//! Port of `api/simple-options.ts`.

use crate::models::clamp_thinking_level;
use crate::types::{
    Model, ModelThinkingLevel, SamplingParams, SimpleStreamOptions, StreamOptions, ThinkingBudgets,
    ThinkingLevel, TranscriptContext,
};
use crate::utils::estimate::estimate_context_tokens;

const CONTEXT_SAFETY_TOKENS: i64 = 4096;
const MIN_MAX_TOKENS: u32 = 1;

pub fn clamp_max_tokens_to_context(
    model: &Model,
    context: &TranscriptContext,
    max_tokens: u32,
) -> u32 {
    if model.context_window == 0 {
        return MIN_MAX_TOKENS.max(max_tokens);
    }
    let available = i64::from(model.context_window)
        - i64::from(estimate_context_tokens(&context.messages).tokens)
        - CONTEXT_SAFETY_TOKENS;
    let available = available.max(i64::from(MIN_MAX_TOKENS));
    i64::from(max_tokens).min(available) as u32
}

pub fn resolve_sampling_params(
    model: &Model,
    thinking_level: ModelThinkingLevel,
    request_params: Option<&SamplingParams>,
) -> Option<SamplingParams> {
    let effective_thinking_level = clamp_thinking_level(model, thinking_level);
    let thinking_level_params = model
        .sampling_params_by_thinking_level
        .as_ref()
        .and_then(|params| params.get(&effective_thinking_level));
    if model.sampling_params.is_none()
        && thinking_level_params.is_none()
        && request_params.is_none()
    {
        return None;
    }
    let mut merged = SamplingParams::new();
    for params in [
        model.sampling_params.as_ref(),
        thinking_level_params,
        request_params,
    ]
    .into_iter()
    .flatten()
    {
        for (key, value) in params {
            merged.insert(key.clone(), value.clone());
        }
    }
    Some(merged)
}

/// `buildBaseOptions()`. `telemetryContext` is not ported; `fetch` is
/// `http_client`.
pub fn build_base_options(
    model: &Model,
    context: &TranscriptContext,
    options: Option<&SimpleStreamOptions>,
    api_key: Option<&str>,
) -> StreamOptions {
    let reasoning = options
        .and_then(|options| options.reasoning)
        .map(ModelThinkingLevel::from)
        .unwrap_or(ModelThinkingLevel::Off);
    let sampling_params = resolve_sampling_params(
        model,
        reasoning,
        options.and_then(|options| options.sampling_params.as_ref()),
    );
    let max_tokens = clamp_max_tokens_to_context(
        model,
        context,
        options
            .and_then(|options| options.max_tokens)
            .unwrap_or(model.max_tokens),
    );
    let Some(options) = options else {
        return StreamOptions {
            sampling_params,
            max_tokens: Some(max_tokens),
            api_key: api_key.filter(|key| !key.is_empty()).map(str::to_string),
            ..Default::default()
        };
    };
    StreamOptions {
        temperature: options.temperature,
        sampling_params,
        max_tokens: Some(max_tokens),
        signal: options.signal.clone(),
        api_key: api_key
            .filter(|key| !key.is_empty())
            .map(str::to_string)
            .or_else(|| options.api_key.clone()),
        http_client: options.http_client.clone(),
        transport: options.transport,
        cache_retention: options.cache_retention,
        session_id: options.session_id.clone(),
        headers: options.headers.clone(),
        on_payload: options.on_payload.clone(),
        on_response: options.on_response.clone(),
        on_provider_stream_event: options.on_provider_stream_event.clone(),
        timeout_ms: options.timeout_ms,
        websocket_connect_timeout_ms: options.websocket_connect_timeout_ms,
        max_retries: options.max_retries,
        max_retry_delay_ms: options.max_retry_delay_ms,
        metadata: options.metadata.clone(),
        env: options.env.clone(),
        provider_options: Default::default(),
    }
}

/// Tokens always left for the answer when a thinking budget shares the response ceiling.
pub const MIN_ANSWER_TOKENS: u32 = 1024;

pub const DEFAULT_THINKING_BUDGETS: ThinkingBudgets = ThinkingBudgets {
    minimal: Some(1024),
    low: Some(2048),
    medium: Some(8192),
    high: Some(16384),
};

/// `clampReasoning()`: `xhigh` and `max` become `high`.
pub fn clamp_reasoning(effort: Option<ThinkingLevel>) -> Option<ThinkingLevel> {
    match effort {
        Some(ThinkingLevel::Xhigh | ThinkingLevel::Max) => Some(ThinkingLevel::High),
        effort => effort,
    }
}

pub fn thinking_budget_for_level(
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> u32 {
    let custom = custom_budgets.cloned().unwrap_or_default();
    let budget = |custom: Option<u32>, default: Option<u32>| custom.or(default).unwrap_or(0);
    match clamp_reasoning(Some(reasoning_level)).unwrap_or(ThinkingLevel::High) {
        ThinkingLevel::Minimal => budget(custom.minimal, DEFAULT_THINKING_BUDGETS.minimal),
        ThinkingLevel::Low => budget(custom.low, DEFAULT_THINKING_BUDGETS.low),
        ThinkingLevel::Medium => budget(custom.medium, DEFAULT_THINKING_BUDGETS.medium),
        _ => budget(custom.high, DEFAULT_THINKING_BUDGETS.high),
    }
}

/// Cap a thinking budget so at least MIN_ANSWER_TOKENS remain under a shared response ceiling.
pub fn clamp_thinking_budget_to_answer_room(thinking_budget: u32, ceiling: u32) -> u32 {
    thinking_budget.min(ceiling.saturating_sub(MIN_ANSWER_TOKENS))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdjustedMaxTokens {
    pub max_tokens: u32,
    pub thinking_budget: u32,
}

/// `base_max_tokens` of `None` means no explicit caller cap: use the model cap
/// and fit thinking inside it.
pub fn adjust_max_tokens_for_thinking(
    base_max_tokens: Option<u32>,
    model_max_tokens: u32,
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> AdjustedMaxTokens {
    let mut thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets);
    let max_tokens = match base_max_tokens {
        None => model_max_tokens,
        Some(base) => base.saturating_add(thinking_budget).min(model_max_tokens),
    };

    if max_tokens <= thinking_budget {
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens);
    }

    AdjustedMaxTokens {
        max_tokens,
        thinking_budget,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::types::Context;
    use crate::utils::transcript::normalize_context;

    fn model() -> Model {
        Model {
            id: "m".to_string(),
            reasoning: true,
            context_window: 10_000,
            max_tokens: 8_000,
            ..Default::default()
        }
    }

    #[test]
    fn explicit_api_key_wins_unless_empty() {
        let context = normalize_context(&Context::default());
        let simple = SimpleStreamOptions::from(StreamOptions {
            api_key: Some("option-key".to_string()),
            ..Default::default()
        });
        assert_eq!(
            build_base_options(&model(), &context, Some(&simple), Some("explicit"))
                .api_key
                .as_deref(),
            Some("explicit")
        );
        assert_eq!(
            build_base_options(&model(), &context, Some(&simple), Some(""))
                .api_key
                .as_deref(),
            Some("option-key")
        );
    }

    #[test]
    fn sampling_params_merge_model_level_and_request_values() {
        let mut model = model();
        model.sampling_params = Some(
            json!({ "temperature": 0.1, "top_p": 0.9 })
                .as_object()
                .unwrap()
                .clone(),
        );
        model.sampling_params_by_thinking_level = Some(
            [(
                ModelThinkingLevel::High,
                json!({ "top_p": 0.8, "top_k": 20 })
                    .as_object()
                    .unwrap()
                    .clone(),
            )]
            .into_iter()
            .collect(),
        );
        let request = json!({ "top_k": 40 }).as_object().unwrap().clone();
        assert_eq!(
            Value::Object(
                resolve_sampling_params(&model, ModelThinkingLevel::Max, Some(&request)).unwrap()
            ),
            json!({ "temperature": 0.1, "top_p": 0.8, "top_k": 40 })
        );
        assert_eq!(
            resolve_sampling_params(&Model::default(), ModelThinkingLevel::Off, None),
            None
        );
    }

    #[test]
    fn thinking_budgets_fit_inside_the_response_ceiling() {
        assert_eq!(thinking_budget_for_level(ThinkingLevel::Max, None), 16384);
        assert_eq!(
            adjust_max_tokens_for_thinking(Some(1000), 32000, ThinkingLevel::Medium, None),
            AdjustedMaxTokens {
                max_tokens: 9192,
                thinking_budget: 8192
            }
        );
        assert_eq!(
            adjust_max_tokens_for_thinking(None, 4000, ThinkingLevel::High, None),
            AdjustedMaxTokens {
                max_tokens: 4000,
                thinking_budget: 2976
            }
        );
        let custom = ThinkingBudgets {
            low: Some(100),
            ..Default::default()
        };
        assert_eq!(
            thinking_budget_for_level(ThinkingLevel::Low, Some(&custom)),
            100
        );
    }
}
