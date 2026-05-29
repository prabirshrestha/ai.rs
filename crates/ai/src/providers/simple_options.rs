use crate::models::clamp_thinking_level;
use crate::types::{
    Model, ModelThinkingLevel, SimpleStreamOptions, StreamOptions, ThinkingBudgets,
};

pub fn build_base_options(
    _model: &Model,
    options: &SimpleStreamOptions,
    api_key: String,
) -> StreamOptions {
    let mut base = options.stream.clone();
    base.api_key = Some(api_key);
    base
}

pub struct AdjustedThinkingTokens {
    pub max_tokens: Option<u32>,
    pub thinking_budget: u32,
}

pub fn adjust_max_tokens_for_thinking(
    requested_max_tokens: Option<u32>,
    model_max_tokens: u32,
    reasoning: Option<ModelThinkingLevel>,
    budgets: Option<&ThinkingBudgets>,
) -> AdjustedThinkingTokens {
    let thinking_budget = match reasoning {
        Some(ModelThinkingLevel::Minimal) => budgets.and_then(|b| b.minimal).unwrap_or(1_024),
        Some(ModelThinkingLevel::Low) => budgets.and_then(|b| b.low).unwrap_or(2_048),
        Some(ModelThinkingLevel::Medium) => budgets.and_then(|b| b.medium).unwrap_or(8_192),
        Some(ModelThinkingLevel::High) | Some(ModelThinkingLevel::Xhigh) => {
            budgets.and_then(|b| b.high).unwrap_or(16_384)
        }
        _ => 1_024,
    };

    let max_tokens = requested_max_tokens
        .map(|max_tokens| {
            max_tokens
                .saturating_add(thinking_budget)
                .min(model_max_tokens)
        })
        .unwrap_or(model_max_tokens);
    let thinking_budget = if max_tokens <= thinking_budget {
        max_tokens.saturating_sub(1_024)
    } else {
        thinking_budget
    };
    AdjustedThinkingTokens {
        max_tokens: Some(max_tokens),
        thinking_budget,
    }
}

pub fn clamped_reasoning(
    model: &Model,
    options: &SimpleStreamOptions,
) -> Option<ModelThinkingLevel> {
    options.reasoning.and_then(|level| {
        let clamped = clamp_thinking_level(model, level);
        (clamped != ModelThinkingLevel::Off).then_some(clamped)
    })
}
