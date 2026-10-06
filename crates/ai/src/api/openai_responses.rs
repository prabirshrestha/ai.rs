//! Port of `api/openai-responses.ts`: the OpenAI Responses API.
//!
//! The `openai` SDK client is replaced by [`OpenAIClient`]. Pi's
//! `streamSimple()` throws synchronously when no API key is available;
//! [`stream_simple_openai_responses`] returns that error as `Err`, and the
//! [`ProviderStreams`] implementation turns it into an error stream.

use std::sync::Arc;

use serde_json::{Value, json};

use super::constrained_sampling::create_grammar_tool_input_properties;
use super::github_copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use super::lazy::error_stream;
use super::openai_client::{OpenAIClient, OpenAIRequestOptions, sse_json_events};
use super::openai_prompt_cache::clamp_openai_prompt_cache_key;
use super::openai_responses_shared::{
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, GrammarToolInputProperties,
    OpenAIResponsesStreamOptions, convert_responses_messages, convert_responses_tools,
    process_responses_stream,
};
use super::simple_options::{build_base_options, resolve_sampling_params};
use crate::models::clamp_thinking_level;
use crate::types::{
    AssistantMessage, AssistantMessageEvent, CacheRetention, Model, ModelThinkingLevel,
    ProviderEnv, ProviderHeaders, ProviderResponse, ProviderStreams, SessionAffinityFormat,
    SimpleStreamOptions, StopReason, StreamOptions, ThinkingLevel, ToolChoice, TranscriptContext,
    Usage,
};
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::headers::{has_non_empty_header, headers_to_record};
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};
use crate::utils::time::now_millis;
use crate::utils::transcript::{get_declared_tools, resolve_transcript, resolve_transcript_tools};
use crate::{Error, Result};

const OPENAI_TOOL_CALL_PROVIDERS: &[&str] = &["openai", "openai-codex", "opencode"];
// OpenAI Responses rejects max_output_tokens below 16: https://github.com/earendil-works/pi/issues/6265
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u32 = 16;
const CHATGPT_USAGE_URL: &str = "https://chatgpt.com/settings/usage";

/// OpenAI API keys start with `sk-`; a different credential sent directly to OpenAI
/// is a Sign in with ChatGPT access token.
fn is_chatgpt_sign_in(model: &Model, api_key: Option<&str>) -> bool {
    model.provider == "openai"
        && model.base_url == "https://api.openai.com/v1"
        && api_key.is_some_and(|api_key| !api_key.starts_with("sk-"))
}

/// `getClientApiKey()`, shared with the Chat Completions module.
pub(crate) fn get_client_api_key(
    provider: &str,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
) -> Result<String> {
    if let Some(api_key) = api_key.filter(|api_key| !api_key.is_empty()) {
        return Ok(api_key.to_string());
    }
    if headers.is_some_and(|headers| {
        has_non_empty_header(headers, "authorization")
            || has_non_empty_header(headers, "cf-aig-authorization")
    }) {
        return Ok("unused".to_string());
    }
    Err(Error::message(format!(
        "No API key for provider: {provider}"
    )))
}

fn detect_session_affinity_format(model: &Model) -> SessionAffinityFormat {
    if model.provider == "openrouter" || model.base_url.contains("openrouter.ai") {
        SessionAffinityFormat::Openrouter
    } else {
        SessionAffinityFormat::Openai
    }
}

/// Resolve cache retention preference.
/// Defaults to "short" and uses PI_CACHE_RETENTION for backward compatibility.
pub(crate) fn resolve_cache_retention(
    cache_retention: Option<CacheRetention>,
    env: Option<&ProviderEnv>,
) -> CacheRetention {
    if let Some(cache_retention) = cache_retention {
        return cache_retention;
    }
    if get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
        return CacheRetention::Long;
    }
    CacheRetention::Short
}

/// `Required<OpenAIResponsesCompat>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedOpenAIResponsesCompat {
    pub supports_developer_role: bool,
    pub supports_mid_convo_system_messages: bool,
    pub session_affinity_format: SessionAffinityFormat,
    pub supports_long_cache_retention: bool,
    pub supports_strict_mode: bool,
    pub supports_openai_grammar_tools: bool,
    pub supports_additional_tools: bool,
    pub supports_tool_search: bool,
    pub supports_explicit_prompt_cache_mode: bool,
    pub supports_max_output_tokens: bool,
}

pub fn get_compat(model: &Model) -> ResolvedOpenAIResponsesCompat {
    let compat = model.compat.clone().unwrap_or_default();
    ResolvedOpenAIResponsesCompat {
        supports_developer_role: compat.supports_developer_role.unwrap_or(true),
        supports_mid_convo_system_messages: compat
            .supports_mid_convo_system_messages
            .unwrap_or(false),
        session_affinity_format: compat
            .session_affinity_format
            .unwrap_or_else(|| detect_session_affinity_format(model)),
        supports_long_cache_retention: compat.supports_long_cache_retention.unwrap_or(true),
        supports_strict_mode: compat.supports_strict_mode.unwrap_or(false),
        supports_openai_grammar_tools: compat.supports_openai_grammar_tools.unwrap_or(false),
        supports_additional_tools: compat.supports_additional_tools.unwrap_or(false),
        supports_tool_search: compat.supports_tool_search.unwrap_or(false),
        supports_explicit_prompt_cache_mode: compat
            .supports_explicit_prompt_cache_mode
            .unwrap_or(false),
        supports_max_output_tokens: compat.supports_max_output_tokens.unwrap_or(true),
    }
}

fn get_prompt_cache_retention(
    compat: &ResolvedOpenAIResponsesCompat,
    cache_retention: CacheRetention,
) -> Option<&'static str> {
    (cache_retention == CacheRetention::Long
        && compat.supports_long_cache_retention
        && !compat.supports_explicit_prompt_cache_mode)
        .then_some("24h")
}

fn get_prompt_cache_options(
    compat: &ResolvedOpenAIResponsesCompat,
    cache_retention: CacheRetention,
) -> Option<Value> {
    if !compat.supports_explicit_prompt_cache_mode {
        return None;
    }
    if cache_retention == CacheRetention::None {
        return Some(json!({ "mode": "explicit" }));
    }
    if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
        return Some(json!({ "ttl": "30m" }));
    }
    None
}

/// OpenAI Responses-specific options (`OpenAIResponsesOptions extends StreamOptions`).
#[derive(Debug, Clone, Default)]
pub struct OpenAIResponsesOptions {
    pub stream: StreamOptions,
    pub reasoning_effort: Option<ThinkingLevel>,
    /// `"auto" | "detailed" | "concise" | null`: `Some(None)` is `null`.
    pub reasoning_summary: Option<Option<String>>,
    pub service_tier: Option<String>,
    pub tool_choice: Option<Value>,
}

impl std::ops::Deref for OpenAIResponsesOptions {
    type Target = StreamOptions;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

impl std::ops::DerefMut for OpenAIResponsesOptions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stream
    }
}

impl From<StreamOptions> for OpenAIResponsesOptions {
    /// Reads the Responses options from `provider_options` under their Pi
    /// names (`reasoningEffort`, `reasoningSummary`, `serviceTier`,
    /// `toolChoice`).
    fn from(stream: StreamOptions) -> Self {
        let provider_options = &stream.provider_options;
        let reasoning_effort = provider_options
            .get("reasoningEffort")
            .and_then(|value| serde_json::from_value(value.clone()).ok());
        let reasoning_summary = provider_options
            .get("reasoningSummary")
            .map(|value| value.as_str().map(str::to_string));
        let service_tier = provider_options
            .get("serviceTier")
            .and_then(Value::as_str)
            .map(str::to_string);
        let tool_choice = provider_options.get("toolChoice").cloned();
        Self {
            reasoning_effort,
            reasoning_summary,
            service_tier,
            tool_choice,
            stream,
        }
    }
}

/// Generate function for OpenAI Responses API
pub fn stream_openai_responses(
    model: Model,
    context: TranscriptContext,
    options: OpenAIResponsesOptions,
) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let normalized_context = resolve_transcript(
        &context,
        Some(get_compat(&model).supports_mid_convo_system_messages),
    );

    let events = stream.clone();
    tokio::spawn(async move {
        let mut output = AssistantMessage::empty_for(&model);
        output.stop_reason = StopReason::Pending;
        output.timestamp = now_millis();

        match run_responses(&model, &normalized_context, &options, &mut output, &events).await {
            Ok(()) => {
                events.push(AssistantMessageEvent::Done {
                    reason: output.stop_reason,
                    message: output.clone(),
                });
                events.end(None);
            }
            Err(error) => {
                let aborted = options
                    .signal
                    .as_ref()
                    .is_some_and(|signal| signal.is_cancelled());
                output.stop_reason = if aborted {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                let prefix = format!(
                    "{} API error",
                    if model.provider == "openai" {
                        "OpenAI"
                    } else {
                        model.provider.as_str()
                    }
                );
                let error_message =
                    format_provider_error(&normalize_provider_error(&error), Some(&prefix));
                // Sign in with ChatGPT shares the subscription's usage limit with other apps.
                output.error_message = Some(
                    if error_message.contains("subscription_sharing_usage_limit_exceeded") {
                        format!("{error_message}\nCheck your ChatGPT usage: {CHATGPT_USAGE_URL}")
                    } else {
                        error_message
                    },
                );
                events.push(AssistantMessageEvent::Error {
                    reason: output.stop_reason,
                    error: output.clone(),
                });
                events.end(None);
            }
        }
    });

    stream
}

async fn run_responses(
    model: &Model,
    context: &TranscriptContext,
    options: &OpenAIResponsesOptions,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> Result<()> {
    // Create OpenAI client
    let api_key = get_client_api_key(
        &model.provider,
        options.api_key.as_deref(),
        options.headers.as_ref(),
    )?;
    let cache_retention = resolve_cache_retention(options.cache_retention, options.env.as_ref());
    let cache_session_id = if cache_retention == CacheRetention::None {
        None
    } else {
        options.session_id.as_deref()
    };
    let compat = get_compat(model);
    let grammar_tool_input_properties = create_grammar_tool_input_properties(
        Some(&get_declared_tools(&context.messages)),
        compat.supports_openai_grammar_tools,
    )?;
    let client = create_client(
        model,
        context,
        api_key,
        options.headers.as_ref(),
        options.http_client.as_ref(),
        cache_session_id,
    );
    let mut params = build_params(
        model,
        context,
        Some(options),
        &compat,
        &grammar_tool_input_properties,
    )?;
    if let Some(on_payload) = &options.on_payload
        && let Some(next_params) = on_payload(params.clone(), model).await?
    {
        params = next_params;
    }
    let request_options = OpenAIRequestOptions {
        signal: options.signal.clone(),
        timeout_ms: options.timeout_ms,
    };
    let response = retry_provider_request(
        || client.post("/responses", &params, &request_options),
        &ProviderRetryOptions {
            max_retries: options.max_retries,
            max_retry_delay_ms: options.max_retry_delay_ms,
            signal: options.signal.clone(),
        },
    )
    .await?;
    if let Some(on_response) = &options.on_response {
        on_response(
            ProviderResponse {
                status: response.status().as_u16(),
                headers: headers_to_record(response.headers()),
            },
            model,
        )
        .await?;
    }
    stream.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let model_id = model.id.clone();
    let stream_options = OpenAIResponsesStreamOptions {
        on_provider_stream_event: options.on_provider_stream_event.clone(),
        service_tier: options.service_tier.clone(),
        grammar_tool_input_properties: Some(grammar_tool_input_properties),
        resolve_service_tier: None,
        apply_service_tier_pricing: Some(Arc::new(move |usage: &mut Usage, service_tier| {
            apply_service_tier_pricing(usage, service_tier, &model_id);
        })),
    };
    process_responses_stream(
        sse_json_events(response, options.signal.clone()),
        output,
        stream,
        model,
        Some(&stream_options),
    )
    .await?;

    if options
        .signal
        .as_ref()
        .is_some_and(|signal| signal.is_cancelled())
    {
        return Err(Error::message("Request was aborted"));
    }

    if output.stop_reason == StopReason::Pending {
        return Err(Error::message(
            "OpenAI Responses stream ended without a stop reason",
        ));
    }
    if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
        return Err(Error::message(
            output
                .error_message
                .clone()
                .filter(|message| !message.is_empty())
                .unwrap_or_else(|| "An unknown error occurred".to_string()),
        ));
    }
    Ok(())
}

pub(crate) fn model_thinking_level_to_effort(level: ModelThinkingLevel) -> Option<ThinkingLevel> {
    match level {
        ModelThinkingLevel::Off => None,
        ModelThinkingLevel::Minimal => Some(ThinkingLevel::Minimal),
        ModelThinkingLevel::Low => Some(ThinkingLevel::Low),
        ModelThinkingLevel::Medium => Some(ThinkingLevel::Medium),
        ModelThinkingLevel::High => Some(ThinkingLevel::High),
        ModelThinkingLevel::Xhigh => Some(ThinkingLevel::Xhigh),
        ModelThinkingLevel::Max => Some(ThinkingLevel::Max),
    }
}

pub(crate) fn tool_choice_value(tool_choice: Option<ToolChoice>) -> Option<Value> {
    tool_choice.map(|tool_choice| match tool_choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
    })
}

pub fn stream_simple_openai_responses(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions,
) -> Result<AssistantMessageEventStream> {
    get_client_api_key(
        &model.provider,
        options.api_key.as_deref(),
        options.headers.as_ref(),
    )?;

    let base = build_base_options(&model, &context, Some(&options), options.api_key.as_deref());
    let clamped_reasoning = options
        .reasoning
        .map(|reasoning| clamp_thinking_level(&model, reasoning.into()));
    let reasoning_effort = clamped_reasoning.and_then(model_thinking_level_to_effort);

    Ok(stream_openai_responses(
        model,
        context,
        OpenAIResponsesOptions {
            stream: base,
            tool_choice: tool_choice_value(options.tool_choice),
            reasoning_effort,
            ..Default::default()
        },
    ))
}

fn create_client(
    model: &Model,
    context: &TranscriptContext,
    api_key: String,
    options_headers: Option<&ProviderHeaders>,
    http_client: Option<&reqwest::Client>,
    session_id: Option<&str>,
) -> OpenAIClient {
    let compat = get_compat(model);
    let mut headers = ProviderHeaders::new();
    headers.insert("User-Agent", Some(get_pi_user_agent()));
    for (name, value) in model.headers.iter().flatten() {
        headers.insert(name.clone(), Some(value.clone()));
    }
    if model.provider == "github-copilot" {
        let has_images = has_copilot_vision_input(&context.messages);
        let copilot_headers = build_copilot_dynamic_headers(&context.messages, has_images);
        for (name, value) in copilot_headers.iter() {
            headers.insert(name.clone(), value.clone());
        }
    }

    if let Some(session_id) = session_id.filter(|session_id| !session_id.is_empty()) {
        if compat.session_affinity_format == SessionAffinityFormat::Openrouter {
            headers.insert("x-session-id", Some(session_id.to_string()));
        } else {
            if compat.session_affinity_format == SessionAffinityFormat::Openai {
                headers.insert("session_id", Some(session_id.to_string()));
            }
            headers.insert("x-client-request-id", Some(session_id.to_string()));
        }
    }

    // Merge options headers last so they can override defaults
    if let Some(options_headers) = options_headers {
        for (name, value) in options_headers.iter() {
            headers.insert(name.clone(), value.clone());
        }
    }

    OpenAIClient::new(api_key, &model.base_url, http_client, headers)
}

pub fn build_params(
    model: &Model,
    context: &TranscriptContext,
    options: Option<&OpenAIResponsesOptions>,
    compat: &ResolvedOpenAIResponsesCompat,
    grammar_tool_input_properties: &GrammarToolInputProperties,
) -> Result<Value> {
    let transcript_tools = resolve_transcript_tools(
        &context.messages,
        compat.supports_additional_tools || compat.supports_tool_search,
    );
    let messages = convert_responses_messages(
        model,
        context,
        OPENAI_TOOL_CALL_PROVIDERS,
        Some(&ConvertResponsesMessagesOptions {
            include_system_prompt: None,
            grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
            supports_mid_convo_system_messages: Some(compat.supports_mid_convo_system_messages),
            supports_additional_tools: Some(compat.supports_additional_tools),
            supports_tool_search: Some(compat.supports_tool_search),
            tool_options: Some(ConvertResponsesToolsOptions {
                strict: None,
                supports_strict_mode: Some(compat.supports_strict_mode),
                supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
                tool_search_result: None,
            }),
        }),
    )?;

    let cache_retention = resolve_cache_retention(
        options.and_then(|options| options.cache_retention),
        options.and_then(|options| options.env.as_ref()),
    );
    // Sign in with ChatGPT rejects these request fields.
    let omit_unsupported_fields = is_chatgpt_sign_in(
        model,
        options.and_then(|options| options.api_key.as_deref()),
    );
    let mut params = serde_json::Map::new();
    params.insert("model".to_string(), json!(model.id));
    params.insert("input".to_string(), Value::Array(messages));
    params.insert("stream".to_string(), json!(true));
    if cache_retention != CacheRetention::None
        && let Some(key) =
            clamp_openai_prompt_cache_key(options.and_then(|options| options.session_id.as_deref()))
    {
        params.insert("prompt_cache_key".to_string(), json!(key));
    }
    if !omit_unsupported_fields {
        if let Some(retention) = get_prompt_cache_retention(compat, cache_retention) {
            params.insert("prompt_cache_retention".to_string(), json!(retention));
        }
        if let Some(cache_options) = get_prompt_cache_options(compat, cache_retention) {
            params.insert("prompt_cache_options".to_string(), cache_options);
        }
    }
    params.insert("store".to_string(), json!(false));

    if let Some(max_tokens) = options
        .and_then(|options| options.max_tokens)
        .filter(|max_tokens| *max_tokens > 0)
        && compat.supports_max_output_tokens
        && !omit_unsupported_fields
    {
        params.insert(
            "max_output_tokens".to_string(),
            json!(max_tokens.max(OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)),
        );
    }

    if let Some(temperature) = options.and_then(|options| options.temperature)
        && !omit_unsupported_fields
    {
        params.insert("temperature".to_string(), json!(temperature));
    }

    if let Some(service_tier) = options.and_then(|options| options.service_tier.as_ref()) {
        params.insert("service_tier".to_string(), json!(service_tier));
    }

    if !transcript_tools.request_tools.is_empty() {
        params.insert(
            "tools".to_string(),
            Value::Array(convert_responses_tools(
                &transcript_tools.request_tools,
                Some(&ConvertResponsesToolsOptions {
                    supports_strict_mode: Some(compat.supports_strict_mode),
                    supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
                    ..Default::default()
                }),
            )?),
        );
    }

    if let Some(tool_choice) = options.and_then(|options| options.tool_choice.as_ref()) {
        params.insert("tool_choice".to_string(), tool_choice.clone());
    }

    let requested_effort = options.and_then(|options| options.reasoning_effort);
    let reasoning_summary = options
        .and_then(|options| options.reasoning_summary.clone())
        .flatten()
        .filter(|summary| !summary.is_empty());
    let reasoning_effort: Option<&str> = requested_effort
        .map(ThinkingLevel::as_str)
        .or_else(|| reasoning_summary.as_ref().map(|_| "medium"));
    if model.reasoning {
        if let Some(reasoning_effort) = reasoning_effort {
            let effort = match requested_effort {
                Some(requested) => model
                    .thinking_level_map
                    .as_ref()
                    .and_then(|map| map.get(&ModelThinkingLevel::from(requested)))
                    .and_then(|effort| effort.clone())
                    .unwrap_or_else(|| requested.as_str().to_string()),
                None => reasoning_effort.to_string(),
            };
            params.insert(
                "reasoning".to_string(),
                json!({
                    "effort": effort,
                    "summary": reasoning_summary.clone().unwrap_or_else(|| "auto".to_string()),
                }),
            );
            params.insert(
                "include".to_string(),
                json!(["reasoning.encrypted_content"]),
            );
        } else {
            let off = model
                .thinking_level_map
                .as_ref()
                .and_then(|map| map.get(&ModelThinkingLevel::Off));
            // `thinkingLevelMap?.off !== null`: a missing entry is undefined.
            if model.provider != "github-copilot" && !matches!(off, Some(None)) {
                let effort = off.cloned().flatten().unwrap_or_else(|| "none".to_string());
                params.insert("reasoning".to_string(), json!({ "effort": effort }));
            }
        }
        if model.provider == "xai" {
            params.insert(
                "include".to_string(),
                json!(["reasoning.encrypted_content"]),
            );
        }
    }

    // Last so model and request sampling parameters override named request fields.
    let sampling_level = reasoning_effort
        .and_then(|effort| serde_json::from_value::<ModelThinkingLevel>(json!(effort)).ok())
        .unwrap_or(ModelThinkingLevel::Off);
    if let Some(sampling_params) = resolve_sampling_params(
        model,
        sampling_level,
        options.and_then(|options| options.sampling_params.as_ref()),
    ) {
        for (key, value) in sampling_params {
            params.insert(key, value);
        }
    }

    Ok(Value::Object(params))
}

fn get_service_tier_cost_multiplier(model_id: &str, service_tier: Option<&str>) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority" | "fast") => {
            if model_id == "gpt-5.5" {
                2.5
            } else {
                2.0
            }
        }
        _ => 1.0,
    }
}

fn apply_service_tier_pricing(usage: &mut Usage, service_tier: Option<&str>, model_id: &str) {
    let multiplier = get_service_tier_cost_multiplier(model_id, service_tier);
    if multiplier == 1.0 {
        return;
    }

    usage.cost.input *= multiplier;
    usage.cost.output *= multiplier;
    usage.cost.cache_read *= multiplier;
    usage.cost.cache_write *= multiplier;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

struct OpenAIResponsesApi;

impl ProviderStreams for OpenAIResponsesApi {
    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        stream_openai_responses(model, context, options.into())
    }

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let error_model = model.clone();
        stream_simple_openai_responses(model, context, options)
            .unwrap_or_else(|error| error_stream(&error_model, error))
    }
}

/// The `openai-responses` implementation as `ProviderStreams`.
pub fn openai_responses_api() -> Arc<dyn ProviderStreams> {
    Arc::new(OpenAIResponsesApi)
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::api::openai_client::test_support::{
        MockResponse, MockServer, capture_payload, collect, collect_aborting, context,
        copilot_model, gpt5_mini, model, openai_model, serve_stalled_sse, stream_event_hook,
    };
    use crate::types::{AssistantMessageEvent, ModelCompat};

    fn hi() -> TranscriptContext {
        context(json!({
            "systemPrompt": "sys",
            "messages": [{ "role": "user", "content": "hi", "timestamp": 1 }],
        }))
    }

    async fn payload_for(
        model: &Model,
        ctx: TranscriptContext,
        mut base: OpenAIResponsesOptions,
    ) -> Value {
        let model = model.clone();
        if base.api_key.is_none() {
            base.api_key = Some("sk-test-key".to_string());
        }
        capture_payload(move |hook| {
            base.on_payload = Some(hook);
            stream_openai_responses(model, ctx, base)
        })
        .await
    }

    fn completed_response() -> MockResponse {
        MockResponse::sse(&[json!({
            "type": "response.completed", "sequence_number": 0,
            "response": { "id": "resp_test", "status": "completed", "output": [],
                "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                    "input_tokens_details": { "cached_tokens": 0 } } },
        })])
    }

    /// Stream against a mock server and return the captured request and result.
    async fn request_with(
        mut model: Model,
        ctx: TranscriptContext,
        options: OpenAIResponsesOptions,
        response: MockResponse,
    ) -> (
        crate::api::openai_client::test_support::CapturedRequest,
        AssistantMessage,
    ) {
        let server = MockServer::start(vec![response]).await;
        model.base_url = server.url.clone();
        let (_, result) = collect(stream_openai_responses(model, ctx, options)).await;
        (server.last(), result)
    }

    fn with_key(api_key: &str) -> OpenAIResponsesOptions {
        let mut options = OpenAIResponsesOptions::default();
        options.api_key = Some(api_key.to_string());
        options
    }

    // openai-responses-compat.test.ts
    #[tokio::test]
    async fn omits_reasoning_for_copilot_when_none_is_requested() {
        let payload = payload_for(&copilot_model("gpt-5-mini"), hi(), Default::default()).await;
        assert!(payload.get("reasoning").is_none());
    }

    #[tokio::test]
    async fn forwards_required_tool_choice() {
        let ctx = context(json!({
            "messages": [{ "role": "user", "content": "Do not call ping. Respond with text instead.", "timestamp": 1 }],
            "tools": [{ "name": "ping", "description": "Ping",
                "parameters": { "type": "object", "properties": { "value": { "type": "string" } }, "required": ["value"] } }],
        }));
        let mut base = OpenAIResponsesOptions::default();
        base.tool_choice = Some(json!("required"));
        let payload = payload_for(&openai_model("gpt-5.4"), ctx, base).await;
        assert_eq!(payload["tool_choice"], "required");
        assert_eq!(payload["tools"][0]["name"], "ping");
    }

    #[tokio::test]
    async fn sets_strict_mode_explicitly_when_supported() {
        // Cloudflare AI Gateway's gpt-5.6-sol: an OpenAI Responses model with strict mode.
        let mut model = openai_model("gpt-5.6-sol");
        model.provider = "cloudflare-ai-gateway".to_string();
        model
            .compat
            .get_or_insert_with(ModelCompat::default)
            .supports_strict_mode = Some(true);
        let ctx = context(json!({
            "messages": [{ "role": "user", "content": "Use a tool.", "timestamp": 1 }],
            "tools": [
                { "name": "ordinary", "description": "An ordinary tool", "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" }, "offset": { "type": "number" } },
                    "required": ["path"] } },
                { "name": "constrained", "description": "A constrained tool", "parameters": {
                    "type": "object", "properties": { "value": { "type": "string" } }, "required": ["value"] },
                  "constrainedSampling": { "type": "json_schema", "strict": "prefer" } },
            ],
        }));
        let payload = payload_for(&model, ctx, Default::default()).await;
        assert_eq!(payload["tools"][0]["name"], "ordinary");
        assert_eq!(payload["tools"][0]["strict"], false);
        assert_eq!(payload["tools"][1]["name"], "constrained");
        assert_eq!(payload["tools"][1]["strict"], true);
    }

    #[tokio::test]
    async fn sends_none_reasoning_effort_when_off_is_supported() {
        for id in [
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.3-codex",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.5",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-6-sol",
            "gpt-6-luna",
        ] {
            let payload = payload_for(&openai_model(id), hi(), Default::default()).await;
            assert_eq!(payload["reasoning"], json!({ "effort": "none" }), "{id}");
        }
    }

    #[tokio::test]
    async fn omits_reasoning_effort_when_off_is_unsupported() {
        for id in [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-pro",
            "gpt-5.2-pro",
            "gpt-5.4-pro",
            "gpt-5.5-pro",
        ] {
            let payload = payload_for(&openai_model(id), hi(), Default::default()).await;
            assert!(payload.get("reasoning").is_none(), "{id}");
        }
    }

    async fn affinity_headers(
        model: Model,
        options: OpenAIResponsesOptions,
    ) -> (Option<String>, Option<String>, Option<String>) {
        let (request, _) = request_with(model, hi(), options, MockResponse::sse(&[])).await;
        let header = |name: &str| request.header(name).map(str::to_string);
        (
            header("session_id"),
            header("x-client-request-id"),
            header("x-session-id"),
        )
    }

    fn session(session_id: &str) -> OpenAIResponsesOptions {
        let mut options = with_key("sk-test-key");
        options.session_id = Some(session_id.to_string());
        options
    }

    fn proxy(provider: &str, format: Option<SessionAffinityFormat>) -> Model {
        let mut model = openai_model("gpt-5.4");
        model.provider = provider.to_string();
        if let Some(format) = format {
            model.compat = Some(ModelCompat {
                session_affinity_format: Some(format),
                ..Default::default()
            });
        }
        model
    }

    #[tokio::test]
    async fn sets_cache_affinity_headers_with_a_session_id() {
        let headers = affinity_headers(openai_model("gpt-5.4"), session("session-123")).await;
        assert_eq!(
            headers,
            (Some("session-123".into()), Some("session-123".into()), None)
        );
        let headers = affinity_headers(proxy("opencode", None), session("session-123")).await;
        assert_eq!(
            headers,
            (Some("session-123".into()), Some("session-123".into()), None)
        );
    }

    #[tokio::test]
    async fn clamps_prompt_cache_key_to_64_characters() {
        let payload = payload_for(&openai_model("gpt-5.4"), hi(), session(&"x".repeat(67))).await;
        assert_eq!(payload["prompt_cache_key"], json!("x".repeat(64)));
    }

    #[tokio::test]
    async fn uses_openrouter_session_affinity_when_configured_or_detected() {
        let configured = proxy("proxy", Some(SessionAffinityFormat::Openrouter));
        let headers = affinity_headers(configured.clone(), session("session-proxy")).await;
        assert_eq!(headers, (None, None, Some("session-proxy".into())));
        let payload = payload_for(&configured, hi(), session("session-proxy")).await;
        assert!(payload.get("session_id").is_none());
        assert_eq!(payload["prompt_cache_key"], "session-proxy");

        let detected = proxy("openrouter", None);
        let headers = affinity_headers(detected, session("session-openrouter")).await;
        assert_eq!(headers, (None, None, Some("session-openrouter".into())));
    }

    #[tokio::test]
    async fn uses_the_no_session_format_when_configured() {
        for provider in ["proxy", "opencode"] {
            let model = proxy(provider, Some(SessionAffinityFormat::OpenaiNosession));
            let headers = affinity_headers(model.clone(), session("session-proxy")).await;
            assert_eq!(headers, (None, Some("session-proxy".into()), None));
            let payload = payload_for(&model, hi(), session("session-proxy")).await;
            assert_eq!(payload["prompt_cache_key"], "session-proxy");
        }
    }

    #[tokio::test]
    async fn explicit_headers_override_affinity_headers() {
        let mut options = session("session-123");
        let mut headers = ProviderHeaders::new();
        headers.insert("session_id", Some("override-session".to_string()));
        headers.insert("x-client-request-id", Some("override-request".to_string()));
        options.headers = Some(headers);
        let headers = affinity_headers(openai_model("gpt-5.4"), options).await;
        assert_eq!(headers.0.as_deref(), Some("override-session"));
        assert_eq!(headers.1.as_deref(), Some("override-request"));
    }

    #[tokio::test]
    async fn omits_affinity_headers_when_cache_retention_is_none() {
        let mut options = session("session-123");
        options.cache_retention = Some(CacheRetention::None);
        let headers = affinity_headers(openai_model("gpt-5.4"), options).await;
        assert_eq!(headers, (None, None, None));
    }

    #[tokio::test]
    async fn applies_service_tier_cost_multipliers() {
        for (id, tier, response_tier, multiplier) in [
            ("gpt-5.4", "priority", "priority", 2.0),
            ("gpt-5.5", "priority", "priority", 2.5),
            ("gpt-5.5", "flex", "flex", 0.5),
            // GPT-6 models report Fast mode as "fast" even when "priority" is requested (#10034)
            ("gpt-6-luna", "priority", "fast", 2.0),
            ("gpt-6-luna", "fast", "fast", 2.0),
        ] {
            let model = openai_model(id);
            let tokens = 100_000;
            let scale = f64::from(tokens) / 1_000_000.0;
            let response = MockResponse::sse(&[json!({
                "type": "response.completed",
                "response": { "status": "completed", "service_tier": response_tier,
                    "usage": { "input_tokens": tokens, "output_tokens": tokens, "total_tokens": tokens * 2,
                        "input_tokens_details": { "cached_tokens": 0 } } },
            })]);
            let mut options = with_key("sk-test-key");
            options.service_tier = Some(tier.to_string());
            let (_, result) = request_with(model.clone(), hi(), options, response).await;
            let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
            assert!(
                close(
                    result.usage.cost.input,
                    model.cost.input * multiplier * scale
                ),
                "{id} {tier}"
            );
            assert!(
                close(
                    result.usage.cost.output,
                    model.cost.output * multiplier * scale
                ),
                "{id} {tier}"
            );
            assert!(close(
                result.usage.cost.total,
                (model.cost.input + model.cost.output) * multiplier * scale
            ));
        }
    }

    #[tokio::test]
    async fn sends_max_output_tokens_unless_unsupported() {
        let mut base = OpenAIResponsesOptions::default();
        base.max_tokens = Some(1024);
        let payload = payload_for(&openai_model("gpt-5.4"), hi(), base.clone()).await;
        assert_eq!(payload["max_output_tokens"], 1024);

        let mut model = openai_model("gpt-5.4");
        model
            .compat
            .get_or_insert_with(ModelCompat::default)
            .supports_max_output_tokens = Some(false);
        let payload = payload_for(&model, hi(), base.clone()).await;
        assert!(payload.get("max_output_tokens").is_none());

        // OpenAI Responses rejects max_output_tokens below 16.
        base.max_tokens = Some(4);
        let payload = payload_for(&openai_model("gpt-5.4"), hi(), base).await;
        assert_eq!(payload["max_output_tokens"], 16);
    }

    // openai-responses-chatgpt-sign-in.test.ts
    fn sign_in_context() -> TranscriptContext {
        context(json!({
            "systemPrompt": "",
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }], "timestamp": 0 }],
            "tools": [],
        }))
    }

    async fn sign_in_payload(api_key: &str, model: &Model) -> Value {
        let mut base = with_key(api_key);
        base.max_tokens = Some(1000);
        base.temperature = Some(0.5);
        base.cache_retention = Some(CacheRetention::Long);
        payload_for(model, sign_in_context(), base).await
    }

    #[tokio::test]
    async fn chatgpt_sign_in_omits_rejected_fields() {
        let model = gpt5_mini("openai-responses");
        let payload = sign_in_payload("chatgpt-access-token", &model).await;
        assert!(payload.get("max_output_tokens").is_none());
        assert!(payload.get("temperature").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());

        let mut explicit = model.clone();
        explicit.compat = Some(ModelCompat {
            supports_explicit_prompt_cache_mode: Some(true),
            ..Default::default()
        });
        assert!(
            sign_in_payload("chatgpt-access-token", &explicit)
                .await
                .get("prompt_cache_options")
                .is_none()
        );
        assert_eq!(
            sign_in_payload("sk-proj-test", &explicit).await["prompt_cache_options"],
            json!({ "ttl": "30m" })
        );

        let mut gateway = model.clone();
        gateway.base_url = "https://gateway.example.com/v1".to_string();
        for (api_key, model) in [("sk-proj-test", &model), ("gateway-key", &gateway)] {
            let payload = sign_in_payload(api_key, model).await;
            assert_eq!(payload["max_output_tokens"], 1000);
            assert_eq!(payload["temperature"], 0.5);
            assert_eq!(payload["prompt_cache_retention"], "24h");
        }
    }

    // openai-responses-usage-limit.test.ts
    #[tokio::test]
    async fn usage_limit_errors_link_to_chatgpt_usage() {
        let usage_limit = json!({ "code": "subscription_sharing_usage_limit_exceeded", "message": "Usage limit reached." });
        let rejected = MockResponse::status(
            429,
            &[("content-type", "application/json")],
            json!({ "error": { "code": "subscription_sharing_usage_limit_exceeded",
                "message": "Usage limit reached.", "type": "rate_limit_error" } })
            .to_string(),
        );
        let failed = MockResponse::sse_raw(format!(
            "event: response.failed\ndata: {}\n\n",
            json!({ "type": "response.failed", "sequence_number": 0,
                "response": { "id": "resp_failed", "status": "failed", "error": usage_limit } })
        ));
        for (response, expected) in [
            (rejected, "subscription_sharing_usage_limit_exceeded"),
            (
                failed,
                "subscription_sharing_usage_limit_exceeded: Usage limit reached.",
            ),
        ] {
            let (_, result) = request_with(
                gpt5_mini("openai-responses"),
                sign_in_context(),
                with_key("test"),
                response,
            )
            .await;
            assert_eq!(result.stop_reason, StopReason::Error);
            let message = result.error_message.unwrap();
            assert!(message.contains(expected), "{message}");
            assert!(
                message.contains("Check your ChatGPT usage: https://chatgpt.com/settings/usage")
            );
        }
    }

    // openai-responses-terminal-event.test.ts (wrapper)
    fn early_eof_response() -> MockResponse {
        MockResponse::sse(&[
            json!({ "type": "response.created", "sequence_number": 0, "response": { "id": "resp_wrapper_early_eof" } }),
            json!({ "type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
                "item": { "type": "reasoning", "id": "rs_wrapper_early_eof", "summary": [] } }),
            json!({ "type": "response.reasoning_text.delta", "sequence_number": 2, "output_index": 0,
                "content_index": 0, "item_id": "rs_wrapper_early_eof",
                "delta": "partial reasoning before the wrapper stream ends" }),
        ])
    }

    #[tokio::test]
    async fn forwards_parsed_provider_stream_events_in_order() {
        let (hook, events) = stream_event_hook();
        let mut options = with_key("test");
        options.on_provider_stream_event = Some(hook);
        request_with(
            gpt5_mini("openai-responses"),
            sign_in_context(),
            options,
            early_eof_response(),
        )
        .await;
        let events = events.lock().clone();
        let types: Vec<_> = events
            .iter()
            .map(|(event, _)| event["type"].clone())
            .collect();
        assert_eq!(
            types,
            [
                "response.created",
                "response.output_item.added",
                "response.reasoning_text.delta"
            ]
        );
        assert!(events.iter().all(|(_, model)| model == "gpt-5-mini"));
    }

    #[tokio::test]
    async fn early_eof_ends_with_an_error_result() {
        let mut model = gpt5_mini("openai-responses");
        let server = MockServer::start(vec![early_eof_response()]).await;
        model.base_url = server.url.clone();
        let (events, result) = collect(stream_openai_responses(
            model,
            sign_in_context(),
            with_key("test"),
        ))
        .await;
        let AssistantMessageEvent::Start { partial } = &events[0] else {
            panic!("expected start");
        };
        assert_eq!(partial.stop_reason, StopReason::Pending);
        assert_eq!(events.last().unwrap().event_type(), "error");
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("OpenAI Responses stream ended before a terminal response event")
        );
    }

    #[tokio::test]
    async fn abort_mid_stream_fails_like_an_early_end_of_stream() {
        // The SDK swallows the abort and ends the stream, so Pi's
        // processResponsesStream reports the missing terminal event.
        let head: String = [
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({ "type": "response.output_item.added", "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] } }),
            json!({ "type": "response.output_text.delta", "output_index": 0, "delta": "Hel" }),
        ]
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect();
        let mut model = gpt5_mini("openai-responses");
        model.base_url = serve_stalled_sse(head).await;
        let signal = tokio_util::sync::CancellationToken::new();
        let mut options = with_key("test");
        options.signal = Some(signal.clone());
        let (events, result) = collect_aborting(
            stream_openai_responses(model, hi(), options),
            signal,
            "text_delta",
        )
        .await;
        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type())
                .collect::<Vec<_>>(),
            ["start", "text_start", "text_delta", "error"]
        );
        assert_eq!(result.stop_reason, StopReason::Aborted);
        assert_eq!(
            result.error_message.as_deref(),
            Some("OpenAI Responses stream ended before a terminal response event")
        );
    }

    #[tokio::test]
    async fn named_error_frames_fail_with_the_payload_message() {
        // OpenAI SDK `core/streaming.js`: `event: error` throws
        // `APIError(undefined, data?.error ?? data)`, so the message is the
        // payload's, not processResponsesStream's "Error Code ..." text.
        let body: String = [
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({ "type": "error", "code": "server_error", "message": "boom", "param": null, "sequence_number": 1 }),
        ]
        .iter()
        .map(|event| format!("event: {}\ndata: {event}\n\n", event["type"].as_str().unwrap()))
        .collect();
        let (_, result) = request_with(
            gpt5_mini("openai-responses"),
            hi(),
            with_key("test"),
            MockResponse::sse_raw(body),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn streams_text_end_to_end_with_bearer_auth() {
        let response = MockResponse::sse(&[
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({ "type": "response.output_item.added", "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] } }),
            json!({ "type": "response.output_text.delta", "output_index": 0, "delta": "Hel" }),
            json!({ "type": "response.output_text.delta", "output_index": 0, "delta": "lo" }),
            json!({ "type": "response.output_item.done", "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant",
                    "content": [{ "type": "output_text", "text": "Hello", "annotations": [] }] } }),
            json!({ "type": "response.completed", "response": { "id": "resp_1", "status": "completed" } }),
        ]);
        let (request, result) = request_with(
            gpt5_mini("openai-responses"),
            hi(),
            with_key("sk-abc"),
            response,
        )
        .await;
        assert_eq!(request.path, "/v1/responses");
        assert_eq!(request.header("authorization"), Some("Bearer sk-abc"));
        assert_eq!(
            request.header("user-agent"),
            Some(get_pi_user_agent().as_str())
        );
        assert_eq!(request.body["stream"], true);
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.response_id.as_deref(), Some("resp_1"));
        assert_eq!(
            serde_json::to_value(&result.content).unwrap(),
            json!([{ "type": "text", "text": "Hello", "textSignature": "{\"v\":1,\"id\":\"msg_1\"}" }])
        );
    }

    // cache-retention.test.ts (OpenAI Responses)
    fn cache_context() -> TranscriptContext {
        context(json!({
            "systemPrompt": "You are a helpful assistant.",
            "messages": [{ "role": "user", "content": "Hello", "timestamp": 1 }],
        }))
    }

    fn cache_options(
        retention: Option<CacheRetention>,
        session_id: Option<&str>,
    ) -> OpenAIResponsesOptions {
        let mut options = with_key("sk-fake-key");
        options.cache_retention = retention;
        options.session_id = session_id.map(str::to_string);
        options.env = Some(Default::default());
        options
    }

    #[tokio::test]
    async fn cache_retention_long_sets_prompt_cache_retention() {
        // PI_CACHE_RETENTION=long through the provider env.
        let mut options = cache_options(None, None);
        options.env = Some([("PI_CACHE_RETENTION".to_string(), "long".to_string())].into());
        let mut proxy = openai_model("gpt-4o-mini");
        proxy.base_url = "https://my-proxy.example.com/v1".to_string();
        let payload = payload_for(&proxy, cache_context(), options).await;
        assert_eq!(payload["prompt_cache_retention"], "24h");

        let mut model = openai_model("gpt-4o-mini");
        model.compat = Some(ModelCompat {
            supports_long_cache_retention: Some(false),
            ..Default::default()
        });
        let payload = payload_for(
            &model,
            cache_context(),
            cache_options(Some(CacheRetention::Long), Some("session-compat-false")),
        )
        .await;
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[tokio::test]
    async fn cache_retention_none_disables_prompt_caching() {
        let options = cache_options(Some(CacheRetention::None), Some("session-1"));
        let payload = payload_for(
            &openai_model("gpt-5.6-sol"),
            cache_context(),
            options.clone(),
        )
        .await;
        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
        assert_eq!(
            payload["prompt_cache_options"],
            json!({ "mode": "explicit" })
        );

        let payload = payload_for(&openai_model("gpt-4o-mini"), cache_context(), options).await;
        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_options").is_none());
    }

    #[tokio::test]
    async fn uses_the_supported_long_cache_field() {
        for (id, retention, cache_options_value) in [
            ("gpt-4o-mini", Some("24h"), None),
            ("gpt-6-astra", None, Some(json!({ "ttl": "30m" }))),
            ("gpt-6-sol", None, Some(json!({ "ttl": "30m" }))),
            ("gpt-6-luna", None, Some(json!({ "ttl": "30m" }))),
        ] {
            let payload = payload_for(
                &openai_model(id),
                cache_context(),
                cache_options(Some(CacheRetention::Long), Some("session-2")),
            )
            .await;
            assert_eq!(payload["prompt_cache_key"], "session-2");
            assert_eq!(
                payload
                    .get("prompt_cache_retention")
                    .and_then(Value::as_str),
                retention,
                "{id}"
            );
            assert_eq!(
                payload.get("prompt_cache_options").cloned(),
                cache_options_value,
                "{id}"
            );
        }
    }

    // sampling-options.test.ts (OpenAI Responses)
    fn sampling_model(sampling: Option<Value>, extra: Value) -> Model {
        let mut value = json!({
            "id": "custom-model", "name": "Custom Model", "api": "openai-responses", "provider": "custom-provider",
            "baseUrl": "http://127.0.0.1:9/v1", "reasoning": false, "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 128000, "maxTokens": 16384,
        });
        if let Some(sampling) = sampling {
            value["samplingParams"] = sampling;
        }
        for (key, entry) in extra.as_object().unwrap() {
            value[key] = entry.clone();
        }
        model(value)
    }

    fn sampling(value: Value) -> Option<crate::types::SamplingParams> {
        value.as_object().cloned()
    }

    #[tokio::test]
    async fn sampling_params_merge_model_level_thinking_and_request_keys() {
        let mut base = with_key("fake-key");
        base.sampling_params = sampling(json!({ "top_p": 0.5 }));
        let payload = payload_for(
            &sampling_model(Some(json!({ "top_p": 0.95, "min_p": 0.05 })), json!({})),
            cache_context(),
            base,
        )
        .await;
        assert_eq!(payload["top_p"], 0.5);
        assert_eq!(payload["min_p"], 0.05);

        let mut base = with_key("fake-key");
        base.reasoning_effort = Some(ThinkingLevel::Low);
        base.sampling_params = sampling(json!({ "top_p": 0.5 }));
        let payload = payload_for(
            &sampling_model(
                Some(json!({ "temperature": 1, "top_p": 0.95 })),
                json!({ "reasoning": true, "samplingParamsByThinkingLevel": { "low": { "temperature": 0.6, "top_k": 64 } } }),
            ),
            cache_context(),
            base,
        )
        .await;
        assert_eq!(payload["temperature"], 0.6);
        assert_eq!(payload["top_p"], 0.5);
        assert_eq!(payload["top_k"], 64);
    }

    #[tokio::test]
    async fn summary_only_requests_use_medium_sampling_params() {
        let mut base = with_key("fake-key");
        base.reasoning_summary = Some(Some("auto".to_string()));
        let payload = payload_for(
            &sampling_model(
                None,
                json!({ "reasoning": true, "samplingParamsByThinkingLevel": {
                    "off": { "temperature": 0.7 }, "medium": { "temperature": 0.8 } } }),
            ),
            cache_context(),
            base,
        )
        .await;
        assert_eq!(payload["reasoning"]["effort"], "medium");
        assert_eq!(payload["temperature"], 0.8);
    }

    // xai-responses.test.ts (request shaping)
    fn grok(id: &str) -> Model {
        model(json!({
            "id": id, "name": id, "api": "openai-responses", "provider": "xai",
            "baseUrl": "https://api.x.ai/v1", "reasoning": true, "input": ["text"],
            "thinkingLevelMap": { "off": null },
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 256000, "maxTokens": 64000,
            "compat": { "supportsLongCacheRetention": false },
        }))
    }

    #[tokio::test]
    async fn xai_requests_use_bearer_auth_and_encrypted_reasoning() {
        let mut options = with_key("xai-test-token");
        options.session_id = Some("pi-session-123".to_string());
        options.cache_retention = Some(CacheRetention::Long);
        options.reasoning_effort = Some(ThinkingLevel::Medium);
        let ctx = context(json!({
            "systemPrompt": "You are a careful coding assistant.",
            "messages": [{ "role": "user", "content": "hello", "timestamp": 1 }],
        }));
        let (request, result) =
            request_with(grok("grok-4.5"), ctx, options, completed_response()).await;
        assert_eq!(
            result.stop_reason,
            StopReason::Stop,
            "{:?}",
            result.error_message
        );
        assert_eq!(request.path, "/v1/responses");
        assert_eq!(
            request.header("authorization"),
            Some("Bearer xai-test-token")
        );
        assert_eq!(
            request.header("user-agent"),
            Some(get_pi_user_agent().as_str())
        );
        assert_eq!(request.header("session_id"), Some("pi-session-123"));
        assert_eq!(request.body["model"], "grok-4.5");
        assert_eq!(request.body["store"], false);
        assert_eq!(request.body["prompt_cache_key"], "pi-session-123");
        assert_eq!(request.body["reasoning"]["effort"], "medium");
        assert_eq!(
            request.body["include"],
            json!(["reasoning.encrypted_content"])
        );
        assert!(request.body.get("prompt_cache_retention").is_none());
        assert!(request.body["input"].as_array().unwrap().contains(&json!({
            "role": "developer", "content": "You are a careful coding assistant.",
        })));

        let ctx = context(
            json!({ "messages": [{ "role": "user", "content": "hello", "timestamp": 1 }] }),
        );
        let (request, _) = request_with(
            grok("grok-4.5"),
            ctx.clone(),
            with_key("xai-test-token"),
            completed_response(),
        )
        .await;
        assert_eq!(
            request.body["include"],
            json!(["reasoning.encrypted_content"])
        );
        assert!(request.body.get("reasoning").is_none());

        let mut options = with_key("xai-test-token");
        options.reasoning_effort = Some(ThinkingLevel::Xhigh);
        let (request, _) =
            request_with(grok("grok-4.7"), ctx.clone(), options, completed_response()).await;
        assert_eq!(request.body["reasoning"]["effort"], "xhigh");

        let mut options = with_key("xai-test-token");
        let mut headers = ProviderHeaders::new();
        headers.insert("User-Agent", Some("custom-agent".to_string()));
        options.headers = Some(headers);
        let (request, _) = request_with(grok("grok-4.5"), ctx, options, completed_response()).await;
        assert_eq!(request.header("user-agent"), Some("custom-agent"));
    }

    // provider-error-body-regression.test.ts (openai-responses)
    #[tokio::test]
    async fn http_errors_keep_the_prefix_and_surface_the_body() {
        let mut model = gpt5_mini("openai-responses");
        model.reasoning = false;
        let response = MockResponse::status(
            403,
            &[],
            json!({ "error": "blocked by gateway WAF" }).to_string(),
        );
        let (_, result) = request_with(model, sign_in_context(), with_key("test"), response).await;
        assert_eq!(result.stop_reason, StopReason::Error);
        let message = result.error_message.unwrap();
        assert!(message.contains("OpenAI API error (403)"), "{message}");
        assert!(message.contains("blocked by gateway WAF"));
    }

    #[tokio::test]
    async fn retries_retryable_http_errors() {
        let server = MockServer::start(vec![
            MockResponse::status(503, &[("retry-after-ms", "1")], "busy"),
            completed_response(),
        ])
        .await;
        let mut model = gpt5_mini("openai-responses");
        model.base_url = server.url.clone();
        let mut options = with_key("sk-test");
        options.max_retries = Some(1);
        let (_, result) = collect(stream_openai_responses(model, hi(), options)).await;
        assert_eq!(
            result.stop_reason,
            StopReason::Stop,
            "{:?}",
            result.error_message
        );
        assert_eq!(server.requests().len(), 2);
    }

    // pre-generation-error.test.ts
    #[test]
    fn stream_simple_fails_synchronously_without_auth() {
        let mut model = gpt5_mini("openai-responses");
        model.provider = "test-provider".to_string();
        let error = stream_simple_openai_responses(
            model,
            context(json!({ "messages": [] })),
            Default::default(),
        )
        .err()
        .unwrap();
        assert_eq!(error.to_string(), "No API key for provider: test-provider");
    }

    #[tokio::test]
    async fn stream_simple_maps_reasoning_and_tool_choice() {
        let model = openai_model("gpt-5.4");
        let payload = capture_payload(|hook| {
            let mut options = SimpleStreamOptions::default();
            options.api_key = Some("sk-test".to_string());
            options.on_payload = Some(hook);
            options.reasoning = Some(ThinkingLevel::High);
            options.tool_choice = Some(ToolChoice::None);
            stream_simple_openai_responses(model, hi(), options).unwrap()
        })
        .await;
        assert_eq!(
            payload["reasoning"],
            json!({ "effort": "high", "summary": "auto" })
        );
        assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(payload["tool_choice"], "none");
    }

    #[test]
    fn provider_options_map_to_responses_options() {
        let mut stream = StreamOptions::default();
        stream
            .provider_options
            .insert("reasoningEffort".to_string(), json!("low"));
        stream
            .provider_options
            .insert("reasoningSummary".to_string(), Value::Null);
        stream
            .provider_options
            .insert("serviceTier".to_string(), json!("flex"));
        stream
            .provider_options
            .insert("toolChoice".to_string(), json!("required"));
        let options = OpenAIResponsesOptions::from(stream);
        assert_eq!(options.reasoning_effort, Some(ThinkingLevel::Low));
        assert_eq!(options.reasoning_summary, Some(None));
        assert_eq!(options.service_tier.as_deref(), Some("flex"));
        assert_eq!(options.tool_choice, Some(json!("required")));
    }
}
