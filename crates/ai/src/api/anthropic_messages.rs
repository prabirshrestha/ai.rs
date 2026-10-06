//! Port of `api/anthropic-messages.ts`: the Anthropic Messages API.
//!
//! Divergences from Pi:
//! - Pi drives the request through the `@anthropic-ai/sdk` client. The port
//!   issues the same request itself: `POST {baseUrl}/v1/messages?beta=true`
//!   with the SDK's `anthropic-version: 2023-06-01`, `x-api-key` (API key) or
//!   `Authorization: Bearer` (auth token) headers, the client's default
//!   headers, and the payload's `betas` moved into the `anthropic-beta`
//!   header, as `client.beta.messages.create()` does. The SDK's
//!   `X-Stainless-*` telemetry headers are not sent.
//! - The `client` option (a pre-built SDK client) is not ported; point
//!   `Model::base_url` and `http_client` at another endpoint instead.
//! - Workload identity federation (the SDK's `config` credential exchange) is
//!   deferred: a request that would use it fails with an explicit error.
//! - SSE decoding uses the shared [`crate::utils::sse::events`] decoder in
//!   its Pi mode, which follows the line rules of Pi's inline decoder
//!   (`consumeLine`, `decodeSseLine`).
//! - Provider-specific options travel in `StreamOptions::provider_options`
//!   under Pi's field names; [`AnthropicOptions`] reads and writes them.
//!   Values of the wrong type are ignored.
//! - Pi keeps the streaming scratch fields (`index`, `partialJson`) on the
//!   content blocks; the port keeps them beside the blocks, so they never
//!   appear in `partial` snapshots.

use std::collections::HashMap;
use std::sync::Arc;

use async_stream::try_stream;
use futures::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use super::constrained_sampling::{
    get_json_schema_tool_parameters, resolve_json_schema_strict_sampling,
};
use super::github_copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use super::lazy::error_stream;
use super::simple_options::{
    adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context,
};
use super::transform_messages::transform_messages;
use crate::env_api_keys::{
    ANTHROPIC_FEDERATION_RULE_ID_ENV, ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
    ANTHROPIC_ORGANIZATION_ID_ENV, ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, ANTHROPIC_WORKSPACE_ID_ENV,
};
use crate::models::calculate_cost;
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, CacheRetention, Message, Model,
    ModelThinkingLevel, ProviderEnv, ProviderHeaders, ProviderResponse, ProviderStreams,
    SessionAffinityFormat, SimpleStreamOptions, StopReason, StreamOptions, TextContent,
    ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolChoice, ToolResultMessage,
    TranscriptContext, UserContent, UserMessageContent,
};
use crate::utils::diagnostics::{AssistantMessageDiagnostic, append_assistant_message_diagnostic};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::headers::{apply_provider_headers, has_non_empty_header, headers_to_record};
use crate::utils::http::{http_client, send_with_retries};
use crate::utils::json_parse::{parse_json_with_repair, parse_streaming_json};
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::text::{get_system_message_text, render_system_message_update};
use crate::utils::time::now_millis;
use crate::utils::transcript::{get_current_tools, get_initial_system_message, resolve_transcript};
use crate::{Error, Result};

/// Resolve cache retention preference.
/// Defaults to "short" and uses PI_CACHE_RETENTION for backward compatibility.
fn resolve_cache_retention(
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

fn get_cache_control(
    model: &Model,
    cache_retention: Option<CacheRetention>,
    env: Option<&ProviderEnv>,
) -> (CacheRetention, Option<Value>) {
    let retention = resolve_cache_retention(cache_retention, env);
    if retention == CacheRetention::None {
        return (retention, None);
    }
    let ttl = (retention == CacheRetention::Long
        && get_anthropic_compat(model).supports_long_cache_retention)
        .then_some("1h");
    let mut cache_control = Map::new();
    cache_control.insert("type".to_string(), json!("ephemeral"));
    if let Some(ttl) = ttl {
        cache_control.insert("ttl".to_string(), json!(ttl));
    }
    (retention, Some(Value::Object(cache_control)))
}

// Stealth mode: Mimic Claude Code's tool naming exactly
const CLAUDE_CODE_VERSION: &str = "2.1.280";

// Claude Code 2.x tool names (canonical casing)
// Source: https://cchistory.mariozechner.at/data/prompts-2.1.11.md
// To update: https://github.com/badlogic/cchistory
const CLAUDE_CODE_TOOLS: [&str; 17] = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

// Convert tool name to CC canonical casing if it matches (case-insensitive)
fn to_claude_code_name(name: &str) -> String {
    let lower_name = name.to_lowercase();
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|tool| tool.to_lowercase() == lower_name)
        .map_or_else(|| name.to_string(), |tool| tool.to_string())
}

fn from_claude_code_name(name: &str, tools: &[Tool]) -> String {
    if !tools.is_empty() {
        let lower_name = name.to_lowercase();
        if let Some(matched_tool) = tools
            .iter()
            .find(|tool| tool.name.to_lowercase() == lower_name)
        {
            return matched_tool.name.clone();
        }
    }
    name.to_string()
}

fn image_block(mime_type: &str, data: &str) -> Value {
    json!({
        "type": "image",
        "source": { "type": "base64", "media_type": mime_type, "data": data },
    })
}

/// Convert content blocks to Anthropic API format
fn convert_content_blocks(content: &[UserContent]) -> Value {
    // If only text blocks, return as concatenated string for simplicity
    let has_images = content
        .iter()
        .any(|block| matches!(block, UserContent::Image(_)));
    if !has_images {
        let text = content
            .iter()
            .filter_map(|block| match block {
                UserContent::Text(text) => Some(text.text.as_str()),
                UserContent::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Value::String(sanitize_surrogates(&text));
    }

    // If we have images, convert to content block array
    let mut blocks: Vec<Value> = content
        .iter()
        .map(|block| match block {
            UserContent::Text(text) => {
                json!({ "type": "text", "text": sanitize_surrogates(&text.text) })
            }
            UserContent::Image(image) => image_block(&image.mime_type, &image.data),
        })
        .collect();

    // If only images (no text), add placeholder text block
    let has_text = content
        .iter()
        .any(|block| matches!(block, UserContent::Text(_)));
    if !has_text {
        blocks.insert(0, json!({ "type": "text", "text": "(see attached image)" }));
    }

    Value::Array(blocks)
}

/// `AnthropicEffort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnthropicEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl AnthropicEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }

    /// `isAnthropicEffort()` as a parser.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::Xhigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }
}

/// `AnthropicThinkingDisplay`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnthropicThinkingDisplay {
    Summarized,
    Omitted,
}

impl AnthropicThinkingDisplay {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Summarized => "summarized",
            Self::Omitted => "omitted",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "summarized" => Some(Self::Summarized),
            "omitted" => Some(Self::Omitted),
            _ => None,
        }
    }
}

/// Anthropic tool choice (`"auto" | "any" | "none" | { type: "tool"; name }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

impl AnthropicToolChoice {
    /// The option value in Pi's shape.
    pub fn to_value(&self) -> Value {
        match self {
            Self::Auto => json!("auto"),
            Self::Any => json!("any"),
            Self::None => json!("none"),
            Self::Tool { name } => json!({ "type": "tool", "name": name }),
        }
    }

    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::String(choice) => match choice.as_str() {
                "auto" => Some(Self::Auto),
                "any" => Some(Self::Any),
                "none" => Some(Self::None),
                _ => None,
            },
            Value::Object(choice) if choice.get("type") == Some(&json!("tool")) => choice
                .get("name")
                .and_then(Value::as_str)
                .map(|name| Self::Tool {
                    name: name.to_string(),
                }),
            _ => None,
        }
    }
}

impl From<ToolChoice> for AnthropicToolChoice {
    fn from(value: ToolChoice) -> Self {
        match value {
            ToolChoice::Auto => Self::Auto,
            ToolChoice::None => Self::None,
        }
    }
}

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const SERVER_SIDE_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const MID_CONVERSATION_OUTPUT_CONFIG_BETA: &str = "mid-conversation-output-config-2026-07-01";
const THINKING_BINDING_CONTROLS_BETA: &str = "thinking-binding-controls-2026-08-01";
const INLINE_TOOLS_BETA: &str = "inline-tools-2026-09-15";

/// Stable deferred tool declared whenever native tool changes are in use. Anthropic adds
/// hidden prompt scaffolding for mid-conversation tool changes; declaring this placeholder
/// from the first request keeps that scaffolding in the cached prefix, so the first tool
/// change does not invalidate the cache (measured: full miss without it). It is never
/// activated and the model cannot see it.
fn deferred_tool_placeholder() -> Value {
    json!({
        "name": "__pi_deferred_placeholder__",
        "description": "Reserved placeholder. Never available. Never call this.",
        "input_schema": { "type": "object", "properties": {}, "required": [] },
        "defer_loading": true,
    })
}

fn should_use_server_side_fallback_beta(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.allowed_fallback_models.as_ref())
        .is_some_and(|fallbacks| !fallbacks.is_empty())
}

#[derive(Debug, Clone, Copy)]
struct AnthropicCompat {
    supports_eager_tool_input_streaming: bool,
    supports_long_cache_retention: bool,
    send_session_affinity_headers: bool,
    session_affinity_format: Option<SessionAffinityFormat>,
    supports_cache_control_on_tools: bool,
    supports_temperature: bool,
    allow_empty_signature: bool,
    supports_strict_tools: bool,
    supports_mid_convo_system_messages: bool,
    supports_mid_convo_tool_changes: bool,
}

fn get_anthropic_compat(model: &Model) -> AnthropicCompat {
    let is_open_router = model.provider == "openrouter" || model.base_url.contains("openrouter.ai");
    let compat = model.compat.clone().unwrap_or_default();
    AnthropicCompat {
        supports_eager_tool_input_streaming: compat
            .supports_eager_tool_input_streaming
            .unwrap_or(true),
        supports_long_cache_retention: compat.supports_long_cache_retention.unwrap_or(true),
        send_session_affinity_headers: compat
            .send_session_affinity_headers
            .unwrap_or(is_open_router),
        session_affinity_format: compat
            .session_affinity_format
            .or(is_open_router.then_some(SessionAffinityFormat::Openrouter)),
        supports_cache_control_on_tools: compat.supports_cache_control_on_tools.unwrap_or(true),
        supports_temperature: compat.supports_temperature.unwrap_or(true),
        allow_empty_signature: compat.allow_empty_signature.unwrap_or(false),
        supports_strict_tools: compat.supports_strict_tools.unwrap_or(false),
        supports_mid_convo_system_messages: compat
            .supports_mid_convo_system_messages
            .unwrap_or(false),
        supports_mid_convo_tool_changes: compat.supports_mid_convo_tool_changes.unwrap_or(false),
    }
}

fn supports_mid_convo_effort(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .is_some_and(|compat| compat.supports_mid_convo_effort == Some(true))
}

fn force_adaptive_thinking(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .is_some_and(|compat| compat.force_adaptive_thinking == Some(true))
}

/// `AnthropicOptions`: [`StreamOptions`] plus the Anthropic-specific fields.
/// Derefs to the inherited options.
#[derive(Clone, Default, Debug)]
pub struct AnthropicOptions {
    pub base: StreamOptions,
    /// Enable extended thinking.
    /// For adaptive thinking models: the model decides when/how much to think.
    /// For older models: uses budget-based thinking with `thinking_budget_tokens`.
    /// Default: `None` (thinking is omitted unless `stream_simple()` maps
    /// a simple reasoning level to this option, or callers set it explicitly).
    pub thinking_enabled: Option<bool>,
    /// Token budget for extended thinking (older models only).
    /// Ignored for adaptive thinking models.
    /// Default: 1024 when `thinking_enabled` is true and no budget is provided.
    pub thinking_budget_tokens: Option<u32>,
    /// Effort level for adaptive thinking models. Ignored for older models.
    /// Default: omitted unless `stream_simple()` maps a simple reasoning
    /// level to this option.
    pub effort: Option<AnthropicEffort>,
    /// Controls how thinking content is returned in API responses.
    /// Default: `Summarized` when thinking is enabled.
    pub thinking_display: Option<AnthropicThinkingDisplay>,
    /// Whether to request the interleaved thinking beta header for non-adaptive
    /// thinking models. Default: true.
    pub interleaved_thinking: Option<bool>,
    /// Anthropic tool choice behavior.
    /// Default: omitted (Anthropic default behavior, currently equivalent to auto).
    pub tool_choice: Option<AnthropicToolChoice>,
}

impl std::ops::Deref for AnthropicOptions {
    type Target = StreamOptions;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl std::ops::DerefMut for AnthropicOptions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl From<StreamOptions> for AnthropicOptions {
    fn from(options: StreamOptions) -> Self {
        Self::from_stream_options(options)
    }
}

impl AnthropicOptions {
    /// Read the Anthropic fields from `provider_options` (Pi's field names).
    pub fn from_stream_options(mut base: StreamOptions) -> Self {
        let mut take = |key: &str| base.provider_options.remove(key);
        let thinking_enabled = take("thinkingEnabled").and_then(|value| value.as_bool());
        let thinking_budget_tokens = take("thinkingBudgetTokens")
            .and_then(|value| value.as_u64())
            .map(|value| value.min(u64::from(u32::MAX)) as u32);
        let effort =
            take("effort").and_then(|value| value.as_str().and_then(AnthropicEffort::parse));
        let thinking_display = take("thinkingDisplay")
            .and_then(|value| value.as_str().and_then(AnthropicThinkingDisplay::parse));
        let interleaved_thinking = take("interleavedThinking").and_then(|value| value.as_bool());
        let tool_choice =
            take("toolChoice").and_then(|value| AnthropicToolChoice::from_value(&value));
        Self {
            base,
            thinking_enabled,
            thinking_budget_tokens,
            effort,
            thinking_display,
            interleaved_thinking,
            tool_choice,
        }
    }

    /// Fold the Anthropic fields into `provider_options` (Pi's field names).
    pub fn into_stream_options(self) -> StreamOptions {
        let mut base = self.base;
        let options = &mut base.provider_options;
        if let Some(value) = self.thinking_enabled {
            options.insert("thinkingEnabled".to_string(), json!(value));
        }
        if let Some(value) = self.thinking_budget_tokens {
            options.insert("thinkingBudgetTokens".to_string(), json!(value));
        }
        if let Some(value) = self.effort {
            options.insert("effort".to_string(), json!(value.as_str()));
        }
        if let Some(value) = self.thinking_display {
            options.insert("thinkingDisplay".to_string(), json!(value.as_str()));
        }
        if let Some(value) = self.interleaved_thinking {
            options.insert("interleavedThinking".to_string(), json!(value));
        }
        if let Some(value) = self.tool_choice {
            options.insert("toolChoice".to_string(), value.to_value());
        }
        base
    }
}

/// `mergeHeaders()`: `Object.assign` semantics, so names are matched
/// exactly; later sources replace earlier values in place.
fn merge_headers(header_sources: &[Option<&ProviderHeaders>]) -> ProviderHeaders {
    let mut merged = ProviderHeaders::new();
    for headers in header_sources.iter().flatten() {
        for (name, value) in headers.iter() {
            merged.insert(name.clone(), value.clone());
        }
    }
    merged
}

fn merge_client_headers(header_sources: &[Option<&ProviderHeaders>]) -> ProviderHeaders {
    let user_agent: ProviderHeaders = [("User-Agent", get_pi_user_agent())].into_iter().collect();
    let mut sources = vec![Some(&user_agent)];
    sources.extend_from_slice(header_sources);
    merge_headers(&sources)
}

fn has_header(headers: Option<&ProviderHeaders>, name: &str) -> bool {
    headers.is_some_and(|headers| has_non_empty_header(headers, name))
}

fn has_request_auth(api_key: Option<&str>, headers: Option<&ProviderHeaders>) -> bool {
    api_key.is_some_and(|api_key| !api_key.is_empty())
        || has_header(headers, "authorization")
        || has_header(headers, "x-api-key")
        || has_header(headers, "cf-aig-authorization")
}

fn assert_request_auth(
    provider: &str,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
) -> Result<()> {
    if !has_request_auth(api_key, headers) {
        return Err(Error::message(format!(
            "No API key for provider: {provider}"
        )));
    }
    Ok(())
}

/// Workload identity federation config from the ANTHROPIC_* variables the
/// Anthropic SDK documents (the SDK's `ClientOptions.config`). Only for the
/// anthropic provider, since the exchange is an Anthropic API endpoint, and
/// only when no key or auth header was resolved.
fn get_anthropic_federation(
    model: &Model,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
    env: Option<&ProviderEnv>,
) -> Option<Value> {
    if model.provider != "anthropic" || has_request_auth(api_key, headers) {
        return None;
    }
    let federation_rule_id = get_provider_env_value(ANTHROPIC_FEDERATION_RULE_ID_ENV, env)?;
    let organization_id = get_provider_env_value(ANTHROPIC_ORGANIZATION_ID_ENV, env)?;
    let identity_token_file = get_provider_env_value(ANTHROPIC_IDENTITY_TOKEN_FILE_ENV, env)?;
    let mut config = Map::new();
    config.insert("organization_id".to_string(), json!(organization_id));
    if let Some(workspace_id) = get_provider_env_value(ANTHROPIC_WORKSPACE_ID_ENV, env) {
        config.insert("workspace_id".to_string(), json!(workspace_id));
    }
    let mut authentication = Map::new();
    authentication.insert("type".to_string(), json!("oidc_federation"));
    authentication.insert("federation_rule_id".to_string(), json!(federation_rule_id));
    if let Some(service_account_id) = get_provider_env_value(ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, env)
    {
        authentication.insert("service_account_id".to_string(), json!(service_account_id));
    }
    authentication.insert(
        "identity_token".to_string(),
        json!({ "source": "file", "path": identity_token_file }),
    );
    config.insert("authentication".to_string(), Value::Object(authentication));
    Some(Value::Object(config))
}

const ANTHROPIC_MESSAGE_EVENTS: [&str; 6] = [
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
];

fn iterate_anthropic_events(
    response: reqwest::Response,
    signal: Option<CancellationToken>,
) -> impl Stream<Item = Result<Value>> + Send + 'static {
    try_stream! {
        let mut saw_message_start = false;
        let mut saw_message_end = false;

        let sse_events = crate::utils::sse::events(response, signal);
        futures::pin_mut!(sse_events);
        while let Some(sse) = sse_events.next().await {
            let sse = sse?;
            if sse.event.as_deref() == Some("error") {
                Err(Error::message(sse.data.clone()))?;
            }

            if !ANTHROPIC_MESSAGE_EVENTS.contains(&sse.event.as_deref().unwrap_or("")) {
                continue;
            }

            let event = match parse_json_with_repair::<Value>(&sse.data) {
                Ok(event) => event,
                Err(error) => Err(Error::message(format!(
                    "Could not parse Anthropic SSE event {}: {error}; data={}; raw={}",
                    sse.event.as_deref().unwrap_or("null"),
                    sse.data,
                    sse.raw.join("\\n")
                )))?,
            };
            match event.get("type").and_then(Value::as_str) {
                Some("message_start") => saw_message_start = true,
                Some("message_stop") => saw_message_end = true,
                _ => {}
            }
            yield event;
        }

        if saw_message_start && !saw_message_end {
            Err(Error::message("Anthropic stream ended before message_stop"))?;
        }
    }
}

fn token_count(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|count| count.min(u64::from(u32::MAX)) as u32)
}

fn update_total_tokens(output: &mut AssistantMessage) {
    // Anthropic doesn't provide total_tokens, compute from components
    let usage = &mut output.usage;
    usage.total_tokens = usage
        .input
        .saturating_add(usage.output)
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write);
}

/// Streaming scratch state Pi keeps on the content blocks (`index`,
/// `partialJson`), aligned with `output.content`.
#[derive(Default)]
struct BlockScratch {
    indices: Vec<Option<Value>>,
    partial_json: Vec<Option<String>>,
}

impl BlockScratch {
    fn push(&mut self, index: Value, partial_json: Option<String>) {
        self.indices.push(Some(index));
        self.partial_json.push(partial_json);
    }

    fn find(&self, index: &Value) -> Option<usize> {
        self.indices
            .iter()
            .position(|candidate| candidate.as_ref() == Some(index))
    }
}

/// `stream()` for the Anthropic Messages API.
pub fn stream_anthropic(
    model: Model,
    context: TranscriptContext,
    options: AnthropicOptions,
) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let normalized_context = resolve_transcript(
        &context,
        Some(get_anthropic_compat(&model).supports_mid_convo_system_messages),
    );
    let current_tools = get_current_tools(&normalized_context.messages);

    let target = stream.clone();
    tokio::spawn(async move {
        let provider_thinking_level = supports_mid_convo_effort(&model).then(|| {
            options
                .effort
                .unwrap_or(AnthropicEffort::High)
                .as_str()
                .to_string()
        });
        let mut output = AssistantMessage {
            provider_thinking_level,
            stop_reason: StopReason::Pending,
            timestamp: now_millis(),
            ..AssistantMessage::empty_for(&model)
        };

        if let Err(error) = run_stream(
            &model,
            &normalized_context,
            &current_tools,
            &options,
            &mut output,
            &target,
        )
        .await
        {
            output.stop_reason = if options
                .signal
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                StopReason::Aborted
            } else {
                StopReason::Error
            };
            output.error_message = Some(error.to_string());
            target.push(AssistantMessageEvent::Error {
                reason: output.stop_reason,
                error: output.clone(),
            });
            target.end(None);
        }
    });

    stream
}

async fn run_stream(
    model: &Model,
    normalized_context: &TranscriptContext,
    current_tools: &[Tool],
    options: &AnthropicOptions,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> Result<()> {
    let mut usage_model = model.clone();
    let mut input_transformations: Option<Vec<Value>> = None;

    let api_key = options.api_key.clone();
    let federation = get_anthropic_federation(
        model,
        api_key.as_deref(),
        options.headers.as_ref(),
        options.env.as_ref(),
    );
    if federation.is_none() {
        assert_request_auth(
            &model.provider,
            api_key.as_deref(),
            options.headers.as_ref(),
        )?;
    }

    let copilot_dynamic_headers = (model.provider == "github-copilot").then(|| {
        let has_images = has_copilot_vision_input(&normalized_context.messages);
        build_copilot_dynamic_headers(&normalized_context.messages, has_images)
    });

    let cache_retention = resolve_cache_retention(options.cache_retention, options.env.as_ref());
    let cache_session_id = if cache_retention == CacheRetention::None {
        None
    } else {
        options.session_id.clone()
    };

    let (client, is_oauth) = create_client(
        model,
        api_key,
        options.headers.as_ref(),
        options.http_client.as_ref(),
        copilot_dynamic_headers.as_ref(),
        cache_session_id.as_deref(),
        federation,
    )?;
    let mut params = build_params(model, normalized_context, is_oauth, options)?;
    if let Some(on_payload) = &options.on_payload
        && let Some(next_params) = on_payload(params.clone(), model).await?
    {
        let mut next_params = match next_params {
            Value::Object(next_params) => next_params,
            _ => Map::new(),
        };
        next_params.insert("stream".to_string(), Value::Bool(true));
        params = Value::Object(next_params);
    }
    let response = client.create_messages_response(params, options).await?;
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

    let mut blocks = BlockScratch::default();
    let events = iterate_anthropic_events(response, options.signal.clone());
    futures::pin_mut!(events);
    while let Some(event) = events.next().await {
        let event = event?;
        if let Some(on_provider_stream_event) = &options.on_provider_stream_event {
            on_provider_stream_event(&event, model).await;
        }
        let index = event.get("index").cloned().unwrap_or(Value::Null);
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "message_start" => {
                let message = event.get("message").cloned().unwrap_or(Value::Null);
                output.response_id = message
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                if let Some(transformations) = message
                    .get("input_transformations")
                    .and_then(Value::as_array)
                {
                    input_transformations = Some(transformations.clone());
                }
                let response_model = message
                    .get("model")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                if response_model.as_deref() != Some(model.id.as_str()) {
                    output.response_model = response_model.clone();
                }
                let fallback_cost = if response_model.as_deref() == Some(model.id.as_str()) {
                    None
                } else {
                    model
                        .compat
                        .as_ref()
                        .and_then(|compat| compat.allowed_fallback_models.as_ref())
                        .and_then(|fallbacks| {
                            fallbacks.iter().find(|fallback| {
                                fallback.provider == model.provider
                                    && Some(fallback.model.as_str()) == response_model.as_deref()
                            })
                        })
                        .map(|fallback| fallback.cost.clone())
                };
                usage_model = match fallback_cost {
                    Some(cost) => Model {
                        id: response_model.clone().unwrap_or_default(),
                        cost,
                        ..model.clone()
                    },
                    None => model.clone(),
                };
                // Capture initial token usage from message_start event
                // This ensures we have input token counts even if the stream is aborted early
                let usage = message.get("usage").cloned().unwrap_or(Value::Null);
                output.usage.input = token_count(&usage, "input_tokens").unwrap_or(0);
                output.usage.output = token_count(&usage, "output_tokens").unwrap_or(0);
                output.usage.cache_read =
                    token_count(&usage, "cache_read_input_tokens").unwrap_or(0);
                output.usage.cache_write =
                    token_count(&usage, "cache_creation_input_tokens").unwrap_or(0);
                output.usage.cache_write_1h = Some(
                    usage
                        .get("cache_creation")
                        .map(|cache_creation| {
                            token_count(cache_creation, "ephemeral_1h_input_tokens").unwrap_or(0)
                        })
                        .unwrap_or(0),
                );
                update_total_tokens(output);
                calculate_cost(&usage_model, &mut output.usage);
            }
            "content_block_start" => {
                let content_block = event.get("content_block").cloned().unwrap_or(Value::Null);
                let string_field = |key: &str| {
                    content_block
                        .get(key)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                };
                match content_block.get("type").and_then(Value::as_str) {
                    Some("fallback") => {
                        if !output.content.is_empty() {
                            return Err(Error::message(
                                "Anthropic performed an unsupported mid-output model fallback",
                            ));
                        }
                        continue;
                    }
                    Some("text") => {
                        output.content.push(AssistantContent::Text(TextContent::new(
                            string_field("text"),
                        )));
                        blocks.push(index, None);
                        stream.push(AssistantMessageEvent::TextStart {
                            content_index: output.content.len() - 1,
                            partial: output.clone(),
                        });
                    }
                    Some("thinking") => {
                        output
                            .content
                            .push(AssistantContent::Thinking(ThinkingContent {
                                thinking: string_field("thinking"),
                                thinking_signature: Some(string_field("signature")),
                                redacted: None,
                            }));
                        blocks.push(index, None);
                        stream.push(AssistantMessageEvent::ThinkingStart {
                            content_index: output.content.len() - 1,
                            partial: output.clone(),
                        });
                    }
                    Some("redacted_thinking") => {
                        output
                            .content
                            .push(AssistantContent::Thinking(ThinkingContent {
                                thinking: "[Reasoning redacted]".to_string(),
                                thinking_signature: content_block
                                    .get("data")
                                    .and_then(Value::as_str)
                                    .map(str::to_string),
                                redacted: Some(true),
                            }));
                        blocks.push(index, None);
                        stream.push(AssistantMessageEvent::ThinkingStart {
                            content_index: output.content.len() - 1,
                            partial: output.clone(),
                        });
                    }
                    Some("tool_use") => {
                        let name = string_field("name");
                        let arguments = match content_block.get("input") {
                            Some(input) if !input.is_null() => input.clone(),
                            _ => json!({}),
                        };
                        output.content.push(AssistantContent::ToolCall(ToolCall {
                            id: string_field("id"),
                            name: if is_oauth {
                                from_claude_code_name(&name, current_tools)
                            } else {
                                name
                            },
                            arguments,
                            thought_signature: None,
                            namespace: None,
                        }));
                        blocks.push(index, Some(String::new()));
                        stream.push(AssistantMessageEvent::ToolCallStart {
                            content_index: output.content.len() - 1,
                            partial: output.clone(),
                        });
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let delta = event.get("delta").cloned().unwrap_or(Value::Null);
                let delta_text = |key: &str| {
                    delta
                        .get(key)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                };
                let Some(content_index) = blocks.find(&index) else {
                    continue;
                };
                match (
                    delta.get("type").and_then(Value::as_str),
                    &mut output.content[content_index],
                ) {
                    (Some("text_delta"), AssistantContent::Text(block)) => {
                        let text = delta_text("text");
                        block.text.push_str(&text);
                        stream.push(AssistantMessageEvent::TextDelta {
                            content_index,
                            delta: text,
                            partial: output.clone(),
                        });
                    }
                    (Some("thinking_delta"), AssistantContent::Thinking(block)) => {
                        let thinking = delta_text("thinking");
                        block.thinking.push_str(&thinking);
                        stream.push(AssistantMessageEvent::ThinkingDelta {
                            content_index,
                            delta: thinking,
                            partial: output.clone(),
                        });
                    }
                    (Some("input_json_delta"), AssistantContent::ToolCall(block)) => {
                        let partial_json = delta_text("partial_json");
                        let buffer =
                            blocks.partial_json[content_index].get_or_insert_with(String::new);
                        buffer.push_str(&partial_json);
                        block.arguments = parse_streaming_json(Some(buffer));
                        stream.push(AssistantMessageEvent::ToolCallDelta {
                            content_index,
                            delta: partial_json,
                            partial: output.clone(),
                        });
                    }
                    (Some("signature_delta"), AssistantContent::Thinking(block)) => {
                        block
                            .thinking_signature
                            .get_or_insert_with(String::new)
                            .push_str(&delta_text("signature"));
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let Some(content_index) = blocks.find(&index) else {
                    continue;
                };
                blocks.indices[content_index] = None;
                let partial_json = blocks.partial_json[content_index].take();
                match &mut output.content[content_index] {
                    AssistantContent::Text(block) => {
                        let content = block.text.clone();
                        stream.push(AssistantMessageEvent::TextEnd {
                            content_index,
                            content,
                            partial: output.clone(),
                        });
                    }
                    AssistantContent::Thinking(block) => {
                        let content = block.thinking.clone();
                        stream.push(AssistantMessageEvent::ThinkingEnd {
                            content_index,
                            content,
                            partial: output.clone(),
                        });
                    }
                    AssistantContent::ToolCall(block) => {
                        // Finalize in-place and strip the scratch buffer so replay only
                        // carries parsed arguments.
                        block.arguments = parse_streaming_json(partial_json.as_deref());
                        let tool_call = block.clone();
                        stream.push(AssistantMessageEvent::ToolCallEnd {
                            content_index,
                            tool_call,
                            partial: output.clone(),
                        });
                    }
                }
            }
            "message_delta" => {
                if let Some(transformations) =
                    event.get("input_transformations").and_then(Value::as_array)
                {
                    input_transformations = Some(transformations.clone());
                }
                let delta = event.get("delta").cloned().unwrap_or(Value::Null);
                if let Some(stop_reason) = delta
                    .get("stop_reason")
                    .and_then(Value::as_str)
                    .filter(|stop_reason| !stop_reason.is_empty())
                {
                    output.raw_stop_reason = Some(stop_reason.to_string());
                    let (mapped, error_message) =
                        map_stop_reason(stop_reason, delta.get("stop_details"))?;
                    output.stop_reason = mapped;
                    if let Some(error_message) = error_message {
                        output.error_message = Some(error_message);
                    }
                }
                // Only update usage fields if present (not null).
                // Preserves input_tokens from message_start when proxies omit it in message_delta.
                if let Some(usage) = event.get("usage").filter(|usage| usage.is_object()) {
                    if let Some(input) = token_count(usage, "input_tokens") {
                        output.usage.input = input;
                    }
                    if let Some(output_tokens) = token_count(usage, "output_tokens") {
                        output.usage.output = output_tokens;
                    }
                    if let Some(cache_read) = token_count(usage, "cache_read_input_tokens") {
                        output.usage.cache_read = cache_read;
                    }
                    if let Some(cache_write) = token_count(usage, "cache_creation_input_tokens") {
                        output.usage.cache_write = cache_write;
                    }
                    // Vercel AI Gateway includes the TTL breakdown in deltas, though the SDK only types it on message_start.
                    if let Some(cache_write_1h) =
                        usage.get("cache_creation").and_then(|cache_creation| {
                            token_count(cache_creation, "ephemeral_1h_input_tokens")
                        })
                    {
                        output.usage.cache_write_1h = Some(cache_write_1h);
                    }
                    // Anthropic reports reasoning tokens as a subset of output tokens.
                    if let Some(thinking_tokens) = usage
                        .get("output_tokens_details")
                        .and_then(|details| token_count(details, "thinking_tokens"))
                    {
                        output.usage.reasoning = Some(thinking_tokens);
                    }
                }
                update_total_tokens(output);
                calculate_cost(&usage_model, &mut output.usage);
            }
            _ => {}
        }
    }

    if options
        .signal
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Err(Error::Aborted("Request was aborted".to_string()));
    }

    if output.stop_reason == StopReason::Pending {
        return Err(Error::message(
            "Anthropic stream ended without a stop reason",
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
    if let Some(transformations) = input_transformations.filter(|list| !list.is_empty()) {
        let transformations: Vec<Value> = transformations
            .iter()
            .map(|transformation| {
                let mut entry = Map::new();
                for key in ["type", "path", "reason"] {
                    if let Some(value) = transformation.get(key).filter(|value| !value.is_null()) {
                        entry.insert(key.to_string(), value.clone());
                    }
                }
                Value::Object(entry)
            })
            .collect();
        let mut details = Map::new();
        details.insert("transformations".to_string(), Value::Array(transformations));
        append_assistant_message_diagnostic(
            output,
            AssistantMessageDiagnostic {
                diagnostic_type: "anthropic_input_transformations".to_string(),
                timestamp: now_millis(),
                error: None,
                details: Some(details),
            },
        );
    }

    stream.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output.clone(),
    });
    stream.end(None);
    Ok(())
}

/// Map ThinkingLevel to Anthropic effort levels for adaptive thinking.
/// Note: effort "max" is available on all adaptive-thinking Claude models, while native
/// "xhigh" is only available on Opus 4.7/4.8, Sonnet 5, and Fable 5.
///
/// Pi casts any mapped string to an effort; the port only accepts the five
/// known efforts and falls back to the default mapping otherwise.
fn map_thinking_level_to_effort(model: &Model, level: ThinkingLevel) -> AnthropicEffort {
    if let Some(Some(mapped)) = model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&ModelThinkingLevel::from(level)))
        && let Some(effort) = AnthropicEffort::parse(mapped)
    {
        return effort;
    }

    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => AnthropicEffort::Low,
        ThinkingLevel::Medium => AnthropicEffort::Medium,
        ThinkingLevel::High => AnthropicEffort::High,
        _ => AnthropicEffort::High,
    }
}

/// `streamSimple()` for the Anthropic Messages API. Fails synchronously when
/// no request auth is available, like Pi's throw.
pub fn stream_simple_anthropic(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions,
) -> Result<AssistantMessageEventStream> {
    if get_anthropic_federation(
        &model,
        options.api_key.as_deref(),
        options.headers.as_ref(),
        options.env.as_ref(),
    )
    .is_none()
    {
        assert_request_auth(
            &model.provider,
            options.api_key.as_deref(),
            options.headers.as_ref(),
        )?;
    }

    let base = AnthropicOptions {
        base: build_base_options(&model, &context, Some(&options), options.api_key.as_deref()),
        tool_choice: options.tool_choice.map(AnthropicToolChoice::from),
        ..Default::default()
    };
    let Some(reasoning) = options.reasoning else {
        return Ok(stream_anthropic(
            model,
            context,
            AnthropicOptions {
                thinking_enabled: Some(false),
                ..base
            },
        ));
    };

    // For models with adaptive thinking: use an effort level.
    // For older models: use budget-based thinking.
    if force_adaptive_thinking(&model) {
        let effort = map_thinking_level_to_effort(&model, reasoning);
        return Ok(stream_anthropic(
            model,
            context,
            AnthropicOptions {
                thinking_enabled: Some(true),
                effort: Some(effort),
                ..base
            },
        ));
    }

    // `None` means the caller did not request an output cap; let the helper use the model cap.
    // Do not coerce to 0 here, or the thinking budget would become the entire max_tokens value.
    let adjusted = adjust_max_tokens_for_thinking(
        base.max_tokens,
        model.max_tokens,
        reasoning,
        options.thinking_budgets.as_ref(),
    );

    let max_tokens = clamp_max_tokens_to_context(&model, &context, adjusted.max_tokens);

    let mut base = base;
    base.base.max_tokens = Some(max_tokens);
    Ok(stream_anthropic(
        model,
        context,
        AnthropicOptions {
            thinking_enabled: Some(true),
            thinking_budget_tokens: Some(
                adjusted
                    .thinking_budget
                    .min(max_tokens.saturating_sub(1024)),
            ),
            ..base
        },
    ))
}

fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

/// The request half of the SDK client Pi builds in `createClient()`.
struct AnthropicClient {
    api_key: Option<String>,
    auth_token: Option<String>,
    base_url: String,
    http_client: reqwest::Client,
    default_headers: ProviderHeaders,
}

impl AnthropicClient {
    fn request_headers(&self, betas: Option<String>) -> Result<HeaderMap> {
        fn insert(headers: &mut HeaderMap, name: &'static str, value: &str) -> Result<()> {
            let value = HeaderValue::from_str(value)
                .map_err(|error| Error::InvalidHeaderValue(name.to_string(), error))?;
            headers.insert(HeaderName::from_static(name), value);
            Ok(())
        }

        let mut headers = HeaderMap::new();
        insert(&mut headers, "accept", "application/json")?;
        if let Some(api_key) = &self.api_key {
            insert(&mut headers, "x-api-key", api_key)?;
        }
        if let Some(auth_token) = &self.auth_token {
            insert(
                &mut headers,
                "authorization",
                &format!("Bearer {auth_token}"),
            )?;
        }
        insert(&mut headers, "anthropic-version", "2023-06-01")?;
        apply_provider_headers(&mut headers, &self.default_headers)?;
        insert(&mut headers, "content-type", "application/json")?;
        if let Some(betas) = betas {
            insert(&mut headers, "anthropic-beta", &betas)?;
        }
        Ok(headers)
    }

    /// `client.beta.messages.create(params, { signal, timeout, maxRetries: 0 }).asResponse()`
    /// wrapped in `retryProviderRequest()`.
    async fn create_messages_response(
        &self,
        params: Value,
        options: &StreamOptions,
    ) -> Result<reqwest::Response> {
        let mut body = match params {
            Value::Object(body) => body,
            _ => Map::new(),
        };
        let betas = body.shift_remove("betas").and_then(|betas| match betas {
            Value::Array(betas) if !betas.is_empty() => Some(
                betas
                    .iter()
                    .map(|beta| match beta {
                        Value::String(beta) => beta.clone(),
                        other => other.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            Value::String(betas) => Some(betas),
            _ => None,
        });
        let headers = self.request_headers(betas)?;
        let body = serde_json::to_vec(&Value::Object(body))?;
        let url = format!(
            "{}/v1/messages?beta=true",
            self.base_url.trim_end_matches('/')
        );
        // The SDK `timeout` (`fetchWithTimeout`) only runs until the response
        // headers arrive; `send_with_retries` applies it the same way.
        send_with_retries(&options.request_options(), || {
            self.http_client
                .post(&url)
                .headers(headers.clone())
                .body(body.clone())
        })
        .await
    }
}

fn create_client(
    model: &Model,
    api_key: Option<String>,
    options_headers: Option<&ProviderHeaders>,
    fetch: Option<&reqwest::Client>,
    dynamic_headers: Option<&ProviderHeaders>,
    session_id: Option<&str>,
    federation: Option<Value>,
) -> Result<(AnthropicClient, bool)> {
    let model_headers: Option<ProviderHeaders> =
        model.headers.as_ref().map(|headers| headers.clone().into());
    let http_client = http_client(fetch);

    // Copilot: Bearer auth.
    if model.provider == "github-copilot" {
        let base: ProviderHeaders = [
            ("accept", "application/json"),
            ("anthropic-dangerous-direct-browser-access", "true"),
        ]
        .into_iter()
        .map(|(name, value)| (name, value.to_string()))
        .collect();
        let client = AnthropicClient {
            api_key: None,
            auth_token: api_key,
            base_url: model.base_url.clone(),
            http_client,
            default_headers: merge_client_headers(&[
                Some(&base),
                model_headers.as_ref(),
                dynamic_headers,
                options_headers,
            ]),
        };

        return Ok((client, false));
    }

    // OAuth: Bearer auth, Claude Code identity headers
    if let Some(api_key) = api_key.as_deref().filter(|api_key| is_oauth_token(api_key)) {
        let base: ProviderHeaders = [
            ("accept", "application/json".to_string()),
            (
                "anthropic-dangerous-direct-browser-access",
                "true".to_string(),
            ),
            ("user-agent", format!("claude-cli/{CLAUDE_CODE_VERSION}")),
            ("x-app", "cli".to_string()),
        ]
        .into_iter()
        .collect();
        let client = AnthropicClient {
            api_key: None,
            auth_token: Some(api_key.to_string()),
            base_url: model.base_url.clone(),
            http_client,
            default_headers: merge_client_headers(&[
                Some(&base),
                model_headers.as_ref(),
                options_headers,
            ]),
        };

        return Ok((client, true));
    }

    // API key, header-owned auth, or workload identity federation.
    let compat = get_anthropic_compat(model);
    let mut session_affinity_headers = ProviderHeaders::new();
    if let Some(session_id) = session_id.filter(|session_id| !session_id.is_empty())
        && compat.send_session_affinity_headers
    {
        let header = if compat.session_affinity_format == Some(SessionAffinityFormat::Openrouter) {
            "x-session-id"
        } else {
            "x-session-affinity"
        };
        session_affinity_headers.insert(header, session_id.to_string());
    }
    let base: ProviderHeaders = [
        ("accept", "application/json"),
        ("anthropic-dangerous-direct-browser-access", "true"),
    ]
    .into_iter()
    .map(|(name, value)| (name, value.to_string()))
    .collect();
    let default_headers = merge_client_headers(&[
        Some(&base),
        Some(&session_affinity_headers),
        model_headers.as_ref(),
        options_headers,
    ]);
    if federation.is_some() {
        return Err(Error::message(
            "Anthropic workload identity federation is not supported yet; set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN instead",
        ));
    }

    let client = AnthropicClient {
        api_key: api_key.filter(|api_key| !api_key.is_empty()),
        auth_token: None,
        base_url: model.base_url.clone(),
        http_client,
        default_headers,
    };

    Ok((client, false))
}

fn get_beta_features(
    model: &Model,
    context: &TranscriptContext,
    is_oauth_token: bool,
    native_tool_changes: bool,
    options: &AnthropicOptions,
) -> Vec<String> {
    // `undefined` (no header), `null` (suppressed) or the configured value.
    let mut configured_features: Option<Option<String>> = None;
    for (name, value) in model.headers.iter().flatten() {
        if name.to_lowercase() == "anthropic-beta" {
            configured_features = Some(Some(value.clone()));
        }
    }
    for (name, value) in options.headers.iter().flat_map(|headers| headers.iter()) {
        if name.to_lowercase() == "anthropic-beta" {
            configured_features = Some(value.clone());
        }
    }
    match configured_features {
        Some(None) => return Vec::new(),
        Some(Some(configured_features)) => {
            return dedupe(
                configured_features
                    .split(',')
                    .map(str::trim)
                    .filter(|feature| !feature.is_empty())
                    .map(str::to_string)
                    .collect(),
            );
        }
        None => {}
    }

    let mut features: Vec<String> = Vec::new();
    if is_oauth_token {
        features.push("claude-code-20250219".to_string());
        features.push("oauth-2025-04-20".to_string());
    }
    if should_use_fine_grained_tool_streaming_beta(model, context) {
        features.push(FINE_GRAINED_TOOL_STREAMING_BETA.to_string());
    }
    if model.reasoning
        && options.thinking_enabled == Some(true)
        && options.interleaved_thinking.unwrap_or(true)
        && !force_adaptive_thinking(model)
    {
        features.push(INTERLEAVED_THINKING_BETA.to_string());
    }
    if should_use_server_side_fallback_beta(model) {
        features.push(SERVER_SIDE_FALLBACK_BETA.to_string());
    }
    if supports_mid_convo_effort(model) {
        features.push(MID_CONVERSATION_OUTPUT_CONFIG_BETA.to_string());
        features.push(THINKING_BINDING_CONTROLS_BETA.to_string());
    }
    if native_tool_changes {
        features.push(INLINE_TOOLS_BETA.to_string());
    }
    dedupe(features)
}

/// `[...new Set(values)]`.
fn dedupe(values: Vec<String>) -> Vec<String> {
    let mut unique: Vec<String> = Vec::with_capacity(values.len());
    for value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    unique
}

fn text_block(text: String, cache_control: Option<&Value>) -> Value {
    let mut block = Map::new();
    block.insert("type".to_string(), json!("text"));
    block.insert("text".to_string(), Value::String(text));
    if let Some(cache_control) = cache_control {
        block.insert("cache_control".to_string(), cache_control.clone());
    }
    Value::Object(block)
}

fn build_params(
    model: &Model,
    context: &TranscriptContext,
    is_oauth_token: bool,
    options: &AnthropicOptions,
) -> Result<Value> {
    let (_, cache_control) =
        get_cache_control(model, options.cache_retention, options.env.as_ref());
    let compat = get_anthropic_compat(model);
    let initial_system_message = get_initial_system_message(&context.messages);
    let initial_system_text = initial_system_message
        .map(get_system_message_text)
        .unwrap_or_default();
    let normalize = |id: &str, _: &Model, _: &AssistantMessage| normalize_tool_call_id(id);
    let transformed_messages = transform_messages(&context.messages, model, Some(&normalize));
    let conversation_messages = if initial_system_message.is_some() {
        &transformed_messages[1..]
    } else {
        &transformed_messages[..]
    };
    // Native tool changes keep the request-level tool list fixed and define every later tool
    // by value in a `tool_addition` block, which also expresses same-name redefinitions.
    // Anthropic rejects a tool list where every tool is deferred, so there must be an initial
    // active tool to anchor the placeholder. Otherwise the current tool list is sent.
    let initial_tools: Vec<Tool> = initial_system_message
        .and_then(|message| message.tools_added.clone())
        .unwrap_or_default();
    let native_tool_changes = compat.supports_mid_convo_system_messages
        && compat.supports_mid_convo_tool_changes
        && !initial_tools.is_empty();
    let convert_tool_definitions = |tools: &[Tool]| {
        convert_tools(
            tools,
            is_oauth_token,
            compat.supports_eager_tool_input_streaming,
            compat.supports_strict_tools,
            None,
        )
    };
    let managed_provider = supports_mid_convo_effort(model).then_some(model.provider.as_str());
    let converted = convert_messages(
        conversation_messages,
        is_oauth_token,
        cache_control.as_ref(),
        compat.allow_empty_signature,
        managed_provider,
        native_tool_changes
            .then_some(&convert_tool_definitions as &dyn Fn(&[Tool]) -> Result<Vec<Value>>),
    )?;
    let active_effort = options.effort.unwrap_or(AnthropicEffort::High);
    let beta_features =
        get_beta_features(model, context, is_oauth_token, native_tool_changes, options);
    let mut params = Map::new();
    params.insert("model".to_string(), json!(model.id));
    params.insert(
        "messages".to_string(),
        Value::Array(if supports_mid_convo_effort(model) {
            insert_thinking_level_messages(converted, active_effort)
        } else {
            converted.messages
        }),
    );
    params.insert(
        "max_tokens".to_string(),
        json!(options.max_tokens.unwrap_or(model.max_tokens)),
    );
    params.insert("stream".to_string(), json!(true));
    if !beta_features.is_empty() {
        params.insert("betas".to_string(), json!(beta_features));
    }

    // For OAuth tokens, we MUST include Claude Code identity
    if is_oauth_token {
        let mut system = vec![text_block(
            "You are Claude Code, Anthropic's official CLI for Claude.".to_string(),
            cache_control.as_ref(),
        )];
        if !initial_system_text.is_empty() {
            system.push(text_block(
                sanitize_surrogates(&initial_system_text),
                cache_control.as_ref(),
            ));
        }
        params.insert("system".to_string(), Value::Array(system));
    } else if !initial_system_text.is_empty() {
        // Add cache control to system prompt for non-OAuth tokens
        params.insert(
            "system".to_string(),
            json!([text_block(
                sanitize_surrogates(&initial_system_text),
                cache_control.as_ref()
            )]),
        );
    }

    // Temperature is incompatible with extended thinking and unsupported on Claude Opus 4.7+.
    if let Some(temperature) = options.temperature
        && options.thinking_enabled != Some(true)
        && !supports_mid_convo_effort(model)
        && compat.supports_temperature
    {
        params.insert("temperature".to_string(), json!(temperature));
    }

    let tool_cache_control = if compat.supports_cache_control_on_tools {
        cache_control.as_ref()
    } else {
        None
    };
    if native_tool_changes {
        // Initial tools stay active with the cache breakpoint on the last one, followed by the
        // placeholder. The list never changes afterwards: later tools are defined by value in
        // `tool_addition` blocks and withdrawn by `tool_removal`, so the cached prefix survives
        // every tool change.
        let mut tools = convert_tools(
            &initial_tools,
            is_oauth_token,
            compat.supports_eager_tool_input_streaming,
            compat.supports_strict_tools,
            tool_cache_control,
        )?;
        tools.push(deferred_tool_placeholder());
        params.insert("tools".to_string(), Value::Array(tools));
    } else {
        let tools = get_current_tools(&context.messages);
        if !tools.is_empty() {
            params.insert(
                "tools".to_string(),
                Value::Array(convert_tools(
                    &tools,
                    is_oauth_token,
                    compat.supports_eager_tool_input_streaming,
                    compat.supports_strict_tools,
                    tool_cache_control,
                )?),
            );
        }
    }

    // Managed effort models always use adaptive thinking so prefix mismatches can
    // be dropped instead of surfacing as persistent 400 responses.
    if supports_mid_convo_effort(model) {
        params.insert(
            "thinking".to_string(),
            json!({
                "type": "adaptive",
                "display": options
                    .thinking_display
                    .unwrap_or(AnthropicThinkingDisplay::Summarized)
                    .as_str(),
                "block_binding": { "prefix_mismatch_behavior": "drop_block" },
            }),
        );
        params.insert("output_config".to_string(), json!({ "effort": "high" }));
    } else if model.reasoning {
        if options.thinking_enabled == Some(true) {
            // Default to "summarized" so Opus 4.7 and Mythos Preview behave like
            // older Claude 4 models (whose API default is also "summarized").
            let display = options
                .thinking_display
                .unwrap_or(AnthropicThinkingDisplay::Summarized)
                .as_str();
            if force_adaptive_thinking(model) {
                // Adaptive thinking: Claude decides when and how much to think.
                params.insert(
                    "thinking".to_string(),
                    json!({ "type": "adaptive", "display": display }),
                );
                if let Some(effort) = options.effort {
                    params.insert(
                        "output_config".to_string(),
                        json!({ "effort": effort.as_str() }),
                    );
                }
            } else {
                // Budget-based thinking for older models
                params.insert(
                    "thinking".to_string(),
                    json!({
                        "type": "enabled",
                        "budget_tokens": options
                            .thinking_budget_tokens
                            .filter(|budget| *budget != 0)
                            .unwrap_or(1024),
                        "display": display,
                    }),
                );
            }
        } else if options.thinking_enabled == Some(false)
            && !matches!(
                model
                    .thinking_level_map
                    .as_ref()
                    .and_then(|map| map.get(&ModelThinkingLevel::Off)),
                Some(None)
            )
        {
            params.insert("thinking".to_string(), json!({ "type": "disabled" }));
        }
    }

    if let Some(metadata) = &options.metadata
        && let Some(user_id) = metadata.get("user_id").and_then(Value::as_str)
    {
        params.insert("metadata".to_string(), json!({ "user_id": user_id }));
    }

    if let Some(tool_choice) = &options.tool_choice {
        let tool_choice = match tool_choice.to_value() {
            Value::String(choice) => json!({ "type": choice }),
            choice => choice,
        };
        params.insert("tool_choice".to_string(), tool_choice);
    }

    if let Some(allowed_fallback_models) = model
        .compat
        .as_ref()
        .and_then(|compat| compat.allowed_fallback_models.as_ref())
        .filter(|fallbacks| !fallbacks.is_empty())
    {
        params.insert(
            "fallbacks".to_string(),
            Value::Array(
                allowed_fallback_models
                    .iter()
                    .map(|fallback| json!({ "model": fallback.model }))
                    .collect(),
            ),
        );
    }

    Ok(Value::Object(params))
}

/// Normalize tool call IDs to match Anthropic's required pattern and length.
/// JavaScript replaces and slices UTF-16 code units, so a character outside
/// the BMP becomes two underscores.
pub(crate) fn normalize_tool_call_id(id: &str) -> String {
    let mut normalized = String::with_capacity(id.len());
    for character in id.chars() {
        if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
            normalized.push(character);
        } else {
            for _ in 0..character.len_utf16() {
                normalized.push('_');
            }
        }
    }
    normalized.truncate(64);
    normalized
}

fn convert_tool_result(message: &ToolResultMessage) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": convert_content_blocks(&message.content),
        "is_error": message.is_error,
    })
}

struct ConvertedAnthropicMessages {
    messages: Vec<Value>,
    assistant_levels: HashMap<usize, AnthropicEffort>,
}

type ConvertToolDefinitions<'a> = &'a dyn Fn(&[Tool]) -> Result<Vec<Value>>;

fn convert_messages(
    transformed_messages: &[Message],
    is_oauth_token: bool,
    cache_control: Option<&Value>,
    allow_empty_signature: bool,
    managed_provider: Option<&str>,
    // Converts tool definitions for native `tool_addition` blocks; `None` when tool changes are not native.
    convert_tool_definitions: Option<ConvertToolDefinitions>,
) -> Result<ConvertedAnthropicMessages> {
    let mut params: Vec<Value> = Vec::new();
    let mut assistant_levels = HashMap::new();
    // Later system messages are held back and emitted directly before the next assistant
    // message (or at the end of the transcript). Anthropic requires `tool_result` blocks to
    // immediately follow their `tool_use`, so a system message between them is rejected; this
    // also mirrors where the managed-effort system messages are inserted. As a result an
    // update placed before a user message in the transcript lands after it on the wire.
    let mut pending_system_messages: Vec<Value> = Vec::new();

    let mut i = 0;
    while i < transformed_messages.len() {
        match &transformed_messages[i] {
            Message::System(message) => {
                // Later system messages only reach this point when the model accepts them natively;
                // otherwise the transcript was collapsed into the leading message before conversion.
                let text = render_system_message_update(message);
                let mut blocks: Vec<Value> = Vec::new();
                if !text.is_empty() {
                    blocks.push(json!({ "type": "text", "text": sanitize_surrogates(&text) }));
                }
                if let Some(convert_tool_definitions) = convert_tool_definitions {
                    let added = message.tools_added.clone().unwrap_or_default();
                    let redefined: Vec<&str> =
                        added.iter().map(|tool| tool.name.as_str()).collect();
                    for tool in message.tools_removed.iter().flatten() {
                        // A new definition under the same name replaces the old one, so no removal is needed.
                        if redefined.contains(&tool.name.as_str()) {
                            continue;
                        }
                        let name = if is_oauth_token {
                            to_claude_code_name(&tool.name)
                        } else {
                            tool.name.clone()
                        };
                        blocks.push(json!({
                            "type": "tool_removal",
                            "tool": { "type": "tool_reference", "name": name },
                        }));
                    }
                    for definition in convert_tool_definitions(&added)? {
                        blocks.push(json!({
                            "type": "tool_addition",
                            "tool": { "type": "tool_definition", "definition": definition },
                        }));
                    }
                }
                if !blocks.is_empty() {
                    pending_system_messages.push(json!({ "role": "system", "content": blocks }));
                }
            }
            Message::User(message) => match &message.content {
                UserMessageContent::Text(content) => {
                    if !content.trim().is_empty() {
                        params.push(json!({
                            "role": "user",
                            "content": sanitize_surrogates(content),
                        }));
                    }
                }
                UserMessageContent::Parts(parts) => {
                    let filtered_blocks: Vec<Value> = parts
                        .iter()
                        .filter_map(|item| {
                            match item {
                            UserContent::Text(text) => (!text.text.trim().is_empty()).then(|| {
                                json!({ "type": "text", "text": sanitize_surrogates(&text.text) })
                            }),
                            UserContent::Image(image) => {
                                Some(image_block(&image.mime_type, &image.data))
                            }
                        }
                        })
                        .collect();
                    if !filtered_blocks.is_empty() {
                        params.push(json!({ "role": "user", "content": filtered_blocks }));
                    }
                }
            },
            Message::Assistant(message) => {
                params.append(&mut pending_system_messages);
                let mut blocks: Vec<Value> = Vec::new();

                for block in &message.content {
                    match block {
                        AssistantContent::Text(text) => {
                            if text.text.trim().is_empty() {
                                continue;
                            }
                            blocks.push(
                                json!({ "type": "text", "text": sanitize_surrogates(&text.text) }),
                            );
                        }
                        AssistantContent::Thinking(thinking) => {
                            // Redacted thinking: pass the opaque payload back as redacted_thinking
                            if thinking.redacted == Some(true) {
                                blocks.push(json!({
                                    "type": "redacted_thinking",
                                    "data": thinking.thinking_signature,
                                }));
                                continue;
                            }
                            let thinking_signature = thinking
                                .thinking_signature
                                .as_deref()
                                .filter(|signature| !signature.trim().is_empty());
                            if thinking.thinking.trim().is_empty() && thinking_signature.is_none() {
                                continue;
                            }
                            // If thinking signature is missing/empty (e.g., from aborted stream),
                            // convert to plain text for Anthropic. Some compatible providers emit
                            // and accept empty signatures, so let marked models preserve the block.
                            match thinking_signature {
                                None if allow_empty_signature => blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(&thinking.thinking),
                                    "signature": "",
                                })),
                                None => blocks.push(json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&thinking.thinking),
                                })),
                                Some(signature) => blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(&thinking.thinking),
                                    "signature": signature,
                                })),
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let name = if is_oauth_token {
                                to_claude_code_name(&tool_call.name)
                            } else {
                                tool_call.name.clone()
                            };
                            let input = if tool_call.arguments.is_null() {
                                json!({})
                            } else {
                                tool_call.arguments.clone()
                            };
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": name,
                                "input": input,
                            }));
                        }
                    }
                }
                if !blocks.is_empty() {
                    let message_index = params.len();
                    params.push(json!({ "role": "assistant", "content": blocks }));
                    if let Some(managed_provider) = managed_provider
                        && message.api == "anthropic-messages"
                        && message.provider == managed_provider
                        && let Some(effort) = message
                            .provider_thinking_level
                            .as_deref()
                            .and_then(AnthropicEffort::parse)
                    {
                        assistant_levels.insert(message_index, effort);
                    }
                }
            }
            Message::ToolResult(_) => {
                // Collect all consecutive toolResult messages, needed for z.ai Anthropic endpoint.
                let mut tool_results: Vec<Value> = Vec::new();
                let mut j = i;
                while let Some(Message::ToolResult(result)) = transformed_messages.get(j) {
                    tool_results.push(convert_tool_result(result));
                    j += 1;
                }

                // Skip the messages we've already processed.
                i = j - 1;

                params.push(json!({ "role": "user", "content": tool_results }));
            }
        }
        i += 1;
    }

    params.append(&mut pending_system_messages);

    // Add cache_control to the last user or system message to cache conversation history
    if let Some(cache_control) = cache_control
        && let Some(last_message) = params.last_mut()
    {
        let role = last_message.get("role").and_then(Value::as_str);
        if matches!(role, Some("user") | Some("system")) {
            match last_message.get_mut("content") {
                Some(Value::Array(content)) => {
                    if let Some(last_block) = content.last_mut()
                        && matches!(
                            last_block.get("type").and_then(Value::as_str),
                            Some("text")
                                | Some("image")
                                | Some("tool_result")
                                | Some("tool_addition")
                                | Some("tool_removal")
                        )
                        && let Some(last_block) = last_block.as_object_mut()
                    {
                        last_block.insert("cache_control".to_string(), cache_control.clone());
                    }
                }
                Some(content @ Value::String(_)) => {
                    let text = content.as_str().unwrap_or_default().to_string();
                    *content = json!([text_block(text, Some(cache_control))]);
                }
                _ => {}
            }
        }
    }

    Ok(ConvertedAnthropicMessages {
        messages: params,
        assistant_levels,
    })
}

fn insert_thinking_level_messages(
    converted: ConvertedAnthropicMessages,
    active_effort: AnthropicEffort,
) -> Vec<Value> {
    let mut messages = Vec::with_capacity(converted.messages.len() + 1);
    for (index, message) in converted.messages.into_iter().enumerate() {
        if let Some(historical_effort) = converted.assistant_levels.get(&index) {
            messages.push(json!({
                "role": "system",
                "content": [],
                "output_config": { "effort": historical_effort.as_str() },
            }));
        }
        messages.push(message);
    }
    messages.push(json!({
        "role": "system",
        "content": [],
        "output_config": { "effort": active_effort.as_str() },
    }));
    messages
}

fn should_use_fine_grained_tool_streaming_beta(model: &Model, context: &TranscriptContext) -> bool {
    !get_current_tools(&context.messages).is_empty()
        && !get_anthropic_compat(model).supports_eager_tool_input_streaming
}

// Keywords Anthropic strict tool use rejects with a 400 for the whole request.
// https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
const ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS: [&str; 11] = [
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "maxItems",
    "uniqueItems",
    "minContains",
    "maxContains",
    "minProperties",
    "maxProperties",
];
const ANTHROPIC_STRICT_STRING_FORMATS: [&str; 10] = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

fn is_anthropic_strict_unsupported_keyword(key: &str, value: &Value) -> bool {
    if ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS.contains(&key) {
        return true;
    }
    if key == "minItems" {
        return !value
            .as_f64()
            .is_some_and(|value| value == 0.0 || value == 1.0);
    }
    if key == "format" {
        return !value
            .as_str()
            .is_some_and(|format| ANTHROPIC_STRICT_STRING_FORMATS.contains(&format));
    }
    false
}

fn convert_tools(
    tools: &[Tool],
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: Option<&Value>,
) -> Result<Vec<Value>> {
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let strict = resolve_json_schema_strict_sampling(
                tool,
                supports_strict_tools,
                Some(&is_anthropic_strict_unsupported_keyword),
            )?;
            let parameters = get_json_schema_tool_parameters(tool, strict)?;
            let properties = parameters
                .get("properties")
                .filter(|properties| !properties.is_null())
                .cloned()
                .unwrap_or_else(|| json!({}));
            let required = parameters
                .get("required")
                .filter(|required| !required.is_null())
                .cloned()
                .unwrap_or_else(|| json!([]));
            let mut input_schema = match (strict == Some(true), parameters) {
                (true, Value::Object(parameters)) => parameters,
                _ => Map::new(),
            };
            input_schema.insert("type".to_string(), json!("object"));
            input_schema.insert("properties".to_string(), properties);
            input_schema.insert("required".to_string(), required);

            let mut converted = Map::new();
            converted.insert(
                "name".to_string(),
                json!(if is_oauth_token {
                    to_claude_code_name(&tool.name)
                } else {
                    tool.name.clone()
                }),
            );
            converted.insert("description".to_string(), json!(tool.description));
            if supports_eager_tool_input_streaming {
                converted.insert("eager_input_streaming".to_string(), json!(true));
            }
            if strict == Some(true) {
                converted.insert("strict".to_string(), json!(true));
            }
            converted.insert("input_schema".to_string(), Value::Object(input_schema));
            if let Some(cache_control) = cache_control
                && index == tools.len() - 1
            {
                converted.insert("cache_control".to_string(), cache_control.clone());
            }
            Ok(Value::Object(converted))
        })
        .collect()
}

fn map_stop_reason(
    reason: &str,
    stop_details: Option<&Value>,
) -> Result<(StopReason, Option<String>)> {
    Ok(match reason {
        "end_turn" => (StopReason::Stop, None),
        "max_tokens" => (StopReason::Length, None),
        "tool_use" => (StopReason::ToolUse, None),
        "refusal" => (
            StopReason::Error,
            Some(
                stop_details
                    .and_then(|details| details.get("explanation"))
                    .and_then(Value::as_str)
                    .filter(|explanation| !explanation.is_empty())
                    .unwrap_or("The model refused to complete the request")
                    .to_string(),
            ),
        ),
        // Stop is good enough -> resubmit
        "pause_turn" => (StopReason::Stop, None),
        // We don't supply stop sequences, so this should never happen
        "stop_sequence" => (StopReason::Stop, None),
        // Content flagged by safety filters (not yet in SDK types)
        "sensitive" => (
            StopReason::Error,
            Some("Provider stopped with: sensitive".to_string()),
        ),
        // Handle unknown stop reasons gracefully (API may add new values)
        _ => {
            return Err(Error::message(format!("Unhandled stop reason: {reason}")));
        }
    })
}

/// The `anthropic-messages` implementation as `ProviderStreams`.
struct AnthropicMessagesApi;

impl ProviderStreams for AnthropicMessagesApi {
    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        stream_anthropic(
            model,
            context,
            AnthropicOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let fallback = model.clone();
        stream_simple_anthropic(model, context, options)
            .unwrap_or_else(|error| error_stream(&fallback, error))
    }
}

/// The `anthropic-messages` implementation as `ProviderStreams`.
pub fn anthropic_messages_api() -> Arc<dyn ProviderStreams> {
    Arc::new(AnthropicMessagesApi)
}

#[cfg(test)]
// Tests set inherited options through `Deref`, which this lint cannot see.
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::auth::AuthContext;
    use crate::models::{CreateModelsOptions, create_models};
    use crate::providers::all::get_builtin_model;
    use crate::providers::anthropic::anthropic_provider;
    use crate::types::{Context, PayloadHook};
    use crate::utils::transcript::normalize_context;

    #[derive(Debug, Clone)]
    struct CapturedRequest {
        path: String,
        headers: Vec<(String, String)>,
        body: Value,
    }

    impl CapturedRequest {
        fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .iter()
                .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.as_str())
        }
    }

    type Requests = Arc<Mutex<Vec<CapturedRequest>>>;

    /// A local HTTP server answering every request with `body` as an SSE
    /// stream, standing in for Pi's fake SDK clients and fetch mocks.
    async fn serve_sse(body: String) -> (String, Requests) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests: Requests = Arc::default();
        let captured = Arc::clone(&requests);
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let header_end = loop {
                    let read = socket.read(&mut chunk).await.unwrap_or(0);
                    if read == 0 {
                        break None;
                    }
                    buffer.extend_from_slice(&chunk[..read]);
                    if let Some(position) =
                        buffer.windows(4).position(|window| window == b"\r\n\r\n")
                    {
                        break Some(position + 4);
                    }
                };
                let Some(header_end) = header_end else {
                    continue;
                };
                let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
                let mut lines = head.split("\r\n");
                let path = lines
                    .next()
                    .and_then(|line| line.split(' ').nth(1))
                    .unwrap_or_default()
                    .to_string();
                let headers: Vec<(String, String)> = lines
                    .filter_map(|line| line.split_once(": "))
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect();
                let length = headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                    .and_then(|(_, value)| value.parse::<usize>().ok())
                    .unwrap_or(0);
                while buffer.len() < header_end + length {
                    let read = socket.read(&mut chunk).await.unwrap_or(0);
                    if read == 0 {
                        break;
                    }
                    buffer.extend_from_slice(&chunk[..read]);
                }
                let body_value =
                    serde_json::from_slice(&buffer[header_end..]).unwrap_or(Value::Null);
                captured.lock().unwrap().push(CapturedRequest {
                    path,
                    headers,
                    body: body_value,
                });
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        (format!("http://{address}"), requests)
    }

    fn sse(events: &[Value]) -> String {
        events
            .iter()
            .map(|event| {
                format!(
                    "event: {}\ndata: {event}\n\n",
                    event["type"].as_str().unwrap_or_default()
                )
            })
            .collect()
    }

    fn sse_raw(events: &[(&str, String)]) -> String {
        events
            .iter()
            .map(|(event, data)| format!("event: {event}\ndata: {data}\n\n"))
            .collect()
    }

    fn minimal_events() -> Vec<Value> {
        vec![
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            }),
            json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
            json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "Hello" } }),
            json!({ "type": "content_block_stop", "index": 0 }),
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            }),
            json!({ "type": "message_stop" }),
        ]
    }

    fn response_model_events(model: &str, content_block: Value) -> Vec<Value> {
        vec![
            json!({
                "type": "message_start",
                "message": { "id": "msg_response_model", "model": model, "usage": { "input_tokens": 100, "output_tokens": 0 } },
            }),
            json!({ "type": "content_block_start", "index": 0, "content_block": content_block }),
            json!({ "type": "content_block_stop", "index": 0 }),
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "input_tokens": 100, "output_tokens": 20 },
            }),
            json!({ "type": "message_stop" }),
        ]
    }

    fn builtin(provider: &str, id: &str) -> Model {
        get_builtin_model(provider, id).unwrap()
    }

    fn model_from(value: Value) -> Model {
        serde_json::from_value(value).unwrap()
    }

    fn custom_model(id: &str, provider: &str, compat: Value) -> Model {
        let mut model = json!({
            "id": id,
            "name": id,
            "api": "anthropic-messages",
            "provider": provider,
            "baseUrl": "http://127.0.0.1:9",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 200000,
            "maxTokens": 32000,
        });
        if !compat.is_null() {
            model["compat"] = compat;
        }
        model_from(model)
    }

    fn context(value: Value) -> TranscriptContext {
        normalize_context(&serde_json::from_value::<Context>(value).unwrap())
    }

    fn hello() -> TranscriptContext {
        context(json!({ "messages": [{ "role": "user", "content": "Hello", "timestamp": 1 }] }))
    }

    fn system_hello() -> TranscriptContext {
        context(json!({
            "systemPrompt": "You are a helpful assistant.",
            "messages": [{ "role": "user", "content": "Hello", "timestamp": 1 }],
        }))
    }

    fn assistant_json(provider: &str, model: &str, content: Value, extra: Value) -> Value {
        let mut message = json!({
            "role": "assistant",
            "content": content,
            "api": "anthropic-messages",
            "provider": provider,
            "model": model,
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
            },
            "stopReason": "stop",
            "timestamp": 1,
        });
        if let (Value::Object(message), Value::Object(extra)) = (&mut message, extra) {
            message.extend(extra);
        }
        message
    }

    type Slot = Arc<Mutex<Option<Value>>>;

    /// Pi's `onPayload` capture: record the payload, then fail the request.
    fn capture_hook() -> (PayloadHook, Slot) {
        let slot: Slot = Arc::default();
        let target = Arc::clone(&slot);
        let hook: PayloadHook = Arc::new(move |payload, _| {
            *target.lock().unwrap() = Some(payload);
            Box::pin(async { Err(Error::message("payload captured")) })
        });
        (hook, slot)
    }

    async fn capture(
        model: Model,
        context: TranscriptContext,
        mut options: AnthropicOptions,
    ) -> (Value, AssistantMessage) {
        let (hook, slot) = capture_hook();
        options.on_payload = Some(hook);
        if options.api_key.is_none() {
            options.api_key = Some("test-key".to_string());
        }
        let message = stream_anthropic(model, context, options).result().await;
        let payload = slot.lock().unwrap().take().expect("payload captured");
        (payload, message)
    }

    async fn capture_simple(
        model: Model,
        context: TranscriptContext,
        mut options: SimpleStreamOptions,
    ) -> Value {
        let (hook, slot) = capture_hook();
        options.stream.on_payload = Some(hook);
        options.stream.api_key = Some("fake-key".to_string());
        stream_simple_anthropic(model, context, options)
            .unwrap()
            .result()
            .await;
        slot.lock().unwrap().take().expect("payload captured")
    }

    fn api_key(key: &str) -> AnthropicOptions {
        AnthropicOptions {
            base: StreamOptions {
                api_key: Some(key.to_string()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    async fn run(
        mut model: Model,
        context: TranscriptContext,
        mut options: AnthropicOptions,
        body: String,
    ) -> (AssistantMessage, Option<CapturedRequest>) {
        let (base_url, requests) = serve_sse(body).await;
        model.base_url = base_url;
        if options.api_key.is_none() && options.headers.is_none() {
            options.api_key = Some("test-key".to_string());
        }
        let message = stream_anthropic(model, context, options).result().await;
        let request = requests.lock().unwrap().first().cloned();
        (message, request)
    }

    /// A server that sends the response headers and the first `split` SSE
    /// events, waits `delay`, then sends the rest. With `headers_delay`, it
    /// waits before sending anything. Counts the requests it accepted.
    async fn serve_slow_sse(
        events: Vec<Value>,
        split: usize,
        headers_delay: Option<std::time::Duration>,
        delay: std::time::Duration,
    ) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let (head, tail) = events.split_at(split);
                let (head, tail) = (sse(head), sse(tail));
                tokio::spawn(async move {
                    let mut chunk = [0u8; 8192];
                    let _ = socket.read(&mut chunk).await;
                    if let Some(headers_delay) = headers_delay {
                        tokio::time::sleep(headers_delay).await;
                    }
                    let _ = socket
                        .write_all(
                            format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{head}")
                                .as_bytes(),
                        )
                        .await;
                    let _ = socket.flush().await;
                    tokio::time::sleep(delay).await;
                    let _ = socket.write_all(tail.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        (format!("http://{address}"), attempts)
    }

    #[tokio::test]
    async fn timeout_only_bounds_the_wait_for_response_headers() {
        let mut model = builtin("anthropic", "claude-haiku-4-5");
        let (base_url, attempts) = serve_slow_sse(
            minimal_events(),
            2,
            None,
            std::time::Duration::from_millis(400),
        )
        .await;
        model.base_url = base_url;
        let mut options = api_key("test-key");
        options.timeout_ms = Some(100);
        let result = stream_anthropic(model, hello(), options).result().await;
        assert_eq!(
            result.stop_reason,
            StopReason::Stop,
            "{:?}",
            result.error_message
        );
        assert_eq!(
            result.content,
            vec![AssistantContent::Text(TextContent::new("Hello"))]
        );
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn header_timeouts_read_like_the_sdk_and_are_retried() {
        let mut model = builtin("anthropic", "claude-haiku-4-5");
        let (base_url, attempts) = serve_slow_sse(
            minimal_events(),
            0,
            Some(std::time::Duration::from_secs(5)),
            std::time::Duration::ZERO,
        )
        .await;
        model.base_url = base_url;
        let mut options = api_key("test-key");
        options.timeout_ms = Some(50);
        options.max_retries = Some(1);
        let result = stream_anthropic(model, hello(), options).result().await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some("Request timed out."));
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    // anthropic-sse-parsing.test.ts

    #[tokio::test]
    async fn forwards_parsed_provider_stream_events_in_order() {
        let model = builtin("anthropic", "claude-haiku-4-5");
        let seen: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
        let target = Arc::clone(&seen);
        let mut options = AnthropicOptions::default();
        options.on_provider_stream_event = Some(Arc::new(move |event, model| {
            target.lock().unwrap().push((
                event["type"].as_str().unwrap_or_default().to_string(),
                model.id.clone(),
            ));
            Box::pin(async {})
        }));
        let (result, _) = run(model, hello(), options, sse(&minimal_events())).await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        let seen = seen.lock().unwrap().clone();
        assert_eq!(
            seen.iter()
                .map(|(kind, _)| kind.as_str())
                .collect::<Vec<_>>(),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert!(seen.iter().all(|(_, model)| model == "claude-haiku-4-5"));
    }

    #[tokio::test]
    async fn keeps_signed_thinking_replayable_when_a_proxy_relabels_the_model() {
        let model = builtin("anthropic", "claude-opus-5");
        let initial = hello();
        let (first, _) = run(
            model.clone(),
            initial.clone(),
            AnthropicOptions::default(),
            sse(&response_model_events(
                "kimi-for-coding",
                json!({ "type": "thinking", "thinking": "reasoning", "signature": "signature" }),
            )),
        )
        .await;
        assert_eq!(first.model, model.id);
        assert_eq!(first.response_model.as_deref(), Some("kimi-for-coding"));

        let mut messages = initial.messages.clone();
        messages.push(Message::Assistant(first));
        let transformed = transform_messages(&messages, &model, None);
        let Some(Message::Assistant(replayed)) = transformed
            .iter()
            .find(|message| matches!(message, Message::Assistant(_)))
        else {
            panic!("expected an assistant message");
        };
        assert_eq!(
            replayed.content,
            vec![AssistantContent::Thinking(ThinkingContent {
                thinking: "reasoning".to_string(),
                thinking_signature: Some("signature".to_string()),
                redacted: None,
            })]
        );
    }

    #[tokio::test]
    async fn uses_a_returned_fallback_model_for_cost_attribution() {
        let mut model = builtin("anthropic", "claude-opus-5");
        model.compat = Some(
            serde_json::from_value(json!({
                "allowedFallbackModels": [{
                    "provider": "anthropic",
                    "model": "fallback-model",
                    "cost": { "input": 3, "output": 5, "cacheRead": 0, "cacheWrite": 0 },
                }],
            }))
            .unwrap(),
        );
        let (result, request) = run(
            model.clone(),
            hello(),
            AnthropicOptions::default(),
            sse(&response_model_events(
                "fallback-model",
                json!({ "type": "text", "text": "done" }),
            )),
        )
        .await;
        assert_eq!(result.model, model.id);
        assert_eq!(result.response_model.as_deref(), Some("fallback-model"));
        assert!((result.usage.cost.input - 0.0003).abs() < 1e-10);
        assert!((result.usage.cost.output - 0.0001).abs() < 1e-10);
        let request = request.unwrap();
        assert_eq!(
            request.body["fallbacks"],
            json!([{ "model": "fallback-model" }])
        );
        assert!(
            request
                .header("anthropic-beta")
                .unwrap()
                .contains(SERVER_SIDE_FALLBACK_BETA)
        );
    }

    #[tokio::test]
    async fn fails_safely_when_anthropic_falls_back_after_output_begins() {
        let events = vec![
            json!({
                "type": "message_start",
                "message": { "id": "msg_fallback", "model": "claude-opus-5", "usage": { "input_tokens": 1, "output_tokens": 0 } },
            }),
            json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "partial" } }),
            json!({ "type": "content_block_stop", "index": 0 }),
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": { "type": "fallback", "from": { "model": "claude-opus-5" }, "to": { "model": "claude-opus-4-8" } },
            }),
        ];
        let (result, _) = run(
            builtin("anthropic", "claude-opus-5"),
            hello(),
            AnthropicOptions::default(),
            sse(&events),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(
            result
                .error_message
                .unwrap()
                .contains("unsupported mid-output model fallback")
        );
    }

    #[tokio::test]
    async fn forces_streaming_after_an_on_payload_replacement() {
        let mut options = AnthropicOptions::default();
        options.on_payload = Some(Arc::new(|payload, _| {
            Box::pin(async move {
                let mut payload = payload;
                payload["stream"] = json!(false);
                Ok(Some(payload))
            })
        }));
        let (_, request) = run(
            builtin("anthropic", "claude-fable-5-1"),
            hello(),
            options,
            sse(&minimal_events()),
        )
        .await;
        assert_eq!(request.unwrap().body["stream"], json!(true));
    }

    #[tokio::test]
    async fn omits_the_interleaved_thinking_beta_when_thinking_is_disabled() {
        let model = model_from(json!({
            "id": "anthropic/claude-haiku-4.5",
            "name": "Claude Haiku 4.5",
            "api": "anthropic-messages",
            "provider": "openrouter",
            "baseUrl": "https://openrouter.ai/api",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 1, "output": 5, "cacheRead": 0.1, "cacheWrite": 1.25 },
            "contextWindow": 200000,
            "maxTokens": 64000,
        }));
        let (_, request) = run(
            model,
            hello(),
            AnthropicOptions {
                thinking_enabled: Some(false),
                ..Default::default()
            },
            sse(&minimal_events()),
        )
        .await;
        let request = request.unwrap();
        assert!(
            !request
                .header("anthropic-beta")
                .unwrap_or_default()
                .contains(INTERLEAVED_THINKING_BETA)
        );
        assert_eq!(request.path, "/v1/messages?beta=true");
    }

    #[tokio::test]
    async fn passes_managed_beta_features() {
        let (result, request) = run(
            builtin("anthropic", "claude-fable-5-1"),
            hello(),
            AnthropicOptions::default(),
            sse(&minimal_events()),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        let request = request.unwrap();
        let betas = request.header("anthropic-beta").unwrap();
        assert!(betas.contains(MID_CONVERSATION_OUTPUT_CONFIG_BETA));
        assert!(betas.contains(THINKING_BINDING_CONTROLS_BETA));
        // The SDK moves `betas` into the header.
        assert!(request.body.get("betas").is_none());
    }

    #[tokio::test]
    async fn uses_the_serving_model_input_transformations_from_the_final_stream_event() {
        let mut events = minimal_events();
        events[0] = json!({
            "type": "message_start",
            "message": {
                "id": "msg_transformations",
                "model": "claude-fable-5-1",
                "usage": { "input_tokens": 12, "output_tokens": 0 },
                "input_transformations": [
                    { "type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch" },
                ],
            },
        });
        events[4]["input_transformations"] = json!([
            { "type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch" },
        ]);
        let (result, _) = run(
            builtin("anthropic", "claude-fable-5-1"),
            hello(),
            AnthropicOptions::default(),
            sse(&events),
        )
        .await;
        let diagnostics = result.diagnostics.unwrap();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].diagnostic_type,
            "anthropic_input_transformations"
        );
        assert!(diagnostics[0].error.is_none());
        assert_eq!(
            Value::Object(diagnostics[0].details.clone().unwrap()),
            json!({
                "transformations": [
                    { "type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch" },
                ],
            })
        );
    }

    #[tokio::test]
    async fn repairs_malformed_sse_json_and_malformed_streamed_tool_json() {
        let context = context(json!({
            "messages": [{ "role": "user", "content": "Use the edit tool.", "timestamp": 1 }],
            "tools": [{
                "name": "edit",
                "description": "Edit a file.",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" }, "text": { "type": "string" } },
                    "required": ["path", "text"],
                },
            }],
        }));
        let malformed = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"A\\H\\\",\\\"text\\\":\\\"col1\tcol2\\\"}\"}}";
        let body = sse_raw(&[
            ("message_start", minimal_events()[0].to_string()),
            (
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "tool_use", "id": "toolu_test", "name": "edit", "input": {} },
                })
                .to_string(),
            ),
            ("content_block_delta", malformed.to_string()),
            ("content_block_stop", json!({ "type": "content_block_stop", "index": 0 }).to_string()),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "tool_use" },
                    "usage": { "input_tokens": 12, "output_tokens": 5 },
                })
                .to_string(),
            ),
            ("message_stop", json!({ "type": "message_stop" }).to_string()),
        ]);
        let (result, _) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            context,
            AnthropicOptions::default(),
            body,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::ToolUse);
        assert_eq!(result.error_message, None);
        let tool_call = result
            .content
            .iter()
            .find_map(|block| match block {
                AssistantContent::ToolCall(tool_call) => Some(tool_call),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            tool_call.arguments,
            json!({ "path": "A\\H", "text": "col1\tcol2" })
        );
    }

    #[tokio::test]
    async fn preserves_content_from_content_block_start_events() {
        let events = vec![
            minimal_events()[0].clone(),
            json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "Initial text" } }),
            json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": " plus delta" } }),
            json!({ "type": "content_block_stop", "index": 0 }),
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": { "type": "thinking", "thinking": "Initial thinking", "signature": "initial signature" },
            }),
            json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "thinking_delta", "thinking": " plus delta" } }),
            json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "signature_delta", "signature": " plus delta" } }),
            json!({ "type": "content_block_stop", "index": 1 }),
            minimal_events()[4].clone(),
            json!({ "type": "message_stop" }),
        ];
        let (result, _) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            hello(),
            AnthropicOptions::default(),
            sse(&events),
        )
        .await;
        assert_eq!(
            result.content,
            vec![
                AssistantContent::Text(TextContent::new("Initial text plus delta")),
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "Initial thinking plus delta".to_string(),
                    thinking_signature: Some("initial signature plus delta".to_string()),
                    redacted: None,
                }),
            ]
        );
    }

    fn stop_events(stop_reason: &str, stop_details: Option<Value>) -> Vec<Value> {
        let mut delta = json!({ "stop_reason": stop_reason });
        if let Some(stop_details) = stop_details {
            delta["stop_details"] = stop_details;
        }
        vec![
            minimal_events()[0].clone(),
            json!({ "type": "message_delta", "delta": delta, "usage": { "input_tokens": 12, "output_tokens": 0 } }),
            json!({ "type": "message_stop" }),
        ]
    }

    #[tokio::test]
    async fn preserves_refusal_stop_details_from_message_delta() {
        let explanation = "This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage Policy.";
        let (result, _) = run(
            builtin("anthropic", "claude-fable-5"),
            hello(),
            AnthropicOptions::default(),
            sse(&stop_events(
                "refusal",
                Some(json!({ "type": "refusal", "category": "cyber", "explanation": explanation })),
            )),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.raw_stop_reason.as_deref(), Some("refusal"));
        assert_eq!(result.error_message.as_deref(), Some(explanation));

        let (result, _) = run(
            builtin("anthropic", "claude-fable-5"),
            hello(),
            AnthropicOptions::default(),
            sse(&stop_events("refusal", None)),
        )
        .await;
        assert_eq!(
            result.error_message.as_deref(),
            Some("The model refused to complete the request")
        );
    }

    #[tokio::test]
    async fn preserves_sensitive_stop_reasons_with_a_descriptive_error_message() {
        let (result, _) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            hello(),
            AnthropicOptions::default(),
            sse(&stop_events("sensitive", None)),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.raw_stop_reason.as_deref(), Some("sensitive"));
        assert_eq!(
            result.error_message.as_deref(),
            Some("Provider stopped with: sensitive")
        );
    }

    #[tokio::test]
    async fn reports_unhandled_stop_reasons() {
        let (result, _) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            hello(),
            AnthropicOptions::default(),
            sse(&stop_events("brand_new", None)),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Unhandled stop reason: brand_new")
        );
    }

    #[tokio::test]
    async fn treats_message_delta_without_usage_as_a_no_op_for_usage_accumulation() {
        let mut events = minimal_events();
        events[4] = json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" } });
        let (result, _) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            hello(),
            AnthropicOptions::default(),
            sse(&events),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.error_message, None);
        assert_eq!(
            result.content,
            vec![AssistantContent::Text(TextContent::new("Hello"))]
        );
        assert_eq!(result.usage.input, 12);
        assert_eq!(result.usage.total_tokens, 12);
    }

    #[tokio::test]
    async fn ignores_unknown_sse_events_after_message_stop() {
        let mut body = sse(&minimal_events());
        body.push_str(&sse_raw(&[
            ("done", "[DONE]".to_string()),
            ("proxy.stats", "not json".to_string()),
        ]));
        let (result, _) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            hello(),
            AnthropicOptions::default(),
            body,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.error_message, None);
        assert_eq!(
            result.content,
            vec![AssistantContent::Text(TextContent::new("Hello"))]
        );
    }

    #[tokio::test]
    async fn reports_streams_that_end_early_or_carry_errors() {
        let model = builtin("anthropic", "claude-haiku-4-5");
        let (result, _) = run(
            model.clone(),
            hello(),
            AnthropicOptions::default(),
            sse(&minimal_events()[..5]),
        )
        .await;
        assert_eq!(
            result.error_message.as_deref(),
            Some("Anthropic stream ended before message_stop")
        );

        let (result, _) = run(
            model.clone(),
            hello(),
            AnthropicOptions::default(),
            String::new(),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Anthropic stream ended without a stop reason")
        );

        let body = sse_raw(&[(
            "error",
            "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\"}}".to_string(),
        )]);
        let (result, _) = run(model, hello(), AnthropicOptions::default(), body).await;
        assert_eq!(
            result.error_message.as_deref(),
            Some("{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\"}}")
        );
    }

    #[tokio::test]
    async fn reports_aborted_requests() {
        let signal = CancellationToken::new();
        signal.cancel();
        let mut options = AnthropicOptions::default();
        options.signal = Some(signal);
        let (result, _) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            hello(),
            options,
            sse(&minimal_events()),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Aborted);
        // Raised by the shared retry helper before the request is sent.
        assert_eq!(result.error_message.as_deref(), Some("Request aborted"));
    }

    // anthropic-cache-write-1h-cost.test.ts

    fn cache_creation_events(cache_creation: Option<Value>) -> Vec<Value> {
        let mut start_usage = json!({
            "input_tokens": 100,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 1_000_000,
        });
        if let Some(cache_creation) = cache_creation {
            start_usage["cache_creation"] = cache_creation;
        }
        let mut events = minimal_events();
        events[0] = json!({ "type": "message_start", "message": { "id": "msg_test", "usage": start_usage } });
        events[4] = json!({
            "type": "message_delta",
            "delta": { "stop_reason": "end_turn" },
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 1_000_000,
            },
        });
        events
    }

    #[tokio::test]
    async fn prices_the_1h_portion_at_2x_input_and_the_rest_at_the_5m_rate() {
        let (result, _) = run(
            builtin("anthropic", "claude-opus-4-8"),
            hello(),
            AnthropicOptions::default(),
            sse(&cache_creation_events(Some(json!({
                "ephemeral_5m_input_tokens": 600_000,
                "ephemeral_1h_input_tokens": 400_000,
            })))),
        )
        .await;
        assert_eq!(result.usage.cache_write, 1_000_000);
        assert_eq!(result.usage.cache_write_1h, Some(400_000));
        // 600k * 6.25/Mtok + 400k * 10/Mtok = 3.75 + 4.0 = 7.75
        assert!((result.usage.cost.cache_write - 7.75).abs() < 1e-10);
    }

    #[tokio::test]
    async fn prices_1h_cache_writes_reported_only_in_message_delta() {
        let model = model_from(json!({
            "id": "anthropic/claude-haiku-4.5",
            "name": "Claude Haiku 4.5",
            "api": "anthropic-messages",
            "provider": "vercel-ai-gateway",
            "baseUrl": "https://ai-gateway.vercel.sh",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 1, "output": 5, "cacheRead": 0.1, "cacheWrite": 1.25 },
            "contextWindow": 200000,
            "maxTokens": 64000,
            "compat": { "allowEmptySignature": true },
        }));
        let events = vec![
            json!({ "type": "message_start", "message": { "id": "msg_test", "usage": { "input_tokens": 0, "output_tokens": 0 } } }),
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "cache_creation_input_tokens": 6535,
                    "cache_creation": { "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 6535 },
                },
            }),
            json!({ "type": "message_stop" }),
        ];
        let (result, _) = run(model, hello(), AnthropicOptions::default(), sse(&events)).await;
        assert_eq!(result.usage.cache_write, 6535);
        assert_eq!(result.usage.cache_write_1h, Some(6535));
        assert!((result.usage.cost.cache_write - 6535.0 * 2.0 / 1_000_000.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn falls_back_to_the_5m_rate_when_no_breakdown_is_reported() {
        let (result, _) = run(
            builtin("anthropic", "claude-opus-4-8"),
            hello(),
            AnthropicOptions::default(),
            sse(&cache_creation_events(None)),
        )
        .await;
        assert_eq!(result.usage.cache_write, 1_000_000);
        assert_eq!(result.usage.cache_write_1h.unwrap_or(0), 0);
        assert!((result.usage.cost.cache_write - 6.25).abs() < 1e-10);
    }

    #[tokio::test]
    async fn reports_reasoning_tokens_from_output_token_details() {
        let mut events = minimal_events();
        events[4]["usage"]["output_tokens_details"] = json!({ "thinking_tokens": 3 });
        let (result, _) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            hello(),
            AnthropicOptions::default(),
            sse(&events),
        )
        .await;
        assert_eq!(result.usage.reasoning, Some(3));
        assert_eq!(result.usage.total_tokens, 17);
    }

    // anthropic-eager-tool-input-compat.test.ts

    fn eager_model(compat: Value) -> Model {
        let mut merged = json!({ "forceAdaptiveThinking": true });
        if let Value::Object(compat) = compat {
            merged.as_object_mut().unwrap().extend(compat);
        }
        custom_model("claude-opus-4-8", "test-anthropic", merged)
    }

    fn lookup_tool() -> Value {
        json!({
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"],
            },
        })
    }

    fn tool_context(tools: Value) -> TranscriptContext {
        context(json!({
            "messages": [{ "role": "user", "content": "Use the tool", "timestamp": 1 }],
            "tools": tools,
        }))
    }

    fn no_cache() -> AnthropicOptions {
        let mut options = AnthropicOptions::default();
        options.cache_retention = Some(CacheRetention::None);
        options
    }

    #[tokio::test]
    async fn sends_per_tool_eager_input_streaming_by_default() {
        let (_, request) = run(
            eager_model(Value::Null),
            tool_context(json!([lookup_tool()])),
            no_cache(),
            String::new(),
        )
        .await;
        let request = request.unwrap();
        assert_eq!(
            request.body["tools"][0]["eager_input_streaming"],
            json!(true)
        );
        assert_eq!(request.header("anthropic-beta"), None);
    }

    #[tokio::test]
    async fn uses_the_legacy_fine_grained_tool_streaming_beta_when_eager_streaming_is_disabled() {
        let model = eager_model(json!({ "supportsEagerToolInputStreaming": false }));
        let (_, request) = run(
            model.clone(),
            tool_context(json!([lookup_tool()])),
            no_cache(),
            String::new(),
        )
        .await;
        let request = request.unwrap();
        assert!(
            request.body["tools"][0]
                .get("eager_input_streaming")
                .is_none()
        );
        assert_eq!(
            request.header("anthropic-beta"),
            Some(FINE_GRAINED_TOOL_STREAMING_BETA)
        );

        let (_, request) = run(model, tool_context(json!([])), no_cache(), String::new()).await;
        let request = request.unwrap();
        assert!(request.body.get("tools").is_none());
        assert_eq!(request.header("anthropic-beta"), None);
    }

    // anthropic-mid-conversation-effort.test.ts

    fn managed_model(provider: &str) -> Model {
        let mut model = custom_model(
            "claude-fable-5-1",
            provider,
            json!({ "forceAdaptiveThinking": true, "supportsMidConvoEffort": true }),
        );
        model.thinking_level_map = Some(
            serde_json::from_value(json!({
                "off": null, "minimal": "low", "low": "low", "medium": "medium", "high": "high", "max": "max",
            }))
            .unwrap(),
        );
        model
    }

    fn managed_assistant(model: &Model, level: Option<&str>) -> Value {
        assistant_json(
            &model.provider,
            &model.id,
            json!([
                { "type": "thinking", "thinking": "reasoning", "thinkingSignature": "signature" },
                { "type": "text", "text": "answer" },
            ]),
            level.map_or_else(
                || json!({}),
                |level| json!({ "providerThinkingLevel": level }),
            ),
        )
    }

    fn user(text: &str, timestamp: u64) -> Value {
        json!({ "role": "user", "content": text, "timestamp": timestamp })
    }

    async fn capture_effort(
        model: &Model,
        messages: Value,
        effort: Option<AnthropicEffort>,
    ) -> (Value, AssistantMessage) {
        let mut options = no_cache();
        options.thinking_enabled = Some(true);
        options.effort = effort;
        capture(
            model.clone(),
            context(json!({ "messages": messages })),
            options,
        )
        .await
    }

    fn effort_messages(payload: &Value) -> Vec<Value> {
        payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message["role"] == "system")
            .cloned()
            .collect()
    }

    #[tokio::test]
    async fn reconstructs_an_exact_historical_marker_prefix_and_appends_the_current_marker() {
        let model = managed_model("anthropic");
        let (first, first_message) =
            capture_effort(&model, json!([user("one", 1)]), Some(AnthropicEffort::Low)).await;
        let (second, _) = capture_effort(
            &model,
            json!([
                user("one", 1),
                managed_assistant(&model, Some("low")),
                user("two", 2)
            ]),
            Some(AnthropicEffort::High),
        )
        .await;

        assert_eq!(
            first["messages"],
            json!([
                { "role": "user", "content": "one" },
                { "role": "system", "content": [], "output_config": { "effort": "low" } },
            ])
        );
        let first_messages = first["messages"].as_array().unwrap();
        let second_messages = second["messages"].as_array().unwrap();
        assert_eq!(
            &second_messages[..first_messages.len()],
            &first_messages[..]
        );
        assert_eq!(
            second_messages.last().unwrap(),
            &json!({ "role": "system", "content": [], "output_config": { "effort": "high" } })
        );
        assert_eq!(first["output_config"], json!({ "effort": "high" }));
        assert_eq!(second["output_config"], json!({ "effort": "high" }));
        assert_eq!(
            second["thinking"],
            json!({
                "type": "adaptive",
                "display": "summarized",
                "block_binding": { "prefix_mismatch_behavior": "drop_block" },
            })
        );
        assert_eq!(
            first_message.provider_thinking_level.as_deref(),
            Some("low")
        );
    }

    #[tokio::test]
    async fn preserves_native_efforts() {
        let model = managed_model("anthropic");
        for effort in [
            AnthropicEffort::Low,
            AnthropicEffort::Medium,
            AnthropicEffort::High,
            AnthropicEffort::Xhigh,
            AnthropicEffort::Max,
        ] {
            let (payload, message) =
                capture_effort(&model, json!([user("one", 1)]), Some(effort)).await;
            assert_eq!(
                effort_messages(&payload),
                vec![
                    json!({ "role": "system", "content": [], "output_config": { "effort": effort.as_str() } })
                ]
            );
            assert_eq!(
                message.provider_thinking_level.as_deref(),
                Some(effort.as_str())
            );
        }
    }

    #[tokio::test]
    async fn defaults_omitted_effort_to_high_and_still_enables_drop_block() {
        let (payload, message) =
            capture_effort(&managed_model("anthropic"), json!([user("one", 1)]), None).await;
        assert_eq!(
            payload["messages"].as_array().unwrap().last().unwrap(),
            &json!({ "role": "system", "content": [], "output_config": { "effort": "high" } })
        );
        assert_eq!(
            payload["thinking"]["block_binding"]["prefix_mismatch_behavior"],
            "drop_block"
        );
        assert_eq!(message.provider_thinking_level.as_deref(), Some("high"));
    }

    #[tokio::test]
    async fn does_not_invent_markers_for_legacy_or_other_provider_assistants() {
        let model = managed_model("anthropic");
        let mut other_provider = managed_assistant(&model, Some("low"));
        other_provider["provider"] = json!("other-provider");
        let (payload, _) = capture_effort(
            &model,
            json!([
                user("one", 1),
                managed_assistant(&model, None),
                user("two", 2),
                other_provider,
                user("three", 3),
            ]),
            Some(AnthropicEffort::Medium),
        )
        .await;
        assert_eq!(
            effort_messages(&payload),
            vec![
                json!({ "role": "system", "content": [], "output_config": { "effort": "medium" } })
            ]
        );
    }

    #[tokio::test]
    async fn leaves_unsupported_models_on_top_level_effort() {
        let mut model = managed_model("anthropic");
        model.compat =
            Some(serde_json::from_value(json!({ "forceAdaptiveThinking": true })).unwrap());
        let (payload, message) =
            capture_effort(&model, json!([user("one", 1)]), Some(AnthropicEffort::Low)).await;
        assert_eq!(
            payload["messages"],
            json!([{ "role": "user", "content": "one" }])
        );
        assert_eq!(payload["output_config"], json!({ "effort": "low" }));
        assert_eq!(
            payload["thinking"],
            json!({ "type": "adaptive", "display": "summarized" })
        );
        assert_eq!(message.provider_thinking_level, None);
    }

    #[tokio::test]
    async fn generates_exact_model_and_transport_gates() {
        let direct = builtin("anthropic", "claude-fable-5-1");
        assert_eq!(
            direct.compat.as_ref().unwrap().supports_mid_convo_effort,
            Some(true)
        );
        assert_eq!(
            direct
                .thinking_level_map
                .as_ref()
                .unwrap()
                .get(&ModelThinkingLevel::Off),
            Some(&None)
        );
        let unsupported = builtin("anthropic", "claude-opus-4-8");
        assert_eq!(
            unsupported
                .compat
                .as_ref()
                .unwrap()
                .supports_mid_convo_effort,
            None
        );
        assert!(
            builtin("anthropic", "claude-opus-5")
                .compat
                .unwrap()
                .allowed_fallback_models
                .is_none()
        );
    }

    // anthropic-strict-tool-schema.test.ts

    async fn capture_first_tool(tool: Value) -> Value {
        let model = custom_model(
            "claude-opus-4-8",
            "test-anthropic",
            json!({ "forceAdaptiveThinking": true, "supportsStrictTools": true }),
        );
        let (payload, _) = capture(model, tool_context(json!([tool])), no_cache()).await;
        payload["tools"][0].clone()
    }

    fn strict_tool(parameters: Value) -> Value {
        json!({
            "name": "lookup",
            "description": "Look up a value",
            "parameters": parameters,
            "constrainedSampling": { "type": "json_schema", "strict": "prefer" },
        })
    }

    #[tokio::test]
    async fn only_sends_the_full_input_schema_for_strict_json_schema_tools() {
        let legacy_parameters = json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false,
            "title": "LookupInput",
        });
        let legacy = capture_first_tool(json!({
            "name": "lookup",
            "description": "Look up a value",
            "parameters": legacy_parameters,
        }))
        .await;
        assert!(legacy.get("strict").is_none());
        assert_eq!(
            legacy["input_schema"],
            json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"],
            })
        );

        let strict = capture_first_tool(strict_tool(json!({
            "type": "object",
            "properties": { "value": { "type": "string" }, "optional": { "type": "number" } },
            "required": ["value"],
            "title": "StrictLookupInput",
        })))
        .await;
        assert_eq!(strict["strict"], json!(true));
        let schema = &strict["input_schema"];
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(schema["required"], json!(["value", "optional"]));
        assert_eq!(
            schema["properties"]["optional"],
            json!({ "anyOf": [{ "type": "number" }, { "type": "null" }] })
        );
        assert_eq!(schema["title"], json!("StrictLookupInput"));
    }

    #[tokio::test]
    async fn sends_prefer_tools_non_strict_when_they_use_keywords_anthropic_strict_mode_rejects() {
        let unsupported = [
            json!({
                "type": "object",
                "properties": { "timeoutMs": { "type": "integer", "minimum": 1, "maximum": 300000 } },
            }),
            json!({
                "type": "object",
                "properties": {
                    "options": {
                        "type": "object",
                        "properties": { "tags": { "type": "array", "items": { "type": "string" }, "minItems": 2 } },
                        "required": ["tags"],
                    },
                },
                "required": ["options"],
            }),
            json!({
                "type": "object",
                "properties": { "expression": { "type": "string", "format": "regex" } },
                "required": ["expression"],
            }),
        ];
        for parameters in unsupported {
            let tool = capture_first_tool(strict_tool(parameters)).await;
            assert!(tool.get("strict").is_none());
        }

        let supported = capture_first_tool(strict_tool(json!({
            "type": "object",
            "properties": {
                "code": { "type": "string", "minLength": 1, "maxLength": 1000, "pattern": "^[a-z]+$" },
                "url": { "type": "string", "format": "uri" },
                "tags": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
            },
            "required": ["code", "url", "tags"],
        })))
        .await;
        assert_eq!(supported["strict"], json!(true));
    }

    // anthropic-temperature-compat.test.ts

    fn temperature(value: f64) -> SimpleStreamOptions {
        let mut options = SimpleStreamOptions::default();
        options.stream.temperature = Some(value);
        options
    }

    #[tokio::test]
    async fn applies_temperature_compatibility() {
        for (model, value, expected) in [
            (builtin("anthropic", "claude-opus-4-7"), 0.0, None),
            (builtin("anthropic", "claude-opus-4-8"), 0.0, None),
            (builtin("anthropic", "claude-opus-4-7"), 1.0, None),
            (
                builtin("anthropic", "claude-opus-4-6"),
                0.0,
                Some(json!(0.0)),
            ),
            (
                builtin("anthropic", "claude-sonnet-4-6"),
                0.0,
                Some(json!(0.0)),
            ),
            (
                custom_model(
                    "vendor--claude-opus-4-7",
                    "vendor-proxy",
                    json!({ "supportsTemperature": false }),
                ),
                0.0,
                None,
            ),
        ] {
            let payload = capture_simple(model, hello(), temperature(value)).await;
            assert_eq!(payload.get("temperature").cloned(), expected);
        }
    }

    // anthropic-thinking-disable.test.ts

    fn reasoning(level: ThinkingLevel) -> SimpleStreamOptions {
        SimpleStreamOptions {
            reasoning: Some(level),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn sends_thinking_disabled_when_thinking_is_off() {
        for id in ["claude-sonnet-4-5", "claude-opus-4-6", "claude-opus-4-8"] {
            let payload = capture_simple(
                builtin("anthropic", id),
                hello(),
                SimpleStreamOptions::default(),
            )
            .await;
            assert_eq!(payload["thinking"], json!({ "type": "disabled" }), "{id}");
            assert!(payload.get("output_config").is_none());
        }

        let payload = capture_simple(
            builtin("anthropic", "claude-fable-5"),
            hello(),
            SimpleStreamOptions::default(),
        )
        .await;
        assert!(payload.get("thinking").is_none());
        assert!(payload.get("output_config").is_none());
    }

    #[tokio::test]
    async fn uses_adaptive_thinking_when_reasoning_is_enabled() {
        for (id, level, effort) in [
            ("claude-opus-4-8", ThinkingLevel::High, "high"),
            ("claude-sonnet-5", ThinkingLevel::High, "high"),
            ("claude-opus-4-8", ThinkingLevel::Xhigh, "xhigh"),
            ("claude-fable-5", ThinkingLevel::Xhigh, "xhigh"),
        ] {
            let payload = capture_simple(builtin("anthropic", id), hello(), reasoning(level)).await;
            assert_eq!(
                payload["thinking"],
                json!({ "type": "adaptive", "display": "summarized" })
            );
            assert_eq!(payload["output_config"], json!({ "effort": effort }));
        }
    }

    // anthropic-force-adaptive-thinking.test.ts

    fn vendor_model(compat: Value) -> Model {
        custom_model("vendor--claude-opus-latest", "vendor-proxy", compat)
    }

    #[tokio::test]
    async fn sends_legacy_thinking_payload_for_custom_model_ids_by_default() {
        let payload = capture_simple(
            vendor_model(Value::Null),
            hello(),
            reasoning(ThinkingLevel::Medium),
        )
        .await;
        assert_eq!(payload["thinking"]["type"], "enabled");
        assert!(payload.get("output_config").is_none());
        let max_tokens = payload["max_tokens"].as_u64().unwrap();
        let budget = payload["thinking"]["budget_tokens"].as_u64().unwrap();
        assert!(budget <= max_tokens - 1024);
        let betas = payload["betas"].as_array().unwrap();
        assert!(betas.contains(&json!(INTERLEAVED_THINKING_BETA)));
    }

    #[tokio::test]
    async fn sends_adaptive_thinking_payload_when_force_adaptive_thinking_is_true() {
        let payload = capture_simple(
            vendor_model(json!({ "forceAdaptiveThinking": true })),
            hello(),
            reasoning(ThinkingLevel::Medium),
        )
        .await;
        assert_eq!(
            payload["thinking"],
            json!({ "type": "adaptive", "display": "summarized" })
        );
        assert_eq!(payload["output_config"], json!({ "effort": "medium" }));
    }

    #[tokio::test]
    async fn uses_adaptive_thinking_effort_from_the_thinking_level_map() {
        let mut model = vendor_model(json!({ "forceAdaptiveThinking": true }));
        model.thinking_level_map =
            Some(serde_json::from_value(json!({ "max": "max", "high": "xhigh" })).unwrap());
        let payload = capture_simple(model.clone(), hello(), reasoning(ThinkingLevel::Max)).await;
        assert_eq!(payload["output_config"], json!({ "effort": "max" }));
        let payload = capture_simple(model, hello(), reasoning(ThinkingLevel::High)).await;
        assert_eq!(payload["output_config"], json!({ "effort": "xhigh" }));
    }

    #[tokio::test]
    async fn allows_built_in_adaptive_models_to_opt_out() {
        let mut model = builtin("anthropic", "claude-opus-4-8");
        model.compat =
            Some(serde_json::from_value(json!({ "forceAdaptiveThinking": false })).unwrap());
        let payload = capture_simple(model, hello(), reasoning(ThinkingLevel::Medium)).await;
        assert_eq!(payload["thinking"]["type"], "enabled");
        assert!(payload.get("output_config").is_none());
    }

    #[tokio::test]
    async fn preserves_thinking_disabled_when_reasoning_is_off_regardless_of_override() {
        let payload = capture_simple(
            vendor_model(json!({ "forceAdaptiveThinking": true })),
            hello(),
            SimpleStreamOptions::default(),
        )
        .await;
        assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
        assert!(payload.get("output_config").is_none());
    }

    // anthropic-empty-thinking-signature-compat.test.ts

    fn signature_model(provider: &str, id: &str, allow_empty_signature: Option<bool>) -> Model {
        let mut model = custom_model(
            id,
            provider,
            allow_empty_signature
                .map_or(Value::Null, |allow| json!({ "allowEmptySignature": allow })),
        );
        model.base_url = "http://127.0.0.1:9/anthropic".to_string();
        model.max_tokens = 1024;
        model
    }

    fn signature_context(
        signature: &str,
        thinking: &str,
        provider: &str,
        model: &str,
        extra_content: Option<Value>,
    ) -> TranscriptContext {
        let mut content = vec![
            json!({ "type": "thinking", "thinking": thinking, "thinkingSignature": signature }),
        ];
        content.extend(extra_content);
        context(json!({
            "messages": [
                user("first", 1),
                assistant_json(provider, model, Value::Array(content), json!({})),
                user("second", 2),
            ],
        }))
    }

    fn assistant_content(payload: &Value) -> Value {
        payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "assistant")
            .map(|message| message["content"].clone())
            .unwrap_or(Value::Null)
    }

    #[tokio::test]
    async fn handles_empty_thinking_signatures() {
        let provider = "xiaomi-token-plan-ams";
        let id = "mimo-v2.5-pro";
        let payload = capture_simple(
            signature_model(provider, id, None),
            signature_context("", "internal reasoning", provider, id, None),
            SimpleStreamOptions::default(),
        )
        .await;
        assert_eq!(
            assistant_content(&payload),
            json!([{ "type": "text", "text": "internal reasoning" }])
        );

        let payload = capture_simple(
            signature_model(provider, id, None),
            signature_context("signed-thinking", "", provider, id, None),
            SimpleStreamOptions::default(),
        )
        .await;
        assert_eq!(
            assistant_content(&payload),
            json!([{ "type": "thinking", "thinking": "", "signature": "signed-thinking" }])
        );

        let payload = capture_simple(
            signature_model(provider, id, Some(true)),
            signature_context(" ", "internal reasoning", provider, id, None),
            SimpleStreamOptions::default(),
        )
        .await;
        assert_eq!(
            assistant_content(&payload),
            json!([{ "type": "thinking", "thinking": "internal reasoning", "signature": "" }])
        );
    }

    #[tokio::test]
    async fn preserves_unsigned_thinking_only_for_the_same_model() {
        let id = "accounts/fireworks/models/deepseek-v4p1-flash";
        let payload = capture_simple(
            signature_model("fireworks", id, Some(true)),
            signature_context(
                "",
                "internal reasoning",
                "fireworks",
                id,
                Some(json!({ "type": "text", "text": "answer" })),
            ),
            SimpleStreamOptions::default(),
        )
        .await;
        assert_eq!(
            assistant_content(&payload),
            json!([
                { "type": "thinking", "thinking": "internal reasoning", "signature": "" },
                { "type": "text", "text": "answer" },
            ])
        );

        let payload = capture_simple(
            signature_model("fireworks", id, Some(true)),
            signature_context(
                "",
                "internal reasoning",
                "fireworks",
                "accounts/fireworks/models/nemotron-3-ultra-nvfp4",
                None,
            ),
            SimpleStreamOptions::default(),
        )
        .await;
        assert_eq!(
            assistant_content(&payload),
            json!([{ "type": "text", "text": "internal reasoning" }])
        );
    }

    // anthropic-auth-token.test.ts

    fn test_model() -> Model {
        model_from(json!({
            "id": "claude-test",
            "name": "Claude Test",
            "api": "anthropic-messages",
            "provider": "anthropic",
            "baseUrl": "https://api.anthropic.com",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 100000,
            "maxTokens": 4096,
        }))
    }

    fn system_prompt_context() -> TranscriptContext {
        context(json!({
            "systemPrompt": "System prompt.",
            "messages": [{ "role": "user", "content": "Hello", "timestamp": 1 }],
        }))
    }

    fn auth_events() -> String {
        sse(&[
            json!({ "type": "message_start", "message": { "id": "msg_test", "usage": { "input_tokens": 1, "output_tokens": 0 } } }),
            json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 1 } }),
            json!({ "type": "message_stop" }),
        ])
    }

    fn with_headers(headers: &[(&str, Option<&str>)], key: Option<&str>) -> AnthropicOptions {
        let mut options = AnthropicOptions::default();
        options.headers = Some(
            headers
                .iter()
                .map(|(name, value)| (name.to_string(), value.map(str::to_string)))
                .collect(),
        );
        options.api_key = key.map(str::to_string);
        options
    }

    fn system_texts(body: &Value) -> Vec<String> {
        body["system"]
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .map(|block| block["text"].as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn uses_authorization_headers_without_oauth_mode_request_shaping() {
        let (result, request) = run(
            test_model(),
            system_prompt_context(),
            with_headers(&[("Authorization", Some("Bearer gateway-token"))], None),
            auth_events(),
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        let request = request.unwrap();
        assert_eq!(
            request.header("authorization"),
            Some("Bearer gateway-token")
        );
        assert_eq!(request.header("x-api-key"), None);
        assert!(
            !request
                .header("anthropic-beta")
                .unwrap_or_default()
                .contains("oauth-2025-04-20")
        );
        assert_eq!(system_texts(&request.body), ["System prompt."]);
    }

    struct EnvContext(Vec<(&'static str, &'static str)>);

    #[async_trait]
    impl AuthContext for EnvContext {
        async fn env(&self, name: &str) -> Option<String> {
            self.0
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }

        async fn file_exists(&self, _path: &str) -> bool {
            false
        }
    }

    async fn run_through_models(
        env: Vec<(&'static str, &'static str)>,
        options: SimpleStreamOptions,
    ) -> CapturedRequest {
        let (base_url, requests) = serve_sse(auth_events()).await;
        let models = create_models(CreateModelsOptions {
            auth_context: Some(Arc::new(EnvContext(env))),
            ..Default::default()
        });
        models.set_provider(anthropic_provider());
        let mut model = test_model();
        model.base_url = base_url;
        let context: Context = serde_json::from_value(json!({
            "systemPrompt": "System prompt.",
            "messages": [{ "role": "user", "content": "Hello", "timestamp": 1 }],
        }))
        .unwrap();
        let result = models
            .stream_simple(&model, &context, options)
            .result()
            .await;
        assert_eq!(
            result.stop_reason,
            StopReason::Stop,
            "{:?}",
            result.error_message
        );
        requests.lock().unwrap().first().cloned().unwrap()
    }

    #[tokio::test]
    async fn threads_auth_context_tokens_through_request_headers() {
        let request = run_through_models(
            vec![("ANTHROPIC_AUTH_TOKEN", "ctx-token")],
            SimpleStreamOptions::default(),
        )
        .await;
        assert_eq!(request.header("authorization"), Some("Bearer ctx-token"));
        assert_eq!(request.header("x-api-key"), None);
        assert!(
            !request
                .header("anthropic-beta")
                .unwrap_or_default()
                .contains("oauth-2025-04-20")
        );
        assert_eq!(system_texts(&request.body), ["System prompt."]);

        let request = run_through_models(
            vec![("ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat-test")],
            SimpleStreamOptions::default(),
        )
        .await;
        assert_eq!(
            request.header("authorization"),
            Some("Bearer sk-ant-oat-test")
        );
        assert_eq!(request.header("x-api-key"), None);
        assert!(
            request
                .header("anthropic-beta")
                .unwrap()
                .contains("oauth-2025-04-20")
        );
        assert_eq!(
            system_texts(&request.body),
            [
                "You are Claude Code, Anthropic's official CLI for Claude.",
                "System prompt."
            ]
        );
        assert_eq!(
            request.header("user-agent"),
            Some(format!("claude-cli/{CLAUDE_CODE_VERSION}").as_str())
        );
        assert_eq!(request.header("x-app"), Some("cli"));

        let mut options = SimpleStreamOptions::default();
        options.stream.headers = Some(
            [("Authorization", "Bearer explicit-token".to_string())]
                .into_iter()
                .collect(),
        );
        let request =
            run_through_models(vec![("ANTHROPIC_AUTH_TOKEN", "ctx-token")], options).await;
        assert_eq!(
            request.header("authorization"),
            Some("Bearer explicit-token")
        );
    }

    #[tokio::test]
    async fn uses_pi_user_agent_and_explicit_header_overrides() {
        let (_, request) = run(
            test_model(),
            system_prompt_context(),
            api_key("anthropic-key"),
            auth_events(),
        )
        .await;
        let request = request.unwrap();
        assert_eq!(
            request.header("user-agent"),
            Some(get_pi_user_agent().as_str())
        );
        assert_eq!(request.header("x-api-key"), Some("anthropic-key"));
        assert_eq!(request.header("anthropic-version"), Some("2023-06-01"));

        let mut kimi = test_model();
        kimi.id = "kimi-for-coding".to_string();
        kimi.provider = "kimi-coding".to_string();
        let (_, request) = run(
            kimi,
            system_prompt_context(),
            with_headers(&[("User-Agent", Some("custom-client"))], Some("kimi-key")),
            auth_events(),
        )
        .await;
        assert_eq!(request.unwrap().header("user-agent"), Some("custom-client"));
    }

    #[tokio::test]
    async fn preserves_explicit_anthropic_beta_header_replacement_and_suppression() {
        let (_, request) = run(
            test_model(),
            system_prompt_context(),
            with_headers(
                &[("anthropic-beta", Some("custom-beta"))],
                Some("anthropic-key"),
            ),
            auth_events(),
        )
        .await;
        assert_eq!(
            request.unwrap().header("anthropic-beta"),
            Some("custom-beta")
        );

        let mut options = with_headers(&[("anthropic-beta", None)], Some("anthropic-key"));
        options.thinking_enabled = Some(true);
        let (_, request) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            system_prompt_context(),
            options,
            auth_events(),
        )
        .await;
        assert_eq!(request.unwrap().header("anthropic-beta"), None);
    }

    // anthropic-tool-name-normalization.test.ts (offline)

    #[tokio::test]
    async fn maps_tool_names_to_claude_code_casing_for_oauth_tokens() {
        let context = context(json!({
            "messages": [{ "role": "user", "content": "Use the tool", "timestamp": 1 }],
            "tools": [
                { "name": "todowrite", "description": "Todo", "parameters": { "type": "object", "properties": {} } },
                { "name": "find", "description": "Find", "parameters": { "type": "object", "properties": {} } },
            ],
        }));
        let events = vec![
            minimal_events()[0].clone(),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "tool_use", "id": "toolu_1", "name": "TodoWrite", "input": {} },
            }),
            json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": "{\"a\":1}" } }),
            json!({ "type": "content_block_stop", "index": 0 }),
            json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 3 } }),
            json!({ "type": "message_stop" }),
        ];
        let (result, request) = run(
            builtin("anthropic", "claude-sonnet-4-6"),
            context,
            api_key("sk-ant-oat-test"),
            sse(&events),
        )
        .await;
        let request = request.unwrap();
        let names: Vec<&str> = request.body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["TodoWrite", "find"]);
        let AssistantContent::ToolCall(tool_call) = &result.content[0] else {
            panic!("expected a tool call");
        };
        assert_eq!(tool_call.name, "todowrite");
        assert_eq!(tool_call.arguments, json!({ "a": 1 }));
    }

    // cache-retention.test.ts (Anthropic)

    async fn capture_cache(model: Model, options: AnthropicOptions) -> Value {
        capture(model, system_hello(), options).await.0
    }

    #[tokio::test]
    async fn applies_cache_retention() {
        let haiku = builtin("anthropic", "claude-haiku-4-5");
        let mut proxy = haiku.clone();
        proxy.base_url = "https://my-proxy.example.com/v1".to_string();

        let mut options = AnthropicOptions::default();
        options.env = Some([("PI_CACHE_RETENTION".to_string(), "long".to_string())].into());
        let payload = capture_cache(proxy.clone(), options).await;
        assert_eq!(
            payload["system"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );

        let mut no_long = proxy;
        no_long.compat =
            Some(serde_json::from_value(json!({ "supportsLongCacheRetention": false })).unwrap());
        let mut options = AnthropicOptions::default();
        options.cache_retention = Some(CacheRetention::Long);
        let payload = capture_cache(no_long, options).await;
        assert_eq!(
            payload["system"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );

        let payload = capture_cache(haiku.clone(), no_cache()).await;
        assert!(payload["system"][0].get("cache_control").is_none());

        let payload = capture_cache(haiku.clone(), AnthropicOptions::default()).await;
        assert_eq!(
            payload["system"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        let last_message = payload["messages"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .clone();
        assert_eq!(
            last_message["content"],
            json!([{ "type": "text", "text": "Hello", "cache_control": { "type": "ephemeral" } }])
        );

        let mut options = AnthropicOptions::default();
        options.cache_retention = Some(CacheRetention::Long);
        let payload = capture_cache(haiku, options).await;
        assert_eq!(
            payload["system"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
    }

    #[tokio::test]
    async fn sends_session_affinity_headers_only_when_caching() {
        let model = model_from(json!({
            "id": "anthropic/claude-haiku-4.5",
            "name": "Claude Haiku 4.5",
            "api": "anthropic-messages",
            "provider": "openrouter",
            "baseUrl": "https://openrouter.ai/api",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 1, "output": 5, "cacheRead": 0.1, "cacheWrite": 1.25 },
            "contextWindow": 200000,
            "maxTokens": 64000,
        }));
        let mut options = AnthropicOptions::default();
        options.session_id = Some("session-1".to_string());
        let (_, request) = run(
            model.clone(),
            hello(),
            options.clone(),
            sse(&minimal_events()),
        )
        .await;
        assert_eq!(request.unwrap().header("x-session-id"), Some("session-1"));

        options.cache_retention = Some(CacheRetention::None);
        let (_, request) = run(model, hello(), options.clone(), sse(&minimal_events())).await;
        assert_eq!(request.unwrap().header("x-session-id"), None);

        options.cache_retention = None;
        let (_, request) = run(
            builtin("anthropic", "claude-haiku-4-5"),
            hello(),
            options,
            sse(&minimal_events()),
        )
        .await;
        let request = request.unwrap();
        assert_eq!(request.header("x-session-id"), None);
        assert_eq!(request.header("x-session-affinity"), None);
    }

    // transcript-tool-changes.test.ts (Anthropic)

    fn empty_tool(name: &str) -> Value {
        json!({ "name": name, "description": format!("{name} tool"), "parameters": { "type": "object", "properties": {} } })
    }

    fn tool_change_context() -> Value {
        json!({
            "messages": [
                {
                    "role": "system",
                    "content": "base prompt",
                    "sections": { "rules": "<rules>\nold rules\n</rules>", "docs": "<docs>\nread docs\n</docs>" },
                    "toolsAdded": [empty_tool("base_tool")],
                    "timestamp": 0,
                },
                { "role": "user", "content": "before", "timestamp": 1 },
                {
                    "role": "system",
                    "content": "updated guidance",
                    "sections": { "rules": "<rules>\nnew rules\n</rules>", "docs": null },
                    "toolsRemoved": [{ "name": "base_tool" }],
                    "toolsAdded": [empty_tool("late_tool")],
                    "timestamp": 2,
                },
            ],
        })
    }

    fn tool_change_model(compat: Value) -> Model {
        let mut model = custom_model("claude-opus-5", "anthropic", compat);
        model.max_tokens = 1000;
        model.context_window = 100000;
        model
    }

    fn native_model() -> Model {
        tool_change_model(json!({
            "supportsMidConvoSystemMessages": true,
            "supportsMidConvoToolChanges": true,
        }))
    }

    async fn capture_transcript(model: Model, messages: Value) -> Value {
        capture_simple(model, context(messages), SimpleStreamOptions::default()).await
    }

    fn betas(payload: &Value) -> Vec<String> {
        payload["betas"]
            .as_array()
            .map(|betas| {
                betas
                    .iter()
                    .map(|beta| beta.as_str().unwrap().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn sends_anthropic_updates_and_tool_changes_in_native_system_messages() {
        let payload = capture_transcript(native_model(), tool_change_context()).await;
        assert!(betas(&payload).contains(&INLINE_TOOLS_BETA.to_string()));
        assert_eq!(
            system_texts(&payload),
            ["base prompt\n\n<rules>\nold rules\n</rules>\n\n<docs>\nread docs\n</docs>"]
        );
        let tools = payload["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"], "base_tool");
        assert_eq!(tools[0]["cache_control"], json!({ "type": "ephemeral" }));
        assert!(tools[0].get("defer_loading").is_none());
        assert_eq!(tools[1]["name"], "__pi_deferred_placeholder__");
        assert_eq!(tools[1]["defer_loading"], json!(true));
        assert!(tools[1].get("cache_control").is_none());

        let update = payload["messages"].as_array().unwrap().last().unwrap();
        assert_eq!(update["role"], "system");
        let content = update["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(
            content[1],
            json!({ "type": "tool_removal", "tool": { "type": "tool_reference", "name": "base_tool" } })
        );
        assert_eq!(content[2]["type"], "tool_addition");
        assert_eq!(content[2]["tool"]["type"], "tool_definition");
        assert_eq!(content[2]["tool"]["definition"]["name"], "late_tool");
        assert_eq!(
            content[2]["tool"]["definition"]["description"],
            "late_tool tool"
        );
        assert_eq!(content[2]["cache_control"], json!({ "type": "ephemeral" }));
        assert!(
            content[2]["tool"]["definition"]
                .get("cache_control")
                .is_none()
        );
        assert!(
            content[2]["tool"]["definition"]
                .get("defer_loading")
                .is_none()
        );
        let text = content[0]["text"].as_str().unwrap();
        assert!(text.contains("updated guidance"));
        assert!(text.contains("<rules>\nnew rules\n</rules>"));
        assert!(text.contains("Removed system prompt section \"docs\""));

        let mut initial = tool_change_context();
        initial["messages"].as_array_mut().unwrap().truncate(2);
        let payload = capture_transcript(native_model(), initial).await;
        let names: Vec<&str> = payload["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["base_tool", "__pi_deferred_placeholder__"]);
    }

    #[tokio::test]
    async fn redefines_an_anthropic_tool_by_value_under_the_same_name() {
        let mut redefined = empty_tool("base_tool");
        redefined["description"] = json!("changed");
        let payload = capture_transcript(
            native_model(),
            json!({
                "messages": [
                    { "role": "system", "content": "base prompt", "toolsAdded": [empty_tool("base_tool")], "timestamp": 0 },
                    { "role": "user", "content": "before", "timestamp": 1 },
                    {
                        "role": "system",
                        "content": "",
                        "toolsRemoved": [{ "name": "base_tool" }],
                        "toolsAdded": [redefined],
                        "timestamp": 2,
                    },
                ],
            }),
        )
        .await;
        assert!(betas(&payload).contains(&INLINE_TOOLS_BETA.to_string()));
        assert_eq!(payload["tools"][0]["description"], "base_tool tool");
        assert_eq!(payload["tools"][1]["name"], "__pi_deferred_placeholder__");
        let update = payload["messages"].as_array().unwrap().last().unwrap();
        let content = update["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_addition");
        assert_eq!(content[0]["tool"]["definition"]["name"], "base_tool");
        assert_eq!(content[0]["tool"]["definition"]["description"], "changed");
    }

    #[tokio::test]
    async fn sends_the_current_anthropic_tool_list_without_an_initial_tool() {
        let payload = capture_transcript(
            native_model(),
            json!({
                "messages": [
                    { "role": "system", "content": "base prompt", "timestamp": 0 },
                    { "role": "user", "content": "before", "timestamp": 1 },
                    { "role": "system", "content": "updated guidance", "toolsAdded": [empty_tool("late_tool")], "timestamp": 2 },
                ],
            }),
        )
        .await;
        assert!(!betas(&payload).contains(&INLINE_TOOLS_BETA.to_string()));
        let tools = payload["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "late_tool");
        assert_eq!(tools[0]["cache_control"], json!({ "type": "ephemeral" }));
        assert!(tools[0].get("defer_loading").is_none());
        let update = payload["messages"].as_array().unwrap().last().unwrap();
        let types: Vec<&str> = update["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|block| block["type"].as_str().unwrap())
            .collect();
        assert_eq!(types, ["text"]);
    }

    #[tokio::test]
    async fn folds_anthropic_updates_without_native_support() {
        let mut sonnet = tool_change_model(Value::Null);
        sonnet.id = "claude-sonnet-4-5".to_string();
        let payload = capture_transcript(sonnet, tool_change_context()).await;
        assert!(!betas(&payload).contains(&INLINE_TOOLS_BETA.to_string()));
        assert_eq!(
            system_texts(&payload),
            ["base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>"]
        );
        assert_eq!(payload["tools"][0]["name"], "late_tool");
        assert_eq!(payload["tools"].as_array().unwrap().len(), 1);
        let roles: Vec<&str> = payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user"]);

        let partial = tool_change_model(json!({ "supportsMidConvoToolChanges": true }));
        let payload = capture_transcript(partial, tool_change_context()).await;
        assert!(!betas(&payload).contains(&INLINE_TOOLS_BETA.to_string()));
        assert_eq!(payload["tools"].as_array().unwrap().len(), 1);
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn holds_system_updates_until_the_next_assistant_message() {
        let model = native_model();
        let tool_call = assistant_json(
            "anthropic",
            "claude-opus-5",
            json!([{ "type": "toolCall", "id": "call_1", "name": "base_tool", "arguments": {} }]),
            json!({ "stopReason": "toolUse" }),
        );
        let payload = capture_transcript(
            model,
            json!({
                "messages": [
                    { "role": "system", "content": "base prompt", "toolsAdded": [empty_tool("base_tool")], "timestamp": 0 },
                    user("before", 1),
                    tool_call,
                    { "role": "system", "content": "between", "timestamp": 2 },
                    {
                        "role": "toolResult",
                        "toolCallId": "call_1",
                        "toolName": "base_tool",
                        "content": [{ "type": "text", "text": "done" }],
                        "isError": false,
                        "timestamp": 3,
                    },
                    assistant_json("anthropic", "claude-opus-5", json!([{ "type": "text", "text": "ok" }]), json!({})),
                ],
            }),
        )
        .await;
        let roles: Vec<&str> = payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user", "assistant", "user", "system", "assistant"]);
        assert_eq!(payload["messages"][2]["content"][0]["type"], "tool_result");
    }

    // anthropic-adaptive-thinking-models.test.ts (in-scope providers only)

    #[test]
    fn marks_built_in_anthropic_messages_models_that_use_adaptive_thinking() {
        let mut flagged: Vec<String> = crate::providers::all::get_builtin_providers()
            .into_iter()
            .flat_map(crate::providers::all::get_builtin_models)
            .filter(|model| model.api == "anthropic-messages")
            .filter(|model| {
                model
                    .compat
                    .as_ref()
                    .and_then(|compat| compat.force_adaptive_thinking)
                    == Some(true)
            })
            .map(|model| format!("{}/{}", model.provider, model.id))
            .collect();
        flagged.sort();
        for expected in [
            "anthropic/claude-fable-5",
            "anthropic/claude-opus-4-8",
            "anthropic/claude-opus-5",
            "anthropic/claude-sonnet-5",
        ] {
            assert!(
                flagged.iter().any(|id| id == expected),
                "{expected} in {flagged:?}"
            );
        }
        // /(opus[-.](4[-.][678]|5)|sonnet[-.]4[-.]6|sonnet[-.]5|fable[-.]5|kimi-coding\/)/
        let mut patterns = Vec::new();
        for a in ['-', '.'] {
            patterns.push(format!("opus{a}5"));
            patterns.push(format!("sonnet{a}5"));
            patterns.push(format!("fable{a}5"));
            for b in ['-', '.'] {
                for minor in ['6', '7', '8'] {
                    patterns.push(format!("opus{a}4{b}{minor}"));
                }
                patterns.push(format!("sonnet{a}4{b}6"));
            }
        }
        patterns.push("kimi-coding/".to_string());
        for id in &flagged {
            assert!(
                patterns.iter().any(|pattern| id.contains(pattern.as_str())),
                "{id} is flagged but not an adaptive-thinking model"
            );
        }
    }

    // github-copilot-anthropic.test.ts

    #[test]
    fn applies_copilot_specific_adaptive_thinking_effort_overrides() {
        use crate::models::get_supported_thinking_levels;
        use crate::types::ModelThinkingLevel as Level;

        let map_contains = |model: &Model, entries: &[(Level, &str)]| {
            let map = model
                .thinking_level_map
                .as_ref()
                .expect("thinking level map");
            for (level, effort) in entries {
                assert_eq!(
                    map.get(level).cloned().flatten().as_deref(),
                    Some(*effort),
                    "{} {level:?}",
                    model.id
                );
            }
        };

        let opus47 = builtin("github-copilot", "claude-opus-4.7");
        map_contains(
            &opus47,
            &[
                (Level::Minimal, "low"),
                (Level::Xhigh, "xhigh"),
                (Level::Max, "max"),
            ],
        );
        assert!(get_supported_thinking_levels(&opus47).contains(&Level::Xhigh));
        assert!(get_supported_thinking_levels(&opus47).contains(&Level::Max));

        let opus5 = builtin("github-copilot", "claude-opus-5");
        assert_eq!(opus5.api, "anthropic-messages");
        assert_eq!(opus5.context_window, 1_000_000);
        map_contains(
            &opus5,
            &[
                (Level::Minimal, "low"),
                (Level::Xhigh, "xhigh"),
                (Level::Max, "max"),
            ],
        );
        assert!(get_supported_thinking_levels(&opus5).contains(&Level::Xhigh));
        assert!(get_supported_thinking_levels(&opus5).contains(&Level::Max));

        let opus55 = builtin("github-copilot", "claude-opus-5.5");
        assert_eq!(opus55.api, "anthropic-messages");
        assert_eq!(opus55.context_window, 1_000_000);
        assert_eq!(
            get_supported_thinking_levels(&opus55),
            [
                Level::Low,
                Level::Medium,
                Level::High,
                Level::Xhigh,
                Level::Max
            ]
        );

        let sonnet46 = builtin("github-copilot", "claude-sonnet-4.6");
        map_contains(&sonnet46, &[(Level::Minimal, "low"), (Level::Max, "max")]);
        assert!(get_supported_thinking_levels(&sonnet46).contains(&Level::Max));
        assert!(!get_supported_thinking_levels(&sonnet46).contains(&Level::Xhigh));
    }

    #[tokio::test]
    async fn uses_bearer_auth_copilot_headers_and_a_valid_messages_payload() {
        let model = builtin("github-copilot", "claude-sonnet-4.6");
        assert_eq!(model.api, "anthropic-messages");
        let user_agent = model.headers.as_ref().unwrap().get("User-Agent").cloned();
        let max_tokens = model.max_tokens;
        let mut options = api_key("tid_copilot_session_test_token");
        options.interleaved_thinking = Some(true);
        options.thinking_enabled = Some(true);
        let (_, request) = run(model, system_hello(), options, auth_events()).await;
        let request = request.unwrap();
        assert_eq!(
            request.header("authorization"),
            Some("Bearer tid_copilot_session_test_token")
        );
        assert_eq!(request.header("x-api-key"), None);
        assert!(
            request
                .header("user-agent")
                .unwrap()
                .contains("GitHubCopilotChat")
        );
        assert_eq!(request.header("user-agent"), user_agent.as_deref());
        assert_eq!(
            request.header("copilot-integration-id"),
            Some("vscode-chat")
        );
        assert_eq!(request.header("x-initiator"), Some("user"));
        assert_eq!(request.header("openai-intent"), Some("conversation-edits"));
        let betas = request.header("anthropic-beta").unwrap_or_default();
        assert!(!betas.contains(FINE_GRAINED_TOOL_STREAMING_BETA));
        assert!(!betas.contains(INTERLEAVED_THINKING_BETA));
        assert_eq!(request.body["model"], "claude-sonnet-4.6");
        assert_eq!(request.body["stream"], json!(true));
        assert_eq!(request.body["max_tokens"], json!(max_tokens));
        assert!(request.body["messages"].is_array());
    }

    // pre-generation-error.test.ts

    #[test]
    fn stream_simple_throws_synchronously_when_auth_is_missing() {
        let model = custom_model("test-model", "test-provider", Value::Null);
        let error = stream_simple_anthropic(
            model,
            context(json!({ "messages": [] })),
            SimpleStreamOptions::default(),
        )
        .err()
        .unwrap();
        assert_eq!(error.to_string(), "No API key for provider: test-provider");
    }

    #[tokio::test]
    async fn stream_reports_missing_auth_and_unsupported_federation() {
        let model = custom_model("test-model", "test-provider", Value::Null);
        let result = stream_anthropic(model, hello(), AnthropicOptions::default())
            .result()
            .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("No API key for provider: test-provider")
        );

        let mut options = AnthropicOptions::default();
        options.env = Some(
            [
                (ANTHROPIC_FEDERATION_RULE_ID_ENV, "rule"),
                (ANTHROPIC_ORGANIZATION_ID_ENV, "org"),
                (ANTHROPIC_IDENTITY_TOKEN_FILE_ENV, "/token"),
            ]
            .into_iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect(),
        );
        let result = stream_anthropic(test_model(), hello(), options)
            .result()
            .await;
        assert!(
            result
                .error_message
                .unwrap()
                .contains("workload identity federation is not supported yet")
        );
    }

    // fetch-option.test.ts (Anthropic)

    #[tokio::test]
    async fn passes_the_http_client_through_stream_simple() {
        let (base_url, requests) = serve_sse(sse(&minimal_events())).await;
        let mut model = builtin("anthropic", "claude-haiku-4-5");
        model.base_url = base_url;
        let mut options = SimpleStreamOptions::default();
        options.stream.api_key = Some("test-key".to_string());
        options.stream.http_client = Some(
            reqwest::Client::builder()
                .user_agent("custom-http-client")
                .build()
                .unwrap(),
        );
        options.stream.headers = Some([("User-Agent", None::<String>)].into_iter().collect());
        let result = stream_simple_anthropic(model, hello(), options)
            .unwrap()
            .result()
            .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        let request = requests.lock().unwrap().first().cloned().unwrap();
        assert_eq!(request.header("user-agent"), Some("custom-http-client"));
    }

    // The public handle API.

    #[tokio::test]
    async fn streams_through_a_provider_handle() {
        let (base_url, requests) = serve_sse(sse(&minimal_events())).await;
        let handle = crate::providers::anthropic::builder()
            .api_key("handle-key")
            .base_url(base_url)
            .build()
            .unwrap();
        let model = handle.model("claude-haiku-4-5").build().unwrap();
        let context: Context = serde_json::from_value(json!({
            "messages": [{ "role": "user", "content": "Hello", "timestamp": 1 }],
        }))
        .unwrap();
        let result = handle
            .models()
            .complete_simple(&model, &context, SimpleStreamOptions::default())
            .await;
        assert_eq!(
            result.stop_reason,
            StopReason::Stop,
            "{:?}",
            result.error_message
        );
        assert_eq!(
            result.content,
            vec![AssistantContent::Text(TextContent::new("Hello"))]
        );
        let request = requests.lock().unwrap().first().cloned().unwrap();
        assert_eq!(request.header("x-api-key"), Some("handle-key"));
        assert_eq!(request.body["model"], "claude-haiku-4-5");
    }

    // Unit helpers.

    #[test]
    fn normalizes_tool_call_ids_like_javascript() {
        assert_eq!(normalize_tool_call_id("call|abc.def"), "call_abc_def");
        assert_eq!(normalize_tool_call_id("a\u{1F600}b"), "a__b");
        assert_eq!(normalize_tool_call_id(&"x".repeat(80)).len(), 64);
    }

    #[test]
    fn round_trips_anthropic_options_through_provider_options() {
        let options = AnthropicOptions {
            thinking_enabled: Some(true),
            thinking_budget_tokens: Some(2048),
            effort: Some(AnthropicEffort::Xhigh),
            thinking_display: Some(AnthropicThinkingDisplay::Omitted),
            interleaved_thinking: Some(false),
            tool_choice: Some(AnthropicToolChoice::Tool {
                name: "lookup".to_string(),
            }),
            ..Default::default()
        };
        let stream_options = options.into_stream_options();
        assert_eq!(
            Value::Object(stream_options.provider_options.clone()),
            json!({
                "thinkingEnabled": true,
                "thinkingBudgetTokens": 2048,
                "effort": "xhigh",
                "thinkingDisplay": "omitted",
                "interleavedThinking": false,
                "toolChoice": { "type": "tool", "name": "lookup" },
            })
        );
        let parsed = AnthropicOptions::from_stream_options(stream_options);
        assert_eq!(parsed.effort, Some(AnthropicEffort::Xhigh));
        assert_eq!(
            parsed.tool_choice,
            Some(AnthropicToolChoice::Tool {
                name: "lookup".to_string()
            })
        );
        assert!(parsed.provider_options.is_empty());
    }

    #[tokio::test]
    async fn sends_tool_choice_and_metadata() {
        let mut options = no_cache();
        options.tool_choice = Some(AnthropicToolChoice::Any);
        options.metadata = Some(
            json!({ "user_id": "user-1", "other": 1 })
                .as_object()
                .unwrap()
                .clone(),
        );
        let (payload, _) = capture(
            builtin("anthropic", "claude-haiku-4-5"),
            tool_context(json!([lookup_tool()])),
            options,
        )
        .await;
        assert_eq!(payload["tool_choice"], json!({ "type": "any" }));
        assert_eq!(payload["metadata"], json!({ "user_id": "user-1" }));
        let keys: Vec<&str> = payload
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "model",
                "messages",
                "max_tokens",
                "stream",
                "tools",
                "metadata",
                "tool_choice"
            ]
        );
    }
}
