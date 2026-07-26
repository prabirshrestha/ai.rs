use futures::{StreamExt, pin_mut};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

use crate::env_api_keys::{KnownProvider, get_anthropic_auth_token, get_env_api_key};
use crate::event_stream::AssistantMessageEventStreamSender;
use crate::models::calculate_cost;
use crate::provider::{LanguageModelApi, ModelBuilder, Provider, ProviderCapabilities};
use crate::providers::constrained_sampling::resolve_json_schema_strict_sampling;
use crate::providers::github_copilot_headers::{
    build_copilot_dynamic_headers, has_copilot_vision_input,
};
use crate::providers::simple_options;
use crate::providers::simple_options::{adjust_max_tokens_for_thinking, clamped_reasoning};
use crate::providers::transform_messages::transform_messages;
use crate::types::{
    AnthropicMessagesCompat, AssistantContent, AssistantMessage, AssistantMessageEvent,
    CacheRetention, Context, Model, ModelCompat, ModelThinkingLevel, SimpleStreamOptions,
    StopReason, StreamOptions, TextContent, ThinkingContent, Tool, ToolCall, ToolResultContent,
    UserContent, UserMessageContent,
};
use crate::utils::http::{request_timeout, send_with_retries};
use crate::utils::json::{parse_json_with_repair, parse_streaming_json};
use crate::utils::sse;
use crate::{Error, Result};

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const CLAUDE_CODE_VERSION: &str = "2.1.75";
const DEFAULT_PROVIDER_ID: KnownProvider = KnownProvider::Anthropic;
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

#[derive(Clone)]
pub struct Anthropic {
    provider_id: String,
    api_key: Option<String>,
    auth_token: Option<String>,
    base_url: String,
    http_client: Option<reqwest::Client>,
}

impl Anthropic {
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::default()
    }

    pub fn from_env() -> Result<Self> {
        if let Some(auth_token) = get_anthropic_auth_token() {
            return Self::builder().auth_token(auth_token).build();
        }
        let api_key = get_env_api_key(DEFAULT_PROVIDER_ID)
            .ok_or_else(|| Error::MissingApiKey(DEFAULT_PROVIDER_ID.into()))?;
        Self::builder().api_key(api_key).build()
    }

    pub fn model(&self, id: &str) -> ModelBuilder {
        <Self as Provider>::model(self, id)
    }
}

impl Provider for Anthropic {
    fn id(&self) -> &str {
        &self.provider_id
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            language_models: true,
            image_models: false,
        }
    }

    fn model(&self, id: &str) -> ModelBuilder {
        let runtime = Arc::new(AnthropicLanguageModelApi {
            api_key: self.api_key.clone(),
            auth_token: self.auth_token.clone(),
            http_client: self.http_client.clone(),
        });
        ModelBuilder::new(&self.provider_id, id, runtime)
            .base_url(self.base_url.clone())
            .input(vec![crate::ModelInput::Text, crate::ModelInput::Image])
            .context_window(1_000_000)
            .max_tokens(16_384)
            .compat(ModelCompat {
                anthropic_messages: AnthropicMessagesCompat {
                    supports_strict_tools: Some(true),
                    ..Default::default()
                },
                ..Default::default()
            })
    }
}

#[derive(Default)]
pub struct AnthropicBuilder {
    provider_id: Option<String>,
    api_key: Option<String>,
    auth_token: Option<String>,
    base_url: Option<String>,
    http_client: Option<reqwest::Client>,
}

impl AnthropicBuilder {
    pub fn provider_id(mut self, provider_id: impl Into<String>) -> Self {
        self.provider_id = Some(provider_id.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Use Anthropic bearer authentication without enabling Claude Code OAuth
    /// request shaping.
    pub fn auth_token(mut self, auth_token: impl Into<String>) -> Self {
        self.auth_token = Some(auth_token.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn build(self) -> Result<Anthropic> {
        Ok(Anthropic {
            provider_id: self
                .provider_id
                .unwrap_or_else(|| DEFAULT_PROVIDER_ID.into()),
            api_key: self.api_key,
            auth_token: self.auth_token,
            base_url: self
                .base_url
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            http_client: self.http_client,
        })
    }
}

#[derive(Clone)]
struct AnthropicLanguageModelApi {
    api_key: Option<String>,
    auth_token: Option<String>,
    http_client: Option<reqwest::Client>,
}

impl AnthropicLanguageModelApi {
    fn with_api_key(&self, mut options: StreamOptions) -> StreamOptions {
        if options
            .api_key
            .as_deref()
            .is_none_or(|api_key| api_key.trim().is_empty())
        {
            options.api_key = self.api_key.clone();
        }
        if options.api_key.is_none()
            && !has_authorization_override(&options.headers)
            && let Some(auth_token) = self
                .auth_token
                .as_deref()
                .filter(|auth_token| !auth_token.trim().is_empty())
        {
            options
                .headers
                .insert("Authorization", Some(format!("Bearer {auth_token}")));
        }
        if options.http_client.is_none() {
            options.http_client = self.http_client.clone();
        }
        options
    }

    fn with_api_key_simple(&self, mut options: SimpleStreamOptions) -> SimpleStreamOptions {
        options.stream = self.with_api_key(options.stream);
        options
    }
}

impl LanguageModelApi for AnthropicLanguageModelApi {
    fn id(&self) -> &str {
        "anthropic-messages"
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: StreamOptions,
    ) -> Result<crate::AssistantEventStream> {
        Ok(stream_anthropic(
            model,
            context,
            simple_options::anthropic_options_from_stream_options(self.with_api_key(options)),
        ))
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<crate::AssistantEventStream> {
        stream_simple_anthropic(model, context, self.with_api_key_simple(options))
    }
}

pub fn builder() -> AnthropicBuilder {
    Anthropic::builder()
}

pub fn from_env() -> Result<Anthropic> {
    Anthropic::from_env()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl AnthropicEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicThinkingDisplay {
    Summarized,
    Omitted,
}

impl AnthropicThinkingDisplay {
    fn as_str(self) -> &'static str {
        match self {
            Self::Summarized => "summarized",
            Self::Omitted => "omitted",
        }
    }
}

#[derive(Clone)]
pub struct AnthropicOptions {
    pub base: StreamOptions,
    pub thinking_enabled: Option<bool>,
    pub thinking_budget_tokens: Option<u32>,
    pub effort: Option<AnthropicEffort>,
    pub thinking_display: Option<AnthropicThinkingDisplay>,
    pub interleaved_thinking: bool,
    pub tool_choice: Option<Value>,
}

impl Default for AnthropicOptions {
    fn default() -> Self {
        Self {
            base: StreamOptions::default(),
            thinking_enabled: None,
            thinking_budget_tokens: None,
            effort: None,
            thinking_display: None,
            interleaved_thinking: true,
            tool_choice: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedAnthropicCompat {
    pub supports_eager_tool_input_streaming: bool,
    pub supports_long_cache_retention: bool,
    pub send_session_affinity_headers: bool,
    pub supports_cache_control_on_tools: bool,
    pub supports_temperature: bool,
    pub supports_strict_tools: bool,
}

pub fn stream_simple_anthropic(
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
) -> crate::Result<crate::AssistantEventStream> {
    let api_key = options
        .stream
        .api_key
        .clone()
        .filter(|key| !key.trim().is_empty());
    if api_key.is_none() && !has_authorization_header(&options.stream.headers) {
        return Err(crate::Error::MissingApiKey(model.provider));
    }
    let mut base = options.stream.clone();
    base.api_key = api_key;

    let Some(reasoning) = clamped_reasoning(&model, &options) else {
        return Ok(stream_anthropic(
            model,
            context,
            AnthropicOptions {
                base,
                thinking_enabled: Some(false),
                ..Default::default()
            },
        ));
    };

    if model.compat.anthropic_messages.force_adaptive_thinking == Some(true) {
        return Ok(stream_anthropic(
            model.clone(),
            context,
            AnthropicOptions {
                base,
                thinking_enabled: Some(true),
                effort: Some(map_thinking_level_to_effort(&model, reasoning)),
                ..Default::default()
            },
        ));
    }

    let adjusted = adjust_max_tokens_for_thinking(
        base.max_tokens,
        model.max_tokens,
        Some(reasoning),
        options.thinking_budgets.as_ref(),
    );
    let mut adjusted_base = base;
    adjusted_base.max_tokens = adjusted.max_tokens;
    Ok(stream_anthropic(
        model,
        context,
        AnthropicOptions {
            base: adjusted_base,
            thinking_enabled: Some(true),
            thinking_budget_tokens: Some(adjusted.thinking_budget),
            ..Default::default()
        },
    ))
}

pub fn stream_anthropic(
    model: Model,
    context: Context,
    options: AnthropicOptions,
) -> crate::AssistantEventStream {
    crate::event_stream::stream_from_producer(
        move |mut sender| async move {
            let output = AssistantMessage::empty_for(&model);
            run_stream(model, context, options, output, &mut sender).await?;
            Ok(())
        },
        |error: StreamFailure| {
            let mut message = error.output;
            message.stop_reason = if error.cancelled {
                StopReason::Aborted
            } else {
                StopReason::Error
            };
            message.error_message = Some(error.message);
            AssistantMessageEvent::Error {
                reason: message.stop_reason,
                error: message,
            }
        },
    )
}

struct StreamFailure {
    output: AssistantMessage,
    message: String,
    cancelled: bool,
}

impl StreamFailure {
    fn new(output: AssistantMessage, error: impl std::fmt::Display) -> Self {
        Self {
            output,
            message: error.to_string(),
            cancelled: false,
        }
    }

    fn cancelled(output: AssistantMessage) -> Self {
        Self {
            output,
            message: "Request was aborted".to_string(),
            cancelled: true,
        }
    }
}

async fn run_stream(
    model: Model,
    context: Context,
    options: AnthropicOptions,
    mut output: AssistantMessage,
    sender: &mut AssistantMessageEventStreamSender,
) -> std::result::Result<(), StreamFailure> {
    if options
        .base
        .cancellation_token
        .as_ref()
        .is_some_and(|token| token.is_cancelled())
    {
        return Err(StreamFailure::cancelled(output));
    }

    let api_key = options
        .base
        .api_key
        .clone()
        .filter(|key| !key.trim().is_empty());
    if api_key.is_none() && !has_authorization_header(&options.base.headers) {
        return Err(StreamFailure::new(
            output,
            format!("No API key for provider: {}", model.provider),
        ));
    }
    let is_oauth = api_key.as_deref().is_some_and(is_oauth_token);
    let compat = get_anthropic_compat(&model);
    let cache_retention = resolve_cache_retention(options.base.cache_retention);
    let cache_control = cache_control(&model, cache_retention, compat);
    let mut payload = match try_build_anthropic_payload(
        &model,
        &context,
        &options,
        is_oauth,
        cache_control.clone(),
    ) {
        Ok(payload) => payload,
        Err(error) => return Err(StreamFailure::new(output, error)),
    };
    if let Some(on_payload) = &options.base.on_payload {
        match on_payload(payload.clone(), &model).await {
            Ok(Some(next)) => payload = next,
            Ok(None) => {}
            Err(error) => return Err(StreamFailure::new(output, error)),
        }
    }

    let base_url = match request_base_url(&model) {
        Ok(base_url) => base_url,
        Err(error) => return Err(StreamFailure::new(output, error)),
    };
    let request_url = format!("{}/messages", trim_end_slash(&base_url));
    let request_headers = match headers(
        &model,
        &context,
        &options,
        api_key.as_deref(),
        is_oauth,
        compat,
        cache_retention,
    ) {
        Ok(headers) => headers,
        Err(error) => return Err(StreamFailure::new(output, error)),
    };
    let client = options.base.http_client.clone().unwrap_or_default();
    let response_result = send_with_retries(&options.base, || {
        client
            .post(request_url.as_str())
            .headers(request_headers.clone())
            .json(&payload)
            .timeout(request_timeout(options.base.timeout_ms))
    })
    .await;
    let response = match response_result {
        Ok(response) => response,
        Err(Error::Cancelled) => return Err(StreamFailure::cancelled(output)),
        Err(error) => return Err(StreamFailure::new(output, error)),
    };
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(StreamFailure::new(
            output,
            Error::ApiStatus { status, body },
        ));
    }
    if let Some(on_response) = &options.base.on_response {
        let provider_response = crate::types::ProviderResponse {
            status: response.status().as_u16(),
            headers: response_headers(response.headers()),
        };
        if let Err(error) = on_response(provider_response, &model).await {
            return Err(StreamFailure::new(output, error));
        }
    }

    sender.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let mut blocks_by_anthropic_index: HashMap<i64, usize> = HashMap::new();
    let mut partial_json: HashMap<usize, String> = HashMap::new();
    let mut saw_message_start = false;
    let mut saw_message_stop = false;
    let events = sse::events(response, options.base.cancellation_token.clone());
    pin_mut!(events);
    while let Some(event) = events.next().await {
        if options
            .base
            .cancellation_token
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(StreamFailure::cancelled(output));
        }
        let event = match event {
            Ok(event) => event,
            Err(Error::Cancelled) => return Err(StreamFailure::cancelled(output)),
            Err(error) => return Err(StreamFailure::new(output, error)),
        };
        if event.event.as_deref() == Some("error") {
            return Err(StreamFailure::new(output, event.data));
        }
        if !matches!(
            event.event.as_deref(),
            Some(
                "message_start"
                    | "message_delta"
                    | "message_stop"
                    | "content_block_start"
                    | "content_block_delta"
                    | "content_block_stop"
            )
        ) {
            continue;
        }
        let parsed: Value = match parse_json_with_repair(&event.data) {
            Ok(value) => value,
            Err(error) => {
                return Err(StreamFailure::new(
                    output,
                    format!(
                        "Could not parse Anthropic SSE event {}: {}; data={}; raw={}",
                        event.event.as_deref().unwrap_or_default(),
                        error,
                        event.data,
                        event.raw.join("\\n")
                    ),
                ));
            }
        };
        match parsed.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                saw_message_start = true;
                if let Some(id) = parsed.pointer("/message/id").and_then(Value::as_str) {
                    output.response_id = Some(id.to_string());
                }
                if let Some(usage) = parsed.pointer("/message/usage") {
                    update_anthropic_start_usage(&mut output, usage, &model);
                }
            }
            Some("content_block_start") => {
                let index = parsed.get("index").and_then(Value::as_i64).unwrap_or(0);
                let Some(content_block) = parsed.get("content_block") else {
                    continue;
                };
                match content_block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        output.content.push(AssistantContent::Text(TextContent {
                            text: String::new(),
                            text_signature: None,
                        }));
                        let content_index = output.content.len() - 1;
                        blocks_by_anthropic_index.insert(index, content_index);
                        sender.push(AssistantMessageEvent::TextStart {
                            content_index,
                            partial: output.clone(),
                        });
                    }
                    Some("thinking") => {
                        output
                            .content
                            .push(AssistantContent::Thinking(ThinkingContent {
                                thinking: String::new(),
                                thinking_signature: Some(String::new()),
                                redacted: None,
                            }));
                        let content_index = output.content.len() - 1;
                        blocks_by_anthropic_index.insert(index, content_index);
                        sender.push(AssistantMessageEvent::ThinkingStart {
                            content_index,
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
                                    .map(ToString::to_string),
                                redacted: Some(true),
                            }));
                        let content_index = output.content.len() - 1;
                        blocks_by_anthropic_index.insert(index, content_index);
                        sender.push(AssistantMessageEvent::ThinkingStart {
                            content_index,
                            partial: output.clone(),
                        });
                    }
                    Some("tool_use") => {
                        output.content.push(AssistantContent::ToolCall(ToolCall {
                            id: content_block
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            name: from_claude_code_name(
                                content_block
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default(),
                                &context.tools,
                                is_oauth,
                            ),
                            arguments: content_block
                                .get("input")
                                .cloned()
                                .unwrap_or_else(|| Value::Object(Default::default())),
                            thought_signature: None,
                        }));
                        let content_index = output.content.len() - 1;
                        blocks_by_anthropic_index.insert(index, content_index);
                        partial_json.insert(content_index, String::new());
                        sender.push(AssistantMessageEvent::ToolCallStart {
                            content_index,
                            partial: output.clone(),
                        });
                    }
                    _ => {}
                }
            }
            Some("content_block_delta") => {
                let anthropic_index = parsed.get("index").and_then(Value::as_i64).unwrap_or(0);
                let Some(content_index) = blocks_by_anthropic_index.get(&anthropic_index).copied()
                else {
                    continue;
                };
                let Some(delta) = parsed.get("delta") else {
                    continue;
                };
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let text = delta
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if let Some(AssistantContent::Text(block)) =
                            output.content.get_mut(content_index)
                        {
                            block.text.push_str(text);
                            sender.push(AssistantMessageEvent::TextDelta {
                                content_index,
                                delta: text.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                    Some("thinking_delta") => {
                        let thinking = delta
                            .get("thinking")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if let Some(AssistantContent::Thinking(block)) =
                            output.content.get_mut(content_index)
                        {
                            block.thinking.push_str(thinking);
                            sender.push(AssistantMessageEvent::ThinkingDelta {
                                content_index,
                                delta: thinking.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                    Some("input_json_delta") => {
                        let delta_json = delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let entry = partial_json.entry(content_index).or_default();
                        entry.push_str(delta_json);
                        if let Some(AssistantContent::ToolCall(block)) =
                            output.content.get_mut(content_index)
                        {
                            block.arguments = parse_streaming_json(Some(entry));
                            sender.push(AssistantMessageEvent::ToolCallDelta {
                                content_index,
                                delta: delta_json.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                    Some("signature_delta") => {
                        let signature = delta
                            .get("signature")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if let Some(AssistantContent::Thinking(block)) =
                            output.content.get_mut(content_index)
                        {
                            let existing = block.thinking_signature.get_or_insert_with(String::new);
                            existing.push_str(signature);
                        }
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                let anthropic_index = parsed.get("index").and_then(Value::as_i64).unwrap_or(0);
                let Some(content_index) = blocks_by_anthropic_index.get(&anthropic_index).copied()
                else {
                    continue;
                };
                match output.content.get_mut(content_index) {
                    Some(AssistantContent::Text(block)) => {
                        sender.push(AssistantMessageEvent::TextEnd {
                            content_index,
                            content: block.text.clone(),
                            partial: output.clone(),
                        });
                    }
                    Some(AssistantContent::Thinking(block)) => {
                        sender.push(AssistantMessageEvent::ThinkingEnd {
                            content_index,
                            content: block.thinking.clone(),
                            partial: output.clone(),
                        });
                    }
                    Some(AssistantContent::ToolCall(block)) => {
                        if let Some(args) = partial_json.get(&content_index) {
                            block.arguments = parse_streaming_json(Some(args));
                        }
                        sender.push(AssistantMessageEvent::ToolCallEnd {
                            content_index,
                            tool_call: block.clone(),
                            partial: output.clone(),
                        });
                    }
                    None => {}
                }
            }
            Some("message_delta") => {
                if let Some(reason) = parsed.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    let stop_details = parsed.pointer("/delta/stop_details");
                    match map_stop_reason(reason, stop_details) {
                        Ok((stop_reason, error_message)) => {
                            output.stop_reason = stop_reason;
                            output.error_message = error_message;
                        }
                        Err(message) => return Err(StreamFailure::new(output, message)),
                    }
                }
                if let Some(usage) = parsed.get("usage") {
                    update_anthropic_usage(&mut output, usage, &model);
                }
            }
            Some("message_stop") => {
                saw_message_stop = true;
            }
            _ => {}
        }
    }

    if saw_message_start && !saw_message_stop {
        return Err(StreamFailure::new(
            output,
            "Anthropic stream ended before message_stop",
        ));
    }
    if options
        .base
        .cancellation_token
        .as_ref()
        .is_some_and(|token| token.is_cancelled())
    {
        return Err(StreamFailure::cancelled(output));
    }
    if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
        let message = output
            .error_message
            .clone()
            .unwrap_or_else(|| "An unknown error occurred".to_string());
        return Err(StreamFailure::new(output, message));
    }
    sender.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output,
    });
    Ok(())
}

fn try_build_anthropic_payload(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
    is_oauth_token: bool,
    cache_control: Option<Value>,
) -> Result<Value> {
    let mut payload = json!({
        "model": model.id,
        "messages": convert_messages(
            &context.messages,
            model,
            is_oauth_token,
            cache_control.clone(),
            model.compat.anthropic_messages.allow_empty_signature.unwrap_or(false),
        ),
        "max_tokens": options.base.max_tokens.unwrap_or(model.max_tokens),
        "stream": true
    });
    let object = payload.as_object_mut().expect("payload object");

    if is_oauth_token {
        let mut system = vec![json!({
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude."
        })];
        if let Some(cache_control) = &cache_control {
            system[0]["cache_control"] = cache_control.clone();
        }
        if let Some(system_prompt) = &context.system_prompt
            && !system_prompt.is_empty()
        {
            let mut item = json!({ "type": "text", "text": system_prompt });
            if let Some(cache_control) = &cache_control {
                item["cache_control"] = cache_control.clone();
            }
            system.push(item);
        }
        object.insert("system".to_string(), Value::Array(system));
    } else if let Some(system_prompt) = &context.system_prompt
        && !system_prompt.is_empty()
    {
        let mut item = json!({ "type": "text", "text": system_prompt });
        if let Some(cache_control) = &cache_control {
            item["cache_control"] = cache_control.clone();
        }
        object.insert("system".to_string(), json!([item]));
    }

    if let Some(temperature) = options.base.temperature
        && options.thinking_enabled != Some(true)
        && get_anthropic_compat(model).supports_temperature
    {
        object.insert("temperature".to_string(), json!(temperature));
    }
    if !context.tools.is_empty() {
        let compat = get_anthropic_compat(model);
        object.insert(
            "tools".to_string(),
            Value::Array(convert_tools(
                &context.tools,
                is_oauth_token,
                compat.supports_eager_tool_input_streaming,
                compat.supports_strict_tools,
                if compat.supports_cache_control_on_tools {
                    cache_control.clone()
                } else {
                    None
                },
            )?),
        );
    }
    if model.reasoning {
        if options.thinking_enabled == Some(true) {
            let display = options
                .thinking_display
                .unwrap_or(AnthropicThinkingDisplay::Summarized)
                .as_str();
            if model.compat.anthropic_messages.force_adaptive_thinking == Some(true) {
                object.insert(
                    "thinking".to_string(),
                    json!({ "type": "adaptive", "display": display }),
                );
                if let Some(effort) = options.effort {
                    object.insert(
                        "output_config".to_string(),
                        json!({ "effort": effort.as_str() }),
                    );
                }
            } else {
                object.insert(
                    "thinking".to_string(),
                    json!({
                        "type": "enabled",
                        "budget_tokens": options.thinking_budget_tokens.unwrap_or(1024),
                        "display": display
                    }),
                );
            }
        } else if options.thinking_enabled == Some(false) {
            object.insert("thinking".to_string(), json!({ "type": "disabled" }));
        }
    }
    if let Some(metadata) = &options.base.metadata
        && let Some(user_id) = metadata.get("user_id").and_then(Value::as_str)
    {
        object.insert("metadata".to_string(), json!({ "user_id": user_id }));
    }
    if let Some(tool_choice) = &options.tool_choice {
        let value = tool_choice
            .as_str()
            .map(|choice| json!({ "type": choice }))
            .unwrap_or_else(|| tool_choice.clone());
        object.insert("tool_choice".to_string(), value);
    }
    Ok(payload)
}

#[cfg(test)]
fn build_anthropic_payload(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
    is_oauth_token: bool,
    cache_control: Option<Value>,
) -> Value {
    try_build_anthropic_payload(model, context, options, is_oauth_token, cache_control).unwrap()
}

fn convert_messages(
    messages: &[crate::types::Message],
    model: &Model,
    is_oauth_token: bool,
    cache_control: Option<Value>,
    allow_empty_signature: bool,
) -> Vec<Value> {
    let transformed = transform_messages(messages, model, |id, _model, _source| {
        normalize_tool_call_id(id)
    });
    let mut params = Vec::new();
    let mut index = 0usize;
    while index < transformed.len() {
        match &transformed[index] {
            crate::types::Message::User(user) => match &user.content {
                UserMessageContent::Text(text) => {
                    if !text.trim().is_empty() {
                        params.push(json!({ "role": "user", "content": text }));
                    }
                }
                UserMessageContent::Parts(parts) => {
                    let blocks: Vec<Value> = parts
                        .iter()
                        .filter_map(|item| match item {
                            UserContent::Text(text) => (!text.text.trim().is_empty())
                                .then(|| json!({ "type": "text", "text": &text.text })),
                            UserContent::Image(image) => Some(json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image.mime_type,
                                    "data": image.data
                                }
                            })),
                        })
                        .collect();
                    if !blocks.is_empty() {
                        params.push(json!({ "role": "user", "content": blocks }));
                    }
                }
            },
            crate::types::Message::Assistant(assistant) => {
                let mut blocks = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(text) if !text.text.trim().is_empty() => {
                            blocks.push(json!({ "type": "text", "text": &text.text }));
                        }
                        AssistantContent::Thinking(thinking) if thinking.redacted == Some(true) => {
                            if let Some(signature) = &thinking.thinking_signature {
                                blocks.push(json!({ "type": "redacted_thinking", "data": signature }));
                            }
                        }
                        AssistantContent::Thinking(thinking) => {
                            let signature = thinking
                                .thinking_signature
                                .as_deref()
                                .filter(|signature| !signature.trim().is_empty());
                            if thinking.thinking.trim().is_empty() && signature.is_none() {
                                continue;
                            }
                            match signature {
                                Some(signature) => blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": &thinking.thinking,
                                    "signature": signature
                                })),
                                None if allow_empty_signature => blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": &thinking.thinking,
                                    "signature": ""
                                })),
                                None => blocks.push(json!({
                                    "type": "text",
                                    "text": &thinking.thinking
                                })),
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => blocks.push(json!({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": if is_oauth_token { to_claude_code_name(&tool_call.name) } else { tool_call.name.clone() },
                            "input": tool_call.arguments
                        })),
                        _ => {}
                    }
                }
                if !blocks.is_empty() {
                    params.push(json!({ "role": "assistant", "content": blocks }));
                }
            }
            crate::types::Message::ToolResult(tool_result) => {
                let mut tool_results = vec![json!({
                    "type": "tool_result",
                    "tool_use_id": tool_result.tool_call_id,
                    "content": convert_content_blocks(&tool_result.content),
                    "is_error": tool_result.is_error
                })];
                let mut cursor = index + 1;
                while cursor < transformed.len() {
                    let crate::types::Message::ToolResult(next) = &transformed[cursor] else {
                        break;
                    };
                    tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": next.tool_call_id,
                        "content": convert_content_blocks(&next.content),
                        "is_error": next.is_error
                    }));
                    cursor += 1;
                }
                index = cursor - 1;
                params.push(json!({ "role": "user", "content": tool_results }));
            }
            crate::types::Message::Custom(_) => {}
        }
        index += 1;
    }

    if let Some(cache_control) = cache_control
        && let Some(last) = params
            .last_mut()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("user"))
    {
        match last.get_mut("content") {
            Some(Value::Array(blocks)) => {
                if let Some(block) = blocks.last_mut()
                    && matches!(
                        block.get("type").and_then(Value::as_str),
                        Some("text" | "image" | "tool_result")
                    )
                {
                    block["cache_control"] = cache_control;
                }
            }
            Some(Value::String(text)) => {
                let text = std::mem::take(text);
                last["content"] =
                    json!([{ "type": "text", "text": text, "cache_control": cache_control }]);
            }
            _ => {}
        }
    }

    params
}

fn convert_content_blocks(content: &[ToolResultContent]) -> Value {
    let has_images = content
        .iter()
        .any(|content| matches!(content, ToolResultContent::Image(_)));
    if !has_images {
        let text = content
            .iter()
            .filter_map(|content| match content {
                ToolResultContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        return json!(&text);
    }

    let mut blocks: Vec<Value> = content
        .iter()
        .map(|block| match block {
            ToolResultContent::Text(text) => {
                json!({ "type": "text", "text": &text.text })
            }
            ToolResultContent::Image(image) => json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type,
                    "data": image.data
                }
            }),
        })
        .collect();
    let has_text = blocks
        .iter()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("text"));
    if !has_text {
        blocks.insert(0, json!({ "type": "text", "text": "(see attached image)" }));
    }
    Value::Array(blocks)
}

fn convert_tools(
    tools: &[Tool],
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: Option<Value>,
) -> Result<Vec<Value>> {
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let strict = resolve_json_schema_strict_sampling(tool, supports_strict_tools)?;
            let legacy_input_schema = json!({
                "type": "object",
                "properties": tool.parameters.get("properties").cloned().unwrap_or_else(|| json!({})),
                "required": tool.parameters.get("required").cloned().unwrap_or_else(|| json!([]))
            });
            let input_schema = if strict == Some(true) {
                let mut input_schema = tool.parameters.clone();
                let object = input_schema.as_object_mut().expect("tool parameters object");
                object.insert("type".to_string(), json!("object"));
                object.insert(
                    "properties".to_string(),
                    legacy_input_schema["properties"].clone(),
                );
                object.insert(
                    "required".to_string(),
                    legacy_input_schema["required"].clone(),
                );
                input_schema
            } else {
                legacy_input_schema
            };
            let mut value = json!({
                "name": if is_oauth_token { to_claude_code_name(&tool.name) } else { tool.name.clone() },
                "description": tool.description,
                "input_schema": input_schema
            });
            if strict == Some(true) {
                value["strict"] = json!(true);
            }
            if supports_eager_tool_input_streaming {
                value["eager_input_streaming"] = json!(true);
            }
            if index == tools.len() - 1
                && let Some(cache_control) = &cache_control {
                    value["cache_control"] = cache_control.clone();
                }
            Ok(value)
        })
        .collect()
}

fn update_anthropic_usage(output: &mut AssistantMessage, usage: &Value, model: &Model) {
    if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
        output.usage.input = input as u32;
    }
    if let Some(output_tokens) = usage.get("output_tokens").and_then(Value::as_u64) {
        output.usage.output = output_tokens as u32;
    }
    if let Some(cache_read) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
        output.usage.cache_read = cache_read as u32;
    }
    if let Some(cache_write) = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
    {
        output.usage.cache_write = cache_write as u32;
    }
    if let Some(cache_write_1h) = usage
        .pointer("/cache_creation/ephemeral_1h_input_tokens")
        .and_then(Value::as_u64)
    {
        output.usage.cache_write_1h = Some(cache_write_1h as u32);
    }
    if let Some(reasoning) = usage
        .pointer("/output_tokens_details/thinking_tokens")
        .and_then(Value::as_u64)
    {
        output.usage.reasoning = Some(reasoning as u32);
    }
    output.usage.total_tokens = output.usage.input
        + output.usage.output
        + output.usage.cache_read
        + output.usage.cache_write;
    calculate_cost(model, &mut output.usage);
}

fn update_anthropic_start_usage(output: &mut AssistantMessage, usage: &Value, model: &Model) {
    // pi exposes this provider-supported breakdown as zero when Anthropic
    // omits it from message_start, while leaving it absent for providers that
    // do not report the breakdown at all.
    output.usage.cache_write_1h = Some(
        usage
            .pointer("/cache_creation/ephemeral_1h_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
    );
    update_anthropic_usage(output, usage, model);
}

fn get_anthropic_compat(model: &Model) -> ResolvedAnthropicCompat {
    let compat = &model.compat.anthropic_messages;
    ResolvedAnthropicCompat {
        supports_eager_tool_input_streaming: compat
            .supports_eager_tool_input_streaming
            .unwrap_or(true),
        supports_long_cache_retention: compat.supports_long_cache_retention.unwrap_or(true),
        send_session_affinity_headers: compat.send_session_affinity_headers.unwrap_or(false),
        supports_cache_control_on_tools: compat.supports_cache_control_on_tools.unwrap_or(true),
        supports_temperature: compat.supports_temperature.unwrap_or(true),
        supports_strict_tools: compat.supports_strict_tools.unwrap_or(false),
    }
}

fn cache_control(
    _model: &Model,
    retention: CacheRetention,
    compat: ResolvedAnthropicCompat,
) -> Option<Value> {
    if retention == CacheRetention::None {
        return None;
    }
    let mut value = json!({ "type": "ephemeral" });
    if retention == CacheRetention::Long && compat.supports_long_cache_retention {
        value["ttl"] = json!("1h");
    }
    Some(value)
}

fn resolve_cache_retention(cache_retention: Option<CacheRetention>) -> CacheRetention {
    // Intentionally do not port pi's PI_CACHE_RETENTION environment fallback.
    // ai.rs is a library: applications that want environment-driven policy can
    // translate it into StreamOptions::cache_retention at their boundary.
    cache_retention.unwrap_or(CacheRetention::Short)
}

fn should_use_fine_grained_tool_streaming_beta(model: &Model, context: &Context) -> bool {
    !context.tools.is_empty() && !get_anthropic_compat(model).supports_eager_tool_input_streaming
}

fn headers(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
    api_key: Option<&str>,
    is_oauth: bool,
    compat: ResolvedAnthropicCompat,
    cache_retention: CacheRetention,
) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        HeaderName::from_static("accept"),
        HeaderValue::from_static("application/json"),
    );
    headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );
    headers.insert(
        HeaderName::from_static("anthropic-dangerous-direct-browser-access"),
        HeaderValue::from_static("true"),
    );

    let mut beta_features = Vec::new();
    if should_use_fine_grained_tool_streaming_beta(model, context) {
        beta_features.push(FINE_GRAINED_TOOL_STREAMING_BETA);
    }
    if options.interleaved_thinking
        && model.compat.anthropic_messages.force_adaptive_thinking != Some(true)
    {
        beta_features.push(INTERLEAVED_THINKING_BETA);
    }

    if model.provider == "github-copilot" || is_oauth {
        let api_key = api_key.unwrap_or_default();
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| Error::InvalidHeaderValue("authorization".to_string(), e))?,
        );
        if is_oauth {
            beta_features.insert(0, "oauth-2025-04-20");
            beta_features.insert(0, "claude-code-20250219");
            headers.insert(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_str(&format!("claude-cli/{CLAUDE_CODE_VERSION}"))
                    .map_err(|e| Error::InvalidHeaderValue("user-agent".to_string(), e))?,
            );
            headers.insert(
                HeaderName::from_static("x-app"),
                HeaderValue::from_static("cli"),
            );
        }
    } else if let Some(api_key) = api_key.filter(|api_key| !api_key.is_empty()) {
        headers.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(api_key)
                .map_err(|e| Error::InvalidHeaderValue("x-api-key".to_string(), e))?,
        );
    }

    if !beta_features.is_empty() {
        headers.insert(
            HeaderName::from_static("anthropic-beta"),
            HeaderValue::from_str(&beta_features.join(","))
                .map_err(|e| Error::InvalidHeaderValue("anthropic-beta".to_string(), e))?,
        );
    }
    if let Some(session_id) = &options.base.session_id
        && cache_retention != CacheRetention::None
        && compat.send_session_affinity_headers
    {
        headers.insert(
            HeaderName::from_static("x-session-affinity"),
            HeaderValue::from_str(session_id)
                .map_err(|e| Error::InvalidHeaderValue("x-session-affinity".to_string(), e))?,
        );
    }
    for (name, value) in &model.headers {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let value = HeaderValue::from_str(value)
            .map_err(|e| Error::InvalidHeaderValue(name.to_string(), e))?;
        headers.insert(name, value);
    }
    if model.provider == "github-copilot" {
        for (name, value) in build_copilot_dynamic_headers(
            &context.messages,
            has_copilot_vision_input(&context.messages),
        ) {
            let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
                continue;
            };
            let value = HeaderValue::from_str(&value)
                .map_err(|e| Error::InvalidHeaderValue(name.to_string(), e))?;
            headers.insert(name, value);
        }
    }
    crate::utils::headers::apply_provider_headers(&mut headers, &options.base.headers)?;
    Ok(headers)
}

fn response_headers(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
        .collect()
}

fn has_authorization_header(headers: &crate::types::ProviderHeaders) -> bool {
    headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("authorization")
            && value
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
    })
}

fn has_authorization_override(headers: &crate::types::ProviderHeaders) -> bool {
    headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
}

fn map_thinking_level_to_effort(model: &Model, level: ModelThinkingLevel) -> AnthropicEffort {
    if let Some(Some(mapped)) = model.thinking_level_map.get(level.as_str()) {
        return match mapped.as_str() {
            "low" => AnthropicEffort::Low,
            "medium" => AnthropicEffort::Medium,
            "high" => AnthropicEffort::High,
            "xhigh" => AnthropicEffort::Xhigh,
            "max" => AnthropicEffort::Max,
            _ => AnthropicEffort::High,
        };
    }
    match level {
        ModelThinkingLevel::Minimal | ModelThinkingLevel::Low => AnthropicEffort::Low,
        ModelThinkingLevel::Medium => AnthropicEffort::Medium,
        ModelThinkingLevel::High => AnthropicEffort::High,
        ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max => AnthropicEffort::High,
        _ => AnthropicEffort::High,
    }
}

fn normalize_tool_call_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

fn map_stop_reason(
    reason: &str,
    stop_details: Option<&Value>,
) -> std::result::Result<(StopReason, Option<String>), String> {
    match reason {
        "end_turn" | "pause_turn" | "stop_sequence" => Ok((StopReason::Stop, None)),
        "max_tokens" => Ok((StopReason::Length, None)),
        "tool_use" => Ok((StopReason::ToolUse, None)),
        "refusal" => Ok((
            StopReason::Error,
            Some(
                stop_details
                    .and_then(|details| details.get("explanation"))
                    .and_then(Value::as_str)
                    .unwrap_or("The model refused to complete the request")
                    .to_string(),
            ),
        )),
        "sensitive" => Ok((StopReason::Error, None)),
        _ => Err(format!("Unhandled stop reason: {reason}")),
    }
}

fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

fn trim_end_slash(url: &str) -> &str {
    url.trim_end_matches('/')
}

fn request_base_url(model: &Model) -> Result<String> {
    Ok(model.base_url.clone())
}

fn to_claude_code_name(name: &str) -> String {
    let canonical = [
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
    canonical
        .iter()
        .find(|candidate| candidate.eq_ignore_ascii_case(name))
        .copied()
        .unwrap_or(name)
        .to_string()
}

fn from_claude_code_name(name: &str, tools: &[Tool], is_oauth: bool) -> String {
    if !is_oauth {
        return name.to_string();
    }
    tools
        .iter()
        .find(|tool| tool.name.eq_ignore_ascii_case(name))
        .map(|tool| tool.name.clone())
        .unwrap_or_else(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use futures::StreamExt;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::types::{
        AssistantContent, CacheRetention, ModelCost, ModelInput, ResponseHook, Usage,
    };

    fn anthropic_model(id: &str) -> Model {
        Model {
            id: id.to_string(),
            name: id.to_string(),
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            base_url: "http://127.0.0.1:9".to_string(),
            reasoning: true,
            input: vec![ModelInput::Text],
            cost: ModelCost::default(),
            context_window: 1_000_000,
            max_tokens: 1024,
            ..Default::default()
        }
    }

    struct SavedEnv {
        key: &'static str,
        value: Option<String>,
    }

    impl SavedEnv {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                value: std::env::var(key).ok(),
            }
        }

        fn restore(self) {
            unsafe {
                if let Some(value) = self.value {
                    std::env::set_var(self.key, value);
                } else {
                    std::env::remove_var(self.key);
                }
            }
        }
    }

    #[test]
    fn from_env_prefers_anthropic_auth_token_as_bearer_credentials() {
        use crate::env_api_keys::{
            ANTHROPIC_API_KEY_ENV_VAR, ANTHROPIC_AUTH_TOKEN_ENV_VAR, ANTHROPIC_OAUTH_TOKEN_ENV_VAR,
            ENV_LOCK,
        };

        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let auth = SavedEnv::capture(ANTHROPIC_AUTH_TOKEN_ENV_VAR);
        let oauth = SavedEnv::capture(ANTHROPIC_OAUTH_TOKEN_ENV_VAR);
        let api_key = SavedEnv::capture(ANTHROPIC_API_KEY_ENV_VAR);
        unsafe {
            std::env::set_var(ANTHROPIC_AUTH_TOKEN_ENV_VAR, "auth-token");
            std::env::set_var(ANTHROPIC_OAUTH_TOKEN_ENV_VAR, "oauth-token");
            std::env::set_var(ANTHROPIC_API_KEY_ENV_VAR, "api-key");
        }

        let provider = from_env().expect("provider");

        assert_eq!(provider.api_key, None);
        assert_eq!(provider.auth_token.as_deref(), Some("auth-token"));

        auth.restore();
        oauth.restore();
        api_key.restore();
    }

    #[tokio::test]
    async fn from_env_sends_auth_token_as_bearer_without_oauth_shaping() {
        use crate::env_api_keys::{
            ANTHROPIC_API_KEY_ENV_VAR, ANTHROPIC_AUTH_TOKEN_ENV_VAR, ANTHROPIC_OAUTH_TOKEN_ENV_VAR,
            ENV_LOCK,
        };

        let (base_url, request) =
            spawn_request_capture_server(successful_anthropic_sse_body()).await;
        let stream = {
            let _guard = ENV_LOCK.lock().expect("env lock poisoned");
            let auth = SavedEnv::capture(ANTHROPIC_AUTH_TOKEN_ENV_VAR);
            let oauth = SavedEnv::capture(ANTHROPIC_OAUTH_TOKEN_ENV_VAR);
            let api_key = SavedEnv::capture(ANTHROPIC_API_KEY_ENV_VAR);
            unsafe {
                std::env::set_var(ANTHROPIC_AUTH_TOKEN_ENV_VAR, "auth-token");
                std::env::set_var(ANTHROPIC_OAUTH_TOKEN_ENV_VAR, "oauth-token");
                std::env::set_var(ANTHROPIC_API_KEY_ENV_VAR, "api-key");
            }
            let mut provider = from_env().expect("provider");
            provider.base_url = base_url;
            let model = provider.model("claude-test").build().expect("model");
            let stream = crate::stream_simple(
                model,
                Context {
                    system_prompt: Some("System prompt.".to_string()),
                    messages: vec![crate::types::Message::user_text("Hello")],
                    ..Default::default()
                },
                None,
            )
            .expect("stream");
            auth.restore();
            oauth.restore();
            api_key.restore();
            stream
        };
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .expect("response");
        let request = request.await.expect("captured request");
        let request_lower = request.to_ascii_lowercase();

        assert_eq!(result.stop_reason, StopReason::Stop);
        assert!(request_lower.contains("authorization: bearer auth-token"));
        assert!(!request_lower.contains("x-api-key:"));
        assert!(!request_lower.contains("oauth-2025-04-20"));
        assert!(request.contains("System prompt."));
        assert!(!request.contains("You are Claude Code"));
    }

    #[test]
    fn bearer_authorization_does_not_enable_oauth_request_shaping() {
        let model = anthropic_model("claude-test");
        let context = Context {
            system_prompt: Some("System prompt.".to_string()),
            messages: vec![crate::types::Message::user_text("Hello")],
            ..Default::default()
        };
        let mut options = AnthropicOptions::default();
        options
            .base
            .headers
            .insert("Authorization", Some("Bearer gateway-token".to_string()));
        let request_headers = headers(
            &model,
            &context,
            &options,
            None,
            false,
            get_anthropic_compat(&model),
            CacheRetention::None,
        )
        .unwrap();
        let payload = build_anthropic_payload(&model, &context, &options, false, None);

        assert_eq!(
            request_headers
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer gateway-token")
        );
        assert!(request_headers.get("x-api-key").is_none());
        assert!(
            request_headers
                .get("anthropic-beta")
                .and_then(|value| value.to_str().ok())
                .is_none_or(|value| !value.contains("oauth-2025-04-20"))
        );
        assert_eq!(
            payload["system"],
            json!([{ "type": "text", "text": "System prompt." }])
        );
    }

    #[test]
    fn explicit_authorization_overrides_provider_auth_token() {
        let runtime = AnthropicLanguageModelApi {
            api_key: None,
            auth_token: Some("provider-token".to_string()),
            http_client: None,
        };
        let mut options = StreamOptions::default();
        options
            .headers
            .insert("Authorization", Some("Bearer explicit-token".to_string()));

        let options = runtime.with_api_key(options);

        assert_eq!(
            options.headers.get("Authorization"),
            Some(&Some("Bearer explicit-token".to_string()))
        );
    }

    #[test]
    fn prices_one_hour_cache_writes_and_reports_reasoning_usage() {
        let mut model = anthropic_model("claude-opus-4-8");
        model.cost = ModelCost {
            input: 5.0,
            output: 25.0,
            cache_read: 0.5,
            cache_write: 6.25,
            ..Default::default()
        };
        let mut output = AssistantMessage::empty_for(&model);

        update_anthropic_usage(
            &mut output,
            &json!({
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 1_000_000,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 600_000,
                    "ephemeral_1h_input_tokens": 400_000
                },
                "output_tokens_details": { "thinking_tokens": 40 }
            }),
            &model,
        );

        assert_eq!(output.usage.cache_write, 1_000_000);
        assert_eq!(output.usage.cache_write_1h, Some(400_000));
        assert_eq!(output.usage.reasoning, Some(40));
        assert_eq!(output.usage.output, 50);
        assert_eq!(output.usage.cost.cache_write, 7.75);
    }

    #[test]
    fn cache_write_pricing_falls_back_when_anthropic_omits_breakdown() {
        let mut model = anthropic_model("claude-opus-4-8");
        model.cost = ModelCost {
            input: 5.0,
            cache_write: 6.25,
            ..Default::default()
        };
        let mut output = AssistantMessage::empty_for(&model);

        update_anthropic_start_usage(
            &mut output,
            &json!({ "cache_creation_input_tokens": 1_000_000 }),
            &model,
        );

        assert_eq!(output.usage.cache_write_1h, Some(0));
        assert_eq!(output.usage.cost.cache_write, 6.25);
    }

    #[tokio::test]
    async fn anthropic_stream_reports_reasoning_and_one_hour_cache_write_usage() {
        let body = sse_body(&[
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_usage",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 1_000_000,
                            "cache_creation": {
                                "ephemeral_5m_input_tokens": 600_000,
                                "ephemeral_1h_input_tokens": 400_000
                            }
                        }
                    }
                })
                .to_string(),
            ),
            (
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                })
                .to_string(),
            ),
            (
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "Hi" }
                })
                .to_string(),
            ),
            (
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }).to_string(),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn" },
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 1_000_000,
                        "output_tokens_details": { "thinking_tokens": 40 }
                    }
                })
                .to_string(),
            ),
            (
                "message_stop",
                json!({ "type": "message_stop" }).to_string(),
            ),
        ]);
        let mut model = anthropic_model("claude-opus-4-8");
        model.base_url = spawn_sse_server(body).await;
        model.cost = ModelCost {
            input: 5.0,
            output: 25.0,
            cache_read: 0.5,
            cache_write: 6.25,
            ..Default::default()
        };

        let result = crate::stream::final_message_from_stream(stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("hi")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        ))
        .await
        .unwrap();

        assert_eq!(result.usage.output, 50);
        assert_eq!(result.usage.reasoning, Some(40));
        assert_eq!(result.usage.cache_write, 1_000_000);
        assert_eq!(result.usage.cache_write_1h, Some(400_000));
        assert_eq!(result.usage.cost.cache_write, 7.75);
    }

    #[tokio::test]
    async fn preserves_refusal_stop_details_from_message_delta() {
        let explanation = "This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage Policy.";
        let body = sse_body(&[
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_refusal",
                        "usage": {
                            "input_tokens": 412,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0
                        }
                    }
                })
                .to_string(),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "refusal",
                        "stop_details": {
                            "type": "refusal",
                            "category": "cyber",
                            "explanation": explanation
                        }
                    },
                    "usage": {
                        "input_tokens": 412,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                })
                .to_string(),
            ),
            (
                "message_stop",
                json!({ "type": "message_stop" }).to_string(),
            ),
        ]);
        let mut model = anthropic_model("claude-fable-5");
        model.base_url = spawn_sse_server(body).await;

        let result = crate::stream::final_message_from_stream(stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("blocked request")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        ))
        .await
        .unwrap();

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some(explanation));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn model_carries_runtime_dispatch() {
        let anthropic = builder()
            .provider_id("test-anthropic-runtime")
            .build()
            .expect("provider");
        let mut model = anthropic.model("claude-test").build().expect("model");
        model.api = "not-registered".to_string();

        let error = match crate::stream_simple(model, Context::default(), None) {
            Ok(_) => panic!("missing API key should fail before stream creation"),
            Err(error) => error,
        };
        assert!(
            matches!(error, crate::Error::MissingApiKey(provider) if provider == "test-anthropic-runtime")
        );
    }

    #[test]
    fn provider_models_enable_pi_strict_tools() {
        let anthropic = builder().api_key("test-token").build().expect("provider");
        let model = anthropic.model("claude-opus-4-8").build().expect("model");

        assert_eq!(
            model.compat.anthropic_messages.supports_strict_tools,
            Some(true)
        );
    }

    #[test]
    fn should_handle_empty_content_array() {
        let model = anthropic_model("claude-sonnet-4-5");
        let messages = vec![crate::types::Message::User(crate::types::UserMessage {
            content: UserMessageContent::Parts(Vec::new()),
            timestamp: 0,
        })];

        let converted = convert_messages(&messages, &model, false, None, false);

        assert!(converted.is_empty());
    }

    #[test]
    fn should_handle_empty_string_content() {
        let model = anthropic_model("claude-sonnet-4-5");
        let messages = vec![crate::types::Message::user_text("")];

        let converted = convert_messages(&messages, &model, false, None, false);

        assert!(converted.is_empty());
    }

    #[test]
    fn should_handle_whitespace_only_content() {
        let model = anthropic_model("claude-sonnet-4-5");
        let messages = vec![crate::types::Message::user_text("   \n\t  ")];

        let converted = convert_messages(&messages, &model, false, None, false);

        assert!(converted.is_empty());
    }

    #[test]
    fn should_handle_empty_assistant_message_in_conversation() {
        let model = anthropic_model("claude-sonnet-4-5");
        let messages = vec![
            crate::types::Message::user_text("Hello, how are you?"),
            crate::types::Message::Assistant(AssistantMessage::empty_for(&model)),
            crate::types::Message::user_text("Please respond this time."),
        ];

        let converted = convert_messages(&messages, &model, false, None, false);
        let roles = converted
            .iter()
            .filter_map(|message| message.get("role").and_then(Value::as_str))
            .collect::<Vec<_>>();

        assert_eq!(roles, ["user", "user"]);
    }

    fn counting_on_response(calls: Arc<AtomicUsize>) -> ResponseHook {
        Arc::new(move |_response, _model| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        })
    }

    #[test]
    fn stream_simple_missing_api_key_names_provider() {
        let error = match stream_simple_anthropic(
            anthropic_model("claude-sonnet-4-5"),
            Context::default(),
            SimpleStreamOptions::default(),
        ) {
            Ok(_) => panic!("missing API key should fail before stream creation"),
            Err(error) => error,
        };

        assert!(matches!(error, crate::Error::MissingApiKey(provider) if provider == "anthropic"));
    }

    fn assistant_with_thinking(signature: &str) -> crate::types::Message {
        assistant_with_thinking_text(signature, "internal reasoning")
    }

    fn assistant_with_thinking_text(signature: &str, thinking: &str) -> crate::types::Message {
        crate::types::Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::Thinking(ThinkingContent {
                thinking: thinking.to_string(),
                thinking_signature: Some(signature.to_string()),
                redacted: None,
            })],
            api: "anthropic-messages".to_string(),
            provider: "xiaomi-token-plan-ams".to_string(),
            model: "mimo-v2.5-pro".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: crate::utils::time::now_millis(),
        })
    }

    #[test]
    fn converts_empty_signature_thinking_to_text_by_default() {
        let mut model = anthropic_model("mimo-v2.5-pro");
        model.provider = "xiaomi-token-plan-ams".to_string();
        let messages = vec![
            crate::types::Message::user_text("first"),
            assistant_with_thinking(""),
            crate::types::Message::user_text("second"),
        ];

        let converted = convert_messages(&messages, &model, false, None, false);
        let assistant = converted
            .iter()
            .find(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
            .expect("assistant message");

        assert_eq!(
            assistant["content"],
            json!([{ "type": "text", "text": "internal reasoning" }])
        );
    }

    #[test]
    fn preserves_empty_signature_thinking_when_allow_empty_signature_is_enabled() {
        let mut model = anthropic_model("mimo-v2.5-pro");
        model.provider = "xiaomi-token-plan-ams".to_string();
        model.compat.anthropic_messages.allow_empty_signature = Some(true);
        let messages = vec![
            crate::types::Message::user_text("first"),
            assistant_with_thinking(" "),
            crate::types::Message::user_text("second"),
        ];

        let converted = convert_messages(
            &messages,
            &model,
            false,
            None,
            model
                .compat
                .anthropic_messages
                .allow_empty_signature
                .unwrap_or(false),
        );
        let assistant = converted
            .iter()
            .find(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
            .expect("assistant message");

        assert_eq!(
            assistant["content"],
            json!([{ "type": "thinking", "thinking": "internal reasoning", "signature": "" }])
        );
    }

    #[test]
    fn preserves_empty_thinking_text_when_signature_is_present() {
        let mut model = anthropic_model("mimo-v2.5-pro");
        model.provider = "xiaomi-token-plan-ams".to_string();
        let messages = vec![
            crate::types::Message::user_text("first"),
            assistant_with_thinking_text("signed-thinking", ""),
            crate::types::Message::user_text("second"),
        ];

        let converted = convert_messages(&messages, &model, false, None, false);
        let assistant = converted
            .iter()
            .find(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
            .expect("assistant message");

        assert_eq!(
            assistant["content"],
            json!([{ "type": "thinking", "thinking": "", "signature": "signed-thinking" }])
        );
    }

    #[test]
    fn empty_user_text_is_filtered_for_anthropic_messages() {
        let mut model = anthropic_model("claude-sonnet-4-5");
        model.input.push(ModelInput::Image);
        let messages = vec![
            crate::types::Message::user_text(""),
            crate::types::Message::User(crate::types::UserMessage {
                content: crate::types::UserMessageContent::Parts(vec![
                    crate::types::UserContent::text(""),
                    crate::types::UserContent::Image(crate::types::ImageContent {
                        data: "abc".to_string(),
                        mime_type: "image/png".to_string(),
                    }),
                ]),
                timestamp: crate::utils::time::now_millis(),
            }),
        ];

        let converted = convert_messages(&messages, &model, false, None, false);

        assert_eq!(
            converted,
            vec![json!({
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc"
                    }
                }]
            })]
        );
    }

    #[test]
    fn text_only_tool_result_content_uses_concatenated_string() {
        let model = anthropic_model("claude-haiku-4-5");
        let converted = convert_messages(
            &[crate::types::Message::ToolResult(
                crate::types::ToolResultMessage {
                    tool_call_id: "toolu_123".to_string(),
                    tool_name: "lookup".to_string(),
                    content: vec![
                        ToolResultContent::text("first"),
                        ToolResultContent::text("second"),
                    ],
                    details: None,
                    is_error: false,
                    timestamp: 1,
                },
            )],
            &model,
            false,
            None,
            false,
        );

        assert_eq!(
            converted[0]["content"][0]["content"],
            json!("first\nsecond")
        );
    }

    #[test]
    fn sends_thinking_type_disabled_for_budget_based_reasoning_models_when_thinking_is_off() {
        let model = anthropic_model("claude-sonnet-4-5");
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(false),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
        assert!(payload.get("output_config").is_none());
    }

    #[test]
    fn disables_thinking_for_claude_reasoning_models() {
        let model = anthropic_model("claude-sonnet-4-5");
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(false),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
        assert!(payload.get("output_config").is_none());
    }

    #[test]
    fn sends_thinking_type_disabled_for_adaptive_reasoning_models_when_thinking_is_off() {
        let mut model = anthropic_model("claude-opus-4-8");
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(false),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
        assert!(payload.get("output_config").is_none());
    }

    #[test]
    fn sends_thinking_type_disabled_for_claude_opus_4_8_when_thinking_is_off() {
        let mut model = anthropic_model("claude-opus-4-8");
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(false),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
        assert!(payload.get("output_config").is_none());
    }

    #[test]
    fn string_tool_choice_is_wrapped_for_anthropic_payload() {
        let model = anthropic_model("claude-sonnet-4-5");
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("Use lookup")],
                tools: vec![Tool {
                    name: "lookup".to_string(),
                    description: "Look up a value".to_string(),
                    parameters: json!({ "type": "object" }),
                    constrained_sampling: None,
                }],
                ..Default::default()
            },
            &AnthropicOptions {
                tool_choice: Some(json!("any")),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(payload["tool_choice"], json!({ "type": "any" }));
    }

    #[test]
    fn object_tool_choice_passes_through_for_anthropic_payload() {
        let model = anthropic_model("claude-sonnet-4-5");
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("Use lookup")],
                tools: vec![Tool {
                    name: "lookup".to_string(),
                    description: "Look up a value".to_string(),
                    parameters: json!({ "type": "object" }),
                    constrained_sampling: None,
                }],
                ..Default::default()
            },
            &AnthropicOptions {
                tool_choice: Some(json!({ "type": "tool", "name": "lookup" })),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(
            payload["tool_choice"],
            json!({ "type": "tool", "name": "lookup" })
        );
    }

    #[test]
    fn metadata_includes_string_user_id_only_for_anthropic_payload() {
        let model = anthropic_model("claude-sonnet-4-5");
        let context = Context {
            messages: vec![crate::types::Message::user_text("hello")],
            ..Default::default()
        };
        let payload = build_anthropic_payload(
            &model,
            &context,
            &AnthropicOptions {
                base: StreamOptions {
                    metadata: Some(json!({ "user_id": "user-123", "other": "ignored" })),
                    ..Default::default()
                },
                ..Default::default()
            },
            false,
            None,
        );
        let invalid_payload = build_anthropic_payload(
            &model,
            &context,
            &AnthropicOptions {
                base: StreamOptions {
                    metadata: Some(json!({ "user_id": 123 })),
                    ..Default::default()
                },
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(payload["metadata"], json!({ "user_id": "user-123" }));
        assert!(invalid_payload.get("metadata").is_none());
    }

    #[test]
    fn temperature_is_omitted_when_anthropic_thinking_is_enabled() {
        let model = anthropic_model("claude-sonnet-4-5");
        let context = Context {
            messages: vec![crate::types::Message::user_text("hello")],
            ..Default::default()
        };
        let with_thinking = build_anthropic_payload(
            &model,
            &context,
            &AnthropicOptions {
                base: StreamOptions {
                    temperature: Some(0.2),
                    ..Default::default()
                },
                thinking_enabled: Some(true),
                ..Default::default()
            },
            false,
            None,
        );
        let without_thinking = build_anthropic_payload(
            &model,
            &context,
            &AnthropicOptions {
                base: StreamOptions {
                    temperature: Some(0.2),
                    ..Default::default()
                },
                thinking_enabled: Some(false),
                ..Default::default()
            },
            false,
            None,
        );

        assert!(with_thinking.get("temperature").is_none());
        assert_eq!(without_thinking["temperature"], json!(0.2));
    }

    #[test]
    fn temperature_is_omitted_when_compat_disables_it() {
        let mut model = anthropic_model("vendor--claude-opus-4-7");
        model.compat.anthropic_messages.supports_temperature = Some(false);
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                base: StreamOptions {
                    temperature: Some(0.0),
                    ..Default::default()
                },
                thinking_enabled: Some(false),
                ..Default::default()
            },
            false,
            None,
        );

        assert!(payload.get("temperature").is_none());
    }

    #[test]
    fn uses_adaptive_thinking_for_claude_opus_4_8_when_reasoning_is_enabled() {
        let mut model = anthropic_model("claude-opus-4-8");
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(true),
                effort: Some(AnthropicEffort::Xhigh),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(
            payload["thinking"],
            json!({ "type": "adaptive", "display": "summarized" })
        );
        assert_eq!(payload["output_config"], json!({ "effort": "xhigh" }));
    }

    #[test]
    fn sends_legacy_thinking_payload_for_custom_model_ids_by_default() {
        let mut model = anthropic_model("vendor--claude-opus-latest");
        model.provider = "vendor-proxy".to_string();
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(true),
                effort: Some(AnthropicEffort::Medium),
                thinking_budget_tokens: Some(2048),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(
            payload["thinking"],
            json!({ "type": "enabled", "budget_tokens": 2048, "display": "summarized" })
        );
        assert!(payload.get("output_config").is_none());
    }

    #[test]
    fn sends_adaptive_thinking_payload_when_compat_force_adaptive_thinking_is_true() {
        let mut model = anthropic_model("vendor--claude-opus-latest");
        model.provider = "vendor-proxy".to_string();
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(true),
                effort: Some(AnthropicEffort::Medium),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(
            payload["thinking"],
            json!({ "type": "adaptive", "display": "summarized" })
        );
        assert_eq!(payload["output_config"], json!({ "effort": "medium" }));
    }

    #[test]
    fn preserves_thinking_type_disabled_when_reasoning_is_off_regardless_of_override() {
        let mut model = anthropic_model("vendor--claude-opus-latest");
        model.provider = "vendor-proxy".to_string();
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(false),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
        assert!(payload.get("output_config").is_none());
    }

    #[test]
    fn allows_built_in_adaptive_models_to_opt_out_with_compat_force_adaptive_thinking_false() {
        let mut model = anthropic_model("claude-opus-4-8");
        model.compat.anthropic_messages.force_adaptive_thinking = Some(false);
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(true),
                effort: Some(AnthropicEffort::Medium),
                thinking_budget_tokens: Some(2048),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(
            payload["thinking"],
            json!({ "type": "enabled", "budget_tokens": 2048, "display": "summarized" })
        );
        assert!(payload.get("output_config").is_none());
    }

    #[test]
    fn maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_4_8() {
        let mut model = anthropic_model("claude-opus-4-8");
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        model
            .thinking_level_map
            .insert("xhigh".to_string(), Some("xhigh".to_string()));
        assert_eq!(
            model.compat.anthropic_messages.force_adaptive_thinking,
            Some(true)
        );
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(true),
                effort: Some(map_thinking_level_to_effort(
                    &model,
                    ModelThinkingLevel::Xhigh,
                )),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(
            payload["thinking"],
            json!({ "type": "adaptive", "display": "summarized" })
        );
        assert_eq!(payload["output_config"], json!({ "effort": "xhigh" }));
    }

    #[test]
    fn maps_max_reasoning_to_effort_max_for_adaptive_models() {
        let mut model = anthropic_model("vendor--claude-opus-latest");
        model.provider = "vendor-proxy".to_string();
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        model
            .thinking_level_map
            .insert("max".to_string(), Some("max".to_string()));
        let payload = build_anthropic_payload(
            &model,
            &Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions {
                thinking_enabled: Some(true),
                effort: Some(map_thinking_level_to_effort(
                    &model,
                    ModelThinkingLevel::Max,
                )),
                ..Default::default()
            },
            false,
            None,
        );

        assert_eq!(payload["output_config"], json!({ "effort": "max" }));
    }

    #[test]
    fn unmapped_extended_reasoning_levels_fall_back_to_high() {
        let model = anthropic_model("vendor--claude-opus-latest");

        assert_eq!(
            map_thinking_level_to_effort(&model, ModelThinkingLevel::Xhigh),
            AnthropicEffort::High
        );
        assert_eq!(
            map_thinking_level_to_effort(&model, ModelThinkingLevel::Max),
            AnthropicEffort::High
        );
    }

    #[test]
    fn cache_control_retention_uses_long_ttl_only_when_supported() {
        let model = anthropic_model("claude-haiku-4-5");
        assert_eq!(
            cache_control(&model, CacheRetention::Short, get_anthropic_compat(&model),),
            Some(json!({ "type": "ephemeral" }))
        );
        assert_eq!(
            cache_control(&model, CacheRetention::Long, get_anthropic_compat(&model)),
            Some(json!({ "type": "ephemeral", "ttl": "1h" }))
        );
        assert_eq!(
            cache_control(&model, CacheRetention::None, get_anthropic_compat(&model)),
            None
        );

        let mut unsupported = model.clone();
        unsupported
            .compat
            .anthropic_messages
            .supports_long_cache_retention = Some(false);
        assert_eq!(
            cache_control(
                &unsupported,
                CacheRetention::Long,
                get_anthropic_compat(&unsupported),
            ),
            Some(json!({ "type": "ephemeral" }))
        );
    }

    #[test]
    fn payload_applies_cache_control_to_system_and_last_user_text() {
        let model = anthropic_model("claude-haiku-4-5");
        let payload = build_anthropic_payload(
            &model,
            &Context {
                system_prompt: Some("You are helpful.".to_string()),
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions::default(),
            false,
            Some(json!({ "type": "ephemeral" })),
        );

        assert_eq!(
            payload["system"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(
            payload["messages"][0]["content"][0],
            json!({
                "type": "text",
                "text": "hello",
                "cache_control": { "type": "ephemeral" }
            })
        );
    }

    #[test]
    fn payload_skips_empty_system_prompt() {
        let model = anthropic_model("claude-haiku-4-5");
        let payload = build_anthropic_payload(
            &model,
            &Context {
                system_prompt: Some(String::new()),
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions::default(),
            false,
            Some(json!({ "type": "ephemeral" })),
        );

        assert!(payload.get("system").is_none());
    }

    #[test]
    fn payload_omits_cache_control_when_cache_retention_is_none() {
        let model = anthropic_model("claude-haiku-4-5");
        let payload = build_anthropic_payload(
            &model,
            &Context {
                system_prompt: Some("You are helpful.".to_string()),
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &AnthropicOptions::default(),
            false,
            None,
        );

        assert!(payload["system"][0].get("cache_control").is_none());
        assert!(
            payload["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[test]
    fn session_affinity_headers_follow_cache_retention_and_overrides() {
        let mut model = anthropic_model("claude-haiku-4-5");
        model
            .compat
            .anthropic_messages
            .send_session_affinity_headers = Some(true);
        let context = Context {
            messages: vec![crate::types::Message::user_text("hello")],
            ..Default::default()
        };
        let mut options = AnthropicOptions::default();
        options.base.session_id = Some("session-123".to_string());
        let compat = get_anthropic_compat(&model);

        let short_headers = headers(
            &model,
            &context,
            &options,
            Some("test-key"),
            false,
            compat,
            CacheRetention::Short,
        )
        .unwrap();
        assert_eq!(
            short_headers
                .get("x-session-affinity")
                .and_then(|value| value.to_str().ok()),
            Some("session-123")
        );

        let none_headers = headers(
            &model,
            &context,
            &options,
            Some("test-key"),
            false,
            compat,
            CacheRetention::None,
        )
        .unwrap();
        assert!(none_headers.get("x-session-affinity").is_none());

        options
            .base
            .headers
            .insert("x-session-affinity".to_string(), "override".to_string());
        let override_headers = headers(
            &model,
            &context,
            &options,
            Some("test-key"),
            false,
            compat,
            CacheRetention::Short,
        )
        .unwrap();
        assert_eq!(
            override_headers
                .get("x-session-affinity")
                .and_then(|value| value.to_str().ok()),
            Some("override")
        );
    }

    #[test]
    fn provider_name_does_not_enable_out_of_scope_anthropic_compat() {
        let mut model = anthropic_model("claude-haiku-4-5");
        model.provider = "fireworks".to_string();
        let compat = get_anthropic_compat(&model);

        assert!(compat.supports_eager_tool_input_streaming);
        assert!(compat.supports_long_cache_retention);
        assert!(!compat.send_session_affinity_headers);
        assert!(compat.supports_cache_control_on_tools);
    }

    #[test]
    fn tool_cache_control_respects_tool_support_compat() {
        let model = anthropic_model("claude-haiku-4-5");
        let mut second_tool = lookup_tool();
        second_tool.name = "lookup_again".to_string();
        let context = Context {
            messages: vec![crate::types::Message::user_text("hello")],
            tools: vec![lookup_tool(), second_tool],
            ..Default::default()
        };
        let payload = build_anthropic_payload(
            &model,
            &context,
            &AnthropicOptions::default(),
            false,
            Some(json!({ "type": "ephemeral" })),
        );

        assert!(
            payload["tools"][0].get("cache_control").is_none(),
            "only the last tool should receive a cache marker"
        );
        assert_eq!(
            payload["tools"][1]["cache_control"],
            json!({ "type": "ephemeral" })
        );

        let mut unsupported = model;
        unsupported
            .compat
            .anthropic_messages
            .supports_cache_control_on_tools = Some(false);
        let unsupported_payload = build_anthropic_payload(
            &unsupported,
            &context,
            &AnthropicOptions::default(),
            false,
            Some(json!({ "type": "ephemeral" })),
        );

        assert!(
            unsupported_payload["tools"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            unsupported_payload["tools"][1]
                .get("cache_control")
                .is_none()
        );
    }

    fn lookup_tool() -> Tool {
        Tool {
            name: "lookup".to_string(),
            description: "Look up a value".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            }),
            constrained_sampling: None,
        }
    }

    #[test]
    fn strict_tools_preserve_the_full_schema() {
        use crate::types::{
            ConstrainedSampling, ConstrainedSamplingConfig, ConstrainedSamplingStrict,
        };

        let mut model = anthropic_model("claude-haiku-4-5");
        model.compat.anthropic_messages.supports_strict_tools = Some(true);
        let context = Context {
            messages: vec![crate::types::Message::user_text("Use the tool")],
            tools: vec![Tool {
                name: "lookup".to_string(),
                description: "Look up a value".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"],
                    "additionalProperties": false
                }),
                constrained_sampling: Some(ConstrainedSampling::Config(
                    ConstrainedSamplingConfig::JsonSchema {
                        strict: ConstrainedSamplingStrict::Prefer,
                    },
                )),
            }],
            ..Default::default()
        };
        let payload =
            build_anthropic_payload(&model, &context, &AnthropicOptions::default(), false, None);

        assert_eq!(payload["tools"][0]["strict"], json!(true));
        assert_eq!(
            payload["tools"][0]["input_schema"]["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn sends_per_tool_eager_input_streaming_by_default() {
        let mut model = anthropic_model("claude-opus-4-8");
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        let context = Context {
            messages: vec![crate::types::Message::user_text("Use the tool")],
            tools: vec![lookup_tool()],
            ..Default::default()
        };
        let payload =
            build_anthropic_payload(&model, &context, &AnthropicOptions::default(), false, None);
        let request_headers = headers(
            &model,
            &context,
            &AnthropicOptions::default(),
            Some("test-key"),
            false,
            get_anthropic_compat(&model),
            CacheRetention::None,
        )
        .unwrap();

        assert_eq!(payload["tools"][0]["eager_input_streaming"], json!(true));
        assert!(request_headers.get("anthropic-beta").is_none());
    }

    #[test]
    fn uses_the_legacy_fine_grained_tool_streaming_beta_when_eager_tool_input_streaming_is_disabled()
     {
        let mut model = anthropic_model("claude-opus-4-8");
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        model
            .compat
            .anthropic_messages
            .supports_eager_tool_input_streaming = Some(false);
        let context = Context {
            messages: vec![crate::types::Message::user_text("Use the tool")],
            tools: vec![lookup_tool()],
            ..Default::default()
        };
        let payload =
            build_anthropic_payload(&model, &context, &AnthropicOptions::default(), false, None);
        let request_headers = headers(
            &model,
            &context,
            &AnthropicOptions::default(),
            Some("test-key"),
            false,
            get_anthropic_compat(&model),
            CacheRetention::None,
        )
        .unwrap();

        assert!(payload["tools"][0].get("eager_input_streaming").is_none());
        assert_eq!(
            request_headers
                .get("anthropic-beta")
                .and_then(|value| value.to_str().ok()),
            Some(FINE_GRAINED_TOOL_STREAMING_BETA)
        );
    }

    #[test]
    fn does_not_send_the_legacy_fine_grained_tool_streaming_beta_when_there_are_no_tools() {
        let mut model = anthropic_model("claude-opus-4-8");
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        model
            .compat
            .anthropic_messages
            .supports_eager_tool_input_streaming = Some(false);
        let context = Context {
            messages: vec![crate::types::Message::user_text("No tools")],
            ..Default::default()
        };
        let request_headers = headers(
            &model,
            &context,
            &AnthropicOptions::default(),
            Some("test-key"),
            false,
            get_anthropic_compat(&model),
            CacheRetention::None,
        )
        .unwrap();

        assert!(request_headers.get("anthropic-beta").is_none());
    }

    #[test]
    fn uses_bearer_auth_copilot_headers_and_valid_anthropic_messages_payload() {
        let mut model = anthropic_model("claude-sonnet-4.6");
        model.provider = "github-copilot".to_string();
        model.headers.insert(
            "Copilot-Integration-Id".to_string(),
            "vscode-chat".to_string(),
        );
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        let context = Context {
            messages: vec![crate::types::Message::User(crate::types::UserMessage {
                content: UserMessageContent::Parts(vec![
                    UserContent::text("describe this"),
                    UserContent::Image(crate::types::ImageContent {
                        data: "abc".to_string(),
                        mime_type: "image/png".to_string(),
                    }),
                ]),
                timestamp: 1,
            })],
            tools: vec![lookup_tool()],
            ..Default::default()
        };
        let request_headers = headers(
            &model,
            &context,
            &AnthropicOptions::default(),
            Some("tid_copilot_session_test_token"),
            false,
            get_anthropic_compat(&model),
            CacheRetention::Short,
        )
        .unwrap();

        assert_eq!(
            request_headers
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer tid_copilot_session_test_token")
        );
        assert!(request_headers.get("x-api-key").is_none());
        assert_eq!(
            request_headers
                .get("x-initiator")
                .and_then(|value| value.to_str().ok()),
            Some("user")
        );
        assert_eq!(
            request_headers
                .get("openai-intent")
                .and_then(|value| value.to_str().ok()),
            Some("conversation-edits")
        );
        assert_eq!(
            request_headers
                .get("copilot-vision-request")
                .and_then(|value| value.to_str().ok()),
            Some("true")
        );
        assert_eq!(
            request_headers
                .get("copilot-integration-id")
                .and_then(|value| value.to_str().ok()),
            Some("vscode-chat")
        );
        assert!(request_headers.get("anthropic-beta").is_none());

        let payload =
            build_anthropic_payload(&model, &context, &AnthropicOptions::default(), false, None);
        assert_eq!(payload["model"], json!("claude-sonnet-4.6"));
        assert_eq!(payload["stream"], json!(true));
        assert_eq!(payload["max_tokens"], json!(model.max_tokens));
        assert!(payload["messages"].as_array().is_some());
    }

    #[test]
    fn omits_interleaved_thinking_beta_for_adaptive_thinking_models() {
        let mut model = anthropic_model("claude-sonnet-4.6");
        model.provider = "github-copilot".to_string();
        model.compat.anthropic_messages.force_adaptive_thinking = Some(true);
        let context = Context {
            messages: vec![crate::types::Message::user_text("Hello")],
            ..Default::default()
        };

        let request_headers = headers(
            &model,
            &context,
            &AnthropicOptions {
                interleaved_thinking: true,
                ..Default::default()
            },
            Some("tid_copilot_session_test_token"),
            false,
            get_anthropic_compat(&model),
            CacheRetention::Short,
        )
        .unwrap();

        assert!(
            request_headers
                .get("anthropic-beta")
                .and_then(|value| value.to_str().ok())
                .is_none_or(|value| !value.contains(INTERLEAVED_THINKING_BETA))
        );
    }

    #[test]
    fn oauth_tool_names_round_trip_by_case_insensitive_lookup_only() {
        let model = anthropic_model("claude-sonnet-4-6");
        let tools = vec![
            Tool {
                name: "todowrite".to_string(),
                description: "Write todos".to_string(),
                parameters: json!({ "type": "object", "properties": {}, "required": [] }),
                constrained_sampling: None,
            },
            Tool {
                name: "find".to_string(),
                description: "Find files".to_string(),
                parameters: json!({ "type": "object", "properties": {}, "required": [] }),
                constrained_sampling: None,
            },
        ];
        let context = Context {
            messages: vec![crate::types::Message::user_text("hello")],
            tools: tools.clone(),
            ..Default::default()
        };
        let payload =
            build_anthropic_payload(&model, &context, &AnthropicOptions::default(), true, None);
        let tool_names: Vec<_> = payload["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();

        assert_eq!(tool_names, vec!["TodoWrite", "find"]);
        assert_eq!(
            from_claude_code_name("TodoWrite", &tools, true),
            "todowrite"
        );
        assert_eq!(from_claude_code_name("find", &tools, true), "find");
        assert_eq!(from_claude_code_name("Glob", &tools, true), "Glob");
    }

    #[test]
    fn oauth_tool_names_preserve_builtin_original_names_and_custom_names() {
        let model = anthropic_model("claude-sonnet-4-6");
        let tools = ["read", "write", "edit", "bash", "my_custom_tool"]
            .into_iter()
            .map(|name| Tool {
                name: name.to_string(),
                description: format!("{name} tool"),
                parameters: json!({ "type": "object", "properties": {}, "required": [] }),
                constrained_sampling: None,
            })
            .collect::<Vec<_>>();
        let context = Context {
            messages: vec![crate::types::Message::user_text("hello")],
            tools: tools.clone(),
            ..Default::default()
        };
        let payload =
            build_anthropic_payload(&model, &context, &AnthropicOptions::default(), true, None);
        let tool_names: Vec<_> = payload["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();

        assert_eq!(
            tool_names,
            vec!["Read", "Write", "Edit", "Bash", "my_custom_tool"]
        );
        for (provider_name, original_name) in [
            ("Read", "read"),
            ("Write", "write"),
            ("Edit", "edit"),
            ("Bash", "bash"),
            ("my_custom_tool", "my_custom_tool"),
        ] {
            assert_eq!(
                from_claude_code_name(provider_name, &tools, true),
                original_name
            );
        }
        assert_eq!(from_claude_code_name("Read", &tools, false), "Read");
    }

    fn sse_body(events: &[(&str, String)]) -> String {
        events
            .iter()
            .map(|(event, data)| format!("event: {event}\ndata: {data}\n"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn successful_anthropic_sse_body() -> String {
        sse_body(&[
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_retry",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0
                        }
                    }
                })
                .to_string(),
            ),
            (
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                })
                .to_string(),
            ),
            (
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "ok" }
                })
                .to_string(),
            ),
            (
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }).to_string(),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn" },
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                })
                .to_string(),
            ),
            (
                "message_stop",
                json!({ "type": "message_stop" }).to_string(),
            ),
        ])
    }

    async fn spawn_sse_server(body: String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 4096];
            let _ = socket.read(&mut buffer).await.unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_request_capture_server(
        body: String,
    ) -> (String, tokio::sync::oneshot::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (request_tx, request_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 16 * 1024];
            let read = socket.read(&mut buffer).await.unwrap();
            let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
            let _ = request_tx.send(request);
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        (format!("http://{addr}"), request_rx)
    }

    async fn spawn_retrying_sse_server(body: String, attempts: Arc<AtomicUsize>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                let mut buffer = vec![0u8; 4096];
                let _ = socket.read(&mut buffer).await.unwrap();
                let response = if attempt == 0 {
                    "HTTP/1.1 500 Internal Server Error\r\nretry-after-ms: 0\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_string()
                } else {
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                };
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }

    async fn spawn_status_server(status: u16, reason: &str, body: String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let reason = reason.to_string();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 4096];
            let _ = socket.read(&mut buffer).await.unwrap();
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn spawn_hanging_sse_server(body: String) -> (String, Arc<tokio::sync::Notify>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let release = Arc::new(tokio::sync::Notify::new());
        let release_task = Arc::clone(&release);
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 4096];
            let _ = socket.read(&mut buffer).await.unwrap();
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: keep-alive\r\n\r\n",
                )
                .await
                .unwrap();
            socket.write_all(body.as_bytes()).await.unwrap();
            socket.write_all(b"\n").await.unwrap();
            socket.flush().await.unwrap();
            release_task.notified().await;
        });
        (format!("http://{addr}"), release)
    }

    #[tokio::test]
    async fn repairs_malformed_sse_json_and_malformed_streamed_tool_json() {
        let body = sse_body(&[
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0
                        }
                    }
                })
                .to_string(),
            ),
            (
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_test",
                        "name": "edit",
                        "input": {}
                    }
                })
                .to_string(),
            ),
            (
                "content_block_delta",
                String::from(
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1	col2\"}"}}"#,
                ),
            ),
            (
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }).to_string(),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "tool_use" },
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                })
                .to_string(),
            ),
            (
                "message_stop",
                json!({ "type": "message_stop" }).to_string(),
            ),
        ]);
        let base_url = spawn_sse_server(body).await;
        let mut model = anthropic_model("claude-haiku-4-5");
        model.base_url = base_url;
        model.reasoning = false;

        let stream = stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("Use edit")],
                tools: vec![Tool {
                    name: "edit".to_string(),
                    description: "Edit a file".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" },
                            "text": { "type": "string" }
                        },
                        "required": ["path", "text"]
                    }),
                    constrained_sampling: None,
                }],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();
        assert_eq!(result.stop_reason, StopReason::ToolUse, "{result:#?}");

        let tool_call = result
            .content
            .iter()
            .find_map(|block| match block {
                AssistantContent::ToolCall(tool_call) => Some(tool_call),
                _ => None,
            })
            .expect("tool call");
        assert_eq!(
            tool_call.arguments,
            json!({ "path": "A\\H", "text": "col1\tcol2" })
        );
    }

    #[tokio::test]
    async fn response_id_is_exposed_from_message_start() {
        let body = sse_body(&[
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_response_id",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0
                        }
                    }
                })
                .to_string(),
            ),
            (
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                })
                .to_string(),
            ),
            (
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "Hello" }
                })
                .to_string(),
            ),
            (
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }).to_string(),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn" },
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                })
                .to_string(),
            ),
            (
                "message_stop",
                json!({ "type": "message_stop" }).to_string(),
            ),
        ]);
        let base_url = spawn_sse_server(body).await;
        let mut model = anthropic_model("claude-haiku-4-5");
        model.base_url = base_url;
        model.reasoning = false;

        let stream = stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("Say hello.")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.response_id.as_deref(), Some("msg_response_id"));
    }

    #[tokio::test]
    async fn unknown_stop_reason_reports_provider_reason() {
        let body = sse_body(&[
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_unknown_stop",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0
                        }
                    }
                })
                .to_string(),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "model_overheated" },
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                })
                .to_string(),
            ),
        ]);
        let base_url = spawn_sse_server(body).await;
        let mut model = anthropic_model("claude-haiku-4-5");
        model.base_url = base_url;
        model.reasoning = false;

        let stream = stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("Say hello.")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Unhandled stop reason: model_overheated")
        );
    }

    #[tokio::test]
    async fn anthropic_provider_skips_on_response_for_api_errors() {
        let response_calls = Arc::new(AtomicUsize::new(0));
        let base_url = spawn_status_server(
            500,
            "Internal Server Error",
            json!({ "error": { "message": "upstream unavailable" } }).to_string(),
        )
        .await;
        let mut model = anthropic_model("claude-haiku-4-5");
        model.base_url = base_url;
        model.reasoning = false;

        let stream = stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    on_response: Some(counting_on_response(Arc::clone(&response_calls))),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(response_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn anthropic_provider_does_not_retry_by_default() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let base_url =
            spawn_retrying_sse_server(successful_anthropic_sse_body(), Arc::clone(&attempts)).await;
        let mut model = anthropic_model("claude-haiku-4-5");
        model.base_url = base_url;
        model.reasoning = false;

        let stream = stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(
            result
                .error_message
                .as_deref()
                .is_some_and(|message| message.contains("500"))
        );
    }

    #[tokio::test]
    async fn anthropic_provider_honors_explicit_retry_settings() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let base_url =
            spawn_retrying_sse_server(successful_anthropic_sse_body(), Arc::clone(&attempts)).await;
        let mut model = anthropic_model("claude-haiku-4-5");
        model.base_url = base_url;
        model.reasoning = false;

        let stream = stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    max_retries: Some(1),
                    max_retry_delay_ms: Some(0),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.response_id.as_deref(), Some("msg_retry"));
    }

    #[tokio::test]
    async fn should_handle_immediate_abort() {
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        cancellation_token.cancel();
        let stream = stream_anthropic(
            anthropic_model("claude-haiku-4-5"),
            Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    cancellation_token: Some(cancellation_token),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Aborted);
        assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
        assert!(result.content.is_empty());
    }

    #[tokio::test]
    async fn should_abort_mid_stream() {
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        let (base_url, release_server) = spawn_hanging_sse_server(sse_body(&[
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_abort",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0
                        }
                    }
                })
                .to_string(),
            ),
            (
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                })
                .to_string(),
            ),
            (
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "partial" }
                })
                .to_string(),
            ),
        ]))
        .await;
        let mut model = anthropic_model("claude-haiku-4-5");
        model.base_url = base_url;
        model.reasoning = false;
        let mut stream = stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    cancellation_token: Some(cancellation_token.clone()),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let mut result = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event");
            match event {
                AssistantMessageEvent::TextDelta { .. } => {
                    cancellation_token.cancel();
                }
                AssistantMessageEvent::Done { message, .. } => result = Some(message),
                AssistantMessageEvent::Error { error, .. } => result = Some(error),
                _ => {}
            }
        }
        let result = result.expect("final message");
        release_server.notify_waiters();

        assert_eq!(result.stop_reason, StopReason::Aborted);
        assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
        assert_eq!(
            result.content,
            vec![AssistantContent::Text(TextContent {
                text: "partial".to_string(),
                text_signature: None,
            })]
        );
    }

    #[tokio::test]
    async fn ignores_unknown_sse_events_after_message_stop() {
        let body = sse_body(&[
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0
                        }
                    }
                })
                .to_string(),
            ),
            (
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                })
                .to_string(),
            ),
            (
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "Hello" }
                })
                .to_string(),
            ),
            (
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }).to_string(),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn" },
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0
                    }
                })
                .to_string(),
            ),
            (
                "message_stop",
                json!({ "type": "message_stop" }).to_string(),
            ),
            ("done", "[DONE]".to_string()),
            ("proxy.stats", "not json".to_string()),
        ]);
        let base_url = spawn_sse_server(body).await;
        let mut model = anthropic_model("claude-haiku-4-5");
        model.base_url = base_url;
        model.reasoning = false;

        let stream = stream_anthropic(
            model,
            Context {
                messages: vec![crate::types::Message::user_text("Say hello")],
                ..Default::default()
            },
            AnthropicOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();
        assert_eq!(result.stop_reason, StopReason::Stop, "{result:#?}");
        assert_eq!(
            result.content,
            vec![AssistantContent::Text(TextContent {
                text: "Hello".to_string(),
                text_signature: None,
            })]
        );
    }
}
