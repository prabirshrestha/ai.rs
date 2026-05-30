use std::collections::HashMap;
use std::time::Duration;

use futures::{StreamExt, pin_mut};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::event_stream::AssistantMessageEventStreamSender;
use crate::models::calculate_cost;
use crate::providers::cloudflare::{is_cloudflare_provider, resolve_cloudflare_base_url};
use crate::providers::github_copilot_headers::{
    build_copilot_dynamic_headers, has_copilot_vision_input,
};
use crate::providers::simple_options::{
    adjust_max_tokens_for_thinking, build_base_options, clamped_reasoning,
};
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, CacheRetention, Context, Model,
    ModelThinkingLevel, SimpleStreamOptions, StopReason, StreamOptions, TextContent,
    ThinkingContent, Tool, ToolCall, ToolResultContent, UserContent, UserMessageContent,
};
use crate::utils::json::{parse_json_with_repair, parse_streaming_json};
use crate::utils::sanitize::sanitize_surrogates;
use crate::utils::sse;
use crate::utils::transform_messages::transform_messages;
use crate::{Error, Result};

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const CLAUDE_CODE_VERSION: &str = "2.1.75";

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
pub struct ResolvedAnthropicCompat {
    pub supports_eager_tool_input_streaming: bool,
    pub supports_long_cache_retention: bool,
    pub send_session_affinity_headers: bool,
    pub supports_cache_control_on_tools: bool,
    pub allow_empty_signature: bool,
}

pub fn stream_simple_anthropic(
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
) -> crate::AssistantMessageEventStream {
    let api_key = options
        .stream
        .api_key
        .clone()
        .filter(|key| !key.trim().is_empty());
    let Some(api_key) = api_key else {
        return immediate_error(model, "No API key for provider");
    };
    let base = build_base_options(&model, &options, api_key);
    let tool_choice = options.tool_choice.clone();

    let Some(reasoning) = clamped_reasoning(&model, &options) else {
        return stream_anthropic(
            model,
            context,
            AnthropicOptions {
                base,
                thinking_enabled: Some(false),
                tool_choice,
                ..Default::default()
            },
        );
    };

    if model.compat.anthropic_messages.force_adaptive_thinking == Some(true) {
        return stream_anthropic(
            model.clone(),
            context,
            AnthropicOptions {
                base,
                thinking_enabled: Some(true),
                effort: Some(map_thinking_level_to_effort(&model, reasoning)),
                tool_choice,
                ..Default::default()
            },
        );
    }

    let adjusted = adjust_max_tokens_for_thinking(
        base.max_tokens,
        model.max_tokens,
        Some(reasoning),
        options.thinking_budgets.as_ref(),
    );
    let mut adjusted_base = base;
    adjusted_base.max_tokens = adjusted.max_tokens;
    stream_anthropic(
        model,
        context,
        AnthropicOptions {
            base: adjusted_base,
            thinking_enabled: Some(true),
            thinking_budget_tokens: Some(adjusted.thinking_budget),
            tool_choice,
            ..Default::default()
        },
    )
}

pub fn stream_anthropic(
    model: Model,
    context: Context,
    options: AnthropicOptions,
) -> crate::AssistantMessageEventStream {
    let (mut sender, stream) = crate::AssistantMessageEventStream::channel();
    tokio::spawn(async move {
        let output = AssistantMessage::empty_for(&model);
        if let Err(error) = run_stream(model, context, options, output, &mut sender).await {
            let mut message = error.output;
            message.stop_reason = if error.cancelled {
                StopReason::Aborted
            } else {
                StopReason::Error
            };
            message.error_message = Some(error.message);
            sender.push(AssistantMessageEvent::Error {
                reason: message.stop_reason,
                error: message,
            });
        }
    });
    stream
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

    let Some(api_key) = options
        .base
        .api_key
        .clone()
        .filter(|key| !key.trim().is_empty())
    else {
        return Err(StreamFailure::new(
            output,
            format!("No API key for provider: {}", model.provider),
        ));
    };
    let is_oauth = is_oauth_token(&api_key);
    let compat = get_anthropic_compat(&model);
    let cache_retention = resolve_cache_retention(options.base.cache_retention);
    let cache_control = cache_control(&model, cache_retention, compat);
    let mut payload =
        build_anthropic_payload(&model, &context, &options, is_oauth, cache_control.clone());
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
        &api_key,
        is_oauth,
        compat,
        cache_retention,
    ) {
        Ok(headers) => headers,
        Err(error) => return Err(StreamFailure::new(output, error)),
    };
    let client = reqwest::Client::new();
    let response = match crate::utils::http::send_with_retries(&options.base, || {
        let mut request = client
            .post(request_url.as_str())
            .headers(request_headers.clone())
            .json(&payload);
        if let Some(timeout_ms) = options.base.timeout_ms {
            request = request.timeout(Duration::from_millis(timeout_ms));
        }
        request
    })
    .await
    {
        Ok(response) => response,
        Err(Error::Cancelled) => return Err(StreamFailure::cancelled(output)),
        Err(error) => return Err(StreamFailure::new(output, error)),
    };
    if let Some(on_response) = &options.base.on_response {
        let provider_response = crate::types::ProviderResponse {
            status: response.status().as_u16(),
            headers: response_headers(response.headers()),
        };
        if let Err(error) = on_response(provider_response, &model).await {
            return Err(StreamFailure::new(output, error));
        }
    }
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(StreamFailure::new(
            output,
            Error::ApiStatus { status, body },
        ));
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
                        "Could not parse Anthropic SSE event {:?}: {}; data={}; raw={}",
                        event.event,
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
                    update_anthropic_usage(&mut output, usage, &model);
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
                    output.stop_reason = map_stop_reason(reason);
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
        return Err(StreamFailure::new(output, "An unknown error occurred"));
    }
    sender.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output,
    });
    Ok(())
}

pub fn build_anthropic_payload(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
    is_oauth_token: bool,
    cache_control: Option<Value>,
) -> Value {
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
        if let Some(system_prompt) = &context.system_prompt {
            let mut item = json!({ "type": "text", "text": sanitize_surrogates(system_prompt) });
            if let Some(cache_control) = &cache_control {
                item["cache_control"] = cache_control.clone();
            }
            system.push(item);
        }
        object.insert("system".to_string(), Value::Array(system));
    } else if let Some(system_prompt) = &context.system_prompt {
        let mut item = json!({ "type": "text", "text": sanitize_surrogates(system_prompt) });
        if let Some(cache_control) = &cache_control {
            item["cache_control"] = cache_control.clone();
        }
        object.insert("system".to_string(), json!([item]));
    }

    if let Some(temperature) = options.base.temperature {
        if options.thinking_enabled != Some(true) {
            object.insert("temperature".to_string(), json!(temperature));
        }
    }
    if !context.tools.is_empty() {
        let compat = get_anthropic_compat(model);
        object.insert(
            "tools".to_string(),
            Value::Array(convert_tools(
                &context.tools,
                is_oauth_token,
                compat.supports_eager_tool_input_streaming,
                if compat.supports_cache_control_on_tools {
                    cache_control.clone()
                } else {
                    None
                },
            )),
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
    if let Some(metadata) = &options.base.metadata {
        if let Some(user_id) = metadata.get("user_id").and_then(Value::as_str) {
            object.insert("metadata".to_string(), json!({ "user_id": user_id }));
        }
    }
    if let Some(tool_choice) = &options.tool_choice {
        let value = tool_choice
            .as_str()
            .map(|choice| json!({ "type": choice }))
            .unwrap_or_else(|| tool_choice.clone());
        object.insert("tool_choice".to_string(), value);
    }
    payload
}

pub fn convert_messages(
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
                        params
                            .push(json!({ "role": "user", "content": sanitize_surrogates(text) }));
                    }
                }
                UserMessageContent::Parts(parts) => {
                    let blocks: Vec<Value> = parts
                        .iter()
                        .filter_map(|item| match item {
                            UserContent::Text(text) if !text.text.trim().is_empty() => Some(
                                json!({ "type": "text", "text": sanitize_surrogates(&text.text) }),
                            ),
                            UserContent::Text(_) => None,
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
                            blocks.push(json!({ "type": "text", "text": sanitize_surrogates(&text.text) }));
                        }
                        AssistantContent::Thinking(thinking) if thinking.redacted == Some(true) => {
                            if let Some(signature) = &thinking.thinking_signature {
                                blocks.push(json!({ "type": "redacted_thinking", "data": signature }));
                            }
                        }
                        AssistantContent::Thinking(thinking) if !thinking.thinking.trim().is_empty() => {
                            match thinking.thinking_signature.as_deref().filter(|s| !s.trim().is_empty()) {
                                Some(signature) => blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(&thinking.thinking),
                                    "signature": signature
                                })),
                                None if allow_empty_signature => blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(&thinking.thinking),
                                    "signature": ""
                                })),
                                None => blocks.push(json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&thinking.thinking)
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
        }
        index += 1;
    }

    if let Some(cache_control) = cache_control {
        if let Some(last) = params
            .last_mut()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        {
            match last.get_mut("content") {
                Some(Value::Array(blocks)) => {
                    if let Some(block) = blocks.last_mut() {
                        if matches!(
                            block.get("type").and_then(Value::as_str),
                            Some("text" | "image" | "tool_result")
                        ) {
                            block["cache_control"] = cache_control;
                        }
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
    }

    params
}

fn convert_content_blocks(content: &[ToolResultContent]) -> Value {
    let has_images = content
        .iter()
        .any(|content| matches!(content, ToolResultContent::Image(_)));
    if !has_images {
        return json!(
            content
                .iter()
                .filter_map(|content| match content {
                    ToolResultContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    let mut blocks: Vec<Value> = content
        .iter()
        .map(|block| match block {
            ToolResultContent::Text(text) => {
                json!({ "type": "text", "text": sanitize_surrogates(&text.text) })
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
    cache_control: Option<Value>,
) -> Vec<Value> {
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let mut value = json!({
                "name": if is_oauth_token { to_claude_code_name(&tool.name) } else { tool.name.clone() },
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": tool.parameters.get("properties").cloned().unwrap_or_else(|| json!({})),
                    "required": tool.parameters.get("required").cloned().unwrap_or_else(|| json!([]))
                }
            });
            if supports_eager_tool_input_streaming {
                value["eager_input_streaming"] = json!(true);
            }
            if index == tools.len() - 1 {
                if let Some(cache_control) = &cache_control {
                    value["cache_control"] = cache_control.clone();
                }
            }
            value
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
    output.usage.total_tokens = output.usage.input
        + output.usage.output
        + output.usage.cache_read
        + output.usage.cache_write;
    calculate_cost(model, &mut output.usage);
}

fn get_anthropic_compat(model: &Model) -> ResolvedAnthropicCompat {
    let is_fireworks = model.provider == "fireworks";
    let is_cloudflare_anthropic =
        model.provider == "cloudflare-ai-gateway" && model.base_url.contains("anthropic");
    let compat = &model.compat.anthropic_messages;
    ResolvedAnthropicCompat {
        supports_eager_tool_input_streaming: compat
            .supports_eager_tool_input_streaming
            .unwrap_or(!is_fireworks),
        supports_long_cache_retention: compat
            .supports_long_cache_retention
            .unwrap_or(!is_fireworks),
        send_session_affinity_headers: compat
            .send_session_affinity_headers
            .unwrap_or(is_fireworks || is_cloudflare_anthropic),
        supports_cache_control_on_tools: compat
            .supports_cache_control_on_tools
            .unwrap_or(!is_fireworks),
        allow_empty_signature: compat.allow_empty_signature.unwrap_or(false),
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
    cache_retention
        .or_else(|| {
            (std::env::var("PI_CACHE_RETENTION").ok().as_deref() == Some("long"))
                .then_some(CacheRetention::Long)
        })
        .unwrap_or(CacheRetention::Short)
}

fn should_use_fine_grained_tool_streaming_beta(model: &Model, context: &Context) -> bool {
    !context.tools.is_empty() && !get_anthropic_compat(model).supports_eager_tool_input_streaming
}

fn headers(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
    api_key: &str,
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

    if model.provider == "cloudflare-ai-gateway" {
        headers.insert(
            HeaderName::from_static("cf-aig-authorization"),
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| Error::InvalidHeaderValue("cf-aig-authorization".to_string(), e))?,
        );
    } else if model.provider == "github-copilot" || is_oauth {
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
    } else if !api_key.is_empty() {
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
    if let Some(session_id) = &options.base.session_id {
        if cache_retention != CacheRetention::None && compat.send_session_affinity_headers {
            headers.insert(
                HeaderName::from_static("x-session-affinity"),
                HeaderValue::from_str(session_id)
                    .map_err(|e| Error::InvalidHeaderValue("x-session-affinity".to_string(), e))?,
            );
        }
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
    for (name, value) in &options.base.headers {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let value = HeaderValue::from_str(value)
            .map_err(|e| Error::InvalidHeaderValue(name.to_string(), e))?;
        headers.insert(name, value);
    }
    Ok(headers)
}

fn response_headers(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
        .collect()
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

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "pause_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "refusal" | "sensitive" => StopReason::Error,
        _ => StopReason::Error,
    }
}

fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

fn trim_end_slash(url: &str) -> &str {
    url.trim_end_matches('/')
}

fn request_base_url(model: &Model) -> Result<String> {
    if is_cloudflare_provider(&model.provider) {
        resolve_cloudflare_base_url(model)
    } else {
        Ok(model.base_url.clone())
    }
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

fn immediate_error(model: Model, message: &str) -> crate::AssistantMessageEventStream {
    let (mut sender, stream) = crate::AssistantMessageEventStream::channel();
    let mut output = AssistantMessage::empty_for(&model);
    output.stop_reason = StopReason::Error;
    output.error_message = Some(format!("{message}: {}", model.provider));
    sender.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: output,
    });
    stream
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::types::{AssistantContent, CacheRetention, ModelCost, ModelInput, Usage};

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

    fn assistant_with_thinking(signature: &str) -> crate::types::Message {
        crate::types::Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::Thinking(ThinkingContent {
                thinking: "internal reasoning".to_string(),
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
    fn empty_thinking_signature_converts_to_text_by_default() {
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
    fn empty_thinking_signature_can_be_preserved_with_compat() {
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
    fn thinking_disabled_payload_for_reasoning_models() {
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
    fn adaptive_thinking_payload_uses_effort() {
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
    fn custom_model_ids_use_legacy_thinking_payload_by_default() {
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
    fn force_adaptive_thinking_enables_adaptive_payload_for_custom_model_ids() {
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
    fn force_adaptive_thinking_preserves_disabled_thinking_when_reasoning_is_off() {
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
    fn adaptive_thinking_can_be_disabled_by_compat_override() {
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
    fn built_in_adaptive_thinking_model_uses_xhigh_effort() {
        let model = crate::get_model("anthropic", "claude-opus-4-8").expect("claude-opus-4-8");
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
    fn payload_uses_pi_cache_retention_for_long_cache_control() {
        let _env = crate::test_env::EnvVarGuard::set("PI_CACHE_RETENTION", "long");
        let model = anthropic_model("claude-haiku-4-5");
        let options = AnthropicOptions::default();
        let cache_control = cache_control(
            &model,
            resolve_cache_retention(options.base.cache_retention),
            get_anthropic_compat(&model),
        );
        let payload = build_anthropic_payload(
            &model,
            &Context {
                system_prompt: Some("You are helpful.".to_string()),
                messages: vec![crate::types::Message::user_text("hello")],
                ..Default::default()
            },
            &options,
            false,
            cache_control,
        );

        assert_eq!(
            payload["system"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
        assert_eq!(
            payload["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
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
        }
    }

    #[test]
    fn eager_tool_input_streaming_is_enabled_by_default() {
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
            "test-key",
            false,
            get_anthropic_compat(&model),
            CacheRetention::None,
        )
        .unwrap();

        assert_eq!(payload["tools"][0]["eager_input_streaming"], json!(true));
        assert!(request_headers.get("anthropic-beta").is_none());
    }

    #[test]
    fn fine_grained_tool_streaming_beta_is_used_when_eager_input_is_disabled() {
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
            "test-key",
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
    fn fine_grained_tool_streaming_beta_is_omitted_without_tools() {
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
            "test-key",
            false,
            get_anthropic_compat(&model),
            CacheRetention::None,
        )
        .unwrap();

        assert!(request_headers.get("anthropic-beta").is_none());
    }

    #[test]
    fn copilot_anthropic_headers_use_bearer_auth_and_dynamic_headers() {
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
            "tid_copilot_session_test_token",
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
    }

    #[test]
    fn copilot_anthropic_omits_interleaved_thinking_beta_for_adaptive_models() {
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
            "tid_copilot_session_test_token",
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
            },
            Tool {
                name: "find".to_string(),
                description: "Find files".to_string(),
                parameters: json!({ "type": "object", "properties": {}, "required": [] }),
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

    fn sse_body(events: &[(&str, String)]) -> String {
        events
            .iter()
            .map(|(event, data)| format!("event: {event}\ndata: {data}\n"))
            .collect::<Vec<_>>()
            .join("\n")
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

    #[tokio::test]
    async fn repairs_malformed_sse_json_and_streamed_tool_json() {
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

        let mut stream = stream_anthropic(
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
        while stream.next().await.is_some() {}
        let result = stream.result().await.unwrap();
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

        let mut stream = stream_anthropic(
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
        while stream.next().await.is_some() {}
        let result = stream.result().await.unwrap();

        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.response_id.as_deref(), Some("msg_response_id"));
    }

    #[tokio::test]
    async fn anthropic_immediate_cancellation_returns_aborted_message() {
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        cancellation_token.cancel();
        let mut stream = stream_anthropic(
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

        let result = stream.result().await.unwrap();

        assert_eq!(result.stop_reason, StopReason::Aborted);
        assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
        assert!(result.content.is_empty());
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

        let mut stream = stream_anthropic(
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
        while stream.next().await.is_some() {}
        let result = stream.result().await.unwrap();
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
