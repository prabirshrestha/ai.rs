use std::collections::{HashMap, HashSet};
use std::time::Duration;

use futures::{StreamExt, pin_mut};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::event_stream::AssistantMessageEventStreamSender;
use crate::models::{calculate_cost, clamp_thinking_level};
use crate::providers::cloudflare::{is_cloudflare_provider, resolve_cloudflare_base_url};
use crate::providers::github_copilot_headers::{
    build_copilot_dynamic_headers, has_copilot_vision_input,
};
use crate::providers::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::providers::simple_options::build_base_options;
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, CacheRetention, Context,
    ImageContent, Model, ModelInput, ModelThinkingLevel, SimpleStreamOptions, StopReason,
    StreamOptions, TextContent, TextPhase, TextSignatureV1, ThinkingContent, Tool, ToolCall,
    ToolResultContent, Usage, UserContent, UserMessageContent,
};
use crate::utils::hash::short_hash;
use crate::utils::json::parse_streaming_json;
use crate::utils::sanitize::sanitize_surrogates;
use crate::utils::sse;
use crate::utils::transform_messages::transform_messages;
use crate::{Error, Result};

const OPENAI_TOOL_CALL_PROVIDERS: &[&str] = &[
    "openai",
    "openai-codex",
    "opencode",
    "azure-openai-responses",
];

#[derive(Clone, Default)]
pub struct OpenAIResponsesOptions {
    pub base: StreamOptions,
    pub reasoning_effort: Option<ModelThinkingLevel>,
    pub reasoning_summary: Option<Option<String>>,
    pub service_tier: Option<String>,
    pub request_url: Option<String>,
    pub request_model: Option<String>,
    pub payload_override: Option<Value>,
    pub include_store: Option<bool>,
    pub auth_header: OpenAIResponsesAuthHeader,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAIResponsesAuthHeader {
    #[default]
    Bearer,
    ApiKey,
}

#[derive(Debug, Clone, Copy)]
pub struct ResolvedOpenAIResponsesCompat {
    pub send_session_id_header: bool,
    pub supports_long_cache_retention: bool,
}

pub fn stream_simple_openai_responses(
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
    let reasoning_effort = options.reasoning.and_then(|reasoning| {
        let clamped = clamp_thinking_level(&model, reasoning);
        (clamped != ModelThinkingLevel::Off).then_some(clamped)
    });
    stream_openai_responses(
        model,
        context,
        OpenAIResponsesOptions {
            base,
            reasoning_effort,
            reasoning_summary: None,
            service_tier: None,
            ..Default::default()
        },
    )
}

pub fn stream_openai_responses(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions,
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
    options: OpenAIResponsesOptions,
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
    let compat = get_compat(&model);
    let cache_retention = resolve_cache_retention(options.base.cache_retention);
    let mut payload = options.payload_override.clone().unwrap_or_else(|| {
        build_responses_payload(&model, &context, &options, &compat, cache_retention)
    });
    if let Some(on_payload) = &options.base.on_payload {
        match on_payload(payload.clone(), &model).await {
            Ok(Some(next)) => payload = next,
            Ok(None) => {}
            Err(error) => return Err(StreamFailure::new(output, error)),
        }
    }

    let request_url = if let Some(request_url) = options.request_url.clone() {
        request_url
    } else {
        match request_base_url(&model) {
            Ok(base_url) => format!("{}/responses", trim_end_slash(&base_url)),
            Err(error) => return Err(StreamFailure::new(output, error)),
        }
    };
    let request_headers = match headers(
        &model,
        &context,
        &options.base,
        &api_key,
        &compat,
        cache_retention,
        options.auth_header,
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

    let mut current_item: Option<Value> = None;
    let mut current_block: Option<usize> = None;
    let mut current_text_part: Option<String> = None;
    let mut partial_json: HashMap<usize, String> = HashMap::new();
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
        if event.data.trim().is_empty() || event.data.trim() == "[DONE]" {
            continue;
        }
        let parsed: Value = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(error) => return Err(StreamFailure::new(output, error)),
        };
        let event_type = parsed
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref())
            .unwrap_or_default();
        match event_type {
            "response.created" => {
                if let Some(id) = parsed.pointer("/response/id").and_then(Value::as_str) {
                    output.response_id = Some(id.to_string());
                }
            }
            "response.output_item.added" => {
                let Some(item) = parsed.get("item") else {
                    continue;
                };
                current_item = Some(item.clone());
                match item.get("type").and_then(Value::as_str) {
                    Some("reasoning") => {
                        output
                            .content
                            .push(AssistantContent::Thinking(ThinkingContent {
                                thinking: String::new(),
                                thinking_signature: None,
                                redacted: None,
                            }));
                        let index = output.content.len() - 1;
                        current_block = Some(index);
                        sender.push(AssistantMessageEvent::ThinkingStart {
                            content_index: index,
                            partial: output.clone(),
                        });
                    }
                    Some("message") => {
                        current_text_part = item
                            .get("content")
                            .and_then(Value::as_array)
                            .and_then(|parts| parts.last())
                            .and_then(response_text_part_type);
                        output.content.push(AssistantContent::Text(TextContent {
                            text: String::new(),
                            text_signature: None,
                        }));
                        let index = output.content.len() - 1;
                        current_block = Some(index);
                        sender.push(AssistantMessageEvent::TextStart {
                            content_index: index,
                            partial: output.clone(),
                        });
                    }
                    Some("function_call") => {
                        let index = output.content.len();
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let item_id = item.get("id").and_then(Value::as_str).unwrap_or_default();
                        let args = item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        output.content.push(AssistantContent::ToolCall(ToolCall {
                            id: format!("{call_id}|{item_id}"),
                            name: item
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            arguments: parse_streaming_json(Some(args)),
                            thought_signature: None,
                        }));
                        partial_json.insert(index, args.to_string());
                        current_block = Some(index);
                        sender.push(AssistantMessageEvent::ToolCallStart {
                            content_index: index,
                            partial: output.clone(),
                        });
                    }
                    _ => {}
                }
            }
            "response.content_part.added" => {
                if current_item
                    .as_ref()
                    .and_then(|item| item.get("type"))
                    .and_then(Value::as_str)
                    == Some("message")
                {
                    if let Some(part_type) = parsed.get("part").and_then(response_text_part_type) {
                        current_text_part = Some(part_type);
                    }
                }
            }
            "response.reasoning_summary_part.added" => {
                if let Some(item) = current_item.as_mut() {
                    item["summary"].as_array_mut().map(|summary| {
                        if let Some(part) = parsed.get("part") {
                            summary.push(part.clone());
                        }
                    });
                    if item.get("summary").is_none() {
                        item["summary"] =
                            json!([parsed.get("part").cloned().unwrap_or(Value::Null)]);
                    }
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let delta = parsed
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(index) = current_block {
                    if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(index) {
                        block.thinking.push_str(delta);
                        sender.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: index,
                            delta: delta.to_string(),
                            partial: output.clone(),
                        });
                    }
                }
            }
            "response.reasoning_summary_part.done" => {
                if let Some(index) = current_block {
                    if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(index) {
                        block.thinking.push_str("\n\n");
                        sender.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: index,
                            delta: "\n\n".to_string(),
                            partial: output.clone(),
                        });
                    }
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let expected_part = if event_type == "response.output_text.delta" {
                    "output_text"
                } else {
                    "refusal"
                };
                if current_text_part.as_deref() != Some(expected_part) {
                    continue;
                }
                let delta = parsed
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(index) = current_block {
                    if let Some(AssistantContent::Text(block)) = output.content.get_mut(index) {
                        block.text.push_str(delta);
                        sender.push(AssistantMessageEvent::TextDelta {
                            content_index: index,
                            delta: delta.to_string(),
                            partial: output.clone(),
                        });
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                let delta = parsed
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(index) = current_block {
                    let entry = partial_json.entry(index).or_default();
                    entry.push_str(delta);
                    if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(index) {
                        block.arguments = parse_streaming_json(Some(entry));
                        sender.push(AssistantMessageEvent::ToolCallDelta {
                            content_index: index,
                            delta: delta.to_string(),
                            partial: output.clone(),
                        });
                    }
                }
            }
            "response.function_call_arguments.done" => {
                let arguments = parsed
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(index) = current_block {
                    let previous = partial_json
                        .insert(index, arguments.to_string())
                        .unwrap_or_default();
                    if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(index) {
                        block.arguments = parse_streaming_json(Some(arguments));
                        if let Some(delta) =
                            arguments.strip_prefix(&previous).filter(|s| !s.is_empty())
                        {
                            sender.push(AssistantMessageEvent::ToolCallDelta {
                                content_index: index,
                                delta: delta.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                }
            }
            "response.output_item.done" => {
                let item = parsed.get("item").cloned().unwrap_or_default();
                if let Some(index) = current_block {
                    match item.get("type").and_then(Value::as_str) {
                        Some("reasoning") => {
                            if let Some(AssistantContent::Thinking(block)) =
                                output.content.get_mut(index)
                            {
                                let summary_text = item
                                    .get("summary")
                                    .and_then(Value::as_array)
                                    .map(|parts| {
                                        parts
                                            .iter()
                                            .filter_map(|part| {
                                                part.get("text").and_then(Value::as_str)
                                            })
                                            .collect::<Vec<_>>()
                                            .join("\n\n")
                                    })
                                    .unwrap_or_default();
                                let content_text = item
                                    .get("content")
                                    .and_then(Value::as_array)
                                    .map(|parts| {
                                        parts
                                            .iter()
                                            .filter_map(|part| {
                                                part.get("text").and_then(Value::as_str)
                                            })
                                            .collect::<Vec<_>>()
                                            .join("\n\n")
                                    })
                                    .unwrap_or_default();
                                if !summary_text.is_empty() || !content_text.is_empty() {
                                    block.thinking = if summary_text.is_empty() {
                                        content_text
                                    } else {
                                        summary_text
                                    };
                                }
                                block.thinking_signature = Some(item.to_string());
                                sender.push(AssistantMessageEvent::ThinkingEnd {
                                    content_index: index,
                                    content: block.thinking.clone(),
                                    partial: output.clone(),
                                });
                            }
                        }
                        Some("message") => {
                            if let Some(AssistantContent::Text(block)) =
                                output.content.get_mut(index)
                            {
                                let text = item
                                    .get("content")
                                    .and_then(Value::as_array)
                                    .map(|parts| {
                                        parts
                                            .iter()
                                            .filter_map(|part| {
                                                part.get("text")
                                                    .or_else(|| part.get("refusal"))
                                                    .and_then(Value::as_str)
                                            })
                                            .collect::<String>()
                                    })
                                    .unwrap_or_else(|| block.text.clone());
                                block.text = text;
                                if let Some(id) = item.get("id").and_then(Value::as_str) {
                                    let phase = match item.get("phase").and_then(Value::as_str) {
                                        Some("commentary") => Some(TextPhase::Commentary),
                                        Some("final_answer") => Some(TextPhase::FinalAnswer),
                                        _ => None,
                                    };
                                    block.text_signature =
                                        serde_json::to_string(&TextSignatureV1 {
                                            v: 1,
                                            id: id.to_string(),
                                            phase,
                                        })
                                        .ok();
                                }
                                sender.push(AssistantMessageEvent::TextEnd {
                                    content_index: index,
                                    content: block.text.clone(),
                                    partial: output.clone(),
                                });
                            }
                        }
                        Some("function_call") => {
                            if let Some(AssistantContent::ToolCall(block)) =
                                output.content.get_mut(index)
                            {
                                let args = partial_json
                                    .get(&index)
                                    .map(String::as_str)
                                    .or_else(|| item.get("arguments").and_then(Value::as_str))
                                    .unwrap_or("{}");
                                block.arguments = parse_streaming_json(Some(args));
                                sender.push(AssistantMessageEvent::ToolCallEnd {
                                    content_index: index,
                                    tool_call: block.clone(),
                                    partial: output.clone(),
                                });
                            }
                        }
                        _ => {}
                    }
                }
                current_block = None;
                current_item = None;
                current_text_part = None;
            }
            "response.completed" | "response.done" | "response.incomplete" => {
                let response = parsed.get("response").unwrap_or(&parsed);
                if let Some(id) = response.get("id").and_then(Value::as_str) {
                    output.response_id = Some(id.to_string());
                }
                if let Some(usage) = response.get("usage") {
                    output.usage = parse_response_usage(usage, &model);
                    if let Some(service_tier) = response
                        .get("service_tier")
                        .and_then(Value::as_str)
                        .or(options.service_tier.as_deref())
                    {
                        apply_service_tier_pricing(&mut output.usage, service_tier, &model);
                    }
                }
                output.stop_reason = map_status(response.get("status").and_then(Value::as_str));
                if output
                    .content
                    .iter()
                    .any(|block| matches!(block, AssistantContent::ToolCall(_)))
                    && output.stop_reason == StopReason::Stop
                {
                    output.stop_reason = StopReason::ToolUse;
                }
            }
            "error" => {
                let code = parsed
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let message = parsed
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Unknown error");
                return Err(StreamFailure::new(
                    output,
                    format!("Error Code {code}: {message}"),
                ));
            }
            "response.failed" => {
                let response = parsed.get("response").unwrap_or(&parsed);
                let message = response
                    .get("error")
                    .map(|error| {
                        format!(
                            "{}: {}",
                            error
                                .get("code")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown"),
                            error
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("no message")
                        )
                    })
                    .or_else(|| {
                        response
                            .pointer("/incomplete_details/reason")
                            .and_then(Value::as_str)
                            .map(|reason| format!("incomplete: {reason}"))
                    })
                    .unwrap_or_else(|| "Unknown error (no error details in response)".to_string());
                return Err(StreamFailure::new(output, message));
            }
            _ => {}
        }
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

pub fn build_responses_payload(
    model: &Model,
    context: &Context,
    options: &OpenAIResponsesOptions,
    compat: &ResolvedOpenAIResponsesCompat,
    cache_retention: CacheRetention,
) -> Value {
    let messages = convert_responses_messages(
        model,
        context,
        &OPENAI_TOOL_CALL_PROVIDERS.iter().copied().collect(),
        true,
    );
    let mut payload = json!({
        "model": options.request_model.as_deref().unwrap_or(&model.id),
        "input": messages,
        "stream": true
    });
    let object = payload.as_object_mut().expect("payload object");
    if options.include_store.unwrap_or(true) {
        object.insert("store".to_string(), json!(false));
    }
    if let Some(max_tokens) = options.base.max_tokens {
        object.insert("max_output_tokens".to_string(), json!(max_tokens));
    }
    if let Some(temperature) = options.base.temperature {
        object.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(service_tier) = &options.service_tier {
        object.insert("service_tier".to_string(), json!(service_tier));
    }
    if !context.tools.is_empty() {
        object.insert(
            "tools".to_string(),
            Value::Array(convert_responses_tools(&context.tools, Some(false))),
        );
    }
    if cache_retention != CacheRetention::None {
        if let Some(session_id) = &options.base.session_id {
            object.insert(
                "prompt_cache_key".to_string(),
                json!(clamp_openai_prompt_cache_key(Some(session_id))),
            );
        }
    }
    if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
        object.insert("prompt_cache_retention".to_string(), json!("24h"));
    }
    if model.reasoning {
        if options.reasoning_effort.is_some() || options.reasoning_summary.is_some() {
            let effort = options
                .reasoning_effort
                .and_then(|effort| {
                    model
                        .thinking_level_map
                        .get(effort.as_str())
                        .cloned()
                        .flatten()
                })
                .or_else(|| {
                    options
                        .reasoning_effort
                        .map(|effort| effort.as_str().to_string())
                })
                .unwrap_or_else(|| "medium".to_string());
            let summary = options
                .reasoning_summary
                .clone()
                .flatten()
                .unwrap_or_else(|| "auto".to_string());
            object.insert(
                "reasoning".to_string(),
                json!({ "effort": effort, "summary": summary }),
            );
            object.insert(
                "include".to_string(),
                json!(["reasoning.encrypted_content"]),
            );
        } else if model.provider != "github-copilot"
            && model.thinking_level_map.get("off") != Some(&None)
        {
            object.insert(
                "reasoning".to_string(),
                json!({ "effort": model.thinking_level_map.get("off").and_then(Clone::clone).unwrap_or_else(|| "none".to_string()) }),
            );
        }
    }
    payload
}

pub fn convert_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &HashSet<&str>,
    include_system_prompt: bool,
) -> Vec<Value> {
    let mut messages = Vec::new();
    let transformed = transform_messages(
        context.messages.as_slice(),
        model,
        |id, target_model, source| {
            normalize_responses_tool_call_id(id, target_model, source, allowed_tool_call_providers)
        },
    );
    if include_system_prompt {
        if let Some(system_prompt) = &context.system_prompt {
            messages.push(json!({
                "role": if model.reasoning { "developer" } else { "system" },
                "content": sanitize_surrogates(system_prompt),
            }));
        }
    }

    let mut msg_index = 0usize;
    for msg in transformed {
        match msg {
            crate::types::Message::User(user) => match user.content {
                UserMessageContent::Text(text) => messages.push(json!({
                    "role": "user",
                    "content": [{ "type": "input_text", "text": sanitize_surrogates(&text) }]
                })),
                UserMessageContent::Parts(parts) => {
                    let content: Vec<Value> = parts
                        .iter()
                        .map(|item| match item {
                            UserContent::Text(text) => {
                                json!({ "type": "input_text", "text": sanitize_surrogates(&text.text) })
                            }
                            UserContent::Image(image) => json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!("data:{};base64,{}", image.mime_type, image.data)
                            }),
                        })
                        .collect();
                    if content.is_empty() {
                        continue;
                    }
                    messages.push(json!({ "role": "user", "content": content }));
                }
            },
            crate::types::Message::Assistant(assistant) => {
                let mut output = Vec::new();
                let is_different_model = assistant.model != model.id
                    && assistant.provider == model.provider
                    && assistant.api == model.api;
                let mut text_block_index = 0usize;
                for block in assistant.content {
                    match block {
                        AssistantContent::Thinking(thinking) => {
                            if let Some(signature) = thinking.thinking_signature {
                                if let Ok(reasoning_item) =
                                    serde_json::from_str::<Value>(&signature)
                                {
                                    output.push(reasoning_item);
                                }
                            }
                        }
                        AssistantContent::Text(text) => {
                            let parsed_signature =
                                parse_text_signature(text.text_signature.as_deref());
                            let fallback_id = if text_block_index == 0 {
                                format!("msg_pi_{msg_index}")
                            } else {
                                format!("msg_pi_{msg_index}_{text_block_index}")
                            };
                            text_block_index += 1;
                            let msg_id = parsed_signature
                                .as_ref()
                                .map(|sig| sig.0.clone())
                                .unwrap_or(fallback_id);
                            let msg_id = if msg_id.len() > 64 {
                                format!("msg_{}", short_hash(&msg_id))
                            } else {
                                msg_id
                            };
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{ "type": "output_text", "text": sanitize_surrogates(&text.text), "annotations": [] }],
                                "status": "completed",
                                "id": msg_id,
                            });
                            if let Some((_, Some(phase))) = parsed_signature {
                                item["phase"] = json!(match phase {
                                    TextPhase::Commentary => "commentary",
                                    TextPhase::FinalAnswer => "final_answer",
                                });
                            }
                            output.push(item);
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let (call_id, item_id_raw) = tool_call
                                .id
                                .split_once('|')
                                .map(|(call_id, item_id)| {
                                    (call_id.to_string(), Some(item_id.to_string()))
                                })
                                .unwrap_or_else(|| (tool_call.id.clone(), None));
                            let item_id = if is_different_model
                                && item_id_raw
                                    .as_deref()
                                    .is_some_and(|id| id.starts_with("fc_"))
                            {
                                None
                            } else {
                                item_id_raw
                            };
                            output.push(json!({
                                "type": "function_call",
                                "id": item_id,
                                "call_id": call_id,
                                "name": tool_call.name,
                                "arguments": serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string())
                            }));
                        }
                    }
                }
                if output.is_empty() {
                    continue;
                }
                messages.extend(output);
            }
            crate::types::Message::ToolResult(tool_result) => {
                let text_result = tool_result
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        ToolResultContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_images = tool_result
                    .content
                    .iter()
                    .any(|content| matches!(content, ToolResultContent::Image(_)));
                let (call_id, _) = tool_result
                    .tool_call_id
                    .split_once('|')
                    .unwrap_or((&tool_result.tool_call_id, ""));
                let output = if has_images && model.input.contains(&ModelInput::Image) {
                    let mut content = Vec::new();
                    if !text_result.is_empty() {
                        content.push(json!({ "type": "input_text", "text": sanitize_surrogates(&text_result) }));
                    }
                    for block in tool_result.content {
                        if let ToolResultContent::Image(ImageContent { data, mime_type }) = block {
                            content.push(json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!("data:{mime_type};base64,{data}")
                            }));
                        }
                    }
                    Value::Array(content)
                } else {
                    json!(sanitize_surrogates(if text_result.is_empty() {
                        "(see attached image)"
                    } else {
                        &text_result
                    }))
                };
                messages.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output
                }));
            }
        }
        msg_index += 1;
    }
    messages
}

pub fn convert_responses_tools(tools: &[Tool], strict: Option<bool>) -> Vec<Value> {
    let strict = strict.unwrap_or(false);
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": strict
            })
        })
        .collect()
}

fn normalize_responses_tool_call_id(
    id: &str,
    model: &Model,
    source: &AssistantMessage,
    allowed_tool_call_providers: &HashSet<&str>,
) -> String {
    let normalize_id_part = |part: &str| {
        let sanitized: String = part
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .collect();
        sanitized
            .chars()
            .take(64)
            .collect::<String>()
            .trim_end_matches('_')
            .to_string()
    };
    if !allowed_tool_call_providers.contains(model.provider.as_str()) {
        return normalize_id_part(id);
    }
    if !id.contains('|') {
        return normalize_id_part(id);
    }
    let (call_id, item_id) = id.split_once('|').unwrap_or((id, ""));
    let normalized_call_id = normalize_id_part(call_id);
    let is_foreign = source.provider != model.provider || source.api != model.api;
    let mut normalized_item_id = if is_foreign {
        format!("fc_{}", short_hash(item_id))
    } else {
        normalize_id_part(item_id)
    };
    if !normalized_item_id.starts_with("fc_") {
        normalized_item_id = normalize_id_part(&format!("fc_{normalized_item_id}"));
    }
    format!("{normalized_call_id}|{normalized_item_id}")
}

fn parse_text_signature(signature: Option<&str>) -> Option<(String, Option<TextPhase>)> {
    let signature = signature?;
    if signature.starts_with('{') {
        if let Ok(parsed) = serde_json::from_str::<TextSignatureV1>(signature) {
            return Some((parsed.id, parsed.phase));
        }
    }
    Some((signature.to_string(), None))
}

fn response_text_part_type(part: &Value) -> Option<String> {
    part.get("type")
        .and_then(Value::as_str)
        .filter(|part_type| matches!(*part_type, "output_text" | "refusal"))
        .map(ToString::to_string)
}

fn parse_response_usage(raw: &Value, model: &Model) -> Usage {
    let input_tokens = raw.get("input_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
    let output_tokens = raw
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let cached_tokens = raw
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let mut usage = Usage {
        input: input_tokens.saturating_sub(cached_tokens),
        output: output_tokens,
        cache_read: cached_tokens,
        cache_write: 0,
        total_tokens: raw.get("total_tokens").and_then(Value::as_u64).unwrap_or(0) as u32,
        cost: Default::default(),
    };
    calculate_cost(model, &mut usage);
    usage
}

fn map_status(status: Option<&str>) -> StopReason {
    match status {
        Some("completed") | None => StopReason::Stop,
        Some("incomplete") => StopReason::Length,
        Some("failed") | Some("cancelled") => StopReason::Error,
        Some("in_progress") | Some("queued") => StopReason::Stop,
        Some(_) => StopReason::Error,
    }
}

pub(crate) fn get_compat(model: &Model) -> ResolvedOpenAIResponsesCompat {
    ResolvedOpenAIResponsesCompat {
        send_session_id_header: model
            .compat
            .openai_responses
            .send_session_id_header
            .unwrap_or(true),
        supports_long_cache_retention: model
            .compat
            .openai_responses
            .supports_long_cache_retention
            .unwrap_or(true),
    }
}

fn resolve_cache_retention(cache_retention: Option<CacheRetention>) -> CacheRetention {
    cache_retention
        .or_else(|| {
            (std::env::var("PI_CACHE_RETENTION").ok().as_deref() == Some("long"))
                .then_some(CacheRetention::Long)
        })
        .unwrap_or(CacheRetention::Short)
}

fn headers(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    api_key: &str,
    compat: &ResolvedOpenAIResponsesCompat,
    cache_retention: CacheRetention,
    auth_header: OpenAIResponsesAuthHeader,
) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let is_cloudflare_ai_gateway = model.provider == "cloudflare-ai-gateway";
    if !api_key.is_empty() && !is_cloudflare_ai_gateway {
        match auth_header {
            OpenAIResponsesAuthHeader::Bearer => {
                headers.insert(
                    AUTHORIZATION,
                    HeaderValue::from_str(&format!("Bearer {api_key}"))
                        .map_err(|e| Error::InvalidHeaderValue("authorization".to_string(), e))?,
                );
            }
            OpenAIResponsesAuthHeader::ApiKey => {
                headers.insert(
                    HeaderName::from_static("api-key"),
                    HeaderValue::from_str(api_key)
                        .map_err(|e| Error::InvalidHeaderValue("api-key".to_string(), e))?,
                );
            }
        }
    }
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
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
    if let Some(session_id) = &options.session_id {
        if cache_retention != CacheRetention::None {
            if compat.send_session_id_header {
                headers.insert(
                    HeaderName::from_static("session_id"),
                    HeaderValue::from_str(session_id)
                        .map_err(|e| Error::InvalidHeaderValue("session_id".to_string(), e))?,
                );
            }
            headers.insert(
                HeaderName::from_static("x-client-request-id"),
                HeaderValue::from_str(session_id)
                    .map_err(|e| Error::InvalidHeaderValue("x-client-request-id".to_string(), e))?,
            );
        }
    }
    for (name, value) in &options.headers {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let value = HeaderValue::from_str(value)
            .map_err(|e| Error::InvalidHeaderValue(name.to_string(), e))?;
        headers.insert(name, value);
    }
    if !api_key.is_empty() && is_cloudflare_ai_gateway {
        headers.insert(
            HeaderName::from_static("cf-aig-authorization"),
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| Error::InvalidHeaderValue("cf-aig-authorization".to_string(), e))?,
        );
    }
    Ok(headers)
}

fn response_headers(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
        .collect()
}

fn get_service_tier_multiplier(model: &Model, service_tier: &str) -> f64 {
    match service_tier {
        "flex" => 0.5,
        "priority" if model.id == "gpt-5.5" => 2.5,
        "priority" => 2.0,
        _ => 1.0,
    }
}

fn apply_service_tier_pricing(usage: &mut Usage, service_tier: &str, model: &Model) {
    let multiplier = get_service_tier_multiplier(model, service_tier);
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
    use super::*;
    use crate::types::{Message, ModelCost, ToolResultMessage};
    use futures::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const COPILOT_RAW_TOOL_CALL_ID: &str = "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA==";

    fn model() -> Model {
        Model {
            id: "gpt-5.5".to_string(),
            name: "GPT 5.5".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "http://localhost:4141/v1".to_string(),
            reasoning: true,
            input: vec![ModelInput::Text, ModelInput::Image],
            cost: ModelCost::default(),
            context_window: 1_000_000,
            max_tokens: 4096,
            ..Default::default()
        }
    }

    #[test]
    fn builds_responses_payload_with_reasoning() {
        let model = model();
        let context = Context {
            system_prompt: Some("sys".to_string()),
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
        };
        let options = OpenAIResponsesOptions {
            reasoning_effort: Some(ModelThinkingLevel::Low),
            ..Default::default()
        };
        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );
        assert_eq!(payload["input"][0]["role"], "developer");
        assert_eq!(payload["reasoning"]["effort"], "low");
        assert_eq!(payload["include"][0], "reasoning.encrypted_content");
    }

    #[test]
    fn response_headers_let_explicit_headers_override_session_affinity() {
        let model = model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
        };
        let mut options = StreamOptions {
            session_id: Some("session-123".to_string()),
            ..Default::default()
        };
        options
            .headers
            .insert("session_id".to_string(), "override-session".to_string());
        options.headers.insert(
            "x-client-request-id".to_string(),
            "override-request".to_string(),
        );

        let headers = headers(
            &model,
            &context,
            &options,
            "test-key",
            &get_compat(&model),
            CacheRetention::Short,
            OpenAIResponsesAuthHeader::Bearer,
        )
        .unwrap();

        assert_eq!(
            headers
                .get("session_id")
                .and_then(|value| value.to_str().ok()),
            Some("override-session")
        );
        assert_eq!(
            headers
                .get("x-client-request-id")
                .and_then(|value| value.to_str().ok()),
            Some("override-request")
        );
    }

    #[test]
    fn response_headers_use_cloudflare_ai_gateway_authorization() {
        let mut model = model();
        model.provider = "cloudflare-ai-gateway".to_string();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
        };

        let headers = headers(
            &model,
            &context,
            &StreamOptions::default(),
            "test-key",
            &get_compat(&model),
            CacheRetention::Short,
            OpenAIResponsesAuthHeader::Bearer,
        )
        .unwrap();

        assert!(headers.get(AUTHORIZATION).is_none());
        assert_eq!(
            headers
                .get("cf-aig-authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer test-key")
        );
    }

    #[test]
    fn response_headers_add_copilot_dynamic_headers() {
        let mut model = model();
        model.provider = "github-copilot".to_string();
        model.headers.insert(
            "Copilot-Integration-Id".to_string(),
            "vscode-chat".to_string(),
        );
        let context = Context {
            system_prompt: None,
            messages: vec![Message::User(crate::types::UserMessage {
                content: crate::types::UserMessageContent::Parts(vec![
                    crate::types::UserContent::text("describe this"),
                    crate::types::UserContent::Image(ImageContent {
                        data: "abc".to_string(),
                        mime_type: "image/png".to_string(),
                    }),
                ]),
                timestamp: 1,
            })],
            tools: Vec::new(),
        };
        let headers = headers(
            &model,
            &context,
            &StreamOptions::default(),
            "test-key",
            &get_compat(&model),
            CacheRetention::Short,
            OpenAIResponsesAuthHeader::Bearer,
        )
        .unwrap();

        assert_eq!(
            headers
                .get("x-initiator")
                .and_then(|value| value.to_str().ok()),
            Some("user")
        );
        assert_eq!(
            headers
                .get("openai-intent")
                .and_then(|value| value.to_str().ok()),
            Some("conversation-edits")
        );
        assert_eq!(
            headers
                .get("copilot-vision-request")
                .and_then(|value| value.to_str().ok()),
            Some("true")
        );
        assert_eq!(
            headers
                .get("copilot-integration-id")
                .and_then(|value| value.to_str().ok()),
            Some("vscode-chat")
        );
    }

    #[test]
    fn response_payload_omits_default_reasoning_for_copilot() {
        let mut model = model();
        model.provider = "github-copilot".to_string();
        let context = Context {
            system_prompt: Some("sys".to_string()),
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
        };
        let payload = build_responses_payload(
            &model,
            &context,
            &OpenAIResponsesOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("reasoning").is_none());
        assert!(payload.get("include").is_none());
    }

    #[tokio::test]
    async fn response_text_delta_ignores_unsupported_content_parts() {
        let body = sse_body(&[
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "content": []
                }
            }),
            json!({
                "type": "response.content_part.added",
                "part": { "type": "reasoning_text", "text": "" }
            }),
            json!({
                "type": "response.output_text.delta",
                "delta": "hidden"
            }),
            json!({
                "type": "response.content_part.added",
                "part": { "type": "output_text", "text": "" }
            }),
            json!({
                "type": "response.output_text.delta",
                "delta": "visible"
            }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "visible" }]
                }
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_test",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": { "cached_tokens": 0 }
                    }
                }
            }),
        ]);
        let base_url = spawn_sse_server(body).await;
        let mut model = model();
        model.base_url = base_url;

        let mut stream = stream_openai_responses(
            model,
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAIResponsesOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let mut deltas = Vec::new();
        while let Some(event) = stream.next().await {
            if let AssistantMessageEvent::TextDelta { delta, .. } = event {
                deltas.push(delta);
            }
        }
        let result = stream.result().await.unwrap();
        assert_eq!(deltas, vec!["visible"]);
        assert_eq!(
            result.content,
            vec![AssistantContent::Text(TextContent {
                text: "visible".to_string(),
                text_signature: Some(
                    serde_json::to_string(&TextSignatureV1 {
                        v: 1,
                        id: "msg_test".to_string(),
                        phase: None,
                    })
                    .unwrap()
                ),
            })]
        );
    }

    #[tokio::test]
    async fn response_service_tier_overrides_requested_tier_for_pricing() {
        let body = sse_body(&[json!({
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "status": "completed",
                "service_tier": "flex",
                "usage": {
                    "input_tokens": 1_000_000,
                    "output_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                    "input_tokens_details": { "cached_tokens": 0 }
                }
            }
        })]);
        let base_url = spawn_sse_server(body).await;
        let mut model = model();
        model.base_url = base_url;
        model.cost = ModelCost {
            input: 2.0,
            output: 4.0,
            cache_read: 0.0,
            cache_write: 0.0,
        };

        let mut stream = stream_openai_responses(
            model,
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAIResponsesOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                service_tier: Some("priority".to_string()),
                ..Default::default()
            },
        );
        while stream.next().await.is_some() {}
        let result = stream.result().await.unwrap();

        assert_eq!(result.usage.cost.input, 1.0);
        assert_eq!(result.usage.cost.output, 2.0);
        assert_eq!(result.usage.cost.total, 3.0);
    }

    #[test]
    fn generates_unique_fallback_message_ids_for_multiple_text_blocks() {
        let model = crate::get_model("openai-codex", "gpt-5.5").expect("gpt-5.5");
        let assistant = AssistantMessage {
            content: vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "private reasoning".to_string(),
                    thinking_signature: None,
                    redacted: None,
                }),
                AssistantContent::Text(TextContent {
                    text: "visible answer".to_string(),
                    text_signature: None,
                }),
            ],
            api: "anthropic-messages".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-opus-4-8".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 2,
        };
        let context = Context {
            system_prompt: Some("You are concise.".to_string()),
            messages: vec![Message::user_text("hello"), Message::Assistant(assistant)],
            tools: Vec::new(),
        };

        let input = convert_responses_messages(
            &model,
            &context,
            &["openai", "openai-codex", "opencode"].into_iter().collect(),
            true,
        );
        let message_ids = input
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
            .filter_map(|item| item.get("id").and_then(Value::as_str))
            .collect::<Vec<_>>();

        assert_eq!(message_ids, ["msg_pi_1", "msg_pi_1_1"]);
    }

    #[test]
    fn fallback_message_ids_skip_empty_converted_assistant_turns() {
        let model = model();
        let mut empty_assistant = AssistantMessage::empty_for(&model);
        empty_assistant
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: "transient reasoning".to_string(),
                thinking_signature: None,
                redacted: None,
            }));
        let mut text_assistant = AssistantMessage::empty_for(&model);
        text_assistant
            .content
            .push(AssistantContent::Text(TextContent {
                text: "visible".to_string(),
                text_signature: None,
            }));

        let messages = convert_responses_messages(
            &model,
            &Context {
                messages: vec![
                    Message::Assistant(empty_assistant),
                    Message::Assistant(text_assistant),
                ],
                ..Default::default()
            },
            &["openai", "openai-codex", "opencode"].into_iter().collect(),
            true,
        );

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["id"], json!("msg_pi_0"));
        assert_eq!(messages[0]["content"][0]["text"], json!("visible"));
    }

    #[test]
    fn hashes_foreign_tool_item_ids_for_responses_models() {
        let model = crate::get_model("openai-codex", "gpt-5.5").expect("gpt-5.5");
        let assistant = AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: COPILOT_RAW_TOOL_CALL_ID.to_string(),
                name: "edit".to_string(),
                arguments: json!({ "path": "src/styles/app.css" }),
                thought_signature: None,
            })],
            api: "openai-responses".to_string(),
            provider: "github-copilot".to_string(),
            model: "gpt-5.5".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 2,
        };
        let tool_result = ToolResultMessage {
            tool_call_id: COPILOT_RAW_TOOL_CALL_ID.to_string(),
            tool_name: "edit".to_string(),
            content: vec![ToolResultContent::text("ok")],
            details: None,
            is_error: false,
            timestamp: 3,
        };
        let context = Context {
            system_prompt: Some("You are concise.".to_string()),
            messages: vec![
                Message::user_text("Use the tool."),
                Message::Assistant(assistant),
                Message::ToolResult(tool_result),
            ],
            tools: Vec::new(),
        };

        let input = convert_responses_messages(
            &model,
            &context,
            &["openai", "openai-codex", "opencode"].into_iter().collect(),
            true,
        );
        let function_call = input
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .expect("function_call item");
        let item_id = function_call
            .get("id")
            .and_then(Value::as_str)
            .expect("function call item id");
        let raw_item_id = COPILOT_RAW_TOOL_CALL_ID
            .split_once('|')
            .expect("raw item id")
            .1;
        let expected_item_id = format!("fc_{}", crate::utils::hash::short_hash(raw_item_id));

        assert_eq!(item_id, expected_item_id);
        assert!(item_id.len() <= 64);
        assert!(item_id.starts_with("fc_"));
        assert!(
            item_id
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        );
    }

    #[test]
    fn tool_result_images_stay_inside_function_call_output() {
        let model = model();
        let tool_result = ToolResultMessage {
            tool_call_id: "call-1".to_string(),
            tool_name: "get_image".to_string(),
            content: vec![
                ToolResultContent::text("A red circle."),
                ToolResultContent::Image(ImageContent {
                    data: "iVBORw0KGgo=".to_string(),
                    mime_type: "image/png".to_string(),
                }),
            ],
            details: None,
            is_error: false,
            timestamp: 2,
        };
        let context = Context {
            system_prompt: None,
            messages: vec![Message::ToolResult(tool_result)],
            tools: Vec::new(),
        };

        let input = convert_responses_messages(
            &model,
            &context,
            &["openai", "openai-codex", "opencode"].into_iter().collect(),
            true,
        );
        let function_output = input
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output"))
            .expect("function_call_output item");
        let output = function_output
            .get("output")
            .and_then(Value::as_array)
            .expect("content array output");

        assert!(output.iter().any(|item| {
            item.get("type").and_then(Value::as_str) == Some("input_text")
                && item
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.contains("A red circle."))
        }));
        assert!(output.iter().any(|item| {
            item.get("type").and_then(Value::as_str) == Some("input_image")
                && item
                    .get("image_url")
                    .and_then(Value::as_str)
                    .is_some_and(|url| url.starts_with("data:image/png;base64,"))
        }));
    }

    fn sse_body(events: &[Value]) -> String {
        events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>()
    }

    async fn spawn_sse_server(body: String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 4096];
            let _ = socket.read(&mut buffer).await.unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{addr}")
    }
}
