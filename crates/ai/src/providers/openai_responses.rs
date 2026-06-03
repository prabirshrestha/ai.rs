use std::collections::{HashMap, HashSet};

use futures::{StreamExt, pin_mut};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::event_stream::AssistantMessageEventStreamSender;
use crate::models::{calculate_cost, clamp_thinking_level};
use crate::providers::github_copilot_headers::{
    build_copilot_dynamic_headers, has_copilot_vision_input,
};
use crate::providers::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::providers::simple_options::build_base_options;
use crate::providers::transform_messages::transform_messages;
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, CacheRetention, Context,
    ImageContent, Model, ModelInput, ModelThinkingLevel, SimpleStreamOptions, StopReason,
    StreamOptions, TextContent, TextPhase, TextSignatureV1, ThinkingContent, Tool, ToolCall,
    ToolResultContent, Usage, UserContent, UserMessageContent,
};
use crate::utils::hash::short_hash;
use crate::utils::http::{request_timeout, send_with_retries};
use crate::utils::json::parse_streaming_json;
use crate::utils::sse;
use crate::{Error, Result};

const OPENAI_TOOL_CALL_PROVIDERS: &[&str] = &["openai"];

#[derive(Clone, Default)]
pub struct OpenAIResponsesOptions {
    pub base: StreamOptions,
    pub reasoning_effort: Option<ModelThinkingLevel>,
    pub reasoning_summary: Option<Option<String>>,
    pub service_tier: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedOpenAIResponsesCompat {
    pub send_session_id_header: bool,
    pub supports_long_cache_retention: bool,
}

pub fn stream_simple_openai_responses(
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
) -> crate::Result<crate::AssistantEventStream> {
    let Some(api_key) = options.stream.api_key.clone() else {
        return Err(crate::Error::MissingApiKey(model.provider));
    };
    let base = build_base_options(&model, &options, api_key);
    let reasoning_effort = options.reasoning.and_then(|reasoning| {
        let clamped = clamp_thinking_level(&model, reasoning);
        (clamped != ModelThinkingLevel::Off).then_some(clamped)
    });
    Ok(stream_openai_responses(
        model,
        context,
        OpenAIResponsesOptions {
            base,
            reasoning_effort,
            reasoning_summary: None,
            service_tier: None,
        },
    ))
}

pub fn stream_openai_responses(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions,
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

    fn api_status(output: AssistantMessage, status: reqwest::StatusCode, body: String) -> Self {
        Self {
            output,
            message: format_openai_responses_api_error(status, &body),
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

    let Some(api_key) = options.base.api_key.clone() else {
        return Err(StreamFailure::new(
            output,
            format!("No API key for provider: {}", model.provider),
        ));
    };
    let compat = get_compat(&model);
    let cache_retention = resolve_cache_retention(options.base.cache_retention);
    let mut payload =
        match try_build_responses_payload(&model, &context, &options, &compat, cache_retention) {
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

    let request_url = match request_base_url(&model) {
        Ok(base_url) => format!("{}/responses", trim_end_slash(&base_url)),
        Err(error) => return Err(StreamFailure::new(output, error)),
    };
    let request_headers = match headers(
        &model,
        &context,
        &options.base,
        &api_key,
        &compat,
        cache_retention,
    ) {
        Ok(headers) => headers,
        Err(error) => return Err(StreamFailure::new(output, error)),
    };
    let client = options.base.http_client.clone().unwrap_or_default();
    let response = match send_with_retries(&options.base, || {
        client
            .post(request_url.as_str())
            .headers(request_headers.clone())
            .json(&payload)
            .timeout(request_timeout(options.base.timeout_ms))
    })
    .await
    {
        Ok(response) => response,
        Err(Error::Cancelled) => return Err(StreamFailure::cancelled(output)),
        Err(error) => return Err(StreamFailure::new(output, error)),
    };
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(StreamFailure::api_status(output, status, body));
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
            Err(Error::Cancelled) => return Err(StreamFailure::cancelled(output)),
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
                            arguments: json!({}),
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
                    && let Some(part_type) = parsed.get("part").and_then(response_text_part_type)
                {
                    current_text_part = Some(part_type);
                }
            }
            "response.reasoning_summary_part.added" => {
                if current_item
                    .as_ref()
                    .and_then(|item| item.get("type"))
                    .and_then(Value::as_str)
                    == Some("reasoning")
                    && let Some(item) = current_item.as_mut()
                {
                    let part = parsed.get("part").cloned().unwrap_or(Value::Null);
                    if let Some(Value::Array(summary)) = item.get_mut("summary") {
                        summary.push(part);
                    } else {
                        item["summary"] = json!([part]);
                    }
                }
            }
            "response.reasoning_summary_text.delta" => {
                let delta = parsed
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let has_summary_part = current_item
                    .as_mut()
                    .and_then(|item| item.get_mut("summary"))
                    .and_then(Value::as_array_mut)
                    .and_then(|summary| summary.last_mut())
                    .map(|part| {
                        if let Some(Value::String(text)) = part.get_mut("text") {
                            text.push_str(delta);
                        }
                    })
                    .is_some();
                if has_summary_part
                    && let Some(index) = current_block
                    && let Some(AssistantContent::Thinking(block)) = output.content.get_mut(index)
                {
                    block.thinking.push_str(delta);
                    sender.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: index,
                        delta: delta.to_string(),
                        partial: output.clone(),
                    });
                }
            }
            "response.reasoning_text.delta" => {
                let delta = parsed
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(index) = current_block
                    && let Some(AssistantContent::Thinking(block)) = output.content.get_mut(index)
                {
                    block.thinking.push_str(delta);
                    sender.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: index,
                        delta: delta.to_string(),
                        partial: output.clone(),
                    });
                }
            }
            "response.reasoning_summary_part.done" => {
                let has_summary_part = current_item
                    .as_mut()
                    .and_then(|item| item.get_mut("summary"))
                    .and_then(Value::as_array_mut)
                    .and_then(|summary| summary.last_mut())
                    .map(|part| {
                        if let Some(Value::String(text)) = part.get_mut("text") {
                            text.push_str("\n\n");
                        }
                    })
                    .is_some();
                if has_summary_part
                    && let Some(index) = current_block
                    && let Some(AssistantContent::Thinking(block)) = output.content.get_mut(index)
                {
                    block.thinking.push_str("\n\n");
                    sender.push(AssistantMessageEvent::ThinkingDelta {
                        content_index: index,
                        delta: "\n\n".to_string(),
                        partial: output.clone(),
                    });
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
                if let Some(index) = current_block
                    && let Some(AssistantContent::Text(block)) = output.content.get_mut(index)
                {
                    block.text.push_str(delta);
                    sender.push(AssistantMessageEvent::TextDelta {
                        content_index: index,
                        delta: delta.to_string(),
                        partial: output.clone(),
                    });
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
                                    .filter(|args| !args.is_empty())
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
                output.stop_reason =
                    match map_status(response.get("status").and_then(Value::as_str)) {
                        Ok(stop_reason) => stop_reason,
                        Err(message) => return Err(StreamFailure::new(output, message)),
                    };
                if output
                    .content
                    .iter()
                    .any(|block| matches!(block, AssistantContent::ToolCall(_)))
                    && output.stop_reason == StopReason::Stop
                {
                    output.stop_reason = StopReason::ToolUse;
                }
                break;
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

#[cfg(test)]
fn build_responses_payload(
    model: &Model,
    context: &Context,
    options: &OpenAIResponsesOptions,
    compat: &ResolvedOpenAIResponsesCompat,
    cache_retention: CacheRetention,
) -> Value {
    try_build_responses_payload(model, context, options, compat, cache_retention)
        .expect("valid OpenAI Responses payload")
}

fn try_build_responses_payload(
    model: &Model,
    context: &Context,
    options: &OpenAIResponsesOptions,
    compat: &ResolvedOpenAIResponsesCompat,
    cache_retention: CacheRetention,
) -> Result<Value> {
    let messages = try_convert_responses_messages(
        model,
        context,
        &OPENAI_TOOL_CALL_PROVIDERS.iter().copied().collect(),
        true,
    )?;
    let mut payload = json!({
        "model": model.id,
        "input": messages,
        "stream": true
    });
    let object = payload.as_object_mut().expect("payload object");
    object.insert("store".to_string(), json!(false));
    if let Some(max_tokens) = options.base.max_tokens.filter(|max_tokens| *max_tokens > 0) {
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
    if cache_retention != CacheRetention::None
        && let Some(session_id) = &options.base.session_id
    {
        object.insert(
            "prompt_cache_key".to_string(),
            json!(clamp_openai_prompt_cache_key(Some(session_id))),
        );
    }
    if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
        object.insert("prompt_cache_retention".to_string(), json!("24h"));
    }
    if model.reasoning {
        let reasoning_effort = options
            .reasoning_effort
            .filter(|effort| *effort != ModelThinkingLevel::Off);
        let has_reasoning_summary = options
            .reasoning_summary
            .as_ref()
            .is_some_and(Option::is_some);
        if reasoning_effort.is_some() || has_reasoning_summary {
            let effort = reasoning_effort
                .and_then(|effort| {
                    model
                        .thinking_level_map
                        .get(effort.as_str())
                        .cloned()
                        .flatten()
                })
                .or_else(|| reasoning_effort.map(|effort| effort.as_str().to_string()))
                .unwrap_or_else(|| "medium".to_string());
            let summary = options
                .reasoning_summary
                .clone()
                .flatten()
                .filter(|summary| !summary.is_empty())
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
    Ok(payload)
}

#[cfg(test)]
fn convert_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &HashSet<&str>,
    include_system_prompt: bool,
) -> Vec<Value> {
    try_convert_responses_messages(
        model,
        context,
        allowed_tool_call_providers,
        include_system_prompt,
    )
    .expect("valid OpenAI Responses message history")
}

fn try_convert_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &HashSet<&str>,
    include_system_prompt: bool,
) -> Result<Vec<Value>> {
    let mut messages = Vec::new();
    let transformed = transform_messages(
        context.messages.as_slice(),
        model,
        |id, target_model, source| {
            normalize_responses_tool_call_id(id, target_model, source, allowed_tool_call_providers)
        },
    );
    if include_system_prompt
        && let Some(system_prompt) = &context.system_prompt
        && !system_prompt.is_empty()
    {
        messages.push(json!({
            "role": if model.reasoning { "developer" } else { "system" },
            "content": system_prompt,
        }));
    }

    let mut msg_index = 0usize;
    for msg in transformed {
        match msg {
            crate::types::Message::User(user) => match user.content {
                UserMessageContent::Text(text) => messages.push(json!({
                    "role": "user",
                    "content": [{ "type": "input_text", "text": &text }]
                })),
                UserMessageContent::Parts(parts) => {
                    let content: Vec<Value> = parts
                        .iter()
                        .map(|item| match item {
                            UserContent::Text(text) => {
                                json!({ "type": "input_text", "text": &text.text })
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
                            if let Some(signature) = thinking.thinking_signature
                                && !signature.is_empty()
                            {
                                let reasoning_item = serde_json::from_str::<Value>(&signature)?;
                                output.push(reasoning_item);
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
                                .and_then(|sig| (!sig.0.is_empty()).then(|| sig.0.clone()))
                                .unwrap_or(fallback_id);
                            let msg_id = if msg_id.len() > 64 {
                                format!("msg_{}", short_hash(&msg_id))
                            } else {
                                msg_id
                            };
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{ "type": "output_text", "text": &text.text, "annotations": [] }],
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
                        content.push(json!({ "type": "input_text", "text": &text_result }));
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
                    json!(if text_result.is_empty() {
                        "(see attached image)"
                    } else {
                        &text_result
                    })
                };
                messages.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output
                }));
            }
            crate::types::Message::Custom(_) => {}
        }
        msg_index += 1;
    }
    Ok(messages)
}

fn convert_responses_tools(tools: &[Tool], strict: Option<bool>) -> Vec<Value> {
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
    if signature.is_empty() {
        return None;
    }
    if signature.starts_with('{')
        && let Ok(parsed) = serde_json::from_str::<Value>(signature)
        && parsed.get("v").and_then(Value::as_u64) == Some(1)
        && let Some(id) = parsed.get("id").and_then(Value::as_str)
    {
        let phase = match parsed.get("phase").and_then(Value::as_str) {
            Some("commentary") => Some(TextPhase::Commentary),
            Some("final_answer") => Some(TextPhase::FinalAnswer),
            _ => None,
        };
        return Some((id.to_string(), phase));
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

fn map_status(status: Option<&str>) -> std::result::Result<StopReason, String> {
    match status {
        Some("completed") | None => Ok(StopReason::Stop),
        Some("incomplete") => Ok(StopReason::Length),
        Some("failed") | Some("cancelled") => Ok(StopReason::Error),
        Some("in_progress") | Some("queued") => Ok(StopReason::Stop),
        Some(status) => Err(format!("Unhandled stop reason: {status}")),
    }
}

fn get_compat(model: &Model) -> ResolvedOpenAIResponsesCompat {
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
) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    if !api_key.is_empty() {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| Error::InvalidHeaderValue("authorization".to_string(), e))?,
        );
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
    if let Some(session_id) = &options.session_id
        && cache_retention != CacheRetention::None
    {
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
    for (name, value) in &options.headers {
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
    Ok(model.base_url.clone())
}

fn format_openai_responses_api_error(status: reqwest::StatusCode, body: &str) -> String {
    let message = openai_error_message(body).unwrap_or_else(|| {
        let body = body.trim();
        if body.is_empty() {
            format!("{} status code (no body)", status.as_u16())
        } else {
            body.to_string()
        }
    });
    format!("OpenAI API error ({}): {}", status.as_u16(), message)
}

fn openai_error_message(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| value.get("message").and_then(Value::as_str))
        .or_else(|| value.get("error").and_then(Value::as_str))
        .or_else(|| value.as_str())
        .filter(|message| !message.trim().is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::types::{Message, ModelCost, ResponseHook, ToolResultMessage};
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

    fn reasoning_model_with_off_support(id: &str) -> Model {
        let mut model = model();
        model.id = id.to_string();
        model
            .thinking_level_map
            .insert("off".to_string(), Some("none".to_string()));
        model
    }

    fn reasoning_model_without_off_support(id: &str) -> Model {
        let mut model = model();
        model.id = id.to_string();
        model.thinking_level_map.insert("off".to_string(), None);
        model
    }

    #[test]
    fn should_handle_empty_content_array() {
        let model = model();
        let context = Context {
            messages: vec![Message::User(crate::types::UserMessage {
                content: UserMessageContent::Parts(Vec::new()),
                timestamp: 0,
            })],
            ..Default::default()
        };

        let messages =
            convert_responses_messages(&model, &context, &["openai"].into_iter().collect(), true);

        assert!(messages.is_empty());
    }

    #[test]
    fn should_handle_empty_string_content() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("")],
            ..Default::default()
        };

        let messages =
            convert_responses_messages(&model, &context, &["openai"].into_iter().collect(), true);

        assert_eq!(
            messages,
            vec![json!({
                "role": "user",
                "content": [{ "type": "input_text", "text": "" }]
            })]
        );
    }

    #[test]
    fn should_handle_whitespace_only_content() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("   \n\t  ")],
            ..Default::default()
        };

        let messages =
            convert_responses_messages(&model, &context, &["openai"].into_iter().collect(), true);

        assert_eq!(
            messages,
            vec![json!({
                "role": "user",
                "content": [{ "type": "input_text", "text": "   \n\t  " }]
            })]
        );
    }

    #[test]
    fn should_handle_empty_assistant_message_in_conversation() {
        let model = model();
        let context = Context {
            messages: vec![
                Message::user_text("Hello, how are you?"),
                Message::Assistant(AssistantMessage::empty_for(&model)),
                Message::user_text("Please respond this time."),
            ],
            ..Default::default()
        };

        let messages =
            convert_responses_messages(&model, &context, &["openai"].into_iter().collect(), true);
        let roles = messages
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
        let error = match stream_simple_openai_responses(
            model(),
            Context::default(),
            SimpleStreamOptions::default(),
        ) {
            Ok(_) => panic!("missing API key should fail before stream creation"),
            Err(error) => error,
        };

        assert!(matches!(error, crate::Error::MissingApiKey(provider) if provider == "openai"));
    }

    #[tokio::test]
    async fn invalid_replayed_reasoning_signature_errors_before_request() {
        let model = model();
        let mut assistant = AssistantMessage::empty_for(&model);
        assistant
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some("{".to_string()),
                redacted: None,
            }));
        let stream = stream_openai_responses(
            model,
            Context {
                messages: vec![Message::Assistant(assistant)],
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

        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(
            result
                .error_message
                .as_deref()
                .is_some_and(|message| message.contains("line 1 column"))
        );
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
    fn responses_payload_treats_off_reasoning_effort_as_unspecified() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAIResponsesOptions {
            reasoning_effort: Some(ModelThinkingLevel::Off),
            reasoning_summary: Some(Some("concise".to_string())),
            ..Default::default()
        };

        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(
            payload["reasoning"],
            json!({ "effort": "medium", "summary": "concise" })
        );
    }

    #[test]
    fn responses_payload_skips_empty_system_prompt() {
        let model = model();
        let context = Context {
            system_prompt: Some(String::new()),
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

        assert_eq!(payload["input"][0]["role"], "user");
    }

    #[test]
    fn response_payload_sends_store_false_by_default() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };

        let payload = build_responses_payload(
            &model,
            &context,
            &OpenAIResponsesOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["store"], json!(false));
    }

    #[test]
    fn response_payload_forwards_requested_service_tier() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAIResponsesOptions {
            service_tier: Some("priority".to_string()),
            ..Default::default()
        };

        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["service_tier"], json!("priority"));
    }

    #[test]
    fn response_payload_null_reasoning_summary_alone_does_not_enable_reasoning() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let payload = build_responses_payload(
            &model,
            &context,
            &OpenAIResponsesOptions {
                reasoning_summary: Some(None),
                ..Default::default()
            },
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["reasoning"], json!({ "effort": "none" }));
        assert!(payload.get("include").is_none());
    }

    #[test]
    fn response_payload_reasoning_summary_uses_medium_effort_without_explicit_effort() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let payload = build_responses_payload(
            &model,
            &context,
            &OpenAIResponsesOptions {
                reasoning_summary: Some(Some("concise".to_string())),
                ..Default::default()
            },
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(
            payload["reasoning"],
            json!({ "effort": "medium", "summary": "concise" })
        );
        assert_eq!(payload["include"][0], "reasoning.encrypted_content");
    }

    #[test]
    fn response_payload_empty_reasoning_summary_defaults_to_auto() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let payload = build_responses_payload(
            &model,
            &context,
            &OpenAIResponsesOptions {
                reasoning_summary: Some(Some(String::new())),
                ..Default::default()
            },
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(
            payload["reasoning"],
            json!({ "effort": "medium", "summary": "auto" })
        );
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
    fn response_payload_sets_openai_prompt_cache_fields() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAIResponsesOptions {
            base: StreamOptions {
                session_id: Some(format!("{}tail", "x".repeat(64))),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Long,
        );

        assert_eq!(payload["prompt_cache_key"], json!("x".repeat(64)));
        assert_eq!(payload["prompt_cache_retention"], json!("24h"));
    }

    #[test]
    fn response_payload_sets_openai_prompt_cache_key_for_short_retention() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAIResponsesOptions {
            base: StreamOptions {
                session_id: Some("session-short".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["prompt_cache_key"], json!("session-short"));
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn response_payload_uses_pi_cache_retention_for_openai_requests() {
        let _env = crate::test_env::EnvVarGuard::set("PI_CACHE_RETENTION", "long");
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAIResponsesOptions {
            base: StreamOptions {
                session_id: Some("session-env".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            resolve_cache_retention(options.base.cache_retention),
        );

        assert_eq!(payload["prompt_cache_key"], json!("session-env"));
        assert_eq!(payload["prompt_cache_retention"], json!("24h"));
    }

    #[test]
    fn response_payload_omits_long_retention_when_compat_disables_it() {
        let mut model = model();
        model.compat.openai_responses.supports_long_cache_retention = Some(false);
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAIResponsesOptions {
            base: StreamOptions {
                session_id: Some("session-compat-false".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Long,
        );

        assert_eq!(payload["prompt_cache_key"], json!("session-compat-false"));
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn response_payload_omits_prompt_cache_fields_when_retention_is_none() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAIResponsesOptions {
            base: StreamOptions {
                session_id: Some("session-123".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::None,
        );

        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn response_payload_omits_default_max_output_tokens() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let payload = build_responses_payload(
            &model,
            &context,
            &OpenAIResponsesOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("max_output_tokens").is_none());
    }

    #[test]
    fn response_payload_sends_explicit_max_output_tokens() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAIResponsesOptions {
            base: StreamOptions {
                max_tokens: Some(1234),
                ..Default::default()
            },
            ..Default::default()
        };
        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["max_output_tokens"], json!(1234));
    }

    #[test]
    fn response_payload_omits_zero_max_output_tokens() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAIResponsesOptions {
            base: StreamOptions {
                max_tokens: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };
        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("max_output_tokens").is_none());
    }

    #[test]
    fn response_headers_set_and_omit_cache_affinity_by_cache_retention() {
        let model = model();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
        };
        let options = StreamOptions {
            session_id: Some("session-123".to_string()),
            ..Default::default()
        };
        let compat = get_compat(&model);

        let request_headers = headers(
            &model,
            &context,
            &options,
            "test-key",
            &compat,
            CacheRetention::Short,
        )
        .unwrap();
        assert_eq!(
            request_headers
                .get("session_id")
                .and_then(|value| value.to_str().ok()),
            Some("session-123")
        );
        assert_eq!(
            request_headers
                .get("x-client-request-id")
                .and_then(|value| value.to_str().ok()),
            Some("session-123")
        );

        let request_headers = headers(
            &model,
            &context,
            &options,
            "test-key",
            &compat,
            CacheRetention::None,
        )
        .unwrap();
        assert!(request_headers.get("session_id").is_none());
        assert!(request_headers.get("x-client-request-id").is_none());
    }

    #[test]
    fn response_headers_can_omit_session_id_header_only() {
        let mut model = model();
        model.compat.openai_responses.send_session_id_header = Some(false);
        let context = Context {
            system_prompt: None,
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
        };
        let options = StreamOptions {
            session_id: Some("session-123".to_string()),
            ..Default::default()
        };
        let request_headers = headers(
            &model,
            &context,
            &options,
            "test-key",
            &get_compat(&model),
            CacheRetention::Short,
        )
        .unwrap();

        assert!(request_headers.get("session_id").is_none());
        assert_eq!(
            request_headers
                .get("x-client-request-id")
                .and_then(|value| value.to_str().ok()),
            Some("session-123")
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

    #[test]
    fn response_payload_sends_none_reasoning_for_openai_models_that_support_off() {
        for model_id in [
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.3-codex",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.5",
        ] {
            let model = reasoning_model_with_off_support(model_id);
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

            assert_eq!(
                payload["reasoning"],
                json!({ "effort": "none" }),
                "{model_id}"
            );
        }
    }

    #[test]
    fn response_payload_omits_default_reasoning_when_off_is_unsupported() {
        for model_id in [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-pro",
            "gpt-5.2-pro",
            "gpt-5.4-pro",
            "gpt-5.5-pro",
        ] {
            let model = reasoning_model_without_off_support(model_id);
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

            assert!(payload.get("reasoning").is_none(), "{model_id}");
        }
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
        let mut result = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event");
            match event {
                AssistantMessageEvent::TextDelta { delta, .. } => deltas.push(delta),
                AssistantMessageEvent::Done { message, .. } => result = Some(message),
                AssistantMessageEvent::Error { error, .. } => result = Some(error),
                _ => {}
            }
        }
        let result = result.expect("final message");
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
    async fn removes_partial_json_from_persisted_tool_call_blocks_at_output_item_done() {
        let arguments = r#"{"path":"README.md","content":"updated"}"#;
        let body = sse_body(&[
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "edit",
                    "arguments": ""
                }
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "delta": r#"{"path":"README.md""#
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "delta": r#","content":"updated"}"#
            }),
            json!({
                "type": "response.function_call_arguments.done",
                "arguments": arguments
            }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "edit",
                    "arguments": arguments
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
        let mut ended_tool_call = None;
        let mut result = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event");
            match event {
                AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                    ended_tool_call = Some(tool_call);
                }
                AssistantMessageEvent::Done { message, .. } => result = Some(message),
                AssistantMessageEvent::Error { error, .. } => result = Some(error),
                _ => {}
            }
        }
        let result = result.expect("final message");
        let tool_call = match result.content.first().expect("tool call content") {
            AssistantContent::ToolCall(tool_call) => tool_call,
            other => panic!("expected tool call, got {other:?}"),
        };

        assert_eq!(tool_call.id, "call_test|fc_test");
        assert_eq!(tool_call.name, "edit");
        assert_eq!(
            tool_call.arguments,
            json!({ "path": "README.md", "content": "updated" })
        );
        assert_eq!(
            ended_tool_call.expect("toolcall_end event"),
            tool_call.clone()
        );
        let serialized = serde_json::to_value(tool_call).expect("serialize tool call");
        assert!(serialized.get("partialJson").is_none());
        assert!(serialized.get("partial_json").is_none());
    }

    #[tokio::test]
    async fn reasoning_summary_delta_ignores_missing_summary_part() {
        let body = sse_body(&[
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "reasoning",
                    "id": "rs_test",
                    "summary": []
                }
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "delta": "ignored"
            }),
            json!({
                "type": "response.reasoning_summary_part.done"
            }),
            json!({
                "type": "response.reasoning_text.delta",
                "delta": "raw reasoning"
            }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_test",
                    "summary": []
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
        let mut result = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event");
            match event {
                AssistantMessageEvent::ThinkingDelta { delta, .. } => deltas.push(delta),
                AssistantMessageEvent::Done { message, .. } => result = Some(message),
                AssistantMessageEvent::Error { error, .. } => result = Some(error),
                _ => {}
            }
        }
        let result = result.expect("final message");

        assert_eq!(deltas, vec!["raw reasoning"]);
        assert_eq!(
            result.content,
            vec![AssistantContent::Thinking(ThinkingContent {
                thinking: "raw reasoning".to_string(),
                thinking_signature: Some(
                    json!({
                        "type": "reasoning",
                        "id": "rs_test",
                        "summary": []
                    })
                    .to_string()
                ),
                redacted: None,
            })]
        );
    }

    #[tokio::test]
    async fn reasoning_summary_part_added_initializes_null_summary() {
        let body = sse_body(&[
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "reasoning",
                    "id": "rs_test",
                    "summary": null
                }
            }),
            json!({
                "type": "response.reasoning_summary_part.added",
                "part": { "type": "summary_text", "text": "" }
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "delta": "kept"
            }),
            json!({
                "type": "response.reasoning_summary_part.done"
            }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_test",
                    "summary": null
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
        let mut result = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event");
            match event {
                AssistantMessageEvent::ThinkingDelta { delta, .. } => deltas.push(delta),
                AssistantMessageEvent::Done { message, .. } => result = Some(message),
                AssistantMessageEvent::Error { error, .. } => result = Some(error),
                _ => {}
            }
        }
        let result = result.expect("final message");

        assert_eq!(deltas, vec!["kept", "\n\n"]);
        assert_eq!(
            result.content,
            vec![AssistantContent::Thinking(ThinkingContent {
                thinking: "kept\n\n".to_string(),
                thinking_signature: Some(
                    json!({
                        "type": "reasoning",
                        "id": "rs_test",
                        "summary": null
                    })
                    .to_string()
                ),
                redacted: None,
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

        let stream = stream_openai_responses(
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
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.usage.cost.input, 1.0);
        assert_eq!(result.usage.cost.output, 2.0);
        assert_eq!(result.usage.cost.total, 3.0);
    }

    #[tokio::test]
    async fn response_id_is_exposed_from_responses_stream_events() {
        let body = sse_body(&[
            json!({
                "type": "response.created",
                "response": { "id": "resp_created" }
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_completed",
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

        let stream = stream_openai_responses(
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
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.response_id.as_deref(), Some("resp_completed"));
    }

    #[tokio::test]
    async fn terminal_response_events_finish_without_waiting_for_sse_close() {
        for (event_type, status, expected_stop_reason) in [
            ("response.completed", "completed", StopReason::Stop),
            ("response.incomplete", "incomplete", StopReason::Length),
        ] {
            let body = sse_body(&[json!({
                "type": event_type,
                "response": {
                    "id": format!("resp_{status}"),
                    "status": status,
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": { "cached_tokens": 0 }
                    }
                }
            })]);
            let (base_url, release_server) = spawn_hanging_sse_server(body).await;
            let mut model = model();
            model.base_url = base_url;

            let stream = stream_openai_responses(
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

            let result = match tokio::time::timeout(
                std::time::Duration::from_secs(1),
                crate::stream::final_message_from_stream(stream),
            )
            .await
            {
                Ok(result) => result.unwrap(),
                Err(error) => {
                    release_server.notify_waiters();
                    panic!("stream did not finish after {event_type}: {error}");
                }
            };
            release_server.notify_waiters();

            let expected_response_id = format!("resp_{status}");
            assert_eq!(result.stop_reason, expected_stop_reason, "{event_type}");
            assert_eq!(
                result.response_id.as_deref(),
                Some(expected_response_id.as_str()),
                "{event_type}"
            );
        }
    }

    #[tokio::test]
    async fn unknown_terminal_status_reports_provider_status() {
        let body = sse_body(&[json!({
            "type": "response.completed",
            "response": {
                "id": "resp_unknown",
                "status": "stalled",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": { "cached_tokens": 0 }
                }
            }
        })]);
        let base_url = spawn_sse_server(body).await;
        let mut model = model();
        model.base_url = base_url;

        let stream = stream_openai_responses(
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
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Unhandled stop reason: stalled")
        );
    }

    #[tokio::test]
    async fn should_handle_immediate_abort() {
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        cancellation_token.cancel();
        let stream = stream_openai_responses(
            model(),
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAIResponsesOptions {
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
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "message",
                    "id": "msg_abort",
                    "role": "assistant",
                    "content": []
                }
            }),
            json!({
                "type": "response.content_part.added",
                "part": { "type": "output_text", "text": "" }
            }),
            json!({
                "type": "response.output_text.delta",
                "delta": "partial"
            }),
        ]))
        .await;
        let mut response_model = model();
        response_model.base_url = base_url;
        let mut stream = stream_openai_responses(
            response_model,
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAIResponsesOptions {
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

    #[test]
    fn response_service_tier_pricing_multipliers_match_openai_models() {
        for (model_id, service_tier, multiplier) in [
            ("gpt-5.4", "priority", 2.0),
            ("gpt-5.5", "priority", 2.5),
            ("gpt-5.5", "flex", 0.5),
        ] {
            let mut model = model();
            model.id = model_id.to_string();
            let mut usage = Usage {
                cost: crate::types::UsageCost {
                    input: 2.0,
                    output: 4.0,
                    cache_read: 1.0,
                    cache_write: 3.0,
                    total: 10.0,
                },
                ..Default::default()
            };

            apply_service_tier_pricing(&mut usage, service_tier, &model);

            assert_eq!(usage.cost.input, 2.0 * multiplier);
            assert_eq!(usage.cost.output, 4.0 * multiplier);
            assert_eq!(usage.cost.cache_read, multiplier);
            assert_eq!(usage.cost.cache_write, 3.0 * multiplier);
            assert_eq!(usage.cost.total, 10.0 * multiplier);
        }
    }

    #[tokio::test]
    async fn response_function_call_done_uses_final_arguments_without_deltas() {
        let body = sse_body(&[
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "read",
                    "arguments": ""
                }
            }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "read",
                    "arguments": "{\"path\":\"README.md\"}"
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

        let stream = stream_openai_responses(
            model,
            Context {
                messages: vec![Message::user_text("read")],
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
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::ToolUse);
        assert_eq!(
            result.content,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "call_test|fc_test".to_string(),
                name: "read".to_string(),
                arguments: json!({ "path": "README.md" }),
                thought_signature: None,
            })]
        );
    }

    #[tokio::test]
    async fn response_function_call_start_keeps_initial_arguments_private_until_done() {
        let body = sse_body(&[
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "read",
                    "arguments": "{\"path\":\"README.md\"}"
                }
            }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "read",
                    "arguments": "{\"path\":\"README.md\"}"
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
                messages: vec![Message::user_text("read")],
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

        let mut start_arguments = None;
        let mut ended_tool_call = None;
        let mut result = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event");
            match event {
                AssistantMessageEvent::ToolCallStart { partial, .. } => {
                    start_arguments = partial.content.iter().find_map(|block| match block {
                        AssistantContent::ToolCall(tool_call) => Some(tool_call.arguments.clone()),
                        _ => None,
                    });
                }
                AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                    ended_tool_call = Some(tool_call);
                }
                AssistantMessageEvent::Done { message, .. } => result = Some(message),
                AssistantMessageEvent::Error { error, .. } => result = Some(error),
                _ => {}
            }
        }
        let result = result.expect("final message");

        assert_eq!(start_arguments, Some(json!({})));
        let expected = ToolCall {
            id: "call_test|fc_test".to_string(),
            name: "read".to_string(),
            arguments: json!({ "path": "README.md" }),
            thought_signature: None,
        };
        assert_eq!(ended_tool_call, Some(expected.clone()));
        assert_eq!(result.content, vec![AssistantContent::ToolCall(expected)]);
    }

    #[tokio::test]
    async fn response_function_call_done_replaces_delta_arguments() {
        let body = sse_body(&[
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "edit",
                    "arguments": ""
                }
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "delta": "{\"path\":\"README.md\""
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "delta": ",\"content\":\"updated\"}"
            }),
            json!({
                "type": "response.function_call_arguments.done",
                "arguments": "{\"path\":\"README.md\",\"content\":\"updated\"}"
            }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "edit",
                    "arguments": "{\"path\":\"README.md\",\"content\":\"updated\"}"
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
                messages: vec![Message::user_text("edit")],
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

        let mut ended_tool_call = None;
        let mut result = None;
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event");
            match event {
                AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                    ended_tool_call = Some(tool_call);
                }
                AssistantMessageEvent::Done { message, .. } => result = Some(message),
                AssistantMessageEvent::Error { error, .. } => result = Some(error),
                _ => {}
            }
        }
        let result = result.expect("final message");
        let expected = ToolCall {
            id: "call_test|fc_test".to_string(),
            name: "edit".to_string(),
            arguments: json!({ "path": "README.md", "content": "updated" }),
            thought_signature: None,
        };

        assert_eq!(ended_tool_call, Some(expected.clone()));
        assert_eq!(result.content, vec![AssistantContent::ToolCall(expected)]);
    }

    #[tokio::test]
    async fn formats_http_status_errors_like_openai_responses() {
        let base_url = spawn_status_server(
            400,
            "Bad Request",
            json!({
                "error": {
                    "message": "Your input exceeds the context window of this model"
                }
            })
            .to_string(),
        )
        .await;
        let mut model = model();
        model.base_url = base_url;

        let stream = stream_openai_responses(
            model,
            Context {
                messages: vec![Message::user_text("too much")],
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
        let result = crate::stream::final_message_from_stream(stream)
            .await
            .unwrap();

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("OpenAI API error (400): Your input exceeds the context window of this model")
        );
    }

    #[tokio::test]
    async fn responses_provider_skips_on_response_for_api_errors() {
        let response_calls = Arc::new(AtomicUsize::new(0));
        let base_url = spawn_status_server(
            500,
            "Internal Server Error",
            json!({ "error": { "message": "upstream unavailable" } }).to_string(),
        )
        .await;
        let mut model = model();
        model.base_url = base_url;

        let stream = stream_openai_responses(
            model,
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAIResponsesOptions {
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
    async fn responses_provider_does_not_retry_by_default() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let base_url = spawn_retrying_sse_server(
            sse_body(&[json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_retry",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": { "cached_tokens": 0 }
                    }
                }
            })]),
            Arc::clone(&attempts),
        )
        .await;
        let mut model = model();
        model.base_url = base_url;

        let stream = stream_openai_responses(
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
    async fn responses_provider_honors_explicit_retry_settings() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let base_url = spawn_retrying_sse_server(
            sse_body(&[json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_retry",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": { "cached_tokens": 0 }
                    }
                }
            })]),
            Arc::clone(&attempts),
        )
        .await;
        let mut model = model();
        model.base_url = base_url;

        let stream = stream_openai_responses(
            model,
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAIResponsesOptions {
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
        assert_eq!(result.response_id.as_deref(), Some("resp_retry"));
    }

    #[test]
    fn generates_unique_fallback_message_ids_for_multiple_text_blocks_in_one_assistant_turn() {
        let model = model();
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

        let input =
            convert_responses_messages(&model, &context, &["openai"].into_iter().collect(), true);
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
            &["openai"].into_iter().collect(),
            true,
        );

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["id"], json!("msg_pi_0"));
        assert_eq!(messages[0]["content"][0]["text"], json!("visible"));
    }

    #[test]
    fn text_signature_unknown_phase_preserves_message_id() {
        let model = model();
        let mut assistant = AssistantMessage::empty_for(&model);
        assistant.content.push(AssistantContent::Text(TextContent {
            text: "visible".to_string(),
            text_signature: Some(
                json!({
                    "v": 1,
                    "id": "msg_valid",
                    "phase": "future_phase"
                })
                .to_string(),
            ),
        }));

        let messages = convert_responses_messages(
            &model,
            &Context {
                messages: vec![Message::Assistant(assistant)],
                ..Default::default()
            },
            &["openai"].into_iter().collect(),
            true,
        );

        assert_eq!(messages[0]["id"], json!("msg_valid"));
        assert!(messages[0].get("phase").is_none());
    }

    #[test]
    fn empty_text_signature_ids_use_fallback_message_ids() {
        let model = model();
        let mut assistant = AssistantMessage::empty_for(&model);
        assistant.content.push(AssistantContent::Text(TextContent {
            text: "empty signature".to_string(),
            text_signature: Some(String::new()),
        }));
        assistant.content.push(AssistantContent::Text(TextContent {
            text: "empty parsed id".to_string(),
            text_signature: Some(
                json!({
                    "v": 1,
                    "id": "",
                    "phase": "commentary"
                })
                .to_string(),
            ),
        }));

        let messages = convert_responses_messages(
            &model,
            &Context {
                messages: vec![Message::Assistant(assistant)],
                ..Default::default()
            },
            &["openai"].into_iter().collect(),
            true,
        );

        assert_eq!(messages[0]["id"], json!("msg_pi_0"));
        assert!(messages[0].get("phase").is_none());
        assert_eq!(messages[1]["id"], json!("msg_pi_0_1"));
        assert_eq!(messages[1]["phase"], json!("commentary"));
    }

    #[test]
    fn skips_aborted_reasoning_only_history() {
        let model = model();
        let mut aborted_assistant = AssistantMessage::empty_for(&model);
        aborted_assistant.stop_reason = StopReason::Aborted;
        aborted_assistant
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(
                    json!({
                        "type": "reasoning",
                        "id": "rs_aborted",
                        "summary": []
                    })
                    .to_string(),
                ),
                redacted: None,
            }));
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::user_text("Use the tool."),
                Message::Assistant(aborted_assistant),
                Message::user_text("Continue."),
            ],
            tools: Vec::new(),
        };

        let input =
            convert_responses_messages(&model, &context, &["openai"].into_iter().collect(), true);

        assert_eq!(input.len(), 2);
        assert!(input.iter().all(|item| {
            item.get("type").and_then(Value::as_str) != Some("reasoning")
                && item.get("id").and_then(Value::as_str) != Some("rs_aborted")
        }));
    }

    #[test]
    fn omits_paired_function_call_item_id_for_same_provider_model_handoff() {
        let mut source_model = model();
        source_model.id = "gpt-5-mini".to_string();
        let mut target_model = model();
        target_model.id = "gpt-5.2".to_string();
        let assistant = AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: "call_abc|fc_paired".to_string(),
                name: "double_number".to_string(),
                arguments: json!({ "value": 21 }),
                thought_signature: None,
            })],
            api: source_model.api.clone(),
            provider: source_model.provider.clone(),
            model: source_model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 2,
        };
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::user_text("Double 21."),
                Message::Assistant(assistant),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_abc|fc_paired".to_string(),
                    tool_name: "double_number".to_string(),
                    content: vec![ToolResultContent::text("42")],
                    details: None,
                    is_error: false,
                    timestamp: 3,
                }),
            ],
            tools: Vec::new(),
        };

        let input = convert_responses_messages(
            &target_model,
            &context,
            &["openai"].into_iter().collect(),
            true,
        );
        let function_call = input
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .expect("function_call");

        assert_eq!(function_call["call_id"], json!("call_abc"));
        assert!(function_call.get("id").is_none_or(Value::is_null));
        let function_call_output = input
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output"))
            .expect("function_call_output");
        assert_eq!(function_call_output["call_id"], json!("call_abc"));
    }

    #[test]
    fn preserves_responses_tool_item_ids_for_upstream_openai_tool_call_provider() {
        let target_model = model();
        let assistant = AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: "call_abc|fc_existing".to_string(),
                name: "double_number".to_string(),
                arguments: json!({ "value": 21 }),
                thought_signature: None,
            })],
            api: target_model.api.clone(),
            provider: target_model.provider.clone(),
            model: target_model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 2,
        };
        let context = Context {
            messages: vec![Message::Assistant(assistant)],
            ..Default::default()
        };

        let input = convert_responses_messages(
            &target_model,
            &context,
            &OPENAI_TOOL_CALL_PROVIDERS.iter().copied().collect(),
            true,
        );
        let function_call = input
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .expect("function_call");

        assert_eq!(function_call["call_id"], json!("call_abc"));
        assert_eq!(function_call["id"], json!("fc_existing"));
    }

    #[test]
    fn hashes_foreign_copilot_tool_item_ids_into_a_bounded_responses_safe_fc_hash_shape() {
        let model = model();
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

        let input =
            convert_responses_messages(&model, &context, &["openai"].into_iter().collect(), true);
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
    fn should_send_tool_result_images_in_function_call_output() {
        let model = model();
        let assistant = AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: "call-1".to_string(),
                name: "get_image".to_string(),
                arguments: json!({}),
                thought_signature: None,
            })],
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 1,
        };
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
            messages: vec![
                Message::user_text("Describe the tool image."),
                Message::Assistant(assistant),
                Message::ToolResult(tool_result),
            ],
            tools: Vec::new(),
        };

        let input =
            convert_responses_messages(&model, &context, &["openai"].into_iter().collect(), true);
        let function_call_index = input
            .iter()
            .position(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .expect("function_call item");
        let function_output_index = input
            .iter()
            .position(|item| {
                item.get("type").and_then(Value::as_str) == Some("function_call_output")
            })
            .expect("function_call_output item");
        assert!(
            function_output_index > function_call_index,
            "tool result output should follow its function call"
        );
        let function_output = input
            .get(function_output_index)
            .expect("function_call_output index");
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
        assert!(
            input
                .iter()
                .skip(function_output_index + 1)
                .all(|item| item.get("role").and_then(Value::as_str) != Some("user")),
            "tool result images must not be split into a later user message"
        );
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
            socket.flush().await.unwrap();
            release_task.notified().await;
        });
        (format!("http://{addr}"), release)
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
}
