use std::collections::HashMap;
use std::time::Duration;

use futures::{StreamExt, pin_mut};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value, json};

use crate::event_stream::AssistantMessageEventStreamSender;
use crate::models::{calculate_cost, clamp_thinking_level};
use crate::providers::simple_options::build_base_options;
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Context, ImageContent, Message,
    Model, ModelInput, ModelThinkingLevel, SimpleStreamOptions, StopReason, StreamOptions,
    TextContent, ThinkingContent, Tool, ToolCall, ToolResultContent, Usage, UserContent,
    UserMessageContent,
};
use crate::utils::hash::short_hash;
use crate::utils::headers::headers_to_record;
use crate::utils::json::parse_streaming_json;
use crate::utils::sanitize::sanitize_surrogates;
use crate::utils::sse;
use crate::utils::transform_messages::transform_messages;
use crate::{Error, Result};

const MISTRAL_TOOL_CALL_ID_LENGTH: usize = 9;
const MAX_MISTRAL_ERROR_BODY_CHARS: usize = 4000;

#[derive(Clone, Default)]
pub struct MistralOptions {
    pub base: StreamOptions,
    pub tool_choice: Option<Value>,
    pub prompt_mode: Option<String>,
    pub reasoning_effort: Option<String>,
}

pub fn stream_simple_mistral(
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
) -> crate::AssistantMessageEventStream {
    let api_key = options
        .stream
        .api_key
        .clone()
        .or_else(|| env_api_key(&model.provider));
    let Some(api_key) = api_key else {
        return immediate_error(model, "No API key for provider");
    };

    let base = build_base_options(&model, &options, api_key);
    let reasoning = options.reasoning.and_then(|reasoning| {
        let clamped = clamp_thinking_level(&model, reasoning);
        (clamped != ModelThinkingLevel::Off).then_some(clamped)
    });
    let should_use_reasoning = model.reasoning && reasoning.is_some();
    let prompt_mode = (should_use_reasoning && uses_prompt_mode_reasoning(&model))
        .then(|| "reasoning".to_string());
    let reasoning_effort = if should_use_reasoning && uses_reasoning_effort(&model) {
        reasoning.map(|level| map_reasoning_effort(&model, level))
    } else {
        None
    };

    stream_mistral(
        model,
        context,
        MistralOptions {
            base,
            tool_choice: None,
            prompt_mode,
            reasoning_effort,
        },
    )
}

pub fn stream_mistral(
    model: Model,
    context: Context,
    options: MistralOptions,
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

    fn api_status(output: AssistantMessage, status: reqwest::StatusCode, body: String) -> Self {
        let body = truncate_error_text(body.trim(), MAX_MISTRAL_ERROR_BODY_CHARS);
        Self {
            output,
            message: if body.is_empty() {
                format!("Mistral API error ({status}): {status}")
            } else {
                format!("Mistral API error ({}): {}", status.as_u16(), body)
            },
            cancelled: false,
        }
    }
}

async fn run_stream(
    model: Model,
    context: Context,
    options: MistralOptions,
    mut output: AssistantMessage,
    sender: &mut AssistantMessageEventStreamSender,
) -> std::result::Result<(), StreamFailure> {
    if is_cancelled(&options.base) {
        return Err(StreamFailure::cancelled(output));
    }

    let api_key = options
        .base
        .api_key
        .clone()
        .or_else(|| env_api_key(&model.provider))
        .ok_or_else(|| {
            StreamFailure::new(
                output.clone(),
                format!("No API key for provider: {}", model.provider),
            )
        })?;

    let normalizer = std::cell::RefCell::new(MistralToolCallIdNormalizer::default());
    let transformed_messages = transform_messages(&context.messages, &model, |id, _, _| {
        normalizer.borrow_mut().normalize(id)
    });
    let mut payload = build_chat_payload(&model, &context, &transformed_messages, &options);
    if let Some(on_payload) = &options.base.on_payload {
        match on_payload(payload.clone(), &model).await {
            Ok(Some(next)) => payload = next,
            Ok(None) => {}
            Err(error) => return Err(StreamFailure::new(output, error)),
        }
    }

    let client = reqwest::Client::new();
    let mut request = client
        .post(mistral_chat_completions_url(&model))
        .headers(match headers(&model, &options.base, &api_key) {
            Ok(headers) => headers,
            Err(error) => return Err(StreamFailure::new(output, error)),
        })
        .json(&to_mistral_api_payload(payload));
    if let Some(timeout_ms) = options.base.timeout_ms {
        request = request.timeout(Duration::from_millis(timeout_ms));
    }

    let response = if let Some(cancellation_token) = options.base.cancellation_token.as_ref() {
        match tokio::select! {
            _ = cancellation_token.cancelled() => Err(Error::Cancelled),
            response = request.send() => response.map_err(Error::from),
        } {
            Ok(response) => response,
            Err(Error::Cancelled) => return Err(StreamFailure::cancelled(output)),
            Err(error) => return Err(StreamFailure::new(output, error)),
        }
    } else {
        match request.send().await {
            Ok(response) => response,
            Err(error) => return Err(StreamFailure::new(output, error)),
        }
    };

    if let Some(on_response) = &options.base.on_response {
        let provider_response = crate::types::ProviderResponse {
            status: response.status().as_u16(),
            headers: headers_to_record(response.headers()),
        };
        if let Err(error) = on_response(provider_response, &model).await {
            return Err(StreamFailure::new(output, error));
        }
    }

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(StreamFailure::api_status(output, status, body));
    }

    sender.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });
    consume_chat_stream(&model, &mut output, sender, response, &options).await?;

    if is_cancelled(&options.base) {
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

#[derive(Default)]
struct MistralToolCallIdNormalizer {
    id_map: HashMap<String, String>,
    reverse_map: HashMap<String, String>,
}

impl MistralToolCallIdNormalizer {
    fn normalize(&mut self, id: &str) -> String {
        if let Some(existing) = self.id_map.get(id) {
            return existing.clone();
        }

        let mut attempt = 0;
        loop {
            let candidate = derive_mistral_tool_call_id(id, attempt);
            let owner = self.reverse_map.get(&candidate);
            if owner.is_none_or(|owner| owner == id) {
                self.id_map.insert(id.to_string(), candidate.clone());
                self.reverse_map.insert(candidate.clone(), id.to_string());
                return candidate;
            }
            attempt += 1;
        }
    }
}

fn derive_mistral_tool_call_id(id: &str, attempt: u32) -> String {
    let normalized = id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    if attempt == 0 && normalized.len() == MISTRAL_TOOL_CALL_ID_LENGTH {
        return normalized;
    }
    let seed_base = if normalized.is_empty() {
        id
    } else {
        &normalized
    };
    let seed = if attempt == 0 {
        seed_base.to_string()
    } else {
        format!("{seed_base}:{attempt}")
    };
    short_hash(&seed)
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(MISTRAL_TOOL_CALL_ID_LENGTH)
        .collect()
}

fn build_chat_payload(
    model: &Model,
    context: &Context,
    messages: &[Message],
    options: &MistralOptions,
) -> Value {
    let mut object = Map::from_iter([
        ("model".to_string(), json!(model.id)),
        ("stream".to_string(), json!(true)),
        (
            "messages".to_string(),
            Value::Array(to_chat_messages(
                messages,
                model.input.contains(&ModelInput::Image),
            )),
        ),
    ]);

    if !context.tools.is_empty() {
        object.insert(
            "tools".to_string(),
            Value::Array(to_function_tools(&context.tools)),
        );
    }
    if let Some(temperature) = options.base.temperature {
        object.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(max_tokens) = options.base.max_tokens {
        object.insert("maxTokens".to_string(), json!(max_tokens));
    }
    if let Some(tool_choice) = &options.tool_choice {
        object.insert("toolChoice".to_string(), tool_choice.clone());
    }
    if let Some(prompt_mode) = &options.prompt_mode {
        object.insert("promptMode".to_string(), json!(prompt_mode));
    }
    if let Some(reasoning_effort) = &options.reasoning_effort {
        object.insert("reasoningEffort".to_string(), json!(reasoning_effort));
    }

    if let Some(system_prompt) = &context.system_prompt {
        let mut messages = object
            .remove("messages")
            .and_then(|value| match value {
                Value::Array(messages) => Some(messages),
                _ => None,
            })
            .unwrap_or_default();
        messages.insert(
            0,
            json!({
                "role": "system",
                "content": sanitize_surrogates(system_prompt),
            }),
        );
        object.insert("messages".to_string(), Value::Array(messages));
    }

    Value::Object(object)
}

fn to_function_tools(tools: &[Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": false,
                }
            })
        })
        .collect()
}

fn to_chat_messages(messages: &[Message], supports_images: bool) -> Vec<Value> {
    let mut result = Vec::new();
    for message in messages {
        match message {
            Message::User(message) => match &message.content {
                UserMessageContent::Text(text) => {
                    result.push(json!({
                        "role": "user",
                        "content": sanitize_surrogates(text),
                    }));
                }
                UserMessageContent::Parts(parts) => {
                    let had_images = parts
                        .iter()
                        .any(|part| matches!(part, UserContent::Image(_)));
                    let content = parts
                        .iter()
                        .filter_map(|part| match part {
                            UserContent::Text(text) => Some(json!({
                                "type": "text",
                                "text": sanitize_surrogates(&text.text),
                            })),
                            UserContent::Image(image) if supports_images => Some(json!({
                                "type": "image_url",
                                "imageUrl": data_url(image),
                            })),
                            UserContent::Image(_) => None,
                        })
                        .collect::<Vec<_>>();
                    if !content.is_empty() {
                        result.push(json!({
                            "role": "user",
                            "content": content,
                        }));
                    } else if had_images && !supports_images {
                        result.push(json!({
                            "role": "user",
                            "content": "(image omitted: model does not support images)",
                        }));
                    }
                }
            },
            Message::Assistant(message) => {
                let mut content_parts = Vec::new();
                let mut tool_calls = Vec::new();
                for block in &message.content {
                    match block {
                        AssistantContent::Text(text) => {
                            if !text.text.trim().is_empty() {
                                content_parts.push(json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&text.text),
                                }));
                            }
                        }
                        AssistantContent::Thinking(thinking) => {
                            if !thinking.thinking.trim().is_empty() {
                                content_parts.push(json!({
                                    "type": "thinking",
                                    "thinking": [
                                        {
                                            "type": "text",
                                            "text": sanitize_surrogates(&thinking.thinking),
                                        }
                                    ],
                                }));
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(json!({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string()),
                                },
                            }));
                        }
                    }
                }
                let mut object = Map::from_iter([("role".to_string(), json!("assistant"))]);
                if !content_parts.is_empty() {
                    object.insert("content".to_string(), Value::Array(content_parts));
                }
                if !tool_calls.is_empty() {
                    object.insert("toolCalls".to_string(), Value::Array(tool_calls));
                }
                if object.len() > 1 {
                    result.push(Value::Object(object));
                }
            }
            Message::ToolResult(message) => {
                let text_result = message
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ToolResultContent::Text(text) => Some(sanitize_surrogates(&text.text)),
                        ToolResultContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_images = message
                    .content
                    .iter()
                    .any(|part| matches!(part, ToolResultContent::Image(_)));
                let mut content = vec![json!({
                    "type": "text",
                    "text": build_tool_result_text(&text_result, has_images, supports_images, message.is_error),
                })];
                if supports_images {
                    for part in &message.content {
                        if let ToolResultContent::Image(image) = part {
                            content.push(json!({
                                "type": "image_url",
                                "imageUrl": data_url(image),
                            }));
                        }
                    }
                }
                result.push(json!({
                    "role": "tool",
                    "toolCallId": message.tool_call_id,
                    "name": message.tool_name,
                    "content": content,
                }));
            }
        }
    }
    result
}

fn build_tool_result_text(
    text: &str,
    has_images: bool,
    supports_images: bool,
    is_error: bool,
) -> String {
    let trimmed = text.trim();
    let error_prefix = if is_error { "[tool error] " } else { "" };

    if !trimmed.is_empty() {
        let image_suffix = if has_images && !supports_images {
            "\n[tool image omitted: model does not support images]"
        } else {
            ""
        };
        return format!("{error_prefix}{trimmed}{image_suffix}");
    }

    if has_images {
        if supports_images {
            return if is_error {
                "[tool error] (see attached image)"
            } else {
                "(see attached image)"
            }
            .to_string();
        }
        return if is_error {
            "[tool error] (image omitted: model does not support images)"
        } else {
            "(image omitted: model does not support images)"
        }
        .to_string();
    }

    if is_error {
        "[tool error] (no tool output)".to_string()
    } else {
        "(no tool output)".to_string()
    }
}

fn data_url(image: &ImageContent) -> String {
    format!("data:{};base64,{}", image.mime_type, image.data)
}

async fn consume_chat_stream(
    model: &Model,
    output: &mut AssistantMessage,
    sender: &mut AssistantMessageEventStreamSender,
    response: reqwest::Response,
    options: &MistralOptions,
) -> std::result::Result<(), StreamFailure> {
    let mut active_block: Option<ActiveBlock> = None;
    let mut tool_blocks_by_key: HashMap<String, usize> = HashMap::new();
    let mut tool_order = Vec::new();
    let mut partial_args: HashMap<usize, String> = HashMap::new();

    let events = sse::events(response, options.base.cancellation_token.clone());
    pin_mut!(events);
    while let Some(event) = events.next().await {
        if is_cancelled(&options.base) {
            return Err(StreamFailure::cancelled(output.clone()));
        }

        let event = match event {
            Ok(event) => event,
            Err(Error::Cancelled) => return Err(StreamFailure::cancelled(output.clone())),
            Err(error) => return Err(StreamFailure::new(output.clone(), error)),
        };
        if event.data.trim() == "[DONE]" || event.data.trim().is_empty() {
            continue;
        }

        let chunk = match serde_json::from_str::<Value>(&event.data) {
            Ok(value) => value,
            Err(error) => return Err(StreamFailure::new(output.clone(), error)),
        };
        let chunk = chunk
            .get("data")
            .filter(|data| data.is_object())
            .unwrap_or(&chunk);
        if let Some(id) = string_field(chunk, &["id"]) {
            output.response_id.get_or_insert_with(|| id.to_string());
        }
        if let Some(response_model) = string_field(chunk, &["model"]) {
            if response_model != model.id && output.response_model.is_none() {
                output.response_model = Some(response_model.to_string());
            }
        }
        if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
            output.usage = parse_chunk_usage(usage, model);
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            continue;
        };
        if let Some(reason) = string_field(choice, &["finishReason", "finish_reason"]) {
            output.stop_reason = map_chat_stop_reason(reason);
        }

        let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
            continue;
        };

        if let Some(content) = delta.get("content").filter(|value| !value.is_null()) {
            consume_content_delta(content, output, sender, &mut active_block);
        }

        let tool_calls = delta
            .get("toolCalls")
            .or_else(|| delta.get("tool_calls"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for tool_call in tool_calls {
            finish_active_block(output, sender, active_block.take());
            consume_tool_call_delta(
                &tool_call,
                output,
                sender,
                &mut tool_blocks_by_key,
                &mut tool_order,
                &mut partial_args,
            );
        }
    }

    finish_active_block(output, sender, active_block.take());
    for index in tool_order {
        let arguments = parse_streaming_json(partial_args.get(&index).map(String::as_str));
        if let Some(AssistantContent::ToolCall(tool_call)) = output.content.get_mut(index) {
            tool_call.arguments = arguments;
            sender.push(AssistantMessageEvent::ToolCallEnd {
                content_index: index,
                tool_call: tool_call.clone(),
                partial: output.clone(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveBlock {
    Text(usize),
    Thinking(usize),
}

fn consume_content_delta(
    content: &Value,
    output: &mut AssistantMessage,
    sender: &mut AssistantMessageEventStreamSender,
    active_block: &mut Option<ActiveBlock>,
) {
    match content {
        Value::String(text) => {
            if !text.is_empty() {
                push_text_delta(text, output, sender, active_block);
            }
        }
        Value::Array(items) => {
            for item in items {
                if let Some(text) = item.as_str() {
                    if !text.is_empty() {
                        push_text_delta(text, output, sender, active_block);
                    }
                    continue;
                }
                if item.get("type").and_then(Value::as_str) == Some("thinking") {
                    let thinking = item
                        .get("thinking")
                        .and_then(Value::as_array)
                        .map(|parts| {
                            parts
                                .iter()
                                .filter_map(|part| string_field(part, &["text"]))
                                .collect::<String>()
                        })
                        .unwrap_or_default();
                    if !thinking.is_empty() {
                        push_thinking_delta(&thinking, output, sender, active_block);
                    }
                    continue;
                }
                if item.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(text) = string_field(item, &["text"])
                    && !text.is_empty()
                {
                    push_text_delta(text, output, sender, active_block);
                }
            }
        }
        _ => {}
    }
}

fn push_text_delta(
    text: &str,
    output: &mut AssistantMessage,
    sender: &mut AssistantMessageEventStreamSender,
    active_block: &mut Option<ActiveBlock>,
) {
    let text = sanitize_surrogates(text);
    let index = if let Some(ActiveBlock::Text(index)) = *active_block {
        index
    } else {
        finish_active_block(output, sender, active_block.take());
        let index = output.content.len();
        output.content.push(AssistantContent::Text(TextContent {
            text: String::new(),
            text_signature: None,
        }));
        sender.push(AssistantMessageEvent::TextStart {
            content_index: index,
            partial: output.clone(),
        });
        *active_block = Some(ActiveBlock::Text(index));
        index
    };
    if let Some(AssistantContent::Text(block)) = output.content.get_mut(index) {
        block.text.push_str(&text);
    }
    sender.push(AssistantMessageEvent::TextDelta {
        content_index: index,
        delta: text,
        partial: output.clone(),
    });
}

fn push_thinking_delta(
    thinking: &str,
    output: &mut AssistantMessage,
    sender: &mut AssistantMessageEventStreamSender,
    active_block: &mut Option<ActiveBlock>,
) {
    let thinking = sanitize_surrogates(thinking);
    let index = if let Some(ActiveBlock::Thinking(index)) = *active_block {
        index
    } else {
        finish_active_block(output, sender, active_block.take());
        let index = output.content.len();
        output
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: None,
                redacted: None,
            }));
        sender.push(AssistantMessageEvent::ThinkingStart {
            content_index: index,
            partial: output.clone(),
        });
        *active_block = Some(ActiveBlock::Thinking(index));
        index
    };
    if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(index) {
        block.thinking.push_str(&thinking);
    }
    sender.push(AssistantMessageEvent::ThinkingDelta {
        content_index: index,
        delta: thinking,
        partial: output.clone(),
    });
}

fn finish_active_block(
    output: &AssistantMessage,
    sender: &mut AssistantMessageEventStreamSender,
    active_block: Option<ActiveBlock>,
) {
    match active_block {
        Some(ActiveBlock::Text(index)) => {
            if let Some(AssistantContent::Text(block)) = output.content.get(index) {
                sender.push(AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content: block.text.clone(),
                    partial: output.clone(),
                });
            }
        }
        Some(ActiveBlock::Thinking(index)) => {
            if let Some(AssistantContent::Thinking(block)) = output.content.get(index) {
                sender.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content: block.thinking.clone(),
                    partial: output.clone(),
                });
            }
        }
        None => {}
    }
}

fn consume_tool_call_delta(
    tool_call: &Value,
    output: &mut AssistantMessage,
    sender: &mut AssistantMessageEventStreamSender,
    tool_blocks_by_key: &mut HashMap<String, usize>,
    tool_order: &mut Vec<usize>,
    partial_args: &mut HashMap<usize, String>,
) {
    let index_value = tool_call.get("index").and_then(Value::as_i64).unwrap_or(0);
    let call_id = string_field(tool_call, &["id"])
        .filter(|id| *id != "null")
        .map(ToString::to_string)
        .unwrap_or_else(|| derive_mistral_tool_call_id(&format!("toolcall:{index_value}"), 0));
    let key = format!("{call_id}:{index_value}");
    let block_index = if let Some(index) = tool_blocks_by_key.get(&key).copied() {
        index
    } else {
        let index = output.content.len();
        let name = tool_call
            .get("function")
            .and_then(|function| string_field(function, &["name"]))
            .unwrap_or("")
            .to_string();
        output.content.push(AssistantContent::ToolCall(ToolCall {
            id: call_id,
            name,
            arguments: json!({}),
            thought_signature: None,
        }));
        tool_blocks_by_key.insert(key, index);
        tool_order.push(index);
        partial_args.insert(index, String::new());
        sender.push(AssistantMessageEvent::ToolCallStart {
            content_index: index,
            partial: output.clone(),
        });
        index
    };

    if let Some(name) = tool_call
        .get("function")
        .and_then(|function| string_field(function, &["name"]))
        && let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(block_index)
        && block.name.is_empty()
    {
        block.name = name.to_string();
    }

    let args_delta = tool_call
        .get("function")
        .and_then(|function| function.get("arguments"))
        .map(|arguments| match arguments {
            Value::String(arguments) => arguments.clone(),
            Value::Null => String::new(),
            other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
        })
        .unwrap_or_default();
    let partial = partial_args.entry(block_index).or_default();
    partial.push_str(&args_delta);
    if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(block_index) {
        block.arguments = parse_streaming_json(Some(partial));
    }
    sender.push(AssistantMessageEvent::ToolCallDelta {
        content_index: block_index,
        delta: args_delta,
        partial: output.clone(),
    });
}

fn parse_chunk_usage(usage: &Value, model: &Model) -> Usage {
    let input = numeric_field(usage, &["promptTokens", "prompt_tokens"]);
    let output = numeric_field(usage, &["completionTokens", "completion_tokens"]);
    let total_tokens = numeric_field(usage, &["totalTokens", "total_tokens"]).max(input + output);
    let mut usage = Usage {
        input,
        output,
        cache_read: 0,
        cache_write: 0,
        total_tokens,
        cost: Default::default(),
    };
    calculate_cost(model, &mut usage);
    usage
}

fn map_chat_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "length" | "model_length" => StopReason::Length,
        "tool_calls" => StopReason::ToolUse,
        "error" => StopReason::Error,
        _ => StopReason::Stop,
    }
}

fn uses_reasoning_effort(model: &Model) -> bool {
    matches!(
        model.id.as_str(),
        "mistral-small-2603" | "mistral-small-latest" | "mistral-medium-3.5"
    )
}

fn uses_prompt_mode_reasoning(model: &Model) -> bool {
    model.reasoning && !uses_reasoning_effort(model)
}

fn map_reasoning_effort(model: &Model, level: ModelThinkingLevel) -> String {
    model
        .thinking_level_map
        .get(level.as_str())
        .and_then(Clone::clone)
        .unwrap_or_else(|| "high".to_string())
}

fn headers(model: &Model, options: &StreamOptions, api_key: &str) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|e| Error::InvalidHeaderValue("authorization".to_string(), e))?,
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    for (name, value) in model.headers.iter().chain(options.headers.iter()) {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let value = HeaderValue::from_str(value)
            .map_err(|e| Error::InvalidHeaderValue(name.to_string(), e))?;
        headers.insert(name, value);
    }
    if let Some(session_id) = &options.session_id
        && !headers.contains_key("x-affinity")
    {
        headers.insert(
            HeaderName::from_static("x-affinity"),
            HeaderValue::from_str(session_id)
                .map_err(|e| Error::InvalidHeaderValue("x-affinity".to_string(), e))?,
        );
    }
    Ok(headers)
}

fn mistral_chat_completions_url(model: &Model) -> String {
    let base_url = model.base_url.trim_end_matches('/');
    if base_url.ends_with("/v1") {
        format!("{base_url}/chat/completions")
    } else {
        format!("{base_url}/v1/chat/completions")
    }
}

fn to_mistral_api_payload(value: Value) -> Value {
    match value {
        Value::Array(values) => {
            Value::Array(values.into_iter().map(to_mistral_api_payload).collect())
        }
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .map(|(key, value)| {
                    (
                        mistral_api_key(&key).to_string(),
                        to_mistral_api_payload(value),
                    )
                })
                .collect(),
        ),
        other => other,
    }
}

fn mistral_api_key(key: &str) -> &str {
    match key {
        "maxTokens" => "max_tokens",
        "toolChoice" => "tool_choice",
        "promptMode" => "prompt_mode",
        "reasoningEffort" => "reasoning_effort",
        "toolCalls" => "tool_calls",
        "toolCallId" => "tool_call_id",
        "imageUrl" => "image_url",
        other => other,
    }
}

fn string_field<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
}

fn numeric_field(value: &Value, keys: &[&str]) -> u32 {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_u64))
        .unwrap_or(0) as u32
}

fn truncate_error_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated = text.chars().take(max_chars).collect::<String>();
    let omitted = text.chars().count() - max_chars;
    format!("{truncated}... [truncated {omitted} chars]")
}

fn is_cancelled(options: &StreamOptions) -> bool {
    options
        .cancellation_token
        .as_ref()
        .is_some_and(|token| token.is_cancelled())
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

fn env_api_key(provider: &str) -> Option<String> {
    crate::env_api_keys::get_env_api_key(provider)
}

#[cfg(test)]
mod tests {
    use std::io;

    use futures::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    use crate::models::get_model;
    use crate::stream::{complete, complete_simple};
    use crate::types::{
        ModelCost, Tool, ToolResultMessage, UserContent, UserMessage, UserMessageContent,
    };

    use super::*;

    #[derive(Debug)]
    struct CapturedRequest {
        path: String,
        authorization: Option<String>,
        affinity: Option<String>,
        body: Value,
    }

    async fn serve_once(response_body: String) -> (String, oneshot::Receiver<CapturedRequest>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let (sender, receiver) = oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept request");
            let mut buffer = Vec::new();
            let header_end;
            loop {
                let mut chunk = [0_u8; 1024];
                let n = socket.read(&mut chunk).await.expect("read request");
                if n == 0 {
                    panic!("connection closed before headers");
                }
                buffer.extend_from_slice(&chunk[..n]);
                if let Some(index) = find_subsequence(&buffer, b"\r\n\r\n") {
                    header_end = index + 4;
                    break;
                }
            }

            let header_text = String::from_utf8_lossy(&buffer[..header_end]).to_string();
            let content_length = header_text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            while buffer.len() < header_end + content_length {
                let mut chunk = [0_u8; 1024];
                let n = socket.read(&mut chunk).await.expect("read body");
                if n == 0 {
                    break;
                }
                buffer.extend_from_slice(&chunk[..n]);
            }

            let request_line = header_text.lines().next().expect("request line");
            let path = request_line
                .split_whitespace()
                .nth(1)
                .expect("request path")
                .to_string();
            let authorization = header_text.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("authorization")
                    .then(|| value.trim().to_string())
            });
            let affinity = header_text.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("x-affinity")
                    .then(|| value.trim().to_string())
            });
            let body =
                serde_json::from_slice::<Value>(&buffer[header_end..header_end + content_length])
                    .expect("request json");
            sender
                .send(CapturedRequest {
                    path,
                    authorization,
                    affinity,
                    body,
                })
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "receiver dropped"))
                .expect("send captured request");

            let http_response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket
                .write_all(http_response.as_bytes())
                .await
                .expect("write response");
        });

        (format!("http://{addr}"), receiver)
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    fn mistral_model(id: &str) -> Model {
        Model {
            id: id.to_string(),
            name: id.to_string(),
            api: "mistral-conversations".to_string(),
            provider: "mistral".to_string(),
            base_url: "http://127.0.0.1:9".to_string(),
            reasoning: id.starts_with("magistral")
                || id == "mistral-small-2603"
                || id == "mistral-medium-3.5",
            input: vec![ModelInput::Text, ModelInput::Image],
            cost: ModelCost {
                input: 1.0,
                output: 2.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            context_window: 128_000,
            max_tokens: 16_384,
            ..Model::default()
        }
    }

    #[tokio::test]
    async fn stream_simple_selects_mistral_reasoning_controls() {
        let response = complete_simple(
            mistral_model("mistral-small-2603"),
            Context {
                messages: vec![Message::user_text("Hello")],
                ..Context::default()
            },
            Some(SimpleStreamOptions {
                reasoning: Some(ModelThinkingLevel::Medium),
                stream: StreamOptions {
                    api_key: Some("fake-key".to_string()),
                    on_payload: Some(std::sync::Arc::new(|payload, _model| {
                        assert_eq!(payload["reasoningEffort"], "high");
                        assert!(payload.get("promptMode").is_none());
                        Box::pin(async move { Ok(Some(payload)) })
                    })),
                    ..StreamOptions::default()
                },
                ..SimpleStreamOptions::default()
            }),
        )
        .await
        .expect("stream result");
        assert_eq!(response.stop_reason, StopReason::Error);

        let mut stream = stream_simple_mistral(
            mistral_model("magistral-medium-latest"),
            Context {
                messages: vec![Message::user_text("Hello")],
                ..Context::default()
            },
            SimpleStreamOptions {
                reasoning: Some(ModelThinkingLevel::Medium),
                stream: StreamOptions {
                    api_key: Some("fake-key".to_string()),
                    on_payload: Some(std::sync::Arc::new(|payload, _model| {
                        assert_eq!(payload["promptMode"], "reasoning");
                        assert!(payload.get("reasoningEffort").is_none());
                        Box::pin(async move { Ok(Some(payload)) })
                    })),
                    ..StreamOptions::default()
                },
                ..SimpleStreamOptions::default()
            },
        );
        while stream.next().await.is_some() {}
        let _ = stream.result().await.expect("stream result");

        let mut stream = stream_simple_mistral(
            mistral_model("mistral-medium-3.5"),
            Context {
                messages: vec![Message::user_text("Hello")],
                ..Context::default()
            },
            SimpleStreamOptions {
                reasoning: Some(ModelThinkingLevel::Medium),
                stream: StreamOptions {
                    api_key: Some("fake-key".to_string()),
                    on_payload: Some(std::sync::Arc::new(|payload, _model| {
                        assert_eq!(payload["reasoningEffort"], "high");
                        assert!(payload.get("promptMode").is_none());
                        Box::pin(async move { Ok(Some(payload)) })
                    })),
                    ..StreamOptions::default()
                },
                ..SimpleStreamOptions::default()
            },
        );
        while stream.next().await.is_some() {}
        let _ = stream.result().await.expect("stream result");
    }

    #[tokio::test]
    async fn streams_text_thinking_tool_calls_usage_and_sends_api_payload() {
        let body = [
            r#"data: {"id":"chat_1","model":"devstral-medium-latest","choices":[{"delta":{"content":"Hello "}}]}"#,
            "",
            r#"data: {"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"plan"}]},{"type":"text","text":"world"}]}}]}"#,
            "",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"toolabc12","function":{"name":"echo","arguments":"{\"text\""}}]}}]}"#,
            "",
            r#"data: {"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7},"choices":[{"finish_reason":"tool_calls","delta":{"tool_calls":[{"index":0,"id":"toolabc12","function":{"arguments":":\"hi\"}"}}]}}]}"#,
            "",
            "data: [DONE]",
            "",
        ]
        .join("\n");
        let (base_url, request_receiver) = serve_once(body).await;
        let mut model = mistral_model("devstral-medium-latest");
        model.base_url = base_url;
        let mut stream = stream_mistral(
            model,
            Context {
                system_prompt: Some("Be concise.".to_string()),
                messages: vec![
                    Message::User(UserMessage {
                        content: UserMessageContent::Parts(vec![
                            UserContent::text("Describe this."),
                            UserContent::Image(ImageContent {
                                data: "YWJj".to_string(),
                                mime_type: "image/png".to_string(),
                            }),
                        ]),
                        timestamp: 1,
                    }),
                    Message::ToolResult(ToolResultMessage {
                        tool_call_id: "foreign tool id".to_string(),
                        tool_name: "lookup".to_string(),
                        content: vec![ToolResultContent::text("tool text")],
                        details: None,
                        is_error: false,
                        timestamp: 2,
                    }),
                ],
                tools: vec![Tool {
                    name: "echo".to_string(),
                    description: "Echo text".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": { "text": { "type": "string" } },
                    }),
                }],
            },
            MistralOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    session_id: Some("session-1".to_string()),
                    max_tokens: Some(32),
                    ..StreamOptions::default()
                },
                tool_choice: Some(json!("auto")),
                prompt_mode: Some("reasoning".to_string()),
                reasoning_effort: None,
            },
        );

        while stream.next().await.is_some() {}
        let message = stream.result().await.expect("stream result");

        assert_eq!(message.response_id.as_deref(), Some("chat_1"));
        assert_eq!(message.response_model.as_deref(), None);
        assert_eq!(message.stop_reason, StopReason::ToolUse);
        assert_eq!(message.usage.input, 3);
        assert_eq!(message.usage.output, 4);
        assert_eq!(message.usage.total_tokens, 7);
        assert_eq!(
            message.content,
            vec![
                AssistantContent::Text(TextContent {
                    text: "Hello ".to_string(),
                    text_signature: None,
                }),
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "plan".to_string(),
                    thinking_signature: None,
                    redacted: None,
                }),
                AssistantContent::Text(TextContent {
                    text: "world".to_string(),
                    text_signature: None,
                }),
                AssistantContent::ToolCall(ToolCall {
                    id: "toolabc12".to_string(),
                    name: "echo".to_string(),
                    arguments: json!({ "text": "hi" }),
                    thought_signature: None,
                }),
            ]
        );

        let request = request_receiver.await.expect("captured request");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.authorization.as_deref(), Some("Bearer test-key"));
        assert_eq!(request.affinity.as_deref(), Some("session-1"));
        assert_eq!(request.body["max_tokens"], 32);
        assert_eq!(request.body["tool_choice"], "auto");
        assert_eq!(request.body["prompt_mode"], "reasoning");
        assert_eq!(request.body["messages"][0]["role"], "system");
        assert_eq!(
            request.body["messages"][1]["content"][1]["image_url"],
            "data:image/png;base64,YWJj"
        );
        assert_eq!(request.body["tools"][0]["function"]["strict"], false);
    }

    #[tokio::test]
    async fn registered_builtin_dispatches_mistral_provider() {
        let mut model = get_model("mistral", "devstral-medium-latest").expect("mistral model");
        model.base_url = "http://127.0.0.1:9".to_string();
        let message = complete(
            model,
            Context {
                messages: vec![Message::user_text("Hello")],
                ..Context::default()
            },
            Some(StreamOptions {
                api_key: Some("fake-key".to_string()),
                ..StreamOptions::default()
            }),
        )
        .await
        .expect("mistral result");

        assert_eq!(message.stop_reason, StopReason::Error);
        assert!(crate::api_registry::get_api_provider("mistral-conversations").is_some());
    }
}
