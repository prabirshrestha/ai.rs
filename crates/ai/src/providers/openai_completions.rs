use std::collections::HashMap;

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
    AssistantContent, AssistantMessage, AssistantMessageEvent, CacheControlFormat, CacheRetention,
    Context, ImageContent, MaxTokensField, Model, ModelInput, ModelThinkingLevel,
    OpenAIThinkingFormat, SimpleStreamOptions, StopReason, StreamOptions, TextContent,
    ThinkingContent, Tool, ToolCall, ToolResultContent, Usage, UserContent, UserMessageContent,
};
use crate::utils::http::{request_timeout, send_with_retries};
use crate::utils::json::parse_streaming_json;
use crate::utils::sanitize::sanitize_surrogates;
use crate::utils::sse;
use crate::{Error, Result};

#[derive(Clone, Default)]
pub struct OpenAICompletionsOptions {
    pub base: StreamOptions,
    pub tool_choice: Option<Value>,
    pub reasoning_effort: Option<ModelThinkingLevel>,
}

#[derive(Debug, Clone)]
pub struct ResolvedOpenAICompletionsCompat {
    pub supports_store: bool,
    pub supports_developer_role: bool,
    pub supports_reasoning_effort: bool,
    pub supports_usage_in_streaming: bool,
    pub max_tokens_field: MaxTokensField,
    pub requires_tool_result_name: bool,
    pub requires_assistant_after_tool_result: bool,
    pub requires_thinking_as_text: bool,
    pub requires_reasoning_content_on_assistant_messages: bool,
    pub thinking_format: OpenAIThinkingFormat,
    pub open_router_routing: Option<Value>,
    pub vercel_gateway_routing: Option<Value>,
    pub zai_tool_stream: bool,
    pub supports_strict_mode: bool,
    pub cache_control_format: Option<CacheControlFormat>,
    pub send_session_affinity_headers: bool,
    pub supports_long_cache_retention: bool,
}

pub fn stream_simple_openai_completions(
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
        let provider = model.provider.clone();
        return immediate_error(model, &format!("No API key for provider: {provider}"));
    };

    let base = build_base_options(&model, &options, api_key);
    let reasoning_effort = options.reasoning.and_then(|reasoning| {
        let clamped = clamp_thinking_level(&model, reasoning);
        (clamped != ModelThinkingLevel::Off).then_some(clamped)
    });

    stream_openai_completions(
        model,
        context,
        OpenAICompletionsOptions {
            base,
            tool_choice: simple_tool_choice(&options),
            reasoning_effort,
        },
    )
}

fn simple_tool_choice(options: &SimpleStreamOptions) -> Option<Value> {
    options.stream.provider_options.get("toolChoice").cloned()
}

pub fn stream_openai_completions(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions,
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
    options: OpenAICompletionsOptions,
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
    let mut payload =
        build_chat_completions_payload(&model, &context, &options, &compat, cache_retention);
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
    let request_url = format!("{}/chat/completions", trim_end_slash(&base_url));
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
    let client = reqwest::Client::new();
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

    let mut text_block: Option<usize> = None;
    let mut thinking_block: Option<usize> = None;
    let mut has_finish_reason = false;
    let mut tool_blocks_by_index: HashMap<i64, usize> = HashMap::new();
    let mut tool_blocks_by_id: HashMap<String, usize> = HashMap::new();
    let mut partial_args: HashMap<usize, String> = HashMap::new();

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
        if event.data.trim() == "[DONE]" || event.data.trim().is_empty() {
            continue;
        }
        let chunk: Value = match serde_json::from_str(&event.data) {
            Ok(chunk) => chunk,
            Err(error) => return Err(StreamFailure::new(output, error)),
        };

        if let Some(id) = chunk
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            output.response_id.get_or_insert_with(|| id.to_string());
        }
        if let Some(response_model) = chunk
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
            && response_model != model.id
            && output.response_model.is_none()
        {
            output.response_model = Some(response_model.to_string());
        }
        let chunk_has_usage =
            if let Some(usage) = chunk.get("usage").filter(|value| !value.is_null()) {
                output.usage = parse_chunk_usage(usage, &model);
                true
            } else {
                false
            };

        let choice = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first());
        let Some(choice) = choice else {
            continue;
        };
        if !chunk_has_usage
            && let Some(usage) = choice.get("usage").filter(|value| !value.is_null())
        {
            output.usage = parse_chunk_usage(usage, &model);
        }

        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            let mapped = map_stop_reason(reason);
            output.stop_reason = mapped.0;
            output.error_message = mapped.1;
            has_finish_reason = true;
        }

        let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
            continue;
        };

        if let Some(content) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            let index = ensure_text_block(&mut output, &mut text_block, sender);
            if let Some(AssistantContent::Text(block)) = output.content.get_mut(index) {
                block.text.push_str(content);
            }
            sender.push(AssistantMessageEvent::TextDelta {
                content_index: index,
                delta: content.to_string(),
                partial: output.clone(),
            });
        }

        let reasoning_field = ["reasoning_content", "reasoning", "reasoning_text"]
            .iter()
            .find_map(|field| {
                delta
                    .get(*field)
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(|value| (*field, value))
            });
        if let Some((field, reasoning)) = reasoning_field {
            let index = ensure_thinking_block(&mut output, &mut thinking_block, field, sender);
            if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(index) {
                block.thinking.push_str(reasoning);
            }
            sender.push(AssistantMessageEvent::ThinkingDelta {
                content_index: index,
                delta: reasoning.to_string(),
                partial: output.clone(),
            });
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call_delta in tool_calls {
                let index = ensure_tool_call_block(
                    &mut output,
                    tool_call_delta,
                    &mut tool_blocks_by_index,
                    &mut tool_blocks_by_id,
                    sender,
                );

                if let Some(id) = tool_call_delta
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(index)
                        && block.id.is_empty()
                    {
                        block.id = id.to_string();
                    }
                    tool_blocks_by_id.insert(id.to_string(), index);
                }
                if let Some(name) = tool_call_delta
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    && let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(index)
                    && block.name.is_empty()
                {
                    block.name = name.to_string();
                }

                let delta_args = tool_call_delta
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if !delta_args.is_empty() {
                    let entry = partial_args.entry(index).or_default();
                    entry.push_str(delta_args);
                    if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(index) {
                        block.arguments = parse_streaming_json(Some(entry));
                    }
                }
                sender.push(AssistantMessageEvent::ToolCallDelta {
                    content_index: index,
                    delta: delta_args.to_string(),
                    partial: output.clone(),
                });
            }
        }

        if let Some(details) = delta.get("reasoning_details").and_then(Value::as_array) {
            for detail in details {
                if detail.get("type").and_then(Value::as_str) != Some("reasoning.encrypted") {
                    continue;
                }
                let Some(id) = detail.get("id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(data) = detail.get("data") else {
                    continue;
                };
                if data.is_null() {
                    continue;
                }
                for block in output.content.iter_mut() {
                    if let AssistantContent::ToolCall(tool_call) = block
                        && tool_call.id == id
                    {
                        tool_call.thought_signature = Some(detail.to_string());
                    }
                }
            }
        }
    }

    finish_open_blocks(&mut output, &partial_args, sender);

    if options
        .base
        .cancellation_token
        .as_ref()
        .is_some_and(|token| token.is_cancelled())
    {
        return Err(StreamFailure::cancelled(output));
    }
    if output.stop_reason == StopReason::Aborted {
        return Err(StreamFailure::cancelled(output));
    }
    if output.stop_reason == StopReason::Error {
        return Err(StreamFailure::new(
            output.clone(),
            output
                .error_message
                .clone()
                .unwrap_or_else(|| "Provider returned an error stop reason".to_string()),
        ));
    }
    if !has_finish_reason {
        return Err(StreamFailure::new(
            output,
            "Stream ended without finish_reason",
        ));
    }

    sender.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output,
    });
    Ok(())
}

fn ensure_text_block(
    output: &mut AssistantMessage,
    text_block: &mut Option<usize>,
    sender: &mut AssistantMessageEventStreamSender,
) -> usize {
    if let Some(index) = *text_block {
        return index;
    }
    output.content.push(AssistantContent::Text(TextContent {
        text: String::new(),
        text_signature: None,
    }));
    let index = output.content.len() - 1;
    *text_block = Some(index);
    sender.push(AssistantMessageEvent::TextStart {
        content_index: index,
        partial: output.clone(),
    });
    index
}

fn ensure_thinking_block(
    output: &mut AssistantMessage,
    thinking_block: &mut Option<usize>,
    signature: &str,
    sender: &mut AssistantMessageEventStreamSender,
) -> usize {
    if let Some(index) = *thinking_block {
        return index;
    }
    output
        .content
        .push(AssistantContent::Thinking(ThinkingContent {
            thinking: String::new(),
            thinking_signature: Some(signature.to_string()),
            redacted: None,
        }));
    let index = output.content.len() - 1;
    *thinking_block = Some(index);
    sender.push(AssistantMessageEvent::ThinkingStart {
        content_index: index,
        partial: output.clone(),
    });
    index
}

fn ensure_tool_call_block(
    output: &mut AssistantMessage,
    tool_call_delta: &Value,
    by_index: &mut HashMap<i64, usize>,
    by_id: &mut HashMap<String, usize>,
    sender: &mut AssistantMessageEventStreamSender,
) -> usize {
    let stream_index = tool_call_delta.get("index").and_then(Value::as_i64);
    if let Some(index) = stream_index.and_then(|index| by_index.get(&index).copied()) {
        return index;
    }
    if let Some(id) = tool_call_delta.get("id").and_then(Value::as_str)
        && let Some(index) = by_id.get(id).copied()
    {
        if let Some(stream_index) = stream_index {
            by_index.insert(stream_index, index);
        }
        return index;
    }

    let id = tool_call_delta
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let name = tool_call_delta
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    output.content.push(AssistantContent::ToolCall(ToolCall {
        id: id.clone(),
        name,
        arguments: Value::Object(Default::default()),
        thought_signature: None,
    }));
    let index = output.content.len() - 1;
    if let Some(stream_index) = stream_index {
        by_index.insert(stream_index, index);
    }
    if !id.is_empty() {
        by_id.insert(id, index);
    }
    sender.push(AssistantMessageEvent::ToolCallStart {
        content_index: index,
        partial: output.clone(),
    });
    index
}

fn finish_open_blocks(
    output: &mut AssistantMessage,
    partial_args: &HashMap<usize, String>,
    sender: &mut AssistantMessageEventStreamSender,
) {
    for index in 0..output.content.len() {
        match output.content.get_mut(index) {
            Some(AssistantContent::Text(block)) => {
                sender.push(AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content: block.text.clone(),
                    partial: output.clone(),
                });
            }
            Some(AssistantContent::Thinking(block)) => {
                sender.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content: block.thinking.clone(),
                    partial: output.clone(),
                });
            }
            Some(AssistantContent::ToolCall(block)) => {
                if let Some(args) = partial_args.get(&index) {
                    block.arguments = parse_streaming_json(Some(args));
                }
                sender.push(AssistantMessageEvent::ToolCallEnd {
                    content_index: index,
                    tool_call: block.clone(),
                    partial: output.clone(),
                });
            }
            None => {}
        }
    }
}

fn build_chat_completions_payload(
    model: &Model,
    context: &Context,
    options: &OpenAICompletionsOptions,
    compat: &ResolvedOpenAICompletionsCompat,
    cache_retention: CacheRetention,
) -> Value {
    let messages = convert_messages(model, context, compat);
    let mut payload = json!({
        "model": model.id,
        "messages": messages,
        "stream": true
    });
    let object = payload.as_object_mut().expect("payload object");

    if compat.supports_usage_in_streaming {
        object.insert(
            "stream_options".to_string(),
            json!({ "include_usage": true }),
        );
    }
    if compat.supports_store {
        object.insert("store".to_string(), json!(false));
    }
    if let Some(max_tokens) = options.base.max_tokens.filter(|max_tokens| *max_tokens > 0) {
        let field = match compat.max_tokens_field {
            MaxTokensField::MaxTokens => "max_tokens",
            MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
        };
        object.insert(field.to_string(), json!(max_tokens));
    }
    if let Some(temperature) = options.base.temperature {
        object.insert("temperature".to_string(), json!(temperature));
    }
    if !context.tools.is_empty() {
        object.insert(
            "tools".to_string(),
            Value::Array(convert_tools(&context.tools, compat)),
        );
        if compat.zai_tool_stream {
            object.insert("tool_stream".to_string(), json!(true));
        }
    } else if has_tool_history(&context.messages) {
        object.insert("tools".to_string(), Value::Array(Vec::new()));
    }
    if let Some(tool_choice) = &options.tool_choice {
        object.insert("tool_choice".to_string(), tool_choice.clone());
    }

    apply_reasoning_options(object, model, options, compat);

    if (model.base_url.contains("api.openai.com") && cache_retention != CacheRetention::None
        || (cache_retention == CacheRetention::Long && compat.supports_long_cache_retention))
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

    if model.base_url.contains("openrouter.ai")
        && let Some(routing) = &compat.open_router_routing
    {
        object.insert("provider".to_string(), routing.clone());
    }
    if model.base_url.contains("ai-gateway.vercel.sh")
        && let Some(routing) = &compat.vercel_gateway_routing
        && let Some(gateway_options) = gateway_provider_options(routing)
    {
        object.insert(
            "providerOptions".to_string(),
            json!({ "gateway": gateway_options }),
        );
    }

    if let Some(cache_control) = compat_cache_control(compat, cache_retention) {
        apply_anthropic_cache_control(&mut payload, cache_control);
    }

    payload
}

fn gateway_provider_options(routing: &Value) -> Option<Value> {
    let mut gateway = serde_json::Map::new();
    for key in ["only", "order"] {
        if let Some(value) = routing.get(key).filter(|value| !value.is_null()) {
            gateway.insert(key.to_string(), value.clone());
        }
    }
    (!gateway.is_empty()).then_some(Value::Object(gateway))
}

fn apply_reasoning_options(
    object: &mut serde_json::Map<String, Value>,
    model: &Model,
    options: &OpenAICompletionsOptions,
    compat: &ResolvedOpenAICompletionsCompat,
) {
    if !model.reasoning {
        return;
    }
    let effort = options
        .reasoning_effort
        .filter(|effort| *effort != ModelThinkingLevel::Off);
    let mapped_effort = effort
        .and_then(|effort| {
            model
                .thinking_level_map
                .get(effort.as_str())
                .cloned()
                .flatten()
        })
        .or_else(|| effort.map(|effort| effort.as_str().to_string()));

    match compat.thinking_format {
        OpenAIThinkingFormat::Zai | OpenAIThinkingFormat::Qwen => {
            object.insert(
                "enable_thinking".to_string(),
                json!(mapped_effort.is_some()),
            );
        }
        OpenAIThinkingFormat::QwenChatTemplate => {
            object.insert(
                "chat_template_kwargs".to_string(),
                json!({ "enable_thinking": mapped_effort.is_some(), "preserve_thinking": true }),
            );
        }
        OpenAIThinkingFormat::Deepseek => {
            object.insert(
                "thinking".to_string(),
                json!({ "type": if mapped_effort.is_some() { "enabled" } else { "disabled" } }),
            );
            if let Some(effort) = mapped_effort.filter(|_| compat.supports_reasoning_effort) {
                object.insert("reasoning_effort".to_string(), json!(effort));
            }
        }
        OpenAIThinkingFormat::Openrouter => {
            if let Some(effort) = mapped_effort {
                object.insert("reasoning".to_string(), json!({ "effort": effort }));
            } else if model.thinking_level_map.get("off") != Some(&None) {
                object.insert(
                    "reasoning".to_string(),
                    json!({ "effort": model.thinking_level_map.get("off").and_then(Clone::clone).unwrap_or_else(|| "none".to_string()) }),
                );
            }
        }
        OpenAIThinkingFormat::Together => {
            object.insert(
                "reasoning".to_string(),
                json!({ "enabled": mapped_effort.is_some() }),
            );
            if let Some(effort) = mapped_effort.filter(|_| compat.supports_reasoning_effort) {
                object.insert("reasoning_effort".to_string(), json!(effort));
            }
        }
        OpenAIThinkingFormat::StringThinking => {
            if let Some(effort) = mapped_effort {
                object.insert("thinking".to_string(), json!(effort));
            } else if model.thinking_level_map.get("off") != Some(&None) {
                object.insert(
                    "thinking".to_string(),
                    json!(
                        model
                            .thinking_level_map
                            .get("off")
                            .and_then(Clone::clone)
                            .unwrap_or_else(|| "none".to_string())
                    ),
                );
            }
        }
        OpenAIThinkingFormat::Openai => {
            if let Some(effort) = mapped_effort.filter(|_| compat.supports_reasoning_effort) {
                object.insert("reasoning_effort".to_string(), json!(effort));
            } else if compat.supports_reasoning_effort
                && let Some(Some(off)) = model.thinking_level_map.get("off")
            {
                object.insert("reasoning_effort".to_string(), json!(off));
            }
        }
    }
}

pub fn convert_messages(
    model: &Model,
    context: &Context,
    compat: &ResolvedOpenAICompletionsCompat,
) -> Vec<Value> {
    let mut params = Vec::new();
    let transformed = transform_messages(&context.messages, model, |id, target_model, _source| {
        normalize_chat_tool_call_id(id, target_model)
    });

    if let Some(system_prompt) = &context.system_prompt
        && !system_prompt.is_empty()
    {
        let role = if model.reasoning && compat.supports_developer_role {
            "developer"
        } else {
            "system"
        };
        params.push(json!({ "role": role, "content": sanitize_surrogates(system_prompt) }));
    }

    let mut last_role: Option<&str> = None;
    let mut index = 0usize;
    while index < transformed.len() {
        let msg = &transformed[index];
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(msg, crate::types::Message::User(_))
        {
            params.push(
                json!({ "role": "assistant", "content": "I have processed the tool results." }),
            );
        }

        match msg {
            crate::types::Message::User(user) => match &user.content {
                UserMessageContent::Text(text) => {
                    params.push(json!({ "role": "user", "content": sanitize_surrogates(text) }));
                    last_role = Some("user");
                }
                UserMessageContent::Parts(parts) => {
                    let content: Vec<Value> = parts
                            .iter()
                            .map(|part| match part {
                                UserContent::Text(text) => {
                                    json!({ "type": "text", "text": sanitize_surrogates(&text.text) })
                                }
                                UserContent::Image(image) => json!({
                                    "type": "image_url",
                                    "image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) }
                                }),
                            })
                            .collect();
                    if !content.is_empty() {
                        params.push(json!({ "role": "user", "content": content }));
                        last_role = Some("user");
                    }
                }
            },
            crate::types::Message::Assistant(assistant) => {
                let mut assistant_msg = json!({
                    "role": "assistant",
                    "content": if compat.requires_assistant_after_tool_result { json!("") } else { Value::Null },
                });
                let assistant_obj = assistant_msg.as_object_mut().expect("assistant object");
                let text_parts: Vec<Value> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Text(text) if !text.text.trim().is_empty() => {
                            Some(json!({ "type": "text", "text": sanitize_surrogates(&text.text) }))
                        }
                        _ => None,
                    })
                    .collect();
                let assistant_text = text_parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<String>();
                let thinking_blocks: Vec<_> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Thinking(thinking)
                            if !thinking.thinking.trim().is_empty() =>
                        {
                            Some(thinking)
                        }
                        _ => None,
                    })
                    .collect();
                if !thinking_blocks.is_empty() {
                    if compat.requires_thinking_as_text {
                        let thinking_text = thinking_blocks
                            .iter()
                            .map(|block| sanitize_surrogates(&block.thinking))
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        let mut content = vec![json!({ "type": "text", "text": thinking_text })];
                        content.extend(text_parts);
                        assistant_obj.insert("content".to_string(), Value::Array(content));
                    } else {
                        if !assistant_text.is_empty() {
                            assistant_obj.insert("content".to_string(), json!(assistant_text));
                        }
                        let signature = thinking_blocks
                            .first()
                            .and_then(|block| block.thinking_signature.as_deref());
                        if let Some(signature) = signature.filter(|signature| !signature.is_empty())
                        {
                            assistant_obj.insert(
                                signature.to_string(),
                                json!(
                                    thinking_blocks
                                        .iter()
                                        .map(|block| block.thinking.as_str())
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                ),
                            );
                        }
                    }
                } else if !assistant_text.is_empty() {
                    assistant_obj.insert("content".to_string(), json!(assistant_text));
                }

                let tool_calls: Vec<Value> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::ToolCall(tool_call) => Some(json!({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string())
                            }
                        })),
                        _ => None,
                    })
                    .collect();
                if !tool_calls.is_empty() {
                    assistant_obj.insert("tool_calls".to_string(), Value::Array(tool_calls));
                    let reasoning_details: Vec<Value> = assistant
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            AssistantContent::ToolCall(tool_call) => tool_call
                                .thought_signature
                                .as_deref()
                                .and_then(|raw| serde_json::from_str(raw).ok()),
                            _ => None,
                        })
                        .collect();
                    if !reasoning_details.is_empty() {
                        assistant_obj.insert(
                            "reasoning_details".to_string(),
                            Value::Array(reasoning_details),
                        );
                    }
                }
                if compat.requires_reasoning_content_on_assistant_messages
                    && model.reasoning
                    && !assistant_obj.contains_key("reasoning_content")
                {
                    assistant_obj.insert("reasoning_content".to_string(), json!(""));
                }
                let has_content = assistant_obj
                    .get("content")
                    .filter(|content| !content.is_null())
                    .is_some_and(|content| match content {
                        Value::String(text) => !text.is_empty(),
                        Value::Array(parts) => !parts.is_empty(),
                        _ => true,
                    });
                if has_content || assistant_obj.contains_key("tool_calls") {
                    params.push(assistant_msg);
                    last_role = Some("assistant");
                }
            }
            crate::types::Message::ToolResult(_) => {
                let mut image_blocks = Vec::new();
                let mut cursor = index;
                while cursor < transformed.len() {
                    let crate::types::Message::ToolResult(tool_msg) = &transformed[cursor] else {
                        break;
                    };
                    let text_result = tool_msg
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ToolResultContent::Text(text) => Some(text.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let has_text = !text_result.is_empty();
                    let mut tool_result = json!({
                        "role": "tool",
                        "content": sanitize_surrogates(if has_text { &text_result } else { "(see attached image)" }),
                        "tool_call_id": tool_msg.tool_call_id
                    });
                    if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
                        tool_result["name"] = json!(tool_msg.tool_name);
                    }
                    params.push(tool_result);

                    if model.input.contains(&ModelInput::Image) {
                        for block in &tool_msg.content {
                            if let ToolResultContent::Image(ImageContent { data, mime_type }) =
                                block
                            {
                                image_blocks.push(json!({
                                    "type": "image_url",
                                    "image_url": { "url": format!("data:{mime_type};base64,{data}") }
                                }));
                            }
                        }
                    }
                    cursor += 1;
                }
                index = cursor - 1;
                if !image_blocks.is_empty() {
                    if compat.requires_assistant_after_tool_result {
                        params.push(json!({ "role": "assistant", "content": "I have processed the tool results." }));
                    }
                    let mut content = vec![
                        json!({ "type": "text", "text": "Attached image(s) from tool result:" }),
                    ];
                    content.extend(image_blocks);
                    params.push(json!({ "role": "user", "content": content }));
                    last_role = Some("user");
                } else {
                    last_role = Some("toolResult");
                }
            }
            crate::types::Message::Custom(_) => {}
        }
        index += 1;
    }

    params
}

fn convert_tools(tools: &[Tool], compat: &ResolvedOpenAICompletionsCompat) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let mut function = json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            });
            if compat.supports_strict_mode {
                function["strict"] = json!(false);
            }
            json!({ "type": "function", "function": function })
        })
        .collect()
}

fn parse_chunk_usage(raw: &Value, model: &Model) -> Usage {
    let prompt_tokens = raw
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let completion_tokens = raw
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let prompt_details = raw.get("prompt_tokens_details");
    let cache_read = prompt_details
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| raw.get("prompt_cache_hit_tokens").and_then(Value::as_u64))
        .unwrap_or(0) as u32;
    let cache_write = prompt_details
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let mut usage = Usage {
        input: prompt_tokens
            .saturating_sub(cache_read)
            .saturating_sub(cache_write),
        output: completion_tokens,
        cache_read,
        cache_write,
        total_tokens: prompt_tokens + completion_tokens,
        cost: Default::default(),
    };
    usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
    calculate_cost(model, &mut usage);
    usage
}

fn map_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        "content_filter" => (
            StopReason::Error,
            Some("Provider finish_reason: content_filter".to_string()),
        ),
        "network_error" => (
            StopReason::Error,
            Some("Provider finish_reason: network_error".to_string()),
        ),
        other => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {other}")),
        ),
    }
}

fn detect_compat(_model: &Model) -> ResolvedOpenAICompletionsCompat {
    ResolvedOpenAICompletionsCompat {
        supports_store: true,
        supports_developer_role: true,
        supports_reasoning_effort: true,
        supports_usage_in_streaming: true,
        max_tokens_field: MaxTokensField::MaxCompletionTokens,
        requires_tool_result_name: false,
        requires_assistant_after_tool_result: false,
        requires_thinking_as_text: false,
        requires_reasoning_content_on_assistant_messages: false,
        thinking_format: OpenAIThinkingFormat::Openai,
        open_router_routing: None,
        vercel_gateway_routing: None,
        zai_tool_stream: false,
        supports_strict_mode: true,
        cache_control_format: None,
        send_session_affinity_headers: false,
        supports_long_cache_retention: true,
    }
}

fn get_compat(model: &Model) -> ResolvedOpenAICompletionsCompat {
    let detected = detect_compat(model);
    let compat = &model.compat.openai_completions;
    ResolvedOpenAICompletionsCompat {
        supports_store: compat.supports_store.unwrap_or(detected.supports_store),
        supports_developer_role: compat
            .supports_developer_role
            .unwrap_or(detected.supports_developer_role),
        supports_reasoning_effort: compat
            .supports_reasoning_effort
            .unwrap_or(detected.supports_reasoning_effort),
        supports_usage_in_streaming: compat
            .supports_usage_in_streaming
            .unwrap_or(detected.supports_usage_in_streaming),
        max_tokens_field: compat.max_tokens_field.unwrap_or(detected.max_tokens_field),
        requires_tool_result_name: compat
            .requires_tool_result_name
            .unwrap_or(detected.requires_tool_result_name),
        requires_assistant_after_tool_result: compat
            .requires_assistant_after_tool_result
            .unwrap_or(detected.requires_assistant_after_tool_result),
        requires_thinking_as_text: compat
            .requires_thinking_as_text
            .unwrap_or(detected.requires_thinking_as_text),
        requires_reasoning_content_on_assistant_messages: compat
            .requires_reasoning_content_on_assistant_messages
            .unwrap_or(detected.requires_reasoning_content_on_assistant_messages),
        thinking_format: compat.thinking_format.unwrap_or(detected.thinking_format),
        open_router_routing: compat
            .open_router_routing
            .clone()
            .or(detected.open_router_routing),
        vercel_gateway_routing: compat
            .vercel_gateway_routing
            .clone()
            .or(detected.vercel_gateway_routing),
        zai_tool_stream: compat.zai_tool_stream.unwrap_or(detected.zai_tool_stream),
        supports_strict_mode: compat
            .supports_strict_mode
            .unwrap_or(detected.supports_strict_mode),
        cache_control_format: compat
            .cache_control_format
            .or(detected.cache_control_format),
        send_session_affinity_headers: compat
            .send_session_affinity_headers
            .unwrap_or(detected.send_session_affinity_headers),
        supports_long_cache_retention: compat
            .supports_long_cache_retention
            .unwrap_or(detected.supports_long_cache_retention),
    }
}

fn has_tool_history(messages: &[crate::types::Message]) -> bool {
    messages.iter().any(|message| match message {
        crate::types::Message::ToolResult(_) => true,
        crate::types::Message::Assistant(assistant) => assistant
            .content
            .iter()
            .any(|block| matches!(block, AssistantContent::ToolCall(_))),
        _ => false,
    })
}

fn normalize_chat_tool_call_id(id: &str, model: &Model) -> String {
    if let Some((call_id, _)) = id.split_once('|') {
        return sanitize_id(call_id).chars().take(40).collect();
    }
    if model.provider == "openai" && id.len() > 40 {
        id.chars().take(40).collect()
    } else {
        id.to_string()
    }
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn resolve_cache_retention(cache_retention: Option<CacheRetention>) -> CacheRetention {
    cache_retention
        .or_else(|| {
            (std::env::var("PI_CACHE_RETENTION").ok().as_deref() == Some("long"))
                .then_some(CacheRetention::Long)
        })
        .unwrap_or(CacheRetention::Short)
}

fn compat_cache_control(
    compat: &ResolvedOpenAICompletionsCompat,
    cache_retention: CacheRetention,
) -> Option<Value> {
    if compat.cache_control_format != Some(CacheControlFormat::Anthropic)
        || cache_retention == CacheRetention::None
    {
        return None;
    }
    let mut value = json!({ "type": "ephemeral" });
    if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
        value["ttl"] = json!("1h");
    }
    Some(value)
}

fn apply_anthropic_cache_control(payload: &mut Value, cache_control: Value) {
    let Some(messages) = payload.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for message in messages.iter_mut() {
        if matches!(
            message.get("role").and_then(Value::as_str),
            Some("system" | "developer")
        ) {
            add_cache_control_to_text_content(message, cache_control.clone());
            break;
        }
    }
    for message in messages.iter_mut().rev() {
        if matches!(
            message.get("role").and_then(Value::as_str),
            Some("user" | "assistant")
        ) && add_cache_control_to_text_content(message, cache_control.clone())
        {
            break;
        }
    }
    if let Some(tools) = payload.get_mut("tools").and_then(Value::as_array_mut)
        && let Some(last_tool) = tools.last_mut()
    {
        last_tool["cache_control"] = cache_control;
    }
}

fn add_cache_control_to_text_content(message: &mut Value, cache_control: Value) -> bool {
    match message.get_mut("content") {
        Some(Value::String(text)) if !text.is_empty() => {
            let text = std::mem::take(text);
            message["content"] =
                json!([{ "type": "text", "text": text, "cache_control": cache_control }]);
            true
        }
        Some(Value::Array(parts)) => {
            for part in parts.iter_mut().rev() {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    part["cache_control"] = cache_control;
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

fn headers(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    api_key: &str,
    compat: &ResolvedOpenAICompletionsCompat,
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
        && compat.send_session_affinity_headers
        && cache_retention != CacheRetention::None
    {
        headers.insert(
            HeaderName::from_static("session_id"),
            HeaderValue::from_str(session_id)
                .map_err(|e| Error::InvalidHeaderValue("session_id".to_string(), e))?,
        );
        headers.insert(
            HeaderName::from_static("x-client-request-id"),
            HeaderValue::from_str(session_id)
                .map_err(|e| Error::InvalidHeaderValue("x-client-request-id".to_string(), e))?,
        );
        headers.insert(
            HeaderName::from_static("x-session-affinity"),
            HeaderValue::from_str(session_id)
                .map_err(|e| Error::InvalidHeaderValue("x-session-affinity".to_string(), e))?,
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

fn trim_end_slash(url: &str) -> &str {
    url.trim_end_matches('/')
}

fn request_base_url(model: &Model) -> Result<String> {
    Ok(model.base_url.clone())
}

fn immediate_error(model: Model, message: &str) -> crate::AssistantMessageEventStream {
    let (mut sender, stream) = crate::AssistantMessageEventStream::channel();
    let mut output = AssistantMessage::empty_for(&model);
    output.stop_reason = StopReason::Error;
    output.error_message = Some(message.to_string());
    sender.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: output,
    });
    stream
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        Message, ModelCompat, ModelCost, OpenAICompletionsCompat, PayloadHook, ResponseHook,
        ToolResultMessage,
    };
    use futures::StreamExt;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn model() -> Model {
        Model {
            id: "gpt-5.5".to_string(),
            name: "GPT 5.5".to_string(),
            api: "openai-completions".to_string(),
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
    fn should_handle_empty_content_array() {
        let model = model();
        let context = Context {
            messages: vec![Message::User(crate::types::UserMessage {
                content: UserMessageContent::Parts(Vec::new()),
                timestamp: 0,
            })],
            ..Default::default()
        };

        let messages = convert_messages(&model, &context, &get_compat(&model));

        assert!(messages.is_empty());
    }

    #[test]
    fn should_handle_empty_string_content() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("")],
            ..Default::default()
        };

        let messages = convert_messages(&model, &context, &get_compat(&model));

        assert_eq!(messages, vec![json!({ "role": "user", "content": "" })]);
    }

    #[test]
    fn should_handle_whitespace_only_content() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("   \n\t  ")],
            ..Default::default()
        };

        let messages = convert_messages(&model, &context, &get_compat(&model));

        assert_eq!(
            messages,
            vec![json!({ "role": "user", "content": "   \n\t  " })]
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

        let messages = convert_messages(&model, &context, &get_compat(&model));
        let roles = messages
            .iter()
            .filter_map(|message| message.get("role").and_then(Value::as_str))
            .collect::<Vec<_>>();

        assert_eq!(roles, ["user", "user"]);
    }

    #[test]
    fn simple_tool_choice_uses_upstream_camel_case_name() {
        let camel = SimpleStreamOptions {
            stream: StreamOptions {
                provider_options: [("toolChoice".to_string(), json!("required"))]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(simple_tool_choice(&camel), Some(json!("required")));

        let snake = SimpleStreamOptions {
            stream: StreamOptions {
                provider_options: [("tool_choice".to_string(), json!("required"))]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(simple_tool_choice(&snake), None);
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

    #[tokio::test]
    async fn stream_simple_missing_api_key_names_provider() {
        let mut stream = stream_simple_openai_completions(
            model(),
            Context::default(),
            SimpleStreamOptions::default(),
        );
        while stream.next().await.is_some() {}
        let message = stream.result().await.unwrap();

        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            message.error_message.as_deref(),
            Some("No API key for provider: openai")
        );
    }

    #[test]
    fn builds_developer_role_and_reasoning_effort() {
        let model = model();
        let compat = get_compat(&model);
        let context = Context {
            system_prompt: Some("You are terse.".to_string()),
            messages: vec![Message::user_text("hello")],
            tools: Vec::new(),
        };
        let options = OpenAICompletionsOptions {
            reasoning_effort: Some(ModelThinkingLevel::Low),
            ..Default::default()
        };
        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &compat,
            CacheRetention::Short,
        );
        assert_eq!(payload["messages"][0]["role"], "developer");
        assert_eq!(payload["reasoning_effort"], "low");
        assert_eq!(payload["stream"], true);
    }

    #[test]
    fn chat_payload_treats_off_reasoning_effort_as_unspecified() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hello")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            reasoning_effort: Some(ModelThinkingLevel::Off),
            ..Default::default()
        };
        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("reasoning_effort").is_none());
    }

    #[test]
    fn chat_payload_skips_empty_system_prompt() {
        let model = model();
        let context = Context {
            system_prompt: Some(String::new()),
            messages: vec![Message::user_text("hello")],
            tools: Vec::new(),
        };
        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["messages"][0]["role"], "user");
    }

    #[test]
    fn chat_headers_let_explicit_headers_override_session_affinity() {
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
        options.headers.insert(
            "x-session-affinity".to_string(),
            "override-affinity".to_string(),
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
        assert_eq!(
            headers
                .get("x-session-affinity")
                .and_then(|value| value.to_str().ok()),
            Some("override-affinity")
        );
    }

    #[test]
    fn chat_payload_sets_openai_prompt_cache_fields() {
        let mut model = model();
        model.base_url = "https://api.openai.com/v1".to_string();
        let compat = get_compat(&model);
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            base: StreamOptions {
                session_id: Some(format!("{}tail", "x".repeat(64))),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &compat,
            CacheRetention::Long,
        );

        assert_eq!(payload["prompt_cache_key"], json!("x".repeat(64)));
        assert_eq!(payload["prompt_cache_retention"], json!("24h"));
    }

    #[test]
    fn chat_payload_sets_openai_prompt_cache_key_for_short_retention() {
        let mut model = model();
        model.base_url = "https://api.openai.com/v1".to_string();
        let compat = get_compat(&model);
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            base: StreamOptions {
                session_id: Some("session-short".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &compat,
            CacheRetention::Short,
        );

        assert_eq!(payload["prompt_cache_key"], json!("session-short"));
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn chat_payload_uses_pi_cache_retention_for_direct_openai_requests() {
        let _env = crate::test_env::EnvVarGuard::set("PI_CACHE_RETENTION", "long");
        let mut model = model();
        model.base_url = "https://api.openai.com/v1".to_string();
        let compat = get_compat(&model);
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            base: StreamOptions {
                session_id: Some("session-env".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &compat,
            resolve_cache_retention(options.base.cache_retention),
        );

        assert_eq!(payload["prompt_cache_key"], json!("session-env"));
        assert_eq!(payload["prompt_cache_retention"], json!("24h"));
    }

    #[test]
    fn chat_payload_omits_prompt_cache_fields_when_retention_is_none() {
        let mut model = model();
        model.base_url = "https://api.openai.com/v1".to_string();
        let compat = get_compat(&model);
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            base: StreamOptions {
                session_id: Some("session-123".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &compat,
            CacheRetention::None,
        );

        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn chat_payload_omits_proxy_prompt_cache_without_long_retention_support() {
        let mut model = model();
        model.base_url = "https://proxy.example.com/v1".to_string();
        model
            .compat
            .openai_completions
            .supports_long_cache_retention = Some(false);
        let compat = get_compat(&model);
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            base: StreamOptions {
                session_id: Some("session-123".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &compat,
            CacheRetention::Long,
        );

        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn chat_payload_sets_proxy_prompt_cache_when_long_retention_is_supported() {
        let mut model = model();
        model.base_url = "https://proxy.example.com/v1".to_string();
        let compat = get_compat(&model);
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            base: StreamOptions {
                session_id: Some("session-proxy".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &compat,
            CacheRetention::Long,
        );

        assert_eq!(payload["prompt_cache_key"], json!("session-proxy"));
        assert_eq!(payload["prompt_cache_retention"], json!("24h"));
    }

    #[test]
    fn does_not_send_default_max_token_fields() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("max_tokens").is_none());
        assert!(payload.get("max_completion_tokens").is_none());
    }

    #[test]
    fn sends_explicit_max_tokens() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            base: StreamOptions {
                max_tokens: Some(1234),
                ..Default::default()
            },
            ..Default::default()
        };
        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("max_tokens").is_none());
        assert_eq!(payload["max_completion_tokens"], json!(1234));
    }

    #[test]
    fn chat_payload_omits_zero_max_tokens() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            base: StreamOptions {
                max_tokens: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };
        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("max_tokens").is_none());
        assert!(payload.get("max_completion_tokens").is_none());
    }

    #[test]
    fn chat_payload_uses_explicit_nested_reasoning_compat() {
        let mut model = model();
        model.provider = "custom-openai-compatible".to_string();
        model.id = "custom-reasoning-model".to_string();
        model.base_url = "https://example.compat/v1".to_string();
        model.reasoning = true;
        model.compat.openai_completions.thinking_format = Some(OpenAIThinkingFormat::Openrouter);
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            reasoning_effort: Some(ModelThinkingLevel::High),
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::None,
        );

        assert_eq!(payload["reasoning"], json!({ "effort": "high" }));
        assert!(payload.get("reasoning_effort").is_none());
    }

    #[test]
    fn provider_name_does_not_enable_out_of_scope_openai_compat() {
        let mut model = model();
        model.provider = "openrouter".to_string();
        model.id = "deepseek/deepseek-r1".to_string();
        model.base_url = "https://openrouter.ai/api/v1".to_string();
        model.reasoning = true;
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };
        let options = OpenAICompletionsOptions {
            reasoning_effort: Some(ModelThinkingLevel::High),
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["reasoning_effort"], json!("high"));
        assert!(payload.get("reasoning").is_none());
        assert!(
            payload["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[test]
    fn chat_payload_omits_empty_gateway_routing() {
        let mut model = model();
        model.base_url = "https://ai-gateway.vercel.sh/v1".to_string();
        model.compat = ModelCompat {
            openai_completions: OpenAICompletionsCompat {
                vercel_gateway_routing: Some(json!({})),
                ..Default::default()
            },
            ..Default::default()
        };
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("providerOptions").is_none());
    }

    #[test]
    fn chat_payload_maps_gateway_routing_to_provider_options() {
        let mut model = model();
        model.base_url = "https://ai-gateway.vercel.sh/v1".to_string();
        model.compat = ModelCompat {
            openai_completions: OpenAICompletionsCompat {
                vercel_gateway_routing: Some(json!({
                    "only": ["openai"],
                    "order": ["openai", "anthropic"],
                    "ignored": ["not-forwarded"]
                })),
                ..Default::default()
            },
            ..Default::default()
        };
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(
            payload["providerOptions"],
            json!({ "gateway": { "only": ["openai"], "order": ["openai", "anthropic"] } })
        );
    }

    #[test]
    fn omits_tools_field_when_context_tools_is_an_empty_array() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("tools").is_none());
    }

    #[test]
    fn omits_tools_field_when_context_tools_is_undefined() {
        let model = model();
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert!(payload.get("tools").is_none());
    }

    #[test]
    fn still_emits_tools_for_anthropic_litellm_proxy_when_conversation_has_tool_history() {
        let model = model();
        let mut assistant = AssistantMessage::empty_for(&model);
        assistant.content.push(AssistantContent::ToolCall(ToolCall {
            id: "tool-1".to_string(),
            name: "read".to_string(),
            arguments: json!({ "path": "README.md" }),
            thought_signature: None,
        }));
        assistant.stop_reason = StopReason::ToolUse;

        let context = Context {
            messages: vec![
                Message::user_text("read"),
                Message::Assistant(assistant),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "tool-1".to_string(),
                    tool_name: "read".to_string(),
                    content: vec![ToolResultContent::text("done")],
                    details: None,
                    is_error: false,
                    timestamp: 0,
                }),
            ],
            tools: Vec::new(),
            ..Default::default()
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["tools"], json!([]));
    }

    #[test]
    fn normalizes_prefilled_context_with_long_pipe_separated_ids() {
        let mut source_model = model();
        source_model.provider = "github-copilot".to_string();
        source_model.api = "openai-responses".to_string();
        source_model.id = "gpt-5.2-codex".to_string();

        let mut target_model = model();
        target_model.provider = "custom-openai-compatible".to_string();
        target_model.id = "gpt-5.2-codex-compatible".to_string();

        let raw_tool_call_id = concat!(
            "call_pAYbIr76hXIjncD9UE4eGfnS|",
            "t5nnb2qYMFWGSsr13fhCd1CaCu3t3qONEPuOudu4HSVEtA8YJSL6FAZUxvoOoD792VIJWl91g87EdqsCWp9krVsd",
            "BysQoDaf9lMCLb8BS4EYi4gQd5kBQBYLlgD71PYwvf+TbMD9J9/5OMD42oxSRj8H+vRf78/l2Xla33LWz4nOgsd",
            "dBlbvabICRs8GHt5C9PK5keFtzyi3lsyVKNlfduK3iphsZqs4MLv4zyGJnvZo/+QzShyk5xnMSQX/f98+aEoNfl",
            "EApCdEOXipipgeiNWnpFSHbcwmMkZoJhURNu+JEz3xCh1mrXeYoN5o+trLL3IXJacSsLYXDrYTipZZbJFRPAucg",
            "bnjYBC+/ZzJOfkwCs+Gkw7EoZR7ZQgJ8ma+9586n4tT4cI8DEhBSZsWMjrCt8dxKg=="
        );
        let assistant = AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: raw_tool_call_id.to_string(),
                name: "echo".to_string(),
                arguments: json!({ "message": "hello" }),
                thought_signature: Some(json!({ "provider": "copilot" }).to_string()),
            })],
            api: source_model.api,
            provider: source_model.provider,
            model: source_model.id,
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 2,
        };
        let context = Context {
            messages: vec![
                Message::user_text("Use echo."),
                Message::Assistant(assistant),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: raw_tool_call_id.to_string(),
                    tool_name: "echo".to_string(),
                    content: vec![ToolResultContent::text("hello")],
                    details: None,
                    is_error: false,
                    timestamp: 3,
                }),
                Message::user_text("Say hi."),
            ],
            tools: vec![Tool {
                name: "echo".to_string(),
                description: "Echo a message".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                }),
            }],
            ..Default::default()
        };

        let messages = convert_messages(&target_model, &context, &get_compat(&target_model));
        let assistant_message = messages
            .iter()
            .find(|message| message["role"] == "assistant")
            .expect("assistant message");
        let tool_call_id = assistant_message["tool_calls"][0]["id"]
            .as_str()
            .expect("tool call id");
        let tool_result_message = messages
            .iter()
            .find(|message| message["role"] == "tool")
            .expect("tool result message");

        assert_eq!(tool_call_id, "call_pAYbIr76hXIjncD9UE4eGfnS");
        assert_eq!(tool_result_message["tool_call_id"], json!(tool_call_id));
        assert!(tool_call_id.len() <= 40);
        assert!(!tool_call_id.contains('|'));
        assert!(assistant_message.get("reasoning_details").is_none());
    }

    #[test]
    fn chat_headers_set_and_omit_session_affinity_by_cache_retention() {
        let mut model = model();
        model.base_url = "https://proxy.example.com/v1".to_string();
        model
            .compat
            .openai_completions
            .send_session_affinity_headers = Some(true);
        let context = Context {
            system_prompt: None,
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
        };
        let options = StreamOptions {
            session_id: Some("session-affinity".to_string()),
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
            Some("session-affinity")
        );
        assert_eq!(
            request_headers
                .get("x-client-request-id")
                .and_then(|value| value.to_str().ok()),
            Some("session-affinity")
        );
        assert_eq!(
            request_headers
                .get("x-session-affinity")
                .and_then(|value| value.to_str().ok()),
            Some("session-affinity")
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
        assert!(request_headers.get("x-session-affinity").is_none());
    }

    #[test]
    fn chat_headers_add_copilot_dynamic_headers_and_allow_overrides() {
        let mut model = model();
        model.provider = "github-copilot".to_string();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::ToolResult(ToolResultMessage {
                tool_call_id: "call-1".to_string(),
                tool_name: "screenshot".to_string(),
                content: vec![ToolResultContent::Image(ImageContent {
                    data: "abc".to_string(),
                    mime_type: "image/png".to_string(),
                })],
                details: None,
                is_error: false,
                timestamp: 1,
            })],
            tools: Vec::new(),
        };
        let mut options = StreamOptions::default();
        options
            .headers
            .insert("Openai-Intent".to_string(), "override-intent".to_string());

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
                .get("x-initiator")
                .and_then(|value| value.to_str().ok()),
            Some("agent")
        );
        assert_eq!(
            headers
                .get("openai-intent")
                .and_then(|value| value.to_str().ok()),
            Some("override-intent")
        );
        assert_eq!(
            headers
                .get("copilot-vision-request")
                .and_then(|value| value.to_str().ok()),
            Some("true")
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

    fn chat_sse_body(chunks: &[Value]) -> String {
        let mut body = chunks
            .iter()
            .map(|chunk| format!("data: {chunk}\n\n"))
            .collect::<String>();
        body.push_str("data: [DONE]\n\n");
        body
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

    async fn spawn_capturing_sse_server(
        body: String,
        captured_payload: Arc<Mutex<Option<Value>>>,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut socket).await;
            *captured_payload.lock().unwrap() = Some(request_body_json(&request));
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = vec![0u8; 4096];
        let mut expected_len = None;
        loop {
            let read = socket.read(&mut buffer).await.unwrap();
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            if expected_len.is_none() {
                expected_len = expected_request_len(&request);
            }
            if expected_len.is_some_and(|len| request.len() >= len) {
                break;
            }
        }
        request
    }

    fn expected_request_len(request: &[u8]) -> Option<usize> {
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")?;
        let headers = std::str::from_utf8(&request[..header_end]).ok()?;
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        Some(header_end + 4 + content_length)
    }

    fn request_body_json(request: &[u8]) -> Value {
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("request headers");
        let expected_len = expected_request_len(request).expect("request length");
        serde_json::from_slice(&request[header_end + 4..expected_len]).expect("request JSON")
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
                let _ = socket.read(&mut buffer).await;
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

    #[tokio::test]
    async fn surfaces_routed_chunk_model_as_response_model() {
        let mut routed_model = model();
        routed_model.id = "openrouter/auto".to_string();
        routed_model.provider = "openrouter".to_string();
        routed_model.reasoning = false;
        routed_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-1",
                "model": "anthropic/claude-opus-4.8",
                "choices": [{ "index": 0, "delta": { "content": "hi" } }]
            }),
            json!({
                "id": "chatcmpl-1",
                "model": "anthropic/claude-opus-4.8",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            routed_model,
            Context {
                messages: vec![Message::user_text("hi")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.model, "openrouter/auto");
        assert_eq!(
            message.response_model.as_deref(),
            Some("anthropic/claude-opus-4.8")
        );
        assert_eq!(message.provider, "openrouter");
        assert_eq!(message.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn leaves_response_model_empty_when_chunks_echo_requested_model() {
        let mut routed_model = model();
        routed_model.id = "openrouter/auto".to_string();
        routed_model.provider = "openrouter".to_string();
        routed_model.reasoning = false;
        routed_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-echo",
                "model": "openrouter/auto",
                "choices": [{ "index": 0, "delta": { "content": "hi" } }]
            }),
            json!({
                "id": "chatcmpl-echo",
                "model": "openrouter/auto",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            routed_model,
            Context {
                messages: vec![Message::user_text("hi")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.model, "openrouter/auto");
        assert_eq!(message.response_model, None);
        assert_eq!(message.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn ignores_empty_or_missing_chunk_model_for_response_model() {
        let mut routed_model = model();
        routed_model.id = "openrouter/auto".to_string();
        routed_model.provider = "openrouter".to_string();
        routed_model.reasoning = false;
        routed_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-2",
                "choices": [{ "index": 0, "delta": { "content": "hi" } }]
            }),
            json!({
                "id": "chatcmpl-2",
                "model": "",
                "choices": [{ "index": 0, "delta": { "content": "!" } }]
            }),
            json!({
                "id": "chatcmpl-2",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            routed_model,
            Context {
                messages: vec![Message::user_text("hi")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.model, "openrouter/auto");
        assert_eq!(message.response_model, None);
        assert_eq!(message.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn ignores_empty_chunk_id_for_response_id() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "",
                "choices": [{ "index": 0, "delta": { "content": "hi" } }]
            }),
            json!({
                "id": "chatcmpl-final",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("hi")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.response_id.as_deref(), Some("chatcmpl-final"));
        assert_eq!(message.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn chat_provider_does_not_retry_by_default() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_retrying_sse_server(
            chat_sse_body(&[json!({
                "id": "chatcmpl-retry",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            })]),
            Arc::clone(&attempts),
        )
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        while stream.next().await.is_some() {}
        let message = stream.result().await.unwrap();

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(message.stop_reason, StopReason::Error);
        assert!(
            message
                .error_message
                .as_deref()
                .is_some_and(|message| message.contains("500"))
        );
    }

    #[tokio::test]
    async fn chat_provider_skips_on_response_for_api_errors() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let response_calls = Arc::new(AtomicUsize::new(0));
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url =
            spawn_retrying_sse_server(chat_sse_body(&[]), Arc::clone(&attempts)).await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    on_response: Some(counting_on_response(Arc::clone(&response_calls))),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        while stream.next().await.is_some() {}
        let message = stream.result().await.unwrap();

        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(response_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn chat_provider_honors_explicit_retry_settings() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_retrying_sse_server(
            chat_sse_body(&[
                json!({
                    "id": "chatcmpl-retry",
                    "choices": [{ "index": 0, "delta": { "content": "ok" }, "finish_reason": null }]
                }),
                json!({
                    "id": "chatcmpl-retry",
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "prompt_tokens_details": { "cached_tokens": 0 },
                        "completion_tokens_details": { "reasoning_tokens": 0 }
                    }
                }),
            ]),
            Arc::clone(&attempts),
        )
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
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
        while stream.next().await.is_some() {}
        let message = stream.result().await.unwrap();

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(message.stop_reason, StopReason::Stop);
        assert_eq!(
            message.content,
            vec![AssistantContent::Text(TextContent {
                text: "ok".to_string(),
                text_signature: None,
            })]
        );
    }

    #[tokio::test]
    async fn should_handle_immediate_abort() {
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        cancellation_token.cancel();
        let mut stream = stream_openai_completions(
            model(),
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
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
    async fn should_abort_mid_stream() {
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        let (base_url, release_server) = spawn_hanging_sse_server(chat_sse_body(&[json!({
            "id": "chatcmpl-abort",
            "choices": [{ "index": 0, "delta": { "content": "partial" }, "finish_reason": null }]
        })]))
        .await;
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = base_url;
        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    cancellation_token: Some(cancellation_token.clone()),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        while let Some(event) = stream.next().await {
            if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
                cancellation_token.cancel();
            }
        }
        let result = stream.result().await.unwrap();
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
    async fn choice_usage_fallback_updates_from_later_chunks() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-choice-usage",
                "choices": [{
                    "index": 0,
                    "delta": { "content": "OK" },
                    "finish_reason": null,
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "prompt_tokens_details": { "cached_tokens": 0 },
                        "completion_tokens_details": { "reasoning_tokens": 0 }
                    }
                }]
            }),
            json!({
                "id": "chatcmpl-choice-usage",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "prompt_tokens_details": { "cached_tokens": 0 },
                        "completion_tokens_details": { "reasoning_tokens": 0 }
                    }
                }]
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("hi")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.usage.input, 10);
        assert_eq!(message.usage.output, 5);
        assert_eq!(message.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn chat_usage_does_not_double_count_reasoning_tokens() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[json!({
            "id": "chatcmpl-reasoning-usage",
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 33,
                "prompt_tokens_details": { "cached_tokens": 0 },
                "completion_tokens_details": { "reasoning_tokens": 21 }
            }
        })]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Use reasoning.")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.usage.input, 10);
        assert_eq!(message.usage.output, 33);
        assert_eq!(message.usage.total_tokens, 43);
    }

    #[tokio::test]
    async fn chat_usage_preserves_cache_read_and_write_from_chunk_usage() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-cache-write",
                "choices": [{ "index": 0, "delta": { "content": "OK" }, "finish_reason": null }]
            }),
            json!({
                "id": "chatcmpl-cache-write",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": { "cached_tokens": 50, "cache_write_tokens": 30 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Reply with exactly OK")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.usage.input, 20);
        assert_eq!(message.usage.cache_read, 50);
        assert_eq!(message.usage.cache_write, 30);
        assert_eq!(message.usage.output, 5);
        assert_eq!(message.usage.total_tokens, 105);
    }

    #[tokio::test]
    async fn chat_usage_preserves_cache_read_and_write_from_choice_usage() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-cache-write-choice",
                "choices": [{ "index": 0, "delta": { "content": "OK" }, "finish_reason": null }]
            }),
            json!({
                "id": "chatcmpl-cache-write-choice",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 5,
                        "prompt_tokens_details": { "cached_tokens": 50, "cache_write_tokens": 30 },
                        "completion_tokens_details": { "reasoning_tokens": 0 }
                    }
                }]
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Reply with exactly OK")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.usage.input, 20);
        assert_eq!(message.usage.cache_read, 50);
        assert_eq!(message.usage.cache_write, 30);
        assert_eq!(message.usage.output, 5);
        assert_eq!(message.usage.total_tokens, 105);
    }

    #[tokio::test]
    async fn forwards_tool_choice_from_simple_options_to_payload() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[json!({
            "id": "chatcmpl-3",
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "prompt_tokens_details": { "cached_tokens": 0 },
                "completion_tokens_details": { "reasoning_tokens": 0 }
            }
        })]))
        .await;

        let captured_payload = Arc::new(Mutex::new(None));
        let hook_capture = Arc::clone(&captured_payload);
        let on_payload: PayloadHook = Arc::new(move |payload, _model| {
            let hook_capture = Arc::clone(&hook_capture);
            Box::pin(async move {
                *hook_capture.lock().unwrap() = Some(payload.clone());
                Ok(Some(payload))
            })
        });

        let mut stream = stream_simple_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Call lookup")],
                tools: vec![lookup_tool()],
                ..Default::default()
            },
            SimpleStreamOptions {
                stream: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    on_payload: Some(on_payload),
                    provider_options: [("toolChoice".to_string(), json!("required"))]
                        .into_iter()
                        .collect(),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.stop_reason, StopReason::Stop);
        let payload = captured_payload.lock().unwrap().take().expect("payload");
        assert_eq!(payload["tool_choice"], json!("required"));
        assert!(
            payload["tools"]
                .as_array()
                .is_some_and(|tools| !tools.is_empty())
        );
    }

    #[tokio::test]
    async fn ignores_null_stream_chunks_from_compatible_providers() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            Value::Null,
            json!({
                "id": "chatcmpl-null",
                "choices": [{ "index": 0, "delta": { "content": "OK" }, "finish_reason": null }]
            }),
            json!({
                "id": "chatcmpl-null",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Reply OK")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.stop_reason, StopReason::Stop);
        assert_eq!(message.error_message, None);
        assert_eq!(message.response_id.as_deref(), Some("chatcmpl-null"));
        assert_eq!(message.usage.total_tokens, 4);
        assert_eq!(
            message.content,
            vec![AssistantContent::Text(TextContent {
                text: "OK".to_string(),
                text_signature: None,
            })]
        );
    }

    #[tokio::test]
    async fn errors_when_stream_ends_without_finish_reason() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-truncated",
                "choices": [{ "index": 0, "delta": { "content": "partial" }, "finish_reason": null }]
            }),
            json!({
                "id": "chatcmpl-truncated",
                "choices": [{ "index": 0, "delta": { "content": " answer" }, "finish_reason": null }]
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Reply longer")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            message.error_message.as_deref(),
            Some("Stream ended without finish_reason")
        );
    }

    #[tokio::test]
    async fn maps_provider_finish_reason_errors() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-network-error",
                "choices": [{ "index": 0, "delta": { "content": "partial" }, "finish_reason": null }]
            }),
            json!({
                "id": "chatcmpl-network-error",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "network_error" }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Hi")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let message = stream.result().await.unwrap();
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            message.error_message.as_deref(),
            Some("Provider finish_reason: network_error")
        );
    }

    #[tokio::test]
    async fn coalesces_tool_call_deltas_by_index_when_provider_mutates_ids() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-mutating-tools",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "functions.read:0",
                            "type": "function",
                            "function": { "name": "read", "arguments": "" }
                        }]
                    },
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-mutating-tools",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "chatcmpl-tool-a",
                            "type": "function",
                            "function": { "name": null, "arguments": "{\"path\":\"README" }
                        }]
                    },
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-mutating-tools",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "chatcmpl-tool-b",
                            "type": "function",
                            "function": { "name": null, "arguments": ".md\"}" }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Read README.md")],
                tools: vec![lookup_tool()],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let mut tool_call_indexes = Vec::new();
        while let Some(event) = stream.next().await {
            match event {
                AssistantMessageEvent::ToolCallStart { content_index, .. }
                | AssistantMessageEvent::ToolCallDelta { content_index, .. }
                | AssistantMessageEvent::ToolCallEnd { content_index, .. } => {
                    tool_call_indexes.push(content_index);
                }
                _ => {}
            }
        }
        let message = stream.result().await.unwrap();

        assert_eq!(message.stop_reason, StopReason::ToolUse);
        assert_eq!(tool_call_indexes, vec![0, 0, 0, 0, 0]);
        assert_eq!(
            message.content,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "functions.read:0".to_string(),
                name: "read".to_string(),
                arguments: json!({ "path": "README.md" }),
                thought_signature: None,
            })]
        );
    }

    #[tokio::test]
    async fn binds_late_tool_call_index_to_existing_id() {
        let mut chat_model = model();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-late-index",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "id": "tc_late",
                            "type": "function",
                            "function": { "name": "read", "arguments": "{\"path\"" }
                        }]
                    },
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-late-index",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "tc_late",
                            "type": "function",
                            "function": { "arguments": ":\"README" }
                        }]
                    },
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-late-index",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "type": "function",
                            "function": { "arguments": ".md\"}" }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Read README.md")],
                tools: vec![lookup_tool()],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let mut tool_call_indexes = Vec::new();
        while let Some(event) = stream.next().await {
            match event {
                AssistantMessageEvent::ToolCallStart { content_index, .. }
                | AssistantMessageEvent::ToolCallDelta { content_index, .. }
                | AssistantMessageEvent::ToolCallEnd { content_index, .. } => {
                    tool_call_indexes.push(content_index);
                }
                _ => {}
            }
        }
        let message = stream.result().await.unwrap();

        assert_eq!(message.stop_reason, StopReason::ToolUse);
        assert_eq!(tool_call_indexes, vec![0, 0, 0, 0, 0]);
        assert_eq!(
            message.content,
            vec![AssistantContent::ToolCall(ToolCall {
                id: "tc_late".to_string(),
                name: "read".to_string(),
                arguments: json!({ "path": "README.md" }),
                thought_signature: None,
            })]
        );
    }

    #[tokio::test]
    async fn accumulates_mixed_text_reasoning_and_parallel_tool_deltas_independently() {
        let mut chat_model = model();
        chat_model.id = "gpt-4o-mini".to_string();
        chat_model.reasoning = false;
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[
            json!({
                "id": "chatcmpl-mixed-deltas",
                "choices": [{
                    "delta": {
                        "content": "answer 1",
                        "reasoning_content": "think 1",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc_read_initial",
                                "type": "function",
                                "function": { "name": "read", "arguments": "{\"path\":\"README" }
                            },
                            {
                                "index": 1,
                                "id": "tc_grep_initial",
                                "type": "function",
                                "function": { "name": "grep", "arguments": "{\"pattern\":\"TODO" }
                            },
                            {
                                "id": "tc_list_no_index",
                                "type": "function",
                                "function": { "name": "list", "arguments": "{\"path\":\"packages" }
                            },
                            {
                                "id": "tc_write_no_index",
                                "type": "function",
                                "function": { "name": "write", "arguments": "{\"path\":\"out" }
                            }
                        ]
                    },
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-mixed-deltas",
                "choices": [{
                    "delta": {
                        "content": " answer 2",
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "tc_grep_changed",
                                "type": "function",
                                "function": { "arguments": "\",\"path\":\"src" }
                            },
                            {
                                "id": "tc_write_no_index",
                                "type": "function",
                                "function": { "arguments": ".txt\",\"content\":\"ok\"}" }
                            },
                            {
                                "id": "tc_list_no_index",
                                "type": "function",
                                "function": { "arguments": "/ai\"}" }
                            }
                        ]
                    },
                    "finish_reason": null
                }]
            }),
            json!({
                "id": "chatcmpl-mixed-deltas",
                "choices": [{
                    "delta": {
                        "content": "\n",
                        "reasoning_content": " think 2",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc_read_changed",
                                "type": "function",
                                "function": { "arguments": ".md\"}" }
                            },
                            {
                                "index": 1,
                                "type": "function",
                                "function": { "arguments": "\"}" }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 2 }
                }
            }),
        ]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Think, answer, and use tools.")],
                tools: vec![
                    Tool {
                        name: "read".to_string(),
                        description: "Read a file".to_string(),
                        parameters: json!({
                            "type": "object",
                            "properties": { "path": { "type": "string" } },
                            "required": ["path"]
                        }),
                    },
                    Tool {
                        name: "grep".to_string(),
                        description: "Search a file".to_string(),
                        parameters: json!({
                            "type": "object",
                            "properties": {
                                "pattern": { "type": "string" },
                                "path": { "type": "string" }
                            },
                            "required": ["pattern", "path"]
                        }),
                    },
                    Tool {
                        name: "list".to_string(),
                        description: "List a directory".to_string(),
                        parameters: json!({
                            "type": "object",
                            "properties": { "path": { "type": "string" } },
                            "required": ["path"]
                        }),
                    },
                    Tool {
                        name: "write".to_string(),
                        description: "Write a file".to_string(),
                        parameters: json!({
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" },
                                "content": { "type": "string" }
                            },
                            "required": ["path", "content"]
                        }),
                    },
                ],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let mut event_names = Vec::new();
        let mut tool_events_by_content_index: HashMap<usize, Vec<&'static str>> = HashMap::new();
        while let Some(event) = stream.next().await {
            let name = match &event {
                AssistantMessageEvent::TextStart { .. } => "text_start",
                AssistantMessageEvent::TextDelta { .. } => "text_delta",
                AssistantMessageEvent::TextEnd { .. } => "text_end",
                AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
                AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
                AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
                AssistantMessageEvent::ToolCallStart { content_index, .. } => {
                    tool_events_by_content_index
                        .entry(*content_index)
                        .or_default()
                        .push("toolcall_start");
                    "toolcall_start"
                }
                AssistantMessageEvent::ToolCallDelta { content_index, .. } => {
                    tool_events_by_content_index
                        .entry(*content_index)
                        .or_default()
                        .push("toolcall_delta");
                    "toolcall_delta"
                }
                AssistantMessageEvent::ToolCallEnd { content_index, .. } => {
                    tool_events_by_content_index
                        .entry(*content_index)
                        .or_default()
                        .push("toolcall_end");
                    "toolcall_end"
                }
                _ => "other",
            };
            event_names.push(name);
        }
        let message = stream.result().await.unwrap();

        assert_eq!(message.stop_reason, StopReason::ToolUse);
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "text_start")
                .count(),
            1
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "text_delta")
                .count(),
            3
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "text_end")
                .count(),
            1
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "thinking_start")
                .count(),
            1
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "thinking_delta")
                .count(),
            2
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "thinking_end")
                .count(),
            1
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "toolcall_start")
                .count(),
            4
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "toolcall_delta")
                .count(),
            9
        );
        assert_eq!(
            event_names
                .iter()
                .filter(|name| **name == "toolcall_end")
                .count(),
            4
        );
        assert_eq!(
            tool_events_by_content_index.get(&2).map(Vec::as_slice),
            Some(
                &[
                    "toolcall_start",
                    "toolcall_delta",
                    "toolcall_delta",
                    "toolcall_end",
                ][..]
            )
        );
        assert_eq!(
            tool_events_by_content_index.get(&3).map(Vec::as_slice),
            Some(
                &[
                    "toolcall_start",
                    "toolcall_delta",
                    "toolcall_delta",
                    "toolcall_delta",
                    "toolcall_end",
                ][..]
            )
        );
        assert_eq!(
            tool_events_by_content_index.get(&4).map(Vec::as_slice),
            Some(
                &[
                    "toolcall_start",
                    "toolcall_delta",
                    "toolcall_delta",
                    "toolcall_end",
                ][..]
            )
        );
        assert_eq!(
            tool_events_by_content_index.get(&5).map(Vec::as_slice),
            Some(
                &[
                    "toolcall_start",
                    "toolcall_delta",
                    "toolcall_delta",
                    "toolcall_end",
                ][..]
            )
        );

        assert_eq!(message.content.len(), 6);
        assert_eq!(
            message.content[0],
            AssistantContent::Text(TextContent {
                text: "answer 1 answer 2\n".to_string(),
                text_signature: None,
            })
        );
        assert_eq!(
            message.content[1],
            AssistantContent::Thinking(ThinkingContent {
                thinking: "think 1 think 2".to_string(),
                thinking_signature: Some("reasoning_content".to_string()),
                redacted: None,
            })
        );
        assert_eq!(
            message.content[2],
            AssistantContent::ToolCall(ToolCall {
                id: "tc_read_initial".to_string(),
                name: "read".to_string(),
                arguments: json!({ "path": "README.md" }),
                thought_signature: None,
            })
        );
        assert_eq!(
            message.content[3],
            AssistantContent::ToolCall(ToolCall {
                id: "tc_grep_initial".to_string(),
                name: "grep".to_string(),
                arguments: json!({ "pattern": "TODO", "path": "src" }),
                thought_signature: None,
            })
        );
        assert_eq!(
            message.content[4],
            AssistantContent::ToolCall(ToolCall {
                id: "tc_list_no_index".to_string(),
                name: "list".to_string(),
                arguments: json!({ "path": "packages/ai" }),
                thought_signature: None,
            })
        );
        assert_eq!(
            message.content[5],
            AssistantContent::ToolCall(ToolCall {
                id: "tc_write_no_index".to_string(),
                name: "write".to_string(),
                arguments: json!({ "path": "out.txt", "content": "ok" }),
                thought_signature: None,
            })
        );
    }

    #[tokio::test]
    async fn reasoning_deltas_keep_source_signature() {
        let mut chat_model = model();
        chat_model.id = "gpt-4o-mini".to_string();
        chat_model.base_url = spawn_sse_server(chat_sse_body(&[json!({
            "id": "chatcmpl-reasoning",
            "choices": [{
                "delta": { "reasoning": "think" },
                "finish_reason": "stop"
            }]
        })]))
        .await;

        let mut stream = stream_openai_completions(
            chat_model,
            Context {
                messages: vec![Message::user_text("Use reasoning.")],
                ..Default::default()
            },
            OpenAICompletionsOptions {
                base: StreamOptions {
                    api_key: Some("test-key".to_string()),
                    cache_retention: Some(CacheRetention::None),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        while stream.next().await.is_some() {}
        let message = stream.result().await.unwrap();

        assert_eq!(
            message.content,
            vec![AssistantContent::Thinking(ThinkingContent {
                thinking: "think".to_string(),
                thinking_signature: Some("reasoning".to_string()),
                redacted: None,
            })]
        );
    }

    #[test]
    fn omits_strict_when_compat_disables_strict_mode() {
        let mut model = model();
        model.compat.openai_completions.supports_strict_mode = Some(false);
        let compat = get_compat(&model);
        let payload = build_chat_completions_payload(
            &model,
            &Context {
                messages: vec![Message::user_text("Use the tool")],
                tools: vec![lookup_tool()],
                ..Default::default()
            },
            &OpenAICompletionsOptions::default(),
            &compat,
            CacheRetention::Short,
        );

        let function = &payload["tools"][0]["function"];
        assert_eq!(function["name"], json!("lookup"));
        assert!(function.get("strict").is_none());
    }

    #[test]
    fn enables_explicit_tool_stream_for_supported_models_with_tools() {
        let mut model = model();
        model.provider = "custom-openai-compatible".to_string();
        model.base_url = "https://example.compat/v1".to_string();
        model.compat.openai_completions.zai_tool_stream = Some(true);
        let compat = get_compat(&model);
        let payload = build_chat_completions_payload(
            &model,
            &Context {
                messages: vec![Message::user_text("Use the tool")],
                tools: vec![lookup_tool()],
                ..Default::default()
            },
            &OpenAICompletionsOptions::default(),
            &compat,
            CacheRetention::Short,
        );

        assert_eq!(payload["tool_stream"], json!(true));
    }

    #[test]
    fn omits_explicit_tool_stream_without_tools() {
        let mut model = model();
        model.provider = "custom-openai-compatible".to_string();
        model.base_url = "https://example.compat/v1".to_string();
        model.compat.openai_completions.zai_tool_stream = Some(true);
        let compat = get_compat(&model);
        let payload = build_chat_completions_payload(
            &model,
            &Context {
                messages: vec![Message::user_text("No tools")],
                tools: Vec::new(),
                ..Default::default()
            },
            &OpenAICompletionsOptions::default(),
            &compat,
            CacheRetention::Short,
        );

        assert!(payload.get("tool_stream").is_none());
    }

    #[test]
    fn thinking_level_map_can_remap_reasoning_effort() {
        let mut model = model();
        model.id = "custom-reasoning-model".to_string();
        model.provider = "custom-openai-compatible".to_string();
        model
            .thinking_level_map
            .insert("high".to_string(), Some("default".to_string()));
        let payload = build_chat_completions_payload(
            &model,
            &Context {
                messages: vec![Message::user_text("Think carefully.")],
                ..Default::default()
            },
            &OpenAICompletionsOptions {
                reasoning_effort: Some(ModelThinkingLevel::High),
                ..Default::default()
            },
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["reasoning_effort"], json!("default"));
    }

    #[test]
    fn reasoning_levels_without_mapping_keep_reasoning_effort() {
        let mut model = model();
        model.id = "custom-reasoning-model".to_string();
        model.provider = "custom-openai-compatible".to_string();
        let payload = build_chat_completions_payload(
            &model,
            &Context {
                messages: vec![Message::user_text("Think carefully.")],
                ..Default::default()
            },
            &OpenAICompletionsOptions {
                reasoning_effort: Some(ModelThinkingLevel::Medium),
                ..Default::default()
            },
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["reasoning_effort"], json!("medium"));
    }

    #[test]
    fn custom_thinking_payload_respects_reasoning_effort_compat() {
        let mut unsupported_effort = model();
        unsupported_effort.id = "custom-thinking-model".to_string();
        unsupported_effort.provider = "custom-openai-compatible".to_string();
        unsupported_effort.compat.openai_completions.thinking_format =
            Some(OpenAIThinkingFormat::Deepseek);
        unsupported_effort
            .compat
            .openai_completions
            .supports_reasoning_effort = Some(false);
        let unsupported_compat = get_compat(&unsupported_effort);
        let unsupported_disabled = build_chat_completions_payload(
            &unsupported_effort,
            &Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            &OpenAICompletionsOptions::default(),
            &unsupported_compat,
            CacheRetention::Short,
        );
        let unsupported_enabled = build_chat_completions_payload(
            &unsupported_effort,
            &Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            &OpenAICompletionsOptions {
                reasoning_effort: Some(ModelThinkingLevel::High),
                ..Default::default()
            },
            &unsupported_compat,
            CacheRetention::Short,
        );

        assert_eq!(
            unsupported_disabled["thinking"],
            json!({ "type": "disabled" })
        );
        assert!(unsupported_disabled.get("reasoning_effort").is_none());
        assert_eq!(
            unsupported_enabled["thinking"],
            json!({ "type": "enabled" })
        );
        assert!(unsupported_enabled.get("reasoning_effort").is_none());

        let mut supported_effort = model();
        supported_effort.id = "custom-thinking-model".to_string();
        supported_effort.provider = "custom-openai-compatible".to_string();
        supported_effort.compat.openai_completions.thinking_format =
            Some(OpenAIThinkingFormat::Deepseek);
        supported_effort
            .compat
            .openai_completions
            .supports_reasoning_effort = Some(true);
        let supported_compat = get_compat(&supported_effort);
        let supported_payload = build_chat_completions_payload(
            &supported_effort,
            &Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            &OpenAICompletionsOptions {
                reasoning_effort: Some(ModelThinkingLevel::High),
                ..Default::default()
            },
            &supported_compat,
            CacheRetention::Short,
        );

        assert_eq!(supported_payload["thinking"], json!({ "type": "enabled" }));
        assert_eq!(supported_payload["reasoning_effort"], json!("high"));
    }

    #[test]
    fn openai_thinking_payload_respects_reasoning_effort_compat() {
        let mut model = model();
        model.compat.openai_completions.supports_reasoning_effort = Some(false);
        let compat = get_compat(&model);
        let payload = build_chat_completions_payload(
            &model,
            &Context {
                messages: vec![Message::user_text("hello")],
                ..Default::default()
            },
            &OpenAICompletionsOptions {
                reasoning_effort: Some(ModelThinkingLevel::High),
                ..Default::default()
            },
            &compat,
            CacheRetention::Short,
        );

        assert!(payload.get("reasoning_effort").is_none());
    }

    fn assistant_message(content: Vec<AssistantContent>, model: &Model) -> AssistantMessage {
        AssistantMessage {
            content,
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 2,
        }
    }

    #[test]
    fn explicit_metadata_requires_reasoning_content_replay() {
        let mut model = model();
        model.id = "custom-thinking-model".to_string();
        model.provider = "custom-openai-compatible".to_string();
        model
            .compat
            .openai_completions
            .requires_reasoning_content_on_assistant_messages = Some(true);
        model.compat.openai_completions.thinking_format = Some(OpenAIThinkingFormat::Deepseek);
        let compat = get_compat(&model);

        assert!(compat.requires_reasoning_content_on_assistant_messages);
        assert_eq!(compat.thinking_format, OpenAIThinkingFormat::Deepseek);
        assert!(model.compat.openai_completions.max_tokens_field.is_none());
        assert!(
            model
                .compat
                .openai_completions
                .supports_developer_role
                .is_none()
        );
    }

    #[test]
    fn explicit_tool_call_replay_includes_empty_reasoning_content() {
        let mut model = model();
        model.id = "custom-thinking-model".to_string();
        model.provider = "custom-openai-compatible".to_string();
        model
            .compat
            .openai_completions
            .requires_reasoning_content_on_assistant_messages = Some(true);
        let compat = get_compat(&model);
        let assistant = assistant_message(
            vec![AssistantContent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                arguments: json!({ "path": "README.md" }),
                thought_signature: None,
            })],
            &model,
        );
        let context = Context {
            messages: vec![
                Message::user_text("Read README.md"),
                Message::Assistant(assistant),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "read".to_string(),
                    content: vec![ToolResultContent::text("contents")],
                    details: None,
                    is_error: false,
                    timestamp: 3,
                }),
            ],
            ..Default::default()
        };

        let messages = convert_messages(&model, &context, &compat);
        let replayed_assistant = messages
            .iter()
            .find(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
            .expect("assistant message");

        assert_eq!(replayed_assistant["reasoning_content"], json!(""));
    }

    #[test]
    fn replays_thinking_with_source_signature() {
        let mut model = model();
        model.id = "gpt-4o-mini".to_string();
        let compat = get_compat(&model);
        let assistant = assistant_message(
            vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "think".to_string(),
                    thinking_signature: Some("reasoning_content".to_string()),
                    redacted: None,
                }),
                AssistantContent::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "read".to_string(),
                    arguments: json!({ "path": "README.md" }),
                    thought_signature: None,
                }),
            ],
            &model,
        );
        let context = Context {
            messages: vec![Message::Assistant(assistant)],
            ..Default::default()
        };

        let messages = convert_messages(&model, &context, &compat);

        assert_eq!(messages[0]["reasoning_content"], json!("think"));
        assert!(messages[0].get("reasoning").is_none());
    }

    fn tool_result(tool_call_id: &str, timestamp: u64) -> ToolResultMessage {
        ToolResultMessage {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "read".to_string(),
            content: vec![
                ToolResultContent::text("Read image file [image/png]"),
                ToolResultContent::Image(ImageContent {
                    data: "ZmFrZQ==".to_string(),
                    mime_type: "image/png".to_string(),
                }),
            ],
            details: None,
            is_error: false,
            timestamp,
        }
    }

    #[test]
    fn serializes_same_model_thinking_plus_text_as_text_parts() {
        let model = model();
        let mut compat = get_compat(&model);
        compat.requires_thinking_as_text = true;
        let assistant = assistant_message(
            vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "internal reasoning".to_string(),
                    thinking_signature: None,
                    redacted: None,
                }),
                AssistantContent::Text(TextContent {
                    text: "visible answer".to_string(),
                    text_signature: None,
                }),
            ],
            &model,
        );
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::user_text("hello"),
                Message::Assistant(assistant),
                Message::user_text("continue"),
            ],
            tools: Vec::new(),
        };

        let messages = convert_messages(&model, &context, &compat);

        assert_eq!(
            messages[1],
            json!({
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "internal reasoning" },
                    { "type": "text", "text": "visible answer" }
                ]
            })
        );
    }

    #[test]
    fn serializes_same_model_thinking_only_as_text_parts() {
        let model = model();
        let mut compat = get_compat(&model);
        compat.requires_thinking_as_text = true;
        let assistant = assistant_message(
            vec![AssistantContent::Thinking(ThinkingContent {
                thinking: "internal reasoning".to_string(),
                thinking_signature: None,
                redacted: None,
            })],
            &model,
        );
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::user_text("hello"),
                Message::Assistant(assistant),
                Message::user_text("continue"),
            ],
            tools: Vec::new(),
        };

        let messages = convert_messages(&model, &context, &compat);

        assert_eq!(
            messages[1],
            json!({
                "role": "assistant",
                "content": [{ "type": "text", "text": "internal reasoning" }]
            })
        );
    }

    #[tokio::test]
    async fn thinking_as_text_replay_reaches_endpoint() {
        let captured_payload = Arc::new(Mutex::new(None));
        let body = chat_sse_body(&[
            json!({
                "id": "chatcmpl-thinking-text",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-5.5",
                "choices": [{ "index": 0, "delta": { "content": "ok" }, "finish_reason": null }]
            }),
            json!({
                "id": "chatcmpl-thinking-text",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-5.5",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
            }),
        ]);
        let base_url = spawn_capturing_sse_server(body, Arc::clone(&captured_payload)).await;
        let mut model = model();
        model.base_url = base_url;
        model.compat.openai_completions.requires_thinking_as_text = Some(true);
        let assistant = assistant_message(
            vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "internal reasoning".to_string(),
                    thinking_signature: None,
                    redacted: None,
                }),
                AssistantContent::Text(TextContent {
                    text: "visible answer".to_string(),
                    text_signature: None,
                }),
            ],
            &model,
        );

        let mut stream = stream_openai_completions(
            model,
            Context {
                messages: vec![
                    Message::user_text("hello"),
                    Message::Assistant(assistant),
                    Message::user_text("continue"),
                ],
                ..Default::default()
            },
            OpenAICompletionsOptions {
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
        let payload = captured_payload
            .lock()
            .unwrap()
            .clone()
            .expect("captured payload");

        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(
            payload["messages"][1],
            json!({
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "internal reasoning" },
                    { "type": "text", "text": "visible answer" }
                ]
            })
        );
    }

    #[test]
    fn batches_tool_result_images_after_consecutive_tool_results() {
        let model = model();
        let compat = get_compat(&model);
        let assistant = assistant_message(
            vec![
                AssistantContent::ToolCall(ToolCall {
                    id: "tool-1".to_string(),
                    name: "read".to_string(),
                    arguments: json!({ "path": "img-1.png" }),
                    thought_signature: None,
                }),
                AssistantContent::ToolCall(ToolCall {
                    id: "tool-2".to_string(),
                    name: "read".to_string(),
                    arguments: json!({ "path": "img-2.png" }),
                    thought_signature: None,
                }),
            ],
            &model,
        );
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::user_text("Read the images"),
                Message::Assistant(assistant),
                Message::ToolResult(tool_result("tool-1", 3)),
                Message::ToolResult(tool_result("tool-2", 4)),
            ],
            tools: Vec::new(),
        };

        let messages = convert_messages(&model, &context, &compat);
        let roles = messages
            .iter()
            .filter_map(|message| message.get("role").and_then(Value::as_str))
            .collect::<Vec<_>>();
        let image_parts = messages
            .last()
            .and_then(|message| message.get("content"))
            .and_then(Value::as_array)
            .expect("image user content")
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("image_url"))
            .count();

        assert_eq!(roles, ["user", "assistant", "tool", "tool", "user"]);
        assert_eq!(image_parts, 2);
    }

    #[test]
    fn skipped_empty_assistant_preserves_tool_result_bridge_state() {
        let model = model();
        let mut compat = get_compat(&model);
        compat.requires_assistant_after_tool_result = true;
        let empty_assistant = AssistantMessage::empty_for(&model);
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "tool-1".to_string(),
                    tool_name: "read".to_string(),
                    content: vec![ToolResultContent::text("done")],
                    details: None,
                    is_error: false,
                    timestamp: 1,
                }),
                Message::Assistant(empty_assistant),
                Message::user_text("next"),
            ],
            tools: Vec::new(),
        };

        let messages = convert_messages(&model, &context, &compat);
        let roles = messages
            .iter()
            .filter_map(|message| message.get("role").and_then(Value::as_str))
            .collect::<Vec<_>>();

        assert_eq!(roles, ["tool", "assistant", "user"]);
        assert_eq!(
            messages[1]["content"],
            json!("I have processed the tool results.")
        );
    }

    #[test]
    fn applies_anthropic_cache_markers_when_compat_enables_them() {
        let mut model = model();
        model.provider = "openrouter".to_string();
        model.compat = ModelCompat {
            openai_completions: OpenAICompletionsCompat {
                cache_control_format: Some(CacheControlFormat::Anthropic),
                ..Default::default()
            },
            ..Default::default()
        };
        let compat = get_compat(&model);
        let context = Context {
            system_prompt: Some("System prompt".to_string()),
            messages: vec![Message::user_text("Hello")],
            tools: vec![Tool {
                name: "read".to_string(),
                description: "Read a file".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            }],
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &compat,
            CacheRetention::Short,
        );

        assert_eq!(
            payload["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(
            payload["tools"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(
            payload["messages"][1]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn applies_anthropic_cache_markers_when_explicit_compat_enables_them() {
        let mut model = model();
        model.provider = "openrouter".to_string();
        model.id = "anthropic/claude-sonnet-4.5".to_string();
        model.base_url = "https://openrouter.ai/api/v1".to_string();
        model.compat.openai_completions.cache_control_format = Some(CacheControlFormat::Anthropic);
        let context = Context {
            system_prompt: Some("System prompt".to_string()),
            messages: vec![Message::user_text("Hello")],
            tools: vec![Tool {
                name: "read".to_string(),
                description: "Read a file".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            }],
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(
            payload["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(
            payload["tools"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(
            payload["messages"][1]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn omits_anthropic_cache_markers_when_cache_retention_is_none() {
        let mut model = model();
        model.provider = "openrouter".to_string();
        model.compat = ModelCompat {
            openai_completions: OpenAICompletionsCompat {
                cache_control_format: Some(CacheControlFormat::Anthropic),
                ..Default::default()
            },
            ..Default::default()
        };
        let compat = get_compat(&model);
        let context = Context {
            system_prompt: Some("System prompt".to_string()),
            messages: vec![Message::user_text("Hello")],
            tools: vec![Tool {
                name: "read".to_string(),
                description: "Read a file".to_string(),
                parameters: json!({ "type": "object" }),
            }],
        };

        let payload = build_chat_completions_payload(
            &model,
            &context,
            &OpenAICompletionsOptions::default(),
            &compat,
            CacheRetention::None,
        );

        assert!(payload["messages"][0]["content"].as_str().is_some());
        assert!(payload["tools"][0].get("cache_control").is_none());
        assert!(payload["messages"][1]["content"].as_str().is_some());
    }
}
