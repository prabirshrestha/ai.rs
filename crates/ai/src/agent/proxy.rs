//! Port of `packages/agent/src/proxy.ts`.
//!
//! Proxy stream function for apps that route LLM calls through a server.
//! The server manages auth and proxies requests to LLM providers.
//!
//! Rust adaptations: `fetch` becomes an optional `http_client`; content
//! indices past the end of the partial message are padded with empty text
//! blocks (JavaScript arrays are sparse); the streaming JSON of a tool call
//! is tracked beside the partial message instead of on a hidden
//! `partialJson` field. Pi's `errorMessage` for an abort during `fetch` is
//! fetch's own `AbortError` message; Rust has no `fetch`, so every abort
//! reports "Request aborted by user" (the stop reason is `aborted` in both).

use std::collections::HashMap;
use std::future::Future;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    CacheRetention, Model, ProviderHeaders, SamplingParams, StopReason, TextContent,
    ThinkingBudgets, ThinkingContent, ThinkingLevel, ToolCall, TranscriptContext, Transport, Usage,
};
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::time::now_millis;

/// Proxy event types: the server sends these with the `partial` field
/// stripped to reduce bandwidth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProxyAssistantMessageEvent {
    #[serde(rename = "start")]
    Start,
    #[serde(rename = "text_start")]
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "text_end")]
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(
            rename = "contentSignature",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        content_signature: Option<String>,
    },
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(
            rename = "contentSignature",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        content_signature: Option<String>,
    },
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
    },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "toolCall")]
        tool_call: ToolCall,
    },
    /// `reason` is `Stop`, `Length` or `ToolUse`.
    ///
    /// A missing `usage` reads as zero usage: Pi copies the absent field
    /// through as `undefined`, which a Rust `Usage` cannot hold.
    #[serde(rename = "done")]
    Done {
        reason: StopReason,
        #[serde(default)]
        usage: Usage,
        #[serde(
            rename = "providerThinkingLevel",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        provider_thinking_level: Option<String>,
    },
    /// `reason` is `Aborted` or `Error`.
    #[serde(rename = "error")]
    Error {
        reason: StopReason,
        #[serde(
            rename = "errorMessage",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        error_message: Option<String>,
        /// A missing `usage` reads as zero usage (see `Done`).
        #[serde(default)]
        usage: Usage,
        #[serde(
            rename = "providerThinkingLevel",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        provider_thinking_level: Option<String>,
    },
}

/// Options for [`stream_proxy`].
#[derive(Clone, Default)]
pub struct ProxyStreamOptions {
    pub temperature: Option<f64>,
    pub sampling_params: Option<SamplingParams>,
    pub max_tokens: Option<u32>,
    pub reasoning: Option<ThinkingLevel>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub headers: Option<ProviderHeaders>,
    pub metadata: Option<serde_json::Map<String, Value>>,
    pub transport: Option<Transport>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
    /// Local abort signal for the proxy request.
    pub signal: Option<CancellationToken>,
    /// Auth token for the proxy server.
    pub auth_token: String,
    /// Proxy server URL (e.g. `"https://genai.example.com"`).
    pub proxy_url: String,
    /// Optional HTTP client (Pi's global `fetch`).
    pub http_client: Option<reqwest::Client>,
}

/// The JSON-serializable subset of [`ProxyStreamOptions`] sent to the server.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProxySerializableStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling_params: Option<SamplingParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ThinkingLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_retention: Option<CacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    headers: Option<ProviderHeaders>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transport: Option<Transport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budgets: Option<ThinkingBudgets>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_retry_delay_ms: Option<u64>,
}

fn build_proxy_request_options(options: &ProxyStreamOptions) -> ProxySerializableStreamOptions {
    ProxySerializableStreamOptions {
        temperature: options.temperature,
        sampling_params: options.sampling_params.clone(),
        max_tokens: options.max_tokens,
        reasoning: options.reasoning,
        cache_retention: options.cache_retention,
        session_id: options.session_id.clone(),
        headers: options.headers.clone(),
        metadata: options.metadata.clone(),
        transport: options.transport,
        thinking_budgets: options.thinking_budgets.clone(),
        max_retry_delay_ms: options.max_retry_delay_ms,
    }
}

#[derive(Serialize)]
struct ProxyRequestBody<'a> {
    model: &'a Model,
    context: &'a TranscriptContext,
    options: ProxySerializableStreamOptions,
}

/// Stream function that proxies through a server instead of calling LLM
/// providers directly. The server strips the partial field from delta
/// events to reduce bandwidth; the partial message is rebuilt client-side.
///
/// Use it as the `stream_fn` of an agent that needs to go through a proxy:
///
/// ```no_run
/// use ai::agent::{ProxyStreamOptions, stream_fn, stream_proxy};
///
/// let stream_fn = stream_fn(|model, context, options| {
///     stream_proxy(
///         model,
///         context,
///         ProxyStreamOptions {
///             reasoning: options.reasoning,
///             signal: options.signal.clone(),
///             auth_token: "token".to_string(),
///             proxy_url: "https://genai.example.com".to_string(),
///             ..Default::default()
///         },
///     )
/// });
/// ```
pub fn stream_proxy(
    model: Model,
    context: TranscriptContext,
    options: ProxyStreamOptions,
) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let producer = stream.clone();

    tokio::spawn(async move {
        // Initialize the partial message that we'll build up from events.
        let mut state = ProxyPartialState {
            partial: AssistantMessage {
                stop_reason: StopReason::Pending,
                timestamp: now_millis(),
                ..AssistantMessage::empty_for(&model)
            },
            partial_json: HashMap::new(),
        };

        match run_proxy_request(&producer, &model, &context, &options, &mut state).await {
            Ok(()) => producer.end(None),
            Err(error_message) => {
                let reason = if options
                    .signal
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled)
                {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                let mut partial = state.partial;
                partial.stop_reason = reason;
                partial.error_message = Some(error_message);
                producer.push(AssistantMessageEvent::Error {
                    reason,
                    error: partial,
                });
                producer.end(None);
            }
        }
    });

    stream
}

const ABORTED_BY_USER: &str = "Request aborted by user";

async fn run_proxy_request(
    stream: &AssistantMessageEventStream,
    model: &Model,
    context: &TranscriptContext,
    options: &ProxyStreamOptions,
    state: &mut ProxyPartialState,
) -> Result<(), String> {
    let client = options.http_client.clone().unwrap_or_default();
    let body = ProxyRequestBody {
        model,
        context,
        options: build_proxy_request_options(options),
    };
    let request = client
        .post(format!("{}/api/stream", options.proxy_url))
        .header("Authorization", format!("Bearer {}", options.auth_token))
        .header("Content-Type", "application/json")
        .json(&body)
        .send();
    let response = with_signal(options.signal.as_ref(), request)
        .await?
        .map_err(|error| error.to_string())?;

    if !response.status().is_success() {
        let status = response.status();
        let mut error_message = format!(
            "Proxy error: {} {}",
            status.as_u16(),
            status.canonical_reason().unwrap_or_default()
        );
        if let Ok(error_data) = response.json::<Value>().await
            && let Some(error) = error_data.get("error").and_then(Value::as_str)
            && !error.is_empty()
        {
            error_message = format!("Proxy error: {error}");
        }
        return Err(error_message);
    }

    let mut body = response.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();
    let mut saw_terminal_event = false;

    loop {
        let chunk = with_signal(options.signal.as_ref(), body.next()).await?;
        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|error| error.to_string())?;

        if is_cancelled(options.signal.as_ref()) {
            return Err(ABORTED_BY_USER.to_string());
        }

        buffer.extend_from_slice(&chunk);
        while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = buffer.drain(..=newline).collect();
            process_line(
                stream,
                &String::from_utf8_lossy(&line[..line.len() - 1]),
                state,
                &mut saw_terminal_event,
            )?;
        }
    }

    if is_cancelled(options.signal.as_ref()) {
        return Err(ABORTED_BY_USER.to_string());
    }

    // The final event may not be newline-terminated; process whatever is
    // left in the buffer.
    if !buffer.is_empty() {
        process_line(
            stream,
            &String::from_utf8_lossy(&buffer),
            state,
            &mut saw_terminal_event,
        )?;
    }

    if !saw_terminal_event {
        // A clean EOF without a done/error event means the server dropped the
        // response mid-stream. Surface it as an error instead of leaving
        // consumers waiting on a result that never arrives.
        state.partial.stop_reason = StopReason::Error;
        state.partial.error_message =
            Some("Connection closed by proxy server before the response completed".to_string());
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: state.partial.clone(),
        });
    }
    Ok(())
}

fn is_cancelled(signal: Option<&CancellationToken>) -> bool {
    signal.is_some_and(CancellationToken::is_cancelled)
}

async fn with_signal<T>(
    signal: Option<&CancellationToken>,
    future: impl Future<Output = T>,
) -> Result<T, String> {
    match signal {
        Some(signal) => tokio::select! {
            _ = signal.cancelled() => Err(ABORTED_BY_USER.to_string()),
            value = future => Ok(value),
        },
        None => Ok(future.await),
    }
}

/// The `type` tags of [`ProxyAssistantMessageEvent`].
const PROXY_EVENT_TYPES: [&str; 12] = [
    "start",
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
    "done",
    "error",
];

fn process_line(
    stream: &AssistantMessageEventStream,
    line: &str,
    state: &mut ProxyPartialState,
    saw_terminal_event: &mut bool,
) -> Result<(), String> {
    let Some(data) = line.strip_prefix("data: ") else {
        return Ok(());
    };
    let data = data.trim();
    if data.is_empty() {
        return Ok(());
    }
    let value: Value = serde_json::from_str(data).map_err(|error| error.to_string())?;
    // Pi's `switch` warns about an unknown event type and skips the event.
    let event_type = value.get("type").and_then(Value::as_str).unwrap_or("");
    if !PROXY_EVENT_TYPES.contains(&event_type) {
        eprintln!("Unhandled proxy event type: {event_type}");
        return Ok(());
    }
    let proxy_event: ProxyAssistantMessageEvent =
        serde_json::from_value(value).map_err(|error| error.to_string())?;
    if let Some(event) = process_proxy_event(proxy_event, state)? {
        if matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        ) {
            *saw_terminal_event = true;
        }
        stream.push(event);
    }
    Ok(())
}

struct ProxyPartialState {
    partial: AssistantMessage,
    /// `partialJson` of streaming tool calls, by content index.
    partial_json: HashMap<usize, String>,
}

impl ProxyPartialState {
    fn set_content(&mut self, index: usize, content: AssistantContent) {
        let blocks = &mut self.partial.content;
        while blocks.len() < index {
            blocks.push(AssistantContent::Text(TextContent::new("")));
        }
        if index < blocks.len() {
            blocks[index] = content;
        } else {
            blocks.push(content);
        }
    }
}

/// Process a proxy event and update the partial message.
fn process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    state: &mut ProxyPartialState,
) -> Result<Option<AssistantMessageEvent>, String> {
    Ok(Some(match proxy_event {
        ProxyAssistantMessageEvent::Start => AssistantMessageEvent::Start {
            partial: state.partial.clone(),
        },

        ProxyAssistantMessageEvent::TextStart { content_index } => {
            state.set_content(content_index, AssistantContent::Text(TextContent::new("")));
            AssistantMessageEvent::TextStart {
                content_index,
                partial: state.partial.clone(),
            }
        }

        ProxyAssistantMessageEvent::TextDelta {
            content_index,
            delta,
        } => match state.partial.content.get_mut(content_index) {
            Some(AssistantContent::Text(content)) => {
                content.text.push_str(&delta);
                AssistantMessageEvent::TextDelta {
                    content_index,
                    delta,
                    partial: state.partial.clone(),
                }
            }
            _ => return Err("Received text_delta for non-text content".to_string()),
        },

        ProxyAssistantMessageEvent::TextEnd {
            content_index,
            content_signature,
        } => match state.partial.content.get_mut(content_index) {
            Some(AssistantContent::Text(content)) => {
                content.text_signature = content_signature;
                let text = content.text.clone();
                AssistantMessageEvent::TextEnd {
                    content_index,
                    content: text,
                    partial: state.partial.clone(),
                }
            }
            _ => return Err("Received text_end for non-text content".to_string()),
        },

        ProxyAssistantMessageEvent::ThinkingStart { content_index } => {
            state.set_content(
                content_index,
                AssistantContent::Thinking(ThinkingContent::default()),
            );
            AssistantMessageEvent::ThinkingStart {
                content_index,
                partial: state.partial.clone(),
            }
        }

        ProxyAssistantMessageEvent::ThinkingDelta {
            content_index,
            delta,
        } => match state.partial.content.get_mut(content_index) {
            Some(AssistantContent::Thinking(content)) => {
                content.thinking.push_str(&delta);
                AssistantMessageEvent::ThinkingDelta {
                    content_index,
                    delta,
                    partial: state.partial.clone(),
                }
            }
            _ => return Err("Received thinking_delta for non-thinking content".to_string()),
        },

        ProxyAssistantMessageEvent::ThinkingEnd {
            content_index,
            content_signature,
        } => match state.partial.content.get_mut(content_index) {
            Some(AssistantContent::Thinking(content)) => {
                content.thinking_signature = content_signature;
                let thinking = content.thinking.clone();
                AssistantMessageEvent::ThinkingEnd {
                    content_index,
                    content: thinking,
                    partial: state.partial.clone(),
                }
            }
            _ => return Err("Received thinking_end for non-thinking content".to_string()),
        },

        ProxyAssistantMessageEvent::ToolCallStart {
            content_index,
            id,
            tool_name,
        } => {
            state.set_content(
                content_index,
                AssistantContent::ToolCall(ToolCall {
                    id,
                    name: tool_name,
                    arguments: Value::Object(Default::default()),
                    thought_signature: None,
                    namespace: None,
                }),
            );
            state.partial_json.insert(content_index, String::new());
            AssistantMessageEvent::ToolCallStart {
                content_index,
                partial: state.partial.clone(),
            }
        }

        ProxyAssistantMessageEvent::ToolCallDelta {
            content_index,
            delta,
        } => match state.partial.content.get_mut(content_index) {
            Some(AssistantContent::ToolCall(content)) => {
                let partial_json = state.partial_json.entry(content_index).or_default();
                partial_json.push_str(&delta);
                content.arguments = parse_streaming_json(Some(partial_json));
                AssistantMessageEvent::ToolCallDelta {
                    content_index,
                    delta,
                    partial: state.partial.clone(),
                }
            }
            _ => return Err("Received toolcall_delta for non-toolCall content".to_string()),
        },

        ProxyAssistantMessageEvent::ToolCallEnd {
            content_index,
            tool_call,
        } => match state.partial.content.get_mut(content_index) {
            Some(AssistantContent::ToolCall(content)) => {
                // `Object.assign(content, proxyEvent.toolCall)`.
                content.id = tool_call.id;
                content.name = tool_call.name;
                content.arguments = tool_call.arguments;
                if tool_call.thought_signature.is_some() {
                    content.thought_signature = tool_call.thought_signature;
                }
                if tool_call.namespace.is_some() {
                    content.namespace = tool_call.namespace;
                }
                let tool_call = content.clone();
                state.partial_json.remove(&content_index);
                AssistantMessageEvent::ToolCallEnd {
                    content_index,
                    tool_call,
                    partial: state.partial.clone(),
                }
            }
            _ => return Ok(None),
        },

        ProxyAssistantMessageEvent::Done {
            reason,
            usage,
            provider_thinking_level,
        } => {
            state.partial.stop_reason = reason;
            state.partial.usage = usage;
            if provider_thinking_level.is_some() {
                state.partial.provider_thinking_level = provider_thinking_level;
            }
            AssistantMessageEvent::Done {
                reason,
                message: state.partial.clone(),
            }
        }

        ProxyAssistantMessageEvent::Error {
            reason,
            error_message,
            usage,
            provider_thinking_level,
        } => {
            state.partial.stop_reason = reason;
            state.partial.error_message = error_message;
            state.partial.usage = usage;
            if provider_thinking_level.is_some() {
                state.partial.provider_thinking_level = provider_thinking_level;
            }
            AssistantMessageEvent::Error {
                reason,
                error: state.partial.clone(),
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    //! Port of `test/proxy.test.ts`. A local TCP server stands in for the
    //! stubbed `fetch`.

    use futures::StreamExt;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::types::{Context, ModelCost, ModelInput};
    use crate::utils::transcript::normalize_context;

    fn model() -> Model {
        Model {
            id: "gpt-5.4".to_string(),
            name: "GPT-5.4".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            reasoning: true,
            input: vec![ModelInput::Text],
            cost: ModelCost::default(),
            context_window: 400_000,
            max_tokens: 128_000,
            ..Default::default()
        }
    }

    fn usage() -> Value {
        serde_json::to_value(Usage::default()).unwrap()
    }

    async fn spawn_proxy(body: String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 4096];
            // Read the headers and the JSON body.
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                request.extend_from_slice(&buffer[..read]);
                let text = String::from_utf8_lossy(&request);
                if let Some(header_end) = text.find("\r\n\r\n") {
                    let length = text[..header_end]
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if request.len() >= header_end + 4 + length {
                        break;
                    }
                }
                if read == 0 {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
        });
        format!("http://{addr}")
    }

    fn options(proxy_url: String) -> ProxyStreamOptions {
        ProxyStreamOptions {
            auth_token: "test-token".to_string(),
            proxy_url,
            ..Default::default()
        }
    }

    fn empty_context() -> TranscriptContext {
        normalize_context(&Context {
            system_prompt: Some(String::new()),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn preserves_tool_call_metadata_received_only_on_toolcall_end() {
        let proxy_events = [
            json!({ "type": "start" }),
            json!({ "type": "toolcall_start", "contentIndex": 0, "id": "call_test|fc_test", "toolName": "lookup" }),
            json!({ "type": "toolcall_delta", "contentIndex": 0, "delta": "{\"value\":\"hello\"}" }),
            json!({
                "type": "toolcall_end",
                "contentIndex": 0,
                "toolCall": {
                    "type": "toolCall",
                    "id": "call_test|fc_test",
                    "name": "lookup",
                    "arguments": { "value": "hello" },
                    "namespace": "dynamic_tools",
                },
            }),
            json!({ "type": "done", "reason": "toolUse", "usage": usage() }),
        ];
        let body: String = proxy_events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect();
        let url = spawn_proxy(body).await;

        let stream = stream_proxy(model(), empty_context(), options(url));
        let events: Vec<_> = stream.clone().collect().await;
        let result = stream.result().await;
        let end_event = events
            .iter()
            .find_map(|event| match event {
                AssistantMessageEvent::ToolCallEnd { tool_call, .. } => Some(tool_call.clone()),
                _ => None,
            })
            .expect("toolcall_end");

        assert_eq!(end_event.namespace.as_deref(), Some("dynamic_tools"));
        let AssistantContent::ToolCall(tool_call) = &result.content[0] else {
            panic!("expected tool call");
        };
        assert_eq!(tool_call.arguments, json!({ "value": "hello" }));
        assert_eq!(tool_call.namespace.as_deref(), Some("dynamic_tools"));
    }

    // Regression tests for https://github.com/earendil-works/pi/issues/8996
    #[tokio::test]
    async fn processes_terminal_metadata_when_the_event_is_not_newline_terminated() {
        let start = format!("data: {}\n\n", json!({ "type": "start" }));
        let done = format!(
            "data: {}",
            json!({ "type": "done", "reason": "stop", "usage": usage(), "providerThinkingLevel": "high" })
        );
        let url = spawn_proxy(start + &done).await;

        let stream = stream_proxy(model(), empty_context(), options(url));
        let events: Vec<_> = stream.clone().collect().await;
        let result = stream.result().await;

        let types: Vec<_> = events
            .iter()
            .map(AssistantMessageEvent::event_type)
            .collect();
        assert_eq!(types, ["start", "done"]);
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.provider_thinking_level.as_deref(), Some("high"));
    }

    #[tokio::test]
    async fn emits_an_error_instead_of_hanging_when_the_stream_ends_without_a_terminal_event() {
        let body = format!("data: {}\n\n", json!({ "type": "start" }));
        let url = spawn_proxy(body).await;

        let stream = stream_proxy(model(), empty_context(), options(url));
        let events: Vec<_> = stream.clone().collect().await;
        let result = stream.result().await;

        let types: Vec<_> = events
            .iter()
            .map(AssistantMessageEvent::event_type)
            .collect();
        assert_eq!(types, ["start", "error"]);
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(
            result
                .error_message
                .as_deref()
                .unwrap_or_default()
                .contains("Connection closed by proxy server")
        );
    }

    // Rust-only: Pi's `switch` skips unknown event types with a warning, and
    // a `done` without `usage` does not fail `JSON.parse`.
    #[tokio::test]
    async fn skips_unknown_event_types_and_accepts_done_without_usage() {
        let events = [
            json!({ "type": "start" }),
            json!({ "type": "future_event", "contentIndex": 0 }),
            json!({ "contentIndex": 0 }),
            json!({ "type": "text_start", "contentIndex": 0 }),
            json!({ "type": "text_delta", "contentIndex": 0, "delta": "hi" }),
            json!({ "type": "text_end", "contentIndex": 0 }),
            json!({ "type": "done", "reason": "stop" }),
        ];
        let body: String = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect();
        let url = spawn_proxy(body).await;

        let stream = stream_proxy(model(), empty_context(), options(url));
        let events: Vec<_> = stream.clone().collect().await;
        let result = stream.result().await;

        let types: Vec<_> = events
            .iter()
            .map(AssistantMessageEvent::event_type)
            .collect();
        assert_eq!(
            types,
            ["start", "text_start", "text_delta", "text_end", "done"]
        );
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.usage, Usage::default());
        let AssistantContent::Text(text) = &result.content[0] else {
            panic!("expected text");
        };
        assert_eq!(text.text, "hi");
    }
}
