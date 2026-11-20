//! Port of `harness-fx/src/fx-transport.ts`: the fetch fx's core sends its
//! model requests through. It points AI Gateway requests at the configured
//! base URL with the session's headers, applies the turn's response format,
//! numbers the model requests and reads each response's complete tool calls
//! before the core sees them.
//!
//! [rust] Not ported: the request fingerprints and recorded responses a
//! continued turn replays, since ai.rs sessions do not suspend turns.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures::StreamExt as _;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::libfx::fetch::{FxBody, FxFetch, FxFetchError, FxHttpRequest, FxHttpResponse};
use crate::harness::session::ResponseFormat;

/// `FxProvider`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FxProvider {
    Gateway,
    Codex,
    Grok,
}

impl FxProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gateway => "gateway",
            Self::Codex => "codex",
            Self::Grok => "grok",
        }
    }
}

/// One complete tool call of a model response.
#[derive(Debug, Clone, PartialEq)]
pub struct FxToolCall {
    pub tool_call_id: String,
    pub tool_name: String,
    /// The arguments, as the provider sent them (usually JSON text).
    pub input: Value,
}

/// What the transport tells the session, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum FxTransportSignal {
    /// `onRequest`: model request `index` starts.
    Request { index: usize, model_id: Option<String> },
    /// `beforeRelease`: request `index` finished with these tool calls,
    /// which the core is about to see.
    ToolCalls { index: usize, calls: Vec<FxToolCall> },
    /// A subscription provider rejected request `index`.
    Rejected {
        index: usize,
        model_id: Option<String>,
        status: u16,
    },
}

pub type FxTransportSink = Arc<dyn Fn(FxTransportSignal) + Send + Sync>;

const GATEWAY_ORIGIN: &str = "https://ai-gateway.vercel.sh";
const CODEX_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex/responses";
const GROK_ENDPOINT: &str = "https://cli-chat-proxy.grok.com/v1/responses";

/// `parseFxEvent(frame)`: an SSE frame's JSON data.
pub fn parse_fx_event(frame: &str) -> Option<Value> {
    let data = frame
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() || data == "[DONE]" {
        return None;
    }
    serde_json::from_str(&data).ok()
}

fn part_type(part: Option<&Value>) -> &str {
    part.and_then(|part| part.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_default()
}

/// The URL without its query.
fn without_query(url: &str) -> &str {
    url.split(['?', '#']).next().unwrap_or(url)
}

fn origin_and_path(url: &str) -> Option<(String, String, String)> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let suffix = parsed.query().map(|query| format!("?{query}")).unwrap_or_default();
    Some((parsed.origin().ascii_serialization(), parsed.path().to_string(), suffix))
}

/// `createFxTransportProtocol({ provider })`.
struct Protocol {
    provider: FxProvider,
}

impl Protocol {
    fn label(&self) -> &'static str {
        match self.provider {
            FxProvider::Gateway => "Gateway",
            FxProvider::Codex => "codex",
            FxProvider::Grok => "grok",
        }
    }

    fn preserve_rejected_responses(&self) -> bool {
        self.provider != FxProvider::Gateway
    }

    fn is_gateway_request(&self, url: &str) -> bool {
        self.provider == FxProvider::Gateway
            && origin_and_path(url).is_some_and(|(origin, _, _)| origin == GATEWAY_ORIGIN)
    }

    fn rewrite_url(&self, url: &str, base_url: Option<&str>) -> String {
        if !self.is_gateway_request(url) {
            return url.to_string();
        }
        let Some((_, path, query)) = origin_and_path(url) else {
            return url.to_string();
        };
        let base = base_url.unwrap_or(GATEWAY_ORIGIN);
        let base = if base.ends_with('/') {
            base.to_string()
        } else {
            format!("{base}/")
        };
        reqwest::Url::parse(&base)
            .and_then(|base| base.join(&format!("{}{query}", &path[1..])))
            .map_or_else(|_| url.to_string(), |url| url.to_string())
    }

    fn is_model_request(&self, url: &str, gateway_request: bool) -> bool {
        match self.provider {
            FxProvider::Gateway => {
                gateway_request
                    && origin_and_path(url).is_some_and(|(_, path, _)| path.ends_with("/ai/language-model"))
            }
            FxProvider::Codex => without_query(url) == CODEX_ENDPOINT,
            FxProvider::Grok => without_query(url) == GROK_ENDPOINT,
        }
    }

    fn prepare_body(&self, body: String, response_format: Option<&ResponseFormat>) -> String {
        let Some(ResponseFormat::Json {
            schema,
            name,
            description,
        }) = response_format
        else {
            return body;
        };
        let Ok(Value::Object(mut parsed)) = serde_json::from_str::<Value>(&body) else {
            return body;
        };
        if self.provider == FxProvider::Gateway {
            let mut format = Map::new();
            format.insert("type".into(), json!("json"));
            if let Some(schema) = schema {
                format.insert("schema".into(), schema.clone());
            }
            if let Some(name) = name {
                format.insert("name".into(), json!(name));
            }
            if let Some(description) = description {
                format.insert("description".into(), json!(description));
            }
            parsed.insert("responseFormat".into(), Value::Object(format));
            return Value::Object(parsed).to_string();
        }
        let mut text = match parsed.get("text") {
            Some(Value::Object(text)) => text.clone(),
            _ => Map::new(),
        };
        let format = match schema {
            None => json!({ "type": "json_object" }),
            Some(schema) => {
                let mut format = Map::new();
                format.insert("type".into(), json!("json_schema"));
                format.insert("name".into(), json!(name.clone().unwrap_or_else(|| "response".into())));
                if let Some(description) = description {
                    format.insert("description".into(), json!(description));
                }
                format.insert("schema".into(), schema.clone());
                format.insert("strict".into(), json!(true));
                Value::Object(format)
            }
        };
        text.insert("format".into(), format);
        parsed.insert("text".into(), Value::Object(text));
        Value::Object(parsed).to_string()
    }

    fn model_id(&self, request: &FxHttpRequest, body: &str) -> Option<String> {
        match self.provider {
            FxProvider::Gateway => request.header("ai-language-model-id").map(str::to_string),
            _ => serde_json::from_str::<Value>(body)
                .ok()?
                .get("model")?
                .as_str()
                .map(str::to_string),
        }
    }

    fn is_terminal_event(&self, part: Option<&Value>) -> bool {
        let kind = part_type(part);
        match self.provider {
            FxProvider::Gateway => kind == "finish" || kind == "error",
            _ => matches!(
                kind,
                "response.completed" | "response.done" | "response.incomplete" | "response.failed"
            ),
        }
    }

    fn should_hold_frame(&self, part: Option<&Value>, terminal: bool) -> bool {
        let Some(part) = part else {
            return false;
        };
        let kind = part_type(Some(part));
        let tool = kind.starts_with("tool-") || terminal;
        match self.provider {
            FxProvider::Gateway => tool,
            _ => {
                tool || kind.starts_with("response.function_call_arguments.")
                    || part
                        .get("item")
                        .and_then(|item| item.get("type"))
                        .and_then(Value::as_str)
                        == Some("function_call")
            }
        }
    }
}

/// `collectToolCalls({ record, index })`: the complete tool calls of a
/// response body.
pub fn collect_tool_calls(body: &str) -> Vec<FxToolCall> {
    let mut calls: IndexMap<String, FxToolCall> = IndexMap::new();
    let normalized = body.replace("\r\n", "\n");
    for frame in normalized.split("\n\n") {
        let Some(part) = parse_fx_event(frame) else {
            continue;
        };
        let kind = part_type(Some(&part));
        if kind == "tool-call"
            && part.get("providerExecuted") != Some(&Value::Bool(true))
            && let (Some(id), Some(name)) = (
                part.get("toolCallId").and_then(Value::as_str),
                part.get("toolName").and_then(Value::as_str),
            )
        {
            calls.insert(
                id.to_string(),
                FxToolCall {
                    tool_call_id: id.to_string(),
                    tool_name: name.to_string(),
                    input: part.get("input").cloned().unwrap_or(Value::Null),
                },
            );
        }
        if !matches!(kind, "response.completed" | "response.done") {
            continue;
        }
        let Some(response) = part.get("response").filter(|response| response.is_object()) else {
            continue;
        };
        let completed = match response.get("status") {
            None | Some(Value::Null) => true,
            Some(status) => status.as_str() == Some("completed"),
        };
        let Some(output) = response.get("output").and_then(Value::as_array) else {
            continue;
        };
        if !completed {
            continue;
        }
        for item in output {
            if item.get("type").and_then(Value::as_str) == Some("function_call")
                && let (Some(id), Some(name)) = (
                    item.get("call_id").and_then(Value::as_str),
                    item.get("name").and_then(Value::as_str),
                )
            {
                calls.insert(
                    id.to_string(),
                    FxToolCall {
                        tool_call_id: id.to_string(),
                        tool_name: name.to_string(),
                        input: item.get("arguments").cloned().unwrap_or(Value::Null),
                    },
                );
            }
        }
    }
    calls.into_values().collect()
}

/// `createFxTransport(options)` options.
pub struct FxTransportOptions {
    pub base_url: Option<String>,
    pub provider: FxProvider,
    pub headers: IndexMap<String, String>,
    pub response_format: Option<ResponseFormat>,
    pub signals: FxTransportSink,
    pub fetch: FxFetch,
}

struct State {
    request_index: Option<usize>,
}

/// `createFxTransport`.
pub struct FxTransport {
    options: FxTransportOptions,
    protocol: Protocol,
    state: Mutex<State>,
    active: AtomicBool,
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    stopped: CancellationToken,
}

fn suspended() -> FxFetchError {
    FxFetchError::Failed("fx turn is suspended.".into())
}

/// The first `\r?\n\r?\n` in `text`: where it starts and ends.
fn frame_end(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\n' {
            let mut next = index + 1;
            if bytes.get(next) == Some(&b'\r') {
                next += 1;
            }
            if bytes.get(next) == Some(&b'\n') {
                return Some(next + 1);
            }
        }
        index += 1;
    }
    None
}

impl FxTransport {
    pub fn new(options: FxTransportOptions) -> Arc<Self> {
        let protocol = Protocol {
            provider: options.provider,
        };
        Arc::new(Self {
            options,
            protocol,
            state: Mutex::new(State { request_index: None }),
            active: AtomicBool::new(true),
            tasks: Mutex::default(),
            stopped: CancellationToken::new(),
        })
    }

    /// The index of the last model request, if any.
    pub fn request_index(&self) -> Option<usize> {
        self.state.lock().unwrap().request_index
    }

    /// `stop()`.
    pub async fn stop(&self) {
        self.active.store(false, Ordering::SeqCst);
        self.stopped.cancel();
        let tasks = std::mem::take(&mut *self.tasks.lock().unwrap());
        for task in tasks {
            let _ = task.await;
        }
    }

    /// The transport's `fetch`.
    pub fn fetch(self: &Arc<Self>) -> FxFetch {
        let transport = self.clone();
        Arc::new(move |request, signal| {
            let transport = transport.clone();
            // The request is numbered when the core sends it, so its number
            // is in order with the output the core sent before it.
            let prepared = transport.prepare(request);
            Box::pin(async move { transport.send(prepared, signal).await })
        })
    }

    fn prepare(&self, mut request: FxHttpRequest) -> Result<Prepared, FxFetchError> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(suspended());
        }
        let original = request.url.clone();
        let gateway_request = self.protocol.is_gateway_request(&original);
        request.url = self
            .protocol
            .rewrite_url(&original, self.options.base_url.as_deref());
        if gateway_request {
            for (name, value) in &self.options.headers {
                request.set_header(name, value.clone());
            }
        }
        request.method = request.method.to_ascii_uppercase();
        if request.method != "POST" || !self.protocol.is_model_request(&original, gateway_request) {
            return Ok(Prepared::Passthrough(request));
        }
        let index = {
            let mut state = self.state.lock().unwrap();
            let index = state.request_index.map_or(0, |index| index + 1);
            state.request_index = Some(index);
            index
        };
        let body = self
            .protocol
            .prepare_body(request.body_text(), self.options.response_format.as_ref());
        let model_id = self.protocol.model_id(&request, &body);
        request.body = Some(Bytes::from(body));
        (self.options.signals)(FxTransportSignal::Request {
            index,
            model_id: model_id.clone(),
        });
        Ok(Prepared::Model {
            request,
            index,
            model_id,
        })
    }

    /// Takes back request `index`'s number.
    fn unnumber(&self, index: usize) {
        let mut state = self.state.lock().unwrap();
        if state.request_index == Some(index) {
            state.request_index = index.checked_sub(1);
        }
    }

    async fn send(
        self: Arc<Self>,
        prepared: Result<Prepared, FxFetchError>,
        signal: CancellationToken,
    ) -> Result<FxHttpResponse, FxFetchError> {
        let (request, index, model_id) = match prepared? {
            Prepared::Passthrough(request) => return (self.options.fetch)(request, signal).await,
            Prepared::Model {
                request,
                index,
                model_id,
            } => (request, index, model_id),
        };
        let response = match (self.options.fetch)(request, signal.clone()).await {
            Ok(response) => response,
            Err(error) => {
                self.unnumber(index);
                return Err(error);
            }
        };
        // libfx authorizes subscription requests around this transport: it
        // gets a rejected response as it was, to refresh a token or mask it.
        if self.protocol.preserve_rejected_responses() && !response.ok() {
            (self.options.signals)(FxTransportSignal::Rejected {
                index,
                model_id,
                status: response.status,
            });
            self.unnumber(index);
            return Ok(response);
        }
        if !response.ok() {
            return Ok(response);
        }
        let FxHttpResponse {
            status,
            headers,
            body,
        } = response;
        let (sender, receiver) = mpsc::channel::<Result<Bytes, FxFetchError>>(16);
        let transport = self.clone();
        let task = tokio::spawn(async move {
            transport.pump(index, body, sender, signal).await;
        });
        self.tasks.lock().unwrap().push(task);
        let body: FxBody = futures::stream::unfold(receiver, |mut receiver| async move {
            receiver.recv().await.map(|item| (item, receiver))
        })
        .boxed();
        Ok(FxHttpResponse {
            status,
            headers,
            body,
        })
    }

    /// Streams the response to the core, holding its tool and terminal
    /// frames until the whole response arrived and its tool calls are out.
    async fn pump(
        &self,
        index: usize,
        mut body: FxBody,
        sender: mpsc::Sender<Result<Bytes, FxFetchError>>,
        signal: CancellationToken,
    ) {
        let mut record = Vec::<u8>::new();
        let mut undecoded = Vec::<u8>::new();
        let mut pending = String::new();
        let mut holding = false;
        let mut terminal = false;
        let mut held = String::new();
        let result: Result<(), FxFetchError> = async {
            loop {
                let chunk = tokio::select! {
                    chunk = body.next() => chunk,
                    _ = signal.cancelled() => return Err(FxFetchError::Aborted),
                    _ = self.stopped.cancelled() => return Err(suspended()),
                };
                if !self.active.load(Ordering::SeqCst) {
                    return Err(suspended());
                }
                let done = chunk.is_none();
                if let Some(chunk) = chunk {
                    let chunk = chunk?;
                    record.extend_from_slice(&chunk);
                    undecoded.extend_from_slice(&chunk);
                }
                // `TextDecoder` with `stream: true`: a character split across
                // chunks waits for its last bytes.
                let valid = match std::str::from_utf8(&undecoded) {
                    Ok(_) => undecoded.len(),
                    Err(error) if !done && error.error_len().is_none() => error.valid_up_to(),
                    Err(_) => undecoded.len(),
                };
                let rest = undecoded.split_off(valid);
                pending.push_str(&String::from_utf8_lossy(&undecoded));
                undecoded = rest;
                while let Some(end) = frame_end(&pending) {
                    let frame: String = pending.drain(..end).collect();
                    let part = parse_fx_event(&frame);
                    let terminal_part = self.protocol.is_terminal_event(part.as_ref());
                    if terminal_part {
                        terminal = true;
                    }
                    if self.protocol.should_hold_frame(part.as_ref(), terminal_part) {
                        holding = true;
                    }
                    if holding {
                        held.push_str(&frame);
                    } else if sender.send(Ok(Bytes::from(frame))).await.is_err() {
                        return Err(FxFetchError::Aborted);
                    }
                }
                if done {
                    break;
                }
            }
            if !pending.trim().is_empty() {
                return Err(FxFetchError::Failed(format!(
                    "{} stream ended with an incomplete event.",
                    self.protocol.label()
                )));
            }
            if !terminal {
                return Err(FxFetchError::Failed(format!(
                    "{} stream ended before a terminal event.",
                    self.protocol.label()
                )));
            }
            let calls = collect_tool_calls(&String::from_utf8_lossy(&record));
            (self.options.signals)(FxTransportSignal::ToolCalls { index, calls });
            if !self.active.load(Ordering::SeqCst) {
                return Err(suspended());
            }
            if !held.is_empty() {
                let _ = sender.send(Ok(Bytes::from(std::mem::take(&mut held)))).await;
            }
            Ok(())
        }
        .await;
        if let Err(error) = result {
            let _ = sender.send(Err(error)).await;
        }
    }
}

enum Prepared {
    Passthrough(FxHttpRequest),
    Model {
        request: FxHttpRequest,
        index: usize,
        model_id: Option<String>,
    },
}

/// The tool calls of each request, as the session reads them.
pub type FxToolCalls = HashMap<usize, Vec<FxToolCall>>;

#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;
