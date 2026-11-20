//! [rust] The MCP client fx uses for remote servers, standing in for
//! `@ai-sdk/mcp`'s `createMCPClient`: JSON-RPC over Streamable HTTP
//! (`http`) or the legacy HTTP+SSE transport (`sse`), with the requests fx
//! needs (`initialize`, `tools/list`, `tools/call`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt as _;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::harness::error::{HarnessError, HarnessResult};

const PROTOCOL_VERSION: &str = "2025-06-18";

/// `transport.type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxMcpTransportKind {
    Http,
    Sse,
}

/// `createMCPClient({ transport })`'s transport.
#[derive(Debug, Clone, PartialEq)]
pub struct FxMcpTransportConfig {
    pub kind: FxMcpTransportKind,
    pub url: String,
    pub headers: Option<IndexMap<String, String>>,
}

/// The parts of an MCP client fx uses.
#[async_trait]
pub trait FxMcpClient: Send + Sync {
    /// The server's `instructions` from `initialize`.
    fn instructions(&self) -> Option<String>;
    /// `tools/list` with an optional cursor; the raw result.
    async fn list_tools(&self, cursor: Option<String>) -> HarnessResult<Value>;
    /// `tools/call`; the raw result.
    async fn call_tool(&self, name: &str, arguments: Value, signal: CancellationToken) -> HarnessResult<Value>;
    async fn close(&self);
}

/// `createMCPClient`: connects and initializes a client.
pub type FxMcpConnector = Arc<
    dyn Fn(FxMcpTransportConfig, Option<CancellationToken>) -> BoxFuture<'static, HarnessResult<Arc<dyn FxMcpClient>>>
        + Send
        + Sync,
>;

/// The connector over [`reqwest`].
pub fn default_mcp_connector() -> FxMcpConnector {
    Arc::new(|config, signal| {
        Box::pin(async move {
            let client: Arc<dyn FxMcpClient> = match config.kind {
                FxMcpTransportKind::Http => Arc::new(HttpMcpClient::connect(config, signal).await?),
                FxMcpTransportKind::Sse => Arc::new(SseMcpClient::connect(config, signal).await?),
            };
            Ok(client)
        })
    })
}

fn request_body(id: u64, method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}

fn initialize_params() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": { "name": "ai-sdk-mcp-client", "version": env!("CARGO_PKG_VERSION") },
    })
}

/// The `result` of a JSON-RPC response, or its error.
fn response_result(message: Value) -> HarnessResult<Value> {
    if let Some(error) = message.get("error") {
        let text = error.get("message").and_then(Value::as_str).unwrap_or("MCP request failed");
        return Err(HarnessError::new(format!("MCP error: {text}")).into());
    }
    Ok(message.get("result").cloned().unwrap_or(Value::Null))
}

fn apply_headers(mut request: reqwest::RequestBuilder, headers: &Option<IndexMap<String, String>>) -> reqwest::RequestBuilder {
    for (name, value) in headers.iter().flatten() {
        request = request.header(name, value);
    }
    request
}

async fn cancellable<T>(signal: Option<&CancellationToken>, future: impl Future<Output = HarnessResult<T>>) -> HarnessResult<T> {
    match signal {
        Some(signal) => tokio::select! {
            _ = signal.cancelled() => Err(HarnessError::new("This operation was aborted").into()),
            result = future => result,
        },
        None => future.await,
    }
}

/// Streamable HTTP.
struct HttpMcpClient {
    http: reqwest::Client,
    config: FxMcpTransportConfig,
    session_id: Mutex<Option<String>>,
    next_id: AtomicU64,
    instructions: Option<String>,
}

impl HttpMcpClient {
    async fn connect(config: FxMcpTransportConfig, signal: Option<CancellationToken>) -> HarnessResult<Self> {
        let mut client = Self {
            http: reqwest::Client::new(),
            config,
            session_id: Mutex::new(None),
            next_id: AtomicU64::new(0),
            instructions: None,
        };
        let result = cancellable(signal.as_ref(), client.request("initialize", initialize_params(), signal.clone())).await?;
        client.instructions = result.get("instructions").and_then(Value::as_str).map(str::to_string);
        client
            .send(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }), None)
            .await?;
        Ok(client)
    }

    fn post(&self, body: &Value) -> reqwest::RequestBuilder {
        let mut request = apply_headers(self.http.post(&self.config.url), &self.config.headers)
            .header("accept", "application/json, text/event-stream")
            .header("content-type", "application/json")
            .header("mcp-protocol-version", PROTOCOL_VERSION)
            .body(body.to_string());
        if let Some(session_id) = self.session_id.lock().unwrap().clone() {
            request = request.header("mcp-session-id", session_id);
        }
        request
    }

    /// Sends a message and returns the response, if the server sent one.
    async fn send(&self, body: Value, signal: Option<CancellationToken>) -> HarnessResult<Option<Value>> {
        let id = body.get("id").cloned();
        let response = self.post(&body).send().await?;
        if let Some(session_id) = response.headers().get("mcp-session-id").and_then(|value| value.to_str().ok()) {
            *self.session_id.lock().unwrap() = Some(session_id.to_string());
        }
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(HarnessError::new(format!("MCP HTTP Error: {status} {text}")).into());
        }
        let Some(id) = id else {
            return Ok(None);
        };
        let streamed = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream"));
        if !streamed {
            let message: Value = serde_json::from_slice(&response.bytes().await?)?;
            let messages = match message {
                Value::Array(messages) => messages,
                message => vec![message],
            };
            return Ok(messages.into_iter().find(|message| message.get("id") == Some(&id)));
        }
        let mut events = std::pin::pin!(crate::utils::sse::events(response, signal));
        while let Some(event) = events.next().await {
            let event = event.map_err(|error| HarnessError::new(error.to_string()))?;
            if event.event.as_deref().is_some_and(|name| name != "message") {
                continue;
            }
            let Ok(message) = serde_json::from_str::<Value>(&event.data) else {
                continue;
            };
            if message.get("id") == Some(&id) && message.get("method").is_none() {
                return Ok(Some(message));
            }
        }
        Err(HarnessError::new("MCP server closed the stream without a response").into())
    }

    async fn request(&self, method: &str, params: Value, signal: Option<CancellationToken>) -> HarnessResult<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let response = self.send(request_body(id, method, params), signal).await?;
        response_result(response.ok_or_else(|| HarnessError::new("MCP server sent no response"))?)
    }
}

#[async_trait]
impl FxMcpClient for HttpMcpClient {
    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }

    async fn list_tools(&self, cursor: Option<String>) -> HarnessResult<Value> {
        let params = cursor.map_or_else(|| json!({}), |cursor| json!({ "cursor": cursor }));
        self.request("tools/list", params, None).await
    }

    async fn call_tool(&self, name: &str, arguments: Value, signal: CancellationToken) -> HarnessResult<Value> {
        let params = json!({ "name": name, "arguments": arguments });
        cancellable(Some(&signal), self.request("tools/call", params, Some(signal.clone()))).await
    }

    async fn close(&self) {
        let session_id = self.session_id.lock().unwrap().take();
        if let Some(session_id) = session_id {
            let request = apply_headers(self.http.delete(&self.config.url), &self.config.headers)
                .header("mcp-session-id", session_id)
                .header("mcp-protocol-version", PROTOCOL_VERSION);
            let _ = request.send().await;
        }
    }
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;

/// The legacy HTTP+SSE transport: responses arrive on the event stream; requests
/// go to the endpoint its first `endpoint` event names.
struct SseMcpClient {
    http: reqwest::Client,
    headers: Option<IndexMap<String, String>>,
    endpoint: String,
    pending: Pending,
    next_id: AtomicU64,
    instructions: Option<String>,
    stop: CancellationToken,
}

impl SseMcpClient {
    async fn connect(config: FxMcpTransportConfig, signal: Option<CancellationToken>) -> HarnessResult<Self> {
        let http = reqwest::Client::new();
        let base = reqwest::Url::parse(&config.url).map_err(|error| HarnessError::new(format!("Invalid MCP URL: {error}")))?;
        let response = cancellable(signal.as_ref(), async {
            Ok(apply_headers(http.get(base.clone()), &config.headers)
                .header("accept", "text/event-stream")
                .send()
                .await?)
        })
        .await?;
        if !response.status().is_success() {
            return Err(HarnessError::new(format!("MCP SSE Transport Error: {}", response.status())).into());
        }
        let stop = CancellationToken::new();
        let pending: Pending = Arc::default();
        let (endpoint_sender, endpoint) = oneshot::channel::<String>();
        let reader_pending = pending.clone();
        let mut events = Box::pin(crate::utils::sse::events(response, Some(stop.clone())));
        tokio::spawn(async move {
            let mut endpoint_sender = Some(endpoint_sender);
            while let Some(Ok(event)) = events.next().await {
                match event.event.as_deref() {
                    Some("endpoint") => {
                        if let Some(sender) = endpoint_sender.take() {
                            let _ = sender.send(event.data);
                        }
                    }
                    None | Some("message") => {
                        let Ok(message) = serde_json::from_str::<Value>(&event.data) else {
                            continue;
                        };
                        if message.get("method").is_some() {
                            continue;
                        }
                        let Some(id) = message.get("id").and_then(Value::as_u64) else {
                            continue;
                        };
                        if let Some(sender) = reader_pending.lock().unwrap().remove(&id) {
                            let _ = sender.send(message);
                        }
                    }
                    Some(_) => {}
                }
            }
            reader_pending.lock().unwrap().clear();
        });
        let endpoint = cancellable(signal.as_ref(), async {
            endpoint
                .await
                .map_err(|_| HarnessError::new("MCP SSE Transport Error: the stream ended before an endpoint event").into())
        })
        .await?;
        let endpoint = base
            .join(endpoint.trim())
            .map_err(|error| HarnessError::new(format!("Invalid MCP endpoint: {error}")))?;
        if endpoint.origin() != base.origin() {
            stop.cancel();
            return Err(HarnessError::new("MCP SSE endpoint origin does not match the connection origin").into());
        }
        let mut client = Self {
            http,
            headers: config.headers,
            endpoint: endpoint.to_string(),
            pending,
            next_id: AtomicU64::new(0),
            instructions: None,
            stop,
        };
        let result = cancellable(signal.as_ref(), client.request("initialize", initialize_params())).await;
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                client.stop.cancel();
                return Err(error);
            }
        };
        client.instructions = result.get("instructions").and_then(Value::as_str).map(str::to_string);
        client
            .post(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))
            .await?;
        Ok(client)
    }

    async fn post(&self, body: Value) -> HarnessResult<()> {
        let response = apply_headers(self.http.post(&self.endpoint), &self.headers)
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(HarnessError::new(format!("MCP SSE Transport Error: {status} {text}")).into());
        }
        Ok(())
    }

    async fn request(&self, method: &str, params: Value) -> HarnessResult<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, sender);
        if let Err(error) = self.post(request_body(id, method, params)).await {
            self.pending.lock().unwrap().remove(&id);
            return Err(error);
        }
        let message = receiver
            .await
            .map_err(|_| HarnessError::new("MCP connection closed"))?;
        response_result(message)
    }
}

#[async_trait]
impl FxMcpClient for SseMcpClient {
    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }

    async fn list_tools(&self, cursor: Option<String>) -> HarnessResult<Value> {
        let params = cursor.map_or_else(|| json!({}), |cursor| json!({ "cursor": cursor }));
        self.request("tools/list", params).await
    }

    async fn call_tool(&self, name: &str, arguments: Value, signal: CancellationToken) -> HarnessResult<Value> {
        let params = json!({ "name": name, "arguments": arguments });
        cancellable(Some(&signal), self.request("tools/call", params)).await
    }

    async fn close(&self) {
        self.stop.cancel();
    }
}

impl Drop for SseMcpClient {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}
