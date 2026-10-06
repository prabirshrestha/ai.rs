//! Rust stand-in for the `openai` SDK client that Pi's OpenAI API modules
//! (`openai-responses.ts`, `openai-completions.ts`) create per request. No Pi
//! counterpart: it reproduces the SDK behaviour those modules rely on.
//!
//! - Headers: `Authorization: Bearer <apiKey>` first, then the client's
//!   default headers in order, where a `None` value removes a header (the SDK
//!   drops `null` header values), so `Authorization: null` suppresses auth.
//! - URL: `baseURL + path`, joining a trailing slash like the SDK.
//! - One attempt per call (Pi passes `maxRetries: 0`); `retryProviderRequest`
//!   drives retries.
//! - The SDK `timeout` bounds the request until the response headers arrive;
//!   expiry is the SDK's `APIConnectionTimeoutError` (`"Request timed out."`,
//!   no status, so `retryProviderRequest` retries it).
//! - SSE: `data: [DONE]` ends the stream, each other event is parsed as JSON,
//!   and an event whose JSON has a truthy `error` field fails the stream like
//!   the SDK's `APIError` (message from `error.message`), as a status-less
//!   `ProviderHttpError`.

use futures::{Stream, StreamExt};
use reqwest::Response;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::types::ProviderHeaders;
use crate::utils::headers::apply_provider_headers;
use crate::utils::http::{http_client, send_checked_with_timeout};
use crate::utils::provider_retry::ProviderHttpError;
use crate::utils::sse;
use crate::{Error, Result};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Per-request SDK options (`{ signal, timeout, maxRetries: 0 }`).
#[derive(Debug, Clone, Default)]
pub struct OpenAIRequestOptions {
    pub signal: Option<CancellationToken>,
    pub timeout_ms: Option<u64>,
}

/// `new OpenAI({ apiKey, baseURL, fetch, defaultHeaders })`.
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    pub api_key: String,
    pub base_url: String,
    pub default_headers: ProviderHeaders,
    http_client: reqwest::Client,
}

impl OpenAIClient {
    pub fn new(
        api_key: impl Into<String>,
        base_url: &str,
        http: Option<&reqwest::Client>,
        default_headers: ProviderHeaders,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: if base_url.is_empty() {
                DEFAULT_BASE_URL.to_string()
            } else {
                base_url.to_string()
            },
            default_headers,
            http_client: http_client(http),
        }
    }

    pub fn build_url(&self, path: &str) -> String {
        if self.base_url.ends_with('/') && path.starts_with('/') {
            format!("{}{}", self.base_url, &path[1..])
        } else {
            format!("{}{}", self.base_url, path)
        }
    }

    pub fn build_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|error| Error::InvalidHeaderValue("authorization".to_string(), error))?,
        );
        apply_provider_headers(&mut headers, &self.default_headers)?;
        Ok(headers)
    }

    /// POST `body` to `path` once and return the streaming response. A
    /// non-2xx response becomes a `ProviderHttpError`.
    pub async fn post(
        &self,
        path: &str,
        body: &Value,
        options: &OpenAIRequestOptions,
    ) -> Result<Response> {
        let request = self
            .http_client
            .post(self.build_url(path))
            .headers(self.build_headers()?)
            .json(body);
        send_checked_with_timeout(request, options.signal.as_ref(), options.timeout_ms).await
    }
}

/// The SDK's `Stream.fromSSEResponse()`: parsed JSON events until `[DONE]`.
pub fn sse_json_events(
    response: Response,
    signal: Option<CancellationToken>,
) -> impl Stream<Item = Result<Value>> + Send + 'static {
    let mut done = false;
    sse::events(response, signal).filter_map(move |event| {
        let item = match event {
            Err(error) => Some(Err(error)),
            Ok(_) if done => None,
            Ok(event) if event.data.starts_with("[DONE]") => {
                done = true;
                None
            }
            Ok(event) => Some(parse_sse_data(&event.data)),
        };
        futures::future::ready(item)
    })
}

fn parse_sse_data(data: &str) -> Result<Value> {
    let value: Value = serde_json::from_str(data)?;
    if let Some(error) = value.get("error").filter(|error| is_truthy(error)) {
        // Keep the error object as the body (wrapped like an HTTP error body,
        // `{ "error": ... }`) so callers can read fields such as
        // `error.metadata.raw`, like the SDK's `APIError.error`.
        return Err(Error::ProviderHttp(Box::new(ProviderHttpError {
            status: None,
            headers: Default::default(),
            body: Some(serde_json::json!({ "error": error }).to_string()),
            message: api_error_message(error),
        })));
    }
    Ok(value)
}

/// `APIError.makeMessage(undefined, error, undefined)`.
fn api_error_message(error: &Value) -> String {
    match error.get("message") {
        Some(Value::String(message)) if !message.is_empty() => message.clone(),
        Some(message) if is_truthy(message) => message.to_string(),
        _ => error.to_string(),
    }
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// JavaScript template-literal stringification of an optional JSON value
/// (`${value}`): strings verbatim, `undefined` for a missing value.
pub(crate) fn js_template_string(value: Option<&Value>) -> String {
    match value {
        None => "undefined".to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) => "null".to_string(),
        Some(Value::Object(_)) => "[object Object]".to_string(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::Null => String::new(),
                item => js_template_string(Some(item)),
            })
            .collect::<Vec<_>>()
            .join(","),
        Some(other) => other.to_string(),
    }
}

/// JavaScript truthiness of an optional JSON value.
pub(crate) fn js_truthy(value: Option<&Value>) -> bool {
    value.is_some_and(is_truthy)
}

#[cfg(test)]
pub(crate) mod test_support {
    //! A tiny HTTP/1.1 server for the OpenAI API module tests: it records
    //! every request and answers with queued responses.

    use std::collections::VecDeque;
    use std::sync::Arc;

    use futures::StreamExt;
    use parking_lot::Mutex;
    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use crate::Error;
    use crate::types::{
        AssistantMessage, AssistantMessageEvent, Context, Model, PayloadHook,
        ProviderStreamEventHook, TranscriptContext,
    };
    use crate::utils::event_stream::AssistantMessageEventStream;
    use crate::utils::transcript::normalize_context;

    /// `normalizeContext(<Pi Context literal>)`.
    pub fn context(value: Value) -> TranscriptContext {
        let context: Context = serde_json::from_value(value).expect("valid Pi context");
        normalize_context(&context)
    }

    /// A model from a Pi `Model` literal.
    pub fn model(value: Value) -> Model {
        serde_json::from_value(value).expect("valid Pi model")
    }

    /// The literal `gpt-5-mini` model most Pi OpenAI tests construct.
    pub fn gpt5_mini(api: &str) -> Model {
        model(json!({
            "id": "gpt-5-mini",
            "name": "GPT-5 Mini",
            "api": api,
            "provider": "openai",
            "baseUrl": "https://api.openai.com/v1",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 400000,
            "maxTokens": 128000,
        }))
    }

    pub fn openai_model(id: &str) -> Model {
        crate::providers::catalog::openai_models()
            .get(id)
            .cloned()
            .unwrap_or_else(|| panic!("openai catalog has {id}"))
    }

    pub fn copilot_model(id: &str) -> Model {
        crate::providers::catalog::github_copilot_models()
            .get(id)
            .cloned()
            .unwrap_or_else(|| panic!("github-copilot catalog has {id}"))
    }

    /// An empty `output` like the Pi tests' `createOutput()`.
    pub fn pending_output(model: &Model) -> AssistantMessage {
        let mut output = AssistantMessage::empty_for(model);
        output.stop_reason = crate::types::StopReason::Pending;
        output
    }

    /// An `onPayload` hook recording every payload; `abort` makes it fail the
    /// request so nothing is sent.
    pub fn payload_hook(abort: bool) -> (PayloadHook, Arc<Mutex<Vec<Value>>>) {
        let payloads = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&payloads);
        let hook: PayloadHook = Arc::new(move |payload, _model| {
            captured.lock().push(payload);
            Box::pin(async move {
                if abort {
                    Err(Error::message("payload captured"))
                } else {
                    Ok(None)
                }
            })
        });
        (hook, payloads)
    }

    /// Provider stream events with the id of the model that produced them.
    pub type CapturedStreamEvents = Arc<Mutex<Vec<(Value, String)>>>;

    /// An `onProviderStreamEvent` hook recording every event.
    pub fn stream_event_hook() -> (ProviderStreamEventHook, CapturedStreamEvents) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let hook: ProviderStreamEventHook = Arc::new(move |event, model| {
            captured.lock().push((event.clone(), model.id.clone()));
            Box::pin(async {})
        });
        (hook, events)
    }

    /// Drain a stream: its events and final message.
    pub async fn collect(
        stream: AssistantMessageEventStream,
    ) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
        let events: Vec<_> = stream.clone().collect().await;
        (events, stream.result().await)
    }

    /// Run `stream` with a payload hook that aborts the request, and return
    /// the captured payload.
    pub async fn capture_payload(
        run: impl FnOnce(PayloadHook) -> AssistantMessageEventStream,
    ) -> Value {
        let (hook, payloads) = payload_hook(true);
        let (_, result) = collect(run(hook)).await;
        let payload = payloads.lock().first().cloned();
        payload.unwrap_or_else(|| panic!("no payload captured: {:?}", result.error_message))
    }

    #[derive(Debug, Clone)]
    pub struct CapturedRequest {
        pub path: String,
        pub headers: Vec<(String, String)>,
        pub body: Value,
    }

    impl CapturedRequest {
        pub fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.as_str())
        }
    }

    #[derive(Debug, Clone)]
    pub struct MockResponse {
        pub status: u16,
        pub headers: Vec<(String, String)>,
        pub body: String,
    }

    impl MockResponse {
        pub fn sse(events: &[Value]) -> Self {
            let mut body: String = events
                .iter()
                .map(|event| format!("data: {event}\n\n"))
                .collect();
            body.push_str("data: [DONE]\n\n");
            Self::sse_raw(body)
        }

        pub fn sse_raw(body: impl Into<String>) -> Self {
            Self {
                status: 200,
                headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
                body: body.into(),
            }
        }

        pub fn status(status: u16, headers: &[(&str, &str)], body: impl Into<String>) -> Self {
            Self {
                status,
                headers: headers
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
                body: body.into(),
            }
        }
    }

    pub struct MockServer {
        pub url: String,
        pub requests: Arc<Mutex<Vec<CapturedRequest>>>,
    }

    impl MockServer {
        /// Serve `responses` in order; once they run out, the last one repeats.
        pub async fn start(responses: Vec<MockResponse>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let captured = Arc::clone(&requests);
            let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
            tokio::spawn(async move {
                let mut last: Option<MockResponse> = None;
                loop {
                    let Ok((mut socket, _)) = listener.accept().await else {
                        break;
                    };
                    let Some(request) = read_request(&mut socket).await else {
                        continue;
                    };
                    captured.lock().push(request);
                    let response = {
                        let mut queue = queue.lock();
                        match queue.pop_front() {
                            Some(response) => {
                                last = Some(response.clone());
                                response
                            }
                            None => last.clone().unwrap_or_else(|| MockResponse::sse(&[])),
                        }
                    };
                    let reason = match response.status {
                        200 => "OK",
                        _ => "Error",
                    };
                    let mut raw = format!("HTTP/1.1 {} {reason}\r\n", response.status);
                    for (name, value) in &response.headers {
                        raw.push_str(&format!("{name}: {value}\r\n"));
                    }
                    raw.push_str(&format!(
                        "content-length: {}\r\nconnection: close\r\n\r\n{}",
                        response.body.len(),
                        response.body
                    ));
                    let _ = socket.write_all(raw.as_bytes()).await;
                    let _ = socket.shutdown().await;
                }
            });
            Self {
                url: format!("http://{addr}/v1"),
                requests,
            }
        }

        pub fn requests(&self) -> Vec<CapturedRequest> {
            self.requests.lock().clone()
        }

        pub fn last(&self) -> CapturedRequest {
            self.requests
                .lock()
                .last()
                .cloned()
                .expect("a request was captured")
        }
    }

    async fn read_request(socket: &mut tokio::net::TcpStream) -> Option<CapturedRequest> {
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 8192];
        let header_end = loop {
            let read = socket.read(&mut chunk).await.ok()?;
            if read == 0 {
                return None;
            }
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };
        let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
        let mut lines = head.split("\r\n");
        let request_line = lines.next()?;
        let path = request_line.split(' ').nth(1)?.to_string();
        let headers: Vec<(String, String)> = lines
            .filter(|line| !line.is_empty())
            .filter_map(|line| {
                let (name, value) = line.split_once(':')?;
                Some((name.trim().to_string(), value.trim().to_string()))
            })
            .collect();
        let length = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, value)| value.parse::<usize>().ok())
            .unwrap_or(0);
        while buffer.len() < header_end + length {
            let read = socket.read(&mut chunk).await.ok()?;
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
        }
        let body = serde_json::from_slice(&buffer[header_end..]).unwrap_or(Value::Null);
        Some(CapturedRequest {
            path,
            headers,
            body,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::ProviderHeaders;

    #[test]
    fn builds_urls_and_headers_like_the_sdk() {
        let mut defaults = ProviderHeaders::new();
        defaults.insert("User-Agent", Some("pi".to_string()));
        defaults.insert("Authorization", None::<String>);
        let client = OpenAIClient::new("key", "http://host/v1/", None, defaults);
        assert_eq!(client.build_url("/responses"), "http://host/v1/responses");
        let headers = client.build_headers().unwrap();
        assert!(headers.get(AUTHORIZATION).is_none());
        assert_eq!(headers.get("user-agent").unwrap(), "pi");

        let client = OpenAIClient::new("key", "", None, ProviderHeaders::new());
        assert_eq!(
            client.build_url("/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            client.build_headers().unwrap().get(AUTHORIZATION).unwrap(),
            "Bearer key"
        );
    }

    #[test]
    fn sse_error_payloads_fail_like_api_errors() {
        assert_eq!(
            parse_sse_data(r#"{"error":{"message":"boom"}}"#)
                .unwrap_err()
                .to_string(),
            "boom"
        );
        assert_eq!(
            parse_sse_data(r#"{"type":"error","error":null}"#).unwrap(),
            json!({ "type": "error", "error": null })
        );
    }

    #[test]
    fn template_strings_follow_javascript() {
        assert_eq!(js_template_string(None), "undefined");
        assert_eq!(js_template_string(Some(&json!("x"))), "x");
        assert_eq!(js_template_string(Some(&json!(42))), "42");
    }
}
