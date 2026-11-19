//! [rust] The `opencode serve` HTTP API, in place of `@opencode-ai/sdk`'s
//! `createOpencodeClient`: JSON requests with the server's Basic auth and the
//! session directory header, and the `/event` server-sent event stream.

use futures::StreamExt as _;
use indexmap::IndexMap;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::harness::error::{HarnessError, HarnessResult};

/// A client for one OpenCode server and working directory.
#[derive(Clone)]
pub(super) struct OpenCodeClient {
    http: reqwest::Client,
    base_url: String,
    directory: String,
    headers: IndexMap<String, String>,
}

/// The `{ data, error }` of an SDK call.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Response {
    Data(Value),
    Error(Value),
}

fn is_loopback(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|url| {
        matches!(
            url.host_str(),
            Some("127.0.0.1" | "localhost" | "[::1]" | "::1")
        )
    })
}

/// `encodeURIComponent`.
fn encode_uri_component(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'!'
            | b'~'
            | b'*'
            | b'\''
            | b'('
            | b')' => out.push(byte as char),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

impl OpenCodeClient {
    /// `base_url` is the server URL; `headers` are the port endpoint's
    /// headers, sent with every request.
    pub fn new(
        base_url: &str,
        username: &str,
        password: &str,
        directory: &str,
        endpoint_headers: Option<&IndexMap<String, String>>,
    ) -> HarnessResult<Self> {
        let mut builder = reqwest::Client::builder();
        if is_loopback(base_url) {
            builder = builder.no_proxy();
        }
        let http = builder.build()?;
        let mut headers = endpoint_headers.cloned().unwrap_or_default();
        use base64::Engine as _;
        headers.insert(
            "Authorization".into(),
            format!(
                "Basic {}",
                base64::engine::general_purpose::STANDARD.encode(format!("{username}:{password}"))
            ),
        );
        headers.insert(
            "x-opencode-directory".into(),
            encode_uri_component(directory),
        );
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            directory: directory.to_string(),
            headers,
        })
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let mut request = self
            .http
            .request(method, format!("{}{path}", self.base_url));
        for (name, value) in &self.headers {
            request = request.header(name, value);
        }
        request
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> HarnessResult<Response> {
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;
        let body = if text.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&text).unwrap_or(Value::String(text))
        };
        Ok(if status.is_success() {
            Response::Data(body)
        } else if body.is_null() {
            Response::Error(json!({ "status": status.as_u16() }))
        } else {
            Response::Error(body)
        })
    }

    /// The SDK moves the directory header of a `GET` into the query.
    fn get_request(&self, path: &str) -> reqwest::RequestBuilder {
        let directory = encode_uri_component(&self.directory);
        let mut query = format!("directory={directory}");
        if path.starts_with("/api/") {
            query.push_str(&format!("&location%5Bdirectory%5D={directory}"));
        }
        let separator = if path.contains('?') { '&' } else { '?' };
        let mut request = self
            .http
            .get(format!("{}{path}{separator}{query}", self.base_url));
        for (name, value) in &self.headers {
            if name != "x-opencode-directory" {
                request = request.header(name, value);
            }
        }
        request
    }

    pub async fn get(&self, path: &str) -> HarnessResult<Response> {
        self.send(self.get_request(path)).await
    }

    pub async fn post(&self, path: &str, body: Value) -> HarnessResult<Response> {
        self.send(self.request(reqwest::Method::POST, path).json(&body))
            .await
    }

    /// `GET /event`: each event's JSON on `events` until the stream ends or
    /// `cancel` fires. Returns once the stream is open.
    pub async fn subscribe(
        &self,
        cancel: CancellationToken,
    ) -> HarnessResult<mpsc::UnboundedReceiver<Value>> {
        let response = self
            .get_request("/event")
            .header("Accept", "text/event-stream")
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(HarnessError::new(format!(
                "OpenCode event subscription failed with status {}.",
                response.status().as_u16()
            ))
            .into());
        }
        let (sender, receiver) = mpsc::unbounded_channel();
        let mut stream = response.bytes_stream();
        tokio::spawn(async move {
            let mut buffer: Vec<u8> = Vec::new();
            let mut data = String::new();
            loop {
                let chunk = tokio::select! {
                    _ = cancel.cancelled() => break,
                    chunk = stream.next() => chunk,
                };
                let Some(Ok(chunk)) = chunk else {
                    break;
                };
                buffer.extend_from_slice(&chunk);
                while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
                    let line: Vec<u8> = buffer.drain(..=position).collect();
                    let line = String::from_utf8_lossy(&line);
                    let line = line.trim_end_matches(['\n', '\r']);
                    if line.is_empty() {
                        if !data.is_empty() {
                            if let Ok(value) = serde_json::from_str::<Value>(&data) {
                                let _ = sender.send(value);
                            }
                            data.clear();
                        }
                    } else if let Some(rest) = line.strip_prefix("data:") {
                        if !data.is_empty() {
                            data.push('\n');
                        }
                        data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
                    }
                }
            }
        });
        Ok(receiver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_directories_like_encode_uri_component() {
        assert_eq!(
            encode_uri_component("/work dir/ä"),
            "%2Fwork%20dir%2F%C3%A4"
        );
        assert_eq!(encode_uri_component("a-b_c.d!~*'()"), "a-b_c.d!~*'()");
    }
}
