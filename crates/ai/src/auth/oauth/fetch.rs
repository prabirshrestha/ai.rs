//! Minimal `fetch()` seam for the OAuth flows (no Pi counterpart).
//!
//! Pi's flows call the global `fetch`, and its tests replace it with
//! `vi.stubGlobal("fetch", ...)`. The Rust flows take an [`OAuthFetch`]
//! instead: [`default_oauth_fetch`] sends through `reqwest`, and tests (or apps
//! that need a proxy-aware client) inject their own.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::types::BoxFuture;
use crate::utils::headers::redact_header;
use crate::{Error, Result};

/// The message of a DOM `TimeoutError` (`AbortSignal.timeout()`).
pub(crate) const TIMEOUT_MESSAGE: &str = "The operation was aborted due to timeout";

/// One HTTP request (`fetch(url, init)`). `Debug` redacts credential
/// headers and the body (token grants carry codes, verifiers and refresh
/// tokens).
#[derive(Clone, PartialEq, Eq)]
pub struct FetchRequest {
    pub method: String,
    pub url: String,
    /// Header names keep the casing the flow uses.
    pub headers: Vec<(String, String)>,
    pub body: Option<String>,
}

impl fmt::Debug for FetchRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FetchRequest")
            .field("method", &self.method)
            .field("url", &self.url)
            .field(
                "headers",
                &self
                    .headers
                    .iter()
                    .map(|(name, value)| redact_header(name, value))
                    .collect::<Vec<_>>(),
            )
            .field("body", &self.body.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl FetchRequest {
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: "GET".to_string(),
            url: url.into(),
            headers: Vec::new(),
            body: None,
        }
    }

    pub fn post(url: impl Into<String>) -> Self {
        Self {
            method: "POST".to_string(),
            ..Self::get(url)
        }
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Case-insensitive header lookup.
    pub fn header_value(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// A fully read HTTP response. `Debug` redacts credential headers and the
/// body (token responses carry access and refresh tokens).
#[derive(Clone, Default, PartialEq, Eq)]
pub struct FetchResponse {
    pub status: u16,
    pub status_text: String,
    /// Lower-cased header names.
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl fmt::Debug for FetchResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FetchResponse")
            .field("status", &self.status)
            .field("status_text", &self.status_text)
            .field(
                "headers",
                &self
                    .headers
                    .iter()
                    .map(|(name, value)| redact_header(name, value))
                    .collect::<Vec<_>>(),
            )
            .field("body", &"<redacted>")
            .finish()
    }
}

impl FetchResponse {
    /// A response with the canonical status text for `status`.
    pub fn new(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            status_text: reqwest::StatusCode::from_u16(status)
                .ok()
                .and_then(|status| status.canonical_reason())
                .unwrap_or_default()
                .to_string(),
            headers: HashMap::new(),
            body: body.into(),
        }
    }

    /// A JSON response (`new Response(JSON.stringify(body), { status })`).
    pub fn json(status: u16, body: &serde_json::Value) -> Self {
        Self::new(status, body.to_string()).with_header("content-type", "application/json")
    }

    pub fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.insert(name.to_ascii_lowercase(), value.into());
        self
    }

    /// `response.ok`.
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }
}

/// `fetch(request, { signal })`. Rejects when the signal aborts.
pub type OAuthFetch =
    Arc<dyn Fn(FetchRequest, CancellationToken) -> BoxFuture<Result<FetchResponse>> + Send + Sync>;

/// `fetch` through a default `reqwest` client.
pub fn default_oauth_fetch() -> OAuthFetch {
    reqwest_oauth_fetch(reqwest::Client::new())
}

/// `fetch` through the given `reqwest` client.
pub fn reqwest_oauth_fetch(client: reqwest::Client) -> OAuthFetch {
    Arc::new(move |request, signal| {
        let client = client.clone();
        Box::pin(async move {
            let send = async {
                let method = reqwest::Method::from_bytes(request.method.as_bytes())
                    .map_err(|_| Error::message(format!("Invalid method: {}", request.method)))?;
                let mut builder = client.request(method, &request.url);
                for (name, value) in &request.headers {
                    builder = builder.header(name, value);
                }
                if let Some(body) = request.body {
                    builder = builder.body(body);
                }
                let response = builder.send().await?;
                let status = response.status();
                let headers = response
                    .headers()
                    .iter()
                    .filter_map(|(name, value)| {
                        Some((name.as_str().to_string(), value.to_str().ok()?.to_string()))
                    })
                    .collect();
                let body = response.text().await?;
                Ok(FetchResponse {
                    status: status.as_u16(),
                    status_text: status.canonical_reason().unwrap_or_default().to_string(),
                    headers,
                    body,
                })
            };
            if signal.is_cancelled() {
                return Err(Error::aborted());
            }
            tokio::select! {
                _ = signal.cancelled() => Err(Error::aborted()),
                result = send => result,
            }
        })
    })
}

/// `fetch(request, { signal: AbortSignal.any([signal, AbortSignal.timeout(ms)]) })`.
pub(crate) async fn fetch_with_timeout(
    fetch: &OAuthFetch,
    request: FetchRequest,
    signal: &CancellationToken,
    timeout: Duration,
) -> Result<FetchResponse> {
    let request_signal = signal.child_token();
    let response = fetch(request, request_signal.clone());
    tokio::select! {
        result = response => result,
        _ = tokio::time::sleep(timeout) => {
            request_signal.cancel();
            Err(Error::Aborted(TIMEOUT_MESSAGE.to_string()))
        }
    }
}

/// `new URLSearchParams(fields).toString()`.
pub(crate) fn url_search_params(fields: &[(&str, &str)]) -> String {
    let mut url = reqwest::Url::parse("http://localhost/").expect("static url");
    url.query_pairs_mut().extend_pairs(fields);
    url.query().unwrap_or_default().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_redacts_credentials() {
        let request = FetchRequest::post("https://example.test/token")
            .header("Authorization", "Bearer secret-access")
            .header("Content-Type", "application/json")
            .body(r#"{"refresh_token":"secret-refresh"}"#);
        let debug = format!("{request:?}");
        assert!(!debug.contains("secret"), "{debug}");
        assert!(debug.contains("application/json"), "{debug}");

        let response = FetchResponse {
            status: 200,
            headers: [("set-cookie".to_string(), "secret-cookie".to_string())].into(),
            body: r#"{"access_token":"secret-access"}"#.to_string(),
            ..Default::default()
        };
        assert!(!format!("{response:?}").contains("secret"));

        let pkce = crate::auth::oauth::pkce::generate_pkce().unwrap();
        assert!(!format!("{pkce:?}").contains(&pkce.verifier));

        let mut env = crate::types::ProviderEnv::new();
        env.insert("API_TOKEN".to_string(), "secret-env".to_string());
        let result = crate::auth::AuthResult {
            env: Some(env),
            ..Default::default()
        };
        let debug = format!("{result:?}");
        assert!(
            !debug.contains("secret") && debug.contains("API_TOKEN"),
            "{debug}"
        );
    }

    #[test]
    fn encodes_search_params_like_url_search_params() {
        assert_eq!(
            url_search_params(&[
                ("client_id", "Iv1.b507a08c87ecfe98"),
                ("scope", "read:user"),
                ("device_code", "abc def/ghi*"),
            ]),
            "client_id=Iv1.b507a08c87ecfe98&scope=read%3Auser&device_code=abc+def%2Fghi*"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn times_out_pending_requests() {
        let fetch: OAuthFetch = Arc::new(|_, signal| {
            Box::pin(async move {
                signal.cancelled().await;
                Err(Error::aborted())
            })
        });
        let error = fetch_with_timeout(
            &fetch,
            FetchRequest::get("http://x"),
            &CancellationToken::new(),
            Duration::from_millis(10),
        )
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), TIMEOUT_MESSAGE);
    }

    #[test]
    fn response_helpers_report_status_and_headers() {
        let response =
            FetchResponse::json(429, &serde_json::json!({})).with_header("Retry-After", "1");
        assert_eq!(response.status_text, "Too Many Requests");
        assert!(!response.ok());
        assert_eq!(response.header("retry-after"), Some("1"));
        assert_eq!(
            FetchRequest::post("u")
                .header("Content-Type", "x")
                .header_value("content-type"),
            Some("x")
        );
    }
}
