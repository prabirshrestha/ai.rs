//! [rust] `typeof globalThis.fetch`, reduced to what the fx core and its
//! transport use: a request with a byte body, and a response whose body
//! streams.

use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream};
use futures::{StreamExt as _, TryStreamExt as _};
use tokio_util::sync::CancellationToken;

/// One HTTP request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FxHttpRequest {
    pub url: String,
    pub method: String,
    /// In order; a name may repeat.
    pub headers: Vec<(String, String)>,
    pub body: Option<Bytes>,
}

impl FxHttpRequest {
    /// The first value of a header, by case-insensitive name.
    pub fn header(&self, name: &str) -> Option<&str> {
        header(&self.headers, name)
    }

    /// Set a header, replacing every value it had (`Headers.set`).
    pub fn set_header(&mut self, name: &str, value: impl Into<String>) {
        set_header(&mut self.headers, name, value.into());
    }

    pub fn body_text(&self) -> String {
        self.body
            .as_ref()
            .map(|body| String::from_utf8_lossy(body).into_owned())
            .unwrap_or_default()
    }
}

pub(crate) fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

pub(crate) fn set_header(headers: &mut Vec<(String, String)>, name: &str, value: String) {
    headers.retain(|(key, _)| !key.eq_ignore_ascii_case(name));
    headers.push((name.to_ascii_lowercase(), value));
}

/// The error a body stream or a fetch fails with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FxFetchError {
    /// The request was cancelled (`AbortError`).
    Aborted,
    Failed(String),
}

impl fmt::Display for FxFetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aborted => f.write_str("This operation was aborted"),
            Self::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for FxFetchError {}

/// A response body.
pub type FxBody = BoxStream<'static, Result<Bytes, FxFetchError>>;

/// One HTTP response.
pub struct FxHttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: FxBody,
}

impl fmt::Debug for FxHttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FxHttpResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

impl FxHttpResponse {
    /// A response with a whole body.
    pub fn new(status: u16, body: impl Into<Bytes>) -> Self {
        let body: Bytes = body.into();
        Self {
            status,
            headers: Vec::new(),
            body: stream::once(async move { Ok(body) }).boxed(),
        }
    }

    pub fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        set_header(&mut self.headers, name, value.into());
        self
    }

    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        header(&self.headers, name)
    }

    /// The whole body.
    pub async fn bytes(self) -> Result<Bytes, FxFetchError> {
        let chunks: Vec<Bytes> = self.body.try_collect().await?;
        Ok(chunks.concat().into())
    }

    pub async fn text(self) -> Result<String, FxFetchError> {
        Ok(String::from_utf8_lossy(&self.bytes().await?).into_owned())
    }
}

/// `fetch(input, { signal })`. The token cancels the request and its body.
pub type FxFetch = Arc<
    dyn Fn(
            FxHttpRequest,
            CancellationToken,
        ) -> BoxFuture<'static, Result<FxHttpResponse, FxFetchError>>
        + Send
        + Sync,
>;

/// The host's `fetch`: a `reqwest` client.
pub fn reqwest_fetch() -> FxFetch {
    let client = reqwest::Client::new();
    Arc::new(move |request: FxHttpRequest, signal: CancellationToken| {
        let client = client.clone();
        Box::pin(async move {
            let method = reqwest::Method::from_bytes(request.method.as_bytes())
                .map_err(|error| FxFetchError::Failed(error.to_string()))?;
            let mut builder = client.request(method, &request.url);
            for (name, value) in &request.headers {
                builder = builder.header(name, value);
            }
            if let Some(body) = request.body {
                builder = builder.body(body);
            }
            let response = tokio::select! {
                response = builder.send() => response.map_err(|error| FxFetchError::Failed(error.to_string()))?,
                _ = signal.cancelled() => return Err(FxFetchError::Aborted),
            };
            let status = response.status().as_u16();
            let headers = response
                .headers()
                .iter()
                .map(|(name, value)| {
                    (
                        name.as_str().to_string(),
                        String::from_utf8_lossy(value.as_bytes()).into_owned(),
                    )
                })
                .collect();
            let body = response
                .bytes_stream()
                .map_err(|error| FxFetchError::Failed(error.to_string()));
            let body = stream::unfold(
                (body.boxed(), signal),
                |(mut body, signal)| async move {
                    tokio::select! {
                        chunk = body.next() => chunk.map(|chunk| (chunk, (body, signal))),
                        _ = signal.cancelled() => Some((Err(FxFetchError::Aborted), (futures::stream::empty().boxed(), CancellationToken::new()))),
                    }
                },
            )
            .boxed();
            Ok(FxHttpResponse {
                status,
                headers,
                body,
            })
        })
    })
}
