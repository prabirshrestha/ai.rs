//! Rust HTTP plumbing shared by the API modules (no Pi counterpart: Pi uses
//! the provider SDKs and `fetch`).

use std::time::Duration;

use reqwest::{RequestBuilder, Response};
use tokio_util::sync::CancellationToken;

use crate::types::ProviderRequestOptions;
use crate::utils::headers::headers_to_record;
use crate::utils::provider_retry::{
    ProviderHttpError, ProviderRetryOptions, retry_provider_request,
};
use crate::{Error, Result};

/// Default HTTP request timeout of the OpenAI and Anthropic SDK clients.
pub const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 600_000;

pub fn request_timeout(timeout_ms: Option<u64>) -> Duration {
    Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_REQUEST_TIMEOUT_MS))
}

/// The HTTP client to use: the caller's, or a default client.
pub fn http_client(client: Option<&reqwest::Client>) -> reqwest::Client {
    client.cloned().unwrap_or_default()
}

/// Send one request, racing it with `signal`. A non-2xx response becomes a
/// [`ProviderHttpError`] with its body read, like an SDK `APIError`.
pub async fn send_checked(
    request: RequestBuilder,
    signal: Option<&CancellationToken>,
) -> Result<Response> {
    let send = async {
        let response = request.send().await?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let headers = headers_to_record(response.headers());
        let body = response.text().await.ok();
        Err(ProviderHttpError::new(status.as_u16(), headers, body).into())
    };
    match signal {
        Some(signal) => {
            if signal.is_cancelled() {
                return Err(Error::Aborted("Request was aborted".to_string()));
            }
            tokio::select! {
                _ = signal.cancelled() => Err(Error::Aborted("Request was aborted".to_string())),
                result = send => result,
            }
        }
        None => send.await,
    }
}

/// [`send_checked`] wrapped in [`retry_provider_request`] with the request
/// options' retry settings.
pub async fn send_with_retries<F>(
    options: &ProviderRequestOptions,
    mut build: F,
) -> Result<Response>
where
    F: FnMut() -> RequestBuilder,
{
    let retry_options = ProviderRetryOptions {
        max_retries: options.max_retries,
        max_retry_delay_ms: options.max_retry_delay_ms,
        signal: options.signal.clone(),
    };
    let signal = options.signal.clone();
    retry_provider_request(
        || {
            let request = build();
            let signal = signal.clone();
            async move { send_checked(request, signal.as_ref()).await }
        },
        &retry_options,
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    async fn spawn_server(
        attempts: Arc<AtomicUsize>,
        first_status: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let first_status = first_status.to_string();
        let headers = headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\r\n"))
            .collect::<String>();
        let body = body.to_string();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                let mut buffer = vec![0u8; 1024];
                let _ = socket.read(&mut buffer).await;
                let response = if attempt == 0 {
                    format!(
                        "HTTP/1.1 {first_status}\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                } else {
                    "HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_string()
                };
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn retries_retryable_status_when_enabled() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_server(
            Arc::clone(&attempts),
            "500 Internal Server Error",
            &[("retry-after-ms", "0")],
            "",
        )
        .await;
        let client = reqwest::Client::new();
        let options = ProviderRequestOptions {
            max_retries: Some(1),
            ..Default::default()
        };
        let response = send_with_retries(&options, || client.get(&url))
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn does_not_retry_by_default_and_reports_the_body() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_server(
            Arc::clone(&attempts),
            "500 Internal Server Error",
            &[],
            "{\"error\":\"boom\"}",
        )
        .await;
        let client = reqwest::Client::new();
        let error = send_with_retries(&ProviderRequestOptions::default(), || client.get(&url))
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "500 {\"error\":\"boom\"}");
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn rejects_server_retry_delay_above_cap_with_provider_message() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_server(
            Arc::clone(&attempts),
            "429 Too Many Requests",
            &[("retry-after", "277403")],
            "",
        )
        .await;
        let client = reqwest::Client::new();
        let options = ProviderRequestOptions {
            max_retries: Some(2),
            max_retry_delay_ms: Some(1_000),
            ..Default::default()
        };
        let error = send_with_retries(&options, || client.get(&url))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("Server requested 277403s retry delay (max: 1s)"));
        assert!(error.contains("429 status code (no body)"));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}
