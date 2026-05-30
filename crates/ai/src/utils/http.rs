use std::time::Duration;

use reqwest::{RequestBuilder, Response, StatusCode, header::HeaderMap};

use crate::types::StreamOptions;
use crate::{Error, Result};

const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 60_000;

pub async fn send_with_retries<F>(options: &StreamOptions, mut build: F) -> Result<Response>
where
    F: FnMut() -> RequestBuilder,
{
    let max_retries = options.max_retries.unwrap_or(0);
    let mut attempt = 0;

    loop {
        if options
            .cancellation_token
            .as_ref()
            .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
        {
            return Err(Error::Cancelled);
        }

        let send = build().send();
        let result = if let Some(cancellation_token) = options.cancellation_token.as_ref() {
            tokio::select! {
                _ = cancellation_token.cancelled() => Err(Error::Cancelled),
                response = send => response.map_err(Error::from),
            }
        } else {
            send.await.map_err(Error::from)
        };

        match result {
            Ok(response) if attempt < max_retries && is_retryable_status(response.status()) => {
                sleep_before_retry(options, response.headers(), attempt).await?;
                attempt += 1;
            }
            Ok(response) => return Ok(response),
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(error) if attempt < max_retries => {
                sleep_before_retry(options, &HeaderMap::new(), attempt).await?;
                attempt += 1;
                let _ = error;
            }
            Err(error) => return Err(error),
        }
    }
}

fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT | StatusCode::CONFLICT | StatusCode::TOO_MANY_REQUESTS
    ) || status.is_server_error()
}

async fn sleep_before_retry(
    options: &StreamOptions,
    headers: &HeaderMap,
    attempt: u32,
) -> Result<()> {
    let delay_ms = retry_delay_ms(headers, attempt, options.max_retry_delay_ms);
    if delay_ms == 0 {
        return Ok(());
    }

    if let Some(cancellation_token) = options.cancellation_token.as_ref() {
        tokio::select! {
            _ = cancellation_token.cancelled() => Err(Error::Cancelled),
            _ = tokio::time::sleep(Duration::from_millis(delay_ms)) => Ok(()),
        }
    } else {
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        Ok(())
    }
}

fn retry_delay_ms(headers: &HeaderMap, attempt: u32, max_retry_delay_ms: Option<u64>) -> u64 {
    let delay = retry_after_ms(headers).unwrap_or_else(|| exponential_delay_ms(attempt));
    match max_retry_delay_ms {
        Some(0) => delay,
        Some(max) => delay.min(max),
        None => delay.min(DEFAULT_MAX_RETRY_DELAY_MS),
    }
}

fn retry_after_ms(headers: &HeaderMap) -> Option<u64> {
    headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .or_else(|| {
            headers
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .map(|seconds| seconds.saturating_mul(1000))
        })
}

fn exponential_delay_ms(attempt: u32) -> u64 {
    500u64.saturating_mul(2u64.saturating_pow(attempt.min(6)))
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn retries_retryable_status_when_enabled() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_retry_server(Arc::clone(&attempts)).await;
        let client = reqwest::Client::new();
        let options = StreamOptions {
            max_retries: Some(1),
            max_retry_delay_ms: Some(0),
            ..Default::default()
        };

        let response = send_with_retries(&options, || client.get(&url))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn does_not_retry_by_default() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_retry_server(Arc::clone(&attempts)).await;
        let client = reqwest::Client::new();

        let response = send_with_retries(&StreamOptions::default(), || client.get(&url))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    async fn spawn_retry_server(attempts: Arc<AtomicUsize>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                let mut buffer = vec![0u8; 1024];
                let _ = socket.read(&mut buffer).await;
                let response = if attempt == 0 {
                    "HTTP/1.1 500 Internal Server Error\r\nretry-after-ms: 0\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                } else {
                    "HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                };
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }
}
