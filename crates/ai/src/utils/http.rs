use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::{RequestBuilder, Response, StatusCode, header::HeaderMap};
use ring::rand::{SecureRandom, SystemRandom};

use crate::types::StreamOptions;
use crate::{Error, Result};

pub const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 600_000;
const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 60_000;

pub fn request_timeout(timeout_ms: Option<u64>) -> Duration {
    Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_REQUEST_TIMEOUT_MS))
}

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
            Ok(response) if attempt < max_retries && is_retryable_response(&response) => {
                let delay_ms = match retry_delay_ms(
                    response.headers(),
                    attempt,
                    options.max_retry_delay_ms,
                ) {
                    Ok(delay_ms) => delay_ms,
                    Err(Error::Provider(message)) => {
                        let status = response.status();
                        let read_body = response.text();
                        let body = if let Some(cancellation_token) =
                            options.cancellation_token.as_ref()
                        {
                            tokio::select! {
                                _ = cancellation_token.cancelled() => return Err(Error::Cancelled),
                                body = read_body => body.unwrap_or_default(),
                            }
                        } else {
                            read_body.await.unwrap_or_default()
                        };
                        return Err(Error::Provider(format!(
                            "{message} {}",
                            Error::ApiStatus { status, body }
                        )));
                    }
                    Err(error) => return Err(error),
                };
                sleep_before_retry(options, delay_ms).await?;
                attempt += 1;
            }
            Ok(response) => return Ok(response),
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(error) if attempt < max_retries => {
                let delay_ms =
                    retry_delay_ms(&HeaderMap::new(), attempt, options.max_retry_delay_ms)?;
                sleep_before_retry(options, delay_ms).await?;
                attempt += 1;
                let _ = error;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Mirrors the pinned OpenAI/Anthropic SDK retry policy. Provider directives
/// take precedence over the status-based fallback.
fn is_retryable_response(response: &Response) -> bool {
    if response.status().is_success() {
        return false;
    }
    match response
        .headers()
        .get("x-should-retry")
        .and_then(|value| value.to_str().ok())
    {
        Some("true") => true,
        Some("false") => false,
        _ => is_retryable_status(response.status()),
    }
}

fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT | StatusCode::CONFLICT | StatusCode::TOO_MANY_REQUESTS
    ) || status.is_server_error()
}

async fn sleep_before_retry(options: &StreamOptions, delay_ms: f64) -> Result<()> {
    if delay_ms.is_nan() || delay_ms <= 0.0 {
        return Ok(());
    }
    let delay = if delay_ms.is_finite() {
        Duration::from_secs_f64(delay_ms / 1000.0)
    } else {
        Duration::MAX
    };

    if let Some(cancellation_token) = options.cancellation_token.as_ref() {
        tokio::select! {
            _ = cancellation_token.cancelled() => Err(Error::Cancelled),
            _ = tokio::time::sleep(delay) => Ok(()),
        }
    } else {
        tokio::time::sleep(delay).await;
        Ok(())
    }
}

fn retry_delay_ms(
    headers: &HeaderMap,
    attempt: u32,
    max_retry_delay_ms: Option<u64>,
) -> Result<f64> {
    let Some(delay_ms) = retry_after_ms(headers) else {
        return Ok(exponential_delay_ms(attempt, random_unit_interval()));
    };
    let max_delay_ms = max_retry_delay_ms.unwrap_or(DEFAULT_MAX_RETRY_DELAY_MS);
    if max_delay_ms > 0 && delay_ms > max_delay_ms as f64 {
        let requested_seconds = (delay_ms / 1000.0).ceil();
        let max_seconds = max_delay_ms.saturating_add(999) / 1000;
        return Err(Error::Provider(format!(
            "Server requested {requested_seconds}s retry delay (max: {max_seconds}s)."
        )));
    }
    Ok(delay_ms)
}

fn retry_after_ms(headers: &HeaderMap) -> Option<f64> {
    headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(parse_retry_delay_number)
        .or_else(|| {
            let value = headers
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty())?;
            Some(
                parse_retry_delay_seconds(value)
                    .or_else(|| retry_after_http_date_ms(value.trim()))
                    .unwrap_or(f64::NAN),
            )
        })
}

fn parse_retry_delay_number(value: &str) -> Option<f64> {
    parse_float_prefix(value)
}

fn parse_retry_delay_seconds(value: &str) -> Option<f64> {
    parse_float_prefix(value).map(|seconds| seconds * 1000.0)
}

/// Equivalent to JavaScript's `Number.parseFloat` for the decimal forms that
/// can occur in HTTP headers. In particular, a valid numeric prefix is accepted.
fn parse_float_prefix(value: &str) -> Option<f64> {
    let value = value.trim_start();
    let (sign, unsigned) = match value.as_bytes().first() {
        Some(b'+') => (1.0, &value[1..]),
        Some(b'-') => (-1.0, &value[1..]),
        _ => (1.0, value),
    };
    if unsigned.starts_with("Infinity") {
        return Some(sign * f64::INFINITY);
    }
    let bytes = value.as_bytes();
    let mut end = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let integer_start = end;
    while bytes.get(end).is_some_and(u8::is_ascii_digit) {
        end += 1;
    }
    let mut has_digit = end > integer_start;
    if bytes.get(end) == Some(&b'.') {
        end += 1;
        let fraction_start = end;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        has_digit |= end > fraction_start;
    }
    if !has_digit {
        return None;
    }

    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        let exponent_mark = end;
        end += 1;
        if matches!(bytes.get(end), Some(b'+' | b'-')) {
            end += 1;
        }
        let exponent_start = end;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        if end == exponent_start {
            end = exponent_mark;
        }
    }

    value[..end].parse().ok()
}

fn exponential_delay_ms(attempt: u32, random: f64) -> f64 {
    let exponential_delay = (500.0 * 2f64.powi(attempt.min(4) as i32)).min(8_000.0);
    exponential_delay * (1.0 - random.clamp(0.0, 1.0) * 0.25)
}

fn random_unit_interval() -> f64 {
    let mut bytes = [0; 8];
    if SystemRandom::new().fill(&mut bytes).is_err() {
        return 0.5;
    }

    // Use the high 53 bits so every possible result is exactly representable
    // as an f64 in the same [0, 1) range as Math.random().
    let value = u64::from_ne_bytes(bytes) >> 11;
    value as f64 / (1u64 << 53) as f64
}

fn retry_after_http_date_ms(value: &str) -> Option<f64> {
    let target_ms = system_time_epoch_ms(httpdate::parse_http_date(value).ok()?);
    let now_ms = system_time_epoch_ms(SystemTime::now());
    Some(target_ms.saturating_sub(now_ms) as f64)
}

fn system_time_epoch_ms(time: SystemTime) -> i128 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as i128,
        Err(error) => -(error.duration().as_millis() as i128),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use reqwest::header::HeaderValue;
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

    #[tokio::test(flavor = "current_thread")]
    async fn rejects_server_retry_delay_above_configured_cap_without_retrying() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_status_server(
            Arc::clone(&attempts),
            "429 Too Many Requests",
            &[
                ("retry-after", "277403"),
                ("content-type", "application/json"),
            ],
            "",
        )
        .await;
        let client = reqwest::Client::new();
        let options = StreamOptions {
            max_retries: Some(2),
            max_retry_delay_ms: Some(1_000),
            ..Default::default()
        };

        let error = send_with_retries(&options, || client.get(&url))
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Server requested 277403s retry delay (max: 1s)")
        );
        assert!(error.to_string().contains("429 Too Many Requests"));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rejects_server_retry_delay_above_default_cap() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_status_server(
            Arc::clone(&attempts),
            "503 Service Unavailable",
            &[("retry-after", "61")],
            "",
        )
        .await;
        let client = reqwest::Client::new();
        let options = StreamOptions {
            max_retries: Some(1),
            ..Default::default()
        };

        let error = send_with_retries(&options, || client.get(&url))
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Server requested 61s retry delay (max: 60s)")
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn zero_disables_server_retry_delay_cap() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url =
            spawn_retry_server_with_headers(Arc::clone(&attempts), &[("retry-after-ms", "1")])
                .await;
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
    async fn cancellation_interrupts_uncapped_server_backoff() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_status_server(
            Arc::clone(&attempts),
            "429 Too Many Requests",
            &[("retry-after", "277403")],
            "",
        )
        .await;
        let client = reqwest::Client::new();
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        let options = StreamOptions {
            max_retries: Some(2),
            max_retry_delay_ms: Some(0),
            cancellation_token: Some(cancellation_token.clone()),
            ..Default::default()
        };
        let request = send_with_retries(&options, || client.get(&url));
        tokio::pin!(request);

        tokio::select! {
            result = &mut request => panic!("request finished before cancellation: {result:?}"),
            _ = wait_for_attempts(&attempts, 1) => {}
        }
        cancellation_token.cancel();

        assert!(matches!(request.await, Err(Error::Cancelled)));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn x_should_retry_false_prevents_retry() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_status_server(
            Arc::clone(&attempts),
            "429 Too Many Requests",
            &[("x-should-retry", "false")],
            "",
        )
        .await;
        let client = reqwest::Client::new();
        let options = StreamOptions {
            max_retries: Some(2),
            max_retry_delay_ms: Some(0),
            ..Default::default()
        };

        let response = send_with_retries(&options, || client.get(&url))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn x_should_retry_true_allows_non_retryable_status() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let url = spawn_retry_server_with_status_and_headers(
            Arc::clone(&attempts),
            "400 Bad Request",
            &[("x-should-retry", "true"), ("retry-after-ms", "0")],
        )
        .await;
        let client = reqwest::Client::new();
        let options = StreamOptions {
            max_retries: Some(1),
            ..Default::default()
        };

        let response = send_with_retries(&options, || client.get(&url))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn standard_retryable_statuses_remain_retryable() {
        for status in [
            "408 Request Timeout",
            "409 Conflict",
            "429 Too Many Requests",
            "500 Internal Server Error",
            "503 Service Unavailable",
        ] {
            let attempts = Arc::new(AtomicUsize::new(0));
            let url = spawn_retry_server_with_status_and_headers(
                Arc::clone(&attempts),
                status,
                &[("retry-after-ms", "0")],
            )
            .await;
            let client = reqwest::Client::new();
            let options = StreamOptions {
                max_retries: Some(1),
                ..Default::default()
            };

            let response = send_with_retries(&options, || client.get(&url))
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK, "status {status}");
            assert_eq!(attempts.load(Ordering::SeqCst), 2, "status {status}");
        }
    }

    #[test]
    fn parses_retry_after_http_date() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "retry-after",
            HeaderValue::from_static("Fri, 31 Dec 9999 23:59:59 GMT"),
        );

        let delay_ms = retry_after_ms(&headers).unwrap();

        assert!(delay_ms > DEFAULT_MAX_RETRY_DELAY_MS as f64);
    }

    #[test]
    fn retry_after_http_date_in_past_returns_zero() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "retry-after",
            HeaderValue::from_static("Thu, 01 Jan 1970 00:00:00 GMT"),
        );

        assert!(retry_after_ms(&headers).is_some_and(|delay| delay < 0.0));
    }

    #[test]
    fn retry_after_ms_header_takes_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after-ms", HeaderValue::from_static("25"));
        headers.insert(
            "retry-after",
            HeaderValue::from_static("Fri, 31 Dec 9999 23:59:59 GMT"),
        );

        assert_eq!(retry_after_ms(&headers), Some(25.0));
    }

    #[test]
    fn parses_fractional_provider_retry_delays_like_pi() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after-ms", HeaderValue::from_static("25.1"));
        assert_eq!(retry_after_ms(&headers), Some(25.1));

        headers.remove("retry-after-ms");
        headers.insert("retry-after", HeaderValue::from_static("0.025"));
        assert_eq!(retry_after_ms(&headers), Some(25.0));
    }

    #[test]
    fn parses_numeric_prefixes_like_javascript_parse_float() {
        assert_eq!(parse_float_prefix("  -1.5e2junk"), Some(-150.0));
        assert_eq!(parse_float_prefix("1e+oops"), Some(1.0));
        assert_eq!(parse_float_prefix(".25 seconds"), Some(0.25));
        assert_eq!(parse_float_prefix("Infinity"), Some(f64::INFINITY));
        assert_eq!(parse_float_prefix("junk1.5"), None);
    }

    #[test]
    fn invalid_retry_after_is_an_immediate_delay_like_javascript() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("not-a-date"));

        assert!(retry_after_ms(&headers).is_some_and(f64::is_nan));
    }

    #[test]
    fn empty_retry_after_uses_exponential_fallback_like_javascript() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static(""));

        assert!(retry_after_ms(&headers).is_none());
    }

    #[test]
    fn exponential_backoff_matches_pi_curve_and_jitter() {
        assert_eq!(exponential_delay_ms(0, 0.0), 500.0);
        assert_eq!(exponential_delay_ms(0, 1.0), 375.0);
        assert_eq!(exponential_delay_ms(1, 0.0), 1_000.0);
        assert_eq!(exponential_delay_ms(4, 0.0), 8_000.0);
        assert_eq!(exponential_delay_ms(4, 1.0), 6_000.0);
        assert_eq!(exponential_delay_ms(20, 0.0), 8_000.0);
        assert_eq!(exponential_delay_ms(20, 1.0), 6_000.0);
    }

    async fn spawn_retry_server(attempts: Arc<AtomicUsize>) -> String {
        spawn_retry_server_with_headers(attempts, &[("retry-after-ms", "0")]).await
    }

    async fn spawn_retry_server_with_headers(
        attempts: Arc<AtomicUsize>,
        headers: &[(&str, &str)],
    ) -> String {
        spawn_retry_server_with_status_and_headers(attempts, "500 Internal Server Error", headers)
            .await
    }

    async fn spawn_retry_server_with_status_and_headers(
        attempts: Arc<AtomicUsize>,
        status: &str,
        headers: &[(&str, &str)],
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let status = status.to_string();
        let headers = headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\r\n"))
            .collect::<String>();
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
                        "HTTP/1.1 {status}\r\n{headers}content-length: 0\r\nconnection: close\r\n\r\n"
                    )
                } else {
                    "HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_string()
                };
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }

    async fn spawn_status_server(
        attempts: Arc<AtomicUsize>,
        status: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let status = status.to_string();
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
                attempts.fetch_add(1, Ordering::SeqCst);
                let mut buffer = vec![0u8; 1024];
                let _ = socket.read(&mut buffer).await;
                let response = format!(
                    "HTTP/1.1 {status}\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }

    async fn wait_for_attempts(attempts: &AtomicUsize, expected: usize) {
        while attempts.load(Ordering::SeqCst) < expected {
            tokio::task::yield_now().await;
        }
    }
}
