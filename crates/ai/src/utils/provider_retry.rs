//! Port of `utils/provider-retry.ts`: the OpenAI/Anthropic SDK retry policy
//! with an interruptible backoff sleep.
//!
//! Pi wraps SDK calls (made with `maxRetries: 0`) in `retryProviderRequest()`.
//! The Rust API modules issue raw HTTP requests, so a non-2xx response is
//! surfaced as [`ProviderHttpError`] (the SDK `APIError` shape: status,
//! headers, body) and transport failures as [`Error::Http`], which the SDK
//! reports as connection errors without a status.

use std::fmt;
use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use indexmap::IndexMap;
use ring::rand::{SecureRandom, SystemRandom};
use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 60_000;

#[derive(Debug, Clone, Default)]
pub struct ProviderRetryOptions {
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
    pub signal: Option<CancellationToken>,
}

/// A provider HTTP error carrying the response status, headers and body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderHttpError {
    pub status: Option<u16>,
    /// Response headers with lower-cased names.
    pub headers: IndexMap<String, String>,
    /// Raw response body, when one was read.
    pub body: Option<String>,
    /// Display message, `"<status> <body>"` or `"<status> status code (no body)"`
    /// like the OpenAI and Anthropic SDKs.
    pub message: String,
}

impl ProviderHttpError {
    pub fn new(status: u16, headers: IndexMap<String, String>, body: Option<String>) -> Self {
        let body = body.filter(|body| !body.trim().is_empty());
        let message = match &body {
            Some(body) => format!("{status} {body}"),
            None => format!("{status} status code (no body)"),
        };
        Self {
            status: Some(status),
            headers: headers
                .into_iter()
                .map(|(name, value)| (name.to_ascii_lowercase(), value))
                .collect(),
            body,
            message,
        }
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }
}

impl fmt::Display for ProviderHttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderHttpError {}

type ProviderErrorParts<'a> = (Option<u16>, Option<&'a IndexMap<String, String>>);

/// `isProviderError()`: an error with an HTTP status and headers, or a
/// transport failure (no status, no headers).
fn as_provider_error(error: &Error) -> Option<ProviderErrorParts<'_>> {
    match error {
        Error::ProviderHttp(error) => Some((error.status, Some(&error.headers))),
        Error::Http(error) if error.status().is_none() => Some((None, None)),
        _ => None,
    }
}

/// Mirrors the pinned OpenAI/Anthropic SDK retry policy.
fn is_retryable_provider_error(
    status: Option<u16>,
    headers: Option<&IndexMap<String, String>>,
) -> bool {
    match headers
        .and_then(|headers| headers.get("x-should-retry"))
        .map(String::as_str)
    {
        Some("true") => return true,
        Some("false") => return false,
        _ => {}
    }
    match status {
        None => true,
        Some(status) => status == 408 || status == 409 || status == 429 || status >= 500,
    }
}

fn validate_server_retry_delay_ms(
    delay_ms: f64,
    max_retry_delay_ms: Option<u64>,
    provider_error_message: &str,
) -> Result<f64> {
    let max_delay_ms = max_retry_delay_ms.unwrap_or(DEFAULT_MAX_RETRY_DELAY_MS);
    if max_delay_ms > 0 && delay_ms > max_delay_ms as f64 {
        return Err(Error::message(format!(
            "Server requested {}s retry delay (max: {}s). {provider_error_message}",
            (delay_ms / 1000.0).ceil(),
            max_delay_ms.div_ceil(1000)
        )));
    }
    Ok(delay_ms)
}

fn get_retry_delay_ms(
    headers: Option<&IndexMap<String, String>>,
    retry_index: u32,
    max_retry_delay_ms: Option<u64>,
    message: &str,
) -> Result<f64> {
    let header = |name: &str| {
        headers
            .and_then(|headers| headers.get(name))
            .map(String::as_str)
            .filter(|value| !value.is_empty())
    };

    if let Some(retry_after_ms) = header("retry-after-ms")
        && let Some(value) = parse_float(retry_after_ms)
        && value.is_finite()
    {
        return validate_server_retry_delay_ms(value, max_retry_delay_ms, message);
    }

    if let Some(retry_after) = header("retry-after") {
        let delay_ms = match parse_float(retry_after) {
            Some(seconds) => Some(seconds * 1000.0),
            None => parse_http_date_ms(retry_after).map(|target| target - now_ms()),
        };
        if let Some(delay_ms) = delay_ms.filter(|delay| delay.is_finite()) {
            return validate_server_retry_delay_ms(delay_ms, max_retry_delay_ms, message);
        }
    }

    Ok(exponential_delay_ms(retry_index, random_unit_interval()))
}

/// `Math.min(0.5 * 2 ** retryIndex, 8) * 1000 * (1 - Math.random() * 0.25)`.
pub(crate) fn exponential_delay_ms(retry_index: u32, random: f64) -> f64 {
    let exponential_delay = (0.5 * 2f64.powi(retry_index.min(64) as i32)).min(8.0) * 1000.0;
    exponential_delay * (1.0 - random * 0.25)
}

/// `Math.random()` from the system CSPRNG.
pub(crate) fn random_unit_interval() -> f64 {
    let mut bytes = [0; 8];
    if SystemRandom::new().fill(&mut bytes).is_err() {
        return 0.5;
    }
    // The high 53 bits give every result exactly in [0, 1).
    let value = u64::from_ne_bytes(bytes) >> 11;
    value as f64 / (1u64 << 53) as f64
}

/// JavaScript `Number.parseFloat` for the decimal forms that occur in HTTP
/// headers: a valid numeric prefix is accepted. Returns `None` for `NaN`.
pub(crate) fn parse_float(value: &str) -> Option<f64> {
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

/// `Date.parse()` for HTTP dates. Unparseable dates give `None` (`NaN`).
fn parse_http_date_ms(value: &str) -> Option<f64> {
    let time = httpdate::parse_http_date(value.trim()).ok()?;
    Some(system_time_ms(time))
}

fn now_ms() -> f64 {
    system_time_ms(SystemTime::now())
}

fn system_time_ms(time: SystemTime) -> f64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as f64,
        Err(error) => -(error.duration().as_millis() as f64),
    }
}

fn create_abort_error() -> Error {
    Error::Aborted("Request aborted".to_string())
}

async fn abortable_sleep(ms: f64, signal: Option<&CancellationToken>) -> Result<()> {
    if signal.is_some_and(CancellationToken::is_cancelled) {
        return Err(create_abort_error());
    }
    let delay = if ms.is_nan() || ms <= 0.0 {
        Duration::ZERO
    } else if ms.is_finite() {
        Duration::from_secs_f64(ms / 1000.0)
    } else {
        Duration::MAX
    };
    match signal {
        Some(signal) => tokio::select! {
            _ = signal.cancelled() => Err(create_abort_error()),
            _ = tokio::time::sleep(delay) => Ok(()),
        },
        None => {
            tokio::time::sleep(delay).await;
            Ok(())
        }
    }
}

/// Reproduce the retry behavior used by the OpenAI and Anthropic SDKs while
/// making their backoff sleep interruptible. Provider-requested delays above
/// `max_retry_delay_ms` fail immediately (60 seconds by default); zero
/// disables the limit.
pub async fn retry_provider_request<T, F, Fut>(
    mut request: F,
    options: &ProviderRetryOptions,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let max_retries = options.max_retries.unwrap_or(0);
    let mut retries_remaining = max_retries;
    let signal = options.signal.as_ref();

    loop {
        let error = match request().await {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        if signal.is_some_and(CancellationToken::is_cancelled) {
            return Err(create_abort_error());
        }
        let Some((status, headers)) = as_provider_error(&error) else {
            return Err(error);
        };
        if retries_remaining == 0 || !is_retryable_provider_error(status, headers) {
            return Err(error);
        }

        let retry_index = max_retries - retries_remaining;
        retries_remaining -= 1;
        let delay_ms = get_retry_delay_ms(
            headers,
            retry_index,
            options.max_retry_delay_ms,
            &error.to_string(),
        )?;
        abortable_sleep(delay_ms, signal).await?;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    fn provider_error(status: u16, headers: &[(&str, &str)]) -> Error {
        let mut error = ProviderHttpError::new(
            status,
            headers
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
            None,
        );
        error.message = format!("Provider error: {status}");
        Error::from(error)
    }

    fn options(max_retries: u32, max_retry_delay_ms: Option<u64>) -> ProviderRetryOptions {
        ProviderRetryOptions {
            max_retries: Some(max_retries),
            max_retry_delay_ms,
            signal: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn retries_retryable_provider_errors() {
        let calls = Arc::new(AtomicU32::new(0));
        let observed = Arc::clone(&calls);
        let started = tokio::time::Instant::now();
        let result = retry_provider_request(
            || {
                let call = observed.fetch_add(1, Ordering::SeqCst);
                async move {
                    if call == 0 {
                        Err(provider_error(429, &[("retry-after-ms", "1000")]))
                    } else {
                        Ok("ok")
                    }
                }
            },
            &options(1, None),
        )
        .await
        .unwrap();
        assert_eq!(result, "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(started.elapsed(), Duration::from_millis(1000));
    }

    #[tokio::test]
    async fn does_not_retry_errors_the_provider_marks_as_non_retryable() {
        let calls = AtomicU32::new(0);
        let error = retry_provider_request::<(), _, _>(
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err(provider_error(429, &[("x-should-retry", "false")])) }
            },
            &options(2, None),
        )
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "Provider error: 429");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn rejects_a_provider_requested_retry_delay_above_the_limit() {
        let calls = AtomicU32::new(0);
        let error = retry_provider_request::<(), _, _>(
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err(provider_error(429, &[("retry-after", "277403")])) }
            },
            &options(1, Some(1000)),
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Server requested 277403s retry delay (max: 1s)")
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn allows_disabling_the_provider_requested_retry_delay_cap() {
        let calls = AtomicU32::new(0);
        let started = tokio::time::Instant::now();
        let result = retry_provider_request(
            || {
                let call = calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    if call == 0 {
                        Err(provider_error(429, &[("retry-after", "2")]))
                    } else {
                        Ok("ok")
                    }
                }
            },
            &options(1, Some(0)),
        )
        .await
        .unwrap();
        assert_eq!(result, "ok");
        assert_eq!(started.elapsed(), Duration::from_millis(2000));
    }

    #[tokio::test(start_paused = true)]
    async fn aborts_a_provider_requested_retry_delay() {
        let calls = Arc::new(AtomicU32::new(0));
        let controller = CancellationToken::new();
        let observed = Arc::clone(&calls);
        let abort = controller.clone();
        tokio::spawn(async move {
            while observed.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
            abort.cancel();
        });
        let error = retry_provider_request::<(), _, _>(
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err(provider_error(429, &[("retry-after", "277403")])) }
            },
            &ProviderRetryOptions {
                max_retries: Some(2),
                max_retry_delay_ms: Some(0),
                signal: Some(controller),
            },
        )
        .await
        .unwrap_err();
        assert!(error.is_abort());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn unparseable_retry_after_dates_fall_back_to_exponential_backoff() {
        let calls = AtomicU32::new(0);
        let started = tokio::time::Instant::now();
        retry_provider_request(
            || {
                let call = calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    if call == 0 {
                        Err(provider_error(503, &[("retry-after", "not-a-date")]))
                    } else {
                        Ok(())
                    }
                }
            },
            &options(1, None),
        )
        .await
        .unwrap();
        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_millis(375) && elapsed <= Duration::from_millis(500));
    }

    #[tokio::test]
    async fn standard_retryable_statuses_remain_retryable() {
        for status in [408, 409, 429, 500, 503] {
            let calls = AtomicU32::new(0);
            retry_provider_request(
                || {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    async move {
                        if call == 0 {
                            Err(provider_error(status, &[("retry-after-ms", "0")]))
                        } else {
                            Ok(())
                        }
                    }
                },
                &options(1, None),
            )
            .await
            .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 2, "status {status}");
        }
        let calls = AtomicU32::new(0);
        retry_provider_request(
            || {
                let call = calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    if call == 0 {
                        Err(provider_error(
                            400,
                            &[("x-should-retry", "true"), ("retry-after-ms", "0")],
                        ))
                    } else {
                        Ok(())
                    }
                }
            },
            &options(1, None),
        )
        .await
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn parses_numeric_prefixes_like_javascript_parse_float() {
        assert_eq!(parse_float("  -1.5e2junk"), Some(-150.0));
        assert_eq!(parse_float("1e+oops"), Some(1.0));
        assert_eq!(parse_float(".25 seconds"), Some(0.25));
        assert_eq!(parse_float("Infinity"), Some(f64::INFINITY));
        assert_eq!(parse_float("junk1.5"), None);
    }

    #[test]
    fn exponential_backoff_matches_sdk_curve_and_jitter() {
        assert_eq!(exponential_delay_ms(0, 0.0), 500.0);
        assert_eq!(exponential_delay_ms(0, 1.0), 375.0);
        assert_eq!(exponential_delay_ms(1, 0.0), 1_000.0);
        assert_eq!(exponential_delay_ms(4, 0.0), 8_000.0);
        assert_eq!(exponential_delay_ms(20, 1.0), 6_000.0);
    }

    #[test]
    fn provider_http_error_messages_match_sdk_wording() {
        assert_eq!(
            ProviderHttpError::new(400, IndexMap::new(), None).to_string(),
            "400 status code (no body)"
        );
        let error = ProviderHttpError::new(
            429,
            [("Retry-After".to_string(), "1".to_string())]
                .into_iter()
                .collect(),
            Some("{\"error\":\"slow down\"}".to_string()),
        );
        assert_eq!(error.to_string(), "429 {\"error\":\"slow down\"}");
        assert_eq!(error.header("retry-after"), Some("1"));
    }
}
