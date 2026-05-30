use std::time::Duration;

use reqwest::{RequestBuilder, Response, StatusCode, header::HeaderMap};

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
            let value = headers
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .map(str::trim)?;
            value
                .parse::<u64>()
                .ok()
                .map(|seconds| seconds.saturating_mul(1000))
                .or_else(|| retry_after_http_date_ms(value))
        })
}

fn exponential_delay_ms(attempt: u32) -> u64 {
    500u64.saturating_mul(2u64.saturating_pow(attempt.min(6)))
}

fn retry_after_http_date_ms(value: &str) -> Option<u64> {
    let epoch_seconds = parse_imf_fixdate_epoch_seconds(value)?;
    let target_ms = i128::from(epoch_seconds).saturating_mul(1000);
    let now_ms = i128::from(crate::utils::time::now_millis());
    Some(target_ms.saturating_sub(now_ms).max(0) as u64)
}

fn parse_imf_fixdate_epoch_seconds(value: &str) -> Option<i64> {
    let parts: Vec<&str> = value.split_ascii_whitespace().collect();
    let [weekday, day, month, year, time, zone] = parts.as_slice() else {
        return None;
    };
    if !weekday.ends_with(',') || !zone.eq_ignore_ascii_case("GMT") {
        return None;
    }

    let day = day.parse::<u32>().ok()?;
    let month = parse_month(month)?;
    let year = year.parse::<i32>().ok()?;
    let (hour, minute, second) = parse_hms(time)?;
    if day == 0 || day > days_in_month(year, month) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }

    let days = days_from_civil(year, month, day);
    Some(
        days.saturating_mul(86_400)
            .saturating_add(i64::from(hour) * 3600)
            .saturating_add(i64::from(minute) * 60)
            .saturating_add(i64::from(second)),
    )
}

fn parse_month(value: &str) -> Option<u32> {
    match value {
        "Jan" => Some(1),
        "Feb" => Some(2),
        "Mar" => Some(3),
        "Apr" => Some(4),
        "May" => Some(5),
        "Jun" => Some(6),
        "Jul" => Some(7),
        "Aug" => Some(8),
        "Sep" => Some(9),
        "Oct" => Some(10),
        "Nov" => Some(11),
        "Dec" => Some(12),
        _ => None,
    }
}

fn parse_hms(value: &str) -> Option<(u32, u32, u32)> {
    let mut parts = value.split(':');
    let hour = parts.next()?.parse::<u32>().ok()?;
    let minute = parts.next()?.parse::<u32>().ok()?;
    let second = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((hour, minute, second))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month_prime = i64::from(month) + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month_prime + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
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

    #[test]
    fn parses_retry_after_http_date() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "retry-after",
            HeaderValue::from_static("Fri, 31 Dec 9999 23:59:59 GMT"),
        );

        let delay_ms = retry_after_ms(&headers).unwrap();

        assert!(delay_ms > DEFAULT_MAX_RETRY_DELAY_MS);
    }

    #[test]
    fn retry_after_http_date_in_past_returns_zero() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "retry-after",
            HeaderValue::from_static("Thu, 01 Jan 1970 00:00:00 GMT"),
        );

        assert_eq!(retry_after_ms(&headers), Some(0));
    }

    #[test]
    fn retry_after_ms_header_takes_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after-ms", HeaderValue::from_static("25"));
        headers.insert(
            "retry-after",
            HeaderValue::from_static("Fri, 31 Dec 9999 23:59:59 GMT"),
        );

        assert_eq!(retry_after_ms(&headers), Some(25));
    }

    #[test]
    fn parses_imf_fixdate_epoch_seconds() {
        assert_eq!(
            parse_imf_fixdate_epoch_seconds("Thu, 01 Jan 1970 00:00:00 GMT"),
            Some(0)
        );
        assert_eq!(
            parse_imf_fixdate_epoch_seconds("Wed, 21 Oct 2015 07:28:00 GMT"),
            Some(1_445_412_480)
        );
        assert_eq!(
            parse_imf_fixdate_epoch_seconds("Wed, 31 Feb 2015 07:28:00 GMT"),
            None
        );
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
