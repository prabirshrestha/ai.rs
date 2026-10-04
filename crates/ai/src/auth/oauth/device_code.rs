//! Port of `auth/oauth/device-code.ts` (RFC 8628 device authorization polling).
//!
//! Deadlines use the tokio clock instead of `Date.now()`, so paused-time tests
//! drive them the way Pi's fake timers do.

use std::future::Future;
use std::time::Duration;

use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

const CANCEL_MESSAGE: &str = "Login cancelled";
const TIMEOUT_MESSAGE: &str = "Device flow timed out";
const SLOW_DOWN_TIMEOUT_MESSAGE: &str = "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.";
const MINIMUM_INTERVAL_MS: u64 = 1000;
// RFC 8628 section 3.2: if the authorization server omits `interval`, the client must use 5 seconds.
const DEFAULT_POLL_INTERVAL_SECONDS: f64 = 5.0;
// RFC 8628 section 3.5: `slow_down` means the polling interval must increase by 5 seconds.
const SLOW_DOWN_INTERVAL_INCREMENT_MS: u64 = 5000;

/// Result of one device-token poll.
#[derive(Debug, Clone, PartialEq)]
pub enum OAuthDeviceCodePollResult<T> {
    Pending,
    SlowDown { interval_seconds: Option<f64> },
    Failed { message: String },
    Complete(T),
}

/// Options of [`poll_oauth_device_code_flow`].
#[derive(Debug, Clone, Default)]
pub struct OAuthDeviceCodePollOptions {
    pub interval_seconds: Option<f64>,
    pub expires_in_seconds: Option<f64>,
    pub wait_before_first_poll: bool,
    pub signal: CancellationToken,
}

/// Sleep for `ms`, rejecting with `cancel_message` when `signal` aborts.
pub async fn abortable_sleep(
    ms: u64,
    signal: &CancellationToken,
    cancel_message: &str,
) -> Result<()> {
    if signal.is_cancelled() {
        return Err(Error::message(cancel_message));
    }
    tokio::select! {
        _ = signal.cancelled() => Err(Error::message(cancel_message)),
        _ = tokio::time::sleep(Duration::from_millis(ms)) => Ok(()),
    }
}

fn interval_ms(seconds: f64) -> u64 {
    MINIMUM_INTERVAL_MS.max((seconds * 1000.0).floor().max(0.0) as u64)
}

fn remaining_ms(deadline: Option<Instant>) -> Option<u64> {
    deadline.map(|deadline| {
        deadline
            .saturating_duration_since(Instant::now())
            .as_millis() as u64
    })
}

/// `pollOAuthDeviceCodeFlow()`: poll until the flow completes, fails, the
/// device code expires, or `signal` aborts.
pub async fn poll_oauth_device_code_flow<T, F, Fut>(
    options: OAuthDeviceCodePollOptions,
    mut poll: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<OAuthDeviceCodePollResult<T>>>,
{
    let deadline = options
        .expires_in_seconds
        .map(|seconds| Instant::now() + Duration::from_millis((seconds * 1000.0).max(0.0) as u64));
    let mut interval = interval_ms(
        options
            .interval_seconds
            .unwrap_or(DEFAULT_POLL_INTERVAL_SECONDS),
    );

    let mut slow_down_responses = 0u32;
    if options.wait_before_first_poll {
        let remaining = remaining_ms(deadline).unwrap_or(u64::MAX);
        if remaining > 0 {
            abortable_sleep(interval.min(remaining), &options.signal, CANCEL_MESSAGE).await?;
        }
    }

    while deadline.is_none_or(|deadline| Instant::now() < deadline) {
        if options.signal.is_cancelled() {
            return Err(Error::message(CANCEL_MESSAGE));
        }

        match poll().await? {
            OAuthDeviceCodePollResult::Complete(value) => return Ok(value),
            OAuthDeviceCodePollResult::Failed { message } => return Err(Error::message(message)),
            OAuthDeviceCodePollResult::Pending => {}
            OAuthDeviceCodePollResult::SlowDown { interval_seconds } => {
                slow_down_responses += 1;
                // Use the server-provided interval when given (GitHub reports the new required
                // minimum in `interval`); trusting only a client-tracked value risks polling early
                // forever under WSL/VM clock drift. Otherwise apply RFC 8628 section 3.5: increase
                // by 5 seconds.
                interval = match interval_seconds {
                    Some(seconds) if seconds.is_finite() && seconds > 0.0 => interval_ms(seconds),
                    _ => MINIMUM_INTERVAL_MS.max(interval + SLOW_DOWN_INTERVAL_INCREMENT_MS),
                };
            }
        }

        let remaining = remaining_ms(deadline).unwrap_or(u64::MAX);
        if remaining == 0 {
            break;
        }

        abortable_sleep(interval.min(remaining), &options.signal, CANCEL_MESSAGE).await?;
    }

    Err(Error::message(if slow_down_responses > 0 {
        SLOW_DOWN_TIMEOUT_MESSAGE
    } else {
        TIMEOUT_MESSAGE
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use parking_lot::Mutex;

    use super::*;

    type Times = Arc<Mutex<Vec<u64>>>;

    fn spawn_poll(
        options: OAuthDeviceCodePollOptions,
        results: Vec<OAuthDeviceCodePollResult<&'static str>>,
    ) -> (Times, tokio::task::JoinHandle<Result<&'static str>>) {
        let times: Times = Arc::default();
        let start = Instant::now();
        let results = Arc::new(Mutex::new(VecDeque::from(results)));
        let poll_times = times.clone();
        let handle = tokio::spawn(async move {
            poll_oauth_device_code_flow(options, move || {
                let poll_times = poll_times.clone();
                let results = results.clone();
                async move {
                    poll_times
                        .lock()
                        .push(Instant::now().duration_since(start).as_millis() as u64);
                    results
                        .lock()
                        .pop_front()
                        .ok_or_else(|| Error::message("Unexpected extra poll"))
                }
            })
            .await
        });
        (times, handle)
    }

    async fn advance(ms: u64) {
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_millis(ms)).await;
        tokio::task::yield_now().await;
    }

    fn options(interval: f64, expires: f64) -> OAuthDeviceCodePollOptions {
        OAuthDeviceCodePollOptions {
            interval_seconds: Some(interval),
            expires_in_seconds: Some(expires),
            ..Default::default()
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn polls_immediately_and_returns_the_completed_value() {
        let (times, handle) = spawn_poll(
            options(2.0, 30.0),
            vec![
                OAuthDeviceCodePollResult::Pending,
                OAuthDeviceCodePollResult::Complete("token"),
            ],
        );
        advance(0).await;
        assert_eq!(*times.lock(), vec![0]);
        advance(1999).await;
        assert_eq!(*times.lock(), vec![0]);
        advance(1).await;
        assert_eq!(handle.await.unwrap().unwrap(), "token");
        assert_eq!(*times.lock(), vec![0, 2000]);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn can_wait_before_the_first_poll() {
        let (times, handle) = spawn_poll(
            OAuthDeviceCodePollOptions {
                wait_before_first_poll: true,
                ..options(2.0, 30.0)
            },
            vec![OAuthDeviceCodePollResult::Complete("token")],
        );
        advance(1999).await;
        assert!(times.lock().is_empty());
        advance(1).await;
        assert_eq!(handle.await.unwrap().unwrap(), "token");
        assert_eq!(*times.lock(), vec![2000]);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn increases_the_interval_by_5_seconds_after_slow_down_without_a_server_interval() {
        let (times, handle) = spawn_poll(
            options(2.0, 900.0),
            vec![
                OAuthDeviceCodePollResult::SlowDown {
                    interval_seconds: None,
                },
                OAuthDeviceCodePollResult::Complete("token"),
            ],
        );
        advance(0).await;
        assert_eq!(*times.lock(), vec![0]);
        advance(6999).await;
        assert_eq!(*times.lock(), vec![0]);
        advance(1).await;
        assert_eq!(handle.await.unwrap().unwrap(), "token");
        assert_eq!(*times.lock(), vec![0, 7000]);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn honors_a_server_provided_slow_down_interval() {
        let (times, handle) = spawn_poll(
            options(2.0, 900.0),
            vec![
                OAuthDeviceCodePollResult::SlowDown {
                    interval_seconds: Some(30.0),
                },
                OAuthDeviceCodePollResult::Complete("token"),
            ],
        );
        advance(0).await;
        assert_eq!(*times.lock(), vec![0]);
        advance(29_999).await;
        assert_eq!(*times.lock(), vec![0]);
        advance(1).await;
        assert_eq!(handle.await.unwrap().unwrap(), "token");
        assert_eq!(*times.lock(), vec![0, 30_000]);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn cancels_an_in_flight_wait() {
        let signal = CancellationToken::new();
        let (_, handle) = spawn_poll(
            OAuthDeviceCodePollOptions {
                signal: signal.clone(),
                ..options(5.0, 30.0)
            },
            vec![OAuthDeviceCodePollResult::Pending; 10],
        );
        advance(0).await;
        signal.cancel();
        let error = handle.await.unwrap().unwrap_err();
        assert_eq!(error.to_string(), "Login cancelled");
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn reports_failures_and_slow_down_timeouts() {
        let (_, handle) = spawn_poll(
            options(1.0, 1.0),
            vec![OAuthDeviceCodePollResult::Failed {
                message: "nope".to_string(),
            }],
        );
        assert_eq!(handle.await.unwrap().unwrap_err().to_string(), "nope");

        let (times, handle) = spawn_poll(
            options(1.0, 1.0),
            vec![OAuthDeviceCodePollResult::SlowDown {
                interval_seconds: None,
            }],
        );
        let error = handle.await.unwrap().unwrap_err();
        assert_eq!(times.lock().len(), 1);
        assert_eq!(error.to_string(), SLOW_DOWN_TIMEOUT_MESSAGE);

        let (_, handle) = spawn_poll(options(1.0, 1.0), vec![OAuthDeviceCodePollResult::Pending]);
        assert_eq!(
            handle.await.unwrap().unwrap_err().to_string(),
            TIMEOUT_MESSAGE
        );
    }
}
