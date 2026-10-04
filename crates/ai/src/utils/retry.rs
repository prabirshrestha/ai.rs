//! Port of `utils/retry.ts`: assistant-level retry classification and the
//! policy-driven retry loop.

use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use regex::{Regex, RegexBuilder};
use tokio_util::sync::CancellationToken;

use crate::Result;
use crate::types::{AssistantMessage, BoxFuture, StopReason};

fn build_provider_error_pattern(patterns: &[&str]) -> Regex {
    RegexBuilder::new(&patterns.join("|"))
        .case_insensitive(true)
        .build()
        .expect("provider error pattern compiles")
}

fn non_retryable_provider_limit_error_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        build_provider_error_pattern(&[
            // OpenCode Go/free-tier limits returned as 429 JSON error types.
            "GoUsageLimitError",
            "FreeUsageLimitError",
            // OpenCode Go subscription-limit text.
            "Monthly usage limit reached",
            "available balance",
            // Generic quota/budget/billing exhaustion.
            "insufficient_quota",
            "out of budget",
            "quota exceeded",
            "billing",
            // Sign in with ChatGPT: the subscription's shared usage limit.
            "subscription_sharing_usage_limit_exceeded",
        ])
    })
}

fn retryable_provider_error_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        build_provider_error_pattern(&[
            // Generic provider load, HTTP status, and server-side transient failures.
            "overloaded",
            "currently experiencing high demand",
            "model is at capacity",
            "rate.?limit",
            "too many requests",
            "429",
            "500",
            "502",
            "503",
            "504",
            "520",
            "524",
            "service.?unavailable",
            "server.?error",
            "internal.?error",
            // Wrapper/provider text for transient upstream failures.
            "provider.?returned.?error",
            "exceeded request buffer limit while retrying upstream",
            // Network, proxy, and fetch transport failures.
            "network.?error",
            "connection.?error",
            "connection.?refused",
            "connection.?lost",
            "other side closed",
            "fetch failed",
            "getaddrinfo",
            "ENOTFOUND",
            "EAI_AGAIN",
            "upstream.?connect",
            "reset before headers",
            "socket hang up",
            "socket connection was closed",
            "timed? out",
            "timeout",
            "terminated",
            // WebSocket transports.
            "websocket.?closed",
            "websocket.?error",
            // Premature stream endings from SDKs and transports.
            "ended without",
            "stream ended before message_stop",
            "stream ended before a terminal response event",
            "http2 request did not get a response",
            // Provider-requested retry delay cap failures.
            "retry delay",
            // Explicit retry guidance emitted mid-stream.
            "you can retry your request",
            "try your request again",
            "please retry your request",
            // gRPC based providers (e.g. NVIDIA NIM)
            "ResourceExhausted",
            // Sign in with ChatGPT: usage or user data temporarily unavailable.
            "subscription_sharing_usage_unavailable",
            "subscription_sharing_user_unavailable",
        ])
    })
}

/// Retry policy: bounded attempts with exponential backoff
/// (`base_delay_ms * 2^(attempt-1)`), each delay capped by
/// `max_agent_delay_ms` (default 60 seconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub enabled: bool,
    /// Max retry attempts (0 = no retries). The initial call never counts as a retry.
    pub max_retries: u32,
    /// Base delay in ms.
    pub base_delay_ms: u64,
    /// Optional cap for agent-level retry delays in ms. Defaults to 60 seconds.
    pub max_agent_delay_ms: Option<u64>,
}

pub const DEFAULT_MAX_AGENT_RETRY_DELAY_MS: u64 = 60_000;
const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

pub fn retry_delay_ms(base_delay_ms: u64, max_agent_delay_ms: Option<u64>, attempt: u32) -> u64 {
    let exponent = attempt.saturating_sub(1);
    let delay = 2u64
        .checked_pow(exponent)
        .and_then(|factor| base_delay_ms.checked_mul(factor))
        .filter(|delay| *delay <= MAX_SAFE_INTEGER)
        .unwrap_or(MAX_SAFE_INTEGER);
    delay.min(max_agent_delay_ms.unwrap_or(DEFAULT_MAX_AGENT_RETRY_DELAY_MS))
}

/// Optional callbacks emitted by [`retry_assistant_call`] around each retry.
#[derive(Clone, Default)]
#[allow(clippy::type_complexity)]
pub struct RetryCallbacks {
    /// Emitted before the backoff sleep of each retry attempt (1-indexed):
    /// `(attempt, max_attempts, delay_ms, error_message)`.
    pub on_retry_scheduled:
        Option<Arc<dyn Fn(u32, u32, u64, String) -> BoxFuture<()> + Send + Sync>>,
    /// Emitted after the backoff sleep, immediately before the retried call starts.
    pub on_retry_attempt_start: Option<Arc<dyn Fn() -> BoxFuture<()> + Send + Sync>>,
    /// Emitted once when the loop ends: `(success, attempt, final_error)`.
    pub on_retry_finished:
        Option<Arc<dyn Fn(bool, u32, Option<String>) -> BoxFuture<()> + Send + Sync>>,
}

impl RetryCallbacks {
    async fn retry_finished(&self, success: bool, attempt: u32, final_error: Option<String>) {
        if let Some(callback) = &self.on_retry_finished {
            callback(success, attempt, final_error).await;
        }
    }
}

/// Run a single assistant-producing call with bounded retry on transient
/// errors. Aborts are terminal and never retried; aborts during the backoff
/// sleep return the last error message converted to an aborted message.
/// When `policy` is `None` or disabled, the first response is returned
/// unchanged.
pub async fn retry_assistant_call<F, Fut>(
    mut produce: F,
    policy: Option<&RetryPolicy>,
    signal: Option<&CancellationToken>,
    callbacks: Option<&RetryCallbacks>,
) -> Result<AssistantMessage>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<AssistantMessage>>,
{
    let default_callbacks = RetryCallbacks::default();
    let callbacks = callbacks.unwrap_or(&default_callbacks);
    let max_attempts = policy
        .filter(|policy| policy.enabled)
        .map_or(0, |policy| policy.max_retries);

    let mut attempt = 0;
    let mut last_retry: Option<(u32, String)> = None;
    loop {
        let response = produce().await?;

        // Abort: terminal but not successful. Never retry an aborted message.
        if response.stop_reason == StopReason::Aborted {
            if let Some((attempt, _)) = &last_retry {
                callbacks.retry_finished(false, *attempt, None).await;
            }
            return Ok(response);
        }

        // Success: non-error, non-abort responses return as-is.
        if response.stop_reason != StopReason::Error {
            if let Some((attempt, _)) = &last_retry {
                callbacks.retry_finished(true, *attempt, None).await;
            }
            return Ok(response);
        }

        // Non-retryable, or budget exhausted: return the final error message.
        if attempt >= max_attempts || !is_retryable_assistant_error(&response) {
            if let Some((attempt, _)) = &last_retry {
                callbacks
                    .retry_finished(false, *attempt, response.error_message.clone())
                    .await;
            }
            return Ok(response);
        }

        attempt += 1;
        let error_message = response
            .error_message
            .clone()
            .filter(|message| !message.is_empty())
            .unwrap_or_else(|| "Unknown error".to_string());
        last_retry = Some((attempt, error_message.clone()));
        let policy = policy.expect("retries require a policy");
        let delay_ms = retry_delay_ms(policy.base_delay_ms, policy.max_agent_delay_ms, attempt);
        if let Some(callback) = &callbacks.on_retry_scheduled {
            callback(attempt, max_attempts, delay_ms, error_message.clone()).await;
        }

        // Normalize aborts during retry backoff to the same message shape as
        // provider stream aborts.
        if !sleep(delay_ms, signal).await {
            callbacks
                .retry_finished(false, attempt, Some(error_message))
                .await;
            let mut aborted = response;
            aborted.error_message = None;
            aborted.stop_reason = StopReason::Aborted;
            return Ok(aborted);
        }
        if let Some(callback) = &callbacks.on_retry_attempt_start {
            callback().await;
        }
    }
}

/// Returns false when `signal` aborted the sleep.
async fn sleep(ms: u64, signal: Option<&CancellationToken>) -> bool {
    match signal {
        Some(signal) => {
            if signal.is_cancelled() {
                return false;
            }
            tokio::select! {
                _ = signal.cancelled() => false,
                _ = tokio::time::sleep(Duration::from_millis(ms)) => true,
            }
        }
        None => {
            tokio::time::sleep(Duration::from_millis(ms)).await;
            true
        }
    }
}

/// Classifies whether a failed assistant message looks like a transient
/// provider or transport error. Does not implement retry policy.
pub fn is_retryable_assistant_error(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    let Some(error_message) = message
        .error_message
        .as_deref()
        .filter(|error| !error.is_empty())
    else {
        return false;
    };
    if non_retryable_provider_limit_error_pattern().is_match(error_message) {
        return false;
    }
    retryable_provider_error_pattern().is_match(error_message)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use parking_lot::Mutex;

    use super::*;
    use crate::types::{AssistantContent, Model};

    fn message(
        text: &str,
        stop_reason: StopReason,
        error_message: Option<&str>,
    ) -> AssistantMessage {
        let mut message = AssistantMessage::empty_for(&Model::default());
        if !text.is_empty() {
            message.content = vec![AssistantContent::text(text)];
        }
        message.stop_reason = stop_reason;
        message.error_message = error_message.map(str::to_string);
        message
    }

    fn error(error_message: &str) -> AssistantMessage {
        message("", StopReason::Error, Some(error_message))
    }

    fn retryable(error_message: &str) -> bool {
        is_retryable_assistant_error(&error(error_message))
    }

    #[test]
    fn matches_explicit_provider_retry_guidance() {
        assert!(retryable(
            "An error occurred while processing your request. You can retry your request, or contact us through our help center at help.openai.com if the error persists. Please include the request ID req_******** in your message."
        ));
        assert!(retryable(
            r#"{"message":"The system encountered an unexpected error during processing. Try your request again."}"#
        ));
        assert!(retryable(
            "ResourceExhausted: Worker local total request limit reached (288/48)"
        ));
    }

    #[test]
    fn matches_bun_fetch_socket_drop_wording() {
        assert!(retryable(
            "The socket connection was closed unexpectedly. For more information, pass `verbose: true` in the second argument to fetch()"
        ));
    }

    #[test]
    fn matches_upstream_request_buffer_exhaustion_wording() {
        assert!(retryable(
            "Error: exceeded request buffer limit while retrying upstream"
        ));
    }

    #[test]
    fn matches_dns_transport_failure_wording() {
        for error_message in [
            "The pending stream has been canceled (caused by: getaddrinfo ENOTFOUND bedrock-runtime.us-east-1.amazonaws.com)",
            "connect ENOTFOUND api.example.com",
            "EAI_AGAIN api.example.com",
            "getaddrinfo failed for api.example.com",
        ] {
            assert!(retryable(error_message), "{error_message}");
        }
    }

    #[test]
    fn matches_openai_responses_streams_that_end_before_terminal_events() {
        assert!(retryable(
            "OpenAI Responses stream ended before a terminal response event"
        ));
    }

    #[test]
    fn matches_azure_peak_load_capacity_errors() {
        assert!(retryable(
            "The system is currently experiencing high demand and cannot process your request. Your request exceeds the maximum usage size allowed during peak load. For improved capacity reliability, consider switching to Provisioned Throughput."
        ));
    }

    #[test]
    fn keeps_provider_limit_errors_non_retryable() {
        assert!(!retryable("429 quota exceeded"));
        assert!(!retryable(
            r#"OpenAI API error (429): {"code":"subscription_sharing_usage_limit_exceeded","message":"Usage limit reached."}"#
        ));
    }

    #[test]
    fn retries_temporary_chatgpt_subscription_errors() {
        assert!(retryable(
            "subscription_sharing_usage_unavailable: Usage cannot be checked."
        ));
        assert!(retryable(
            "subscription_sharing_user_unavailable: User cannot be loaded."
        ));
    }

    #[test]
    fn classifies_assistant_error_messages() {
        assert!(retryable("overloaded_error"));
        assert!(retryable("520 status code (no body)"));
        assert!(retryable("524 status code (no body)"));
        assert!(!is_retryable_assistant_error(&message(
            "not an error",
            StopReason::Stop,
            None
        )));
    }

    #[test]
    fn caps_agent_retry_delay() {
        assert_eq!(retry_delay_ms(2000, None, 6), 60000);
        assert_eq!(retry_delay_ms(2000, Some(5000), 5), 5000);
        assert_eq!(retry_delay_ms(2000, Some(0), 5), 0);
    }

    const ENABLED: RetryPolicy = RetryPolicy {
        enabled: true,
        max_retries: 3,
        base_delay_ms: 0,
        max_agent_delay_ms: None,
    };

    #[derive(Default)]
    struct Recorder {
        scheduled: Mutex<Vec<(u32, u32, u64, String)>>,
        finished: Mutex<Vec<(bool, u32, Option<String>)>>,
        events: Arc<Mutex<Vec<String>>>,
    }

    fn callbacks(recorder: &Arc<Recorder>) -> RetryCallbacks {
        let scheduled = Arc::clone(recorder);
        let finished = Arc::clone(recorder);
        let started = Arc::clone(recorder);
        RetryCallbacks {
            on_retry_scheduled: Some(Arc::new(move |attempt, max, delay, error| {
                scheduled.events.lock().push(format!("retry:{attempt}"));
                scheduled
                    .scheduled
                    .lock()
                    .push((attempt, max, delay, error));
                Box::pin(async {})
            })),
            on_retry_attempt_start: Some(Arc::new(move || {
                started.events.lock().push("attempt-start".to_string());
                Box::pin(async {})
            })),
            on_retry_finished: Some(Arc::new(move |success, attempt, error| {
                finished.finished.lock().push((success, attempt, error));
                Box::pin(async {})
            })),
        }
    }

    #[tokio::test]
    async fn returns_a_successful_response_immediately_without_retrying() {
        let calls = AtomicU32::new(0);
        let response = retry_assistant_call(
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(message("ok", StopReason::Stop, None)) }
            },
            Some(&ENABLED),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(response.content, vec![AssistantContent::text("ok")]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn does_not_retry_an_aborted_message() {
        let recorder = Arc::new(Recorder::default());
        let calls = AtomicU32::new(0);
        let response = retry_assistant_call(
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(message("", StopReason::Aborted, None)) }
            },
            Some(&ENABLED),
            None,
            Some(&callbacks(&recorder)),
        )
        .await
        .unwrap();
        assert_eq!(response.stop_reason, StopReason::Aborted);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(recorder.scheduled.lock().is_empty());
    }

    #[tokio::test]
    async fn does_not_retry_a_non_retryable_error() {
        let recorder = Arc::new(Recorder::default());
        let calls = AtomicU32::new(0);
        let response = retry_assistant_call(
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(error("insufficient_quota")) }
            },
            Some(&ENABLED),
            None,
            Some(&callbacks(&recorder)),
        )
        .await
        .unwrap();
        assert_eq!(response.stop_reason, StopReason::Error);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(recorder.scheduled.lock().is_empty());
        assert!(recorder.finished.lock().is_empty());
    }

    #[tokio::test]
    async fn retries_a_transient_error_up_to_max_retries_then_returns_the_final_error() {
        let recorder = Arc::new(Recorder::default());
        let calls = AtomicU32::new(0);
        let response = retry_assistant_call(
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(error("terminated")) }
            },
            Some(&ENABLED),
            None,
            Some(&callbacks(&recorder)),
        )
        .await
        .unwrap();
        assert_eq!(response.stop_reason, StopReason::Error);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert_eq!(recorder.scheduled.lock().len(), 3);
        assert_eq!(
            *recorder.finished.lock(),
            vec![(false, 3, Some("terminated".to_string()))]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn reports_capped_retry_delays() {
        let recorder = Arc::new(Recorder::default());
        let calls = AtomicU32::new(0);
        let policy = RetryPolicy {
            enabled: true,
            max_retries: 4,
            base_delay_ms: 10,
            max_agent_delay_ms: Some(15),
        };
        retry_assistant_call(
            || {
                let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    Ok(if n < 5 {
                        error("terminated")
                    } else {
                        message("recovered", StopReason::Stop, None)
                    })
                }
            },
            Some(&policy),
            None,
            Some(&callbacks(&recorder)),
        )
        .await
        .unwrap();
        let delays: Vec<u64> = recorder
            .scheduled
            .lock()
            .iter()
            .map(|call| call.2)
            .collect();
        assert_eq!(delays, vec![10, 15, 15, 15]);
    }

    #[tokio::test]
    async fn stops_retrying_once_a_call_succeeds() {
        let recorder = Arc::new(Recorder::default());
        let calls = AtomicU32::new(0);
        let response = retry_assistant_call(
            || {
                let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    Ok(if n < 3 {
                        error("terminated")
                    } else {
                        message("recovered", StopReason::Stop, None)
                    })
                }
            },
            Some(&ENABLED),
            None,
            Some(&callbacks(&recorder)),
        )
        .await
        .unwrap();
        assert_eq!(response.content, vec![AssistantContent::text("recovered")]);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(*recorder.finished.lock(), vec![(true, 2, None)]);
    }

    #[tokio::test]
    async fn reports_an_aborted_retried_call_as_unsuccessful() {
        let recorder = Arc::new(Recorder::default());
        let calls = AtomicU32::new(0);
        let response = retry_assistant_call(
            || {
                let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    Ok(if n == 1 {
                        error("terminated")
                    } else {
                        message("", StopReason::Aborted, None)
                    })
                }
            },
            Some(&ENABLED),
            None,
            Some(&callbacks(&recorder)),
        )
        .await
        .unwrap();
        assert_eq!(response.stop_reason, StopReason::Aborted);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(*recorder.finished.lock(), vec![(false, 1, None)]);
    }

    #[tokio::test]
    async fn does_not_retry_when_policy_is_disabled() {
        let recorder = Arc::new(Recorder::default());
        let calls = AtomicU32::new(0);
        let disabled = RetryPolicy {
            enabled: false,
            ..ENABLED
        };
        let response = retry_assistant_call(
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(error("terminated")) }
            },
            Some(&disabled),
            None,
            Some(&callbacks(&recorder)),
        )
        .await
        .unwrap();
        assert_eq!(response.stop_reason, StopReason::Error);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(recorder.scheduled.lock().is_empty());
        assert!(recorder.finished.lock().is_empty());
    }

    #[tokio::test]
    async fn emits_on_retry_attempt_start_after_backoff_before_each_retried_call() {
        let recorder = Arc::new(Recorder::default());
        let events = Arc::clone(&recorder.events);
        let calls = AtomicU32::new(0);
        let response = retry_assistant_call(
            || {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                events.lock().push(format!("produce:{n}"));
                async move {
                    Ok(if n + 1 < 3 {
                        error("terminated")
                    } else {
                        message("recovered", StopReason::Stop, None)
                    })
                }
            },
            Some(&ENABLED),
            None,
            Some(&callbacks(&recorder)),
        )
        .await
        .unwrap();
        assert_eq!(response.content, vec![AssistantContent::text("recovered")]);
        assert_eq!(
            *recorder.events.lock(),
            vec![
                "produce:0",
                "retry:1",
                "attempt-start",
                "produce:1",
                "retry:2",
                "attempt-start",
                "produce:2",
            ]
        );
    }

    #[tokio::test]
    async fn aborts_backoff_sleep_via_signal_and_returns_an_aborted_message() {
        let recorder = Arc::new(Recorder::default());
        let controller = CancellationToken::new();
        let calls = Arc::new(AtomicU32::new(0));
        let policy = RetryPolicy {
            enabled: true,
            max_retries: 5,
            base_delay_ms: 10_000,
            max_agent_delay_ms: None,
        };
        let abort = controller.clone();
        let observed_calls = Arc::clone(&calls);
        tokio::spawn(async move {
            while observed_calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
            abort.cancel();
        });
        let response = retry_assistant_call(
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(error("terminated")) }
            },
            Some(&policy),
            Some(&controller),
            Some(&callbacks(&recorder)),
        )
        .await
        .unwrap();
        assert_eq!(response.stop_reason, StopReason::Aborted);
        assert_eq!(response.error_message, None);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            *recorder.finished.lock(),
            vec![(false, 1, Some("terminated".to_string()))]
        );
    }
}
