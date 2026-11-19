//! Port of `utils/bridge-ready.ts`: waiting for a spawned bridge to listen.

use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::harness::errors::{AbortError, BoxError, HarnessResult};
use crate::harness::sandbox::{
    SandboxProcess, SandboxProcessControl, SandboxReadTextFileOptions, SandboxSession,
    SandboxWriteTextFileOptions,
};
use crate::harness::spec::HarnessV1BridgeReady;
use crate::harness::utils::bridge_diagnostics::{LineDecoder, push_tail};
use crate::harness::utils::plain_error;

const STDOUT_TAIL_LIMIT: usize = 20;

/// `BridgeReadySource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BridgeReadySource {
    Stdout,
    Metadata,
}

/// `BridgeReadyErrorContext`.
///
/// [rust] `proc` is the process's wait and kill half; the waiter still holds
/// its streams.
#[derive(Clone)]
pub struct BridgeReadyErrorContext {
    pub proc: Arc<dyn SandboxProcessControl>,
    pub stdout_tail: Vec<String>,
}

/// `createTimeoutError` / `createExitError`.
pub type BridgeReadyErrorFactory =
    Box<dyn FnOnce(BridgeReadyErrorContext) -> BoxFuture<'static, BoxError> + Send>;

/// `WaitForBridgeReadyOptions`.
pub struct WaitForBridgeReadyOptions<'a> {
    /// The bridge process. Its stdout is read until ready; when readiness
    /// comes from elsewhere, stdout is cancelled (left empty).
    pub proc: &'a mut SandboxProcess,
    pub sandbox: Arc<dyn SandboxSession>,
    pub bridge_state_dir: String,
    pub bridge_type: String,
    pub timeout_ms: u64,
    pub abort_signal: Option<CancellationToken>,
    /// Defaults to 100.
    pub poll_interval_ms: Option<u64>,
    pub create_timeout_error: Option<BridgeReadyErrorFactory>,
    pub create_exit_error: Option<BridgeReadyErrorFactory>,
}

/// `WaitForBridgeReadyResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaitForBridgeReadyResult {
    pub port: u16,
    pub source: BridgeReadySource,
    pub stdout_tail: Vec<String>,
}

/// `markBridgeStarting(...)`: best-effort marker that keeps a stale
/// `bridge-meta.json` from reporting readiness.
pub async fn mark_bridge_starting(
    sandbox: &dyn SandboxSession,
    bridge_state_dir: &str,
    bridge_type: &str,
    abort_signal: Option<CancellationToken>,
) {
    // The bridge's own metadata writes are best-effort. This host-side marker
    // only prevents stale readiness fallback when possible; stdout remains
    // the primary readiness signal.
    let _ = sandbox
        .write_text_file(SandboxWriteTextFileOptions {
            path: bridge_meta_path(bridge_state_dir),
            content: serde_json::json!({ "type": bridge_type, "state": "starting" }).to_string(),
            abort_signal,
            encoding: None,
        })
        .await;
}

/// `waitForBridgeReady(options)`: resolves once the bridge prints its
/// `bridge-ready` line or writes `waiting` metadata, whichever comes first.
pub async fn wait_for_bridge_ready(
    options: WaitForBridgeReadyOptions<'_>,
) -> HarnessResult<WaitForBridgeReadyResult> {
    let WaitForBridgeReadyOptions {
        proc,
        sandbox,
        bridge_state_dir,
        bridge_type,
        timeout_ms,
        abort_signal,
        poll_interval_ms,
        create_timeout_error,
        create_exit_error,
    } = options;
    let poll_interval = Duration::from_millis(poll_interval_ms.unwrap_or(100));
    let control = proc.control.clone();
    let mut decoder = LineDecoder::default();
    let mut stdout_tail: Vec<String> = Vec::new();
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut pending_meta_read: Option<BoxFuture<'static, Option<u16>>> = None;
    let mut next_meta_read_at = Instant::now();

    enum Event {
        Stdout(Option<HarnessResult<bytes::Bytes>>),
        Metadata(Option<u16>),
        Tick,
    }

    loop {
        if abort_signal
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            cancel_stdout(proc);
            let _ = control.kill().await;
            break Err(AbortError.into());
        }

        let now = Instant::now();
        if now >= deadline {
            cancel_stdout(proc);
            let _ = control.kill().await;
            break Err(make_bridge_ready_error(
                create_timeout_error,
                "bridge did not become ready in time.",
                &control,
                &stdout_tail,
            )
            .await);
        }
        let remaining = deadline - now;

        let meta_read_delay = next_meta_read_at.saturating_duration_since(now);
        if pending_meta_read.is_none() && meta_read_delay.is_zero() {
            pending_meta_read = Some(
                read_bridge_meta_ready(
                    sandbox.clone(),
                    bridge_meta_path(&bridge_state_dir),
                    bridge_type.clone(),
                    abort_signal.clone(),
                )
                .boxed(),
            );
        }
        let tick = remaining.min(if pending_meta_read.is_none() {
            meta_read_delay
        } else {
            poll_interval
        });

        let event = {
            let meta = async {
                match pending_meta_read.as_mut() {
                    Some(read) => read.await,
                    None => futures::future::pending().await,
                }
            };
            let aborted = async {
                match &abort_signal {
                    Some(signal) => signal.cancelled().await,
                    None => futures::future::pending().await,
                }
            };
            tokio::select! {
                biased;
                read = proc.stdout.next() => Event::Stdout(read),
                port = meta => Event::Metadata(port),
                _ = tokio::time::sleep(tick) => Event::Tick,
                _ = aborted => Event::Tick,
            }
        };

        match event {
            Event::Tick => continue,
            Event::Metadata(port) => {
                pending_meta_read = None;
                if let Some(port) = port {
                    cancel_stdout(proc);
                    break Ok(WaitForBridgeReadyResult {
                        port,
                        source: BridgeReadySource::Metadata,
                        stdout_tail: stdout_tail.clone(),
                    });
                }
                next_meta_read_at = Instant::now() + poll_interval;
            }
            Event::Stdout(None) => {
                for line in decoder.flush() {
                    push_tail(&mut stdout_tail, line, STDOUT_TAIL_LIMIT);
                }
                break Err(make_bridge_ready_error(
                    create_exit_error,
                    "bridge exited before becoming ready.",
                    &control,
                    &stdout_tail,
                )
                .await);
            }
            Event::Stdout(Some(Err(error))) => break Err(error),
            Event::Stdout(Some(Ok(chunk))) => {
                for line in decoder.push(&chunk) {
                    push_tail(&mut stdout_tail, line.clone(), STDOUT_TAIL_LIMIT);
                    if let Ok(ready) = serde_json::from_str::<HarnessV1BridgeReady>(&line) {
                        return Ok(WaitForBridgeReadyResult {
                            port: ready.port,
                            source: BridgeReadySource::Stdout,
                            stdout_tail,
                        });
                    }
                }
            }
        }
    }
}

/// `reader.cancel()`.
fn cancel_stdout(proc: &mut SandboxProcess) {
    proc.stdout = futures::stream::empty().boxed();
}

async fn read_bridge_meta_ready(
    sandbox: Arc<dyn SandboxSession>,
    path: String,
    bridge_type: String,
    abort_signal: Option<CancellationToken>,
) -> Option<u16> {
    let raw = sandbox
        .read_text_file(SandboxReadTextFileOptions {
            path,
            abort_signal,
            encoding: None,
            start_line: None,
            end_line: None,
        })
        .await
        .ok()??;
    let Value::Object(meta) = serde_json::from_str::<Value>(&raw).ok()? else {
        return None;
    };
    // `bridgeMetaSchema`: every field is optional but must have its type.
    let field = |name: &str| meta.get(name);
    let string_field = |name: &str| match field(name) {
        None => Some(None),
        Some(Value::String(value)) => Some(Some(value.as_str())),
        Some(_) => None,
    };
    let number_field = |name: &str| match field(name) {
        None => Some(None),
        Some(Value::Number(value)) => Some(value.as_f64()),
        Some(_) => None,
    };
    let r#type = string_field("type")?;
    let state = string_field("state")?;
    let port = number_field("port")?;
    number_field("pid")?;
    if r#type != Some(bridge_type.as_str()) || state != Some("waiting") {
        return None;
    }
    let port = port.filter(|port| *port > 0.0)?;
    (port.fract() == 0.0 && port <= f64::from(u16::MAX)).then_some(port as u16)
}

async fn make_bridge_ready_error(
    create_error: Option<BridgeReadyErrorFactory>,
    fallback_message: &str,
    proc: &Arc<dyn SandboxProcessControl>,
    stdout_tail: &[String],
) -> BoxError {
    match create_error {
        Some(create_error) => {
            create_error(BridgeReadyErrorContext {
                proc: proc.clone(),
                stdout_tail: stdout_tail.to_vec(),
            })
            .await
        }
        None => plain_error(fallback_message),
    }
}

fn bridge_meta_path(bridge_state_dir: &str) -> String {
    format!("{bridge_state_dir}/bridge-meta.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sandbox::SandboxRunResult;
    use crate::harness::test_support::{MockSandbox, mock_process};
    use parking_lot::Mutex;
    use serde_json::json;
    use std::collections::VecDeque;

    fn sandbox() -> MockSandbox {
        MockSandbox::with_result(SandboxRunResult::default()).with_read_text(|_| Ok(None))
    }

    fn options<'a>(
        proc: &'a mut SandboxProcess,
        sandbox: MockSandbox,
        poll_interval_ms: Option<u64>,
    ) -> WaitForBridgeReadyOptions<'a> {
        WaitForBridgeReadyOptions {
            proc,
            sandbox: Arc::new(sandbox),
            bridge_state_dir: "/state".into(),
            bridge_type: "claude-code".into(),
            timeout_ms: 1_000,
            abort_signal: None,
            poll_interval_ms,
            create_timeout_error: None,
            create_exit_error: None,
        }
    }

    fn reads(values: Vec<Value>) -> MockSandbox {
        let reads = Mutex::new(
            values
                .into_iter()
                .map(|v| v.to_string())
                .collect::<VecDeque<_>>(),
        );
        MockSandbox::with_result(SandboxRunResult::default())
            .with_read_text(move |_| Ok(reads.lock().pop_front()))
    }

    #[tokio::test]
    async fn resolves_from_stdout_bridge_ready() {
        let (mut proc, _) = mock_process(Some(vec!["{\"type\":\"bridge-ready\",\"port\":4319}\n"]));
        let result = wait_for_bridge_ready(options(&mut proc, sandbox(), None))
            .await
            .unwrap();
        assert_eq!(
            (result.port, result.source),
            (4319, BridgeReadySource::Stdout)
        );
    }

    #[tokio::test]
    async fn resolves_from_bridge_metadata_when_stdout_does_not_deliver() {
        let (mut proc, _) = mock_process(None);
        let sandbox = MockSandbox::with_result(SandboxRunResult::default()).with_read_text(|_| {
            Ok(Some(
                json!({ "type": "claude-code", "port": 4319, "state": "waiting", "pid": 123 })
                    .to_string(),
            ))
        });
        let result = wait_for_bridge_ready(options(&mut proc, sandbox, Some(1)))
            .await
            .unwrap();
        assert_eq!(
            (result.port, result.source),
            (4319, BridgeReadySource::Metadata)
        );
    }

    #[tokio::test]
    async fn ignores_non_waiting_metadata() {
        let (mut proc, _) = mock_process(None);
        let sandbox = reads(vec![
            json!({ "type": "claude-code", "state": "starting" }),
            json!({ "type": "claude-code", "port": 4319, "state": "waiting" }),
        ]);
        let text_reads = sandbox.text_reads.clone();
        let result = wait_for_bridge_ready(options(&mut proc, sandbox, Some(1)))
            .await
            .unwrap();
        assert_eq!(
            (result.port, result.source),
            (4319, BridgeReadySource::Metadata)
        );
        assert_eq!(text_reads.lock().len(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn waits_for_the_poll_interval_between_metadata_reads() {
        let (mut proc, _) = mock_process(None);
        let sandbox = reads(vec![
            json!({ "type": "claude-code", "state": "starting" }),
            json!({ "type": "claude-code", "port": 4319, "state": "waiting" }),
        ]);
        let text_reads = sandbox.text_reads.clone();
        let result = tokio::spawn(async move {
            wait_for_bridge_ready(options(&mut proc, sandbox, Some(50)))
                .await
                .unwrap()
        });

        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        assert_eq!(text_reads.lock().len(), 1);

        tokio::time::sleep(Duration::from_millis(49)).await;
        assert_eq!(text_reads.lock().len(), 1);

        tokio::time::sleep(Duration::from_millis(1)).await;
        let result = result.await.unwrap();
        assert_eq!(
            (result.port, result.source),
            (4319, BridgeReadySource::Metadata)
        );
        assert_eq!(text_reads.lock().len(), 2);
    }

    #[tokio::test]
    async fn marks_bridge_startup_metadata() {
        let sandbox = sandbox();
        mark_bridge_starting(&sandbox, "/state", "codex", None).await;
        assert_eq!(
            sandbox.state.lock().writes,
            vec![(
                "/state/bridge-meta.json".to_string(),
                json!({ "type": "codex", "state": "starting" }).to_string()
            )]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn times_out_and_kills_the_process() {
        let (mut proc, control) = mock_process(None);
        let error = wait_for_bridge_ready(WaitForBridgeReadyOptions {
            timeout_ms: 10,
            ..options(&mut proc, sandbox(), Some(1))
        })
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "bridge did not become ready in time.");
        assert_eq!(control.kills.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fails_when_the_bridge_exits_before_ready() {
        let (mut proc, _) = mock_process(Some(vec!["booting\n", "bye"]));
        let error = wait_for_bridge_ready(WaitForBridgeReadyOptions {
            create_exit_error: Some(Box::new(|context| {
                async move { plain_error(context.stdout_tail.join("|")) }.boxed()
            })),
            ..options(&mut proc, sandbox(), None)
        })
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "booting|bye");
    }
}
