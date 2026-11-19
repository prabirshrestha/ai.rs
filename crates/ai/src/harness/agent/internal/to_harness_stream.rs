//! Port of `agent/internal/to-harness-stream.ts`.

use std::future::Future;
use std::sync::Arc;

use futures::stream::BoxStream;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use crate::harness::errors::HarnessResult;
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::spec::harness::{HarnessV1Emit, HarnessV1PromptControl};
use crate::harness::spec::stream_part::HarnessV1StreamPart;

/// What [`to_harness_stream`] returns.
pub struct HarnessStream {
    pub stream: BoxStream<'static, HarnessV1StreamPart>,
    /// The control surface `invoke` produced.
    pub control: Arc<dyn HarnessV1PromptControl>,
}

/// `toHarnessStream({ invoke })`: bridge an adapter's emit-based events
/// into a pull-based stream.
///
/// Calls `invoke(emit)` once. Every emitted part becomes a stream item. The
/// stream ends when `control.done()` resolves; when it rejects, an `error`
/// part is pushed first. Parts emitted after that are dropped.
///
/// [rust] `done` is driven on a spawned task, as a TS promise runs without
/// a consumer; this needs a Tokio runtime.
pub async fn to_harness_stream<F, Fut>(invoke: F) -> HarnessResult<HarnessStream>
where
    F: FnOnce(HarnessV1Emit) -> Fut,
    Fut: Future<Output = HarnessResult<Box<dyn HarnessV1PromptControl>>>,
{
    let (tx, rx) = mpsc::unbounded_channel::<HarnessV1StreamPart>();
    let sender = Arc::new(Mutex::new(Some(tx)));
    let emit_sender = sender.clone();
    let emit: HarnessV1Emit = Arc::new(move |part| {
        if let Some(tx) = &*emit_sender.lock() {
            let _ = tx.send(part);
        }
    });

    let control: Arc<dyn HarnessV1PromptControl> = Arc::from(invoke(emit).await?);

    let done = control.done();
    tokio::spawn(async move {
        let result = done.await;
        let tx = sender.lock().take();
        if let (Err(error), Some(tx)) = (result, tx) {
            let _ = tx.send(HarnessV1StreamPart::Error {
                error: ErrorValue::from(error),
            });
        }
    });

    let stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|part| (part, rx))
    });
    Ok(HarnessStream {
        stream: Box::pin(stream),
        control,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::spec::harness::HarnessV1SubmitToolResult;
    use async_trait::async_trait;
    use futures::StreamExt as _;
    use futures::future::{BoxFuture, FutureExt as _, Shared};
    use serde_json::json;
    use tokio::sync::oneshot;

    type Done = Shared<BoxFuture<'static, Result<(), String>>>;

    struct Control {
        done: Done,
        submitted: Arc<Mutex<Vec<HarnessV1SubmitToolResult>>>,
    }

    impl Control {
        fn boxed(done: BoxFuture<'static, Result<(), String>>) -> Box<dyn HarnessV1PromptControl> {
            Self::with_log(done, Arc::default())
        }

        fn with_log(
            done: BoxFuture<'static, Result<(), String>>,
            submitted: Arc<Mutex<Vec<HarnessV1SubmitToolResult>>>,
        ) -> Box<dyn HarnessV1PromptControl> {
            Box::new(Self {
                done: done.shared(),
                submitted,
            })
        }
    }

    #[async_trait]
    impl HarnessV1PromptControl for Control {
        async fn submit_tool_result(&self, input: HarnessV1SubmitToolResult) -> HarnessResult<()> {
            self.submitted.lock().push(input);
            Ok(())
        }

        fn done(&self) -> BoxFuture<'static, HarnessResult<()>> {
            let done = self.done.clone();
            Box::pin(async move { done.await.map_err(Into::into) })
        }
    }

    fn part(value: serde_json::Value) -> HarnessV1StreamPart {
        HarnessV1StreamPart::parse(value).unwrap()
    }

    #[tokio::test]
    async fn forwards_emitted_events_in_order_and_closes_when_done_resolves() {
        let HarnessStream { stream, .. } = to_harness_stream(|emit| async move {
            emit(part(json!({"type": "stream-start"})));
            emit(part(json!({"type": "text-start", "id": "t1"})));
            emit(part(
                json!({"type": "text-delta", "id": "t1", "delta": "hello"}),
            ));
            emit(part(json!({"type": "text-end", "id": "t1"})));
            emit(part(json!({
                "type": "finish",
                "finishReason": {"unified": "stop", "raw": "end_turn"},
                "totalUsage": {
                    "inputTokens": {"total": 10, "noCache": 10},
                    "outputTokens": {"total": 5, "text": 5}
                }
            })));
            Ok(Control::boxed(Box::pin(async { Ok(()) })))
        })
        .await
        .unwrap();
        let types: Vec<&str> = stream
            .collect::<Vec<_>>()
            .await
            .iter()
            .map(|p| p.type_name())
            .collect();
        assert_eq!(
            types,
            vec![
                "stream-start",
                "text-start",
                "text-delta",
                "text-end",
                "finish"
            ]
        );
    }

    #[tokio::test]
    async fn returns_the_same_control_handle_invoke_produced() {
        let submitted: Arc<Mutex<Vec<HarnessV1SubmitToolResult>>> = Arc::default();
        let log = submitted.clone();
        let HarnessStream { control, .. } =
            to_harness_stream(
                |_| async move { Ok(Control::with_log(Box::pin(async { Ok(()) }), log)) },
            )
            .await
            .unwrap();
        control
            .submit_tool_result(HarnessV1SubmitToolResult {
                tool_call_id: "c1".into(),
                output: json!("ok"),
                is_error: None,
                tool_result: None,
            })
            .await
            .unwrap();
        let submitted = submitted.lock();
        assert_eq!(submitted[0].tool_call_id, "c1");
        assert_eq!(submitted[0].output, json!("ok"));
    }

    #[tokio::test]
    async fn enqueues_an_error_part_and_closes_the_stream_when_done_rejects() {
        let (signal_failure, failure) = oneshot::channel::<String>();
        let HarnessStream { stream, .. } = to_harness_stream(|emit| async move {
            emit(part(
                json!({"type": "text-delta", "id": "t", "delta": "partial"}),
            ));
            Ok(Control::boxed(Box::pin(async move {
                Err(failure.await.unwrap())
            })))
        })
        .await
        .unwrap();
        signal_failure.send("boom".into()).unwrap();
        let parts: Vec<HarnessV1StreamPart> = stream.collect().await;
        assert_eq!(parts.len(), 2);
        assert!(
            matches!(&parts[0], HarnessV1StreamPart::TextDelta { delta, .. } if delta == "partial")
        );
        match &parts[1] {
            HarnessV1StreamPart::Error { error } => {
                assert_eq!(error.as_error().unwrap().to_string(), "boom")
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invokes_the_thunk_once_with_a_working_emit() {
        let calls = Arc::new(Mutex::new(0));
        let counter = calls.clone();
        to_harness_stream(move |emit| {
            *counter.lock() += 1;
            emit(part(json!({"type": "stream-start"})));
            async { Ok(Control::boxed(Box::pin(async { Ok(()) }))) }
        })
        .await
        .unwrap();
        assert_eq!(*calls.lock(), 1);
    }
}
