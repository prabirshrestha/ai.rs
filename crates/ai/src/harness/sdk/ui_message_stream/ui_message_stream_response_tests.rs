//! Ports of `ai` `ui-message-stream/create-ui-message-stream-response.test.ts`
//! and `create-sse-stream-with-keep-alive.test.ts`.
//!
//! [rust] Not ported: 'should retain one pending source read across many
//! idle keep-alives' (a Rust stream has no separate pending read to leak;
//! `next()` is polled in place), 'should handle synchronous
//! consumeSseStream' (`ConsumeSseStreamFn` is always async) and the
//! `toBeInstanceOf(Response)` checks. Cancelling is dropping the stream,
//! which drops the source; there is no cancel reason.

use std::sync::Arc;
use std::time::Duration;

use futures::{FutureExt, StreamExt};
use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::harness::sdk::text_stream_part::TextStreamPart;
use crate::harness::sdk::ui_message_stream::to_ui_message_stream::{
    UIMessageStreamOptions, to_ui_message_stream,
};
use crate::harness::sdk::ui_message_stream::ui_message_chunks::UIMessageChunk;

fn chunks(values: Vec<serde_json::Value>) -> UIMessageStream {
    let chunks: Vec<_> = values
        .into_iter()
        .map(|value| Ok(serde_json::from_value::<UIMessageChunk>(value).unwrap()))
        .collect();
    futures::stream::iter(chunks).boxed()
}

async fn body(response: UIMessageStreamResponse) -> Vec<String> {
    response
        .body
        .map(|bytes| String::from_utf8(bytes.unwrap().to_vec()).unwrap())
        .collect()
        .await
}

fn strings(values: &[&str]) -> SseStream {
    let values: Vec<_> = values.iter().map(|v| Ok(v.to_string())).collect();
    futures::stream::iter(values).boxed()
}

// ─── createUIMessageStreamResponse ─────────────────────────────────────

#[tokio::test]
async fn should_create_a_response_with_correct_headers_and_encoded_stream() {
    let response = create_ui_message_stream_response(
        chunks(vec![
            json!({ "type": "text-delta", "id": "1", "delta": "test-data" }),
        ]),
        UIMessageStreamResponseInit {
            status: Some(200),
            status_text: Some("OK".into()),
            headers: Some([("Custom-Header".to_string(), "test".to_string())].into()),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(response.status, 200);
    assert_eq!(response.status_text.as_deref(), Some("OK"));
    let mut headers: Vec<_> = response
        .headers
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    headers.sort();
    assert_eq!(
        headers,
        vec![
            ("cache-control", "no-cache"),
            ("connection", "keep-alive"),
            ("content-type", "text/event-stream"),
            ("custom-header", "test"),
            ("x-accel-buffering", "no"),
            ("x-vercel-ai-ui-message-stream", "v1"),
        ]
    );
    assert_eq!(
        body(response).await,
        vec![
            "data: {\"type\":\"text-delta\",\"id\":\"1\",\"delta\":\"test-data\"}\n\n",
            "data: [DONE]\n\n",
        ]
    );
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn should_send_opening_and_keep_alive_comments_for_an_idle_stream() {
    let mut response = create_ui_message_stream_response(
        futures::stream::pending().boxed(),
        UIMessageStreamResponseInit {
            keep_alive_ms: Some(100),
            ..Default::default()
        },
    )
    .unwrap();

    let open = response.body.next().await.unwrap().unwrap();
    assert_eq!(&open[..], b": stream-open\n\n");
    let start = tokio::time::Instant::now();
    let keep_alive = response.body.next().await.unwrap().unwrap();
    assert_eq!(&keep_alive[..], b": keep-alive\n\n");
    assert_eq!(start.elapsed(), Duration::from_millis(100));
}

#[tokio::test]
async fn can_respond_with_a_stream_created_by_to_ui_message_stream() {
    let parts = vec![
        TextStreamPart::Start,
        TextStreamPart::TextStart {
            id: "t1".into(),
            provider_metadata: None,
        },
        TextStreamPart::TextDelta {
            id: "t1".into(),
            text: "Hello".into(),
            provider_metadata: None,
        },
        TextStreamPart::TextEnd {
            id: "t1".into(),
            citations: None,
            provider_metadata: None,
        },
    ];
    let stream = to_ui_message_stream(
        futures::stream::iter(parts).boxed(),
        None,
        UIMessageStreamOptions {
            generate_message_id: Some(Arc::new(|| "msg-123".into())),
            ..Default::default()
        },
    );
    let response = create_ui_message_stream_response(
        stream,
        UIMessageStreamResponseInit {
            status: Some(200),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(
        body(response).await,
        vec![
            "data: {\"type\":\"start\",\"messageId\":\"msg-123\"}\n\n",
            "data: {\"type\":\"text-start\",\"id\":\"t1\"}\n\n",
            "data: {\"type\":\"text-delta\",\"id\":\"t1\",\"delta\":\"Hello\"}\n\n",
            "data: {\"type\":\"text-end\",\"id\":\"t1\"}\n\n",
            "data: [DONE]\n\n",
        ]
    );
}

#[tokio::test]
async fn should_handle_errors_in_the_stream() {
    let response = create_ui_message_stream_response(
        chunks(vec![
            json!({ "type": "error", "errorText": "Custom error message" }),
        ]),
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        body(response).await,
        vec![
            "data: {\"type\":\"error\",\"errorText\":\"Custom error message\"}\n\n",
            "data: [DONE]\n\n",
        ]
    );
}

type Consumed = Arc<Mutex<Vec<String>>>;

fn consumer(
    consumed: Consumed,
    calls: Arc<std::sync::atomic::AtomicUsize>,
    after: impl Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync + 'static,
) -> ConsumeSseStreamFn {
    let after = Arc::new(after);
    Arc::new(move |stream: SseStream| {
        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (consumed, after) = (consumed.clone(), after.clone());
        async move {
            let data: Vec<String> = stream.map(|item| item.unwrap()).collect().await;
            consumed.lock().extend(data);
            after().await;
        }
        .boxed()
    })
}

const TWO_DELTAS: [&str; 3] = [
    "data: {\"type\":\"text-delta\",\"id\":\"1\",\"delta\":\"test-data-1\"}\n\n",
    "data: {\"type\":\"text-delta\",\"id\":\"1\",\"delta\":\"test-data-2\"}\n\n",
    "data: [DONE]\n\n",
];

#[tokio::test]
async fn should_call_consume_sse_stream_with_a_teed_stream() {
    let consumed = Consumed::default();
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let response = create_ui_message_stream_response(
        chunks(vec![
            json!({ "type": "text-delta", "id": "1", "delta": "test-data-1" }),
            json!({ "type": "text-delta", "id": "1", "delta": "test-data-2" }),
        ]),
        UIMessageStreamResponseInit {
            status: Some(200),
            consume_sse_stream: Some(consumer(consumed.clone(), calls.clone(), || {
                async {}.boxed()
            })),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(body(response).await, TWO_DELTAS);
    crate::harness::agent::internal::run_prompt::tests::wait_for(|| consumed.lock().len() == 3)
        .await;
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(*consumed.lock(), TWO_DELTAS);
}

#[tokio::test]
async fn should_not_block_the_response_when_consume_sse_stream_takes_time() {
    let consumed = Consumed::default();
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let response = create_ui_message_stream_response(
        chunks(vec![
            json!({ "type": "text-delta", "id": "1", "delta": "test-data" }),
        ]),
        UIMessageStreamResponseInit {
            status: Some(200),
            consume_sse_stream: Some(consumer(consumed, calls.clone(), || {
                futures::future::pending().boxed()
            })),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(response.status, 200);
    assert_eq!(
        body(response).await,
        vec![
            "data: {\"type\":\"text-delta\",\"id\":\"1\",\"delta\":\"test-data\"}\n\n",
            "data: [DONE]\n\n",
        ]
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn should_handle_consume_sse_stream_errors_gracefully() {
    // [rust] a failing consumer is a panicking task.
    let response = create_ui_message_stream_response(
        chunks(vec![
            json!({ "type": "text-delta", "id": "1", "delta": "error-test" }),
        ]),
        UIMessageStreamResponseInit {
            consume_sse_stream: Some(Arc::new(|_| {
                async { panic!("consumeSseStream error") }.boxed()
            })),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        body(response).await,
        vec![
            "data: {\"type\":\"text-delta\",\"id\":\"1\",\"delta\":\"error-test\"}\n\n",
            "data: [DONE]\n\n",
        ]
    );
}

// ─── createSseStreamWithKeepAlive ──────────────────────────────────────

#[tokio::test]
async fn should_pass_through_the_original_stream_when_keep_alive_ms_is_undefined() {
    let stream = create_sse_stream_with_keep_alive(strings(&["data"]), None).unwrap();
    let items: Vec<_> = stream.map(|item| item.unwrap()).collect().await;
    assert_eq!(items, vec!["data"]);
}

#[tokio::test]
async fn rejects_an_invalid_keep_alive() {
    // [rust] the TS checks `Number.isFinite` and the timer limit as well.
    for ms in [0, 2_147_483_648] {
        assert_eq!(
            create_sse_stream_with_keep_alive(strings(&[]), Some(ms)).err(),
            Some(InvalidKeepAliveError)
        );
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn should_send_an_opening_comment_immediately_and_comments_while_the_source_is_idle() {
    let mut stream =
        create_sse_stream_with_keep_alive(futures::stream::pending().boxed(), Some(100)).unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap(), ": stream-open\n\n");
    let start = tokio::time::Instant::now();
    assert_eq!(stream.next().await.unwrap().unwrap(), ": keep-alive\n\n");
    assert_eq!(start.elapsed(), Duration::from_millis(100));
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn should_reset_the_keep_alive_timer_after_source_activity() {
    let (tx, rx) = futures::channel::mpsc::unbounded::<Result<String, UIMessageStreamError>>();
    let mut stream = create_sse_stream_with_keep_alive(rx.boxed(), Some(100)).unwrap();
    stream.next().await;

    let sender = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        tx.unbounded_send(Ok("data".into())).unwrap();
        tx
    });
    assert_eq!(stream.next().await.unwrap().unwrap(), "data");
    let _tx = sender.await.unwrap();

    let start = tokio::time::Instant::now();
    assert_eq!(stream.next().await.unwrap().unwrap(), ": keep-alive\n\n");
    // the timer restarted at the source chunk, not at 100ms
    assert_eq!(start.elapsed(), Duration::from_millis(100));
}

#[tokio::test]
async fn should_preserve_source_chunks_and_completion() {
    let stream =
        create_sse_stream_with_keep_alive(strings(&["data 1", "data 2"]), Some(100)).unwrap();
    let items: Vec<_> = stream.map(|item| item.unwrap()).collect().await;
    assert_eq!(items, vec![": stream-open\n\n", "data 1", "data 2"]);
}

#[tokio::test]
async fn should_cancel_the_source_stream() {
    struct Source(Arc<std::sync::atomic::AtomicBool>);
    impl Drop for Source {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let guard = Source(cancelled.clone());
    let source = futures::stream::pending::<Result<String, UIMessageStreamError>>()
        .map(move |item| {
            let _ = &guard;
            item
        })
        .boxed();
    let mut stream = create_sse_stream_with_keep_alive(source, Some(100)).unwrap();
    stream.next().await;
    drop(stream);
    assert!(cancelled.load(std::sync::atomic::Ordering::SeqCst));
}
