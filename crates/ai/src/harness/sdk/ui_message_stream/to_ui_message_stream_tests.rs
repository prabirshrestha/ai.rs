//! Port of `ai` `ui-message-stream/to-ui-message-stream.test.ts`.
//!
//! [rust] Not ported, because the Rust source stream and callbacks cannot
//! fail: 'reports an errored source stream as failed exactly once', 'does
//! not let an earlier finish part hide a source stream failure', 'reports
//! message metadata failures as failed exactly once', 'reports UI chunk
//! conversion failures as failed exactly once' and 'does not fail the
//! stream when the tool error cannot be serialized'. The reader-lock tests
//! have no counterpart (dropping the stream releases the source); the
//! cancellation one keeps its `onEnd` assertions. 'sends a … that converts
//! back to the tool result the model received' needs `readUIMessageStream`
//! and `convertToModelMessages`, which are not ported.

use std::sync::Arc;
use std::time::SystemTime;

use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::harness::provider::JsonValue;
use crate::harness::sdk::content::{FinishReason, TypedToolCall, TypedToolError};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::telemetry::callback;
use crate::harness::sdk::text_stream_part::{FinishStepResponse, TextStreamPart};
use crate::harness::sdk::ui_message_stream::ui_message_chunks::UIMessageChunk;
use crate::harness::sdk::ui_message_stream::ui_messages::UIMessage;

fn test_usage() -> crate::harness::sdk::usage::LanguageModelUsage {
    serde_json::from_value(json!({
        "inputTokens": 1,
        "outputTokens": 1,
        "totalTokens": 2,
        "inputTokenDetails": {},
        "outputTokenDetails": {},
    }))
    .unwrap()
}

fn test_step() -> TextStreamPart {
    TextStreamPart::FinishStep {
        response: FinishStepResponse {
            id: "r".into(),
            timestamp: SystemTime::UNIX_EPOCH,
            model_id: "m".into(),
            headers: None,
        },
        usage: test_usage(),
        performance: Default::default(),
        finish_reason: FinishReason::Stop,
        raw_finish_reason: Some("stop".into()),
        provider_metadata: None,
    }
}

fn start_step() -> TextStreamPart {
    TextStreamPart::StartStep {
        request: Default::default(),
        warnings: Vec::new(),
    }
}

fn finish() -> TextStreamPart {
    TextStreamPart::Finish {
        finish_reason: FinishReason::Stop,
        raw_finish_reason: Some("stop".into()),
        total_usage: test_usage(),
        provider_metadata: None,
    }
}

fn text(id: &str, delta: &str) -> Vec<TextStreamPart> {
    vec![
        TextStreamPart::TextStart {
            id: id.into(),
            provider_metadata: None,
        },
        TextStreamPart::TextDelta {
            id: id.into(),
            text: delta.into(),
            provider_metadata: None,
        },
        TextStreamPart::TextEnd {
            id: id.into(),
            citations: None,
            provider_metadata: None,
        },
    ]
}

fn reasoning() -> Vec<TextStreamPart> {
    vec![
        TextStreamPart::ReasoningStart {
            id: "r1".into(),
            provider_metadata: None,
        },
        TextStreamPart::ReasoningDelta {
            id: "r1".into(),
            text: "thinking".into(),
            provider_metadata: None,
        },
        TextStreamPart::ReasoningEnd {
            id: "r1".into(),
            provider_metadata: None,
        },
    ]
}

fn source(parts: Vec<TextStreamPart>) -> futures::stream::BoxStream<'static, TextStreamPart> {
    futures::stream::iter(parts).boxed()
}

async fn collect(parts: Vec<TextStreamPart>, options: UIMessageStreamOptions) -> Vec<JsonValue> {
    to_ui_message_stream(source(parts), None, options)
        .map(|chunk| chunk.unwrap().to_json())
        .collect()
        .await
}

type Recorded<E> = Arc<Mutex<Vec<E>>>;

fn recorder<E: Clone + Send + 'static>() -> (Callback<E>, Recorded<E>) {
    let calls: Recorded<E> = Arc::default();
    let log = calls.clone();
    (
        callback(move |event: E| {
            log.lock().push(event);
            async {}
        }),
        calls,
    )
}

fn user_message() -> UIMessage {
    serde_json::from_value(
        json!({ "id": "user-msg-1", "role": "user", "parts": [{ "type": "text", "text": "Hi" }] }),
    )
    .unwrap()
}

fn id(text: &'static str) -> Option<IdGeneratorFn> {
    Some(Arc::new(move || text.to_string()))
}

fn json_of<T: serde::Serialize>(value: &T) -> JsonValue {
    serde_json::to_value(value).unwrap()
}

#[tokio::test]
async fn includes_finish_step_metadata_in_the_snapshot_before_forwarding_the_boundary() {
    let (on_step_end, step_ends) = recorder::<UIMessageStreamOnStepEndEvent>();
    let (on_end, ends) = recorder::<UIMessageStreamOnEndEvent>();
    let chunks = collect(
        vec![TextStreamPart::Start, start_step(), test_step(), finish()],
        UIMessageStreamOptions {
            generate_message_id: id("assistant-1"),
            message_metadata: Some(Arc::new(|part| match part {
                TextStreamPart::FinishStep { usage, .. } => {
                    Some(json!({ "totalTokens": usage.total_tokens }))
                }
                _ => None,
            })),
            on_step_end: Some(on_step_end),
            on_end: Some(on_end),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        chunks,
        vec![
            json!({ "type": "start", "messageId": "assistant-1" }),
            json!({ "type": "start-step" }),
            json!({ "type": "message-metadata", "messageMetadata": { "totalTokens": 2 } }),
            json!({ "type": "finish-step" }),
            json!({ "type": "finish", "finishReason": "stop" }),
        ]
    );
    let step_ends = step_ends.lock();
    assert_eq!(step_ends.len(), 1);
    assert_eq!(
        step_ends[0].response_message.metadata,
        Some(json!({ "totalTokens": 2 }))
    );
    assert_eq!(
        step_ends[0].response_message,
        ends.lock()[0].response_message
    );
}

#[tokio::test]
async fn resolves_the_deprecated_step_callback_alias() {
    for has_on_step_end in [false, true] {
        let (on_step_end, step_ends) = recorder::<UIMessageStreamOnStepEndEvent>();
        let (on_step_finish, step_finishes) = recorder::<UIMessageStreamOnStepEndEvent>();
        collect(
            vec![TextStreamPart::Start, test_step()],
            UIMessageStreamOptions {
                on_step_end: has_on_step_end.then_some(on_step_end),
                on_step_finish: Some(on_step_finish),
                ..Default::default()
            },
        )
        .await;
        assert_eq!(step_ends.lock().len(), usize::from(has_on_step_end));
        assert_eq!(step_finishes.lock().len(), usize::from(!has_on_step_end));
    }
}

#[tokio::test]
async fn awaits_the_ui_callback_before_delivering_finish_step() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let (s, r) = (started.clone(), release.clone());
    let mut stream = to_ui_message_stream(
        source(vec![TextStreamPart::Start, test_step()]),
        None,
        UIMessageStreamOptions {
            on_step_end: Some(callback(move |_: UIMessageStreamOnStepEndEvent| {
                let (s, r) = (s.clone(), r.clone());
                async move {
                    s.notify_one();
                    r.notified().await;
                }
            })),
            ..Default::default()
        },
    );

    assert_eq!(
        stream.next().await.unwrap().unwrap().to_json(),
        json!({ "type": "start" })
    );
    let next = tokio::spawn(async move {
        let chunk = stream.next().await;
        (chunk, stream)
    });
    started.notified().await;
    tokio::task::yield_now().await;
    assert!(!next.is_finished());
    release.notify_one();
    let (chunk, mut stream) = next.await.unwrap();
    assert_eq!(chunk.unwrap().unwrap(), UIMessageChunk::FinishStep);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn maps_text_and_lifecycle_parts_to_ui_message_chunks() {
    let mut parts = vec![TextStreamPart::Start, start_step()];
    parts.push(TextStreamPart::TextStart {
        id: "t1".into(),
        provider_metadata: None,
    });
    for delta in ["Hello", ", world!"] {
        parts.push(TextStreamPart::TextDelta {
            id: "t1".into(),
            text: delta.into(),
            provider_metadata: None,
        });
    }
    parts.push(TextStreamPart::TextEnd {
        id: "t1".into(),
        citations: None,
        provider_metadata: None,
    });
    parts.extend([test_step(), finish()]);

    assert_eq!(
        collect(parts, Default::default()).await,
        vec![
            json!({ "type": "start" }),
            json!({ "type": "start-step" }),
            json!({ "type": "text-start", "id": "t1" }),
            json!({ "type": "text-delta", "id": "t1", "delta": "Hello" }),
            json!({ "type": "text-delta", "id": "t1", "delta": ", world!" }),
            json!({ "type": "text-end", "id": "t1" }),
            json!({ "type": "finish-step" }),
            json!({ "type": "finish", "finishReason": "stop" }),
        ]
    );
}

#[tokio::test]
async fn attaches_the_generated_message_id_to_the_start_chunk_when_provided() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = calls.clone();
    let chunks = collect(
        vec![TextStreamPart::Start],
        UIMessageStreamOptions {
            generate_message_id: Some(Arc::new(move || {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                "msg-123".into()
            })),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(
        chunks,
        vec![json!({ "type": "start", "messageId": "msg-123" })]
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn suppresses_start_and_finish_chunks_when_send_start_and_send_finish_are_false() {
    let parts = vec![
        TextStreamPart::Start,
        TextStreamPart::TextStart {
            id: "t1".into(),
            provider_metadata: None,
        },
        TextStreamPart::TextEnd {
            id: "t1".into(),
            citations: None,
            provider_metadata: None,
        },
        finish(),
    ];
    let chunks = collect(
        parts,
        UIMessageStreamOptions {
            send_start: Some(false),
            send_finish: Some(false),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(
        chunks,
        vec![
            json!({ "type": "text-start", "id": "t1" }),
            json!({ "type": "text-end", "id": "t1" }),
        ]
    );
}

#[tokio::test]
async fn skips_reasoning_parts_when_send_reasoning_is_false() {
    let chunks = collect(
        reasoning(),
        UIMessageStreamOptions {
            send_reasoning: Some(false),
            ..Default::default()
        },
    )
    .await;
    assert!(chunks.is_empty());
}

#[tokio::test]
async fn routes_error_parts_through_on_error() {
    let chunks = collect(
        vec![TextStreamPart::Error {
            error: ErrorValue::message("boom"),
        }],
        UIMessageStreamOptions {
            on_error: Some(Arc::new(|error| {
                format!("handled: {}", error.as_error().unwrap())
            })),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(
        chunks,
        vec![json!({ "type": "error", "errorText": "handled: boom" })]
    );
}

fn tool_call(input: &str) -> TypedToolCall {
    let mut call = TypedToolCall::new("call-1", "test-tool", json!({ "value": input }));
    call.dynamic = Some(true);
    call
}

fn tool_error(input: &str, error: ErrorValue) -> TypedToolError {
    let mut tool_error = TypedToolError::new(
        "call-1",
        "test-tool",
        Some(json!({ "value": input })),
        error,
    );
    tool_error.dynamic = Some(true);
    tool_error
}

fn recording_on_error(
    text: &'static str,
) -> (UIMessageStreamOnErrorFn, Arc<Mutex<Vec<ErrorValue>>>) {
    let calls: Arc<Mutex<Vec<ErrorValue>>> = Arc::default();
    let log = calls.clone();
    (
        Arc::new(move |error| {
            log.lock().push(error.clone());
            text.to_string()
        }),
        calls,
    )
}

#[tokio::test]
async fn routes_an_invalid_tool_call_through_on_error_once() {
    let invalid_error = ErrorValue::message("invalid tool input");
    let mut call = tool_call("invalid");
    call.invalid = Some(true);
    call.error = Some(invalid_error.clone());
    let (on_error, calls) = recording_on_error("formatted tool error");
    let chunks = collect(
        vec![
            TextStreamPart::ToolCall(call),
            TextStreamPart::ToolError(tool_error(
                "invalid",
                ErrorValue::Value(json!("Error: invalid tool input")),
            )),
        ],
        UIMessageStreamOptions {
            on_error: Some(on_error),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        chunks,
        vec![json!({
            "type": "tool-input-error",
            "toolCallId": "call-1",
            "toolName": "test-tool",
            "input": { "value": "invalid" },
            "dynamic": true,
            "errorText": "formatted tool error",
        })]
    );
    assert_eq!(*calls.lock(), vec![invalid_error]);
}

#[tokio::test]
async fn forwards_tool_errors_for_valid_tool_calls() {
    let (on_error, calls) = recording_on_error("formatted tool error");
    let chunks = collect(
        vec![
            TextStreamPart::ToolCall(tool_call("valid")),
            TextStreamPart::ToolError(tool_error("valid", ErrorValue::message("execution failed"))),
        ],
        UIMessageStreamOptions {
            on_error: Some(on_error),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        chunks,
        vec![
            json!({
                "type": "tool-input-available",
                "toolCallId": "call-1",
                "toolName": "test-tool",
                "input": { "value": "valid" },
                "dynamic": true,
            }),
            json!({
                "type": "tool-output-error",
                "toolCallId": "call-1",
                "dynamic": true,
                "errorText": "formatted tool error",
            }),
        ]
    );
    assert_eq!(calls.lock().len(), 1);
}

#[tokio::test]
async fn emits_separate_metadata_chunks_for_non_lifecycle_parts() {
    let chunks = collect(
        vec![
            TextStreamPart::Start,
            TextStreamPart::TextDelta {
                id: "t1".into(),
                text: "Hello".into(),
                provider_metadata: None,
            },
            finish(),
        ],
        UIMessageStreamOptions {
            message_metadata: Some(Arc::new(|part| {
                Some(json!({ "partType": part.type_name() }))
            })),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        chunks,
        vec![
            json!({ "type": "start", "messageMetadata": { "partType": "start" } }),
            json!({ "type": "text-delta", "id": "t1", "delta": "Hello" }),
            json!({ "type": "message-metadata", "messageMetadata": { "partType": "text-delta" } }),
            json!({ "type": "finish", "finishReason": "stop", "messageMetadata": { "partType": "finish" } }),
        ]
    );
}

fn hello_turn() -> Vec<TextStreamPart> {
    let mut parts = vec![TextStreamPart::Start];
    parts.extend(text("t1", "Hello"));
    parts.push(finish());
    parts
}

#[tokio::test]
async fn injects_generated_message_id_and_calls_on_finish() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = calls.clone();
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    let chunks = collect(
        hello_turn(),
        UIMessageStreamOptions {
            original_messages: Some(vec![user_message()]),
            generate_message_id: Some(Arc::new(move || {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                "msg-123".into()
            })),
            on_finish: Some(on_finish),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        chunks[0],
        json!({ "type": "start", "messageId": "msg-123" })
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    let finishes = finishes.lock();
    assert_eq!(finishes.len(), 1);
    let event = &finishes[0];
    assert!(!event.is_aborted);
    assert!(!event.is_continuation);
    assert_eq!(event.finish_reason, Some(FinishReason::Stop));
    let response = json!({ "id": "msg-123", "role": "assistant", "parts": [{ "type": "text", "text": "Hello" }] });
    assert!(crate::harness::agent::internal::run_prompt::tests::matches(
        &json_of(&event.response_message),
        &response
    ));
    assert!(crate::harness::agent::internal::run_prompt::tests::matches(
        &json_of(&event.messages),
        &json!([json_of(&user_message()), response])
    ));
}

#[tokio::test]
async fn calls_on_step_end_with_the_accumulated_ui_message() {
    let mut parts = vec![TextStreamPart::Start, start_step()];
    parts.extend(text("t1", "First"));
    parts.extend([test_step(), start_step()]);
    parts.extend(text("t2", "Second"));
    parts.extend([test_step(), finish()]);
    let (on_step_end, step_ends) = recorder::<UIMessageStreamOnStepEndEvent>();
    collect(
        parts,
        UIMessageStreamOptions {
            original_messages: Some(vec![user_message()]),
            generate_message_id: id("assistant-msg-1"),
            on_step_end: Some(on_step_end),
            ..Default::default()
        },
    )
    .await;

    let step_ends = step_ends.lock();
    assert_eq!(step_ends.len(), 2);
    assert!(!step_ends[0].is_continuation);
    let first = json!({
        "id": "assistant-msg-1",
        "role": "assistant",
        "parts": [{ "type": "step-start" }, { "type": "text", "text": "First" }],
    });
    let matches = crate::harness::agent::internal::run_prompt::tests::matches;
    assert!(matches(&json_of(&step_ends[0].response_message), &first));
    assert!(matches(
        &json_of(&step_ends[0].messages),
        &json!([json_of(&user_message()), first])
    ));
    assert!(matches(
        &json_of(&step_ends[1].response_message.parts),
        &json!([
            { "type": "step-start" },
            { "type": "text", "text": "First" },
            { "type": "step-start" },
            { "type": "text", "text": "Second" },
        ])
    ));
}

#[tokio::test]
async fn calls_on_end_when_stream_finishes() {
    let (on_end, ends) = recorder::<UIMessageStreamOnEndEvent>();
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    collect(
        hello_turn(),
        UIMessageStreamOptions {
            generate_message_id: id("msg-123"),
            on_end: Some(on_end),
            on_finish: Some(on_finish),
            ..Default::default()
        },
    )
    .await;

    let ends = ends.lock();
    assert_eq!(ends.len(), 1);
    assert!(!ends[0].is_aborted);
    assert!(!ends[0].is_continuation);
    assert_eq!(ends[0].finish_reason, Some(FinishReason::Stop));
    assert!(crate::harness::agent::internal::run_prompt::tests::matches(
        &json_of(&ends[0].response_message),
        &json!({ "id": "msg-123", "role": "assistant", "parts": [{ "type": "text", "text": "Hello" }] })
    ));
    assert!(finishes.lock().is_empty());
}

#[tokio::test]
async fn reports_the_source_outcome_to_on_end() {
    async fn observe(parts: Vec<TextStreamPart>) -> (String, Option<String>) {
        let (on_end, ends) = recorder::<UIMessageStreamOnEndEvent>();
        collect(
            parts,
            UIMessageStreamOptions {
                on_end: Some(on_end),
                ..Default::default()
            },
        )
        .await;
        let outcome = ends.lock()[0].outcome.clone();
        let message = match &outcome {
            UIMessageStreamOutcome::Failed { error: Some(error) } => {
                error.as_error().map(|e| e.to_string())
            }
            _ => None,
        };
        (outcome.status().to_string(), message)
    }
    let error = || TextStreamPart::Error {
        error: ErrorValue::message("generation failed"),
    };
    let abort = || TextStreamPart::Abort {
        reason: Some("user cancelled".into()),
    };

    assert_eq!(observe(vec![finish()]).await, ("completed".into(), None));
    assert_eq!(
        observe(vec![error()]).await,
        ("failed".into(), Some("generation failed".into()))
    );
    assert_eq!(observe(vec![abort()]).await, ("aborted".into(), None));
    assert_eq!(observe(vec![]).await, ("unknown".into(), None));
    assert_eq!(
        observe(vec![finish(), error()]).await,
        ("completed".into(), None)
    );
    assert_eq!(
        observe(vec![error(), finish()]).await,
        ("completed".into(), None)
    );
}

#[tokio::test]
async fn reports_a_consumer_cancellation_to_on_end() {
    let (on_end, ends) = recorder::<UIMessageStreamOnEndEvent>();
    // a source that never ends
    let source = futures::stream::iter([TextStreamPart::Start])
        .chain(futures::stream::pending())
        .boxed();
    let mut stream = to_ui_message_stream(
        source,
        None,
        UIMessageStreamOptions {
            on_end: Some(on_end),
            ..Default::default()
        },
    );
    stream.next().await;
    drop(stream);

    crate::harness::agent::internal::run_prompt::tests::wait_for(|| !ends.lock().is_empty()).await;
    let ends = ends.lock();
    assert_eq!(ends.len(), 1);
    assert!(!ends[0].is_aborted);
    assert_eq!(ends[0].is_cancelled, Some(true));
    assert_eq!(ends[0].outcome, UIMessageStreamOutcome::Unknown);
}

// ─── tool error text mismatch warning ──────────────────────────────────

fn tool_error_then_reasoning() -> Vec<TextStreamPart> {
    let mut parts = vec![
        start_step(),
        TextStreamPart::ToolCall(tool_call("valid")),
        TextStreamPart::ToolError(tool_error("valid", ErrorValue::message("execution failed"))),
        test_step(),
        start_step(),
    ];
    parts.extend(reasoning());
    parts.push(test_step());
    parts
}

/// Consume the stream and return the warnings it logged.
async fn warnings(parts: Vec<TextStreamPart>, options: UIMessageStreamOptions) -> Vec<String> {
    CAPTURED_WARNINGS.with(|captured| *captured.borrow_mut() = Some(Vec::new()));
    collect(parts, options).await;
    CAPTURED_WARNINGS.with(|captured| captured.borrow_mut().take().unwrap())
}

fn redacted() -> UIMessageStreamOptions {
    UIMessageStreamOptions {
        redact_tool_errors: Some(true),
        ..Default::default()
    }
}

fn message_on_error() -> Option<UIMessageStreamOnErrorFn> {
    Some(Arc::new(|error| error.as_error().unwrap().to_string()))
}

#[tokio::test]
async fn warns_when_a_masked_tool_error_is_followed_by_reasoning_from_a_later_step() {
    let warnings = warnings(tool_error_then_reasoning(), redacted()).await;
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains(
        "The error text sent to the UI for tool call \"call-1\" differs from the error text the model received"
    ));
}

#[tokio::test]
async fn warns_once_per_stream() {
    let mut parts = tool_error_then_reasoning();
    let mut call = tool_call("valid");
    call.tool_call_id = "call-2".into();
    let mut error = tool_error("valid", ErrorValue::message("execution failed"));
    error.tool_call_id = "call-2".into();
    parts.extend([
        start_step(),
        TextStreamPart::ToolCall(call),
        TextStreamPart::ToolError(error),
        test_step(),
        start_step(),
    ]);
    parts.extend(reasoning());
    parts.push(test_step());
    assert_eq!(warnings(parts, redacted()).await.len(), 1);
}

#[tokio::test]
async fn does_not_warn_by_default_because_the_ui_receives_the_text_the_model_received() {
    assert!(
        warnings(tool_error_then_reasoning(), Default::default())
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn does_not_warn_when_a_provided_on_error_does_not_redact_tool_errors() {
    let options = UIMessageStreamOptions {
        on_error: message_on_error(),
        redact_tool_errors: Some(false),
        ..Default::default()
    };
    assert!(
        warnings(tool_error_then_reasoning(), options)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn warns_when_on_error_returns_a_text_that_differs_from_the_text_the_model_received() {
    let options = UIMessageStreamOptions {
        on_error: message_on_error(),
        ..Default::default()
    };
    assert_eq!(
        warnings(tool_error_then_reasoning(), options).await.len(),
        1
    );
}

#[tokio::test]
async fn does_not_warn_when_on_error_returns_the_text_the_model_received() {
    let options = UIMessageStreamOptions {
        on_error: Some(Arc::new(|error| error.get_error_message())),
        ..Default::default()
    };
    assert!(
        warnings(tool_error_then_reasoning(), options)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn does_not_warn_when_reasoning_is_not_sent() {
    let options = UIMessageStreamOptions {
        send_reasoning: Some(false),
        ..Default::default()
    };
    assert!(
        warnings(tool_error_then_reasoning(), options)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn does_not_warn_when_the_reasoning_precedes_the_tool_error() {
    let mut parts = vec![start_step()];
    parts.extend(reasoning());
    parts.extend([
        TextStreamPart::ToolCall(tool_call("valid")),
        TextStreamPart::ToolError(tool_error("valid", ErrorValue::message("execution failed"))),
        test_step(),
        start_step(),
    ]);
    parts.extend(text("t1", "The tool failed."));
    parts.push(test_step());
    assert!(warnings(parts, Default::default()).await.is_empty());
}

#[tokio::test]
async fn does_not_warn_when_the_reasoning_follows_the_tool_error_in_the_same_step() {
    let mut parts = vec![
        start_step(),
        TextStreamPart::ToolCall(tool_call("valid")),
        TextStreamPart::ToolError(tool_error("valid", ErrorValue::message("execution failed"))),
    ];
    parts.extend(reasoning());
    parts.push(test_step());
    assert!(warnings(parts, Default::default()).await.is_empty());
}

#[tokio::test]
async fn warns_when_a_masked_tool_error_precedes_the_first_step() {
    // approved tool calls run before the first step of the next call
    let mut parts = vec![
        TextStreamPart::ToolError(tool_error("valid", ErrorValue::message("execution failed"))),
        start_step(),
    ];
    parts.extend(reasoning());
    parts.push(test_step());
    assert_eq!(warnings(parts, redacted()).await.len(), 1);
}

#[tokio::test]
async fn warns_when_a_masked_invalid_tool_call_is_followed_by_reasoning_from_a_later_step() {
    let mut call = tool_call("valid");
    call.invalid = Some(true);
    call.error = Some(ErrorValue::message("invalid tool input"));
    let mut parts = vec![
        start_step(),
        TextStreamPart::ToolCall(call),
        TextStreamPart::ToolError(tool_error(
            "valid",
            ErrorValue::Value(json!("Error: invalid tool input")),
        )),
        test_step(),
        start_step(),
    ];
    parts.extend(reasoning());
    parts.push(test_step());
    assert_eq!(warnings(parts, redacted()).await.len(), 1);
}

#[tokio::test]
async fn does_not_warn_for_provider_executed_tool_errors() {
    let mut call = tool_call("valid");
    call.provider_executed = Some(true);
    let mut error = tool_error("valid", ErrorValue::Value(json!("provider error")));
    error.provider_executed = Some(true);
    let mut parts = vec![
        start_step(),
        TextStreamPart::ToolCall(call),
        TextStreamPart::ToolError(error),
        test_step(),
        start_step(),
    ];
    parts.extend(reasoning());
    parts.push(test_step());
    assert!(warnings(parts, Default::default()).await.is_empty());
}

// ─── getResponseUIMessageId (get-response-ui-message-id.test.ts) ──────

fn ui(id: &str, role: &str) -> UIMessage {
    serde_json::from_value(json!({ "id": id, "role": role, "parts": [] })).unwrap()
}

fn new_id() -> IdGeneratorFn {
    Arc::new(|| "new-id".into())
}

#[test]
fn should_return_undefined_when_original_messages_is_null() {
    assert_eq!(get_response_ui_message_id(None, &new_id()), None);
}

#[test]
fn should_return_the_last_assistant_message_id_when_present() {
    let messages = [ui("msg-1", "user"), ui("msg-2", "assistant")];
    assert_eq!(
        get_response_ui_message_id(Some(&messages), &new_id()).as_deref(),
        Some("msg-2")
    );
}

#[test]
fn should_generate_new_id_when_last_message_is_not_from_assistant() {
    let messages = [ui("msg-1", "assistant"), ui("msg-2", "user")];
    assert_eq!(
        get_response_ui_message_id(Some(&messages), &new_id()).as_deref(),
        Some("new-id")
    );
}

#[test]
fn should_generate_new_id_when_messages_array_is_empty() {
    assert_eq!(
        get_response_ui_message_id(Some(&[]), &new_id()).as_deref(),
        Some("new-id")
    );
}

#[test]
fn should_use_the_response_message_id_when_it_is_a_string() {
    // [rust] `responseMessageId` is always a generator; a constant one
    // stands in for the string.
    let id: IdGeneratorFn = Arc::new(|| "response-id".into());
    assert_eq!(
        get_response_ui_message_id(Some(&[]), &id).as_deref(),
        Some("response-id")
    );
}
