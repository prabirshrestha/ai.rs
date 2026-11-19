//! Port of `ai` `ui-message-stream/handle-ui-message-stream-finish.test.ts`.
//!
//! [rust] Not ported: 'should handle onStepFinish errors by logging and
//! continuing' (Rust callbacks cannot fail). Cancelling is dropping the
//! stream, after which `onEnd` runs on a spawned task. Chunks are moved
//! values, so 'without mutating frozen start chunks' keeps only its
//! output and outcome checks, and each callback receives its own copy of
//! the message (the deep-clone test mutates that copy).

use std::sync::Arc;

use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::harness::agent::internal::run_prompt::tests::wait_for;
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::telemetry::callback;
use crate::harness::sdk::ui_message_stream::ui_message_chunks::UIMessageChunk;
use crate::harness::sdk::ui_message_stream::ui_messages::{UIMessage, UIMessagePart};

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

fn chunks(values: &[serde_json::Value]) -> Vec<UIMessageChunk> {
    values
        .iter()
        .map(|value| serde_json::from_value(value.clone()).unwrap())
        .collect()
}

fn source(chunks: Vec<UIMessageChunk>) -> BoxStream<'static, UIMessageChunk> {
    futures::stream::iter(chunks).boxed()
}

struct ErrorHandler(Arc<Mutex<Vec<ErrorValue>>>);

fn error_handler() -> (UIMessageStreamOnErrorFn, ErrorHandler) {
    let calls: Arc<Mutex<Vec<ErrorValue>>> = Arc::default();
    let log = calls.clone();
    (
        Arc::new(move |error| {
            log.lock().push(error.clone());
            String::new()
        }),
        ErrorHandler(calls),
    )
}

fn options(
    message_id: Option<&str>,
    original_messages: Vec<UIMessage>,
) -> (HandleUIMessageStreamFinishOptions, ErrorHandler) {
    let (on_error, errors) = error_handler();
    (
        HandleUIMessageStreamFinishOptions {
            message_id: message_id.map(str::to_string),
            original_messages,
            on_step_end: None,
            on_step_finish: None,
            on_end: None,
            on_finish: None,
            on_error,
            get_outcome: None,
        },
        errors,
    )
}

async fn run(
    chunks: Vec<UIMessageChunk>,
    options: HandleUIMessageStreamFinishOptions,
) -> Vec<UIMessageChunk> {
    handle_ui_message_stream_finish(source(chunks), options)
        .map(|chunk| chunk.unwrap())
        .collect()
        .await
}

fn message(id: &str, role: &str, text: Option<&str>) -> UIMessage {
    let parts = match text {
        Some(text) => json!([{ "type": "text", "text": text }]),
        None => json!([]),
    };
    serde_json::from_value(json!({ "id": id, "role": role, "parts": parts })).unwrap()
}

fn hello(id: &str) -> Vec<UIMessageChunk> {
    chunks(&[
        json!({ "type": "start", "messageId": id }),
        json!({ "type": "text-start", "id": "text-1" }),
        json!({ "type": "text-delta", "id": "text-1", "delta": "Hello" }),
        json!({ "type": "text-delta", "id": "text-1", "delta": " World" }),
        json!({ "type": "text-end", "id": "text-1" }),
        json!({ "type": "finish" }),
    ])
}

// ─── stream pass-through without onFinish ──────────────────────────────

#[tokio::test]
async fn should_pass_through_stream_chunks_without_processing_when_on_finish_is_not_provided() {
    let input = hello("msg-123");
    let (options, errors) = options(Some("msg-123"), vec![]);
    assert_eq!(run(input.clone(), options).await, input);
    assert!(errors.0.lock().is_empty());
}

#[tokio::test]
async fn should_inject_message_id_when_not_present_in_start_chunk() {
    let input = chunks(&[
        json!({ "type": "start" }),
        json!({ "type": "text-start", "id": "text-1" }),
        json!({ "type": "text-delta", "id": "text-1", "delta": "Test" }),
        json!({ "type": "text-end", "id": "text-1" }),
        json!({ "type": "finish" }),
    ]);
    let (options, _) = options(Some("injected-123"), vec![]);
    let result = run(input.clone(), options).await;
    assert_eq!(
        result[0].to_json(),
        json!({ "type": "start", "messageId": "injected-123" })
    );
    assert_eq!(result[1..], input[1..]);
}

// ─── stream processing with onFinish callback ──────────────────────────

#[tokio::test]
async fn should_process_stream_and_call_on_finish_with_correct_parameters() {
    let input = hello("msg-456");
    let original = vec![message("user-msg-1", "user", Some("Hello"))];
    let (mut options, errors) = options(Some("msg-456"), original.clone());
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_finish);

    assert_eq!(run(input.clone(), options).await, input);
    let finishes = finishes.lock();
    assert_eq!(finishes.len(), 1);
    let event = &finishes[0];
    assert!(!event.is_continuation);
    assert_eq!(event.response_message.id, "msg-456");
    assert_eq!(event.response_message.role, UIMessageRole::Assistant);
    assert_eq!(event.messages.len(), 2);
    assert_eq!(event.messages[0], original[0]);
    assert_eq!(event.messages[1], event.response_message);
    assert!(errors.0.lock().is_empty());
}

#[tokio::test]
async fn should_prefer_on_end_over_deprecated_on_finish() {
    let (mut options, _) = options(Some("msg-456"), vec![]);
    let (on_end, ends) = recorder::<UIMessageStreamOnEndEvent>();
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_end = Some(on_end);
    options.on_finish = Some(on_finish);
    run(hello("msg-456"), options).await;
    assert_eq!(ends.lock().len(), 1);
    assert!(finishes.lock().is_empty());
}

#[tokio::test]
async fn should_pass_the_stream_owner_outcome_to_on_end() {
    let error = ErrorValue::message("stream failed");
    let (mut options, _) = options(Some("msg-456"), vec![]);
    let (on_end, ends) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_end = Some(on_end);
    let declared = UIMessageStreamOutcome::Failed {
        error: Some(error.clone()),
    };
    let outcome = declared.clone();
    options.get_outcome = Some(Arc::new(move || Some(outcome.clone())));
    run(
        chunks(&[
            json!({ "type": "start", "messageId": "msg-456" }),
            json!({ "type": "error", "errorText": "masked error" }),
        ]),
        options,
    )
    .await;

    let ends = ends.lock();
    assert!(!ends[0].is_aborted);
    assert_eq!(ends[0].outcome, declared);
}

#[tokio::test]
async fn should_report_ui_message_processing_failures_before_and_after_a_declared_outcome() {
    for input in [
        chunks(&[json!({ "type": "text-delta", "id": "missing", "delta": "text" })]),
        chunks(&[
            json!({ "type": "finish" }),
            json!({ "type": "text-delta", "id": "missing", "delta": "text" }),
        ]),
    ] {
        let (mut options, _) = options(Some("msg-processing-error"), vec![]);
        let (on_end, ends) = recorder::<UIMessageStreamOnEndEvent>();
        options.on_end = Some(on_end);
        options.get_outcome = Some(Arc::new(|| Some(UIMessageStreamOutcome::Completed)));

        let results: Vec<_> = handle_ui_message_stream_finish(source(input), options)
            .collect()
            .await;
        let processing_error = results
            .into_iter()
            .find_map(Result::err)
            .expect("the stream errors");

        let ends = ends.lock();
        assert_eq!(ends.len(), 1);
        match &ends[0].outcome {
            UIMessageStreamOutcome::Failed { error: Some(error) } => {
                assert_eq!(
                    error.as_error().unwrap().to_string(),
                    processing_error.to_string()
                );
            }
            other => panic!("expected failed, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn should_inject_message_ids_into_start_chunks() {
    let (mut options, _) = options(Some("msg-injected"), vec![]);
    let (on_end, ends) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_end = Some(on_end);
    options.get_outcome = Some(Arc::new(|| Some(UIMessageStreamOutcome::Completed)));
    let result = run(chunks(&[json!({ "type": "start" })]), options).await;
    assert_eq!(
        result,
        chunks(&[json!({ "type": "start", "messageId": "msg-injected" })])
    );
    let ends = ends.lock();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].outcome, UIMessageStreamOutcome::Completed);
}

#[tokio::test]
async fn should_handle_empty_original_messages_array() {
    let (mut options, _) = options(Some("msg-789"), vec![]);
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_finish);
    run(hello("msg-789"), options).await;
    let finishes = finishes.lock();
    assert_eq!(finishes.len(), 1);
    assert!(!finishes[0].is_continuation);
    assert_eq!(
        finishes[0].messages,
        vec![finishes[0].response_message.clone()]
    );
}

// ─── continuation ──────────────────────────────────────────────────────

fn continued() -> Vec<UIMessageChunk> {
    chunks(&[
        json!({ "type": "start", "messageId": "assistant-msg-1" }),
        json!({ "type": "text-start", "id": "text-1" }),
        json!({ "type": "text-delta", "id": "text-1", "delta": " continued" }),
        json!({ "type": "text-end", "id": "text-1" }),
        json!({ "type": "finish-step" }),
        json!({ "type": "finish" }),
    ])
}

fn continuation_messages() -> Vec<UIMessage> {
    vec![
        message("user-msg-1", "user", Some("Continue this")),
        message("assistant-msg-1", "assistant", Some("This is")),
    ]
}

#[tokio::test]
async fn should_handle_continuation_when_last_message_is_assistant() {
    let original = continuation_messages();
    // this id is ignored since we are continuing
    let (mut options, _) = options(Some("msg-999"), original.clone());
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_finish);
    run(continued(), options).await;

    let finishes = finishes.lock();
    assert_eq!(finishes.len(), 1);
    let event = &finishes[0];
    assert!(event.is_continuation);
    assert_eq!(event.response_message.id, "assistant-msg-1");
    assert_eq!(event.messages.len(), 2);
    assert_eq!(event.messages[0], original[0]);
    assert_eq!(event.messages[1], event.response_message);
}

#[tokio::test]
async fn should_not_treat_user_message_as_continuation() {
    let original = vec![
        message("user-msg-1", "user", Some("Question")),
        message("user-msg-2", "user", Some("Another question")),
    ];
    let (mut options, _) = options(Some("msg-001"), original);
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_finish);
    run(hello("msg-001"), options).await;

    let finishes = finishes.lock();
    assert!(!finishes[0].is_continuation);
    assert_eq!(finishes[0].response_message.id, "msg-001");
    assert_eq!(finishes[0].messages.len(), 3);
}

// ─── abort scenarios ───────────────────────────────────────────────────

fn aborted(id: &str, abort: serde_json::Value) -> Vec<UIMessageChunk> {
    chunks(&[
        json!({ "type": "start", "messageId": id }),
        json!({ "type": "text-start", "id": "text-1" }),
        json!({ "type": "text-delta", "id": "text-1", "delta": "Starting text" }),
        abort,
        json!({ "type": "finish" }),
    ])
}

#[tokio::test]
async fn should_set_is_aborted_to_true_when_abort_chunk_is_encountered() {
    let input = aborted("msg-abort-1", json!({ "type": "abort" }));
    let (mut options, _) = options(
        Some("msg-abort-1"),
        vec![message("user-msg-1", "user", Some("Test request"))],
    );
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_finish);
    assert_eq!(run(input.clone(), options).await, input);

    let finishes = finishes.lock();
    assert_eq!(finishes.len(), 1);
    assert!(finishes[0].is_aborted);
    assert!(!finishes[0].is_continuation);
    assert_eq!(finishes[0].response_message.id, "msg-abort-1");
    assert_eq!(finishes[0].messages.len(), 2);
}

#[tokio::test]
async fn should_pass_through_abort_reason_when_provided() {
    let input = aborted(
        "msg-abort-reason",
        json!({ "type": "abort", "reason": "manual abort" }),
    );
    let (mut options, _) = options(Some("msg-abort-reason"), vec![]);
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_finish);
    assert_eq!(run(input.clone(), options).await, input);
    assert_eq!(finishes.lock().len(), 1);
}

#[tokio::test]
async fn should_set_is_aborted_to_false_when_no_abort_chunk_is_encountered() {
    let (mut options, _) = options(
        Some("msg-normal"),
        vec![message("user-msg-1", "user", Some("Test request"))],
    );
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_finish);
    run(hello("msg-normal"), options).await;

    let finishes = finishes.lock();
    assert!(!finishes[0].is_aborted);
    assert!(!finishes[0].is_continuation);
    assert_eq!(finishes[0].response_message.id, "msg-normal");
}

#[tokio::test]
async fn should_handle_abort_chunk_in_pass_through_mode_without_on_finish() {
    let input = aborted("msg-abort-passthrough", json!({ "type": "abort" }));
    let (options, errors) = options(Some("msg-abort-passthrough"), vec![]);
    assert_eq!(run(input.clone(), options).await, input);
    assert!(errors.0.lock().is_empty());
}

#[tokio::test]
async fn should_handle_multiple_abort_chunks_correctly() {
    let input = chunks(&[
        json!({ "type": "start", "messageId": "msg-multiple-abort" }),
        json!({ "type": "text-start", "id": "text-1" }),
        json!({ "type": "abort" }),
        json!({ "type": "text-delta", "id": "text-1", "delta": "Some text" }),
        json!({ "type": "abort" }),
        json!({ "type": "finish" }),
    ]);
    let (mut options, _) = options(Some("msg-multiple-abort"), vec![]);
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_finish);
    assert_eq!(run(input.clone(), options).await, input);
    let finishes = finishes.lock();
    assert_eq!(finishes.len(), 1);
    assert!(finishes[0].is_aborted);
}

/// Read `reads` chunks, then drop (cancel) the stream and return the
/// single `onEnd` event.
async fn cancel_after(
    chunks: Vec<UIMessageChunk>,
    reads: usize,
    get_outcome: Option<GetOutcomeFn>,
) -> UIMessageStreamOnEndEvent {
    let (mut options, _) = options(Some("msg-1"), vec![]);
    let (on_end, ends) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_end);
    options.get_outcome = get_outcome;
    // a source that stays open after its chunks
    let source = futures::stream::iter(chunks)
        .chain(futures::stream::pending())
        .boxed();
    let mut stream = handle_ui_message_stream_finish(source, options);
    for _ in 0..reads {
        stream.next().await.unwrap().unwrap();
    }
    drop(stream);
    wait_for(|| !ends.lock().is_empty()).await;
    tokio::task::yield_now().await;
    let ends = ends.lock();
    assert_eq!(ends.len(), 1);
    ends[0].clone()
}

#[tokio::test]
async fn should_report_consumer_cancellation_when_the_reader_is_cancelled_before_an_outcome_is_declared()
 {
    let input = chunks(&[
        json!({ "type": "start", "messageId": "msg-1" }),
        json!({ "type": "text-start", "id": "text-1" }),
        json!({ "type": "text-delta", "id": "text-1", "delta": "Hello" }),
    ]);
    let event = cancel_after(input, 1, None).await;
    assert!(!event.is_aborted);
    assert_eq!(event.is_cancelled, Some(true));
    assert_eq!(event.outcome, UIMessageStreamOutcome::Unknown);
    assert_eq!(event.response_message.id, "msg-1");
}

#[tokio::test]
async fn should_preserve_a_declared_outcome_when_the_reader_is_cancelled() {
    for declared in [
        UIMessageStreamOutcome::Completed,
        UIMessageStreamOutcome::Failed {
            error: Some(ErrorValue::message("stream failed")),
        },
        UIMessageStreamOutcome::Aborted,
    ] {
        let outcome = declared.clone();
        let event = cancel_after(
            chunks(&[json!({ "type": "start", "messageId": "msg-1" })]),
            1,
            Some(Arc::new(move || Some(outcome.clone()))),
        )
        .await;
        assert_eq!(event.outcome, declared);
        assert_eq!(event.is_cancelled, None);
    }
}

#[tokio::test]
async fn should_prefer_an_observed_abort_over_cancellation() {
    let event = cancel_after(
        chunks(&[
            json!({ "type": "start", "messageId": "msg-1" }),
            json!({ "type": "abort" }),
        ]),
        2,
        None,
    )
    .await;
    assert!(event.is_aborted);
    assert_eq!(event.outcome, UIMessageStreamOutcome::Aborted);
    assert_eq!(event.is_cancelled, None);
}

// ─── onStepFinish callback ─────────────────────────────────────────────

fn one_step(id: &str) -> Vec<UIMessageChunk> {
    chunks(&[
        json!({ "type": "start", "messageId": id }),
        json!({ "type": "text-start", "id": "text-1" }),
        json!({ "type": "text-delta", "id": "text-1", "delta": "Hello" }),
        json!({ "type": "text-end", "id": "text-1" }),
        json!({ "type": "finish-step" }),
        json!({ "type": "finish" }),
    ])
}

#[tokio::test]
async fn should_call_on_step_end_when_finish_step_chunk_is_encountered() {
    let (mut options, _) = options(None, vec![]);
    let (on_step_end, step_ends) = recorder::<UIMessageStreamOnStepEndEvent>();
    options.on_step_end = Some(on_step_end);
    run(one_step("msg-step-end"), options).await;
    let step_ends = step_ends.lock();
    assert_eq!(step_ends.len(), 1);
    assert_eq!(step_ends[0].response_message.id, "msg-step-end");
}

#[tokio::test]
async fn should_prefer_on_step_end_over_deprecated_on_step_finish() {
    let (mut options, _) = options(None, vec![]);
    let (on_step_end, step_ends) = recorder::<UIMessageStreamOnStepEndEvent>();
    let (on_step_finish, step_finishes) = recorder::<UIMessageStreamOnStepEndEvent>();
    options.on_step_end = Some(on_step_end);
    options.on_step_finish = Some(on_step_finish);
    run(
        chunks(&[
            json!({ "type": "start", "messageId": "msg-step-end" }),
            json!({ "type": "finish-step" }),
            json!({ "type": "finish" }),
        ]),
        options,
    )
    .await;
    assert_eq!(step_ends.lock().len(), 1);
    assert!(step_finishes.lock().is_empty());
}

#[tokio::test]
async fn should_call_on_step_finish_when_finish_step_chunk_is_encountered() {
    let input = one_step("msg-step-1");
    let original = vec![message("user-msg-1", "user", Some("Hello"))];
    let (mut options, _) = options(Some("msg-step-1"), original.clone());
    let (on_step_finish, step_finishes) = recorder::<UIMessageStreamOnStepEndEvent>();
    options.on_step_finish = Some(on_step_finish);
    assert_eq!(run(input.clone(), options).await, input);

    let step_finishes = step_finishes.lock();
    assert_eq!(step_finishes.len(), 1);
    let event = &step_finishes[0];
    assert!(!event.is_continuation);
    assert_eq!(event.response_message.id, "msg-step-1");
    assert_eq!(event.response_message.role, UIMessageRole::Assistant);
    assert_eq!(event.messages.len(), 2);
    assert_eq!(event.messages[0], original[0]);
    assert_eq!(event.messages[1].id, "msg-step-1");
}

#[tokio::test]
async fn should_call_on_step_finish_multiple_times_for_multiple_steps() {
    let mut values = vec![json!({ "type": "start", "messageId": "msg-multi-step" })];
    for step in 1..=3 {
        if step > 1 {
            values.push(json!({ "type": "start-step" }));
        }
        let id = format!("text-{step}");
        values.extend([
            json!({ "type": "text-start", "id": id }),
            json!({ "type": "text-delta", "id": id, "delta": format!("Step {step}") }),
            json!({ "type": "text-end", "id": id }),
            json!({ "type": "finish-step" }),
        ]);
    }
    values.push(json!({ "type": "finish" }));
    let (mut options, _) = options(Some("msg-multi-step"), vec![]);
    let (on_step_finish, step_finishes) = recorder::<UIMessageStreamOnStepEndEvent>();
    options.on_step_finish = Some(on_step_finish);
    run(chunks(&values), options).await;

    let lengths: Vec<_> = step_finishes
        .lock()
        .iter()
        .map(|event| event.response_message.parts.len())
        .collect();
    // step-starts + text parts
    assert_eq!(lengths, vec![1, 3, 5]);
}

#[tokio::test]
async fn should_call_both_on_step_finish_and_on_finish_when_both_are_provided() {
    let (mut options, _) = options(Some("msg-both"), vec![]);
    let (on_step_finish, step_finishes) = recorder::<UIMessageStreamOnStepEndEvent>();
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_step_finish = Some(on_step_finish);
    options.on_finish = Some(on_finish);
    run(one_step("msg-both"), options).await;
    assert_eq!(step_finishes.lock().len(), 1);
    assert_eq!(finishes.lock().len(), 1);
}

#[tokio::test]
async fn should_handle_continuation_scenario_with_on_step_finish() {
    let (mut options, _) = options(Some("msg-999"), continuation_messages());
    let (on_step_finish, step_finishes) = recorder::<UIMessageStreamOnStepEndEvent>();
    options.on_step_finish = Some(on_step_finish);
    run(continued(), options).await;

    let step_finishes = step_finishes.lock();
    assert_eq!(step_finishes.len(), 1);
    assert!(step_finishes[0].is_continuation);
    assert_eq!(step_finishes[0].response_message.id, "assistant-msg-1");
    assert_eq!(step_finishes[0].messages.len(), 2);
}

#[tokio::test]
async fn should_provide_deep_cloned_messages_in_on_step_finish_to_prevent_mutation() {
    let (mut options, _) = options(Some("msg-clone"), vec![]);
    let step_messages: Arc<Mutex<Vec<UIMessage>>> = Arc::default();
    let log = step_messages.clone();
    options.on_step_finish = Some(callback(move |mut event: UIMessageStreamOnStepEndEvent| {
        // mutate the message in the callback
        event.response_message.parts.push(
            serde_json::from_value::<UIMessagePart>(json!({ "type": "text", "text": "MUTATION!" }))
                .unwrap(),
        );
        log.lock().push(event.response_message);
        async {}
    }));
    let (on_finish, finishes) = recorder::<UIMessageStreamOnEndEvent>();
    options.on_finish = Some(on_finish);
    run(one_step("msg-clone"), options).await;

    // onStepFinish saw its mutation (original + mutation)
    assert_eq!(step_messages.lock()[0].parts.len(), 2);
    // onFinish does not see the mutation
    assert_eq!(finishes.lock()[0].response_message.parts.len(), 1);
}

#[tokio::test]
async fn should_not_process_stream_when_neither_on_finish_nor_on_step_finish_is_provided() {
    let input = one_step("msg-passthrough");
    let (options, errors) = options(Some("msg-passthrough"), vec![]);
    assert_eq!(run(input.clone(), options).await, input);
    assert!(errors.0.lock().is_empty());
}
