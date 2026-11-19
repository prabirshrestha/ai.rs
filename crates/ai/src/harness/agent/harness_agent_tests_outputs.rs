//! `harness-agent.test.ts`: typed output, streamed parts, reasoning, turn
//! release after failures, and host results across suspension.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::harness::agent::internal::run_prompt::tests::tool_outputs::{
    approval_response, custom_approval, submitted_json,
};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::output::{array, object};
use crate::harness::sdk::tool::Tool;

fn answer_schema() -> crate::harness::sdk::tool::Schema {
    z_object(&[("answer", "string")])
}

fn structured_turn(delta: &str) -> Vec<HarnessV1StreamPart> {
    let mut values = vec![
        json!({ "type": "text-start", "id": "structured" }),
        json!({ "type": "text-delta", "id": "structured", "delta": delta }),
        json!({ "type": "text-end", "id": "structured" }),
    ];
    values.extend(stop_finish());
    parts(values)
}

async fn part_types(
    result: &crate::harness::agent::internal::harness_stream_text_result::HarnessStreamTextResult,
) -> Vec<String> {
    result
        .full_stream()
        .map(|part| part.to_json()["type"].as_str().unwrap().to_string())
        .collect()
        .await
}

fn count_of(types: &[String], ty: &str) -> usize {
    types.iter().filter(|t| *t == ty).count()
}

#[test]
fn reports_whether_typed_output_is_configured() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let text_agent = agent(settings(&mock));
    let mut s = settings(&mock);
    s.output = Some(object(answer_schema(), None, None));
    let output_agent = agent(s);

    assert!(!text_agent.has_output());
    assert!(output_agent.has_output());
}

#[tokio::test]
async fn generates_typed_output_and_sends_its_response_format_on_every_turn() {
    let mock = MockHarnessOptions::script(|| structured_turn(r#"{"answer":"yes"}"#)).build();
    let mut s = settings(&mock);
    s.output = Some(object(
        answer_schema(),
        Some("answer".into()),
        Some("A yes or no answer.".into()),
    ));
    let agent = agent(s);
    let session = new_session(&agent).await;

    let first = agent.generate(call(&session, "answer")).await.unwrap();
    let second = agent
        .generate(call(&session, "answer again"))
        .await
        .unwrap();

    assert_eq!(first.output, Some(json!({ "answer": "yes" })));
    assert_eq!(second.output, Some(json!({ "answer": "yes" })));
    let response_formats: Vec<JsonValue> = mock
        .record
        .prompt_turns
        .lock()
        .iter()
        .map(|turn| to_json(&turn.response_format))
        .collect();
    assert_eq!(response_formats.len(), 2);
    for response_format in response_formats {
        assert!(
            matches(
                &response_format,
                &json!({
                    "type": "json",
                    "name": "answer",
                    "description": "A yes or no answer.",
                })
            ),
            "{response_format}"
        );
        assert_eq!(response_format["schema"]["type"], "object");
        assert_eq!(
            response_format["schema"]["properties"],
            json!({ "answer": { "type": "string" } })
        );
        assert_eq!(response_format["schema"]["required"], json!(["answer"]));
    }

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn streams_partial_typed_output() {
    let mock = MockHarnessOptions::script(|| structured_turn(r#"{"answer":"yes"}"#)).build();
    let mut s = settings(&mock);
    s.output = Some(object(answer_schema(), None, None));
    let agent = agent(s);
    let session = new_session(&agent).await;

    let result = agent.stream(call(&session, "answer")).await.unwrap();
    let partial_outputs: Vec<JsonValue> = result.partial_output_stream().collect().await;

    assert_eq!(partial_outputs, vec![json!({ "answer": "yes" })]);
    assert_eq!(result.output().await.unwrap(), json!({ "answer": "yes" }));
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn streams_array_elements_from_typed_output() {
    let mock = MockHarnessOptions::script(|| {
        structured_turn(r#"{"elements":[{"answer":"yes"},{"answer":"no"}]}"#)
    })
    .build();
    let mut s = settings(&mock);
    s.output = Some(array(answer_schema(), None, None, None, None).unwrap());
    let agent = agent(s);
    let session = new_session(&agent).await;

    let result = agent.stream(call(&session, "answer twice")).await.unwrap();
    let elements: Vec<JsonValue> = result
        .element_stream()
        .unwrap()
        .map(|element| element.unwrap())
        .collect()
        .await;

    assert_eq!(
        elements,
        vec![json!({ "answer": "yes" }), json!({ "answer": "no" })]
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn stream_returns_a_result_whose_full_stream_emits_translated_parts() {
    let mock = MockHarnessOptions::script(|| {
        parts(vec![
            json!({ "type": "stream-start" }),
            json!({ "type": "text-delta", "id": "t1", "delta": "Hi" }),
            json!({ "type": "finish-step", "finishReason": { "unified": "stop" }, "usage": zero_usage() }),
            json!({ "type": "finish", "finishReason": { "unified": "stop" }, "totalUsage": zero_usage() }),
        ])
    })
    .build();

    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "hi")).await.unwrap();

    let types = part_types(&result).await;

    // The message-level `start` part must be the first chunk, mirroring
    // `streamText`: UI message stream consumers rely on it.
    assert_eq!(types[0], "start");
    assert!(types.contains(&"text-delta".to_string()));
    assert!(types.contains(&"finish-step".to_string()));
    assert!(types.contains(&"finish".to_string()));
    assert_eq!(result.text().await.unwrap(), "Hi");

    session.destroy().await.unwrap();
}

/// `twoStepEvents()`.
fn two_step_events() -> Vec<HarnessV1StreamPart> {
    parts(vec![
        json!({ "type": "text-start", "id": "first" }),
        json!({ "type": "text-delta", "id": "first", "delta": "First" }),
        json!({ "type": "text-end", "id": "first" }),
        json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "tool_use" }, "usage": zero_usage() }),
        json!({ "type": "text-start", "id": "second" }),
        json!({ "type": "text-delta", "id": "second", "delta": "Second" }),
        json!({ "type": "text-end", "id": "second" }),
        json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "end_turn" }, "usage": zero_usage() }),
        json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "end_turn" }, "totalUsage": zero_usage() }),
    ])
}

#[tokio::test]
async fn ui_streams_forward_on_step_end_and_the_deprecated_on_step_finish() {
    use crate::harness::sdk::telemetry::callback;
    use crate::harness::sdk::ui_message_stream::{
        UIMessageStreamOnStepEndEvent, UIMessageStreamOptions,
    };

    for (name, response, use_on_step_end, use_on_step_finish, expected_end, expected_finish) in [
        (
            "direct UI stream forwards onStepEnd",
            false,
            true,
            false,
            2,
            0,
        ),
        (
            "response UI stream forwards onStepEnd",
            true,
            true,
            false,
            2,
            0,
        ),
        (
            "direct UI stream forwards deprecated onStepFinish",
            false,
            false,
            true,
            0,
            2,
        ),
        (
            "response UI stream forwards deprecated onStepFinish",
            true,
            false,
            true,
            0,
            2,
        ),
        (
            "direct UI stream prefers onStepEnd over onStepFinish",
            false,
            true,
            true,
            2,
            0,
        ),
        (
            "response UI stream prefers onStepEnd over onStepFinish",
            true,
            true,
            true,
            2,
            0,
        ),
    ] {
        let mock = MockHarnessOptions::script(two_step_events).build();
        let agent = agent(settings(&mock));
        let session = new_session(&agent).await;
        let result = agent.stream(call(&session, "two steps")).await.unwrap();
        let on_step_end_calls = Arc::new(AtomicUsize::new(0));
        let on_step_finish_calls = Arc::new(AtomicUsize::new(0));
        let counter = |count: &Arc<AtomicUsize>| {
            let count = count.clone();
            callback(move |_: UIMessageStreamOnStepEndEvent| {
                count.fetch_add(1, Ordering::SeqCst);
                async {}
            })
        };
        let options = UIMessageStreamOptions {
            on_step_end: use_on_step_end.then(|| counter(&on_step_end_calls)),
            on_step_finish: use_on_step_finish.then(|| counter(&on_step_finish_calls)),
            ..Default::default()
        };

        let finished_steps = if response {
            let response = result
                .to_ui_message_stream_response(options, Default::default())
                .unwrap();
            let bytes: Vec<_> = response.body.map(|b| b.unwrap()).collect().await;
            String::from_utf8(bytes.concat())
                .unwrap()
                .split('\n')
                .filter(|line| line.contains(r#""type":"finish-step""#))
                .count()
        } else {
            result
                .to_ui_message_stream(options)
                .filter(|chunk| {
                    let is_finish_step = matches!(
                        chunk,
                        Ok(crate::harness::sdk::ui_message_stream::UIMessageChunk::FinishStep)
                    );
                    async move { is_finish_step }
                })
                .count()
                .await
        };

        assert_eq!(finished_steps, 2, "{name}");
        assert_eq!(
            on_step_end_calls.load(Ordering::SeqCst),
            expected_end,
            "{name}"
        );
        assert_eq!(
            on_step_finish_calls.load(Ordering::SeqCst),
            expected_finish,
            "{name}"
        );
        session.destroy().await.unwrap();
    }
}

#[tokio::test]
async fn stream_exposes_reasoning_from_the_final_step() {
    let mock = MockHarnessOptions::script(|| {
        let mut values = vec![
            json!({ "type": "reasoning-start", "id": "r1" }),
            json!({ "type": "reasoning-delta", "id": "r1", "delta": "First thought." }),
            json!({ "type": "reasoning-end", "id": "r1" }),
            json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "tool_use" }, "usage": zero_usage() }),
            json!({ "type": "reasoning-start", "id": "r2" }),
            json!({ "type": "reasoning-delta", "id": "r2", "delta": "Final thought." }),
            json!({ "type": "reasoning-end", "id": "r2" }),
            json!({ "type": "text-start", "id": "t1" }),
            json!({ "type": "text-delta", "id": "t1", "delta": "Answer." }),
            json!({ "type": "text-end", "id": "t1" }),
        ];
        values.extend(stop_finish());
        parts(values)
    })
    .build();

    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "think")).await.unwrap();

    let streamed_reasoning: Vec<String> = result
        .full_stream()
        .filter_map(|part| async move {
            let part = part.to_json();
            (part["type"] == "reasoning-delta").then(|| part["text"].as_str().unwrap().to_string())
        })
        .collect()
        .await;

    assert_eq!(streamed_reasoning, vec!["First thought.", "Final thought."]);
    assert_eq!(
        to_json(&result.reasoning().await.unwrap()),
        // [rust] `ReasoningPart` is the struct inside the tagged `reasoning`
        // content part, so it serializes without `type`.
        json!([{ "text": "Final thought." }])
    );
    assert_eq!(
        result.reasoning_text().await.unwrap().as_deref(),
        Some("Final thought.")
    );

    let steps = result.steps().await.unwrap();
    assert_eq!(
        steps
            .iter()
            .map(|step| step.reasoning_text())
            .collect::<Vec<_>>(),
        vec![
            Some("First thought.".to_string()),
            Some("Final thought.".to_string())
        ]
    );
    let final_step = result.final_step().await.unwrap();
    assert_eq!(result.reasoning().await.unwrap(), final_step.reasoning());
    assert_eq!(
        result.reasoning_text().await.unwrap(),
        final_step.reasoning_text()
    );

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn stream_exposes_empty_reasoning_accessors_when_no_reasoning_is_emitted() {
    let mock = MockHarnessOptions::script(|| text_turn("t1", "Answer.")).build();

    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "answer")).await.unwrap();

    assert!(result.reasoning().await.unwrap().is_empty());
    assert_eq!(result.reasoning_text().await.unwrap(), None);

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn releases_the_session_after_a_turn_start_failure() {
    let prompt_turn_count = Arc::new(AtomicUsize::new(0));
    let mut options = MockHarnessOptions::script(finish_events);
    let count = prompt_turn_count.clone();
    options.on_prompt_turn = Some(Arc::new(move |_| {
        if count.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(plain_error("failed to start turn"));
        }
        Ok(())
    }));
    let mock = options.build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let failed = agent.stream(call(&session, "fail")).await.unwrap();
    let error = failed.text().await.unwrap_err();
    assert!(
        error.to_string().contains("failed to start turn"),
        "{error}"
    );

    assert!(!session.has_unfinished_turn());
    assert!(agent.generate(call(&session, "recover")).await.is_ok());

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn releases_the_session_after_a_continued_turn_stream_error() {
    let mock = MockHarnessOptions::script(finish_events)
        .with_continue(|| {
            vec![HarnessV1StreamPart::Error {
                error: ErrorValue::from(plain_error("continued turn failed")),
            }]
        })
        .build();
    let agent = agent(settings(&mock));
    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            continue_from: Some(mock_continue_state()),
            ..with_sandbox(&make_sandbox_session())
        })
        .await
        .unwrap();

    let failed = agent
        .continue_stream(HarnessAgentContinueParameters::new(session.clone()))
        .await
        .unwrap();
    let error = failed.text().await.unwrap_err();
    assert!(
        error.to_string().contains("continued turn failed"),
        "{error}"
    );

    assert!(!session.has_unfinished_turn());
    assert!(agent.generate(call(&session, "recover")).await.is_ok());

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn releases_the_session_after_an_aborted_turn_closes_without_finish() {
    let prompt_turn_count = Arc::new(AtomicUsize::new(0));
    let abort_controller = tokio_util::sync::CancellationToken::new();
    let count = prompt_turn_count.clone();
    let mut options = MockHarnessOptions {
        script: Some(Arc::new({
            let count = prompt_turn_count.clone();
            move || {
                if count.load(Ordering::SeqCst) == 1 {
                    Vec::new()
                } else {
                    finish_events()
                }
            }
        })),
        ..Default::default()
    };
    options.on_prompt_turn = Some(Arc::new(move |_| {
        count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }));
    let count = prompt_turn_count.clone();
    options.prompt_done = Some(Arc::new(move |options| {
        if count.load(Ordering::SeqCst) > 1 {
            return Box::pin(async { Ok(()) });
        }
        let abort_signal = options
            .abort_signal
            .clone()
            .expect("Expected an abort signal.");
        Box::pin(async move {
            abort_signal.cancelled().await;
            Ok(())
        })
    }));
    let mock = options.build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let mut params = call(&session, "abort");
    params.abort_signal = Some(abort_controller.clone());
    let aborted = agent.stream(params).await.unwrap();
    abort_controller.cancel();
    aborted.consume_stream().await;

    assert!(!session.has_unfinished_turn());
    assert!(agent.generate(call(&session, "recover")).await.is_ok());

    session.destroy().await.unwrap();
}

/// The doubles of the suspension race tests: a tool whose work the test
/// releases, and an adapter channel that closes on `doSuspendTurn`.
#[derive(Clone, Default)]
struct SuspensionRace {
    work: Gate,
    started: Gate,
    stream_done: Gate,
    closed: Gate,
    channel_closed: Arc<AtomicBool>,
    executions: Arc<AtomicUsize>,
    completed: Arc<Mutex<Vec<String>>>,
}

impl SuspensionRace {
    /// `onDoStart`, `onSuspendTurn`, `onSubmitToolResult` (and `promptDone`
    /// when `close_stream`).
    fn configure(
        &self,
        options: &mut MockHarnessOptions,
        close_stream: bool,
        message: &'static str,
    ) {
        let race = self.clone();
        options.on_do_start = Some(Arc::new(move |_| {
            race.channel_closed.store(false, Ordering::SeqCst)
        }));
        let race = self.clone();
        options.on_suspend_turn = Some(Arc::new(move || {
            race.channel_closed.store(true, Ordering::SeqCst);
            if close_stream {
                race.stream_done.open();
            }
            race.closed.open();
            Box::pin(async { Ok(()) })
        }));
        let race = self.clone();
        options.on_submit_tool_result = Some(Arc::new(move |_| {
            let closed = race.channel_closed.load(Ordering::SeqCst);
            Box::pin(async move {
                if closed {
                    return Err(plain_error(message));
                }
                Ok(())
            })
        }));
        if close_stream {
            let gate = self.stream_done.clone();
            options.prompt_done = Some(Arc::new(move |_| {
                let gate = gate.clone();
                Box::pin(async move {
                    gate.wait().await;
                    Ok(())
                })
            }));
        }
    }
}

fn research_call(index: usize) -> JsonValue {
    json!({
        "type": "tool-call", "toolCallId": format!("call-{index}"), "toolName": "research",
        "input": json!({ "query": index.to_string() }).to_string(),
    })
}

#[tokio::test]
async fn preserves_dispatched_host_results_when_the_adapter_suspends() {
    // [rust] The `suspendDuringValidation` case is not ported: schema
    // validation is synchronous in Rust, so a suspension cannot interleave
    // with it.
    for (count, error) in [(1usize, false), (3, true)] {
        let race = SuspensionRace::default();
        let calls: Vec<JsonValue> = (0..count).map(research_call).collect();
        let script_calls = calls.clone();
        let continue_calls = calls.clone();
        let mut options = MockHarnessOptions::script(move || parts(script_calls.clone()))
            .with_continue(move || {
                let mut values: Vec<JsonValue> = continue_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "type": "tool-result", "toolCallId": call["toolCallId"], "toolName": call["toolName"],
                            "result": { "answer": call["toolCallId"] },
                        })
                    })
                    .collect();
                values.extend(finish_events().iter().map(to_json));
                parts(values)
            });
        race.configure(
            &mut options,
            true,
            "SandboxChannel: cannot send tool-result — channel is closed.",
        );
        let mock = options.build();
        let r = race.clone();
        let research = tool(&[("query", "string")]).with_execute(move |input, _| {
            let r = r.clone();
            async move {
                if r.executions.fetch_add(1, Ordering::SeqCst) + 1 == count {
                    r.started.open();
                }
                r.work.wait().await;
                let query = input["query"].as_str().unwrap().to_string();
                r.completed.lock().push(query.clone());
                if error && query == "1" {
                    return Err(plain_error("tool unavailable"));
                }
                Ok(json!({ "answer": query }))
            }
        });
        let mut s = settings(&mock);
        s.tools = Some(tools(vec![("research", research)]));
        let agent = agent(s);
        let sandbox = make_sandbox_session();
        let session = agent.create_session(with_sandbox(&sandbox)).await.unwrap();
        let first = agent.stream(call(&session, "research")).await.unwrap();
        let first_stream = first.full_stream();
        let first_read = tokio::spawn(async move {
            first_stream
                .map(|part| part.to_json()["type"].as_str().unwrap().to_string())
                .collect::<Vec<_>>()
                .await
        });
        race.started.wait().await;
        let suspending = session.clone();
        let suspension = tokio::spawn(async move { suspending.suspend_turn().await });
        race.closed.wait().await;
        race.work.open();
        let continue_from = suspension.await.unwrap().unwrap();
        let first_parts = first_read.await.unwrap();

        assert_eq!(race.completed.lock().len(), count);
        assert_eq!(
            continue_from
                .pending_tool_results
                .as_ref()
                .map_or(0, |r| r.len()),
            count
        );
        assert!(mock.tool_results().is_empty());
        assert!(
            !first_parts.contains(&"error".to_string()),
            "{first_parts:?}"
        );
        assert_eq!(
            count_of(&first_parts, "tool-result"),
            count - usize::from(error)
        );
        assert_eq!(count_of(&first_parts, "tool-error"), usize::from(error));

        let resumed = agent
            .create_session(HarnessAgentCreateSessionOptions {
                session_id: Some(session.session_id().to_string()),
                continue_from: Some(continue_from.clone()),
                ..with_sandbox(&sandbox)
            })
            .await
            .unwrap();
        let second = agent
            .continue_stream(HarnessAgentContinueParameters::new(resumed.clone()))
            .await
            .unwrap();
        let second_parts = part_types(&second).await;

        assert_eq!(race.executions.load(Ordering::SeqCst), count);
        assert_eq!(mock.prompts().len(), 1);
        assert!(matches!(&mock.prompts()[0], HarnessV1Prompt::Text(t) if t == "research"));
        assert_eq!(mock.continue_count(), 1);
        let submitted: Vec<JsonValue> = mock.tool_results().iter().map(submitted_json).collect();
        assert_eq!(submitted.len(), count);
        for index in 0..count {
            let expected = if error && index == 1 {
                json!({ "toolCallId": "call-1", "output": { "error": "Error: tool unavailable" }, "isError": true })
            } else {
                json!({ "toolCallId": format!("call-{index}"), "output": { "answer": index.to_string() } })
            };
            assert!(
                submitted.contains(&expected),
                "{submitted:?} missing {expected}"
            );
        }
        assert!(!second_parts.contains(&"tool-result".to_string()));
        assert!(!second_parts.contains(&"tool-error".to_string()));
        assert_eq!(count_of(&second_parts, "finish-step"), 1);
        assert_eq!(second.steps().await.unwrap().len(), 1);
        resumed.destroy().await.unwrap();
    }
}

#[tokio::test]
async fn preserves_a_resumed_approved_host_call_across_another_suspension() {
    let race = SuspensionRace::default();
    let continuation_count = Arc::new(AtomicUsize::new(0));
    let mut options = MockHarnessOptions::script(Vec::new).with_continue(move || {
        if continuation_count.fetch_add(1, Ordering::SeqCst) == 0 {
            Vec::new()
        } else {
            let mut script = parts(vec![json!({
                "type": "tool-result", "toolCallId": "call-1", "toolName": "research",
                "result": { "answer": "retained" },
            })]);
            script.extend(finish_events());
            script
        }
    });
    race.configure(&mut options, false, "channel is closed");
    let mock = options.build();
    let r = race.clone();
    let research = Tool::new(z_object(&[])).with_execute(move |_, _| {
        let r = r.clone();
        async move {
            r.executions.fetch_add(1, Ordering::SeqCst);
            r.started.open();
            r.work.wait().await;
            Ok(json!({ "answer": "retained" }))
        }
    });
    let mut s = settings(&mock);
    s.tools = Some(tools(vec![("research", research)]));
    let agent = agent(s);
    let sandbox = make_sandbox_session();
    let mut continue_from = mock_continue_state();
    continue_from.pending_tool_approvals = Some(vec![custom_approval(
        "approval-1",
        "call-1",
        "research",
        "{}",
    )]);
    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            continue_from: Some(continue_from),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    let mut params = HarnessAgentContinueParameters::new(session.clone());
    params.tool_approval_continuations = vec![approval_response("approval-1", true)];
    let first = agent.continue_stream(params).await.unwrap();
    let first_stream = first.full_stream();
    let consume = tokio::spawn(async move {
        first_stream
            .map(|part| part.to_json()["type"].as_str().unwrap().to_string())
            .collect::<Vec<_>>()
            .await
    });
    race.started.wait().await;
    let suspending = session.clone();
    let suspension = tokio::spawn(async move { suspending.suspend_turn().await });
    race.closed.wait().await;
    race.work.open();
    let continue_from = suspension.await.unwrap().unwrap();
    let first_parts = consume.await.unwrap();

    assert!(
        !first_parts.contains(&"error".to_string()),
        "{first_parts:?}"
    );
    assert_eq!(count_of(&first_parts, "tool-result"), 1);
    assert_eq!(
        continue_from
            .pending_tool_results
            .as_ref()
            .map_or(0, |r| r.len()),
        1
    );
    assert!(
        to_json(&continue_from)
            .get("pendingToolApprovals")
            .is_none()
    );

    let resumed = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some(session.session_id().to_string()),
            continue_from: Some(continue_from),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    let second = agent
        .continue_stream(HarnessAgentContinueParameters::new(resumed.clone()))
        .await
        .unwrap();
    second.consume_stream().await;

    assert_eq!(race.executions.load(Ordering::SeqCst), 1);
    assert_eq!(
        mock.tool_results()
            .iter()
            .map(submitted_json)
            .collect::<Vec<_>>(),
        vec![json!({ "toolCallId": "call-1", "output": { "answer": "retained" } })]
    );
    resumed.destroy().await.unwrap();
}

#[tokio::test]
async fn session_detach_and_stop_preserve_an_in_flight_host_result_in_their_nested_continuation() {
    for lifecycle_method in ["detach", "stop"] {
        let race = SuspensionRace::default();
        let tool_call = json!({ "type": "tool-call", "toolCallId": "call-1", "toolName": "research", "input": "{}" });
        let script_call = tool_call.clone();
        let mut options = MockHarnessOptions::script(move || parts(vec![script_call.clone()]))
            .with_continue(|| {
                let mut script = parts(vec![json!({
                    "type": "tool-result", "toolCallId": "call-1", "toolName": "research",
                    "result": { "answer": "retained" },
                })]);
                script.extend(finish_events());
                script
            });
        race.configure(&mut options, true, "channel is closed");
        let mock = options.build();
        let r = race.clone();
        let research = Tool::new(z_object(&[])).with_execute(move |_, _| {
            let r = r.clone();
            async move {
                r.executions.fetch_add(1, Ordering::SeqCst);
                r.started.open();
                r.work.wait().await;
                Ok(json!({ "answer": "retained" }))
            }
        });
        let mut s = settings(&mock);
        s.tools = Some(tools(vec![("research", research)]));
        let agent = agent(s);
        let sandbox = make_sandbox_session();
        let session = agent.create_session(with_sandbox(&sandbox)).await.unwrap();
        let first = agent.stream(call(&session, "research")).await.unwrap();
        let first_read = tokio::spawn(async move { first.consume_stream().await });
        race.started.wait().await;

        let ending_session = session.clone();
        let ending = tokio::spawn(async move {
            if lifecycle_method == "detach" {
                ending_session.detach().await
            } else {
                ending_session.stop().await
            }
        });
        race.closed.wait().await;
        race.work.open();
        let resume_from = ending.await.unwrap().unwrap();
        first_read.await.unwrap();

        assert_eq!(
            to_json(
                &resume_from
                    .continue_from
                    .as_ref()
                    .unwrap()
                    .pending_tool_results
            ),
            json!([{
                "toolCallId": "call-1",
                "toolName": "research",
                "input": "{}",
                "completedResult": { "output": { "answer": "retained" } },
            }])
        );
        assert!(mock.tool_results().is_empty());

        let resumed = agent
            .create_session(HarnessAgentCreateSessionOptions {
                session_id: Some(session.session_id().to_string()),
                resume_from: Some(resume_from),
                ..with_sandbox(&sandbox)
            })
            .await
            .unwrap();
        let second = agent
            .continue_stream(HarnessAgentContinueParameters::new(resumed.clone()))
            .await
            .unwrap();
        second.consume_stream().await;

        assert_eq!(race.executions.load(Ordering::SeqCst), 1);
        assert_eq!(mock.continue_count(), 1);
        assert_eq!(
            mock.tool_results()
                .iter()
                .map(submitted_json)
                .collect::<Vec<_>>(),
            vec![json!({ "toolCallId": "call-1", "output": { "answer": "retained" } })]
        );
        resumed.destroy().await.unwrap();
    }
}

#[tokio::test]
async fn keeps_a_turn_unfinished_when_suspension_closes_its_stream_mid_step() {
    let prompt_done = Gate::default();
    let mut options = MockHarnessOptions::script(|| {
        parts(vec![
            json!({ "type": "text-start", "id": "t1" }),
            json!({ "type": "text-delta", "id": "t1", "delta": "partial" }),
        ])
    });
    let gate = prompt_done.clone();
    options.prompt_done = Some(Arc::new(move |_| {
        let gate = gate.clone();
        Box::pin(async move {
            gate.wait().await;
            Ok(())
        })
    }));
    let gate = prompt_done.clone();
    options.on_suspend_turn = Some(Arc::new(move || {
        gate.open();
        Box::pin(async { Ok(()) })
    }));
    let mock = options.build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    let result = agent.stream(call(&session, "work")).await.unwrap();

    let continue_from = session.suspend_turn().await.unwrap();
    result.consume_stream().await;

    assert_eq!(
        to_json(&continue_from),
        json!({
            "type": "continue-turn",
            "harnessId": "mock",
            "specificationVersion": "harness-v1",
            "data": {},
            "turnSettings": { "skills": [], "tools": [] },
        })
    );
    assert!(session.has_unfinished_turn());
    assert!(result.steps().await.unwrap().is_empty());
}

fn aborting_mock() -> Mock {
    let prompt_turn_count = Arc::new(AtomicUsize::new(0));
    let count = prompt_turn_count.clone();
    let mut options = MockHarnessOptions::script(move || {
        if prompt_turn_count.load(Ordering::SeqCst) == 1 {
            parts(vec![
                json!({ "type": "error", "error": "AbortError: This operation was aborted" }),
            ])
        } else {
            finish_events()
        }
    });
    options.on_prompt_turn = Some(Arc::new(move |_| {
        count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }));
    options.build()
}

#[tokio::test]
async fn settles_an_aborted_turn_with_an_abort_part_and_releases_the_session_for_the_next_turn() {
    let abort_controller = tokio_util::sync::CancellationToken::new();
    abort_controller.cancel();
    let mock = aborting_mock();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let mut params = call(&session, "abort");
    params.abort_signal = Some(abort_controller);
    let aborted = agent.stream(params).await.unwrap();

    let types = part_types(&aborted).await;

    // The user stop surfaces as an `abort` part, not an `error` part …
    assert!(types.contains(&"abort".to_string()), "{types:?}");
    assert!(!types.contains(&"error".to_string()), "{types:?}");
    // … and the turn-settlement lifecycle ran: the session returns to idle
    // and accepts the next turn.
    assert!(!session.has_unfinished_turn());
    assert!(agent.generate(call(&session, "recover")).await.is_ok());

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn does_not_log_a_bridge_error_to_stderr_for_a_turn_the_caller_aborted() {
    // [rust] The process's stderr cannot be intercepted from a test, so this
    // checks the turn settles as an abort, the condition under which the
    // bridge error is not logged.
    let abort_controller = tokio_util::sync::CancellationToken::new();
    abort_controller.cancel();
    let mock = MockHarnessOptions::script(|| {
        parts(vec![
            json!({ "type": "error", "error": "AbortError: This operation was aborted" }),
        ])
    })
    .build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let mut params = call(&session, "abort");
    params.abort_signal = Some(abort_controller);
    let aborted = agent.stream(params).await.unwrap();
    let types = part_types(&aborted).await;
    assert!(types.contains(&"abort".to_string()), "{types:?}");
    assert!(!types.contains(&"error".to_string()), "{types:?}");

    session.destroy().await.unwrap();
}
