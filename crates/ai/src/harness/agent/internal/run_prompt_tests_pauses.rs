//! `run-prompt.test.ts`: `runPrompt suspension lifecycle`, `runPrompt abort
//! semantics` and `runPrompt host input pauses`.

use super::tool_outputs::{approval_config, approval_response, recording_tool, submitted_list};
use super::*;

fn call(tool_call_id: &str, tool_name: &str, input: JsonValue) -> HarnessV1StreamPart {
    p(
        json!({ "type": "tool-call", "toolCallId": tool_call_id, "toolName": tool_name, "input": input.to_string() }),
    )
}

type Counter = Arc<AtomicUsize>;

fn counter() -> (Counter, Arc<dyn Fn() + Send + Sync>) {
    let count: Counter = Arc::default();
    let c = count.clone();
    (
        count,
        Arc::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
        }),
    )
}

fn aborted_token() -> CancellationToken {
    let token = CancellationToken::new();
    token.cancel();
    token
}

fn abort_error_script() -> Vec<HarnessV1StreamPart> {
    parts(vec![
        json!({ "type": "error", "error": "AbortError: This operation was aborted" }),
    ])
}

// ─── runPrompt suspension lifecycle ────────────────────────────────────

struct SuspendingRun {
    output: RunPromptOutput,
    parts: Vec<JsonValue>,
    finished: Counter,
    failed: Counter,
}

async fn suspending_run(
    script: Vec<HarnessV1StreamPart>,
    abort_signal: Option<CancellationToken>,
) -> SuspendingRun {
    let (finished, on_turn_finished) = counter();
    let (failed, on_turn_failed) = counter();
    let mut run = base_input(fake_harness(), FakeSession::new(script), ToolSet::new());
    run.abort_signal = abort_signal;
    run.on_turn_finished = Some(on_turn_finished);
    run.on_turn_failed = Some(on_turn_failed);
    run.is_turn_suspending = Some(Arc::new(|| true));
    let (parts, output) = run_collect(run).await;
    SuspendingRun {
        output,
        parts,
        finished,
        failed,
    }
}

#[tokio::test]
async fn settles_a_result_with_a_terminal_finish_without_finishing_the_suspended_turn() {
    let run = suspending_run(finish_events(), None).await;
    assert_eq!(
        to_json(&run.output.result.finish_reason().await.unwrap()),
        "stop"
    );
    assert_eq!(run.finished.load(Ordering::SeqCst), 0);
    assert_eq!(run.failed.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn settles_a_cleanly_closed_result_without_failing_the_suspended_turn() {
    let run = suspending_run(Vec::new(), None).await;
    assert_eq!(run.finished.load(Ordering::SeqCst), 0);
    assert_eq!(run.failed.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn preserves_error_result_semantics_without_failing_the_suspended_turn() {
    let run = suspending_run(
        parts(vec![json!({ "type": "error", "error": "boom" })]),
        None,
    )
    .await;
    assert!(run.output.result.finish_reason().await.is_err());
    assert_eq!(run.finished.load(Ordering::SeqCst), 0);
    assert_eq!(run.failed.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn preserves_abort_result_semantics_without_failing_the_suspended_turn() {
    let run = suspending_run(abort_error_script(), Some(aborted_token())).await;
    assert!(of_type(&run.parts, "error").is_empty());
    assert_eq!(run.parts.last().unwrap()["type"], "abort");
    assert!(run.output.result.finish_reason().await.is_err());
    assert_eq!(run.finished.load(Ordering::SeqCst), 0);
    assert_eq!(run.failed.load(Ordering::SeqCst), 0);
}

// ─── runPrompt abort semantics ─────────────────────────────────────────

/// Records `onAbort` and `onError` events; optionally aborts on
/// `onLanguageModelCallStart`.
#[derive(Default)]
struct AbortTelemetry {
    abort_on_call_start: Option<CancellationToken>,
    aborts: Arc<Mutex<Vec<TelemetryEvent>>>,
    errors: Arc<Mutex<Vec<TelemetryEvent>>>,
}

impl Telemetry for AbortTelemetry {
    fn on_language_model_call_start(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        if let Some(token) = &self.abort_on_call_start {
            assert!(!token.is_cancelled());
            token.cancel();
        }
        Box::pin(async {})
    }
    fn on_abort(&self, event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.aborts.lock().push(event.clone());
        Box::pin(async {})
    }
    fn on_error(&self, event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.errors.lock().push(event.clone());
        Box::pin(async {})
    }
}

fn assert_abort_event(event: &TelemetryEvent) {
    let event = JsonValue::Object(event.clone());
    assert!(event["aiVersion"].is_string(), "{event}");
    assert!(event["callId"].is_string(), "{event}");
    assert_eq!(event["steps"], json!([]));
    assert!(!event["reason"].is_null(), "{event}");
    let mut keys: Vec<&String> = event.as_object().unwrap().keys().collect();
    keys.sort();
    assert_eq!(keys, ["aiVersion", "callId", "reason", "steps"]);
}

#[tokio::test]
async fn dispatches_abort_telemetry_when_the_caller_stops_an_active_turn() {
    let token = CancellationToken::new();
    let telemetry = Arc::new(AbortTelemetry {
        abort_on_call_start: Some(token.clone()),
        ..Default::default()
    });
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(parts(vec![
            json!({ "type": "stream-start", "modelId": "mock-model" }),
            json!({ "type": "text-start", "id": "t1" }),
            json!({ "type": "error", "error": "adapter stopped" }),
        ])),
        ToolSet::new(),
    );
    run.abort_signal = Some(token);
    run.telemetry = Some(telemetry_with(telemetry.clone()));
    let (parts, _) = run_collect(run).await;

    // [rust] A `CancellationToken` carries no reason, so `reason` is the
    // `AbortError` rather than the caller's `Error('user stopped')`.
    let aborts = telemetry.aborts.lock().clone();
    assert_eq!(aborts.len(), 1);
    assert_abort_event(&aborts[0]);
    assert!(telemetry.errors.lock().is_empty());
    assert_eq!(parts.last().unwrap()["type"], "abort");
}

fn aborted_run(script: Vec<HarnessV1StreamPart>) -> RunPromptInput {
    let mut run = base_input(fake_harness(), FakeSession::new(script), ToolSet::new());
    run.abort_signal = Some(aborted_token());
    run
}

#[tokio::test]
async fn settles_with_an_abort_part_instead_of_an_error_part_when_the_abort_signal_has_fired() {
    let (parts, output) = run_collect(aborted_run(parts(vec![
        json!({ "type": "text-start", "id": "t1" }),
        json!({ "type": "text-delta", "id": "t1", "delta": "partial " }),
        json!({ "type": "error", "error": "AbortError: This operation was aborted" }),
    ])))
    .await;

    assert!(of_type(&parts, "error").is_empty());
    let last = parts.last().unwrap();
    assert_eq!(last["type"], "abort");
    assert!(
        last["reason"].as_str().unwrap().contains("aborted"),
        "{last}"
    );
    // Awaiting consumers still settle (rejected with the underlying error).
    assert!(output.result.finish_reason().await.is_err());
}

#[tokio::test]
async fn keeps_a_real_error_part_when_the_abort_signal_has_not_fired() {
    let (parts, output) = run_collect(base_input(
        fake_harness(),
        FakeSession::new(parts(vec![json!({ "type": "error", "error": "boom" })])),
        ToolSet::new(),
    ))
    .await;

    assert!(of_type(&parts, "abort").is_empty());
    assert_eq!(parts.last().unwrap()["type"], "error");
    assert!(output.result.finish_reason().await.is_err());
}

#[tokio::test]
async fn notifies_on_turn_failed_when_an_aborted_turn_settles() {
    let (failed, on_turn_failed) = counter();
    let mut run = aborted_run(abort_error_script());
    run.on_turn_failed = Some(on_turn_failed);
    run_collect(run).await;
    assert_eq!(failed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dispatches_abort_telemetry_instead_of_error_telemetry() {
    let telemetry = Arc::new(AbortTelemetry::default());
    let mut run = aborted_run(abort_error_script());
    run.telemetry = Some(telemetry_with(telemetry.clone()));
    run_collect(run).await;

    let aborts = telemetry.aborts.lock().clone();
    assert_eq!(aborts.len(), 1);
    assert_abort_event(&aborts[0]);
    assert!(telemetry.errors.lock().is_empty());
}

// [rust] `toUIMessageStream emits an abort chunk, skips onError, and
// reports isAborted to onEnd for an aborted turn` comes with the UI message
// stream port.

// ─── runPrompt host input pauses ───────────────────────────────────────

fn boundary(total: u64) -> HarnessV1StreamPart {
    p(json!({
        "type": "finish-step",
        "finishReason": { "unified": "tool-calls", "raw": "host-input" },
        "usage": { "inputTokens": { "total": total }, "outputTokens": {} },
    }))
}

/// Wraps `emit` so that a `finish` part settles the turn.
fn finishing_emit(emit: HarnessV1Emit, turn_done: Gate) -> HarnessV1Emit {
    Arc::new(move |part: HarnessV1StreamPart| {
        let finished = matches!(part, HarnessV1StreamPart::Finish { .. });
        emit(part);
        if finished {
            turn_done.open();
        }
    })
}

/// `scriptedSession({ start, onSubmitToolResult })`: the turn is done once
/// a `finish` part is emitted.
fn scripted_session(
    start: impl Fn(&HarnessV1Emit) + Send + Sync + 'static,
    on_submit: impl Fn(HarnessV1SubmitToolResult, HarnessV1Emit) + Send + Sync + 'static,
) -> FakeSession {
    let turn_done = Gate::default();
    let mut session = FakeSession::new(Vec::new());
    let gate = turn_done.clone();
    session.on_turn = Some(Arc::new(move |emit| {
        start(&finishing_emit(emit, gate.clone()))
    }));
    let gate = turn_done.clone();
    session = session.with_submit(move |submission, emit| {
        on_submit(submission, finishing_emit(emit, gate.clone()))
    });
    session.done = Some(
        async move {
            turn_done.wait().await;
            Ok(())
        }
        .boxed()
        .shared(),
    );
    session
}

fn weather() -> Tool {
    tool(&[("city", "string")])
        .with_description("Get weather")
        .with_execute(
            |input, _| async move { Ok(json!({ "city": input["city"], "temperature": 72 })) },
        )
}

fn step_content_types(steps: &JsonValue, index: usize) -> Vec<String> {
    types(steps[index]["content"].as_array().unwrap())
}

#[tokio::test]
async fn surfaces_the_rest_of_the_step_and_pauses_at_the_next_finish_step() {
    let mut script = vec![
        call("c1", "weather", json!({ "city": "SF" })),
        p(json!({
            "type": "tool-call", "toolCallId": "native-1", "toolName": "read",
            "input": json!({ "path": "a.txt" }).to_string(), "providerExecuted": true, "dynamic": true,
        })),
        p(
            json!({ "type": "tool-result", "toolCallId": "native-1", "toolName": "read", "result": "file contents" }),
        ),
        call("c2", "weather", json!({ "city": "NYC" })),
        boundary(7),
    ];
    script.extend(parts(vec![
        json!({ "type": "text-start", "id": "t1" }),
        json!({ "type": "text-delta", "id": "t1", "delta": "must not appear" }),
        json!({ "type": "text-end", "id": "t1" }),
    ]));
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(script)),
        tools(vec![("weather", weather())]),
    );
    run.tool_approval = Some(approval_config(json!({ "weather": "user-approval" })));
    let pending: Arc<Mutex<Vec<HarnessV1PendingToolApproval>>> = Arc::default();
    let log = pending.clone();
    run.on_pending_tool_approval = Some(Arc::new(move |a| log.lock().push(a)));
    let (parts, output) = run_collect(run).await;

    assert_eq!(
        pending
            .lock()
            .iter()
            .map(|a| a.tool_call_id.clone())
            .collect::<Vec<_>>(),
        ["c1", "c2"]
    );
    assert!(of_type(&parts, "text-delta").is_empty());
    let steps = to_json(&output.result.steps().await.unwrap());
    assert_eq!(steps.as_array().unwrap().len(), 1);
    assert_eq!(steps[0]["finishReason"], "tool-calls");
    assert_eq!(steps[0]["usage"]["inputTokens"], 7);
    assert_eq!(
        step_content_types(&steps, 0),
        [
            "tool-call",
            "tool-approval-request",
            "tool-call",
            "tool-result",
            "tool-call",
            "tool-approval-request",
        ]
    );
}

#[tokio::test]
async fn ignores_a_finish_step_while_a_host_tool_has_not_been_echoed_yet() {
    let session = scripted_session(
        |emit| {
            emit(p(
                json!({ "type": "stream-start", "modelId": "fake-model" }),
            ));
            emit(call("c1", "weather", json!({ "city": "SF" })));
            emit(boundary(3));
        },
        |submission, emit| {
            emit(p(json!({
                "type": "tool-result", "toolCallId": submission.tool_call_id,
                "toolName": "weather", "result": submission.output,
            })));
            emit(boundary(5));
            emit(finish());
        },
    );
    let (_, output) = run_collect(base_input(
        fake_harness(),
        session,
        tools(vec![("weather", weather())]),
    ))
    .await;

    let steps = to_json(&output.result.steps().await.unwrap());
    assert_eq!(steps.as_array().unwrap().len(), 1);
    assert_eq!(steps[0]["usage"]["inputTokens"], 5);
    assert_eq!(step_content_types(&steps, 0), ["tool-call", "tool-result"]);
}

#[tokio::test]
async fn waits_for_the_echo_of_host_tool_results_before_pausing() {
    let local = tool(&[("city", "string")]).with_execute(|_, _| async {
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(json!({ "local": true }))
    });
    let session = scripted_session(
        |emit| {
            emit(p(
                json!({ "type": "stream-start", "modelId": "fake-model" }),
            ));
            emit(call("local-1", "local", json!({ "city": "SF" })));
            emit(call("c1", "weather", json!({ "city": "SF" })));
            emit(boundary(2));
        },
        |submission, emit| {
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                emit(p(json!({
                    "type": "tool-result", "toolCallId": submission.tool_call_id,
                    "toolName": "local", "result": submission.output,
                })));
            });
        },
    );
    let mut run = base_input(
        fake_harness(),
        session,
        tools(vec![("weather", weather()), ("local", local)]),
    );
    run.tool_approval = Some(approval_config(json!({ "weather": "user-approval" })));
    let (parts, output) = run_collect(run).await;

    assert_eq!(
        of_type(&parts, "tool-result")
            .iter()
            .map(|p| p["toolCallId"].clone())
            .collect::<Vec<_>>(),
        [json!("local-1")]
    );
    let steps = to_json(&output.result.steps().await.unwrap());
    assert_eq!(steps.as_array().unwrap().len(), 1);
    assert_eq!(steps[0]["usage"]["inputTokens"], 2);
    assert_eq!(
        step_content_types(&steps, 0),
        [
            "tool-call",
            "tool-call",
            "tool-approval-request",
            "tool-result"
        ]
    );
}

#[tokio::test]
async fn pauses_for_a_host_approval_before_waiting_for_another_host_result_echo() {
    let (gated_write, gated_calls) = recording_tool(z_object(&[("value", "string")]), |i| {
        Ok(json!({ "value": i["value"] }))
    });
    let (allowed_write, allowed_calls) = recording_tool(z_object(&[("value", "string")]), |i| {
        Ok(json!({ "value": i["value"] }))
    });
    let submitted: Arc<Mutex<Vec<HarnessV1SubmitToolResult>>> = Arc::default();
    let echo_received = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let pending_before_echo = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let echo_timer: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>> = Arc::default();
    let (log, received, timer) = (submitted.clone(), echo_received.clone(), echo_timer.clone());
    let session = scripted_session(
        |emit| {
            emit(p(
                json!({ "type": "stream-start", "modelId": "fake-model" }),
            ));
            emit(call("gated-1", "gated_write", json!({ "value": "gated" })));
            emit(call(
                "allowed-1",
                "allowed_write",
                json!({ "value": "allowed" }),
            ));
            emit(boundary(6));
        },
        move |submission, emit| {
            log.lock().push(submission.clone());
            let received = received.clone();
            *timer.lock() = Some(tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                received.store(true, Ordering::SeqCst);
                emit(p(json!({
                    "type": "tool-result", "toolCallId": submission.tool_call_id,
                    "toolName": "allowed_write", "result": submission.output,
                })));
            }));
        },
    );
    let mut run = base_input(
        fake_harness(),
        session,
        tools(vec![
            ("gated_write", gated_write),
            ("allowed_write", allowed_write),
        ]),
    );
    run.tool_approval = Some(approval_config(
        json!({ "gated_write": "user-approval", "allowed_write": "approved" }),
    ));
    let pending_approvals: Arc<Mutex<Vec<HarnessV1PendingToolApproval>>> = Arc::default();
    let log = pending_approvals.clone();
    run.on_pending_tool_approval = Some(Arc::new(move |a| log.lock().push(a)));
    let pending_results: Arc<Mutex<Vec<HarnessV1PendingToolResult>>> = Arc::default();
    let (log, received, before) = (
        pending_results.clone(),
        echo_received.clone(),
        pending_before_echo.clone(),
    );
    run.on_pending_tool_result = Some(Arc::new(move |r| {
        log.lock().push(r);
        if !received.load(Ordering::SeqCst) {
            before.store(true, Ordering::SeqCst);
        }
    }));
    let (parts, _) = run_collect(run).await;
    let echo_was_received = echo_received.load(Ordering::SeqCst);
    if let Some(timer) = echo_timer.lock().take() {
        timer.abort();
    }

    assert!(gated_calls.lock().is_empty());
    assert_eq!(allowed_calls.lock().len(), 1);
    let submitted = submitted_list(&submitted);
    assert_eq!(submitted.len(), 1);
    assert!(matches(
        &submitted[0],
        &json!({ "toolCallId": "allowed-1", "output": { "value": "allowed" } })
    ));
    let approvals = to_json(&*pending_approvals.lock());
    assert_eq!(approvals.as_array().unwrap().len(), 1);
    assert!(matches(
        &approvals[0],
        &json!({ "toolCallId": "gated-1", "toolName": "gated_write", "kind": "custom" })
    ));
    let results = to_json(&*pending_results.lock());
    assert_eq!(results.as_array().unwrap().len(), 1);
    assert!(matches(
        &results[0],
        &json!({
            "toolCallId": "allowed-1",
            "toolName": "allowed_write",
            "input": json!({ "value": "allowed" }).to_string(),
            "completedResult": { "output": { "value": "allowed" } },
        })
    ));
    assert!(pending_before_echo.load(Ordering::SeqCst));
    assert!(!echo_was_received);
    assert_eq!(of_type(&parts, "tool-approval-request").len(), 1);
    assert_eq!(
        of_type(&parts, "tool-result")
            .iter()
            .map(|p| p["toolCallId"].clone())
            .collect::<Vec<_>>(),
        [json!("allowed-1")]
    );
}

type ApprovalLog = Arc<Mutex<Vec<HarnessV1SubmitToolApproval>>>;

/// A continue-turn session for the repeated built-in approval test.
fn continued_session(
    script: Vec<HarnessV1StreamPart>,
    submitted: &Arc<Mutex<Vec<HarnessV1SubmitToolResult>>>,
    approvals: &ApprovalLog,
) -> FakeSession {
    let (submitted, approvals) = (submitted.clone(), approvals.clone());
    FakeSession::new(script)
        .with_submit(move |submission, _| submitted.lock().push(submission))
        .with_approval(move |approval, _| approvals.lock().push(approval))
}

#[tokio::test]
async fn pauses_for_repeated_builtin_approvals_without_a_host_result_echo() {
    for approved in [true, false] {
        for fails in [false, true] {
            for replay in [false, true] {
                repeated_builtin_approvals(approved, fails, replay).await;
            }
        }
    }
}

async fn repeated_builtin_approvals(approved: bool, fails: bool, replay: bool) {
    let label = format!("approved: {approved}, fails: {fails}, replay: {replay}");
    let pending_approvals: Arc<Mutex<Vec<HarnessV1PendingToolApproval>>> = Arc::default();
    let pending_results: Arc<Mutex<Vec<HarnessV1PendingToolResult>>> = Arc::default();
    let submitted: Arc<Mutex<Vec<HarnessV1SubmitToolResult>>> = Arc::default();
    let executions = Arc::new(AtomicUsize::new(0));
    let bash = tool(&[("command", "string")]);
    let counter = executions.clone();
    let list_project_tickets = tool(&[])
        .with_execute(move |_, _| {
            counter.fetch_add(1, Ordering::SeqCst);
            async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                if fails {
                    return Err(plain_error("Ticket lookup failed"));
                }
                Ok(json!([{ "id": "AI-23503" }]))
            }
        })
        .with_to_model_output(|options| async move {
            Ok(serde_json::from_value(
                json!({ "type": "json", "value": { "tickets": options.output } }),
            )
            .unwrap())
        });
    let approval_harness =
        FakeHarness::with_builtin_approvals("fake", &tools(vec![("bash", bash.clone())]));
    let tool_set = || {
        tools(vec![
            ("bash", bash.clone()),
            ("listProjectTickets", list_project_tickets.clone()),
        ])
    };
    let bash_call = |id: &str| {
        p(json!({
            "type": "tool-call", "toolCallId": id, "toolName": "bash", "nativeName": "Bash",
            "input": r#"{"command":"pwd"}"#, "providerExecuted": true,
        }))
    };
    let host_call = || {
        p(json!({
            "type": "tool-call", "toolCallId": "host-1", "toolName": "listProjectTickets", "input": "{}",
            "providerExecuted": false, "providerMetadata": { "test": { "source": "original-call" } },
        }))
    };
    let approval_request = |id: &str| {
        p(json!({ "type": "tool-approval-request", "approvalId": id, "toolCallId": id }))
    };
    let input = |harness: Arc<FakeHarness>, session: FakeSession| {
        let mut run = RunPromptInput::new(harness, Arc::new(session), sandbox(), WORK_DIR);
        run.tools = tool_set();
        run
    };

    // The first turn pauses for the Bash approval.
    let (start_call, start_host, start_request) =
        (bash_call("bash-1"), host_call(), approval_request("bash-1"));
    let log = submitted.clone();
    let session = scripted_session(
        move |emit| {
            emit(p(
                json!({ "type": "stream-start", "modelId": "fake-model" }),
            ));
            emit(start_call.clone());
            emit(start_host.clone());
            emit(start_request.clone());
            emit(boundary(7));
        },
        move |submission, _| log.lock().push(submission),
    );
    let mut paused = input(approval_harness.clone(), session);
    paused.prompt = Some("Run Bash and list the project tickets.".into());
    paused.tool_approval = Some(approval_config(json!({ "listProjectTickets": "approved" })));
    let log = pending_approvals.clone();
    paused.on_pending_tool_approval = Some(Arc::new(move |a| log.lock().push(a)));
    let log = pending_results.clone();
    paused.on_pending_tool_result = Some(Arc::new(move |r| log.lock().push(r)));
    let (paused_parts, output) = run_collect(paused).await;

    assert_eq!(executions.load(Ordering::SeqCst), 1, "{label}");
    assert_eq!(submitted.lock().len(), 1, "{label}");
    let first_submission = submitted.lock()[0].clone();
    let approvals_json = to_json(&*pending_approvals.lock());
    assert_eq!(approvals_json.as_array().unwrap().len(), 1, "{label}");
    assert!(matches(
        &approvals_json[0],
        &json!({ "approvalId": "bash-1", "toolCallId": "bash-1", "kind": "builtin" })
    ));
    let results_json = to_json(&*pending_results.lock());
    assert_eq!(results_json.as_array().unwrap().len(), 1, "{label}");
    assert!(matches(
        &results_json[0],
        &json!({
            "toolCallId": "host-1",
            "providerOptions": { "test": { "source": "original-call" } },
            "completedResult": { "output": first_submission.output },
        })
    ));
    assert_eq!(
        of_type(&paused_parts, "tool-approval-request").len(),
        1,
        "{label}"
    );
    let outcome_type = if fails { "tool-error" } else { "tool-result" };
    let outcomes = of_type(&paused_parts, outcome_type);
    assert_eq!(outcomes.len(), 1, "{label}: {paused_parts:?}");
    assert!(matches(outcomes[0], &json!({ "toolCallId": "host-1" })));
    let steps = to_json(&output.result.steps().await.unwrap());
    assert_eq!(steps.as_array().unwrap().len(), 1, "{label}");
    assert_eq!(steps[0]["usage"]["inputTokens"], 7);
    assert_eq!(steps[0]["finishReason"], "tool-calls");
    assert_eq!(
        step_content_types(&steps, 0),
        [
            "tool-call",
            "tool-call",
            "tool-approval-request",
            outcome_type
        ],
        "{label}"
    );

    // The continued turn pauses again for a second Bash approval.
    let settled_results: Log = Arc::default();
    let next_approvals: Arc<Mutex<Vec<HarnessV1PendingToolApproval>>> = Arc::default();
    let next_results: Arc<Mutex<Vec<HarnessV1PendingToolResult>>> = Arc::default();
    let tool_approvals: ApprovalLog = Arc::default();
    let mut script = Vec::new();
    if replay {
        script.push(host_call());
    }
    script.extend([bash_call("bash-2"), approval_request("bash-2"), boundary(3)]);
    let mut paused_again = input(
        approval_harness.clone(),
        continued_session(script, &submitted, &tool_approvals),
    );
    paused_again.mode = RunPromptMode::Continue;
    paused_again.pending_tool_approvals = pending_approvals.lock().clone();
    paused_again.pending_tool_results = pending_results.lock().clone();
    paused_again.tool_approval_continuations = vec![approval_response("bash-1", approved)];
    let log = next_approvals.clone();
    paused_again.on_pending_tool_approval = Some(Arc::new(move |a| log.lock().push(a)));
    let log = next_results.clone();
    paused_again.on_pending_tool_result = Some(Arc::new(move |r| log.lock().push(r)));
    let log = settled_results.clone();
    paused_again.on_tool_result_settled = Some(Arc::new(move |id| log.lock().push(id)));
    let (paused_again_parts, _) = run_collect(paused_again).await;

    assert_eq!(*next_results.lock(), *pending_results.lock(), "{label}");
    let next_json = to_json(&*next_approvals.lock());
    assert_eq!(next_json.as_array().unwrap().len(), 1, "{label}");
    assert!(matches(
        &next_json[0],
        &json!({ "approvalId": "bash-2", "kind": "builtin" })
    ));
    assert!(
        !paused_again_parts
            .iter()
            .any(|p| p["toolCallId"] == "host-1"),
        "{label}: {paused_again_parts:?}"
    );
    assert_eq!(executions.load(Ordering::SeqCst), 1, "{label}");
    assert_eq!(submitted.lock().len(), 2, "{label}");

    // The resumed turn settles both approvals and finishes.
    let host_result = || {
        let mut part = json!({
            "type": "tool-result", "toolCallId": "host-1", "toolName": "listProjectTickets",
            "result": first_submission.output,
        });
        if fails {
            part["isError"] = json!(true);
        }
        p(part)
    };
    let mut script = vec![
        bash_call("bash-2"),
        host_call(),
        approval_request("bash-2"),
        p(json!({
            "type": "tool-result", "toolCallId": "bash-2", "toolName": "bash",
            "result": if approved { WORK_DIR } else { "Denied" },
        })),
        host_result(),
        host_result(),
        host_call(),
        boundary(0),
    ];
    script.extend(parts(vec![
        json!({ "type": "text-start", "id": "t1" }),
        json!({ "type": "text-delta", "id": "t1", "delta": "Done" }),
        json!({ "type": "text-end", "id": "t1" }),
    ]));
    let mut resumed = input(
        approval_harness,
        continued_session(with_finish(script), &submitted, &tool_approvals),
    );
    resumed.mode = RunPromptMode::Continue;
    resumed.pending_tool_approvals = next_approvals.lock().clone();
    resumed.pending_tool_results = next_results.lock().clone();
    resumed.tool_approval_continuations = vec![approval_response("bash-2", approved)];
    let log = settled_results.clone();
    resumed.on_tool_result_settled = Some(Arc::new(move |id| log.lock().push(id)));
    let (resumed_parts, resumed_output) = run_collect(resumed).await;

    let approval = |id: &str| HarnessV1SubmitToolApproval {
        approval_id: id.into(),
        approved,
        reason: None,
    };
    assert!(
        tool_approvals.lock().contains(&approval("bash-1")),
        "{label}"
    );
    assert_eq!(executions.load(Ordering::SeqCst), 1, "{label}");
    assert!(
        tool_approvals.lock().contains(&approval("bash-2")),
        "{label}"
    );
    assert_eq!(*settled_results.lock(), ["host-1", "host-1"], "{label}");
    assert_eq!(
        *submitted.lock(),
        vec![
            first_submission.clone(),
            first_submission.clone(),
            first_submission
        ],
        "{label}"
    );
    if !fails {
        let completed = pending_results.lock()[0].completed_result.clone().unwrap();
        assert_eq!(
            to_json(&completed.tool_result.unwrap().output),
            json!({ "type": "json", "value": { "tickets": [{ "id": "AI-23503" }] } })
        );
    }
    assert!(
        !resumed_parts
            .iter()
            .any(|p| p["type"] == "tool-call" || p["type"] == "tool-approval-request"),
        "{label}: {resumed_parts:?}"
    );
    assert!(
        !resumed_parts.iter().any(
            |p| (p["type"] == "tool-result" || p["type"] == "tool-error")
                && p["toolCallId"] == "host-1"
        ),
        "{label}"
    );
    assert_eq!(resumed_output.result.text().await.unwrap(), "Done");
    assert_eq!(
        to_json(&resumed_output.result.finish_reason().await.unwrap()),
        "stop"
    );
}

#[tokio::test]
async fn closes_a_step_whose_finish_step_was_ignored_when_the_turn_finishes() {
    let run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(vec![call(
            "c1",
            "weather",
            json!({ "city": "SF" }),
        )])),
        tools(vec![("weather", weather())]),
    );
    let (_, output) = run_collect(run).await;

    let steps = to_json(&output.result.steps().await.unwrap());
    assert_eq!(steps.as_array().unwrap().len(), 1);
    assert_eq!(step_content_types(&steps, 0), ["tool-call"]);
    assert_eq!(
        to_json(&output.result.finish_reason().await.unwrap()),
        "stop"
    );
}
