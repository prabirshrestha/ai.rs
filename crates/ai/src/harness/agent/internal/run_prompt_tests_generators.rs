//! `run-prompt.test.ts`: `runPrompt host tool generator results`.

use super::tool_outputs::{
    Calls, approval_config, approval_response, custom_approval, pending_result, recording_tool,
    submitted_list,
};
use super::*;

fn call(tool_call_id: &str, tool_name: &str, input: JsonValue) -> HarnessV1StreamPart {
    let input = match input {
        JsonValue::String(s) => s,
        other => other.to_string(),
    };
    p(
        json!({ "type": "tool-call", "toolCallId": tool_call_id, "toolName": tool_name, "input": input }),
    )
}

fn is_input_part(part: &JsonValue) -> bool {
    matches!(
        part["type"].as_str(),
        Some("tool-input-start" | "tool-input-delta" | "tool-input-end" | "tool-call")
    )
}

fn weather_tool() -> (Tool, Calls) {
    recording_tool(z_object(&[("city", "string")]), |input| {
        Ok(json!({ "city": input["city"], "temperature": 72 }))
    })
}

/// `executeTool` and tool lifecycle events of a telemetry integration.
#[derive(Default)]
struct WrapperTelemetry {
    events: Log,
    call_ids: Log,
    fail: bool,
    /// Whether the integration implements `onToolExecutionEnd`.
    on_end: bool,
}

impl Telemetry for WrapperTelemetry {
    fn on_tool_execution_start(&self, event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let (events, call_ids) = (self.events.clone(), self.call_ids.clone());
        let call_id = event["callId"].as_str().unwrap_or_default().to_string();
        Box::pin(async move {
            tokio::task::yield_now().await;
            call_ids.lock().push(call_id);
            events.lock().push("tool-start".into());
        })
    }
    fn on_tool_execution_end(&self, _event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        if self.on_end {
            self.events.lock().push("tool-end".into());
        }
        Box::pin(async {})
    }
    fn execute_tool(
        &self,
        event: &TelemetryEvent,
        execute: crate::harness::sdk::telemetry::ExecuteThunk,
    ) -> BoxFuture<'static, HarnessResult<JsonValue>> {
        if self.fail {
            return Box::pin(async { Err(plain_error("telemetry wrapper failed")) });
        }
        self.call_ids
            .lock()
            .push(event["callId"].as_str().unwrap_or_default().to_string());
        assert_eq!(event["toolCallId"], "c1");
        self.events.lock().push("wrapper-start".into());
        let events = self.events.clone();
        Box::pin(async move {
            let output = execute().await;
            events.lock().push("wrapper-end".into());
            output
        })
    }
    fn supports_execute_tool(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn suppresses_replayed_tool_input_for_settled_host_calls() {
    let (session, submitted) = FakeSession::recording(with_finish(parts(vec![
        json!({ "type": "tool-input-start", "id": "c1", "toolName": "weather", "providerExecuted": false }),
        json!({ "type": "tool-input-delta", "id": "c1", "delta": r#"{"city":"SF"}"# }),
        json!({ "type": "tool-input-end", "id": "c1" }),
        json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "weather", "input": r#"{"city":"SF"}"# }),
    ])));
    let mut run = base_input(fake_harness(), session, ToolSet::new());
    run.mode = RunPromptMode::Continue;
    run.prompt = None;
    run.pending_tool_results = vec![pending_result("c1", "weather", r#"{"city":"SF"}"#)];
    run.tool_result_continuations = vec![
        serde_json::from_value(json!({
            "toolCallId": "c1",
            "toolName": "weather",
            "output": { "type": "json", "value": { "city": "SF", "temperature": 72 } },
        }))
        .unwrap(),
    ];
    let (parts, _) = run_collect(run).await;

    assert_eq!(
        submitted_list(&submitted),
        vec![json!({
            "toolCallId": "c1",
            "output": { "city": "SF", "temperature": 72 },
            "toolResult": {
                "type": "tool-result",
                "toolCallId": "c1",
                "toolName": "weather",
                "output": { "type": "json", "value": { "city": "SF", "temperature": 72 } },
            },
        })]
    );
    assert!(!parts.iter().any(is_input_part), "{parts:?}");
}

#[tokio::test]
async fn executes_independent_host_tool_calls_concurrently() {
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let second_started = Gate::default();
    let start_tool = {
        let (active, max_active) = (active.clone(), max_active.clone());
        move || {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(now, Ordering::SeqCst);
        }
    };
    let first = {
        let (start_tool, active, observed, gate) = (
            start_tool.clone(),
            active.clone(),
            observed.clone(),
            second_started.clone(),
        );
        tool(&[])
            .with_description("First independent tool")
            .with_execute(move |_, _| {
                start_tool();
                let (active, observed, gate) = (active.clone(), observed.clone(), gate.clone());
                async move {
                    let saw = tokio::time::timeout(Duration::from_millis(100), gate.wait())
                        .await
                        .is_ok();
                    observed.store(saw, Ordering::SeqCst);
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(json!({ "tool": "first" }))
                }
            })
    };
    let second = {
        let (active, gate) = (active.clone(), second_started.clone());
        tool(&[])
            .with_description("Second independent tool")
            .with_execute(move |_, _| {
                start_tool();
                gate.open();
                active.fetch_sub(1, Ordering::SeqCst);
                async { Ok(json!({ "tool": "second" })) }
            })
    };
    let (session, submitted) = FakeSession::recording(with_finish(vec![
        call("c1", "first", json!("{}")),
        call("c2", "second", json!("{}")),
    ]));
    run_collect(base_input(
        fake_harness(),
        session,
        tools(vec![("first", first), ("second", second)]),
    ))
    .await;

    assert!(observed.load(Ordering::SeqCst));
    assert_eq!(max_active.load(Ordering::SeqCst), 2);
    let mut ids: Vec<String> = submitted
        .lock()
        .iter()
        .map(|s| s.tool_call_id.clone())
        .collect();
    ids.sort();
    assert_eq!(ids, ["c1", "c2"]);
}

#[tokio::test]
async fn waits_for_every_concurrent_host_tool_when_result_submission_fails() {
    let submitted_ids: Log = Arc::default();
    let slow_can_finish = Gate::default();
    let slow_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let failed_submission = Gate::default();
    let attempts = Arc::new(AtomicUsize::new(0));
    let fast = tool(&[]).with_execute(|_, _| async { Ok(json!({ "tool": "fast" })) });
    let slow = {
        let (gate, finished) = (slow_can_finish.clone(), slow_finished.clone());
        tool(&[]).with_execute(move |_, _| {
            let (gate, finished) = (gate.clone(), finished.clone());
            async move {
                gate.wait().await;
                finished.store(true, Ordering::SeqCst);
                Ok(json!({ "tool": "slow" }))
            }
        })
    };
    let mut session = FakeSession::new(with_finish(vec![
        call("fast-call", "fast", json!("{}")),
        call("slow-call", "slow", json!("{}")),
    ]));
    let (ids, failed, attempts_log) = (
        submitted_ids.clone(),
        failed_submission.clone(),
        attempts.clone(),
    );
    session.on_submit = Some(Arc::new(move |input, _| {
        ids.lock().push(input.tool_call_id.clone());
        if input.tool_call_id == "fast-call" {
            if attempts_log.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                failed.open();
            }
            return Box::pin(async { Err(plain_error("result submission failed")) });
        }
        Box::pin(async { Ok(()) })
    }));
    let output = run_prompt(base_input(
        fake_harness(),
        session,
        tools(vec![("fast", fast), ("slow", slow)]),
    ));
    let settled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (done, flag) = (output.done.clone(), settled.clone());
    let watcher = tokio::spawn(async move {
        done.await;
        flag.store(true, Ordering::SeqCst);
    });
    let result = output.result.clone();
    let consume = tokio::spawn(async move { result.consume_stream().await });

    tokio::time::timeout(Duration::from_secs(5), failed_submission.wait())
        .await
        .expect("second failed submission");
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!slow_finished.load(Ordering::SeqCst));
    assert!(!settled.load(Ordering::SeqCst));
    slow_can_finish.open();
    tokio::time::timeout(Duration::from_secs(5), async {
        watcher.await.unwrap();
        consume.await.unwrap();
    })
    .await
    .expect("turn settled");

    assert!(slow_finished.load(Ordering::SeqCst));
    assert!(settled.load(Ordering::SeqCst));
    assert_eq!(
        *submitted_ids.lock(),
        ["fast-call", "fast-call", "slow-call"]
    );
    let err = output.result.finish_reason().await.unwrap_err();
    assert!(
        err.to_string().contains("result submission failed"),
        "{err}"
    );
}

#[tokio::test]
async fn pauses_custom_tool_execution_when_approval_is_required() {
    let stop_condition_calls = Arc::new(AtomicUsize::new(0));
    let stop_boundary_calls = Arc::new(AtomicUsize::new(0));
    let (weather, _) = recording_tool(z_object(&[("city", "string")]), |_| {
        Ok(json!({ "temperature": 72 }))
    });
    let (session, submitted) =
        FakeSession::recording(vec![call("c1", "weather", json!({ "city": "SF" }))]);
    let mut run = base_input(fake_harness(), session, tools(vec![("weather", weather)]));
    run.tool_approval = Some(approval_config(json!({ "weather": "user-approval" })));
    let calls = stop_condition_calls.clone();
    run.stop_conditions = Some(vec![stop_condition(move |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        true
    })]);
    let boundary = stop_boundary_calls.clone();
    run.on_stop_condition_met = Some(Arc::new(move || {
        boundary.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }));
    let pending: Arc<Mutex<Vec<HarnessV1PendingToolApproval>>> = Arc::default();
    let log = pending.clone();
    run.on_pending_tool_approval = Some(Arc::new(move |a| log.lock().push(a)));
    let (parts, output) = run_collect(run).await;

    assert!(submitted.lock().is_empty());
    assert_eq!(
        to_json(&output.result.finish_reason().await.unwrap()),
        "tool-calls"
    );
    let pending = pending.lock().clone();
    assert_eq!(pending.len(), 1);
    assert!(!pending[0].approval_id.is_empty());
    let mut pending_json = to_json(&pending[0]);
    pending_json.as_object_mut().unwrap().remove("approvalId");
    assert_eq!(
        pending_json,
        json!({
            "toolCallId": "c1",
            "toolName": "weather",
            "input": json!({ "city": "SF" }).to_string(),
            "kind": "custom",
            "providerExecuted": false,
        })
    );
    let request = of_type(&parts, "tool-approval-request");
    assert_eq!(request[0]["toolCall"]["toolName"], "weather");
    assert_eq!(request[0]["toolCall"]["input"], json!({ "city": "SF" }));
    let steps = to_json(&output.result.steps().await.unwrap());
    assert_eq!(steps.as_array().unwrap().len(), 1);
    assert_eq!(steps[0]["finishReason"], "tool-calls");
    assert_eq!(
        types(steps[0]["content"].as_array().unwrap()),
        ["tool-call", "tool-approval-request"]
    );
    assert_eq!(stop_condition_calls.load(Ordering::SeqCst), 0);
    assert_eq!(stop_boundary_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn surfaces_every_approval_request_of_a_step_before_pausing() {
    let (weather, _) = recording_tool(z_object(&[("city", "string")]), |_| {
        Ok(json!({ "temperature": 72 }))
    });
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(vec![
            call("c1", "weather", json!({ "city": "SF" })),
            call("c2", "weather", json!({ "city": "NYC" })),
        ]),
        tools(vec![("weather", weather)]),
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
    assert_eq!(
        of_type(&parts, "tool-approval-request")
            .iter()
            .map(|p| p["toolCall"]["toolCallId"].clone())
            .collect::<Vec<_>>(),
        [json!("c1"), json!("c2")]
    );
    let steps = to_json(&output.result.steps().await.unwrap());
    assert_eq!(
        types(steps[0]["content"].as_array().unwrap()),
        [
            "tool-call",
            "tool-approval-request",
            "tool-call",
            "tool-approval-request"
        ]
    );
}

#[tokio::test]
async fn denies_custom_tools_configured_with_denied_approval_status() {
    let (weather, _) = recording_tool(z_object(&[("city", "string")]), |_| {
        Ok(json!({ "temperature": 72 }))
    });
    let (session, submitted) = FakeSession::recording(with_finish(vec![call(
        "c1",
        "weather",
        json!({ "city": "SF" }),
    )]));
    let mut run = base_input(fake_harness(), session, tools(vec![("weather", weather)]));
    run.tool_approval = Some(approval_config(
        json!({ "weather": { "type": "denied", "reason": "weather disabled" } }),
    ));
    let (parts, output) = run_collect(run).await;

    let denied = json!({ "type": "execution-denied", "reason": "weather disabled" });
    assert_eq!(
        submitted_list(&submitted),
        vec![json!({
            "toolCallId": "c1",
            "output": denied,
            "toolResult": { "type": "tool-result", "toolCallId": "c1", "toolName": "weather", "output": denied },
        })]
    );
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-approval-request", "isAutomatic": true })
    ));
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-approval-response", "approved": false, "reason": "weather disabled", "providerExecuted": false })
    ));
    let steps = to_json(&output.result.steps().await.unwrap());
    assert_eq!(steps.as_array().unwrap().len(), 1);
    assert_eq!(
        types(steps[0]["content"].as_array().unwrap()),
        [
            "tool-call",
            "tool-approval-request",
            "tool-approval-response"
        ]
    );
}

#[tokio::test]
async fn emits_one_final_result_after_approved_pending_custom_tool_execution() {
    let (weather, _) = weather_tool();
    let (session, submitted) = FakeSession::recording(parts(vec![
        json!({ "type": "tool-input-start", "id": "c1", "toolName": "weather", "providerExecuted": false }),
        json!({ "type": "tool-input-delta", "id": "c1", "delta": r#"{"city":"SF"}"# }),
        json!({ "type": "tool-input-end", "id": "c1" }),
        json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "weather", "input": r#"{"city":"SF"}"# }),
        json!({ "type": "tool-approval-request", "approvalId": "approval-1", "toolCallId": "c1" }),
        json!({ "type": "tool-result", "toolCallId": "c1", "toolName": "weather", "result": { "city": "SF", "temperature": 72 } }),
    ]));
    let mut run = base_input(fake_harness(), session, tools(vec![("weather", weather)]));
    run.mode = RunPromptMode::Continue;
    run.prompt = None;
    run.pending_tool_approvals = vec![custom_approval(
        "approval-1",
        "c1",
        "weather",
        &json!({ "city": "SF" }).to_string(),
    )];
    run.tool_approval_continuations = vec![approval_response("approval-1", true)];
    let settled: Log = Arc::default();
    let log = settled.clone();
    run.on_tool_approval_settled = Some(Arc::new(move |id| log.lock().push(id)));
    let telemetry = Arc::new(WrapperTelemetry {
        on_end: true,
        ..Default::default()
    });
    run.telemetry = Some(telemetry_with(telemetry.clone()));
    let (parts, output) = run_collect(run).await;

    assert_eq!(*settled.lock(), ["approval-1"]);
    assert_eq!(
        submitted_list(&submitted),
        vec![json!({ "toolCallId": "c1", "output": { "city": "SF", "temperature": 72 } })]
    );
    assert_eq!(
        *telemetry.events.lock(),
        ["tool-start", "wrapper-start", "wrapper-end", "tool-end"]
    );
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-approval-response", "approvalId": "approval-1", "approved": true })
    ));
    let results = of_type(&parts, "tool-result");
    assert_eq!(results.len(), 1);
    assert!(matches(
        results[0],
        &json!({ "toolCallId": "c1", "toolName": "weather", "output": { "city": "SF", "temperature": 72 } })
    ));
    assert!(
        !parts
            .iter()
            .any(|p| is_input_part(p) || p["type"] == "tool-approval-request")
    );
    assert!(of_type(&parts, "error").is_empty());
    assert!(output.result.steps().await.unwrap().is_empty());
}

#[tokio::test]
async fn emits_the_provider_result_after_approved_pending_builtin_tool_execution() {
    let approvals: Arc<Mutex<Vec<HarnessV1SubmitToolApproval>>> = Arc::default();
    let log = approvals.clone();
    let session = FakeSession::new(with_finish(parts(vec![
        json!({ "type": "tool-input-start", "id": "c1", "toolName": "bash", "providerExecuted": true }),
        json!({ "type": "tool-input-delta", "id": "c1", "delta": r#"{"command":"printf ok"}"# }),
        json!({ "type": "tool-input-end", "id": "c1" }),
        json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "bash", "input": json!({ "command": "printf ok" }).to_string(), "providerExecuted": true }),
        json!({ "type": "tool-approval-request", "approvalId": "approval-1", "toolCallId": "c1" }),
        json!({ "type": "tool-result", "toolCallId": "c1", "toolName": "bash", "result": "ok" }),
    ])))
    .with_approval(move |input, _| log.lock().push(input));
    let mut run = base_input(fake_harness(), session, ToolSet::new());
    run.mode = RunPromptMode::Continue;
    run.prompt = None;
    run.pending_tool_approvals = vec![
        serde_json::from_value(json!({
            "approvalId": "approval-1",
            "toolCallId": "c1",
            "toolName": "bash",
            "input": json!({ "command": "printf ok" }).to_string(),
            "kind": "builtin",
            "providerExecuted": true,
        }))
        .unwrap(),
    ];
    run.tool_approval_continuations = vec![approval_response("approval-1", true)];
    let (parts, _) = run_collect(run).await;

    assert_eq!(
        *approvals.lock(),
        vec![HarnessV1SubmitToolApproval {
            approval_id: "approval-1".into(),
            approved: true,
            reason: None,
        }]
    );
    assert!(
        !parts
            .iter()
            .any(|p| is_input_part(p) || p["type"] == "tool-approval-request")
    );
    let results = of_type(&parts, "tool-result");
    assert_eq!(results.len(), 1);
    assert!(matches(
        results[0],
        &json!({ "toolCallId": "c1", "toolName": "bash", "output": "ok" })
    ));
}

#[tokio::test]
async fn emits_an_error_after_approved_pending_custom_tool_execution_fails() {
    let weather = tool(&[("city", "string")])
        .with_execute(|_, _| async { Err::<JsonValue, _>(plain_error("weather unavailable")) });
    let (session, submitted) = FakeSession::recording(Vec::new());
    let mut run = base_input(fake_harness(), session, tools(vec![("weather", weather)]));
    run.mode = RunPromptMode::Continue;
    run.prompt = None;
    run.pending_tool_approvals = vec![custom_approval(
        "approval-1",
        "c1",
        "weather",
        &json!({ "city": "SF" }).to_string(),
    )];
    run.tool_approval_continuations = vec![approval_response("approval-1", true)];
    let (parts, _) = run_collect(run).await;

    assert_eq!(
        submitted_list(&submitted),
        vec![
            json!({ "toolCallId": "c1", "output": { "error": "Error: weather unavailable" }, "isError": true })
        ]
    );
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-error", "toolCallId": "c1", "toolName": "weather", "error": { "message": "weather unavailable" } })
    ));
}

#[tokio::test]
async fn does_not_reuse_a_consumed_approval_for_replayed_custom_tool_calls() {
    let (weather, _) = weather_tool();
    let (session, submitted) = FakeSession::recording(vec![
        call("c1", "weather", json!({ "city": "SF" })),
        call("c2", "weather", json!({ "city": "Austin" })),
    ]);
    let mut run = base_input(fake_harness(), session, tools(vec![("weather", weather)]));
    run.mode = RunPromptMode::Continue;
    run.prompt = None;
    run.tool_approval = Some(approval_config(json!({ "weather": "user-approval" })));
    run.pending_tool_approvals = vec![custom_approval(
        "approval-1",
        "c1",
        "weather",
        &json!({ "city": "SF" }).to_string(),
    )];
    run.tool_approval_continuations = vec![approval_response("approval-1", true)];
    let pending: Arc<Mutex<Vec<HarnessV1PendingToolApproval>>> = Arc::default();
    let log = pending.clone();
    run.on_pending_tool_approval = Some(Arc::new(move |a| log.lock().push(a)));
    let settled: Log = Arc::default();
    let log = settled.clone();
    run.on_tool_approval_settled = Some(Arc::new(move |id| log.lock().push(id)));
    let (parts, _) = run_collect(run).await;

    assert_eq!(*settled.lock(), ["approval-1"]);
    assert_eq!(
        submitted_list(&submitted),
        vec![json!({ "toolCallId": "c1", "output": { "city": "SF", "temperature": 72 } })]
    );
    let pending = pending.lock().clone();
    assert_eq!(pending.len(), 1);
    assert!(!pending[0].approval_id.is_empty());
    assert!(matches(
        &to_json(&pending[0]),
        &json!({
            "toolCallId": "c2",
            "toolName": "weather",
            "input": json!({ "city": "Austin" }).to_string(),
            "kind": "custom",
            "providerExecuted": false,
        })
    ));
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-approval-request", "toolCall": { "toolCallId": "c2" } })
    ));
}

#[tokio::test]
async fn surfaces_each_generator_yield_as_a_preliminary_result_and_submits_the_last_yield() {
    let weather = tool(&[("city", "string")])
        .with_description("Get the weather")
        .with_streaming_execute(|_, _| {
            futures::stream::iter([
                Ok(json!({ "state": "loading" })),
                Ok(json!({ "state": "ready", "temperature": 72 })),
            ])
            .boxed()
        });
    let (session, submitted) = FakeSession::recording(with_finish(vec![call(
        "c1",
        "weather",
        json!({ "city": "SF" }),
    )]));
    let (parts, _) = run_collect(base_input(
        fake_harness(),
        session,
        tools(vec![("weather", weather)]),
    ))
    .await;

    // Every `yield` is surfaced as a preliminary result, including the last,
    // which `executeTool` re-emits as the final value. The non-preliminary
    // final result reaches consumers only when the adapter echoes the
    // submitted result back, which the fake session does not model.
    let results = of_type(&parts, "tool-result");
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r["preliminary"] == true));
    assert_eq!(results[0]["output"], json!({ "state": "loading" }));
    assert_eq!(
        results[1]["output"],
        json!({ "state": "ready", "temperature": 72 })
    );
    // Only the final value is submitted back to the runtime.
    assert_eq!(
        submitted_list(&submitted),
        vec![json!({ "toolCallId": "c1", "output": { "state": "ready", "temperature": 72 } })]
    );
}

#[tokio::test]
async fn submits_a_plain_execute_return_value_with_no_preliminary_results() {
    let (echo, _) = recording_tool(z_object(&[("text", "string")]), |input| {
        Ok(json!({ "echoed": input["text"] }))
    });
    let (session, submitted) = FakeSession::recording(with_finish(vec![call(
        "c1",
        "echo",
        json!({ "text": "hi" }),
    )]));
    let (parts, _) = run_collect(base_input(
        fake_harness(),
        session,
        tools(vec![("echo", echo)]),
    ))
    .await;

    assert!(of_type(&parts, "tool-result").is_empty());
    assert_eq!(
        submitted_list(&submitted),
        vec![json!({ "toolCallId": "c1", "output": { "echoed": "hi" } })]
    );
}

#[tokio::test]
async fn executes_host_tools_through_telemetry_context_wrappers() {
    let telemetry = Arc::new(WrapperTelemetry::default());
    let events = telemetry.events.clone();
    let echo = tool(&[("text", "string")]).with_execute(move |input, _| {
        events.lock().push("execute".into());
        async move { Ok(json!({ "echoed": input["text"] })) }
    });
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(vec![call(
            "c1",
            "echo",
            json!({ "text": "hi" }),
        )])),
        tools(vec![("echo", echo)]),
    );
    run.telemetry = Some(telemetry_with(telemetry.clone()));
    run_collect(run).await;

    assert_eq!(
        *telemetry.events.lock(),
        ["tool-start", "wrapper-start", "execute", "wrapper-end"]
    );
    let call_ids: HashSet<String> = telemetry.call_ids.lock().iter().cloned().collect();
    assert_eq!(call_ids.len(), 1);
}

#[tokio::test]
async fn reports_telemetry_wrapper_failures_as_tool_errors() {
    let (echo, calls) = recording_tool(z_object(&[("text", "string")]), |_| Ok(JsonValue::Null));
    let (session, submitted) = FakeSession::recording(with_finish(vec![call(
        "c1",
        "echo",
        json!({ "text": "hi" }),
    )]));
    let mut run = base_input(fake_harness(), session, tools(vec![("echo", echo)]));
    run.telemetry = Some(telemetry_with(Arc::new(WrapperTelemetry {
        fail: true,
        ..Default::default()
    })));
    run_collect(run).await;

    assert!(calls.lock().is_empty());
    assert_eq!(
        submitted_list(&submitted),
        vec![
            json!({ "toolCallId": "c1", "output": { "error": "Error: telemetry wrapper failed" }, "isError": true })
        ]
    );
}

#[tokio::test]
async fn strips_the_work_dir_from_preliminary_results_before_they_reach_consumers() {
    let find = tool(&[("name", "string")]).with_streaming_execute(|_, _| {
        futures::stream::iter([Ok(json!({ "path": format!("{WORK_DIR}/src/foo.ts") }))]).boxed()
    });
    let (parts, _) = run_collect(base_input(
        fake_harness(),
        FakeSession::new(with_finish(vec![call(
            "c1",
            "find",
            json!({ "name": "foo.ts" }),
        )])),
        tools(vec![("find", find)]),
    ))
    .await;

    let results = of_type(&parts, "tool-result");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["preliminary"], true);
    assert_eq!(results[0]["output"], json!({ "path": "src/foo.ts" }));
}
