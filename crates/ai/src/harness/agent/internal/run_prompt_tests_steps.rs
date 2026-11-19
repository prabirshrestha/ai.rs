//! `run-prompt.test.ts`: `runPrompt step accounting`.

use super::*;

#[tokio::test]
async fn preserves_adapter_warnings_on_the_step_and_aggregate_result() {
    let warnings = json!([
        { "type": "unsupported-setting", "setting": "temperature", "details": "The adapter does not support temperature." },
        { "type": "unsupported-tool", "tool": "web_search", "details": "The adapter does not support web search." },
        { "type": "other", "message": "The adapter used a fallback." },
    ]);
    let run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(parts(vec![
            json!({ "type": "stream-start", "warnings": warnings }),
            json!({ "type": "text-delta", "id": "t1", "delta": "done" }),
        ]))),
        ToolSet::new(),
    );
    let (parts, output) = run_collect(run).await;
    assert_eq!(of_type(&parts, "start-step")[0]["warnings"], warnings);
    let steps = output.result.steps().await.unwrap();
    assert_eq!(to_json(&steps[0].warnings), warnings);
    assert_eq!(to_json(&output.result.warnings().await.unwrap()), warnings);
}

#[tokio::test]
async fn records_one_step_per_finish_step_without_counting_terminal_finish() {
    let mut script = parts(vec![json!({ "type": "text-delta", "id": "t1", "delta": "first" })]);
    script.push(resumable_finish_step());
    script.push(p(json!({ "type": "text-delta", "id": "t2", "delta": "second" })));
    let run = base_input(fake_harness(), FakeSession::new(with_finish(script)), ToolSet::new());
    let (_, output) = run_collect(run).await;
    let steps = output.result.steps().await.unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps.iter().map(|s| s.step_number).collect::<Vec<_>>(), [0, 1]);
    assert_eq!(steps.iter().map(|s| s.text()).collect::<Vec<_>>(), ["first", "second"]);
    assert!(is_step_count(2)(&steps).await);
}

#[tokio::test]
async fn keeps_tool_calls_and_results_in_one_predicate_compatible_step() {
    let weather = tool(&[("city", "string")])
        .with_description("Get weather")
        .with_execute(|_, _| async { Ok(json!({ "temperature": 72 })) });
    let run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(parts(vec![
            json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "weather", "input": "{\"city\":\"SF\"}" }),
            json!({ "type": "tool-result", "toolCallId": "c1", "toolName": "weather", "result": { "temperature": 72 } }),
        ]))),
        tools(vec![("weather", weather)]),
    );
    let (_, output) = run_collect(run).await;
    let steps = output.result.steps().await.unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0].content.iter().map(|p| p.type_name()).collect::<Vec<_>>(),
        ["tool-call", "tool-result"]
    );
    assert_eq!(steps[0].tool_calls().len(), 1);
    assert_eq!(steps[0].tool_results().len(), 1);
    assert!(has_tool_call(vec!["weather".into()])(&steps).await);
}

#[tokio::test]
async fn preserves_input_and_execution_origin_on_interleaved_tool_results() {
    let host_lookup =
        tool(&[("query", "string")]).with_execute(|_, _| async { Ok(json!("host found")) });
    let run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(parts(vec![
            json!({ "type": "tool-call", "toolCallId": "native-success", "toolName": "nativeLookup", "input": "{\"query\":\"native success\"}", "providerExecuted": true, "dynamic": true }),
            json!({ "type": "tool-call", "toolCallId": "host-success", "toolName": "hostLookup", "input": "{\"query\":\"host success\"}", "providerExecuted": false }),
            json!({ "type": "tool-call", "toolCallId": "native-failure", "toolName": "nativeLookup", "input": "{\"query\":\"native failure\"}", "providerExecuted": true, "dynamic": true }),
            json!({ "type": "tool-result", "toolCallId": "host-success", "toolName": "hostLookup", "result": "host found" }),
            json!({ "type": "tool-result", "toolCallId": "native-failure", "toolName": "nativeLookup", "result": "native failed", "isError": true, "dynamic": true }),
            json!({ "type": "tool-result", "toolCallId": "native-success", "toolName": "nativeLookup", "result": "native found", "dynamic": true }),
        ]))),
        tools(vec![("hostLookup", host_lookup)]),
    );
    let (parts, output) = run_collect(run).await;
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-result", "toolCallId": "native-success", "input": { "query": "native success" }, "output": "native found", "providerExecuted": true })
    ));
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-result", "toolCallId": "host-success", "input": { "query": "host success" }, "output": "host found", "providerExecuted": false })
    ));
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-error", "toolCallId": "native-failure", "input": { "query": "native failure" }, "error": "native failed", "providerExecuted": true })
    ));
    let results = to_json(&output.result.tool_results().await.unwrap());
    let results = results.as_array().unwrap();
    assert!(contains_matching(
        results,
        json!({ "toolCallId": "native-success", "input": { "query": "native success" }, "providerExecuted": true })
    ));
    assert!(contains_matching(
        results,
        json!({ "toolCallId": "host-success", "input": { "query": "host success" }, "providerExecuted": false })
    ));
    let messages = to_json(&output.result.response_messages().await.unwrap());
    let messages = messages.as_array().unwrap();
    let assistant = messages.iter().find(|m| m["role"] == "assistant").unwrap();
    let tool_message = messages.iter().find(|m| m["role"] == "tool").unwrap();
    let has = |m: &JsonValue, id: &str| {
        contains_matching(
            m["content"].as_array().unwrap(),
            json!({ "type": "tool-result", "toolCallId": id }),
        )
    };
    assert!(has(assistant, "native-success"));
    assert!(has(tool_message, "host-success"));
    assert!(!has(tool_message, "native-success"));
}

#[tokio::test]
async fn surfaces_a_failed_provider_executed_tool_result_as_a_tool_error_carrying_the_runtime_message()
 {
    let weather = tool(&[("city", "string")]).with_description("Get weather");
    let run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(parts(vec![
            json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "weather", "input": "{\"city\":\"SF\"}", "providerExecuted": true }),
            json!({ "type": "tool-result", "toolCallId": "c1", "toolName": "weather", "result": "weather service unreachable", "isError": true }),
        ]))),
        tools(vec![("weather", weather)]),
    );
    let (parts, output) = run_collect(run).await;
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-error", "toolCallId": "c1", "toolName": "weather", "error": "weather service unreachable", "providerExecuted": true })
    ));
    assert!(!contains_matching(&parts, json!({ "type": "tool-result", "toolCallId": "c1" })));
    let steps = output.result.steps().await.unwrap();
    assert_eq!(
        steps[0].content.iter().map(|p| p.type_name()).collect::<Vec<_>>(),
        ["tool-call", "tool-error"]
    );
}

/// A host tool's failure is echoed back on the same wire event as a
/// provider-executed one; only the originating `tool-call` tells them apart.
async fn stream_failed_host_tool(provider_executed: Option<bool>) -> Vec<JsonValue> {
    let weather = tool(&[("city", "string")])
        .with_description("Get weather")
        .with_execute(|_, _| async { Err::<JsonValue, BoxError>(plain_error("weather unavailable")) });
    let mut call = json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "weather", "input": "{\"city\":\"SF\"}" });
    if let Some(pe) = provider_executed {
        call["providerExecuted"] = json!(pe);
    }
    let run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(parts(vec![
            call,
            json!({ "type": "tool-result", "toolCallId": "c1", "toolName": "weather", "result": { "error": "Error: weather unavailable" }, "isError": true }),
        ]))),
        tools(vec![("weather", weather)]),
    );
    run_collect(run).await.0
}

fn expect_host_tool_failure(parts: &[JsonValue]) {
    // Marking this provider-executed would bypass the consumer's `onError`.
    assert!(!contains_matching(parts, json!({ "type": "tool-error", "toolCallId": "c1" })));
    assert!(contains_matching(
        parts,
        json!({ "type": "tool-result", "toolCallId": "c1", "output": { "error": "Error: weather unavailable" } })
    ));
}

#[tokio::test]
async fn does_not_mark_a_failed_host_tool_result_as_provider_executed() {
    expect_host_tool_failure(&stream_failed_host_tool(Some(false)).await);
}

#[tokio::test]
async fn treats_a_failed_host_tool_result_as_host_executed_when_provider_executed_is_omitted() {
    expect_host_tool_failure(&stream_failed_host_tool(None).await);
}

#[tokio::test]
async fn falls_back_to_provider_executed_when_the_originating_tool_call_is_not_in_this_slice() {
    let run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(parts(vec![json!({
            "type": "tool-result", "toolCallId": "c1", "toolName": "bash",
            "result": "bash: command not found: pnpmm", "isError": true
        })]))),
        ToolSet::new(),
    );
    let (parts, _) = run_collect(run).await;
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-error", "toolCallId": "c1", "error": "bash: command not found: pnpmm", "providerExecuted": true })
    ));
}

#[tokio::test]
async fn does_not_expose_provider_executed_tool_calls_as_pending_client_results() {
    let pending: Arc<Mutex<Vec<HarnessV1PendingToolResult>>> = Arc::default();
    let log = pending.clone();
    let weather = tool(&[("city", "string")]).with_description("Get weather");
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(parts(vec![json!({
            "type": "tool-call", "toolCallId": "c1", "toolName": "weather",
            "input": "{\"city\":\"SF\"}", "providerExecuted": true
        })]))),
        tools(vec![("weather", weather)]),
    );
    run.on_pending_tool_result = Some(Arc::new(move |p| log.lock().push(p)));
    let (_, output) = run_collect(run).await;
    assert!(pending.lock().is_empty());
    assert_eq!(output.result.steps().await.unwrap().len(), 1);
}

fn partial_text_script() -> Vec<HarnessV1StreamPart> {
    parts(vec![
        json!({ "type": "text-start", "id": "t1" }),
        json!({ "type": "text-delta", "id": "t1", "delta": "partial" }),
    ])
}

fn counter() -> (Arc<AtomicUsize>, Arc<dyn Fn() + Send + Sync>) {
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    (
        count,
        Arc::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
        }),
    )
}

#[tokio::test]
async fn closes_a_deliberately_suspended_mid_step_stream_without_failing_the_turn() {
    let (failed, on_turn_failed) = counter();
    let mut run = base_input(fake_harness(), FakeSession::new(partial_text_script()), ToolSet::new());
    run.is_turn_suspending = Some(Arc::new(|| true));
    run.on_turn_failed = Some(on_turn_failed);
    let (parts, output) = run_collect(run).await;
    assert!(of_type(&parts, "error").is_empty());
    assert!(contains_matching(&parts, json!({ "type": "text-delta", "text": "partial" })));
    assert_eq!(failed.load(Ordering::SeqCst), 0);
    assert!(output.result.steps().await.unwrap().is_empty());
}

#[tokio::test]
async fn fails_a_mid_step_stream_that_closes_without_an_intentional_suspension() {
    let (failed, on_turn_failed) = counter();
    let mut run = base_input(fake_harness(), FakeSession::new(partial_text_script()), ToolSet::new());
    run.on_turn_failed = Some(on_turn_failed);
    let (parts, output) = run_collect(run).await;
    let errors = of_type(&parts, "error");
    assert_eq!(errors.len(), 1);
    assert!(
        errors[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unclosed step content")
    );
    assert_eq!(failed.load(Ordering::SeqCst), 1);
    let err = output.result.steps().await.unwrap_err();
    assert!(err.to_string().contains("unclosed step content"));
}

#[tokio::test]
async fn fails_when_terminal_finish_receives_unclosed_step_content() {
    let run = base_input(
        fake_harness(),
        FakeSession::new(vec![
            p(json!({ "type": "text-delta", "id": "t1", "delta": "unclosed" })),
            finish(),
        ]),
        ToolSet::new(),
    );
    let (parts, output) = run_collect(run).await;
    assert!(
        of_type(&parts, "error")[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unclosed step content")
    );
    assert!(
        output
            .result
            .steps()
            .await
            .unwrap_err()
            .to_string()
            .contains("unclosed step content")
    );
}

#[tokio::test]
async fn allows_an_empty_terminal_turn_without_recording_a_step() {
    let run = base_input(fake_harness(), FakeSession::new(vec![finish()]), ToolSet::new());
    let (_, output) = run_collect(run).await;
    assert!(output.result.steps().await.unwrap().is_empty());
    assert_eq!(output.result.text().await.unwrap(), "");
}

fn stop_counter() -> (Arc<AtomicUsize>, OnStopConditionMetFn) {
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    (
        count,
        Arc::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {})
        }),
    )
}

#[tokio::test]
async fn evaluates_stop_conditions_only_after_real_finish_step_events() {
    let step_counts: Arc<Mutex<Vec<usize>>> = Arc::default();
    let (stops, on_stop) = stop_counter();
    let log = step_counts.clone();
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(vec![
            p(json!({ "type": "text-delta", "id": "t1", "delta": "first" })),
            resumable_finish_step(),
            p(json!({ "type": "text-delta", "id": "t2", "delta": "second" })),
            resumable_finish_step(),
            p(json!({ "type": "text-delta", "id": "t3", "delta": "third" })),
        ])),
        ToolSet::new(),
    );
    run.stop_conditions = Some(vec![Arc::new(move |steps: &[StepResult]| {
        log.lock().push(steps.len());
        let met = steps.len() == 2;
        Box::pin(async move { met })
    })]);
    run.on_stop_condition_met = Some(on_stop);
    let (_, output) = run_collect(run).await;
    assert_eq!(*step_counts.lock(), [1, 2]);
    assert_eq!(stops.load(Ordering::SeqCst), 1);
    assert_eq!(
        output.result.steps().await.unwrap().iter().map(|s| s.text()).collect::<Vec<_>>(),
        ["first", "second"]
    );
}

#[tokio::test]
async fn lets_a_terminal_text_only_step_emit_finish_even_when_stop_when_matches_its_step_count() {
    let predicate_count = Arc::new(AtomicUsize::new(0));
    let (stops, on_stop) = stop_counter();
    let count = predicate_count.clone();
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(vec![p(
            json!({ "type": "text-delta", "id": "t1", "delta": "done" }),
        )])),
        ToolSet::new(),
    );
    run.stop_conditions = Some(vec![stop_condition(move |steps| {
        count.fetch_add(1, Ordering::SeqCst);
        steps.len() == 1
    })]);
    run.on_stop_condition_met = Some(on_stop);
    let (parts, output) = run_collect(run).await;
    assert_eq!(predicate_count.load(Ordering::SeqCst), 0);
    assert_eq!(stops.load(Ordering::SeqCst), 0);
    let types = types(&parts);
    assert_eq!(types[types.len() - 2..], ["finish-step", "finish"]);
    assert_eq!(
        output.result.finish_reason().await.unwrap(),
        LanguageModelV4UnifiedFinishReason::Stop
    );
    assert_eq!(output.result.text().await.unwrap(), "done");
}

#[tokio::test]
async fn stops_before_a_next_step_tool_call_when_the_completed_step_has_no_tool_calls() {
    let weather = tool(&[("city", "string")]).with_description("Get weather");
    let (stops, on_stop) = stop_counter();
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(vec![
            p(json!({ "type": "text-delta", "id": "t1", "delta": "first" })),
            finish_step(),
            p(json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "weather", "input": "{\"city\":\"Lima\"}", "providerExecuted": true })),
        ]),
        tools(vec![("weather", weather)]),
    );
    run.stop_conditions = Some(vec![stop_condition(|steps| steps.len() == 1)]);
    run.on_stop_condition_met = Some(on_stop);
    let (parts, output) = run_collect(run).await;
    assert_eq!(stops.load(Ordering::SeqCst), 1);
    assert!(of_type(&parts, "tool-call").is_empty());
    assert_eq!(
        output.result.steps().await.unwrap().iter().map(|s| s.text()).collect::<Vec<_>>(),
        ["first"]
    );
}

#[tokio::test]
async fn processes_a_lookahead_event_exactly_once_when_stop_conditions_do_not_match() {
    let step_counts: Arc<Mutex<Vec<usize>>> = Arc::default();
    let log = step_counts.clone();
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(vec![
            p(json!({ "type": "text-delta", "id": "t1", "delta": "first" })),
            finish_step(),
            p(json!({ "type": "text-start", "id": "t2" })),
            p(json!({ "type": "text-delta", "id": "t2", "delta": "second" })),
            p(json!({ "type": "text-end", "id": "t2" })),
        ])),
        ToolSet::new(),
    );
    run.stop_conditions = Some(vec![stop_condition(move |steps| {
        log.lock().push(steps.len());
        false
    })]);
    let (parts, output) = run_collect(run).await;
    assert_eq!(*step_counts.lock(), [1]);
    assert_eq!(
        parts
            .iter()
            .filter(|p| p["type"] == "text-delta" && p["text"] == "second")
            .count(),
        1
    );
    assert_eq!(
        output.result.steps().await.unwrap().iter().map(|s| s.text()).collect::<Vec<_>>(),
        ["first", "second"]
    );
}
