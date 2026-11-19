//! `run-prompt.test.ts`: `runPrompt tool model output` and `runPrompt host
//! tool input validation`.

use super::*;
use crate::harness::provider::DataContent;
use crate::harness::sdk::error_value::ThrownValue;
use crate::harness::sdk::messages::{FileData, ToolResultContentPart};
use crate::harness::sdk::tool::ToModelOutputOptions;

// ─── helpers ───────────────────────────────────────────────────────────

/// A submitted result as TS `toEqual` sees it (undefined keys dropped).
pub(super) fn submitted_json(submission: &HarnessV1SubmitToolResult) -> JsonValue {
    let mut value = json!({ "toolCallId": submission.tool_call_id, "output": submission.output });
    if let Some(is_error) = submission.is_error {
        value["isError"] = json!(is_error);
    }
    if let Some(tool_result) = &submission.tool_result {
        let mut part = json!({ "type": "tool-result" });
        for (k, v) in to_json(tool_result).as_object().unwrap() {
            part[k] = v.clone();
        }
        value["toolResult"] = part;
    }
    value
}

pub(super) fn submitted_list(log: &Mutex<Vec<HarnessV1SubmitToolResult>>) -> Vec<JsonValue> {
    log.lock().iter().map(submitted_json).collect()
}

/// Inputs a tool's `execute` received.
pub(super) type Calls = Arc<Mutex<Vec<JsonValue>>>;

/// A tool whose `execute` records its input and returns `output(input)`.
pub(super) fn recording_tool(
    schema: Schema,
    output: impl Fn(&JsonValue) -> HarnessResult<JsonValue> + Send + Sync + 'static,
) -> (Tool, Calls) {
    let calls: Calls = Arc::default();
    let log = calls.clone();
    let output = Arc::new(output);
    let tool = Tool::new(schema).with_execute(move |input, _| {
        log.lock().push(input.clone());
        let result = output(&input);
        async move { result }
    });
    (tool, calls)
}

/// A schema with a custom validator (zod refinements, transforms and
/// defaults).
pub(super) fn custom_schema(
    validate: impl Fn(&JsonValue) -> Result<JsonValue, String> + Send + Sync + 'static,
) -> Schema {
    json_schema(json!({ "type": "object" }))
        .with_validate(move |value| validate(value).map_err(Into::into))
}

pub(super) fn pending_result(
    tool_call_id: &str,
    tool_name: &str,
    input: &str,
) -> HarnessV1PendingToolResult {
    HarnessV1PendingToolResult {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        input: input.into(),
        provider_options: None,
        completed_result: None,
    }
}

pub(super) fn custom_approval(
    approval_id: &str,
    tool_call_id: &str,
    tool_name: &str,
    input: &str,
) -> HarnessV1PendingToolApproval {
    serde_json::from_value(json!({
        "approvalId": approval_id,
        "toolCallId": tool_call_id,
        "toolName": tool_name,
        "input": input,
        "kind": "custom",
        "providerExecuted": false,
    }))
    .unwrap()
}

pub(super) fn approval_config(value: JsonValue) -> HarnessAgentToolApprovalConfiguration {
    serde_json::from_value(value).unwrap()
}

pub(super) fn approval_response(approval_id: &str, approved: bool) -> ToolApprovalResponse {
    ToolApprovalResponse {
        approval_id: approval_id.into(),
        approved,
        reason: None,
        provider_executed: None,
    }
}

/// `runHostToolScript`: run a scripted turn and record what the control
/// was asked to submit.
pub(super) async fn run_host_tool_script(
    events: Vec<HarnessV1StreamPart>,
    tools: ToolSet,
    configure: impl FnOnce(&mut RunPromptInput),
) -> (Vec<JsonValue>, Vec<JsonValue>) {
    let (session, submitted) = FakeSession::recording(events);
    let mut run = base_input(fake_harness(), session, tools);
    configure(&mut run);
    if run.mode == RunPromptMode::Continue {
        run.prompt = None;
    }
    let (parts, _) = run_collect(run).await;
    let submitted = submitted_list(&submitted);
    (parts, submitted)
}

fn call(tool_call_id: &str, tool_name: &str, input: &str) -> HarnessV1StreamPart {
    p(
        json!({ "type": "tool-call", "toolCallId": tool_call_id, "toolName": tool_name, "input": input }),
    )
}

const VALIDATION_FAILED: &str = "Tool input validation failed.";

fn validation_failed_submission(tool_call_id: &str) -> JsonValue {
    json!({ "toolCallId": tool_call_id, "output": { "error": VALIDATION_FAILED }, "isError": true })
}

fn empty_object_schema() -> Schema {
    z_object(&[])
}

fn counted_to_model_output(
    tool: Tool,
    output: impl Fn(&ToModelOutputOptions) -> HarnessResult<ToolResultOutput> + Send + Sync + 'static,
) -> (Tool, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let counter = count.clone();
    let tool = tool.with_to_model_output(move |options| {
        counter.fetch_add(1, Ordering::SeqCst);
        let result = output(&options);
        async move { result }
    });
    (tool, count)
}

// ─── runPrompt tool model output ───────────────────────────────────────

fn unsupported_parts() -> Vec<JsonValue> {
    vec![
        json!({ "type": "file", "mediaType": "image/png", "data": { "type": "url", "url": "https://example.com/image.png" } }),
        json!({ "type": "image-url", "url": "https://example.com/image.png" }),
        json!({ "type": "file-url", "url": "https://example.com/file.pdf", "mediaType": "application/pdf" }),
        json!({ "type": "file", "mediaType": "image/png", "data": { "type": "reference", "reference": { "test": "image-id" } } }),
        json!({ "type": "file", "mediaType": "application/pdf", "data": { "type": "data", "data": "cGRm" } }),
    ]
}

fn output_of(value: JsonValue) -> ToolResultOutput {
    serde_json::from_value(value).unwrap()
}

#[tokio::test]
async fn serializes_an_unsupported_client_continuation_without_failing() {
    for part in unsupported_parts() {
        let output = output_of(json!({
            "type": "content",
            "value": [{ "type": "text", "text": "keep the whole output" }, part],
        }));
        let (visualize, calls) =
            recording_tool(empty_object_schema(), |_| Ok(json!("not reached")));
        let (visualize, callback) = counted_to_model_output(visualize, |_| {
            Ok(output_of(json!({ "type": "text", "value": "not reached" })))
        });
        let (session, submitted) = FakeSession::recording(finish_events());
        let mut run = base_input(
            fake_harness(),
            session,
            tools(vec![("visualize", visualize)]),
        );
        run.mode = RunPromptMode::Continue;
        run.prompt = None;
        run.pending_tool_results = vec![pending_result("c1", "visualize", "{}")];
        run.tool_result_continuations = vec![
            serde_json::from_value(json!({
                "toolCallId": "c1",
                "toolName": "visualize",
                "output": output,
                "providerOptions": { "test": { "marker": "preserved" } },
            }))
            .unwrap(),
        ];
        let (parts, result) = run_collect(run).await;
        assert_eq!(
            submitted_list(&submitted),
            vec![json!({
                "toolCallId": "c1",
                "output": output,
                "toolResult": {
                    "type": "tool-result",
                    "toolCallId": "c1",
                    "toolName": "visualize",
                    "providerOptions": { "test": { "marker": "preserved" } },
                    "output": { "type": "text", "value": serde_json::to_string(&output).unwrap() },
                },
            })]
        );
        assert!(calls.lock().is_empty());
        assert_eq!(callback.load(Ordering::SeqCst), 0);
        assert!(of_type(&parts, "error").is_empty());
        assert!(result.result.steps().await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn persists_normalized_client_output_on_suspension_and_submits_it_exactly_once_on_resume() {
    for (output, normalized) in [
        (
            json!({ "type": "content", "value": [unsupported_parts()[0].clone()] }),
            None,
        ),
        (
            json!({ "type": "content", "value": [{ "type": "image-data", "mediaType": "image/png", "data": "iVBORw==" }] }),
            Some(json!({
                "type": "content",
                "value": [{ "type": "file", "mediaType": "image/png", "data": { "type": "data", "data": "iVBORw==" } }],
            })),
        ),
    ] {
        let output = output_of(output);
        let expected = normalized.unwrap_or_else(
            || json!({ "type": "text", "value": serde_json::to_string(&output).unwrap() }),
        );
        let (session, submitted) = FakeSession::recording(finish_events());
        let pending: Arc<Mutex<Vec<HarnessV1PendingToolResult>>> = Arc::default();
        let make = |session: FakeSession| {
            let mut run = base_input(fake_harness(), session, ToolSet::new());
            run.mode = RunPromptMode::Continue;
            run.prompt = None;
            run
        };
        let mut suspended = make(session.clone());
        suspended.pending_tool_results = vec![pending_result("c1", "visualize", "{}")];
        suspended.tool_result_continuations = vec![ToolResultPart {
            tool_call_id: "c1".into(),
            tool_name: "visualize".into(),
            output: output.clone(),
            provider_options: None,
        }];
        suspended.is_turn_suspending = Some(Arc::new(|| true));
        let log = pending.clone();
        suspended.on_pending_tool_result = Some(Arc::new(move |part| log.lock().push(part)));
        run_collect(suspended).await;
        assert!(submitted.lock().is_empty());
        let pending = pending.lock().clone();
        assert_eq!(pending.len(), 1);
        assert_eq!(
            to_json(
                &pending[0]
                    .completed_result
                    .as_ref()
                    .unwrap()
                    .tool_result
                    .as_ref()
                    .unwrap()
                    .output
            ),
            expected
        );

        let mut resumed = make(session);
        resumed.pending_tool_results = pending;
        run_collect(resumed).await;
        let submitted = submitted.lock().clone();
        assert_eq!(submitted.len(), 1);
        assert_eq!(submitted[0].output, to_json(&output));
        assert_eq!(
            to_json(&submitted[0].tool_result.as_ref().unwrap().output),
            expected
        );
    }
}

// [rust] `does not swallow fallback serialization errors` is not ported:
// it builds a circular object, which serde_json values cannot represent.

#[tokio::test]
async fn keeps_unsupported_host_callback_output_strict() {
    let visualize = tool(&[]).with_execute(|_, _| async { Ok(json!({ "ok": true })) });
    let (visualize, _) = counted_to_model_output(visualize, |_| {
        Ok(output_of(
            json!({ "type": "content", "value": [unsupported_parts()[0].clone()] }),
        ))
    });
    let (_, submitted) = run_host_tool_script(
        with_finish(vec![call("c1", "visualize", "{}")]),
        tools(vec![("visualize", visualize)]),
        |_| {},
    )
    .await;
    assert_eq!(submitted.len(), 1);
    assert!(matches(
        &submitted[0],
        &json!({ "toolCallId": "c1", "isError": true })
    ));
    assert!(submitted[0].get("toolResult").is_none());
    assert!(
        submitted[0]["output"]["error"]
            .as_str()
            .unwrap()
            .contains("Harnesses support only text and inline images"),
        "{submitted:?}"
    );
}

#[tokio::test]
async fn converts_once_and_preserves_raw_results() {
    for generator in [false, true] {
        let raw = json!({ "status": "ready" });
        let executions = Arc::new(AtomicUsize::new(0));
        let visualize = if generator {
            let (executions, raw) = (executions.clone(), raw.clone());
            tool(&[]).with_streaming_execute(move |_, _| {
                executions.fetch_add(1, Ordering::SeqCst);
                futures::stream::iter([Ok(json!({ "status": "loading" })), Ok(raw.clone())]).boxed()
            })
        } else {
            let (executions, raw) = (executions.clone(), raw.clone());
            tool(&[]).with_execute(move |_, _| {
                executions.fetch_add(1, Ordering::SeqCst);
                let raw = raw.clone();
                async move { Ok(raw) }
            })
        };
        let expected_raw = raw.clone();
        let (visualize, callback) = counted_to_model_output(visualize, move |options| {
            assert_eq!(options.tool_call_id, "visual");
            assert_eq!(options.input, json!({}));
            assert_eq!(options.output, expected_raw);
            Ok(ToolResultOutput::Content {
                value: vec![
                    serde_json::from_value(json!({ "type": "text", "text": "model-only" }))
                        .unwrap(),
                    ToolResultContentPart::File {
                        data: FileData::Data {
                            data: DataContent::Bytes(vec![137, 80, 78, 71]),
                        },
                        media_type: "image".into(),
                        filename: None,
                        provider_options: None,
                    },
                ],
            })
        });
        let (session, submitted) = FakeSession::recording(with_finish(parts(vec![
            json!({ "type": "tool-call", "toolCallId": "visual", "toolName": "visualize", "input": "{}" }),
            json!({ "type": "tool-result", "toolCallId": "visual", "toolName": "visualize", "result": raw }),
        ])));
        let run = base_input(
            fake_harness(),
            session,
            tools(vec![("visualize", visualize)]),
        );
        let (_, output) = run_collect(run).await;
        assert_eq!(executions.load(Ordering::SeqCst), 1);
        assert_eq!(callback.load(Ordering::SeqCst), 1);
        let submitted = submitted.lock().clone();
        assert_eq!(submitted[0].output, raw);
        let model_output = to_json(&submitted[0].tool_result.as_ref().unwrap().output);
        assert_eq!(
            model_output,
            json!({
                "type": "content",
                "value": [
                    { "type": "text", "text": "model-only" },
                    { "type": "file", "mediaType": "image/png", "data": { "type": "data", "data": "iVBORw==" } },
                ],
            })
        );
        let tool_results = output.result.tool_results().await.unwrap();
        let final_result = tool_results
            .iter()
            .find(|r| r.preliminary != Some(true))
            .unwrap();
        assert_eq!(final_result.output, raw);
        let messages = to_json(&output.result.response_messages().await.unwrap());
        let content: Vec<JsonValue> = messages
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["content"].as_array().cloned())
            .flatten()
            .collect();
        assert!(
            contains_matching(
                &content,
                json!({ "type": "tool-result", "output": model_output })
            ),
            "{content:?}"
        );
        assert_eq!(callback.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn submits_a_callback_failure_as_a_tool_error_without_re_execution() {
    let (visualize, calls) = recording_tool(empty_object_schema(), |_| Ok(json!({ "ok": true })));
    let (visualize, callback) =
        counted_to_model_output(visualize, |_| Err(plain_error("conversion failed")));
    let (_, submitted) = run_host_tool_script(
        with_finish(vec![call("c1", "visualize", "{}")]),
        tools(vec![("visualize", visualize)]),
        |_| {},
    )
    .await;
    assert_eq!(calls.lock().len(), 1);
    assert_eq!(callback.load(Ordering::SeqCst), 1);
    assert_eq!(
        submitted,
        vec![
            json!({ "toolCallId": "c1", "output": { "error": "Error: conversion failed" }, "isError": true })
        ]
    );
}

// ─── runPrompt host tool input validation ──────────────────────────────

/// `z.object({ city: z.string().min(3), mode: z.enum(['read']),
/// values: z.array(z.number()).max(2) })`.
fn restricted_schema() -> Schema {
    custom_schema(|value| {
        let city = value["city"].as_str().ok_or("city: expected string")?;
        if city.chars().count() < 3 {
            return Err("city: too small".into());
        }
        if value["mode"] != "read" {
            return Err("mode: invalid option".into());
        }
        let values = value["values"].as_array().ok_or("values: expected array")?;
        if values.len() > 2 || !values.iter().all(JsonValue::is_number) {
            return Err("values: invalid".into());
        }
        Ok(json!({ "city": city, "mode": "read", "values": values }))
    })
}

fn invalid_error_part(tool_call_id: &str) -> JsonValue {
    json!({ "toolCallId": tool_call_id, "error": { "message": VALIDATION_FAILED } })
}

#[tokio::test]
async fn rejects_invalid_input_before_executing_or_requesting_approval() {
    for (_, input) in [
        (
            "wrong primitive type",
            r#"{"city":1,"mode":"read","values":[1]}"#,
        ),
        (
            "rejected enum",
            r#"{"city":"Paris","mode":"write","values":[1]}"#,
        ),
        ("short string", r#"{"city":"X","mode":"read","values":[1]}"#),
        (
            "long array",
            r#"{"city":"Paris","mode":"read","values":[1,2,3]}"#,
        ),
        ("malformed JSON", r#"{"city":"#),
    ] {
        let (restricted, calls) = recording_tool(restricted_schema(), |_| Ok(json!("not reached")));
        let (parts, submitted) = run_host_tool_script(
            with_finish(vec![
                call("c1", "restricted", input),
                p(json!({ "type": "tool-approval-request", "approvalId": "a1", "toolCallId": "c1" })),
            ]),
            tools(vec![("restricted", restricted)]),
            |run| {
                run.tool_approval = Some(approval_config(json!({ "restricted": "user-approval" })));
            },
        )
        .await;
        assert!(calls.lock().is_empty());
        assert!(contains_matching(
            &parts,
            json!({ "type": "tool-call", "toolCallId": "c1", "invalid": true })
        ));
        let errors = of_type(&parts, "tool-error");
        assert_eq!(errors.len(), 1, "{input}: {parts:?}");
        assert!(matches(errors[0], &invalid_error_part("c1")));
        assert!(of_type(&parts, "tool-result").is_empty());
        assert!(of_type(&parts, "tool-approval-request").is_empty());
        assert_eq!(submitted, vec![validation_failed_submission("c1")]);
    }
}

fn lookup_account() -> (Tool, Calls) {
    let (tool, calls) = recording_tool(empty_object_schema(), |_| Ok(json!("not reached")));
    (
        tool.with_context_schema(z_object(&[("userId", "string")])),
        calls,
    )
}

fn secret_tools_context() -> ToolsContext {
    json!({ "lookupAccount": { "userId": 123, "apiKey": "host-secret" } })
        .as_object()
        .unwrap()
        .clone()
}

/// Records the host-facing error of each failed tool execution.
fn error_recorder(run: &mut RunPromptInput) -> Arc<Mutex<Vec<ErrorValue>>> {
    let errors: Arc<Mutex<Vec<ErrorValue>>> = Arc::default();
    let log = errors.clone();
    run.callbacks = Some(HarnessAgentLifecycleCallbacks {
        on_tool_execution_end: Some(Arc::new(move |event| {
            if let ToolOutput::ToolError(error) = &event.tool_output {
                log.lock().push(error.error.clone());
            }
            Box::pin(async {})
        })),
        ..Default::default()
    });
    errors
}

#[tokio::test]
async fn keeps_host_tool_context_out_of_the_stream_when_context_validation_fails() {
    let (lookup, calls) = lookup_account();
    // The turn pauses for the client-side tool before the runtime echoes the
    // result of the host tool, so the harness reports the host tool outcome
    // itself.
    let (session, submitted) = FakeSession::recording(with_finish(vec![
        call("c1", "lookupAccount", "{}"),
        call("c2", "askUser", "{}"),
    ]));
    let mut run = base_input(
        fake_harness(),
        session,
        tools(vec![("lookupAccount", lookup), ("askUser", tool(&[]))]),
    );
    run.tools_context = Some(secret_tools_context());
    let host_errors = error_recorder(&mut run);
    let (parts, output) = run_collect(run).await;

    assert!(calls.lock().is_empty());
    assert_eq!(
        submitted_list(&submitted),
        vec![
            json!({ "toolCallId": "c1", "output": { "error": "Tool context validation failed." }, "isError": true })
        ]
    );
    // The callback is host-only and keeps the detailed validation error.
    let host_errors = host_errors.lock().clone();
    assert!(
        host_errors[0].js_string().contains("host-secret"),
        "{host_errors:?}"
    );
    // Stream parts can reach the client. They carry the error that the
    // runtime received.
    let errors = of_type(&parts, "tool-error");
    assert_eq!(errors.len(), 1);
    assert!(matches(
        errors[0],
        &json!({ "toolCallId": "c1", "error": { "message": "Tool context validation failed." } })
    ));
    // [rust] The `toUIMessageStream` assertions come with the UI message
    // stream port.
    assert!(
        !serde_json::to_string(&parts)
            .unwrap()
            .contains("host-secret")
    );
    let steps = output.result.steps().await.unwrap();
    assert!(
        !serde_json::to_string(&steps[0].content)
            .unwrap()
            .contains("host-secret")
    );
}

#[tokio::test]
async fn keeps_host_tool_context_out_of_the_stream_when_an_approved_host_tool_fails_context_validation()
 {
    let (lookup, calls) = lookup_account();
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(Vec::new()),
        tools(vec![("lookupAccount", lookup)]),
    );
    run.mode = RunPromptMode::Continue;
    run.prompt = None;
    run.tools_context = Some(secret_tools_context());
    run.pending_tool_approvals = vec![custom_approval("a1", "c1", "lookupAccount", "{}")];
    run.tool_approval_continuations = vec![approval_response("a1", true)];
    let (parts, _) = run_collect(run).await;

    assert!(calls.lock().is_empty());
    let errors = of_type(&parts, "tool-error");
    assert_eq!(errors.len(), 1, "{parts:?}");
    assert!(matches(
        errors[0],
        &json!({ "toolCallId": "c1", "error": { "message": "Tool context validation failed." } })
    ));
    assert!(
        !serde_json::to_string(&parts)
            .unwrap()
            .contains("host-secret")
    );
}

#[tokio::test]
async fn sends_the_text_that_the_runtime_received_when_a_host_tool_throws_an_object() {
    let failing = tool(&[]).with_execute(|_, _| async {
        Err::<JsonValue, _>(
            Box::new(ThrownValue(json!({ "status": 500, "body": "host-secret" }))) as BoxError,
        )
    });
    // The turn pauses for the client-side tool, so the harness reports the
    // host tool outcome itself.
    let (session, submitted) = FakeSession::recording(with_finish(vec![
        call("c1", "failing", "{}"),
        call("c2", "askUser", "{}"),
    ]));
    let mut run = base_input(
        fake_harness(),
        session,
        tools(vec![("failing", failing), ("askUser", tool(&[]))]),
    );
    let host_errors = error_recorder(&mut run);
    let (parts, _) = run_collect(run).await;

    assert_eq!(
        submitted_list(&submitted),
        vec![
            json!({ "toolCallId": "c1", "output": { "error": "[object Object]" }, "isError": true })
        ]
    );
    assert_eq!(
        to_json(&host_errors.lock()[0]),
        json!({ "status": 500, "body": "host-secret" })
    );
    // [rust] The `toUIMessageStream` assertions come with the UI message
    // stream port.
    assert!(
        !serde_json::to_string(&parts)
            .unwrap()
            .contains("host-secret")
    );
}

// [rust] `redacts a host tool error in toUIMessageStream(Response) when
// redactToolErrors is true` comes with the UI message stream port.

#[tokio::test]
async fn reports_a_rejected_host_call_without_an_adapter_echo_in_the_step() {
    let (restricted, calls) = recording_tool(z_object(&[("city", "string")]), |_| {
        Ok(json!("not reached"))
    });
    let run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(vec![call(
            "c1",
            "restricted",
            r#"{"city":42}"#,
        )])),
        tools(vec![("restricted", restricted)]),
    );
    let (parts, output) = run_collect(run).await;

    assert!(calls.lock().is_empty());
    let errors = of_type(&parts, "tool-error");
    assert_eq!(errors.len(), 1);
    assert!(matches(errors[0], &invalid_error_part("c1")));
    assert!(of_type(&parts, "tool-result").is_empty());
    let steps = to_json(&output.result.steps().await.unwrap());
    assert!(contains_matching(
        steps[0]["content"].as_array().unwrap(),
        json!({ "type": "tool-error", "toolCallId": "c1" })
    ));
    assert!(output.result.tool_results().await.unwrap().is_empty());
    // [rust] The `toUIMessageStream` assertions come with the UI message
    // stream port.
}

#[tokio::test]
async fn surfaces_an_echoed_rejected_host_result_as_a_tool_error() {
    for is_error in [true, false] {
        let (restricted, calls) = recording_tool(z_object(&[("city", "string")]), |_| {
            Ok(json!("not reached"))
        });
        let turn_done = Gate::default();
        let submitted: Arc<Mutex<Vec<HarnessV1SubmitToolResult>>> = Arc::default();
        let mut session = FakeSession::new(Vec::new());
        session.on_turn = Some(Arc::new(|emit| {
            emit(call("c1", "restricted", r#"{"city":42}"#));
        }));
        let (gate, log) = (turn_done.clone(), submitted.clone());
        session = session.with_submit(move |submission, emit| {
            let echoed = p(json!({
                "type": "tool-result",
                "toolCallId": submission.tool_call_id,
                "toolName": "restricted",
                "result": submission.output,
                "isError": is_error,
            }));
            log.lock().push(submission);
            emit(echoed.clone());
            emit(echoed);
            for event in finish_events() {
                emit(event);
            }
            gate.open();
        });
        let gate = turn_done.clone();
        session.done = Some(
            async move {
                gate.wait().await;
                Ok(())
            }
            .boxed()
            .shared(),
        );
        let run = base_input(
            fake_harness(),
            session,
            tools(vec![("restricted", restricted)]),
        );
        let (parts, output) = run_collect(run).await;

        assert!(calls.lock().is_empty());
        assert_eq!(
            submitted_list(&submitted),
            vec![validation_failed_submission("c1")]
        );
        assert!(contains_matching(
            &parts,
            json!({ "type": "tool-call", "toolCallId": "c1", "invalid": true })
        ));
        let errors = of_type(&parts, "tool-error");
        assert_eq!(errors.len(), 1, "{parts:?}");
        assert!(matches(errors[0], &invalid_error_part("c1")));
        let steps = output.result.steps().await.unwrap();
        assert_eq!(
            steps[0]
                .content
                .iter()
                .filter(|c| to_json(c)["type"] == "tool-error")
                .count(),
            1
        );
        assert!(of_type(&parts, "tool-result").is_empty());
        // [rust] The `toUIMessageStream` assertions come with the UI message
        // stream port.
    }
}

#[tokio::test]
async fn reports_a_rejected_host_call_once_when_the_turn_is_suspending() {
    let (restricted, calls) = recording_tool(z_object(&[("city", "string")]), |_| {
        Ok(json!("not reached"))
    });
    let pending: Arc<Mutex<Vec<HarnessV1PendingToolResult>>> = Arc::default();
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(vec![call(
            "c1",
            "restricted",
            r#"{"city":42}"#,
        )])),
        tools(vec![("restricted", restricted)]),
    );
    run.is_turn_suspending = Some(Arc::new(|| true));
    let log = pending.clone();
    run.on_pending_tool_result = Some(Arc::new(move |part| log.lock().push(part)));
    let (parts, output) = run_collect(run).await;

    assert!(calls.lock().is_empty());
    let pending = to_json(&*pending.lock());
    assert_eq!(pending.as_array().unwrap().len(), 1);
    assert!(matches(&pending[0], &json!({ "toolCallId": "c1" })));
    assert_eq!(
        pending[0]["completedResult"],
        json!({ "output": { "error": VALIDATION_FAILED }, "isError": true })
    );
    let errors = of_type(&parts, "tool-error");
    assert_eq!(errors.len(), 1);
    assert!(matches(errors[0], &json!({ "toolCallId": "c1" })));
    let steps = to_json(&output.result.steps().await.unwrap());
    assert!(contains_matching(
        steps[0]["content"].as_array().unwrap(),
        json!({ "type": "tool-error", "toolCallId": "c1" })
    ));
}

/// `z.string().transform(f)` on `city`, plus an optional `unit` enum
/// defaulting to `metric` when `with_unit` is set.
fn city_schema(transform: fn(&str) -> String, with_unit: bool) -> Schema {
    custom_schema(move |value| {
        let city = value["city"].as_str().ok_or("city: expected string")?;
        let mut out = json!({ "city": transform(city) });
        if with_unit {
            out["unit"] = match value.get("unit") {
                None => json!("metric"),
                Some(unit) if unit == "metric" || unit == "imperial" => unit.clone(),
                Some(_) => return Err("unit: invalid option".into()),
            };
        }
        Ok(out)
    })
}

fn trim_upper(city: &str) -> String {
    city.trim().to_uppercase()
}

#[tokio::test]
async fn executes_with_defaults_and_transformations_from_the_schema() {
    let (weather, calls) = recording_tool(city_schema(trim_upper, true), |input| Ok(input.clone()));
    let (parts, submitted) = run_host_tool_script(
        with_finish(vec![call("c1", "weather", r#"{"city":" Paris "}"#)]),
        tools(vec![("weather", weather)]),
        |_| {},
    )
    .await;

    assert_eq!(
        *calls.lock(),
        vec![json!({ "city": "PARIS", "unit": "metric" })]
    );
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-call", "toolCallId": "c1", "input": { "city": "PARIS", "unit": "metric" } })
    ));
    assert_eq!(
        submitted,
        vec![json!({ "toolCallId": "c1", "output": { "city": "PARIS", "unit": "metric" } })]
    );
}

#[tokio::test]
async fn validates_empty_input_as_an_object() {
    let (empty, calls) = recording_tool(empty_object_schema(), |input| Ok(input.clone()));
    let (_, submitted) = run_host_tool_script(
        with_finish(vec![call("c1", "empty", "")]),
        tools(vec![("empty", empty)]),
        |_| {},
    )
    .await;

    assert_eq!(*calls.lock(), vec![json!({})]);
    assert_eq!(submitted, vec![json!({ "toolCallId": "c1", "output": {} })]);
}

#[tokio::test]
async fn awaits_asynchronous_validation_for_accepted_and_rejected_inputs() {
    // [rust] Schema validation is synchronous; the refinement runs inline.
    let schema = custom_schema(|value| {
        let city = value["city"].as_str().ok_or("city: expected string")?;
        if city == "rejected" {
            return Err("Invalid input".into());
        }
        Ok(json!({ "city": city }))
    });
    let (weather, calls) = recording_tool(schema, |input| Ok(input.clone()));
    let (_, submitted) = run_host_tool_script(
        with_finish(vec![
            call("c1", "weather", r#"{"city":"rejected"}"#),
            call("c2", "weather", r#"{"city":"accepted"}"#),
        ]),
        tools(vec![("weather", weather)]),
        |_| {},
    )
    .await;

    assert_eq!(*calls.lock(), vec![json!({ "city": "accepted" })]);
    assert_eq!(
        submitted,
        vec![
            validation_failed_submission("c1"),
            json!({ "toolCallId": "c2", "output": { "city": "accepted" } }),
        ]
    );
}

fn path_starting_with(prefix: &'static str) -> Schema {
    custom_schema(move |value| {
        let path = value["path"].as_str().ok_or("path: expected string")?;
        if !path.starts_with(prefix) {
            return Err(format!("path: must start with {prefix}"));
        }
        Ok(json!({ "path": path }))
    })
}

#[tokio::test]
async fn validates_raw_paths_while_keeping_the_stream_input_relative() {
    let (read_file, calls) =
        recording_tool(path_starting_with(WORK_DIR), |input| Ok(input.clone()));
    let (parts, _) = run_host_tool_script(
        with_finish(vec![call(
            "c1",
            "readFile",
            &json!({ "path": format!("{WORK_DIR}/src/foo.ts") }).to_string(),
        )]),
        tools(vec![("readFile", read_file)]),
        |_| {},
    )
    .await;

    assert_eq!(
        *calls.lock(),
        vec![json!({ "path": format!("{WORK_DIR}/src/foo.ts") })]
    );
    assert!(contains_matching(
        &parts,
        json!({ "type": "tool-call", "input": { "path": "src/foo.ts" } })
    ));
}

// [rust] `executes deeply nested validated input without recursively
// projecting it` is not ported: serde_json caps parsing at 128 levels, so a
// 12,000-level input cannot be represented.
//
// [rust] `executes cyclic schema output while projecting a separate cyclic
// display input` is not ported: serde_json values cannot be cyclic.

#[tokio::test]
async fn rejects_raw_paths_that_only_satisfy_the_schema_after_display_stripping() {
    let (read_file, calls) =
        recording_tool(path_starting_with("src/"), |_| Ok(json!("not reached")));
    let (parts, submitted) = run_host_tool_script(
        with_finish(vec![call(
            "c1",
            "readFile",
            &json!({ "path": format!("{WORK_DIR}/src/foo.ts") }).to_string(),
        )]),
        tools(vec![("readFile", read_file)]),
        |_| {},
    )
    .await;

    assert!(calls.lock().is_empty());
    assert!(contains_matching(
        &parts,
        json!({
            "type": "tool-call",
            "toolCallId": "c1",
            "input": { "path": "src/foo.ts" },
            "invalid": true,
            "error": { "message": VALIDATION_FAILED },
        })
    ));
    assert_eq!(submitted, vec![validation_failed_submission("c1")]);
}

#[tokio::test]
async fn keeps_concurrent_calls_with_identical_input_isolated_by_tool_call_id() {
    let (upper, upper_calls) =
        recording_tool(city_schema(|c| c.to_uppercase(), false), |i| Ok(i.clone()));
    let (lower, lower_calls) =
        recording_tool(city_schema(|c| c.to_lowercase(), false), |i| Ok(i.clone()));
    let (invalid, invalid_calls) = recording_tool(z_object(&[("city", "number")]), |_| {
        Ok(json!("not reached"))
    });
    let (_, submitted) = run_host_tool_script(
        with_finish(vec![
            call("c1", "upper", r#"{"city":"Paris"}"#),
            call("c2", "lower", r#"{"city":"Paris"}"#),
            call("c3", "invalid", r#"{"city":"Paris"}"#),
        ]),
        tools(vec![
            ("upper", upper),
            ("lower", lower),
            ("invalid", invalid),
        ]),
        |_| {},
    )
    .await;

    assert_eq!(upper_calls.lock()[0], json!({ "city": "PARIS" }));
    assert_eq!(lower_calls.lock()[0], json!({ "city": "paris" }));
    assert!(invalid_calls.lock().is_empty());
    assert!(submitted.contains(&validation_failed_submission("c3")));
}

#[tokio::test]
async fn preserves_generator_results_with_transformed_input() {
    let weather =
        Tool::new(city_schema(|c| c.to_uppercase(), false)).with_streaming_execute(|input, _| {
            let city = input["city"].clone();
            futures::stream::iter([
                Ok(json!({ "city": city, "state": "loading" })),
                Ok(json!({ "city": city, "state": "done" })),
            ])
            .boxed()
        });
    let (parts, submitted) = run_host_tool_script(
        with_finish(vec![call("c1", "weather", r#"{"city":"Paris"}"#)]),
        tools(vec![("weather", weather)]),
        |_| {},
    )
    .await;

    let results = of_type(&parts, "tool-result");
    assert_eq!(
        results
            .iter()
            .map(|r| r["output"].clone())
            .collect::<Vec<_>>(),
        vec![
            json!({ "city": "PARIS", "state": "loading" }),
            json!({ "city": "PARIS", "state": "done" }),
        ]
    );
    assert!(results.iter().all(|r| r["preliminary"] == true));
    assert_eq!(
        submitted,
        vec![json!({ "toolCallId": "c1", "output": { "city": "PARIS", "state": "done" } })]
    );
}

#[tokio::test]
async fn revalidates_approved_continuation_input() {
    for (input, valid) in [(r#"{"city":" Paris "}"#, true), (r#"{"city":42}"#, false)] {
        let (weather, calls) = recording_tool(city_schema(trim_upper, false), |i| Ok(i.clone()));
        let (parts, submitted) =
            run_host_tool_script(Vec::new(), tools(vec![("weather", weather)]), |run| {
                run.mode = RunPromptMode::Continue;
                run.pending_tool_approvals = vec![custom_approval("a1", "c1", "weather", input)];
                run.tool_approval_continuations = vec![approval_response("a1", true)];
            })
            .await;

        assert_eq!(calls.lock().len(), usize::from(valid));
        assert_eq!(
            submitted,
            vec![if valid {
                json!({ "toolCallId": "c1", "output": { "city": "PARIS" } })
            } else {
                validation_failed_submission("c1")
            }]
        );
        assert!(contains_matching(
            &parts,
            json!({ "type": if valid { "tool-result" } else { "tool-error" }, "toolCallId": "c1" })
        ));
    }
}
