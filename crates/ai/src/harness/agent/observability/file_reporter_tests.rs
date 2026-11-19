use std::path::Path;

use serde_json::{Value, json};

use super::*;
use crate::harness::agent::observability::types::{HarnessDiagnosticKind, HarnessDiagnosticStream};

fn tmp() -> PathBuf {
    std::env::temp_dir().join(format!(
        "harness-file-reporter-{}",
        crate::harness::sdk::error_value::generate_id()
    ))
}

fn read_lines(dir: &Path) -> Vec<Value> {
    std::fs::read_to_string(dir.join("events.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn diag(level: HarnessDebugLevel, message: &str) -> HarnessDiagnostic {
    HarnessDiagnostic {
        level,
        message: message.into(),
        subsystem: "sandbox.log.test".into(),
        kind: HarnessDiagnosticKind::Log,
        source: Some("test".into()),
        stream: Some(HarnessDiagnosticStream::Stdout),
        attrs: None,
        error: None,
        session_id: Some("s1".into()),
        timestamp: 1,
    }
}

fn ev(value: Value) -> TelemetryEvent {
    value.as_object().unwrap().clone()
}

fn start_event() -> TelemetryEvent {
    ev(json!({
        "callId": "call-1", "operationId": "ai.harness", "modelId": "m",
        "messages": [{"role": "user", "content": "what is 2+2?"}],
        "instructions": "be terse"
    }))
}

fn step_start_event() -> TelemetryEvent {
    ev(json!({"callId": "call-1", "stepNumber": 0}))
}

fn step_finish_event() -> TelemetryEvent {
    ev(json!({"callId": "call-1", "usage": {}, "content": [{"type": "text", "text": "4"}]}))
}

fn end_event() -> TelemetryEvent {
    ev(json!({"callId": "call-1", "finishReason": "stop", "totalUsage": {}}))
}

fn reporter(dir: &Path, fail_only: bool) -> FileReporter {
    create_file_reporter(FileReporterOptions {
        dir: dir.to_path_buf(),
        fail_only: Some(fail_only),
        file_name: None,
    })
}

#[tokio::test]
async fn records_error_details_and_flushes_the_failed_turn_without_on_end() {
    let dir = tmp();
    let r = reporter(&dir, true);
    r.on_start(&start_event()).await;
    r.on_error(&ev(
        json!({"callId": "call-1", "error": {"name": "Error", "message": "bridge failed"}}),
    ))
    .await;
    let lines = read_lines(&dir);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["kind"], "turn-start");
    assert_eq!(lines[0]["callId"], "call-1");
    assert_eq!(lines[1]["kind"], "error");
    assert_eq!(
        lines[1]["error"],
        json!({"name": "Error", "message": "bridge failed"})
    );
}

#[tokio::test]
async fn preserves_non_error_thrown_values() {
    for error in [
        Some(json!(null)),
        None,
        Some(json!("failed")),
        Some(json!(0)),
        Some(json!(false)),
    ] {
        let dir = tmp();
        let r = reporter(&dir, false);
        r.on_start(&start_event()).await;
        let mut e = ev(json!({"callId": "call-1"}));
        if let Some(error) = &error {
            e.insert("error".into(), error.clone());
        }
        r.on_error(&e).await;
        let lines = read_lines(&dir);
        let line = lines.iter().find(|l| l["kind"] == "error").unwrap();
        assert_eq!(line.get("error"), error.as_ref());
    }
}

#[tokio::test]
async fn settles_errors_by_call_id_when_turns_overlap() {
    let dir = tmp();
    let r = reporter(&dir, true);
    r.on_start(&start_event()).await;
    r.on_start(&ev(json!({"callId": "call-2"}))).await;
    r.on_error(&ev(
        json!({"callId": "call-1", "error": {"name": "Error", "message": "first failed"}}),
    ))
    .await;
    r.on_end(&ev(json!({"callId": "call-2"}))).await;
    let ids: Vec<Value> = read_lines(&dir)
        .iter()
        .map(|l| l["callId"].clone())
        .collect();
    assert_eq!(ids, vec![json!("call-1"), json!("call-1")]);
}

#[tokio::test]
async fn records_an_abort_and_flushes_the_turn_without_on_end() {
    for reason in [
        json!("user stopped"),
        json!({"name": "Error", "message": "user stopped"}),
    ] {
        let dir = tmp();
        let r = reporter(&dir, false);
        r.on_start(&start_event()).await;
        r.on_abort(&ev(
            json!({"callId": "call-1", "steps": [], "reason": reason}),
        ))
        .await;
        let lines = read_lines(&dir);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["kind"], "turn-start");
        assert_eq!(lines[1]["kind"], "turn-abort");
        assert_eq!(lines[1]["reason"], reason);
    }
}

#[tokio::test]
async fn fail_only_skips_a_clean_aborted_turn_and_flushes_an_errored_aborted_turn() {
    let dir = tmp();
    let r = reporter(&dir, true);
    r.on_start(&start_event()).await;
    r.on_abort(&ev(json!({"callId": "call-1", "steps": []})))
        .await;
    assert!(!dir.join("events.jsonl").exists());

    r.on_start(&ev(json!({"callId": "call-2"}))).await;
    r.ingest_diagnostic(&diag(HarnessDebugLevel::Error, "boom"));
    r.on_abort(&ev(json!({"callId": "call-2", "steps": []})))
        .await;
    let kinds: Vec<Value> = read_lines(&dir).iter().map(|l| l["kind"].clone()).collect();
    assert_eq!(
        kinds,
        vec![
            json!("turn-start"),
            json!("diagnostic"),
            json!("turn-abort")
        ]
    );
}

#[tokio::test]
async fn writes_a_unified_non_lossy_events_jsonl_with_spans_and_diagnostics() {
    let dir = tmp();
    let r = reporter(&dir, false);
    r.on_start(&start_event()).await;
    r.on_step_start(&step_start_event()).await;
    r.ingest_diagnostic(&diag(HarnessDebugLevel::Info, "hello from sandbox"));
    r.on_tool_execution_start(&ev(json!({
        "callId": "call-1", "toolCall": {"toolName": "bash", "toolCallId": "c1", "input": "{}"}
    })))
    .await;
    r.on_tool_execution_end(&ev(json!({
        "callId": "call-1", "toolCall": {"toolCallId": "c1"},
        "toolOutput": {"type": "tool-result", "output": {"ok": true}}
    })))
    .await;
    r.on_step_finish(&step_finish_event()).await;
    r.on_end(&end_event()).await;

    let lines = read_lines(&dir);
    let kinds: Vec<&str> = lines.iter().map(|l| l["kind"].as_str().unwrap()).collect();
    assert_eq!(
        kinds,
        vec![
            "turn-start",
            "step-start",
            "diagnostic",
            "tool-start",
            "tool-end",
            "step-finish",
            "turn-finish"
        ]
    );
    let d = lines.iter().find(|l| l["kind"] == "diagnostic").unwrap();
    assert_eq!(d["diagnostic"]["message"], "hello from sandbox");
    let turn_start = lines.iter().find(|l| l["kind"] == "turn-start").unwrap();
    assert_eq!(
        turn_start["input"]["messages"],
        json!([{"role": "user", "content": "what is 2+2?"}])
    );
    let step_finish = lines.iter().find(|l| l["kind"] == "step-finish").unwrap();
    assert_eq!(
        step_finish["output"],
        json!([{"type": "text", "text": "4"}])
    );
}

#[tokio::test]
async fn record_inputs_and_outputs_false_suppress_message_content() {
    let dir = tmp();
    let r = reporter(&dir, false);
    let mut start = start_event();
    start.insert("recordInputs".into(), false.into());
    r.on_start(&start).await;
    r.on_step_start(&step_start_event()).await;
    let mut finish = step_finish_event();
    finish.insert("recordOutputs".into(), false.into());
    r.on_step_finish(&finish).await;
    r.on_end(&end_event()).await;
    let lines = read_lines(&dir);
    assert!(
        lines
            .iter()
            .find(|l| l["kind"] == "turn-start")
            .unwrap()
            .get("input")
            .is_none()
    );
    assert!(
        lines
            .iter()
            .find(|l| l["kind"] == "step-finish")
            .unwrap()
            .get("output")
            .is_none()
    );
}

#[tokio::test]
async fn fail_only_skips_a_clean_turn_but_writes_a_turn_that_errored() {
    let dir = tmp();
    let r = reporter(&dir, true);
    r.on_start(&start_event()).await;
    r.on_step_start(&step_start_event()).await;
    r.on_step_finish(&step_finish_event()).await;
    r.on_end(&end_event()).await;
    assert!(!dir.join("events.jsonl").exists());

    r.on_start(&ev(
        json!({"callId": "call-2", "operationId": "ai.harness", "modelId": "m"}),
    ))
    .await;
    r.ingest_diagnostic(&diag(HarnessDebugLevel::Error, "boom"));
    r.on_end(&ev(
        json!({"callId": "call-2", "finishReason": "error", "totalUsage": {}}),
    ))
    .await;
    let lines = read_lines(&dir);
    assert!(lines.iter().any(|l| l["kind"] == "diagnostic"));
    assert!(
        lines
            .iter()
            .all(|l| l["callId"] == "call-2" || l["kind"] == "diagnostic")
    );
}
