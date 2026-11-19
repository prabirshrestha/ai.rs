//! Port of `agent/observability/file-reporter.ts`.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde_json::json;

use super::types::{HarnessDebugLevel, HarnessDiagnostic, HarnessDiagnosticConsumer};
use crate::harness::provider::{JsonObject, JsonValue};
use crate::harness::sdk::telemetry::{Telemetry, TelemetryEvent};

/// `FileReporterOptions`.
#[derive(Debug, Clone)]
pub struct FileReporterOptions {
    /// Directory for `events.jsonl` (created if absent).
    pub dir: PathBuf,
    /// Write a turn's records only if the turn errored. Default false.
    pub fail_only: Option<bool>,
    /// File name within `dir`. Default `events.jsonl`.
    pub file_name: Option<String>,
}

#[derive(Default)]
struct Bucket {
    lines: Vec<JsonObject>,
    errored: bool,
}

#[derive(Default)]
struct State {
    turns: HashMap<String, Bucket>,
    last_open_call_id: Option<String>,
    dir_ready: bool,
}

/// `FileReporter`: a [`Telemetry`] integration and diagnostic consumer that
/// writes turn spans and bridge diagnostics to one `events.jsonl`.
pub struct FileReporter {
    dir: PathBuf,
    path: PathBuf,
    fail_only: bool,
    state: Mutex<State>,
}

/// `createFileReporter(options)`.
pub fn create_file_reporter(options: FileReporterOptions) -> FileReporter {
    let file_name = options.file_name.unwrap_or_else(|| "events.jsonl".into());
    FileReporter {
        path: options.dir.join(file_name),
        dir: options.dir,
        fail_only: options.fail_only.unwrap_or(false),
        state: Mutex::default(),
    }
}

fn now_ms() -> JsonValue {
    json!(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    )
}

fn call_id(e: &TelemetryEvent) -> String {
    e.get("callId")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

fn rec(fields: JsonValue) -> JsonObject {
    match fields {
        JsonValue::Object(map) => map,
        _ => JsonObject::new(),
    }
}

/// Copy `key` from `from` when present (TS: `undefined` fields vanish in
/// `JSON.stringify`).
fn copy(into: &mut JsonObject, key: &str, value: Option<&JsonValue>) {
    if let Some(v) = value {
        into.insert(key.into(), v.clone());
    }
}

impl FileReporter {
    fn flush_lines(&self, state: &mut State, lines: &[JsonObject]) {
        if lines.is_empty() {
            return;
        }
        // Best-effort: never let observability break a turn.
        let _ = (|| -> std::io::Result<()> {
            if !state.dir_ready {
                std::fs::create_dir_all(&self.dir)?;
                state.dir_ready = true;
            }
            let mut text = lines
                .iter()
                .map(|l| serde_json::to_string(l).unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n");
            text.push('\n');
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?
                .write_all(text.as_bytes())
        })();
    }

    fn record(&self, state: &mut State, call_id: Option<String>, rec: JsonObject) {
        match call_id.or_else(|| state.last_open_call_id.clone()) {
            None => self.flush_lines(state, &[rec]),
            Some(id) => state.turns.entry(id).or_default().lines.push(rec),
        }
    }

    fn finish_turn(&self, state: &mut State, call_id: &str) {
        let Some(bucket) = state.turns.remove(call_id) else {
            return;
        };
        if state.last_open_call_id.as_deref() == Some(call_id) {
            state.last_open_call_id = None;
        }
        if self.fail_only && !bucket.errored {
            return;
        }
        self.flush_lines(state, &bucket.lines);
    }

    fn done() -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

impl Telemetry for FileReporter {
    fn on_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let id = call_id(e);
        let mut r = rec(json!({ "ts": now_ms(), "kind": "turn-start", "callId": id }));
        copy(&mut r, "operationId", e.get("operationId"));
        copy(&mut r, "provider", e.get("provider"));
        copy(&mut r, "modelId", e.get("modelId"));
        if e.get("recordInputs") != Some(&JsonValue::Bool(false)) {
            let mut input = JsonObject::new();
            copy(&mut input, "messages", e.get("messages"));
            copy(&mut input, "instructions", e.get("instructions"));
            r.insert("input".into(), JsonValue::Object(input));
        }
        let mut s = self.state.lock();
        s.last_open_call_id = Some(id.clone());
        self.record(&mut s, Some(id), r);
        Self::done()
    }

    fn on_step_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let id = call_id(e);
        let mut r = rec(json!({ "ts": now_ms(), "kind": "step-start", "callId": id }));
        copy(&mut r, "step", e.get("stepNumber"));
        self.record(&mut self.state.lock(), Some(id), r);
        Self::done()
    }

    fn on_tool_execution_start(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let id = call_id(e);
        let call = e.get("toolCall").cloned().unwrap_or_default();
        let mut r = rec(json!({
            "ts": now_ms(), "kind": "tool-start", "callId": id,
            "toolName": call["toolName"], "toolCallId": call["toolCallId"],
        }));
        copy(&mut r, "input", call.get("input"));
        self.record(&mut self.state.lock(), Some(id), r);
        Self::done()
    }

    fn on_tool_execution_end(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let id = call_id(e);
        let output = e.get("toolOutput").cloned().unwrap_or_default();
        // [rust] Mirrors TS, which compares `toolOutput.type` with `'error'`.
        let is_error = output["type"] == "error";
        let mut r = rec(json!({
            "ts": now_ms(), "kind": "tool-end", "callId": id,
            "toolCallId": e.get("toolCall").map(|c| c["toolCallId"].clone()),
            "isError": is_error,
        }));
        copy(
            &mut r,
            "output",
            output.get(if is_error { "error" } else { "output" }),
        );
        let mut s = self.state.lock();
        if is_error {
            s.turns.entry(id.clone()).or_default().errored = true;
        }
        self.record(&mut s, Some(id), r);
        Self::done()
    }

    fn on_step_finish(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let id = call_id(e);
        let mut r = rec(json!({ "ts": now_ms(), "kind": "step-finish", "callId": id }));
        copy(&mut r, "usage", e.get("usage"));
        if e.get("recordOutputs") != Some(&JsonValue::Bool(false)) {
            copy(&mut r, "output", e.get("content").filter(|c| !c.is_null()));
        }
        self.record(&mut self.state.lock(), Some(id), r);
        Self::done()
    }

    fn on_end(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let id = call_id(e);
        let mut r = rec(json!({ "ts": now_ms(), "kind": "turn-finish", "callId": id }));
        copy(&mut r, "finishReason", e.get("finishReason"));
        copy(
            &mut r,
            "usage",
            e.get("totalUsage")
                .filter(|v| !v.is_null())
                .or(e.get("usage")),
        );
        let mut s = self.state.lock();
        self.record(&mut s, Some(id.clone()), r);
        self.finish_turn(&mut s, &id);
        Self::done()
    }

    fn on_error(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let id = call_id(e);
        // The dispatcher already serializes an `Error` as `{ name, message }`.
        let mut r = rec(json!({ "ts": now_ms(), "kind": "error", "callId": id }));
        copy(&mut r, "error", e.get("error"));
        let mut s = self.state.lock();
        s.turns.entry(id.clone()).or_default().errored = true;
        self.record(&mut s, Some(id.clone()), r);
        self.finish_turn(&mut s, &id);
        Self::done()
    }

    fn on_abort(&self, e: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let id = call_id(e);
        let mut r = rec(json!({ "ts": now_ms(), "kind": "turn-abort", "callId": id }));
        copy(&mut r, "reason", e.get("reason"));
        let mut s = self.state.lock();
        self.record(&mut s, Some(id.clone()), r);
        self.finish_turn(&mut s, &id);
        Self::done()
    }

    fn as_diagnostic_consumer(&self) -> Option<&dyn HarnessDiagnosticConsumer> {
        Some(self)
    }
}

impl HarnessDiagnosticConsumer for FileReporter {
    fn ingest_diagnostic(&self, diagnostic: &HarnessDiagnostic) {
        let mut s = self.state.lock();
        if diagnostic.level == HarnessDebugLevel::Error
            && let Some(id) = s.last_open_call_id.clone()
        {
            s.turns.entry(id).or_default().errored = true;
        }
        let r = rec(json!({
            "ts": diagnostic.timestamp, "kind": "diagnostic", "diagnostic": diagnostic,
        }));
        let open = s.last_open_call_id.clone();
        self.record(&mut s, open, r);
    }
}

#[cfg(test)]
#[path = "file_reporter_tests.rs"]
mod tests;
