//! Port of `bridge/opencode-events.ts`, `bridge/create-emit-stream-event.ts`,
//! `bridge/opencode-finish-step.ts` and `bridge/opencode-usage.ts`: OpenCode
//! server events as stream parts, which [`super::turn`] applies to the turn.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};

/// `asOpenCodeObject`.
pub(super) fn as_object(value: Option<&Value>) -> Option<&Map<String, Value>> {
    value.and_then(Value::as_object)
}

/// `stringValue`: a non-empty string.
pub(super) fn string_value(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// `String(value ?? fallback)`.
fn js_string(value: Option<&Value>, fallback: &str) -> String {
    match value {
        None | Some(Value::Null) => fallback.to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}

/// `OpenCodeEvent`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct OpenCodeEvent {
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub properties: Map<String, Value>,
}

impl OpenCodeEvent {
    pub fn prop(&self, name: &str) -> Option<&Value> {
        self.properties.get(name).filter(|value| !value.is_null())
    }

    /// `String(props.<name> ?? event.id)`.
    fn prop_or_id(&self, name: &str) -> String {
        match self.prop(name) {
            Some(value) => js_string(Some(value), ""),
            None => self.id.clone().unwrap_or_else(|| "undefined".into()),
        }
    }
}

fn strip_sync_version(event_type: &str) -> String {
    match event_type.rsplit_once('.') {
        Some((head, tail)) if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) => {
            head.to_string()
        }
        _ => event_type.to_string(),
    }
}

/// `unwrapOpenCodeEvent`.
pub(super) fn unwrap_opencode_event(raw: &Value) -> Option<OpenCodeEvent> {
    let raw = raw.as_object()?;
    if raw.get("type").and_then(Value::as_str) == Some("sync")
        && raw.get("syncEvent").is_some_and(|sync| !sync.is_null())
    {
        let sync = raw.get("syncEvent")?.as_object()?;
        let id = match sync.get("id").or_else(|| raw.get("id")) {
            None | Some(Value::Null) => String::new(),
            Some(value) => js_string(Some(value), ""),
        };
        return Some(OpenCodeEvent {
            id: Some(id),
            r#type: Some(strip_sync_version(&js_string(sync.get("type"), ""))),
            properties: as_object(sync.get("data")).cloned().unwrap_or_default(),
        });
    }
    Some(OpenCodeEvent {
        id: raw.get("id").and_then(Value::as_str).map(str::to_string),
        r#type: raw
            .get("type")
            .and_then(Value::as_str)
            .map(strip_sync_version),
        properties: as_object(raw.get("properties"))
            .or_else(|| as_object(raw.get("data")))
            .cloned()
            .unwrap_or_default(),
    })
}

/// `getOpenCodeEventSessionId`.
pub(super) fn get_opencode_event_session_id(event: &OpenCodeEvent) -> Option<String> {
    let props = &event.properties;
    if let Some(id) = props.get("sessionID").and_then(Value::as_str) {
        return Some(id.to_string());
    }
    if let Some(id) = props.get("sessionId").and_then(Value::as_str) {
        return Some(id.to_string());
    }
    let event_type = event.r#type.as_deref().unwrap_or_default();
    if event_type.starts_with("session.")
        && let Some(id) = props.get("id").and_then(Value::as_str)
    {
        return Some(id.to_string());
    }
    let info = as_object(props.get("info"));
    if let Some(id) = info
        .and_then(|info| info.get("sessionID"))
        .and_then(Value::as_str)
    {
        return Some(id.to_string());
    }
    if matches!(
        event_type,
        "session.created" | "session.updated" | "session.deleted"
    ) && let Some(id) = info.and_then(|info| info.get("id")).and_then(Value::as_str)
    {
        return Some(id.to_string());
    }
    as_object(props.get("part"))
        .and_then(|part| part.get("sessionID"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// `OpenCodeTokenUsage`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct OpenCodeTokens {
    pub input: f64,
    pub output: f64,
    pub reasoning: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// `HarnessUsage` of a `finish-step`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct StepUsage {
    pub input_total: f64,
    pub input_no_cache: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub output_total: f64,
    pub output_text: f64,
    pub output_reasoning: f64,
}

impl StepUsage {
    /// The ai.rs usage: `input` excludes cached input, `output` includes
    /// reasoning.
    pub fn to_usage(self) -> crate::Usage {
        let to = |value: f64| value.max(0.0).min(u32::MAX as f64) as u32;
        crate::Usage {
            input: to(self.input_no_cache),
            output: to(self.output_total),
            cache_read: to(self.cache_read),
            cache_write: to(self.cache_write),
            reasoning: (self.output_reasoning > 0.0).then(|| to(self.output_reasoning)),
            total_tokens: to(self.input_total + self.output_total),
            ..Default::default()
        }
    }
}

fn number_value(value: Option<&Value>) -> f64 {
    value
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
}

/// `extractOpenCodeTokens`.
fn extract_opencode_tokens(value: Option<&Value>) -> Option<OpenCodeTokens> {
    let record = as_object(value)?;
    let cache = as_object(record.get("cache"))?;
    Some(OpenCodeTokens {
        input: number_value(record.get("input")),
        output: number_value(record.get("output")),
        reasoning: number_value(record.get("reasoning")),
        cache_read: number_value(cache.get("read")),
        cache_write: number_value(cache.get("write")),
    })
}

/// `mapUsage`.
pub(super) fn map_usage(tokens: Option<&Value>) -> StepUsage {
    map_tokens(extract_opencode_tokens(tokens).unwrap_or_default())
}

pub(super) fn map_tokens(value: OpenCodeTokens) -> StepUsage {
    StepUsage {
        input_total: value.input,
        input_no_cache: (value.input - value.cache_read).max(0.0),
        cache_read: value.cache_read,
        cache_write: value.cache_write,
        output_total: value.output + value.reasoning,
        output_text: value.output,
        output_reasoning: value.reasoning,
    }
}

/// `extractSessionTokens`.
// [rust] Session-token totals are not wired into the turn yet; usage comes
// from each step.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn extract_session_tokens(value: &Value) -> Option<OpenCodeTokens> {
    let record = value.as_object()?;
    extract_opencode_tokens(record.get("tokens"))
        .or_else(|| extract_opencode_tokens(record.get("info").and_then(|info| info.get("tokens"))))
        .or_else(|| extract_opencode_tokens(record.get("data").and_then(|data| data.get("tokens"))))
        .or_else(|| {
            extract_opencode_tokens(
                record
                    .get("data")
                    .and_then(|data| data.get("data"))
                    .and_then(|data| data.get("tokens")),
            )
        })
}

/// `subtractSessionTokens`.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn subtract_session_tokens(
    before: OpenCodeTokens,
    after: OpenCodeTokens,
) -> OpenCodeTokens {
    let diff = |before: f64, after: f64| (after - before).max(0.0);
    OpenCodeTokens {
        input: diff(before.input, after.input),
        output: diff(before.output, after.output),
        reasoning: diff(before.reasoning, after.reasoning),
        cache_read: diff(before.cache_read, after.cache_read),
        cache_write: diff(before.cache_write, after.cache_write),
    }
}

/// The unified finish reason of `mapOpenCodeFinishReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Error,
    Other,
}

/// `mapOpenCodeFinishReason`.
pub(super) fn map_opencode_finish_reason(reason: &str) -> FinishReason {
    let normalized = reason.to_lowercase();
    if normalized.contains("length") {
        FinishReason::Length
    } else if normalized.contains("filter") {
        FinishReason::ContentFilter
    } else if normalized.contains("tool") {
        FinishReason::ToolCalls
    } else if normalized.contains("error") || normalized.contains("fail") {
        FinishReason::Error
    } else if normalized == "stop" || normalized == "end" {
        FinishReason::Stop
    } else {
        FinishReason::Other
    }
}

/// One harness stream part (`HarnessV1StreamPart`).
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Part {
    StreamStart {
        model_id: Option<String>,
    },
    TextStart {
        id: String,
    },
    TextDelta {
        id: String,
        delta: String,
    },
    TextEnd {
        id: String,
    },
    ReasoningStart {
        id: String,
    },
    ReasoningDelta {
        id: String,
        delta: String,
    },
    ReasoningEnd {
        id: String,
    },
    /// A tool OpenCode runs itself (`providerExecuted: true`).
    ToolCall {
        tool_call_id: String,
        tool_name: String,
        native_name: Option<String>,
        input: Value,
        dynamic: bool,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        result: Value,
        is_error: bool,
    },
    FileChange {
        path: String,
    },
    FinishStep {
        reason: FinishReason,
        raw: String,
        usage: StepUsage,
        cost: Option<f64>,
    },
    Compaction {
        summary: String,
    },
    /// `raw` parts (compaction progress, subagent usage).
    Raw(Value),
    Warning(String),
    Error(String),
}

/// What the translator asks of the turn, besides parts.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Output {
    Part(Part),
    /// `authorizeHostToolCall`: OpenCode called a host tool.
    AuthorizeHostToolCall {
        call_id: String,
        tool_name: String,
        input: Value,
    },
    /// `onSubagentSession`.
    SubagentSession {
        parent_session_id: String,
        session_id: String,
    },
}

/// `NATIVE_TO_COMMON` and `OPENCODE_TO_WIRE`: `toWireToolName`.
pub(super) fn to_wire_tool_name(native_name: &str) -> String {
    match native_name {
        "view" | "read" => "read",
        "write" => "write",
        "edit" => "edit",
        "bash" => "bash",
        "glob" => "glob",
        "grep" => "grep",
        "question" => "askUserQuestions",
        "list" | "ls" => "ls",
        "webfetch" => "webfetch",
        "task" | "agent" | "subtask" => "agent",
        "askUserQuestions" => "question",
        other => other,
    }
    .to_string()
}

/// `nativeNameField`.
pub(super) fn native_name_field(native_name: &str, tool_name: &str) -> Option<String> {
    (!native_name.is_empty() && native_name != tool_name && tool_name != "agent")
        .then(|| native_name.to_string())
}

/// The prefix OpenCode gives the host tool MCP server's tools.
pub(super) const HOST_TOOL_PREFIX: &str = "harness-tools_";

/// What the translator needs to know about tools.
#[derive(Debug, Clone, Default)]
pub(super) struct ToolNaming {
    /// Host tool names.
    pub host_tools: HashSet<String>,
    /// Tool name prefixes of connected user MCP servers.
    pub mcp_prefixes: Vec<String>,
    pub work_dir: String,
}

impl ToolNaming {
    /// `getHostToolName`.
    pub fn host_tool_name(&self, tool_name: &str, raw_tool_name: Option<&str>) -> Option<String> {
        if self.host_tools.contains(tool_name) {
            return Some(tool_name.to_string());
        }
        let raw = raw_tool_name?;
        if self.host_tools.contains(raw) {
            return Some(raw.to_string());
        }
        raw.strip_prefix(HOST_TOOL_PREFIX)
            .filter(|name| self.host_tools.contains(*name))
            .map(str::to_string)
    }

    fn is_mcp_tool_name(&self, name: &str) -> bool {
        self.mcp_prefixes
            .iter()
            .any(|prefix| name.starts_with(prefix))
    }

    /// `stripWorkDir`.
    pub fn strip_work_dir(&self, file: &str) -> String {
        if file.is_empty() {
            return file.to_string();
        }
        let normalized = crate::harness::paths::normalize_posix(file);
        let root = crate::harness::paths::normalize_posix(&self.work_dir);
        match normalized.strip_prefix(&format!("{}/", root.trim_end_matches('/'))) {
            Some(relative) if crate::harness::paths::is_absolute_posix(file) => {
                relative.to_string()
            }
            _ => file.to_string(),
        }
    }
}

/// `TranslationState`.
#[derive(Debug, Default)]
pub(super) struct TranslationState {
    pub stream_started: bool,
    text_deltas: HashMap<String, String>,
    reasoning_deltas: HashMap<String, String>,
    tool_inputs: HashMap<String, String>,
    tool_names: HashMap<String, (String, String)>,
    tool_calls_emitted: HashSet<String>,
    tool_results_emitted: HashSet<String>,
    host_tool_calls_authorized: HashSet<String>,
    shell_commands: HashMap<String, String>,
    message_roles: HashMap<String, String>,
    /// Insertion-ordered like the TS `Set`s.
    legacy_text_part_ids: Vec<String>,
    legacy_reasoning_part_ids: Vec<String>,
    legacy_step_finish_part_ids: HashSet<String>,
    dynamic_tool_call_ids: HashSet<String>,
}

struct Compaction {
    id: String,
    /// Part id → text, in insertion order.
    text: Vec<(String, String)>,
}

/// `createEmitStreamEvent`: one translator per OpenCode session.
pub(super) struct Translator {
    pub state: TranslationState,
    naming: ToolNaming,
    compaction_messages: HashSet<String>,
    compaction: Option<Compaction>,
    out: Vec<Output>,
}

fn emit_missing_final_delta(
    out: &mut Vec<Output>,
    id: &str,
    full_text: Option<&str>,
    emitted: &str,
    reasoning: bool,
) {
    let Some(full) = full_text.filter(|full| !full.is_empty()) else {
        return;
    };
    if full == emitted || !full.starts_with(emitted) {
        return;
    }
    let delta = full[emitted.len()..].to_string();
    out.push(Output::Part(if reasoning {
        Part::ReasoningDelta {
            id: id.to_string(),
            delta,
        }
    } else {
        Part::TextDelta {
            id: id.to_string(),
            delta,
        }
    }));
}

/// The subset of `asOpenCodeErrorFromValue` the retry message uses.
fn retry_message(event: &OpenCodeEvent) -> String {
    let mut details = Vec::new();
    if let Some(attempt) = event.prop("attempt").and_then(Value::as_f64) {
        details.push(format!("attempt {}", format_number(attempt)));
    }
    match event.prop("error") {
        Some(Value::Object(error)) => {
            if let Some(status) = error.get("statusCode").and_then(Value::as_f64) {
                details.push(format!("HTTP {}", format_number(status)));
            }
            if let Some(message) = string_value(error.get("message"))
                .or_else(|| string_value(error.get("data").and_then(|data| data.get("message"))))
            {
                details.push(message);
            }
        }
        Some(error) => details.push(format_error(error)),
        None => {}
    }
    if details.is_empty() {
        "OpenCode session retry".to_string()
    } else {
        format!("OpenCode session retry: {}", details.join("; "))
    }
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

/// `formatError`.
pub(super) fn format_error(error: &Value) -> String {
    match error {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// The message of an `emitError({ error, message })`: the error's own
/// message after the context.
fn error_message(context: &str, error: Option<&Value>) -> String {
    let detail = error.and_then(|error| match error {
        Value::String(text) => Some(text.clone()),
        Value::Object(map) => string_value(map.get("message"))
            .or_else(|| string_value(map.get("data").and_then(|data| data.get("message"))))
            .or_else(|| string_value(map.get("name")))
            .or_else(|| Some(error.to_string())),
        Value::Null => None,
        other => Some(other.to_string()),
    });
    match detail {
        Some(detail) => format!("{context}: {detail}"),
        None => context.to_string(),
    }
}

impl Translator {
    pub fn new(naming: ToolNaming) -> Self {
        Self {
            state: TranslationState::default(),
            naming,
            compaction_messages: HashSet::new(),
            compaction: None,
            out: Vec::new(),
        }
    }

    fn part(&mut self, part: Part) {
        self.out.push(Output::Part(part));
    }

    fn finish_compaction(&mut self, failed: bool) {
        let Some(compaction) = self.compaction.take() else {
            return;
        };
        if failed {
            self.part(Part::Raw(json!({
                "type": "opencode.compaction",
                "messageId": compaction.id,
                "status": "failed",
            })));
        } else {
            let summary: String = compaction
                .text
                .iter()
                .map(|(_, text)| text.as_str())
                .collect();
            self.part(Part::Compaction { summary });
        }
    }

    /// `emitOpenCodeStreamStart`.
    pub fn stream_start(&mut self, info: Option<&Value>) -> Vec<Output> {
        self.emit_stream_start(info);
        std::mem::take(&mut self.out)
    }

    fn emit_stream_start(&mut self, info: Option<&Value>) {
        if self.state.stream_started {
            return;
        }
        let Some(message) = as_object(info) else {
            return;
        };
        let is_assistant = message.get("role").and_then(Value::as_str) == Some("assistant")
            || message.get("type").and_then(Value::as_str) == Some("assistant");
        if !is_assistant {
            return;
        }
        let model_id = match (
            string_value(message.get("providerID")),
            string_value(message.get("modelID")),
        ) {
            (Some(provider), Some(model)) => Some(format!("{provider}/{model}")),
            _ => None,
        };
        self.state.stream_started = true;
        self.part(Part::StreamStart { model_id });
    }

    /// Translate one event.
    pub fn translate(&mut self, event: &OpenCodeEvent) -> Vec<Output> {
        self.translate_event(event);
        std::mem::take(&mut self.out)
    }

    fn translate_event(&mut self, event: &OpenCodeEvent) {
        let event_type = event.r#type.as_deref().unwrap_or_default();
        let props = &event.properties;

        if event_type == "message.updated" {
            if let Some(info) = as_object(props.get("info")) {
                let id = string_value(info.get("id"));
                if let (Some(id), Some(role)) = (&id, string_value(info.get("role"))) {
                    self.state.message_roles.insert(id.clone(), role);
                }
                if let Some(id) = id
                    && info.get("summary") == Some(&Value::Bool(true))
                {
                    if self.compaction_messages.insert(id.clone()) {
                        self.compaction = Some(Compaction {
                            id: id.clone(),
                            text: Vec::new(),
                        });
                        self.emit_stream_start(props.get("info"));
                        self.part(Part::Raw(json!({
                            "type": "opencode.compaction",
                            "messageId": id,
                            "status": "started",
                        })));
                    }
                    let has_error = info.get("error").is_some_and(|error| {
                        !error.is_null() && error != &Value::Bool(false) && error != ""
                    });
                    if has_error && self.compaction.as_ref().is_some_and(|c| c.id == id) {
                        self.finish_compaction(true);
                    }
                }
            }
            return;
        }
        if event_type == "session.compacted" {
            self.finish_compaction(false);
            return;
        }
        if event_type == "message.part.updated" || event_type == "message.part.delta" {
            let part = as_object(props.get("part"));
            if event_type == "message.part.updated"
                && part
                    .and_then(|part| part.get("type"))
                    .and_then(Value::as_str)
                    == Some("compaction")
            {
                return;
            }
            let message_id = if event_type == "message.part.updated" {
                string_value(part.and_then(|part| part.get("messageID")))
            } else {
                string_value(props.get("messageID"))
            };
            if let Some(message_id) = message_id
                && self.compaction_messages.contains(&message_id)
            {
                let current = self.compaction.as_ref().is_some_and(|c| c.id == message_id);
                if event_type == "message.part.updated" {
                    if current
                        && part
                            .and_then(|part| part.get("type"))
                            .and_then(Value::as_str)
                            == Some("text")
                        && let (Some(id), Some(text)) = (
                            string_value(part.and_then(|part| part.get("id"))),
                            part.and_then(|part| part.get("text"))
                                .and_then(Value::as_str),
                        )
                    {
                        let compaction = self.compaction.as_mut().unwrap();
                        match compaction.text.iter_mut().find(|(key, _)| *key == id) {
                            Some(entry) => entry.1 = text.to_string(),
                            None => compaction.text.push((id, text.to_string())),
                        }
                    }
                    self.emit_legacy_step_finish_part(props.get("part"));
                } else if current && props.get("field").and_then(Value::as_str) == Some("text") {
                    if let (Some(id), Some(delta)) = (
                        string_value(props.get("partID")),
                        props.get("delta").and_then(Value::as_str),
                    ) {
                        let compaction = self.compaction.as_mut().unwrap();
                        if let Some(entry) = compaction.text.iter_mut().find(|(key, _)| *key == id)
                        {
                            entry.1.push_str(delta);
                        }
                    }
                }
                return;
            }
        }
        match event_type {
            "message.part.delta" => self.emit_legacy_part_delta(props),
            "message.part.updated" => {
                if self.emit_legacy_text_part_update(props.get("part")) {
                    return;
                }
                if self.emit_legacy_step_finish_part(props.get("part")) {
                    return;
                }
                self.emit_legacy_tool_part(props.get("part"));
            }
            "session.next.text.started" => {
                let id = event.prop_or_id("textID");
                self.part(Part::TextStart { id });
            }
            "session.next.text.delta" => {
                let id = event.prop_or_id("textID");
                let delta = js_string(props.get("delta"), "");
                self.state
                    .text_deltas
                    .entry(id.clone())
                    .or_default()
                    .push_str(&delta);
                self.part(Part::TextDelta { id, delta });
            }
            "session.next.text.ended" => {
                let id = event.prop_or_id("textID");
                let emitted = self.state.text_deltas.get(&id).cloned().unwrap_or_default();
                emit_missing_final_delta(
                    &mut self.out,
                    &id,
                    props.get("text").and_then(Value::as_str),
                    &emitted,
                    false,
                );
                self.part(Part::TextEnd { id });
            }
            "session.next.reasoning.started" => {
                let id = event.prop_or_id("reasoningID");
                self.part(Part::ReasoningStart { id });
            }
            "session.next.reasoning.delta" => {
                let id = event.prop_or_id("reasoningID");
                let delta = js_string(props.get("delta"), "");
                self.state
                    .reasoning_deltas
                    .entry(id.clone())
                    .or_default()
                    .push_str(&delta);
                self.part(Part::ReasoningDelta { id, delta });
            }
            "session.next.reasoning.ended" => {
                let id = event.prop_or_id("reasoningID");
                let emitted = self
                    .state
                    .reasoning_deltas
                    .get(&id)
                    .cloned()
                    .unwrap_or_default();
                emit_missing_final_delta(
                    &mut self.out,
                    &id,
                    props.get("text").and_then(Value::as_str),
                    &emitted,
                    true,
                );
                self.part(Part::ReasoningEnd { id });
            }
            "session.next.shell.started" => {
                let call_id = event.prop_or_id("callID");
                let command = js_string(props.get("command"), "");
                self.state
                    .shell_commands
                    .insert(call_id.clone(), command.clone());
                self.part(Part::ToolCall {
                    tool_call_id: call_id,
                    tool_name: "bash".into(),
                    native_name: Some("bash".into()),
                    input: json!({ "command": command }),
                    dynamic: false,
                });
            }
            "session.next.shell.ended" => {
                let call_id = event.prop_or_id("callID");
                let command = self
                    .state
                    .shell_commands
                    .get(&call_id)
                    .cloned()
                    .unwrap_or_default();
                self.part(Part::ToolResult {
                    tool_call_id: call_id,
                    tool_name: "bash".into(),
                    result: json!({
                        "command": command,
                        "output": js_string(props.get("output"), ""),
                    }),
                    is_error: false,
                });
            }
            "session.next.tool.input.delta" => {
                let call_id = event.prop_or_id("callID");
                let delta = js_string(props.get("delta"), "");
                self.state
                    .tool_inputs
                    .entry(call_id)
                    .or_default()
                    .push_str(&delta);
            }
            "session.next.tool.input.ended" => {
                let call_id = event.prop_or_id("callID");
                self.state
                    .tool_inputs
                    .insert(call_id, js_string(props.get("text"), ""));
            }
            "session.next.tool.called" => {
                let call_id = event.prop_or_id("callID");
                let raw_tool_name = js_string(props.get("tool"), "unknown");
                if raw_tool_name == "StructuredOutput" || raw_tool_name == "question" {
                    return;
                }
                let tool_name = to_wire_tool_name(&raw_tool_name);
                self.state
                    .tool_names
                    .insert(call_id.clone(), (raw_tool_name.clone(), tool_name.clone()));
                let input = match event.prop("input") {
                    Some(input) => input.clone(),
                    None => self.parse_tool_input(event),
                };
                if let Some(host) = self
                    .naming
                    .host_tool_name(&tool_name, props.get("tool").and_then(Value::as_str))
                {
                    self.authorize_host_tool_call(call_id, host, input);
                    return;
                }
                let dynamic = self.naming.is_mcp_tool_name(&raw_tool_name);
                self.part(Part::ToolCall {
                    tool_call_id: call_id.clone(),
                    native_name: native_name_field(&raw_tool_name, &tool_name),
                    tool_name,
                    input,
                    dynamic,
                });
                if dynamic {
                    self.state.dynamic_tool_call_ids.insert(call_id);
                }
            }
            "session.next.tool.success" | "session.next.tool.failed" => {
                let call_id = event.prop_or_id("callID");
                let cached = self.state.tool_names.get(&call_id).cloned();
                let raw_tool_name = cached
                    .as_ref()
                    .map(|(raw, _)| raw.clone())
                    .unwrap_or_else(|| js_string(props.get("tool"), ""));
                if raw_tool_name == "StructuredOutput" || raw_tool_name == "question" {
                    return;
                }
                let tool_name = cached.map(|(_, name)| name).unwrap_or_else(|| {
                    to_wire_tool_name(if raw_tool_name.is_empty() {
                        "unknown"
                    } else {
                        &raw_tool_name
                    })
                });
                if self
                    .naming
                    .host_tool_name(&tool_name, Some(&raw_tool_name))
                    .is_some()
                {
                    return;
                }
                let result = event
                    .prop("result")
                    .or_else(|| event.prop("structured"))
                    .or_else(|| event.prop("content"))
                    .cloned()
                    .unwrap_or(Value::Null);
                self.state.dynamic_tool_call_ids.remove(&call_id);
                self.part(Part::ToolResult {
                    tool_call_id: call_id,
                    tool_name,
                    result,
                    is_error: event_type == "session.next.tool.failed",
                });
            }
            "session.next.retried" => {
                let error = event.prop("error");
                let retryable = error
                    .and_then(Value::as_object)
                    .and_then(|error| error.get("isRetryable"));
                if retryable == Some(&Value::Bool(false)) {
                    let message = error_message("OpenCode session retry failed", error);
                    self.part(Part::Error(message));
                } else {
                    self.part(Part::Warning(retry_message(event)));
                }
            }
            "session.next.step.ended" => {
                self.close_legacy_open_parts();
                let raw = js_string(props.get("finish"), "stop");
                self.part(Part::FinishStep {
                    reason: map_opencode_finish_reason(&raw),
                    raw,
                    usage: map_usage(props.get("tokens")),
                    cost: props.get("cost").and_then(Value::as_f64),
                });
            }
            "session.next.compaction.ended" => {
                let summary = js_string(props.get("text"), "");
                self.part(Part::Compaction { summary });
            }
            "file.edited" => {
                let path = self
                    .naming
                    .strip_work_dir(&js_string(props.get("file"), ""));
                self.part(Part::FileChange { path });
            }
            "session.error" | "session.next.step.failed" => {
                self.finish_compaction(true);
                let context = if event_type == "session.error" {
                    "OpenCode session error"
                } else {
                    "OpenCode step failed"
                };
                let message = error_message(context, event.prop("error"));
                self.part(Part::Error(message));
            }
            _ => {}
        }
    }

    fn parse_tool_input(&self, event: &OpenCodeEvent) -> Value {
        let call_id = js_string(event.prop("callID"), "");
        match self.state.tool_inputs.get(&call_id) {
            None => json!({}),
            Some(text) if text.is_empty() => json!({}),
            Some(text) => serde_json::from_str(text).unwrap_or_else(|_| json!({ "input": text })),
        }
    }

    fn authorize_host_tool_call(&mut self, call_id: String, tool_name: String, input: Value) {
        if !self
            .state
            .host_tool_calls_authorized
            .insert(call_id.clone())
        {
            return;
        }
        self.out.push(Output::AuthorizeHostToolCall {
            call_id,
            tool_name,
            input,
        });
    }

    fn start_legacy_part(&mut self, id: &str, reasoning: bool) {
        let ids = if reasoning {
            &mut self.state.legacy_reasoning_part_ids
        } else {
            &mut self.state.legacy_text_part_ids
        };
        if ids.iter().any(|known| known == id) {
            return;
        }
        ids.push(id.to_string());
        self.part(if reasoning {
            Part::ReasoningStart { id: id.to_string() }
        } else {
            Part::TextStart { id: id.to_string() }
        });
    }

    fn is_user_message(&self, message_id: Option<String>) -> bool {
        message_id
            .is_some_and(|id| self.state.message_roles.get(&id).map(String::as_str) == Some("user"))
    }

    /// `emitLegacyPartDelta`.
    fn emit_legacy_part_delta(&mut self, props: &Map<String, Value>) {
        let field = js_string(props.get("field"), "");
        let delta = js_string(props.get("delta"), "");
        if delta.is_empty() || self.is_user_message(string_value(props.get("messageID"))) {
            return;
        }
        let part_id = |fallback: &str| {
            string_value(props.get("partID"))
                .or_else(|| string_value(props.get("id")))
                .unwrap_or_else(|| fallback.to_string())
        };
        if field == "text" {
            let id = part_id("legacy-text");
            // OpenCode publishes reasoning deltas with field "text". A part
            // already announced as reasoning stays reasoning.
            if self.state.legacy_reasoning_part_ids.contains(&id) {
                self.state
                    .reasoning_deltas
                    .entry(id.clone())
                    .or_default()
                    .push_str(&delta);
                self.part(Part::ReasoningDelta { id, delta });
                return;
            }
            self.start_legacy_part(&id, false);
            self.state
                .text_deltas
                .entry(id.clone())
                .or_default()
                .push_str(&delta);
            self.part(Part::TextDelta { id, delta });
        } else if field == "reasoning" {
            let id = part_id("legacy-reasoning");
            self.start_legacy_part(&id, true);
            self.state
                .reasoning_deltas
                .entry(id.clone())
                .or_default()
                .push_str(&delta);
            self.part(Part::ReasoningDelta { id, delta });
        }
    }

    /// `emitLegacyTextPartUpdate`: whether the part was a text or reasoning
    /// part.
    fn emit_legacy_text_part_update(&mut self, part: Option<&Value>) -> bool {
        let Some(part) = as_object(part) else {
            return false;
        };
        let reasoning = match part.get("type").and_then(Value::as_str) {
            Some("text") => false,
            Some("reasoning") => true,
            _ => return false,
        };
        let Some(id) = string_value(part.get("id")) else {
            return true;
        };
        if self.is_user_message(string_value(part.get("messageID"))) {
            return true;
        }
        self.start_legacy_part(&id, reasoning);
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            let emitted = if reasoning {
                self.state.reasoning_deltas.get(&id)
            } else {
                self.state.text_deltas.get(&id)
            }
            .cloned()
            .unwrap_or_default();
            emit_missing_final_delta(&mut self.out, &id, Some(text), &emitted, reasoning);
            let map = if reasoning {
                &mut self.state.reasoning_deltas
            } else {
                &mut self.state.text_deltas
            };
            map.insert(id.clone(), text.to_string());
        }
        let ended = as_object(part.get("time"))
            .and_then(|time| time.get("end"))
            .is_some_and(|end| !end.is_null());
        if ended {
            if reasoning {
                self.state
                    .legacy_reasoning_part_ids
                    .retain(|known| *known != id);
                self.state.reasoning_deltas.remove(&id);
                self.part(Part::ReasoningEnd { id });
            } else {
                self.state.legacy_text_part_ids.retain(|known| *known != id);
                self.state.text_deltas.remove(&id);
                self.part(Part::TextEnd { id });
            }
        }
        true
    }

    /// `closeLegacyOpenParts`.
    fn close_legacy_open_parts(&mut self) {
        for id in std::mem::take(&mut self.state.legacy_reasoning_part_ids) {
            self.state.reasoning_deltas.remove(&id);
            self.part(Part::ReasoningEnd { id });
        }
        for id in std::mem::take(&mut self.state.legacy_text_part_ids) {
            self.state.text_deltas.remove(&id);
            self.part(Part::TextEnd { id });
        }
    }

    /// `emitLegacyStepFinishPart` (with `translateLegacyStepFinishPart`).
    fn emit_legacy_step_finish_part(&mut self, part: Option<&Value>) -> bool {
        let Some(part) = as_object(part) else {
            return false;
        };
        if part.get("type").and_then(Value::as_str) != Some("step-finish") {
            return false;
        }
        if let Some(id) = part.get("id").and_then(Value::as_str)
            && !self
                .state
                .legacy_step_finish_part_ids
                .insert(id.to_string())
        {
            return true;
        }
        self.close_legacy_open_parts();
        let raw = part
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("stop")
            .to_string();
        self.part(Part::FinishStep {
            reason: map_opencode_finish_reason(&raw),
            raw,
            usage: map_usage(part.get("tokens")),
            cost: part.get("cost").and_then(Value::as_f64),
        });
        true
    }

    /// `emitLegacyToolPart`.
    fn emit_legacy_tool_part(&mut self, part: Option<&Value>) {
        let Some(tool_part) = LegacyToolPart::from_value(part) else {
            return;
        };
        let status = tool_part.status.as_deref().unwrap_or_default();
        if !matches!(status, "running" | "completed" | "error") {
            return;
        }
        let call_id = tool_part.call_id.clone();
        let raw_tool_name = tool_part.tool.clone();
        if raw_tool_name == "StructuredOutput" || raw_tool_name == "question" {
            return;
        }
        let tool_name = to_wire_tool_name(&raw_tool_name);
        if tool_name == "agent" {
            let mut metadata = tool_part.metadata.clone();
            if let Some(state_metadata) = &tool_part.state_metadata {
                metadata.extend(state_metadata.clone());
            }
            if let (Some(parent_session_id), Some(session_id)) = (
                string_value(metadata.get("parentSessionId")),
                string_value(metadata.get("sessionId")),
            ) {
                self.out.push(Output::SubagentSession {
                    parent_session_id,
                    session_id,
                });
            }
        }
        self.state
            .tool_names
            .insert(call_id.clone(), (raw_tool_name.clone(), tool_name.clone()));
        if let Some(host) = self.naming.host_tool_name(&tool_name, Some(&raw_tool_name)) {
            if status == "running" {
                let input = Value::Object(tool_part.input.clone().unwrap_or_default());
                self.authorize_host_tool_call(call_id, host, input);
            }
            return;
        }
        let dynamic = self.naming.is_mcp_tool_name(&raw_tool_name);
        if self.state.tool_calls_emitted.insert(call_id.clone()) {
            self.part(Part::ToolCall {
                tool_call_id: call_id.clone(),
                native_name: native_name_field(&raw_tool_name, &tool_name),
                tool_name: tool_name.clone(),
                input: Value::Object(tool_part.input.clone().unwrap_or_default()),
                dynamic,
            });
            if dynamic {
                self.state.dynamic_tool_call_ids.insert(call_id.clone());
            }
        }
        if matches!(status, "completed" | "error")
            && self.state.tool_results_emitted.insert(call_id.clone())
        {
            self.state.dynamic_tool_call_ids.remove(&call_id);
            self.part(Part::ToolResult {
                tool_call_id: call_id,
                tool_name,
                result: tool_part.output(),
                is_error: status == "error",
            });
        }
    }
}

/// `LegacyToolPart`.
struct LegacyToolPart {
    call_id: String,
    tool: String,
    status: Option<String>,
    state: Option<Map<String, Value>>,
    input: Option<Map<String, Value>>,
    state_metadata: Option<Map<String, Value>>,
    metadata: Map<String, Value>,
    error: Option<Value>,
}

impl LegacyToolPart {
    /// `legacyToolPartFromValue`.
    fn from_value(value: Option<&Value>) -> Option<Self> {
        let part = as_object(value)?;
        if part.get("type").and_then(Value::as_str) != Some("tool") {
            return None;
        }
        let tool = part.get("tool")?.as_str()?.to_string();
        let call_id = part.get("callID")?.as_str()?.to_string();
        let state = as_object(part.get("state")).cloned();
        if state
            .as_ref()
            .and_then(|state| as_object(state.get("time")))
            .and_then(|time| time.get("compacted"))
            .is_some_and(|compacted| !compacted.is_null())
        {
            return None;
        }
        let status = match part.get("state") {
            Some(Value::String(status)) => Some(status.clone()),
            _ => state
                .as_ref()
                .and_then(|state| string_value(state.get("status"))),
        };
        Some(Self {
            call_id,
            tool,
            status,
            input: state
                .as_ref()
                .and_then(|state| as_object(state.get("input")))
                .cloned(),
            state_metadata: state
                .as_ref()
                .and_then(|state| as_object(state.get("metadata")))
                .cloned(),
            state,
            metadata: as_object(part.get("metadata")).cloned().unwrap_or_default(),
            error: part.get("error").filter(|error| !error.is_null()).cloned(),
        })
    }

    /// `legacyToolPartOutput`.
    fn output(&self) -> Value {
        let get = |key: &str| {
            self.state
                .as_ref()
                .and_then(|state| state.get(key))
                .filter(|value| !value.is_null())
                .cloned()
        };
        let state_status = self
            .state
            .as_ref()
            .and_then(|state| string_value(state.get("status")));
        if state_status.as_deref() == Some("error") {
            return get("error")
                .or_else(|| self.error.clone())
                .or_else(|| get("result"))
                .unwrap_or_else(|| json!("tool failed"));
        }
        get("output")
            .or_else(|| get("result"))
            .or_else(|| get("structured"))
            .or_else(|| get("content"))
            .unwrap_or(Value::Null)
    }
}

/// `permissionToolCallFromPart`: the native tool name and input of a running
/// or settled tool part.
pub(super) fn permission_tool_call_from_part(
    value: Option<&Value>,
) -> Option<(String, Map<String, Value>)> {
    let part = as_object(value)?;
    let state = as_object(part.get("state"))?;
    let input = as_object(state.get("input"))?;
    if part.get("type").and_then(Value::as_str) != Some("tool") {
        return None;
    }
    let tool = part.get("tool")?.as_str()?;
    if !matches!(
        state.get("status").and_then(Value::as_str),
        Some("running" | "completed" | "error")
    ) {
        return None;
    }
    Some((tool.to_string(), input.clone()))
}

#[cfg(test)]
#[path = "events_tests.rs"]
mod tests;
