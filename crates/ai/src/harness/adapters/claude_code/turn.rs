//! One Claude Code turn: the `claude` CLI in the sandbox, driven over its
//! stream-json stdio protocol the way the Claude Agent SDK drives it, with
//! its output translated into ai.rs events (the TS bridge's `runTurn` and
//! `create-emit-stream-event.ts`).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::questions::{
    ClaudeCodeQuestionResult, to_claude_code_question_result, to_harness_questions_input,
};
use super::tools::{
    ApprovalContext, HOST_TOOL_PREFIX, HOST_TOOL_SERVER, is_protected_claude_file_write,
    needs_host_approval, to_harness_name,
};
use crate::harness::adapters::host_tools::{
    ApprovalDecision, HostTools, tool_result_mcp_content, tool_result_text,
};
use crate::harness::adapters::mcp_host::{McpCallFn, McpHostServer, McpToolExtras};
use crate::harness::adapters::stdio_process::{JsonLineProcess, process_exit_error};
use crate::harness::adapters::turn_builder::TurnBuilder;
use crate::harness::questions_tool::QuestionsToolOutput;
use crate::harness::sandbox::{SandboxProcessOptions, SandboxSession};
use crate::harness::session::{HarnessEvent, HarnessTurnSender, NoticeLevel};
use crate::harness::tools::PermissionMode;
use crate::{
    AgentMessage, AgentResult, AgentToolResult, AssistantMessage, ImageContent, Message,
    StopReason, ToolCall, ToolResultContent, Usage, UserContent, UserMessageContent,
};

const INTERRUPT_GRACE: Duration = Duration::from_secs(5);
const EXIT_GRACE: Duration = Duration::from_secs(10);
const UNRECOVERABLE_API_RETRY_STATUSES: [i64; 3] = [401, 403, 404];
const APPROVAL_HOOK: &str = "harness_approval";
const QUESTION_HOOK: &str = "harness_question";
const POST_COMPACT_HOOK: &str = "harness_post_compact";
const QUESTIONS_TOOL: &str = "askUserQuestions";

/// Everything one turn needs.
pub(super) struct TurnConfig {
    pub sandbox: Arc<dyn SandboxSession>,
    pub command: String,
    pub work_dir: String,
    pub env: IndexMap<String, String>,
    pub permission_mode: PermissionMode,
    pub auto_mode: bool,
    pub inactive_native_tools: Vec<String>,
    pub readable_skill_roots: Vec<String>,
    pub additional_directories: Vec<String>,
    pub question_timeout_ms: Option<u64>,
    pub append_system_prompt: Option<String>,
    pub exclude_dynamic_sections: Option<bool>,
    pub structured_output: bool,
    pub host: HostTools,
    pub tool_hints: IndexMap<String, super::ClaudeCodeToolHint>,
    pub model: String,
    /// The Claude conversation id, updated as the CLI reports it.
    pub claude_session_id: Arc<StdMutex<Option<String>>>,
    pub steer: Arc<StdMutex<Option<SteerHandle>>>,
}

/// Lets callers add user messages to the running turn.
#[derive(Clone)]
pub(super) struct SteerHandle {
    process: Arc<JsonLineProcess>,
    internal: mpsc::UnboundedSender<Internal>,
}

impl SteerHandle {
    pub async fn steer(&self, message: AgentMessage) -> AgentResult<()> {
        let uuid = crate::harness::utils::random_uuid();
        let mut wire = user_message_wire(&message, &uuid);
        wire["priority"] = json!("next");
        self.process
            .send(&wire)
            .await
            .map_err(crate::harness::error::into_agent_error)?;
        let _ = self.internal.send(Internal::Steer { message, uuid });
        Ok(())
    }
}

/// Events from the control request handlers to the turn loop.
enum Internal {
    ToolStart {
        id: String,
        name: String,
        args: Value,
    },
    ToolUpdate {
        id: String,
        partial: AgentToolResult,
    },
    ToolEnd {
        id: String,
        name: String,
        result: AgentToolResult,
        is_error: bool,
    },
    CompactSummary(String),
    Steer {
        message: AgentMessage,
        uuid: String,
    },
}

/// Host tool calls Claude reported, to match MCP calls that come without a
/// tool use id.
#[derive(Default)]
struct HostCalls {
    reported: IndexMap<String, (String, Value)>,
    claimed: HashSet<String>,
}

impl HostCalls {
    fn resolve(&mut self, meta_id: Option<String>, name: &str, input: &Value) -> String {
        let id = meta_id
            .or_else(|| {
                self.reported
                    .iter()
                    .find(|(id, (reported_name, reported_input))| {
                        !self.claimed.contains(*id)
                            && reported_name == name
                            && reported_input == input
                    })
                    .map(|(id, _)| id.clone())
            })
            .unwrap_or_else(crate::harness::utils::random_uuid);
        self.claimed.insert(id.clone());
        id
    }
}

/// The text Claude Code takes for a user message.
fn user_message_wire(message: &AgentMessage, uuid: &str) -> Value {
    let content: Vec<Value> = match message {
        Message::User(user) => match &user.content {
            UserMessageContent::Text(text) => vec![json!({ "type": "text", "text": text })],
            UserMessageContent::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    UserContent::Text(text) => json!({ "type": "text", "text": text.text }),
                    UserContent::Image(image) => json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.mime_type,
                            "data": image.data,
                        },
                    }),
                })
                .collect(),
        },
        other => vec![json!({
            "type": "text",
            "text": serde_json::to_string(other).unwrap_or_default(),
        })],
    };
    json!({
        "type": "user",
        "message": { "role": "user", "content": content },
        "parent_tool_use_id": null,
        "uuid": uuid,
    })
}

fn usage_from(value: Option<&Value>) -> Option<Usage> {
    let value = value?.as_object()?;
    let get = |key: &str| value.get(key).and_then(Value::as_u64).unwrap_or(0) as u32;
    let input = get("input_tokens");
    let output = get("output_tokens");
    let cache_read = get("cache_read_input_tokens");
    let cache_write = get("cache_creation_input_tokens");
    Some(Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: input + output + cache_read + cache_write,
        ..Default::default()
    })
}

fn merge_usage(current: &mut Map<String, Value>, update: &Value) {
    if let Some(update) = update.as_object() {
        for (key, value) in update {
            if !value.is_null() {
                current.insert(key.clone(), value.clone());
            }
        }
    }
}

fn stop_reason(value: Option<&str>) -> Option<StopReason> {
    match value? {
        "max_tokens" | "model_context_window_exceeded" => Some(StopReason::Length),
        "refusal" => Some(StopReason::Error),
        _ => None,
    }
}

fn str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

/// A `tool_result` block's content as a tool result.
fn tool_result_from_content(content: Option<&Value>) -> AgentToolResult {
    let mut parts = Vec::new();
    match content {
        Some(Value::String(text)) => parts.push(ToolResultContent::text(text.clone())),
        Some(Value::Array(entries)) => {
            for entry in entries {
                match str_field(entry, "type") {
                    Some("text") => parts.push(ToolResultContent::text(
                        str_field(entry, "text").unwrap_or_default(),
                    )),
                    Some("image") => {
                        let source = entry.get("source").cloned().unwrap_or_default();
                        parts.push(ToolResultContent::Image(ImageContent {
                            data: str_field(&source, "data").unwrap_or_default().to_string(),
                            mime_type: str_field(&source, "media_type")
                                .unwrap_or("image/png")
                                .to_string(),
                        }));
                    }
                    _ => parts.push(ToolResultContent::text(entry.to_string())),
                }
            }
        }
        Some(other) if !other.is_null() => parts.push(ToolResultContent::text(other.to_string())),
        _ => {}
    }
    AgentToolResult {
        content: parts,
        details: None,
        usage: None,
        added_tool_names: Vec::new(),
        terminate: false,
    }
}

/// Work deferred until the open assistant message closes, so tool events
/// follow the message that called them.
enum PostAction {
    Start {
        id: String,
        name: String,
        args: Value,
    },
    Update {
        id: String,
        partial: AgentToolResult,
    },
    End {
        id: String,
        name: String,
        result: AgentToolResult,
        is_error: bool,
    },
}

/// `createCompactionLatch`: `compact_boundary` and the `PostCompact` hook
/// arrive in either order; emit one compaction event.
#[derive(Default)]
struct CompactionLatch {
    boundary: Option<(Option<u64>, Option<u64>)>,
    summary: Option<String>,
}

/// Translates the CLI's messages into the turn.
struct Translator {
    turn: TurnBuilder,
    structured_output: bool,
    /// Stream content block index → block key.
    blocks: HashMap<u64, String>,
    stream_message_id: Option<String>,
    streamed_message_ids: HashSet<String>,
    /// Assistant messages built without stream events.
    fallback_message_id: Option<String>,
    response_usage: Map<String, Value>,
    response_stop_reason: Option<String>,
    post: Vec<PostAction>,
    started_tools: HashSet<String>,
    native_names: HashMap<String, String>,
    structured_output_ids: HashSet<String>,
    host_calls: Arc<StdMutex<HostCalls>>,
    last_assistant: Arc<StdMutex<Option<AssistantMessage>>>,
    observed_error: Option<String>,
    terminal_error: Option<String>,
    compaction: CompactionLatch,
    claude_session_id: Arc<StdMutex<Option<String>>>,
    sent_message_ids: HashSet<String>,
    active_steers: HashSet<String>,
    cli_echoes_ids: bool,
    done: bool,
}

impl Translator {
    fn emit_notice(&self, level: NoticeLevel, message: String) {
        self.turn
            .emit_harness(HarnessEvent::Notice { level, message });
    }

    fn sync_assistant(&self) {
        *self.last_assistant.lock().unwrap() = self.turn.current_assistant();
    }

    fn apply(&mut self, action: PostAction) {
        if self.turn.has_open_message() {
            self.post.push(action);
            return;
        }
        match action {
            PostAction::Start { id, name, args } => {
                self.started_tools.insert(id.clone());
                self.turn.tool_start(&id, &name, args);
            }
            PostAction::Update { id, partial } => self.turn.tool_update(&id, partial),
            PostAction::End {
                id,
                name,
                result,
                is_error,
            } => self.turn.tool_end(&id, &name, result, is_error),
        }
    }

    fn close_message(&mut self, stop: Option<StopReason>, usage: Option<Usage>) {
        if !self.turn.has_open_message() {
            return;
        }
        self.turn.close_message(stop, usage, None);
        self.sync_assistant();
        for action in std::mem::take(&mut self.post) {
            self.apply(action);
        }
    }

    fn close_fallback_message(&mut self) {
        if self.fallback_message_id.take().is_some() {
            self.close_message(None, None);
        }
    }

    fn terminal(&mut self, message: impl Into<String>) {
        let message = message.into();
        let message = message.trim().to_string();
        if self.terminal_error.is_none() && !message.is_empty() {
            self.terminal_error = Some(message);
        }
        self.done = true;
    }

    fn flush_compaction(&mut self, force: bool) {
        let ready = self.compaction.boundary.is_some() && self.compaction.summary.is_some();
        if !(ready
            || force && (self.compaction.boundary.is_some() || self.compaction.summary.is_some()))
        {
            return;
        }
        let (before, after) = self.compaction.boundary.take().unwrap_or((None, None));
        let summary = self.compaction.summary.take();
        self.turn.emit_harness(HarnessEvent::Compaction {
            tokens_before: before,
            tokens_after: after,
            summary,
        });
    }

    fn on_internal(&mut self, event: Internal) {
        match event {
            Internal::ToolStart { id, name, args } => {
                self.apply(PostAction::Start { id, name, args });
            }
            Internal::ToolUpdate { id, partial } => self.apply(PostAction::Update { id, partial }),
            Internal::ToolEnd {
                id,
                name,
                result,
                is_error,
            } => self.apply(PostAction::End {
                id,
                name,
                result,
                is_error,
            }),
            Internal::CompactSummary(summary) => {
                self.compaction.summary = Some(summary);
                self.flush_compaction(false);
            }
            Internal::Steer { message, uuid } => {
                self.close_fallback_message();
                self.sent_message_ids.insert(uuid.clone());
                self.active_steers.insert(uuid);
                self.turn.user_message(message);
            }
        }
    }

    fn on_message(&mut self, message: &Value) {
        let kind = str_field(message, "type").unwrap_or_default();
        if let Some(session_id) = str_field(message, "session_id").filter(|id| !id.is_empty()) {
            *self.claude_session_id.lock().unwrap() = Some(session_id.to_string());
        }
        let is_same_fallback = kind == "assistant"
            && message
                .get("message")
                .and_then(|m| str_field(m, "id"))
                .is_some_and(|id| self.fallback_message_id.as_deref() == Some(id));
        if !is_same_fallback && kind != "stream_event" && kind != "command_lifecycle" {
            self.close_fallback_message();
        }
        let subtype = str_field(message, "subtype").unwrap_or_default();
        match (kind, subtype) {
            ("system", "init") => {
                if let Some(model) = str_field(message, "model") {
                    self.turn.set_model(model);
                }
                return;
            }
            ("system", "status") => return,
            ("system", "api_retry") => {
                let status = message.get("error_status").and_then(Value::as_i64);
                let error = str_field(message, "error");
                if let Some(status) =
                    status.filter(|status| UNRECOVERABLE_API_RETRY_STATUSES.contains(status))
                {
                    self.terminal(format!(
                        "HTTP {status}: {}",
                        error.unwrap_or("provider request failed")
                    ));
                    return;
                }
                self.emit_notice(NoticeLevel::Warning, format_api_retry_warning(message));
                return;
            }
            _ => {}
        }
        if let Some(error) = str_field(message, "error")
            .map(str::trim)
            .filter(|e| !e.is_empty())
        {
            self.observed_error = Some(error.to_string());
            if kind == "auth_status" {
                self.terminal(error);
                return;
            }
        }
        if kind == "system" && subtype == "task_updated" {
            let patch = message.get("patch");
            if patch.and_then(|p| str_field(p, "status")) == Some("failed")
                && let Some(error) = patch.and_then(|p| str_field(p, "error"))
            {
                self.terminal(error);
            }
            return;
        }
        if kind == "system" && subtype == "compact_boundary" {
            if let Some(meta) = message.get("compact_metadata") {
                self.compaction.boundary = Some((
                    meta.get("pre_tokens").and_then(Value::as_u64),
                    meta.get("post_tokens").and_then(Value::as_u64),
                ));
                self.flush_compaction(false);
            }
            return;
        }
        if message
            .get("parent_tool_use_id")
            .is_some_and(|id| !id.is_null())
        {
            // A subagent's messages: they do not belong to this conversation.
            return;
        }
        match kind {
            "stream_event" => self.on_stream_event(message.get("event").unwrap_or(&Value::Null)),
            "assistant" => self.on_assistant(message),
            "user" => self.on_user(message),
            "result" => self.on_result(message),
            "command_lifecycle" => self.on_lifecycle(message),
            _ => {}
        }
    }

    fn on_lifecycle(&mut self, message: &Value) {
        let (Some(uuid), Some(state)) = (
            str_field(message, "command_uuid"),
            str_field(message, "state"),
        ) else {
            return;
        };
        if self.sent_message_ids.contains(uuid) {
            self.cli_echoes_ids = true;
        }
        if matches!(state, "completed" | "cancelled" | "discarded") {
            self.active_steers.remove(uuid);
        }
    }

    fn on_stream_event(&mut self, event: &Value) {
        let kind = str_field(event, "type").unwrap_or_default();
        match kind {
            "message_start" => {
                self.close_message(None, None);
                let message = event.get("message").cloned().unwrap_or_default();
                self.turn.begin_message();
                if let Some(id) = str_field(&message, "id") {
                    self.turn.set_response_id(id);
                    self.streamed_message_ids.insert(id.to_string());
                    self.stream_message_id = Some(id.to_string());
                }
                self.response_usage = Map::new();
                if let Some(usage) = message.get("usage") {
                    merge_usage(&mut self.response_usage, usage);
                }
                self.response_stop_reason = None;
                self.blocks.clear();
                self.sync_assistant();
            }
            "message_delta" => {
                if let Some(usage) = event.get("usage") {
                    merge_usage(&mut self.response_usage, usage);
                }
                if let Some(reason) = event
                    .get("delta")
                    .and_then(|delta| str_field(delta, "stop_reason"))
                {
                    self.response_stop_reason = Some(reason.to_string());
                }
            }
            "message_stop" => {
                let usage = usage_from(Some(&Value::Object(std::mem::take(
                    &mut self.response_usage,
                ))));
                let stop = stop_reason(self.response_stop_reason.as_deref());
                self.stream_message_id = None;
                self.close_message(stop, usage);
            }
            "content_block_start" => {
                let Some(index) = event.get("index").and_then(Value::as_u64) else {
                    return;
                };
                self.turn.begin_message();
                let block = event.get("content_block").cloned().unwrap_or_default();
                match str_field(&block, "type") {
                    Some("text") => {
                        let key = format!("t{index}");
                        self.turn.text_delta(&key, "");
                        self.blocks.insert(index, key);
                    }
                    Some("thinking") => {
                        let key = format!("r{index}");
                        self.turn.thinking_delta(&key, "");
                        self.blocks.insert(index, key);
                    }
                    Some("tool_use") => {
                        let (Some(id), Some(name)) =
                            (str_field(&block, "id"), str_field(&block, "name"))
                        else {
                            return;
                        };
                        if name == "StructuredOutput" {
                            self.structured_output_ids.insert(id.to_string());
                            return;
                        }
                        self.native_names.insert(id.to_string(), name.to_string());
                        self.turn.tool_call_start(id, id, &to_harness_name(name));
                        self.blocks.insert(index, id.to_string());
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let Some(key) = event
                    .get("index")
                    .and_then(Value::as_u64)
                    .and_then(|index| self.blocks.get(&index))
                    .cloned()
                else {
                    return;
                };
                let delta = event.get("delta").cloned().unwrap_or_default();
                match str_field(&delta, "type") {
                    Some("text_delta") => {
                        self.turn
                            .text_delta(&key, str_field(&delta, "text").unwrap_or_default());
                    }
                    Some("thinking_delta") => {
                        self.turn.thinking_delta(
                            &key,
                            str_field(&delta, "thinking").unwrap_or_default(),
                        );
                    }
                    Some("signature_delta") => {
                        self.turn.thinking_signature(
                            &key,
                            str_field(&delta, "signature").unwrap_or_default(),
                        );
                    }
                    Some("input_json_delta") => {
                        self.turn.tool_call_delta(
                            &key,
                            str_field(&delta, "partial_json").unwrap_or_default(),
                        );
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                if let Some(key) = event
                    .get("index")
                    .and_then(Value::as_u64)
                    .and_then(|index| self.blocks.remove(&index))
                {
                    self.turn.end_block(&key);
                }
            }
            _ => {}
        }
    }

    fn register_tool_use(&mut self, id: &str, name: &str, input: &Value) {
        if name == "StructuredOutput" {
            self.structured_output_ids.insert(id.to_string());
            return;
        }
        self.native_names.insert(id.to_string(), name.to_string());
        if let Some(host_name) = name.strip_prefix(HOST_TOOL_PREFIX) {
            self.host_calls
                .lock()
                .unwrap()
                .reported
                .insert(id.to_string(), (host_name.to_string(), input.clone()));
            return;
        }
        if name == "AskUserQuestion" || self.started_tools.contains(id) {
            return;
        }
        self.started_tools.insert(id.to_string());
        self.post.push(PostAction::Start {
            id: id.to_string(),
            name: to_harness_name(name),
            args: input.clone(),
        });
    }

    fn tool_arguments(name: &str, input: &Value) -> Value {
        if name == "AskUserQuestion" {
            to_harness_questions_input(input)
        } else {
            input.clone()
        }
    }

    fn on_assistant(&mut self, message: &Value) {
        let Some(body) = message.get("message") else {
            return;
        };
        let Some(blocks) = body.get("content").and_then(Value::as_array) else {
            return;
        };
        let id = str_field(body, "id").map(str::to_string);
        let streamed = id
            .as_ref()
            .is_some_and(|id| self.streamed_message_ids.contains(id));
        if !streamed {
            // No stream events for this message: build it from the blocks.
            if self.fallback_message_id.is_none() || self.fallback_message_id != id {
                self.close_message(None, None);
                self.turn.begin_message();
                if let Some(id) = &id {
                    self.turn.set_response_id(id.clone());
                }
                if let Some(usage) = usage_from(body.get("usage")) {
                    self.turn.set_usage(usage);
                }
                self.fallback_message_id = Some(id.clone().unwrap_or_default());
            }
        }
        for (index, block) in blocks.iter().enumerate() {
            match str_field(block, "type") {
                Some("text") if !streamed => {
                    let key = format!("f{}t{index}", self.turn.messages().len());
                    self.turn
                        .text(&key, str_field(block, "text").unwrap_or_default());
                }
                Some("thinking") if !streamed => {
                    let key = format!("f{}r{index}", self.turn.messages().len());
                    self.turn.thinking(
                        &key,
                        str_field(block, "thinking").unwrap_or_default(),
                        str_field(block, "signature"),
                    );
                }
                Some("tool_use") => {
                    let (Some(tool_id), Some(name)) =
                        (str_field(block, "id"), str_field(block, "name"))
                    else {
                        continue;
                    };
                    let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                    if name != "StructuredOutput" {
                        let arguments = Self::tool_arguments(name, &input);
                        if streamed {
                            self.turn.tool_call_arguments(tool_id, arguments);
                        } else {
                            self.turn.tool_call(
                                tool_id,
                                tool_id,
                                &to_harness_name(name),
                                arguments,
                            );
                        }
                    }
                    self.register_tool_use(tool_id, name, &input);
                }
                _ => {}
            }
        }
        self.sync_assistant();
        if !self.turn.has_open_message() {
            for action in std::mem::take(&mut self.post) {
                self.apply(action);
            }
        }
    }

    fn on_user(&mut self, message: &Value) {
        let Some(blocks) = message
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(Value::as_array)
        else {
            return;
        };
        let results: Vec<&Value> = blocks
            .iter()
            .filter(|block| str_field(block, "type") == Some("tool_result"))
            .collect();
        let tool_use_result = (results.len() == 1)
            .then(|| message.get("tool_use_result").cloned())
            .flatten();
        for block in results {
            let Some(id) = str_field(block, "tool_use_id") else {
                continue;
            };
            // Host tools and questions report their own results.
            if self.structured_output_ids.remove(id)
                || self.turn.is_tool_ended(id)
                || self.host_calls.lock().unwrap().claimed.contains(id)
            {
                continue;
            }
            let native = self
                .native_names
                .get(id)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            let name = to_harness_name(&native);
            let is_error = block.get("is_error").and_then(Value::as_bool) == Some(true);
            let mut result = tool_result_from_content(block.get("content"));
            result.details = match &tool_use_result {
                Some(details) => Some(details.clone()),
                None if name == "bash" => Some(json!({
                    "exitCode": if is_error { 1 } else { 0 },
                    "stdout": tool_result_text(&result),
                })),
                None => None,
            };
            self.apply(PostAction::End {
                id: id.to_string(),
                name,
                result,
                is_error,
            });
        }
    }

    fn answers_sent_message(&mut self, message: &Value) -> bool {
        let mut ids: Vec<&str> = Vec::new();
        if let Some(id) = str_field(message, "user_message_uuid") {
            ids.push(id);
        }
        if let Some(list) = message.get("user_message_uuids").and_then(Value::as_array) {
            ids.extend(list.iter().filter_map(Value::as_str));
        }
        let echoed = ids.iter().any(|id| self.sent_message_ids.contains(*id));
        if echoed {
            self.cli_echoes_ids = true;
        }
        let origin = message
            .get("origin")
            .and_then(|origin| str_field(origin, "kind"));
        echoed || (!self.cli_echoes_ids && origin != Some("task-notification"))
    }

    fn on_result(&mut self, message: &Value) {
        self.close_message(None, None);
        let subtype = str_field(message, "subtype").unwrap_or_default();
        if subtype != "success" {
            let errors = message
                .get("errors")
                .and_then(Value::as_array)
                .map(|errors| {
                    errors
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .filter(|errors| !errors.is_empty());
            let error = errors
                .or_else(|| self.observed_error.clone())
                .or_else(|| str_field(message, "result").map(str::to_string))
                .unwrap_or_else(|| "Unknown error".to_string());
            self.terminal(error);
            return;
        }
        let result = str_field(message, "result").unwrap_or_default().trim();
        if message.get("is_error").and_then(Value::as_bool) == Some(true) {
            let error = if !result.is_empty() {
                result.to_string()
            } else if let Some(observed) = &self.observed_error {
                observed.clone()
            } else if let Some(status) = message.get("api_error_status").and_then(Value::as_i64) {
                format!("Claude Code reported an API error (HTTP {status})")
            } else {
                "Claude Code reported a failed result".to_string()
            };
            self.terminal(error);
            return;
        }
        if result.is_empty()
            && let Some(observed) = self.observed_error.clone()
        {
            self.terminal(observed);
            return;
        }
        if self.structured_output
            && let Some(output) = message.get("structured_output").filter(|v| !v.is_null())
        {
            self.turn.begin_message();
            let key = format!("structured{}", self.turn.messages().len());
            self.turn.text(&key, &output.to_string());
            self.close_message(Some(StopReason::Stop), None);
        }
        if self.answers_sent_message(message) && self.active_steers.is_empty() {
            self.done = true;
        }
    }
}

fn format_api_retry_warning(message: &Value) -> String {
    let mut details = Vec::new();
    if let Some(attempt) = message.get("attempt").and_then(Value::as_i64) {
        let max = message
            .get("max_retries")
            .and_then(Value::as_i64)
            .map(|max| format!("/{max}"))
            .unwrap_or_default();
        details.push(format!("attempt {attempt}{max}"));
    }
    if let Some(status) = message.get("error_status").and_then(Value::as_i64) {
        details.push(format!("HTTP {status}"));
    }
    if let Some(delay) = message.get("retry_delay_ms").and_then(Value::as_i64) {
        details.push(format!("retrying in {delay}ms"));
    }
    if let Some(error) = str_field(message, "error") {
        details.push(error.to_string());
    }
    if details.is_empty() {
        "Claude Code API retry".to_string()
    } else {
        format!("Claude Code API retry: {}", details.join("; "))
    }
}

/// Answers the CLI's control requests.
struct ControlContext {
    process: Arc<JsonLineProcess>,
    host: HostTools,
    mcp: Option<McpHostServer>,
    sandbox: Arc<dyn SandboxSession>,
    work_dir: String,
    permission_mode: PermissionMode,
    auto_mode: bool,
    inactive_native_tools: Vec<String>,
    readable_skill_roots: Vec<String>,
    additional_directories: Vec<String>,
    approved: StdMutex<HashSet<String>>,
    host_calls: Arc<StdMutex<HostCalls>>,
    tool_cwds: StdMutex<HashMap<String, String>>,
    last_assistant: Arc<StdMutex<Option<AssistantMessage>>>,
    model: String,
    internal: mpsc::UnboundedSender<Internal>,
    cancel: CancellationToken,
}

impl ControlContext {
    fn approval_context<'a>(&'a self, cwd: &'a str) -> ApprovalContext<'a> {
        ApprovalContext {
            sandbox: self.sandbox.as_ref(),
            work_dir: &self.work_dir,
            cwd,
            readable_skill_roots: &self.readable_skill_roots,
            additional_directories: &self.additional_directories,
            permission_mode: self.permission_mode,
            inactive_native_tools: &self.inactive_native_tools,
            abort: Some(self.cancel.clone()),
        }
    }

    fn assistant(&self) -> AssistantMessage {
        self.last_assistant
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| empty_assistant(&self.model))
    }

    /// Ask the caller about a native tool call.
    async fn ask(&self, id: &str, native_name: &str, input: &Value) -> ApprovalDecision {
        let name = to_harness_name(native_name);
        let call = ToolCall {
            id: id.to_string(),
            name: name.clone(),
            arguments: input.clone(),
            thought_signature: None,
        };
        self.host
            .approve(
                &call,
                &self.assistant(),
                ApprovalDecision::Deny {
                    reason: format!(
                        "Tool '{name}' requires approval, but no approval handler was provided."
                    ),
                },
            )
            .await
    }

    async fn can_use_tool(&self, request: &Value) -> Value {
        let tool_name = str_field(request, "tool_name").unwrap_or_default();
        let input = request.get("input").cloned().unwrap_or_else(|| json!({}));
        let id = str_field(request, "tool_use_id")
            .unwrap_or_default()
            .to_string();
        let allow =
            |input: Value| json!({ "behavior": "allow", "updatedInput": input, "toolUseID": id });
        if tool_name.starts_with(HOST_TOOL_PREFIX) {
            return allow(input);
        }
        if self.approved.lock().unwrap().remove(&id) {
            return allow(input);
        }
        if !self.auto_mode {
            let cwd = self
                .tool_cwds
                .lock()
                .unwrap()
                .get(&id)
                .cloned()
                .unwrap_or_else(|| self.work_dir.clone());
            let blocked = str_field(request, "blocked_path");
            if !needs_host_approval(&self.approval_context(&cwd), tool_name, &input, blocked).await
            {
                return allow(input);
            }
        }
        match self.ask(&id, tool_name, &input).await {
            ApprovalDecision::Allow { args } => allow(args),
            ApprovalDecision::Deny { reason } => {
                json!({ "behavior": "deny", "message": reason, "toolUseID": id })
            }
        }
    }

    async fn approval_hook(&self, input: &Value, tool_use_id: Option<&str>) -> Value {
        if str_field(input, "hook_event_name") != Some("PreToolUse") {
            return json!({});
        }
        let (Some(id), Some(tool_name)) = (
            tool_use_id.or_else(|| str_field(input, "tool_use_id")),
            str_field(input, "tool_name"),
        ) else {
            return json!({});
        };
        if tool_name == "AskUserQuestion" {
            return json!({});
        }
        let cwd = str_field(input, "cwd")
            .map(str::to_string)
            .unwrap_or_else(|| self.work_dir.clone());
        self.tool_cwds
            .lock()
            .unwrap()
            .insert(id.to_string(), cwd.clone());
        let tool_input = input
            .get("tool_input")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let context = self.approval_context(&cwd);
        let needs = if self.auto_mode {
            is_protected_claude_file_write(&context, tool_name, &tool_input).await
        } else {
            needs_host_approval(&context, tool_name, &tool_input, None).await
        };
        if !needs {
            return json!({});
        }
        match self.ask(id, tool_name, &tool_input).await {
            ApprovalDecision::Allow { args } => {
                self.approved.lock().unwrap().insert(id.to_string());
                json!({
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "updatedInput": args,
                    }
                })
            }
            ApprovalDecision::Deny { reason } => json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }),
        }
    }

    async fn question_hook(&self, input: &Value, tool_use_id: Option<&str>) -> Value {
        if str_field(input, "hook_event_name") != Some("PreToolUse")
            || str_field(input, "tool_name") != Some("AskUserQuestion")
        {
            return json!({});
        }
        let deny = |reason: String| {
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            })
        };
        let native_input = input
            .get("tool_input")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let id = tool_use_id
            .or_else(|| str_field(input, "tool_use_id"))
            .map(str::to_string)
            .unwrap_or_else(crate::harness::utils::random_uuid);
        if self.host.find(QUESTIONS_TOOL).is_none() {
            return deny(format!(
                "No '{QUESTIONS_TOOL}' tool was provided to answer the questions."
            ));
        }
        self.host_calls.lock().unwrap().claimed.insert(id.clone());
        let canonical = to_harness_questions_input(&native_input);
        let call = ToolCall {
            id: id.clone(),
            name: QUESTIONS_TOOL.to_string(),
            arguments: canonical.clone(),
            thought_signature: None,
        };
        let _ = self.internal.send(Internal::ToolStart {
            id: id.clone(),
            name: QUESTIONS_TOOL.to_string(),
            args: canonical,
        });
        let (result, is_error) = self.host.run(&call, &self.assistant(), None).await;
        let _ = self.internal.send(Internal::ToolEnd {
            id: id.clone(),
            name: QUESTIONS_TOOL.to_string(),
            result: result.clone(),
            is_error,
        });
        if is_error {
            return deny(tool_result_text(&result));
        }
        let output = result
            .details
            .clone()
            .filter(|details| details.get("action").is_some())
            .or_else(|| serde_json::from_str(&tool_result_text(&result)).ok());
        let Some(output) = output.and_then(|output| QuestionsToolOutput::parse(output).ok()) else {
            return deny("The questions tool returned an invalid answer.".to_string());
        };
        match to_claude_code_question_result(&native_input, &output) {
            ClaudeCodeQuestionResult::Allow { updated_input } => json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": updated_input,
                }
            }),
            ClaudeCodeQuestionResult::Deny { message } => deny(message),
        }
    }

    async fn handle(self: Arc<Self>, message: Value) {
        let request_id = message.get("request_id").cloned().unwrap_or(Value::Null);
        let request = message.get("request").cloned().unwrap_or_default();
        let subtype = str_field(&request, "subtype").unwrap_or_default();
        let response: Result<Value, String> = match subtype {
            "can_use_tool" => Ok(self.can_use_tool(&request).await),
            "hook_callback" => {
                let input = request.get("input").cloned().unwrap_or_default();
                let tool_use_id = str_field(&request, "tool_use_id");
                Ok(
                    match str_field(&request, "callback_id").unwrap_or_default() {
                        APPROVAL_HOOK => self.approval_hook(&input, tool_use_id).await,
                        QUESTION_HOOK => self.question_hook(&input, tool_use_id).await,
                        POST_COMPACT_HOOK => {
                            if let Some(summary) = str_field(&input, "compact_summary") {
                                let _ = self
                                    .internal
                                    .send(Internal::CompactSummary(summary.to_string()));
                            }
                            json!({})
                        }
                        other => json!({ "_unknown": other }),
                    },
                )
            }
            "mcp_message" => {
                let server = str_field(&request, "server_name").unwrap_or_default();
                match (&self.mcp, server == HOST_TOOL_SERVER) {
                    (Some(mcp), true) => {
                        let rpc = request.get("message").cloned().unwrap_or_default();
                        let answer = mcp
                            .handle(&rpc)
                            .await
                            .unwrap_or_else(|| json!({ "jsonrpc": "2.0", "result": {}, "id": 0 }));
                        Ok(json!({ "mcp_response": answer }))
                    }
                    _ => Err(format!("SDK MCP server not found: {server}")),
                }
            }
            other => Err(format!("Unsupported control request: {other}")),
        };
        let reply = match response {
            Ok(response) => json!({
                "type": "control_response",
                "response": { "subtype": "success", "request_id": request_id, "response": response },
            }),
            Err(error) => json!({
                "type": "control_response",
                "response": { "subtype": "error", "request_id": request_id, "error": error },
            }),
        };
        let _ = self.process.send(&reply).await;
    }
}

fn empty_assistant(model: &str) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: super::CLAUDE_CODE_API.to_string(),
        provider: super::CLAUDE_CODE_PROVIDER.to_string(),
        model: model.to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: crate::utils::time::now_millis(),
    }
}

/// The caller's tools as the `harness-tools` MCP server.
fn host_mcp_server(
    config: &TurnConfig,
    host_calls: Arc<StdMutex<HostCalls>>,
    last_assistant: Arc<StdMutex<Option<AssistantMessage>>>,
    internal: mpsc::UnboundedSender<Internal>,
) -> Option<McpHostServer> {
    let tools: Vec<_> = config
        .host
        .tools
        .iter()
        .map(|tool| tool.definition())
        .filter(|tool| tool.name != QUESTIONS_TOOL)
        .map(|tool| {
            let hint = config.tool_hints.get(&tool.name);
            let extras = McpToolExtras {
                annotations: hint
                    .and_then(|hint| hint.read_only)
                    .map(|read_only| json!({ "readOnlyHint": read_only })),
                meta: hint
                    .and_then(|hint| hint.always_load)
                    .map(|always| json!({ "anthropic/alwaysLoad": always })),
            };
            (tool, extras)
        })
        .collect();
    if tools.is_empty() {
        return None;
    }
    let host = config.host.clone();
    let model = config.model.clone();
    let call: McpCallFn = Arc::new(move |name, args, meta| {
        let host = host.clone();
        let host_calls = host_calls.clone();
        let last_assistant = last_assistant.clone();
        let internal = internal.clone();
        let model = model.clone();
        Box::pin(async move {
            let meta_id = meta
                .as_ref()
                .and_then(|meta| meta.get("claudecode/toolUseId"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let id = host_calls.lock().unwrap().resolve(meta_id, &name, &args);
            let _ = internal.send(Internal::ToolStart {
                id: id.clone(),
                name: name.clone(),
                args: args.clone(),
            });
            let call = ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: args,
                thought_signature: None,
            };
            let assistant = last_assistant
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| empty_assistant(&model));
            let update_sender = internal.clone();
            let update_id = id.clone();
            let on_update: crate::AgentToolUpdateCallback = Arc::new(move |partial| {
                let _ = update_sender.send(Internal::ToolUpdate {
                    id: update_id.clone(),
                    partial,
                });
                Box::pin(async {})
            });
            let (result, is_error) = host.run(&call, &assistant, Some(on_update)).await;
            let content = tool_result_mcp_content(&result);
            let _ = internal.send(Internal::ToolEnd {
                id,
                name,
                result,
                is_error,
            });
            json!({ "content": content, "isError": is_error })
        })
    });
    Some(McpHostServer {
        name: HOST_TOOL_SERVER.to_string(),
        tools,
        call,
    })
}

/// The `initialize` control request.
fn initialize_request(config: &TurnConfig, has_mcp: bool) -> Value {
    let mut pre_tool_use = Vec::new();
    if config.permission_mode != PermissionMode::AllowAll {
        pre_tool_use.push(json!({ "hookCallbackIds": [APPROVAL_HOOK] }));
    }
    let mut question = json!({ "matcher": "AskUserQuestion", "hookCallbackIds": [QUESTION_HOOK] });
    if let Some(timeout) = config.question_timeout_ms {
        question["timeout"] = json!(timeout.div_ceil(1000));
    }
    pre_tool_use.push(question);
    let mut request = json!({
        "subtype": "initialize",
        "hooks": {
            "PreToolUse": pre_tool_use,
            "PostCompact": [{ "hookCallbackIds": [POST_COMPACT_HOOK] }],
        },
    });
    if has_mcp {
        request["sdkMcpServers"] = json!([HOST_TOOL_SERVER]);
    }
    if let Some(append) = &config.append_system_prompt {
        request["appendSystemPrompt"] = json!(append);
    }
    if let Some(exclude) = config.exclude_dynamic_sections {
        request["excludeDynamicSections"] = json!(exclude);
    }
    json!({ "type": "control_request", "request_id": "initialize", "request": request })
}

/// Run one turn. Returns the turn's new messages.
pub(super) async fn run_turn(
    config: TurnConfig,
    prompt: AgentMessage,
    sender: HarnessTurnSender,
    cancel: CancellationToken,
) -> AgentResult<Vec<AgentMessage>> {
    let mut turn = TurnBuilder::new(
        sender,
        super::CLAUDE_CODE_API,
        super::CLAUDE_CODE_PROVIDER,
        config.model.clone(),
    );
    turn.start(prompt.clone());
    if cancel.is_cancelled() {
        return Ok(turn.finish(Some((StopReason::Aborted, "Operation aborted".into()))));
    }

    let spawned = JsonLineProcess::spawn(
        config.sandbox.as_ref(),
        SandboxProcessOptions {
            command: config.command.clone(),
            working_directory: Some(config.work_dir.clone()),
            env: Some(config.env.clone()),
            abort_signal: None,
        },
        "claude",
    )
    .await;
    let (process, mut output) = match spawned {
        Ok(spawned) => spawned,
        Err(error) => return Ok(turn.finish(Some((StopReason::Error, error.to_string())))),
    };

    let (internal_sender, mut internal) = mpsc::unbounded_channel();
    let host_calls: Arc<StdMutex<HostCalls>> = Arc::default();
    let last_assistant: Arc<StdMutex<Option<AssistantMessage>>> = Arc::default();
    let mcp = host_mcp_server(
        &config,
        host_calls.clone(),
        last_assistant.clone(),
        internal_sender.clone(),
    );
    let control = Arc::new(ControlContext {
        process: process.clone(),
        host: config.host.clone(),
        mcp: mcp.clone(),
        sandbox: config.sandbox.clone(),
        work_dir: config.work_dir.clone(),
        permission_mode: config.permission_mode,
        auto_mode: config.auto_mode,
        inactive_native_tools: config.inactive_native_tools.clone(),
        readable_skill_roots: config.readable_skill_roots.clone(),
        additional_directories: config.additional_directories.clone(),
        approved: StdMutex::default(),
        host_calls: host_calls.clone(),
        tool_cwds: StdMutex::default(),
        last_assistant: last_assistant.clone(),
        model: config.model.clone(),
        internal: internal_sender.clone(),
        cancel: cancel.clone(),
    });

    let prompt_id = crate::harness::utils::random_uuid();
    let mut translator = Translator {
        turn,
        structured_output: config.structured_output,
        blocks: HashMap::new(),
        stream_message_id: None,
        streamed_message_ids: HashSet::new(),
        fallback_message_id: None,
        response_usage: Map::new(),
        response_stop_reason: None,
        post: Vec::new(),
        started_tools: HashSet::new(),
        native_names: HashMap::new(),
        structured_output_ids: HashSet::new(),
        host_calls,
        last_assistant,
        observed_error: None,
        terminal_error: None,
        compaction: CompactionLatch::default(),
        claude_session_id: config.claude_session_id.clone(),
        sent_message_ids: HashSet::from([prompt_id.clone()]),
        active_steers: HashSet::new(),
        cli_echoes_ids: false,
        done: false,
    };

    let init = initialize_request(&config, mcp.is_some());
    let sent = async {
        process.send(&init).await?;
        process.send(&user_message_wire(&prompt, &prompt_id)).await
    }
    .await;
    if let Err(error) = sent {
        process.kill().await;
        let error = process_exit_error(&process, &format!("could not start: {error}"));
        return Ok(translator
            .turn
            .finish(Some((StopReason::Error, error.to_string()))));
    }
    *config.steer.lock().unwrap() = Some(SteerHandle {
        process: process.clone(),
        internal: internal_sender.clone(),
    });

    let mut interrupted = false;
    let mut kill_at: Option<tokio::time::Instant> = None;
    let mut exited_early = false;
    loop {
        let kill_sleep = async {
            match kill_at {
                Some(at) => tokio::time::sleep_until(at).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            message = output.recv() => {
                let Some(message) = message else {
                    exited_early = !translator.done;
                    break;
                };
                match str_field(&message, "type") {
                    Some("control_request") => {
                        tokio::spawn(control.clone().handle(message));
                    }
                    Some("control_response") => {
                        let response = message.get("response").cloned().unwrap_or_default();
                        if str_field(&response, "request_id") == Some("initialize")
                            && str_field(&response, "subtype") == Some("error")
                        {
                            translator.terminal(
                                str_field(&response, "error")
                                    .unwrap_or("Claude Code failed to initialize"),
                            );
                        }
                    }
                    Some("control_cancel_request") | Some("keep_alive") => {}
                    _ => translator.on_message(&message),
                }
            }
            Some(event) = internal.recv() => translator.on_internal(event),
            _ = cancel.cancelled(), if !interrupted => {
                interrupted = true;
                let _ = process
                    .send(&json!({
                        "type": "control_request",
                        "request_id": "interrupt",
                        "request": { "subtype": "interrupt" },
                    }))
                    .await;
                kill_at = Some(tokio::time::Instant::now() + INTERRUPT_GRACE);
            }
            _ = kill_sleep => {
                process.kill().await;
                kill_at = None;
            }
        }
        if translator.done {
            break;
        }
    }
    *config.steer.lock().unwrap() = None;
    // Apply what the handlers reported before the process ended.
    while let Ok(event) = internal.try_recv() {
        translator.on_internal(event);
    }
    translator.close_fallback_message();
    translator.flush_compaction(true);

    process.close_stdin().await;
    let exited = !(translator.terminal_error.is_some() || interrupted)
        && tokio::time::timeout(EXIT_GRACE, process.wait())
            .await
            .is_ok();
    if !exited {
        process.kill().await;
    }

    let error = if interrupted {
        Some((StopReason::Aborted, "Operation aborted".to_string()))
    } else if let Some(error) = translator.terminal_error.take() {
        Some((StopReason::Error, error))
    } else if exited_early {
        let _ = process.wait().await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        Some((
            StopReason::Error,
            process_exit_error(&process, "exited before the turn finished").to_string(),
        ))
    } else {
        None
    };
    Ok(translator.turn.finish(error))
}
