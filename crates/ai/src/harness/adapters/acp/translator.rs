//! Port of `bridge/stream-translator.ts`: ACP `session/update`s as ai.rs
//! turn events.
//!
//! Tool calls the agent runs itself become tool calls of the assistant
//! message with their results, named by the builtin tool they match or
//! `acp_tool_*`. Tool calls that run a host tool are left to the host tool
//! relay, which reports them when the agent calls the tool.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};

use crate::harness::adapters::turn_builder::TurnBuilder;
use crate::harness::session::{FileChangeKind, HarnessEvent, NoticeLevel};
use crate::harness::tools::{BuiltinTool, BuiltinToolKind};
use crate::{AgentToolResult, AssistantMessage, StopReason, Usage};

use super::settings::{
    AcpResolveHostToolCall, AcpResolveHostToolCallOptions, AcpResolvedHostToolCall,
    HOST_TOOL_MCP_SERVER_NAME,
};

/// `ACPBuiltinToolMapping`.
#[derive(Debug, Clone, PartialEq)]
pub struct AcpBuiltinToolMapping {
    pub tool_name: String,
    pub native_name: Option<String>,
    pub title: Option<String>,
    pub tool_use_kind: Option<BuiltinToolKind>,
    pub input_schema: Option<Value>,
}

/// `serializeBuiltinTools`.
pub fn serialize_builtin_tools(
    tools: &indexmap::IndexMap<String, BuiltinTool>,
    titles: &indexmap::IndexMap<String, String>,
) -> Vec<AcpBuiltinToolMapping> {
    tools
        .iter()
        .map(|(name, tool)| AcpBuiltinToolMapping {
            tool_name: name.clone(),
            native_name: tool.native_name.clone(),
            title: titles.get(name).cloned(),
            tool_use_kind: tool.kind,
            input_schema: Some(tool.tool.parameters.clone()),
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockType {
    Text,
    Reasoning,
}

struct OpenBlock {
    block_type: BlockType,
    id: String,
    message_id: Option<String>,
}

#[derive(Default)]
struct ToolState {
    values: Map<String, Value>,
    emitted_file_paths: HashSet<String>,
    emitted_call: bool,
    emitted_result: bool,
    tool_name: Option<String>,
    host: Option<AcpResolvedHostToolCall>,
}

/// The finish of a prompt.
pub(super) struct Finish {
    pub stop_reason: StopReason,
    pub error: Option<String>,
}

/// `createACPStreamTranslator`, writing to a [`TurnBuilder`].
pub(super) struct Translator {
    pub turn: TurnBuilder,
    open_block: Option<OpenBlock>,
    block_counter: u64,
    block_id_counts: HashMap<String, u64>,
    tools: HashMap<String, ToolState>,
    builtins: Vec<AcpBuiltinToolMapping>,
    builtins_by_name: HashMap<String, usize>,
    resolve_host_tool_call: AcpResolveHostToolCall,
    host_tool_names: Vec<String>,
    /// Host tool calls seen in updates, for the relay to take ids from.
    host_calls: Vec<(String, String, Value)>,
    open_has_tools: bool,
    pending_starts: Vec<(String, String, Value)>,
}

impl Translator {
    pub fn new(
        turn: TurnBuilder,
        builtins: Vec<AcpBuiltinToolMapping>,
        resolve_host_tool_call: AcpResolveHostToolCall,
        host_tool_names: Vec<String>,
    ) -> Self {
        let builtins_by_name = index_builtin_tools(&builtins);
        Self {
            turn,
            open_block: None,
            block_counter: 0,
            block_id_counts: HashMap::new(),
            tools: HashMap::new(),
            builtins,
            builtins_by_name,
            resolve_host_tool_call,
            host_tool_names,
            host_calls: Vec::new(),
            open_has_tools: false,
            pending_starts: Vec::new(),
        }
    }

    pub fn notice(&self, level: NoticeLevel, message: impl Into<String>) {
        self.turn.emit_harness(HarnessEvent::Notice {
            level,
            message: message.into(),
        });
    }

    /// The assistant message tool calls belong to.
    pub fn current_assistant(&self) -> Option<AssistantMessage> {
        self.turn.current_assistant()
    }

    fn close_block(&mut self) {
        if let Some(block) = self.open_block.take() {
            self.turn.end_block(&block.id);
        }
    }

    /// End the open assistant message and start its pending tools.
    pub fn close_message(&mut self) {
        self.close_block();
        if self.turn.has_open_message() {
            self.turn.close_message(None, None, None);
        }
        self.open_has_tools = false;
        for (id, name, args) in std::mem::take(&mut self.pending_starts) {
            self.turn.tool_start(&id, &name, args);
        }
    }

    fn create_block_id(&mut self, block_type: BlockType, message_id: Option<&str>) -> String {
        let base = match message_id {
            Some(id) => id.to_string(),
            None => {
                self.block_counter += 1;
                let kind = match block_type {
                    BlockType::Text => "text",
                    BlockType::Reasoning => "reasoning",
                };
                format!("acp-{kind}-{}", self.block_counter)
            }
        };
        let count = self.block_id_counts.entry(base.clone()).or_insert(0);
        *count += 1;
        if *count == 1 {
            base
        } else {
            format!("{base}-{count}")
        }
    }

    fn emit_content(&mut self, block_type: BlockType, text: &str, message_id: Option<&str>) {
        let reopen = match &self.open_block {
            Some(block) => {
                block.block_type != block_type || block.message_id.as_deref() != message_id
            }
            None => true,
        };
        if reopen {
            self.close_block();
            if self.open_has_tools {
                self.close_message();
            }
            let id = self.create_block_id(block_type, message_id);
            self.open_block = Some(OpenBlock {
                block_type,
                id,
                message_id: message_id.map(str::to_string),
            });
        }
        let id = self.open_block.as_ref().unwrap().id.clone();
        match block_type {
            BlockType::Text => self.turn.text_delta(&id, text),
            BlockType::Reasoning => self.turn.thinking_delta(&id, text),
        }
    }

    /// Translate one `session/update`.
    pub fn update(&mut self, update: &Value) {
        let kind = update
            .get("sessionUpdate")
            .and_then(Value::as_str)
            .unwrap_or("");
        let text = update
            .get("content")
            .filter(|content| content.get("type").and_then(Value::as_str) == Some("text"))
            .and_then(|content| content.get("text"))
            .and_then(Value::as_str);
        let message_id = update.get("messageId").and_then(Value::as_str);
        match kind {
            "agent_message_chunk" | "agent_thought_chunk" => match text {
                Some(text) => {
                    let block_type = if kind == "agent_message_chunk" {
                        BlockType::Text
                    } else {
                        BlockType::Reasoning
                    };
                    self.emit_content(block_type, text, message_id);
                }
                None => self.close_block(),
            },
            "tool_call" | "tool_call_update" => self.tool_update(update, false),
            _ => {}
        }
    }

    /// `permissionToolCall`: a tool call the agent asks approval for, shown
    /// even when it is still pending. Returns its harness name and input,
    /// or `None` for a host tool call.
    pub fn permission_tool_call(&mut self, tool_call: &Value) -> Option<(String, Value)> {
        self.tool_update(tool_call, true);
        let id = tool_call.get("toolCallId").and_then(Value::as_str)?;
        let state = self.tools.get(id)?;
        if state.host.is_some() {
            return None;
        }
        Some((
            state.tool_name.clone()?,
            state
                .values
                .get("rawInput")
                .cloned()
                .unwrap_or_else(|| json!({})),
        ))
    }

    /// Merge a tool call without emitting it; returns its host tool call.
    pub fn observe_tool_call(&mut self, tool_call: &Value) -> Option<AcpResolvedHostToolCall> {
        let id = tool_call
            .get("toolCallId")
            .and_then(Value::as_str)?
            .to_string();
        let mut state = self.tools.remove(&id).unwrap_or_default();
        merge_tool_update(&mut state, tool_call);
        self.resolve_host(&id, &mut state);
        let host = state.host.clone();
        self.tools.insert(id, state);
        host
    }

    fn resolve_host(&mut self, id: &str, state: &mut ToolState) {
        if state.host.is_some() || state.emitted_call || self.host_tool_names.is_empty() {
            return;
        }
        let snapshot = create_acp_tool_call(id, state);
        let resolved = (self.resolve_host_tool_call)(AcpResolveHostToolCallOptions {
            tool_call: &snapshot,
            server_name: HOST_TOOL_MCP_SERVER_NAME,
            tool_names: &self.host_tool_names,
        });
        if let Some(resolved) = resolved.filter(|resolved| {
            self.host_tool_names.contains(&resolved.tool_name) && resolved.input.is_object()
        }) {
            self.host_calls.push((
                id.to_string(),
                resolved.tool_name.clone(),
                resolved.input.clone(),
            ));
            state.host = Some(resolved);
        }
    }

    /// The id for a host tool call the relay reports: the ACP tool call
    /// with the same tool and input, when one was seen.
    pub fn claim_host_call_id(&mut self, name: &str, args: &Value) -> String {
        match self
            .host_calls
            .iter()
            .position(|(_, tool, input)| tool == name && input == args)
        {
            Some(index) => self.host_calls.remove(index).0,
            None => format!("acp-host-{}", crate::harness::utils::random_uuid()),
        }
    }

    fn tool_update(&mut self, update: &Value, force: bool) {
        let Some(id) = update.get("toolCallId").and_then(Value::as_str) else {
            return;
        };
        let id = id.to_string();
        self.close_block();
        let mut state = self.tools.remove(&id).unwrap_or_default();
        merge_tool_update(&mut state, update);
        self.resolve_host(&id, &mut state);
        if state.host.is_none() {
            self.emit_tool_state(&id, &mut state, update, force);
        }
        self.tools.insert(id, state);
    }

    fn emit_tool_state(&mut self, id: &str, state: &mut ToolState, update: &Value, force: bool) {
        let status = state.values.get("status").and_then(Value::as_str);
        let is_new_pending = update.get("sessionUpdate").and_then(Value::as_str)
            == Some("tool_call")
            && update.get("status").and_then(Value::as_str) == Some("pending");
        if !force && !state.emitted_call && is_new_pending {
            return;
        }
        if !state.emitted_call {
            let programmatic_name = state.values.get("name").and_then(Value::as_str);
            let builtin = self.resolve_builtin(state).cloned();
            let finished = matches!(status, Some("completed" | "failed"));
            if !force
                && !finished
                && builtin.as_ref().is_some_and(|builtin| {
                    builtin.input_schema.as_ref().is_some_and(|schema| {
                        !has_required_builtin_tool_input(state.values.get("rawInput"), schema)
                    })
                })
            {
                return;
            }
            let name = match &builtin {
                Some(builtin) => builtin.tool_name.clone(),
                None => create_dynamic_tool_name(programmatic_name, id),
            };
            let input = to_safe_json(state.values.get("rawInput"), json!({}));
            if !self.turn.has_open_message() || self.open_block.is_some() {
                self.close_block();
            }
            self.turn.begin_message();
            self.turn
                .tool_call(&format!("c:{id}"), id, &name, input.clone());
            self.open_has_tools = true;
            self.pending_starts
                .push((id.to_string(), name.clone(), input));
            state.tool_name = Some(name);
            state.emitted_call = true;
        }
        self.emit_file_changes(state);
        let status = state.values.get("status").and_then(Value::as_str);
        if !state.emitted_result && matches!(status, Some("completed" | "failed")) {
            let failed = status == Some("failed");
            state.emitted_result = true;
            let name = state.tool_name.clone().unwrap_or_default();
            self.tool_result(id, &name, create_tool_result(state), failed);
        }
    }

    /// End the open message with `stop_reason`.
    pub fn close_message_with(&mut self, stop_reason: StopReason) {
        self.close_block();
        if self.turn.has_open_message() {
            self.turn.close_message(
                Some(stop_reason).filter(|reason| *reason != StopReason::Stop),
                None,
                None,
            );
        }
        self.close_message();
    }

    /// A host tool call the relay reports.
    pub fn host_tool_call(&mut self, id: &str, name: &str, args: Value) {
        self.close_block();
        self.turn.begin_message();
        self.turn
            .tool_call(&format!("c:{id}"), id, name, args.clone());
        self.open_has_tools = true;
        self.pending_starts
            .push((id.to_string(), name.to_string(), args));
    }

    /// A tool result: it follows the message holding the call.
    pub fn tool_result(&mut self, id: &str, name: &str, result: AgentToolResult, is_error: bool) {
        self.close_message();
        self.turn.tool_end(id, name, result, is_error);
    }

    fn emit_file_changes(&mut self, state: &mut ToolState) {
        let mut changes = Vec::new();
        if let Some(Value::Array(content)) = state.values.get("content") {
            for item in content {
                if item.get("type").and_then(Value::as_str) != Some("diff") {
                    continue;
                }
                let (Some(path), Some(_)) = (
                    item.get("path").and_then(Value::as_str),
                    item.get("newText").and_then(Value::as_str),
                ) else {
                    continue;
                };
                let kind = if item.get("oldText").is_none_or(Value::is_null) {
                    FileChangeKind::Create
                } else {
                    FileChangeKind::Modify
                };
                changes.push((path.to_string(), kind));
            }
        }
        if state.values.get("kind").and_then(Value::as_str) == Some("edit")
            && let Some(Value::Array(locations)) = state.values.get("locations")
        {
            for location in locations {
                if let Some(path) = location.get("path").and_then(Value::as_str) {
                    changes.push((path.to_string(), FileChangeKind::Modify));
                }
            }
        }
        for (path, kind) in changes {
            if state.emitted_file_paths.insert(path.clone()) {
                self.turn
                    .emit_harness(HarnessEvent::FileChange { kind, path });
            }
        }
    }

    fn resolve_builtin(&self, state: &ToolState) -> Option<&AcpBuiltinToolMapping> {
        if let Some(name) = state.values.get("name").and_then(Value::as_str) {
            return self
                .builtins_by_name
                .get(name)
                .map(|index| &self.builtins[*index]);
        }
        let (has_programmatic_name, matches) =
            find_builtin_tools_in_metadata(state.values.get("_meta"), self);
        if has_programmatic_name {
            return (matches.len() == 1).then(|| matches[0]);
        }
        let title = state.values.get("title").and_then(Value::as_str);
        let kind = state.values.get("kind").and_then(Value::as_str);
        if let Some(found) = self.find_builtin_by_title(title, kind) {
            return Some(found);
        }
        let raw_input = state.values.get("rawInput");
        let schema_matches: Vec<&AcpBuiltinToolMapping> = self
            .builtins
            .iter()
            .filter(|tool| matches_builtin_tool_input(raw_input, tool.input_schema.as_ref()))
            .collect();
        if schema_matches.len() <= 1 {
            return schema_matches.first().copied();
        }
        find_most_specific_schema_match(raw_input, &schema_matches)
    }

    fn find_builtin_by_title(
        &self,
        title: Option<&str>,
        kind: Option<&str>,
    ) -> Option<&AcpBuiltinToolMapping> {
        let title = title?;
        let matches: Vec<&AcpBuiltinToolMapping> = self
            .builtins
            .iter()
            .filter(|tool| {
                tool.title
                    .as_deref()
                    .is_some_and(|prefix| title.starts_with(prefix))
                    && is_compatible_tool_kind(tool.tool_use_kind, kind)
            })
            .collect();
        let longest = matches
            .iter()
            .map(|tool| tool.title.as_ref().unwrap().len())
            .max()?;
        let longest: Vec<_> = matches
            .into_iter()
            .filter(|tool| tool.title.as_ref().unwrap().len() == longest)
            .collect();
        (longest.len() == 1).then(|| longest[0])
    }

    /// The prompt finished.
    pub fn finish(&mut self, response: &Value) -> Finish {
        self.close_block();
        let stop_reason = response
            .get("stopReason")
            .and_then(Value::as_str)
            .unwrap_or("end_turn");
        if let Some(usage) = response.get("usage").filter(|usage| usage.is_object()) {
            let number = |name: &str| {
                usage
                    .get(name)
                    .and_then(Value::as_u64)
                    .map(|value| value.min(u32::MAX as u64) as u32)
            };
            let input = number("inputTokens").unwrap_or(0);
            let output = number("outputTokens").unwrap_or(0);
            let usage = Usage {
                input,
                output,
                cache_read: number("cachedReadTokens").unwrap_or(0),
                cache_write: number("cachedWriteTokens").unwrap_or(0),
                reasoning: number("thoughtTokens"),
                total_tokens: number("totalTokens").unwrap_or(input + output),
                ..Default::default()
            };
            self.turn.begin_message();
            self.turn.add_usage(&usage);
        }
        match map_acp_finish_reason(stop_reason) {
            AcpFinishReason::Stop | AcpFinishReason::Other => Finish {
                stop_reason: StopReason::Stop,
                error: None,
            },
            AcpFinishReason::Length => Finish {
                stop_reason: StopReason::Length,
                error: None,
            },
            AcpFinishReason::ContentFilter => Finish {
                stop_reason: StopReason::Error,
                error: Some("The agent refused to continue (refusal).".to_string()),
            },
            AcpFinishReason::Cancelled => Finish {
                stop_reason: StopReason::Aborted,
                error: Some("Operation aborted".to_string()),
            },
        }
    }
}

/// `mapACPFinishReason`'s unified reasons, plus `cancelled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcpFinishReason {
    Stop,
    Length,
    ContentFilter,
    /// [rust] `cancelled` is `other` in TS, where the abort decides.
    Cancelled,
    Other,
}

/// `mapACPFinishReason`.
pub fn map_acp_finish_reason(stop_reason: &str) -> AcpFinishReason {
    match stop_reason {
        "end_turn" => AcpFinishReason::Stop,
        "max_tokens" | "max_turn_requests" => AcpFinishReason::Length,
        "refusal" => AcpFinishReason::ContentFilter,
        "cancelled" => AcpFinishReason::Cancelled,
        _ => AcpFinishReason::Other,
    }
}

fn index_builtin_tools(builtins: &[AcpBuiltinToolMapping]) -> HashMap<String, usize> {
    let mut matches: HashMap<String, Option<usize>> = HashMap::new();
    let mut add = |name: &str, index: usize| match matches.get(name) {
        None => {
            matches.insert(name.to_string(), Some(index));
        }
        Some(Some(existing)) if builtins[*existing].tool_name != builtins[index].tool_name => {
            matches.insert(name.to_string(), None);
        }
        _ => {}
    };
    for (index, tool) in builtins.iter().enumerate() {
        add(&tool.tool_name, index);
        if let Some(native) = &tool.native_name {
            add(native, index);
        }
    }
    matches
        .into_iter()
        .filter_map(|(name, index)| index.map(|index| (name, index)))
        .collect()
}

fn find_builtin_tools_in_metadata<'a>(
    metadata: Option<&Value>,
    translator: &'a Translator,
) -> (bool, Vec<&'a AcpBuiltinToolMapping>) {
    let mut has_programmatic_name = false;
    let mut matches: Vec<&AcpBuiltinToolMapping> = Vec::new();
    fn visit<'a>(
        value: &Value,
        translator: &'a Translator,
        has_name: &mut bool,
        matches: &mut Vec<&'a AcpBuiltinToolMapping>,
    ) {
        let children: Vec<(Option<&str>, &Value)> = match value {
            Value::Object(map) => map.iter().map(|(k, v)| (Some(k.as_str()), v)).collect(),
            Value::Array(items) => items.iter().map(|v| (None, v)).collect(),
            _ => return,
        };
        for (property, child) in children {
            if matches!(property, Some("name" | "toolName"))
                && let Some(name) = child.as_str()
            {
                *has_name = true;
                if let Some(index) = translator.builtins_by_name.get(name) {
                    let builtin = &translator.builtins[*index];
                    if !matches.iter().any(|m| m.tool_name == builtin.tool_name) {
                        matches.push(builtin);
                    }
                }
            }
            visit(child, translator, has_name, matches);
        }
    }
    if let Some(metadata) = metadata {
        visit(
            metadata,
            translator,
            &mut has_programmatic_name,
            &mut matches,
        );
    }
    (has_programmatic_name, matches)
}

fn is_compatible_tool_kind(tool_use_kind: Option<BuiltinToolKind>, kind: Option<&str>) -> bool {
    let (Some(tool_use_kind), Some(kind)) = (tool_use_kind, kind) else {
        return true;
    };
    if !is_acp_tool_kind(kind) {
        return true;
    }
    match tool_use_kind {
        BuiltinToolKind::Readonly => matches!(kind, "read" | "search" | "fetch" | "think"),
        BuiltinToolKind::Edit => matches!(kind, "edit" | "delete" | "move"),
        BuiltinToolKind::Bash => kind == "execute",
    }
}

fn is_acp_tool_kind(kind: &str) -> bool {
    matches!(
        kind,
        "read"
            | "edit"
            | "delete"
            | "move"
            | "search"
            | "execute"
            | "think"
            | "fetch"
            | "switch_mode"
            | "other"
    )
}

fn find_most_specific_schema_match<'a>(
    raw_input: Option<&Value>,
    schema_matches: &[&'a AcpBuiltinToolMapping],
) -> Option<&'a AcpBuiltinToolMapping> {
    let raw_input = raw_input?.as_object()?;
    let coverage = |tool: &AcpBuiltinToolMapping| {
        let properties = tool
            .input_schema
            .as_ref()
            .and_then(|schema| schema.get("properties"))
            .and_then(Value::as_object);
        raw_input
            .keys()
            .filter(|key| properties.is_some_and(|properties| properties.contains_key(*key)))
            .count()
    };
    let highest = schema_matches.iter().map(|tool| coverage(tool)).max()?;
    let best: Vec<_> = schema_matches
        .iter()
        .filter(|tool| coverage(tool) == highest)
        .collect();
    (best.len() == 1).then(|| *best[0])
}

fn matches_builtin_tool_input(raw_input: Option<&Value>, schema: Option<&Value>) -> bool {
    let (Some(Value::Object(raw_input)), Some(Value::Object(schema))) = (raw_input, schema) else {
        return false;
    };
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return false;
    }
    let Some(Value::Array(required)) = schema.get("required") else {
        return false;
    };
    if required.is_empty() || !required.iter().all(Value::is_string) {
        return false;
    }
    let schema_value = Value::Object(schema.clone());
    let raw = Value::Object(raw_input.clone());
    if !has_required_builtin_tool_input(Some(&raw), &schema_value) {
        return false;
    }
    let Some(Value::Object(properties)) = schema.get("properties") else {
        return true;
    };
    properties
        .iter()
        .filter(|(_, property)| is_json_schema_discriminator(property))
        .all(|(name, property)| {
            raw_input
                .get(name)
                .is_some_and(|value| matches_json_schema_value(value, property))
        })
}

fn has_required_builtin_tool_input(raw_input: Option<&Value>, schema: &Value) -> bool {
    let Some(schema) = schema.as_object() else {
        return true;
    };
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return true;
    }
    let Some(Value::Object(raw_input)) = raw_input else {
        return false;
    };
    let Some(Value::Array(required)) = schema.get("required") else {
        return true;
    };
    if required.is_empty() || !required.iter().all(Value::is_string) {
        return true;
    }
    let properties = schema.get("properties").and_then(Value::as_object);
    required.iter().filter_map(Value::as_str).all(|name| {
        raw_input.get(name).is_some_and(|value| {
            matches_json_schema_value(
                value,
                properties
                    .and_then(|properties| properties.get(name))
                    .unwrap_or(&Value::Null),
            )
        })
    })
}

fn is_json_schema_discriminator(schema: &Value) -> bool {
    schema.as_object().is_some_and(|schema| {
        schema.contains_key("const")
            || schema
                .get("enum")
                .and_then(Value::as_array)
                .is_some_and(|values| values.len() == 1)
    })
}

fn matches_json_schema_value(value: &Value, schema: &Value) -> bool {
    match schema {
        Value::Bool(allowed) => return *allowed,
        Value::Object(_) => {}
        _ => return false,
    }
    if let Some(constant) = schema.get("const")
        && constant != value
    {
        return false;
    }
    if let Some(Value::Array(values)) = schema.get("enum")
        && !values.contains(value)
    {
        return false;
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(Value::Array(alternatives)) = schema.get(key)
            && !alternatives
                .iter()
                .any(|alternative| matches_json_schema_value(value, alternative))
        {
            return false;
        }
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("null") => value.is_null(),
        Some("boolean") => value.is_boolean(),
        Some("integer") => value.as_f64().is_some_and(|n| n.fract() == 0.0),
        Some("number") => value.is_number(),
        Some("string") => value.is_string(),
        Some("array") => value.is_array(),
        Some("object") => value.is_object(),
        _ => true,
    }
}

fn merge_tool_update(state: &mut ToolState, update: &Value) {
    for property in [
        "title",
        "kind",
        "status",
        "content",
        "locations",
        "rawInput",
        "rawOutput",
        "_meta",
    ] {
        if let Some(value) = update.get(property) {
            state.values.insert(property.to_string(), value.clone());
        }
    }
    if let Some(name) = update.get("name").and_then(Value::as_str) {
        state.values.insert("name".into(), json!(name));
    }
}

/// `createACPToolCall`: the tool call snapshot host tool resolvers see.
fn create_acp_tool_call(id: &str, state: &ToolState) -> Value {
    let mut call = Map::new();
    call.insert("toolCallId".into(), json!(id));
    if let Some(name) = state.values.get("name").filter(|v| v.is_string()) {
        call.insert("name".into(), name.clone());
    }
    let title = state
        .values
        .get("title")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| state.tool_name.clone())
        .unwrap_or_else(|| format!("Tool {id}"));
    call.insert("title".into(), json!(title));
    if let Some(kind) = state
        .values
        .get("kind")
        .filter(|kind| kind.as_str().is_some_and(is_acp_tool_kind))
    {
        call.insert("kind".into(), kind.clone());
    }
    if let Some(status) = state.values.get("status").filter(|status| {
        matches!(
            status.as_str(),
            Some("pending" | "in_progress" | "completed" | "failed")
        )
    }) {
        call.insert("status".into(), status.clone());
    }
    for key in ["content", "locations"] {
        if let Some(value) = state.values.get(key).filter(|v| v.is_array()) {
            call.insert(key.into(), value.clone());
        }
    }
    for key in ["rawInput", "rawOutput"] {
        if let Some(value) = state.values.get(key) {
            call.insert(key.into(), value.clone());
        }
    }
    if let Some(meta) = state
        .values
        .get("_meta")
        .filter(|meta| meta.is_null() || meta.is_object())
    {
        call.insert("_meta".into(), meta.clone());
    }
    Value::Object(call)
}

/// `createDynamicToolName`.
pub fn create_dynamic_tool_name(programmatic_name: Option<&str>, tool_call_id: &str) -> String {
    let source = programmatic_name
        .map(str::to_string)
        .unwrap_or_else(|| format!("acp_tool_{tool_call_id}"));
    let mut normalized = String::new();
    let mut in_run = false;
    for c in source.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            normalized.push(c);
            in_run = false;
        } else if !in_run {
            normalized.push('_');
            in_run = true;
        }
    }
    let normalized: String = normalized.trim_matches('_').chars().take(128).collect();
    if normalized.is_empty() {
        return "acp_tool".to_string();
    }
    if normalized.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        normalized
    } else {
        format!("acp_{normalized}")
    }
}

fn to_safe_json(value: Option<&Value>, fallback: Value) -> Value {
    match value {
        None | Some(Value::Null) => fallback,
        Some(value) => value.clone(),
    }
}

/// `createToolResult`: `rawOutput`, else `content`.
fn create_tool_result(state: &ToolState) -> AgentToolResult {
    let value = match state.values.get("rawOutput") {
        Some(output) if !output.is_null() => output.clone(),
        _ => to_safe_json(state.values.get("content"), json!({})),
    };
    let text = result_text(&value);
    let mut result = AgentToolResult::text(text);
    result.details = Some(value);
    result
}

/// [rust] The model-facing text of a native tool result.
fn result_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => {
            let texts: Vec<&str> = items
                .iter()
                .filter_map(|item| {
                    item.get("content")
                        .filter(|content| content.get("type") == Some(&json!("text")))
                        .and_then(|content| content.get("text"))
                        .and_then(Value::as_str)
                })
                .collect();
            if texts.is_empty() {
                value.to_string()
            } else {
                texts.join("\n")
            }
        }
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_finish_reasons() {
        assert_eq!(map_acp_finish_reason("end_turn"), AcpFinishReason::Stop);
        assert_eq!(map_acp_finish_reason("max_tokens"), AcpFinishReason::Length);
        assert_eq!(
            map_acp_finish_reason("max_turn_requests"),
            AcpFinishReason::Length
        );
        assert_eq!(
            map_acp_finish_reason("refusal"),
            AcpFinishReason::ContentFilter
        );
        assert_eq!(map_acp_finish_reason("other"), AcpFinishReason::Other);
    }

    #[test]
    fn creates_dynamic_tool_names() {
        assert_eq!(create_dynamic_tool_name(None, "call 1"), "acp_tool_call_1");
        assert_eq!(create_dynamic_tool_name(Some("mcp.tool"), "x"), "mcp_tool");
        assert_eq!(create_dynamic_tool_name(Some("1tool"), "x"), "acp_1tool");
        assert_eq!(create_dynamic_tool_name(Some("!!!"), "x"), "acp_tool");
    }

    fn mapping(
        name: &str,
        title: &str,
        kind: Option<BuiltinToolKind>,
        schema: Value,
    ) -> AcpBuiltinToolMapping {
        AcpBuiltinToolMapping {
            tool_name: name.into(),
            native_name: Some(format!("{name}ToolCall")),
            title: Some(title.into()),
            tool_use_kind: kind,
            input_schema: Some(schema),
        }
    }

    fn translator() -> Translator {
        let (sender, _events) = crate::harness::session::test_turn_sender();
        Translator::new(
            TurnBuilder::new(sender, "acp", "acp", "m"),
            vec![
                mapping(
                    "read",
                    "Read",
                    Some(BuiltinToolKind::Readonly),
                    json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }),
                ),
                mapping(
                    "bash",
                    "Terminal",
                    Some(BuiltinToolKind::Bash),
                    json!({ "type": "object", "properties": { "command": { "type": "string" } }, "required": ["command"] }),
                ),
                mapping(
                    "stopBash",
                    "Stop",
                    Some(BuiltinToolKind::Bash),
                    json!({ "type": "object", "properties": { "shellId": { "type": "string" }, "command": { "type": "string" } }, "required": ["shellId"] }),
                ),
            ],
            std::sync::Arc::new(|_| None),
            Vec::new(),
        )
    }

    #[test]
    fn resolves_builtin_tools() {
        let translator = translator();
        let state = |values: Value| ToolState {
            values: values.as_object().unwrap().clone(),
            ..Default::default()
        };
        let name = |values: Value| {
            translator
                .resolve_builtin(&state(values))
                .map(|tool| tool.tool_name.clone())
        };
        // By programmatic name, native or harness.
        assert_eq!(
            name(json!({ "name": "readToolCall" })).as_deref(),
            Some("read")
        );
        assert_eq!(name(json!({ "name": "unknown", "title": "Read" })), None);
        // By metadata names.
        assert_eq!(
            name(json!({ "_meta": { "tool": { "toolName": "bash" } } })).as_deref(),
            Some("bash")
        );
        // By title prefix with a compatible kind.
        assert_eq!(
            name(json!({ "title": "Read src/a.rs", "kind": "read" })).as_deref(),
            Some("read")
        );
        assert_eq!(
            name(json!({ "title": "Read src/a.rs", "kind": "execute" })),
            None
        );
        // By schema, most specific first.
        assert_eq!(
            name(json!({ "rawInput": { "command": "ls" } })).as_deref(),
            Some("bash")
        );
        assert_eq!(
            name(json!({ "rawInput": { "command": "ls", "shellId": "1" } })).as_deref(),
            Some("stopBash")
        );
    }

    #[test]
    fn matches_json_schema_values() {
        assert!(matches_json_schema_value(
            &json!("a"),
            &json!({ "const": "a" })
        ));
        assert!(!matches_json_schema_value(
            &json!("b"),
            &json!({ "enum": ["a"] })
        ));
        assert!(matches_json_schema_value(
            &json!(1),
            &json!({ "anyOf": [{ "type": "string" }, { "type": "integer" }] })
        ));
        assert!(!matches_json_schema_value(
            &json!(1.5),
            &json!({ "type": "integer" })
        ));
        assert!(matches_json_schema_value(&json!(null), &json!(true)));
    }
}
