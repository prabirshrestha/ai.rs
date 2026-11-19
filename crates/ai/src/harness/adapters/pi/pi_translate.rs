//! Port of `harness-pi/src/pi-translate.ts`.

use std::collections::{HashMap, HashSet};

use indexmap::{IndexMap, IndexSet};
use serde_json::{Map, Value as JsonValue, json};

use crate::harness::provider::{
    LanguageModelV4FinishReason, LanguageModelV4InputTokens, LanguageModelV4OutputTokens,
    LanguageModelV4UnifiedFinishReason, LanguageModelV4Usage,
};
use crate::harness::spec::stream_part::{
    HarnessV1CompactionTrigger, HarnessV1Metadata, HarnessV1StreamPart,
};

use super::pi_events::{PiSessionEvent, PiUsage, extract_assistant_text, get_pi_terminal_error};
use super::pi_utils::serialize_tool_output;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DynamicToolKind {
    Mcp,
    Extension,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolKind {
    Host,
    Builtin,
    Dynamic(DynamicToolKind),
}

/// `PiTranslatorState`: translator state shared across all events of a
/// single turn. Reset at the start of every `doPromptTurn`.
#[derive(Debug, Clone, Default)]
pub struct PiTranslatorState {
    /// True once a `turn_start` or assistant `message_start` was observed.
    pub prompt_started: bool,
    /// Accumulated assistant text from `text_delta` events.
    pub streamed_assistant_text: String,
    /// Stream-part id for the active text block.
    pub current_text_id: Option<String>,
    /// Stream-part id for the active reasoning block.
    pub current_reasoning_id: Option<String>,
    /// Whether a `reasoning-start` event has already been emitted.
    pub reasoning_started: bool,
    /// Tool-call id → tool name.
    pub observed_tool_names: HashMap<String, String>,
    /// Content-block index → tool-call id for streaming tool inputs.
    pub streaming_tool_input_ids: HashMap<i64, String>,
    /// Tool ids requested by the current assistant message but not yet
    /// completed, in request order.
    pub pending_step_tool_call_ids: IndexSet<String>,
    /// Id of the first tool call the current assistant message streamed.
    pub first_streamed_tool_call_id: Option<String>,
    /// Tool-call id → buffered `tool-input-*` parts for not-yet-started calls.
    pub buffered_tool_input_parts: HashMap<String, Vec<HarnessV1StreamPart>>,
    /// Tool calls already covered by an early `finish-step` of the step.
    pub early_finish_step_tool_call_ids: HashSet<String>,
    /// Whether the current assistant message has opened a visible step.
    pub step_open: bool,
    pub step_usage: Option<PiUsage>,
    /// Pi's session stats do not track reasoning.
    pub turn_reasoning_tokens: Option<f64>,
    /// The error of the last assistant message, if it failed.
    pub last_assistant_error: Option<String>,
    /// Tool-call id → the exact output the host submitted for a host tool.
    pub host_tool_results: HashMap<String, Option<JsonValue>>,
    dynamic_tool_calls: HashMap<String, DynamicToolKind>,
    /// Tools Pi executes natively.
    pub builtin_tool_names: HashSet<String>,
    pub host_tool_names: HashSet<String>,
    /// Native tool name → common name.
    pub native_to_common_name_map: HashMap<String, String>,
}

/// `PiTranslatorStateOptions`.
#[derive(Debug, Clone, Default)]
pub struct PiTranslatorStateOptions {
    pub builtin_tool_names: Vec<String>,
    pub host_tool_names: Vec<String>,
    pub native_to_common: IndexMap<String, String>,
}

/// `createPiTranslatorState(options)`.
pub fn create_pi_translator_state(options: PiTranslatorStateOptions) -> PiTranslatorState {
    PiTranslatorState {
        builtin_tool_names: options.builtin_tool_names.into_iter().collect(),
        host_tool_names: options.host_tool_names.into_iter().collect(),
        native_to_common_name_map: options.native_to_common.into_iter().collect(),
        ..Default::default()
    }
}

fn new_id() -> String {
    use ring::rand::SecureRandom as _;
    let mut bytes = [0u8; 8];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .expect("system randomness");
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_text_part(part: &JsonValue) -> bool {
    part.get("type").and_then(JsonValue::as_str) == Some("text")
}

/// `unwrapPiToolResult(event)`.
///
/// Pi's `tool_execution_end` payload (`result`) is an `AgentToolResult`
/// envelope; `tool_result` uses a flat shape. Text-only results without
/// details keep the string projection, including empty text. Envelopes
/// carrying details or non-text content are preserved.
///
/// [rust] An absent `details` key is TS `undefined`.
fn unwrap_pi_tool_result(event: &PiSessionEvent) -> JsonValue {
    let mut candidates: Vec<&Vec<JsonValue>> = Vec::new();
    let result = event.get("result");
    if let Some(envelope) = result.and_then(JsonValue::as_object)
        && let Some(inner) = envelope.get("content").and_then(JsonValue::as_array)
    {
        let details = envelope.get("details");
        if details.is_some_and(|details| !details.is_null())
            || inner.is_empty()
            || envelope.keys().any(|key| key != "content" && key != "details")
            || inner.iter().any(|part| !is_text_part(part))
        {
            return JsonValue::Object(envelope.clone());
        }
        candidates.push(inner);
    }
    if let Some(content) = event.get("content").and_then(JsonValue::as_array) {
        let details = event.get("details");
        if details.is_some_and(|details| !details.is_null())
            || content.is_empty()
            || content.iter().any(|part| !is_text_part(part))
        {
            let mut envelope = Map::new();
            envelope.insert("content".into(), JsonValue::Array(content.clone()));
            if let Some(details) = details {
                envelope.insert("details".into(), details.clone());
            }
            return JsonValue::Object(envelope);
        }
        candidates.push(content);
    }
    for content in candidates {
        let texts: Vec<&str> = content
            .iter()
            .filter(|part| is_text_part(part))
            .filter_map(|part| part.get("text").and_then(JsonValue::as_str))
            .collect();
        if !texts.is_empty() {
            return JsonValue::String(texts.concat());
        }
    }
    if let Some(JsonValue::String(result)) = result {
        return JsonValue::String(result.clone());
    }
    if let Some(JsonValue::String(content)) = event.get("content") {
        return JsonValue::String(content.clone());
    }
    result
        .filter(|value| !value.is_null())
        .or_else(|| event.get("content"))
        .cloned()
        .unwrap_or(JsonValue::Null)
}

fn parse_mcp_tool_result(result: JsonValue) -> JsonValue {
    let text = match &result {
        JsonValue::String(text) => Some(text.clone()),
        other => mcp_result_text(other),
    };
    let Some(text) = text else {
        return result;
    };
    serde_json::from_str(&text).unwrap_or(JsonValue::String(text))
}

fn mcp_result_text(result: &JsonValue) -> Option<String> {
    let content = result.get("content")?.as_array()?;
    if content.is_empty() {
        return None;
    }
    let mut texts = String::new();
    for part in content {
        if !is_text_part(part) {
            return None;
        }
        texts.push_str(part.get("text")?.as_str()?);
    }
    Some(texts)
}

fn resolve_tool_name(state: &PiTranslatorState, native_name: &str) -> (String, String) {
    let wire = state
        .native_to_common_name_map
        .get(native_name)
        .cloned()
        .unwrap_or_else(|| native_name.to_string());
    (wire, native_name.to_string())
}

/// `resolveToolKind`: how a tool call is dispatched, from its native name.
fn resolve_tool_kind(state: &PiTranslatorState, native_name: &str) -> ToolKind {
    if state.host_tool_names.contains(native_name) {
        ToolKind::Host
    } else if state.builtin_tool_names.contains(native_name) {
        ToolKind::Builtin
    } else if native_name.starts_with("mcp__") {
        ToolKind::Dynamic(DynamicToolKind::Mcp)
    } else {
        ToolKind::Dynamic(DynamicToolKind::Extension)
    }
}

/// `readStreamingToolCall(event)`: the `{ id, name }` of the tool call a
/// `toolcall_*` event refers to.
///
/// [rust] Pi's RPC stream drops `partial` and carries `id` and `toolName`
/// on `toolcall_start` instead (`docs/json.md`). Both shapes are read.
fn read_streaming_tool_call(event: &PiSessionEvent) -> Option<(i64, String, String)> {
    let update = event.assistant_message_event()?;
    let content_index = update.get("contentIndex")?.as_f64()? as i64;
    let non_empty = |value: Option<&JsonValue>| {
        value
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
            .map(String::from)
    };
    if let Some(block) = update
        .get("partial")
        .and_then(|partial| partial.get("content"))
        .and_then(|content| content.get(usize::try_from(content_index).ok()?))
    {
        let block = block.as_object()?;
        if block.get("type").and_then(JsonValue::as_str) != Some("toolCall") {
            return None;
        }
        return Some((
            content_index,
            non_empty(block.get("id"))?,
            non_empty(block.get("name"))?,
        ));
    }
    Some((
        content_index,
        non_empty(update.get("id"))?,
        non_empty(update.get("toolName"))?,
    ))
}

fn tokens(value: f64) -> u64 {
    value.max(0.0).round() as u64
}

/// `toHarnessUsage(usage)`.
pub fn to_harness_usage(usage: &PiUsage) -> LanguageModelV4Usage {
    LanguageModelV4Usage {
        input_tokens: LanguageModelV4InputTokens {
            total: Some(tokens(usage.input + usage.cache_read + usage.cache_write)),
            no_cache: Some(tokens(usage.input)),
            cache_read: Some(tokens(usage.cache_read)),
            cache_write: Some(tokens(usage.cache_write)),
        },
        output_tokens: LanguageModelV4OutputTokens {
            total: Some(tokens(usage.output)),
            text: usage
                .reasoning
                .map(|reasoning| tokens(usage.output - reasoning)),
            reasoning: usage.reasoning.map(tokens),
        },
        raw: None,
    }
}

const ZERO_PI_USAGE: PiUsage = PiUsage {
    input: 0.0,
    output: 0.0,
    cache_read: 0.0,
    cache_write: 0.0,
    reasoning: Some(0.0),
};

pub(crate) fn inferred_step_metadata() -> HarnessV1Metadata {
    let mut metadata = HarnessV1Metadata::new();
    metadata.insert(
        "pi".into(),
        json!({ "inferredStep": true })
            .as_object()
            .cloned()
            .unwrap_or_default(),
    );
    metadata
}

fn finish_step_part(
    unified: LanguageModelV4UnifiedFinishReason,
    raw: &str,
    usage: LanguageModelV4Usage,
) -> HarnessV1StreamPart {
    HarnessV1StreamPart::FinishStep {
        finish_reason: LanguageModelV4FinishReason {
            unified,
            raw: Some(raw.into()),
        },
        usage,
        harness_metadata: Some(inferred_step_metadata()),
        checkpoint: None,
    }
}

fn create_inferred_finish_step(usage: Option<LanguageModelV4Usage>) -> HarnessV1StreamPart {
    finish_step_part(
        LanguageModelV4UnifiedFinishReason::Stop,
        "stop",
        usage.unwrap_or_else(|| to_harness_usage(&ZERO_PI_USAGE)),
    )
}

fn finish_step(state: &mut PiTranslatorState) -> Vec<HarnessV1StreamPart> {
    if !state.step_open || !state.pending_step_tool_call_ids.is_empty() {
        return Vec::new();
    }
    state.step_open = false;
    state.pending_step_tool_call_ids.clear();
    state.early_finish_step_tool_call_ids.clear();
    let usage = to_harness_usage(&state.step_usage.take().unwrap_or(ZERO_PI_USAGE));
    vec![create_inferred_finish_step(Some(usage))]
}

/// `takePiEarlyFinishStep(state, { waitingToolCallIds, sequential })`.
///
/// Emit an early `finish-step` when Pi cannot make progress on the current
/// tool batch without host input, while keeping the step open.
pub fn take_pi_early_finish_step(
    state: &mut PiTranslatorState,
    waiting_tool_call_ids: &HashSet<String>,
    sequential: bool,
) -> Vec<HarnessV1StreamPart> {
    if !state.step_open || state.pending_step_tool_call_ids.is_empty() {
        return Vec::new();
    }
    let unsettled: Vec<String> = state.pending_step_tool_call_ids.iter().cloned().collect();
    let blocking = if sequential {
        &unsettled[..1]
    } else {
        &unsettled[..]
    };
    if !blocking.iter().all(|id| waiting_tool_call_ids.contains(id)) {
        return Vec::new();
    }
    let waiting: Vec<&String> = unsettled
        .iter()
        .filter(|id| waiting_tool_call_ids.contains(*id))
        .collect();
    if waiting
        .iter()
        .all(|id| state.early_finish_step_tool_call_ids.contains(*id))
    {
        return Vec::new();
    }
    for id in waiting {
        state.early_finish_step_tool_call_ids.insert(id.clone());
    }
    vec![finish_step_part(
        LanguageModelV4UnifiedFinishReason::ToolCalls,
        "host-input",
        to_harness_usage(&state.step_usage.unwrap_or(ZERO_PI_USAGE)),
    )]
}

fn tool_input_part_id(part: &HarnessV1StreamPart) -> &str {
    match part {
        HarnessV1StreamPart::ToolInputStart { id, .. }
        | HarnessV1StreamPart::ToolInputDelta { id, .. }
        | HarnessV1StreamPart::ToolInputEnd { id, .. } => id,
        _ => "",
    }
}

fn buffer_tool_input_part(
    state: &mut PiTranslatorState,
    part: HarnessV1StreamPart,
) -> Vec<HarnessV1StreamPart> {
    let id = tool_input_part_id(&part).to_string();
    if state.first_streamed_tool_call_id.as_deref() == Some(id.as_str()) {
        return vec![part];
    }
    state
        .buffered_tool_input_parts
        .entry(id)
        .or_default()
        .push(part);
    Vec::new()
}

fn extract_pi_tool_call_ids(event: &PiSessionEvent) -> Vec<String> {
    if event.message_role() != Some("assistant") {
        return Vec::new();
    }
    let Some(content) = event
        .message()
        .and_then(|message| message.get("content"))
        .and_then(JsonValue::as_array)
    else {
        return Vec::new();
    };
    content
        .iter()
        .filter_map(|part| {
            let block = part.as_object()?;
            if block.get("type").and_then(JsonValue::as_str) != Some("toolCall") {
                return None;
            }
            block
                .get("id")
                .or_else(|| block.get("toolCallId"))
                .and_then(JsonValue::as_str)
                .filter(|id| !id.is_empty())
                .map(String::from)
        })
        .collect()
}

fn flag(value: bool) -> Option<bool> {
    value.then_some(true)
}

/// `translatePiEvent(event, state)`: translate one Pi event into zero or
/// more stream parts, updating the translator state in place.
///
/// The translator does NOT emit `stream-start`/`finish`.
pub fn translate_pi_event(
    event: &PiSessionEvent,
    state: &mut PiTranslatorState,
) -> Vec<HarnessV1StreamPart> {
    match event.event_type() {
        "turn_start" | "message_start" => {
            let is_message_start = event.event_type() == "message_start";
            if is_message_start && event.message_role() != Some("assistant") {
                return Vec::new();
            }
            state.prompt_started = true;
            if is_message_start {
                state.step_open = true;
                state.pending_step_tool_call_ids.clear();
                state.early_finish_step_tool_call_ids.clear();
                state.step_usage = None;
                state.first_streamed_tool_call_id = None;
                state.buffered_tool_input_parts.clear();
            }
            state.streamed_assistant_text.clear();
            state.current_text_id = None;
            state.current_reasoning_id = None;
            state.reasoning_started = false;
            // Content-block indices restart with every assistant message.
            state.streaming_tool_input_ids.clear();
            Vec::new()
        }

        "message_update" => {
            if !state.prompt_started {
                return Vec::new();
            }
            let Some(update) = event.assistant_message_event() else {
                return Vec::new();
            };
            let update_type = update.get("type").and_then(JsonValue::as_str);
            let delta = update.get("delta").and_then(JsonValue::as_str);
            match (update_type, delta) {
                (Some("text_delta"), Some(delta)) => {
                    let mut parts = Vec::new();
                    if state.reasoning_started
                        && let Some(id) = state.current_reasoning_id.take()
                    {
                        parts.push(HarnessV1StreamPart::ReasoningEnd {
                            id,
                            harness_metadata: None,
                        });
                        state.reasoning_started = false;
                    }
                    let id = match &state.current_text_id {
                        Some(id) => id.clone(),
                        None => {
                            let id = new_id();
                            state.current_text_id = Some(id.clone());
                            parts.push(HarnessV1StreamPart::TextStart {
                                id: id.clone(),
                                harness_metadata: None,
                            });
                            id
                        }
                    };
                    state.streamed_assistant_text.push_str(delta);
                    parts.push(HarnessV1StreamPart::TextDelta {
                        id,
                        delta: delta.to_string(),
                        harness_metadata: None,
                    });
                    parts
                }
                (Some("thinking_delta"), Some(delta)) => {
                    let mut parts = Vec::new();
                    if let Some(id) = state.current_text_id.take() {
                        parts.push(HarnessV1StreamPart::TextEnd {
                            id,
                            harness_metadata: None,
                        });
                    }
                    let id = state
                        .current_reasoning_id
                        .get_or_insert_with(new_id)
                        .clone();
                    if !state.reasoning_started {
                        state.reasoning_started = true;
                        parts.push(HarnessV1StreamPart::ReasoningStart {
                            id: id.clone(),
                            harness_metadata: None,
                        });
                    }
                    parts.push(HarnessV1StreamPart::ReasoningDelta {
                        id,
                        delta: delta.to_string(),
                        harness_metadata: None,
                    });
                    parts
                }
                (Some("toolcall_start"), _) => {
                    let Some((content_index, id, name)) = read_streaming_tool_call(event) else {
                        return Vec::new();
                    };
                    let (wire, native) = resolve_tool_name(state, &name);
                    let kind = resolve_tool_kind(state, &native);
                    state
                        .streaming_tool_input_ids
                        .insert(content_index, id.clone());
                    state
                        .first_streamed_tool_call_id
                        .get_or_insert_with(|| id.clone());
                    buffer_tool_input_part(
                        state,
                        HarnessV1StreamPart::ToolInputStart {
                            id,
                            tool_name: wire,
                            provider_metadata: None,
                            provider_executed: flag(kind != ToolKind::Host),
                            dynamic: flag(matches!(kind, ToolKind::Dynamic(_))),
                            title: None,
                        },
                    )
                }
                (Some(kind @ ("toolcall_delta" | "toolcall_end")), _) => {
                    let Some(content_index) = update
                        .get("contentIndex")
                        .and_then(JsonValue::as_f64)
                        .map(|index| index as i64)
                    else {
                        return Vec::new();
                    };
                    // Without a start there is no id to attach the input to.
                    let Some(id) = state.streaming_tool_input_ids.get(&content_index).cloned()
                    else {
                        return Vec::new();
                    };
                    if kind == "toolcall_end" {
                        state.streaming_tool_input_ids.remove(&content_index);
                        return buffer_tool_input_part(
                            state,
                            HarnessV1StreamPart::ToolInputEnd {
                                id,
                                provider_metadata: None,
                            },
                        );
                    }
                    let Some(delta) = delta else {
                        return Vec::new();
                    };
                    buffer_tool_input_part(
                        state,
                        HarnessV1StreamPart::ToolInputDelta {
                            id,
                            delta: delta.to_string(),
                            provider_metadata: None,
                        },
                    )
                }
                _ => Vec::new(),
            }
        }

        kind @ ("message_end" | "turn_end") => {
            if !state.prompt_started {
                return Vec::new();
            }
            let mut parts = Vec::new();
            let full_text = extract_assistant_text(event.message());
            if let Some(id) = &state.current_text_id
                && full_text.starts_with(&state.streamed_assistant_text)
                && full_text.len() > state.streamed_assistant_text.len()
            {
                let missing = full_text[state.streamed_assistant_text.len()..].to_string();
                state.streamed_assistant_text = full_text;
                parts.push(HarnessV1StreamPart::TextDelta {
                    id: id.clone(),
                    delta: missing,
                    harness_metadata: None,
                });
            }
            if let Some(id) = state.current_text_id.take() {
                parts.push(HarnessV1StreamPart::TextEnd {
                    id,
                    harness_metadata: None,
                });
            }
            if state.reasoning_started
                && let Some(id) = state.current_reasoning_id.take()
            {
                parts.push(HarnessV1StreamPart::ReasoningEnd {
                    id,
                    harness_metadata: None,
                });
                state.reasoning_started = false;
            }
            if kind == "message_end" {
                let is_assistant = event.message_role() == Some("assistant");
                if is_assistant {
                    state.last_assistant_error = get_pi_terminal_error(event);
                }
                if is_assistant && let Some(usage) = event.message_usage() {
                    state.step_usage = Some(usage);
                    if let Some(reasoning) = usage.reasoning {
                        state.turn_reasoning_tokens =
                            Some(state.turn_reasoning_tokens.unwrap_or(0.0) + reasoning);
                    }
                }
                for tool_call_id in extract_pi_tool_call_ids(event) {
                    state.pending_step_tool_call_ids.insert(tool_call_id);
                }
            } else {
                state.pending_step_tool_call_ids.clear();
                if get_pi_terminal_error(event).is_none() {
                    parts.extend(finish_step(state));
                }
            }
            parts
        }

        "tool_execution_start" => {
            let (Some(tool_call_id), Some(tool_name)) = (event.tool_call_id(), event.tool_name())
            else {
                return Vec::new();
            };
            if tool_call_id.is_empty() || tool_name.is_empty() {
                return Vec::new();
            }
            let (wire, native) = resolve_tool_name(state, tool_name);
            state
                .observed_tool_names
                .insert(tool_call_id.to_string(), wire.clone());
            let kind = resolve_tool_kind(state, &native);
            if let ToolKind::Dynamic(dynamic) = kind {
                state
                    .dynamic_tool_calls
                    .insert(tool_call_id.to_string(), dynamic);
            }
            let args = event
                .get("args")
                .filter(|value| !value.is_null())
                .or_else(|| event.get("input").filter(|value| !value.is_null()));
            let input = serialize_tool_output(Some(args.unwrap_or(&json!({}))));
            let mut parts = state
                .buffered_tool_input_parts
                .remove(tool_call_id)
                .unwrap_or_default();
            parts.push(HarnessV1StreamPart::ToolCall {
                tool_call_id: tool_call_id.to_string(),
                tool_name: wire.clone(),
                input,
                provider_executed: flag(kind != ToolKind::Host),
                dynamic: flag(matches!(kind, ToolKind::Dynamic(_))),
                provider_metadata: None,
                native_name: (wire != native).then_some(native),
            });
            parts
        }

        "tool_execution_end" | "tool_result" => {
            let Some(tool_call_id) = event.tool_call_id().filter(|id| !id.is_empty()) else {
                return Vec::new();
            };
            let recorded = state.observed_tool_names.get(tool_call_id).cloned();
            let wire = recorded.or_else(|| {
                event
                    .tool_name()
                    .filter(|name| !name.is_empty())
                    .map(|name| resolve_tool_name(state, name).0)
            });
            let Some(wire) = wire else {
                return Vec::new();
            };
            let dynamic_kind = state.dynamic_tool_calls.remove(tool_call_id);
            let result = match state.host_tool_results.remove(tool_call_id) {
                Some(submitted) => submitted.unwrap_or(JsonValue::Null),
                None if dynamic_kind == Some(DynamicToolKind::Mcp) => {
                    parse_mcp_tool_result(unwrap_pi_tool_result(event))
                }
                None => unwrap_pi_tool_result(event),
            };
            state.pending_step_tool_call_ids.shift_remove(tool_call_id);
            let mut parts = vec![HarnessV1StreamPart::ToolResult {
                tool_call_id: tool_call_id.to_string(),
                tool_name: wire,
                result,
                is_error: flag(event.is_error()),
                preliminary: None,
                dynamic: flag(dynamic_kind.is_some()),
                provider_metadata: None,
            }];
            parts.extend(finish_step(state));
            parts
        }

        "compaction_end" => {
            // Pi performs the compaction itself; we observe its result.
            if event.aborted() {
                return Vec::new();
            }
            let Some(result) = event.get("result").and_then(JsonValue::as_object) else {
                return Vec::new();
            };
            let summary = result
                .get("summary")
                .and_then(JsonValue::as_str)
                .unwrap_or("(no summary provided)")
                .to_string();
            let compaction = HarnessV1StreamPart::Compaction {
                trigger: if event.reason() == Some("manual") {
                    HarnessV1CompactionTrigger::Manual
                } else {
                    HarnessV1CompactionTrigger::Auto
                },
                summary,
                tokens_before: result
                    .get("tokensBefore")
                    .and_then(JsonValue::as_f64)
                    .map(tokens),
                tokens_after: None,
                harness_metadata: None,
            };
            if state.step_open {
                vec![compaction]
            } else {
                vec![compaction, create_inferred_finish_step(None)]
            }
        }

        _ => Vec::new(),
    }
}
