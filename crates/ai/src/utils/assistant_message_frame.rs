//! Port of `utils/assistant-message-frame.ts`: a compact, replayable delta
//! frame format for streaming assistant messages.
//!
//! In Pi `partial` is one shared live accumulator that can run ahead of the
//! event being encoded; the encoder's per-block offsets drop deltas already
//! visible in a start snapshot. Rust events carry partial snapshots, so the
//! offsets usually start at zero, but the logic is kept as is.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, StopReason, TextContent,
    ThinkingContent, ToolCall,
};
use crate::utils::estimate::js_length;
use crate::utils::json_parse::parse_streaming_json;
use crate::{Error, Result};

/// Compact, replayable assistant-message progress. Terminal settlement is
/// intentionally excluded and must be persisted separately.
#[allow(clippy::large_enum_variant)] // mirrors Pi's plain union
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all_fields = "camelCase")]
pub enum AssistantMessageFrame {
    #[serde(rename = "start")]
    Start { partial: AssistantMessage },
    /// `content` is a `text` block.
    #[serde(rename = "text_start")]
    TextStart {
        content_index: usize,
        content: AssistantContent,
    },
    #[serde(rename = "text_delta")]
    TextDelta { content_index: usize, delta: String },
    #[serde(rename = "text_end")]
    TextEnd {
        content_index: usize,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text_signature: Option<String>,
    },
    /// `content` is a `thinking` block.
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        content_index: usize,
        content: AssistantContent,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { content_index: usize, delta: String },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        content_index: usize,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        redacted: Option<bool>,
    },
    /// `tool_call` is a `toolCall` block.
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        content_index: usize,
        tool_call: AssistantContent,
    },
    #[serde(rename = "toolcall_checkpoint")]
    ToolCallCheckpoint { content_index: usize, json: String },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta { content_index: usize, delta: String },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        content_index: usize,
        id: String,
        name: String,
        arguments: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
    },
}

impl AssistantMessageFrame {
    pub fn frame_type(&self) -> &'static str {
        match self {
            Self::Start { .. } => "start",
            Self::TextStart { .. } => "text_start",
            Self::TextDelta { .. } => "text_delta",
            Self::TextEnd { .. } => "text_end",
            Self::ThinkingStart { .. } => "thinking_start",
            Self::ThinkingDelta { .. } => "thinking_delta",
            Self::ThinkingEnd { .. } => "thinking_end",
            Self::ToolCallStart { .. } => "toolcall_start",
            Self::ToolCallCheckpoint { .. } => "toolcall_checkpoint",
            Self::ToolCallDelta { .. } => "toolcall_delta",
            Self::ToolCallEnd { .. } => "toolcall_end",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolCall,
}

impl BlockKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Thinking => "thinking",
            Self::ToolCall => "toolCall",
        }
    }

    fn of(block: &AssistantContent) -> Self {
        match block {
            AssistantContent::Text(_) => Self::Text,
            AssistantContent::Thinking(_) => Self::Thinking,
            AssistantContent::ToolCall(_) => Self::ToolCall,
        }
    }
}

enum EncoderBlockState {
    Text {
        kind: BlockKind,
        covered_chars: usize,
        delta_chars: usize,
    },
    ToolCall {
        caught_up: bool,
        catchup_json: String,
        snapshot_arguments: String,
    },
}

impl EncoderBlockState {
    fn kind(&self) -> BlockKind {
        match self {
            Self::Text { kind, .. } => *kind,
            Self::ToolCall { .. } => BlockKind::ToolCall,
        }
    }
}

fn error(message: impl Into<String>) -> Error {
    Error::message(message)
}

fn clone_text_content(content: &TextContent) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: content.text.clone(),
        text_signature: content.text_signature.clone(),
    })
}

fn clone_thinking_content(content: &ThinkingContent) -> AssistantContent {
    AssistantContent::Thinking(ThinkingContent {
        thinking: content.thinking.clone(),
        thinking_signature: content.thinking_signature.clone(),
        redacted: content.redacted,
    })
}

fn clone_start_message(message: &AssistantMessage) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: message.api.clone(),
        provider: message.provider.clone(),
        model: message.model.clone(),
        response_model: message.response_model.clone(),
        response_id: message.response_id.clone(),
        provider_thinking_level: message.provider_thinking_level.clone(),
        thinking_level: None,
        diagnostics: message.diagnostics.clone(),
        usage: message.usage.clone(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: message.timestamp,
    }
}

fn event_block<'a>(
    event_type: &str,
    content_index: usize,
    partial: &'a AssistantMessage,
) -> Result<&'a AssistantContent> {
    partial.content.get(content_index).ok_or_else(|| {
        error(format!(
            "{event_type} event has no content block at index {content_index}"
        ))
    })
}

fn wrong_kind(event_type: &str, block: &AssistantContent, content_index: usize) -> Error {
    error(format!(
        "{event_type} event points to {} block at index {content_index}",
        BlockKind::of(block).as_str()
    ))
}

fn serialized_arguments(arguments: &Value) -> String {
    serde_json::to_string(arguments).unwrap_or_default()
}

fn empty_parsed_tool_arguments() -> String {
    serialized_arguments(&parse_streaming_json(Some("")))
}

fn is_json_prefix(snapshot: &Value, current: &Value) -> bool {
    match snapshot {
        Value::String(snapshot) => current
            .as_str()
            .is_some_and(|current| current.starts_with(snapshot.as_str())),
        Value::Array(snapshot) => current.as_array().is_some_and(|current| {
            snapshot.len() <= current.len()
                && snapshot
                    .iter()
                    .zip(current)
                    .all(|(value, current)| is_json_prefix(value, current))
        }),
        Value::Object(snapshot) => current.as_object().is_some_and(|current| {
            snapshot.iter().all(|(key, value)| {
                current
                    .get(key)
                    .is_some_and(|current| is_json_prefix(value, current))
            })
        }),
        _ => snapshot == current,
    }
}

/// Encodes one assistant stream into frames.
#[derive(Default)]
pub struct AssistantMessageFrameEncoder {
    started: bool,
    terminal: bool,
    blocks: BTreeMap<usize, EncoderBlockState>,
}

impl AssistantMessageFrameEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn encode(
        &mut self,
        event: &AssistantMessageEvent,
    ) -> Result<Option<AssistantMessageFrame>> {
        if self.terminal {
            return Err(error(format!(
                "Assistant message event {} follows a terminal event",
                event.event_type()
            )));
        }

        match event {
            AssistantMessageEvent::Start { partial } => {
                if self.started {
                    return Err(error(
                        "Assistant message stream contains more than one start event",
                    ));
                }
                self.started = true;
                return Ok(Some(AssistantMessageFrame::Start {
                    partial: clone_start_message(partial),
                }));
            }
            AssistantMessageEvent::Done { .. } => {
                if !self.started {
                    return Err(error("Assistant message done event appears before start"));
                }
                self.terminal = true;
                return Ok(None);
            }
            AssistantMessageEvent::Error { .. } => {
                self.terminal = true;
                return Ok(None);
            }
            _ => {}
        }

        if !self.started {
            return Err(error(format!(
                "Assistant message {} event appears before start",
                event.event_type()
            )));
        }

        let event_type = event.event_type();
        match event {
            AssistantMessageEvent::TextStart {
                content_index,
                partial,
            } => {
                let block = event_block(event_type, *content_index, partial)?;
                let AssistantContent::Text(content) = block else {
                    return Err(wrong_kind(event_type, block, *content_index));
                };
                self.start_block(
                    *content_index,
                    EncoderBlockState::Text {
                        kind: BlockKind::Text,
                        covered_chars: js_length(&content.text),
                        delta_chars: 0,
                    },
                )?;
                Ok(Some(AssistantMessageFrame::TextStart {
                    content_index: *content_index,
                    content: clone_text_content(content),
                }))
            }
            AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                ..
            } => self.encode_text_delta(*content_index, delta, BlockKind::Text),
            AssistantMessageEvent::TextEnd {
                content_index,
                content,
                partial,
            } => {
                let block = event_block(event_type, *content_index, partial)?;
                let AssistantContent::Text(block) = block else {
                    return Err(wrong_kind(event_type, block, *content_index));
                };
                self.end_block(*content_index, BlockKind::Text)?;
                Ok(Some(AssistantMessageFrame::TextEnd {
                    content_index: *content_index,
                    content: content.clone(),
                    text_signature: block.text_signature.clone(),
                }))
            }
            AssistantMessageEvent::ThinkingStart {
                content_index,
                partial,
            } => {
                let block = event_block(event_type, *content_index, partial)?;
                let AssistantContent::Thinking(content) = block else {
                    return Err(wrong_kind(event_type, block, *content_index));
                };
                self.start_block(
                    *content_index,
                    EncoderBlockState::Text {
                        kind: BlockKind::Thinking,
                        covered_chars: js_length(&content.thinking),
                        delta_chars: 0,
                    },
                )?;
                Ok(Some(AssistantMessageFrame::ThinkingStart {
                    content_index: *content_index,
                    content: clone_thinking_content(content),
                }))
            }
            AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                ..
            } => self.encode_text_delta(*content_index, delta, BlockKind::Thinking),
            AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
                partial,
            } => {
                let block = event_block(event_type, *content_index, partial)?;
                let AssistantContent::Thinking(block) = block else {
                    return Err(wrong_kind(event_type, block, *content_index));
                };
                self.end_block(*content_index, BlockKind::Thinking)?;
                Ok(Some(AssistantMessageFrame::ThinkingEnd {
                    content_index: *content_index,
                    content: content.clone(),
                    thinking_signature: block.thinking_signature.clone(),
                    redacted: block.redacted,
                }))
            }
            AssistantMessageEvent::ToolCallStart {
                content_index,
                partial,
            } => {
                let block = event_block(event_type, *content_index, partial)?;
                let AssistantContent::ToolCall(content) = block else {
                    return Err(wrong_kind(event_type, block, *content_index));
                };
                let snapshot_arguments = serialized_arguments(&content.arguments);
                let caught_up = snapshot_arguments == empty_parsed_tool_arguments();
                self.start_block(
                    *content_index,
                    EncoderBlockState::ToolCall {
                        caught_up,
                        catchup_json: String::new(),
                        snapshot_arguments: if caught_up {
                            String::new()
                        } else {
                            snapshot_arguments
                        },
                    },
                )?;
                Ok(Some(AssistantMessageFrame::ToolCallStart {
                    content_index: *content_index,
                    tool_call: AssistantContent::ToolCall(content.clone()),
                }))
            }
            AssistantMessageEvent::ToolCallDelta {
                content_index,
                delta,
                ..
            } => {
                let state = self.block(*content_index, BlockKind::ToolCall)?;
                let EncoderBlockState::ToolCall {
                    caught_up,
                    catchup_json,
                    snapshot_arguments,
                } = state
                else {
                    return Err(error("Unreachable tool-call encoder state"));
                };
                if *caught_up {
                    return Ok(
                        (!delta.is_empty()).then(|| AssistantMessageFrame::ToolCallDelta {
                            content_index: *content_index,
                            delta: delta.clone(),
                        }),
                    );
                }
                catchup_json.push_str(delta);
                let arguments = parse_streaming_json(Some(catchup_json));
                if serialized_arguments(&arguments) != *snapshot_arguments {
                    // Legacy grammar calls include the initial input in
                    // toolcall_start, but their JSON delta stream still begins at
                    // an empty input, so the parsed arguments can extend the
                    // start snapshot.
                    let snapshot = parse_streaming_json(Some(snapshot_arguments));
                    if !is_json_prefix(&snapshot, &arguments) {
                        return Ok(None);
                    }
                }
                *caught_up = true;
                snapshot_arguments.clear();
                let json = std::mem::take(catchup_json);
                Ok(
                    (!json.is_empty()).then(|| AssistantMessageFrame::ToolCallCheckpoint {
                        content_index: *content_index,
                        json,
                    }),
                )
            }
            AssistantMessageEvent::ToolCallEnd {
                content_index,
                tool_call,
                partial,
            } => {
                let block = event_block(event_type, *content_index, partial)?;
                if !matches!(block, AssistantContent::ToolCall(_)) {
                    return Err(wrong_kind(event_type, block, *content_index));
                }
                self.end_block(*content_index, BlockKind::ToolCall)?;
                Ok(Some(AssistantMessageFrame::ToolCallEnd {
                    content_index: *content_index,
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                    thought_signature: tool_call.thought_signature.clone(),
                    namespace: tool_call.namespace.clone(),
                }))
            }
            AssistantMessageEvent::Start { .. }
            | AssistantMessageEvent::Done { .. }
            | AssistantMessageEvent::Error { .. } => unreachable!("handled above"),
        }
    }

    fn start_block(&mut self, content_index: usize, state: EncoderBlockState) -> Result<()> {
        if self.blocks.contains_key(&content_index) {
            return Err(error(format!(
                "Assistant message block {content_index} starts more than once"
            )));
        }
        self.blocks.insert(content_index, state);
        Ok(())
    }

    fn block(&mut self, content_index: usize, kind: BlockKind) -> Result<&mut EncoderBlockState> {
        let Some(state) = self.blocks.get_mut(&content_index) else {
            return Err(error(format!(
                "Assistant message {} block {content_index} has not started",
                kind.as_str()
            )));
        };
        if state.kind() != kind {
            return Err(error(format!(
                "Assistant message block {content_index} is {}, not {}",
                state.kind().as_str(),
                kind.as_str()
            )));
        }
        Ok(state)
    }

    fn end_block(&mut self, content_index: usize, kind: BlockKind) -> Result<()> {
        self.block(content_index, kind)?;
        self.blocks.remove(&content_index);
        Ok(())
    }

    fn encode_text_delta(
        &mut self,
        content_index: usize,
        delta: &str,
        kind: BlockKind,
    ) -> Result<Option<AssistantMessageFrame>> {
        let state = self.block(content_index, kind)?;
        let EncoderBlockState::Text {
            covered_chars,
            delta_chars,
            ..
        } = state
        else {
            return Err(error("Unreachable text encoder state"));
        };
        let delta_start = *delta_chars;
        let delta_length = js_length(delta);
        *delta_chars += delta_length;
        let covered = covered_chars.saturating_sub(delta_start);
        if covered >= delta_length {
            return Ok(None);
        }
        let uncovered = if covered == 0 {
            delta.to_string()
        } else {
            js_slice_from(delta, covered)
        };
        Ok(Some(match kind {
            BlockKind::Text => AssistantMessageFrame::TextDelta {
                content_index,
                delta: uncovered,
            },
            _ => AssistantMessageFrame::ThinkingDelta {
                content_index,
                delta: uncovered,
            },
        }))
    }
}

/// `string.slice(start)` in UTF-16 code units. A character whose surrogate
/// pair straddles `start` is skipped (JS would keep its lone low surrogate):
/// the result starts at the first character boundary at or after `start`.
fn js_slice_from(text: &str, start: usize) -> String {
    let mut units = 0;
    for (index, ch) in text.char_indices() {
        if units >= start {
            return text[index..].to_string();
        }
        units += ch.len_utf16();
    }
    String::new()
}

struct ReducerBlockState {
    kind: BlockKind,
    ended: bool,
    json: String,
}

fn append_block(
    message: &mut AssistantMessage,
    states: &mut BTreeMap<usize, ReducerBlockState>,
    content_index: usize,
    block: AssistantContent,
) -> Result<()> {
    if content_index != message.content.len() {
        let reason = if content_index < message.content.len() {
            "already exists"
        } else {
            "would leave a gap"
        };
        return Err(error(format!(
            "Cannot start assistant message block at index {content_index}: {reason}"
        )));
    }
    states.insert(
        content_index,
        ReducerBlockState {
            kind: BlockKind::of(&block),
            ended: false,
            json: String::new(),
        },
    );
    message.content.push(block);
    Ok(())
}

fn active_block<'a>(
    message: &'a mut AssistantMessage,
    states: &'a mut BTreeMap<usize, ReducerBlockState>,
    content_index: usize,
    expected_kind: BlockKind,
    frame_type: &str,
) -> Result<(&'a mut AssistantContent, &'a mut ReducerBlockState)> {
    let (Some(state), Some(block)) = (
        states.get_mut(&content_index),
        message.content.get_mut(content_index),
    ) else {
        return Err(error(format!(
            "{frame_type} frame has no started block at index {content_index}"
        )));
    };
    if state.kind != expected_kind || BlockKind::of(block) != expected_kind {
        return Err(error(format!(
            "{frame_type} frame expected {} block at index {content_index}, found {}",
            expected_kind.as_str(),
            BlockKind::of(block).as_str()
        )));
    }
    if state.ended {
        return Err(error(format!(
            "{frame_type} frame follows the end of block at index {content_index}"
        )));
    }
    Ok((block, state))
}

/// Replay compact frames. Returns `None` when there is no start frame.
pub fn reduce_assistant_message_frames<'a>(
    frames: impl IntoIterator<Item = &'a AssistantMessageFrame>,
) -> Result<Option<AssistantMessage>> {
    let mut message: Option<AssistantMessage> = None;
    let mut frame_before_start: Option<&'static str> = None;
    let mut states: BTreeMap<usize, ReducerBlockState> = BTreeMap::new();

    for frame in frames {
        if let AssistantMessageFrame::Start { partial } = frame {
            if message.is_some() {
                return Err(error(
                    "Assistant message frame sequence contains more than one start frame",
                ));
            }
            if let Some(frame_type) = frame_before_start {
                return Err(error(format!(
                    "{frame_type} frame appears before the start frame"
                )));
            }
            message = Some(partial.clone());
            continue;
        }
        let Some(message) = message.as_mut() else {
            frame_before_start.get_or_insert(frame.frame_type());
            continue;
        };
        let frame_type = frame.frame_type();

        match frame {
            AssistantMessageFrame::Start { .. } => unreachable!("handled above"),
            AssistantMessageFrame::TextStart {
                content_index,
                content,
            } => {
                if !matches!(content, AssistantContent::Text(_)) {
                    return Err(error(format!(
                        "text_start frame contains {} content",
                        BlockKind::of(content).as_str()
                    )));
                }
                append_block(message, &mut states, *content_index, content.clone())?;
            }
            AssistantMessageFrame::TextDelta {
                content_index,
                delta,
            } => {
                let (block, _) = active_block(
                    message,
                    &mut states,
                    *content_index,
                    BlockKind::Text,
                    frame_type,
                )?;
                if let AssistantContent::Text(block) = block {
                    block.text.push_str(delta);
                }
            }
            AssistantMessageFrame::TextEnd {
                content_index,
                content,
                text_signature,
            } => {
                let (block, state) = active_block(
                    message,
                    &mut states,
                    *content_index,
                    BlockKind::Text,
                    frame_type,
                )?;
                if let AssistantContent::Text(block) = block {
                    block.text = content.clone();
                    block.text_signature = text_signature.clone();
                }
                state.ended = true;
            }
            AssistantMessageFrame::ThinkingStart {
                content_index,
                content,
            } => {
                if !matches!(content, AssistantContent::Thinking(_)) {
                    return Err(error(format!(
                        "thinking_start frame contains {} content",
                        BlockKind::of(content).as_str()
                    )));
                }
                append_block(message, &mut states, *content_index, content.clone())?;
            }
            AssistantMessageFrame::ThinkingDelta {
                content_index,
                delta,
            } => {
                let (block, _) = active_block(
                    message,
                    &mut states,
                    *content_index,
                    BlockKind::Thinking,
                    frame_type,
                )?;
                if let AssistantContent::Thinking(block) = block {
                    block.thinking.push_str(delta);
                }
            }
            AssistantMessageFrame::ThinkingEnd {
                content_index,
                content,
                thinking_signature,
                redacted,
            } => {
                let (block, state) = active_block(
                    message,
                    &mut states,
                    *content_index,
                    BlockKind::Thinking,
                    frame_type,
                )?;
                if let AssistantContent::Thinking(block) = block {
                    block.thinking = content.clone();
                    block.thinking_signature = thinking_signature.clone();
                    block.redacted = *redacted;
                }
                state.ended = true;
            }
            AssistantMessageFrame::ToolCallStart {
                content_index,
                tool_call,
            } => {
                if !matches!(tool_call, AssistantContent::ToolCall(_)) {
                    return Err(error(format!(
                        "toolcall_start frame contains {} content",
                        BlockKind::of(tool_call).as_str()
                    )));
                }
                append_block(message, &mut states, *content_index, tool_call.clone())?;
            }
            AssistantMessageFrame::ToolCallCheckpoint {
                content_index,
                json,
            } => {
                let (block, state) = active_block(
                    message,
                    &mut states,
                    *content_index,
                    BlockKind::ToolCall,
                    frame_type,
                )?;
                state.json = json.clone();
                if let AssistantContent::ToolCall(block) = block {
                    block.arguments = parse_streaming_json(Some(json));
                }
            }
            AssistantMessageFrame::ToolCallDelta {
                content_index,
                delta,
            } => {
                let (_, state) = active_block(
                    message,
                    &mut states,
                    *content_index,
                    BlockKind::ToolCall,
                    frame_type,
                )?;
                state.json.push_str(delta);
            }
            AssistantMessageFrame::ToolCallEnd {
                content_index,
                id,
                name,
                arguments,
                thought_signature,
                namespace,
            } => {
                let (block, state) = active_block(
                    message,
                    &mut states,
                    *content_index,
                    BlockKind::ToolCall,
                    frame_type,
                )?;
                if let AssistantContent::ToolCall(block) = block {
                    *block = ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                        thought_signature: thought_signature.clone(),
                        namespace: namespace.clone(),
                    };
                }
                state.ended = true;
            }
        }
    }

    let Some(mut message) = message else {
        return Ok(None);
    };
    for (content_index, state) in &states {
        if state.kind != BlockKind::ToolCall || state.ended || state.json.is_empty() {
            continue;
        }
        let Some(AssistantContent::ToolCall(block)) = message.content.get_mut(*content_index)
        else {
            return Err(error("Unreachable tool-call frame state"));
        };
        block.arguments = parse_streaming_json(Some(&state.json));
    }

    Ok(Some(message))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::Model;
    use crate::utils::diagnostics::AssistantMessageDiagnostic;

    fn seed() -> AssistantMessage {
        let mut message = AssistantMessage::empty_for(&Model {
            id: "test-model".to_string(),
            api: "test-api".to_string(),
            provider: "test-provider".to_string(),
            ..Default::default()
        });
        message.stop_reason = StopReason::Pending;
        message.timestamp = 1;
        message
    }

    fn frame(
        encoder: &mut AssistantMessageFrameEncoder,
        event: AssistantMessageEvent,
    ) -> AssistantMessageFrame {
        encoder
            .encode(&event)
            .unwrap()
            .unwrap_or_else(|| panic!("Expected {} event to produce a frame", event.event_type()))
    }

    fn text(text: &str) -> AssistantContent {
        AssistantContent::text(text)
    }

    fn tool_call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
            thought_signature: None,
            namespace: None,
        }
    }

    fn reduce(frames: &[AssistantMessageFrame]) -> Option<AssistantMessage> {
        reduce_assistant_message_frames(frames).unwrap()
    }

    fn reduce_error(frames: &[AssistantMessageFrame]) -> String {
        reduce_assistant_message_frames(frames)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn uses_authoritative_text_end_content_and_signature() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        partial.content.push(text("Hello "));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::TextStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        partial.content[0] = AssistantContent::Text(TextContent {
            text: "Hello world".to_string(),
            text_signature: Some("sig-text".to_string()),
        });
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "incorrect".to_string(),
                partial: partial.clone(),
            },
        ));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::TextEnd {
                content_index: 0,
                content: "Hello world".to_string(),
                partial: partial.clone(),
            },
        ));

        assert_eq!(
            frames.last().unwrap(),
            &AssistantMessageFrame::TextEnd {
                content_index: 0,
                content: "Hello world".to_string(),
                text_signature: Some("sig-text".to_string()),
            }
        );
        assert_eq!(
            reduce(&frames).unwrap().content,
            vec![AssistantContent::Text(TextContent {
                text: "Hello world".to_string(),
                text_signature: Some("sig-text".to_string()),
            })]
        );
    }

    #[test]
    fn preserves_provider_thinking_level_from_the_stream_start() {
        let mut partial = seed();
        partial.provider_thinking_level = Some("high".to_string());
        let mut encoder = AssistantMessageFrameEncoder::new();
        let start = frame(&mut encoder, AssistantMessageEvent::Start { partial });
        assert_eq!(
            reduce(&[start]).unwrap().provider_thinking_level.as_deref(),
            Some("high")
        );
    }

    #[test]
    fn preserves_initial_and_final_thinking_metadata_including_redaction() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        partial
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: "[redacted]".to_string(),
                thinking_signature: Some("encrypted-start".to_string()),
                redacted: Some(true),
            }));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ThinkingStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        let final_block = ThinkingContent {
            thinking: "[redacted]".to_string(),
            thinking_signature: Some("encrypted-final".to_string()),
            redacted: Some(true),
        };
        partial.content[0] = AssistantContent::Thinking(final_block.clone());
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ThinkingEnd {
                content_index: 0,
                content: "[redacted]".to_string(),
                partial,
            },
        ));

        assert_eq!(
            frames.last().unwrap(),
            &AssistantMessageFrame::ThinkingEnd {
                content_index: 0,
                content: "[redacted]".to_string(),
                thinking_signature: Some("encrypted-final".to_string()),
                redacted: Some(true),
            }
        );
        assert_eq!(
            reduce(&frames).unwrap().content[0],
            AssistantContent::Thinking(final_block)
        );
    }

    #[test]
    fn parses_unfinished_tool_json_once_and_uses_authoritative_completed_arguments() {
        let mut frames = vec![
            AssistantMessageFrame::Start { partial: seed() },
            AssistantMessageFrame::ToolCallStart {
                content_index: 0,
                tool_call: AssistantContent::ToolCall(tool_call("initial-id", "write", json!({}))),
            },
            AssistantMessageFrame::ToolCallDelta {
                content_index: 0,
                delta: r#"{"path":"READ"#.to_string(),
            },
        ];
        let AssistantContent::ToolCall(partial_call) = &reduce(&frames).unwrap().content[0] else {
            panic!("expected tool call");
        };
        assert_eq!(partial_call.arguments, json!({ "path": "READ" }));

        frames.push(AssistantMessageFrame::ToolCallDelta {
            content_index: 0,
            delta: r#"ME.md","lines":[1,2]}"#.to_string(),
        });
        frames.push(AssistantMessageFrame::ToolCallEnd {
            content_index: 0,
            id: "final-id".to_string(),
            name: "write_file".to_string(),
            arguments: json!({ "path": "final.md", "lines": [3] }),
            thought_signature: Some("thought".to_string()),
            namespace: Some("files".to_string()),
        });
        assert_eq!(
            reduce(&frames).unwrap().content[0],
            AssistantContent::ToolCall(ToolCall {
                thought_signature: Some("thought".to_string()),
                namespace: Some("files".to_string()),
                ..tool_call(
                    "final-id",
                    "write_file",
                    json!({ "path": "final.md", "lines": [3] })
                )
            })
        );
    }

    #[test]
    fn reconciles_queued_text_events_against_one_advanced_live_partial_without_duplicate_content() {
        // Pi queues events that share one live partial, which has advanced to
        // the final text by the time they are encoded. Rust partials are
        // snapshots, so every event carries that advanced snapshot.
        let mut advanced = seed();
        advanced.content.push(text("Hello world"));
        let mut events = vec![
            AssistantMessageEvent::Start {
                partial: advanced.clone(),
            },
            AssistantMessageEvent::TextStart {
                content_index: 0,
                partial: advanced.clone(),
            },
        ];
        for delta in ["Hel", "lo", " ", "world"] {
            events.push(AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: delta.to_string(),
                partial: advanced.clone(),
            });
        }

        let mut encoder = AssistantMessageFrameEncoder::new();
        let frames: Vec<_> = events
            .iter()
            .filter_map(|event| encoder.encode(event).unwrap())
            .collect();
        assert_eq!(
            frames
                .iter()
                .map(AssistantMessageFrame::frame_type)
                .collect::<Vec<_>>(),
            ["start", "text_start"]
        );
        let AssistantMessageFrame::Start { partial } = &frames[0] else {
            panic!("expected a start frame");
        };
        assert!(partial.content.is_empty());
        assert_eq!(partial.stop_reason, StopReason::Pending);
        assert_eq!(reduce(&frames).unwrap().content, vec![text("Hello world")]);
    }

    #[test]
    fn trims_only_the_covered_prefix_when_a_start_snapshot_lands_inside_a_delta() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        partial.content.push(text("Hel"));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::TextStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        let covered = encoder
            .encode(&AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "He".to_string(),
                partial: partial.clone(),
            })
            .unwrap();
        assert_eq!(covered, None);
        let remainder = frame(
            &mut encoder,
            AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "llo".to_string(),
                partial,
            },
        );
        assert_eq!(
            remainder,
            AssistantMessageFrame::TextDelta {
                content_index: 0,
                delta: "lo".to_string()
            }
        );
        frames.push(remainder);
        assert_eq!(reduce(&frames).unwrap().content, vec![text("Hello")]);
    }

    #[test]
    fn checkpoints_queued_tool_json_without_replaying_covered_deltas() {
        let mut partial = seed();
        partial.content.push(AssistantContent::ToolCall(tool_call(
            "call",
            "write",
            json!({}),
        )));
        let mut events = vec![
            AssistantMessageEvent::Start { partial: seed() },
            AssistantMessageEvent::ToolCallStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ];
        partial.content[0] =
            AssistantContent::ToolCall(tool_call("call", "write", json!({ "path": "README.md" })));
        events.push(AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta: r#"{"path":"READ"#.to_string(),
            partial: partial.clone(),
        });
        events.push(AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta: r#"ME.md"}"#.to_string(),
            partial,
        });

        let mut encoder = AssistantMessageFrameEncoder::new();
        let frames: Vec<_> = events
            .iter()
            .filter_map(|event| encoder.encode(event).unwrap())
            .collect();
        assert_eq!(
            frames
                .iter()
                .map(AssistantMessageFrame::frame_type)
                .collect::<Vec<_>>(),
            vec![
                "start",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta"
            ]
        );
        assert_eq!(
            reduce(&frames).unwrap().content,
            vec![AssistantContent::ToolCall(tool_call(
                "call",
                "write",
                json!({ "path": "README.md" })
            ))]
        );
    }

    #[test]
    fn resumes_legacy_grammar_tool_json_from_initial_arguments() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        partial.content.push(AssistantContent::ToolCall(tool_call(
            "call",
            "bash",
            json!({ "input": "a" }),
        )));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolCallStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolCallDelta {
                content_index: 0,
                delta: r#"{"input":"ab"#.to_string(),
                partial: partial.clone(),
            },
        ));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolCallDelta {
                content_index: 0,
                delta: r#"c"}"#.to_string(),
                partial,
            },
        ));

        assert_eq!(
            frames[2..],
            [
                AssistantMessageFrame::ToolCallCheckpoint {
                    content_index: 0,
                    json: r#"{"input":"ab"#.to_string()
                },
                AssistantMessageFrame::ToolCallDelta {
                    content_index: 0,
                    delta: r#"c"}"#.to_string()
                },
            ]
        );
        assert_eq!(
            reduce(&frames).unwrap().content,
            vec![AssistantContent::ToolCall(tool_call(
                "call",
                "bash",
                json!({ "input": "abc" })
            ))]
        );
    }

    #[test]
    fn streams_tool_json_compactly_from_an_empty_argument_start() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        let mut frames = vec![frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        )];
        partial.content.push(AssistantContent::ToolCall(tool_call(
            "call",
            "bash",
            json!({}),
        )));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolCallStart {
                content_index: 0,
                partial: partial.clone(),
            },
        ));
        frames.push(frame(
            &mut encoder,
            AssistantMessageEvent::ToolCallDelta {
                content_index: 0,
                delta: r#"{"command":"ls -la /tmp"}"#.to_string(),
                partial,
            },
        ));
        assert_eq!(
            frames.last().unwrap(),
            &AssistantMessageFrame::ToolCallDelta {
                content_index: 0,
                delta: r#"{"command":"ls -la /tmp"}"#.to_string()
            }
        );
        let AssistantContent::ToolCall(call) = &reduce(&frames).unwrap().content[0] else {
            panic!("expected tool call");
        };
        assert_eq!(call.arguments, json!({ "command": "ls -la /tmp" }));
    }

    #[test]
    fn accepts_a_pre_generation_error_but_rejects_success_or_updates_before_start() {
        let mut failed = seed();
        failed.stop_reason = StopReason::Error;
        failed.error_message = Some("setup failed".to_string());
        assert_eq!(
            AssistantMessageFrameEncoder::new()
                .encode(&AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: failed,
                })
                .unwrap(),
            None
        );

        let mut completed = seed();
        completed.stop_reason = StopReason::Stop;
        let error = AssistantMessageFrameEncoder::new()
            .encode(&AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: completed,
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("done event appears before start")
        );
        let error = AssistantMessageFrameEncoder::new()
            .encode(&AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "x".to_string(),
                partial: seed(),
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("text_delta event appears before start")
        );
    }

    #[test]
    fn treats_end_signature_metadata_including_absence_as_authoritative() {
        let frames = vec![
            AssistantMessageFrame::Start { partial: seed() },
            AssistantMessageFrame::TextStart {
                content_index: 0,
                content: AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: Some("stale-text".to_string()),
                }),
            },
            AssistantMessageFrame::TextEnd {
                content_index: 0,
                content: String::new(),
                text_signature: None,
            },
            AssistantMessageFrame::ThinkingStart {
                content_index: 1,
                content: AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some("stale-thinking".to_string()),
                    redacted: Some(true),
                }),
            },
            AssistantMessageFrame::ThinkingEnd {
                content_index: 1,
                content: String::new(),
                thinking_signature: Some(String::new()),
                redacted: Some(false),
            },
            AssistantMessageFrame::ToolCallStart {
                content_index: 2,
                tool_call: AssistantContent::ToolCall(ToolCall {
                    thought_signature: Some("stale-tool".to_string()),
                    namespace: Some("stale-namespace".to_string()),
                    ..tool_call("call", "read", json!({}))
                }),
            },
            AssistantMessageFrame::ToolCallEnd {
                content_index: 2,
                id: "call".to_string(),
                name: "read".to_string(),
                arguments: json!({}),
                thought_signature: None,
                namespace: None,
            },
        ];
        assert_eq!(
            reduce(&frames).unwrap().content,
            vec![
                text(""),
                AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some(String::new()),
                    redacted: Some(false),
                }),
                AssistantContent::ToolCall(tool_call("call", "read", json!({}))),
            ]
        );
    }

    #[test]
    fn stores_authoritative_final_arguments_in_toolcall_end_frames() {
        let call = ToolCall {
            thought_signature: Some("thought".to_string()),
            namespace: Some("files".to_string()),
            ..tool_call("call-1", "read", json!({ "path": "README.md" }))
        };
        let mut partial = seed();
        partial
            .content
            .push(AssistantContent::ToolCall(call.clone()));
        let mut encoder = AssistantMessageFrameEncoder::new();
        frame(
            &mut encoder,
            AssistantMessageEvent::Start {
                partial: partial.clone(),
            },
        );
        frame(
            &mut encoder,
            AssistantMessageEvent::ToolCallStart {
                content_index: 0,
                partial: partial.clone(),
            },
        );
        let end = frame(
            &mut encoder,
            AssistantMessageEvent::ToolCallEnd {
                content_index: 0,
                tool_call: call,
                partial,
            },
        );
        assert_eq!(
            serde_json::to_value(end).unwrap(),
            json!({
                "type": "toolcall_end",
                "contentIndex": 0,
                "id": "call-1",
                "name": "read",
                "arguments": { "path": "README.md" },
                "thoughtSignature": "thought",
                "namespace": "files"
            })
        );
    }

    #[test]
    fn supports_interleaved_streams_by_content_index() {
        let frames = vec![
            AssistantMessageFrame::Start { partial: seed() },
            AssistantMessageFrame::TextStart {
                content_index: 0,
                content: text(""),
            },
            AssistantMessageFrame::ToolCallStart {
                content_index: 1,
                tool_call: AssistantContent::ToolCall(tool_call("call", "lookup", json!({}))),
            },
            AssistantMessageFrame::ThinkingStart {
                content_index: 2,
                content: AssistantContent::Thinking(ThinkingContent::default()),
            },
            AssistantMessageFrame::TextDelta {
                content_index: 0,
                delta: "answer".to_string(),
            },
            AssistantMessageFrame::ToolCallDelta {
                content_index: 1,
                delta: r#"{"query":"pi"}"#.to_string(),
            },
            AssistantMessageFrame::ThinkingDelta {
                content_index: 2,
                delta: "check".to_string(),
            },
            AssistantMessageFrame::ToolCallEnd {
                content_index: 1,
                id: "call".to_string(),
                name: "lookup".to_string(),
                arguments: json!({ "query": "pi" }),
                thought_signature: None,
                namespace: None,
            },
            AssistantMessageFrame::TextEnd {
                content_index: 0,
                content: "answer".to_string(),
                text_signature: None,
            },
            AssistantMessageFrame::ThinkingEnd {
                content_index: 2,
                content: "check".to_string(),
                thinking_signature: None,
                redacted: None,
            },
        ];
        assert_eq!(
            reduce(&frames).unwrap().content,
            vec![
                text("answer"),
                AssistantContent::ToolCall(tool_call("call", "lookup", json!({ "query": "pi" }))),
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "check".to_string(),
                    ..Default::default()
                }),
            ]
        );
    }

    #[test]
    fn snapshots_start_data_and_omits_stream_scratch() {
        let mut partial = seed();
        partial.diagnostics = Some(vec![AssistantMessageDiagnostic {
            diagnostic_type: "test".to_string(),
            timestamp: 2,
            error: None,
            details: Some(json!({ "value": "original" }).as_object().unwrap().clone()),
        }]);
        partial.content.push(text("already streamed"));
        partial.stop_reason = StopReason::Stop;
        let mut encoder = AssistantMessageFrameEncoder::new();
        let AssistantMessageFrame::Start { partial: start } =
            frame(&mut encoder, AssistantMessageEvent::Start { partial })
        else {
            panic!("expected start frame");
        };
        assert!(start.content.is_empty());
        assert_eq!(start.stop_reason, StopReason::Pending);
        assert_eq!(
            start.diagnostics.unwrap()[0].details.as_ref().unwrap()["value"],
            "original"
        );
    }

    #[test]
    fn omits_terminal_events_because_settlement_is_separate() {
        let mut message = seed();
        let mut completed = AssistantMessageFrameEncoder::new();
        completed
            .encode(&AssistantMessageEvent::Start {
                partial: message.clone(),
            })
            .unwrap();
        message.stop_reason = StopReason::Stop;
        assert_eq!(
            completed
                .encode(&AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message: message.clone(),
                })
                .unwrap(),
            None
        );
        assert!(
            completed
                .encode(&AssistantMessageEvent::Start { partial: message })
                .unwrap_err()
                .to_string()
                .contains("follows a terminal event")
        );
    }

    #[test]
    fn returns_none_when_there_is_no_start_frame() {
        assert_eq!(reduce(&[]), None);
        assert_eq!(
            reduce(&[AssistantMessageFrame::TextDelta {
                content_index: 0,
                delta: "x".to_string()
            }]),
            None
        );
    }

    #[test]
    fn rejects_frames_before_start_wrong_block_kinds_duplicate_ends_and_index_gaps() {
        assert!(
            reduce_error(&[
                AssistantMessageFrame::TextDelta {
                    content_index: 0,
                    delta: "x".to_string()
                },
                AssistantMessageFrame::Start { partial: seed() },
            ])
            .contains("before the start frame")
        );
        assert!(
            reduce_error(&[
                AssistantMessageFrame::Start { partial: seed() },
                AssistantMessageFrame::ToolCallStart {
                    content_index: 0,
                    tool_call: AssistantContent::ToolCall(tool_call("call", "run", json!({}))),
                },
                AssistantMessageFrame::TextDelta {
                    content_index: 0,
                    delta: "wrong".to_string()
                },
            ])
            .contains("expected text block")
        );
        assert!(
            reduce_error(&[
                AssistantMessageFrame::Start { partial: seed() },
                AssistantMessageFrame::TextStart {
                    content_index: 0,
                    content: text("")
                },
                AssistantMessageFrame::TextEnd {
                    content_index: 0,
                    content: String::new(),
                    text_signature: None
                },
                AssistantMessageFrame::TextEnd {
                    content_index: 0,
                    content: String::new(),
                    text_signature: None
                },
            ])
            .contains("follows the end")
        );
        assert!(
            reduce_error(&[
                AssistantMessageFrame::Start { partial: seed() },
                AssistantMessageFrame::TextStart {
                    content_index: 1,
                    content: text("")
                },
            ])
            .contains("would leave a gap")
        );
    }

    #[test]
    fn rejects_conversion_events_whose_content_index_points_to_the_wrong_block_kind() {
        let mut partial = seed();
        let mut encoder = AssistantMessageFrameEncoder::new();
        encoder
            .encode(&AssistantMessageEvent::Start {
                partial: partial.clone(),
            })
            .unwrap();
        partial
            .content
            .push(AssistantContent::Thinking(ThinkingContent::default()));
        let error = encoder
            .encode(&AssistantMessageEvent::TextStart {
                content_index: 0,
                partial,
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("text_start event points to thinking block")
        );
    }

    #[test]
    fn frames_serialize_with_pi_field_names() {
        let value = serde_json::to_value(AssistantMessageFrame::TextStart {
            content_index: 0,
            content: text(""),
        })
        .unwrap();
        assert_eq!(
            value,
            json!({ "type": "text_start", "contentIndex": 0, "content": { "type": "text", "text": "" } })
        );
    }
}
