//! Mirror of `ai` `ui/process-ui-message-stream.ts` and
//! `error/ui-message-stream-error.ts`.
//!
//! [rust] Only the part `handleUIMessageStreamFinish` drives is ported: the
//! chat-client options (`messageMetadataSchema`, `dataPartSchemas`,
//! `onToolCall`, `onData`, `resetStateOnMessageIdChange` and
//! `resetStateOnFirstMessageStart`) are left out, so applying a chunk is
//! synchronous.

use indexmap::IndexMap;

use crate::harness::provider::{JsonObject, JsonValue};
use crate::harness::sdk::content::{FinishReason, ProviderMetadata};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::output::parse_partial_json;
use crate::harness::sdk::ui_message_stream::ui_message_chunks::UIMessageChunk;
use crate::harness::sdk::ui_message_stream::ui_messages::{
    InputEncoding, ReasoningUIPart, TextUIPart, UIMessage, UIMessagePart, UIMessageRole,
    UIPartState, UIToolApproval, UIToolPart, UIToolState,
};

/// `UIMessageStreamError`: a chunk does not fit the message being built.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct UIMessageStreamError {
    pub chunk_type: String,
    pub chunk_id: String,
    pub message: String,
}

impl UIMessageStreamError {
    fn new(chunk_type: &str, chunk_id: &str, message: String) -> Self {
        Self {
            chunk_type: chunk_type.into(),
            chunk_id: chunk_id.into(),
            message,
        }
    }
}

/// `StreamingUIMessageState['partialToolCalls'][id]`.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialToolCall {
    pub text: String,
    pub input_encoding: Option<InputEncoding>,
    pub index: usize,
    pub tool_name: String,
    pub dynamic: Option<bool>,
    pub title: Option<String>,
    pub tool_metadata: Option<JsonObject>,
}

/// `StreamingUIMessageState`.
///
/// [rust] The active text and reasoning parts are indexes into
/// `message.parts` (TS keeps references to the part objects).
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingUIMessageState {
    pub message: UIMessage,
    pub active_text_parts: IndexMap<String, usize>,
    pub active_reasoning_parts: IndexMap<String, usize>,
    pub partial_tool_calls: IndexMap<String, PartialToolCall>,
    pub finish_reason: Option<FinishReason>,
}

/// `createStreamingUIMessageState({ lastMessage, messageId })`.
pub fn create_streaming_ui_message_state(
    last_message: Option<UIMessage>,
    message_id: &str,
) -> StreamingUIMessageState {
    let message = match last_message {
        Some(message) if message.role == UIMessageRole::Assistant => message,
        _ => UIMessage {
            id: message_id.to_string(),
            role: UIMessageRole::Assistant,
            metadata: None,
            parts: Vec::new(),
        },
    };
    let mut partial_tool_calls = IndexMap::new();
    let last_step_start = message
        .parts
        .iter()
        .rposition(|part| matches!(part, UIMessagePart::StepStart));
    let mut static_tool_index = 0;
    let after = last_step_start.map_or(0, |i| i + 1);
    for part in &message.parts[after..] {
        let Some(tool) = part.as_tool() else {
            continue;
        };
        let index = static_tool_index;
        if !tool.dynamic {
            static_tool_index += 1;
        }
        if tool.state != UIToolState::InputStreaming {
            continue;
        }
        partial_tool_calls.insert(
            tool.tool_call_id.clone(),
            PartialToolCall {
                text: match &tool.raw_input {
                    Some(JsonValue::String(text)) => text.clone(),
                    _ => String::new(),
                },
                input_encoding: tool.input_encoding,
                index,
                tool_name: tool.tool_name.clone(),
                dynamic: Some(tool.dynamic),
                title: tool.title.clone(),
                tool_metadata: tool.tool_metadata.clone(),
            },
        );
    }
    StreamingUIMessageState {
        message,
        active_text_parts: IndexMap::new(),
        active_reasoning_parts: IndexMap::new(),
        partial_tool_calls,
        finish_reason: None,
    }
}

/// `mergeObjects(base, overrides)`.
pub fn merge_objects(base: &JsonValue, overrides: &JsonValue) -> JsonValue {
    let (JsonValue::Object(base), JsonValue::Object(overrides)) = (base, overrides) else {
        return overrides.clone();
    };
    let mut result = base.clone();
    for (key, value) in overrides {
        // Skip prototype-polluting keys when merging untrusted input
        if matches!(key.as_str(), "__proto__" | "constructor" | "prototype") {
            continue;
        }
        let merged = match (result.get(key), value) {
            (Some(base_value @ JsonValue::Object(_)), JsonValue::Object(_)) => {
                merge_objects(base_value, value)
            }
            _ => value.clone(),
        };
        result.insert(key.clone(), merged);
    }
    JsonValue::Object(result)
}

/// The fields an `updateToolPart` / `updateDynamicToolPart` call sets.
#[derive(Default)]
struct ToolUpdate {
    tool_name: String,
    tool_call_id: String,
    state: Option<UIToolState>,
    input: Option<JsonValue>,
    output: Option<JsonValue>,
    error_text: Option<String>,
    raw_input: Option<JsonValue>,
    input_encoding: Option<InputEncoding>,
    preliminary: Option<bool>,
    provider_executed: Option<bool>,
    provider_metadata: Option<ProviderMetadata>,
    title: Option<String>,
    tool_metadata: Option<JsonObject>,
}

impl StreamingUIMessageState {
    /// Index of the first part of the current step.
    fn current_step_start(&self) -> usize {
        self.message
            .parts
            .iter()
            .rposition(|part| matches!(part, UIMessagePart::StepStart))
            .map_or(0, |i| i + 1)
    }

    /// `getToolInvocation(toolCallId)`: the current step's part, else the
    /// latest one in the message.
    fn tool_invocation(&self, tool_call_id: &str) -> Result<usize, UIMessageStreamError> {
        let start = self.current_step_start();
        let parts = &self.message.parts;
        let has_id = |part: &UIMessagePart| {
            part.as_tool()
                .is_some_and(|t| t.tool_call_id == tool_call_id)
        };
        (start..parts.len())
            .find(|&i| has_id(&parts[i]))
            .or_else(|| (0..parts.len()).rev().find(|&i| has_id(&parts[i])))
            .ok_or_else(|| {
                UIMessageStreamError::new(
                    "tool-invocation",
                    tool_call_id,
                    format!("No tool invocation found for tool call ID \"{tool_call_id}\"."),
                )
            })
    }

    /// `getToolInvocationByApprovalId(approvalId)`.
    fn tool_invocation_by_approval_id(
        &self,
        approval_id: &str,
    ) -> Result<usize, UIMessageStreamError> {
        self.message
            .parts
            .iter()
            .position(|part| {
                part.as_tool()
                    .and_then(|t| t.approval.as_ref())
                    .is_some_and(|a| a.id == approval_id)
            })
            .ok_or_else(|| {
                UIMessageStreamError::new(
                    "tool-approval-response",
                    approval_id,
                    format!("No tool invocation found for approval ID \"{approval_id}\"."),
                )
            })
    }

    fn tool_mut(&mut self, index: usize) -> &mut UIToolPart {
        self.message.parts[index]
            .as_tool_mut()
            .expect("index of a tool part")
    }

    /// `updateToolPart` (`dynamic: false`) and `updateDynamicToolPart`
    /// (`dynamic: true`).
    fn update_tool_part(&mut self, dynamic: bool, update: ToolUpdate, existing: Option<usize>) {
        let state = update.state.expect("tool update state");
        let is_result = matches!(
            state,
            UIToolState::OutputAvailable | UIToolState::OutputError
        );
        let start = self.current_step_start();
        let index = existing.or_else(|| {
            (start..self.message.parts.len()).find(|&i| {
                self.message.parts[i]
                    .as_tool()
                    .is_some_and(|t| t.dynamic == dynamic && t.tool_call_id == update.tool_call_id)
            })
        });
        match index {
            Some(index) => {
                let part = self.tool_mut(index);
                part.state = state;
                part.tool_name = update.tool_name;
                part.input = update.input;
                part.output = update.output;
                part.error_text = update.error_text;
                part.raw_input = update.raw_input;
                part.input_encoding = update.input_encoding;
                part.preliminary = update.preliminary;
                if update.title.is_some() {
                    part.title = update.title;
                }
                if update.tool_metadata.is_some() {
                    part.tool_metadata = update.tool_metadata;
                }
                // once providerExecuted is set, it stays for streaming
                part.provider_executed = update.provider_executed.or(part.provider_executed);
                if let Some(metadata) = update.provider_metadata {
                    if is_result {
                        part.result_provider_metadata = Some(metadata);
                    } else {
                        part.call_provider_metadata = Some(metadata);
                    }
                }
            }
            None => {
                let mut part =
                    UIToolPart::new(dynamic, update.tool_name, update.tool_call_id, state);
                part.title = update.title;
                part.tool_metadata = update.tool_metadata;
                part.input = update.input;
                part.input_encoding = update.input_encoding;
                part.output = update.output;
                part.raw_input = if dynamic { None } else { update.raw_input };
                part.error_text = update.error_text;
                part.provider_executed = update.provider_executed;
                part.preliminary = update.preliminary;
                if is_result {
                    part.result_provider_metadata = update.provider_metadata;
                } else {
                    part.call_provider_metadata = update.provider_metadata;
                }
                self.message.parts.push(UIMessagePart::Tool(part));
            }
        }
    }

    /// `updateMessageMetadata(metadata)`.
    fn update_message_metadata(&mut self, metadata: Option<&JsonValue>) {
        let Some(metadata) = metadata.filter(|m| !m.is_null()) else {
            return;
        };
        let merged = match &self.message.metadata {
            Some(existing) if !existing.is_null() => merge_objects(existing, metadata),
            _ => metadata.clone(),
        };
        self.message.metadata = Some(merged);
    }

    /// One `processUIMessageStream` transform step: apply `chunk` to the
    /// message. `on_error` receives `error` chunks.
    pub fn apply_chunk(
        &mut self,
        chunk: &UIMessageChunk,
        on_error: &dyn Fn(&ErrorValue),
    ) -> Result<(), UIMessageStreamError> {
        match chunk {
            UIMessageChunk::TextStart {
                id,
                provider_metadata,
            } => {
                self.message.parts.push(UIMessagePart::Text(TextUIPart {
                    text: String::new(),
                    provider_metadata: provider_metadata.clone(),
                    state: Some(UIPartState::Streaming),
                    citations: None,
                }));
                self.active_text_parts
                    .insert(id.clone(), self.message.parts.len() - 1);
            }
            UIMessageChunk::TextDelta {
                id,
                delta,
                provider_metadata,
            } => {
                let part = self.active_text_part(id, "text-delta")?;
                part.text.push_str(delta);
                if provider_metadata.is_some() {
                    part.provider_metadata = provider_metadata.clone();
                }
            }
            UIMessageChunk::TextEnd {
                id,
                citations,
                provider_metadata,
            } => {
                let part = self.active_text_part(id, "text-end")?;
                part.state = Some(UIPartState::Done);
                if citations.is_some() {
                    part.citations = citations.clone();
                }
                if provider_metadata.is_some() {
                    part.provider_metadata = provider_metadata.clone();
                }
                self.active_text_parts.shift_remove(id);
            }
            UIMessageChunk::ReasoningStart {
                id,
                provider_metadata,
            } => {
                self.message
                    .parts
                    .push(UIMessagePart::Reasoning(ReasoningUIPart {
                        id: Some(id.clone()),
                        text: String::new(),
                        provider_metadata: provider_metadata.clone(),
                        state: Some(UIPartState::Streaming),
                    }));
                self.active_reasoning_parts
                    .insert(id.clone(), self.message.parts.len() - 1);
            }
            UIMessageChunk::ReasoningDelta {
                id,
                delta,
                provider_metadata,
            } => {
                let part = self.active_reasoning_part(id, "reasoning-delta")?;
                part.text.push_str(delta);
                if provider_metadata.is_some() {
                    part.provider_metadata = provider_metadata.clone();
                }
            }
            UIMessageChunk::ReasoningEnd {
                id,
                provider_metadata,
            } => {
                let part = self.active_reasoning_part(id, "reasoning-end")?;
                if provider_metadata.is_some() {
                    part.provider_metadata = provider_metadata.clone();
                }
                part.state = Some(UIPartState::Done);
                self.active_reasoning_parts.shift_remove(id);
            }
            UIMessageChunk::ToolInputStart {
                tool_call_id,
                tool_name,
                input_encoding,
                provider_executed,
                provider_metadata,
                tool_metadata,
                dynamic,
                title,
            } => {
                let start = self.current_step_start();
                let index = self.message.parts[start..]
                    .iter()
                    .filter(|part| part.is_static_tool())
                    .count();
                self.partial_tool_calls.insert(
                    tool_call_id.clone(),
                    PartialToolCall {
                        text: String::new(),
                        input_encoding: *input_encoding,
                        tool_name: tool_name.clone(),
                        index,
                        dynamic: *dynamic,
                        title: title.clone(),
                        tool_metadata: tool_metadata.clone(),
                    },
                );
                self.update_tool_part(
                    *dynamic == Some(true),
                    ToolUpdate {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        state: Some(UIToolState::InputStreaming),
                        input_encoding: *input_encoding,
                        provider_executed: *provider_executed,
                        title: title.clone(),
                        tool_metadata: tool_metadata.clone(),
                        provider_metadata: provider_metadata.clone(),
                        ..Default::default()
                    },
                    None,
                );
            }
            UIMessageChunk::ToolInputDelta {
                tool_call_id,
                input_text_delta,
            } => {
                let Some(partial) = self.partial_tool_calls.get_mut(tool_call_id) else {
                    return Err(UIMessageStreamError::new(
                        "tool-input-delta",
                        tool_call_id,
                        format!(
                            "Received tool-input-delta for missing tool call with ID \"{tool_call_id}\". Ensure a \"tool-input-start\" chunk is sent before any \"tool-input-delta\" chunks."
                        ),
                    ));
                };
                partial.text.push_str(input_text_delta);
                let partial = partial.clone();
                let partial_args = if partial.input_encoding == Some(InputEncoding::Text) {
                    Some(JsonValue::String(partial.text.clone()))
                } else {
                    parse_partial_json(Some(&partial.text)).0
                };
                self.update_tool_part(
                    partial.dynamic == Some(true),
                    ToolUpdate {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: partial.tool_name,
                        state: Some(UIToolState::InputStreaming),
                        input_encoding: partial.input_encoding,
                        input: partial_args,
                        raw_input: Some(JsonValue::String(partial.text)),
                        title: partial.title,
                        tool_metadata: partial.tool_metadata,
                        ..Default::default()
                    },
                    None,
                );
            }
            UIMessageChunk::ToolInputAvailable {
                tool_call_id,
                tool_name,
                input,
                provider_executed,
                provider_metadata,
                tool_metadata,
                dynamic,
                title,
            } => {
                self.update_tool_part(
                    *dynamic == Some(true),
                    ToolUpdate {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        state: Some(UIToolState::InputAvailable),
                        input: Some(input.clone()),
                        provider_executed: *provider_executed,
                        provider_metadata: provider_metadata.clone(),
                        title: title.clone(),
                        tool_metadata: tool_metadata.clone(),
                        ..Default::default()
                    },
                    None,
                );
            }
            UIMessageChunk::ToolInputError {
                tool_call_id,
                tool_name,
                input,
                provider_executed,
                provider_metadata,
                tool_metadata,
                dynamic,
                error_text,
                ..
            } => {
                // An existing part for this toolCallId keeps its type, so it
                // is updated in place.
                let start = self.current_step_start();
                let existing = self.message.parts[start..]
                    .iter()
                    .filter_map(UIMessagePart::as_tool)
                    .find(|t| t.tool_call_id == *tool_call_id);
                let is_dynamic = match existing {
                    Some(part) => part.dynamic,
                    None => *dynamic == Some(true),
                };
                self.update_tool_part(
                    is_dynamic,
                    ToolUpdate {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        state: Some(UIToolState::OutputError),
                        input: Some(input.clone()),
                        error_text: Some(error_text.clone()),
                        provider_executed: *provider_executed,
                        provider_metadata: provider_metadata.clone(),
                        tool_metadata: tool_metadata.clone(),
                        ..Default::default()
                    },
                    None,
                );
            }
            UIMessageChunk::ToolApprovalRequest {
                approval_id,
                tool_call_id,
                approval_descriptor,
                input_schema_input,
                reason,
                is_automatic,
                signature,
            } => {
                let index = self.tool_invocation(tool_call_id)?;
                let part = self.tool_mut(index);
                part.state = UIToolState::ApprovalRequested;
                part.approval = Some(UIToolApproval {
                    id: approval_id.clone(),
                    descriptor: approval_descriptor.clone(),
                    input_schema_input: input_schema_input.clone(),
                    request_reason: reason.clone(),
                    is_automatic: (*is_automatic == Some(true)).then_some(true),
                    signature: signature.clone(),
                    ..Default::default()
                });
            }
            UIMessageChunk::ToolApprovalResponse {
                approval_id,
                approved,
                reason,
                provider_executed,
                provider_metadata,
            } => {
                let index = self.tool_invocation_by_approval_id(approval_id)?;
                let part = self.tool_mut(index);
                let mut approval = part.approval.clone().unwrap_or_else(|| UIToolApproval {
                    id: approval_id.clone(),
                    ..Default::default()
                });
                part.state = UIToolState::ApprovalResponded;
                approval.id = approval_id.clone();
                approval.approved = Some(*approved);
                if reason.is_some() {
                    approval.reason = reason.clone();
                }
                part.approval = Some(approval);
                if provider_executed.is_some() {
                    part.provider_executed = *provider_executed;
                }
                if provider_metadata.is_some() {
                    part.call_provider_metadata = provider_metadata.clone();
                }
            }
            UIMessageChunk::ToolOutputDenied { tool_call_id } => {
                let index = self.tool_invocation(tool_call_id)?;
                self.tool_mut(index).state = UIToolState::OutputDenied;
            }
            UIMessageChunk::ToolOutputAvailable {
                tool_call_id,
                output,
                provider_executed,
                provider_metadata,
                tool_metadata,
                preliminary,
                ..
            } => {
                let index = self.tool_invocation(tool_call_id)?;
                let existing = self.tool_mut(index).clone();
                self.update_tool_part(
                    existing.dynamic,
                    ToolUpdate {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: existing.tool_name,
                        state: Some(UIToolState::OutputAvailable),
                        input: existing.input,
                        output: Some(output.clone()),
                        preliminary: *preliminary,
                        provider_executed: *provider_executed,
                        provider_metadata: provider_metadata.clone(),
                        title: existing.title,
                        tool_metadata: tool_metadata.clone().or(existing.tool_metadata),
                        ..Default::default()
                    },
                    Some(index),
                );
            }
            UIMessageChunk::ToolOutputError {
                tool_call_id,
                error_text,
                provider_executed,
                provider_metadata,
                tool_metadata,
                ..
            } => {
                let index = self.tool_invocation(tool_call_id)?;
                let existing = self.tool_mut(index).clone();
                self.update_tool_part(
                    existing.dynamic,
                    ToolUpdate {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: existing.tool_name,
                        state: Some(UIToolState::OutputError),
                        input: existing.input,
                        // the static updater keeps the raw input
                        raw_input: if existing.dynamic {
                            None
                        } else {
                            existing.raw_input
                        },
                        error_text: Some(error_text.clone()),
                        provider_executed: *provider_executed,
                        provider_metadata: provider_metadata.clone(),
                        title: existing.title,
                        tool_metadata: tool_metadata.clone().or(existing.tool_metadata),
                        ..Default::default()
                    },
                    Some(index),
                );
            }
            UIMessageChunk::StartStep => {
                // add a step boundary part to the message
                self.message.parts.push(UIMessagePart::StepStart);
            }
            UIMessageChunk::FinishStep => {
                // Active parts are closed by their explicit end chunks.
            }
            UIMessageChunk::Start {
                message_id,
                message_metadata,
            } => {
                if let Some(id) = message_id {
                    self.message.id = id.clone();
                }
                self.update_message_metadata(message_metadata.as_ref());
            }
            UIMessageChunk::Finish {
                finish_reason,
                message_metadata,
            } => {
                if finish_reason.is_some() {
                    self.finish_reason = *finish_reason;
                }
                self.update_message_metadata(message_metadata.as_ref());
            }
            UIMessageChunk::MessageMetadata { message_metadata } => {
                self.update_message_metadata(Some(message_metadata));
            }
            UIMessageChunk::Error { error_text } => {
                on_error(&ErrorValue::message(error_text.clone()));
            }
            UIMessageChunk::Abort { .. } => {}
        }
        Ok(())
    }

    fn active_text_part(
        &mut self,
        id: &str,
        chunk_type: &str,
    ) -> Result<&mut TextUIPart, UIMessageStreamError> {
        let index = self.active_text_parts.get(id).copied().ok_or_else(|| {
            UIMessageStreamError::new(
                chunk_type,
                id,
                format!(
                    "Received {chunk_type} for missing text part with ID \"{id}\". Ensure a \"text-start\" chunk is sent before any \"{chunk_type}\" chunks."
                ),
            )
        })?;
        match &mut self.message.parts[index] {
            UIMessagePart::Text(part) => Ok(part),
            _ => unreachable!("active text part index"),
        }
    }

    fn active_reasoning_part(
        &mut self,
        id: &str,
        chunk_type: &str,
    ) -> Result<&mut ReasoningUIPart, UIMessageStreamError> {
        let index = self.active_reasoning_parts.get(id).copied().ok_or_else(|| {
            UIMessageStreamError::new(
                chunk_type,
                id,
                format!(
                    "Received {chunk_type} for missing reasoning part with ID \"{id}\". Ensure a \"reasoning-start\" chunk is sent before any \"{chunk_type}\" chunks."
                ),
            )
        })?;
        match &mut self.message.parts[index] {
            UIMessagePart::Reasoning(part) => Ok(part),
            _ => unreachable!("active reasoning part index"),
        }
    }
}

#[cfg(test)]
#[path = "process_ui_message_stream_tests.rs"]
mod tests;
