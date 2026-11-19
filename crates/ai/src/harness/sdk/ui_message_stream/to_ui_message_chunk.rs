//! Mirror of `ai` `ui-message-stream/to-ui-message-chunk.ts` and
//! `get-model-tool-error-text.ts`.

use std::sync::Arc;

use crate::harness::provider::JsonValue;
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::text_stream_part::TextStreamPart;
use crate::harness::sdk::tool::{ToolSet, ToolType};
use crate::harness::sdk::ui_message_stream::ui_message_chunks::UIMessageChunk;

/// `onError: (error: unknown) => string`.
pub type UIMessageStreamOnErrorFn = Arc<dyn Fn(&ErrorValue) -> String + Send + Sync>;

/// The default `onError`: stream errors never reach the model, so they are
/// masked.
pub const DEFAULT_ERROR_TEXT: &str = "An error occurred.";

/// `ToUIMessageChunkOptions`.
#[derive(Clone, Default)]
pub struct ToUIMessageChunkOptions<'a> {
    pub tools: Option<&'a ToolSet>,
    /// Defaults to `true`.
    pub send_reasoning: Option<bool>,
    /// Defaults to `false`.
    ///
    /// [rust] The harness never emits `source` parts, so it has no effect.
    pub send_sources: Option<bool>,
    /// Defaults to `true`.
    pub send_start: Option<bool>,
    /// Defaults to `true`.
    pub send_finish: Option<bool>,
    /// Formats stream errors. Also formats invalid tool calls and
    /// application-executed tool errors when `redact_tool_errors` is
    /// enabled. Provider-executed tool execution errors bypass it.
    pub on_error: Option<UIMessageStreamOnErrorFn>,
    /// Replace the error text of failed tool calls with the result of
    /// `on_error`. Defaults to `true` when `on_error` is set.
    pub redact_tool_errors: Option<bool>,
    pub message_metadata: Option<JsonValue>,
    pub response_message_id: Option<String>,
}

/// `getModelToolErrorText(error)`: the error text the model receives for a
/// failed tool call.
///
/// [rust] `getErrorMessage` cannot fail here, so this is always `Some`.
pub fn get_model_tool_error_text(error: &ErrorValue) -> Option<String> {
    Some(error.get_error_message())
}

/// `toUIMessageChunk(part, options)`: one stream part as a UI message
/// chunk, or `None` for parts that produce none.
pub fn to_ui_message_chunk(
    part: &TextStreamPart,
    options: &ToUIMessageChunkOptions<'_>,
) -> Option<UIMessageChunk> {
    let send_reasoning = options.send_reasoning.unwrap_or(true);
    let send_start = options.send_start.unwrap_or(true);
    let send_finish = options.send_finish.unwrap_or(true);
    // an existing `onError` keeps handling tool errors, as it did before
    // tool errors stopped being redacted by default
    let redact_tool_errors = options
        .redact_tool_errors
        .unwrap_or(options.on_error.is_some());
    let on_error = |error: &ErrorValue| match &options.on_error {
        Some(f) => f(error),
        None => DEFAULT_ERROR_TEXT.to_string(),
    };
    // The error text of a failed tool call is not altered by default (the
    // model has already received it).
    let get_tool_error_text = |error: &ErrorValue| {
        if redact_tool_errors {
            on_error(error)
        } else {
            get_model_tool_error_text(error).unwrap_or_else(|| on_error(error))
        }
    };
    let is_dynamic = |tool_name: &str, dynamic: Option<bool>| -> Option<bool> {
        match options.tools.and_then(|tools| tools.get(tool_name)) {
            // provider-executed, dynamic tools are not listed in the tools
            None => dynamic,
            Some(tool) => matches!(tool.r#type, ToolType::Dynamic).then_some(true),
        }
    };

    Some(match part {
        TextStreamPart::TextStart {
            id,
            provider_metadata,
        } => UIMessageChunk::TextStart {
            id: id.clone(),
            provider_metadata: provider_metadata.clone(),
        },
        TextStreamPart::TextDelta {
            id,
            text,
            provider_metadata,
        } => UIMessageChunk::TextDelta {
            id: id.clone(),
            delta: text.clone(),
            provider_metadata: provider_metadata.clone(),
        },
        TextStreamPart::TextEnd {
            id,
            citations,
            provider_metadata,
        } => UIMessageChunk::TextEnd {
            id: id.clone(),
            citations: citations.clone(),
            provider_metadata: provider_metadata.clone(),
        },
        TextStreamPart::ReasoningStart {
            id,
            provider_metadata,
        } => {
            if !send_reasoning {
                return None;
            }
            UIMessageChunk::ReasoningStart {
                id: id.clone(),
                provider_metadata: provider_metadata.clone(),
            }
        }
        TextStreamPart::ReasoningEnd {
            id,
            provider_metadata,
        } => {
            if !send_reasoning {
                return None;
            }
            UIMessageChunk::ReasoningEnd {
                id: id.clone(),
                provider_metadata: provider_metadata.clone(),
            }
        }
        TextStreamPart::ReasoningDelta {
            id,
            text,
            provider_metadata,
        } => {
            if !send_reasoning {
                return None;
            }
            UIMessageChunk::ReasoningDelta {
                id: id.clone(),
                delta: text.clone(),
                provider_metadata: provider_metadata.clone(),
            }
        }
        TextStreamPart::ToolInputStart {
            id,
            tool_name,
            provider_metadata,
            provider_executed,
            dynamic,
            title,
        } => UIMessageChunk::ToolInputStart {
            tool_call_id: id.clone(),
            tool_name: tool_name.clone(),
            // [rust] The harness's tool-input-start parts carry no
            // `inputEncoding` or `toolMetadata`.
            input_encoding: None,
            provider_executed: *provider_executed,
            provider_metadata: provider_metadata.clone(),
            tool_metadata: None,
            dynamic: is_dynamic(tool_name, *dynamic),
            title: title.clone(),
        },
        TextStreamPart::ToolInputDelta { id, delta, .. } => UIMessageChunk::ToolInputDelta {
            tool_call_id: id.clone(),
            input_text_delta: delta.clone(),
        },
        TextStreamPart::ToolCall(call) => {
            let dynamic = is_dynamic(&call.tool_name, call.dynamic);
            if call.is_invalid() {
                let error = call
                    .error
                    .clone()
                    .unwrap_or(ErrorValue::Value(JsonValue::Null));
                return Some(UIMessageChunk::ToolInputError {
                    tool_call_id: call.tool_call_id.clone(),
                    tool_name: call.tool_name.clone(),
                    input: call.input.clone(),
                    provider_executed: call.provider_executed,
                    provider_metadata: call.provider_metadata.clone(),
                    tool_metadata: call.tool_metadata.clone(),
                    dynamic,
                    // streamText sends the model a tool result for an invalid
                    // tool call only when the tool is not provider-executed
                    error_text: if call.provider_executed == Some(true) {
                        on_error(&error)
                    } else {
                        get_tool_error_text(&error)
                    },
                    title: call.title.clone(),
                });
            }
            UIMessageChunk::ToolInputAvailable {
                tool_call_id: call.tool_call_id.clone(),
                tool_name: call.tool_name.clone(),
                input: call.input.clone(),
                provider_executed: call.provider_executed,
                provider_metadata: call.provider_metadata.clone(),
                tool_metadata: call.tool_metadata.clone(),
                dynamic,
                title: call.title.clone(),
            }
        }
        TextStreamPart::ToolApprovalRequest(request) => UIMessageChunk::ToolApprovalRequest {
            approval_id: request.approval_id.clone(),
            tool_call_id: request.tool_call.tool_call_id.clone(),
            approval_descriptor: None,
            input_schema_input: request
                .tool_call
                .input_schema_input()
                .filter(|value| **value != request.tool_call.input)
                .cloned(),
            reason: request.reason.clone(),
            is_automatic: request.is_automatic,
            signature: request.signature.clone(),
        },
        TextStreamPart::ToolApprovalResponse(response) => UIMessageChunk::ToolApprovalResponse {
            approval_id: response.approval_id.clone(),
            approved: response.approved,
            reason: response.reason.clone(),
            provider_executed: response.provider_executed,
            provider_metadata: None,
        },
        TextStreamPart::ToolResult(result) => UIMessageChunk::ToolOutputAvailable {
            tool_call_id: result.tool_call_id.clone(),
            // UI stream chunks are serialized as JSON, which drops undefined
            // properties; the output is always present (`null` at worst).
            output: result.output.clone(),
            provider_executed: result.provider_executed,
            provider_metadata: result.provider_metadata.clone(),
            tool_metadata: result.tool_metadata.clone(),
            dynamic: is_dynamic(&result.tool_name, result.dynamic),
            preliminary: result.preliminary,
        },
        TextStreamPart::ToolError(error) => UIMessageChunk::ToolOutputError {
            tool_call_id: error.tool_call_id.clone(),
            // Preserve provider error codes for model-message round trips.
            // Provider-executed execution errors always bypass onError.
            error_text: if error.provider_executed == Some(true) {
                match &error.error {
                    ErrorValue::Value(JsonValue::String(text)) => text.clone(),
                    other => other.to_js_json().to_string(),
                }
            } else {
                get_tool_error_text(&error.error)
            },
            provider_executed: error.provider_executed,
            provider_metadata: error.provider_metadata.clone(),
            tool_metadata: error.tool_metadata.clone(),
            dynamic: is_dynamic(&error.tool_name, error.dynamic),
        },
        TextStreamPart::Error { error } => UIMessageChunk::Error {
            error_text: on_error(error),
        },
        TextStreamPart::StartStep { .. } => UIMessageChunk::StartStep,
        TextStreamPart::FinishStep { .. } => UIMessageChunk::FinishStep,
        TextStreamPart::Start => {
            if !send_start {
                return None;
            }
            UIMessageChunk::Start {
                message_id: options.response_message_id.clone(),
                message_metadata: options.message_metadata.clone(),
            }
        }
        TextStreamPart::Finish { finish_reason, .. } => {
            if !send_finish {
                return None;
            }
            UIMessageChunk::Finish {
                finish_reason: Some(*finish_reason),
                message_metadata: options.message_metadata.clone(),
            }
        }
        TextStreamPart::Abort { reason } => UIMessageChunk::Abort {
            reason: reason.clone(),
        },
        TextStreamPart::ToolInputEnd { .. } | TextStreamPart::Raw { .. } => return None,
    })
}

#[cfg(test)]
#[path = "to_ui_message_chunk_tests.rs"]
mod tests;
