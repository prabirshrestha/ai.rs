//! Port of `agent/internal/translate-stream-part.ts`.

use serde_json::json;

use crate::harness::provider::JsonValue;
use crate::harness::sdk::content::{TypedToolCall, TypedToolError, TypedToolResult};
use crate::harness::sdk::error_value::{ErrorValue, generate_id};
use crate::harness::sdk::text_stream_part::TextStreamPart;
use crate::harness::spec::stream_part::HarnessV1StreamPart;

/// What `getToolCall` returns: the originating call's parsed input and
/// execution origin.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslatedToolCallInfo {
    pub input: JsonValue,
    pub provider_executed: Option<bool>,
}

/// `getToolCall`.
pub type GetToolCallFn<'a> = dyn Fn(&str) -> Option<TranslatedToolCallInfo> + 'a;

/// `translateStreamPart` options.
#[derive(Default)]
pub struct TranslateOptions<'a> {
    /// Returns the parsed tool call that produced a result.
    pub get_tool_call: Option<&'a GetToolCallFn<'a>>,
    /// Whether the tool call that produced this event ran inside the
    /// harness runtime. Omit to treat every failure as provider-executed.
    pub is_provider_executed: Option<&'a (dyn Fn(&str) -> bool + 'a)>,
}

/// Translate one harness event to the AI SDK `TextStreamPart` shape.
///
/// `tool-call`, `tool-approval-request`, `stream-start`, `finish-step` and
/// `finish` give no parts; `file-change` and `compaction` fan out into a
/// synthetic dynamic, provider-executed tool-call / tool-result pair.
pub fn translate_stream_part(
    event: &HarnessV1StreamPart,
    options: &TranslateOptions<'_>,
) -> Vec<TextStreamPart> {
    use HarnessV1StreamPart as H;
    match event {
        H::StreamStart { .. }
        | H::ToolCall { .. }
        | H::ToolApprovalRequest { .. }
        | H::FinishStep { .. }
        | H::Finish { .. } => vec![],
        H::TextStart {
            id,
            harness_metadata,
        } => vec![TextStreamPart::TextStart {
            id: id.clone(),
            provider_metadata: harness_metadata.clone(),
        }],
        H::TextDelta {
            id,
            delta,
            harness_metadata,
        } => vec![TextStreamPart::TextDelta {
            id: id.clone(),
            text: delta.clone(),
            provider_metadata: harness_metadata.clone(),
        }],
        H::TextEnd {
            id,
            harness_metadata,
        } => vec![TextStreamPart::TextEnd {
            id: id.clone(),
            citations: None,
            provider_metadata: harness_metadata.clone(),
        }],
        H::ReasoningStart {
            id,
            harness_metadata,
        } => vec![TextStreamPart::ReasoningStart {
            id: id.clone(),
            provider_metadata: harness_metadata.clone(),
        }],
        H::ReasoningDelta {
            id,
            delta,
            harness_metadata,
        } => vec![TextStreamPart::ReasoningDelta {
            id: id.clone(),
            text: delta.clone(),
            provider_metadata: harness_metadata.clone(),
        }],
        H::ReasoningEnd {
            id,
            harness_metadata,
        } => vec![TextStreamPart::ReasoningEnd {
            id: id.clone(),
            provider_metadata: harness_metadata.clone(),
        }],
        H::ToolInputStart {
            id,
            tool_name,
            provider_metadata,
            provider_executed,
            dynamic,
            title,
        } => vec![TextStreamPart::ToolInputStart {
            id: id.clone(),
            tool_name: tool_name.clone(),
            provider_metadata: provider_metadata.clone(),
            provider_executed: *provider_executed,
            dynamic: *dynamic,
            title: title.clone(),
        }],
        H::ToolInputDelta {
            id,
            delta,
            provider_metadata,
        } => vec![TextStreamPart::ToolInputDelta {
            id: id.clone(),
            delta: delta.clone(),
            provider_metadata: provider_metadata.clone(),
        }],
        H::ToolInputEnd {
            id,
            provider_metadata,
        } => vec![TextStreamPart::ToolInputEnd {
            id: id.clone(),
            provider_metadata: provider_metadata.clone(),
        }],
        H::ToolResult {
            tool_call_id,
            tool_name,
            result,
            is_error,
            preliminary,
            dynamic,
            provider_metadata,
        } => {
            let tool_call = options.get_tool_call.and_then(|f| f(tool_call_id));
            let provider_executed = tool_call
                .as_ref()
                .and_then(|c| c.provider_executed)
                .or_else(|| options.is_provider_executed.map(|f| f(tool_call_id)))
                .unwrap_or(true);
            let input = tool_call.as_ref().map(|c| c.input.clone());
            if *is_error == Some(true) && provider_executed {
                let mut error = TypedToolError::new(
                    tool_call_id,
                    tool_name,
                    input,
                    ErrorValue::Value(result.clone()),
                );
                error.provider_executed = Some(true);
                error.dynamic = *dynamic;
                error.provider_metadata = provider_metadata.clone();
                return vec![TextStreamPart::ToolError(error)];
            }
            let mut out = TypedToolResult::new(tool_call_id, tool_name, input, result.clone());
            out.provider_executed = tool_call.and_then(|c| c.provider_executed);
            out.preliminary = *preliminary;
            out.dynamic = *dynamic;
            out.provider_metadata = provider_metadata.clone();
            vec![TextStreamPart::ToolResult(out)]
        }
        H::FileChange {
            event,
            path,
            harness_metadata,
        } => {
            let payload = json!({ "event": event, "path": path });
            synthetic_pair(
                format!("harness-file-change-{}", generate_id()),
                "fileChange",
                payload.clone(),
                payload,
                harness_metadata.clone(),
            )
        }
        H::Compaction {
            trigger,
            summary,
            tokens_before,
            tokens_after,
            harness_metadata,
        } => {
            let mut output = json!({ "trigger": trigger, "summary": summary });
            if let Some(n) = tokens_before {
                output["tokensBefore"] = json!(n);
            }
            if let Some(n) = tokens_after {
                output["tokensAfter"] = json!(n);
            }
            synthetic_pair(
                format!("harness-compaction-{}", generate_id()),
                "compaction",
                json!({}),
                output,
                harness_metadata.clone(),
            )
        }
        H::Error { error } => vec![TextStreamPart::Error {
            error: ErrorValue::Value(error.clone()),
        }],
        H::Raw { raw_value } => vec![TextStreamPart::Raw {
            raw_value: raw_value.clone(),
        }],
    }
}

fn synthetic_pair(
    tool_call_id: String,
    tool_name: &str,
    input: JsonValue,
    output: JsonValue,
    metadata: Option<crate::harness::sdk::content::ProviderMetadata>,
) -> Vec<TextStreamPart> {
    let mut call = TypedToolCall::new(&tool_call_id, tool_name, input.clone());
    call.dynamic = Some(true);
    call.provider_executed = Some(true);
    call.provider_metadata = metadata.clone();
    let mut result = TypedToolResult::new(&tool_call_id, tool_name, Some(input), output);
    result.dynamic = Some(true);
    result.provider_executed = Some(true);
    result.provider_metadata = metadata;
    vec![
        TextStreamPart::ToolCall(call),
        TextStreamPart::ToolResult(result),
    ]
}

#[cfg(test)]
#[path = "translate_stream_part_tests.rs"]
mod tests;
