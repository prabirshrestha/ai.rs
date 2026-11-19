//! Mirror of `ai` `TextStreamPart<TOOLS>`: what `HarnessStreamTextResult`
//! streams.
//!
//! [rust] Carries the variants a harness turn produces. `source`, `file`,
//! `reasoning-file`, `custom`, `tool-output-denied`, `steering-message` and
//! `ping` are never emitted by the harness and are left out.

use serde::Serialize;

use crate::harness::provider::JsonValue;
use crate::harness::sdk::content::{
    CallWarning, FinishReason, LanguageModelRequestMetadata, LanguageModelResponseMetadata,
    ProviderMetadata, StepResultPerformance, ToolApprovalRequestOutput, ToolApprovalResponseOutput,
    TypedToolCall, TypedToolError, TypedToolResult,
};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::usage::LanguageModelUsage;

/// `TextStreamFinishStepPart['response']`: response metadata without
/// `messages` and `body`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FinishStepResponse {
    pub id: String,
    #[serde(with = "crate::harness::sdk::content::iso_timestamp")]
    pub timestamp: std::time::SystemTime,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<indexmap::IndexMap<String, String>>,
}

impl From<&LanguageModelResponseMetadata> for FinishStepResponse {
    fn from(r: &LanguageModelResponseMetadata) -> Self {
        Self {
            id: r.id.clone(),
            timestamp: r.timestamp,
            model_id: r.model_id.clone(),
            headers: r.headers.clone(),
        }
    }
}

/// `TextStreamPart<TOOLS>`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum TextStreamPart {
    TextStart {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    TextEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<JsonValue>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    TextDelta {
        id: String,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ReasoningStart {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ReasoningEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ReasoningDelta {
        id: String,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolInputStart {
        id: String,
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    ToolInputEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolInputDelta {
        id: String,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolCall(TypedToolCall),
    ToolResult(TypedToolResult),
    ToolError(TypedToolError),
    ToolApprovalRequest(ToolApprovalRequestOutput),
    ToolApprovalResponse(ToolApprovalResponseOutput),
    StartStep {
        request: LanguageModelRequestMetadata,
        warnings: Vec<CallWarning>,
    },
    FinishStep {
        response: FinishStepResponse,
        usage: LanguageModelUsage,
        performance: StepResultPerformance,
        finish_reason: FinishReason,
        raw_finish_reason: Option<String>,
        provider_metadata: Option<ProviderMetadata>,
    },
    Start,
    Finish {
        finish_reason: FinishReason,
        raw_finish_reason: Option<String>,
        total_usage: LanguageModelUsage,
        /// [rust] TS spreads `providerMetadata` onto the finish part too.
        provider_metadata: Option<ProviderMetadata>,
    },
    Abort {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Error {
        error: ErrorValue,
    },
    Raw {
        raw_value: JsonValue,
    },
}

impl TextStreamPart {
    pub fn tool_call(call: TypedToolCall) -> Self {
        Self::ToolCall(call)
    }

    pub fn tool_result(result: TypedToolResult) -> Self {
        Self::ToolResult(result)
    }

    pub fn tool_error(error: TypedToolError) -> Self {
        Self::ToolError(error)
    }

    pub fn approval_request(request: ToolApprovalRequestOutput) -> Self {
        Self::ToolApprovalRequest(request)
    }

    pub fn approval_response(response: ToolApprovalResponseOutput) -> Self {
        Self::ToolApprovalResponse(response)
    }

    /// `part.type`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::TextStart { .. } => "text-start",
            Self::TextEnd { .. } => "text-end",
            Self::TextDelta { .. } => "text-delta",
            Self::ReasoningStart { .. } => "reasoning-start",
            Self::ReasoningEnd { .. } => "reasoning-end",
            Self::ReasoningDelta { .. } => "reasoning-delta",
            Self::ToolInputStart { .. } => "tool-input-start",
            Self::ToolInputEnd { .. } => "tool-input-end",
            Self::ToolInputDelta { .. } => "tool-input-delta",
            Self::ToolCall(_) => "tool-call",
            Self::ToolResult(_) => "tool-result",
            Self::ToolError(_) => "tool-error",
            Self::ToolApprovalRequest(_) => "tool-approval-request",
            Self::ToolApprovalResponse(_) => "tool-approval-response",
            Self::StartStep { .. } => "start-step",
            Self::FinishStep { .. } => "finish-step",
            Self::Start => "start",
            Self::Finish { .. } => "finish",
            Self::Abort { .. } => "abort",
            Self::Error { .. } => "error",
            Self::Raw { .. } => "raw",
        }
    }

    /// The part as JSON (what TS consumers see).
    pub fn to_json(&self) -> JsonValue {
        serde_json::to_value(self).unwrap_or(JsonValue::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serializes_tool_parts_with_their_type() {
        let part = TextStreamPart::tool_call(TypedToolCall::new("c", "t", json!({})));
        assert_eq!(
            part.to_json(),
            json!({"type": "tool-call", "toolCallId": "c", "toolName": "t", "input": {}})
        );
        let part = TextStreamPart::TextDelta {
            id: "1".into(),
            text: "x".into(),
            provider_metadata: None,
        };
        assert_eq!(
            part.to_json(),
            json!({"type": "text-delta", "id": "1", "text": "x"})
        );
        assert_eq!(TextStreamPart::Start.to_json(), json!({"type": "start"}));
        assert_eq!(
            TextStreamPart::Raw {
                raw_value: json!(1)
            }
            .to_json(),
            json!({"type": "raw", "rawValue": 1})
        );
    }
}
