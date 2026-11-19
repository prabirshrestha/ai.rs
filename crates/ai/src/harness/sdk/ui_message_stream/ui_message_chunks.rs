//! Mirror of `ai` `ui-message-stream/ui-message-chunks.ts`.
//!
//! [rust] Carries the chunks a harness UI message stream produces, like
//! [`TextStreamPart`](crate::harness::sdk::text_stream_part::TextStreamPart).
//! `custom`, `source-url`, `source-document`, `file`, `reasoning-file`,
//! `data-*`, `steering-message` and `reset-step` chunks come from stream
//! parts the harness never emits and are left out. Metadata is JSON.

use serde::{Deserialize, Serialize};

use crate::harness::provider::{JsonObject, JsonValue};
use crate::harness::sdk::content::{FinishReason, ProviderMetadata};
use crate::harness::sdk::ui_message_stream::ui_messages::InputEncoding;

/// `UIMessageChunk`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum UIMessageChunk {
    TextStart {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    TextDelta {
        id: String,
        delta: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    TextEnd {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<JsonValue>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ReasoningStart {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ReasoningDelta {
        id: String,
        delta: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ReasoningEnd {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    Error {
        error_text: String,
    },
    ToolInputAvailable {
        tool_call_id: String,
        tool_name: String,
        input: JsonValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_metadata: Option<JsonObject>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    ToolInputError {
        tool_call_id: String,
        tool_name: String,
        input: JsonValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_metadata: Option<JsonObject>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        error_text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    ToolApprovalRequest {
        approval_id: String,
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        approval_descriptor: Option<JsonValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input_schema_input: Option<JsonValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_automatic: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolApprovalResponse {
        approval_id: String,
        approved: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolOutputAvailable {
        tool_call_id: String,
        output: JsonValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_metadata: Option<JsonObject>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preliminary: Option<bool>,
    },
    ToolOutputError {
        tool_call_id: String,
        error_text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_metadata: Option<JsonObject>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
    },
    ToolOutputDenied {
        tool_call_id: String,
    },
    ToolInputStart {
        tool_call_id: String,
        tool_name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input_encoding: Option<InputEncoding>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_metadata: Option<JsonObject>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    ToolInputDelta {
        tool_call_id: String,
        input_text_delta: String,
    },
    StartStep,
    FinishStep,
    Start {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message_metadata: Option<JsonValue>,
    },
    Finish {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        finish_reason: Option<FinishReason>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message_metadata: Option<JsonValue>,
    },
    Abort {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    MessageMetadata {
        message_metadata: JsonValue,
    },
}

impl UIMessageChunk {
    /// `chunk.type`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::TextStart { .. } => "text-start",
            Self::TextDelta { .. } => "text-delta",
            Self::TextEnd { .. } => "text-end",
            Self::ReasoningStart { .. } => "reasoning-start",
            Self::ReasoningDelta { .. } => "reasoning-delta",
            Self::ReasoningEnd { .. } => "reasoning-end",
            Self::Error { .. } => "error",
            Self::ToolInputAvailable { .. } => "tool-input-available",
            Self::ToolInputError { .. } => "tool-input-error",
            Self::ToolApprovalRequest { .. } => "tool-approval-request",
            Self::ToolApprovalResponse { .. } => "tool-approval-response",
            Self::ToolOutputAvailable { .. } => "tool-output-available",
            Self::ToolOutputError { .. } => "tool-output-error",
            Self::ToolOutputDenied { .. } => "tool-output-denied",
            Self::ToolInputStart { .. } => "tool-input-start",
            Self::ToolInputDelta { .. } => "tool-input-delta",
            Self::StartStep => "start-step",
            Self::FinishStep => "finish-step",
            Self::Start { .. } => "start",
            Self::Finish { .. } => "finish",
            Self::Abort { .. } => "abort",
            Self::MessageMetadata { .. } => "message-metadata",
        }
    }

    /// The chunk as JSON.
    pub fn to_json(&self) -> JsonValue {
        serde_json::to_value(self).unwrap_or(JsonValue::Null)
    }
}
