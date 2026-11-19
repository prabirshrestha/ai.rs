//! Port of `harness-v1-message.ts`: persisted harness messages using the V4
//! prompt content shapes without `providerOptions`.

use serde::{Deserialize, Serialize};

use crate::harness::provider::{JsonValue, SharedV4FileData, SharedV4ReasoningFileData};
use crate::harness::spec::stream_part::HarnessV1Metadata;

/// `HarnessV1TextPart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessV1TextPart {
    pub text: String,
}

/// `HarnessV1FilePart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1FilePart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    pub data: SharedV4FileData,
    pub media_type: String,
}

/// `HarnessV1CustomPart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessV1CustomPart {
    pub kind: String,
}

/// `HarnessV1ReasoningPart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessV1ReasoningPart {
    pub text: String,
}

/// `HarnessV1ReasoningFilePart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1ReasoningFilePart {
    pub data: SharedV4ReasoningFileData,
    pub media_type: String,
}

/// `HarnessV1ToolCallPart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1ToolCallPart {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_name: Option<String>,
}

/// A `content` part of [`HarnessV1ToolResultOutput`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum HarnessV1ToolResultContentPart {
    Text {
        text: String,
    },
    File {
        data: SharedV4FileData,
        media_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    Custom {},
}

/// `HarnessV1ToolResultOutput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum HarnessV1ToolResultOutput {
    Text {
        value: String,
    },
    Json {
        value: JsonValue,
    },
    ExecutionDenied {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    ErrorText {
        value: String,
    },
    ErrorJson {
        value: JsonValue,
    },
    Content {
        value: Vec<HarnessV1ToolResultContentPart>,
    },
}

/// `HarnessV1ToolResultPart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1ToolResultPart {
    pub tool_call_id: String,
    pub tool_name: String,
    pub output: HarnessV1ToolResultOutput,
}

/// `HarnessV1ToolApprovalResponsePart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1ToolApprovalResponsePart {
    pub approval_id: String,
    pub approved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// `HarnessV1MessagePart`: any part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum HarnessV1MessagePart {
    Text(HarnessV1TextPart),
    File(HarnessV1FilePart),
    Custom(HarnessV1CustomPart),
    Reasoning(HarnessV1ReasoningPart),
    ReasoningFile(HarnessV1ReasoningFilePart),
    ToolCall(HarnessV1ToolCallPart),
    ToolResult(HarnessV1ToolResultPart),
    ToolApprovalResponse(HarnessV1ToolApprovalResponsePart),
}

/// Parts allowed in a user message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum HarnessV1UserMessagePart {
    Text(HarnessV1TextPart),
    File(HarnessV1FilePart),
}

/// Parts allowed in an assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum HarnessV1AssistantMessagePart {
    Text(HarnessV1TextPart),
    File(HarnessV1FilePart),
    Custom(HarnessV1CustomPart),
    Reasoning(HarnessV1ReasoningPart),
    ReasoningFile(HarnessV1ReasoningFilePart),
    ToolCall(HarnessV1ToolCallPart),
    ToolResult(HarnessV1ToolResultPart),
}

/// Parts allowed in a tool message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum HarnessV1ToolMessagePart {
    ToolResult(HarnessV1ToolResultPart),
    ToolApprovalResponse(HarnessV1ToolApprovalResponsePart),
}

/// `HarnessV1UserMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1UserMessage {
    pub content: Vec<HarnessV1UserMessagePart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harness_metadata: Option<HarnessV1Metadata>,
}

/// `HarnessV1AssistantMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1AssistantMessage {
    pub content: Vec<HarnessV1AssistantMessagePart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harness_metadata: Option<HarnessV1Metadata>,
}

/// `HarnessV1ToolMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1ToolMessage {
    pub content: Vec<HarnessV1ToolMessagePart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harness_metadata: Option<HarnessV1Metadata>,
}

/// `HarnessV1Message`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum HarnessV1Message {
    User(HarnessV1UserMessage),
    Assistant(HarnessV1AssistantMessage),
    Tool(HarnessV1ToolMessage),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Runtime counterpart of harness-v1-message.test-d.ts: every message shape
    // keeps the V4 JSON layout.
    #[test]
    fn messages_round_trip() {
        for message in [
            json!({ "role": "user", "content": [
                { "type": "text", "text": "hi" },
                { "type": "file", "data": { "type": "data", "data": "AA==" }, "mediaType": "image/png", "filename": "a.png" }
            ], "at": "2026-01-01T00:00:00Z" }),
            json!({ "role": "assistant", "content": [
                { "type": "reasoning", "text": "r" },
                { "type": "reasoning-file", "data": { "type": "url", "url": "https://x" }, "mediaType": "image/png" },
                { "type": "custom", "kind": "x.y" },
                { "type": "tool-call", "toolCallId": "c", "toolName": "bash", "input": { "command": "ls" }, "providerExecuted": true, "nativeName": "Bash" },
                { "type": "tool-result", "toolCallId": "c", "toolName": "bash", "output": { "type": "content", "value": [
                    { "type": "text", "text": "ok" }, { "type": "custom" }
                ] } }
            ], "harnessMetadata": { "codex": { "turn": 1 } } }),
            json!({ "role": "tool", "content": [
                { "type": "tool-result", "toolCallId": "c", "toolName": "x", "output": { "type": "execution-denied" } },
                { "type": "tool-approval-response", "approvalId": "a", "approved": true }
            ] }),
        ] {
            let parsed: HarnessV1Message = serde_json::from_value(message.clone()).unwrap();
            assert_eq!(serde_json::to_value(parsed).unwrap(), message);
        }
    }

    #[test]
    fn user_messages_reject_assistant_parts() {
        let bad = json!({ "role": "user", "content": [{ "type": "reasoning", "text": "r" }] });
        assert!(serde_json::from_value::<HarnessV1Message>(bad).is_err());
    }
}
