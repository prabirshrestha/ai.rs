//! Mirrors of `@ai-sdk/provider-utils` message and content-part types
//! (`ModelMessage`, `TextPart`, `ToolResultPart`, `ToolResultOutput`, …).

use serde::{Deserialize, Serialize};

use crate::harness::provider::{
    DataContent, JsonValue, SharedV4FileData, SharedV4ProviderOptions, SharedV4ProviderReference,
};

/// `ProviderOptions`.
pub type ProviderOptions = SharedV4ProviderOptions;
/// `ProviderReference`.
pub type ProviderReference = SharedV4ProviderReference;
/// `FileData`.
pub type FileData = SharedV4FileData;

/// `TextPart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextPart {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// `ImagePart.image`: `DataContent | URL | ProviderReference`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImagePartImage {
    /// Base64 data, bytes, or a URL string.
    Content(DataContent),
    Reference(ProviderReference),
}

/// `ImagePart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagePart {
    pub image: ImagePartImage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// `FilePart.data`: `FileData | DataContent | URL | ProviderReference`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilePartData {
    FileData(FileData),
    /// Base64 data, bytes, or a URL string.
    Content(DataContent),
    Reference(ProviderReference),
}

/// `FilePart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FilePart {
    pub data: FilePartData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// `ReasoningPart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningPart {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// `CustomPart`. `kind` is `${string}.${string}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomPart {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// `ReasoningFilePart.data`: `FileDataData | FileDataUrl | DataContent | URL`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReasoningFilePartData {
    FileData(crate::harness::provider::SharedV4ReasoningFileData),
    Content(DataContent),
}

/// `ReasoningFilePart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningFilePart {
    pub data: ReasoningFilePartData,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// `ToolCallPart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallPart {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
}

/// `ToolResultPart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultPart {
    pub tool_call_id: String,
    pub tool_name: String,
    pub output: ToolResultOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// `fileId: string | Record<string, string>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FileId {
    Id(String),
    ByProvider(indexmap::IndexMap<String, String>),
}

/// One part of a `content` [`ToolResultOutput`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum ToolResultContentPart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    File {
        data: FileData,
        media_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    FileData {
        data: String,
        media_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    FileUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    FileId {
        file_id: FileId,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    FileReference {
        provider_reference: ProviderReference,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    ImageData {
        data: String,
        media_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    ImageUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    ImageFileId {
        file_id: FileId,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    ImageFileReference {
        provider_reference: ProviderReference,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    Custom {
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
}

/// `ToolResultOutput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum ToolResultOutput {
    Text {
        value: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    Json {
        value: JsonValue,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    ExecutionDenied {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    ErrorText {
        value: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    ErrorJson {
        value: JsonValue,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<ProviderOptions>,
    },
    Content {
        value: Vec<ToolResultContentPart>,
    },
}

/// `ToolApprovalRequest` (assistant content part).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolApprovalRequest {
    pub approval_id: String,
    pub tool_call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_automatic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema_input: Option<JsonValue>,
}

/// `ToolApprovalResponse` (tool content part).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolApprovalResponse {
    pub approval_id: String,
    pub approved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
}

/// `TextPart | ImagePart | FilePart`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum UserContentPart {
    Text(TextPart),
    Image(ImagePart),
    File(FilePart),
}

/// `UserContent`: `string | Array<TextPart | ImagePart | FilePart>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Parts(Vec<UserContentPart>),
}

/// `UserModelMessage` (`role: 'user'`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserModelMessage {
    pub content: UserContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// `SystemModelMessage` (`role: 'system'`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemModelMessage {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// One part of `AssistantContent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AssistantContentPart {
    Text(TextPart),
    Custom(CustomPart),
    File(FilePart),
    Reasoning(ReasoningPart),
    ReasoningFile(ReasoningFilePart),
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
    ToolApprovalRequest(ToolApprovalRequest),
}

/// `AssistantContent`: `string | Array<…>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssistantContent {
    Text(String),
    Parts(Vec<AssistantContentPart>),
}

/// `AssistantModelMessage` (`role: 'assistant'`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantModelMessage {
    pub content: AssistantContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// One part of `ToolContent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolContentPart {
    ToolResult(ToolResultPart),
    ToolApprovalResponse(ToolApprovalResponse),
}

/// `ToolModelMessage` (`role: 'tool'`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolModelMessage {
    pub content: Vec<ToolContentPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

/// `ModelMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ModelMessage {
    System(SystemModelMessage),
    User(UserModelMessage),
    Assistant(AssistantModelMessage),
    Tool(ToolModelMessage),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip<T: Serialize + serde::de::DeserializeOwned>(value: serde_json::Value) {
        let parsed: T = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), value);
    }

    #[test]
    fn model_messages_round_trip() {
        round_trip::<ModelMessage>(json!({ "role": "system", "content": "be brief" }));
        round_trip::<ModelMessage>(json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "hi" },
                { "type": "image", "image": "aGVsbG8=", "mediaType": "image/png" },
                { "type": "file", "data": { "type": "url", "url": "https://x/y.pdf" }, "mediaType": "application/pdf" }
            ]
        }));
        round_trip::<ModelMessage>(json!({
            "role": "assistant",
            "content": [
                { "type": "reasoning", "text": "think" },
                { "type": "tool-call", "toolCallId": "c1", "toolName": "bash", "input": { "command": "ls" } },
                { "type": "tool-approval-request", "approvalId": "a1", "toolCallId": "c1" }
            ]
        }));
        round_trip::<ModelMessage>(json!({
            "role": "tool",
            "content": [
                { "type": "tool-result", "toolCallId": "c1", "toolName": "bash",
                  "output": { "type": "content", "value": [
                      { "type": "text", "text": "ok" },
                      { "type": "image-data", "data": "AA==", "mediaType": "image/png" },
                      { "type": "file-id", "fileId": { "openai": "f1" } }
                  ] } },
                { "type": "tool-approval-response", "approvalId": "a1", "approved": false, "reason": "no" }
            ]
        }));
    }

    #[test]
    fn user_content_accepts_a_plain_string() {
        round_trip::<UserModelMessage>(json!({ "content": "hello" }));
    }
}
