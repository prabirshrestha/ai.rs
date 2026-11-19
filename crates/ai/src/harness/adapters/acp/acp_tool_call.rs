//! Port of `harness-acp/src/acp-tool-call.ts`.

use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::harness::errors::BoxError;
use crate::harness::provider::JsonValue;

/// `ACPMetadata`.
pub type ACPMetadata = serde_json::Map<String, JsonValue>;

/// `ACPRole`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ACPRole {
    Assistant,
    User,
}

/// `ACPAnnotations`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPAnnotations {
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub audience: Option<Option<Vec<ACPRole>>>,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub last_modified: Option<Option<String>>,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub priority: Option<Option<f64>>,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

/// `ACPTextContent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ACPTextContent {
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub annotations: Option<Option<ACPAnnotations>>,
    pub text: String,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

/// `ACPImageContent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPImageContent {
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub annotations: Option<Option<ACPAnnotations>>,
    pub data: String,
    pub mime_type: String,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub uri: Option<Option<String>>,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

/// `ACPAudioContent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPAudioContent {
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub annotations: Option<Option<ACPAnnotations>>,
    pub data: String,
    pub mime_type: String,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

/// `ACPResourceLink`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPResourceLink {
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub annotations: Option<Option<ACPAnnotations>>,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<Option<String>>,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub mime_type: Option<Option<String>>,
    pub name: String,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub size: Option<Option<f64>>,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub title: Option<Option<String>>,
    pub uri: String,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

/// `ACPTextResourceContents`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPTextResourceContents {
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub mime_type: Option<Option<String>>,
    pub text: String,
    pub uri: String,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

/// `ACPBlobResourceContents`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPBlobResourceContents {
    pub blob: String,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub mime_type: Option<Option<String>>,
    pub uri: String,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

/// `ACPTextResourceContents | ACPBlobResourceContents`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ACPResourceContents {
    Text(ACPTextResourceContents),
    Blob(ACPBlobResourceContents),
}

/// `ACPEmbeddedResource`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ACPEmbeddedResource {
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub annotations: Option<Option<ACPAnnotations>>,
    pub resource: ACPResourceContents,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

/// `ACPContentBlock`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ACPContentBlock {
    Text(ACPTextContent),
    Image(ACPImageContent),
    Audio(ACPAudioContent),
    ResourceLink(ACPResourceLink),
    Resource(ACPEmbeddedResource),
}

/// `ACPToolCallContent`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "lowercase",
    rename_all_fields = "camelCase"
)]
pub enum ACPToolCallContent {
    Content {
        content: ACPContentBlock,
        #[serde(
            rename = "_meta",
            default,
            deserialize_with = "super::acp_auth::deserialize_present",
            skip_serializing_if = "Option::is_none"
        )]
        meta: Option<Option<ACPMetadata>>,
    },
    Diff {
        path: String,
        #[serde(
            default,
            deserialize_with = "super::acp_auth::deserialize_present",
            skip_serializing_if = "Option::is_none"
        )]
        old_text: Option<Option<String>>,
        new_text: String,
        #[serde(
            rename = "_meta",
            default,
            deserialize_with = "super::acp_auth::deserialize_present",
            skip_serializing_if = "Option::is_none"
        )]
        meta: Option<Option<ACPMetadata>>,
    },
    Terminal {
        terminal_id: String,
        #[serde(
            rename = "_meta",
            default,
            deserialize_with = "super::acp_auth::deserialize_present",
            skip_serializing_if = "Option::is_none"
        )]
        meta: Option<Option<ACPMetadata>>,
    },
}

/// `ACPToolKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ACPToolKind {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Think,
    Fetch,
    SwitchMode,
    Other,
}

/// `ACPToolCallStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ACPToolCallStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// `ACPToolCallLocation`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ACPToolCallLocation {
    pub path: String,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub line: Option<Option<f64>>,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

/// `ACPToolCall`.
///
/// [rust] `null` and absent stay distinct (`Option<Option<_>>`) where TS
/// allows both, so a tool call round-trips unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPToolCall {
    pub tool_call_id: String,
    #[serde(
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub name: Option<Option<String>>,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<ACPToolKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ACPToolCallStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ACPToolCallContent>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<ACPToolCallLocation>>,
    /// `rawInput?: unknown`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<JsonValue>,
    /// `rawOutput?: unknown`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<JsonValue>,
    #[serde(
        rename = "_meta",
        default,
        deserialize_with = "super::acp_auth::deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub meta: Option<Option<ACPMetadata>>,
}

impl ACPToolCall {
    /// A tool call with only the required fields.
    pub fn new(tool_call_id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            name: None,
            title: title.into(),
            kind: None,
            status: None,
            content: None,
            locations: None,
            raw_input: None,
            raw_output: None,
            meta: None,
        }
    }

    /// `name` when it is a string.
    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().and_then(|name| name.as_deref())
    }
}

/// `ACPResolvedHostToolCall`: a tool call from the ACP agent, resolved to one
/// of the host tools that the harness exposes through its own MCP server.
#[derive(Debug, Clone, PartialEq)]
pub struct ACPResolvedHostToolCall {
    /// Name of the host tool. Must be one of the `tool_names` passed to the
    /// resolver.
    pub tool_name: String,
    /// Input for the host tool, with any agent-specific wrapper removed.
    pub input: IndexMap<String, JsonValue>,
}

/// Input of [`ACPResolveHostToolCall`].
#[derive(Debug, Clone, Copy)]
pub struct ACPResolveHostToolCallOptions<'a> {
    pub tool_call: &'a ACPToolCall,
    pub server_name: &'a str,
    pub tool_names: &'a [String],
}

/// `ACPResolveHostToolCall`: resolves an ACP tool call to the host tool it
/// invokes.
///
/// Called with the merged snapshot of all updates seen so far for a tool
/// call, and with standalone permission-request tool calls. Return `None` if
/// the tool call does not invoke a host tool, or cannot be resolved from the
/// data available so far.
///
/// [rust] A TS resolver can throw; here it returns the error.
pub type ACPResolveHostToolCall = Arc<
    dyn Fn(ACPResolveHostToolCallOptions<'_>) -> Result<Option<ACPResolvedHostToolCall>, BoxError>
        + Send
        + Sync,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_calls_round_trip_with_null_and_absent_fields() {
        let value = json!({
            "toolCallId": "call-1",
            "name": null,
            "title": "Read file",
            "kind": "switch_mode",
            "status": "in_progress",
            "content": [
                { "type": "content", "content": { "type": "text", "text": "hi" } },
                { "type": "diff", "path": "a.txt", "oldText": null, "newText": "b" },
                { "type": "terminal", "terminalId": "t1" },
                { "type": "content", "content": { "type": "resource", "resource": { "uri": "file:///a", "blob": "AA==" } } },
            ],
            "locations": [{ "path": "a.txt", "line": 3.0 }],
            "rawInput": { "path": "a.txt" },
            "_meta": { "x": 1 },
        });
        let call: ACPToolCall = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(call.name(), None);
        assert_eq!(serde_json::to_value(&call).unwrap(), value);
    }
}
