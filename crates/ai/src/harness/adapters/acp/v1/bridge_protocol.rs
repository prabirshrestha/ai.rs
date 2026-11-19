//! Port of `harness-acp/src/v1/acp-v1-bridge-protocol.ts`.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::harness::adapters::acp::acp_tool_call::ACPToolCall;
use crate::harness::provider::{JsonValue, TypeValidationError};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::spec::bridge_protocol::{
    HarnessV1BridgeInboundCommand, HarnessV1BridgeOutboundMessage, HarnessV1BridgeStartBase,
};
use crate::harness::spec::diagnostic::HarnessV1DebugConfig;
use crate::harness::spec::literal::string_literal;
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::spec::tools::{
    HarnessV1BuiltinToolUseKind, HarnessV1PermissionMode, HarnessV1ToolSpec,
};
use crate::harness::utils::SandboxChannelEventCheckpoint;
use crate::harness::utils::SandboxChannelMessage;

use super::prompt::ACPContentBlock;
use super::settings::{
    ACPInstructionMapping, ACPModelMapping, ACPOutputSchemaMapping, ACPPermissionModeMapping,
};

/// `ACPBuiltinToolMapping`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPBuiltinToolMapping {
    /// Key of the tool in `HarnessV1.builtinTools`, which is also emitted as
    /// the Harness V1 `toolName`. It is the common tool name when the tool
    /// has a standardized Harness V1 equivalent; otherwise it is the native
    /// tool name.
    pub tool_name: String,
    /// Name used by the underlying agent runtime. It is present when
    /// `tool_name` is a Harness V1 common tool name that differs from the
    /// runtime's name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_name: Option<String>,
    /// Stable prefix of the display title emitted by an ACP agent. Used only
    /// when the agent omits a programmatic tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Broad capability category used to compare a built-in with the ACP
    /// tool kind when resolving unnamed calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_kind: Option<HarnessV1BuiltinToolUseKind>,
    /// JSON Schema used to identify tool calls when an ACP agent does not
    /// provide a programmatic tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<JsonValue>,
}

/// `ACPSerializableToolSpec`: a tool spec whose input schema, when present,
/// is a JSON object.
pub type ACPSerializableToolSpec = HarnessV1ToolSpec;

/// `acpSerializableToolSpecSchema.parse(tool)`.
pub fn parse_acp_serializable_tool_spec(
    tool: &HarnessV1ToolSpec,
) -> Result<ACPSerializableToolSpec, TypeValidationError> {
    if let Some(schema) = &tool.input_schema
        && !schema.is_object()
    {
        return Err(TypeValidationError::new(
            serde_json::to_value(tool).unwrap_or_default(),
            "Invalid input: expected a JSON object inputSchema",
        ));
    }
    Ok(tool.clone())
}

/// `version: 1`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ACPTurnStartConfigVersion;

impl Serialize for ACPTurnStartConfigVersion {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u8(1)
    }
}

impl<'de> Deserialize<'de> for ACPTurnStartConfigVersion {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = JsonValue::deserialize(d)?;
        if value.as_f64() == Some(1.0) {
            Ok(Self)
        } else {
            Err(D::Error::custom("Invalid input: expected 1"))
        }
    }
}

/// `ACPTurnStartConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPTurnStartConfig {
    pub version: ACPTurnStartConfigVersion,
    pub configuration_fingerprint: String,
    pub prompt: Vec<ACPContentBlock>,
    #[serde(default)]
    pub tools: Vec<ACPSerializableToolSpec>,
    #[serde(default)]
    pub builtin_tools: Vec<ACPBuiltinToolMapping>,
    pub permission_mode: HarnessV1PermissionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_mode_mapping: Option<ACPPermissionModeMapping>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_mapping: Option<ACPModelMapping>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<HarnessV1DebugConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<HarnessV1ResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema_mapping: Option<ACPOutputSchemaMapping>,
}

/// `ACPColdSessionState`: [`ACPTurnStartConfig`] without `prompt`, `debug`,
/// `model` and `modelMapping`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPColdSessionState {
    pub version: ACPTurnStartConfigVersion,
    pub configuration_fingerprint: String,
    #[serde(default)]
    pub tools: Vec<ACPSerializableToolSpec>,
    #[serde(default)]
    pub builtin_tools: Vec<ACPBuiltinToolMapping>,
    pub permission_mode: HarnessV1PermissionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_mode_mapping: Option<ACPPermissionModeMapping>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<HarnessV1ResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema_mapping: Option<ACPOutputSchemaMapping>,
}

/// The `min(1)` checks of the turn start config schemas, which serde does
/// not express.
pub(crate) fn validate_turn_start_mappings(
    model_mapping: Option<&ACPModelMapping>,
    output_schema_mapping: Option<&ACPOutputSchemaMapping>,
) -> Result<(), String> {
    if let Some(
        ACPModelMapping::SessionConfigOption { path } | ACPModelMapping::SessionModel { path },
    ) = model_mapping
        && path.is_empty()
    {
        return Err("Too small: expected modelMapping.path to have >=1 characters".into());
    }
    if let Some(ACPOutputSchemaMapping::SessionPromptMeta { path }) = output_schema_mapping
        && (path.is_empty() || path.iter().any(String::is_empty))
    {
        return Err("Too small: expected outputSchemaMapping.path to be a non-empty list of non-empty strings".into());
    }
    Ok(())
}

/// `recoveryMode` of a [`ACPStartMessage`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum ACPRecoveryMode {
    LossyRerun {
        acp_session_id: String,
        reason: String,
    },
    ColdRestore {
        acp_session_id: String,
    },
}

/// `StartMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPStartMessage {
    #[serde(flatten)]
    pub base: HarnessV1BridgeStartBase<Vec<ACPContentBlock>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruction_mapping: Option<ACPInstructionMapping>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema_mapping: Option<ACPOutputSchemaMapping>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<indexmap::IndexMap<String, JsonValue>>,
    #[serde(default)]
    pub builtin_tools: Vec<ACPBuiltinToolMapping>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_mode_mapping: Option<ACPPermissionModeMapping>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_mapping: Option<ACPModelMapping>,
    pub turn_start_config: ACPTurnStartConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_mode: Option<ACPRecoveryMode>,
}

/// `InboundMessage`: the `start` payload or a shared bridge command.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ACPInboundMessage {
    Start(Box<ACPStartMessage>),
    Command(HarnessV1BridgeInboundCommand),
}

impl From<HarnessV1BridgeInboundCommand> for ACPInboundMessage {
    fn from(command: HarnessV1BridgeInboundCommand) -> Self {
        Self::Command(command)
    }
}

impl From<ACPStartMessage> for ACPInboundMessage {
    fn from(start: ACPStartMessage) -> Self {
        Self::Start(Box::new(start))
    }
}

string_literal!(
    /// `type: 'acp-tool-call-candidate'`.
    ToolCallCandidateType = "acp-tool-call-candidate"
);
string_literal!(
    /// `type: 'acp-question-request'`.
    QuestionRequestType = "acp-question-request"
);
string_literal!(
    /// `type: 'acp-question-resolved'`.
    QuestionResolvedType = "acp-question-resolved"
);
string_literal!(
    /// `type: 'acp-host-tool-call-resolution-request'`.
    HostToolCallResolutionRequestType = "acp-host-tool-call-resolution-request"
);

/// `acpToolCallCandidateSchema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPToolCallCandidate {
    pub r#type: ToolCallCandidateType,
    pub request_id: String,
    pub tool_call: ACPToolCall,
}

/// `acpQuestionRequestSchema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPQuestionRequest {
    pub r#type: QuestionRequestType,
    pub request_id: String,
    #[serde(default)]
    pub native_request: JsonValue,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_tool_call: Option<ACPToolCall>,
}

/// `acpQuestionResolvedSchema`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPQuestionResolved {
    pub r#type: QuestionResolvedType,
    pub request_id: String,
}

/// `acpHostToolCallResolutionRequestSchema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPHostToolCallResolutionRequest {
    pub r#type: HostToolCallResolutionRequestType,
    pub request_id: String,
    pub tool_call: ACPToolCall,
    pub server_name: String,
    pub tool_names: Vec<String>,
}

/// `OutboundMessage` (`outboundMessageSchema`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ACPOutboundMessage {
    Bridge(HarnessV1BridgeOutboundMessage),
    ToolCallCandidate(ACPToolCallCandidate),
    QuestionRequest(ACPQuestionRequest),
    QuestionResolved(ACPQuestionResolved),
    HostToolCallResolutionRequest(ACPHostToolCallResolutionRequest),
}

impl<'de> Deserialize<'de> for ACPOutboundMessage {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = JsonValue::deserialize(d)?;
        let parse = |result: Result<Self, serde_json::Error>| result.map_err(D::Error::custom);
        match value.get("type").and_then(JsonValue::as_str) {
            Some(ToolCallCandidateType::VALUE) => {
                parse(serde_json::from_value(value).map(Self::ToolCallCandidate))
            }
            Some(QuestionRequestType::VALUE) => {
                parse(serde_json::from_value(value).map(Self::QuestionRequest))
            }
            Some(QuestionResolvedType::VALUE) => {
                parse(serde_json::from_value(value).map(Self::QuestionResolved))
            }
            Some(HostToolCallResolutionRequestType::VALUE) => {
                parse(serde_json::from_value(value).map(Self::HostToolCallResolutionRequest))
            }
            _ => parse(serde_json::from_value(value).map(Self::Bridge)),
        }
    }
}

impl ACPOutboundMessage {
    /// The `type` discriminator.
    pub fn type_name(&self) -> &str {
        match self {
            Self::Bridge(message) => match message {
                HarnessV1BridgeOutboundMessage::StreamPart(part) => part.type_name(),
                HarnessV1BridgeOutboundMessage::Hello(_) => "bridge-hello",
                HarnessV1BridgeOutboundMessage::UserMessageResponse(_) => "user-message-response",
                HarnessV1BridgeOutboundMessage::Stop(_) => "bridge-stop",
                HarnessV1BridgeOutboundMessage::Thread(_) => "bridge-thread",
                HarnessV1BridgeOutboundMessage::SandboxLog(_) => "sandbox-log",
                HarnessV1BridgeOutboundMessage::DebugEvent(_) => "debug-event",
            },
            Self::ToolCallCandidate(_) => ToolCallCandidateType::VALUE,
            Self::QuestionRequest(_) => QuestionRequestType::VALUE,
            Self::QuestionResolved(_) => QuestionResolvedType::VALUE,
            Self::HostToolCallResolutionRequest(_) => HostToolCallResolutionRequestType::VALUE,
        }
    }

    /// The stream part, when this frame is one.
    pub fn stream_part(&self) -> Option<&HarnessV1StreamPart> {
        match self {
            Self::Bridge(HarnessV1BridgeOutboundMessage::StreamPart(part)) => Some(part),
            _ => None,
        }
    }
}

/// An [`ACPOutboundMessage`] as the channel delivers it, with its event
/// checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct ACPOutboundFrame {
    pub message: ACPOutboundMessage,
    checkpoint: Option<SandboxChannelEventCheckpoint>,
}

impl ACPOutboundFrame {
    pub fn new(message: ACPOutboundMessage) -> Self {
        Self {
            message,
            checkpoint: None,
        }
    }
}

impl<'de> Deserialize<'de> for ACPOutboundFrame {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        ACPOutboundMessage::deserialize(d).map(Self::new)
    }
}

impl SandboxChannelMessage for ACPOutboundFrame {
    fn type_name(&self) -> &str {
        self.message.type_name()
    }

    fn from_validation_error(error: TypeValidationError) -> Self {
        Self::new(ACPOutboundMessage::Bridge(
            HarnessV1BridgeOutboundMessage::StreamPart(HarnessV1StreamPart::Error {
                error: ErrorValue::from_box(Box::new(error)),
            }),
        ))
    }

    fn set_event_checkpoint(&mut self, checkpoint: SandboxChannelEventCheckpoint) {
        self.checkpoint = Some(checkpoint);
    }

    fn event_checkpoint(&self) -> Option<&SandboxChannelEventCheckpoint> {
        self.checkpoint.as_ref()
    }
}

/// `acpHostToolCallResolutionResultSchema`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPHostToolCallResolutionResult {
    pub host_tool_call: Option<ACPHostToolCallResolution>,
}

/// `hostToolCall` of [`ACPHostToolCallResolutionResult`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPHostToolCallResolution {
    pub tool_name: String,
    pub input: indexmap::IndexMap<String, JsonValue>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_acp_specific_and_shared_outbound_frames() {
        let candidate: ACPOutboundMessage = serde_json::from_value(json!({
            "type": "acp-tool-call-candidate",
            "requestId": "r1",
            "toolCall": { "toolCallId": "c1", "title": "Read" },
        }))
        .unwrap();
        assert_eq!(candidate.type_name(), "acp-tool-call-candidate");

        let question: ACPOutboundMessage = serde_json::from_value(json!({
            "type": "acp-question-request",
            "requestId": "r2",
            "nativeRequest": { "questions": [] },
        }))
        .unwrap();
        assert_eq!(question.type_name(), "acp-question-request");

        let finish: ACPOutboundMessage =
            serde_json::from_value(json!({ "type": "text-delta", "id": "t", "delta": "x" }))
                .unwrap();
        assert_eq!(finish.type_name(), "text-delta");
        assert!(matches!(finish, ACPOutboundMessage::Bridge(_)));

        assert!(
            serde_json::from_value::<ACPOutboundMessage>(json!({
                "type": "acp-host-tool-call-resolution-request",
                "requestId": "r3",
            }))
            .is_err()
        );
    }

    #[test]
    fn turn_start_config_requires_version_one_and_defaults_tool_lists() {
        let config: ACPTurnStartConfig = serde_json::from_value(json!({
            "version": 1,
            "configurationFingerprint": "f",
            "prompt": [{ "type": "text", "text": "hi" }],
            "permissionMode": "allow-all",
        }))
        .unwrap();
        assert!(config.tools.is_empty() && config.builtin_tools.is_empty());
        assert!(
            serde_json::from_value::<ACPTurnStartConfig>(json!({
                "version": 2,
                "configurationFingerprint": "f",
                "prompt": [],
                "permissionMode": "allow-all",
            }))
            .is_err()
        );
    }

    #[test]
    fn start_messages_serialize_with_the_shared_start_fields() {
        let mut base =
            HarnessV1BridgeStartBase::new(vec![ACPContentBlock::Text { text: "hi".into() }]);
        base.permission_mode = Some(HarnessV1PermissionMode::AllowAll);
        let start = ACPStartMessage {
            base,
            instructions: None,
            instruction_mapping: None,
            output_schema_mapping: None,
            mcp_servers: None,
            builtin_tools: Vec::new(),
            permission_mode_mapping: None,
            model_mapping: None,
            turn_start_config: serde_json::from_value(json!({
                "version": 1,
                "configurationFingerprint": "f",
                "prompt": [],
                "permissionMode": "allow-all",
            }))
            .unwrap(),
            recovery_mode: Some(ACPRecoveryMode::ColdRestore {
                acp_session_id: "s".into(),
            }),
        };
        let value = serde_json::to_value(ACPInboundMessage::from(start)).unwrap();
        assert_eq!(value["type"], "start");
        assert_eq!(value["prompt"], json!([{ "type": "text", "text": "hi" }]));
        assert_eq!(value["permissionMode"], "allow-all");
        assert_eq!(value["builtinTools"], json!([]));
        assert_eq!(
            value["recoveryMode"],
            json!({ "type": "cold-restore", "acpSessionId": "s" })
        );
        assert_eq!(
            serde_json::to_value(ACPInboundMessage::from(
                HarnessV1BridgeInboundCommand::Abort
            ))
            .unwrap(),
            json!({ "type": "abort" })
        );
    }
}
