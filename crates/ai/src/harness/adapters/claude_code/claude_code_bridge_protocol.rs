//! Port of `harness-claude-code/src/claude-code-bridge-protocol.ts`.
//!
//! Claude Code's bridge wire protocol. The outbound events, transport
//! frames, shared inbound commands, and `bridge-ready` line all come from
//! the shared harness protocol. The only Claude-specific piece is the
//! `start` payload, which carries Claude SDK configuration.

use indexmap::IndexMap;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::harness::provider::{JsonValue, TypeValidationError};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::spec::bridge_protocol::{
    HarnessV1BridgeInboundCommand, HarnessV1BridgeOutboundMessage, HarnessV1BridgeReady,
    HarnessV1BridgeStartBase,
};
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::utils::{
    HarnessPromptContent, SandboxChannelEventCheckpoint, SandboxChannelMessage,
};

use super::claude_code_thinking::ClaudeCodeThinkingConfig;

/// `OutboundMessage`: a shared outbound frame plus the optional `startId`
/// of the turn it belongs to.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeCodeOutboundMessage {
    pub message: HarnessV1BridgeOutboundMessage,
    pub start_id: Option<String>,
    checkpoint: Option<SandboxChannelEventCheckpoint>,
}

impl ClaudeCodeOutboundMessage {
    pub fn new(message: HarnessV1BridgeOutboundMessage, start_id: Option<String>) -> Self {
        Self {
            message,
            start_id,
            checkpoint: None,
        }
    }

    /// The stream part, when this frame is one.
    pub fn stream_part(&self) -> Option<&HarnessV1StreamPart> {
        match &self.message {
            HarnessV1BridgeOutboundMessage::StreamPart(part) => Some(part),
            _ => None,
        }
    }
}

impl Serialize for ClaudeCodeOutboundMessage {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(&self.message).map_err(serde::ser::Error::custom)?;
        if let (Some(start_id), Some(object)) = (&self.start_id, value.as_object_mut()) {
            object.insert("startId".into(), JsonValue::String(start_id.clone()));
        }
        value.serialize(s)
    }
}

impl<'de> Deserialize<'de> for ClaudeCodeOutboundMessage {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let mut value = JsonValue::deserialize(d)?;
        let start_id = match value.as_object_mut().and_then(|o| o.remove("startId")) {
            None => None,
            Some(JsonValue::String(start_id)) => Some(start_id),
            Some(_) => {
                return Err(D::Error::custom(
                    "Invalid input: expected string at startId",
                ));
            }
        };
        let message = serde_json::from_value(value).map_err(D::Error::custom)?;
        Ok(Self::new(message, start_id))
    }
}

impl SandboxChannelMessage for ClaudeCodeOutboundMessage {
    fn type_name(&self) -> &str {
        match &self.message {
            HarnessV1BridgeOutboundMessage::StreamPart(part) => part.type_name(),
            HarnessV1BridgeOutboundMessage::Hello(_) => "bridge-hello",
            HarnessV1BridgeOutboundMessage::UserMessageResponse(_) => "user-message-response",
            HarnessV1BridgeOutboundMessage::Stop(_) => "bridge-stop",
            HarnessV1BridgeOutboundMessage::Thread(_) => "bridge-thread",
            HarnessV1BridgeOutboundMessage::SandboxLog(_) => "sandbox-log",
            HarnessV1BridgeOutboundMessage::DebugEvent(_) => "debug-event",
        }
    }

    fn from_validation_error(error: TypeValidationError) -> Self {
        Self::new(
            HarnessV1BridgeOutboundMessage::StreamPart(HarnessV1StreamPart::Error {
                error: ErrorValue::from_box(Box::new(error)),
            }),
            None,
        )
    }

    fn set_event_checkpoint(&mut self, checkpoint: SandboxChannelEventCheckpoint) {
        self.checkpoint = Some(checkpoint);
    }

    fn event_checkpoint(&self) -> Option<&SandboxChannelEventCheckpoint> {
        self.checkpoint.as_ref()
    }
}

/// `effort` of a [`ClaudeCodeStartMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClaudeCodeEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// A value of `skillOverrides`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClaudeCodeSkillOverride {
    On,
    NameOnly,
    UserInvocableOnly,
    Off,
}

/// A value of `toolHints`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaudeCodeToolHint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub always_load: Option<bool>,
}

/// `StartMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaudeCodeStartMessage {
    #[serde(flatten)]
    pub base: HarnessV1BridgeStartBase<HarnessPromptContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readable_skill_roots: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub thinking: ClaudeCodeThinkingConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ClaudeCodeEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_overrides: Option<IndexMap<String, ClaudeCodeSkillOverride>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_budget_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_directories: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_mcp_config: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_mode: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question_timeout_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_dynamic_sections: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_progress_summaries: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_subagent_text: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<IndexMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<IndexMap<String, JsonValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_hints: Option<IndexMap<String, ClaudeCodeToolHint>>,
    /// Resume signal. When true, the bridge passes `{ continue: true }` to
    /// the Claude SDK so the in-workdir thread state is rehydrated. The host
    /// sets this on the first prompt after a cross-process resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#continue: Option<bool>,
    /// Exact conversation to resume, as the Claude SDK's `resume` option.
    ///
    /// Preferred over `continue` whenever the host knows which conversation
    /// it means: `continue` resolves to the most recent thread in the
    /// workdir, which is the wrong one as soon as anything else has run
    /// there since, including a resume that itself started a new thread.
    /// Mutually exclusive with `continue` in the SDK, so the bridge sends one
    /// or the other.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_session_id: Option<String>,
    /// A tool call the previous bridge deferred while it waited for host
    /// approval. The bridge resumes `resumeSessionId` without a prompt once
    /// the host answers, so Claude Code runs that same call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred_tool_use_id: Option<String>,
}

impl ClaudeCodeStartMessage {
    pub fn new(prompt: HarnessPromptContent, thinking: ClaudeCodeThinkingConfig) -> Self {
        Self {
            base: HarnessV1BridgeStartBase::new(prompt),
            readable_skill_roots: None,
            instructions: None,
            thinking,
            effort: None,
            max_turns: None,
            skill_overrides: None,
            max_budget_usd: None,
            additional_directories: None,
            strict_mcp_config: None,
            auto_mode: None,
            question_timeout_ms: None,
            exclude_dynamic_sections: None,
            agent_progress_summaries: None,
            forward_subagent_text: None,
            env: None,
            skills: None,
            mcp_servers: None,
            tool_hints: None,
            r#continue: None,
            resume_session_id: None,
            deferred_tool_use_id: None,
        }
    }
}

/// `InboundMessage`: the `start` payload or a shared bridge command.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ClaudeCodeInboundMessage {
    Start(Box<ClaudeCodeStartMessage>),
    Command(HarnessV1BridgeInboundCommand),
}

impl<'de> Deserialize<'de> for ClaudeCodeInboundMessage {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = JsonValue::deserialize(d)?;
        if value.get("type").and_then(JsonValue::as_str) == Some("start") {
            serde_json::from_value(value)
                .map(|start| Self::Start(Box::new(start)))
                .map_err(D::Error::custom)
        } else {
            serde_json::from_value(value)
                .map(Self::Command)
                .map_err(D::Error::custom)
        }
    }
}

impl From<HarnessV1BridgeInboundCommand> for ClaudeCodeInboundMessage {
    fn from(command: HarnessV1BridgeInboundCommand) -> Self {
        Self::Command(command)
    }
}

impl From<ClaudeCodeStartMessage> for ClaudeCodeInboundMessage {
    fn from(start: ClaudeCodeStartMessage) -> Self {
        Self::Start(Box::new(start))
    }
}

/// `BridgeReady`.
pub type ClaudeCodeBridgeReady = HarnessV1BridgeReady;

#[cfg(test)]
#[path = "claude_code_bridge_protocol_tests.rs"]
mod tests;
