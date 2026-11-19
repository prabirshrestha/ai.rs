//! Port of `harness-codex/src/codex-bridge-protocol.ts`.
//!
//! Codex's bridge wire protocol. The outbound events (including
//! `file-change` and the `bridge-thread` resume coordinate), transport
//! frames, shared inbound commands, and `bridge-ready` line all come from
//! the shared harness protocol. The only Codex-specific piece is the
//! `start` payload.

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

/// `OutboundMessage`: `harnessV1BridgeOutboundMessageSchema`.
///
/// [rust] A wrapper so the channel can store each frame's checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct CodexOutboundMessage {
    pub message: HarnessV1BridgeOutboundMessage,
    checkpoint: Option<SandboxChannelEventCheckpoint>,
}

impl CodexOutboundMessage {
    pub fn new(message: HarnessV1BridgeOutboundMessage) -> Self {
        Self {
            message,
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

impl Serialize for CodexOutboundMessage {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.message.serialize(s)
    }
}

impl<'de> Deserialize<'de> for CodexOutboundMessage {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        HarnessV1BridgeOutboundMessage::deserialize(d).map(Self::new)
    }
}

impl SandboxChannelMessage for CodexOutboundMessage {
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
        Self::new(HarnessV1BridgeOutboundMessage::StreamPart(
            HarnessV1StreamPart::Error {
                error: ErrorValue::from_box(Box::new(error)),
            },
        ))
    }

    fn set_event_checkpoint(&mut self, checkpoint: SandboxChannelEventCheckpoint) {
        self.checkpoint = Some(checkpoint);
    }

    fn event_checkpoint(&self) -> Option<&SandboxChannelEventCheckpoint> {
        self.checkpoint.as_ref()
    }
}

/// `reasoningEffort` of a [`CodexStartMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CodexReasoningEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// `StartMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexStartMessage {
    #[serde(flatten)]
    pub base: HarnessV1BridgeStartBase<HarnessPromptContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<CodexReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_config: Option<IndexMap<String, JsonValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<IndexMap<String, JsonValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    /// The host carries this app-server thread ID in lifecycle state.
    /// Supplying it makes the bridge call `thread/resume` instead of
    /// starting a fresh thread.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restart_thread: Option<bool>,
}

impl CodexStartMessage {
    pub fn new(prompt: HarnessPromptContent) -> Self {
        Self {
            base: HarnessV1BridgeStartBase::new(prompt),
            instructions: None,
            reasoning_effort: None,
            web_search: None,
            codex_config: None,
            mcp_servers: None,
            headers: None,
            resume_thread_id: None,
            restart_thread: None,
        }
    }
}

/// `InboundMessage`: the `start` payload or a shared bridge command.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum CodexInboundMessage {
    Start(Box<CodexStartMessage>),
    Command(HarnessV1BridgeInboundCommand),
}

impl<'de> Deserialize<'de> for CodexInboundMessage {
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

impl From<HarnessV1BridgeInboundCommand> for CodexInboundMessage {
    fn from(command: HarnessV1BridgeInboundCommand) -> Self {
        Self::Command(command)
    }
}

impl From<CodexStartMessage> for CodexInboundMessage {
    fn from(start: CodexStartMessage) -> Self {
        Self::Start(Box::new(start))
    }
}

/// `BridgeReady`.
pub type CodexBridgeReady = HarnessV1BridgeReady;

#[cfg(test)]
#[path = "codex_bridge_protocol_tests.rs"]
mod tests;
