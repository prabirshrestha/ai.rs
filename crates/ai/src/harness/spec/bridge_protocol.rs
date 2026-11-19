//! Port of `harness-v1-bridge-protocol.ts`: the wire protocol shared by
//! every bridge-backed adapter (host <-> in-sandbox bridge over WebSocket).
//!
//! Three tiers:
//! 1. outbound events: [`HarnessV1StreamPart`];
//! 2. outbound transport and control frames (`bridge-hello`, `bridge-stop`,
//!    `bridge-thread`, diagnostics);
//! 3. inbound commands. Each adapter defines its own `start` payload around
//!    [`HarnessV1BridgeStartBase`] and builds its inbound union with
//!    [`HarnessV1BridgeInboundCommand`].

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::harness::provider::{JsonObject, JsonValue, TypeValidationError, parse_json};
use crate::harness::spec::diagnostic::{
    HarnessV1DebugConfig, HarnessV1DebugLevel, HarnessV1Diagnostic, HarnessV1DiagnosticError,
    HarnessV1DiagnosticKind, HarnessV1LogStream,
};
use crate::harness::spec::literal::string_literal;
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::spec::tools::{HarnessV1BuiltinToolFiltering, HarnessV1PermissionMode};

string_literal!(
    /// `type: 'start'`.
    StartType = "start"
);
string_literal!(
    /// `type: 'bridge-hello'`.
    BridgeHelloType = "bridge-hello"
);
string_literal!(
    /// `type: 'user-message-response'`.
    UserMessageResponseType = "user-message-response"
);
string_literal!(
    /// `type: 'bridge-stop'`.
    BridgeStopType = "bridge-stop"
);
string_literal!(
    /// `type: 'bridge-thread'`.
    BridgeThreadType = "bridge-thread"
);
string_literal!(
    /// `type: 'sandbox-log'`.
    SandboxLogType = "sandbox-log"
);
string_literal!(
    /// `type: 'debug-event'`.
    DebugEventType = "debug-event"
);
string_literal!(
    /// `type: 'user-message'`.
    UserMessageType = "user-message"
);
string_literal!(
    /// `type: 'bridge-ready'`.
    BridgeReadyType = "bridge-ready"
);

/// `nullish()` fields: absent, `null`, or a value, kept apart on round trip.
mod nullish {
    use super::*;

    pub fn serialize<T: Serialize, S: Serializer>(
        value: &Option<Option<T>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(Some(v)) => v.serialize(s),
            _ => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Option<T>>, D::Error> {
        Ok(Some(Option::<T>::deserialize(d)?))
    }
}

/// `HarnessV1BridgeToolWire`: the part of a host tool sent on `start`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1BridgeToolWire {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<JsonValue>,
}

/// `harnessV1BridgePermissionModeSchema`.
pub type HarnessV1BridgePermissionMode = HarnessV1PermissionMode;
/// `harnessV1BridgeBuiltinToolFilteringSchema`.
pub type HarnessV1BridgeBuiltinToolFiltering = HarnessV1BuiltinToolFiltering;
/// `harnessV1BridgeResponseFormatSchema`.
pub type HarnessV1BridgeResponseFormat = HarnessV1ResponseFormat;

/// `harnessV1BridgeStartBaseSchema`: common fields of the inbound `start`.
///
/// Adapters `#[serde(flatten)]` this into their own `start` payload. `P` is
/// the prompt type: `String` in the base, content parts in adapters that
/// override it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1BridgeStartBase<P = String> {
    pub r#type: StartType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_id: Option<String>,
    pub prompt: P,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<HarnessV1BridgeToolWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<HarnessV1DebugConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permission_mode: Option<HarnessV1BridgePermissionMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub builtin_tool_filtering: Option<HarnessV1BridgeBuiltinToolFiltering>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<HarnessV1BridgeResponseFormat>,
}

impl<P> HarnessV1BridgeStartBase<P> {
    pub fn new(prompt: P) -> Self {
        Self {
            r#type: StartType,
            start_id: None,
            prompt,
            tools: None,
            model: None,
            debug: None,
            permission_mode: None,
            builtin_tool_filtering: None,
            response_format: None,
        }
    }
}

/// `bridge-hello` capabilities.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1BridgeHelloCapabilities {
    #[serde(
        rename = "idempotentStart",
        default,
        skip_serializing_if = "Option::is_none",
        with = "nullish"
    )]
    pub idempotent_start: Option<Option<bool>>,
    #[serde(
        rename = "experimental_userMessageResponses",
        skip_serializing_if = "Option::is_none"
    )]
    pub experimental_user_message_responses: Option<bool>,
}

/// `harnessV1BridgeHelloSchema`: sent as soon as the bridge accepts an
/// authenticated connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1BridgeHello {
    pub r#type: BridgeHelloType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<HarnessV1BridgeHelloCapabilities>,
}

/// `error` of a user-message response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1BridgeErrorMessage {
    pub message: String,
}

/// `experimental_harnessV1BridgeUserMessageResponseSchema`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentalHarnessV1BridgeUserMessageResponse {
    pub r#type: UserMessageResponseType,
    pub message_id: String,
    pub accepted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<HarnessV1BridgeErrorMessage>,
}

/// `harnessV1BridgeStopSchema`: reply to an inbound `stop`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessV1BridgeStop {
    pub r#type: BridgeStopType,
    #[serde(default)]
    pub data: JsonValue,
}

/// `harnessV1BridgeThreadSchema`: a resume coordinate the bridge announces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1BridgeThread {
    pub r#type: BridgeThreadType,
    pub thread_id: String,
}

/// `harnessV1BridgeSandboxLogSchema`: one captured console line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1BridgeSandboxLog {
    pub r#type: SandboxLogType,
    pub source: String,
    pub stream: HarnessV1LogStream,
    pub line: String,
}

/// `harnessV1BridgeDebugEventSchema`: a structured bridge diagnostic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessV1BridgeDebugEvent {
    pub r#type: DebugEventType,
    pub level: HarnessV1DebugLevel,
    pub subsystem: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attrs: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<HarnessV1DiagnosticError>,
}

/// `HarnessV1BridgeOutboundMessage`: every frame a bridge can send.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HarnessV1BridgeOutboundMessage {
    StreamPart(HarnessV1StreamPart),
    Hello(HarnessV1BridgeHello),
    UserMessageResponse(ExperimentalHarnessV1BridgeUserMessageResponse),
    Stop(HarnessV1BridgeStop),
    Thread(HarnessV1BridgeThread),
    SandboxLog(HarnessV1BridgeSandboxLog),
    DebugEvent(HarnessV1BridgeDebugEvent),
}

impl HarnessV1BridgeOutboundMessage {
    /// `harnessV1BridgeOutboundMessageSchema.parse(value)`.
    pub fn parse(value: JsonValue) -> Result<Self, TypeValidationError> {
        parse_json(value)
    }
}

/// A diagnostics frame: sandbox log or debug event.
#[derive(Debug, Clone, PartialEq)]
pub enum HarnessV1BridgeDiagnosticFrame<'a> {
    SandboxLog(&'a HarnessV1BridgeSandboxLog),
    DebugEvent(&'a HarnessV1BridgeDebugEvent),
}

/// `harnessV1DiagnosticFromBridgeFrame`: `stderr` lines become `warn`,
/// `stdout` lines `info`; structured events pass through.
pub fn harness_v1_diagnostic_from_bridge_frame(
    frame: HarnessV1BridgeDiagnosticFrame<'_>,
    session_id: Option<String>,
    timestamp: i64,
) -> HarnessV1Diagnostic {
    match frame {
        HarnessV1BridgeDiagnosticFrame::SandboxLog(log) => HarnessV1Diagnostic {
            level: match log.stream {
                HarnessV1LogStream::Stderr => HarnessV1DebugLevel::Warn,
                HarnessV1LogStream::Stdout => HarnessV1DebugLevel::Info,
            },
            message: log.line.clone(),
            subsystem: format!("sandbox.log.{}", log.source),
            kind: HarnessV1DiagnosticKind::Log,
            source: Some(log.source.clone()),
            stream: Some(log.stream),
            attrs: None,
            error: None,
            session_id,
            timestamp,
        },
        HarnessV1BridgeDiagnosticFrame::DebugEvent(event) => HarnessV1Diagnostic {
            level: event.level,
            message: event.message.clone(),
            subsystem: event.subsystem.clone(),
            kind: HarnessV1DiagnosticKind::Event,
            source: None,
            stream: None,
            attrs: event.attrs.clone(),
            error: event.error.clone(),
            session_id,
            timestamp,
        },
    }
}

/// `harnessV1BridgeUserMessageInboundSchema`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1BridgeUserMessageInbound {
    pub r#type: UserMessageType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    pub text: String,
}

/// `experimental_harnessV1BridgeUserMessageInboundSchema`: `messageId` is
/// required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentalHarnessV1BridgeUserMessageInbound {
    pub r#type: UserMessageType,
    pub message_id: String,
    pub text: String,
}

/// `harnessV1BridgeInboundCommandSchemas`: the shared host -> bridge
/// commands. Adapters combine these with their own `start` payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum HarnessV1BridgeInboundCommand {
    /// `harnessV1BridgeToolResultInboundSchema`.
    ToolResult {
        tool_call_id: String,
        #[serde(default)]
        output: JsonValue,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_result: Option<JsonValue>,
    },
    /// `harnessV1BridgeToolApprovalResponseInboundSchema`.
    ToolApprovalResponse {
        approval_id: String,
        approved: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// `harnessV1BridgeUserMessageInboundSchema`.
    UserMessage {
        #[serde(skip_serializing_if = "Option::is_none")]
        message_id: Option<String>,
        text: String,
    },
    /// `harnessV1BridgeAbortInboundSchema`.
    Abort,
    /// `harnessV1BridgeDestroyInboundSchema`.
    Destroy,
    /// `harnessV1BridgeResumeInboundSchema`: replay events after
    /// `last_seen_event_id`.
    Resume { last_seen_event_id: u64 },
    /// `harnessV1BridgeStopInboundSchema`.
    Stop,
}

/// `harnessV1BridgeReadySchema`: the stdout line announcing the bridge port.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1BridgeReady {
    pub r#type: BridgeReadyType,
    pub port: u16,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip<T: Serialize + serde::de::DeserializeOwned>(value: JsonValue) -> T {
        let parsed: T = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), value);
        parsed
    }

    // harness-v1-bridge-protocol.test.ts
    #[test]
    fn preserves_start_identifiers_and_the_retry_capability() {
        round_trip::<HarnessV1BridgeStartBase>(
            json!({ "type": "start", "startId": "turn-1", "prompt": "Hello" }),
        );
        let hello = round_trip::<HarnessV1BridgeHello>(
            json!({ "type": "bridge-hello", "capabilities": { "idempotentStart": true } }),
        );
        assert_eq!(
            hello.capabilities.unwrap().idempotent_start,
            Some(Some(true))
        );
        round_trip::<HarnessV1BridgeHello>(
            json!({ "type": "bridge-hello", "capabilities": { "idempotentStart": null } }),
        );
    }

    #[test]
    fn preserves_json_schema_response_formats_across_the_bridge_boundary() {
        round_trip::<HarnessV1BridgeResponseFormat>(json!({
            "type": "json",
            "name": "answer",
            "description": "A structured answer.",
            "schema": {
                "type": "object",
                "properties": { "answer": { "type": "string", "enum": ["yes", "no"] } },
                "required": ["answer"],
                "additionalProperties": false
            }
        }));
    }

    #[test]
    fn accepts_the_original_user_message_payload_without_a_message_id() {
        round_trip::<HarnessV1BridgeUserMessageInbound>(
            json!({ "type": "user-message", "text": "/compact" }),
        );
        round_trip::<HarnessV1BridgeInboundCommand>(
            json!({ "type": "user-message", "text": "/compact" }),
        );
    }

    #[test]
    fn requires_a_message_id_on_the_experimental_acknowledged_payload() {
        assert!(
            serde_json::from_value::<ExperimentalHarnessV1BridgeUserMessageInbound>(
                json!({ "type": "user-message", "text": "Change course." })
            )
            .is_err()
        );
        round_trip::<ExperimentalHarnessV1BridgeUserMessageInbound>(json!({
            "type": "user-message", "messageId": "message-1", "text": "Change course."
        }));
    }

    #[test]
    fn rejects_the_wrong_literal_type() {
        assert!(
            serde_json::from_value::<HarnessV1BridgeStartBase>(
                json!({ "type": "stop", "prompt": "x" })
            )
            .is_err()
        );
    }

    #[test]
    fn outbound_frames_parse_into_the_right_variant() {
        let cases = [
            (
                json!({ "type": "text-delta", "id": "t", "delta": "x" }),
                "part",
            ),
            (
                json!({ "type": "bridge-hello", "state": "idle", "lastSeq": 3 }),
                "hello",
            ),
            (
                json!({ "type": "user-message-response", "messageId": "m", "accepted": false, "error": { "message": "busy" } }),
                "umr",
            ),
            (
                json!({ "type": "bridge-stop", "data": { "sessionId": "s" } }),
                "stop",
            ),
            (
                json!({ "type": "bridge-thread", "threadId": "th" }),
                "thread",
            ),
            (
                json!({ "type": "sandbox-log", "source": "bridge", "stream": "stderr", "line": "l" }),
                "log",
            ),
            (
                json!({ "type": "debug-event", "level": "debug", "subsystem": "a.b", "message": "m" }),
                "debug",
            ),
        ];
        for (value, kind) in cases {
            let parsed = HarnessV1BridgeOutboundMessage::parse(value.clone()).unwrap();
            let got = match &parsed {
                HarnessV1BridgeOutboundMessage::StreamPart(_) => "part",
                HarnessV1BridgeOutboundMessage::Hello(_) => "hello",
                HarnessV1BridgeOutboundMessage::UserMessageResponse(_) => "umr",
                HarnessV1BridgeOutboundMessage::Stop(_) => "stop",
                HarnessV1BridgeOutboundMessage::Thread(_) => "thread",
                HarnessV1BridgeOutboundMessage::SandboxLog(_) => "log",
                HarnessV1BridgeOutboundMessage::DebugEvent(_) => "debug",
            };
            assert_eq!(got, kind);
            assert_eq!(serde_json::to_value(&parsed).unwrap(), value);
        }
        assert!(HarnessV1BridgeOutboundMessage::parse(json!({ "type": "bogus" })).is_err());
    }

    #[test]
    fn inbound_commands_round_trip() {
        for value in [
            json!({ "type": "tool-result", "toolCallId": "c", "output": { "a": 1 }, "isError": true }),
            json!({ "type": "tool-approval-response", "approvalId": "a", "approved": true, "reason": "ok" }),
            json!({ "type": "abort" }),
            json!({ "type": "destroy" }),
            json!({ "type": "resume", "lastSeenEventId": 7 }),
            json!({ "type": "stop" }),
        ] {
            round_trip::<HarnessV1BridgeInboundCommand>(value);
        }
        round_trip::<HarnessV1BridgeReady>(json!({ "type": "bridge-ready", "port": 8123 }));
    }

    #[test]
    fn diagnostic_from_bridge_frame() {
        let log: HarnessV1BridgeSandboxLog = serde_json::from_value(
            json!({ "type": "sandbox-log", "source": "claude", "stream": "stderr", "line": "oops" }),
        )
        .unwrap();
        let d = harness_v1_diagnostic_from_bridge_frame(
            HarnessV1BridgeDiagnosticFrame::SandboxLog(&log),
            Some("s".into()),
            42,
        );
        assert_eq!(
            serde_json::to_value(d).unwrap(),
            json!({ "level": "warn", "message": "oops", "subsystem": "sandbox.log.claude",
                    "kind": "log", "source": "claude", "stream": "stderr", "sessionId": "s", "timestamp": 42 })
        );
        let event: HarnessV1BridgeDebugEvent = serde_json::from_value(json!({
            "type": "debug-event", "level": "error", "subsystem": "bridge.ws", "message": "m",
            "attrs": { "k": 1 }, "error": { "message": "e" }
        }))
        .unwrap();
        let d = harness_v1_diagnostic_from_bridge_frame(
            HarnessV1BridgeDiagnosticFrame::DebugEvent(&event),
            None,
            1,
        );
        assert_eq!(
            serde_json::to_value(d).unwrap(),
            json!({ "level": "error", "message": "m", "subsystem": "bridge.ws", "kind": "event",
                    "attrs": { "k": 1 }, "error": { "message": "e" }, "timestamp": 1 })
        );
    }
}
