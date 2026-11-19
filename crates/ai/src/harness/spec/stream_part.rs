//! Port of `harness-v1-stream-part.ts`, `harness-v1-call-warning.ts` and
//! `harness-v1-metadata.ts`.
//!
//! The zod schemas (`harnessV1StreamPartSchema` and the per-part schemas)
//! are the serde impls; [`HarnessV1StreamPart::parse`] is
//! `harnessV1StreamPartSchema.parse`.

use serde::{Deserialize, Serialize};

use crate::harness::sdk::error_value::ErrorValue;

use crate::harness::provider::{
    JsonObject, JsonValue, LanguageModelV4FinishReason, LanguageModelV4Usage,
    SharedV4ProviderMetadata, TypeValidationError, parse_json,
};

/// `HarnessV1Metadata`: adapter-namespaced opaque data, keyed by harness id.
pub type HarnessV1Metadata = indexmap::IndexMap<String, JsonObject>;

/// `HarnessV1CallWarning`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum HarnessV1CallWarning {
    UnsupportedSetting {
        setting: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    UnsupportedTool {
        tool: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    Other {
        message: String,
    },
}

/// `file-change` `event`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HarnessV1FileChangeEvent {
    Create,
    Modify,
    Delete,
}

/// `compaction` `trigger`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HarnessV1CompactionTrigger {
    Manual,
    Auto,
}

/// One event emitted by a harness adapter during a prompt turn.
///
/// Mirrors `LanguageModelV4StreamPart` on the variants it shares. The
/// metadata field is named `harnessMetadata` because a harness is a peer to
/// a provider, not a kind of provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum HarnessV1StreamPart {
    StreamStart {
        #[serde(skip_serializing_if = "Option::is_none")]
        warnings: Option<Vec<HarnessV1CallWarning>>,
        /// The model the runtime actually resolved to for this turn.
        #[serde(skip_serializing_if = "Option::is_none")]
        model_id: Option<String>,
    },
    TextStart {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
    },
    TextDelta {
        id: String,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
    },
    TextEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
    },
    ReasoningStart {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
    },
    ReasoningDelta {
        id: String,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
    },
    ReasoningEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
    },
    ToolInputStart {
        id: String,
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedV4ProviderMetadata>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    ToolInputDelta {
        id: String,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    ToolInputEnd {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    /// `LanguageModelV4ToolCall & { nativeName? }`. `input` is JSON text.
    ToolCall {
        tool_call_id: String,
        tool_name: String,
        input: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedV4ProviderMetadata>,
        #[serde(skip_serializing_if = "Option::is_none")]
        native_name: Option<String>,
    },
    /// `LanguageModelV4ToolApprovalRequest`.
    ToolApprovalRequest {
        approval_id: String,
        tool_call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    /// `LanguageModelV4ToolResult`. The type is non-null JSON; the runtime
    /// schema also accepts `null` for tools that produced no output.
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        result: JsonValue,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        preliminary: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<SharedV4ProviderMetadata>,
    },
    /// Step boundary inside a multi-step turn. Also emitted when every
    /// unsettled tool call of the step waits for host input.
    FinishStep {
        finish_reason: LanguageModelV4FinishReason,
        usage: LanguageModelV4Usage,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
        /// [rust] The non-enumerable event checkpoint a sandbox channel
        /// attaches to a dispatched event (`pinSandboxChannelEventCheckpoint`).
        /// Never serialized, and equal to every other checkpoint.
        #[serde(skip)]
        checkpoint: Option<crate::harness::utils::sandbox_channel::SandboxChannelEventCheckpoint>,
    },
    /// Turn end.
    Finish {
        finish_reason: LanguageModelV4FinishReason,
        total_usage: LanguageModelV4Usage,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
    },
    /// Workspace file mutation through an opaque mechanism. Path only.
    FileChange {
        event: HarnessV1FileChangeEvent,
        path: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
    },
    /// Context compaction performed by the underlying runtime.
    Compaction {
        trigger: HarnessV1CompactionTrigger,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens_before: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens_after: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        harness_metadata: Option<HarnessV1Metadata>,
    },
    /// Errors. Multiple may be emitted in a single turn. `error: unknown`.
    ///
    /// [rust] An [`ErrorValue`]: adapters can emit an `Error` instance, and
    /// parsed JSON becomes [`ErrorValue::Value`].
    Error {
        #[serde(default = "null_error")]
        error: ErrorValue,
    },
    /// Adapter-specific passthrough. `rawValue: unknown`.
    Raw {
        #[serde(default)]
        raw_value: JsonValue,
    },
}

fn null_error() -> ErrorValue {
    ErrorValue::Value(JsonValue::Null)
}

impl HarnessV1StreamPart {
    /// `harnessV1StreamPartSchema.parse(value)`.
    pub fn parse(value: JsonValue) -> Result<Self, TypeValidationError> {
        parse_json(value)
    }

    /// The `type` discriminator.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::StreamStart { .. } => "stream-start",
            Self::TextStart { .. } => "text-start",
            Self::TextDelta { .. } => "text-delta",
            Self::TextEnd { .. } => "text-end",
            Self::ReasoningStart { .. } => "reasoning-start",
            Self::ReasoningDelta { .. } => "reasoning-delta",
            Self::ReasoningEnd { .. } => "reasoning-end",
            Self::ToolInputStart { .. } => "tool-input-start",
            Self::ToolInputDelta { .. } => "tool-input-delta",
            Self::ToolInputEnd { .. } => "tool-input-end",
            Self::ToolCall { .. } => "tool-call",
            Self::ToolApprovalRequest { .. } => "tool-approval-request",
            Self::ToolResult { .. } => "tool-result",
            Self::FinishStep { .. } => "finish-step",
            Self::Finish { .. } => "finish",
            Self::FileChange { .. } => "file-change",
            Self::Compaction { .. } => "compaction",
            Self::Error { .. } => "error",
            Self::Raw { .. } => "raw",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // harness-v1-stream-part.test.ts
    #[test]
    fn accepts_an_interrupted_finish_step_and_finish() {
        for (ty, usage_key) in [("finish-step", "usage"), ("finish", "totalUsage")] {
            // `raw: undefined` in TS is an absent key in JSON.
            let part = json!({
                "type": ty,
                "finishReason": { "unified": "interrupted" },
                usage_key: { "inputTokens": {}, "outputTokens": {} },
            });
            let parsed = HarnessV1StreamPart::parse(part.clone()).unwrap();
            assert_eq!(serde_json::to_value(&parsed).unwrap(), part);
            assert_eq!(parsed.type_name(), ty);
        }
    }

    #[test]
    fn every_variant_round_trips() {
        let parts = [
            json!({ "type": "stream-start", "warnings": [
                { "type": "unsupported-setting", "setting": "temperature" },
                { "type": "unsupported-tool", "tool": "x", "details": "d" },
                { "type": "other", "message": "m" }
            ], "modelId": "m1" }),
            json!({ "type": "text-start", "id": "t", "harnessMetadata": { "claude-code": { "a": 1 } } }),
            json!({ "type": "text-delta", "id": "t", "delta": "hi" }),
            json!({ "type": "text-end", "id": "t" }),
            json!({ "type": "reasoning-start", "id": "r" }),
            json!({ "type": "reasoning-delta", "id": "r", "delta": "hm" }),
            json!({ "type": "reasoning-end", "id": "r" }),
            json!({ "type": "tool-input-start", "id": "c", "toolName": "bash", "providerExecuted": true, "dynamic": false, "title": "Bash" }),
            json!({ "type": "tool-input-delta", "id": "c", "delta": "{" }),
            json!({ "type": "tool-input-end", "id": "c" }),
            json!({ "type": "tool-call", "toolCallId": "c", "toolName": "bash", "input": "{\"command\":\"ls\"}", "providerExecuted": true, "nativeName": "Bash" }),
            json!({ "type": "tool-approval-request", "approvalId": "a", "toolCallId": "c" }),
            json!({ "type": "tool-result", "toolCallId": "c", "toolName": "bash", "result": { "ok": true }, "isError": false, "preliminary": true }),
            json!({ "type": "tool-result", "toolCallId": "c", "toolName": "bash", "result": null }),
            json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "tool_use" },
                    "usage": { "inputTokens": { "total": 3, "noCache": 1, "cacheRead": 2, "cacheWrite": 0 },
                               "outputTokens": { "total": 4, "text": 3, "reasoning": 1 }, "raw": { "x": 1 } } }),
            json!({ "type": "finish", "finishReason": { "unified": "stop" }, "totalUsage": { "inputTokens": {}, "outputTokens": {} } }),
            json!({ "type": "file-change", "event": "modify", "path": "a.txt" }),
            json!({ "type": "compaction", "trigger": "auto", "summary": "s", "tokensBefore": 10, "tokensAfter": 2 }),
            json!({ "type": "error", "error": { "message": "boom" } }),
            json!({ "type": "raw", "rawValue": [1, 2] }),
        ];
        for part in parts {
            let parsed = HarnessV1StreamPart::parse(part.clone()).unwrap();
            assert_eq!(serde_json::to_value(&parsed).unwrap(), part);
        }
    }

    #[test]
    fn rejects_unknown_types_and_bad_shapes() {
        assert!(HarnessV1StreamPart::parse(json!({ "type": "nope" })).is_err());
        assert!(HarnessV1StreamPart::parse(json!({ "type": "text-delta", "id": "t" })).is_err());
        assert!(
            HarnessV1StreamPart::parse(json!({
                "type": "finish", "finishReason": { "unified": "weird" },
                "totalUsage": { "inputTokens": {}, "outputTokens": {} }
            }))
            .is_err()
        );
    }
}
