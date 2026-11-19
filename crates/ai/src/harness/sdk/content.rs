//! Mirrors of the `ai` generate-text content types the harness agent builds:
//! `TypedToolCall`, `TypedToolResult`, `TypedToolError`, the approval
//! outputs, `ContentPart`, `StepResult` and its metadata.
//!
//! [rust] The harness has no typed tool sets, so the static / dynamic split
//! is the `dynamic` flag on one struct. `ContentPart` carries the variants
//! a harness turn produces (it never emits sources, files or custom parts).

use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::harness::provider::{
    JsonObject, JsonValue, LanguageModelV4UnifiedFinishReason, SharedV4ProviderMetadata,
};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::messages::{ModelMessage, ReasoningPart};
use crate::harness::sdk::usage::LanguageModelUsage;
use crate::harness::spec::stream_part::HarnessV1CallWarning;

/// `ProviderMetadata`.
pub type ProviderMetadata = SharedV4ProviderMetadata;
/// `FinishReason`.
pub type FinishReason = LanguageModelV4UnifiedFinishReason;
/// `CallWarning` (the harness passes `HarnessV1CallWarning`s through).
pub type CallWarning = HarnessV1CallWarning;
/// `Context`: runtime context objects are JSON objects.
pub type Context = JsonObject;
/// `InferToolSetContext<TOOLS>`: per-tool context keyed by tool name.
pub type ToolsContext = JsonObject;

/// `TypedToolCall` (static or dynamic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TypedToolCall {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_metadata: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invalid: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The parsed JSON the input schema received, before its transforms
    /// (`setToolCallInputSchemaInput`). TS keeps it under a non-enumerable
    /// symbol, so it is neither serialized nor compared.
    #[serde(skip)]
    pub input_schema_input: InputSchemaInput,
}

/// The hidden `inputSchemaInput` of a [`TypedToolCall`]. Equality ignores
/// it, as `toEqual` ignores symbol keys.
#[derive(Debug, Clone, Default)]
pub struct InputSchemaInput(pub Option<JsonValue>);

impl PartialEq for InputSchemaInput {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl TypedToolCall {
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: JsonValue,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input,
            provider_executed: None,
            provider_metadata: None,
            tool_metadata: None,
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            input_schema_input: InputSchemaInput::default(),
        }
    }

    /// `getToolCallInputSchemaInput(toolCall)`.
    pub fn input_schema_input(&self) -> Option<&JsonValue> {
        self.input_schema_input.0.as_ref()
    }

    /// `toolCall.dynamic === true`.
    pub fn is_dynamic(&self) -> bool {
        self.dynamic == Some(true)
    }

    /// `toolCall.invalid === true`.
    pub fn is_invalid(&self) -> bool {
        self.invalid == Some(true)
    }

    /// `{ ...toolCall, type: 'tool-result', output }`.
    pub fn to_result(&self, output: JsonValue) -> TypedToolResult {
        TypedToolResult {
            tool_call_id: self.tool_call_id.clone(),
            tool_name: self.tool_name.clone(),
            input: Some(self.input.clone()),
            output,
            provider_executed: self.provider_executed,
            provider_metadata: self.provider_metadata.clone(),
            tool_metadata: self.tool_metadata.clone(),
            dynamic: self.dynamic,
            preliminary: None,
            title: self.title.clone(),
        }
    }

    /// `{ ...toolCall, type: 'tool-error', error }`.
    pub fn to_error(&self, error: ErrorValue) -> TypedToolError {
        TypedToolError {
            tool_call_id: self.tool_call_id.clone(),
            tool_name: self.tool_name.clone(),
            input: Some(self.input.clone()),
            error,
            provider_executed: self.provider_executed,
            provider_metadata: self.provider_metadata.clone(),
            tool_metadata: self.tool_metadata.clone(),
            dynamic: self.dynamic,
            title: self.title.clone(),
        }
    }
}

/// `TypedToolResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TypedToolResult {
    pub tool_call_id: String,
    pub tool_name: String,
    /// `input: unknown` (`undefined` on preliminary results).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<JsonValue>,
    pub output: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_metadata: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl TypedToolResult {
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: Option<JsonValue>,
        output: JsonValue,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input,
            output,
            provider_executed: None,
            provider_metadata: None,
            tool_metadata: None,
            dynamic: None,
            preliminary: None,
            title: None,
        }
    }
}

/// `TypedToolError`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TypedToolError {
    pub tool_call_id: String,
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<JsonValue>,
    pub error: ErrorValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_metadata: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl TypedToolError {
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: Option<JsonValue>,
        error: ErrorValue,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            input,
            error,
            provider_executed: None,
            provider_metadata: None,
            tool_metadata: None,
            dynamic: None,
            title: None,
        }
    }
}

/// `TypedToolResult<TOOLS> | TypedToolError<TOOLS>`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolOutput {
    ToolResult(TypedToolResult),
    ToolError(TypedToolError),
}

impl ToolOutput {
    pub fn tool_call_id(&self) -> &str {
        match self {
            Self::ToolResult(r) => &r.tool_call_id,
            Self::ToolError(e) => &e.tool_call_id,
        }
    }

    /// The output as a content part.
    pub fn into_content_part(self) -> ContentPart {
        match self {
            Self::ToolResult(r) => ContentPart::ToolResult(r),
            Self::ToolError(e) => ContentPart::ToolError(e),
        }
    }
}

/// `ToolApprovalRequestOutput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolApprovalRequestOutput {
    pub approval_id: String,
    pub tool_call: TypedToolCall,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_automatic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// `ToolApprovalResponseOutput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolApprovalResponseOutput {
    pub approval_id: String,
    pub tool_call: TypedToolCall,
    pub approved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
}

/// `{ type: 'text' }` content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    /// `citations?: Citation[]` (sources, passed through as JSON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<JsonValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

/// `ReasoningOutput`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningOutput {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

/// `ContentPart<TOOLS>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ContentPart {
    Text(TextContent),
    Reasoning(ReasoningOutput),
    ToolCall(TypedToolCall),
    ToolResult(TypedToolResult),
    ToolError(TypedToolError),
    ToolApprovalRequest(ToolApprovalRequestOutput),
    ToolApprovalResponse(ToolApprovalResponseOutput),
}

impl ContentPart {
    /// `part.type`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::Reasoning(_) => "reasoning",
            Self::ToolCall(_) => "tool-call",
            Self::ToolResult(_) => "tool-result",
            Self::ToolError(_) => "tool-error",
            Self::ToolApprovalRequest(_) => "tool-approval-request",
            Self::ToolApprovalResponse(_) => "tool-approval-response",
        }
    }
}

/// `LanguageModelRequestMetadata`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelRequestMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<ModelMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JsonValue>,
}

/// `LanguageModelResponseMetadata`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelResponseMetadata {
    pub id: String,
    #[serde(with = "iso_timestamp")]
    pub timestamp: SystemTime,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<indexmap::IndexMap<String, String>>,
    /// `messages: Array<ResponseMessage>`. Omitted on `finish-step` parts.
    #[serde(default)]
    pub messages: Vec<ModelMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JsonValue>,
}

/// `OutputChunkTimingStats`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputChunkTimingStats {
    pub min: f64,
    pub p10: f64,
    pub median: f64,
    pub avg: f64,
    pub p90: f64,
    pub max: f64,
}

/// `StepResultPerformance`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepResultPerformance {
    pub effective_output_tokens_per_second: f64,
    pub output_tokens_per_second: Option<f64>,
    pub input_tokens_per_second: Option<f64>,
    pub effective_total_tokens_per_second: f64,
    pub step_time_ms: f64,
    pub response_time_ms: f64,
    pub tool_execution_ms: indexmap::IndexMap<String, f64>,
    pub time_to_first_output_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_between_output_chunks_ms: Option<OutputChunkTimingStats>,
}

/// `StepResult['model']`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepModel {
    pub provider: String,
    pub model_id: String,
}

/// `StepResult` / `DefaultStepResult`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StepResult {
    pub call_id: String,
    pub step_number: u32,
    pub model: StepModel,
    pub tools_context: ToolsContext,
    pub runtime_context: Context,
    pub content: Vec<ContentPart>,
    pub finish_reason: FinishReason,
    pub raw_finish_reason: Option<String>,
    pub usage: LanguageModelUsage,
    pub performance: StepResultPerformance,
    pub warnings: Option<Vec<CallWarning>>,
    pub request: LanguageModelRequestMetadata,
    pub response: LanguageModelResponseMetadata,
    pub provider_metadata: Option<ProviderMetadata>,
}

impl StepResult {
    /// `text`.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// `reasoning` (`convertFromReasoningOutputs`).
    pub fn reasoning(&self) -> Vec<ReasoningPart> {
        self.content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Reasoning(r) => Some(ReasoningPart {
                    text: r.text.clone(),
                    provider_options: r.provider_metadata.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    /// `reasoningText` (`asReasoningText`).
    pub fn reasoning_text(&self) -> Option<String> {
        let text: String = self.reasoning().iter().map(|r| r.text.as_str()).collect();
        (!text.is_empty()).then_some(text)
    }

    /// `toolCalls`.
    pub fn tool_calls(&self) -> Vec<TypedToolCall> {
        self.content
            .iter()
            .filter_map(|p| match p {
                ContentPart::ToolCall(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }

    /// `staticToolCalls`.
    pub fn static_tool_calls(&self) -> Vec<TypedToolCall> {
        self.tool_calls()
            .into_iter()
            .filter(|c| !c.is_dynamic())
            .collect()
    }

    /// `dynamicToolCalls`.
    pub fn dynamic_tool_calls(&self) -> Vec<TypedToolCall> {
        self.tool_calls()
            .into_iter()
            .filter(|c| c.is_dynamic())
            .collect()
    }

    /// `toolResults`.
    pub fn tool_results(&self) -> Vec<TypedToolResult> {
        self.content
            .iter()
            .filter_map(|p| match p {
                ContentPart::ToolResult(r) => Some(r.clone()),
                _ => None,
            })
            .collect()
    }

    /// `staticToolResults`.
    pub fn static_tool_results(&self) -> Vec<TypedToolResult> {
        self.tool_results()
            .into_iter()
            .filter(|r| r.dynamic != Some(true))
            .collect()
    }

    /// `dynamicToolResults`.
    pub fn dynamic_tool_results(&self) -> Vec<TypedToolResult> {
        self.tool_results()
            .into_iter()
            .filter(|r| r.dynamic == Some(true))
            .collect()
    }
}

/// ISO-8601 (`Date.prototype.toISOString`) serialization of `SystemTime`.
pub mod iso_timestamp {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde::{Deserialize, Deserializer, Serializer};

    /// `date.toISOString()`.
    pub fn format(time: &SystemTime) -> String {
        let millis = match time.duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_millis() as i64,
            Err(e) => -(e.duration().as_millis() as i64),
        };
        let days = millis.div_euclid(86_400_000);
        let ms_of_day = millis.rem_euclid(86_400_000);
        let (y, m, d) = civil_from_days(days);
        format!(
            "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z",
            ms_of_day / 3_600_000,
            (ms_of_day / 60_000) % 60,
            (ms_of_day / 1000) % 60,
            ms_of_day % 1000
        )
    }

    /// Parse an ISO-8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SS(.mmm)Z`).
    pub fn parse(text: &str) -> Option<SystemTime> {
        let text = text.strip_suffix('Z')?;
        let (date, time) = text.split_once('T')?;
        let mut dp = date.split('-');
        let y: i64 = dp.next()?.parse().ok()?;
        let m: u32 = dp.next()?.parse().ok()?;
        let d: u32 = dp.next()?.parse().ok()?;
        let (hms, frac) = time.split_once('.').unwrap_or((time, "0"));
        let mut tp = hms.split(':');
        let hh: i64 = tp.next()?.parse().ok()?;
        let mm: i64 = tp.next()?.parse().ok()?;
        let ss: i64 = tp.next()?.parse().ok()?;
        let ms: i64 = format!("{frac:0<3}")[..3].parse().ok()?;
        let days = days_from_civil(y, m, d);
        let millis = days * 86_400_000 + hh * 3_600_000 + mm * 60_000 + ss * 1000 + ms;
        Some(if millis >= 0 {
            UNIX_EPOCH + Duration::from_millis(millis as u64)
        } else {
            UNIX_EPOCH - Duration::from_millis((-millis) as u64)
        })
    }

    fn civil_from_days(z: i64) -> (i64, u32, u32) {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        (if m <= 2 { y + 1 } else { y }, m, d)
    }

    fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
        let y = if m <= 2 { y - 1 } else { y };
        let era = y.div_euclid(400);
        let yoe = y.rem_euclid(400);
        let m = m as i64;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    pub fn serialize<S: Serializer>(time: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format(time))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let text = String::deserialize(d)?;
        parse(&text).ok_or_else(|| serde::de::Error::custom("invalid ISO timestamp"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn iso_timestamps_round_trip() {
        let t = UNIX_EPOCH + Duration::from_millis(1_700_000_000_123);
        assert_eq!(iso_timestamp::format(&t), "2023-11-14T22:13:20.123Z");
        assert_eq!(iso_timestamp::parse("2023-11-14T22:13:20.123Z"), Some(t));
        assert_eq!(
            iso_timestamp::format(&UNIX_EPOCH),
            "1970-01-01T00:00:00.000Z"
        );
    }

    #[test]
    fn content_parts_serialize_with_type_tags() {
        let part = ContentPart::ToolCall(TypedToolCall::new("c1", "t", json!({"a": 1})));
        assert_eq!(
            serde_json::to_value(&part).unwrap(),
            json!({"type": "tool-call", "toolCallId": "c1", "toolName": "t", "input": {"a": 1}})
        );
        let part = ContentPart::Text(TextContent {
            text: "hi".into(),
            ..Default::default()
        });
        assert_eq!(
            serde_json::to_value(&part).unwrap(),
            json!({"type": "text", "text": "hi"})
        );
    }

    fn step(content: Vec<ContentPart>) -> StepResult {
        StepResult {
            call_id: "call".into(),
            step_number: 0,
            model: StepModel {
                provider: "harness:test".into(),
                model_id: "m".into(),
            },
            tools_context: Default::default(),
            runtime_context: Default::default(),
            content,
            finish_reason: FinishReason::Stop,
            raw_finish_reason: None,
            usage: Default::default(),
            performance: Default::default(),
            warnings: None,
            request: Default::default(),
            response: LanguageModelResponseMetadata {
                id: "call-0".into(),
                timestamp: UNIX_EPOCH,
                model_id: "m".into(),
                headers: None,
                messages: vec![],
                body: None,
            },
            provider_metadata: None,
        }
    }

    #[test]
    fn step_result_getters_filter_content() {
        let mut dynamic_call = TypedToolCall::new("c2", "d", json!({}));
        dynamic_call.dynamic = Some(true);
        let s = step(vec![
            ContentPart::Text(TextContent {
                text: "a".into(),
                ..Default::default()
            }),
            ContentPart::Reasoning(ReasoningOutput {
                text: "r".into(),
                provider_metadata: None,
            }),
            ContentPart::Text(TextContent {
                text: "b".into(),
                ..Default::default()
            }),
            ContentPart::ToolCall(TypedToolCall::new("c1", "s", json!({}))),
            ContentPart::ToolCall(dynamic_call),
        ]);
        assert_eq!(s.text(), "ab");
        assert_eq!(s.reasoning_text().as_deref(), Some("r"));
        assert_eq!(s.tool_calls().len(), 2);
        assert_eq!(s.static_tool_calls()[0].tool_call_id, "c1");
        assert_eq!(s.dynamic_tool_calls()[0].tool_call_id, "c2");
        assert_eq!(step(vec![]).reasoning_text(), None);
    }
}
