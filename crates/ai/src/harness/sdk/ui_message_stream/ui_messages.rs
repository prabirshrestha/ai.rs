//! Mirror of `ai` `ui/ui-messages.ts`: `UIMessage` and its parts.
//!
//! [rust] Metadata and data are JSON (`METADATA`, `DATA_PARTS` and `TOOLS`
//! are not generic). The parts a UI message stream builds (text, reasoning,
//! tool and step-start parts) are typed; any other part (files, sources,
//! data parts, custom and steering parts) is kept as JSON in
//! [`UIMessagePart::Other`] so original messages round-trip unchanged.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::harness::provider::{JsonObject, JsonValue};
use crate::harness::sdk::content::ProviderMetadata;

/// `UIMessage['role']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UIMessageRole {
    System,
    User,
    Assistant,
}

/// `UIMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UIMessage {
    pub id: String,
    pub role: UIMessageRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonValue>,
    pub parts: Vec<UIMessagePart>,
}

/// `state?: 'streaming' | 'done'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UIPartState {
    Streaming,
    Done,
}

/// `TextUIPart`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextUIPart {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<JsonValue>>,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<UIPartState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

/// `ReasoningUIPart`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningUIPart {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<UIPartState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}

/// `UIToolInvocation['state']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum UIToolState {
    InputStreaming,
    InputAvailable,
    ApprovalRequested,
    ApprovalResponded,
    OutputAvailable,
    OutputError,
    OutputDenied,
}

/// `inputEncoding?: 'json' | 'text'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputEncoding {
    Json,
    Text,
}

/// `UIToolInvocation['approval']`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UIToolApproval {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descriptor: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_automatic: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_schema_input: Option<JsonValue>,
}

/// `ToolUIPart` (`type: 'tool-${name}'`) and `DynamicToolUIPart`
/// (`type: 'dynamic-tool'`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UIToolPart {
    /// `true` for a `dynamic-tool` part.
    #[serde(skip)]
    pub dynamic: bool,
    /// The tool name (`getToolName(part)`).
    #[serde(skip)]
    pub tool_name: String,
    pub tool_call_id: String,
    pub state: UIToolState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_metadata: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub input: Option<JsonValue>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub raw_input: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_encoding: Option<InputEncoding>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub output: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_provider_metadata: Option<ProviderMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_provider_metadata: Option<ProviderMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval: Option<UIToolApproval>,
}

impl UIToolPart {
    /// An empty part in `state` (the other fields unset).
    pub fn new(
        dynamic: bool,
        tool_name: impl Into<String>,
        tool_call_id: impl Into<String>,
        state: UIToolState,
    ) -> Self {
        Self {
            dynamic,
            tool_name: tool_name.into(),
            tool_call_id: tool_call_id.into(),
            state,
            title: None,
            tool_metadata: None,
            provider_executed: None,
            input: None,
            raw_input: None,
            input_encoding: None,
            output: None,
            error_text: None,
            preliminary: None,
            call_provider_metadata: None,
            result_provider_metadata: None,
            approval: None,
        }
    }

    /// `part.type`: `tool-${name}` or `dynamic-tool`.
    pub fn part_type(&self) -> String {
        if self.dynamic {
            "dynamic-tool".into()
        } else {
            format!("tool-{}", self.tool_name)
        }
    }
}

/// `UIMessagePart`.
#[derive(Debug, Clone, PartialEq)]
pub enum UIMessagePart {
    Text(TextUIPart),
    Reasoning(ReasoningUIPart),
    /// `ToolUIPart | DynamicToolUIPart`.
    Tool(UIToolPart),
    /// `StepStartUIPart`.
    StepStart,
    /// Any other part, as JSON (it has a `type`).
    Other(JsonValue),
}

impl UIMessagePart {
    /// `part.type`.
    pub fn part_type(&self) -> String {
        match self {
            Self::Text(_) => "text".into(),
            Self::Reasoning(_) => "reasoning".into(),
            Self::Tool(tool) => tool.part_type(),
            Self::StepStart => "step-start".into(),
            Self::Other(value) => value["type"].as_str().unwrap_or_default().to_string(),
        }
    }

    /// `isToolUIPart(part)`.
    pub fn as_tool(&self) -> Option<&UIToolPart> {
        match self {
            Self::Tool(tool) => Some(tool),
            _ => None,
        }
    }

    pub fn as_tool_mut(&mut self) -> Option<&mut UIToolPart> {
        match self {
            Self::Tool(tool) => Some(tool),
            _ => None,
        }
    }

    /// `isStaticToolUIPart(part)`.
    pub fn is_static_tool(&self) -> bool {
        matches!(self, Self::Tool(tool) if !tool.dynamic)
    }
}

impl Serialize for UIMessagePart {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let tagged = |value: Result<JsonValue, serde_json::Error>, ty: String| {
            let mut value = value.map_err(serde::ser::Error::custom)?;
            if let JsonValue::Object(map) = &mut value {
                map.insert("type".into(), ty.into());
            }
            Ok::<_, S::Error>(value)
        };
        let value = match self {
            Self::Text(part) => tagged(serde_json::to_value(part), "text".into())?,
            Self::Reasoning(part) => tagged(serde_json::to_value(part), "reasoning".into())?,
            Self::Tool(part) => {
                let mut value = tagged(serde_json::to_value(part), part.part_type())?;
                if part.dynamic {
                    value["toolName"] = part.tool_name.clone().into();
                }
                value
            }
            Self::StepStart => serde_json::json!({ "type": "step-start" }),
            Self::Other(value) => value.clone(),
        };
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UIMessagePart {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = JsonValue::deserialize(deserializer)?;
        let ty = value
            .get("type")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| D::Error::custom("UI message part without a type"))?
            .to_string();
        Ok(match ty.as_str() {
            "text" => Self::Text(parse::<_, D>(value)?),
            "reasoning" => Self::Reasoning(parse::<_, D>(value)?),
            "step-start" => Self::StepStart,
            "dynamic-tool" => {
                let tool_name = value
                    .get("toolName")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default()
                    .to_string();
                let mut part: UIToolPart = parse::<_, D>(value)?;
                part.dynamic = true;
                part.tool_name = tool_name;
                Self::Tool(part)
            }
            _ if ty.starts_with("tool-") => {
                let mut part: UIToolPart = parse::<_, D>(value)?;
                // `getStaticToolName`: the type after `tool-`.
                part.tool_name = ty["tool-".len()..].to_string();
                Self::Tool(part)
            }
            _ => Self::Other(value),
        })
    }
}

fn parse<'de, T: serde::de::DeserializeOwned, D: Deserializer<'de>>(
    value: JsonValue,
) -> Result<T, D::Error> {
    serde_json::from_value(value).map_err(D::Error::custom)
}

/// Deserialize a present key as `Some`, even when it is `null` (TS keeps
/// `output: null` apart from a missing output).
fn present<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<JsonValue>, D::Error> {
    JsonValue::deserialize(deserializer).map(Some)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn tool_parts_round_trip_with_their_type() {
        let parts = json!([
            { "type": "tool-weather", "toolCallId": "c1", "state": "output-available", "input": { "city": "SF" }, "output": null },
            { "type": "dynamic-tool", "toolName": "bash", "toolCallId": "c2", "state": "input-streaming" },
            { "type": "step-start" },
            { "type": "data-weather", "id": "d1", "data": { "temp": 1 } },
        ]);
        let parsed: Vec<UIMessagePart> = serde_json::from_value(parts.clone()).unwrap();
        assert_eq!(parsed[0].as_tool().unwrap().tool_name, "weather");
        assert_eq!(parsed[0].as_tool().unwrap().output, Some(JsonValue::Null));
        assert!(parsed[1].as_tool().unwrap().dynamic);
        assert_eq!(serde_json::to_value(&parsed).unwrap(), parts);
    }
}
