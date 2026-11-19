//! [rust] Mirrors of the `@ai-sdk/provider` V4 types the harness spec reuses
//! verbatim (`LanguageModelV4Usage`, `LanguageModelV4FinishReason`, shared
//! metadata, options and file data, and the V4 prompt parts).
//!
//! Each type serializes to exactly the TS JSON shape.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// `JSONValue`.
pub type JsonValue = Value;
/// `JSONObject`.
pub type JsonObject = Map<String, Value>;

/// `SharedV4ProviderMetadata`: `Record<string, Record<string, JSONValue>>`.
pub type SharedV4ProviderMetadata = indexmap::IndexMap<String, JsonObject>;
/// `SharedV4ProviderOptions`: `Record<string, JSONObject>`.
pub type SharedV4ProviderOptions = indexmap::IndexMap<String, JsonObject>;
/// `SharedV4ProviderReference`: `Record<string, string>`.
pub type SharedV4ProviderReference = indexmap::IndexMap<String, String>;

/// `LanguageModelV4Usage.inputTokens`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelV4InputTokens {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_cache: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<u64>,
}

/// `LanguageModelV4Usage.outputTokens`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelV4OutputTokens {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
}

/// `LanguageModelV4Usage`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelV4Usage {
    pub input_tokens: LanguageModelV4InputTokens,
    pub output_tokens: LanguageModelV4OutputTokens,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<JsonObject>,
}

/// `LanguageModelV4FinishReason['unified']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LanguageModelV4UnifiedFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Error,
    Interrupted,
    Other,
}

/// `LanguageModelV4FinishReason`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageModelV4FinishReason {
    pub unified: LanguageModelV4UnifiedFinishReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

impl LanguageModelV4FinishReason {
    pub fn new(unified: LanguageModelV4UnifiedFinishReason) -> Self {
        Self { unified, raw: None }
    }
}

/// [rust] `Uint8Array | string` (and `DataContent`). Bytes serialize as a
/// base64 string, which is how the AI SDK puts binary data on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataContent {
    /// A string: base64 data (or text for `FileDataText`).
    String(String),
    /// Raw bytes.
    Bytes(Vec<u8>),
}

impl Serialize for DataContent {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            DataContent::String(v) => s.serialize_str(v),
            DataContent::Bytes(b) => {
                use base64::Engine as _;
                s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(b))
            }
        }
    }
}

impl<'de> Deserialize<'de> for DataContent {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            S(String),
            B(Vec<u8>),
        }
        Ok(match Repr::deserialize(d)? {
            Repr::S(s) => DataContent::String(s),
            Repr::B(b) => DataContent::Bytes(b),
        })
    }
}

/// `SharedV4FileData` (`data` | `url` | `reference` | `text`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum SharedV4FileData {
    Data {
        data: DataContent,
    },
    /// [rust] `URL` is carried as its string form.
    Url {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        original_url: Option<String>,
    },
    Reference {
        reference: SharedV4ProviderReference,
    },
    Text {
        text: String,
    },
}

/// `SharedV4FileDataData | SharedV4FileDataUrl` (reasoning files).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum SharedV4ReasoningFileData {
    Data {
        data: DataContent,
    },
    Url {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        original_url: Option<String>,
    },
}

/// One part of a `content` tool result output (`LanguageModelV4ToolResultOutput`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum LanguageModelV4ToolResultContentPart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<SharedV4ProviderOptions>,
    },
    File {
        data: SharedV4FileData,
        media_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<SharedV4ProviderOptions>,
    },
    Custom {
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_options: Option<SharedV4ProviderOptions>,
    },
}

/// `TypeValidationError` from `@ai-sdk/provider`: a value failed a schema.
///
/// [rust] Raised where TS runs a zod `parse` (stream parts, bridge frames,
/// question tool input and output).
#[derive(Debug, Clone, thiserror::Error)]
#[error("Type validation failed: Value: {value}.\nError message: {message}")]
pub struct TypeValidationError {
    pub value: JsonValue,
    pub message: String,
}

impl TypeValidationError {
    pub fn new(value: JsonValue, message: impl Into<String>) -> Self {
        Self {
            value,
            message: message.into(),
        }
    }
}

/// Parse a JSON value into `T`, the way a zod `parse` does: unknown keys are
/// stripped, a mismatch is a [`TypeValidationError`].
pub fn parse_json<T: serde::de::DeserializeOwned>(
    value: JsonValue,
) -> Result<T, TypeValidationError> {
    serde_json::from_value::<T>(value.clone())
        .map_err(|e| TypeValidationError::new(value, e.to_string()))
}
