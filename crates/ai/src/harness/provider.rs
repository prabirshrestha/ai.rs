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

/// `TypeValidationContext`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TypeValidationContext {
    pub field: Option<String>,
    pub entity_name: Option<String>,
    pub entity_id: Option<String>,
}

/// `TypeValidationError` from `@ai-sdk/provider`: a value failed a schema.
///
/// [rust] Raised where TS runs a zod `parse` (stream parts, bridge frames,
/// question tool input and output) and by [`validate_types`]. `message` is
/// the cause's `getErrorMessage` text.
#[derive(Debug, Clone)]
pub struct TypeValidationError {
    pub value: JsonValue,
    pub message: String,
    pub context: Option<Box<TypeValidationContext>>,
}

impl std::fmt::Display for TypeValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut prefix = String::from("Type validation failed");
        if let Some(context) = &self.context {
            if let Some(field) = context.field.as_deref().filter(|s| !s.is_empty()) {
                prefix.push_str(&format!(" for {field}"));
            }
            let name = context.entity_name.as_deref().filter(|s| !s.is_empty());
            let id = context.entity_id.as_deref().filter(|s| !s.is_empty());
            if name.is_some() || id.is_some() {
                let mut parts = Vec::new();
                if let Some(name) = name {
                    parts.push(name.to_string());
                }
                if let Some(id) = id {
                    parts.push(format!("id: \"{id}\""));
                }
                prefix.push_str(&format!(" ({})", parts.join(", ")));
            }
        }
        write!(
            f,
            "{prefix}: Value: {}.\nError message: {}",
            self.value, self.message
        )
    }
}

impl std::error::Error for TypeValidationError {}

impl TypeValidationError {
    pub fn new(value: JsonValue, message: impl Into<String>) -> Self {
        Self {
            value,
            message: message.into(),
            context: None,
        }
    }

    /// `TypeValidationError.wrap({ value, cause, context })`: a
    /// `TypeValidationError` cause for the same value and context is kept.
    pub fn wrap(
        value: JsonValue,
        cause: &(dyn std::error::Error + 'static),
        context: Option<TypeValidationContext>,
    ) -> Self {
        if let Some(existing) = cause.downcast_ref::<TypeValidationError>()
            && existing.value == value
            && existing.context.as_deref().cloned().unwrap_or_default()
                == context.clone().unwrap_or_default()
        {
            return existing.clone();
        }
        Self {
            value,
            message: crate::harness::sdk::error_value::get_error_message(cause),
            context: context.map(Box::new),
        }
    }
}

/// `JSONParseError` from `@ai-sdk/provider`.
#[derive(Debug, Clone, thiserror::Error)]
#[error("JSON parsing failed: Text: {text}.\nError message: {message}")]
pub struct JsonParseError {
    pub text: String,
    pub message: String,
}

/// `validateTypes({ value, schema, context })`.
pub fn validate_types(
    value: &JsonValue,
    schema: &crate::harness::sdk::Schema,
    context: Option<TypeValidationContext>,
) -> Result<JsonValue, TypeValidationError> {
    schema
        .validate_value(value)
        .map_err(|cause| TypeValidationError::wrap(value.clone(), &*cause, context))
}

/// `safeParseJSON({ text })` / `parseJSON` failure.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParseJsonError {
    #[error(transparent)]
    Json(#[from] JsonParseError),
    #[error(transparent)]
    Validation(#[from] TypeValidationError),
}

/// `safeParseJSON({ text, schema })`. `secureJsonParse` rejects nothing
/// `serde_json` accepts except `__proto__` / `constructor.prototype` keys.
pub fn safe_parse_json(
    text: &str,
    schema: Option<&crate::harness::sdk::Schema>,
) -> Result<JsonValue, ParseJsonError> {
    let value: JsonValue = serde_json::from_str(text).map_err(|e| JsonParseError {
        text: text.to_string(),
        message: format!("SyntaxError: {e}"),
    })?;
    if has_forbidden_prototype_keys(&value) {
        return Err(JsonParseError {
            text: text.to_string(),
            message: "SyntaxError: Object contains forbidden prototype property".into(),
        }
        .into());
    }
    match schema {
        None => Ok(value),
        Some(schema) => Ok(validate_types(&value, schema, None)?),
    }
}

fn has_forbidden_prototype_keys(value: &JsonValue) -> bool {
    match value {
        JsonValue::Object(map) => {
            map.contains_key("__proto__")
                || map
                    .get("constructor")
                    .and_then(|c| c.as_object())
                    .is_some_and(|c| c.contains_key("prototype"))
                || map.values().any(has_forbidden_prototype_keys)
        }
        JsonValue::Array(items) => items.iter().any(has_forbidden_prototype_keys),
        _ => false,
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
