//! Port of `harness-v1-response-format.ts`.

use serde::{Deserialize, Serialize};

/// `HarnessV1JSONValue`.
pub type HarnessV1JsonValue = serde_json::Value;
/// `HarnessV1JSONArray`.
pub type HarnessV1JsonArray = Vec<serde_json::Value>;
/// `HarnessV1JSONObject`.
pub type HarnessV1JsonObject = serde_json::Map<String, serde_json::Value>;
/// `HarnessV1JSONSchema`.
pub type HarnessV1JsonSchema = HarnessV1JsonObject;

/// `HarnessV1ResponseFormat`: requested response format for one turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum HarnessV1ResponseFormat {
    Text,
    Json {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<HarnessV1JsonSchema>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}
