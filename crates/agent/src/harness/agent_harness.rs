use std::collections::HashMap;

use ai::{CacheRetention, Model, Transport};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentHarnessErrorCode {
    Busy,
    InvalidState,
    InvalidArgument,
    Auth,
    Session,
    Compaction,
    BranchSummary,
    Hook,
    Unknown,
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct AgentHarnessError {
    pub code: AgentHarnessErrorCode,
    message: String,
}

impl AgentHarnessError {
    pub fn new(code: AgentHarnessErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentHarnessPhase {
    Idle,
    Turn,
    Compaction,
    BranchSummary,
    Retry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessAuth {
    pub api_key: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
}

impl Default for AgentHarnessStreamOptions {
    fn default() -> Self {
        Self {
            transport: None,
            timeout_ms: None,
            max_retries: None,
            max_retry_delay_ms: None,
            headers: HashMap::new(),
            metadata: None,
            cache_retention: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MapPatch<T> {
    Clear,
    Merge(HashMap<String, Option<T>>),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessStreamOptionsPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<Option<Transport>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<Option<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<Option<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<Option<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<Option<CacheRetention>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<MapPatch<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MapPatch<Value>>,
}

pub fn clone_stream_options(
    stream_options: &AgentHarnessStreamOptions,
) -> AgentHarnessStreamOptions {
    stream_options.clone()
}

pub fn merge_headers(
    sources: impl IntoIterator<Item = Option<HashMap<String, String>>>,
) -> HashMap<String, String> {
    let mut merged = HashMap::new();
    for source in sources.into_iter().flatten() {
        merged.extend(source);
    }
    merged
}

pub fn apply_stream_options_patch(
    base: &AgentHarnessStreamOptions,
    patch: Option<&AgentHarnessStreamOptionsPatch>,
) -> AgentHarnessStreamOptions {
    let mut result = clone_stream_options(base);
    let Some(patch) = patch else {
        return result;
    };

    apply_scalar_patch(&mut result.transport, &patch.transport);
    apply_scalar_patch(&mut result.timeout_ms, &patch.timeout_ms);
    apply_scalar_patch(&mut result.max_retries, &patch.max_retries);
    apply_scalar_patch(&mut result.max_retry_delay_ms, &patch.max_retry_delay_ms);
    apply_scalar_patch(&mut result.cache_retention, &patch.cache_retention);

    if let Some(headers) = patch.headers.as_ref() {
        apply_map_patch(&mut result.headers, headers);
    }
    if let Some(metadata) = patch.metadata.as_ref() {
        apply_metadata_patch(&mut result.metadata, metadata);
    }

    result
}

pub fn validate_unique_names(
    names: impl IntoIterator<Item = impl AsRef<str>>,
    message: &str,
) -> Result<(), AgentHarnessError> {
    let mut seen = std::collections::HashSet::new();
    let mut duplicates = Vec::new();
    for name in names {
        let name = name.as_ref().to_string();
        if !seen.insert(name.clone()) && !duplicates.contains(&name) {
            duplicates.push(name);
        }
    }
    if duplicates.is_empty() {
        Ok(())
    } else {
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::InvalidArgument,
            format!("{message}: {}", duplicates.join(", ")),
        ))
    }
}

pub fn create_failure_message(model: &Model, error: impl ToString, aborted: bool) -> ai::Message {
    ai::Message::Assistant(ai::AssistantMessage {
        content: vec![ai::AssistantContent::Text(ai::TextContent {
            text: String::new(),
            text_signature: None,
        })],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: ai::Usage::default(),
        stop_reason: if aborted {
            ai::StopReason::Aborted
        } else {
            ai::StopReason::Error
        },
        error_message: Some(error.to_string()),
        timestamp: ai::utils::time::now_millis(),
    })
}

fn apply_scalar_patch<T: Clone>(target: &mut Option<T>, patch: &Option<Option<T>>) {
    if let Some(value) = patch {
        *target = value.clone();
    }
}

fn apply_map_patch<T: Clone>(target: &mut HashMap<String, T>, patch: &MapPatch<T>) {
    match patch {
        MapPatch::Clear => target.clear(),
        MapPatch::Merge(values) => {
            for (key, value) in values {
                match value {
                    Some(value) => {
                        target.insert(key.clone(), value.clone());
                    }
                    None => {
                        target.remove(key);
                    }
                }
            }
        }
    }
}

fn apply_metadata_patch(target: &mut Option<Value>, patch: &MapPatch<Value>) {
    match patch {
        MapPatch::Clear => {
            *target = None;
        }
        MapPatch::Merge(values) => {
            let mut object = target
                .take()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            for (key, value) in values {
                match value {
                    Some(value) => {
                        object.insert(key.clone(), value.clone());
                    }
                    None => {
                        object.remove(key);
                    }
                }
            }
            *target = if object.is_empty() {
                None
            } else {
                Some(Value::Object(object))
            };
        }
    }
}
