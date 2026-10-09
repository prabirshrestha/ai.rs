//! Port of `utils/diagnostics.ts`.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Error;
use crate::types::AssistantMessage;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticErrorInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    /// `string | number`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageDiagnostic {
    #[serde(rename = "type")]
    pub diagnostic_type: String,
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticErrorInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Map<String, Value>>,
}

/// `formatThrownValue()`: the error's message.
pub fn format_thrown_value(value: &dyn fmt::Display) -> String {
    value.to_string()
}

/// `extractDiagnosticError()`. Rust errors have no stack; `name` follows
/// Pi's error classes (`AbortError`, `ModelsError`, `Error`) and `code` is
/// the HTTP status of provider errors.
pub fn extract_diagnostic_error(error: &Error) -> DiagnosticErrorInfo {
    let name = match error {
        Error::Aborted(_) => "AbortError",
        Error::Models(_) => "ModelsError",
        _ => "Error",
    };
    let code = match error {
        Error::ProviderHttp(error) => error.status.map(Value::from),
        _ => None,
    };
    DiagnosticErrorInfo {
        name: Some(name.to_string()),
        message: error.to_string(),
        stack: None,
        code,
    }
}

/// `extractDiagnosticError()` for a thrown non-`Error` value.
pub fn extract_thrown_value_diagnostic(message: impl Into<String>) -> DiagnosticErrorInfo {
    DiagnosticErrorInfo {
        name: Some("ThrownValue".to_string()),
        message: message.into(),
        stack: None,
        code: None,
    }
}

pub fn create_assistant_message_diagnostic(
    diagnostic_type: impl Into<String>,
    error: DiagnosticErrorInfo,
    details: Option<Map<String, Value>>,
) -> AssistantMessageDiagnostic {
    AssistantMessageDiagnostic {
        diagnostic_type: diagnostic_type.into(),
        timestamp: crate::utils::time::now_millis(),
        error: Some(error),
        details,
    }
}

pub fn append_assistant_message_diagnostic(
    message: &mut AssistantMessage,
    diagnostic: AssistantMessageDiagnostic,
) {
    message
        .diagnostics
        .get_or_insert_with(Vec::new)
        .push(diagnostic);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Model;

    #[test]
    fn appends_diagnostics_and_serializes_pi_shape() {
        let mut message = AssistantMessage::empty_for(&Model::default());
        let diagnostic = create_assistant_message_diagnostic(
            "stream_error",
            extract_diagnostic_error(&Error::message("boom")),
            None,
        );
        append_assistant_message_diagnostic(&mut message, diagnostic);
        let value = serde_json::to_value(&message).unwrap();
        assert_eq!(value["diagnostics"][0]["type"], "stream_error");
        assert_eq!(value["diagnostics"][0]["error"]["message"], "boom");
        assert_eq!(value["diagnostics"][0]["error"]["name"], "Error");
        assert_eq!(
            extract_diagnostic_error(&Error::aborted()).name.as_deref(),
            Some("AbortError")
        );
    }
}
