use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::AssistantMessage;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticErrorInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageDiagnostic {
    #[serde(rename = "type")]
    pub diagnostic_type: String,
    pub timestamp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticErrorInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

pub fn format_thrown_value(value: impl ToString) -> String {
    value.to_string()
}

pub fn extract_diagnostic_error(error: &(dyn std::error::Error + 'static)) -> DiagnosticErrorInfo {
    DiagnosticErrorInfo {
        name: Some(std::any::type_name_of_val(error).to_string()),
        message: error.to_string(),
        stack: None,
        code: None,
    }
}

pub fn diagnostic_error_from_message(message: impl Into<String>) -> DiagnosticErrorInfo {
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
    details: Option<Value>,
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
    if let Ok(value) = serde_json::to_value(diagnostic) {
        message.diagnostics.push(value);
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::types::{Model, ModelCost};

    use super::*;

    #[test]
    fn creates_and_appends_diagnostic() {
        let diagnostic = create_assistant_message_diagnostic(
            "provider-retry",
            diagnostic_error_from_message("temporary failure"),
            Some(json!({ "attempt": 1 })),
        );
        assert_eq!(diagnostic.diagnostic_type, "provider-retry");
        assert_eq!(
            diagnostic
                .error
                .as_ref()
                .map(|error| error.message.as_str()),
            Some("temporary failure")
        );

        let model = Model {
            id: "test".to_string(),
            name: "test".to_string(),
            api: "test-api".to_string(),
            provider: "test-provider".to_string(),
            base_url: "http://localhost".to_string(),
            cost: ModelCost::default(),
            ..Model::default()
        };
        let mut message = crate::types::AssistantMessage::empty_for(&model);
        append_assistant_message_diagnostic(&mut message, diagnostic);

        assert_eq!(message.diagnostics.len(), 1);
        assert_eq!(message.diagnostics[0]["type"], "provider-retry");
        assert_eq!(message.diagnostics[0]["details"]["attempt"], 1);
    }
}
