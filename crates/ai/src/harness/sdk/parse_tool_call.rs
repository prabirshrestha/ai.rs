//! Port of `ai` `parseToolCall`, `validateToolContext` and the tool errors
//! they raise (`NoSuchToolError`, `InvalidToolInputError`).
//!
//! [rust] The harness calls `parseToolCall` with no `repairToolCall` and no
//! `refineToolInput`, so the repair path is not mirrored.

use crate::harness::provider::{
    JsonValue, ParseJsonError, SharedV4ProviderMetadata, TypeValidationContext,
    TypeValidationError, safe_parse_json, validate_types,
};
use crate::harness::sdk::content::TypedToolCall;
use crate::harness::sdk::error_value::{ErrorValue, get_error_message};
use crate::harness::sdk::tool::{Schema, ToolSet, ToolType};

/// `NoSuchToolError` (`AI_NoSuchToolError`).
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct NoSuchToolError {
    pub tool_name: String,
    pub available_tools: Option<Vec<String>>,
    pub message: String,
}

impl NoSuchToolError {
    pub fn new(tool_name: impl Into<String>, available_tools: Option<Vec<String>>) -> Self {
        let tool_name = tool_name.into();
        let message = format!(
            "Model tried to call unavailable tool '{tool_name}'. {}",
            match &available_tools {
                None => "No tools are available.".to_string(),
                Some(tools) => format!("Available tools: {}.", tools.join(", ")),
            }
        );
        Self {
            tool_name,
            available_tools,
            message,
        }
    }

    /// `new NoSuchToolError({ toolName, message })`.
    pub fn with_message(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            available_tools: None,
            message: message.into(),
        }
    }
}

/// `InvalidToolInputError` (`AI_InvalidToolInputError`).
#[derive(Debug, Clone, thiserror::Error)]
#[error("Invalid input for tool {tool_name}: {cause_message}")]
pub struct InvalidToolInputError {
    pub tool_name: String,
    pub tool_input: String,
    pub cause_message: String,
}

/// `LanguageModelV4ToolCall`.
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageModelV4ToolCall {
    pub tool_call_id: String,
    pub tool_name: String,
    pub input: String,
    pub provider_executed: Option<bool>,
    pub dynamic: Option<bool>,
    pub provider_metadata: Option<SharedV4ProviderMetadata>,
}

/// Why `doParseToolCall` failed.
#[derive(Debug, Clone, thiserror::Error)]
enum ParseFailure {
    #[error(transparent)]
    NoSuchTool(#[from] NoSuchToolError),
    #[error(transparent)]
    InvalidInput(#[from] InvalidToolInputError),
}

fn invalid_input(call: &LanguageModelV4ToolCall, cause: &ParseJsonError) -> InvalidToolInputError {
    let cause_message = match cause {
        ParseJsonError::Json(e) => get_error_message(e),
        ParseJsonError::Validation(e) => get_error_message(e),
    };
    InvalidToolInputError {
        tool_name: call.tool_name.clone(),
        tool_input: call.input.clone(),
        cause_message,
    }
}

fn parse_provider_executed_dynamic(
    call: &LanguageModelV4ToolCall,
) -> Result<TypedToolCall, ParseFailure> {
    let input = if call.input.trim().is_empty() {
        JsonValue::Object(Default::default())
    } else {
        safe_parse_json(&call.input, None).map_err(|e| invalid_input(call, &e))?
    };
    let mut parsed = TypedToolCall::new(&call.tool_call_id, &call.tool_name, input);
    parsed.provider_executed = Some(true);
    parsed.dynamic = Some(true);
    parsed.provider_metadata = call.provider_metadata.clone();
    Ok(parsed)
}

fn do_parse_tool_call(
    call: &LanguageModelV4ToolCall,
    tools: &ToolSet,
) -> Result<TypedToolCall, ParseFailure> {
    let Some(tool) = tools.get(&call.tool_name) else {
        if call.provider_executed == Some(true) && call.dynamic == Some(true) {
            return parse_provider_executed_dynamic(call);
        }
        return Err(
            NoSuchToolError::new(&call.tool_name, Some(tools.keys().cloned().collect())).into(),
        );
    };
    let schema: &Schema = &tool.input_schema;
    let parsed = if call.input.trim().is_empty() {
        validate_types(&JsonValue::Object(Default::default()), schema, None)
            .map_err(ParseJsonError::from)
    } else {
        safe_parse_json(&call.input, Some(schema))
    }
    .map_err(|e| invalid_input(call, &e))?;
    // `setToolCallInputSchemaInput`: the JSON the schema received.
    let schema_input = if call.input.trim().is_empty() {
        Some(JsonValue::Object(Default::default()))
    } else {
        safe_parse_json(&call.input, None).ok()
    };
    let mut out = TypedToolCall::new(&call.tool_call_id, &call.tool_name, parsed);
    out.input_schema_input = crate::harness::sdk::content::InputSchemaInput(schema_input);
    out.provider_executed = call.provider_executed;
    out.provider_metadata = call.provider_metadata.clone();
    out.tool_metadata = tool.metadata.clone();
    out.title = tool.title.clone();
    if matches!(tool.r#type, ToolType::Dynamic) {
        out.dynamic = Some(true);
    }
    Ok(out)
}

/// `parseToolCall({ toolCall, tools, repairToolCall: undefined, ... })`.
///
/// Never fails: an unknown tool or invalid input yields a dynamic tool call
/// with `invalid: true` and the error.
pub fn parse_tool_call(call: &LanguageModelV4ToolCall, tools: Option<&ToolSet>) -> TypedToolCall {
    let result = match tools {
        None => {
            if call.provider_executed == Some(true) && call.dynamic == Some(true) {
                parse_provider_executed_dynamic(call)
            } else {
                Err(NoSuchToolError::new(&call.tool_name, None).into())
            }
        }
        Some(tools) => do_parse_tool_call(call, tools),
    };
    match result {
        Ok(parsed) => parsed,
        Err(error) => {
            let input = safe_parse_json(&call.input, None)
                .unwrap_or_else(|_| JsonValue::String(call.input.clone()));
            let tool = tools.and_then(|t| t.get(&call.tool_name));
            let error: crate::harness::errors::BoxError = match error {
                ParseFailure::NoSuchTool(e) => Box::new(e),
                ParseFailure::InvalidInput(e) => Box::new(e),
            };
            let mut parsed = TypedToolCall::new(&call.tool_call_id, &call.tool_name, input);
            parsed.dynamic = Some(true);
            parsed.invalid = Some(true);
            parsed.error = Some(ErrorValue::from(error));
            parsed.title = tool.and_then(|t| t.title.clone());
            parsed.provider_executed = call.provider_executed;
            parsed.provider_metadata = call.provider_metadata.clone();
            parsed.tool_metadata = tool.and_then(|t| t.metadata.clone());
            parsed
        }
    }
}

/// `validateToolContext({ toolName, context, contextSchema })`.
///
/// [rust] A missing context (`undefined`) is `None`; it validates as JSON
/// `null`.
pub fn validate_tool_context(
    tool_name: &str,
    context: Option<&JsonValue>,
    context_schema: Option<&Schema>,
) -> Result<Option<JsonValue>, TypeValidationError> {
    let Some(schema) = context_schema else {
        return Ok(context.cloned());
    };
    let value = context.cloned().unwrap_or(JsonValue::Null);
    validate_types(
        &value,
        schema,
        Some(TypeValidationContext {
            field: Some("tool context".into()),
            entity_name: Some(tool_name.into()),
            entity_id: None,
        }),
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sdk::tool::{Tool, json_schema};
    use serde_json::json;

    fn call(name: &str, input: &str) -> LanguageModelV4ToolCall {
        LanguageModelV4ToolCall {
            tool_call_id: "c1".into(),
            tool_name: name.into(),
            input: input.into(),
            provider_executed: None,
            dynamic: None,
            provider_metadata: None,
        }
    }

    fn tools() -> ToolSet {
        let schema = json_schema(json!({"type": "object"})).with_validate(|v| {
            if v.get("x").is_some_and(|x| x.is_number()) {
                Ok(v.clone())
            } else {
                Err("x must be a number".into())
            }
        });
        let mut tools = ToolSet::new();
        tools.insert("t".into(), Tool::new(schema));
        tools
    }

    #[test]
    fn parses_valid_input() {
        let parsed = parse_tool_call(&call("t", r#"{"x":1}"#), Some(&tools()));
        assert_eq!(parsed.input, json!({"x": 1}));
        assert_eq!(parsed.invalid, None);
        assert_eq!(parsed.dynamic, None);
    }

    #[test]
    fn invalid_input_becomes_an_invalid_dynamic_call() {
        let parsed = parse_tool_call(&call("t", r#"{"x":"a"}"#), Some(&tools()));
        assert_eq!(parsed.invalid, Some(true));
        assert_eq!(parsed.dynamic, Some(true));
        assert_eq!(parsed.input, json!({"x": "a"}));
        let error = parsed.error.unwrap();
        assert_eq!(
            error.as_error().unwrap().to_string(),
            "Invalid input for tool t: AI_TypeValidationError: Type validation failed: Value: {\"x\":\"a\"}.\nError message: Error: x must be a number"
        );
    }

    #[test]
    fn unknown_tools_name_the_available_tools() {
        let parsed = parse_tool_call(&call("u", "{}"), Some(&tools()));
        assert_eq!(
            parsed.error.unwrap().as_error().unwrap().to_string(),
            "Model tried to call unavailable tool 'u'. Available tools: t."
        );
        let parsed = parse_tool_call(&call("u", "not json"), None);
        assert_eq!(parsed.input, json!("not json"));
        assert_eq!(
            parsed.error.unwrap().as_error().unwrap().to_string(),
            "Model tried to call unavailable tool 'u'. No tools are available."
        );
    }

    #[test]
    fn provider_executed_dynamic_calls_need_no_tool() {
        let mut c = call("builtin", "");
        c.provider_executed = Some(true);
        c.dynamic = Some(true);
        let parsed = parse_tool_call(&c, None);
        assert_eq!(parsed.input, json!({}));
        assert_eq!(parsed.provider_executed, Some(true));
        assert_eq!(parsed.invalid, None);
    }

    #[test]
    fn validates_tool_context_with_its_entity() {
        let schema = json_schema(json!({})).with_validate(|v| {
            if v.is_object() {
                Ok(v.clone())
            } else {
                Err("expected object".into())
            }
        });
        assert_eq!(
            validate_tool_context("t", Some(&json!({"a": 1})), Some(&schema)).unwrap(),
            Some(json!({"a": 1}))
        );
        let err = validate_tool_context("t", None, Some(&schema)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Type validation failed for tool context (t): Value: null.\nError message: Error: expected object"
        );
        assert_eq!(validate_tool_context("t", None, None).unwrap(), None);
    }
}
