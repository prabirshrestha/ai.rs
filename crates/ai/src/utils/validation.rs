use serde_json::{Map, Number, Value};

use crate::types::{Tool, ToolCall};
use crate::{Error, Result};

pub fn validate_tool_call(tools: &[Tool], tool_call: &ToolCall) -> Result<Value> {
    let tool = tools
        .iter()
        .find(|tool| tool.name == tool_call.name)
        .ok_or_else(|| Error::Validation(format!("Tool \"{}\" not found", tool_call.name)))?;
    validate_tool_arguments(tool, tool_call)
}

pub fn validate_tool_arguments(tool: &Tool, tool_call: &ToolCall) -> Result<Value> {
    let args = coerce_with_json_schema(tool_call.arguments.clone(), &tool.parameters);
    let mut errors = Vec::new();
    validate_value(&args, &tool.parameters, "root", &mut errors);
    if errors.is_empty() {
        return Ok(args);
    }

    Err(Error::Validation(format!(
        "Validation failed for tool \"{}\":\n{}\n\nReceived arguments:\n{}",
        tool_call.name,
        errors
            .iter()
            .map(|error| format!("  - {error}"))
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::to_string_pretty(&tool_call.arguments).unwrap_or_else(|_| "null".to_string())
    )))
}

fn schema_types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(value)) => vec![value.as_str()],
        Some(Value::Array(values)) => values.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn matches_json_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn coerce_primitive_by_type(value: Value, schema_type: &str) -> Value {
    match schema_type {
        "number" => match value {
            Value::Null => json_number(0.0),
            Value::String(text) if !text.trim().is_empty() => text
                .parse::<f64>()
                .ok()
                .and_then(Number::from_f64)
                .map(Value::Number)
                .unwrap_or(Value::String(text)),
            Value::Bool(flag) => json_number(if flag { 1.0 } else { 0.0 }),
            other => other,
        },
        "integer" => match value {
            Value::Null => Value::Number(Number::from(0)),
            Value::String(text) if !text.trim().is_empty() => text
                .parse::<i64>()
                .ok()
                .map(Number::from)
                .map(Value::Number)
                .unwrap_or(Value::String(text)),
            Value::Bool(flag) => Value::Number(Number::from(if flag { 1 } else { 0 })),
            other => other,
        },
        "boolean" => match value {
            Value::Null => Value::Bool(false),
            Value::String(text) if text == "true" => Value::Bool(true),
            Value::String(text) if text == "false" => Value::Bool(false),
            Value::Number(number) if number.as_i64() == Some(1) => Value::Bool(true),
            Value::Number(number) if number.as_i64() == Some(0) => Value::Bool(false),
            other => other,
        },
        "string" => match value {
            Value::Null => Value::String(String::new()),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Bool(flag) => Value::String(flag.to_string()),
            other => other,
        },
        "null" => match value {
            Value::String(text) if text.is_empty() => Value::Null,
            Value::Number(number) if number.as_i64() == Some(0) => Value::Null,
            Value::Bool(false) => Value::Null,
            other => other,
        },
        _ => value,
    }
}

fn json_number(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn coerce_with_json_schema(value: Value, schema: &Value) -> Value {
    let mut next = value;

    if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
        for nested in all_of {
            next = coerce_with_json_schema(next, nested);
        }
    }

    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        next = coerce_with_union_schema(next, any_of);
    }

    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        next = coerce_with_union_schema(next, one_of);
    }

    let types = schema_types(schema);
    let matches_union_member = types.len() > 1
        && types
            .iter()
            .any(|schema_type| matches_json_type(&next, schema_type));
    if !types.is_empty() && !matches_union_member {
        for schema_type in &types {
            let candidate = coerce_primitive_by_type(next.clone(), schema_type);
            if candidate != next {
                next = candidate;
                break;
            }
        }
    }

    if types.contains(&"object") {
        if let Value::Object(object) = next {
            next = Value::Object(coerce_object(object, schema));
        }
    }

    if types.contains(&"array") {
        if let Value::Array(array) = next {
            next = Value::Array(coerce_array(array, schema));
        }
    }

    next
}

fn coerce_with_union_schema(value: Value, schemas: &[Value]) -> Value {
    for schema in schemas {
        let candidate = coerce_with_json_schema(value.clone(), schema);
        let mut errors = Vec::new();
        validate_value(&candidate, schema, "root", &mut errors);
        if errors.is_empty() {
            return candidate;
        }
    }
    value
}

fn coerce_object(mut object: Map<String, Value>, schema: &Value) -> Map<String, Value> {
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (key, property_schema) in properties {
            if let Some(value) = object.remove(key) {
                object.insert(key.clone(), coerce_with_json_schema(value, property_schema));
            }
        }
    }

    if let Some(additional_schema) = schema
        .get("additionalProperties")
        .filter(|value| value.is_object())
    {
        let defined = schema
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| properties.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        for (key, value) in object.clone() {
            if !defined.contains(&key) {
                object.insert(key, coerce_with_json_schema(value, additional_schema));
            }
        }
    }

    object
}

fn coerce_array(mut array: Vec<Value>, schema: &Value) -> Vec<Value> {
    match schema.get("items") {
        Some(Value::Array(items)) => {
            for (index, item_schema) in items.iter().enumerate() {
                if let Some(value) = array.get_mut(index) {
                    *value = coerce_with_json_schema(value.clone(), item_schema);
                }
            }
        }
        Some(item_schema) if item_schema.is_object() => {
            for value in &mut array {
                *value = coerce_with_json_schema(value.clone(), item_schema);
            }
        }
        _ => {}
    }
    array
}

fn validate_value(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
        for nested in all_of {
            validate_value(value, nested, path, errors);
        }
    }

    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        if any_of.iter().all(|nested| {
            let mut nested_errors = Vec::new();
            validate_value(value, nested, path, &mut nested_errors);
            !nested_errors.is_empty()
        }) {
            errors.push(format!("{path}: must match at least one schema"));
        }
    }

    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        let matches = one_of
            .iter()
            .filter(|nested| {
                let mut nested_errors = Vec::new();
                validate_value(value, nested, path, &mut nested_errors);
                nested_errors.is_empty()
            })
            .count();
        if matches != 1 {
            errors.push(format!("{path}: must match exactly one schema"));
        }
    }

    let types = schema_types(schema);
    if !types.is_empty()
        && !types
            .iter()
            .any(|schema_type| matches_json_type(value, schema_type))
    {
        errors.push(format!("{path}: expected {}", types.join(" or ")));
        return;
    }

    if types.contains(&"object") {
        validate_object(value, schema, path, errors);
    }

    if types.contains(&"array") {
        validate_array(value, schema, path, errors);
    }
}

fn validate_object(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(object) = value.as_object() else {
        return;
    };

    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for key in required.iter().filter_map(Value::as_str) {
            if !object.contains_key(key) {
                errors.push(format!(
                    "{}: missing required property",
                    child_path(path, key)
                ));
            }
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (key, property_schema) in properties {
            if let Some(property_value) = object.get(key) {
                validate_value(
                    property_value,
                    property_schema,
                    &child_path(path, key),
                    errors,
                );
            }
        }
    }

    if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for key in object.keys() {
                if !properties.contains_key(key) {
                    errors.push(format!("{}: unexpected property", child_path(path, key)));
                }
            }
        }
    }
}

fn validate_array(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(array) = value.as_array() else {
        return;
    };
    match schema.get("items") {
        Some(Value::Array(items)) => {
            for (index, item_schema) in items.iter().enumerate() {
                if let Some(item) = array.get(index) {
                    validate_value(item, item_schema, &format!("{path}.{index}"), errors);
                }
            }
        }
        Some(item_schema) if item_schema.is_object() => {
            for (index, item) in array.iter().enumerate() {
                validate_value(item, item_schema, &format!("{path}.{index}"), errors);
            }
        }
        _ => {}
    }
}

fn child_path(parent: &str, child: &str) -> String {
    if parent == "root" {
        child.to_string()
    } else {
        format!("{parent}.{child}")
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn create_tool_call_with_plain_schema(schema: Value, value: Value) -> (Tool, ToolCall) {
        let tool = Tool {
            name: "echo".to_string(),
            description: "Echo tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": schema
                },
                "required": ["value"]
            }),
        };
        let tool_call = ToolCall {
            id: "tool-1".to_string(),
            name: "echo".to_string(),
            arguments: json!({ "value": value }),
            thought_signature: None,
        };
        (tool, tool_call)
    }

    #[test]
    fn coerces_serialized_plain_json_schemas_with_primitive_rules() {
        let cases = [
            (json!({ "type": "number" }), json!("42"), json!(42.0)),
            (json!({ "type": "number" }), json!(true), json!(1.0)),
            (json!({ "type": "number" }), Value::Null, json!(0.0)),
            (json!({ "type": "integer" }), json!("42"), json!(42)),
            (json!({ "type": "boolean" }), json!("true"), json!(true)),
            (json!({ "type": "boolean" }), json!("false"), json!(false)),
            (json!({ "type": "boolean" }), json!(1), json!(true)),
            (json!({ "type": "boolean" }), json!(0), json!(false)),
            (json!({ "type": "string" }), Value::Null, json!("")),
            (json!({ "type": "string" }), json!(true), json!("true")),
            (json!({ "type": "null" }), json!(""), Value::Null),
            (json!({ "type": "null" }), json!(0), Value::Null),
            (json!({ "type": "null" }), json!(false), Value::Null),
            (
                json!({ "type": ["number", "string"] }),
                json!("1"),
                json!("1"),
            ),
            (
                json!({ "type": ["boolean", "number"] }),
                json!("1"),
                json!(1.0),
            ),
        ];

        for (schema, input, expected) in cases {
            let (tool, tool_call) = create_tool_call_with_plain_schema(schema, input);
            assert_eq!(
                validate_tool_arguments(&tool, &tool_call).unwrap(),
                json!({ "value": expected })
            );
        }
    }

    #[test]
    fn rejects_invalid_coercions_for_serialized_plain_json_schemas() {
        let cases = [
            (json!({ "type": "boolean" }), json!("1")),
            (json!({ "type": "boolean" }), json!("0")),
            (json!({ "type": "null" }), json!("null")),
            (json!({ "type": "integer" }), json!("42.1")),
        ];

        for (schema, input) in cases {
            let (tool, tool_call) = create_tool_call_with_plain_schema(schema, input);
            let error = validate_tool_arguments(&tool, &tool_call)
                .expect_err("expected validation error")
                .to_string();
            assert!(error.contains("Validation failed"));
        }
    }
}
