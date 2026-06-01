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

    if types.contains(&"object")
        && let Value::Object(object) = next
    {
        next = Value::Object(coerce_object(object, schema));
    }

    if types.contains(&"array")
        && let Value::Array(array) = next
    {
        next = Value::Array(coerce_array(array, schema));
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

    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array)
        && any_of.iter().all(|nested| {
            let mut nested_errors = Vec::new();
            validate_value(value, nested, path, &mut nested_errors);
            !nested_errors.is_empty()
        })
    {
        errors.push(format!("{path}: must match at least one schema"));
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

    validate_const_and_enum(value, schema, path, errors);

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

    if value.is_number() {
        validate_number(value, schema, path, errors);
    }

    if value.is_string() {
        validate_string(value, schema, path, errors);
    }
}

fn validate_object(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(object) = value.as_object() else {
        return;
    };

    if let Some(min_properties) = schema.get("minProperties").and_then(Value::as_u64)
        && (object.len() as u64) < min_properties
    {
        errors.push(format!(
            "{path}: must have at least {min_properties} properties"
        ));
    }

    if let Some(max_properties) = schema.get("maxProperties").and_then(Value::as_u64)
        && (object.len() as u64) > max_properties
    {
        errors.push(format!(
            "{path}: must have at most {max_properties} properties"
        ));
    }

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

    match schema.get("additionalProperties") {
        Some(Value::Bool(false)) => {
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                for key in object.keys() {
                    if !properties.contains_key(key) {
                        errors.push(format!("{}: unexpected property", child_path(path, key)));
                    }
                }
            }
        }
        Some(additional_schema) if additional_schema.is_object() => {
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .map(|properties| properties.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            for (key, property_value) in object {
                if !properties.contains(key) {
                    validate_value(
                        property_value,
                        additional_schema,
                        &child_path(path, key),
                        errors,
                    );
                }
            }
        }
        _ => {}
    }
}

fn validate_array(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(array) = value.as_array() else {
        return;
    };

    if let Some(min_items) = schema.get("minItems").and_then(Value::as_u64)
        && (array.len() as u64) < min_items
    {
        errors.push(format!("{path}: must have at least {min_items} items"));
    }

    if let Some(max_items) = schema.get("maxItems").and_then(Value::as_u64)
        && (array.len() as u64) > max_items
    {
        errors.push(format!("{path}: must have at most {max_items} items"));
    }

    if schema.get("uniqueItems") == Some(&Value::Bool(true)) {
        for (index, item) in array.iter().enumerate() {
            if array
                .iter()
                .skip(index + 1)
                .any(|other| json_schema_equal(item, other))
            {
                errors.push(format!("{path}: must have unique items"));
                break;
            }
        }
    }

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

fn validate_const_and_enum(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    if let Some(expected) = schema.get("const")
        && !json_schema_equal(value, expected)
    {
        errors.push(format!("{path}: must equal const value"));
    }

    if let Some(variants) = schema.get("enum").and_then(Value::as_array)
        && !variants
            .iter()
            .any(|variant| json_schema_equal(value, variant))
    {
        errors.push(format!("{path}: must equal one of the allowed values"));
    }
}

fn validate_number(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(number) = value.as_f64() else {
        return;
    };

    if let Some(minimum) = schema.get("minimum").and_then(Value::as_f64)
        && number < minimum
    {
        errors.push(format!("{path}: must be >= {minimum}"));
    }

    if let Some(maximum) = schema.get("maximum").and_then(Value::as_f64)
        && number > maximum
    {
        errors.push(format!("{path}: must be <= {maximum}"));
    }

    if let Some(exclusive_minimum) = schema.get("exclusiveMinimum").and_then(Value::as_f64)
        && number <= exclusive_minimum
    {
        errors.push(format!("{path}: must be > {exclusive_minimum}"));
    }

    if let Some(exclusive_maximum) = schema.get("exclusiveMaximum").and_then(Value::as_f64)
        && number >= exclusive_maximum
    {
        errors.push(format!("{path}: must be < {exclusive_maximum}"));
    }

    if let Some(multiple_of) = schema.get("multipleOf").and_then(Value::as_f64)
        && multiple_of > 0.0
    {
        let quotient = number / multiple_of;
        if (quotient - quotient.round()).abs() > 1e-9 {
            errors.push(format!("{path}: must be a multiple of {multiple_of}"));
        }
    }
}

fn validate_string(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(text) = value.as_str() else {
        return;
    };
    let len = text.chars().count() as u64;

    if let Some(min_length) = schema.get("minLength").and_then(Value::as_u64)
        && len < min_length
    {
        errors.push(format!("{path}: must be at least {min_length} characters"));
    }

    if let Some(max_length) = schema.get("maxLength").and_then(Value::as_u64)
        && len > max_length
    {
        errors.push(format!("{path}: must be at most {max_length} characters"));
    }

    if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
        match regex::Regex::new(pattern) {
            Ok(regex) => {
                if !regex.is_match(text) {
                    errors.push(format!("{path}: must match pattern {pattern:?}"));
                }
            }
            Err(error) => errors.push(format!("{path}: invalid regex pattern: {error}")),
        }
    }
}

fn json_schema_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => match (left.as_f64(), right.as_f64()) {
            (Some(left), Some(right)) => (left - right).abs() <= f64::EPSILON,
            _ => left == right,
        },
        _ => left == right,
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
    fn still_validates_when_function_constructor_is_unavailable() {
        let tool = Tool {
            name: "echo".to_string(),
            description: "Echo tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "count": { "type": "number" }
                },
                "required": ["count"]
            }),
        };
        let tool_call = ToolCall {
            id: "tool-1".to_string(),
            name: "echo".to_string(),
            arguments: json!({ "count": "42" }),
            thought_signature: None,
        };

        assert_eq!(
            validate_tool_arguments(&tool, &tool_call).unwrap(),
            json!({ "count": 42.0 })
        );
    }

    #[test]
    fn coerces_serialized_plain_json_schemas_with_ajv_compatible_primitive_rules() {
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

    #[test]
    fn validates_common_json_schema_constraints() {
        let tool = Tool {
            name: "configure".to_string(),
            description: "Configure a task".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "mode": { "type": "string", "enum": ["fast", "safe"] },
                    "count": { "type": "integer", "minimum": 1, "maximum": 5 },
                    "tag": { "type": "string", "minLength": 2, "maxLength": 4, "pattern": "^[a-z]+$" },
                    "items": { "type": "array", "minItems": 1, "maxItems": 2, "uniqueItems": true, "items": { "type": "string" } },
                    "fixed": { "const": "yes" }
                },
                "required": ["mode", "count", "tag", "items", "fixed"],
                "additionalProperties": { "type": "number" }
            }),
        };
        let valid = ToolCall {
            id: "tool-1".to_string(),
            name: "configure".to_string(),
            arguments: json!({
                "mode": "safe",
                "count": "3",
                "tag": "ab",
                "items": ["one"],
                "fixed": "yes",
                "extra": "4"
            }),
            thought_signature: None,
        };

        assert_eq!(
            validate_tool_arguments(&tool, &valid).unwrap(),
            json!({
                "mode": "safe",
                "count": 3,
                "tag": "ab",
                "items": ["one"],
                "fixed": "yes",
                "extra": 4.0
            })
        );

        let invalid = ToolCall {
            id: "tool-2".to_string(),
            name: "configure".to_string(),
            arguments: json!({
                "mode": "unsafe",
                "count": 6,
                "tag": "A",
                "items": ["one", "one", "two"],
                "fixed": "no",
                "extra": "not-a-number"
            }),
            thought_signature: None,
        };
        let error = validate_tool_arguments(&tool, &invalid)
            .expect_err("expected validation error")
            .to_string();
        assert!(error.contains("mode: must equal one of the allowed values"));
        assert!(error.contains("count: must be <= 5"));
        assert!(error.contains("tag: must be at least 2 characters"));
        assert!(error.contains("tag: must match pattern"));
        assert!(error.contains("items: must have at most 2 items"));
        assert!(error.contains("items: must have unique items"));
        assert!(error.contains("fixed: must equal const value"));
        assert!(error.contains("extra: expected number"));
    }
}
