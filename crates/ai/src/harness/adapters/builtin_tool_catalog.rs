//! [rust] An adapter's built-in tool catalog (`builtinTools`), loaded from
//! the `builtin-tools.json` that `scripts/refresh-harness-bridges.sh` dumps
//! from the TS adapter. Each schema is what `asSchema` reduces the adapter's
//! Zod schema to, and is validated with [`validate_json_schema`], so the
//! Rust catalog matches the TS one field for field.

#![allow(dead_code)]

use indexmap::IndexMap;
use regex::Regex;
use serde::Deserialize;
use serde_json::Value as JsonValue;

use crate::harness::sdk::tool::{Schema, Tool};
use crate::harness::spec::tools::{
    HarnessV1BuiltinTool, HarnessV1BuiltinToolName, HarnessV1BuiltinToolUseKind,
    harness_v1_builtin_tools,
};
use crate::harness::utils::plain_error;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CatalogEntry {
    title: Option<String>,
    description: Option<String>,
    native_name: Option<String>,
    common_name: Option<HarnessV1BuiltinToolName>,
    tool_use_kind: Option<HarnessV1BuiltinToolUseKind>,
    input_schema: JsonValue,
    output_schema: Option<JsonValue>,
}

/// A JSON Schema plus a validator that enforces it.
pub(crate) fn validated_schema(json_schema: JsonValue) -> Schema {
    let validated = json_schema.clone();
    Schema {
        json_schema,
        validate: None,
    }
    .with_validate(move |value| {
        validate_json_schema(&validated, value)
            .map(|()| value.clone())
            .map_err(plain_error)
    })
}

/// Parses a `builtin-tools.json` catalog, keeping its order.
///
/// The harness `askUserQuestions` tool keeps the harness validator, as the
/// TS adapters spread `HARNESS_V1_BUILTIN_TOOLS.askUserQuestions`.
pub(crate) fn load_builtin_tool_catalog(json: &str) -> IndexMap<String, HarnessV1BuiltinTool> {
    let entries: IndexMap<String, CatalogEntry> =
        serde_json::from_str(json).expect("the vendored built-in tool catalog is valid");
    entries
        .into_iter()
        .map(|(key, entry)| {
            let mut tool = if key == "askUserQuestions" && entry.common_name.is_none() {
                harness_v1_builtin_tools()[&HarnessV1BuiltinToolName::AskUserQuestions].clone()
            } else {
                let mut tool = Tool::new(validated_schema(entry.input_schema));
                tool.output_schema = entry.output_schema.map(validated_schema);
                tool
            };
            tool.title = entry.title;
            tool.description = entry.description;
            (
                key,
                HarnessV1BuiltinTool {
                    tool,
                    native_name: entry.native_name,
                    common_name: entry.common_name,
                    tool_use_kind: entry.tool_use_kind,
                },
            )
        })
        .collect()
}

fn type_matches(expected: &str, value: &JsonValue) -> bool {
    match expected {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_f64().is_some_and(|number| number.fract() == 0.0),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn at(path: &str) -> &str {
    if path.is_empty() { "input" } else { path }
}

/// Validates `value` against the draft-07 subset Zod's JSON Schema output
/// uses: `type`, `properties`, `required`, `additionalProperties`,
/// `propertyNames`, `items`, `enum`, `const`, `anyOf`, `oneOf`, `allOf`,
/// string and array lengths, `pattern` and numeric bounds.
pub(crate) fn validate_json_schema(schema: &JsonValue, value: &JsonValue) -> Result<(), String> {
    validate(schema, value, "")
}

fn validate(schema: &JsonValue, value: &JsonValue, path: &str) -> Result<(), String> {
    let Some(schema) = schema.as_object() else {
        return match schema {
            JsonValue::Bool(false) => Err(format!("Invalid {}: not allowed", at(path))),
            _ => Ok(()),
        };
    };
    match schema.get("type") {
        Some(JsonValue::String(expected)) if !type_matches(expected, value) => {
            return Err(format!("Invalid {}: expected {expected}", at(path)));
        }
        Some(JsonValue::Array(types))
            if !types
                .iter()
                .filter_map(JsonValue::as_str)
                .any(|expected| type_matches(expected, value)) =>
        {
            return Err(format!("Invalid {}: unexpected type", at(path)));
        }
        _ => {}
    }
    if let Some(constant) = schema.get("const")
        && constant != value
    {
        return Err(format!("Invalid {}: expected {constant}", at(path)));
    }
    if let Some(values) = schema.get("enum").and_then(JsonValue::as_array)
        && !values.contains(value)
    {
        return Err(format!(
            "Invalid {}: expected one of {}",
            at(path),
            JsonValue::Array(values.clone())
        ));
    }
    if let Some(options) = schema.get("anyOf").and_then(JsonValue::as_array)
        && !options
            .iter()
            .any(|option| validate(option, value, path).is_ok())
    {
        return Err(format!("Invalid {}: no union member matches", at(path)));
    }
    if let Some(options) = schema.get("oneOf").and_then(JsonValue::as_array)
        && options
            .iter()
            .filter(|option| validate(option, value, path).is_ok())
            .count()
            != 1
    {
        return Err(format!(
            "Invalid {}: expected exactly one union member",
            at(path)
        ));
    }
    if let Some(parts) = schema.get("allOf").and_then(JsonValue::as_array) {
        for part in parts {
            validate(part, value, path)?;
        }
    }
    if let Some(text) = value.as_str() {
        let length = text.encode_utf16().count() as u64;
        if let Some(min) = schema.get("minLength").and_then(JsonValue::as_u64)
            && length < min
        {
            return Err(format!("Invalid {}: too short", at(path)));
        }
        if let Some(max) = schema.get("maxLength").and_then(JsonValue::as_u64)
            && length > max
        {
            return Err(format!("Invalid {}: too long", at(path)));
        }
        if let Some(pattern) = schema.get("pattern").and_then(JsonValue::as_str)
            && let Ok(regex) = Regex::new(pattern)
            && !regex.is_match(text)
        {
            return Err(format!("Invalid {}: does not match {pattern}", at(path)));
        }
    }
    if let Some(number) = value.as_f64() {
        let bound = |key: &str| schema.get(key).and_then(JsonValue::as_f64);
        if bound("minimum").is_some_and(|min| number < min)
            || bound("maximum").is_some_and(|max| number > max)
            || bound("exclusiveMinimum").is_some_and(|min| number <= min)
            || bound("exclusiveMaximum").is_some_and(|max| number >= max)
        {
            return Err(format!("Invalid {}: out of range", at(path)));
        }
    }
    if let Some(items) = value.as_array() {
        let count = items.len() as u64;
        if let Some(min) = schema.get("minItems").and_then(JsonValue::as_u64)
            && count < min
        {
            return Err(format!("Invalid {}: too few items", at(path)));
        }
        if let Some(max) = schema.get("maxItems").and_then(JsonValue::as_u64)
            && count > max
        {
            return Err(format!("Invalid {}: too many items", at(path)));
        }
        if let Some(item_schema) = schema.get("items") {
            for (index, item) in items.iter().enumerate() {
                validate(item_schema, item, &format!("{path}[{index}]"))?;
            }
        }
    }
    if let Some(object) = value.as_object() {
        let properties = schema.get("properties").and_then(JsonValue::as_object);
        if let Some(required) = schema.get("required").and_then(JsonValue::as_array) {
            for name in required.iter().filter_map(JsonValue::as_str) {
                if !object.contains_key(name) {
                    return Err(format!("Invalid input: {path}.{name} is required"));
                }
            }
        }
        for (name, child) in object {
            let child_path = format!("{path}.{name}");
            if let Some(names) = schema.get("propertyNames") {
                validate(names, &JsonValue::String(name.clone()), &child_path)?;
            }
            match properties.and_then(|properties| properties.get(name)) {
                Some(property) => validate(property, child, &child_path)?,
                None => {
                    if let Some(additional) = schema.get("additionalProperties") {
                        if additional == &JsonValue::Bool(false) {
                            return Err(format!("Invalid input: unrecognized key {name:?}"));
                        }
                        validate(additional, child, &child_path)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_the_zod_json_schema_subset() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "minLength": 1 },
                "mode": { "type": "string", "enum": ["a", "b"] },
                "count": { "type": "integer", "exclusiveMinimum": 0 },
                "message": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "array", "items": { "type": "string" }, "minItems": 1 },
                    ],
                },
            },
            "required": ["path"],
            "additionalProperties": false,
        });
        assert!(validate_json_schema(&schema, &json!({ "path": "a" })).is_ok());
        assert!(
            validate_json_schema(
                &schema,
                &json!({ "path": "a", "mode": "b", "count": 2, "message": ["x"] })
            )
            .is_ok()
        );
        for invalid in [
            json!({}),
            json!({ "path": "" }),
            json!({ "path": "a", "extra": 1 }),
            json!({ "path": "a", "mode": "c" }),
            json!({ "path": "a", "count": 0 }),
            json!({ "path": "a", "count": 1.5 }),
            json!({ "path": "a", "message": [] }),
        ] {
            assert!(
                validate_json_schema(&schema, &invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn loads_a_catalog_in_order() {
        let catalog = load_builtin_tool_catalog(
            r#"{
                "b": { "title": "B", "nativeName": "B", "toolUseKind": "edit",
                       "inputSchema": { "type": "object" } },
                "a": { "description": "A", "inputSchema": { "type": "object" } }
            }"#,
        );
        assert_eq!(catalog.keys().collect::<Vec<_>>(), ["b", "a"]);
        assert_eq!(catalog["b"].tool.title.as_deref(), Some("B"));
        assert_eq!(
            catalog["b"].tool_use_kind,
            Some(HarnessV1BuiltinToolUseKind::Edit)
        );
        assert_eq!(catalog["a"].tool.description.as_deref(), Some("A"));
    }
}
