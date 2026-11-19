//! [rust] The Zod schemas the adapters declare their built-in tool inputs
//! with (`z.looseObject`, `z.string()`, ...), reduced to what `asSchema`
//! produces: the draft-07 JSON Schema plus a validator.

#![allow(dead_code)]

use serde_json::{Value as JsonValue, json};

use crate::harness::sdk::tool::Schema;
use crate::harness::utils::plain_error;

/// One `z.looseObject` property.
#[derive(Debug, Clone)]
pub(crate) struct Field {
    pub name: &'static str,
    pub schema: JsonValue,
    pub optional: bool,
}

/// A required property.
pub(crate) fn required(name: &'static str, schema: JsonValue) -> Field {
    Field {
        name,
        schema,
        optional: false,
    }
}

/// A `.optional()` property.
pub(crate) fn optional(name: &'static str, schema: JsonValue) -> Field {
    Field {
        name,
        schema,
        optional: true,
    }
}

/// `z.string()`.
pub(crate) fn string() -> JsonValue {
    json!({ "type": "string" })
}

/// `z.number()`.
pub(crate) fn number() -> JsonValue {
    json!({ "type": "number" })
}

/// `z.boolean()`.
pub(crate) fn boolean() -> JsonValue {
    json!({ "type": "boolean" })
}

/// `z.unknown()`.
pub(crate) fn unknown() -> JsonValue {
    json!({})
}

/// `z.literal(value)` for a string.
pub(crate) fn literal(value: &str) -> JsonValue {
    json!({ "type": "string", "const": value })
}

/// `z.enum(values)`.
pub(crate) fn enumeration(values: &[&str]) -> JsonValue {
    json!({ "type": "string", "enum": values })
}

/// `z.number().int().nonnegative()`.
pub(crate) fn nonnegative_integer() -> JsonValue {
    json!({ "type": "integer", "minimum": 0, "maximum": 9_007_199_254_740_991_u64 })
}

/// `z.array(items)`.
pub(crate) fn array(items: JsonValue) -> JsonValue {
    json!({ "type": "array", "items": items })
}

/// A nested `z.looseObject(shape)` as a JSON Schema value.
pub(crate) fn loose_object_json(fields: &[Field]) -> JsonValue {
    let properties: serde_json::Map<String, JsonValue> = fields
        .iter()
        .map(|field| (field.name.to_string(), field.schema.clone()))
        .collect();
    let required: Vec<&str> = fields
        .iter()
        .filter(|field| !field.optional)
        .map(|field| field.name)
        .collect();
    let mut schema = json!({
        "type": "object",
        "properties": properties,
        "additionalProperties": {},
    });
    if !required.is_empty() {
        schema["required"] = json!(required);
    }
    schema
}

/// `z.looseObject(shape)` as a top-level [`Schema`] that validates its input.
pub(crate) fn loose_object(fields: &[Field]) -> Schema {
    let mut schema = loose_object_json(fields);
    schema["$schema"] = json!("http://json-schema.org/draft-07/schema#");
    let validated = schema.clone();
    Schema {
        json_schema: schema,
        validate: None,
    }
    .with_validate(move |value| {
        validate(&validated, value, "")
            .map(|()| value.clone())
            .map_err(plain_error)
    })
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

fn validate(schema: &JsonValue, value: &JsonValue, path: &str) -> Result<(), String> {
    let at = if path.is_empty() { "input" } else { path };
    if let Some(expected) = schema.get("type").and_then(JsonValue::as_str)
        && !type_matches(expected, value)
    {
        return Err(format!("Invalid {at}: expected {expected}"));
    }
    if let Some(constant) = schema.get("const")
        && constant != value
    {
        return Err(format!("Invalid {at}: expected {constant}"));
    }
    if let Some(values) = schema.get("enum").and_then(JsonValue::as_array)
        && !values.contains(value)
    {
        return Err(format!(
            "Invalid {at}: expected one of {}",
            JsonValue::Array(values.clone())
        ));
    }
    if let (Some(minimum), Some(number)) = (
        schema.get("minimum").and_then(JsonValue::as_f64),
        value.as_f64(),
    ) && number < minimum
    {
        return Err(format!("Invalid {at}: expected a number >= {minimum}"));
    }
    if let (Some(maximum), Some(number)) = (
        schema.get("maximum").and_then(JsonValue::as_f64),
        value.as_f64(),
    ) && number > maximum
    {
        return Err(format!("Invalid {at}: expected a number <= {maximum}"));
    }
    if let (Some(items), Some(values)) = (schema.get("items"), value.as_array()) {
        for (index, item) in values.iter().enumerate() {
            validate(items, item, &format!("{path}[{index}]"))?;
        }
    }
    if let Some(object) = value.as_object() {
        let properties = schema.get("properties").and_then(JsonValue::as_object);
        let required = schema
            .get("required")
            .and_then(JsonValue::as_array)
            .cloned()
            .unwrap_or_default();
        for name in required.iter().filter_map(JsonValue::as_str) {
            if !object.contains_key(name) {
                return Err(format!("Invalid input: {path}.{name} is required"));
            }
        }
        if let Some(properties) = properties {
            for (name, property) in properties {
                if let Some(child) = object.get(name) {
                    validate(property, child, &format!("{path}.{name}"))?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_the_zod_json_schema_and_validates() {
        let schema = loose_object(&[required("path", string()), optional("count", number())]);
        assert_eq!(
            schema.json_schema,
            json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "count": { "type": "number" },
                },
                "required": ["path"],
                "additionalProperties": {},
            })
        );
        let validate = schema.validate.unwrap();
        assert!(validate(&json!({ "path": "a", "extra": true })).is_ok());
        assert!(validate(&json!({})).is_err());
        assert!(validate(&json!({ "path": 1 })).is_err());
        assert!(validate(&json!({ "path": "a", "count": "x" })).is_err());
    }
}
