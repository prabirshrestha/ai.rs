//! Port of `api/constrained-sampling.ts`: strict JSON-schema tool conversion
//! and OpenAI grammar tool helpers.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};

use crate::types::{
    ConstrainedSampling, ConstrainedSamplingConfig, ConstrainedSamplingStrict, Tool,
};
use crate::utils::text::js_trim;
use crate::{Error, Result};

/// `UnsupportedStrictJsonSchemaError`: the schema cannot be converted to
/// the strict subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedStrictJsonSchemaError(pub String);

impl std::fmt::Display for UnsupportedStrictJsonSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnsupportedStrictJsonSchemaError {}

impl From<UnsupportedStrictJsonSchemaError> for Error {
    fn from(error: UnsupportedStrictJsonSchemaError) -> Self {
        Error::message(error.0)
    }
}

type StrictResult<T> = std::result::Result<T, UnsupportedStrictJsonSchemaError>;

fn unsupported<T>(message: impl Into<String>) -> StrictResult<T> {
    Err(UnsupportedStrictJsonSchemaError(message.into()))
}

/// Returns true when a provider's strict mode rejects this schema keyword with this value.
pub type UnsupportedStrictSchemaKeywordCheck<'a> = &'a dyn Fn(&str, &Value) -> bool;

const UNSUPPORTED_STRICT_SCHEMA_KEYS: [&str; 16] = [
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

fn is_structured_schema(schema: &Value) -> bool {
    let Some(schema) = schema.as_object() else {
        return false;
    };
    let types: Vec<&Value> = match schema.get("type") {
        Some(Value::String(_)) => vec![&schema["type"]],
        Some(Value::Array(types)) => types.iter().collect(),
        _ => Vec::new(),
    };
    types
        .iter()
        .any(|value| value.as_str() == Some("object") || value.as_str() == Some("array"))
        || schema.contains_key("properties")
        || schema.contains_key("items")
}

fn schema_allows_null(schema: &Value) -> bool {
    let Some(schema) = schema.as_object() else {
        return false;
    };
    match schema.get("type") {
        Some(Value::String(value)) if value == "null" => return true,
        Some(Value::Array(types)) if types.iter().any(|value| value.as_str() == Some("null")) => {
            return true;
        }
        _ => {}
    }
    if schema.get("const") == Some(&Value::Null)
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.contains(&Value::Null))
    {
        return true;
    }
    schema
        .get("anyOf")
        .and_then(Value::as_array)
        .is_some_and(|variants| variants.iter().any(schema_allows_null))
}

fn make_json_schema_node_strict(
    schema: &mut Value,
    is_unsupported_keyword: Option<UnsupportedStrictSchemaKeywordCheck>,
) -> StrictResult<()> {
    let Some(node) = schema.as_object_mut() else {
        return unsupported("boolean schemas are unsupported");
    };
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS {
        if node.contains_key(key) {
            return unsupported(format!("{key} schemas are unsupported"));
        }
    }
    if let Some(is_unsupported_keyword) = is_unsupported_keyword {
        for (key, value) in node.iter() {
            if is_unsupported_keyword(key, value) {
                return unsupported(format!("{key}: {value} is unsupported"));
            }
        }
    }

    if let Some(any_of) = node.get_mut("anyOf") {
        let Some(variants) = any_of
            .as_array_mut()
            .filter(|variants| !variants.is_empty())
        else {
            return unsupported("anyOf must contain at least one schema");
        };
        for variant in variants {
            if is_structured_schema(variant) {
                return unsupported("object and array unions are unsupported");
            }
            make_json_schema_node_strict(variant, is_unsupported_keyword)?;
        }
    }

    if let Some(items) = node.get_mut("items") {
        if items.is_array() {
            return unsupported("tuple schemas are unsupported");
        }
        make_json_schema_node_strict(items, is_unsupported_keyword)?;
    }

    let is_object_schema = node.get("type").and_then(Value::as_str) == Some("object");
    if node.contains_key("properties") && !is_object_schema {
        return unsupported("properties require type object");
    }
    if !is_object_schema {
        return Ok(());
    }
    if node
        .get("additionalProperties")
        .is_some_and(|value| *value != Value::Bool(false))
    {
        return unsupported("schema-valued or true additionalProperties is unsupported");
    }
    if node
        .get("properties")
        .is_some_and(|properties| !properties.is_object())
    {
        return unsupported("object properties must be a schema map");
    }
    if let Some(required) = node.get("required")
        && !required
            .as_array()
            .is_some_and(|keys| keys.iter().all(Value::is_string))
    {
        return unsupported("object required must be a string array");
    }

    let required: HashSet<String> = node
        .get("required")
        .and_then(Value::as_array)
        .map(|keys| {
            keys.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mut detached = Map::new();
    let properties = match node.get_mut("properties") {
        Some(Value::Object(properties)) => properties,
        _ => &mut detached,
    };
    let property_names: Vec<String> = properties.keys().cloned().collect();
    if required.iter().any(|key| !property_names.contains(key)) {
        return unsupported("required contains an unknown property");
    }
    for key in &property_names {
        let property = properties
            .get_mut(key)
            .expect("property names come from the map");
        make_json_schema_node_strict(property, is_unsupported_keyword)?;
        if !required.contains(key) && !schema_allows_null(property) {
            let original = property.take();
            *property = json!({ "anyOf": [original, { "type": "null" }] });
        }
    }
    node.insert(
        "required".to_string(),
        Value::Array(property_names.into_iter().map(Value::String).collect()),
    );
    node.insert("additionalProperties".to_string(), Value::Bool(false));
    Ok(())
}

/// Convert a tool schema to the strict subset expected by provider constrained sampling.
pub fn make_strict_json_schema(
    schema: &Value,
    is_unsupported_keyword: Option<UnsupportedStrictSchemaKeywordCheck>,
) -> StrictResult<Value> {
    let mut cloned = schema.clone();
    if !cloned.is_object() {
        return unsupported("root schema must have type object");
    }
    make_json_schema_node_strict(&mut cloned, is_unsupported_keyword)?;
    if cloned.get("type").and_then(Value::as_str) != Some("object") {
        return unsupported("root schema must have type object");
    }
    Ok(cloned)
}

pub fn get_json_schema_tool_parameters(tool: &Tool, strict: Option<bool>) -> Result<Value> {
    if strict == Some(true) {
        Ok(make_strict_json_schema(&tool.parameters, None)?)
    } else {
        Ok(tool.parameters.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarSyntax {
    Lark,
    Regex,
}

impl GrammarSyntax {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lark => "lark",
            Self::Regex => "regex",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarConstrainedSampling {
    pub format: GrammarSyntax,
    pub definition: String,
    pub input_property: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GrammarToolInputJsonBuffer {
    pub input: String,
    pub started: bool,
    pub closed: bool,
}

pub fn get_grammar_tool_input<'a>(
    tool_name: &str,
    arguments: &'a Value,
    input_property: &str,
) -> Result<&'a str> {
    arguments
        .get(input_property)
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error::message(format!(
                "Grammar tool call \"{tool_name}\" requires argument \"{input_property}\" to be a string."
            ))
        })
}

pub fn append_grammar_tool_input_json_delta(
    buffer: &mut GrammarToolInputJsonBuffer,
    input_property: &str,
    next_input: &str,
    close: bool,
) -> Result<Option<String>> {
    if buffer.closed {
        if close && next_input == buffer.input {
            return Ok(None);
        }
        return Err(Error::message(format!(
            "grammar tool input for property \"{input_property}\" changed after it was closed"
        )));
    }
    let Some(input_delta) = next_input.strip_prefix(buffer.input.as_str()) else {
        return Err(Error::message(format!(
            "grammar tool input for property \"{input_property}\" changed non-monotonically"
        )));
    };
    if !close && input_delta.is_empty() {
        return Ok(None);
    }

    let mut delta = String::new();
    if !buffer.started {
        delta.push('{');
        delta.push_str(&serde_json::to_string(input_property)?);
        delta.push_str(":\"");
        buffer.started = true;
    }
    let encoded = serde_json::to_string(input_delta)?;
    delta.push_str(&encoded[1..encoded.len() - 1]);
    buffer.input = next_input.to_string();

    if close {
        delta.push_str("\"}");
        buffer.closed = true;
    }
    Ok(Some(delta))
}

fn infer_grammar_input_property(tool: &Tool) -> std::result::Result<String, String> {
    let schema = &tool.parameters;
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err("grammar constrained sampling requires an object parameter schema".to_string());
    }
    let input_property = match schema.get("required").and_then(Value::as_array) {
        Some(required) if required.len() == 1 && required[0].is_string() => {
            required[0].as_str().unwrap_or_default().to_string()
        }
        _ => {
            return Err(
                "grammar constrained sampling requires exactly one required string property"
                    .to_string(),
            );
        }
    };
    let Some(property) = schema
        .get("properties")
        .and_then(|properties| properties.get(&input_property))
        .filter(|property| !property.is_null())
    else {
        return Err(format!(
            "grammar constrained sampling requires a properties entry for {input_property}"
        ));
    };
    if property.get("type").and_then(Value::as_str) != Some("string") {
        return Err(format!(
            "grammar constrained sampling property {input_property} must have type string"
        ));
    }
    Ok(input_property)
}

fn constrained_sampling_config(tool: &Tool) -> Option<&ConstrainedSamplingConfig> {
    match &tool.constrained_sampling {
        Some(ConstrainedSampling::Config(config)) => Some(config),
        _ => None,
    }
}

/// Decide whether a JSON-schema tool is sent in strict mode. `is_unsupported_keyword` lets a provider
/// reject extra keywords its strict mode does not accept, so "prefer" tools fall back to non-strict.
pub fn resolve_json_schema_strict_sampling(
    tool: &Tool,
    supports_strict_mode: bool,
    is_unsupported_keyword: Option<UnsupportedStrictSchemaKeywordCheck>,
) -> Result<Option<bool>> {
    let Some(ConstrainedSamplingConfig::JsonSchema { strict }) = constrained_sampling_config(tool)
    else {
        return Ok(None);
    };

    if supports_strict_mode {
        return match make_strict_json_schema(&tool.parameters, is_unsupported_keyword) {
            Ok(_) => Ok(Some(true)),
            Err(error) => {
                if *strict != ConstrainedSamplingStrict::Require {
                    return Ok(None);
                }
                Err(Error::message(format!(
                    "Tool \"{}\" requires JSON-schema constrained sampling, but {error}.",
                    tool.name
                )))
            }
        };
    }
    if *strict == ConstrainedSamplingStrict::Require {
        return Err(Error::message(format!(
            "Tool \"{}\" requires JSON-schema constrained sampling, but strict tools are unsupported.",
            tool.name
        )));
    }
    Ok(None)
}

pub fn resolve_grammar_constrained_sampling(
    tool: &Tool,
    supports_openai_grammar_tools: bool,
) -> Result<Option<GrammarConstrainedSampling>> {
    let Some(ConstrainedSamplingConfig::Grammar { variants }) = constrained_sampling_config(tool)
    else {
        return Ok(None);
    };

    if !supports_openai_grammar_tools {
        return Ok(None);
    }

    let lark_definition = variants
        .openai_lark
        .as_deref()
        .filter(|definition| !js_trim(definition).is_empty());
    let regex_definition = variants
        .openai_regex
        .as_deref()
        .filter(|definition| !js_trim(definition).is_empty());
    let (format, definition) = match (lark_definition, regex_definition) {
        (Some(definition), _) => (GrammarSyntax::Lark, definition),
        (None, Some(definition)) => (GrammarSyntax::Regex, definition),
        (None, None) => {
            return Err(Error::message(format!(
                "Tool \"{}\" cannot use grammar constrained sampling: no supported grammar variant was provided.",
                tool.name
            )));
        }
    };

    let input_property = infer_grammar_input_property(tool).map_err(|message| {
        Error::message(format!(
            "Tool \"{}\" cannot use grammar constrained sampling: {message}.",
            tool.name
        ))
    })?;
    Ok(Some(GrammarConstrainedSampling {
        format,
        definition: definition.to_string(),
        input_property,
    }))
}

pub fn create_grammar_tool_input_properties(
    tools: Option<&[Tool]>,
    supports_openai_grammar_tools: bool,
) -> Result<HashMap<String, String>> {
    let mut properties = HashMap::new();
    for tool in tools.into_iter().flatten() {
        if let Some(grammar) =
            resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)?
        {
            properties.insert(tool.name.clone(), grammar.input_property);
        }
    }
    Ok(properties)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(parameters: Value, strict: ConstrainedSamplingStrict) -> Tool {
        Tool {
            name: "sample_tool".to_string(),
            description: "Sample tool".to_string(),
            parameters,
            constrained_sampling: Some(ConstrainedSampling::Config(
                ConstrainedSamplingConfig::JsonSchema { strict },
            )),
        }
    }

    // constrained-sampling.test.ts
    #[test]
    fn derives_strict_provider_schemas_without_changing_tool_definitions() {
        let parameters = json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "offset": { "type": "number" },
                "metadata": {
                    "type": "object",
                    "properties": { "enabled": { "type": "boolean" } },
                    "required": []
                },
                "nullable": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["path", "metadata"]
        });

        let strict = make_strict_json_schema(&parameters, None).unwrap();

        assert!(parameters.get("additionalProperties").is_none());
        assert_eq!(parameters["required"], json!(["path", "metadata"]));
        assert_eq!(strict["additionalProperties"], json!(false));
        assert_eq!(
            strict["required"],
            json!(["path", "offset", "metadata", "nullable"])
        );
        assert_eq!(
            strict["properties"]["offset"],
            json!({ "anyOf": [{ "type": "number" }, { "type": "null" }] })
        );
        let metadata = &strict["properties"]["metadata"];
        assert_eq!(metadata["additionalProperties"], json!(false));
        assert_eq!(metadata["required"], json!(["enabled"]));
        assert_eq!(
            metadata["properties"]["enabled"],
            json!({ "anyOf": [{ "type": "boolean" }, { "type": "null" }] })
        );
        assert_eq!(
            strict["properties"]["nullable"],
            json!({ "anyOf": [{ "type": "string" }, { "type": "null" }] })
        );
    }

    #[test]
    fn falls_back_or_rejects_schemas_that_cannot_be_safely_converted() {
        let cases = [
            (
                json!({
                    "type": "object",
                    "properties": {
                        "metadata": { "type": "object", "properties": {}, "additionalProperties": { "type": "string" } }
                    },
                    "required": ["metadata"]
                }),
                "additionalProperties is unsupported",
            ),
            (
                json!({
                    "allOf": [
                        { "type": "object", "properties": { "a": { "type": "string" } }, "required": ["a"] },
                        { "type": "object", "properties": { "b": { "type": "number" } }, "required": ["b"] }
                    ]
                }),
                "allOf schemas are unsupported",
            ),
            (
                json!({
                    "type": "object",
                    "properties": {
                        "value": { "anyOf": [
                            { "type": "object", "properties": { "nested": { "type": "string" } }, "required": ["nested"] },
                            { "type": "null" }
                        ] }
                    },
                    "required": ["value"]
                }),
                "object and array unions are unsupported",
            ),
            (
                json!({
                    "type": "object",
                    "properties": { "child": { "$ref": "https://example.com/child.json" } },
                    "required": ["child"]
                }),
                "$ref schemas are unsupported",
            ),
        ];

        for (parameters, error) in cases {
            let message = make_strict_json_schema(&parameters, None)
                .unwrap_err()
                .to_string();
            assert!(message.contains(error), "{message} should contain {error}");
            let prefer = tool(parameters.clone(), ConstrainedSamplingStrict::Prefer);
            assert_eq!(
                resolve_json_schema_strict_sampling(&prefer, true, None).unwrap(),
                None
            );
            let require = tool(parameters, ConstrainedSamplingStrict::Require);
            let message = resolve_json_schema_strict_sampling(&require, true, None)
                .unwrap_err()
                .to_string();
            assert!(message.contains(error), "{message} should contain {error}");
        }
    }

    #[test]
    fn require_without_strict_support_is_rejected() {
        let require = tool(
            json!({ "type": "object", "properties": {} }),
            ConstrainedSamplingStrict::Require,
        );
        assert_eq!(
            resolve_json_schema_strict_sampling(&require, false, None)
                .unwrap_err()
                .to_string(),
            "Tool \"sample_tool\" requires JSON-schema constrained sampling, but strict tools are unsupported."
        );
    }

    #[test]
    fn keeps_grammar_input_json_deltas_append_only() {
        let mut buffer = GrammarToolInputJsonBuffer::default();
        let first = append_grammar_tool_input_json_delta(&mut buffer, "payload", "a\"", false)
            .unwrap()
            .unwrap();
        let second = append_grammar_tool_input_json_delta(&mut buffer, "payload", "a\"\nb", true)
            .unwrap()
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&format!("{first}{second}")).unwrap(),
            json!({ "payload": "a\"\nb" })
        );
        assert_eq!(
            append_grammar_tool_input_json_delta(&mut buffer, "payload", "a\"\nb", true).unwrap(),
            None
        );
        assert_eq!(
            append_grammar_tool_input_json_delta(&mut buffer, "payload", "changed", true)
                .unwrap_err()
                .to_string(),
            "grammar tool input for property \"payload\" changed after it was closed"
        );
    }

    #[test]
    fn resolves_grammar_variants() {
        let grammar_tool = Tool {
            name: "sample_tool".to_string(),
            description: "Sample tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "payload": { "type": "string" } },
                "required": ["payload"]
            }),
            constrained_sampling: Some(ConstrainedSampling::Config(
                ConstrainedSamplingConfig::Grammar {
                    variants: crate::types::GrammarVariants {
                        openai_lark: Some("start: /[a-z]+/".to_string()),
                        openai_regex: None,
                    },
                },
            )),
        };
        assert_eq!(
            resolve_grammar_constrained_sampling(&grammar_tool, true).unwrap(),
            Some(GrammarConstrainedSampling {
                format: GrammarSyntax::Lark,
                definition: "start: /[a-z]+/".to_string(),
                input_property: "payload".to_string(),
            })
        );
        assert_eq!(
            resolve_grammar_constrained_sampling(&grammar_tool, false).unwrap(),
            None
        );
        let mut empty = grammar_tool.clone();
        empty.constrained_sampling = Some(ConstrainedSampling::Config(
            ConstrainedSamplingConfig::Grammar {
                variants: Default::default(),
            },
        ));
        assert_eq!(
            resolve_grammar_constrained_sampling(&empty, true)
                .unwrap_err()
                .to_string(),
            "Tool \"sample_tool\" cannot use grammar constrained sampling: no supported grammar variant was provided."
        );
        assert_eq!(
            get_grammar_tool_input("sample_tool", &json!({ "payload": 42 }), "payload")
                .unwrap_err()
                .to_string(),
            "Grammar tool call \"sample_tool\" requires argument \"payload\" to be a string."
        );
    }
}
