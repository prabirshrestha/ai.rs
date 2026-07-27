use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::types::{
    ConstrainedSampling, ConstrainedSamplingConfig, ConstrainedSamplingStrict, Tool,
};
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GrammarFormat {
    Lark,
    Regex,
}

impl GrammarFormat {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Lark => "lark",
            Self::Regex => "regex",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GrammarConstrainedSampling {
    pub(crate) format: GrammarFormat,
    pub(crate) definition: String,
    pub(crate) input_property: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct GrammarToolInputJsonBuffer {
    input: String,
    started: bool,
    closed: bool,
}

pub(crate) fn get_grammar_tool_input<'a>(
    tool_name: &str,
    arguments: &'a Value,
    input_property: &str,
) -> Result<&'a str> {
    arguments
        .get(input_property)
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error::Validation(format!(
                "Grammar tool call \"{tool_name}\" requires argument \"{input_property}\" to be a string."
            ))
        })
}

pub(crate) fn append_grammar_tool_input_json_delta(
    buffer: &mut GrammarToolInputJsonBuffer,
    input_property: &str,
    next_input: &str,
    close: bool,
) -> Result<Option<String>> {
    if buffer.closed {
        if close && next_input == buffer.input {
            return Ok(None);
        }
        return Err(Error::Validation(format!(
            "grammar tool input for property \"{input_property}\" changed after it was closed"
        )));
    }
    let Some(input_delta) = next_input.strip_prefix(&buffer.input) else {
        return Err(Error::Validation(format!(
            "grammar tool input for property \"{input_property}\" changed non-monotonically"
        )));
    };
    if !close && input_delta.is_empty() {
        return Ok(None);
    }

    let mut delta = String::new();
    if !buffer.started {
        let property = serde_json::to_string(input_property)?;
        delta.push('{');
        delta.push_str(&property);
        delta.push_str(":\"");
        buffer.started = true;
    }
    let encoded_delta = serde_json::to_string(input_delta)?;
    delta.push_str(&encoded_delta[1..encoded_delta.len() - 1]);
    buffer.input = next_input.to_string();

    if close {
        delta.push_str("\"}");
        buffer.closed = true;
    }
    Ok(Some(delta))
}

fn infer_grammar_input_property(tool: &Tool) -> Result<String> {
    let schema = tool.parameters.as_object().ok_or_else(|| {
        Error::Validation(
            "grammar constrained sampling requires an object parameter schema".to_string(),
        )
    })?;
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err(Error::Validation(
            "grammar constrained sampling requires an object parameter schema".to_string(),
        ));
    }
    let required = schema.get("required").and_then(Value::as_array);
    let Some([required]) = required.map(Vec::as_slice) else {
        return Err(Error::Validation(
            "grammar constrained sampling requires exactly one required string property"
                .to_string(),
        ));
    };
    let Some(input_property) = required.as_str() else {
        return Err(Error::Validation(
            "grammar constrained sampling requires exactly one required string property"
                .to_string(),
        ));
    };
    let property = schema
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(input_property))
        .ok_or_else(|| {
            Error::Validation(format!(
                "grammar constrained sampling requires a properties entry for {input_property}"
            ))
        })?;
    if property.get("type").and_then(Value::as_str) != Some("string") {
        return Err(Error::Validation(format!(
            "grammar constrained sampling property {input_property} must have type string"
        )));
    }
    Ok(input_property.to_string())
}

pub(crate) fn resolve_json_schema_strict_sampling(
    tool: &Tool,
    supports_strict_mode: bool,
) -> Result<Option<bool>> {
    let Some(ConstrainedSampling::Config(ConstrainedSamplingConfig::JsonSchema { strict })) =
        &tool.constrained_sampling
    else {
        return Ok(None);
    };
    if supports_strict_mode {
        return Ok(Some(true));
    }
    if *strict == ConstrainedSamplingStrict::Require {
        return Err(Error::Validation(format!(
            "Tool \"{}\" requires JSON-schema constrained sampling, but strict tools are unsupported.",
            tool.name
        )));
    }
    Ok(None)
}

pub(crate) fn resolve_grammar_constrained_sampling(
    tool: &Tool,
    supports_openai_grammar_tools: bool,
) -> Result<Option<GrammarConstrainedSampling>> {
    let Some(ConstrainedSampling::Config(ConstrainedSamplingConfig::Grammar { variants })) =
        &tool.constrained_sampling
    else {
        return Ok(None);
    };
    if !supports_openai_grammar_tools {
        return Ok(None);
    }

    let lark = variants
        .openai_lark
        .as_deref()
        .filter(|definition| !definition.trim().is_empty());
    let regex = variants
        .openai_regex
        .as_deref()
        .filter(|definition| !definition.trim().is_empty());
    let (format, definition) = match (lark, regex) {
        (Some(definition), _) => (GrammarFormat::Lark, definition),
        (None, Some(definition)) => (GrammarFormat::Regex, definition),
        (None, None) => {
            return Err(Error::Validation(format!(
                "Tool \"{}\" cannot use grammar constrained sampling: no supported grammar variant was provided.",
                tool.name
            )));
        }
    };
    let input_property = infer_grammar_input_property(tool).map_err(|error| {
        Error::Validation(format!(
            "Tool \"{}\" cannot use grammar constrained sampling: {error}.",
            tool.name
        ))
    })?;
    Ok(Some(GrammarConstrainedSampling {
        format,
        definition: definition.to_string(),
        input_property,
    }))
}

pub(crate) fn create_grammar_tool_input_properties(
    tools: &[Tool],
    supports_openai_grammar_tools: bool,
) -> Result<HashMap<String, String>> {
    tools
        .iter()
        .filter_map(|tool| {
            resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
                .transpose()
                .map(|result| result.map(|grammar| (tool.name.clone(), grammar.input_property)))
        })
        .collect()
}

pub(crate) fn grammar_arguments(input_property: &str, input: impl Into<String>) -> Value {
    let mut arguments = Map::new();
    arguments.insert(input_property.to_string(), Value::String(input.into()));
    Value::Object(arguments)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            serde_json::json!({"payload": "a\"\nb"})
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
}
