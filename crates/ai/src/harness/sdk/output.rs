//! Port of `ai` `Output` (`output.ts`), `parsePartialJson` and `fixJson`.
//!
//! [rust] `Output` is a trait object. `parsePartialOutput` is synchronous:
//! every built-in implementation is, and the harness runs it inline while it
//! enqueues stream parts. `responseFormat` is the harness response format,
//! which has the same shape as `LanguageModelV4CallOptions['responseFormat']`.

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::harness::errors::{BoxError, HarnessResult};
use crate::harness::provider::{JsonValue, TypeValidationError, safe_parse_json, validate_types};
use crate::harness::sdk::content::{FinishReason, LanguageModelResponseMetadata};
use crate::harness::sdk::error_value::get_error_message;
use crate::harness::sdk::tool::Schema;
use crate::harness::sdk::usage::LanguageModelUsage;
use crate::harness::spec::response_format::{HarnessV1JsonSchema, HarnessV1ResponseFormat};

/// The `context` argument of `parseCompleteOutput`.
#[derive(Debug, Clone)]
pub struct OutputContext {
    pub response: LanguageModelResponseMetadata,
    pub usage: LanguageModelUsage,
    pub finish_reason: FinishReason,
}

/// `{ partial: PARTIAL }`.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialOutput {
    pub partial: JsonValue,
}

/// `createElementStreamTransform()`: maps each partial output to the
/// elements to publish.
pub trait ElementStreamTransform: Send {
    /// Elements to publish for this partial output, or an error that ends
    /// the stream.
    fn transform(&mut self, partial_output: Option<&JsonValue>)
    -> Result<Vec<JsonValue>, BoxError>;
}

/// `Output<OUTPUT, PARTIAL, ELEMENT>`.
pub trait Output: Send + Sync {
    /// `name`.
    fn name(&self) -> &str;
    /// `responseFormat`.
    fn response_format(&self) -> BoxFuture<'static, HarnessResult<HarnessV1ResponseFormat>>;
    /// `parseCompleteOutput({ text }, context)`.
    fn parse_complete_output(&self, text: &str, context: OutputContext)
    -> HarnessResult<JsonValue>;
    /// `parsePartialOutput({ text })`.
    fn parse_partial_output(&self, text: &str) -> Option<PartialOutput>;
    /// `createElementStreamTransform()`.
    fn create_element_stream_transform(&self) -> Option<Box<dyn ElementStreamTransform>> {
        None
    }
}

/// A shared [`Output`].
pub type SharedOutput = Arc<dyn Output>;

/// `NoObjectGeneratedError` (`AI_NoObjectGeneratedError`).
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct NoObjectGeneratedError {
    pub message: String,
    pub cause_message: Option<String>,
    pub text: Option<String>,
    pub response: Option<LanguageModelResponseMetadata>,
    pub usage: Option<LanguageModelUsage>,
    pub finish_reason: Option<FinishReason>,
}

impl NoObjectGeneratedError {
    fn boxed(
        message: &str,
        cause: &(dyn std::error::Error + 'static),
        text: &str,
        context: OutputContext,
    ) -> BoxError {
        Box::new(Self {
            message: message.into(),
            cause_message: Some(get_error_message(cause)),
            text: Some(text.into()),
            response: Some(context.response),
            usage: Some(context.usage),
            finish_reason: Some(context.finish_reason),
        })
    }
}

const COULD_NOT_PARSE: &str = "No object generated: could not parse the response.";
const DID_NOT_MATCH: &str = "No object generated: response did not match schema.";

/// The `state` of `parsePartialJson`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParsePartialJsonState {
    UndefinedInput,
    SuccessfulParse,
    RepairedParse,
    FailedParse,
}

/// `parsePartialJson(jsonText)`.
pub fn parse_partial_json(json_text: Option<&str>) -> (Option<JsonValue>, ParsePartialJsonState) {
    let Some(text) = json_text else {
        return (None, ParsePartialJsonState::UndefinedInput);
    };
    if let Ok(value) = safe_parse_json(text, None) {
        return (Some(value), ParsePartialJsonState::SuccessfulParse);
    }
    if let Ok(value) = safe_parse_json(&fix_json(text), None) {
        return (Some(value), ParsePartialJsonState::RepairedParse);
    }
    (None, ParsePartialJsonState::FailedParse)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Root,
    Finish,
    InsideString,
    InsideStringEscape,
    InsideStringUnicodeEscape,
    InsideLiteral,
    InsideNumber,
    InsideObjectStart,
    InsideObjectKey,
    InsideObjectAfterKey,
    InsideObjectBeforeValue,
    InsideObjectAfterValue,
    InsideObjectAfterComma,
    InsideArrayStart,
    InsideArrayAfterValue,
    InsideArrayAfterComma,
}

/// `fixJson(input)`: close a truncated JSON text.
pub fn fix_json(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut stack = vec![State::Root];
    let mut last_valid: isize = -1;
    let mut literal_start: usize = 0;
    let mut unicode_digits = 0;

    fn value_start(
        stack: &mut Vec<State>,
        last_valid: &mut isize,
        literal_start: &mut usize,
        c: char,
        i: usize,
        swap: State,
    ) {
        let push = |stack: &mut Vec<State>, inner: State| {
            stack.pop();
            stack.push(swap);
            stack.push(inner);
        };
        match c {
            '"' => {
                *last_valid = i as isize;
                push(stack, State::InsideString);
            }
            'f' | 't' | 'n' => {
                *last_valid = i as isize;
                *literal_start = i;
                push(stack, State::InsideLiteral);
            }
            '-' => push(stack, State::InsideNumber),
            '0'..='9' => {
                *last_valid = i as isize;
                push(stack, State::InsideNumber);
            }
            '{' => {
                *last_valid = i as isize;
                push(stack, State::InsideObjectStart);
            }
            '[' => {
                *last_valid = i as isize;
                push(stack, State::InsideArrayStart);
            }
            _ => {}
        }
    }

    fn after_object_value(stack: &mut Vec<State>, last_valid: &mut isize, c: char, i: usize) {
        match c {
            ',' => {
                stack.pop();
                stack.push(State::InsideObjectAfterComma);
            }
            '}' => {
                *last_valid = i as isize;
                stack.pop();
            }
            _ => {}
        }
    }

    fn after_array_value(stack: &mut Vec<State>, last_valid: &mut isize, c: char, i: usize) {
        match c {
            ',' => {
                stack.pop();
                stack.push(State::InsideArrayAfterComma);
            }
            ']' => {
                *last_valid = i as isize;
                stack.pop();
            }
            _ => {}
        }
    }

    for (i, &c) in chars.iter().enumerate() {
        let current = *stack.last().unwrap_or(&State::Finish);
        match current {
            State::Root => value_start(
                &mut stack,
                &mut last_valid,
                &mut literal_start,
                c,
                i,
                State::Finish,
            ),
            State::InsideObjectStart => match c {
                '"' => {
                    stack.pop();
                    stack.push(State::InsideObjectKey);
                }
                '}' => {
                    last_valid = i as isize;
                    stack.pop();
                }
                _ => {}
            },
            State::InsideObjectAfterComma => {
                if c == '"' {
                    stack.pop();
                    stack.push(State::InsideObjectKey);
                }
            }
            State::InsideObjectKey => {
                if c == '"' {
                    stack.pop();
                    stack.push(State::InsideObjectAfterKey);
                }
            }
            State::InsideObjectAfterKey => {
                if c == ':' {
                    stack.pop();
                    stack.push(State::InsideObjectBeforeValue);
                }
            }
            State::InsideObjectBeforeValue => value_start(
                &mut stack,
                &mut last_valid,
                &mut literal_start,
                c,
                i,
                State::InsideObjectAfterValue,
            ),
            State::InsideObjectAfterValue => after_object_value(&mut stack, &mut last_valid, c, i),
            State::InsideString => match c {
                '"' => {
                    stack.pop();
                    last_valid = i as isize;
                }
                '\\' => stack.push(State::InsideStringEscape),
                _ => last_valid = i as isize,
            },
            State::InsideArrayStart => match c {
                ']' => {
                    last_valid = i as isize;
                    stack.pop();
                }
                _ => value_start(
                    &mut stack,
                    &mut last_valid,
                    &mut literal_start,
                    c,
                    i,
                    State::InsideArrayAfterValue,
                ),
            },
            State::InsideArrayAfterValue => match c {
                ',' => {
                    stack.pop();
                    stack.push(State::InsideArrayAfterComma);
                }
                ']' => {
                    last_valid = i as isize;
                    stack.pop();
                }
                _ => last_valid = i as isize,
            },
            State::InsideArrayAfterComma => value_start(
                &mut stack,
                &mut last_valid,
                &mut literal_start,
                c,
                i,
                State::InsideArrayAfterValue,
            ),
            State::InsideStringEscape => {
                stack.pop();
                if c == 'u' {
                    unicode_digits = 0;
                    stack.push(State::InsideStringUnicodeEscape);
                } else {
                    last_valid = i as isize;
                }
            }
            State::InsideStringUnicodeEscape => {
                if c.is_ascii_hexdigit() {
                    unicode_digits += 1;
                    if unicode_digits == 4 {
                        stack.pop();
                        last_valid = i as isize;
                    }
                }
            }
            State::InsideNumber => match c {
                '0'..='9' => last_valid = i as isize,
                'e' | 'E' | '-' | '.' => {}
                ',' => {
                    stack.pop();
                    if stack.last() == Some(&State::InsideArrayAfterValue) {
                        after_array_value(&mut stack, &mut last_valid, c, i);
                    }
                    if stack.last() == Some(&State::InsideObjectAfterValue) {
                        after_object_value(&mut stack, &mut last_valid, c, i);
                    }
                }
                '}' => {
                    stack.pop();
                    if stack.last() == Some(&State::InsideObjectAfterValue) {
                        after_object_value(&mut stack, &mut last_valid, c, i);
                    }
                }
                ']' => {
                    stack.pop();
                    if stack.last() == Some(&State::InsideArrayAfterValue) {
                        after_array_value(&mut stack, &mut last_valid, c, i);
                    }
                }
                _ => {
                    stack.pop();
                }
            },
            State::InsideLiteral => {
                let partial: String = chars[literal_start..=i].iter().collect();
                if !"false".starts_with(&partial)
                    && !"true".starts_with(&partial)
                    && !"null".starts_with(&partial)
                {
                    stack.pop();
                    if stack.last() == Some(&State::InsideObjectAfterValue) {
                        after_object_value(&mut stack, &mut last_valid, c, i);
                    } else if stack.last() == Some(&State::InsideArrayAfterValue) {
                        after_array_value(&mut stack, &mut last_valid, c, i);
                    }
                } else {
                    last_valid = i as isize;
                }
            }
            State::Finish => {}
        }
    }

    let mut result: String = chars[..((last_valid + 1) as usize)].iter().collect();
    for state in stack.iter().rev() {
        match state {
            State::InsideString => result.push('"'),
            State::InsideObjectKey
            | State::InsideObjectAfterKey
            | State::InsideObjectAfterComma
            | State::InsideObjectStart
            | State::InsideObjectBeforeValue
            | State::InsideObjectAfterValue => result.push('}'),
            State::InsideArrayStart
            | State::InsideArrayAfterComma
            | State::InsideArrayAfterValue => result.push(']'),
            State::InsideLiteral => {
                let partial: String = chars[literal_start..].iter().collect();
                for literal in ["true", "false", "null"] {
                    if literal.starts_with(&partial) {
                        result.push_str(&literal[partial.len()..]);
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    result
}

/// `Output.text()`.
pub fn text() -> SharedOutput {
    struct Text;
    impl Output for Text {
        fn name(&self) -> &str {
            "text"
        }
        fn response_format(&self) -> BoxFuture<'static, HarnessResult<HarnessV1ResponseFormat>> {
            Box::pin(async { Ok(HarnessV1ResponseFormat::Text) })
        }
        fn parse_complete_output(&self, text: &str, _: OutputContext) -> HarnessResult<JsonValue> {
            Ok(JsonValue::String(text.into()))
        }
        fn parse_partial_output(&self, text: &str) -> Option<PartialOutput> {
            Some(PartialOutput {
                partial: JsonValue::String(text.into()),
            })
        }
    }
    Arc::new(Text)
}

fn schema_object(schema: &JsonValue) -> Option<HarnessV1JsonSchema> {
    schema.as_object().cloned()
}

/// `Output.object({ schema, name, description })`.
pub fn object(schema: Schema, name: Option<String>, description: Option<String>) -> SharedOutput {
    struct Object {
        schema: Schema,
        name: Option<String>,
        description: Option<String>,
    }
    impl Output for Object {
        fn name(&self) -> &str {
            "object"
        }
        fn response_format(&self) -> BoxFuture<'static, HarnessResult<HarnessV1ResponseFormat>> {
            let format = HarnessV1ResponseFormat::Json {
                schema: schema_object(&self.schema.json_schema),
                name: self.name.clone(),
                description: self.description.clone(),
            };
            Box::pin(async move { Ok(format) })
        }
        fn parse_complete_output(
            &self,
            text: &str,
            context: OutputContext,
        ) -> HarnessResult<JsonValue> {
            let value = safe_parse_json(text, None).map_err(|e| {
                NoObjectGeneratedError::boxed(COULD_NOT_PARSE, &e, text, context.clone())
            })?;
            validate_types(&value, &self.schema, None)
                .map_err(|e| NoObjectGeneratedError::boxed(DID_NOT_MATCH, &e, text, context))
        }
        fn parse_partial_output(&self, text: &str) -> Option<PartialOutput> {
            match parse_partial_json(Some(text)) {
                (
                    Some(value),
                    ParsePartialJsonState::SuccessfulParse | ParsePartialJsonState::RepairedParse,
                ) => Some(PartialOutput { partial: value }),
                _ => None,
            }
        }
    }
    Arc::new(Object {
        schema,
        name,
        description,
    })
}

/// `InvalidArgumentError` raised by `Output.array` bounds.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct InvalidArgumentError {
    pub parameter: String,
    pub message: String,
}

fn array_length_error(
    value: &[JsonValue],
    min: Option<usize>,
    max: Option<usize>,
) -> Option<TypeValidationError> {
    let json = JsonValue::Array(value.to_vec());
    if let Some(min) = min
        && value.len() < min
    {
        return Some(TypeValidationError::new(
            json,
            format!("elements array must contain at least {min} items"),
        ));
    }
    if let Some(max) = max
        && value.len() > max
    {
        return Some(TypeValidationError::new(
            json,
            format!("elements array must contain at most {max} items"),
        ));
    }
    None
}

/// `Output.array({ element, minItems, maxItems, name, description })`.
pub fn array(
    element: Schema,
    min_items: Option<usize>,
    max_items: Option<usize>,
    name: Option<String>,
    description: Option<String>,
) -> Result<SharedOutput, InvalidArgumentError> {
    if let (Some(min), Some(max)) = (min_items, max_items)
        && min > max
    {
        return Err(InvalidArgumentError {
            parameter: "minItems".into(),
            message: "minItems must be less than or equal to maxItems".into(),
        });
    }
    struct Array {
        element: Schema,
        min: Option<usize>,
        max: Option<usize>,
        name: Option<String>,
        description: Option<String>,
    }
    struct Elements {
        published: usize,
        max: Option<usize>,
    }
    impl ElementStreamTransform for Elements {
        fn transform(&mut self, partial: Option<&JsonValue>) -> Result<Vec<JsonValue>, BoxError> {
            let mut out = Vec::new();
            if let Some(JsonValue::Array(items)) = partial {
                while self.published < items.len() {
                    if let Some(max) = self.max
                        && self.published >= max
                    {
                        return Err(Box::new(
                            array_length_error(items, None, Some(max)).expect("over max"),
                        ));
                    }
                    out.push(items[self.published].clone());
                    self.published += 1;
                }
            }
            Ok(out)
        }
    }
    impl Output for Array {
        fn name(&self) -> &str {
            "array"
        }
        fn response_format(&self) -> BoxFuture<'static, HarnessResult<HarnessV1ResponseFormat>> {
            let mut item = self
                .element
                .json_schema
                .as_object()
                .cloned()
                .unwrap_or_default();
            item.remove("$schema");
            let definitions = item.remove("definitions");
            let defs = item.remove("$defs");
            let mut elements = serde_json::Map::new();
            elements.insert("type".into(), "array".into());
            elements.insert("items".into(), JsonValue::Object(item));
            if let Some(min) = self.min {
                elements.insert("minItems".into(), min.into());
            }
            if let Some(max) = self.max {
                elements.insert("maxItems".into(), max.into());
            }
            let mut schema = serde_json::Map::new();
            schema.insert(
                "$schema".into(),
                "http://json-schema.org/draft-07/schema#".into(),
            );
            if let Some(d) = definitions {
                schema.insert("definitions".into(), d);
            }
            if let Some(d) = defs {
                schema.insert("$defs".into(), d);
            }
            schema.insert("type".into(), "object".into());
            let mut props = serde_json::Map::new();
            props.insert("elements".into(), JsonValue::Object(elements));
            schema.insert("properties".into(), JsonValue::Object(props));
            schema.insert("required".into(), serde_json::json!(["elements"]));
            schema.insert("additionalProperties".into(), false.into());
            let format = HarnessV1ResponseFormat::Json {
                schema: Some(schema),
                name: self.name.clone(),
                description: self.description.clone(),
            };
            Box::pin(async move { Ok(format) })
        }
        fn parse_complete_output(
            &self,
            text: &str,
            context: OutputContext,
        ) -> HarnessResult<JsonValue> {
            let outer = safe_parse_json(text, None).map_err(|e| {
                NoObjectGeneratedError::boxed(COULD_NOT_PARSE, &e, text, context.clone())
            })?;
            let Some(elements) = outer.get("elements").and_then(|e| e.as_array()) else {
                let cause = TypeValidationError::new(
                    outer.clone(),
                    "response must be an object with an elements array",
                );
                return Err(NoObjectGeneratedError::boxed(
                    DID_NOT_MATCH,
                    &cause,
                    text,
                    context,
                ));
            };
            if let Some(cause) = array_length_error(elements, self.min, self.max) {
                return Err(NoObjectGeneratedError::boxed(
                    DID_NOT_MATCH,
                    &cause,
                    text,
                    context,
                ));
            }
            let mut validated = Vec::new();
            for element in elements {
                validated.push(validate_types(element, &self.element, None).map_err(|e| {
                    NoObjectGeneratedError::boxed(DID_NOT_MATCH, &e, text, context.clone())
                })?);
            }
            Ok(JsonValue::Array(validated))
        }
        fn parse_partial_output(&self, text: &str) -> Option<PartialOutput> {
            let (value, state) = parse_partial_json(Some(text));
            let value = match state {
                ParsePartialJsonState::SuccessfulParse | ParsePartialJsonState::RepairedParse => {
                    value?
                }
                _ => return None,
            };
            let elements = value.get("elements")?.as_array()?;
            let raw = if state == ParsePartialJsonState::RepairedParse && !elements.is_empty() {
                &elements[..elements.len() - 1]
            } else {
                &elements[..]
            };
            let parsed: Vec<JsonValue> = raw
                .iter()
                .filter_map(|e| validate_types(e, &self.element, None).ok())
                .collect();
            Some(PartialOutput {
                partial: JsonValue::Array(parsed),
            })
        }
        fn create_element_stream_transform(&self) -> Option<Box<dyn ElementStreamTransform>> {
            Some(Box::new(Elements {
                published: 0,
                max: self.max,
            }))
        }
    }
    Ok(Arc::new(Array {
        element,
        min: min_items,
        max: max_items,
        name,
        description,
    }))
}

/// `Output.choice({ options, name, description })`.
pub fn choice(
    options: Vec<String>,
    name: Option<String>,
    description: Option<String>,
) -> SharedOutput {
    struct Choice {
        options: Vec<String>,
        name: Option<String>,
        description: Option<String>,
    }
    impl Output for Choice {
        fn name(&self) -> &str {
            "choice"
        }
        fn response_format(&self) -> BoxFuture<'static, HarnessResult<HarnessV1ResponseFormat>> {
            let schema = serde_json::json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": { "result": { "type": "string", "enum": self.options } },
                "required": ["result"],
                "additionalProperties": false
            });
            let format = HarnessV1ResponseFormat::Json {
                schema: schema.as_object().cloned(),
                name: self.name.clone(),
                description: self.description.clone(),
            };
            Box::pin(async move { Ok(format) })
        }
        fn parse_complete_output(
            &self,
            text: &str,
            context: OutputContext,
        ) -> HarnessResult<JsonValue> {
            let outer = safe_parse_json(text, None).map_err(|e| {
                NoObjectGeneratedError::boxed(COULD_NOT_PARSE, &e, text, context.clone())
            })?;
            match outer.get("result").and_then(|r| r.as_str()) {
                Some(result) if self.options.iter().any(|o| o == result) => {
                    Ok(JsonValue::String(result.into()))
                }
                _ => {
                    let cause = TypeValidationError::new(
                        outer.clone(),
                        "response must be an object that contains a choice value.",
                    );
                    Err(NoObjectGeneratedError::boxed(
                        DID_NOT_MATCH,
                        &cause,
                        text,
                        context,
                    ))
                }
            }
        }
        fn parse_partial_output(&self, text: &str) -> Option<PartialOutput> {
            let (value, state) = parse_partial_json(Some(text));
            let value = match state {
                ParsePartialJsonState::SuccessfulParse | ParsePartialJsonState::RepairedParse => {
                    value?
                }
                _ => return None,
            };
            let result = value.get("result")?.as_str()?;
            let matches: Vec<&String> = self
                .options
                .iter()
                .filter(|o| o.starts_with(result))
                .collect();
            if state == ParsePartialJsonState::SuccessfulParse {
                matches
                    .iter()
                    .any(|m| m.as_str() == result)
                    .then(|| PartialOutput {
                        partial: JsonValue::String(result.into()),
                    })
            } else if matches.len() == 1 {
                Some(PartialOutput {
                    partial: JsonValue::String(matches[0].clone()),
                })
            } else {
                None
            }
        }
    }
    Arc::new(Choice {
        options,
        name,
        description,
    })
}

/// `Output.json({ name, description })`.
pub fn json(name: Option<String>, description: Option<String>) -> SharedOutput {
    struct Json {
        name: Option<String>,
        description: Option<String>,
    }
    impl Output for Json {
        fn name(&self) -> &str {
            "json"
        }
        fn response_format(&self) -> BoxFuture<'static, HarnessResult<HarnessV1ResponseFormat>> {
            let format = HarnessV1ResponseFormat::Json {
                schema: None,
                name: self.name.clone(),
                description: self.description.clone(),
            };
            Box::pin(async move { Ok(format) })
        }
        fn parse_complete_output(
            &self,
            text: &str,
            context: OutputContext,
        ) -> HarnessResult<JsonValue> {
            safe_parse_json(text, None)
                .map_err(|e| NoObjectGeneratedError::boxed(COULD_NOT_PARSE, &e, text, context))
        }
        fn parse_partial_output(&self, text: &str) -> Option<PartialOutput> {
            match parse_partial_json(Some(text)) {
                (
                    Some(value),
                    ParsePartialJsonState::SuccessfulParse | ParsePartialJsonState::RepairedParse,
                ) => Some(PartialOutput { partial: value }),
                _ => None,
            }
        }
    }
    Arc::new(Json { name, description })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // fix-json.test.ts (representative cases)
    #[test]
    fn fix_json_closes_truncated_values() {
        let cases = [
            ("", ""),
            ("{", "{}"),
            ("[", "[]"),
            ("\"abc", "\"abc\""),
            ("\"abc\\", "\"abc\""),
            ("\"\\u00", "\"\""),
            ("{\"a\": 1", "{\"a\": 1}"),
            ("{\"a\": 1,", "{\"a\": 1}"),
            ("{\"a\"", "{}"),
            ("{\"a\":", "{}"),
            ("[1, 2", "[1, 2]"),
            ("[1, 2,", "[1, 2]"),
            ("[1, 2.", "[1, 2]"),
            ("-", ""),
            ("12.3e", "12.3"),
            ("t", "true"),
            ("fal", "false"),
            ("[nu", "[null]"),
            ("{\"a\": tr", "{\"a\": true}"),
            ("{\"a\": [{\"b\": \"c", "{\"a\": [{\"b\": \"c\"}]}"),
            ("{\"a\": 1} ", "{\"a\": 1}"),
        ];
        for (input, expected) in cases {
            assert_eq!(fix_json(input), expected, "input {input:?}");
        }
    }

    // parse-partial-json.test.ts
    #[test]
    fn parse_partial_json_states() {
        assert_eq!(
            parse_partial_json(None),
            (None, ParsePartialJsonState::UndefinedInput)
        );
        assert_eq!(
            parse_partial_json(Some("{\"a\":1}")),
            (
                Some(json!({"a": 1})),
                ParsePartialJsonState::SuccessfulParse
            )
        );
        assert_eq!(
            parse_partial_json(Some("{\"a\":1")),
            (Some(json!({"a": 1})), ParsePartialJsonState::RepairedParse)
        );
        assert_eq!(
            parse_partial_json(Some("x")),
            (None, ParsePartialJsonState::FailedParse)
        );
    }

    #[test]
    fn choice_partials_resolve_unique_prefixes() {
        let out = choice(
            vec!["sunny".into(), "rainy".into(), "snowy".into()],
            None,
            None,
        );
        assert_eq!(
            out.parse_partial_output("{\"result\":\"su")
                .unwrap()
                .partial,
            json!("sunny")
        );
        assert!(out.parse_partial_output("{\"result\":\"s").is_none());
        assert!(
            out.parse_partial_output("{\"result\":\"cloudy\"}")
                .is_none()
        );
    }

    #[test]
    fn array_partials_drop_the_last_repaired_element() {
        let out = array(
            crate::harness::sdk::tool::json_schema(json!({})),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            out.parse_partial_output("{\"elements\":[{\"a\":1},{\"a\":")
                .unwrap()
                .partial,
            json!([{"a": 1}])
        );
        let mut t = out.create_element_stream_transform().unwrap();
        assert_eq!(
            t.transform(Some(&json!([1, 2]))).unwrap(),
            vec![json!(1), json!(2)]
        );
        assert_eq!(
            t.transform(Some(&json!([1, 2, 3]))).unwrap(),
            vec![json!(3)]
        );
    }

    #[test]
    fn object_complete_output_errors_name_the_failure() {
        let out = object(
            crate::harness::sdk::tool::json_schema(json!({})).with_validate(|_| Err("nope".into())),
            None,
            None,
        );
        let context = OutputContext {
            response: LanguageModelResponseMetadata {
                id: "r".into(),
                timestamp: std::time::UNIX_EPOCH,
                model_id: "m".into(),
                headers: None,
                messages: vec![],
                body: None,
            },
            usage: Default::default(),
            finish_reason: FinishReason::Stop,
        };
        assert_eq!(
            out.parse_complete_output("x", context.clone())
                .unwrap_err()
                .to_string(),
            "No object generated: could not parse the response."
        );
        assert_eq!(
            out.parse_complete_output("{}", context)
                .unwrap_err()
                .to_string(),
            "No object generated: response did not match schema."
        );
    }
}
