//! Port of `utils/json-parse.ts`, including a port of the `partial-json`
//! (0.1.7) parser Pi uses for incomplete streaming JSON.
//!
//! `JSON.parse` becomes `serde_json::from_str`. Two edge cases differ: serde
//! rejects lone UTF-16 surrogate escapes (`"\ud800"`), and `Infinity`/`NaN`
//! accepted by `partial-json` become `null` (their `JSON.stringify` form).

use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::Result;

const VALID_JSON_ESCAPES: &[char] = &['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'];

fn is_control_character(ch: char) -> bool {
    (ch as u32) <= 0x1f
}

fn escape_control_character(ch: char) -> String {
    match ch {
        '\u{0008}' => "\\b".to_string(),
        '\u{000c}' => "\\f".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        _ => format!("\\u{:04x}", ch as u32),
    }
}

/// Repairs malformed JSON string literals by:
/// - escaping raw control characters inside strings
/// - doubling backslashes before invalid escape characters
pub fn repair_json(json: &str) -> String {
    let chars: Vec<char> = json.chars().collect();
    let mut repaired = String::with_capacity(json.len());
    let mut in_string = false;
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];

        if !in_string {
            repaired.push(ch);
            if ch == '"' {
                in_string = true;
            }
            index += 1;
            continue;
        }

        if ch == '"' {
            repaired.push(ch);
            in_string = false;
            index += 1;
            continue;
        }

        if ch == '\\' {
            let Some(&next_char) = chars.get(index + 1) else {
                repaired.push_str("\\\\");
                index += 1;
                continue;
            };

            if next_char == 'u' {
                let unicode_digits =
                    &chars[(index + 2).min(chars.len())..(index + 6).min(chars.len())];
                if unicode_digits.len() == 4 && unicode_digits.iter().all(char::is_ascii_hexdigit) {
                    repaired.push_str("\\u");
                    repaired.extend(unicode_digits);
                    index += 6;
                    continue;
                }
            }

            if VALID_JSON_ESCAPES.contains(&next_char) {
                repaired.push('\\');
                repaired.push(next_char);
                index += 2;
                continue;
            }

            repaired.push_str("\\\\");
            index += 1;
            continue;
        }

        if is_control_character(ch) {
            repaired.push_str(&escape_control_character(ch));
        } else {
            repaired.push(ch);
        }
        index += 1;
    }

    repaired
}

pub fn parse_json_with_repair<T: DeserializeOwned>(json: &str) -> Result<T> {
    match serde_json::from_str(json) {
        Ok(value) => Ok(value),
        Err(error) => {
            let repaired_json = repair_json(json);
            if repaired_json != json {
                return serde_json::from_str(&repaired_json).map_err(Into::into);
            }
            Err(error.into())
        }
    }
}

/// Attempts to parse potentially incomplete JSON during streaming. Always
/// returns a value, `{}` when parsing fails.
pub fn parse_streaming_json(partial_json: Option<&str>) -> Value {
    let Some(partial_json) = partial_json.filter(|json| !json.trim().is_empty()) else {
        return Value::Object(Map::new());
    };

    if let Ok(value) = parse_json_with_repair::<Value>(partial_json) {
        return value;
    }
    if let Ok(value) = partial_parse(partial_json) {
        return non_null_or_empty(value);
    }
    if let Ok(value) = partial_parse(&repair_json(partial_json)) {
        return non_null_or_empty(value);
    }
    Value::Object(Map::new())
}

/// `result ?? {}`.
fn non_null_or_empty(value: Value) -> Value {
    if value.is_null() {
        Value::Object(Map::new())
    } else {
        value
    }
}

/// `partial-json`'s `PartialJSON` / `MalformedJSON` errors.
#[derive(Debug)]
struct PartialJsonError;

type PartialResult<T> = std::result::Result<T, PartialJsonError>;

/// Port of `partial-json`'s `parse(jsonString, Allow.ALL)`.
fn partial_parse(json_string: &str) -> PartialResult<Value> {
    if json_string.trim().is_empty() {
        return Err(PartialJsonError);
    }
    let chars: Vec<char> = json_string.trim().chars().collect();
    PartialJsonParser {
        chars: &chars,
        index: 0,
    }
    .parse_any()
}

struct PartialJsonParser<'a> {
    chars: &'a [char],
    index: usize,
}

impl PartialJsonParser<'_> {
    fn length(&self) -> usize {
        self.chars.len()
    }

    fn at(&self, index: usize) -> Option<char> {
        self.chars.get(index).copied()
    }

    /// JavaScript `String.prototype.substring` with clamping and argument swap.
    fn substring(&self, start: isize, end: isize) -> String {
        let length = self.length() as isize;
        let start = start.clamp(0, length) as usize;
        let end = end.clamp(0, length) as usize;
        let (start, end) = if start > end {
            (end, start)
        } else {
            (start, end)
        };
        self.chars[start..end].iter().collect()
    }

    fn rest_matches(&self, literal: &str) -> bool {
        let literal: Vec<char> = literal.chars().collect();
        self.chars[self.index..].starts_with(&literal)
    }

    /// `length - index < literal.length && literal.startsWith(rest)`.
    fn rest_is_prefix_of(&self, literal: &str) -> bool {
        let rest = &self.chars[self.index..];
        let literal: Vec<char> = literal.chars().collect();
        rest.len() < literal.len() && literal.starts_with(rest)
    }

    fn last_index_of(&self, ch: char) -> isize {
        self.chars
            .iter()
            .rposition(|candidate| *candidate == ch)
            .map_or(-1, |index| index as isize)
    }

    fn json_parse(text: &str) -> PartialResult<Value> {
        serde_json::from_str(text).map_err(|_| PartialJsonError)
    }

    fn parse_any(&mut self) -> PartialResult<Value> {
        self.skip_blank();
        if self.index >= self.length() {
            return Err(PartialJsonError);
        }
        match self.chars[self.index] {
            '"' => return self.parse_str().map(Value::String),
            '{' => return self.parse_obj(),
            '[' => return self.parse_arr(),
            _ => {}
        }
        if self.rest_matches("null") || self.rest_is_prefix_of("null") {
            self.index += 4;
            return Ok(Value::Null);
        }
        if self.rest_matches("true") || self.rest_is_prefix_of("true") {
            self.index += 4;
            return Ok(Value::Bool(true));
        }
        if self.rest_matches("false") || self.rest_is_prefix_of("false") {
            self.index += 5;
            return Ok(Value::Bool(false));
        }
        if self.rest_matches("Infinity") || self.rest_is_prefix_of("Infinity") {
            self.index += 8;
            return Ok(Value::Null);
        }
        let rest = self.length() - self.index;
        if self.rest_matches("-Infinity") || (1 < rest && self.rest_is_prefix_of("-Infinity")) {
            self.index += 9;
            return Ok(Value::Null);
        }
        if self.rest_matches("NaN") || self.rest_is_prefix_of("NaN") {
            self.index += 3;
            return Ok(Value::Null);
        }
        self.parse_num()
    }

    fn parse_str(&mut self) -> PartialResult<String> {
        let start = self.index;
        let mut escape = false;
        self.index += 1; // skip initial quote
        while self.index < self.length()
            && (self.chars[self.index] != '"'
                || (escape && self.at(self.index.wrapping_sub(1)) == Some('\\')))
        {
            escape = if self.chars[self.index] == '\\' {
                !escape
            } else {
                false
            };
            self.index += 1;
        }
        let parse_string = |text: String| -> PartialResult<String> {
            match Self::json_parse(&text)? {
                Value::String(value) => Ok(value),
                _ => Err(PartialJsonError),
            }
        };
        if self.at(self.index) == Some('"') {
            self.index += 1;
            let end = self.index as isize - isize::from(escape);
            return parse_string(self.substring(start as isize, end));
        }
        // Allow.STR
        let end = self.index as isize - isize::from(escape);
        match parse_string(format!("{}\"", self.substring(start as isize, end))) {
            Ok(value) => Ok(value),
            // SyntaxError: Invalid escape sequence
            Err(_) => parse_string(format!(
                "{}\"",
                self.substring(start as isize, self.last_index_of('\\'))
            )),
        }
    }

    fn parse_obj(&mut self) -> PartialResult<Value> {
        self.index += 1; // skip initial brace
        self.skip_blank();
        let mut object = Map::new();
        // Allow.OBJ: every failure returns the object parsed so far.
        while self.at(self.index) != Some('}') {
            self.skip_blank();
            if self.index >= self.length() {
                return Ok(Value::Object(object));
            }
            let Ok(key) = self.parse_str() else {
                return Ok(Value::Object(object));
            };
            self.skip_blank();
            self.index += 1; // skip colon
            match self.parse_any() {
                Ok(value) => {
                    object.insert(key, value);
                }
                Err(_) => return Ok(Value::Object(object)),
            }
            self.skip_blank();
            if self.at(self.index) == Some(',') {
                self.index += 1; // skip comma
            }
        }
        self.index += 1; // skip final brace
        Ok(Value::Object(object))
    }

    fn parse_arr(&mut self) -> PartialResult<Value> {
        self.index += 1; // skip initial bracket
        let mut array = Vec::new();
        // Allow.ARR: every failure returns the array parsed so far.
        while self.at(self.index) != Some(']') {
            match self.parse_any() {
                Ok(value) => array.push(value),
                Err(_) => return Ok(Value::Array(array)),
            }
            self.skip_blank();
            if self.at(self.index) == Some(',') {
                self.index += 1; // skip comma
            }
        }
        self.index += 1; // skip final bracket
        Ok(Value::Array(array))
    }

    fn parse_num(&mut self) -> PartialResult<Value> {
        if self.index == 0 {
            let whole: String = self.chars.iter().collect();
            if whole == "-" {
                return Err(PartialJsonError);
            }
            return match Self::json_parse(&whole) {
                Ok(value) => Ok(value),
                // Allow.NUM
                Err(_) => Self::json_parse(&self.substring(0, self.last_index_of('e'))),
            };
        }

        let start = self.index;
        if self.at(self.index) == Some('-') {
            self.index += 1;
        }
        while let Some(ch) = self.at(self.index) {
            if ",]}".contains(ch) {
                break;
            }
            self.index += 1;
        }
        let raw = self.substring(start as isize, self.index as isize);
        match Self::json_parse(&raw) {
            Ok(value) => Ok(value),
            Err(_) => {
                if raw == "-" {
                    return Err(PartialJsonError);
                }
                Self::json_parse(&self.substring(start as isize, self.last_index_of('e')))
            }
        }
    }

    fn skip_blank(&mut self) {
        while self.index < self.length() && " \n\r\t".contains(self.chars[self.index]) {
            self.index += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn repairs_control_characters_and_invalid_escapes_inside_strings() {
        assert_eq!(repair_json("{\"a\":\"x\ny\"}"), "{\"a\":\"x\\ny\"}");
        assert_eq!(repair_json("{\"a\":\"\u{1}\"}"), "{\"a\":\"\\u0001\"}");
        assert_eq!(repair_json(r#"{"path":"A\H"}"#), r#"{"path":"A\\H"}"#);
        // `u` is a valid escape letter, so a short `\u` sequence is kept.
        assert_eq!(repair_json(r#""\u12""#), r#""\u12""#);
        assert_eq!(repair_json(r#""\u00e9""#), r#""\u00e9""#);
        assert_eq!(repair_json("\"\\"), "\"\\\\");
        // DEL is not a JSON control character.
        assert_eq!(repair_json("\"\u{7f}\""), "\"\u{7f}\"");
        // Outside strings nothing changes.
        assert_eq!(repair_json("{\n}"), "{\n}");
    }

    #[test]
    fn parse_json_with_repair_retries_with_the_repaired_text() {
        let value: Value = parse_json_with_repair(r#"{"path":"C:\Users"}"#).unwrap();
        assert_eq!(value, json!({ "path": r"C:\Users" }));
        assert!(parse_json_with_repair::<Value>("{").is_err());
    }

    #[test]
    fn parse_streaming_json_returns_completed_object_members() {
        assert_eq!(parse_streaming_json(None), json!({}));
        assert_eq!(parse_streaming_json(Some("  ")), json!({}));
        assert_eq!(
            parse_streaming_json(Some(r#"{"path":"README.md","content""#)),
            json!({ "path": "README.md" })
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"path":"README.md","#)),
            json!({ "path": "README.md" })
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"path":"README.md","content":"hel"#)),
            json!({ "path": "README.md", "content": "hel" })
        );
    }

    #[test]
    fn parse_streaming_json_returns_completed_array_items() {
        assert_eq!(
            parse_streaming_json(Some(r#"[{"path":"a"},{"path""#)),
            json!([{ "path": "a" }, {}])
        );
        assert_eq!(
            parse_streaming_json(Some(r#"[{"path":"a"},{"path":"b"#)),
            json!([{ "path": "a" }, { "path": "b" }])
        );
    }

    #[test]
    fn parse_streaming_json_partial_parses_the_raw_text_before_the_repaired_text() {
        // partial-json drops the malformed member instead of throwing, so the
        // raw parse wins, as in Pi.
        assert_eq!(
            parse_streaming_json(Some(r#"{"path":"A\H","next""#)),
            json!({})
        );
        // A raw control character makes the raw partial parse return the
        // members before it; the repaired text is never consulted.
        assert_eq!(
            parse_streaming_json(Some("{\"a\":1,\"b\":\"x\ny")),
            json!({ "a": 1 })
        );
    }

    #[test]
    fn parse_streaming_json_matches_partial_json_number_and_literal_recovery() {
        assert_eq!(
            parse_streaming_json(Some(r#"{"count":1e,"next""#)),
            json!({})
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"count":1e"#)),
            json!({ "count": 1 })
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"count":123.,"next""#)),
            json!({})
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"a":tr"#)),
            json!({ "a": true })
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"a":nul"#)),
            json!({ "a": null })
        );
        assert_eq!(parse_streaming_json(Some("12e")), json!(12));
        assert_eq!(
            parse_streaming_json(Some(r#"{"s":"ab\"#)),
            json!({ "s": "ab" })
        );
    }
}
