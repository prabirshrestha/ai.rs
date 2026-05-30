use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::Result;

pub fn parse_json_with_repair<T: DeserializeOwned>(json: &str) -> Result<T> {
    match serde_json::from_str(json) {
        Ok(value) => Ok(value),
        Err(first_error) => {
            let repaired = repair_json(json);
            if repaired != json {
                serde_json::from_str(&repaired).map_err(Into::into)
            } else {
                Err(first_error.into())
            }
        }
    }
}

pub fn parse_streaming_json(partial_json: Option<&str>) -> Value {
    let Some(partial_json) = partial_json else {
        return Value::Object(Default::default());
    };
    if partial_json.trim().is_empty() {
        return Value::Object(Default::default());
    }

    if let Ok(value) = serde_json::from_str(partial_json) {
        return value;
    }
    let repaired = repair_json(partial_json);
    if let Ok(value) = serde_json::from_str(&repaired) {
        return value;
    }
    if let Some(value) = PartialJsonParser::new(&repaired).parse() {
        return value;
    }
    let completed = complete_partial_json(&repaired);
    serde_json::from_str(&completed).unwrap_or_else(|_| Value::Object(Default::default()))
}

pub fn repair_json(json: &str) -> String {
    let mut repaired = String::with_capacity(json.len());
    let mut in_string = false;
    let mut escaped = false;

    for ch in json.chars() {
        if !in_string {
            repaired.push(ch);
            if ch == '"' {
                in_string = true;
            }
            continue;
        }

        if escaped {
            match ch {
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u' => {
                    repaired.push('\\');
                    repaired.push(ch);
                }
                _ => {
                    repaired.push_str("\\\\");
                    repaired.push(ch);
                }
            }
            escaped = false;
            continue;
        }

        match ch {
            '\\' => escaped = true,
            '"' => {
                repaired.push(ch);
                in_string = false;
            }
            '\u{0008}' => repaired.push_str("\\b"),
            '\u{000c}' => repaired.push_str("\\f"),
            '\n' => repaired.push_str("\\n"),
            '\r' => repaired.push_str("\\r"),
            '\t' => repaired.push_str("\\t"),
            ch if ch.is_control() => {
                use std::fmt::Write;
                let _ = write!(repaired, "\\u{:04x}", ch as u32);
            }
            ch => repaired.push(ch),
        }
    }

    if escaped {
        repaired.push_str("\\\\");
    }

    repaired
}

fn complete_partial_json(json: &str) -> String {
    let mut result = json.to_string();
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for ch in json.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                if stack.last() == Some(&ch) {
                    stack.pop();
                }
            }
            _ => {}
        }
    }

    if in_string {
        result.push('"');
    }
    while let Some(ch) = stack.pop() {
        result.push(ch);
    }
    result
}

struct PartialJsonParser<'a> {
    json: &'a str,
    index: usize,
}

impl<'a> PartialJsonParser<'a> {
    fn new(json: &'a str) -> Self {
        Self {
            json: json.trim(),
            index: 0,
        }
    }

    fn parse(mut self) -> Option<Value> {
        if self.json.is_empty() {
            return None;
        }
        self.parse_any().ok()
    }

    fn parse_any(&mut self) -> std::result::Result<Value, ()> {
        self.skip_blank();
        let Some(byte) = self.peek() else {
            return Err(());
        };
        match byte {
            b'"' => self.parse_string().map(Value::String),
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'n' => self.parse_partial_literal("null", Value::Null),
            b't' => self.parse_partial_literal("true", Value::Bool(true)),
            b'f' => self.parse_partial_literal("false", Value::Bool(false)),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(()),
        }
    }

    fn parse_object(&mut self) -> std::result::Result<Value, ()> {
        self.index += 1;
        let mut object = serde_json::Map::new();

        loop {
            self.skip_blank();
            match self.peek() {
                Some(b'}') => {
                    self.index += 1;
                    return Ok(Value::Object(object));
                }
                None => return Ok(Value::Object(object)),
                _ => {}
            }

            let key = match self.parse_string() {
                Ok(key) => key,
                Err(()) => return Ok(Value::Object(object)),
            };

            self.skip_blank();
            if self.peek() == Some(b':') {
                self.index += 1;
            }

            match self.parse_any() {
                Ok(value) => {
                    object.insert(key, value);
                }
                Err(()) => return Ok(Value::Object(object)),
            }

            self.skip_blank();
            if self.peek() == Some(b',') {
                self.index += 1;
            }
        }
    }

    fn parse_array(&mut self) -> std::result::Result<Value, ()> {
        self.index += 1;
        let mut array = Vec::new();

        loop {
            self.skip_blank();
            match self.peek() {
                Some(b']') => {
                    self.index += 1;
                    return Ok(Value::Array(array));
                }
                None => return Ok(Value::Array(array)),
                _ => {}
            }

            match self.parse_any() {
                Ok(value) => array.push(value),
                Err(()) => return Ok(Value::Array(array)),
            }

            self.skip_blank();
            if self.peek() == Some(b',') {
                self.index += 1;
            }
        }
    }

    fn parse_string(&mut self) -> std::result::Result<String, ()> {
        if self.peek() != Some(b'"') {
            return Err(());
        }
        let start = self.index;
        self.index += 1;
        let mut escaped = false;

        while let Some(byte) = self.peek() {
            self.index += 1;
            if escaped {
                escaped = false;
                continue;
            }
            match byte {
                b'\\' => escaped = true,
                b'"' => {
                    return serde_json::from_str(&self.json[start..self.index]).map_err(|_| ());
                }
                _ => {}
            }
        }

        let end = if escaped {
            self.json[..self.index]
                .rfind('\\')
                .filter(|slash| *slash >= start)
                .unwrap_or(self.index)
        } else {
            self.index
        };
        let candidate = format!("{}\"", &self.json[start..end]);
        serde_json::from_str(&candidate).map_err(|_| ())
    }

    fn parse_partial_literal(
        &mut self,
        literal: &str,
        value: Value,
    ) -> std::result::Result<Value, ()> {
        let rest = &self.json[self.index..];
        if literal.starts_with(rest) || rest.starts_with(literal) {
            self.index = self
                .index
                .saturating_add(literal.len())
                .min(self.json.len());
            Ok(value)
        } else {
            Err(())
        }
    }

    fn parse_number(&mut self) -> std::result::Result<Value, ()> {
        let start = self.index;
        while let Some(byte) = self.peek() {
            if matches!(byte, b',' | b']' | b'}') {
                break;
            }
            self.index += 1;
        }

        let raw = self.json[start..self.index].trim_end();
        if raw == "-" || raw.is_empty() {
            return Err(());
        }
        if let Ok(value) = serde_json::from_str(raw) {
            return Ok(value);
        }

        if let Some(exponent) = raw.rfind('e') {
            let candidate = raw[..exponent].trim_end();
            if candidate != "-" && !candidate.is_empty() {
                if let Ok(value) = serde_json::from_str(candidate) {
                    return Ok(value);
                }
            }
        }

        Err(())
    }

    fn skip_blank(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.index += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.json.as_bytes().get(self.index).copied()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_streaming_json_returns_completed_object_members() {
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
    fn parse_streaming_json_repairs_invalid_escapes_before_partial_parse() {
        assert_eq!(
            parse_streaming_json(Some(r#"{"path":"A\H","next""#)),
            json!({ "path": r#"A\H"# })
        );
    }

    #[test]
    fn parse_streaming_json_matches_partial_json_number_recovery() {
        assert_eq!(
            parse_streaming_json(Some(r#"{"count":1e,"next""#)),
            json!({ "count": 1 })
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"count":123.,"next""#)),
            json!({})
        );
    }
}
