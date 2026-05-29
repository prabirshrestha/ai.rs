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
