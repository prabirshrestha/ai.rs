//! [rust] `parse(text, errors, { allowTrailingComma: true })` from
//! `jsonc-parser`, reduced to what the adapters use: JSON with `//` and
//! `/* */` comments and trailing commas. Returns `None` where
//! `jsonc-parser` would report an error.

#![allow(dead_code)]

use serde_json::Value as JsonValue;

/// Parses JSONC, allowing comments and trailing commas.
pub(crate) fn parse_jsonc(text: &str) -> Option<JsonValue> {
    let without_comments = strip_comments(text)?;
    let json = strip_trailing_commas(&without_comments);
    serde_json::from_str(&json).ok()
}

fn strip_comments(text: &str) -> Option<String> {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_string = false;
    while let Some(c) = chars.next() {
        if in_string {
            output.push(c);
            match c {
                '\\' => output.extend(chars.next()),
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match (c, chars.peek()) {
            ('"', _) => {
                in_string = true;
                output.push(c);
            }
            ('/', Some('/')) => {
                for next in chars.by_ref() {
                    if next == '\n' || next == '\r' {
                        output.push(next);
                        break;
                    }
                }
            }
            ('/', Some('*')) => {
                chars.next();
                let mut closed = false;
                let mut previous = '\0';
                for next in chars.by_ref() {
                    if previous == '*' && next == '/' {
                        closed = true;
                        break;
                    }
                    previous = next;
                }
                if !closed {
                    return None;
                }
                output.push(' ');
            }
            _ => output.push(c),
        }
    }
    Some(output)
}

fn strip_trailing_commas(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut in_string = false;
    let mut index = 0;
    while index < chars.len() {
        let c = chars[index];
        if in_string {
            output.push(c);
            if c == '\\' {
                if let Some(next) = chars.get(index + 1) {
                    output.push(*next);
                    index += 1;
                }
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
            output.push(c);
        } else if c == ',' {
            let next = chars[index + 1..].iter().find(|next| !next.is_whitespace());
            if !matches!(next, Some('}') | Some(']')) {
                output.push(c);
            }
        } else {
            output.push(c);
        }
        index += 1;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_comments_and_trailing_commas() {
        assert_eq!(
            parse_jsonc(
                r#"
                // line comment
                {
                  /* block */ "a": "x // not a comment",
                  "b": [1, 2,],
                  "c": "\"/*\"",
                }
                "#
            ),
            Some(json!({ "a": "x // not a comment", "b": [1, 2], "c": "\"/*\"" }))
        );
    }

    #[test]
    fn rejects_invalid_input() {
        assert_eq!(parse_jsonc("{ invalid"), None);
        assert_eq!(parse_jsonc("{} /* open"), None);
    }
}
