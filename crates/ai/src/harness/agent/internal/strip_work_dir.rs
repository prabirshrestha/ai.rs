//! Port of `agent/internal/strip-work-dir.ts`: rewrite absolute sandbox
//! work-dir paths to relative ones in what the agent shows the caller.
//!
//! [rust] Strings are processed as `char` sequences (TS indexes UTF-16
//! code units; the two agree for every non-astral character).

use std::collections::HashMap;

use crate::harness::provider::JsonValue;
use crate::harness::spec::stream_part::HarnessV1StreamPart;

#[derive(Default, Clone)]
struct StreamState {
    pending: Vec<char>,
    preceding: Option<char>,
}

/// `createToolInputWorkDirStripper({ sessionWorkDir })`.
///
/// The returned function takes `tool-input-start`, `tool-input-delta` and
/// `tool-input-end` parts (other parts pass through).
pub fn create_tool_input_work_dir_stripper(
    session_work_dir: &str,
) -> impl FnMut(HarnessV1StreamPart) -> Vec<HarnessV1StreamPart> + Send + 'static {
    let work_dir: Vec<char> = session_work_dir.chars().collect();
    let mut states: HashMap<String, StreamState> = HashMap::new();
    move |part| {
        if work_dir.is_empty() {
            return vec![part];
        }
        match part {
            HarnessV1StreamPart::ToolInputStart { ref id, .. } => {
                states.insert(id.clone(), StreamState::default());
                vec![part]
            }
            HarnessV1StreamPart::ToolInputDelta {
                id,
                delta,
                provider_metadata,
            } => {
                let state = states.get(&id).cloned().unwrap_or_default();
                let mut value = state.pending.clone();
                value.extend(delta.chars());
                let stripped = strip_streaming(&value, &work_dir, false, state.preceding);
                let pending_start = value.len() - stripped.pending.len();
                let preceding = if !stripped.pending.is_empty() {
                    if pending_start > 0 {
                        Some(value[pending_start - 1])
                    } else {
                        state.preceding
                    }
                } else if !value.is_empty() {
                    value.last().copied()
                } else {
                    state.preceding
                };
                states.insert(
                    id.clone(),
                    StreamState {
                        pending: stripped.pending,
                        preceding,
                    },
                );
                if stripped.output.is_empty() {
                    vec![]
                } else {
                    vec![HarnessV1StreamPart::ToolInputDelta {
                        id,
                        delta: stripped.output.into_iter().collect(),
                        provider_metadata,
                    }]
                }
            }
            HarnessV1StreamPart::ToolInputEnd { ref id, .. } => {
                let state = states.remove(id);
                let Some(state) = state.filter(|s| !s.pending.is_empty()) else {
                    return vec![part];
                };
                let stripped = strip_streaming(&state.pending, &work_dir, true, state.preceding);
                if stripped.output.is_empty() {
                    vec![part]
                } else {
                    vec![
                        HarnessV1StreamPart::ToolInputDelta {
                            id: id.clone(),
                            delta: stripped.output.into_iter().collect(),
                            provider_metadata: None,
                        },
                        part,
                    ]
                }
            }
            other => vec![other],
        }
    }
}

/// `stripWorkDir(part, sessionWorkDir)`.
pub fn strip_work_dir(part: &HarnessV1StreamPart, session_work_dir: &str) -> HarnessV1StreamPart {
    if session_work_dir.is_empty() {
        return part.clone();
    }
    let mut part = part.clone();
    match &mut part {
        HarnessV1StreamPart::ToolInputDelta { delta, .. } => {
            *delta = strip_string(delta, session_work_dir)
        }
        HarnessV1StreamPart::ToolCall { input, .. } => {
            *input = strip_string(input, session_work_dir)
        }
        HarnessV1StreamPart::ToolResult { result, .. } => {
            *result = strip_deep(result, session_work_dir)
        }
        HarnessV1StreamPart::FileChange { path, .. } => {
            *path = strip_string(path, session_work_dir)
        }
        _ => {}
    }
    part
}

/// `stripParsedToolInputWorkDir({ value, sessionWorkDir })`: a stripped
/// copy of a parsed tool input. Iterative, so deep inputs cannot overflow
/// the stack.
pub fn strip_parsed_tool_input_work_dir(value: &JsonValue, session_work_dir: &str) -> JsonValue {
    if session_work_dir.is_empty() {
        return value.clone();
    }
    strip_value_iteratively(value, session_work_dir)
}

/// Strip a JSON value without recursion.
fn strip_value_iteratively(value: &JsonValue, work_dir: &str) -> JsonValue {
    enum Frame<'a> {
        Array(std::slice::Iter<'a, JsonValue>, Vec<JsonValue>),
        Object(
            serde_json::map::Iter<'a>,
            serde_json::Map<String, JsonValue>,
            Option<String>,
        ),
    }
    let leaf = |v: &JsonValue| -> Option<JsonValue> {
        match v {
            JsonValue::String(s) => Some(JsonValue::String(strip_string(s, work_dir))),
            JsonValue::Array(_) | JsonValue::Object(_) => None,
            other => Some(other.clone()),
        }
    };
    if let Some(v) = leaf(value) {
        return v;
    }
    fn open(v: &JsonValue) -> Frame<'_> {
        match v {
            JsonValue::Array(items) => Frame::Array(items.iter(), Vec::with_capacity(items.len())),
            JsonValue::Object(map) => Frame::Object(map.iter(), serde_json::Map::new(), None),
            _ => unreachable!(),
        }
    }
    let mut stack: Vec<Frame<'_>> = vec![open(value)];
    loop {
        let next = match stack.last_mut().expect("non-empty") {
            Frame::Array(iter, _) => iter.next().map(|v| (None, v)),
            Frame::Object(iter, _, key) => iter.next().map(|(k, v)| {
                *key = Some(k.clone());
                (Some(k.clone()), v)
            }),
        };
        match next {
            Some((key, child)) => match leaf(child) {
                Some(v) => match stack.last_mut().expect("non-empty") {
                    Frame::Array(_, out) => out.push(v),
                    Frame::Object(_, out, _) => {
                        out.insert(key.expect("object key"), v);
                    }
                },
                None => stack.push(open(child)),
            },
            None => {
                let done = match stack.pop().expect("non-empty") {
                    Frame::Array(_, out) => JsonValue::Array(out),
                    Frame::Object(_, out, _) => JsonValue::Object(out),
                };
                match stack.last_mut() {
                    None => return done,
                    Some(Frame::Array(_, out)) => out.push(done),
                    Some(Frame::Object(_, out, key)) => {
                        out.insert(key.clone().expect("object key"), done);
                    }
                }
            }
        }
    }
}

fn strip_string(value: &str, work_dir: &str) -> String {
    let value: Vec<char> = value.chars().collect();
    let work_dir: Vec<char> = work_dir.chars().collect();
    strip_streaming(&value, &work_dir, true, None)
        .output
        .into_iter()
        .collect()
}

struct Stripped {
    output: Vec<char>,
    pending: Vec<char>,
}

fn index_of(haystack: &[char], needle: &[char]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn strip_streaming(
    value: &[char],
    work_dir: &[char],
    final_: bool,
    preceding: Option<char>,
) -> Stripped {
    let mut remaining: &[char] = value;
    let mut output: Vec<char> = Vec::new();
    let mut before = preceding;
    while !remaining.is_empty() {
        if let Some(match_index) = index_of(remaining, work_dir) {
            let following = match_index + work_dir.len();
            let boundary = is_path_boundary(remaining, match_index, before);
            if boundary && following == remaining.len() && !final_ {
                output.extend_from_slice(&remaining[..match_index]);
                return Stripped {
                    output,
                    pending: remaining[match_index..].to_vec(),
                };
            }
            let next = remaining.get(following).copied();
            let is_path = boundary
                && (following == remaining.len() || next == Some('/') || is_path_terminator(next));
            if !is_path {
                output.extend_from_slice(&remaining[..following]);
                before = Some(remaining[following - 1]);
                remaining = &remaining[following..];
            } else if next == Some('/') {
                output.extend_from_slice(&remaining[..match_index]);
                before = Some('/');
                remaining = &remaining[following + 1..];
            } else {
                output.extend_from_slice(&remaining[..match_index]);
                output.push('.');
                remaining = &remaining[following..];
            }
            continue;
        }
        if final_ {
            output.extend_from_slice(remaining);
            return Stripped {
                output,
                pending: vec![],
            };
        }
        let mut pending_len = remaining.len().min(work_dir.len().saturating_sub(1));
        while pending_len > 0
            && (!work_dir.starts_with(&remaining[remaining.len() - pending_len..])
                || !is_path_boundary(remaining, remaining.len() - pending_len, before))
        {
            pending_len -= 1;
        }
        let output_len = remaining.len() - pending_len;
        output.extend_from_slice(&remaining[..output_len]);
        return Stripped {
            output,
            pending: remaining[output_len..].to_vec(),
        };
    }
    Stripped {
        output,
        pending: vec![],
    }
}

fn is_path_boundary(value: &[char], index: usize, preceding: Option<char>) -> bool {
    let c = if index == 0 {
        preceding
    } else {
        Some(value[index - 1])
    };
    match c {
        None => true,
        Some(c) => c.is_whitespace() || matches!(c, '"' | '\'' | '`' | '='),
    }
}

fn is_path_terminator(c: Option<char>) -> bool {
    match c {
        None => true,
        Some(c) => {
            c.is_whitespace()
                || matches!(
                    c,
                    '"' | '\'' | '`' | ';' | '|' | '&' | '<' | '>' | '(' | ')'
                )
        }
    }
}

fn strip_deep(value: &JsonValue, work_dir: &str) -> JsonValue {
    match value {
        JsonValue::String(s) => JsonValue::String(strip_string(s, work_dir)),
        JsonValue::Array(items) => {
            JsonValue::Array(items.iter().map(|v| strip_deep(v, work_dir)).collect())
        }
        JsonValue::Object(map) => JsonValue::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), strip_deep(v, work_dir)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
#[path = "strip_work_dir_tests.rs"]
mod tests;
