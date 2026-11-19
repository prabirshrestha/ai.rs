//! Port of `harness-pi/src/pi-events.ts`.
//!
//! [rust] `piSessionEventSchema` is a loose zod object. The port keeps the
//! raw JSON object and validates the recognised fields the same way, so
//! unknown fields still pass through.

use serde_json::{Map, Value as JsonValue};

/// `PiUsage`: Pi's `input` excludes cached tokens; `reasoning` is a subset
/// of `output`, set only by providers that report it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PiUsage {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub reasoning: Option<f64>,
}

impl PiUsage {
    /// `piUsageSchema` (`.optional().catch(undefined)`).
    pub fn parse(value: Option<&JsonValue>) -> Option<Self> {
        let usage = value?.as_object()?;
        let number = |key: &str| usage.get(key).and_then(JsonValue::as_f64);
        let reasoning = match usage.get("reasoning") {
            None => None,
            Some(value) => Some(value.as_f64()?),
        };
        Some(Self {
            input: number("input")?,
            output: number("output")?,
            cache_read: number("cacheRead")?,
            cache_write: number("cacheWrite")?,
            reasoning,
        })
    }
}

/// `PiSessionEvent`: a Pi session event that passed `piSessionEventSchema`.
#[derive(Debug, Clone, PartialEq)]
pub struct PiSessionEvent {
    raw: Map<String, JsonValue>,
}

fn optional_is(value: Option<&JsonValue>, check: impl Fn(&JsonValue) -> bool) -> bool {
    value.is_none_or(check)
}

fn optional_loose_object(
    value: Option<&JsonValue>,
    check: impl Fn(&Map<String, JsonValue>) -> bool,
) -> bool {
    optional_is(value, |value| value.as_object().is_some_and(&check))
}

impl PiSessionEvent {
    /// Build an event from a JSON object without validation (tests and
    /// synthesized events).
    pub fn from_json(raw: JsonValue) -> Option<Self> {
        parse_native_event(&raw)
    }

    /// The raw event object.
    pub fn raw(&self) -> &Map<String, JsonValue> {
        &self.raw
    }

    /// `type`.
    pub fn event_type(&self) -> &str {
        self.raw["type"].as_str().unwrap_or_default()
    }

    /// Any field, as JSON.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.raw.get(key)
    }

    fn string(&self, key: &str) -> Option<&str> {
        self.raw.get(key).and_then(JsonValue::as_str)
    }

    /// `toolCallId`.
    pub fn tool_call_id(&self) -> Option<&str> {
        self.string("toolCallId")
    }

    /// `toolName`.
    pub fn tool_name(&self) -> Option<&str> {
        self.string("toolName")
    }

    /// `reason`.
    pub fn reason(&self) -> Option<&str> {
        self.string("reason")
    }

    /// `aborted`.
    pub fn aborted(&self) -> bool {
        self.raw.get("aborted").and_then(JsonValue::as_bool) == Some(true)
    }

    /// `isError`.
    pub fn is_error(&self) -> bool {
        self.raw.get("isError").and_then(JsonValue::as_bool) == Some(true)
    }

    /// `assistantMessageEvent`.
    pub fn assistant_message_event(&self) -> Option<&Map<String, JsonValue>> {
        self.raw
            .get("assistantMessageEvent")
            .and_then(JsonValue::as_object)
    }

    /// `message`.
    pub fn message(&self) -> Option<&Map<String, JsonValue>> {
        self.raw.get("message").and_then(JsonValue::as_object)
    }

    /// `message.role`.
    pub fn message_role(&self) -> Option<&str> {
        self.message()?.get("role")?.as_str()
    }

    /// `message.usage` (invalid usage parses as absent).
    pub fn message_usage(&self) -> Option<PiUsage> {
        PiUsage::parse(self.message()?.get("usage"))
    }
}

/// `parseNativeEvent(raw)`: decode an unknown raw event into a
/// `PiSessionEvent` if it looks like one.
pub fn parse_native_event(raw: &JsonValue) -> Option<PiSessionEvent> {
    let object = raw.as_object()?;
    object.get("type")?.as_str()?;
    let string = |key: &str| optional_is(object.get(key), JsonValue::is_string);
    let valid = optional_loose_object(object.get("assistantMessageEvent"), |update| {
        optional_is(update.get("type"), JsonValue::is_string)
            && optional_is(update.get("delta"), JsonValue::is_string)
            && optional_is(update.get("contentIndex"), JsonValue::is_number)
            && optional_loose_object(update.get("partial"), |partial| {
                optional_is(partial.get("content"), JsonValue::is_array)
            })
    }) && string("toolCallId")
        && string("toolName")
        && optional_is(object.get("isError"), JsonValue::is_boolean)
        && string("reason")
        && optional_is(object.get("aborted"), JsonValue::is_boolean)
        && optional_is(object.get("error"), |error| {
            error.is_string()
                || error.as_object().is_some_and(|error| {
                    optional_is(error.get("errorMessage"), JsonValue::is_string)
                        && optional_is(error.get("stopReason"), JsonValue::is_string)
                })
        })
        && optional_loose_object(object.get("message"), |message| {
            optional_is(message.get("role"), JsonValue::is_string)
                && optional_is(message.get("stopReason"), JsonValue::is_string)
                && optional_is(message.get("errorMessage"), JsonValue::is_string)
        });
    if !valid {
        return None;
    }
    let mut raw = object.clone();
    // `usage: piUsageSchema` catches to `undefined`.
    if let Some(message) = raw.get_mut("message").and_then(JsonValue::as_object_mut)
        && message.contains_key("usage")
        && PiUsage::parse(message.get("usage")).is_none()
    {
        message.remove("usage");
    }
    Some(PiSessionEvent { raw })
}

fn trimmed(value: Option<&JsonValue>) -> Option<&str> {
    value
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// `getPiTerminalError(event)`: whether a Pi event signals a terminal error
/// for the current turn, with its message.
pub fn get_pi_terminal_error(event: &PiSessionEvent) -> Option<String> {
    let is_terminal_stop_reason = |value: Option<&str>| matches!(value, Some("error" | "aborted"));

    if let Some(error) = trimmed(event.get("error")) {
        return Some(error.to_string());
    }
    if let Some(error) = event.get("error").and_then(JsonValue::as_object) {
        if let Some(message) = trimmed(error.get("errorMessage")) {
            return Some(message.to_string());
        }
        let stop_reason = error
            .get("stopReason")
            .and_then(JsonValue::as_str)
            .map(str::trim);
        if is_terminal_stop_reason(stop_reason) {
            return stop_reason.map(String::from);
        }
    }
    let message = event.message();
    if let Some(message) = trimmed(message.and_then(|message| message.get("errorMessage"))) {
        return Some(message.to_string());
    }
    let stop_reason = message
        .and_then(|message| message.get("stopReason"))
        .and_then(JsonValue::as_str)
        .map(str::trim);
    if is_terminal_stop_reason(stop_reason) {
        return stop_reason.map(String::from);
    }
    if event.is_error()
        && let Some(content) = trimmed(event.get("content"))
    {
        return Some(content.to_string());
    }
    None
}

/// `extractAssistantText(message)`: the assistant text of a `turn_end` /
/// `message_end` event payload.
pub fn extract_assistant_text(message: Option<&Map<String, JsonValue>>) -> String {
    let Some(message) = message else {
        return String::new();
    };
    if message.get("role").and_then(JsonValue::as_str) != Some("assistant") {
        return String::new();
    }
    match message.get("content") {
        Some(JsonValue::String(content)) => content.clone(),
        Some(JsonValue::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                let part = part.as_object()?;
                (part.get("type").and_then(JsonValue::as_str) == Some("text"))
                    .then(|| part.get("text").and_then(JsonValue::as_str))
                    .flatten()
            })
            .collect(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn event(value: JsonValue) -> PiSessionEvent {
        PiSessionEvent {
            raw: value.as_object().unwrap().clone(),
        }
    }

    #[test]
    fn parse_parses_a_well_formed_event() {
        let event = parse_native_event(&json!({ "type": "turn_start" })).unwrap();
        assert_eq!(event.event_type(), "turn_start");
    }

    #[test]
    fn parse_passes_unknown_fields_through() {
        let event =
            parse_native_event(&json!({ "type": "custom", "extra": { "nested": true } })).unwrap();
        assert_eq!(event.get("extra"), Some(&json!({ "nested": true })));
    }

    #[test]
    fn parse_returns_undefined_for_malformed_input() {
        assert!(parse_native_event(&JsonValue::Null).is_none());
        assert!(parse_native_event(&json!({})).is_none());
        assert!(parse_native_event(&json!({ "type": 42 })).is_none());
    }

    #[test]
    fn terminal_error_extracts_a_string_error() {
        assert_eq!(
            get_pi_terminal_error(&event(json!({ "type": "x", "error": "boom" }))).as_deref(),
            Some("boom")
        );
    }

    #[test]
    fn terminal_error_extracts_error_error_message() {
        assert_eq!(
            get_pi_terminal_error(&event(
                json!({ "type": "x", "error": { "errorMessage": "oops" } })
            ))
            .as_deref(),
            Some("oops")
        );
    }

    #[test]
    fn terminal_error_returns_stop_reason_when_it_is_error_or_aborted() {
        for reason in ["error", "aborted"] {
            assert_eq!(
                get_pi_terminal_error(&event(
                    json!({ "type": "x", "error": { "stopReason": reason } })
                ))
                .as_deref(),
                Some(reason)
            );
        }
    }

    #[test]
    fn terminal_error_ignores_non_terminal_stop_reasons() {
        assert_eq!(
            get_pi_terminal_error(&event(
                json!({ "type": "x", "error": { "stopReason": "tool_use" } })
            )),
            None
        );
    }

    #[test]
    fn terminal_error_extracts_message_error_message() {
        assert_eq!(
            get_pi_terminal_error(&event(
                json!({ "type": "x", "message": { "errorMessage": "msg" } })
            ))
            .as_deref(),
            Some("msg")
        );
    }

    #[test]
    fn terminal_error_extracts_is_error_and_content() {
        assert_eq!(
            get_pi_terminal_error(&event(
                json!({ "type": "x", "isError": true, "content": "failed" })
            ))
            .as_deref(),
            Some("failed")
        );
    }

    #[test]
    fn terminal_error_returns_undefined_for_normal_events() {
        assert_eq!(
            get_pi_terminal_error(&event(json!({ "type": "text_delta" }))),
            None
        );
    }

    fn message(value: JsonValue) -> Map<String, JsonValue> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn assistant_text_returns_empty_when_no_message() {
        assert_eq!(extract_assistant_text(None), "");
    }

    #[test]
    fn assistant_text_returns_empty_for_non_assistant_messages() {
        assert_eq!(
            extract_assistant_text(Some(&message(json!({ "role": "user", "content": "hi" })))),
            ""
        );
    }

    #[test]
    fn assistant_text_returns_string_content_directly() {
        assert_eq!(
            extract_assistant_text(Some(&message(
                json!({ "role": "assistant", "content": "hello" })
            ))),
            "hello"
        );
    }

    #[test]
    fn assistant_text_concatenates_text_parts_from_array_content() {
        assert_eq!(
            extract_assistant_text(Some(&message(json!({
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "one" },
                    { "type": "tool_use", "name": "bash" },
                    { "type": "text", "text": "two" },
                ],
            })))),
            "onetwo"
        );
    }
}
