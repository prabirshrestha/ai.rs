//! Port of `agent/internal/validate-tool-call.test.ts`.
//!
//! [rust] `validateToolCall` is synchronous here, and `TypedToolCall` has
//! no `type` field, so `toMatchObject` checks compare its JSON without it.

use serde_json::json;

use super::tests::{matches, p, to_json, tool, tools, z_object};
use super::*;
use crate::harness::sdk::tool::Tool;

fn bash_tools() -> ToolSet {
    tools(vec![(
        "bash",
        tool(&[("command", "string")]).with_description("Run a shell command"),
    )])
}

fn call(harness_id: &str, event: JsonValue, tools: Option<&ToolSet>) -> JsonValue {
    let mut event = event;
    event["type"] = json!("tool-call");
    to_json(&validate_tool_call(&p(event), tools, harness_id))
}

fn assert_matches(actual: &JsonValue, expected: JsonValue) {
    assert!(matches(actual, &expected), "{actual:#}\n!~ {expected:#}");
}

#[test]
fn parses_a_valid_tool_call_input_against_the_merged_tool_schema() {
    let result = call(
        "claude-code",
        json!({ "toolCallId": "c1", "toolName": "bash", "input": "{\"command\":\"ls\"}", "nativeName": "Bash", "providerExecuted": true }),
        Some(&bash_tools()),
    );
    assert_matches(
        &result,
        json!({
            "toolCallId": "c1",
            "toolName": "bash",
            "input": { "command": "ls" },
            "providerExecuted": true,
            "providerMetadata": { "claude-code": { "nativeToolName": "Bash" } },
        }),
    );
    assert!(result.get("invalid").is_none());
}

#[test]
fn marks_the_part_invalid_when_the_input_fails_the_schema() {
    let result = call(
        "claude-code",
        json!({ "toolCallId": "c2", "toolName": "bash", "input": "{\"foo\":\"bar\"}", "nativeName": "Bash", "providerExecuted": true }),
        Some(&bash_tools()),
    );
    assert_matches(
        &result,
        json!({
            "toolCallId": "c2",
            "toolName": "bash",
            "invalid": true,
            "providerMetadata": { "claude-code": { "nativeToolName": "Bash" } },
        }),
    );
    // raw parsed JSON preserved on the invalid path
    assert_eq!(result["input"], json!({ "foo": "bar" }));
}

#[test]
fn marks_the_part_invalid_and_dynamic_when_the_tool_is_not_declared() {
    let tools = bash_tools();
    let event = p(
        json!({ "type": "tool-call", "toolCallId": "c3", "toolName": "unknown", "input": "{\"x\":1}" }),
    );
    let typed = validate_tool_call(&event, Some(&tools), "fake");
    assert_matches(
        &to_json(&typed),
        json!({ "toolCallId": "c3", "toolName": "unknown", "invalid": true, "dynamic": true }),
    );
    // [rust] `toBeInstanceOf(Error)`: an error value is present.
    assert!(typed.error.is_some());
}

#[test]
fn accepts_an_undeclared_provider_executed_dynamic_tool() {
    let result = call(
        "fake",
        json!({ "toolCallId": "c4", "toolName": "runtimeTool", "input": "{\"value\":\"test\"}", "providerExecuted": true, "dynamic": true }),
        Some(&bash_tools()),
    );
    assert_matches(
        &result,
        json!({
            "toolCallId": "c4",
            "toolName": "runtimeTool",
            "input": { "value": "test" },
            "providerExecuted": true,
            "dynamic": true,
        }),
    );
    assert!(result.get("invalid").is_none());
}

#[test]
fn parses_a_provider_executed_dynamic_tool_without_a_static_tool_set() {
    let result = call(
        "fake",
        json!({ "toolCallId": "c4-replacement", "toolName": "bash", "input": "{\"query\":\"test\"}", "providerExecuted": true, "dynamic": true }),
        None,
    );
    assert_matches(
        &result,
        json!({
            "toolCallId": "c4-replacement",
            "toolName": "bash",
            "input": { "query": "test" },
            "providerExecuted": true,
            "dynamic": true,
        }),
    );
    assert!(result.get("invalid").is_none());
}

#[test]
fn marks_malformed_provider_executed_dynamic_input_invalid_without_a_static_tool_set() {
    let result = call(
        "fake",
        json!({ "toolCallId": "c4-malformed", "toolName": "bash", "input": "{", "providerExecuted": true, "dynamic": true }),
        None,
    );
    assert_matches(
        &result,
        json!({ "toolCallId": "c4-malformed", "toolName": "bash", "invalid": true, "dynamic": true }),
    );
}

#[test]
fn omits_provider_executed_when_the_bridge_did_not_set_it() {
    let result = call(
        "fake",
        json!({ "toolCallId": "c5", "toolName": "bash", "input": "{\"command\":\"ls\"}" }),
        Some(&bash_tools()),
    );
    assert!(result.get("providerExecuted").is_none());
}

#[test]
fn treats_an_empty_input_string_as_an_empty_object_before_validating() {
    let no_args = tools(vec![(
        "empty",
        Tool::new(z_object(&[])).with_description("No args"),
    )]);
    let result = call(
        "fake",
        json!({ "toolCallId": "c6", "toolName": "empty", "input": "", "providerExecuted": true }),
        Some(&no_args),
    );
    assert_matches(
        &result,
        json!({ "toolCallId": "c6", "toolName": "empty", "input": {} }),
    );
}

#[test]
fn merges_the_native_tool_name_into_the_harness_namespace_without_mutating_existing_metadata() {
    let provider_metadata = json!({
        "runtime": { "existing": "value" },
        "other": { "existing": "other value" },
    });
    let event = p(json!({
        "type": "tool-call",
        "toolCallId": "c1",
        "toolName": "bash",
        "input": "{\"command\":\"echo hello\"}",
        "nativeName": "Bash",
        "providerExecuted": true,
        "providerMetadata": provider_metadata,
    }));
    let result = to_json(&validate_tool_call(&event, Some(&bash_tools()), "runtime"));
    assert_matches(
        &result,
        json!({
            "providerMetadata": {
                "runtime": { "existing": "value", "nativeToolName": "Bash" },
                "other": { "existing": "other value" },
            },
        }),
    );
    let HarnessV1StreamPart::ToolCall {
        provider_metadata: original,
        ..
    } = &event
    else {
        unreachable!()
    };
    assert_eq!(to_json(original), provider_metadata);
}

#[test]
fn preserves_existing_metadata_when_native_name_is_absent_or_matches() {
    for native_name in [None, Some("bash")] {
        let provider_metadata = json!({ "runtime": { "existing": "value" } });
        let mut event = json!({
            "toolCallId": "c1",
            "toolName": "bash",
            "input": "{\"command\":\"echo hello\"}",
            "providerMetadata": provider_metadata,
        });
        if let Some(name) = native_name {
            event["nativeName"] = json!(name);
        }
        let result = call("runtime", event, Some(&bash_tools()));
        assert_eq!(
            result["providerMetadata"], provider_metadata,
            "{native_name:?}"
        );
    }
}

#[test]
fn does_not_add_metadata_when_native_name_is_absent_or_matches() {
    for native_name in [None, Some("bash")] {
        let mut event = json!({
            "toolCallId": "c1",
            "toolName": "bash",
            "input": "{\"command\":\"echo hello\"}",
        });
        if let Some(name) = native_name {
            event["nativeName"] = json!(name);
        }
        let result = call("runtime", event, Some(&bash_tools()));
        assert!(result.get("providerMetadata").is_none(), "{native_name:?}");
    }
}
