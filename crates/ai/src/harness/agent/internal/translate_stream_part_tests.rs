use serde_json::json;

use super::*;

fn ev(value: JsonValue) -> HarnessV1StreamPart {
    HarnessV1StreamPart::parse(value).unwrap()
}

fn tr(value: JsonValue) -> Vec<JsonValue> {
    to_json(translate_stream_part(&ev(value), &TranslateOptions::default()))
}

fn to_json(parts: Vec<TextStreamPart>) -> Vec<JsonValue> {
    parts
        .into_iter()
        .map(|p| serde_json::to_value(p).unwrap())
        .collect()
}

fn contains(actual: &JsonValue, expected: JsonValue) {
    for (k, v) in expected.as_object().unwrap() {
        assert_eq!(&actual[k], v, "field {k} of {actual}");
    }
}

#[test]
fn forwards_streaming_tool_input_events() {
    let events = vec![
        json!({"type": "tool-input-start", "id": "c1", "toolName": "write", "providerExecuted": true}),
        json!({"type": "tool-input-delta", "id": "c1", "delta": "{\"path\":\"notes.md\""}),
        json!({"type": "tool-input-end", "id": "c1"}),
    ];
    let out: Vec<JsonValue> = events.iter().flat_map(|e| tr(e.clone())).collect();
    assert_eq!(out, events);
}

#[test]
fn returns_no_parts_for_a_tool_call_event() {
    let out = tr(json!({
        "type": "tool-call", "toolCallId": "c1", "toolName": "bash",
        "input": "{\"command\":\"ls\"}", "nativeName": "Bash", "providerExecuted": true
    }));
    assert!(out.is_empty());
}

#[test]
fn preserves_dynamic_on_a_tool_result_event() {
    let out = tr(json!({
        "type": "tool-result", "toolCallId": "c1", "toolName": "mcp__weather__current",
        "result": {"temperature": 72}, "isError": false, "dynamic": true
    }));
    assert_eq!(out.len(), 1);
    contains(
        &out[0],
        json!({"type": "tool-result", "toolCallId": "c1", "toolName": "mcp__weather__current", "dynamic": true}),
    );
}

#[test]
fn restores_tool_result_input_and_execution_origin_from_its_tool_call() {
    let get = |_: &str| {
        Some(TranslatedToolCallInfo {
            input: json!({"city": "San Francisco"}),
            provider_executed: Some(true),
        })
    };
    let out = to_json(translate_stream_part(
        &ev(json!({"type": "tool-result", "toolCallId": "c1", "toolName": "weather", "result": {"temperature": 72}})),
        &TranslateOptions {
            get_tool_call: Some(&get),
            ..Default::default()
        },
    ));
    assert_eq!(out.len(), 1);
    contains(
        &out[0],
        json!({
            "type": "tool-result", "toolCallId": "c1", "input": {"city": "San Francisco"},
            "output": {"temperature": 72}, "providerExecuted": true
        }),
    );
}

#[test]
fn translates_a_failed_tool_result_into_a_provider_executed_tool_error() {
    let out = tr(json!({
        "type": "tool-result", "toolCallId": "c1", "toolName": "bash",
        "result": "bash: command not found: pnpmm", "isError": true,
        "providerMetadata": {"claude-code": {"subtype": "error_during_execution"}}
    }));
    assert_eq!(
        out,
        vec![json!({
            "type": "tool-error", "toolCallId": "c1", "toolName": "bash",
            "error": "bash: command not found: pnpmm", "providerExecuted": true,
            "providerMetadata": {"claude-code": {"subtype": "error_during_execution"}}
        })]
    );
}

#[test]
fn leaves_a_failed_host_tool_result_as_a_tool_result() {
    let host = |_: &str| false;
    let out = to_json(translate_stream_part(
        &ev(json!({
            "type": "tool-result", "toolCallId": "c1", "toolName": "weather",
            "result": {"error": "Error: boom"}, "isError": true
        })),
        &TranslateOptions {
            is_provider_executed: Some(&host),
            ..Default::default()
        },
    ));
    assert_eq!(out.len(), 1);
    contains(
        &out[0],
        json!({"type": "tool-result", "toolCallId": "c1", "output": {"error": "Error: boom"}}),
    );
}

#[test]
fn preserves_dynamic_on_a_failed_tool_result() {
    let out = tr(json!({
        "type": "tool-result", "toolCallId": "c1", "toolName": "mcp__weather__current",
        "result": {"message": "upstream timeout"}, "isError": true, "dynamic": true
    }));
    assert_eq!(out.len(), 1);
    contains(
        &out[0],
        json!({
            "type": "tool-error", "toolCallId": "c1", "toolName": "mcp__weather__current",
            "error": {"message": "upstream timeout"}, "providerExecuted": true, "dynamic": true
        }),
    );
}

#[test]
fn restores_tool_error_input_from_its_provider_executed_tool_call() {
    let get = |_: &str| {
        Some(TranslatedToolCallInfo {
            input: json!({"city": "San Francisco"}),
            provider_executed: Some(true),
        })
    };
    let out = to_json(translate_stream_part(
        &ev(json!({
            "type": "tool-result", "toolCallId": "c1", "toolName": "weather",
            "result": "upstream timeout", "isError": true
        })),
        &TranslateOptions {
            get_tool_call: Some(&get),
            ..Default::default()
        },
    ));
    assert_eq!(out.len(), 1);
    contains(
        &out[0],
        json!({
            "type": "tool-error", "toolCallId": "c1", "input": {"city": "San Francisco"},
            "error": "upstream timeout", "providerExecuted": true
        }),
    );
}

#[test]
fn fans_file_change_out_into_a_tool_call_and_tool_result_pair() {
    let out = tr(json!({"type": "file-change", "event": "create", "path": "notes.md"}));
    assert_eq!(out.len(), 2);
    let (call, result) = (&out[0], &out[1]);
    let payload = json!({"event": "create", "path": "notes.md"});
    contains(
        call,
        json!({"type": "tool-call", "toolName": "fileChange", "input": payload, "dynamic": true, "providerExecuted": true}),
    );
    contains(
        result,
        json!({"type": "tool-result", "toolName": "fileChange", "output": payload, "dynamic": true, "providerExecuted": true}),
    );
    assert_eq!(result["toolCallId"], call["toolCallId"]);
    assert!(
        call["toolCallId"]
            .as_str()
            .unwrap()
            .starts_with("harness-file-change-")
    );
}

#[test]
fn fans_compaction_out_into_a_pair_with_empty_input_and_metadata_output() {
    let out = tr(json!({
        "type": "compaction", "trigger": "auto",
        "summary": "Summarized the earlier conversation.",
        "tokensBefore": 120000, "tokensAfter": 38000
    }));
    assert_eq!(out.len(), 2);
    let (call, result) = (&out[0], &out[1]);
    contains(
        call,
        json!({"type": "tool-call", "toolName": "compaction", "input": {}, "dynamic": true, "providerExecuted": true}),
    );
    contains(
        result,
        json!({
            "type": "tool-result", "toolName": "compaction", "input": {},
            "output": {
                "trigger": "auto", "summary": "Summarized the earlier conversation.",
                "tokensBefore": 120000, "tokensAfter": 38000
            },
            "dynamic": true, "providerExecuted": true
        }),
    );
    assert_eq!(result["toolCallId"], call["toolCallId"]);
    assert!(
        call["toolCallId"]
            .as_str()
            .unwrap()
            .starts_with("harness-compaction-")
    );
}

#[test]
fn omits_optional_token_fields_on_the_compaction_output_when_absent() {
    let out = tr(json!({"type": "compaction", "trigger": "manual", "summary": "Compacted."}));
    assert_eq!(
        out[1]["output"],
        json!({"trigger": "manual", "summary": "Compacted."})
    );
}

#[test]
fn returns_empty_for_events_consumed_internally() {
    assert!(tr(json!({"type": "stream-start"})).is_empty());
    let usage = json!({"inputTokens": {"total": 0}, "outputTokens": {"total": 0}});
    assert!(
        tr(json!({"type": "finish-step", "finishReason": {"unified": "stop", "raw": "stop"}, "usage": usage}))
            .is_empty()
    );
    assert!(
        tr(json!({"type": "finish", "finishReason": {"unified": "stop", "raw": "stop"}, "totalUsage": usage}))
            .is_empty()
    );
}
