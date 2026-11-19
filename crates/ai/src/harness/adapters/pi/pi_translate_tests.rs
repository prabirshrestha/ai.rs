//! Port of `harness-pi/src/pi-translate.test.ts`.
//!
//! [rust] Parts are compared as their JSON serialization. TS
//! `details: undefined` is an absent key; the "keeps `details` on the
//! input" checks have no counterpart because the input is not mutated.

use std::collections::HashSet;

use serde_json::{Value, json};

use super::pi_events::PiSessionEvent;
use super::pi_translate::*;
use crate::harness::adapters::bridge_test_support::assert_match_object;

fn ev(value: Value) -> PiSessionEvent {
    PiSessionEvent::from_json(value).expect("valid Pi event")
}

fn to_json(parts: Vec<crate::harness::spec::stream_part::HarnessV1StreamPart>) -> Vec<Value> {
    parts
        .into_iter()
        .map(|part| serde_json::to_value(part).unwrap())
        .collect()
}

fn translate(event: Value, state: &mut PiTranslatorState) -> Vec<Value> {
    to_json(translate_pi_event(&ev(event), state))
}

fn emit(events: Vec<Value>, state: &mut PiTranslatorState) -> Vec<Value> {
    events
        .into_iter()
        .flat_map(|event| translate(event, state))
        .collect()
}

fn state() -> PiTranslatorState {
    create_pi_translator_state(PiTranslatorStateOptions::default())
}

fn state_with(builtin: &[&str], host: &[&str]) -> PiTranslatorState {
    create_pi_translator_state(PiTranslatorStateOptions {
        builtin_tool_names: builtin.iter().map(|name| name.to_string()).collect(),
        host_tool_names: host.iter().map(|name| name.to_string()).collect(),
        native_to_common: Default::default(),
    })
}

fn types(parts: &[Value]) -> Vec<&str> {
    parts
        .iter()
        .map(|part| part["type"].as_str().unwrap())
        .collect()
}

fn update(value: Value) -> Value {
    json!({ "type": "message_update", "assistantMessageEvent": value })
}

fn set(ids: &[&str]) -> HashSet<String> {
    ids.iter().map(|id| id.to_string()).collect()
}

fn start_tool(state: &mut PiTranslatorState, id: &str, name: &str) {
    translate(json!({ "type": "turn_start" }), state);
    translate(
        json!({ "type": "tool_execution_start", "toolCallId": id, "toolName": name, "args": {} }),
        state,
    );
}

#[test]
fn projects_text_without_an_undefined_details_member() {
    for text in ["", "native output"] {
        let mut state = state();
        start_tool(&mut state, "file-call", "ls");
        let events = translate(
            json!({
                "type": "tool_execution_end",
                "toolCallId": "file-call",
                "toolName": "ls",
                "result": { "content": [{ "type": "text", "text": text }] },
            }),
            &mut state,
        );
        assert_match_object(
            &events[0],
            &json!({ "type": "tool-result", "toolCallId": "file-call", "result": text }),
        );
    }
}

#[test]
fn retains_native_result_details_and_error_semantics() {
    for is_error in [false, true] {
        let mut state = state();
        start_tool(&mut state, "file-call", "grep");
        let result = json!({
            "content": [{ "type": "text", "text": "partial output" }],
            "details": { "truncated": true, "nextOffset": 20 },
        });
        let events = translate(
            json!({
                "type": "tool_execution_end",
                "toolCallId": "file-call",
                "toolName": "grep",
                "result": result,
                "isError": is_error,
            }),
            &mut state,
        );
        assert_match_object(&events[0], &json!({ "type": "tool-result", "result": result }));
        assert_eq!(
            events[0].get("isError").and_then(Value::as_bool).unwrap_or(false),
            is_error
        );
    }
}

#[test]
fn retains_non_text_results_in_a_json_safe_envelope() {
    let cases = [
        (
            "image content",
            json!({ "content": [{ "type": "image", "data": "AA==", "mimeType": "image/png" }] }),
            json!({ "content": [{ "type": "image", "data": "AA==", "mimeType": "image/png" }] }),
        ),
        (
            "mixed text and image content",
            json!({ "content": [
                { "type": "text", "text": "caption" },
                { "type": "image", "data": "AA==", "mimeType": "image/png" },
            ] }),
            json!({ "content": [
                { "type": "text", "text": "caption" },
                { "type": "image", "data": "AA==", "mimeType": "image/png" },
            ] }),
        ),
        (
            "an empty content array",
            json!({ "content": [] }),
            json!({ "content": [] }),
        ),
        (
            "additional envelope fields",
            json!({ "content": [{ "type": "text", "text": "done" }], "terminate": true }),
            json!({ "content": [{ "type": "text", "text": "done" }], "terminate": true }),
        ),
    ];
    for (name, result, expected) in cases {
        let mut state = state();
        start_tool(&mut state, "file-call", "read");
        let events = translate(
            json!({
                "type": "tool_execution_end",
                "toolCallId": "file-call",
                "toolName": "read",
                "result": result,
            }),
            &mut state,
        );
        assert_eq!(events[0]["result"], expected, "{name}");
    }
}

#[test]
fn projects_empty_text_from_flat_tool_results() {
    let mut state = state();
    start_tool(&mut state, "file-call", "ls");
    let events = translate(
        json!({
            "type": "tool_result",
            "toolCallId": "file-call",
            "toolName": "ls",
            "content": [{ "type": "text", "text": "" }],
        }),
        &mut state,
    );
    assert_match_object(&events[0], &json!({ "type": "tool-result", "result": "" }));
}

#[test]
fn drops_events_before_turn_start() {
    let mut state = state();
    let out = translate(
        update(json!({ "type": "text_delta", "delta": "hi" })),
        &mut state,
    );
    assert!(out.is_empty());
}

#[test]
fn emits_text_start_before_the_first_text_delta() {
    let mut state = state();
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({ "type": "text_delta", "delta": "Hello " })),
            update(json!({ "type": "text_delta", "delta": "world" })),
        ],
        &mut state,
    );
    assert_eq!(out[0]["type"], "text-start");
    let id = out[0]["id"].clone();
    assert_match_object(
        &out[1],
        &json!({ "type": "text-delta", "id": id, "delta": "Hello " }),
    );
    assert_match_object(
        &out[2],
        &json!({ "type": "text-delta", "id": id, "delta": "world" }),
    );
}

#[test]
fn streams_tool_input_as_the_model_writes_it() {
    let mut state = state_with(&[], &["visualize"]);
    let partial = |args: &str| {
        json!({ "content": [{ "type": "toolCall", "id": "call-1", "name": "visualize", "args": args }] })
    };
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({ "type": "toolcall_start", "contentIndex": 0, "partial": partial("") })),
            update(json!({
                "type": "toolcall_delta", "contentIndex": 0, "delta": "{\"html\":\"<h1>",
                "partial": partial("{\"html\":\"<h1>"),
            })),
            update(json!({
                "type": "toolcall_delta", "contentIndex": 0, "delta": "hi</h1>\"}",
                "partial": partial("{\"html\":\"<h1>hi</h1>\"}"),
            })),
            update(json!({
                "type": "toolcall_end", "contentIndex": 0,
                "partial": partial("{\"html\":\"<h1>hi</h1>\"}"),
            })),
        ],
        &mut state,
    );
    assert_eq!(
        out,
        vec![
            json!({ "type": "tool-input-start", "id": "call-1", "toolName": "visualize" }),
            json!({ "type": "tool-input-delta", "id": "call-1", "delta": "{\"html\":\"<h1>" }),
            json!({ "type": "tool-input-delta", "id": "call-1", "delta": "hi</h1>\"}" }),
            json!({ "type": "tool-input-end", "id": "call-1" }),
        ]
    );
}

/// [rust] Pi's RPC wire format drops `partial` and carries `id` and
/// `toolName` on `toolcall_start` (`docs/json.md`).
#[test]
fn streams_tool_input_from_the_rpc_wire_shape() {
    let mut state = state_with(&[], &["visualize"]);
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({
                "type": "toolcall_start", "contentIndex": 0, "id": "call-1", "toolName": "visualize",
            })),
            update(json!({ "type": "toolcall_delta", "contentIndex": 0, "delta": "{}" })),
            update(json!({ "type": "toolcall_end", "contentIndex": 0, "toolCall": {} })),
        ],
        &mut state,
    );
    assert_eq!(
        out,
        vec![
            json!({ "type": "tool-input-start", "id": "call-1", "toolName": "visualize" }),
            json!({ "type": "tool-input-delta", "id": "call-1", "delta": "{}" }),
            json!({ "type": "tool-input-end", "id": "call-1" }),
        ]
    );
}

#[test]
fn reports_the_same_dispatch_on_tool_input_start_as_on_the_tool_call() {
    let mut state = state_with(&["read"], &[]);
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({
                "type": "toolcall_start", "contentIndex": 0,
                "partial": { "content": [{ "type": "toolCall", "id": "call-1", "name": "read" }] },
            })),
            json!({
                "type": "tool_execution_start", "toolCallId": "call-1", "toolName": "read",
                "args": { "path": "a.txt" },
            }),
        ],
        &mut state,
    );
    assert_match_object(
        &out[0],
        &json!({ "type": "tool-input-start", "id": "call-1", "providerExecuted": true }),
    );
    assert_match_object(
        &out[1],
        &json!({ "type": "tool-call", "toolCallId": "call-1", "providerExecuted": true }),
    );
}

#[test]
fn marks_a_streamed_mcp_tool_input_as_dynamic() {
    let mut state = state();
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({
                "type": "toolcall_start", "contentIndex": 0,
                "partial": { "content": [{ "type": "toolCall", "id": "call-1", "name": "mcp__linear__issue" }] },
            })),
        ],
        &mut state,
    );
    assert_match_object(
        &out[0],
        &json!({ "type": "tool-input-start", "id": "call-1", "dynamic": true, "providerExecuted": true }),
    );
}

#[test]
fn keeps_tool_inputs_of_concurrent_calls_on_their_own_ids() {
    let mut state = state();
    let content = json!([
        { "type": "toolCall", "id": "call-1", "name": "read" },
        { "type": "toolCall", "id": "call-2", "name": "read" },
    ]);
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(
                json!({ "type": "toolcall_start", "contentIndex": 0, "partial": { "content": content } }),
            ),
            update(
                json!({ "type": "toolcall_start", "contentIndex": 1, "partial": { "content": content } }),
            ),
            update(json!({
                "type": "toolcall_delta", "contentIndex": 1, "delta": "b", "partial": { "content": content },
            })),
            update(json!({
                "type": "toolcall_delta", "contentIndex": 0, "delta": "a", "partial": { "content": content },
            })),
            json!({ "type": "tool_execution_start", "toolCallId": "call-2", "toolName": "read", "args": {} }),
        ],
        &mut state,
    );
    let ids: Vec<(String, String)> = out
        .iter()
        .map(|part| {
            (
                part["type"].as_str().unwrap().to_string(),
                part.get("id")
                    .or_else(|| part.get("toolCallId"))
                    .and_then(Value::as_str)
                    .unwrap()
                    .to_string(),
            )
        })
        .collect();
    let expected: Vec<(String, String)> = [
        ("tool-input-start", "call-1"),
        ("tool-input-delta", "call-1"),
        ("tool-input-start", "call-2"),
        ("tool-input-delta", "call-2"),
        ("tool-call", "call-2"),
    ]
    .iter()
    .map(|(kind, id)| (kind.to_string(), id.to_string()))
    .collect();
    assert_eq!(ids, expected);
    assert_eq!(
        out[3],
        json!({ "type": "tool-input-delta", "id": "call-2", "delta": "b" })
    );
}

#[test]
fn drops_tool_input_deltas_that_arrive_without_a_start() {
    let mut state = state();
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({ "type": "toolcall_delta", "contentIndex": 0, "delta": "{\"a\":1}" })),
        ],
        &mut state,
    );
    assert!(out.is_empty());
}

#[test]
fn does_not_stream_a_tool_input_whose_id_is_not_known_yet() {
    let mut state = state();
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({
                "type": "toolcall_start", "contentIndex": 0,
                "partial": { "content": [{ "type": "toolCall", "id": "", "name": "" }] },
            })),
            update(json!({ "type": "toolcall_delta", "contentIndex": 0, "delta": "{" })),
        ],
        &mut state,
    );
    assert!(out.is_empty());
}

#[test]
fn does_not_carry_tool_input_ids_across_assistant_messages() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({
                "type": "toolcall_start", "contentIndex": 0,
                "partial": { "content": [{ "type": "toolCall", "id": "call-1", "name": "read" }] },
            })),
        ],
        &mut state,
    );
    let out = emit(
        vec![
            json!({ "type": "message_start", "message": { "role": "assistant" } }),
            update(json!({ "type": "toolcall_delta", "contentIndex": 0, "delta": "{" })),
        ],
        &mut state,
    );
    assert!(out.is_empty());
}

#[test]
fn gap_fills_missing_text_at_turn_end_and_emits_text_end() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({ "type": "text_delta", "delta": "partial" })),
        ],
        &mut state,
    );
    let closing = translate(
        json!({ "type": "turn_end", "message": { "role": "assistant", "content": "partial and more" } }),
        &mut state,
    );
    assert_match_object(
        &closing[0],
        &json!({ "type": "text-delta", "delta": " and more" }),
    );
    assert_eq!(closing[1]["type"], "text-end");
}

#[test]
fn emits_text_end_without_gap_fill_when_nothing_is_missing() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({ "type": "text_delta", "delta": "complete" })),
        ],
        &mut state,
    );
    let closing = translate(
        json!({ "type": "message_end", "message": { "role": "assistant", "content": "complete" } }),
        &mut state,
    );
    assert!(!types(&closing).contains(&"text-delta"));
    assert!(types(&closing).contains(&"text-end"));
}

#[test]
fn emits_finish_step_at_turn_end_when_the_assistant_requested_no_tools() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({ "type": "message_start", "message": { "role": "assistant", "content": [] } }),
            update(json!({ "type": "text_delta", "delta": "done" })),
        ],
        &mut state,
    );
    let closing = translate(
        json!({ "type": "message_end", "message": {
            "role": "assistant", "content": [{ "type": "text", "text": "done" }],
        } }),
        &mut state,
    );
    assert_eq!(types(&closing), vec!["text-end"]);
    let turn_end = translate(json!({ "type": "turn_end" }), &mut state);
    assert_eq!(types(&turn_end), vec!["finish-step"]);
}

#[test]
fn does_not_emit_finish_step_when_turn_end_reports_a_terminal_error() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({ "type": "message_start", "message": { "role": "assistant", "content": [] } }),
            update(json!({ "type": "text_delta", "delta": "partial" })),
        ],
        &mut state,
    );
    let closing = translate(
        json!({ "type": "turn_end", "message": {
            "role": "assistant",
            "content": [{ "type": "text", "text": "partial" }],
            "stopReason": "error",
            "errorMessage": "Provider rejection",
        } }),
        &mut state,
    );
    assert_eq!(types(&closing), vec!["text-end"]);
}

#[test]
fn takes_the_turn_error_from_the_last_assistant_message_so_a_retried_request_recovers() {
    let mut state = state();
    let failed = json!({
        "role": "assistant",
        "content": [{ "type": "text", "text": "Coffee beans grow" }],
        "stopReason": "error",
        "errorMessage": "529 overloaded_error",
    });
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({ "type": "message_start", "message": failed }),
            json!({ "type": "message_end", "message": failed }),
            json!({ "type": "turn_end", "message": failed }),
            json!({ "type": "agent_end", "messages": [], "willRetry": true }),
        ],
        &mut state,
    );
    assert_eq!(
        state.last_assistant_error.as_deref(),
        Some("529 overloaded_error")
    );
    let answer = json!({
        "role": "assistant",
        "content": [{ "type": "text", "text": "Coffee beans come from cherries." }],
        "stopReason": "stop",
    });
    let parts = emit(
        vec![
            json!({ "type": "auto_retry_start", "attempt": 1 }),
            json!({ "type": "turn_start" }),
            json!({ "type": "message_start", "message": answer }),
            json!({ "type": "message_end", "message": answer }),
            json!({ "type": "turn_end", "message": answer }),
        ],
        &mut state,
    );
    assert_eq!(state.last_assistant_error, None);
    assert_eq!(types(&parts), vec!["finish-step"]);
}

#[test]
fn waits_for_requested_tool_executions_before_emitting_finish_step() {
    let mut state = state_with(&["bash"], &[]);
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({ "type": "message_start", "message": { "role": "assistant", "content": [] } }),
            json!({ "type": "message_end", "message": {
                "role": "assistant", "content": [{ "type": "toolCall", "id": "c-step", "name": "bash" }],
            } }),
        ],
        &mut state,
    );
    let start = translate(
        json!({
            "type": "tool_execution_start", "toolCallId": "c-step", "toolName": "bash",
            "args": { "command": "pwd" },
        }),
        &mut state,
    );
    let end = translate(
        json!({ "type": "tool_execution_end", "toolCallId": "c-step", "result": "ok" }),
        &mut state,
    );
    assert_eq!(types(&start), vec!["tool-call"]);
    assert_eq!(types(&end), vec!["tool-result", "finish-step"]);
}

#[test]
fn buffers_tool_input_of_later_calls_until_pi_starts_them() {
    let mut state = state_with(&[], &["deploy"]);
    let content = json!([
        { "type": "toolCall", "id": "call-1", "name": "deploy" },
        { "type": "toolCall", "id": "call-2", "name": "deploy" },
    ]);
    let mut events = vec![
        json!({ "type": "turn_start" }),
        json!({ "type": "message_start", "message": { "role": "assistant", "content": [] } }),
    ];
    for content_index in [0, 1] {
        events.push(update(json!({
            "type": "toolcall_start", "contentIndex": content_index, "partial": { "content": content },
        })));
        events.push(update(json!({
            "type": "toolcall_end", "contentIndex": content_index, "partial": { "content": content },
        })));
    }
    events.push(json!({ "type": "message_end", "message": { "role": "assistant", "content": content } }));
    let streamed = emit(events, &mut state);
    assert_eq!(
        streamed
            .iter()
            .map(|part| (part["type"].clone(), part["id"].clone()))
            .collect::<Vec<_>>(),
        vec![
            (json!("tool-input-start"), json!("call-1")),
            (json!("tool-input-end"), json!("call-1")),
        ]
    );
    let second = translate(
        json!({
            "type": "tool_execution_start", "toolCallId": "call-2", "toolName": "deploy",
            "args": { "target": "two" },
        }),
        &mut state,
    );
    assert_eq!(
        types(&second),
        vec!["tool-input-start", "tool-input-end", "tool-call"]
    );
    assert!(second[2].get("stepToolCallCount").is_none());
}

#[test]
fn reports_each_steps_usage_from_its_assistant_message() {
    let mut state = state_with(&["bash"], &[]);
    let tool_step = emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({ "type": "message_start", "message": { "role": "assistant", "content": [] } }),
            json!({ "type": "message_end", "message": {
                "role": "assistant",
                "content": [{ "type": "toolCall", "id": "c-usage", "name": "bash" }],
                "usage": { "input": 4, "output": 137, "cacheRead": 0, "cacheWrite": 13343, "totalTokens": 13484 },
            } }),
            json!({
                "type": "tool_execution_start", "toolCallId": "c-usage", "toolName": "bash",
                "args": { "command": "pwd" },
            }),
            json!({ "type": "tool_execution_end", "toolCallId": "c-usage", "result": "ok" }),
        ],
        &mut state,
    );
    let text_step = emit(
        vec![
            json!({ "type": "message_start", "message": { "role": "assistant", "content": [] } }),
            json!({ "type": "message_end", "message": {
                "role": "assistant",
                "content": [{ "type": "text", "text": "done" }],
                "usage": {
                    "input": 2, "output": 137, "cacheRead": 13403, "cacheWrite": 0,
                    "reasoning": 40, "totalTokens": 13542,
                },
            } }),
            json!({ "type": "turn_end" }),
        ],
        &mut state,
    );
    let finish = |parts: &[Value]| {
        parts
            .iter()
            .find(|part| part["type"] == "finish-step")
            .cloned()
            .unwrap()
    };
    assert_eq!(
        finish(&tool_step)["usage"],
        json!({
            "inputTokens": { "total": 13347, "noCache": 4, "cacheRead": 0, "cacheWrite": 13343 },
            "outputTokens": { "total": 137 },
        })
    );
    assert_eq!(
        finish(&text_step)["usage"],
        json!({
            "inputTokens": { "total": 13405, "noCache": 2, "cacheRead": 13403, "cacheWrite": 0 },
            "outputTokens": { "total": 137, "text": 97, "reasoning": 40 },
        })
    );
}

fn state_with_batch(ids: &[&str]) -> PiTranslatorState {
    let mut state = state_with(&[], &["weather"]);
    let content: Vec<Value> = ids
        .iter()
        .map(|id| json!({ "type": "toolCall", "id": id, "name": "weather" }))
        .collect();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({ "type": "message_start", "message": { "role": "assistant", "content": [] } }),
            json!({ "type": "message_end", "message": { "role": "assistant", "content": content } }),
        ],
        &mut state,
    );
    state
}

#[test]
fn early_finish_step_waits_until_every_call_of_a_parallel_batch_waits_on_host_input() {
    let mut state = state_with_batch(&["a", "b"]);
    assert!(take_pi_early_finish_step(&mut state, &set(&["a"]), false).is_empty());
    let parts = to_json(take_pi_early_finish_step(&mut state, &set(&["a", "b"]), false));
    assert_eq!(parts.len(), 1);
    assert_match_object(
        &parts[0],
        &json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "host-input" } }),
    );
}

#[test]
fn early_finish_step_does_not_repeat_for_calls_that_were_already_covered() {
    let mut state = state_with_batch(&["a", "b"]);
    assert_eq!(
        take_pi_early_finish_step(&mut state, &set(&["a", "b"]), false).len(),
        1
    );
    assert!(take_pi_early_finish_step(&mut state, &set(&["b"]), false).is_empty());
}

#[test]
fn early_finish_step_only_needs_the_first_unsettled_call_of_a_sequential_batch() {
    let mut state = state_with_batch(&["a", "b"]);
    assert!(take_pi_early_finish_step(&mut state, &set(&["b"]), true).is_empty());
    assert_eq!(
        take_pi_early_finish_step(&mut state, &set(&["a"]), true).len(),
        1
    );
    translate(
        json!({ "type": "tool_execution_end", "toolCallId": "a", "toolName": "weather", "result": "ok" }),
        &mut state,
    );
    assert_eq!(
        take_pi_early_finish_step(&mut state, &set(&["b"]), true).len(),
        1
    );
}

#[test]
fn emits_reasoning_start_lazily_on_first_thinking_delta() {
    let mut state = state();
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({ "type": "thinking_delta", "delta": "pondering..." })),
        ],
        &mut state,
    );
    assert_eq!(out[0]["type"], "reasoning-start");
    assert_match_object(
        &out[1],
        &json!({ "type": "reasoning-delta", "delta": "pondering..." }),
    );
}

#[test]
fn emits_reasoning_end_on_turn_end_if_reasoning_was_started() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({ "type": "thinking_delta", "delta": "think" })),
        ],
        &mut state,
    );
    let closing = translate(
        json!({ "type": "turn_end", "message": { "role": "assistant", "content": "" } }),
        &mut state,
    );
    assert!(types(&closing).contains(&"reasoning-end"));
}

#[test]
fn emits_tool_call_with_provider_executed_for_builtin_tools() {
    let mut state = state_with(&["bash"], &[]);
    emit(vec![json!({ "type": "turn_start" })], &mut state);
    let out = translate(
        json!({
            "type": "tool_execution_start", "toolCallId": "call_1", "toolName": "bash",
            "args": { "command": "ls" },
        }),
        &mut state,
    );
    assert_match_object(
        &out[0],
        &json!({
            "type": "tool-call", "toolCallId": "call_1", "toolName": "bash",
            "input": "{\"command\":\"ls\"}", "providerExecuted": true,
        }),
    );
}

#[test]
fn uses_common_name_as_wire_tool_name_when_native_to_common_contains_it() {
    let mut state = create_pi_translator_state(PiTranslatorStateOptions {
        builtin_tool_names: vec!["find".into()],
        host_tool_names: vec![],
        native_to_common: [("find".to_string(), "glob".to_string())].into_iter().collect(),
    });
    emit(vec![json!({ "type": "turn_start" })], &mut state);
    let out = translate(
        json!({
            "type": "tool_execution_start", "toolCallId": "c2", "toolName": "find",
            "args": { "pattern": "*.ts" },
        }),
        &mut state,
    );
    assert_match_object(
        &out[0],
        &json!({ "type": "tool-call", "toolName": "glob", "nativeName": "find", "providerExecuted": true }),
    );
}

#[test]
fn emits_tool_call_with_provider_executed_unset_for_user_registered_tools() {
    let mut state = state_with(&["bash"], &["deploy"]);
    emit(vec![json!({ "type": "turn_start" })], &mut state);
    let out = translate(
        json!({
            "type": "tool_execution_start", "toolCallId": "c3", "toolName": "deploy",
            "args": { "env": "staging" },
        }),
        &mut state,
    );
    assert!(out[0].get("providerExecuted").is_none());
}

#[test]
fn marks_mcp_prefixed_tool_calls_and_results_as_dynamic_and_parses_json_results() {
    let mut state = state();
    emit(vec![json!({ "type": "turn_start" })], &mut state);
    let call = translate(
        json!({
            "type": "tool_execution_start", "toolCallId": "mcp-call", "toolName": "mcp__memory_search",
            "args": { "query": "AI SDK" },
        }),
        &mut state,
    );
    let result = translate(
        json!({
            "type": "tool_execution_end", "toolCallId": "mcp-call",
            "result": { "content": [{ "type": "text", "text": "{\"matches\":[\"AI SDK Core\",\"AI SDK UI\"]}" }] },
        }),
        &mut state,
    );
    assert_eq!(
        [call[0].clone(), result[0].clone()],
        [
            json!({
                "dynamic": true,
                "input": "{\"query\":\"AI SDK\"}",
                "providerExecuted": true,
                "toolCallId": "mcp-call",
                "toolName": "mcp__memory_search",
                "type": "tool-call",
            }),
            json!({
                "dynamic": true,
                "result": { "matches": ["AI SDK Core", "AI SDK UI"] },
                "toolCallId": "mcp-call",
                "toolName": "mcp__memory_search",
                "type": "tool-result",
            }),
        ]
    );
}

#[test]
fn parses_a_json_mcp_result_that_carries_native_mcp_details() {
    let mut state = state();
    emit(vec![json!({ "type": "turn_start" })], &mut state);
    emit(
        vec![json!({
            "type": "tool_execution_start", "toolCallId": "mcp-call",
            "toolName": "mcp__brand_ai__brand_check", "args": {},
        })],
        &mut state,
    );
    let result = translate(
        json!({
            "type": "tool_execution_end", "toolCallId": "mcp-call",
            "result": {
                "content": [{ "type": "text", "text": "{\"ok\":true,\"issues\":[]}" }],
                "details": { "server": "brand_ai", "tool": "brand_check" },
            },
        }),
        &mut state,
    );
    assert_match_object(
        &result[0],
        &json!({
            "type": "tool-result", "toolName": "mcp__brand_ai__brand_check",
            "result": { "ok": true, "issues": [] },
        }),
    );
}

#[test]
fn keeps_non_json_mcp_results_and_json_native_results_as_text() {
    let mut state = state_with(&["read"], &[]);
    emit(vec![json!({ "type": "turn_start" })], &mut state);
    emit(
        vec![
            json!({ "type": "tool_execution_start", "toolCallId": "mcp-call", "toolName": "mcp__memory_search", "args": {} }),
            json!({ "type": "tool_execution_start", "toolCallId": "native-call", "toolName": "read", "args": {} }),
        ],
        &mut state,
    );
    let mcp_result = translate(
        json!({
            "type": "tool_execution_end", "toolCallId": "mcp-call",
            "result": { "content": [{ "type": "text", "text": "not JSON" }] },
        }),
        &mut state,
    );
    let native_result = translate(
        json!({
            "type": "tool_execution_end", "toolCallId": "native-call",
            "result": { "content": [{ "type": "text", "text": "{\"path\":\"README.md\"}" }] },
        }),
        &mut state,
    );
    assert_eq!(
        [mcp_result[0].clone(), native_result[0].clone()],
        [
            json!({
                "dynamic": true,
                "result": "not JSON",
                "toolCallId": "mcp-call",
                "toolName": "mcp__memory_search",
                "type": "tool-result",
            }),
            json!({
                "result": "{\"path\":\"README.md\"}",
                "toolCallId": "native-call",
                "toolName": "read",
                "type": "tool-result",
            }),
        ]
    );
}

#[test]
fn keeps_explicitly_typed_host_tools_static_even_with_an_mcp_prefix() {
    let mut state = state_with(&[], &["mcp__custom_tool"]);
    emit(vec![json!({ "type": "turn_start" })], &mut state);
    let out = translate(
        json!({ "type": "tool_execution_start", "toolCallId": "host-call", "toolName": "mcp__custom_tool", "args": {} }),
        &mut state,
    );
    assert_eq!(
        out[0],
        json!({
            "input": "{}",
            "toolCallId": "host-call",
            "toolName": "mcp__custom_tool",
            "type": "tool-call",
        })
    );
}

#[test]
fn marks_tools_registered_by_an_extension_as_dynamic_and_provider_executed() {
    let mut state = state_with(&["read"], &["deploy"]);
    let out = emit(
        vec![
            json!({ "type": "turn_start" }),
            update(json!({
                "type": "toolcall_start", "contentIndex": 0,
                "partial": { "content": [{ "type": "toolCall", "id": "task-call", "name": "Task" }] },
            })),
            json!({
                "type": "tool_execution_start", "toolCallId": "task-call", "toolName": "Task",
                "args": { "prompt": "Summarize the repo" },
            }),
            json!({
                "type": "tool_execution_end", "toolCallId": "task-call",
                "result": { "content": [{ "type": "text", "text": "{\"not\":\"mcp\"}" }] },
            }),
        ],
        &mut state,
    );
    assert_eq!(
        out,
        vec![
            json!({
                "dynamic": true,
                "id": "task-call",
                "providerExecuted": true,
                "toolName": "Task",
                "type": "tool-input-start",
            }),
            json!({
                "dynamic": true,
                "input": "{\"prompt\":\"Summarize the repo\"}",
                "providerExecuted": true,
                "toolCallId": "task-call",
                "toolName": "Task",
                "type": "tool-call",
            }),
            json!({
                "dynamic": true,
                "result": "{\"not\":\"mcp\"}",
                "toolCallId": "task-call",
                "toolName": "Task",
                "type": "tool-result",
            }),
        ]
    );
}

#[test]
fn correlates_tool_result_with_the_prior_tool_call_by_id() {
    let mut state = state_with(&["bash"], &[]);
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({
                "type": "tool_execution_start", "toolCallId": "c4", "toolName": "bash",
                "args": { "command": "echo" },
            }),
        ],
        &mut state,
    );
    let out = translate(
        json!({ "type": "tool_execution_end", "toolCallId": "c4", "result": "hello" }),
        &mut state,
    );
    assert_match_object(
        &out[0],
        &json!({ "type": "tool-result", "toolCallId": "c4", "toolName": "bash", "result": "hello" }),
    );
}

#[test]
fn surfaces_the_original_host_submitted_output_object_instead_of_the_echoed_text() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({
                "type": "tool_execution_start", "toolCallId": "c6", "toolName": "weather",
                "args": { "city": "SF" },
            }),
        ],
        &mut state,
    );
    let submitted = json!({ "state": "ready", "temperature": 72, "weather": "sunny" });
    state
        .host_tool_results
        .insert("c6".into(), Some(submitted.clone()));
    let out = translate(
        json!({
            "type": "tool_execution_end", "toolCallId": "c6",
            "result": { "content": [{ "type": "text", "text": submitted.to_string() }] },
        }),
        &mut state,
    );
    assert_match_object(
        &out[0],
        &json!({ "type": "tool-result", "toolCallId": "c6", "toolName": "weather", "result": submitted }),
    );
    assert!(!state.host_tool_results.contains_key("c6"));
}

#[test]
fn marks_tool_result_as_error_when_is_error_is_true() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({ "type": "tool_execution_start", "toolCallId": "c5", "toolName": "bash", "args": {} }),
        ],
        &mut state,
    );
    let out = translate(
        json!({ "type": "tool_execution_end", "toolCallId": "c5", "result": "boom", "isError": true }),
        &mut state,
    );
    assert_match_object(&out[0], &json!({ "type": "tool-result", "isError": true }));
}

#[test]
fn drops_unrecognised_event_types() {
    let mut state = state();
    emit(vec![json!({ "type": "turn_start" })], &mut state);
    assert!(translate(json!({ "type": "queue_update" }), &mut state).is_empty());
}

#[test]
fn translates_compaction_end_into_a_compaction_part() {
    let mut state = state();
    let out = translate(
        json!({
            "type": "compaction_end", "reason": "threshold", "aborted": false,
            "result": { "summary": "Condensed history.", "tokensBefore": 90000 },
        }),
        &mut state,
    );
    assert_eq!(
        out[0],
        json!({
            "type": "compaction",
            "trigger": "auto",
            "summary": "Condensed history.",
            "tokensBefore": 90000,
        })
    );
    assert_eq!(types(&out), vec!["compaction", "finish-step"]);
}

#[test]
fn closes_a_compaction_that_arrives_after_the_step_finished_in_its_own_step() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({ "type": "message_start", "message": { "role": "assistant", "content": [] } }),
            json!({ "type": "turn_end", "message": { "role": "assistant", "content": [] } }),
        ],
        &mut state,
    );
    let out = translate(
        json!({
            "type": "compaction_end", "reason": "threshold", "aborted": false,
            "result": { "summary": "Condensed history.", "tokensBefore": 90000 },
        }),
        &mut state,
    );
    assert_eq!(types(&out), vec!["compaction", "finish-step"]);
    assert_match_object(
        &out[1],
        &json!({ "harnessMetadata": { "pi": { "inferredStep": true } } }),
    );
    assert!(!state.step_open);
}

#[test]
fn leaves_a_compaction_inside_an_open_step_for_that_step_to_close() {
    let mut state = state();
    emit(
        vec![
            json!({ "type": "turn_start" }),
            json!({ "type": "message_start", "message": { "role": "assistant", "content": [] } }),
        ],
        &mut state,
    );
    let out = translate(
        json!({
            "type": "compaction_end", "reason": "threshold", "aborted": false,
            "result": { "summary": "Condensed history." },
        }),
        &mut state,
    );
    assert_eq!(types(&out), vec!["compaction"]);
    assert!(state.step_open);
}

#[test]
fn maps_reason_manual_to_trigger_manual() {
    let mut state = state();
    let out = translate(
        json!({
            "type": "compaction_end", "reason": "manual", "aborted": false,
            "result": { "summary": "Manual compaction." },
        }),
        &mut state,
    );
    assert_eq!(
        out[0],
        json!({ "type": "compaction", "trigger": "manual", "summary": "Manual compaction." })
    );
}

#[test]
fn drops_aborted_or_result_less_compaction_end_events() {
    let mut state = state();
    assert!(
        translate(
            json!({ "type": "compaction_end", "reason": "manual", "aborted": true, "result": { "summary": "s" } }),
            &mut state,
        )
        .is_empty()
    );
    assert!(
        translate(
            json!({ "type": "compaction_end", "reason": "overflow", "aborted": false }),
            &mut state,
        )
        .is_empty()
    );
}

#[test]
fn emits_with_a_placeholder_summary_when_the_result_has_no_summary() {
    let mut state = state();
    let out = translate(
        json!({
            "type": "compaction_end", "reason": "threshold", "aborted": false,
            "result": { "tokensBefore": 50000 },
        }),
        &mut state,
    );
    assert_eq!(
        out[0],
        json!({
            "type": "compaction",
            "trigger": "auto",
            "summary": "(no summary provided)",
            "tokensBefore": 50000,
        })
    );
    assert_eq!(types(&out), vec!["compaction", "finish-step"]);
}
