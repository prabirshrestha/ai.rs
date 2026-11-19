//! Port of `opencode-events.test.ts`, `create-emit-stream-event.test.ts`,
//! `opencode-usage.test.ts` and `opencode-finish-step.test.ts`.

use serde_json::{Value, json};

use super::*;

fn event(event_type: &str, properties: Value) -> OpenCodeEvent {
    OpenCodeEvent {
        id: None,
        r#type: Some(event_type.to_string()),
        properties: properties.as_object().cloned().unwrap_or_default(),
    }
}

fn translator(host_tools: &[&str], mcp_prefixes: &[&str]) -> Translator {
    Translator::new(ToolNaming {
        host_tools: host_tools.iter().map(|name| name.to_string()).collect(),
        mcp_prefixes: mcp_prefixes.iter().map(|name| name.to_string()).collect(),
        work_dir: "/work".into(),
    })
}

fn parts(outputs: Vec<Output>) -> Vec<Part> {
    outputs
        .into_iter()
        .filter_map(|output| match output {
            Output::Part(part) => Some(part),
            _ => None,
        })
        .collect()
}

fn run(translator: &mut Translator, events: Vec<OpenCodeEvent>) -> Vec<Output> {
    events
        .iter()
        .flat_map(|event| translator.translate(event))
        .collect()
}

// describe('OpenCode event helpers')

#[test]
fn unwraps_native_v2_events_with_data_payloads() {
    let event = unwrap_opencode_event(&json!({
        "id": "event-1",
        "type": "session.next.step.ended",
        "data": { "sessionID": "session-1", "finish": "stop" },
    }))
    .unwrap();
    assert_eq!(event.id.as_deref(), Some("event-1"));
    assert_eq!(event.r#type.as_deref(), Some("session.next.step.ended"));
    assert_eq!(
        Value::Object(event.properties.clone()),
        json!({ "sessionID": "session-1", "finish": "stop" })
    );
    assert_eq!(
        get_opencode_event_session_id(&event).as_deref(),
        Some("session-1")
    );
}

#[test]
fn unwraps_synchronized_events_and_strips_version_suffixes() {
    let event = unwrap_opencode_event(&json!({
        "type": "sync",
        "id": "outer-event",
        "syncEvent": {
            "id": "sync-event",
            "type": "session.next.text.ended.1",
            "data": { "sessionID": "session-1", "textID": "text-1", "text": "done" },
        },
    }))
    .unwrap();
    assert_eq!(event.id.as_deref(), Some("sync-event"));
    assert_eq!(event.r#type.as_deref(), Some("session.next.text.ended"));
    assert_eq!(event.properties["text"], "done");
}

#[test]
fn rejects_malformed_event_envelopes() {
    assert!(unwrap_opencode_event(&Value::Null).is_none());
    assert!(unwrap_opencode_event(&json!([])).is_none());
    assert!(
        unwrap_opencode_event(&json!({ "type": "sync", "syncEvent": "not-an-event" })).is_none()
    );
}

#[test]
fn finds_session_ids() {
    assert_eq!(
        get_opencode_event_session_id(&event(
            "message.part.updated",
            json!({ "part": { "type": "tool", "sessionID": "session-1" } })
        ))
        .as_deref(),
        Some("session-1")
    );
    for (event_type, properties) in [
        (
            "message.updated",
            json!({ "info": { "sessionID": "child-session" } }),
        ),
        (
            "session.created",
            json!({ "info": { "id": "child-session" } }),
        ),
        (
            "session.updated",
            json!({ "info": { "id": "child-session" } }),
        ),
        (
            "session.deleted",
            json!({ "info": { "id": "child-session" } }),
        ),
    ] {
        assert_eq!(
            get_opencode_event_session_id(&event(event_type, properties)).as_deref(),
            Some("child-session"),
            "{event_type}"
        );
    }
}

#[test]
fn emits_the_resolved_assistant_model_once() {
    let mut translator = translator(&[], &[]);
    let mut out = Vec::new();
    for info in [
        json!({ "role": "user", "providerID": "anthropic", "modelID": "claude-sonnet-4-6" }),
        json!({ "role": "assistant", "providerID": "anthropic", "modelID": "claude-sonnet-4-6" }),
        json!({ "role": "assistant", "providerID": "openai", "modelID": "gpt-5.3-codex" }),
    ] {
        out.extend(translator.stream_start(Some(&info)));
    }
    assert_eq!(
        parts(out),
        [Part::StreamStart {
            model_id: Some("anthropic/claude-sonnet-4-6".into())
        }]
    );

    let mut translator = super::super::events::Translator::new(ToolNaming::default());
    let mut out = translator.stream_start(Some(&json!({ "role": "assistant" })));
    out.extend(translator.stream_start(Some(
        &json!({ "role": "assistant", "providerID": "anthropic", "modelID": "x" }),
    )));
    assert_eq!(parts(out), [Part::StreamStart { model_id: None }]);
}

#[test]
fn emits_only_the_final_text_that_has_not_already_streamed() {
    let mut out = Vec::new();
    emit_missing_final_delta(&mut out, "text-1", Some("hello world"), "hello ", false);
    emit_missing_final_delta(
        &mut out,
        "text-1",
        Some("hello world"),
        "hello world",
        false,
    );
    emit_missing_final_delta(&mut out, "text-1", Some("rewritten"), "hello", false);
    assert_eq!(
        parts(out),
        [Part::TextDelta {
            id: "text-1".into(),
            delta: "world".into()
        }]
    );
}

// describe('legacy reasoning part translation')

#[test]
fn keeps_a_reasoning_part_streamed_under_field_text_as_reasoning() {
    let mut translator = translator(&[], &[]);
    let out = parts(run(
        &mut translator,
        vec![
            event(
                "message.part.updated",
                json!({ "part": { "id": "p1", "type": "reasoning", "text": "" } }),
            ),
            event(
                "message.part.delta",
                json!({ "partID": "p1", "field": "text", "delta": "thinking about the answer" }),
            ),
            event(
                "message.part.updated",
                json!({ "part": {
                    "id": "p1", "type": "reasoning",
                    "text": "thinking about the answer", "time": { "end": 1 },
                } }),
            ),
        ],
    ));
    assert!(
        !out.iter()
            .any(|part| matches!(part, Part::TextDelta { .. } | Part::TextStart { .. }))
    );
    let reasoning: String = out
        .iter()
        .filter_map(|part| match part {
            Part::ReasoningDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "thinking about the answer");
}

#[test]
fn still_emits_text_delta_for_a_plain_field_text_delta() {
    let mut translator = translator(&[], &[]);
    let out = parts(translator.translate(&event(
        "message.part.delta",
        json!({ "partID": "p2", "field": "text", "delta": "hello world" }),
    )));
    assert_eq!(
        out,
        [
            Part::TextStart { id: "p2".into() },
            Part::TextDelta {
                id: "p2".into(),
                delta: "hello world".into()
            },
        ]
    );
}

#[test]
fn does_not_treat_a_malformed_time_payload_as_a_completed_part() {
    let mut translator = translator(&[], &[]);
    let out = parts(translator.translate(&event(
        "message.part.updated",
        json!({ "part": { "id": "p3", "type": "text", "text": "hello", "time": "invalid" } }),
    )));
    assert_eq!(
        out,
        [
            Part::TextStart { id: "p3".into() },
            Part::TextDelta {
                id: "p3".into(),
                delta: "hello".into()
            },
        ]
    );
}

// describe('createEmitStreamEvent')

#[test]
fn keeps_mcp_arguments_separate_from_observation_metadata() {
    let mut translator = translator(&[], &["fixture_"]);
    let part = json!({
        "type": "tool",
        "callID": "native-call",
        "tool": "fixture_read",
        "metadata": { "providerItemId": "observe-1", "path": "wrong-path" },
        "state": {
            "status": "running",
            "metadata": { "billing": "observation", "exact": false },
            "input": { "path": "a.ts", "exact": true },
        },
    });
    let mut completed = part.clone();
    completed["state"]["status"] = json!("completed");
    completed["state"]["output"] = json!("native result");
    let out = parts(run(
        &mut translator,
        vec![
            event("message.part.updated", json!({ "part": part })),
            event("message.part.updated", json!({ "part": completed })),
        ],
    ));
    assert_eq!(
        out,
        [
            Part::ToolCall {
                tool_call_id: "native-call".into(),
                tool_name: "fixture_read".into(),
                native_name: None,
                input: json!({ "path": "a.ts", "exact": true }),
                dynamic: true,
            },
            Part::ToolResult {
                tool_call_id: "native-call".into(),
                tool_name: "fixture_read".into(),
                result: json!("native result"),
                is_error: false,
            },
        ]
    );
}

#[test]
fn reports_legacy_compaction_without_streaming_its_summary_as_answer_text() {
    let mut translator = translator(&[], &[]);
    let out = parts(run(
        &mut translator,
        vec![
            event(
                "message.part.updated",
                json!({ "part": { "type": "compaction", "auto": false } }),
            ),
            event(
                "message.updated",
                json!({ "info": { "id": "summary-1", "role": "assistant", "summary": true } }),
            ),
            event(
                "message.part.updated",
                json!({ "part": { "id": "t1", "type": "text", "messageID": "summary-1", "text": "Sum" } }),
            ),
            event(
                "message.part.delta",
                json!({ "messageID": "summary-1", "partID": "t1", "field": "text", "delta": "mary" }),
            ),
            event("session.compacted", json!({})),
        ],
    ));
    assert!(
        !out.iter()
            .any(|part| matches!(part, Part::TextDelta { .. }))
    );
    assert_eq!(
        out.last(),
        Some(&Part::Compaction {
            summary: "Summary".into()
        })
    );
}

#[test]
fn emits_text_final_deltas_and_step_usage() {
    let mut translator = translator(&[], &[]);
    let out = parts(run(
        &mut translator,
        vec![
            event("session.next.text.started", json!({ "textID": "text-1" })),
            event(
                "session.next.text.delta",
                json!({ "textID": "text-1", "delta": "hello " }),
            ),
            event(
                "session.next.text.ended",
                json!({ "textID": "text-1", "text": "hello world" }),
            ),
            event(
                "session.next.step.ended",
                json!({
                    "finish": "stop",
                    "tokens": { "input": 3, "output": 2, "reasoning": 0, "cache": { "read": 0, "write": 0 } },
                    "cost": 0.01,
                }),
            ),
        ],
    ));
    assert_eq!(
        out,
        [
            Part::TextStart {
                id: "text-1".into()
            },
            Part::TextDelta {
                id: "text-1".into(),
                delta: "hello ".into()
            },
            Part::TextDelta {
                id: "text-1".into(),
                delta: "world".into()
            },
            Part::TextEnd {
                id: "text-1".into()
            },
            Part::FinishStep {
                reason: FinishReason::Stop,
                raw: "stop".into(),
                usage: StepUsage {
                    input_total: 3.0,
                    input_no_cache: 3.0,
                    output_total: 2.0,
                    output_text: 2.0,
                    ..Default::default()
                },
                cost: Some(0.01),
            },
        ]
    );
}

#[test]
fn translates_provider_and_host_tool_events_without_duplication() {
    let mut translator = translator(&["weather"], &[]);
    let out = run(
        &mut translator,
        vec![
            event(
                "session.next.tool.called",
                json!({ "callID": "tool-1", "tool": "view", "input": { "file": "README.md" } }),
            ),
            event(
                "session.next.tool.success",
                json!({ "callID": "tool-1", "result": "contents" }),
            ),
            event(
                "session.next.tool.called",
                json!({ "callID": "tool-2", "tool": "weather", "input": { "city": "Berlin" } }),
            ),
            event(
                "session.next.tool.success",
                json!({ "callID": "tool-2", "result": "sunny" }),
            ),
        ],
    );
    assert_eq!(
        out,
        [
            Output::Part(Part::ToolCall {
                tool_call_id: "tool-1".into(),
                tool_name: "read".into(),
                native_name: Some("view".into()),
                input: json!({ "file": "README.md" }),
                dynamic: false,
            }),
            Output::Part(Part::ToolResult {
                tool_call_id: "tool-1".into(),
                tool_name: "read".into(),
                result: json!("contents"),
                is_error: false,
            }),
            Output::AuthorizeHostToolCall {
                call_id: "tool-2".into(),
                tool_name: "weather".into(),
                input: json!({ "city": "Berlin" }),
            },
        ]
    );
}

#[test]
fn resolves_prefixed_host_tool_names() {
    let mut translator = translator(&["weather"], &[]);
    let out = translator.translate(&event(
        "message.part.updated",
        json!({ "part": {
            "type": "tool", "callID": "c1", "tool": "harness-tools_weather",
            "state": { "status": "running", "input": {} },
        } }),
    ));
    assert_eq!(
        out,
        [Output::AuthorizeHostToolCall {
            call_id: "c1".into(),
            tool_name: "weather".into(),
            input: json!({}),
        }]
    );
}

#[test]
fn normalizes_legacy_tool_parts_before_translating_them() {
    let mut translator = translator(&[], &[]);
    let out = parts(translator.translate(&event(
        "message.part.updated",
        json!({ "part": {
            "type": "tool", "callID": "tool-legacy", "tool": "view", "metadata": ["ignored"],
            "state": { "status": "completed", "input": { "file": "README.md" }, "metadata": "ignored", "output": "contents" },
        } }),
    )));
    assert_eq!(
        out,
        [
            Part::ToolCall {
                tool_call_id: "tool-legacy".into(),
                tool_name: "read".into(),
                native_name: Some("view".into()),
                input: json!({ "file": "README.md" }),
                dynamic: false,
            },
            Part::ToolResult {
                tool_call_id: "tool-legacy".into(),
                tool_name: "read".into(),
                result: json!("contents"),
                is_error: false,
            },
        ]
    );
}

#[test]
fn ignores_tool_parts_that_opencode_has_compacted() {
    let mut translator = translator(&[], &[]);
    let part = json!({
        "type": "tool", "callID": "tool-earlier-turn", "tool": "view",
        "state": { "status": "completed", "input": { "file": "README.md" }, "output": "contents", "time": { "start": 1, "end": 2 } },
    });
    let mut compacted = part.clone();
    compacted["state"]["time"]["compacted"] = json!(3);
    assert!(
        translator
            .translate(&event("message.part.updated", json!({ "part": compacted })))
            .is_empty()
    );
    assert_eq!(
        translator
            .translate(&event("message.part.updated", json!({ "part": part })))
            .len(),
        2
    );
}

#[test]
fn does_not_expose_the_internal_structured_output_tool() {
    let mut translator = translator(&[], &[]);
    let out = run(
        &mut translator,
        vec![
            event(
                "session.next.tool.called",
                json!({ "callID": "structured-next", "tool": "StructuredOutput", "input": { "answer": "yes" } }),
            ),
            event(
                "session.next.tool.success",
                json!({ "callID": "structured-next", "tool": "StructuredOutput", "result": { "answer": "yes" } }),
            ),
            event(
                "message.part.updated",
                json!({ "part": {
                    "type": "tool", "callID": "structured-legacy", "tool": "StructuredOutput",
                    "state": { "status": "completed", "input": { "answer": "yes" }, "output": { "answer": "yes" } },
                } }),
            ),
        ],
    );
    assert!(out.is_empty());
}

#[test]
fn preserves_retry_error_compaction_and_file_events() {
    let mut translator = translator(&[], &[]);
    let out = parts(run(
        &mut translator,
        vec![
            event(
                "session.next.retried",
                json!({ "attempt": 2, "error": { "statusCode": 500, "message": "temporary" } }),
            ),
            event(
                "session.next.compaction.ended",
                json!({ "reason": "auto", "text": "summary", "recent": "recent" }),
            ),
            event("file.edited", json!({ "file": "/work/src/index.ts" })),
            event("session.next.step.failed", json!({ "error": "failed" })),
        ],
    ));
    assert_eq!(
        out,
        [
            Part::Warning("OpenCode session retry: attempt 2; HTTP 500; temporary".into()),
            Part::Compaction {
                summary: "summary".into()
            },
            Part::FileChange {
                path: "src/index.ts".into()
            },
            Part::Error("OpenCode step failed: failed".into()),
        ]
    );
}

#[test]
fn reports_subagent_sessions() {
    let mut translator = translator(&[], &[]);
    let out = translator.translate(&event(
        "message.part.updated",
        json!({ "part": {
            "type": "tool", "callID": "task-1", "tool": "task",
            "state": { "status": "running", "input": { "prompt": "go" },
                       "metadata": { "parentSessionId": "parent", "sessionId": "child" } },
        } }),
    ));
    assert_eq!(
        out[0],
        Output::SubagentSession {
            parent_session_id: "parent".into(),
            session_id: "child".into()
        }
    );
    assert!(matches!(
        &out[1],
        Output::Part(Part::ToolCall { tool_name, native_name: None, .. }) if tool_name == "agent"
    ));
}

// describe('OpenCode usage helpers')

#[test]
fn extracts_session_tokens_from_legacy_and_v2_response_shapes() {
    let tokens =
        json!({ "input": 10, "output": 5, "reasoning": 1, "cache": { "read": 2, "write": 3 } });
    let expected = OpenCodeTokens {
        input: 10.0,
        output: 5.0,
        reasoning: 1.0,
        cache_read: 2.0,
        cache_write: 3.0,
    };
    for shape in [
        json!({ "tokens": tokens }),
        json!({ "info": { "tokens": tokens } }),
        json!({ "data": { "tokens": tokens } }),
        json!({ "data": { "data": { "tokens": tokens } } }),
    ] {
        assert_eq!(extract_session_tokens(&shape), Some(expected));
    }
}

#[test]
fn ignores_malformed_token_envelopes() {
    assert_eq!(extract_session_tokens(&json!(null)), None);
    assert_eq!(
        extract_session_tokens(&json!({ "tokens": { "input": 1 } })),
        None
    );
    assert_eq!(
        extract_session_tokens(&json!({ "tokens": { "input": "1", "cache": {} } })),
        Some(OpenCodeTokens::default())
    );
}

#[test]
fn maps_the_turn_local_session_token_delta_to_harness_usage() {
    let before = OpenCodeTokens {
        input: 100.0,
        output: 20.0,
        reasoning: 5.0,
        cache_read: 40.0,
        cache_write: 10.0,
    };
    let after = OpenCodeTokens {
        input: 160.0,
        output: 30.0,
        reasoning: 7.0,
        cache_read: 70.0,
        cache_write: 10.0,
    };
    let usage = map_tokens(subtract_session_tokens(before, after));
    assert_eq!(
        usage,
        StepUsage {
            input_total: 60.0,
            input_no_cache: 30.0,
            cache_read: 30.0,
            cache_write: 0.0,
            output_total: 12.0,
            output_text: 10.0,
            output_reasoning: 2.0,
        }
    );
    let usage = usage.to_usage();
    assert_eq!((usage.input, usage.output, usage.cache_read), (30, 12, 30));
    assert_eq!((usage.reasoning, usage.total_tokens), (Some(2), 72));
}

// describe('OpenCode finish-step helpers')

#[test]
fn maps_legacy_step_finish_parts_and_ignores_others() {
    let mut translator = translator(&[], &[]);
    let finish = json!({
        "id": "finish-1", "type": "step-finish", "reason": "tool-calls", "cost": 0.5,
        "tokens": { "input": 1, "output": 2, "reasoning": 0, "cache": { "read": 0, "write": 0 } },
    });
    let out = parts(run(
        &mut translator,
        vec![
            event("message.part.updated", json!({ "part": finish })),
            event("message.part.updated", json!({ "part": finish })),
            event(
                "message.part.updated",
                json!({ "part": { "type": "step-start" } }),
            ),
        ],
    ));
    assert_eq!(out.len(), 1);
    assert!(matches!(
        &out[0],
        Part::FinishStep { reason: FinishReason::ToolCalls, raw, cost: Some(cost), .. }
            if raw == "tool-calls" && *cost == 0.5
    ));
}

#[test]
fn normalizes_opencode_finish_reasons() {
    for (raw, expected) in [
        ("max-length", FinishReason::Length),
        ("content_filter", FinishReason::ContentFilter),
        ("tool-calls", FinishReason::ToolCalls),
        ("error", FinishReason::Error),
        ("failed", FinishReason::Error),
        ("stop", FinishReason::Stop),
        ("end", FinishReason::Stop),
        ("unknown", FinishReason::Other),
    ] {
        assert_eq!(map_opencode_finish_reason(raw), expected, "{raw}");
    }
}
