//! Port of `harness-codex/src/codex-bridge-protocol.test.ts`.

use serde_json::{Value as JsonValue, json};

use super::*;

fn parse_outbound(value: JsonValue) -> Result<CodexOutboundMessage, serde_json::Error> {
    serde_json::from_value(value)
}

fn parse_inbound(value: JsonValue) -> Result<CodexInboundMessage, serde_json::Error> {
    serde_json::from_value(value)
}

fn parse_start(value: JsonValue) -> Result<CodexStartMessage, serde_json::Error> {
    serde_json::from_value(value)
}

// describe('outboundMessageSchema')

#[test]
fn outbound_accepts_every_sample() {
    let usage = json!({
        "inputTokens": { "total": 0, "noCache": 0 },
        "outputTokens": { "total": 0, "text": 0 },
    });
    for sample in [
        json!({ "type": "stream-start" }),
        json!({ "type": "text-start", "id": "x" }),
        json!({ "type": "text-delta", "id": "x", "delta": "hi" }),
        json!({ "type": "text-end", "id": "x" }),
        json!({ "type": "reasoning-start", "id": "r" }),
        json!({ "type": "reasoning-delta", "id": "r", "delta": "thinking" }),
        json!({ "type": "reasoning-end", "id": "r" }),
        json!({
            "type": "tool-call",
            "toolCallId": "t1",
            "toolName": "bash",
            "input": "{\"command\":\"ls\"}",
            "nativeName": "Bash",
            "providerExecuted": true,
        }),
        json!({
            "type": "tool-result",
            "toolCallId": "t1",
            "toolName": "bash",
            "result": { "exitCode": 0, "output": "ok" },
        }),
        json!({
            "type": "finish-step",
            "finishReason": { "unified": "stop", "raw": "stop" },
            "usage": usage,
        }),
        json!({
            "type": "finish",
            "finishReason": { "unified": "stop", "raw": "stop" },
            "totalUsage": usage,
        }),
        json!({ "type": "file-change", "event": "create", "path": "notes.md" }),
        json!({ "type": "file-change", "event": "modify", "path": "src/lib.ts" }),
        json!({ "type": "file-change", "event": "delete", "path": "old.txt" }),
        json!({ "type": "error", "error": "boom" }),
        json!({ "type": "raw", "rawValue": { "hello": "world" } }),
    ] {
        assert!(parse_outbound(sample.clone()).is_ok(), "{sample}");
    }
}

#[test]
fn outbound_rejects_unknown_types() {
    assert!(parse_outbound(json!({ "type": "mystery", "error": 1 })).is_err());
}

// describe('inboundMessageSchema')

#[test]
fn inbound_accepts_a_start_message() {
    assert!(matches!(
        parse_inbound(json!({
            "type": "start",
            "prompt": "hi",
            "instructions": "Be concise.",
            "restartThread": true,
            "tools": [{ "name": "deploy" }],
            "model": "gpt-5.1",
            "reasoningEffort": "high",
            "webSearch": true,
            "codexConfig": {
                "model_verbosity": "low",
                "features": { "multi_agent": false },
            },
        })),
        Ok(CodexInboundMessage::Start(_))
    ));
}

#[test]
fn inbound_accepts_a_start_message_with_text_and_image_prompt_content() {
    assert!(
        parse_start(json!({
            "type": "start",
            "prompt": [
                { "type": "text", "text": "What is in this image?" },
                { "type": "image", "mediaType": "image/png", "data": "iVBORw0KGgo=" },
            ],
        }))
        .is_ok()
    );
}

#[test]
fn inbound_accepts_a_tool_result_message() {
    assert!(
        parse_inbound(json!({
            "type": "tool-result",
            "toolCallId": "t1",
            "output": { "ok": true },
        }))
        .is_ok()
    );
}

#[test]
fn inbound_accepts_a_tool_approval_response_message() {
    assert!(
        parse_inbound(json!({
            "type": "tool-approval-response",
            "approvalId": "a1",
            "approved": false,
            "reason": "not allowed",
        }))
        .is_ok()
    );
}

#[test]
fn inbound_accepts_user_message_abort_stop_and_destroy() {
    for sample in [
        json!({ "type": "user-message", "messageId": "message-1", "text": "hi" }),
        json!({ "type": "abort" }),
        json!({ "type": "stop" }),
        json!({ "type": "destroy" }),
    ] {
        assert!(
            matches!(
                parse_inbound(sample.clone()),
                Ok(CodexInboundMessage::Command(_))
            ),
            "{sample}"
        );
    }
}

#[test]
fn inbound_accepts_xhigh_and_max_reasoning_effort_in_a_start_message() {
    for (effort, expected) in [
        ("xhigh", CodexReasoningEffort::Xhigh),
        ("max", CodexReasoningEffort::Max),
    ] {
        let parsed = parse_start(json!({
            "type": "start",
            "prompt": "Solve a difficult problem.",
            "reasoningEffort": effort,
        }))
        .unwrap();
        assert_eq!(parsed.reasoning_effort, Some(expected));
    }
}

// describe('bridgeReadySchema')

#[test]
fn bridge_ready_accepts_the_ready_handshake() {
    assert!(
        serde_json::from_value::<CodexBridgeReady>(json!({
            "type": "bridge-ready",
            "port": 12345,
        }))
        .is_ok()
    );
}

#[test]
fn bridge_ready_rejects_a_non_ready_type() {
    assert!(
        serde_json::from_value::<CodexBridgeReady>(json!({ "type": "nope", "port": 12345 }))
            .is_err()
    );
}
