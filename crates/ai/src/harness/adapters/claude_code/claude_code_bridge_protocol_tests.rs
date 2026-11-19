//! Port of `harness-claude-code/src/claude-code-bridge-protocol.test.ts`.

use serde_json::{Value as JsonValue, json};

use super::*;

fn parse_outbound(value: JsonValue) -> Result<ClaudeCodeOutboundMessage, serde_json::Error> {
    serde_json::from_value(value)
}

fn parse_inbound(value: JsonValue) -> Result<ClaudeCodeInboundMessage, serde_json::Error> {
    serde_json::from_value(value)
}

fn outbound_cases() -> Vec<JsonValue> {
    let usage = json!({
        "inputTokens": { "total": 0, "noCache": 0 },
        "outputTokens": { "total": 0, "text": 0 },
    });
    vec![
        json!({ "type": "stream-start" }),
        json!({ "type": "text-start", "id": "x" }),
        json!({ "type": "text-delta", "id": "x", "delta": "hi" }),
        json!({ "type": "text-end", "id": "x" }),
        json!({ "type": "reasoning-start", "id": "r" }),
        json!({ "type": "reasoning-delta", "id": "r", "delta": "thinking" }),
        json!({ "type": "reasoning-end", "id": "r" }),
        json!({
            "type": "tool-input-start",
            "id": "t1",
            "toolName": "bash",
            "providerExecuted": true,
        }),
        json!({ "type": "tool-input-delta", "id": "t1", "delta": "{\"command\":\"ls\"}" }),
        json!({ "type": "tool-input-end", "id": "t1" }),
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
        json!({ "type": "error", "error": "boom" }),
        json!({ "type": "raw", "rawValue": { "hello": "world" } }),
        json!({
            "type": "raw",
            "rawValue": {
                "type": "stream_event",
                "event": { "type": "message_stop" },
                "usage": { "input_tokens": 1, "output_tokens": 2 },
            },
        }),
    ]
}

#[test]
fn outbound_accepts_every_sample() {
    for sample in outbound_cases() {
        assert!(parse_outbound(sample.clone()).is_ok(), "{sample}");
    }
}

#[test]
fn outbound_preserves_the_turn_identifier() {
    for sample in outbound_cases() {
        let mut with_start_id = sample.clone();
        with_start_id["startId"] = json!("turn-1");
        let parsed = parse_outbound(with_start_id.clone()).unwrap();
        assert_eq!(parsed.start_id.as_deref(), Some("turn-1"));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), with_start_id);
    }
}

#[test]
fn outbound_rejects_a_non_string_turn_identifier() {
    assert!(parse_outbound(json!({ "type": "stream-start", "startId": 7 })).is_err());
}

#[test]
fn outbound_rejects_unknown_types() {
    assert!(parse_outbound(json!({ "type": "mystery", "error": 1 })).is_err());
}

#[test]
fn outbound_preserves_structured_claude_code_tool_results() {
    let message = json!({
        "type": "tool-result",
        "toolCallId": "t1",
        "toolName": "TaskCreate",
        "result": { "task": { "id": "1", "subject": "probe-task" } },
    });
    assert_eq!(
        serde_json::to_value(parse_outbound(message.clone()).unwrap()).unwrap(),
        message
    );
}

fn start(parsed: ClaudeCodeInboundMessage) -> ClaudeCodeStartMessage {
    match parsed {
        ClaudeCodeInboundMessage::Start(start) => *start,
        other => panic!("expected a start message, got {other:?}"),
    }
}

#[test]
fn inbound_keeps_the_question_timeout_and_exclude_dynamic_sections() {
    let parsed = start(
        parse_inbound(json!({
            "type": "start",
            "prompt": "hi",
            "tools": [],
            "thinking": { "type": "adaptive", "display": "summarized" },
            "questionTimeoutMs": 600_000,
            "excludeDynamicSections": true,
        }))
        .unwrap(),
    );
    assert_eq!(parsed.question_timeout_ms, Some(600_000));
    assert_eq!(parsed.exclude_dynamic_sections, Some(true));
}

#[test]
fn inbound_keeps_additional_directories_strict_mcp_config_and_the_budget_limit() {
    let parsed = start(
        parse_inbound(json!({
            "type": "start",
            "prompt": "hi",
            "tools": [],
            "thinking": { "type": "adaptive", "display": "summarized" },
            "additionalDirectories": ["/vercel/sandbox/shared"],
            "strictMcpConfig": true,
            "maxBudgetUsd": 0.5,
        }))
        .unwrap(),
    );
    assert_eq!(
        parsed.additional_directories,
        Some(vec!["/vercel/sandbox/shared".to_string()])
    );
    assert_eq!(parsed.strict_mcp_config, Some(true));
    assert_eq!(parsed.max_budget_usd, Some(0.5));
}

#[test]
fn inbound_accepts_a_start_message() {
    assert!(
        parse_inbound(json!({
            "type": "start",
            "prompt": "hi",
            "instructions": "Be concise.",
            "tools": [{ "name": "deploy" }],
            "model": "claude-sonnet-4-5",
            "maxTurns": 5,
            "agentProgressSummaries": true,
            "forwardSubagentText": true,
            "env": { "DEPLOYMENT_ENV": "staging" },
            "thinking": { "type": "adaptive", "display": "summarized" },
            "skills": ["weather-forecast", "weather-codes"],
            "permissionMode": "allow-edits",
            "builtinToolFiltering": { "mode": "deny", "toolNames": ["bash"] },
        }))
        .is_ok()
    );
}

#[test]
fn inbound_accepts_a_start_message_with_text_and_image_prompt_content() {
    assert!(
        parse_inbound(json!({
            "type": "start",
            "prompt": [
                { "type": "text", "text": "What is in this image?" },
                { "type": "image", "mediaType": "image/png", "data": "iVBORw0KGgo=" },
            ],
            "thinking": { "type": "disabled" },
        }))
        .is_ok()
    );
}

#[test]
fn inbound_accepts_a_start_message_naming_the_exact_conversation_to_resume() {
    assert!(
        parse_inbound(json!({
            "type": "start",
            "prompt": "hi",
            "thinking": { "type": "disabled" },
            "resumeSessionId": "claude-session-1",
        }))
        .is_ok()
    );
}

#[test]
fn inbound_preserves_skill_overrides() {
    let message = json!({
        "type": "start",
        "prompt": "hi",
        "thinking": { "type": "disabled" },
        "skillOverrides": {
            "enabled-skill": "on",
            "name-only-skill": "name-only",
            "user-invocable-skill": "user-invocable-only",
            "disabled-skill": "off",
        },
    });
    assert_eq!(
        serde_json::to_value(parse_inbound(message.clone()).unwrap()).unwrap(),
        message
    );
}

#[test]
fn inbound_rejects_unsupported_skill_override_modes() {
    assert!(
        parse_inbound(json!({
            "type": "start",
            "prompt": "hi",
            "thinking": { "type": "disabled" },
            "skillOverrides": { "init": "disabled" },
        }))
        .is_err()
    );
}

#[test]
fn inbound_does_not_preserve_arbitrary_native_settings() {
    let parsed = parse_inbound(json!({
        "type": "start",
        "prompt": "hi",
        "thinking": { "type": "disabled" },
        "settings": { "permissions": { "defaultMode": "bypassPermissions" } },
    }))
    .unwrap();
    assert!(
        serde_json::to_value(parsed)
            .unwrap()
            .get("settings")
            .is_none()
    );
}

#[test]
fn inbound_accepts_auto_mode() {
    for auto_mode in [Some(true), Some(false), None] {
        let mut message =
            json!({ "type": "start", "prompt": "hi", "thinking": { "type": "disabled" } });
        if let Some(auto_mode) = auto_mode {
            message["autoMode"] = json!(auto_mode);
        }
        assert_eq!(start(parse_inbound(message).unwrap()).auto_mode, auto_mode);
    }
}

#[test]
fn inbound_rejects_invalid_start_fields() {
    for (key, value) in [
        ("autoMode", json!("auto")),
        ("resumeSessionId", json!(7)),
        ("env", json!({ "RETRY_COUNT": 3 })),
    ] {
        let mut message =
            json!({ "type": "start", "prompt": "hi", "thinking": { "type": "disabled" } });
        message[key] = value;
        assert!(parse_inbound(message).is_err(), "{key}");
    }
}

#[test]
fn inbound_rejects_legacy_string_thinking_values() {
    assert!(
        parse_inbound(json!({ "type": "start", "prompt": "hi", "thinking": "adaptive" })).is_err()
    );
}

#[test]
fn inbound_accepts_the_shared_commands() {
    for sample in [
        json!({ "type": "tool-result", "toolCallId": "t1", "output": { "ok": true } }),
        json!({ "type": "tool-approval-response", "approvalId": "a1", "approved": true }),
        json!({ "type": "user-message", "messageId": "message-1", "text": "hi" }),
        json!({ "type": "abort" }),
        json!({ "type": "stop" }),
        json!({ "type": "destroy" }),
    ] {
        assert!(
            matches!(
                parse_inbound(sample.clone()),
                Ok(ClaudeCodeInboundMessage::Command(_))
            ),
            "{sample}"
        );
    }
}

#[test]
fn bridge_ready_accepts_the_ready_handshake() {
    assert!(
        serde_json::from_value::<ClaudeCodeBridgeReady>(json!({
            "type": "bridge-ready",
            "port": 12345,
        }))
        .is_ok()
    );
    assert!(
        serde_json::from_value::<ClaudeCodeBridgeReady>(json!({ "type": "nope", "port": 12345 }))
            .is_err()
    );
}
