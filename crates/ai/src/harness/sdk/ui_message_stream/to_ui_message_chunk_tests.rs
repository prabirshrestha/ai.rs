//! Port of `ai` `ui-message-stream/to-ui-message-chunk.test.ts`.
//!
//! [rust] The file, reasoning-file, steering-message, source, custom and
//! tool-output-denied cases are not ported: the harness's
//! `TextStreamPart` has no such parts. 'uses onError when the tool error
//! cannot be serialized' and 'throws for unknown part types' have no
//! counterpart either (`getErrorMessage` cannot fail, and the part enum is
//! closed).

use std::sync::Arc;
use std::time::SystemTime;

use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::harness::provider::{JsonObject, JsonValue};
use crate::harness::sdk::content::{
    FinishReason, ProviderMetadata, ToolApprovalRequestOutput, ToolApprovalResponseOutput,
    TypedToolCall, TypedToolError, TypedToolResult,
};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::text_stream_part::{FinishStepResponse, TextStreamPart};
use crate::harness::sdk::tool::{Tool, ToolSet, ToolType, json_schema};

fn provider_metadata() -> ProviderMetadata {
    serde_json::from_value(json!({ "testProvider": { "signature": "sig-1" } })).unwrap()
}

fn tool_metadata() -> JsonObject {
    serde_json::from_value(json!({ "clientName": "test-client" })).unwrap()
}

fn test_tools() -> ToolSet {
    ToolSet::from([
        (
            "staticTool".to_string(),
            Tool::new(json_schema(json!({ "type": "object" }))),
        ),
        (
            "dynamicTool".to_string(),
            Tool::new(json_schema(json!({}))).with_type(ToolType::Dynamic),
        ),
    ])
}

fn chunk(part: TextStreamPart, options: &ToUIMessageChunkOptions<'_>) -> Option<JsonValue> {
    to_ui_message_chunk(&part, options).map(|c| c.to_json())
}

fn plain(part: TextStreamPart) -> Option<JsonValue> {
    chunk(part, &Default::default())
}

fn on_error(
    f: impl Fn(&ErrorValue) -> String + Send + Sync + 'static,
) -> Option<UIMessageStreamOnErrorFn> {
    Some(Arc::new(f))
}

/// `` error => `handled: ${error.message}` ``.
fn handled() -> Option<UIMessageStreamOnErrorFn> {
    on_error(|error| format!("handled: {}", error.as_error().unwrap()))
}

#[test]
fn maps_text_parts_and_preserves_provider_metadata() {
    let pm = json!({ "testProvider": { "signature": "sig-1" } });
    assert_eq!(
        plain(TextStreamPart::TextStart {
            id: "text-1".into(),
            provider_metadata: Some(provider_metadata()),
        }),
        Some(json!({ "type": "text-start", "id": "text-1", "providerMetadata": pm }))
    );
    assert_eq!(
        plain(TextStreamPart::TextDelta {
            id: "text-1".into(),
            text: "hello".into(),
            provider_metadata: Some(provider_metadata()),
        }),
        Some(
            json!({ "type": "text-delta", "id": "text-1", "delta": "hello", "providerMetadata": pm })
        )
    );
    assert_eq!(
        plain(TextStreamPart::TextEnd {
            id: "text-1".into(),
            citations: None,
            provider_metadata: Some(provider_metadata()),
        }),
        Some(json!({ "type": "text-end", "id": "text-1", "providerMetadata": pm }))
    );
}

#[test]
fn maps_reasoning_parts_by_default_and_suppresses_them_when_disabled() {
    let parts = || {
        vec![
            TextStreamPart::ReasoningStart {
                id: "reasoning-1".into(),
                provider_metadata: Some(provider_metadata()),
            },
            TextStreamPart::ReasoningDelta {
                id: "reasoning-1".into(),
                text: "thinking".into(),
                provider_metadata: Some(provider_metadata()),
            },
            TextStreamPart::ReasoningEnd {
                id: "reasoning-1".into(),
                provider_metadata: Some(provider_metadata()),
            },
        ]
    };
    let pm = json!({ "testProvider": { "signature": "sig-1" } });
    assert_eq!(
        parts().into_iter().map(plain).collect::<Vec<_>>(),
        vec![
            Some(json!({ "id": "reasoning-1", "providerMetadata": pm, "type": "reasoning-start" })),
            Some(
                json!({ "delta": "thinking", "id": "reasoning-1", "providerMetadata": pm, "type": "reasoning-delta" })
            ),
            Some(json!({ "id": "reasoning-1", "providerMetadata": pm, "type": "reasoning-end" })),
        ]
    );
    let disabled = ToUIMessageChunkOptions {
        send_reasoning: Some(false),
        ..Default::default()
    };
    assert_eq!(
        parts()
            .into_iter()
            .map(|p| chunk(p, &disabled))
            .collect::<Vec<_>>(),
        vec![None, None, None]
    );
}

fn usage() -> crate::harness::sdk::usage::LanguageModelUsage {
    serde_json::from_value(json!({
        "inputTokens": 1,
        "outputTokens": 2,
        "totalTokens": 3,
        "inputTokenDetails": {},
        "outputTokenDetails": {},
    }))
    .unwrap()
}

fn finish() -> TextStreamPart {
    TextStreamPart::Finish {
        finish_reason: FinishReason::Stop,
        raw_finish_reason: Some("stop".into()),
        total_usage: usage(),
        provider_metadata: None,
    }
}

#[test]
fn maps_lifecycle_parts() {
    let metadata = ToUIMessageChunkOptions {
        message_metadata: Some(json!({ "model": "test-model" })),
        response_message_id: Some("msg-1".into()),
        ..Default::default()
    };
    assert_eq!(
        chunk(TextStreamPart::Start, &metadata),
        Some(
            json!({ "type": "start", "messageId": "msg-1", "messageMetadata": { "model": "test-model" } })
        )
    );
    let no_start = ToUIMessageChunkOptions {
        send_start: Some(false),
        ..Default::default()
    };
    assert_eq!(chunk(TextStreamPart::Start, &no_start), None);

    let finish_metadata = ToUIMessageChunkOptions {
        message_metadata: Some(json!({ "model": "test-model" })),
        ..Default::default()
    };
    assert_eq!(
        chunk(finish(), &finish_metadata),
        Some(
            json!({ "type": "finish", "finishReason": "stop", "messageMetadata": { "model": "test-model" } })
        )
    );
    let no_finish = ToUIMessageChunkOptions {
        send_finish: Some(false),
        ..Default::default()
    };
    assert_eq!(chunk(finish(), &no_finish), None);

    assert_eq!(
        plain(TextStreamPart::StartStep {
            request: Default::default(),
            warnings: Vec::new(),
        }),
        Some(json!({ "type": "start-step" }))
    );
    assert_eq!(
        plain(TextStreamPart::FinishStep {
            response: FinishStepResponse {
                id: "response-id".into(),
                timestamp: SystemTime::UNIX_EPOCH,
                model_id: "model-id".into(),
                headers: None,
            },
            usage: usage(),
            performance: Default::default(),
            finish_reason: FinishReason::Stop,
            raw_finish_reason: Some("stop".into()),
            provider_metadata: Some(provider_metadata()),
        }),
        Some(json!({ "type": "finish-step" }))
    );
    assert_eq!(
        plain(TextStreamPart::Abort {
            reason: Some("user".into())
        }),
        Some(json!({ "type": "abort", "reason": "user" }))
    );
}

#[test]
fn maps_tool_input_streaming_parts() {
    let tools = test_tools();
    let with_tools = ToUIMessageChunkOptions {
        tools: Some(&tools),
        ..Default::default()
    };
    // [rust] The harness's tool-input-start part has no `inputEncoding` or
    // `toolMetadata`.
    assert_eq!(
        chunk(
            TextStreamPart::ToolInputStart {
                id: "call-1".into(),
                tool_name: "dynamicTool".into(),
                provider_metadata: Some(provider_metadata()),
                provider_executed: Some(true),
                dynamic: None,
                title: Some("Dynamic Tool".into()),
            },
            &with_tools
        ),
        Some(json!({
            "type": "tool-input-start",
            "toolCallId": "call-1",
            "toolName": "dynamicTool",
            "providerExecuted": true,
            "providerMetadata": { "testProvider": { "signature": "sig-1" } },
            "dynamic": true,
            "title": "Dynamic Tool",
        }))
    );
    assert_eq!(
        chunk(
            TextStreamPart::ToolInputStart {
                id: "call-2".into(),
                tool_name: "providerTool".into(),
                provider_metadata: None,
                provider_executed: None,
                dynamic: Some(true),
                title: None,
            },
            &with_tools
        ),
        Some(
            json!({ "type": "tool-input-start", "toolCallId": "call-2", "toolName": "providerTool", "dynamic": true })
        )
    );
    assert_eq!(
        plain(TextStreamPart::ToolInputDelta {
            id: "call-1".into(),
            delta: "{\"value\"".into(),
            provider_metadata: None,
        }),
        Some(
            json!({ "type": "tool-input-delta", "toolCallId": "call-1", "inputTextDelta": "{\"value\"" })
        )
    );
}

#[test]
fn maps_valid_and_invalid_tool_call_parts() {
    let tools = test_tools();
    let with_tools = ToUIMessageChunkOptions {
        tools: Some(&tools),
        ..Default::default()
    };
    let mut valid = TypedToolCall::new("call-1", "staticTool", json!({ "value": "input" }));
    valid.provider_executed = Some(true);
    valid.provider_metadata = Some(provider_metadata());
    valid.tool_metadata = Some(tool_metadata());
    valid.title = Some("Static Tool".into());
    assert_eq!(
        chunk(TextStreamPart::ToolCall(valid), &with_tools),
        Some(json!({
            "type": "tool-input-available",
            "toolCallId": "call-1",
            "toolName": "staticTool",
            "input": { "value": "input" },
            "providerExecuted": true,
            "providerMetadata": { "testProvider": { "signature": "sig-1" } },
            "toolMetadata": { "clientName": "test-client" },
            "title": "Static Tool",
        }))
    );

    let mut runtime = TypedToolCall::new("call-2", "runtimeTool", json!({ "value": "input" }));
    runtime.dynamic = Some(true);
    assert_eq!(
        plain(TextStreamPart::ToolCall(runtime)),
        Some(json!({
            "type": "tool-input-available",
            "toolCallId": "call-2",
            "toolName": "runtimeTool",
            "input": { "value": "input" },
            "dynamic": true,
        }))
    );

    let mut invalid = TypedToolCall::new("call-3", "runtimeTool", json!("{broken"));
    invalid.dynamic = Some(true);
    invalid.invalid = Some(true);
    invalid.error = Some(ErrorValue::message("invalid input"));
    invalid.provider_executed = Some(true);
    invalid.provider_metadata = Some(provider_metadata());
    invalid.tool_metadata = Some(tool_metadata());
    invalid.title = Some("Invalid Tool".into());
    assert_eq!(
        chunk(
            TextStreamPart::ToolCall(invalid),
            &ToUIMessageChunkOptions {
                on_error: handled(),
                ..Default::default()
            }
        ),
        Some(json!({
            "type": "tool-input-error",
            "toolCallId": "call-3",
            "toolName": "runtimeTool",
            "input": "{broken",
            "providerExecuted": true,
            "providerMetadata": { "testProvider": { "signature": "sig-1" } },
            "toolMetadata": { "clientName": "test-client" },
            "dynamic": true,
            "errorText": "handled: invalid input",
            "title": "Invalid Tool",
        }))
    );
}

#[test]
fn preserves_schema_input_for_transformed_tool_approval_requests() {
    let mut tool_call = TypedToolCall::new("call-1", "staticTool", json!({ "value": "trimmed" }));
    tool_call.input_schema_input =
        crate::harness::sdk::content::InputSchemaInput(Some(json!({ "value": " trimmed " })));
    assert_eq!(
        plain(TextStreamPart::ToolApprovalRequest(
            ToolApprovalRequestOutput {
                approval_id: "approval-1".into(),
                tool_call,
                reason: None,
                is_automatic: None,
                signature: None,
            }
        )),
        Some(json!({
            "type": "tool-approval-request",
            "approvalId": "approval-1",
            "toolCallId": "call-1",
            "inputSchemaInput": { "value": " trimmed " },
        }))
    );
}

#[test]
fn maps_tool_result_tool_error_and_approval_parts() {
    let tools = test_tools();
    let with_tools = ToUIMessageChunkOptions {
        tools: Some(&tools),
        ..Default::default()
    };
    let mut result = TypedToolResult::new(
        "call-1",
        "dynamicTool",
        Some(json!({ "value": "input" })),
        json!({ "value": "output" }),
    );
    result.provider_executed = Some(true);
    result.provider_metadata = Some(provider_metadata());
    result.tool_metadata = Some(tool_metadata());
    result.dynamic = Some(true);
    result.preliminary = Some(true);
    assert_eq!(
        chunk(TextStreamPart::ToolResult(result), &with_tools),
        Some(json!({
            "type": "tool-output-available",
            "toolCallId": "call-1",
            "output": { "value": "output" },
            "providerExecuted": true,
            "providerMetadata": { "testProvider": { "signature": "sig-1" } },
            "toolMetadata": { "clientName": "test-client" },
            "dynamic": true,
            "preliminary": true,
        }))
    );

    // The output must be `null` so the property survives JSON.
    let undefined_output = TypedToolResult::new(
        "call-undefined",
        "dynamicTool",
        Some(json!({ "value": "input" })),
        JsonValue::Null,
    );
    assert_eq!(
        chunk(TextStreamPart::ToolResult(undefined_output), &with_tools),
        Some(
            json!({ "type": "tool-output-available", "toolCallId": "call-undefined", "output": null, "dynamic": true })
        )
    );

    let mut provider_error = TypedToolError::new(
        "call-2",
        "dynamicTool",
        Some(json!({ "value": "input" })),
        ErrorValue::Value(json!({ "code": "provider-error" })),
    );
    provider_error.provider_executed = Some(true);
    provider_error.provider_metadata = Some(provider_metadata());
    provider_error.tool_metadata = Some(tool_metadata());
    provider_error.dynamic = Some(true);
    assert_eq!(
        chunk(
            TextStreamPart::ToolError(provider_error),
            &ToUIMessageChunkOptions {
                on_error: on_error(|_| "should not be used for provider-executed errors".into()),
                tools: Some(&tools),
                ..Default::default()
            }
        ),
        Some(json!({
            "type": "tool-output-error",
            "toolCallId": "call-2",
            "errorText": "{\"code\":\"provider-error\"}",
            "providerExecuted": true,
            "providerMetadata": { "testProvider": { "signature": "sig-1" } },
            "toolMetadata": { "clientName": "test-client" },
            "dynamic": true,
        }))
    );

    let mut string_error = TypedToolError::new(
        "call-string-error",
        "dynamicTool",
        Some(json!({ "value": "input" })),
        ErrorValue::Value(json!("provider string error")),
    );
    string_error.provider_executed = Some(true);
    string_error.dynamic = Some(true);
    assert_eq!(
        plain(TextStreamPart::ToolError(string_error)),
        Some(json!({
            "type": "tool-output-error",
            "toolCallId": "call-string-error",
            "errorText": "provider string error",
            "providerExecuted": true,
            "dynamic": true,
        }))
    );

    let tool_error = TypedToolError::new(
        "call-3",
        "staticTool",
        Some(json!({ "value": "input" })),
        ErrorValue::message("tool failed"),
    );
    assert_eq!(
        chunk(
            TextStreamPart::ToolError(tool_error),
            &ToUIMessageChunkOptions {
                on_error: handled(),
                tools: Some(&tools),
                ..Default::default()
            }
        ),
        Some(
            json!({ "type": "tool-output-error", "toolCallId": "call-3", "errorText": "handled: tool failed" })
        )
    );

    let call = TypedToolCall::new("call-5", "staticTool", json!({ "value": "input" }));
    assert_eq!(
        plain(TextStreamPart::ToolApprovalRequest(
            ToolApprovalRequestOutput {
                approval_id: "approval-1".into(),
                tool_call: call.clone(),
                reason: Some("requires operator review".into()),
                is_automatic: Some(true),
                signature: None,
            }
        )),
        Some(json!({
            "type": "tool-approval-request",
            "approvalId": "approval-1",
            "toolCallId": "call-5",
            "reason": "requires operator review",
            "isAutomatic": true,
        }))
    );
    assert_eq!(
        plain(TextStreamPart::ToolApprovalResponse(
            ToolApprovalResponseOutput {
                approval_id: "approval-1".into(),
                tool_call: call,
                approved: false,
                reason: Some("not allowed".into()),
                provider_executed: Some(true),
            }
        )),
        Some(json!({
            "type": "tool-approval-response",
            "approvalId": "approval-1",
            "approved": false,
            "reason": "not allowed",
            "providerExecuted": true,
        }))
    );
}

#[test]
fn maps_error_parts_through_on_error() {
    let calls: Arc<Mutex<Vec<ErrorValue>>> = Arc::default();
    let log = calls.clone();
    let error = ErrorValue::message("boom");
    assert_eq!(
        chunk(
            TextStreamPart::Error {
                error: error.clone()
            },
            &ToUIMessageChunkOptions {
                on_error: on_error(move |e| {
                    log.lock().push(e.clone());
                    "handled error".into()
                }),
                ..Default::default()
            }
        ),
        Some(json!({ "type": "error", "errorText": "handled error" }))
    );
    assert_eq!(*calls.lock(), vec![error]);
}

#[test]
fn masks_error_parts_by_default_and_when_redact_tool_errors_is_false() {
    let error = || TextStreamPart::Error {
        error: ErrorValue::message("boom"),
    };
    let masked = Some(json!({ "type": "error", "errorText": "An error occurred." }));
    assert_eq!(plain(error()), masked);
    assert_eq!(
        chunk(
            error(),
            &ToUIMessageChunkOptions {
                redact_tool_errors: Some(false),
                ..Default::default()
            }
        ),
        masked
    );
}

fn tool_error_part() -> TextStreamPart {
    let mut error = TypedToolError::new(
        "call-1",
        "runtimeTool",
        Some(json!({ "value": "input" })),
        ErrorValue::message("tool failed"),
    );
    error.dynamic = Some(true);
    TextStreamPart::ToolError(error)
}

fn invalid_tool_call() -> TypedToolCall {
    let mut call = TypedToolCall::new("call-1", "runtimeTool", json!("{broken"));
    call.dynamic = Some(true);
    call.invalid = Some(true);
    call.error = Some(ErrorValue::message("tool failed"));
    call
}

#[test]
fn error_text_of_failed_tool_calls() {
    for (name, on_error, redact, expected) in [
        (
            "keeps the text that the model received by default",
            None,
            None,
            "Error: tool failed",
        ),
        (
            "uses the default onError when tool errors are redacted",
            None,
            Some(true),
            "An error occurred.",
        ),
        (
            "uses a provided onError",
            handled(),
            None,
            "handled: tool failed",
        ),
        (
            "keeps the text that the model received when a provided onError does not redact tool errors",
            handled(),
            Some(false),
            "Error: tool failed",
        ),
    ] {
        let options = ToUIMessageChunkOptions {
            on_error,
            redact_tool_errors: redact,
            ..Default::default()
        };
        assert_eq!(
            chunk(tool_error_part(), &options),
            Some(
                json!({ "type": "tool-output-error", "toolCallId": "call-1", "dynamic": true, "errorText": expected })
            ),
            "{name}"
        );
        assert_eq!(
            chunk(TextStreamPart::ToolCall(invalid_tool_call()), &options),
            Some(json!({
                "type": "tool-input-error",
                "toolCallId": "call-1",
                "toolName": "runtimeTool",
                "input": "{broken",
                "dynamic": true,
                "errorText": expected,
            })),
            "{name}"
        );
    }
}

#[test]
fn uses_on_error_for_invalid_tool_calls_of_provider_executed_tools() {
    let mut call = invalid_tool_call();
    call.provider_executed = Some(true);
    assert_eq!(
        plain(TextStreamPart::ToolCall(call)),
        Some(json!({
            "type": "tool-input-error",
            "toolCallId": "call-1",
            "toolName": "runtimeTool",
            "input": "{broken",
            "providerExecuted": true,
            "dynamic": true,
            "errorText": "An error occurred.",
        }))
    );
}

#[test]
fn returns_none_for_parts_that_do_not_produce_ui_message_chunks() {
    assert_eq!(
        plain(TextStreamPart::ToolInputEnd {
            id: "call-1".into(),
            provider_metadata: None,
        }),
        None
    );
    assert_eq!(
        plain(TextStreamPart::Raw {
            raw_value: json!({ "provider": "raw" })
        }),
        None
    );
}
