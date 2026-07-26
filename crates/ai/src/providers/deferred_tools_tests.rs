use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::{anthropic, openai, openai_completions, openai_responses};
use crate::types::{
    AnthropicMessagesCompat, AssistantContent, AssistantMessage, Context, DeferredToolsMode,
    ImageContent, Message, Model, ModelCost, ModelInput, OpenAICompletionsCompat,
    OpenAIResponsesCompat, PayloadHook, SimpleStreamOptions, StopReason, StreamOptions, Tool,
    ToolCall, ToolResultContent, ToolResultMessage, Usage, UserMessage, UserMessageContent,
};

fn tool(name: &str) -> Tool {
    Tool {
        name: name.to_string(),
        description: format!("The {name} tool"),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"]
        }),
        constrained_sampling: None,
    }
}

fn user(timestamp: u64) -> Message {
    Message::User(UserMessage {
        content: UserMessageContent::Text("Hello".to_string()),
        timestamp,
    })
}

fn assistant_tool_call() -> Message {
    Message::Assistant(AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: "call_1".to_string(),
            name: "base_tool".to_string(),
            arguments: json!({}),
            thought_signature: None,
        })],
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        model: "claude-opus-4-6".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 2,
    })
}

fn tool_result(tool_call_id: &str, added_tool_names: &[&str]) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call_id.to_string(),
        tool_name: "base_tool".to_string(),
        content: vec![ToolResultContent::text("done")],
        details: None,
        added_tool_names: added_tool_names.iter().map(ToString::to_string).collect(),
        is_error: false,
        timestamp: 3,
    })
}

fn context(tools: Vec<Tool>) -> Context {
    Context {
        messages: vec![
            user(1),
            assistant_tool_call(),
            tool_result("call_1", &["late_tool"]),
            user(4),
        ],
        tools,
        ..Default::default()
    }
}

fn capture_options(captured: Arc<Mutex<Option<Value>>>) -> SimpleStreamOptions {
    capture_options_with_key(captured, "test-key")
}

fn capture_options_with_key(
    captured: Arc<Mutex<Option<Value>>>,
    api_key: &str,
) -> SimpleStreamOptions {
    let on_payload: PayloadHook = Arc::new(move |payload, _model| {
        let captured = Arc::clone(&captured);
        Box::pin(async move {
            *captured.lock().unwrap() = Some(payload);
            Err(crate::Error::Provider("payload captured".to_string()))
        })
    });
    SimpleStreamOptions {
        stream: StreamOptions {
            api_key: Some(api_key.to_string()),
            on_payload: Some(on_payload),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn anthropic_model(id: &str) -> Model {
    anthropic::builder()
        .api_key("test-key")
        .build()
        .expect("provider")
        .model(id)
        .build()
        .expect("model")
}

async fn capture_payload(
    stream: crate::Result<crate::AssistantEventStream>,
    captured: Arc<Mutex<Option<Value>>>,
) -> Value {
    let _ = crate::stream::final_message_from_stream(stream.expect("stream")).await;
    captured.lock().unwrap().take().expect("captured payload")
}

fn tool_names(payload: &Value) -> Vec<&str> {
    payload["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            tool.get("name")
                .or_else(|| tool.pointer("/function/name"))
                .and_then(Value::as_str)
        })
        .collect()
}

#[tokio::test]
async fn loads_an_anthropic_tool_at_its_tool_result_marker() {
    let captured = Arc::new(Mutex::new(None));
    let model = anthropic::builder()
        .api_key("test-key")
        .build()
        .expect("provider")
        .model("claude-opus-4-6")
        .build()
        .expect("model");
    let payload = capture_payload(
        anthropic::stream_simple_anthropic(
            model,
            context(vec![tool("base_tool"), tool("late_tool")]),
            capture_options(Arc::clone(&captured)),
        ),
        captured,
    )
    .await;

    assert_eq!(tool_names(&payload), ["base_tool", "late_tool"]);
    assert_eq!(payload["tools"][1]["defer_loading"], json!(true));
    let reference = payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .find(|block| block["type"] == "tool_result")
        .expect("tool result");
    assert_eq!(
        reference["content"],
        json!([{ "type": "tool_reference", "tool_name": "late_tool" }])
    );
}

#[tokio::test]
async fn anthropic_preserves_tool_output_as_sibling_content_after_references() {
    let captured = Arc::new(Mutex::new(None));
    let mut context = context(vec![tool("base_tool"), tool("late_tool")]);
    let Message::Assistant(assistant) = &mut context.messages[1] else {
        unreachable!();
    };
    assistant.content.push(AssistantContent::ToolCall(ToolCall {
        id: "call_2".to_string(),
        name: "base_tool".to_string(),
        arguments: json!({}),
        thought_signature: None,
    }));
    let Message::ToolResult(first_result) = &mut context.messages[2] else {
        unreachable!();
    };
    first_result.content = vec![
        ToolResultContent::text("work completed"),
        ToolResultContent::Image(ImageContent {
            data: "aW1hZ2U=".to_string(),
            mime_type: "image/png".to_string(),
        }),
    ];
    let mut second_result = match tool_result("call_2", &[]) {
        Message::ToolResult(result) => result,
        _ => unreachable!(),
    };
    second_result.content = vec![ToolResultContent::text("second result")];
    context
        .messages
        .insert(3, Message::ToolResult(second_result));

    let payload = capture_payload(
        anthropic::stream_simple_anthropic(
            anthropic_model("claude-opus-4-6"),
            context,
            capture_options(Arc::clone(&captured)),
        ),
        captured,
    )
    .await;
    let content = payload["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|message| {
            message["content"]
                .as_array()
                .filter(|blocks| blocks.iter().any(|block| block["type"] == "tool_result"))
        })
        .expect("tool result content");

    assert_eq!(
        content,
        &[
            json!({
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": [{ "type": "tool_reference", "tool_name": "late_tool" }],
                "is_error": false
            }),
            json!({
                "type": "tool_result",
                "tool_use_id": "call_2",
                "content": "second result",
                "is_error": false
            }),
            json!({ "type": "text", "text": "work completed" }),
            json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aW1hZ2U="
                }
            })
        ]
    );
}

#[tokio::test]
async fn anthropic_handles_cross_provider_history_missing_and_previously_used_tools() {
    for scenario in ["cross-provider", "missing", "used"] {
        let captured = Arc::new(Mutex::new(None));
        let mut context = context(if scenario == "missing" {
            vec![tool("base_tool")]
        } else {
            vec![tool("base_tool"), tool("late_tool")]
        });
        let Message::Assistant(assistant) = &mut context.messages[1] else {
            unreachable!();
        };
        if scenario == "cross-provider" {
            assistant.api = "openai-responses".to_string();
            assistant.provider = "openai".to_string();
            assistant.model = "gpt-5.4".to_string();
        } else if scenario == "used" {
            let AssistantContent::ToolCall(call) = &mut assistant.content[0] else {
                unreachable!();
            };
            call.name = "late_tool".to_string();
        }
        let payload = capture_payload(
            anthropic::stream_simple_anthropic(
                anthropic_model("claude-opus-4-8"),
                context,
                capture_options(Arc::clone(&captured)),
            ),
            captured,
        )
        .await;

        match scenario {
            "cross-provider" => {
                assert_eq!(payload["tools"][1]["defer_loading"], true);
                assert!(payload.to_string().contains("tool_reference"));
            }
            "missing" => {
                assert_eq!(tool_names(&payload), ["base_tool"]);
                assert!(!payload.to_string().contains("tool_reference"));
            }
            "used" => {
                assert_eq!(tool_names(&payload), ["base_tool", "late_tool"]);
                assert!(
                    payload["tools"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|tool| tool.get("defer_loading").is_none())
                );
                assert!(!payload.to_string().contains("tool_reference"));
            }
            _ => unreachable!(),
        }
    }
}

#[tokio::test]
async fn anthropic_oauth_normalizes_markers_usage_and_duplicate_definitions() {
    for scenario in ["used", "marker", "dedupe"] {
        let captured = Arc::new(Mutex::new(None));
        let mut context = if scenario == "dedupe" {
            let mut canonical = tool("Read");
            canonical.description = "Canonical definition".to_string();
            Context {
                messages: vec![user(1)],
                tools: vec![tool("read"), canonical],
                ..Default::default()
            }
        } else {
            let mut context = context(vec![tool("base_tool"), tool("read")]);
            let Message::ToolResult(result) = &mut context.messages[2] else {
                unreachable!();
            };
            result.added_tool_names = vec![if scenario == "marker" {
                "Read".to_string()
            } else {
                "read".to_string()
            }];
            if scenario == "used" {
                let Message::Assistant(assistant) = &mut context.messages[1] else {
                    unreachable!();
                };
                let AssistantContent::ToolCall(call) = &mut assistant.content[0] else {
                    unreachable!();
                };
                call.name = "Read".to_string();
            }
            context
        };
        let payload = capture_payload(
            anthropic::stream_simple_anthropic(
                anthropic_model("claude-opus-4-6"),
                std::mem::take(&mut context),
                capture_options_with_key(Arc::clone(&captured), "sk-ant-oat-fake"),
            ),
            captured,
        )
        .await;

        match scenario {
            "used" => {
                assert_eq!(tool_names(&payload), ["base_tool", "Read"]);
                assert!(
                    payload["tools"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|tool| tool.get("defer_loading").is_none())
                );
                assert!(!payload.to_string().contains("tool_reference"));
            }
            "marker" => {
                assert_eq!(tool_names(&payload), ["base_tool", "Read"]);
                assert_eq!(payload["tools"][1]["defer_loading"], true);
                assert!(payload.to_string().contains("\"tool_name\":\"Read\""));
            }
            "dedupe" => {
                assert_eq!(tool_names(&payload), ["Read"]);
                assert_eq!(payload["tools"][0]["description"], "Canonical definition");
            }
            _ => unreachable!(),
        }
    }
}

#[tokio::test]
async fn anthropic_uses_normal_or_immediate_tools_for_unsupported_and_all_deferred_cases() {
    for (model_id, tools, expected_names) in [
        (
            "claude-haiku-4-5",
            vec![tool("base_tool"), tool("late_tool")],
            vec!["base_tool", "late_tool"],
        ),
        (
            "claude-sonnet-4-20250514",
            vec![tool("base_tool"), tool("late_tool")],
            vec!["base_tool", "late_tool"],
        ),
        (
            "claude-opus-4-6",
            vec![tool("late_tool")],
            vec!["late_tool"],
        ),
    ] {
        let captured = Arc::new(Mutex::new(None));
        let payload = capture_payload(
            anthropic::stream_simple_anthropic(
                anthropic_model(model_id),
                context(tools),
                capture_options(Arc::clone(&captured)),
            ),
            captured,
        )
        .await;

        assert_eq!(tool_names(&payload), expected_names);
        assert!(
            payload["tools"]
                .as_array()
                .unwrap()
                .iter()
                .all(|tool| tool.get("defer_loading").is_none())
        );
        assert!(!payload.to_string().contains("tool_reference"));
    }
}

#[tokio::test]
async fn anthropic_supports_explicit_tool_reference_compatibility_override() {
    let captured = Arc::new(Mutex::new(None));
    let mut model = anthropic_model("vendor-claude");
    model.provider = "anthropic-proxy".to_string();
    model.compat.anthropic_messages = AnthropicMessagesCompat {
        supports_tool_references: Some(true),
        ..Default::default()
    };
    let payload = capture_payload(
        anthropic::stream_simple_anthropic(
            model,
            context(vec![tool("base_tool"), tool("late_tool")]),
            capture_options(Arc::clone(&captured)),
        ),
        captured,
    )
    .await;

    assert_eq!(payload["tools"][1]["defer_loading"], true);
}

#[tokio::test]
async fn loads_an_openai_responses_tool_through_client_tool_search() {
    let captured = Arc::new(Mutex::new(None));
    let model = openai::builder()
        .api_key(Some("test-key"))
        .build()
        .expect("provider")
        .model("gpt-5.4")
        .build()
        .expect("model");
    let payload = capture_payload(
        openai_responses::stream_simple_openai_responses(
            model,
            context(vec![tool("base_tool"), tool("late_tool")]),
            capture_options(Arc::clone(&captured)),
        ),
        captured,
    )
    .await;

    assert_eq!(tool_names(&payload), ["base_tool"]);
    let input = payload["input"].as_array().expect("input");
    let search_call = input
        .iter()
        .find(|item| item["type"] == "tool_search_call")
        .expect("search call");
    let search_output = input
        .iter()
        .find(|item| item["type"] == "tool_search_output")
        .expect("search output");
    assert_eq!(search_output["call_id"], search_call["call_id"]);
    assert_eq!(search_output["tools"][0]["name"], "late_tool");
    assert_eq!(search_output["tools"][0]["defer_loading"], true);
}

#[tokio::test]
async fn openai_responses_uses_normal_tools_when_search_is_unsupported_or_disabled() {
    for (model_id, override_support) in [
        ("gpt-5.2", None),
        ("gpt-5.4-nano", None),
        ("gpt-5.5-pro", None),
        ("gpt-5.4", Some(false)),
    ] {
        let captured = Arc::new(Mutex::new(None));
        let mut model = openai::builder()
            .api_key(Some("test-key"))
            .build()
            .expect("provider")
            .model(model_id)
            .build()
            .expect("model");
        if let Some(supports_tool_search) = override_support {
            model.provider = "openai-proxy".to_string();
            model.compat.openai_responses.supports_tool_search = Some(supports_tool_search);
        }
        let payload = capture_payload(
            openai_responses::stream_simple_openai_responses(
                model,
                context(vec![tool("base_tool"), tool("late_tool")]),
                capture_options(Arc::clone(&captured)),
            ),
            captured,
        )
        .await;

        assert_eq!(tool_names(&payload), ["base_tool", "late_tool"]);
        assert!(!payload.to_string().contains("tool_search_output"));
    }
}

#[tokio::test]
async fn serializes_kimi_deferred_tools_as_system_tool_definitions() {
    let captured = Arc::new(Mutex::new(None));
    let mut model = Model {
        id: "deferred-tools-model".to_string(),
        name: "Deferred Tools Model".to_string(),
        api: "openai-completions".to_string(),
        provider: "moonshotai".to_string(),
        base_url: "http://127.0.0.1:9/v1".to_string(),
        reasoning: false,
        input: vec![ModelInput::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4_096,
        ..Default::default()
    };
    model.compat.openai_completions = OpenAICompletionsCompat {
        deferred_tools_mode: Some(DeferredToolsMode::Kimi),
        ..Default::default()
    };
    let payload = capture_payload(
        openai_completions::stream_simple_openai_completions(
            model,
            context(vec![tool("base_tool"), tool("late_tool")]),
            capture_options(Arc::clone(&captured)),
        ),
        captured,
    )
    .await;

    assert_eq!(tool_names(&payload), ["base_tool"]);
    let messages = payload["messages"].as_array().expect("messages");
    let tool_result_index = messages
        .iter()
        .position(|message| message["role"] == "tool")
        .expect("tool result");
    let system_tool_index = messages
        .iter()
        .position(|message| message.get("tools").is_some())
        .expect("system tools");
    assert!(system_tool_index > tool_result_index);
    assert_eq!(
        messages[system_tool_index]["tools"][0]["function"]["name"],
        "late_tool"
    );
}

fn kimi_model(deferred: bool) -> Model {
    let mut model = Model {
        id: "deferred-tools-model".to_string(),
        name: "Deferred Tools Model".to_string(),
        api: "openai-completions".to_string(),
        provider: "moonshotai".to_string(),
        base_url: "http://127.0.0.1:9/v1".to_string(),
        reasoning: false,
        input: vec![ModelInput::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4_096,
        ..Default::default()
    };
    model.compat.openai_completions = OpenAICompletionsCompat {
        deferred_tools_mode: deferred.then_some(DeferredToolsMode::Kimi),
        ..Default::default()
    };
    model
}

#[tokio::test]
async fn kimi_emits_batched_deferred_schemas_in_marker_order() {
    let captured = Arc::new(Mutex::new(None));
    let mut context = context(vec![
        tool("base_tool"),
        tool("late_tool"),
        tool("later_tool"),
    ]);
    context
        .messages
        .insert(3, tool_result("call_2", &["later_tool"]));
    let payload = capture_payload(
        openai_completions::stream_simple_openai_completions(
            kimi_model(true),
            context,
            capture_options(Arc::clone(&captured)),
        ),
        captured,
    )
    .await;
    let messages = payload["messages"].as_array().expect("messages");

    assert_eq!(
        messages
            .iter()
            .filter_map(|message| message["role"].as_str())
            .collect::<Vec<_>>(),
        ["user", "assistant", "tool", "tool", "system", "user"]
    );
    assert_eq!(
        messages[4]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
            .collect::<Vec<_>>(),
        ["late_tool", "later_tool"]
    );
}

#[tokio::test]
async fn chat_completions_leaves_tools_unchanged_without_kimi_mode() {
    let captured = Arc::new(Mutex::new(None));
    let payload = capture_payload(
        openai_completions::stream_simple_openai_completions(
            kimi_model(false),
            context(vec![tool("base_tool"), tool("late_tool")]),
            capture_options(Arc::clone(&captured)),
        ),
        captured,
    )
    .await;

    assert_eq!(tool_names(&payload), ["base_tool", "late_tool"]);
    assert!(
        payload["messages"]
            .as_array()
            .unwrap()
            .iter()
            .all(|message| message.get("tools").is_none())
    );
}

#[test]
fn estimator_counts_definitions_marked_after_latest_usage_checkpoint() {
    let mut assistant = match assistant_tool_call() {
        Message::Assistant(assistant) => assistant,
        _ => unreachable!(),
    };
    assistant.content = vec![AssistantContent::Text(crate::TextContent {
        text: "done".to_string(),
        text_signature: None,
    })];
    assistant.usage.input = 100;
    assistant.usage.total_tokens = 100;
    assistant.stop_reason = StopReason::Stop;
    let plain = crate::utils::estimate::estimate_context_tokens(&Context {
        messages: vec![Message::Assistant(assistant.clone()), user(4)],
        ..Default::default()
    });
    let mut late_tool = tool("late_tool");
    late_tool.description = "x".repeat(4_000);
    let marked = crate::utils::estimate::estimate_context_tokens(&Context {
        messages: vec![
            Message::Assistant(assistant),
            tool_result("call_1", &["late_tool"]),
        ],
        tools: vec![late_tool],
        ..Default::default()
    });

    assert!(marked.tokens > plain.tokens + 500);
    assert!(marked.trailing_tokens > plain.trailing_tokens + 500);
}

#[test]
fn compat_metadata_round_trips() {
    let compat = OpenAIResponsesCompat {
        supports_tool_search: Some(true),
        ..Default::default()
    };
    let value = serde_json::to_value(&compat).expect("serialize compat");
    assert_eq!(value["supportsToolSearch"], true);
    assert_eq!(
        serde_json::from_value::<OpenAIResponsesCompat>(value).expect("deserialize compat"),
        compat
    );
}
