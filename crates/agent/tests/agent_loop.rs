use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use agent::{
    AfterToolCallResult, AgentContext, AgentEvent, AgentEventSink, AgentLoopConfig,
    AgentLoopTurnUpdate, AgentTool, AgentToolResult, BeforeToolCallResult, StreamFn,
    ToolExecutionMode, run_agent_loop,
};
use ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, Message, Model, ModelCost, SimpleStreamOptions, StopReason, TextContent, Tool,
    ToolCall, ToolResultContent, Usage,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::{Mutex, Notify};
use tokio_util::sync::CancellationToken;

fn create_model() -> Model {
    Model {
        id: "mock".to_string(),
        name: "mock".to_string(),
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        base_url: "https://example.invalid".to_string(),
        reasoning: false,
        input: vec![ai::ModelInput::Text],
        cost: ModelCost::default(),
        context_window: 8192,
        max_tokens: 2048,
        ..Model::default()
    }
}

fn assistant_message(content: Vec<AssistantContent>, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content,
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        model: "mock".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason,
        error_message: None,
        timestamp: ai::utils::time::now_millis(),
    }
}

fn assistant_text(text: &str) -> AssistantMessage {
    assistant_message(
        vec![AssistantContent::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        })],
        StopReason::Stop,
    )
}

fn assistant_tool_call(id: &str, name: &str, args: Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments: args,
        thought_signature: None,
    })
}

fn scripted_stream(messages: Vec<AssistantMessage>) -> StreamFn {
    let messages = Arc::new(Mutex::new(VecDeque::from(messages)));
    Arc::new(
        move |_model: Model, _context: Context, _options: SimpleStreamOptions| {
            let messages = messages.clone();
            Box::pin(async move {
                let message = messages
                    .lock()
                    .await
                    .pop_front()
                    .expect("scripted stream exhausted");
                let reason = message.stop_reason;
                let (mut sender, stream) = AssistantMessageEventStream::channel();
                sender.push(AssistantMessageEvent::Done { reason, message });
                Ok(stream)
            })
        },
    )
}

fn collecting_sink(events: Arc<Mutex<Vec<AgentEvent>>>) -> AgentEventSink {
    Arc::new(move |event| {
        let events = events.clone();
        Box::pin(async move {
            events.lock().await.push(event);
            Ok(())
        })
    })
}

fn event_type(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart => "agent_start",
        AgentEvent::AgentEnd { .. } => "agent_end",
        AgentEvent::TurnStart => "turn_start",
        AgentEvent::TurnEnd { .. } => "turn_end",
        AgentEvent::MessageStart { .. } => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd { .. } => "message_end",
        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
    }
}

fn message_role(message: &Message) -> &'static str {
    match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

fn tool_result_text(content: &[ToolResultContent]) -> Option<&str> {
    content.iter().find_map(|content| match content {
        ToolResultContent::Text(text) => Some(text.text.as_str()),
        ToolResultContent::Image(_) => None,
    })
}

#[derive(Clone)]
struct EchoTool {
    executed: Arc<Mutex<Vec<String>>>,
    first_started: Arc<Notify>,
    release_first: Arc<Notify>,
    first_finished: Arc<AtomicBool>,
    parallel_observed: Arc<AtomicBool>,
}

impl EchoTool {
    fn new(executed: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            executed,
            first_started: Arc::new(Notify::new()),
            release_first: Arc::new(Notify::new()),
            first_finished: Arc::new(AtomicBool::new(false)),
            parallel_observed: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[async_trait]
impl AgentTool for EchoTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "echo".to_string(),
            description: "Echo tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            }),
        }
    }

    fn label(&self) -> &str {
        "Echo"
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: Value,
        _cancellation_token: Option<CancellationToken>,
        _on_update: Option<agent::AgentToolUpdateCallback>,
    ) -> agent::Result<AgentToolResult> {
        let value = args
            .get("value")
            .and_then(Value::as_str)
            .expect("value argument")
            .to_string();

        if value == "first" {
            self.first_started.notify_one();
            self.release_first.notified().await;
            self.first_finished.store(true, Ordering::SeqCst);
        } else if value == "second" {
            self.first_started.notified().await;
            if !self.first_finished.load(Ordering::SeqCst) {
                self.parallel_observed.store(true, Ordering::SeqCst);
            }
            self.release_first.notify_one();
        }

        self.executed.lock().await.push(value.clone());
        Ok(AgentToolResult {
            content: vec![ToolResultContent::text(format!("echoed: {value}"))],
            details: Some(json!({ "value": value })),
            terminate: false,
        })
    }
}

#[tokio::test]
async fn emits_prompt_and_assistant_lifecycle_events() {
    let context = AgentContext {
        system_prompt: Some("You are helpful.".to_string()),
        messages: Vec::new(),
        tools: Vec::new(),
    };
    let config = AgentLoopConfig::new(create_model());
    let events = Arc::new(Mutex::new(Vec::new()));

    let messages = run_agent_loop(
        vec![Message::user_text("Hello")],
        context,
        config,
        collecting_sink(events.clone()),
        None,
        Some(scripted_stream(vec![assistant_text("Hi there!")])),
    )
    .await
    .unwrap();

    assert_eq!(
        messages.iter().map(message_role).collect::<Vec<_>>(),
        ["user", "assistant"]
    );
    assert_eq!(
        events
            .lock()
            .await
            .iter()
            .map(event_type)
            .collect::<Vec<_>>(),
        [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end"
        ]
    );
}

#[tokio::test]
async fn applies_transform_context_before_convert_to_llm() {
    let context = AgentContext {
        system_prompt: Some("You are helpful.".to_string()),
        messages: vec![
            Message::user_text("old message 1"),
            Message::Assistant(assistant_text("old response 1")),
            Message::user_text("old message 2"),
            Message::Assistant(assistant_text("old response 2")),
        ],
        tools: Vec::new(),
    };
    let transformed_seen = Arc::new(Mutex::new(Vec::new()));
    let converted_seen = Arc::new(Mutex::new(Vec::new()));
    let mut config = AgentLoopConfig::new(create_model());

    config.transform_context = Some(Arc::new({
        let transformed_seen = transformed_seen.clone();
        move |mut messages, _signal| {
            let transformed_seen = transformed_seen.clone();
            Box::pin(async move {
                let keep_from = messages.len().saturating_sub(2);
                let transformed = messages.split_off(keep_from);
                *transformed_seen.lock().await = transformed.clone();
                transformed
            })
        }
    }));
    config.convert_to_llm = Some(Arc::new({
        let converted_seen = converted_seen.clone();
        move |messages| {
            let converted_seen = converted_seen.clone();
            Box::pin(async move {
                *converted_seen.lock().await = messages.clone();
                messages
            })
        }
    }));

    run_agent_loop(
        vec![Message::user_text("new message")],
        context,
        config,
        collecting_sink(Arc::new(Mutex::new(Vec::new()))),
        None,
        Some(scripted_stream(vec![assistant_text("Response")])),
    )
    .await
    .unwrap();

    let transformed_roles = transformed_seen
        .lock()
        .await
        .iter()
        .map(message_role)
        .collect::<Vec<_>>();
    let converted_roles = converted_seen
        .lock()
        .await
        .iter()
        .map(message_role)
        .collect::<Vec<_>>();

    assert_eq!(transformed_roles, ["assistant", "user"]);
    assert_eq!(converted_roles, transformed_roles);
}

#[tokio::test]
async fn executes_tool_calls_and_emits_tool_result_messages() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = Arc::new(EchoTool::new(executed.clone()));
    let context = AgentContext {
        system_prompt: Some(String::new()),
        messages: Vec::new(),
        tools: vec![tool],
    };
    let config = AgentLoopConfig::new(create_model());
    let events = Arc::new(Mutex::new(Vec::new()));

    let messages = run_agent_loop(
        vec![Message::user_text("echo something")],
        context,
        config,
        collecting_sink(events.clone()),
        None,
        Some(scripted_stream(vec![
            assistant_message(
                vec![assistant_tool_call(
                    "tool-1",
                    "echo",
                    json!({ "value": "hello" }),
                )],
                StopReason::ToolUse,
            ),
            assistant_text("done"),
        ])),
    )
    .await
    .unwrap();

    assert_eq!(executed.lock().await.as_slice(), ["hello"]);
    assert_eq!(
        messages.iter().map(message_role).collect::<Vec<_>>(),
        ["user", "assistant", "toolResult", "assistant"]
    );

    let events = events.lock().await;
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            ..
        } if tool_call_id == "tool-1" && tool_name == "echo"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            is_error,
            ..
        } if tool_call_id == "tool-1" && !is_error
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::MessageEnd {
            message: Message::ToolResult(result),
        } if result.tool_call_id == "tool-1" && !result.is_error
    )));
}

#[tokio::test]
async fn parallel_tool_end_events_follow_completion_order_but_results_use_source_order() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = Arc::new(EchoTool::new(executed));
    let parallel_observed = tool.parallel_observed.clone();
    let context = AgentContext {
        system_prompt: Some(String::new()),
        messages: Vec::new(),
        tools: vec![tool],
    };
    let mut config = AgentLoopConfig::new(create_model());
    config.tool_execution = ToolExecutionMode::Parallel;
    let events = Arc::new(Mutex::new(Vec::new()));

    run_agent_loop(
        vec![Message::user_text("echo both")],
        context,
        config,
        collecting_sink(events.clone()),
        None,
        Some(scripted_stream(vec![
            assistant_message(
                vec![
                    assistant_tool_call("tool-1", "echo", json!({ "value": "first" })),
                    assistant_tool_call("tool-2", "echo", json!({ "value": "second" })),
                ],
                StopReason::ToolUse,
            ),
            assistant_text("done"),
        ])),
    )
    .await
    .unwrap();

    let events = events.lock().await;
    let tool_end_ids = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let tool_result_ids = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd {
                message: Message::ToolResult(result),
            } => Some(result.tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(parallel_observed.load(Ordering::SeqCst));
    assert_eq!(tool_end_ids, ["tool-2", "tool-1"]);
    assert_eq!(tool_result_ids, ["tool-1", "tool-2"]);
}

#[tokio::test]
async fn before_tool_call_can_override_arguments_without_revalidating() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = Arc::new(EchoTool::new(executed.clone()));
    let context = AgentContext {
        system_prompt: Some(String::new()),
        messages: Vec::new(),
        tools: vec![tool],
    };
    let mut config = AgentLoopConfig::new(create_model());
    config.before_tool_call = Some(Arc::new(|context, _signal| {
        Box::pin(async move {
            assert_eq!(context.tool_call.id, "tool-1");
            assert_eq!(context.args, json!({ "value": "original" }));
            Ok(Some(BeforeToolCallResult {
                args: Some(json!({ "value": "changed" })),
                ..BeforeToolCallResult::default()
            }))
        })
    }));

    run_agent_loop(
        vec![Message::user_text("echo something")],
        context,
        config,
        collecting_sink(Arc::new(Mutex::new(Vec::new()))),
        None,
        Some(scripted_stream(vec![
            assistant_message(
                vec![assistant_tool_call(
                    "tool-1",
                    "echo",
                    json!({ "value": "original" }),
                )],
                StopReason::ToolUse,
            ),
            assistant_text("done"),
        ])),
    )
    .await
    .unwrap();

    assert_eq!(executed.lock().await.as_slice(), ["changed"]);
}

#[tokio::test]
async fn before_tool_call_can_block_execution_with_error_tool_result() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = Arc::new(EchoTool::new(executed.clone()));
    let context = AgentContext {
        system_prompt: Some(String::new()),
        messages: Vec::new(),
        tools: vec![tool],
    };
    let mut config = AgentLoopConfig::new(create_model());
    config.before_tool_call = Some(Arc::new(|_context, _signal| {
        Box::pin(async move {
            Ok(Some(BeforeToolCallResult {
                block: true,
                reason: Some("blocked by policy".to_string()),
                args: None,
            }))
        })
    }));
    let events = Arc::new(Mutex::new(Vec::new()));

    let messages = run_agent_loop(
        vec![Message::user_text("echo something")],
        context,
        config,
        collecting_sink(events.clone()),
        None,
        Some(scripted_stream(vec![
            assistant_message(
                vec![assistant_tool_call(
                    "tool-1",
                    "echo",
                    json!({ "value": "hello" }),
                )],
                StopReason::ToolUse,
            ),
            assistant_text("done"),
        ])),
    )
    .await
    .unwrap();

    assert!(executed.lock().await.is_empty());
    assert_eq!(
        messages.iter().map(message_role).collect::<Vec<_>>(),
        ["user", "assistant", "toolResult", "assistant"]
    );
    let events = events.lock().await;
    let tool_result = events.iter().find_map(|event| match event {
        AgentEvent::MessageEnd {
            message: Message::ToolResult(result),
        } => Some(result),
        _ => None,
    });
    let tool_result = tool_result.expect("tool result event");
    assert!(tool_result.is_error);
    assert_eq!(
        tool_result_text(&tool_result.content),
        Some("blocked by policy")
    );
}

#[tokio::test]
async fn after_tool_call_can_override_result_and_terminate_batch() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = Arc::new(EchoTool::new(executed.clone()));
    let context = AgentContext {
        system_prompt: Some(String::new()),
        messages: Vec::new(),
        tools: vec![tool],
    };
    let mut config = AgentLoopConfig::new(create_model());
    config.after_tool_call = Some(Arc::new(|context, _signal| {
        Box::pin(async move {
            assert!(!context.is_error);
            assert_eq!(context.args, json!({ "value": "hello" }));
            Ok(Some(AfterToolCallResult {
                content: Some(vec![ToolResultContent::text("overridden")]),
                details: Some(json!({ "source": "after" })),
                is_error: Some(false),
                terminate: Some(true),
            }))
        })
    }));
    let events = Arc::new(Mutex::new(Vec::new()));

    let messages = run_agent_loop(
        vec![Message::user_text("echo something")],
        context,
        config,
        collecting_sink(events.clone()),
        None,
        Some(scripted_stream(vec![assistant_message(
            vec![assistant_tool_call(
                "tool-1",
                "echo",
                json!({ "value": "hello" }),
            )],
            StopReason::ToolUse,
        )])),
    )
    .await
    .unwrap();

    assert_eq!(executed.lock().await.as_slice(), ["hello"]);
    assert_eq!(
        messages.iter().map(message_role).collect::<Vec<_>>(),
        ["user", "assistant", "toolResult"]
    );
    let events = events.lock().await;
    let tool_result = events.iter().find_map(|event| match event {
        AgentEvent::MessageEnd {
            message: Message::ToolResult(result),
        } => Some(result),
        _ => None,
    });
    let tool_result = tool_result.expect("tool result event");
    assert_eq!(tool_result_text(&tool_result.content), Some("overridden"));
    assert_eq!(tool_result.details, Some(json!({ "source": "after" })));
}

#[tokio::test]
async fn prepare_next_turn_updates_context_before_continuing() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = Arc::new(EchoTool::new(executed));
    let context = AgentContext {
        system_prompt: Some("first prompt".to_string()),
        messages: Vec::new(),
        tools: vec![tool],
    };
    let prepared = Arc::new(AtomicBool::new(false));
    let mut config = AgentLoopConfig::new(create_model());
    config.prepare_next_turn = Some(Arc::new({
        let prepared = prepared.clone();
        move |context| {
            let prepared = prepared.clone();
            Box::pin(async move {
                if prepared.swap(true, Ordering::SeqCst) {
                    return None;
                }
                Some(AgentLoopTurnUpdate {
                    context: Some(AgentContext {
                        system_prompt: Some("second prompt".to_string()),
                        messages: context.context.messages,
                        tools: context.context.tools,
                    }),
                    model: None,
                    reasoning_level: None,
                })
            })
        }
    }));

    let calls = Arc::new(AtomicUsize::new(0));
    let second_system_prompt = Arc::new(Mutex::new(None));
    let stream_fn: StreamFn = Arc::new({
        let calls = calls.clone();
        let second_system_prompt = second_system_prompt.clone();
        move |_model, context, _options| {
            let calls = calls.clone();
            let second_system_prompt = second_system_prompt.clone();
            Box::pin(async move {
                let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                if call == 2 {
                    *second_system_prompt.lock().await = context.system_prompt.clone();
                }
                let message = if call == 1 {
                    assistant_message(
                        vec![assistant_tool_call(
                            "tool-1",
                            "echo",
                            json!({ "value": "hello" }),
                        )],
                        StopReason::ToolUse,
                    )
                } else {
                    assistant_text("done")
                };
                let reason = message.stop_reason;
                let (mut sender, stream) = AssistantMessageEventStream::channel();
                sender.push(AssistantMessageEvent::Done { reason, message });
                Ok(stream)
            })
        }
    });

    run_agent_loop(
        vec![Message::user_text("echo something")],
        context,
        config,
        collecting_sink(Arc::new(Mutex::new(Vec::new()))),
        None,
        Some(stream_fn),
    )
    .await
    .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        second_system_prompt.lock().await.as_deref(),
        Some("second prompt")
    );
}
