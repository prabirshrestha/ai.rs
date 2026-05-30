use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ai::{
    Agent, AgentContext, AgentError, AgentEvent, AgentEventSink, AgentLoopConfig, AgentOptions,
    AgentResult, AgentTool, AgentToolResult, BeforeToolCallContext, BeforeToolCallResult,
    FauxAssistantMessageOptions, Message, StopReason, Tool, agent_loop, agent_loop_continue,
    faux_assistant_message, faux_text, faux_tool_call, register_faux_provider, run_agent_loop,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn agent_prompt_records_user_and_assistant_messages() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("hello", None)]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));

    agent.prompt_text("hi", Vec::new()).await.unwrap();

    let state = agent.state().await;
    assert_eq!(state.messages.len(), 2);
    assert!(matches!(state.messages[0], Message::User(_)));
    let Message::Assistant(assistant) = &state.messages[1] else {
        panic!("expected assistant message");
    };
    assert_eq!(assistant.content, vec![faux_text("hello")]);

    registration.unregister();
}

#[tokio::test]
async fn run_agent_loop_is_exported_from_ai_crate() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("done", None)]);
    let events = Arc::new(Mutex::new(Vec::new()));
    let emit: AgentEventSink = Arc::new({
        let events = Arc::clone(&events);
        move |event| {
            let events = Arc::clone(&events);
            Box::pin(async move {
                events.lock().await.push(event_name(&event));
                Ok(())
            })
        }
    });

    let messages = run_agent_loop(
        vec![Message::user_text("hi")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: Vec::new(),
        },
        AgentLoopConfig::new(registration.get_model()),
        emit,
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(messages.len(), 2);
    assert!(events.lock().await.contains(&"agent_end"));

    registration.unregister();
}

#[tokio::test]
async fn agent_loop_event_stream_is_exported_from_ai_crate() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("streamed", None)]);

    let mut stream = agent_loop(
        vec![Message::user_text("hi")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: Vec::new(),
        },
        AgentLoopConfig::new(registration.get_model()),
        None,
        None,
    );

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event_name(&event));
    }
    let messages = stream.result().await.unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(events.last(), Some(&"agent_end"));

    registration.unregister();
}

#[tokio::test]
async fn agent_loop_continue_rejects_invalid_contexts() {
    let registration = register_faux_provider(None);
    let config = AgentLoopConfig::new(registration.get_model());

    let empty = agent_loop_continue(
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: Vec::new(),
        },
        config.clone(),
        None,
        None,
    )
    .err();
    assert!(matches!(empty, Some(AgentError::NoMessagesToContinue)));

    let assistant_tail = agent_loop_continue(
        AgentContext {
            system_prompt: None,
            messages: vec![Message::Assistant(faux_assistant_message("done", None))],
            tools: Vec::new(),
        },
        config,
        None,
        None,
    )
    .err();
    assert!(matches!(
        assistant_tail,
        Some(AgentError::CannotContinueFromAssistant)
    ));

    registration.unregister();
}

#[tokio::test]
async fn parallel_tool_execution_prepares_tool_calls_sequentially() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "echo",
                    json!({ "value": "first" }),
                    Some("tool-1".to_string()),
                ),
                faux_tool_call(
                    "echo",
                    json!({ "value": "second" }),
                    Some("tool-2".to_string()),
                ),
            ],
            Some(FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            }),
        ),
        faux_assistant_message("done", None),
    ]);

    let active_before_hooks = Arc::new(AtomicUsize::new(0));
    let saw_concurrent_before_hook = Arc::new(AtomicBool::new(false));
    let before_active = Arc::clone(&active_before_hooks);
    let before_concurrent = Arc::clone(&saw_concurrent_before_hook);
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.before_tool_call = Some(Arc::new(
        move |_context: BeforeToolCallContext, _token: Option<CancellationToken>| {
            let before_active = Arc::clone(&before_active);
            let before_concurrent = Arc::clone(&before_concurrent);
            Box::pin(async move {
                if before_active.fetch_add(1, Ordering::SeqCst) > 0 {
                    before_concurrent.store(true, Ordering::SeqCst);
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                before_active.fetch_sub(1, Ordering::SeqCst);
                Ok::<Option<BeforeToolCallResult>, AgentError>(None)
            })
        },
    ));

    let messages = run_agent_loop(
        vec![Message::user_text("run tools")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool::default())],
        },
        config,
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    assert!(!saw_concurrent_before_hook.load(Ordering::SeqCst));
    assert_eq!(messages.len(), 5);

    registration.unregister();
}

#[tokio::test]
async fn parallel_tool_execution_end_events_follow_completion_order() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "echo",
                    json!({ "value": "first" }),
                    Some("tool-1".to_string()),
                ),
                faux_tool_call(
                    "echo",
                    json!({ "value": "second" }),
                    Some("tool-2".to_string()),
                ),
            ],
            Some(FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            }),
        ),
        faux_assistant_message("done", None),
    ]);
    let events = Arc::new(Mutex::new(Vec::new()));
    let emit: AgentEventSink = Arc::new({
        let events = Arc::clone(&events);
        move |event| {
            let events = Arc::clone(&events);
            Box::pin(async move {
                events.lock().await.push(event);
                Ok(())
            })
        }
    });

    let messages = run_agent_loop(
        vec![Message::user_text("run tools")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool { delay_first: true })],
        },
        AgentLoopConfig::new(registration.get_model()),
        emit,
        None,
        None,
    )
    .await
    .unwrap();

    let events = events.lock().await;
    let tool_execution_end_ids = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let tool_result_message_ids = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd {
                message: Message::ToolResult(tool_result),
            } => Some(tool_result.tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(tool_execution_end_ids, vec!["tool-2", "tool-1"]);
    assert_eq!(tool_result_message_ids, vec!["tool-1", "tool-2"]);
    assert!(matches!(messages[2], Message::ToolResult(_)));
    assert!(matches!(messages[3], Message::ToolResult(_)));

    registration.unregister();
}

fn event_name(event: &AgentEvent) -> &'static str {
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

fn quiet_sink() -> AgentEventSink {
    Arc::new(|_event| Box::pin(async { Ok(()) }))
}

#[derive(Default)]
struct EchoTool {
    delay_first: bool,
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
        _on_update: Option<ai::AgentToolUpdateCallback>,
    ) -> AgentResult<AgentToolResult> {
        let value = args
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if self.delay_first && value == "first" {
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        }
        Ok(AgentToolResult::text(format!("echoed: {value}")))
    }
}
