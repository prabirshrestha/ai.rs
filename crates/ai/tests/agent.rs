use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ai::{
    AfterToolCallContext, Agent, AgentContext, AgentError, AgentEvent, AgentEventSink,
    AgentLoopConfig, AgentLoopTurnUpdate, AgentOptions, AgentResult, AgentTool, AgentToolResult,
    BeforeToolCallContext, BeforeToolCallResult, FauxAssistantMessageOptions, FauxResponseStep,
    Message, StopReason, Tool, ToolExecutionMode, agent_loop, agent_loop_continue,
    faux_assistant_message, faux_text, faux_tool_call, register_faux_provider, run_agent_loop,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::sync::{Mutex, Notify};
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
            tools: vec![Arc::new(EchoTool {
                delay_first: true,
                ..Default::default()
            })],
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

#[tokio::test]
async fn prepare_next_turn_updates_context_before_tool_continuation() {
    let registration = register_faux_provider(None);
    let second_turn_system_prompt = Arc::new(Mutex::new(None));
    let captured_prompt = Arc::clone(&second_turn_system_prompt);
    registration.set_responses(vec![
        tool_use_response(vec![faux_tool_call(
            "echo",
            json!({ "value": "hello" }),
            Some("tool-1".to_string()),
        )])
        .into(),
        FauxResponseStep::factory(move |context, _options, _state, _model| {
            let captured_prompt = Arc::clone(&captured_prompt);
            async move {
                *captured_prompt.lock().await = context.system_prompt;
                Ok(faux_assistant_message("done", None))
            }
        }),
    ]);

    let prepared = Arc::new(AtomicBool::new(false));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.prepare_next_turn = Some(Arc::new({
        let prepared = Arc::clone(&prepared);
        move |context| {
            let prepared = Arc::clone(&prepared);
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

    let messages = run_agent_loop(
        vec![Message::user_text("echo something")],
        AgentContext {
            system_prompt: Some("first prompt".to_string()),
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

    assert_eq!(registration.state.call_count(), 2);
    assert_eq!(
        *second_turn_system_prompt.lock().await,
        Some("second prompt".to_string())
    );
    assert_eq!(
        message_roles(&messages),
        vec!["user", "assistant", "toolResult", "assistant"]
    );

    registration.unregister();
}

#[tokio::test]
async fn should_stop_after_turn_stops_before_follow_up_poll() {
    let registration = register_faux_provider(None);
    registration.set_responses([tool_use_response(vec![faux_tool_call(
        "echo",
        json!({ "value": "hello" }),
        Some("tool-1".to_string()),
    )])]);

    let executions = Arc::new(Mutex::new(Vec::new()));
    let steering_polls = Arc::new(AtomicUsize::new(0));
    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let callback_tool_result_ids = Arc::new(Mutex::new(Vec::new()));
    let callback_context_roles = Arc::new(Mutex::new(Vec::new()));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.get_steering_messages = Some(Arc::new({
        let steering_polls = Arc::clone(&steering_polls);
        move || {
            let steering_polls = Arc::clone(&steering_polls);
            Box::pin(async move {
                steering_polls.fetch_add(1, Ordering::SeqCst);
                Vec::new()
            })
        }
    }));
    config.get_follow_up_messages = Some(Arc::new({
        let follow_up_polls = Arc::clone(&follow_up_polls);
        move || {
            let follow_up_polls = Arc::clone(&follow_up_polls);
            Box::pin(async move {
                follow_up_polls.fetch_add(1, Ordering::SeqCst);
                vec![Message::user_text("follow up should stay queued")]
            })
        }
    }));
    config.should_stop_after_turn = Some(Arc::new({
        let callback_tool_result_ids = Arc::clone(&callback_tool_result_ids);
        let callback_context_roles = Arc::clone(&callback_context_roles);
        move |context| {
            let callback_tool_result_ids = Arc::clone(&callback_tool_result_ids);
            let callback_context_roles = Arc::clone(&callback_context_roles);
            Box::pin(async move {
                *callback_tool_result_ids.lock().await = context
                    .tool_results
                    .iter()
                    .map(|result| result.tool_call_id.clone())
                    .collect();
                *callback_context_roles.lock().await = message_roles(&context.context.messages);
                true
            })
        }
    }));
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
        vec![Message::user_text("echo something")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                executions: Some(Arc::clone(&executions)),
                ..Default::default()
            })],
        },
        config,
        emit,
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(*executions.lock().await, vec!["hello"]);
    assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
    assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
    assert_eq!(*callback_tool_result_ids.lock().await, vec!["tool-1"]);
    assert_eq!(
        *callback_context_roles.lock().await,
        vec!["user", "assistant", "toolResult"]
    );
    assert_eq!(
        message_roles(&messages),
        vec!["user", "assistant", "toolResult"]
    );
    assert_eq!(
        events
            .lock()
            .await
            .iter()
            .map(event_name)
            .filter(|name| *name != "message_update")
            .collect::<Vec<_>>(),
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "tool_execution_start",
            "tool_execution_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );

    registration.unregister();
}

#[tokio::test]
async fn terminating_tool_batch_stops_without_next_assistant_turn() {
    let registration = register_faux_provider(None);
    registration.set_responses([tool_use_response(vec![faux_tool_call(
        "echo",
        json!({ "value": "hello" }),
        Some("tool-1".to_string()),
    )])]);

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
        vec![Message::user_text("echo something")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                terminate_values: vec!["hello".to_string()],
                ..Default::default()
            })],
        },
        AgentLoopConfig::new(registration.get_model()),
        emit,
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(registration.state.call_count(), 1);
    assert_eq!(
        message_roles(&messages),
        vec!["user", "assistant", "toolResult"]
    );
    assert_eq!(
        events
            .lock()
            .await
            .iter()
            .filter(|event| matches!(event, AgentEvent::TurnEnd { .. }))
            .count(),
        1
    );

    registration.unregister();
}

#[tokio::test]
async fn parallel_tool_batch_continues_when_not_all_results_terminate() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        tool_use_response(vec![
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
        ]),
        faux_assistant_message("done", None),
    ]);

    let messages = run_agent_loop(
        vec![Message::user_text("echo both")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                terminate_values: vec!["first".to_string()],
                ..Default::default()
            })],
        },
        AgentLoopConfig::new(registration.get_model()),
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(registration.state.call_count(), 2);
    assert_eq!(
        message_roles(&messages),
        vec!["user", "assistant", "toolResult", "toolResult", "assistant"]
    );

    registration.unregister();
}

#[tokio::test]
async fn after_tool_call_can_mark_batch_terminating() {
    let registration = register_faux_provider(None);
    registration.set_responses([tool_use_response(vec![faux_tool_call(
        "echo",
        json!({ "value": "hello" }),
        Some("tool-1".to_string()),
    )])]);

    let mut config = AgentLoopConfig::new(registration.get_model());
    config.after_tool_call = Some(Arc::new(
        |_context: AfterToolCallContext, _token: Option<CancellationToken>| {
            Box::pin(async move {
                Ok(Some(ai::AfterToolCallResult {
                    terminate: Some(true),
                    ..Default::default()
                }))
            })
        },
    ));

    let messages = run_agent_loop(
        vec![Message::user_text("echo something")],
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

    assert_eq!(registration.state.call_count(), 1);
    assert_eq!(
        message_roles(&messages),
        vec!["user", "assistant", "toolResult"]
    );

    registration.unregister();
}

#[tokio::test]
async fn queued_steering_messages_wait_until_tool_batch_finishes() {
    let registration = register_faux_provider(None);
    let second_turn_roles = Arc::new(Mutex::new(Vec::new()));
    let captured_roles = Arc::clone(&second_turn_roles);
    registration.set_responses(vec![
        tool_use_response(vec![
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
        ])
        .into(),
        FauxResponseStep::factory(move |context, _options, _state, _model| {
            let captured_roles = Arc::clone(&captured_roles);
            async move {
                *captured_roles.lock().await = message_roles(&context.messages);
                Ok(faux_assistant_message("done", None))
            }
        }),
    ]);

    let steering_polls = Arc::new(AtomicUsize::new(0));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.get_steering_messages = Some(Arc::new({
        let steering_polls = Arc::clone(&steering_polls);
        move || {
            let steering_polls = Arc::clone(&steering_polls);
            Box::pin(async move {
                match steering_polls.fetch_add(1, Ordering::SeqCst) {
                    0 => Vec::new(),
                    1 => vec![Message::user_text("interrupt")],
                    _ => Vec::new(),
                }
            })
        }
    }));

    let messages = run_agent_loop(
        vec![Message::user_text("run tools")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                delay_first: true,
                ..Default::default()
            })],
        },
        config,
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(
        *second_turn_roles.lock().await,
        vec!["user", "assistant", "toolResult", "toolResult", "user"]
    );
    assert_eq!(
        message_roles(&messages),
        vec![
            "user",
            "assistant",
            "toolResult",
            "toolResult",
            "user",
            "assistant",
        ]
    );

    registration.unregister();
}

#[tokio::test]
async fn before_tool_call_args_override_executes_without_revalidation() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        tool_use_response(vec![faux_tool_call(
            "raw",
            json!({ "value": "valid" }),
            Some("tool-1".to_string()),
        )]),
        faux_assistant_message("done", None),
    ]);

    let received_args = Arc::new(Mutex::new(Vec::new()));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.before_tool_call = Some(Arc::new(
        |_context: BeforeToolCallContext, _token: Option<CancellationToken>| {
            Box::pin(async move {
                Ok(Some(BeforeToolCallResult {
                    args: Some(json!({ "value": 42 })),
                    ..Default::default()
                }))
            })
        },
    ));

    let messages = run_agent_loop(
        vec![Message::user_text("run raw")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(RecordingArgsTool {
                received_args: Arc::clone(&received_args),
            })],
        },
        config,
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(*received_args.lock().await, vec![json!({ "value": 42 })]);
    let Message::ToolResult(result) = &messages[2] else {
        panic!("expected tool result");
    };
    assert!(!result.is_error);

    registration.unregister();
}

#[tokio::test]
async fn sequential_tool_execution_mode_forces_batch_sequential() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        tool_use_response(vec![
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
        ]),
        faux_assistant_message("done", None),
    ]);

    let release_first = Arc::new(Notify::new());
    let first_done = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    tokio::spawn({
        let release_first = Arc::clone(&release_first);
        async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            release_first.notify_waiters();
        }
    });

    run_agent_loop(
        vec![Message::user_text("echo both")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                mode: Some(ToolExecutionMode::Sequential),
                first_done: Some(Arc::clone(&first_done)),
                parallel_observed: Some(Arc::clone(&parallel_observed)),
                release_first: Some(Arc::clone(&release_first)),
                ..Default::default()
            })],
        },
        AgentLoopConfig::new(registration.get_model()),
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    assert!(!parallel_observed.load(Ordering::SeqCst));

    registration.unregister();
}

#[tokio::test]
async fn parallel_tool_execution_mode_allows_concurrent_tools() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        tool_use_response(vec![
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
        ]),
        faux_assistant_message("done", None),
    ]);

    let release_first = Arc::new(Notify::new());
    let first_done = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    tokio::spawn({
        let release_first = Arc::clone(&release_first);
        async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            release_first.notify_waiters();
        }
    });

    run_agent_loop(
        vec![Message::user_text("echo both")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                mode: Some(ToolExecutionMode::Parallel),
                first_done: Some(Arc::clone(&first_done)),
                parallel_observed: Some(Arc::clone(&parallel_observed)),
                release_first: Some(Arc::clone(&release_first)),
                ..Default::default()
            })],
        },
        AgentLoopConfig::new(registration.get_model()),
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    assert!(parallel_observed.load(Ordering::SeqCst));

    registration.unregister();
}

fn tool_use_response(content: Vec<ai::AssistantContent>) -> ai::AssistantMessage {
    faux_assistant_message(
        content,
        Some(FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..Default::default()
        }),
    )
}

fn message_roles(messages: &[Message]) -> Vec<&'static str> {
    messages
        .iter()
        .map(|message| match message {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "toolResult",
        })
        .collect()
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
    terminate_values: Vec<String>,
    mode: Option<ToolExecutionMode>,
    executions: Option<Arc<Mutex<Vec<String>>>>,
    first_done: Option<Arc<AtomicBool>>,
    parallel_observed: Option<Arc<AtomicBool>>,
    release_first: Option<Arc<Notify>>,
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

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.mode
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
        if let Some(executions) = &self.executions {
            executions.lock().await.push(value.clone());
        }
        if value == "first" {
            if let Some(release_first) = &self.release_first {
                release_first.notified().await;
            }
            if let Some(first_done) = &self.first_done {
                first_done.store(true, Ordering::SeqCst);
            }
        }
        if value == "second"
            && self
                .first_done
                .as_ref()
                .is_some_and(|first_done| !first_done.load(Ordering::SeqCst))
        {
            if let Some(parallel_observed) = &self.parallel_observed {
                parallel_observed.store(true, Ordering::SeqCst);
            }
        }
        if self.delay_first && value == "first" {
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        }
        let mut result = AgentToolResult::text(format!("echoed: {value}"));
        result.terminate = self.terminate_values.iter().any(|item| item == &value);
        Ok(result)
    }
}

struct RecordingArgsTool {
    received_args: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl AgentTool for RecordingArgsTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "raw".to_string(),
            description: "Record raw args".to_string(),
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
        "Raw"
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: Value,
        _cancellation_token: Option<CancellationToken>,
        _on_update: Option<ai::AgentToolUpdateCallback>,
    ) -> AgentResult<AgentToolResult> {
        self.received_args.lock().await.push(args);
        Ok(AgentToolResult::text("recorded"))
    }
}
