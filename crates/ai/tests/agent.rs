use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ai::{
    AfterToolCallContext, Agent, AgentContext, AgentError, AgentEvent, AgentEventSink,
    AgentLoopConfig, AgentLoopTurnUpdate, AgentOptions, AgentResult, AgentState, AgentTool,
    AgentToolResult, BeforeToolCallContext, BeforeToolCallResult, FauxAssistantMessageOptions,
    FauxResponseStep, Message, QueueMode, StopReason, Tool, ToolExecutionMode, agent_loop,
    agent_loop_continue, faux_assistant_message, faux_text, faux_tool_call, register_faux_provider,
    run_agent_loop, run_agent_loop_continue,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::sync::{Mutex, Notify};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn agent_starts_with_default_state() {
    let registration = register_faux_provider(None);
    let model = registration.get_model();
    let agent = Agent::new(AgentOptions::new(model.clone()));

    let state = agent.state().await;
    assert_eq!(state.system_prompt, None);
    assert_eq!(state.model, model);
    assert_eq!(state.reasoning_level, None);
    assert!(state.tools.is_empty());
    assert!(state.messages.is_empty());
    assert!(!state.is_streaming);
    assert!(state.streaming_message.is_none());
    assert!(state.pending_tool_calls.is_empty());
    assert!(state.error_message.is_none());

    registration.unregister();
}

#[tokio::test]
async fn agent_starts_with_custom_initial_state() {
    let registration = register_faux_provider(None);
    let model = registration.get_model();
    let initial_user = Message::user_text("initial prompt");
    let mut options = AgentOptions::new(model.clone());
    options.initial_state = AgentState::new(model.clone());
    options.initial_state.system_prompt = Some("You are helpful.".to_string());
    options.initial_state.reasoning_level = Some(ai::ModelThinkingLevel::Low);
    options.initial_state.messages = vec![initial_user.clone()];
    let agent = Agent::new(options);

    let state = agent.state().await;
    assert_eq!(state.system_prompt.as_deref(), Some("You are helpful."));
    assert_eq!(state.model, model);
    assert_eq!(state.reasoning_level, Some(ai::ModelThinkingLevel::Low));
    assert_eq!(state.messages, vec![initial_user]);
    assert!(!state.is_streaming);

    registration.unregister();
}

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
async fn agent_prompt_emits_full_lifecycle_for_provider_failures() {
    let registration = register_faux_provider(None);
    let mut options = AgentOptions::new(registration.get_model());
    options.stream_fn = Some(Arc::new(|_model, _context, _options| {
        Box::pin(async move { Err(ai::Error::Validation("provider exploded".to_string())) })
    }));
    let agent = Agent::new(options);
    let events = Arc::new(Mutex::new(Vec::new()));

    agent
        .subscribe(Arc::new({
            let events = Arc::clone(&events);
            move |event, _token| {
                let events = Arc::clone(&events);
                Box::pin(async move {
                    events.lock().await.push(event_name(&event));
                    Ok(())
                })
            }
        }))
        .await;

    agent.prompt_text("hello", Vec::new()).await.unwrap();

    assert_eq!(
        *events.lock().await,
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
    let state = agent.state().await;
    assert!(!state.is_streaming);
    assert_eq!(state.error_message.as_deref(), Some("provider exploded"));
    assert_eq!(state.messages.len(), 2);
    assert!(matches!(state.messages[0], Message::User(_)));
    let Message::Assistant(assistant) = &state.messages[1] else {
        panic!("expected failure assistant message");
    };
    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert_eq!(
        assistant.error_message.as_deref(),
        Some("provider exploded")
    );

    registration.unregister();
}

#[tokio::test]
async fn agent_subscribe_can_unsubscribe_listener() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("hello", None)]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let event_count = Arc::new(AtomicUsize::new(0));

    let listener_id = agent
        .subscribe(Arc::new({
            let event_count = Arc::clone(&event_count);
            move |_event, _token| {
                let event_count = Arc::clone(&event_count);
                Box::pin(async move {
                    event_count.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            }
        }))
        .await;

    assert!(agent.unsubscribe(listener_id).await);
    assert!(!agent.unsubscribe(listener_id).await);

    agent.prompt_text("hi", Vec::new()).await.unwrap();
    assert_eq!(event_count.load(Ordering::SeqCst), 0);

    registration.unregister();
}

#[tokio::test]
async fn agent_wait_for_idle_waits_for_async_listeners() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("hello", None)]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let listener_entered = Arc::new(Notify::new());
    let release_listener = Arc::new(Notify::new());

    agent
        .subscribe(Arc::new({
            let listener_entered = Arc::clone(&listener_entered);
            let release_listener = Arc::clone(&release_listener);
            move |event, _token| {
                let listener_entered = Arc::clone(&listener_entered);
                let release_listener = Arc::clone(&release_listener);
                Box::pin(async move {
                    if matches!(event, AgentEvent::AgentEnd { .. }) {
                        listener_entered.notify_waiters();
                        release_listener.notified().await;
                    }
                    Ok(())
                })
            }
        }))
        .await;

    let prompt = agent.prompt_text("hi", Vec::new());
    tokio::pin!(prompt);
    tokio::select! {
        _ = listener_entered.notified() => {}
        result = &mut prompt => panic!("prompt completed before agent_end listener blocked: {result:?}"),
    }

    let idle = agent.wait_for_idle();
    tokio::pin!(idle);
    tokio::select! {
        _ = &mut idle => panic!("agent became idle before awaited listener settled"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
    }

    release_listener.notify_waiters();
    prompt.await.unwrap();
    idle.await;
    assert!(!agent.state().await.is_streaming);

    registration.unregister();
}

#[tokio::test]
async fn agent_exposes_active_cancellation_token() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("hello", None)]);
    let agent = Arc::new(Agent::new(AgentOptions::new(registration.get_model())));
    let saw_active_token = Arc::new(AtomicBool::new(false));

    agent
        .subscribe(Arc::new({
            let saw_active_token = Arc::clone(&saw_active_token);
            move |event, token| {
                let saw_active_token = Arc::clone(&saw_active_token);
                Box::pin(async move {
                    if matches!(event, AgentEvent::AgentStart) {
                        saw_active_token.store(!token.is_cancelled(), Ordering::SeqCst);
                    }
                    Ok(())
                })
            }
        }))
        .await;

    agent.prompt_text("hi", Vec::new()).await.unwrap();
    assert!(saw_active_token.load(Ordering::SeqCst));
    assert!(agent.cancellation_token().await.is_none());

    registration.unregister();
}

#[tokio::test]
async fn agent_abort_cancels_active_prompt() {
    let registration = register_faux_provider(Some(ai::RegisterFauxProviderOptions {
        tokens_per_second: Some(100.0),
        token_size: Some(ai::FauxTokenSize {
            min: Some(3),
            max: Some(3),
        }),
        ..Default::default()
    }));
    registration.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz", None)]);
    let agent = Arc::new(Agent::new(AgentOptions::new(registration.get_model())));
    let abort_requested = Arc::new(AtomicBool::new(false));
    let listener_token = Arc::new(Mutex::new(None::<CancellationToken>));

    agent
        .subscribe(Arc::new({
            let agent = Arc::clone(&agent);
            let abort_requested = Arc::clone(&abort_requested);
            let listener_token = Arc::clone(&listener_token);
            move |event, token| {
                let agent = Arc::clone(&agent);
                let abort_requested = Arc::clone(&abort_requested);
                let listener_token = Arc::clone(&listener_token);
                Box::pin(async move {
                    if matches!(event, AgentEvent::AgentStart) {
                        *listener_token.lock().await = Some(token.clone());
                    }
                    if matches!(
                        event,
                        AgentEvent::MessageUpdate {
                            assistant_message_event: ai::AssistantMessageEvent::TextDelta { .. },
                            ..
                        }
                    ) && !abort_requested.swap(true, Ordering::SeqCst)
                    {
                        agent.abort().await;
                    }
                    Ok(())
                })
            }
        }))
        .await;

    agent.prompt_text("hi", Vec::new()).await.unwrap();

    let state = agent.state().await;
    assert!(!state.is_streaming);
    assert!(agent.cancellation_token().await.is_none());
    assert!(
        listener_token
            .lock()
            .await
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    );
    assert_eq!(state.error_message.as_deref(), Some("Request was aborted"));
    let Message::Assistant(assistant) = state.messages.last().expect("assistant message") else {
        panic!("expected assistant message");
    };
    assert_eq!(assistant.stop_reason, StopReason::Aborted);
    assert_eq!(
        assistant.error_message.as_deref(),
        Some("Request was aborted")
    );

    registration.unregister();
}

#[tokio::test]
async fn agent_continue_run_rejects_while_processing_before_context_validation() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("hello", None)]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let listener_entered = Arc::new(Notify::new());
    let release_listener = Arc::new(Notify::new());

    agent
        .subscribe(Arc::new({
            let listener_entered = Arc::clone(&listener_entered);
            let release_listener = Arc::clone(&release_listener);
            move |event, _token| {
                let listener_entered = Arc::clone(&listener_entered);
                let release_listener = Arc::clone(&release_listener);
                Box::pin(async move {
                    if matches!(event, AgentEvent::AgentStart) {
                        listener_entered.notify_waiters();
                        release_listener.notified().await;
                    }
                    Ok(())
                })
            }
        }))
        .await;

    let prompt = agent.prompt_text("hi", Vec::new());
    tokio::pin!(prompt);
    tokio::select! {
        _ = listener_entered.notified() => {}
        result = &mut prompt => panic!("prompt completed before agent_start listener blocked: {result:?}"),
    }

    let error = agent.continue_run().await.unwrap_err();
    assert!(matches!(error, AgentError::AlreadyProcessing));

    release_listener.notify_waiters();
    prompt.await.unwrap();

    registration.unregister();
}

#[tokio::test]
async fn agent_prompt_rejects_while_processing() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("hello", None)]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));
    let listener_entered = Arc::new(Notify::new());
    let release_listener = Arc::new(Notify::new());

    agent
        .subscribe(Arc::new({
            let listener_entered = Arc::clone(&listener_entered);
            let release_listener = Arc::clone(&release_listener);
            move |event, _token| {
                let listener_entered = Arc::clone(&listener_entered);
                let release_listener = Arc::clone(&release_listener);
                Box::pin(async move {
                    if matches!(event, AgentEvent::AgentStart) {
                        listener_entered.notify_waiters();
                        release_listener.notified().await;
                    }
                    Ok(())
                })
            }
        }))
        .await;

    let first_prompt = agent.prompt_text("first", Vec::new());
    tokio::pin!(first_prompt);
    tokio::select! {
        _ = listener_entered.notified() => {}
        result = &mut first_prompt => panic!("prompt completed before agent_start listener blocked: {result:?}"),
    }

    let error = agent.prompt_text("second", Vec::new()).await.unwrap_err();
    assert!(matches!(error, AgentError::AlreadyProcessing));

    release_listener.notify_waiters();
    first_prompt.await.unwrap();

    registration.unregister();
}

#[tokio::test]
async fn agent_state_mutators_update_runtime_configuration() {
    let registration = register_faux_provider(None);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));

    agent
        .set_system_prompt(Some("Custom prompt".to_string()))
        .await;
    assert_eq!(
        agent.state().await.system_prompt.as_deref(),
        Some("Custom prompt")
    );

    let mut next_model = registration.get_model();
    next_model.id = "next-model".to_string();
    agent.set_model(next_model).await;
    assert_eq!(agent.state().await.model.id, "next-model");

    agent
        .set_reasoning_level(Some(ai::ModelThinkingLevel::High))
        .await;
    assert_eq!(
        agent.state().await.reasoning_level,
        Some(ai::ModelThinkingLevel::High)
    );

    let tools = vec![Arc::new(EchoTool::default()) as Arc<dyn AgentTool>];
    agent.set_tools(tools.clone()).await;
    assert_eq!(agent.state().await.tools.len(), 1);
    drop(tools);
    assert_eq!(agent.state().await.tools.len(), 1);
    agent.clear_tools().await;
    assert!(agent.state().await.tools.is_empty());

    let first = Message::user_text("hello");
    agent.set_messages(vec![first.clone()]).await;
    assert_eq!(agent.state().await.messages, vec![first]);

    let second = Message::user_text("again");
    agent.push_message(second.clone()).await;
    assert_eq!(agent.state().await.messages.len(), 2);
    assert_eq!(agent.state().await.messages[1], second);

    agent.clear_messages().await;
    assert!(agent.state().await.messages.is_empty());

    registration.unregister();
}

#[tokio::test]
async fn agent_runtime_options_are_forwarded_to_stream_fn() {
    let registration = register_faux_provider(None);
    let observed_options = Arc::new(Mutex::new(Vec::new()));
    let mut options = AgentOptions::new(registration.get_model());
    options.session_id = Some("session-abc".to_string());
    options.options = ai::SimpleStreamOptions {
        stream: ai::StreamOptions {
            transport: Some(ai::Transport::Sse),
            max_retry_delay_ms: Some(10),
            ..Default::default()
        },
        thinking_budgets: Some(ai::ThinkingBudgets {
            low: Some(123),
            ..Default::default()
        }),
        ..Default::default()
    };
    options.stream_fn = Some(Arc::new({
        let observed_options = Arc::clone(&observed_options);
        move |model, _context, options| {
            let observed_options = Arc::clone(&observed_options);
            Box::pin(async move {
                observed_options.lock().await.push(options);
                let mut message = faux_assistant_message("ok", None);
                message.api = model.api;
                message.provider = model.provider;
                message.model = model.id;
                let (mut sender, stream) = ai::AssistantMessageEventStream::channel();
                sender.push(ai::AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message,
                });
                Ok(stream)
            })
        }
    }));
    let agent = Agent::new(options);

    agent.prompt_text("hello", Vec::new()).await.unwrap();
    agent.set_session_id(Some("session-def".to_string())).await;
    agent.set_transport(Some(ai::Transport::Websocket)).await;
    agent.set_max_retry_delay_ms(Some(20)).await;
    agent
        .set_thinking_budgets(Some(ai::ThinkingBudgets {
            high: Some(456),
            ..Default::default()
        }))
        .await;
    agent.prompt_text("again", Vec::new()).await.unwrap();

    let observed = observed_options.lock().await;
    assert_eq!(observed.len(), 2);
    assert_eq!(
        observed[0].stream.session_id.as_deref(),
        Some("session-abc")
    );
    assert_eq!(observed[0].stream.transport, Some(ai::Transport::Sse));
    assert_eq!(observed[0].stream.max_retry_delay_ms, Some(10));
    assert_eq!(
        observed[0]
            .thinking_budgets
            .as_ref()
            .and_then(|budgets| budgets.low),
        Some(123)
    );
    assert_eq!(
        observed[1].stream.session_id.as_deref(),
        Some("session-def")
    );
    assert_eq!(observed[1].stream.transport, Some(ai::Transport::Websocket));
    assert_eq!(observed[1].stream.max_retry_delay_ms, Some(20));
    assert_eq!(
        observed[1]
            .thinking_budgets
            .as_ref()
            .and_then(|budgets| budgets.high),
        Some(456)
    );

    registration.unregister();
}

#[tokio::test]
async fn agent_get_api_key_falls_back_to_static_api_key() {
    let registration = register_faux_provider(None);
    let observed_api_keys = Arc::new(Mutex::new(Vec::new()));
    let mut options = AgentOptions::new(registration.get_model());
    options.options = ai::SimpleStreamOptions {
        stream: ai::StreamOptions {
            api_key: Some("static-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    options.get_api_key = Some(Arc::new(|_provider| Box::pin(async { None })));
    options.stream_fn = Some(Arc::new({
        let observed_api_keys = Arc::clone(&observed_api_keys);
        move |model, _context, options| {
            let observed_api_keys = Arc::clone(&observed_api_keys);
            Box::pin(async move {
                observed_api_keys.lock().await.push(options.stream.api_key);
                let mut message = faux_assistant_message("ok", None);
                message.api = model.api;
                message.provider = model.provider;
                message.model = model.id;
                let (mut sender, stream) = ai::AssistantMessageEventStream::channel();
                sender.push(ai::AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message,
                });
                Ok(stream)
            })
        }
    }));
    let agent = Agent::new(options);

    agent.prompt_text("hello", Vec::new()).await.unwrap();

    assert_eq!(
        *observed_api_keys.lock().await,
        vec![Some("static-key".to_string())]
    );

    registration.unregister();
}

#[tokio::test]
async fn agent_loop_ignores_message_updates_before_stream_start() {
    let registration = register_faux_provider(None);
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
    let stream_fn: ai::StreamFn = Arc::new(|model, _context, _options| {
        Box::pin(async move {
            let mut message = faux_assistant_message("final", None);
            message.api = model.api;
            message.provider = model.provider;
            message.model = model.id;
            let (mut sender, stream) = ai::AssistantMessageEventStream::channel();
            sender.push(ai::AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "ignored".to_string(),
                partial: message.clone(),
            });
            sender.push(ai::AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message,
            });
            Ok(stream)
        })
    });

    let messages = run_agent_loop(
        vec![Message::user_text("hello")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: Vec::new(),
        },
        AgentLoopConfig::new(registration.get_model()),
        emit,
        None,
        Some(stream_fn),
    )
    .await
    .unwrap();

    assert_eq!(message_roles(&messages), vec!["user", "assistant"]);
    assert!(!events.lock().await.contains(&"message_update"));

    registration.unregister();
}

#[tokio::test]
async fn agent_queue_modes_can_be_changed_after_construction() {
    let registration = register_faux_provider(None);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));

    assert_eq!(agent.steering_mode().await, QueueMode::OneAtATime);
    assert_eq!(agent.follow_up_mode().await, QueueMode::OneAtATime);

    agent.set_steering_mode(QueueMode::All).await;
    agent.set_follow_up_mode(QueueMode::All).await;

    assert_eq!(agent.steering_mode().await, QueueMode::All);
    assert_eq!(agent.follow_up_mode().await, QueueMode::All);

    registration.unregister();
}

#[tokio::test]
async fn agent_queues_can_be_cleared_independently() {
    let registration = register_faux_provider(None);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));

    agent.steer(Message::user_text("steer")).await;
    agent.follow_up(Message::user_text("follow")).await;
    assert!(agent.has_queued_messages().await);

    agent.clear_steering_queue().await;
    assert!(agent.has_queued_messages().await);

    agent.clear_follow_up_queue().await;
    assert!(!agent.has_queued_messages().await);

    agent.steer(Message::user_text("steer again")).await;
    agent.follow_up(Message::user_text("follow again")).await;
    agent.clear_all_queues().await;
    assert!(!agent.has_queued_messages().await);

    registration.unregister();
}

#[tokio::test]
async fn agent_continue_run_processes_queued_follow_up_after_assistant_tail() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("processed", None)]);
    let mut options = AgentOptions::new(registration.get_model());
    options.initial_state.messages = vec![
        Message::user_text("initial"),
        Message::Assistant(faux_assistant_message("initial response", None)),
    ];
    let agent = Agent::new(options);

    agent
        .follow_up(Message::user_text("queued follow-up"))
        .await;
    agent.continue_run().await.unwrap();

    let state = agent.state().await;
    assert_eq!(
        message_roles(&state.messages),
        vec!["user", "assistant", "user", "assistant"]
    );
    assert!(matches!(state.messages[2], Message::User(_)));
    assert!(matches!(state.messages[3], Message::Assistant(_)));

    registration.unregister();
}

#[tokio::test]
async fn agent_continue_run_keeps_one_at_a_time_steering_from_assistant_tail() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        faux_assistant_message("processed 1", None),
        faux_assistant_message("processed 2", None),
    ]);
    let mut options = AgentOptions::new(registration.get_model());
    options.initial_state.messages = vec![
        Message::user_text("initial"),
        Message::Assistant(faux_assistant_message("initial response", None)),
    ];
    let agent = Agent::new(options);

    agent.steer(Message::user_text("steering 1")).await;
    agent.steer(Message::user_text("steering 2")).await;
    agent.continue_run().await.unwrap();

    let state = agent.state().await;
    assert_eq!(
        message_roles(&state.messages),
        vec![
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant"
        ]
    );
    assert_eq!(registration.state.call_count(), 2);

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
async fn transform_context_runs_before_convert_to_llm() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("hello", None)]);

    let transformed_messages = Arc::new(Mutex::new(Vec::new()));
    let converted_messages = Arc::new(Mutex::new(Vec::new()));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.transform_context = Some(Arc::new({
        let transformed_messages = Arc::clone(&transformed_messages);
        move |messages, _token| {
            let transformed_messages = Arc::clone(&transformed_messages);
            Box::pin(async move {
                let pruned = messages
                    .into_iter()
                    .rev()
                    .take(2)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();
                *transformed_messages.lock().await = pruned.clone();
                pruned
            })
        }
    }));
    config.convert_to_llm = Some(Arc::new({
        let converted_messages = Arc::clone(&converted_messages);
        move |messages| {
            let converted_messages = Arc::clone(&converted_messages);
            Box::pin(async move {
                *converted_messages.lock().await = messages.clone();
                messages
            })
        }
    }));

    run_agent_loop(
        vec![Message::user_text("new")],
        AgentContext {
            system_prompt: None,
            messages: vec![
                Message::user_text("old 1"),
                Message::Assistant(faux_assistant_message("old response 1", None)),
                Message::user_text("old 2"),
            ],
            tools: Vec::new(),
        },
        config,
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    let transformed = transformed_messages.lock().await.clone();
    let converted = converted_messages.lock().await.clone();
    assert_eq!(message_roles(&transformed), vec!["user", "user"]);
    assert_eq!(converted, transformed);

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
async fn agent_loop_continue_returns_only_new_messages_without_reemitting_context() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("continued", None)]);

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

    let messages = run_agent_loop_continue(
        AgentContext {
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![Message::user_text("existing prompt")],
            tools: Vec::new(),
        },
        AgentLoopConfig::new(registration.get_model()),
        emit,
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(message_roles(&messages), vec!["assistant"]);
    let message_end_roles = events
        .lock()
        .await
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd { message } => Some(match message {
                Message::User(_) => "user",
                Message::Assistant(_) => "assistant",
                Message::ToolResult(_) => "toolResult",
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(message_end_roles, vec!["assistant"]);

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
async fn abort_during_tool_preflight_stops_remaining_tool_preflight() {
    let registration = register_faux_provider(None);
    registration.set_responses([tool_use_response(vec![
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
    ])]);

    let before_count = Arc::new(AtomicUsize::new(0));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.before_tool_call = Some(Arc::new({
        let before_count = Arc::clone(&before_count);
        move |_context: BeforeToolCallContext, token: Option<CancellationToken>| {
            let before_count = Arc::clone(&before_count);
            Box::pin(async move {
                before_count.fetch_add(1, Ordering::SeqCst);
                if let Some(token) = token {
                    token.cancel();
                }
                Ok::<Option<BeforeToolCallResult>, AgentError>(None)
            })
        }
    }));
    config.should_stop_after_turn = Some(Arc::new(|_context| Box::pin(async { true })));

    let messages = run_agent_loop(
        vec![Message::user_text("run tools")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool::default())],
        },
        config,
        quiet_sink(),
        Some(CancellationToken::new()),
        None,
    )
    .await
    .unwrap();

    assert_eq!(before_count.load(Ordering::SeqCst), 1);
    let tool_results = messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_results.len(), 1);
    assert_eq!(tool_results[0].tool_call_id, "tool-1");
    assert!(tool_results[0].is_error);
    assert_eq!(
        tool_results[0].content[0],
        ai::ToolResultContent::text("Operation aborted")
    );

    registration.unregister();
}

#[tokio::test]
async fn abort_during_blocking_tool_preflight_reports_abort() {
    let registration = register_faux_provider(None);
    registration.set_responses([tool_use_response(vec![faux_tool_call(
        "echo",
        json!({ "value": "first" }),
        Some("tool-1".to_string()),
    )])]);

    let mut config = AgentLoopConfig::new(registration.get_model());
    config.before_tool_call = Some(Arc::new(
        |_context: BeforeToolCallContext, token: Option<CancellationToken>| {
            Box::pin(async move {
                if let Some(token) = token {
                    token.cancel();
                }
                Ok(Some(BeforeToolCallResult {
                    block: true,
                    reason: Some("blocked after abort".to_string()),
                    args: None,
                }))
            })
        },
    ));
    config.should_stop_after_turn = Some(Arc::new(|_context| Box::pin(async { true })));

    let messages = run_agent_loop(
        vec![Message::user_text("run tools")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool::default())],
        },
        config,
        quiet_sink(),
        Some(CancellationToken::new()),
        None,
    )
    .await
    .unwrap();

    let tool_result = messages
        .iter()
        .find_map(|message| match message {
            Message::ToolResult(result) => Some(result),
            _ => None,
        })
        .expect("tool result");

    assert!(tool_result.is_error);
    assert_eq!(
        tool_result.content[0],
        ai::ToolResultContent::text("Operation aborted")
    );

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
async fn after_tool_call_null_details_keeps_original_details() {
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
                    details: Some(Value::Null),
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
            tools: vec![Arc::new(EchoTool {
                include_details: true,
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

    let Message::ToolResult(result) = &messages[2] else {
        panic!("expected tool result");
    };
    assert_eq!(result.details, Some(json!({ "value": "hello" })));

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
async fn prepare_tool_arguments_runs_before_validation() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        tool_use_response(vec![faux_tool_call(
            "edit",
            json!({ "oldText": "before", "newText": "after" }),
            Some("tool-1".to_string()),
        )]),
        faux_assistant_message("done", None),
    ]);

    let received_edits = Arc::new(Mutex::new(Vec::new()));
    let messages = run_agent_loop(
        vec![Message::user_text("edit something")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![Arc::new(PreparingEditTool {
                received_edits: Arc::clone(&received_edits),
            })],
        },
        AgentLoopConfig::new(registration.get_model()),
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(
        *received_edits.lock().await,
        vec![json!([{ "oldText": "before", "newText": "after" }])]
    );
    let Message::ToolResult(result) = &messages[2] else {
        panic!("expected tool result");
    };
    assert!(!result.is_error);

    registration.unregister();
}

#[tokio::test]
async fn tool_preflight_errors_include_empty_details() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        tool_use_response(vec![faux_tool_call(
            "missing",
            json!({ "value": "hello" }),
            Some("tool-1".to_string()),
        )]),
        faux_assistant_message("done", None),
    ]);

    let messages = run_agent_loop(
        vec![Message::user_text("run missing")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: Vec::new(),
        },
        AgentLoopConfig::new(registration.get_model()),
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    let Message::ToolResult(result) = &messages[2] else {
        panic!("expected tool result");
    };
    assert!(result.is_error);
    assert_eq!(result.details, Some(json!({})));

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
async fn sequential_tool_execution_mode_on_one_tool_forces_mixed_batch_sequential() {
    let registration = register_faux_provider(None);
    registration.set_responses([
        tool_use_response(vec![
            faux_tool_call(
                "slow",
                json!({ "value": "first" }),
                Some("tool-1".to_string()),
            ),
            faux_tool_call(
                "fast",
                json!({ "value": "second" }),
                Some("tool-2".to_string()),
            ),
        ]),
        faux_assistant_message("done", None),
    ]);

    let release_first = Arc::new(Notify::new());
    let first_done = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let executions = Arc::new(Mutex::new(Vec::new()));
    tokio::spawn({
        let release_first = Arc::clone(&release_first);
        async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            release_first.notify_waiters();
        }
    });

    run_agent_loop(
        vec![Message::user_text("run both")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: vec![
                Arc::new(EchoTool {
                    name: Some("slow"),
                    mode: Some(ToolExecutionMode::Sequential),
                    first_done: Some(Arc::clone(&first_done)),
                    parallel_observed: Some(Arc::clone(&parallel_observed)),
                    release_first: Some(Arc::clone(&release_first)),
                    executions: Some(Arc::clone(&executions)),
                    ..Default::default()
                }),
                Arc::new(EchoTool {
                    name: Some("fast"),
                    mode: Some(ToolExecutionMode::Parallel),
                    first_done: Some(Arc::clone(&first_done)),
                    parallel_observed: Some(Arc::clone(&parallel_observed)),
                    executions: Some(Arc::clone(&executions)),
                    ..Default::default()
                }),
            ],
        },
        AgentLoopConfig::new(registration.get_model()),
        quiet_sink(),
        None,
        None,
    )
    .await
    .unwrap();

    assert!(!parallel_observed.load(Ordering::SeqCst));
    assert_eq!(*executions.lock().await, vec!["first", "second"]);

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
    name: Option<&'static str>,
    delay_first: bool,
    include_details: bool,
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
            name: self.name.unwrap_or("echo").to_string(),
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
        if self.include_details {
            result.details = Some(json!({ "value": value }));
        }
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

struct PreparingEditTool {
    received_edits: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl AgentTool for PreparingEditTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "edit".to_string(),
            description: "Edit tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": { "type": "string" },
                                "newText": { "type": "string" }
                            },
                            "required": ["oldText", "newText"]
                        }
                    }
                },
                "required": ["edits"]
            }),
        }
    }

    fn label(&self) -> &str {
        "Edit"
    }

    fn prepare_arguments(&self, args: Value) -> AgentResult<Value> {
        let Some(old_text) = args.get("oldText").and_then(Value::as_str) else {
            return Ok(args);
        };
        let Some(new_text) = args.get("newText").and_then(Value::as_str) else {
            return Ok(args);
        };

        Ok(json!({
            "edits": [{ "oldText": old_text, "newText": new_text }]
        }))
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: Value,
        _cancellation_token: Option<CancellationToken>,
        _on_update: Option<ai::AgentToolUpdateCallback>,
    ) -> AgentResult<AgentToolResult> {
        self.received_edits.lock().await.push(args["edits"].clone());
        Ok(AgentToolResult::text("edited"))
    }
}
