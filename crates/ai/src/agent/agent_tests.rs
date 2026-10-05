//! Port of `packages/agent/test/agent.test.ts` and `test/e2e.test.ts`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::Notify;

use super::*;
use crate::agent::agent_loop::tests::{
    create_assistant_message, create_model, create_user_message, empty_schema, finished_stream,
    roles, text, text_stream, tool_call, user_texts,
};
use crate::agent::stream_fn::{DEFAULT_STREAM_FN_TEST_LOCK, set_default_stream_fn, stream_fn};
use crate::agent::types::{
    AgentToolBuilder, AgentToolResult, AgentToolUpdateCallback, AgentTurnDecision,
};
use crate::types::{
    AssistantMessageEvent, AssistantMessageEventStream, SystemMessage, SystemMessageContent, Tool,
    ToolReference, ToolResultContent, ToolResultMessage, UserMessageContent,
};

fn create_tool(name: &str) -> DynAgentTool {
    let result = name.to_string();
    AgentToolBuilder::new(name)
        .label(name)
        .description(format!("{name} tool"))
        .parameters(empty_schema())
        .execute(move |_| {
            let result = result.clone();
            async move { Ok(AgentToolResult::text(result)) }
        })
        .build()
        .unwrap()
}

fn echo_tool() -> DynAgentTool {
    AgentToolBuilder::new("echo")
        .label("Echo")
        .description("Echo input")
        .parameters(empty_schema())
        .execute(|_| async { Ok(AgentToolResult::text("echo")) })
        .build()
        .unwrap()
}

fn declaration(name: &str, description: &str) -> Tool {
    Tool {
        name: name.to_string(),
        description: description.to_string(),
        parameters: empty_schema(),
        constrained_sampling: None,
    }
}

fn tool_use_message(calls: Vec<crate::types::AssistantContent>) -> AssistantMessage {
    create_assistant_message(calls, StopReason::ToolUse)
}

/// A stream function that is never called (Pi's `unusedStreamFunction`).
fn unused_stream_function() -> StreamFn {
    stream_fn(|_, _, _| panic!("Unexpected stream call"))
}

fn agent_with(options: AgentOptions) -> Agent {
    Agent::new(options)
}

fn options_with_stream(stream: StreamFn) -> AgentOptions {
    AgentOptions {
        stream_fn: Some(stream),
        ..Default::default()
    }
}

/// A stream that starts and then waits until `signal` aborts it.
fn stream_until_aborted(
    options: &crate::types::SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    stream.push(AssistantMessageEvent::Start {
        partial: create_assistant_message(vec![text("")], StopReason::Stop),
    });
    let producer = stream.clone();
    let signal = options.signal.clone();
    tokio::spawn(async move {
        loop {
            if signal.as_ref().is_some_and(|signal| signal.is_cancelled()) {
                let error = create_assistant_message(vec![text("Aborted")], StopReason::Aborted);
                producer.push(AssistantMessageEvent::Error {
                    reason: StopReason::Aborted,
                    error,
                });
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    stream
}

/// Stream function recording the user texts of each request.
fn recording_stream(requests: Arc<Mutex<Vec<Vec<String>>>>) -> StreamFn {
    stream_fn(move |_, context, _| {
        requests.lock().push(user_texts(&context.messages));
        text_stream("done")
    })
}

#[tokio::test]
async fn uses_the_configured_default_when_a_legacy_caller_omits_stream_fn() {
    let _lock = DEFAULT_STREAM_FN_TEST_LOCK.lock().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    set_default_stream_fn(Some(stream_fn(move |_, _, _| {
        counter.fetch_add(1, Ordering::SeqCst);
        text_stream("fallback")
    })));

    let agent = Agent::new(AgentOptions::default());
    let result = agent.prompt_text("Hello", Vec::new()).await;
    set_default_stream_fn(None);

    result.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn should_create_an_agent_instance_with_default_state() {
    let agent = agent_with(options_with_stream(unused_stream_function()));
    let state = agent.state();

    assert_eq!(state.model.id, "unknown");
    assert_eq!(state.thinking_level, ModelThinkingLevel::Off);
    assert!(state.tools.is_empty());
    assert!(state.messages.is_empty());
    assert!(!state.is_streaming);
    assert!(state.streaming_message.is_none());
    assert!(state.pending_tool_calls.is_empty());
    assert!(state.error_message.is_none());
}

#[test]
fn should_create_an_agent_instance_with_custom_initial_state() {
    let custom_model = Model {
        id: "gpt-4o-mini".to_string(),
        ..create_model()
    };
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are a helpful assistant.".to_string()),
            model: Some(custom_model.clone()),
            thinking_level: Some(ModelThinkingLevel::Low),
            ..Default::default()
        },
        ..options_with_stream(unused_stream_function())
    });
    let state = agent.state();

    assert_eq!(
        state.messages,
        [Message::System(SystemMessage {
            content: SystemMessageContent::Text("You are a helpful assistant.".to_string()),
            timestamp: 0,
            ..Default::default()
        })]
    );
    assert_eq!(state.system_prompt, "You are a helpful assistant.");
    assert_eq!(state.model, custom_model);
    assert_eq!(state.thinking_level, ModelThinkingLevel::Low);
}

#[test]
fn converts_initial_prompt_and_tools_into_transcript_state() {
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".to_string()),
            tools: vec![echo_tool()],
            ..Default::default()
        },
        ..options_with_stream(unused_stream_function())
    });

    let Some(Message::System(initial)) = agent.messages().first().cloned() else {
        panic!("expected initial system message");
    };
    assert_eq!(
        initial.content,
        SystemMessageContent::Text("You are helpful.".to_string())
    );
    let names: Vec<_> = initial
        .tools_added
        .unwrap_or_default()
        .into_iter()
        .map(|tool| tool.name)
        .collect();
    assert_eq!(names, ["echo"]);
}

#[tokio::test]
async fn declares_tool_loadout_changes_to_the_model_before_the_next_request() {
    let requests = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let recorded = Arc::clone(&requests);
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".to_string()),
            tools: vec![create_tool("first")],
            ..Default::default()
        },
        ..options_with_stream(stream_fn(move |_, context, _| {
            let entries = context
                .messages
                .iter()
                .filter_map(|message| match message {
                    Message::System(system) => Some(system),
                    _ => None,
                })
                .flat_map(|system| {
                    let added: Vec<_> = system
                        .tools_added
                        .iter()
                        .flatten()
                        .map(|tool| tool.name.clone())
                        .collect();
                    let removed: Vec<_> = system
                        .tools_removed
                        .iter()
                        .flatten()
                        .map(|tool| tool.name.clone())
                        .collect();
                    [
                        format!("+{}", added.join(",")),
                        format!("-{}", removed.join(",")),
                    ]
                })
                .collect();
            recorded.lock().push(entries);
            text_stream("done")
        }))
    });

    agent.prompt_text("one", Vec::new()).await.unwrap();
    agent.set_tools(vec![create_tool("second")]);
    agent.prompt_text("two", Vec::new()).await.unwrap();
    agent.prompt_text("three", Vec::new()).await.unwrap();

    assert_eq!(
        requests.lock().as_slice(),
        [
            vec!["+first", "-"],
            vec!["+first", "-", "+second", "-first"],
            vec!["+first", "-", "+second", "-first"],
        ]
    );
    let update = agent
        .messages()
        .into_iter()
        .find_map(|message| match message {
            Message::System(system) if system.tools_removed.is_some() => Some(system),
            _ => None,
        })
        .expect("update message");
    assert_eq!(
        update,
        SystemMessage {
            content: SystemMessageContent::Text(String::new()),
            tools_added: Some(vec![declaration("second", "second tool")]),
            tools_removed: Some(vec![ToolReference {
                name: "first".to_string(),
            }]),
            timestamp: update.timestamp,
            ..Default::default()
        }
    );
}

#[tokio::test]
async fn merges_tool_changes_into_a_pending_system_message() {
    let system_counts = Arc::new(Mutex::new(Vec::new()));
    let counts = Arc::clone(&system_counts);
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".to_string()),
            ..Default::default()
        },
        ..options_with_stream(stream_fn(move |_, context, _| {
            counts.lock().push(
                context
                    .messages
                    .iter()
                    .filter(|message| matches!(message, Message::System(_)))
                    .count(),
            );
            text_stream("done")
        }))
    });

    agent.set_tools(vec![echo_tool()]);
    let sections = [("skills".to_string(), Some("<skills>x</skills>".to_string()))]
        .into_iter()
        .collect();
    agent
        .prompt_messages(vec![
            Message::System(SystemMessage {
                content: SystemMessageContent::Text(String::new()),
                sections: Some(sections),
                timestamp: 1,
                ..Default::default()
            }),
            Message::User(crate::types::UserMessage {
                content: UserMessageContent::Text("hi".to_string()),
                timestamp: 2,
            }),
        ])
        .await
        .unwrap();

    assert_eq!(system_counts.lock().as_slice(), [2]);
    let sections = [("skills".to_string(), Some("<skills>x</skills>".to_string()))]
        .into_iter()
        .collect();
    assert_eq!(
        agent.messages()[1],
        Message::System(SystemMessage {
            content: SystemMessageContent::Text(String::new()),
            sections: Some(sections),
            tools_added: Some(vec![declaration("echo", "Echo input")]),
            timestamp: 1,
            ..Default::default()
        })
    );
}

#[tokio::test]
async fn rewrites_pending_tool_declarations_to_match_the_executable_set() {
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".to_string()),
            tools: vec![create_tool("first")],
            ..Default::default()
        },
        ..options_with_stream(stream_fn(|_, _, _| text_stream("done")))
    });

    // The pending message claims to add `second` and remove `first`, but the executable
    // set still has `first` and lacks `second`: the executable set wins.
    let sections = || {
        [("note".to_string(), Some("<note>x</note>".to_string()))]
            .into_iter()
            .collect()
    };
    agent
        .prompt_messages(vec![
            Message::System(SystemMessage {
                content: SystemMessageContent::Text(String::new()),
                sections: Some(sections()),
                tools_added: Some(vec![to_tool_declaration(
                    &create_tool("second").definition(),
                )]),
                tools_removed: Some(vec![ToolReference {
                    name: "first".to_string(),
                }]),
                timestamp: 1,
            }),
            Message::User(crate::types::UserMessage {
                content: UserMessageContent::Text("hi".to_string()),
                timestamp: 2,
            }),
        ])
        .await
        .unwrap();

    assert_eq!(
        agent.messages()[1],
        Message::System(SystemMessage {
            content: SystemMessageContent::Text(String::new()),
            sections: Some(sections()),
            timestamp: 1,
            ..Default::default()
        })
    );
    let current: Vec<_> = get_current_system_message(&agent.messages())
        .and_then(|message| message.tools_added)
        .unwrap_or_default()
        .into_iter()
        .map(|tool| tool.name)
        .collect();
    assert_eq!(current, ["first"]);
}

#[test]
fn restores_the_transcript_baseline_when_reset() {
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            system_prompt: Some("You are helpful.".to_string()),
            tools: vec![echo_tool()],
            messages: vec![Message::User(crate::types::UserMessage {
                content: UserMessageContent::Text("old".to_string()),
                timestamp: 1,
            })],
            ..Default::default()
        },
        ..options_with_stream(unused_stream_function())
    });

    agent.reset().unwrap();

    let messages = agent.messages();
    assert_eq!(messages.len(), 1);
    let Message::System(initial) = &messages[0] else {
        panic!("expected initial system message");
    };
    assert_eq!(
        initial.content,
        SystemMessageContent::Text("You are helpful.".to_string())
    );
    let names: Vec<_> = initial
        .tools_added
        .iter()
        .flatten()
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(names, ["echo"]);
}

#[test]
fn should_subscribe_to_events() {
    let agent = agent_with(options_with_stream(unused_stream_function()));
    let event_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&event_count);
    let subscription = agent.subscribe(move |_, _| {
        counter.fetch_add(1, Ordering::SeqCst);
        async { Ok(()) }
    });

    // No initial event on subscribe.
    assert_eq!(event_count.load(Ordering::SeqCst), 0);

    // State mutators don't emit events.
    agent.set_thinking_level(ModelThinkingLevel::Low);
    assert_eq!(event_count.load(Ordering::SeqCst), 0);
    assert_eq!(agent.state().thinking_level, ModelThinkingLevel::Low);

    // Unsubscribe should work.
    assert!(subscription.unsubscribe());
    agent.set_thinking_level(ModelThinkingLevel::High);
    assert_eq!(event_count.load(Ordering::SeqCst), 0);
}

/// Pi's stream function throws; a Rust stream function cannot fail, so a
/// missing default stream function supplies the thrown error.
#[tokio::test]
async fn emits_full_lifecycle_events_for_thrown_run_failures() {
    let _lock = DEFAULT_STREAM_FN_TEST_LOCK.lock().await;
    set_default_stream_fn(None);
    let agent = Agent::new(AgentOptions::default());
    let events = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&events);
    let _subscription = agent.subscribe(move |event, _| {
        recorded.lock().push(event.event_type());
        async { Ok(()) }
    });

    agent.prompt_text("hello", Vec::new()).await.unwrap();

    assert_eq!(
        events.lock().as_slice(),
        [
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
    let expected = AgentError::NoDefaultStreamFn.to_string();
    let Some(Message::Assistant(last_message)) = agent.messages().last().cloned() else {
        panic!("Expected assistant message");
    };
    assert_eq!(last_message.stop_reason, StopReason::Error);
    assert_eq!(
        last_message.error_message.as_deref(),
        Some(expected.as_str())
    );
    assert_eq!(agent.state().error_message, Some(expected));
}

#[tokio::test]
async fn should_await_async_subscribers_before_prompt_resolves() {
    let barrier = Arc::new(Notify::new());
    let agent = agent_with(options_with_stream(stream_fn(|_, _, _| text_stream("ok"))));
    let listener_finished = Arc::new(AtomicBool::new(false));
    let _subscription = {
        let barrier = Arc::clone(&barrier);
        let finished = Arc::clone(&listener_finished);
        agent.subscribe(move |event, _| {
            let barrier = Arc::clone(&barrier);
            let finished = Arc::clone(&finished);
            async move {
                if matches!(event, AgentEvent::AgentEnd { .. }) {
                    barrier.notified().await;
                    finished.store(true, Ordering::SeqCst);
                }
                Ok(())
            }
        })
    };

    let prompt = tokio::spawn({
        let agent = agent.clone();
        async move { agent.prompt_text("hello", Vec::new()).await }
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!prompt.is_finished());
    assert!(!listener_finished.load(Ordering::SeqCst));
    assert!(agent.state().is_streaming);

    barrier.notify_one();
    prompt.await.unwrap().unwrap();

    assert!(listener_finished.load(Ordering::SeqCst));
    assert!(!agent.state().is_streaming);
}

#[tokio::test]
async fn wait_for_idle_should_wait_for_async_subscribers() {
    let barrier = Arc::new(Notify::new());
    let agent = agent_with(options_with_stream(stream_fn(|_, _, _| text_stream("ok"))));
    let _subscription = {
        let barrier = Arc::clone(&barrier);
        agent.subscribe(move |event, _| {
            let barrier = Arc::clone(&barrier);
            async move {
                if matches!(
                    event,
                    AgentEvent::MessageEnd {
                        message: Message::Assistant(_)
                    }
                ) {
                    barrier.notified().await;
                }
                Ok(())
            }
        })
    };

    let prompt = tokio::spawn({
        let agent = agent.clone();
        async move { agent.prompt_text("hello", Vec::new()).await }
    });
    tokio::time::sleep(Duration::from_millis(1)).await;
    let idle = tokio::spawn({
        let agent = agent.clone();
        async move { agent.wait_for_idle().await }
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!idle.is_finished());
    assert!(agent.state().is_streaming);

    barrier.notify_one();
    prompt.await.unwrap().unwrap();
    idle.await.unwrap();

    assert!(!agent.state().is_streaming);
}

#[tokio::test]
async fn should_pass_the_active_abort_signal_to_subscribers() {
    let agent = agent_with(options_with_stream(stream_fn(|_, _, options| {
        stream_until_aborted(&options)
    })));
    let received_signal = Arc::new(Mutex::new(None));
    let _subscription = {
        let received = Arc::clone(&received_signal);
        agent.subscribe(move |event, signal| {
            if matches!(event, AgentEvent::AgentStart) {
                *received.lock() = Some(signal);
            }
            async { Ok(()) }
        })
    };

    let prompt = tokio::spawn({
        let agent = agent.clone();
        async move { agent.prompt_text("hello", Vec::new()).await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;

    let signal = received_signal.lock().clone().expect("signal");
    assert!(!signal.is_cancelled());

    agent.abort();
    prompt.await.unwrap().unwrap();

    assert!(signal.is_cancelled());
}

#[tokio::test]
async fn should_ignore_tool_updates_after_the_tool_execution_settles() {
    let delayed_update: Arc<Mutex<Option<AgentToolUpdateCallback>>> = Arc::new(Mutex::new(None));
    let events = Arc::new(Mutex::new(Vec::new()));
    let tool = {
        let delayed_update = Arc::clone(&delayed_update);
        AgentToolBuilder::new("delayed_tool")
            .label("Delayed Tool")
            .description("Captures progress callbacks")
            .parameters(empty_schema())
            .execute_with_context(move |_, _, _, on_update| {
                let delayed_update = Arc::clone(&delayed_update);
                async move {
                    let on_update = on_update.expect("on_update");
                    *delayed_update.lock() = Some(Arc::clone(&on_update));
                    // Not awaited, like Pi's fire-and-forget `onUpdate?.(...)`.
                    drop(on_update(AgentToolResult {
                        details: Some(json!({ "status": "running" })),
                        ..AgentToolResult::text("running")
                    }));
                    Ok(AgentToolResult {
                        details: Some(json!({ "status": "done" })),
                        terminate: true,
                        ..AgentToolResult::text("ok")
                    })
                }
            })
            .build()
            .unwrap()
    };
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            tools: vec![tool],
            ..Default::default()
        },
        ..options_with_stream(stream_fn(|_, _, _| {
            finished_stream(tool_use_message(vec![tool_call(
                "call-1",
                "delayed_tool",
                json!({}),
            )]))
        }))
    });
    let _subscription = {
        let events = Arc::clone(&events);
        agent.subscribe(move |event, _| {
            events.lock().push(event);
            async { Ok(()) }
        })
    };

    agent.prompt_text("run tool", Vec::new()).await.unwrap();
    let event_count_after_prompt = events.lock().len();

    let update = delayed_update.lock().clone().expect("update callback");
    update(AgentToolResult {
        details: Some(json!({ "status": "late" })),
        ..AgentToolResult::text("late")
    })
    .await;
    tokio::time::sleep(Duration::from_millis(1)).await;

    let events = events.lock();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolExecutionUpdate { .. }))
            .count(),
        1
    );
    assert_eq!(events.len(), event_count_after_prompt);
}

#[tokio::test]
async fn should_ignore_a_settled_parallel_tool_update_while_another_tool_is_still_running() {
    let slow_started = Arc::new(Notify::new());
    let settled_tool_ended = Arc::new(Notify::new());
    let release_slow = Arc::new(Notify::new());
    let settled_tool_update: Arc<Mutex<Option<AgentToolUpdateCallback>>> =
        Arc::new(Mutex::new(None));
    let events = Arc::new(Mutex::new(Vec::new()));
    let settled_tool = {
        let settled_tool_update = Arc::clone(&settled_tool_update);
        AgentToolBuilder::new("settled_tool")
            .label("Settled Tool")
            .description("Captures progress callbacks")
            .parameters(empty_schema())
            .execute_with_context(move |_, _, _, on_update| {
                let settled_tool_update = Arc::clone(&settled_tool_update);
                async move {
                    *settled_tool_update.lock() = on_update;
                    Ok(AgentToolResult {
                        details: Some(json!({ "status": "done" })),
                        terminate: true,
                        ..AgentToolResult::text("done")
                    })
                }
            })
            .build()
            .unwrap()
    };
    let slow_tool = {
        let slow_started = Arc::clone(&slow_started);
        let release_slow = Arc::clone(&release_slow);
        AgentToolBuilder::new("slow_tool")
            .label("Slow Tool")
            .description("Keeps the agent run active")
            .parameters(empty_schema())
            .execute(move |_| {
                let slow_started = Arc::clone(&slow_started);
                let release_slow = Arc::clone(&release_slow);
                async move {
                    slow_started.notify_one();
                    release_slow.notified().await;
                    Ok(AgentToolResult {
                        details: Some(json!({ "status": "done" })),
                        terminate: true,
                        ..AgentToolResult::text("done")
                    })
                }
            })
            .build()
            .unwrap()
    };
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            tools: vec![settled_tool, slow_tool],
            ..Default::default()
        },
        ..options_with_stream(stream_fn(|_, _, _| {
            finished_stream(tool_use_message(vec![
                tool_call("call-1", "settled_tool", json!({})),
                tool_call("call-2", "slow_tool", json!({})),
            ]))
        }))
    });
    let _subscription = {
        let events = Arc::clone(&events);
        let settled_tool_ended = Arc::clone(&settled_tool_ended);
        agent.subscribe(move |event, _| {
            if matches!(&event, AgentEvent::ToolExecutionEnd { tool_call_id, .. } if tool_call_id == "call-1")
            {
                settled_tool_ended.notify_one();
            }
            events.lock().push(event);
            async { Ok(()) }
        })
    };

    let prompt = tokio::spawn({
        let agent = agent.clone();
        async move { agent.prompt_text("run tools", Vec::new()).await }
    });
    slow_started.notified().await;
    settled_tool_ended.notified().await;
    let event_count_before_late_update = events.lock().len();

    let update = settled_tool_update.lock().clone().expect("update callback");
    update(AgentToolResult {
        details: Some(json!({ "status": "late" })),
        ..AgentToolResult::text("late")
    })
    .await;
    tokio::time::sleep(Duration::from_millis(1)).await;
    assert_eq!(events.lock().len(), event_count_before_late_update);

    release_slow.notify_one();
    prompt.await.unwrap().unwrap();
    assert_eq!(
        events
            .lock()
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolExecutionUpdate { .. }))
            .count(),
        0
    );
}

#[test]
fn should_update_state_with_mutators() {
    let agent = agent_with(options_with_stream(unused_stream_function()));

    let new_model = Model {
        id: "gemini-2.5-flash".to_string(),
        provider: "google".to_string(),
        ..create_model()
    };
    agent.set_model(new_model.clone());
    assert_eq!(agent.state().model, new_model);

    agent.set_thinking_level(ModelThinkingLevel::High);
    assert_eq!(agent.state().thinking_level, ModelThinkingLevel::High);

    agent.set_tools(vec![create_tool("test")]);
    assert_eq!(agent.state().tools.len(), 1);

    let messages = vec![create_user_message("Hello")];
    agent.set_messages(messages.clone());
    assert_eq!(agent.messages(), messages);

    let new_message =
        Message::Assistant(create_assistant_message(vec![text("Hi")], StopReason::Stop));
    agent.push_message(new_message.clone());
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[1], new_message);

    agent.set_messages(Vec::new());
    assert!(agent.messages().is_empty());
}

#[test]
fn should_support_steering_message_queue() {
    let agent = agent_with(options_with_stream(unused_stream_function()));
    let message = create_user_message("Steering message");
    agent.steer(message.clone());

    // The message is queued but not yet in the transcript.
    assert!(!agent.messages().contains(&message));
    assert!(agent.has_queued_messages());
}

#[test]
fn should_support_follow_up_message_queue() {
    let agent = agent_with(options_with_stream(unused_stream_function()));
    let message = create_user_message("Follow-up message");
    agent.follow_up(message.clone());

    assert!(!agent.messages().contains(&message));
    assert!(agent.has_queued_messages());
}

#[test]
fn should_handle_abort_controller() {
    let agent = agent_with(options_with_stream(unused_stream_function()));
    // Should not panic even if nothing is running.
    agent.abort();
    assert!(agent.signal().is_none());
}

#[tokio::test]
async fn should_reject_reset_while_processing_without_corrupting_the_transcript() {
    let stream_started = Arc::new(Notify::new());
    let release_response = Arc::new(Notify::new());
    let agent = {
        let stream_started = Arc::clone(&stream_started);
        let release_response = Arc::clone(&release_response);
        agent_with(options_with_stream(stream_fn(move |_, _, _| {
            let stream = AssistantMessageEventStream::new();
            let producer = stream.clone();
            let stream_started = Arc::clone(&stream_started);
            let release_response = Arc::clone(&release_response);
            tokio::spawn(async move {
                producer.push(AssistantMessageEvent::Start {
                    partial: create_assistant_message(vec![text("")], StopReason::Stop),
                });
                stream_started.notify_one();
                release_response.notified().await;
                let message = create_assistant_message(vec![text("Done")], StopReason::Stop);
                producer.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message,
                });
            });
            stream
        })))
    };

    let prompt = tokio::spawn({
        let agent = agent.clone();
        async move { agent.prompt_text("Hello", Vec::new()).await }
    });
    stream_started.notified().await;
    // Let the loop observe the `start` event.
    tokio::time::sleep(Duration::from_millis(5)).await;

    assert!(agent.state().is_streaming);
    assert_eq!(roles(&agent.messages()), ["user"]);
    let error = agent.reset().unwrap_err();
    assert_eq!(
        error.to_string(),
        "Agent is already processing. Wait for completion before resetting."
    );
    assert!(agent.state().is_streaming);
    assert_eq!(roles(&agent.messages()), ["user"]);

    release_response.notify_one();
    prompt.await.unwrap().unwrap();

    assert!(!agent.state().is_streaming);
    assert_eq!(roles(&agent.messages()), ["user", "assistant"]);
}

#[tokio::test]
async fn should_throw_when_prompt_called_while_streaming() {
    let agent = agent_with(options_with_stream(stream_fn(|_, _, options| {
        stream_until_aborted(&options)
    })));

    let first_prompt = tokio::spawn({
        let agent = agent.clone();
        async move { agent.prompt_text("First message", Vec::new()).await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(agent.state().is_streaming);

    let error = agent
        .prompt_text("Second message", Vec::new())
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
    );

    agent.abort();
    first_prompt.await.unwrap().unwrap();
}

#[tokio::test]
async fn should_throw_when_continue_called_while_streaming() {
    let agent = agent_with(options_with_stream(stream_fn(|_, _, options| {
        stream_until_aborted(&options)
    })));

    let first_prompt = tokio::spawn({
        let agent = agent.clone();
        async move { agent.prompt_text("First message", Vec::new()).await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(agent.state().is_streaming);

    let error = agent.continue_run().await.unwrap_err();
    assert_eq!(
        error.to_string(),
        "Agent is already processing. Wait for completion before continuing."
    );

    agent.abort();
    first_prompt.await.unwrap().unwrap();
}

#[tokio::test]
async fn continue_should_process_queued_follow_up_messages_after_an_assistant_turn() {
    let agent = agent_with(options_with_stream(stream_fn(|_, _, _| {
        text_stream("Processed")
    })));
    agent.set_messages(vec![
        create_user_message("Initial"),
        Message::Assistant(create_assistant_message(
            vec![text("Initial response")],
            StopReason::Stop,
        )),
    ]);
    agent.follow_up(create_user_message("Queued follow-up"));

    agent.continue_run().await.unwrap();

    let messages = agent.messages();
    assert!(user_texts(&messages).contains(&"Queued follow-up".to_string()));
    assert_eq!(messages.last().map(Message::role), Some("assistant"));
}

async fn continue_keeps_steering_semantics_for_assistant_tail(
    mode: QueueMode,
    expected_requests: usize,
) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = agent_with(AgentOptions {
        steering_mode: mode,
        ..options_with_stream(recording_stream(Arc::clone(&requests)))
    });
    agent.set_messages(vec![
        create_user_message("Initial"),
        Message::Assistant(create_assistant_message(
            vec![text("Initial response")],
            StopReason::Stop,
        )),
    ]);
    agent.steer(create_user_message("Steering 1"));
    agent.steer(create_user_message("Steering 2"));

    agent.continue_run().await.unwrap();

    let requests = requests.lock();
    assert_eq!(requests.len(), expected_requests);
    assert!(requests[0].contains(&"Steering 1".to_string()));
    if mode == QueueMode::OneAtATime {
        assert!(!requests[0].contains(&"Steering 2".to_string()));
        assert!(requests[1].contains(&"Steering 2".to_string()));
    } else {
        assert!(requests[0].contains(&"Steering 2".to_string()));
    }
}

#[tokio::test]
async fn continue_keeps_one_at_a_time_steering_semantics_for_assistant_tail_fallback() {
    continue_keeps_steering_semantics_for_assistant_tail(QueueMode::OneAtATime, 2).await;
}

#[tokio::test]
async fn continue_keeps_all_steering_semantics_for_assistant_tail_fallback() {
    continue_keeps_steering_semantics_for_assistant_tail(QueueMode::All, 1).await;
}

fn noop_tool() -> DynAgentTool {
    AgentToolBuilder::new("noop")
        .label("Noop")
        .description("Noop tool")
        .parameters(empty_schema())
        .execute(|_| async { Ok(AgentToolResult::text("ok")) })
        .build()
        .unwrap()
}

/// Tool use on the first request, `final_text` afterwards.
fn tool_use_then_text(final_text: &'static str) -> (StreamFn, Arc<AtomicUsize>) {
    let request_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&request_count);
    let stream_fn = stream_fn(move |_, _, _| {
        if counter.fetch_add(1, Ordering::SeqCst) == 0 {
            finished_stream(tool_use_message(vec![tool_call(
                "tool-1",
                "noop",
                json!({}),
            )]))
        } else {
            text_stream(final_text)
        }
    });
    (stream_fn, request_count)
}

#[tokio::test]
async fn keeps_legacy_prepare_next_turn_signal_callback_behavior() {
    let saw_abort_signal = Arc::new(AtomicBool::new(false));
    let (stream, request_count) = tool_use_then_text("done");
    let saw = Arc::clone(&saw_abort_signal);
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            tools: vec![noop_tool()],
            ..Default::default()
        },
        prepare_next_turn: Some(Arc::new(move |signal| {
            saw.store(signal.is_some(), Ordering::SeqCst);
            Box::pin(async { None })
        })),
        ..options_with_stream(stream)
    });

    agent.prompt_text("start", Vec::new()).await.unwrap();

    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    assert!(saw_abort_signal.load(Ordering::SeqCst));
}

#[tokio::test]
async fn forwards_finish_turn_through_agent_options_with_the_active_abort_signal() {
    let saw_abort_signal = Arc::new(AtomicBool::new(false));
    let callback_context_roles = Arc::new(Mutex::new(Vec::new()));
    let (stream, request_count) = tool_use_then_text("should not run");
    let saw = Arc::clone(&saw_abort_signal);
    let context_roles = Arc::clone(&callback_context_roles);
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            tools: vec![noop_tool()],
            ..Default::default()
        },
        finish_turn: Some(Arc::new(move |turn, signal| {
            saw.store(signal.is_some(), Ordering::SeqCst);
            *context_roles.lock() = roles(&turn.context.messages);
            Box::pin(async { Some(AgentTurnDecision::End) })
        })),
        ..options_with_stream(stream)
    });

    agent.prompt_text("start", Vec::new()).await.unwrap();

    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    assert!(saw_abort_signal.load(Ordering::SeqCst));
    assert_eq!(
        callback_context_roles.lock().as_slice(),
        ["system", "user", "assistant", "toolResult"]
    );
}

async fn rejects_a_queued_continuation_without_draining_queues(messages: Vec<AgentMessage>) {
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            messages,
            ..Default::default()
        },
        ..options_with_stream(unused_stream_function())
    });
    let steering = create_user_message("steering");
    let follow_up = create_user_message("follow-up");
    agent.steer(steering.clone());
    agent.follow_up(follow_up.clone());

    let error = agent.continue_run().await.unwrap_err();
    assert_eq!(error.to_string(), "No messages to continue from");
    assert_eq!(agent.peek_queued_messages(), [steering]);
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), [follow_up]);
    assert!(!agent.state().is_streaming);
}

#[tokio::test]
async fn rejects_a_queued_continuation_from_empty_context_without_draining_queues() {
    rejects_a_queued_continuation_without_draining_queues(Vec::new()).await;
}

#[tokio::test]
async fn rejects_a_queued_continuation_from_system_only_context_without_draining_queues() {
    rejects_a_queued_continuation_without_draining_queues(vec![Message::System(SystemMessage {
        content: SystemMessageContent::Text("system only".to_string()),
        timestamp: 1,
        ..Default::default()
    })])
    .await;
}

async fn defers_follow_up_input_on_the_first_continuation_request(messages: Vec<AgentMessage>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            messages,
            ..Default::default()
        },
        ..options_with_stream(recording_stream(Arc::clone(&requests)))
    });
    agent.follow_up(create_user_message("follow-up"));

    agent.continue_run().await.unwrap();

    let requests = requests.lock();
    assert_eq!(requests.len(), 2);
    assert!(!requests[0].contains(&"follow-up".to_string()));
    assert!(requests[1].contains(&"follow-up".to_string()));
}

#[tokio::test]
async fn defers_follow_up_input_on_the_first_continuation_request_from_a_user_tail() {
    defers_follow_up_input_on_the_first_continuation_request(vec![create_user_message(
        "existing user",
    )])
    .await;
}

#[tokio::test]
async fn defers_follow_up_input_on_the_first_continuation_request_from_a_tool_result_tail() {
    defers_follow_up_input_on_the_first_continuation_request(vec![
        create_user_message("existing user"),
        Message::Assistant(tool_use_message(vec![tool_call(
            "call-1",
            "noop",
            json!({}),
        )])),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "call-1".to_string(),
            tool_name: "noop".to_string(),
            content: vec![ToolResultContent::text("done")],
            details: None,
            usage: None,
            nested_calls: None,
            is_error: false,
            timestamp: 1,
        }),
    ])
    .await;
}

async fn polls_steering_at_continuation_startup(mode: QueueMode, expected_requests: usize) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            messages: vec![create_user_message("existing")],
            ..Default::default()
        },
        steering_mode: mode,
        ..options_with_stream(recording_stream(Arc::clone(&requests)))
    });
    agent.steer(create_user_message("first"));
    agent.steer(create_user_message("second"));

    agent.continue_run().await.unwrap();

    let requests = requests.lock();
    assert_eq!(requests.len(), expected_requests);
    assert!(requests[0].contains(&"first".to_string()));
    if mode == QueueMode::OneAtATime {
        assert!(!requests[0].contains(&"second".to_string()));
        assert!(requests[1].contains(&"second".to_string()));
    } else {
        assert!(requests[0].contains(&"second".to_string()));
    }
}

#[tokio::test]
async fn polls_one_at_a_time_steering_at_continuation_startup() {
    polls_steering_at_continuation_startup(QueueMode::OneAtATime, 2).await;
}

#[tokio::test]
async fn polls_all_steering_at_continuation_startup() {
    polls_steering_at_continuation_startup(QueueMode::All, 1).await;
}

#[tokio::test]
async fn keeps_steering_ahead_of_follow_up_from_a_non_assistant_continuation_tail() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            messages: vec![create_user_message("existing")],
            ..Default::default()
        },
        ..options_with_stream(recording_stream(Arc::clone(&requests)))
    });
    agent.steer(create_user_message("steering"));
    agent.follow_up(create_user_message("follow-up"));

    agent.continue_run().await.unwrap();

    let requests = requests.lock();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].contains(&"steering".to_string()));
    assert!(!requests[0].contains(&"follow-up".to_string()));
    assert!(requests[1].contains(&"follow-up".to_string()));
}

/// Subscribe a listener that steers `message` when the assistant message ends.
fn steer_on_assistant_end(agent: &Agent, message: AgentMessage) -> AgentSubscription {
    let steering_agent = agent.clone();
    agent.subscribe(move |event, _| {
        if matches!(
            event,
            AgentEvent::MessageEnd {
                message: Message::Assistant(_)
            }
        ) {
            steering_agent.steer(message.clone());
        }
        async { Ok(()) }
    })
}

async fn keeps_queues_on_a_failed_response_even_when_finish_turn_requests_continuation(
    stop_reason: StopReason,
) {
    let queued_during_response = create_user_message("steering");
    let follow_up = create_user_message("follow-up");
    let agent = agent_with(AgentOptions {
        finish_turn: Some(Arc::new(|_, _| {
            Box::pin(async { Some(AgentTurnDecision::Continue) })
        })),
        ..options_with_stream(stream_fn(move |_, _, _| {
            finished_stream(AssistantMessage {
                error_message: Some(stop_reason.as_str().to_string()),
                ..create_assistant_message(vec![text(stop_reason.as_str())], stop_reason)
            })
        }))
    });
    agent.follow_up(follow_up.clone());
    let _subscription = steer_on_assistant_end(&agent, queued_during_response.clone());

    agent.prompt_text("start", Vec::new()).await.unwrap();

    assert_eq!(agent.peek_queued_messages(), [queued_during_response]);
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), [follow_up]);
}

#[tokio::test]
async fn keeps_queues_on_a_error_response_even_when_finish_turn_requests_continuation() {
    keeps_queues_on_a_failed_response_even_when_finish_turn_requests_continuation(
        StopReason::Error,
    )
    .await;
}

#[tokio::test]
async fn keeps_queues_on_a_aborted_response_even_when_finish_turn_requests_continuation() {
    keeps_queues_on_a_failed_response_even_when_finish_turn_requests_continuation(
        StopReason::Aborted,
    )
    .await;
}

#[tokio::test]
async fn keeps_queues_when_finish_turn_ends_the_run() {
    let queued_during_response = create_user_message("steering");
    let follow_up = create_user_message("follow-up");
    let agent = agent_with(AgentOptions {
        finish_turn: Some(Arc::new(|_, _| {
            Box::pin(async { Some(AgentTurnDecision::End) })
        })),
        ..options_with_stream(stream_fn(|_, _, _| text_stream("done")))
    });
    agent.follow_up(follow_up.clone());
    let _subscription = steer_on_assistant_end(&agent, queued_during_response.clone());

    agent.prompt_text("start", Vec::new()).await.unwrap();

    assert_eq!(agent.peek_queued_messages(), [queued_during_response]);
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), [follow_up]);
}

#[test]
fn previews_the_next_selected_queued_messages_without_consuming_them() {
    let agent = agent_with(AgentOptions {
        steering_mode: QueueMode::OneAtATime,
        follow_up_mode: QueueMode::All,
        ..options_with_stream(unused_stream_function())
    });
    let first = create_user_message("first steering");
    let second = create_user_message("second steering");
    let follow_up = create_user_message("follow-up");
    agent.steer(first.clone());
    agent.steer(second);
    agent.follow_up(follow_up.clone());

    assert_eq!(agent.peek_queued_messages(), std::slice::from_ref(&first));
    assert_eq!(agent.peek_queued_messages(), [first]);
    agent.clear_steering_queue();
    assert_eq!(agent.peek_queued_messages(), [follow_up]);
}

#[tokio::test]
async fn forwards_provider_stream_event_observers_through_agent_options() {
    let provider_events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let recorded = Arc::clone(&provider_events);
    let agent = agent_with(AgentOptions {
        on_provider_stream_event: Some(Arc::new(move |data: &Value, _model: &Model| {
            recorded.lock().push(data.clone());
            Box::pin(async {})
        })),
        ..options_with_stream(stream_fn(|model, _, options| {
            let stream = AssistantMessageEventStream::new();
            let producer = stream.clone();
            let hook = options.on_provider_stream_event.clone();
            tokio::spawn(async move {
                if let Some(hook) = hook {
                    hook(&json!({ "request_cost": 0.01 }), &model).await;
                }
                let message = create_assistant_message(vec![text("ok")], StopReason::Stop);
                producer.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message,
                });
            });
            stream
        }))
    });

    agent.prompt_text("hello", Vec::new()).await.unwrap();

    assert_eq!(
        provider_events.lock().as_slice(),
        [json!({ "request_cost": 0.01 })]
    );
}

#[tokio::test]
async fn forwards_session_id_to_stream_function_options() {
    let received_session_id = Arc::new(Mutex::new(None));
    let received = Arc::clone(&received_session_id);
    let agent = agent_with(AgentOptions {
        session_id: Some("session-abc".to_string()),
        ..options_with_stream(stream_fn(move |_, _, options| {
            *received.lock() = options.session_id.clone();
            text_stream("ok")
        }))
    });

    agent.prompt_text("hello", Vec::new()).await.unwrap();
    assert_eq!(received_session_id.lock().as_deref(), Some("session-abc"));

    agent.set_session_id(Some("session-def".to_string()));
    assert_eq!(agent.session_id().as_deref(), Some("session-def"));

    agent.prompt_text("hello again", Vec::new()).await.unwrap();
    assert_eq!(received_session_id.lock().as_deref(), Some("session-def"));
}

/// Rust addition: the requested thinking level is stamped on the response.
#[tokio::test]
async fn stamps_the_requested_thinking_level_on_assistant_messages() {
    let agent = agent_with(AgentOptions {
        initial_state: AgentInitialState {
            thinking_level: Some(ModelThinkingLevel::Medium),
            ..Default::default()
        },
        ..options_with_stream(stream_fn(|_, _, _| text_stream("ok")))
    });

    agent.prompt_text("hello", Vec::new()).await.unwrap();

    let Some(Message::Assistant(message)) = agent.messages().last().cloned() else {
        panic!("expected assistant message");
    };
    assert_eq!(message.thinking_level, Some(ModelThinkingLevel::Medium));
}

/// Port of `test/e2e.test.ts`: the agent against the faux provider through
/// a `Models` collection's `stream_simple`.
///
/// Divergence: Pi registers the faux provider in the compat api-registry
/// (`registerFauxProvider`); ai.rs has no global registry, so each test
/// registers it in its own `Models` and passes `stream_simple_fn(models)`.
mod faux_e2e {
    use super::*;
    use crate::models::{Models, create_models};
    use crate::providers::faux::{
        FauxMessageOptions, FauxModelDefinition, FauxProviderHandle, FauxResponseStep,
        FauxTokenSize, RegisterFauxProviderOptions, faux_assistant_message, faux_provider,
        faux_text, faux_thinking, faux_tool_call,
    };
    use crate::stream_simple_fn;

    /// A faux provider registered in its own `Models` collection.
    struct Faux {
        handle: FauxProviderHandle,
        models: Models,
    }

    impl std::ops::Deref for Faux {
        type Target = FauxProviderHandle;

        fn deref(&self) -> &FauxProviderHandle {
            &self.handle
        }
    }

    fn register_faux_provider(options: RegisterFauxProviderOptions) -> Faux {
        let handle = faux_provider(options);
        let models = create_models(Default::default());
        models.set_provider(handle.provider.clone());
        Faux { handle, models }
    }

    /// `test/utils/calculate.ts`: evaluates `a <op> b`.
    fn calculate(expression: &str) -> crate::agent::AgentResult<AgentToolResult> {
        let parts: Vec<_> = expression.split_whitespace().collect();
        let [left, operator, right] = parts.as_slice() else {
            return Err(AgentError::message(format!(
                "Unsupported expression: {expression}"
            )));
        };
        let left: f64 = left
            .parse()
            .map_err(|_| AgentError::message("bad number"))?;
        let right: f64 = right
            .parse()
            .map_err(|_| AgentError::message("bad number"))?;
        let result = match *operator {
            "+" => left + right,
            "-" => left - right,
            "*" => left * right,
            "/" => left / right,
            other => {
                return Err(AgentError::message(format!(
                    "Unsupported operator: {other}"
                )));
            }
        };
        Ok(AgentToolResult::text(format!("{expression} = {result}")))
    }

    fn calculate_tool() -> DynAgentTool {
        AgentToolBuilder::new("calculate")
            .label("Calculator")
            .description("Evaluate mathematical expressions")
            .parameters(json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }))
            .execute(
                |args| async move { calculate(args["expression"].as_str().unwrap_or_default()) },
            )
            .build()
            .unwrap()
    }

    fn text_content(content: &[crate::types::AssistantContent]) -> String {
        content
            .iter()
            .filter_map(|block| match block {
                crate::types::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn tool_result_text(content: &[ToolResultContent]) -> String {
        content
            .iter()
            .filter_map(|block| match block {
                ToolResultContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn faux_agent(faux: &Faux, system_prompt: &str, tools: Vec<DynAgentTool>) -> Agent {
        Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some(system_prompt.to_string()),
                model: Some(faux.get_model()),
                thinking_level: Some(ModelThinkingLevel::Off),
                tools,
                ..Default::default()
            },
            ..options_with_stream(stream_simple_fn(faux.models.clone()))
        })
    }

    fn message(content: impl Into<crate::providers::faux::FauxContent>) -> FauxResponseStep {
        faux_assistant_message(content, FauxMessageOptions::default()).into()
    }

    #[tokio::test]
    async fn handles_a_basic_text_prompt() {
        let faux = register_faux_provider(Default::default());
        faux.set_responses([message("4")]);
        let agent = faux_agent(
            &faux,
            "You are a helpful assistant. Keep your responses concise.",
            Vec::new(),
        );

        agent
            .prompt_text("What is 2+2? Answer with just the number.", Vec::new())
            .await
            .unwrap();

        assert!(!agent.state().is_streaming);
        let messages = agent.messages();
        assert_eq!(roles(&messages), ["system", "user", "assistant"]);
        let Message::Assistant(assistant) = &messages[2] else {
            panic!("Expected assistant message");
        };
        assert!(text_content(&assistant.content).contains('4'));
    }

    #[tokio::test]
    async fn executes_tools_and_tracks_pending_tool_calls() {
        let faux = register_faux_provider(Default::default());
        faux.set_responses([
            faux_assistant_message(
                vec![
                    faux_text("Let me calculate that."),
                    faux_tool_call(
                        "calculate",
                        json!({ "expression": "123 * 456" }),
                        Some("calc-1"),
                    ),
                ],
                FauxMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            )
            .into(),
            message("The result is 56088."),
        ]);
        let agent = faux_agent(
            &faux,
            "You are a helpful assistant. Always use the calculator tool for math.",
            vec![calculate_tool()],
        );
        let pending_during_events = Arc::new(Mutex::new(Vec::new()));
        let _subscription = {
            let observer = agent.clone();
            let pending = Arc::clone(&pending_during_events);
            agent.subscribe(move |event, _| {
                if matches!(
                    event,
                    AgentEvent::ToolExecutionStart { .. } | AgentEvent::ToolExecutionEnd { .. }
                ) {
                    let mut ids: Vec<_> = observer.state().pending_tool_calls.into_iter().collect();
                    ids.sort();
                    pending.lock().push((event.event_type(), ids));
                }
                async { Ok(()) }
            })
        };

        agent
            .prompt_text("Calculate 123 * 456 using the calculator tool.", Vec::new())
            .await
            .unwrap();

        let state = agent.state();
        assert!(!state.is_streaming);
        assert!(state.messages.len() >= 4);
        let tool_result = state
            .messages
            .iter()
            .find_map(|message| match message {
                Message::ToolResult(result) => Some(result.clone()),
                _ => None,
            })
            .expect("Expected tool result message");
        assert!(tool_result_text(&tool_result.content).contains("123 * 456 = 56088"));
        let Some(Message::Assistant(final_message)) = state.messages.last() else {
            panic!("Expected final assistant message");
        };
        assert!(text_content(&final_message.content).contains("56088"));
        assert!(state.pending_tool_calls.is_empty());
        assert_eq!(
            pending_during_events.lock().as_slice(),
            [
                ("tool_execution_start", vec!["calc-1".to_string()]),
                ("tool_execution_end", Vec::new()),
            ]
        );
    }

    #[tokio::test]
    async fn handles_abort_during_streaming() {
        let faux = register_faux_provider(RegisterFauxProviderOptions {
            tokens_per_second: Some(20.0),
            token_size: Some(FauxTokenSize {
                min: Some(2),
                max: Some(2),
            }),
            ..Default::default()
        });
        faux.set_responses([message(
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen",
        )]);
        let agent = faux_agent(&faux, "You are a helpful assistant.", Vec::new());

        let aborter = agent.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            aborter.abort();
        });
        agent
            .prompt_text("Count slowly from 1 to 20.", Vec::new())
            .await
            .unwrap();

        let state = agent.state();
        assert!(!state.is_streaming);
        assert!(state.messages.len() >= 2);
        let Some(Message::Assistant(last_message)) = state.messages.last() else {
            panic!("Expected assistant message");
        };
        assert_eq!(last_message.stop_reason, StopReason::Aborted);
        assert!(last_message.error_message.is_some());
        assert_eq!(state.error_message, last_message.error_message);
    }

    #[tokio::test]
    async fn emits_lifecycle_updates_while_streaming() {
        let faux = register_faux_provider(RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            ..Default::default()
        });
        faux.set_responses([message("1 2 3 4 5")]);
        let agent = faux_agent(&faux, "You are a helpful assistant.", Vec::new());
        let events = Arc::new(Mutex::new(Vec::new()));
        let _subscription = {
            let events = Arc::clone(&events);
            agent.subscribe(move |event, _| {
                events.lock().push(event.event_type());
                async { Ok(()) }
            })
        };

        agent
            .prompt_text("Count from 1 to 5.", Vec::new())
            .await
            .unwrap();

        let events = events.lock();
        for expected in [
            "agent_start",
            "turn_start",
            "message_start",
            "message_update",
            "message_end",
            "turn_end",
            "agent_end",
        ] {
            assert!(events.contains(&expected), "missing {expected}");
        }
        let first = |name: &str| events.iter().position(|event| *event == name).unwrap();
        let last = |name: &str| events.iter().rposition(|event| *event == name).unwrap();
        assert!(first("agent_start") < first("message_start"));
        assert!(first("message_start") < first("message_end"));
        assert!(first("message_end") < last("agent_end"));
        assert!(!agent.state().is_streaming);
        assert_eq!(agent.messages().len(), 3);
    }

    #[tokio::test]
    async fn maintains_context_across_multiple_turns() {
        let faux = register_faux_provider(Default::default());
        faux.set_responses([
            message("Nice to meet you, Alice."),
            FauxResponseStep::factory(|context, _, _, _| {
                let has_alice = context.messages.iter().any(|message| match message {
                    Message::User(user) => match &user.content {
                        UserMessageContent::Text(text) => text.contains("Alice"),
                        UserMessageContent::Parts(parts) => parts.iter().any(|part| {
                            matches!(part, crate::types::UserContent::Text(text) if text.text.contains("Alice"))
                        }),
                    },
                    _ => false,
                });
                Ok(faux_assistant_message(
                    if has_alice {
                        "Your name is Alice."
                    } else {
                        "I do not know your name."
                    },
                    FauxMessageOptions::default(),
                ))
            }),
        ]);
        let agent = faux_agent(&faux, "You are a helpful assistant.", Vec::new());

        agent
            .prompt_text("My name is Alice.", Vec::new())
            .await
            .unwrap();
        assert_eq!(agent.messages().len(), 3);
        agent
            .prompt_text("What is my name?", Vec::new())
            .await
            .unwrap();
        let messages = agent.messages();
        assert_eq!(messages.len(), 5);
        let Message::Assistant(last_message) = &messages[4] else {
            panic!("Expected assistant message");
        };
        assert!(
            text_content(&last_message.content)
                .to_lowercase()
                .contains("alice")
        );
    }

    #[tokio::test]
    async fn preserves_thinking_content_blocks() {
        let faux = register_faux_provider(RegisterFauxProviderOptions {
            models: vec![FauxModelDefinition {
                reasoning: Some(true),
                ..FauxModelDefinition::new("faux-reasoning")
            }],
            ..Default::default()
        });
        faux.set_responses([faux_assistant_message(
            vec![faux_thinking("step by step"), faux_text("4")],
            FauxMessageOptions::default(),
        )
        .into()]);
        let agent = faux_agent(&faux, "You are a helpful assistant.", Vec::new());
        agent.set_thinking_level(ModelThinkingLevel::Low);

        agent.prompt_text("What is 2+2?", Vec::new()).await.unwrap();

        let Some(Message::Assistant(assistant)) = agent.messages().get(2).cloned() else {
            panic!("Expected assistant message");
        };
        assert_eq!(
            assistant.content,
            [faux_thinking("step by step"), faux_text("4")]
        );
    }

    #[tokio::test]
    async fn continue_throws_when_no_messages_in_context() {
        let faux = register_faux_provider(Default::default());
        let agent = Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("Test".to_string()),
                model: Some(faux.get_model()),
                ..Default::default()
            },
            ..options_with_stream(stream_simple_fn(faux.models.clone()))
        });

        let error = agent.continue_run().await.unwrap_err();
        assert_eq!(error.to_string(), "No messages to continue from");
    }

    #[tokio::test]
    async fn continue_throws_when_last_message_is_assistant() {
        let faux = register_faux_provider(Default::default());
        let model = faux.get_model();
        let agent = Agent::new(AgentOptions {
            initial_state: AgentInitialState {
                system_prompt: Some("Test".to_string()),
                model: Some(model.clone()),
                ..Default::default()
            },
            ..options_with_stream(stream_simple_fn(faux.models.clone()))
        });
        let mut assistant_message = AssistantMessage::empty_for(&model);
        assistant_message.content = vec![text("Hello")];
        agent.set_messages(vec![Message::Assistant(assistant_message)]);

        let error = agent.continue_run().await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Cannot continue from message role: assistant"
        );
    }

    #[tokio::test]
    async fn continues_and_gets_a_response_when_last_message_is_user() {
        let faux = register_faux_provider(Default::default());
        faux.set_responses([message("HELLO WORLD")]);
        let agent = faux_agent(
            &faux,
            "You are a helpful assistant. Follow instructions exactly.",
            Vec::new(),
        );
        agent.set_messages(vec![Message::User(crate::types::UserMessage {
            content: UserMessageContent::Parts(vec![crate::types::UserContent::text(
                "Say exactly: HELLO WORLD",
            )]),
            timestamp: now_millis(),
        })]);

        agent.continue_run().await.unwrap();

        assert!(!agent.state().is_streaming);
        let messages = agent.messages();
        assert_eq!(roles(&messages), ["user", "assistant"]);
        let Message::Assistant(assistant) = &messages[1] else {
            panic!("Expected assistant message");
        };
        assert!(
            text_content(&assistant.content)
                .to_uppercase()
                .contains("HELLO WORLD")
        );
    }

    #[tokio::test]
    async fn continues_and_processes_tool_results() {
        let faux = register_faux_provider(Default::default());
        let model = faux.get_model();
        faux.set_responses([message("The answer is 8.")]);
        let agent = faux_agent(
            &faux,
            "You are a helpful assistant. After getting a calculation result, state the answer clearly.",
            vec![calculate_tool()],
        );
        let mut assistant_message = AssistantMessage::empty_for(&model);
        assistant_message.content = vec![
            text("Let me calculate that."),
            tool_call("calc-1", "calculate", json!({ "expression": "5 + 3" })),
        ];
        assistant_message.stop_reason = StopReason::ToolUse;
        agent.set_messages(vec![
            Message::User(crate::types::UserMessage {
                content: UserMessageContent::Parts(vec![crate::types::UserContent::text(
                    "What is 5 + 3?",
                )]),
                timestamp: now_millis(),
            }),
            Message::Assistant(assistant_message),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "calc-1".to_string(),
                tool_name: "calculate".to_string(),
                content: vec![ToolResultContent::text("5 + 3 = 8")],
                details: None,
                usage: None,
                nested_calls: None,
                is_error: false,
                timestamp: now_millis(),
            }),
        ]);

        agent.continue_run().await.unwrap();

        assert!(!agent.state().is_streaming);
        let messages = agent.messages();
        assert!(messages.len() >= 4);
        let Some(Message::Assistant(last_message)) = messages.last() else {
            panic!("Expected assistant message");
        };
        assert!(text_content(&last_message.content).contains('8'));
    }
}
