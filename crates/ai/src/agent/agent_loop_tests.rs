//! Port of `packages/agent/test/agent-loop.test.ts`.
//!
//! Pi's `MockAssistantStream` is an [`AssistantMessageEventStream`] the
//! test fills directly. The two tests built on custom (non-LLM) message
//! roles use system messages instead, since [`AgentMessage`] has no custom
//! roles in Rust.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::agent::stream_fn::{
    DEFAULT_STREAM_FN_TEST_LOCK, set_default_stream_fn, stream_fn as make_stream_fn,
};
use crate::agent::types::{
    AfterToolCallResult, AgentLoopTurnUpdate, AgentRequestUpdate, AgentToolBuilder,
    BeforeToolCallResult, MessageQueueFn,
};
use crate::types::{
    AssistantMessageEventStream, ModelCost, ModelInput, ThinkingLevel, TranscriptContext, Usage,
    UsageCost, UserMessage, UserMessageContent,
};

pub(crate) fn create_usage() -> Usage {
    Usage::default()
}

pub(crate) fn create_model() -> crate::types::Model {
    crate::types::Model {
        id: "mock".to_string(),
        name: "mock".to_string(),
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        base_url: "https://example.invalid".to_string(),
        reasoning: false,
        input: vec![ModelInput::Text],
        cost: ModelCost::default(),
        context_window: 8192,
        max_tokens: 2048,
        ..Default::default()
    }
}

pub(crate) fn create_assistant_message(
    content: Vec<crate::types::AssistantContent>,
    stop_reason: StopReason,
) -> AssistantMessage {
    AssistantMessage {
        content,
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        model: "mock".to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        diagnostics: None,
        usage: create_usage(),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_millis(),
    }
}

pub(crate) fn text(text: &str) -> crate::types::AssistantContent {
    crate::types::AssistantContent::text(text)
}

pub(crate) fn tool_call(id: &str, name: &str, arguments: Value) -> crate::types::AssistantContent {
    crate::types::AssistantContent::ToolCall(AgentToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
        thought_signature: None,
        namespace: None,
    })
}

pub(crate) fn create_user_message(text: &str) -> AgentMessage {
    Message::User(UserMessage {
        content: UserMessageContent::Text(text.to_string()),
        timestamp: now_millis(),
    })
}

/// A stream that ends with `done` (or `error` for error/aborted messages).
pub(crate) fn finished_stream(message: AssistantMessage) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let reason = message.stop_reason;
    if matches!(reason, StopReason::Error | StopReason::Aborted) {
        stream.push(AssistantMessageEvent::Error {
            reason,
            error: message,
        });
    } else {
        stream.push(AssistantMessageEvent::Done { reason, message });
    }
    stream
}

pub(crate) fn text_stream(response: &str) -> AssistantMessageEventStream {
    finished_stream(create_assistant_message(
        vec![text(response)],
        StopReason::Stop,
    ))
}

pub(crate) fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::User(UserMessage {
                content: UserMessageContent::Text(text),
                ..
            }) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

pub(crate) fn roles(messages: &[Message]) -> Vec<&'static str> {
    messages.iter().map(Message::role).collect()
}

pub(crate) fn value_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "value": { "type": "string" } },
        "required": ["value"]
    })
}

pub(crate) fn empty_schema() -> Value {
    json!({ "type": "object", "properties": {} })
}

fn arg_value(args: &Value) -> String {
    args["value"].as_str().unwrap_or_default().to_string()
}

pub(crate) fn text_result(text: impl Into<String>, details: Value) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContent::text(text)],
        details: Some(details),
        ..Default::default()
    }
}

/// An `echo` tool recording the `value` argument of each call.
fn echo_tool(executed: Arc<Mutex<Vec<String>>>) -> DynAgentTool {
    AgentToolBuilder::new("echo")
        .label("Echo")
        .description("Echo tool")
        .parameters(value_schema())
        .execute(move |args| {
            let executed = Arc::clone(&executed);
            async move {
                let value = arg_value(&args);
                executed.lock().push(value.clone());
                Ok(text_result(
                    format!("echoed: {value}"),
                    json!({ "value": value }),
                ))
            }
        })
        .build()
        .unwrap()
}

fn noop_tool() -> DynAgentTool {
    AgentToolBuilder::new("noop")
        .label("Noop")
        .description("Noop tool")
        .parameters(empty_schema())
        .execute(|_| async { Ok(AgentToolResult::text("done")) })
        .build()
        .unwrap()
}

fn config() -> AgentLoopConfig {
    AgentLoopConfig::new(create_model())
}

fn context(tools: Vec<DynAgentTool>) -> AgentContext {
    AgentContext {
        messages: Vec::new(),
        tools,
    }
}

/// A stream function answering with `respond(call_index, context)`.
fn scripted<F>(respond: F) -> (StreamFn, Arc<AtomicUsize>)
where
    F: Fn(usize, &TranscriptContext) -> AssistantMessage + Send + Sync + 'static,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let stream_fn = make_stream_fn(move |_model, context, _options| {
        let index = counter.fetch_add(1, Ordering::SeqCst);
        finished_stream(respond(index, &context))
    });
    (stream_fn, calls)
}

/// First request returns `first`, later requests a final "done" text.
fn tool_then_done(first: Vec<crate::types::AssistantContent>) -> (StreamFn, Arc<AtomicUsize>) {
    scripted(move |index, _| {
        if index == 0 {
            create_assistant_message(first.clone(), StopReason::ToolUse)
        } else {
            create_assistant_message(vec![text("done")], StopReason::Stop)
        }
    })
}

async fn collect(mut stream: AgentEventStream) -> (Vec<AgentEvent>, Vec<AgentMessage>) {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    let messages = stream.result().await.expect("loop result");
    (events, messages)
}

fn event_types(events: &[AgentEvent]) -> Vec<&'static str> {
    events.iter().map(AgentEvent::event_type).collect()
}

fn sink(events: Arc<Mutex<Vec<AgentEvent>>>) -> AgentEventSink {
    Arc::new(move |event| {
        events.lock().push(event);
        Box::pin(async { Ok(()) })
    })
}

fn queue(messages: impl Fn() -> Vec<AgentMessage> + Send + Sync + 'static) -> MessageQueueFn {
    Arc::new(move || {
        let messages = messages();
        Box::pin(async move { messages })
    })
}

fn finish_turn(
    decide: impl Fn(&AgentTurnContext) -> Option<AgentTurnDecision> + Send + Sync + 'static,
) -> crate::agent::types::FinishTurnFn {
    Arc::new(move |turn, _signal| {
        let decision = decide(&turn);
        Box::pin(async move { Ok(decision) })
    })
}

mod default_stream_function_compatibility {
    use super::*;

    #[tokio::test]
    async fn uses_the_configured_default_when_a_legacy_caller_omits_stream_fn() {
        let _lock = DEFAULT_STREAM_FN_TEST_LOCK.lock().await;
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        set_default_stream_fn(Some(make_stream_fn(move |_, _, _| {
            counter.fetch_add(1, Ordering::SeqCst);
            text_stream("fallback")
        })));

        let mut stream = agent_loop(
            vec![create_user_message("Hello")],
            context(Vec::new()),
            config(),
            None,
            None,
        );
        let result = stream.result().await;
        set_default_stream_fn(None);

        result.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

mod agent_loop_with_agent_message {
    use super::*;

    #[tokio::test]
    async fn should_emit_events_with_agent_message_types() {
        let (stream_fn, _) =
            scripted(|_, _| create_assistant_message(vec![text("Hi there!")], StopReason::Stop));
        let stream = agent_loop(
            vec![create_user_message("Hello")],
            context(Vec::new()),
            config(),
            None,
            Some(stream_fn),
        );
        let (events, messages) = collect(stream).await;

        assert_eq!(roles(&messages), ["user", "assistant"]);
        let types = event_types(&events);
        for expected in [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ] {
            assert!(types.contains(&expected), "missing {expected}");
        }
    }

    #[tokio::test]
    async fn should_build_provider_context_exclusively_from_transcript_messages() {
        let initial_system = Message::System(SystemMessage {
            content: SystemMessageContent::Text("Transcript prompt".to_string()),
            tools_added: Some(Vec::new()),
            timestamp: 1,
            ..Default::default()
        });
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_in_fn = Arc::clone(&seen);
        let stream_fn = make_stream_fn(move |_, provider_context, _| {
            // The provider receives a transcript: no top-level prompt or tool fields.
            seen_in_fn.lock().push(provider_context.messages[0].clone());
            text_stream("done")
        });
        let mut stream = agent_loop(
            vec![initial_system.clone(), create_user_message("Hello")],
            context(Vec::new()),
            config(),
            None,
            Some(stream_fn),
        );
        stream.result().await.unwrap();

        assert_eq!(seen.lock().as_slice(), [initial_system]);
    }

    /// Pi passes a custom `notification` message and filters it in
    /// `convertToLlm`; a system message stands in for it here.
    #[tokio::test]
    async fn should_filter_messages_via_convert_to_llm() {
        let notification = Message::System(SystemMessage {
            content: SystemMessageContent::Text("This is a notification".to_string()),
            timestamp: now_millis(),
            ..Default::default()
        });
        let converted = Arc::new(Mutex::new(Vec::new()));
        let mut config = config();
        let converted_in_fn = Arc::clone(&converted);
        config.convert_to_llm = Arc::new(move |messages: Vec<AgentMessage>| {
            let filtered: Vec<Message> = messages
                .into_iter()
                .filter(|message| !matches!(message, Message::System(_)))
                .collect();
            *converted_in_fn.lock() = filtered.clone();
            Box::pin(async move { filtered })
        });
        let (stream_fn, _) =
            scripted(|_, _| create_assistant_message(vec![text("Response")], StopReason::Stop));

        let stream = agent_loop(
            vec![create_user_message("Hello")],
            AgentContext {
                messages: vec![notification],
                tools: Vec::new(),
            },
            config,
            None,
            Some(stream_fn),
        );
        collect(stream).await;

        // The notification should have been filtered out in convert_to_llm.
        assert_eq!(roles(&converted.lock()), ["user"]);
    }

    #[tokio::test]
    async fn should_apply_transform_context_before_convert_to_llm() {
        let transformed = Arc::new(Mutex::new(Vec::new()));
        let converted = Arc::new(Mutex::new(Vec::new()));
        let mut config = config();
        let transformed_in_fn = Arc::clone(&transformed);
        config.transform_context = Some(Arc::new(move |messages: Vec<AgentMessage>, _| {
            // Keep only the last 2 messages (prune old ones).
            let kept = messages[messages.len() - 2..].to_vec();
            *transformed_in_fn.lock() = kept.clone();
            Box::pin(async move { kept })
        }));
        let converted_in_fn = Arc::clone(&converted);
        config.convert_to_llm = Arc::new(move |messages: Vec<AgentMessage>| {
            *converted_in_fn.lock() = messages.clone();
            Box::pin(async move { messages })
        });
        let (stream_fn, _) =
            scripted(|_, _| create_assistant_message(vec![text("Response")], StopReason::Stop));

        let stream = agent_loop(
            vec![create_user_message("new message")],
            AgentContext {
                messages: vec![
                    create_user_message("old message 1"),
                    Message::Assistant(create_assistant_message(
                        vec![text("old response 1")],
                        StopReason::Stop,
                    )),
                    create_user_message("old message 2"),
                    Message::Assistant(create_assistant_message(
                        vec![text("old response 2")],
                        StopReason::Stop,
                    )),
                ],
                tools: Vec::new(),
            },
            config,
            None,
            Some(stream_fn),
        );
        collect(stream).await;

        assert_eq!(transformed.lock().len(), 2);
        assert_eq!(converted.lock().len(), 2);
    }

    #[tokio::test]
    async fn should_handle_tool_calls_and_results() {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let tool_usage = Usage {
            input: 1,
            output: 2,
            cache_read: 3,
            cache_write: 4,
            total_tokens: 10,
            cost: UsageCost {
                input: 0.1,
                output: 0.2,
                cache_read: 0.3,
                cache_write: 0.4,
                total: 1.0,
            },
            ..Default::default()
        };
        let patched_tool_usage = Usage {
            input: 5,
            output: 6,
            cache_read: 7,
            cache_write: 8,
            total_tokens: 26,
            cost: UsageCost {
                input: 0.5,
                output: 0.6,
                cache_read: 0.7,
                cache_write: 0.8,
                total: 2.6,
            },
            ..Default::default()
        };
        let observed_tool_usage = Arc::new(Mutex::new(None));
        let tool = {
            let executed = Arc::clone(&executed);
            let tool_usage = tool_usage.clone();
            AgentToolBuilder::new("echo")
                .label("Echo")
                .description("Echo tool")
                .parameters(value_schema())
                .execute(move |args| {
                    let executed = Arc::clone(&executed);
                    let tool_usage = tool_usage.clone();
                    async move {
                        let value = arg_value(&args);
                        executed.lock().push(value.clone());
                        Ok(AgentToolResult {
                            usage: Some(tool_usage),
                            ..text_result(format!("echoed: {value}"), json!({ "value": value }))
                        })
                    }
                })
                .build()
                .unwrap()
        };
        let mut config = config();
        let observed = Arc::clone(&observed_tool_usage);
        let patched = patched_tool_usage.clone();
        config.after_tool_call = Some(Arc::new(move |context: AfterToolCallContext, _| {
            *observed.lock() = context.result.usage.clone();
            let patched = patched.clone();
            Box::pin(async move {
                Ok(Some(AfterToolCallResult {
                    usage: Some(patched),
                    ..Default::default()
                }))
            })
        }));
        let (stream_fn, _) = tool_then_done(vec![tool_call(
            "tool-1",
            "echo",
            json!({ "value": "hello" }),
        )]);

        let stream = agent_loop(
            vec![create_user_message("echo something")],
            context(vec![tool]),
            config,
            None,
            Some(stream_fn),
        );
        let (events, messages) = collect(stream).await;

        assert_eq!(executed.lock().as_slice(), ["hello"]);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolExecutionStart { .. }))
        );
        let tool_end = events
            .iter()
            .find_map(|event| match event {
                AgentEvent::ToolExecutionEnd { is_error, .. } => Some(*is_error),
                _ => None,
            })
            .expect("tool_execution_end");
        assert!(!tool_end);
        assert_eq!(observed_tool_usage.lock().clone(), Some(tool_usage));
        let tool_result = messages
            .iter()
            .find_map(|message| match message {
                Message::ToolResult(result) => Some(result.clone()),
                _ => None,
            })
            .expect("tool result");
        assert_eq!(tool_result.usage, Some(patched_tool_usage));
    }

    #[tokio::test]
    async fn should_not_execute_tool_calls_from_a_length_truncated_assistant_message() {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let (stream_fn, calls) = scripted(|index, _| {
            if index == 0 {
                // Output hit the token limit mid tool call. The salvage parser can
                // produce arguments that validate but are silently truncated, so
                // nothing in this message may execute.
                create_assistant_message(
                    vec![tool_call("tool-1", "echo", json!({ "value": "hel" }))],
                    StopReason::Length,
                )
            } else {
                create_assistant_message(vec![text("done")], StopReason::Stop)
            }
        });

        let stream = agent_loop(
            vec![create_user_message("echo something")],
            context(vec![echo_tool(Arc::clone(&executed))]),
            config(),
            None,
            Some(stream_fn),
        );
        let (events, messages) = collect(stream).await;

        // The tool must never execute with potentially truncated arguments.
        assert!(executed.lock().is_empty());
        let (result, is_error) = events
            .iter()
            .find_map(|event| match event {
                AgentEvent::ToolExecutionEnd {
                    result, is_error, ..
                } => Some((result.clone(), *is_error)),
                _ => None,
            })
            .expect("tool_execution_end");
        assert!(is_error);
        let ToolResultContent::Text(text) = &result.content[0] else {
            panic!("expected text");
        };
        assert!(text.text.contains("output token limit"));

        // The loop continues so the model can re-issue the tool call.
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(messages.last().map(Message::role), Some("assistant"));
    }

    #[tokio::test]
    async fn should_execute_mutated_before_tool_call_args_without_revalidation() {
        let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
        let tool = {
            let executed = Arc::clone(&executed);
            AgentToolBuilder::new("echo")
                .label("Echo")
                .description("Echo tool")
                .parameters(value_schema())
                .execute(move |args| {
                    let executed = Arc::clone(&executed);
                    async move {
                        executed.lock().push(args["value"].clone());
                        Ok(text_result(
                            format!("echoed: {}", args["value"]),
                            json!({ "value": args["value"] }),
                        ))
                    }
                })
                .build()
                .unwrap()
        };
        let mut config = config();
        config.before_tool_call = Some(Arc::new(|context: BeforeToolCallContext, _| {
            context.args.lock()["value"] = json!(123);
            Box::pin(async { Ok(None) })
        }));
        let (stream_fn, _) = tool_then_done(vec![tool_call(
            "tool-1",
            "echo",
            json!({ "value": "hello" }),
        )]);

        let stream = agent_loop(
            vec![create_user_message("echo something")],
            context(vec![tool]),
            config,
            None,
            Some(stream_fn),
        );
        collect(stream).await;

        assert_eq!(executed.lock().as_slice(), [json!(123)]);
    }

    #[tokio::test]
    async fn should_prepare_tool_arguments_for_validation() {
        let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
        let tool = {
            let executed = Arc::clone(&executed);
            AgentToolBuilder::new("edit")
                .label("Edit")
                .description("Edit tool")
                .parameters(json!({
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
                }))
                .prepare_arguments(|args| {
                    let (Some(old_text), Some(new_text)) = (
                        args.get("oldText").and_then(Value::as_str),
                        args.get("newText").and_then(Value::as_str),
                    ) else {
                        return Ok(args);
                    };
                    let mut edits = args
                        .get("edits")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    edits.push(json!({ "oldText": old_text, "newText": new_text }));
                    Ok(json!({ "edits": edits }))
                })
                .execute(move |args| {
                    let executed = Arc::clone(&executed);
                    async move {
                        let edits = args["edits"].clone();
                        let count = edits.as_array().map_or(0, Vec::len);
                        executed.lock().push(edits);
                        Ok(text_result(
                            format!("edited {count}"),
                            json!({ "count": count }),
                        ))
                    }
                })
                .build()
                .unwrap()
        };
        let (stream_fn, _) = tool_then_done(vec![tool_call(
            "tool-1",
            "edit",
            json!({ "oldText": "before", "newText": "after" }),
        )]);

        let stream = agent_loop(
            vec![create_user_message("edit something")],
            context(vec![tool]),
            config(),
            None,
            Some(stream_fn),
        );
        collect(stream).await;

        assert_eq!(
            executed.lock().as_slice(),
            [json!([{ "oldText": "before", "newText": "after" }])]
        );
    }

    /// A tool whose `first` call waits for `release`; records whether
    /// `second` ran before `first` finished.
    fn gated_tool(
        name: &str,
        release: Arc<Notify>,
        parallel_observed: Arc<AtomicBool>,
        execution_mode: Option<ToolExecutionMode>,
    ) -> DynAgentTool {
        let first_resolved = Arc::new(AtomicBool::new(false));
        let mut builder = AgentToolBuilder::new(name)
            .description(format!("{name} tool"))
            .parameters(value_schema());
        if let Some(execution_mode) = execution_mode {
            builder = builder.execution_mode(execution_mode);
        }
        builder
            .execute(move |args| {
                let release = Arc::clone(&release);
                let first_resolved = Arc::clone(&first_resolved);
                let parallel_observed = Arc::clone(&parallel_observed);
                async move {
                    let value = arg_value(&args);
                    if value == "first" {
                        release.notified().await;
                        first_resolved.store(true, Ordering::SeqCst);
                    }
                    if value == "second" && !first_resolved.load(Ordering::SeqCst) {
                        parallel_observed.store(true, Ordering::SeqCst);
                    }
                    Ok(text_result(
                        format!("echoed: {value}"),
                        json!({ "value": value }),
                    ))
                }
            })
            .build()
            .unwrap()
    }

    /// Two tool calls on the first request, releasing `release` 20ms later.
    fn two_calls_stream(name: &'static str, release: Arc<Notify>) -> StreamFn {
        scripted(move |index, _| {
            if index == 0 {
                let release = Arc::clone(&release);
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    release.notify_one();
                });
                create_assistant_message(
                    vec![
                        tool_call("tool-1", name, json!({ "value": "first" })),
                        tool_call("tool-2", name, json!({ "value": "second" })),
                    ],
                    StopReason::ToolUse,
                )
            } else {
                create_assistant_message(vec![text("done")], StopReason::Stop)
            }
        })
        .0
    }

    fn tool_end_ids(events: &[AgentEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
                _ => None,
            })
            .collect()
    }

    fn tool_result_ids(events: &[AgentEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::MessageEnd {
                    message: Message::ToolResult(result),
                } => Some(result.tool_call_id.clone()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn should_emit_tool_execution_end_in_completion_order_but_persist_tool_results_in_source_order()
     {
        let release = Arc::new(Notify::new());
        let parallel_observed = Arc::new(AtomicBool::new(false));
        let tool = gated_tool(
            "echo",
            Arc::clone(&release),
            Arc::clone(&parallel_observed),
            None,
        );
        let mut config = config();
        config.tool_execution = ToolExecutionMode::Parallel;

        let stream = agent_loop(
            vec![create_user_message("echo both")],
            context(vec![tool]),
            config,
            None,
            Some(two_calls_stream("echo", release)),
        );
        let (events, _) = collect(stream).await;

        let turn_tool_result_ids: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::TurnEnd { tool_results, .. } => Some(tool_results),
                _ => None,
            })
            .flatten()
            .map(|result| result.tool_call_id.clone())
            .collect();
        assert!(parallel_observed.load(Ordering::SeqCst));
        assert_eq!(tool_end_ids(&events), ["tool-2", "tool-1"]);
        assert_eq!(tool_result_ids(&events), ["tool-1", "tool-2"]);
        assert_eq!(turn_tool_result_ids, ["tool-1", "tool-2"]);
    }

    #[tokio::test]
    async fn should_inject_queued_messages_after_all_tool_calls_complete() {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let queued_delivered = Arc::new(AtomicBool::new(false));
        let saw_interrupt_in_context = Arc::new(AtomicBool::new(false));
        let mut config = config();
        config.tool_execution = ToolExecutionMode::Sequential;
        {
            let executed = Arc::clone(&executed);
            let queued_delivered = Arc::clone(&queued_delivered);
            config.get_steering_messages = Some(queue(move || {
                // Return a steering message after tool execution has started.
                if !executed.lock().is_empty() && !queued_delivered.swap(true, Ordering::SeqCst) {
                    vec![create_user_message("interrupt")]
                } else {
                    Vec::new()
                }
            }));
        }
        let saw = Arc::clone(&saw_interrupt_in_context);
        let (stream_fn, _) = scripted(move |index, context| {
            if index == 1 {
                saw.store(
                    user_texts(&context.messages).contains(&"interrupt".to_string()),
                    Ordering::SeqCst,
                );
            }
            if index == 0 {
                create_assistant_message(
                    vec![
                        tool_call("tool-1", "echo", json!({ "value": "first" })),
                        tool_call("tool-2", "echo", json!({ "value": "second" })),
                    ],
                    StopReason::ToolUse,
                )
            } else {
                create_assistant_message(vec![text("done")], StopReason::Stop)
            }
        });

        let stream = agent_loop(
            vec![create_user_message("start")],
            context(vec![echo_tool(Arc::clone(&executed))]),
            config,
            None,
            Some(stream_fn),
        );
        let (events, _) = collect(stream).await;

        // Both tools should execute before steering is injected.
        assert_eq!(executed.lock().as_slice(), ["first", "second"]);
        let tool_errors: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolExecutionEnd { is_error, .. } => Some(*is_error),
                _ => None,
            })
            .collect();
        assert_eq!(tool_errors, [false, false]);

        // The queued message should appear after both tool result messages.
        let sequence: Vec<String> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::MessageStart {
                    message: Message::ToolResult(result),
                } => Some(format!("tool:{}", result.tool_call_id)),
                AgentEvent::MessageStart { message } => {
                    user_texts(std::slice::from_ref(message)).pop()
                }
                _ => None,
            })
            .collect();
        let position = |entry: &str| sequence.iter().position(|value| value == entry);
        assert!(position("interrupt").is_some());
        assert!(position("tool:tool-1") < position("interrupt"));
        assert!(position("tool:tool-2") < position("interrupt"));

        // The interrupt message should be in context when the second LLM call is made.
        assert!(saw_interrupt_in_context.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn should_force_sequential_execution_when_a_tool_has_execution_mode_sequential_even_with_default_parallel_config()
     {
        let release = Arc::new(Notify::new());
        let parallel_observed = Arc::new(AtomicBool::new(false));
        let tool = gated_tool(
            "slow",
            Arc::clone(&release),
            Arc::clone(&parallel_observed),
            Some(ToolExecutionMode::Sequential),
        );

        let stream = agent_loop(
            vec![create_user_message("run both")],
            context(vec![tool]),
            config(),
            None,
            Some(two_calls_stream("slow", release)),
        );
        let (events, _) = collect(stream).await;

        // With sequential execution, the second tool should not start before the first finishes.
        assert!(!parallel_observed.load(Ordering::SeqCst));
        assert_eq!(tool_result_ids(&events), ["tool-1", "tool-2"]);
    }

    #[tokio::test]
    async fn should_force_sequential_execution_when_one_of_multiple_tools_has_execution_mode_sequential()
     {
        let execution_order = Arc::new(Mutex::new(Vec::new()));
        let release = Arc::new(Notify::new());
        let slow_tool = {
            let execution_order = Arc::clone(&execution_order);
            let release = Arc::clone(&release);
            AgentToolBuilder::new("slow")
                .description("Slow tool")
                .parameters(value_schema())
                .execution_mode(ToolExecutionMode::Sequential)
                .execute(move |args| {
                    let execution_order = Arc::clone(&execution_order);
                    let release = Arc::clone(&release);
                    async move {
                        let value = arg_value(&args);
                        execution_order.lock().push(format!("slow:{value}"));
                        if value == "a" {
                            release.notified().await;
                        }
                        Ok(text_result(
                            format!("slow: {value}"),
                            json!({ "value": value }),
                        ))
                    }
                })
                .build()
                .unwrap()
        };
        let fast_tool = {
            let execution_order = Arc::clone(&execution_order);
            AgentToolBuilder::new("fast")
                .description("Fast tool")
                .parameters(value_schema())
                .execute(move |args| {
                    let execution_order = Arc::clone(&execution_order);
                    async move {
                        let value = arg_value(&args);
                        execution_order.lock().push(format!("fast:{value}"));
                        Ok(text_result(
                            format!("fast: {value}"),
                            json!({ "value": value }),
                        ))
                    }
                })
                .build()
                .unwrap()
        };
        let (stream_fn, _) = scripted(move |index, _| {
            if index == 0 {
                let release = Arc::clone(&release);
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    release.notify_one();
                });
                create_assistant_message(
                    vec![
                        tool_call("tool-1", "slow", json!({ "value": "a" })),
                        tool_call("tool-2", "fast", json!({ "value": "b" })),
                    ],
                    StopReason::ToolUse,
                )
            } else {
                create_assistant_message(vec![text("done")], StopReason::Stop)
            }
        });

        let stream = agent_loop(
            vec![create_user_message("run both")],
            context(vec![slow_tool, fast_tool]),
            config(),
            None,
            Some(stream_fn),
        );
        collect(stream).await;

        // The fast tool should not run before the slow tool finishes.
        let order = execution_order.lock().clone();
        assert_eq!(order[0], "slow:a");
        assert!(order.contains(&"fast:b".to_string()));
    }

    #[tokio::test]
    async fn should_allow_parallel_execution_when_all_tools_have_execution_mode_parallel() {
        let release = Arc::new(Notify::new());
        let parallel_observed = Arc::new(AtomicBool::new(false));
        let tool = gated_tool(
            "echo",
            Arc::clone(&release),
            Arc::clone(&parallel_observed),
            Some(ToolExecutionMode::Parallel),
        );

        let stream = agent_loop(
            vec![create_user_message("echo both")],
            context(vec![tool]),
            config(),
            None,
            Some(two_calls_stream("echo", release)),
        );
        collect(stream).await;

        assert!(parallel_observed.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn runs_finish_turn_after_tool_result_messages_and_before_turn_end() {
        let tool = AgentToolBuilder::new("echo")
            .description("Echo tool")
            .parameters(value_schema())
            .execute(|args| async move {
                Ok(AgentToolResult {
                    terminate: true,
                    ..text_result(arg_value(&args), json!({ "value": arg_value(&args) }))
                })
            })
            .build()
            .unwrap();
        let ordering = Arc::new(Mutex::new(Vec::<String>::new()));
        let mut config = config();
        {
            let ordering = Arc::clone(&ordering);
            config.finish_turn = Some(finish_turn(move |turn| {
                ordering.lock().push("finishTurn".to_string());
                assert_eq!(turn.tool_results.len(), 1);
                assert_eq!(
                    turn.context.messages.last().map(Message::role),
                    Some("toolResult")
                );
                None
            }));
        }
        let emit_ordering = Arc::clone(&ordering);
        let emit: AgentEventSink = Arc::new(move |event| {
            match &event {
                AgentEvent::MessageEnd { message } => emit_ordering
                    .lock()
                    .push(format!("message_end:{}", message.role())),
                AgentEvent::TurnEnd { .. } => emit_ordering.lock().push("turn_end".to_string()),
                _ => {}
            }
            Box::pin(async { Ok(()) })
        });
        let (stream_fn, _) = scripted(|_, _| {
            create_assistant_message(
                vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
                StopReason::ToolUse,
            )
        });

        run_agent_loop(
            vec![create_user_message("echo")],
            context(vec![tool]),
            config,
            emit,
            None,
            Some(stream_fn),
        )
        .await
        .unwrap();

        let ordering = ordering.lock().clone();
        assert_eq!(
            ordering[ordering.len() - 3..],
            ["message_end:toolResult", "finishTurn", "turn_end"]
        );
    }

    async fn finish_turn_for_failed_assistant(reason: StopReason) {
        let ordering = Arc::new(Mutex::new(Vec::<&str>::new()));
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let follow_up_polls = Arc::new(AtomicUsize::new(0));
        let mut config = config();
        {
            let ordering = Arc::clone(&ordering);
            config.finish_turn = Some(finish_turn(move |turn| {
                assert_eq!(turn.message.stop_reason, reason);
                ordering.lock().push("finishTurn");
                Some(AgentTurnDecision::Continue)
            }));
        }
        {
            let steering_polls = Arc::clone(&steering_polls);
            config.get_steering_messages = Some(queue(move || {
                steering_polls.fetch_add(1, Ordering::SeqCst);
                Vec::new()
            }));
        }
        {
            let follow_up_polls = Arc::clone(&follow_up_polls);
            config.get_follow_up_messages = Some(queue(move || {
                follow_up_polls.fetch_add(1, Ordering::SeqCst);
                vec![create_user_message("queued")]
            }));
        }
        let emit_ordering = Arc::clone(&ordering);
        let emit: AgentEventSink = Arc::new(move |event| {
            if matches!(event, AgentEvent::TurnEnd { .. }) {
                emit_ordering.lock().push("turn_end");
            }
            Box::pin(async { Ok(()) })
        });
        let (stream_fn, provider_calls) = scripted(move |_, _| AssistantMessage {
            error_message: Some(reason.as_str().to_string()),
            ..create_assistant_message(Vec::new(), reason)
        });

        run_agent_loop(
            vec![create_user_message("run")],
            context(Vec::new()),
            config,
            emit,
            None,
            Some(stream_fn),
        )
        .await
        .unwrap();

        assert_eq!(ordering.lock().as_slice(), ["finishTurn", "turn_end"]);
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
        assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn runs_finish_turn_for_an_error_assistant_before_turn_end_without_changing_the_hard_exit()
     {
        finish_turn_for_failed_assistant(StopReason::Error).await;
    }

    #[tokio::test]
    async fn runs_finish_turn_for_an_aborted_assistant_before_turn_end_without_changing_the_hard_exit()
     {
        finish_turn_for_failed_assistant(StopReason::Aborted).await;
    }

    #[tokio::test]
    async fn action_end_skips_queue_polling_and_next_turn_preparation() {
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let follow_up_polls = Arc::new(AtomicUsize::new(0));
        let prepare_next_turn_calls = Arc::new(AtomicUsize::new(0));
        let mut config = config();
        config.finish_turn = Some(finish_turn(|_| Some(AgentTurnDecision::End)));
        {
            let calls = Arc::clone(&prepare_next_turn_calls);
            config.prepare_next_turn = Some(Arc::new(move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Ok(None) })
            }));
        }
        {
            let steering_polls = Arc::clone(&steering_polls);
            config.get_steering_messages = Some(queue(move || {
                steering_polls.fetch_add(1, Ordering::SeqCst);
                Vec::new()
            }));
        }
        {
            let follow_up_polls = Arc::clone(&follow_up_polls);
            config.get_follow_up_messages = Some(queue(move || {
                follow_up_polls.fetch_add(1, Ordering::SeqCst);
                vec![create_user_message("queued")]
            }));
        }
        let (stream_fn, provider_calls) = scripted(|_, _| {
            create_assistant_message(
                vec![tool_call("tool-1", "noop", json!({}))],
                StopReason::ToolUse,
            )
        });

        let mut stream = agent_loop(
            vec![create_user_message("run")],
            context(vec![noop_tool()]),
            config,
            None,
            Some(stream_fn),
        );
        stream.result().await.unwrap();

        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
        assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
        assert_eq!(prepare_next_turn_calls.load(Ordering::SeqCst), 0);
    }

    /// `finishTurn` that asks to continue on its first call only.
    fn continue_once(finish_calls: Arc<AtomicUsize>) -> crate::agent::types::FinishTurnFn {
        finish_turn(move |_| {
            let calls = finish_calls.fetch_add(1, Ordering::SeqCst) + 1;
            (calls == 1).then_some(AgentTurnDecision::Continue)
        })
    }

    #[tokio::test]
    async fn makes_exactly_one_context_only_request_when_no_natural_request_satisfies_continuation()
    {
        let finish_calls = Arc::new(AtomicUsize::new(0));
        let mut config = config();
        config.finish_turn = Some(continue_once(Arc::clone(&finish_calls)));
        let (stream_fn, provider_calls) = scripted(|index, _| {
            create_assistant_message(
                vec![text(&format!("response {}", index + 1))],
                StopReason::Stop,
            )
        });

        let mut stream = agent_loop(
            vec![create_user_message("run")],
            context(Vec::new()),
            config,
            None,
            Some(stream_fn),
        );
        stream.result().await.unwrap();

        assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
        assert_eq!(finish_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn lets_a_natural_tool_result_request_satisfy_continuation() {
        let finish_calls = Arc::new(AtomicUsize::new(0));
        let mut config = config();
        config.finish_turn = Some(continue_once(Arc::clone(&finish_calls)));
        let (stream_fn, provider_calls) =
            tool_then_done(vec![tool_call("tool-1", "noop", json!({}))]);

        let mut stream = agent_loop(
            vec![create_user_message("run")],
            context(vec![noop_tool()]),
            config,
            None,
            Some(stream_fn),
        );
        stream.result().await.unwrap();

        assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
        assert_eq!(finish_calls.load(Ordering::SeqCst), 2);
    }

    async fn natural_queue_request_satisfies_continuation(steering: bool) {
        let queue_kind = if steering { "steering" } else { "follow-up" };
        let finish_calls = Arc::new(AtomicUsize::new(0));
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let follow_up_delivered = Arc::new(AtomicBool::new(false));
        let second_request_users = Arc::new(Mutex::new(Vec::new()));
        let mut config = config();
        config.finish_turn = Some(continue_once(Arc::clone(&finish_calls)));
        {
            let steering_polls = Arc::clone(&steering_polls);
            config.get_steering_messages = Some(queue(move || {
                let polls = steering_polls.fetch_add(1, Ordering::SeqCst) + 1;
                if steering && polls == 2 {
                    vec![create_user_message(queue_kind)]
                } else {
                    Vec::new()
                }
            }));
        }
        {
            let delivered = Arc::clone(&follow_up_delivered);
            config.get_follow_up_messages = Some(queue(move || {
                if steering || delivered.swap(true, Ordering::SeqCst) {
                    Vec::new()
                } else {
                    vec![create_user_message(queue_kind)]
                }
            }));
        }
        let users = Arc::clone(&second_request_users);
        let (stream_fn, provider_calls) = scripted(move |index, context| {
            if index == 1 {
                users.lock().extend(user_texts(&context.messages));
            }
            create_assistant_message(vec![text("done")], StopReason::Stop)
        });

        let mut stream = agent_loop(
            vec![create_user_message("run")],
            context(Vec::new()),
            config,
            None,
            Some(stream_fn),
        );
        stream.result().await.unwrap();

        assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
        assert_eq!(finish_calls.load(Ordering::SeqCst), 2);
        assert!(
            second_request_users
                .lock()
                .contains(&queue_kind.to_string())
        );
    }

    #[tokio::test]
    async fn lets_a_natural_steering_request_satisfy_continuation() {
        natural_queue_request_satisfies_continuation(true).await;
    }

    #[tokio::test]
    async fn lets_a_natural_follow_up_request_satisfy_continuation() {
        natural_queue_request_satisfies_continuation(false).await;
    }

    #[tokio::test]
    async fn prepares_the_initial_request_after_pending_messages_and_can_replace_request_state() {
        let replacement_model = crate::types::Model {
            id: "replacement".to_string(),
            name: "replacement".to_string(),
            ..create_model()
        };
        let canonical_message = create_user_message("canonical projection");
        let steering_message = create_user_message("steering");
        let completed_messages = Arc::new(Mutex::new(Vec::new()));
        let steering_delivered = Arc::new(AtomicBool::new(false));
        let prepare_calls = Arc::new(AtomicUsize::new(0));
        let mut config = config();
        {
            let steering_message = steering_message.clone();
            let delivered = Arc::clone(&steering_delivered);
            config.get_steering_messages = Some(queue(move || {
                if delivered.swap(true, Ordering::SeqCst) {
                    Vec::new()
                } else {
                    vec![steering_message.clone()]
                }
            }));
        }
        {
            let completed = Arc::clone(&completed_messages);
            let prepare_calls = Arc::clone(&prepare_calls);
            let steering_message = steering_message.clone();
            let canonical_message = canonical_message.clone();
            let replacement_model = replacement_model.clone();
            config.prepare_request = Some(Arc::new(move |request: PrepareRequestContext, _| {
                prepare_calls.fetch_add(1, Ordering::SeqCst);
                assert!(completed.lock().contains(&steering_message));
                assert!(request.context.messages.contains(&steering_message));
                let update = AgentRequestUpdate {
                    context: Some(AgentContext {
                        messages: vec![canonical_message.clone()],
                        ..request.context
                    }),
                    model: Some(replacement_model.clone()),
                    thinking_level: Some(ModelThinkingLevel::High),
                };
                Box::pin(async move { Ok(Some(update)) })
            }));
        }
        let completed = Arc::clone(&completed_messages);
        let emit: AgentEventSink = Arc::new(move |event| {
            if let AgentEvent::MessageEnd { message } = event {
                completed.lock().push(message);
            }
            Box::pin(async { Ok(()) })
        });
        let checks = Arc::new(Mutex::new(Vec::new()));
        let checks_in_fn = Arc::clone(&checks);
        let stream_fn = make_stream_fn(move |model, context, options| {
            checks_in_fn.lock().push((
                model.id.clone(),
                context.messages.clone(),
                options.reasoning,
            ));
            text_stream("done")
        });

        run_agent_loop(
            vec![create_user_message("prompt")],
            context(Vec::new()),
            config,
            emit,
            None,
            Some(stream_fn),
        )
        .await
        .unwrap();

        assert_eq!(
            checks.lock().as_slice(),
            [(
                "replacement".to_string(),
                vec![canonical_message],
                Some(ThinkingLevel::High)
            )]
        );
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn does_not_poll_steering_after_prepare_request() {
        let queued = Arc::new(Mutex::new(Vec::new()));
        let late_steering = create_user_message("late steering");
        let request_included_steering = Arc::new(Mutex::new(Vec::new()));
        let request_preparations = Arc::new(AtomicUsize::new(0));
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let mut config = config();
        {
            let queued = Arc::clone(&queued);
            let steering_polls = Arc::clone(&steering_polls);
            config.get_steering_messages = Some(queue(move || {
                steering_polls.fetch_add(1, Ordering::SeqCst);
                std::mem::take(&mut *queued.lock())
            }));
        }
        {
            let queued = Arc::clone(&queued);
            let preparations = Arc::clone(&request_preparations);
            let late_steering = late_steering.clone();
            config.prepare_request = Some(Arc::new(move |_, _| {
                if preparations.fetch_add(1, Ordering::SeqCst) == 0 {
                    queued.lock().push(late_steering.clone());
                }
                Box::pin(async { Ok(None) })
            }));
        }
        let included = Arc::clone(&request_included_steering);
        let (stream_fn, _) = scripted(move |_, context| {
            included
                .lock()
                .push(context.messages.contains(&late_steering));
            create_assistant_message(vec![text("done")], StopReason::Stop)
        });

        let mut stream = agent_loop(
            vec![create_user_message("run")],
            context(Vec::new()),
            config,
            None,
            Some(stream_fn),
        );
        stream.result().await.unwrap();

        assert_eq!(request_included_steering.lock().as_slice(), [false, true]);
        assert_eq!(request_preparations.load(Ordering::SeqCst), 2);
        // Startup, post-turn delivery, then the final natural-stop check.
        assert_eq!(steering_polls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn should_use_prepare_next_turn_snapshot_before_continuing() {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let prepare_calls = Arc::new(AtomicUsize::new(0));
        let prepared = Arc::new(AtomicBool::new(false));
        let mut config = config();
        {
            let prepare_calls = Arc::clone(&prepare_calls);
            let prepared = Arc::clone(&prepared);
            config.prepare_next_turn = Some(Arc::new(move |turn: PrepareNextTurnContext| {
                prepare_calls.fetch_add(1, Ordering::SeqCst);
                let update =
                    (!prepared.swap(true, Ordering::SeqCst)).then(|| AgentLoopTurnUpdate {
                        context: Some(AgentContext {
                            messages: turn.context.messages.clone(),
                            tools: turn.context.tools.clone(),
                        }),
                        messages: Some(vec![Message::System(SystemMessage {
                            content: SystemMessageContent::Text("updated guidance".to_string()),
                            timestamp: 1,
                            ..Default::default()
                        })]),
                        ..Default::default()
                    });
                Box::pin(async move { Ok(update) })
            }));
        }
        let second_turn_has_update = Arc::new(AtomicBool::new(false));
        let has_update = Arc::clone(&second_turn_has_update);
        let (stream_fn, llm_calls) = scripted(move |index, context| {
            if index == 1 {
                has_update.store(
                    context.messages.iter().any(|message| {
                        matches!(message, Message::System(system)
                            if system.content == SystemMessageContent::Text("updated guidance".to_string()))
                    }),
                    Ordering::SeqCst,
                );
            }
            if index == 0 {
                create_assistant_message(
                    vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
                    StopReason::ToolUse,
                )
            } else {
                create_assistant_message(vec![text("done")], StopReason::Stop)
            }
        });

        let stream = agent_loop(
            vec![create_user_message("echo something")],
            context(vec![echo_tool(executed)]),
            config,
            None,
            Some(stream_fn),
        );
        collect(stream).await;

        assert_eq!(llm_calls.load(Ordering::SeqCst), 2);
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
        assert!(second_turn_has_update.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn picks_up_steering_queued_during_prepare_next_turn_before_the_next_request() {
        let queued = Arc::new(Mutex::new(Vec::new()));
        let late_steering = create_user_message("late steering");
        let mut config = config();
        {
            let queued = Arc::clone(&queued);
            let late_steering = late_steering.clone();
            config.prepare_next_turn = Some(Arc::new(move |_| {
                queued.lock().push(late_steering.clone());
                Box::pin(async { Ok(None) })
            }));
        }
        {
            let queued = Arc::clone(&queued);
            config.get_steering_messages = Some(queue(move || std::mem::take(&mut *queued.lock())));
        }
        let included = Arc::new(AtomicBool::new(false));
        let included_in_fn = Arc::clone(&included);
        let (stream_fn, provider_calls) = scripted(move |index, context| {
            if index == 1 {
                included_in_fn.store(context.messages.contains(&late_steering), Ordering::SeqCst);
            }
            if index == 0 {
                create_assistant_message(
                    vec![tool_call("tool-1", "noop", json!({}))],
                    StopReason::ToolUse,
                )
            } else {
                create_assistant_message(vec![text("done")], StopReason::Stop)
            }
        });

        let mut stream = agent_loop(
            vec![create_user_message("run")],
            context(vec![noop_tool()]),
            config,
            None,
            Some(stream_fn),
        );
        stream.result().await.unwrap();

        assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
        assert!(included.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn action_end_receives_finalized_turn_context_and_stops_before_queue_polling() {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let follow_up_polls = Arc::new(AtomicUsize::new(0));
        let callback_tool_result_ids = Arc::new(Mutex::new(Vec::new()));
        let callback_context_roles = Arc::new(Mutex::new(Vec::new()));
        let mut config = config();
        {
            let ids = Arc::clone(&callback_tool_result_ids);
            let context_roles = Arc::clone(&callback_context_roles);
            config.finish_turn = Some(finish_turn(move |turn| {
                *ids.lock() = turn
                    .tool_results
                    .iter()
                    .map(|result| result.tool_call_id.clone())
                    .collect();
                *context_roles.lock() = roles(&turn.context.messages);
                Some(AgentTurnDecision::End)
            }));
        }
        {
            let steering_polls = Arc::clone(&steering_polls);
            config.get_steering_messages = Some(queue(move || {
                steering_polls.fetch_add(1, Ordering::SeqCst);
                Vec::new()
            }));
        }
        {
            let follow_up_polls = Arc::clone(&follow_up_polls);
            config.get_follow_up_messages = Some(queue(move || {
                follow_up_polls.fetch_add(1, Ordering::SeqCst);
                vec![create_user_message("follow up should stay queued")]
            }));
        }
        let (stream_fn, llm_calls) = scripted(|index, _| {
            if index == 0 {
                create_assistant_message(
                    vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
                    StopReason::ToolUse,
                )
            } else {
                create_assistant_message(vec![text("should not run")], StopReason::Stop)
            }
        });

        let stream = agent_loop(
            vec![create_user_message("echo something")],
            context(vec![echo_tool(Arc::clone(&executed))]),
            config,
            None,
            Some(stream_fn),
        );
        let (events, messages) = collect(stream).await;

        assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
        assert_eq!(executed.lock().as_slice(), ["hello"]);
        assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
        assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
        assert_eq!(callback_tool_result_ids.lock().as_slice(), ["tool-1"]);
        assert_eq!(
            callback_context_roles.lock().as_slice(),
            ["system", "user", "assistant", "toolResult"]
        );
        // The context declares no tools, so the loop announces the loadout with a system message.
        assert_eq!(
            roles(&messages),
            ["system", "user", "assistant", "toolResult"]
        );
        assert_eq!(
            event_types(&events),
            [
                "agent_start",
                "turn_start",
                "message_start",
                "message_end",
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
    }

    fn terminating_echo(terminate_value: Option<&'static str>) -> DynAgentTool {
        AgentToolBuilder::new("echo")
            .description("Echo tool")
            .parameters(value_schema())
            .execute(move |args| async move {
                let value = arg_value(&args);
                Ok(AgentToolResult {
                    terminate: terminate_value.is_none_or(|expected| value == expected),
                    ..text_result(format!("echoed: {value}"), json!({ "value": value }))
                })
            })
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn should_stop_after_a_tool_batch_when_every_tool_result_sets_terminate_true() {
        let (stream_fn, llm_calls) = scripted(|_, _| {
            create_assistant_message(
                vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
                StopReason::ToolUse,
            )
        });

        let stream = agent_loop(
            vec![create_user_message("echo something")],
            context(vec![terminating_echo(None)]),
            config(),
            None,
            Some(stream_fn),
        );
        let (events, messages) = collect(stream).await;

        assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            roles(&messages),
            ["system", "user", "assistant", "toolResult"]
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::TurnEnd { .. }))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn should_stop_after_a_blocked_tool_call_when_before_tool_call_sets_terminate_true() {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let mut config = config();
        config.before_tool_call = Some(Arc::new(|_, _| {
            Box::pin(async {
                Ok(Some(BeforeToolCallResult {
                    block: true,
                    reason: Some("Blocked by policy".to_string()),
                    terminate: true,
                }))
            })
        }));
        let (stream_fn, llm_calls) = tool_then_done(vec![tool_call(
            "tool-1",
            "echo",
            json!({ "value": "hello" }),
        )]);

        let stream = agent_loop(
            vec![create_user_message("echo something")],
            context(vec![echo_tool(Arc::clone(&executed))]),
            config,
            None,
            Some(stream_fn),
        );
        let (_, messages) = collect(stream).await;

        let tool_result = messages
            .iter()
            .find_map(|message| match message {
                Message::ToolResult(result) => Some(result.clone()),
                _ => None,
            })
            .expect("tool result");
        assert!(executed.lock().is_empty());
        assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
        assert!(tool_result.is_error);
        assert!(
            tool_result
                .content
                .contains(&ToolResultContent::text("Blocked by policy"))
        );
    }

    #[tokio::test]
    async fn should_continue_after_a_mixed_batch_with_one_terminating_blocked_call() {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let mut config = config();
        config.tool_execution = ToolExecutionMode::Parallel;
        config.before_tool_call = Some(Arc::new(|context: BeforeToolCallContext, _| {
            let value = arg_value(&context.args.lock());
            Box::pin(async move {
                Ok((value == "first").then(|| BeforeToolCallResult {
                    block: true,
                    reason: Some("Blocked first".to_string()),
                    terminate: true,
                }))
            })
        }));
        let (stream_fn, llm_calls) = tool_then_done(vec![
            tool_call("tool-1", "echo", json!({ "value": "first" })),
            tool_call("tool-2", "echo", json!({ "value": "second" })),
        ]);

        let stream = agent_loop(
            vec![create_user_message("echo both")],
            context(vec![echo_tool(Arc::clone(&executed))]),
            config,
            None,
            Some(stream_fn),
        );
        collect(stream).await;

        assert_eq!(executed.lock().as_slice(), ["second"]);
        assert_eq!(llm_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn should_continue_after_parallel_tool_calls_when_not_all_tool_results_terminate() {
        let mut config = config();
        config.tool_execution = ToolExecutionMode::Parallel;
        let (stream_fn, llm_calls) = tool_then_done(vec![
            tool_call("tool-1", "echo", json!({ "value": "first" })),
            tool_call("tool-2", "echo", json!({ "value": "second" })),
        ]);

        let stream = agent_loop(
            vec![create_user_message("echo both")],
            context(vec![terminating_echo(Some("first"))]),
            config,
            None,
            Some(stream_fn),
        );
        let (_, messages) = collect(stream).await;

        assert_eq!(llm_calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            roles(&messages),
            [
                "system",
                "user",
                "assistant",
                "toolResult",
                "toolResult",
                "assistant"
            ]
        );
    }

    #[tokio::test]
    async fn should_allow_after_tool_call_to_mark_a_tool_batch_as_terminating() {
        let mut config = config();
        config.after_tool_call = Some(Arc::new(|_, _| {
            Box::pin(async {
                Ok(Some(AfterToolCallResult {
                    terminate: Some(true),
                    ..Default::default()
                }))
            })
        }));
        let (stream_fn, llm_calls) = scripted(|_, _| {
            create_assistant_message(
                vec![tool_call("tool-1", "echo", json!({ "value": "hello" }))],
                StopReason::ToolUse,
            )
        });

        let stream = agent_loop(
            vec![create_user_message("echo something")],
            context(vec![echo_tool(Arc::new(Mutex::new(Vec::new())))]),
            config,
            None,
            Some(stream_fn),
        );
        collect(stream).await;

        assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
    }

    /// Rust addition: an abort before deferred parallel execution reports
    /// "Operation aborted" without running the tool.
    #[tokio::test]
    async fn parallel_calls_prepared_before_an_abort_are_reported_as_aborted() {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let signal = CancellationToken::new();
        let mut config = config();
        let abort = signal.clone();
        config.before_tool_call = Some(Arc::new(move |context: BeforeToolCallContext, _| {
            // Abort after the second call is prepared.
            if arg_value(&context.args.lock()) == "second" {
                abort.cancel();
            }
            Box::pin(async { Ok(None) })
        }));
        let (stream_fn, _) = tool_then_done(vec![
            tool_call("tool-1", "echo", json!({ "value": "first" })),
            tool_call("tool-2", "echo", json!({ "value": "second" })),
        ]);

        let stream = agent_loop(
            vec![create_user_message("echo both")],
            context(vec![echo_tool(Arc::clone(&executed))]),
            config,
            Some(signal),
            Some(stream_fn),
        );
        let (_, messages) = collect(stream).await;

        assert!(executed.lock().is_empty());
        let results: Vec<_> = messages
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult(result) => Some((result.content.clone(), result.is_error)),
                _ => None,
            })
            .collect();
        assert_eq!(
            results,
            [
                (vec![ToolResultContent::text("Operation aborted")], true),
                (vec![ToolResultContent::text("Operation aborted")], true),
            ]
        );
    }
}

mod agent_loop_continue_with_agent_message {
    use super::*;

    #[tokio::test]
    async fn should_throw_when_context_has_no_messages() {
        let error = agent_loop_continue(context(Vec::new()), config(), None, None)
            .err()
            .expect("error");
        assert_eq!(error.to_string(), "Cannot continue: no messages in context");
    }

    #[tokio::test]
    async fn should_continue_from_existing_context_without_emitting_user_message_events() {
        let (stream_fn, _) =
            scripted(|_, _| create_assistant_message(vec![text("Response")], StopReason::Stop));
        let stream = agent_loop_continue(
            AgentContext {
                messages: vec![create_user_message("Hello")],
                tools: Vec::new(),
            },
            config(),
            None,
            Some(stream_fn),
        )
        .unwrap();
        let (events, messages) = collect(stream).await;

        // Only the new assistant message (not the existing user message).
        assert_eq!(roles(&messages), ["assistant"]);
        let message_end_roles: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::MessageEnd { message } => Some(message.role()),
                _ => None,
            })
            .collect();
        assert_eq!(message_end_roles, ["assistant"]);
    }

    /// Pi ends the context with a custom message that `convertToLlm` turns
    /// into a user message; a system message stands in for it here.
    #[tokio::test]
    async fn should_allow_non_llm_last_messages_converted_by_convert_to_llm() {
        let mut config = config();
        config.convert_to_llm = Arc::new(|messages: Vec<AgentMessage>| {
            let converted = messages
                .into_iter()
                .map(|message| match message {
                    Message::System(system) => Message::User(UserMessage {
                        content: UserMessageContent::Text(crate::utils::text::content_text(
                            &system.content,
                        )),
                        timestamp: system.timestamp,
                    }),
                    other => other,
                })
                .collect();
            Box::pin(async move { converted })
        });
        let (stream_fn, _) = scripted(|_, _| {
            create_assistant_message(vec![text("Response to custom message")], StopReason::Stop)
        });

        let stream = agent_loop_continue(
            AgentContext {
                messages: vec![Message::System(SystemMessage {
                    content: SystemMessageContent::Text("Hook content".to_string()),
                    timestamp: now_millis(),
                    ..Default::default()
                })],
                tools: Vec::new(),
            },
            config,
            None,
            Some(stream_fn),
        )
        .unwrap();
        let (_, messages) = collect(stream).await;

        assert_eq!(roles(&messages), ["assistant"]);
    }
}

mod run_tool_call_tests {
    use super::*;

    fn echo() -> DynAgentTool {
        AgentToolBuilder::new("echo")
            .label("Echo")
            .description("Echo tool")
            .parameters(value_schema())
            .output_schema(value_schema())
            .execute_with_context(|_id, args, _signal, on_update| async move {
                if let Some(on_update) = on_update {
                    on_update(text_result("partial", json!({}))).await;
                }
                let value = arg_value(&args);
                Ok(AgentToolResult {
                    structured_content: Some(json!({ "value": value })),
                    ..text_result(value.clone(), json!({}))
                })
            })
            .build()
            .unwrap()
    }

    fn failing() -> DynAgentTool {
        AgentToolBuilder::new("failing")
            .label("Failing")
            .description("Returns an error result")
            .parameters(empty_schema())
            .execute(|_| async {
                Ok(AgentToolResult {
                    is_error: true,
                    ..text_result("bad", json!({ "partial": true }))
                })
            })
            .build()
            .unwrap()
    }

    fn call(id: &str, name: &str, arguments: Value) -> AgentToolCall {
        AgentToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
            thought_signature: None,
            namespace: None,
        }
    }

    fn assistant_message() -> AssistantMessage {
        create_assistant_message(Vec::new(), StopReason::Stop)
    }

    #[tokio::test]
    async fn validates_runs_the_hooks_and_reports_failures_as_error_outcomes() {
        let hook_calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let updates = Arc::new(Mutex::new(Vec::new()));
        let mut options = RunToolCallOptions::new(
            vec![echo(), failing()],
            assistant_message(),
            AgentContext::default(),
        );
        {
            let hook_calls = Arc::clone(&hook_calls);
            options.before_tool_call = Some(Arc::new(move |context: BeforeToolCallContext, _| {
                hook_calls
                    .lock()
                    .push(format!("before {}", context.tool_call.id));
                let blocked = context.args.lock()["value"] == "blocked";
                Box::pin(async move {
                    Ok(blocked.then(|| BeforeToolCallResult {
                        block: true,
                        reason: Some("nope".to_string()),
                        terminate: false,
                    }))
                })
            }));
        }
        {
            let hook_calls = Arc::clone(&hook_calls);
            options.after_tool_call = Some(Arc::new(move |context: AfterToolCallContext, _| {
                hook_calls
                    .lock()
                    .push(format!("after {}", context.tool_call.id));
                Box::pin(async { Ok(None) })
            }));
        }
        {
            let updates = Arc::clone(&updates);
            options.on_update = Some(Arc::new(move |partial| {
                updates.lock().push(partial);
                Box::pin(async {})
            }));
        }

        let outcome =
            run_tool_call(call("a", "echo", json!({ "value": "a" })), options.clone()).await;
        assert_eq!(outcome.tool_call.id, "a");
        assert_eq!(
            outcome.result.structured_content,
            Some(json!({ "value": "a" }))
        );
        assert!(!outcome.is_error);

        let outcome = run_tool_call(
            call("b", "echo", json!({ "value": { "nested": true } })),
            options.clone(),
        )
        .await;
        assert!(outcome.is_error);

        let outcome = run_tool_call(
            call("c", "echo", json!({ "value": "blocked" })),
            options.clone(),
        )
        .await;
        assert_eq!(outcome.result.content, [ToolResultContent::text("nope")]);
        assert!(outcome.is_error);

        let outcome = run_tool_call(call("d", "missing", json!({})), options.clone()).await;
        assert_eq!(
            outcome.result.content,
            [ToolResultContent::text("Tool missing not found")]
        );
        assert!(outcome.is_error);

        // Error results keep their details.
        let outcome = run_tool_call(call("e", "failing", json!({})), options.clone()).await;
        assert_eq!(outcome.result.details, Some(json!({ "partial": true })));
        assert!(outcome.is_error);

        assert_eq!(
            updates.lock().as_slice(),
            [text_result("partial", json!({}))]
        );
        // Validation failures and unknown tools never reach the hooks; blocked calls skip after_tool_call.
        assert_eq!(
            hook_calls.lock().as_slice(),
            ["before a", "after a", "before c", "before e", "after e"]
        );
    }

    #[tokio::test]
    async fn lets_after_tool_call_replace_structured_content_and_drops_it_when_only_content_is_replaced()
     {
        let redacted = vec![ToolResultContent::text("redacted")];
        let results = [
            AfterToolCallResult {
                content: Some(redacted.clone()),
                ..Default::default()
            },
            AfterToolCallResult {
                structured_content: Some(json!({ "value": "replaced" })),
                ..Default::default()
            },
            AfterToolCallResult {
                content: Some(redacted),
                structured_content: Some(json!({ "value": "both" })),
                ..Default::default()
            },
            AfterToolCallResult {
                details: Some(json!({ "note": "kept" })),
                ..Default::default()
            },
        ];
        let mut seen = Vec::new();
        for after_result in results {
            let mut options =
                RunToolCallOptions::new(vec![echo()], assistant_message(), AgentContext::default());
            options.after_tool_call = Some(Arc::new(move |_, _| {
                let after_result = after_result.clone();
                Box::pin(async move { Ok(Some(after_result)) })
            }));
            let outcome =
                run_tool_call(call("x", "echo", json!({ "value": "original" })), options).await;
            seen.push(outcome.result.structured_content);
        }
        assert_eq!(
            seen,
            [
                None,
                Some(json!({ "value": "replaced" })),
                Some(json!({ "value": "both" })),
                Some(json!({ "value": "original" })),
            ]
        );
    }
}

/// Rust addition: `run_agent_loop` emits nothing for an empty prompt batch
/// besides the lifecycle events, and a declared tool set yields no extra
/// system message.
#[tokio::test]
async fn declared_tools_need_no_update_message() {
    let tool = noop_tool();
    let declarations = vec![to_tool_declaration(&tool.definition())];
    let (stream_fn, _) =
        scripted(|_, _| create_assistant_message(vec![text("ok")], StopReason::Stop));
    let events = Arc::new(Mutex::new(Vec::new()));
    let messages = run_agent_loop(
        vec![create_user_message("hi")],
        AgentContext {
            messages: vec![Message::System(SystemMessage {
                tools_added: Some(declarations),
                ..Default::default()
            })],
            tools: vec![tool],
        },
        config(),
        sink(Arc::clone(&events)),
        None,
        Some(stream_fn),
    )
    .await
    .unwrap();
    assert_eq!(roles(&messages), ["user", "assistant"]);
    assert_eq!(
        events.lock().last().map(AgentEvent::event_type),
        Some("agent_end")
    );
}

// Rust-only: a stream that ends without `done`/`error` and without a final
// result fails the loop with `StreamClosed` (Pi would wait forever).
#[tokio::test]
async fn a_stream_ending_without_a_terminal_event_fails_with_stream_closed() {
    let stream_fn = make_stream_fn(|_, _, _| {
        let stream = AssistantMessageEventStream::new();
        stream.push(AssistantMessageEvent::Start {
            partial: create_assistant_message(vec![text("")], StopReason::Stop),
        });
        stream.end(None);
        stream
    });
    let events = Arc::new(Mutex::new(Vec::new()));

    let result = run_agent_loop(
        vec![create_user_message("hello")],
        context(Vec::new()),
        config(),
        sink(Arc::clone(&events)),
        None,
        Some(stream_fn),
    )
    .await;

    assert!(matches!(result, Err(AgentError::StreamClosed)));
    assert_eq!(
        event_types(&events.lock()),
        [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
        ]
    );
}
