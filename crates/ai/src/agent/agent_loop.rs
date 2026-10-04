//! Port of `packages/agent/src/agent-loop.ts`.
//!
//! Agent loop that works with agent messages throughout and transforms them
//! to LLM messages only at the call boundary.
//!
//! Divergences from Pi:
//! - Pi's `agentLoop` returns an `EventStream` whose result never resolves
//!   when the loop rejects. [`AgentEventStream`] is an event channel plus a
//!   result that resolves with the loop's `Err` instead.
//! - A stream function whose stream ends without `done`/`error` and without
//!   a final result fails the run with [`AgentError::StreamClosed`] (Pi
//!   would wait forever on `response.result()`).
//! - Tool progress updates are delivered through a channel that the loop
//!   drains while the tool runs, so updates keep their order; an error from
//!   delivering an update surfaces after the tool settles, as in Pi.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context as TaskContext, Poll};

use futures::future::join_all;
use futures::{FutureExt, Stream, StreamExt};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::stream_fn::get_default_stream_fn;
use crate::agent::types::{
    AfterToolCallContext, AfterToolCallFn, AgentContext, AgentEvent, AgentEventSink,
    AgentLoopConfig, AgentMessage, AgentToolCall, AgentToolCallOutcome, AgentToolResult,
    AgentToolUpdateCallback, AgentTurnContext, AgentTurnDecision, BeforeToolCallContext,
    BeforeToolCallFn, DynAgentTool, PrepareNextTurnContext, PrepareRequestContext, StreamFn,
    ToolExecutionMode, assistant_tool_calls,
};
use crate::types::{
    AssistantMessage, AssistantMessageEvent, BoxFuture, Context, Message, ModelThinkingLevel,
    StopReason, SystemMessage, SystemMessageContent, ToolResultContent, ToolResultMessage,
};
use crate::utils::time::now_millis;
use crate::utils::transcript::{
    ToolStateChanges, get_current_tools, get_tool_state_changes, normalize_context,
    to_tool_declaration,
};
use crate::utils::validation::validate_tool_arguments;

/// Events of one loop invocation plus its final messages.
pub struct AgentEventStream {
    receiver: mpsc::UnboundedReceiver<AgentEvent>,
    result_receiver: Option<oneshot::Receiver<AgentResult<Vec<AgentMessage>>>>,
    result: Option<Vec<AgentMessage>>,
}

impl AgentEventStream {
    /// The messages the loop added (Pi: the `agent_end` messages).
    pub async fn result(&mut self) -> AgentResult<Vec<AgentMessage>> {
        if let Some(result) = &self.result {
            return Ok(result.clone());
        }
        let receiver = self
            .result_receiver
            .take()
            .ok_or(AgentError::StreamClosed)?;
        let result = receiver.await.map_err(|_| AgentError::StreamClosed)??;
        self.result = Some(result.clone());
        Ok(result)
    }
}

impl Stream for AgentEventStream {
    type Item = AgentEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

fn create_agent_stream() -> (
    AgentEventSink,
    oneshot::Sender<AgentResult<Vec<AgentMessage>>>,
    AgentEventStream,
) {
    let (event_sender, receiver) = mpsc::unbounded_channel();
    let (result_sender, result_receiver) = oneshot::channel();
    let emit: AgentEventSink = Arc::new(move |event| {
        let _ = event_sender.send(event);
        Box::pin(async { Ok(()) })
    });
    (
        emit,
        result_sender,
        AgentEventStream {
            receiver,
            result_receiver: Some(result_receiver),
            result: None,
        },
    )
}

/// Start an agent loop with new prompt messages. The prompts are added to
/// the context and events are emitted for them.
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    stream_fn: Option<StreamFn>,
) -> AgentEventStream {
    let (emit, result_sender, stream) = create_agent_stream();
    tokio::spawn(async move {
        let result = run_agent_loop(prompts, context, config, emit, signal, stream_fn).await;
        let _ = result_sender.send(result);
    });
    stream
}

/// Continue an agent loop from the current context without adding a new
/// message. Used for retries: the context already has a user message or
/// tool results.
///
/// The last message in context must convert to a `user` or `toolResult`
/// message via `convert_to_llm`; this cannot be validated here.
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    stream_fn: Option<StreamFn>,
) -> AgentResult<AgentEventStream> {
    check_continue_context(&context)?;
    let (emit, result_sender, stream) = create_agent_stream();
    tokio::spawn(async move {
        let result = run_agent_loop_continue(context, config, emit, signal, stream_fn).await;
        let _ = result_sender.send(result);
    });
    Ok(stream)
}

fn check_continue_context(context: &AgentContext) -> AgentResult<()> {
    match context.messages.last() {
        None => Err(AgentError::NoMessagesInContext),
        Some(Message::Assistant(_)) => Err(AgentError::CannotContinueFromAssistant),
        Some(_) => Ok(()),
    }
}

pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Option<CancellationToken>,
    stream_fn: Option<StreamFn>,
) -> AgentResult<Vec<AgentMessage>> {
    let initial_messages = declare_tool_changes(&context, &prompts);
    let mut new_messages = initial_messages.clone();
    let mut messages = context.messages;
    messages.extend(initial_messages.iter().cloned());
    let current_context = AgentContext {
        messages,
        tools: context.tools,
    };

    emit(AgentEvent::AgentStart).await?;
    emit(AgentEvent::TurnStart).await?;
    for message in initial_messages {
        emit(AgentEvent::MessageStart {
            message: message.clone(),
        })
        .await?;
        emit(AgentEvent::MessageEnd { message }).await?;
    }

    let stream_function = match stream_fn {
        Some(stream_fn) => stream_fn,
        None => get_default_stream_fn()?,
    };
    run_loop(
        current_context,
        &mut new_messages,
        config,
        signal,
        &emit,
        stream_function,
    )
    .await?;
    Ok(new_messages)
}

pub async fn run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Option<CancellationToken>,
    stream_fn: Option<StreamFn>,
) -> AgentResult<Vec<AgentMessage>> {
    check_continue_context(&context)?;

    let mut new_messages = Vec::new();
    emit(AgentEvent::AgentStart).await?;
    emit(AgentEvent::TurnStart).await?;

    let stream_function = match stream_fn {
        Some(stream_fn) => stream_fn,
        None => get_default_stream_fn()?,
    };
    run_loop(
        context,
        &mut new_messages,
        config,
        signal,
        &emit,
        stream_function,
    )
    .await?;
    Ok(new_messages)
}

async fn poll_queue(queue: &Option<crate::agent::types::MessageQueueFn>) -> Vec<AgentMessage> {
    match queue {
        Some(get) => get().await,
        None => Vec::new(),
    }
}

fn requested_thinking_level(config: &AgentLoopConfig) -> ModelThinkingLevel {
    config
        .options
        .reasoning
        .map_or(ModelThinkingLevel::Off, Into::into)
}

/// Main loop logic shared by `agent_loop` and `agent_loop_continue`.
async fn run_loop(
    initial_context: AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    initial_config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    emit: &AgentEventSink,
    stream_function: StreamFn,
) -> AgentResult<()> {
    let mut current_context = initial_context;
    let mut config = initial_config;
    let mut last_completed_turn: Option<PrepareNextTurnContext> = None;
    let mut explicit_continuation = false;
    // Check for steering messages at start (user may have typed while waiting).
    let mut pending_messages = poll_queue(&config.get_steering_messages).await;

    // Outer loop: continues when queued follow-up messages arrive after the agent would stop.
    loop {
        let mut has_more_tool_calls = true;

        // Inner loop: process tool calls and steering messages.
        while has_more_tool_calls || !pending_messages.is_empty() {
            let mut prepared_messages = Vec::new();
            if let Some(turn) = &last_completed_turn {
                let next_turn_snapshot = match &config.prepare_next_turn {
                    Some(prepare_next_turn) => prepare_next_turn(turn.clone()).await,
                    None => None,
                };
                if let Some(snapshot) = next_turn_snapshot {
                    if let Some(context) = snapshot.context {
                        current_context = context;
                    }
                    prepared_messages = snapshot.messages.unwrap_or_default();
                    if let Some(model) = snapshot.model {
                        config.model = model;
                    }
                    if let Some(thinking_level) = snapshot.thinking_level {
                        config.options.reasoning = thinking_level.thinking_level();
                    }
                }
                // Preparation can be long-running (for example, compaction). Pick up steering
                // queued while it ran. Only poll again if the earlier poll returned nothing;
                // otherwise one-at-a-time mode would deliver two messages in this turn.
                if pending_messages.is_empty() {
                    pending_messages = poll_queue(&config.get_steering_messages).await;
                }
                emit(AgentEvent::TurnStart).await?;
            }

            // Process prepared and queued messages before the next assistant response.
            prepared_messages.append(&mut pending_messages);
            for message in declare_tool_changes(&current_context, &prepared_messages) {
                emit(AgentEvent::MessageStart {
                    message: message.clone(),
                })
                .await?;
                emit(AgentEvent::MessageEnd {
                    message: message.clone(),
                })
                .await?;
                current_context.messages.push(message.clone());
                new_messages.push(message);
            }

            if let Some(prepare_request) = &config.prepare_request
                && let Some(request_update) = prepare_request(
                    PrepareRequestContext {
                        context: current_context.clone(),
                        model: config.model.clone(),
                        thinking_level: requested_thinking_level(&config),
                    },
                    signal.clone(),
                )
                .await
            {
                if let Some(context) = request_update.context {
                    current_context = context;
                }
                if let Some(model) = request_update.model {
                    config.model = model;
                }
                if let Some(thinking_level) = request_update.thinking_level {
                    config.options.reasoning = thinking_level.thinking_level();
                }
            }

            // Stream assistant response.
            let message = stream_assistant_response(
                &mut current_context,
                &config,
                signal.clone(),
                emit,
                &stream_function,
            )
            .await?;
            new_messages.push(Message::Assistant(message.clone()));

            if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
                let turn = AgentTurnContext {
                    message: message.clone(),
                    tool_results: Vec::new(),
                    context: current_context.clone(),
                    new_messages: new_messages.clone(),
                };
                if let Some(finish_turn) = &config.finish_turn {
                    finish_turn(turn, signal.clone()).await;
                }
                emit(AgentEvent::TurnEnd {
                    message: Message::Assistant(message),
                    tool_results: Vec::new(),
                })
                .await?;
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await?;
                return Ok(());
            }

            // Check for tool calls.
            let tool_calls = assistant_tool_calls(&message);

            let mut tool_results = Vec::new();
            has_more_tool_calls = false;
            if !tool_calls.is_empty() {
                // A "length" stop means the output was cut off by the token limit, so
                // every tool call in the message may carry truncated arguments. Fail
                // them all instead of executing potentially borked calls.
                let executed_tool_batch = if message.stop_reason == StopReason::Length {
                    fail_tool_calls_from_truncated_message(tool_calls, emit).await?
                } else {
                    execute_tool_calls(&current_context, &message, &config, signal.clone(), emit)
                        .await?
                };
                tool_results.extend(executed_tool_batch.messages);
                has_more_tool_calls = !executed_tool_batch.terminate;

                for result in &tool_results {
                    current_context
                        .messages
                        .push(Message::ToolResult(result.clone()));
                    new_messages.push(Message::ToolResult(result.clone()));
                }
            }

            let turn = AgentTurnContext {
                message: message.clone(),
                tool_results: tool_results.clone(),
                context: current_context.clone(),
                new_messages: new_messages.clone(),
            };
            last_completed_turn = Some(turn.clone());
            let decision = match &config.finish_turn {
                Some(finish_turn) => finish_turn(turn, signal.clone()).await,
                None => None,
            };
            emit(AgentEvent::TurnEnd {
                message: Message::Assistant(message),
                tool_results,
            })
            .await?;

            if decision == Some(AgentTurnDecision::End) {
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await?;
                return Ok(());
            }

            explicit_continuation = decision == Some(AgentTurnDecision::Continue);
            pending_messages = poll_queue(&config.get_steering_messages).await;
            if has_more_tool_calls || !pending_messages.is_empty() {
                explicit_continuation = false;
            }
        }

        // Agent would stop here. Check for follow-up messages.
        let follow_up_messages = poll_queue(&config.get_follow_up_messages).await;
        if !follow_up_messages.is_empty() {
            // Set as pending so the inner loop processes them.
            explicit_continuation = false;
            pending_messages = follow_up_messages;
            continue;
        }

        // No natural request was selected, so fulfill the continuation decision with one context-only turn.
        if explicit_continuation {
            explicit_continuation = false;
            continue;
        }

        // No more messages, exit.
        break;
    }

    emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await?;
    Ok(())
}

/// Declare tool loadout changes to the model.
///
/// `context.tools` is what the runtime can execute; the transcript's system
/// messages declare what the model may call. Before each request the
/// difference becomes `tools_added` and `tools_removed` on a system message.
/// When a pending system message exists, its tool fields are treated as
/// intent and replaced with the delta between the committed transcript and
/// the executable set, so replay always yields exactly `context.tools`.
/// Otherwise a new system message is inserted before the first non-system
/// pending message.
fn declare_tool_changes(
    context: &AgentContext,
    pending_messages: &[AgentMessage],
) -> Vec<AgentMessage> {
    let system_index = pending_messages
        .iter()
        .rposition(|message| matches!(message, Message::System(_)));
    let pending = system_index.and_then(|index| match &pending_messages[index] {
        Message::System(message) => Some(message.clone()),
        _ => None,
    });
    let baseline: Vec<AgentMessage> = match (&pending, system_index) {
        (Some(pending), Some(system_index)) => pending_messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                if index == system_index {
                    Message::System(with_tool_changes(pending, &NO_CHANGES))
                } else {
                    message.clone()
                }
            })
            .collect(),
        _ => pending_messages.to_vec(),
    };
    let declared: Vec<Message> = context
        .messages
        .iter()
        .chain(baseline.iter())
        .cloned()
        .collect();
    let executable: Vec<_> = context
        .tools
        .iter()
        .map(|tool| to_tool_declaration(&tool.definition()))
        .collect();
    let changes = get_tool_state_changes(&get_current_tools(&declared), &executable);
    let unchanged = changes.tools_added.is_empty() && changes.tools_removed.is_empty();

    if let (Some(pending), Some(system_index)) = (pending, system_index) {
        // Keep the caller's message when it already declares no tool changes.
        if unchanged
            && pending.tools_added.as_ref().is_none_or(Vec::is_empty)
            && pending.tools_removed.as_ref().is_none_or(Vec::is_empty)
        {
            return pending_messages.to_vec();
        }
        let mut messages = baseline;
        messages[system_index] = Message::System(with_tool_changes(&pending, &changes));
        return messages;
    }
    if unchanged {
        return pending_messages.to_vec();
    }
    let update = with_tool_changes(
        &SystemMessage {
            content: SystemMessageContent::Text(String::new()),
            timestamp: now_millis(),
            ..Default::default()
        },
        &changes,
    );
    let index = pending_messages
        .iter()
        .position(|message| !matches!(message, Message::System(_)))
        .unwrap_or(pending_messages.len());
    let mut messages = pending_messages.to_vec();
    messages.insert(index, Message::System(update));
    messages
}

const NO_CHANGES: ToolStateChanges = ToolStateChanges {
    tools_added: Vec::new(),
    tools_removed: Vec::new(),
};

/// Copy a system message with its tool fields replaced by `changes`; empty
/// lists omit the field.
fn with_tool_changes(message: &SystemMessage, changes: &ToolStateChanges) -> SystemMessage {
    SystemMessage {
        tools_added: (!changes.tools_added.is_empty()).then(|| changes.tools_added.clone()),
        tools_removed: (!changes.tools_removed.is_empty()).then(|| changes.tools_removed.clone()),
        ..message.clone()
    }
}

/// Stream an assistant response from the LLM. This is where agent messages
/// get transformed to LLM messages.
async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    signal: Option<CancellationToken>,
    emit: &AgentEventSink,
    stream_function: &StreamFn,
) -> AgentResult<AssistantMessage> {
    // Apply context transform if configured (agent messages -> agent messages).
    let mut messages = context.messages.clone();
    if let Some(transform_context) = &config.transform_context {
        messages = transform_context(messages, signal.clone()).await;
    }

    // Convert to LLM-compatible messages.
    let llm_messages = (config.convert_to_llm)(messages).await;

    let llm_context = normalize_context(&Context {
        system_prompt: None,
        messages: llm_messages,
        tools: None,
    });

    // Resolve API key (important for expiring tokens).
    let resolved_api_key = match &config.get_api_key {
        Some(get_api_key) => get_api_key(config.model.provider.clone()).await,
        None => None,
    }
    .filter(|api_key| !api_key.is_empty())
    .or_else(|| config.options.api_key.clone());

    let mut options = config.options.clone();
    options.api_key = resolved_api_key;
    options.signal = signal;
    let mut response = stream_function(config.model.clone(), llm_context, options).await;
    // Record the requested level, whichever stream function answered.
    let thinking_level = requested_thinking_level(config);
    let stamp = |mut message: AssistantMessage| {
        message.thinking_level = Some(thinking_level);
        message
    };

    let mut partial_message: Option<AssistantMessage> = None;
    let mut added_partial = false;

    while let Some(event) = response.next().await {
        match &event {
            AssistantMessageEvent::Start { partial } => {
                partial_message = Some(partial.clone());
                context.messages.push(Message::Assistant(partial.clone()));
                added_partial = true;
                emit(AgentEvent::MessageStart {
                    message: Message::Assistant(partial.clone()),
                })
                .await?;
            }
            AssistantMessageEvent::TextStart { partial, .. }
            | AssistantMessageEvent::TextDelta { partial, .. }
            | AssistantMessageEvent::TextEnd { partial, .. }
            | AssistantMessageEvent::ThinkingStart { partial, .. }
            | AssistantMessageEvent::ThinkingDelta { partial, .. }
            | AssistantMessageEvent::ThinkingEnd { partial, .. }
            | AssistantMessageEvent::ToolCallStart { partial, .. }
            | AssistantMessageEvent::ToolCallDelta { partial, .. }
            | AssistantMessageEvent::ToolCallEnd { partial, .. } => {
                if partial_message.is_some() {
                    partial_message = Some(partial.clone());
                    if let Some(last) = context.messages.last_mut() {
                        *last = Message::Assistant(partial.clone());
                    }
                    emit(AgentEvent::MessageUpdate {
                        message: Message::Assistant(partial.clone()),
                        assistant_message_event: event.clone(),
                    })
                    .await?;
                }
            }
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {
                let final_message = stamp(response.result().await);
                return finish_assistant_response(context, final_message, added_partial, emit)
                    .await;
            }
        }
    }

    let final_message = stamp(
        response
            .result()
            .now_or_never()
            .ok_or(AgentError::StreamClosed)?,
    );
    finish_assistant_response(context, final_message, added_partial, emit).await
}

async fn finish_assistant_response(
    context: &mut AgentContext,
    final_message: AssistantMessage,
    added_partial: bool,
    emit: &AgentEventSink,
) -> AgentResult<AssistantMessage> {
    if added_partial {
        if let Some(last) = context.messages.last_mut() {
            *last = Message::Assistant(final_message.clone());
        }
    } else {
        context
            .messages
            .push(Message::Assistant(final_message.clone()));
        emit(AgentEvent::MessageStart {
            message: Message::Assistant(final_message.clone()),
        })
        .await?;
    }
    emit(AgentEvent::MessageEnd {
        message: Message::Assistant(final_message.clone()),
    })
    .await?;
    Ok(final_message)
}

struct ExecutedToolCallBatch {
    messages: Vec<ToolResultMessage>,
    terminate: bool,
}

/// Fail all tool calls from an assistant message that was truncated by the
/// output token limit. Streamed tool-call arguments are finalized with a
/// best-effort JSON salvage parser, so a truncated message can yield tool
/// calls whose arguments parse and validate but are silently incomplete.
/// None of them are safe to execute; report each as an error so the model
/// can re-issue them.
async fn fail_tool_calls_from_truncated_message(
    tool_calls: Vec<AgentToolCall>,
    emit: &AgentEventSink,
) -> AgentResult<ExecutedToolCallBatch> {
    let mut messages = Vec::new();
    for tool_call in tool_calls {
        emit_tool_execution_start(&tool_call, emit).await?;
        let result = create_error_tool_result(format!(
            "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
            tool_call.name
        ));
        let finalized = AgentToolCallOutcome {
            tool_call,
            result,
            is_error: true,
        };
        emit_tool_execution_end(&finalized, emit).await?;
        let tool_result_message = create_tool_result_message(finalized);
        emit_tool_result_message(&tool_result_message, emit).await?;
        messages.push(tool_result_message);
    }
    Ok(ExecutedToolCallBatch {
        messages,
        terminate: false,
    })
}

/// Execute tool calls from an assistant message.
async fn execute_tool_calls(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    config: &AgentLoopConfig,
    signal: Option<CancellationToken>,
    emit: &AgentEventSink,
) -> AgentResult<ExecutedToolCallBatch> {
    let tool_calls = assistant_tool_calls(assistant_message);
    let has_sequential_tool_call = tool_calls.iter().any(|tool_call| {
        find_tool(&current_context.tools, &tool_call.name).and_then(|tool| tool.execution_mode())
            == Some(ToolExecutionMode::Sequential)
    });
    let hooks = ToolCallHooks {
        before_tool_call: config.before_tool_call.clone(),
        after_tool_call: config.after_tool_call.clone(),
    };
    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential_tool_call {
        execute_tool_calls_sequential(
            current_context,
            assistant_message,
            tool_calls,
            &hooks,
            signal,
            emit,
        )
        .await
    } else {
        execute_tool_calls_parallel(
            current_context,
            assistant_message,
            tool_calls,
            &hooks,
            signal,
            emit,
        )
        .await
    }
}

fn find_tool(tools: &[DynAgentTool], name: &str) -> Option<DynAgentTool> {
    tools
        .iter()
        .find(|tool| tool.definition().name == name)
        .cloned()
}

fn is_aborted(signal: &Option<CancellationToken>) -> bool {
    signal.as_ref().is_some_and(CancellationToken::is_cancelled)
}

async fn execute_tool_calls_sequential(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: Vec<AgentToolCall>,
    hooks: &ToolCallHooks,
    signal: Option<CancellationToken>,
    emit: &AgentEventSink,
) -> AgentResult<ExecutedToolCallBatch> {
    let mut finalized_calls = Vec::new();
    let mut messages = Vec::new();

    for tool_call in tool_calls {
        emit_tool_execution_start(&tool_call, emit).await?;

        let preparation = prepare_tool_call(
            current_context,
            assistant_message,
            &tool_call,
            hooks,
            signal.clone(),
            &current_context.tools,
        )
        .await;
        let finalized = match preparation {
            ToolCallPreparation::Immediate { result, is_error } => AgentToolCallOutcome {
                tool_call,
                result,
                is_error,
            },
            ToolCallPreparation::Prepared(prepared) => {
                let executed = execute_prepared_tool_call(
                    &prepared,
                    signal.clone(),
                    emit_tool_execution_update(&tool_call, emit),
                )
                .await?;
                finalize_executed_tool_call(
                    current_context,
                    assistant_message,
                    prepared,
                    executed,
                    hooks,
                    signal.clone(),
                )
                .await
            }
        };

        emit_tool_execution_end(&finalized, emit).await?;
        let terminate = finalized.result.terminate;
        let tool_result_message = create_tool_result_message(finalized);
        emit_tool_result_message(&tool_result_message, emit).await?;
        finalized_calls.push(terminate);
        messages.push(tool_result_message);

        if is_aborted(&signal) {
            break;
        }
    }

    Ok(ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_tool_batch(&finalized_calls),
    })
}

type FinalizedToolCallEntry<'a> = futures::future::BoxFuture<'a, AgentResult<AgentToolCallOutcome>>;

async fn execute_tool_calls_parallel(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: Vec<AgentToolCall>,
    hooks: &ToolCallHooks,
    signal: Option<CancellationToken>,
    emit: &AgentEventSink,
) -> AgentResult<ExecutedToolCallBatch> {
    let mut finalized_calls: Vec<FinalizedToolCallEntry<'_>> = Vec::new();

    for tool_call in tool_calls {
        emit_tool_execution_start(&tool_call, emit).await?;

        let preparation = prepare_tool_call(
            current_context,
            assistant_message,
            &tool_call,
            hooks,
            signal.clone(),
            &current_context.tools,
        )
        .await;
        let prepared = match preparation {
            ToolCallPreparation::Immediate { result, is_error } => {
                let finalized = AgentToolCallOutcome {
                    tool_call,
                    result,
                    is_error,
                };
                emit_tool_execution_end(&finalized, emit).await?;
                finalized_calls.push(Box::pin(async move { Ok(finalized) }));
                if is_aborted(&signal) {
                    break;
                }
                continue;
            }
            ToolCallPreparation::Prepared(prepared) => prepared,
        };

        let signal_for_call = signal.clone();
        finalized_calls.push(Box::pin(async move {
            if is_aborted(&signal_for_call) {
                let finalized = AgentToolCallOutcome {
                    tool_call,
                    result: create_error_tool_result("Operation aborted"),
                    is_error: true,
                };
                emit_tool_execution_end(&finalized, emit).await?;
                return Ok(finalized);
            }
            let executed = execute_prepared_tool_call(
                &prepared,
                signal_for_call.clone(),
                emit_tool_execution_update(&tool_call, emit),
            )
            .await?;
            let finalized = finalize_executed_tool_call(
                current_context,
                assistant_message,
                prepared,
                executed,
                hooks,
                signal_for_call,
            )
            .await;
            emit_tool_execution_end(&finalized, emit).await?;
            Ok(finalized)
        }));
        if is_aborted(&signal) {
            break;
        }
    }

    let ordered_finalized_calls = join_all(finalized_calls)
        .await
        .into_iter()
        .collect::<AgentResult<Vec<_>>>()?;
    let terminate = should_terminate_tool_batch(
        &ordered_finalized_calls
            .iter()
            .map(|finalized| finalized.result.terminate)
            .collect::<Vec<_>>(),
    );
    let mut messages = Vec::new();
    for finalized in ordered_finalized_calls {
        let tool_result_message = create_tool_result_message(finalized);
        emit_tool_result_message(&tool_result_message, emit).await?;
        messages.push(tool_result_message);
    }

    Ok(ExecutedToolCallBatch {
        messages,
        terminate,
    })
}

struct PreparedToolCall {
    tool_call: AgentToolCall,
    tool: DynAgentTool,
    args: Value,
}

enum ToolCallPreparation {
    Prepared(PreparedToolCall),
    Immediate {
        result: AgentToolResult,
        is_error: bool,
    },
}

struct ExecutedToolCallOutcome {
    result: AgentToolResult,
    is_error: bool,
}

/// The `before_tool_call` and `after_tool_call` hooks of [`AgentLoopConfig`].
#[derive(Clone, Default)]
pub struct ToolCallHooks {
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
}

/// Sink for partial tool results (`ToolUpdateSink`).
type ToolUpdateSink = Arc<dyn Fn(AgentToolResult) -> BoxFuture<AgentResult<()>> + Send + Sync>;

fn should_terminate_tool_batch(terminate_flags: &[bool]) -> bool {
    !terminate_flags.is_empty() && terminate_flags.iter().all(|terminate| *terminate)
}

fn prepare_tool_call_arguments(
    tool: &DynAgentTool,
    tool_call: &AgentToolCall,
) -> AgentResult<AgentToolCall> {
    let prepared_arguments = tool.prepare_arguments(tool_call.arguments.clone())?;
    Ok(AgentToolCall {
        arguments: prepared_arguments,
        ..tool_call.clone()
    })
}

async fn prepare_tool_call(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_call: &AgentToolCall,
    hooks: &ToolCallHooks,
    signal: Option<CancellationToken>,
    tools: &[DynAgentTool],
) -> ToolCallPreparation {
    let Some(tool) = find_tool(tools, &tool_call.name) else {
        return ToolCallPreparation::Immediate {
            result: create_error_tool_result(format!("Tool {} not found", tool_call.name)),
            is_error: true,
        };
    };

    let preparation = async {
        let prepared_tool_call = prepare_tool_call_arguments(&tool, tool_call)?;
        let mut args = validate_tool_arguments(&tool.definition(), &prepared_tool_call)?;
        if let Some(before_tool_call) = &hooks.before_tool_call {
            let shared_args = Arc::new(Mutex::new(args));
            let before_result = before_tool_call(
                BeforeToolCallContext {
                    assistant_message: assistant_message.clone(),
                    tool_call: tool_call.clone(),
                    args: Arc::clone(&shared_args),
                    context: current_context.clone(),
                },
                signal.clone(),
            )
            .await?;
            if is_aborted(&signal) {
                return Ok(ToolCallPreparation::Immediate {
                    result: create_error_tool_result("Operation aborted"),
                    is_error: true,
                });
            }
            if let Some(before_result) = before_result
                && before_result.block
            {
                let mut result = create_error_tool_result(
                    before_result
                        .reason
                        .filter(|reason| !reason.is_empty())
                        .unwrap_or_else(|| "Tool execution was blocked".to_string()),
                );
                if before_result.terminate {
                    result.terminate = true;
                }
                return Ok(ToolCallPreparation::Immediate {
                    result,
                    is_error: true,
                });
            }
            args = shared_args.lock().clone();
        }
        if is_aborted(&signal) {
            return Ok(ToolCallPreparation::Immediate {
                result: create_error_tool_result("Operation aborted"),
                is_error: true,
            });
        }
        Ok::<_, AgentError>(ToolCallPreparation::Prepared(PreparedToolCall {
            tool_call: tool_call.clone(),
            tool,
            args,
        }))
    };
    match preparation.await {
        Ok(preparation) => preparation,
        Err(error) => ToolCallPreparation::Immediate {
            result: create_error_tool_result(error.to_string()),
            is_error: true,
        },
    }
}

fn emit_tool_execution_update(tool_call: &AgentToolCall, emit: &AgentEventSink) -> ToolUpdateSink {
    let tool_call = tool_call.clone();
    let emit = Arc::clone(emit);
    Arc::new(move |partial_result| {
        emit(AgentEvent::ToolExecutionUpdate {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
            partial_result,
        })
    })
}

/// Callback for [`RunToolCallOptions::on_update`].
pub type ToolUpdateCallback = Arc<dyn Fn(AgentToolResult) -> BoxFuture<()> + Send + Sync>;

/// Options for [`run_tool_call`].
#[derive(Clone)]
pub struct RunToolCallOptions {
    /// Tools the call resolves against.
    pub tools: Vec<DynAgentTool>,
    /// Passed to the hooks as the message that issued the call.
    pub assistant_message: AssistantMessage,
    /// Passed to the hooks as the current agent context.
    pub context: AgentContext,
    pub signal: Option<CancellationToken>,
    pub on_update: Option<ToolUpdateCallback>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
}

impl RunToolCallOptions {
    pub fn new(
        tools: Vec<DynAgentTool>,
        assistant_message: AssistantMessage,
        context: AgentContext,
    ) -> Self {
        Self {
            tools,
            assistant_message,
            context,
            signal: None,
            on_update: None,
            before_tool_call: None,
            after_tool_call: None,
        }
    }
}

/// Run one tool call through the same steps as a model-issued call:
/// argument preparation, schema validation, `before_tool_call`, execution,
/// and `after_tool_call`. Emits no events and adds no messages. Tools that
/// call other tools use this so the hooks (for example permission checks)
/// apply to those calls too.
///
/// Never fails for tool failures: unknown tools, validation errors, blocked
/// calls, and tool errors come back as `is_error: true`.
pub async fn run_tool_call(
    tool_call: AgentToolCall,
    options: RunToolCallOptions,
) -> AgentToolCallOutcome {
    let hooks = ToolCallHooks {
        before_tool_call: options.before_tool_call.clone(),
        after_tool_call: options.after_tool_call.clone(),
    };
    let preparation = prepare_tool_call(
        &options.context,
        &options.assistant_message,
        &tool_call,
        &hooks,
        options.signal.clone(),
        &options.tools,
    )
    .await;
    let prepared = match preparation {
        ToolCallPreparation::Immediate { result, is_error } => {
            return AgentToolCallOutcome {
                tool_call,
                result,
                is_error,
            };
        }
        ToolCallPreparation::Prepared(prepared) => prepared,
    };
    let on_update = options.on_update.clone();
    let sink: ToolUpdateSink = Arc::new(move |partial_result| {
        let on_update = on_update.clone();
        Box::pin(async move {
            if let Some(on_update) = on_update {
                on_update(partial_result).await;
            }
            Ok(())
        })
    });
    let executed = match execute_prepared_tool_call(&prepared, options.signal.clone(), sink).await {
        Ok(executed) => executed,
        // The sink above never fails.
        Err(error) => ExecutedToolCallOutcome {
            result: create_error_tool_result(error.to_string()),
            is_error: true,
        },
    };
    finalize_executed_tool_call(
        &options.context,
        &options.assistant_message,
        prepared,
        executed,
        &hooks,
        options.signal,
    )
    .await
}

type PendingUpdate = (AgentToolResult, oneshot::Sender<()>);

async fn execute_prepared_tool_call(
    prepared: &PreparedToolCall,
    signal: Option<CancellationToken>,
    on_update: ToolUpdateSink,
) -> AgentResult<ExecutedToolCallOutcome> {
    let (update_sender, mut update_receiver) = mpsc::unbounded_channel::<PendingUpdate>();
    let accepting_updates = Arc::new(AtomicBool::new(true));
    let callback: AgentToolUpdateCallback = {
        let accepting_updates = Arc::clone(&accepting_updates);
        Arc::new(move |partial_result| {
            if !accepting_updates.load(Ordering::SeqCst) {
                return Box::pin(async {});
            }
            let (delivered_sender, delivered_receiver) = oneshot::channel();
            if update_sender
                .send((partial_result, delivered_sender))
                .is_err()
            {
                return Box::pin(async {});
            }
            Box::pin(async move {
                let _ = delivered_receiver.await;
            })
        })
    };

    let mut update_error = None;
    let mut deliver = async |(partial_result, delivered): PendingUpdate| {
        if let Err(error) = on_update(partial_result).await {
            update_error.get_or_insert(error);
        }
        let _ = delivered.send(());
    };

    let execution = prepared.tool.execute(
        &prepared.tool_call.id,
        prepared.args.clone(),
        signal,
        Some(callback),
    );
    tokio::pin!(execution);
    let result = loop {
        tokio::select! {
            biased;
            result = &mut execution => break result,
            Some(update) = update_receiver.recv() => deliver(update).await,
        }
    };
    accepting_updates.store(false, Ordering::SeqCst);
    while let Ok(update) = update_receiver.try_recv() {
        deliver(update).await;
    }
    if let Some(error) = update_error {
        return Err(error);
    }

    Ok(match result {
        Ok(result) => {
            let is_error = result.is_error;
            ExecutedToolCallOutcome { result, is_error }
        }
        Err(error) => ExecutedToolCallOutcome {
            result: create_error_tool_result(error.to_string()),
            is_error: true,
        },
    })
}

fn non_null(value: Option<Value>) -> Option<Value> {
    value.filter(|value| !value.is_null())
}

async fn finalize_executed_tool_call(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    prepared: PreparedToolCall,
    executed: ExecutedToolCallOutcome,
    hooks: &ToolCallHooks,
    signal: Option<CancellationToken>,
) -> AgentToolCallOutcome {
    let mut result = executed.result;
    let mut is_error = executed.is_error;

    if let Some(after_tool_call) = &hooks.after_tool_call {
        let after_result = after_tool_call(
            AfterToolCallContext {
                assistant_message: assistant_message.clone(),
                tool_call: prepared.tool_call.clone(),
                args: prepared.args.clone(),
                result: result.clone(),
                is_error,
                context: current_context.clone(),
            },
            signal,
        )
        .await;
        match after_result {
            Ok(Some(after_result)) => {
                // Structured content not replaced along with the content may no longer match it.
                let structured_content = non_null(after_result.structured_content).or_else(|| {
                    if after_result.content.is_some() {
                        None
                    } else {
                        result.structured_content.clone()
                    }
                });
                if let Some(content) = after_result.content {
                    result.content = content;
                }
                if let Some(details) = non_null(after_result.details) {
                    result.details = Some(details);
                }
                if let Some(usage) = after_result.usage {
                    result.usage = Some(usage);
                }
                if let Some(terminate) = after_result.terminate {
                    result.terminate = terminate;
                }
                result.structured_content = structured_content;
                if let Some(next_is_error) = after_result.is_error {
                    is_error = next_is_error;
                }
            }
            Ok(None) => {}
            Err(error) => {
                result = create_error_tool_result(error.to_string());
                is_error = true;
            }
        }
    }

    AgentToolCallOutcome {
        tool_call: prepared.tool_call,
        result,
        is_error,
    }
}

fn create_error_tool_result(message: impl Into<String>) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContent::text(message)],
        details: Some(json!({})),
        ..Default::default()
    }
}

async fn emit_tool_execution_start(
    tool_call: &AgentToolCall,
    emit: &AgentEventSink,
) -> AgentResult<()> {
    emit(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        args: tool_call.arguments.clone(),
    })
    .await
}

async fn emit_tool_execution_end(
    finalized: &AgentToolCallOutcome,
    emit: &AgentEventSink,
) -> AgentResult<()> {
    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        result: finalized.result.clone(),
        is_error: finalized.is_error,
    })
    .await
}

fn create_tool_result_message(finalized: AgentToolCallOutcome) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: finalized.tool_call.id,
        tool_name: finalized.tool_call.name,
        content: finalized.result.content,
        details: finalized.result.details,
        usage: finalized.result.usage,
        nested_calls: None,
        is_error: finalized.is_error,
        timestamp: now_millis(),
    }
}

async fn emit_tool_result_message(
    tool_result_message: &ToolResultMessage,
    emit: &AgentEventSink,
) -> AgentResult<()> {
    let message = Message::ToolResult(tool_result_message.clone());
    emit(AgentEvent::MessageStart {
        message: message.clone(),
    })
    .await?;
    emit(AgentEvent::MessageEnd { message }).await
}

#[cfg(test)]
#[path = "agent_loop_tests.rs"]
pub(super) mod tests;
