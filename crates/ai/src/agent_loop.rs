use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};
use std::task::{Context as TaskContext, Poll};

use crate::{AssistantMessageEvent, StopReason, ToolResultMessage};
use futures::{Stream, StreamExt};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::agent_types::{
    AfterToolCallContext, AgentContext, AgentEvent, AgentEventSink, AgentLoopConfig, AgentMessage,
    AgentToolResult, BeforeToolCallContext, DynAgentTool, ToolExecutionMode, assistant_tool_calls,
};
use crate::{AgentError, AgentResult};

pub struct AgentEventStream {
    receiver: mpsc::UnboundedReceiver<AgentEvent>,
    result_receiver: Option<oneshot::Receiver<AgentResult<Vec<AgentMessage>>>>,
    result: Option<Vec<AgentMessage>>,
}

impl AgentEventStream {
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

pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    cancellation_token: Option<CancellationToken>,
    stream_fn: Option<crate::agent_types::StreamFn>,
) -> AgentEventStream {
    let (event_sender, receiver) = mpsc::unbounded_channel();
    let (result_sender, result_receiver) = oneshot::channel();
    let emit: AgentEventSink = Arc::new(move |event| {
        let event_sender = event_sender.clone();
        Box::pin(async move {
            let _ = event_sender.send(event);
            Ok(())
        })
    });

    tokio::spawn(async move {
        let result = run_agent_loop(
            prompts,
            context,
            config,
            emit,
            cancellation_token,
            stream_fn,
        )
        .await;
        let _ = result_sender.send(result);
    });

    AgentEventStream {
        receiver,
        result_receiver: Some(result_receiver),
        result: None,
    }
}

pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancellation_token: Option<CancellationToken>,
    stream_fn: Option<crate::agent_types::StreamFn>,
) -> AgentResult<AgentEventStream> {
    if context.messages.is_empty() {
        return Err(AgentError::NoMessagesToContinue);
    }
    if matches!(context.messages.last(), Some(crate::Message::Assistant(_))) {
        return Err(AgentError::CannotContinueFromAssistant);
    }

    let (event_sender, receiver) = mpsc::unbounded_channel();
    let (result_sender, result_receiver) = oneshot::channel();
    let emit: AgentEventSink = Arc::new(move |event| {
        let event_sender = event_sender.clone();
        Box::pin(async move {
            let _ = event_sender.send(event);
            Ok(())
        })
    });

    tokio::spawn(async move {
        let result =
            run_agent_loop_continue(context, config, emit, cancellation_token, stream_fn).await;
        let _ = result_sender.send(result);
    });

    Ok(AgentEventStream {
        receiver,
        result_receiver: Some(result_receiver),
        result: None,
    })
}

pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    mut context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
    stream_fn: Option<crate::agent_types::StreamFn>,
) -> AgentResult<Vec<AgentMessage>> {
    let mut new_messages = prompts.clone();
    context.messages.extend(prompts.clone());

    emit(AgentEvent::AgentStart).await?;
    emit(AgentEvent::TurnStart).await?;
    for prompt in prompts {
        emit(AgentEvent::MessageStart {
            message: prompt.clone(),
        })
        .await?;
        emit(AgentEvent::MessageEnd { message: prompt }).await?;
    }

    run_loop(
        &mut context,
        &mut new_messages,
        config,
        emit.clone(),
        cancellation_token,
        stream_fn,
    )
    .await?;
    emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await?;
    Ok(new_messages)
}

pub async fn run_agent_loop_continue(
    mut context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
    stream_fn: Option<crate::agent_types::StreamFn>,
) -> AgentResult<Vec<AgentMessage>> {
    if context.messages.is_empty() {
        return Err(AgentError::NoMessagesToContinue);
    }
    if matches!(context.messages.last(), Some(crate::Message::Assistant(_))) {
        return Err(AgentError::CannotContinueFromAssistant);
    }

    let mut new_messages = Vec::new();
    emit(AgentEvent::AgentStart).await?;
    emit(AgentEvent::TurnStart).await?;
    run_loop(
        &mut context,
        &mut new_messages,
        config,
        emit.clone(),
        cancellation_token,
        stream_fn,
    )
    .await?;
    emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await?;
    Ok(new_messages)
}

async fn run_loop(
    context: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    mut config: AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
    stream_fn: Option<crate::agent_types::StreamFn>,
) -> AgentResult<()> {
    let mut first_turn = true;
    let mut pending_messages = if let Some(get) = &config.get_steering_messages {
        get().await
    } else {
        Vec::new()
    };

    loop {
        let mut has_more_tool_calls = true;
        while has_more_tool_calls || !pending_messages.is_empty() {
            if first_turn {
                first_turn = false;
            } else {
                emit(AgentEvent::TurnStart).await?;
            }

            for message in std::mem::take(&mut pending_messages) {
                emit(AgentEvent::MessageStart {
                    message: message.clone(),
                })
                .await?;
                emit(AgentEvent::MessageEnd {
                    message: message.clone(),
                })
                .await?;
                context.messages.push(message.clone());
                new_messages.push(message);
            }

            let assistant = stream_assistant_response(
                context,
                &config,
                &emit,
                cancellation_token.clone(),
                stream_fn.clone(),
            )
            .await?;
            new_messages.push(crate::Message::Assistant(assistant.clone()));

            if matches!(
                assistant.stop_reason,
                StopReason::Error | StopReason::Aborted
            ) {
                emit(AgentEvent::TurnEnd {
                    message: crate::Message::Assistant(assistant),
                    tool_results: Vec::new(),
                })
                .await?;
                return Ok(());
            }

            let tool_calls = assistant_tool_calls(&assistant);
            let mut tool_results = Vec::new();
            has_more_tool_calls = false;
            if !tool_calls.is_empty() {
                let executed = execute_tool_calls(
                    context,
                    &assistant,
                    &config,
                    emit.clone(),
                    cancellation_token.clone(),
                )
                .await?;
                has_more_tool_calls = !executed.terminate;
                tool_results.extend(executed.messages);
                for result in &tool_results {
                    context
                        .messages
                        .push(crate::Message::ToolResult(result.clone()));
                    new_messages.push(crate::Message::ToolResult(result.clone()));
                }
            }

            emit(AgentEvent::TurnEnd {
                message: crate::Message::Assistant(assistant.clone()),
                tool_results: tool_results.clone(),
            })
            .await?;

            if let Some(prepare_next_turn) = config.prepare_next_turn.clone()
                && let Some(update) = prepare_next_turn(
                    crate::agent_types::PrepareNextTurnContext {
                        message: assistant.clone(),
                        tool_results: tool_results.clone(),
                        context: context.clone(),
                        new_messages: new_messages.clone(),
                    },
                    cancellation_token.clone(),
                )
                .await
            {
                if let Some(next_context) = update.context {
                    *context = next_context;
                }
                if let Some(next_model) = update.model {
                    config.model = next_model;
                }
                if let Some(reasoning_level) = update.reasoning_level {
                    config.options.reasoning = if reasoning_level == crate::ModelThinkingLevel::Off
                    {
                        None
                    } else {
                        Some(reasoning_level)
                    };
                }
            }

            if let Some(should_stop) = &config.should_stop_after_turn
                && should_stop(crate::agent_types::ShouldStopAfterTurnContext {
                    message: assistant,
                    tool_results,
                    context: context.clone(),
                    new_messages: new_messages.clone(),
                })
                .await
            {
                return Ok(());
            }

            pending_messages = if let Some(get) = &config.get_steering_messages {
                get().await
            } else {
                Vec::new()
            };
        }

        let follow_ups = if let Some(get) = &config.get_follow_up_messages {
            get().await
        } else {
            Vec::new()
        };
        if follow_ups.is_empty() {
            break;
        }
        pending_messages = follow_ups;
    }

    Ok(())
}

async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    emit: &AgentEventSink,
    cancellation_token: Option<CancellationToken>,
    stream_fn: Option<crate::agent_types::StreamFn>,
) -> AgentResult<crate::AssistantMessage> {
    let mut messages = context.messages.clone();
    if let Some(transform) = &config.transform_context {
        messages = transform(messages, cancellation_token.clone()).await;
    }
    let llm_messages = if let Some(convert) = &config.convert_to_llm {
        convert(messages).await
    } else {
        messages
            .into_iter()
            .filter(|message| message.is_llm_message())
            .collect()
    };

    let mut llm_context = context.llm_context();
    llm_context.messages = llm_messages;

    let mut options = config.options.clone();
    options.stream.cancellation_token = cancellation_token.clone();
    if let Some(get_api_key) = &config.get_api_key
        && let Some(api_key) = get_api_key(config.model.provider.clone())
            .await
            .filter(|key| !key.is_empty())
    {
        options.stream.api_key = Some(api_key);
    }

    let mut response = if let Some(stream_fn) = stream_fn {
        stream_fn(config.model.clone(), llm_context, options).await?
    } else {
        crate::stream_simple(config.model.clone(), llm_context, Some(options))?
    };

    let mut partial_added = false;
    while let Some(event) = response.next().await {
        match &event {
            AssistantMessageEvent::Start { partial } => {
                context
                    .messages
                    .push(crate::Message::Assistant(partial.clone()));
                partial_added = true;
                emit(AgentEvent::MessageStart {
                    message: crate::Message::Assistant(partial.clone()),
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
                if partial_added {
                    if let Some(last) = context.messages.last_mut() {
                        *last = crate::Message::Assistant(partial.clone());
                    }
                    emit(AgentEvent::MessageUpdate {
                        message: crate::Message::Assistant(partial.clone()),
                        assistant_message_event: event.clone(),
                    })
                    .await?;
                }
            }
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {
                let final_message = response.result().await?;
                if partial_added {
                    if let Some(last) = context.messages.last_mut() {
                        *last = crate::Message::Assistant(final_message.clone());
                    }
                } else {
                    context
                        .messages
                        .push(crate::Message::Assistant(final_message.clone()));
                    emit(AgentEvent::MessageStart {
                        message: crate::Message::Assistant(final_message.clone()),
                    })
                    .await?;
                }
                emit(AgentEvent::MessageEnd {
                    message: crate::Message::Assistant(final_message.clone()),
                })
                .await?;
                return Ok(final_message);
            }
        }
    }

    let final_message = response.result().await?;
    if partial_added {
        if let Some(last) = context.messages.last_mut() {
            *last = crate::Message::Assistant(final_message.clone());
        }
    } else {
        context
            .messages
            .push(crate::Message::Assistant(final_message.clone()));
        emit(AgentEvent::MessageStart {
            message: crate::Message::Assistant(final_message.clone()),
        })
        .await?;
    }
    emit(AgentEvent::MessageEnd {
        message: crate::Message::Assistant(final_message.clone()),
    })
    .await?;
    Ok(final_message)
}

struct ExecutedToolBatch {
    messages: Vec<ToolResultMessage>,
    terminate: bool,
}

async fn execute_tool_calls(
    context: &AgentContext,
    assistant: &crate::AssistantMessage,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> AgentResult<ExecutedToolBatch> {
    let tool_calls = assistant_tool_calls(assistant);
    let has_sequential = tool_calls.iter().any(|tool_call| {
        context
            .tools
            .iter()
            .find(|tool| tool.definition().name == tool_call.name)
            .and_then(|tool| tool.execution_mode())
            == Some(ToolExecutionMode::Sequential)
    });
    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential {
        execute_tool_calls_sequential(
            context,
            assistant,
            tool_calls,
            config,
            emit,
            cancellation_token,
        )
        .await
    } else {
        execute_tool_calls_parallel(
            context,
            assistant,
            tool_calls,
            config,
            emit,
            cancellation_token,
        )
        .await
    }
}

async fn execute_tool_calls_sequential(
    context: &AgentContext,
    assistant: &crate::AssistantMessage,
    tool_calls: Vec<crate::ToolCall>,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> AgentResult<ExecutedToolBatch> {
    let mut results = Vec::new();
    let mut terminate_flags = Vec::new();
    for tool_call in tool_calls {
        let finalized = execute_one_tool(
            context,
            assistant,
            tool_call,
            config,
            emit.clone(),
            cancellation_token.clone(),
        )
        .await?;
        terminate_flags.push(finalized.2.terminate);
        let result_message = create_tool_result_message(finalized);
        emit_tool_result_message(&result_message, &emit).await?;
        results.push(result_message);
        if cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            break;
        }
    }
    Ok(ExecutedToolBatch {
        messages: results,
        terminate: !terminate_flags.is_empty() && terminate_flags.iter().all(|flag| *flag),
    })
}

async fn execute_tool_calls_parallel(
    context: &AgentContext,
    assistant: &crate::AssistantMessage,
    tool_calls: Vec<crate::ToolCall>,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> AgentResult<ExecutedToolBatch> {
    let mut finalized_by_source = vec![None; tool_calls.len()];
    let mut prepared_calls = Vec::new();
    for (source_index, tool_call) in tool_calls.into_iter().enumerate() {
        match prepare_tool_call(
            context,
            assistant,
            source_index,
            tool_call,
            config,
            emit.clone(),
            cancellation_token.clone(),
        )
        .await?
        {
            PreparedToolCallOutcome::Immediate(finalized) => {
                emit_tool_execution_end(&finalized, &emit).await?;
                finalized_by_source[source_index] = Some(finalized);
                if cancellation_token
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled)
                {
                    break;
                }
            }
            PreparedToolCallOutcome::Prepared(prepared) => {
                prepared_calls.push(prepared);
                if cancellation_token
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled)
                {
                    break;
                }
            }
        }
    }

    let mut futures = futures::stream::FuturesUnordered::new();
    for prepared in prepared_calls {
        let context = context.clone();
        let assistant = assistant.clone();
        let config = config.clone();
        let emit = emit.clone();
        let cancellation_token = cancellation_token.clone();
        futures.push(async move {
            let source_index = prepared.source_index;
            execute_prepared_tool_call(
                &context,
                &assistant,
                prepared,
                &config,
                emit,
                cancellation_token,
            )
            .await
            .map(|finalized| (source_index, finalized))
        });
    }
    while let Some(item) = futures.next().await {
        let (source_index, finalized) = item?;
        finalized_by_source[source_index] = Some(finalized);
    }

    let mut messages = Vec::new();
    let mut terminate_flags = Vec::new();
    for finalized in finalized_by_source.into_iter().flatten() {
        terminate_flags.push(finalized.2.terminate);
        messages.push(create_tool_result_message(finalized));
    }
    for message in &messages {
        emit_tool_result_message(message, &emit).await?;
    }
    Ok(ExecutedToolBatch {
        messages,
        terminate: !terminate_flags.is_empty() && terminate_flags.iter().all(|flag| *flag),
    })
}

type FinalizedToolCall = (crate::ToolCall, bool, AgentToolResult);

struct PreparedToolCall {
    source_index: usize,
    tool_call: crate::ToolCall,
    tool: DynAgentTool,
    args: Value,
}

enum PreparedToolCallOutcome {
    Immediate(FinalizedToolCall),
    Prepared(PreparedToolCall),
}

async fn execute_one_tool(
    context: &AgentContext,
    assistant: &crate::AssistantMessage,
    tool_call: crate::ToolCall,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> AgentResult<FinalizedToolCall> {
    match prepare_tool_call(
        context,
        assistant,
        0,
        tool_call,
        config,
        emit.clone(),
        cancellation_token.clone(),
    )
    .await?
    {
        PreparedToolCallOutcome::Immediate(finalized) => {
            emit_tool_execution_end(&finalized, &emit).await?;
            Ok(finalized)
        }
        PreparedToolCallOutcome::Prepared(prepared) => {
            execute_prepared_tool_call(
                context,
                assistant,
                prepared,
                config,
                emit,
                cancellation_token,
            )
            .await
        }
    }
}

async fn prepare_tool_call(
    context: &AgentContext,
    assistant: &crate::AssistantMessage,
    source_index: usize,
    tool_call: crate::ToolCall,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> AgentResult<PreparedToolCallOutcome> {
    emit(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        args: tool_call.arguments.clone(),
    })
    .await?;

    let Some(tool) = context
        .tools
        .iter()
        .find(|tool| tool.definition().name == tool_call.name)
        .cloned()
    else {
        let tool_name = tool_call.name.clone();
        return Ok(PreparedToolCallOutcome::Immediate(finalized_error(
            tool_call,
            format!("Tool {tool_name} not found"),
        )));
    };

    let mut prepared_args = match tool.prepare_arguments(tool_call.arguments.clone()) {
        Ok(args) => args,
        Err(error) => {
            return Ok(PreparedToolCallOutcome::Immediate(finalized_error(
                tool_call,
                error.to_string(),
            )));
        }
    };

    let mut prepared_tool_call = tool_call.clone();
    prepared_tool_call.arguments = prepared_args;
    prepared_args = match crate::utils::validation::validate_tool_arguments(
        &tool.definition(),
        &prepared_tool_call,
    ) {
        Ok(args) => args,
        Err(error) => {
            return Ok(PreparedToolCallOutcome::Immediate(finalized_error(
                tool_call,
                error.to_string(),
            )));
        }
    };

    if let Some(before_tool_call) = &config.before_tool_call {
        let shared_args = Arc::new(StdMutex::new(prepared_args.clone()));
        match before_tool_call(
            BeforeToolCallContext {
                assistant_message: assistant.clone(),
                tool_call: tool_call.clone(),
                args: Arc::clone(&shared_args),
                context: context.clone(),
            },
            cancellation_token.clone(),
        )
        .await
        {
            Ok(Some(before_result)) => {
                if cancellation_token
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled)
                {
                    return Ok(PreparedToolCallOutcome::Immediate(finalized_error(
                        tool_call,
                        "Operation aborted",
                    )));
                }
                if before_result.block {
                    let reason = before_result
                        .reason
                        .unwrap_or_else(|| "Tool execution was blocked".to_string());
                    return Ok(PreparedToolCallOutcome::Immediate(finalized_error(
                        tool_call, reason,
                    )));
                }
            }
            Ok(None) => {}
            Err(error) => {
                return Ok(PreparedToolCallOutcome::Immediate(finalized_error(
                    tool_call,
                    error.to_string(),
                )));
            }
        }
        prepared_args = shared_args
            .lock()
            .map_err(|error| AgentError::Other(format!("before tool args lock poisoned: {error}")))?
            .clone();
    }

    if cancellation_token
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Ok(PreparedToolCallOutcome::Immediate(finalized_error(
            tool_call,
            "Operation aborted",
        )));
    }

    Ok(PreparedToolCallOutcome::Prepared(PreparedToolCall {
        source_index,
        tool_call,
        tool,
        args: prepared_args,
    }))
}

async fn execute_prepared_tool_call(
    context: &AgentContext,
    assistant: &crate::AssistantMessage,
    prepared: PreparedToolCall,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> AgentResult<FinalizedToolCall> {
    let tool = prepared.tool;
    let tool_call = prepared.tool_call;
    let prepared_args = prepared.args;
    let emit_for_update = emit.clone();
    let update_tool_call = tool_call.clone();
    let update_tasks = Arc::new(StdMutex::new(Vec::new()));
    let update_tasks_for_callback = update_tasks.clone();
    let on_update = Arc::new(move |partial_result: AgentToolResult| {
        let emit = emit_for_update.clone();
        let tool_call = update_tool_call.clone();
        let (done_sender, done_receiver) = oneshot::channel();
        let handle = tokio::spawn(async move {
            let result = emit(AgentEvent::ToolExecutionUpdate {
                tool_call_id: tool_call.id,
                tool_name: tool_call.name,
                args: tool_call.arguments,
                partial_result,
            })
            .await;
            let _ = done_sender.send(());
            result
        });
        if let Ok(mut tasks) = update_tasks_for_callback.lock() {
            tasks.push(handle);
        }
        Box::pin(async move {
            let _ = done_receiver.await;
        }) as Pin<Box<dyn std::future::Future<Output = ()> + Send>>
    });

    let (mut is_error, mut result) = match tool
        .execute(
            &tool_call.id,
            prepared_args.clone(),
            cancellation_token.clone(),
            Some(on_update),
        )
        .await
    {
        Ok(result) => (false, result),
        Err(error) => (true, error_tool_result(error.to_string())),
    };

    await_tool_update_tasks(&update_tasks).await?;

    if let Some(after_tool_call) = &config.after_tool_call {
        match after_tool_call(
            AfterToolCallContext {
                assistant_message: assistant.clone(),
                tool_call: tool_call.clone(),
                args: prepared_args,
                result: result.clone(),
                is_error,
                context: context.clone(),
            },
            cancellation_token.clone(),
        )
        .await
        {
            Ok(Some(after_result)) => {
                if let Some(content) = after_result.content {
                    result.content = content;
                }
                if let Some(details) = after_result.details.filter(|details| !details.is_null()) {
                    result.details = Some(details);
                }
                if let Some(terminate) = after_result.terminate {
                    result.terminate = terminate;
                }
                if let Some(next_is_error) = after_result.is_error {
                    is_error = next_is_error;
                }
            }
            Ok(None) => {}
            Err(error) => {
                result = error_tool_result(error.to_string());
                is_error = true;
            }
        }
    }

    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        result: result.clone(),
        is_error,
    })
    .await?;
    Ok((tool_call, is_error, result))
}

async fn await_tool_update_tasks(
    tasks: &Arc<StdMutex<Vec<tokio::task::JoinHandle<AgentResult<()>>>>>,
) -> AgentResult<()> {
    let handles = match tasks.lock() {
        Ok(mut tasks) => tasks.drain(..).collect::<Vec<_>>(),
        Err(error) => {
            return Err(AgentError::Other(format!(
                "tool update task lock poisoned: {error}"
            )));
        }
    };
    for handle in handles {
        match handle.await {
            Ok(result) => result?,
            Err(error) => {
                return Err(AgentError::Other(format!(
                    "tool update task failed: {error}"
                )));
            }
        }
    }
    Ok(())
}

fn finalized_error(tool_call: crate::ToolCall, message: impl Into<String>) -> FinalizedToolCall {
    (tool_call, true, error_tool_result(message))
}

fn error_tool_result(message: impl Into<String>) -> AgentToolResult {
    let mut result = AgentToolResult::text(message);
    result.details = Some(json!({}));
    result
}

async fn emit_tool_execution_end(
    finalized: &FinalizedToolCall,
    emit: &AgentEventSink,
) -> AgentResult<()> {
    let (tool_call, is_error, result) = finalized;
    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        result: result.clone(),
        is_error: *is_error,
    })
    .await
}

fn create_tool_result_message(
    finalized: (crate::ToolCall, bool, AgentToolResult),
) -> ToolResultMessage {
    let (tool_call, is_error, result) = finalized;
    ToolResultMessage {
        tool_call_id: tool_call.id,
        tool_name: tool_call.name,
        content: result.content,
        details: result.details,
        is_error,
        timestamp: crate::utils::time::now_millis(),
    }
}

async fn emit_tool_result_message(
    message: &ToolResultMessage,
    emit: &AgentEventSink,
) -> AgentResult<()> {
    let message = crate::Message::ToolResult(message.clone());
    emit(AgentEvent::MessageStart {
        message: message.clone(),
    })
    .await?;
    emit(AgentEvent::MessageEnd { message }).await
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::FutureExt;
    use serde_json::{Value, json};
    use tokio_util::sync::CancellationToken;

    use super::run_agent_loop;
    use super::run_agent_loop_continue;
    use crate::agent_types::{
        AfterToolCallContext, AfterToolCallResult, AgentContext, AgentEvent, AgentEventSink,
        AgentLoopConfig, AgentLoopTurnUpdate, AgentTool, AgentToolResult, AgentToolUpdateCallback,
        BeforeToolCallContext, BeforeToolCallResult, StreamFn, ToolExecutionMode,
    };
    use crate::event_stream::create_assistant_message_event_stream;
    use crate::providers::faux::{
        FauxAssistantMessageOptions, faux_assistant_message, faux_tool_call, register_faux_provider,
    };
    use crate::{
        AssistantContent, AssistantMessage, AssistantMessageEvent, Message, Model, StopReason,
        TextContent, Tool, ToolResultContent, ToolResultMessage, UserMessageContent,
    };

    fn collect_events() -> (Arc<StdMutex<Vec<AgentEvent>>>, AgentEventSink) {
        let events = Arc::new(StdMutex::new(Vec::new()));
        let sink_events = Arc::clone(&events);
        let sink = Arc::new(move |event: AgentEvent| {
            let sink_events = Arc::clone(&sink_events);
            Box::pin(async move {
                sink_events.lock().unwrap().push(event);
                Ok(())
            })
                as Pin<Box<dyn std::future::Future<Output = crate::AgentResult<()>> + Send>>
        });
        (events, sink)
    }

    fn text_from_tool_result(message: &ToolResultMessage) -> String {
        message
            .content
            .iter()
            .filter_map(|content| match content {
                ToolResultContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn user_text(text: &str) -> Message {
        Message::User(crate::UserMessage {
            content: UserMessageContent::Text(text.to_string()),
            timestamp: 0,
        })
    }

    fn user_text_value(message: &Message) -> Option<&str> {
        let Message::User(user) = message else {
            return None;
        };
        match &user.content {
            UserMessageContent::Text(text) => Some(text.as_str()),
            UserMessageContent::Parts(parts) => parts.iter().find_map(|part| match part {
                crate::UserContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            }),
        }
    }

    fn assistant_text_message(model: &Model, text: impl Into<String>) -> AssistantMessage {
        let mut message = AssistantMessage::empty_for(model);
        message.content = vec![AssistantContent::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })];
        message
    }

    fn immediate_stream_fn(text: impl Into<String>) -> StreamFn {
        let text = Arc::new(text.into());
        Arc::new(move |model, _context, _options| {
            let text = Arc::clone(&text);
            async move {
                let (mut sender, stream) = create_assistant_message_event_stream();
                sender.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message: assistant_text_message(&model, text.as_str()),
                });
                Ok(stream)
            }
            .boxed()
        })
    }

    struct EditTool {
        executed: Arc<StdMutex<Vec<Value>>>,
    }

    #[async_trait]
    impl AgentTool for EditTool {
        fn definition(&self) -> Tool {
            Tool {
                name: "edit".to_string(),
                description: "Apply edits.".to_string(),
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

        fn prepare_arguments(&self, args: Value) -> crate::AgentResult<Value> {
            let Some(old_text) = args.get("oldText").and_then(Value::as_str) else {
                return Ok(args);
            };
            let Some(new_text) = args.get("newText").and_then(Value::as_str) else {
                return Ok(args);
            };
            Ok(json!({
                "edits": [{
                    "oldText": old_text,
                    "newText": new_text
                }]
            }))
        }

        async fn execute(
            &self,
            _tool_call_id: &str,
            args: Value,
            _cancellation_token: Option<CancellationToken>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> crate::AgentResult<AgentToolResult> {
            self.executed.lock().unwrap().push(args);
            Ok(AgentToolResult::text("edited"))
        }
    }

    struct EchoTool {
        executed: Arc<StdMutex<Vec<String>>>,
        delay_first: bool,
        execution_mode: Option<ToolExecutionMode>,
        terminate: bool,
    }

    #[async_trait]
    impl AgentTool for EchoTool {
        fn definition(&self) -> Tool {
            Tool {
                name: "echo".to_string(),
                description: "Echo a value.".to_string(),
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
            self.execution_mode
        }

        async fn execute(
            &self,
            _tool_call_id: &str,
            args: Value,
            _cancellation_token: Option<CancellationToken>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> crate::AgentResult<AgentToolResult> {
            let value = args
                .get("value")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if self.delay_first && value == "first" {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            self.executed.lock().unwrap().push(value.clone());
            let mut result = AgentToolResult::text(format!("echoed: {value}"));
            result.terminate = self.terminate;
            Ok(result)
        }
    }

    fn echo_tool(executed: Arc<StdMutex<Vec<String>>>) -> EchoTool {
        EchoTool {
            executed,
            delay_first: false,
            execution_mode: None,
            terminate: false,
        }
    }

    fn tool_result_message_ids(events: &[AgentEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::MessageEnd {
                    message: Message::ToolResult(tool_result),
                } => Some(tool_result.tool_call_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
    }

    struct MutatedArgsTool {
        executed: Arc<StdMutex<Vec<Value>>>,
    }

    #[async_trait]
    impl AgentTool for MutatedArgsTool {
        fn definition(&self) -> Tool {
            Tool {
                name: "echo".to_string(),
                description: "Echo a value.".to_string(),
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
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> crate::AgentResult<AgentToolResult> {
            self.executed
                .lock()
                .unwrap()
                .push(args.get("value").cloned().unwrap_or(Value::Null));
            Ok(AgentToolResult::text("echoed"))
        }
    }

    #[tokio::test]
    async fn prepare_arguments_runs_before_validation_like_upstream() {
        let registration = register_faux_provider(None);
        registration.set_responses([
            faux_assistant_message(
                vec![faux_tool_call(
                    "edit",
                    json!({ "oldText": "before", "newText": "after" }),
                    Some("tool-1".to_string()),
                )],
                Some(FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                }),
            ),
            faux_assistant_message("done", None),
        ]);
        let executed = Arc::new(StdMutex::new(Vec::new()));
        let context = AgentContext {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: vec![Arc::new(EditTool {
                executed: Arc::clone(&executed),
            })],
        };
        let (_events, emit) = collect_events();

        run_agent_loop(
            vec![user_text("edit something")],
            context,
            AgentLoopConfig::new(registration.get_model()),
            emit,
            None,
            None,
        )
        .await
        .expect("loop succeeds");

        assert_eq!(
            *executed.lock().unwrap(),
            vec![json!({ "edits": [{ "oldText": "before", "newText": "after" }] })]
        );
        registration.unregister();
    }

    #[tokio::test]
    async fn before_tool_call_blocks_without_executing_like_upstream() {
        let registration = register_faux_provider(None);
        registration.set_responses([
            faux_assistant_message(
                vec![faux_tool_call(
                    "echo",
                    json!({ "value": "hello" }),
                    Some("tool-1".to_string()),
                )],
                Some(FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                }),
            ),
            faux_assistant_message("done", None),
        ]);
        let executed = Arc::new(StdMutex::new(Vec::new()));
        let mut config = AgentLoopConfig::new(registration.get_model());
        config.before_tool_call = Some(Arc::new(|_context: BeforeToolCallContext, _token| {
            Box::pin(async {
                Ok(Some(BeforeToolCallResult {
                    block: true,
                    reason: Some("blocked by test".to_string()),
                }))
            })
        }));
        let context = AgentContext {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                executed: Arc::clone(&executed),
                delay_first: false,
                execution_mode: None,
                terminate: false,
            })],
        };
        let (events, emit) = collect_events();

        let messages = run_agent_loop(
            vec![user_text("echo something")],
            context,
            config,
            emit,
            None,
            None,
        )
        .await
        .expect("loop succeeds");

        assert!(executed.lock().unwrap().is_empty());
        let tool_result = messages
            .iter()
            .find_map(|message| match message {
                Message::ToolResult(tool_result) => Some(tool_result),
                _ => None,
            })
            .expect("tool result");
        assert!(tool_result.is_error);
        assert_eq!(text_from_tool_result(tool_result), "blocked by test");
        assert!(
            events
                .lock()
                .unwrap()
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolExecutionEnd { is_error: true, .. }))
        );
        registration.unregister();
    }

    #[tokio::test]
    async fn before_tool_call_mutated_args_execute_without_revalidation_like_upstream() {
        let registration = register_faux_provider(None);
        registration.set_responses([
            faux_assistant_message(
                vec![faux_tool_call(
                    "echo",
                    json!({ "value": "hello" }),
                    Some("tool-1".to_string()),
                )],
                Some(FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                }),
            ),
            faux_assistant_message("done", None),
        ]);
        let executed = Arc::new(StdMutex::new(Vec::new()));
        let mut config = AgentLoopConfig::new(registration.get_model());
        config.before_tool_call = Some(Arc::new(|context: BeforeToolCallContext, _token| {
            Box::pin(async move {
                context.args.lock().unwrap()["value"] = json!(123);
                Ok(None)
            })
        }));
        let context = AgentContext {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: vec![Arc::new(MutatedArgsTool {
                executed: Arc::clone(&executed),
            })],
        };
        let (_events, emit) = collect_events();

        run_agent_loop(
            vec![user_text("echo something")],
            context,
            config,
            emit,
            None,
            None,
        )
        .await
        .expect("loop succeeds");

        assert_eq!(*executed.lock().unwrap(), [json!(123)]);
        registration.unregister();
    }

    #[tokio::test]
    async fn parallel_tool_end_events_complete_order_but_results_source_order() {
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
        let executed = Arc::new(StdMutex::new(Vec::new()));
        let mut config = AgentLoopConfig::new(registration.get_model());
        config.tool_execution = ToolExecutionMode::Parallel;
        let context = AgentContext {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                executed,
                delay_first: true,
                execution_mode: None,
                terminate: false,
            })],
        };
        let (events, emit) = collect_events();

        run_agent_loop(
            vec![user_text("echo both")],
            context,
            config,
            emit,
            None,
            None,
        )
        .await
        .expect("loop succeeds");

        let events = events.lock().unwrap();
        let tool_execution_end_ids = events
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
                    message: Message::ToolResult(tool_result),
                } => Some(tool_result.tool_call_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tool_execution_end_ids, ["tool-2", "tool-1"]);
        assert_eq!(tool_result_ids, ["tool-1", "tool-2"]);
        registration.unregister();
    }

    #[tokio::test]
    async fn transform_context_runs_before_convert_to_llm_like_upstream() {
        let mut config = AgentLoopConfig::new(Model {
            id: "test-model".to_string(),
            api: "test".to_string(),
            provider: "test".to_string(),
            ..Default::default()
        });
        let transformed_seen = Arc::new(StdMutex::new(Vec::new()));
        config.transform_context = Some(Arc::new({
            let transformed_seen = Arc::clone(&transformed_seen);
            move |messages, _token| {
                let transformed_seen = Arc::clone(&transformed_seen);
                async move {
                    transformed_seen.lock().unwrap().extend(
                        messages
                            .iter()
                            .filter_map(user_text_value)
                            .map(str::to_string),
                    );
                    vec![user_text("transformed")]
                }
                .boxed()
            }
        }));
        let converted_seen = Arc::new(StdMutex::new(Vec::new()));
        config.convert_to_llm = Some(Arc::new({
            let converted_seen = Arc::clone(&converted_seen);
            move |messages| {
                let converted_seen = Arc::clone(&converted_seen);
                async move {
                    converted_seen.lock().unwrap().extend(
                        messages
                            .iter()
                            .filter_map(user_text_value)
                            .map(str::to_string),
                    );
                    messages
                        .into_iter()
                        .filter(|message| message.is_llm_message())
                        .collect()
                }
                .boxed()
            }
        }));
        let streamed_seen = Arc::new(StdMutex::new(Vec::new()));
        let stream_fn: StreamFn = Arc::new({
            let streamed_seen = Arc::clone(&streamed_seen);
            move |model, context, _options| {
                let streamed_seen = Arc::clone(&streamed_seen);
                async move {
                    streamed_seen.lock().unwrap().extend(
                        context
                            .messages
                            .iter()
                            .filter_map(user_text_value)
                            .map(str::to_string),
                    );
                    let (mut sender, stream) = create_assistant_message_event_stream();
                    sender.push(AssistantMessageEvent::Done {
                        reason: StopReason::Stop,
                        message: assistant_text_message(&model, "done"),
                    });
                    Ok(stream)
                }
                .boxed()
            }
        });
        let (_events, emit) = collect_events();

        run_agent_loop(
            vec![user_text("original")],
            AgentContext {
                system_prompt: Some(String::new()),
                messages: Vec::new(),
                tools: Vec::new(),
            },
            config,
            emit,
            None,
            Some(stream_fn),
        )
        .await
        .expect("loop succeeds");

        assert_eq!(*transformed_seen.lock().unwrap(), ["original"]);
        assert_eq!(*converted_seen.lock().unwrap(), ["transformed"]);
        assert_eq!(*streamed_seen.lock().unwrap(), ["transformed"]);
    }

    #[tokio::test]
    async fn steering_messages_inject_after_all_tool_calls_complete_like_upstream() {
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
            faux_assistant_message("after steering", None),
        ]);
        let poll_count = Arc::new(AtomicUsize::new(0));
        let mut config = AgentLoopConfig::new(registration.get_model());
        config.get_steering_messages = Some(Arc::new({
            let poll_count = Arc::clone(&poll_count);
            move || {
                let poll_count = Arc::clone(&poll_count);
                async move {
                    if poll_count.fetch_add(1, Ordering::SeqCst) == 1 {
                        vec![user_text("steer now")]
                    } else {
                        Vec::new()
                    }
                }
                .boxed()
            }
        }));
        let context = AgentContext {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: vec![Arc::new(echo_tool(Arc::new(StdMutex::new(Vec::new()))))],
        };
        let (events, emit) = collect_events();

        run_agent_loop(
            vec![user_text("echo both")],
            context,
            config,
            emit,
            None,
            None,
        )
        .await
        .expect("loop succeeds");

        let events = events.lock().unwrap();
        let last_tool_result_index = events
            .iter()
            .rposition(|event| {
                matches!(
                    event,
                    AgentEvent::MessageEnd {
                        message: Message::ToolResult(_)
                    }
                )
            })
            .expect("tool result event");
        let steering_index = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    AgentEvent::MessageStart { message }
                        if user_text_value(message) == Some("steer now")
                )
            })
            .expect("steering event");
        assert!(last_tool_result_index < steering_index);
        assert_eq!(tool_result_message_ids(&events), ["tool-1", "tool-2"]);
        registration.unregister();
    }

    #[tokio::test]
    async fn tool_execution_mode_sequential_overrides_parallel_config_like_upstream() {
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
        let mut config = AgentLoopConfig::new(registration.get_model());
        config.tool_execution = ToolExecutionMode::Parallel;
        let context = AgentContext {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                executed: Arc::new(StdMutex::new(Vec::new())),
                delay_first: true,
                execution_mode: Some(ToolExecutionMode::Sequential),
                terminate: false,
            })],
        };
        let (events, emit) = collect_events();

        run_agent_loop(
            vec![user_text("echo both")],
            context,
            config,
            emit,
            None,
            None,
        )
        .await
        .expect("loop succeeds");

        let events = events.lock().unwrap();
        let tool_execution_end_ids = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tool_execution_end_ids, ["tool-1", "tool-2"]);
        assert_eq!(tool_result_message_ids(&events), ["tool-1", "tool-2"]);
        registration.unregister();
    }

    #[tokio::test]
    async fn prepare_next_turn_snapshot_is_used_before_continuing_like_upstream() {
        let calls = Arc::new(AtomicUsize::new(0));
        let streamed_contexts = Arc::new(StdMutex::new(Vec::<Vec<String>>::new()));
        let stream_fn: StreamFn = Arc::new({
            let calls = Arc::clone(&calls);
            let streamed_contexts = Arc::clone(&streamed_contexts);
            move |model, context, _options| {
                let calls = Arc::clone(&calls);
                let streamed_contexts = Arc::clone(&streamed_contexts);
                async move {
                    streamed_contexts.lock().unwrap().push(
                        context
                            .messages
                            .iter()
                            .filter_map(user_text_value)
                            .map(str::to_string)
                            .collect(),
                    );
                    let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                    let (mut sender, stream) = create_assistant_message_event_stream();
                    sender.push(AssistantMessageEvent::Done {
                        reason: StopReason::Stop,
                        message: assistant_text_message(&model, format!("turn {call}")),
                    });
                    Ok(stream)
                }
                .boxed()
            }
        });
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let mut config = AgentLoopConfig::new(Model {
            id: "test-model".to_string(),
            api: "test".to_string(),
            provider: "test".to_string(),
            ..Default::default()
        });
        config.get_steering_messages = Some(Arc::new({
            let steering_polls = Arc::clone(&steering_polls);
            move || {
                let steering_polls = Arc::clone(&steering_polls);
                async move {
                    if steering_polls.fetch_add(1, Ordering::SeqCst) == 1 {
                        vec![user_text("steer")]
                    } else {
                        Vec::new()
                    }
                }
                .boxed()
            }
        }));
        let prepare_calls = Arc::new(AtomicUsize::new(0));
        config.prepare_next_turn = Some(Arc::new({
            let prepare_calls = Arc::clone(&prepare_calls);
            move |_context, _token| {
                let prepare_calls = Arc::clone(&prepare_calls);
                async move {
                    if prepare_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        Some(AgentLoopTurnUpdate {
                            context: Some(AgentContext {
                                system_prompt: Some(String::new()),
                                messages: vec![user_text("prepared context")],
                                tools: Vec::new(),
                            }),
                            model: None,
                            reasoning_level: None,
                        })
                    } else {
                        None
                    }
                }
                .boxed()
            }
        }));
        let (_events, emit) = collect_events();

        run_agent_loop(
            vec![user_text("initial")],
            AgentContext {
                system_prompt: Some(String::new()),
                messages: Vec::new(),
                tools: Vec::new(),
            },
            config,
            emit,
            None,
            Some(stream_fn),
        )
        .await
        .expect("loop succeeds");

        assert_eq!(
            *streamed_contexts.lock().unwrap(),
            vec![
                vec!["initial".to_string()],
                vec!["prepared context".to_string(), "steer".to_string()],
            ]
        );
    }

    #[tokio::test]
    async fn stop_and_terminate_controls_exit_before_more_messages_like_upstream() {
        let registration = register_faux_provider(None);
        registration.set_responses([
            faux_assistant_message(
                vec![faux_tool_call(
                    "echo",
                    json!({ "value": "stop" }),
                    Some("tool-1".to_string()),
                )],
                Some(FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                }),
            ),
            faux_assistant_message("should not be reached", None),
        ]);
        let context = AgentContext {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: vec![Arc::new(EchoTool {
                executed: Arc::new(StdMutex::new(Vec::new())),
                delay_first: false,
                execution_mode: None,
                terminate: true,
            })],
        };
        let (_events, emit) = collect_events();

        let messages = run_agent_loop(
            vec![user_text("echo once")],
            context,
            AgentLoopConfig::new(registration.get_model()),
            emit,
            None,
            None,
        )
        .await
        .expect("loop succeeds");

        assert_eq!(
            messages
                .iter()
                .filter(|message| matches!(message, Message::Assistant(_)))
                .count(),
            1
        );
        registration.unregister();

        let mut config = AgentLoopConfig::new(Model {
            id: "test-model".to_string(),
            api: "test".to_string(),
            provider: "test".to_string(),
            ..Default::default()
        });
        let follow_up_polls = Arc::new(AtomicUsize::new(0));
        config.get_follow_up_messages = Some(Arc::new({
            let follow_up_polls = Arc::clone(&follow_up_polls);
            move || {
                let follow_up_polls = Arc::clone(&follow_up_polls);
                async move {
                    follow_up_polls.fetch_add(1, Ordering::SeqCst);
                    vec![user_text("follow up")]
                }
                .boxed()
            }
        }));
        config.should_stop_after_turn = Some(Arc::new(|_context| async { true }.boxed()));
        let (_events, emit) = collect_events();
        let messages = run_agent_loop(
            vec![user_text("stop after turn")],
            AgentContext {
                system_prompt: Some(String::new()),
                messages: Vec::new(),
                tools: Vec::new(),
            },
            config,
            emit,
            None,
            Some(immediate_stream_fn("stopped")),
        )
        .await
        .expect("loop succeeds");

        assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
        assert!(
            !messages
                .iter()
                .any(|message| user_text_value(message) == Some("follow up"))
        );
    }

    #[tokio::test]
    async fn continues_when_not_all_parallel_tool_results_terminate_like_upstream() {
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
            faux_assistant_message("continued", None),
        ]);
        let mut config = AgentLoopConfig::new(registration.get_model());
        config.after_tool_call = Some(Arc::new(|context: AfterToolCallContext, _token| {
            Box::pin(async move {
                Ok(Some(AfterToolCallResult {
                    terminate: Some(context.tool_call.id == "tool-1"),
                    ..AfterToolCallResult::default()
                }))
            })
        }));
        let context = AgentContext {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: vec![Arc::new(echo_tool(Arc::new(StdMutex::new(Vec::new()))))],
        };
        let (_events, emit) = collect_events();

        let messages = run_agent_loop(
            vec![user_text("echo both")],
            context,
            config,
            emit,
            None,
            None,
        )
        .await
        .expect("loop succeeds");

        assert_eq!(
            messages
                .iter()
                .filter(|message| matches!(message, Message::Assistant(_)))
                .count(),
            2
        );
        registration.unregister();
    }

    #[tokio::test]
    async fn after_tool_call_can_mark_batch_terminating_like_upstream() {
        let registration = register_faux_provider(None);
        registration.set_responses([
            faux_assistant_message(
                vec![faux_tool_call(
                    "echo",
                    json!({ "value": "stop" }),
                    Some("tool-1".to_string()),
                )],
                Some(FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                }),
            ),
            faux_assistant_message("should not be reached", None),
        ]);
        let mut config = AgentLoopConfig::new(registration.get_model());
        config.after_tool_call = Some(Arc::new(|_context: AfterToolCallContext, _token| {
            Box::pin(async {
                Ok(Some(AfterToolCallResult {
                    terminate: Some(true),
                    ..AfterToolCallResult::default()
                }))
            })
        }));
        let context = AgentContext {
            system_prompt: Some(String::new()),
            messages: Vec::new(),
            tools: vec![Arc::new(echo_tool(Arc::new(StdMutex::new(Vec::new()))))],
        };
        let (_events, emit) = collect_events();

        let messages = run_agent_loop(
            vec![user_text("echo once")],
            context,
            config,
            emit,
            None,
            None,
        )
        .await
        .expect("loop succeeds");

        assert_eq!(
            messages
                .iter()
                .filter(|message| matches!(message, Message::Assistant(_)))
                .count(),
            1
        );
        registration.unregister();
    }

    #[tokio::test]
    async fn loop_continue_validation_and_existing_context_events_match_upstream() {
        let (_events, emit) = collect_events();
        let empty_result = run_agent_loop_continue(
            AgentContext {
                system_prompt: Some(String::new()),
                messages: Vec::new(),
                tools: Vec::new(),
            },
            AgentLoopConfig::new(Model::default()),
            emit,
            None,
            Some(immediate_stream_fn("unused")),
        )
        .await;
        assert!(matches!(
            empty_result,
            Err(crate::AgentError::NoMessagesToContinue)
        ));

        let (_events, emit) = collect_events();
        let assistant_result = run_agent_loop_continue(
            AgentContext {
                system_prompt: Some(String::new()),
                messages: vec![Message::Assistant(assistant_text_message(
                    &Model::default(),
                    "assistant tail",
                ))],
                tools: Vec::new(),
            },
            AgentLoopConfig::new(Model::default()),
            emit,
            None,
            Some(immediate_stream_fn("unused")),
        )
        .await;
        assert!(matches!(
            assistant_result,
            Err(crate::AgentError::CannotContinueFromAssistant)
        ));

        let (events, emit) = collect_events();
        run_agent_loop_continue(
            AgentContext {
                system_prompt: Some(String::new()),
                messages: vec![user_text("existing")],
                tools: Vec::new(),
            },
            AgentLoopConfig::new(Model {
                id: "test-model".to_string(),
                api: "test".to_string(),
                provider: "test".to_string(),
                ..Default::default()
            }),
            emit,
            None,
            Some(immediate_stream_fn("continued")),
        )
        .await
        .expect("continue succeeds");
        assert!(!events.lock().unwrap().iter().any(|event| {
            matches!(
                event,
                AgentEvent::MessageStart { message }
                    if user_text_value(message) == Some("existing")
            )
        }));

        let (_events, emit) = collect_events();
        run_agent_loop_continue(
            AgentContext {
                system_prompt: Some(String::new()),
                messages: vec![Message::Custom(json!({ "kind": "ui" }))],
                tools: Vec::new(),
            },
            AgentLoopConfig::new(Model {
                id: "test-model".to_string(),
                api: "test".to_string(),
                provider: "test".to_string(),
                ..Default::default()
            }),
            emit,
            None,
            Some(immediate_stream_fn("custom tail")),
        )
        .await
        .expect("custom tail continue succeeds");
    }
}
