use std::pin::Pin;
use std::sync::Arc;

use ai::{AssistantMessageEvent, StopReason, ToolResultMessage};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::types::{
    AfterToolCallContext, AgentContext, AgentEvent, AgentEventSink, AgentLoopConfig, AgentMessage,
    AgentToolResult, BeforeToolCallContext, ToolExecutionMode, assistant_tool_calls,
};
use crate::{AgentError, Result};

pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    mut context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
    stream_fn: Option<crate::types::StreamFn>,
) -> Result<Vec<AgentMessage>> {
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
    stream_fn: Option<crate::types::StreamFn>,
) -> Result<Vec<AgentMessage>> {
    if context.messages.is_empty() {
        return Err(AgentError::NoMessagesToContinue);
    }
    if matches!(context.messages.last(), Some(ai::Message::Assistant(_))) {
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
    stream_fn: Option<crate::types::StreamFn>,
) -> Result<()> {
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
            new_messages.push(ai::Message::Assistant(assistant.clone()));

            if matches!(
                assistant.stop_reason,
                StopReason::Error | StopReason::Aborted
            ) {
                emit(AgentEvent::TurnEnd {
                    message: ai::Message::Assistant(assistant),
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
                        .push(ai::Message::ToolResult(result.clone()));
                    new_messages.push(ai::Message::ToolResult(result.clone()));
                }
            }

            emit(AgentEvent::TurnEnd {
                message: ai::Message::Assistant(assistant.clone()),
                tool_results: tool_results.clone(),
            })
            .await?;

            if let Some(prepare_next_turn) = config.prepare_next_turn.clone() {
                if let Some(update) = prepare_next_turn(crate::types::PrepareNextTurnContext {
                    message: assistant.clone(),
                    tool_results: tool_results.clone(),
                    context: context.clone(),
                    new_messages: new_messages.clone(),
                })
                .await
                {
                    if let Some(next_context) = update.context {
                        *context = next_context;
                    }
                    if let Some(next_model) = update.model {
                        config.model = next_model;
                    }
                    if let Some(reasoning_level) = update.reasoning_level {
                        config.options.reasoning = if reasoning_level == ai::ModelThinkingLevel::Off
                        {
                            None
                        } else {
                            Some(reasoning_level)
                        };
                    }
                }
            }

            if let Some(should_stop) = &config.should_stop_after_turn {
                if should_stop(crate::types::ShouldStopAfterTurnContext {
                    message: assistant,
                    tool_results,
                    context: context.clone(),
                    new_messages: new_messages.clone(),
                })
                .await
                {
                    return Ok(());
                }
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
    stream_fn: Option<crate::types::StreamFn>,
) -> Result<ai::AssistantMessage> {
    let mut messages = context.messages.clone();
    if let Some(transform) = &config.transform_context {
        messages = transform(messages, cancellation_token.clone()).await;
    }
    let llm_messages = if let Some(convert) = &config.convert_to_llm {
        convert(messages).await
    } else {
        messages
    };

    let mut llm_context = context.llm_context();
    llm_context.messages = llm_messages;

    let mut options = config.options.clone();
    options.stream.cancellation_token = cancellation_token.clone();
    if let Some(get_api_key) = &config.get_api_key {
        options.stream.api_key = get_api_key(config.model.provider.clone()).await;
    }

    let mut response = if let Some(stream_fn) = stream_fn {
        stream_fn(config.model.clone(), llm_context, options).await?
    } else {
        ai::stream_simple(config.model.clone(), llm_context, Some(options))?
    };

    let mut partial_added = false;
    while let Some(event) = response.next().await {
        match &event {
            AssistantMessageEvent::Start { partial } => {
                context
                    .messages
                    .push(ai::Message::Assistant(partial.clone()));
                partial_added = true;
                emit(AgentEvent::MessageStart {
                    message: ai::Message::Assistant(partial.clone()),
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
                        *last = ai::Message::Assistant(partial.clone());
                    }
                }
                emit(AgentEvent::MessageUpdate {
                    message: ai::Message::Assistant(partial.clone()),
                    assistant_message_event: event.clone(),
                })
                .await?;
            }
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {
                let final_message = response.result().await?;
                if partial_added {
                    if let Some(last) = context.messages.last_mut() {
                        *last = ai::Message::Assistant(final_message.clone());
                    }
                } else {
                    context
                        .messages
                        .push(ai::Message::Assistant(final_message.clone()));
                    emit(AgentEvent::MessageStart {
                        message: ai::Message::Assistant(final_message.clone()),
                    })
                    .await?;
                }
                emit(AgentEvent::MessageEnd {
                    message: ai::Message::Assistant(final_message.clone()),
                })
                .await?;
                return Ok(final_message);
            }
        }
    }

    let final_message = response.result().await?;
    if partial_added {
        if let Some(last) = context.messages.last_mut() {
            *last = ai::Message::Assistant(final_message.clone());
        }
    } else {
        context
            .messages
            .push(ai::Message::Assistant(final_message.clone()));
        emit(AgentEvent::MessageStart {
            message: ai::Message::Assistant(final_message.clone()),
        })
        .await?;
    }
    emit(AgentEvent::MessageEnd {
        message: ai::Message::Assistant(final_message.clone()),
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
    assistant: &ai::AssistantMessage,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> Result<ExecutedToolBatch> {
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
    assistant: &ai::AssistantMessage,
    tool_calls: Vec<ai::ToolCall>,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> Result<ExecutedToolBatch> {
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
    assistant: &ai::AssistantMessage,
    tool_calls: Vec<ai::ToolCall>,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> Result<ExecutedToolBatch> {
    let futures = tool_calls
        .into_iter()
        .map(|tool_call| {
            let context = context.clone();
            let assistant = assistant.clone();
            let config = config.clone();
            let emit = emit.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                execute_one_tool(
                    &context,
                    &assistant,
                    tool_call,
                    &config,
                    emit,
                    cancellation_token,
                )
                .await
            }
        })
        .collect::<Vec<_>>();
    let finalized = futures::future::join_all(futures).await;
    let mut messages = Vec::new();
    let mut terminate_flags = Vec::new();
    for item in finalized {
        let finalized = item?;
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

async fn execute_one_tool(
    context: &AgentContext,
    assistant: &ai::AssistantMessage,
    tool_call: ai::ToolCall,
    config: &AgentLoopConfig,
    emit: AgentEventSink,
    cancellation_token: Option<CancellationToken>,
) -> Result<(ai::ToolCall, bool, AgentToolResult)> {
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
        let result = AgentToolResult::text(format!("Tool {} not found", tool_call.name));
        emit(AgentEvent::ToolExecutionEnd {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            result: result.clone(),
            is_error: true,
        })
        .await?;
        return Ok((tool_call, true, result));
    };

    let mut prepared_args = match tool.prepare_arguments(tool_call.arguments.clone()) {
        Ok(args) => args,
        Err(error) => {
            let result = AgentToolResult::text(error.to_string());
            emit(AgentEvent::ToolExecutionEnd {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                result: result.clone(),
                is_error: true,
            })
            .await?;
            return Ok((tool_call, true, result));
        }
    };

    if let Some(before_tool_call) = &config.before_tool_call {
        match before_tool_call(
            BeforeToolCallContext {
                assistant_message: assistant.clone(),
                tool_call: tool_call.clone(),
                args: prepared_args.clone(),
                context: context.clone(),
            },
            cancellation_token.clone(),
        )
        .await
        {
            Ok(Some(before_result)) => {
                if before_result.block {
                    let result = AgentToolResult::text(
                        before_result
                            .reason
                            .unwrap_or_else(|| "Tool execution was blocked".to_string()),
                    );
                    emit(AgentEvent::ToolExecutionEnd {
                        tool_call_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        result: result.clone(),
                        is_error: true,
                    })
                    .await?;
                    return Ok((tool_call, true, result));
                }
                if let Some(args) = before_result.args {
                    prepared_args = args;
                }
            }
            Ok(None) => {}
            Err(error) => {
                let result = AgentToolResult::text(error.to_string());
                emit(AgentEvent::ToolExecutionEnd {
                    tool_call_id: tool_call.id.clone(),
                    tool_name: tool_call.name.clone(),
                    result: result.clone(),
                    is_error: true,
                })
                .await?;
                return Ok((tool_call, true, result));
            }
        }
    }

    if cancellation_token
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        let result = AgentToolResult::text("Operation aborted");
        emit(AgentEvent::ToolExecutionEnd {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            result: result.clone(),
            is_error: true,
        })
        .await?;
        return Ok((tool_call, true, result));
    }

    let emit_for_update = emit.clone();
    let update_tool_call = tool_call.clone();
    let on_update = Arc::new(move |partial_result: AgentToolResult| {
        let emit = emit_for_update.clone();
        let tool_call = update_tool_call.clone();
        Box::pin(async move {
            let _ = emit(AgentEvent::ToolExecutionUpdate {
                tool_call_id: tool_call.id,
                tool_name: tool_call.name,
                args: tool_call.arguments,
                partial_result,
            })
            .await;
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
        Err(error) => (true, AgentToolResult::text(error.to_string())),
    };

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
                if let Some(details) = after_result.details {
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
                result = AgentToolResult::text(error.to_string());
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

fn create_tool_result_message(
    finalized: (ai::ToolCall, bool, AgentToolResult),
) -> ToolResultMessage {
    let (tool_call, is_error, result) = finalized;
    ToolResultMessage {
        tool_call_id: tool_call.id,
        tool_name: tool_call.name,
        content: result.content,
        details: result.details,
        is_error,
        timestamp: ai::utils::time::now_millis(),
    }
}

async fn emit_tool_result_message(
    message: &ToolResultMessage,
    emit: &AgentEventSink,
) -> Result<()> {
    let message = ai::Message::ToolResult(message.clone());
    emit(AgentEvent::MessageStart {
        message: message.clone(),
    })
    .await?;
    emit(AgentEvent::MessageEnd { message }).await
}
