//! Port of durable `src/harness/generation.ts`: the built-in generation task.
//!
//! Divergences from Pi:
//! - `beforeRequest` receives and returns the message list (TS `{ messages }`), and `onYield` returns the continuation
//!   input (TS `{ continue }`).
//! - The partial throttle runs its commits as Tokio tasks; drafts are diffed when the commit prepares, so the partial is
//!   assigned as a whole value (TS `assignJson`).

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::chord::Context;
use crate::durable::entries::{ASSISTANT_ENTRY, RESET_ENTRY, SYSTEM_ENTRY, USER_ENTRY};
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, EntryId, SubmissionId, TaskId};
use crate::durable::session::{DocDraft, Transaction};
use crate::durable::tasks::{NextTaskState, RunningTask, Task, TaskDefinition, define_task};
use crate::durable::types::{
    EntryHead, EntryQuery, EntryRecord, JoinPolicy, JsonObject, SubmissionSettlement, TaskOptions,
    TaskOutcome, TaskOutcomeError, TaskOwnership, TypedEntryDraft,
};
use crate::models::Models;
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Context as ModelContext,
    DeferredFetchOptions, DeferredHandle, Message, Model, ModelThinkingLevel,
    ProviderRequestOptions, SimpleStreamOptions, StopReason, ToolCall, UserMessage,
    UserMessageContent,
};
use crate::utils::overflow::is_context_overflow;
use crate::utils::retry::{is_retryable_assistant_error, retry_delay_ms};
use crate::utils::transcript::get_current_tools;

use super::agent::add_tools;
use super::compaction::{
    CompactionInput, create_compaction, estimate_context, no_model_message, no_model_outcome,
    select_cut, simple_stream_options, stop_reason_str,
};
use super::inbox::{BoundaryKind, QueueModes, apply_boundary, prepare_boundary};
use super::live::{
    GenerationStatus, LIVE_DOC, LiveState, RetryStatus, RunState, SlotStatus, ToolSlot, end_run,
};
use super::prompt::{SystemDraft, plan_system_entries, render_sections, replay_sections};
use super::provider::ensure_provider_session_id;
use super::scheduler::TaskRuntime;
use super::tool::{TOOL_TASK, ToolTaskInput, ToolTaskResult, append_tool_result, harness_error};
use super::types::{
    CompactionPolicy, CompactionReason, CompactionResult, ContextView, ConversationStreamOptions,
    GenerationHooks, ModelRef, PromptInput, ToolControl, ToolExecutionMode, UserInput,
};
use super::usage::{UsageBucket, record_usage};

pub type GenerationInput = JsonObject;

/// `GenerationCheckpoint`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
pub enum GenerationCheckpoint {
    #[serde(rename_all = "camelCase")]
    Prepare {
        attempt: u32,
        /// The blocking compaction this generation waited for; it starts no other compaction (spec §8.3).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compacted: Option<TaskId<CompactionResult>>,
        /// Error text of the overflow that started `compacted`; checked once when `prepare` resumes.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        overflow: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    Request {
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compacted: Option<TaskId<CompactionResult>>,
        model: ModelRef,
        thinking_level: ModelThinkingLevel,
        /// The settings' request options when preparation committed; a resend after recovery uses them unchanged.
        stream_options: ConversationStreamOptions,
        /// Newest entry included in the request.
        cutoff: EntryId,
    },
    #[serde(rename_all = "camelCase")]
    Retry {
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compacted: Option<TaskId<CompactionResult>>,
        until: u64,
    },
    #[serde(rename_all = "camelCase")]
    Poll {
        attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compacted: Option<TaskId<CompactionResult>>,
        model: ModelRef,
        cutoff: EntryId,
        handle: DeferredHandle,
        poll_at: u64,
    },
    /// Waiting on the round's tool tasks, which the generation owns (spec §8.5).
    #[serde(rename_all = "camelCase")]
    Tools {
        /// The tool-calling answer.
        assistant: EntryId,
        /// Tool tasks created so far, in call order; grows by one per started call of a sequential round.
        tools: Vec<TaskId<ToolTaskResult>>,
        /// Calls of a sequential round not started yet, in call order.
        pending: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationResult {
    pub entry_id: EntryId,
}

type Runtime =
    TaskRuntime<GenerationInput, GenerationCheckpoint, GenerationResult, GenerationHooks>;
type Next = NextTaskState<GenerationCheckpoint>;
type Current = RunningTask<GenerationInput, GenerationCheckpoint, GenerationResult>;

pub type GenerationTaskType =
    Task<GenerationInput, GenerationCheckpoint, GenerationResult, GenerationHooks>;

/// What classification needs from the request that produced a message.
struct Request {
    attempt: u32,
    compacted: Option<TaskId<CompactionResult>>,
    model: ModelRef,
    cutoff: EntryId,
    /// Committed model context through `cutoff`, when the phase already derived it.
    messages: Option<Vec<Message>>,
    /// Set when the message came from polling, so a still deferred result polls strictly later.
    poll_at: Option<u64>,
}

const PARTIAL_THROTTLE_MS: u64 = 100;
const DEFAULT_POLL_AFTER_MS: u64 = 5000;

/// Built-in generation task: prepares the positional system prompt and tool loadout, requests or polls the model,
/// retries, and classifies the response. The run's inputs live in `pi.live.run`.
pub static GENERATION_TASK: LazyLock<GenerationTaskType> = LazyLock::new(|| {
    define_task(
        TaskDefinition::new("pi.generation", 1, |_: &GenerationInput| {
            GenerationCheckpoint::Prepare {
                attempt: 1,
                compacted: None,
                overflow: None,
            }
        })
        .phase("prepare", |task, runtime, context| async move {
            prepare_phase(task, runtime, context).await
        })
        .phase("request", |task, runtime, context| async move {
            request_phase(task, runtime, context).await
        })
        .phase("retry", |task, runtime: Runtime, context| async move {
            let GenerationCheckpoint::Retry {
                attempt,
                compacted,
                until,
            } = task.checkpoint
            else {
                return Err(wrong_phase("retry"));
            };
            runtime.sleep(until, &context).await?;
            let conversation_id = runtime.conversation_id();
            runtime
                .commit(
                    move |tx, _| async move {
                        let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                        live.edit(|live| {
                            live.generation = Some(GenerationStatus::attempt(attempt + 1))
                        })?;
                        Ok(Some(Next::running(GenerationCheckpoint::Prepare {
                            attempt: attempt + 1,
                            compacted,
                            overflow: None,
                        })))
                    },
                    &context,
                )
                .await
        })
        .phase("poll", |task, runtime: Runtime, context| async move {
            let GenerationCheckpoint::Poll {
                attempt,
                compacted,
                model: ref_,
                cutoff,
                handle,
                poll_at,
            } = task.checkpoint
            else {
                return Err(wrong_phase("poll"));
            };
            let Some(model) = runtime.models().get_model(&ref_.provider, &ref_.model_id) else {
                return fail_no_model(&runtime, Some(&ref_), &context).await;
            };
            runtime.sleep(poll_at, &context).await?;
            let options = DeferredFetchOptions {
                request: ProviderRequestOptions {
                    signal: Some(runtime.signal().token()),
                    ..ProviderRequestOptions::default()
                },
                wait: None,
            };
            let message = runtime
                .models()
                .fetch_deferred(&model, &handle, options)
                .await;
            let request = Request {
                attempt,
                compacted,
                model: ref_,
                cutoff,
                messages: None,
                poll_at: Some(poll_at),
            };
            classify(&runtime, request, message, &context).await
        })
        .phase("tools", |task, runtime: Runtime, context| async move {
            let GenerationCheckpoint::Tools {
                assistant,
                tools,
                pending,
            } = task.checkpoint
            else {
                return Err(wrong_phase("tools"));
            };
            let mut rest = pending.into_iter();
            let Some(next) = rest.next() else {
                return finish_tool_round(&runtime, assistant, tools, &context).await;
            };
            let rest: Vec<String> = rest.collect();
            // Sequential round: start the next call and wait for it.
            let conversation_id = runtime.conversation_id();
            let owner = runtime.task_id().erase();
            runtime
                .commit(
                    move |tx, _| async move {
                        let task_id = create_tool_task(&tx, owner, assistant, next.clone()).await?;
                        let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                        live.edit(|live| {
                            if let Some(slot) = live.tools.as_mut().and_then(|slots| {
                                slots
                                    .iter_mut()
                                    .find(|slot| slot.call_id == next && slot.task_id.is_none())
                            }) {
                                slot.task_id = Some(task_id.erase());
                            }
                        })?;
                        let mut tools = tools;
                        tools.push(task_id);
                        Ok(Some(Next::waiting(
                            GenerationCheckpoint::Tools {
                                assistant,
                                tools,
                                pending: rest,
                            },
                            vec![task_id.erase()],
                            JoinPolicy::AllSettled,
                        )))
                    },
                    &context,
                )
                .await
        })
        .abort(|task, runtime: Runtime, context| async move {
            abort_generation(task, runtime, context).await
        }),
    )
});

fn wrong_phase(phase: &str) -> Error {
    Error::type_error(format!("Generation task is not in its {phase} phase"))
}

async fn abort_generation(task: Current, runtime: Runtime, context: Context) -> Result<()> {
    let checkpoint = task.checkpoint;
    if let GenerationCheckpoint::Poll { model, handle, .. } = &checkpoint
        && let Some(model) = runtime.models().get_model(&model.provider, &model.model_id)
    {
        let options = ProviderRequestOptions {
            signal: Some(runtime.signal().token()),
            ..ProviderRequestOptions::default()
        };
        if let Err(error) = runtime
            .models()
            .cancel_deferred(&model, handle, options)
            .await
        {
            runtime.report(Error::message(error.to_string()))?;
        }
    }
    let conversation_id = runtime.conversation_id();
    // Runs after the round's tool tasks are terminal; calls never started get `aborted` results (spec §8.5).
    let unstarted = match &checkpoint {
        GenerationCheckpoint::Tools {
            assistant, pending, ..
        } => read_calls(&runtime, *assistant, pending, &context).await?,
        _ => Vec::new(),
    };
    let task_id = runtime.task_id().erase();
    let now = runtime.now()?;
    runtime
        .commit(
            move |tx, _| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                convert_partial(&tx, &live, conversation_id).await?;
                for call in &unstarted {
                    let result =
                        harness_error("aborted", &format!("Tool {} was aborted", call.name));
                    append_tool_result(&tx, conversation_id, call, result, now).await?;
                }
                let mut state = live.get()?;
                end_run(&tx, &mut state, task_id, unanswered("aborted", None))?;
                live.set(state)?;
                Ok(Some(Next::Terminal {
                    outcome: TaskOutcome::Aborted {
                        reason: None,
                        result: None,
                    },
                }))
            },
            &context,
        )
        .await
}

fn unanswered(reason: &str, detail: Option<String>) -> SubmissionSettlement {
    SubmissionSettlement::Unanswered {
        reason: reason.to_string(),
        detail: detail.map(serde_json::Value::String),
    }
}

/// Render the system prompt and tool loadout and append the positional `pi.system` entries they need, then move to
/// `request`. The agent and settings resolved here are fixed for this request. Only the Harness writes to a busy
/// conversation, so the transcript read here is still the tail at the commit.
async fn prepare_phase(task: Current, runtime: Runtime, context: Context) -> Result<()> {
    let conversation_id = runtime.conversation_id();
    let agent = runtime.agent(&context).await?;
    let settings = runtime.settings();
    let thinking_level = agent.thinking_level;
    let model = agent.model.clone();
    let resolved = model
        .as_ref()
        .and_then(|model| runtime.models().get_model(&model.provider, &model.model_id));
    let (Some(model), Some(resolved)) = (model.clone(), resolved) else {
        return fail_no_model(&runtime, model.as_ref(), &context).await;
    };
    let GenerationCheckpoint::Prepare {
        attempt,
        compacted,
        overflow,
    } = task.checkpoint
    else {
        return Err(wrong_phase("prepare"));
    };
    if let (Some(compacted), Some(overflow)) = (compacted, &overflow) {
        let outcomes = runtime.outcomes(&[compacted], &context).await?;
        let summarized = matches!(
            outcomes.first(),
            Some(TaskOutcome::Completed { result })
                if result.get("entryId").is_some_and(|id| !id.is_null())
        );
        if !summarized {
            return fail_model_error(&runtime, overflow.clone(), &context).await;
        }
    }
    let view = runtime.context(conversation_id, &context, None).await?;
    let shown = replay_sections(&view.messages);
    let report_runtime = runtime.clone();
    let report = move |error: Error| {
        let _ = report_runtime.report(error);
    };
    let env = match runtime.env(&context).await {
        Ok(env) => env,
        Err(error) => {
            if context
                .abort_signal()
                .is_some_and(|signal| signal.aborted())
            {
                return Err(error);
            }
            report(error);
            None
        }
    };
    let input = PromptInput {
        conversation_id,
        agent: agent.clone(),
        env,
        shown: shown.clone(),
        read: runtime.reader(),
    };
    let desired = render_sections(&agent.sections, &input, &shown, &report, &context).await?;
    let tools: Vec<_> = agent.tools.iter().map(|tool| tool.tool()).collect();
    let entries = plan_system_entries(&view, &desired, &tools, runtime.now()?);
    let threshold = match compacted {
        None => threshold_compaction(
            &view,
            &entries,
            resolved.context_window,
            settings.compaction,
        ),
        Some(_) => None,
    };
    let owner = runtime.task_id().erase();
    if threshold == Some(Threshold::Blocking) {
        // Compact first and prepare again; the transcript is unchanged until the compaction appends.
        return runtime
            .commit(
                move |tx, _| async move {
                    let child =
                        create_compaction(&tx, conversation_id, threshold_input(), Some(owner))
                            .await?;
                    Ok(Some(Next::waiting(
                        GenerationCheckpoint::Prepare {
                            attempt,
                            compacted: Some(child),
                            overflow: None,
                        },
                        vec![child.erase()],
                        JoinPolicy::AllSettled,
                    )))
                },
                &context,
            )
            .await;
    }
    let stream_options = settings.stream.clone();
    runtime
        .commit(
            move |tx, _| async move {
                let mut cutoff = tx
                    .scan_entries(
                        EntryQuery {
                            conversation_id,
                            min_entry_id: None,
                            max_entry_id: None,
                        },
                        1,
                        None,
                    )
                    .await?
                    .items
                    .first()
                    .map(|entry| entry.id);
                for entry in entries {
                    cutoff = Some(
                        tx.append_entry_of(&SYSTEM_ENTRY, conversation_id, entry)
                            .await?
                            .id,
                    );
                }
                let Some(cutoff) = cutoff else {
                    return Err(Error::message(format!(
                        "Conversation {conversation_id} has no entries to send"
                    )));
                };
                // Checked in this commit, so a compaction admitted during preparation counts.
                if threshold == Some(Threshold::Background)
                    && tx
                        .doc(&*LIVE_DOC, conversation_id)
                        .await?
                        .get()?
                        .compactions
                        .is_none()
                {
                    create_compaction(&tx, conversation_id, threshold_input(), None).await?;
                }
                Ok(Some(Next::running(GenerationCheckpoint::Request {
                    attempt,
                    compacted,
                    model,
                    thinking_level,
                    stream_options,
                    cutoff,
                })))
            },
            &context,
        )
        .await
}

fn threshold_input() -> CompactionInput {
    CompactionInput {
        reason: CompactionReason::Threshold,
        instructions: None,
    }
}

async fn request_phase(task: Current, runtime: Runtime, context: Context) -> Result<()> {
    let GenerationCheckpoint::Request {
        attempt,
        compacted,
        model: ref_,
        thinking_level,
        stream_options,
        cutoff,
    } = task.checkpoint
    else {
        return Err(wrong_phase("request"));
    };
    let conversation_id = runtime.conversation_id();
    runtime
        .commit(
            move |tx, _| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                convert_partial(&tx, &live, conversation_id).await?;
                live.edit(|live| live.generation = Some(GenerationStatus::attempt(attempt)))?;
                Ok(None)
            },
            &context,
        )
        .await?;
    let Some(model) = runtime.models().get_model(&ref_.provider, &ref_.model_id) else {
        return fail_no_model(&runtime, Some(&ref_), &context).await;
    };
    let view = runtime
        .context(conversation_id, &context, Some(cutoff))
        .await?;
    let messages = Arc::new(Mutex::new(view.messages.clone()));
    runtime
        .hooks()
        .each(
            |hooks| hooks.before_request.clone(),
            |hook| {
                let messages = messages.clone();
                let api = runtime.hook_api();
                let context = context.clone();
                async move {
                    let current = messages.lock().clone();
                    if let Some(replaced) = hook(current, api, context).await? {
                        *messages.lock() = replaced;
                    }
                    Ok(())
                }
            },
        )
        .await?;
    let messages = messages.lock().clone();
    let mut options = simple_stream_options(&stream_options, thinking_level);
    options.stream.signal = Some(runtime.signal().token());
    options.stream.session_id = Some(ensure_provider_session_id(runtime.core(), &context).await?);
    let message = stream_response(&runtime, &model, messages, options, attempt, &context).await?;
    let request = Request {
        attempt,
        compacted,
        model: ref_,
        cutoff,
        messages: Some(view.messages),
        poll_at: None,
    };
    classify(&runtime, request, message, &context).await
}

/// The calls `call_ids` of the assistant entry, in the given order.
async fn read_calls(
    runtime: &Runtime,
    assistant: EntryId,
    call_ids: &[String],
    context: &Context,
) -> Result<Vec<ToolCall>> {
    let entry = runtime
        .entry_of(&ASSISTANT_ENTRY, assistant, context)
        .await?;
    let calls: Vec<ToolCall> = match entry
        .and_then(|entry| entry.model)
        .and_then(|model| model.into_iter().next())
    {
        Some(Message::Assistant(message)) => tool_calls(&message),
        _ => Vec::new(),
    };
    Ok(call_ids
        .iter()
        .filter_map(|id| calls.iter().find(|call| &call.id == id).cloned())
        .collect())
}

fn tool_calls(message: &AssistantMessage) -> Vec<ToolCall> {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

/// A tool task for call `call_id`, owned by the generation.
async fn create_tool_task(
    tx: &Transaction,
    owner: TaskId,
    assistant: EntryId,
    call_id: String,
) -> Result<TaskId<ToolTaskResult>> {
    tx.create_task(
        &TOOL_TASK,
        ToolTaskInput { assistant, call_id },
        TaskOptions {
            ownership: TaskOwnership::Task { task_id: owner },
            conversation_id: None,
            background: None,
        },
    )
    .await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Threshold {
    Blocking,
    Background,
}

/// Which threshold compaction preparation starts before its request (spec §8.3): `blocking` above
/// `context_window - reserve_tokens`, `background` above the background threshold, and only when range selection finds
/// a cut. The caller starts a background one only while no compaction is listed.
fn threshold_compaction(
    view: &ContextView,
    planned: &[SystemDraft],
    context_window: u32,
    policy: CompactionPolicy,
) -> Option<Threshold> {
    if !policy.enabled || context_window == 0 {
        return None;
    }
    let extra: Vec<Message> = planned
        .iter()
        .flat_map(|entry| entry.model.clone().unwrap_or_default())
        .collect();
    let tokens = estimate_context(view, &extra) as i128;
    let blocking = i128::from(context_window) - i128::from(policy.reserve_tokens);
    let background = blocking - i128::from(policy.background_tokens);
    let over = if tokens > blocking {
        Some(Threshold::Blocking)
    } else if policy.background_tokens > 0 && tokens > background {
        Some(Threshold::Background)
    } else {
        None
    };
    over.filter(|_| select_cut(view, policy.keep_recent_tokens).is_some())
}

fn model_error(text: String) -> TaskOutcome {
    TaskOutcome::Failed {
        error: TaskOutcomeError {
            message: text,
            detail: Some(serde_json::json!({ "reason": "model_error" })),
        },
        result: None,
    }
}

/// Settle the run's inputs `unanswered` with `model_error` and fail with `text`.
async fn fail_model_error(runtime: &Runtime, text: String, context: &Context) -> Result<()> {
    let conversation_id = runtime.conversation_id();
    let task_id = runtime.task_id().erase();
    runtime
        .commit(
            move |tx, _| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                let mut state = live.get()?;
                end_run(
                    &tx,
                    &mut state,
                    task_id,
                    unanswered("model_error", Some(text.clone())),
                )?;
                live.set(state)?;
                Ok(Some(Next::Terminal {
                    outcome: model_error(text),
                }))
            },
            context,
        )
        .await
}

/// Settle the run's inputs `unanswered` with `no_model` and fail.
async fn fail_no_model(
    runtime: &Runtime,
    ref_: Option<&ModelRef>,
    context: &Context,
) -> Result<()> {
    let message = no_model_message(ref_);
    let conversation_id = runtime.conversation_id();
    let task_id = runtime.task_id().erase();
    runtime
        .commit(
            move |tx, _| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                let mut state = live.get()?;
                end_run(&tx, &mut state, task_id, unanswered("no_model", None))?;
                live.set(state)?;
                Ok(Some(Next::Terminal {
                    outcome: no_model_outcome(message),
                }))
            },
            context,
        )
        .await
}

/// Append a committed partial left by an interrupted, aborted, faulted, or orphaned attempt as an aborted assistant
/// entry; the caller replaces or removes `generation`.
pub async fn convert_partial(
    tx: &Transaction,
    live: &DocDraft<LiveState>,
    conversation_id: ConversationId,
) -> Result<()> {
    let partial = live
        .get()?
        .generation
        .and_then(|generation| generation.message);
    let Some(mut message) = partial else {
        return Ok(());
    };
    message.stop_reason = StopReason::Aborted;
    append_assistant(tx, conversation_id, message).await?;
    Ok(())
}

struct Throttle {
    pending: Option<AssistantMessage>,
    timer: Option<tokio::task::JoinHandle<()>>,
    in_flight: Option<Shared<BoxFuture<'static, ()>>>,
    stopped: bool,
}

/// Partial commits of one streamed response: trailing writes at most every 100 ms with one commit in flight.
#[derive(Clone)]
struct PartialThrottle {
    runtime: Runtime,
    attempt: u32,
    context: Context,
    state: Arc<Mutex<Throttle>>,
}

impl PartialThrottle {
    fn offer(&self, partial: AssistantMessage) {
        let mut state = self.state.lock();
        state.pending = Some(partial);
        if state.timer.is_none() && state.in_flight.is_none() && !state.stopped {
            state.timer = Some(self.start_timer());
        }
    }

    fn start_timer(&self) -> tokio::task::JoinHandle<()> {
        let this = self.clone();
        let deadline = tokio::time::Instant::now() + Duration::from_millis(PARTIAL_THROTTLE_MS);
        tokio::spawn(async move {
            tokio::time::sleep_until(deadline).await;
            this.flush();
        })
    }

    fn flush(&self) {
        let mut state = self.state.lock();
        state.timer = None;
        let partial = state.pending.take();
        let Some(message) = partial else {
            return;
        };
        if state.stopped {
            return;
        }
        let this = self.clone();
        let commit = async move {
            let conversation_id = this.runtime.conversation_id();
            let attempt = this.attempt;
            let committed = this
                .runtime
                .commit(
                    move |tx, _| async move {
                        let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                        live.edit(|live| {
                            live.generation
                                .get_or_insert_with(|| GenerationStatus::attempt(attempt))
                                .message = Some(message);
                        })?;
                        Ok(None)
                    },
                    &this.context,
                )
                .await;
            if let Err(error) = committed {
                // Rejections after an abort mark or close are expected; the committed state stays consistent.
                if !this.runtime.signal().aborted() {
                    let _ = this.runtime.report(error);
                }
            }
            let mut state = this.state.lock();
            state.in_flight = None;
            if state.pending.is_some() && !state.stopped {
                state.timer = Some(this.start_timer());
            }
        }
        .boxed()
        .shared();
        state.in_flight = Some(commit.clone());
        drop(state);
        tokio::spawn(commit);
    }

    async fn stop(&self) {
        let in_flight = {
            let mut state = self.state.lock();
            state.stopped = true;
            if let Some(timer) = state.timer.take() {
                timer.abort();
            }
            state.in_flight.clone()
        };
        if let Some(in_flight) = in_flight {
            in_flight.await;
        }
    }
}

fn event_partial(event: &AssistantMessageEvent) -> Option<&AssistantMessage> {
    match event {
        AssistantMessageEvent::Start { partial }
        | AssistantMessageEvent::TextStart { partial, .. }
        | AssistantMessageEvent::TextDelta { partial, .. }
        | AssistantMessageEvent::TextEnd { partial, .. }
        | AssistantMessageEvent::ThinkingStart { partial, .. }
        | AssistantMessageEvent::ThinkingDelta { partial, .. }
        | AssistantMessageEvent::ThinkingEnd { partial, .. }
        | AssistantMessageEvent::ToolCallStart { partial, .. }
        | AssistantMessageEvent::ToolCallDelta { partial, .. }
        | AssistantMessageEvent::ToolCallEnd { partial, .. } => Some(partial),
        AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => None,
    }
}

/// Stream one request and return the terminal message. Partials commit as trailing writes at most every 100 ms with
/// one commit in flight; the throttle stops and awaits that commit before returning, so no stale partial lands after
/// the outcome.
async fn stream_response(
    runtime: &Runtime,
    model: &Model,
    messages: Vec<Message>,
    options: SimpleStreamOptions,
    attempt: u32,
    context: &Context,
) -> Result<AssistantMessage> {
    let throttle = PartialThrottle {
        runtime: runtime.clone(),
        attempt,
        context: context.clone(),
        state: Arc::new(Mutex::new(Throttle {
            pending: None,
            timer: None,
            in_flight: None,
            stopped: false,
        })),
    };
    let models: Models = runtime.models();
    let mut events = models.stream_simple(
        model,
        &ModelContext {
            system_prompt: None,
            messages,
            tools: None,
        },
        options,
    );
    while let Some(event) = events.next().await {
        // A partial without content, such as pi-ai's opening `start` event, shows nothing; a deferred response
        // never gets past it, so it never leaves a partial.
        let Some(partial) = event_partial(&event) else {
            continue;
        };
        if partial.content.is_empty() {
            continue;
        }
        throttle.offer(partial.clone());
    }
    let message = events.result().await;
    throttle.stop().await;
    Ok(message)
}

/// Classify a terminal provider message in one commit that also clears the partial.
async fn classify(
    runtime: &Runtime,
    request: Request,
    message: AssistantMessage,
    context: &Context,
) -> Result<()> {
    // An abort mark or close: the abort invocation or the reopened run handles the committed state.
    runtime.signal().throw_if_aborted()?;
    let conversation_id = runtime.conversation_id();
    let task_id = runtime.task_id().erase();
    let Request {
        attempt,
        compacted,
        model: ref_,
        cutoff,
        ..
    } = request.clone_head();
    if message.stop_reason == StopReason::Deferred
        && let Some(handle) = message.deferred.clone()
    {
        let earliest = runtime.now()? + handle.poll_after_ms.unwrap_or(DEFAULT_POLL_AFTER_MS);
        let poll_at = match request.poll_at {
            None => earliest,
            Some(previous) => earliest.max(previous + 1),
        };
        return runtime
            .commit(
                move |tx, _| async move {
                    let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                    live.edit(|live| {
                        live.generation = Some(GenerationStatus {
                            attempt,
                            message: None,
                            retry: None,
                            deferred: Some(super::live::DeferredStatus { poll_at }),
                        })
                    })?;
                    Ok(Some(Next::running(GenerationCheckpoint::Poll {
                        attempt,
                        compacted,
                        model: ref_,
                        cutoff,
                        handle,
                        poll_at,
                    })))
                },
                context,
            )
            .await;
    }
    runtime
        .hooks()
        .each(
            |hooks| hooks.after_response.clone(),
            |hook| hook(message.clone(), runtime.hook_api(), context.clone()),
        )
        .await?;
    let calls = tool_calls(&message);
    if message.stop_reason == StopReason::ToolUse && !calls.is_empty() {
        return start_tool_round(runtime, request, message, calls, context).await;
    }
    if matches!(
        message.stop_reason,
        StopReason::Stop | StopReason::Length | StopReason::ToolUse
    ) {
        return answer(runtime, message, context).await;
    }
    // The retry and compaction policies govern the next attempt, so they are read now rather than pinned at
    // preparation.
    let settings = runtime.settings();
    let overflow = message.stop_reason == StopReason::Error && is_context_overflow(&message, None);
    if overflow && compacted.is_none() && settings.compaction.enabled {
        let policy = settings.compaction;
        let view = runtime
            .context(conversation_id, context, Some(cutoff))
            .await?;
        if select_cut(&view, policy.keep_recent_tokens).is_some() {
            let text = message
                .error_message
                .clone()
                .unwrap_or_else(|| "Context overflow".to_string());
            return runtime
                .commit(
                    move |tx, _| async move {
                        let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                        append_assistant(&tx, conversation_id, message).await?;
                        live.edit(|live| live.generation = None)?;
                        let child = create_compaction(
                            &tx,
                            conversation_id,
                            CompactionInput {
                                reason: CompactionReason::Overflow,
                                instructions: None,
                            },
                            Some(task_id),
                        )
                        .await?;
                        Ok(Some(Next::waiting(
                            GenerationCheckpoint::Prepare {
                                attempt,
                                compacted: Some(child),
                                overflow: Some(text),
                            },
                            vec![child.erase()],
                            JoinPolicy::AllSettled,
                        )))
                    },
                    context,
                )
                .await;
        }
    }
    let policy = settings.retry;
    // An overflow is never retried: only a compaction can make the next request fit.
    let retry = message.stop_reason == StopReason::Error
        && !overflow
        && is_retryable_assistant_error(&message)
        && policy.enabled
        && attempt <= policy.max_retries;
    let until = if retry {
        runtime.now()? + retry_delay_ms(policy.base_delay_ms, policy.max_agent_delay_ms, attempt)
    } else {
        0
    };
    runtime
        .commit(
            move |tx, _| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                let error_message = message.error_message.clone();
                let stop_reason = message.stop_reason;
                append_assistant(&tx, conversation_id, message).await?;
                if retry {
                    live.edit(|live| {
                        live.generation = Some(GenerationStatus {
                            attempt,
                            message: None,
                            retry: Some(RetryStatus {
                                at: until,
                                error: error_message.clone().unwrap_or_default(),
                            }),
                            deferred: None,
                        })
                    })?;
                    return Ok(Some(Next::running(GenerationCheckpoint::Retry {
                        attempt,
                        compacted,
                        until,
                    })));
                }
                let text = error_message.unwrap_or_else(|| {
                    format!(
                        "Model response ended with stop reason {}",
                        stop_reason_str(stop_reason)
                    )
                });
                let mut state = live.get()?;
                end_run(
                    &tx,
                    &mut state,
                    task_id,
                    unanswered("model_error", Some(text.clone())),
                )?;
                live.set(state)?;
                Ok(Some(Next::Terminal {
                    outcome: model_error(text),
                }))
            },
            context,
        )
        .await
}

impl Request {
    fn clone_head(&self) -> Request {
        Request {
            attempt: self.attempt,
            compacted: self.compacted,
            model: self.model.clone(),
            cutoff: self.cutoff,
            messages: None,
            poll_at: self.poll_at,
        }
    }
}

/// A final answer; the final boundary places queued items (spec §6). The first `onYield` continuation appends a user
/// message and hands the run to a successor generation, but only when the boundary selected no user item and no
/// reset. Otherwise the run's inputs settle `done`, and selected user items start the next run.
async fn answer(runtime: &Runtime, message: AssistantMessage, context: &Context) -> Result<()> {
    let continuation: Arc<Mutex<Option<UserInput>>> = Arc::default();
    runtime
        .hooks()
        .each(
            |hooks| hooks.on_yield.clone(),
            |hook| {
                let continuation = continuation.clone();
                let message = message.clone();
                let api = runtime.hook_api();
                let context = context.clone();
                async move {
                    if continuation.lock().is_some() {
                        return Ok(());
                    }
                    let next = hook(message, api, context).await?;
                    *continuation.lock() = next;
                    Ok(())
                }
            },
        )
        .await?;
    let continuation = continuation.lock().take();
    let conversation_id = runtime.conversation_id();
    let task_id = runtime.task_id().erase();
    let line_runtime = runtime.clone();
    runtime
        .commit(
            move |tx, _| async move {
                // Queue modes are read on the Session line, when the boundary is decided.
                let mut boundary = prepare_boundary(
                    &tx,
                    conversation_id,
                    QueueModes::from(&line_runtime.settings()),
                )
                .await?;
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                let entry = append_assistant(&tx, conversation_id, message).await?;
                let result = Next::completed(GenerationResult { entry_id: entry.id });
                let now = line_runtime.now()?;
                let placed = apply_boundary(&tx, &mut boundary, BoundaryKind::Final, now).await?;
                if let Some(continuation) = continuation
                    && placed.users.is_empty()
                    && !placed.reset
                {
                    let user = UserMessage {
                        content: continuation,
                        timestamp: line_runtime.now()?,
                    };
                    tx.append_entry_of(
                        &USER_ENTRY,
                        conversation_id,
                        TypedEntryDraft {
                            model: Some(vec![Message::User(user)]),
                            ..TypedEntryDraft::default()
                        },
                    )
                    .await?;
                    let successor = create_generation(&tx, conversation_id).await?;
                    live.edit(|live| {
                        hand_over(live, task_id, successor.erase());
                        live.generation = None;
                    })?;
                    return Ok(Some(result));
                }
                let mut state = live.get()?;
                end_run(
                    &tx,
                    &mut state,
                    task_id,
                    SubmissionSettlement::Done { answer: entry.id },
                )?;
                live.set(state)?;
                if !placed.users.is_empty() {
                    start_run(&tx, conversation_id, &live, placed.users).await?;
                }
                Ok(Some(result))
            },
            context,
        )
        .await
}

/// Append the tool-calling answer and start its tool round in one commit (spec §8.3). A call to a tool the request did
/// not offer gets its `tool_unavailable` result here; every other call gets a tool task owned by the generation, only
/// the first one now when the round is sequential. The generation then waits for them in its `tools` phase, keeping
/// the run.
async fn start_tool_round(
    runtime: &Runtime,
    request: Request,
    message: AssistantMessage,
    calls: Vec<ToolCall>,
    context: &Context,
) -> Result<()> {
    let conversation_id = runtime.conversation_id();
    let messages = match request.messages {
        Some(messages) => messages,
        None => {
            runtime
                .context(conversation_id, context, Some(request.cutoff))
                .await?
                .messages
        }
    };
    let offered: HashSet<String> = get_current_tools(&messages)
        .into_iter()
        .map(|tool| tool.name)
        .collect();
    // Read as the round starts; a tool is resolved as its tool task resolves it.
    let tools = runtime.agent(context).await?.tools;
    let sequential = runtime.settings().tool_execution == ToolExecutionMode::Sequential
        || calls.iter().any(|call| {
            offered.contains(&call.name)
                && tools
                    .iter()
                    .find(|tool| tool.name == call.name)
                    .and_then(|tool| tool.execution_mode)
                    == Some(ToolExecutionMode::Sequential)
        });
    let owner = runtime.task_id().erase();
    let now = runtime.now()?;
    runtime
        .commit(
            move |tx, _| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                let entry = append_assistant(&tx, conversation_id, message).await?;
                let mut slots = Vec::new();
                let mut tools: Vec<TaskId<ToolTaskResult>> = Vec::new();
                let mut pending = Vec::new();
                for call in &calls {
                    if !offered.contains(&call.name) {
                        let unavailable = harness_error(
                            "tool_unavailable",
                            &format!("Tool {} is not available", call.name),
                        );
                        let result =
                            append_tool_result(&tx, conversation_id, call, unavailable, now)
                                .await?;
                        let mut slot = ToolSlot::new(&call.id, &call.name, SlotStatus::Done);
                        slot.entry = Some(result.id);
                        slots.push(slot);
                        continue;
                    }
                    if sequential && !tools.is_empty() {
                        pending.push(call.id.clone());
                        slots.push(ToolSlot::new(&call.id, &call.name, SlotStatus::Pending));
                        continue;
                    }
                    let task_id = create_tool_task(&tx, owner, entry.id, call.id.clone()).await?;
                    tools.push(task_id);
                    let mut slot = ToolSlot::new(&call.id, &call.name, SlotStatus::Pending);
                    slot.task_id = Some(task_id.erase());
                    slots.push(slot);
                }
                live.edit(|live| {
                    live.generation = None;
                    live.tools = Some(slots);
                })?;
                let on = tools.iter().map(|id| id.erase()).collect();
                Ok(Some(Next::waiting(
                    GenerationCheckpoint::Tools {
                        assistant: entry.id,
                        tools,
                        pending,
                    },
                    on,
                    JoinPolicy::AllSettled,
                )))
            },
            context,
        )
        .await
}

/// The round's tools are terminal: apply their controls and either end the run at the final boundary (`terminate`,
/// `handoff`, or a queued reset) or hand it to the next generation at the `postTools` boundary (spec §8.5).
async fn finish_tool_round(
    runtime: &Runtime,
    assistant: EntryId,
    tools: Vec<TaskId<ToolTaskResult>>,
    context: &Context,
) -> Result<()> {
    let conversation_id = runtime.conversation_id();
    let outcomes = runtime.outcomes(&tools, context).await?;
    let controls: Vec<(TaskId, Option<ToolControl>)> = tools
        .iter()
        .zip(outcomes)
        .map(|(id, outcome)| {
            let control = match outcome {
                TaskOutcome::Completed { result } => {
                    serde_json::from_value::<ToolTaskResult>(result)
                        .ok()
                        .and_then(|result| result.control)
                }
                _ => None,
            };
            (id.erase(), control)
        })
        .collect();
    let slots = runtime
        .snapshot(&*LIVE_DOC, conversation_id, context)
        .await?
        .and_then(|live| live.tools)
        .unwrap_or_default();
    let results: Vec<EntryId> = slots.iter().filter_map(|slot| slot.entry).collect();
    runtime
        .hooks()
        .each(
            |hooks| hooks.after_tools.clone(),
            |hook| {
                hook(
                    assistant,
                    results.clone(),
                    runtime.hook_api(),
                    context.clone(),
                )
            },
        )
        .await?;
    let control_of = |id: TaskId| {
        controls
            .iter()
            .find(|(task, _)| *task == id)
            .and_then(|(_, control)| control.as_ref())
    };
    // Every call of the round, including those answered without a task, must ask to terminate.
    let terminate = !slots.is_empty()
        && slots.iter().all(|slot| {
            slot.task_id.is_some_and(|id| {
                control_of(id).and_then(|control| control.terminate) == Some(true)
            })
        });
    let added: Vec<String> = controls
        .iter()
        .flat_map(|(_, control)| {
            control
                .as_ref()
                .and_then(|control| control.add_tools.clone())
                .unwrap_or_default()
        })
        .collect();
    // The last handoff in call order wins.
    let handoff = controls
        .iter()
        .rev()
        .find_map(|(_, control)| control.as_ref().and_then(|control| control.handoff.clone()));
    let task_id = runtime.task_id().erase();
    let line_runtime = runtime.clone();
    runtime
        .commit(
            move |tx, _| async move {
                let mut boundary = prepare_boundary(
                    &tx,
                    conversation_id,
                    QueueModes::from(&line_runtime.settings()),
                )
                .await?;
                if !added.is_empty() {
                    add_tools(&tx, conversation_id, &added).await?;
                }
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                let now = line_runtime.now()?;
                if terminate || handoff.is_some() {
                    if let Some(handoff) = handoff {
                        let message = UserMessage {
                            content: UserMessageContent::Text(handoff),
                            timestamp: now,
                        };
                        let entry = tx
                            .append_entry_of(
                                &RESET_ENTRY,
                                conversation_id,
                                TypedEntryDraft {
                                    head: Some(EntryHead::SelfEntry),
                                    model: Some(vec![Message::User(message)]),
                                    ..TypedEntryDraft::default()
                                },
                            )
                            .await?;
                        boundary.head = Some(entry.id);
                    }
                    let placed =
                        apply_boundary(&tx, &mut boundary, BoundaryKind::Final, now).await?;
                    let mut state = live.get()?;
                    end_run(
                        &tx,
                        &mut state,
                        task_id,
                        SubmissionSettlement::Done { answer: assistant },
                    )?;
                    live.set(state)?;
                    if !placed.users.is_empty() {
                        start_run(&tx, conversation_id, &live, placed.users).await?;
                    }
                } else {
                    let placed =
                        apply_boundary(&tx, &mut boundary, BoundaryKind::PostTools, now).await?;
                    if placed.reset {
                        // The queued reset cut the run's context before an answer.
                        let mut state = live.get()?;
                        end_run(&tx, &mut state, task_id, unanswered("reset", None))?;
                        live.set(state)?;
                        if !placed.users.is_empty() {
                            start_run(&tx, conversation_id, &live, placed.users).await?;
                        }
                    } else {
                        let successor = create_generation(&tx, conversation_id).await?;
                        live.edit(|live| {
                            live.tools = None;
                            if let Some(run) = &mut live.run
                                && run.task_id == task_id
                            {
                                run.inputs.extend(placed.users);
                            }
                            hand_over(live, task_id, successor.erase());
                        })?;
                    }
                }
                Ok(Some(Next::completed(GenerationResult {
                    entry_id: assistant,
                })))
            },
            context,
        )
        .await
}

/// Append a provider result and add its usage to `pi.usage` in the same commit.
/// REMINDER: every built-in writer of assistant entries goes through here, so the usage ledger stays complete.
pub async fn append_assistant(
    tx: &Transaction,
    conversation_id: ConversationId,
    message: AssistantMessage,
) -> Result<EntryRecord> {
    let key = format!("{}/{}", message.provider, message.model);
    record_usage(
        tx,
        conversation_id,
        UsageBucket::Models,
        &key,
        &message.usage,
    )
    .await?;
    tx.append_entry_of(
        &ASSISTANT_ENTRY,
        conversation_id,
        TypedEntryDraft {
            model: Some(vec![Message::Assistant(message)]),
            ..TypedEntryDraft::default()
        },
    )
    .await
}

/// Start a run for `inputs`, placed input submissions: a new generation takes `pi.live.run`.
pub async fn start_run(
    tx: &Transaction,
    conversation_id: ConversationId,
    live: &DocDraft<LiveState>,
    inputs: Vec<SubmissionId>,
) -> Result<()> {
    let task_id = create_generation(tx, conversation_id).await?;
    live.edit(|live| {
        live.run = Some(RunState {
            task_id: task_id.erase(),
            inputs,
        })
    })
}

/// A generation owned by its conversation.
pub async fn create_generation(
    tx: &Transaction,
    conversation_id: ConversationId,
) -> Result<TaskId<GenerationResult>> {
    tx.create_task(
        &GENERATION_TASK,
        JsonObject::new(),
        TaskOptions {
            ownership: TaskOwnership::Conversation,
            conversation_id: Some(conversation_id),
            background: None,
        },
    )
    .await
}

/// Hand run control from `from` to `to`; the run's inputs move with it.
pub fn hand_over(live: &mut LiveState, from: TaskId, to: TaskId) {
    if let Some(run) = &mut live.run
        && run.task_id == from
    {
        run.task_id = to;
    }
}
