//! Port of durable `src/harness/compaction.ts`: the built-in compaction task (spec §8.7).
//!
//! Divergence from Pi: a negative `reserve_tokens` gives a summary request `max_tokens` of 0 (TS sends the negative
//! `Math.floor(0.8 * reserveTokens)` as is).

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::chord::{Context, JsonValue};
use crate::durable::entries::COMPACTION_ENTRY;
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, EntryId, TaskId};
use crate::durable::session::Transaction;
use crate::durable::tasks::{NextTaskState, RunningTask, Task, TaskDefinition, define_task};
use crate::durable::types::{
    EntryDraft, EntryHead, TaskOptions, TaskOutcome, TaskOutcomeError, TaskOwnership,
};
use crate::types::{
    AssistantContent, AssistantMessage, CacheRetention, Context as ModelContext, Message,
    ModelThinkingLevel, SimpleStreamOptions, StopReason, SystemMessage, SystemMessageContent,
    UserContent, UserMessage, UserMessageContent,
};
use crate::utils::estimate::{calculate_context_tokens, estimate_message_tokens};
use crate::utils::retry::{is_retryable_assistant_error, retry_delay_ms};

use super::context::order_tool_results;
use super::inbox::QueueModes;
use super::live::{
    CompactionStatus, LIVE_DOC, LiveState, RetryStatus, add_compaction_status, compaction_status,
    remove_compaction_status,
};
use super::provider::ensure_provider_session_id;
use super::scheduler::TaskRuntime;
use super::submissions::admit_submission;
use super::types::{
    CompactionDecision, CompactionHooks, CompactionReason, CompactionRequest, CompactionResult,
    ContextView, ConversationStreamOptions, ModelRef, SubmissionDraft,
};
use super::usage::{UsageBucket, record_usage};

/// `CompactionInput`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionInput {
    pub reason: CompactionReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// The pinned summarization request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryRequest {
    pub attempt: u32,
    pub model: ModelRef,
    pub thinking_level: ModelThinkingLevel,
    pub stream_options: ConversationStreamOptions,
    pub max_tokens: u64,
    /// Newest entry of the context the range was selected from.
    pub tail: EntryId,
    /// First entry kept verbatim; the summary's `head`.
    pub first_kept: EntryId,
}

/// `CompactionCheckpoint`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
pub enum CompactionCheckpoint {
    Select,
    Summarize(SummaryRequest),
    Retry {
        until: u64,
        #[serde(flatten)]
        request: SummaryRequest,
    },
}

type Runtime =
    TaskRuntime<CompactionInput, CompactionCheckpoint, CompactionResult, CompactionHooks>;
type Next = NextTaskState<CompactionCheckpoint>;
type Current = RunningTask<CompactionInput, CompactionCheckpoint, CompactionResult>;

pub type CompactionTaskType =
    Task<CompactionInput, CompactionCheckpoint, CompactionResult, CompactionHooks>;

/// Longest tool result text a serialized summary source keeps, in UTF-16 code units.
const TOOL_RESULT_MAX_CHARS: usize = 2000;

const SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
const SUMMARY_SUFFIX: &str = "\n</summary>";

const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work. If the conversation starts with an earlier summary, preserve its information and fold the newer messages into it.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or \"(none)\" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or \"(none)\" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.";

/// Built-in compaction task (spec §8.7): select an old prefix of the model context, summarize it, and place a summary
/// entry whose `head` is the first kept entry. A compaction the generation owns blocks it and appends directly; a
/// conversation-owned one places its summary through a write submission.
pub static COMPACTION_TASK: LazyLock<CompactionTaskType> = LazyLock::new(|| {
    define_task(
        TaskDefinition::new("pi.compaction", 1, |_: &CompactionInput| {
            CompactionCheckpoint::Select
        })
        .phase("select", |task, runtime, context| async move {
            select_phase(task, runtime, context).await
        })
        .phase("summarize", |task, runtime, context| async move {
            summarize_phase(task, runtime, context).await
        })
        .phase("retry", |task, runtime: Runtime, context| async move {
            let CompactionCheckpoint::Retry { until, request } = task.checkpoint else {
                return Err(Error::type_error(
                    "Compaction task is not in its retry phase",
                ));
            };
            runtime.sleep(until, &context).await?;
            let attempt = request.attempt + 1;
            let conversation_id = runtime.conversation_id();
            let task_id = runtime.task_id().erase();
            runtime
                .commit(
                    move |tx, _| async move {
                        let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                        live.edit(|live| {
                            if let Some(status) = compaction_status(live, task_id) {
                                status.attempt = attempt;
                                status.retry = None;
                            }
                        })?;
                        Ok(Some(Next::running(CompactionCheckpoint::Summarize(
                            SummaryRequest { attempt, ..request },
                        ))))
                    },
                    &context,
                )
                .await
        })
        .abort(|_task, runtime: Runtime, context| async move {
            let conversation_id = runtime.conversation_id();
            let task_id = runtime.task_id().erase();
            runtime
                .commit(
                    move |tx, _| async move {
                        let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                        live.edit(|live| remove_compaction_status(live, task_id))?;
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
        }),
    )
});

async fn select_phase(task: Current, runtime: Runtime, context: Context) -> Result<()> {
    let conversation_id = runtime.conversation_id();
    let agent = runtime.agent(&context).await?;
    let settings = runtime.settings();
    let ref_ = agent.model.clone();
    let model = ref_
        .as_ref()
        .and_then(|ref_| runtime.models().get_model(&ref_.provider, &ref_.model_id));
    let (Some(ref_), Some(model)) = (ref_.clone(), model) else {
        return fail_no_model(&runtime, ref_.as_ref(), &context).await;
    };
    let policy = settings.compaction;
    let view = runtime.context(conversation_id, &context, None).await?;
    let Some(cut) = select_cut(&view, policy.keep_recent_tokens) else {
        return complete(&runtime, &context).await;
    };
    let first_kept = view.entries[cut].id;
    let CompactionInput {
        reason,
        instructions,
    } = task.input;
    let compaction = CompactionRequest {
        reason,
        entries: view.entries[..cut].to_vec(),
        messages: summarized_messages(&view, cut),
        first_kept,
        instructions,
    };
    let decision: Arc<Mutex<Option<CompactionDecision>>> = Arc::default();
    runtime
        .hooks()
        .each(
            |hooks| hooks.before_compact.clone(),
            |hook| {
                let decision = decision.clone();
                let compaction = compaction.clone();
                let api = runtime.hook_api();
                let context = context.clone();
                async move {
                    if decision.lock().is_none() {
                        let decided = hook(compaction, api, context).await?;
                        *decision.lock() = decided;
                    }
                    Ok(())
                }
            },
        )
        .await?;
    let decision = decision.lock().take();
    match decision {
        Some(CompactionDecision::Decline) => return complete(&runtime, &context).await,
        Some(CompactionDecision::Summary(summary)) => {
            return place(&runtime, first_kept, summary, &context).await;
        }
        None => {}
    }
    // TS passes a negative `Math.floor(0.8 * reserveTokens)` on as is; the unsigned request clamps it to 0.
    let reserve = (0.8 * policy.reserve_tokens as f64).floor().max(0.0) as u64;
    let request = SummaryRequest {
        attempt: 1,
        model: ref_,
        thinking_level: agent.thinking_level,
        stream_options: settings.stream,
        max_tokens: if model.max_tokens > 0 {
            reserve.min(u64::from(model.max_tokens))
        } else {
            reserve
        },
        tail: view
            .entries
            .iter()
            .fold(first_kept, |tail, entry| entry.id.max(tail)),
        first_kept,
    };
    runtime
        .commit(
            move |_, _| async move {
                Ok(Some(Next::running(CompactionCheckpoint::Summarize(
                    request,
                ))))
            },
            &context,
        )
        .await
}

async fn summarize_phase(task: Current, runtime: Runtime, context: Context) -> Result<()> {
    let CompactionCheckpoint::Summarize(request) = task.checkpoint else {
        return Err(Error::type_error(
            "Compaction task is not in its summarize phase",
        ));
    };
    let Some(model) = runtime
        .models()
        .get_model(&request.model.provider, &request.model.model_id)
    else {
        return fail_no_model(&runtime, Some(&request.model), &context).await;
    };
    // The context at `tail` is immutable, so this is the range `select` chose.
    let view = runtime
        .context(runtime.conversation_id(), &context, Some(request.tail))
        .await?;
    let cut = view
        .entries
        .iter()
        .position(|entry| entry.id == request.first_kept)
        // Pi's `findIndex` gives -1, and `slice(0, -1)` drops the last contribution.
        .unwrap_or(view.contributions.len().saturating_sub(1));
    let now = runtime.now()?;
    let messages = vec![
        Message::System(SystemMessage {
            content: SystemMessageContent::Text(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
            sections: None,
            tools_added: None,
            tools_removed: None,
            timestamp: now,
        }),
        Message::User(UserMessage {
            content: UserMessageContent::Parts(vec![UserContent::text(summary_prompt(
                &summarized_messages(&view, cut),
                task.input.instructions.as_deref(),
            ))]),
            timestamp: now,
        }),
    ];
    let mut forwarded = request.stream_options.clone();
    forwarded.deferred = None;
    let mut options = simple_stream_options(&forwarded, request.thinking_level);
    options.stream.cache_retention = Some(CacheRetention::None);
    options.stream.max_tokens = Some(request.max_tokens.min(u64::from(u32::MAX)) as u32);
    options.stream.signal = Some(runtime.signal().token());
    options.stream.session_id = Some(ensure_provider_session_id(runtime.core(), &context).await?);
    let message = runtime
        .models()
        .complete_simple(
            &model,
            &ModelContext {
                system_prompt: None,
                messages,
                tools: None,
            },
            options,
        )
        .await;
    // An abort mark or close: the abort invocation or the reopened task handles the committed state.
    runtime.signal().throw_if_aborted()?;
    let summary = summary_text(&message);
    let policy = runtime.settings().retry;
    let attempt = request.attempt;
    let retry = message.stop_reason == StopReason::Error
        && is_retryable_assistant_error(&message)
        && policy.enabled
        && attempt <= policy.max_retries;
    let until = if retry {
        runtime.now()? + retry_delay_ms(policy.base_delay_ms, policy.max_agent_delay_ms, attempt)
    } else {
        0
    };
    let conversation_id = runtime.conversation_id();
    let task_id = runtime.task_id().erase();
    let place_runtime = runtime.clone();
    runtime
        .commit(
            move |tx, current| async move {
                record_usage(
                    &tx,
                    conversation_id,
                    UsageBucket::Models,
                    &format!("{}/{}", message.provider, message.model),
                    &message.usage,
                )
                .await?;
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                if let Some(summary) = summary {
                    let first_kept = request.first_kept;
                    return place_summary(
                        &tx,
                        &place_runtime,
                        &current,
                        &live,
                        first_kept,
                        &summary,
                    )
                    .await
                    .map(Some);
                }
                if retry {
                    live.edit(|live| {
                        if let Some(status) = compaction_status(live, task_id) {
                            status.retry = Some(RetryStatus {
                                at: until,
                                error: message.error_message.clone().unwrap_or_default(),
                            });
                        }
                    })?;
                    return Ok(Some(Next::running(CompactionCheckpoint::Retry {
                        until,
                        request,
                    })));
                }
                live.edit(|live| remove_compaction_status(live, task_id))?;
                let text = summary_failure(&message);
                Ok(Some(Next::Terminal {
                    outcome: model_error(text),
                }))
            },
            &context,
        )
        .await
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

/// pi-ai request options from the curated settings options, with `reasoning` unless thinking is off.
pub fn simple_stream_options(
    options: &ConversationStreamOptions,
    thinking_level: ModelThinkingLevel,
) -> SimpleStreamOptions {
    let mut simple = SimpleStreamOptions::default();
    simple.stream.transport = options.transport;
    simple.stream.timeout_ms = options.timeout_ms;
    simple.stream.max_retries = options.max_retries;
    simple.stream.max_retry_delay_ms = options.max_retry_delay_ms;
    simple.stream.headers = options.headers.as_ref().map(|headers| {
        headers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    });
    simple.stream.metadata = options.metadata.clone();
    simple.stream.cache_retention = options.cache_retention;
    simple.deferred = options.deferred.as_ref().map(|deferred| match deferred {
        super::types::DeferredOption::Enabled(enabled) => {
            crate::types::DeferredRequest::Flag(*enabled)
        }
        super::types::DeferredOption::Window { window } => {
            crate::types::DeferredRequest::Window(window.as_deref().and_then(|window| {
                serde_json::from_value(JsonValue::String(window.to_string())).ok()
            }))
        }
    });
    simple.reasoning = thinking_level.thinking_level();
    simple
}

/// Create a compaction task with its status in this commit. `owner` is the generation that waits for it (a blocking
/// compaction); without one it is conversation-owned, and `background` unless it is manual.
pub async fn create_compaction(
    tx: &Transaction,
    conversation_id: ConversationId,
    input: CompactionInput,
    owner: Option<TaskId>,
) -> Result<TaskId<CompactionResult>> {
    let ownership = match owner {
        None => TaskOwnership::Conversation,
        Some(task_id) => TaskOwnership::Task { task_id },
    };
    let background = owner.is_none() && input.reason != CompactionReason::Manual;
    let reason = input.reason;
    let task_id = tx
        .create_task(
            &COMPACTION_TASK,
            input,
            TaskOptions {
                ownership,
                conversation_id: Some(conversation_id),
                background: Some(background),
            },
        )
        .await?;
    let status = CompactionStatus {
        task_id,
        reason,
        blocking: owner.is_some(),
        attempt: 1,
        retry: None,
    };
    let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
    live.edit(|live| add_compaction_status(live, status))?;
    Ok(task_id)
}

/// Index in `view.entries` of the first entry a summary keeps, or `None` when there is nothing to compact
/// (spec §8.7). Walks back from the tail until `keep_recent_tokens` are kept, then cuts at the first candidate at or
/// after that entry: an entry whose contribution starts with a user or assistant message, never a tool result, and
/// never a user entry that a result of the preceding assistant's calls still follows.
pub fn select_cut(view: &ContextView, keep_recent_tokens: u64) -> Option<usize> {
    let contributions = &view.contributions;
    let start = usize::from(view.head.is_some());
    let candidates: Vec<usize> = (start..contributions.len())
        .filter(|index| is_candidate(contributions, *index))
        .collect();
    let mut kept: u64 = 0;
    let mut cut = None;
    for index in (start..contributions.len()).rev() {
        for message in &contributions[index] {
            kept += u64::from(estimate_message_tokens(message));
        }
        if kept < keep_recent_tokens {
            continue;
        }
        cut = candidates
            .iter()
            .copied()
            .find(|candidate| *candidate >= index)
            .or_else(|| candidates.last().copied());
        break;
    }
    let cut = cut?;
    (start..cut)
        .any(|index| !contributions[index].is_empty())
        .then_some(cut)
}

fn is_candidate(contributions: &[Vec<Message>], index: usize) -> bool {
    match contributions[index].first() {
        Some(Message::Assistant(_)) => return true,
        Some(Message::User(_)) => {}
        _ => return false,
    }
    // A result of the preceding assistant's calls that follows this entry, before the next assistant, belongs before
    // it.
    let mut calls: HashSet<&str> = HashSet::new();
    for before in (0..index).rev() {
        let assistant = contributions[before]
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::Assistant(assistant) => Some(assistant),
                _ => None,
            });
        let Some(assistant) = assistant else {
            continue;
        };
        calls = assistant
            .content
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(call.id.as_str()),
                _ => None,
            })
            .collect();
        break;
    }
    if calls.is_empty() {
        return true;
    }
    for (after, contribution) in contributions.iter().enumerate().skip(index) {
        for (position, message) in contribution.iter().enumerate() {
            match message {
                Message::Assistant(_) if after > index || position > 0 => return true,
                Message::ToolResult(result) if calls.contains(result.tool_call_id.as_str()) => {
                    return false;
                }
                _ => {}
            }
        }
    }
    true
}

/// Model messages of the entries before `cut`: the head marker first, ordered like model context (spec §2.1).
pub fn summarized_messages(view: &ContextView, cut: usize) -> Vec<Message> {
    let flat: Vec<Message> = view.contributions[..cut.min(view.contributions.len())]
        .iter()
        .flatten()
        .cloned()
        .collect();
    order_tool_results(&flat)
}

/// Size of a request over `view` followed by `extra` (spec §8.3): the usage of the newest assistant appended after the
/// head marker, whose request included the marker, plus estimates of the messages after it; without one, estimates of
/// every message.
pub fn estimate_context(view: &ContextView, extra: &[Message]) -> u64 {
    let mut measured: Option<&AssistantMessage> = None;
    for index in (0..view.entries.len()).rev() {
        if measured.is_some() {
            break;
        }
        if view
            .head
            .as_ref()
            .is_some_and(|head| view.entries[index].id <= head.id)
        {
            continue;
        }
        measured = view.contributions[index]
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::Assistant(assistant) if calculate_context_tokens(&assistant.usage) > 0 => {
                    Some(assistant)
                }
                _ => None,
            });
    }
    let (from, mut tokens) = match measured {
        None => (0, 0u64),
        // Pi's `lastIndexOf` finds the measured object by identity; Rust compares by value, which differs only when
        // a deep-equal assistant message follows it.
        Some(measured) => (
            view.messages
                .iter()
                .rposition(|message| matches!(message, Message::Assistant(m) if m == measured))
                .map_or(0, |index| index + 1),
            u64::from(calculate_context_tokens(&measured.usage)),
        ),
    };
    for message in view.messages.iter().skip(from) {
        tokens += u64::from(estimate_message_tokens(message));
    }
    for message in extra {
        tokens += u64::from(estimate_message_tokens(message));
    }
    tokens
}

/// The summary of a clean `stop` with text and no tool call; anything else is not a summary.
fn summary_text(message: &AssistantMessage) -> Option<String> {
    if message.stop_reason != StopReason::Stop
        || message
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_)))
    {
        return None;
    }
    let text: Vec<&str> = message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    let text = text.join("\n").trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn summary_failure(message: &AssistantMessage) -> String {
    if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
        return format!(
            "Summarization failed: {}",
            message
                .error_message
                .clone()
                .unwrap_or_else(|| stop_reason_str(message.stop_reason).to_string())
        );
    }
    if message.stop_reason == StopReason::Length {
        return "Summarization hit the token limit; the summary is incomplete".to_string();
    }
    if message
        .content
        .iter()
        .any(|content| matches!(content, AssistantContent::ToolCall(_)))
    {
        return "Summarization attempted to call a tool".to_string();
    }
    "Summarization produced no text".to_string()
}

/// The Pi string of a stop reason.
pub fn stop_reason_str(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Pending => "pending",
        StopReason::Stop => "stop",
        StopReason::Length => "length",
        StopReason::ToolUse => "toolUse",
        StopReason::Error => "error",
        StopReason::Aborted => "aborted",
        StopReason::Deferred => "deferred",
    }
}

/// The summarizer's user message: the serialized conversation, the prompt, and any instructions.
fn summary_prompt(messages: &[Message], instructions: Option<&str>) -> String {
    let focus = instructions.map_or(String::new(), |instructions| {
        format!("\n\nAdditional focus: {instructions}")
    });
    format!(
        "<conversation>\n{}\n</conversation>\n\n{SUMMARIZATION_PROMPT}{focus}",
        serialize_conversation(messages)
    )
}

/// Messages as plain text, so the summarizer reads a transcript instead of continuing it. System messages are
/// omitted.
pub fn serialize_conversation(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    for message in messages {
        match message {
            Message::User(user) => {
                let text = match &user.content {
                    UserMessageContent::Text(text) => text.clone(),
                    UserMessageContent::Parts(parts) => content_text(parts),
                };
                if !text.is_empty() {
                    parts.push(format!("[User]: {text}"));
                }
            }
            Message::Assistant(assistant) => {
                let mut thinking = Vec::new();
                let mut text = Vec::new();
                let mut calls = Vec::new();
                for content in &assistant.content {
                    match content {
                        AssistantContent::Thinking(block) => thinking.push(block.thinking.clone()),
                        AssistantContent::Text(block) => text.push(block.text.clone()),
                        AssistantContent::ToolCall(call) => {
                            let args: Vec<String> = call
                                .arguments
                                .as_object()
                                .map(|object| {
                                    object
                                        .iter()
                                        .map(|(key, value)| {
                                            format!(
                                                "{key}={}",
                                                serde_json::to_string(value).unwrap_or_default()
                                            )
                                        })
                                        .collect()
                                })
                                .unwrap_or_default();
                            calls.push(format!("{}({})", call.name, args.join(", ")));
                        }
                    }
                }
                if !thinking.is_empty() {
                    parts.push(format!("[Assistant thinking]: {}", thinking.join("\n")));
                }
                if !text.is_empty() {
                    parts.push(format!("[Assistant]: {}", text.join("\n")));
                }
                if !calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
                }
            }
            Message::ToolResult(result) => {
                let text = content_text(&result.content);
                if !text.is_empty() {
                    parts.push(format!(
                        "[Tool result]: {}",
                        truncate(&text, TOOL_RESULT_MAX_CHARS)
                    ));
                }
            }
            _ => {}
        }
    }
    parts.join("\n\n")
}

fn content_text(content: &[UserContent]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Truncate to `max_chars` UTF-16 code units (TS string length). Divergence: when a surrogate pair straddles
/// `max_chars`, Pi's `slice` keeps the lone high surrogate; a Rust `String` cannot hold one, so the pair is dropped
/// whole (the reported count is the same).
fn truncate(text: &str, max_chars: usize) -> String {
    let length: usize = text.chars().map(char::len_utf16).sum();
    if length <= max_chars {
        return text.to_string();
    }
    let mut units = 0;
    let mut end = text.len();
    for (index, c) in text.char_indices() {
        if units + c.len_utf16() > max_chars {
            end = index;
            break;
        }
        units += c.len_utf16();
    }
    format!(
        "{}\n\n[... {} more characters truncated]",
        &text[..end],
        length - max_chars
    )
}

/// Place a summary supplied by a hook in its own commit.
async fn place(
    runtime: &Runtime,
    first_kept: EntryId,
    summary: String,
    context: &Context,
) -> Result<()> {
    let place_runtime = runtime.clone();
    let conversation_id = runtime.conversation_id();
    runtime
        .commit(
            move |tx, current| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                place_summary(&tx, &place_runtime, &current, &live, first_kept, &summary)
                    .await
                    .map(Some)
            },
            context,
        )
        .await
}

/// `{ reason }` of a `pi.compaction` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionData {
    pub reason: CompactionReason,
}

/// Place the summary entry and complete (spec §8.7). A blocking compaction, owned by its generation, appends it: the
/// generation holds the run and waits. A conversation-owned one admits it as a write submission, placed at once when
/// idle, otherwise at the next boundary, or settled `stale`.
/// REMINDER: nothing else may append to a busy conversation, so every non-blocking summary goes through admission.
async fn place_summary(
    tx: &Transaction,
    runtime: &Runtime,
    current: &Current,
    live: &crate::durable::session::DocDraft<LiveState>,
    first_kept: EntryId,
    summary: &str,
) -> Result<Next> {
    let task_id = runtime.task_id().erase();
    live.edit(|live| remove_compaction_status(live, task_id))?;
    let text = format!("{SUMMARY_PREFIX}{summary}{SUMMARY_SUFFIX}");
    let now = runtime.now()?;
    let entry = EntryDraft {
        kind: COMPACTION_ENTRY.kind().to_string(),
        head: Some(EntryHead::Id(first_kept)),
        model: Some(vec![Message::User(UserMessage {
            content: UserMessageContent::Parts(vec![UserContent::text(text)]),
            timestamp: now,
        })]),
        data: Some(serde_json::to_value(CompactionData {
            reason: current.input.reason,
        })?),
        edits: None,
    };
    let conversation_id = runtime.conversation_id();
    let result = match current.owner {
        None => CompactionResult {
            entry_id: None,
            submission_id: Some(
                admit_submission(
                    tx,
                    conversation_id,
                    SubmissionDraft::write(entry).with_request_id(format!("compaction:{task_id}")),
                    now,
                    QueueModes::from(&runtime.settings()),
                )
                .await?,
            ),
        },
        Some(_) => CompactionResult {
            entry_id: Some(tx.append_entry(conversation_id, entry).await?.id),
            submission_id: None,
        },
    };
    Ok(Next::completed(result))
}

/// Remove the status and complete without a summary.
async fn complete(runtime: &Runtime, context: &Context) -> Result<()> {
    let conversation_id = runtime.conversation_id();
    let task_id = runtime.task_id().erase();
    runtime
        .commit(
            move |tx, _| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                live.edit(|live| remove_compaction_status(live, task_id))?;
                Ok(Some(Next::completed(CompactionResult::default())))
            },
            context,
        )
        .await
}

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
                live.edit(|live| remove_compaction_status(live, task_id))?;
                Ok(Some(Next::Terminal {
                    outcome: no_model_outcome(message),
                }))
            },
            context,
        )
        .await
}

/// `No model is configured` or `Model <provider>/<id> is not available`.
pub fn no_model_message(ref_: Option<&ModelRef>) -> String {
    match ref_ {
        None => "No model is configured".to_string(),
        Some(ref_) => format!("Model {}/{} is not available", ref_.provider, ref_.model_id),
    }
}

/// `failed` with `{ reason: "no_model" }`.
pub fn no_model_outcome(message: String) -> TaskOutcome {
    TaskOutcome::Failed {
        error: TaskOutcomeError {
            message,
            detail: Some(serde_json::json!({ "reason": "no_model" })),
        },
        result: None,
    }
}
