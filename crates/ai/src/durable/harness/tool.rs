//! Port of durable `src/harness/tool.ts`: the built-in tool task.
//!
//! Divergences from Pi:
//! - `ToolExecutionApi` is a cloneable handle; `output()` and `diagnostic()` return `Err` after the call settled (TS
//!   throws). `output_bytes()` takes the `Uint8Array` form of `output()`.
//! - `ToolExecutionApi.memo` is `memo` and `memo_with` (the TS overloads).
//! - Drafts are diffed when the commit prepares, so the `pi.live` slot is assigned as a whole value; the diff still
//!   writes an append for a grown output string (TS `assignJson`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::chord::delta::overlap;
use crate::chord::{Context, JsonValue, await_with_context};
use crate::durable::documents::{DocAccess, RewindableDocAccess};
use crate::durable::entries::{ASSISTANT_ENTRY, TOOL_RESULT_ENTRY};
use crate::durable::env::ExecutionEnv;
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, EntryId, TaskId};
use crate::durable::session::{DocumentWatch, Transaction};
use crate::durable::tasks::{NextTaskState, RunningTask, Task, TaskDefinition, define_task};
use crate::durable::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};
use crate::durable::types::{
    EntryRecord, JsonObject, TaskOptions, TaskOutcome, TaskOutcomeError, TaskRecord,
    TypedEntryDraft,
};
use crate::types::{
    AssistantContent, Message, TextContent, ToolCall, ToolResultMessage, UserContent,
};
use crate::utils::validation::validate_tool_arguments;

use super::harness::ConversationHandle;
use super::live::{LIVE_DOC, SlotStatus, ToolSlot, clear_progress, finish_slot, tool_slot};
use super::output::{OutputBuffer, OutputLimits, Progress, bound_output};
use super::registry::RegistrySnapshot;
use super::scheduler::TaskRuntime;
use super::types::{
    Agent, DiagnosticSeverity, Replay, Retain, ToolControl, ToolDiagnostic, ToolExecutionResult,
    ToolHooks, ToolRegistration,
};
use super::usage::{UsageBucket, record_usage};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolTaskInput {
    pub assistant: EntryId,
    pub call_id: String,
}

/// `ToolTaskCheckpoint`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
pub enum ToolTaskCheckpoint {
    Call,
    /// Durable intent: the final arguments and the replay policy recorded before execution.
    Execute {
        arguments: JsonObject,
        replay: Replay,
    },
}

/// `ToolTaskResult`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolTaskResult {
    pub entry_id: EntryId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control: Option<ToolControl>,
}

type Runtime = TaskRuntime<ToolTaskInput, ToolTaskCheckpoint, ToolTaskResult, ToolHooks>;
type Next = NextTaskState<ToolTaskCheckpoint>;
type Content = Vec<UserContent>;

pub type ToolTaskType = Task<ToolTaskInput, ToolTaskCheckpoint, ToolTaskResult, ToolHooks>;

/// Built-in tool task: resolves the called tool among its phase agent's tools, validates, runs `beforeTool`, records
/// intent, executes, runs `afterTool`, and appends the result, all in one `call` handler so nothing separates
/// resolution from settlement. `execute` is reached only by recovery and applies the replay rule.
pub static TOOL_TASK: LazyLock<ToolTaskType> = LazyLock::new(|| {
    define_task(
        TaskDefinition::new("pi.tool", 1, |_: &ToolTaskInput| ToolTaskCheckpoint::Call)
            .phase("call", |task, runtime, context| async move {
                call_phase(task, runtime, context).await
            })
            .phase("execute", |task, runtime, context| async move {
                execute_phase(task, runtime, context).await
            })
            .abort(|task, runtime: Runtime, context| async move {
                let call = read_call(&runtime, &task.input, &context).await?;
                let message = format!("Tool {} was aborted", call.name);
                settle(
                    &runtime,
                    &call,
                    Ending::Aborted,
                    move |slot| from_slot(slot, "aborted", &message),
                    &context,
                )
                .await
            }),
    )
});

async fn call_phase(
    task: RunningTask<ToolTaskInput, ToolTaskCheckpoint, ToolTaskResult>,
    runtime: Runtime,
    context: Context,
) -> Result<()> {
    let call = read_call(&runtime, &task.input, &context).await?;
    let tool = runtime
        .agent(&context)
        .await?
        .tools
        .into_iter()
        .find(|each| each.name == call.name);
    let Some(tool) = tool else {
        let error = harness_error(
            "tool_unavailable",
            &format!("Tool {} is not available", call.name),
        );
        return settle(&runtime, &call, Ending::Completed, move |_| error, &context).await;
    };
    let checked =
        prepare(&tool, call.arguments.clone()).and_then(|args| validate(&tool, &call, args));
    let args = match checked {
        Ok(args) => args,
        Err(error) => {
            return settle(
                &runtime,
                &call,
                Ending::Completed,
                move |_| invalid(&error),
                &context,
            )
            .await;
        }
    };
    let decided: Arc<Mutex<(JsonValue, Option<String>)>> = Arc::new(Mutex::new((args, None)));
    let signal = runtime.signal();
    runtime
        .hooks()
        .each(
            |hooks| hooks.before_tool.clone(),
            |hook| {
                let decided = decided.clone();
                let api = runtime.hook_api();
                let context = context.clone();
                let signal = signal.clone();
                let mut call = call.clone();
                async move {
                    if decided.lock().1.is_some() {
                        return Ok(());
                    }
                    call.arguments = decided.lock().0.clone();
                    match hook(call, api, context).await {
                        Ok(Some(decision)) => {
                            let mut decided = decided.lock();
                            if let Some(block) = decision.block {
                                decided.1 = Some(block);
                            } else if let Some(arguments) = decision.arguments {
                                decided.0 = JsonValue::Object(arguments);
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            if signal.aborted() {
                                return Err(error);
                            }
                            decided.lock().1 = Some(error_text(&error));
                        }
                    }
                    Ok(())
                }
            },
        )
        .await?;
    let (args, block) = decided.lock().clone();
    if let Some(block) = block {
        let blocked = harness_error("blocked", &format!("Tool call blocked: {block}"));
        return settle(
            &runtime,
            &call,
            Ending::Completed,
            move |_| blocked,
            &context,
        )
        .await;
    }
    let final_args = match validate(&tool, &call, args) {
        Ok(args) => args,
        Err(error) => {
            return settle(
                &runtime,
                &call,
                Ending::Completed,
                move |_| invalid(&error),
                &context,
            )
            .await;
        }
    };
    let conversation_id = runtime.conversation_id();
    let task_id = runtime.task_id().erase();
    let intent = ToolTaskCheckpoint::Execute {
        arguments: as_object(&final_args),
        replay: tool.replay.unwrap_or_default(),
    };
    runtime
        .commit(
            move |tx, _| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                live.edit(|live| {
                    if let Some(slot) = tool_slot(live, task_id) {
                        slot.status = SlotStatus::Running;
                    }
                })?;
                Ok(Some(Next::running(intent)))
            },
            &context,
        )
        .await?;
    run(&runtime, &call, &tool, final_args, &context).await
}

/// Recovery after intent: rerun only when the stored and the current policy both say `safe`.
async fn execute_phase(
    task: RunningTask<ToolTaskInput, ToolTaskCheckpoint, ToolTaskResult>,
    runtime: Runtime,
    context: Context,
) -> Result<()> {
    let ToolTaskCheckpoint::Execute { arguments, replay } = task.checkpoint else {
        return Err(Error::type_error("Tool task is not in its execute phase"));
    };
    let call = read_call(&runtime, &task.input, &context).await?;
    let tool = runtime
        .agent(&context)
        .await?
        .tools
        .into_iter()
        .find(|each| each.name == call.name);
    if let Some(tool) = tool
        && replay == Replay::Safe
        && tool.replay == Some(Replay::Safe)
    {
        // The rerun reports from scratch; clear what the interrupted attempt published.
        let conversation_id = runtime.conversation_id();
        let task_id = runtime.task_id().erase();
        runtime
            .commit(
                move |tx, _| async move {
                    let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                    live.edit(|live| {
                        if let Some(slot) = tool_slot(live, task_id) {
                            clear_progress(slot);
                        }
                    })?;
                    Ok(None)
                },
                &context,
            )
            .await?;
        return run(
            &runtime,
            &call,
            &tool,
            JsonValue::Object(arguments),
            &context,
        )
        .await;
    }
    let message = format!(
        "Tool {} was interrupted and may have partially run",
        call.name
    );
    // `failed` records cancellation intent, so the call's owned conversations, left unsupervised, are aborted.
    let ending = Ending::Failed(message.clone());
    settle(
        &runtime,
        &call,
        ending,
        move |slot| from_slot(slot, "interrupted", &message),
        &context,
    )
    .await
}

/// The tool call `call_id` of the assistant entry.
async fn read_call(
    runtime: &Runtime,
    input: &ToolTaskInput,
    context: &Context,
) -> Result<ToolCall> {
    let entry = runtime
        .entry_of(&ASSISTANT_ENTRY, input.assistant, context)
        .await?;
    let call = entry
        .and_then(|entry| entry.model)
        .and_then(|model| model.into_iter().next())
        .and_then(|message| match message {
            Message::Assistant(message) => {
                message
                    .content
                    .into_iter()
                    .find_map(|content| match content {
                        AssistantContent::ToolCall(call) if call.id == input.call_id => Some(call),
                        _ => None,
                    })
            }
            _ => None,
        });
    call.ok_or_else(|| {
        Error::message(format!(
            "Entry {} has no tool call {}",
            input.assistant, input.call_id
        ))
    })
}

fn as_object(value: &JsonValue) -> JsonObject {
    match value {
        JsonValue::Object(object) => object.clone(),
        _ => JsonObject::new(),
    }
}

/// The call's arguments as repaired by the tool; a failing repair makes them invalid.
fn prepare(tool: &ToolRegistration, args: JsonValue) -> std::result::Result<JsonValue, String> {
    match &tool.prepare_arguments {
        None => Ok(args),
        Some(prepare) => prepare(args).map_err(|error| error_text(&error)),
    }
}

/// Arguments validated and coerced against the implementation's schema.
fn validate(
    tool: &ToolRegistration,
    call: &ToolCall,
    args: JsonValue,
) -> std::result::Result<JsonValue, String> {
    let call = ToolCall {
        arguments: args,
        ..call.clone()
    };
    validate_tool_arguments(&tool.tool(), &call).map_err(|error| error.to_string())
}

fn invalid(message: &str) -> ToolExecutionResult {
    harness_error("invalid_arguments", message)
}

/// What a running tool reported through its api: output, the last details, and diagnostics.
struct Reported {
    output: OutputBuffer,
    diagnostics: Vec<ToolDiagnostic>,
    details: Option<JsonValue>,
}

struct ApiInner {
    runtime: Runtime,
    call_id: String,
    env: Option<Arc<dyn ExecutionEnv>>,
    reported: Arc<Mutex<Reported>>,
    progress: Progress,
    ended: Arc<AtomicBool>,
}

/// What a tool's execute function may use (`ToolExecutionApi`).
#[derive(Clone)]
pub struct ToolExecutionApi(Arc<ApiInner>);

impl ToolExecutionApi {
    fn assert_live(&self) -> Result<()> {
        if self.0.ended.load(Ordering::SeqCst) {
            return Err(Error::message(format!(
                "Tool call {} has settled",
                self.0.call_id
            )));
        }
        Ok(())
    }

    pub fn task_id(&self) -> TaskId {
        self.0.runtime.task_id().erase()
    }

    pub fn conversation_id(&self) -> ConversationId {
        self.0.runtime.conversation_id()
    }

    pub fn call_id(&self) -> &str {
        &self.0.call_id
    }

    /// The tool task's phase snapshot.
    pub fn registry(&self) -> RegistrySnapshot {
        self.0.runtime.registry()
    }

    /// The calling conversation's agent, as the tool task's phase resolved it.
    pub async fn agent(&self, context: &Context) -> Result<Agent> {
        self.0.runtime.agent(context).await
    }

    /// Built by `HarnessOptions.env` for this call; `None` without an environment.
    pub fn env(&self) -> Option<Arc<dyn ExecutionEnv>> {
        self.0.env.clone()
    }

    /// Append running output; it becomes the result content when the result omits `content`.
    pub fn output(&self, chunk: &str) -> Result<()> {
        self.assert_live()?;
        let accepted = self.0.reported.lock().output.push(chunk);
        if accepted {
            self.0.progress.mark();
        }
        Ok(())
    }

    /// `output(chunk)` with a byte chunk; incomplete characters wait for the next chunk.
    pub fn output_bytes(&self, chunk: &[u8]) -> Result<()> {
        self.assert_live()?;
        let accepted = self.0.reported.lock().output.push_bytes(chunk);
        if accepted {
            self.0.progress.mark();
        }
        Ok(())
    }

    /// Record a model-visible remark about this call.
    pub fn diagnostic(&self, diagnostic: ToolDiagnostic) -> Result<()> {
        self.assert_live()?;
        self.0.reported.lock().diagnostics.push(diagnostic);
        self.0.progress.mark();
        Ok(())
    }

    /// Replace running details; the last value becomes the result details when the result omits `details`.
    pub async fn details(&self, value: JsonValue, context: &Context) -> Result<()> {
        self.assert_live()?;
        if let Some(signal) = context.abort_signal() {
            signal.throw_if_aborted()?;
        }
        self.0.reported.lock().details = Some(value);
        // Cancelling the wait leaves the update in place; the commit's own outcome stays observed.
        let committed = self.0.progress.mark_and_wait();
        await_with_context(committed, context).await?
    }

    /// Commit on the Session line through the tool task; returns what `change` returns.
    pub async fn commit<T, F, Fut>(&self, change: F, context: &Context) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(Transaction) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        let slot: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));
        let out = slot.clone();
        self.0
            .runtime
            .commit(
                move |tx, _| async move {
                    let value = change(tx).await?;
                    *out.lock() = Some(value);
                    Ok(None)
                },
                context,
            )
            .await?;
        let value = slot.lock().take();
        value.ok_or_else(|| Error::message("Tool commit produced no value"))
    }

    pub async fn memo<T: DeserializeOwned>(&self, name: &str) -> Result<Option<T>> {
        self.0.runtime.memo(name).await
    }

    pub async fn memo_with<T: Serialize + DeserializeOwned>(
        &self,
        name: &str,
        candidate: T,
        context: &Context,
    ) -> Result<T> {
        self.0.runtime.memo_with(name, candidate, context).await
    }

    /// Create a task in a commit of its own; `options.conversation_id` is ignored (TS omits it).
    pub async fn create_task<I, S, R, H>(
        &self,
        task: &Task<I, S, R, H>,
        input: I,
        options: TaskOptions,
        context: &Context,
    ) -> Result<TaskId<R>>
    where
        I: Serialize + Send + 'static,
        S: Serialize + 'static,
        R: Send + 'static,
        H: Send + Sync + 'static,
    {
        let task = task.clone();
        let options = TaskOptions {
            conversation_id: None,
            ..options
        };
        self.commit(
            move |tx| async move { tx.create_task(&task, input, options).await },
            context,
        )
        .await
    }

    pub async fn get_task<T>(
        &self,
        id: TaskId<T>,
        context: &Context,
    ) -> Result<Option<TaskRecord>> {
        self.0.runtime.get_task(id, context).await
    }

    pub async fn wait_for_task<T>(&self, id: TaskId<T>, context: &Context) -> Result<TaskRecord> {
        self.0.runtime.wait_for_task(id, context).await
    }

    /// Invocation-bound handle of an existing conversation, such as one this tool created in `commit()`.
    pub async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<ConversationHandle>> {
        self.0.runtime.conversation(id, context).await
    }

    pub async fn snapshot<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> Result<Option<A::Value>> {
        self.0.runtime.snapshot(token, address, context).await
    }

    pub async fn snapshot_as_of<A: RewindableDocAccess>(
        &self,
        token: &A,
        address: A::Address,
        at: EntryId,
        context: &Context,
    ) -> Result<Option<A::Value>> {
        self.0
            .runtime
            .snapshot_as_of(token, address, at, context)
            .await
    }

    pub async fn watch_doc<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> Result<Option<DocumentWatch>> {
        self.0.runtime.watch_doc(token, address, context).await
    }
}

/// Execute with the resolved implementation, then settle its result.
async fn run(
    runtime: &Runtime,
    call: &ToolCall,
    tool: &ToolRegistration,
    args: JsonValue,
    context: &Context,
) -> Result<()> {
    let limits = OutputLimits {
        max_bytes: tool
            .output_limits
            .and_then(|limits| limits.max_bytes)
            .unwrap_or(DEFAULT_MAX_BYTES),
        max_lines: tool
            .output_limits
            .and_then(|limits| limits.max_lines)
            .unwrap_or(DEFAULT_MAX_LINES),
        retain: tool
            .output_limits
            .and_then(|limits| limits.retain)
            .unwrap_or(Retain::Head),
    };
    let reported = Arc::new(Mutex::new(Reported {
        output: OutputBuffer::new(limits),
        diagnostics: Vec::new(),
        details: None,
    }));
    let progress = publish_progress(runtime, reported.clone(), context);
    let ended = Arc::new(AtomicBool::new(false));

    let mut ending = Ending::Completed;
    // Built for this call, so a rerun after recovery gets the conversation's environment at that time.
    let executed = match runtime.env(context).await {
        Err(error) => Err(error),
        Ok(env) => {
            let api = ToolExecutionApi(Arc::new(ApiInner {
                runtime: runtime.clone(),
                call_id: call.id.clone(),
                env,
                reported: reported.clone(),
                progress: progress.clone(),
                ended: ended.clone(),
            }));
            (tool.execute)(args, api, context.clone()).await
        }
    };
    let result = match executed {
        Ok(result) => result,
        Err(error) => {
            if runtime.signal().aborted() {
                ended.store(true, Ordering::SeqCst);
                for waiter in progress.stop().await {
                    let _ = waiter.send(Err(error.clone()));
                }
                return Err(error);
            }
            // A failure, from `execute()` or from building the environment, ends the task `failed`, which cancels
            // what the call owned; it no longer supervises it. The error text is already in the result entry.
            ending = Ending::Failed(format!("Tool {} threw", call.name));
            ToolExecutionResult {
                is_error: Some(true),
                diagnostics: Some(vec![tool_diagnostic("tool_error", &error_text(&error))]),
                ..ToolExecutionResult::default()
            }
        }
    };
    ended.store(true, Ordering::SeqCst);
    reported.lock().output.end();
    // Details still waiting for a progress commit settle with the terminal commit, the final flush.
    let pending = progress.stop().await;
    let settled = async {
        let settled = final_result(runtime, call, result, &reported, limits, context).await?;
        settle(runtime, call, ending, move |_| settled, context).await
    }
    .await;
    if let Err(error) = settled {
        for waiter in pending {
            let _ = waiter.send(Err(error.clone()));
        }
        return Err(error);
    }
    for waiter in pending {
        let _ = waiter.send(Ok(()));
    }
    Ok(())
}

/// UTF-8 byte offset of the first `units` UTF-16 code units of `text`.
fn utf16_offset(text: &str, units: usize) -> usize {
    let mut seen = 0;
    for (index, c) in text.char_indices() {
        if seen >= units {
            return index;
        }
        seen += c.len_utf16();
    }
    text.len()
}

#[derive(Default, Clone)]
struct Written {
    text: String,
    details: Option<JsonValue>,
    diagnostics: usize,
}

/// Throttled commits of what the tool reported into its `pi.live.tools` slot, each writing only what changed since
/// the last one.
fn publish_progress(
    runtime: &Runtime,
    reported: Arc<Mutex<Reported>>,
    context: &Context,
) -> Progress {
    let written = Arc::new(Mutex::new(Written::default()));
    let write_runtime = runtime.clone();
    let context = context.clone();
    let report_runtime = runtime.clone();
    Progress::new(
        Arc::new(move || {
            // Capture everything synchronously: the tool keeps reporting while the commit is in flight.
            let (snapshot, details, diagnostics) = {
                let mut reported = reported.lock();
                let snapshot = reported.output.snapshot();
                (
                    snapshot,
                    reported.details.clone(),
                    reported.diagnostics.clone(),
                )
            };
            let previous = written.lock().clone();
            let added: Vec<ToolDiagnostic> = diagnostics
                .get(previous.diagnostics..)
                .unwrap_or_default()
                .to_vec();
            let details_changed = details != previous.details;
            // What the commit writes, as the diff stores the string: an append, a trim plus an append of what follows
            // the shared part, or the whole window when its bounded overlap search finds nothing.
            let mut bytes = 0;
            if snapshot.text != previous.text {
                let shared = if snapshot.text.starts_with(&previous.text) {
                    previous.text.len()
                } else {
                    utf16_offset(
                        &snapshot.text,
                        overlap(&previous.text, &snapshot.text, 65_536),
                    )
                };
                bytes += snapshot.text.len() - shared;
            }
            if details_changed {
                bytes += serde_json::to_string(details.as_ref().unwrap_or(&JsonValue::Null))
                    .map_or(0, |text| text.len());
            }
            if !added.is_empty() {
                bytes += serde_json::to_string(&added).map_or(0, |text| text.len());
            }
            let current = Written {
                text: snapshot.text.clone(),
                details: details.clone(),
                diagnostics: diagnostics.len(),
            };
            let runtime = write_runtime.clone();
            let context = context.clone();
            let written = written.clone();
            Box::pin(async move {
                let conversation_id = runtime.conversation_id();
                let task_id = runtime.task_id().erase();
                runtime
                    .commit(
                        move |tx, _| async move {
                            let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                            live.edit(|live| {
                                let Some(slot) = tool_slot(live, task_id) else {
                                    return;
                                };
                                if slot.output.as_deref().unwrap_or("") != snapshot.text {
                                    slot.output = Some(snapshot.text);
                                }
                                if snapshot.dropped_bytes > 0 {
                                    slot.dropped_bytes = Some(snapshot.dropped_bytes as u64);
                                }
                                if snapshot.dropped_lines > 0 {
                                    slot.dropped_lines = Some(snapshot.dropped_lines as u64);
                                }
                                if details_changed && details.is_some() {
                                    slot.details = details;
                                }
                                if !added.is_empty() {
                                    slot.diagnostics.get_or_insert_with(Vec::new).extend(added);
                                }
                            })?;
                            Ok(None)
                        },
                        &context,
                    )
                    .await?;
                *written.lock() = current;
                Ok(bytes)
            })
        }),
        Arc::new(move |error| {
            // Rejections after an abort mark or close are expected; the committed state stays consistent.
            if !report_runtime.signal().aborted() {
                let _ = report_runtime.report(error);
            }
        }),
    )
}

/// The settled result: the tool's result with the retained output and last details as fallbacks, its diagnostics
/// after those reported through the api, `afterTool` applied, and explicit text bounded, with the Harness's truncation
/// diagnostic last.
async fn final_result(
    runtime: &Runtime,
    call: &ToolCall,
    result: ToolExecutionResult,
    reported: &Arc<Mutex<Reported>>,
    limits: OutputLimits,
    context: &Context,
) -> Result<ToolExecutionResult> {
    let mut harness = Vec::new();
    let (retained, reported_details, reported_diagnostics) = {
        let mut reported = reported.lock();
        let retained = result.content.is_none().then(|| reported.output.snapshot());
        (
            retained,
            reported.details.clone(),
            reported.diagnostics.clone(),
        )
    };
    let content: Content = match &retained {
        None => result.content.clone().unwrap_or_default(),
        Some(retained) if retained.text.is_empty() => Vec::new(),
        Some(retained) => vec![UserContent::text(retained.text.clone())],
    };
    let mut diagnostics = reported_diagnostics;
    diagnostics.extend(result.diagnostics.clone().unwrap_or_default());
    let initial = ToolExecutionResult {
        content: Some(content.clone()),
        details: result.details.clone().or(reported_details),
        diagnostics: Some(diagnostics),
        ..result
    };
    let final_ = Arc::new(Mutex::new(initial));
    runtime
        .hooks()
        .each(
            |hooks| hooks.after_tool.clone(),
            |hook| {
                let final_ = final_.clone();
                let call = call.clone();
                let api = runtime.hook_api();
                let context = context.clone();
                async move {
                    let current = final_.lock().clone();
                    if let Some(next) = hook(call, current, api, context).await? {
                        *final_.lock() = next;
                    }
                    Ok(())
                }
            },
        )
        .await?;
    let final_ = final_.lock().clone();
    // The retained output's truncation applies only while afterTool kept that content (TS compares the array by
    // identity; here by value).
    if let Some(retained) = &retained
        && retained.dropped_bytes > 0
        && final_.content.as_ref() == Some(&content)
    {
        harness.push(truncated(
            retained.dropped_lines,
            retained.dropped_bytes,
            Some(limits.retain),
        ));
    }
    let bounded = bound_content(final_.content.clone().unwrap_or_default(), limits);
    if bounded.dropped_bytes > 0 {
        harness.push(truncated(
            bounded.dropped_lines,
            bounded.dropped_bytes,
            Some(limits.retain),
        ));
    }
    let mut diagnostics = final_.diagnostics.clone().unwrap_or_default();
    diagnostics.extend(harness);
    Ok(ToolExecutionResult {
        content: Some(bounded.content),
        diagnostics: Some(diagnostics),
        ..final_
    })
}

/// How a tool task ends; the result entry is appended either way. `Failed` (execution failed or was interrupted)
/// records cancellation intent for the conversations the call owns; a result with `is_error` still completes.
#[derive(Debug, Clone)]
enum Ending {
    Completed,
    Aborted,
    Failed(String),
}

/// Commit the tool's terminal state: append its result entry, mark its slot done, and complete or end aborted with
/// the entry ID. `build` receives the slot so interruption and abort can report the durable partial output.
async fn settle(
    runtime: &Runtime,
    call: &ToolCall,
    ending: Ending,
    build: impl FnOnce(Option<&ToolSlot>) -> ToolExecutionResult + Send + 'static,
    context: &Context,
) -> Result<()> {
    let conversation_id = runtime.conversation_id();
    let task_id = runtime.task_id().erase();
    let call = call.clone();
    let now = runtime.now()?;
    runtime
        .commit(
            move |tx, _| async move {
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
                let mut state = live.get()?;
                let result = build(tool_slot(&mut state, task_id).map(|slot| &*slot));
                let control = result.control.clone();
                let entry = append_tool_result(&tx, conversation_id, &call, result, now).await?;
                if let Some(slot) = tool_slot(&mut state, task_id) {
                    finish_slot(slot, Some(entry.id));
                    live.set(state)?;
                }
                let entry_id = entry.id;
                let outcome = match ending {
                    Ending::Aborted => TaskOutcome::Aborted {
                        reason: None,
                        result: Some(serde_json::to_value(ToolTaskResult {
                            entry_id,
                            control: None,
                        })?),
                    },
                    Ending::Failed(message) => TaskOutcome::Failed {
                        error: TaskOutcomeError {
                            message,
                            detail: None,
                        },
                        result: Some(serde_json::to_value(ToolTaskResult {
                            entry_id,
                            control: None,
                        })?),
                    },
                    Ending::Completed => TaskOutcome::Completed {
                        result: serde_json::to_value(ToolTaskResult { entry_id, control })?,
                    },
                };
                Ok(Some(Next::Terminal { outcome }))
            },
            context,
        )
        .await
}

/// An error result from the slot's durable partial output, details, and diagnostics.
fn from_slot(slot: Option<&ToolSlot>, code: &str, message: &str) -> ToolExecutionResult {
    let mut diagnostics = slot
        .and_then(|slot| slot.diagnostics.clone())
        .unwrap_or_default();
    let dropped_bytes = slot.and_then(|slot| slot.dropped_bytes).unwrap_or(0);
    if dropped_bytes > 0 {
        let dropped_lines = slot.and_then(|slot| slot.dropped_lines).unwrap_or(0);
        diagnostics.push(truncated(
            dropped_lines as usize,
            dropped_bytes as usize,
            None,
        ));
    }
    diagnostics.push(tool_diagnostic(code, message));
    let output = slot
        .and_then(|slot| slot.output.clone())
        .unwrap_or_default();
    ToolExecutionResult {
        content: Some(if output.is_empty() {
            Vec::new()
        } else {
            vec![UserContent::text(output)]
        }),
        is_error: Some(true),
        details: slot.and_then(|slot| slot.details.clone()),
        diagnostics: Some(diagnostics),
        ..ToolExecutionResult::default()
    }
}

/// An error result the Harness writes itself: no content and one `error` diagnostic with `code`.
pub fn harness_error(code: &str, message: &str) -> ToolExecutionResult {
    ToolExecutionResult {
        content: Some(Vec::new()),
        is_error: Some(true),
        diagnostics: Some(vec![tool_diagnostic(code, message)]),
        ..ToolExecutionResult::default()
    }
}

fn tool_diagnostic(code: &str, message: &str) -> ToolDiagnostic {
    ToolDiagnostic {
        severity: DiagnosticSeverity::Error,
        message: message.to_string(),
        code: Some(code.to_string()),
    }
}

/// The Harness's truncation diagnostic; `retain` is unknown when rebuilt from a slot after recovery.
fn truncated(dropped_lines: usize, dropped_bytes: usize, retain: Option<Retain>) -> ToolDiagnostic {
    let kept = match retain {
        None => "",
        Some(Retain::Head) => " to its beginning",
        Some(Retain::Tail) => " to its end",
    };
    ToolDiagnostic {
        severity: DiagnosticSeverity::Warn,
        message: format!(
            "Output truncated{kept}: {dropped_lines} lines, {dropped_bytes} bytes dropped"
        ),
        code: Some("truncated".to_string()),
    }
}

/// `{ diagnostics }` of a `pi.tool-result` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultData {
    pub diagnostics: Vec<ToolDiagnostic>,
}

/// Append a `pi.tool-result` entry. The content ends with the rendered diagnostics, so the stored message is exactly
/// what the model sees; `data` keeps the structured list. A result's usage is added to `pi.usage` in the same commit.
pub async fn append_tool_result(
    tx: &Transaction,
    conversation_id: ConversationId,
    call: &ToolCall,
    result: ToolExecutionResult,
    timestamp: u64,
) -> Result<EntryRecord> {
    let diagnostics = result.diagnostics.unwrap_or_default();
    let mut content = result.content.unwrap_or_default();
    if !diagnostics.is_empty() {
        content.push(UserContent::text(render_diagnostics(&diagnostics)));
    }
    let message = ToolResultMessage {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content,
        details: result.details,
        usage: result.usage.clone(),
        nested_calls: None,
        is_error: result.is_error.unwrap_or(false),
        timestamp,
    };
    if let Some(usage) = &result.usage {
        record_usage(tx, conversation_id, UsageBucket::Tools, &call.name, usage).await?;
    }
    tx.append_entry_of(
        &TOOL_RESULT_ENTRY,
        conversation_id,
        TypedEntryDraft {
            model: Some(vec![Message::ToolResult(message)]),
            data: Some(serde_json::to_value(ToolResultData { diagnostics })?),
            head: None,
            edits: None,
        },
    )
    .await
}

fn render_diagnostics(diagnostics: &[ToolDiagnostic]) -> String {
    let lines: Vec<String> = diagnostics
        .iter()
        .map(|diagnostic| format!("[{}] {}", diagnostic.severity.as_str(), diagnostic.message))
        .collect();
    format!("<harness>\n{}\n</harness>", lines.join("\n"))
}

struct BoundedContent {
    content: Content,
    dropped_bytes: usize,
    dropped_lines: usize,
}

/// Bound the text of result content. When the joined text exceeds the limits, the text items are replaced by one
/// bounded item at the position of the first (head) or last (tail) text item; other content is kept.
fn bound_content(content: Content, limits: OutputLimits) -> BoundedContent {
    let texts: Vec<usize> = content
        .iter()
        .enumerate()
        .filter(|(_, item)| matches!(item, UserContent::Text(_)))
        .map(|(index, _)| index)
        .collect();
    let joined: String = content
        .iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    let bounded = bound_output(&joined, limits);
    if bounded.dropped_bytes == 0 {
        return BoundedContent {
            content,
            dropped_bytes: 0,
            dropped_lines: 0,
        };
    }
    let keep = match limits.retain {
        Retain::Head => texts.first().copied(),
        Retain::Tail => texts.last().copied(),
    };
    let mut result = Vec::new();
    for (index, item) in content.into_iter().enumerate() {
        match item {
            UserContent::Text(text) => {
                if Some(index) == keep {
                    result.push(UserContent::Text(TextContent {
                        text: bounded.text.clone(),
                        ..text
                    }));
                }
            }
            other => result.push(other),
        }
    }
    BoundedContent {
        content: result,
        dropped_bytes: bounded.dropped_bytes,
        dropped_lines: bounded.dropped_lines,
    }
}

fn error_text(error: &Error) -> String {
    error.to_string()
}
