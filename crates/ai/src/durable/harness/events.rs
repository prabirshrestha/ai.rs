//! Port of durable `src/harness/events.ts`: experimental agent events (spec §9.4).

use std::collections::HashSet;
use std::sync::Arc;

use futures::future::{BoxFuture, Shared};
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::Serialize;

use crate::chord::delta::{Op, Path, Seg};
use crate::chord::{Context, JsonValue};
use crate::durable::errors::Result;
use crate::durable::ids::{ConversationId, EntryId, SubmissionId, TaskId};
use crate::durable::session::observation::{CommittedWatch, ObservedValue};
use crate::durable::types::{
    CommitChange, CommitPublication, EntryRecord, JsonObject, SubmissionRecord, TaskOutcome,
    TaskQuery, TaskRecord, TaskState, TaskStatus, WatchEnd,
};
use crate::types::{AssistantContent, AssistantMessage, Message, Usage};

use super::harness::Harness;
use super::inbox::{InboxItem, InboxState};
use super::live::{CompactionStatus, GenerationStatus, LiveState, SlotStatus, ToolSlot};
use super::types::{AgentState, CompactionReason, ToolDiagnostic};
use super::usage::UsageState;
use super::util::{abort_error, scan_all};
use super::view::{ConversationView, ViewObserver};

/// A queued submission's ID and mode.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueuedItem {
    pub id: SubmissionId,
    /// `steer`, `followUp`, or `write`.
    pub mode: &'static str,
}

/// One change to the in-flight assistant message, relative to that message.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
#[allow(clippy::large_enum_variant)]
pub enum MessageChange {
    TextStart {
        content_index: usize,
        block: AssistantContent,
    },
    ThinkingStart {
        content_index: usize,
        block: AssistantContent,
    },
    ToolcallStart {
        content_index: usize,
        block: AssistantContent,
    },
    TextDelta {
        content_index: usize,
        delta: String,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
    },
    ToolcallDelta {
        content_index: usize,
        path: Path,
        delta: String,
    },
    Block {
        content_index: usize,
        block: AssistantContent,
    },
    Message {
        message: AssistantMessage,
    },
}

/// `{ inputs }` of the current run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunInputs {
    pub inputs: Vec<SubmissionId>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotEvent {
    pub entries: Vec<EntryRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<RunInputs>,
    /// Current generation attempt: its in-flight partial, retry backoff, or deferred poll.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationStatus>,
    pub tools: Vec<ToolSlot>,
    /// `pi.live.compactions`: live compactions with their attempt and retry backoff.
    pub compactions: Vec<CompactionStatus>,
    pub inbox: Vec<QueuedItem>,
    /// `pi.agent`; the initial value when absent.
    pub agent: AgentState,
    pub usage: UsageState,
}

/// A tool's output change: a front trim and then an append of the retained window, or its replacement.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ToolOutputUpdate {
    Delta {
        #[serde(rename = "trimStart", skip_serializing_if = "Option::is_none")]
        trim_start: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        append: Option<String>,
    },
    Set {
        set: String,
    },
}

/// Experimental agent event, shaped like the coding agent's session events (spec §9.4).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
#[allow(clippy::large_enum_variant)]
pub enum AgentEvent {
    Snapshot(SnapshotEvent),
    RunStart {
        inputs: Vec<SubmissionId>,
    },
    RunEnd {
        inputs: Vec<SubmissionId>,
    },
    TurnStart,
    TurnEnd,
    MessageStart {
        message: Message,
    },
    /// `usage` is the partial's current usage, as in the coding agent's JSON mode.
    MessageUpdate {
        usage: Usage,
        changes: Vec<MessageChange>,
    },
    MessageEnd {
        entry: EntryRecord,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: JsonObject,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<ToolOutputUpdate>,
        /// `Some(null)` when a safe replay removed the details.
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
        #[serde(skip_serializing_if = "Option::is_none")]
        diagnostics: Option<Vec<ToolDiagnostic>>,
    },
    /// `entry` is absent when the tool task faulted or was orphaned.
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        entry: Option<EntryRecord>,
    },
    InboxUpdate {
        items: Vec<QueuedItem>,
    },
    Submission {
        record: SubmissionRecord,
    },
    AutoRetryStart {
        attempt: u32,
        at: u64,
        error_message: String,
    },
    AutoRetryEnd {
        attempt: u32,
    },
    DeferredPoll {
        poll_at: u64,
    },
    EntryAppended {
        entry: EntryRecord,
    },
    AgentChanged {
        agent: AgentState,
    },
    UsageChanged {
        usage: UsageState,
    },
    TaskFailed {
        task_id: TaskId,
        kind: String,
        message: String,
    },
    CompactionStart {
        task_id: TaskId,
        reason: CompactionReason,
        blocking: bool,
    },
    /// The task's receipt tells whether it produced a summary; the summary entry has its own events.
    CompactionEnd {
        task_id: TaskId,
        reason: CompactionReason,
    },
}

impl AgentEvent {
    /// The TS `type` discriminator.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Snapshot(_) => "snapshot",
            Self::RunStart { .. } => "run_start",
            Self::RunEnd { .. } => "run_end",
            Self::TurnStart => "turn_start",
            Self::TurnEnd => "turn_end",
            Self::MessageStart { .. } => "message_start",
            Self::MessageUpdate { .. } => "message_update",
            Self::MessageEnd { .. } => "message_end",
            Self::ToolExecutionStart { .. } => "tool_execution_start",
            Self::ToolExecutionUpdate { .. } => "tool_execution_update",
            Self::ToolExecutionEnd { .. } => "tool_execution_end",
            Self::InboxUpdate { .. } => "inbox_update",
            Self::Submission { .. } => "submission",
            Self::AutoRetryStart { .. } => "auto_retry_start",
            Self::AutoRetryEnd { .. } => "auto_retry_end",
            Self::DeferredPoll { .. } => "deferred_poll",
            Self::EntryAppended { .. } => "entry_appended",
            Self::AgentChanged { .. } => "agent_changed",
            Self::UsageChanged { .. } => "usage_changed",
            Self::TaskFailed { .. } => "task_failed",
            Self::CompactionStart { .. } => "compaction_start",
            Self::CompactionEnd { .. } => "compaction_end",
        }
    }
}

/// One commit's events: the values of the stream's watch.
pub type EventBatch = Arc<[AgentEvent]>;

impl ObservedValue for EventBatch {
    fn is_retired(&self) -> bool {
        false
    }

    fn root_value(&self) -> JsonValue {
        serde_json::to_value(&**self).expect("agent events serialize")
    }
}

/// Serialized stream of one conversation's event batches, one per commit.
pub struct AgentEventStream {
    /// The `snapshot` event at attachment. Divergence: Pi's is an `AgentEvent` with `type: "snapshot"`; this is the
    /// variant's payload, which serializes without the tag (wrap it in `AgentEvent::Snapshot` for Pi's JSON).
    pub snapshot: SnapshotEvent,
    watch: CommittedWatch<EventBatch>,
}

impl std::fmt::Debug for AgentEventStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentEventStream").finish_non_exhaustive()
    }
}

impl AgentEventStream {
    /// Install the sole listener; never invoked inline.
    pub fn start(
        &self,
        listener: impl Fn(EventBatch, Context) -> BoxFuture<'static, Result<()>> + Send + Sync + 'static,
    ) -> Result<()> {
        self.watch
            .start(move |events, _ops, context| listener(events, context))
    }

    pub fn stop(&self) -> Shared<BoxFuture<'static, WatchEnd>> {
        self.watch.stop()
    }

    pub fn closed(&self) -> Shared<BoxFuture<'static, WatchEnd>> {
        self.watch.closed()
    }
}

/// The typed parts of a view the events read.
struct Parts {
    live: LiveState,
    inbox: Option<Arc<JsonValue>>,
    agent: Option<Arc<JsonValue>>,
    usage: Option<Arc<JsonValue>>,
}

fn parts(view: &ConversationView) -> Parts {
    Parts {
        live: view
            .doc("pi.live")
            .map(|value| decode(value))
            .unwrap_or_default(),
        inbox: view.doc("pi.inbox").cloned(),
        agent: view.doc("pi.agent").cloned(),
        usage: view.doc("pi.usage").cloned(),
    }
}

fn decode<T: serde::de::DeserializeOwned + Default>(value: &JsonValue) -> T {
    serde_json::from_value(value.clone()).unwrap_or_default()
}

fn decode_doc<T: serde::de::DeserializeOwned + Default>(value: &Option<Arc<JsonValue>>) -> T {
    value.as_deref().map(decode).unwrap_or_default()
}

/// Same committed revision of an optional document (TS identity).
fn same_doc(a: &Option<Arc<JsonValue>>, b: &Option<Arc<JsonValue>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

fn snapshot_of(view: &ConversationView) -> SnapshotEvent {
    let Parts {
        live,
        inbox,
        agent,
        usage,
    } = parts(view);
    SnapshotEvent {
        entries: view.entries.to_vec(),
        run: live.run.map(|run| RunInputs { inputs: run.inputs }),
        generation: live.generation,
        tools: live.tools.unwrap_or_default(),
        compactions: live.compactions.unwrap_or_default(),
        inbox: queued(&inbox),
        agent: decode_doc(&agent),
        usage: decode_doc(&usage),
    }
}

fn queued(inbox: &Option<Arc<JsonValue>>) -> Vec<QueuedItem> {
    decode_doc::<InboxState>(inbox)
        .items
        .iter()
        .map(|item| QueuedItem {
            id: item.id(),
            mode: match item {
                InboxItem::Steer { .. } => "steer",
                InboxItem::FollowUp { .. } => "followUp",
                InboxItem::Write { .. } => "write",
            },
        })
        .collect()
}

#[derive(Clone)]
struct EventsObserver {
    conversation_id: ConversationId,
    current: Arc<Mutex<ConversationView>>,
    held: Arc<Mutex<HashSet<TaskId>>>,
    watch: CommittedWatch<EventBatch>,
}

impl ViewObserver for EventsObserver {
    fn publication(
        &self,
        before: &ConversationView,
        after: &ConversationView,
        ops: &Arc<[Op]>,
        publication: &CommitPublication,
        context: &Context,
    ) {
        *self.current.lock() = after.clone();
        let events = translate(
            self.conversation_id,
            before,
            after,
            ops,
            publication,
            &mut self.held.lock(),
        );
        if !events.is_empty() {
            self.watch
                .advance(events.into(), Arc::from(Vec::new()), context.clone());
        }
    }

    fn close_session(&self) {
        self.watch.close_session();
    }
}

/// Experimental: attach to one conversation's agent events (spec §9.4). The snapshot and the registration for later
/// commits are captured atomically on the Session line; overflow replaces undelivered batches with one snapshot.
pub async fn watch_events(
    harness: &Harness,
    conversation_id: ConversationId,
    context: &Context,
) -> Result<AgentEventStream> {
    let scan_context = context.clone();
    let (observer, _) = harness
        .views()
        .attach(
            conversation_id,
            move |initial, release, storage| {
                Box::pin(async move {
                    // Generations whose held outcome already ended their turn, read on the line with the snapshot.
                    let query = TaskQuery {
                        conversation_id: Some(conversation_id),
                        kind: Some("pi.generation".into()),
                        status: Some(TaskStatus::Completing),
                        ..TaskQuery::default()
                    };
                    let completing = scan_all(|cursor| {
                        let storage = storage.clone();
                        let query = query.clone();
                        let context = scan_context.clone();
                        async move {
                            storage
                                .scan_tasks(&query, 100, cursor.as_ref(), &context)
                                .await
                        }
                    })
                    .await?;
                    let held: HashSet<TaskId> = completing.iter().map(|record| record.id).collect();
                    let current = Arc::new(Mutex::new(initial));
                    let replace_current = current.clone();
                    // Batches are the watch's values; an overflow delivers a snapshot of the newest view instead.
                    let watch = CommittedWatch::new(
                        EventBatch::from(Vec::new()),
                        release,
                        Some(Arc::new(move || {
                            EventBatch::from(vec![AgentEvent::Snapshot(snapshot_of(
                                &replace_current.lock(),
                            ))])
                        })),
                    );
                    Ok(EventsObserver {
                        conversation_id,
                        current,
                        held: Arc::new(Mutex::new(held)),
                        watch,
                    })
                })
            },
            context,
        )
        .await?;
    let snapshot = snapshot_of(&observer.current.lock());
    let watch = observer.watch;
    // Like a watch, the acquisition context governs the stream's lifetime.
    if let Some(signal) = context.abort_signal() {
        if let Err(reason) = signal.throw_if_aborted() {
            watch.cancel();
            return Err(abort_error(reason));
        }
        watch.observe_cancellation(signal)?;
    }
    Ok(AgentEventStream { snapshot, watch })
}

impl Harness {
    /// Experimental: attach to one conversation's agent events (spec §9.4).
    pub async fn watch_events(
        &self,
        conversation_id: ConversationId,
        context: &Context,
    ) -> Result<AgentEventStream> {
        watch_events(self, conversation_id, context).await
    }
}

/// The tool result for `call_id` among `entries`.
fn result_of<'a>(entries: &[&'a EntryRecord], call_id: &str) -> Option<&'a EntryRecord> {
    entries.iter().copied().find(|entry| {
        matches!(
            entry.model.as_ref().and_then(|model| model.first()),
            Some(Message::ToolResult(message)) if message.tool_call_id == call_id
        )
    })
}

fn first_message(entry: &EntryRecord) -> Option<&Message> {
    entry.model.as_ref().and_then(|model| model.first())
}

/// Every event one publication causes, in the order of spec §9.4.
fn translate(
    conversation_id: ConversationId,
    before: &ConversationView,
    after: &ConversationView,
    view_ops: &[Op],
    publication: &CommitPublication,
    held: &mut HashSet<TaskId>,
) -> Vec<AgentEvent> {
    let mut entries: Vec<&EntryRecord> = Vec::new();
    let mut tasks: IndexMap<TaskId, &TaskRecord> = IndexMap::new();
    let mut submissions: Vec<&SubmissionRecord> = Vec::new();
    for change in &publication.changes {
        match change {
            CommitChange::Entry(entry) if entry.conversation_id == conversation_id => {
                entries.push(entry)
            }
            CommitChange::Task(task) if task.conversation_id == conversation_id => {
                tasks.insert(task.id, task);
            }
            CommitChange::Submission(record) if record.conversation_id == conversation_id => {
                submissions.push(record)
            }
            _ => {}
        }
    }
    if view_ops.is_empty() && entries.is_empty() && tasks.is_empty() && submissions.is_empty() {
        return Vec::new();
    }
    // Entries are appended in ID order; submission records are published in the order the commit first touched them.
    submissions.sort_by_key(|record| record.id);
    let was = parts(before);
    let now = parts(after);
    let mut events = Vec::new();

    // Progress: tool starts, the in-flight message, tool updates, retry and deferred state.
    let no_slots = Vec::new();
    let slots_before: IndexMap<&str, &ToolSlot> = was
        .live
        .tools
        .as_ref()
        .unwrap_or(&no_slots)
        .iter()
        .map(|slot| (slot.call_id.as_str(), slot))
        .collect();
    let slots = now.live.tools.as_ref().unwrap_or(&no_slots);
    for slot in slots {
        let running_before = slots_before
            .get(slot.call_id.as_str())
            .is_some_and(|previous| previous.status == SlotStatus::Running);
        if slot.status != SlotStatus::Running || running_before {
            continue;
        }
        let checkpoint = slot
            .task_id
            .and_then(|id| tasks.get(&id))
            .and_then(|task| task.state.checkpoint());
        let args = checkpoint
            .and_then(|checkpoint| checkpoint.get("arguments"))
            .and_then(JsonValue::as_object)
            .cloned()
            .unwrap_or_default();
        events.push(AgentEvent::ToolExecutionStart {
            tool_call_id: slot.call_id.clone(),
            tool_name: slot.name.clone(),
            args,
        });
    }
    let partial_before = was
        .live
        .generation
        .as_ref()
        .and_then(|generation| generation.message.as_ref());
    let partial = now
        .live
        .generation
        .as_ref()
        .and_then(|generation| generation.message.as_ref());
    if let Some(partial) = partial {
        match partial_before {
            None => events.push(AgentEvent::MessageStart {
                message: Message::Assistant(partial.clone()),
            }),
            Some(previous) if previous != partial => events.push(AgentEvent::MessageUpdate {
                usage: partial.usage.clone(),
                changes: message_changes(view_ops, partial),
            }),
            Some(_) => {}
        }
    }
    for (index, slot) in slots.iter().enumerate() {
        let Some(previous) = slots_before.get(slot.call_id.as_str()) else {
            continue;
        };
        if slot.status != SlotStatus::Running || previous.status != SlotStatus::Running {
            continue;
        }
        if let Some((output, details, diagnostics)) = tool_update(view_ops, index, slot, previous) {
            events.push(AgentEvent::ToolExecutionUpdate {
                tool_call_id: slot.call_id.clone(),
                tool_name: slot.name.clone(),
                output,
                details,
                diagnostics,
            });
        }
    }
    let generation = now.live.generation.as_ref();
    let generation_before = was.live.generation.as_ref();
    let retry = generation.and_then(|generation| generation.retry.as_ref());
    let retry_before = generation_before.and_then(|generation| generation.retry.as_ref());
    if let (Some(generation), Some(retry), None) = (generation, retry, retry_before) {
        events.push(AgentEvent::AutoRetryStart {
            attempt: generation.attempt,
            at: retry.at,
            error_message: retry.error.clone(),
        });
    }
    if let (Some(generation_before), Some(_), None) = (generation_before, retry_before, retry) {
        events.push(AgentEvent::AutoRetryEnd {
            attempt: generation_before.attempt,
        });
    }
    if let Some(deferred) = generation.and_then(|generation| generation.deferred.as_ref()) {
        let poll_before = generation_before
            .and_then(|generation| generation.deferred.as_ref())
            .map(|deferred| deferred.poll_at);
        if Some(deferred.poll_at) != poll_before {
            events.push(AgentEvent::DeferredPoll {
                poll_at: deferred.poll_at,
            });
        }
    }

    // Tools that end in this commit: a slot that becomes done, one created done (a call not offered), or an unfinished
    // one that vanishes because its run ended. A done slot that vanishes ended earlier.
    // Each end holds the index of its result among this commit's entries.
    let mut tool_ends: Vec<(String, String, Option<usize>)> = Vec::new();
    let position = |entry_id: Option<EntryId>| {
        entries
            .iter()
            .position(|candidate| Some(candidate.id) == entry_id)
    };
    for previous in slots_before.values() {
        if previous.status == SlotStatus::Done {
            continue;
        }
        match slots
            .iter()
            .find(|candidate| candidate.call_id == previous.call_id)
        {
            Some(slot) if slot.status == SlotStatus::Done => tool_ends.push((
                previous.call_id.clone(),
                previous.name.clone(),
                position(slot.entry),
            )),
            // A slot whose run ended in this commit may have had its result appended with it, as for unstarted
            // calls.
            None => tool_ends.push((
                previous.call_id.clone(),
                previous.name.clone(),
                position(result_of(&entries, &previous.call_id).map(|entry| entry.id)),
            )),
            Some(_) => {}
        }
    }
    for slot in slots {
        if slot.status == SlotStatus::Done && !slots_before.contains_key(slot.call_id.as_str()) {
            tool_ends.push((
                slot.call_id.clone(),
                slot.name.clone(),
                position(slot.entry),
            ));
        }
    }
    let end_event =
        |(call_id, name, entry): &(String, String, Option<usize>)| AgentEvent::ToolExecutionEnd {
            tool_call_id: call_id.clone(),
            tool_name: name.clone(),
            entry: entry.map(|index| entries[index].clone()),
        };

    // Entries in append order; a tool's end directly precedes its result's message, as in the coding agent.
    let mut assistant_appended = false;
    for (index, entry) in entries.iter().enumerate() {
        events.extend(
            tool_ends
                .iter()
                .filter(|(_, _, end)| *end == Some(index))
                .map(end_event),
        );
        let Some(message) = first_message(entry) else {
            events.push(AgentEvent::EntryAppended {
                entry: (*entry).clone(),
            });
            continue;
        };
        let assistant = matches!(message, Message::Assistant(_));
        // A streamed answer already started with its first partial.
        let streamed = assistant && partial_before.is_some() && !assistant_appended;
        if assistant {
            assistant_appended = true;
        }
        if !streamed {
            events.push(AgentEvent::MessageStart {
                message: message.clone(),
            });
        }
        events.push(AgentEvent::MessageEnd {
            entry: (*entry).clone(),
        });
    }
    // Ends without a result entry: a faulted or orphaned tool, or one whose run ended.
    events.extend(
        tool_ends
            .iter()
            .filter(|(_, _, end)| end.is_none())
            .map(end_event),
    );

    // Compaction ends, task failures, then turn and run ends.
    let no_compactions = Vec::new();
    let compactions_before = was.live.compactions.as_ref().unwrap_or(&no_compactions);
    let compactions = now.live.compactions.as_ref().unwrap_or(&no_compactions);
    for status in compactions_before {
        if !compactions
            .iter()
            .any(|current| current.task_id == status.task_id)
        {
            events.push(AgentEvent::CompactionEnd {
                task_id: status.task_id.erase(),
                reason: status.reason,
            });
        }
    }
    // A generation's turn ends when its outcome is committed: at a `completing` hold or at terminal, whichever comes
    // first, so a successor created at the hold starts after it.
    let mut turn_ended = false;
    for task in tasks.values() {
        let generation = task.kind == "pi.generation";
        if generation && matches!(task.state, TaskState::Completing { .. }) && held.insert(task.id)
        {
            turn_ended = true;
        }
        let TaskState::Terminal { outcome } = &task.state else {
            continue;
        };
        if generation && !held.remove(&task.id) {
            turn_ended = true;
        }
        let message = match outcome {
            TaskOutcome::Faulted { error } => Some(error.message.clone()),
            TaskOutcome::Orphaned { reason } => Some(reason.clone()),
            _ => None,
        };
        if let Some(message) = message {
            events.push(AgentEvent::TaskFailed {
                task_id: task.id,
                kind: task.kind.clone(),
                message,
            });
        }
    }
    if turn_ended {
        events.push(AgentEvent::TurnEnd);
    }
    let run = now.live.run.as_ref();
    let run_before = was.live.run.as_ref();
    let run_changed =
        run.and_then(|run| run.inputs.first()) != run_before.and_then(|run| run.inputs.first());
    if let (Some(run_before), true) = (run_before, run_changed) {
        events.push(AgentEvent::RunEnd {
            inputs: run_before.inputs.clone(),
        });
    }

    // Submissions, document state, then what began.
    for record in submissions {
        events.push(AgentEvent::Submission {
            record: record.clone(),
        });
    }
    if !same_doc(&now.inbox, &was.inbox) {
        events.push(AgentEvent::InboxUpdate {
            items: queued(&now.inbox),
        });
    }
    // A retired document reads as its initial value, as in a snapshot.
    if !same_doc(&now.agent, &was.agent) {
        events.push(AgentEvent::AgentChanged {
            agent: decode_doc(&now.agent),
        });
    }
    if !same_doc(&now.usage, &was.usage) {
        events.push(AgentEvent::UsageChanged {
            usage: decode_doc(&now.usage),
        });
    }
    for status in compactions {
        if !compactions_before
            .iter()
            .any(|previous| previous.task_id == status.task_id)
        {
            events.push(AgentEvent::CompactionStart {
                task_id: status.task_id.erase(),
                reason: status.reason,
                blocking: status.blocking,
            });
        }
    }
    if let (Some(run), true) = (run, run_changed) {
        events.push(AgentEvent::RunStart {
            inputs: run.inputs.clone(),
        });
    }
    if let Some(run) = run
        && Some(run.task_id) != run_before.map(|run| run.task_id)
        && tasks
            .get(&run.task_id)
            .is_some_and(|task| task.kind == "pi.generation")
    {
        events.push(AgentEvent::TurnStart);
    }
    events
}

fn key_path(keys: &[&str]) -> Path {
    keys.iter().map(|key| Seg::from(*key)).collect()
}

fn op_path(op: &Op) -> Option<&Path> {
    match op {
        Op::R(_) => None,
        Op::S(path, _)
        | Op::D(path)
        | Op::A(path, _)
        | Op::T(path, _)
        | Op::P(path, ..)
        | Op::M(path, _) => Some(path),
    }
}

fn starts_with(path: &[Seg], prefix: &[Seg]) -> bool {
    prefix.len() <= path.len() && prefix.iter().zip(path).all(|(a, b)| a == b)
}

/// Translate the view operations on the in-flight message into message changes (spec §9.4).
fn message_changes(view_ops: &[Op], message: &AssistantMessage) -> Vec<MessageChange> {
    let partial_path = key_path(&["docs", "pi.live", "generation", "message"]);
    let whole_message = || {
        vec![MessageChange::Message {
            message: message.clone(),
        }]
    };
    let mut changes = Vec::new();
    // A block sent whole already holds every later change to it in this batch.
    let mut whole: HashSet<usize> = HashSet::new();
    for op in view_ops {
        // View operations never replace the root.
        let Some(path) = op_path(op) else {
            return whole_message();
        };
        if !starts_with(path, &partial_path) {
            // The whole message or generation was replaced.
            if starts_with(&partial_path, path) {
                return whole_message();
            }
            continue;
        }
        let rest = &path[partial_path.len()..];
        match rest.first() {
            Some(Seg::Key(key)) if key == "usage" => continue,
            Some(Seg::Key(key)) if key == "content" => {}
            _ => return whole_message(),
        }
        if rest.len() == 1 {
            let Op::P(_, index, 0, items) = op else {
                return whole_message();
            };
            for (offset, item) in items.iter().enumerate() {
                let Ok(block) = serde_json::from_value::<AssistantContent>(item.clone()) else {
                    return whole_message();
                };
                let content_index = index + offset;
                changes.push(match block {
                    AssistantContent::Text(_) => MessageChange::TextStart {
                        content_index,
                        block,
                    },
                    AssistantContent::Thinking(_) => MessageChange::ThinkingStart {
                        content_index,
                        block,
                    },
                    AssistantContent::ToolCall(_) => MessageChange::ToolcallStart {
                        content_index,
                        block,
                    },
                });
            }
            continue;
        }
        let Seg::Index(content_index) = rest[1] else {
            return whole_message();
        };
        let field = match rest.get(2) {
            Some(Seg::Key(key)) => Some(key.as_str()),
            _ => None,
        };
        if whole.contains(&content_index) {
            continue;
        }
        match (op, field) {
            (Op::A(_, delta), Some("text")) if rest.len() == 3 => {
                changes.push(MessageChange::TextDelta {
                    content_index,
                    delta: delta.clone(),
                })
            }
            (Op::A(_, delta), Some("thinking")) if rest.len() == 3 => {
                changes.push(MessageChange::ThinkingDelta {
                    content_index,
                    delta: delta.clone(),
                })
            }
            (Op::A(_, delta), Some("arguments")) => changes.push(MessageChange::ToolcallDelta {
                content_index,
                path: rest[3..].to_vec(),
                delta: delta.clone(),
            }),
            _ => {
                whole.insert(content_index);
                let Some(block) = message.content.get(content_index) else {
                    return whole_message();
                };
                changes.push(MessageChange::Block {
                    content_index,
                    block: block.clone(),
                });
            }
        }
    }
    changes
}

type ToolUpdate = (
    Option<ToolOutputUpdate>,
    Option<JsonValue>,
    Option<Vec<ToolDiagnostic>>,
);

/// Output, details, and diagnostics changes of a running slot, from the view operations on it.
fn tool_update(
    view_ops: &[Op],
    index: usize,
    slot: &ToolSlot,
    previous: &ToolSlot,
) -> Option<ToolUpdate> {
    let mut output_path = key_path(&["docs", "pi.live", "tools"]);
    output_path.push(Seg::from(index));
    output_path.push(Seg::from("output"));
    let mut trim_start = 0;
    let mut append = String::new();
    let mut set = false;
    for op in view_ops {
        if !op_path(op).is_some_and(|path| starts_with(path, &output_path)) {
            continue;
        }
        match op {
            Op::T(_, count) => trim_start += count,
            Op::A(_, text) => append.push_str(text),
            _ => set = true,
        }
    }
    let output = if set || (slot.output != previous.output && trim_start == 0 && append.is_empty())
    {
        Some(ToolOutputUpdate::Set {
            set: slot.output.clone().unwrap_or_default(),
        })
    } else if trim_start > 0 || !append.is_empty() {
        Some(ToolOutputUpdate::Delta {
            trim_start: (trim_start > 0).then_some(trim_start),
            append: (!append.is_empty()).then_some(append),
        })
    } else {
        None
    };
    // A safe replay clears a running slot's progress: removed details send `null`, removed diagnostics `[]`.
    let details =
        (slot.details != previous.details).then(|| slot.details.clone().unwrap_or(JsonValue::Null));
    let diagnostics = (slot.diagnostics != previous.diagnostics)
        .then(|| slot.diagnostics.clone().unwrap_or_default());
    if output.is_none() && details.is_none() && diagnostics.is_none() {
        return None;
    }
    Some((output, details, diagnostics))
}
