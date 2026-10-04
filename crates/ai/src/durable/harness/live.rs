//! Port of durable `src/harness/live.ts`.

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::chord::JsonValue;
use crate::durable::documents::{DocToken, define_doc};
use crate::durable::errors::Result;
use crate::durable::ids::{EntryId, SubmissionId, TaskId};
use crate::durable::session::Transaction;
use crate::durable::types::{
    DocDefinition, LatestConversation, LatestFork, SubmissionSettlement, TaskOutcome, TaskRecord,
};
use crate::types::AssistantMessage;

use super::generation::convert_partial;
use super::types::{CompactionReason, CompactionResult, ToolDiagnostic};

/// `ToolSlot["status"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SlotStatus {
    Pending,
    Running,
    Done,
}

/// Presentation of one tool call of the current round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSlot {
    pub call_id: String,
    pub name: String,
    /// Absent for a call not started yet (sequential round) and for a call its request did not offer, which starts
    /// `done` with the `entry` generation wrote.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    pub status: SlotStatus,
    /// Retained running output and what the bounds dropped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropped_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropped_lines: Option<u64>,
    /// Last `details()` value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
    /// Diagnostics recorded through `api.diagnostic()`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<ToolDiagnostic>>,
    /// Result entry once done; absent when the tool task faulted or was orphaned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<EntryId>,
}

impl ToolSlot {
    pub fn new(call_id: impl Into<String>, name: impl Into<String>, status: SlotStatus) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            task_id: None,
            status,
            output: None,
            dropped_bytes: None,
            dropped_lines: None,
            details: None,
            diagnostics: None,
            entry: None,
        }
    }
}

/// A durable backoff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryStatus {
    pub at: u64,
    pub error: String,
}

/// Presentation of one live compaction task (spec §8.7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionStatus {
    pub task_id: TaskId<CompactionResult>,
    pub reason: CompactionReason,
    /// Whether a generation waits for it: a compaction the generation owns.
    pub blocking: bool,
    pub attempt: u32,
    /// Durable backoff before the next summarization attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryStatus>,
}

/// Run control: the task that settles the run's inputs, and those inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunState {
    pub task_id: TaskId,
    pub inputs: Vec<SubmissionId>,
}

/// `{ pollAt }` of a deferred response being polled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredStatus {
    pub poll_at: u64,
}

/// Presentation of the current generation attempt.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationStatus {
    pub attempt: u32,
    /// Committed throttled partial of the in-flight response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<AssistantMessage>,
    /// Durable backoff before the next attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryStatus>,
    /// Provider-side deferred response being polled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredStatus>,
}

impl GenerationStatus {
    pub fn attempt(attempt: u32) -> Self {
        Self {
            attempt,
            ..Self::default()
        }
    }
}

/// Built-in live conversation state: run control and presentation of the current generation and tool round.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveState {
    /// Present exactly while busy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<RunState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationStatus>,
    /// The current tool round in call order, from the tool-calling answer until the generation's `tools` phase ends it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSlot>>,
    /// Live compaction tasks in task ID order; absent when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compactions: Option<Vec<CompactionStatus>>,
}

pub static LIVE_DOC: LazyLock<DocToken<LiveState, LatestConversation>> = LazyLock::new(|| {
    define_doc(
        DocDefinition::new(
            "pi.live",
            1,
            LatestConversation {
                fork: LatestFork::Initial,
            },
            LiveState::default,
        )
        // REMINDER: a complete base whenever nothing runs (spec §8.2): no generation and no running tool slot.
        .checkpoint_when(|value, _, _| {
            let running = value["tools"].as_array().is_some_and(|slots| {
                slots
                    .iter()
                    .any(|slot| slot["status"].as_str() == Some("running"))
            });
            Ok(value.get("generation").is_none() && !running)
        }),
    )
    .expect("valid pi.live definition")
});

/// Built-in task kinds that can own `pi.live.run`.
const RUN_TASK_KINDS: &[&str] = &["pi.generation"];
const TOOL_TASK_KIND: &str = "pi.tool";
const COMPACTION_TASK_KIND: &str = "pi.compaction";

/// End the run owned by `task_id`: settle each of its inputs and remove `run`. Always removes `generation` and `tools`,
/// whose presentation belongs to the ending run.
pub fn end_run(
    tx: &Transaction,
    live: &mut LiveState,
    task_id: TaskId,
    settlement: SubmissionSettlement,
) -> Result<()> {
    if live.run.as_ref().is_some_and(|run| run.task_id == task_id) {
        for id in &live.run.as_ref().expect("run").inputs {
            tx.settle_submission(*id, settlement.clone())?;
        }
        live.run = None;
    }
    live.generation = None;
    live.tools = None;
    Ok(())
}

/// Add the status of a compaction task created in this commit; statuses stay in task ID order.
pub fn add_compaction_status(live: &mut LiveState, status: CompactionStatus) {
    live.compactions.get_or_insert_with(Vec::new).push(status);
}

/// The status of compaction task `task_id`, if listed.
pub fn compaction_status(live: &mut LiveState, task_id: TaskId) -> Option<&mut CompactionStatus> {
    live.compactions
        .as_mut()?
        .iter_mut()
        .find(|status| status.task_id == task_id)
}

/// Remove the status of compaction task `task_id`, and the list once empty.
pub fn remove_compaction_status(live: &mut LiveState, task_id: TaskId) {
    let Some(statuses) = &mut live.compactions else {
        return;
    };
    if let Some(index) = statuses.iter().position(|status| status.task_id == task_id) {
        statuses.remove(index);
    }
    if statuses.is_empty() {
        live.compactions = None;
    }
}

/// The slot of tool task `task_id` in the current round, if the round still lists it.
pub fn tool_slot(live: &mut LiveState, task_id: TaskId) -> Option<&mut ToolSlot> {
    live.tools
        .as_mut()?
        .iter_mut()
        .find(|slot| slot.task_id == Some(task_id))
}

/// Mark a slot done: the result entry, if any, now carries its running output, details, and diagnostics.
pub fn finish_slot(slot: &mut ToolSlot, entry: Option<EntryId>) {
    slot.status = SlotStatus::Done;
    if entry.is_some() {
        slot.entry = entry;
    }
    clear_progress(slot);
}

/// Remove what a tool published while running; its result entry or a rerun replaces it.
pub fn clear_progress(slot: &mut ToolSlot) {
    slot.output = None;
    slot.dropped_bytes = None;
    slot.dropped_lines = None;
    slot.details = None;
    slot.diagnostics = None;
}

/// Harness cleanup for a terminal outcome the scheduler writes itself (`faulted` or `orphaned`). A run task ends its
/// run; a tool task's slot is marked done without an entry, and context derivation synthesizes the missing result; a
/// compaction task's status is removed. Ignores other kinds so it never creates `pi.live` elsewhere.
pub async fn settle_scheduler_outcome(
    tx: &Transaction,
    record: &TaskRecord,
    outcome: &TaskOutcome,
) -> Result<()> {
    if record.kind == TOOL_TASK_KIND {
        let live = tx.doc(&*LIVE_DOC, record.conversation_id).await?;
        return live.edit(|live| {
            if let Some(slot) = tool_slot(live, record.id) {
                finish_slot(slot, None);
            }
        });
    }
    if record.kind == COMPACTION_TASK_KIND {
        let live = tx.doc(&*LIVE_DOC, record.conversation_id).await?;
        return live.edit(|live| remove_compaction_status(live, record.id));
    }
    if !RUN_TASK_KINDS.contains(&record.kind.as_str()) {
        return Ok(());
    }
    let live = tx.doc(&*LIVE_DOC, record.conversation_id).await?;
    if live.get()?.run.is_none_or(|run| run.task_id != record.id) {
        return Ok(());
    }
    convert_partial(tx, &live, record.conversation_id).await?;
    let settlement = match outcome {
        TaskOutcome::Faulted { error } => SubmissionSettlement::Unanswered {
            reason: "faulted".into(),
            detail: Some(JsonValue::String(error.message.clone())),
        },
        TaskOutcome::Orphaned { reason } => SubmissionSettlement::Unanswered {
            reason: reason.clone(),
            detail: None,
        },
        _ => return Ok(()),
    };
    let mut state = live.get()?;
    end_run(tx, &mut state, record.id, settlement)?;
    live.set(state)
}
