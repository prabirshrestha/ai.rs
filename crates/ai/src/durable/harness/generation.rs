//! Port of durable `src/harness/generation.ts` (milestone 6 subset: the generation task's identity, run control, and
//! partial conversion; its phases arrive with milestone 7).

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::durable::entries::ASSISTANT_ENTRY;
use crate::durable::errors::Result;
use crate::durable::ids::{ConversationId, EntryId, SubmissionId, TaskId};
use crate::durable::session::{DocDraft, Transaction};
use crate::durable::tasks::{Task, TaskDefinition, define_task};
use crate::durable::types::{EntryRecord, JsonObject, TaskOptions, TaskOwnership, TypedEntryDraft};
use crate::types::{AssistantMessage, Message, StopReason};

use super::live::{LiveState, RunState};
use super::types::GenerationHooks;
use super::usage::{UsageBucket, record_usage};

pub type GenerationInput = JsonObject;

/// `GenerationCheckpoint`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
pub enum GenerationCheckpoint {
    #[serde(rename_all = "camelCase")]
    Prepare { attempt: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationResult {
    pub entry_id: EntryId,
}

pub type GenerationTaskType =
    Task<GenerationInput, GenerationCheckpoint, GenerationResult, GenerationHooks>;

/// Built-in generation task.
pub static GENERATION_TASK: LazyLock<GenerationTaskType> = LazyLock::new(|| {
    define_task(TaskDefinition::new(
        "pi.generation",
        1,
        |_: &GenerationInput| GenerationCheckpoint::Prepare { attempt: 1 },
    ))
});

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
