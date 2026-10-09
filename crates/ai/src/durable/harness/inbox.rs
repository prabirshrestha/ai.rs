//! Port of durable `src/harness/inbox.ts`.

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::chord::JsonValue;
use crate::durable::documents::{DocToken, define_doc};
use crate::durable::entries::USER_ENTRY;
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, EntryId, SubmissionId};
use crate::durable::session::{DocDraft, Transaction};
use crate::durable::types::{
    ContextEdit, DocDefinition, EntryDraft, EntryHead, JsonObject, LatestConversation, LatestFork,
    SubmissionSettlement, TypedEntryDraft,
};
use crate::types::{Message, UserMessage};

use super::types::{QueueMode, Settings, UserInput};

/// A queued submission: user input for a run, or a passive entry write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode")]
pub enum InboxItem {
    #[serde(rename = "steer")]
    Steer {
        id: SubmissionId,
        content: UserInput,
    },
    #[serde(rename = "followUp")]
    FollowUp {
        id: SubmissionId,
        content: UserInput,
    },
    /// `entry` is an `EntryDraft`, stored as plain JSON.
    #[serde(rename = "write")]
    Write { id: SubmissionId, entry: JsonObject },
}

impl InboxItem {
    pub fn id(&self) -> SubmissionId {
        match self {
            Self::Steer { id, .. } | Self::FollowUp { id, .. } | Self::Write { id, .. } => *id,
        }
    }

    fn is_write(&self) -> bool {
        matches!(self, Self::Write { .. })
    }
}

/// Built-in queue of one conversation's submissions waiting for a boundary, in ID order.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct InboxState {
    pub items: Vec<InboxItem>,
}

pub static INBOX_DOC: LazyLock<DocToken<InboxState, LatestConversation>> = LazyLock::new(|| {
    define_doc(
        DocDefinition::new(
            "pi.inbox",
            1,
            LatestConversation {
                fork: LatestFork::Initial,
            },
            InboxState::default,
        )
        .checkpoint_when(|value, _, _| {
            Ok(value["items"]
                .as_array()
                .is_none_or(|items| items.is_empty()))
        }),
    )
    .expect("valid pi.inbox definition")
});

/// The settings a boundary reads, on the Session line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueModes {
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
}

impl From<&Settings> for QueueModes {
    fn from(settings: &Settings) -> Self {
        Self {
            steering_mode: settings.steering_mode,
            follow_up_mode: settings.follow_up_mode,
        }
    }
}

/// What a boundary reads before the commit's first table write, and the newest head it has seen so far.
pub struct Boundary {
    pub conversation_id: ConversationId,
    pub inbox: DocDraft<InboxState>,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    /// Start of the active range, the newest head marker's `head`; advanced by heads written in this commit.
    pub head: Option<EntryId>,
}

/// Selected user items, in ID order, and whether a `head: "self"` write (a reset) was placed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoundaryResult {
    pub users: Vec<SubmissionId>,
    pub reset: bool,
}

/// Which boundary is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind {
    PostTools,
    Final,
}

/// Read what a boundary needs. Table reads must precede the commit's first table write, so callers prepare the
/// boundary at the start of their commit.
pub async fn prepare_boundary(
    tx: &Transaction,
    conversation_id: ConversationId,
    modes: QueueModes,
) -> Result<Boundary> {
    let head = tx
        .latest_head_marker(conversation_id)
        .await?
        .and_then(|marker| marker.head);
    let inbox = tx.doc(&*INBOX_DOC, conversation_id).await?;
    Ok(Boundary {
        conversation_id,
        inbox,
        steering_mode: modes.steering_mode,
        follow_up_mode: modes.follow_up_mode,
        head,
    })
}

/// An `EntryDraft` as the plain JSON the inbox stores.
pub fn entry_draft_json(draft: &EntryDraft) -> Result<JsonObject> {
    let mut object = JsonObject::new();
    object.insert("kind".into(), JsonValue::String(draft.kind.clone()));
    if let Some(model) = &draft.model {
        object.insert("model".into(), serde_json::to_value(model)?);
    }
    if let Some(data) = &draft.data {
        object.insert("data".into(), data.clone());
    }
    if let Some(head) = draft.head {
        object.insert(
            "head".into(),
            match head {
                EntryHead::SelfEntry => JsonValue::String("self".into()),
                EntryHead::Id(id) => serde_json::to_value(id)?,
            },
        );
    }
    if let Some(edits) = &draft.edits {
        object.insert("edits".into(), serde_json::to_value(edits)?);
    }
    Ok(object)
}

/// Decode an `EntryDraft` stored as plain JSON.
pub fn entry_draft_from_json(object: &JsonObject) -> Result<EntryDraft> {
    let kind = object
        .get("kind")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| Error::type_error("Queued entry has no kind"))?
        .to_string();
    let model = object
        .get("model")
        .map(|model| serde_json::from_value::<Vec<Message>>(model.clone()))
        .transpose()?;
    let head = match object.get("head") {
        None => None,
        Some(JsonValue::String(text)) if text == "self" => Some(EntryHead::SelfEntry),
        Some(value) => Some(EntryHead::Id(serde_json::from_value(value.clone())?)),
    };
    let edits = object
        .get("edits")
        .map(|edits| serde_json::from_value::<Vec<ContextEdit>>(edits.clone()))
        .transpose()?;
    Ok(EntryDraft {
        kind,
        model,
        data: object.get("data").cloned(),
        head,
        edits,
    })
}

/// A user message entry with `content`.
pub async fn append_user(
    tx: &Transaction,
    conversation_id: ConversationId,
    content: UserInput,
    now: u64,
) -> Result<crate::durable::types::EntryRecord> {
    let message = Message::User(UserMessage {
        content,
        timestamp: now,
    });
    tx.append_entry_of(
        &USER_ENTRY,
        conversation_id,
        TypedEntryDraft {
            model: Some(vec![message]),
            ..TypedEntryDraft::default()
        },
    )
    .await
}

/// Place the queued items a boundary selects (spec §6): every write, the first or all steers, and at `final` the first
/// or all follow-ups. A selected reset turns a `postTools` boundary into `final`. Writes are placed first and user
/// items after them, each in ID order, so user items queued before a reset run in the new context. A write whose head
/// targets an entry before the active range, including a range started earlier in this commit, is stale. Selected and
/// stale items are removed positionally.
pub async fn apply_boundary(
    tx: &Transaction,
    boundary: &mut Boundary,
    at: BoundaryKind,
    now: u64,
) -> Result<BoundaryResult> {
    let conversation_id = boundary.conversation_id;
    let items = boundary.inbox.get()?.items;
    let reset = items.iter().any(|item| match item {
        InboxItem::Write { entry, .. } => {
            entry.get("head").and_then(JsonValue::as_str) == Some("self")
        }
        _ => false,
    });
    let final_ = at == BoundaryKind::Final || reset;
    let pick = |follow_up: bool, mode: QueueMode| -> Vec<usize> {
        let indexes: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|(_, item)| match item {
                InboxItem::Steer { .. } => !follow_up,
                InboxItem::FollowUp { .. } => follow_up,
                InboxItem::Write { .. } => false,
            })
            .map(|(index, _)| index)
            .collect();
        match mode {
            QueueMode::All => indexes,
            QueueMode::OneAtATime => indexes.into_iter().take(1).collect(),
        }
    };
    let writes: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, item)| item.is_write())
        .map(|(index, _)| index)
        .collect();
    let mut users = pick(false, boundary.steering_mode);
    if final_ {
        users.extend(pick(true, boundary.follow_up_mode));
    }
    users.sort_unstable();

    for &index in &writes {
        let InboxItem::Write { id, entry } = &items[index] else {
            unreachable!("write index")
        };
        let draft = entry_draft_from_json(entry)?;
        if is_stale(boundary, &draft) {
            tx.settle_submission(
                *id,
                SubmissionSettlement::Unanswered {
                    reason: "stale".into(),
                    detail: None,
                },
            )?;
            continue;
        }
        let head = draft.head;
        let appended = tx.append_entry(conversation_id, draft).await?;
        if let Some(head) = head {
            boundary.head = Some(match head {
                EntryHead::SelfEntry => appended.id,
                EntryHead::Id(head) => head,
            });
        }
        tx.place_submission(*id, appended.id)?;
    }
    let mut placed = Vec::new();
    for &index in &users {
        let (InboxItem::Steer { id, content } | InboxItem::FollowUp { id, content }) =
            &items[index]
        else {
            unreachable!("user index")
        };
        let entry = append_user(tx, conversation_id, content.clone(), now).await?;
        tx.place_submission(*id, entry.id)?;
        placed.push(*id);
    }
    let mut removed: Vec<usize> = writes.into_iter().chain(users).collect();
    removed.sort_unstable_by(|a, b| b.cmp(a));
    boundary.inbox.edit(|inbox| {
        for index in removed {
            inbox.items.remove(index);
        }
    })?;
    Ok(BoundaryResult {
        users: placed,
        reset,
    })
}

/// Whether a head write targets an entry before the active range, so placing it would bring back cut history.
pub fn is_stale(boundary: &Boundary, entry: &EntryDraft) -> bool {
    match (entry.head, boundary.head) {
        (Some(EntryHead::Id(head)), Some(start)) => head < start,
        _ => false,
    }
}

/// Remove a withdrawn submission's item; the caller settles the submission.
pub async fn remove_inbox_item(
    tx: &Transaction,
    conversation_id: ConversationId,
    id: SubmissionId,
) -> Result<()> {
    let inbox = tx.doc(&*INBOX_DOC, conversation_id).await?;
    inbox.edit(|inbox| {
        if let Some(index) = inbox.items.iter().position(|item| item.id() == id) {
            inbox.items.remove(index);
        }
    })
}

/// Withdraw every queued input of a conversation, as `Conversation.abort()` and abort cascades do: each settles
/// `unanswered` with `aborted` and leaves the inbox; queued writes stay for later placement.
pub async fn withdraw_queued_inputs(
    tx: &Transaction,
    conversation_id: ConversationId,
) -> Result<()> {
    let inbox = tx.doc(&*INBOX_DOC, conversation_id).await?;
    let mut withdrawn = Vec::new();
    inbox.edit(|inbox| {
        inbox.items.retain(|item| {
            if item.is_write() {
                return true;
            }
            withdrawn.push(item.id());
            false
        })
    })?;
    for id in withdrawn.into_iter().rev() {
        tx.settle_submission(
            id,
            SubmissionSettlement::Unanswered {
                reason: "aborted".into(),
                detail: None,
            },
        )?;
    }
    Ok(())
}
