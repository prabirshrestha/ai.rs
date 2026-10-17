//! Port of durable `src/types.ts`: the core data model and the [`Storage`]
//! interface.
//!
//! Records serialize to exactly Pi's JSON shapes (camelCase keys, `type` /
//! `status` / `kind` discriminators, omitted optional fields), so stored
//! records stay byte-compatible with the TS backends. Field order follows the
//! TS type declarations.
//!
//! Divergences from Pi:
//! - Unions whose members share fields (`SubmissionRecord`, `DocumentRecord`,
//!   `DocumentCreate`) are flat structs with optional fields. Pi's
//!   per-status field exclusions are conventions the Session upholds, not types.
//! - `TaskRecord` is the stored, erased `TaskRecord<JsonValue, JsonValue, JsonValue>`;
//!   typed task records arrive with the scheduler.
//! - `Tx`, `Session`, `DocumentObserver` and `WatchHandle` are concrete types in
//!   [`super::session`]; their overloads are expressed through
//!   [`super::documents::DocAccess`].
//! - Rust has no `undefined`. Pi writes records through
//!   `copyJson(.., { omitUndefinedProperties })`, so a void task input,
//!   checkpoint or completed result has no key at all, while a `null` one is
//!   written. Those `JsonValue` fields read a missing key as `null`; Rust
//!   writes `null` (a Rust `()` serializes to `null`), never an omitted key,
//!   so a void value written from Rust reads back in TS as `null`. Omitting
//!   `null` instead would lose the `null` inputs and results Pi's own
//!   records carry.
//! - `Storage.entry(conversationId, id)` is [`Storage::entry_in`]; `mintId<I>()`
//!   returns the raw number, branded by the caller with
//!   [`super::ids::id_from_number`].

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize, Serializer};

use crate::chord::delta::Op;
use crate::chord::{Context, JsonValue};
use crate::types::Message;

use super::errors::{Error, Result};
use super::ids::{ConversationId, DocumentId, EntryId, Seq, SubmissionId, TaskId};

/// JSON object used as the root of every durable document.
pub type JsonObject = serde_json::Map<String, JsonValue>;

/// Deserialize a present field, including an explicit `null`, as `Some`.
pub(crate) fn present<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<JsonValue>, D::Error> {
    JsonValue::deserialize(deserializer).map(Some)
}

// ─── Document semantics ──────────────────────────────────────────────────────

/// `history`: what a conversation document retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum History {
    /// Retain only current state.
    Latest,
    /// Retain history needed for as-of reads.
    Rewindable,
}

/// `fork`: how a forked conversation initializes a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ForkPolicy {
    /// From the source state at the fork entry (rewindable only).
    AsOf,
    /// From the source's current state.
    Current,
    /// From the definition's initial value (the document starts absent).
    Initial,
}

impl ForkPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AsOf => "asOf",
            Self::Current => "current",
            Self::Initial => "initial",
        }
    }
}

/// `DocumentSemantics["scope"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeKind {
    Session,
    Conversation,
    Task,
}

impl ScopeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Conversation => "conversation",
            Self::Task => "task",
        }
    }
}

impl fmt::Display for ScopeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Ownership and lifetime of a document; only conversation documents declare history and fork behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DocumentSemantics {
    Session,
    Conversation { history: History, fork: ForkPolicy },
    Task,
}

impl DocumentSemantics {
    pub fn scope(&self) -> ScopeKind {
        match self {
            Self::Session => ScopeKind::Session,
            Self::Conversation { .. } => ScopeKind::Conversation,
            Self::Task => ScopeKind::Task,
        }
    }

    pub fn history(&self) -> Option<History> {
        match self {
            Self::Conversation { history, .. } => Some(*history),
            _ => None,
        }
    }

    pub fn fork(&self) -> Option<ForkPolicy> {
        match self {
            Self::Conversation { fork, .. } => Some(*fork),
            _ => None,
        }
    }
}

/// `fork` of a latest-only conversation document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LatestFork {
    Current,
    Initial,
}

/// `fork` of a rewindable conversation document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RewindableFork {
    AsOf,
    Current,
    Initial,
}

/// A document scope marker. TS selects `defineDoc` overloads by the literal
/// `scope`/`history`/`fork` fields; Rust selects the typed token by this marker.
pub trait DocScope: Copy + Send + Sync + 'static {
    /// The owner argument: `()` for Session documents, a conversation or task ID otherwise.
    type Owner: Copy + Send + Sync + 'static;
    fn semantics(&self) -> DocumentSemantics;
    /// The erased owner ID passed to address resolution.
    fn owner_id(owner: Self::Owner) -> Option<u64>;
}

/// `{ scope: "session" }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SessionScope;

/// `{ scope: "task" }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TaskScope;

/// `LatestConversationSemantics`: `{ scope: "conversation", history: "latest", fork }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatestConversation {
    pub fork: LatestFork,
}

/// `RewindableConversationSemantics`: `{ scope: "conversation", history: "rewindable", fork }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RewindableConversation {
    pub fork: RewindableFork,
}

/// Conversation scopes (latest or rewindable).
pub trait ConversationScope: DocScope<Owner = ConversationId> {}

impl DocScope for SessionScope {
    type Owner = ();
    fn semantics(&self) -> DocumentSemantics {
        DocumentSemantics::Session
    }
    fn owner_id((): ()) -> Option<u64> {
        None
    }
}

impl DocScope for TaskScope {
    type Owner = TaskId;
    fn semantics(&self) -> DocumentSemantics {
        DocumentSemantics::Task
    }
    fn owner_id(owner: TaskId) -> Option<u64> {
        Some(owner.0)
    }
}

impl DocScope for LatestConversation {
    type Owner = ConversationId;
    fn semantics(&self) -> DocumentSemantics {
        DocumentSemantics::Conversation {
            history: History::Latest,
            fork: match self.fork {
                LatestFork::Current => ForkPolicy::Current,
                LatestFork::Initial => ForkPolicy::Initial,
            },
        }
    }
    fn owner_id(owner: ConversationId) -> Option<u64> {
        Some(owner.0)
    }
}

impl DocScope for RewindableConversation {
    type Owner = ConversationId;
    fn semantics(&self) -> DocumentSemantics {
        DocumentSemantics::Conversation {
            history: History::Rewindable,
            fork: match self.fork {
                RewindableFork::AsOf => ForkPolicy::AsOf,
                RewindableFork::Current => ForkPolicy::Current,
                RewindableFork::Initial => ForkPolicy::Initial,
            },
        }
    }
    fn owner_id(owner: ConversationId) -> Option<u64> {
        Some(owner.0)
    }
}

impl ConversationScope for LatestConversation {}
impl ConversationScope for RewindableConversation {}

/// Stored replay state supplied to a document's checkpoint predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointInfo {
    /// Deltas already stored after the newest base, excluding the change being evaluated.
    pub deltas_since_base: u64,
}

/// `checkpointWhen(value, ops, info)`: return true to store this ordinary change as a complete base.
pub type CheckpointWhen =
    Arc<dyn Fn(&JsonValue, &[Op], CheckpointInfo) -> Result<bool> + Send + Sync>;

/// `migrate(value, fromVersion)`.
pub type Migrate<T> = Arc<dyn Fn(JsonObject, u32) -> Result<T> + Send + Sync>;

/// Singleton document definition (`CommonDocDefinition<T> & DocumentSemantics`).
pub struct DocDefinition<T, S: DocScope> {
    /// Stable persisted kind; part of the public protocol.
    pub kind: String,
    /// Positive integer version of the stored value shape.
    pub version: u32,
    pub scope: S,
    pub initial: Arc<dyn Fn() -> T + Send + Sync>,
    pub migrate: Option<Migrate<T>>,
    pub checkpoint_when: Option<CheckpointWhen>,
}

impl<T, S: DocScope> DocDefinition<T, S> {
    pub fn new(
        kind: impl Into<String>,
        version: u32,
        scope: S,
        initial: impl Fn() -> T + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind: kind.into(),
            version,
            scope,
            initial: Arc::new(initial),
            migrate: None,
            checkpoint_when: None,
        }
    }

    pub fn migrate(
        mut self,
        migrate: impl Fn(JsonObject, u32) -> Result<T> + Send + Sync + 'static,
    ) -> Self {
        self.migrate = Some(Arc::new(migrate));
        self
    }

    pub fn checkpoint_when(
        mut self,
        checkpoint_when: impl Fn(&JsonValue, &[Op], CheckpointInfo) -> Result<bool>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.checkpoint_when = Some(Arc::new(checkpoint_when));
        self
    }
}

/// Keyed document family definition; `initial(seed)` runs only when a member is absent.
pub struct DocFamilyDefinition<T, I, S: DocScope> {
    pub kind: String,
    pub version: u32,
    pub scope: S,
    pub initial: Arc<dyn Fn(I) -> T + Send + Sync>,
    pub migrate: Option<Migrate<T>>,
    pub checkpoint_when: Option<CheckpointWhen>,
}

impl<T, I, S: DocScope> DocFamilyDefinition<T, I, S> {
    pub fn new(
        kind: impl Into<String>,
        version: u32,
        scope: S,
        initial: impl Fn(I) -> T + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind: kind.into(),
            version,
            scope,
            initial: Arc::new(initial),
            migrate: None,
            checkpoint_when: None,
        }
    }

    pub fn migrate(
        mut self,
        migrate: impl Fn(JsonObject, u32) -> Result<T> + Send + Sync + 'static,
    ) -> Self {
        self.migrate = Some(Arc::new(migrate));
        self
    }

    pub fn checkpoint_when(
        mut self,
        checkpoint_when: impl Fn(&JsonValue, &[Op], CheckpointInfo) -> Result<bool>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.checkpoint_when = Some(Arc::new(checkpoint_when));
        self
    }
}

// ─── Records ─────────────────────────────────────────────────────────────────

/// Ownership selected explicitly whenever a conversation is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationOwnership {
    Ownerless,
    Task { task_id: TaskId },
}

/// Fork source and inclusive parent entry through which history is inherited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationParent {
    pub conversation_id: ConversationId,
    pub at: EntryId,
}

/// Creator edge used for attribution, subtree abort, and subtree idle waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationOwner {
    pub conversation_id: ConversationId,
    pub task_id: TaskId,
}

/// Immutable identity, history ancestry, and task ownership of a transcript scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationRecord {
    pub id: ConversationId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<ConversationParent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<ConversationOwner>,
}

impl ConversationRecord {
    pub fn new(id: ConversationId) -> Self {
        Self {
            id,
            parent: None,
            owner: None,
        }
    }
}

/// `ContextEdit["action"]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "camelCase")]
pub enum ContextEditAction {
    Omit,
    Replace { messages: Vec<Message> },
}

/// An immutable override of one visible entry's contribution to model context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextEdit {
    /// Entry whose model messages are omitted or replaced.
    pub target: EntryId,
    #[serde(flatten)]
    pub action: ContextEditAction,
}

/// Immutable transcript event with separate model-facing and application-facing payloads.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryRecord {
    pub id: EntryId,
    pub conversation_id: ConversationId,
    /// Application-defined entry discriminator.
    pub kind: String,
    /// Messages contributed to model context; absent for display or bookkeeping entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<Vec<Message>>,
    /// JSON payload consumed by views, extensions, or bookkeeping logic.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub data: Option<JsonValue>,
    /// First entry in the active context selected by this entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<EntryId>,
    /// Context-only overrides of earlier visible entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edits: Option<Vec<ContextEdit>>,
    /// Task that appended this entry, when it was produced by durable work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by_task_id: Option<TaskId>,
}

/// `EntryDraft["head"]`: an entry ID, or `"self"` to start active context at the new entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryHead {
    Id(EntryId),
    SelfEntry,
}

/// Entry content supplied before the Session assigns identity and task attribution.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EntryDraft {
    pub kind: String,
    pub model: Option<Vec<Message>>,
    pub data: Option<JsonValue>,
    pub head: Option<EntryHead>,
    pub edits: Option<Vec<ContextEdit>>,
}

impl EntryDraft {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            ..Self::default()
        }
    }

    pub fn data(mut self, data: impl Into<JsonValue>) -> Self {
        self.data = Some(data.into());
        self
    }

    pub fn head(mut self, head: EntryHead) -> Self {
        self.head = Some(head);
        self
    }
}

/// Entry content of a typed kind; the token supplies `kind` (`TypedEntryDraft<D>`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TypedEntryDraft<D> {
    pub model: Option<Vec<Message>>,
    pub data: Option<D>,
    pub head: Option<EntryHead>,
    pub edits: Option<Vec<ContextEdit>>,
}

/// `SubmissionRecord["type"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SubmissionType {
    #[default]
    Input,
    Write,
}

/// `SubmissionRecord["status"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SubmissionStatus {
    /// Admitted but not yet represented in the transcript.
    #[default]
    Queued,
    /// Added to the transcript and owned by an active run (input only).
    Placed,
    /// Successfully answered input or appended passive entry.
    Done,
    /// Terminal submission that can no longer be answered or placed.
    Unanswered,
}

/// Durable lifecycle of one admitted user input or passive entry write.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmissionRecord {
    pub id: SubmissionId,
    pub conversation_id: ConversationId,
    /// Host-provided deduplication key, scoped to the conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(rename = "type")]
    pub type_: SubmissionType,
    pub status: SubmissionStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<EntryId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<EntryId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub detail: Option<JsonValue>,
}

/// Submission fields supplied before the Session assigns an ID (`SubmissionCreate`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmissionCreate {
    pub conversation_id: ConversationId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(rename = "type")]
    pub type_: SubmissionType,
    pub status: SubmissionStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<EntryId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<EntryId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub detail: Option<JsonValue>,
}

impl SubmissionCreate {
    pub fn with_id(self, id: SubmissionId) -> SubmissionRecord {
        SubmissionRecord {
            id,
            conversation_id: self.conversation_id,
            request_id: self.request_id,
            type_: self.type_,
            status: self.status,
            entry: self.entry,
            answer: self.answer,
            reason: self.reason,
            detail: self.detail,
        }
    }
}

/// Terminal status staged for a submission; identity, type, and entry come from its current record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum SubmissionSettlement {
    Done {
        answer: EntryId,
    },
    Unanswered {
        reason: String,
        #[serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )]
        detail: Option<JsonValue>,
    },
}

/// JSON-safe error snapshot persisted instead of a runtime `Error` object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskOutcomeError {
    pub message: String,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub detail: Option<JsonValue>,
}

/// Durable reason and optional result recorded when a task becomes terminal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum TaskOutcome {
    Completed {
        /// A missing key (Pi's `undefined`, a void result) reads as `null`.
        #[serde(default)]
        result: JsonValue,
    },
    /// Expected task or domain failure explicitly committed by its implementation.
    Failed {
        error: TaskOutcomeError,
        #[serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )]
        result: Option<JsonValue>,
    },
    /// Explicit cancellation handled by the task's abort protocol.
    Aborted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(
            default,
            deserialize_with = "present",
            skip_serializing_if = "Option::is_none"
        )]
        result: Option<JsonValue>,
    },
    /// Task that cannot resume because its definition or migration is unavailable.
    Orphaned { reason: String },
    /// Runtime-detected contract failure, such as an uncaught throw or no durable progress.
    Faulted { error: TaskOutcomeError },
}

/// How a waiting task treats the tasks it waits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JoinPolicy {
    FailFast,
    AllSettled,
}

/// `TaskState["status"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TaskStatus {
    Pending,
    Running,
    Waiting,
    Completing,
    Terminal,
}

impl TaskStatus {
    pub const ALL: [TaskStatus; 5] = [
        TaskStatus::Pending,
        TaskStatus::Running,
        TaskStatus::Waiting,
        TaskStatus::Completing,
        TaskStatus::Terminal,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Waiting => "waiting",
            Self::Completing => "completing",
            Self::Terminal => "terminal",
        }
    }
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Complete durable execution state of a task (erased checkpoint and result).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum TaskState {
    /// Eligible for scheduling.
    Pending {
        /// A missing key (Pi's `undefined`) reads as `null`.
        #[serde(default)]
        checkpoint: JsonValue,
    },
    /// Reserved by one in-memory task invocation.
    Running {
        #[serde(default)]
        checkpoint: JsonValue,
    },
    /// Parked until every task in `on` is terminal; then resumes at `checkpoint`.
    Waiting {
        #[serde(default)]
        checkpoint: JsonValue,
        on: Vec<TaskId>,
        policy: JoinPolicy,
    },
    /// Outcome decided; becomes terminal once no ordinary owned work below is live.
    Completing { outcome: TaskOutcome },
    /// Permanently settled durable result receipt.
    Terminal { outcome: TaskOutcome },
}

impl TaskState {
    pub fn status(&self) -> TaskStatus {
        match self {
            Self::Pending { .. } => TaskStatus::Pending,
            Self::Running { .. } => TaskStatus::Running,
            Self::Waiting { .. } => TaskStatus::Waiting,
            Self::Completing { .. } => TaskStatus::Completing,
            Self::Terminal { .. } => TaskStatus::Terminal,
        }
    }

    pub fn checkpoint(&self) -> Option<&JsonValue> {
        match self {
            Self::Pending { checkpoint }
            | Self::Running { checkpoint }
            | Self::Waiting { checkpoint, .. } => Some(checkpoint),
            _ => None,
        }
    }

    pub fn outcome(&self) -> Option<&TaskOutcome> {
        match self {
            Self::Completing { outcome } | Self::Terminal { outcome } => Some(outcome),
            _ => None,
        }
    }
}

/// Complete replacement record for one durable task state machine
/// (`TaskRecord<JsonValue, JsonValue, JsonValue>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskRecord {
    pub id: TaskId,
    pub conversation_id: ConversationId,
    /// Registered task definition name.
    pub kind: String,
    /// Definition version used to migrate live input and checkpoints.
    pub version: u32,
    /// Original task input retained while the task is live or terminal.
    /// A missing key (Pi's `undefined`, a void input) reads as `null`.
    #[serde(default)]
    pub input: JsonValue,
    /// Owning task of a child task; absent for a task its conversation owns. Immutable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<TaskId>,
    /// Whether this conversation-owned task is excluded from ordinary idle waits, conversation aborts, and cascades.
    pub background: bool,
    /// Durable abort mark checked before run-mode progress is committed.
    pub abort_requested: bool,
    pub state: TaskState,
    /// Small first-writer-wins values retained while the task can run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memos: Option<JsonObject>,
}

/// Who owns a task: its conversation (a top-level task) or another task of the same conversation (a child task).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOwnership {
    Conversation,
    Task { task_id: TaskId },
}

/// Creation options for a durable task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskOptions {
    /// Required: a task always names its owner.
    pub ownership: TaskOwnership,
    /// Default: the owner task's conversation, or the transaction's bound conversation.
    pub conversation_id: Option<ConversationId>,
    /// Conversation-owned tasks only: excluded from ordinary idle waits, conversation aborts, and cascades.
    pub background: Option<bool>,
}

impl TaskOptions {
    pub fn conversation(conversation_id: Option<ConversationId>) -> Self {
        Self {
            ownership: TaskOwnership::Conversation,
            conversation_id,
            background: None,
        }
    }
}

/// `DocumentRecord["scope"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum DocumentScope {
    Session,
    #[serde(rename_all = "camelCase")]
    Conversation {
        conversation_id: ConversationId,
    },
    #[serde(rename_all = "camelCase")]
    Task {
        task_id: TaskId,
    },
}

impl DocumentScope {
    pub fn kind(&self) -> ScopeKind {
        match self {
            Self::Session => ScopeKind::Session,
            Self::Conversation { .. } => ScopeKind::Conversation,
            Self::Task { .. } => ScopeKind::Task,
        }
    }
}

/// Persisted lifecycle record for one create-to-retire document incarnation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentRecord {
    /// Unique incarnation ID; never reused when the same logical document is recreated.
    pub id: DocumentId,
    /// Stable document definition kind.
    pub kind: String,
    /// Family member key; absent for singleton documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Commit that created the incarnation, stamped by storage.
    pub created_at: Seq,
    /// Commit that retired the incarnation; absent while it is current.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retired_at: Option<Seq>,
    pub scope: DocumentScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history: Option<History>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork: Option<ForkPolicy>,
}

/// Fields supplied when storage creates and stamps a new `DocumentRecord`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentCreate {
    pub id: DocumentId,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub scope: DocumentScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history: Option<History>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork: Option<ForkPolicy>,
}

impl DocumentCreate {
    /// Stamp the record with its creating commit.
    pub fn stamp(&self, created_at: Seq) -> DocumentRecord {
        DocumentRecord {
            id: self.id,
            kind: self.kind.clone(),
            key: self.key.clone(),
            created_at,
            retired_at: None,
            scope: self.scope,
            history: self.history,
            fork: self.fork,
        }
    }

    pub fn address(&self) -> DocumentAddress {
        DocumentAddress {
            kind: self.kind.clone(),
            scope: self.scope,
            key: self.key.clone(),
        }
    }
}

impl DocumentRecord {
    pub fn address(&self) -> DocumentAddress {
        DocumentAddress {
            kind: self.kind.clone(),
            scope: self.scope,
            key: self.key.clone(),
        }
    }

    /// The creation fields of this record.
    pub fn create(&self) -> DocumentCreate {
        DocumentCreate {
            id: self.id,
            kind: self.kind.clone(),
            key: self.key.clone(),
            scope: self.scope,
            history: self.history,
            fork: self.fork,
        }
    }
}

/// One ordered scan result and its optional continuation state.
#[derive(Debug, Clone, PartialEq)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub next: Option<Cursor>,
}

/// Backend-owned JSON continuation state that callers only round-trip to the same scan.
pub type Cursor = JsonObject;

/// Optional filters for an ordered conversation scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConversationQuery {
    pub owner_conversation_id: Option<ConversationId>,
    pub owner_task_id: Option<TaskId>,
}

/// Inclusive ID bounds for a newest-first scan of one conversation's fork-aware history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryQuery {
    pub conversation_id: ConversationId,
    /// Oldest entry ID that may be returned.
    pub min_entry_id: Option<EntryId>,
    /// Newest entry ID that may be returned.
    pub max_entry_id: Option<EntryId>,
}

impl EntryQuery {
    pub fn conversation(conversation_id: ConversationId) -> Self {
        Self {
            conversation_id,
            min_entry_id: None,
            max_entry_id: None,
        }
    }
}

/// Optional filters for an ordered scan of durable task records.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TaskQuery {
    pub conversation_id: Option<ConversationId>,
    pub kind: Option<String>,
    pub status: Option<TaskStatus>,
    pub abort_requested: Option<bool>,
    pub background: Option<bool>,
}

/// Optional filters for an ordered scan of submission records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubmissionQuery {
    pub conversation_id: Option<ConversationId>,
    pub status: Option<SubmissionStatus>,
}

/// Current state or one historical commit sequence used for document membership and content reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DocumentPoint {
    Seq(Seq),
    Current,
}

impl From<Seq> for DocumentPoint {
    fn from(seq: Seq) -> Self {
        Self::Seq(seq)
    }
}

impl Serialize for DocumentPoint {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Seq(seq) => seq.serialize(serializer),
            Self::Current => serializer.serialize_str("current"),
        }
    }
}

impl<'de> Deserialize<'de> for DocumentPoint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        match JsonValue::deserialize(deserializer)? {
            JsonValue::String(text) if text == "current" => Ok(Self::Current),
            JsonValue::Number(number) => number
                .as_u64()
                .map(|value| Self::Seq(Seq(value)))
                .ok_or_else(|| serde::de::Error::custom("Invalid document point")),
            _ => Err(serde::de::Error::custom("Invalid document point")),
        }
    }
}

/// Exact logical identity of a singleton or one keyed family member.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentAddress {
    pub kind: String,
    pub scope: DocumentScope,
    /// Absent selects the singleton; present selects one family member.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// Ordered scan of document incarnations alive in one exact scope at one point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentQuery {
    pub scope: DocumentScope,
    pub at: DocumentPoint,
    pub kind: Option<String>,
}

/// Complete checkpoint or Chord operation batch selected by the owning Session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum DocumentContent {
    Base { version: u32, value: JsonObject },
    Delta { version: u32, ops: Vec<Op> },
}

impl DocumentContent {
    pub fn version(&self) -> u32 {
        match self {
            Self::Base { version, .. } | Self::Delta { version, .. } => *version,
        }
    }

    pub fn is_base(&self) -> bool {
        matches!(self, Self::Base { .. })
    }
}

/// Exact persisted source selected for a definition-free document copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentCopySource {
    pub id: DocumentId,
    pub at: DocumentPoint,
}

/// Detached materialized value and stored definition version at a selected point.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredDocument {
    pub record: DocumentRecord,
    pub version: u32,
    pub value: JsonObject,
    /// Deltas replayed after the selected base to materialize `value`.
    pub deltas_since_base: u64,
}

/// One global entry and the sequence of the commit that persisted it.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredEntry {
    pub entry: EntryRecord,
    pub commit_seq: Seq,
}

/// One record or document mutation in an atomic storage commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StorageWrite {
    #[serde(rename = "conversation")]
    Conversation { value: ConversationRecord },
    #[serde(rename = "entry")]
    Entry { value: EntryRecord },
    #[serde(rename = "task")]
    Task { value: TaskRecord },
    #[serde(rename = "submission")]
    Submission { value: SubmissionRecord },
    /// `content` is always a base.
    #[serde(rename = "document.create")]
    DocumentCreate {
        record: DocumentCreate,
        content: DocumentContent,
    },
    #[serde(rename = "document.copy")]
    DocumentCopy {
        record: DocumentCreate,
        source: DocumentCopySource,
    },
    #[serde(rename = "document.change")]
    DocumentChange {
        id: DocumentId,
        content: DocumentContent,
    },
    #[serde(rename = "document.retire")]
    DocumentRetire { id: DocumentId },
}

impl StorageWrite {
    /// The TS `type` discriminator.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Conversation { .. } => "conversation",
            Self::Entry { .. } => "entry",
            Self::Task { .. } => "task",
            Self::Submission { .. } => "submission",
            Self::DocumentCreate { .. } => "document.create",
            Self::DocumentCopy { .. } => "document.copy",
            Self::DocumentChange { .. } => "document.change",
            Self::DocumentRetire { .. } => "document.retire",
        }
    }
}

/// `Extract<DocumentCommitChange, { type: "document" }>`.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentChange {
    pub record: DocumentRecord,
    /// Conversation owning the document; task documents derive it from their task record. `None` only for Session documents.
    pub conversation_id: Option<ConversationId>,
    /// Definition version of `value`; absent when this commit retired the incarnation.
    pub version: Option<u32>,
    /// Exact adopted immutable revision, or `None` (TS `null`) when this commit retired the incarnation.
    pub value: Option<Arc<JsonValue>>,
    /// Exact adopted operations for an ordinary update; empty for creation and retirement.
    pub ops: Arc<[Op]>,
}

/// Definition-free child initialization; consumers hydrate through state or watch acquisition.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentCopyChange {
    pub record: DocumentRecord,
    pub conversation_id: ConversationId,
    pub source: DocumentCopySource,
}

/// Committed change of one document incarnation.
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentCommitChange {
    Document(DocumentChange),
    Copy(DocumentCopyChange),
}

/// One immutable change of a successful Session commit (`CommitChange`):
/// a complete table record or a document change.
#[derive(Debug, Clone, PartialEq)]
pub enum CommitChange {
    Conversation(ConversationRecord),
    Entry(EntryRecord),
    Task(TaskRecord),
    Submission(SubmissionRecord),
    Document(DocumentChange),
    DocumentCopy(DocumentCopyChange),
}

impl CommitChange {
    /// The TS `type` discriminator.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Conversation(_) => "conversation",
            Self::Entry(_) => "entry",
            Self::Task(_) => "task",
            Self::Submission(_) => "submission",
            Self::Document(_) => "document",
            Self::DocumentCopy(_) => "document.copy",
        }
    }
}

impl From<DocumentCommitChange> for CommitChange {
    fn from(change: DocumentCommitChange) -> Self {
        match change {
            DocumentCommitChange::Document(change) => Self::Document(change),
            DocumentCommitChange::Copy(change) => Self::DocumentCopy(change),
        }
    }
}

/// Every immutable change from one successful Session commit. Change order is unspecified.
#[derive(Debug, Clone, PartialEq)]
pub struct CommitPublication {
    pub seq: Seq,
    pub changes: Vec<CommitChange>,
}

/// Terminal result of one document watch.
#[derive(Debug, Clone)]
pub enum WatchEnd {
    Stopped,
    Cancelled,
    SessionClosed,
    Retired,
    ListenerError(Error),
}

impl WatchEnd {
    /// The TS `reason`.
    pub fn reason(&self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
            Self::Cancelled => "cancelled",
            Self::SessionClosed => "session_closed",
            Self::Retired => "retired",
            Self::ListenerError(_) => "listener_error",
        }
    }
}

impl PartialEq for WatchEnd {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ListenerError(left), Self::ListenerError(right)) => {
                left.to_string() == right.to_string()
            }
            _ => self.reason() == other.reason(),
        }
    }
}

/// Atomic persistence boundary for Session records.
///
/// Storage trusts the owning Session to supply semantically valid records, references,
/// ancestry, and transitions. Implementations enforce atomicity, global ID ownership,
/// immutable conversation/entry creation, document record consistency, and detached values;
/// Session serializes commits.
#[async_trait]
pub trait Storage: Send + Sync {
    /// Atomically persist one batch and return its sequence. Once resolved, later reads through this storage observe it.
    async fn commit(&self, writes: &[StorageWrite], context: &Context) -> Result<Seq>;

    /// Return a fresh candidate from the Session-global numeric ID namespace.
    async fn mint_id(&self) -> Result<u64>;

    /// Look up one conversation by exact ID.
    async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<ConversationRecord>>;

    /// Scan conversations in ascending ID order.
    async fn scan_conversations(
        &self,
        query: &ConversationQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<ConversationRecord>>;

    /// Look up one global entry and the sequence of the commit that persisted it.
    async fn entry(&self, id: EntryId, context: &Context) -> Result<Option<StoredEntry>>;

    /// Look up one entry only when it is visible through the requested conversation's ancestry
    /// (TS `entry(conversationId, id, context)`).
    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        context: &Context,
    ) -> Result<Option<StoredEntry>>;

    /// Return the newest visible entry with a `head` at or below the optional inclusive cutoff.
    /// The returned entry is the marker; its `head` value is the range's actual lower bound.
    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
        context: &Context,
    ) -> Result<Option<EntryRecord>>;

    /// Scan the inclusive visible range newest-first, returning at most `limit` entries.
    async fn scan_entries(
        &self,
        query: &EntryQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<EntryRecord>>;

    /// Look up the latest complete record for one task.
    async fn task(&self, id: TaskId, context: &Context) -> Result<Option<TaskRecord>>;

    /// Scan task records matching every supplied filter.
    async fn scan_tasks(
        &self,
        query: &TaskQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<TaskRecord>>;

    /// Look up the latest complete record for one admitted submission.
    async fn submission(
        &self,
        id: SubmissionId,
        context: &Context,
    ) -> Result<Option<SubmissionRecord>>;

    /// Scan submissions matching every supplied filter in ascending ID order.
    async fn scan_submissions(
        &self,
        query: &SubmissionQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<SubmissionRecord>>;

    /// Find a submission by its conversation-scoped host deduplication key.
    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        context: &Context,
    ) -> Result<Option<SubmissionRecord>>;

    /// Resolve the incarnation occupying one exact logical address at the selected point.
    async fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        context: &Context,
    ) -> Result<Option<DocumentRecord>>;

    /// Materialize one specific incarnation by ID at the selected point without following a replacement at its address.
    async fn document(
        &self,
        id: DocumentId,
        at: DocumentPoint,
        context: &Context,
    ) -> Result<Option<StoredDocument>>;

    /// Scan incarnations alive in one exact scope at the selected point.
    async fn scan_documents(
        &self,
        query: &DocumentQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<DocumentRecord>>;

    /// Release backend resources; all later operations must reject.
    async fn close(&self, context: &Context) -> Result<()>;
}

#[cfg(test)]
mod tests {
    //! JSON shapes of the stored records. Pi's `types.test.ts` checks the
    //! discriminated shapes at the type level (`expectTypeOf`, the
    //! `@ts-expect-error` cases have no Rust counterpart); here each shape
    //! must round-trip byte for byte, which pins the on-disk format.

    use serde::Serialize;
    use serde::de::DeserializeOwned;
    use serde_json::json;

    use super::*;

    fn round_trip<T: Serialize + DeserializeOwned>(text: &str) -> T {
        let value: T = serde_json::from_str(text).unwrap();
        assert_eq!(serde_json::to_string(&value).unwrap(), text);
        value
    }

    // Port of "encodes discriminator-dependent fields" (runtime half).
    #[test]
    fn encodes_discriminator_dependent_fields() {
        let omit: ContextEdit = round_trip(r#"{"target":2,"action":"omit"}"#);
        assert_eq!(omit.action, ContextEditAction::Omit);
        let replace: ContextEdit = round_trip(r#"{"target":2,"action":"replace","messages":[]}"#);
        assert!(matches!(replace.action, ContextEditAction::Replace { .. }));

        let pending: TaskState =
            round_trip(r#"{"status":"pending","checkpoint":{"phase":"ready"}}"#);
        assert_eq!(pending.status(), TaskStatus::Pending);
        let terminal: TaskState = round_trip(
            r#"{"status":"terminal","outcome":{"status":"completed","result":{"value":1}}}"#,
        );
        assert_eq!(
            terminal.outcome(),
            Some(&TaskOutcome::Completed {
                result: json!({ "value": 1 })
            })
        );

        let completed_input: SubmissionRecord = round_trip(
            r#"{"id":5,"conversationId":1,"type":"input","status":"done","entry":2,"answer":3}"#,
        );
        assert_eq!(completed_input.type_, SubmissionType::Input);
        let completed_write: SubmissionRecord =
            round_trip(r#"{"id":5,"conversationId":1,"type":"write","status":"done","entry":2}"#);
        assert_eq!(completed_write.answer, None);
        let queued_write: SubmissionCreate =
            round_trip(r#"{"conversationId":1,"type":"write","status":"queued"}"#);
        assert_eq!(queued_write.status, SubmissionStatus::Queued);

        let base: DocumentContent =
            round_trip(r#"{"kind":"base","version":1,"value":{"count":1}}"#);
        assert!(matches!(base, DocumentContent::Base { .. }));
        let delta: DocumentContent =
            round_trip(r#"{"kind":"delta","version":1,"ops":[["s",["count"],2]]}"#);
        assert!(matches!(delta, DocumentContent::Delta { .. }));
        let conversation_document: DocumentCreate = round_trip(
            r#"{"id":6,"kind":"test","scope":{"kind":"conversation","conversationId":1},"history":"rewindable","fork":"asOf"}"#,
        );
        assert_eq!(conversation_document.fork, Some(ForkPolicy::AsOf));
    }

    // TS writes records through `copyJson(.., { omitUndefinedProperties })`,
    // so a void task input, checkpoint or result has no key. These lines are
    // the output of chord's `copyJson` on void task records (Pi 1.0.2).
    #[test]
    fn reads_ts_task_records_that_omit_void_fields() {
        let records = [
            r#"{"id":5,"conversationId":1,"kind":"void.task","version":1,"background":false,"abortRequested":false,"state":{"status":"pending"}}"#,
            r#"{"id":5,"conversationId":1,"kind":"void.task","version":1,"background":false,"abortRequested":false,"state":{"status":"running"}}"#,
            r#"{"id":5,"conversationId":1,"kind":"void.task","version":1,"background":false,"abortRequested":false,"state":{"status":"waiting","on":[6],"policy":"failFast"}}"#,
            r#"{"id":5,"conversationId":1,"kind":"void.task","version":1,"background":false,"abortRequested":false,"state":{"status":"terminal","outcome":{"status":"completed"}}}"#,
            r#"{"id":5,"conversationId":1,"kind":"void.task","version":1,"owner":4,"background":true,"abortRequested":true,"state":{"status":"completing","outcome":{"status":"aborted"}}}"#,
        ];
        for text in records {
            let record: TaskRecord = serde_json::from_str(text).unwrap();
            assert_eq!(record.input, JsonValue::Null);
            if let Some(checkpoint) = record.state.checkpoint() {
                assert_eq!(checkpoint, &JsonValue::Null);
            }
        }
        let record: TaskRecord = serde_json::from_str(records[3]).unwrap();
        assert_eq!(
            record.state.outcome(),
            Some(&TaskOutcome::Completed {
                result: JsonValue::Null
            })
        );
        // Rust has no `undefined`: the record is written back with `null`s,
        // which TS reads as the `null` input and result its own tests use.
        assert_eq!(
            serde_json::to_string(&record).unwrap(),
            r#"{"id":5,"conversationId":1,"kind":"void.task","version":1,"input":null,"background":false,"abortRequested":false,"state":{"status":"terminal","outcome":{"status":"completed","result":null}}}"#
        );
    }
}
