//! Port of durable `src/session/transaction.ts`: the transaction handed to one
//! Session commit callback.
//!
//! Divergences from Pi:
//! - TS operations are eager promises. Every asynchronous [`Transaction`]
//!   operation here runs its synchronous checks at call time and then runs on
//!   a spawned tokio task, so an operation the callback never awaits still
//!   runs and is drained exactly like a pending TS promise. Operations return
//!   [`TxFuture`]s; a commit therefore needs a tokio runtime.
//! - `Tx` and the internal `Transaction` are one type; the internal operations
//!   (`set_task`, `create_root_conversation`, `staged_tasks`, ...) are marked.
//! - Drafts are [`DocDraft`] handles instead of proxies: typed values are read
//!   with [`DocDraft::get`] and changed with [`DocDraft::edit`]; Chord diffs the
//!   draft at preparation. A settled draft fails with "Cannot use a settled
//!   overlay". An edit that replaces the root with a non-object fails with
//!   "Document values must be JSON objects" (Rust-only: a TS proxy draft
//!   cannot replace its root).
//! - Storage writes own their JSON, so a written base is a copy of the adopted
//!   revision rather than the same object, and delta ops are a copy of the
//!   published `Arc<[Op]>`.

use std::collections::HashSet;
use std::marker::PhantomData;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::Notify;

use crate::chord::delta::{Change, Op, Prepared, Tracker, track};
use crate::chord::{Context, JsonValue, copy_json};

use crate::durable::documents::{
    AnyDocDefinition, DocAccess, ResolvedAddress, address_id, check_record_scope,
    check_record_version, document_create, from_json, materialize_document_value, resolve_address,
};
use crate::durable::entries::Entry;
use crate::durable::errors::{Error, ReadAfterWrite, Result};
use crate::durable::ids::{
    ConversationId, DocumentId, EntryId, ROOT_CONVERSATION_ID, Seq, SubmissionId, TaskId,
};
use crate::durable::tasks::Task;
use crate::durable::types::{
    CheckpointInfo, ConversationOwner, ConversationOwnership, ConversationParent,
    ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentChange,
    DocumentCommitChange, DocumentContent, DocumentCopyChange, DocumentCreate, DocumentPoint,
    DocumentQuery, DocumentRecord, DocumentScope, EntryDraft, EntryHead, EntryQuery, EntryRecord,
    ForkPolicy, Page, Storage, StorageWrite, SubmissionCreate, SubmissionRecord,
    SubmissionSettlement, SubmissionStatus, SubmissionType, TaskOptions, TaskOwnership, TaskQuery,
    TaskRecord, TaskState, TaskStatus, TypedEntryDraft,
};

use super::forks::{ForkDocumentCopy, prepare_fork_document_copies};

/// The future of one asynchronous transaction operation.
pub type TxFuture<T> = BoxFuture<'static, Result<T>>;

type SharedResult<T> = Shared<BoxFuture<'static, Result<T>>>;

/// A staged submission change: a settlement, or the placement of a queued submission at its entry.
#[derive(Debug, Clone, PartialEq)]
enum SubmissionChange {
    Settlement(SubmissionSettlement),
    Placed { entry: EntryId },
}

/// Complete record after applying one change. Placement turns a queued input `placed` and a queued write `done`; only a
/// placed input can be answered. A settled record stays (`None`).
fn apply_submission_change(
    current: &SubmissionRecord,
    change: &SubmissionChange,
) -> Result<Option<SubmissionRecord>> {
    if matches!(
        current.status,
        SubmissionStatus::Done | SubmissionStatus::Unanswered
    ) {
        return Ok(None);
    }
    let mut next = current.clone();
    match change {
        SubmissionChange::Placed { entry } => {
            if current.status != SubmissionStatus::Queued {
                return Err(Error::message(format!(
                    "Submission {} is not queued",
                    current.id
                )));
            }
            next.status = match current.type_ {
                SubmissionType::Input => SubmissionStatus::Placed,
                SubmissionType::Write => SubmissionStatus::Done,
            };
            next.entry = Some(*entry);
        }
        SubmissionChange::Settlement(SubmissionSettlement::Done { answer }) => {
            if current.status != SubmissionStatus::Placed {
                return Err(Error::message(format!(
                    "Submission {} is not a placed input",
                    current.id
                )));
            }
            next.status = SubmissionStatus::Done;
            next.answer = Some(*answer);
        }
        // Queued and placed records carry no answer, reason, or detail; an unanswered input keeps its entry.
        SubmissionChange::Settlement(SubmissionSettlement::Unanswered { reason, detail }) => {
            next.status = SubmissionStatus::Unanswered;
            next.reason = Some(reason.clone());
            if detail.is_some() {
                next.detail = detail.clone();
            }
        }
    }
    Ok(Some(next))
}

const INTERNAL_SCAN_PAGE_SIZE: usize = 256;

fn empty_operations() -> Arc<[Op]> {
    Arc::from(Vec::new())
}

/// One committed document incarnation owned by the Session tracker cache.
pub(crate) struct LoadedDocument {
    pub address_id: String,
    pub record: DocumentRecord,
    /// Persisted definition version; older while the tracked value is migrated only in memory.
    pub stored_version: u32,
    /// Definition version whose shape the tracked value has; access with another version reloads from Storage.
    pub value_version: u32,
    /// Stored deltas after the newest base; advanced by adoption so the next predicate call needs no read.
    pub deltas_since_base: u64,
    pub tracker: Tracker,
}

/// Shared handle of one cached incarnation.
pub(crate) type LoadedRef = Arc<Mutex<LoadedDocument>>;

/// Session services used by a transaction while it holds the mutation line.
pub(crate) trait TransactionHost: Send + Sync {
    fn storage(&self) -> Arc<dyn Storage>;
    /// Return the cached current incarnation without loading.
    fn cached(&self, address_id: &str) -> Option<LoadedRef>;
    /// Return the cached current incarnation, cold-loading and migrating it when necessary.
    fn load(
        &self,
        definition: Arc<AnyDocDefinition>,
        address_id: String,
        address: DocumentAddress,
        context: Context,
    ) -> BoxFuture<'static, Result<Option<LoadedRef>>>;
    /// Install a newly committed incarnation.
    fn install(&self, document: LoadedDocument);
    /// Remove a retired incarnation if it is still the cached occupant of its address.
    fn evict(&self, address_id: &str, record_id: DocumentId);
    /// Stage writes that belong to every newly created or forked conversation, in its creating transaction.
    fn conversation_created(
        &self,
        tx: &Transaction,
        record: &ConversationRecord,
    ) -> BoxFuture<'static, Result<()>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskWriteKind {
    Create,
    Replace,
}

#[derive(Debug, Clone)]
struct TaskWrite {
    kind: TaskWriteKind,
    record: TaskRecord,
}

/// Committed and candidate state for one task touched by this transaction.
#[derive(Default)]
struct TransactionTask {
    committed_read: Option<SharedResult<Option<TaskRecord>>>,
    write: Option<TaskWrite>,
    publication_conversation_id: Option<ConversationId>,
}

/// Defaults a commit binds to: `tx.create_task()` conversation and the task attributed to appended entries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransactionScope {
    pub conversation_id: Option<ConversationId>,
    /// Task whose runtime commit this is; stamped as `by_task_id` on appended entries.
    pub task_id: Option<TaskId>,
}

/// Storage/cache provenance of one staged document incarnation.
enum DocumentTarget {
    Loaded(LoadedRef),
    Created {
        record: DocumentCreate,
        version: u32,
        tracker: Tracker,
    },
    ForkCopy(ForkDocumentCopy),
    RetireOnly(DocumentRecord),
}

/// One document incarnation acquired, created, or retired by this transaction.
struct DocumentEntry {
    address_id: String,
    address: DocumentAddress,
    /// Absent for definition-free fork copies and retirement entries discovered by a terminal-task scan.
    definition: Option<Arc<AnyDocDefinition>>,
    /// Memoized public acquisition; absent for metadata-only retirement.
    draft: Option<SharedResult<()>>,
    /// Set after acquisition or retirement lookup finds the affected incarnation.
    target: Option<DocumentTarget>,
    change: Option<Change>,
    prepared: Option<Prepared>,
    retire_on_commit: bool,
}

/// `record` of a plan: committed for an existing incarnation, a creation for a new one.
#[derive(Debug, Clone)]
enum PlanRecord {
    Create(DocumentCreate),
    Committed(DocumentRecord),
}

impl PlanRecord {
    fn id(&self) -> DocumentId {
        match self {
            Self::Create(record) => record.id,
            Self::Committed(record) => record.id,
        }
    }

    fn scope(&self) -> DocumentScope {
        match self {
            Self::Create(record) => record.scope,
            Self::Committed(record) => record.scope,
        }
    }

    fn fork(&self) -> Option<ForkPolicy> {
        match self {
            Self::Create(record) => record.fork,
            Self::Committed(record) => record.fork,
        }
    }
}

enum PlanTracker {
    /// The cached incarnation this change updates.
    Loaded(LoadedRef),
    /// A new incarnation's tracker, installed on adoption.
    Created(Tracker),
}

/// Prepared change of a tracked incarnation.
struct PlanChange {
    tracker: PlanTracker,
    prepared: Prepared,
    version: u32,
    definition: Option<Arc<AnyDocDefinition>>,
}

impl PlanChange {
    fn loaded(&self) -> Option<&LoadedRef> {
        match &self.tracker {
            PlanTracker::Loaded(loaded) => Some(loaded),
            PlanTracker::Created(_) => None,
        }
    }
}

/// What one staged incarnation writes and publishes, decided once before Storage admission so adoption only applies it.
struct DocumentPlan {
    address_id: String,
    record: PlanRecord,
    retire: bool,
    /// Creation, copy, or change content; absent when only retirement is written.
    content: Option<StorageWrite>,
    /// Prepared change of a tracked incarnation; absent for fork copies and retirement-only entries.
    change: Option<PlanChange>,
    /// Resolved before Storage admission so adoption performs no reads.
    conversation_id: Option<ConversationId>,
}

#[derive(Default)]
struct TxState {
    sealed: bool,
    has_table_write: bool,
    /// Atomic batch; conversation and entry writes stage eagerly, while task and document writes assemble later.
    writes: Vec<StorageWrite>,
    created_conversation_ids: HashSet<ConversationId>,
    fork_source_conversation_ids: HashSet<ConversationId>,
    fork_source_document_ids: HashSet<DocumentId>,
    /// One entry per task touched by a public read, candidate write, or document-owner lookup.
    tasks_by_id: IndexMap<TaskId, TransactionTask>,
    /// Submissions created by this transaction, by ID.
    submissions: IndexMap<SubmissionId, SubmissionRecord>,
    /// Submission settlements and placements in staging order.
    submission_changes: Vec<(SubmissionId, SubmissionChange)>,
    /// Write and publication plans of every staged incarnation, built during assembly.
    plans: Vec<DocumentPlan>,
    /// Every document acquisition or retirement marker in staging order.
    documents: Vec<DocumentEntry>,
    /// Latest transaction-local incarnation or retirement marker at each logical address.
    latest_document_by_address: std::collections::HashMap<String, usize>,
}

fn sealed_error() -> Error {
    Error::message("Transaction has settled")
}

impl TxState {
    fn assert_open(&self) -> Result<()> {
        if self.sealed {
            return Err(sealed_error());
        }
        Ok(())
    }

    fn task_entry(&mut self, id: TaskId) -> &mut TransactionTask {
        self.tasks_by_id.entry(id).or_default()
    }

    fn abort_changes(&mut self) {
        for document in &mut self.documents {
            if let Some(change) = &mut document.change {
                change.abort();
            }
        }
    }

    fn assert_task_documents_open(&self, resolved: &ResolvedAddress) -> Result<()> {
        if let DocumentScope::Task { task_id } = resolved.address.scope
            && self
                .tasks_by_id
                .get(&task_id)
                .and_then(|task| task.write.as_ref())
                .is_some_and(|write| write.record.state.status() == TaskStatus::Terminal)
        {
            return Err(Error::message(format!("Task {task_id} is terminal")));
        }
        Ok(())
    }
}

struct TxInner {
    host: Arc<dyn TransactionHost>,
    storage: Arc<dyn Storage>,
    context: Context,
    scope: TransactionScope,
    pending_operations: AtomicUsize,
    drained: Notify,
    state: Mutex<TxState>,
}

/// Transaction for one Session commit callback (`Tx` and the internal `Transaction`).
///
/// Every asynchronous operation is tracked so callback settlement can reject and drain unfinished work. Session calls
/// one settlement method, then either discards prepared changes or adopts them once after Storage succeeds.
#[derive(Clone)]
pub struct Transaction {
    inner: Arc<TxInner>,
}

/// `Tx`: the public transaction surface.
pub type Tx = Transaction;

struct PendingGuard(Arc<TxInner>);

impl Drop for PendingGuard {
    fn drop(&mut self) {
        if self.0.pending_operations.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.0.drained.notify_waiters();
        }
    }
}

/// Await a spawned job, resuming its panic on the caller.
pub(crate) async fn join<T>(handle: tokio::task::JoinHandle<T>) -> T {
    match handle.await {
        Ok(value) => value,
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => panic!("durable task was cancelled: {error}"),
    }
}

fn ready<T: Send + 'static>(result: Result<T>) -> TxFuture<T> {
    Box::pin(futures::future::ready(result))
}

impl Transaction {
    pub(crate) fn new(
        host: Arc<dyn TransactionHost>,
        context: Context,
        scope: TransactionScope,
    ) -> Self {
        let storage = host.storage();
        Self {
            inner: Arc::new(TxInner {
                host,
                storage,
                context,
                scope,
                pending_operations: AtomicUsize::new(0),
                drained: Notify::new(),
                state: Mutex::new(TxState::default()),
            }),
        }
    }

    /// The commit's context.
    pub fn context(&self) -> &Context {
        &self.inner.context
    }

    /// The defaults this commit binds to.
    pub fn scope(&self) -> TransactionScope {
        self.inner.scope
    }

    /// Whether two handles are the same transaction.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn assert_open(&self) -> Result<()> {
        self.inner.state.lock().assert_open()
    }

    // ─── Table reads ────────────────────────────────────────────────────────

    pub fn conversation(&self, id: ConversationId) -> TxFuture<Option<ConversationRecord>> {
        let storage = self.inner.storage.clone();
        let context = self.inner.context.clone();
        self.read("conversation", async move {
            storage.conversation(id, &context).await
        })
    }

    pub fn entry(&self, id: EntryId) -> TxFuture<Option<EntryRecord>> {
        self.entry_with_kind(None, id)
    }

    /// `entry(token, id)`: the entry only when it has the token's kind. Read its data with `token.data()`.
    pub fn entry_of<D>(&self, token: &Entry<D>, id: EntryId) -> TxFuture<Option<EntryRecord>> {
        self.entry_with_kind(Some(token.kind().to_string()), id)
    }

    fn entry_with_kind(&self, kind: Option<String>, id: EntryId) -> TxFuture<Option<EntryRecord>> {
        let storage = self.inner.storage.clone();
        let context = self.inner.context.clone();
        self.read("entry", async move {
            let entry = storage
                .entry(id, &context)
                .await?
                .map(|stored| stored.entry);
            Ok(match kind {
                None => entry,
                Some(kind) => entry.filter(|entry| entry.kind == kind),
            })
        })
    }

    pub fn task(&self, id: TaskId) -> TxFuture<Option<TaskRecord>> {
        let tx = self.clone();
        self.read("task", async move { tx.committed_task(id).await })
    }

    pub fn scan_conversations(
        &self,
        query: ConversationQuery,
        limit: usize,
        cursor: Option<Cursor>,
    ) -> TxFuture<Page<ConversationRecord>> {
        let storage = self.inner.storage.clone();
        let context = self.inner.context.clone();
        self.read("scanConversations", async move {
            storage
                .scan_conversations(&query, limit, cursor.as_ref(), &context)
                .await
        })
    }

    pub fn scan_entries(
        &self,
        query: EntryQuery,
        limit: usize,
        cursor: Option<Cursor>,
    ) -> TxFuture<Page<EntryRecord>> {
        let storage = self.inner.storage.clone();
        let context = self.inner.context.clone();
        self.read("scanEntries", async move {
            storage
                .scan_entries(&query, limit, cursor.as_ref(), &context)
                .await
        })
    }

    pub fn latest_head_marker(
        &self,
        conversation_id: ConversationId,
    ) -> TxFuture<Option<EntryRecord>> {
        let storage = self.inner.storage.clone();
        let context = self.inner.context.clone();
        self.read("latestHeadMarker", async move {
            storage
                .find_latest_head_marker(conversation_id, None, &context)
                .await
        })
    }

    pub fn scan_tasks(
        &self,
        query: TaskQuery,
        limit: usize,
        cursor: Option<Cursor>,
    ) -> TxFuture<Page<TaskRecord>> {
        let storage = self.inner.storage.clone();
        let context = self.inner.context.clone();
        self.read("scanTasks", async move {
            storage
                .scan_tasks(&query, limit, cursor.as_ref(), &context)
                .await
        })
    }

    /// Internal: committed submission record.
    pub fn submission(&self, id: SubmissionId) -> TxFuture<Option<SubmissionRecord>> {
        let storage = self.inner.storage.clone();
        let context = self.inner.context.clone();
        self.read("submission", async move {
            storage.submission(id, &context).await
        })
    }

    /// Committed submission with a conversation-scoped request ID.
    pub fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: impl Into<String>,
    ) -> TxFuture<Option<SubmissionRecord>> {
        let storage = self.inner.storage.clone();
        let context = self.inner.context.clone();
        let request_id = request_id.into();
        self.read("submissionByRequest", async move {
            storage
                .submission_by_request(conversation_id, &request_id, &context)
                .await
        })
    }

    // ─── Table writes ───────────────────────────────────────────────────────

    pub fn create_conversation(
        &self,
        ownership: ConversationOwnership,
    ) -> TxFuture<ConversationRecord> {
        let tx = self.clone();
        self.write(async move { tx.stage_conversation(None, ownership, None).await })
    }

    /// Internal final-form bootstrap path for the reserved root identity.
    pub fn create_root_conversation(&self) -> TxFuture<ConversationRecord> {
        let tx = self.clone();
        self.write(async move {
            tx.stage_conversation(
                None,
                ConversationOwnership::Ownerless,
                Some(ROOT_CONVERSATION_ID),
            )
            .await
        })
    }

    pub fn fork_conversation(
        &self,
        parent_conversation_id: ConversationId,
        at: EntryId,
        ownership: ConversationOwnership,
    ) -> TxFuture<ConversationRecord> {
        let tx = self.clone();
        self.write(async move {
            tx.stage_conversation(
                Some(ConversationParent {
                    conversation_id: parent_conversation_id,
                    at,
                }),
                ownership,
                None,
            )
            .await
        })
    }

    async fn stage_conversation(
        &self,
        parent: Option<ConversationParent>,
        ownership: ConversationOwnership,
        reserved_id: Option<ConversationId>,
    ) -> Result<ConversationRecord> {
        let owner_task_id = match ownership {
            ConversationOwnership::Task { task_id } => Some(task_id),
            ConversationOwnership::Ownerless => None,
        };
        let id = match reserved_id {
            Some(id) => id,
            None => ConversationId(self.inner.storage.mint_id().await?),
        };
        self.assert_open()?;
        let mut owner = None;
        if let Some(task_id) = owner_task_id {
            let task = self.current_task(task_id).await?;
            self.assert_open()?;
            let Some(task) = task else {
                return Err(Error::message(format!(
                    "Conversation owner task {task_id} does not exist"
                )));
            };
            owner = Some(ConversationOwner {
                conversation_id: task.conversation_id,
                task_id,
            });
        }
        let record = ConversationRecord { id, parent, owner };
        let copies = match parent {
            None => Vec::new(),
            Some(parent) => {
                prepare_fork_document_copies(
                    &*self.inner.storage,
                    parent.conversation_id,
                    parent.at,
                    id,
                    &self.inner.context,
                )
                .await?
            }
        };
        {
            let mut state = self.inner.state.lock();
            state.assert_open()?;
            for copy in copies {
                state.fork_source_document_ids.insert(copy.source.id);
                let address = copy.record.address();
                let entry = DocumentEntry {
                    address_id: address_id(&address),
                    address,
                    definition: None,
                    draft: None,
                    target: Some(DocumentTarget::ForkCopy(copy)),
                    change: None,
                    prepared: None,
                    retire_on_commit: false,
                };
                let index = state.documents.len();
                state
                    .latest_document_by_address
                    .insert(entry.address_id.clone(), index);
                state.documents.push(entry);
            }
            if let Some(parent) = parent {
                state
                    .fork_source_conversation_ids
                    .insert(parent.conversation_id);
            }
            state.created_conversation_ids.insert(id);
            state.writes.push(StorageWrite::Conversation {
                value: record.clone(),
            });
        }
        self.inner.host.conversation_created(self, &record).await?;
        self.assert_open()?;
        Ok(record)
    }

    pub fn append_entry(
        &self,
        conversation_id: ConversationId,
        value: EntryDraft,
    ) -> TxFuture<EntryRecord> {
        let tx = self.clone();
        self.write(async move { tx.stage_entry(conversation_id, value).await })
    }

    /// `appendEntry(token, conversationId, value)`: the token supplies the kind and types the data.
    pub fn append_entry_of<D: Serialize>(
        &self,
        token: &Entry<D>,
        conversation_id: ConversationId,
        value: TypedEntryDraft<D>,
    ) -> TxFuture<EntryRecord> {
        let data = value
            .data
            .as_ref()
            .map(|data| copy_json(data, None))
            .transpose()
            .map_err(Error::from);
        let draft = data.map(|data| EntryDraft {
            kind: token.kind().to_string(),
            model: value.model,
            data,
            head: value.head,
            edits: value.edits,
        });
        let tx = self.clone();
        self.write(async move { tx.stage_entry(conversation_id, draft?).await })
    }

    async fn stage_entry(
        &self,
        conversation_id: ConversationId,
        value: EntryDraft,
    ) -> Result<EntryRecord> {
        self.require_conversation(conversation_id).await?;
        self.assert_open()?;
        let id = EntryId(self.inner.storage.mint_id().await?);
        let mut state = self.inner.state.lock();
        state.assert_open()?;
        let record = EntryRecord {
            id,
            conversation_id,
            kind: value.kind,
            model: value.model,
            data: value.data,
            head: value.head.map(|head| match head {
                EntryHead::SelfEntry => id,
                EntryHead::Id(head) => head,
            }),
            edits: value.edits,
            by_task_id: self.inner.scope.task_id,
        };
        state.writes.push(StorageWrite::Entry {
            value: record.clone(),
        });
        Ok(record)
    }

    pub fn create_task<I, S, R, H>(
        &self,
        task: &Task<I, S, R, H>,
        input: I,
        options: TaskOptions,
    ) -> TxFuture<TaskId<R>>
    where
        I: Serialize + Send + 'static,
        S: Serialize + 'static,
        R: 'static,
    {
        let definition = &task.definition;
        let initial = definition.initial.clone();
        let name = definition.name.clone();
        let version = definition.version;
        let tx = self.clone();
        self.write(async move {
            let mut owner = None;
            if let TaskOwnership::Task { task_id } = options.ownership {
                // Validated again against the owner's final candidate during assembly.
                let found = tx.current_task(task_id).await?;
                tx.assert_open()?;
                let Some(found) = found else {
                    return Err(Error::message(format!(
                        "Task owner {task_id} does not exist"
                    )));
                };
                if options.background == Some(true) {
                    return Err(Error::type_error("A child task cannot be background"));
                }
                if options
                    .conversation_id
                    .is_some_and(|conversation_id| conversation_id != found.conversation_id)
                {
                    return Err(Error::message(format!(
                        "A child task lives in its owner's conversation {}",
                        found.conversation_id
                    )));
                }
                owner = Some(found);
            }
            let Some(conversation_id) = owner
                .as_ref()
                .map(|owner| owner.conversation_id)
                .or(options.conversation_id)
                .or(tx.inner.scope.conversation_id)
            else {
                return Err(Error::type_error(
                    "Tx.createTask() requires options.conversationId",
                ));
            };
            tx.require_conversation(conversation_id).await?;
            tx.assert_open()?;
            let checkpoint = copy_json(&initial(&input), None)?;
            let id = TaskId::new(tx.inner.storage.mint_id().await?);
            tx.assert_open()?;
            let record = TaskRecord {
                id,
                conversation_id,
                kind: name,
                version,
                input: copy_json(&input, None)?,
                owner: owner.map(|owner| owner.id),
                background: options.background.unwrap_or(false),
                abort_requested: false,
                state: TaskState::Pending { checkpoint },
                memos: None,
            };
            let mut state = tx.inner.state.lock();
            state.assert_open()?;
            state.tasks_by_id.insert(
                id,
                TransactionTask {
                    write: Some(TaskWrite {
                        kind: TaskWriteKind::Create,
                        record,
                    }),
                    ..TransactionTask::default()
                },
            );
            Ok(id.cast())
        })
    }

    /// Create a raw submission record with a fresh ID; no admission rules apply.
    pub fn create_submission(&self, create: SubmissionCreate) -> TxFuture<SubmissionRecord> {
        let tx = self.clone();
        self.write(async move {
            tx.require_conversation(create.conversation_id).await?;
            tx.assert_open()?;
            let id = SubmissionId(tx.inner.storage.mint_id().await?);
            let mut state = tx.inner.state.lock();
            state.assert_open()?;
            let record = create.with_id(id);
            state.submissions.insert(id, record.clone());
            Ok(record)
        })
    }

    /// Settle a submission. Resolved during assembly against the transaction's latest candidate record, falling back to
    /// committed state, so it is not a caller table read and works after the first table write.
    pub fn settle_submission(
        &self,
        id: SubmissionId,
        settlement: SubmissionSettlement,
    ) -> Result<()> {
        let mut state = self.inner.state.lock();
        state.assert_open()?;
        state.has_table_write = true;
        state
            .submission_changes
            .push((id, SubmissionChange::Settlement(settlement)));
        Ok(())
    }

    /// Place a queued submission at `entry`; resolved during assembly like `settle_submission()`.
    pub fn place_submission(&self, id: SubmissionId, entry: EntryId) -> Result<()> {
        let mut state = self.inner.state.lock();
        state.assert_open()?;
        state.has_table_write = true;
        state
            .submission_changes
            .push((id, SubmissionChange::Placed { entry }));
        Ok(())
    }

    /// Internal: replace one task record completely. Tasks change their own state through their runtime.
    pub fn set_task(&self, value: TaskRecord) -> Result<()> {
        let mut state = self.inner.state.lock();
        state.assert_open()?;
        state.has_table_write = true;
        let task = state.task_entry(value.id);
        if let Some(candidate) = &task.write {
            if candidate.record.state.status() == TaskStatus::Terminal {
                return Err(Error::message(format!(
                    "Task {} already has a terminal candidate",
                    value.id
                )));
            }
            if candidate.record.conversation_id != value.conversation_id {
                return Err(Error::message(format!(
                    "Task {} cannot change conversations",
                    value.id
                )));
            }
        }
        let kind = match &task.write {
            Some(write) if write.kind == TaskWriteKind::Create => TaskWriteKind::Create,
            _ => TaskWriteKind::Replace,
        };
        task.write = Some(TaskWrite {
            kind,
            record: value,
        });
        Ok(())
    }

    /// Internal: candidate records of the tasks this transaction created or replaced so far.
    pub fn staged_tasks(&self) -> Vec<TaskRecord> {
        self.inner
            .state
            .lock()
            .tasks_by_id
            .values()
            .filter_map(|task| task.write.as_ref().map(|write| write.record.clone()))
            .collect()
    }

    /// Internal: conversations this transaction created or forked so far.
    pub fn staged_conversations(&self) -> Vec<ConversationRecord> {
        self.inner
            .state
            .lock()
            .writes
            .iter()
            .filter_map(|write| match write {
                StorageWrite::Conversation { value } => Some(value.clone()),
                _ => None,
            })
            .collect()
    }

    // ─── Documents ──────────────────────────────────────────────────────────

    /// Acquire a document draft, creating the document on first access.
    pub fn doc<A: DocAccess>(&self, token: &A, args: A::Args) -> TxFuture<DocDraft<A::Value>> {
        let acquired = token.acquire_args(args).and_then(|(args, seed)| {
            let definition = token.definition().clone();
            let resolved = resolve_address(&definition, &args)?;
            self.acquire_document(definition, resolved, seed)
        });
        match acquired {
            Err(error) => ready(Err(error)),
            Ok((draft, index)) => {
                let tx = self.clone();
                Box::pin(async move {
                    draft.await?;
                    Ok(DocDraft {
                        tx,
                        index,
                        _value: PhantomData,
                    })
                })
            }
        }
    }

    /// Erased `doc()`: the memoized acquisition of one address and its entry index.
    fn acquire_document(
        &self,
        definition: Arc<AnyDocDefinition>,
        resolved: ResolvedAddress,
        seed: Option<JsonValue>,
    ) -> Result<(SharedResult<()>, usize)> {
        let mut state = self.inner.state.lock();
        state.assert_open()?;
        state.assert_task_documents_open(&resolved)?;
        let latest = state.latest_document_by_address.get(&resolved.id).copied();
        if let Some(index) = latest {
            let entry = &state.documents[index];
            if !entry.retire_on_commit {
                if let Some(draft) = &entry.draft {
                    return Ok((draft.clone(), index));
                }
                if let Some(DocumentTarget::ForkCopy(copy)) = &entry.target {
                    let copy = copy.clone();
                    let draft = self
                        .track(acquire_fork_copy(self.clone(), index, definition, copy))
                        .shared();
                    state.documents[index].draft = Some(draft.clone());
                    return Ok((draft, index));
                }
            }
        }
        let skip_load = latest.is_some_and(|index| state.documents[index].retire_on_commit);
        let seed = if definition.family { seed } else { None };
        let index = state.documents.len();
        state.documents.push(DocumentEntry {
            address_id: resolved.id.clone(),
            address: resolved.address,
            definition: Some(definition),
            draft: None,
            target: None,
            change: None,
            prepared: None,
            retire_on_commit: false,
        });
        state.latest_document_by_address.insert(resolved.id, index);
        // Capture retirement before awaiting so a pending old acquisition and its replacement stay distinct.
        let draft = self
            .track(acquire(self.clone(), index, seed, skip_load))
            .shared();
        state.documents[index].draft = Some(draft.clone());
        Ok((draft, index))
    }

    /// Retire the current incarnation at an address when this transaction commits.
    pub fn retire_doc<A: DocAccess>(&self, token: &A, address: A::Address) -> TxFuture<()> {
        let definition = token.definition().clone();
        let resolved = match resolve_address(&definition, &token.address_args(address)) {
            Ok(resolved) => resolved,
            Err(error) => return ready(Err(error)),
        };
        let mut state = self.inner.state.lock();
        if let Err(error) = state.assert_open() {
            return ready(Err(error));
        }
        let latest = state.latest_document_by_address.get(&resolved.id).copied();
        if let Some(index) = latest {
            let entry = &mut state.documents[index];
            if entry.retire_on_commit {
                return ready(Ok(()));
            }
            if let Some(DocumentTarget::ForkCopy(copy)) = &entry.target {
                if let Err(error) = check_record_scope(&definition, &copy.record) {
                    return ready(Err(error));
                }
                entry.retire_on_commit = true;
                return ready(Ok(()));
            }
            if let Some(draft) = entry.draft.clone() {
                // Retirement of an acquired draft persists its final content before retirement.
                entry.retire_on_commit = true;
                return self.track(draft);
            }
        }
        let index = state.documents.len();
        state.documents.push(DocumentEntry {
            address_id: resolved.id.clone(),
            address: resolved.address,
            definition: Some(definition),
            draft: None,
            target: None,
            change: None,
            prepared: None,
            retire_on_commit: true,
        });
        state.latest_document_by_address.insert(resolved.id, index);
        self.track(find_retirement(self.clone(), index))
    }

    // ─── Settlement ─────────────────────────────────────────────────────────

    async fn pending_drained(&self) {
        loop {
            let notified = self.inner.drained.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.inner.pending_operations.load(Ordering::SeqCst) == 0 {
                return;
            }
            notified.await;
        }
    }

    /// Seal after callback failure: abort every change and observe every pending operation.
    pub(crate) async fn settle_failure(&self) {
        {
            let mut state = self.inner.state.lock();
            state.sealed = true;
            state.abort_changes();
        }
        self.pending_drained().await;
    }

    /// Seal after callback success, prepare every open change, and assemble the atomic batch.
    /// Any failure aborts every change before Storage admission.
    pub(crate) async fn settle_success(&self) -> Result<Vec<StorageWrite>> {
        let pending = {
            let mut state = self.inner.state.lock();
            state.sealed = true;
            let pending = self.inner.pending_operations.load(Ordering::SeqCst) > 0;
            if pending {
                state.abort_changes();
            }
            pending
        };
        if pending {
            self.pending_drained().await;
            return Err(Error::message(
                "Session commit callback settled before its pending Tx operations",
            ));
        }
        let prepared = {
            let mut state = self.inner.state.lock();
            // Synchronously prepare every open change; this revokes every draft.
            let mut result = Ok(());
            for document in &mut state.documents {
                if let Some(change) = &mut document.change {
                    match change.prepare() {
                        Ok(prepared) => document.prepared = Some(prepared),
                        Err(error) => {
                            result = Err(Error::from(error));
                            break;
                        }
                    }
                }
            }
            result
        };
        let assembled = match prepared {
            Ok(()) => self.assemble().await,
            Err(error) => Err(error),
        };
        if assembled.is_err() {
            self.inner.state.lock().abort_changes();
        }
        assembled
    }

    /// Abort every prepared change after Storage failure or when no write is required.
    pub(crate) fn discard(&self) {
        self.inner.state.lock().abort_changes();
    }

    /// Adopt every prepared change by pointer swap after Storage success and describe the publication.
    pub(crate) fn adopt(&self, seq: Seq) -> Result<Vec<DocumentCommitChange>> {
        let plans = std::mem::take(&mut self.inner.state.lock().plans);
        let mut publications = Vec::new();
        for plan in plans {
            let committed = matches!(plan.record, PlanRecord::Committed(_));
            let mut record = match &plan.record {
                PlanRecord::Committed(record) => record.clone(),
                PlanRecord::Create(record) => record.stamp(seq),
            };
            if plan.retire {
                record.retired_at = Some(seq);
            }
            let publishes = publishes(&plan);
            let is_change_write = matches!(plan.content, Some(StorageWrite::DocumentChange { .. }));
            let is_base = matches!(
                plan.content,
                Some(StorageWrite::DocumentChange {
                    content: DocumentContent::Base { .. },
                    ..
                })
            );
            let mut published_change = None;
            if let Some(change) = plan.change {
                let PlanChange {
                    tracker,
                    prepared,
                    version,
                    ..
                } = change;
                match tracker {
                    PlanTracker::Loaded(loaded) => {
                        let mut loaded = loaded.lock();
                        // A loaded incarnation is adopted only when it changed.
                        if !prepared.ops().is_empty() {
                            loaded.tracker.adopt(&prepared)?;
                        } else {
                            prepared.abort();
                        }
                        if loaded.stored_version < version {
                            loaded.stored_version = version;
                        }
                        if is_change_write {
                            if is_base {
                                loaded.deltas_since_base = 0;
                            } else {
                                loaded.deltas_since_base += 1;
                            }
                        }
                        published_change =
                            Some((version, prepared.value().clone(), prepared.ops().clone()));
                    }
                    PlanTracker::Created(mut tracker) => {
                        // A new incarnation is adopted unless it retires in the same commit.
                        if !plan.retire {
                            tracker.adopt(&prepared)?;
                            self.inner.host.install(LoadedDocument {
                                address_id: plan.address_id.clone(),
                                record: record.clone(),
                                stored_version: version,
                                value_version: version,
                                deltas_since_base: 0,
                                tracker,
                            });
                        } else {
                            prepared.abort();
                        }
                        published_change =
                            Some((version, prepared.value().clone(), empty_operations()));
                    }
                }
            }
            let conversation_id = plan.conversation_id;
            if plan.retire {
                if committed {
                    self.inner.host.evict(&plan.address_id, record.id);
                }
                publications.push(DocumentCommitChange::Document(DocumentChange {
                    record,
                    conversation_id,
                    version: None,
                    value: None,
                    ops: empty_operations(),
                }));
            } else if let Some(StorageWrite::DocumentCopy { source, .. }) = plan.content {
                publications.push(DocumentCommitChange::Copy(DocumentCopyChange {
                    record,
                    conversation_id: conversation_id.expect("a copy belongs to a conversation"),
                    source,
                }));
            } else if let Some((version, value, ops)) = published_change
                && publishes
            {
                publications.push(DocumentCommitChange::Document(DocumentChange {
                    record,
                    conversation_id,
                    version: Some(version),
                    value: Some(value),
                    ops,
                }));
            }
        }
        Ok(publications)
    }

    async fn assemble(&self) -> Result<Vec<StorageWrite>> {
        let storage = self.inner.storage.clone();
        let context = self.inner.context.clone();
        let mut plans: Vec<DocumentPlan> = {
            let mut state = self.inner.state.lock();
            state
                .documents
                .iter_mut()
                .filter_map(plan_document)
                .collect()
        };
        self.reject_fork_source_writes(&plans)?;
        self.validate_owners().await?;
        let replaced: Vec<(TaskId, ConversationId)> = {
            let state = self.inner.state.lock();
            state
                .tasks_by_id
                .iter()
                .filter_map(|(id, task)| match &task.write {
                    Some(write) if write.kind == TaskWriteKind::Replace => {
                        Some((*id, write.record.conversation_id))
                    }
                    _ => None,
                })
                .collect()
        };
        for (id, conversation_id) in replaced {
            let Some(committed) = self.committed_task(id).await? else {
                return Err(Error::message(format!("Task {id} does not exist")));
            };
            if committed.state.status() == TaskStatus::Terminal {
                return Err(Error::message(format!("Task {id} is already terminal")));
            }
            if committed.conversation_id != conversation_id {
                return Err(Error::message(format!(
                    "Task {id} cannot change conversations"
                )));
            }
        }

        // Terminal settlement retires every task document, including ones created by this transaction.
        let terminal_tasks: Vec<(TaskId, bool)> = {
            let state = self.inner.state.lock();
            state
                .tasks_by_id
                .values()
                .filter_map(|task| task.write.as_ref())
                .filter(|write| write.record.state.status() == TaskStatus::Terminal)
                .map(|write| (write.record.id, write.kind == TaskWriteKind::Create))
                .collect()
        };
        if !terminal_tasks.is_empty() {
            let mut retiring = HashSet::new();
            for plan in &mut plans {
                if let DocumentScope::Task { task_id } = plan.record.scope()
                    && terminal_tasks.iter().any(|(id, _)| *id == task_id)
                {
                    plan.retire = true;
                    retiring.insert(plan.record.id());
                }
            }
            for (task_id, created) in &terminal_tasks {
                if *created {
                    continue;
                }
                let query = DocumentQuery {
                    scope: DocumentScope::Task { task_id: *task_id },
                    at: DocumentPoint::Current,
                    kind: None,
                };
                let mut cursor: Option<Cursor> = None;
                loop {
                    let page = storage
                        .scan_documents(&query, INTERNAL_SCAN_PAGE_SIZE, cursor.as_ref(), &context)
                        .await?;
                    for record in page.items {
                        if !retiring.insert(record.id) {
                            continue;
                        }
                        plans.push(DocumentPlan {
                            address_id: address_id(&record.address()),
                            record: PlanRecord::Committed(record),
                            retire: true,
                            content: None,
                            change: None,
                            conversation_id: None,
                        });
                    }
                    cursor = page.next;
                    if cursor.is_none() {
                        break;
                    }
                }
            }
        }

        // Resolve publication ownership before Storage admission so adoption remains synchronous.
        for plan in &mut plans {
            if !publishes(plan) {
                continue;
            }
            match plan.record.scope() {
                DocumentScope::Conversation { conversation_id } => {
                    plan.conversation_id = Some(conversation_id);
                }
                DocumentScope::Task { task_id } => {
                    let known = self
                        .inner
                        .state
                        .lock()
                        .task_entry(task_id)
                        .publication_conversation_id;
                    let resolved = match known {
                        Some(conversation_id) => Some(conversation_id),
                        None => {
                            let current = self.current_task(task_id).await?;
                            let conversation_id = current.map(|task| task.conversation_id);
                            if conversation_id.is_some() {
                                self.inner
                                    .state
                                    .lock()
                                    .task_entry(task_id)
                                    .publication_conversation_id = conversation_id;
                            }
                            conversation_id
                        }
                    };
                    plan.conversation_id = resolved;
                }
                DocumentScope::Session => {}
            }
        }

        let submission_changes = self.inner.state.lock().submission_changes.clone();
        for (id, change) in submission_changes {
            let staged = self.inner.state.lock().submissions.get(&id).cloned();
            let current = match staged {
                Some(current) => Some(current),
                None => storage.submission(id, &context).await?,
            };
            let Some(current) = current else {
                return Err(Error::message(format!("Submission {id} does not exist")));
            };
            if let Some(next) = apply_submission_change(&current, &change)? {
                self.inner.state.lock().submissions.insert(id, next);
            }
        }

        let mut state = self.inner.state.lock();
        let mut writes = std::mem::take(&mut state.writes);
        for value in state.submissions.values() {
            writes.push(StorageWrite::Submission {
                value: value.clone(),
            });
        }
        for task in state.tasks_by_id.values() {
            if let Some(write) = &task.write {
                writes.push(StorageWrite::Task {
                    value: write.record.clone(),
                });
            }
        }
        drop(state);
        for plan in &mut plans {
            // Checkpoint predicates run last, after every validation.
            if let (
                Some(StorageWrite::DocumentChange {
                    id,
                    content: DocumentContent::Delta { .. },
                }),
                Some(change),
            ) = (&plan.content, &plan.change)
                && let Some(loaded) = change.loaded()
            {
                let info = CheckpointInfo {
                    deltas_since_base: loaded.lock().deltas_since_base,
                };
                let checkpoint = match &change.definition {
                    Some(definition) => definition.checkpoint_when(
                        change.prepared.value(),
                        change.prepared.ops(),
                        info,
                    )?,
                    None => false,
                };
                if checkpoint {
                    plan.content = Some(StorageWrite::DocumentChange {
                        id: *id,
                        content: DocumentContent::Base {
                            version: change.version,
                            value: json_object(change.prepared.value()),
                        },
                    });
                }
            }
            if let Some(content) = &plan.content {
                writes.push(content.clone());
            }
            if plan.retire {
                writes.push(StorageWrite::DocumentRetire {
                    id: plan.record.id(),
                });
            }
        }
        self.inner.state.lock().plans = plans;
        Ok(writes)
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    /// New owned work needs a live owner, judged on the owner's final candidate: not `completing`, terminal, or
    /// abort-marked. A task therefore cannot create owned work in the commit that finishes it (spec §5.5).
    async fn validate_owners(&self) -> Result<()> {
        let owners: Vec<(&'static str, TaskId)> = {
            let state = self.inner.state.lock();
            let mut owners = Vec::new();
            for write in &state.writes {
                if let StorageWrite::Conversation { value } = write
                    && let Some(owner) = value.owner
                {
                    owners.push(("Conversation owner task", owner.task_id));
                }
            }
            for task in state.tasks_by_id.values() {
                if let Some(write) = &task.write
                    && write.kind == TaskWriteKind::Create
                    && let Some(owner) = write.record.owner
                {
                    owners.push(("Task owner", owner));
                }
            }
            owners
        };
        for (what, task_id) in owners {
            let Some(task) = self.current_task(task_id).await? else {
                return Err(Error::message(format!("{what} {task_id} does not exist")));
            };
            let status = task.state.status();
            if matches!(status, TaskStatus::Terminal | TaskStatus::Completing) {
                return Err(Error::message(format!("{what} {task_id} is {status}")));
            }
            if task.abort_requested {
                return Err(Error::message(format!("{what} {task_id} is abort-marked")));
            }
        }
        Ok(())
    }

    fn reject_fork_source_writes(&self, plans: &[DocumentPlan]) -> Result<()> {
        let state = self.inner.state.lock();
        for plan in plans {
            if plan.content.is_none() && !plan.retire {
                continue;
            }
            let id = plan.record.id();
            if state.fork_source_document_ids.contains(&id) {
                return Err(Error::message(format!(
                    "Cannot change fork source document {id} in the fork transaction"
                )));
            }
            if let DocumentScope::Conversation { conversation_id } = plan.record.scope()
                && state
                    .fork_source_conversation_ids
                    .contains(&conversation_id)
                && plan.record.fork() == Some(ForkPolicy::Current)
            {
                return Err(Error::message(format!(
                    "Cannot fork conversation {conversation_id} while changing its current-policy documents"
                )));
            }
        }
        Ok(())
    }

    /// Register an operation so callback settlement can reject and drain it.
    fn track<T: Send + 'static>(
        &self,
        operation: impl Future<Output = Result<T>> + Send + 'static,
    ) -> TxFuture<T> {
        self.inner.pending_operations.fetch_add(1, Ordering::SeqCst);
        let guard = PendingGuard(self.inner.clone());
        let handle = tokio::spawn(async move {
            let result = AssertUnwindSafe(operation).catch_unwind().await;
            drop(guard);
            match result {
                Ok(result) => result,
                Err(panic) => std::panic::resume_unwind(panic),
            }
        });
        Box::pin(join(handle))
    }

    fn read<T: Send + 'static>(
        &self,
        method: &str,
        read: impl Future<Output = Result<T>> + Send + 'static,
    ) -> TxFuture<T> {
        {
            let state = self.inner.state.lock();
            if let Err(error) = state.assert_open() {
                return ready(Err(error));
            }
            if state.has_table_write {
                return ready(Err(ReadAfterWrite::new(method).into()));
            }
        }
        self.track(read)
    }

    fn write<T: Send + 'static>(
        &self,
        write: impl Future<Output = Result<T>> + Send + 'static,
    ) -> TxFuture<T> {
        {
            let mut state = self.inner.state.lock();
            if let Err(error) = state.assert_open() {
                return ready(Err(error));
            }
            state.has_table_write = true;
        }
        self.track(write)
    }

    async fn require_conversation(&self, id: ConversationId) -> Result<()> {
        if self
            .inner
            .state
            .lock()
            .created_conversation_ids
            .contains(&id)
        {
            return Ok(());
        }
        if self
            .inner
            .storage
            .conversation(id, &self.inner.context)
            .await?
            .is_none()
        {
            return Err(Error::message(format!("Conversation {id} does not exist")));
        }
        Ok(())
    }

    /// Latest candidate task record, falling back to committed state; not a caller table read.
    async fn current_task(&self, id: TaskId) -> Result<Option<TaskRecord>> {
        let candidate = self
            .inner
            .state
            .lock()
            .tasks_by_id
            .get(&id)
            .and_then(|task| task.write.as_ref().map(|write| write.record.clone()));
        match candidate {
            Some(candidate) => Ok(Some(candidate)),
            None => self.committed_task(id).await,
        }
    }

    fn committed_task(&self, id: TaskId) -> SharedResult<Option<TaskRecord>> {
        let mut state = self.inner.state.lock();
        let task = state.task_entry(id);
        task.committed_read
            .get_or_insert_with(|| {
                let storage = self.inner.storage.clone();
                let context = self.inner.context.clone();
                async move { storage.task(id, &context).await }
                    .boxed()
                    .shared()
            })
            .clone()
    }
}

fn json_object(value: &JsonValue) -> crate::durable::types::JsonObject {
    match value {
        JsonValue::Object(object) => object.clone(),
        _ => unreachable!("documents are JSON objects"),
    }
}

async fn acquire(
    tx: Transaction,
    index: usize,
    seed: Option<JsonValue>,
    skip_load: bool,
) -> Result<()> {
    let (definition, address_id, address) = {
        let state = tx.inner.state.lock();
        let entry = &state.documents[index];
        (
            entry
                .definition
                .clone()
                .expect("an acquisition has a definition"),
            entry.address_id.clone(),
            entry.address.clone(),
        )
    };
    let loaded = if skip_load {
        None
    } else {
        tx.inner
            .host
            .load(
                definition.clone(),
                address_id,
                address.clone(),
                tx.inner.context.clone(),
            )
            .await?
    };
    tx.assert_open()?;
    if let Some(loaded) = loaded {
        let change = {
            let mut document = loaded.lock();
            check_record_scope(&definition, &document.record)?;
            check_record_version(&definition, &document.record, document.stored_version)?;
            document.tracker.begin_change()
        };
        let mut state = tx.inner.state.lock();
        state.assert_open()?;
        let entry = &mut state.documents[index];
        entry.target = Some(DocumentTarget::Loaded(loaded));
        entry.change = Some(change);
        return Ok(());
    }
    match address.scope {
        DocumentScope::Conversation { conversation_id } => {
            tx.require_conversation(conversation_id).await?;
        }
        DocumentScope::Task { task_id } => {
            let Some(task) = tx.current_task(task_id).await? else {
                return Err(Error::message(format!("Task {task_id} does not exist")));
            };
            if task.state.status() == TaskStatus::Terminal {
                return Err(Error::message(format!("Task {task_id} is terminal")));
            }
        }
        DocumentScope::Session => {}
    }
    tx.assert_open()?;
    let value = definition.initial(seed.as_ref())?;
    let id = DocumentId(tx.inner.storage.mint_id().await?);
    let mut state = tx.inner.state.lock();
    state.assert_open()?;
    let mut tracker = track(JsonValue::Object(value));
    let change = tracker.begin_change();
    let entry = &mut state.documents[index];
    entry.target = Some(DocumentTarget::Created {
        record: document_create(&definition, &address, id),
        version: definition.version,
        tracker,
    });
    entry.change = Some(change);
    Ok(())
}

async fn acquire_fork_copy(
    tx: Transaction,
    index: usize,
    definition: Arc<AnyDocDefinition>,
    target: ForkDocumentCopy,
) -> Result<()> {
    let stored = tx
        .inner
        .storage
        .document(target.source.id, target.source.at, &tx.inner.context)
        .await?;
    tx.assert_open()?;
    let Some(stored) = stored else {
        return Err(Error::message(format!(
            "Fork source document {} cannot be read",
            target.source.id
        )));
    };
    if !matches!(stored.record.scope, DocumentScope::Conversation { .. })
        || stored.record.kind != target.record.kind
        || stored.record.key != target.record.key
        || stored.record.history != target.record.history
        || stored.record.fork != target.record.fork
    {
        return Err(Error::message(format!(
            "Fork source document {} does not match the copied record",
            target.source.id
        )));
    }
    let value =
        materialize_document_value(&definition, &target.record, stored.version, stored.value)?;
    let mut tracker = track(JsonValue::Object(value));
    let change = tracker.begin_change();
    let mut state = tx.inner.state.lock();
    state.assert_open()?;
    let entry = &mut state.documents[index];
    entry.definition = Some(definition.clone());
    entry.target = Some(DocumentTarget::Created {
        record: target.record,
        version: definition.version,
        tracker,
    });
    entry.change = Some(change);
    Ok(())
}

async fn find_retirement(tx: Transaction, index: usize) -> Result<()> {
    let (definition, address_id, address) = {
        let state = tx.inner.state.lock();
        let entry = &state.documents[index];
        (
            entry
                .definition
                .clone()
                .expect("a retirement has a definition"),
            entry.address_id.clone(),
            entry.address.clone(),
        )
    };
    let cached = tx
        .inner
        .host
        .cached(&address_id)
        .map(|loaded| loaded.lock().record.clone());
    let record = match cached {
        Some(record) => Some(record),
        None => {
            tx.inner
                .storage
                .find_document(&address, DocumentPoint::Current, &tx.inner.context)
                .await?
        }
    };
    tx.assert_open()?;
    let Some(record) = record else {
        return Ok(());
    };
    check_record_scope(&definition, &record)?;
    let mut state = tx.inner.state.lock();
    state.assert_open()?;
    state.documents[index].target = Some(DocumentTarget::RetireOnly(record));
    Ok(())
}

/// Plan of one staged document: its record, content write, and prepared change. Retirement is decided later.
fn plan_document(document: &mut DocumentEntry) -> Option<DocumentPlan> {
    let target = document.target.take()?;
    let address_id = document.address_id.clone();
    let retire = document.retire_on_commit;
    let plan = match target {
        DocumentTarget::Created {
            record,
            version,
            tracker,
        } => {
            let prepared = document
                .prepared
                .clone()
                .expect("a created document is prepared");
            let content = StorageWrite::DocumentCreate {
                record: record.clone(),
                content: DocumentContent::Base {
                    version,
                    value: json_object(prepared.value()),
                },
            };
            DocumentPlan {
                address_id,
                record: PlanRecord::Create(record),
                retire,
                content: Some(content),
                change: Some(PlanChange {
                    tracker: PlanTracker::Created(tracker),
                    prepared,
                    version,
                    definition: document.definition.clone(),
                }),
                conversation_id: None,
            }
        }
        DocumentTarget::ForkCopy(copy) => {
            let content = StorageWrite::DocumentCopy {
                record: copy.record.clone(),
                source: copy.source,
            };
            // Keep the copy discoverable for a later retirement check.
            document.target = Some(DocumentTarget::ForkCopy(copy.clone()));
            DocumentPlan {
                address_id,
                record: PlanRecord::Create(copy.record),
                retire,
                content: Some(content),
                change: None,
                conversation_id: None,
            }
        }
        DocumentTarget::RetireOnly(record) => DocumentPlan {
            address_id,
            record: PlanRecord::Committed(record),
            retire,
            content: None,
            change: None,
            conversation_id: None,
        },
        DocumentTarget::Loaded(loaded) => {
            let definition = document
                .definition
                .clone()
                .expect("a loaded document has a definition");
            let prepared = document
                .prepared
                .clone()
                .expect("a loaded document is prepared");
            let version = definition.version;
            let (record, stored_version) = {
                let loaded = loaded.lock();
                (loaded.record.clone(), loaded.stored_version)
            };
            let id = record.id;
            // A version change stores a base even without operations; otherwise only a change stores a delta.
            let content = if stored_version < version {
                Some(StorageWrite::DocumentChange {
                    id,
                    content: DocumentContent::Base {
                        version,
                        value: json_object(prepared.value()),
                    },
                })
            } else if !prepared.ops().is_empty() {
                Some(StorageWrite::DocumentChange {
                    id,
                    content: DocumentContent::Delta {
                        version,
                        ops: prepared.ops().to_vec(),
                    },
                })
            } else {
                None
            };
            document.target = Some(DocumentTarget::Loaded(loaded.clone()));
            DocumentPlan {
                address_id,
                record: PlanRecord::Committed(record),
                retire,
                content,
                change: Some(PlanChange {
                    tracker: PlanTracker::Loaded(loaded),
                    prepared,
                    version,
                    definition: Some(definition),
                }),
                conversation_id: None,
            }
        }
    };
    Some(plan)
}

/// Whether adoption publishes the plan: every creation, copy, and retirement, and a loaded incarnation that writes
/// content, which includes a migration-only base so observers of the older shape receive the new value.
fn publishes(plan: &DocumentPlan) -> bool {
    plan.retire
        || plan
            .change
            .as_ref()
            .is_none_or(|change| change.loaded().is_none())
        || plan.content.is_some()
}

// ─── Drafts ──────────────────────────────────────────────────────────────────

/// An acquired document draft (`Draft<T>`): the transaction-local candidate of one incarnation.
///
/// Reads and edits go through the open Chord change; once the callback settles every access fails with
/// "Cannot use a settled overlay".
pub struct DocDraft<T> {
    tx: Transaction,
    index: usize,
    _value: PhantomData<fn() -> T>,
}

impl<T> Clone for DocDraft<T> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            index: self.index,
            _value: PhantomData,
        }
    }
}

impl<T> std::fmt::Debug for DocDraft<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocDraft")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

fn settled_overlay() -> Error {
    Error::type_error("Cannot use a settled overlay")
}

impl<T> DocDraft<T> {
    /// Whether two drafts are the same acquisition (TS draft identity).
    pub fn same(&self, other: &Self) -> bool {
        self.tx.ptr_eq(&other.tx) && self.index == other.index
    }

    /// The draft's current JSON value.
    pub fn json(&self) -> Result<JsonValue> {
        let state = self.tx.inner.state.lock();
        let change = state.documents[self.index]
            .change
            .as_ref()
            .ok_or_else(settled_overlay)?;
        Ok(change.state()?.clone())
    }

    /// Mutate the draft's JSON value in place.
    pub fn edit_json<R>(&self, mutate: impl FnOnce(&mut JsonValue) -> R) -> Result<R> {
        let mut state = self.tx.inner.state.lock();
        let change = state.documents[self.index]
            .change
            .as_mut()
            .ok_or_else(settled_overlay)?;
        let draft = change.state_mut()?;
        let mut candidate = draft.clone();
        let result = mutate(&mut candidate);
        if !candidate.is_object() {
            return Err(Error::type_error("Document values must be JSON objects"));
        }
        *draft = candidate;
        Ok(result)
    }
}

impl<T: Serialize + DeserializeOwned> DocDraft<T> {
    /// Decode the draft's current value.
    pub fn get(&self) -> Result<T> {
        let value = self.json()?;
        from_json("draft", &value)
    }

    /// Decode, mutate, and store the draft. The result is strict-checked before it replaces the draft.
    pub fn edit<R>(&self, mutate: impl FnOnce(&mut T) -> R) -> Result<R> {
        let mut state = self.tx.inner.state.lock();
        let change = state.documents[self.index]
            .change
            .as_mut()
            .ok_or_else(settled_overlay)?;
        let draft = change.state_mut()?;
        let mut typed: T = from_json("draft", draft)?;
        let result = mutate(&mut typed);
        let value = copy_json(&typed, None)?;
        if !value.is_object() {
            return Err(Error::type_error("Document values must be JSON objects"));
        }
        *draft = value;
        Ok(result)
    }

    /// Replace the draft's value.
    pub fn set(&self, value: T) -> Result<()> {
        self.edit(|draft| *draft = value)
    }
}
