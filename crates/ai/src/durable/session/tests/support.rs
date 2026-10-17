//! Port of `test/session-support.ts`.

//!
//! Divergence: TS keeps both the exact admitted batches (for identity checks)
//! and detached copies; Rust storage receives borrowed batches it cannot keep,
//! so [`ControlledStorage::commits`] holds copies and identity checks compare
//! values.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::watch;

use crate::chord::{BACKGROUND_CONTEXT, Context, JsonValue};
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, DocumentId, EntryId, Seq, SubmissionId, TaskId};
use crate::durable::session::{SessionImpl, Transaction};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{
    CommitChange, CommitPublication, ConversationOwnership, ConversationQuery, ConversationRecord,
    Cursor, DocumentAddress, DocumentChange, DocumentCopyChange, DocumentPoint, DocumentQuery,
    DocumentRecord, EntryQuery, EntryRecord, Page, Storage, StorageWrite, StoredDocument,
    StoredEntry, SubmissionQuery, SubmissionRecord, TaskQuery, TaskRecord,
};

pub fn context() -> Context {
    BACKGROUND_CONTEXT.clone()
}

struct Held {
    gate: watch::Sender<bool>,
    entered: watch::Sender<bool>,
}

impl Held {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            gate: watch::channel(false).0,
            entered: watch::channel(false).0,
        })
    }

    async fn hold(&self) {
        self.entered.send_replace(true);
        let mut gate = self.gate.subscribe();
        let _ = gate.wait_for(|released| *released).await;
    }
}

/// A gate that holds calls until released and reports when the first held call arrives.
pub struct Gate {
    held: Arc<Held>,
    slot: Arc<Mutex<Option<Arc<Held>>>>,
}

impl Gate {
    pub async fn entered(&self) {
        let mut entered = self.held.entered.subscribe();
        let _ = entered.wait_for(|entered| *entered).await;
    }

    pub fn release(&self) {
        let mut slot = self.slot.lock();
        if slot
            .as_ref()
            .is_some_and(|held| Arc::ptr_eq(held, &self.held))
        {
            *slot = None;
        }
        self.held.gate.send_replace(true);
    }
}

/// Memory storage with observable commits, held calls, and injected commit failures.
#[derive(Default)]
pub struct ControlledStorage {
    inner: MemoryStorage,
    /// Batches admitted by Session.
    pub commits: Mutex<Vec<Vec<StorageWrite>>>,
    pub mint_count: AtomicUsize,
    pub document_read_count: AtomicUsize,
    commit_gate: Arc<Mutex<Option<Arc<Held>>>>,
    find_gate: Arc<Mutex<Option<Arc<Held>>>>,
    submission_gate: Arc<Mutex<Option<Arc<Held>>>>,
    commit_failure: Mutex<Option<Error>>,
    close_failure: Mutex<Option<Error>>,
    persistent: std::sync::atomic::AtomicBool,
    commit_filter: Mutex<Option<CommitFilter>>,
}

/// Decides per batch whether a commit fails before reaching the inner storage.
pub type CommitFilter = Box<dyn FnMut(&[StorageWrite]) -> Option<Error> + Send>;

impl ControlledStorage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn hold_commits(&self) -> Gate {
        let held = Held::new();
        *self.commit_gate.lock() = Some(held.clone());
        Gate {
            held,
            slot: self.commit_gate.clone(),
        }
    }

    pub fn hold_find_document(&self) -> Gate {
        let held = Held::new();
        *self.find_gate.lock() = Some(held.clone());
        Gate {
            held,
            slot: self.find_gate.clone(),
        }
    }

    /// Hold `submission()` reads until released.
    #[allow(dead_code)] // Used by the Harness submissions suite.
    pub fn hold_submission_reads(&self) -> Gate {
        let held = Held::new();
        *self.submission_gate.lock() = Some(held.clone());
        Gate {
            held,
            slot: self.submission_gate.clone(),
        }
    }

    /// Simulate a crash during the held commit: it never reaches storage, and later commits proceed.
    #[allow(dead_code)] // Used by the Harness recovery suites.
    pub fn crash(&self) {
        *self.commit_gate.lock() = None;
    }

    pub fn fail_next_commit(&self, error: Error) {
        *self.commit_failure.lock() = Some(error);
    }

    /// Fail `close()` after closing the inner storage.
    #[allow(dead_code)] // Used by the Harness lifecycle suite.
    pub fn fail_close(&self, error: Error) {
        *self.close_failure.lock() = Some(error);
    }

    /// Keep the data readable after `close()`, as a durable backend reopened at the same path would be.
    #[allow(dead_code)] // Used by the Harness recovery suites.
    pub fn persistent() -> Self {
        let storage = Self::default();
        storage.persistent.store(true, Ordering::SeqCst);
        storage
    }

    /// Fail the batches `filter` returns an error for.
    #[allow(dead_code)] // Used by the Harness ownership suite.
    pub fn filter_commits(&self, filter: CommitFilter) {
        *self.commit_filter.lock() = Some(filter);
    }

    pub fn commit_count(&self) -> usize {
        self.commits.lock().len()
    }

    pub fn last_commit(&self) -> Vec<StorageWrite> {
        self.commits.lock().last().cloned().expect("a commit")
    }

    pub fn mints(&self) -> usize {
        self.mint_count.load(Ordering::SeqCst)
    }

    pub fn document_reads(&self) -> usize {
        self.document_read_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Storage for ControlledStorage {
    async fn commit(&self, writes: &[StorageWrite], context: &Context) -> Result<Seq> {
        self.commits.lock().push(writes.to_vec());
        let held = self.commit_gate.lock().clone();
        if let Some(held) = held {
            held.hold().await;
        }
        let failure = self.commit_failure.lock().take();
        if let Some(failure) = failure {
            return Err(failure);
        }
        let filtered = self
            .commit_filter
            .lock()
            .as_mut()
            .and_then(|filter| filter(writes));
        if let Some(failure) = filtered {
            return Err(failure);
        }
        self.inner.commit(writes, context).await
    }

    async fn mint_id(&self) -> Result<u64> {
        self.mint_count.fetch_add(1, Ordering::SeqCst);
        self.inner.mint_id().await
    }

    async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<ConversationRecord>> {
        self.inner.conversation(id, context).await
    }

    async fn scan_conversations(
        &self,
        query: &ConversationQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<ConversationRecord>> {
        self.inner
            .scan_conversations(query, limit, cursor, context)
            .await
    }

    async fn entry(&self, id: EntryId, context: &Context) -> Result<Option<StoredEntry>> {
        self.inner.entry(id, context).await
    }

    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        context: &Context,
    ) -> Result<Option<StoredEntry>> {
        self.inner.entry_in(conversation_id, id, context).await
    }

    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
        context: &Context,
    ) -> Result<Option<EntryRecord>> {
        self.inner
            .find_latest_head_marker(conversation_id, at_or_before_entry_id, context)
            .await
    }

    async fn scan_entries(
        &self,
        query: &EntryQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<EntryRecord>> {
        self.inner.scan_entries(query, limit, cursor, context).await
    }

    async fn task(&self, id: TaskId, context: &Context) -> Result<Option<TaskRecord>> {
        self.inner.task(id, context).await
    }

    async fn scan_tasks(
        &self,
        query: &TaskQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<TaskRecord>> {
        self.inner.scan_tasks(query, limit, cursor, context).await
    }

    async fn submission(
        &self,
        id: SubmissionId,
        context: &Context,
    ) -> Result<Option<SubmissionRecord>> {
        let held = self.submission_gate.lock().clone();
        if let Some(held) = held {
            held.hold().await;
        }
        self.inner.submission(id, context).await
    }

    async fn scan_submissions(
        &self,
        query: &SubmissionQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<SubmissionRecord>> {
        self.inner
            .scan_submissions(query, limit, cursor, context)
            .await
    }

    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        context: &Context,
    ) -> Result<Option<SubmissionRecord>> {
        self.inner
            .submission_by_request(conversation_id, request_id, context)
            .await
    }

    async fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        context: &Context,
    ) -> Result<Option<DocumentRecord>> {
        let held = self.find_gate.lock().clone();
        if let Some(held) = held {
            held.hold().await;
        }
        self.inner.find_document(address, at, context).await
    }

    async fn document(
        &self,
        id: DocumentId,
        at: DocumentPoint,
        context: &Context,
    ) -> Result<Option<StoredDocument>> {
        self.document_read_count.fetch_add(1, Ordering::SeqCst);
        self.inner.document(id, at, context).await
    }

    async fn scan_documents(
        &self,
        query: &DocumentQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<DocumentRecord>> {
        self.inner
            .scan_documents(query, limit, cursor, context)
            .await
    }

    async fn close(&self, context: &Context) -> Result<()> {
        if !self.persistent.load(Ordering::SeqCst) {
            self.inner.close(context).await?;
        }
        match self.close_failure.lock().take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Session kernel plus its controlled storage and every committed publication.
pub struct TestSession {
    pub storage: Arc<ControlledStorage>,
    pub session: SessionImpl,
    pub publications: Arc<Mutex<Vec<CommitPublication>>>,
}

impl TestSession {
    pub fn published(&self) -> usize {
        self.publications.lock().len()
    }

    pub fn last_publication(&self) -> CommitPublication {
        self.publications
            .lock()
            .last()
            .cloned()
            .expect("a publication")
    }
}

pub fn open_test_session() -> TestSession {
    let storage = Arc::new(ControlledStorage::new());
    let session = SessionImpl::new(storage.clone());
    let publications: Arc<Mutex<Vec<CommitPublication>>> = Arc::default();
    let sink = publications.clone();
    let _unsubscribe = session
        .subscribe_commits(move |publication, _| sink.lock().push(publication.clone()))
        .expect("subscribe");
    TestSession {
        storage,
        session,
        publications,
    }
}

pub fn document_changes(publication: &CommitPublication) -> Vec<DocumentChange> {
    publication
        .changes
        .iter()
        .filter_map(|change| match change {
            CommitChange::Document(change) => Some(change.clone()),
            _ => None,
        })
        .collect()
}

pub fn document_copy_changes(publication: &CommitPublication) -> Vec<DocumentCopyChange> {
    publication
        .changes
        .iter()
        .filter_map(|change| match change {
            CommitChange::DocumentCopy(change) => Some(change.clone()),
            _ => None,
        })
        .collect()
}

/// Create one conversation and return its ID.
pub async fn create_conversation(session: &SessionImpl) -> ConversationId {
    session
        .commit(
            |tx| async move {
                Ok(tx
                    .create_conversation(ConversationOwnership::Ownerless)
                    .await?
                    .id)
            },
            &context(),
        )
        .await
        .expect("create conversation")
}

/// Resolve after pending tasks and one timer turn.
pub async fn flush() {
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(Duration::from_millis(1)).await;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
}

/// JSON encoding of a value, which is what TS `toEqual` compares.
pub fn to_json<T: Serialize>(value: &T) -> JsonValue {
    serde_json::to_value(value).expect("serializable")
}

/// Decode a JSON literal in Pi's wire shape.
pub fn from_json<T: DeserializeOwned>(value: JsonValue) -> T {
    serde_json::from_value(value).unwrap_or_else(|error| panic!("invalid literal: {error}"))
}

/// The `type` tag of a storage write.
pub fn write_type(write: &StorageWrite) -> String {
    to_json(write)["type"]
        .as_str()
        .expect("tagged write")
        .to_string()
}

/// Assert `actual` contains every field of `expected` (TS `toMatchObject`).
#[track_caller]
pub fn assert_matches(actual: &JsonValue, expected: &JsonValue) {
    fn matches(actual: &JsonValue, expected: &JsonValue) -> bool {
        match (actual, expected) {
            (JsonValue::Object(actual), JsonValue::Object(expected)) => expected
                .iter()
                .all(|(key, value)| actual.get(key).is_some_and(|actual| matches(actual, value))),
            (JsonValue::Array(actual), JsonValue::Array(expected)) => {
                actual.len() == expected.len()
                    && actual
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| matches(actual, expected))
            }
            _ => actual == expected,
        }
    }
    assert!(
        matches(actual, expected),
        "{actual:#} does not match {expected:#}"
    );
}

/// Expect a commit or operation to fail with a message containing `needle`.
#[track_caller]
pub fn assert_err<T: std::fmt::Debug>(result: Result<T>, needle: &str) -> Error {
    match result {
        Ok(value) => panic!("expected an error containing {needle:?}, got {value:?}"),
        Err(error) => {
            assert!(
                error.to_string().contains(needle),
                "expected an error containing {needle:?}, got {error:?}"
            );
            error
        }
    }
}

/// Run one commit and unwrap it.
pub async fn commit<T, F, Fut>(session: &SessionImpl, change: F) -> T
where
    T: Send + 'static,
    F: FnOnce(Transaction) -> Fut + Send + 'static,
    Fut: Future<Output = Result<T>> + Send + 'static,
{
    session
        .commit(change, &context())
        .await
        .unwrap_or_else(|error| panic!("commit failed: {error:?}"))
}

/// Run one internal commit and unwrap it.
pub async fn commit_with<T, F, Fut>(session: &SessionImpl, change: F) -> T
where
    T: Send + 'static,
    F: FnOnce(Transaction) -> Fut + Send + 'static,
    Fut: Future<Output = Result<T>> + Send + 'static,
{
    session
        .commit_with(change, &context(), Default::default())
        .await
        .unwrap_or_else(|error| panic!("commit failed: {error:?}"))
}

/// A one-shot signal tests resolve and await (TS `deferred()`).
#[derive(Clone)]
pub struct Deferred(Arc<watch::Sender<bool>>);

impl Default for Deferred {
    fn default() -> Self {
        Self(Arc::new(watch::channel(false).0))
    }
}

impl Deferred {
    pub fn resolve(&self) {
        self.0.send_replace(true);
    }

    pub async fn wait(&self) {
        let mut receiver = self.0.subscribe();
        let _ = receiver.wait_for(|resolved| *resolved).await;
    }
}
