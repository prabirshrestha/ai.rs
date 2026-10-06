//! Port of durable `src/session/session.ts`: the Session kernel.
//!
//! Divergences from Pi:
//! - The mutation line is a chain of spawned tokio tasks: each queued job waits
//!   for its predecessor, so jobs run in order even when the caller drops the
//!   returned future. Public operations run their synchronous checks at call
//!   time and return `'static` futures, like TS eager promises.
//! - TS subclasses override the protected `conversationCreated()` and
//!   `beforeClose()`; Rust passes them as [`SessionHooks`].
//! - Snapshot, state, and watch overloads collapse into one method per
//!   operation taking a [`DocAccess`] token and its address. Typed snapshots
//!   decode the committed value; [`SessionImpl::snapshot_json`] returns the
//!   shared committed revision itself. Document states and watches carry the
//!   untyped committed revision (`ObservedDocumentValue`).
//! - A commit callback receives the transaction by value and returns a future;
//!   a panic in the callback settles the transaction as a failure before it
//!   resumes on the caller.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use indexmap::IndexMap;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::chord::delta::{Op, track};
use crate::chord::{
    AbortSignal, Context, JsonValue, await_with_context, replicated_state, without_abort_signal,
};

use crate::durable::documents::{
    AnyDocDefinition, DocAccess, RewindableDocAccess, check_record_scope, check_record_version,
    from_json, materialize_document, resolve_address,
};
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, EntryId, Seq};
use crate::durable::types::{
    CommitChange, CommitPublication, ConversationRecord, DocumentAddress, DocumentChange,
    DocumentCommitChange, DocumentPoint, DocumentRecord, DocumentScope, Storage, StorageWrite,
};

use super::observation::{
    CommittedStateSource, CommittedWatch, ObservedDocumentValue, RETIREMENT_OPERATIONS,
};
use super::transaction::{
    LoadedDocument, LoadedRef, Transaction, TransactionHost, TransactionScope, join,
};
use super::{DocumentState, DocumentWatch};

/// Open a Session kernel over one storage backend.
pub fn create_session(storage: Arc<dyn Storage>) -> Session {
    SessionImpl::new(storage)
}

/// The public Session surface.
pub type Session = SessionImpl;

/// Synchronous post-adoption listener. It must not block or call Session operations.
pub type CommitListener = Arc<dyn Fn(&CommitPublication, &Context) + Send + Sync>;

/// Listener called synchronously when close begins.
pub type CloseListener = Arc<dyn Fn() + Send + Sync>;

/// Removes one listener.
pub type Unsubscribe = Box<dyn Fn() + Send + Sync>;

/// The protected TS `SessionImpl` hooks a Harness overrides.
pub trait SessionHooks: Send + Sync + 'static {
    /// Runs inside every transaction that creates or forks a conversation, after the conversation record is staged. A
    /// plain Session stages nothing; a Harness stages its built-in documents.
    fn conversation_created(
        &self,
        _tx: &Transaction,
        _record: &ConversationRecord,
    ) -> BoxFuture<'static, Result<()>> {
        Box::pin(futures::future::ready(Ok(())))
    }

    /// Runs after close seals admission and before the line closes Storage; must not fail.
    fn before_close(&self) -> BoxFuture<'static, ()> {
        Box::pin(futures::future::ready(()))
    }
}

struct PlainSession;

impl SessionHooks for PlainSession {}

/// A conversation document's current incarnation and value, read on the line.
#[derive(Debug, Clone)]
pub struct LineDocument {
    pub record: DocumentRecord,
    pub version: u32,
    pub value: Arc<JsonValue>,
}

type Documents = Arc<Mutex<HashMap<String, LoadedRef>>>;

struct SessionInner {
    storage: Arc<dyn Storage>,
    hooks: Arc<dyn SessionHooks>,
    documents: Documents,
    commit_listeners: Mutex<IndexMap<u64, CommitListener>>,
    close_listeners: Mutex<IndexMap<u64, CloseListener>>,
    next_listener: Mutex<u64>,
    tail: Mutex<Option<oneshot::Receiver<()>>>,
    closing: Mutex<Option<Shared<BoxFuture<'static, Result<()>>>>>,
    poison: Mutex<Option<Error>>,
}

impl TransactionHost for SessionInner {
    fn storage(&self) -> Arc<dyn Storage> {
        self.storage.clone()
    }

    fn cached(&self, address_id: &str) -> Option<LoadedRef> {
        self.documents.lock().get(address_id).cloned()
    }

    fn load(
        &self,
        definition: Arc<AnyDocDefinition>,
        address_id: String,
        address: DocumentAddress,
        context: Context,
    ) -> BoxFuture<'static, Result<Option<LoadedRef>>> {
        Box::pin(load_document(
            self.storage.clone(),
            self.documents.clone(),
            definition,
            address_id,
            address,
            context,
        ))
    }

    fn install(&self, document: LoadedDocument) {
        self.documents
            .lock()
            .insert(document.address_id.clone(), Arc::new(Mutex::new(document)));
    }

    fn evict(&self, address_id: &str, record_id: crate::durable::ids::DocumentId) {
        let mut documents = self.documents.lock();
        if documents
            .get(address_id)
            .is_some_and(|loaded| loaded.lock().record.id == record_id)
        {
            documents.remove(address_id);
        }
    }

    fn conversation_created(
        &self,
        tx: &Transaction,
        record: &ConversationRecord,
    ) -> BoxFuture<'static, Result<()>> {
        self.hooks.conversation_created(tx, record)
    }
}

impl SessionInner {
    fn assert_usable(&self) -> Result<()> {
        if self.closing.lock().is_some() {
            return Err(Error::message("Session is closed"));
        }
        self.assert_healthy()
    }

    fn assert_healthy(&self) -> Result<()> {
        if let Some(poison) = &*self.poison.lock() {
            return Err(Error::with_cause(
                "Session is poisoned by a failed commit after storage admission; reopen it",
                poison.clone(),
            ));
        }
        Ok(())
    }

    fn enqueue<T: Send + 'static>(
        &self,
        job: impl Future<Output = T> + Send + 'static,
    ) -> BoxFuture<'static, T> {
        let (release, done) = oneshot::channel::<()>();
        let previous = self.tail.lock().replace(done);
        let handle = tokio::spawn(async move {
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            let result = job.await;
            drop(release);
            result
        });
        Box::pin(join(handle))
    }

    fn next_listener_id(&self) -> u64 {
        let mut next = self.next_listener.lock();
        *next += 1;
        *next
    }

    fn subscribe_commits(self: &Arc<Self>, listener: CommitListener) -> Result<Unsubscribe> {
        self.assert_usable()?;
        let id = self.next_listener_id();
        self.commit_listeners.lock().insert(id, listener);
        let weak = Arc::downgrade(self);
        Ok(Box::new(move || {
            if let Some(inner) = weak.upgrade() {
                inner.commit_listeners.lock().shift_remove(&id);
            }
        }))
    }

    fn subscribe_close(self: &Arc<Self>, listener: CloseListener) -> Result<Unsubscribe> {
        self.assert_usable()?;
        let id = self.next_listener_id();
        self.close_listeners.lock().insert(id, listener);
        let weak = Arc::downgrade(self);
        Ok(Box::new(move || {
            if let Some(inner) = weak.upgrade() {
                inner.close_listeners.lock().shift_remove(&id);
            }
        }))
    }

    fn publish(
        &self,
        seq: Seq,
        writes: Vec<StorageWrite>,
        documents: Vec<DocumentCommitChange>,
        context: &Context,
    ) {
        let listeners: Vec<CommitListener> =
            self.commit_listeners.lock().values().cloned().collect();
        if listeners.is_empty() {
            return;
        }
        let mut changes = Vec::new();
        for write in writes {
            match write {
                StorageWrite::Conversation { value } => {
                    changes.push(CommitChange::Conversation(value))
                }
                StorageWrite::Entry { value } => changes.push(CommitChange::Entry(value)),
                StorageWrite::Task { value } => changes.push(CommitChange::Task(value)),
                StorageWrite::Submission { value } => changes.push(CommitChange::Submission(value)),
                _ => {}
            }
        }
        for document in documents {
            changes.push(match document {
                DocumentCommitChange::Document(change) => CommitChange::Document(change),
                DocumentCommitChange::Copy(change) => CommitChange::DocumentCopy(change),
            });
        }
        let publication = CommitPublication { seq, changes };
        for listener in listeners {
            listener(&publication, context);
        }
    }

    /// Attach an observer to one committed incarnation: check the definition, then forward this incarnation's committed
    /// changes and close. `detach` removes both subscriptions.
    fn attach_document<O: SessionObserver>(
        self: &Arc<Self>,
        definition: &AnyDocDefinition,
        loaded: &LoadedRef,
        create: impl FnOnce(ObservedDocumentValue, Detach) -> O,
    ) -> Result<(O, Detach)> {
        let (record_id, value, value_version) = {
            let loaded = loaded.lock();
            check_record_scope(definition, &loaded.record)?;
            check_record_version(definition, &loaded.record, loaded.stored_version)?;
            (
                loaded.record.id,
                loaded.tracker.value().clone(),
                loaded.value_version,
            )
        };
        let subscriptions: Arc<Mutex<Vec<Unsubscribe>>> = Arc::default();
        let detach: Detach = {
            let subscriptions = subscriptions.clone();
            Arc::new(move || {
                for unsubscribe in std::mem::take(&mut *subscriptions.lock()) {
                    unsubscribe();
                }
            })
        };
        let observer = create(Some(value), detach.clone());
        let observed = Mutex::new(value_version);
        let commit_observer = observer.clone();
        let unsubscribe_commit =
            self.subscribe_commits(Arc::new(move |publication, context| {
                for change in &publication.changes {
                    let CommitChange::Document(change) = change else {
                        continue;
                    };
                    if change.record.id != record_id {
                        continue;
                    }
                    // A document state's frames carry no caller cancellation; a watch observes its own cancellation.
                    let frame_context = if O::STRIP_ABORT_SIGNAL {
                        without_abort_signal(context)
                    } else {
                        context.clone()
                    };
                    let ops = observed_operations(&mut observed.lock(), change);
                    // A migration-only base changes nothing for an observer of the new version.
                    if ops.is_empty() {
                        continue;
                    }
                    commit_observer.advance(change.value.clone(), ops, frame_context);
                }
            }))?;
        subscriptions.lock().push(unsubscribe_commit);
        let close_observer = observer.clone();
        let unsubscribe_close =
            self.subscribe_close(Arc::new(move || close_observer.close_session()))?;
        subscriptions.lock().push(unsubscribe_close);
        Ok((observer, detach))
    }
}

type Detach = Arc<dyn Fn() + Send + Sync>;

/// The two committed observers a document can attach.
trait SessionObserver: Clone + Send + Sync + 'static {
    const STRIP_ABORT_SIGNAL: bool;
    fn advance(&self, value: ObservedDocumentValue, ops: Arc<[Op]>, context: Context);
    fn close_session(&self);
}

impl SessionObserver for CommittedStateSource<ObservedDocumentValue> {
    const STRIP_ABORT_SIGNAL: bool = true;

    fn advance(&self, value: ObservedDocumentValue, ops: Arc<[Op]>, context: Context) {
        CommittedStateSource::advance(self, value, ops, context);
    }

    fn close_session(&self) {
        CommittedStateSource::close_session(self);
    }
}

impl SessionObserver for CommittedWatch<ObservedDocumentValue> {
    const STRIP_ABORT_SIGNAL: bool = false;

    fn advance(&self, value: ObservedDocumentValue, ops: Arc<[Op]>, context: Context) {
        CommittedWatch::advance(self, value, ops, context);
    }

    fn close_session(&self) {
        CommittedWatch::close_session(self);
    }
}

/// Session kernel: one mutation line, the loaded document tracker cache, and committed publication.
///
/// Only committed state is observable. Every commit callback, preparation, Storage settlement, adoption, and
/// publication enqueue runs while the line is held; listeners run later.
#[derive(Clone)]
pub struct SessionImpl {
    inner: Arc<SessionInner>,
}

impl std::fmt::Debug for SessionImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionImpl").finish_non_exhaustive()
    }
}

fn rejected<T: Send + 'static>(error: Error) -> BoxFuture<'static, Result<T>> {
    Box::pin(futures::future::ready(Err(error)))
}

impl SessionImpl {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self::with_hooks(storage, Arc::new(PlainSession))
    }

    /// A Session whose protected hooks are `hooks` (TS subclassing).
    pub fn with_hooks(storage: Arc<dyn Storage>, hooks: Arc<dyn SessionHooks>) -> Self {
        Self {
            inner: Arc::new(SessionInner {
                storage,
                hooks,
                documents: Arc::default(),
                commit_listeners: Mutex::default(),
                close_listeners: Mutex::default(),
                next_listener: Mutex::new(0),
                tail: Mutex::new(None),
                closing: Mutex::new(None),
                poison: Mutex::new(None),
            }),
        }
    }

    /// The Session's storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.inner.storage
    }

    /// Whether two handles are the same Session.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn commit<T, F, Fut>(&self, change: F, context: &Context) -> BoxFuture<'static, Result<T>>
    where
        T: Send + 'static,
        F: FnOnce(Transaction) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        self.commit_with(change, context, TransactionScope::default())
    }

    /// Internal commit exposing the concrete transaction and its internal operations, such as the reserved-ID root
    /// bootstrap and task replacement. `scope` sets the default `tx.create_task()` conversation and the task attributed
    /// to appended entries.
    pub fn commit_with<T, F, Fut>(
        &self,
        change: F,
        context: &Context,
        scope: TransactionScope,
    ) -> BoxFuture<'static, Result<T>>
    where
        T: Send + 'static,
        F: FnOnce(Transaction) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        if let Err(error) = self.inner.assert_usable() {
            return rejected(error);
        }
        let inner = self.inner.clone();
        let context = context.clone();
        self.inner
            .enqueue(run_commit(inner, change, context, scope))
    }

    /// Internal: run a read-only job on the mutation line so multi-read derivations observe one committed state.
    pub fn read_on_line<T: Send + 'static>(
        &self,
        job: impl Future<Output = Result<T>> + Send + 'static,
    ) -> BoxFuture<'static, Result<T>> {
        if let Err(error) = self.inner.assert_usable() {
            return rejected(error);
        }
        let inner = self.inner.clone();
        self.inner.enqueue(async move {
            inner.assert_healthy()?;
            job.await
        })
    }

    /// Internal: a conversation document's current incarnation and value, for a job already running on the line (see
    /// `read_on_line()`). Absent documents are `None`.
    pub async fn conversation_document_on_line<A: DocAccess<Address = ConversationId>>(
        &self,
        token: &A,
        conversation_id: ConversationId,
        context: &Context,
    ) -> Result<Option<LineDocument>> {
        let definition = token.definition().clone();
        let resolved = resolve_address(&definition, &token.address_args(conversation_id))?;
        let loaded = load_document(
            self.inner.storage.clone(),
            self.inner.documents.clone(),
            definition.clone(),
            resolved.id,
            resolved.address,
            context.clone(),
        )
        .await?;
        let Some(loaded) = loaded else {
            return Ok(None);
        };
        let loaded = loaded.lock();
        check_record_scope(&definition, &loaded.record)?;
        check_record_version(&definition, &loaded.record, loaded.stored_version)?;
        Ok(Some(LineDocument {
            record: loaded.record.clone(),
            version: loaded.value_version,
            value: loaded.tracker.value().clone(),
        }))
    }

    /// The committed value of a document, decoded as the token's type.
    pub fn snapshot<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<A::Value>>> {
        let kind = token.definition().kind.clone();
        let snapshot = self.snapshot_json(token, address, context);
        Box::pin(async move {
            match snapshot.await? {
                None => Ok(None),
                Some(value) => from_json(&kind, &value).map(Some),
            }
        })
    }

    /// The committed immutable revision of a document (TS `snapshot()` identity).
    pub fn snapshot_json<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<Arc<JsonValue>>>> {
        if let Err(error) = self.inner.assert_usable() {
            return rejected(error);
        }
        let definition = token.definition().clone();
        let resolved = match resolve_address(&definition, &token.address_args(address)) {
            Ok(resolved) => resolved,
            Err(error) => return rejected(error),
        };
        let cached = self
            .inner
            .documents
            .lock()
            .get(&resolved.id)
            .filter(|cached| cached.lock().value_version == definition.version)
            .cloned();
        let checked = definition.clone();
        let read = move |loaded: &LoadedRef| -> Result<Arc<JsonValue>> {
            let loaded = loaded.lock();
            check_record_scope(&checked, &loaded.record)?;
            check_record_version(&checked, &loaded.record, loaded.stored_version)?;
            Ok(loaded.tracker.value().clone())
        };
        if let Some(cached) = cached {
            return Box::pin(futures::future::ready(read(&cached).map(Some)));
        }
        let inner = self.inner.clone();
        let context = context.clone();
        let loading = self.inner.enqueue({
            let definition = definition.clone();
            async move {
                inner.assert_healthy()?;
                load_document(
                    inner.storage.clone(),
                    inner.documents.clone(),
                    definition,
                    resolved.id,
                    resolved.address,
                    context,
                )
                .await
            }
        });
        Box::pin(async move {
            match loading.await? {
                None => Ok(None),
                Some(loaded) => read(&loaded).map(Some),
            }
        })
    }

    /// Attach a replicated state to the document's current committed incarnation.
    pub fn document_state<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<DocumentState>>> {
        if let Err(error) = self.inner.assert_usable() {
            return rejected(error);
        }
        let definition = token.definition().clone();
        let resolved = match resolve_address(&definition, &token.address_args(address)) {
            Ok(resolved) => resolved,
            Err(error) => return rejected(error),
        };
        let inner = self.inner.clone();
        let context = context.clone();
        self.inner.enqueue(async move {
            inner.assert_healthy()?;
            let Some(loaded) = load_document(
                inner.storage.clone(),
                inner.documents.clone(),
                definition.clone(),
                resolved.id,
                resolved.address,
                context,
            )
            .await?
            else {
                return Ok(None);
            };
            let (source, detach) =
                inner.attach_document(&definition, &loaded, |value, detach| {
                    CommittedStateSource::new(value, move || detach())
                })?;
            match replicated_state(&source, None) {
                Ok(state) => Ok(Some(state)),
                Err(error) => {
                    detach();
                    Err(error.into())
                }
            }
        })
    }

    /// Watch the exact committed frames of the document's current incarnation.
    pub fn watch_doc<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<DocumentWatch>>> {
        if let Err(error) = self.inner.assert_usable() {
            return rejected(error);
        }
        let definition = token.definition().clone();
        let resolved = match resolve_address(&definition, &token.address_args(address)) {
            Ok(resolved) => resolved,
            Err(error) => return rejected(error),
        };
        let signal = context.abort_signal().cloned();
        let cancelled = Arc::new(AtomicBool::new(
            signal.as_ref().is_some_and(AbortSignal::aborted),
        ));
        let listener = signal.as_ref().map(|signal| {
            let cancelled = cancelled.clone();
            signal.add_listener(move |_| cancelled.store(true, Ordering::SeqCst))
        });
        let inner = self.inner.clone();
        let context = context.clone();
        let watching = {
            let signal = signal.clone();
            let cancelled = cancelled.clone();
            self.inner.enqueue(async move {
                inner.assert_healthy()?;
                if cancelled.load(Ordering::SeqCst) {
                    return Err(cancellation_error(signal.as_ref()));
                }
                let loaded = load_document(
                    inner.storage.clone(),
                    inner.documents.clone(),
                    definition.clone(),
                    resolved.id,
                    resolved.address,
                    context,
                )
                .await?;
                if cancelled.load(Ordering::SeqCst) {
                    return Err(cancellation_error(signal.as_ref()));
                }
                let Some(loaded) = loaded else {
                    return Ok(None);
                };
                let (watch, _) = inner.attach_document(&definition, &loaded, |value, detach| {
                    CommittedWatch::new(value, move || detach(), None)
                })?;
                Ok(Some(watch))
            })
        };
        Box::pin(async move {
            let result = async {
                let Some(watch) = watching.await? else {
                    return Ok(None);
                };
                if cancelled.load(Ordering::SeqCst) {
                    watch.cancel();
                    return Err(cancellation_error(signal.as_ref()));
                }
                if let Some(signal) = &signal {
                    watch.observe_cancellation(signal)?;
                }
                Ok(Some(watch))
            }
            .await;
            if let (Some(signal), Some(listener)) = (&signal, listener) {
                signal.remove_listener(listener);
            }
            result
        })
    }

    /// The value of a rewindable conversation document as of `at`, decoded as the token's type.
    pub fn snapshot_as_of<A: RewindableDocAccess>(
        &self,
        token: &A,
        address: A::Address,
        at: EntryId,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<A::Value>>> {
        if let Err(error) = self.inner.assert_usable() {
            return rejected(error);
        }
        let definition = token.definition().clone();
        let resolved = match resolve_address(&definition, &token.address_args(address)) {
            Ok(resolved) => resolved,
            Err(error) => return rejected(error),
        };
        let DocumentScope::Conversation { conversation_id } = resolved.address.scope else {
            return rejected(Error::type_error(
                "Session.snapshotAsOf() requires a conversation document",
            ));
        };
        let inner = self.inner.clone();
        let context = context.clone();
        self.inner.enqueue(async move {
            inner.assert_healthy()?;
            let Some(stored_entry) = inner
                .storage
                .entry_in(conversation_id, at, &context)
                .await?
            else {
                return Err(Error::message(format!(
                    "Entry {at} is not visible from conversation {conversation_id}"
                )));
            };
            let address = DocumentAddress {
                scope: DocumentScope::Conversation {
                    conversation_id: stored_entry.entry.conversation_id,
                },
                ..resolved.address
            };
            let point = DocumentPoint::Seq(stored_entry.commit_seq);
            let Some(record) = inner
                .storage
                .find_document(&address, point, &context)
                .await?
            else {
                return Ok(None);
            };
            let Some(stored) = inner.storage.document(record.id, point, &context).await? else {
                return Err(Error::message(format!(
                    "Historical document {} ({}) cannot be read",
                    record.id, record.kind
                )));
            };
            let value = materialize_document(&definition, stored)?;
            from_json(&definition.kind, &JsonValue::Object(value)).map(Some)
        })
    }

    /// Seal admission, stop observers, settle admitted work, and close Storage. Idempotent.
    pub fn close(&self, context: &Context) -> BoxFuture<'static, Result<()>> {
        let closing = {
            let mut closing = self.inner.closing.lock();
            match &*closing {
                Some(closing) => (closing.clone(), false),
                None => {
                    let cleanup = without_abort_signal(context);
                    let inner = self.inner.clone();
                    // Seal admission before anything else runs, then stop observers; admitted work settles before
                    // Storage closes.
                    let handle = tokio::spawn(async move {
                        inner.hooks.before_close().await;
                        let line = inner.clone();
                        inner
                            .enqueue(async move {
                                line.commit_listeners.lock().clear();
                                line.documents.lock().clear();
                                line.storage.close(&cleanup).await
                            })
                            .await
                    });
                    let shared = join(handle).boxed().shared();
                    *closing = Some(shared.clone());
                    (shared, true)
                }
            }
        };
        let (closing, first) = closing;
        if first {
            let listeners: Vec<CloseListener> =
                std::mem::take(&mut *self.inner.close_listeners.lock())
                    .into_values()
                    .collect();
            for listener in listeners {
                listener();
            }
        }
        let context = context.clone();
        Box::pin(async move { await_with_context(closing, &context).await? })
    }

    /// Register a synchronous post-adoption listener. It must not block or call Session operations.
    pub fn subscribe_commits(
        &self,
        listener: impl Fn(&CommitPublication, &Context) + Send + Sync + 'static,
    ) -> Result<Unsubscribe> {
        self.inner.subscribe_commits(Arc::new(listener))
    }

    /// Register a listener called synchronously when close begins. It must not block or call Session operations.
    pub fn subscribe_close(
        &self,
        listener: impl Fn() + Send + Sync + 'static,
    ) -> Result<Unsubscribe> {
        self.inner.subscribe_close(Arc::new(listener))
    }

    /// The number of commit subscriptions (tests only).
    #[cfg(test)]
    pub(crate) fn commit_listener_count(&self) -> usize {
        self.inner.commit_listeners.lock().len()
    }

    /// Drop every loaded tracker on the mutation line; later access cold-loads from Storage.
    pub fn unload_documents(&self) -> BoxFuture<'static, ()> {
        let documents = self.inner.documents.clone();
        self.inner.enqueue(async move {
            documents.lock().clear();
        })
    }

    /// A weak handle that does not keep the Session open.
    pub fn downgrade(&self) -> WeakSession {
        WeakSession(Arc::downgrade(&self.inner))
    }
}

/// A Session handle that does not keep it alive.
#[derive(Clone)]
pub struct WeakSession(Weak<SessionInner>);

impl WeakSession {
    pub fn upgrade(&self) -> Option<SessionImpl> {
        self.0.upgrade().map(|inner| SessionImpl { inner })
    }
}

async fn run_commit<T, F, Fut>(
    inner: Arc<SessionInner>,
    change: F,
    context: Context,
    scope: TransactionScope,
) -> Result<T>
where
    F: FnOnce(Transaction) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    inner.assert_healthy()?;
    if let Some(signal) = context.abort_signal() {
        signal.throw_if_aborted()?;
    }
    let host: Arc<dyn TransactionHost> = inner.clone();
    let tx = Transaction::new(host, context.clone(), scope);
    let outcome = AssertUnwindSafe(async { change(tx.clone()).await })
        .catch_unwind()
        .await;
    let result = match outcome {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            tx.settle_failure().await;
            return Err(error);
        }
        Err(panic) => {
            tx.settle_failure().await;
            std::panic::resume_unwind(panic);
        }
    };
    let writes = tx.settle_success().await?;
    if writes.is_empty() {
        tx.discard();
        return Ok(result);
    }
    // Once admitted, caller cancellation does not interrupt Storage settlement.
    let seq = match inner
        .storage
        .commit(&writes, &without_abort_signal(&context))
        .await
    {
        Ok(seq) => seq,
        Err(error) => {
            tx.discard();
            // Callback errors never reach this branch; StorageRejected alone guarantees that no batch effect committed.
            if !error.is_storage_rejected() {
                *inner.poison.lock() = Some(error.clone());
            }
            return Err(error);
        }
    };
    let documents = match tx.adopt(seq) {
        Ok(documents) => documents,
        Err(error) => {
            // Storage already committed; a failed adoption leaves memory behind durable state.
            *inner.poison.lock() = Some(error.clone());
            return Err(error);
        }
    };
    inner.publish(seq, writes, documents, &context);
    Ok(result)
}

async fn load_document(
    storage: Arc<dyn Storage>,
    documents: Documents,
    definition: Arc<AnyDocDefinition>,
    address_id: String,
    address: DocumentAddress,
    context: Context,
) -> Result<Option<LoadedRef>> {
    {
        let mut cache = documents.lock();
        if let Some(cached) = cache.get(&address_id) {
            // A tracker serves only tokens of the version its value was materialized for; others reload from Storage.
            if cached.lock().value_version == definition.version {
                return Ok(Some(cached.clone()));
            }
            cache.remove(&address_id);
        }
    }
    let Some(record) = storage
        .find_document(&address, DocumentPoint::Current, &context)
        .await?
    else {
        return Ok(None);
    };
    let Some(stored) = storage
        .document(record.id, DocumentPoint::Current, &context)
        .await?
    else {
        return Err(Error::message(format!(
            "Current document {} ({}) cannot be read",
            record.id, record.kind
        )));
    };
    let record = stored.record.clone();
    let stored_version = stored.version;
    let deltas_since_base = stored.deltas_since_base;
    let value = materialize_document(&definition, stored)?;
    let loaded = Arc::new(Mutex::new(LoadedDocument {
        address_id: address_id.clone(),
        record,
        stored_version,
        value_version: definition.version,
        deltas_since_base,
        tracker: track(JsonValue::Object(value)),
    }));
    documents.lock().insert(address_id, loaded.clone());
    Ok(Some(loaded))
}

/// Operations an observer applies for one committed change. An observer hydrated under another definition version holds
/// a differently shaped value, so it receives the new value as a root replacement instead of operations for that shape.
fn observed_operations(observed: &mut u32, change: &DocumentChange) -> Arc<[Op]> {
    let Some(value) = &change.value else {
        return RETIREMENT_OPERATIONS.clone();
    };
    if change.version == Some(*observed) {
        return change.ops.clone();
    }
    if let Some(version) = change.version {
        *observed = version;
    }
    Arc::from(vec![Op::R((**value).clone())])
}

fn cancellation_error(signal: Option<&AbortSignal>) -> Error {
    Error::Aborted(
        signal
            .and_then(AbortSignal::reason)
            .unwrap_or_else(crate::chord::AbortReason::abort_error),
    )
}
