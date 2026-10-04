//! Port of durable `src/harness/view.ts`: structural conversation view mounts (spec §9.3).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use futures::future::BoxFuture;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::json;

use crate::chord::delta::{Op, Path, Seg};
use crate::chord::{
    AttachedReplicatedState, Context, JsonValue, replicated_state, without_abort_signal,
};
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, DocumentId};
use crate::durable::session::SessionImpl;
use crate::durable::session::observation::{
    CommittedStateSource, CommittedWatch, ObservedValue, WatchListener,
};
use crate::durable::types::{
    CommitChange, CommitPublication, ConversationRecord, EntryRecord, Storage,
};

use super::agent::AGENT_DOC;
use super::context::{active_entries, capture_context_bounds};
use super::inbox::INBOX_DOC;
use super::live::LIVE_DOC;
use super::provider::PROVIDER_DOC;
use super::usage::USAGE_DOC;
use super::util::{abort_error, closed_error};

/// Structural mount of one conversation's active transcript and built-in documents (spec §9.3).
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationView {
    pub conversation: ConversationRecord,
    /// Raw active entries, as `ContextView.entries`: the head marker, then the non-head entries from its head.
    pub entries: Arc<[EntryRecord]>,
    /// Built-in conversation documents keyed by kind, in mount order; absent documents are absent.
    pub docs: Arc<IndexMap<String, Arc<JsonValue>>>,
}

impl ConversationView {
    /// The committed value of the mounted document `kind`.
    pub fn doc(&self, kind: &str) -> Option<&Arc<JsonValue>> {
        self.docs.get(kind)
    }

    /// The JSON form of the view, as the TS object.
    pub fn to_json(&self) -> JsonValue {
        let docs: serde_json::Map<String, JsonValue> = self
            .docs
            .iter()
            .map(|(kind, value)| (kind.clone(), (**value).clone()))
            .collect();
        json!({
            "conversation": self.conversation,
            "entries": &*self.entries,
            "docs": docs,
        })
    }
}

impl ObservedValue for ConversationView {
    fn is_retired(&self) -> bool {
        false
    }

    fn root_value(&self) -> JsonValue {
        self.to_json()
    }
}

/// Receives each next revision of a mount, and the Session's close.
pub trait ViewObserver: Send + Sync + 'static {
    fn advance(&self, _value: &ConversationView, _ops: &Arc<[Op]>, _context: &Context) {}
    /// Every publication, after the mount took it; `ops` are the mount's, possibly none.
    fn publication(
        &self,
        _before: &ConversationView,
        _after: &ConversationView,
        _ops: &Arc<[Op]>,
        _publication: &CommitPublication,
        _context: &Context,
    ) {
    }
    fn close_session(&self);
}

impl ViewObserver for CommittedStateSource<ConversationView> {
    fn advance(&self, value: &ConversationView, ops: &Arc<[Op]>, context: &Context) {
        CommittedStateSource::advance(self, value.clone(), ops.clone(), context.clone());
    }

    fn close_session(&self) {
        CommittedStateSource::close_session(self);
    }
}

impl ViewObserver for CommittedWatch<ConversationView> {
    fn advance(&self, value: &ConversationView, ops: &Arc<[Op]>, context: &Context) {
        CommittedWatch::advance(self, value.clone(), ops.clone(), context.clone());
    }

    fn close_session(&self) {
        CommittedWatch::close_session(self);
    }
}

/// A conversation view watch (`ConversationWatch`).
pub type ConversationWatch = CommittedWatch<ConversationView>;
/// The listener of a [`ConversationWatch`].
pub type ConversationWatchListener = WatchListener<ConversationView>;

/// One conversation's mount: its current revision, the document incarnations it shows, and its observers.
struct Mount {
    value: ConversationView,
    /// Mounted incarnation and definition version per kind; another incarnation or version is set whole.
    docs: HashMap<String, (DocumentId, u32)>,
    observers: IndexMap<u64, Arc<dyn ViewObserver>>,
}

type MountRef = Arc<Mutex<Mount>>;

struct ViewsInner {
    session: SessionImpl,
    mounts: Mutex<HashMap<ConversationId, MountRef>>,
    next_observer: AtomicU64,
    closed: AtomicBool,
}

/// Drops an attached observer, and the mount with its last observer.
pub type ViewDetach = Arc<dyn Fn() + Send + Sync>;

/// The release callback handed to an observer: drops it, and the mount with its last observer.
pub type ViewRelease = Box<dyn FnOnce() + Send>;

/// The Harness's conversation view mounts: at most one per conversation, built on the Session line by its first
/// observer and dropped with its last. Each mount advances from the Session's commit publications, which are durable.
#[derive(Clone)]
pub struct ConversationViews {
    inner: Arc<ViewsInner>,
}

impl std::fmt::Debug for ConversationViews {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationViews").finish_non_exhaustive()
    }
}

impl ConversationViews {
    pub fn new(session: SessionImpl) -> Result<Self> {
        let inner = Arc::new(ViewsInner {
            session: session.clone(),
            mounts: Mutex::new(HashMap::new()),
            next_observer: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        });
        let weak: Weak<ViewsInner> = Arc::downgrade(&inner);
        let commits = weak.clone();
        let unsubscribe = session.subscribe_commits(move |publication, context| {
            let Some(inner) = commits.upgrade() else {
                return;
            };
            let mounts: Vec<(ConversationId, MountRef)> = inner
                .mounts
                .lock()
                .iter()
                .map(|(id, mount)| (*id, mount.clone()))
                .collect();
            for (id, mount) in mounts {
                advance(id, &mount, publication, context);
            }
        })?;
        // The subscription lives as long as the Session.
        std::mem::forget(unsubscribe);
        let unsubscribe = session.subscribe_close(move || {
            let Some(inner) = weak.upgrade() else {
                return;
            };
            inner.closed.store(true, Ordering::SeqCst);
            let mounts: Vec<MountRef> = inner.mounts.lock().drain().map(|(_, m)| m).collect();
            for mount in mounts {
                let observers: Vec<Arc<dyn ViewObserver>> =
                    mount.lock().observers.values().cloned().collect();
                for observer in observers {
                    observer.close_session();
                }
            }
        })?;
        std::mem::forget(unsubscribe);
        Ok(Self { inner })
    }

    /// A disposable read-only Chord state of the view.
    pub async fn state(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<AttachedReplicatedState<ConversationView>> {
        let (source, detach) = self
            .attach(
                id,
                |value, release, _| {
                    Box::pin(async move { Ok(CommittedStateSource::new(value, release)) })
                },
                context,
            )
            .await?;
        match replicated_state(&source, None) {
            Ok(state) => Ok(state),
            Err(error) => {
                detach();
                Err(error.into())
            }
        }
    }

    /// A serialized exact-frame watch of the view; cancelling `context` stops it.
    pub async fn watch(&self, id: ConversationId, context: &Context) -> Result<ConversationWatch> {
        let (watch, _) = self
            .attach(
                id,
                |value, release, _| {
                    Box::pin(async move { Ok(CommittedWatch::new(value, release, None)) })
                },
                context,
            )
            .await?;
        if let Some(signal) = context.abort_signal() {
            if let Err(reason) = signal.throw_if_aborted() {
                watch.cancel();
                return Err(abort_error(reason));
            }
            watch.observe_cancellation(signal)?;
        }
        Ok(watch)
    }

    /// Register an observer created from the current revision, atomically on the Session line: it sees every later
    /// publication and nothing earlier. `create` may read committed Storage, still on the line. The returned detach
    /// drops it, and the mount with its last observer.
    pub fn attach<O, F>(
        &self,
        id: ConversationId,
        create: F,
        context: &Context,
    ) -> BoxFuture<'static, Result<(O, ViewDetach)>>
    where
        O: ViewObserver + Clone,
        F: FnOnce(ConversationView, ViewRelease, Arc<dyn Storage>) -> BoxFuture<'static, Result<O>>
            + Send
            + 'static,
    {
        let inner = self.inner.clone();
        let context = context.clone();
        let job = async move {
            let existing = inner.mounts.lock().get(&id).cloned();
            let mount = match existing {
                Some(mount) => mount,
                None => Arc::new(Mutex::new(build(&inner.session, id, &context).await?)),
            };
            let key = inner.next_observer.fetch_add(1, Ordering::SeqCst);
            let weak = Arc::downgrade(&inner);
            let detach_mount = mount.clone();
            let detach: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                let empty = {
                    let mut mount = detach_mount.lock();
                    mount.observers.shift_remove(&key);
                    mount.observers.is_empty()
                };
                if !empty {
                    return;
                }
                if let Some(inner) = weak.upgrade() {
                    let mut mounts = inner.mounts.lock();
                    if mounts
                        .get(&id)
                        .is_some_and(|current| Arc::ptr_eq(current, &detach_mount))
                    {
                        mounts.remove(&id);
                    }
                }
            });
            let value = mount.lock().value.clone();
            let release = detach.clone();
            let observer = create(
                value,
                Box::new(move || release()),
                inner.session.storage().clone(),
            )
            .await?;
            // Close or cancellation may begin while the mount hydrates; register nothing then.
            if inner.closed.load(Ordering::SeqCst) {
                return Err(closed_error());
            }
            if let Some(signal) = context.abort_signal() {
                signal.throw_if_aborted().map_err(abort_error)?;
            }
            inner.mounts.lock().insert(id, mount.clone());
            mount
                .lock()
                .observers
                .insert(key, Arc::new(observer.clone()) as Arc<dyn ViewObserver>);
            Ok((observer, detach))
        };
        self.inner.session.read_on_line(job)
    }
}

async fn build(session: &SessionImpl, id: ConversationId, context: &Context) -> Result<Mount> {
    let storage = session.storage().clone();
    let Some(conversation) = storage.conversation(id, context).await? else {
        return Err(Error::message(format!("Conversation {id} does not exist")));
    };
    let bounds = capture_context_bounds(&storage, id, context, None).await?;
    let entries = active_entries(&storage, id, bounds.as_ref(), context).await?;
    let mut docs = IndexMap::new();
    let mut incarnations = HashMap::new();
    macro_rules! mount {
        ($token:expr) => {{
            let token = &*$token;
            let kind = token.definition().kind.clone();
            if let Some(loaded) = session
                .conversation_document_on_line(token, id, context)
                .await?
            {
                incarnations.insert(kind.clone(), (loaded.record.id, loaded.version));
                docs.insert(kind, loaded.value);
            }
        }};
    }
    mount!(AGENT_DOC);
    mount!(LIVE_DOC);
    mount!(INBOX_DOC);
    mount!(PROVIDER_DOC);
    mount!(USAGE_DOC);
    Ok(Mount {
        value: ConversationView {
            conversation,
            entries: entries.into(),
            docs: Arc::new(docs),
        },
        docs: incarnations,
        observers: IndexMap::new(),
    })
}

/// Kinds of the mounted built-in documents.
pub const MOUNTED_KINDS: [&str; 5] = ["pi.agent", "pi.live", "pi.inbox", "pi.provider", "pi.usage"];

/// Derive the mount's operations from one publication, apply them, and hand the revision to every observer.
fn advance(
    id: ConversationId,
    mount: &MountRef,
    publication: &CommitPublication,
    context: &Context,
) {
    let (before, after, ops, observers) = {
        let mut mount = mount.lock();
        let mut doc_ops: Vec<Op> = Vec::new();
        let mut entry_ops: Vec<Op> = Vec::new();
        let mut entries: Option<Vec<EntryRecord>> = None;
        let mut docs: Option<IndexMap<String, Arc<JsonValue>>> = None;
        // Entry writes are published in ID order.
        for change in &publication.changes {
            match change {
                CommitChange::Entry(entry) if entry.conversation_id == id => {
                    let list = entries.get_or_insert_with(|| mount.value.entries.to_vec());
                    let value = serde_json::to_value(entry).expect("entry records serialize");
                    let Some(target) = entry.head else {
                        entry_ops.push(Op::P(
                            vec![Seg::from("entries")],
                            list.len(),
                            0,
                            vec![value],
                        ));
                        list.push(entry.clone());
                        continue;
                    };
                    // A head marker keeps the non-head entries from its head, which are always a suffix, and goes
                    // in front.
                    let kept = list
                        .iter()
                        .position(|candidate| candidate.head.is_none() && candidate.id >= target)
                        .unwrap_or(list.len());
                    entry_ops.push(Op::P(vec![Seg::from("entries")], 0, kept, vec![value]));
                    list.drain(..kept);
                    list.insert(0, entry.clone());
                }
                CommitChange::Document(change) if change.conversation_id == Some(id) => {
                    let kind = &change.record.kind;
                    if !MOUNTED_KINDS.contains(&kind.as_str()) || change.record.key.is_some() {
                        continue;
                    }
                    let path: Path = vec![Seg::from("docs"), Seg::from(kind.clone())];
                    let mounted = mount.docs.get(kind).copied();
                    let current = mount.value.docs.clone();
                    match &change.value {
                        None => {
                            if mounted.map(|(doc, _)| doc) != Some(change.record.id) {
                                continue;
                            }
                            mount.docs.remove(kind);
                            docs.get_or_insert_with(|| (*current).clone())
                                .shift_remove(kind);
                            doc_ops.push(Op::D(path));
                        }
                        Some(value) => {
                            if mounted == change.version.map(|version| (change.record.id, version))
                            {
                                if change.ops.is_empty() {
                                    continue;
                                }
                                doc_ops.extend(change.ops.iter().map(|op| prefixed(op, &path)));
                            } else {
                                mount.docs.insert(
                                    kind.clone(),
                                    (change.record.id, change.version.unwrap_or_default()),
                                );
                                doc_ops.push(Op::S(path, (**value).clone()));
                            }
                            // The adopted revision is the mounted value after its operations; an unchanged document
                            // keeps its revision (TS identity).
                            docs.get_or_insert_with(|| (*current).clone())
                                .insert(kind.clone(), value.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        let before = mount.value.clone();
        let ops: Vec<Op> = doc_ops.into_iter().chain(entry_ops).collect();
        if !ops.is_empty() {
            if let Some(entries) = entries {
                mount.value.entries = entries.into();
            }
            if let Some(docs) = docs {
                mount.value.docs = Arc::new(docs);
            }
        }
        let observers: Vec<Arc<dyn ViewObserver>> = mount.observers.values().cloned().collect();
        (
            before,
            mount.value.clone(),
            Arc::<[Op]>::from(ops),
            observers,
        )
    };
    let frame_context = without_abort_signal(context);
    if !ops.is_empty() {
        for observer in &observers {
            observer.advance(&after, &ops, &frame_context);
        }
    }
    for observer in &observers {
        observer.publication(&before, &after, &ops, publication, &frame_context);
    }
}

/// `op` moved under `prefix`; a root replacement becomes a set of the prefix.
fn prefixed(op: &Op, prefix: &Path) -> Op {
    let at = |path: &Path| -> Path { prefix.iter().cloned().chain(path.iter().cloned()).collect() };
    match op {
        Op::R(value) => Op::S(prefix.clone(), value.clone()),
        Op::P(path, index, remove, items) => Op::P(at(path), *index, *remove, items.clone()),
        Op::M(path, permutation) => Op::M(at(path), permutation.clone()),
        Op::S(path, value) => Op::S(at(path), value.clone()),
        Op::D(path) => Op::D(at(path)),
        Op::A(path, text) => Op::A(at(path), text.clone()),
        Op::T(path, count) => Op::T(at(path), *count),
    }
}
