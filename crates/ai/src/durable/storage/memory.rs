//! Port of durable `src/storage/memory.ts`: the detached in-memory reference
//! implementation of [`Storage`].
//!
//! Divergences from Pi: reads and retained writes are owned clones by
//! construction, so `clone`/`freeze` have no counterpart (a prepared commit's
//! writes are only reachable through `&`). State lives behind one mutex;
//! every method runs synchronously under it, as the JS methods do between
//! awaits.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::Mutex;

use crate::chord::delta::{Op, apply_immutable_batches};
use crate::chord::{Context, JsonValue};

use crate::durable::errors::{Error, Result, StorageRejected};
use crate::durable::ids::{
    ConversationId, DocumentId, EntryId, MAX_SAFE_INTEGER, Seq, SubmissionId, TaskId,
};
use crate::durable::types::{
    ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentContent,
    DocumentCreate, DocumentPoint, DocumentQuery, DocumentRecord, DocumentScope, EntryQuery,
    EntryRecord, History, Page, Storage, StorageWrite, StoredDocument, StoredEntry,
    SubmissionQuery, SubmissionRecord, SubmissionStatus, TaskQuery, TaskRecord, TaskStatus,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableName {
    Conversation,
    Entry,
    Task,
    Submission,
    Document,
}

impl TableName {
    fn as_str(self) -> &'static str {
        match self {
            Self::Conversation => "conversation",
            Self::Entry => "entry",
            Self::Task => "task",
            Self::Submission => "submission",
            Self::Document => "document",
        }
    }
}

#[derive(Debug, Clone)]
struct DocumentRevision {
    content: DocumentContent,
    seq: Seq,
}

#[derive(Debug, Clone)]
struct StoredDocumentState {
    record: DocumentRecord,
    revisions: Vec<DocumentRevision>,
}

#[derive(Debug, Clone, Default)]
struct DocumentAction {
    create: Option<DocumentCreate>,
    content: Option<DocumentContent>,
    retire: bool,
}

#[derive(Debug, Clone, Default)]
struct DocumentAddressIndex {
    ids: Vec<u64>,
    current_id: Option<DocumentId>,
}

#[derive(Debug, Default)]
struct State {
    record_types: HashMap<u64, TableName>,
    conversations: HashMap<u64, ConversationRecord>,
    conversation_ids: Vec<u64>,
    conversation_ids_by_owner_conversation: HashMap<u64, Vec<u64>>,
    conversation_ids_by_owner_task: HashMap<u64, Vec<u64>>,
    entries: HashMap<u64, EntryRecord>,
    entry_ids: HashMap<u64, Vec<u64>>,
    head_entry_ids: HashMap<u64, Vec<u64>>,
    entry_commit_seqs: HashMap<u64, Seq>,
    tasks: HashMap<u64, TaskRecord>,
    task_ids: Vec<u64>,
    task_ids_by_status: HashMap<TaskStatus, Vec<u64>>,
    submissions: HashMap<u64, SubmissionRecord>,
    submission_ids: Vec<u64>,
    submission_ids_by_status: HashMap<SubmissionStatus, Vec<u64>>,
    submission_ids_by_request: HashMap<u64, HashMap<String, SubmissionId>>,
    documents: HashMap<u64, StoredDocumentState>,
    document_addresses: HashMap<String, DocumentAddressIndex>,
    document_ids_by_scope: HashMap<String, Vec<u64>>,
}

#[derive(Debug)]
struct Inner {
    state: State,
    next_id: u64,
    next_seq: u64,
    closed: bool,
}

fn document_delta_batches<'a>(
    id: DocumentId,
    version: u32,
    revisions: &[&'a DocumentRevision],
) -> Result<Vec<&'a [Op]>> {
    revisions
        .iter()
        .map(|revision| match &revision.content {
            DocumentContent::Delta {
                version: delta_version,
                ops,
            } if *delta_version == version => Ok(ops.as_slice()),
            _ => Err(Error::message(format!(
                "Document {id} crosses a stored version boundary without a base"
            ))),
        })
        .collect()
}

fn cursor_id(cursor: Option<&Cursor>) -> Result<Option<u64>> {
    let Some(after) = cursor.and_then(|cursor| cursor.get("after")) else {
        return Ok(None);
    };
    super::cursor_after(after)
}

fn lower_bound(ids: &[u64], target: u64) -> usize {
    ids.partition_point(|&id| id < target)
}

fn upper_bound(ids: &[u64], target: u64) -> usize {
    ids.partition_point(|&id| id <= target)
}

fn insert_sorted(ids: &mut Vec<u64>, id: u64) {
    if ids.last().is_none_or(|&last| last < id) {
        ids.push(id);
    } else {
        let index = lower_bound(ids, id);
        ids.insert(index, id);
    }
}

fn remove_sorted(ids: &mut Vec<u64>, id: u64) {
    let index = lower_bound(ids, id);
    if ids.get(index) == Some(&id) {
        ids.remove(index);
    }
}

fn insert_map_id(index: &mut HashMap<u64, Vec<u64>>, key: u64, id: u64) {
    insert_sorted(index.entry(key).or_default(), id);
}

fn scope_key(scope: &DocumentScope) -> String {
    let key = match scope {
        DocumentScope::Session => serde_json::json!(["session"]),
        DocumentScope::Conversation { conversation_id } => {
            serde_json::json!(["conversation", conversation_id.0])
        }
        DocumentScope::Task { task_id } => serde_json::json!(["task", task_id.0]),
    };
    key.to_string()
}

fn address_key(address: &DocumentAddress) -> String {
    let key = match &address.key {
        None => serde_json::json!(["singleton"]),
        Some(key) => serde_json::json!(["family", key]),
    };
    serde_json::json!([address.kind, scope_key(&address.scope), key]).to_string()
}

fn record_address_key(kind: &str, scope: &DocumentScope, key: &Option<String>) -> String {
    address_key(&DocumentAddress {
        kind: kind.to_string(),
        scope: *scope,
        key: key.clone(),
    })
}

fn is_alive_at(record: &DocumentRecord, at: DocumentPoint) -> bool {
    match at {
        DocumentPoint::Current => record.retired_at.is_none(),
        DocumentPoint::Seq(at) => {
            record.created_at <= at && record.retired_at.is_none_or(|retired| at < retired)
        }
    }
}

fn is_current_only(record: &DocumentRecord) -> bool {
    !matches!(record.scope, DocumentScope::Conversation { .. })
        || record.history == Some(History::Latest)
}

fn page<T: Clone>(values: Vec<&T>, limit: usize, id: impl Fn(&T) -> u64) -> Result<Page<T>> {
    if values.len() <= limit {
        return Ok(Page {
            items: values.into_iter().cloned().collect(),
            next: None,
        });
    }
    // Pi reads `items.at(-1)!.id`, which throws for a zero limit.
    let Some(last) = limit.checked_sub(1).map(|index| values[index]) else {
        return Err(Error::type_error(PAGE_LIMIT_ZERO));
    };
    let mut cursor = Cursor::new();
    cursor.insert("after".into(), JsonValue::from(id(last)));
    Ok(Page {
        items: values.into_iter().take(limit).cloned().collect(),
        next: Some(cursor),
    })
}

/// The `TypeError` Pi's `page()` throws for a zero limit over a non-empty scan.
pub(crate) const PAGE_LIMIT_ZERO: &str = "Cannot read properties of undefined (reading 'id')";

/// A fully validated, detached state mutation whose application performs no fallible preparation.
pub struct PreparedMemoryCommit<'a> {
    pub seq: Seq,
    /// Detached writes for persistence.
    pub writes: Vec<StorageWrite>,
    storage: &'a MemoryStorage,
    document_actions: IndexMap<u64, DocumentAction>,
    applied: AtomicBool,
}

impl PreparedMemoryCommit<'_> {
    /// Apply the prepared mutation once; later calls return the same sequence.
    pub fn apply(&self) -> Seq {
        if !self.applied.swap(true, Ordering::SeqCst) {
            let mut inner = self.storage.inner.lock();
            apply_prepared_commit(&mut inner, &self.writes, &self.document_actions, self.seq);
        }
        self.seq
    }
}

/// Detached in-memory reference implementation of `Storage`.
///
/// Reads and retained writes are detached to match the ownership boundary
/// of serialization-backed stores. This is backend conformance, not validation.
#[derive(Debug)]
pub struct MemoryStorage {
    inner: Mutex<Inner>,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                state: State::default(),
                next_id: 2,
                next_seq: 1,
                closed: false,
            }),
        }
    }

    /// Validate and detach one commit without changing observable state.
    pub fn prepare_commit(
        &self,
        writes: &[StorageWrite],
        seq: Option<Seq>,
    ) -> Result<PreparedMemoryCommit<'_>> {
        let inner = self.inner.lock();
        assert_open(&inner)?;
        let seq = seq.unwrap_or(Seq(inner.next_seq));
        if seq.0 > MAX_SAFE_INTEGER || seq.0 < inner.next_seq {
            return Err(Error::message(format!(
                "Commit sequence {seq} does not strictly increase"
            )));
        }
        let detached_writes = resolve_document_copies(&inner.state, writes.to_vec())?;
        check_global_ids(&inner.state, &detached_writes)?;
        let document_actions = prepare_document_actions(&detached_writes)?;
        check_document_actions(&inner.state, &document_actions)?;
        Ok(PreparedMemoryCommit {
            seq,
            writes: detached_writes,
            storage: self,
            document_actions,
            applied: AtomicBool::new(false),
        })
    }

    fn read<T>(&self, read: impl FnOnce(&State) -> Result<T>) -> Result<T> {
        let inner = self.inner.lock();
        assert_open(&inner)?;
        read(&inner.state)
    }
}

fn assert_open(inner: &Inner) -> Result<()> {
    if inner.closed {
        return Err(Error::message("MemoryStorage is closed"));
    }
    Ok(())
}

fn resolve_document_copies(state: &State, writes: Vec<StorageWrite>) -> Result<Vec<StorageWrite>> {
    if !writes
        .iter()
        .any(|write| matches!(write, StorageWrite::DocumentCopy { .. }))
    {
        return Ok(writes);
    }
    let mut changed_document_ids = std::collections::HashSet::new();
    for write in &writes {
        match write {
            StorageWrite::DocumentCreate { record, .. }
            | StorageWrite::DocumentCopy { record, .. } => {
                changed_document_ids.insert(record.id.0);
            }
            StorageWrite::DocumentChange { id, .. } | StorageWrite::DocumentRetire { id } => {
                changed_document_ids.insert(id.0);
            }
            _ => {}
        }
    }
    writes
        .into_iter()
        .map(|write| {
            let StorageWrite::DocumentCopy { record, source } = write else {
                return Ok(write);
            };
            let resolved = (|| {
                if changed_document_ids.contains(&source.id.0) {
                    return Err(Error::message(format!(
                        "Fork source document {} is changed in the copy batch",
                        source.id
                    )));
                }
                let Some(stored) = materialize_document(state, source.id, source.at)? else {
                    return Err(Error::message(format!(
                        "Fork source document {} cannot be read",
                        source.id
                    )));
                };
                if !matches!(stored.record.scope, DocumentScope::Conversation { .. })
                    || !matches!(record.scope, DocumentScope::Conversation { .. })
                    || stored.record.kind != record.kind
                    || stored.record.key != record.key
                    || stored.record.history != record.history
                    || stored.record.fork != record.fork
                {
                    return Err(Error::message(format!(
                        "Fork source document {} does not match the copied record",
                        source.id
                    )));
                }
                Ok(StorageWrite::DocumentCreate {
                    record: record.clone(),
                    content: DocumentContent::Base {
                        version: stored.version,
                        value: stored.value,
                    },
                })
            })();
            resolved.map_err(|error| match error {
                Error::StorageRejected(_) => error,
                error => StorageRejected::with_cause(
                    format!("Document copy {} was rejected", record.id),
                    error,
                )
                .into(),
            })
        })
        .collect()
}

fn apply_prepared_commit(
    inner: &mut Inner,
    prepared: &[StorageWrite],
    document_actions: &IndexMap<u64, DocumentAction>,
    seq: Seq,
) {
    let state = &mut inner.state;
    for write in prepared {
        match write {
            StorageWrite::Conversation { value } => {
                let id = value.id.0;
                state.record_types.insert(id, TableName::Conversation);
                state.conversations.insert(id, value.clone());
                insert_sorted(&mut state.conversation_ids, id);
                if let Some(owner) = value.owner {
                    insert_map_id(
                        &mut state.conversation_ids_by_owner_conversation,
                        owner.conversation_id.0,
                        id,
                    );
                    insert_map_id(
                        &mut state.conversation_ids_by_owner_task,
                        owner.task_id.0,
                        id,
                    );
                }
                inner.next_id = inner.next_id.max(id + 1);
            }
            StorageWrite::Entry { value } => {
                let id = value.id.0;
                state.record_types.insert(id, TableName::Entry);
                state.entries.insert(id, value.clone());
                state.entry_commit_seqs.insert(id, seq);
                insert_sorted(
                    state.entry_ids.entry(value.conversation_id.0).or_default(),
                    id,
                );
                if value.head.is_some() {
                    insert_sorted(
                        state
                            .head_entry_ids
                            .entry(value.conversation_id.0)
                            .or_default(),
                        id,
                    );
                }
                inner.next_id = inner.next_id.max(id + 1);
            }
            StorageWrite::Task { value } => {
                let id = value.id.0;
                state.record_types.insert(id, TableName::Task);
                let status = value.state.status();
                match state.tasks.get(&id) {
                    None => {
                        insert_sorted(&mut state.task_ids, id);
                        insert_sorted(state.task_ids_by_status.entry(status).or_default(), id);
                    }
                    Some(previous) if previous.state.status() != status => {
                        let previous = previous.state.status();
                        remove_sorted(state.task_ids_by_status.entry(previous).or_default(), id);
                        insert_sorted(state.task_ids_by_status.entry(status).or_default(), id);
                    }
                    Some(_) => {}
                }
                state.tasks.insert(id, value.clone());
                inner.next_id = inner.next_id.max(id + 1);
            }
            StorageWrite::Submission { value } => {
                let id = value.id.0;
                state.record_types.insert(id, TableName::Submission);
                let previous = state.submissions.get(&id).cloned();
                match &previous {
                    None => {
                        insert_sorted(&mut state.submission_ids, id);
                        insert_sorted(
                            state
                                .submission_ids_by_status
                                .entry(value.status)
                                .or_default(),
                            id,
                        );
                    }
                    Some(previous) if previous.status != value.status => {
                        remove_sorted(
                            state
                                .submission_ids_by_status
                                .entry(previous.status)
                                .or_default(),
                            id,
                        );
                        insert_sorted(
                            state
                                .submission_ids_by_status
                                .entry(value.status)
                                .or_default(),
                            id,
                        );
                    }
                    Some(_) => {}
                }
                if let Some(previous) = &previous
                    && let Some(request_id) = &previous.request_id
                    && let Some(requests) = state
                        .submission_ids_by_request
                        .get_mut(&previous.conversation_id.0)
                {
                    if requests.get(request_id) == Some(&value.id) {
                        requests.remove(request_id);
                    }
                    if requests.is_empty() {
                        state
                            .submission_ids_by_request
                            .remove(&previous.conversation_id.0);
                    }
                }
                state.submissions.insert(id, value.clone());
                if let Some(request_id) = &value.request_id {
                    state
                        .submission_ids_by_request
                        .entry(value.conversation_id.0)
                        .or_default()
                        .insert(request_id.clone(), value.id);
                }
                inner.next_id = inner.next_id.max(id + 1);
            }
            StorageWrite::DocumentCopy { .. } => {
                unreachable!("Prepared document copy was not resolved")
            }
            StorageWrite::DocumentCreate { .. }
            | StorageWrite::DocumentChange { .. }
            | StorageWrite::DocumentRetire { .. } => {}
        }
    }
    apply_document_actions(inner, document_actions, seq);
    inner.next_seq = seq.0 + 1;
}

fn materialize_document(
    state: &State,
    id: DocumentId,
    at: DocumentPoint,
) -> Result<Option<StoredDocument>> {
    let Some(stored) = state.documents.get(&id.0) else {
        return Ok(None);
    };
    if at != DocumentPoint::Current && is_current_only(&stored.record) {
        return Err(Error::message(format!(
            "Document {id} does not retain historical content"
        )));
    }
    if !is_alive_at(&stored.record, at) {
        return Ok(None);
    }
    let revisions: Vec<&DocumentRevision> = match at {
        DocumentPoint::Current => stored.revisions.iter().collect(),
        DocumentPoint::Seq(at) => stored
            .revisions
            .iter()
            .filter(|revision| revision.seq <= at)
            .collect(),
    };
    let Some(base_index) = revisions
        .iter()
        .rposition(|revision| revision.content.is_base())
    else {
        return Err(Error::message(format!(
            "Document {id} is missing a required base"
        )));
    };
    let DocumentContent::Base {
        version,
        value: base,
    } = &revisions[base_index].content
    else {
        unreachable!("base_index selects a base");
    };
    let batches = document_delta_batches(id, *version, &revisions[base_index + 1..])?;
    let value = apply_immutable_batches(&JsonValue::Object(base.clone()), batches)?;
    let JsonValue::Object(value) = value else {
        return Err(Error::message(format!(
            "Document {id} does not materialize a JSON object"
        )));
    };
    Ok(Some(StoredDocument {
        record: stored.record.clone(),
        version: *version,
        value,
        deltas_since_base: (revisions.len() - base_index - 1) as u64,
    }))
}

/// Visit the visible entries of a conversation newest-first; `visit` returns false to stop.
fn visible_entries<'s>(
    state: &'s State,
    conversation_id: ConversationId,
    min_entry_id: Option<u64>,
    max_entry_id: Option<u64>,
    mut visit: impl FnMut(&'s EntryRecord) -> bool,
) -> Result<()> {
    if !state.conversations.contains_key(&conversation_id.0) {
        return Err(Error::message(format!(
            "Unknown conversation: {conversation_id}"
        )));
    }
    let min_entry_id = min_entry_id.unwrap_or(0);
    let mut current_id = conversation_id.0;
    let mut upper_entry_id = max_entry_id.unwrap_or(u64::MAX);
    loop {
        if let Some(ids) = state.entry_ids.get(&current_id) {
            for &id in ids[..upper_bound(ids, upper_entry_id)].iter().rev() {
                if id < min_entry_id {
                    break;
                }
                if !visit(&state.entries[&id]) {
                    return Ok(());
                }
            }
        }
        let conversation = &state.conversations[&current_id];
        let Some(parent) = conversation.parent else {
            break;
        };
        upper_entry_id = upper_entry_id.min(parent.at.0);
        if upper_entry_id < min_entry_id {
            break;
        }
        current_id = parent.conversation_id.0;
    }
    Ok(())
}

fn check_global_ids(state: &State, writes: &[StorageWrite]) -> Result<()> {
    let mut claimed: HashMap<u64, TableName> = HashMap::new();
    for write in writes {
        let (table, id) = match write {
            StorageWrite::DocumentChange { .. } | StorageWrite::DocumentRetire { .. } => continue,
            StorageWrite::DocumentCreate { record, .. }
            | StorageWrite::DocumentCopy { record, .. } => (TableName::Document, record.id.0),
            StorageWrite::Conversation { value } => (TableName::Conversation, value.id.0),
            StorageWrite::Entry { value } => (TableName::Entry, value.id.0),
            StorageWrite::Task { value } => (TableName::Task, value.id.0),
            StorageWrite::Submission { value } => (TableName::Submission, value.id.0),
        };
        let existing = state.record_types.get(&id).copied();
        let earlier = claimed.get(&id).copied();
        if matches!(
            table,
            TableName::Conversation | TableName::Entry | TableName::Document
        ) {
            if let Some(existing) = existing {
                return Err(Error::message(format!(
                    "ID {id} already belongs to {}",
                    existing.as_str()
                )));
            }
            if earlier.is_some() {
                return Err(Error::message(format!("ID {id} is written more than once")));
            }
        } else {
            if let Some(existing) = existing
                && existing != table
            {
                return Err(Error::message(format!(
                    "ID {id} already belongs to {}",
                    existing.as_str()
                )));
            }
            if let Some(earlier) = earlier
                && earlier != table
            {
                return Err(Error::message(format!(
                    "ID {id} is written as two record types"
                )));
            }
        }
        claimed.insert(id, table);
    }
    Ok(())
}

fn prepare_document_actions(writes: &[StorageWrite]) -> Result<IndexMap<u64, DocumentAction>> {
    let mut actions: IndexMap<u64, DocumentAction> = IndexMap::new();
    for write in writes {
        let id = match write {
            StorageWrite::DocumentCreate { record, .. } => record.id,
            StorageWrite::DocumentChange { id, .. } | StorageWrite::DocumentRetire { id } => *id,
            _ => continue,
        };
        let action = actions.entry(id.0).or_default();
        match write {
            StorageWrite::DocumentCreate { record, content } => {
                if action.create.is_some() || action.content.is_some() {
                    return Err(Error::message(format!(
                        "Document {id} has more than one content command"
                    )));
                }
                action.create = Some(record.clone());
                action.content = Some(content.clone());
            }
            StorageWrite::DocumentChange { content, .. } => {
                if action.content.is_some() {
                    return Err(Error::message(format!(
                        "Document {id} has more than one content command"
                    )));
                }
                action.content = Some(content.clone());
            }
            StorageWrite::DocumentRetire { .. } => {
                if action.retire {
                    return Err(Error::message(format!(
                        "Document {id} is retired more than once"
                    )));
                }
                action.retire = true;
            }
            _ => unreachable!(),
        }
    }
    Ok(actions)
}

fn check_document_actions(state: &State, actions: &IndexMap<u64, DocumentAction>) -> Result<()> {
    let mut live_counts: IndexMap<String, i64> = IndexMap::new();
    for (&id, action) in actions {
        let existing = state.documents.get(&id);
        if action.create.is_none() && existing.is_none() {
            return Err(Error::message(format!("Unknown document: {id}")));
        }
        if action.create.is_some() && existing.is_some() {
            return Err(Error::message(format!("Document {id} already exists")));
        }
        if existing.is_some_and(|existing| existing.record.retired_at.is_some()) {
            return Err(Error::message(format!("Document {id} is retired")));
        }
        let previous = existing.and_then(|existing| existing.revisions.last());
        if let Some(DocumentContent::Delta { version, .. }) = &action.content {
            let Some(previous) = previous else {
                return Err(Error::message(format!("Document {id} delta has no base")));
            };
            if previous.content.version() != *version {
                return Err(Error::message(format!(
                    "Document {id} version transition requires a base"
                )));
            }
        }

        let key = match (&action.create, existing) {
            (Some(create), _) => record_address_key(&create.kind, &create.scope, &create.key),
            (None, Some(existing)) => record_address_key(
                &existing.record.kind,
                &existing.record.scope,
                &existing.record.key,
            ),
            (None, None) => unreachable!(),
        };
        let current_id = state
            .document_addresses
            .get(&key)
            .and_then(|address| address.current_id);
        let live = live_counts
            .entry(key)
            .or_insert(if current_id.is_none() { 0 } else { 1 });
        if action.retire && current_id == Some(DocumentId(id)) {
            *live -= 1;
        }
        if action.create.is_some() && !action.retire {
            *live += 1;
        }
    }
    if live_counts.values().any(|&live| live > 1) {
        return Err(Error::message(
            "Document address already has a current incarnation",
        ));
    }
    Ok(())
}

fn apply_document_actions(inner: &mut Inner, actions: &IndexMap<u64, DocumentAction>, seq: Seq) {
    let state = &mut inner.state;
    for (&id, action) in actions {
        if let Some(create) = &action.create {
            let mut record = create.stamp(seq);
            if action.retire {
                record.retired_at = Some(seq);
            }
            let content = action.content.clone().expect("a create carries content");
            let key = record_address_key(&record.kind, &record.scope, &record.key);
            let scope = scope_key(&record.scope);
            state.record_types.insert(id, TableName::Document);
            state.documents.insert(
                id,
                StoredDocumentState {
                    record,
                    revisions: vec![DocumentRevision { content, seq }],
                },
            );
            insert_sorted(
                &mut state.document_addresses.entry(key).or_default().ids,
                id,
            );
            insert_sorted(state.document_ids_by_scope.entry(scope).or_default(), id);
            inner.next_id = inner.next_id.max(id + 1);
        } else if let Some(content) = &action.content {
            let stored = state.documents.get_mut(&id).expect("checked document");
            let revision = DocumentRevision {
                content: content.clone(),
                seq,
            };
            if revision.content.is_base() && is_current_only(&stored.record) {
                stored.revisions = vec![revision];
            } else {
                stored.revisions.push(revision);
            }
        }

        let stored = state.documents.get_mut(&id).expect("checked document");
        if action.retire && action.create.is_none() {
            stored.record.retired_at = Some(seq);
        }
        if action.retire && is_current_only(&stored.record) {
            stored.revisions.clear();
        }
        if action.create.is_some() || action.retire {
            let key = record_address_key(
                &stored.record.kind,
                &stored.record.scope,
                &stored.record.key,
            );
            let address = state
                .document_addresses
                .get_mut(&key)
                .expect("indexed document address");
            if action.retire && address.current_id == Some(DocumentId(id)) {
                address.current_id = None;
            }
            if action.create.is_some() && !action.retire {
                address.current_id = Some(DocumentId(id));
            }
        }
    }
}

#[async_trait]
impl Storage for MemoryStorage {
    async fn commit(&self, writes: &[StorageWrite], _context: &Context) -> Result<Seq> {
        Ok(self.prepare_commit(writes, None)?.apply())
    }

    async fn mint_id(&self) -> Result<u64> {
        let mut inner = self.inner.lock();
        assert_open(&inner)?;
        if inner.next_id > MAX_SAFE_INTEGER {
            return Err(Error::message("ID space is exhausted"));
        }
        let id = inner.next_id;
        inner.next_id += 1;
        Ok(id)
    }

    async fn conversation(
        &self,
        id: ConversationId,
        _context: &Context,
    ) -> Result<Option<ConversationRecord>> {
        self.read(|state| Ok(state.conversations.get(&id.0).cloned()))
    }

    async fn scan_conversations(
        &self,
        query: &ConversationQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<ConversationRecord>> {
        self.read(|state| {
            let empty = Vec::new();
            let ids = if let Some(owner_task_id) = query.owner_task_id {
                state
                    .conversation_ids_by_owner_task
                    .get(&owner_task_id.0)
                    .unwrap_or(&empty)
            } else if let Some(owner_conversation_id) = query.owner_conversation_id {
                state
                    .conversation_ids_by_owner_conversation
                    .get(&owner_conversation_id.0)
                    .unwrap_or(&empty)
            } else {
                &state.conversation_ids
            };
            let start = cursor_id(cursor)?.map_or(0, |after| upper_bound(ids, after));
            let mut values = Vec::new();
            for id in &ids[start..] {
                if values.len() > limit {
                    break;
                }
                let value = &state.conversations[id];
                if let Some(owner_conversation_id) = query.owner_conversation_id
                    && value.owner.map(|owner| owner.conversation_id) != Some(owner_conversation_id)
                {
                    continue;
                }
                values.push(value);
            }
            page(values, limit, |value| value.id.0)
        })
    }

    async fn entry(&self, id: EntryId, _context: &Context) -> Result<Option<StoredEntry>> {
        self.read(|state| {
            Ok(state.entries.get(&id.0).map(|entry| StoredEntry {
                entry: entry.clone(),
                commit_seq: state.entry_commit_seqs[&id.0],
            }))
        })
    }

    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        _context: &Context,
    ) -> Result<Option<StoredEntry>> {
        self.read(|state| {
            let mut found = None;
            visible_entries(state, conversation_id, Some(id.0), Some(id.0), |entry| {
                found = Some(entry.clone());
                false
            })?;
            Ok(found.map(|entry| StoredEntry {
                entry,
                commit_seq: state.entry_commit_seqs[&id.0],
            }))
        })
    }

    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
        _context: &Context,
    ) -> Result<Option<EntryRecord>> {
        self.read(|state| {
            if !state.conversations.contains_key(&conversation_id.0) {
                return Err(Error::message(format!(
                    "Unknown conversation: {conversation_id}"
                )));
            }
            let mut current_id = conversation_id.0;
            let mut upper_entry_id = at_or_before_entry_id.map_or(u64::MAX, |id| id.0);
            loop {
                if let Some(ids) = state.head_entry_ids.get(&current_id) {
                    let index = upper_bound(ids, upper_entry_id);
                    if index > 0 {
                        return Ok(Some(state.entries[&ids[index - 1]].clone()));
                    }
                }
                let conversation = &state.conversations[&current_id];
                let Some(parent) = conversation.parent else {
                    return Ok(None);
                };
                upper_entry_id = upper_entry_id.min(parent.at.0);
                current_id = parent.conversation_id.0;
            }
        })
    }

    async fn scan_entries(
        &self,
        query: &EntryQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<EntryRecord>> {
        self.read(|state| {
            let max_entry_id = match cursor_id(cursor)? {
                None => query.max_entry_id.map(|id| id.0),
                Some(0) => {
                    return Ok(Page {
                        items: Vec::new(),
                        next: None,
                    });
                }
                Some(after) => Some(
                    query
                        .max_entry_id
                        .map_or(u64::MAX, |id| id.0)
                        .min(after - 1),
                ),
            };
            let mut visible = Vec::new();
            visible_entries(
                state,
                query.conversation_id,
                query.min_entry_id.map(|id| id.0),
                max_entry_id,
                |entry| {
                    visible.push(entry);
                    visible.len() <= limit
                },
            )?;
            page(visible, limit, |entry| entry.id.0)
        })
    }

    async fn task(&self, id: TaskId, _context: &Context) -> Result<Option<TaskRecord>> {
        self.read(|state| Ok(state.tasks.get(&id.0).cloned()))
    }

    async fn scan_tasks(
        &self,
        query: &TaskQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<TaskRecord>> {
        self.read(|state| {
            let empty = Vec::new();
            let ids = match query.status {
                None => &state.task_ids,
                Some(status) => state.task_ids_by_status.get(&status).unwrap_or(&empty),
            };
            let start = cursor_id(cursor)?.map_or(0, |after| upper_bound(ids, after));
            let mut values = Vec::new();
            for id in &ids[start..] {
                if values.len() > limit {
                    break;
                }
                let value = &state.tasks[id];
                if query
                    .conversation_id
                    .is_some_and(|conversation_id| value.conversation_id != conversation_id)
                {
                    continue;
                }
                if query.kind.as_ref().is_some_and(|kind| &value.kind != kind) {
                    continue;
                }
                if query
                    .abort_requested
                    .is_some_and(|abort_requested| value.abort_requested != abort_requested)
                {
                    continue;
                }
                if query
                    .background
                    .is_some_and(|background| value.background != background)
                {
                    continue;
                }
                values.push(value);
            }
            page(values, limit, |value| value.id.0)
        })
    }

    async fn submission(
        &self,
        id: SubmissionId,
        _context: &Context,
    ) -> Result<Option<SubmissionRecord>> {
        self.read(|state| Ok(state.submissions.get(&id.0).cloned()))
    }

    async fn scan_submissions(
        &self,
        query: &SubmissionQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<SubmissionRecord>> {
        self.read(|state| {
            let empty = Vec::new();
            let ids = match query.status {
                None => &state.submission_ids,
                Some(status) => state
                    .submission_ids_by_status
                    .get(&status)
                    .unwrap_or(&empty),
            };
            let start = cursor_id(cursor)?.map_or(0, |after| upper_bound(ids, after));
            let mut values = Vec::new();
            for id in &ids[start..] {
                if values.len() > limit {
                    break;
                }
                let value = &state.submissions[id];
                if query
                    .conversation_id
                    .is_some_and(|conversation_id| value.conversation_id != conversation_id)
                {
                    continue;
                }
                values.push(value);
            }
            page(values, limit, |value| value.id.0)
        })
    }

    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        _context: &Context,
    ) -> Result<Option<SubmissionRecord>> {
        self.read(|state| {
            Ok(state
                .submission_ids_by_request
                .get(&conversation_id.0)
                .and_then(|requests| requests.get(request_id))
                .map(|id| state.submissions[&id.0].clone()))
        })
    }

    async fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        _context: &Context,
    ) -> Result<Option<DocumentRecord>> {
        self.read(|state| {
            let index = state.document_addresses.get(&address_key(address));
            if at == DocumentPoint::Current {
                return Ok(index
                    .and_then(|index| index.current_id)
                    .map(|id| state.documents[&id.0].record.clone()));
            }
            for id in index.map(|index| index.ids.as_slice()).unwrap_or_default() {
                let record = &state.documents[id].record;
                if is_alive_at(record, at) {
                    return Ok(Some(record.clone()));
                }
            }
            Ok(None)
        })
    }

    async fn document(
        &self,
        id: DocumentId,
        at: DocumentPoint,
        _context: &Context,
    ) -> Result<Option<StoredDocument>> {
        self.read(|state| materialize_document(state, id, at))
    }

    async fn scan_documents(
        &self,
        query: &DocumentQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<DocumentRecord>> {
        self.read(|state| {
            let empty = Vec::new();
            let ids = state
                .document_ids_by_scope
                .get(&scope_key(&query.scope))
                .unwrap_or(&empty);
            let start = cursor_id(cursor)?.map_or(0, |after| upper_bound(ids, after));
            let mut values = Vec::new();
            for id in &ids[start..] {
                if values.len() > limit {
                    break;
                }
                let record = &state.documents[id].record;
                if query.kind.as_ref().is_some_and(|kind| &record.kind != kind) {
                    continue;
                }
                if is_alive_at(record, query.at) {
                    values.push(record);
                }
            }
            page(values, limit, |record| record.id.0)
        })
    }

    async fn close(&self, _context: &Context) -> Result<()> {
        self.inner.lock().closed = true;
        Ok(())
    }
}
