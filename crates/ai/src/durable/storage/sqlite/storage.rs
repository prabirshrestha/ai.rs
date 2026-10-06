//! Port of durable `src/storage/sqlite/storage.ts`: the portable
//! [`SqliteStorage`] core over any [`SqliteDatabase`] facade.
//!
//! Divergences from Pi: records are encoded with serde (field order follows
//! the Rust record structs, values are identical JSON), and the overloaded
//! `entry` is the trait's `entry`/`entry_in` pair. `close` is memoized as a
//! shared future so a repeated close settles with the first.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use tokio::sync::Notify;

use crate::chord::delta::{Op, apply};
use crate::chord::{Context, JsonValue};
use crate::durable::errors::{Error, Result, StorageRejected};
use crate::durable::ids::{
    ConversationId, DocumentId, EntryId, MAX_SAFE_INTEGER, Seq, SubmissionId, TaskId,
};
use crate::durable::types::{
    ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentContent,
    DocumentCopySource, DocumentCreate, DocumentPoint, DocumentQuery, DocumentRecord,
    DocumentScope, EntryQuery, EntryRecord, History, JsonObject, Page, Storage, StorageWrite,
    StoredDocument, StoredEntry, SubmissionQuery, SubmissionRecord, TaskQuery, TaskRecord,
};

use super::database::{SqliteDatabase, SqliteDatabaseExt, SqliteExecutor, SqliteRow, SqliteValue};
use super::migrations::apply_sqlite_migrations;

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

#[derive(Debug, Clone, Default)]
struct DocumentAction {
    create: Option<DocumentCreate>,
    copy: Option<DocumentCopySource>,
    content: Option<DocumentContent>,
    retire: bool,
}

struct ScopeColumns {
    scope_kind: &'static str,
    owner_id: u64,
}

struct AddressParts {
    kind: String,
    scope_kind: &'static str,
    owner_id: u64,
    family: i64,
    key_value: String,
}

fn parse_json<T: DeserializeOwned>(value: &str) -> Result<T> {
    Ok(serde_json::from_str(value)?)
}

fn encode_json<T: serde::Serialize + ?Sized>(value: &T) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

// Some SQLite bindings replace lone UTF-16 surrogates. JSON encoding keeps indexed identities lossless.
fn encode_indexed_string(value: &str) -> String {
    serde_json::to_string(value).expect("strings encode")
}

fn column<'r>(row: &'r SqliteRow, name: &str) -> Result<&'r SqliteValue> {
    row.get(name)
        .ok_or_else(|| Error::message(format!("SQLite row is missing column {name}")))
}

fn text_column<'r>(row: &'r SqliteRow, name: &str) -> Result<&'r str> {
    column(row, name)?
        .as_str()
        .ok_or_else(|| Error::type_error(format!("SQLite column {name} is not text")))
}

fn integer_column(row: &SqliteRow, name: &str) -> Result<i64> {
    column(row, name)?
        .as_i64()
        .ok_or_else(|| Error::type_error(format!("SQLite column {name} is not an integer")))
}

fn record<T: DeserializeOwned>(row: &SqliteRow) -> Result<T> {
    parse_json(text_column(row, "record")?)
}

fn cursor_id(cursor: Option<&Cursor>) -> Result<Option<u64>> {
    let Some(after) = cursor.and_then(|cursor| cursor.get("after")) else {
        return Ok(None);
    };
    crate::durable::storage::cursor_after(after)
}

/// `cursorId(cursor) ?? -1` as a binding.
fn cursor_binding(cursor: Option<&Cursor>) -> Result<SqliteValue> {
    Ok(cursor_id(cursor)?.map_or(SqliteValue::Integer(-1), SqliteValue::from))
}

fn page<T>(mut values: Vec<T>, limit: usize, id: impl Fn(&T) -> u64) -> Result<Page<T>> {
    if values.len() <= limit {
        return Ok(Page {
            items: values,
            next: None,
        });
    }
    values.truncate(limit);
    // Pi reads `items.at(-1)!.id`, which throws for a zero limit.
    let Some(last) = values.last() else {
        return Err(Error::type_error(
            crate::durable::storage::memory::PAGE_LIMIT_ZERO,
        ));
    };
    let mut cursor = Cursor::new();
    cursor.insert("after".into(), JsonValue::from(id(last)));
    Ok(Page {
        items: values,
        next: Some(cursor),
    })
}

fn scope_columns(scope: &DocumentScope) -> ScopeColumns {
    match scope {
        DocumentScope::Session => ScopeColumns {
            scope_kind: "session",
            owner_id: 0,
        },
        DocumentScope::Conversation { conversation_id } => ScopeColumns {
            scope_kind: "conversation",
            owner_id: conversation_id.0,
        },
        DocumentScope::Task { task_id } => ScopeColumns {
            scope_kind: "task",
            owner_id: task_id.0,
        },
    }
}

fn address_parts(kind: &str, scope: &DocumentScope, key: Option<&str>) -> AddressParts {
    let scope = scope_columns(scope);
    AddressParts {
        kind: encode_indexed_string(kind),
        scope_kind: scope.scope_kind,
        owner_id: scope.owner_id,
        family: i64::from(key.is_some()),
        key_value: encode_indexed_string(key.unwrap_or("")),
    }
}

fn address_key(parts: &AddressParts) -> String {
    serde_json::to_string(&(
        &parts.kind,
        parts.scope_kind,
        parts.owner_id,
        parts.family,
        &parts.key_value,
    ))
    .expect("address keys encode")
}

fn is_alive_at(record: &DocumentRecord, at: DocumentPoint) -> bool {
    match at {
        DocumentPoint::Current => record.retired_at.is_none(),
        DocumentPoint::Seq(at) => {
            record.created_at <= at && record.retired_at.is_none_or(|retired| at < retired)
        }
    }
}

fn is_current_only(scope: &DocumentScope, history: Option<History>) -> bool {
    !matches!(scope, DocumentScope::Conversation { .. }) || history == Some(History::Latest)
}

fn write_id(write: &StorageWrite) -> Option<u64> {
    match write {
        StorageWrite::Conversation { value } => Some(value.id.0),
        StorageWrite::Entry { value } => Some(value.id.0),
        StorageWrite::Task { value } => Some(value.id.0),
        StorageWrite::Submission { value } => Some(value.id.0),
        StorageWrite::DocumentCreate { record, .. } | StorageWrite::DocumentCopy { record, .. } => {
            Some(record.id.0)
        }
        StorageWrite::DocumentChange { .. } | StorageWrite::DocumentRetire { .. } => None,
    }
}

fn point_binding(at: DocumentPoint) -> SqliteValue {
    match at {
        DocumentPoint::Current => SqliteValue::Integer(MAX_SAFE_INTEGER as i64),
        DocumentPoint::Seq(seq) => seq.0.into(),
    }
}

struct Inner {
    db: Arc<dyn SqliteDatabase>,
    next_id: Mutex<u64>,
    closed: AtomicBool,
    closing: Mutex<Option<Shared<BoxFuture<'static, Result<()>>>>>,
    admitted_reads: Mutex<usize>,
    reads_drained: Notify,
}

/// Portable SQLite implementation of the Pico storage contract (`SqliteStorage`).
#[derive(Clone)]
pub struct SqliteStorage {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for SqliteStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStorage")
            .field("next_id", &*self.inner.next_id.lock())
            .field("closed", &self.inner.closed.load(Ordering::SeqCst))
            .finish()
    }
}

/// Decrements the admitted-read count when a multi-query read finishes or is dropped.
struct AdmittedRead<'a>(&'a Inner);

impl Drop for AdmittedRead<'_> {
    fn drop(&mut self) {
        let mut reads = self.0.admitted_reads.lock();
        *reads -= 1;
        if *reads == 0 {
            self.0.reads_drained.notify_waiters();
        }
    }
}

impl SqliteStorage {
    /// Initialize storage over an owned SQLite database facade (`SqliteStorage.open`).
    pub async fn open(db: Arc<dyn SqliteDatabase>) -> Result<Self> {
        let opened = async {
            apply_sqlite_migrations(&*db, None).await?;
            let metadata = db
                .get(
                    "SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1",
                    vec![],
                )
                .await?
                .ok_or_else(|| Error::message("Durable SQLite metadata is missing"))?;
            parse_next_id(&metadata)
        }
        .await;
        match opened {
            Ok(next_id) => Ok(Self {
                inner: Arc::new(Inner {
                    db,
                    next_id: Mutex::new(next_id),
                    closed: AtomicBool::new(false),
                    closing: Mutex::new(None),
                    admitted_reads: Mutex::new(0),
                    reads_drained: Notify::new(),
                }),
            }),
            Err(error) => {
                // Preserve the initialization failure.
                let _ = db.close().await;
                Err(error)
            }
        }
    }

    fn assert_open(&self) -> Result<()> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(Error::message("SqliteStorage is closed"));
        }
        Ok(())
    }

    fn db(&self) -> &dyn SqliteDatabase {
        &*self.inner.db
    }

    /// Run a read that issues several queries. Close waits for admitted reads, so their later queries never reach a
    /// closed database. Single-query reads and transactions are already ordered before close by the database.
    async fn admit_read<T>(&self, read: impl Future<Output = Result<T>>) -> Result<T> {
        self.assert_open()?;
        *self.inner.admitted_reads.lock() += 1;
        let _admitted = AdmittedRead(&self.inner);
        read.await
    }

    async fn read_conversation(&self, id: ConversationId) -> Result<Option<ConversationRecord>> {
        read_conversation(self.db(), id).await
    }

    async fn read_entry(
        &self,
        conversation_id: Option<ConversationId>,
        id: EntryId,
    ) -> Result<Option<StoredEntry>> {
        let mut conversation = match conversation_id {
            Some(conversation_id) => Some(
                self.read_conversation(conversation_id)
                    .await?
                    .ok_or_else(|| {
                        Error::message(format!("Unknown conversation: {conversation_id}"))
                    })?,
            ),
            None => None,
        };
        let Some(row) = self
            .db()
            .get(
                "SELECT record, commit_seq FROM entries WHERE id = ?",
                sqlite_params![id.0],
            )
            .await?
        else {
            return Ok(None);
        };
        let entry: EntryRecord = record(&row)?;
        if let Some(current) = conversation.as_mut() {
            let mut upper_entry_id = u64::MAX;
            while current.id != entry.conversation_id {
                let Some(parent) = current.parent else {
                    return Ok(None);
                };
                upper_entry_id = upper_entry_id.min(parent.at.0);
                *current = self
                    .read_conversation(parent.conversation_id)
                    .await?
                    .ok_or_else(|| {
                        Error::message(format!("Unknown conversation: {}", parent.conversation_id))
                    })?;
            }
            if entry.id.0 > upper_entry_id {
                return Ok(None);
            }
        }
        let commit_seq = integer_column(&row, "commit_seq")?;
        Ok(Some(StoredEntry {
            entry,
            commit_seq: Seq(commit_seq as u64),
        }))
    }

    async fn read_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
    ) -> Result<Option<EntryRecord>> {
        let mut conversation = self
            .read_conversation(conversation_id)
            .await?
            .ok_or_else(|| Error::message(format!("Unknown conversation: {conversation_id}")))?;
        let mut upper = at_or_before_entry_id.map(|id| id.0);
        loop {
            let row = match upper {
                None => {
                    self.db()
                        .get(
                            "SELECT record FROM entries WHERE conversation_id = ? AND head IS NOT NULL ORDER BY id DESC LIMIT 1",
                            sqlite_params![conversation.id.0],
                        )
                        .await?
                }
                Some(upper) => {
                    self.db()
                        .get(
                            "SELECT record FROM entries WHERE conversation_id = ? AND head IS NOT NULL AND id <= ? ORDER BY id DESC LIMIT 1",
                            sqlite_params![conversation.id.0, upper],
                        )
                        .await?
                }
            };
            if let Some(row) = row {
                return Ok(Some(record(&row)?));
            }
            let Some(parent) = conversation.parent else {
                return Ok(None);
            };
            upper = Some(upper.map_or(parent.at.0, |upper| upper.min(parent.at.0)));
            conversation = self
                .read_conversation(parent.conversation_id)
                .await?
                .ok_or_else(|| {
                    Error::message(format!("Unknown conversation: {}", parent.conversation_id))
                })?;
        }
    }

    async fn read_entries(
        &self,
        query: &EntryQuery,
        limit: usize,
        cursor: Option<&Cursor>,
    ) -> Result<Page<EntryRecord>> {
        let mut conversation = self
            .read_conversation(query.conversation_id)
            .await?
            .ok_or_else(|| {
                Error::message(format!("Unknown conversation: {}", query.conversation_id))
            })?;
        let after = cursor_id(cursor)?;
        let mut upper = query.max_entry_id.map(|id| id.0);
        if let Some(after) = after {
            upper = Some(
                upper
                    .unwrap_or(MAX_SAFE_INTEGER)
                    .min(after.saturating_sub(1)),
            );
        }
        let mut values: Vec<EntryRecord> = Vec::new();
        loop {
            let mut clauses = vec!["conversation_id = ?"];
            let mut params = sqlite_params![conversation.id.0];
            if let Some(min) = query.min_entry_id {
                clauses.push("id >= ?");
                params.push(min.0.into());
            }
            if let Some(upper) = upper {
                clauses.push("id <= ?");
                params.push(upper.into());
            }
            params.push(((limit + 1 - values.len()) as u64).into());
            let rows = self
                .db()
                .all(
                    &format!(
                        "SELECT record FROM entries WHERE {} ORDER BY id DESC LIMIT ?",
                        clauses.join(" AND ")
                    ),
                    params,
                )
                .await?;
            for row in &rows {
                values.push(record(row)?);
            }
            let Some(parent) = conversation.parent else {
                break;
            };
            if values.len() > limit {
                break;
            }
            let next_upper = upper.map_or(parent.at.0, |upper| upper.min(parent.at.0));
            upper = Some(next_upper);
            if query.min_entry_id.is_some_and(|min| next_upper < min.0) {
                break;
            }
            conversation = self
                .read_conversation(parent.conversation_id)
                .await?
                .ok_or_else(|| {
                    Error::message(format!("Unknown conversation: {}", parent.conversation_id))
                })?;
        }
        page(values, limit, |entry| entry.id.0)
    }

    fn candidate_next_id(&self, writes: &[StorageWrite]) -> u64 {
        let mut next_id = *self.inner.next_id.lock();
        for write in writes {
            if let Some(id) = write_id(write) {
                next_id = next_id.max(id + 1);
            }
        }
        next_id
    }

    async fn close_database(inner: Arc<Inner>) -> Result<()> {
        loop {
            let drained = inner.reads_drained.notified();
            if *inner.admitted_reads.lock() == 0 {
                break;
            }
            drained.await;
        }
        inner.db.close().await
    }
}

fn parse_next_id(metadata: &SqliteRow) -> Result<u64> {
    let next_id = text_column(metadata, "next_id")?;
    next_id
        .parse::<u64>()
        .map_err(|_| Error::type_error(format!("Invalid durable SQLite next_id: {next_id}")))
}

async fn read_conversation(
    executor: &dyn SqliteExecutor,
    id: ConversationId,
) -> Result<Option<ConversationRecord>> {
    let row = executor
        .get(
            "SELECT record FROM conversations WHERE id = ?",
            sqlite_params![id.0],
        )
        .await?;
    row.as_ref().map(record).transpose()
}

async fn materialize_document(
    executor: &dyn SqliteExecutor,
    id: DocumentId,
    at: DocumentPoint,
) -> Result<Option<StoredDocument>> {
    let Some(row) = executor
        .get(
            "SELECT record FROM documents WHERE id = ?",
            sqlite_params![id.0],
        )
        .await?
    else {
        return Ok(None);
    };
    let record: DocumentRecord = record(&row)?;
    if at != DocumentPoint::Current && is_current_only(&record.scope, record.history) {
        return Err(Error::message(format!(
            "Document {id} does not retain historical content"
        )));
    }
    if !is_alive_at(&record, at) {
        return Ok(None);
    }
    let upper = point_binding(at);
    let base = executor
        .get(
            "SELECT seq, kind, version, content FROM document_revisions
				WHERE document_id = ? AND kind = 'base' AND seq <= ? ORDER BY seq DESC LIMIT 1",
            vec![id.0.into(), upper.clone()],
        )
        .await?
        .ok_or_else(|| Error::message(format!("Document {id} is missing a required base")))?;
    let base_seq = integer_column(&base, "seq")?;
    let base_version = integer_column(&base, "version")?;
    let mut value = JsonValue::Object(parse_json::<JsonObject>(text_column(&base, "content")?)?);
    let tail = executor
        .all(
            "SELECT seq, kind, version, content FROM document_revisions
				WHERE document_id = ? AND seq > ? AND seq <= ? ORDER BY seq",
            vec![id.0.into(), base_seq.into(), upper],
        )
        .await?;
    for revision in &tail {
        if text_column(revision, "kind")? != "delta"
            || integer_column(revision, "version")? != base_version
        {
            return Err(Error::message(format!(
                "Document {id} crosses a stored version boundary without a base"
            )));
        }
        let ops: Vec<Op> = parse_json(text_column(revision, "content")?)?;
        value = apply(value, &ops)?;
    }
    let JsonValue::Object(value) = value else {
        return Err(Error::message(format!(
            "Document {id} does not materialize a JSON object"
        )));
    };
    Ok(Some(StoredDocument {
        record,
        version: base_version as u32,
        value,
        deltas_since_base: tail.len() as u64,
    }))
}

async fn check_global_ids(executor: &dyn SqliteExecutor, writes: &[StorageWrite]) -> Result<()> {
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
        let existing = executor
            .get(
                "SELECT record_type FROM record_ids WHERE id = ?",
                sqlite_params![id],
            )
            .await?
            .map(|row| text_column(&row, "record_type").map(str::to_owned))
            .transpose()?;
        let earlier = claimed.get(&id).copied();
        if matches!(
            table,
            TableName::Conversation | TableName::Entry | TableName::Document
        ) {
            if let Some(existing) = existing {
                return Err(Error::message(format!(
                    "ID {id} already belongs to {existing}"
                )));
            }
            if earlier.is_some() {
                return Err(Error::message(format!("ID {id} is written more than once")));
            }
        } else {
            if let Some(existing) = existing.filter(|existing| existing != table.as_str()) {
                return Err(Error::message(format!(
                    "ID {id} already belongs to {existing}"
                )));
            }
            if earlier.is_some_and(|earlier| earlier != table) {
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
            StorageWrite::DocumentCreate { record, .. }
            | StorageWrite::DocumentCopy { record, .. } => record.id,
            StorageWrite::DocumentChange { id, .. } | StorageWrite::DocumentRetire { id } => *id,
            _ => continue,
        };
        let action = actions.entry(id.0).or_default();
        match write {
            StorageWrite::DocumentCreate { record, content } => {
                if action.create.is_some() || action.content.is_some() || action.copy.is_some() {
                    return Err(Error::message(format!(
                        "Document {id} has more than one content command"
                    )));
                }
                action.create = Some(record.clone());
                action.content = Some(content.clone());
            }
            StorageWrite::DocumentCopy { record, source } => {
                if action.create.is_some() || action.content.is_some() || action.copy.is_some() {
                    return Err(Error::message(format!(
                        "Document {id} has more than one content command"
                    )));
                }
                action.create = Some(record.clone());
                action.copy = Some(*source);
            }
            StorageWrite::DocumentChange { content, .. } => {
                if action.content.is_some() || action.copy.is_some() {
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
            _ => unreachable!("only document writes reach here"),
        }
    }
    Ok(actions)
}

async fn read_document_record(
    executor: &dyn SqliteExecutor,
    id: u64,
) -> Result<Option<DocumentRecord>> {
    let row = executor
        .get(
            "SELECT record FROM documents WHERE id = ?",
            sqlite_params![id],
        )
        .await?;
    row.as_ref().map(record).transpose()
}

async fn check_document_actions(
    executor: &dyn SqliteExecutor,
    actions: &IndexMap<u64, DocumentAction>,
) -> Result<()> {
    let mut live_counts: HashMap<String, i64> = HashMap::new();
    for (&id, action) in actions {
        if let Some(copy) = &action.copy
            && actions.contains_key(&copy.id.0)
        {
            return Err(StorageRejected::new(format!(
                "Document copy {id} source is changed in the copy batch"
            ))
            .into());
        }
        let existing = read_document_record(executor, id).await?;
        if action.create.is_none() && existing.is_none() {
            return Err(Error::message(format!("Unknown document: {id}")));
        }
        if action.create.is_some() && existing.is_some() {
            return Err(Error::message(format!("Document {id} already exists")));
        }
        if existing
            .as_ref()
            .is_some_and(|existing| existing.retired_at.is_some())
        {
            return Err(Error::message(format!("Document {id} is retired")));
        }
        if let Some(DocumentContent::Delta { version, .. }) = &action.content {
            let previous = executor
                .get(
                    "SELECT version FROM document_revisions WHERE document_id = ? ORDER BY seq DESC LIMIT 1",
                    sqlite_params![id],
                )
                .await?
                .ok_or_else(|| Error::message(format!("Document {id} delta has no base")))?;
            if integer_column(&previous, "version")? != i64::from(*version) {
                return Err(Error::message(format!(
                    "Document {id} version transition requires a base"
                )));
            }
        }
        let parts = match (&action.create, &existing) {
            (Some(create), _) => address_parts(&create.kind, &create.scope, create.key.as_deref()),
            (None, Some(existing)) => {
                address_parts(&existing.kind, &existing.scope, existing.key.as_deref())
            }
            (None, None) => unreachable!("checked above"),
        };
        let key = address_key(&parts);
        let mut live = match live_counts.get(&key) {
            Some(live) => *live,
            None => i64::from(current_document_id(executor, &parts).await?.is_some()),
        };
        if action.retire && existing.is_some() {
            live -= 1;
        }
        if action.create.is_some() && !action.retire {
            live += 1;
        }
        live_counts.insert(key, live);
    }
    if live_counts.values().any(|&live| live > 1) {
        return Err(Error::message(
            "Document address already has a current incarnation",
        ));
    }
    Ok(())
}

async fn current_document_id(
    executor: &dyn SqliteExecutor,
    parts: &AddressParts,
) -> Result<Option<DocumentId>> {
    let row = executor
        .get(
            "SELECT id FROM documents
				WHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ? AND retired_at IS NULL
				LIMIT 1",
            sqlite_params![
                parts.kind.as_str(),
                parts.scope_kind,
                parts.owner_id,
                parts.family,
                parts.key_value.as_str()
            ],
        )
        .await?;
    row.map(|row| integer_column(&row, "id").map(|id| DocumentId(id as u64)))
        .transpose()
}

async fn claim_id(executor: &dyn SqliteExecutor, id: u64, table: TableName) -> Result<()> {
    executor
        .run(
            "INSERT OR IGNORE INTO record_ids (id, record_type) VALUES (?, ?)",
            sqlite_params![id, table.as_str()],
        )
        .await
}

async fn apply_table_write(
    executor: &dyn SqliteExecutor,
    write: &StorageWrite,
    seq: Seq,
) -> Result<()> {
    match write {
        StorageWrite::Conversation { value } => {
            claim_id(executor, value.id.0, TableName::Conversation).await?;
            executor
                .run(
                    "INSERT INTO conversations (id, owner_conversation_id, owner_task_id, record) VALUES (?, ?, ?, ?)",
                    sqlite_params![
                        value.id.0,
                        value.owner.map(|owner| owner.conversation_id.0),
                        value.owner.map(|owner| owner.task_id.0),
                        encode_json(value)?
                    ],
                )
                .await
        }
        StorageWrite::Entry { value } => {
            claim_id(executor, value.id.0, TableName::Entry).await?;
            executor
                .run(
                    "INSERT INTO entries (id, conversation_id, head, commit_seq, record) VALUES (?, ?, ?, ?, ?)",
                    sqlite_params![
                        value.id.0,
                        value.conversation_id.0,
                        value.head.map(|head| head.0),
                        seq.0,
                        encode_json(value)?
                    ],
                )
                .await
        }
        StorageWrite::Task { value } => {
            claim_id(executor, value.id.0, TableName::Task).await?;
            executor
                .run(
                    "INSERT INTO tasks (id, conversation_id, kind, status, abort_requested, background, record)
						VALUES (?, ?, ?, ?, ?, ?, ?)
						ON CONFLICT(id) DO UPDATE SET conversation_id = excluded.conversation_id, kind = excluded.kind,
						status = excluded.status, abort_requested = excluded.abort_requested,
						background = excluded.background, record = excluded.record",
                    sqlite_params![
                        value.id.0,
                        value.conversation_id.0,
                        encode_indexed_string(&value.kind),
                        value.state.status().as_str(),
                        i64::from(value.abort_requested),
                        i64::from(value.background),
                        encode_json(value)?
                    ],
                )
                .await
        }
        StorageWrite::Submission { value } => {
            claim_id(executor, value.id.0, TableName::Submission).await?;
            executor
                .run(
                    "INSERT INTO submissions (id, conversation_id, request_id, status, record) VALUES (?, ?, ?, ?, ?)
						ON CONFLICT(id) DO UPDATE SET conversation_id = excluded.conversation_id,
						request_id = excluded.request_id, status = excluded.status, record = excluded.record",
                    sqlite_params![
                        value.id.0,
                        value.conversation_id.0,
                        value.request_id.as_deref().map(encode_indexed_string),
                        submission_status(value),
                        encode_json(value)?
                    ],
                )
                .await
        }
        StorageWrite::DocumentCreate { .. }
        | StorageWrite::DocumentCopy { .. }
        | StorageWrite::DocumentChange { .. }
        | StorageWrite::DocumentRetire { .. } => Ok(()),
    }
}

fn submission_status(value: &SubmissionRecord) -> String {
    match serde_json::to_value(value.status) {
        Ok(JsonValue::String(status)) => status,
        _ => unreachable!("submission statuses serialize as strings"),
    }
}

async fn apply_document_actions(
    executor: &dyn SqliteExecutor,
    actions: &IndexMap<u64, DocumentAction>,
    seq: Seq,
) -> Result<()> {
    for (&id, action) in actions {
        let mut content = action.content.clone();
        if let Some(copy) = &action.copy {
            let copied = async {
                let stored = materialize_document(executor, copy.id, copy.at)
                    .await?
                    .ok_or_else(|| {
                        Error::message(format!("Fork source document {} cannot be read", copy.id))
                    })?;
                let create = action.create.as_ref().expect("copies create a record");
                if !matches!(stored.record.scope, DocumentScope::Conversation { .. })
                    || !matches!(create.scope, DocumentScope::Conversation { .. })
                    || stored.record.kind != create.kind
                    || stored.record.key != create.key
                    || stored.record.history != create.history
                    || stored.record.fork != create.fork
                {
                    return Err(Error::message(format!(
                        "Fork source document {} does not match the copied record",
                        copy.id
                    )));
                }
                Ok(DocumentContent::Base {
                    version: stored.version,
                    value: stored.value,
                })
            }
            .await;
            content = Some(match copied {
                Ok(content) => content,
                Err(error @ Error::StorageRejected(_)) => return Err(error),
                Err(error) => {
                    return Err(StorageRejected::with_cause(
                        format!("Document copy {id} was rejected"),
                        error,
                    )
                    .into());
                }
            });
        }
        let mut record = match &action.create {
            Some(create) => {
                let mut record = create.stamp(seq);
                if action.retire {
                    record.retired_at = Some(seq);
                }
                let parts = address_parts(&record.kind, &record.scope, record.key.as_deref());
                claim_id(executor, id, TableName::Document).await?;
                executor
                    .run(
                        "INSERT INTO documents
						(id, kind, family, key_value, scope_kind, owner_id, created_at, retired_at, record)
						VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        sqlite_params![
                            id,
                            parts.kind,
                            parts.family,
                            parts.key_value,
                            parts.scope_kind,
                            parts.owner_id,
                            seq.0,
                            action.retire.then_some(seq.0),
                            encode_json(&record)?
                        ],
                    )
                    .await?;
                record
            }
            None => read_document_record(executor, id)
                .await?
                .ok_or_else(|| Error::message(format!("Unknown document: {id}")))?,
        };

        if let Some(content) = &content {
            if content.is_base() && is_current_only(&record.scope, record.history) {
                executor
                    .run(
                        "DELETE FROM document_revisions WHERE document_id = ?",
                        sqlite_params![id],
                    )
                    .await?;
            }
            let (kind, version, encoded) = match content {
                DocumentContent::Base { version, value } => ("base", *version, encode_json(value)?),
                DocumentContent::Delta { version, ops } => ("delta", *version, encode_json(ops)?),
            };
            executor
                .run(
                    "INSERT INTO document_revisions (document_id, seq, kind, version, content) VALUES (?, ?, ?, ?, ?)",
                    sqlite_params![id, seq.0, kind, version, encoded],
                )
                .await?;
        }

        if action.retire {
            if action.create.is_none() {
                record.retired_at = Some(seq);
                executor
                    .run(
                        "UPDATE documents SET retired_at = ?, record = ? WHERE id = ?",
                        sqlite_params![seq.0, encode_json(&record)?, id],
                    )
                    .await?;
            }
            if is_current_only(&record.scope, record.history) {
                executor
                    .run(
                        "DELETE FROM document_revisions WHERE document_id = ?",
                        sqlite_params![id],
                    )
                    .await?;
            }
        }
    }
    Ok(())
}

#[async_trait]
impl Storage for SqliteStorage {
    async fn commit(&self, writes: &[StorageWrite], _context: &Context) -> Result<Seq> {
        self.assert_open()?;
        let document_actions = prepare_document_actions(writes)?;
        let candidate_next_id = self.candidate_next_id(writes);
        let actions = &document_actions;
        let seq = self
            .db()
            .transaction(move |transaction| async move {
                let transaction = &*transaction;
                let metadata = transaction
                    .get(
                        "SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1",
                        vec![],
                    )
                    .await?
                    .ok_or_else(|| Error::message("Durable SQLite metadata is missing"))?;
                let stored_next_id = parse_next_id(&metadata)?;
                let committed_seq = Seq(integer_column(&metadata, "next_seq")? as u64);
                check_global_ids(transaction, writes).await?;
                check_document_actions(transaction, actions).await?;
                for write in writes {
                    apply_table_write(transaction, write, committed_seq).await?;
                }
                apply_document_actions(transaction, actions, committed_seq).await?;
                transaction
                    .run(
                        "UPDATE durable_metadata SET next_id = ?, next_seq = ? WHERE singleton = 1",
                        sqlite_params![
                            stored_next_id.max(candidate_next_id).to_string(),
                            committed_seq.0 + 1
                        ],
                    )
                    .await?;
                Ok(committed_seq)
            })
            .await?;
        let mut next_id = self.inner.next_id.lock();
        *next_id = (*next_id).max(candidate_next_id);
        Ok(seq)
    }

    async fn mint_id(&self) -> Result<u64> {
        self.assert_open()?;
        let mut next_id = self.inner.next_id.lock();
        if *next_id > MAX_SAFE_INTEGER {
            return Err(Error::message("ID space is exhausted"));
        }
        let id = *next_id;
        *next_id += 1;
        Ok(id)
    }

    async fn conversation(
        &self,
        id: ConversationId,
        _context: &Context,
    ) -> Result<Option<ConversationRecord>> {
        self.assert_open()?;
        self.read_conversation(id).await
    }

    async fn scan_conversations(
        &self,
        query: &ConversationQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<ConversationRecord>> {
        self.assert_open()?;
        let mut clauses = vec!["id > ?"];
        let mut params = vec![cursor_binding(cursor)?];
        if let Some(owner) = query.owner_conversation_id {
            clauses.push("owner_conversation_id = ?");
            params.push(owner.0.into());
        }
        if let Some(owner) = query.owner_task_id {
            clauses.push("owner_task_id = ?");
            params.push(owner.0.into());
        }
        params.push(((limit + 1) as u64).into());
        let rows = self
            .db()
            .all(
                &format!(
                    "SELECT record FROM conversations WHERE {} ORDER BY id LIMIT ?",
                    clauses.join(" AND ")
                ),
                params,
            )
            .await?;
        let values = rows
            .iter()
            .map(record)
            .collect::<Result<Vec<ConversationRecord>>>()?;
        page(values, limit, |conversation| conversation.id.0)
    }

    async fn entry(&self, id: EntryId, _context: &Context) -> Result<Option<StoredEntry>> {
        self.admit_read(self.read_entry(None, id)).await
    }

    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        _context: &Context,
    ) -> Result<Option<StoredEntry>> {
        self.admit_read(self.read_entry(Some(conversation_id), id))
            .await
    }

    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
        _context: &Context,
    ) -> Result<Option<EntryRecord>> {
        self.admit_read(self.read_latest_head_marker(conversation_id, at_or_before_entry_id))
            .await
    }

    async fn scan_entries(
        &self,
        query: &EntryQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<EntryRecord>> {
        self.admit_read(self.read_entries(query, limit, cursor))
            .await
    }

    async fn task(&self, id: TaskId, _context: &Context) -> Result<Option<TaskRecord>> {
        self.assert_open()?;
        let row = self
            .db()
            .get(
                "SELECT record FROM tasks WHERE id = ?",
                sqlite_params![id.0],
            )
            .await?;
        row.as_ref().map(record).transpose()
    }

    async fn scan_tasks(
        &self,
        query: &TaskQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<TaskRecord>> {
        self.assert_open()?;
        let mut clauses = vec!["id > ?"];
        let mut params = vec![cursor_binding(cursor)?];
        if let Some(conversation_id) = query.conversation_id {
            clauses.push("conversation_id = ?");
            params.push(conversation_id.0.into());
        }
        if let Some(kind) = &query.kind {
            clauses.push("kind = ?");
            params.push(encode_indexed_string(kind).into());
        }
        if let Some(status) = query.status {
            clauses.push("status = ?");
            params.push(status.as_str().into());
        }
        if let Some(abort_requested) = query.abort_requested {
            clauses.push("abort_requested = ?");
            params.push(i64::from(abort_requested).into());
        }
        if let Some(background) = query.background {
            clauses.push("background = ?");
            params.push(i64::from(background).into());
        }
        params.push(((limit + 1) as u64).into());
        let rows = self
            .db()
            .all(
                &format!(
                    "SELECT record FROM tasks WHERE {} ORDER BY id LIMIT ?",
                    clauses.join(" AND ")
                ),
                params,
            )
            .await?;
        let values = rows
            .iter()
            .map(record)
            .collect::<Result<Vec<TaskRecord>>>()?;
        page(values, limit, |task| task.id.0)
    }

    async fn submission(
        &self,
        id: SubmissionId,
        _context: &Context,
    ) -> Result<Option<SubmissionRecord>> {
        self.assert_open()?;
        let row = self
            .db()
            .get(
                "SELECT record FROM submissions WHERE id = ?",
                sqlite_params![id.0],
            )
            .await?;
        row.as_ref().map(record).transpose()
    }

    async fn scan_submissions(
        &self,
        query: &SubmissionQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<SubmissionRecord>> {
        self.assert_open()?;
        let mut clauses = vec!["id > ?"];
        let mut params = vec![cursor_binding(cursor)?];
        if let Some(conversation_id) = query.conversation_id {
            clauses.push("conversation_id = ?");
            params.push(conversation_id.0.into());
        }
        if let Some(status) = query.status {
            clauses.push("status = ?");
            params.push(
                match serde_json::to_value(status)? {
                    JsonValue::String(status) => status,
                    _ => unreachable!("submission statuses serialize as strings"),
                }
                .into(),
            );
        }
        params.push(((limit + 1) as u64).into());
        let rows = self
            .db()
            .all(
                &format!(
                    "SELECT record FROM submissions WHERE {} ORDER BY id LIMIT ?",
                    clauses.join(" AND ")
                ),
                params,
            )
            .await?;
        let values = rows
            .iter()
            .map(record)
            .collect::<Result<Vec<SubmissionRecord>>>()?;
        page(values, limit, |submission| submission.id.0)
    }

    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        _context: &Context,
    ) -> Result<Option<SubmissionRecord>> {
        self.assert_open()?;
        let row = self
            .db()
            .get(
                "SELECT record FROM submissions WHERE conversation_id = ? AND request_id = ?",
                sqlite_params![conversation_id.0, encode_indexed_string(request_id)],
            )
            .await?;
        row.as_ref().map(record).transpose()
    }

    async fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        _context: &Context,
    ) -> Result<Option<DocumentRecord>> {
        self.assert_open()?;
        let parts = address_parts(&address.kind, &address.scope, address.key.as_deref());
        let mut params = sqlite_params![
            parts.kind,
            parts.scope_kind,
            parts.owner_id,
            parts.family,
            parts.key_value
        ];
        let sql = match at {
            DocumentPoint::Current => {
                "SELECT record FROM documents
					WHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ?
					AND retired_at IS NULL ORDER BY created_at DESC LIMIT 1"
            }
            DocumentPoint::Seq(at) => {
                params.push(at.0.into());
                params.push(at.0.into());
                "SELECT record FROM documents
					WHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ?
					AND created_at <= ? AND (retired_at IS NULL OR retired_at > ?)
					ORDER BY created_at DESC LIMIT 1"
            }
        };
        let row = self.db().get(sql, params).await?;
        row.as_ref().map(record).transpose()
    }

    async fn document(
        &self,
        id: DocumentId,
        at: DocumentPoint,
        _context: &Context,
    ) -> Result<Option<StoredDocument>> {
        self.assert_open()?;
        // The record and revision queries must observe one committed state; a commit between them can replace the base.
        self.db()
            .transaction(move |transaction| async move {
                materialize_document(&*transaction, id, at).await
            })
            .await
    }

    async fn scan_documents(
        &self,
        query: &DocumentQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        _context: &Context,
    ) -> Result<Page<DocumentRecord>> {
        self.assert_open()?;
        let scope = scope_columns(&query.scope);
        let mut clauses = vec!["scope_kind = ?", "owner_id = ?", "id > ?"];
        let mut params = vec![
            scope.scope_kind.into(),
            scope.owner_id.into(),
            cursor_binding(cursor)?,
        ];
        if let Some(kind) = &query.kind {
            clauses.push("kind = ?");
            params.push(encode_indexed_string(kind).into());
        }
        match query.at {
            DocumentPoint::Current => clauses.push("retired_at IS NULL"),
            DocumentPoint::Seq(at) => {
                clauses.push("created_at <= ?");
                clauses.push("(retired_at IS NULL OR retired_at > ?)");
                params.push(at.0.into());
                params.push(at.0.into());
            }
        }
        params.push(((limit + 1) as u64).into());
        let rows = self
            .db()
            .all(
                &format!(
                    "SELECT record FROM documents WHERE {} ORDER BY id LIMIT ?",
                    clauses.join(" AND ")
                ),
                params,
            )
            .await?;
        let values = rows
            .iter()
            .map(record)
            .collect::<Result<Vec<DocumentRecord>>>()?;
        page(values, limit, |document| document.id.0)
    }

    async fn close(&self, _context: &Context) -> Result<()> {
        let closing = {
            let mut closing = self.inner.closing.lock();
            closing
                .get_or_insert_with(|| {
                    self.inner.closed.store(true, Ordering::SeqCst);
                    Self::close_database(self.inner.clone()).boxed().shared()
                })
                .clone()
        };
        closing.await
    }
}
