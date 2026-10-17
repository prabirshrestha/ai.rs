//! Ports of `test/sqlite-storage.test.ts`, `test/sqlite-facade.test.ts` and
//! `test/sqlite-migrations.test.ts`, plus the cross-runtime check against a
//! database written by Pi's TypeScript `SqliteStorage` (`fixtures/`).
//!
//! Divergences: TS `DatabaseSync` inspection uses a plain `rusqlite`
//! connection; the circular-JSON rollback case (unrepresentable in Rust) fails
//! the batch with a rejected document copy after its table rows were written;
//! the prepare-counting case keeps its behavior checks but cannot count
//! `rusqlite` prepares; promise call-order cases poll futures with
//! `tokio::join!` or spawn them, since enqueueing happens when an operation is
//! called.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ::rusqlite::{Connection, OpenFlags};
use async_trait::async_trait;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde_json::json;

use crate::chord::{Context, JsonValue};
use crate::durable::errors::{Error, Result, StorageRejected};
use crate::durable::ids::{
    ConversationId, DocumentId, EntryId, ROOT_CONVERSATION_ID, Seq, SubmissionId, TaskId,
};
use crate::durable::storage::test_support::{
    TempDir, context, document_json, entry_json, json, page_json, pending_task, root_write, writes,
};
use crate::durable::testing::StorageConformanceProvider;
use crate::durable::types::{
    ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentPoint, DocumentQuery,
    DocumentRecord, DocumentScope, EntryQuery, EntryRecord, Page, Storage, StorageWrite,
    StoredDocument, StoredEntry, SubmissionQuery, SubmissionRecord, SubmissionStatus, TaskQuery,
    TaskRecord, TaskStatus,
};

use super::rusqlite::{
    RusqliteDatabase, RusqliteStorageOptions, open_rusqlite_database, open_rusqlite_storage,
};
use super::{
    AggregateError, CURRENT_SQLITE_SCHEMA_VERSION, SqliteDatabase, SqliteDatabaseExt,
    SqliteExecutor, SqliteMigration, SqliteRow, SqliteStorage, SqliteTransactionCallback,
    SqliteValue, apply_sqlite_migrations, sqlite_migrations,
};

fn assert_rejects<T: std::fmt::Debug>(result: Result<T>, message_includes: &str) -> Error {
    match result {
        Ok(value) => panic!("expected a rejection including {message_includes:?}, got {value:?}"),
        Err(error) => {
            assert!(
                error.to_string().contains(message_includes),
                "expected {:?} to include {message_includes:?}",
                error.to_string()
            );
            error
        }
    }
}

fn row(values: &[(&str, SqliteValue)]) -> SqliteRow {
    values
        .iter()
        .map(|(name, value)| ((*name).to_owned(), value.clone()))
        .collect()
}

fn int(value: i64) -> SqliteValue {
    SqliteValue::Integer(value)
}

async fn create_sqlite_storage(
    directory: &TempDir,
    options: RusqliteStorageOptions,
) -> (SqliteStorage, std::path::PathBuf) {
    let path = directory.join("storage.sqlite");
    let storage = open_rusqlite_storage(&path, options).await.unwrap();
    (storage, path)
}

fn read_only(path: &std::path::Path) -> Connection {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap()
}

fn scalar(connection: &Connection, sql: &str) -> i64 {
    connection.query_row(sql, [], |row| row.get(0)).unwrap()
}

fn revision_count(path: &std::path::Path, document_id: u64) -> i64 {
    read_only(path)
        .query_row(
            "SELECT count(*) AS count FROM document_revisions WHERE document_id = ?",
            [document_id as i64],
            |row| row.get(0),
        )
        .unwrap()
}

fn entry(id: u64, conversation_id: u64, data: Option<JsonValue>) -> JsonValue {
    let mut value = json!({ "id": id, "conversationId": conversation_id, "kind": "message" });
    if let Some(data) = data {
        value["data"] = data;
    }
    value
}

async fn create_root(storage: &dyn Storage) -> Seq {
    storage.commit(&root_write(), context()).await.unwrap()
}

async fn commit(storage: &dyn Storage, value: JsonValue) -> Result<Seq> {
    storage.commit(&writes(value), context()).await
}

// ─── Conformance ─────────────────────────────────────────────────────────────

mod sqlite_storage {
    use super::*;

    crate::storage_conformance_tests!({
        let provider: StorageConformanceProvider = Arc::new(|test| {
            Box::pin(async move {
                let directory = TempDir::new("pi-durable-sqlite-");
                let storage =
                    open_rusqlite_storage(directory.join("storage.sqlite"), Default::default())
                        .await
                        .unwrap();
                test(Arc::new(storage.clone())).await;
                storage.close(context()).await.unwrap();
            })
        });
        provider
    });
}

/// Closes and reopens the file after every commit (`ReopeningStorage`).
struct ReopeningStorage {
    current: Mutex<SqliteStorage>,
    path: std::path::PathBuf,
    closed: AtomicBool,
}

impl ReopeningStorage {
    fn current(&self) -> SqliteStorage {
        self.current.lock().clone()
    }
}

#[async_trait]
impl Storage for ReopeningStorage {
    async fn commit(&self, writes: &[StorageWrite], commit_context: &Context) -> Result<Seq> {
        let current = self.current();
        if self.closed.load(Ordering::SeqCst) {
            return current.commit(writes, commit_context).await;
        }
        let result = current.commit(writes, commit_context).await;
        current.close(context()).await.unwrap();
        *self.current.lock() = open_rusqlite_storage(&self.path, Default::default())
            .await
            .unwrap();
        result
    }

    async fn mint_id(&self) -> Result<u64> {
        self.current().mint_id().await
    }

    async fn conversation(
        &self,
        id: ConversationId,
        c: &Context,
    ) -> Result<Option<ConversationRecord>> {
        self.current().conversation(id, c).await
    }

    async fn scan_conversations(
        &self,
        query: &ConversationQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        c: &Context,
    ) -> Result<Page<ConversationRecord>> {
        self.current()
            .scan_conversations(query, limit, cursor, c)
            .await
    }

    async fn entry(&self, id: EntryId, c: &Context) -> Result<Option<StoredEntry>> {
        self.current().entry(id, c).await
    }

    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        c: &Context,
    ) -> Result<Option<StoredEntry>> {
        self.current().entry_in(conversation_id, id, c).await
    }

    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at: Option<EntryId>,
        c: &Context,
    ) -> Result<Option<EntryRecord>> {
        self.current()
            .find_latest_head_marker(conversation_id, at, c)
            .await
    }

    async fn scan_entries(
        &self,
        query: &EntryQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        c: &Context,
    ) -> Result<Page<EntryRecord>> {
        self.current().scan_entries(query, limit, cursor, c).await
    }

    async fn task(&self, id: TaskId, c: &Context) -> Result<Option<TaskRecord>> {
        self.current().task(id, c).await
    }

    async fn scan_tasks(
        &self,
        query: &TaskQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        c: &Context,
    ) -> Result<Page<TaskRecord>> {
        self.current().scan_tasks(query, limit, cursor, c).await
    }

    async fn submission(&self, id: SubmissionId, c: &Context) -> Result<Option<SubmissionRecord>> {
        self.current().submission(id, c).await
    }

    async fn scan_submissions(
        &self,
        query: &SubmissionQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        c: &Context,
    ) -> Result<Page<SubmissionRecord>> {
        self.current()
            .scan_submissions(query, limit, cursor, c)
            .await
    }

    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        c: &Context,
    ) -> Result<Option<SubmissionRecord>> {
        self.current()
            .submission_by_request(conversation_id, request_id, c)
            .await
    }

    async fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        c: &Context,
    ) -> Result<Option<DocumentRecord>> {
        self.current().find_document(address, at, c).await
    }

    async fn document(
        &self,
        id: DocumentId,
        at: DocumentPoint,
        c: &Context,
    ) -> Result<Option<StoredDocument>> {
        self.current().document(id, at, c).await
    }

    async fn scan_documents(
        &self,
        query: &DocumentQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        c: &Context,
    ) -> Result<Page<DocumentRecord>> {
        self.current().scan_documents(query, limit, cursor, c).await
    }

    async fn close(&self, c: &Context) -> Result<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        self.current().close(c).await
    }
}

mod sqlite_storage_across_reopen {
    use super::*;

    crate::storage_conformance_tests!({
        let provider: StorageConformanceProvider = Arc::new(|test| {
            Box::pin(async move {
                let directory = TempDir::new("pi-durable-sqlite-conformance-");
                let path = directory.join("storage.sqlite");
                let created = open_rusqlite_storage(&path, Default::default())
                    .await
                    .unwrap();
                created.close(context()).await.unwrap();
                let storage = Arc::new(ReopeningStorage {
                    current: Mutex::new(
                        open_rusqlite_storage(&path, Default::default())
                            .await
                            .unwrap(),
                    ),
                    path,
                    closed: AtomicBool::new(false),
                });
                test(storage.clone()).await;
                storage.close(context()).await.unwrap();
            })
        });
        provider
    });
}

// ─── Pico SqliteStorage ──────────────────────────────────────────────────────

#[tokio::test]
async fn persists_records_sequence_allocation_and_global_id_allocation_across_reopen() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    assert_eq!(create_root(&storage).await, Seq(1));
    let entry_id = storage.mint_id().await.unwrap();
    assert_eq!(
        commit(
            &storage,
            json!([{ "type": "entry", "value": entry(entry_id, 1, None) }])
        )
        .await
        .unwrap(),
        Seq(2)
    );
    storage.close(context()).await.unwrap();

    let reopened = open_rusqlite_storage(&path, Default::default())
        .await
        .unwrap();
    assert_eq!(
        entry_json(
            &reopened
                .entry(EntryId(entry_id), context())
                .await
                .unwrap()
                .unwrap()
        ),
        json!({ "entry": entry(entry_id, 1, None), "commitSeq": 2 })
    );
    assert_eq!(reopened.mint_id().await.unwrap(), entry_id + 1);
    let task_id = reopened.mint_id().await.unwrap();
    assert_eq!(
        commit(
            &reopened,
            json!([{ "type": "task", "value": pending_task(task_id, "ready") }])
        )
        .await
        .unwrap(),
        Seq(3)
    );
    reopened.close(context()).await.unwrap();
}

#[tokio::test]
async fn rejects_persisted_metadata_corruption_on_reopen() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    storage.close(context()).await.unwrap();
    Connection::open(&path)
        .unwrap()
        .execute_batch("DELETE FROM durable_metadata")
        .unwrap();
    assert_rejects(
        open_rusqlite_storage(&path, Default::default()).await,
        "Durable SQLite metadata is missing",
    );
}

#[tokio::test]
async fn rejects_a_document_whose_required_base_is_missing() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    create_root(&storage).await;
    let id = storage.mint_id().await.unwrap();
    commit(
        &storage,
        json!([{
            "type": "document.create",
            "record": { "id": id, "kind": "corrupt", "scope": { "kind": "session" } },
            "content": { "kind": "base", "version": 1, "value": { "retained": true } },
        }]),
    )
    .await
    .unwrap();
    Connection::open(&path)
        .unwrap()
        .execute(
            "DELETE FROM document_revisions WHERE document_id = ?",
            [id as i64],
        )
        .unwrap();
    assert_rejects(
        storage
            .document(DocumentId(id), DocumentPoint::Current, context())
            .await,
        &format!("Document {id} is missing a required base"),
    );
    storage.close(context()).await.unwrap();
}

#[tokio::test]
async fn replays_detached_root_replacements_and_follow_up_edits_while_rejecting_corrupt_operations()
{
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    create_root(&storage).await;
    let id = storage.mint_id().await.unwrap();
    for value in [
        json!([{
            "type": "document.create",
            "record": { "id": id, "kind": "replay", "scope": { "kind": "session" } },
            "content": { "kind": "base", "version": 1, "value": { "nested": { "value": 1 }, "rows": [] } },
        }]),
        json!([{
            "type": "document.change",
            "id": id,
            "content": { "kind": "delta", "version": 1, "ops": [["r", { "nested": { "value": 2 }, "rows": [{ "id": 1 }] }]] },
        }]),
        json!([{
            "type": "document.change",
            "id": id,
            "content": {
                "kind": "delta",
                "version": 1,
                "ops": [["s", ["nested", "value"], 3], ["p", ["rows"], 1, 0, [{ "id": 2 }]], ["m", ["rows"], [1, 0]]],
            },
        }]),
        json!([{
            "type": "document.change",
            "id": id,
            "content": { "kind": "delta", "version": 1, "ops": [["s", ["nested", "value"], 4]] },
        }]),
    ] {
        commit(&storage, value).await.unwrap();
    }

    let expected = json!({ "nested": { "value": 4 }, "rows": [{ "id": 2 }, { "id": 1 }] });
    let mut first = storage
        .document(DocumentId(id), DocumentPoint::Current, context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(JsonValue::Object(first.value.clone()), expected);
    first.value["nested"]["value"] = json!(99);
    first.value["rows"][0]["id"] = json!(99);
    let second = storage
        .document(DocumentId(id), DocumentPoint::Current, context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(JsonValue::Object(second.value), expected);

    Connection::open(&path)
        .unwrap()
        .execute(
            "UPDATE document_revisions SET content = ? WHERE document_id = ? AND seq =
				(SELECT max(seq) FROM document_revisions WHERE document_id = ?)",
            ::rusqlite::params![r#"[["unknown"]]"#, id as i64, id as i64],
        )
        .unwrap();
    assert_rejects(
        storage
            .document(DocumentId(id), DocumentPoint::Current, context())
            .await,
        "unknown op verb",
    );
    storage.close(context()).await.unwrap();
}

#[tokio::test]
async fn rolls_sql_rows_and_sequence_allocation_back_as_one_transaction() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    create_root(&storage).await;
    let transient_id = storage.mint_id().await.unwrap();
    let transient_task_id = storage.mint_id().await.unwrap();
    let copy_id = storage.mint_id().await.unwrap();
    // Divergence: TS fails JSON encoding of a circular task input; Rust fails
    // the batch after its table rows were written, with an unreadable copy source.
    let error = assert_rejects(
        commit(
            &storage,
            json!([
                { "type": "entry", "value": entry(transient_id, 1, None) },
                { "type": "task", "value": pending_task(transient_task_id, "ready") },
                {
                    "type": "document.copy",
                    "record": {
                        "id": copy_id,
                        "kind": "copy",
                        "scope": { "kind": "conversation", "conversationId": 1 },
                        "history": "rewindable",
                        "fork": "asOf",
                    },
                    "source": { "id": 999, "at": 1 },
                },
            ]),
        )
        .await,
        &format!("Document copy {copy_id} was rejected"),
    );
    assert!(error.is_storage_rejected());
    assert!(
        storage
            .entry(EntryId(transient_id), context())
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        storage
            .task(TaskId::new(transient_task_id), context())
            .await
            .unwrap()
            .is_none()
    );
    let committed_id = storage.mint_id().await.unwrap();
    assert_eq!(
        commit(
            &storage,
            json!([{ "type": "entry", "value": entry(committed_id, 1, None) }])
        )
        .await
        .unwrap(),
        Seq(2)
    );

    let db = read_only(&path);
    assert_eq!(scalar(&db, "SELECT count(*) AS value FROM entries"), 1);
    assert_eq!(
        scalar(
            &db,
            "SELECT next_seq AS value FROM durable_metadata WHERE singleton = 1"
        ),
        3
    );
    drop(db);
    storage.close(context()).await.unwrap();
}

#[tokio::test]
async fn reconstructs_recent_and_ancient_rewindable_points_after_reopen() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    create_root(&storage).await;
    let id = storage.mint_id().await.unwrap();
    let created_at = commit(
        &storage,
        json!([{
            "type": "document.create",
            "record": {
                "id": id,
                "kind": "history",
                "scope": { "kind": "conversation", "conversationId": 1 },
                "history": "rewindable",
                "fork": "asOf",
            },
            "content": { "kind": "base", "version": 1, "value": { "count": 0 } },
        }]),
    )
    .await
    .unwrap();
    let mut ancient_at = created_at;
    let mut recent_at = created_at;
    for count in 1..=40 {
        let content = if count == 20 {
            json!({ "kind": "base", "version": 1, "value": { "count": count } })
        } else {
            json!({ "kind": "delta", "version": 1, "ops": [["s", ["count"], count]] })
        };
        recent_at = commit(
            &storage,
            json!([{ "type": "document.change", "id": id, "content": content }]),
        )
        .await
        .unwrap();
        if count == 5 {
            ancient_at = recent_at;
        }
    }
    storage.close(context()).await.unwrap();
    let reopened = open_rusqlite_storage(&path, Default::default())
        .await
        .unwrap();
    let value = |stored: Option<StoredDocument>| JsonValue::Object(stored.unwrap().value);
    assert_eq!(
        value(
            reopened
                .document(DocumentId(id), ancient_at.into(), context())
                .await
                .unwrap()
        ),
        json!({ "count": 5 })
    );
    assert_eq!(
        value(
            reopened
                .document(DocumentId(id), recent_at.into(), context())
                .await
                .unwrap()
        ),
        json!({ "count": 40 })
    );
    reopened.close(context()).await.unwrap();
}

#[tokio::test]
async fn uses_indexes_for_exact_addresses_exact_scopes_entry_history_and_document_revision_tails() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    storage.close(context()).await.unwrap();
    let db = read_only(&path);
    let plan = |sql: &str, params: Vec<::rusqlite::types::Value>| -> String {
        let mut statement = db.prepare(&format!("EXPLAIN QUERY PLAN {sql}")).unwrap();
        let details = statement
            .query_map(::rusqlite::params_from_iter(params), |row| {
                row.get::<_, String>("detail")
            })
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        details.join("\n")
    };
    use ::rusqlite::types::Value::{Integer as I, Text as T};
    let details = [
        plan(
            "SELECT record FROM documents
				WHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ?
				AND retired_at IS NULL ORDER BY created_at DESC LIMIT 1",
            vec![
                T("kind".into()),
                T("session".into()),
                I(0),
                I(0),
                T(String::new()),
            ],
        ),
        plan(
            "SELECT record FROM documents
				WHERE kind = ? AND scope_kind = ? AND owner_id = ? AND family = ? AND key_value = ?
				AND created_at <= ? AND (retired_at IS NULL OR retired_at > ?)
				ORDER BY created_at DESC LIMIT 1",
            vec![
                T("kind".into()),
                T("conversation".into()),
                I(1),
                I(0),
                T(String::new()),
                I(10),
                I(10),
            ],
        ),
        plan(
            "SELECT record FROM documents
				WHERE scope_kind = ? AND owner_id = ? AND kind = ? AND id > ? ORDER BY id LIMIT ?",
            vec![T("task".into()), I(1), T("kind".into()), I(0), I(10)],
        ),
        plan(
            "SELECT record FROM entries
				WHERE conversation_id = ? AND id <= ? ORDER BY id DESC LIMIT ?",
            vec![I(1), I(10), I(10)],
        ),
        plan(
            "SELECT record FROM entries
				WHERE conversation_id = ? AND head IS NOT NULL AND id <= ? ORDER BY id DESC LIMIT 1",
            vec![I(1), I(10)],
        ),
        plan(
            "SELECT record FROM tasks WHERE status = ? AND id > ? ORDER BY id LIMIT ?",
            vec![T("pending".into()), I(0), I(10)],
        ),
        plan(
            "SELECT seq, kind, version, content FROM document_revisions
				WHERE document_id = ? AND kind = 'base' AND seq <= ? ORDER BY seq DESC LIMIT 1",
            vec![I(1), I(10)],
        ),
        plan(
            "SELECT seq, kind, version, content FROM document_revisions
				WHERE document_id = ? AND seq > ? AND seq <= ? ORDER BY seq",
            vec![I(1), I(5), I(10)],
        ),
    ];
    assert!(
        details[0].contains("documents_by_address"),
        "{}",
        details[0]
    );
    assert!(
        details[1].contains("documents_by_address"),
        "{}",
        details[1]
    );
    assert!(
        details[2].contains("documents_by_scope_kind"),
        "{}",
        details[2]
    );
    assert!(
        details[3].contains("entries_by_conversation"),
        "{}",
        details[3]
    );
    assert!(
        details[4].contains("entry_heads_by_conversation"),
        "{}",
        details[4]
    );
    assert!(details[5].contains("tasks_by_status"), "{}", details[5]);
    assert!(
        details[6].contains("document_revisions_by_kind"),
        "{}",
        details[6]
    );
    assert!(
        details[7].contains("sqlite_autoindex_document_revisions_1"),
        "{}",
        details[7]
    );
    for detail in &details[..3] {
        assert!(!detail.contains("SCAN documents"), "{detail}");
    }
    assert!(!details[7].contains("SCAN document_revisions"));
}

#[tokio::test]
async fn reclaims_current_only_revisions_only_after_a_base_or_retirement() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    create_root(&storage).await;
    let id = storage.mint_id().await.unwrap();
    commit(
        &storage,
        json!([{
            "type": "document.create",
            "record": { "id": id, "kind": "latest", "scope": { "kind": "session" } },
            "content": { "kind": "base", "version": 1, "value": { "count": 0 } },
        }]),
    )
    .await
    .unwrap();
    for count in 1..=10 {
        commit(
            &storage,
            json!([{ "type": "document.change", "id": id, "content": { "kind": "delta", "version": 1, "ops": [["s", ["count"], count]] } }]),
        )
        .await
        .unwrap();
    }
    assert_eq!(revision_count(&path, id), 11);
    let content: String = read_only(&path)
        .query_row(
            "SELECT content FROM document_revisions WHERE document_id = ? AND kind = 'delta' ORDER BY seq DESC LIMIT 1",
            [id as i64],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<JsonValue>(&content).unwrap(),
        json!([["s", ["count"], 10]])
    );
    commit(
        &storage,
        json!([{ "type": "document.change", "id": id, "content": { "kind": "base", "version": 1, "value": { "count": 11 } } }]),
    )
    .await
    .unwrap();
    assert_eq!(revision_count(&path, id), 1);
    commit(
        &storage,
        json!([{ "type": "document.change", "id": id, "content": { "kind": "delta", "version": 1, "ops": [["r", { "count": 12 }]] } }]),
    )
    .await
    .unwrap();
    assert_eq!(revision_count(&path, id), 2);
    commit(&storage, json!([{ "type": "document.retire", "id": id }]))
        .await
        .unwrap();
    assert_eq!(revision_count(&path, id), 0);
    storage.close(context()).await.unwrap();
}

#[tokio::test]
async fn auto_checkpoints_wal_frames_and_truncates_the_wal_on_close() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(
        &directory,
        RusqliteStorageOptions {
            wal_auto_checkpoint_pages: Some(1),
            busy_timeout_ms: None,
        },
    )
    .await;
    create_root(&storage).await;
    for index in 0..20 {
        let id = storage.mint_id().await.unwrap();
        commit(
            &storage,
            json!([{ "type": "entry", "value": entry(id, 1, Some(json!({ "text": "x".repeat(32 * 1024), "index": index }))) }]),
        )
        .await
        .unwrap();
    }
    let wal_path = format!("{}-wal", path.display());
    assert!(std::fs::metadata(&wal_path).unwrap().len() < 512 * 1024);
    let observer = read_only(&path);
    assert_eq!(
        scalar(&observer, "SELECT count(*) AS value FROM entries"),
        20
    );
    storage.close(context()).await.unwrap();
    assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), 0);
    drop(observer);
}

#[tokio::test]
async fn reuses_pages_released_by_current_only_checkpoints() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    create_root(&storage).await;
    let id = storage.mint_id().await.unwrap();
    let large = "x".repeat(512 * 1024);
    commit(
        &storage,
        json!([{
            "type": "document.create",
            "record": { "id": id, "kind": "reuse", "scope": { "kind": "session" } },
            "content": { "kind": "base", "version": 1, "value": { "text": large } },
        }]),
    )
    .await
    .unwrap();
    commit(
        &storage,
        json!([{ "type": "document.change", "id": id, "content": { "kind": "base", "version": 1, "value": { "text": "small" } } }]),
    )
    .await
    .unwrap();
    let before = read_only(&path);
    let pages_after_delete = scalar(
        &before,
        "SELECT page_count AS value FROM pragma_page_count() ",
    );
    let free_after_delete = scalar(
        &before,
        "SELECT freelist_count AS value FROM pragma_freelist_count() ",
    );
    drop(before);
    assert!(free_after_delete > 0);
    commit(
        &storage,
        json!([{ "type": "document.change", "id": id, "content": { "kind": "base", "version": 1, "value": { "text": large } } }]),
    )
    .await
    .unwrap();
    let after = read_only(&path);
    let pages_after_reuse = scalar(
        &after,
        "SELECT page_count AS value FROM pragma_page_count() ",
    );
    let free_after_reuse = scalar(
        &after,
        "SELECT freelist_count AS value FROM pragma_freelist_count() ",
    );
    drop(after);
    assert!(pages_after_reuse <= pages_after_delete + 2);
    assert!(free_after_reuse < free_after_delete);
    storage.close(context()).await.unwrap();
}

#[tokio::test]
async fn keeps_representative_row_and_document_storage_bounded() {
    let directory = TempDir::new("pi-durable-sqlite-");
    let (storage, path) = create_sqlite_storage(&directory, Default::default()).await;
    create_root(&storage).await;
    for index in 0..100 {
        let id = storage.mint_id().await.unwrap();
        commit(
            &storage,
            json!([{ "type": "entry", "value": entry(id, 1, Some(json!({ "index": index, "text": "x".repeat(1_024) }))) }]),
        )
        .await
        .unwrap();
    }
    let document_id = storage.mint_id().await.unwrap();
    commit(
        &storage,
        json!([{
            "type": "document.create",
            "record": {
                "id": document_id,
                "kind": "size.history",
                "scope": { "kind": "conversation", "conversationId": 1 },
                "history": "rewindable",
                "fork": "asOf",
            },
            "content": { "kind": "base", "version": 1, "value": { "count": 0 } },
        }]),
    )
    .await
    .unwrap();
    for count in 1..=100 {
        commit(
            &storage,
            json!([{ "type": "document.change", "id": document_id, "content": { "kind": "delta", "version": 1, "ops": [["s", ["count"], count]] } }]),
        )
        .await
        .unwrap();
    }
    storage.close(context()).await.unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() < 1024 * 1024);
}

// ─── Portable SQLite facade settlement ───────────────────────────────────────

async fn memory_database() -> Arc<RusqliteDatabase> {
    Arc::new(
        open_rusqlite_database(":memory:", Default::default())
            .await
            .unwrap(),
    )
}

#[tokio::test]
async fn prepares_each_storage_statement_once_per_connection_and_reuses_it_across_transactions() {
    // Divergence: rusqlite's statement cache cannot be observed, so this keeps the behavior checks only.
    let storage = SqliteStorage::open(memory_database().await).await.unwrap();
    create_root(&storage).await;
    let entries: Vec<JsonValue> = (0..100)
        .map(|index| json!({ "type": "entry", "value": { "id": index + 2, "conversationId": 1, "kind": "cached" } }))
        .collect();
    commit(&storage, JsonValue::Array(entries)).await.unwrap();
    commit(
        &storage,
        json!([{ "type": "entry", "value": { "id": 102, "conversationId": 1, "kind": "cached-again" } }]),
    )
    .await
    .unwrap();
    let kind = |stored: Option<StoredEntry>| stored.unwrap().entry.kind;
    assert_eq!(
        kind(storage.entry(EntryId(2), context()).await.unwrap()),
        "cached"
    );
    assert_eq!(
        kind(storage.entry(EntryId(102), context()).await.unwrap()),
        "cached-again"
    );
    assert_rejects(
        storage.commit(&root_write(), context()).await,
        "ID 1 already belongs to conversation",
    );
    assert_eq!(
        kind(storage.entry(EntryId(2), context()).await.unwrap()),
        "cached"
    );
    storage.close(context()).await.unwrap();
}

#[tokio::test]
async fn commits_work_done_through_the_transaction_handle_and_closes_idempotently() {
    let database = memory_database().await;
    database
        .transaction(|transaction| async move {
            transaction
                .exec("CREATE TABLE async_probe (value INTEGER)")
                .await?;
            transaction
                .run("INSERT INTO async_probe (value) VALUES (?)", vec![int(1)])
                .await
        })
        .await
        .unwrap();
    assert_eq!(
        database
            .get("SELECT value FROM async_probe", vec![])
            .await
            .unwrap(),
        Some(row(&[("value", int(1))]))
    );
    database.close().await.unwrap();
    database.close().await.unwrap();
}

#[tokio::test]
async fn serializes_concurrent_transactions() {
    let database = memory_database().await;
    database
        .exec("CREATE TABLE transaction_queue (value INTEGER)")
        .await
        .unwrap();
    let (mark_first_started, first_started) = tokio::sync::oneshot::channel::<()>();
    let (release_first, first_gate) = tokio::sync::oneshot::channel::<()>();
    let first = database.transaction(|transaction| async move {
        transaction
            .exec("INSERT INTO transaction_queue (value) VALUES (1)")
            .await?;
        mark_first_started.send(()).unwrap();
        first_gate.await.unwrap();
        Ok(())
    });
    let second_started = Arc::new(AtomicBool::new(false));
    let second_flag = second_started.clone();
    let second = database.transaction(move |transaction| async move {
        second_flag.store(true, Ordering::SeqCst);
        transaction
            .exec("INSERT INTO transaction_queue (value) VALUES (2)")
            .await
    });
    let control = async {
        first_started.await.unwrap();
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(!second_started.load(Ordering::SeqCst));
        release_first.send(()).unwrap();
    };
    let (first, second, ()) = tokio::join!(first, second, control);
    first.unwrap();
    second.unwrap();
    assert_eq!(
        database
            .all("SELECT value FROM transaction_queue ORDER BY value", vec![])
            .await
            .unwrap(),
        vec![row(&[("value", int(1))]), row(&[("value", int(2))])]
    );
    database.close().await.unwrap();
}

#[tokio::test]
async fn runs_operations_in_call_order_whether_they_start_immediately_or_wait() {
    let database = memory_database().await;
    database
        .exec("CREATE TABLE call_order (value INTEGER)")
        .await
        .unwrap();
    // Operations called during a transaction must neither see its uncommitted rows nor join its rollback.
    let transaction = database.transaction(|handle| async move {
        handle
            .run("INSERT INTO call_order (value) VALUES (?)", vec![int(1)])
            .await?;
        tokio::task::yield_now().await;
        Err::<(), _>(Error::message("roll back"))
    });
    let before_write = database.all("SELECT value FROM call_order ORDER BY value", vec![]);
    let write = database.run("INSERT INTO call_order (value) VALUES (?)", vec![int(2)]);
    let after_write = database.all("SELECT value FROM call_order ORDER BY value", vec![]);
    let (transaction, write, before_write, after_write) =
        tokio::join!(transaction, write, before_write, after_write);
    assert_rejects(transaction, "roll back");
    write.unwrap();
    assert!(before_write.unwrap().is_empty());
    assert_eq!(after_write.unwrap(), vec![row(&[("value", int(2))])]);

    let storage = SqliteStorage::open(database).await.unwrap();
    let root = root_write();
    let commit = storage.commit(&root, context());
    let read = storage.conversation(ROOT_CONVERSATION_ID, context());
    let (commit, read) = tokio::join!(commit, read);
    commit.unwrap();
    assert_eq!(json(&read.unwrap().unwrap()), json!({ "id": 1 }));
    storage.close(context()).await.unwrap();
}

#[tokio::test]
async fn queues_ordinary_operations_behind_an_active_transaction() {
    let database = memory_database().await;
    database
        .exec("CREATE TABLE operation_queue (value INTEGER)")
        .await
        .unwrap();
    let (mark_started, started) = tokio::sync::oneshot::channel::<()>();
    let (release, gate) = tokio::sync::oneshot::channel::<()>();
    let pending = database.transaction(|transaction| async move {
        transaction
            .exec("INSERT INTO operation_queue (value) VALUES (1)")
            .await?;
        mark_started.send(()).unwrap();
        gate.await.unwrap();
        Ok(())
    });
    let write_settled = Arc::new(AtomicBool::new(false));
    let read_settled = Arc::new(AtomicBool::new(false));
    let control = async {
        started.await.unwrap();
        let write = database.exec("INSERT INTO operation_queue (value) VALUES (2)");
        let read = database.all("SELECT value FROM operation_queue ORDER BY value", vec![]);
        let write = {
            let settled = write_settled.clone();
            async move {
                let result = write.await;
                settled.store(true, Ordering::SeqCst);
                result
            }
        };
        let read = {
            let settled = read_settled.clone();
            async move {
                let result = read.await;
                settled.store(true, Ordering::SeqCst);
                result
            }
        };
        let observe = async {
            for _ in 0..10 {
                tokio::task::yield_now().await;
            }
            assert!(!write_settled.load(Ordering::SeqCst));
            assert!(!read_settled.load(Ordering::SeqCst));
            release.send(()).unwrap();
        };
        let (write, read, ()) = tokio::join!(write, read, observe);
        write.unwrap();
        read.unwrap()
    };
    let (pending, read) = tokio::join!(pending, control);
    pending.unwrap();
    assert_eq!(
        read,
        vec![row(&[("value", int(1))]), row(&[("value", int(2))])]
    );
    database.close().await.unwrap();
}

#[tokio::test]
async fn queues_database_calls_made_synchronously_by_a_transaction_that_started_immediately() {
    let database = memory_database().await;
    database
        .exec("CREATE TABLE barrier_probe (value INTEGER)")
        .await
        .unwrap();
    let outside: Arc<Mutex<Option<BoxFuture<'static, Result<()>>>>> = Arc::new(Mutex::new(None));
    let slot = outside.clone();
    let db = database.clone();
    let transaction = database
        .transaction(move |handle| async move {
            // Misuse: this call must wait for the transaction instead of joining it.
            *slot.lock() =
                Some(db.run("INSERT INTO barrier_probe (value) VALUES (?)", vec![int(2)]));
            handle
                .run("INSERT INTO barrier_probe (value) VALUES (?)", vec![int(1)])
                .await?;
            tokio::task::yield_now().await;
            Err::<(), _>(Error::message("roll back"))
        })
        .await;
    assert_rejects(transaction, "roll back");
    let outside = outside.lock().take().unwrap();
    outside.await.unwrap();
    assert_eq!(
        database
            .all("SELECT value FROM barrier_probe", vec![])
            .await
            .unwrap(),
        vec![row(&[("value", int(2))])]
    );
    database.close().await.unwrap();
}

#[tokio::test]
async fn lets_admitted_multi_query_reads_finish_before_storage_closes() {
    let storage = SqliteStorage::open(memory_database().await).await.unwrap();
    commit(
        &storage,
        json!([
            { "type": "conversation", "value": { "id": 1 } },
            { "type": "entry", "value": { "id": 2, "conversationId": 1, "kind": "probe" } },
        ]),
    )
    .await
    .unwrap();
    let scan = tokio::spawn({
        let storage = storage.clone();
        async move {
            storage
                .scan_entries(
                    &EntryQuery::conversation(ROOT_CONVERSATION_ID),
                    10,
                    None,
                    context(),
                )
                .await
        }
    });
    let entry = tokio::spawn({
        let storage = storage.clone();
        async move {
            storage
                .entry_in(ROOT_CONVERSATION_ID, EntryId(2), context())
                .await
        }
    });
    let head = tokio::spawn({
        let storage = storage.clone();
        async move {
            storage
                .find_latest_head_marker(ROOT_CONVERSATION_ID, None, context())
                .await
        }
    });
    // Rust futures are lazy: let the reads be admitted before closing.
    tokio::task::yield_now().await;
    let closed = tokio::spawn({
        let storage = storage.clone();
        async move { storage.close(context()).await }
    });
    // A repeated close settles only when the database is closed.
    let repeated = storage.close(context());
    let scan = scan.await.unwrap().unwrap();
    assert_eq!(
        scan.items.iter().map(|item| item.id).collect::<Vec<_>>(),
        vec![EntryId(2)]
    );
    assert_eq!(entry.await.unwrap().unwrap().unwrap().entry.kind, "probe");
    assert!(head.await.unwrap().unwrap().is_none());
    closed.await.unwrap().unwrap();
    repeated.await.unwrap();
    assert_rejects(
        storage
            .scan_entries(
                &EntryQuery::conversation(ROOT_CONVERSATION_ID),
                10,
                None,
                context(),
            )
            .await,
        "SqliteStorage is closed",
    );
}

#[tokio::test]
async fn rejects_a_transaction_handle_used_after_its_transaction_settles() {
    let database = memory_database().await;
    database
        .exec("CREATE TABLE stale_probe (value INTEGER)")
        .await
        .unwrap();
    let handle: Arc<Mutex<Option<Arc<dyn SqliteExecutor>>>> = Arc::new(Mutex::new(None));
    let slot = handle.clone();
    database
        .transaction(move |transaction| async move {
            *slot.lock() = Some(transaction.clone());
            transaction
                .run("INSERT INTO stale_probe (value) VALUES (?)", vec![int(1)])
                .await
        })
        .await
        .unwrap();
    let handle = handle.lock().take().unwrap();
    let stale = "SQLite transaction handle is no longer active";
    assert_rejects(
        handle
            .exec("INSERT INTO stale_probe (value) VALUES (2)")
            .await,
        stale,
    );
    assert_rejects(
        handle
            .run("INSERT INTO stale_probe (value) VALUES (?)", vec![int(3)])
            .await,
        stale,
    );
    assert_eq!(
        database
            .all("SELECT value FROM stale_probe", vec![])
            .await
            .unwrap(),
        vec![row(&[("value", int(1))])]
    );
    database.close().await.unwrap();
}

#[tokio::test]
async fn does_not_preserve_a_guaranteed_rejection_when_rollback_itself_fails() {
    let database = memory_database().await;
    database
        .exec("CREATE TABLE rollback_probe (value INTEGER)")
        .await
        .unwrap();
    let error = database
        .transaction(|transaction| async move {
            transaction
                .exec("INSERT INTO rollback_probe (value) VALUES (1)")
                .await?;
            transaction.exec("COMMIT").await?;
            Err::<(), _>(StorageRejected::new("rejected after an escaped commit").into())
        })
        .await
        .unwrap_err();
    let Error::Thrown(thrown) = &error else {
        panic!("expected an AggregateError, got {error:?}");
    };
    let aggregate = thrown.downcast_ref::<AggregateError>().unwrap();
    assert!(aggregate.errors[0].is_storage_rejected());
    assert!(!error.is_storage_rejected());
    assert_eq!(
        database
            .get("SELECT value FROM rollback_probe", vec![])
            .await
            .unwrap(),
        Some(row(&[("value", int(1))]))
    );
    database.close().await.unwrap();
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettlementMode {
    Immediate,
    Delay,
    Reject,
}

struct ControlledSettlementDatabase {
    delegate: Arc<RusqliteDatabase>,
    mode: Mutex<SettlementMode>,
    pending: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

impl ControlledSettlementDatabase {
    fn control_next_settlement(&self, mode: SettlementMode) {
        assert!(
            self.pending.lock().is_none(),
            "A settlement is already pending"
        );
        *self.mode.lock() = mode;
    }

    fn settle(&self) {
        let settle = self
            .pending
            .lock()
            .take()
            .expect("No settlement is pending");
        settle.send(()).unwrap();
    }
}

impl SqliteExecutor for ControlledSettlementDatabase {
    fn exec(&self, sql: &str) -> BoxFuture<'static, Result<()>> {
        self.delegate.exec(sql)
    }
    fn run(&self, sql: &str, params: Vec<SqliteValue>) -> BoxFuture<'static, Result<()>> {
        self.delegate.run(sql, params)
    }
    fn get(
        &self,
        sql: &str,
        params: Vec<SqliteValue>,
    ) -> BoxFuture<'static, Result<Option<SqliteRow>>> {
        self.delegate.get(sql, params)
    }
    fn all(
        &self,
        sql: &str,
        params: Vec<SqliteValue>,
    ) -> BoxFuture<'static, Result<Vec<SqliteRow>>> {
        self.delegate.all(sql, params)
    }
}

impl SqliteDatabase for ControlledSettlementDatabase {
    fn transaction_dyn<'a>(
        &'a self,
        callback: SqliteTransactionCallback<'a>,
    ) -> BoxFuture<'a, Result<Box<dyn Any + Send>>> {
        let mode = std::mem::replace(&mut *self.mode.lock(), SettlementMode::Immediate);
        if mode == SettlementMode::Immediate {
            return self.delegate.transaction_dyn(callback);
        }
        let settlement = self.delegate.transaction_dyn(Box::new(move |transaction| {
            Box::pin(async move {
                let value = callback(transaction).await?;
                if mode == SettlementMode::Reject {
                    return Err(Error::message("controlled settlement rejection"));
                }
                Ok(value)
            })
        }));
        let (settle, settled) = tokio::sync::oneshot::channel();
        *self.pending.lock() = Some(settle);
        Box::pin(async move {
            let result = settlement.await;
            settled.await.unwrap();
            result
        })
    }

    fn close(&self) -> BoxFuture<'static, Result<()>> {
        self.delegate.close()
    }
}

#[tokio::test]
async fn awaits_async_transaction_settlement_and_adopts_ids_only_after_success() {
    let database = Arc::new(ControlledSettlementDatabase {
        delegate: memory_database().await,
        mode: Mutex::new(SettlementMode::Immediate),
        pending: Mutex::new(None),
    });
    database.control_next_settlement(SettlementMode::Delay);
    let opening = tokio::spawn(SqliteStorage::open(database.clone()));
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert!(!opening.is_finished());
    database.settle();
    let storage = opening.await.unwrap().unwrap();

    database.control_next_settlement(SettlementMode::Delay);
    let committing = tokio::spawn({
        let storage = storage.clone();
        async move {
            commit(
                &storage,
                json!([{ "type": "entry", "value": { "id": 100, "conversationId": 1, "kind": "settled" } }]),
            )
            .await
        }
    });
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert!(!committing.is_finished());
    assert_eq!(storage.mint_id().await.unwrap(), 2);
    database.settle();
    assert_eq!(committing.await.unwrap().unwrap(), Seq(1));
    assert_eq!(storage.mint_id().await.unwrap(), 101);

    database.control_next_settlement(SettlementMode::Reject);
    let rejected = tokio::spawn({
        let storage = storage.clone();
        async move {
            commit(
                &storage,
                json!([{ "type": "entry", "value": { "id": 200, "conversationId": 1, "kind": "rejected" } }]),
            )
            .await
        }
    });
    tokio::task::yield_now().await;
    assert_eq!(storage.mint_id().await.unwrap(), 102);
    // The pending settlement exists once the spawned commit entered its transaction.
    while database.pending.lock().is_none() {
        tokio::task::yield_now().await;
    }
    database.settle();
    assert_rejects(rejected.await.unwrap(), "controlled settlement rejection");
    assert_eq!(storage.mint_id().await.unwrap(), 103);
    assert!(
        storage
            .entry(EntryId(200), context())
            .await
            .unwrap()
            .is_none()
    );
    storage.close(context()).await.unwrap();
}

#[tokio::test]
async fn reads_a_document_from_one_committed_state_while_a_commit_replaces_its_base() {
    let id = DocumentId(5);
    // Each yield count starts the commit at a different point of the read's record and revision queries.
    for yields in 0..16 {
        let storage = SqliteStorage::open(memory_database().await).await.unwrap();
        commit(
            &storage,
            json!([{
                "type": "document.create",
                "record": { "id": 5, "kind": "replaced", "scope": { "kind": "session" } },
                "content": { "kind": "base", "version": 1, "value": { "value": 1 } },
            }]),
        )
        .await
        .unwrap();
        let read = tokio::spawn({
            let storage = storage.clone();
            async move {
                storage
                    .document(id, DocumentPoint::Current, context())
                    .await
            }
        });
        for _ in 0..yields {
            tokio::task::yield_now().await;
        }
        let replace = tokio::spawn({
            let storage = storage.clone();
            async move {
                commit(
                    &storage,
                    json!([{ "type": "document.change", "id": 5, "content": { "kind": "base", "version": 1, "value": { "value": 2 } } }]),
                )
                .await
            }
        });
        let stored = read.await.unwrap().unwrap().unwrap();
        replace.await.unwrap().unwrap();
        let value = JsonValue::Object(stored.value);
        assert!(
            value == json!({ "value": 1 }) || value == json!({ "value": 2 }),
            "{value}"
        );
        storage.close(context()).await.unwrap();
    }
}

// ─── Durable SQLite migrations ───────────────────────────────────────────────

fn database_path(directory: &TempDir) -> std::path::PathBuf {
    directory.join("storage.sqlite")
}

#[tokio::test]
async fn creates_the_current_schema_and_can_be_applied_repeatedly() {
    let directory = TempDir::new("pi-durable-migrations-");
    let database = open_rusqlite_database(database_path(&directory), Default::default())
        .await
        .unwrap();
    apply_sqlite_migrations(&database, None).await.unwrap();
    apply_sqlite_migrations(&database, None).await.unwrap();
    assert_eq!(
        database
            .get(
                "SELECT version FROM durable_schema WHERE singleton = 1",
                vec![]
            )
            .await
            .unwrap(),
        Some(row(&[("version", int(CURRENT_SQLITE_SCHEMA_VERSION))]))
    );
    assert_eq!(
        database
            .get(
                "SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1",
                vec![]
            )
            .await
            .unwrap(),
        Some(row(&[
            ("next_id", SqliteValue::Text("2".into())),
            ("next_seq", int(1))
        ]))
    );
    database.close().await.unwrap();
}

#[tokio::test]
async fn rejects_a_database_newer_than_the_portable_core() {
    let directory = TempDir::new("pi-durable-migrations-");
    let path = database_path(&directory);
    let database = open_rusqlite_database(&path, Default::default())
        .await
        .unwrap();
    apply_sqlite_migrations(&database, None).await.unwrap();
    database
        .run(
            "UPDATE durable_schema SET version = ? WHERE singleton = 1",
            vec![int(CURRENT_SQLITE_SCHEMA_VERSION + 1)],
        )
        .await
        .unwrap();
    database.close().await.unwrap();
    assert_rejects(
        open_rusqlite_storage(&path, Default::default()).await,
        "is newer than supported version",
    );
}

#[tokio::test]
async fn rolls_initial_bootstrap_and_every_pending_migration_back_together() {
    let directory = TempDir::new("pi-durable-migrations-");
    let database = open_rusqlite_database(database_path(&directory), Default::default())
        .await
        .unwrap();
    let first = SqliteMigration::new(
        1,
        [
            "CREATE TABLE migration_first (value TEXT) STRICT",
            "INSERT INTO migration_first (value) VALUES ('retained')",
        ],
    );
    let failed = [
        first.clone(),
        SqliteMigration::new(
            2,
            [
                "CREATE TABLE migration_second (value TEXT) STRICT",
                "THIS IS NOT SQL",
            ],
        ),
    ];
    assert!(
        apply_sqlite_migrations(&database, Some(&failed))
            .await
            .is_err()
    );
    assert_eq!(
        database
            .get(
                "SELECT count(*) AS count FROM sqlite_schema WHERE name IN ('durable_schema', 'migration_first', 'migration_second')",
                vec![],
            )
            .await
            .unwrap(),
        Some(row(&[("count", int(0))]))
    );

    apply_sqlite_migrations(
        &database,
        Some(&[
            first,
            SqliteMigration::new(2, ["CREATE TABLE migration_second (value TEXT) STRICT"]),
        ]),
    )
    .await
    .unwrap();
    assert_eq!(
        database
            .get(
                "SELECT version FROM durable_schema WHERE singleton = 1",
                vec![]
            )
            .await
            .unwrap(),
        Some(row(&[("version", int(2))]))
    );
    assert_eq!(
        database
            .get("SELECT value FROM migration_first", vec![])
            .await
            .unwrap(),
        Some(row(&[("value", SqliteValue::Text("retained".into()))]))
    );
    database.close().await.unwrap();
}

#[tokio::test]
async fn rolls_a_failed_migration_back_and_preserves_stored_data_for_a_successful_retry() {
    let directory = TempDir::new("pi-durable-migrations-");
    let path = database_path(&directory);
    let storage = open_rusqlite_storage(&path, Default::default())
        .await
        .unwrap();
    commit(
        &storage,
        json!([
            { "type": "conversation", "value": { "id": 1 } },
            { "type": "entry", "value": { "id": 2, "conversationId": 1, "kind": "retained", "data": { "retained": true } } },
        ]),
    )
    .await
    .unwrap();
    storage.close(context()).await.unwrap();

    let database = open_rusqlite_database(&path, Default::default())
        .await
        .unwrap();
    let next_version = CURRENT_SQLITE_SCHEMA_VERSION + 1;
    let mut failed_migrations = sqlite_migrations();
    failed_migrations.push(SqliteMigration::new(
        next_version,
        [
            "CREATE TABLE migration_probe (value TEXT) STRICT",
            "THIS IS NOT SQL",
        ],
    ));
    assert!(
        apply_sqlite_migrations(&database, Some(&failed_migrations))
            .await
            .is_err()
    );
    assert_eq!(
        database
            .get(
                "SELECT version FROM durable_schema WHERE singleton = 1",
                vec![]
            )
            .await
            .unwrap(),
        Some(row(&[("version", int(CURRENT_SQLITE_SCHEMA_VERSION))]))
    );
    assert_eq!(
        database
            .get(
                "SELECT count(*) AS count FROM sqlite_schema WHERE type = 'table' AND name = 'migration_probe'",
                vec![],
            )
            .await
            .unwrap(),
        Some(row(&[("count", int(0))]))
    );

    let mut successful_migrations = sqlite_migrations();
    successful_migrations.push(SqliteMigration::new(
        next_version,
        ["CREATE TABLE migration_probe (value TEXT) STRICT"],
    ));
    apply_sqlite_migrations(&database, Some(&successful_migrations))
        .await
        .unwrap();
    assert_eq!(
        database
            .get(
                "SELECT version FROM durable_schema WHERE singleton = 1",
                vec![]
            )
            .await
            .unwrap(),
        Some(row(&[("version", int(next_version))]))
    );
    let entry = database
        .get(
            "SELECT record, commit_seq FROM entries WHERE id = 2",
            vec![],
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<JsonValue>(entry["record"].as_str().unwrap()).unwrap(),
        json!({ "id": 2, "conversationId": 1, "kind": "retained", "data": { "retained": true } })
    );
    assert_eq!(entry["commit_seq"], int(1));
    assert_eq!(
        database
            .get(
                "SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1",
                vec![]
            )
            .await
            .unwrap(),
        Some(row(&[
            ("next_id", SqliteValue::Text("3".into())),
            ("next_seq", int(2))
        ]))
    );
    database.close().await.unwrap();
}

// ─── Cross-runtime compatibility ─────────────────────────────────────────────

const TS_WRITTEN: &[u8] = include_bytes!("fixtures/ts-written.sqlite");
const TS_READS: &str = include_str!("fixtures/ts-reads.json");

fn schema(path: &std::path::Path) -> JsonValue {
    let connection = read_only(path);
    let mut statement = connection
        .prepare("SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY name")
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok(json!({
                "type": row.get::<_, String>(0)?,
                "name": row.get::<_, String>(1)?,
                "tbl_name": row.get::<_, String>(2)?,
                "sql": row.get::<_, Option<String>>(3)?,
            }))
        })
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    JsonValue::Array(rows)
}

#[tokio::test]
async fn creates_the_same_schema_as_the_typescript_adapter() {
    let expected: JsonValue = serde_json::from_str(TS_READS).unwrap();
    let directory = TempDir::new("pi-durable-sqlite-schema-");
    let path = directory.join("storage.sqlite");
    open_rusqlite_storage(&path, Default::default())
        .await
        .unwrap()
        .close(context())
        .await
        .unwrap();
    assert_eq!(schema(&path), expected["schema"]);
}

#[tokio::test]
async fn reads_a_database_written_by_the_typescript_storage() {
    let expected: JsonValue = serde_json::from_str(TS_READS).unwrap();
    let directory = TempDir::new("pi-durable-sqlite-ts-");
    let path = directory.join("ts-written.sqlite");
    std::fs::write(&path, TS_WRITTEN).unwrap();
    assert_eq!(schema(&path), expected["schema"]);

    let storage = open_rusqlite_storage(&path, Default::default())
        .await
        .unwrap();
    let seq = |name: &str| DocumentPoint::Seq(Seq(expected["seqs"][name].as_u64().unwrap()));
    let c = context();
    let conversation_scope = DocumentScope::Conversation {
        conversation_id: ROOT_CONVERSATION_ID,
    };
    let session_address = |kind: &str| DocumentAddress {
        kind: kind.into(),
        scope: DocumentScope::Session,
        key: None,
    };
    let optional = |value: Option<JsonValue>| value.unwrap_or(JsonValue::Null);
    let document = |stored: Option<StoredDocument>| optional(stored.as_ref().map(document_json));
    let actual = [
        ("minted", json!(storage.mint_id().await.unwrap())),
        (
            "conversation4",
            json(&storage.conversation(ConversationId(4), c).await.unwrap()),
        ),
        (
            "conversation6",
            json(&storage.conversation(ConversationId(6), c).await.unwrap()),
        ),
        (
            "scanConversations",
            page_json(
                &storage
                    .scan_conversations(&ConversationQuery::default(), 2, None, c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "scanConversationsByTask",
            page_json(
                &storage
                    .scan_conversations(
                        &ConversationQuery {
                            owner_task_id: Some(TaskId::new(5)),
                            ..Default::default()
                        },
                        10,
                        None,
                        c,
                    )
                    .await
                    .unwrap(),
            ),
        ),
        (
            "entry2",
            optional(
                storage
                    .entry(EntryId(2), c)
                    .await
                    .unwrap()
                    .as_ref()
                    .map(entry_json),
            ),
        ),
        (
            "entry2InFork",
            optional(
                storage
                    .entry_in(ConversationId(4), EntryId(2), c)
                    .await
                    .unwrap()
                    .as_ref()
                    .map(entry_json),
            ),
        ),
        (
            "entry3InFork",
            optional(
                storage
                    .entry_in(ConversationId(4), EntryId(3), c)
                    .await
                    .unwrap()
                    .as_ref()
                    .map(entry_json),
            ),
        ),
        (
            "scanForkEntries",
            page_json(
                &storage
                    .scan_entries(&EntryQuery::conversation(ConversationId(4)), 10, None, c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "rootHeadMarker",
            json(
                &storage
                    .find_latest_head_marker(ROOT_CONVERSATION_ID, None, c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "forkHeadMarkerBefore",
            json(
                &storage
                    .find_latest_head_marker(ConversationId(4), Some(EntryId(7)), c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "task5",
            json(&storage.task(TaskId::new(5), c).await.unwrap()),
        ),
        (
            "terminalTasks",
            page_json(
                &storage
                    .scan_tasks(
                        &TaskQuery {
                            status: Some(TaskStatus::Terminal),
                            kind: Some("fixture.task".into()),
                            ..Default::default()
                        },
                        10,
                        None,
                        c,
                    )
                    .await
                    .unwrap(),
            ),
        ),
        (
            "submission8",
            json(&storage.submission(SubmissionId(8), c).await.unwrap()),
        ),
        (
            "submissionByRequest",
            json(
                &storage
                    .submission_by_request(ROOT_CONVERSATION_ID, "req-é\u{0}", c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "doneSubmissions",
            page_json(
                &storage
                    .scan_submissions(
                        &SubmissionQuery {
                            status: Some(SubmissionStatus::Done),
                            ..Default::default()
                        },
                        10,
                        None,
                        c,
                    )
                    .await
                    .unwrap(),
            ),
        ),
        (
            "findHistoryCurrent",
            json(
                &storage
                    .find_document(
                        &DocumentAddress {
                            kind: "fixture.history".into(),
                            scope: conversation_scope,
                            key: None,
                        },
                        DocumentPoint::Current,
                        c,
                    )
                    .await
                    .unwrap(),
            ),
        ),
        (
            "findFamilyMember",
            json(
                &storage
                    .find_document(
                        &DocumentAddress {
                            kind: "fixture.family".into(),
                            scope: DocumentScope::Task {
                                task_id: TaskId::new(5),
                            },
                            key: Some("member/é".into()),
                        },
                        DocumentPoint::Current,
                        c,
                    )
                    .await
                    .unwrap(),
            ),
        ),
        (
            "findRetiredBefore",
            json(
                &storage
                    .find_document(&session_address("fixture.retired"), seq("documents"), c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "findRetiredAfter",
            json(
                &storage
                    .find_document(&session_address("fixture.retired"), seq("deltas"), c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "historyAtDocuments",
            document(
                storage
                    .document(DocumentId(10), seq("documents"), c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "historyAtDeltas",
            document(
                storage
                    .document(DocumentId(10), seq("deltas"), c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "historyCurrent",
            document(
                storage
                    .document(DocumentId(10), DocumentPoint::Current, c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "sessionCurrent",
            document(
                storage
                    .document(DocumentId(11), DocumentPoint::Current, c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "copiedCurrent",
            document(
                storage
                    .document(DocumentId(14), DocumentPoint::Current, c)
                    .await
                    .unwrap(),
            ),
        ),
        (
            "sessionDocuments",
            page_json(
                &storage
                    .scan_documents(
                        &DocumentQuery {
                            scope: DocumentScope::Session,
                            at: seq("documents"),
                            kind: None,
                        },
                        10,
                        None,
                        c,
                    )
                    .await
                    .unwrap(),
            ),
        ),
        (
            "rootDocuments",
            page_json(
                &storage
                    .scan_documents(
                        &DocumentQuery {
                            scope: conversation_scope,
                            at: DocumentPoint::Current,
                            kind: Some("fixture.history".into()),
                        },
                        10,
                        None,
                        c,
                    )
                    .await
                    .unwrap(),
            ),
        ),
    ];
    for (name, value) in actual {
        assert_eq!(value, expected[name], "{name}");
    }

    // Rust appends to the TS-written file and reads its own commit back.
    let next = expected["seqs"]["fork"].as_u64().unwrap() + 1;
    assert_eq!(
        commit(
            &storage,
            json!([{ "type": "document.change", "id": 10, "content": { "kind": "delta", "version": 1, "ops": [["s", ["count"], 3]] } }]),
        )
        .await
        .unwrap(),
        Seq(next)
    );
    assert_eq!(
        storage
            .document(DocumentId(10), DocumentPoint::Current, c)
            .await
            .unwrap()
            .unwrap()
            .value["count"],
        json!(3)
    );
    storage.close(context()).await.unwrap();
}
