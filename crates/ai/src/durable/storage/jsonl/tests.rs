//! Port of `test/jsonl-storage.test.ts`, over [`LocalExecutionEnv`].
//!
//! Divergence: the BigInt serialization failure (unrepresentable in Rust) is a
//! preparation failure (an entry reusing the root conversation's ID); it
//! likewise performs no I/O and leaves the storage usable.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::json;

use crate::chord::{Context, JsonValue};
use crate::durable::env::local::LocalExecutionEnv;
use crate::durable::env::{
    CreateDirOptions, FileError, FileErrorCode, FileInfo, FileSystem, ReadTextLinesOptions,
    RemoveOptions, TempFileOptions, TextLineReader,
};
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{
    ConversationId, DocumentId, EntryId, ROOT_CONVERSATION_ID, Seq, SubmissionId, TaskId,
};
use crate::durable::storage::test_support::{
    TempDir, context, json, pending_task, root_write, terminal_task, writes,
};
use crate::durable::testing::StorageConformanceProvider;
use crate::durable::types::{
    ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentPoint, DocumentQuery,
    DocumentRecord, DocumentScope, EntryQuery, EntryRecord, Page, Storage, StorageWrite,
    StoredDocument, StoredEntry, SubmissionQuery, SubmissionRecord, TaskQuery, TaskRecord,
};

use super::{
    JsonlCorruptionError, JsonlStorage, JsonlStorageOptions, JsonlStoragePoisonedError,
    open_local_jsonl_storage,
};

fn dir_path(directory: &TempDir) -> String {
    directory
        .join("")
        .to_string_lossy()
        .trim_end_matches('/')
        .to_owned()
}

fn local_env(directory: &str) -> Arc<dyn FileSystem> {
    Arc::new(LocalExecutionEnv::at(directory))
}

async fn open_storage(
    directory: &str,
    fs: Arc<dyn FileSystem>,
    options: JsonlStorageOptions,
) -> JsonlStorage {
    JsonlStorage::open(directory, fs, context(), options)
        .await
        .unwrap()
}

async fn open_default(directory: &str) -> JsonlStorage {
    open_storage(directory, local_env(directory), Default::default()).await
}

async fn open_result(directory: &str) -> Result<JsonlStorage> {
    JsonlStorage::open(
        directory,
        local_env(directory),
        context(),
        Default::default(),
    )
    .await
}

fn fsync() -> JsonlStorageOptions {
    JsonlStorageOptions { fsync: Some(true) }
}

async fn create_root(storage: &dyn Storage) {
    storage.commit(&root_write(), context()).await.unwrap();
}

async fn commit(storage: &dyn Storage, value: JsonValue) -> Result<Seq> {
    storage.commit(&writes(value), context()).await
}

async fn mint(storage: &dyn Storage) -> u64 {
    storage.mint_id().await.unwrap()
}

fn session_document(id: u64, kind: &str) -> JsonValue {
    json!({ "id": id, "kind": kind, "scope": { "kind": "session" } })
}

fn create_document(record: JsonValue, value: JsonValue) -> JsonValue {
    json!({
        "type": "document.create",
        "record": record,
        "content": { "kind": "base", "version": 1, "value": value },
    })
}

fn change_base(id: u64, value: JsonValue) -> JsonValue {
    json!({ "type": "document.change", "id": id, "content": { "kind": "base", "version": 1, "value": value } })
}

fn change_delta(id: u64, ops: JsonValue) -> JsonValue {
    json!({ "type": "document.change", "id": id, "content": { "kind": "delta", "version": 1, "ops": ops } })
}

fn task_write(task: JsonValue) -> JsonValue {
    json!({ "type": "task", "value": task })
}

async fn current_value(storage: &dyn Storage, id: u64) -> Option<JsonValue> {
    storage
        .document(DocumentId(id), DocumentPoint::Current, context())
        .await
        .unwrap()
        .map(|document| JsonValue::Object(document.value))
}

async fn task_json(storage: &dyn Storage, id: u64) -> Option<JsonValue> {
    storage
        .task(TaskId::new(id), context())
        .await
        .unwrap()
        .map(|task| json(&task))
}

fn read_lines(path: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap();
    if text.is_empty() {
        return Vec::new();
    }
    text.trim_end().split('\n').map(str::to_owned).collect()
}

fn file_exists(path: &str) -> bool {
    std::path::Path::new(path).exists()
}

fn reclaim_files(directory: &str) -> Vec<String> {
    std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".reclaim"))
        .collect()
}

fn append_bytes(path: &str, bytes: &[u8]) {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
}

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

// ─── Conformance ─────────────────────────────────────────────────────────────

mod jsonl_storage {
    use super::*;

    crate::storage_conformance_tests!({
        let provider: StorageConformanceProvider = Arc::new(|test| {
            Box::pin(async move {
                let directory = TempDir::new("pi-durable-jsonl-");
                let storage = open_default(&dir_path(&directory)).await;
                test(Arc::new(storage.clone())).await;
                storage.close(context()).await.unwrap();
            })
        });
        provider
    });
}

/// Closes and reopens the directory after every commit (`ReopeningStorage`).
struct ReopeningStorage {
    current: Mutex<JsonlStorage>,
    directory: String,
    closed: AtomicBool,
}

impl ReopeningStorage {
    fn current(&self) -> JsonlStorage {
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
        *self.current.lock() = open_default(&self.directory).await;
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

mod jsonl_storage_across_reopen {
    use super::*;

    crate::storage_conformance_tests!({
        let provider: StorageConformanceProvider = Arc::new(|test| {
            Box::pin(async move {
                let directory = TempDir::new("pi-durable-jsonl-conformance-");
                let path = dir_path(&directory);
                let storage = Arc::new(ReopeningStorage {
                    current: Mutex::new(open_default(&path).await),
                    directory: path,
                    closed: AtomicBool::new(false),
                });
                test(storage.clone()).await;
                storage.close(context()).await.unwrap();
            })
        });
        provider
    });
}

// ─── InstrumentedEnv ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Operation {
    Append,
    Flush,
    Write,
    Rename,
    Remove,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Before,
    After,
    Short,
}

#[derive(Debug, Clone, Copy)]
struct Failure {
    operation: Operation,
    call: usize,
    mode: Mode,
}

const fn failure(operation: Operation, call: usize, mode: Mode) -> Failure {
    Failure {
        operation,
        call,
        mode,
    }
}

#[derive(Default)]
struct Observations {
    operations: Vec<String>,
    failure: Option<Failure>,
    calls: HashMap<Operation, usize>,
}

/// A [`LocalExecutionEnv`] that records file mutations and injects one failure.
struct InstrumentedEnv {
    inner: LocalExecutionEnv,
    observations: Mutex<Observations>,
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn injected(message: &str, path: &str) -> FileError {
    FileError::new(FileErrorCode::Unknown, message, Some(path.to_owned()))
}

fn partial(content: &[u8]) -> &[u8] {
    &content[..(content.len() / 2).max(1).min(content.len())]
}

impl InstrumentedEnv {
    fn new(directory: &str) -> Arc<Self> {
        Arc::new(Self {
            inner: LocalExecutionEnv::at(directory),
            observations: Mutex::new(Observations::default()),
        })
    }

    fn fail(&self, failure: Failure) {
        let mut observations = self.observations.lock();
        *observations = Observations::default();
        observations.failure = Some(failure);
    }

    fn clear(&self) {
        *self.observations.lock() = Observations::default();
    }

    fn operations(&self) -> Vec<String> {
        self.observations.lock().operations.clone()
    }

    /// Count one call and return the failure mode injected into it.
    fn observe(&self, operation: Operation, label: String) -> Option<Mode> {
        let mut observations = self.observations.lock();
        let calls = observations.calls.entry(operation).or_default();
        *calls += 1;
        let call = *calls;
        observations.operations.push(label);
        observations
            .failure
            .filter(|failure| failure.operation == operation && failure.call == call)
            .map(|failure| failure.mode)
    }
}

#[async_trait]
impl FileSystem for InstrumentedEnv {
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn cwd(&self) -> String {
        self.inner.cwd()
    }
    fn set_cwd(&self, cwd: String) {
        self.inner.set_cwd(cwd);
    }
    async fn absolute_path(&self, path: &str, c: &Context) -> Result<String, FileError> {
        self.inner.absolute_path(path, c).await
    }
    async fn join_path(&self, parts: &[&str], c: &Context) -> Result<String, FileError> {
        self.inner.join_path(parts, c).await
    }
    async fn read_text_file(&self, path: &str, c: &Context) -> Result<String, FileError> {
        self.inner.read_text_file(path, c).await
    }
    async fn open_text_line_reader(
        &self,
        path: &str,
        c: &Context,
    ) -> Result<Box<dyn TextLineReader>, FileError> {
        self.inner.open_text_line_reader(path, c).await
    }
    async fn read_text_lines(
        &self,
        path: &str,
        options: ReadTextLinesOptions,
        c: &Context,
    ) -> Result<Vec<String>, FileError> {
        self.inner.read_text_lines(path, options, c).await
    }
    async fn read_binary_file(&self, path: &str, c: &Context) -> Result<Vec<u8>, FileError> {
        self.inner.read_binary_file(path, c).await
    }

    async fn write_file(&self, path: &str, content: &[u8], c: &Context) -> Result<(), FileError> {
        match self.observe(Operation::Write, format!("write:{}", basename(path))) {
            None => self.inner.write_file(path, content, c).await,
            Some(Mode::Before) => Err(injected("injected write failure", path)),
            Some(Mode::Short) => {
                self.inner.write_file(path, partial(content), c).await?;
                Err(injected("injected short write", path))
            }
            Some(Mode::After) => {
                self.inner.write_file(path, content, c).await?;
                Err(injected("injected post-write failure", path))
            }
        }
    }

    async fn append_file(&self, path: &str, content: &[u8], c: &Context) -> Result<(), FileError> {
        match self.observe(Operation::Append, format!("append:{}", basename(path))) {
            None => self.inner.append_file(path, content, c).await,
            Some(Mode::Before) => Err(injected("injected append failure", path)),
            Some(Mode::Short) => {
                self.inner.append_file(path, partial(content), c).await?;
                Err(injected("injected short append", path))
            }
            Some(Mode::After) => {
                self.inner.append_file(path, content, c).await?;
                Err(injected("injected post-append failure", path))
            }
        }
    }

    async fn truncate_file(&self, path: &str, size: u64, c: &Context) -> Result<(), FileError> {
        self.inner.truncate_file(path, size, c).await
    }

    async fn flush_file(&self, path: &str, c: &Context) -> Result<(), FileError> {
        match self.observe(Operation::Flush, format!("flush:{}", basename(path))) {
            None => self.inner.flush_file(path, c).await,
            Some(mode) => {
                if mode == Mode::After {
                    self.inner.flush_file(path, c).await?;
                }
                Err(injected("injected flush failure", path))
            }
        }
    }

    async fn rename_file(
        &self,
        source_path: &str,
        destination_path: &str,
        c: &Context,
    ) -> Result<(), FileError> {
        let label = format!(
            "rename:{}->{}",
            basename(source_path),
            basename(destination_path)
        );
        match self.observe(Operation::Rename, label) {
            None => {
                self.inner
                    .rename_file(source_path, destination_path, c)
                    .await
            }
            Some(mode) => {
                if mode == Mode::After {
                    self.inner
                        .rename_file(source_path, destination_path, c)
                        .await?;
                }
                Err(injected("injected rename failure", source_path))
            }
        }
    }

    async fn file_info(&self, path: &str, c: &Context) -> Result<FileInfo, FileError> {
        self.inner.file_info(path, c).await
    }
    async fn list_dir(&self, path: &str, c: &Context) -> Result<Vec<FileInfo>, FileError> {
        self.inner.list_dir(path, c).await
    }
    async fn canonical_path(&self, path: &str, c: &Context) -> Result<String, FileError> {
        self.inner.canonical_path(path, c).await
    }
    async fn exists(&self, path: &str, c: &Context) -> Result<bool, FileError> {
        self.inner.exists(path, c).await
    }
    async fn create_dir(
        &self,
        path: &str,
        options: CreateDirOptions,
        c: &Context,
    ) -> Result<(), FileError> {
        self.inner.create_dir(path, options, c).await
    }

    async fn remove(
        &self,
        path: &str,
        options: RemoveOptions,
        c: &Context,
    ) -> Result<(), FileError> {
        match self.observe(Operation::Remove, format!("remove:{}", basename(path))) {
            None => self.inner.remove(path, options, c).await,
            Some(mode) => {
                if mode == Mode::After {
                    self.inner.remove(path, options, c).await?;
                }
                Err(injected("injected remove failure", path))
            }
        }
    }

    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        c: &Context,
    ) -> Result<String, FileError> {
        self.inner.create_temp_dir(prefix, c).await
    }
    async fn create_temp_file(
        &self,
        options: TempFileOptions,
        c: &Context,
    ) -> Result<String, FileError> {
        self.inner.create_temp_file(options, c).await
    }
    async fn cleanup(&self, c: &Context) {
        FileSystem::cleanup(&self.inner, c).await;
    }
}

// ─── Pico JsonlStorage publication and recovery ──────────────────────────────

#[tokio::test]
async fn opens_through_the_local_adapter() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let storage = open_local_jsonl_storage(&dir_path(&directory), context(), Default::default())
        .await
        .unwrap();
    create_root(&storage).await;
    assert_eq!(
        storage
            .conversation(ROOT_CONVERSATION_ID, context())
            .await
            .unwrap()
            .map(|record| json(&record)),
        Some(json!({ "id": 1 }))
    );
}

#[tokio::test]
async fn publishes_every_commit_before_reclaiming_current_only_document_and_terminal_task_sidecars()
{
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let task_id = mint(&storage).await;
    let document_id = mint(&storage).await;
    commit(
        &storage,
        json!([task_write(pending_task(task_id, "ready"))]),
    )
    .await
    .unwrap();
    commit(
        &storage,
        json!([create_document(
            session_document(document_id, "test.document"),
            json!({ "count": 0 })
        )]),
    )
    .await
    .unwrap();
    commit(&storage, json!([change_delta(document_id, json!([]))]))
        .await
        .unwrap();
    commit(
        &storage,
        json!([change_base(document_id, json!({ "count": 1 }))]),
    )
    .await
    .unwrap();
    assert_eq!(read_lines(&format!("{dir}/task-{task_id}.jsonl")).len(), 1);
    assert_eq!(
        read_lines(&format!("{dir}/doc-{document_id}.jsonl")).len(),
        1
    );

    commit(
        &storage,
        json!([{ "type": "document.retire", "id": document_id }]),
    )
    .await
    .unwrap();
    commit(&storage, json!([task_write(terminal_task(task_id))]))
        .await
        .unwrap();

    let main = read_lines(&format!("{dir}/main.jsonl"));
    assert_eq!(main.len(), 7);
    assert!(!file_exists(&format!("{dir}/task-{task_id}.jsonl")));
    assert!(!file_exists(&format!("{dir}/doc-{document_id}.jsonl")));
    for line in main {
        let marker: JsonValue = serde_json::from_str(&line).unwrap();
        assert_eq!(marker["type"], "commit");
    }
}

#[tokio::test]
async fn orders_multiple_live_task_replacements_in_one_sidecar_by_commit_ordinal() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let task_id = mint(&storage).await;
    commit(
        &storage,
        json!([
            task_write(pending_task(task_id, "first")),
            task_write(pending_task(task_id, "second")),
        ]),
    )
    .await
    .unwrap();
    assert_eq!(read_lines(&format!("{dir}/task-{task_id}.jsonl")).len(), 2);

    let reopened = open_default(&dir).await;
    assert_eq!(
        task_json(&reopened, task_id).await,
        Some(pending_task(task_id, "second"))
    );
}

#[tokio::test]
async fn removes_a_complete_plus_torn_unconfirmed_multi_record_sidecar_append() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let env = InstrumentedEnv::new(&dir);
    let storage = open_storage(&dir, env.clone(), Default::default()).await;
    create_root(&storage).await;
    let task_id = mint(&storage).await;
    env.fail(failure(Operation::Append, 1, Mode::Short));
    let second = format!("second-{}", "x".repeat(512));
    assert_rejects(
        commit(
            &storage,
            json!([
                task_write(pending_task(task_id, "first")),
                task_write(pending_task(task_id, &second)),
            ]),
        )
        .await,
        "poisoned",
    );
    let path = format!("{dir}/task-{task_id}.jsonl");
    let partial = std::fs::read(&path).unwrap();
    assert_ne!(partial.last(), Some(&b'\n'));
    assert_eq!(partial.iter().filter(|&&byte| byte == b'\n').count(), 1);

    let reopened = open_default(&dir).await;
    assert_eq!(task_json(&reopened, task_id).await, None);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
}

#[tokio::test]
async fn serializes_the_complete_candidate_before_io_and_leaves_preparation_failures_usable() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let env = InstrumentedEnv::new(&dir);
    let storage = open_storage(&dir, env.clone(), Default::default()).await;
    create_root(&storage).await;
    env.clear();
    // The root conversation already owns ID 1.
    assert!(
        commit(
            &storage,
            json!([{ "type": "entry", "value": { "id": 1, "conversationId": 1, "kind": "bad" } }]),
        )
        .await
        .is_err()
    );
    assert_eq!(env.operations(), Vec::<String>::new());
    assert!(
        storage
            .entry(EntryId(1), context())
            .await
            .unwrap()
            .is_none()
    );
    let id = mint(&storage).await;
    assert_eq!(
        commit(
            &storage,
            json!([{ "type": "entry", "value": { "id": id, "conversationId": 1, "kind": "good" } }]),
        )
        .await
        .unwrap(),
        Seq(2)
    );
}

#[tokio::test]
async fn poisons_after_append_failures_and_recovers_only_confirmed_state() {
    for case in [
        failure(Operation::Append, 1, Mode::Before),
        failure(Operation::Append, 1, Mode::After),
        failure(Operation::Append, 1, Mode::Short),
        failure(Operation::Append, 2, Mode::Before),
        failure(Operation::Append, 2, Mode::After),
        failure(Operation::Append, 3, Mode::Before),
        failure(Operation::Append, 3, Mode::Short),
        failure(Operation::Append, 3, Mode::After),
    ] {
        let directory = TempDir::new("pi-durable-jsonl-");
        let dir = dir_path(&directory);
        let env = InstrumentedEnv::new(&dir);
        let storage = open_storage(&dir, env.clone(), Default::default()).await;
        create_root(&storage).await;
        let first_id = mint(&storage).await;
        let second_id = mint(&storage).await;
        env.fail(case);
        let error = assert_rejects(
            commit(
                &storage,
                json!([
                    create_document(session_document(first_id, "first"), json!({ "text": "α" })),
                    create_document(
                        session_document(second_id, "second"),
                        json!({ "text": "β" })
                    ),
                ]),
            )
            .await,
            "poisoned",
        );
        assert!(
            matches!(&error, Error::Thrown(thrown) if thrown.downcast_ref::<JsonlStoragePoisonedError>().is_some())
        );
        assert_rejects(
            storage
                .document(DocumentId(first_id), DocumentPoint::Current, context())
                .await,
            "poisoned",
        );

        let reopened = open_default(&dir).await;
        let marker_survived = case.call == 3 && case.mode == Mode::After;
        assert_eq!(
            current_value(&reopened, first_id).await,
            marker_survived.then(|| json!({ "text": "α" })),
            "{case:?}"
        );
        assert_eq!(
            current_value(&reopened, second_id).await,
            marker_survived.then(|| json!({ "text": "β" })),
            "{case:?}"
        );
    }
}

#[tokio::test]
async fn poisons_after_flush_failures_and_never_writes_a_marker() {
    for case in [
        failure(Operation::Flush, 1, Mode::Before),
        failure(Operation::Flush, 1, Mode::After),
        failure(Operation::Flush, 2, Mode::Before),
        failure(Operation::Flush, 2, Mode::After),
    ] {
        let directory = TempDir::new("pi-durable-jsonl-");
        let dir = dir_path(&directory);
        let env = InstrumentedEnv::new(&dir);
        let storage = open_storage(&dir, env.clone(), fsync()).await;
        create_root(&storage).await;
        let first_id = mint(&storage).await;
        let second_id = mint(&storage).await;
        env.fail(case);
        assert_rejects(
            commit(
                &storage,
                json!([
                    create_document(session_document(first_id, "flush.first"), json!({})),
                    create_document(session_document(second_id, "flush.second"), json!({})),
                ]),
            )
            .await,
            "poisoned",
        );
        let reopened = open_default(&dir).await;
        assert_eq!(current_value(&reopened, first_id).await, None, "{case:?}");
        assert_eq!(current_value(&reopened, second_id).await, None, "{case:?}");
    }
}

#[tokio::test]
async fn orders_publication_flushes_exactly_and_flushes_main_only_to_authorize_reclamation() {
    for fsync_enabled in [false, true] {
        let directory = TempDir::new("pi-durable-jsonl-");
        let dir = dir_path(&directory);
        let env = InstrumentedEnv::new(&dir);
        let storage = open_storage(
            &dir,
            env.clone(),
            JsonlStorageOptions {
                fsync: Some(fsync_enabled),
            },
        )
        .await;
        create_root(&storage).await;
        let first_id = mint(&storage).await;
        let second_id = mint(&storage).await;
        env.clear();
        commit(
            &storage,
            json!([
                create_document(session_document(second_id, "second"), json!({})),
                create_document(session_document(first_id, "first"), json!({})),
            ]),
        )
        .await
        .unwrap();
        let mut expected = vec![
            format!("append:doc-{second_id}.jsonl"),
            format!("append:doc-{first_id}.jsonl"),
        ];
        if fsync_enabled {
            expected.push(format!("flush:doc-{second_id}.jsonl"));
            expected.push(format!("flush:doc-{first_id}.jsonl"));
        }
        expected.push("append:main.jsonl".into());
        assert_eq!(env.operations(), expected);

        env.clear();
        commit(
            &storage,
            json!([change_base(first_id, json!({ "checkpoint": true }))]),
        )
        .await
        .unwrap();
        let mut expected = vec![format!("append:doc-{first_id}.jsonl")];
        if fsync_enabled {
            expected.push(format!("flush:doc-{first_id}.jsonl"));
        }
        expected.push("append:main.jsonl".into());
        if fsync_enabled {
            expected.push("flush:main.jsonl".into());
        }
        expected.push(format!("write:doc-{first_id}.jsonl.reclaim"));
        if fsync_enabled {
            expected.push(format!("flush:doc-{first_id}.jsonl.reclaim"));
        }
        expected.push(format!(
            "rename:doc-{first_id}.jsonl.reclaim->doc-{first_id}.jsonl"
        ));
        assert_eq!(env.operations(), expected);

        env.clear();
        let entry_id = mint(&storage).await;
        commit(
            &storage,
            json!([{ "type": "entry", "value": { "id": entry_id, "conversationId": 1, "kind": "main-only" } }]),
        )
        .await
        .unwrap();
        assert_eq!(env.operations(), vec!["append:main.jsonl"]);

        let task_id = mint(&storage).await;
        commit(
            &storage,
            json!([task_write(pending_task(task_id, "ready"))]),
        )
        .await
        .unwrap();
        env.clear();
        commit(&storage, json!([task_write(terminal_task(task_id))]))
            .await
            .unwrap();
        let mut expected = vec!["append:main.jsonl".to_owned()];
        if fsync_enabled {
            expected.push("flush:main.jsonl".into());
        }
        expected.push(format!("remove:task-{task_id}.jsonl"));
        assert_eq!(env.operations(), expected);
    }
}

#[tokio::test]
async fn recovers_a_committed_base_across_reclaim_failures() {
    for case in [
        failure(Operation::Write, 1, Mode::Before),
        failure(Operation::Write, 1, Mode::Short),
        failure(Operation::Write, 1, Mode::After),
        failure(Operation::Rename, 1, Mode::Before),
        failure(Operation::Rename, 1, Mode::After),
    ] {
        let directory = TempDir::new("pi-durable-jsonl-");
        let dir = dir_path(&directory);
        let env = InstrumentedEnv::new(&dir);
        let storage = open_storage(&dir, env.clone(), Default::default()).await;
        create_root(&storage).await;
        let id = mint(&storage).await;
        commit(
            &storage,
            json!([create_document(
                session_document(id, "test.document"),
                json!({ "count": 0 })
            )]),
        )
        .await
        .unwrap();
        commit(
            &storage,
            json!([change_delta(id, json!([["s", ["count"], 1]]))]),
        )
        .await
        .unwrap();

        env.fail(case);
        assert_eq!(
            commit(&storage, json!([change_base(id, json!({ "count": 2 }))]))
                .await
                .unwrap(),
            Seq(4)
        );
        assert_eq!(
            current_value(&storage, id).await,
            Some(json!({ "count": 2 }))
        );
        storage.close(context()).await.unwrap();

        let reopened = open_default(&dir).await;
        assert_eq!(
            current_value(&reopened, id).await,
            Some(json!({ "count": 2 })),
            "{case:?}"
        );
        assert_eq!(
            read_lines(&format!("{dir}/doc-{id}.jsonl")).len(),
            1,
            "{case:?}"
        );
        assert_eq!(reclaim_files(&dir), Vec::<String>::new());
    }
}

#[tokio::test]
async fn recovers_document_retirement_reclamation_across_remove_failures() {
    for mode in [Mode::Before, Mode::After] {
        let directory = TempDir::new("pi-durable-jsonl-");
        let dir = dir_path(&directory);
        let env = InstrumentedEnv::new(&dir);
        let storage = open_storage(&dir, env.clone(), Default::default()).await;
        create_root(&storage).await;
        let task_id = mint(&storage).await;
        let id = mint(&storage).await;
        commit(
            &storage,
            json!([
                task_write(pending_task(task_id, "ready")),
                create_document(
                    json!({ "id": id, "kind": "task.document", "scope": { "kind": "task", "taskId": task_id } }),
                    json!({ "count": 1 }),
                ),
            ]),
        )
        .await
        .unwrap();
        env.fail(failure(Operation::Remove, 1, mode));
        assert_eq!(
            commit(&storage, json!([{ "type": "document.retire", "id": id }]))
                .await
                .unwrap(),
            Seq(3)
        );
        assert_eq!(current_value(&storage, id).await, None);
        storage.close(context()).await.unwrap();

        let reopened = open_default(&dir).await;
        assert_eq!(current_value(&reopened, id).await, None);
        assert_eq!(
            task_json(&reopened, task_id).await,
            Some(pending_task(task_id, "ready"))
        );
        assert!(!file_exists(&format!("{dir}/doc-{id}.jsonl")), "{mode:?}");
        assert_eq!(reclaim_files(&dir), Vec::<String>::new());
    }
}

#[tokio::test]
async fn defers_reclamation_after_authorizing_main_flush_failure() {
    for mode in [Mode::Before, Mode::After] {
        let directory = TempDir::new("pi-durable-jsonl-");
        let dir = dir_path(&directory);
        let env = InstrumentedEnv::new(&dir);
        let storage = open_storage(&dir, env.clone(), fsync()).await;
        create_root(&storage).await;
        let id = mint(&storage).await;
        commit(
            &storage,
            json!([create_document(
                session_document(id, "test.document"),
                json!({ "count": 0 })
            )]),
        )
        .await
        .unwrap();
        env.fail(failure(Operation::Flush, 2, mode));
        assert_eq!(
            commit(&storage, json!([change_base(id, json!({ "count": 2 }))]))
                .await
                .unwrap(),
            Seq(3)
        );
        assert_eq!(
            env.operations(),
            vec![
                format!("append:doc-{id}.jsonl"),
                format!("flush:doc-{id}.jsonl"),
                "append:main.jsonl".into(),
                "flush:main.jsonl".into(),
            ]
        );
        assert_eq!(
            current_value(&storage, id).await,
            Some(json!({ "count": 2 }))
        );
        assert_eq!(read_lines(&format!("{dir}/doc-{id}.jsonl")).len(), 2);
        storage.close(context()).await.unwrap();

        let recovery_env = InstrumentedEnv::new(&dir);
        recovery_env.fail(failure(Operation::Flush, 1, mode));
        let deferred = open_storage(&dir, recovery_env.clone(), fsync()).await;
        assert_eq!(
            current_value(&deferred, id).await,
            Some(json!({ "count": 2 }))
        );
        assert_eq!(recovery_env.operations(), vec!["flush:main.jsonl"]);
        assert_eq!(read_lines(&format!("{dir}/doc-{id}.jsonl")).len(), 2);
        deferred.close(context()).await.unwrap();

        let reclaimed = open_storage(&dir, local_env(&dir), fsync()).await;
        assert_eq!(
            current_value(&reclaimed, id).await,
            Some(json!({ "count": 2 }))
        );
        assert_eq!(read_lines(&format!("{dir}/doc-{id}.jsonl")).len(), 1);
    }
}

#[tokio::test]
async fn keeps_a_committed_base_usable_after_reclaim_temp_flush_failure() {
    for mode in [Mode::Before, Mode::After] {
        let directory = TempDir::new("pi-durable-jsonl-");
        let dir = dir_path(&directory);
        let env = InstrumentedEnv::new(&dir);
        let storage = open_storage(&dir, env.clone(), fsync()).await;
        create_root(&storage).await;
        let id = mint(&storage).await;
        commit(
            &storage,
            json!([create_document(
                session_document(id, "test.document"),
                json!({ "count": 0 })
            )]),
        )
        .await
        .unwrap();
        env.fail(failure(Operation::Flush, 3, mode));
        assert_eq!(
            commit(&storage, json!([change_base(id, json!({ "count": 2 }))]))
                .await
                .unwrap(),
            Seq(3)
        );
        env.clear();
        commit(
            &storage,
            json!([change_delta(id, json!([["s", ["count"], 3]]))]),
        )
        .await
        .unwrap();
        storage.close(context()).await.unwrap();

        let reopened = open_storage(&dir, local_env(&dir), fsync()).await;
        assert_eq!(
            current_value(&reopened, id).await,
            Some(json!({ "count": 3 }))
        );
        assert_eq!(
            read_lines(&format!("{dir}/doc-{id}.jsonl")).len(),
            2,
            "{mode:?}"
        );
    }
}

#[tokio::test]
async fn recovers_terminal_task_reclamation_across_remove_failures() {
    for mode in [Mode::Before, Mode::After] {
        let directory = TempDir::new("pi-durable-jsonl-");
        let dir = dir_path(&directory);
        let env = InstrumentedEnv::new(&dir);
        let storage = open_storage(&dir, env.clone(), Default::default()).await;
        create_root(&storage).await;
        let id = mint(&storage).await;
        commit(&storage, json!([task_write(pending_task(id, "ready"))]))
            .await
            .unwrap();
        env.fail(failure(Operation::Remove, 1, mode));
        assert_eq!(
            commit(&storage, json!([task_write(terminal_task(id))]))
                .await
                .unwrap(),
            Seq(3)
        );
        assert_eq!(task_json(&storage, id).await, Some(terminal_task(id)));
        storage.close(context()).await.unwrap();

        let reopened = open_default(&dir).await;
        assert_eq!(task_json(&reopened, id).await, Some(terminal_task(id)));
        assert!(!file_exists(&format!("{dir}/task-{id}.jsonl")), "{mode:?}");
        assert_eq!(reclaim_files(&dir), Vec::<String>::new());
    }
}

#[tokio::test]
async fn appends_later_deltas_to_the_replacement_sidecar_after_a_current_only_base() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let id = mint(&storage).await;
    commit(
        &storage,
        json!([create_document(
            session_document(id, "test.document"),
            json!({ "count": 0 })
        )]),
    )
    .await
    .unwrap();
    commit(&storage, json!([change_base(id, json!({ "count": 10 }))]))
        .await
        .unwrap();
    commit(
        &storage,
        json!([change_delta(id, json!([["s", ["count"], 11]]))]),
    )
    .await
    .unwrap();
    assert_eq!(read_lines(&format!("{dir}/doc-{id}.jsonl")).len(), 2);
    let reopened = open_default(&dir).await;
    assert_eq!(
        current_value(&reopened, id).await,
        Some(json!({ "count": 11 }))
    );
}

#[tokio::test]
async fn never_reclaims_rewindable_document_history_including_after_a_base_and_retirement() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let id = mint(&storage).await;
    let record = json!({
        "id": id,
        "kind": "rewindable",
        "scope": { "kind": "conversation", "conversationId": 1 },
        "history": "rewindable",
        "fork": "asOf",
    });
    let created_at = commit(
        &storage,
        json!([create_document(record, json!({ "count": 0 }))]),
    )
    .await
    .unwrap();
    let changed_at = commit(
        &storage,
        json!([change_delta(id, json!([["s", ["count"], 1]]))]),
    )
    .await
    .unwrap();
    commit(&storage, json!([change_base(id, json!({ "count": 2 }))]))
        .await
        .unwrap();
    commit(&storage, json!([{ "type": "document.retire", "id": id }]))
        .await
        .unwrap();

    let path = format!("{dir}/doc-{id}.jsonl");
    assert_eq!(read_lines(&path).len(), 3);
    let reopened = open_default(&dir).await;
    let value_at = |seq: Seq| {
        let reopened = reopened.clone();
        async move {
            reopened
                .document(DocumentId(id), DocumentPoint::Seq(seq), context())
                .await
                .unwrap()
                .map(|document| JsonValue::Object(document.value))
        }
    };
    assert_eq!(value_at(created_at).await, Some(json!({ "count": 0 })));
    assert_eq!(value_at(changed_at).await, Some(json!({ "count": 1 })));
    assert_eq!(current_value(&reopened, id).await, None);
    assert_eq!(read_lines(&path).len(), 3);
}

#[tokio::test]
async fn reclaims_retired_task_session_and_latest_conversation_document_sidecars() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let task_id = mint(&storage).await;
    let session_id = mint(&storage).await;
    let latest_id = mint(&storage).await;
    let task_document_id = mint(&storage).await;
    let created_at = commit(
        &storage,
        json!([
            task_write(pending_task(task_id, "ready")),
            create_document(session_document(session_id, "session"), json!({})),
            create_document(
                json!({
                    "id": latest_id,
                    "kind": "latest",
                    "scope": { "kind": "conversation", "conversationId": 1 },
                    "history": "latest",
                    "fork": "current",
                }),
                json!({}),
            ),
            create_document(
                json!({ "id": task_document_id, "kind": "task", "scope": { "kind": "task", "taskId": task_id } }),
                json!({}),
            ),
        ]),
    )
    .await
    .unwrap();
    let retired_at = commit(
        &storage,
        json!([
            { "type": "document.retire", "id": session_id },
            { "type": "document.retire", "id": latest_id },
            { "type": "document.retire", "id": task_document_id },
            task_write(terminal_task(task_id)),
        ]),
    )
    .await
    .unwrap();
    for file in [
        format!("doc-{session_id}.jsonl"),
        format!("doc-{latest_id}.jsonl"),
        format!("doc-{task_document_id}.jsonl"),
        format!("task-{task_id}.jsonl"),
    ] {
        assert!(!file_exists(&format!("{dir}/{file}")), "{file}");
    }

    let reopened = open_default(&dir).await;
    assert_eq!(
        task_json(&reopened, task_id).await,
        Some(terminal_task(task_id))
    );
    assert_eq!(current_value(&reopened, session_id).await, None);
    assert_eq!(current_value(&reopened, latest_id).await, None);
    assert_eq!(current_value(&reopened, task_document_id).await, None);
    let address = DocumentAddress {
        kind: "session".into(),
        scope: DocumentScope::Session,
        key: None,
    };
    let found = reopened
        .find_document(&address, DocumentPoint::Seq(created_at), context())
        .await
        .unwrap()
        .expect("alive at creation");
    assert_eq!(found.id, DocumentId(session_id));
    assert_eq!(found.created_at, created_at);
    assert_eq!(found.retired_at, Some(retired_at));
    assert!(
        reopened
            .find_document(&address, DocumentPoint::Seq(retired_at), context())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn truncates_torn_utf8_tails_at_exact_byte_offsets_and_reuses_the_unconfirmed_sequence() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let document_id = mint(&storage).await;
    commit(
        &storage,
        json!([create_document(
            session_document(document_id, "test.document"),
            json!({ "text": "kept" })
        )]),
    )
    .await
    .unwrap();
    let sidecar_path = format!("{dir}/doc-{document_id}.jsonl");
    let main_path = format!("{dir}/main.jsonl");
    let sidecar_size = std::fs::metadata(&sidecar_path).unwrap().len();
    let main_size = std::fs::metadata(&main_path).unwrap().len();
    let torn = "{\"text\":\"€".as_bytes();
    append_bytes(&sidecar_path, &torn[..torn.len() - 1]);
    append_bytes(&main_path, &torn[..torn.len() - 1]);

    let reopened = open_default(&dir).await;
    assert_eq!(
        std::fs::metadata(&sidecar_path).unwrap().len(),
        sidecar_size
    );
    assert_eq!(std::fs::metadata(&main_path).unwrap().len(), main_size);
    assert_eq!(
        commit(&reopened, json!([change_delta(document_id, json!([]))]))
            .await
            .unwrap(),
        Seq(3)
    );
}

#[tokio::test]
async fn removes_complete_unconfirmed_sidecar_tails_without_resurrecting_terminal_tasks() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let task_id = mint(&storage).await;
    commit(
        &storage,
        json!([task_write(pending_task(task_id, "ready"))]),
    )
    .await
    .unwrap();
    commit(&storage, json!([task_write(terminal_task(task_id))]))
        .await
        .unwrap();
    let sidecar_path = format!("{dir}/task-{task_id}.jsonl");
    assert!(!file_exists(&sidecar_path));
    let stale = json!({
        "format": 1,
        "type": "record",
        "seq": 4,
        "ordinal": 0,
        "payload": { "type": "task", "value": pending_task(task_id, "stale") },
    });
    append_bytes(&sidecar_path, format!("{stale}\n").as_bytes());

    let reopened = open_default(&dir).await;
    assert!(!file_exists(&sidecar_path));
    assert_eq!(
        task_json(&reopened, task_id).await,
        Some(terminal_task(task_id))
    );
    assert_eq!(reopened.commit(&[], context()).await.unwrap(), Seq(4));
}

async fn committed_document(dir: &str, value: JsonValue) -> u64 {
    let storage = open_default(dir).await;
    create_root(&storage).await;
    let document_id = mint(&storage).await;
    commit(
        &storage,
        json!([create_document(
            session_document(document_id, "test.document"),
            value
        )]),
    )
    .await
    .unwrap();
    document_id
}

fn assert_corruption(error: &Error) {
    assert!(
        matches!(error, Error::Thrown(thrown) if thrown.downcast_ref::<JsonlCorruptionError>().is_some()),
        "expected a JsonlCorruptionError, got {error:?}"
    );
}

#[tokio::test]
async fn fails_open_when_confirmed_sidecar_data_is_missing() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let document_id = committed_document(&dir, json!({})).await;
    std::fs::write(format!("{dir}/doc-{document_id}.jsonl"), "").unwrap();
    let error = assert_rejects(open_result(&dir).await, "Missing confirmed sidecar record");
    assert_corruption(&error);
}

#[tokio::test]
async fn rejects_a_confirmed_record_after_an_unconfirmed_sidecar_record() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let document_id = committed_document(&dir, json!({ "count": 0 })).await;
    let storage = open_default(&dir).await;
    commit(
        &storage,
        json!([change_delta(document_id, json!([["s", ["count"], 1]]))]),
    )
    .await
    .unwrap();
    let path = format!("{dir}/doc-{document_id}.jsonl");
    let lines = read_lines(&path);
    let unconfirmed = json!({
        "format": 1,
        "type": "record",
        "seq": 2,
        "ordinal": 999,
        "payload": { "type": "document", "id": document_id, "content": { "kind": "delta", "version": 1, "ops": [] } },
    });
    std::fs::write(
        &path,
        format!("{}\n{unconfirmed}\n{}\n", lines[0], lines[1]),
    )
    .unwrap();
    assert_rejects(
        open_result(&dir).await,
        "Confirmed record follows an unconfirmed tail",
    );
}

#[tokio::test]
async fn rejects_non_increasing_main_commit_sequences() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let path = format!("{dir}/main.jsonl");
    let content = std::fs::read(&path).unwrap();
    append_bytes(&path, &content);
    assert_rejects(
        open_result(&dir).await,
        "Commit sequence does not strictly increase",
    );
}

#[tokio::test]
async fn rejects_structurally_invalid_confirmed_document_content() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let document_id = committed_document(&dir, json!({})).await;
    let path = format!("{dir}/doc-{document_id}.jsonl");
    let mut record: JsonValue =
        serde_json::from_str(std::fs::read_to_string(&path).unwrap().trim()).unwrap();
    record["payload"]["content"]
        .as_object_mut()
        .unwrap()
        .remove("value");
    std::fs::write(&path, format!("{record}\n")).unwrap();
    assert_rejects(open_result(&dir).await, "Invalid document content");
}

#[tokio::test]
async fn rejects_malformed_complete_main_and_sidecar_lines() {
    let main_directory = TempDir::new("pi-durable-jsonl-");
    let main_dir = dir_path(&main_directory);
    let main_storage = open_default(&main_dir).await;
    create_root(&main_storage).await;
    append_bytes(&format!("{main_dir}/main.jsonl"), b"{bad}\n");
    assert_rejects(
        open_result(&main_dir).await,
        "Malformed complete main.jsonl",
    );

    let sidecar_directory = TempDir::new("pi-durable-jsonl-");
    let sidecar_dir = dir_path(&sidecar_directory);
    let sidecar_storage = open_default(&sidecar_dir).await;
    create_root(&sidecar_storage).await;
    std::fs::write(format!("{sidecar_dir}/doc-99.jsonl"), "{bad}\n").unwrap();
    assert_rejects(
        open_result(&sidecar_dir).await,
        "Malformed complete doc-99.jsonl",
    );
}

#[tokio::test]
async fn rejects_use_after_close() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let storage = open_default(&dir_path(&directory)).await;
    storage.close(context()).await.unwrap();
    storage.close(context()).await.unwrap();
    assert_rejects(storage.mint_id().await, "JsonlStorage is closed");
    assert_rejects(
        storage.commit(&root_write(), context()).await,
        "JsonlStorage is closed",
    );
}

// Rust-only: TS writes records through `copyJson(.., { omitUndefinedProperties })`,
// so a void task's sidecar line has no `input` or `checkpoint` key. Such a
// store must load (the keys read as `null`).
#[tokio::test]
async fn reads_ts_task_sidecar_lines_that_omit_void_fields() {
    fn strip_null_void_fields(value: &mut JsonValue) {
        match value {
            JsonValue::Object(map) => {
                for key in ["input", "checkpoint", "result"] {
                    if map.get(key) == Some(&JsonValue::Null) {
                        map.remove(key);
                    }
                }
                map.values_mut().for_each(strip_null_void_fields);
            }
            JsonValue::Array(items) => items.iter_mut().for_each(strip_null_void_fields),
            _ => {}
        }
    }

    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let task_id = mint(&storage).await;
    let mut task = pending_task(task_id, "unused");
    task["state"]["checkpoint"] = JsonValue::Null;
    commit(&storage, json!([task_write(task.clone())]))
        .await
        .unwrap();
    drop(storage);

    let path = format!("{dir}/task-{task_id}.jsonl");
    let lines = read_lines(&path);
    assert_eq!(lines.len(), 1);
    let mut line: JsonValue = serde_json::from_str(&lines[0]).unwrap();
    strip_null_void_fields(&mut line);
    let ts_line = line.to_string();
    assert!(!ts_line.contains("\"input\"") && !ts_line.contains("\"checkpoint\""));
    std::fs::write(&path, format!("{ts_line}\n")).unwrap();

    let reopened = open_default(&dir).await;
    assert_eq!(task_json(&reopened, task_id).await, Some(task));
}

// Rust-only: TS decodes each line with `TextDecoder`, which drops a leading
// byte order mark, so a BOM-prefixed line parses.
#[tokio::test]
async fn reads_a_line_with_a_leading_byte_order_mark() {
    let directory = TempDir::new("pi-durable-jsonl-");
    let dir = dir_path(&directory);
    let storage = open_default(&dir).await;
    create_root(&storage).await;
    let task_id = mint(&storage).await;
    commit(&storage, json!([task_write(pending_task(task_id, "bom"))]))
        .await
        .unwrap();
    drop(storage);

    let path = format!("{dir}/task-{task_id}.jsonl");
    let lines = read_lines(&path);
    assert_eq!(lines.len(), 1);
    std::fs::write(&path, format!("\u{FEFF}{}\n", lines[0])).unwrap();

    let reopened = open_default(&dir).await;
    assert_eq!(
        task_json(&reopened, task_id).await,
        Some(pending_task(task_id, "bom"))
    );
}
