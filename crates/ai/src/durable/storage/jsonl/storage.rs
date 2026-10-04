//! Port of durable `src/storage/jsonl/storage.ts`: [`JsonlStorage`], the
//! portable JSONL implementation of the storage contract over any
//! [`FileSystem`].
//!
//! Layout: `main.jsonl` holds one commit marker per committed batch; live task
//! records and document contents go to `task-{id}.jsonl` / `doc-{id}.jsonl`
//! sidecars, appended (and optionally flushed) before the marker that confirms
//! them. Recovery replays confirmed state into a [`MemoryStorage`], truncates
//! torn and unconfirmed tails, and reclaims sidecars that only hold
//! superseded current-only state.
//!
//! Divergences from Pi:
//! - Commits are serialized by an async lock. Pi relies on its single owning
//!   Session never overlapping commits; overlapping Rust callers would
//!   otherwise prepare the same sequence.
//! - Persisted lines are validated with Pi's structural checks and messages,
//!   then decoded into typed records; a line that passes the checks but does
//!   not decode is reported with the same message as the failed check.
//! - Main and sidecar lines are serialized from typed records, so key order
//!   follows the Rust field order. Both runtimes read either order.
//! - `JsonlCorruptionError` and `JsonlStoragePoisonedError` surface as
//!   [`Error::Thrown`]; downcast them to inspect. `errorFromFile` failures are
//!   [`Error::Message`] with the [`FileError`] as their cause.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::chord::{Context, JsonValue};
use crate::durable::env::{
    CreateDirOptions, FileError, FileErrorCode, FileKind, FileSystem, RemoveOptions,
};
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{
    ConversationId, DocumentId, EntryId, MAX_SAFE_INTEGER, Seq, SubmissionId, TaskId,
};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{
    ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentContent,
    DocumentCreate, DocumentPoint, DocumentQuery, DocumentRecord, DocumentScope, EntryQuery,
    EntryRecord, History, Page, Storage, StorageWrite, StoredDocument, StoredEntry,
    SubmissionQuery, SubmissionRecord, TaskQuery, TaskRecord, TaskStatus,
};

const FORMAT_VERSION: u64 = 1;
const MAIN_FILE: &str = "main.jsonl";
const RECLAIM_SUFFIX: &str = ".reclaim";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum MainOperation {
    #[serde(rename = "conversation")]
    Conversation { value: ConversationRecord },
    #[serde(rename = "entry")]
    Entry { value: EntryRecord },
    #[serde(rename = "submission")]
    Submission { value: SubmissionRecord },
    #[serde(rename = "document.retire")]
    DocumentRetire { id: DocumentId },
    /// A terminal task, kept in the main log.
    #[serde(rename = "task")]
    Task { value: TaskRecord },
    #[serde(rename = "task.sidecar")]
    TaskSidecar { id: TaskId, ordinal: u64 },
    #[serde(rename = "document.create")]
    DocumentCreate {
        record: DocumentCreate,
        ordinal: u64,
    },
    #[serde(rename = "document.change")]
    DocumentChange { id: DocumentId, ordinal: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MainMarker {
    format: u64,
    #[serde(rename = "type")]
    marker_type: String,
    seq: Seq,
    writes: Vec<MainOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum SidecarPayload {
    #[serde(rename = "task")]
    Task { value: Box<TaskRecord> },
    #[serde(rename = "document")]
    Document {
        id: DocumentId,
        content: DocumentContent,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SidecarRecord {
    format: u64,
    #[serde(rename = "type")]
    record_type: String,
    seq: Seq,
    ordinal: u64,
    payload: SidecarPayload,
}

#[derive(Debug, Clone)]
struct ParsedLine<T> {
    value: T,
    start: u64,
}

#[derive(Debug, Clone)]
struct ParsedFile<T> {
    path: String,
    lines: Vec<ParsedLine<T>>,
}

#[derive(Debug, Clone)]
struct EncodedCommit {
    marker: String,
    sidecars: IndexMap<String, String>,
}

/// `JsonlStorageOptions`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JsonlStorageOptions {
    /// Flush every affected sidecar before appending the main marker. Defaults to false.
    pub fsync: Option<bool>,
}

/// Persisted JSONL data that cannot be recovered (`JsonlCorruptionError`).
#[derive(Debug, Clone)]
pub struct JsonlCorruptionError {
    pub message: String,
    pub cause: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl JsonlCorruptionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            cause: None,
        }
    }

    pub fn with_cause(
        message: impl Into<String>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            cause: Some(Arc::new(cause)),
        }
    }

    /// The JS `error.name`.
    pub fn name(&self) -> &'static str {
        "JsonlCorruptionError"
    }
}

impl fmt::Display for JsonlCorruptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for JsonlCorruptionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_deref()
            .map(|cause| cause as &(dyn std::error::Error + 'static))
    }
}

/// A failed publication left unknown on-disk state; reopen to recover (`JsonlStoragePoisonedError`).
#[derive(Debug, Clone)]
pub struct JsonlStoragePoisonedError {
    pub cause: Error,
}

impl JsonlStoragePoisonedError {
    pub fn new(cause: Error) -> Self {
        Self { cause }
    }

    /// The JS `error.name`.
    pub fn name(&self) -> &'static str {
        "JsonlStoragePoisonedError"
    }
}

impl fmt::Display for JsonlStoragePoisonedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("JSONL storage is poisoned and must be reopened")
    }
}

impl std::error::Error for JsonlStoragePoisonedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

fn corruption(message: impl Into<String>) -> Error {
    Error::thrown(JsonlCorruptionError::new(message))
}

fn is_safe_integer(value: Option<&JsonValue>) -> bool {
    match value {
        Some(JsonValue::Number(number)) => {
            if let Some(integer) = number.as_u64() {
                integer <= MAX_SAFE_INTEGER
            } else if let Some(integer) = number.as_i64() {
                integer.unsigned_abs() <= MAX_SAFE_INTEGER
            } else {
                number.as_f64().is_some_and(|float| {
                    float.is_finite()
                        && float.fract() == 0.0
                        && float.abs() <= MAX_SAFE_INTEGER as f64
                })
            }
        }
        _ => false,
    }
}

fn number(value: Option<&JsonValue>) -> f64 {
    value.and_then(JsonValue::as_f64).unwrap_or(f64::NAN)
}

fn is_object(value: Option<&JsonValue>) -> bool {
    matches!(value, Some(JsonValue::Object(_)))
}

fn field<'a>(value: Option<&'a JsonValue>, key: &str) -> Option<&'a JsonValue> {
    value.and_then(|value| value.get(key))
}

fn sidecar_file_name(kind: &str, id: u64) -> String {
    format!("{kind}-{id}.jsonl")
}

/// `^(?:doc|task)-(?:0|[1-9]\d*)\.jsonl$`, plus the `.reclaim` suffix when `reclaim`.
fn matches_sidecar_name(name: &str, reclaim: bool) -> bool {
    let name = if reclaim {
        match name.strip_suffix(RECLAIM_SUFFIX) {
            Some(name) => name,
            None => return false,
        }
    } else {
        name
    };
    let Some(rest) = name
        .strip_prefix("doc-")
        .or_else(|| name.strip_prefix("task-"))
    else {
        return false;
    };
    let Some(digits) = rest.strip_suffix(".jsonl") else {
        return false;
    };
    !digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
        && (digits == "0" || !digits.starts_with('0'))
}

fn is_sidecar_file_name(name: &str) -> bool {
    matches_sidecar_name(name, false)
}

fn is_reclaim_file_name(name: &str) -> bool {
    matches_sidecar_name(name, true)
}

fn is_current_only(record: &DocumentCreate) -> bool {
    !matches!(record.scope, DocumentScope::Conversation { .. })
        || record.history == Some(History::Latest)
}

type SidecarKey = (String, u64, u64);

fn sidecar_key(file: &str, seq: Seq, ordinal: u64) -> SidecarKey {
    (file.to_owned(), seq.0, ordinal)
}

fn json_line<T: Serialize>(value: &T) -> String {
    let mut line = serde_json::to_string(value).expect("JSONL records serialize");
    line.push('\n');
    line
}

fn error_from_file(action: &str, error: FileError) -> Error {
    Error::with_cause(
        format!("JSONL {action} failed: {}", error.message),
        Error::thrown(error),
    )
}

fn parse_json(text: &str, description: &str) -> Result<JsonValue> {
    serde_json::from_str(text).map_err(|error| {
        Error::thrown(JsonlCorruptionError::with_cause(
            format!("Malformed complete {description}"),
            error,
        ))
    })
}

fn decode<T: serde::de::DeserializeOwned>(value: JsonValue, message: String) -> Result<T> {
    serde_json::from_value(value)
        .map_err(|error| Error::thrown(JsonlCorruptionError::with_cause(message, error)))
}

fn validate_main_operation(value: JsonValue, description: &str) -> Result<MainOperation> {
    let Some(write_type) = value
        .get("type")
        .and_then(JsonValue::as_str)
        .map(str::to_owned)
    else {
        return Err(corruption(format!("Invalid write in {description}")));
    };
    let v = Some(&value);
    let ordinal_ok = || is_safe_integer(field(v, "ordinal")) && number(field(v, "ordinal")) >= 0.0;
    let message = match write_type.as_str() {
        "conversation" | "entry" | "submission" => {
            let message = format!("Invalid {write_type} write in {description}");
            if !is_object(field(v, "value")) || !is_safe_integer(field(field(v, "value"), "id")) {
                return Err(corruption(message));
            }
            message
        }
        "task" => {
            let message = format!("Invalid terminal task write in {description}");
            let task = field(v, "value");
            if !is_object(task)
                || !is_safe_integer(field(task, "id"))
                || !is_object(field(task, "state"))
                || field(field(task, "state"), "status").and_then(JsonValue::as_str)
                    != Some("terminal")
            {
                return Err(corruption(message));
            }
            message
        }
        "document.retire" => {
            let message = format!("Invalid document retirement in {description}");
            if !is_safe_integer(field(v, "id")) {
                return Err(corruption(message));
            }
            message
        }
        "task.sidecar" => {
            let message = format!("Invalid task sidecar write in {description}");
            if !is_safe_integer(field(v, "id")) || !ordinal_ok() {
                return Err(corruption(message));
            }
            message
        }
        "document.create" => {
            let message = format!("Invalid document creation in {description}");
            if !is_object(field(v, "record"))
                || !is_safe_integer(field(field(v, "record"), "id"))
                || !ordinal_ok()
            {
                return Err(corruption(message));
            }
            message
        }
        "document.change" => {
            let message = format!("Invalid document change in {description}");
            if !is_safe_integer(field(v, "id")) || !ordinal_ok() {
                return Err(corruption(message));
            }
            message
        }
        _ => return Err(corruption(format!("Unknown write type in {description}"))),
    };
    decode(value, message)
}

fn parse_main_marker(text: &str, line: usize) -> Result<MainMarker> {
    let description = format!("{MAIN_FILE} line {line}");
    let value = parse_json(text, &description)?;
    let v = Some(&value);
    if !is_object(v)
        || field(v, "format").and_then(JsonValue::as_f64) != Some(FORMAT_VERSION as f64)
        || field(v, "type").and_then(JsonValue::as_str) != Some("commit")
        || !is_safe_integer(field(v, "seq"))
        || number(field(v, "seq")) < 1.0
        || !matches!(field(v, "writes"), Some(JsonValue::Array(_)))
    {
        return Err(corruption(format!(
            "Invalid commit marker in {description}"
        )));
    }
    let seq = decode::<Seq>(
        value["seq"].clone(),
        format!("Invalid commit marker in {description}"),
    )?;
    let JsonValue::Array(writes) = value["writes"].clone() else {
        unreachable!("checked above");
    };
    let writes = writes
        .into_iter()
        .map(|write| validate_main_operation(write, &description))
        .collect::<Result<Vec<_>>>()?;
    Ok(MainMarker {
        format: FORMAT_VERSION,
        marker_type: "commit".into(),
        seq,
        writes,
    })
}

fn validate_document_content(value: Option<&JsonValue>, description: &str) -> Result<()> {
    let message = || corruption(format!("Invalid document content in {description}"));
    if !is_object(value)
        || !is_safe_integer(field(value, "version"))
        || number(field(value, "version")) < 1.0
    {
        return Err(message());
    }
    match field(value, "kind").and_then(JsonValue::as_str) {
        Some("base") if is_object(field(value, "value")) => Ok(()),
        Some("delta") if matches!(field(value, "ops"), Some(JsonValue::Array(_))) => Ok(()),
        _ => Err(message()),
    }
}

fn parse_sidecar_record(text: &str, file: &str, line: usize) -> Result<SidecarRecord> {
    let description = format!("{file} line {line}");
    let value = parse_json(text, &description)?;
    let v = Some(&value);
    let payload = field(v, "payload");
    if !is_object(v)
        || field(v, "format").and_then(JsonValue::as_f64) != Some(FORMAT_VERSION as f64)
        || field(v, "type").and_then(JsonValue::as_str) != Some("record")
        || !is_safe_integer(field(v, "seq"))
        || number(field(v, "seq")) < 1.0
        || !is_safe_integer(field(v, "ordinal"))
        || number(field(v, "ordinal")) < 0.0
        || !is_object(payload)
        || !matches!(field(payload, "type"), Some(JsonValue::String(_)))
    {
        return Err(corruption(format!(
            "Invalid sidecar record in {description}"
        )));
    }
    let message = match field(payload, "type").and_then(JsonValue::as_str) {
        Some("task") => {
            let message = format!("Invalid live task record in {description}");
            let task = field(payload, "value");
            if !is_object(task)
                || !is_safe_integer(field(task, "id"))
                || !is_object(field(task, "state"))
                || field(field(task, "state"), "status").and_then(JsonValue::as_str)
                    == Some("terminal")
            {
                return Err(corruption(message));
            }
            message
        }
        Some("document") => {
            if !is_safe_integer(field(payload, "id")) {
                return Err(corruption(format!(
                    "Invalid document record in {description}"
                )));
            }
            validate_document_content(field(payload, "content"), &description)?;
            format!("Invalid document content in {description}")
        }
        _ => {
            return Err(corruption(format!(
                "Unknown sidecar record type in {description}"
            )));
        }
    };
    decode(value, message)
}

/// Sidecar bookkeeping adopted after each publication.
#[derive(Debug, Default)]
struct SidecarState {
    current_only_documents: HashSet<u64>,
    live_task_sidecars: HashSet<u64>,
}

struct Inner {
    fs: Arc<dyn FileSystem>,
    directory: String,
    main_path: String,
    fsync: bool,
    memory: MemoryStorage,
    state: Mutex<SidecarState>,
    closed: AtomicBool,
    poison_error: Mutex<Option<Arc<JsonlStoragePoisonedError>>>,
    commit_lock: tokio::sync::Mutex<()>,
}

/// Portable JSONL implementation of the storage contract (`JsonlStorage`). Clones share one storage.
#[derive(Clone)]
pub struct JsonlStorage {
    inner: Arc<Inner>,
}

impl fmt::Debug for JsonlStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonlStorage")
            .field("directory", &self.inner.directory)
            .field("fsync", &self.inner.fsync)
            .finish_non_exhaustive()
    }
}

impl JsonlStorage {
    /// Open or create a JSONL storage directory using the supplied filesystem.
    pub async fn open(
        directory: &str,
        fs: Arc<dyn FileSystem>,
        context: &Context,
        options: JsonlStorageOptions,
    ) -> Result<Self> {
        let absolute = fs
            .absolute_path(directory, context)
            .await
            .map_err(|error| error_from_file("path resolution", error))?;
        fs.create_dir(
            &absolute,
            CreateDirOptions {
                recursive: Some(true),
            },
            context,
        )
        .await
        .map_err(|error| error_from_file("directory creation", error))?;
        let main_path = fs
            .join_path(&[&absolute, MAIN_FILE], context)
            .await
            .map_err(|error| error_from_file("path join", error))?;
        let storage = Self {
            inner: Arc::new(Inner {
                fs,
                directory: absolute,
                main_path,
                fsync: options.fsync.unwrap_or(false),
                memory: MemoryStorage::new(),
                state: Mutex::new(SidecarState::default()),
                closed: AtomicBool::new(false),
                poison_error: Mutex::new(None),
                commit_lock: tokio::sync::Mutex::new(()),
            }),
        };
        storage.recover(context).await?;
        Ok(storage)
    }

    fn encode_commit(seq: Seq, writes: &[StorageWrite]) -> EncodedCommit {
        let mut main_writes = Vec::new();
        let mut records: IndexMap<String, Vec<SidecarRecord>> = IndexMap::new();
        let mut next_ordinal = 0;
        let mut add_sidecar = |file: String, payload: SidecarPayload| -> u64 {
            let ordinal = next_ordinal;
            next_ordinal += 1;
            records.entry(file).or_default().push(SidecarRecord {
                format: FORMAT_VERSION,
                record_type: "record".into(),
                seq,
                ordinal,
                payload,
            });
            ordinal
        };

        for write in writes {
            match write {
                StorageWrite::Conversation { value } => {
                    main_writes.push(MainOperation::Conversation {
                        value: value.clone(),
                    });
                }
                StorageWrite::Entry { value } => {
                    main_writes.push(MainOperation::Entry {
                        value: value.clone(),
                    });
                }
                StorageWrite::Submission { value } => {
                    main_writes.push(MainOperation::Submission {
                        value: value.clone(),
                    });
                }
                StorageWrite::DocumentRetire { id } => {
                    main_writes.push(MainOperation::DocumentRetire { id: *id });
                }
                StorageWrite::Task { value } => {
                    if value.state.status() == TaskStatus::Terminal {
                        main_writes.push(MainOperation::Task {
                            value: value.clone(),
                        });
                    } else {
                        let ordinal = add_sidecar(
                            sidecar_file_name("task", value.id.0),
                            SidecarPayload::Task {
                                value: Box::new(value.clone()),
                            },
                        );
                        main_writes.push(MainOperation::TaskSidecar {
                            id: value.id,
                            ordinal,
                        });
                    }
                }
                StorageWrite::DocumentCreate { record, content } => {
                    let ordinal = add_sidecar(
                        sidecar_file_name("doc", record.id.0),
                        SidecarPayload::Document {
                            id: record.id,
                            content: content.clone(),
                        },
                    );
                    main_writes.push(MainOperation::DocumentCreate {
                        record: record.clone(),
                        ordinal,
                    });
                }
                StorageWrite::DocumentChange { id, content } => {
                    let ordinal = add_sidecar(
                        sidecar_file_name("doc", id.0),
                        SidecarPayload::Document {
                            id: *id,
                            content: content.clone(),
                        },
                    );
                    main_writes.push(MainOperation::DocumentChange { id: *id, ordinal });
                }
                // Prepared writes resolve copies into creations.
                StorageWrite::DocumentCopy { .. } => unreachable!("document copies are resolved"),
            }
        }

        let sidecars = records
            .into_iter()
            .map(|(file, file_records)| (file, file_records.iter().map(json_line).collect()))
            .collect();
        let marker = MainMarker {
            format: FORMAT_VERSION,
            marker_type: "commit".into(),
            seq,
            writes: main_writes,
        };
        EncodedCommit {
            marker: json_line(&marker),
            sidecars,
        }
    }

    fn plan_reclamations(
        &self,
        writes: &[StorageWrite],
        encoded: &EncodedCommit,
    ) -> IndexMap<String, String> {
        let mut created_current_only_documents = HashSet::new();
        let mut retired_documents = indexmap::IndexSet::new();
        let mut base_documents = indexmap::IndexSet::new();
        let mut final_tasks: IndexMap<u64, &TaskRecord> = IndexMap::new();
        for write in writes {
            match write {
                StorageWrite::DocumentCreate { record, .. } => {
                    if is_current_only(record) {
                        created_current_only_documents.insert(record.id.0);
                    }
                }
                StorageWrite::DocumentChange { id, content } => {
                    if content.is_base() {
                        base_documents.insert(id.0);
                    }
                }
                StorageWrite::DocumentRetire { id } => {
                    retired_documents.insert(id.0);
                }
                StorageWrite::Task { value } => {
                    final_tasks.insert(value.id.0, value);
                }
                _ => {}
            }
        }

        let state = self.inner.state.lock();
        let mut replacements = IndexMap::new();
        let is_current_only_document = |id: u64| {
            state.current_only_documents.contains(&id)
                || created_current_only_documents.contains(&id)
        };
        for &id in &retired_documents {
            if is_current_only_document(id) {
                replacements.insert(sidecar_file_name("doc", id), String::new());
            }
        }
        for &id in &base_documents {
            if !is_current_only_document(id) || retired_documents.contains(&id) {
                continue;
            }
            let file = sidecar_file_name("doc", id);
            if let Some(content) = encoded.sidecars.get(&file) {
                replacements.insert(file, content.clone());
            }
        }
        for (&id, task) in &final_tasks {
            if task.state.status() == TaskStatus::Terminal
                && (state.live_task_sidecars.contains(&id)
                    || encoded
                        .sidecars
                        .contains_key(&sidecar_file_name("task", id)))
            {
                replacements.insert(sidecar_file_name("task", id), String::new());
            }
        }
        replacements
    }

    fn adopt_sidecar_state(&self, writes: &[StorageWrite]) {
        let mut state = self.inner.state.lock();
        for write in writes {
            match write {
                StorageWrite::DocumentCreate { record, .. } => {
                    if is_current_only(record) {
                        state.current_only_documents.insert(record.id.0);
                    }
                }
                StorageWrite::Task { value } => {
                    if value.state.status() == TaskStatus::Terminal {
                        state.live_task_sidecars.remove(&value.id.0);
                    } else {
                        state.live_task_sidecars.insert(value.id.0);
                    }
                }
                _ => {}
            }
        }
    }

    /// The marker already published this state, so reclamation is retryable best-effort maintenance.
    async fn reclaim_sidecars(&self, replacements: &IndexMap<String, String>, context: &Context) {
        if replacements.is_empty() {
            return;
        }
        if self.inner.fsync
            && self
                .inner
                .fs
                .flush_file(&self.inner.main_path, context)
                .await
                .is_err()
        {
            return;
        }
        for (file, content) in replacements {
            self.replace_sidecar(file, content, context).await;
        }
    }

    async fn replace_sidecar(&self, file: &str, content: &str, context: &Context) {
        let fs = &self.inner.fs;
        let Ok(path) = fs.join_path(&[&self.inner.directory, file], context).await else {
            return;
        };
        if content.is_empty() {
            let _ = fs
                .remove(
                    &path,
                    RemoveOptions {
                        recursive: None,
                        force: Some(true),
                    },
                    context,
                )
                .await;
            return;
        }
        let Ok(temporary_path) = fs
            .join_path(
                &[&self.inner.directory, &format!("{file}{RECLAIM_SUFFIX}")],
                context,
            )
            .await
        else {
            return;
        };
        if fs
            .write_file(&temporary_path, content.as_bytes(), context)
            .await
            .is_err()
        {
            return;
        }
        if self.inner.fsync && fs.flush_file(&temporary_path, context).await.is_err() {
            return;
        }
        let _ = fs.rename_file(&temporary_path, &path, context).await;
    }

    async fn recover(&self, context: &Context) -> Result<()> {
        let fs = &*self.inner.fs;
        let directory = &self.inner.directory;
        let memory = &self.inner.memory;
        let main = Self::read_lines(
            fs,
            &self.inner.main_path,
            MAIN_FILE,
            context,
            parse_main_marker,
        )
        .await?;
        let mut previous_seq = 0;
        for marker in &main.lines {
            if marker.value.seq.0 <= previous_seq {
                return Err(corruption(format!(
                    "Commit sequence does not strictly increase in {MAIN_FILE}"
                )));
            }
            previous_seq = marker.value.seq.0;
        }

        let listed = fs
            .list_dir(directory, context)
            .await
            .map_err(|error| error_from_file("directory listing", error))?;
        for info in &listed {
            if info.kind == FileKind::File && is_reclaim_file_name(&info.name) {
                let _ = fs
                    .remove(
                        &info.path,
                        RemoveOptions {
                            recursive: None,
                            force: Some(true),
                        },
                        context,
                    )
                    .await;
            }
        }
        let mut sidecar_files: Vec<String> = listed
            .iter()
            .filter(|info| info.kind == FileKind::File && is_sidecar_file_name(&info.name))
            .map(|info| info.name.clone())
            .collect();
        sidecar_files.sort();
        let mut parsed_files: IndexMap<String, ParsedFile<SidecarRecord>> = IndexMap::new();
        let mut record_by_key: HashMap<SidecarKey, SidecarRecord> = HashMap::new();
        for file in &sidecar_files {
            let path = fs
                .join_path(&[directory, file], context)
                .await
                .map_err(|error| error_from_file("path join", error))?;
            let parsed = Self::read_lines(fs, &path, file, context, |text, line| {
                parse_sidecar_record(text, file, line)
            })
            .await?;
            let mut previous: Option<&SidecarRecord> = None;
            for line in &parsed.lines {
                if let Some(previous) = previous
                    && (line.value.seq < previous.seq
                        || (line.value.seq == previous.seq
                            && line.value.ordinal <= previous.ordinal))
                {
                    return Err(corruption(format!(
                        "Sidecar records are out of order in {file}"
                    )));
                }
                previous = Some(&line.value);
                record_by_key.insert(
                    sidecar_key(file, line.value.seq, line.value.ordinal),
                    line.value.clone(),
                );
            }
            parsed_files.insert(file.clone(), parsed);
        }

        let mut current_only_documents = indexmap::IndexSet::new();
        let mut retired_documents = indexmap::IndexSet::new();
        let mut final_task_is_live: IndexMap<u64, bool> = IndexMap::new();
        for line in &main.lines {
            for operation in &line.value.writes {
                match operation {
                    MainOperation::DocumentCreate { record, .. } => {
                        if is_current_only(record) {
                            current_only_documents.insert(record.id.0);
                        }
                    }
                    MainOperation::DocumentRetire { id } => {
                        retired_documents.insert(id.0);
                    }
                    MainOperation::Task { value } => {
                        final_task_is_live.insert(value.id.0, false);
                    }
                    MainOperation::TaskSidecar { id, .. } => {
                        final_task_is_live.insert(id.0, true);
                    }
                    _ => {}
                }
            }
        }
        let retired_current_only_documents: HashSet<u64> = retired_documents
            .iter()
            .copied()
            .filter(|id| current_only_documents.contains(id))
            .collect();

        let mut latest_bases: HashMap<u64, (Seq, u64)> = HashMap::new();
        for line in &main.lines {
            let marker = &line.value;
            for operation in &marker.writes {
                let (id, ordinal) = match operation {
                    MainOperation::DocumentCreate { record, ordinal } => (record.id.0, *ordinal),
                    MainOperation::DocumentChange { id, ordinal } => (id.0, *ordinal),
                    _ => continue,
                };
                if !current_only_documents.contains(&id) {
                    continue;
                }
                let Some(record) = record_by_key.get(&sidecar_key(
                    &sidecar_file_name("doc", id),
                    marker.seq,
                    ordinal,
                )) else {
                    continue;
                };
                let SidecarPayload::Document {
                    id: payload_id,
                    content,
                } = &record.payload
                else {
                    continue;
                };
                if payload_id.0 != id || !content.is_base() {
                    continue;
                }
                let replace = match latest_bases.get(&id) {
                    None => true,
                    Some(&(seq, previous_ordinal)) => {
                        record.seq > seq || (record.seq == seq && record.ordinal > previous_ordinal)
                    }
                };
                if replace {
                    latest_bases.insert(id, (record.seq, record.ordinal));
                }
            }
        }

        let is_before_latest_base = |id: u64, seq: Seq, ordinal: u64| -> bool {
            latest_bases
                .get(&id)
                .is_some_and(|&(base_seq, base_ordinal)| {
                    seq < base_seq || (seq == base_seq && ordinal < base_ordinal)
                })
        };
        let terminal_tasks: HashSet<u64> = final_task_is_live
            .iter()
            .filter(|&(_, &live)| !live)
            .map(|(&id, _)| id)
            .collect();
        let mut confirmed: HashSet<SidecarKey> = HashSet::new();
        for line in &main.lines {
            let marker = &line.value;
            let mut writes: Vec<StorageWrite> = Vec::new();
            for operation in &marker.writes {
                match operation {
                    MainOperation::Conversation { value } => {
                        writes.push(StorageWrite::Conversation {
                            value: value.clone(),
                        });
                    }
                    MainOperation::Entry { value } => writes.push(StorageWrite::Entry {
                        value: value.clone(),
                    }),
                    MainOperation::Submission { value } => {
                        writes.push(StorageWrite::Submission {
                            value: value.clone(),
                        });
                    }
                    MainOperation::Task { value } => writes.push(StorageWrite::Task {
                        value: value.clone(),
                    }),
                    MainOperation::DocumentRetire { id } => {
                        writes.push(StorageWrite::DocumentRetire { id: *id });
                    }
                    MainOperation::TaskSidecar { id, ordinal } => {
                        let optional = terminal_tasks.contains(&id.0);
                        let record = Self::confirm_record(
                            marker,
                            *ordinal,
                            &sidecar_file_name("task", id.0),
                            &record_by_key,
                            &mut confirmed,
                            optional,
                        )?;
                        if let Some(record) = record {
                            let SidecarPayload::Task { value } = &record.payload else {
                                return Err(corruption(format!(
                                    "Confirmed task sidecar data does not match commit {}",
                                    marker.seq
                                )));
                            };
                            if value.id != *id {
                                return Err(corruption(format!(
                                    "Confirmed task sidecar data does not match commit {}",
                                    marker.seq
                                )));
                            }
                            if !optional {
                                writes.push(StorageWrite::Task {
                                    value: (**value).clone(),
                                });
                            }
                        }
                    }
                    MainOperation::DocumentCreate { record: _, ordinal }
                    | MainOperation::DocumentChange { ordinal, .. } => {
                        let id = match operation {
                            MainOperation::DocumentCreate { record, .. } => record.id,
                            MainOperation::DocumentChange { id, .. } => *id,
                            _ => unreachable!(),
                        };
                        let reclaimed = retired_current_only_documents.contains(&id.0)
                            || is_before_latest_base(id.0, marker.seq, *ordinal);
                        let record = Self::confirm_record(
                            marker,
                            *ordinal,
                            &sidecar_file_name("doc", id.0),
                            &record_by_key,
                            &mut confirmed,
                            reclaimed,
                        )?;
                        let mut content: Option<DocumentContent> = None;
                        if let Some(record) = record {
                            match &record.payload {
                                SidecarPayload::Document {
                                    id: payload_id,
                                    content: payload_content,
                                } if *payload_id == id => content = Some(payload_content.clone()),
                                _ => {
                                    return Err(corruption(format!(
                                        "Confirmed document sidecar data does not match commit {}",
                                        marker.seq
                                    )));
                                }
                            }
                        }
                        if let MainOperation::DocumentCreate { record, .. } = operation {
                            if content.as_ref().is_some_and(|content| !content.is_base()) {
                                return Err(corruption(format!(
                                    "Document creation lacks a confirmed base in commit {}",
                                    marker.seq
                                )));
                            }
                            let content = match content {
                                Some(content) if !reclaimed => content,
                                _ => DocumentContent::Base {
                                    version: 1,
                                    value: Default::default(),
                                },
                            };
                            writes.push(StorageWrite::DocumentCreate {
                                record: record.clone(),
                                content,
                            });
                        } else if !reclaimed && let Some(content) = content {
                            writes.push(StorageWrite::DocumentChange { id, content });
                        }
                    }
                }
            }
            memory
                .prepare_commit(&writes, Some(marker.seq))
                .map(|prepared| {
                    prepared.apply();
                })
                .map_err(|error| {
                    Error::thrown(JsonlCorruptionError::with_cause(
                        format!("Invalid committed state at sequence {}", marker.seq),
                        error,
                    ))
                })?;
        }

        let mut reclamations = IndexMap::new();
        for (file, parsed) in &parsed_files {
            let mut unconfirmed_at: Option<u64> = None;
            for line in &parsed.lines {
                let key = sidecar_key(file, line.value.seq, line.value.ordinal);
                if confirmed.contains(&key) {
                    if unconfirmed_at.is_some() {
                        return Err(corruption(format!(
                            "Confirmed record follows an unconfirmed tail in {file}"
                        )));
                    }
                } else if unconfirmed_at.is_none() {
                    unconfirmed_at = Some(line.start);
                }
            }
            if let Some(unconfirmed_at) = unconfirmed_at {
                fs.truncate_file(&parsed.path, unconfirmed_at, context)
                    .await
                    .map_err(|error| {
                        error_from_file(&format!("tail truncation of {file}"), error)
                    })?;
            }

            let numeric_id: u64 = file
                [file.find('-').map_or(0, |index| index + 1)..file.len() - ".jsonl".len()]
                .parse()
                .unwrap_or(u64::MAX);
            let confirmed_lines: Vec<&ParsedLine<SidecarRecord>> = parsed
                .lines
                .iter()
                .filter(|line| {
                    confirmed.contains(&sidecar_key(file, line.value.seq, line.value.ordinal))
                })
                .collect();
            let mut retained_lines: Option<Vec<&ParsedLine<SidecarRecord>>> = None;
            if file.starts_with("task-") && terminal_tasks.contains(&numeric_id) {
                retained_lines = Some(Vec::new());
            } else if file.starts_with("doc-") {
                if retired_current_only_documents.contains(&numeric_id) {
                    retained_lines = Some(Vec::new());
                } else if latest_bases.contains_key(&numeric_id) {
                    retained_lines = Some(
                        confirmed_lines
                            .iter()
                            .copied()
                            .filter(|line| {
                                !is_before_latest_base(
                                    numeric_id,
                                    line.value.seq,
                                    line.value.ordinal,
                                )
                            })
                            .collect(),
                    );
                }
            }
            if let Some(retained_lines) = retained_lines
                && (retained_lines.len() < confirmed_lines.len() || retained_lines.is_empty())
            {
                reclamations.insert(
                    file.clone(),
                    retained_lines
                        .iter()
                        .map(|line| json_line(&line.value))
                        .collect(),
                );
            }
        }
        self.reclaim_sidecars(&reclamations, context).await;

        let mut state = self.inner.state.lock();
        state.current_only_documents.extend(current_only_documents);
        for (id, live) in final_task_is_live {
            if live {
                state.live_task_sidecars.insert(id);
            }
        }
        Ok(())
    }

    fn confirm_record<'a>(
        marker: &MainMarker,
        ordinal: u64,
        file: &str,
        record_by_key: &'a HashMap<SidecarKey, SidecarRecord>,
        confirmed: &mut HashSet<SidecarKey>,
        optional: bool,
    ) -> Result<Option<&'a SidecarRecord>> {
        let key = sidecar_key(file, marker.seq, ordinal);
        if confirmed.contains(&key) {
            return Err(corruption("Sidecar record is confirmed more than once"));
        }
        let Some(record) = record_by_key.get(&key) else {
            if optional {
                return Ok(None);
            }
            return Err(corruption(format!(
                "Missing confirmed sidecar record {file} at sequence {}",
                marker.seq
            )));
        };
        confirmed.insert(key);
        Ok(Some(record))
    }

    async fn read_lines<T>(
        fs: &dyn FileSystem,
        path: &str,
        name: &str,
        context: &Context,
        parse: impl Fn(&str, usize) -> Result<T>,
    ) -> Result<ParsedFile<T>> {
        let bytes = match fs.read_binary_file(path, context).await {
            Ok(bytes) => bytes,
            Err(error) if error.code == FileErrorCode::NotFound => {
                return Ok(ParsedFile {
                    path: path.to_owned(),
                    lines: Vec::new(),
                });
            }
            Err(error) => return Err(error_from_file(&format!("read of {name}"), error)),
        };
        let mut complete_size = bytes.len();
        if complete_size > 0 && bytes[complete_size - 1] != b'\n' {
            complete_size = bytes
                .iter()
                .rposition(|&byte| byte == b'\n')
                .map_or(0, |index| index + 1);
            fs.truncate_file(path, complete_size as u64, context)
                .await
                .map_err(|error| {
                    error_from_file(&format!("torn-line truncation of {name}"), error)
                })?;
        }
        let mut lines = Vec::new();
        let mut start = 0;
        let mut line_number = 1;
        for end in 0..complete_size {
            if bytes[end] != b'\n' {
                continue;
            }
            let text = std::str::from_utf8(&bytes[start..end]).map_err(|error| {
                Error::thrown(JsonlCorruptionError::with_cause(
                    format!("Invalid UTF-8 in complete {name} line {line_number}"),
                    error,
                ))
            })?;
            lines.push(ParsedLine {
                value: parse(text, line_number)?,
                start: start as u64,
            });
            start = end + 1;
            line_number += 1;
        }
        Ok(ParsedFile {
            path: path.to_owned(),
            lines,
        })
    }

    async fn resolve_file(&self, file: &str, context: &Context) -> Result<String> {
        self.inner
            .fs
            .join_path(&[&self.inner.directory, file], context)
            .await
            .map_err(|error| error_from_file("path join", error))
    }

    fn store(&self) -> Result<&MemoryStorage> {
        self.assert_usable()?;
        Ok(&self.inner.memory)
    }

    fn poison(&self, cause: Error) -> Error {
        let mut poison = self.inner.poison_error.lock();
        let error = poison.get_or_insert_with(|| Arc::new(JsonlStoragePoisonedError::new(cause)));
        Error::Thrown(error.clone())
    }

    fn assert_usable(&self) -> Result<()> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(Error::message("JsonlStorage is closed"));
        }
        if let Some(error) = &*self.inner.poison_error.lock() {
            return Err(Error::Thrown(error.clone()));
        }
        Ok(())
    }
}

#[async_trait]
impl Storage for JsonlStorage {
    async fn commit(&self, writes: &[StorageWrite], context: &Context) -> Result<Seq> {
        let _commit = self.inner.commit_lock.lock().await;
        self.assert_usable()?;
        let prepared = self.inner.memory.prepare_commit(writes, None)?;
        let encoded = Self::encode_commit(prepared.seq, &prepared.writes);
        let reclamations = self.plan_reclamations(&prepared.writes, &encoded);
        let mut sidecars = Vec::with_capacity(encoded.sidecars.len());
        for (file, content) in &encoded.sidecars {
            sidecars.push((file, content, self.resolve_file(file, context).await?));
        }

        let fs = &self.inner.fs;
        for (file, content, path) in &sidecars {
            if let Err(error) = fs.append_file(path, content.as_bytes(), context).await {
                return Err(self.poison(error_from_file(&format!("append to {file}"), error)));
            }
        }
        if self.inner.fsync {
            for (file, _, path) in &sidecars {
                if let Err(error) = fs.flush_file(path, context).await {
                    return Err(self.poison(error_from_file(&format!("flush of {file}"), error)));
                }
            }
        }
        if let Err(error) = fs
            .append_file(&self.inner.main_path, encoded.marker.as_bytes(), context)
            .await
        {
            return Err(self.poison(error_from_file(&format!("append to {MAIN_FILE}"), error)));
        }
        let seq = prepared.apply();
        self.adopt_sidecar_state(&prepared.writes);
        self.reclaim_sidecars(&reclamations, context).await;
        Ok(seq)
    }

    async fn mint_id(&self) -> Result<u64> {
        self.store()?.mint_id().await
    }

    async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<ConversationRecord>> {
        self.store()?.conversation(id, context).await
    }

    async fn scan_conversations(
        &self,
        query: &ConversationQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<ConversationRecord>> {
        self.store()?
            .scan_conversations(query, limit, cursor, context)
            .await
    }

    async fn entry(&self, id: EntryId, context: &Context) -> Result<Option<StoredEntry>> {
        self.store()?.entry(id, context).await
    }

    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        context: &Context,
    ) -> Result<Option<StoredEntry>> {
        self.store()?.entry_in(conversation_id, id, context).await
    }

    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
        context: &Context,
    ) -> Result<Option<EntryRecord>> {
        self.store()?
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
        self.store()?
            .scan_entries(query, limit, cursor, context)
            .await
    }

    async fn task(&self, id: TaskId, context: &Context) -> Result<Option<TaskRecord>> {
        self.store()?.task(id, context).await
    }

    async fn scan_tasks(
        &self,
        query: &TaskQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<TaskRecord>> {
        self.store()?
            .scan_tasks(query, limit, cursor, context)
            .await
    }

    async fn submission(
        &self,
        id: SubmissionId,
        context: &Context,
    ) -> Result<Option<SubmissionRecord>> {
        self.store()?.submission(id, context).await
    }

    async fn scan_submissions(
        &self,
        query: &SubmissionQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<SubmissionRecord>> {
        self.store()?
            .scan_submissions(query, limit, cursor, context)
            .await
    }

    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        context: &Context,
    ) -> Result<Option<SubmissionRecord>> {
        self.store()?
            .submission_by_request(conversation_id, request_id, context)
            .await
    }

    async fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        context: &Context,
    ) -> Result<Option<DocumentRecord>> {
        self.store()?.find_document(address, at, context).await
    }

    async fn document(
        &self,
        id: DocumentId,
        at: DocumentPoint,
        context: &Context,
    ) -> Result<Option<StoredDocument>> {
        self.store()?.document(id, at, context).await
    }

    async fn scan_documents(
        &self,
        query: &DocumentQuery,
        limit: usize,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Result<Page<DocumentRecord>> {
        self.store()?
            .scan_documents(query, limit, cursor, context)
            .await
    }

    async fn close(&self, context: &Context) -> Result<()> {
        if self.inner.closed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        self.inner.memory.close(context).await
    }
}
