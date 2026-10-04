//! Shared helpers for the storage and environment tests.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

use crate::chord::{BACKGROUND_CONTEXT, Context, JsonValue};
use crate::durable::types::{Page, StorageWrite, StoredDocument, StoredEntry};

pub(crate) fn context() -> &'static Context {
    &BACKGROUND_CONTEXT
}

/// A unique temporary directory removed on drop (`mkdtemp` + `rm -rf` in `afterEach`).
pub(crate) struct TempDir(PathBuf);

impl TempDir {
    pub(crate) fn new(prefix: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "{prefix}{}-{nanos}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    pub(crate) fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Decode storage writes from Pi's JSON wire shape.
pub(crate) fn writes(value: JsonValue) -> Vec<StorageWrite> {
    serde_json::from_value(value).unwrap_or_else(|error| panic!("invalid storage writes: {error}"))
}

pub(crate) fn root_write() -> Vec<StorageWrite> {
    writes(json!([{ "type": "conversation", "value": { "id": 1 } }]))
}

pub(crate) fn pending_task(id: u64, phase: &str) -> JsonValue {
    json!({
        "id": id,
        "conversationId": 1,
        "kind": "test.task",
        "version": 1,
        "input": null,
        "state": { "status": "pending", "checkpoint": { "phase": phase } },
        "background": false,
        "abortRequested": false,
    })
}

pub(crate) fn json<T: serde::Serialize>(value: &T) -> JsonValue {
    serde_json::to_value(value).expect("serializable")
}

pub(crate) fn entry_json(stored: &StoredEntry) -> JsonValue {
    json!({ "entry": json(&stored.entry), "commitSeq": stored.commit_seq })
}

pub(crate) fn document_json(stored: &StoredDocument) -> JsonValue {
    json!({
        "record": json(&stored.record),
        "version": stored.version,
        "value": stored.value,
        "deltasSinceBase": stored.deltas_since_base,
    })
}

pub(crate) fn page_json<T: serde::Serialize>(page: &Page<T>) -> JsonValue {
    let mut value = json!({ "items": json(&page.items) });
    if let Some(next) = &page.next {
        value["next"] = JsonValue::Object(next.clone());
    }
    value
}
