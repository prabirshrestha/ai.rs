//! Pi's session log: ports of `pi-resume-state.ts` and
//! `pi-session-storage.ts`, with the parts of Pi's `SessionManager` the
//! harness uses (a header, then a tree of entries, and the branch that ends at
//! the leaf).
//!
//! [rust] Message entries hold ai.rs [`Message`]s. Entries written under an
//! older session version are read as they are, and there are no pending tool
//! batches: approvals run inline, so a turn never parks.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::utils::{iso_timestamp_now, random_uuid};
use crate::{AgentMessage, AgentResult};

/// `CURRENT_SESSION_VERSION`.
pub const CURRENT_PI_SESSION_VERSION: u64 = 3;

/// One entry of a session log (`PiSessionStorageEntry`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiSessionStorageEntry {
    pub entry_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_entry_id: Option<String>,
    /// The session version the entry was written under.
    pub version: u64,
    pub data: Value,
}

/// `PiSessionStorage`: a caller-provided append-only log, such as an object
/// store, a database table or a stream. With it, the resume state holds only
/// the id of the entry the session ends at.
#[async_trait]
pub trait PiSessionStorage: Send + Sync {
    /// Append one entry. An entry whose id the log already holds is a retry
    /// and can be ignored.
    async fn append(
        &self,
        session_id: &str,
        entry: PiSessionStorageEntry,
        abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<()>;

    /// Append several entries in order.
    async fn append_many(
        &self,
        session_id: &str,
        entries: Vec<PiSessionStorageEntry>,
        abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<()> {
        for entry in entries {
            self.append(session_id, entry, abort_signal.clone()).await?;
        }
        Ok(())
    }

    /// The entries that lead to `leaf_id`, in append order. Returning every
    /// entry of the log is also valid.
    async fn read(
        &self,
        session_id: &str,
        leaf_id: &str,
        abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<Vec<Value>>;
}

/// `PiResumeStateData`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiResumeStateData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entries: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leaf_id: Option<String>,
}

/// `PI_SESSION_ID_PATTERN`.
pub fn is_pi_session_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    let edge = |b: u8| b.is_ascii_alphanumeric();
    !bytes.is_empty()
        && edge(bytes[0])
        && edge(bytes[bytes.len() - 1])
        && bytes
            .iter()
            .all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn is_header(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("session")
        && value.get("id").is_some_and(Value::is_string)
        && value.get("timestamp").is_some_and(Value::is_string)
        && value.get("cwd").is_some_and(Value::is_string)
}

fn is_entry(value: &Value) -> bool {
    value.is_object()
        && value
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind != "session")
}

/// `piSessionEntriesSchema`: a header, then entries.
pub fn validate_pi_session_entries(entries: &[Value]) -> Result<(), HarnessError> {
    match entries.split_first() {
        Some((header, rest)) if is_header(header) => {
            match rest.iter().position(|entry| !is_entry(entry)) {
                Some(index) => Err(HarnessError::new(format!(
                    "Pi session entry {} must be an object with a type.",
                    index + 1
                ))),
                None => Ok(()),
            }
        }
        _ => Err(HarnessError::new(
            "Pi session entries must start with a session header.",
        )),
    }
}

/// `withSessionId(entries, sessionId)`: rewrite the header id.
pub fn with_session_id(mut entries: Vec<Value>, session_id: &str) -> Vec<Value> {
    if let Some(header) = entries.first_mut().and_then(Value::as_object_mut) {
        header.insert("id".into(), Value::String(session_id.into()));
    }
    entries
}

fn now_iso() -> String {
    iso_timestamp_now()
}

fn entry_id(entry: &Value) -> Option<&str> {
    entry.get("id").and_then(Value::as_str)
}

fn parent_id(entry: &Value) -> Option<&str> {
    entry.get("parentId").and_then(Value::as_str)
}

/// An in-memory Pi session (`SessionManager.inMemory`).
#[derive(Debug, Clone)]
pub struct PiSessionLog {
    header: Value,
    entries: Vec<Value>,
    leaf_id: Option<String>,
    /// Entries already in the session storage.
    saved_count: usize,
}

impl PiSessionLog {
    /// A fresh session.
    pub fn new(session_id: Option<&str>, cwd: &str) -> Self {
        Self {
            header: json!({
                "type": "session",
                "version": CURRENT_PI_SESSION_VERSION,
                "id": session_id.map_or_else(random_uuid, String::from),
                "timestamp": now_iso(),
                "cwd": cwd,
            }),
            entries: Vec::new(),
            leaf_id: None,
            saved_count: 0,
        }
    }

    /// A session seeded with `entries` (header first), on the branch that
    /// ends at `leaf_id` or at the last entry.
    pub fn from_entries(
        entries: Vec<Value>,
        leaf_id: Option<&str>,
        saved_count: usize,
    ) -> Result<Self, HarnessError> {
        validate_pi_session_entries(&entries)?;
        let mut entries = entries.into_iter();
        let header = entries.next().expect("validated");
        let entries: Vec<Value> = entries.collect();
        let leaf_id = match leaf_id {
            Some(leaf_id) => {
                if !entries.iter().any(|entry| entry_id(entry) == Some(leaf_id)) {
                    return Err(HarnessError::new(format!("Entry {leaf_id} not found")));
                }
                Some(leaf_id.to_string())
            }
            None => entries.last().and_then(entry_id).map(String::from),
        };
        Ok(Self {
            header,
            entries,
            leaf_id,
            saved_count,
        })
    }

    pub fn leaf_id(&self) -> Option<&str> {
        self.leaf_id.as_deref()
    }

    pub fn header_version(&self) -> u64 {
        self.header
            .get("version")
            .and_then(Value::as_u64)
            .unwrap_or(CURRENT_PI_SESSION_VERSION)
    }

    /// The header followed by every entry.
    pub fn entries(&self) -> Vec<Value> {
        std::iter::once(self.header.clone())
            .chain(self.entries.iter().cloned())
            .collect()
    }

    /// The entries of the current branch, root first.
    pub fn branch(&self) -> Vec<&Value> {
        let by_id: HashMap<&str, &Value> = self
            .entries
            .iter()
            .filter_map(|entry| entry_id(entry).map(|id| (id, entry)))
            .collect();
        let mut branch = Vec::new();
        let mut next = self.leaf_id.as_deref();
        while let Some(id) = next {
            let Some(entry) = by_id.get(id) else { break };
            branch.push(*entry);
            next = parent_id(entry);
        }
        branch.reverse();
        branch
    }

    /// The messages of the current branch (`buildSessionContext`).
    pub fn messages(&self) -> AgentResult<Vec<AgentMessage>> {
        self.branch()
            .into_iter()
            .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("message"))
            .filter_map(|entry| entry.get("message").cloned())
            .map(|message| {
                serde_json::from_value(message).map_err(|error| {
                    HarnessError::with_cause(format!("Invalid Pi session message: {error}"), error)
                        .into()
                })
            })
            .collect()
    }

    fn generate_id(&self) -> String {
        loop {
            let id: String = random_uuid().chars().take(8).collect();
            if !self
                .entries
                .iter()
                .any(|entry| entry_id(entry) == Some(&id))
            {
                return id;
            }
        }
    }

    /// `appendMessage(message)`.
    pub fn append_message(&mut self, message: &AgentMessage) -> String {
        let id = self.generate_id();
        self.entries.push(json!({
            "type": "message",
            "id": id,
            "parentId": self.leaf_id,
            "timestamp": now_iso(),
            "message": message,
        }));
        self.leaf_id = Some(id.clone());
        id
    }

    /// The header and the entries not yet in the storage.
    pub fn unsaved_entries(&self) -> Vec<Value> {
        std::iter::once(self.header.clone())
            .chain(self.entries[self.saved_count..].iter().cloned())
            .collect()
    }

    /// `createLifecycleData`: append unsaved entries to `storage`, then the
    /// resume data.
    pub async fn resume_data(
        &mut self,
        storage: Option<&dyn PiSessionStorage>,
        session_id: &str,
    ) -> HarnessResult<PiResumeStateData> {
        match storage {
            None => Ok(PiResumeStateData {
                entries: Some(self.unsaved_entries()),
                leaf_id: None,
            }),
            Some(storage) => {
                let unsaved = &self.entries[self.saved_count..];
                if !unsaved.is_empty() {
                    let version = self.header_version();
                    let stored = unsaved
                        .iter()
                        .map(|entry| PiSessionStorageEntry {
                            entry_id: entry_id(entry).unwrap_or_default().to_string(),
                            parent_entry_id: parent_id(entry).map(String::from),
                            version,
                            data: entry.clone(),
                        })
                        .collect();
                    storage.append_many(session_id, stored, None).await?;
                    self.saved_count = self.entries.len();
                }
                Ok(PiResumeStateData {
                    entries: None,
                    leaf_id: self.leaf_id.clone(),
                })
            }
        }
    }
}

/// `readSessionEntries`: read and check the entries that lead to `leaf_id`.
pub async fn read_pi_session_entries(
    storage: &dyn PiSessionStorage,
    session_id: &str,
    leaf_id: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<Vec<Value>> {
    let values = storage.read(session_id, leaf_id, abort_signal).await?;
    values
        .into_iter()
        .enumerate()
        .map(
            |(index, value)| match serde_json::from_value::<PiSessionStorageEntry>(value) {
                Ok(entry) if is_entry(&entry.data) => Ok(entry.data),
                _ => Err(HarnessError::new(format!(
                    "Entry {index} of the Pi session \"{session_id}\" is not a Pi session entry."
                ))
                .into()),
            },
        )
        .collect()
}

#[cfg(test)]
pub(crate) mod memory_storage {
    use std::sync::Mutex;

    use super::*;

    /// A log kept in memory, recording each call.
    #[derive(Default)]
    pub struct MemoryPiSessionStorage {
        pub entries: Mutex<Vec<(String, PiSessionStorageEntry)>>,
        pub append_calls: Mutex<usize>,
    }

    #[async_trait]
    impl PiSessionStorage for MemoryPiSessionStorage {
        async fn append(
            &self,
            session_id: &str,
            entry: PiSessionStorageEntry,
            _abort_signal: Option<CancellationToken>,
        ) -> HarnessResult<()> {
            *self.append_calls.lock().unwrap() += 1;
            self.entries
                .lock()
                .unwrap()
                .push((session_id.to_string(), entry));
            Ok(())
        }

        async fn read(
            &self,
            session_id: &str,
            _leaf_id: &str,
            _abort_signal: Option<CancellationToken>,
        ) -> HarnessResult<Vec<Value>> {
            Ok(self
                .entries
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, _)| id == session_id)
                .map(|(_, entry)| serde_json::to_value(entry).unwrap())
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::memory_storage::MemoryPiSessionStorage;
    use super::*;
    use crate::Message;

    fn header() -> Value {
        json!({ "type": "session", "version": 3, "id": "s", "timestamp": "t", "cwd": "/w" })
    }

    #[test]
    fn validates_entries() {
        assert!(validate_pi_session_entries(&[header(), json!({ "type": "message" })]).is_ok());
        assert!(validate_pi_session_entries(&[json!({ "type": "message" })]).is_err());
        assert!(validate_pi_session_entries(&[]).is_err());
        assert!(validate_pi_session_entries(&[header(), json!({ "type": "session" })]).is_err());
    }

    #[test]
    fn rewrites_only_the_header_id() {
        let entries = with_session_id(
            vec![header(), json!({ "type": "message", "id": "a" })],
            "new",
        );
        assert_eq!(entries[0]["id"], "new");
        assert_eq!(entries[1]["id"], "a");
    }

    #[test]
    fn checks_session_ids() {
        assert!(is_pi_session_id("abc-1.2_x"));
        assert!(!is_pi_session_id("-abc"));
        assert!(!is_pi_session_id("abc/def"));
        assert!(!is_pi_session_id(""));
    }

    #[test]
    fn follows_the_branch_that_ends_at_the_leaf() {
        let mut log = PiSessionLog::new(Some("s"), "/w");
        let first = log.append_message(&Message::user_text("one"));
        log.append_message(&Message::user_text("two"));
        let mut branched = PiSessionLog::from_entries(log.entries(), Some(&first), 0).unwrap();
        branched.append_message(&Message::user_text("three"));
        let texts: Vec<_> = branched
            .messages()
            .unwrap()
            .into_iter()
            .map(|message| match message {
                Message::User(user) => format!("{:?}", user.content),
                _ => String::new(),
            })
            .collect();
        assert_eq!(texts, [r#"Text("one")"#, r#"Text("three")"#]);
        assert!(PiSessionLog::from_entries(log.entries(), Some("missing"), 0).is_err());
    }

    #[tokio::test]
    async fn appends_new_entries_to_the_storage_and_resumes_from_the_leaf() {
        let storage = MemoryPiSessionStorage::default();
        let mut log = PiSessionLog::new(Some("s"), "/w");
        log.append_message(&Message::user_text("one"));
        log.append_message(&Message::user_text("two"));
        let data = log.resume_data(Some(&storage), "s").await.unwrap();
        assert_eq!(data.entries, None);
        assert_eq!(data.leaf_id.as_deref(), log.leaf_id());
        assert_eq!(storage.entries.lock().unwrap().len(), 2);
        // Nothing new: nothing appended.
        log.resume_data(Some(&storage), "s").await.unwrap();
        assert_eq!(*storage.append_calls.lock().unwrap(), 2);

        let stored = read_pi_session_entries(&storage, "s", data.leaf_id.as_deref().unwrap(), None)
            .await
            .unwrap();
        let mut seeded = vec![PiSessionLog::new(Some("s"), "/w").entries().remove(0)];
        seeded.extend(stored);
        let resumed = PiSessionLog::from_entries(seeded, data.leaf_id.as_deref(), 2).unwrap();
        assert_eq!(resumed.messages().unwrap().len(), 2);
        assert_eq!(resumed.unsaved_entries().len(), 1);
    }

    #[tokio::test]
    async fn rejects_entries_that_are_not_pi_entries() {
        let storage = MemoryPiSessionStorage::default();
        storage.entries.lock().unwrap().push((
            "s".into(),
            PiSessionStorageEntry {
                entry_id: "a".into(),
                parent_entry_id: None,
                version: 3,
                data: json!("nope"),
            },
        ));
        let error = read_pi_session_entries(&storage, "s", "a", None)
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Entry 0 of the Pi session \"s\" is not a Pi session entry."
        );
    }

    #[tokio::test]
    async fn holds_the_whole_session_without_a_storage() {
        let mut log = PiSessionLog::new(Some("s"), "/w");
        log.append_message(&Message::user_text("one"));
        let data = log.resume_data(None, "s").await.unwrap();
        assert_eq!(data.entries.unwrap().len(), 2);
        assert_eq!(data.leaf_id, None);
    }
}
