//! Port of `harness-fx/src/fx-session-storage.ts`: a caller's append-only log
//! of fx session states.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::resume_state::FxSessionState;
use super::transport::FxProvider;
use crate::harness::error::{HarnessError, HarnessResult};

/// `FxSessionStorageEntry`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FxSessionStorageEntry {
    pub entry_id: String,
    /// The entry this one continues from; absent on a session's first entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_entry_id: Option<String>,
    /// Opaque JSON written by the harness.
    pub data: Map<String, Value>,
}

/// `FxSessionStorage`: the caller's append-only log for fx sessions, such as
/// an object store, a database table or a stream. When a session detaches or
/// stops, the harness appends one entry holding what changed since the
/// previous one, and the resume state holds only that entry's id.
#[async_trait]
pub trait FxSessionStorage: Send + Sync {
    /// Appends one entry. An entry id the log already holds is a retry.
    async fn append(&self, session_id: &str, entry: FxSessionStorageEntry) -> HarnessResult<()>;

    /// The entries leading to `leaf_id`: it, its parent, and so on back to
    /// the first. Other entries of the session may come too, in any order.
    async fn read(
        &self,
        session_id: &str,
        leaf_id: &str,
        cancellation_token: Option<CancellationToken>,
    ) -> HarnessResult<Vec<Value>>;
}

/// An entry's data: the fields that changed. `turn: null` records that the
/// unfinished turn ended.
#[derive(Debug, Default, Deserialize)]
struct EntryData {
    provider: Option<FxProvider>,
    checkpoint: Option<String>,
    #[serde(default, deserialize_with = "present")]
    turn: Option<Value>,
}

/// Distinguishes `turn: null` (`Some(Null)`) from no `turn` (`None`).
fn present<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Entry {
    entry_id: String,
    parent_entry_id: Option<String>,
    data: EntryData,
}

/// `readFxSessionState({ storage, sessionId, leafId })`.
pub async fn read_fx_session_state(
    storage: &dyn FxSessionStorage,
    session_id: &str,
    leaf_id: &str,
    cancellation_token: Option<CancellationToken>,
) -> HarnessResult<FxSessionState> {
    let values = storage.read(session_id, leaf_id, cancellation_token).await?;
    let mut entries = std::collections::HashMap::new();
    for (index, value) in values.into_iter().enumerate() {
        let entry: Entry = serde_json::from_value(value).map_err(|error| {
            HarnessError::with_cause(
                format!("Entry {index} of the fx session \"{session_id}\" is not an fx session entry."),
                error,
            )
        })?;
        entries.insert(entry.entry_id.clone(), entry);
    }
    let mut chain = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut entry_id = Some(leaf_id.to_string());
    while let Some(id) = entry_id {
        if !visited.insert(id.clone()) {
            return Err(HarnessError::new(format!(
                "The entries of the fx session \"{session_id}\" form a loop at entry \"{id}\"."
            ))
            .into());
        }
        let Some(entry) = entries.remove(&id) else {
            return Err(HarnessError::new(format!(
                "The session storage has no entry \"{id}\" for the fx session \"{session_id}\"."
            ))
            .into());
        };
        entry_id = entry.parent_entry_id.clone();
        chain.push(entry.data);
    }
    let (mut provider, mut checkpoint, mut turn) = (None, None, None);
    for data in chain.into_iter().rev() {
        if data.provider.is_some() {
            provider = data.provider;
        }
        if data.checkpoint.is_some() {
            checkpoint = data.checkpoint;
        }
        match data.turn {
            Some(Value::Null) => turn = None,
            Some(value) => turn = Some(value),
            None => {}
        }
    }
    let Some(provider) = provider else {
        return Err(HarnessError::new(format!(
            "The entries leading to \"{leaf_id}\" do not form an fx session for \"{session_id}\"."
        ))
        .into());
    };
    Ok(FxSessionState {
        provider,
        checkpoint,
        turn,
    })
}

#[derive(Clone, PartialEq)]
struct Saved {
    provider: FxProvider,
    checkpoint: Option<String>,
    turn: Option<String>,
}

impl From<&FxSessionState> for Saved {
    fn from(state: &FxSessionState) -> Self {
        Self {
            provider: state.provider,
            checkpoint: state.checkpoint.clone(),
            turn: state.turn.as_ref().map(Value::to_string),
        }
    }
}

struct WriterState {
    leaf_id: Option<String>,
    saved: Option<Saved>,
}

/// `createFxSessionStorageWriter`: appends the changes since the last
/// written state. Writes run one at a time, in order.
#[derive(Clone)]
pub struct FxSessionStorageWriter {
    storage: Arc<dyn FxSessionStorage>,
    session_id: String,
    state: Arc<Mutex<WriterState>>,
}

impl FxSessionStorageWriter {
    pub fn new(
        storage: Arc<dyn FxSessionStorage>,
        session_id: impl Into<String>,
        leaf_id: Option<String>,
        saved_state: Option<&FxSessionState>,
    ) -> Self {
        Self {
            storage,
            session_id: session_id.into(),
            state: Arc::new(Mutex::new(WriterState {
                leaf_id,
                saved: saved_state.map(Saved::from),
            })),
        }
    }

    /// Appends what changed and returns the id of the entry the session now
    /// ends at.
    pub async fn write(&self, state: &FxSessionState) -> HarnessResult<String> {
        let snapshot = state.clone();
        let mut writer = self.state.lock().await;
        let current = Saved::from(&snapshot);
        let mut data = Map::new();
        if writer.saved.as_ref().map(|saved| saved.provider) != Some(current.provider) {
            data.insert("provider".into(), serde_json::to_value(snapshot.provider)?);
        }
        if let Some(checkpoint) = &current.checkpoint
            && writer.saved.as_ref().and_then(|saved| saved.checkpoint.as_ref()) != Some(checkpoint)
        {
            data.insert("checkpoint".into(), Value::String(checkpoint.clone()));
        }
        if writer.saved.as_ref().map(|saved| &saved.turn) != Some(&current.turn) {
            data.insert("turn".into(), snapshot.turn.clone().unwrap_or(Value::Null));
        }
        if let Some(leaf_id) = &writer.leaf_id
            && data.is_empty()
        {
            return Ok(leaf_id.clone());
        }
        let entry_id = crate::harness::utils::random_uuid();
        self.storage
            .append(
                &self.session_id,
                FxSessionStorageEntry {
                    entry_id: entry_id.clone(),
                    parent_entry_id: writer.leaf_id.clone(),
                    data,
                },
            )
            .await?;
        writer.leaf_id = Some(entry_id.clone());
        writer.saved = Some(current);
        Ok(entry_id)
    }
}

#[cfg(test)]
#[path = "session_storage_tests.rs"]
mod tests;
