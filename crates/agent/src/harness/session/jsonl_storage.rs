use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use super::repo_utils::create_entry_id;
use crate::harness::types::{
    JsonlSessionMetadata, LeafEntry, SessionError, SessionErrorCode, SessionResult, SessionStorage,
    SessionTreeEntry, SessionTreeEntryType,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionHeader {
    #[serde(rename = "type")]
    kind: String,
    version: u8,
    id: String,
    timestamp: String,
    cwd: String,
    #[serde(rename = "parentSession", skip_serializing_if = "Option::is_none")]
    parent_session: Option<String>,
}

#[derive(Debug)]
struct JsonlSessionStorageState {
    entries: Vec<SessionTreeEntry>,
    by_id: HashMap<String, SessionTreeEntry>,
    labels_by_id: HashMap<String, String>,
    current_leaf_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct JsonlSessionStorage {
    file_path: PathBuf,
    metadata: JsonlSessionMetadata,
    state: Arc<Mutex<JsonlSessionStorageState>>,
}

impl JsonlSessionStorage {
    pub async fn open(file_path: impl AsRef<Path>) -> SessionResult<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        let loaded = load_jsonl_storage(&file_path).await?;
        Ok(Self::from_loaded(
            file_path,
            loaded.header,
            loaded.entries,
            loaded.leaf_id,
        ))
    }

    pub async fn create(
        file_path: impl AsRef<Path>,
        cwd: impl Into<String>,
        session_id: impl Into<String>,
        parent_session_path: Option<String>,
    ) -> SessionResult<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        let header = SessionHeader {
            kind: "session".to_string(),
            version: 3,
            id: session_id.into(),
            timestamp: super::create_timestamp(),
            cwd: cwd.into(),
            parent_session: parent_session_path,
        };
        let line = serde_json::to_string(&header).map_err(storage_error)?;
        tokio::fs::write(&file_path, format!("{line}\n"))
            .await
            .map_err(|err| {
                io_error(
                    err,
                    format!("Failed to create session {}", file_path.display()),
                )
            })?;
        Ok(Self::from_loaded(file_path, header, Vec::new(), None))
    }

    fn from_loaded(
        file_path: PathBuf,
        header: SessionHeader,
        entries: Vec<SessionTreeEntry>,
        leaf_id: Option<String>,
    ) -> Self {
        let by_id = entries
            .iter()
            .map(|entry| (entry.id().to_string(), entry.clone()))
            .collect::<HashMap<_, _>>();
        let labels_by_id = build_labels_by_id(&entries);
        let metadata = header_to_session_metadata(&header, &file_path);
        Self {
            file_path,
            metadata,
            state: Arc::new(Mutex::new(JsonlSessionStorageState {
                entries,
                by_id,
                labels_by_id,
                current_leaf_id: leaf_id,
            })),
        }
    }

    async fn append_line(&self, entry: &SessionTreeEntry) -> SessionResult<()> {
        let line = serde_json::to_string(entry).map_err(storage_error)?;
        append_line(
            &self.file_path,
            &line,
            format!("Failed to append session entry {}", entry.id()),
        )
        .await
    }
}

#[async_trait]
impl SessionStorage<JsonlSessionMetadata> for JsonlSessionStorage {
    async fn get_metadata(&self) -> SessionResult<JsonlSessionMetadata> {
        Ok(self.metadata.clone())
    }

    async fn get_leaf_id(&self) -> SessionResult<Option<String>> {
        let state = self.state.lock().await;
        if let Some(leaf_id) = state.current_leaf_id.as_deref()
            && !state.by_id.contains_key(leaf_id)
        {
            return Err(SessionError::new(
                SessionErrorCode::InvalidSession,
                format!("Entry {leaf_id} not found"),
            ));
        }
        Ok(state.current_leaf_id.clone())
    }

    async fn set_leaf_id(&self, leaf_id: Option<String>) -> SessionResult<()> {
        let mut state = self.state.lock().await;
        if let Some(leaf_id) = leaf_id.as_deref()
            && !state.by_id.contains_key(leaf_id)
        {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Entry {leaf_id} not found"),
            ));
        }

        let existing_ids = state.by_id.keys().cloned().collect::<HashSet<_>>();
        let entry = SessionTreeEntry::Leaf(LeafEntry {
            id: create_entry_id(&existing_ids),
            parent_id: state.current_leaf_id.clone(),
            timestamp: super::create_timestamp(),
            target_id: leaf_id.clone(),
        });
        self.append_line(&entry).await?;
        state.entries.push(entry.clone());
        state.by_id.insert(entry.id().to_string(), entry);
        state.current_leaf_id = leaf_id;
        Ok(())
    }

    async fn create_entry_id(&self) -> SessionResult<String> {
        let state = self.state.lock().await;
        let existing_ids = state.by_id.keys().cloned().collect::<HashSet<_>>();
        Ok(create_entry_id(&existing_ids))
    }

    async fn append_entry(&self, entry: SessionTreeEntry) -> SessionResult<()> {
        self.append_line(&entry).await?;
        let mut state = self.state.lock().await;
        state.entries.push(entry.clone());
        state.by_id.insert(entry.id().to_string(), entry.clone());
        update_label_cache(&mut state.labels_by_id, &entry);
        state.current_leaf_id = leaf_id_after_entry(&entry);
        Ok(())
    }

    async fn get_entry(&self, id: &str) -> SessionResult<Option<SessionTreeEntry>> {
        Ok(self.state.lock().await.by_id.get(id).cloned())
    }

    async fn find_entries(
        &self,
        entry_type: SessionTreeEntryType,
    ) -> SessionResult<Vec<SessionTreeEntry>> {
        Ok(self
            .state
            .lock()
            .await
            .entries
            .iter()
            .filter(|entry| entry.entry_type() == entry_type)
            .cloned()
            .collect())
    }

    async fn get_label(&self, id: &str) -> SessionResult<Option<String>> {
        Ok(self.state.lock().await.labels_by_id.get(id).cloned())
    }

    async fn get_path_to_root(
        &self,
        leaf_id: Option<String>,
    ) -> SessionResult<Vec<SessionTreeEntry>> {
        let Some(mut current_id) = leaf_id else {
            return Ok(Vec::new());
        };
        let state = self.state.lock().await;
        let mut path = Vec::new();
        loop {
            let Some(current) = state.by_id.get(&current_id) else {
                return Err(SessionError::new(
                    SessionErrorCode::NotFound,
                    format!("Entry {current_id} not found"),
                ));
            };
            path.push(current.clone());
            let Some(parent_id) = current.parent_id() else {
                break;
            };
            if !state.by_id.contains_key(parent_id) {
                return Err(SessionError::new(
                    SessionErrorCode::InvalidSession,
                    format!("Entry {parent_id} not found"),
                ));
            }
            current_id = parent_id.to_string();
        }
        path.reverse();
        Ok(path)
    }

    async fn get_entries(&self) -> SessionResult<Vec<SessionTreeEntry>> {
        Ok(self.state.lock().await.entries.clone())
    }
}

pub async fn load_jsonl_session_metadata(
    file_path: impl AsRef<Path>,
) -> SessionResult<JsonlSessionMetadata> {
    let file_path = file_path.as_ref();
    let content = tokio::fs::read_to_string(file_path).await.map_err(|err| {
        io_error(
            err,
            format!("Failed to read session header {}", file_path.display()),
        )
    })?;
    let line = content
        .split('\n')
        .next()
        .filter(|line| !line.trim().is_empty())
        .ok_or_else(|| invalid_session(file_path, "missing session header"))?;
    Ok(header_to_session_metadata(
        &parse_header_line(line, file_path)?,
        file_path,
    ))
}

struct LoadedJsonlStorage {
    header: SessionHeader,
    entries: Vec<SessionTreeEntry>,
    leaf_id: Option<String>,
}

async fn load_jsonl_storage(file_path: &Path) -> SessionResult<LoadedJsonlStorage> {
    let content = tokio::fs::read_to_string(file_path).await.map_err(|err| {
        io_error(
            err,
            format!("Failed to read session {}", file_path.display()),
        )
    })?;
    let lines = content
        .split('\n')
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return Err(invalid_session(file_path, "missing session header"));
    }

    let header = parse_header_line(lines[0], file_path)?;
    let mut entries = Vec::new();
    let mut leaf_id = None;
    for (index, line) in lines.iter().enumerate().skip(1) {
        let entry = parse_entry_line(line, file_path, index + 1)?;
        leaf_id = leaf_id_after_entry(&entry);
        entries.push(entry);
    }

    Ok(LoadedJsonlStorage {
        header,
        entries,
        leaf_id,
    })
}

fn parse_header_line(line: &str, file_path: &Path) -> SessionResult<SessionHeader> {
    let value = serde_json::from_str::<serde_json::Value>(line).map_err(|err| {
        invalid_session(file_path, "first line is not a valid session header").with_cause(err)
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid_session(file_path, "first line is not a valid session header"))?;
    if object.get("type").and_then(|value| value.as_str()) != Some("session") {
        return Err(invalid_session(
            file_path,
            "first line is not a valid session header",
        ));
    }
    if object.get("version").and_then(|value| value.as_u64()) != Some(3) {
        return Err(invalid_session(file_path, "unsupported session version"));
    }
    let id = required_string(object, "id")
        .ok_or_else(|| invalid_session(file_path, "session header is missing id"))?;
    let timestamp = required_string(object, "timestamp")
        .ok_or_else(|| invalid_session(file_path, "session header is missing timestamp"))?;
    let cwd = required_string(object, "cwd")
        .ok_or_else(|| invalid_session(file_path, "session header is missing cwd"))?;
    let parent_session = match object.get("parentSession") {
        Some(value) if value.is_string() => value.as_str().map(ToString::to_string),
        Some(_) => {
            return Err(invalid_session(
                file_path,
                "session header parentSession must be a string",
            ));
        }
        None => None,
    };
    Ok(SessionHeader {
        kind: "session".to_string(),
        version: 3,
        id,
        timestamp,
        cwd,
        parent_session,
    })
}

fn parse_entry_line(
    line: &str,
    file_path: &Path,
    line_number: usize,
) -> SessionResult<SessionTreeEntry> {
    let value = serde_json::from_str::<serde_json::Value>(line).map_err(|err| {
        invalid_entry(file_path, line_number, "is not valid JSON").with_cause(err)
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid_entry(file_path, line_number, "is not a valid session entry"))?;
    if object
        .get("type")
        .and_then(|value| value.as_str())
        .is_none()
    {
        return Err(invalid_entry(
            file_path,
            line_number,
            "is missing entry type",
        ));
    }
    if required_string(object, "id").is_none() {
        return Err(invalid_entry(file_path, line_number, "is missing entry id"));
    }
    if let Some(parent_id) = object.get("parentId")
        && !parent_id.is_null()
        && !parent_id.is_string()
    {
        return Err(invalid_entry(
            file_path,
            line_number,
            "has invalid parentId",
        ));
    }
    if required_string(object, "timestamp").is_none() {
        return Err(invalid_entry(
            file_path,
            line_number,
            "is missing timestamp",
        ));
    }
    if object.get("type").and_then(|value| value.as_str()) == Some("leaf")
        && let Some(target_id) = object.get("targetId")
        && !target_id.is_null()
        && !target_id.is_string()
    {
        return Err(invalid_entry(
            file_path,
            line_number,
            "has invalid targetId",
        ));
    }

    serde_json::from_value(value).map_err(|err| {
        invalid_entry(file_path, line_number, "is not a valid session entry").with_cause(err)
    })
}

fn required_string(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<String> {
    object
        .get(key)
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn header_to_session_metadata(header: &SessionHeader, path: &Path) -> JsonlSessionMetadata {
    JsonlSessionMetadata {
        id: header.id.clone(),
        created_at: header.timestamp.clone(),
        cwd: header.cwd.clone(),
        path: path.to_string_lossy().into_owned(),
        parent_session_path: header.parent_session.clone(),
    }
}

fn update_label_cache(labels_by_id: &mut HashMap<String, String>, entry: &SessionTreeEntry) {
    let SessionTreeEntry::Label(entry) = entry else {
        return;
    };
    let label = entry.label.as_deref().map(str::trim).unwrap_or_default();
    if label.is_empty() {
        labels_by_id.remove(&entry.target_id);
    } else {
        labels_by_id.insert(entry.target_id.clone(), label.to_string());
    }
}

fn build_labels_by_id(entries: &[SessionTreeEntry]) -> HashMap<String, String> {
    let mut labels_by_id = HashMap::new();
    for entry in entries {
        update_label_cache(&mut labels_by_id, entry);
    }
    labels_by_id
}

fn leaf_id_after_entry(entry: &SessionTreeEntry) -> Option<String> {
    match entry {
        SessionTreeEntry::Leaf(entry) => entry.target_id.clone(),
        _ => Some(entry.id().to_string()),
    }
}

async fn append_line(file_path: &Path, line: &str, message: String) -> SessionResult<()> {
    use tokio::io::AsyncWriteExt;

    let mut file = tokio::fs::OpenOptions::new()
        .append(true)
        .open(file_path)
        .await
        .map_err(|err| io_error(err, message.clone()))?;
    file.write_all(line.as_bytes())
        .await
        .map_err(|err| io_error(err, message.clone()))?;
    file.write_all(b"\n")
        .await
        .map_err(|err| io_error(err, message))?;
    Ok(())
}

fn invalid_session(file_path: &Path, message: &str) -> SessionError {
    SessionError::new(
        SessionErrorCode::InvalidSession,
        format!(
            "Invalid JSONL session file {}: {message}",
            file_path.display()
        ),
    )
}

fn invalid_entry(file_path: &Path, line_number: usize, message: &str) -> SessionError {
    SessionError::new(
        SessionErrorCode::InvalidEntry,
        format!(
            "Invalid JSONL session file {}: line {line_number} {message}",
            file_path.display()
        ),
    )
}

fn io_error(error: std::io::Error, message: String) -> SessionError {
    let code = if error.kind() == std::io::ErrorKind::NotFound {
        SessionErrorCode::NotFound
    } else {
        SessionErrorCode::Storage
    };
    SessionError::new(code, format!("{message}: {error}"))
}

fn storage_error(error: serde_json::Error) -> SessionError {
    SessionError::new(SessionErrorCode::Storage, error.to_string())
}

trait WithCause {
    fn with_cause(self, cause: impl std::error::Error) -> Self;
}

impl WithCause for SessionError {
    fn with_cause(self, cause: impl std::error::Error) -> Self {
        SessionError::new(self.code, format!("{}: {cause}", self.message()))
    }
}
