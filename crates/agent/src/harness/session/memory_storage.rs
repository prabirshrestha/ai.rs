use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::repo_utils::{create_entry_id, new_metadata};
use crate::harness::types::{
    LeafEntry, SessionError, SessionErrorCode, SessionMetadata, SessionResult, SessionStorage,
    SessionTreeEntry, SessionTreeEntryType,
};

#[derive(Debug, Clone)]
pub struct InMemorySessionStorageOptions<TMetadata = SessionMetadata> {
    pub entries: Vec<SessionTreeEntry>,
    pub metadata: TMetadata,
}

impl Default for InMemorySessionStorageOptions<SessionMetadata> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            metadata: new_metadata(None),
        }
    }
}

#[derive(Debug)]
struct InMemorySessionStorageState<TMetadata> {
    metadata: TMetadata,
    entries: Vec<SessionTreeEntry>,
    by_id: HashMap<String, SessionTreeEntry>,
    labels_by_id: HashMap<String, String>,
    leaf_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InMemorySessionStorage<TMetadata = SessionMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    state: Arc<Mutex<InMemorySessionStorageState<TMetadata>>>,
}

impl InMemorySessionStorage<SessionMetadata> {
    pub fn new() -> SessionResult<Self> {
        Self::with_options(InMemorySessionStorageOptions::default())
    }
}

impl<TMetadata> InMemorySessionStorage<TMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    pub fn with_options(options: InMemorySessionStorageOptions<TMetadata>) -> SessionResult<Self> {
        let mut by_id = HashMap::new();
        let mut labels_by_id = HashMap::new();
        let mut leaf_id = None;

        for entry in &options.entries {
            by_id.insert(entry.id().to_string(), entry.clone());
            update_label_cache(&mut labels_by_id, entry);
            leaf_id = leaf_id_after_entry(entry);
        }

        if let Some(leaf_id) = leaf_id.as_deref()
            && !by_id.contains_key(leaf_id)
        {
            return Err(SessionError::new(
                SessionErrorCode::InvalidSession,
                format!("Entry {leaf_id} not found"),
            ));
        }

        Ok(Self {
            state: Arc::new(Mutex::new(InMemorySessionStorageState {
                metadata: options.metadata,
                entries: options.entries,
                by_id,
                labels_by_id,
                leaf_id,
            })),
        })
    }

    fn lock(
        &self,
    ) -> SessionResult<std::sync::MutexGuard<'_, InMemorySessionStorageState<TMetadata>>> {
        self.state.lock().map_err(|_| {
            SessionError::new(
                SessionErrorCode::Storage,
                "in-memory session storage lock poisoned",
            )
        })
    }
}

#[async_trait]
impl<TMetadata> SessionStorage<TMetadata> for InMemorySessionStorage<TMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    async fn get_metadata(&self) -> SessionResult<TMetadata> {
        Ok(self.lock()?.metadata.clone())
    }

    async fn get_leaf_id(&self) -> SessionResult<Option<String>> {
        let state = self.lock()?;
        if let Some(leaf_id) = state.leaf_id.as_deref()
            && !state.by_id.contains_key(leaf_id)
        {
            return Err(SessionError::new(
                SessionErrorCode::InvalidSession,
                format!("Entry {leaf_id} not found"),
            ));
        }
        Ok(state.leaf_id.clone())
    }

    async fn set_leaf_id(&self, leaf_id: Option<String>) -> SessionResult<()> {
        let mut state = self.lock()?;
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
            parent_id: state.leaf_id.clone(),
            timestamp: super::create_timestamp(),
            target_id: leaf_id.clone(),
        });
        state.entries.push(entry.clone());
        state.by_id.insert(entry.id().to_string(), entry);
        state.leaf_id = leaf_id;
        Ok(())
    }

    async fn create_entry_id(&self) -> SessionResult<String> {
        let state = self.lock()?;
        let existing_ids = state.by_id.keys().cloned().collect::<HashSet<_>>();
        Ok(create_entry_id(&existing_ids))
    }

    async fn append_entry(&self, entry: SessionTreeEntry) -> SessionResult<()> {
        let mut state = self.lock()?;
        state.entries.push(entry.clone());
        state.by_id.insert(entry.id().to_string(), entry.clone());
        update_label_cache(&mut state.labels_by_id, &entry);
        state.leaf_id = leaf_id_after_entry(&entry);
        Ok(())
    }

    async fn get_entry(&self, id: &str) -> SessionResult<Option<SessionTreeEntry>> {
        Ok(self.lock()?.by_id.get(id).cloned())
    }

    async fn find_entries(
        &self,
        entry_type: SessionTreeEntryType,
    ) -> SessionResult<Vec<SessionTreeEntry>> {
        Ok(self
            .lock()?
            .entries
            .iter()
            .filter(|entry| entry.entry_type() == entry_type)
            .cloned()
            .collect())
    }

    async fn get_label(&self, id: &str) -> SessionResult<Option<String>> {
        Ok(self.lock()?.labels_by_id.get(id).cloned())
    }

    async fn get_path_to_root(
        &self,
        leaf_id: Option<String>,
    ) -> SessionResult<Vec<SessionTreeEntry>> {
        let Some(mut current_id) = leaf_id else {
            return Ok(Vec::new());
        };
        let state = self.lock()?;
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
        Ok(self.lock()?.entries.clone())
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

fn leaf_id_after_entry(entry: &SessionTreeEntry) -> Option<String> {
    match entry {
        SessionTreeEntry::Leaf(entry) => entry.target_id.clone(),
        _ => Some(entry.id().to_string()),
    }
}
