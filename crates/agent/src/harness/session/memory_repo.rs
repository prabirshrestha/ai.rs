use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::Session;
use super::memory_storage::{InMemorySessionStorage, InMemorySessionStorageOptions};
use super::repo_utils::{get_entries_to_fork, new_metadata, to_session};
use crate::harness::types::{
    SessionCreateOptions, SessionError, SessionErrorCode, SessionForkOptions, SessionMetadata,
    SessionRepo, SessionResult,
};

#[derive(Clone, Default)]
pub struct InMemorySessionRepo {
    sessions: Arc<Mutex<HashMap<String, Session<SessionMetadata>>>>,
}

impl InMemorySessionRepo {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(
        &self,
    ) -> SessionResult<std::sync::MutexGuard<'_, HashMap<String, Session<SessionMetadata>>>> {
        self.sessions.lock().map_err(|_| {
            SessionError::new(
                SessionErrorCode::Storage,
                "in-memory session repo lock poisoned",
            )
        })
    }
}

#[async_trait]
impl SessionRepo<SessionMetadata, SessionCreateOptions, ()> for InMemorySessionRepo {
    async fn create(
        &self,
        options: SessionCreateOptions,
    ) -> SessionResult<Session<SessionMetadata>> {
        let metadata = new_metadata(options.id);
        let storage = InMemorySessionStorage::with_options(InMemorySessionStorageOptions {
            entries: Vec::new(),
            metadata: metadata.clone(),
        })?;
        let session = to_session(Arc::new(storage));
        self.lock()?.insert(metadata.id.clone(), session.clone());
        Ok(session)
    }

    async fn open(&self, metadata: SessionMetadata) -> SessionResult<Session<SessionMetadata>> {
        self.lock()?.get(&metadata.id).cloned().ok_or_else(|| {
            SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session not found: {}", metadata.id),
            )
        })
    }

    async fn list(&self, _options: Option<()>) -> SessionResult<Vec<SessionMetadata>> {
        let sessions = self.lock()?.values().cloned().collect::<Vec<_>>();
        let mut metadata = Vec::with_capacity(sessions.len());
        for session in sessions {
            metadata.push(session.get_metadata().await?);
        }
        Ok(metadata)
    }

    async fn delete(&self, metadata: SessionMetadata) -> SessionResult<()> {
        self.lock()?.remove(&metadata.id);
        Ok(())
    }

    async fn fork(
        &self,
        source: SessionMetadata,
        options: SessionForkOptions,
    ) -> SessionResult<Session<SessionMetadata>> {
        let source = self.open(source).await?;
        let forked_entries = get_entries_to_fork(source.get_storage(), &options).await?;
        let metadata = new_metadata(options.id);
        let storage = InMemorySessionStorage::with_options(InMemorySessionStorageOptions {
            entries: forked_entries,
            metadata: metadata.clone(),
        })?;
        let session = to_session(Arc::new(storage));
        self.lock()?.insert(metadata.id.clone(), session.clone());
        Ok(session)
    }
}
