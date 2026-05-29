use std::collections::HashSet;
use std::sync::Arc;

use time::format_description::well_known::Rfc3339;
use uuid::Uuid;

use super::Session;
use crate::harness::types::{
    SessionError, SessionErrorCode, SessionForkOptions, SessionForkPosition, SessionMetadata,
    SessionResult, SessionStorage, SessionTreeEntry,
};

pub fn create_session_id() -> String {
    Uuid::now_v7().to_string()
}

pub fn create_timestamp() -> String {
    time::OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| ai::utils::time::now_millis().to_string())
}

pub fn create_entry_id(existing_ids: &HashSet<String>) -> String {
    for _ in 0..100 {
        let id = create_session_id().chars().take(8).collect::<String>();
        if !existing_ids.contains(&id) {
            return id;
        }
    }
    create_session_id()
}

pub fn to_session<TMetadata>(storage: Arc<dyn SessionStorage<TMetadata>>) -> Session<TMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    Session::new(storage)
}

pub async fn get_entries_to_fork<TMetadata>(
    storage: Arc<dyn SessionStorage<TMetadata>>,
    options: &SessionForkOptions,
) -> SessionResult<Vec<SessionTreeEntry>>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    let Some(entry_id) = options.entry_id.as_deref() else {
        return storage.get_entries().await;
    };
    let Some(target) = storage.get_entry(entry_id).await? else {
        return Err(SessionError::new(
            SessionErrorCode::InvalidForkTarget,
            format!("Entry {entry_id} not found"),
        ));
    };

    let effective_leaf_id = if matches!(options.position, Some(SessionForkPosition::At)) {
        Some(target.id().to_string())
    } else {
        match &target {
            SessionTreeEntry::Message(entry) if matches!(entry.message, ai::Message::User(_)) => {
                entry.parent_id.clone()
            }
            _ => {
                return Err(SessionError::new(
                    SessionErrorCode::InvalidForkTarget,
                    format!("Entry {entry_id} is not a user message"),
                ));
            }
        }
    };
    storage.get_path_to_root(effective_leaf_id).await
}

pub fn new_metadata(id: Option<String>) -> SessionMetadata {
    SessionMetadata {
        id: id.unwrap_or_else(create_session_id),
        created_at: create_timestamp(),
    }
}
