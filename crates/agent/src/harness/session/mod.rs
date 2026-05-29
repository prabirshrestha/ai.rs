mod jsonl_repo;
mod jsonl_storage;
mod memory_repo;
mod memory_storage;
mod repo_utils;

use std::sync::Arc;

use ai::{Message, TextContent, UserContent, UserMessage, UserMessageContent};
use serde_json::Value;

pub use jsonl_repo::{JsonlSessionRepo, encode_cwd};
pub use jsonl_storage::{JsonlSessionStorage, load_jsonl_session_metadata};
pub use memory_repo::InMemorySessionRepo;
pub use memory_storage::{InMemorySessionStorage, InMemorySessionStorageOptions};
pub use repo_utils::{create_entry_id, create_session_id, create_timestamp, get_entries_to_fork};

use crate::AgentMessage;

use super::types::{
    ActiveToolsChangeEntry, BranchSummaryEntry, CompactionEntry, CustomEntry, CustomMessageContent,
    CustomMessageEntry, LabelEntry, MessageEntry, ModelChangeEntry, MoveToSummary, SessionContext,
    SessionError, SessionErrorCode, SessionInfoEntry, SessionMetadata, SessionModel, SessionResult,
    SessionStorage, SessionTreeEntry, SessionTreeEntryType, ThinkingLevelChangeEntry,
};

const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";
const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

#[derive(Clone)]
pub struct Session<TMetadata = SessionMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    storage: Arc<dyn SessionStorage<TMetadata>>,
}

impl<TMetadata> Session<TMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    pub fn new(storage: Arc<dyn SessionStorage<TMetadata>>) -> Self {
        Self { storage }
    }

    pub async fn get_metadata(&self) -> SessionResult<TMetadata> {
        self.storage.get_metadata().await
    }

    pub fn get_storage(&self) -> Arc<dyn SessionStorage<TMetadata>> {
        self.storage.clone()
    }

    pub async fn get_leaf_id(&self) -> SessionResult<Option<String>> {
        self.storage.get_leaf_id().await
    }

    pub async fn get_entry(&self, id: &str) -> SessionResult<Option<SessionTreeEntry>> {
        self.storage.get_entry(id).await
    }

    pub async fn get_entries(&self) -> SessionResult<Vec<SessionTreeEntry>> {
        self.storage.get_entries().await
    }

    pub async fn get_branch(&self, from_id: Option<&str>) -> SessionResult<Vec<SessionTreeEntry>> {
        let leaf_id = match from_id {
            Some(id) => Some(id.to_string()),
            None => self.storage.get_leaf_id().await?,
        };
        self.storage.get_path_to_root(leaf_id).await
    }

    pub async fn build_context(&self) -> SessionResult<SessionContext> {
        Ok(build_session_context(&self.get_branch(None).await?))
    }

    pub async fn get_label(&self, id: &str) -> SessionResult<Option<String>> {
        self.storage.get_label(id).await
    }

    pub async fn get_session_name(&self) -> SessionResult<Option<String>> {
        let entries = self
            .storage
            .find_entries(SessionTreeEntryType::SessionInfo)
            .await?;
        Ok(entries.into_iter().rev().find_map(|entry| match entry {
            SessionTreeEntry::SessionInfo(entry) => entry
                .name
                .map(|name| name.trim().to_string())
                .filter(|name| !name.is_empty()),
            _ => None,
        }))
    }

    async fn append_typed_entry(&self, entry: SessionTreeEntry) -> SessionResult<String> {
        let id = entry.id().to_string();
        self.storage.append_entry(entry).await?;
        Ok(id)
    }

    pub async fn append_message(&self, message: AgentMessage) -> SessionResult<String> {
        self.append_typed_entry(SessionTreeEntry::Message(MessageEntry {
            id: self.storage.create_entry_id().await?,
            parent_id: self.storage.get_leaf_id().await?,
            timestamp: create_timestamp(),
            message,
        }))
        .await
    }

    pub async fn append_thinking_level_change(
        &self,
        thinking_level: impl Into<String>,
    ) -> SessionResult<String> {
        self.append_typed_entry(SessionTreeEntry::ThinkingLevelChange(
            ThinkingLevelChangeEntry {
                id: self.storage.create_entry_id().await?,
                parent_id: self.storage.get_leaf_id().await?,
                timestamp: create_timestamp(),
                thinking_level: thinking_level.into(),
            },
        ))
        .await
    }

    pub async fn append_model_change(
        &self,
        provider: impl Into<String>,
        model_id: impl Into<String>,
    ) -> SessionResult<String> {
        self.append_typed_entry(SessionTreeEntry::ModelChange(ModelChangeEntry {
            id: self.storage.create_entry_id().await?,
            parent_id: self.storage.get_leaf_id().await?,
            timestamp: create_timestamp(),
            provider: provider.into(),
            model_id: model_id.into(),
        }))
        .await
    }

    pub async fn append_active_tools_change(
        &self,
        active_tool_names: Vec<String>,
    ) -> SessionResult<String> {
        self.append_typed_entry(SessionTreeEntry::ActiveToolsChange(
            ActiveToolsChangeEntry {
                id: self.storage.create_entry_id().await?,
                parent_id: self.storage.get_leaf_id().await?,
                timestamp: create_timestamp(),
                active_tool_names,
            },
        ))
        .await
    }

    pub async fn append_compaction(
        &self,
        summary: impl Into<String>,
        first_kept_entry_id: impl Into<String>,
        tokens_before: u32,
        details: Option<Value>,
        from_hook: Option<bool>,
    ) -> SessionResult<String> {
        self.append_typed_entry(SessionTreeEntry::Compaction(CompactionEntry {
            id: self.storage.create_entry_id().await?,
            parent_id: self.storage.get_leaf_id().await?,
            timestamp: create_timestamp(),
            summary: summary.into(),
            first_kept_entry_id: first_kept_entry_id.into(),
            tokens_before,
            details,
            from_hook,
        }))
        .await
    }

    pub async fn append_custom_entry(
        &self,
        custom_type: impl Into<String>,
        data: Option<Value>,
    ) -> SessionResult<String> {
        self.append_typed_entry(SessionTreeEntry::Custom(CustomEntry {
            id: self.storage.create_entry_id().await?,
            parent_id: self.storage.get_leaf_id().await?,
            timestamp: create_timestamp(),
            custom_type: custom_type.into(),
            data,
        }))
        .await
    }

    pub async fn append_custom_message_entry(
        &self,
        custom_type: impl Into<String>,
        content: CustomMessageContent,
        display: bool,
        details: Option<Value>,
    ) -> SessionResult<String> {
        self.append_typed_entry(SessionTreeEntry::CustomMessage(CustomMessageEntry {
            id: self.storage.create_entry_id().await?,
            parent_id: self.storage.get_leaf_id().await?,
            timestamp: create_timestamp(),
            custom_type: custom_type.into(),
            content,
            display,
            details,
        }))
        .await
    }

    pub async fn append_label(
        &self,
        target_id: impl Into<String>,
        label: Option<String>,
    ) -> SessionResult<String> {
        let target_id = target_id.into();
        if self.storage.get_entry(&target_id).await?.is_none() {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Entry {target_id} not found"),
            ));
        }
        self.append_typed_entry(SessionTreeEntry::Label(LabelEntry {
            id: self.storage.create_entry_id().await?,
            parent_id: self.storage.get_leaf_id().await?,
            timestamp: create_timestamp(),
            target_id,
            label,
        }))
        .await
    }

    pub async fn append_session_name(&self, name: impl Into<String>) -> SessionResult<String> {
        self.append_typed_entry(SessionTreeEntry::SessionInfo(SessionInfoEntry {
            id: self.storage.create_entry_id().await?,
            parent_id: self.storage.get_leaf_id().await?,
            timestamp: create_timestamp(),
            name: Some(name.into().trim().to_string()),
        }))
        .await
    }

    pub async fn move_to(
        &self,
        entry_id: Option<String>,
        summary: Option<MoveToSummary>,
    ) -> SessionResult<Option<String>> {
        if let Some(entry_id) = entry_id.as_deref()
            && self.storage.get_entry(entry_id).await?.is_none()
        {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Entry {entry_id} not found"),
            ));
        }

        self.storage.set_leaf_id(entry_id.clone()).await?;
        let Some(summary) = summary else {
            return Ok(None);
        };

        let from_id = entry_id.clone().unwrap_or_else(|| "root".to_string());
        self.append_typed_entry(SessionTreeEntry::BranchSummary(BranchSummaryEntry {
            id: self.storage.create_entry_id().await?,
            parent_id: entry_id,
            timestamp: create_timestamp(),
            from_id,
            summary: summary.summary,
            details: summary.details,
            from_hook: summary.from_hook,
        }))
        .await
        .map(Some)
    }
}

pub fn build_session_context(path_entries: &[SessionTreeEntry]) -> SessionContext {
    let mut thinking_level = "off".to_string();
    let mut model = None;
    let mut active_tool_names = None;
    let mut compaction = None;

    for entry in path_entries {
        match entry {
            SessionTreeEntry::ThinkingLevelChange(entry) => {
                thinking_level = entry.thinking_level.clone();
            }
            SessionTreeEntry::ModelChange(entry) => {
                model = Some(SessionModel {
                    provider: entry.provider.clone(),
                    model_id: entry.model_id.clone(),
                });
            }
            SessionTreeEntry::Message(entry) => {
                if let Message::Assistant(message) = &entry.message {
                    model = Some(SessionModel {
                        provider: message.provider.clone(),
                        model_id: message.model.clone(),
                    });
                }
            }
            SessionTreeEntry::ActiveToolsChange(entry) => {
                active_tool_names = Some(entry.active_tool_names.clone());
            }
            SessionTreeEntry::Compaction(entry) => {
                compaction = Some(entry.clone());
            }
            _ => {}
        }
    }

    let mut messages = Vec::new();
    if let Some(compaction) = compaction {
        messages.push(compaction_summary_message(&compaction));

        let Some(compaction_index) = path_entries
            .iter()
            .position(|entry| matches!(entry, SessionTreeEntry::Compaction(candidate) if candidate.id == compaction.id))
        else {
            return SessionContext {
                messages,
                thinking_level,
                model,
                active_tool_names,
            };
        };

        let mut found_first_kept = false;
        for entry in path_entries.iter().take(compaction_index) {
            if entry.id() == compaction.first_kept_entry_id {
                found_first_kept = true;
            }
            if found_first_kept {
                append_context_message(&mut messages, entry);
            }
        }
        for entry in path_entries.iter().skip(compaction_index + 1) {
            append_context_message(&mut messages, entry);
        }
    } else {
        for entry in path_entries {
            append_context_message(&mut messages, entry);
        }
    }

    SessionContext {
        messages,
        thinking_level,
        model,
        active_tool_names,
    }
}

fn append_context_message(messages: &mut Vec<AgentMessage>, entry: &SessionTreeEntry) {
    match entry {
        SessionTreeEntry::Message(entry) => messages.push(entry.message.clone()),
        SessionTreeEntry::CustomMessage(entry) => messages.push(custom_message(entry)),
        SessionTreeEntry::BranchSummary(entry) if !entry.summary.is_empty() => {
            messages.push(branch_summary_message(entry));
        }
        _ => {}
    }
}

fn custom_message(entry: &CustomMessageEntry) -> AgentMessage {
    let timestamp = timestamp_to_millis(&entry.timestamp);
    let content = match &entry.content {
        CustomMessageContent::Text(text) => UserMessageContent::Text(text.clone()),
        CustomMessageContent::Parts(parts) => UserMessageContent::Parts(parts.clone()),
    };
    Message::User(UserMessage { content, timestamp })
}

fn branch_summary_message(entry: &BranchSummaryEntry) -> AgentMessage {
    summary_user_message(
        format!(
            "{BRANCH_SUMMARY_PREFIX}{}{BRANCH_SUMMARY_SUFFIX}",
            entry.summary
        ),
        &entry.timestamp,
    )
}

fn compaction_summary_message(entry: &CompactionEntry) -> AgentMessage {
    summary_user_message(
        format!(
            "{COMPACTION_SUMMARY_PREFIX}{}{COMPACTION_SUMMARY_SUFFIX}",
            entry.summary
        ),
        &entry.timestamp,
    )
}

fn summary_user_message(text: String, timestamp: &str) -> AgentMessage {
    Message::User(UserMessage {
        content: UserMessageContent::Parts(vec![UserContent::Text(TextContent {
            text,
            text_signature: None,
        })]),
        timestamp: timestamp_to_millis(timestamp),
    })
}

fn timestamp_to_millis(timestamp: &str) -> u64 {
    if let Ok(value) = timestamp.parse::<u64>() {
        return value;
    }
    let Ok(parsed) =
        time::OffsetDateTime::parse(timestamp, &time::format_description::well_known::Rfc3339)
    else {
        return 0;
    };
    parsed.unix_timestamp_nanos().max(0) as u64 / 1_000_000
}
