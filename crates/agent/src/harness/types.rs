use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::AgentMessage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionErrorCode {
    NotFound,
    InvalidSession,
    InvalidEntry,
    InvalidForkTarget,
    Storage,
    Unknown,
}

impl SessionErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::InvalidSession => "invalid_session",
            Self::InvalidEntry => "invalid_entry",
            Self::InvalidForkTarget => "invalid_fork_target",
            Self::Storage => "storage",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct SessionError {
    pub code: SessionErrorCode,
    message: String,
}

impl SessionError {
    pub fn new(code: SessionErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

pub type SessionResult<T> = std::result::Result<T, SessionError>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMetadata {
    pub id: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionTreeEntryType {
    Message,
    ThinkingLevelChange,
    ModelChange,
    ActiveToolsChange,
    Compaction,
    BranchSummary,
    Custom,
    CustomMessage,
    Label,
    SessionInfo,
    Leaf,
}

impl SessionTreeEntryType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::ThinkingLevelChange => "thinking_level_change",
            Self::ModelChange => "model_change",
            Self::ActiveToolsChange => "active_tools_change",
            Self::Compaction => "compaction",
            Self::BranchSummary => "branch_summary",
            Self::Custom => "custom",
            Self::CustomMessage => "custom_message",
            Self::Label => "label",
            Self::SessionInfo => "session_info",
            Self::Leaf => "leaf",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub message: AgentMessage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingLevelChangeEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub thinking_level: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelChangeEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub provider: String,
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveToolsChangeEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub active_tool_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_hook: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub from_id: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_hook: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub custom_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CustomMessageContent {
    Text(String),
    Parts(Vec<ai::UserContent>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessageEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub custom_type: String,
    pub content: CustomMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    pub display: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub target_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfoEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeafEntry {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub target_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionTreeEntry {
    Message(MessageEntry),
    ThinkingLevelChange(ThinkingLevelChangeEntry),
    ModelChange(ModelChangeEntry),
    ActiveToolsChange(ActiveToolsChangeEntry),
    Compaction(CompactionEntry),
    BranchSummary(BranchSummaryEntry),
    Custom(CustomEntry),
    CustomMessage(CustomMessageEntry),
    Label(LabelEntry),
    SessionInfo(SessionInfoEntry),
    Leaf(LeafEntry),
}

impl SessionTreeEntry {
    pub fn id(&self) -> &str {
        match self {
            Self::Message(entry) => &entry.id,
            Self::ThinkingLevelChange(entry) => &entry.id,
            Self::ModelChange(entry) => &entry.id,
            Self::ActiveToolsChange(entry) => &entry.id,
            Self::Compaction(entry) => &entry.id,
            Self::BranchSummary(entry) => &entry.id,
            Self::Custom(entry) => &entry.id,
            Self::CustomMessage(entry) => &entry.id,
            Self::Label(entry) => &entry.id,
            Self::SessionInfo(entry) => &entry.id,
            Self::Leaf(entry) => &entry.id,
        }
    }

    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Message(entry) => entry.parent_id.as_deref(),
            Self::ThinkingLevelChange(entry) => entry.parent_id.as_deref(),
            Self::ModelChange(entry) => entry.parent_id.as_deref(),
            Self::ActiveToolsChange(entry) => entry.parent_id.as_deref(),
            Self::Compaction(entry) => entry.parent_id.as_deref(),
            Self::BranchSummary(entry) => entry.parent_id.as_deref(),
            Self::Custom(entry) => entry.parent_id.as_deref(),
            Self::CustomMessage(entry) => entry.parent_id.as_deref(),
            Self::Label(entry) => entry.parent_id.as_deref(),
            Self::SessionInfo(entry) => entry.parent_id.as_deref(),
            Self::Leaf(entry) => entry.parent_id.as_deref(),
        }
    }

    pub fn timestamp(&self) -> &str {
        match self {
            Self::Message(entry) => &entry.timestamp,
            Self::ThinkingLevelChange(entry) => &entry.timestamp,
            Self::ModelChange(entry) => &entry.timestamp,
            Self::ActiveToolsChange(entry) => &entry.timestamp,
            Self::Compaction(entry) => &entry.timestamp,
            Self::BranchSummary(entry) => &entry.timestamp,
            Self::Custom(entry) => &entry.timestamp,
            Self::CustomMessage(entry) => &entry.timestamp,
            Self::Label(entry) => &entry.timestamp,
            Self::SessionInfo(entry) => &entry.timestamp,
            Self::Leaf(entry) => &entry.timestamp,
        }
    }

    pub fn entry_type(&self) -> SessionTreeEntryType {
        match self {
            Self::Message(_) => SessionTreeEntryType::Message,
            Self::ThinkingLevelChange(_) => SessionTreeEntryType::ThinkingLevelChange,
            Self::ModelChange(_) => SessionTreeEntryType::ModelChange,
            Self::ActiveToolsChange(_) => SessionTreeEntryType::ActiveToolsChange,
            Self::Compaction(_) => SessionTreeEntryType::Compaction,
            Self::BranchSummary(_) => SessionTreeEntryType::BranchSummary,
            Self::Custom(_) => SessionTreeEntryType::Custom,
            Self::CustomMessage(_) => SessionTreeEntryType::CustomMessage,
            Self::Label(_) => SessionTreeEntryType::Label,
            Self::SessionInfo(_) => SessionTreeEntryType::SessionInfo,
            Self::Leaf(_) => SessionTreeEntryType::Leaf,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModel {
    pub provider: String,
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionContext {
    pub messages: Vec<AgentMessage>,
    pub thinking_level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<SessionModel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_tool_names: Option<Vec<String>>,
}

#[async_trait]
pub trait SessionStorage<TMetadata = SessionMetadata>: Send + Sync
where
    TMetadata: Clone + Send + Sync + 'static,
{
    async fn get_metadata(&self) -> SessionResult<TMetadata>;
    async fn get_leaf_id(&self) -> SessionResult<Option<String>>;
    async fn set_leaf_id(&self, leaf_id: Option<String>) -> SessionResult<()>;
    async fn create_entry_id(&self) -> SessionResult<String>;
    async fn append_entry(&self, entry: SessionTreeEntry) -> SessionResult<()>;
    async fn get_entry(&self, id: &str) -> SessionResult<Option<SessionTreeEntry>>;
    async fn find_entries(
        &self,
        entry_type: SessionTreeEntryType,
    ) -> SessionResult<Vec<SessionTreeEntry>>;
    async fn get_label(&self, id: &str) -> SessionResult<Option<String>>;
    async fn get_path_to_root(
        &self,
        leaf_id: Option<String>,
    ) -> SessionResult<Vec<SessionTreeEntry>>;
    async fn get_entries(&self) -> SessionResult<Vec<SessionTreeEntry>>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCreateOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionForkPosition {
    Before,
    At,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionForkOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<SessionForkPosition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveToSummary {
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_hook: Option<bool>,
}

#[async_trait]
pub trait SessionRepo<
    TMetadata = SessionMetadata,
    TCreateOptions = SessionCreateOptions,
    TListOptions = (),
>: Send + Sync where
    TMetadata: Clone + Send + Sync + 'static,
    TCreateOptions: Send + Sync + 'static,
    TListOptions: Send + Sync + 'static,
{
    async fn create(
        &self,
        options: TCreateOptions,
    ) -> SessionResult<super::session::Session<TMetadata>>;
    async fn open(&self, metadata: TMetadata) -> SessionResult<super::session::Session<TMetadata>>;
    async fn list(&self, options: Option<TListOptions>) -> SessionResult<Vec<TMetadata>>;
    async fn delete(&self, metadata: TMetadata) -> SessionResult<()>;
    async fn fork(
        &self,
        source: TMetadata,
        options: SessionForkOptions,
    ) -> SessionResult<super::session::Session<TMetadata>>;
}
