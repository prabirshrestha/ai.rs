use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use ai::{CacheRetention, Message, Model, Transport, UserContent, UserMessageContent};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::compaction::{
    BranchSummaryDetails, CompactOptions, DEFAULT_COMPACTION_SETTINGS,
    GenerateBranchSummaryOptions, collect_entries_for_branch_summary, compact,
    generate_branch_summary, prepare_compaction,
};
use super::env::NodeExecutionEnv;
use super::session::Session;
use super::types::{
    BranchSummaryEntry, CustomMessageContent, MoveToSummary, PromptTemplate, SessionMetadata,
    SessionTreeEntry, Skill,
};
use crate::types::DynAgentTool;

pub type AgentHarnessResult<T> = Result<T, AgentHarnessError>;
pub type AgentHarnessAuthFn = Arc<
    dyn Fn(Model) -> Pin<Box<dyn Future<Output = Option<AgentHarnessAuth>> + Send>> + Send + Sync,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentHarnessErrorCode {
    Busy,
    InvalidState,
    InvalidArgument,
    Auth,
    Session,
    Compaction,
    BranchSummary,
    Hook,
    Unknown,
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct AgentHarnessError {
    pub code: AgentHarnessErrorCode,
    message: String,
}

impl AgentHarnessError {
    pub fn new(code: AgentHarnessErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentHarnessPhase {
    Idle,
    Turn,
    Compaction,
    BranchSummary,
    Retry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessAuth {
    pub api_key: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessResources {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_templates: Vec<PromptTemplate>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<Skill>,
}

pub struct AgentHarnessOptions<TMetadata = SessionMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    pub env: NodeExecutionEnv,
    pub session: Session<TMetadata>,
    pub tools: Vec<DynAgentTool>,
    pub resources: AgentHarnessResources,
    pub stream_options: AgentHarnessStreamOptions,
    pub get_api_key_and_headers: Option<AgentHarnessAuthFn>,
    pub model: Model,
    pub thinking_level: ai::ModelThinkingLevel,
    pub active_tool_names: Option<Vec<String>>,
}

impl<TMetadata> AgentHarnessOptions<TMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    pub fn new(env: NodeExecutionEnv, session: Session<TMetadata>, model: Model) -> Self {
        Self {
            env,
            session,
            tools: Vec::new(),
            resources: AgentHarnessResources::default(),
            stream_options: AgentHarnessStreamOptions::default(),
            get_api_key_and_headers: None,
            model,
            thinking_level: ai::ModelThinkingLevel::Off,
            active_tool_names: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NavigateTreeOptions {
    pub summarize: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
    pub replace_instructions: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NavigateTreeResult {
    pub cancelled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub editor_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_entry: Option<BranchSummaryEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessCompactionResult {
    pub summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: usize,
    pub details: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
}

impl Default for AgentHarnessStreamOptions {
    fn default() -> Self {
        Self {
            transport: None,
            timeout_ms: None,
            max_retries: None,
            max_retry_delay_ms: None,
            headers: HashMap::new(),
            metadata: None,
            cache_retention: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MapPatch<T> {
    Clear,
    Merge(HashMap<String, Option<T>>),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentHarnessStreamOptionsPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<Option<Transport>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<Option<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<Option<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<Option<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<Option<CacheRetention>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<MapPatch<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MapPatch<Value>>,
}

pub fn clone_stream_options(
    stream_options: &AgentHarnessStreamOptions,
) -> AgentHarnessStreamOptions {
    stream_options.clone()
}

pub fn merge_headers(
    sources: impl IntoIterator<Item = Option<HashMap<String, String>>>,
) -> HashMap<String, String> {
    let mut merged = HashMap::new();
    for source in sources.into_iter().flatten() {
        merged.extend(source);
    }
    merged
}

pub fn apply_stream_options_patch(
    base: &AgentHarnessStreamOptions,
    patch: Option<&AgentHarnessStreamOptionsPatch>,
) -> AgentHarnessStreamOptions {
    let mut result = clone_stream_options(base);
    let Some(patch) = patch else {
        return result;
    };

    apply_scalar_patch(&mut result.transport, &patch.transport);
    apply_scalar_patch(&mut result.timeout_ms, &patch.timeout_ms);
    apply_scalar_patch(&mut result.max_retries, &patch.max_retries);
    apply_scalar_patch(&mut result.max_retry_delay_ms, &patch.max_retry_delay_ms);
    apply_scalar_patch(&mut result.cache_retention, &patch.cache_retention);

    if let Some(headers) = patch.headers.as_ref() {
        apply_map_patch(&mut result.headers, headers);
    }
    if let Some(metadata) = patch.metadata.as_ref() {
        apply_metadata_patch(&mut result.metadata, metadata);
    }

    result
}

pub fn validate_unique_names(
    names: impl IntoIterator<Item = impl AsRef<str>>,
    message: &str,
) -> Result<(), AgentHarnessError> {
    let mut seen = std::collections::HashSet::new();
    let mut duplicates = Vec::new();
    for name in names {
        let name = name.as_ref().to_string();
        if !seen.insert(name.clone()) && !duplicates.contains(&name) {
            duplicates.push(name);
        }
    }
    if duplicates.is_empty() {
        Ok(())
    } else {
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::InvalidArgument,
            format!("{message}: {}", duplicates.join(", ")),
        ))
    }
}

pub fn create_failure_message(model: &Model, error: impl ToString, aborted: bool) -> ai::Message {
    ai::Message::Assistant(ai::AssistantMessage {
        content: vec![ai::AssistantContent::Text(ai::TextContent {
            text: String::new(),
            text_signature: None,
        })],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: ai::Usage::default(),
        stop_reason: if aborted {
            ai::StopReason::Aborted
        } else {
            ai::StopReason::Error
        },
        error_message: Some(error.to_string()),
        timestamp: ai::utils::time::now_millis(),
    })
}

pub struct AgentHarness<TMetadata = SessionMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    pub env: NodeExecutionEnv,
    session: Session<TMetadata>,
    phase: AgentHarnessPhase,
    model: Model,
    thinking_level: ai::ModelThinkingLevel,
    stream_options: AgentHarnessStreamOptions,
    get_api_key_and_headers: Option<AgentHarnessAuthFn>,
    resources: AgentHarnessResources,
    tools: HashMap<String, DynAgentTool>,
    active_tool_names: Vec<String>,
}

impl<TMetadata> AgentHarness<TMetadata>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    pub fn new(options: AgentHarnessOptions<TMetadata>) -> AgentHarnessResult<Self> {
        let tool_names = options
            .tools
            .iter()
            .map(|tool| tool.definition().name)
            .collect::<Vec<_>>();
        validate_unique_names(&tool_names, "Duplicate tool name(s)")?;
        let tools = options
            .tools
            .into_iter()
            .map(|tool| (tool.definition().name, tool))
            .collect::<HashMap<_, _>>();
        let active_tool_names = options
            .active_tool_names
            .unwrap_or_else(|| tool_names.clone());
        validate_tool_names(&tools, &active_tool_names)?;
        Ok(Self {
            env: options.env,
            session: options.session,
            phase: AgentHarnessPhase::Idle,
            model: options.model,
            thinking_level: options.thinking_level,
            stream_options: options.stream_options,
            get_api_key_and_headers: options.get_api_key_and_headers,
            resources: options.resources,
            tools,
            active_tool_names,
        })
    }

    pub fn phase(&self) -> AgentHarnessPhase {
        self.phase
    }

    pub fn session(&self) -> &Session<TMetadata> {
        &self.session
    }

    pub fn get_model(&self) -> &Model {
        &self.model
    }

    pub async fn set_model(&mut self, model: Model) -> AgentHarnessResult<()> {
        self.session
            .append_model_change(model.provider.clone(), model.id.clone())
            .await
            .map_err(|error| session_harness_error(error.message()))?;
        self.model = model;
        Ok(())
    }

    pub fn get_thinking_level(&self) -> ai::ModelThinkingLevel {
        self.thinking_level
    }

    pub async fn set_thinking_level(
        &mut self,
        level: ai::ModelThinkingLevel,
    ) -> AgentHarnessResult<()> {
        self.session
            .append_thinking_level_change(level.as_str())
            .await
            .map_err(|error| session_harness_error(error.message()))?;
        self.thinking_level = level;
        Ok(())
    }

    pub fn get_tools(&self) -> Vec<DynAgentTool> {
        self.tools.values().cloned().collect()
    }

    pub async fn set_tools(
        &mut self,
        tools: Vec<DynAgentTool>,
        active_tool_names: Option<Vec<String>>,
    ) -> AgentHarnessResult<()> {
        let tool_names = tools
            .iter()
            .map(|tool| tool.definition().name)
            .collect::<Vec<_>>();
        validate_unique_names(&tool_names, "Duplicate tool name(s)")?;
        let next_tools = tools
            .into_iter()
            .map(|tool| (tool.definition().name, tool))
            .collect::<HashMap<_, _>>();
        let next_active_tool_names =
            active_tool_names.unwrap_or_else(|| self.active_tool_names.clone());
        validate_tool_names(&next_tools, &next_active_tool_names)?;
        self.session
            .append_active_tools_change(next_active_tool_names.clone())
            .await
            .map_err(|error| session_harness_error(error.message()))?;
        self.tools = next_tools;
        self.active_tool_names = next_active_tool_names;
        Ok(())
    }

    pub fn get_active_tools(&self) -> Vec<DynAgentTool> {
        self.active_tool_names
            .iter()
            .filter_map(|name| self.tools.get(name).cloned())
            .collect()
    }

    pub async fn set_active_tools(&mut self, tool_names: Vec<String>) -> AgentHarnessResult<()> {
        validate_tool_names(&self.tools, &tool_names)?;
        self.session
            .append_active_tools_change(tool_names.clone())
            .await
            .map_err(|error| session_harness_error(error.message()))?;
        self.active_tool_names = tool_names;
        Ok(())
    }

    pub fn get_resources(&self) -> AgentHarnessResources {
        self.resources.clone()
    }

    pub fn set_resources(&mut self, resources: AgentHarnessResources) {
        self.resources = resources;
    }

    pub fn get_stream_options(&self) -> AgentHarnessStreamOptions {
        clone_stream_options(&self.stream_options)
    }

    pub fn set_stream_options(&mut self, stream_options: AgentHarnessStreamOptions) {
        self.stream_options = clone_stream_options(&stream_options);
    }

    pub async fn append_message(&mut self, message: crate::AgentMessage) -> AgentHarnessResult<()> {
        self.session
            .append_message(message)
            .await
            .map_err(|error| session_harness_error(error.message()))?;
        Ok(())
    }

    pub async fn compact(
        &mut self,
        custom_instructions: Option<String>,
    ) -> AgentHarnessResult<HarnessCompactionResult> {
        self.require_idle("compact() requires idle harness")?;
        self.phase = AgentHarnessPhase::Compaction;
        let result = self.compact_inner(custom_instructions).await;
        self.phase = AgentHarnessPhase::Idle;
        result
    }

    pub async fn navigate_tree(
        &mut self,
        target_id: &str,
        options: NavigateTreeOptions,
    ) -> AgentHarnessResult<NavigateTreeResult> {
        self.require_idle("navigateTree() requires idle harness")?;
        self.phase = AgentHarnessPhase::BranchSummary;
        let result = self.navigate_tree_inner(target_id, options).await;
        self.phase = AgentHarnessPhase::Idle;
        result
    }

    async fn compact_inner(
        &mut self,
        custom_instructions: Option<String>,
    ) -> AgentHarnessResult<HarnessCompactionResult> {
        let model = self.model.clone();
        let auth = self.get_auth(&model).await?.ok_or_else(|| {
            AgentHarnessError::new(
                AgentHarnessErrorCode::Auth,
                "No auth available for compaction",
            )
        })?;
        let branch_entries = self
            .session
            .get_branch(None)
            .await
            .map_err(|error| session_harness_error(error.message()))?;
        let preparation = prepare_compaction(&branch_entries, DEFAULT_COMPACTION_SETTINGS)
            .map_err(compaction_harness_error)?
            .ok_or_else(|| {
                AgentHarnessError::new(AgentHarnessErrorCode::Compaction, "Nothing to compact")
            })?;
        let mut options = CompactOptions::new(model, auth.api_key);
        options.headers = auth.headers;
        options.custom_instructions = custom_instructions;
        options.thinking_level = Some(self.thinking_level);
        let result = compact(preparation, options)
            .await
            .map_err(compaction_harness_error)?;
        let details = serde_json::to_value(&result.details).unwrap_or(Value::Null);
        self.session
            .append_compaction(
                result.summary.clone(),
                result.first_kept_entry_id.clone(),
                result.tokens_before.min(u32::MAX as usize) as u32,
                Some(details.clone()),
                Some(false),
            )
            .await
            .map_err(|error| session_harness_error(error.message()))?;
        Ok(HarnessCompactionResult {
            summary: result.summary,
            first_kept_entry_id: result.first_kept_entry_id,
            tokens_before: result.tokens_before,
            details,
        })
    }

    async fn navigate_tree_inner(
        &mut self,
        target_id: &str,
        options: NavigateTreeOptions,
    ) -> AgentHarnessResult<NavigateTreeResult> {
        let old_leaf_id = self
            .session
            .get_leaf_id()
            .await
            .map_err(|error| session_harness_error(error.message()))?;
        if old_leaf_id.as_deref() == Some(target_id) {
            return Ok(NavigateTreeResult {
                cancelled: false,
                editor_text: None,
                summary_entry: None,
            });
        }
        let target_entry = self
            .session
            .get_entry(target_id)
            .await
            .map_err(|error| session_harness_error(error.message()))?
            .ok_or_else(|| {
                AgentHarnessError::new(
                    AgentHarnessErrorCode::InvalidArgument,
                    format!("Entry {target_id} not found"),
                )
            })?;
        let collected =
            collect_entries_for_branch_summary(&self.session, old_leaf_id.as_deref(), target_id)
                .await
                .map_err(|error| session_harness_error(error.message()))?;

        let mut summary_text = None;
        let mut summary_details = None;
        if options.summarize && !collected.entries.is_empty() {
            let model = self.model.clone();
            let auth = self.get_auth(&model).await?.ok_or_else(|| {
                AgentHarnessError::new(
                    AgentHarnessErrorCode::Auth,
                    "No auth available for branch summary",
                )
            })?;
            let mut branch_options = GenerateBranchSummaryOptions::new(model, auth.api_key);
            branch_options.headers = auth.headers;
            branch_options.custom_instructions = options.custom_instructions.clone();
            branch_options.replace_instructions = options.replace_instructions;
            match generate_branch_summary(&collected.entries, branch_options).await {
                Ok(summary) => {
                    summary_details = serde_json::to_value(BranchSummaryDetails {
                        read_files: summary.read_files,
                        modified_files: summary.modified_files,
                    })
                    .ok();
                    summary_text = Some(summary.summary);
                }
                Err(error) if error.code == super::types::BranchSummaryErrorCode::Aborted => {
                    return Ok(NavigateTreeResult {
                        cancelled: true,
                        editor_text: None,
                        summary_entry: None,
                    });
                }
                Err(error) => return Err(branch_summary_harness_error(error)),
            }
        }

        let (new_leaf_id, editor_text) = tree_target_leaf_and_editor_text(&target_entry);
        let summary_id = self
            .session
            .move_to(
                new_leaf_id,
                summary_text.map(|summary| MoveToSummary {
                    summary,
                    details: summary_details,
                    from_hook: Some(false),
                }),
            )
            .await
            .map_err(|error| session_harness_error(error.message()))?;
        let summary_entry = if let Some(summary_id) = summary_id {
            match self
                .session
                .get_entry(&summary_id)
                .await
                .map_err(|error| session_harness_error(error.message()))?
            {
                Some(SessionTreeEntry::BranchSummary(entry)) => Some(entry),
                _ => None,
            }
        } else {
            None
        };

        Ok(NavigateTreeResult {
            cancelled: false,
            editor_text,
            summary_entry,
        })
    }

    async fn get_auth(&self, model: &Model) -> AgentHarnessResult<Option<AgentHarnessAuth>> {
        let Some(get_auth) = self.get_api_key_and_headers.as_ref() else {
            return Ok(None);
        };
        Ok(get_auth(model.clone()).await)
    }

    fn require_idle(&self, message: &str) -> AgentHarnessResult<()> {
        if self.phase == AgentHarnessPhase::Idle {
            Ok(())
        } else {
            Err(AgentHarnessError::new(AgentHarnessErrorCode::Busy, message))
        }
    }
}

fn apply_scalar_patch<T: Clone>(target: &mut Option<T>, patch: &Option<Option<T>>) {
    if let Some(value) = patch {
        *target = value.clone();
    }
}

fn validate_tool_names(
    tools: &HashMap<String, DynAgentTool>,
    tool_names: &[String],
) -> AgentHarnessResult<()> {
    validate_unique_names(tool_names, "Duplicate active tool name(s)")?;
    let missing = tool_names
        .iter()
        .filter(|name| !tools.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(AgentHarnessError::new(
            AgentHarnessErrorCode::InvalidArgument,
            format!("Unknown tool(s): {}", missing.join(", ")),
        ))
    }
}

fn tree_target_leaf_and_editor_text(entry: &SessionTreeEntry) -> (Option<String>, Option<String>) {
    match entry {
        SessionTreeEntry::Message(entry) if matches!(entry.message, Message::User(_)) => {
            (entry.parent_id.clone(), user_message_text(&entry.message))
        }
        SessionTreeEntry::CustomMessage(entry) => (
            entry.parent_id.clone(),
            Some(custom_message_content_text(&entry.content)),
        ),
        _ => (Some(entry.id().to_string()), None),
    }
}

fn user_message_text(message: &Message) -> Option<String> {
    let Message::User(message) = message else {
        return None;
    };
    Some(match &message.content {
        UserMessageContent::Text(text) => text.clone(),
        UserMessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                UserContent::Text(text) => Some(text.text.as_str()),
                UserContent::Image(_) => None,
            })
            .collect(),
    })
}

fn custom_message_content_text(content: &CustomMessageContent) -> String {
    match content {
        CustomMessageContent::Text(text) => text.clone(),
        CustomMessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                UserContent::Text(text) => Some(text.text.as_str()),
                UserContent::Image(_) => None,
            })
            .collect(),
    }
}

fn session_harness_error(message: impl Into<String>) -> AgentHarnessError {
    AgentHarnessError::new(AgentHarnessErrorCode::Session, message)
}

fn compaction_harness_error(error: super::types::CompactionError) -> AgentHarnessError {
    AgentHarnessError::new(
        AgentHarnessErrorCode::Compaction,
        error.message().to_string(),
    )
}

fn branch_summary_harness_error(error: super::types::BranchSummaryError) -> AgentHarnessError {
    AgentHarnessError::new(
        AgentHarnessErrorCode::BranchSummary,
        error.message().to_string(),
    )
}

fn apply_map_patch<T: Clone>(target: &mut HashMap<String, T>, patch: &MapPatch<T>) {
    match patch {
        MapPatch::Clear => target.clear(),
        MapPatch::Merge(values) => {
            for (key, value) in values {
                match value {
                    Some(value) => {
                        target.insert(key.clone(), value.clone());
                    }
                    None => {
                        target.remove(key);
                    }
                }
            }
        }
    }
}

fn apply_metadata_patch(target: &mut Option<Value>, patch: &MapPatch<Value>) {
    match patch {
        MapPatch::Clear => {
            *target = None;
        }
        MapPatch::Merge(values) => {
            let mut object = target
                .take()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            for (key, value) in values {
                match value {
                    Some(value) => {
                        object.insert(key.clone(), value.clone());
                    }
                    None => {
                        object.remove(key);
                    }
                }
            }
            *target = if object.is_empty() {
                None
            } else {
                Some(Value::Object(object))
            };
        }
    }
}
