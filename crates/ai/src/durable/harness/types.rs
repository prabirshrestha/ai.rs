//! Port of durable `src/harness/types.ts`: the public Harness types.
//!
//! Divergences from Pi:
//! - Extensions are not generic over an application tool type; `Extension`,
//!   `Agent` and the registry hold [`ToolRegistration`]s.
//! - Callbacks (`execute`, `render`, wrappers, hooks, `env`, `init`) are `Arc`
//!   closures returning boxed futures and `Result`s; a TS throw is an `Err`.
//! - Hook interfaces are structs of optional handlers (TS `Partial<H>`).
//! - `AgentChange` fields are `Option<Option<T>>`: `None` keeps the stored
//!   value (TS `undefined`), `Some(None)` clears it (TS `null`).
//! - `HarnessOptions.settings` is a closure read at every resolution (TS getters).
//! - `DocumentReader` is a concrete handle with typed `snapshot`/`snapshot_as_of`.

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::chord::{Context, JsonValue};
use crate::models::Models;
use crate::types::{
    AssistantMessage, CacheRetention, Message, ModelThinkingLevel, Tool, ToolCall, Transport,
    Usage, UserContent, UserMessageContent,
};

use crate::durable::documents::{AnyDocDefinition, DocAccess, DocArgs, RewindableDocAccess};
use crate::durable::env::ExecutionEnv;
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, EntryId, SubmissionId, TaskId};
use crate::durable::session::Transaction;
use crate::durable::tasks::AnyTask;
use crate::durable::types::{
    ConversationOwnership, ConversationRecord, EntryDraft, EntryRecord, JsonObject,
    SubmissionRecord, TaskRecord,
};

use super::registry::RegistryReader;
use super::scheduler::HookApi;
use super::tool::ToolExecutionApi;

/// Provider and model ID resolved through pi-ai `Models`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef {
    pub provider: String,
    pub model_id: String,
}

impl ModelRef {
    pub fn new(provider: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model_id: model_id.into(),
        }
    }
}

/// `UserMessage["content"]`.
pub type UserInput = UserMessageContent;

/// `whenBusy` of an input submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WhenBusy {
    Steer,
    FollowUp,
    Reject,
}

/// Host submission: user input that may start a run, or a passive entry write.
#[derive(Debug, Clone, PartialEq)]
pub enum SubmissionDraft {
    Input {
        request_id: Option<String>,
        content: UserInput,
        when_busy: Option<WhenBusy>,
    },
    Write {
        request_id: Option<String>,
        entry: EntryDraft,
    },
}

impl SubmissionDraft {
    /// `{ type: "input", content }`.
    pub fn input(content: impl Into<UserInput>) -> Self {
        Self::Input {
            request_id: None,
            content: content.into(),
            when_busy: None,
        }
    }

    /// `{ type: "write", entry }`.
    pub fn write(entry: EntryDraft) -> Self {
        Self::Write {
            request_id: None,
            entry,
        }
    }

    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        match &mut self {
            Self::Input { request_id, .. } | Self::Write { request_id, .. } => {
                *request_id = Some(id.into())
            }
        }
        self
    }

    pub fn when_busy(mut self, mode: WhenBusy) -> Self {
        if let Self::Input { when_busy, .. } = &mut self {
            *when_busy = Some(mode);
        }
        self
    }

    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Input { request_id, .. } | Self::Write { request_id, .. } => {
                request_id.as_deref()
            }
        }
    }
}

/// A settled submission record (`status` is `done` or `unanswered`).
pub type SettledSubmissionRecord = SubmissionRecord;

/// A terminal task receipt (`SettledTask<R>`).
pub type SettledTask = TaskRecord;

/// Options of `Conversation.abort()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConversationAbortOptions {
    /// Cross background boundaries: mark every live task reached ignoring the background flag, withdraw the queued
    /// inputs of every conversation reached, and wait until those tasks are terminal and the conversation is idle.
    pub background: bool,
}

/// Post-tools controls requested by a tool result.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolControl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_tools: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminate: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff: Option<String>,
}

/// `ToolDiagnostic["severity"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticSeverity {
    Info,
    Warn,
    Error,
}

impl DiagnosticSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

/// Remark about a call for the model and the UI, such as truncation or a spill path; never part of the tool's data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDiagnostic {
    pub severity: DiagnosticSeverity,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// What a tool execution returns (`ToolExecutionResult`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolExecutionResult {
    /// Omitted: the retained `output()` text becomes the content.
    pub content: Option<Vec<UserContent>>,
    pub is_error: Option<bool>,
    /// Omitted: the last `details()` value becomes the details.
    pub details: Option<JsonValue>,
    /// Added after those recorded through `api.diagnostic()`.
    pub diagnostics: Option<Vec<ToolDiagnostic>>,
    /// Spend of the execution itself; stored on the result and in `pi.usage.tools`.
    pub usage: Option<Usage>,
    pub control: Option<ToolControl>,
}

impl ToolExecutionResult {
    /// A result with text content.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: Some(vec![UserContent::text(text)]),
            ..Self::default()
        }
    }
}

/// Whether the tools of one round run at once or one after another in call order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolExecutionMode {
    #[default]
    Parallel,
    Sequential,
}

/// How many queued items of one mode a boundary places: the first, or all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum QueueMode {
    #[serde(rename = "all")]
    All,
    #[default]
    #[serde(rename = "one-at-a-time")]
    OneAtATime,
}

/// Whether an interrupted execution may rerun on recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Replay {
    Safe,
    #[default]
    Unsafe,
}

/// Which end of a tool's output its limits retain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Retain {
    #[default]
    Head,
    Tail,
}

/// `ToolRegistration["outputLimits"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToolOutputLimits {
    pub max_bytes: Option<usize>,
    pub max_lines: Option<usize>,
    pub retain: Option<Retain>,
}

/// `execute(args, api, context)`.
pub type ToolExecute = Arc<
    dyn Fn(JsonValue, ToolExecutionApi, Context) -> BoxFuture<'static, Result<ToolExecutionResult>>
        + Send
        + Sync,
>;

/// `prepareArguments(args)`: repair arguments before validation; must be pure.
pub type PrepareArguments = Arc<dyn Fn(JsonValue) -> Result<JsonValue> + Send + Sync>;

/// Executable tool registered in a registry. Only pi-ai `Tool` fields enter the transcript.
#[derive(Clone)]
pub struct ToolRegistration {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments (TS TypeBox schema).
    pub parameters: JsonValue,
    /// Whether an interrupted execution may rerun on recovery. Default `unsafe`.
    pub replay: Option<Replay>,
    /// Default: the settings' `toolExecution`. One sequential call makes its whole round sequential.
    pub execution_mode: Option<ToolExecutionMode>,
    pub prepare_arguments: Option<PrepareArguments>,
    pub output_limits: Option<ToolOutputLimits>,
    pub execute: ToolExecute,
}

impl ToolRegistration {
    /// The pi-ai `Tool` declaration of this registration.
    pub fn tool(&self) -> Tool {
        Tool {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            constrained_sampling: None,
        }
    }
}

impl fmt::Debug for ToolRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolRegistration")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// Committed document reads (`DocumentReader = Pick<Session, "snapshot" | "snapshotAsOf">`).
#[derive(Clone)]
pub struct DocumentReader(pub(crate) Arc<dyn ErasedReader>);

/// Erased committed reads behind a [`DocumentReader`].
pub(crate) trait ErasedReader: Send + Sync {
    fn snapshot_json(
        &self,
        definition: Arc<AnyDocDefinition>,
        args: DocArgs,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<Arc<JsonValue>>>>;
    fn snapshot_as_of_json(
        &self,
        definition: Arc<AnyDocDefinition>,
        args: DocArgs,
        at: EntryId,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<JsonValue>>>;
}

impl DocumentReader {
    /// The committed value of a document.
    pub async fn snapshot<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> Result<Option<A::Value>> {
        let definition = token.definition().clone();
        let kind = definition.kind.clone();
        let value = self
            .0
            .snapshot_json(definition, token.address_args(address), context)
            .await?;
        value
            .map(|value| crate::durable::documents::from_json(&kind, &value))
            .transpose()
    }

    /// The value of a rewindable conversation document as of `at`.
    pub async fn snapshot_as_of<A: RewindableDocAccess>(
        &self,
        token: &A,
        address: A::Address,
        at: EntryId,
        context: &Context,
    ) -> Result<Option<A::Value>> {
        let definition = token.definition().clone();
        let kind = definition.kind.clone();
        let value = self
            .0
            .snapshot_as_of_json(definition, token.address_args(address), at, context)
            .await?;
        value
            .map(|value| crate::durable::documents::from_json(&kind, &value))
            .transpose()
    }
}

impl fmt::Debug for DocumentReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DocumentReader")
    }
}

/// Input to system prompt section rendering for one request preparation.
#[derive(Clone)]
pub struct PromptInput {
    pub conversation_id: ConversationId,
    /// The request's resolution; `agent.tools` are the tools offered in this request.
    pub agent: Agent,
    /// Built by `HarnessOptions.env` for this preparation; `None` without an environment.
    pub env: Option<Arc<dyn ExecutionEnv>>,
    /// Sections already in effect after replaying the active transcript.
    pub shown: IndexMap<String, String>,
    /// Committed document reads.
    pub read: DocumentReader,
}

/// `render(input, context)`; `None` omits the section.
pub type SectionRender =
    Arc<dyn Fn(PromptInput, Context) -> BoxFuture<'static, Result<Option<String>>> + Send + Sync>;

/// One system prompt section; the agent's sections render in order before each request.
#[derive(Clone)]
pub struct PromptSection {
    pub key: String,
    pub render: SectionRender,
    /// Default true: wrap the text as `<key>\n...\n</key>`.
    pub tag: Option<bool>,
}

impl fmt::Debug for PromptSection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptSection")
            .field("key", &self.key)
            .field("tag", &self.tag)
            .finish_non_exhaustive()
    }
}

/// Built by `hook()`; matches tasks by name. `handlers` is the task's hooks struct.
#[derive(Clone)]
pub struct HookRegistration {
    pub task: String,
    pub handlers: Arc<dyn std::any::Any + Send + Sync>,
}

impl fmt::Debug for HookRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HookRegistration")
            .field("task", &self.task)
            .finish_non_exhaustive()
    }
}

/// A pure tool wrapper; an `Err` (TS throw) drops the tool.
pub type ToolWrapper = Arc<dyn Fn(ToolRegistration) -> Result<ToolRegistration> + Send + Sync>;
/// A pure section wrapper; an `Err` (TS throw) drops the section.
pub type SectionWrapper = Arc<dyn Fn(PromptSection) -> Result<PromptSection> + Send + Sync>;

/// Built by `wrapTool()` and `wrapSection()`; targets a tool name or a section key.
#[derive(Clone)]
pub enum Wrap {
    Tool {
        tool: String,
        wrap: ToolWrapper,
    },
    Section {
        section: String,
        wrap: SectionWrapper,
    },
}

impl fmt::Debug for Wrap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tool { tool, .. } => f.debug_struct("Wrap::Tool").field("tool", tool).finish(),
            Self::Section { section, .. } => f
                .debug_struct("Wrap::Section")
                .field("section", section)
                .finish(),
        }
    }
}

/// The fields of a named bundle of code (`Extension`).
#[derive(Clone, Default, Debug)]
pub struct ExtensionDefinition {
    pub name: String,
    pub tools: Vec<ToolRegistration>,
    pub sections: Vec<PromptSection>,
    pub hooks: Vec<HookRegistration>,
    /// Apply where this extension is selected, in order.
    pub wraps: Vec<Wrap>,
    /// Resolved by name for every task, whichever conversations select this extension.
    pub tasks: Vec<AnyTask>,
}

impl ExtensionDefinition {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }
}

/// Named bundle of code; installed in a registry and selected by conversations by name. Compares by identity.
#[derive(Clone)]
pub struct Extension(pub(crate) Arc<ExtensionDefinition>);

impl Extension {
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Deref for Extension {
    type Target = ExtensionDefinition;

    fn deref(&self) -> &ExtensionDefinition {
        &self.0
    }
}

impl PartialEq for Extension {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
    }
}

impl fmt::Debug for Extension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Extension").field(&self.0.name).finish()
    }
}

/// `AgentState["extensions"]`: exactly these extensions, or an edit of the host default selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtensionSelection {
    List(Vec<String>),
    Edit {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        add: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        remove: Option<Vec<String>>,
    },
}

/// `AgentState["tools"]`: exactly these tools, or every tool but these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolFilter {
    List(Vec<String>),
    Remove { remove: Vec<String> },
}

/// Stored choices of one conversation; names, not objects. Unset fields follow the host.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ModelThinkingLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<ExtensionSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolFilter>,
    /// Rendered after every extension section, as the section `instructions`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Directory within the environment's file system, passed to `HarnessOptions.env`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

/// `AgentChange["extensions"]`.
#[derive(Debug, Clone)]
pub enum ExtensionsChange {
    List(Vec<Extension>),
    Edit {
        add: Option<Vec<Extension>>,
        remove: Option<Vec<Extension>>,
    },
}

/// `AgentChange["tools"]`.
#[derive(Debug, Clone)]
pub enum ToolsChange {
    List(Vec<ToolRegistration>),
    Remove(Vec<ToolRegistration>),
}

/// A change to `pi.agent`: `Some(Some(v))` replaces the stored field, `Some(None)` clears it, `None` changes nothing.
#[derive(Debug, Clone, Default)]
pub struct AgentChange {
    pub model: Option<Option<ModelRef>>,
    pub thinking_level: Option<Option<ModelThinkingLevel>>,
    pub extensions: Option<Option<ExtensionsChange>>,
    pub tools: Option<Option<ToolsChange>>,
    pub instructions: Option<Option<String>>,
    pub cwd: Option<Option<String>>,
}

impl AgentChange {
    pub fn model(mut self, model: ModelRef) -> Self {
        self.model = Some(Some(model));
        self
    }

    pub fn thinking_level(mut self, level: ModelThinkingLevel) -> Self {
        self.thinking_level = Some(Some(level));
        self
    }

    pub fn extensions(mut self, extensions: Vec<Extension>) -> Self {
        self.extensions = Some(Some(ExtensionsChange::List(extensions)));
        self
    }

    pub fn tools(mut self, tools: Vec<ToolRegistration>) -> Self {
        self.tools = Some(Some(ToolsChange::List(tools)));
        self
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(Some(instructions.into()));
        self
    }

    pub fn cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(Some(cwd.into()));
        self
    }
}

/// A conversation's agent resolved against a registry snapshot and the settings.
#[derive(Debug, Clone)]
pub struct Agent {
    pub model: Option<ModelRef>,
    pub thinking_level: ModelThinkingLevel,
    pub extensions: Vec<Extension>,
    /// The tools a request offers, in order.
    pub tools: Vec<ToolRegistration>,
    /// Extension sections, then `instructions` when set.
    pub sections: Vec<PromptSection>,
    pub instructions: Option<String>,
    pub cwd: Option<String>,
}

/// Runs inside the creating commit, after the creation hook and the `agent` change.
pub type ConversationInit =
    Arc<dyn Fn(Transaction, ConversationId) -> BoxFuture<'static, Result<()>> + Send + Sync>;

/// `ConversationCreateOptions`.
#[derive(Clone)]
pub struct ConversationCreateOptions {
    pub ownership: ConversationOwnership,
    /// Applied in the creating commit after the creation hook's copy, before `init`.
    pub agent: Option<AgentChange>,
    pub init: Option<ConversationInit>,
}

impl ConversationCreateOptions {
    pub fn new(ownership: ConversationOwnership) -> Self {
        Self {
            ownership,
            agent: None,
            init: None,
        }
    }

    pub fn ownerless() -> Self {
        Self::new(ConversationOwnership::Ownerless)
    }
}

impl fmt::Debug for ConversationCreateOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConversationCreateOptions")
            .field("ownership", &self.ownership)
            .finish_non_exhaustive()
    }
}

/// `ConversationStreamOptions["deferred"]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeferredOption {
    Enabled(bool),
    Window {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window: Option<String>,
    },
}

/// Curated pi-ai request options; absent fields use pi-ai defaults.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationStreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Provider/SDK retries inside one request attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredOption>,
}

/// Durable generation attempt retries; the JSON shape of pi-ai `RetryPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationRetryPolicy {
    pub enabled: bool,
    pub max_retries: u32,
    pub base_delay_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_agent_delay_ms: Option<u64>,
}

/// `Partial<ConversationRetryPolicy>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RetryPolicyOverrides {
    pub enabled: Option<bool>,
    pub max_retries: Option<u32>,
    pub base_delay_ms: Option<u64>,
    pub max_agent_delay_ms: Option<u64>,
}

/// Automatic compaction thresholds (spec §8.7); manual compaction ignores `enabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionPolicy {
    /// Threshold and overflow compaction.
    pub enabled: bool,
    /// Room kept free for the answer: generation blocks to compact above `context_window - reserve_tokens`. Signed,
    /// as TS's `number`: a negative reserve puts the blocking threshold above the window.
    pub reserve_tokens: i64,
    /// Approximate size of the recent context a summary keeps verbatim.
    pub keep_recent_tokens: u64,
    /// Background compaction starts `background_tokens` below the blocking threshold; `0` disables it.
    pub background_tokens: u64,
}

/// `Partial<CompactionPolicy>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompactionPolicyOverrides {
    pub enabled: Option<bool>,
    pub reserve_tokens: Option<i64>,
    pub keep_recent_tokens: Option<u64>,
    pub background_tokens: Option<u64>,
}

/// Why a compaction runs: `compact()`, a threshold in generation preparation, or a context overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CompactionReason {
    Manual,
    Threshold,
    Overflow,
}

/// `entryId` of a blocking compaction's summary, or the `submissionId` of a conversation-owned compaction's summary
/// write; both absent when nothing was compacted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<EntryId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submission_id: Option<SubmissionId>,
}

/// Harness-wide run policy. Read at every resolution and never copied.
#[derive(Debug, Clone, Default)]
pub struct HarnessSettings {
    /// Default extension selection; absent: every installed extension, in install order.
    pub extensions: Option<Vec<Extension>>,
    pub stream: Option<ConversationStreamOptions>,
    pub retry: Option<RetryPolicyOverrides>,
    pub compaction: Option<CompactionPolicyOverrides>,
    pub tool_execution: Option<ToolExecutionMode>,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
}

/// Resolved settings: every field over its built-in default, object fields merged.
#[derive(Debug, Clone)]
pub struct Settings {
    pub extensions: Option<Vec<Extension>>,
    pub stream: ConversationStreamOptions,
    pub retry: ConversationRetryPolicy,
    pub compaction: CompactionPolicy,
    pub tool_execution: ToolExecutionMode,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
}

/// What `HarnessOptions.env` builds an environment for.
#[derive(Clone)]
pub struct EnvTarget {
    pub conversation_id: ConversationId,
    /// The conversation's agent `cwd`.
    pub cwd: Option<String>,
    pub read: DocumentReader,
}

/// Builds a conversation's environment at each use. Never called on the Session line.
pub type EnvFactory = Arc<
    dyn Fn(EnvTarget, Context) -> BoxFuture<'static, Result<Option<Arc<dyn ExecutionEnv>>>>
        + Send
        + Sync,
>;

/// `conversationCreated(tx, conversation)`.
pub type ConversationCreated =
    Arc<dyn Fn(Transaction, ConversationRecord) -> BoxFuture<'static, Result<()>> + Send + Sync>;

/// Settings read at every resolution.
pub type SettingsSource = Arc<dyn Fn() -> HarnessSettings + Send + Sync>;

/// Receives extension failures that do not fail the calling operation. Must not panic.
pub type OnReport = Arc<dyn Fn(Error) + Send + Sync>;

/// `HarnessOptions`.
#[derive(Clone)]
pub struct HarnessOptions {
    /// pi-ai model access used by generation.
    pub models: Models,
    pub registry: Arc<dyn RegistryReader>,
    pub settings: Option<SettingsSource>,
    pub env: Option<EnvFactory>,
    pub conversation_created: Option<ConversationCreated>,
    /// The Harness clock in epoch milliseconds.
    pub now: Option<Arc<dyn Fn() -> u64 + Send + Sync>>,
    pub on_report: Option<OnReport>,
}

impl HarnessOptions {
    pub fn new(models: Models, registry: Arc<dyn RegistryReader>) -> Self {
        Self {
            models,
            registry,
            settings: None,
            env: None,
            conversation_created: None,
            now: None,
            on_report: None,
        }
    }
}

/// Why a pending task cannot be reserved under a registry snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockedReason {
    MissingTask,
    TaskTooOld,
    MigrationFailed,
}

impl BlockedReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MissingTask => "missing_task",
            Self::TaskTooOld => "task_too_old",
            Self::MigrationFailed => "migration_failed",
        }
    }
}

/// `TaskInspection["state"]`.
#[derive(Debug, Clone)]
pub enum TaskInspectionState {
    /// An invocation is active.
    Running,
    /// The next scheduling pass reserves it; `migrates` when its definition is newer and has `migrate`.
    Ready { migrates: bool },
    /// Waits for these live tasks.
    Waiting { on: Vec<TaskId> },
    /// Outcome held until its ordinary owned work drains.
    Completing,
    /// No registered definition can take it.
    Blocked {
        reason: BlockedReason,
        error: Option<Error>,
    },
}

/// Live task and what the scheduler would do with it under the current registry.
#[derive(Debug, Clone)]
pub struct TaskInspection {
    pub record: TaskRecord,
    pub state: TaskInspectionState,
}

/// `HarnessInspection["scheduling"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheduling {
    Paused,
    Running,
    Closing,
}

/// Point-in-time view of live work.
#[derive(Debug, Clone)]
pub struct HarnessInspection {
    pub scheduling: Scheduling,
    pub tasks: Vec<TaskInspection>,
    /// Queued and placed submissions, in ID order.
    pub submissions: Vec<SubmissionRecord>,
}

/// Raw active transcript and derived model context.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContextView {
    /// Newest applicable head marker, if any.
    pub head: Option<EntryRecord>,
    /// Raw active entries: the head marker followed by non-head entries from its head through the tail.
    pub entries: Vec<EntryRecord>,
    /// Per entry of `entries`, its model messages after edits and excluded stop reasons.
    pub contributions: Vec<Vec<Message>>,
    /// Model context for the next provider request.
    pub messages: Vec<Message>,
}

/// Hook result future.
pub type HookFuture<T> = BoxFuture<'static, Result<Option<T>>>;

/// `beforeRequest(request, api, context)`: replaces the messages of that request only.
pub type BeforeRequestHook =
    Arc<dyn Fn(Vec<Message>, HookApi, Context) -> HookFuture<Vec<Message>> + Send + Sync>;
/// `afterResponse(message, api, context)`.
pub type AfterResponseHook =
    Arc<dyn Fn(AssistantMessage, HookApi, Context) -> BoxFuture<'static, Result<()>> + Send + Sync>;
/// `onYield(answer, api, context)`: the first `continue` appends a user message and continues the run.
pub type OnYieldHook =
    Arc<dyn Fn(AssistantMessage, HookApi, Context) -> HookFuture<UserInput> + Send + Sync>;
/// `afterTools(assistant, results, api, context)`.
pub type AfterToolsHook = Arc<
    dyn Fn(EntryId, Vec<EntryId>, HookApi, Context) -> BoxFuture<'static, Result<()>> + Send + Sync,
>;

/// Hooks of the built-in generation task.
#[derive(Clone, Default)]
pub struct GenerationHooks {
    pub before_request: Option<BeforeRequestHook>,
    pub after_response: Option<AfterResponseHook>,
    pub on_yield: Option<OnYieldHook>,
    pub after_tools: Option<AfterToolsHook>,
}

/// `beforeTool`'s decision: the first `block` wins, otherwise `arguments` replace the call's arguments.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BeforeToolDecision {
    pub arguments: Option<JsonObject>,
    pub block: Option<String>,
}

/// `beforeTool(call, api, context)`.
pub type BeforeToolHook =
    Arc<dyn Fn(ToolCall, HookApi, Context) -> HookFuture<BeforeToolDecision> + Send + Sync>;
/// `afterTool(call, result, api, context)`: replaces the result.
pub type AfterToolHook = Arc<
    dyn Fn(ToolCall, ToolExecutionResult, HookApi, Context) -> HookFuture<ToolExecutionResult>
        + Send
        + Sync,
>;

/// Hooks of the built-in tool task.
#[derive(Clone, Default)]
pub struct ToolHooks {
    pub before_tool: Option<BeforeToolHook>,
    pub after_tool: Option<AfterToolHook>,
}

/// What `beforeCompact` sees.
#[derive(Debug, Clone)]
pub struct CompactionRequest {
    pub reason: CompactionReason,
    /// The active entries the summary replaces, the head marker first.
    pub entries: Vec<EntryRecord>,
    /// Their model context, the summarizer's source.
    pub messages: Vec<Message>,
    /// The first entry kept verbatim.
    pub first_kept: EntryId,
    pub instructions: Option<String>,
}

/// `beforeCompact`'s decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionDecision {
    Decline,
    Summary(String),
}

/// `beforeCompact(compaction, api, context)`: the first decision wins.
pub type BeforeCompactHook = Arc<
    dyn Fn(CompactionRequest, HookApi, Context) -> HookFuture<CompactionDecision> + Send + Sync,
>;

/// Hooks of the built-in compaction task.
#[derive(Clone, Default)]
pub struct CompactionHooks {
    pub before_compact: Option<BeforeCompactHook>,
}
