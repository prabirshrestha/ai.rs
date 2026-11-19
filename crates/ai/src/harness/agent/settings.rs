//! Port of `agent/harness-agent-settings.ts`.

use std::fmt;
use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use tokio_util::sync::CancellationToken;

use super::internal::resolve_observability::OnLogFn;
use super::observability::types::HarnessDebugConfig;
use crate::harness::errors::HarnessResult;
use crate::harness::provider::JsonValue;
use crate::harness::sandbox::SandboxSession;
use crate::harness::sdk::content::{Context, StepResult, ToolsContext};
use crate::harness::sdk::messages::{ModelMessage, SystemModelMessage};
use crate::harness::sdk::output::SharedOutput;
use crate::harness::sdk::stop_condition::StopCondition;
use crate::harness::sdk::telemetry::{
    Callback, GenerateTextEndEvent, GenerateTextStartEvent, GenerateTextStepStartEvent,
    LanguageModelCallEndEvent, LanguageModelCallStartEvent, TelemetryOptions,
    ToolExecutionEndEvent, ToolExecutionStartEvent,
};
use crate::harness::sdk::tool::{Schema, ToolSet};
use crate::harness::sdk::tool_approval::ToolApprovalStatus;
use crate::harness::spec::harness::{HarnessV1, HarnessV1SandboxSessionRef};
use crate::harness::spec::tools::{HarnessV1PermissionMode, HarnessV1Skill};

/// `HarnessAgentToolApprovalConfiguration`.
pub type HarnessAgentToolApprovalConfiguration = IndexMap<String, ToolApprovalStatus>;

/// `HarnessAgentSandboxCreator['createSandboxSession']`.
pub type CreateSandboxSessionFn = Arc<
    dyn Fn(
            Option<CancellationToken>,
        ) -> BoxFuture<'static, HarnessResult<HarnessV1SandboxSessionRef>>
        + Send
        + Sync,
>;

/// `HarnessAgentSandboxCreator`: creates the sandbox session for
/// `HarnessAgent::create_session` when it is needed instead of upfront.
#[derive(Clone)]
pub struct HarnessAgentSandboxCreator {
    /// Working directory of the sandbox that `create_sandbox_session`
    /// returns.
    pub default_working_directory: String,
    pub create_sandbox_session: CreateSandboxSessionFn,
}

impl fmt::Debug for HarnessAgentSandboxCreator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessAgentSandboxCreator")
            .field("default_working_directory", &self.default_working_directory)
            .finish_non_exhaustive()
    }
}

/// `onBootstrap` options.
#[derive(Clone)]
pub struct OnBootstrapOptions {
    pub session: Arc<dyn SandboxSession>,
    pub work_dir: String,
    pub abort_signal: Option<CancellationToken>,
}

/// `onSession` options.
#[derive(Clone)]
pub struct OnSessionOptions {
    pub session: Arc<dyn SandboxSession>,
    pub session_work_dir: String,
    pub abort_signal: Option<CancellationToken>,
}

/// `HarnessAgentSandboxConfig['onBootstrap']`.
pub type OnBootstrapFn =
    Arc<dyn Fn(OnBootstrapOptions) -> BoxFuture<'static, HarnessResult<()>> + Send + Sync>;

/// `HarnessAgentSandboxConfig['onSession']`.
pub type OnSessionFn =
    Arc<dyn Fn(OnSessionOptions) -> BoxFuture<'static, HarnessResult<()>> + Send + Sync>;

/// `HarnessAgentSandboxConfig`.
#[derive(Clone, Default)]
pub struct HarnessAgentSandboxConfig {
    /// Fixed working directory for all sessions, relative to the sandbox's
    /// default working directory. `"."` uses the default directory itself.
    /// When omitted, sessions use `<harnessId>-<sessionId>`.
    pub work_dir: Option<String>,
    /// Caller-controlled identity for `on_bootstrap`.
    pub bootstrap_hash: Option<String>,
    /// Called during sandbox template creation after the adapter's own
    /// bootstrap. Requires `bootstrap_hash`.
    pub on_bootstrap: Option<OnBootstrapFn>,
    /// Called after each sandbox session is acquired and the session work
    /// directory exists.
    pub on_session: Option<OnSessionFn>,
}

impl fmt::Debug for HarnessAgentSandboxConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessAgentSandboxConfig")
            .field("work_dir", &self.work_dir)
            .field("bootstrap_hash", &self.bootstrap_hash)
            .field("on_bootstrap", &self.on_bootstrap.is_some())
            .field("on_session", &self.on_session.is_some())
            .finish()
    }
}

/// `HarnessAgentSettings['instructions']`: `string | SystemModelMessage`.
#[derive(Debug, Clone, PartialEq)]
pub enum HarnessAgentInstructions {
    Text(String),
    /// Only its `content` is forwarded to the harness adapter.
    System(SystemModelMessage),
}

impl From<&str> for HarnessAgentInstructions {
    fn from(value: &str) -> Self {
        Self::Text(value.into())
    }
}

impl From<String> for HarnessAgentInstructions {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// The AI SDK call `prompt`: `string | Array<ModelMessage>`.
#[derive(Debug, Clone, PartialEq)]
pub enum HarnessAgentCallPrompt {
    Text(String),
    Messages(Vec<ModelMessage>),
}

impl From<&str> for HarnessAgentCallPrompt {
    fn from(value: &str) -> Self {
        Self::Text(value.into())
    }
}

impl From<String> for HarnessAgentCallPrompt {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<ModelMessage>> for HarnessAgentCallPrompt {
    fn from(value: Vec<ModelMessage>) -> Self {
        Self::Messages(value)
    }
}

/// What `prepareCall` receives and returns: the prompt plus the settings
/// that may vary between completed turns.
///
/// [rust] `options` is ignored on the returned value, as TS leaves it out
/// of the return type.
#[derive(Clone, Default)]
pub struct HarnessAgentPrepareCallOptions {
    /// The validated custom call options.
    pub options: Option<JsonValue>,
    pub prompt: Option<HarnessAgentCallPrompt>,
    pub messages: Option<Vec<ModelMessage>>,
    pub model: Option<String>,
    pub skills: Option<Vec<HarnessV1Skill>>,
    pub instructions: Option<HarnessAgentInstructions>,
    pub tools: Option<ToolSet>,
    pub runtime_context: Option<Context>,
    pub tools_context: ToolsContext,
}

/// `HarnessAgentSettings['prepareCall']`.
pub type PrepareCallFn = Arc<
    dyn Fn(
            HarnessAgentPrepareCallOptions,
        ) -> BoxFuture<'static, HarnessResult<HarnessAgentPrepareCallOptions>>
        + Send
        + Sync,
>;

/// `HarnessAgentSettings`: construction-time settings for a `HarnessAgent`.
///
/// Prompt, abort signal, and custom call options belong on the call
/// parameters passed to `generate` / `stream`. Lifecycle callbacks can be
/// configured here for every call, and the call parameters can add their
/// own. `prepare_call` can derive turn-scoped model, skills, instructions,
/// and tools from custom call options.
///
/// [rust] The deprecated `sandbox` provider and `onSandboxSession` are not
/// ported; pass a sandbox session to `create_session` and use
/// `sandbox_config.on_session`. `activeTools` / `inactiveTools` are tool
/// names; passing both is rejected at construction.
#[derive(Clone)]
pub struct HarnessAgentSettings {
    /// The harness adapter driving the underlying agent runtime.
    pub harness: Arc<dyn HarnessV1>,
    /// Stable identifier for this agent instance (`agent.id`).
    pub id: Option<String>,
    /// Model identifier used by the harness adapter.
    pub model: Option<String>,
    /// Tools available to the runtime in addition to the harness built-ins.
    /// User tools take precedence over built-ins on key collision.
    pub tools: Option<ToolSet>,
    /// Per-tool context passed to host-executed tools.
    pub tools_context: Option<ToolsContext>,
    /// Runtime context passed to lifecycle callbacks and telemetry.
    pub runtime_context: Option<Context>,
    /// Skills made available to the underlying runtime.
    pub skills: Option<Vec<HarnessV1Skill>>,
    /// Instructions for the underlying agent runtime.
    pub instructions: Option<HarnessAgentInstructions>,
    /// Additional HTTP headers sent with every model request.
    /// `authorization`, `x-api-key`, `user-agent`, and `x-client-app` are
    /// managed by the harness and are not allowed.
    pub headers: Option<IndexMap<String, Option<String>>>,
    /// Schema for validating the custom options passed to each call.
    pub call_options_schema: Option<Schema>,
    /// Prepares the prompt and the settings that may vary between completed
    /// turns.
    pub prepare_call: Option<PrepareCallFn>,
    /// Typed output, active for every turn.
    pub output: Option<SharedOutput>,
    /// Conditions that stop the current result after a completed harness
    /// tool step that can continue into another model step.
    pub stop_when: Option<Vec<StopCondition>>,
    pub on_start: Option<Callback<GenerateTextStartEvent>>,
    pub on_step_start: Option<Callback<GenerateTextStepStartEvent>>,
    pub on_language_model_call_start: Option<Callback<LanguageModelCallStartEvent>>,
    pub on_language_model_call_end: Option<Callback<LanguageModelCallEndEvent>>,
    pub on_tool_execution_start: Option<Callback<ToolExecutionStartEvent>>,
    pub on_tool_execution_end: Option<Callback<ToolExecutionEndEvent>>,
    pub on_step_end: Option<Callback<StepResult>>,
    pub on_end: Option<Callback<GenerateTextEndEvent>>,
    /// Built-in tool permission mode. Defaults to `allow-all`.
    pub permission_mode: Option<HarnessV1PermissionMode>,
    /// Per custom-tool approval statuses.
    pub tool_approval: Option<HarnessAgentToolApprovalConfiguration>,
    /// Sandbox working-directory and lifecycle hook configuration.
    pub sandbox_config: Option<HarnessAgentSandboxConfig>,
    /// Telemetry configuration.
    pub telemetry: Option<TelemetryOptions>,
    /// Diagnostics configuration.
    pub debug: Option<HarnessDebugConfig>,
    /// Programmatic sink for forwarded bridge diagnostics.
    pub on_log: Option<OnLogFn>,
    /// Limits the tools available for the harness to call.
    pub active_tools: Option<Vec<String>>,
    /// Excludes tools from the set available for the harness to call.
    pub inactive_tools: Option<Vec<String>>,
}

impl HarnessAgentSettings {
    /// Settings with only `harness` set.
    pub fn new(harness: Arc<dyn HarnessV1>) -> Self {
        Self {
            harness,
            id: None,
            model: None,
            tools: None,
            tools_context: None,
            runtime_context: None,
            skills: None,
            instructions: None,
            headers: None,
            call_options_schema: None,
            prepare_call: None,
            output: None,
            stop_when: None,
            on_start: None,
            on_step_start: None,
            on_language_model_call_start: None,
            on_language_model_call_end: None,
            on_tool_execution_start: None,
            on_tool_execution_end: None,
            on_step_end: None,
            on_end: None,
            permission_mode: None,
            tool_approval: None,
            sandbox_config: None,
            telemetry: None,
            debug: None,
            on_log: None,
            active_tools: None,
            inactive_tools: None,
        }
    }
}
