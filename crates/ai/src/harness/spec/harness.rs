//! Port of `harness-v1.ts`, `harness-v1-session.ts`,
//! `harness-v1-prompt-control.ts` and `harness-v1-prompt.ts`.
//!
//! [rust] Optional TS members become a `supports_*` probe (default `false`)
//! plus a default body that returns `HarnessCapabilityUnsupportedError`.
//! `AbortSignal` becomes [`CancellationToken`].

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use tokio_util::sync::CancellationToken;

use crate::harness::errors::{HarnessError, HarnessResult};
use crate::harness::provider::JsonValue;
use crate::harness::sandbox::SandboxSession;
use crate::harness::sdk::messages::{ToolResultPart, UserModelMessage};
use crate::harness::sdk::tool::Schema;
use crate::harness::spec::bootstrap::HarnessV1Bootstrap;
use crate::harness::spec::diagnostic::HarnessV1Observability;
use crate::harness::spec::lifecycle_state::{
    HarnessV1ContinueTurnState, HarnessV1ResumeSessionState, HarnessV1TurnSettings,
};
use crate::harness::spec::message::HarnessV1Message;
use crate::harness::spec::network_sandbox::HarnessV1NetworkSandboxSession;
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::spec::tools::{
    HarnessV1BuiltinTool, HarnessV1BuiltinToolFiltering, HarnessV1PermissionMode,
};

/// `specificationVersion` of [`HarnessV1`].
pub const HARNESS_V1: &str = "harness-v1";

fn unsupported<T>(member: &str, harness_id: Option<&str>) -> HarnessResult<T> {
    Err(HarnessError::capability_unsupported(
        format!("This harness does not support {member}."),
        harness_id.map(str::to_string),
    )
    .into())
}

/// `HarnessV1Prompt`: `string | UserModelMessage`.
#[derive(Debug, Clone, PartialEq)]
pub enum HarnessV1Prompt {
    Text(String),
    Message(UserModelMessage),
}

impl From<&str> for HarnessV1Prompt {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for HarnessV1Prompt {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<UserModelMessage> for HarnessV1Prompt {
    fn from(value: UserModelMessage) -> Self {
        Self::Message(value)
    }
}

/// `sandboxSession: HarnessV1NetworkSandboxSession | SandboxSession`.
///
/// [rust] TS tells the two apart at runtime; Rust uses an enum.
#[derive(Clone)]
pub enum HarnessV1SandboxSessionRef {
    Network(Arc<dyn HarnessV1NetworkSandboxSession>),
    Basic(Arc<dyn SandboxSession>),
}

impl HarnessV1SandboxSessionRef {
    /// The file and process surface of either kind.
    pub fn session(&self) -> Arc<dyn SandboxSession> {
        match self {
            Self::Network(s) => s.clone().restricted(),
            Self::Basic(s) => s.clone(),
        }
    }

    /// The network session, when there is one.
    pub fn network(&self) -> Option<&Arc<dyn HarnessV1NetworkSandboxSession>> {
        match self {
            Self::Network(s) => Some(s),
            Self::Basic(_) => None,
        }
    }
}

impl fmt::Debug for HarnessV1SandboxSessionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Network(s) => write!(f, "Network({})", s.id()),
            Self::Basic(_) => write!(f, "Basic"),
        }
    }
}

/// `HarnessV1StartOptions`.
#[derive(Debug, Clone)]
pub struct HarnessV1StartOptions {
    /// Additional normalized HTTP headers for model requests.
    pub headers: Option<IndexMap<String, String>>,
    /// Stable identifier for this harness session.
    pub session_id: String,
    /// Resume payload from a prior lifecycle method.
    pub resume_from: Option<HarnessV1ResumeSessionState>,
    /// Continuation payload from `doSuspendTurn`, or nested in `resume_from`.
    pub continue_from: Option<HarnessV1ContinueTurnState>,
    /// Approval policy for built-in adapter-native tools.
    pub permission_mode: Option<HarnessV1PermissionMode>,
    /// Adapter-native built-ins available for this session.
    pub builtin_tool_filtering: Option<HarnessV1BuiltinToolFiltering>,
    /// Aborts startup.
    pub abort_signal: Option<CancellationToken>,
    /// Diagnostics wiring, when the consumer enabled diagnostics.
    pub observability: Option<HarnessV1Observability>,
    /// Sandbox the adapter operates against. Adapters must not stop or
    /// destroy it.
    pub sandbox_session: HarnessV1SandboxSessionRef,
    /// Absolute path the adapter runs the agent in.
    pub session_work_dir: String,
}

/// `emit` callback of a prompt or continued turn.
pub type HarnessV1Emit = Arc<dyn Fn(HarnessV1StreamPart) + Send + Sync>;

/// `HarnessV1PromptTurnOptions`: `HarnessV1TurnSettings & { … }`.
#[derive(Clone)]
pub struct HarnessV1PromptTurnOptions {
    pub settings: HarnessV1TurnSettings,
    /// Fresh input for this turn.
    pub prompt: HarnessV1Prompt,
    pub response_format: Option<HarnessV1ResponseFormat>,
    pub abort_signal: Option<CancellationToken>,
    pub emit: HarnessV1Emit,
}

/// `HarnessV1ContinueTurnOptions`: `HarnessV1TurnSettings & { … }`.
#[derive(Clone)]
pub struct HarnessV1ContinueTurnOptions {
    pub settings: HarnessV1TurnSettings,
    pub response_format: Option<HarnessV1ResponseFormat>,
    pub abort_signal: Option<CancellationToken>,
    pub emit: HarnessV1Emit,
}

impl fmt::Debug for HarnessV1PromptTurnOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessV1PromptTurnOptions")
            .field("settings", &self.settings)
            .field("prompt", &self.prompt)
            .field("response_format", &self.response_format)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for HarnessV1ContinueTurnOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessV1ContinueTurnOptions")
            .field("settings", &self.settings)
            .field("response_format", &self.response_format)
            .finish_non_exhaustive()
    }
}

/// `HarnessV1ReadHistoryResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct HarnessV1ReadHistoryResult {
    pub messages: Vec<HarnessV1Message>,
    /// Opaque position; pass back as `since`.
    pub cursor: String,
}

/// Input of `submitToolResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct HarnessV1SubmitToolResult {
    pub tool_call_id: String,
    /// Raw execution result (`unknown`).
    pub output: JsonValue,
    pub is_error: Option<bool>,
    /// Model-facing representation; wins when present.
    pub tool_result: Option<ToolResultPart>,
}

/// Input of `submitToolApproval`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessV1SubmitToolApproval {
    pub approval_id: String,
    pub approved: bool,
    pub reason: Option<String>,
}

/// `HarnessV1PromptControl`: control surface returned by `doPromptTurn`.
#[async_trait]
pub trait HarnessV1PromptControl: Send + Sync {
    /// Provide a result for a `tool-call` the adapter emitted.
    async fn submit_tool_result(&self, input: HarnessV1SubmitToolResult) -> HarnessResult<()>;

    /// Whether `submitToolApproval` is present.
    fn supports_submit_tool_approval(&self) -> bool {
        false
    }

    /// `submitToolApproval?`.
    async fn submit_tool_approval(&self, _input: HarnessV1SubmitToolApproval) -> HarnessResult<()> {
        unsupported("submitToolApproval", None)
    }

    /// Whether `submitUserMessage` is present.
    fn supports_submit_user_message(&self) -> bool {
        false
    }

    /// `submitUserMessage?`: inject a user message into the running turn.
    async fn submit_user_message(&self, _text: String) -> HarnessResult<()> {
        unsupported("submitUserMessage", None)
    }

    /// `done`: resolves when the adapter has finished the turn, rejects with
    /// the underlying error when it fails.
    ///
    /// [rust] A TS promise can be awaited many times; this returns a future
    /// that owns its state so the host can await it while still calling the
    /// submit methods.
    fn done(&self) -> BoxFuture<'static, HarnessResult<()>>;
}

/// `HarnessV1Session`: an active harness session.
#[async_trait]
pub trait HarnessV1Session: Send + Sync {
    /// Same value the host passed as `session_id`.
    fn session_id(&self) -> String;

    /// Whether this session was created from `resume_from` or `continue_from`.
    fn is_resume(&self) -> bool;

    /// Run one prompt turn.
    async fn do_prompt_turn(
        &self,
        options: HarnessV1PromptTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>>;

    /// Ask the runtime to compact its context. Adapters without manual
    /// compaction return `HarnessCapabilityUnsupportedError`.
    async fn do_compact(&self, custom_instructions: Option<String>) -> HarnessResult<()>;

    /// Whether `doReadHistory` is present.
    fn supports_read_history(&self) -> bool {
        false
    }

    /// `doReadHistory?({ since })`.
    async fn do_read_history(
        &self,
        _since: Option<String>,
    ) -> HarnessResult<HarnessV1ReadHistoryResult> {
        unsupported("doReadHistory", None)
    }

    /// Continue the in-flight turn without a new prompt.
    async fn do_continue_turn(
        &self,
        options: HarnessV1ContinueTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>>;

    /// Freeze the active turn at a precise cursor, keeping the runtime alive.
    async fn do_suspend_turn(&self) -> HarnessResult<HarnessV1ContinueTurnState>;

    /// Detach without tearing the runtime down.
    async fn do_detach(&self) -> HarnessResult<HarnessV1ResumeSessionState>;

    /// Persist resume state, then stop the runtime.
    async fn do_stop(&self) -> HarnessResult<HarnessV1ResumeSessionState>;

    /// Stop the runtime without returning lifecycle state.
    async fn do_destroy(&self) -> HarnessResult<()>;
}

/// `HarnessV1`: a versioned harness adapter.
#[async_trait]
pub trait HarnessV1: Send + Sync {
    /// Always `"harness-v1"`.
    fn specification_version(&self) -> &'static str {
        HARNESS_V1
    }

    /// Stable identifier (`"claude-code"`, `"codex"`, …).
    fn harness_id(&self) -> &str;

    /// `builtinTools`, keyed by `commonName ?? nativeName`.
    fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool>;

    /// `supportsBuiltinToolApprovals?` (absent = false).
    fn supports_builtin_tool_approvals(&self) -> bool {
        false
    }

    /// `supportsBuiltinToolFiltering?` (absent = false).
    fn supports_builtin_tool_filtering(&self) -> bool {
        false
    }

    /// `lifecycleStateSchema?`.
    fn lifecycle_state_schema(&self) -> Option<&Schema> {
        None
    }

    /// Whether `getBootstrap` is present.
    fn supports_get_bootstrap(&self) -> bool {
        false
    }

    /// `getBootstrap?({ abortSignal })`.
    async fn get_bootstrap(
        &self,
        _abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<HarnessV1Bootstrap> {
        unsupported("getBootstrap", Some(self.harness_id()))
    }

    /// Start, resume (`resume_from`) or continue (`continue_from`) a session.
    async fn do_start(
        &self,
        options: HarnessV1StartOptions,
    ) -> HarnessResult<Box<dyn HarnessV1Session>>;
}
