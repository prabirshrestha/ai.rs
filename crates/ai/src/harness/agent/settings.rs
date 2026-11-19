//! Port of `agent/harness-agent-settings.ts`.

use std::fmt;
use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use tokio_util::sync::CancellationToken;

use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::SandboxSession;
use crate::harness::sdk::tool_approval::ToolApprovalStatus;
use crate::harness::spec::harness::HarnessV1SandboxSessionRef;

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
