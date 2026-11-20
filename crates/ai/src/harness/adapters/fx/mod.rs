//! fx: an agent whose core is a WebAssembly module (libfx's `fx-core.wasm`)
//! that runs inside this process on wasmtime. Port of `@ai-sdk/harness-fx`.
//! Enabled by the `harness-fx` feature, which is off by default.
//!
//! The core talks ACP to the host. Its model requests go through this
//! process (AI Gateway, or a Codex or Grok subscription), and its tools run
//! here: the builtin file and shell tools in the sandbox's workspace, the
//! caller's tools, and the tools of remote MCP servers. Nothing is installed
//! in the sandbox. Each turn starts the core on the session's libfx
//! checkpoint, which the resume state carries.
//!
//! [rust] Not ported: the `backend` setting (the native libfx binary; only
//! the WebAssembly core runs here) and suspending a turn to continue it
//! later. The core module comes from [`FxCoreSource`]: `AI_FX_CORE_WASM`,
//! the cache, or the verified libfx npm package.

pub(crate) mod libfx;
pub(crate) mod mcp;
pub mod mcp_client;
pub(crate) mod remote_ops;
pub mod resume_state;
mod session;
pub mod session_storage;
pub(crate) mod skills;
pub mod subscription;
pub(crate) mod transport;

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock, Mutex as StdMutex};

use async_trait::async_trait;
use indexmap::IndexMap;
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

pub use libfx::authorizer::{
    FxCredentialSource, FxProviderCredentials, FxSubscriptionProvider, SubscriptionAuthorizer,
};
pub use libfx::engine::{
    FxEngine, FxEngineEvent, FxEngineOptions, FxEngineProvider, FxImage, FxModelEnd, FxModelUsage,
    FxPromptBlock, FxRuntimeEvent, FxRuntimeFactory, FxToolDescriptor, FxToolResult, FxTurn,
    FxTurnControl, FxTurnResult,
};
pub use libfx::fetch::{FxFetch, FxFetchError, FxHttpRequest, FxHttpResponse};
pub use libfx::wasm::{FxCoreSource, LIBFX_VERSION};
pub use mcp_client::{FxMcpClient, FxMcpConnector, FxMcpTransportConfig, FxMcpTransportKind};
pub use resume_state::{FxResumeStateData, FxSessionState};
pub use session::FxSession;
pub use session_storage::{FxSessionStorage, FxSessionStorageEntry};
pub use subscription::{FxAuthenticationMode, FxCodexKeyring};
pub use transport::{FxProvider, FxTransportSignal};

use crate::AgentResult;
use crate::harness::adapters::builtin_catalog::load_builtin_tool_catalog;
use crate::harness::error::{HarnessError, into_agent_error};
use crate::harness::session::{
    Harness, HarnessSession, HarnessStartOptions, validate_start_options,
};
use crate::harness::tools::BuiltinTool;
use crate::harness::utils::get_restricted_sandbox_session;
use remote_ops::FxRemoteOps;
use resume_state::parse_fx_resume_state_data;
use session::SessionConfig;
use session_storage::{FxSessionStorageWriter, read_fx_session_state};
use subscription::{ResolveFxAuthenticationOptions, resolve_fx_authentication};

/// `VERSION` of the ported `@ai-sdk/harness-fx` package.
pub const VERSION: &str = "1.0.104";

/// The harness id.
pub const FX_HARNESS_ID: &str = "fx";

/// The `api` of fx's assistant messages.
pub const FX_API: &str = "fx";

/// `FX_CLIENT_APP`: the `user-agent` and `x-client-app` of model requests.
pub const FX_CLIENT_APP: &str = concat!("ai-sdk-harness-fx/", "1.0.104");

const BUILTIN_TOOLS_JSON: &str = include_str!("builtin_tools.json");

static BUILTIN_TOOLS: LazyLock<IndexMap<String, BuiltinTool>> =
    LazyLock::new(|| load_builtin_tool_catalog(BUILTIN_TOOLS_JSON));

/// `FX_BUILTIN_TOOLS`.
pub fn fx_builtin_tools() -> IndexMap<String, BuiltinTool> {
    BUILTIN_TOOLS.clone()
}

/// `serviceTier`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxServiceTier {
    Default,
    Fast,
    Ultrafast,
}

/// `FxHarnessSettings`.
#[derive(Clone, Default)]
pub struct FxHarnessSettings {
    pub auth: Option<FxAuthenticationMode>,
    /// Remote MCP servers by name: `{ url, type?: "http" | "sse" |
    /// "streamableHttp", headers?, disabled? }`.
    pub mcp_servers: Option<Map<String, Value>>,
    pub headers: Option<IndexMap<String, String>>,
    pub effort: Option<String>,
    pub service_tier: Option<FxServiceTier>,
    /// An append-only log for the session's state. Without it, the resume
    /// state holds the whole session.
    pub session_storage: Option<Arc<dyn FxSessionStorage>>,
    /// [rust] Where the core module comes from.
    pub core: FxCoreSource,
    /// [rust] Creates the engine; defaults to the WebAssembly core.
    pub runtime_factory: Option<FxRuntimeFactory>,
    /// [rust] The HTTP client for model requests; defaults to `reqwest`.
    pub fetch: Option<FxFetch>,
    /// [rust] Connects MCP servers; defaults to the built-in HTTP client.
    pub mcp_connector: Option<FxMcpConnector>,
    /// [rust] The environment authentication reads; defaults to the
    /// process's.
    pub process_env: Option<IndexMap<String, String>>,
    /// [rust] The home directory with the saved subscription credentials and
    /// fx settings; defaults to the user's.
    pub home_directory: Option<PathBuf>,
    /// [rust] Where Codex keeps a keyring credential.
    pub codex_keyring: Option<Arc<dyn FxCodexKeyring>>,
}

/// An fx harness (`createFx`).
#[derive(Clone, Default)]
pub struct FxHarness {
    settings: Arc<FxHarnessSettings>,
}

impl FxHarness {
    pub fn new(settings: FxHarnessSettings) -> Self {
        Self {
            settings: Arc::new(settings),
        }
    }

    pub fn settings(&self) -> &FxHarnessSettings {
        &self.settings
    }

    fn authentication_options(
        &self,
        required_provider: Option<FxProvider>,
    ) -> ResolveFxAuthenticationOptions {
        ResolveFxAuthenticationOptions {
            auth: self.settings.auth.clone(),
            env: self.settings.process_env.clone(),
            home_directory: self.settings.home_directory.clone(),
            platform: None,
            required_provider,
            keyring: self.settings.codex_keyring.clone(),
        }
    }
}

fn harness_error(message: impl Into<String>) -> crate::AgentError {
    into_agent_error(HarnessError::new(message).into())
}

#[async_trait]
impl Harness for FxHarness {
    fn id(&self) -> &str {
        FX_HARNESS_ID
    }

    fn builtin_tools(&self) -> &IndexMap<String, BuiltinTool> {
        &BUILTIN_TOOLS
    }

    fn supports_builtin_tool_approvals(&self) -> bool {
        true
    }

    fn supports_builtin_tool_filtering(&self) -> bool {
        true
    }

    async fn start(&self, options: HarnessStartOptions) -> AgentResult<Box<dyn HarnessSession>> {
        validate_start_options(self, &options)?;
        if options
            .cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(harness_error("This operation was aborted"));
        }
        if let Some(resume) = &options.resume
            && resume.harness_id != FX_HARNESS_ID
        {
            return Err(harness_error(
                "Cannot import lifecycle state from a different harness.",
            ));
        }
        let fresh = match &options.resume {
            Some(_) => None,
            None => Some(
                resolve_fx_authentication(self.authentication_options(None))
                    .await
                    .map_err(into_agent_error)?,
            ),
        };
        let storage = self.settings.session_storage.clone();
        let resume_data = options
            .resume
            .as_ref()
            .map(|resume| parse_fx_resume_state_data(&resume.data))
            .transpose()
            .map_err(|error| into_agent_error(error.into()))?;
        let (restored, leaf_id) = match resume_data {
            Some(FxResumeStateData::Leaf(leaf_id)) => {
                let Some(storage) = &storage else {
                    return Err(harness_error(
                        "The fx lifecycle state was saved through a session storage, but createFx has no sessionStorage to load it from.",
                    ));
                };
                let state = read_fx_session_state(
                    storage.as_ref(),
                    &options.session_id,
                    &leaf_id,
                    options.cancellation_token.clone(),
                )
                .await
                .map_err(into_agent_error)?;
                (Some(state), Some(leaf_id))
            }
            Some(FxResumeStateData::Session(state)) => {
                if storage.is_some() {
                    return Err(harness_error(
                        "createFx has a sessionStorage, so its lifecycle state must hold a leafId.",
                    ));
                }
                (Some(state), None)
            }
            None => (None, None),
        };
        let authentication = match fresh {
            Some(authentication) => authentication,
            None => resolve_fx_authentication(
                self.authentication_options(restored.as_ref().map(|state| state.provider)),
            )
            .await
            .map_err(into_agent_error)?,
        };
        let state = restored
            .clone()
            .unwrap_or_else(|| FxSessionState::new(authentication.provider()));
        let writer = storage.map(|storage| {
            FxSessionStorageWriter::new(
                storage,
                options.session_id.clone(),
                leaf_id.clone(),
                leaf_id.as_ref().and(restored.as_ref()),
            )
        });
        let sandbox = get_restricted_sandbox_session(&options.sandbox);
        let remote = FxRemoteOps::new(sandbox.clone(), &options.work_dir);
        Ok(Box::new(FxSession {
            session_id: options.session_id.clone(),
            is_resume: options.resume.is_some(),
            config: Arc::new(SessionConfig {
                settings: self.settings.clone(),
                sandbox,
                remote,
                work_dir: options.work_dir.clone(),
                permission_mode: options.permission_mode,
                builtin_tool_filter: options.builtin_tool_filter.clone(),
                headers: options.headers.clone(),
                authentication,
                builtins: BUILTIN_TOOLS.clone(),
            }),
            state: Arc::new(StdMutex::new(state)),
            writer,
            running: Arc::default(),
            cancellation_token: options
                .cancellation_token
                .map(|token| token.child_token())
                .unwrap_or_default(),
            stopped: AtomicBool::new(false),
            factory: Arc::default(),
        }))
    }
}
