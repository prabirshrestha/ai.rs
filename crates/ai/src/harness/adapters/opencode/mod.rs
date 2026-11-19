//! OpenCode: `opencode serve` run in the sandbox and driven from Rust over
//! its HTTP API and event stream. Port of `@ai-sdk/harness-opencode`.
//! Enabled by the `harness-opencode` feature.
//!
//! The server runs for the whole session with one OpenCode session, and
//! restarts when its configuration changes. The caller's tools reach it as
//! the `harness-tools` MCP server, which this process answers. Permission
//! requests are approved by the permission mode or asked through
//! `before_tool_call`; questions go to the caller's `askUserQuestions` tool.
//!
//! [rust] The TS bridge process and its WebSocket channel are replaced by
//! direct HTTP calls. In restrictive permission modes the relay serves
//! `read` in place of the TS read plugin. Not ported: suspending and
//! continuing a turn, steering, raw stream parts, the host input tracker
//! and the session token totals (usage comes from each step).

pub mod auth;
mod client;
mod config;
pub mod events;
pub mod permissions;
mod read_tool;
mod server;
pub mod subscription;
mod turn;

#[cfg(test)]
mod tests;

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use async_trait::async_trait;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::builtin_catalog::load_builtin_tool_catalog;
use crate::harness::adapters::host_tools::HostTools;
use crate::harness::auth::HarnessCredentialForwarding;
use crate::harness::bootstrap::{
    HarnessBootstrap, HarnessBootstrapCommand, HarnessBootstrapFile, apply_harness_bootstrap,
};
use crate::harness::error::{BoxError, HarnessError, into_agent_error};
use crate::harness::network::{HarnessPortEndpoint, HarnessRequestTransformationSources};
use crate::harness::paths::{harness_session_data_directory_path, harness_state_directory_path};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxRunOptions};
use crate::harness::session::{
    Harness, HarnessResumeState, HarnessSession, HarnessStartOptions, HarnessTurn,
    HarnessTurnOptions, ResponseFormat, validate_start_options,
};
use crate::harness::tools::{BuiltinTool, BuiltinToolFilter};
use crate::harness::utils::write_skills::{WriteSkillsOptions, write_skills};
use crate::harness::utils::{
    apply_credential_forwarding, get_restricted_sandbox_session,
    resolve_sandbox_credential_environment, resolve_sandbox_home_dir, shell_quote,
    warn_credential_brokering_unavailable,
};
use crate::{AgentMessage, AgentResult};

pub use auth::{
    OPENCODE_CREDENTIAL_ENVIRONMENT_VARIABLES,
    OPENCODE_SUBSCRIPTION_ACCESS_TOKEN_ENVIRONMENT_VARIABLE, OpenCodeAuthenticationChoice,
    OpenCodeAuthenticationMode, OpenCodeResolvedAuthenticationMode,
};
pub use permissions::OPENCODE_WRITE_PROTECTED_PATHS;
pub use server::OpenCodeServerCoords;
pub use subscription::{
    OpenCodeSubscription, ReadOpenCodeSubscriptionFn, read_opencode_subscription,
    resolve_opencode_authentication,
};

/// The harness id.
pub const OPENCODE_HARNESS_ID: &str = "opencode";

/// `VERSION` of the ported `@ai-sdk/harness-opencode` package.
pub const VERSION: &str = "1.0.191";

/// The `api` of messages from OpenCode.
pub const OPENCODE_API: &str = "opencode";

/// The MCP server name of the caller's tools; reserved.
pub const HOST_TOOL_MCP_SERVER_NAME: &str = "harness-tools";

/// `OPENCODE_BOOTSTRAP_DIR`.
pub const OPENCODE_BOOTSTRAP_DIR: &str = ".harness-bootstrap/opencode";

/// The `opencode-ai` version the bootstrap installs.
pub const OPENCODE_VERSION: &str = "1.19.5";

const BUILTIN_TOOLS_JSON: &str = include_str!("builtin_tools.json");

static BUILTIN_TOOLS: LazyLock<IndexMap<String, BuiltinTool>> =
    LazyLock::new(|| load_builtin_tool_catalog(BUILTIN_TOOLS_JSON));

/// `OPENCODE_BUILTIN_TOOLS`.
pub fn opencode_builtin_tools() -> IndexMap<String, BuiltinTool> {
    BUILTIN_TOOLS.clone()
}

/// `OPENCODE_CLIENT_APP`: the `x-client-app` value.
pub fn opencode_client_app() -> String {
    format!("ai-sdk-harness-opencode/{VERSION}")
}

/// `OpenCodeHarnessSettings`.
#[derive(Clone, Default)]
pub struct OpenCodeHarnessSettings {
    pub auth: Option<OpenCodeAuthenticationMode>,
    /// Customizes each credential value before it is forwarded into the
    /// sandbox.
    pub credential_forwarding: Option<HarnessCredentialForwarding>,
    /// Passed to OpenCode as-is, with native key names. Values the adapter
    /// manages win.
    pub opencode_config: Option<Map<String, Value>>,
    /// MCP server definitions in OpenCode's format, keyed by name.
    pub mcp_servers: Option<Map<String, Value>>,
    pub provider: Option<String>,
    /// OpenCode's reasoning variant, such as `"high"`.
    pub reasoning_variant: Option<String>,
    /// The sandbox port the server listens on. Defaults to the sandbox's
    /// first port; required with `port_endpoint`.
    pub port: Option<u16>,
    /// Where this process reaches the server, instead of the sandbox's port
    /// endpoint.
    pub port_endpoint: Option<HarnessPortEndpoint>,
    /// Defaults to 120 seconds.
    pub startup_timeout: Option<Duration>,
    /// [rust] Run this command (shell syntax) instead of the bootstrapped
    /// `opencode`; it receives `serve --hostname=… --port=…`.
    pub executable_override: Option<String>,
    /// [rust] The host environment credentials are read from. Defaults to
    /// this process's environment.
    pub process_env: Option<IndexMap<String, String>>,
    /// [rust] Reads a stored subscription. Defaults to
    /// [`read_opencode_subscription`].
    pub read_subscription: Option<ReadOpenCodeSubscriptionFn>,
}

/// The adapter's resume state (`openCodeResumeStateSchema`, with the
/// server in place of the bridge).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenCodeResumeState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_code_session_id: Option<String>,
    /// The server a detached session left running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server: Option<OpenCodeServerCoords>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_credential_environment: Option<IndexMap<String, String>>,
}

/// An OpenCode harness.
#[derive(Clone)]
pub struct OpenCodeHarness {
    settings: Arc<OpenCodeHarnessSettings>,
}

impl OpenCodeHarness {
    /// `createOpenCode`.
    pub fn new(settings: OpenCodeHarnessSettings) -> Result<Self, BoxError> {
        if settings
            .mcp_servers
            .as_ref()
            .is_some_and(|servers| servers.contains_key(HOST_TOOL_MCP_SERVER_NAME))
        {
            return Err(HarnessError::new(
                "OpenCode MCP server name \"harness-tools\" is reserved for HarnessAgent tools.",
            )
            .into());
        }
        Ok(Self {
            settings: Arc::new(settings),
        })
    }

    pub fn settings(&self) -> &OpenCodeHarnessSettings {
        &self.settings
    }
}

/// `getOpenCodeBootstrap`: installs `opencode-ai` and ripgrep with npm.
pub fn opencode_bootstrap() -> HarnessBootstrap {
    let manifest = json!({
        "name": "harness-opencode",
        "version": "0.0.0",
        "private": true,
        "dependencies": {
            "@vscode/ripgrep": "1.18.0",
            "opencode-ai": OPENCODE_VERSION,
        },
    });
    HarnessBootstrap {
        harness_id: OPENCODE_HARNESS_ID.to_string(),
        bootstrap_dir: OPENCODE_BOOTSTRAP_DIR.to_string(),
        files: vec![HarnessBootstrapFile {
            path: format!("{OPENCODE_BOOTSTRAP_DIR}/package.json"),
            content: serde_json::to_string_pretty(&manifest).unwrap() + "\n",
        }],
        commands: vec![
            HarnessBootstrapCommand {
                command: "npm install --no-audit --no-fund --loglevel=error".into(),
            },
            HarnessBootstrapCommand {
                command: "./node_modules/.bin/opencode --version".into(),
            },
        ],
    }
}

fn harness_error(message: impl Into<String>) -> crate::AgentError {
    HarnessError::new(message).into()
}

fn unsupported(message: impl Into<String>) -> crate::AgentError {
    HarnessError::capability_unsupported(message, Some(OPENCODE_HARNESS_ID.into())).into()
}

/// `validateBuiltinToolFiltering`.
fn validate_builtin_tool_filtering(filter: Option<&BuiltinToolFilter>) -> AgentResult<()> {
    if filter.is_some()
        && BuiltinToolFilter::includes(filter, "write")
            != BuiltinToolFilter::includes(filter, "edit")
    {
        return Err(unsupported(
            "OpenCode built-in filtering requires 'write' and 'edit' to either both be active or both be inactive.",
        ));
    }
    Ok(())
}

/// `resolveInactiveBuiltinToolNames`.
fn inactive_builtin_tool_names(filter: Option<&BuiltinToolFilter>) -> Vec<String> {
    if filter.is_none() {
        return Vec::new();
    }
    BUILTIN_TOOLS
        .keys()
        .filter(|name| !BuiltinToolFilter::includes(filter, name))
        .cloned()
        .collect()
}

#[async_trait]
impl Harness for OpenCodeHarness {
    fn id(&self) -> &str {
        OPENCODE_HARNESS_ID
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

    async fn bootstrap(&self) -> AgentResult<Option<HarnessBootstrap>> {
        Ok(Some(opencode_bootstrap()))
    }

    async fn start(&self, options: HarnessStartOptions) -> AgentResult<Box<dyn HarnessSession>> {
        validate_start_options(self, &options)?;
        validate_builtin_tool_filtering(options.builtin_tool_filter.as_ref())?;
        let settings = self.settings.clone();
        if settings.port_endpoint.is_some() && settings.port.is_none() {
            return Err(harness_error(
                "OpenCode requires `port` together with `port_endpoint`.",
            ));
        }
        let resume: OpenCodeResumeState = match &options.resume {
            Some(state) => serde_json::from_value(state.data.clone()).map_err(|error| {
                harness_error(format!("Invalid OpenCode resume state: {error}"))
            })?,
            None => OpenCodeResumeState::default(),
        };
        let sandbox = get_restricted_sandbox_session(&options.sandbox);
        let abort = options.cancellation_token.clone();
        let process_env = settings
            .process_env
            .clone()
            .unwrap_or_else(|| std::env::vars().collect());
        let authentication =
            resolve_opencode_authentication(subscription::ResolveOpenCodeAuthenticationOptions {
                auth: settings.auth.clone(),
                model: None,
                provider: settings.provider.clone(),
                process_env: Some(process_env.clone()),
                read_subscription: settings.read_subscription.clone(),
            })
            .await
            .map_err(into_agent_error)?;
        let credential_names: Vec<String> = OPENCODE_CREDENTIAL_ENVIRONMENT_VARIABLES
            .iter()
            .map(|name| name.to_string())
            .collect();
        let resolved_env = authentication.environment.clone();
        let mut opencode_config = settings.opencode_config.clone();
        let mut gitlab_brokered = false;
        let mut sandbox_credential_environment = None;
        let network = options
            .sandbox
            .network()
            .filter(|network| network.supports_add_request_transformations());
        let sandbox_auth_env = match network {
            Some(network) => {
                let credentials = resolve_sandbox_credential_environment(
                    &resolved_env,
                    &credential_names,
                    settings.credential_forwarding.as_ref(),
                    resume.sandbox_credential_environment.as_ref(),
                )
                .await
                .map_err(into_agent_error)?;
                let mut sandbox_env = resolved_env.clone();
                sandbox_env.extend(credentials.clone());
                let sandbox_token = sandbox_env
                    .get(OPENCODE_SUBSCRIPTION_ACCESS_TOKEN_ENVIRONMENT_VARIABLE)
                    .cloned();
                let transformations = match (&authentication.subscription, &sandbox_token) {
                    (Some(subscription), Some(token))
                        if subscription.provider_id
                            == OpenCodeResolvedAuthenticationMode::Gitlab =>
                    {
                        let access = subscription::request_opencode_gitlab_direct_access(
                            subscription::RequestOpenCodeGitLabDirectAccessOptions {
                                access_token: subscription.access_token.clone(),
                                instance_url: subscription.enterprise_url.clone(),
                                ai_gateway_url: process_env.get("GITLAB_AI_GATEWAY_URL").cloned(),
                                fetch: None,
                            },
                        )
                        .await
                        .map_err(into_agent_error)?;
                        opencode_config =
                            Some(subscription::create_opencode_gitlab_subscription_config(
                                settings.opencode_config.as_ref(),
                                token,
                                &access.ai_gateway_url,
                                options.headers.as_ref(),
                            ));
                        gitlab_brokered = true;
                        subscription::create_opencode_gitlab_subscription_request_transformations(
                            &access, token,
                        )
                    }
                    (Some(subscription), Some(token)) => {
                        subscription::create_opencode_subscription_request_transformations(
                            subscription,
                            token,
                        )
                    }
                    _ => auth::create_opencode_request_transformations(
                        &HarnessRequestTransformationSources {
                            env: resolved_env.clone(),
                            sandbox_env: sandbox_env.clone(),
                            auth: authentication.authentication_mode,
                        },
                    ),
                }
                .map_err(into_agent_error)?;
                if !transformations.is_empty() {
                    network
                        .add_request_transformations(transformations)
                        .await
                        .map_err(into_agent_error)?;
                }
                sandbox_credential_environment = Some(credentials);
                sandbox_env
            }
            None => {
                let forwarded = apply_credential_forwarding(
                    &resolved_env,
                    &credential_names,
                    settings.credential_forwarding.as_ref(),
                )
                .await
                .map_err(into_agent_error)?;
                warn_credential_brokering_unavailable(&resolved_env, &forwarded, &credential_names);
                forwarded
            }
        };
        let subscription_token = sandbox_auth_env
            .get(OPENCODE_SUBSCRIPTION_ACCESS_TOKEN_ENVIRONMENT_VARIABLE)
            .cloned();
        let mut env: IndexMap<String, String> = sandbox_auth_env
            .into_iter()
            .filter(|(name, _)| name != OPENCODE_SUBSCRIPTION_ACCESS_TOKEN_ENVIRONMENT_VARIABLE)
            .collect();
        if !gitlab_brokered
            && let (Some(subscription), Some(token)) =
                (&authentication.subscription, &subscription_token)
        {
            env.insert(
                "OPENCODE_AUTH_CONTENT".into(),
                subscription::create_opencode_subscription_auth_content(subscription, token),
            );
        }
        let client_app = opencode_client_app();
        env.insert("AI_SDK_HARNESS_CLIENT_APP".into(), client_app.clone());

        let home_dir = resolve_sandbox_home_dir(sandbox.as_ref(), abort.clone())
            .await
            .map_err(into_agent_error)?;
        let state_dir = harness_state_directory_path(&home_dir);
        let private_dir = harness_session_data_directory_path(&state_dir, &options.session_id);
        let skills_dir = format!("{}/.agents/skills", home_dir.trim_end_matches('/'));
        let (command, bootstrap_dir) = match &settings.executable_override {
            Some(command) => (command.clone(), None),
            None => {
                let directory =
                    apply_harness_bootstrap(sandbox.as_ref(), &opencode_bootstrap(), abort.clone())
                        .await
                        .map_err(into_agent_error)?;
                (
                    shell_quote(&format!("{directory}/node_modules/.bin/opencode")),
                    Some(directory),
                )
            }
        };
        let result = sandbox
            .run(SandboxRunOptions {
                process: SandboxProcessOptions {
                    command: format!(
                        "mkdir -p {} {}",
                        shell_quote(&options.work_dir),
                        shell_quote(&private_dir)
                    ),
                    abort_signal: abort.clone(),
                    ..Default::default()
                },
                max_output_length: None,
            })
            .await
            .map_err(into_agent_error)?;
        if result.exit_code != 0 {
            return Err(harness_error(format!(
                "Failed to create the OpenCode session directories: {}",
                result.stderr
            )));
        }
        let gitlab_model = gitlab_brokered;
        let config = turn::SessionConfig {
            sandbox,
            network: options.sandbox.network().cloned(),
            work_dir: options.work_dir.clone(),
            private_dir,
            home_dir: home_dir.clone(),
            skills_dir,
            command,
            bootstrap_dir,
            env,
            opencode_config,
            mcp_servers: settings.mcp_servers.clone(),
            provider: settings.provider.clone(),
            reasoning_variant: settings.reasoning_variant.clone(),
            headers: options.headers.clone(),
            permission_mode: options.permission_mode,
            inactive_builtin_tool_names: inactive_builtin_tool_names(
                options.builtin_tool_filter.as_ref(),
            ),
            port: settings.port,
            port_endpoint: settings.port_endpoint.clone(),
            startup_timeout: settings.startup_timeout.unwrap_or(Duration::from_secs(120)),
            client_app,
        };
        let runtime = turn::Runtime {
            attach: resume.server.clone(),
            resume_session_id: resume.open_code_session_id.clone(),
            ..Default::default()
        };
        Ok(Box::new(OpenCodeSession {
            inner: Arc::new(SessionInner {
                session_id: options.session_id,
                is_resume: options.resume.is_some(),
                config: Arc::new(config),
                provider: settings.provider.clone(),
                gitlab_model,
                home_dir,
                sandbox_credential_environment,
                cancellation_token: options
                    .cancellation_token
                    .map(|token| token.child_token())
                    .unwrap_or_default(),
                runtime: Arc::new(tokio::sync::Mutex::new(runtime)),
            }),
        }))
    }
}

struct SessionInner {
    session_id: String,
    is_resume: bool,
    config: Arc<turn::SessionConfig>,
    provider: Option<String>,
    /// Map models for brokered GitLab subscription credentials.
    gitlab_model: bool,
    home_dir: String,
    sandbox_credential_environment: Option<IndexMap<String, String>>,
    cancellation_token: CancellationToken,
    runtime: Arc<tokio::sync::Mutex<turn::Runtime>>,
}

impl SessionInner {
    fn resume_state(
        &self,
        session_id: Option<String>,
        server: Option<OpenCodeServerCoords>,
    ) -> HarnessResumeState {
        let state = OpenCodeResumeState {
            open_code_session_id: session_id,
            server,
            sandbox_credential_environment: self.sandbox_credential_environment.clone(),
        };
        HarnessResumeState {
            harness_id: OPENCODE_HARNESS_ID.to_string(),
            data: serde_json::to_value(state).unwrap(),
        }
    }

    fn transform_model(&self, model: Option<String>) -> AgentResult<Option<String>> {
        if !self.gitlab_model {
            return Ok(model);
        }
        subscription::resolve_opencode_gitlab_subscription_model(
            model.as_deref(),
            self.provider.as_deref(),
        )
        .map_err(into_agent_error)
    }
}

/// An OpenCode session.
pub struct OpenCodeSession {
    inner: Arc<SessionInner>,
}

#[async_trait]
impl HarnessSession for OpenCodeSession {
    fn harness_id(&self) -> &str {
        OPENCODE_HARNESS_ID
    }

    fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    fn is_resume(&self) -> bool {
        self.inner.is_resume
    }

    async fn prompt(
        &self,
        message: AgentMessage,
        options: HarnessTurnOptions,
    ) -> AgentResult<HarnessTurn> {
        let inner = self.inner.clone();
        let mut runtime = inner
            .runtime
            .clone()
            .try_lock_owned()
            .map_err(|_| harness_error("A turn is already running in this session."))?;
        let response_schema = match &options.response_format {
            Some(ResponseFormat::Json { schema: None, .. }) => {
                return Err(unsupported(
                    "Harness 'opencode' requires a JSON schema for structured output.",
                ));
            }
            Some(ResponseFormat::Json {
                schema: Some(schema),
                ..
            }) => Some(schema.clone()),
            _ => None,
        };
        let turn_settings = options.settings;
        let skills = write_skills({
            let mut write = WriteSkillsOptions::new(
                inner.config.sandbox.as_ref(),
                inner.home_dir.clone(),
                ".agents/skills".to_string(),
                turn_settings.skills.clone(),
            );
            write.abort_signal = options.cancellation_token.clone();
            write.invalid_skill_name_message = Some(Arc::new(|name| {
                format!("Invalid OpenCode skill name: {name}")
            }));
            write.invalid_skill_file_path_message = Some(Arc::new(|skill_name, file_path| {
                format!("Invalid OpenCode skill file path for {skill_name}: {file_path}")
            }));
            write
        })
        .await
        .map_err(into_agent_error)?;
        let model = inner.transform_model(turn_settings.model.clone())?;
        let token = inner.cancellation_token.child_token();
        if let Some(caller) = options.cancellation_token.clone() {
            let token = token.clone();
            tokio::spawn(async move {
                tokio::select! {
                    _ = caller.cancelled() => token.cancel(),
                    _ = token.cancelled() => {}
                }
            });
        }
        let config = turn::TurnConfig {
            prompt_parts: turn::prompt_parts(&message),
            instructions: turn_settings.instructions.clone(),
            model,
            response_schema,
            host: HostTools {
                tools: turn_settings.tools.clone(),
                before_tool_call: options.before_tool_call.clone(),
                after_tool_call: options.after_tool_call.clone(),
                cancellation_token: Some(token.clone()),
            },
            skills_changed: skills.changed,
            readable_skill_roots: turn_settings
                .skills
                .iter()
                .map(|skill| {
                    format!(
                        "{}/.agents/skills/{}",
                        inner.home_dir.trim_end_matches('/'),
                        skill.name
                    )
                })
                .collect(),
        };
        let session_config = inner.config.clone();
        Ok(HarnessTurn::spawn(move |sender| async move {
            let result = turn::run_turn(
                &mut runtime,
                &session_config,
                config,
                message,
                sender,
                token.clone(),
            )
            .await;
            token.cancel();
            result
        }))
    }

    async fn compact(&self, instructions: Option<String>) -> AgentResult<()> {
        if instructions
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
        {
            return Err(unsupported(
                "Harness 'opencode' supports native manual compaction, but OpenCode does not expose custom compaction instructions through the supported API.",
            ));
        }
        let mut runtime = self.inner.runtime.clone().try_lock_owned().map_err(|_| {
            unsupported(
                "Harness 'opencode' supports manual compaction between turns; compacting during an active turn is not supported.",
            )
        })?;
        let token = self.inner.cancellation_token.child_token();
        let result = turn::run_compaction(&mut runtime, &self.inner.config, token.clone()).await;
        token.cancel();
        result.map_err(into_agent_error)
    }

    async fn detach(&self) -> AgentResult<HarnessResumeState> {
        let mut runtime = self.inner.runtime.lock().await;
        // The server keeps running for the next process; its relay does not.
        let server = match runtime.server.take() {
            Some(server) if server.relay.is_none() => Some(server.coords()),
            Some(server) => {
                server.close(self.inner.config.sandbox.as_ref()).await;
                None
            }
            None => runtime.attach.clone(),
        };
        Ok(self.inner.resume_state(runtime.session_id.clone(), server))
    }

    async fn stop(&self) -> AgentResult<HarnessResumeState> {
        self.inner.cancellation_token.cancel();
        let mut runtime = self.inner.runtime.lock().await;
        runtime.close(self.inner.config.sandbox.as_ref()).await;
        let session_id = runtime
            .session_id
            .clone()
            .or(runtime.resume_session_id.clone());
        Ok(self.inner.resume_state(session_id, None))
    }

    async fn destroy(&self) -> AgentResult<()> {
        self.inner.cancellation_token.cancel();
        self.inner
            .runtime
            .lock()
            .await
            .close(self.inner.config.sandbox.as_ref())
            .await;
        Ok(())
    }
}
