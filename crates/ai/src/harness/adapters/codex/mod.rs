//! Codex: `codex app-server` in the sandbox, driven from Rust over its
//! JSON-RPC stdio protocol. Port of `@ai-sdk/harness-codex`. Enabled by the
//! `harness-codex` feature.
//!
//! The app server runs for the whole session and keeps the Codex thread
//! loaded between turns. The caller's tools are Codex dynamic tools, run by
//! the session when Codex calls them. Codex runs without approvals, so the
//! only permission mode is `allow-all`.
//!
//! [rust] Not ported: the TS bridge process and its reconnect, attach and
//! turn-suspension machinery (the app server runs under this process's
//! control), and raw stream parts.

pub mod auth;
mod client;
pub mod subscription;
pub mod tools;
mod turn;

#[cfg(test)]
mod auth_tests;
#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::host_tools::HostTools;
use crate::harness::auth::HarnessCredentialForwarding;
use crate::harness::bootstrap::{
    HarnessBootstrap, HarnessBootstrapCommand, HarnessBootstrapFile, apply_harness_bootstrap,
};
use crate::harness::error::{HarnessError, into_agent_error};
use crate::harness::network::HarnessRequestTransformationSources;
use crate::harness::sandbox::SandboxSession;
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
use crate::{AgentMessage, AgentResult, ThinkingLevel};

pub use auth::{
    CODEX_CREDENTIAL_ENVIRONMENT_VARIABLES, CodexAuthenticationMode,
    CodexResolvedAuthenticationMode, DEFAULT_OPENAI_BASE_URL,
};
pub use tools::codex_builtin_tools;

pub const CODEX_HARNESS_ID: &str = "codex";
/// `VERSION` of the ported `@ai-sdk/harness-codex` package.
pub const VERSION: &str = "1.0.192";
/// The Codex CLI version the bootstrap installs.
pub const CODEX_CLI_VERSION: &str = "0.167.0";
/// `CODEX_BOOTSTRAP_DIR`, under the harness state directory.
pub const CODEX_BOOTSTRAP_DIR: &str = ".harness-bootstrap/codex";
/// `DEFAULT_CODEX_MODEL`: GPT-5.5, since newer models' code-mode tools are
/// not callable through the custom provider the harness uses.
pub const DEFAULT_CODEX_MODEL: &str = "gpt-5.5";

pub(crate) const CODEX_API: &str = "codex";
pub(crate) const CODEX_PROVIDER: &str = "openai";

/// Base URLs that serve Codex's standalone search route.
const STANDALONE_WEB_SEARCH_BASE_URLS: [&str; 2] = [
    "https://api.openai.com/v1",
    "https://chatgpt.com/backend-api/codex",
];
const FINAL_MESSAGE_INSTRUCTION: &str =
    "Only respond with your `final` message once you have fully addressed the user request.";

/// `CODEX_CLIENT_APP`: the `User-Agent` and `x-client-app` attribution.
fn codex_client_app() -> String {
    format!("ai-sdk-harness-codex/{VERSION}")
}

/// Reasoning effort for reasoning-capable models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CodexReasoningEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl CodexReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }

    fn from_thinking_level(level: ThinkingLevel) -> Self {
        match level {
            ThinkingLevel::Minimal | ThinkingLevel::Low => Self::Low,
            ThinkingLevel::Medium => Self::Medium,
            ThinkingLevel::High => Self::High,
            ThinkingLevel::Xhigh => Self::Xhigh,
            ThinkingLevel::Max => Self::Max,
        }
    }
}

/// `CodexHarnessSettings`.
#[derive(Clone, Default)]
pub struct CodexHarnessSettings {
    pub auth: Option<CodexAuthenticationMode>,
    /// Customizes each credential value before it is forwarded into the
    /// sandbox.
    pub credential_forwarding: Option<HarnessCredentialForwarding>,
    /// Codex configuration passed through as-is (snake_case keys). Values
    /// the adapter manages win, except the `model_reasoning_summary`
    /// default.
    pub codex_config: Option<Map<String, Value>>,
    /// MCP server definitions keyed by server name, in Codex's own format.
    pub mcp_servers: Option<IndexMap<String, Value>>,
    /// Effort when no turn sets a thinking level. Unset defers to Codex.
    pub reasoning_effort: Option<CodexReasoningEffort>,
    /// Allow live web search.
    pub web_search: bool,
    /// [rust] Run this `codex` executable (a sandbox path or command)
    /// instead of installing the pinned CLI with the bootstrap.
    pub executable: Option<String>,
    /// [rust] The host environment for credential discovery. Defaults to
    /// this process's environment.
    pub process_env: Option<IndexMap<String, String>>,
    /// [rust] Extra environment variables for the `codex` process. They win
    /// over the credentials the adapter resolves.
    pub env: IndexMap<String, String>,
    /// [rust] Extra arguments, before the `app-server` subcommand.
    pub extra_args: Vec<String>,
    /// [rust] Run the sessions of this harness on one `codex app-server`
    /// process (one per sandbox object, command and environment) instead
    /// of one each. Each thread keeps its session's working directory. Each session gets its own thread's messages; the last session
    /// to stop ends the process.
    pub shared_app_server: bool,
}

/// The adapter's resume state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexResumeState {
    /// The Codex thread to resume (`thread/resume`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_configuration_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_credential_environment: Option<IndexMap<String, String>>,
}

/// The Codex harness.
#[derive(Clone, Default)]
pub struct CodexHarness {
    settings: Arc<CodexHarnessSettings>,
    shared: turn::SharedAppServers,
}

impl CodexHarness {
    pub fn new(settings: CodexHarnessSettings) -> Self {
        Self {
            settings: Arc::new(settings),
            shared: turn::SharedAppServers::default(),
        }
    }
}

/// `getCodexBootstrap`: installs the pinned CLI with npm.
pub fn codex_bootstrap() -> HarnessBootstrap {
    let package = json!({
        "name": "harness-codex",
        "version": "0.0.0",
        "private": true,
        "dependencies": { "@openai/codex": CODEX_CLI_VERSION },
    });
    HarnessBootstrap {
        harness_id: CODEX_HARNESS_ID.to_string(),
        bootstrap_dir: CODEX_BOOTSTRAP_DIR.to_string(),
        files: vec![HarnessBootstrapFile {
            path: format!("{CODEX_BOOTSTRAP_DIR}/package.json"),
            content: serde_json::to_string_pretty(&package).unwrap() + "\n",
        }],
        commands: vec![
            HarnessBootstrapCommand {
                command: "npm install --no-audit --no-fund --loglevel=error".into(),
            },
            HarnessBootstrapCommand {
                command: "./node_modules/.bin/codex --version".into(),
            },
        ],
    }
}

/// `resolveCodexRuntime`: the Codex configuration and model of a turn.
pub(crate) fn resolve_codex_runtime(
    env: &IndexMap<String, String>,
    model: Option<&str>,
    instructions: Option<&str>,
    codex_config: Option<&Map<String, Value>>,
    mcp_servers: Option<&IndexMap<String, Value>>,
    headers: Option<&IndexMap<String, String>>,
) -> AgentResult<(Map<String, Value>, Option<String>)> {
    let get = |name: &str| env.get(name).filter(|value| !value.is_empty());
    let gateway_base_url = get("AI_GATEWAY_BASE_URL");
    let has_gateway_auth = get("AI_GATEWAY_API_KEY").is_some() || gateway_base_url.is_some();
    if has_gateway_auth && gateway_base_url.is_none() {
        return Err(HarnessError::new(
            "AI Gateway auth was selected but AI_GATEWAY_BASE_URL is missing from the Codex environment.",
        )
        .into());
    }
    let api_base_url = if has_gateway_auth {
        gateway_base_url.cloned()
    } else {
        env.get("OPENAI_BASE_URL").cloned().or_else(|| {
            (env.contains_key("CODEX_API_KEY") || headers.is_some())
                .then(|| DEFAULT_OPENAI_BASE_URL.to_string())
        })
    };
    let codex_model = model.map(|model| {
        if has_gateway_auth && !model.contains('/') {
            format!("openai/{model}")
        } else {
            model.to_string()
        }
    });
    let mut config = Map::new();
    config.insert("model_reasoning_summary".into(), json!("detailed"));
    if let Some(user) = codex_config {
        config.extend(user.clone());
    }
    config.insert(
        "developer_instructions".into(),
        json!(
            [
                instructions.filter(|i| !i.is_empty()),
                Some(FINAL_MESSAGE_INSTRUCTION)
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join("\n\n")
        ),
    );
    if let Some(base_url) = api_base_url {
        let client_app = has_gateway_auth.then(codex_client_app);
        let mut provider = Map::new();
        provider.insert(
            "name".into(),
            json!(
                get("CODEX_MODEL_PROVIDER_NAME")
                    .cloned()
                    .unwrap_or_else(|| "Agent Bridge OpenAI".into())
            ),
        );
        provider.insert("base_url".into(), json!(base_url));
        provider.insert("env_key".into(), json!("CODEX_API_KEY"));
        provider.insert("wire_api".into(), json!("responses"));
        provider.insert("supports_websockets".into(), json!(false));
        if STANDALONE_WEB_SEARCH_BASE_URLS.contains(&base_url.trim_end_matches('/')) {
            provider.insert("supports_standalone_web_search".into(), json!(true));
        }
        if headers.is_some() || client_app.is_some() {
            let mut http_headers = Map::new();
            for (name, value) in headers.into_iter().flatten() {
                http_headers.insert(name.clone(), json!(value));
            }
            if let Some(app) = client_app {
                http_headers.insert("User-Agent".into(), json!(app));
                http_headers.insert("x-client-app".into(), json!(app));
            }
            provider.insert("http_headers".into(), Value::Object(http_headers));
        }
        config.insert("model_provider".into(), json!("agent_bridge_openai"));
        config.insert(
            "model_providers".into(),
            json!({ "agent_bridge_openai": provider }),
        );
    }
    if let Some(servers) = mcp_servers {
        config.insert(
            "mcp_servers".into(),
            Value::Object(servers.clone().into_iter().collect()),
        );
    }
    Ok((config, codex_model))
}

fn harness_error(message: impl Into<String>) -> crate::AgentError {
    HarnessError::new(message).into()
}

#[async_trait]
impl Harness for CodexHarness {
    fn id(&self) -> &str {
        CODEX_HARNESS_ID
    }

    fn builtin_tools(&self) -> &IndexMap<String, BuiltinTool> {
        codex_builtin_tools()
    }

    fn supports_builtin_tool_approvals(&self) -> bool {
        false
    }

    fn supports_builtin_tool_filtering(&self) -> bool {
        true
    }

    async fn bootstrap(&self) -> AgentResult<Option<HarnessBootstrap>> {
        Ok(Some(codex_bootstrap()))
    }

    async fn start(&self, options: HarnessStartOptions) -> AgentResult<Box<dyn HarnessSession>> {
        validate_start_options(self, &options)?;
        let settings = self.settings.clone();
        let resume: CodexResumeState = match &options.resume {
            Some(state) => serde_json::from_value(state.data.clone())
                .map_err(|error| harness_error(format!("Invalid Codex resume state: {error}")))?,
            None => CodexResumeState::default(),
        };
        let sandbox = get_restricted_sandbox_session(&options.sandbox);
        let abort = options.cancellation_token.clone();

        let process_env = settings
            .process_env
            .clone()
            .unwrap_or_else(|| std::env::vars().collect());
        let authentication_mode =
            auth::resolve_codex_authentication_mode(settings.auth.as_ref(), &process_env);
        let resolved = subscription::resolve_codex_authentication(
            subscription::ResolveCodexAuthenticationOptions {
                auth: settings.auth.clone(),
                process_env: Some(process_env.clone()),
                auth_credentials_store_mode: settings
                    .codex_config
                    .as_ref()
                    .and_then(|config| config.get("cli_auth_credentials_store"))
                    .cloned(),
                ..Default::default()
            },
        )
        .await
        .map_err(into_agent_error)?;
        let environment = resolved.environment.clone();
        let credential_names: Vec<String> = CODEX_CREDENTIAL_ENVIRONMENT_VARIABLES
            .iter()
            .map(|name| name.to_string())
            .collect();
        let mut sandbox_environment = environment.clone();
        let mut sandbox_credential_environment = None;
        match options
            .sandbox
            .network()
            .filter(|network| network.supports_add_request_transformations())
        {
            Some(network) => {
                let credentials = resolve_sandbox_credential_environment(
                    &environment,
                    &credential_names,
                    settings.credential_forwarding.as_ref(),
                    resume.sandbox_credential_environment.as_ref(),
                )
                .await
                .map_err(into_agent_error)?;
                sandbox_environment.extend(credentials.clone());
                let sources = HarnessRequestTransformationSources {
                    env: environment.clone(),
                    sandbox_env: sandbox_environment.clone(),
                    auth: authentication_mode,
                };
                let transformations =
                    subscription::create_codex_subscription_request_transformations(
                        &sources,
                        resolved.request_headers.as_ref(),
                    )
                    .map_err(into_agent_error)?;
                if !transformations.is_empty() {
                    network
                        .add_request_transformations(transformations.clone())
                        .await
                        .map_err(into_agent_error)?;
                    // Request transformations apply to HTTP only: the custom
                    // provider keeps Codex off WebSockets.
                    if authentication_mode == CodexResolvedAuthenticationMode::Direct
                        && !environment.contains_key("OPENAI_BASE_URL")
                    {
                        sandbox_environment
                            .insert("OPENAI_BASE_URL".into(), DEFAULT_OPENAI_BASE_URL.into());
                    }
                }
                sandbox_credential_environment = Some(credentials);
            }
            None => {
                sandbox_environment = apply_credential_forwarding(
                    &sandbox_environment,
                    &credential_names,
                    settings.credential_forwarding.as_ref(),
                )
                .await
                .map_err(into_agent_error)?;
                warn_credential_brokering_unavailable(
                    &environment,
                    &sandbox_environment,
                    &credential_names,
                );
            }
        }

        sandbox_environment.extend(settings.env.clone());
        let home_dir = resolve_sandbox_home_dir(sandbox.as_ref(), abort.clone())
            .await
            .map_err(into_agent_error)?;
        let executable = match &settings.executable {
            Some(executable) => executable.clone(),
            None => {
                let bootstrap_dir =
                    apply_harness_bootstrap(sandbox.as_ref(), &codex_bootstrap(), abort)
                        .await
                        .map_err(into_agent_error)?;
                shell_quote(&format!("{bootstrap_dir}/node_modules/.bin/codex"))
            }
        };
        let executable = std::iter::once(executable)
            .chain(settings.extra_args.iter().map(|arg| shell_quote(arg)))
            .collect::<Vec<_>>()
            .join(" ");
        let shared = settings.shared_app_server.then(|| self.shared.clone());

        Ok(Box::new(CodexSession {
            inner: Arc::new(SessionInner {
                settings,
                session_id: options.session_id,
                is_resume: options.resume.is_some(),
                sandbox,
                work_dir: options.work_dir,
                home_dir,
                executable,
                environment: sandbox_environment,
                headers: options.headers,
                sandbox_credential_environment,
                builtin_tool_filter: options.builtin_tool_filter,
                thread_id: Arc::new(StdMutex::new(resume.thread_id.filter(|id| !id.is_empty()))),
                fingerprint: StdMutex::new(resume.turn_configuration_fingerprint),
                cancellation_token: options
                    .cancellation_token
                    .map(|token| token.child_token())
                    .unwrap_or_default(),
                runtime: Arc::new(tokio::sync::Mutex::new(turn::Runtime::default())),
                shared,
            }),
        }))
    }
}

struct SessionInner {
    settings: Arc<CodexHarnessSettings>,
    session_id: String,
    is_resume: bool,
    sandbox: Arc<dyn SandboxSession>,
    work_dir: String,
    home_dir: String,
    executable: String,
    environment: IndexMap<String, String>,
    headers: Option<IndexMap<String, String>>,
    sandbox_credential_environment: Option<IndexMap<String, String>>,
    builtin_tool_filter: Option<BuiltinToolFilter>,
    thread_id: Arc<StdMutex<Option<String>>>,
    fingerprint: StdMutex<Option<String>>,
    cancellation_token: CancellationToken,
    /// The app server; also the turn lock.
    runtime: Arc<tokio::sync::Mutex<turn::Runtime>>,
    shared: Option<turn::SharedAppServers>,
}

impl SessionInner {
    fn resume_state(&self) -> HarnessResumeState {
        let state = CodexResumeState {
            thread_id: self.thread_id.lock().unwrap().clone(),
            turn_configuration_fingerprint: self.fingerprint.lock().unwrap().clone(),
            sandbox_credential_environment: self.sandbox_credential_environment.clone(),
        };
        HarnessResumeState {
            harness_id: CODEX_HARNESS_ID.to_string(),
            data: serde_json::to_value(state).unwrap(),
        }
    }

    async fn shut_down(&self) {
        self.cancellation_token.cancel();
        self.runtime.lock().await.close().await;
    }
}

/// A Codex session.
pub struct CodexSession {
    inner: Arc<SessionInner>,
}

#[async_trait]
impl HarnessSession for CodexSession {
    fn harness_id(&self) -> &str {
        CODEX_HARNESS_ID
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
        let output_schema = match &options.response_format {
            Some(ResponseFormat::Json { schema: None, .. }) => {
                return Err(HarnessError::capability_unsupported(
                    "Harness 'codex' requires a JSON schema for structured output.",
                    Some(CODEX_HARNESS_ID.to_string()),
                )
                .into());
            }
            Some(ResponseFormat::Json {
                schema: Some(schema),
                ..
            }) => Some(schema.clone()),
            _ => None,
        };
        let settings = &inner.settings;
        let turn_settings = options.settings;

        // `synchronizeTurnConfiguration`: skills, instructions and tools are
        // fixed when a thread starts, so a change starts a new thread.
        let skills = write_skills({
            let mut write = WriteSkillsOptions::new(
                inner.sandbox.as_ref(),
                inner.home_dir.clone(),
                ".agents/skills",
                turn_settings.skills.clone(),
            );
            write.abort_signal = options.cancellation_token.clone();
            write.invalid_skill_name_message =
                Some(Arc::new(|name| format!("Invalid Codex skill name: {name}")));
            write.invalid_skill_file_path_message = Some(Arc::new(|skill_name, file_path| {
                format!("Invalid Codex skill file path for {skill_name}: {file_path}")
            }));
            write
        })
        .await
        .map_err(into_agent_error)?;
        let definitions: Vec<crate::Tool> = turn_settings
            .tools
            .iter()
            .map(|tool| tool.definition())
            .collect();
        let next_fingerprint = tools::fingerprint_turn_configuration(
            turn_settings.instructions.as_deref(),
            &definitions,
        );
        let restart_thread = {
            let mut fingerprint = inner.fingerprint.lock().unwrap();
            let restart = inner.thread_id.lock().unwrap().is_some()
                && (skills.changed
                    || fingerprint.as_deref().is_some_and(|previous| {
                        previous.starts_with("v2:") && previous != next_fingerprint
                    }));
            *fingerprint = Some(next_fingerprint);
            restart
        };
        if restart_thread {
            *inner.thread_id.lock().unwrap() = None;
        }

        let model = turn_settings
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_CODEX_MODEL.to_string());
        let (codex_config, codex_model) = resolve_codex_runtime(
            &inner.environment,
            Some(&model),
            turn_settings.instructions.as_deref(),
            settings.codex_config.as_ref(),
            settings.mcp_servers.as_ref(),
            inner.headers.as_ref(),
        )?;
        let effort = turn_settings
            .thinking_level
            .map(CodexReasoningEffort::from_thinking_level)
            .or(settings.reasoning_effort)
            .map(CodexReasoningEffort::as_str);

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
        let steer: Arc<StdMutex<Option<turn::SteerHandle>>> = Arc::default();
        let config = turn::TurnConfig {
            sandbox: inner.sandbox.clone(),
            executable: inner.executable.clone(),
            work_dir: inner.work_dir.clone(),
            env: inner.environment.clone(),
            thread_id: inner.thread_id.lock().unwrap().clone(),
            restart_thread,
            codex_model,
            codex_config,
            filter: inner.builtin_tool_filter.clone(),
            web_search: settings.web_search,
            effort,
            output_schema,
            host: HostTools {
                tools: turn_settings.tools.clone(),
                before_tool_call: options.before_tool_call.clone(),
                after_tool_call: options.after_tool_call.clone(),
                cancellation_token: Some(token.clone()),
            },
            model,
            thread_id_out: inner.thread_id.clone(),
            steer: steer.clone(),
            shared: inner.shared.clone(),
        };
        let turn = HarnessTurn::spawn(move |sender| async move {
            let result = turn::run_turn(&mut runtime, config, message, sender, token.clone()).await;
            token.cancel();
            result
        });
        Ok(turn.with_steer(move |message| {
            let handle = steer.lock().unwrap().clone();
            async move {
                match handle {
                    Some(handle) => handle.steer(message).await,
                    None => Err(harness_error("The turn is not running.")),
                }
            }
        }))
    }

    async fn probe(&self) -> AgentResult<crate::harness::probe::HarnessProbe> {
        let inner = &self.inner;
        let (codex_config, _) = resolve_codex_runtime(
            &inner.environment,
            None,
            None,
            inner.settings.codex_config.as_ref(),
            None,
            inner.headers.as_ref(),
        )?;
        turn::probe(
            inner.sandbox.as_ref(),
            &inner.executable,
            &inner.work_dir,
            &inner.environment,
            &codex_config,
        )
        .await
        .map_err(into_agent_error)
    }

    async fn detach(&self) -> AgentResult<HarnessResumeState> {
        Ok(self.inner.resume_state())
    }

    async fn stop(&self) -> AgentResult<HarnessResumeState> {
        self.inner.shut_down().await;
        Ok(self.inner.resume_state())
    }

    async fn destroy(&self) -> AgentResult<()> {
        self.inner.shut_down().await;
        Ok(())
    }
}
