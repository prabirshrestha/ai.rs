//! Claude Code: the `claude` CLI in the sandbox, driven from Rust over its
//! stream-json stdio protocol. Port of `@ai-sdk/harness-claude-code`.
//! Enabled by the `harness-claude-code` feature.
//!
//! Each turn runs `claude` once, resuming the session's Claude conversation
//! after the first turn. The caller's tools are served to Claude Code as the
//! in-process `harness-tools` MCP server, and native tool approvals are
//! answered inline through `before_tool_call`.
//!
//! [rust] Not ported: the TS bridge process and its reconnect, attach and
//! turn-suspension machinery (the CLI runs under this process's control
//! for the whole turn, so approvals are answered while it waits), the
//! `agentProgressSummaries` and `forwardSubagentText` settings (subagent
//! messages are not part of the session's conversation), and raw stream
//! parts.

pub mod auth;
pub mod questions;
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
use crate::harness::tools::{BuiltinTool, BuiltinToolFilter, PermissionMode};
use crate::harness::utils::write_skills::{WriteSkillsOptions, write_skills};
use crate::harness::utils::{
    apply_credential_forwarding, get_restricted_sandbox_session,
    resolve_sandbox_credential_environment, resolve_sandbox_home_dir, shell_quote,
    warn_credential_brokering_unavailable,
};
use crate::{AgentMessage, AgentResult, ThinkingLevel};

pub use auth::{
    CLAUDE_CODE_CREDENTIAL_ENVIRONMENT_VARIABLES, ClaudeCodeAuthenticationMode,
    ClaudeCodeResolvedAuthenticationMode,
};
pub use tools::claude_code_builtin_tools;

pub const CLAUDE_CODE_HARNESS_ID: &str = "claude-code";
/// `VERSION` of the ported `@ai-sdk/harness-claude-code` package.
pub const VERSION: &str = "1.0.192";
/// The Claude Code CLI version the bootstrap installs.
pub const CLAUDE_CODE_CLI_VERSION: &str = "2.1.319";
/// `CLAUDE_CODE_BOOTSTRAP_DIR`, under the harness state directory.
pub const CLAUDE_CODE_BOOTSTRAP_DIR: &str = ".harness-bootstrap/claude-code";

pub(crate) const CLAUDE_CODE_API: &str = "claude-code";
pub(crate) const CLAUDE_CODE_PROVIDER: &str = "anthropic";

/// `CLAUDE_CODE_CLIENT_APP`: the `x-client-app` attribution for AI Gateway.
fn claude_code_client_app() -> String {
    format!("ai-sdk-harness-claude-code/{VERSION}")
}

/// `display` of an adaptive or enabled [`ClaudeCodeThinkingConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClaudeCodeThinkingDisplay {
    Summarized,
    Omitted,
}

/// `ClaudeCodeThinkingConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ClaudeCodeThinkingConfig {
    Adaptive {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display: Option<ClaudeCodeThinkingDisplay>,
    },
    Enabled {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display: Option<ClaudeCodeThinkingDisplay>,
    },
    Disabled,
}

impl Default for ClaudeCodeThinkingConfig {
    fn default() -> Self {
        Self::Adaptive {
            display: Some(ClaudeCodeThinkingDisplay::Summarized),
        }
    }
}

/// How much effort Claude applies when adaptive thinking is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClaudeCodeEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ClaudeCodeEffort {
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

/// A value of `skillOverrides`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClaudeCodeSkillOverride {
    On,
    NameOnly,
    UserInvocableOnly,
    Off,
}

/// Claude Code hints for a host tool.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaudeCodeToolHint {
    /// The tool has no side effects, so Claude Code may run it in parallel
    /// and in plan mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
    /// Send the tool's definition from the first request, even when tool
    /// search defers MCP tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub always_load: Option<bool>,
}

/// `ClaudeCodeHarnessSettings`.
#[derive(Clone, Default)]
pub struct ClaudeCodeHarnessSettings {
    pub auth: Option<ClaudeCodeAuthenticationMode>,
    /// Use Claude Code's classifier-based `auto` permission mode for
    /// `allow-reads` and `allow-edits`. Only escalated calls ask
    /// `before_tool_call`.
    pub auto_mode: bool,
    /// Customizes each credential value before it is forwarded into the
    /// sandbox.
    pub credential_forwarding: Option<HarnessCredentialForwarding>,
    /// MCP server definitions keyed by server name, in Claude Code's own
    /// format. `harness-tools` is reserved.
    pub mcp_servers: Option<IndexMap<String, Value>>,
    /// Hints for host tools, keyed by tool name.
    pub tool_hints: IndexMap<String, ClaudeCodeToolHint>,
    /// Cap on the CLI's internal turns per prompt.
    pub max_turns: Option<u32>,
    /// How named skills are exposed to Claude Code.
    pub skill_overrides: Option<IndexMap<String, ClaudeCodeSkillOverride>>,
    /// Maximum estimated spend in USD per prompt.
    pub max_budget_usd: Option<f64>,
    /// Absolute paths Claude may use besides the working directory.
    pub additional_directories: Option<Vec<String>>,
    /// Use only `mcp_servers`, ignoring other MCP configuration.
    pub strict_mcp_config: Option<bool>,
    /// How long Claude waits for an answer to `AskUserQuestion`.
    pub question_timeout_ms: Option<u64>,
    /// Move per-session sections of the system prompt into the first user
    /// message, so the system prompt caches across sessions.
    pub exclude_dynamic_sections: Option<bool>,
    /// Environment variables for the `claude` process.
    pub env: IndexMap<String, String>,
    /// Defaults to adaptive thinking with summarized display.
    pub thinking: Option<ClaudeCodeThinkingConfig>,
    /// Effort when no turn sets a thinking level.
    pub effort: Option<ClaudeCodeEffort>,
    /// [rust] Run this `claude` executable (a sandbox path or command)
    /// instead of installing the pinned CLI with the bootstrap.
    pub executable: Option<String>,
    /// [rust] The host environment for credential discovery. Defaults to
    /// this process's environment.
    pub process_env: Option<IndexMap<String, String>>,
    /// [rust] Extra arguments for every `claude` run, after the adapter's.
    pub extra_args: Vec<String>,
}

/// The adapter's resume state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaudeCodeResumeState {
    /// The Claude conversation to resume (`claude --resume <id>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claude_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_credential_environment: Option<IndexMap<String, String>>,
}

/// The Claude Code harness.
#[derive(Clone, Default)]
pub struct ClaudeCodeHarness {
    settings: Arc<ClaudeCodeHarnessSettings>,
}

impl ClaudeCodeHarness {
    pub fn new(settings: ClaudeCodeHarnessSettings) -> Self {
        Self {
            settings: Arc::new(settings),
        }
    }
}

/// `getClaudeCodeBootstrap`: installs the pinned CLI with npm.
pub fn claude_code_bootstrap() -> HarnessBootstrap {
    let package = json!({
        "name": "harness-claude-code",
        "version": "0.0.0",
        "private": true,
        "dependencies": { "@anthropic-ai/claude-code": CLAUDE_CODE_CLI_VERSION },
    });
    HarnessBootstrap {
        harness_id: CLAUDE_CODE_HARNESS_ID.to_string(),
        bootstrap_dir: CLAUDE_CODE_BOOTSTRAP_DIR.to_string(),
        files: vec![HarnessBootstrapFile {
            path: format!("{CLAUDE_CODE_BOOTSTRAP_DIR}/package.json"),
            content: serde_json::to_string_pretty(&package).unwrap() + "\n",
        }],
        commands: vec![
            HarnessBootstrapCommand {
                command: "npm install --no-audit --no-fund --loglevel=error".into(),
            },
            HarnessBootstrapCommand {
                command: "./node_modules/.bin/claude --version".into(),
            },
        ],
    }
}

/// The `claude` command line of one turn (the Agent SDK's
/// `ProcessTransport` flags).
#[derive(Debug, Clone, Default)]
pub(crate) struct ClaudeCodeCliOptions {
    pub model: Option<String>,
    pub thinking: ClaudeCodeThinkingConfig,
    pub effort: Option<ClaudeCodeEffort>,
    pub max_turns: Option<u32>,
    pub max_budget_usd: Option<f64>,
    pub json_schema: Option<Value>,
    pub resume: Option<String>,
    pub continue_conversation: bool,
    pub allowed_tools: Vec<String>,
    pub disallowed_tools: Vec<String>,
    pub tools: Option<Vec<String>>,
    pub mcp_servers: Map<String, Value>,
    pub strict_mcp_config: bool,
    pub permission_mode: PermissionMode,
    pub auto_mode: bool,
    pub additional_directories: Vec<String>,
    pub settings: Map<String, Value>,
}

impl ClaudeCodeCliOptions {
    pub fn args(&self) -> Vec<String> {
        let mut args: Vec<String> = [
            "--output-format",
            "stream-json",
            "--verbose",
            "--input-format",
            "stream-json",
        ]
        .map(String::from)
        .to_vec();
        match self.thinking {
            ClaudeCodeThinkingConfig::Adaptive { display }
            | ClaudeCodeThinkingConfig::Enabled { display } => {
                args.push("--thinking=adaptive".into());
                if let Some(display) = display {
                    args.push("--thinking-display".into());
                    args.push(
                        match display {
                            ClaudeCodeThinkingDisplay::Summarized => "summarized",
                            ClaudeCodeThinkingDisplay::Omitted => "omitted",
                        }
                        .into(),
                    );
                }
            }
            ClaudeCodeThinkingConfig::Disabled => args.push("--thinking=disabled".into()),
        }
        let mut flag = |name: &str, value: String| {
            args.push(name.to_string());
            args.push(value);
        };
        if let Some(effort) = self.effort {
            flag("--effort", effort.as_str().into());
        }
        if let Some(max_turns) = self.max_turns {
            flag("--max-turns", max_turns.to_string());
        }
        if let Some(budget) = self.max_budget_usd {
            flag("--max-budget-usd", budget.to_string());
        }
        if let Some(model) = &self.model {
            flag("--model", model.clone());
        }
        if let Some(schema) = &self.json_schema {
            flag("--json-schema", schema.to_string());
        }
        args.push("--permission-prompt-tool=stdio".into());
        if let Some(resume) = &self.resume {
            args.push("--resume".into());
            args.push(resume.clone());
        } else if self.continue_conversation {
            args.push("--continue".into());
        }
        if !self.allowed_tools.is_empty() {
            args.push("--allowedTools".into());
            args.push(self.allowed_tools.join(","));
        }
        if !self.disallowed_tools.is_empty() {
            args.push("--disallowedTools".into());
            args.push(self.disallowed_tools.join(","));
        }
        if let Some(tools) = &self.tools {
            args.push("--tools".into());
            args.push(tools.join(","));
        }
        if !self.mcp_servers.is_empty() {
            args.push("--mcp-config".into());
            args.push(json!({ "mcpServers": self.mcp_servers }).to_string());
        }
        args.push("--setting-sources".into());
        args.push("user".into());
        if self.strict_mcp_config {
            args.push("--strict-mcp-config".into());
        }
        args.push("--permission-mode".into());
        args.push(
            match self.permission_mode {
                PermissionMode::AllowAll => "bypassPermissions",
                _ if self.auto_mode => "auto",
                PermissionMode::AllowEdits => "acceptEdits",
                PermissionMode::AllowReads => "default",
            }
            .into(),
        );
        if self.permission_mode == PermissionMode::AllowAll {
            args.push("--allow-dangerously-skip-permissions".into());
        }
        args.push("--include-partial-messages".into());
        for directory in &self.additional_directories {
            args.push("--add-dir".into());
            args.push(directory.clone());
        }
        if !self.settings.is_empty() {
            args.push("--settings".into());
            args.push(Value::Object(self.settings.clone()).to_string());
        }
        args
    }
}

fn harness_error(message: impl Into<String>) -> crate::AgentError {
    HarnessError::new(message).into()
}

#[async_trait]
impl Harness for ClaudeCodeHarness {
    fn id(&self) -> &str {
        CLAUDE_CODE_HARNESS_ID
    }

    fn builtin_tools(&self) -> &IndexMap<String, BuiltinTool> {
        claude_code_builtin_tools()
    }

    fn supports_builtin_tool_approvals(&self) -> bool {
        true
    }

    fn supports_builtin_tool_filtering(&self) -> bool {
        true
    }

    async fn bootstrap(&self) -> AgentResult<Option<HarnessBootstrap>> {
        Ok(Some(claude_code_bootstrap()))
    }

    async fn start(&self, options: HarnessStartOptions) -> AgentResult<Box<dyn HarnessSession>> {
        validate_start_options(self, &options)?;
        let settings = self.settings.clone();
        if settings
            .mcp_servers
            .as_ref()
            .is_some_and(|servers| servers.contains_key(tools::HOST_TOOL_SERVER))
        {
            return Err(harness_error(
                "Claude Code MCP server name \"harness-tools\" is reserved for HarnessAgent tools.",
            ));
        }
        let resume: ClaudeCodeResumeState = match &options.resume {
            Some(state) => serde_json::from_value(state.data.clone()).map_err(|error| {
                harness_error(format!("Invalid Claude Code resume state: {error}"))
            })?,
            None => ClaudeCodeResumeState::default(),
        };
        let sandbox = get_restricted_sandbox_session(&options.sandbox);
        let abort = options.cancellation_token.clone();

        // Credentials: resolved on the host, brokered through the sandbox
        // network policy when it can, forwarded otherwise.
        let process_env = settings
            .process_env
            .clone()
            .unwrap_or_else(|| std::env::vars().collect());
        let authentication_mode =
            auth::resolve_claude_code_authentication_mode(settings.auth.as_ref(), &process_env);
        let resolved = subscription::resolve_claude_code_authentication(
            subscription::ResolveClaudeCodeAuthenticationOptions {
                auth: settings.auth.clone(),
                process_env: Some(process_env.clone()),
                ..Default::default()
            },
        )
        .await
        .map_err(into_agent_error)?;
        let mut environment = resolved.clone();
        if resolved.contains_key("AI_GATEWAY_BASE_URL") {
            environment.insert(
                "CLAUDE_AGENT_SDK_CLIENT_APP".into(),
                claude_code_client_app(),
            );
        }
        environment.extend(settings.env.clone());
        if let Some(headers) = &options.headers {
            environment.insert(
                "ANTHROPIC_CUSTOM_HEADERS".into(),
                headers
                    .iter()
                    .map(|(name, value)| format!("{name}: {value}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        let credential_names: Vec<String> = CLAUDE_CODE_CREDENTIAL_ENVIRONMENT_VARIABLES
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
                let mut transformations =
                    auth::create_claude_code_request_transformations(&sources)
                        .map_err(into_agent_error)?;
                transformations.extend(
                    subscription::create_claude_code_subscription_request_transformations(&sources)
                        .map_err(into_agent_error)?,
                );
                if !transformations.is_empty() {
                    network
                        .add_request_transformations(transformations)
                        .await
                        .map_err(into_agent_error)?;
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
        sandbox_environment
            .entry("CLAUDE_CODE_ENTRYPOINT".into())
            .or_insert_with(|| "sdk-cli".into());

        let home_dir = resolve_sandbox_home_dir(sandbox.as_ref(), abort.clone())
            .await
            .map_err(into_agent_error)?;
        let executable = match &settings.executable {
            Some(executable) => executable.clone(),
            None => {
                let bootstrap_dir =
                    apply_harness_bootstrap(sandbox.as_ref(), &claude_code_bootstrap(), abort)
                        .await
                        .map_err(into_agent_error)?;
                shell_quote(&format!("{bootstrap_dir}/node_modules/.bin/claude"))
            }
        };

        Ok(Box::new(ClaudeCodeSession {
            inner: Arc::new(SessionInner {
                settings,
                session_id: options.session_id,
                is_resume: options.resume.is_some(),
                sandbox,
                work_dir: options.work_dir,
                home_dir,
                executable,
                environment: sandbox_environment,
                sandbox_credential_environment,
                permission_mode: options.permission_mode,
                builtin_tool_filter: options.builtin_tool_filter,
                claude_session_id: Arc::new(StdMutex::new(resume.claude_session_id)),
                first_turn: StdMutex::new(true),
                cancellation_token: options
                    .cancellation_token
                    .map(|token| token.child_token())
                    .unwrap_or_default(),
                turn_lock: Arc::new(tokio::sync::Mutex::new(())),
            }),
        }))
    }
}

struct SessionInner {
    settings: Arc<ClaudeCodeHarnessSettings>,
    session_id: String,
    is_resume: bool,
    sandbox: Arc<dyn SandboxSession>,
    work_dir: String,
    home_dir: String,
    executable: String,
    environment: IndexMap<String, String>,
    sandbox_credential_environment: Option<IndexMap<String, String>>,
    permission_mode: PermissionMode,
    builtin_tool_filter: Option<BuiltinToolFilter>,
    claude_session_id: Arc<StdMutex<Option<String>>>,
    first_turn: StdMutex<bool>,
    cancellation_token: CancellationToken,
    turn_lock: Arc<tokio::sync::Mutex<()>>,
}

/// A Claude Code session.
pub struct ClaudeCodeSession {
    inner: Arc<SessionInner>,
}

impl SessionInner {
    fn resume_state(&self) -> HarnessResumeState {
        let state = ClaudeCodeResumeState {
            claude_session_id: self.claude_session_id.lock().unwrap().clone(),
            sandbox_credential_environment: self.sandbox_credential_environment.clone(),
        };
        HarnessResumeState {
            harness_id: CLAUDE_CODE_HARNESS_ID.to_string(),
            data: serde_json::to_value(state).unwrap(),
        }
    }
}

#[async_trait]
impl HarnessSession for ClaudeCodeSession {
    fn harness_id(&self) -> &str {
        CLAUDE_CODE_HARNESS_ID
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
        let turn_guard = inner
            .turn_lock
            .clone()
            .try_lock_owned()
            .map_err(|_| harness_error("A turn is already running in this session."))?;
        let structured_output = match &options.response_format {
            Some(ResponseFormat::Json { schema: None, .. }) => {
                return Err(HarnessError::capability_unsupported(
                    "Harness 'claude-code' requires a JSON schema for structured output.",
                    Some(CLAUDE_CODE_HARNESS_ID.to_string()),
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

        write_skills({
            let mut write = WriteSkillsOptions::new(
                inner.sandbox.as_ref(),
                inner.home_dir.clone(),
                ".claude/skills",
                turn_settings.skills.clone(),
            );
            write.abort_signal = options.cancellation_token.clone();
            write.invalid_skill_name_message = Some(Arc::new(|name| {
                format!("Invalid Claude Code skill name: {name}")
            }));
            write.invalid_skill_file_path_message = Some(Arc::new(|skill_name, file_path| {
                format!("Invalid Claude Code skill file path for {skill_name}: {file_path}")
            }));
            write.trailing_newline = true;
            write
        })
        .await
        .map_err(into_agent_error)?;

        let permission_mode = inner.permission_mode;
        let auto_mode = settings.auto_mode && permission_mode != PermissionMode::AllowAll;
        let filter = inner.builtin_tool_filter.as_ref();
        let inactive_native_tools = tools::resolve_inactive_native_tools(filter);
        let host = HostTools {
            tools: turn_settings.tools.clone(),
            before_tool_call: options.before_tool_call.clone(),
            after_tool_call: options.after_tool_call.clone(),
            cancellation_token: None,
        };

        let mcp_servers: Map<String, Value> = settings
            .mcp_servers
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect();
        let mut allowed_tools = Vec::new();
        if !turn_settings.skills.is_empty() {
            allowed_tools.push("Skill".to_string());
        }
        if auto_mode {
            allowed_tools.push(format!("{}*", tools::HOST_TOOL_PREFIX));
        }
        let mut cli_settings =
            tools::create_permission_settings(permission_mode, auto_mode, &inactive_native_tools)
                .unwrap_or_default();
        if let Some(overrides) = &settings.skill_overrides {
            cli_settings.insert(
                "skillOverrides".into(),
                serde_json::to_value(overrides).unwrap(),
            );
        }
        if let Some(overrides) = tools::create_gateway_model_overrides(
            turn_settings.model.as_deref(),
            &inner.environment,
        ) {
            cli_settings.insert("modelOverrides".into(), Value::Object(overrides));
        }
        let first_turn = std::mem::replace(&mut *inner.first_turn.lock().unwrap(), false);
        let resume = inner.claude_session_id.lock().unwrap().clone();
        let cli = ClaudeCodeCliOptions {
            model: turn_settings.model.clone(),
            thinking: settings.thinking.unwrap_or_default(),
            effort: turn_settings
                .thinking_level
                .map(ClaudeCodeEffort::from_thinking_level)
                .or(settings.effort),
            max_turns: settings.max_turns,
            max_budget_usd: settings.max_budget_usd,
            json_schema: structured_output.clone(),
            continue_conversation: resume.is_none() && (inner.is_resume || !first_turn),
            resume,
            allowed_tools,
            disallowed_tools: inactive_native_tools.clone(),
            tools: tools::resolve_native_tools(filter),
            mcp_servers,
            strict_mcp_config: settings.strict_mcp_config == Some(true),
            permission_mode,
            auto_mode,
            additional_directories: settings.additional_directories.clone().unwrap_or_default(),
            settings: cli_settings,
        };
        let command = std::iter::once(inner.executable.clone())
            .chain(cli.args().iter().map(|arg| shell_quote(arg)))
            .chain(settings.extra_args.iter().map(|arg| shell_quote(arg)))
            .collect::<Vec<_>>()
            .join(" ");

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
            command,
            work_dir: inner.work_dir.clone(),
            env: inner.environment.clone(),
            permission_mode,
            auto_mode,
            inactive_native_tools,
            readable_skill_roots: turn_settings
                .skills
                .iter()
                .map(|skill| format!("{}/.claude/skills/{}", inner.home_dir, skill.name))
                .collect(),
            additional_directories: settings.additional_directories.clone().unwrap_or_default(),
            question_timeout_ms: settings.question_timeout_ms,
            append_system_prompt: turn_settings.instructions.clone().filter(|i| !i.is_empty()),
            exclude_dynamic_sections: settings.exclude_dynamic_sections,
            structured_output: structured_output.is_some(),
            host,
            tool_hints: settings.tool_hints.clone(),
            model: turn_settings.model.clone().unwrap_or_default(),
            claude_session_id: inner.claude_session_id.clone(),
            steer: steer.clone(),
        };
        let turn = HarnessTurn::spawn(move |sender| async move {
            let _turn_guard = turn_guard;
            let result = turn::run_turn(config, message, sender, token.clone()).await;
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
        probe_claude_code(&self.inner)
            .await
            .map_err(into_agent_error)
    }

    async fn detach(&self) -> AgentResult<HarnessResumeState> {
        Ok(self.inner.resume_state())
    }

    async fn stop(&self) -> AgentResult<HarnessResumeState> {
        self.inner.cancellation_token.cancel();
        let _ = self.inner.turn_lock.lock().await;
        Ok(self.inner.resume_state())
    }

    async fn destroy(&self) -> AgentResult<()> {
        self.inner.cancellation_token.cancel();
        let _ = self.inner.turn_lock.lock().await;
        Ok(())
    }
}

/// [rust] `claude auth status`, then the models and commands of an
/// `initialize` control request to a short-lived `claude`.
async fn probe_claude_code(
    inner: &SessionInner,
) -> crate::harness::error::HarnessResult<crate::harness::probe::HarnessProbe> {
    use crate::harness::adapters::stdio_process::{JsonLineProcess, process_exit_error};
    use crate::harness::probe::{
        HarnessProbe, claude_code_auth, claude_code_commands, claude_code_models,
    };
    use crate::harness::sandbox::{SandboxProcessOptions, SandboxRunOptions};

    let options = |command: String| SandboxProcessOptions {
        command,
        working_directory: Some(inner.work_dir.clone()),
        env: Some(inner.environment.clone()),
        abort_signal: None,
    };
    let mut probe = HarnessProbe::default();
    let status = inner
        .sandbox
        .run(SandboxRunOptions {
            process: options(format!("{} auth status", inner.executable)),
            max_output_length: Some(64 * 1024),
        })
        .await;
    match status {
        Ok(status) => {
            let (signed_in, account) = claude_code_auth(status.exit_code == 0, &status.stdout);
            probe.signed_in = Some(signed_in);
            probe.account = account;
        }
        Err(error) => probe.error = Some(error.to_string()),
    }

    let command = std::iter::once(inner.executable.clone())
        .chain(
            [
                "--output-format",
                "stream-json",
                "--verbose",
                "--input-format",
                "stream-json",
            ]
            .map(String::from),
        )
        .chain(inner.settings.extra_args.iter().map(|arg| shell_quote(arg)))
        .collect::<Vec<_>>()
        .join(" ");
    let (process, mut output) =
        JsonLineProcess::spawn(inner.sandbox.as_ref(), options(command), "claude").await?;
    let read = async {
        process
            .send(&json!({
                "type": "control_request",
                "request_id": "probe",
                "request": { "subtype": "initialize" },
            }))
            .await?;
        while let Some(message) = output.recv().await {
            if message.get("type").and_then(Value::as_str) == Some("control_response") {
                return Ok(message);
            }
        }
        Err(process_exit_error(&process, "exited before it answered initialize").into())
    };
    let result: crate::harness::error::HarnessResult<Value> =
        match tokio::time::timeout(std::time::Duration::from_secs(30), read).await {
            Ok(result) => result,
            Err(_) => {
                Err(HarnessError::new("Claude Code did not answer the probe in time.").into())
            }
        };
    process.close_stdin().await;
    process.kill().await;
    // `{ type: control_response, response: { subtype, response: {...} } }`.
    let response = result?["response"].clone();
    if response.get("subtype").and_then(Value::as_str) == Some("error") {
        let error = response
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("Claude Code refused initialize.")
            .to_string();
        probe.error.get_or_insert(error);
        return Ok(probe);
    }
    probe.models = claude_code_models(&response);
    probe.commands = claude_code_commands(&response);
    Ok(probe)
}
