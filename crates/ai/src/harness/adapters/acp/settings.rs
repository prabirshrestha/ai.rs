//! Port of `acp-v1-settings.ts` and `acp-tool-call.ts`.

use std::fmt;
use std::net::SocketAddr;
use std::sync::{Arc, LazyLock};

use futures::future::BoxFuture;
use indexmap::{IndexMap, IndexSet};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::harness::auth::HarnessCredentialForwarding;
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::network::HarnessRequestTransformation;
use crate::harness::sandbox::SandboxSession;
use crate::harness::tools::{BuiltinTool, PermissionMode};

use super::auth::{AcpAuthenticationMode, AcpClientApp};

/// The MCP server that serves the caller's tools to the agent.
pub const HOST_TOOL_MCP_SERVER_NAME: &str = "ai-sdk-harness-tools";

/// `ACPSource`: how the bootstrap installs the agent.
///
/// [rust] The TS `npm-locked` source installs the agent together with the
/// JS bridge from a pnpm lockfile. There is no bridge here, so the agent's
/// npm package is installed by name and exact version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AcpSource {
    /// `npm install` of `package@version`; the executable is the package's
    /// `node_modules/.bin` entry.
    Npm { package: String, version: String },
    /// A shell command run with `HOME` set to the bootstrap's private home;
    /// the executable is installed to `$HOME/.local/bin`.
    InstallCommand { command: String },
}

/// `ACPAuthentication`: the ACP `authenticate` call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpAuthentication {
    pub method_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_capabilities: Option<Map<String, Value>>,
}

/// `ACPProviderAuthentication`: agent environment values for AI Gateway
/// routing, as profile values (`{ "$source": ... }` objects are replaced).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcpProviderAuthentication {
    pub gateway: AcpGatewayProviderAuthentication,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcpGatewayProviderAuthentication {
    pub env: IndexMap<String, Value>,
}

/// `ACPModelMapping`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AcpModelMapping {
    /// `session/set_config_option` with `configId: path`.
    SessionConfigOption { path: String },
    /// The non-standard `session/set_model` with the model at `path`.
    SessionModel { path: String },
}

/// Input of [`AcpCredentialBrokering`].
#[derive(Debug, Clone, Copy)]
pub struct AcpCredentialBrokeringOptions<'a> {
    pub env: &'a IndexMap<String, String>,
    pub sandbox_env: Option<&'a IndexMap<String, String>>,
    pub headers: Option<&'a IndexMap<String, String>>,
}

/// `ACPCredentialBrokering`.
pub type AcpCredentialBrokering = Arc<
    dyn Fn(AcpCredentialBrokeringOptions<'_>) -> HarnessResult<Vec<HarnessRequestTransformation>>
        + Send
        + Sync,
>;

/// `ACPAuthenticationFile`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcpAuthenticationFile {
    /// Path relative to the agent's home directory.
    pub path: String,
    pub content: String,
}

/// Input of [`AcpAuthenticationFiles`].
#[derive(Debug, Clone, Copy)]
pub struct AcpAuthenticationFilesOptions<'a> {
    pub env: &'a IndexMap<String, String>,
    pub sandbox_env: &'a IndexMap<String, String>,
    pub credential_brokering_available: bool,
}

/// `ACPAuthenticationFiles`.
pub type AcpAuthenticationFiles =
    Arc<dyn Fn(AcpAuthenticationFilesOptions<'_>) -> Vec<AcpAuthenticationFile> + Send + Sync>;

/// `ACPPermissionModeTarget`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum AcpPermissionModeTarget {
    SessionMode {
        mode_id: String,
    },
    SessionConfigOption {
        config_id: String,
        value: AcpConfigValue,
    },
}

/// `string | boolean`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AcpConfigValue {
    String(String),
    Boolean(bool),
}

/// `ACPPermissionModeMapping`: the session setting of each permission mode,
/// `None` when the agent cannot run in it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcpPermissionModeMapping {
    #[serde(rename = "allow-reads")]
    pub allow_reads: Option<AcpPermissionModeTarget>,
    #[serde(rename = "allow-edits")]
    pub allow_edits: Option<AcpPermissionModeTarget>,
    #[serde(rename = "allow-all")]
    pub allow_all: Option<AcpPermissionModeTarget>,
}

impl AcpPermissionModeMapping {
    pub fn get(&self, mode: PermissionMode) -> Option<&AcpPermissionModeTarget> {
        match mode {
            PermissionMode::AllowReads => self.allow_reads.as_ref(),
            PermissionMode::AllowEdits => self.allow_edits.as_ref(),
            PermissionMode::AllowAll => self.allow_all.as_ref(),
        }
    }
}

/// `ACPInstructionMapping`: where the agent takes native instructions.
/// Session metadata paths are relative to `_meta`; launch environment paths
/// are relative to the JSON object stored in `variable`; a filesystem path is
/// relative to the agent's home directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AcpInstructionMapping {
    SessionMeta { path: Vec<String> },
    LaunchEnvJson { variable: String, path: Vec<String> },
    Filesystem { path: String },
}

/// `ACPOutputSchemaMapping`: where the JSON schema of structured output
/// goes, below the `session/prompt` `_meta`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AcpOutputSchemaMapping {
    SessionPromptMeta { path: Vec<String> },
}

/// `ACPHostToolMCPTransport`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AcpHostToolMcpTransport {
    /// [rust] A stdio MCP server command that relays over FIFOs in the
    /// sandbox to this process.
    #[default]
    Stdio,
    /// [rust] An HTTP MCP endpoint served by this process. The agent must
    /// reach [`AcpHttpRelayOptions::url_host`] from the sandbox.
    Http,
}

/// [rust] Where this process serves the HTTP host tool endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcpHttpRelayOptions {
    /// The listening address. Defaults to `127.0.0.1:0`.
    pub bind: SocketAddr,
    /// The host the agent connects to. Defaults to the bound IP.
    pub url_host: Option<String>,
}

impl Default for AcpHttpRelayOptions {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 0)),
            url_host: None,
        }
    }
}

/// `ACPResolvedHostToolCall`.
#[derive(Debug, Clone, PartialEq)]
pub struct AcpResolvedHostToolCall {
    pub tool_name: String,
    /// The tool input; must be an object.
    pub input: Value,
}

/// Input of [`AcpResolveHostToolCall`].
#[derive(Debug, Clone, Copy)]
pub struct AcpResolveHostToolCallOptions<'a> {
    /// The ACP tool call as seen so far (`ACPToolCall`: `toolCallId`,
    /// `name`, `title`, `kind`, `status`, `content`, `locations`,
    /// `rawInput`, `rawOutput`, `_meta`).
    pub tool_call: &'a Value,
    pub server_name: &'a str,
    pub tool_names: &'a [String],
}

/// `ACPResolveHostToolCall`: which host tool an ACP tool call runs, if any.
pub type AcpResolveHostToolCall =
    Arc<dyn Fn(AcpResolveHostToolCallOptions<'_>) -> Option<AcpResolvedHostToolCall> + Send + Sync>;

/// `resolveAuthenticationEnvironment`: resolves adapter-owned
/// authentication on the host before environment values are forwarded.
pub type AcpResolveAuthenticationEnvironment = Arc<
    dyn Fn(
            Option<AcpAuthenticationMode>,
            IndexMap<String, String>,
        ) -> BoxFuture<'static, HarnessResult<IndexMap<String, String>>>
        + Send
        + Sync,
>;

/// Input of [`AcpPrepareSandbox`].
#[derive(Clone)]
pub struct AcpPrepareSandboxOptions {
    pub sandbox: Arc<dyn SandboxSession>,
    pub work_dir: String,
    pub abort_signal: Option<CancellationToken>,
}

/// `prepareSandbox`: runs after authentication files are written, before
/// the agent starts.
pub type AcpPrepareSandbox =
    Arc<dyn Fn(AcpPrepareSandboxOptions) -> BoxFuture<'static, HarnessResult<()>> + Send + Sync>;

/// `ACPV1Settings`: an ACP agent.
///
/// [rust] Not ported: `askUserQuestions` (no preset uses it), `isMcpToolCall`
/// (it only marks tool calls dynamic, which ai.rs does not distinguish), the
/// bridge port and reconnect settings, and the deprecated `mintBridgeToken`.
#[derive(Clone)]
pub struct AcpHarnessSettings {
    /// A stable kebab-case id.
    pub harness_id: String,
    /// The `clientInfo` sent to the agent, and the AI Gateway attribution.
    pub client_app: AcpClientApp,
    /// The agent's builtin tools, keyed by harness name.
    pub builtin_tools: IndexMap<String, BuiltinTool>,
    /// The ACP tool call title prefix of each builtin tool, used to match
    /// tool calls that carry no programmatic name.
    pub builtin_tool_titles: IndexMap<String, String>,
    /// MCP servers in ACP `McpServer` format, keyed by name.
    pub mcp_servers: Option<IndexMap<String, Value>>,
    pub resolve_host_tool_call: AcpResolveHostToolCall,
    pub auth: Option<AcpAuthenticationMode>,
    pub resolve_authentication_environment: Option<AcpResolveAuthenticationEnvironment>,
    pub source: AcpSource,
    /// A bare command name installed by `source`.
    pub executable: String,
    pub args: Vec<String>,
    /// Host variables passed to the agent as-is.
    pub forward_env: Vec<String>,
    /// Credential variables, brokered or forwarded. Set together with
    /// `credential_brokering`.
    pub credential_env: Option<Vec<String>>,
    pub credential_brokering: Option<AcpCredentialBrokering>,
    pub authentication_files: Option<AcpAuthenticationFiles>,
    pub prepare_sandbox: Option<AcpPrepareSandbox>,
    pub credential_forwarding: Option<HarnessCredentialForwarding>,
    /// Fixed agent environment values.
    pub env: IndexMap<String, String>,
    pub authentication: Option<AcpAuthentication>,
    pub client_capabilities: Option<Map<String, Value>>,
    pub provider_authentication: Option<AcpProviderAuthentication>,
    pub model_mapping: AcpModelMapping,
    /// Relative to the agent's home. Defaults to `.agents/skills`.
    pub skills_directory: Option<String>,
    pub instruction_mapping: Option<AcpInstructionMapping>,
    pub output_schema_mapping: Option<AcpOutputSchemaMapping>,
    pub host_tool_mcp_transport: AcpHostToolMcpTransport,
    /// Whether the agent can take images: as prompts when it does not
    /// advertise them, and in host tool results.
    pub image_content: bool,
    pub permission_mode_mapping: Option<AcpPermissionModeMapping>,
    /// Work-dir-relative paths whose writes always ask outside `allow-all`.
    pub write_protected_paths: Vec<String>,
    /// `session.meta`: `_meta` of `session/new`.
    pub session_meta: Option<Map<String, Value>>,
    /// [rust] Run this agent command (a sandbox path or command, already
    /// shell-quoted) instead of installing `source` with the bootstrap.
    pub executable_override: Option<String>,
    /// [rust] The host environment for credential discovery. Defaults to
    /// this process's environment.
    pub process_env: Option<IndexMap<String, String>>,
    /// [rust] The HTTP host tool endpoint.
    pub http_relay: AcpHttpRelayOptions,
    /// The model reported when the agent reports none and the turn names
    /// none; the harness id otherwise.
    pub fallback_model: Option<String>,
}

impl fmt::Debug for AcpHarnessSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AcpHarnessSettings")
            .field("harness_id", &self.harness_id)
            .field("source", &self.source)
            .field("executable", &self.executable)
            .field("args", &self.args)
            .finish_non_exhaustive()
    }
}

impl AcpHarnessSettings {
    /// Settings with the required fields and defaults for the rest.
    pub fn new(
        harness_id: impl Into<String>,
        client_app: AcpClientApp,
        source: AcpSource,
        executable: impl Into<String>,
        model_mapping: AcpModelMapping,
        resolve_host_tool_call: AcpResolveHostToolCall,
    ) -> Self {
        Self {
            harness_id: harness_id.into(),
            client_app,
            builtin_tools: IndexMap::new(),
            builtin_tool_titles: IndexMap::new(),
            mcp_servers: None,
            resolve_host_tool_call,
            auth: None,
            resolve_authentication_environment: None,
            source,
            executable: executable.into(),
            args: Vec::new(),
            forward_env: Vec::new(),
            credential_env: None,
            credential_brokering: None,
            authentication_files: None,
            prepare_sandbox: None,
            credential_forwarding: None,
            env: IndexMap::new(),
            authentication: None,
            client_capabilities: None,
            provider_authentication: None,
            model_mapping,
            skills_directory: None,
            instruction_mapping: None,
            output_schema_mapping: None,
            host_tool_mcp_transport: AcpHostToolMcpTransport::default(),
            image_content: false,
            permission_mode_mapping: None,
            write_protected_paths: Vec::new(),
            session_meta: None,
            executable_override: None,
            process_env: None,
            fallback_model: None,
            http_relay: AcpHttpRelayOptions::default(),
        }
    }
}

static HARNESS_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").unwrap());
static EXECUTABLE_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._-]*$").unwrap());
static ENVIRONMENT_VARIABLE_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").unwrap());

fn error(message: impl Into<String>) -> crate::harness::error::BoxError {
    HarnessError::new(message).into()
}

/// `validateACPV1Settings`.
pub fn validate_acp_settings(settings: &AcpHarnessSettings) -> HarnessResult<()> {
    if settings.credential_env.is_none() != settings.credential_brokering.is_none() {
        return Err(error(
            "ACP credentialEnv and credentialBrokering must be configured together.",
        ));
    }
    if settings
        .mcp_servers
        .as_ref()
        .is_some_and(|servers| servers.contains_key(HOST_TOOL_MCP_SERVER_NAME))
    {
        return Err(error(format!(
            "ACP MCP server name \"{HOST_TOOL_MCP_SERVER_NAME}\" is reserved for HarnessAgent tools."
        )));
    }
    if !HARNESS_ID.is_match(&settings.harness_id) {
        return Err(error(format!(
            "ACP harnessId must be a stable kebab-case identifier; received {}.",
            serde_json::to_string(&settings.harness_id).unwrap()
        )));
    }
    for entry in &settings.write_protected_paths {
        validate_write_protected_path(entry)?;
    }
    match &settings.source {
        AcpSource::Npm { package, version } => {
            if package.is_empty() || version.is_empty() {
                return Err(error("ACP source package and version must not be empty."));
            }
        }
        AcpSource::InstallCommand { command } => {
            if command.trim().is_empty() {
                return Err(error("ACP source.command must not be empty."));
            }
        }
    }
    if !EXECUTABLE_NAME.is_match(&settings.executable) {
        return Err(error(format!(
            "ACP executable must be a bare command name without a path; received {}.",
            serde_json::to_string(&settings.executable).unwrap()
        )));
    }
    let forwarded: IndexSet<&String> = settings.forward_env.iter().collect();
    let credentials: IndexSet<&String> = settings.credential_env.iter().flatten().collect();
    for name in forwarded.iter().chain(credentials.iter()) {
        validate_environment_variable_name(name)?;
    }
    for key in &credentials {
        if forwarded.contains(key) {
            return Err(error(format!(
                "ACP runtime environment key \"{key}\" cannot be configured in both forwardEnv and credentialEnv."
            )));
        }
    }
    for (key, value) in &settings.env {
        validate_environment_variable_name(key)?;
        if value.contains('\0') {
            return Err(error(format!(
                "ACP runtime environment value for {key} contains NUL."
            )));
        }
        if forwarded.contains(key) {
            return Err(error(format!(
                "ACP runtime environment key \"{key}\" cannot be configured in both forwardEnv and env."
            )));
        }
        if credentials.contains(key) {
            return Err(error(format!(
                "ACP runtime environment key \"{key}\" cannot be configured in both credentialEnv and env."
            )));
        }
    }
    Ok(())
}

/// `resolveACPAgentEnvironment`.
pub fn resolve_acp_agent_environment(
    settings: &AcpHarnessSettings,
    env: &IndexMap<String, String>,
    credential_env: &IndexMap<String, String>,
) -> IndexMap<String, String> {
    let mut environment = IndexMap::new();
    for name in &settings.forward_env {
        if let Some(value) = env.get(name).filter(|value| !value.is_empty()) {
            environment.insert(name.clone(), value.clone());
        }
    }
    for name in settings.credential_env.iter().flatten() {
        if let Some(value) = credential_env.get(name).filter(|value| !value.is_empty()) {
            environment.insert(name.clone(), value.clone());
        }
    }
    environment.extend(settings.env.clone());
    environment
}

fn validate_environment_variable_name(name: &str) -> HarnessResult<()> {
    if ENVIRONMENT_VARIABLE_NAME.is_match(name) {
        Ok(())
    } else {
        Err(error(format!(
            "ACP environment variable name is invalid: {}.",
            serde_json::to_string(name).unwrap()
        )))
    }
}

fn validate_write_protected_path(entry: &str) -> HarnessResult<()> {
    if entry.trim().is_empty()
        || entry.contains('\0')
        || entry.starts_with('/')
        || entry
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return Err(error(format!(
            "ACP writeProtectedPaths entries must be relative paths inside the working directory without \".\" or \"..\" segments; received {}.",
            serde_json::to_string(entry).unwrap()
        )));
    }
    Ok(())
}

/// `resolveACPSkillsDirectory`.
pub fn resolve_acp_skills_directory(
    agent_home_dir: &str,
    skills_directory: Option<&str>,
) -> HarnessResult<String> {
    let directory = skills_directory.unwrap_or(DEFAULT_ACP_SKILLS_DIRECTORY);
    let normalized = crate::harness::paths::normalize_posix(directory);
    let windows_absolute = directory.len() >= 2
        && directory.as_bytes()[1] == b':'
        && directory.as_bytes()[0].is_ascii_alphabetic();
    if directory.is_empty()
        || directory.contains('\\')
        || directory.starts_with('/')
        || windows_absolute
        || directory.split('/').any(|segment| segment == "..")
        || normalized == "."
        || normalized.starts_with("../")
    {
        return Err(error(format!(
            "ACP skillsDirectory {} must be a relative POSIX path without traversal.",
            serde_json::to_string(directory).unwrap()
        )));
    }
    Ok(format!(
        "{}/{normalized}",
        agent_home_dir.trim_end_matches('/')
    ))
}

/// `DEFAULT_ACP_SKILLS_DIRECTORY`.
pub const DEFAULT_ACP_SKILLS_DIRECTORY: &str = ".agents/skills";

/// `ACP_SKILL_NAME_PATTERN`.
pub static ACP_SKILL_NAME_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").unwrap());
