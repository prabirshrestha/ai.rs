//! Port of `harness-acp/src/v1/acp-v1-settings.ts`.

use std::fmt;
use std::sync::{Arc, LazyLock};

use futures::future::BoxFuture;
use indexmap::{IndexMap, IndexSet};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::harness::errors::HarnessResult;
use crate::harness::provider::JsonValue;
use crate::harness::sandbox::SandboxSession;
use crate::harness::sdk::messages::ToolResultPart;
use crate::harness::spec::authentication::HarnessV1CredentialForwarding;
use crate::harness::spec::network_sandbox::HarnessV1RequestTransformation;
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::spec::tools::HarnessV1PermissionMode;
use crate::harness::utils::plain_error;

use crate::harness::adapters::acp::acp_auth::ACPAuthenticationMode;
use crate::harness::adapters::acp::acp_tool_call::{ACPResolveHostToolCall, ACPToolCall};

/// `ACPSerializableValue`: any JSON value.
pub type ACPSerializableValue = JsonValue;

/// `ACPValueSource['$source']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ACPValueSourceKind {
    GatewayApiKey,
    GatewayBaseUrl,
    GatewayAuthorization,
    ClientApp,
    ClientAppName,
    ClientAppVersion,
}

/// `ACPValueSource`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPValueSource {
    #[serde(rename = "$source")]
    pub source: ACPValueSourceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensure_suffix: Option<String>,
}

/// `ACPProfileValue`: a JSON value whose objects may be
/// [`ACPValueSource`]s (`{ "$source": … }`).
///
/// [rust] Kept as JSON, as it crosses into the bridge unchanged.
pub type ACPProfileValue = JsonValue;

/// `ACPNpmLockedSource`: locked NPM installation of the ACP agent. The
/// manifest is the complete bridge installation, so in addition to the agent
/// package it must declare every dependency of the `@ai-sdk/harness-acp`
/// bridge manifest at the same version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPNpmLockedSource {
    pub package_json: String,
    pub pnpm_lock_yaml: String,
    /// Optional pnpm workspace configuration required by the locked install.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pnpm_workspace_yaml: Option<String>,
}

/// `ACPInstallCommandSource`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ACPInstallCommandSource {
    pub command: String,
}

/// `ACPSource`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ACPSource {
    NpmLocked(ACPNpmLockedSource),
    InstallCommand(ACPInstallCommandSource),
}

/// `ACPAuthentication`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPAuthentication {
    pub method_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<IndexMap<String, ACPSerializableValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_capabilities: Option<IndexMap<String, ACPSerializableValue>>,
}

/// `ACPProviderAuthentication['gateway']`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ACPGatewayProviderAuthentication {
    pub env: IndexMap<String, ACPProfileValue>,
}

/// `ACPProviderAuthentication`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ACPProviderAuthentication {
    pub gateway: ACPGatewayProviderAuthentication,
}

/// `ACPModelMapping`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ACPModelMapping {
    SessionConfigOption { path: String },
    SessionModel { path: String },
}

/// Input of [`ACPCredentialBrokering`].
#[derive(Debug, Clone, Copy)]
pub struct ACPCredentialBrokeringOptions<'a> {
    pub env: &'a IndexMap<String, String>,
    pub sandbox_env: Option<&'a IndexMap<String, String>>,
    pub headers: Option<&'a IndexMap<String, String>>,
}

/// `ACPCredentialBrokering`.
pub type ACPCredentialBrokering = Arc<
    dyn Fn(ACPCredentialBrokeringOptions<'_>) -> Vec<HarnessV1RequestTransformation> + Send + Sync,
>;

/// `ACPAuthenticationFile`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ACPAuthenticationFile {
    /// Path relative to the ACP agent's home directory.
    pub path: String,
    pub content: String,
}

/// Input of [`ACPAuthenticationFiles`].
#[derive(Debug, Clone, Copy)]
pub struct ACPAuthenticationFilesOptions<'a> {
    /// Host values resolved for the agent.
    pub env: &'a IndexMap<String, String>,
    /// Values exposed to the sandbox process after credential brokering.
    pub sandbox_env: &'a IndexMap<String, String>,
    pub credential_brokering_available: bool,
}

/// `ACPAuthenticationFiles`.
pub type ACPAuthenticationFiles =
    Arc<dyn Fn(ACPAuthenticationFilesOptions<'_>) -> Vec<ACPAuthenticationFile> + Send + Sync>;

/// `ACPPermissionModeTarget`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum ACPPermissionModeTarget {
    SessionMode {
        mode_id: String,
    },
    SessionConfigOption {
        config_id: String,
        value: ACPPermissionModeTargetValue,
    },
}

/// `value: string | boolean`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ACPPermissionModeTargetValue {
    String(String),
    Boolean(bool),
}

/// `ACPPermissionModeMapping`: `Record<HarnessV1PermissionMode, target | null>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ACPPermissionModeMapping {
    #[serde(rename = "allow-reads")]
    pub allow_reads: Option<ACPPermissionModeTarget>,
    #[serde(rename = "allow-edits")]
    pub allow_edits: Option<ACPPermissionModeTarget>,
    #[serde(rename = "allow-all")]
    pub allow_all: Option<ACPPermissionModeTarget>,
}

impl ACPPermissionModeMapping {
    /// The target of `mode`.
    pub fn get(&self, mode: HarnessV1PermissionMode) -> Option<&ACPPermissionModeTarget> {
        match mode {
            HarnessV1PermissionMode::AllowReads => self.allow_reads.as_ref(),
            HarnessV1PermissionMode::AllowEdits => self.allow_edits.as_ref(),
            HarnessV1PermissionMode::AllowAll => self.allow_all.as_ref(),
        }
    }
}

/// `ACPInstructionMapping`: where an ACP agent accepts native session
/// instructions. Session metadata paths are relative to `_meta`; launch
/// environment paths are relative to the parsed JSON object stored in
/// `variable`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ACPInstructionMapping {
    SessionMeta { path: Vec<String> },
    LaunchEnvJson { variable: String, path: Vec<String> },
    Filesystem { path: String },
}

/// `ACPOutputSchemaMapping`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ACPOutputSchemaMapping {
    SessionPromptMeta { path: Vec<String> },
}

/// `ACPHostToolMCPTransport`: transport of the harness-owned MCP server that
/// exposes host tools to an ACP agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ACPHostToolMCPTransport {
    Stdio,
    Http,
}

/// A `nonTextContentTypes` entry (`'image'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ACPNonTextContentType {
    Image,
}

/// Input of `askUserQuestions.fromNativeRequest`.
#[derive(Debug, Clone, Copy)]
pub struct ACPFromNativeRequestOptions<'a> {
    pub native_request: &'a JsonValue,
    pub native_tool_call: Option<&'a ACPToolCall>,
}

/// Input of `askUserQuestions.toNativeResponse`.
#[derive(Debug, Clone, Copy)]
pub struct ACPToNativeResponseOptions<'a> {
    pub native_request: &'a JsonValue,
    pub tool_result: &'a ToolResultPart,
}

/// Input of `askUserQuestions.matchesNativeRequest`.
#[derive(Debug, Clone, Copy)]
pub struct ACPMatchesNativeRequestOptions<'a> {
    pub previous_native_request: &'a JsonValue,
    pub native_request: &'a JsonValue,
}

/// `ACPAskUserQuestionsSettings`.
///
/// [rust] TS callbacks can throw; these return the error.
#[derive(Clone)]
pub struct ACPAskUserQuestionsSettings {
    pub request_method: String,
    pub is_native_tool_call: Option<
        Arc<dyn Fn(&ACPToolCall) -> crate::harness::errors::HarnessResult<bool> + Send + Sync>,
    >,
    /// Returns a `tool-call` stream part, or `None`.
    pub from_native_request: Arc<
        dyn Fn(ACPFromNativeRequestOptions<'_>) -> HarnessResult<Option<HarnessV1StreamPart>>
            + Send
            + Sync,
    >,
    pub to_native_response:
        Arc<dyn Fn(ACPToNativeResponseOptions<'_>) -> HarnessResult<JsonValue> + Send + Sync>,
    pub matches_native_request:
        Option<Arc<dyn Fn(ACPMatchesNativeRequestOptions<'_>) -> bool + Send + Sync>>,
}

impl fmt::Debug for ACPAskUserQuestionsSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ACPAskUserQuestionsSettings")
            .field("request_method", &self.request_method)
            .finish_non_exhaustive()
    }
}

/// `isMcpToolCall`. [rust] A TS classifier can throw; this returns the error.
pub type ACPIsMcpToolCall = Arc<dyn Fn(&ACPToolCall) -> HarnessResult<bool> + Send + Sync>;

/// Input of `resolveAuthenticationEnvironment`.
#[derive(Debug, Clone)]
pub struct ACPResolveAuthenticationEnvironmentOptions {
    pub auth: Option<ACPAuthenticationMode>,
    pub env: IndexMap<String, String>,
}

/// `resolveAuthenticationEnvironment`.
pub type ACPResolveAuthenticationEnvironment = Arc<
    dyn Fn(
            ACPResolveAuthenticationEnvironmentOptions,
        ) -> BoxFuture<'static, HarnessResult<IndexMap<String, String>>>
        + Send
        + Sync,
>;

/// Input of `prepareSandbox`.
#[derive(Clone)]
pub struct ACPPrepareSandboxOptions {
    pub sandbox: Arc<dyn SandboxSession>,
    pub work_dir: String,
    pub abort_signal: Option<CancellationToken>,
}

/// `prepareSandbox`.
pub type ACPPrepareSandbox =
    Arc<dyn Fn(ACPPrepareSandboxOptions) -> BoxFuture<'static, HarnessResult<()>> + Send + Sync>;

/// `ACPV1Settings['session']`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ACPSessionSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<IndexMap<String, ACPSerializableValue>>,
}

/// `ACPV1Settings`.
///
/// [rust] The deprecated `mintBridgeToken` is not ported.
#[derive(Clone)]
pub struct ACPV1Settings {
    /// The ACP protocol version; only `"v1"` is supported.
    pub version: Option<String>,
    pub harness_id: String,
    pub mcp_servers: Option<IndexMap<String, JsonValue>>,
    /// Classifies an ACP tool call that did not resolve to a host tool as a
    /// dynamic MCP tool call.
    pub is_mcp_tool_call: ACPIsMcpToolCall,
    /// Resolves an ACP tool call to the host tool it invokes. Host tools are
    /// exposed to the ACP agent through the harness-owned MCP server.
    pub resolve_host_tool_call: ACPResolveHostToolCall,
    pub auth: Option<ACPAuthenticationMode>,
    /// Resolves adapter-owned authentication on the host before any
    /// environment values are forwarded to the ACP agent.
    pub resolve_authentication_environment: Option<ACPResolveAuthenticationEnvironment>,
    pub source: ACPSource,
    pub executable: String,
    pub args: Option<Vec<String>>,
    pub forward_env: Option<Vec<String>>,
    pub credential_env: Option<Vec<String>>,
    pub credential_brokering: Option<ACPCredentialBrokering>,
    /// Materializes private authentication files below the ACP agent's home
    /// directory before the agent starts.
    pub authentication_files: Option<ACPAuthenticationFiles>,
    /// Prepares runtime-specific sandbox configuration before a bridge
    /// process starts, after authentication files have been written.
    /// Receives only the restricted sandbox session. Not invoked when
    /// reattaching to a live bridge. Errors reject startup without launching
    /// the bridge.
    pub prepare_sandbox: Option<ACPPrepareSandbox>,
    /// Customizes each credential value before it is forwarded into a
    /// sandbox process. This does not restrict which credentials the harness
    /// adapter can discover, read, or otherwise access in the host process.
    pub credential_forwarding: Option<HarnessV1CredentialForwarding>,
    /// Runtime environment values that are safe to persist in lifecycle
    /// compatibility identity.
    pub env: Option<IndexMap<String, String>>,
    pub authentication: Option<ACPAuthentication>,
    pub client_capabilities: Option<IndexMap<String, ACPSerializableValue>>,
    pub provider_authentication: Option<ACPProviderAuthentication>,
    /// Maps the HarnessAgent model identifier to an ACP session operation.
    pub model_mapping: ACPModelMapping,
    /// Native skills directory relative to the ACP agent's home directory.
    /// Defaults to `.agents/skills`.
    pub skills_directory: Option<String>,
    /// Routes HarnessAgent instructions to a runtime-native system or
    /// developer prompt. Changed instructions are prepended to the next user
    /// prompt when ACP does not expose a native per-turn instruction update.
    pub instruction_mapping: Option<ACPInstructionMapping>,
    /// Maps structured output JSON Schema to an agent-specific path below
    /// the ACP session prompt's `_meta` field.
    pub output_schema_mapping: Option<ACPOutputSchemaMapping>,
    /// Transport used for the harness-owned MCP server that exposes host
    /// tools to the ACP agent. Defaults to `stdio`. Set this to `http` for
    /// agents that only accept HTTP or SSE MCP servers from the client,
    /// which requires the agent to advertise
    /// `agentCapabilities.mcpCapabilities.http`.
    pub host_tool_mcp_transport: Option<ACPHostToolMCPTransport>,
    /// Non-text tool model output the ACP agent can deliver to its model.
    /// Currently accepts images; undeclared image output is omitted with a
    /// warning. Also enables image prompts when the ACP agent does not
    /// advertise image prompt support. Defaults to an empty list.
    pub non_text_content_types: Option<Vec<ACPNonTextContentType>>,
    pub ask_user_questions: Option<ACPAskUserQuestionsSettings>,
    pub permission_mode_mapping: Option<ACPPermissionModeMapping>,
    /// Paths relative to the session working directory (files or
    /// directories) that the ACP agent must not write to without host
    /// approval. Under the `allow-reads` and `allow-edits` permission modes,
    /// `edit`, `delete`, and `move` permission requests touching these paths
    /// always require approval. `.agents/skills` is always protected.
    /// `allow-all` is unaffected.
    pub write_protected_paths: Option<Vec<String>>,
    pub session: Option<ACPSessionSettings>,
}

impl ACPV1Settings {
    /// Settings with the required fields; every optional field unset.
    pub fn new(
        harness_id: impl Into<String>,
        source: ACPSource,
        executable: impl Into<String>,
        model_mapping: ACPModelMapping,
        is_mcp_tool_call: ACPIsMcpToolCall,
        resolve_host_tool_call: ACPResolveHostToolCall,
    ) -> Self {
        Self {
            version: None,
            harness_id: harness_id.into(),
            mcp_servers: None,
            is_mcp_tool_call,
            resolve_host_tool_call,
            auth: None,
            resolve_authentication_environment: None,
            source,
            executable: executable.into(),
            args: None,
            forward_env: None,
            credential_env: None,
            credential_brokering: None,
            authentication_files: None,
            prepare_sandbox: None,
            credential_forwarding: None,
            env: None,
            authentication: None,
            client_capabilities: None,
            provider_authentication: None,
            model_mapping,
            skills_directory: None,
            instruction_mapping: None,
            output_schema_mapping: None,
            host_tool_mcp_transport: None,
            non_text_content_types: None,
            ask_user_questions: None,
            permission_mode_mapping: None,
            write_protected_paths: None,
            session: None,
        }
    }
}

impl fmt::Debug for ACPV1Settings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ACPV1Settings")
            .field("harness_id", &self.harness_id)
            .field("source", &self.source)
            .field("executable", &self.executable)
            .finish_non_exhaustive()
    }
}

static HARNESS_ID_REGEXP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").unwrap());
static EXECUTABLE_NAME_REGEXP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._-]*$").unwrap());
static ENVIRONMENT_VARIABLE_NAME_REGEXP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").unwrap());

fn json(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// `validateACPV1Settings`.
///
/// [rust] `isMcpToolCall` / `resolveHostToolCall` are always functions.
pub fn validate_acp_v1_settings(settings: &ACPV1Settings) -> HarnessResult<()> {
    if settings.credential_env.is_none() != settings.credential_brokering.is_none() {
        return Err(plain_error(
            "ACP credentialEnv and credentialBrokering must be configured together.",
        ));
    }
    if settings
        .mcp_servers
        .as_ref()
        .is_some_and(|servers| servers.contains_key("ai-sdk-harness-tools"))
    {
        return Err(plain_error(
            "ACP MCP server name \"ai-sdk-harness-tools\" is reserved for HarnessAgent tools.",
        ));
    }
    if !HARNESS_ID_REGEXP.is_match(&settings.harness_id) {
        return Err(plain_error(format!(
            "ACP harnessId must be a stable kebab-case identifier; received {}.",
            json(&settings.harness_id)
        )));
    }
    for entry in settings.write_protected_paths.iter().flatten() {
        validate_write_protected_path(entry)?;
    }

    match &settings.source {
        ACPSource::NpmLocked(source) => {
            if source.package_json.is_empty() {
                return Err(plain_error("ACP source.packageJson must not be empty."));
            }
            if source.pnpm_lock_yaml.is_empty() {
                return Err(plain_error("ACP source.pnpmLockYaml must not be empty."));
            }
            if source.pnpm_workspace_yaml.as_deref() == Some("") {
                return Err(plain_error(
                    "ACP source.pnpmWorkspaceYaml must not be empty.",
                ));
            }
        }
        ACPSource::InstallCommand(source) => {
            if source.command.trim().is_empty() {
                return Err(plain_error("ACP source.command must not be empty."));
            }
        }
    }
    if !EXECUTABLE_NAME_REGEXP.is_match(&settings.executable) {
        return Err(plain_error(format!(
            "ACP executable must be a bare command name without a path; received {}.",
            json(&settings.executable)
        )));
    }

    let forwarded_keys: IndexSet<&String> = settings.forward_env.iter().flatten().collect();
    let credential_keys: IndexSet<&String> = settings.credential_env.iter().flatten().collect();
    for name in forwarded_keys.iter().chain(credential_keys.iter()) {
        validate_environment_variable_name(name)?;
    }
    for key in &credential_keys {
        if forwarded_keys.contains(key) {
            return Err(plain_error(format!(
                "ACP runtime environment key {} cannot be configured in both forwardEnv and credentialEnv.",
                json(key)
            )));
        }
    }
    for (key, value) in settings.env.iter().flatten() {
        validate_environment_variable_name(key)?;
        if value.contains('\0') {
            return Err(plain_error(format!(
                "ACP runtime environment value for {key} contains NUL."
            )));
        }
        if forwarded_keys.contains(key) {
            return Err(plain_error(format!(
                "ACP runtime environment key {} cannot be configured in both forwardEnv and env.",
                json(key)
            )));
        }
        if credential_keys.contains(key) {
            return Err(plain_error(format!(
                "ACP runtime environment key {} cannot be configured in both credentialEnv and env.",
                json(key)
            )));
        }
    }
    Ok(())
}

/// `resolveACPAgentEnvironment`: forwarded and credential values that are
/// set and non-empty, then the literal `env`.
pub fn resolve_acp_agent_environment(
    settings: &ACPV1Settings,
    env: &IndexMap<String, String>,
    credential_env: &IndexMap<String, String>,
) -> IndexMap<String, String> {
    let mut environment = IndexMap::new();
    for name in settings.forward_env.iter().flatten() {
        if let Some(value) = env.get(name).filter(|value| !value.is_empty()) {
            environment.insert(name.clone(), value.clone());
        }
    }
    for name in settings.credential_env.iter().flatten() {
        if let Some(value) = credential_env.get(name).filter(|value| !value.is_empty()) {
            environment.insert(name.clone(), value.clone());
        }
    }
    for (key, value) in settings.env.iter().flatten() {
        environment.insert(key.clone(), value.clone());
    }
    environment
}

fn validate_environment_variable_name(name: &str) -> HarnessResult<()> {
    if !ENVIRONMENT_VARIABLE_NAME_REGEXP.is_match(name) {
        return Err(plain_error(format!(
            "ACP environment variable name is invalid: {}.",
            json(name)
        )));
    }
    Ok(())
}

fn validate_write_protected_path(entry: &str) -> HarnessResult<()> {
    if entry.trim().is_empty()
        || entry.contains('\0')
        || entry.starts_with('/')
        || entry
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return Err(plain_error(format!(
            "ACP writeProtectedPaths entries must be relative paths inside the working directory without \".\" or \"..\" segments; received {}.",
            json(entry)
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
