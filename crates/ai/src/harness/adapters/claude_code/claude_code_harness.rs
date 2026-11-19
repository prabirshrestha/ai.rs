//! Port of `harness-claude-code/src/claude-code-harness.ts`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::FutureExt as _;
use futures::StreamExt as _;
use futures::future::{BoxFuture, Shared};
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as JsonValue, json};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::harness::errors::{AbortError, BoxError, HarnessError, HarnessResult};
use crate::harness::sandbox::{
    SandboxProcess, SandboxProcessControl, SandboxProcessOptions, SandboxReadTextFileOptions,
    SandboxRunOptions, SandboxSession, SandboxSpawnOptions,
};
use crate::harness::sdk::error_value::{ErrorValue, generate_id};
use crate::harness::sdk::messages::ToolResultPart;
use crate::harness::sdk::tool::{Schema, json_schema};
use crate::harness::spec::authentication::HarnessV1CredentialForwarding;
use crate::harness::spec::bootstrap::HarnessV1Bootstrap;
use crate::harness::spec::bridge_protocol::{
    HarnessV1BridgeDiagnosticFrame, HarnessV1BridgeInboundCommand, HarnessV1BridgeOutboundMessage,
    HarnessV1BridgeToolWire, harness_v1_diagnostic_from_bridge_frame,
};
use crate::harness::spec::diagnostic::HarnessV1DebugConfig;
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1ContinueTurnOptions, HarnessV1Emit, HarnessV1PromptControl,
    HarnessV1PromptTurnOptions, HarnessV1Session, HarnessV1StartOptions,
    HarnessV1SubmitToolApproval, HarnessV1SubmitToolResult,
};
use crate::harness::spec::lifecycle_state::{
    HarnessV1ContinueTurnState, HarnessV1PendingToolApprovalKind, HarnessV1ResumeSessionState,
    HarnessV1TurnSettings,
};
use crate::harness::spec::network_sandbox::{
    HarnessV1PortEndpoint, HarnessV1RequestTransformationSources,
};
use crate::harness::spec::paths::{
    harness_session_data_directory_path, harness_state_directory_path, resolve_posix,
};
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::spec::tools::{
    HarnessV1BuiltinTool, HarnessV1BuiltinToolFiltering, HarnessV1PermissionMode, HarnessV1ToolSpec,
};
use crate::harness::utils::{
    BRIDGE_AUTH_PUBLIC_KEY_ENV, BridgeHello, BridgeProcessStreamName, BridgeReadyErrorContext,
    BridgeStreamDone, CreateBridgeWebSocketConnectOptions, DiskLogRecoveryMode,
    ExperimentalBridgeUserMessageSubmitter, ExperimentalBridgeUserMessageSubmitterOptions,
    ForwardBridgeProcessStreamOptions, SandboxChannel, SandboxChannelDebugHandler,
    SandboxChannelMessage as _, SandboxChannelOptions, SandboxChannelReconnectOptions,
    SandboxChannelSendOptions, Unsubscribe, WaitForBridgeReadyOptions, WebSocketConstructor,
    WriteSkillsOptions, apply_credential_forwarding, classify_disk_log,
    create_bridge_auth_key_pair, create_bridge_error_handler, create_bridge_startup_error,
    create_bridge_web_socket_connect, drain_bridge_process_stream,
    experimental_create_bridge_user_message_submitter, extract_harness_prompt_content,
    forward_bridge_process_stream, get_restricted_sandbox_session, mark_bridge_starting,
    plain_error, resolve_bridge_endpoint, resolve_bridge_port,
    resolve_sandbox_credential_environment, resolve_sandbox_home_dir, shell_quote,
    validate_bridge_sandbox_settings, wait_for_bridge_ready, warn_credential_brokering_unavailable,
    write_skills,
};

use super::VERSION;
use super::claude_code_auth::{
    CLAUDE_CODE_CREDENTIAL_ENVIRONMENT_VARIABLES, ClaudeCodeAuthenticationMode,
    ResolveClaudeCodeEnvOptions, create_claude_code_request_transformations,
    read_api_key_helper_with, resolve_claude_code_authentication_mode, run_sh, run_system_shell,
};
use super::claude_code_bootstrap::{
    CLAUDE_CODE_BOOTSTRAP_DIR as BOOTSTRAP_DIR, get_claude_code_bootstrap,
};
use super::claude_code_bridge_protocol::{
    ClaudeCodeEffort, ClaudeCodeInboundMessage, ClaudeCodeOutboundMessage, ClaudeCodeSkillOverride,
    ClaudeCodeStartMessage, ClaudeCodeToolHint,
};
use super::claude_code_subscription::{
    ReadClaudeCodeSubscriptionOptions, ResolveClaudeCodeAuthenticationOptions,
    create_claude_code_subscription_request_transformations, read_claude_code_subscription,
    resolve_claude_code_authentication,
};
use super::claude_code_thinking::{ClaudeCodeThinkingConfig, ClaudeCodeThinkingDisplay};

type ClaudeCodeChannel = SandboxChannel<ClaudeCodeOutboundMessage, ClaudeCodeInboundMessage>;

/// `ClaudeCodeRespawnStrategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaudeCodeRespawnStrategy {
    Replay,
    Rerun,
}

/// `CLAUDE_CODE_CLIENT_APP`: value to use in User-Agent and `x-client-app`
/// headers.
fn claude_code_client_app() -> String {
    format!("ai-sdk-harness-claude-code/{VERSION}")
}

/// `ClaudeCodeHarnessSettings`.
///
/// [rust] The deprecated `mintBridgeToken` setting is not ported.
#[derive(Clone, Default)]
pub struct ClaudeCodeHarnessSettings {
    pub auth: Option<ClaudeCodeAuthenticationMode>,
    /// Requests Claude Code's classifier-based `auto` permission mode for
    /// `allow-reads` and `allow-edits`. Only escalated calls ask the host.
    /// Defaults to false; `allow-all` continues to bypass permissions.
    /// Observe raw init/status parts for the effective permission mode.
    pub auto_mode: Option<bool>,
    /// Customizes each credential value before it is forwarded into a
    /// sandbox process. This does not restrict which credentials the harness
    /// adapter can discover, read, or otherwise access in the host process.
    pub credential_forwarding: Option<HarnessV1CredentialForwarding>,
    /// MCP server definitions keyed by server name. Each definition uses the
    /// underlying runtime's native MCP server configuration format.
    pub mcp_servers: Option<IndexMap<String, JsonValue>>,
    /// Claude Code hints for host tools, keyed by tool name.
    ///
    /// - `read_only`: the tool has no side effects. Claude Code can run
    ///   read-only tool calls in parallel, and runs each call to any other
    ///   host tool on its own. It also allows read-only tools in plan mode,
    ///   so mark only tools that change nothing.
    /// - `always_load`: when Claude Code's tool search is on, it defers MCP
    ///   tools until the model looks them up; this sends the tool's
    ///   definition from the first request instead.
    pub tool_hints: Option<IndexMap<String, ClaudeCodeToolHint>>,
    /// Hard cap on how many internal turns the CLI can take before yielding
    /// back to the caller. Unset means the CLI's default.
    pub max_turns: Option<u32>,
    /// Overrides how named skills are exposed to Claude Code. Unspecified
    /// skills retain their Claude Code defaults.
    pub skill_overrides: Option<IndexMap<String, ClaudeCodeSkillOverride>>,
    /// Maximum estimated spend in USD for each Claude query. When it is
    /// exceeded, the query stops and the turn ends with an error. Unset
    /// means no limit.
    pub max_budget_usd: Option<f64>,
    /// Absolute paths of directories Claude can access in addition to the
    /// session working directory.
    pub additional_directories: Option<Vec<String>>,
    /// Only use the MCP servers configured with `mcp_servers`, and ignore
    /// any other MCP configuration such as a project `.mcp.json` or user
    /// settings.
    pub strict_mcp_config: Option<bool>,
    /// Maximum milliseconds Claude waits for an answer to an
    /// `AskUserQuestion` question before the question fails. Unset means
    /// the Claude Code default.
    pub question_timeout_ms: Option<u64>,
    /// Moves per-session sections (working directory, memory, git status)
    /// out of the Claude Code system prompt and into the first user message,
    /// so the system prompt can be cached across sessions.
    pub exclude_dynamic_sections: Option<bool>,
    /// Enables periodic AI-generated progress summaries for running
    /// subagents. The summaries are forwarded in raw `task_progress` stream
    /// parts.
    pub agent_progress_summaries: Option<bool>,
    /// Forwards subagent text and thinking messages in addition to tool
    /// activity. Subagent messages are exposed as raw stream parts.
    pub forward_subagent_text: Option<bool>,
    /// Environment variables for the Claude Code process. These values are
    /// merged over the sandbox bridge process environment.
    pub env: Option<IndexMap<String, String>>,
    /// Controls extended-thinking behavior and whether reasoning is
    /// summarized or omitted. Defaults to
    /// `{ type: 'adaptive', display: 'summarized' }`.
    pub thinking: Option<ClaudeCodeThinkingConfig>,
    /// Controls how much effort Claude applies when adaptive thinking is
    /// enabled. Unset uses the Claude Agent SDK default.
    pub effort: Option<ClaudeCodeEffort>,
    /// Override the port the bridge binds inside the sandbox. By default
    /// the adapter uses the first port the sandbox declares via
    /// `sandbox.ports`. Only set this if the sandbox declares multiple ports
    /// and the first one is reserved for something else.
    pub port: Option<u16>,
    /// Override the host endpoint used to connect to the sandbox bridge.
    /// Required together with `port` when using a basic sandbox session.
    pub port_endpoint: Option<HarnessV1PortEndpoint>,
    /// Maximum milliseconds to wait for the bridge to advertise its port.
    /// Defaults to 120000.
    pub startup_timeout_ms: Option<u64>,
    /// Maximum milliseconds to wait for the first turn event. Defaults to
    /// 120000.
    pub turn_start_timeout_ms: Option<u64>,
    /// Abort a running turn when its host disconnects and does not
    /// reconnect within this many milliseconds, so the turn stops spending.
    /// A suspended turn is exempt. Unset keeps the turn running until it
    /// finishes.
    pub host_disconnect_timeout_ms: Option<u64>,
    /// Configures reconnection attempts after an established bridge
    /// connection drops. The reconnect window includes connection
    /// establishment and backoff delays. Defaults to 30 seconds with
    /// exponential backoff from 50 milliseconds up to 2 seconds.
    pub reconnect: Option<SandboxChannelReconnectOptions>,
    /// WebSocket ping interval in milliseconds. Defaults to 20_000. Set to 0
    /// to disable. Two unanswered pings terminate the socket and trigger
    /// reconnect.
    pub keep_alive_ms: Option<u64>,
    /// Called with bridge connection lifecycle events: reconnect attempts,
    /// successful reconnects, and `reconnect-failed` once the connection is
    /// closed for good. Hosts that cache idle sessions can use it to
    /// reattach before sending the next prompt. The `cause` of a failed
    /// reconnect can contain connection details, so avoid logging it as is.
    pub on_debug: Option<SandboxChannelDebugHandler>,
}

/// `CLAUDE_CODE_BUILTIN_TOOLS`: every native tool the Claude Code CLI can
/// invoke, keyed by what the bridge emits as `toolName` on the wire
/// (`commonName ?? nativeName`).
///
/// [rust] The schemas are the `builtin-tools.json` catalog dumped from the
/// TS adapter by `scripts/refresh-harness-bridges.sh`.
pub fn claude_code_builtin_tools() -> &'static IndexMap<String, HarnessV1BuiltinTool> {
    static TOOLS: LazyLock<IndexMap<String, HarnessV1BuiltinTool>> = LazyLock::new(|| {
        crate::harness::adapters::builtin_tool_catalog::load_builtin_tool_catalog(include_str!(
            "../../../../assets/harness/claude-code/builtin-tools.json"
        ))
    });
    &TOOLS
}

/// `claudeCodeBridgeCoordsSchema`: live bridge coordinates returned by
/// `doDetach()` and `doSuspendTurn()`. A future process uses them to reopen
/// a socket to the still-running bridge (`attach`) instead of re-spawning
/// it. Absent on a `doStop()` payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaudeCodeBridgeCoords {
    pub port: u16,
    pub private_key: String,
    pub last_seen_event_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_id: Option<String>,
}

/// `claudeCodeResumeStateSchema`: the adapter-specific portion of
/// lifecycle state `data`.
///
/// A `doStop()` payload is structurally empty (`{}`): the framework derives
/// the sandbox via `provider.resumeSession({ sessionId })`, and the Claude
/// SDK's `{ continue: true }` flag rehydrates the thread from the workdir. A
/// `doDetach()` payload additionally carries `bridge` coordinates for
/// cross-process `attach`. A loose object keeps both shapes valid.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaudeCodeResumeState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge: Option<ClaudeCodeBridgeCoords>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_credential_environment: Option<IndexMap<String, String>>,
    /// The exact Claude conversation to rehydrate on resume. Written by the
    /// adapter on `doStop()`/`doDetach()`/`doSuspendTurn()` from the session
    /// id the bridge observed, so a resume names the conversation instead of
    /// relying on the SDK's `continue` flag ("most recent thread in this
    /// workdir"), which silently picks the wrong one once a second thread
    /// exists there. Hosts that captured the id themselves (it is surfaced
    /// as `harnessMetadata['claude-code'].sessionId` on `finish` parts) may
    /// also set it explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claude_session_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, JsonValue>,
}

fn claude_code_resume_state_schema() -> Schema {
    json_schema(json!({
        "type": "object",
        "properties": {
            "bridge": {
                "type": "object",
                "properties": {
                    "port": { "type": "number" },
                    "privateKey": { "type": "string" },
                    "lastSeenEventId": { "type": "number" },
                    "startId": { "type": "string" },
                    "sandboxId": { "type": "string" },
                },
                "required": ["port", "privateKey", "lastSeenEventId"],
                "additionalProperties": false,
            },
            "sandboxCredentialEnvironment": {
                "type": "object",
                "additionalProperties": { "type": "string" },
            },
            "claudeSessionId": { "type": "string" },
        },
        "additionalProperties": {},
    }))
    .with_validate(|value| {
        let _: ClaudeCodeResumeState = serde_json::from_value(value.clone())?;
        Ok(value.clone())
    })
}

/// [rust] Host seams the TS tests reach by stubbing `process.env`,
/// `os.homedir()` and the `ws` module.
#[derive(Clone, Default)]
pub(crate) struct ClaudeCodeHost {
    pub web_socket_constructor: Option<WebSocketConstructor>,
    /// Replaces `process.env`; its `HOME` replaces `os.homedir()`.
    pub process_env: Option<IndexMap<String, String>>,
}

/// The harness [`create_claude_code`] returns.
pub struct ClaudeCodeHarness {
    settings: Arc<ClaudeCodeHarnessSettings>,
    thinking: ClaudeCodeThinkingConfig,
    lifecycle_state_schema: Schema,
    host: ClaudeCodeHost,
}

/// `createClaudeCode(settings)`.
pub fn create_claude_code(settings: ClaudeCodeHarnessSettings) -> HarnessResult<ClaudeCodeHarness> {
    create_claude_code_with_host(settings, ClaudeCodeHost::default())
}

/// `claudeCode`: the default `claude-code` harness instance with no
/// overrides, suitable for the common case where the underlying `claude`
/// CLI's defaults are fine. Equivalent to `create_claude_code(Default)`.
pub fn claude_code() -> ClaudeCodeHarness {
    create_claude_code(ClaudeCodeHarnessSettings::default())
        .expect("the default Claude Code settings are valid")
}

pub(crate) fn create_claude_code_with_host(
    settings: ClaudeCodeHarnessSettings,
    host: ClaudeCodeHost,
) -> HarnessResult<ClaudeCodeHarness> {
    if settings
        .mcp_servers
        .as_ref()
        .is_some_and(|servers| servers.contains_key("harness-tools"))
    {
        return Err(plain_error(
            "Claude Code MCP server name \"harness-tools\" is reserved for HarnessAgent tools.",
        ));
    }
    let thinking = settings
        .thinking
        .unwrap_or(ClaudeCodeThinkingConfig::Adaptive {
            display: Some(ClaudeCodeThinkingDisplay::Summarized),
        });
    Ok(ClaudeCodeHarness {
        settings: Arc::new(settings),
        thinking,
        lifecycle_state_schema: claude_code_resume_state_schema(),
        host,
    })
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

fn send_options() -> SandboxChannelSendOptions {
    SandboxChannelSendOptions::default()
}

/// A `ToolResultPart` as TS sends it: with its `type: 'tool-result'` tag,
/// which the Rust struct only carries inside a message's content.
fn tool_result_part_json(tool_result: &ToolResultPart) -> Option<JsonValue> {
    let JsonValue::Object(fields) = serde_json::to_value(tool_result).ok()? else {
        return None;
    };
    let mut tagged = Map::new();
    tagged.insert("type".into(), JsonValue::String("tool-result".into()));
    tagged.extend(fields);
    Some(JsonValue::Object(tagged))
}

fn tool_wires(tools: &[HarnessV1ToolSpec]) -> Vec<HarnessV1BridgeToolWire> {
    tools
        .iter()
        .map(|tool| HarnessV1BridgeToolWire {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        })
        .collect()
}

#[async_trait]
impl HarnessV1 for ClaudeCodeHarness {
    fn harness_id(&self) -> &str {
        "claude-code"
    }

    fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool> {
        claude_code_builtin_tools()
    }

    fn supports_builtin_tool_approvals(&self) -> bool {
        true
    }

    fn supports_builtin_tool_filtering(&self) -> bool {
        true
    }

    fn lifecycle_state_schema(&self) -> Option<&Schema> {
        Some(&self.lifecycle_state_schema)
    }

    fn supports_get_bootstrap(&self) -> bool {
        true
    }

    async fn get_bootstrap(
        &self,
        _abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<HarnessV1Bootstrap> {
        Ok((*get_claude_code_bootstrap()).clone())
    }

    async fn do_start(
        &self,
        start_options: HarnessV1StartOptions,
    ) -> HarnessResult<Box<dyn HarnessV1Session>> {
        Ok(Box::new(self.start(start_options).await?))
    }
}

impl ClaudeCodeHarness {
    /// `resolveClaudeCodeAuthentication({ auth: settings.auth })`, over the
    /// host seams.
    async fn resolve_authentication(
        &self,
        process_env: &IndexMap<String, String>,
    ) -> HarnessResult<IndexMap<String, String>> {
        let mut options = ResolveClaudeCodeAuthenticationOptions {
            auth: self.settings.auth.clone(),
            process_env: Some(process_env.clone()),
            ..Default::default()
        };
        if self.host.process_env.is_some() {
            let home = process_env.get("HOME").cloned();
            options.options = ResolveClaudeCodeEnvOptions {
                read_api_key_helper: Some(Arc::new({
                    let home = home.clone();
                    move || {
                        read_api_key_helper_with(
                            std::path::Path::new(home.as_deref()?),
                            &run_sh,
                            &run_system_shell,
                        )
                    }
                })),
            };
            let env = process_env.clone();
            options.read_subscription = Some(Arc::new(move || {
                let options = ReadClaudeCodeSubscriptionOptions {
                    env: Some(env.clone()),
                    home_directory: Some(home.clone().unwrap_or_default()),
                    ..Default::default()
                };
                read_claude_code_subscription(options).boxed()
            }));
        }
        resolve_claude_code_authentication(options).await
    }

    async fn start(
        &self,
        start_options: HarnessV1StartOptions,
    ) -> HarnessResult<ClaudeCodeSession> {
        let settings = &self.settings;
        let abort_signal = start_options.abort_signal.clone();
        let sandbox_session = &start_options.sandbox_session;
        let tool_safe_sandbox_session = get_restricted_sandbox_session(sandbox_session);
        let sandbox_id = sandbox_session.network().map(|session| session.id());
        validate_bridge_sandbox_settings(
            sandbox_session,
            settings.port,
            settings.port_endpoint.as_ref(),
        )?;
        let lifecycle_state: Option<JsonValue> = start_options
            .continue_from
            .as_ref()
            .map(|state| state.data.clone())
            .or_else(|| {
                start_options
                    .resume_from
                    .as_ref()
                    .map(|state| state.data.clone())
            });
        let is_resume = lifecycle_state.is_some();
        let is_continue = start_options.continue_from.is_some();
        let resume_data: Option<ClaudeCodeResumeState> =
            lifecycle_state.map(|data| serde_json::from_value(data).unwrap_or_default());
        let coords = resume_data.as_ref().and_then(|data| data.bridge.clone());
        let process_env: IndexMap<String, String> = self
            .host
            .process_env
            .clone()
            .unwrap_or_else(|| std::env::vars().collect());
        let authentication_mode =
            resolve_claude_code_authentication_mode(settings.auth.as_ref(), &process_env);
        let resolved_auth_environment = self.resolve_authentication(&process_env).await?;
        let mut claude_environment = resolved_auth_environment.clone();
        // The Claude Agent SDK does not expose arbitrary model-request
        // headers. It reads this environment variable and sends the value
        // as `x-client-app`, while also appending `client-app/<value>` to
        // `User-Agent`, so this is the attribution path for AI Gateway.
        if resolved_auth_environment
            .get("AI_GATEWAY_BASE_URL")
            .is_some_and(|value| !value.is_empty())
        {
            claude_environment.insert(
                "CLAUDE_AGENT_SDK_CLIENT_APP".into(),
                claude_code_client_app(),
            );
        }
        if let Some(env) = &settings.env {
            claude_environment.extend(env.clone());
        }
        if let Some(headers) = &start_options.headers {
            claude_environment.insert(
                "ANTHROPIC_CUSTOM_HEADERS".into(),
                headers
                    .iter()
                    .map(|(name, value)| format!("{name}: {value}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        let credential_environment_variables: Vec<String> =
            CLAUDE_CODE_CREDENTIAL_ENVIRONMENT_VARIABLES
                .iter()
                .map(|name| name.to_string())
                .collect();
        let mut sandbox_claude_environment = claude_environment.clone();
        let mut sandbox_credential_environment: Option<IndexMap<String, String>> = None;
        if let Some(network) = sandbox_session
            .network()
            .filter(|session| session.supports_add_request_transformations())
        {
            let credential_environment = resolve_sandbox_credential_environment(
                &claude_environment,
                &credential_environment_variables,
                settings.credential_forwarding.as_ref(),
                resume_data
                    .as_ref()
                    .and_then(|data| data.sandbox_credential_environment.as_ref()),
            )
            .await?;
            sandbox_claude_environment = claude_environment.clone();
            sandbox_claude_environment.extend(credential_environment.clone());
            let transformation_sources = HarnessV1RequestTransformationSources {
                env: claude_environment.clone(),
                sandbox_env: sandbox_claude_environment.clone(),
                auth: authentication_mode,
            };
            let mut request_transformations =
                create_claude_code_request_transformations(&transformation_sources)?;
            request_transformations.extend(
                create_claude_code_subscription_request_transformations(&transformation_sources)?,
            );
            if !request_transformations.is_empty() {
                network
                    .add_request_transformations(request_transformations)
                    .await?;
            }
            sandbox_credential_environment = Some(credential_environment);
        } else {
            sandbox_claude_environment = apply_credential_forwarding(
                &sandbox_claude_environment,
                &credential_environment_variables,
                settings.credential_forwarding.as_ref(),
            )
            .await?;
            warn_credential_brokering_unavailable(
                &claude_environment,
                &sandbox_claude_environment,
                &credential_environment_variables,
            );
        }
        // Harness SDK state (bootstrap, per-session runs) always lives under
        // the sandbox's own HOME, never the working directory, so it stays
        // out of a user-owned workspace.
        let sandbox_home_dir =
            resolve_sandbox_home_dir(&*tool_safe_sandbox_session, abort_signal.clone()).await?;
        let state_dir = harness_state_directory_path(&sandbox_home_dir);
        let bootstrap_dir = resolve_posix(&state_dir, BOOTSTRAP_DIR);
        // The conversation the host wants back, when it is known. Absent on
        // state written before this field existed; those resumes fall back
        // to the `continue` flag as before.
        let resume_session_id = resume_data
            .as_ref()
            .and_then(|data| data.claude_session_id.clone())
            .filter(|id| !id.is_empty());

        let work_dir = start_options.session_work_dir.clone();
        let session_data_dir =
            harness_session_data_directory_path(&state_dir, &start_options.session_id);
        let bridge_state_dir = format!("{session_data_dir}/bridge");
        let timeout_ms = settings.startup_timeout_ms.unwrap_or(120_000);

        // Normalize each forwarded bridge diagnostics frame into the general
        // `HarnessV1Diagnostic` and report it. The adapter does no telemetry
        // work beyond this transport -> emission mapping.
        let on_diagnostic = start_options
            .observability
            .as_ref()
            .and_then(|observability| observability.report.clone())
            .map(|report| {
                let session_id = start_options.session_id.clone();
                Arc::new(move |frame: &ClaudeCodeOutboundMessage| {
                    let frame = match &frame.message {
                        HarnessV1BridgeOutboundMessage::SandboxLog(log) => {
                            HarnessV1BridgeDiagnosticFrame::SandboxLog(log)
                        }
                        HarnessV1BridgeOutboundMessage::DebugEvent(event) => {
                            HarnessV1BridgeDiagnosticFrame::DebugEvent(event)
                        }
                        _ => return,
                    };
                    report(harness_v1_diagnostic_from_bridge_frame(
                        frame,
                        Some(session_id.clone()),
                        now_ms(),
                    ));
                }) as Arc<dyn Fn(&ClaudeCodeOutboundMessage) + Send + Sync>
            });
        let on_bridge_error = {
            let handler =
                create_bridge_error_handler("claude-code", Some(start_options.session_id.clone()));
            Arc::new(move |frame: &ClaudeCodeOutboundMessage| {
                if let Some(HarnessV1StreamPart::Error { error }) = frame.stream_part() {
                    handler(&error.to_js_json());
                }
            }) as Arc<dyn Fn(&ClaudeCodeOutboundMessage) + Send + Sync>
        };
        let bridge_hello: Arc<Mutex<Option<BridgeHello>>> = Arc::default();
        let on_hello = {
            let bridge_hello = bridge_hello.clone();
            Arc::new(move |hello: &BridgeHello| {
                *bridge_hello.lock() = Some(hello.clone());
            }) as crate::harness::utils::OnBridgeHello
        };
        let new_channel = |endpoint: HarnessV1PortEndpoint,
                           private_key: String,
                           open_retry_timeout_ms: Option<u64>,
                           start_abort_signal: Option<CancellationToken>,
                           initial_last_seen_event_id: Option<u64>|
         -> HarnessResult<ClaudeCodeChannel> {
            let mut connect_options =
                CreateBridgeWebSocketConnectOptions::new(endpoint, private_key);
            connect_options.web_socket_constructor = self.host.web_socket_constructor.clone();
            connect_options.open_retry_timeout_ms = open_retry_timeout_ms;
            connect_options.start_abort_signal = start_abort_signal;
            connect_options.on_hello = Some(on_hello.clone());
            let mut channel_options =
                SandboxChannelOptions::new(create_bridge_web_socket_connect(connect_options)?);
            channel_options.initial_last_seen_event_id = initial_last_seen_event_id;
            channel_options.on_diagnostic = on_diagnostic.clone();
            channel_options.on_bridge_error = Some(on_bridge_error.clone());
            channel_options.reconnect = settings.reconnect;
            channel_options.keep_alive_ms = settings.keep_alive_ms;
            channel_options.on_debug = settings.on_debug.clone();
            Ok(SandboxChannel::new(channel_options))
        };
        let session_config = |channel: ClaudeCodeChannel,
                              proc: Option<Arc<dyn SandboxProcessControl>>,
                              bridge_port: u16,
                              bridge_private_key: String|
         -> SessionConfig {
            SessionConfig {
                session_id: start_options.session_id.clone(),
                channel,
                proc,
                settings: settings.clone(),
                env: sandbox_claude_environment.clone(),
                thinking: self.thinking,
                is_resume: false,
                deferred_tool_use_id: None,
                bridge_port,
                bridge_private_key,
                sandbox_id: sandbox_id.clone(),
                sandbox_credential_environment: sandbox_credential_environment.clone(),
                debug: start_options
                    .observability
                    .as_ref()
                    .and_then(|observability| observability.debug.clone()),
                permission_mode: start_options.permission_mode,
                builtin_tool_filtering: start_options.builtin_tool_filtering.clone(),
                sandbox: tool_safe_sandbox_session.clone(),
                sandbox_home_dir: sandbox_home_dir.clone(),
                bridge_hello: bridge_hello.clone(),
                turn_start_timeout_ms: settings.turn_start_timeout_ms.unwrap_or(120_000),
            }
        };

        // Rung 1 - ATTACH. When lifecycle state carries live bridge
        // coordinates, try to reopen a socket to the still-running bridge.
        // Parked sessions wait for the next `start`; suspended turns request
        // replay of everything past the persisted cursor. No spawn, no fresh
        // key (the existing bridge still authorises the persisted one). If
        // the bridge is gone the open fails and we fall through to a
        // spawn-based recovery.
        if let Some(coords) = &coords {
            let attached: HarnessResult<ClaudeCodeSession> = async {
                let attach_endpoint = resolve_bridge_endpoint(
                    sandbox_session,
                    coords.port,
                    settings.port_endpoint.as_ref(),
                )
                .await?;
                let attach_channel = new_channel(
                    attach_endpoint,
                    coords.private_key.clone(),
                    None,
                    None,
                    Some(coords.last_seen_event_id),
                )?;
                let finish_listener_attachment =
                    is_continue.then(|| attach_channel.begin_listener_attachment());
                if let Err(error) = attach_channel.open(is_continue).await {
                    if let Some(finish) = finish_listener_attachment {
                        finish();
                    }
                    return Err(error);
                }
                // The live bridge was spawned by another process; this one
                // owns no process handle. The session lifecycle method
                // decides whether the sandbox is left running, stopped, or
                // destroyed.
                let mut config = session_config(
                    attach_channel,
                    None,
                    coords.port,
                    coords.private_key.clone(),
                );
                config.is_resume = true;
                Ok(ClaudeCodeSession::new(
                    config,
                    SessionState {
                        stopped: false,
                        stop_promise: None,
                        pending_resume_flag: false,
                        last_claude_session_id: resume_session_id.clone(),
                        active_start_id: coords.start_id.clone(),
                        rerun_continue: false,
                        finish_listener_attachment,
                    },
                ))
            }
            .await;
            // A bridge that is no longer reachable is recovered by
            // respawning below.
            if let Ok(session) = attached {
                return Ok(session);
            }
        }

        // Rungs 2/3 - REPLAY vs RERUN. Respawn the bridge. `replay` is only
        // sound for `continueFrom`: those coordinates include the cursor the
        // on-disk log is replayed *from*. `resumeFrom` is a between-turn
        // resume; even when it carries bridge coordinates, replaying the
        // previous turn would re-deliver stale events into the next turn.
        // Those resumes always `rerun` when attach is unavailable (the CLI
        // continues its own thread from the workdir snapshot via
        // `continue: true`).
        let mut respawn_strategy = is_resume.then_some(ClaudeCodeRespawnStrategy::Rerun);
        if coords.is_some() && is_continue {
            let log_raw = tool_safe_sandbox_session
                .read_text_file(SandboxReadTextFileOptions {
                    path: format!("{bridge_state_dir}/event-log.ndjson"),
                    abort_signal: abort_signal.clone(),
                    ..Default::default()
                })
                .await
                .ok()
                .flatten();
            if classify_disk_log(log_raw.as_deref()) == DiskLogRecoveryMode::Replay {
                respawn_strategy = Some(ClaudeCodeRespawnStrategy::Replay);
            }
        }

        let port = resolve_bridge_port(sandbox_session, settings.port)?;
        let bridge_auth = create_bridge_auth_key_pair()?;
        let mut env = IndexMap::from([
            (
                BRIDGE_AUTH_PUBLIC_KEY_ENV.to_string(),
                bridge_auth.public_key.clone(),
            ),
            ("BRIDGE_WS_PORT".to_string(), port.to_string()),
        ]);
        if respawn_strategy == Some(ClaudeCodeRespawnStrategy::Replay) {
            env.insert("BRIDGE_REPLAY_FROM_DISK".into(), "1".into());
        }

        // On a fresh start the workdir and bridge-state directory must be
        // created. The env is sent fresh on every spawn: the bridge
        // authentication key pair rotates per start.
        if respawn_strategy.is_none() {
            tool_safe_sandbox_session
                .run(SandboxRunOptions {
                    process: SandboxProcessOptions {
                        command: format!(
                            "mkdir -p {} {}",
                            shell_quote(&work_dir),
                            shell_quote(&bridge_state_dir)
                        ),
                        abort_signal: abort_signal.clone(),
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .await?;
        }

        mark_bridge_starting(
            &*tool_safe_sandbox_session,
            &bridge_state_dir,
            "claude-code",
            abort_signal.clone(),
        )
        .await;

        let mut command = format!(
            "node {} --workdir {} --bridge-state-dir {}",
            shell_quote(&format!("{bootstrap_dir}/bridge.mjs")),
            shell_quote(&work_dir),
            shell_quote(&bridge_state_dir)
        );
        if let Some(timeout) = settings.host_disconnect_timeout_ms {
            command.push_str(&format!(
                " --host-disconnect-timeout-ms {}",
                shell_quote(&timeout.to_string())
            ));
        }
        let mut proc: SandboxProcess = tool_safe_sandbox_session
            .spawn(SandboxSpawnOptions {
                process: SandboxProcessOptions {
                    command,
                    env: Some(env),
                    abort_signal: abort_signal.clone(),
                    ..Default::default()
                },
                ..Default::default()
            })
            .await?;

        let stderr_tail: Arc<Mutex<Vec<String>>> = Arc::default();
        let stderr = std::mem::replace(&mut proc.stderr, futures::stream::empty().boxed());
        let bridge_stderr_done: BridgeStreamDone = {
            let handle = tokio::spawn(forward_bridge_process_stream(
                ForwardBridgeProcessStreamOptions {
                    stream: stderr,
                    stream_name: BridgeProcessStreamName::Stderr,
                    source: Some("claude-code".to_string()),
                    collect_tail: Some(stderr_tail.clone()),
                    tail_limit: None,
                },
            ));
            async move {
                let _ = handle.await;
            }
            .boxed()
            .shared()
        };
        let startup_error =
            |message: &'static str| -> crate::harness::utils::BridgeReadyErrorFactory {
                let stderr_tail = stderr_tail.clone();
                let stderr_done = bridge_stderr_done.clone();
                Box::new(move |context: BridgeReadyErrorContext| {
                    async move {
                        create_bridge_startup_error(
                            message,
                            &*context.proc,
                            &context.stdout_tail,
                            &stderr_tail,
                            Some(stderr_done),
                        )
                        .await
                    }
                    .boxed()
                })
            };
        let ready = wait_for_bridge_ready(WaitForBridgeReadyOptions {
            proc: &mut proc,
            sandbox: tool_safe_sandbox_session.clone(),
            bridge_state_dir: bridge_state_dir.clone(),
            bridge_type: "claude-code".to_string(),
            timeout_ms,
            abort_signal: abort_signal.clone(),
            poll_interval_ms: None,
            create_timeout_error: Some(startup_error(
                "claude-code bridge did not become ready in time.",
            )),
            create_exit_error: Some(startup_error(
                "claude-code bridge exited before becoming ready.",
            )),
        })
        .await?;
        let bound_port = ready.port;
        let stdout = std::mem::replace(&mut proc.stdout, futures::stream::empty().boxed());
        tokio::spawn(drain_bridge_process_stream(stdout));
        let proc_control = proc.control.clone();

        let bridge_endpoint =
            resolve_bridge_endpoint(sandbox_session, bound_port, settings.port_endpoint.as_ref())
                .await?;
        let replay = respawn_strategy == Some(ClaudeCodeRespawnStrategy::Replay);
        // In replay mode the respawned bridge reloaded the finished turn
        // from disk; seed the cursor and resume so it streams the tail
        // (incl. `finish`) rather than starting empty.
        let channel = new_channel(
            bridge_endpoint,
            bridge_auth.private_key.clone(),
            Some(timeout_ms),
            abort_signal.clone(),
            replay.then(|| {
                coords
                    .as_ref()
                    .map_or(0, |coords| coords.last_seen_event_id)
            }),
        )?;
        let finish_listener_attachment = replay.then(|| channel.begin_listener_attachment());
        if let Err(error) = channel.open(replay).await {
            if let Some(finish) = finish_listener_attachment {
                finish();
            }
            return Err(error);
        }

        let mut config = session_config(
            channel,
            Some(proc_control),
            bound_port,
            bridge_auth.private_key.clone(),
        );
        config.is_resume = respawn_strategy.is_some();
        if respawn_strategy == Some(ClaudeCodeRespawnStrategy::Rerun) {
            config.deferred_tool_use_id = start_options
                .continue_from
                .as_ref()
                .or_else(|| {
                    start_options
                        .resume_from
                        .as_ref()
                        .and_then(|state| state.continue_from.as_ref())
                })
                .and_then(|state| state.pending_tool_approvals.as_ref())
                .and_then(|approvals| {
                    approvals
                        .iter()
                        .find(|approval| approval.kind == HarnessV1PendingToolApprovalKind::Builtin)
                })
                .map(|approval| approval.tool_call_id.clone());
        }
        Ok(ClaudeCodeSession::new(
            config,
            SessionState {
                stopped: false,
                stop_promise: None,
                pending_resume_flag: respawn_strategy.is_some(),
                last_claude_session_id: resume_session_id,
                active_start_id: if replay {
                    coords.and_then(|coords| coords.start_id)
                } else {
                    None
                },
                rerun_continue: respawn_strategy == Some(ClaudeCodeRespawnStrategy::Rerun),
                finish_listener_attachment,
            },
        ))
    }
}

// ─── Session ───────────────────────────────────────────────────────────

/// The fixed inputs of `createSession`.
struct SessionConfig {
    session_id: String,
    channel: ClaudeCodeChannel,
    /// `None` on `attach`: the live bridge was spawned by another process.
    proc: Option<Arc<dyn SandboxProcessControl>>,
    settings: Arc<ClaudeCodeHarnessSettings>,
    env: IndexMap<String, String>,
    thinking: ClaudeCodeThinkingConfig,
    is_resume: bool,
    /// Built-in tool call the previous bridge deferred while it waited for
    /// host approval. The respawned bridge resumes it once the host answers
    /// instead of starting a new turn.
    deferred_tool_use_id: Option<String>,
    bridge_port: u16,
    bridge_private_key: String,
    sandbox_id: Option<String>,
    sandbox_credential_environment: Option<IndexMap<String, String>>,
    debug: Option<HarnessV1DebugConfig>,
    permission_mode: Option<HarnessV1PermissionMode>,
    builtin_tool_filtering: Option<HarnessV1BuiltinToolFiltering>,
    sandbox: Arc<dyn SandboxSession>,
    sandbox_home_dir: String,
    bridge_hello: Arc<Mutex<Option<BridgeHello>>>,
    turn_start_timeout_ms: u64,
}

impl SessionConfig {
    fn supports_user_message_responses(&self) -> bool {
        self.bridge_hello
            .lock()
            .as_ref()
            .and_then(|hello| hello.capabilities.as_ref())
            .and_then(|capabilities| capabilities.experimental_user_message_responses)
            == Some(true)
    }
}

/// The mutable state of `createSession`.
struct SessionState {
    stopped: bool,
    stop_promise: Option<Shared<BoxFuture<'static, ()>>>,
    /// Force the Claude SDK's `continue: true` on the first prompt only when
    /// the bridge was respawned without an exact conversation id
    /// (rerun/replay): a fresh bridge process treats its first turn as new,
    /// so it must be told to rehydrate the workdir thread. Exact ids are
    /// sent independently of this fallback on every fresh query.
    pending_resume_flag: bool,
    /// The Claude conversation this session currently embodies. Seeded from
    /// the resume state and updated from init raw events and `finish`
    /// metadata. Each turn may fork a new id, so the latest observation is
    /// the one a later stop/detach must record for exact resume.
    last_claude_session_id: Option<String>,
    active_start_id: Option<String>,
    rerun_continue: bool,
    finish_listener_attachment: Option<Unsubscribe>,
}

struct SessionShared {
    config: SessionConfig,
    state: Mutex<SessionState>,
}

/// The session `createSession` returns.
pub struct ClaudeCodeSession {
    shared: Arc<SessionShared>,
}

impl ClaudeCodeSession {
    fn new(config: SessionConfig, state: SessionState) -> Self {
        Self {
            shared: Arc::new(SessionShared {
                config,
                state: Mutex::new(state),
            }),
        }
    }

    /// The `start` payload fields shared by prompt and rerun turns.
    fn start_message(
        &self,
        prompt: crate::harness::utils::HarnessPromptContent,
        settings: &HarnessV1TurnSettings,
        response_format: Option<HarnessV1ResponseFormat>,
    ) -> ClaudeCodeStartMessage {
        let config = &self.shared.config;
        let harness_settings = &config.settings;
        let mut start = ClaudeCodeStartMessage::new(prompt, config.thinking);
        start.base.tools = Some(tool_wires(&settings.tools));
        start.base.response_format = response_format;
        start.instructions = settings
            .instructions
            .clone()
            .filter(|instructions| !instructions.is_empty());
        start.base.model = settings.model.clone();
        start.max_turns = harness_settings.max_turns;
        start.skill_overrides = harness_settings.skill_overrides.clone();
        start.max_budget_usd = harness_settings.max_budget_usd;
        start.additional_directories = harness_settings.additional_directories.clone();
        start.strict_mcp_config = harness_settings.strict_mcp_config;
        start.auto_mode = harness_settings.auto_mode;
        start.question_timeout_ms = harness_settings.question_timeout_ms;
        start.exclude_dynamic_sections = harness_settings.exclude_dynamic_sections;
        start.agent_progress_summaries = harness_settings.agent_progress_summaries;
        start.forward_subagent_text = harness_settings.forward_subagent_text;
        start.env = Some(config.env.clone());
        start.effort = harness_settings.effort;
        if !settings.skills.is_empty() {
            start.skills = Some(
                settings
                    .skills
                    .iter()
                    .map(|skill| skill.name.clone())
                    .collect(),
            );
            start.readable_skill_roots = Some(
                settings
                    .skills
                    .iter()
                    .map(|skill| {
                        format!("{}/.claude/skills/{}", config.sandbox_home_dir, skill.name)
                    })
                    .collect(),
            );
        }
        start.mcp_servers = harness_settings.mcp_servers.clone();
        start.tool_hints = harness_settings.tool_hints.clone();
        start.base.permission_mode = config.permission_mode;
        start.base.builtin_tool_filtering = config.builtin_tool_filtering.clone();
        start.base.debug = config.debug.clone();
        start
    }

    /// `prepareTurn`.
    async fn prepare_turn(
        &self,
        response_format: Option<&HarnessV1ResponseFormat>,
        settings: &HarnessV1TurnSettings,
        emit: HarnessV1Emit,
        abort_signal: Option<CancellationToken>,
        start_id: Option<String>,
    ) -> HarnessResult<Arc<Turn>> {
        if let Some(HarnessV1ResponseFormat::Json { schema: None, .. }) = response_format {
            return Err(Box::new(HarnessError::capability_unsupported(
                "Harness 'claude-code' requires a JSON schema for structured output.",
                Some("claude-code".into()),
            )));
        }
        let config = &self.shared.config;
        let mut options = WriteSkillsOptions::new(
            &*config.sandbox,
            config.sandbox_home_dir.clone(),
            ".claude/skills",
            settings.skills.clone(),
        );
        options.abort_signal = abort_signal.clone();
        options.invalid_skill_name_message = Some(Arc::new(|name: &str| {
            format!("Invalid Claude Code skill name: {name}")
        }));
        options.invalid_skill_file_path_message =
            Some(Arc::new(|skill_name: &str, file_path: &str| {
                format!("Invalid Claude Code skill file path for {skill_name}: {file_path}")
            }));
        options.trailing_newline = true;
        write_skills(options).await?;
        Ok(Turn::wire(&self.shared, emit, abort_signal, start_id))
    }

    fn resume_state_data(&self, last_seen_event_id: u64) -> JsonValue {
        let config = &self.shared.config;
        let state = self.shared.state.lock();
        let mut data = Map::new();
        if let Some(environment) = &config.sandbox_credential_environment {
            data.insert(
                "sandboxCredentialEnvironment".into(),
                serde_json::to_value(environment).unwrap_or_default(),
            );
        }
        let coords = ClaudeCodeBridgeCoords {
            port: config.bridge_port,
            private_key: config.bridge_private_key.clone(),
            last_seen_event_id,
            start_id: state.active_start_id.clone(),
            sandbox_id: config.sandbox_id.clone(),
        };
        data.insert(
            "bridge".into(),
            serde_json::to_value(coords).unwrap_or_default(),
        );
        if let Some(session_id) = state
            .last_claude_session_id
            .as_ref()
            .filter(|id| !id.is_empty())
        {
            data.insert("claudeSessionId".into(), session_id.clone().into());
        }
        JsonValue::Object(data)
    }

    fn mark_stopped(&self, message: impl FnOnce(&str) -> String) -> HarnessResult<()> {
        let mut state = self.shared.state.lock();
        if state.stopped {
            return Err(plain_error(message(&self.shared.config.session_id)));
        }
        state.stopped = true;
        Ok(())
    }

    /// Waits up to five seconds for the bridge process, then reaps it and
    /// closes the channel.
    async fn reap(config: &SessionConfig) {
        if let Some(proc) = &config.proc {
            let _ = tokio::time::timeout(Duration::from_millis(5000), proc.wait()).await;
            let _ = proc.kill().await;
        }
        config.channel.close();
    }
}

#[async_trait]
impl HarnessV1Session for ClaudeCodeSession {
    fn session_id(&self) -> String {
        self.shared.config.session_id.clone()
    }

    fn is_resume(&self) -> bool {
        self.shared.config.is_resume
    }

    async fn do_prompt_turn(
        &self,
        options: HarnessV1PromptTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        let turn = self
            .prepare_turn(
                options.response_format.as_ref(),
                &options.settings,
                options.emit.clone(),
                options.abort_signal.clone(),
                Some(generate_id()),
            )
            .await?;

        // A signal that was already aborted has settled the turn inside
        // `wireTurn`. Sending `start` anyway would run a full unattended
        // turn in the bridge (consuming tokens and possibly acting on the
        // workspace) after the caller has already observed the abort.
        if options
            .abort_signal
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Ok(turn.control());
        }

        let prompt = extract_harness_prompt_content(&options.prompt)?;
        let mut start =
            self.start_message(prompt, &options.settings, options.response_format.clone());
        {
            let mut state = self.shared.state.lock();
            match state
                .last_claude_session_id
                .clone()
                .filter(|id| !id.is_empty())
            {
                Some(session_id) => start.resume_session_id = Some(session_id),
                None if state.pending_resume_flag => start.r#continue = Some(true),
                None => {}
            }
            state.pending_resume_flag = false;
            // Recovery starts a native turn once. Later continuations attach
            // to it.
            state.rerun_continue = false;
        }
        turn.send_start(start)?;
        Ok(turn.control())
    }

    async fn do_continue_turn(
        &self,
        options: HarnessV1ContinueTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        let (rerun_continue, start_id) = {
            let state = self.shared.state.lock();
            (
                state.rerun_continue,
                if state.rerun_continue {
                    Some(generate_id())
                } else {
                    state.active_start_id.clone()
                },
            )
        };
        let turn = self
            .prepare_turn(
                options.response_format.as_ref(),
                &options.settings,
                options.emit.clone(),
                options.abort_signal.clone(),
                start_id,
            )
            .await?;

        // attach / replay: the still-running (or disk-replayed) turn streams
        // into the wired listeners: `doStart` opened the channel with
        // `{ resume: true }` so the bridge replays everything past the
        // persisted cursor (including a `finish` if the turn ended during
        // the gap). No `start` is sent: issuing one would clear the bridge's
        // replay log and begin a new turn. Lossless.
        //
        // rerun: the bridge was respawned with no in-flight turn to attach
        // to, so re-drive the runtime's own thread from the workdir snapshot
        // via `continue: true`. Lossy: work in flight at the suspension is
        // recomputed. This is the rare bridge-died fallback; the common
        // slice path is `attach`.
        //
        // Same as `doPromptTurn`: an already-aborted signal settled the turn
        // in `wireTurn`, so don't re-drive the thread nobody is waiting on.
        if rerun_continue
            && !options
                .abort_signal
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
        {
            // A continuation nudge rather than an empty prompt:
            // `continue: true` rehydrates the prior thread, and this is the
            // new user turn that drives it forward. It must be non-empty: an
            // empty text block is rejected by the Anthropic API once the SDK
            // stamps it with `cache_control`.
            let mut start = self.start_message(
                crate::harness::utils::HarnessPromptContent::Text("Continue.".into()),
                &options.settings,
                options.response_format.clone(),
            );
            {
                let mut state = self.shared.state.lock();
                state.rerun_continue = false;
                state.pending_resume_flag = false;
                match state
                    .last_claude_session_id
                    .clone()
                    .filter(|id| !id.is_empty())
                {
                    Some(session_id) => start.resume_session_id = Some(session_id),
                    None => start.r#continue = Some(true),
                }
            }
            start.deferred_tool_use_id = self.shared.config.deferred_tool_use_id.clone();
            turn.send_start(start)?;
        }
        Ok(turn.control())
    }

    async fn do_compact(&self, custom_instructions: Option<String>) -> HarnessResult<()> {
        // Claude Code has no SDK/control method for compaction: the
        // supported trigger is the `/compact` slash command submitted as
        // user input. Ride the existing user-message rail; the bridge feeds
        // it into the streaming query input and Claude's native compaction
        // handles the rest, emitting a `compact_boundary` + `PostCompact` we
        // observe as a `compaction` event.
        let text = match custom_instructions
            .as_deref()
            .map(str::trim)
            .filter(|instructions| !instructions.is_empty())
        {
            Some(instructions) => format!("/compact {instructions}"),
            None => "/compact".into(),
        };
        self.shared.config.channel.send(
            &HarnessV1BridgeInboundCommand::UserMessage {
                message_id: None,
                text,
            }
            .into(),
            send_options(),
        )
    }

    async fn do_detach(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        self.mark_stopped(|session_id| {
            format!("claude-code session {session_id} is already stopped; cannot detach.")
        })?;
        let last_seen_event_id = self.shared.config.channel.suspend().await;
        Ok(HarnessV1ResumeSessionState::new(
            "claude-code",
            self.resume_state_data(last_seen_event_id),
        ))
    }

    async fn do_destroy(&self) -> HarnessResult<()> {
        let stop_promise = {
            let mut state = self.shared.state.lock();
            if state.stopped {
                state.stop_promise.clone()
            } else {
                state.stopped = true;
                let shared = self.shared.clone();
                let promise = async move {
                    let config = &shared.config;
                    // Tell the channel we are tearing down so the bridge's
                    // post-shutdown socket close finalises instead of
                    // triggering a reconnect.
                    config.channel.begin_close();
                    if !config.channel.is_closed() {
                        let _ = config.channel.send(
                            &HarnessV1BridgeInboundCommand::Destroy.into(),
                            send_options(),
                        );
                    }
                    Self::reap(config).await;
                }
                .boxed()
                .shared();
                state.stop_promise = Some(promise.clone());
                Some(promise)
            }
        };
        if let Some(stop_promise) = stop_promise {
            stop_promise.await;
        }
        Ok(())
    }

    async fn do_stop(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        self.mark_stopped(|session_id| {
            format!("claude-code session {session_id} is already stopped; cannot stop.")
        })?;
        let config = &self.shared.config;
        let channel = &config.channel;
        // If the bridge's channel already closed (e.g. mid-turn WS drop)
        // there is no one to acknowledge a `stop` message. Synthesize an
        // empty payload: for Claude Code the resume state structurally is
        // `{}` (the conversation lives in the workdir, captured by the
        // sandbox snapshot during the subsequent `sandboxSession.stop()`),
        // so we lose nothing by skipping the round-trip.
        //
        // Tell the channel we are tearing down so the bridge's post-stop
        // socket close finalises instead of triggering a reconnect.
        channel.begin_close();
        let data: JsonValue = if channel.is_closed() {
            json!({})
        } else {
            let (data_tx, data_rx) = oneshot::channel::<JsonValue>();
            let data_tx = Mutex::new(Some(data_tx));
            let unsubscribe =
                channel.on("bridge-stop", move |frame: &ClaudeCodeOutboundMessage| {
                    if let HarnessV1BridgeOutboundMessage::Stop(stop) = &frame.message
                        && let Some(data_tx) = data_tx.lock().take()
                    {
                        let _ = data_tx.send(stop.data.clone());
                    }
                });
            if let Err(error) =
                channel.send(&HarnessV1BridgeInboundCommand::Stop.into(), send_options())
            {
                unsubscribe();
                return Err(error);
            }
            let result = tokio::time::timeout(Duration::from_millis(5000), data_rx).await;
            unsubscribe();
            match result {
                Ok(Ok(data)) => data,
                _ => {
                    return Err(plain_error(format!(
                        "claude-code session {} did not reply to stop within 5s.",
                        config.session_id
                    )));
                }
            }
        };

        // The bridge exits itself after sending bridge-stop. Give it a
        // moment, then ensure the process is reaped and the channel closed.
        Self::reap(config).await;

        let lifecycle_data = match data {
            JsonValue::Object(data) => data,
            _ => Map::new(),
        };
        // The bridge's stop reply carries the session id it observed; the
        // adapter's own record backfills it when the reply predates the
        // field or the channel was already closed.
        let mut payload = Map::new();
        if let Some(session_id) = self
            .shared
            .state
            .lock()
            .last_claude_session_id
            .clone()
            .filter(|id| !id.is_empty())
        {
            payload.insert("claudeSessionId".into(), session_id.into());
        }
        payload.extend(lifecycle_data);
        if let Some(environment) = &config.sandbox_credential_environment {
            payload.insert(
                "sandboxCredentialEnvironment".into(),
                serde_json::to_value(environment)?,
            );
        }
        Ok(HarnessV1ResumeSessionState::new(
            "claude-code",
            JsonValue::Object(payload),
        ))
    }

    async fn do_suspend_turn(&self) -> HarnessResult<HarnessV1ContinueTurnState> {
        self.mark_stopped(|session_id| {
            format!("claude-code session {session_id} is stopped; cannot suspend.")
        })?;
        // Freeze the host at a precise cursor without stopping the active
        // model turn. `channel.suspend` stops processing inbound frames,
        // drains what was already dispatched, then closes the host socket
        // with reason `'suspended'`. The bridge keeps the turn running and
        // accumulates events past the cursor for the next slice to replay.
        // The sandbox process is deliberately left alive.
        let last_seen_event_id = self.shared.config.channel.suspend().await;
        Ok(HarnessV1ContinueTurnState::new(
            "claude-code",
            self.resume_state_data(last_seen_event_id),
        ))
    }
}

// ─── Turn ──────────────────────────────────────────────────────────────

#[derive(Default)]
struct TurnState {
    settled: bool,
    expected_start_id: Option<String>,
    pending_start: Option<ClaudeCodeStartMessage>,
    start_cursor: u64,
    start_timer: Option<tokio::task::JoinHandle<()>>,
    subscriptions: Vec<Unsubscribe>,
    abort_task: Option<tokio::task::JoinHandle<()>>,
}

/// One `wireTurn` call: the channel wired into one turn's worth of events.
/// Shared by `doPromptTurn` (which sends a `start` afterwards) and
/// `doContinueTurn` (which attaches to an already-running/replayed turn,
/// or sends a rerun `start`). The only difference between the two entry
/// points is the `start` message, not the listener/abort/settle plumbing.
struct Turn {
    session: Arc<SessionShared>,
    emit: HarnessV1Emit,
    state: Mutex<TurnState>,
    awaiting_start_send: Arc<AtomicBool>,
    user_message_submitter: Mutex<Option<ExperimentalBridgeUserMessageSubmitter>>,
    done_tx: Mutex<Option<oneshot::Sender<Result<(), ErrorValue>>>>,
    done: Shared<BoxFuture<'static, Result<(), ErrorValue>>>,
}

impl Turn {
    fn channel(&self) -> &ClaudeCodeChannel {
        &self.session.config.channel
    }

    fn wire(
        session: &Arc<SessionShared>,
        emit: HarnessV1Emit,
        abort_signal: Option<CancellationToken>,
        start_id: Option<String>,
    ) -> Arc<Self> {
        session.state.lock().active_start_id = start_id.clone();
        let (done_tx, done_rx) = oneshot::channel::<Result<(), ErrorValue>>();
        let done = async move {
            done_rx
                .await
                .unwrap_or_else(|_| Err(ErrorValue::message("claude-code turn was abandoned.")))
        }
        .boxed()
        .shared();
        let turn = Arc::new(Turn {
            session: session.clone(),
            emit,
            state: Mutex::new(TurnState {
                expected_start_id: start_id,
                ..Default::default()
            }),
            awaiting_start_send: Arc::default(),
            user_message_submitter: Mutex::default(),
            done_tx: Mutex::new(Some(done_tx)),
            done,
        });
        let channel = turn.channel().clone();
        let mut subscriptions = Vec::new();

        subscriptions.push(channel.on_reconnect({
            let weak = Arc::downgrade(&turn);
            move || {
                if let Some(turn) = weak.upgrade() {
                    turn.on_reconnect();
                }
            }
        }));

        // Retry the turn start before the submitter replays pending
        // steering.
        if session.config.supports_user_message_responses() {
            let send_channel = channel.clone();
            let response_channel = channel.clone();
            let reconnect_channel = channel.clone();
            *turn.user_message_submitter.lock() =
                Some(experimental_create_bridge_user_message_submitter(
                    ExperimentalBridgeUserMessageSubmitterOptions {
                        // The submitter owns replay, after the turn start has
                        // been retried.
                        send: Arc::new(move |message| {
                            send_channel.send(
                                &HarnessV1BridgeInboundCommand::UserMessage {
                                    message_id: Some(message.message_id.clone()),
                                    text: message.text.clone(),
                                }
                                .into(),
                                SandboxChannelSendOptions {
                                    on_sent: None,
                                    queue_if_disconnected: false,
                                },
                            )
                        }),
                        on_response: Box::new(move |listener| {
                            response_channel.on(
                                "user-message-response",
                                move |frame: &ClaudeCodeOutboundMessage| {
                                    if let HarnessV1BridgeOutboundMessage::UserMessageResponse(
                                        response,
                                    ) = &frame.message
                                    {
                                        listener(response);
                                    }
                                },
                            )
                        }),
                        on_reconnect: Box::new(move |listener| {
                            reconnect_channel.on_reconnect(move || listener())
                        }),
                    },
                ));
        }

        for event_type in [
            "stream-start",
            "text-start",
            "text-delta",
            "text-end",
            "reasoning-start",
            "reasoning-delta",
            "reasoning-end",
            "tool-input-start",
            "tool-input-delta",
            "tool-input-end",
            "tool-call",
            "tool-approval-request",
            "tool-result",
            "finish-step",
            "compaction",
        ] {
            let weak = Arc::downgrade(&turn);
            subscriptions.push(
                channel.on(event_type, move |frame: &ClaudeCodeOutboundMessage| {
                    if let Some(turn) = weak.upgrade()
                        && turn.belongs_to_turn(frame)
                    {
                        turn.forward(frame);
                    }
                }),
            );
        }
        subscriptions.push(channel.on("raw", {
            let weak = Arc::downgrade(&turn);
            move |frame: &ClaudeCodeOutboundMessage| {
                let Some(turn) = weak.upgrade() else { return };
                if !turn.belongs_to_turn(frame) {
                    return;
                }
                if let Some(HarnessV1StreamPart::Raw {
                    raw_value: JsonValue::Object(raw),
                }) = frame.stream_part()
                    && raw.get("type").and_then(JsonValue::as_str) == Some("system")
                    && raw.get("subtype").and_then(JsonValue::as_str) == Some("init")
                    && let Some(session_id) = raw
                        .get("sessionId")
                        .and_then(JsonValue::as_str)
                        .filter(|id| !id.is_empty())
                {
                    turn.session.state.lock().last_claude_session_id = Some(session_id.into());
                }
                turn.forward(frame);
            }
        }));
        subscriptions.push(channel.on("finish", {
            let weak = Arc::downgrade(&turn);
            move |frame: &ClaudeCodeOutboundMessage| {
                let Some(turn) = weak.upgrade() else { return };
                if !turn.belongs_to_turn(frame) {
                    return;
                }
                if let Some(HarnessV1StreamPart::Finish {
                    harness_metadata: Some(metadata),
                    ..
                }) = frame.stream_part()
                    && let Some(session_id) = metadata
                        .get("claude-code")
                        .and_then(|metadata| metadata.get("sessionId"))
                        .and_then(JsonValue::as_str)
                        .filter(|id| !id.is_empty())
                {
                    turn.session.state.lock().last_claude_session_id = Some(session_id.into());
                }
                turn.forward(frame);
                turn.settle(None);
            }
        }));
        subscriptions.push(channel.on("error", {
            let weak = Arc::downgrade(&turn);
            move |frame: &ClaudeCodeOutboundMessage| {
                let Some(turn) = weak.upgrade() else { return };
                if !turn.belongs_to_turn(frame) {
                    return;
                }
                turn.forward(frame);
                if let Some(HarnessV1StreamPart::Error { error }) = frame.stream_part() {
                    turn.settle(Some(error.clone()));
                }
            }
        }));
        turn.state.lock().subscriptions = subscriptions;
        let finish_listener_attachment = session.state.lock().finish_listener_attachment.take();
        if let Some(finish) = finish_listener_attachment {
            finish();
        }

        // A `'suspended'` close is a graceful slice-boundary freeze the host
        // initiated (`doSuspendTurn`): the turn keeps running in the bridge
        // and its tail is replayed to the next process, so wind this turn
        // down cleanly rather than failing it. Any other close mid-turn is
        // an unexpected drop.
        channel.on_close({
            let weak = Arc::downgrade(&turn);
            move |_code, reason| {
                let Some(turn) = weak.upgrade() else { return };
                if reason == "suspended" {
                    turn.settle(None);
                    return;
                }
                turn.settle(Some(ErrorValue::message(
                    "claude-code bridge closed before the turn finished.",
                )));
            }
        });

        if let Some(abort_signal) = &abort_signal {
            if abort_signal.is_cancelled() {
                turn.on_abort();
            } else {
                let token = abort_signal.clone();
                let weak = Arc::downgrade(&turn);
                let task = tokio::spawn(async move {
                    token.cancelled().await;
                    if let Some(turn) = weak.upgrade() {
                        turn.on_abort();
                    }
                });
                turn.state.lock().abort_task = Some(task);
            }
        }
        turn
    }

    fn control(self: &Arc<Self>) -> Box<dyn HarnessV1PromptControl> {
        Box::new(ClaudeCodePromptControl {
            turn: self.clone(),
            supports_user_messages: self.user_message_submitter.lock().is_some(),
        })
    }

    fn belongs_to_turn(&self, frame: &ClaudeCodeOutboundMessage) -> bool {
        let mut state = self.state.lock();
        if state.settled {
            return false;
        }
        let Some(start_id) = &frame.start_id else {
            return true;
        };
        if state.expected_start_id.is_none() {
            state.expected_start_id = Some(start_id.clone());
            self.session.state.lock().active_start_id = Some(start_id.clone());
        }
        state.expected_start_id.as_ref() == Some(start_id)
    }

    fn clear_pending_start(state: &mut TurnState) {
        state.pending_start = None;
        if let Some(timer) = state.start_timer.take() {
            timer.abort();
        }
    }

    /// Forwards a frame's stream part. A `finish-step` keeps the frame's
    /// replay checkpoint, which must reach the consumer for step-boundary
    /// suspension; only the wire-level turn identifier is removed.
    fn forward(&self, frame: &ClaudeCodeOutboundMessage) {
        Self::clear_pending_start(&mut self.state.lock());
        let Some(part) = frame.stream_part() else {
            return;
        };
        let mut part = part.clone();
        if let HarnessV1StreamPart::FinishStep { checkpoint, .. } = &mut part {
            *checkpoint = frame.event_checkpoint().cloned();
        }
        (self.emit)(part);
    }

    fn settle(&self, error: Option<ErrorValue>) {
        let (subscriptions, abort_task) = {
            let mut state = self.state.lock();
            if state.settled {
                return;
            }
            state.settled = true;
            Self::clear_pending_start(&mut state);
            (
                std::mem::take(&mut state.subscriptions),
                state.abort_task.take(),
            )
        };
        if let Some(submitter) = self.user_message_submitter.lock().as_ref() {
            submitter.close(error.as_ref().map(ErrorValue::to_box));
        }
        for unsubscribe in subscriptions {
            unsubscribe();
        }
        if let Some(task) = abort_task {
            task.abort();
        }
        if let Some(done_tx) = self.done_tx.lock().take() {
            let _ = done_tx.send(match error {
                None => Ok(()),
                Some(error) => Err(error),
            });
        }
    }

    fn send_pending_start(&self) -> HarnessResult<()> {
        let Some(pending_start) = self.state.lock().pending_start.clone() else {
            return Ok(());
        };
        self.awaiting_start_send.store(true, Ordering::SeqCst);
        let awaiting = self.awaiting_start_send.clone();
        let result = self.channel().send(
            &pending_start.into(),
            SandboxChannelSendOptions {
                on_sent: Some(Box::new(move || awaiting.store(false, Ordering::SeqCst))),
                queue_if_disconnected: true,
            },
        );
        if result.is_err() {
            self.awaiting_start_send.store(false, Ordering::SeqCst);
        }
        result
    }

    fn fail_start(&self, error: ErrorValue) {
        self.settle(Some(error));
        let channel = self.channel();
        channel.begin_close();
        let _ = channel.send(&HarnessV1BridgeInboundCommand::Abort.into(), send_options());
        channel.close();
    }

    fn on_reconnect(&self) {
        {
            let state = self.state.lock();
            if state.pending_start.is_none()
                || state.settled
                || self.awaiting_start_send.load(Ordering::SeqCst)
            {
                return;
            }
        }
        let hello = self.session.config.bridge_hello.lock().clone();
        if hello
            .as_ref()
            .and_then(|hello| hello.capabilities.as_ref())
            .and_then(|capabilities| capabilities.idempotent_start)
            == Some(Some(true))
        {
            let _ = self.send_pending_start();
            return;
        }
        let start_cursor = self.state.lock().start_cursor;
        let Some(hello) = hello else { return };
        if hello.state.as_deref() != Some("waiting")
            || hello
                .last_seq
                .is_none_or(|last_seq| last_seq > start_cursor)
        {
            return;
        }
        self.fail_start(ErrorValue::message(
            "claude-code bridge is waiting for a start after reconnect and does not support safe start retries.",
        ));
    }

    fn on_abort(&self) {
        if self.state.lock().settled {
            return;
        }
        let _ = self
            .channel()
            .send(&HarnessV1BridgeInboundCommand::Abort.into(), send_options());
        self.settle(Some(ErrorValue::from_box(AbortError.into())));
    }

    /// `sendStart(message)`.
    fn send_start(self: &Arc<Self>, mut message: ClaudeCodeStartMessage) -> HarnessResult<()> {
        let timeout_ms = self.session.config.turn_start_timeout_ms;
        {
            let mut state = self.state.lock();
            if state.settled {
                return Ok(());
            }
            let hello_last_seq = self
                .session
                .config
                .bridge_hello
                .lock()
                .as_ref()
                .and_then(|hello| hello.last_seq)
                .unwrap_or(0);
            state.start_cursor = self.channel().last_seen_event_id().max(hello_last_seq);
            message.base.start_id = state.expected_start_id.clone();
            state.pending_start = Some(message);
            let weak = Arc::downgrade(self);
            state.start_timer = Some(tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
                let Some(turn) = weak.upgrade() else { return };
                {
                    let state = turn.state.lock();
                    if state.settled || state.pending_start.is_none() {
                        return;
                    }
                }
                turn.fail_start(ErrorValue::message(format!(
                    "claude-code bridge did not emit a turn event within {timeout_ms}ms."
                )));
            }));
        }
        if let Err(error) = self.send_pending_start() {
            // The caller never receives control when the initial send
            // fails. Remove the timer and turn listeners.
            let error = ErrorValue::from_box(error);
            self.settle(Some(error.clone()));
            return Err(error.to_box());
        }
        Ok(())
    }
}

struct ClaudeCodePromptControl {
    turn: Arc<Turn>,
    supports_user_messages: bool,
}

#[async_trait]
impl HarnessV1PromptControl for ClaudeCodePromptControl {
    async fn submit_tool_result(&self, input: HarnessV1SubmitToolResult) -> HarnessResult<()> {
        self.turn.channel().send(
            &HarnessV1BridgeInboundCommand::ToolResult {
                tool_call_id: input.tool_call_id,
                output: input.output,
                is_error: input.is_error,
                tool_result: input.tool_result.as_ref().and_then(tool_result_part_json),
            }
            .into(),
            send_options(),
        )
    }

    fn supports_submit_tool_approval(&self) -> bool {
        true
    }

    async fn submit_tool_approval(&self, input: HarnessV1SubmitToolApproval) -> HarnessResult<()> {
        self.turn.channel().send(
            &HarnessV1BridgeInboundCommand::ToolApprovalResponse {
                approval_id: input.approval_id,
                approved: input.approved,
                reason: input.reason,
            }
            .into(),
            send_options(),
        )
    }

    fn supports_submit_user_message(&self) -> bool {
        self.supports_user_messages
    }

    async fn submit_user_message(&self, text: String) -> HarnessResult<()> {
        let submitter = self.turn.user_message_submitter.lock().clone();
        match submitter {
            Some(submitter) => submitter.submit(text).await,
            None => Err(Box::new(HarnessError::capability_unsupported(
                "This prompt control does not support submitUserMessage.",
                None,
            ))),
        }
    }

    fn done(&self) -> BoxFuture<'static, HarnessResult<()>> {
        let done = self.turn.done.clone();
        async move { done.await.map_err(|error| error.to_box()) }.boxed()
    }
}

#[allow(dead_code)]
fn _assert_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<ClaudeCodeSession>();
    check::<ClaudeCodeHarness>();
    let _: Option<BoxError> = None;
}
