//! Port of `harness-codex/src/codex-harness.ts`.

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
use crate::harness::sdk::error_value::ErrorValue;
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
    HarnessV1ContinueTurnState, HarnessV1ResumeSessionState, HarnessV1TurnSettings,
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
    BRIDGE_AUTH_PUBLIC_KEY_ENV, BridgeProcessStreamName, BridgeReadyErrorContext, BridgeStreamDone,
    CreateBridgeWebSocketConnectOptions, DiskLogRecoveryMode,
    ExperimentalBridgeUserMessageSubmitter, ExperimentalBridgeUserMessageSubmitterOptions,
    ForwardBridgeProcessStreamOptions, HarnessPromptContent, SandboxChannel,
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
use super::codex_auth::{
    CODEX_CREDENTIAL_ENVIRONMENT_VARIABLES, CodexAuthenticationMode,
    CodexResolvedAuthenticationMode, DEFAULT_OPENAI_BASE_URL, resolve_codex_authentication_mode,
};
use super::codex_bootstrap::{CODEX_BOOTSTRAP_DIR as BOOTSTRAP_DIR, get_codex_bootstrap};
use super::codex_bridge_protocol::{
    CodexInboundMessage, CodexOutboundMessage, CodexReasoningEffort, CodexStartMessage,
};
use super::codex_subscription::{
    CodexResolvedAuthentication, ReadCodexSubscriptionOptions, ResolveCodexAuthenticationOptions,
    create_codex_subscription_request_transformations, read_codex_subscription,
    resolve_codex_authentication,
};

type CodexChannel = SandboxChannel<CodexOutboundMessage, CodexInboundMessage>;

/// `CodexRespawnStrategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodexRespawnStrategy {
    Replay,
    Rerun,
}

/// `DEFAULT_CODEX_MODEL`.
///
/// This intentionally is not the latest Codex model. Newer GPT-5.6 models
/// use Responses Lite, which does not expose their code-mode tools as
/// callable through the custom model provider used by the harness. Keep
/// GPT-5.5 as the default until the upstream bug is resolved:
/// <https://github.com/openai/codex/issues/31894>
///
/// Passing the model explicitly keeps the runtime behavior deterministic and
/// the telemetry (`gen_ai.request.model`) accurate.
const DEFAULT_CODEX_MODEL: &str = "gpt-5.5";

/// `CODEX_CLIENT_APP`: value to use in User-Agent and `x-client-app`
/// headers.
fn codex_client_app() -> String {
    format!("ai-sdk-harness-codex/{VERSION}")
}

/// `CodexHarnessSettings`.
///
/// [rust] The deprecated `mintBridgeToken` setting is not ported.
#[derive(Clone, Default)]
pub struct CodexHarnessSettings {
    pub auth: Option<CodexAuthenticationMode>,
    /// Customizes each credential value before it is forwarded into a
    /// sandbox process. This does not restrict which credentials the harness
    /// adapter can discover, read, or otherwise access in the host process.
    pub credential_forwarding: Option<HarnessV1CredentialForwarding>,
    /// Additional configuration passed through to Codex as-is. Codex config
    /// keys typically use snake_case and must be provided in that form.
    /// Values managed by this adapter take precedence over conflicting
    /// entries, except for the `model_reasoning_summary` default, which
    /// callers can override.
    pub codex_config: Option<IndexMap<String, JsonValue>>,
    /// MCP server definitions keyed by server name. Each definition uses the
    /// underlying runtime's native MCP server configuration format.
    pub mcp_servers: Option<IndexMap<String, JsonValue>>,
    /// Reasoning effort for reasoning-capable models. Leaving this unset
    /// defers to the CLI's default.
    pub reasoning_effort: Option<CodexReasoningEffort>,
    /// When `true`, allow the underlying runtime to use live web search.
    pub web_search: Option<bool>,
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
}

/// `CODEX_BUILTIN_TOOLS`: every native tool Codex app-server can invoke as
/// a model-callable tool, keyed by what the bridge emits as `toolName` on
/// the wire (`commonName ?? nativeName`). Schemas reflect the corresponding
/// app-server item payloads.
///
/// Other native operations such as todo planning surface as side-effect
/// events and are not model-callable tools.
///
/// [rust] The schemas are the `builtin-tools.json` catalog dumped from the
/// TS adapter by `scripts/refresh-harness-bridges.sh`.
pub fn codex_builtin_tools() -> &'static IndexMap<String, HarnessV1BuiltinTool> {
    static TOOLS: LazyLock<IndexMap<String, HarnessV1BuiltinTool>> = LazyLock::new(|| {
        crate::harness::adapters::builtin_tool_catalog::load_builtin_tool_catalog(include_str!(
            "../../../../assets/harness/codex/builtin-tools.json"
        ))
    });
    &TOOLS
}

/// `codexBridgeCoordsSchema`: live bridge coordinates returned by
/// `doDetach()` and `doSuspendTurn()`. A future process uses them to reopen
/// a socket to the still-running bridge (`attach`) instead of re-spawning
/// it. Absent on a `doStop()` payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexBridgeCoords {
    pub port: u16,
    pub private_key: String,
    pub last_seen_event_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_id: Option<String>,
}

/// `codexResumeStateSchema`: the adapter-specific lifecycle `data` payload
/// Codex produces. `threadId` is what app-server's `thread/resume` method
/// requires for the replay/rerun rungs; the sandbox lookup is handled
/// separately via `provider.resumeSession({ sessionId })`. `bridge` carries
/// live coordinates for cross-process `attach` (present on `doDetach()` and
/// `doSuspendTurn()` payloads).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexResumeState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_configuration_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge: Option<CodexBridgeCoords>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_credential_environment: Option<IndexMap<String, String>>,
}

fn codex_resume_state_schema() -> Schema {
    json_schema(json!({
        "type": "object",
        "properties": {
            "threadId": { "type": "string" },
            "turnConfigurationFingerprint": { "type": "string" },
            "bridge": {
                "type": "object",
                "properties": {
                    "port": { "type": "number" },
                    "privateKey": { "type": "string" },
                    "lastSeenEventId": { "type": "number" },
                    "sandboxId": { "type": "string" },
                },
                "required": ["port", "privateKey", "lastSeenEventId"],
                "additionalProperties": false,
            },
            "sandboxCredentialEnvironment": {
                "type": "object",
                "additionalProperties": { "type": "string" },
            },
        },
        "additionalProperties": false,
    }))
    .with_validate(|value| {
        let state: CodexResumeState = serde_json::from_value(value.clone())?;
        Ok(serde_json::to_value(state)?)
    })
}

/// [rust] Host seams the TS tests reach by stubbing `process.env`,
/// `os.homedir()` and the `ws` module.
#[derive(Clone, Default)]
pub(crate) struct CodexHost {
    pub web_socket_constructor: Option<WebSocketConstructor>,
    /// Replaces `process.env`; its `HOME` replaces `os.homedir()`.
    pub process_env: Option<IndexMap<String, String>>,
}

/// The harness [`create_codex`] returns.
pub struct CodexHarness {
    settings: Arc<CodexHarnessSettings>,
    lifecycle_state_schema: Schema,
    host: CodexHost,
}

/// `createCodex(settings)`.
pub fn create_codex(settings: CodexHarnessSettings) -> CodexHarness {
    create_codex_with_host(settings, CodexHost::default())
}

/// `codex`: the default `codex` harness instance with no overrides.
/// Equivalent to `create_codex(Default)`.
pub fn codex() -> CodexHarness {
    create_codex(CodexHarnessSettings::default())
}

pub(crate) fn create_codex_with_host(
    settings: CodexHarnessSettings,
    host: CodexHost,
) -> CodexHarness {
    CodexHarness {
        settings: Arc::new(settings),
        lifecycle_state_schema: codex_resume_state_schema(),
        host,
    }
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
impl HarnessV1 for CodexHarness {
    fn harness_id(&self) -> &str {
        "codex"
    }

    fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool> {
        codex_builtin_tools()
    }

    fn supports_builtin_tool_approvals(&self) -> bool {
        false
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
        Ok((*get_codex_bootstrap()).clone())
    }

    async fn do_start(
        &self,
        start_options: HarnessV1StartOptions,
    ) -> HarnessResult<Box<dyn HarnessV1Session>> {
        Ok(Box::new(self.start(start_options).await?))
    }
}

/// The lifecycle `data` fields `doStart` reads.
#[derive(Default)]
struct ResumeData {
    thread_id: Option<String>,
    turn_configuration_fingerprint: Option<String>,
    bridge: Option<CodexBridgeCoords>,
    sandbox_credential_environment: Option<IndexMap<String, String>>,
}

impl ResumeData {
    /// `typeof data === 'object'`, read field by field as TS does.
    fn read(data: &JsonValue) -> Self {
        let string = |key: &str| data.get(key).and_then(JsonValue::as_str).map(String::from);
        Self {
            thread_id: string("threadId").filter(|id| !id.is_empty()),
            turn_configuration_fingerprint: string("turnConfigurationFingerprint"),
            bridge: data
                .get("bridge")
                .and_then(|bridge| serde_json::from_value(bridge.clone()).ok()),
            sandbox_credential_environment: data
                .get("sandboxCredentialEnvironment")
                .and_then(|environment| serde_json::from_value(environment.clone()).ok()),
        }
    }
}

impl CodexHarness {
    /// `resolveCodexAuthentication({ auth, authCredentialsStoreMode })`,
    /// over the host seams.
    async fn resolve_authentication(
        &self,
        process_env: &IndexMap<String, String>,
    ) -> HarnessResult<CodexResolvedAuthentication> {
        let mut options = ResolveCodexAuthenticationOptions {
            auth: self.settings.auth.clone(),
            process_env: Some(process_env.clone()),
            auth_credentials_store_mode: self
                .settings
                .codex_config
                .as_ref()
                .and_then(|config| config.get("cli_auth_credentials_store").cloned()),
            read_subscription: None,
        };
        if self.host.process_env.is_some() {
            let env = process_env.clone();
            options.read_subscription = Some(Arc::new(move |auth_credentials_store_mode| {
                read_codex_subscription(ReadCodexSubscriptionOptions {
                    home_directory: Some(env.get("HOME").cloned().unwrap_or_default()),
                    env: Some(env.clone()),
                    auth_credentials_store_mode,
                    ..Default::default()
                })
                .boxed()
            }));
        }
        resolve_codex_authentication(options).await
    }

    async fn start(&self, start_options: HarnessV1StartOptions) -> HarnessResult<CodexSession> {
        let settings = &self.settings;
        let model = DEFAULT_CODEX_MODEL.to_string();
        if start_options
            .permission_mode
            .is_some_and(|mode| mode != HarnessV1PermissionMode::AllowAll)
        {
            return Err(Box::new(HarnessError::capability_unsupported(
                "Harness 'codex' does not support built-in tool approval requests; use permissionMode: 'allow-all'.",
                Some("codex".into()),
            )));
        }
        let abort_signal = start_options.abort_signal.clone();
        let sandbox_session = &start_options.sandbox_session;
        let tool_safe_sandbox_session = get_restricted_sandbox_session(sandbox_session);
        let sandbox_id = sandbox_session.network().map(|session| session.id());
        validate_bridge_sandbox_settings(
            sandbox_session,
            settings.port,
            settings.port_endpoint.as_ref(),
        )?;
        let lifecycle_state: Option<&JsonValue> = start_options
            .continue_from
            .as_ref()
            .map(|state| &state.data)
            .or_else(|| start_options.resume_from.as_ref().map(|state| &state.data));
        let is_resume = lifecycle_state.is_some();
        let is_continue = start_options.continue_from.is_some();
        let resume_data = lifecycle_state
            .filter(|data| data.is_object())
            .map(ResumeData::read)
            .unwrap_or_default();
        let resume_thread_id = resume_data.thread_id.clone();
        let turn_configuration_fingerprint = resume_data.turn_configuration_fingerprint.clone();
        let coords = resume_data.bridge.clone();
        let process_env: IndexMap<String, String> = self
            .host
            .process_env
            .clone()
            .unwrap_or_else(|| std::env::vars().collect());
        let authentication_mode =
            resolve_codex_authentication_mode(settings.auth.as_ref(), &process_env);
        let resolved_authentication = self.resolve_authentication(&process_env).await?;
        let resolved_auth_environment = resolved_authentication.environment.clone();
        let credential_environment_variables: Vec<String> = CODEX_CREDENTIAL_ENVIRONMENT_VARIABLES
            .iter()
            .map(|name| name.to_string())
            .collect();
        let mut sandbox_auth_environment = resolved_auth_environment.clone();
        let mut sandbox_credential_environment: Option<IndexMap<String, String>> = None;
        let mut credentials_brokered = false;
        if let Some(network) = sandbox_session
            .network()
            .filter(|session| session.supports_add_request_transformations())
        {
            let credential_environment = resolve_sandbox_credential_environment(
                &resolved_auth_environment,
                &credential_environment_variables,
                settings.credential_forwarding.as_ref(),
                resume_data.sandbox_credential_environment.as_ref(),
            )
            .await?;
            sandbox_auth_environment = resolved_auth_environment.clone();
            sandbox_auth_environment.extend(credential_environment.clone());
            let request_transformations = create_codex_subscription_request_transformations(
                &HarnessV1RequestTransformationSources {
                    env: resolved_auth_environment.clone(),
                    sandbox_env: sandbox_auth_environment.clone(),
                    auth: authentication_mode,
                },
                resolved_authentication.request_headers.as_ref(),
            )?;
            let has_request_transformations = !request_transformations.is_empty();
            if has_request_transformations {
                network
                    .add_request_transformations(request_transformations)
                    .await?;
            }
            sandbox_credential_environment = Some(credential_environment);
            credentials_brokered = true;
            if has_request_transformations
                && authentication_mode == CodexResolvedAuthenticationMode::Direct
                && !resolved_auth_environment.contains_key("OPENAI_BASE_URL")
            {
                // Vercel Sandbox request transformations apply only to HTTP
                // traffic. Materializing Codex's standard OpenAI URL makes
                // the bridge select its custom provider, where WebSockets
                // are disabled, while keeping the non-brokered path on
                // Codex's built-in OpenAI provider.
                sandbox_auth_environment
                    .insert("OPENAI_BASE_URL".into(), DEFAULT_OPENAI_BASE_URL.into());
            }
        }
        // Harness SDK state (bootstrap, per-session runs) always lives under
        // the sandbox's own HOME, never the working directory, so it stays
        // out of a user-owned workspace.
        let sandbox_home_dir =
            resolve_sandbox_home_dir(&*tool_safe_sandbox_session, abort_signal.clone()).await?;
        let state_dir = harness_state_directory_path(&sandbox_home_dir);
        let bootstrap_dir = resolve_posix(&state_dir, BOOTSTRAP_DIR);

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
                Arc::new(move |frame: &CodexOutboundMessage| {
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
                }) as Arc<dyn Fn(&CodexOutboundMessage) + Send + Sync>
            });
        let on_bridge_error = {
            let handler =
                create_bridge_error_handler("codex", Some(start_options.session_id.clone()));
            Arc::new(move |frame: &CodexOutboundMessage| {
                if let Some(HarnessV1StreamPart::Error { error }) = frame.stream_part() {
                    handler(&error.to_js_json());
                }
            }) as Arc<dyn Fn(&CodexOutboundMessage) + Send + Sync>
        };
        let new_channel = |endpoint: HarnessV1PortEndpoint,
                           private_key: String,
                           open_retry_timeout_ms: Option<u64>,
                           start_abort_signal: Option<CancellationToken>,
                           initial_last_seen_event_id: Option<u64>|
         -> HarnessResult<CodexChannel> {
            let mut connect_options =
                CreateBridgeWebSocketConnectOptions::new(endpoint, private_key);
            connect_options.web_socket_constructor = self.host.web_socket_constructor.clone();
            connect_options.open_retry_timeout_ms = open_retry_timeout_ms;
            connect_options.start_abort_signal = start_abort_signal;
            let mut channel_options =
                SandboxChannelOptions::new(create_bridge_web_socket_connect(connect_options)?);
            channel_options.initial_last_seen_event_id = initial_last_seen_event_id;
            channel_options.on_diagnostic = on_diagnostic.clone();
            channel_options.on_bridge_error = Some(on_bridge_error.clone());
            channel_options.reconnect = settings.reconnect;
            channel_options.keep_alive_ms = settings.keep_alive_ms;
            Ok(SandboxChannel::new(channel_options))
        };
        let session_config = |channel: CodexChannel,
                              proc: Option<Arc<dyn SandboxProcessControl>>,
                              is_resume: bool,
                              bridge_port: u16,
                              bridge_private_key: String|
         -> SessionConfig {
            SessionConfig {
                session_id: start_options.session_id.clone(),
                channel,
                proc,
                model: model.clone(),
                reasoning_effort: settings.reasoning_effort,
                web_search: settings.web_search,
                builtin_tool_filtering: start_options.builtin_tool_filtering.clone(),
                codex_config: settings.codex_config.clone(),
                mcp_servers: settings.mcp_servers.clone(),
                headers: start_options.headers.clone(),
                is_resume,
                bridge_port,
                bridge_private_key,
                sandbox_id: sandbox_id.clone(),
                sandbox_credential_environment: sandbox_credential_environment.clone(),
                debug: start_options
                    .observability
                    .as_ref()
                    .and_then(|observability| observability.debug.clone()),
                permission_mode: start_options.permission_mode,
                sandbox: tool_safe_sandbox_session.clone(),
                sandbox_home_dir: sandbox_home_dir.clone(),
            }
        };

        // Rung 1 - ATTACH. With live coordinates, reopen a socket to the
        // still-running bridge. Parked between-turn sessions just attach and
        // wait for the next `start`; suspended in-flight turns request replay
        // of everything past the persisted cursor. No spawn, no fresh key.
        // If the bridge is gone the open fails and we fall through to a
        // spawn-based recovery.
        if let Some(coords) = &coords {
            let attached: HarnessResult<CodexSession> = async {
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
                attach_channel.open(is_continue).await?;
                // The live bridge was spawned by another process; no process
                // handle.
                Ok(CodexSession::new(
                    session_config(
                        attach_channel,
                        None,
                        true,
                        coords.port,
                        coords.private_key.clone(),
                    ),
                    SessionState {
                        stopped: false,
                        stop_promise: None,
                        pending_resume_thread_id: None,
                        latest_thread_id: resume_thread_id.clone(),
                        latest_turn_configuration_fingerprint: turn_configuration_fingerprint
                            .clone(),
                        rerun_continue: false,
                    },
                ))
            }
            .await;
            // Bridge no longer reachable: recover by respawning below.
            if let Ok(session) = attached {
                return Ok(session);
            }
        }

        // Rungs 2/3 - REPLAY vs RERUN. Respawn the bridge. `replay` is only
        // sound for `continueFrom`: those coordinates include the cursor the
        // on-disk log is replayed *from*. `resumeFrom` is a between-turn
        // resume; even when it carries bridge coordinates, replaying the
        // previous turn would re-deliver stale events into the next turn.
        // Those resumes always `rerun` via app-server's `thread/resume`
        // method when attach is unavailable.
        let mut respawn_strategy = is_resume.then_some(CodexRespawnStrategy::Rerun);
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
                respawn_strategy = Some(CodexRespawnStrategy::Replay);
            }
        }

        let port = resolve_bridge_port(sandbox_session, settings.port)?;
        let bridge_auth = create_bridge_auth_key_pair()?;
        let forwarded_auth_environment = if credentials_brokered {
            sandbox_auth_environment.clone()
        } else {
            apply_credential_forwarding(
                &sandbox_auth_environment,
                &credential_environment_variables,
                settings.credential_forwarding.as_ref(),
            )
            .await?
        };
        if !credentials_brokered {
            warn_credential_brokering_unavailable(
                &resolved_auth_environment,
                &forwarded_auth_environment,
                &credential_environment_variables,
            );
        }
        let mut env = forwarded_auth_environment;
        env.insert("AI_SDK_HARNESS_CLIENT_APP".into(), codex_client_app());
        env.insert(
            BRIDGE_AUTH_PUBLIC_KEY_ENV.to_string(),
            bridge_auth.public_key.clone(),
        );
        env.insert("BRIDGE_WS_PORT".into(), port.to_string());
        if respawn_strategy == Some(CodexRespawnStrategy::Replay) {
            env.insert("BRIDGE_REPLAY_FROM_DISK".into(), "1".into());
        }

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
            "codex",
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
                    source: Some("codex".to_string()),
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
            bridge_type: "codex".to_string(),
            timeout_ms,
            abort_signal: abort_signal.clone(),
            poll_interval_ms: None,
            create_timeout_error: Some(startup_error("codex bridge did not become ready in time.")),
            create_exit_error: Some(startup_error("codex bridge exited before becoming ready.")),
        })
        .await?;
        let bound_port = ready.port;
        let stdout = std::mem::replace(&mut proc.stdout, futures::stream::empty().boxed());
        tokio::spawn(drain_bridge_process_stream(stdout));
        let proc_control = proc.control.clone();

        let bridge_endpoint =
            resolve_bridge_endpoint(sandbox_session, bound_port, settings.port_endpoint.as_ref())
                .await?;
        let replay = respawn_strategy == Some(CodexRespawnStrategy::Replay);
        // In replay mode the respawned bridge reloaded the finished turn
        // from disk; seed the cursor and resume so it streams the tail
        // (incl. `finish`).
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
        channel.open(replay).await?;

        let respawned = respawn_strategy.is_some();
        Ok(CodexSession::new(
            session_config(
                channel,
                Some(proc_control),
                respawned,
                bound_port,
                bridge_auth.private_key.clone(),
            ),
            SessionState {
                stopped: false,
                stop_promise: None,
                // Send the persisted threadId on the first prompt only when
                // the bridge was respawned (rerun/replay) so it takes the
                // app-server `thread/resume` branch. An `attach`ed bridge
                // already holds its threadState in memory and continues on
                // its own, so it needs no seed.
                pending_resume_thread_id: if respawned {
                    resume_thread_id.clone()
                } else {
                    None
                },
                latest_thread_id: resume_thread_id,
                latest_turn_configuration_fingerprint: turn_configuration_fingerprint,
                rerun_continue: respawn_strategy == Some(CodexRespawnStrategy::Rerun),
            },
        ))
    }
}

// ─── Session ───────────────────────────────────────────────────────────

/// The fixed inputs of `createSession`.
struct SessionConfig {
    session_id: String,
    channel: CodexChannel,
    /// `None` on `attach`: the live bridge was spawned by another process.
    proc: Option<Arc<dyn SandboxProcessControl>>,
    model: String,
    reasoning_effort: Option<CodexReasoningEffort>,
    web_search: Option<bool>,
    builtin_tool_filtering: Option<HarnessV1BuiltinToolFiltering>,
    codex_config: Option<IndexMap<String, JsonValue>>,
    mcp_servers: Option<IndexMap<String, JsonValue>>,
    headers: Option<IndexMap<String, String>>,
    is_resume: bool,
    bridge_port: u16,
    bridge_private_key: String,
    sandbox_id: Option<String>,
    sandbox_credential_environment: Option<IndexMap<String, String>>,
    debug: Option<HarnessV1DebugConfig>,
    permission_mode: Option<HarnessV1PermissionMode>,
    sandbox: Arc<dyn SandboxSession>,
    sandbox_home_dir: String,
}

/// The mutable state of `createSession`.
struct SessionState {
    stopped: bool,
    stop_promise: Option<Shared<BoxFuture<'static, ()>>>,
    pending_resume_thread_id: Option<String>,
    /// Latest codex thread id, cached from the bridge's `bridge-thread`
    /// announcements. Seeded from lifecycle state so `doDetach()` and
    /// `doStop()` can include a thread id even before this process has run
    /// a turn.
    latest_thread_id: Option<String>,
    latest_turn_configuration_fingerprint: Option<String>,
    rerun_continue: bool,
}

struct SessionShared {
    config: SessionConfig,
    state: Mutex<SessionState>,
}

/// The session `createSession` returns.
pub struct CodexSession {
    shared: Arc<SessionShared>,
}

impl CodexSession {
    fn new(config: SessionConfig, state: SessionState) -> Self {
        let shared = Arc::new(SessionShared {
            config,
            state: Mutex::new(state),
        });
        let weak = Arc::downgrade(&shared);
        let _ = shared
            .config
            .channel
            .on("bridge-thread", move |frame: &CodexOutboundMessage| {
                if let HarnessV1BridgeOutboundMessage::Thread(thread) = &frame.message
                    && let Some(shared) = weak.upgrade()
                {
                    shared.state.lock().latest_thread_id = Some(thread.thread_id.clone());
                }
            });
        Self { shared }
    }

    /// `synchronizeTurnConfiguration({ skills, instructions, tools })`.
    async fn synchronize_turn_configuration(
        &self,
        settings: &HarnessV1TurnSettings,
        tools: &[HarnessV1BridgeToolWire],
        abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<bool> {
        let config = &self.shared.config;
        let mut options = WriteSkillsOptions::new(
            &*config.sandbox,
            config.sandbox_home_dir.clone(),
            ".agents/skills",
            settings.skills.clone(),
        );
        options.abort_signal = abort_signal;
        options.invalid_skill_name_message = Some(Arc::new(|name: &str| {
            format!("Invalid Codex skill name: {name}")
        }));
        options.invalid_skill_file_path_message =
            Some(Arc::new(|skill_name: &str, file_path: &str| {
                format!("Invalid Codex skill file path for {skill_name}: {file_path}")
            }));
        let skills_result = write_skills(options).await?;
        let next_fingerprint =
            fingerprint_codex_turn_configuration(settings.instructions.as_deref(), tools);
        let mut state = self.shared.state.lock();
        // Legacy fingerprints used order-sensitive JSON, so replace them
        // once instead of restarting a thread during the fingerprint format
        // upgrade.
        let restart_thread = state.latest_thread_id.is_some()
            && (skills_result.changed
                || state
                    .latest_turn_configuration_fingerprint
                    .as_ref()
                    .is_some_and(|previous| {
                        previous.starts_with("v2:") && *previous != next_fingerprint
                    }));
        state.latest_turn_configuration_fingerprint = Some(next_fingerprint);
        if restart_thread {
            state.pending_resume_thread_id = None;
        }
        Ok(restart_thread)
    }

    /// The `start` payload fields shared by prompt and rerun turns.
    fn start_message(
        &self,
        prompt: HarnessPromptContent,
        tools: Vec<HarnessV1BridgeToolWire>,
        settings: &HarnessV1TurnSettings,
        response_format: Option<HarnessV1ResponseFormat>,
        resume_thread_id: Option<String>,
        restart_thread: bool,
    ) -> CodexStartMessage {
        let config = &self.shared.config;
        let mut start = CodexStartMessage::new(prompt);
        start.base.tools = Some(tools);
        start.base.response_format = response_format;
        start.instructions = settings
            .instructions
            .clone()
            .filter(|instructions| !instructions.is_empty());
        start.base.model = Some(
            settings
                .model
                .clone()
                .unwrap_or_else(|| config.model.clone()),
        );
        start.reasoning_effort = config.reasoning_effort;
        start.web_search = config.web_search;
        start.base.builtin_tool_filtering = config.builtin_tool_filtering.clone();
        start.codex_config = config.codex_config.clone();
        start.mcp_servers = config.mcp_servers.clone();
        start.headers = config.headers.clone();
        start.base.permission_mode = config.permission_mode;
        start.resume_thread_id = resume_thread_id.filter(|id| !id.is_empty());
        start.restart_thread = restart_thread.then_some(true);
        start.base.debug = config.debug.clone();
        start
    }

    fn require_json_schema(response_format: Option<&HarnessV1ResponseFormat>) -> HarnessResult<()> {
        if let Some(HarnessV1ResponseFormat::Json { schema: None, .. }) = response_format {
            return Err(Box::new(HarnessError::capability_unsupported(
                "Harness 'codex' requires a JSON schema for structured output.",
                Some("codex".into()),
            )));
        }
        Ok(())
    }

    fn resume_state_data(&self, last_seen_event_id: Option<u64>) -> Map<String, JsonValue> {
        let config = &self.shared.config;
        let state = self.shared.state.lock();
        let mut data = Map::new();
        if let Some(thread_id) = state.latest_thread_id.as_ref().filter(|id| !id.is_empty()) {
            data.insert("threadId".into(), thread_id.clone().into());
        }
        if let Some(fingerprint) = state
            .latest_turn_configuration_fingerprint
            .as_ref()
            .filter(|fingerprint| !fingerprint.is_empty())
        {
            data.insert(
                "turnConfigurationFingerprint".into(),
                fingerprint.clone().into(),
            );
        }
        if let Some(environment) = &config.sandbox_credential_environment {
            data.insert(
                "sandboxCredentialEnvironment".into(),
                serde_json::to_value(environment).unwrap_or_default(),
            );
        }
        if let Some(last_seen_event_id) = last_seen_event_id {
            let coords = CodexBridgeCoords {
                port: config.bridge_port,
                private_key: config.bridge_private_key.clone(),
                last_seen_event_id,
                sandbox_id: config.sandbox_id.clone(),
            };
            data.insert(
                "bridge".into(),
                serde_json::to_value(coords).unwrap_or_default(),
            );
        }
        data
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
impl HarnessV1Session for CodexSession {
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
        Self::require_json_schema(options.response_format.as_ref())?;
        let tools = tool_wires(&options.settings.tools);
        let restart_thread = self
            .synchronize_turn_configuration(&options.settings, &tools, options.abort_signal.clone())
            .await?;
        let turn = Turn::wire(
            &self.shared,
            options.emit.clone(),
            options.abort_signal.clone(),
        );

        let prompt = extract_harness_prompt_content(&options.prompt)?;
        let pending_resume_thread_id = {
            let mut state = self.shared.state.lock();
            // Recovery starts a native turn once. Later continuations attach
            // to it.
            state.rerun_continue = false;
            state.pending_resume_thread_id.take()
        };
        let start = self.start_message(
            prompt,
            tools,
            &options.settings,
            options.response_format.clone(),
            pending_resume_thread_id,
            restart_thread,
        );
        turn.send_start(start);
        Ok(turn.control())
    }

    async fn do_continue_turn(
        &self,
        options: HarnessV1ContinueTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        Self::require_json_schema(options.response_format.as_ref())?;
        let tools = tool_wires(&options.settings.tools);
        let restart_thread = self
            .synchronize_turn_configuration(&options.settings, &tools, options.abort_signal.clone())
            .await?;
        let turn = Turn::wire(
            &self.shared,
            options.emit.clone(),
            options.abort_signal.clone(),
        );

        // attach / replay: the still-running (or disk-replayed) turn streams
        // into the wired listeners: `doStart` opened the channel with
        // `{ resume: true }` so the bridge replays everything past the
        // persisted cursor (including a `finish` if the turn ended during
        // the gap). No `start` is sent: issuing one would clear the bridge's
        // replay log and begin a new turn. Lossless.
        //
        // rerun: the bridge was respawned with no in-flight turn to attach
        // to, so re-drive codex's own thread via `resumeThreadId`. Lossy:
        // work in flight at the suspension is recomputed. This is the rare
        // bridge-died fallback; the common slice path is `attach`.
        let rerun_thread_id = {
            let mut state = self.shared.state.lock();
            if state.rerun_continue {
                state.rerun_continue = false;
                let thread_id = state
                    .pending_resume_thread_id
                    .take()
                    .or_else(|| state.latest_thread_id.clone());
                Some(thread_id)
            } else {
                None
            }
        };
        if let Some(thread_id) = rerun_thread_id {
            // A continuation nudge rather than an empty prompt:
            // `resumeThreadId` rehydrates the prior thread, and this is the
            // new user turn that drives it forward. Keeping it non-empty
            // avoids handing the runtime an empty user message (and mirrors
            // the claude-code adapter, where an empty text block trips the
            // Anthropic API's `cache_control` rule).
            let start = self.start_message(
                HarnessPromptContent::Text("Continue.".into()),
                tools,
                &options.settings,
                options.response_format.clone(),
                thread_id,
                restart_thread,
            );
            turn.send_start(start);
        }
        Ok(turn.control())
    }

    async fn do_compact(&self, _custom_instructions: Option<String>) -> HarnessResult<()> {
        // Codex compacts its context automatically inside the core turn loop
        // (~90% of the model context window). This adapter does not yet
        // expose a manual compaction trigger or compaction event, so manual
        // `compact()` is unsupported while Codex's own auto-compaction
        // continues to run.
        Err(Box::new(HarnessError::capability_unsupported(
            "Harness 'codex' does not support manual compaction; Codex auto-compacts its context internally.",
            Some("codex".into()),
        )))
    }

    async fn do_detach(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        self.mark_stopped(|session_id| {
            format!("codex session {session_id} is already stopped; cannot detach.")
        })?;
        let last_seen_event_id = self.shared.config.channel.suspend().await;
        Ok(HarnessV1ResumeSessionState::new(
            "codex",
            JsonValue::Object(self.resume_state_data(Some(last_seen_event_id))),
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
            format!("codex session {session_id} is already stopped; cannot stop.")
        })?;
        let config = &self.shared.config;
        let channel = &config.channel;
        // If the bridge's channel already closed (e.g. mid-turn WS drop)
        // there is no one to acknowledge a `stop` message. Synthesize an
        // empty payload: the workdir is still captured by the sandbox
        // snapshot during the subsequent `sandboxSession.stop()`, so the
        // next turn can resume the filesystem state. The trade-off: we lose
        // `threadId`, so Codex app-server starts a fresh thread on the
        // preserved workdir rather than resuming the prior conversation
        // inside Codex's runtime. Ability to continue beats throwing.
        //
        // Tell the channel we are tearing down so the bridge's post-stop
        // socket close finalises instead of triggering a reconnect.
        channel.begin_close();
        let data: JsonValue = if channel.is_closed() {
            json!({})
        } else {
            let (data_tx, data_rx) = oneshot::channel::<JsonValue>();
            let data_tx = Mutex::new(Some(data_tx));
            let unsubscribe = channel.on("bridge-stop", move |frame: &CodexOutboundMessage| {
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
                        "codex session {} did not reply to stop within 5s.",
                        config.session_id
                    )));
                }
            }
        };

        Self::reap(config).await;

        let mut payload = match data {
            JsonValue::Object(data) => data,
            _ => Map::new(),
        };
        {
            let state = self.shared.state.lock();
            if let Some(fingerprint) = state
                .latest_turn_configuration_fingerprint
                .as_ref()
                .filter(|fingerprint| !fingerprint.is_empty())
            {
                payload.insert(
                    "turnConfigurationFingerprint".into(),
                    fingerprint.clone().into(),
                );
            }
        }
        if let Some(environment) = &config.sandbox_credential_environment {
            payload.insert(
                "sandboxCredentialEnvironment".into(),
                serde_json::to_value(environment)?,
            );
        }
        Ok(HarnessV1ResumeSessionState::new(
            "codex",
            JsonValue::Object(payload),
        ))
    }

    async fn do_suspend_turn(&self) -> HarnessResult<HarnessV1ContinueTurnState> {
        self.mark_stopped(|session_id| {
            format!("codex session {session_id} is stopped; cannot suspend.")
        })?;
        // Freeze the host at a precise cursor without stopping the active
        // model turn. `channel.suspend` stops processing inbound frames,
        // drains what was already dispatched, then closes the host socket
        // with reason `'suspended'`. The bridge keeps the turn running and
        // accumulates events past the cursor for the next slice to replay.
        // The sandbox process is deliberately left alive.
        let last_seen_event_id = self.shared.config.channel.suspend().await;
        Ok(HarnessV1ContinueTurnState::new(
            "codex",
            JsonValue::Object(self.resume_state_data(Some(last_seen_event_id))),
        ))
    }
}

/// `fingerprintCodexTurnConfiguration({ instructions, tools })`: `v2:` plus
/// the SHA-256 of the configuration as JSON with every object's keys
/// sorted.
fn fingerprint_codex_turn_configuration(
    instructions: Option<&str>,
    tools: &[HarnessV1BridgeToolWire],
) -> String {
    let canonical = sort_keys(&json!({
        "instructions": instructions,
        "tools": tools,
    }))
    .to_string();
    format!(
        "v2:{}",
        crate::harness::agent::internal::sha256_hex(&canonical)
    )
}

/// Every object's keys sorted by UTF-16 code unit, as the TS replacer does.
fn sort_keys(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Array(items) => JsonValue::Array(items.iter().map(sort_keys).collect()),
        JsonValue::Object(record) => {
            let mut entries: Vec<(&String, &JsonValue)> = record.iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.encode_utf16().cmp(right.encode_utf16()));
            JsonValue::Object(
                entries
                    .into_iter()
                    .map(|(key, entry)| (key.clone(), sort_keys(entry)))
                    .collect(),
            )
        }
        _ => value.clone(),
    }
}

// ─── Turn ──────────────────────────────────────────────────────────────

#[derive(Default)]
struct TurnState {
    settled: bool,
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
    user_message_submitter: ExperimentalBridgeUserMessageSubmitter,
    done_tx: Mutex<Option<oneshot::Sender<Result<(), ErrorValue>>>>,
    done: Shared<BoxFuture<'static, Result<(), ErrorValue>>>,
}

impl Turn {
    fn channel(&self) -> &CodexChannel {
        &self.session.config.channel
    }

    fn wire(
        session: &Arc<SessionShared>,
        emit: HarnessV1Emit,
        abort_signal: Option<CancellationToken>,
    ) -> Arc<Self> {
        let (done_tx, done_rx) = oneshot::channel::<Result<(), ErrorValue>>();
        let done = async move {
            done_rx
                .await
                .unwrap_or_else(|_| Err(ErrorValue::message("codex turn was abandoned.")))
        }
        .boxed()
        .shared();
        let channel = session.config.channel.clone();
        let user_message_submitter = {
            let send_channel = channel.clone();
            let response_channel = channel.clone();
            let reconnect_channel = channel.clone();
            experimental_create_bridge_user_message_submitter(
                ExperimentalBridgeUserMessageSubmitterOptions {
                    send: Arc::new(move |message| {
                        send_channel.send(
                            &HarnessV1BridgeInboundCommand::UserMessage {
                                message_id: Some(message.message_id.clone()),
                                text: message.text.clone(),
                            }
                            .into(),
                            send_options(),
                        )
                    }),
                    on_response: Box::new(move |listener| {
                        response_channel.on(
                            "user-message-response",
                            move |frame: &CodexOutboundMessage| {
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
            )
        };
        let turn = Arc::new(Turn {
            session: session.clone(),
            emit,
            state: Mutex::default(),
            user_message_submitter,
            done_tx: Mutex::new(Some(done_tx)),
            done,
        });
        let mut subscriptions = Vec::new();

        for event_type in [
            "stream-start",
            "text-start",
            "text-delta",
            "text-end",
            "reasoning-start",
            "reasoning-delta",
            "reasoning-end",
            "tool-call",
            "tool-approval-request",
            "tool-result",
            "file-change",
            "finish-step",
            "raw",
        ] {
            let weak = Arc::downgrade(&turn);
            subscriptions.push(channel.on(event_type, move |frame: &CodexOutboundMessage| {
                if let Some(turn) = weak.upgrade() {
                    turn.forward(frame);
                }
            }));
        }
        subscriptions.push(channel.on("finish", {
            let weak = Arc::downgrade(&turn);
            move |frame: &CodexOutboundMessage| {
                let Some(turn) = weak.upgrade() else { return };
                turn.forward(frame);
                turn.settle(None);
            }
        }));
        subscriptions.push(channel.on("error", {
            let weak = Arc::downgrade(&turn);
            move |frame: &CodexOutboundMessage| {
                let Some(turn) = weak.upgrade() else { return };
                turn.forward(frame);
                if let Some(HarnessV1StreamPart::Error { error }) = frame.stream_part() {
                    turn.settle(Some(error.clone()));
                }
            }
        }));
        turn.state.lock().subscriptions = subscriptions;

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
                    "codex bridge closed before the turn finished.",
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
        Box::new(CodexPromptControl { turn: self.clone() })
    }

    /// Forwards a frame's stream part. A `finish-step` keeps the frame's
    /// replay checkpoint, which must reach the consumer for step-boundary
    /// suspension.
    fn forward(&self, frame: &CodexOutboundMessage) {
        if self.state.lock().settled {
            return;
        }
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
            (
                std::mem::take(&mut state.subscriptions),
                state.abort_task.take(),
            )
        };
        self.user_message_submitter
            .close(error.as_ref().map(ErrorValue::to_box));
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

    fn on_abort(&self) {
        if self.state.lock().settled {
            return;
        }
        let _ = self
            .channel()
            .send(&HarnessV1BridgeInboundCommand::Abort.into(), send_options());
        self.settle(Some(ErrorValue::from_box(AbortError.into())));
    }

    /// `sendStart(send)`.
    ///
    /// Codex can complete short turns without using tools. Deferring the
    /// start frame gives the harness runner one event-loop turn to finish
    /// wiring the prompt control and stream output before Codex can settle.
    ///
    /// [rust] `setTimeout(send, 0)`, which Node clamps to one millisecond.
    fn send_start(self: &Arc<Self>, message: CodexStartMessage) {
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1)).await;
            let Some(turn) = weak.upgrade() else { return };
            if turn.state.lock().settled {
                return;
            }
            if let Err(error) = turn.channel().send(&message.into(), send_options()) {
                turn.settle(Some(ErrorValue::from_box(error)));
            }
        });
    }
}

struct CodexPromptControl {
    turn: Arc<Turn>,
}

#[async_trait]
impl HarnessV1PromptControl for CodexPromptControl {
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
        true
    }

    async fn submit_user_message(&self, text: String) -> HarnessResult<()> {
        self.turn.user_message_submitter.submit(text).await
    }

    fn done(&self) -> BoxFuture<'static, HarnessResult<()>> {
        let done = self.turn.done.clone();
        async move { done.await.map_err(|error| error.to_box()) }.boxed()
    }
}

#[allow(dead_code)]
fn _assert_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<CodexSession>();
    check::<CodexHarness>();
    let _: Option<BoxError> = None;
}
