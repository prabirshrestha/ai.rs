//! Port of `harness-acp/src/v1/acp-v1-harness.ts`.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::FutureExt as _;
use futures::StreamExt as _;
use futures::future::{BoxFuture, Shared};
use indexmap::{IndexMap, IndexSet};
use parking_lot::Mutex;
use serde_json::{Map, Value, json};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::acp::acp_auth::{
    ACPAuthResolutionOptions, ACPAuthenticationProfileIdentity, ACPClientApp,
    ACPResolvedProviderAuthenticationResult, create_acp_authentication_profile_identity,
    resolve_acp_authentication_environment, resolve_acp_provider_authentication,
    resolve_acp_provider_authentication_compatibility,
};
use crate::harness::adapters::acp::acp_tool_call::{
    ACPResolveHostToolCall, ACPResolveHostToolCallOptions,
};
use crate::harness::errors::{AbortError, BoxError, HarnessError, HarnessResult};
use crate::harness::provider::JsonValue;
use crate::harness::sandbox::{
    SandboxProcess, SandboxProcessControl, SandboxProcessOptions, SandboxReadTextFileOptions,
    SandboxRunOptions, SandboxSession, SandboxSpawnOptions, SandboxWriteTextFileOptions,
};
use crate::harness::sdk::error_value::ErrorValue;
use crate::harness::sdk::messages::ToolResultPart;
use crate::harness::sdk::tool::Schema;
use crate::harness::spec::bootstrap::HarnessV1Bootstrap;
use crate::harness::spec::bridge_protocol::{
    HarnessV1BridgeDiagnosticFrame, HarnessV1BridgeInboundCommand, HarnessV1BridgeOutboundMessage,
    HarnessV1BridgeStartBase, HarnessV1BridgeToolWire, harness_v1_diagnostic_from_bridge_frame,
};
use crate::harness::spec::diagnostic::HarnessV1DebugConfig;
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1ContinueTurnOptions, HarnessV1Emit, HarnessV1PromptControl,
    HarnessV1PromptTurnOptions, HarnessV1Session, HarnessV1StartOptions,
    HarnessV1SubmitToolApproval, HarnessV1SubmitToolResult,
};
use crate::harness::spec::lifecycle_state::{
    HarnessV1ContinueTurnState, HarnessV1ResumeSessionState,
};
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
use crate::harness::spec::paths::{
    harness_state_directory_path, is_absolute_posix, posix_join, resolve_posix,
};
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::spec::tools::{
    HarnessV1BuiltinTool, HarnessV1PermissionMode, HarnessV1Skill, HarnessV1ToolSpec,
};
use crate::harness::utils::{
    BRIDGE_AUTH_PUBLIC_KEY_ENV, BridgeProcessStreamName, BridgeReadyErrorContext, BridgeStreamDone,
    CreateBridgeWebSocketConnectOptions, DiskLogRecoveryMode, ForwardBridgeProcessStreamOptions,
    SandboxChannel, SandboxChannelMessage as _, SandboxChannelOptions,
    SandboxChannelReconnectOptions, SandboxChannelSendOptions, WaitForBridgeReadyOptions,
    WebSocketConstructor, WriteInstructionsOptions, WriteSkillsOptions,
    apply_credential_forwarding, classify_disk_log, create_bridge_auth_key_pair,
    create_bridge_error_handler, create_bridge_startup_error, create_bridge_web_socket_connect,
    drain_bridge_process_stream, forward_bridge_process_stream, get_restricted_sandbox_session,
    harness_bridge_capability_unsupported_error_message, mark_bridge_starting, plain_error,
    resolve_bridge_endpoint, resolve_bridge_port, resolve_sandbox_credential_environment,
    resolve_sandbox_home_dir, shell_quote, validate_bridge_sandbox_settings, wait_for_bridge_ready,
    warn_credential_brokering_unavailable, write_instructions, write_skills,
};

use super::bootstrap::{ACPBootstrap, create_acp_bootstrap, resolve_acp_agent_launch};
use super::bridge::bridge_environment::{
    ACPBridgeConfiguration, ACPResolvedProviderAuthentication, create_acp_bridge_environment,
};
use super::bridge::profile_values::ACPGatewayValues;
use super::bridge::protocol_configuration::resolve_acp_launch_environment;
use super::bridge_protocol::{
    ACPBuiltinToolMapping, ACPColdSessionState, ACPInboundMessage, ACPOutboundFrame,
    ACPOutboundMessage, ACPRecoveryMode, ACPStartMessage, ACPTurnStartConfig,
};
use super::lifecycle::{
    ACPBridgeCoordinates, ACPLifecycleData, ACPPromptGuidanceLifecycleState, ACPRecoveryStatus,
    ACPRecoveryStatusMode, ACPRestoration, ACPRestorationMethod, create_acp_agent_identity,
    resolve_acp_initial_guidance_applied, validate_acp_lifecycle_compatibility,
};
use super::prompt::{
    ACPContentBlock, convert_harness_prompt_to_acp_content_blocks, prepend_acp_instruction_guidance,
};
use super::settings::{
    ACPAskUserQuestionsSettings, ACPAuthenticationFile, ACPAuthenticationFilesOptions,
    ACPCredentialBrokeringOptions, ACPFromNativeRequestOptions, ACPInstructionMapping,
    ACPIsMcpToolCall, ACPMatchesNativeRequest, ACPMatchesNativeRequestOptions, ACPModelMapping,
    ACPOutputSchemaMapping, ACPPermissionModeMapping, ACPPrepareSandboxOptions, ACPProfileValue,
    ACPResolveAuthenticationEnvironmentOptions, ACPSerializableValue, ACPToNativeResponseOptions,
    ACPV1Settings, resolve_acp_agent_environment, validate_acp_v1_settings,
};
use super::skills::{
    ACP_SKILL_NAME_PATTERN, DEFAULT_ACP_SKILLS_DIRECTORY, resolve_acp_private_session_directory,
    resolve_acp_skills_directory, validate_acp_skills,
};
use super::turn_start_config::{
    CreateACPTurnStartConfigOptions, create_acp_cold_session_state, create_acp_turn_start_config,
};

type ACPChannel = SandboxChannel<ACPOutboundFrame, ACPInboundMessage>;

/// `ACPRespawnStrategy`.
#[derive(Debug, Clone)]
enum ACPRespawnStrategy {
    DiskReplay {
        reason: String,
        after_seq: u64,
    },
    LossyRerun {
        reason: String,
        turn_start_config: ACPTurnStartConfig,
        // Carried as TS does; the session reads the id from lifecycle data.
        #[allow(dead_code)]
        acp_session_id: String,
    },
    ColdRestore {
        turn_start_config: ACPTurnStartConfig,
        acp_session_id: String,
    },
}

/// Options of [`create_acp_v1`].
#[derive(Clone)]
pub struct CreateACPV1Options {
    pub settings: ACPV1Settings,
    pub builtin_tools: IndexMap<String, HarnessV1BuiltinTool>,
    pub port: Option<u16>,
    pub port_endpoint: Option<HarnessV1PortEndpoint>,
    pub startup_timeout_ms: Option<u64>,
    pub host_disconnect_timeout_ms: Option<u64>,
    pub reconnect: Option<SandboxChannelReconnectOptions>,
    pub keep_alive_ms: Option<u64>,
    pub client_app: ACPClientApp,
    pub lifecycle_state_schema: Schema,
    /// [rust] Test seam standing in for the `ws` module TS tests mock.
    pub(crate) web_socket_constructor: Option<WebSocketConstructor>,
    /// [rust] Test seam standing in for `vi.stubEnv`: replaces `process.env`.
    pub(crate) process_env: Option<IndexMap<String, String>>,
}

impl CreateACPV1Options {
    /// Options with the required fields; every optional field unset.
    pub fn new(
        settings: ACPV1Settings,
        builtin_tools: IndexMap<String, HarnessV1BuiltinTool>,
        client_app: ACPClientApp,
        lifecycle_state_schema: Schema,
    ) -> Self {
        Self {
            settings,
            builtin_tools,
            port: None,
            port_endpoint: None,
            startup_timeout_ms: None,
            host_disconnect_timeout_ms: None,
            reconnect: None,
            keep_alive_ms: None,
            client_app,
            lifecycle_state_schema,
            web_socket_constructor: None,
            process_env: None,
        }
    }
}

/// The harness [`create_acp_v1`] returns.
pub struct ACPV1Harness {
    settings: Arc<ACPV1Settings>,
    builtin_tools: IndexMap<String, HarnessV1BuiltinTool>,
    builtin_tool_catalog: Vec<ACPBuiltinToolMapping>,
    port_override: Option<u16>,
    port_endpoint_override: Option<HarnessV1PortEndpoint>,
    startup_timeout_ms: Option<u64>,
    host_disconnect_timeout_ms: Option<u64>,
    reconnect: Option<SandboxChannelReconnectOptions>,
    keep_alive_ms: Option<u64>,
    client_app: ACPClientApp,
    lifecycle_state_schema: Schema,
    bootstrap: ACPBootstrap,
    permission_mode_mapping: Option<ACPPermissionModeMapping>,
    web_socket_constructor: Option<WebSocketConstructor>,
    process_env: Option<IndexMap<String, String>>,
}

/// `createACPV1`.
///
/// [rust] The deprecated `mintBridgeToken` warning is not ported.
pub fn create_acp_v1(options: CreateACPV1Options) -> HarnessResult<ACPV1Harness> {
    validate_acp_v1_settings(&options.settings)?;
    let bootstrap = create_acp_bootstrap(&options.settings.harness_id, &options.settings.source);
    // [rust] A typed mapping is always complete (`isCompletePermissionModeMapping`).
    let permission_mode_mapping = options.settings.permission_mode_mapping.clone();
    let builtin_tool_catalog = serialize_builtin_tools(&options.builtin_tools);
    Ok(ACPV1Harness {
        settings: Arc::new(options.settings),
        builtin_tools: options.builtin_tools,
        builtin_tool_catalog,
        port_override: options.port,
        port_endpoint_override: options.port_endpoint,
        startup_timeout_ms: options.startup_timeout_ms,
        host_disconnect_timeout_ms: options.host_disconnect_timeout_ms,
        reconnect: options.reconnect,
        keep_alive_ms: options.keep_alive_ms,
        client_app: options.client_app,
        lifecycle_state_schema: options.lifecycle_state_schema,
        bootstrap,
        permission_mode_mapping,
        web_socket_constructor: options.web_socket_constructor,
        process_env: options.process_env,
    })
}

fn unsupported(harness_id: &str, message: impl Into<String>, cause: Option<BoxError>) -> BoxError {
    Box::new(HarnessError::CapabilityUnsupported {
        message: message.into(),
        harness_id: Some(harness_id.to_string()),
        cause,
    })
}

fn check_aborted(abort_signal: Option<&CancellationToken>) -> HarnessResult<()> {
    if abort_signal.is_some_and(CancellationToken::is_cancelled) {
        return Err(AbortError.into());
    }
    Ok(())
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

fn tool_result_command(
    tool_call_id: impl Into<String>,
    output: JsonValue,
    is_error: Option<bool>,
    tool_result: Option<&ToolResultPart>,
) -> ACPInboundMessage {
    ACPInboundMessage::Command(HarnessV1BridgeInboundCommand::ToolResult {
        tool_call_id: tool_call_id.into(),
        output,
        is_error,
        tool_result: tool_result.and_then(tool_result_part_json),
    })
}

/// A `ToolResultPart` as TS sends it: with its `type: 'tool-result'` tag,
/// which the Rust struct only carries inside a message's content.
fn tool_result_part_json(tool_result: &ToolResultPart) -> Option<JsonValue> {
    let Value::Object(fields) = serde_json::to_value(tool_result).ok()? else {
        return None;
    };
    let mut tagged = Map::new();
    tagged.insert("type".into(), Value::String("tool-result".into()));
    tagged.extend(fields);
    Some(Value::Object(tagged))
}

#[async_trait]
impl HarnessV1 for ACPV1Harness {
    fn harness_id(&self) -> &str {
        &self.settings.harness_id
    }

    fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool> {
        &self.builtin_tools
    }

    fn supports_builtin_tool_approvals(&self) -> bool {
        true
    }

    fn supports_builtin_tool_filtering(&self) -> bool {
        false
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
        Ok((*self.bootstrap.get_bootstrap()).clone())
    }

    async fn do_start(
        &self,
        start_options: HarnessV1StartOptions,
    ) -> HarnessResult<Box<dyn HarnessV1Session>> {
        self.start(start_options).await
    }
}

impl ACPV1Harness {
    async fn start(
        &self,
        start_options: HarnessV1StartOptions,
    ) -> HarnessResult<Box<dyn HarnessV1Session>> {
        let settings = &*self.settings;
        let harness_id = settings.harness_id.as_str();
        if start_options.builtin_tool_filtering.is_some() {
            return Err(unsupported(
                harness_id,
                "ACP built-in tool filtering is not available in this initial ACP v1 implementation.",
                None,
            ));
        }
        let abort_signal = start_options.abort_signal.clone();
        let permission_mode = start_options
            .permission_mode
            .unwrap_or(HarnessV1PermissionMode::AllowAll);
        let env: IndexMap<String, String> = self
            .process_env
            .clone()
            .unwrap_or_else(|| std::env::vars().collect());
        let authentication_environment = match &settings.resolve_authentication_environment {
            None => resolve_acp_authentication_environment(settings.auth.as_ref(), &env),
            Some(resolve) => {
                resolve(ACPResolveAuthenticationEnvironmentOptions {
                    auth: settings.auth.clone(),
                    env: env.clone(),
                })
                .await?
            }
        };
        let provider_authentication_compatibility =
            resolve_acp_provider_authentication_compatibility(
                settings.auth.as_ref(),
                settings.provider_authentication.as_ref(),
                &authentication_environment,
            );
        let agent_identity = create_acp_agent_identity(
            settings,
            &self.client_app,
            provider_authentication_compatibility.as_ref(),
        );
        let authentication_profile = create_acp_authentication_profile_identity(
            settings.authentication.as_ref(),
            provider_authentication_compatibility.as_ref(),
        );
        let resolved_provider_authentication = resolve_acp_provider_authentication(
            ACPAuthResolutionOptions {
                mode: settings.auth.as_ref(),
                provider_authentication: settings.provider_authentication.as_ref(),
                client_app: &self.client_app,
            },
            &authentication_environment,
            provider_authentication_compatibility.as_ref(),
        )?;
        let sandbox_session = &start_options.sandbox_session;
        let tool_safe_sandbox_session = get_restricted_sandbox_session(sandbox_session);
        let sandbox_id = sandbox_session.network().map(|session| session.id());
        validate_bridge_sandbox_settings(
            sandbox_session,
            self.port_override,
            self.port_endpoint_override.as_ref(),
        )?;
        let continue_from = start_options.continue_from.clone().or_else(|| {
            start_options
                .resume_from
                .as_ref()
                .and_then(|resume| resume.continue_from.clone())
        });
        let lifecycle_state: Option<(String, JsonValue)> =
            match (&continue_from, &start_options.resume_from) {
                (Some(state), _) => Some((state.harness_id.clone(), state.data.clone())),
                (None, Some(state)) => Some((state.harness_id.clone(), state.data.clone())),
                (None, None) => None,
            };
        let is_continue = continue_from.is_some();
        let is_resume = lifecycle_state.is_some();
        let lifecycle_data: Option<ACPLifecycleData> = match &lifecycle_state {
            Some((_, data)) if !data.is_null() => Some(serde_json::from_value(data.clone())?),
            _ => None,
        };
        if let (Some(data), Some((lifecycle_harness_id, _))) = (&lifecycle_data, &lifecycle_state) {
            validate_acp_lifecycle_compatibility(
                harness_id,
                lifecycle_harness_id,
                &agent_identity,
                &authentication_profile,
                data,
                sandbox_id.as_deref(),
            )?;
        }
        let mut agent_lookup_env = env.clone();
        agent_lookup_env.extend(authentication_environment.clone());
        let agent_environment =
            resolve_acp_agent_environment(settings, &agent_lookup_env, &authentication_environment);
        let mut sandbox_agent_environment = agent_environment.clone();
        let mut sandbox_provider_authentication_environment =
            resolved_provider_authentication.env.clone();
        let mut sandbox_provider_environment: Option<IndexMap<String, String>> = None;
        let credential_environment_variables: Vec<String> = settings
            .credential_env
            .iter()
            .flatten()
            .cloned()
            .chain([
                "AI_GATEWAY_API_KEY".to_string(),
                "VERCEL_OIDC_TOKEN".to_string(),
            ])
            .collect::<IndexSet<_>>()
            .into_iter()
            .collect();
        let credential_forwarding_environment_variables: Vec<String> =
            credential_environment_variables
                .iter()
                .cloned()
                .chain(resolve_acp_provider_credential_environment_variables(
                    resolved_provider_authentication
                        .provider_authentication
                        .as_ref(),
                ))
                .collect::<IndexSet<_>>()
                .into_iter()
                .collect();
        let mut sandbox_credential_environment: Option<IndexMap<String, String>> = None;
        let is_ai_gateway = matches!(
            resolved_provider_authentication.provider_authentication,
            Some(ACPResolvedProviderAuthentication::AiGateway { .. })
        );

        if let (Some(credential_brokering), Some(network)) = (
            &settings.credential_brokering,
            sandbox_session
                .network()
                .filter(|session| session.supports_add_request_transformations()),
        ) {
            let provider_environment =
                resolve_provider_environment(&resolved_provider_authentication, &self.client_app)?;
            let mut brokering_environment = agent_environment.clone();
            brokering_environment.extend(provider_environment);
            let credential_environment = resolve_sandbox_credential_environment(
                &brokering_environment,
                &credential_forwarding_environment_variables,
                settings.credential_forwarding.as_ref(),
                lifecycle_data
                    .as_ref()
                    .and_then(|data| data.sandbox_credential_environment.as_ref()),
            )
            .await?;
            sandbox_agent_environment = brokering_environment.clone();
            sandbox_agent_environment.extend(credential_environment.clone());
            let request_transformations = credential_brokering(ACPCredentialBrokeringOptions {
                env: &brokering_environment,
                sandbox_env: Some(&sandbox_agent_environment),
                headers: start_options.headers.as_ref(),
            });
            if !request_transformations.is_empty() {
                network
                    .add_request_transformations(request_transformations)
                    .await?;
            }
            sandbox_credential_environment = Some(credential_environment);
            sandbox_provider_environment = is_ai_gateway.then(IndexMap::new);
        }
        if settings.credential_forwarding.is_some()
            && sandbox_provider_environment.is_none()
            && is_ai_gateway
        {
            sandbox_provider_environment = Some(resolve_provider_environment(
                &resolved_provider_authentication,
                &self.client_app,
            )?);
        }
        if let Some(provider_environment) = &sandbox_provider_environment {
            sandbox_agent_environment.extend(provider_environment.clone());
            sandbox_provider_authentication_environment = resolved_provider_authentication
                .env
                .iter()
                .filter(|(key, _)| {
                    key.as_str() != "AI_SDK_ACP_GATEWAY_API_KEY"
                        && key.as_str() != "AI_SDK_ACP_GATEWAY_BASE_URL"
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
        }

        // Harness SDK state always lives under the sandbox's own HOME, never
        // the working directory, so it stays out of a user-owned workspace.
        let sandbox_home_dir =
            resolve_sandbox_home_dir(&*tool_safe_sandbox_session, abort_signal.clone()).await?;
        let state_directory = harness_state_directory_path(&sandbox_home_dir);
        let resolved_bridge_dir = resolve_posix(&state_directory, &self.bootstrap.bootstrap_dir);
        let agent_launch = resolve_acp_agent_launch(
            &resolved_bridge_dir,
            &settings.source,
            &settings.executable,
            settings.args.as_deref(),
        );
        let work_dir = start_options.session_work_dir.clone();
        let private_session_dir =
            resolve_acp_private_session_directory(&state_directory, &start_options.session_id);
        let agent_home_dir = agent_launch
            .home
            .clone()
            .unwrap_or_else(|| sandbox_home_dir.clone());
        let skills_dir = settings
            .skills_directory
            .clone()
            .unwrap_or_else(|| DEFAULT_ACP_SKILLS_DIRECTORY.to_string());
        let skills_directory =
            resolve_acp_skills_directory(&agent_home_dir, settings.skills_directory.as_deref())?;
        let bridge_state_dir = format!("{private_session_dir}/bridge");
        let on_diagnostic = start_options
            .observability
            .as_ref()
            .and_then(|observability| observability.report.clone())
            .map(|report| {
                let session_id = start_options.session_id.clone();
                Arc::new(move |frame: &ACPOutboundFrame| {
                    let frame = match &frame.message {
                        ACPOutboundMessage::Bridge(HarnessV1BridgeOutboundMessage::SandboxLog(
                            log,
                        )) => HarnessV1BridgeDiagnosticFrame::SandboxLog(log),
                        ACPOutboundMessage::Bridge(HarnessV1BridgeOutboundMessage::DebugEvent(
                            event,
                        )) => HarnessV1BridgeDiagnosticFrame::DebugEvent(event),
                        _ => return,
                    };
                    report(harness_v1_diagnostic_from_bridge_frame(
                        frame,
                        Some(session_id.clone()),
                        now_ms(),
                    ));
                }) as Arc<dyn Fn(&ACPOutboundFrame) + Send + Sync>
            });
        let on_bridge_error = {
            let handler = create_bridge_error_handler(
                harness_id.to_string(),
                Some(start_options.session_id.clone()),
            );
            Arc::new(move |frame: &ACPOutboundFrame| {
                if let Some(HarnessV1StreamPart::Error { error }) = frame.message.stream_part() {
                    handler(&error.to_js_json());
                }
            }) as Arc<dyn Fn(&ACPOutboundFrame) + Send + Sync>
        };
        let debug = start_options
            .observability
            .as_ref()
            .and_then(|observability| observability.debug.clone());
        let builtin_tool_catalog = self.builtin_tool_catalog.clone();
        let new_channel = |endpoint: HarnessV1PortEndpoint,
                           private_key: String,
                           open_retry_timeout_ms: Option<u64>,
                           start_abort_signal: Option<CancellationToken>,
                           initial_last_seen_event_id: Option<u64>|
         -> HarnessResult<ACPChannel> {
            let mut connect_options =
                CreateBridgeWebSocketConnectOptions::new(endpoint, private_key);
            connect_options.web_socket_constructor = self.web_socket_constructor.clone();
            connect_options.open_retry_timeout_ms = open_retry_timeout_ms;
            connect_options.start_abort_signal = start_abort_signal;
            let mut channel_options =
                SandboxChannelOptions::new(create_bridge_web_socket_connect(connect_options)?);
            channel_options.initial_last_seen_event_id = initial_last_seen_event_id;
            channel_options.on_diagnostic = on_diagnostic.clone();
            channel_options.on_bridge_error = Some(on_bridge_error.clone());
            channel_options.reconnect = self.reconnect;
            channel_options.keep_alive_ms = self.keep_alive_ms;
            Ok(SandboxChannel::new(channel_options))
        };
        let session_config = |channel: ACPChannel,
                              proc: Option<Arc<dyn SandboxProcessControl>>,
                              bridge_port: u16,
                              bridge_private_key: String|
         -> ACPSessionConfig {
            ACPSessionConfig {
                session_id: start_options.session_id.clone(),
                harness_id: harness_id.to_string(),
                channel,
                proc,
                model_mapping: settings.model_mapping.clone(),
                session_meta: settings
                    .session
                    .as_ref()
                    .and_then(|session| session.meta.clone()),
                instruction_mapping: settings.instruction_mapping.clone(),
                output_schema_mapping: settings.output_schema_mapping.clone(),
                ask_user_questions: settings.ask_user_questions.clone(),
                debug: debug.clone(),
                agent_identity: agent_identity.clone(),
                authentication_profile: authentication_profile.clone(),
                builtin_tools: builtin_tool_catalog.clone(),
                permission_mode,
                permission_mode_mapping: self.permission_mode_mapping.clone(),
                mcp_servers: settings.mcp_servers.clone(),
                is_mcp_tool_call: settings.is_mcp_tool_call.clone(),
                resolve_host_tool_call: settings.resolve_host_tool_call.clone(),
                sandbox: tool_safe_sandbox_session.clone(),
                home_path: agent_home_dir.clone(),
                skills_dir: skills_dir.clone(),
                skills_directory: skills_directory.clone(),
                bridge_port,
                bridge_private_key,
                sandbox_id: sandbox_id.clone(),
                bridge_state_dir: bridge_state_dir.clone(),
                sandbox_credential_environment: sandbox_credential_environment.clone(),
                is_resume,
                turn_in_flight: false,
                initial_guidance_applied: false,
                instructions_fingerprint: None,
                acp_session_id: None,
                turn_start_config: None,
                recovery_status: None,
                restoration: None,
                replay_only: false,
                lossy_rerun: false,
            }
        };
        let mut respawn_strategy: Option<ACPRespawnStrategy> = None;

        if is_resume {
            let Some(lifecycle_data) = &lifecycle_data else {
                return Err(plain_error("ACP lifecycle state data is missing."));
            };
            let coords = lifecycle_data.bridge.clone();
            if coords.is_none() && is_continue {
                return Err(unsupported(
                    harness_id,
                    "ACP continuation state does not contain bridge coordinates required for replay or process-loss rerun.",
                    None,
                ));
            }
            if let Some(coords) = coords {
                let attached: HarnessResult<ACPSession> = async {
                    let attach_endpoint = resolve_bridge_endpoint(
                        sandbox_session,
                        coords.port,
                        self.port_endpoint_override.as_ref(),
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
                    let mut config = session_config(
                        attach_channel,
                        None,
                        coords.port,
                        coords.private_key.clone(),
                    );
                    config.initial_guidance_applied =
                        resolve_acp_initial_guidance_applied(true, Some(&lifecycle_data.guidance));
                    config.instructions_fingerprint =
                        lifecycle_data.guidance.instructions_fingerprint.clone();
                    config.acp_session_id = lifecycle_data.acp_session_id.clone();
                    config.is_resume = true;
                    config.turn_in_flight = is_continue;
                    config.turn_start_config = lifecycle_data.turn_start_config.clone();
                    config.recovery_status = lifecycle_data.recovery.clone();
                    config.restoration = lifecycle_data.restoration;
                    Ok(ACPSession::new(config))
                }
                .await;
                match attached {
                    Ok(session) => return Ok(Box::new(session)),
                    Err(error) => {
                        if is_continue {
                            let event_log = tool_safe_sandbox_session
                                .read_text_file(SandboxReadTextFileOptions {
                                    path: format!("{bridge_state_dir}/event-log.ndjson"),
                                    abort_signal: abort_signal.clone(),
                                    ..Default::default()
                                })
                                .await?;
                            if classify_disk_log(event_log.as_deref())
                                == DiskLogRecoveryMode::Replay
                            {
                                respawn_strategy = Some(ACPRespawnStrategy::DiskReplay {
                                    reason: "completed coherent event log".into(),
                                    after_seq: coords.last_seen_event_id,
                                });
                            } else {
                                let (Some(turn_start_config), Some(acp_session_id)) = (
                                    lifecycle_data.turn_start_config.clone(),
                                    lifecycle_data.acp_session_id.clone(),
                                ) else {
                                    return Err(unsupported(
                                        harness_id,
                                        "ACP process-loss recovery is unavailable because the lifecycle state does not contain the persisted turn start configuration and ACP session identifier.",
                                        Some(error),
                                    ));
                                };
                                validate_acp_turn_start_config(
                                    &turn_start_config,
                                    &TurnStartConfigContext {
                                        authentication_profile: &authentication_profile,
                                        session_meta: settings
                                            .session
                                            .as_ref()
                                            .and_then(|session| session.meta.as_ref()),
                                        instruction_mapping: settings.instruction_mapping.as_ref(),
                                        output_schema_mapping: settings
                                            .output_schema_mapping
                                            .as_ref(),
                                        model_mapping: &settings.model_mapping,
                                        builtin_tools: &builtin_tool_catalog,
                                        permission_mode_mapping: self
                                            .permission_mode_mapping
                                            .as_ref(),
                                        mcp_servers: settings.mcp_servers.as_ref(),
                                    },
                                )?;
                                respawn_strategy = Some(ACPRespawnStrategy::LossyRerun {
                                    reason: "event log not replayable".into(),
                                    turn_start_config,
                                    acp_session_id,
                                });
                            }
                        }
                    }
                }
            }
            if !is_continue {
                let (Some(cold_session), Some(acp_session_id)) = (
                    lifecycle_data.cold_session.clone(),
                    lifecycle_data.acp_session_id.clone(),
                ) else {
                    return Err(unsupported(
                        harness_id,
                        "Cold ACP session restoration requires persisted cold-session configuration and an ACP session identifier.",
                        None,
                    ));
                };
                let turn_start_config = validate_acp_cold_session_configuration(
                    &cold_session,
                    permission_mode,
                    debug.as_ref(),
                    &TurnStartConfigContext {
                        authentication_profile: &authentication_profile,
                        session_meta: settings
                            .session
                            .as_ref()
                            .and_then(|session| session.meta.as_ref()),
                        instruction_mapping: settings.instruction_mapping.as_ref(),
                        output_schema_mapping: settings.output_schema_mapping.as_ref(),
                        model_mapping: &settings.model_mapping,
                        builtin_tools: &builtin_tool_catalog,
                        permission_mode_mapping: self.permission_mode_mapping.as_ref(),
                        mcp_servers: settings.mcp_servers.as_ref(),
                    },
                )?;
                respawn_strategy = Some(ACPRespawnStrategy::ColdRestore {
                    turn_start_config,
                    acp_session_id,
                });
            }
        }

        let forwarded_agent_environment = if sandbox_credential_environment.is_none() {
            apply_credential_forwarding(
                &sandbox_agent_environment,
                &credential_forwarding_environment_variables,
                settings.credential_forwarding.as_ref(),
            )
            .await?
        } else {
            sandbox_agent_environment.clone()
        };
        if settings.credential_brokering.is_some() && sandbox_credential_environment.is_none() {
            let mut environment = sandbox_agent_environment.clone();
            environment.extend(resolved_provider_authentication.env.clone());
            let mut forwarded_environment = forwarded_agent_environment.clone();
            forwarded_environment.extend(sandbox_provider_authentication_environment.clone());
            let mut variables = credential_forwarding_environment_variables.clone();
            variables.push("AI_SDK_ACP_GATEWAY_API_KEY".into());
            warn_credential_brokering_unavailable(&environment, &forwarded_environment, &variables);
        }
        let authentication_files = settings
            .authentication_files
            .as_ref()
            .map(|files| {
                files(ACPAuthenticationFilesOptions {
                    env: &agent_environment,
                    sandbox_env: &forwarded_agent_environment,
                    credential_brokering_available: sandbox_credential_environment.is_some(),
                })
            })
            .unwrap_or_default();
        write_acp_authentication_files(
            &*tool_safe_sandbox_session,
            &agent_home_dir,
            &authentication_files,
            abort_signal.clone(),
        )
        .await?;
        if let Some(prepare_sandbox) = &settings.prepare_sandbox {
            check_aborted(abort_signal.as_ref())?;
            prepare_sandbox(ACPPrepareSandboxOptions {
                sandbox: tool_safe_sandbox_session.clone(),
                work_dir: work_dir.clone(),
                abort_signal: abort_signal.clone(),
            })
            .await?;
            check_aborted(abort_signal.as_ref())?;
        }
        let port = resolve_bridge_port(sandbox_session, self.port_override)?;
        let bridge_auth = create_bridge_auth_key_pair()?;
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
        mark_bridge_starting(
            &*tool_safe_sandbox_session,
            &bridge_state_dir,
            harness_id,
            abort_signal.clone(),
        )
        .await;

        let mut spawn_env = forwarded_agent_environment.clone();
        spawn_env.extend(create_acp_bridge_environment(&ACPBridgeConfiguration {
            agent: agent_launch.clone(),
            authentication: settings.authentication.clone(),
            provider_authentication: resolved_provider_authentication
                .provider_authentication
                .clone(),
            provider_environment: sandbox_provider_environment
                .as_ref()
                .map(|_| IndexMap::new()),
            session_meta: settings
                .session
                .as_ref()
                .and_then(|session| session.meta.clone()),
            client_capabilities: settings.client_capabilities.clone(),
            ask_user_questions_request_method: settings
                .ask_user_questions
                .as_ref()
                .map(|questions| questions.request_method.clone()),
            host_tool_mcp_transport: settings.host_tool_mcp_transport,
            non_text_content_types: settings.non_text_content_types.clone(),
            write_protected_paths: settings.write_protected_paths.clone(),
        }));
        spawn_env.extend(sandbox_provider_authentication_environment.clone());
        spawn_env.insert(
            BRIDGE_AUTH_PUBLIC_KEY_ENV.to_string(),
            bridge_auth.public_key.clone(),
        );
        spawn_env.insert("BRIDGE_WS_PORT".to_string(), port.to_string());
        if matches!(
            respawn_strategy,
            Some(ACPRespawnStrategy::DiskReplay { .. })
        ) {
            spawn_env.insert("BRIDGE_REPLAY_FROM_DISK".to_string(), "1".to_string());
        }
        let mut command = format!(
            "node {} --workdir {} --bridge-state-dir {} --bridge-type {}",
            shell_quote(&format!("{resolved_bridge_dir}/bridge.mjs")),
            shell_quote(&work_dir),
            shell_quote(&bridge_state_dir),
            shell_quote(harness_id)
        );
        if let Some(timeout) = self.host_disconnect_timeout_ms {
            command.push_str(&format!(
                " --host-disconnect-timeout-ms {}",
                shell_quote(&timeout.to_string())
            ));
        }
        let mut proc: SandboxProcess = tool_safe_sandbox_session
            .spawn(SandboxSpawnOptions {
                process: SandboxProcessOptions {
                    command,
                    env: Some(spawn_env),
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
                    source: Some(harness_id.to_string()),
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
        let timeout_ms = self.startup_timeout_ms.unwrap_or(120_000);
        let startup_error = |message: String| -> crate::harness::utils::BridgeReadyErrorFactory {
            let stderr_tail = stderr_tail.clone();
            let stderr_done = bridge_stderr_done.clone();
            Box::new(move |context: BridgeReadyErrorContext| {
                async move {
                    create_bridge_startup_error(
                        &message,
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
            bridge_type: harness_id.to_string(),
            timeout_ms,
            abort_signal: abort_signal.clone(),
            poll_interval_ms: None,
            create_timeout_error: Some(startup_error(format!(
                "{harness_id} ACP bridge did not become ready in time."
            ))),
            create_exit_error: Some(startup_error(format!(
                "{harness_id} ACP bridge exited before becoming ready."
            ))),
        })
        .await?;
        let bound_port = ready.port;
        let stdout = std::mem::replace(&mut proc.stdout, futures::stream::empty().boxed());
        tokio::spawn(drain_bridge_process_stream(stdout));
        let proc_control = proc.control.clone();

        let bridge_endpoint = resolve_bridge_endpoint(
            sandbox_session,
            bound_port,
            self.port_endpoint_override.as_ref(),
        )
        .await?;
        let channel = new_channel(
            bridge_endpoint,
            bridge_auth.private_key.clone(),
            Some(timeout_ms),
            abort_signal.clone(),
            match &respawn_strategy {
                Some(ACPRespawnStrategy::DiskReplay { after_seq, .. }) => Some(*after_seq),
                _ => None,
            },
        )?;
        channel
            .open(matches!(
                respawn_strategy,
                Some(ACPRespawnStrategy::DiskReplay { .. })
            ))
            .await?;
        let mut cold_restoration: Option<ACPRestorationMethod> = None;
        if let Some(ACPRespawnStrategy::ColdRestore {
            turn_start_config,
            acp_session_id,
        }) = &respawn_strategy
        {
            let mut base = HarnessV1BridgeStartBase::new(Vec::<ACPContentBlock>::new());
            base.debug = debug.clone();
            base.permission_mode = Some(permission_mode);
            base.tools = Some(tool_wires(&turn_start_config.tools));
            let start = ACPStartMessage {
                base,
                instructions: None,
                instruction_mapping: settings.instruction_mapping.clone(),
                output_schema_mapping: None,
                mcp_servers: settings.mcp_servers.clone(),
                builtin_tools: builtin_tool_catalog.clone(),
                permission_mode_mapping: self.permission_mode_mapping.clone(),
                model_mapping: None,
                turn_start_config: turn_start_config.clone(),
                recovery_mode: Some(ACPRecoveryMode::ColdRestore {
                    acp_session_id: acp_session_id.clone(),
                }),
            };
            match restore_cold_acp_session(&channel, harness_id, start).await {
                Ok(method) => cold_restoration = Some(method),
                Err(error) => {
                    channel.begin_close();
                    if !channel.is_closed() {
                        let _ = channel.send(
                            &HarnessV1BridgeInboundCommand::Destroy.into(),
                            send_options(),
                        );
                    }
                    let _ = proc_control.kill().await;
                    channel.close();
                    return Err(error);
                }
            }
        }

        let mut config = session_config(
            channel,
            Some(proc_control),
            bound_port,
            bridge_auth.private_key.clone(),
        );
        config.initial_guidance_applied = resolve_acp_initial_guidance_applied(
            is_resume,
            lifecycle_data.as_ref().map(|data| &data.guidance),
        );
        config.instructions_fingerprint = lifecycle_data
            .as_ref()
            .and_then(|data| data.guidance.instructions_fingerprint.clone());
        config.acp_session_id = lifecycle_data
            .as_ref()
            .and_then(|data| data.acp_session_id.clone());
        config.turn_in_flight = matches!(
            respawn_strategy,
            Some(ACPRespawnStrategy::DiskReplay { .. } | ACPRespawnStrategy::LossyRerun { .. })
        );
        config.turn_start_config = match &respawn_strategy {
            Some(
                ACPRespawnStrategy::LossyRerun {
                    turn_start_config, ..
                }
                | ACPRespawnStrategy::ColdRestore {
                    turn_start_config, ..
                },
            ) => Some(turn_start_config.clone()),
            _ => lifecycle_data
                .as_ref()
                .and_then(|data| data.turn_start_config.clone()),
        };
        config.recovery_status = match &respawn_strategy {
            Some(ACPRespawnStrategy::DiskReplay { reason, .. }) => Some(ACPRecoveryStatus {
                mode: ACPRecoveryStatusMode::DiskReplay,
                reason: reason.clone(),
            }),
            Some(ACPRespawnStrategy::LossyRerun { reason, .. }) => Some(ACPRecoveryStatus {
                mode: ACPRecoveryStatusMode::LossyRerun,
                reason: reason.clone(),
            }),
            _ => None,
        };
        config.restoration = match cold_restoration {
            Some(method) => Some(ACPRestoration { method }),
            None => lifecycle_data.as_ref().and_then(|data| data.restoration),
        };
        config.replay_only = matches!(
            respawn_strategy,
            Some(ACPRespawnStrategy::DiskReplay { .. })
        );
        config.lossy_rerun = matches!(
            respawn_strategy,
            Some(ACPRespawnStrategy::LossyRerun { .. })
        );
        Ok(Box::new(ACPSession::new(config)))
    }
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

/// `writeACPAuthenticationFiles`.
async fn write_acp_authentication_files(
    sandbox: &dyn SandboxSession,
    home_path: &str,
    files: &[ACPAuthenticationFile],
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    let target_paths = files
        .iter()
        .map(|file| {
            if file.path.is_empty()
                || is_absolute_posix(&file.path)
                || file.path.split('/').any(|segment| segment == "..")
            {
                return Err(plain_error(format!(
                    "ACP authentication file path must be relative without traversal: {}.",
                    Value::String(file.path.clone())
                )));
            }
            Ok(posix_join(&[home_path, &file.path]))
        })
        .collect::<HarnessResult<Vec<_>>>()?;

    for (path, file) in target_paths.iter().zip(files) {
        sandbox
            .write_text_file(SandboxWriteTextFileOptions {
                path: path.clone(),
                content: file.content.clone(),
                abort_signal: abort_signal.clone(),
                encoding: None,
            })
            .await?;
    }
    if target_paths.is_empty() {
        return Ok(());
    }

    let result = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command: format!(
                    "chmod 600 -- {}",
                    target_paths
                        .iter()
                        .map(|path| shell_quote(path))
                        .collect::<Vec<_>>()
                        .join(" ")
                ),
                abort_signal,
                ..Default::default()
            },
            ..Default::default()
        })
        .await?;
    if result.exit_code != 0 {
        return Err(plain_error(format!(
            "Failed to secure ACP authentication files (exit {}){}",
            result.exit_code,
            if result.stderr.is_empty() {
                String::new()
            } else {
                format!(": {}", result.stderr)
            }
        )));
    }
    Ok(())
}

/// `resolveProviderEnvironment`.
fn resolve_provider_environment(
    resolved_provider_authentication: &ACPResolvedProviderAuthenticationResult,
    client_app: &ACPClientApp,
) -> HarnessResult<IndexMap<String, String>> {
    let Some(provider_authentication @ ACPResolvedProviderAuthentication::AiGateway { .. }) =
        &resolved_provider_authentication.provider_authentication
    else {
        return Ok(IndexMap::new());
    };
    let require = |name: &str| {
        resolved_provider_authentication
            .env
            .get(name)
            .cloned()
            .ok_or_else(|| {
                plain_error(format!(
                    "ACP resolved environment value {name} is unavailable."
                ))
            })
    };
    resolve_acp_launch_environment(
        Some(provider_authentication),
        Some(&ACPGatewayValues {
            api_key: require("AI_SDK_ACP_GATEWAY_API_KEY")?,
            base_url: require("AI_SDK_ACP_GATEWAY_BASE_URL")?,
            client_app_name: client_app.name.clone(),
            client_app_version: client_app.version.clone(),
        }),
    )
}

/// `resolveACPProviderCredentialEnvironmentVariables`.
fn resolve_acp_provider_credential_environment_variables(
    provider_authentication: Option<&ACPResolvedProviderAuthentication>,
) -> Vec<String> {
    let Some(ACPResolvedProviderAuthentication::AiGateway { env }) = provider_authentication else {
        return Vec::new();
    };
    env.iter()
        .filter(|(_, value)| contains_acp_provider_credential(value))
        .map(|(name, _)| name.clone())
        .collect()
}

/// `containsACPProviderCredential`.
fn contains_acp_provider_credential(value: &ACPProfileValue) -> bool {
    match value {
        Value::Array(items) => items.iter().any(contains_acp_provider_credential),
        Value::Object(record) => {
            if let Some(source) = record.get("$source") {
                return source == "gateway-api-key" || source == "gateway-authorization";
            }
            record.values().any(contains_acp_provider_credential)
        }
        _ => false,
    }
}

/// `restoreColdACPSession`.
async fn restore_cold_acp_session(
    channel: &ACPChannel,
    harness_id: &str,
    start: ACPStartMessage,
) -> HarnessResult<ACPRestorationMethod> {
    type Outcome = Result<Option<ACPRestorationMethod>, ErrorValue>;
    struct Restore {
        method: Mutex<Option<ACPRestorationMethod>>,
        complete: Mutex<Option<oneshot::Sender<Outcome>>>,
        subscriptions: Mutex<Vec<crate::harness::utils::Unsubscribe>>,
    }
    impl Restore {
        fn settle(&self, outcome: Outcome) {
            let Some(complete) = self.complete.lock().take() else {
                return;
            };
            for unsubscribe in std::mem::take(&mut *self.subscriptions.lock()) {
                unsubscribe();
            }
            let _ = complete.send(outcome);
        }
    }
    let (complete, done) = oneshot::channel();
    let restore = Arc::new(Restore {
        method: Mutex::new(None),
        complete: Mutex::new(Some(complete)),
        subscriptions: Mutex::default(),
    });
    let subscriptions = vec![
        channel.on("raw", {
            let restore = restore.clone();
            move |frame: &ACPOutboundFrame| {
                if let Some(HarnessV1StreamPart::Raw { raw_value }) = frame.message.stream_part()
                    && raw_value.get("type").and_then(Value::as_str) == Some("acp-session-restored")
                {
                    let method = match raw_value.get("method").and_then(Value::as_str) {
                        Some("resume") => Some(ACPRestorationMethod::Resume),
                        Some("load") => Some(ACPRestorationMethod::Load),
                        _ => None,
                    };
                    if let Some(method) = method {
                        *restore.method.lock() = Some(method);
                    }
                }
            }
        }),
        channel.on("finish", {
            let restore = restore.clone();
            move |_: &ACPOutboundFrame| {
                let method = *restore.method.lock();
                restore.settle(Ok(method));
            }
        }),
        channel.on("error", {
            let restore = restore.clone();
            let harness_id = harness_id.to_string();
            move |frame: &ACPOutboundFrame| {
                if let Some(HarnessV1StreamPart::Error { error }) = frame.message.stream_part() {
                    restore.settle(Err(deserialize_bridge_error(error, &harness_id)));
                }
            }
        }),
    ];
    restore.subscriptions.lock().extend(subscriptions);
    {
        let restore = Arc::downgrade(&restore);
        let harness_id = harness_id.to_string();
        channel.on_close(move |_code, reason| {
            if let Some(restore) = restore.upgrade() {
                restore.settle(Err(ErrorValue::message(format!(
                    "{harness_id} ACP bridge closed during cold restoration: {reason}"
                ))));
            }
        });
    }
    if let Err(error) = channel.send(&start.into(), send_options()) {
        restore.settle(Err(ErrorValue::from_box(error)));
    }
    let outcome = done
        .await
        .unwrap_or_else(|_| Err(ErrorValue::message("ACP cold restoration was abandoned.")));
    match outcome {
        Err(error) => Err(error.to_box()),
        Ok(None) => Err(plain_error(format!(
            "{harness_id} ACP cold restoration completed without identifying the negotiated method."
        ))),
        Ok(Some(method)) => Ok(method),
    }
}

/// `deserializeBridgeError`.
fn deserialize_bridge_error(error: &ErrorValue, harness_id: &str) -> ErrorValue {
    if let ErrorValue::Value(value) = error
        && let Some(message) = harness_bridge_capability_unsupported_error_message(value)
    {
        return ErrorValue::from_box(unsupported(harness_id, message, None));
    }
    error.clone()
}

/// `serializeBuiltinTools`.
pub fn serialize_builtin_tools(
    builtin_tools: &IndexMap<String, HarnessV1BuiltinTool>,
) -> Vec<ACPBuiltinToolMapping> {
    builtin_tools
        .iter()
        .map(|(tool_name, tool)| ACPBuiltinToolMapping {
            tool_name: tool_name.clone(),
            native_name: tool.native_name.clone(),
            title: tool.tool.title.clone(),
            tool_use_kind: tool.tool_use_kind,
            input_schema: Some(tool.tool.input_schema.json_schema.clone()),
        })
        .collect()
}

/// The non-secret configuration a persisted turn start configuration is
/// checked against.
struct TurnStartConfigContext<'a> {
    authentication_profile: &'a ACPAuthenticationProfileIdentity,
    session_meta: Option<&'a IndexMap<String, ACPSerializableValue>>,
    instruction_mapping: Option<&'a ACPInstructionMapping>,
    output_schema_mapping: Option<&'a ACPOutputSchemaMapping>,
    model_mapping: &'a ACPModelMapping,
    builtin_tools: &'a [ACPBuiltinToolMapping],
    permission_mode_mapping: Option<&'a ACPPermissionModeMapping>,
    mcp_servers: Option<&'a IndexMap<String, JsonValue>>,
}

/// `validateACPTurnStartConfig`.
fn validate_acp_turn_start_config(
    turn_start_config: &ACPTurnStartConfig,
    context: &TurnStartConfigContext<'_>,
) -> HarnessResult<()> {
    let current = create_acp_turn_start_config(CreateACPTurnStartConfigOptions {
        prompt: &turn_start_config.prompt,
        tools: &turn_start_config.tools,
        builtin_tools: context.builtin_tools,
        permission_mode: turn_start_config.permission_mode,
        permission_mode_mapping: context.permission_mode_mapping,
        mcp_servers: context.mcp_servers,
        debug: turn_start_config.debug.as_ref(),
        authentication_profile: context.authentication_profile,
        session_meta: context.session_meta,
        instruction_mapping: context.instruction_mapping,
        response_format: turn_start_config.response_format.as_ref(),
        output_schema_mapping: context.output_schema_mapping,
        model: turn_start_config.model.as_deref(),
        model_mapping: context.model_mapping,
    })?;
    if current.configuration_fingerprint != turn_start_config.configuration_fingerprint {
        return Err(plain_error(
            "The persisted ACP turn start configuration is incompatible with the current non-secret start configuration.",
        ));
    }
    Ok(())
}

/// `validateACPColdSessionConfiguration`.
fn validate_acp_cold_session_configuration(
    cold_session: &ACPColdSessionState,
    permission_mode: HarnessV1PermissionMode,
    debug: Option<&HarnessV1DebugConfig>,
    context: &TurnStartConfigContext<'_>,
) -> HarnessResult<ACPTurnStartConfig> {
    let current = create_acp_turn_start_config(CreateACPTurnStartConfigOptions {
        prompt: &[],
        tools: &cold_session.tools,
        builtin_tools: context.builtin_tools,
        permission_mode,
        permission_mode_mapping: context.permission_mode_mapping,
        mcp_servers: context.mcp_servers,
        debug,
        authentication_profile: context.authentication_profile,
        session_meta: context.session_meta,
        instruction_mapping: context.instruction_mapping,
        response_format: cold_session.response_format.as_ref(),
        output_schema_mapping: context.output_schema_mapping,
        model: None,
        model_mapping: context.model_mapping,
    })?;
    if current.configuration_fingerprint != cold_session.configuration_fingerprint
        || cold_session.permission_mode != permission_mode
    {
        return Err(plain_error(
            "ACP cold-session state is incompatible with the current non-secret session configuration.",
        ));
    }
    Ok(current)
}

/// `assertRecoveryToolCatalog`.
fn assert_recovery_tool_catalog(
    persisted: &[HarnessV1ToolSpec],
    current: &[HarnessV1ToolSpec],
) -> HarnessResult<()> {
    if fingerprint_value(&json!(persisted)) != fingerprint_value(&json!(current)) {
        return Err(plain_error(
            "ACP lossy rerun requires the same active host tool catalog as the original turn.",
        ));
    }
    Ok(())
}

/// `fingerprintValue`.
fn fingerprint_value(value: &Value) -> String {
    crate::harness::agent::internal::sha256_hex(&value.to_string())
}

// ─── Session ───────────────────────────────────────────────────────────

/// The inputs of `createSession`.
struct ACPSessionConfig {
    session_id: String,
    harness_id: String,
    channel: ACPChannel,
    proc: Option<Arc<dyn SandboxProcessControl>>,
    model_mapping: ACPModelMapping,
    session_meta: Option<IndexMap<String, ACPSerializableValue>>,
    instruction_mapping: Option<ACPInstructionMapping>,
    output_schema_mapping: Option<ACPOutputSchemaMapping>,
    ask_user_questions: Option<ACPAskUserQuestionsSettings>,
    debug: Option<HarnessV1DebugConfig>,
    agent_identity: String,
    authentication_profile: ACPAuthenticationProfileIdentity,
    builtin_tools: Vec<ACPBuiltinToolMapping>,
    permission_mode: HarnessV1PermissionMode,
    permission_mode_mapping: Option<ACPPermissionModeMapping>,
    mcp_servers: Option<IndexMap<String, JsonValue>>,
    is_mcp_tool_call: ACPIsMcpToolCall,
    resolve_host_tool_call: ACPResolveHostToolCall,
    initial_guidance_applied: bool,
    instructions_fingerprint: Option<String>,
    sandbox: Arc<dyn SandboxSession>,
    home_path: String,
    skills_dir: String,
    skills_directory: String,
    acp_session_id: Option<String>,
    bridge_port: u16,
    bridge_private_key: String,
    sandbox_id: Option<String>,
    is_resume: bool,
    turn_in_flight: bool,
    bridge_state_dir: String,
    turn_start_config: Option<ACPTurnStartConfig>,
    recovery_status: Option<ACPRecoveryStatus>,
    restoration: Option<ACPRestoration>,
    sandbox_credential_environment: Option<IndexMap<String, String>>,
    replay_only: bool,
    lossy_rerun: bool,
}

/// A buffered `askUserQuestions` result.
#[derive(Debug, Clone)]
struct BufferedQuestionResult {
    is_error: Option<bool>,
    tool_result: ToolResultPart,
}

/// The mutable state of `createSession`.
struct SessionState {
    stopped: bool,
    turn_in_flight: bool,
    initial_guidance_applied: bool,
    instructions_fingerprint: Option<String>,
    latest_acp_session_id: Option<String>,
    latest_turn_start_config: Option<ACPTurnStartConfig>,
    buffered_question_results: IndexMap<String, BufferedQuestionResult>,
}

struct SessionShared {
    config: ACPSessionConfig,
    state: Mutex<SessionState>,
}

impl SessionShared {
    fn mark_turn_finished(&self) {
        self.state.lock().turn_in_flight = false;
    }

    fn send(&self, message: ACPInboundMessage) -> HarnessResult<()> {
        self.config.channel.send(&message, send_options())
    }
}

/// The session `createSession` returns.
struct ACPSession {
    shared: Arc<SessionShared>,
}

impl ACPSession {
    fn new(config: ACPSessionConfig) -> Self {
        let state = SessionState {
            stopped: false,
            turn_in_flight: config.turn_in_flight,
            initial_guidance_applied: config.initial_guidance_applied,
            instructions_fingerprint: config.instructions_fingerprint.clone(),
            latest_acp_session_id: config.acp_session_id.clone(),
            latest_turn_start_config: config.turn_start_config.clone(),
            buffered_question_results: IndexMap::new(),
        };
        let turn_in_flight_at_start = config.turn_in_flight;
        let shared = Arc::new(SessionShared {
            config,
            state: Mutex::new(state),
        });
        let channel = &shared.config.channel;
        {
            let weak = Arc::downgrade(&shared);
            // The subscription lives as long as the channel.
            std::mem::forget(
                channel.on("bridge-thread", move |frame: &ACPOutboundFrame| {
                    if let (
                        Some(shared),
                        ACPOutboundMessage::Bridge(HarnessV1BridgeOutboundMessage::Thread(thread)),
                    ) = (weak.upgrade(), &frame.message)
                    {
                        shared.state.lock().latest_acp_session_id = Some(thread.thread_id.clone());
                    }
                }),
            );
        }
        // A resumed continuation replays its buffered events after the new
        // session is created. Let `wire_turn` consume those events so it can
        // forward a terminal replay instead of rejecting the continuation
        // before it starts.
        if !turn_in_flight_at_start {
            for event in ["finish", "error"] {
                let weak = Arc::downgrade(&shared);
                std::mem::forget(channel.on(event, move |_: &ACPOutboundFrame| {
                    if let Some(shared) = weak.upgrade() {
                        shared.mark_turn_finished();
                    }
                }));
            }
            let weak = Arc::downgrade(&shared);
            channel.on_close(move |_, _| {
                if let Some(shared) = weak.upgrade() {
                    shared.mark_turn_finished();
                }
            });
        }
        Self { shared }
    }

    fn harness_id(&self) -> &str {
        &self.shared.config.harness_id
    }

    async fn synchronize_skills(
        &self,
        skills: &[HarnessV1Skill],
        abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<()> {
        validate_acp_skills(skills)?;
        let config = &self.shared.config;
        let mut options = WriteSkillsOptions::new(
            &*config.sandbox,
            config.home_path.clone(),
            config.skills_dir.clone(),
            skills.to_vec(),
        );
        options.abort_signal = abort_signal;
        options.skill_name_pattern = Some(ACP_SKILL_NAME_PATTERN.clone());
        options.invalid_skill_name_message = Some(Arc::new(|name: &str| {
            format!(
                "Invalid ACP skill name {}: expected a kebab-case slug.",
                Value::String(name.to_string())
            )
        }));
        options.invalid_skill_file_path_message = Some(Arc::new(
            |skill_name: &str, file_path: &str| {
                format!(
                    "Invalid ACP skill file path {} for skill {}: expected a relative POSIX path without traversal.",
                    Value::String(file_path.to_string()),
                    Value::String(skill_name.to_string())
                )
            },
        ));
        write_skills(options).await?;
        Ok(())
    }

    async fn synchronize_instructions(
        &self,
        instructions: Option<&str>,
        abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<()> {
        let config = &self.shared.config;
        if let Some(ACPInstructionMapping::Filesystem { path }) = &config.instruction_mapping {
            write_instructions(WriteInstructionsOptions {
                sandbox: &*config.sandbox,
                home_path: config.home_path.clone(),
                instructions_file: path.clone(),
                instructions: instructions.map(str::to_string),
                abort_signal,
            })
            .await?;
        }
        Ok(())
    }

    fn check_response_format(
        &self,
        response_format: Option<&HarnessV1ResponseFormat>,
    ) -> HarnessResult<()> {
        let harness_id = self.harness_id();
        if let Some(HarnessV1ResponseFormat::Json { schema, .. }) = response_format {
            if schema.is_none() {
                return Err(unsupported(
                    harness_id,
                    format!("{harness_id} requires a JSON schema for structured output."),
                    None,
                ));
            }
            if self.shared.config.output_schema_mapping.is_none() {
                return Err(unsupported(
                    harness_id,
                    format!("{harness_id} does not support structured output through ACP."),
                    None,
                ));
            }
        }
        Ok(())
    }

    fn create_lifecycle_data(
        &self,
        bridge: Option<ACPBridgeCoordinates>,
        include_turn_start_config: bool,
    ) -> JsonValue {
        let config = &self.shared.config;
        let state = self.shared.state.lock();
        let data = ACPLifecycleData {
            agent_identity: config.agent_identity.clone(),
            authentication_profile: Some(config.authentication_profile.clone()),
            sandbox_credential_environment: config.sandbox_credential_environment.clone(),
            acp_session_id: state.latest_acp_session_id.clone(),
            bridge,
            cold_session: state
                .latest_turn_start_config
                .as_ref()
                .map(create_acp_cold_session_state),
            turn_start_config: state
                .latest_turn_start_config
                .clone()
                .filter(|_| include_turn_start_config),
            recovery: config.recovery_status.clone(),
            restoration: config.restoration,
            guidance: ACPPromptGuidanceLifecycleState {
                initial_guidance_applied: Some(state.initial_guidance_applied),
                instructions_fingerprint: state.instructions_fingerprint.clone(),
                skills_directory: Some(config.skills_directory.clone()),
            },
        };
        serde_json::to_value(data).unwrap_or(Value::Null)
    }

    fn bridge_coordinates(&self, last_seen_event_id: u64) -> ACPBridgeCoordinates {
        let config = &self.shared.config;
        ACPBridgeCoordinates {
            port: config.bridge_port,
            private_key: config.bridge_private_key.clone(),
            last_seen_event_id,
            sandbox_id: config.sandbox_id.clone(),
            state_dir: Some(config.bridge_state_dir.clone()),
        }
    }

    /// `terminateBridge`.
    async fn terminate_bridge(&self, command: HarnessV1BridgeInboundCommand) {
        {
            let mut state = self.shared.state.lock();
            if state.stopped {
                return;
            }
            state.stopped = true;
        }
        let config = &self.shared.config;
        let channel = &config.channel;
        channel.begin_close();
        let (closed_tx, closed_rx) = oneshot::channel::<()>();
        let closed_tx = Mutex::new(Some(closed_tx));
        channel.on_close(move |_, _| {
            if let Some(closed_tx) = closed_tx.lock().take() {
                let _ = closed_tx.send(());
            }
        });
        if !channel.is_closed() {
            let _ = channel.send(&command.into(), send_options());
        }
        let wait: BoxFuture<'_, ()> = match &config.proc {
            Some(proc) => async move {
                let _ = proc.wait().await;
            }
            .boxed(),
            None => async move {
                let _ = closed_rx.await;
            }
            .boxed(),
        };
        let _ = tokio::time::timeout(Duration::from_millis(5000), wait).await;
        if let Some(proc) = &config.proc {
            let _ = proc.kill().await;
        }
        channel.close();
    }

    /// `wireTurn`.
    fn wire_turn(
        &self,
        emit: HarnessV1Emit,
        abort_signal: Option<CancellationToken>,
        start: impl FnOnce() -> HarnessResult<()>,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        let (done_tx, done_rx) = oneshot::channel::<Result<(), ErrorValue>>();
        let done: Shared<BoxFuture<'static, Result<(), ErrorValue>>> = async move {
            done_rx
                .await
                .unwrap_or_else(|_| Err(ErrorValue::message("ACP turn was abandoned.")))
        }
        .boxed()
        .shared();
        let turn = Arc::new(Turn {
            session: self.shared.clone(),
            emit,
            abort_signal: abort_signal.clone(),
            done_tx: Mutex::new(Some(done_tx)),
            state: Mutex::new(TurnState::default()),
        });
        let channel = &self.shared.config.channel;
        let mut subscriptions = Vec::new();

        subscriptions.push(channel.on("acp-tool-call-candidate", {
            let turn = turn.clone();
            move |frame: &ACPOutboundFrame| {
                if let ACPOutboundMessage::ToolCallCandidate(event) = &frame.message {
                    turn.on_tool_call_candidate(event);
                }
            }
        }));
        subscriptions.push(channel.on("acp-host-tool-call-resolution-request", {
            let turn = turn.clone();
            move |frame: &ACPOutboundFrame| {
                if let ACPOutboundMessage::HostToolCallResolutionRequest(event) = &frame.message {
                    turn.on_host_tool_call_resolution_request(event);
                }
            }
        }));
        subscriptions.push(channel.on("acp-question-request", {
            let turn = turn.clone();
            move |frame: &ACPOutboundFrame| {
                if let ACPOutboundMessage::QuestionRequest(event) = &frame.message {
                    turn.on_question_request(event);
                }
            }
        }));
        subscriptions.push(channel.on("acp-question-resolved", {
            let turn = turn.clone();
            move |frame: &ACPOutboundFrame| {
                if let ACPOutboundMessage::QuestionResolved(event) = &frame.message {
                    let mut state = turn.state.lock();
                    if let Some(tool_call_id) = state
                        .question_tool_call_ids_by_request_id
                        .remove(&event.request_id)
                    {
                        state.active_question_requests.remove(&tool_call_id);
                    }
                }
            }
        }));
        subscriptions.push(channel.on("tool-call", {
            let turn = turn.clone();
            move |frame: &ACPOutboundFrame| {
                if let Some(part) = frame_part(frame) {
                    turn.on_tool_call(part);
                }
            }
        }));
        subscriptions.push(channel.on("tool-result", {
            let turn = turn.clone();
            move |frame: &ACPOutboundFrame| {
                if let Some(part) = frame_part(frame) {
                    turn.on_tool_result(part);
                }
            }
        }));
        for event_type in [
            "stream-start",
            "text-start",
            "text-delta",
            "text-end",
            "reasoning-start",
            "reasoning-delta",
            "reasoning-end",
            "tool-approval-request",
            "file-change",
            "finish-step",
            "raw",
        ] {
            subscriptions.push(channel.on(event_type, {
                let turn = turn.clone();
                move |frame: &ACPOutboundFrame| {
                    if let Some(part) = frame_part(frame) {
                        turn.forward(part);
                    }
                }
            }));
        }
        subscriptions.push(channel.on("finish", {
            let turn = turn.clone();
            move |frame: &ACPOutboundFrame| {
                turn.session.mark_turn_finished();
                turn.close_forwarded_block();
                if let Some(part) = frame_part(frame) {
                    turn.forward(part);
                }
                let error = {
                    let state = turn.state.lock();
                    state
                        .abort_requested
                        .then(|| state.abort_error.clone())
                        .flatten()
                };
                turn.settle(error);
            }
        }));
        subscriptions.push(channel.on("error", {
            let turn = turn.clone();
            move |frame: &ACPOutboundFrame| {
                turn.session.mark_turn_finished();
                turn.close_forwarded_block();
                let Some(HarnessV1StreamPart::Error { error }) = frame.message.stream_part() else {
                    return;
                };
                let error = deserialize_bridge_error(error, &turn.session.config.harness_id);
                turn.forward(HarnessV1StreamPart::Error {
                    error: error.clone(),
                });
                turn.settle(Some(error));
            }
        }));
        turn.state.lock().subscriptions = subscriptions;
        {
            let weak = Arc::downgrade(&turn);
            channel.on_close(move |_code, reason| {
                let Some(turn) = weak.upgrade() else { return };
                turn.session.mark_turn_finished();
                if reason == "suspended" {
                    turn.settle(None);
                    return;
                }
                turn.close_forwarded_block();
                let error = {
                    let state = turn.state.lock();
                    match (&state.abort_error, state.abort_requested) {
                        (Some(error), true) => error.clone(),
                        _ => ErrorValue::message(format!(
                            "{} ACP bridge closed before turn end.",
                            turn.session.config.harness_id
                        )),
                    }
                };
                turn.settle(Some(error));
            });
        }
        if let Err(error) = start() {
            turn.state.lock().settled = true;
            turn.cleanup();
            return Err(error);
        }
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
        Ok(Box::new(ACPPromptControl { turn, done }))
    }
}

/// A stream part frame as the turn forwards it: a `finish-step` carries the
/// frame's event checkpoint, the non-enumerable property TS keeps.
fn frame_part(frame: &ACPOutboundFrame) -> Option<HarnessV1StreamPart> {
    let mut part = frame.message.stream_part()?.clone();
    if let HarnessV1StreamPart::FinishStep { checkpoint, .. } = &mut part {
        *checkpoint = frame.event_checkpoint().cloned();
    }
    Some(part)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockType {
    Text,
    Reasoning,
}

#[derive(Default)]
struct TurnState {
    settled: bool,
    abort_requested: bool,
    abort_error: Option<ErrorValue>,
    open_block: Option<(BlockType, String)>,
    dynamic_tool_calls: HashMap<String, bool>,
    tool_call_classification_errors: HashMap<String, ErrorValue>,
    /// Tool call id -> (request id, native request).
    active_question_requests: HashMap<String, (String, JsonValue)>,
    question_tool_call_ids_by_request_id: HashMap<String, String>,
    subscriptions: Vec<crate::harness::utils::Unsubscribe>,
    abort_task: Option<tokio::task::JoinHandle<()>>,
}

/// One `wireTurn` call.
struct Turn {
    session: Arc<SessionShared>,
    emit: HarnessV1Emit,
    #[allow(dead_code)]
    abort_signal: Option<CancellationToken>,
    done_tx: Mutex<Option<oneshot::Sender<Result<(), ErrorValue>>>>,
    state: Mutex<TurnState>,
}

impl Turn {
    fn harness_id(&self) -> &str {
        &self.session.config.harness_id
    }

    fn send(&self, message: ACPInboundMessage) -> HarnessResult<()> {
        self.session.send(message)
    }

    fn forward(&self, event: HarnessV1StreamPart) {
        {
            let mut state = self.state.lock();
            match &event {
                HarnessV1StreamPart::TextStart { id, .. } => {
                    state.open_block = Some((BlockType::Text, id.clone()));
                }
                HarnessV1StreamPart::ReasoningStart { id, .. } => {
                    state.open_block = Some((BlockType::Reasoning, id.clone()));
                }
                HarnessV1StreamPart::TextEnd { id, .. }
                | HarnessV1StreamPart::ReasoningEnd { id, .. }
                    if state
                        .open_block
                        .as_ref()
                        .is_some_and(|(_, open)| open == id) =>
                {
                    state.open_block = None;
                }
                _ => {}
            }
        }
        (self.emit)(event);
    }

    fn close_forwarded_block(&self) {
        let Some((block_type, id)) = self.state.lock().open_block.take() else {
            return;
        };
        self.forward(match block_type {
            BlockType::Text => HarnessV1StreamPart::TextEnd {
                id,
                harness_metadata: None,
            },
            BlockType::Reasoning => HarnessV1StreamPart::ReasoningEnd {
                id,
                harness_metadata: None,
            },
        });
    }

    fn cleanup(&self) {
        let (subscriptions, abort_task) = {
            let mut state = self.state.lock();
            (
                std::mem::take(&mut state.subscriptions),
                state.abort_task.take(),
            )
        };
        for unsubscribe in subscriptions {
            unsubscribe();
        }
        if let Some(task) = abort_task {
            task.abort();
        }
    }

    fn settle(&self, error: Option<ErrorValue>) {
        {
            let mut state = self.state.lock();
            if state.settled {
                return;
            }
            state.settled = true;
        }
        self.cleanup();
        if let Some(done_tx) = self.done_tx.lock().take() {
            let _ = done_tx.send(match error {
                None => Ok(()),
                Some(error) => Err(error),
            });
        }
    }

    /// Forward `error`, abort the bridge turn and reject.
    fn fail(&self, error: ErrorValue) {
        self.close_forwarded_block();
        self.forward(HarnessV1StreamPart::Error {
            error: error.clone(),
        });
        let _ = self.send(HarnessV1BridgeInboundCommand::Abort.into());
        self.settle(Some(error));
    }

    fn on_abort(&self) {
        let abort_error = ErrorValue::from_box(AbortError.into());
        {
            let mut state = self.state.lock();
            if state.settled || state.abort_requested {
                return;
            }
            state.abort_requested = true;
            state.abort_error = Some(abort_error.clone());
        }
        if self
            .send(HarnessV1BridgeInboundCommand::Abort.into())
            .is_err()
        {
            self.settle(Some(abort_error));
        }
    }

    fn on_tool_call_candidate(&self, event: &super::bridge_protocol::ACPToolCallCandidate) {
        let config = &self.session.config;
        let classified = (|| -> HarnessResult<bool> {
            let suppress = match config
                .ask_user_questions
                .as_ref()
                .and_then(|questions| questions.is_native_tool_call.as_ref())
            {
                Some(is_native_tool_call) => is_native_tool_call(&event.tool_call)?,
                None => false,
            };
            let dynamic = (config.is_mcp_tool_call)(&event.tool_call)?;
            self.state
                .lock()
                .dynamic_tool_calls
                .insert(event.tool_call.tool_call_id.clone(), dynamic);
            Ok(suppress)
        })();
        let suppress = match classified {
            Ok(suppress) => suppress,
            Err(error) => {
                self.state.lock().tool_call_classification_errors.insert(
                    event.tool_call.tool_call_id.clone(),
                    ErrorValue::from_box(error),
                );
                false
            }
        };
        let _ = self.send(tool_result_command(
            event.request_id.clone(),
            json!({ "suppress": suppress }),
            None,
            None,
        ));
    }

    fn on_host_tool_call_resolution_request(
        &self,
        event: &super::bridge_protocol::ACPHostToolCallResolutionRequest,
    ) {
        let harness_id = self.harness_id().to_string();
        let resolved = (self.session.config.resolve_host_tool_call)(ACPResolveHostToolCallOptions {
            tool_call: &event.tool_call,
            server_name: &event.server_name,
            tool_names: &event.tool_names,
        })
        .and_then(|host_tool_call| match host_tool_call {
            Some(call) if !event.tool_names.contains(&call.tool_name) => Err(plain_error(format!(
                "{harness_id} ACP resolveHostToolCall must return undefined or one of the provided toolNames with an object input."
            ))),
            other => Ok(other),
        });
        match resolved {
            Err(error) => {
                let _ = self.send(tool_result_command(
                    event.request_id.clone(),
                    json!({ "hostToolCall": null }),
                    None,
                    None,
                ));
                self.fail(ErrorValue::from_box(error));
            }
            Ok(host_tool_call) => {
                let _ = self.send(tool_result_command(
                    event.request_id.clone(),
                    json!({
                        "hostToolCall": host_tool_call.map(|call| json!({
                            "toolName": call.tool_name,
                            "input": call.input,
                        })),
                    }),
                    None,
                    None,
                ));
            }
        }
    }

    fn on_question_request(&self, event: &super::bridge_protocol::ACPQuestionRequest) {
        let harness_id = self.harness_id().to_string();
        let Some(ask_user_questions) = self.session.config.ask_user_questions.clone() else {
            let _ = self.send(tool_result_command(
                event.request_id.clone(),
                json!({ "type": "unhandled" }),
                None,
                None,
            ));
            return;
        };
        let result = (|| -> HarnessResult<()> {
            let native_tool_call =
                (ask_user_questions.from_native_request)(ACPFromNativeRequestOptions {
                    native_request: &event.native_request,
                    native_tool_call: event.native_tool_call.as_ref(),
                })?;
            let Some(native_tool_call) = native_tool_call else {
                self.send(tool_result_command(
                    event.request_id.clone(),
                    json!({ "type": "unhandled" }),
                    None,
                    None,
                ))?;
                return Ok(());
            };
            let tool_call = match native_tool_call {
                HarnessV1StreamPart::ToolCall {
                    ref tool_name,
                    provider_executed: Some(false),
                    ..
                } if tool_name == "askUserQuestions" => with_native_question_request(
                    &harness_id,
                    &event.native_request,
                    native_tool_call,
                ),
                _ => {
                    return Err(plain_error(format!(
                        "{harness_id} ACP askUserQuestions.fromNativeRequest must return a client-executed askUserQuestions tool call."
                    )));
                }
            };
            let HarnessV1StreamPart::ToolCall { tool_call_id, .. } = &tool_call else {
                unreachable!("checked above");
            };
            let tool_call_id = tool_call_id.clone();
            let buffered_result = take_buffered_question_result(
                &mut self.session.state.lock().buffered_question_results,
                &tool_call_id,
                &event.native_request,
                ask_user_questions.matches_native_request.as_ref(),
                &harness_id,
            );
            {
                let mut state = self.state.lock();
                state.active_question_requests.insert(
                    tool_call_id.clone(),
                    (event.request_id.clone(), event.native_request.clone()),
                );
                state
                    .question_tool_call_ids_by_request_id
                    .insert(event.request_id.clone(), tool_call_id.clone());
            }
            self.send(tool_result_command(
                event.request_id.clone(),
                json!({ "type": "handled", "toolCallId": tool_call_id }),
                None,
                None,
            ))?;

            let Some(buffered_result) = buffered_result else {
                self.forward(tool_call);
                return Ok(());
            };
            let output = (ask_user_questions.to_native_response)(ACPToNativeResponseOptions {
                native_request: &event.native_request,
                tool_result: &buffered_result.tool_result,
            })?;
            self.send(tool_result_command(
                tool_call_id,
                output,
                buffered_result.is_error,
                Some(&buffered_result.tool_result),
            ))?;
            Ok(())
        })();
        if let Err(error) = result {
            self.fail(ErrorValue::from_box(error));
        }
    }

    fn on_tool_call(&self, event: HarnessV1StreamPart) {
        let HarnessV1StreamPart::ToolCall { tool_call_id, .. } = &event else {
            return;
        };
        let (classification_error, dynamic) = {
            let state = self.state.lock();
            (
                state
                    .tool_call_classification_errors
                    .get(tool_call_id)
                    .cloned(),
                state.dynamic_tool_calls.get(tool_call_id) == Some(&true),
            )
        };
        if let Some(error) = classification_error {
            self.fail(error);
            return;
        }
        let mut event = event;
        if dynamic && let HarnessV1StreamPart::ToolCall { dynamic, .. } = &mut event {
            *dynamic = Some(true);
        }
        self.forward(event);
    }

    fn on_tool_result(&self, event: HarnessV1StreamPart) {
        let HarnessV1StreamPart::ToolResult { tool_call_id, .. } = &event else {
            return;
        };
        let dynamic = {
            let mut state = self.state.lock();
            state.tool_call_classification_errors.remove(tool_call_id);
            state.dynamic_tool_calls.remove(tool_call_id) == Some(true)
        };
        let mut event = event;
        if dynamic && let HarnessV1StreamPart::ToolResult { dynamic, .. } = &mut event {
            *dynamic = Some(true);
        }
        self.forward(event);
    }
}

/// `withNativeQuestionRequest`.
fn with_native_question_request(
    harness_id: &str,
    native_request: &JsonValue,
    mut tool_call: HarnessV1StreamPart,
) -> HarnessV1StreamPart {
    if let HarnessV1StreamPart::ToolCall {
        provider_metadata, ..
    } = &mut tool_call
    {
        let metadata = provider_metadata.get_or_insert_with(IndexMap::new);
        let mut harness_metadata: Map<String, Value> =
            metadata.get(harness_id).cloned().unwrap_or_default();
        harness_metadata.insert("nativeRequest".into(), native_request.clone());
        metadata.insert(harness_id.to_string(), harness_metadata);
    }
    tool_call
}

fn previous_native_request<'a>(
    tool_result: &'a ToolResultPart,
    harness_id: &str,
) -> Option<&'a JsonValue> {
    tool_result
        .provider_options
        .as_ref()?
        .get(harness_id)?
        .get("nativeRequest")
}

/// `takeBufferedQuestionResult`.
fn take_buffered_question_result(
    buffered_question_results: &mut IndexMap<String, BufferedQuestionResult>,
    tool_call_id: &str,
    native_request: &JsonValue,
    matches_native_request: Option<&ACPMatchesNativeRequest>,
    harness_id: &str,
) -> Option<BufferedQuestionResult> {
    if let Some(exact) = buffered_question_results.shift_remove(tool_call_id) {
        return Some(exact);
    }
    let matches_native_request = matches_native_request?;
    let key = buffered_question_results
        .iter()
        .find(|(_, buffered)| {
            previous_native_request(&buffered.tool_result, harness_id).is_some_and(|previous| {
                matches_native_request(ACPMatchesNativeRequestOptions {
                    previous_native_request: previous,
                    native_request,
                })
            })
        })
        .map(|(key, _)| key.clone())?;
    buffered_question_results.shift_remove(&key)
}

/// The control `wireTurn` returns.
struct ACPPromptControl {
    turn: Arc<Turn>,
    done: Shared<BoxFuture<'static, Result<(), ErrorValue>>>,
}

#[async_trait]
impl HarnessV1PromptControl for ACPPromptControl {
    async fn submit_tool_result(&self, input: HarnessV1SubmitToolResult) -> HarnessResult<()> {
        let session = &self.turn.session;
        let config = &session.config;
        let harness_id = config.harness_id.as_str();
        if let (Some(ask_user_questions), Some(tool_result)) =
            (&config.ask_user_questions, &input.tool_result)
            && tool_result.tool_name == "askUserQuestions"
        {
            let active_request = self
                .turn
                .state
                .lock()
                .active_question_requests
                .get(&input.tool_call_id)
                .cloned();
            let Some((_, native_request)) = active_request else {
                if let Some(previous) = previous_native_request(tool_result, harness_id)
                    && !config.lossy_rerun
                {
                    let output =
                        (ask_user_questions.to_native_response)(ACPToNativeResponseOptions {
                            native_request: previous,
                            tool_result,
                        })?;
                    return session.send(tool_result_command(
                        input.tool_call_id.clone(),
                        output,
                        input.is_error,
                        Some(tool_result),
                    ));
                }
                session.state.lock().buffered_question_results.insert(
                    input.tool_call_id.clone(),
                    BufferedQuestionResult {
                        is_error: input.is_error,
                        tool_result: tool_result.clone(),
                    },
                );
                return Ok(());
            };
            let output = (ask_user_questions.to_native_response)(ACPToNativeResponseOptions {
                native_request: &native_request,
                tool_result,
            })?;
            return session.send(tool_result_command(
                input.tool_call_id.clone(),
                output,
                input.is_error,
                Some(tool_result),
            ));
        }
        session.send(tool_result_command(
            input.tool_call_id.clone(),
            input.output.clone(),
            input.is_error,
            input.tool_result.as_ref(),
        ))
    }

    fn supports_submit_tool_approval(&self) -> bool {
        true
    }

    async fn submit_tool_approval(&self, input: HarnessV1SubmitToolApproval) -> HarnessResult<()> {
        self.turn.send(
            HarnessV1BridgeInboundCommand::ToolApprovalResponse {
                approval_id: input.approval_id,
                approved: input.approved,
                reason: input.reason,
            }
            .into(),
        )
    }

    fn done(&self) -> BoxFuture<'static, HarnessResult<()>> {
        let done = self.done.clone();
        async move { done.await.map_err(|error| error.to_box()) }.boxed()
    }
}

#[async_trait]
impl HarnessV1Session for ACPSession {
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
        let harness_id = self.harness_id().to_string();
        self.synchronize_skills(&options.settings.skills, options.abort_signal.clone())
            .await?;
        self.synchronize_instructions(
            options.settings.instructions.as_deref(),
            options.abort_signal.clone(),
        )
        .await?;
        self.check_response_format(options.response_format.as_ref())?;
        let config = &self.shared.config;
        if config.replay_only {
            return Err(plain_error(format!(
                "{harness_id} recovered this turn through disk replay only and has no restored ACP process for a subsequent prompt."
            )));
        }
        check_aborted(options.abort_signal.as_ref())?;
        let prompt = convert_harness_prompt_to_acp_content_blocks(&options.prompt, &harness_id)?;
        let model = options.settings.model.clone();
        let turn_start_config = create_acp_turn_start_config(CreateACPTurnStartConfigOptions {
            prompt: &prompt,
            tools: &options.settings.tools,
            builtin_tools: &config.builtin_tools,
            permission_mode: config.permission_mode,
            permission_mode_mapping: config.permission_mode_mapping.as_ref(),
            mcp_servers: config.mcp_servers.as_ref(),
            debug: config.debug.as_ref(),
            authentication_profile: &config.authentication_profile,
            session_meta: config.session_meta.as_ref(),
            instruction_mapping: config.instruction_mapping.as_ref(),
            response_format: options.response_format.as_ref(),
            output_schema_mapping: config.output_schema_mapping.as_ref(),
            model: model.as_deref(),
            model_mapping: &config.model_mapping,
        })?;
        let next_instructions_fingerprint = fingerprint_value(
            &options
                .settings
                .instructions
                .clone()
                .map_or(Value::Null, Value::String),
        );
        let shared = self.shared.clone();
        let instructions = options.settings.instructions.clone();
        let response_format = options.response_format.clone();
        self.wire_turn(
            options.emit.clone(),
            options.abort_signal.clone(),
            move || {
                let config = &shared.config;
                let prompt = {
                    let mut state = shared.state.lock();
                    if state.turn_in_flight {
                        return Err(plain_error(format!(
                            "{harness_id} cannot start a new ACP prompt while a turn is in flight."
                        )));
                    }
                    state.turn_in_flight = true;
                    let filesystem = matches!(
                        config.instruction_mapping,
                        Some(ACPInstructionMapping::Filesystem { .. })
                    );
                    if !filesystem
                        && state.instructions_fingerprint.as_deref()
                            != Some(next_instructions_fingerprint.as_str())
                        && (config.instruction_mapping.is_none() || state.initial_guidance_applied)
                    {
                        prepend_acp_instruction_guidance(&prompt, instructions.as_deref())
                    } else {
                        prompt.clone()
                    }
                };
                let mut base = HarnessV1BridgeStartBase::new(prompt);
                base.debug = config.debug.clone();
                base.permission_mode = Some(config.permission_mode);
                base.model = model.clone();
                base.response_format = response_format.clone();
                base.tools = Some(tool_wires(&turn_start_config.tools));
                let start = ACPStartMessage {
                    base,
                    instructions: config
                        .instruction_mapping
                        .as_ref()
                        .and(instructions.clone()),
                    instruction_mapping: config.instruction_mapping.clone(),
                    output_schema_mapping: turn_start_config.output_schema_mapping.clone(),
                    mcp_servers: config.mcp_servers.clone(),
                    builtin_tools: config.builtin_tools.clone(),
                    permission_mode_mapping: config.permission_mode_mapping.clone(),
                    model_mapping: Some(config.model_mapping.clone()),
                    turn_start_config: turn_start_config.clone(),
                    recovery_mode: None,
                };
                shared.send(start.into())?;
                let mut state = shared.state.lock();
                state.latest_turn_start_config = Some(turn_start_config);
                state.initial_guidance_applied = true;
                state.instructions_fingerprint = Some(next_instructions_fingerprint);
                Ok(())
            },
        )
    }

    async fn do_compact(&self, _custom_instructions: Option<String>) -> HarnessResult<()> {
        Err(unsupported(
            self.harness_id(),
            "ACP v1 does not define manual session compaction.",
            None,
        ))
    }

    async fn do_continue_turn(
        &self,
        options: HarnessV1ContinueTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        let harness_id = self.harness_id().to_string();
        self.synchronize_skills(&options.settings.skills, options.abort_signal.clone())
            .await?;
        self.synchronize_instructions(
            options.settings.instructions.as_deref(),
            options.abort_signal.clone(),
        )
        .await?;
        self.check_response_format(options.response_format.as_ref())?;
        let config = &self.shared.config;
        let (turn_start_config, acp_session_id) = {
            let state = self.shared.state.lock();
            if !state.turn_in_flight {
                return Err(plain_error(format!(
                    "{harness_id} has no in-flight ACP turn to continue."
                )));
            }
            (
                state.latest_turn_start_config.clone(),
                state.latest_acp_session_id.clone(),
            )
        };
        if config.lossy_rerun {
            let (Some(turn_start_config), Some(_)) = (&turn_start_config, &acp_session_id) else {
                return Err(plain_error(format!(
                    "{harness_id} cannot perform lossy ACP rerun without persisted start configuration and an ACP session identifier."
                )));
            };
            assert_recovery_tool_catalog(&turn_start_config.tools, &options.settings.tools)?;
        }
        let shared = self.shared.clone();
        let instructions = options.settings.instructions.clone();
        self.wire_turn(
            options.emit.clone(),
            options.abort_signal.clone(),
            move || {
                let config = &shared.config;
                if !config.lossy_rerun {
                    return Ok(());
                }
                let (Some(turn_start_config), Some(acp_session_id)) =
                    (turn_start_config, acp_session_id)
                else {
                    return Ok(());
                };
                let mut base = HarnessV1BridgeStartBase::new(turn_start_config.prompt.clone());
                base.debug = turn_start_config.debug.clone();
                base.permission_mode = Some(turn_start_config.permission_mode);
                base.model = turn_start_config.model.clone();
                base.response_format = turn_start_config.response_format.clone();
                base.tools = Some(tool_wires(&turn_start_config.tools));
                let start = ACPStartMessage {
                    base,
                    instructions: config
                        .instruction_mapping
                        .as_ref()
                        .and(instructions.clone()),
                    instruction_mapping: config.instruction_mapping.clone(),
                    output_schema_mapping: turn_start_config.output_schema_mapping.clone(),
                    mcp_servers: config.mcp_servers.clone(),
                    builtin_tools: turn_start_config.builtin_tools.clone(),
                    permission_mode_mapping: turn_start_config.permission_mode_mapping.clone(),
                    model_mapping: Some(config.model_mapping.clone()),
                    recovery_mode: Some(ACPRecoveryMode::LossyRerun {
                        acp_session_id,
                        reason: config
                            .recovery_status
                            .as_ref()
                            .map(|status| status.reason.clone())
                            .unwrap_or_else(|| "bridge process loss".into()),
                    }),
                    turn_start_config,
                };
                shared.send(start.into())
            },
        )
    }

    async fn do_suspend_turn(&self) -> HarnessResult<HarnessV1ContinueTurnState> {
        let config = &self.shared.config;
        {
            let mut state = self.shared.state.lock();
            if state.stopped {
                return Err(plain_error(format!(
                    "{} ACP session {} is stopped; cannot suspend.",
                    config.harness_id, config.session_id
                )));
            }
            state.stopped = true;
        }
        let last_seen_event_id = config.channel.suspend().await;
        Ok(HarnessV1ContinueTurnState::new(
            config.harness_id.clone(),
            self.create_lifecycle_data(Some(self.bridge_coordinates(last_seen_event_id)), true),
        ))
    }

    async fn do_detach(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        let config = &self.shared.config;
        {
            let mut state = self.shared.state.lock();
            if state.stopped {
                return Err(plain_error(format!(
                    "{} ACP session {} is stopped; cannot detach.",
                    config.harness_id, config.session_id
                )));
            }
            if state.turn_in_flight {
                return Err(plain_error(format!(
                    "{} ACP session {} has an in-flight turn; suspend it instead.",
                    config.harness_id, config.session_id
                )));
            }
            state.stopped = true;
        }
        let last_seen_event_id = config.channel.suspend().await;
        Ok(HarnessV1ResumeSessionState::new(
            config.harness_id.clone(),
            self.create_lifecycle_data(Some(self.bridge_coordinates(last_seen_event_id)), true),
        ))
    }

    async fn do_stop(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        let data = self.create_lifecycle_data(None, false);
        self.terminate_bridge(HarnessV1BridgeInboundCommand::Stop)
            .await;
        Ok(HarnessV1ResumeSessionState::new(
            self.shared.config.harness_id.clone(),
            data,
        ))
    }

    async fn do_destroy(&self) -> HarnessResult<()> {
        self.terminate_bridge(HarnessV1BridgeInboundCommand::Destroy)
            .await;
        Ok(())
    }
}

#[allow(dead_code)]
fn _assert_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<ACPSession>();
    check::<ACPV1Harness>();
    let _: Option<Weak<Turn>> = None;
}
