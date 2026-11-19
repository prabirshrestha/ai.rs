//! Port of `harness-acp/src/acp-harness.ts`.

use indexmap::IndexMap;
use serde_json::json;

use crate::harness::errors::HarnessResult;
use crate::harness::sdk::tool::{Schema, json_schema};
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
use crate::harness::spec::tools::{
    HarnessV1BuiltinTool, HarnessV1BuiltinToolName, harness_v1_builtin_tools,
};
use crate::harness::utils::{SandboxChannelReconnectOptions, WebSocketConstructor, plain_error};

use super::VERSION;
use super::acp_auth::ACPClientApp;
use super::v1::bridge_protocol::validate_turn_start_mappings;
use super::v1::harness::{ACPV1Harness, CreateACPV1Options, create_acp_v1};
use super::v1::lifecycle::ACPLifecycleData;
use super::v1::settings::ACPV1Settings;

/// `ACPHarnessSettings`: the [`ACPV1Settings`] plus the harness-level
/// options.
#[derive(Clone)]
pub struct ACPHarnessSettings {
    pub settings: ACPV1Settings,
    pub builtin_tools: Option<IndexMap<String, HarnessV1BuiltinTool>>,
    pub port: Option<u16>,
    /// Override the host endpoint used to connect to the sandbox bridge.
    /// Required together with `port` when using a basic sandbox session.
    pub port_endpoint: Option<HarnessV1PortEndpoint>,
    pub startup_timeout_ms: Option<u64>,
    /// Abort a running turn when its host disconnects and does not reconnect
    /// within this many milliseconds, so the turn stops spending. A suspended
    /// turn is exempt. Unset keeps the turn running until it finishes.
    pub host_disconnect_timeout_ms: Option<u64>,
    /// Configures reconnection attempts after an established bridge
    /// connection drops.
    pub reconnect: Option<SandboxChannelReconnectOptions>,
    /// WebSocket ping interval in milliseconds. Defaults to 20_000. Set to 0
    /// to disable.
    pub keep_alive_ms: Option<u64>,
    pub client_app: Option<ACPClientApp>,
    /// [rust] Test seam standing in for the `ws` module TS tests mock.
    pub(crate) web_socket_constructor: Option<WebSocketConstructor>,
    /// [rust] Test seam standing in for `vi.stubEnv`: replaces `process.env`.
    pub(crate) process_env: Option<IndexMap<String, String>>,
}

impl ACPHarnessSettings {
    /// Settings with every harness-level option unset.
    pub fn new(settings: ACPV1Settings) -> Self {
        Self {
            settings,
            builtin_tools: None,
            port: None,
            port_endpoint: None,
            startup_timeout_ms: None,
            host_disconnect_timeout_ms: None,
            reconnect: None,
            keep_alive_ms: None,
            client_app: None,
            web_socket_constructor: None,
            process_env: None,
        }
    }
}

/// `ACP_CLIENT_APP`.
pub fn acp_client_app() -> ACPClientApp {
    ACPClientApp {
        name: "ai-sdk-harness-acp".into(),
        version: VERSION.into(),
    }
}

/// `acpResumeStateSchema`.
pub fn acp_resume_state_schema() -> Schema {
    json_schema(json!({
        "type": "object",
        "properties": { "agentIdentity": { "type": "string" } },
        "required": ["agentIdentity"],
    }))
    .with_validate(|value| {
        let data: ACPLifecycleData = serde_json::from_value(value.clone())?;
        if let Some(config) = &data.turn_start_config {
            validate_turn_start_mappings(
                config.model_mapping.as_ref(),
                config.output_schema_mapping.as_ref(),
            )
            .map_err(plain_error)?;
        }
        if let Some(cold_session) = &data.cold_session {
            validate_turn_start_mappings(None, cold_session.output_schema_mapping.as_ref())
                .map_err(plain_error)?;
        }
        Ok(serde_json::to_value(data)?)
    })
}

/// `createACP`.
pub fn create_acp(settings: ACPHarnessSettings) -> HarnessResult<ACPV1Harness> {
    let version = settings.settings.version.as_deref().unwrap_or("v1");
    if version != "v1" {
        return Err(plain_error(format!(
            "Unsupported ACP protocol version {}. Supported versions: \"v1\".",
            serde_json::Value::String(version.to_string())
        )));
    }
    let client_app = settings.client_app.unwrap_or_else(acp_client_app);
    let mut builtin_tools = settings.builtin_tools.unwrap_or_default();
    if settings.settings.ask_user_questions.is_some() {
        builtin_tools.insert(
            "askUserQuestions".into(),
            HarnessV1BuiltinTool::from(
                harness_v1_builtin_tools()[&HarnessV1BuiltinToolName::AskUserQuestions].clone(),
            ),
        );
    }
    let mut options = CreateACPV1Options::new(
        settings.settings,
        builtin_tools,
        client_app,
        acp_resume_state_schema(),
    );
    options.port = settings.port;
    options.port_endpoint = settings.port_endpoint;
    options.startup_timeout_ms = settings.startup_timeout_ms;
    options.host_disconnect_timeout_ms = settings.host_disconnect_timeout_ms;
    options.reconnect = settings.reconnect;
    options.keep_alive_ms = settings.keep_alive_ms;
    options.web_socket_constructor = settings.web_socket_constructor;
    options.process_env = settings.process_env;
    create_acp_v1(options)
}
