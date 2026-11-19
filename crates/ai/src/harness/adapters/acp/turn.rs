//! Port of `bridge/index.ts`: the ACP connection, the session and each
//! prompt turn, run from this process against the agent's stdio.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde_json::{Map, Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::host_tools::{ApprovalDecision, HostTools};
use crate::harness::adapters::turn_builder::TurnBuilder;
use crate::harness::error::{BoxError, HarnessError, HarnessResult};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxSession};
use crate::harness::session::{HarnessTurnSender, NoticeLevel};
use crate::harness::tools::PermissionMode;
use crate::{
    AgentMessage, AgentResult, AssistantMessage, Message, StopReason, ToolCall, Usage, UserContent,
    UserMessageContent,
};

use super::client::{AcpConnection, METHOD_NOT_FOUND};
use super::permissions::{AutoApproval, cancelled, permission_options, selected};
use super::relay::{HostRelay, RelayEvent, TurnBinding};
use super::settings::{
    AcpConfigValue, AcpHarnessSettings, AcpHostToolMcpTransport, AcpInstructionMapping,
    AcpModelMapping, AcpOutputSchemaMapping, AcpPermissionModeMapping, AcpPermissionModeTarget,
};
use super::translator::{AcpBuiltinToolMapping, Finish, Translator};

/// `acp.PROTOCOL_VERSION`.
pub const ACP_PROTOCOL_VERSION: u64 = 1;
const CANCEL_TIMEOUT: Duration = Duration::from_secs(5);

/// The running agent and its ACP session.
pub(super) struct Agent {
    conn: Arc<AcpConnection>,
    relay: HostRelay,
    session_id: String,
    supports_images: bool,
    prompted: bool,
    current_model: Option<String>,
}

/// The agent of a harness session; also its turn lock.
#[derive(Default)]
pub(super) struct Runtime {
    pub agent: Option<Agent>,
}

impl Runtime {
    pub async fn close(&mut self) {
        if let Some(agent) = self.agent.take() {
            agent.relay.close().await;
            agent.conn.close().await;
        }
    }
}

/// What stays the same for every turn of a session.
pub(super) struct SessionConfig {
    pub settings: Arc<AcpHarnessSettings>,
    pub sandbox: Arc<dyn SandboxSession>,
    pub work_dir: String,
    /// The session's private directory in the sandbox.
    pub private_dir: String,
    /// The agent command, shell-quoted.
    pub command: String,
    pub env: indexmap::IndexMap<String, String>,
    /// `HOME` of an agent installed with an install command.
    pub agent_home: Option<String>,
    pub permission_mode: PermissionMode,
    pub builtins: Vec<AcpBuiltinToolMapping>,
}

/// One turn.
pub(super) struct TurnConfig {
    pub prompt: Vec<Value>,
    pub has_images: bool,
    pub instructions: Option<String>,
    pub model: Option<String>,
    pub prompt_meta: Option<Value>,
    pub host: HostTools,
    /// The ACP session to restore when the agent starts.
    pub restore_session_id: Option<String>,
    pub session_id_out: Arc<StdMutex<Option<String>>>,
}

fn error(message: impl Into<String>) -> BoxError {
    HarnessError::new(message).into()
}

fn unsupported(settings: &AcpHarnessSettings, message: impl Into<String>) -> BoxError {
    HarnessError::capability_unsupported(message, Some(settings.harness_id.clone())).into()
}

/// `convertHarnessPromptToACPContentBlocks`.
pub(super) fn prompt_content_blocks(message: &AgentMessage) -> (Vec<Value>, bool) {
    let text = |text: &str| json!({ "type": "text", "text": text });
    match message {
        Message::User(user) => match &user.content {
            UserMessageContent::Text(value) => (vec![text(value)], false),
            UserMessageContent::Parts(parts) => {
                let mut has_images = false;
                let blocks = parts
                    .iter()
                    .map(|part| match part {
                        UserContent::Text(value) => text(&value.text),
                        UserContent::Image(image) => {
                            has_images = true;
                            json!({ "type": "image", "data": image.data, "mimeType": image.mime_type })
                        }
                    })
                    .collect();
                (blocks, has_images)
            }
        },
        other => (
            vec![text(&serde_json::to_string(other).unwrap_or_default())],
            false,
        ),
    }
}

/// `prependACPInstructionGuidance`.
pub(super) fn prepend_instruction_guidance(
    prompt: Vec<Value>,
    instructions: Option<&str>,
) -> Vec<Value> {
    let Some(instructions) = instructions.filter(|i| !i.is_empty()) else {
        return prompt;
    };
    let text = [
        "<session-guidance>",
        "This block is operating guidance from the harness, not user-authored content.",
        "<instructions>",
        instructions,
        "</instructions>",
        "</session-guidance>",
    ]
    .join("\n");
    let mut blocks = vec![json!({ "type": "text", "text": text })];
    blocks.extend(prompt);
    blocks
}

/// `createOutputSchemaPromptMeta`.
pub(super) fn output_schema_prompt_meta(
    mapping: Option<&AcpOutputSchemaMapping>,
    schema: Option<&Value>,
) -> Option<Value> {
    let (Some(AcpOutputSchemaMapping::SessionPromptMeta { path }), Some(schema)) =
        (mapping, schema)
    else {
        return None;
    };
    let mut value = schema.clone();
    for segment in path.iter().rev() {
        let mut parent = Map::new();
        parent.insert(segment.clone(), value);
        value = Value::Object(parent);
    }
    Some(value)
}

const UNSAFE_PATH_SEGMENTS: [&str; 3] = ["__proto__", "constructor", "prototype"];

fn assert_safe_path(path: &[String]) -> HarnessResult<()> {
    if path.is_empty()
        || path
            .iter()
            .any(|segment| segment.is_empty() || UNSAFE_PATH_SEGMENTS.contains(&segment.as_str()))
    {
        return Err(error(
            "ACP instruction mapping path must contain only safe, non-empty property names.",
        ));
    }
    Ok(())
}

fn set_string_at_path(
    record: &Map<String, Value>,
    path: &[String],
    value: &str,
) -> Map<String, Value> {
    let mut result = record.clone();
    let Some((key, rest)) = path.split_first() else {
        return result;
    };
    let next = if rest.is_empty() {
        json!(value)
    } else {
        let child = record
            .get(key)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        Value::Object(set_string_at_path(&child, rest, value))
    };
    result.insert(key.clone(), next);
    result
}

/// `resolveACPInstructionConfiguration`: the session `_meta` and agent
/// environment with the instructions where the mapping says.
pub(super) fn resolve_instruction_configuration(
    instructions: Option<&str>,
    mapping: Option<&AcpInstructionMapping>,
    session_meta: Option<&Map<String, Value>>,
    environment: &mut indexmap::IndexMap<String, String>,
) -> HarnessResult<Option<Map<String, Value>>> {
    let (Some(mapping), Some(instructions)) = (mapping, instructions.filter(|i| !i.is_empty()))
    else {
        return Ok(session_meta.cloned());
    };
    match mapping {
        AcpInstructionMapping::Filesystem { path } => {
            if path.trim().is_empty() {
                return Err(error(
                    "ACP instruction mapping filesystem path must be a non-empty string.",
                ));
            }
            Ok(session_meta.cloned())
        }
        AcpInstructionMapping::SessionMeta { path } => {
            assert_safe_path(path)?;
            Ok(Some(set_string_at_path(
                &session_meta.cloned().unwrap_or_default(),
                path,
                instructions,
            )))
        }
        AcpInstructionMapping::LaunchEnvJson { variable, path } => {
            assert_safe_path(path)?;
            let configuration = match environment.get(variable).filter(|value| !value.is_empty()) {
                Some(serialized) => match serde_json::from_str::<Value>(serialized) {
                    Ok(Value::Object(map)) => map,
                    _ => {
                        return Err(error(format!(
                            "ACP instruction mapping environment variable \"{variable}\" must contain a JSON object."
                        )));
                    }
                },
                None => Map::new(),
            };
            environment.insert(
                variable.clone(),
                Value::Object(set_string_at_path(&configuration, path, instructions)).to_string(),
            );
            Ok(session_meta.cloned())
        }
    }
}

/// `resolveACPModelId`.
pub(super) fn resolve_acp_model_id(
    session_state: &Value,
    mapping: Option<&AcpModelMapping>,
) -> Option<String> {
    let non_empty = |value: Option<&Value>| {
        value
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let from_config_option = || {
        let options: Vec<&Value> = session_state
            .get("configOptions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|option| option.is_object())
            .collect();
        let by_path = match mapping {
            Some(AcpModelMapping::SessionConfigOption { path }) => options
                .iter()
                .find(|option| option.get("id").and_then(Value::as_str) == Some(path)),
            _ => None,
        };
        let option = by_path.or_else(|| {
            options
                .iter()
                .find(|option| option.get("category").and_then(Value::as_str) == Some("model"))
        });
        non_empty(option.and_then(|option| option.get("currentValue")))
    };
    let from_session_model = || {
        non_empty(
            session_state
                .get("models")
                .filter(|models| models.is_object())
                .and_then(|models| models.get("currentModelId")),
        )
    };
    match mapping {
        Some(AcpModelMapping::SessionModel { .. }) => {
            from_session_model().or_else(from_config_option)
        }
        _ => from_config_option().or_else(from_session_model),
    }
}

fn merge_records(left: &Map<String, Value>, right: &Map<String, Value>) -> Map<String, Value> {
    let mut result = left.clone();
    for (key, value) in right {
        let merged = match (result.get(key), value) {
            (Some(Value::Object(previous)), Value::Object(next)) => {
                Value::Object(merge_records(previous, next))
            }
            _ => value.clone(),
        };
        result.insert(key.clone(), merged);
    }
    result
}

/// `createACPInitializeRequest`.
pub(super) fn initialize_request(settings: &AcpHarnessSettings) -> Value {
    let mut capabilities = merge_records(
        &settings
            .authentication
            .as_ref()
            .and_then(|authentication| authentication.client_capabilities.clone())
            .unwrap_or_default(),
        &settings.client_capabilities.clone().unwrap_or_default(),
    );
    let boolean_options = settings
        .permission_mode_mapping
        .as_ref()
        .into_iter()
        .flat_map(|mapping| {
            [
                &mapping.allow_reads,
                &mapping.allow_edits,
                &mapping.allow_all,
            ]
        })
        .flatten()
        .any(|target| {
            matches!(
                target,
                AcpPermissionModeTarget::SessionConfigOption {
                    value: AcpConfigValue::Boolean(_),
                    ..
                }
            )
        });
    if boolean_options {
        let addition = json!({ "session": { "configOptions": { "boolean": {} } } });
        capabilities = merge_records(&capabilities, addition.as_object().unwrap());
    }
    json!({
        "protocolVersion": ACP_PROTOCOL_VERSION,
        "clientInfo": {
            "name": settings.client_app.name,
            "version": settings.client_app.version,
        },
        "clientCapabilities": capabilities,
    })
}

fn format_choices(values: &[String]) -> String {
    if values.is_empty() {
        "no matching choices".to_string()
    } else {
        values
            .iter()
            .map(|value| serde_json::to_string(value).unwrap())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn mode_name(mode: PermissionMode) -> &'static str {
    match mode {
        PermissionMode::AllowReads => "allow-reads",
        PermissionMode::AllowEdits => "allow-edits",
        PermissionMode::AllowAll => "allow-all",
    }
}

/// `configureACPPermissionMode`'s validation: the request that sets the
/// mode, as (method, params without `sessionId`).
pub(super) fn permission_mode_request(
    settings: &AcpHarnessSettings,
    mapping: &AcpPermissionModeMapping,
    permission_mode: PermissionMode,
    session: &Value,
) -> HarnessResult<(&'static str, Value)> {
    let Some(target) = mapping.get(permission_mode) else {
        let mut message = format!(
            "Permission mode \"{}\" is not supported by this ACP harness.",
            mode_name(permission_mode)
        );
        if mapping.allow_all.is_some() {
            message.push_str(" Use permissionMode: 'allow-all'.");
        }
        return Err(unsupported(settings, message));
    };
    for mode in [
        PermissionMode::AllowReads,
        PermissionMode::AllowEdits,
        PermissionMode::AllowAll,
    ] {
        let Some(mapped) = mapping.get(mode) else {
            continue;
        };
        let name = mode_name(mode);
        match mapped {
            AcpPermissionModeTarget::SessionMode { mode_id } => {
                let available: Vec<String> = session
                    .pointer("/modes/availableModes")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|mode| mode.get("id").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect();
                if !available.contains(mode_id) {
                    return Err(unsupported(
                        settings,
                        format!(
                            "ACP permission mapping for \"{name}\" requires session mode \"{mode_id}\", but the agent advertised {}.",
                            format_choices(&available)
                        ),
                    ));
                }
            }
            AcpPermissionModeTarget::SessionConfigOption { config_id, value } => {
                let options: Vec<&Value> = session
                    .get("configOptions")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .collect();
                let Some(option) = options
                    .iter()
                    .find(|option| option.get("id").and_then(Value::as_str) == Some(config_id))
                else {
                    let ids: Vec<String> = options
                        .iter()
                        .filter_map(|option| option.get("id").and_then(Value::as_str))
                        .map(str::to_string)
                        .collect();
                    return Err(unsupported(
                        settings,
                        format!(
                            "ACP permission mapping for \"{name}\" requires session configuration \"{config_id}\", but the agent advertised {}.",
                            format_choices(&ids)
                        ),
                    ));
                };
                let is_boolean = option.get("type").and_then(Value::as_str) == Some("boolean");
                match (is_boolean, value) {
                    (true, AcpConfigValue::Boolean(_)) => {}
                    (true, AcpConfigValue::String(value)) => {
                        return Err(unsupported(
                            settings,
                            format!(
                                "ACP permission mapping for \"{name}\" assigns string value \"{value}\" to boolean session configuration \"{config_id}\"."
                            ),
                        ));
                    }
                    (false, AcpConfigValue::Boolean(value)) => {
                        return Err(unsupported(
                            settings,
                            format!(
                                "ACP permission mapping for \"{name}\" assigns boolean value {value} to select session configuration \"{config_id}\"."
                            ),
                        ));
                    }
                    (false, AcpConfigValue::String(value)) => {
                        let available: Vec<String> = option
                            .get("options")
                            .and_then(Value::as_array)
                            .into_iter()
                            .flatten()
                            .flat_map(|choice| {
                                match choice.get("options").and_then(Value::as_array) {
                                    Some(group) => group
                                        .iter()
                                        .filter_map(|item| {
                                            item.get("value").and_then(Value::as_str)
                                        })
                                        .map(str::to_string)
                                        .collect::<Vec<_>>(),
                                    None => choice
                                        .get("value")
                                        .and_then(Value::as_str)
                                        .map(str::to_string)
                                        .into_iter()
                                        .collect(),
                                }
                            })
                            .collect();
                        if !available.contains(value) {
                            return Err(unsupported(
                                settings,
                                format!(
                                    "ACP permission mapping for \"{name}\" requires value \"{value}\" for session configuration \"{config_id}\", but the agent advertised {}.",
                                    format_choices(&available)
                                ),
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(match target {
        AcpPermissionModeTarget::SessionMode { mode_id } => {
            ("session/set_mode", json!({ "modeId": mode_id }))
        }
        AcpPermissionModeTarget::SessionConfigOption { config_id, value } => {
            let mut params = json!({ "configId": config_id, "value": value });
            if matches!(value, AcpConfigValue::Boolean(_)) {
                params["type"] = json!("boolean");
            }
            ("session/set_config_option", params)
        }
    })
}

/// `createExternalMcpServers`.
fn external_mcp_servers(
    settings: &AcpHarnessSettings,
    initialization: &Value,
) -> HarnessResult<Vec<Value>> {
    let mut servers = Vec::new();
    for (name, value) in settings.mcp_servers.iter().flatten() {
        let Value::Object(config) = value else {
            return Err(error(format!(
                "ACP MCP server \"{name}\" must be configured with an object value."
            )));
        };
        let kind = config.get("type").and_then(Value::as_str);
        if kind == Some("acp") {
            return Err(unsupported(
                settings,
                "ACP-transport MCP servers require client-side mcp/connect handling, which this harness does not provide.",
            ));
        }
        let capabilities = initialization.pointer("/agentCapabilities/mcpCapabilities");
        let supported = |transport: &str| {
            capabilities.and_then(|caps| caps.get(transport)) == Some(&Value::Bool(true))
        };
        if let Some(transport @ ("http" | "sse")) = kind
            && !supported(transport)
        {
            return Err(unsupported(
                settings,
                format!(
                    "The ACP agent does not advertise support for {} MCP servers.",
                    transport.to_uppercase()
                ),
            ));
        }
        let mut server = config.clone();
        server.insert("name".into(), json!(name));
        servers.push(Value::Object(server));
    }
    Ok(servers)
}

/// Start the agent and its session, or keep the running one.
async fn ensure_agent(
    runtime: &mut Runtime,
    session: &SessionConfig,
    turn: &TurnConfig,
) -> HarnessResult<()> {
    let settings = &session.settings;
    let tools: Vec<crate::Tool> = turn
        .host
        .tools
        .iter()
        .map(|tool| tool.definition())
        .collect();
    let mut restore = turn.restore_session_id.clone();
    if let Some(agent) = &runtime.agent {
        if !agent.conn.is_closed() {
            if agent.relay.set_tools(tools) && agent.prompted {
                agent.relay.notify_tools_changed().await;
            }
            return Ok(());
        }
        // The agent exited between turns: restore its session.
        restore = Some(agent.session_id.clone());
        runtime.close().await;
    }

    let mut env = session.env.clone();
    let session_meta = resolve_instruction_configuration(
        turn.instructions.as_deref(),
        settings.instruction_mapping.as_ref(),
        settings.session_meta.as_ref(),
        &mut env,
    )?;
    let command = match &session.agent_home {
        Some(home) => {
            env.insert("HOME".into(), home.clone());
            format!(
                "PATH={}\"${{PATH:+:$PATH}}\" exec {}",
                crate::harness::utils::shell_quote(&format!("{home}/.local/bin")),
                session.command
            )
        }
        None => format!("exec {}", session.command),
    };
    let conn = AcpConnection::spawn(
        session.sandbox.as_ref(),
        SandboxProcessOptions {
            command,
            working_directory: Some(session.work_dir.clone()),
            env: Some(env),
            abort_signal: None,
        },
        &settings.harness_id,
    )
    .await?;
    let started = start_session(&conn, session, turn, tools, restore, session_meta).await;
    match started {
        Ok(agent) => {
            runtime.agent = Some(agent);
            Ok(())
        }
        Err(error) => {
            conn.close().await;
            Err(error)
        }
    }
}

async fn start_session(
    conn: &Arc<AcpConnection>,
    session: &SessionConfig,
    turn: &TurnConfig,
    tools: Vec<crate::Tool>,
    restore: Option<String>,
    session_meta: Option<Map<String, Value>>,
) -> HarnessResult<Agent> {
    let settings = &session.settings;
    let initialization = conn
        .request("initialize", initialize_request(settings))
        .await?;
    let version = initialization
        .get("protocolVersion")
        .and_then(Value::as_u64);
    if version != Some(ACP_PROTOCOL_VERSION) {
        return Err(error(format!(
            "ACP protocol negotiation failed: requested v{ACP_PROTOCOL_VERSION}, agent selected v{}.",
            initialization
                .get("protocolVersion")
                .map(Value::to_string)
                .unwrap_or_else(|| "undefined".into())
        )));
    }
    let supports_images = initialization.pointer("/agentCapabilities/promptCapabilities/image")
        == Some(&Value::Bool(true))
        || settings.image_content;
    if let Some(authentication) = &settings.authentication {
        let advertised: Vec<String> = initialization
            .get("authMethods")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|method| method.get("id").and_then(Value::as_str))
            .map(str::to_string)
            .collect();
        if !advertised.contains(&authentication.method_id) {
            return Err(error(format!(
                "ACP authentication method \"{}\" is not advertised by the agent. Advertised methods: {}.",
                authentication.method_id,
                if advertised.is_empty() {
                    "none".to_string()
                } else {
                    advertised.join(", ")
                }
            )));
        }
        let mut params = json!({ "methodId": authentication.method_id });
        if let Some(meta) = &authentication.meta {
            params["_meta"] = Value::Object(meta.clone());
        }
        conn.request("authenticate", params).await?;
    }
    let mut mcp_servers = external_mcp_servers(settings, &initialization)?;
    let relay = match settings.host_tool_mcp_transport {
        AcpHostToolMcpTransport::Stdio => {
            HostRelay::start_stdio(
                session.sandbox.as_ref(),
                &format!("{}/host-tools", session.private_dir),
                settings.image_content,
            )
            .await?
        }
        AcpHostToolMcpTransport::Http => {
            if initialization.pointer("/agentCapabilities/mcpCapabilities/http")
                != Some(&Value::Bool(true))
            {
                return Err(unsupported(
                    settings,
                    "This harness exposes host tools through an HTTP MCP server, but the ACP agent does not advertise support for HTTP MCP servers.",
                ));
            }
            HostRelay::start_http(&settings.http_relay, settings.image_content).await?
        }
    };
    relay.set_tools(tools);
    mcp_servers.push(relay.mcp_server_definition());

    let mut params = json!({ "cwd": session.work_dir, "mcpServers": mcp_servers });
    if let Some(meta) = &session_meta {
        params["_meta"] = Value::Object(meta.clone());
    }
    let result = async {
        let (session_id, response) = match &restore {
            Some(id) => {
                let resume = initialization
                    .pointer("/agentCapabilities/sessionCapabilities/resume")
                    .is_some_and(|value| !value.is_null() && value != &Value::Bool(false));
                let load = initialization.pointer("/agentCapabilities/loadSession")
                    == Some(&Value::Bool(true));
                let method = if resume {
                    "session/resume"
                } else if load {
                    "session/load"
                } else {
                    return Err(unsupported(
                        settings,
                        "Cold ACP session restoration requires the agent to advertise sessionCapabilities.resume or loadSession; a fresh unrelated ACP session will not be created.",
                    ));
                };
                params["sessionId"] = json!(id);
                let response = conn.request(method, params).await?;
                // The replayed history is not part of a turn.
                let mut events = conn.events.lock().await;
                while let Ok(event) = events.try_recv() {
                    answer_outside_turn(conn, &event).await;
                }
                (id.clone(), response)
            }
            None => {
                let response = conn.request("session/new", params).await?;
                let id = response
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .ok_or_else(|| error("ACP session/new returned no sessionId."))?
                    .to_string();
                (id, response)
            }
        };
        if let Some(mapping) = &settings.permission_mode_mapping {
            let (method, mut request) =
                permission_mode_request(settings, mapping, session.permission_mode, &response)?;
            request["sessionId"] = json!(session_id);
            conn.request(method, request).await?;
        }
        Ok((session_id, response))
    }
    .await;
    let (session_id, response) = match result {
        Ok(result) => result,
        Err(error) => {
            relay.close().await;
            return Err(error);
        }
    };
    *turn.session_id_out.lock().unwrap() = Some(session_id.clone());
    Ok(Agent {
        conn: conn.clone(),
        relay,
        session_id,
        supports_images,
        prompted: false,
        current_model: resolve_acp_model_id(&response, Some(&settings.model_mapping)),
    })
}

/// Requests that arrive with no turn running are refused.
async fn answer_outside_turn(conn: &AcpConnection, event: &Value) {
    let (Some(id), Some(method)) = (event.get("id"), event.get("method").and_then(Value::as_str))
    else {
        return;
    };
    if method == "session/request_permission" {
        conn.respond(id, Ok(cancelled())).await;
    } else {
        conn.respond(
            id,
            Err((METHOD_NOT_FOUND, format!("Method not found: {method}"))),
        )
        .await;
    }
}

/// `configureACPModel`: returns the model the agent reports afterwards.
async fn configure_model(
    conn: &AcpConnection,
    session_id: &str,
    model: &str,
    mapping: &AcpModelMapping,
) -> HarnessResult<String> {
    let response = match mapping {
        AcpModelMapping::SessionModel { path } => {
            let mut params = json!({ "sessionId": session_id });
            params[path.as_str()] = json!(model);
            conn.request("session/set_model", params).await?
        }
        AcpModelMapping::SessionConfigOption { path } => {
            conn.request(
                "session/set_config_option",
                json!({ "sessionId": session_id, "configId": path, "value": model }),
            )
            .await?
        }
    };
    Ok(resolve_acp_model_id(&response, Some(mapping)).unwrap_or_else(|| model.to_string()))
}

/// A permission request that needs the caller.
struct Ask {
    tool_call: Value,
    reply: oneshot::Sender<Option<(ToolCall, AssistantMessage)>>,
}

fn empty_assistant(api: &str, provider: &str, model: &str) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: api.to_string(),
        provider: provider.to_string(),
        model: model.to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: crate::utils::time::now_millis(),
    }
}

/// Run one turn. Returns the turn's new messages.
pub(super) async fn run_turn(
    runtime: &mut Runtime,
    session: &SessionConfig,
    config: TurnConfig,
    prompt: AgentMessage,
    sender: HarnessTurnSender,
    cancel: CancellationToken,
) -> AgentResult<Vec<AgentMessage>> {
    let settings = session.settings.clone();
    if let Err(error) = ensure_agent(runtime, session, &config).await {
        return Err(crate::harness::error::into_agent_error(error));
    }
    let agent = runtime.agent.as_mut().unwrap();
    if config.has_images && !agent.supports_images {
        return Err(HarnessError::capability_unsupported(
            "The ACP agent does not advertise support for image prompts.",
            Some(settings.harness_id.clone()),
        )
        .into());
    }
    let mut turn = TurnBuilder::new(sender, super::ACP_API, settings.harness_id.clone(), "");
    turn.start(prompt);
    let mut translator = Translator::new(
        turn,
        session.builtins.clone(),
        settings.resolve_host_tool_call.clone(),
        config
            .host
            .tools
            .iter()
            .map(|tool| tool.definition().name)
            .collect(),
    );
    let outcome = drive(agent, session, &config, &mut translator, &cancel).await;
    let error = match outcome {
        Ok(Some(finish)) if finish.error.is_none() => {
            translator.close_message_with(finish.stop_reason);
            None
        }
        Ok(Some(finish)) => finish.error.map(|message| (finish.stop_reason, message)),
        Ok(None) => None,
        Err(error) if cancel.is_cancelled() => {
            let _ = error;
            Some((StopReason::Aborted, "Operation aborted".to_string()))
        }
        Err(error) => Some((StopReason::Error, error.to_string())),
    };
    Ok(translator.turn.finish(error))
}

async fn drive(
    agent: &mut Agent,
    session: &SessionConfig,
    config: &TurnConfig,
    translator: &mut Translator,
    cancel: &CancellationToken,
) -> HarnessResult<Option<Finish>> {
    let settings = session.settings.clone();
    if let Some(model) = &config.model {
        let configured = configure_model(
            &agent.conn,
            &agent.session_id,
            model,
            &settings.model_mapping,
        )
        .await?;
        agent.current_model = Some(configured);
    }
    let model = agent
        .current_model
        .clone()
        .or_else(|| config.model.clone())
        .or_else(|| settings.fallback_model.clone())
        .unwrap_or_else(|| settings.harness_id.clone());
    translator.turn.set_model(model.clone());
    let conn = agent.conn.clone();
    let session_id = agent.session_id.clone();
    let (relay_sender, mut relay_events) = mpsc::unbounded_channel();
    agent.relay.bind(TurnBinding {
        host: config.host.clone(),
        events: relay_sender,
    });
    let mut events = conn.events.lock().await;
    while let Ok(event) = events.try_recv() {
        answer_outside_turn(&conn, &event).await;
    }

    let mut params = json!({ "sessionId": session_id, "prompt": config.prompt });
    if let Some(meta) = &config.prompt_meta {
        params["_meta"] = meta.clone();
    }
    agent.prompted = true;
    let mut prompt = {
        let conn = conn.clone();
        tokio::spawn(async move { conn.request("session/prompt", params).await })
    };
    let approval = AutoApproval {
        sandbox: session.sandbox.clone(),
        work_dir: session.work_dir.clone(),
        permission_mode: session.permission_mode,
        has_permission_mode_mapping: settings.permission_mode_mapping.is_some(),
        write_protected_paths: settings.write_protected_paths.clone(),
        abort: Some(cancel.clone()),
    };
    let permissions = cancel.child_token();
    let (ask_sender, mut asks) = mpsc::unbounded_channel::<Ask>();
    let mut events_open = true;
    let mut cancel_sent = false;
    let mut deadline: Option<tokio::time::Instant> = None;

    let response = loop {
        tokio::select! {
            event = events.recv(), if events_open => match event {
                Some(event) => handle_event(
                    &conn, &session_id, translator, &event, &approval, &config.host,
                    &permissions, &ask_sender,
                ).await,
                None => events_open = false,
            },
            Some(event) = relay_events.recv() => handle_relay(translator, event, &model),
            Some(ask) = asks.recv() => {
                let result = translator.permission_tool_call(&ask.tool_call).map(|(name, input)| {
                    let call = ToolCall {
                        id: ask.tool_call.get("toolCallId").and_then(Value::as_str).unwrap_or("").to_string(),
                        name,
                        arguments: input,
                        thought_signature: None,
                    };
                    let assistant = translator
                        .current_assistant()
                        .unwrap_or_else(|| empty_assistant(super::ACP_API, &settings.harness_id, &model));
                    (call, assistant)
                });
                let _ = ask.reply.send(result);
            }
            result = &mut prompt => break result
                .map_err(|error| HarnessError::new(error.to_string()))
                .and_then(|result| result.map_err(|e| HarnessError::new(e.to_string()))),
            _ = cancel.cancelled(), if !cancel_sent => {
                cancel_sent = true;
                permissions.cancel();
                let _ = conn.notify("session/cancel", json!({ "sessionId": session_id })).await;
                deadline = Some(tokio::time::Instant::now() + CANCEL_TIMEOUT);
            }
            _ = async { tokio::time::sleep_until(deadline.unwrap()).await }, if deadline.is_some() => {
                prompt.abort();
                break Err(HarnessError::new("Operation aborted"));
            }
        }
    };
    permissions.cancel();
    // Updates and tool results that arrived before the response.
    while let Ok(event) = events.try_recv() {
        handle_event(
            &conn,
            &session_id,
            translator,
            &event,
            &approval,
            &config.host,
            &permissions,
            &ask_sender,
        )
        .await;
    }
    drop(events);
    while let Ok(event) = relay_events.try_recv() {
        handle_relay(translator, event, &model);
    }
    agent.relay.unbind();
    match response {
        Ok(response) => Ok(Some(translator.finish(&response))),
        Err(_) if cancel_sent => Ok(Some(Finish {
            stop_reason: StopReason::Aborted,
            error: Some("Operation aborted".to_string()),
        })),
        Err(error) => {
            let tail = conn.stderr_tail();
            let mut message = error.to_string();
            if conn.is_closed() && !tail.is_empty() {
                message = format!("{message}\n{tail}");
            }
            Ok(Some(Finish {
                stop_reason: StopReason::Error,
                error: Some(message),
            }))
        }
    }
}

fn handle_relay(translator: &mut Translator, event: RelayEvent, model: &str) {
    match event {
        RelayEvent::Call { name, args, reply } => {
            let id = translator.claim_host_call_id(&name, &args);
            translator.host_tool_call(&id, &name, args);
            let assistant = translator
                .current_assistant()
                .unwrap_or_else(|| empty_assistant(super::ACP_API, "", model));
            let _ = reply.send((id, assistant));
        }
        RelayEvent::Update { id, partial } => translator.turn.tool_update(&id, partial),
        RelayEvent::End {
            id,
            name,
            result,
            is_error,
        } => translator.tool_result(&id, &name, result, is_error),
        RelayEvent::Notice(level, message) => translator.notice(level, message),
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_event(
    conn: &Arc<AcpConnection>,
    session_id: &str,
    translator: &mut Translator,
    event: &Value,
    approval: &AutoApproval,
    host: &HostTools,
    permissions: &CancellationToken,
    asks: &mpsc::UnboundedSender<Ask>,
) {
    let method = event.get("method").and_then(Value::as_str).unwrap_or("");
    let params = event.get("params").cloned().unwrap_or(Value::Null);
    let Some(id) = event.get("id").cloned() else {
        if method == "session/update"
            && params.get("sessionId").and_then(Value::as_str) == Some(session_id)
            && let Some(update) = params.get("update")
        {
            if update.get("sessionUpdate").and_then(Value::as_str) == Some("config_option_update")
                && let Some(model) = resolve_acp_model_id(update, None)
            {
                translator.turn.set_model(model);
            }
            translator.update(update);
        }
        return;
    };
    if method != "session/request_permission" {
        conn.respond(
            &id,
            Err((METHOD_NOT_FOUND, format!("Method not found: {method}"))),
        )
        .await;
        return;
    }
    let request_session = params
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or("");
    if request_session != session_id {
        translator.notice(
            NoticeLevel::Warning,
            format!(
                "ACP permission request for session \"{request_session}\" was cancelled because the active session is \"{session_id}\"."
            ),
        );
        conn.respond(&id, Ok(cancelled())).await;
        return;
    }
    let tool_call = params.get("toolCall").cloned().unwrap_or(Value::Null);
    let options = match permission_options(&params) {
        Ok(options) => options,
        Err(missing) => {
            translator.notice(
                NoticeLevel::Warning,
                format!(
                    "ACP permission request for tool call \"{}\" was cancelled because it did not advertise {}.",
                    tool_call.get("toolCallId").and_then(Value::as_str).unwrap_or(""),
                    missing.join(" and ")
                ),
            );
            conn.respond(&id, Ok(cancelled())).await;
            return;
        }
    };
    if translator.observe_tool_call(&tool_call).is_some() {
        conn.respond(&id, Ok(selected(&options.allow_once))).await;
        return;
    }
    let conn = conn.clone();
    let approval = approval.clone();
    let host = host.clone();
    let permissions = permissions.clone();
    let asks = asks.clone();
    tokio::spawn(async move {
        let decide = async {
            if approval.approves(&tool_call).await {
                return Some(true);
            }
            let (reply, answer) = oneshot::channel();
            asks.send(Ask {
                tool_call: tool_call.clone(),
                reply,
            })
            .ok()?;
            let (call, assistant) = answer.await.ok()??;
            let name = call.name.clone();
            let decision = host
                .approve(
                    &call,
                    &assistant,
                    ApprovalDecision::Deny {
                        reason: format!(
                            "Tool '{name}' requires approval, but no approval handler was provided."
                        ),
                    },
                )
                .await;
            Some(matches!(decision, ApprovalDecision::Allow { .. }))
        };
        let outcome = tokio::select! {
            outcome = decide => outcome,
            _ = permissions.cancelled() => None,
        };
        let response = match outcome {
            Some(true) => selected(&options.allow_once),
            Some(false) => selected(&options.reject_once),
            None => cancelled(),
        };
        conn.respond(&id, Ok(response)).await;
    });
}
