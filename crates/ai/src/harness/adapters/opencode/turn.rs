//! Port of `bridge/index.ts`: the OpenCode server runtime, the session and
//! each prompt and compaction, run from this process against the server's
//! HTTP API and event stream.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::host_relay::{
    HostRelay, RelayAnswer, RelayConfig, RelayEvent, TurnBinding,
};
use crate::harness::adapters::host_tools::{ApprovalDecision, HostTools, tool_result_text};
use crate::harness::adapters::turn_builder::TurnBuilder;
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::network::{HarnessNetworkSandboxSession, HarnessPortEndpoint};
use crate::harness::questions_tool::QuestionsToolOutput;
use crate::harness::sandbox::SandboxSession;
use crate::harness::session::{FileChangeKind, HarnessEvent, HarnessTurnSender, NoticeLevel};
use crate::harness::tools::{BuiltinToolFilter, PermissionMode};
use crate::{
    AgentMessage, AgentResult, AgentToolResult, AssistantMessage, Message, StopReason, Tool,
    ToolCall, Usage, UserContent, UserMessageContent,
};

use super::client::{OpenCodeClient, Response};
use super::config::{
    ConfigInput, build_opencode_config, resolve_opencode_instructions, sandbox_realpath,
    split_model,
};
use super::events::{
    FinishReason, OpenCodeEvent, Output, Part, ToolNaming, Translator, as_object, format_error,
    get_opencode_event_session_id, map_opencode_finish_reason, map_usage,
    permission_tool_call_from_part, string_value, to_wire_tool_name, unwrap_opencode_event,
};
use super::permissions::{
    PermissionDecision, PermissionReply, PermissionRequest, PermissionToolCall,
    ToolRelayAuthorizer, canonical_json, select_permission_reply, to_harness_questions_input,
    to_opencode_question_response, to_permission_tool_name, tool_relay_call_key,
};
use super::read_tool::{READ_TOOL_NAME, ReadOutput, parse_read_args, read, read_tool_definition};
use super::server::{OpenCodeServerCoords, Server, ServerLaunch, kill_detached};
use super::{HOST_TOOL_MCP_SERVER_NAME, OPENCODE_API, OPENCODE_HARNESS_ID};

const QUESTIONS_TOOL: &str = "askUserQuestions";
const RELAY_AUTHORIZATION_TTL: Duration = Duration::from_secs(10);
const UNAUTHORIZED_RELAY_CALL: &str = "unauthorized tool relay request";

/// What stays the same for every turn of a session.
pub(super) struct SessionConfig {
    pub sandbox: Arc<dyn SandboxSession>,
    pub network: Option<Arc<dyn HarnessNetworkSandboxSession>>,
    pub work_dir: String,
    /// The session's private directory in the sandbox.
    pub private_dir: String,
    pub home_dir: String,
    pub skills_dir: String,
    /// The `opencode` command, shell-quoted.
    pub command: String,
    pub bootstrap_dir: Option<String>,
    /// The server's environment.
    pub env: IndexMap<String, String>,
    pub opencode_config: Option<Map<String, Value>>,
    pub mcp_servers: Option<Map<String, Value>>,
    pub provider: Option<String>,
    pub reasoning_variant: Option<String>,
    pub headers: Option<IndexMap<String, String>>,
    pub permission_mode: PermissionMode,
    pub inactive_builtin_tool_names: Vec<String>,
    pub port: Option<u16>,
    pub port_endpoint: Option<HarnessPortEndpoint>,
    pub startup_timeout: Duration,
    pub client_app: String,
}

impl SessionConfig {
    fn read_guard(&self) -> bool {
        self.permission_mode != PermissionMode::AllowAll
            && !self
                .inactive_builtin_tool_names
                .iter()
                .any(|name| name == READ_TOOL_NAME)
    }
}

/// The server and OpenCode session of a harness session; also its turn
/// lock.
#[derive(Default)]
pub(super) struct Runtime {
    pub server: Option<Server>,
    /// A detached server to try first.
    pub attach: Option<OpenCodeServerCoords>,
    /// The OpenCode session, once known.
    pub session_id: Option<String>,
    /// A session to restore on the first turn.
    pub resume_session_id: Option<String>,
    pub selected_model: Option<String>,
    /// Compaction events for the next turn.
    pub pending_compaction: Vec<HarnessEvent>,
}

impl Runtime {
    pub async fn close(&mut self, sandbox: &dyn SandboxSession) {
        if let Some(server) = self.server.take() {
            server.close(sandbox).await;
        } else if let Some(coords) = self.attach.take() {
            kill_detached(sandbox, coords.pid).await;
        }
    }
}

/// One turn.
pub(super) struct TurnConfig {
    pub prompt_parts: Vec<Value>,
    pub instructions: Option<String>,
    /// The model, after the GitLab subscription mapping.
    pub model: Option<String>,
    pub response_schema: Option<Value>,
    pub host: HostTools,
    pub skills_changed: bool,
    pub readable_skill_roots: Vec<String>,
}

fn error(message: impl Into<String>) -> crate::harness::error::BoxError {
    HarnessError::new(message).into()
}

fn empty_assistant(model: &str) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: OPENCODE_API.to_string(),
        provider: OPENCODE_HARNESS_ID.to_string(),
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

/// `toOpenCodePromptParts`.
pub(super) fn prompt_parts(message: &AgentMessage) -> Vec<Value> {
    let text = |text: &str| json!({ "type": "text", "text": text });
    match message {
        Message::User(user) => match &user.content {
            UserMessageContent::Text(value) => vec![text(value)],
            UserMessageContent::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    UserContent::Text(value) => text(&value.text),
                    UserContent::Image(image) => json!({
                        "type": "file",
                        "mime": image.mime_type,
                        "url": format!("data:{};base64,{}", image.mime_type, image.data),
                    }),
                })
                .collect(),
        },
        other => vec![text(&serde_json::to_string(other).unwrap_or_default())],
    }
}

fn fingerprint(value: &Value) -> String {
    let digest = ring::digest::digest(
        &ring::digest::SHA256,
        canonical_json(value).to_string().as_bytes(),
    );
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn sanitize_mcp_tool_name(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `modelRefFromStart`.
fn model_ref(
    model: Option<&str>,
    provider: Option<&str>,
    env: &IndexMap<String, String>,
) -> Option<(String, String)> {
    let (model_provider, model_id) = split_model(model, provider);
    let model_id = model_id?;
    let provider_id = model_provider
        .or(provider)
        .or(env
            .get("OPENAI_NAME")
            .map(String::as_str)
            .filter(|v| !v.is_empty()))
        .unwrap_or("anthropic");
    Some((provider_id.to_string(), model_id.to_string()))
}

fn model_ref_from_object(value: &Map<String, Value>) -> Option<(String, String)> {
    let provider = string_value(value.get("providerID"))?;
    let model = string_value(value.get("modelID").or(value.get("id")))?;
    Some((provider, model))
}

fn model_ref_from_value(value: Option<&Value>) -> Option<(String, String)> {
    as_object(value).and_then(model_ref_from_object)
}

/// The host tools and internal tools the relay serves this turn.
fn relay_tools(session: &SessionConfig, host: &HostTools) -> HarnessResult<Vec<Tool>> {
    let mut tools: Vec<Tool> = host.tools.iter().map(|tool| tool.definition()).collect();
    if session.read_guard() {
        if tools.iter().any(|tool| tool.name == READ_TOOL_NAME) {
            return Err(HarnessError::capability_unsupported(
                "OpenCode reserves the tool name \"read\" in restrictive permission modes.",
                Some(OPENCODE_HARNESS_ID.into()),
            )
            .into());
        }
        tools.push(read_tool_definition());
    }
    Ok(tools)
}

/// `ensureRuntime`: (re)start the server when its configuration changed.
pub(super) async fn ensure_server(
    runtime: &mut Runtime,
    session: &SessionConfig,
    model: Option<&str>,
    tools: Vec<Tool>,
    abort: Option<CancellationToken>,
) -> HarnessResult<()> {
    let sandbox = session.sandbox.as_ref();
    let instructions = resolve_opencode_instructions(
        sandbox,
        &session.work_dir,
        session
            .opencode_config
            .as_ref()
            .and_then(|config| config.get("instructions")),
        abort.clone(),
    )
    .await?;
    let needs_relay = !tools.is_empty();
    let relay_dir = format!("{}/host-tools", session.private_dir);
    let relay_command =
        needs_relay.then(|| crate::harness::adapters::host_relay::agent_side_command(&relay_dir));
    let input = ConfigInput {
        opencode_config: session.opencode_config.as_ref(),
        mcp_servers: session.mcp_servers.as_ref(),
        permission_mode: session.permission_mode,
        inactive_builtin_tool_names: &session.inactive_builtin_tool_names,
        model,
        provider: session.provider.as_deref(),
        headers: session.headers.as_ref(),
        env: &session.env,
        client_app: &session.client_app,
        skills_dir: &session.skills_dir,
        instructions,
        relay_command,
        read_guard: session.read_guard(),
    };
    let config = build_opencode_config(&input);
    let mut fingerprinted = config.clone();
    // The model switches per turn without a restart.
    fingerprinted.remove("model");
    let next_fingerprint = fingerprint(&json!({ "config": fingerprinted, "env": session.env }));

    if let Some(server) = &runtime.server {
        if server.fingerprint == next_fingerprint {
            if let Some(relay) = &server.relay
                && relay.set_tools(tools)
            {
                relay.notify_tools_changed().await;
            }
            return Ok(());
        }
    }
    if runtime.server.is_none()
        && let Some(coords) = runtime.attach.take()
    {
        if coords.fingerprint == next_fingerprint && !needs_relay {
            let headers = session
                .port_endpoint
                .as_ref()
                .and_then(|e| e.headers.as_ref());
            if let Ok(mut server) = Server::attach(&coords, &session.work_dir, headers)
                && matches!(server.client.get("/mcp").await, Ok(Response::Data(_)))
            {
                server.mcp_prefixes = mcp_prefixes(&server.client).await;
                runtime.server = Some(server);
                return Ok(());
            }
        }
        kill_detached(sandbox, coords.pid).await;
    }
    if let Some(server) = runtime.server.take() {
        server.close(sandbox).await;
    }

    // OpenCode always loads configuration from the home directory and its
    // global config directory; the working directory must not contain them.
    let work_dir = sandbox_realpath(sandbox, &session.work_dir, abort.clone()).await?;
    let config_home = session
        .env
        .get("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .cloned()
        .unwrap_or_else(|| format!("{}/.config", session.home_dir));
    for protected in [session.home_dir.clone(), format!("{config_home}/opencode")] {
        let protected = sandbox_realpath(sandbox, &protected, abort.clone()).await?;
        if crate::harness::adapters::sandbox_paths::relative_inside(&work_dir, &protected) {
            return Err(error(format!(
                "OpenCode session working directory \"{}\" must not be or contain the sandbox home directory or OpenCode's global config directory, because OpenCode always loads configuration from those locations.",
                session.work_dir
            )));
        }
    }

    let relay = if needs_relay {
        let relay = HostRelay::start_stdio(
            sandbox,
            &relay_dir,
            RelayConfig {
                server_name: HOST_TOOL_MCP_SERVER_NAME.to_string(),
                agent_label: "OpenCode".to_string(),
                images: true,
            },
        )
        .await?;
        relay.set_tools(tools);
        Some(relay)
    } else {
        None
    };
    let started = Server::start(ServerLaunch {
        sandbox: &session.sandbox,
        network: session.network.as_ref(),
        work_dir: &session.work_dir,
        command: &session.command,
        bootstrap_dir: session.bootstrap_dir.as_deref(),
        env: &session.env,
        config: &config,
        port: session.port,
        port_endpoint: session.port_endpoint.as_ref(),
        startup_timeout: session.startup_timeout,
    })
    .await;
    let mut server = match started {
        Ok(server) => server,
        Err(error) => {
            if let Some(relay) = relay {
                relay.close().await;
            }
            return Err(error);
        }
    };
    server.relay = relay;
    server.fingerprint = next_fingerprint;
    if server.relay.is_some() {
        let connected = match server.client.get("/mcp").await {
            Ok(Response::Data(status)) => {
                status
                    .get(HOST_TOOL_MCP_SERVER_NAME)
                    .and_then(|status| status.get("status"))
                    .and_then(Value::as_str)
                    == Some("connected")
            }
            _ => false,
        };
        if !connected {
            let tail = server.tail();
            server.close(sandbox).await;
            return Err(error(format!(
                "OpenCode failed to connect the HarnessAgent tool server.{}",
                if tail.is_empty() {
                    String::new()
                } else {
                    format!("\n{tail}")
                }
            )));
        }
    }
    server.mcp_prefixes = mcp_prefixes(&server.client).await;
    runtime.server = Some(server);
    Ok(())
}

async fn mcp_prefixes(client: &OpenCodeClient) -> Vec<String> {
    let Ok(Response::Data(Value::Object(servers))) = client.get("/mcp").await else {
        return Vec::new();
    };
    servers
        .iter()
        .filter(|(name, status)| {
            name.as_str() != HOST_TOOL_MCP_SERVER_NAME
                && status.get("status").and_then(Value::as_str) == Some("connected")
        })
        .map(|(name, _)| format!("{}_", sanitize_mcp_tool_name(name)))
        .collect()
}

fn read_session_id(data: &Value) -> Option<String> {
    string_value(data.get("id"))
        .or_else(|| string_value(data.get("data").and_then(|d| d.get("id"))))
}

/// `ensureSession`.
pub(super) async fn ensure_session(
    runtime: &mut Runtime,
    client: &OpenCodeClient,
) -> HarnessResult<String> {
    if let Some(id) = &runtime.session_id {
        return Ok(id.clone());
    }
    if let Some(id) = runtime.resume_session_id.take()
        && matches!(
            client.get(&format!("/session/{id}")).await,
            Ok(Response::Data(_))
        )
    {
        runtime.session_id = Some(id.clone());
        return Ok(id);
    }
    match client.post("/session", json!({})).await? {
        Response::Error(error) => Err(self::error(format!(
            "OpenCode session create failed: {}",
            format_error(&error)
        ))),
        Response::Data(data) => {
            let id = read_session_id(&data)
                .ok_or_else(|| self::error("OpenCode session create returned no id."))?;
            runtime.session_id = Some(id.clone());
            Ok(id)
        }
    }
}

/// An assistant message read back from the session.
#[derive(Debug, Clone, Default)]
struct AssistantSnapshot {
    info: Map<String, Value>,
    content_parts: Option<Vec<Value>>,
}

impl AssistantSnapshot {
    fn id(&self) -> Option<String> {
        string_value(self.info.get("id"))
    }
}

/// `latestAssistantSnapshot`.
async fn latest_assistant_snapshot(
    client: &OpenCodeClient,
    session_id: &str,
) -> Option<AssistantSnapshot> {
    if let Ok(Response::Data(Value::Array(messages))) = client
        .get(&format!("/session/{session_id}/message?limit=20"))
        .await
    {
        for item in messages.iter().rev() {
            let Some(info) = item.get("info").and_then(Value::as_object) else {
                continue;
            };
            if info.get("role").and_then(Value::as_str) == Some("assistant") {
                return Some(AssistantSnapshot {
                    info: info.clone(),
                    content_parts: item.get("parts").and_then(Value::as_array).cloned(),
                });
            }
        }
    }
    let Ok(Response::Data(data)) = client
        .get(&format!("/api/session/{session_id}/context"))
        .await
    else {
        return None;
    };
    let messages = data
        .get("data")
        .and_then(Value::as_array)
        .or_else(|| data.as_array())?;
    messages.iter().rev().find_map(|message| {
        let record = message.as_object()?;
        (record.get("type").and_then(Value::as_str) == Some("assistant")).then(|| {
            AssistantSnapshot {
                info: record.clone(),
                content_parts: record.get("content").and_then(Value::as_array).cloned(),
            }
        })
    })
}

/// `switchSessionModel`.
async fn switch_session_model(
    client: &OpenCodeClient,
    session_id: &str,
    model: Option<(String, String)>,
) -> HarnessResult<()> {
    let Some((provider_id, model_id)) = model else {
        return Ok(());
    };
    match client
        .post(
            &format!("/api/session/{session_id}/model"),
            json!({ "model": { "id": model_id, "providerID": provider_id } }),
        )
        .await?
    {
        Response::Error(error) => Err(self::error(format_error(&error))),
        Response::Data(_) => Ok(()),
    }
}

/// The turn's output, shared with the permission and question handlers.
struct Sink {
    turn: TurnBuilder,
    recorded_calls: HashSet<String>,
    pending_usage: Option<Usage>,
    saw_content: bool,
    saw_finish_step: bool,
}

fn add_usage(left: Option<Usage>, right: &Usage) -> Usage {
    let mut total = left.unwrap_or_default();
    total.input += right.input;
    total.output += right.output;
    total.cache_read += right.cache_read;
    total.cache_write += right.cache_write;
    total.total_tokens += right.total_tokens;
    if let Some(reasoning) = right.reasoning {
        *total.reasoning.get_or_insert(0) += reasoning;
    }
    total
}

fn tool_result_from_value(result: &Value) -> AgentToolResult {
    match result {
        Value::String(text) => AgentToolResult::text(text.clone()),
        Value::Null => AgentToolResult::text(""),
        other => {
            let mut result = AgentToolResult::text(other.to_string());
            result.details = Some(other.clone());
            result
        }
    }
}

impl Sink {
    fn assistant(&self) -> AssistantMessage {
        self.turn
            .current_assistant()
            .unwrap_or_else(|| empty_assistant(self.turn.model()))
    }

    fn notice(&self, level: NoticeLevel, message: String) {
        self.turn
            .emit_harness(HarnessEvent::Notice { level, message });
    }

    /// Record a tool call once: in the assistant message, then as started.
    fn record_tool_call(&mut self, id: &str, name: &str, args: Value) {
        if !self.recorded_calls.insert(id.to_string()) {
            return;
        }
        if let Some(usage) = self.pending_usage.take() {
            self.turn.add_usage(&usage);
        }
        self.turn.tool_call(id, id, name, args.clone());
        self.turn.tool_start(id, name, args);
    }

    fn apply(&mut self, part: Part) {
        let carry = |sink: &mut Self| {
            if !sink.turn.has_open_message()
                && let Some(usage) = sink.pending_usage.take()
            {
                sink.turn.add_usage(&usage);
            }
        };
        match part {
            Part::StreamStart { model_id } => {
                if let Some(model) = model_id {
                    self.turn.set_model(model);
                }
            }
            Part::TextStart { .. } | Part::ReasoningStart { .. } => {}
            Part::TextDelta { id, delta } => {
                self.saw_content = true;
                carry(self);
                self.turn.text_delta(&format!("text:{id}"), &delta);
            }
            Part::TextEnd { id } => self.turn.end_block(&format!("text:{id}")),
            Part::ReasoningDelta { id, delta } => {
                self.saw_content = true;
                carry(self);
                self.turn.thinking_delta(&format!("reasoning:{id}"), &delta);
            }
            Part::ReasoningEnd { id } => self.turn.end_block(&format!("reasoning:{id}")),
            Part::ToolCall {
                tool_call_id,
                tool_name,
                input,
                ..
            } => self.record_tool_call(&tool_call_id, &tool_name, input),
            Part::ToolResult {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => self.turn.tool_end(
                &tool_call_id,
                &tool_name,
                tool_result_from_value(&result),
                is_error,
            ),
            Part::FileChange { path } => self.turn.emit_harness(HarnessEvent::FileChange {
                kind: FileChangeKind::Modify,
                path,
            }),
            Part::FinishStep { reason, usage, .. } => {
                self.saw_finish_step = true;
                let usage = usage.to_usage();
                if self.turn.has_open_message() {
                    let total = add_usage(self.pending_usage.take(), &usage);
                    self.turn.add_usage(&total);
                    let stop = match reason {
                        FinishReason::Length => Some(StopReason::Length),
                        _ => None,
                    };
                    self.turn.close_message(stop, None, None);
                } else {
                    // The step's message already ended at its tool results.
                    self.pending_usage = Some(add_usage(self.pending_usage.take(), &usage));
                }
            }
            Part::Compaction { summary } => self.turn.emit_harness(HarnessEvent::Compaction {
                tokens_before: None,
                tokens_after: None,
                summary: Some(summary),
            }),
            Part::Raw(_) => {}
            Part::Warning(message) => self.notice(NoticeLevel::Warning, message),
            Part::Error(message) => self.notice(NoticeLevel::Error, message),
        }
    }
}

/// What the handlers share.
struct Handlers {
    client: OpenCodeClient,
    sandbox: Arc<dyn SandboxSession>,
    work_dir: String,
    permission_mode: PermissionMode,
    inactive: Vec<String>,
    skill_roots: Vec<String>,
    host: HostTools,
    sink: Arc<StdMutex<Sink>>,
    /// (session, call id) → the call.
    observed: StdMutex<HashMap<(String, String), PermissionToolCall>>,
    /// One host decision per (session, call, tool call).
    decisions: StdMutex<HashMap<String, Arc<tokio::sync::OnceCell<PermissionReply>>>>,
    abort: CancellationToken,
}

fn approval_reply(decision: ApprovalDecision) -> PermissionReply {
    match decision {
        ApprovalDecision::Allow { .. } => PermissionReply::Once,
        ApprovalDecision::Deny { reason } => {
            PermissionReply::Reject((!reason.is_empty()).then_some(reason))
        }
    }
}

impl Handlers {
    fn observe(&self, session_id: &str, event: &OpenCodeEvent) {
        let part = event.prop("part");
        if let (Some((native_name, input)), Some(call_id)) = (
            permission_tool_call_from_part(part),
            part.and_then(|part| string_value(part.get("callID"))),
        ) {
            self.observed.lock().unwrap().insert(
                (session_id.to_string(), call_id),
                PermissionToolCall { native_name, input },
            );
        }
        if event.r#type.as_deref() == Some("session.next.tool.started") {
            let input = as_object(event.prop("input"));
            let native_name =
                string_value(event.prop("toolName")).or_else(|| string_value(event.prop("tool")));
            let call_id = string_value(event.prop("callID"));
            if let (Some(input), Some(native_name), Some(call_id)) = (input, native_name, call_id) {
                self.observed.lock().unwrap().insert(
                    (session_id.to_string(), call_id),
                    PermissionToolCall {
                        native_name,
                        input: input.clone(),
                    },
                );
            }
        }
    }

    /// `getToolCall`.
    async fn tool_call(
        &self,
        event: &OpenCodeEvent,
        session_id: &str,
    ) -> Option<PermissionToolCall> {
        let source = as_object(event.prop("tool")).or_else(|| as_object(event.prop("source")))?;
        let call_id = string_value(source.get("callID"))?;
        let key = (session_id.to_string(), call_id.clone());
        if let Some(observed) = self.observed.lock().unwrap().get(&key) {
            return Some(observed.clone());
        }
        let message_id = string_value(source.get("messageID"))?;
        if let Ok(Response::Data(data)) = self
            .client
            .get(&format!("/session/{session_id}/message/{message_id}"))
            .await
            && let Some(parts) = data.get("parts").and_then(Value::as_array)
            && let Some(part) = parts
                .iter()
                .find(|part| part.get("callID").and_then(Value::as_str) == Some(call_id.as_str()))
            && let Some((native_name, input)) = permission_tool_call_from_part(Some(part))
        {
            let call = PermissionToolCall { native_name, input };
            self.observed.lock().unwrap().insert(key, call.clone());
            return Some(call);
        }
        self.observed.lock().unwrap().get(&key).cloned()
    }

    /// `requestApproval`: report the call, then ask `before_tool_call`.
    async fn request_approval(
        &self,
        session_id: &str,
        tool_call_id: &str,
        tool_call: Option<&PermissionToolCall>,
        action: &str,
    ) -> PermissionReply {
        let (name, args) = match tool_call {
            Some(call) => (
                to_wire_tool_name(&call.native_name),
                Value::Object(call.input.clone()),
            ),
            None => (to_permission_tool_name(action), json!({})),
        };
        let approve = || async {
            let assistant = {
                let mut sink = self.sink.lock().unwrap();
                if tool_call.is_some() {
                    sink.record_tool_call(tool_call_id, &name, args.clone());
                }
                sink.assistant()
            };
            let call = ToolCall {
                id: tool_call_id.to_string(),
                name: name.clone(),
                arguments: args.clone(),
                thought_signature: None,
            };
            let decision = self
                .host
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
            approval_reply(decision)
        };
        let Some(tool_call) = tool_call else {
            return approve().await;
        };
        let key = json!([
            session_id,
            tool_call_id,
            { "nativeName": tool_call.native_name, "input": tool_call.input },
        ])
        .to_string();
        let cell = self
            .decisions
            .lock()
            .unwrap()
            .entry(key)
            .or_default()
            .clone();
        cell.get_or_init(approve).await.clone()
    }

    /// `selectPermissionReply`.
    async fn select(
        &self,
        session_id: &str,
        tool_call_id: &str,
        request: &PermissionRequest<'_>,
    ) -> PermissionReply {
        let tool_name = to_permission_tool_name(request.action);
        if self.inactive.iter().any(|name| *name == tool_name) {
            return PermissionReply::Reject(Some(BuiltinToolFilter::denial_reason(&tool_name)));
        }
        match select_permission_reply(
            self.sandbox.as_ref(),
            request,
            self.permission_mode,
            &self.inactive,
            &self.work_dir,
            &self.skill_roots,
            Some(self.abort.clone()),
        )
        .await
        {
            PermissionDecision::Reply(reply) => reply,
            PermissionDecision::Ask => {
                self.request_approval(session_id, tool_call_id, request.tool_call, request.action)
                    .await
            }
        }
    }

    /// `handlePermission` and `handlePermissionV2`.
    async fn permission(&self, session_id: String, event: OpenCodeEvent) -> HarnessResult<()> {
        let v2 = event.r#type.as_deref() == Some("permission.v2.asked");
        let Some(request_id) = string_value(event.prop("id")) else {
            return Ok(());
        };
        let tool_call = self.tool_call(&event, &session_id).await;
        let action =
            string_value(event.prop(if v2 { "action" } else { "permission" })).unwrap_or_default();
        let resources: Vec<String> = event
            .prop(if v2 { "resources" } else { "patterns" })
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| match item {
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let tool_call_id = as_object(event.prop(if v2 { "source" } else { "tool" }))
            .and_then(|source| string_value(source.get("callID")))
            .unwrap_or_else(|| request_id.clone());
        let metadata = as_object(event.prop("metadata"));
        let request = PermissionRequest {
            action: &action,
            resources: &resources,
            metadata,
            tool_call: tool_call.as_ref(),
        };
        let reply = self.select(&session_id, &tool_call_id, &request).await;
        let path = if v2 {
            format!("/api/session/{session_id}/permission/{request_id}/reply")
        } else {
            format!("/permission/{request_id}/reply")
        };
        self.client.post(&path, reply.body()).await?;
        Ok(())
    }

    /// `handleQuestion`.
    async fn question(&self, event: OpenCodeEvent) -> HarnessResult<()> {
        let (Some(request_id), Some(_), Some(questions)) = (
            string_value(event.prop("id")),
            string_value(event.prop("sessionID")),
            event.prop("questions").and_then(Value::as_array).cloned(),
        ) else {
            return Ok(());
        };
        let tool_call_id = as_object(event.prop("tool"))
            .and_then(|tool| string_value(tool.get("callID")))
            .unwrap_or_else(|| request_id.clone());
        let input = to_harness_questions_input(&questions);
        let answers = if self.host.find(QUESTIONS_TOOL).is_none() {
            self.sink.lock().unwrap().notice(
                NoticeLevel::Warning,
                format!("No '{QUESTIONS_TOOL}' tool was provided to answer OpenCode's questions; they were rejected."),
            );
            None
        } else {
            let assistant = {
                let mut sink = self.sink.lock().unwrap();
                sink.record_tool_call(&tool_call_id, QUESTIONS_TOOL, input.clone());
                sink.assistant()
            };
            let call = ToolCall {
                id: tool_call_id.clone(),
                name: QUESTIONS_TOOL.to_string(),
                arguments: input,
                thought_signature: None,
            };
            let (result, is_error) = self.host.run(&call, &assistant, None).await;
            self.sink.lock().unwrap().turn.tool_end(
                &tool_call_id,
                QUESTIONS_TOOL,
                result.clone(),
                is_error,
            );
            let output = (!is_error)
                .then(|| {
                    result
                        .details
                        .clone()
                        .filter(|details| details.get("action").is_some())
                        .or_else(|| serde_json::from_str(&tool_result_text(&result)).ok())
                })
                .flatten()
                .and_then(|output| QuestionsToolOutput::parse(output).ok());
            output.and_then(|output| to_opencode_question_response(&questions, &output))
        };
        let response = match answers {
            None => {
                self.client
                    .post(&format!("/question/{request_id}/reject"), json!({}))
                    .await?
            }
            Some(answers) => {
                self.client
                    .post(
                        &format!("/question/{request_id}/reply"),
                        json!({ "answers": answers }),
                    )
                    .await?
            }
        };
        if let Response::Error(error) = response {
            return Err(self::error(format!(
                "OpenCode question response failed: {}",
                format_error(&error)
            )));
        }
        Ok(())
    }

    /// The restrictive-mode `read` tool: the read plugin's permission
    /// request, decided here, then the read through the sandbox.
    async fn read(&self, session_id: &str, call_id: &str, args: Value) -> Value {
        self.sink
            .lock()
            .unwrap()
            .record_tool_call(call_id, READ_TOOL_NAME, args.clone());
        let output = match parse_read_args(&args, &self.work_dir) {
            Err(message) => ReadOutput::error(message),
            Ok(parsed) => {
                let tool_call = PermissionToolCall {
                    native_name: READ_TOOL_NAME.to_string(),
                    input: args.as_object().cloned().unwrap_or_default(),
                };
                let resources = vec![parsed.file_path.clone()];
                let metadata = json!({ "filepath": parsed.file_path });
                let request = PermissionRequest {
                    action: "read",
                    resources: &resources,
                    metadata: metadata.as_object(),
                    tool_call: Some(&tool_call),
                };
                match self.select(session_id, call_id, &request).await {
                    PermissionReply::Reject(reason) => {
                        ReadOutput::error(reason.unwrap_or_else(|| {
                            "The user rejected permission to use this specific tool call.".into()
                        }))
                    }
                    _ => read(self.sandbox.as_ref(), &parsed, Some(self.abort.clone())).await,
                }
            }
        };
        self.sink.lock().unwrap().turn.tool_end(
            call_id,
            READ_TOOL_NAME,
            output.result,
            output.is_error,
        );
        output.mcp
    }
}

/// A host tool request waiting for its tool call event.
struct PendingRelayCall {
    name: String,
    args: Value,
    reply: oneshot::Sender<RelayAnswer>,
}

/// How the turn settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Settlement {
    Event,
    Aborted,
    StreamEnded,
}

/// `legacyStatusType`.
fn status_type(event: &OpenCodeEvent) -> Option<String> {
    event
        .prop("status")
        .and_then(Value::as_object)
        .map(|status| match status.get("type") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        })
}

/// `legacyRetryStatusMessage`.
fn retry_status_message(event: &OpenCodeEvent) -> String {
    let mut details = Vec::new();
    if let Some(status) = event.prop("status").and_then(Value::as_object) {
        if let Some(attempt) = status.get("attempt").and_then(Value::as_f64) {
            details.push(if attempt.fract() == 0.0 {
                format!("attempt {}", attempt as i64)
            } else {
                format!("attempt {attempt}")
            });
        }
        if let Some(message) = status
            .get("message")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|message| !message.is_empty())
        {
            details.push(message.to_string());
        }
    }
    if details.is_empty() {
        "OpenCode session retry".to_string()
    } else {
        format!("OpenCode session retry: {}", details.join("; "))
    }
}

/// What the event loop routes, per OpenCode session.
struct EventLoop {
    session_id: String,
    task_sessions: HashSet<String>,
    translator: Translator,
    descendants: HashMap<String, Translator>,
    naming: ToolNaming,
    handlers: Arc<Handlers>,
    failures: mpsc::UnboundedSender<String>,
    authorizer: ToolRelayAuthorizer<u64>,
    pending: HashMap<u64, PendingRelayCall>,
    next_pending: u64,
    read_guard: bool,
}

impl EventLoop {
    fn spawn_handler<F>(&self, handler: F)
    where
        F: std::future::Future<Output = HarnessResult<()>> + Send + 'static,
    {
        let failures = self.failures.clone();
        tokio::spawn(async move {
            if let Err(error) = handler.await {
                let _ = failures.send(error.to_string());
            }
        });
    }

    fn outputs(&mut self, outputs: Vec<Output>, descendant: Option<&str>) {
        for output in outputs {
            match output {
                Output::Part(part) => {
                    if descendant.is_none() {
                        self.handlers.sink.lock().unwrap().apply(part);
                    }
                }
                Output::AuthorizeHostToolCall {
                    call_id,
                    tool_name,
                    input,
                } => {
                    let key = tool_relay_call_key(&tool_name, &input);
                    if let Some((pending, call_id)) =
                        self.authorizer.authorize(key, call_id, Instant::now())
                    {
                        self.dispatch(pending, call_id);
                    }
                }
                Output::SubagentSession {
                    parent_session_id,
                    session_id,
                } => {
                    let source = descendant.unwrap_or(&self.session_id).to_string();
                    if parent_session_id == source && self.task_sessions.contains(&source) {
                        self.task_sessions.insert(session_id);
                    }
                }
            }
        }
    }

    /// A relay request: run it once its tool call event authorized it.
    fn relay_call(&mut self, name: String, args: Value, reply: oneshot::Sender<RelayAnswer>) {
        let id = self.next_pending;
        self.next_pending += 1;
        let key = tool_relay_call_key(&name, &args);
        self.pending
            .insert(id, PendingRelayCall { name, args, reply });
        if let Some(call_id) = self.authorizer.wait(key, id, Instant::now()) {
            self.dispatch(id, call_id);
        }
    }

    fn dispatch(&mut self, pending: u64, call_id: String) {
        let Some(call) = self.pending.remove(&pending) else {
            return;
        };
        if self.read_guard && call.name == READ_TOOL_NAME {
            let handlers = self.handlers.clone();
            let session_id = self.session_id.clone();
            tokio::spawn(async move {
                let result = handlers.read(&session_id, &call_id, call.args).await;
                let _ = call.reply.send(RelayAnswer::Result(result));
            });
            return;
        }
        let assistant = {
            let mut sink = self.handlers.sink.lock().unwrap();
            sink.record_tool_call(&call_id, &call.name, call.args.clone());
            sink.assistant()
        };
        let _ = call.reply.send(RelayAnswer::Run {
            id: call_id,
            assistant,
        });
    }

    fn deny_all(&mut self, pending: Vec<u64>) {
        for id in pending {
            if let Some(call) = self.pending.remove(&id) {
                let _ = call
                    .reply
                    .send(RelayAnswer::Deny(UNAUTHORIZED_RELAY_CALL.to_string()));
            }
        }
    }

    fn relay_event(&mut self, event: RelayEvent) {
        match event {
            RelayEvent::Call { name, args, reply } => self.relay_call(name, args, reply),
            RelayEvent::Update { id, partial } => self
                .handlers
                .sink
                .lock()
                .unwrap()
                .turn
                .tool_update(&id, partial),
            RelayEvent::End {
                id,
                name,
                result,
                is_error,
            } => self
                .handlers
                .sink
                .lock()
                .unwrap()
                .turn
                .tool_end(&id, &name, result, is_error),
            RelayEvent::Notice(level, message) => {
                self.handlers.sink.lock().unwrap().notice(level, message)
            }
        }
    }

    /// `consumeEvent`: route one event. Returns the event when it belongs to
    /// the session itself.
    fn route(&mut self, raw: &Value) -> Option<OpenCodeEvent> {
        let event = unwrap_opencode_event(raw)?;
        let event_session = get_opencode_event_session_id(&event);
        let scoped = match event_session {
            None => self.session_id.clone(),
            Some(id) if id == self.session_id => id,
            Some(id) if self.task_sessions.contains(&id) => id,
            Some(_) => return None,
        };
        let descendant = scoped != self.session_id;
        self.handlers.observe(&scoped, &event);
        match event.r#type.as_deref().unwrap_or("") {
            "question.asked" => {
                let handlers = self.handlers.clone();
                let event = event.clone();
                self.spawn_handler(async move { handlers.question(event).await });
            }
            "permission.v2.asked" | "permission.asked" => {
                let handlers = self.handlers.clone();
                let event = event.clone();
                let session = scoped.clone();
                self.spawn_handler(async move { handlers.permission(session, event).await });
            }
            _ if descendant => {
                let naming = self.naming.clone();
                let outputs = self
                    .descendants
                    .entry(scoped.clone())
                    .or_insert_with(|| Translator::new(naming))
                    .translate(&event);
                self.outputs(outputs, Some(&scoped));
            }
            _ => {
                let outputs = self.translator.translate(&event);
                self.outputs(outputs, None);
            }
        }
        (!descendant).then_some(event)
    }
}

/// Everything a running turn needs.
struct TurnParts<'a> {
    client: OpenCodeClient,
    session_id: String,
    relay: Option<&'a HostRelay>,
    naming: ToolNaming,
    handlers: Arc<Handlers>,
    read_guard: bool,
}

/// Which operation the event loop serves.
enum Operation<'a> {
    Prompt {
        body: Value,
        json: bool,
    },
    Compact {
        model: (String, String),
        _marker: std::marker::PhantomData<&'a ()>,
    },
}

/// The loop's result.
struct LoopResult {
    settlement: Settlement,
    terminal_error: Option<String>,
    failure: Option<String>,
    request_error: Option<String>,
    translator: Translator,
}

/// `runPrompt` and `runCompaction`'s event loop.
async fn run_event_loop(
    parts: TurnParts<'_>,
    operation: Operation<'_>,
    cancel: &CancellationToken,
) -> HarnessResult<LoopResult> {
    let stream_cancel = cancel.child_token();
    let mut events = parts.client.subscribe(stream_cancel.clone()).await?;
    let (failure_sender, mut failures) = mpsc::unbounded_channel();
    let (relay_sender, mut relay_events) = mpsc::unbounded_channel();
    if let Some(relay) = parts.relay {
        relay.bind(TurnBinding {
            host: parts.handlers.host.clone(),
            events: relay_sender,
        });
    }
    let mut state = EventLoop {
        session_id: parts.session_id.clone(),
        task_sessions: HashSet::from([parts.session_id.clone()]),
        translator: Translator::new(parts.naming.clone()),
        descendants: HashMap::new(),
        naming: parts.naming.clone(),
        handlers: parts.handlers.clone(),
        failures: failure_sender,
        authorizer: ToolRelayAuthorizer::new(RELAY_AUTHORIZATION_TTL),
        pending: HashMap::new(),
        next_pending: 0,
        read_guard: parts.read_guard,
    };
    let client = parts.client.clone();
    let session_id = parts.session_id.clone();
    let (json_format, mut request) = match &operation {
        Operation::Prompt { body, json } => {
            let client = client.clone();
            let path = format!("/session/{session_id}/prompt_async");
            let body = body.clone();
            (
                *json,
                tokio::spawn(async move { client.post(&path, body).await }),
            )
        }
        Operation::Compact { model, .. } => {
            let client = client.clone();
            let path = format!("/session/{session_id}/summarize");
            let body = json!({ "providerID": model.0, "modelID": model.1, "auto": false });
            (
                false,
                tokio::spawn(async move { client.post(&path, body).await }),
            )
        }
    };
    let compact = matches!(operation, Operation::Compact { .. });
    let mut request_done = false;
    let mut request_error = None;
    let mut saw_busy = false;
    let mut terminal_error = None;
    let mut failure = None;
    let mut compact_deadline: Option<tokio::time::Instant> = None;
    let mut tick = tokio::time::interval(Duration::from_millis(500));

    let settlement = loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break Settlement::Aborted,
            result = &mut request, if !request_done => {
                request_done = true;
                match result {
                    Ok(Ok(Response::Data(_))) => {
                        if compact {
                            compact_deadline = Some(tokio::time::Instant::now() + Duration::from_millis(250));
                        }
                    }
                    Ok(Ok(Response::Error(error))) => {
                        request_error = Some(format_error(&error));
                        break Settlement::Event;
                    }
                    Ok(Err(error)) => {
                        request_error = Some(error.to_string());
                        break Settlement::Event;
                    }
                    Err(error) => {
                        request_error = Some(error.to_string());
                        break Settlement::Event;
                    }
                }
            }
            _ = async { tokio::time::sleep_until(compact_deadline.unwrap()).await }, if compact_deadline.is_some() => {
                break Settlement::Event;
            }
            Some(message) = failures.recv() => {
                failure = Some(message);
                break Settlement::Event;
            }
            Some(event) = relay_events.recv() => state.relay_event(event),
            _ = tick.tick() => {
                let expired = state.authorizer.expire(Instant::now());
                state.deny_all(expired);
            }
            event = events.recv() => {
                let Some(raw) = event else {
                    break if cancel.is_cancelled() { Settlement::Aborted } else { Settlement::StreamEnded };
                };
                let Some(event) = state.route(&raw) else { continue };
                let event_type = event.r#type.as_deref().unwrap_or("");
                if compact {
                    if event_type == "session.next.compaction.ended" || event_type == "session.compacted" {
                        break Settlement::Event;
                    }
                    if event_type == "session.error" {
                        terminal_error = Some(format_error(event.prop("error").unwrap_or(&Value::Null)));
                        break Settlement::Event;
                    }
                } else {
                    if event_type == "message.updated" {
                        let outputs = state.translator.stream_start(event.prop("info"));
                        state.outputs(outputs, None);
                        if json_format
                            && let Some(info) = as_object(event.prop("info"))
                            && let Some(structured) = info.get("structured")
                        {
                            let id = string_value(info.get("id"))
                                .unwrap_or_else(crate::harness::utils::random_uuid);
                            let mut sink = state.handlers.sink.lock().unwrap();
                            sink.apply(Part::TextStart { id: id.clone() });
                            sink.apply(Part::TextDelta { id: id.clone(), delta: structured.to_string() });
                            sink.apply(Part::TextEnd { id });
                            sink.apply(Part::FinishStep {
                                reason: FinishReason::Stop,
                                raw: "stop".into(),
                                usage: Default::default(),
                                cost: None,
                            });
                            break Settlement::Event;
                        }
                    }
                    if event_type == "session.next.step.failed" || event_type == "session.error" {
                        let message = format_error(event.prop("error").unwrap_or(&Value::Null));
                        if event_type == "session.error" {
                            terminal_error = Some(message);
                        }
                        break Settlement::Event;
                    }
                }
                match status_type(&event).as_deref() {
                    Some("busy") => saw_busy = true,
                    Some("retry") => {
                        saw_busy = true;
                        state.handlers.sink.lock().unwrap().notice(NoticeLevel::Warning, retry_status_message(&event));
                    }
                    Some("idle") if saw_busy => {
                        saw_busy = false;
                        if compact || !json_format {
                            break Settlement::Event;
                        }
                    }
                    _ => {}
                }
            }
        }
    };
    stream_cancel.cancel();
    if settlement == Settlement::Aborted {
        // `abortSession`, after the prompt request settled.
        if !request_done {
            let _ = request.await;
        }
        let _ = client
            .post(&format!("/session/{session_id}/abort"), json!({}))
            .await;
    } else if !request_done {
        request.abort();
    }
    while let Ok(event) = relay_events.try_recv() {
        state.relay_event(event);
    }
    let remaining = state.authorizer.close();
    state.deny_all(remaining);
    if let Some(relay) = parts.relay {
        relay.unbind();
    }
    Ok(LoopResult {
        settlement,
        terminal_error,
        failure,
        request_error,
        translator: state.translator,
    })
}

fn emit_content_part(sink: &mut Sink, part: &Value) {
    let Some(kind) = part.get("type").and_then(Value::as_str) else {
        return;
    };
    if kind != "text" && kind != "reasoning" {
        return;
    }
    let id = string_value(part.get("id"))
        .unwrap_or_else(|| format!("{kind}-{}", crate::harness::utils::random_uuid()));
    let text = part.get("text").and_then(Value::as_str).unwrap_or("");
    if kind == "text" {
        if !text.is_empty() {
            sink.apply(Part::TextDelta {
                id: id.clone(),
                delta: text.into(),
            });
        }
        sink.apply(Part::TextEnd { id });
    } else {
        if !text.is_empty() {
            sink.apply(Part::ReasoningDelta {
                id: id.clone(),
                delta: text.into(),
            });
        }
        sink.apply(Part::ReasoningEnd { id });
    }
}

/// `emitContextFallback`.
async fn context_fallback(
    client: &OpenCodeClient,
    session_id: &str,
    baseline: &Option<AssistantSnapshot>,
    translator: &mut Translator,
    sink: &Arc<StdMutex<Sink>>,
) -> bool {
    let Some(assistant) = latest_assistant_snapshot(client, session_id).await else {
        return false;
    };
    let Some(id) = assistant.id() else {
        return false;
    };
    if let Some(baseline) = baseline
        && baseline.id().is_none_or(|baseline| baseline == id)
    {
        return false;
    }
    let info = Value::Object(assistant.info.clone());
    let outputs = translator.stream_start(Some(&info));
    let mut sink = sink.lock().unwrap();
    for output in outputs {
        if let Output::Part(part) = output {
            sink.apply(part);
        }
    }
    if !sink.saw_content {
        for part in assistant.content_parts.iter().flatten() {
            emit_content_part(&mut sink, part);
        }
    }
    let raw = match assistant.info.get("finish") {
        Some(Value::String(finish)) => finish.clone(),
        _ if assistant
            .info
            .get("error")
            .is_some_and(|error| !error.is_null() && error != &Value::Bool(false)) =>
        {
            "error".into()
        }
        _ => "stop".into(),
    };
    sink.apply(Part::FinishStep {
        reason: map_opencode_finish_reason(&raw),
        raw,
        usage: map_usage(assistant.info.get("tokens")),
        cost: assistant.info.get("cost").and_then(Value::as_f64),
    });
    true
}

fn handlers(
    session: &SessionConfig,
    client: &OpenCodeClient,
    host: HostTools,
    skill_roots: Vec<String>,
    sink: Arc<StdMutex<Sink>>,
    cancel: &CancellationToken,
) -> Arc<Handlers> {
    Arc::new(Handlers {
        client: client.clone(),
        sandbox: session.sandbox.clone(),
        work_dir: session.work_dir.clone(),
        permission_mode: session.permission_mode,
        inactive: session.inactive_builtin_tool_names.clone(),
        skill_roots,
        host,
        sink,
        observed: StdMutex::default(),
        decisions: StdMutex::default(),
        abort: cancel.clone(),
    })
}

fn naming(session: &SessionConfig, server: &Server, host: &HostTools) -> ToolNaming {
    let mut host_tools: HashSet<String> = host
        .tools
        .iter()
        .map(|tool| tool.definition().name)
        .collect();
    if session.read_guard() {
        host_tools.insert(READ_TOOL_NAME.to_string());
    }
    ToolNaming {
        host_tools,
        mcp_prefixes: server.mcp_prefixes.clone(),
        work_dir: session.work_dir.clone(),
    }
}

/// Run one prompt turn. Returns the turn's new messages.
pub(super) async fn run_turn(
    runtime: &mut Runtime,
    session: &SessionConfig,
    config: TurnConfig,
    prompt: AgentMessage,
    sender: HarnessTurnSender,
    cancel: CancellationToken,
) -> AgentResult<Vec<AgentMessage>> {
    let mut turn = TurnBuilder::new(
        sender,
        OPENCODE_API,
        OPENCODE_HARNESS_ID,
        config.model.clone().unwrap_or_default(),
    );
    turn.start(prompt);
    for event in runtime.pending_compaction.drain(..) {
        turn.emit_harness(event);
    }
    let sink = Arc::new(StdMutex::new(Sink {
        turn,
        recorded_calls: HashSet::new(),
        pending_usage: None,
        saw_content: false,
        saw_finish_step: false,
    }));
    let outcome = drive(runtime, session, &config, &sink, &cancel).await;
    let error = match outcome {
        Ok(()) => None,
        Err(_) if cancel.is_cancelled() => {
            Some((StopReason::Aborted, "Operation aborted".to_string()))
        }
        Err(error) => Some((StopReason::Error, error.to_string())),
    };
    let sink = match Arc::try_unwrap(sink) {
        Ok(sink) => sink.into_inner().unwrap(),
        Err(shared) => {
            // A handler still holds the sink; take the turn out of it.
            let mut guard = shared.lock().unwrap();
            let placeholder = TurnBuilder::new(
                guard.turn.sender().clone(),
                OPENCODE_API,
                OPENCODE_HARNESS_ID,
                "",
            );
            let turn = std::mem::replace(&mut guard.turn, placeholder);
            return Ok(turn.finish(error));
        }
    };
    Ok(sink.turn.finish(error))
}

async fn drive(
    runtime: &mut Runtime,
    session: &SessionConfig,
    config: &TurnConfig,
    sink: &Arc<StdMutex<Sink>>,
    cancel: &CancellationToken,
) -> HarnessResult<()> {
    let tools = relay_tools(session, &config.host)?;
    let model = config
        .model
        .clone()
        .or_else(|| runtime.selected_model.clone());
    if config.model.is_some() {
        runtime.selected_model = config.model.clone();
    }
    ensure_server(
        runtime,
        session,
        model.as_deref(),
        tools,
        Some(cancel.clone()),
    )
    .await?;
    let server = runtime.server.as_ref().unwrap();
    let client = server.client.clone();
    if config.skills_changed {
        client.post("/instance/dispose", json!({})).await?;
    }
    let session_id = ensure_session(runtime, &client).await?;
    let server = runtime.server.as_ref().unwrap();
    switch_session_model(
        &client,
        &session_id,
        model_ref(model.as_deref(), session.provider.as_deref(), &session.env),
    )
    .await?;
    if let Some(model) = &model {
        sink.lock().unwrap().turn.set_model(model.clone());
    }
    let baseline = latest_assistant_snapshot(&client, &session_id).await;
    let handlers = handlers(
        session,
        &client,
        config.host.clone(),
        config.readable_skill_roots.clone(),
        sink.clone(),
        cancel,
    );
    let mut body = json!({ "parts": config.prompt_parts });
    if let Some(instructions) = config.instructions.as_ref().filter(|text| !text.is_empty()) {
        body["system"] = json!(instructions);
    }
    if let Some(variant) = session.reasoning_variant.as_ref().filter(|v| !v.is_empty()) {
        body["variant"] = json!(variant);
    }
    if let Some(schema) = &config.response_schema {
        body["format"] = json!({ "type": "json_schema", "schema": schema });
    }
    let result = run_event_loop(
        TurnParts {
            client: client.clone(),
            session_id: session_id.clone(),
            relay: server.relay.as_ref(),
            naming: naming(session, server, &config.host),
            handlers,
            read_guard: session.read_guard(),
        },
        Operation::Prompt {
            body,
            json: config.response_schema.is_some(),
        },
        cancel,
    )
    .await?;
    if let Some(message) = result.request_error {
        return Err(error(format!("OpenCode prompt failed: {message}")));
    }
    if let Some(message) = result.failure {
        return Err(error(message));
    }
    match result.settlement {
        Settlement::Aborted => return Err(error("Operation aborted")),
        Settlement::StreamEnded => {
            return Err(error(
                "OpenCode event stream ended before the turn settled.",
            ));
        }
        Settlement::Event => {}
    }
    if let Some(message) = result.terminal_error {
        return Err(error(message));
    }
    let saw_finish_step = sink.lock().unwrap().saw_finish_step;
    if !saw_finish_step {
        let mut translator = result.translator;
        if !context_fallback(&client, &session_id, &baseline, &mut translator, sink).await {
            return Err(error(
                "OpenCode turn settled without a correlated assistant response.",
            ));
        }
    }
    Ok(())
}

/// `runCompactOperation` and `runCompaction`: compact between turns; the
/// compaction events go out with the next turn.
pub(super) async fn run_compaction(
    runtime: &mut Runtime,
    session: &SessionConfig,
    cancel: CancellationToken,
) -> HarnessResult<()> {
    let model = runtime.selected_model.clone();
    let tools = relay_tools(session, &HostTools::default())?;
    ensure_server(
        runtime,
        session,
        model.as_deref(),
        tools,
        Some(cancel.clone()),
    )
    .await?;
    let client = runtime.server.as_ref().unwrap().client.clone();
    let session_id = ensure_session(runtime, &client).await?;
    let server = runtime.server.as_ref().unwrap();
    // `resolveCompactionModel`.
    let snapshot = latest_assistant_snapshot(&client, &session_id).await;
    let mut model_ref_value = snapshot.as_ref().and_then(|snapshot| {
        model_ref_from_value(snapshot.info.get("model"))
            .or_else(|| model_ref_from_object(&snapshot.info))
            .or_else(|| {
                model_ref_from_value(
                    snapshot
                        .info
                        .get("metadata")
                        .and_then(|metadata| metadata.get("assistant")),
                )
            })
    });
    if model_ref_value.is_none()
        && let Ok(Response::Data(info)) = client.get(&format!("/session/{session_id}")).await
        && let Some(info) = info.as_object()
    {
        model_ref_value =
            model_ref_from_value(info.get("model")).or_else(|| model_ref_from_object(info));
    }
    let model_ref_value = model_ref_value
        .or_else(|| model_ref(model.as_deref(), session.provider.as_deref(), &session.env))
        .ok_or_else(|| {
            error("OpenCode compaction requires a previous turn or an explicit model.")
        })?;
    let (sender, mut events) = crate::harness::session::HarnessTurnSender::channel();
    let sink = Arc::new(StdMutex::new(Sink {
        turn: TurnBuilder::new(sender, OPENCODE_API, OPENCODE_HARNESS_ID, ""),
        recorded_calls: HashSet::new(),
        pending_usage: None,
        saw_content: false,
        saw_finish_step: false,
    }));
    let handlers = handlers(
        session,
        &client,
        HostTools::default(),
        Vec::new(),
        sink.clone(),
        &cancel,
    );
    let result = run_event_loop(
        TurnParts {
            client: client.clone(),
            session_id,
            relay: server.relay.as_ref(),
            naming: naming(session, server, &HostTools::default()),
            handlers,
            read_guard: session.read_guard(),
        },
        Operation::Compact {
            model: model_ref_value,
            _marker: std::marker::PhantomData,
        },
        &cancel,
    )
    .await?;
    drop(sink);
    let mut compactions = Vec::new();
    while let Ok(event) = events.try_recv() {
        if matches!(event, HarnessEvent::Compaction { .. }) {
            compactions.push(event);
        }
    }
    if let Some(message) = result.request_error {
        return Err(error(format!("OpenCode compaction failed: {message}")));
    }
    if let Some(message) = result.failure.or(result.terminal_error) {
        return Err(error(message));
    }
    if result.settlement == Settlement::Aborted {
        return Err(error("Operation aborted"));
    }
    if compactions.is_empty() {
        compactions.push(HarnessEvent::Compaction {
            tokens_before: None,
            tokens_after: None,
            summary: Some(String::new()),
        });
    }
    runtime.pending_compaction.extend(compactions);
    Ok(())
}
