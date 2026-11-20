//! One Codex turn over the app server (`codex-app-server-driver.ts`), with
//! its notifications translated into ai.rs events
//! (`create-app-server-event-handler.ts`, `create-emit-stream-event.ts` and
//! `codex-step-tracker.ts`).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::client::AppServer;
use super::tools::{
    DynamicTools, assert_trusted_apply_patch_hook, create_dynamic_tools,
    create_trusted_apply_patch_hook, resolve_codex_builtin_tool_policy, to_toml_inline,
};
use crate::harness::adapters::host_tools::HostTools;
use crate::harness::adapters::turn_builder::TurnBuilder;
use crate::harness::error::{HarnessError, HarnessResult, into_agent_error};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxSession};
use crate::harness::session::{FileChangeKind, HarnessEvent, HarnessTurnSender, NoticeLevel};
use crate::harness::tools::BuiltinToolFilter;
use crate::harness::utils::shell_quote;
use crate::{
    AgentMessage, AgentResult, AgentToolResult, ImageContent, Message, StopReason, ToolCall,
    ToolResultContent, Usage, UserContent, UserMessageContent,
};

const BACKGROUND_TERMINAL_POLL_INTERVAL: Duration = Duration::from_secs(1);
const INTERRUPT_GRACE: Duration = Duration::from_secs(5);

/// The app server and the thread it has loaded, kept across turns.
#[derive(Default)]
pub(super) struct Runtime {
    pub client: Option<Arc<AppServer>>,
    /// What the client was started for, to restart it when that changes.
    config_key: String,
    loaded_thread_id: Option<String>,
    /// (requested model, model Codex reported) of the loaded thread.
    loaded_thread_model: Option<(Option<String>, Option<String>)>,
}

impl Runtime {
    pub async fn close(&mut self) {
        let loaded = self.loaded_thread_id.take();
        self.loaded_thread_model = None;
        let Some(client) = self.client.take() else {
            return;
        };
        if !client.shared {
            client.close().await;
            return;
        }
        // [rust] A shared process keeps running for its other sessions: only
        // this session's thread is unloaded, and the last one out stops it.
        if let Some(thread_id) = loaded
            && !client.is_closed()
        {
            let _ = tokio::time::timeout(
                Duration::from_secs(5),
                client.request("thread/unsubscribe", json!({ "threadId": thread_id })),
            )
            .await;
        }
        if Arc::strong_count(&client) == 1 {
            client.close().await;
        }
    }
}

/// [rust] App servers that sessions share, by sandbox, command and
/// environment. Held weakly: the last session to let go stops the process.
#[derive(Clone, Default)]
pub(crate) struct SharedAppServers(
    Arc<tokio::sync::Mutex<HashMap<String, std::sync::Weak<AppServer>>>>,
);

impl SharedAppServers {
    async fn get_or_spawn(
        &self,
        sandbox: &Arc<dyn SandboxSession>,
        options: SandboxProcessOptions,
    ) -> HarnessResult<Arc<AppServer>> {
        let key = json!({
            "sandbox": Arc::as_ptr(sandbox) as *const () as usize,
            // Each thread names its own working directory.
            "command": options.command,
            "env": options.env,
        })
        .to_string();
        let mut servers = self.0.lock().await;
        servers.retain(|_, server| server.strong_count() > 0);
        if let Some(server) = servers.get(&key).and_then(std::sync::Weak::upgrade)
            && !server.is_closed()
        {
            return Ok(server);
        }
        let server = AppServer::spawn(sandbox.as_ref(), options, true).await?;
        servers.insert(key, Arc::downgrade(&server));
        Ok(server)
    }
}

/// Everything one turn needs.
pub(super) struct TurnConfig {
    pub sandbox: Arc<dyn SandboxSession>,
    pub executable: String,
    pub work_dir: String,
    pub env: IndexMap<String, String>,
    /// The thread to continue; `None` starts a new one.
    pub thread_id: Option<String>,
    /// Unload the previous thread before starting a new one.
    pub restart_thread: bool,
    pub codex_model: Option<String>,
    pub codex_config: Map<String, Value>,
    pub filter: Option<BuiltinToolFilter>,
    pub web_search: bool,
    pub effort: Option<&'static str>,
    pub output_schema: Option<Value>,
    pub host: HostTools,
    pub model: String,
    /// The thread id, updated when Codex reports it.
    pub thread_id_out: Arc<StdMutex<Option<String>>>,
    pub steer: Arc<StdMutex<Option<SteerHandle>>>,
    /// [rust] Run on a process other sessions share.
    pub shared: Option<SharedAppServers>,
}

/// Lets callers add user messages to the running turn (`turn/steer`).
#[derive(Clone)]
pub(super) struct SteerHandle {
    client: Arc<AppServer>,
    thread_id: String,
    turn_id: String,
    internal: mpsc::UnboundedSender<Internal>,
}

impl SteerHandle {
    pub async fn steer(&self, message: AgentMessage) -> AgentResult<()> {
        let response = self
            .client
            .request(
                "turn/steer",
                json!({
                    "threadId": self.thread_id,
                    "expectedTurnId": self.turn_id,
                    "clientUserMessageId": crate::harness::utils::random_uuid(),
                    "input": input_items(&message),
                }),
            )
            .await
            .map_err(into_agent_error)?;
        if response.get("turnId").and_then(Value::as_str) != Some(self.turn_id.as_str()) {
            return Err(HarnessError::new(
                "Codex app-server turn/steer returned an unexpected turn ID.",
            )
            .into());
        }
        let _ = self.internal.send(Internal::Steer(message));
        Ok(())
    }
}

enum Internal {
    ToolEnd {
        id: String,
        name: String,
        result: AgentToolResult,
        is_error: bool,
    },
    Steer(AgentMessage),
}

/// The prompt as `turn/start` input items.
pub(super) fn input_items(message: &AgentMessage) -> Vec<Value> {
    let text = |text: &str| json!({ "type": "text", "text": text, "text_elements": [] });
    match message {
        Message::User(user) => match &user.content {
            UserMessageContent::Text(value) => vec![text(value)],
            UserMessageContent::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    UserContent::Text(value) => text(&value.text),
                    UserContent::Image(image) => json!({
                        "type": "image",
                        "url": format!("data:{};base64,{}", image.mime_type, image.data),
                    }),
                })
                .collect(),
        },
        other => vec![text(&serde_json::to_string(other).unwrap_or_default())],
    }
}

/// `splitModelProviderConfig`: the provider goes on the app-server command
/// line, where Codex resolves it.
pub(super) fn split_model_provider_config(
    codex_config: &Map<String, Value>,
) -> HarnessResult<(Map<String, Value>, Vec<String>)> {
    let mut thread_config = codex_config.clone();
    let mut overrides = Vec::new();
    if let Some(provider) = thread_config.remove("model_provider") {
        overrides.push(format!("model_provider={}", to_toml_inline(&provider)?));
    }
    if let Some(providers) = thread_config.remove("model_providers") {
        overrides.push(format!("model_providers={}", to_toml_inline(&providers)?));
    }
    Ok((thread_config, overrides))
}

/// [rust] Sign-in and models from a short-lived app server.
pub(super) async fn probe(
    sandbox: &dyn SandboxSession,
    executable: &str,
    work_dir: &str,
    env: &IndexMap<String, String>,
    codex_config: &Map<String, Value>,
) -> HarnessResult<crate::harness::probe::HarnessProbe> {
    use crate::harness::probe::{HarnessProbe, codex_account, codex_models};
    let (_, overrides) = split_model_provider_config(codex_config)?;
    let command = std::iter::once(executable.to_string())
        .chain(
            app_server_args(&overrides)
                .iter()
                .map(|arg| shell_quote(arg)),
        )
        .collect::<Vec<_>>()
        .join(" ");
    let client = AppServer::spawn(
        sandbox,
        SandboxProcessOptions {
            command,
            working_directory: Some(work_dir.to_string()),
            env: Some(env.clone()),
            abort_signal: None,
        },
        false,
    )
    .await?;
    let read = async {
        client.initialize().await?;
        let mut probe = HarnessProbe::default();
        match client.request("account/read", json!({})).await {
            Ok(account) => {
                let (signed_in, label) = codex_account(&account);
                probe.signed_in = Some(signed_in);
                probe.account = label;
            }
            Err(error) => probe.error = Some(error.to_string()),
        }
        let models = client.request("model/list", json!({})).await?;
        probe.models = codex_models(&models);
        HarnessResult::Ok(probe)
    };
    let result = tokio::time::timeout(PROBE_TIMEOUT, read).await;
    client.close().await;
    match result {
        Ok(result) => result,
        Err(_) => {
            Err(HarnessError::new("Codex app-server did not answer the probe in time.").into())
        }
    }
}

/// How long a probe waits for the harness.
pub(crate) const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// `createCodexAppServerArgs`.
pub(super) fn app_server_args(cli_overrides: &[String]) -> Vec<String> {
    let mut args = vec![
        "--config".to_string(),
        "sandbox_mode=\"danger-full-access\"".to_string(),
        "--config".to_string(),
        "approval_policy=\"never\"".to_string(),
    ];
    for item in cli_overrides {
        args.push("--config".to_string());
        args.push(item.clone());
    }
    args.push("app-server".to_string());
    args.push("--stdio".to_string());
    args
}

fn thread_params(config: &TurnConfig, thread_config: &Map<String, Value>) -> Map<String, Value> {
    let mut params = Map::new();
    if let Some(model) = &config.codex_model {
        params.insert("model".into(), json!(model));
    }
    params.insert("cwd".into(), json!(config.work_dir));
    params.insert("approvalPolicy".into(), json!("never"));
    params.insert("sandbox".into(), json!("danger-full-access"));
    let mut thread_config = thread_config.clone();
    if !thread_config.contains_key("web_search") {
        thread_config.insert(
            "web_search".into(),
            json!(if config.web_search {
                "live"
            } else {
                "disabled"
            }),
        );
    }
    params.insert("config".into(), Value::Object(thread_config));
    params
}

fn assert_thread_permissions(response: &Value, method: &str) -> HarnessResult<()> {
    let sandbox_type = response
        .get("sandbox")
        .and_then(|sandbox| sandbox.get("type"))
        .and_then(Value::as_str);
    if response.get("approvalPolicy").and_then(Value::as_str) != Some("never")
        || sandbox_type != Some("dangerFullAccess")
    {
        return Err(HarnessError::new(format!(
            "Codex app-server {method} did not disable approvals and its platform sandbox."
        ))
        .into());
    }
    Ok(())
}

fn nested_string(value: &Value, path: &[&str], method: &str) -> HarnessResult<String> {
    let mut current = value;
    for segment in path {
        current = current.get(segment).unwrap_or(&Value::Null);
    }
    match current.as_str() {
        Some(text) if !text.is_empty() => Ok(text.to_string()),
        _ => Err(HarnessError::new(format!(
            "Codex app-server {method} returned an invalid response."
        ))
        .into()),
    }
}

fn str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn result_from(value: Value, text: Option<String>) -> AgentToolResult {
    let text = text.unwrap_or_else(|| value_text(&value));
    AgentToolResult {
        content: if text.is_empty() {
            Vec::new()
        } else {
            vec![ToolResultContent::text(text)]
        },
        details: Some(value),
        usage: None,
        added_tool_names: Vec::new(),
        terminate: false,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct UsageBreakdown {
    input: u64,
    cached_input: u64,
    cache_write_input: u64,
    output: u64,
}

impl UsageBreakdown {
    fn read(value: Option<&Value>) -> Option<Self> {
        let value = value?.as_object()?;
        let get = |key: &str| value.get(key).and_then(Value::as_u64).unwrap_or(0);
        Some(Self {
            input: get("inputTokens"),
            cached_input: get("cachedInputTokens"),
            cache_write_input: get("cacheWriteInputTokens"),
            output: get("outputTokens"),
        })
    }

    fn add(&mut self, other: Self) {
        self.input += other.input;
        self.cached_input += other.cached_input;
        self.cache_write_input += other.cache_write_input;
        self.output += other.output;
    }

    fn usage(self) -> Usage {
        let to = |value: u64| value.min(u32::MAX as u64) as u32;
        let no_cache = self
            .input
            .saturating_sub(self.cached_input + self.cache_write_input);
        Usage {
            input: to(no_cache),
            output: to(self.output),
            cache_read: to(self.cached_input),
            cache_write: to(self.cache_write_input),
            total_tokens: to(self.input + self.output),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeToolName {
    ApplyPatch,
    ViewImage,
}

impl NativeToolName {
    fn name(self) -> &'static str {
        match self {
            Self::ApplyPatch => "apply_patch",
            Self::ViewImage => "view_image",
        }
    }
}

struct NativeToolCall {
    name: NativeToolName,
    result_emitted: bool,
    success_confirmed: bool,
}

/// How the turn ended (`AppServerTurnResult`).
struct Completion {
    status: String,
    error: Option<String>,
}

/// Translates app-server notifications into the turn.
struct Translator {
    turn: TurnBuilder,
    thread_id: String,
    turn_id: String,
    /// Item id → text length already emitted.
    text_emitted: HashMap<String, usize>,
    reasoning_emitted: HashMap<String, usize>,
    /// Item id → reasoning text, for items whose summary arrives as deltas.
    reasoning_text: HashMap<String, String>,
    agent_text: HashMap<String, String>,
    /// Whether the open message has tool calls; new model content then
    /// starts the next message.
    open_has_tools: bool,
    /// Tool starts that wait for the open message to close.
    pending_starts: Vec<(String, String, Value)>,
    native_calls: IndexMap<String, NativeToolCall>,
    web_search_started: HashSet<String>,
    started_items: HashSet<String>,
    pending_usage: UsageBreakdown,
    last_usage_key: Option<UsageBreakdown>,
    turn_error: Option<String>,
    completion: Option<Completion>,
}

impl Translator {
    fn notice(&self, level: NoticeLevel, message: impl Into<String>) {
        self.turn.emit_harness(HarnessEvent::Notice {
            level,
            message: message.into(),
        });
    }

    fn matches_turn(&self, params: &Value) -> bool {
        let turn_id = str_field(params, "turnId")
            .or_else(|| params.get("turn").and_then(|turn| str_field(turn, "id")));
        str_field(params, "threadId") == Some(self.thread_id.as_str())
            && turn_id == Some(self.turn_id.as_str())
    }

    fn begin_message(&mut self) {
        if self.turn.has_open_message() {
            return;
        }
        self.turn.begin_message();
        if self.pending_usage != UsageBreakdown::default() {
            let usage = std::mem::take(&mut self.pending_usage).usage();
            self.turn.add_usage(&usage);
        }
    }

    fn close_message(&mut self) {
        if self.turn.has_open_message() {
            self.turn.close_message(None, None, None);
        }
        self.open_has_tools = false;
        for (id, name, args) in std::mem::take(&mut self.pending_starts) {
            self.turn.tool_start(&id, &name, args);
        }
    }

    /// Model content: it follows the tool calls of the open message in the
    /// next message.
    fn model_content(&mut self) {
        if self.open_has_tools {
            self.close_message();
        }
        self.begin_message();
    }

    fn tool_call(&mut self, id: &str, name: &str, args: Value) {
        if self.started_items.contains(id) || self.turn.is_tool_ended(id) {
            return;
        }
        self.started_items.insert(id.to_string());
        self.begin_message();
        self.turn
            .tool_call(&format!("c:{id}"), id, name, args.clone());
        self.open_has_tools = true;
        self.pending_starts
            .push((id.to_string(), name.to_string(), args));
    }

    fn tool_result(&mut self, id: &str, name: &str, result: AgentToolResult, is_error: bool) {
        if !self.started_items.contains(id) {
            self.tool_call(id, name, json!({}));
        }
        self.close_message();
        self.turn.tool_end(id, name, result, is_error);
    }

    fn on_internal(&mut self, event: Internal) {
        match event {
            Internal::ToolEnd {
                id,
                name,
                result,
                is_error,
            } => self.tool_result(&id, &name, result, is_error),
            Internal::Steer(message) => {
                self.close_message();
                self.turn.user_message(message);
            }
        }
    }

    fn streamed(&mut self, id: &str, text: &str, reasoning: bool, completed: bool) {
        let key = if reasoning {
            format!("r:{id}")
        } else {
            format!("t:{id}")
        };
        let emitted = if reasoning {
            &mut self.reasoning_emitted
        } else {
            &mut self.text_emitted
        };
        let last = emitted.get(id).copied();
        if last.is_none() && text.is_empty() && !completed {
            return;
        }
        let delta = match last {
            Some(last) if text.len() > last && text.is_char_boundary(last) => &text[last..],
            Some(_) => "",
            None => text,
        };
        if last.is_none() || !delta.is_empty() {
            let len = text.len().max(last.unwrap_or(0));
            emitted.insert(id.to_string(), len);
            if last.is_none() || !self.turn.has_block(&key) {
                self.model_content();
            }
            if reasoning {
                self.turn.thinking_delta(&key, delta);
            } else {
                self.turn.text_delta(&key, delta);
            }
        }
        if completed {
            self.turn.end_block(&key);
        }
    }

    fn handle(&mut self, message: &Value) {
        let method = str_field(message, "method").unwrap_or_default();
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        match method {
            "item/started" | "item/completed" => {
                if self.matches_turn(&params) {
                    self.on_item(&params, method == "item/completed");
                }
            }
            "rawResponseItem/completed" => {
                if self.matches_turn(&params) {
                    self.on_raw_item(&params);
                }
            }
            "item/agentMessage/delta" | "item/reasoning/summaryTextDelta" => {
                let (Some(id), Some(delta)) =
                    (str_field(&params, "itemId"), str_field(&params, "delta"))
                else {
                    return;
                };
                if !self.matches_turn(&params) {
                    return;
                }
                let reasoning = method == "item/reasoning/summaryTextDelta";
                let store = if reasoning {
                    &mut self.reasoning_text
                } else {
                    &mut self.agent_text
                };
                let text = store.entry(id.to_string()).or_default();
                text.push_str(delta);
                let text = text.clone();
                self.streamed(id, &text, reasoning, false);
            }
            "thread/tokenUsage/updated" => {
                if !self.matches_turn(&params) {
                    return;
                }
                let usage = params.get("tokenUsage");
                let total = UsageBreakdown::read(usage.and_then(|u| u.get("total")));
                let last = UsageBreakdown::read(usage.and_then(|u| u.get("last")));
                if let (Some(total), Some(last)) = (total, last)
                    && self.last_usage_key != Some(total)
                {
                    self.last_usage_key = Some(total);
                    if self.turn.has_open_message() {
                        self.turn.add_usage(&last.usage());
                    } else {
                        self.pending_usage.add(last);
                    }
                }
            }
            "error" => {
                if !self.matches_turn(&params) {
                    return;
                }
                let message = params
                    .get("error")
                    .and_then(|error| str_field(error, "message"))
                    .unwrap_or("Codex app-server reported an error.")
                    .to_string();
                if params.get("willRetry").and_then(Value::as_bool) == Some(true) {
                    self.notice(NoticeLevel::Warning, message);
                } else if self.turn_error.is_none() {
                    self.turn_error = Some(message);
                }
            }
            "warning" | "configWarning" | "deprecationNotice" => {
                if let Some(message) = str_field(&params, "message") {
                    self.notice(NoticeLevel::Warning, message);
                }
            }
            "turn/completed" => {
                if !self.matches_turn(&params) || self.completion.is_some() {
                    return;
                }
                let turn = params.get("turn").cloned().unwrap_or_default();
                let status = str_field(&turn, "status").unwrap_or("failed").to_string();
                if status == "completed" {
                    let pending: Vec<(String, bool, NativeToolName)> = self
                        .native_calls
                        .iter()
                        .filter(|(_, call)| !call.result_emitted)
                        .map(|(id, call)| (id.clone(), call.success_confirmed, call.name))
                        .collect();
                    for (id, confirmed, name) in pending {
                        let text = match (confirmed, name) {
                            (true, NativeToolName::ApplyPatch) => "Patch applied.",
                            (true, NativeToolName::ViewImage) => "Image viewed.",
                            (false, _) => "Codex did not report the tool result.",
                        };
                        self.native_result(&id, json!(text));
                    }
                }
                self.completion = Some(Completion {
                    status,
                    error: turn
                        .get("error")
                        .and_then(|error| str_field(error, "message"))
                        .map(str::to_string),
                });
            }
            _ => {}
        }
    }

    fn native_result(&mut self, id: &str, result: Value) {
        let Some(call) = self.native_calls.get_mut(id) else {
            return;
        };
        if call.result_emitted {
            return;
        }
        call.result_emitted = true;
        let is_error = !call.success_confirmed;
        let name = call.name.name();
        self.tool_result(id, name, result_from(result, None), is_error);
    }

    fn on_raw_item(&mut self, params: &Value) {
        let Some(item) = params.get("item").filter(|item| item.is_object()) else {
            return;
        };
        let call_id = str_field(item, "call_id").filter(|id| !id.is_empty());
        let kind = str_field(item, "type");
        let name = str_field(item, "name");
        if item.get("namespace").is_none_or(Value::is_null)
            && let Some(call_id) = call_id
        {
            let call =
                match (kind, name) {
                    (Some("custom_tool_call"), Some("apply_patch")) => str_field(item, "input")
                        .map(|input| (NativeToolName::ApplyPatch, json!(input))),
                    (Some("function_call"), Some("view_image")) => str_field(item, "arguments")
                        .map(|arguments| {
                            (
                                NativeToolName::ViewImage,
                                serde_json::from_str(arguments).unwrap_or(json!(arguments)),
                            )
                        }),
                    _ => None,
                };
            if let Some((tool, input)) = call {
                if self.native_calls.contains_key(call_id) {
                    return;
                }
                self.native_calls.insert(
                    call_id.to_string(),
                    NativeToolCall {
                        name: tool,
                        result_emitted: false,
                        success_confirmed: false,
                    },
                );
                self.tool_call(call_id, tool.name(), input);
                return;
            }
        }
        let Some(call_id) = str_field(item, "call_id") else {
            return;
        };
        let Some(call) = self.native_calls.get(call_id) else {
            return;
        };
        let expected = match call.name {
            NativeToolName::ApplyPatch => "custom_tool_call_output",
            NativeToolName::ViewImage => "function_call_output",
        };
        let Some(output) = item.get("output").filter(|_| kind == Some(expected)) else {
            return;
        };
        let result = if call.name == NativeToolName::ViewImage
            && call.success_confirmed
            && output.as_array().is_some_and(Vec::is_empty)
        {
            json!("Image viewed.")
        } else {
            output.clone()
        };
        self.native_result(call_id, result);
    }

    fn on_item(&mut self, params: &Value, completed: bool) {
        let Some(item) = params.get("item").filter(|item| item.is_object()) else {
            return;
        };
        let Some(kind) = str_field(item, "type") else {
            return;
        };
        let id = str_field(item, "id")
            .map(str::to_string)
            .unwrap_or_else(crate::harness::utils::random_uuid);
        if completed && let Some(call) = self.native_calls.get_mut(&id) {
            match (call.name, kind) {
                (NativeToolName::ApplyPatch, "fileChange") => {
                    call.success_confirmed = str_field(item, "status") == Some("completed");
                }
                (NativeToolName::ViewImage, "imageView") => call.success_confirmed = true,
                _ => {}
            }
        }
        match kind {
            "agentMessage" => {
                let text = str_field(item, "text").unwrap_or_default().to_string();
                self.agent_text.insert(id.clone(), text.clone());
                self.streamed(&id, &text, false, completed);
            }
            "reasoning" => {
                let join = |key: &str| {
                    item.get(key)
                        .and_then(Value::as_array)
                        .map(|parts| {
                            parts
                                .iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join("\n\n")
                        })
                        .unwrap_or_default()
                };
                let summary = join("summary");
                let content = join("content");
                let streamed = self.reasoning_text.get(&id).cloned().unwrap_or_default();
                let text = [summary, streamed, content]
                    .into_iter()
                    .find(|text| !text.is_empty())
                    .unwrap_or_default();
                self.reasoning_text.insert(id.clone(), text.clone());
                self.streamed(&id, &text, true, completed);
            }
            "commandExecution" => {
                if completed {
                    let status = match str_field(item, "status") {
                        Some("completed") => "completed",
                        Some("inProgress") => "in_progress",
                        _ => "failed",
                    };
                    let output = str_field(item, "aggregatedOutput")
                        .unwrap_or_default()
                        .to_string();
                    let details = json!({
                        "exitCode": item.get("exitCode").cloned().unwrap_or(Value::Null),
                        "output": output,
                        "status": status,
                    });
                    self.tool_result(&id, "bash", result_from(details, Some(output)), false);
                } else {
                    let command = str_field(item, "command").unwrap_or_default();
                    self.tool_call(&id, "bash", json!({ "command": command }));
                }
            }
            "mcpToolCall" => {
                let tool = str_field(item, "tool").unwrap_or("unknown");
                let name = match str_field(item, "server").filter(|server| !server.is_empty()) {
                    Some(server) => format!("mcp__{server}__{tool}"),
                    None => tool.to_string(),
                };
                if completed {
                    let result = match item.get("result").filter(|result| result.is_object()) {
                        Some(result) => result
                            .get("structuredContent")
                            .filter(|value| !value.is_null())
                            .or_else(|| result.get("content"))
                            .cloned()
                            .unwrap_or(Value::Null),
                        None => match item
                            .get("error")
                            .and_then(|error| str_field(error, "message"))
                        {
                            Some(message) => json!({ "error": message }),
                            None => Value::Null,
                        },
                    };
                    let failed = str_field(item, "status") == Some("failed");
                    self.tool_result(&id, &name, result_from(result, None), failed);
                } else {
                    let arguments = item.get("arguments").cloned().unwrap_or_else(|| json!({}));
                    self.tool_call(&id, &name, arguments);
                }
            }
            "webSearch" => {
                let query = str_field(item, "query")
                    .filter(|query| !query.is_empty())
                    .or_else(|| {
                        item.get("action")
                            .and_then(|action| str_field(action, "query"))
                            .filter(|query| !query.is_empty())
                    });
                if completed {
                    if !self.web_search_started.contains(&id) {
                        self.tool_call(&id, "webSearch", json!({ "query": query.unwrap_or("") }));
                    }
                    self.web_search_started.remove(&id);
                    let result = item
                        .get("results")
                        .filter(|value| !value.is_null())
                        .or_else(|| item.get("action").filter(|value| value.is_object()))
                        .cloned()
                        .unwrap_or(Value::Null);
                    self.tool_result(&id, "webSearch", result_from(result, None), false);
                } else if let Some(query) = query
                    && self.web_search_started.insert(id.clone())
                {
                    self.tool_call(&id, "webSearch", json!({ "query": query }));
                }
            }
            "fileChange" if completed => {
                for change in item
                    .get("changes")
                    .and_then(Value::as_array)
                    .map_or(&[][..], Vec::as_slice)
                {
                    let Some(path) = str_field(change, "path") else {
                        continue;
                    };
                    let kind = match change.get("kind").and_then(|kind| str_field(kind, "type")) {
                        Some("add") => FileChangeKind::Create,
                        Some("delete") => FileChangeKind::Delete,
                        Some("update") => FileChangeKind::Modify,
                        _ => continue,
                    };
                    self.turn.emit_harness(HarnessEvent::FileChange {
                        kind,
                        path: path.to_string(),
                    });
                }
            }
            "error" if completed => {
                let message = str_field(item, "message")
                    .filter(|message| !message.trim().is_empty())
                    .unwrap_or("codex reported a non-fatal error item");
                self.notice(NoticeLevel::Warning, message);
            }
            _ => {}
        }
    }
}

/// A dynamic tool call (`item/tool/call`): the caller's tool, run by the
/// host. Returns the call to run, or the error to answer with.
fn dynamic_tool_call(
    params: &Value,
    dynamic_tools: &DynamicTools,
) -> Result<(String, String, Value), String> {
    let (Some(call_id), Some(tool)) = (str_field(params, "callId"), str_field(params, "tool"))
    else {
        return Err("Codex app-server sent an invalid dynamic tool request.".into());
    };
    let Some(name) = dynamic_tools.original_name_by_alias.get(tool) else {
        return Err(format!(
            "Codex app-server requested unknown dynamic tool '{tool}'."
        ));
    };
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    Ok((call_id.to_string(), name.clone(), arguments))
}

/// A host tool result as `item/tool/call` content items.
fn dynamic_tool_response(result: &AgentToolResult, is_error: bool) -> Value {
    let items: Vec<Value> = result
        .content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => json!({ "type": "inputText", "text": text.text }),
            ToolResultContent::Image(ImageContent { data, mime_type }) => json!({
                "type": "inputImage",
                "imageUrl": format!("data:{mime_type};base64,{data}"),
            }),
        })
        .collect();
    json!({ "contentItems": items, "success": !is_error })
}

enum TurnFailure {
    Aborted,
    Error(String),
}

impl From<crate::harness::error::BoxError> for TurnFailure {
    fn from(error: crate::harness::error::BoxError) -> Self {
        Self::Error(error.to_string())
    }
}

async fn race<T>(
    cancel: &CancellationToken,
    operation: impl std::future::Future<Output = HarnessResult<T>>,
) -> Result<T, TurnFailure> {
    tokio::select! {
        result = operation => result.map_err(TurnFailure::from),
        _ = cancel.cancelled() => Err(TurnFailure::Aborted),
    }
}

/// Run one turn. Returns the turn's new messages.
pub(super) async fn run_turn(
    runtime: &mut Runtime,
    config: TurnConfig,
    prompt: AgentMessage,
    sender: HarnessTurnSender,
    cancel: CancellationToken,
) -> AgentResult<Vec<AgentMessage>> {
    let mut turn = TurnBuilder::new(
        sender,
        super::CODEX_API,
        super::CODEX_PROVIDER,
        config.model.clone(),
    );
    turn.start(prompt.clone());
    let mut translator = Translator {
        turn,
        thread_id: String::new(),
        turn_id: String::new(),
        text_emitted: HashMap::new(),
        reasoning_emitted: HashMap::new(),
        reasoning_text: HashMap::new(),
        agent_text: HashMap::new(),
        open_has_tools: false,
        pending_starts: Vec::new(),
        native_calls: IndexMap::new(),
        web_search_started: HashSet::new(),
        started_items: HashSet::new(),
        pending_usage: UsageBreakdown::default(),
        last_usage_key: None,
        turn_error: None,
        completion: None,
    };
    let outcome = drive(runtime, &config, &prompt, &mut translator, &cancel).await;
    *config.steer.lock().unwrap() = None;
    let error = match outcome {
        Ok(()) => None,
        Err(TurnFailure::Aborted) => Some((StopReason::Aborted, "Operation aborted".to_string())),
        Err(TurnFailure::Error(message)) => Some((StopReason::Error, message)),
    };
    if error.is_none() {
        translator.close_message();
    }
    Ok(translator.turn.finish(error))
}

async fn drive(
    runtime: &mut Runtime,
    config: &TurnConfig,
    prompt: &AgentMessage,
    translator: &mut Translator,
    cancel: &CancellationToken,
) -> Result<(), TurnFailure> {
    if cancel.is_cancelled() {
        return Err(TurnFailure::Aborted);
    }
    let (codex_config, provider_overrides) =
        split_model_provider_config(&config.codex_config).map_err(TurnFailure::from)?;
    let policy = resolve_codex_builtin_tool_policy(config.filter.as_ref(), config.web_search);
    let hook = if policy.deny_apply_patch_with_hook {
        Some(create_trusted_apply_patch_hook(&codex_config).map_err(TurnFailure::from)?)
    } else {
        None
    };
    let mut thread_config = hook
        .as_ref()
        .map_or_else(|| codex_config.clone(), |hook| hook.thread_config.clone());
    thread_config.insert(
        "web_search".into(),
        json!(if policy.web_search_live {
            "live"
        } else {
            "disabled"
        }),
    );
    if policy.bash_disabled || policy.view_image_disabled {
        let mut features = codex_config
            .get("features")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        if policy.bash_disabled {
            features.insert("shell_tool".into(), json!(false));
        }
        if policy.view_image_disabled {
            features.insert("view_image".into(), json!(false));
        }
        thread_config.insert("features".into(), Value::Object(features));
    }
    let mut cli_overrides = provider_overrides;
    if let Some(hook) = &hook {
        cli_overrides.extend(hook.cli_overrides.iter().cloned());
    }
    let config_key = json!({ "thread": thread_config, "cliOverrides": cli_overrides }).to_string();
    if runtime.client.is_some() && runtime.config_key != config_key {
        runtime.close().await;
    }
    let host_tool_definitions: Vec<crate::Tool> = config
        .host
        .tools
        .iter()
        .map(|tool| tool.definition())
        .collect();
    let dynamic_tools = create_dynamic_tools(&host_tool_definitions);

    let initialized = runtime.client.is_none();
    if initialized {
        let command = std::iter::once(config.executable.clone())
            .chain(
                app_server_args(&cli_overrides)
                    .iter()
                    .map(|arg| shell_quote(arg)),
            )
            .collect::<Vec<_>>()
            .join(" ");
        let options = SandboxProcessOptions {
            command,
            working_directory: Some(config.work_dir.clone()),
            env: Some(config.env.clone()),
            abort_signal: None,
        };
        let client = match &config.shared {
            Some(shared) => shared
                .get_or_spawn(&config.sandbox, options)
                .await
                .map_err(TurnFailure::from)?,
            None => AppServer::spawn(config.sandbox.as_ref(), options, false)
                .await
                .map_err(TurnFailure::from)?,
        };
        runtime.client = Some(client);
        runtime.config_key = config_key;
    }
    let client = runtime.client.clone().unwrap();
    let result = drive_with_client(
        runtime,
        &client,
        initialized,
        config,
        prompt,
        translator,
        cancel,
        &thread_config,
        hook.as_ref(),
        &dynamic_tools,
        policy.disable_environments,
    )
    .await;
    match result {
        Ok(()) => Ok(()),
        Err(DriveFailure::Kept(failure)) => Err(failure),
        Err(DriveFailure::Closed(failure)) => {
            runtime.close().await;
            Err(failure)
        }
    }
}

enum DriveFailure {
    /// The app server stays usable.
    Kept(TurnFailure),
    Closed(TurnFailure),
}

#[allow(clippy::too_many_arguments)]
async fn drive_with_client(
    runtime: &mut Runtime,
    client: &Arc<AppServer>,
    initialized: bool,
    config: &TurnConfig,
    prompt: &AgentMessage,
    translator: &mut Translator,
    cancel: &CancellationToken,
    thread_config: &Map<String, Value>,
    hook: Option<&super::tools::TrustedApplyPatchHook>,
    dynamic_tools: &DynamicTools,
    disable_environments: bool,
) -> Result<(), DriveFailure> {
    let closed = DriveFailure::Closed;
    if initialized {
        race(cancel, client.initialize()).await.map_err(closed)?;
    }
    if config.restart_thread
        && let Some(loaded) = runtime.loaded_thread_id.take()
    {
        runtime.loaded_thread_model = None;
        race(
            cancel,
            client.request("thread/unsubscribe", json!({ "threadId": loaded })),
        )
        .await
        .map_err(closed)?;
    }
    if let Some(hook) = hook {
        let hooks = race(cancel, client.request("hooks/list", json!({})))
            .await
            .map_err(closed)?;
        assert_trusted_apply_patch_hook(&hooks, hook)
            .map_err(|error| closed(TurnFailure::from(error)))?;
    }
    let thread_id = match &config.thread_id {
        Some(thread_id) if runtime.loaded_thread_id.as_ref() == Some(thread_id) => {
            thread_id.clone()
        }
        _ => {
            let mut params = thread_params(config, thread_config);
            let method = match &config.thread_id {
                None => {
                    params.insert("dynamicTools".into(), json!(dynamic_tools.specs));
                    params.insert("experimentalRawEvents".into(), json!(true));
                    "thread/start"
                }
                Some(thread_id) => {
                    params.insert("threadId".into(), json!(thread_id));
                    params.insert("excludeTurns".into(), json!(true));
                    "thread/resume"
                }
            };
            let response = race(cancel, client.request(method, Value::Object(params)))
                .await
                .map_err(closed)?;
            assert_thread_permissions(&response, method)
                .map_err(|error| closed(TurnFailure::from(error)))?;
            let thread_id = nested_string(&response, &["thread", "id"], method)
                .map_err(|error| closed(TurnFailure::from(error)))?;
            runtime.loaded_thread_id = Some(thread_id.clone());
            runtime.loaded_thread_model = Some((
                config.codex_model.clone(),
                str_field(&response, "model")
                    .filter(|model| !model.is_empty())
                    .map(str::to_string),
            ));
            thread_id
        }
    };
    *config.thread_id_out.lock().unwrap() = Some(thread_id.clone());
    translator.thread_id = thread_id.clone();
    if let Some((requested, Some(reported))) = &runtime.loaded_thread_model
        && *requested == config.codex_model
    {
        translator.turn.set_model(reported.clone());
    }

    let mut params = Map::new();
    params.insert("threadId".into(), json!(thread_id));
    if disable_environments {
        params.insert("environments".into(), json!([]));
    }
    params.insert("input".into(), json!(input_items(prompt)));
    params.insert("approvalPolicy".into(), json!("never"));
    params.insert(
        "sandboxPolicy".into(),
        json!({ "type": "externalSandbox", "networkAccess": "enabled" }),
    );
    if let Some(model) = &config.codex_model {
        params.insert("model".into(), json!(model));
    }
    if let Some(effort) = config.effort {
        params.insert("effort".into(), json!(effort));
    }
    if let Some(schema) = &config.output_schema {
        params.insert("outputSchema".into(), schema.clone());
    }
    // Only this thread's messages, from here on.
    let mut events = client.subscribe(&thread_id);
    let response = race(cancel, client.request("turn/start", Value::Object(params)))
        .await
        .map_err(closed)?;
    let turn_id = nested_string(&response, &["turn", "id"], "turn/start")
        .map_err(|error| closed(TurnFailure::from(error)))?;
    translator.turn_id = turn_id.clone();

    let (internal_sender, mut internal) = mpsc::unbounded_channel();
    *config.steer.lock().unwrap() = Some(SteerHandle {
        client: client.clone(),
        thread_id: thread_id.clone(),
        turn_id: turn_id.clone(),
        internal: internal_sender.clone(),
    });

    let mut interrupted = false;
    let mut give_up_at: Option<tokio::time::Instant> = None;
    let mut protocol_error: Option<String> = None;
    loop {
        let give_up = async {
            match give_up_at {
                Some(at) => tokio::time::sleep_until(at).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            message = events.recv() => {
                let Some(message) = message else {
                    let tail = client.stderr_tail();
                    let mut error = "Codex app-server exited before the turn completed.".to_string();
                    if !tail.is_empty() {
                        error = format!("{error}\n{tail}");
                    }
                    return Err(closed(if interrupted { TurnFailure::Aborted } else { TurnFailure::Error(error) }));
                };
                if message.get("id").is_some() && message.get("method").is_some() {
                    if let Err(error) = on_server_request(
                        client,
                        &message,
                        translator,
                        dynamic_tools,
                        &config.host,
                        &internal_sender,
                    )
                    .await
                    {
                        protocol_error.get_or_insert(error);
                        break;
                    }
                } else {
                    translator.handle(&message);
                }
            }
            Some(event) = internal.recv() => translator.on_internal(event),
            _ = cancel.cancelled(), if !interrupted => {
                interrupted = true;
                let interrupt_client = client.clone();
                let params = json!({ "threadId": thread_id, "turnId": turn_id });
                tokio::spawn(async move {
                    let _ = interrupt_client.request("turn/interrupt", params).await;
                });
                give_up_at = Some(tokio::time::Instant::now() + INTERRUPT_GRACE);
            }
            _ = give_up => {
                return Err(closed(TurnFailure::Aborted));
            }
        }
        if translator.completion.is_some() {
            break;
        }
    }
    *config.steer.lock().unwrap() = None;
    while let Ok(event) = internal.try_recv() {
        translator.on_internal(event);
    }
    drop(events);
    if let Some(error) = protocol_error {
        return Err(closed(TurnFailure::Error(error)));
    }
    if interrupted {
        return Err(DriveFailure::Kept(TurnFailure::Aborted));
    }
    let completion = translator.completion.take().unwrap();
    match completion.status.as_str() {
        "completed" => {}
        "interrupted" => return Err(DriveFailure::Kept(TurnFailure::Aborted)),
        status => {
            let error = completion
                .error
                .or_else(|| translator.turn_error.take())
                .unwrap_or_else(|| format!("Codex turn ended with status '{status}'."));
            return Err(DriveFailure::Kept(TurnFailure::Error(error)));
        }
    }
    if let Some(error) = translator.turn_error.take() {
        return Err(DriveFailure::Kept(TurnFailure::Error(error)));
    }
    // Background terminals the turn started belong to this app server; the
    // turn stays open until they exit.
    loop {
        let response = race(
            cancel,
            client.request(
                "thread/backgroundTerminals/list",
                json!({ "threadId": thread_id, "limit": 1 }),
            ),
        )
        .await
        .map_err(|failure| match failure {
            TurnFailure::Aborted => DriveFailure::Kept(TurnFailure::Aborted),
            other => closed(other),
        })?;
        let Some(terminals) = response.get("data").and_then(Value::as_array) else {
            return Err(closed(TurnFailure::Error(
                "Codex app-server thread/backgroundTerminals/list returned an invalid response."
                    .into(),
            )));
        };
        if terminals.is_empty() {
            break;
        }
        tokio::select! {
            _ = tokio::time::sleep(BACKGROUND_TERMINAL_POLL_INTERVAL) => {}
            _ = cancel.cancelled() => return Err(DriveFailure::Kept(TurnFailure::Aborted)),
        }
    }
    Ok(())
}

/// Answer a server request. `Err` fails the turn.
async fn on_server_request(
    client: &Arc<AppServer>,
    message: &Value,
    translator: &mut Translator,
    dynamic_tools: &DynamicTools,
    host: &HostTools,
    internal: &mpsc::UnboundedSender<Internal>,
) -> Result<(), String> {
    let id = message.get("id").cloned().unwrap_or(Value::Null);
    let method = str_field(message, "method").unwrap_or_default();
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    if !translator.matches_turn(&params) {
        client
            .respond(
                &id,
                Err("Codex app-server requested a tool for an inactive turn.".into()),
            )
            .await;
        return Ok(());
    }
    if method != "item/tool/call" {
        let error = format!("Codex app-server requested unsupported method '{method}'.");
        client.respond(&id, Err(error.clone())).await;
        return Err(error);
    }
    let (call_id, name, arguments) = match dynamic_tool_call(&params, dynamic_tools) {
        Ok(call) => call,
        Err(error) => {
            client.respond(&id, Err(error.clone())).await;
            return Err(error);
        }
    };
    translator.tool_call(&call_id, &name, arguments.clone());
    let assistant = translator.turn.current_assistant();
    let call = ToolCall {
        id: call_id.clone(),
        name: name.clone(),
        arguments,
        thought_signature: None,
    };
    let host = host.clone();
    let client = client.clone();
    let internal = internal.clone();
    let model = translator.turn.model().to_string();
    tokio::spawn(async move {
        let assistant = assistant.unwrap_or_else(|| empty_assistant(&model));
        let (result, is_error) = host.run(&call, &assistant, None).await;
        client
            .respond(&id, Ok(dynamic_tool_response(&result, is_error)))
            .await;
        let _ = internal.send(Internal::ToolEnd {
            id: call_id,
            name,
            result,
            is_error,
        });
    });
    Ok(())
}

fn empty_assistant(model: &str) -> crate::AssistantMessage {
    crate::AssistantMessage {
        content: Vec::new(),
        api: super::CODEX_API.to_string(),
        provider: super::CODEX_PROVIDER.to_string(),
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
