//! Port of the part of libfx's `engine.js` (`createFxEngine`) that fx's
//! harness uses: one session on one fx core, without persistence, speaking
//! ACP to the core over its stdin and stdout.
//!
//! [rust] Not ported: persistence (the journal, images stores, snapshots,
//! follow-ups, resume and handoff), the native addon backend,
//! `resizeImage`, custom providers, `modelCatalog`, `onPermission` and the
//! diagnostic `onEvent` stream. The events a turn yields, the hook events,
//! host tool calls and the transport's own signals reach the caller as one
//! ordered stream of [`FxEngineEvent`]s instead of an iterator, a hook
//! callback and callbacks.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use base64::Engine as _;
use futures::future::BoxFuture;
use serde_json::{Map, Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use super::authorizer::{SubscriptionAuthorizer, endpoint_of};
use super::fetch::{FxFetch, FxFetchError, FxHttpRequest, FxHttpResponse};
use super::host::{self, CoreItem, FxCore, HostToolReply};
use crate::harness::error::{HarnessError, HarnessResult};

/// `fxSdkApiVersion`.
pub const FX_SDK_API_VERSION: u32 = 2;
const MAX_INSTRUCTIONS_BYTES: usize = 64 * 1024;
const MAX_MODEL_BYTES: usize = 1024;
const MAX_EFFORT_BYTES: usize = 64;
const MAX_CHECKPOINT_BYTES: usize = 4 * 1024 * 1024;
const MAX_TOOL_START_INPUT_BYTES: usize = 64 * 1024;
const MAX_PROMPT_IMAGES: usize = 8;
const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// A host tool the core may call.
#[derive(Debug, Clone, PartialEq)]
pub struct FxToolDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// An image in a prompt or a tool result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxImage {
    /// Base64.
    pub data: String,
    pub mime_type: String,
}

/// `FxToolResult`: a typed (`libfx.tool-result`) tool result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FxToolResult {
    pub text: String,
    pub images: Vec<FxImage>,
    pub is_error: bool,
}

impl FxToolResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
            is_error: true,
        }
    }
}

/// One prompt block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FxPromptBlock {
    Text(String),
    Image(FxImage),
}

/// `FxModelEnd`: the `model.end` hook event's data.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FxModelEnd {
    pub finish_reason: Option<String>,
    pub usage: Option<FxModelUsage>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FxModelUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

/// `FxRuntimeEvent`: what a turn yields.
#[derive(Debug, Clone, PartialEq)]
pub enum FxRuntimeEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolStart {
        id: String,
        name: String,
        input: Option<Value>,
    },
    ToolEnd {
        id: String,
        name: String,
        content: Option<String>,
        is_error: bool,
    },
    UserMessage(String),
}

/// What the engine tells its caller, in the order it happens.
pub enum FxEngineEvent {
    /// The running turn's event.
    Turn(FxRuntimeEvent),
    /// A `model.end` hook event.
    ModelEnd(FxModelEnd),
    /// The core runs a host tool. Dropping `reply` cancels the call.
    ToolCall {
        name: String,
        input: Value,
        call_id: String,
        signal: CancellationToken,
        reply: oneshot::Sender<FxToolResult>,
    },
}

/// How the engine reaches the model.
#[derive(Clone)]
pub enum FxEngineProvider {
    /// AI Gateway, with its key.
    Gateway { api_key: String },
    /// A Codex or Grok subscription.
    Subscription(Arc<SubscriptionAuthorizer>),
}

/// `createFxEngine` options.
#[derive(Clone)]
pub struct FxEngineOptions {
    pub provider: FxEngineProvider,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub fast: Option<bool>,
    pub ultrafast: Option<bool>,
    pub instructions: String,
    pub tools: Vec<FxToolDescriptor>,
    pub checkpoint: Option<Vec<u8>>,
    /// The host's fetch; the provider's credentials are added around it.
    pub fetch: FxFetch,
    /// Asks the core for semantic hook events (`model.end`).
    pub hooks: bool,
    pub events: mpsc::UnboundedSender<FxEngineEvent>,
}

/// `{ stopReason, usage }`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FxTurnResult {
    pub stop_reason: String,
    pub usage: FxModelUsage,
}

/// A running turn's controls.
pub trait FxTurnControl: Send + Sync {
    fn steer(&self, text: &str) -> HarnessResult<()>;
    fn cancel(&self);
}

/// A running turn.
pub struct FxTurn {
    pub result: BoxFuture<'static, HarnessResult<FxTurnResult>>,
    pub control: Arc<dyn FxTurnControl>,
}

/// `FxEngine`.
#[async_trait]
pub trait FxEngine: Send + Sync {
    /// Starts a turn.
    fn prompt(&self, prompt: Vec<FxPromptBlock>) -> HarnessResult<FxTurn>;
    /// The session's state as opaque bytes.
    async fn checkpoint(&self) -> HarnessResult<Vec<u8>>;
    async fn close(&self);
}

/// Creates an engine (`createFxRuntime`).
pub type FxRuntimeFactory = Arc<
    dyn Fn(FxEngineOptions) -> BoxFuture<'static, HarnessResult<Arc<dyn FxEngine>>> + Send + Sync,
>;

fn bounded(value: Option<&str>, name: &str, max: usize) -> HarnessResult<()> {
    if let Some(value) = value {
        if value.is_empty() {
            return Err(HarnessError::new(format!("{name} must be a non-empty string")).into());
        }
        if value.len() > max {
            return Err(
                HarnessError::new(format!("{name} exceeds the {max} byte libfx limit")).into(),
            );
        }
    }
    Ok(())
}

fn validate(options: &FxEngineOptions) -> HarnessResult<()> {
    bounded(options.model.as_deref(), "model", MAX_MODEL_BYTES)?;
    bounded(options.effort.as_deref(), "effort", MAX_EFFORT_BYTES)?;
    if let Some(effort) = &options.effort
        && !effort
            .chars()
            .all(|char| char.is_ascii_alphanumeric() || matches!(char, '.' | '_' | '-'))
    {
        return Err(HarnessError::new(
            "effort must use only letters, digits, \".\", \"-\", or \"_\"",
        )
        .into());
    }
    if let FxEngineProvider::Subscription(authorizer) = &options.provider
        && options.model.is_none()
    {
        return Err(HarnessError::new(format!(
            "provider \"{}\" needs a model; fx's default model is an AI Gateway id",
            authorizer.provider().as_str()
        ))
        .into());
    }
    if options.instructions.len() > MAX_INSTRUCTIONS_BYTES {
        return Err(HarnessError::new(format!(
            "instructions exceed the {MAX_INSTRUCTIONS_BYTES} byte libfx limit"
        ))
        .into());
    }
    let mut names = HashSet::new();
    for (index, tool) in options.tools.iter().enumerate() {
        let valid = !tool.name.is_empty()
            && tool.name.len() <= 64
            && tool
                .name
                .chars()
                .all(|char| char.is_ascii_alphanumeric() || matches!(char, '_' | '-'));
        if !valid {
            return Err(HarnessError::new(format!("tool {index} has an invalid name")).into());
        }
        if !names.insert(tool.name.as_str()) {
            return Err(HarnessError::new(format!("duplicate tool name: {}", tool.name)).into());
        }
        if !tool.input_schema.is_object() {
            return Err(HarnessError::new(format!(
                "tool {} requires an object inputSchema",
                tool.name
            ))
            .into());
        }
    }
    if options
        .checkpoint
        .as_ref()
        .is_some_and(|checkpoint| checkpoint.len() > MAX_CHECKPOINT_BYTES)
    {
        return Err(HarnessError::new("libfx checkpoint is too large").into());
    }
    Ok(())
}

/// `agentEnvironment(options)`.
pub(crate) fn agent_environment(options: &FxEngineOptions) -> Vec<(String, String)> {
    let mut env = Vec::new();
    match &options.provider {
        FxEngineProvider::Gateway { api_key } => {
            env.push(("AI_GATEWAY_API_KEY".into(), api_key.clone()));
        }
        FxEngineProvider::Subscription(authorizer) => {
            env.push(("FX_PROVIDER".into(), authorizer.provider().as_str().into()));
        }
    }
    if let Some(model) = &options.model {
        env.push(("FX_MODEL".into(), model.clone()));
    }
    if let Some(effort) = &options.effort {
        env.push(("FX_EFFORT".into(), effort.clone()));
    }
    if let Some(fast) = options.fast {
        env.push(("FX_FAST".into(), fast.to_string()));
    }
    if let Some(ultrafast) = options.ultrafast {
        env.push(("FX_ULTRAFAST".into(), ultrafast.to_string()));
    }
    env
}

fn descriptor_json(tool: &FxToolDescriptor) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    })
}

/// `toolSchemaHash(descriptors)`: FNV-1a of the sorted descriptors' JSON.
pub(crate) fn tool_schema_hash(tools: &[FxToolDescriptor]) -> String {
    let mut sorted: Vec<&FxToolDescriptor> = tools.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let shape = Value::Array(sorted.into_iter().map(descriptor_json).collect()).to_string();
    let mut hash: u32 = 0x811c_9dc5;
    for byte in shape.bytes() {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }
    format!("{hash:08x}")
}

/// The content `hostToolContent` sends the core for a typed tool result.
pub(crate) fn host_tool_content(result: &FxToolResult) -> String {
    let images: Vec<Value> = result
        .images
        .iter()
        .take(8)
        .map(|image| json!({ "type": "image", "data": image.data, "mimeType": image.mime_type }))
        .collect();
    json!({ "text": result.text, "images": images }).to_string()
}

fn usage_of(value: Option<&Value>) -> FxModelUsage {
    let get = |name: &str| value.and_then(|value| value.get(name)?.as_u64());
    FxModelUsage {
        input_tokens: get("inputTokens"),
        output_tokens: get("outputTokens"),
        cache_read_tokens: get("cacheReadTokens"),
        cache_write_tokens: get("cacheWriteTokens"),
        reasoning_tokens: get("reasoningTokens"),
    }
}

/// `toolStartInput(rawInput)`: the input when it fits the preview budget.
fn tool_start_input(raw: Option<&Value>) -> Option<Value> {
    let raw = raw.filter(|raw| !raw.is_null())?;
    (raw.to_string().len() <= MAX_TOOL_START_INPUT_BYTES).then(|| raw.clone())
}

/// `normalizeTurn`'s `eventFor(update)`.
struct TurnEvents {
    tool_names: HashMap<String, String>,
    started: HashSet<String>,
}

impl TurnEvents {
    fn new() -> Self {
        Self {
            tool_names: HashMap::new(),
            started: HashSet::new(),
        }
    }

    fn event_for(&mut self, update: &Value) -> Option<FxRuntimeEvent> {
        let kind = update.get("sessionUpdate")?.as_str()?;
        let text = || {
            update
                .get("content")
                .and_then(|content| content.get("text"))
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(str::to_string)
        };
        match kind {
            "agent_message_chunk" => text()
                .filter(|text| !text.starts_with("[context]"))
                .map(FxRuntimeEvent::TextDelta),
            "agent_thought_chunk" => text().map(FxRuntimeEvent::ReasoningDelta),
            "user_message_chunk" => {
                let content = update.get("content")?;
                if content.get("type").and_then(Value::as_str) != Some("text") {
                    return None;
                }
                Some(FxRuntimeEvent::UserMessage(
                    content.get("text")?.as_str()?.to_string(),
                ))
            }
            "tool_call" => {
                let id = update.get("toolCallId")?.as_str()?.to_string();
                let name = ["name", "toolName", "title"]
                    .iter()
                    .find_map(|key| {
                        update
                            .get(*key)
                            .and_then(Value::as_str)
                            .filter(|name| !name.is_empty())
                    })
                    .unwrap_or("tool")
                    .to_string();
                self.tool_names.insert(id.clone(), name.clone());
                if !self.started.insert(id.clone()) {
                    return None;
                }
                Some(FxRuntimeEvent::ToolStart {
                    id,
                    name,
                    input: tool_start_input(update.get("rawInput")),
                })
            }
            "tool_call_update" => {
                let status = update.get("status").and_then(Value::as_str)?;
                if status != "completed" && status != "failed" {
                    return None;
                }
                let id = update.get("toolCallId")?.as_str()?.to_string();
                let content = update
                    .get("content")
                    .and_then(Value::as_array)
                    .and_then(|entries| {
                        entries.iter().find_map(|entry| {
                            let content = entry.get("content")?;
                            (content.get("type")?.as_str()? == "text")
                                .then(|| content.get("text")?.as_str().map(str::to_string))
                                .flatten()
                        })
                    });
                Some(FxRuntimeEvent::ToolEnd {
                    name: self
                        .tool_names
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| "tool".into()),
                    id,
                    content,
                    is_error: status == "failed",
                })
            }
            _ => None,
        }
    }
}

type Pending = oneshot::Sender<Result<Value, HarnessError>>;

struct ActiveTurn {
    cancelled: CancellationToken,
    events: TurnEvents,
}

/// What the engine and its dispatcher share.
struct Inner {
    core: FxCore,
    pending: Mutex<HashMap<u64, Pending>>,
    next_id: AtomicU64,
    session_id: Mutex<Option<String>>,
    turn: Mutex<Option<ActiveTurn>>,
    closing: AtomicBool,
    events: mpsc::UnboundedSender<FxEngineEvent>,
    exit_error: Mutex<Option<String>>,
    next_attachment: Mutex<u32>,
}

impl Inner {
    fn send(&self, message: &Value) -> HarnessResult<()> {
        if self.closing.load(Ordering::SeqCst) {
            return Err(HarnessError::new("fx agent is closing").into());
        }
        let mut frame = message.to_string();
        frame.push('\n');
        self.core.write(frame.as_bytes())
    }

    async fn request(&self, method: &str, params: Value) -> HarnessResult<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, sender);
        if let Some(error) = self.exit_error.lock().unwrap().clone() {
            self.pending.lock().unwrap().remove(&id);
            return Err(HarnessError::new(error).into());
        }
        if let Err(error) =
            self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))
        {
            self.pending.lock().unwrap().remove(&id);
            return Err(error);
        }
        match receiver.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => Err(error.into()),
            Err(_) => Err(HarnessError::new(
                self.exit_error
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| "fx agent closed".into()),
            )
            .into()),
        }
    }

    /// `attachBytes(payloads)`.
    fn attach(&self, payloads: &[Vec<u8>]) -> HarnessResult<Vec<u32>> {
        self.core.discard_attachments();
        let mut next = self.next_attachment.lock().unwrap();
        payloads
            .iter()
            .map(|bytes| {
                let id = *next;
                *next = if id == 0x7fff_ffff { 1 } else { id + 1 };
                self.core.write_attachment(id, bytes)?;
                Ok(id)
            })
            .collect()
    }

    fn session_id(&self) -> Value {
        self.session_id
            .lock()
            .unwrap()
            .clone()
            .map_or(Value::Null, Value::String)
    }

    fn current_turn_cancelled(&self) -> Option<CancellationToken> {
        self.turn
            .lock()
            .unwrap()
            .as_ref()
            .map(|turn| turn.cancelled.clone())
            .filter(|token| !token.is_cancelled())
    }

    /// `executeHostTool(name, input, sessionId, toolCallId)`.
    async fn execute_host_tool(
        &self,
        name: String,
        input: Value,
        call_id: String,
    ) -> HostToolReply {
        let cancelled = HostToolReply {
            cancelled: true,
            is_error: true,
            ..Default::default()
        };
        let Some(turn) = self.current_turn_cancelled() else {
            return cancelled;
        };
        if self.closing.load(Ordering::SeqCst) {
            return cancelled;
        }
        let signal = turn.child_token();
        let (reply, receiver) = oneshot::channel();
        if self
            .events
            .send(FxEngineEvent::ToolCall {
                name: name.clone(),
                input,
                call_id,
                signal: signal.clone(),
                reply,
            })
            .is_err()
        {
            return cancelled;
        }
        let result = tokio::select! {
            result = receiver => result.ok(),
            _ = signal.cancelled() => None,
        };
        let Some(mut result) = result else {
            return cancelled;
        };
        if turn.is_cancelled() {
            return cancelled;
        }
        // Providers reject an empty error result, which would end the session.
        if result.is_error && result.text.is_empty() && result.images.is_empty() {
            result.text = format!("Tool {name} failed without a message");
        }
        let mut content = host_tool_content(&result);
        if content.len() + 64 > MAX_FRAME_BYTES {
            content = host_tool_content(&FxToolResult {
                images: Vec::new(),
                ..result.clone()
            });
        }
        HostToolReply {
            content,
            is_error: result.is_error,
            rich: true,
            cancelled: false,
        }
    }

    /// The line handler.
    fn on_line(self: &Arc<Self>, message: Value) {
        let method = message.get("method").and_then(Value::as_str);
        let session_id = self.session_id();
        match method {
            Some("libfx/hook") => {
                let params = message.get("params");
                let same =
                    params.and_then(|params| params.get("rootSessionId")) == Some(&session_id);
                let version = params
                    .and_then(|params| params.get("version"))
                    .and_then(Value::as_u64);
                if same
                    && version == Some(1)
                    && params
                        .and_then(|params| params.get("type"))
                        .and_then(Value::as_str)
                        == Some("model.end")
                {
                    let data = params.and_then(|params| params.get("data"));
                    let _ = self.events.send(FxEngineEvent::ModelEnd(FxModelEnd {
                        finish_reason: data
                            .and_then(|data| data.get("finishReason"))
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        usage: data
                            .and_then(|data| data.get("usage"))
                            .filter(|usage| usage.is_object())
                            .map(|usage| usage_of(Some(usage))),
                    }));
                }
            }
            Some("libfx/journal_append") => {}
            Some("session/update") => {
                let params = message.get("params");
                if params.and_then(|params| params.get("sessionId")) != Some(&session_id) {
                    return;
                }
                let Some(update) = params.and_then(|params| params.get("update")) else {
                    return;
                };
                let event = {
                    let mut turn = self.turn.lock().unwrap();
                    match turn.as_mut() {
                        Some(turn) if !turn.cancelled.is_cancelled() => {
                            turn.events.event_for(update)
                        }
                        _ => None,
                    }
                };
                if let Some(event) = event {
                    let _ = self.events.send(FxEngineEvent::Turn(event));
                }
            }
            Some("session/request_permission") => {
                if self.current_turn_cancelled().is_some() {
                    let _ = self.send(&json!({
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "result": { "outcome": { "outcome": "cancelled" } },
                    }));
                }
            }
            Some("libfx/journal_flush") => {
                let _ =
                    self.send(&json!({ "jsonrpc": "2.0", "id": message.get("id"), "result": {} }));
            }
            Some("libfx/tool_call") => {
                let inner = self.clone();
                tokio::spawn(async move {
                    let params = message.get("params").cloned().unwrap_or(Value::Null);
                    let reply = inner
                        .execute_host_tool(
                            params["name"].as_str().unwrap_or_default().to_string(),
                            params.get("input").cloned().unwrap_or(Value::Null),
                            params["toolCallId"]
                                .as_str()
                                .unwrap_or_default()
                                .to_string(),
                        )
                        .await;
                    if reply.cancelled || inner.closing.load(Ordering::SeqCst) {
                        return;
                    }
                    let _ = inner.send(&json!({
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "result": { "content": reply.content, "isError": reply.is_error, "contentType": "rich" },
                    }));
                });
            }
            Some(_) => {}
            None => {
                let Some(id) = message.get("id").and_then(Value::as_u64) else {
                    return;
                };
                let Some(waiter) = self.pending.lock().unwrap().remove(&id) else {
                    return;
                };
                let result = match message.get("error") {
                    Some(error) => Err(HarnessError::new(
                        error
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("fx request failed")
                            .to_string(),
                    )),
                    None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
                };
                let _ = waiter.send(result);
            }
        }
    }
}

/// `transportFetch`: retries a request once after a network error and adds
/// a subscription's credentials.
async fn transport_fetch(
    provider: FxEngineProvider,
    fetch: FxFetch,
    request: FxHttpRequest,
    signal: CancellationToken,
    turn_cancelled: Option<CancellationToken>,
) -> Result<FxHttpResponse, FxFetchError> {
    let endpoint = endpoint_of(&request.url).unwrap_or_else(|| request.url.clone());
    let mut last = FxFetchError::Failed("transport retry exhausted".into());
    for _attempt in 0..2 {
        if turn_cancelled
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(FxFetchError::Aborted);
        }
        let response = match &provider {
            FxEngineProvider::Subscription(authorizer) if authorizer.authorizes(&endpoint) => {
                authorizer
                    .fetch(&endpoint, request.clone(), signal.clone())
                    .await
            }
            _ => fetch(request.clone(), signal.clone()).await,
        };
        match response {
            Ok(response) => return Ok(response),
            Err(error) => {
                if signal.is_cancelled() || error == FxFetchError::Aborted {
                    return Err(FxFetchError::Aborted);
                }
                last = error;
            }
        }
    }
    Err(last)
}

/// The engine on fx's WebAssembly core.
pub struct WasmFxEngine {
    inner: Arc<Inner>,
    checkpoint_meta: Value,
}

impl WasmFxEngine {
    /// `createFxEngine(options)` on `module`.
    pub async fn create(
        module: &wasmtime::Module,
        options: FxEngineOptions,
    ) -> HarnessResult<Self> {
        validate(&options)?;
        let (core, mut items) =
            host::instantiate(module, &["acp"], &agent_environment(&options)).await?;
        let inner = Arc::new(Inner {
            core: core.clone(),
            pending: Mutex::default(),
            next_id: AtomicU64::new(1),
            session_id: Mutex::default(),
            turn: Mutex::default(),
            closing: AtomicBool::new(false),
            events: options.events.clone(),
            exit_error: Mutex::default(),
            next_attachment: Mutex::new(1),
        });
        // The dispatcher: everything the core asks, in order.
        let dispatcher = inner.clone();
        let provider = options.provider.clone();
        let fetch = options.fetch.clone();
        tokio::spawn(async move {
            // Ends once the core exits and its last items are handled.
            let exited = dispatcher.core.clone();
            let mut gone = false;
            loop {
                let item = if gone {
                    match items.try_recv() {
                        Ok(item) => item,
                        Err(_) => break,
                    }
                } else {
                    tokio::select! {
                        biased;
                        item = items.recv() => match item {
                            Some(item) => item,
                            None => break,
                        },
                        _ = exited.exited() => {
                            gone = true;
                            continue;
                        }
                    }
                };
                match item {
                    CoreItem::Line(message, _) => dispatcher.on_line(message),
                    CoreItem::HostToolCall {
                        name,
                        input,
                        call_id,
                        reply,
                    } => {
                        let inner = dispatcher.clone();
                        tokio::spawn(async move {
                            let _ = reply.send(inner.execute_host_tool(name, input, call_id).await);
                        });
                    }
                    CoreItem::Fetch {
                        request,
                        signal,
                        reply,
                    } => {
                        let future = transport_fetch(
                            provider.clone(),
                            fetch.clone(),
                            request,
                            signal,
                            dispatcher.current_turn_cancelled(),
                        );
                        tokio::spawn(async move {
                            let _ = reply.send(future.await);
                        });
                    }
                }
            }
        });
        // The core's exit fails every request still waiting.
        let watcher = inner.clone();
        tokio::spawn(async move {
            let code = watcher.core.exited().await;
            watcher.closing.store(true, Ordering::SeqCst);
            let error = watcher.core.error().unwrap_or_else(|| {
                format!("fx-core exited with code {code} before completing the ACP request")
            });
            *watcher.exit_error.lock().unwrap() = Some(error.clone());
            for (_, waiter) in watcher.pending.lock().unwrap().drain() {
                let _ = waiter.send(Err(HarnessError::new(error.clone())));
            }
        });
        let engine = Self {
            inner,
            checkpoint_meta: json!({
                "libfxVersion": FX_SDK_API_VERSION.to_string(),
                "toolSchemaHash": tool_schema_hash(&options.tools),
                "model": options.model.clone().unwrap_or_default(),
            }),
        };
        if let Err(error) = engine.start(&options).await {
            engine.inner.closing.store(true, Ordering::SeqCst);
            engine.inner.core.abort_host_effects();
            engine.inner.core.close_stdin();
            let _ = engine.inner.core.exited().await;
            return Err(error);
        }
        Ok(engine)
    }

    async fn start(&self, options: &FxEngineOptions) -> HarnessResult<()> {
        let inner = &self.inner;
        let mut capabilities = Map::new();
        if !options.tools.is_empty() || !options.instructions.is_empty() {
            capabilities.insert(
                "libfx".into(),
                json!({
                    "tools": options.tools.iter().map(descriptor_json).collect::<Vec<_>>(),
                    "instructions": options.instructions,
                }),
            );
        }
        inner
            .request(
                "initialize",
                json!({ "protocolVersion": 1, "clientCapabilities": capabilities }),
            )
            .await?;
        let mut params = Map::new();
        if options.hooks {
            params.insert("hooks".into(), json!({ "captureContent": false }));
        }
        let session = inner.request("libfx/new", Value::Object(params)).await?;
        let session_id = session
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| HarnessError::new("fx returned no session id"))?
            .to_string();
        *inner.session_id.lock().unwrap() = Some(session_id.clone());
        if options.hooks
            && session
                .pointer("/_meta/fx/hooksVersion")
                .and_then(Value::as_u64)
                != Some(1)
        {
            return Err(HarnessError::new(
                "Semantic hooks require a matching libfx core with hooks version 1",
            )
            .into());
        }
        if let Some(checkpoint) = &options.checkpoint {
            let ids = inner.attach(std::slice::from_ref(checkpoint))?;
            inner
                .request(
                    "libfx/restore",
                    json!({ "sessionId": session_id, "checkpointAttachment": ids[0] }),
                )
                .await?;
        }
        Ok(())
    }
}

struct WasmTurnControl {
    inner: Arc<Inner>,
    cancelled: CancellationToken,
    finished: Arc<AtomicBool>,
    next_steer: AtomicU64,
}

impl FxTurnControl for WasmTurnControl {
    fn steer(&self, text: &str) -> HarnessResult<()> {
        if self.finished.load(Ordering::SeqCst) || self.cancelled.is_cancelled() {
            return Err(HarnessError::new("no prompt is running").into());
        }
        if text.is_empty() {
            return Err(HarnessError::new("steering text cannot be empty").into());
        }
        let id = format!(
            "in_{}_{}",
            crate::harness::utils::random_uuid(),
            self.next_steer.fetch_add(1, Ordering::SeqCst)
        );
        self.inner.core.steer(text, &id)
    }

    /// `turn.cancel()`.
    fn cancel(&self) {
        if self.finished.load(Ordering::SeqCst) || self.cancelled.is_cancelled() {
            return;
        }
        self.cancelled.cancel();
        self.inner.core.close_steering();
        if !self.inner.closing.load(Ordering::SeqCst) {
            let _ = self.inner.send(&json!({
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": { "sessionId": self.inner.session_id() },
            }));
        }
        self.inner.core.abort_host_effects();
    }
}

/// The prompt as ACP content blocks, with each image's bytes attached.
fn prompt_frame(inner: &Inner, prompt: &[FxPromptBlock]) -> HarnessResult<Vec<Value>> {
    let images: Vec<Vec<u8>> = prompt
        .iter()
        .filter_map(|block| match block {
            FxPromptBlock::Image(image) => Some(
                base64::engine::general_purpose::STANDARD
                    .decode(&image.data)
                    .map_err(|_| {
                        HarnessError::new("image prompt block requires canonical base64 data")
                    }),
            ),
            FxPromptBlock::Text(_) => None,
        })
        .collect::<Result<_, _>>()?;
    if images.len() > MAX_PROMPT_IMAGES {
        return Err(HarnessError::new(format!(
            "prompt cannot contain more than {MAX_PROMPT_IMAGES} images"
        ))
        .into());
    }
    let ids = if images.is_empty() {
        Vec::new()
    } else {
        inner.attach(&images)?
    };
    let mut next = ids.into_iter();
    Ok(prompt
        .iter()
        .map(|block| match block {
            FxPromptBlock::Text(text) => json!({ "type": "text", "text": text }),
            FxPromptBlock::Image(image) => json!({
                "type": "image",
                "mimeType": image.mime_type,
                "_meta": { "fx": { "attachment": next.next() } },
            }),
        })
        .collect())
}

#[async_trait]
impl FxEngine for WasmFxEngine {
    fn prompt(&self, prompt: Vec<FxPromptBlock>) -> HarnessResult<FxTurn> {
        let inner = self.inner.clone();
        if inner.closing.load(Ordering::SeqCst) {
            return Err(HarnessError::new("fx agent is closed").into());
        }
        let cancelled = CancellationToken::new();
        {
            let mut turn = inner.turn.lock().unwrap();
            if turn.is_some() {
                return Err(
                    HarnessError::new("a prompt is already in progress for this session").into(),
                );
            }
            *turn = Some(ActiveTurn {
                cancelled: cancelled.clone(),
                events: TurnEvents::new(),
            });
        }
        let frame = match prompt_frame(&inner, &prompt) {
            Ok(frame) => frame,
            Err(error) => {
                *inner.turn.lock().unwrap() = None;
                return Err(error);
            }
        };
        inner.core.open_steering();
        let finished = Arc::new(AtomicBool::new(false));
        let control = Arc::new(WasmTurnControl {
            inner: inner.clone(),
            cancelled: cancelled.clone(),
            finished: finished.clone(),
            next_steer: AtomicU64::new(0),
        });
        let session_id = inner.session_id();
        let result = Box::pin(async move {
            let response = inner
                .request(
                    "session/prompt",
                    json!({ "sessionId": session_id, "prompt": frame }),
                )
                .await;
            finished.store(true, Ordering::SeqCst);
            *inner.turn.lock().unwrap() = None;
            inner.core.close_steering();
            match response {
                Ok(value) => Ok(FxTurnResult {
                    stop_reason: if cancelled.is_cancelled() {
                        "cancelled".into()
                    } else {
                        value
                            .get("stopReason")
                            .and_then(Value::as_str)
                            .unwrap_or("end_turn")
                            .to_string()
                    },
                    usage: usage_of(value.get("usage")),
                }),
                Err(error) if cancelled.is_cancelled() && error.to_string() == "Cancelled" => {
                    Ok(FxTurnResult {
                        stop_reason: "cancelled".into(),
                        usage: FxModelUsage::default(),
                    })
                }
                Err(error) => Err(error),
            }
        });
        Ok(FxTurn { result, control })
    }

    async fn checkpoint(&self) -> HarnessResult<Vec<u8>> {
        let inner = &self.inner;
        if inner.closing.load(Ordering::SeqCst) {
            return Err(HarnessError::new("fx agent is closed").into());
        }
        if inner.turn.lock().unwrap().is_some() {
            return Err(HarnessError::new("cannot checkpoint while a prompt is active").into());
        }
        let response = inner
            .request(
                "libfx/checkpoint",
                json!({ "sessionId": inner.session_id(), "checkpointMeta": self.checkpoint_meta }),
            )
            .await?;
        let bytes = response
            .get("checkpointAttachment")
            .and_then(Value::as_i64)
            .filter(|id| *id > 0)
            .and_then(|id| inner.core.take_attachment(id));
        match bytes {
            Some(bytes) if !bytes.is_empty() => Ok(bytes),
            _ => Err(HarnessError::new("fx returned an invalid checkpoint").into()),
        }
    }

    async fn close(&self) {
        let inner = &self.inner;
        if let Some(turn) = inner.turn.lock().unwrap().as_ref() {
            turn.cancelled.cancel();
        }
        if !inner.closing.swap(true, Ordering::SeqCst) {
            inner.core.abort_host_effects();
            inner.core.close_stdin();
        }
        inner.core.exited().await;
    }
}

/// The default runtime factory: the WebAssembly core from `module`.
pub fn wasm_runtime_factory(module: wasmtime::Module) -> FxRuntimeFactory {
    Arc::new(move |options| {
        let module = module.clone();
        Box::pin(async move {
            let engine: Arc<dyn FxEngine> = Arc::new(WasmFxEngine::create(&module, options).await?);
            Ok(engine)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_session_updates_to_turn_events() {
        let mut events = TurnEvents::new();
        assert_eq!(
            events.event_for(&json!({ "sessionUpdate": "agent_message_chunk", "content": { "text": "[context] x" } })),
            None
        );
        assert_eq!(
            events.event_for(
                &json!({ "sessionUpdate": "agent_message_chunk", "content": { "text": "hi" } })
            ),
            Some(FxRuntimeEvent::TextDelta("hi".into()))
        );
        assert_eq!(
            events.event_for(&json!({ "sessionUpdate": "tool_call", "toolCallId": "c1", "title": "read", "rawInput": { "a": 1 } })),
            Some(FxRuntimeEvent::ToolStart {
                id: "c1".into(),
                name: "read".into(),
                input: Some(json!({ "a": 1 })),
            })
        );
        assert_eq!(
            events.event_for(&json!({ "sessionUpdate": "tool_call", "toolCallId": "c1" })),
            None
        );
        assert_eq!(
            events.event_for(&json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": "c1",
                "status": "failed",
                "content": [{ "content": { "type": "text", "text": "boom" } }],
            })),
            Some(FxRuntimeEvent::ToolEnd {
                id: "c1".into(),
                // A later `tool_call` without a name renames it, as in libfx.
                name: "tool".into(),
                content: Some("boom".into()),
                is_error: true,
            })
        );
        assert_eq!(
            events.event_for(&json!({ "sessionUpdate": "user_message_chunk", "content": { "type": "text", "text": "more" } })),
            Some(FxRuntimeEvent::UserMessage("more".into()))
        );
    }

    #[test]
    fn hashes_the_sorted_tool_set() {
        let tool = |name: &str| FxToolDescriptor {
            name: name.into(),
            description: "d".into(),
            input_schema: json!({ "type": "object" }),
        };
        assert_eq!(
            tool_schema_hash(&[tool("b"), tool("a")]),
            tool_schema_hash(&[tool("a"), tool("b")])
        );
        assert_eq!(tool_schema_hash(&[]), "741638a5");
    }
}
