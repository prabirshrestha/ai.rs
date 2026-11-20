//! Port of `harness-fx/src/fx-session.ts` and `fx-translate.ts`: one fx
//! session. Each turn starts a fresh engine on the session's checkpoint,
//! runs the prompt, and saves the new checkpoint.
//!
//! [rust] ai.rs turns run to the end or are cancelled; they are never
//! suspended and continued later, so the TS replay records (`state.turn`,
//! emitted keys, recorded tool calls and steers) are not kept. Approvals and
//! host tools are answered inline through `before_tool_call` and the turn's
//! tools.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use base64::Engine as _;
use indexmap::IndexMap;
use serde_json::{Value, json};
use tokio::sync::{OnceCell, mpsc, watch};
use tokio_util::sync::CancellationToken;

use super::libfx::authorizer::{FxCredentialSource, FxProviderCredentials, SubscriptionAuthorizer};
use super::libfx::engine::{
    FxEngine, FxEngineEvent, FxEngineOptions, FxEngineProvider, FxImage, FxModelEnd, FxPromptBlock,
    FxRuntimeEvent, FxRuntimeFactory, FxToolDescriptor, FxToolResult, FxTurn, FxTurnControl,
    wasm_runtime_factory,
};
use super::libfx::fetch::{FxFetch, reqwest_fetch};
use super::libfx::wasm::load_core;
use super::mcp::{FxMcpRuntime, create_fx_mcp_runtime};
use super::mcp_client::default_mcp_connector;
use super::remote_ops::FxRemoteOps;
use super::resume_state::{FxResumeStateData, FxSessionState};
use super::session_storage::FxSessionStorageWriter;
use super::skills::create_fx_skills_instructions;
use super::subscription::{FxResolvedAuthentication, FxSubscription, resolve_fx_model};
use super::transport::{
    FxProvider, FxToolCall, FxTransport, FxTransportOptions, FxTransportSignal,
};
use super::{FX_API, FX_CLIENT_APP, FxHarnessSettings, FxServiceTier};
use crate::harness::adapters::host_tools::{ApprovalDecision, HostTools, tool_result_text};
use crate::harness::adapters::turn_builder::TurnBuilder;
use crate::harness::error::{BoxError, HarnessError, HarnessResult};
use crate::harness::paths::resolve_posix;
use crate::harness::sandbox::SandboxSession;
use crate::harness::session::{
    HarnessResumeState, HarnessSession, HarnessTurn, HarnessTurnOptions, HarnessTurnSender,
};
use crate::harness::tools::{BuiltinTool, BuiltinToolFilter, BuiltinToolKind, PermissionMode};
use crate::harness::utils::is_sandbox_write_protected_path;
use crate::{
    AgentMessage, AgentResult, AgentToolResult, AssistantMessage, ImageContent, Message,
    StopReason, ToolCall, ToolResultContent, Usage, UserContent, UserMessage, UserMessageContent,
};

/// Workspace paths that hold fx configuration or skills, which can run code
/// or change the agent's behavior in later sessions. Writes to them always
/// require approval unless the permission mode is allow-all.
const FX_WRITE_PROTECTED_PATHS: [&str; 2] = [".agents/skills", ".fx"];

/// What a session's turns share.
pub(crate) struct SessionConfig {
    pub settings: Arc<FxHarnessSettings>,
    pub sandbox: Arc<dyn SandboxSession>,
    pub remote: FxRemoteOps,
    pub work_dir: String,
    pub permission_mode: PermissionMode,
    pub builtin_tool_filter: Option<BuiltinToolFilter>,
    pub headers: Option<IndexMap<String, String>>,
    pub authentication: FxResolvedAuthentication,
    pub builtins: IndexMap<String, BuiltinTool>,
}

/// An fx session.
pub struct FxSession {
    pub(crate) session_id: String,
    pub(crate) is_resume: bool,
    pub(crate) config: Arc<SessionConfig>,
    pub(crate) state: Arc<StdMutex<FxSessionState>>,
    pub(crate) writer: Option<FxSessionStorageWriter>,
    pub(crate) running: Arc<tokio::sync::Mutex<()>>,
    pub(crate) cancellation_token: CancellationToken,
    pub(crate) stopped: AtomicBool,
    pub(crate) factory: Arc<OnceCell<FxRuntimeFactory>>,
}

fn harness_error(message: impl Into<String>) -> crate::AgentError {
    crate::harness::error::into_agent_error(HarnessError::new(message).into())
}

/// `toPrompt(prompt)`.
fn prompt_blocks(message: &AgentMessage) -> Vec<FxPromptBlock> {
    match message {
        Message::User(user) => match &user.content {
            UserMessageContent::Text(text) => vec![FxPromptBlock::Text(text.clone())],
            UserMessageContent::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    UserContent::Text(text) => FxPromptBlock::Text(text.text.clone()),
                    UserContent::Image(image) => FxPromptBlock::Image(FxImage {
                        data: image.data.clone(),
                        mime_type: image.mime_type.clone(),
                    }),
                })
                .collect(),
        },
        other => vec![FxPromptBlock::Text(
            serde_json::to_string(other).unwrap_or_default(),
        )],
    }
}

/// The text of a steering message.
fn steer_text(message: &AgentMessage) -> Option<String> {
    let Message::User(user) = message else {
        return None;
    };
    let text = match &user.content {
        UserMessageContent::Text(text) => text.clone(),
        UserMessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                UserContent::Text(text) => Some(text.text.as_str()),
                UserContent::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    (!text.is_empty()).then_some(text)
}

impl FxSession {
    fn resume_data_sync(&self) -> FxSessionState {
        self.state.lock().unwrap().clone()
    }

    /// `createLifecycleData()`.
    async fn resume_state(&self) -> AgentResult<HarnessResumeState> {
        let state = self.resume_data_sync();
        let data = match &self.writer {
            Some(writer) => FxResumeStateData::Leaf(
                writer
                    .write(&state)
                    .await
                    .map_err(crate::harness::error::into_agent_error)?,
            ),
            None => FxResumeStateData::Session(state),
        };
        Ok(HarnessResumeState {
            harness_id: "fx".into(),
            data: data.to_value(),
        })
    }

    /// `shutdown()`: cancels the running turn and waits for it.
    async fn shut_down(&self) {
        if self.stopped.swap(true, Ordering::SeqCst) {
            return;
        }
        self.cancellation_token.cancel();
        let _ = self.running.lock().await;
    }

    async fn runtime_factory(&self) -> HarnessResult<FxRuntimeFactory> {
        let settings = self.config.settings.clone();
        self.factory
            .get_or_try_init(|| async move {
                if let Some(factory) = &settings.runtime_factory {
                    return Ok(factory.clone());
                }
                let wasm = load_core(&settings.core).await?;
                let module =
                    tokio::task::spawn_blocking(move || super::libfx::host::compile(&wasm))
                        .await
                        .map_err(|error| {
                            HarnessError::with_cause("fx core compilation failed", error)
                        })??;
                Ok(wasm_runtime_factory(module))
            })
            .await
            .cloned()
    }
}

#[async_trait]
impl HarnessSession for FxSession {
    fn harness_id(&self) -> &str {
        "fx"
    }

    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn is_resume(&self) -> bool {
        self.is_resume
    }

    async fn prompt(
        &self,
        message: AgentMessage,
        options: HarnessTurnOptions,
    ) -> AgentResult<HarnessTurn> {
        if self.stopped.load(Ordering::SeqCst) {
            return Err(harness_error("fx session has been stopped."));
        }
        if self.state.lock().unwrap().turn.is_some() {
            return Err(harness_error(
                "Continue the unfinished fx turn before prompting again. [rust] ai.rs cannot continue a turn suspended by another runtime.",
            ));
        }
        let running = self
            .running
            .clone()
            .try_lock_owned()
            .map_err(|_| harness_error("A turn is already running in this session."))?;
        let factory = self
            .runtime_factory()
            .await
            .map_err(crate::harness::error::into_agent_error)?;
        let token = self.cancellation_token.child_token();
        if let Some(caller) = options.cancellation_token.clone() {
            let token = token.clone();
            tokio::spawn(async move {
                tokio::select! {
                    _ = caller.cancelled() => token.cancel(),
                    _ = token.cancelled() => {}
                }
            });
        }
        let (control, controls) = watch::channel(None::<Arc<dyn FxTurnControl>>);
        let context = TurnContext {
            config: self.config.clone(),
            state: self.state.clone(),
            factory,
            host: HostTools {
                tools: options.settings.tools.clone(),
                before_tool_call: options.before_tool_call.clone(),
                after_tool_call: options.after_tool_call.clone(),
                cancellation_token: Some(token.clone()),
            },
            options,
            token: token.clone(),
            control,
        };
        Ok(HarnessTurn::spawn(move |sender| async move {
            let _running = running;
            let messages = run_turn(context, message, sender).await;
            token.cancel();
            messages
        })
        .with_steer(move |message| {
            let mut controls = controls.clone();
            async move {
                let text = steer_text(&message)
                    .ok_or_else(|| harness_error("fx can only steer with text."))?;
                loop {
                    let control = controls.borrow_and_update().clone();
                    if let Some(control) = control {
                        return control
                            .steer(&text)
                            .map_err(crate::harness::error::into_agent_error);
                    }
                    if controls.changed().await.is_err() {
                        return Err(harness_error("The fx turn is not running."));
                    }
                }
            }
        }))
    }

    async fn detach(&self) -> AgentResult<HarnessResumeState> {
        self.shut_down().await;
        self.resume_state().await
    }

    async fn stop(&self) -> AgentResult<HarnessResumeState> {
        self.shut_down().await;
        self.resume_state().await
    }

    async fn destroy(&self) -> AgentResult<()> {
        self.shut_down().await;
        Ok(())
    }
}

struct TurnContext {
    config: Arc<SessionConfig>,
    state: Arc<StdMutex<FxSessionState>>,
    factory: FxRuntimeFactory,
    host: HostTools,
    options: HarnessTurnOptions,
    token: CancellationToken,
    control: watch::Sender<Option<Arc<dyn FxTurnControl>>>,
}

/// How a tool offered to fx runs.
#[derive(Clone)]
enum Registered {
    /// A sandbox operation (`providerExecuted`).
    Builtin(BuiltinToolKind),
    /// The caller's tool, `askUserQuestions` included.
    Host,
    /// An MCP server's tool, by its index in the runtime.
    Mcp(usize),
}

/// The subscription's credentials, remembering the first failure so the
/// turn reports it instead of the model request it broke.
struct RecordingSource {
    subscription: FxSubscription,
    failure: Arc<StdMutex<Option<String>>>,
}

impl FxCredentialSource for RecordingSource {
    fn read(&self) -> futures::future::BoxFuture<'_, Result<FxProviderCredentials, String>> {
        Box::pin(async move {
            let result = self
                .subscription
                .read()
                .await
                .map_err(|error| error.to_string());
            if let Err(error) = &result {
                self.failure.lock().unwrap().get_or_insert(error.clone());
            }
            result
        })
    }

    fn write(
        &self,
        credentials: FxProviderCredentials,
    ) -> futures::future::BoxFuture<'_, Result<(), String>> {
        Box::pin(async move {
            let result = self
                .subscription
                .write(&credentials)
                .await
                .map_err(|error| error.to_string());
            if let Err(error) = &result {
                self.failure.lock().unwrap().get_or_insert(error.clone());
            }
            result
        })
    }
}

fn empty_assistant(provider: &str, model: &str) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: FX_API.to_string(),
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

/// `onModelEnd`: a step's usage, with `noCache` and `text` derived.
pub(crate) fn step_usage(end: &FxModelEnd) -> Usage {
    let usage = end.usage.clone().unwrap_or_default();
    let clamp = |value: u64| u32::try_from(value).unwrap_or(u32::MAX);
    let cache_read = usage.cache_read_tokens.unwrap_or(0);
    let cache_write = usage.cache_write_tokens.unwrap_or(0);
    let input = usage
        .input_tokens
        .map_or(0, |total| total.saturating_sub(cache_read + cache_write));
    let output = usage.output_tokens.unwrap_or(0);
    Usage {
        input: clamp(input),
        output: clamp(output),
        cache_read: clamp(cache_read),
        cache_write: clamp(cache_write),
        reasoning: usage.reasoning_tokens.map(clamp),
        total_tokens: clamp(input + cache_read + cache_write + output),
        ..Usage::default()
    }
}

/// A finish reason as a stop reason.
pub(crate) fn stop_reason_of(finish_reason: Option<&str>) -> StopReason {
    match finish_reason {
        Some("length") => StopReason::Length,
        Some("tool-calls") => StopReason::ToolUse,
        Some("error") => StopReason::Error,
        _ => StopReason::Stop,
    }
}

/// An `FxToolResult` as the caller sees it.
fn agent_tool_result(result: &FxToolResult) -> AgentToolResult {
    let mut agent = AgentToolResult::text(result.text.clone());
    if result.text.is_empty() {
        agent.content.clear();
    }
    agent.content.extend(result.images.iter().map(|image| {
        ToolResultContent::Image(ImageContent {
            data: image.data.clone(),
            mime_type: image.mime_type.clone(),
        })
    }));
    agent
}

/// A host tool's result for fx.
fn fx_tool_result(result: &AgentToolResult, is_error: bool) -> FxToolResult {
    FxToolResult {
        text: tool_result_text(result),
        images: result
            .content
            .iter()
            .filter_map(|part| match part {
                ToolResultContent::Image(image) => Some(FxImage {
                    data: image.data.clone(),
                    mime_type: image.mime_type.clone(),
                }),
                ToolResultContent::Text(_) => None,
            })
            .collect(),
        is_error,
    }
}

/// A sandbox operation's output as text.
fn output_text(output: &Value) -> String {
    match output {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// `fxToolRequiresApproval`.
pub(crate) fn fx_tool_requires_approval(
    permission_mode: PermissionMode,
    kind: BuiltinToolKind,
) -> bool {
    permission_mode != PermissionMode::AllowAll
        && (kind == BuiltinToolKind::Bash
            || (kind == BuiltinToolKind::Edit && permission_mode == PermissionMode::AllowReads))
}

/// The tool call ids the transport parsed, and their inputs.
fn registered_calls(
    calls: Vec<FxToolCall>,
    registered: &HashMap<String, Registered>,
) -> Vec<(String, String, Value)> {
    calls
        .into_iter()
        .filter(|call| registered.contains_key(&call.tool_name))
        .filter_map(|call| {
            let input = match call.input {
                Value::String(text) => serde_json::from_str(&text).ok()?,
                value => value,
            };
            input
                .is_object()
                .then_some((call.tool_call_id, call.tool_name, input))
        })
        .collect()
}

struct Turn<'a> {
    context: &'a TurnContext,
    builder: TurnBuilder,
    registered: HashMap<String, Registered>,
    mcp: FxMcpRuntime,
    provider: FxProvider,
    /// The current model request.
    request: Option<usize>,
    /// Request index → (finish reason, usage).
    steps: HashMap<usize, (Option<String>, Usage)>,
    rejected: HashMap<usize, (Option<String>, u16)>,
    /// Tool calls added to the assistant message.
    calls: HashSet<String>,
}

impl Turn<'_> {
    fn close_step(&mut self, index: usize) {
        let (finish, usage) = self.steps.get(&index).cloned().unzip();
        self.builder.close_message(
            finish
                .flatten()
                .as_deref()
                .map(|finish| stop_reason_of(Some(finish))),
            usage,
            None,
        );
    }

    fn assistant(&self) -> AssistantMessage {
        self.builder
            .current_assistant()
            .or_else(|| {
                self.builder
                    .messages()
                    .iter()
                    .rev()
                    .find_map(|message| match message {
                        Message::Assistant(message) => Some(message.clone()),
                        _ => None,
                    })
            })
            .unwrap_or_else(|| empty_assistant(self.provider.as_str(), self.builder.model()))
    }

    /// Handles one event; `Err` fails the turn.
    async fn on_event(&mut self, event: FxEngineEvent) -> HarnessResult<()> {
        match event {
            FxEngineEvent::Transport(FxTransportSignal::Request { index, model_id }) => {
                if let Some(previous) = self.request
                    && index > previous
                {
                    self.close_step(previous);
                }
                self.request = Some(index);
                if let Some(model_id) = model_id {
                    self.builder.set_model(model_id);
                }
            }
            FxEngineEvent::Transport(FxTransportSignal::ToolCalls { calls, .. }) => {
                for (id, name, input) in registered_calls(calls, &self.registered) {
                    if self.calls.insert(id.clone()) {
                        self.builder
                            .tool_call(&format!("tool:{id}"), &id, &name, input);
                    }
                }
            }
            FxEngineEvent::Transport(FxTransportSignal::Rejected {
                index,
                model_id,
                status,
            }) => {
                self.rejected.insert(index, (model_id, status));
            }
            FxEngineEvent::Turn(
                FxRuntimeEvent::TextDelta(delta) | FxRuntimeEvent::ReasoningDelta(delta),
            ) if self.request.is_none() => {
                let _ = delta;
                return Err(self.unmatched_output());
            }
            FxEngineEvent::Turn(FxRuntimeEvent::TextDelta(delta)) => {
                let key = format!("{}:text", self.request.unwrap());
                self.builder.text_delta(&key, &delta);
            }
            FxEngineEvent::Turn(FxRuntimeEvent::ReasoningDelta(delta)) => {
                let key = format!("{}:reasoning", self.request.unwrap());
                self.builder.thinking_delta(&key, &delta);
            }
            FxEngineEvent::Turn(FxRuntimeEvent::UserMessage(text)) => {
                self.builder
                    .user_message(Message::User(UserMessage::text(text)));
            }
            FxEngineEvent::Turn(
                FxRuntimeEvent::ToolStart { .. } | FxRuntimeEvent::ToolEnd { .. },
            ) => {}
            FxEngineEvent::ModelEnd(end) => {
                if let Some(index) = self.request
                    && (end.usage.is_some() || end.finish_reason.is_some())
                {
                    let usage = step_usage(&end);
                    self.builder.set_usage(usage.clone());
                    self.steps.insert(index, (end.finish_reason.clone(), usage));
                }
            }
            FxEngineEvent::ToolCall {
                name,
                input,
                call_id,
                signal,
                reply,
            } => {
                let id = if call_id.is_empty() {
                    crate::harness::utils::random_uuid()
                } else {
                    call_id
                };
                if self.calls.insert(id.clone()) {
                    self.builder
                        .tool_call(&format!("tool:{id}"), &id, &name, input.clone());
                }
                let result = self.run_tool(&id, &name, input, signal).await;
                let _ = reply.send(result);
            }
        }
        Ok(())
    }

    fn unmatched_output(&self) -> BoxError {
        let next = self.request.map_or(0, |index| index + 1);
        match self.rejected.get(&next) {
            Some((model_id, status)) => HarnessError::new(format!(
                "fx {} model request {} failed with HTTP {status}.",
                self.provider.as_str(),
                model_id.as_deref().unwrap_or("(unknown model)")
            ))
            .into(),
            None => {
                HarnessError::new("fx parsed output could not be matched to its model request.")
                    .into()
            }
        }
    }

    /// Runs one tool call and reports it to the caller.
    async fn run_tool(
        &mut self,
        id: &str,
        name: &str,
        input: Value,
        signal: CancellationToken,
    ) -> FxToolResult {
        let call = ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: input.clone(),
            thought_signature: None,
        };
        let assistant = self.assistant();
        self.builder.tool_start(id, name, input.clone());
        let registered = self.registered.get(name).cloned();
        let result = match registered {
            None => FxToolResult::error(format!("Unsupported fx tool {name}")),
            Some(Registered::Host) => {
                let (result, is_error) = self.context.host.run(&call, &assistant, None).await;
                let fx = fx_tool_result(&result, is_error);
                self.builder.tool_end(id, name, result, is_error);
                return fx;
            }
            Some(Registered::Builtin(kind)) => {
                match self.approve(&call, &assistant, kind, &signal).await {
                    Err(reason) => FxToolResult::error(reason),
                    Ok(args) => match self
                        .context
                        .config
                        .remote
                        .call(name, args, Some(signal))
                        .await
                    {
                        Ok(output) => FxToolResult::text(output_text(&output)),
                        Err(error) => FxToolResult::error(error.to_string()),
                    },
                }
            }
            Some(Registered::Mcp(index)) => {
                match self
                    .approve(&call, &assistant, BuiltinToolKind::Bash, &signal)
                    .await
                {
                    Err(reason) => FxToolResult::error(reason),
                    Ok(args) => self.mcp.tools[index].execute(args, signal).await,
                }
            }
        };
        self.builder
            .tool_end(id, name, agent_tool_result(&result), result.is_error);
        result
    }

    /// The arguments to run with, or why the call was denied.
    async fn approve(
        &self,
        call: &ToolCall,
        assistant: &AssistantMessage,
        kind: BuiltinToolKind,
        signal: &CancellationToken,
    ) -> Result<Value, String> {
        let config = &self.context.config;
        let mode = config.permission_mode;
        let required = fx_tool_requires_approval(mode, kind)
            || (mode == PermissionMode::AllowEdits
                && kind == BuiltinToolKind::Edit
                && self.write_protected(&call.arguments, signal).await);
        if !required {
            return Ok(call.arguments.clone());
        }
        let default = ApprovalDecision::Deny {
            reason: format!(
                "Tool '{}' requires approval, but no approval handler was provided.",
                call.name
            ),
        };
        match self.context.host.approve(call, assistant, default).await {
            ApprovalDecision::Allow { args } => Ok(args),
            ApprovalDecision::Deny { reason } if reason.is_empty() => {
                Err("Tool execution denied.".into())
            }
            ApprovalDecision::Deny { reason } => Err(reason),
        }
    }

    /// `isFxWriteProtected({ toolInput })`.
    async fn write_protected(&self, input: &Value, signal: &CancellationToken) -> bool {
        let Some(file_path) = input.get("file_path").and_then(Value::as_str) else {
            return true;
        };
        let config = &self.context.config;
        let protected: Vec<String> = FX_WRITE_PROTECTED_PATHS
            .iter()
            .map(|path| path.to_string())
            .collect();
        is_sandbox_write_protected_path(
            config.sandbox.as_ref(),
            &config.work_dir,
            &[resolve_posix(&config.work_dir, file_path)],
            &protected,
            Some(signal.clone()),
        )
        .await
        .unwrap_or(true)
    }
}

/// The tools fx is offered, and how each runs.
fn turn_tools(
    config: &SessionConfig,
    host: &HostTools,
    mcp: &FxMcpRuntime,
) -> (Vec<FxToolDescriptor>, HashMap<String, Registered>) {
    let mut tools: Vec<FxToolDescriptor> = Vec::new();
    let mut registered = HashMap::new();
    let custom: HashSet<String> = host
        .tools
        .iter()
        .map(|tool| tool.definition().name)
        .collect();
    for (name, builtin) in &config.builtins {
        if custom.contains(name)
            || !BuiltinToolFilter::includes(config.builtin_tool_filter.as_ref(), name)
        {
            continue;
        }
        tools.push(FxToolDescriptor {
            name: name.clone(),
            description: if builtin.tool.description.is_empty() {
                name.clone()
            } else {
                builtin.tool.description.clone()
            },
            input_schema: builtin.tool.parameters.clone(),
        });
        registered.insert(
            name.clone(),
            if name == "askUserQuestions" {
                Registered::Host
            } else {
                Registered::Builtin(builtin.kind.unwrap_or(BuiltinToolKind::Readonly))
            },
        );
    }
    for tool in &host.tools {
        let definition = tool.definition();
        let schema = if definition.parameters.is_object() {
            definition.parameters.clone()
        } else {
            json!({ "type": "object", "properties": {} })
        };
        tools.push(FxToolDescriptor {
            name: definition.name.clone(),
            description: if definition.description.is_empty() {
                definition.name.clone()
            } else {
                definition.description.clone()
            },
            input_schema: schema,
        });
        registered.insert(definition.name, Registered::Host);
    }
    for (index, tool) in mcp.tools.iter().enumerate() {
        let mut name = tool.descriptor.name.clone();
        let mut suffix = 2;
        while tools.iter().any(|existing| existing.name == name) {
            name = format!(
                "{}_{suffix}",
                tool.descriptor.name.chars().take(58).collect::<String>()
            );
            suffix += 1;
        }
        tools.push(FxToolDescriptor {
            name: name.clone(),
            ..tool.descriptor.clone()
        });
        registered.insert(name, Registered::Mcp(index));
    }
    (tools, registered)
}

/// The finish of a turn that ran to its end, or why it failed.
type Outcome = Result<(), BoxError>;

async fn run_turn(
    context: TurnContext,
    message: AgentMessage,
    sender: HarnessTurnSender,
) -> AgentResult<Vec<AgentMessage>> {
    let config = context.config.clone();
    let provider = config.authentication.provider();
    let settings = context.options.settings.clone();
    let mut builder = TurnBuilder::new(
        sender,
        FX_API,
        provider.as_str(),
        settings.model.clone().unwrap_or_default(),
    );
    builder.start(message.clone());
    let credential_failure = Arc::new(StdMutex::new(None::<String>));

    let setup = async {
        let skills = create_fx_skills_instructions(&settings.skills)?;
        let connector = config
            .settings
            .mcp_connector
            .clone()
            .unwrap_or_else(default_mcp_connector);
        let mcp = create_fx_mcp_runtime(
            config.settings.mcp_servers.as_ref(),
            Some(context.token.clone()),
            &connector,
        )
        .await?;
        Ok::<_, BoxError>((skills, mcp))
    }
    .await;
    let (skills, mcp) = match setup {
        Ok(setup) => setup,
        Err(error) => return Ok(builder.finish(Some((StopReason::Error, error.to_string())))),
    };
    let (tools, registered) = turn_tools(&config, &context.host, &mcp);
    let (events, mut receiver) = mpsc::unbounded_channel::<FxEngineEvent>();
    let mut headers = config.settings.headers.clone().unwrap_or_default();
    headers.extend(config.headers.clone().unwrap_or_default());
    headers.insert("user-agent".into(), FX_CLIENT_APP.to_string());
    headers.insert("x-client-app".into(), FX_CLIENT_APP.to_string());
    let signals = events.clone();
    let fetch: FxFetch = config.settings.fetch.clone().unwrap_or_else(reqwest_fetch);
    let transport = FxTransport::new(FxTransportOptions {
        base_url: match &config.authentication {
            FxResolvedAuthentication::Gateway { base_url, .. } => Some(base_url.clone()),
            FxResolvedAuthentication::Subscription(_) => None,
        },
        provider,
        headers,
        response_format: context.options.response_format.clone(),
        signals: Arc::new(move |signal| {
            let _ = signals.send(FxEngineEvent::Transport(signal));
        }),
        fetch,
    });

    let mut turn = Turn {
        context: &context,
        builder,
        registered,
        mcp,
        provider,
        request: None,
        steps: HashMap::new(),
        rejected: HashMap::new(),
        calls: HashSet::new(),
    };
    let mut engine: Option<Arc<dyn FxEngine>> = None;
    let outcome: Outcome = async {
        let model = resolve_fx_model(
            provider,
            settings.model.clone(),
            config.settings.home_directory.as_deref(),
        )
        .await;
        let engine_provider = match &config.authentication {
            FxResolvedAuthentication::Gateway { api_key, .. } => FxEngineProvider::Gateway {
                api_key: api_key.clone(),
            },
            FxResolvedAuthentication::Subscription(subscription) => {
                FxEngineProvider::Subscription(Arc::new(SubscriptionAuthorizer::new(
                    subscription.provider,
                    Arc::new(RecordingSource {
                        subscription: subscription.clone(),
                        failure: credential_failure.clone(),
                    }),
                    transport.fetch(),
                )))
            }
        };
        let instructions = [
            settings.instructions.clone().unwrap_or_default(),
            skills,
            turn.mcp.instructions.clone(),
            format!(
                "All filesystem and shell tools operate in the sandbox workspace {}.",
                config.work_dir
            ),
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
        let checkpoint = context.state.lock().unwrap().checkpoint.clone();
        let checkpoint = checkpoint
            .map(|checkpoint| {
                base64::engine::general_purpose::STANDARD
                    .decode(checkpoint)
                    .map_err(|error| {
                        HarnessError::with_cause("The fx checkpoint is not valid base64.", error)
                    })
            })
            .transpose()?;
        let (fast, ultrafast) = match config.settings.service_tier {
            None | Some(FxServiceTier::Default) => (None, None),
            Some(tier) => (
                Some(tier == FxServiceTier::Fast),
                Some(tier == FxServiceTier::Ultrafast),
            ),
        };
        if context.token.is_cancelled() {
            return Err(HarnessError::new("This operation was aborted").into());
        }
        let created = (context.factory)(FxEngineOptions {
            provider: engine_provider,
            model,
            effort: config.settings.effort.clone(),
            fast,
            ultrafast,
            instructions,
            tools,
            checkpoint: checkpoint.clone(),
            fetch: transport.fetch(),
            hooks: true,
            events: events.clone(),
        })
        .await?;
        engine = Some(created.clone());
        if checkpoint.is_none() {
            let initial = created.checkpoint().await?;
            context.state.lock().unwrap().checkpoint =
                Some(base64::engine::general_purpose::STANDARD.encode(initial));
        }
        let FxTurn { result, control } = created.prompt(prompt_blocks(&message))?;
        let _ = context.control.send(Some(control.clone()));
        let mut result = result;
        let mut cancelled = false;
        let finished = loop {
            tokio::select! {
                biased;
                _ = context.token.cancelled(), if !cancelled => {
                    cancelled = true;
                    control.cancel();
                }
                event = receiver.recv() => {
                    if let Some(event) = event
                        && let Err(error) = turn.on_event(event).await
                    {
                        control.cancel();
                        let _ = (&mut result).await;
                        return Err(error);
                    }
                }
                finished = &mut result => break finished,
            }
        };
        let _ = context.control.send(None);
        while let Ok(event) = receiver.try_recv() {
            turn.on_event(event).await?;
        }
        if cancelled || context.token.is_cancelled() {
            return Err(HarnessError::new("This operation was aborted").into());
        }
        let finished = finished?;
        let last = turn.request;
        let step = last.and_then(|index| turn.steps.get(&index).cloned());
        let (finish, usage) = match step {
            Some((finish, usage)) => (
                finish.map(|finish| stop_reason_of(Some(&finish))),
                Some(usage),
            ),
            None => (
                Some(if finished.stop_reason == "max_output_tokens" {
                    StopReason::Length
                } else {
                    StopReason::Stop
                }),
                None,
            ),
        };
        if turn.builder.has_open_message() {
            turn.builder.close_message(finish, usage, None);
        }
        let saved = created.checkpoint().await?;
        context.state.lock().unwrap().checkpoint =
            Some(base64::engine::general_purpose::STANDARD.encode(saved));
        Ok(())
    }
    .await;

    drop(events);
    if let Some(engine) = &engine {
        engine.close().await;
    }
    transport.stop().await;
    turn.mcp.close().await;
    let error = match outcome {
        Ok(()) => None,
        Err(_) if context.token.is_cancelled() => {
            Some((StopReason::Aborted, "Operation aborted".to_string()))
        }
        Err(error) => {
            let failure = credential_failure.lock().unwrap().clone();
            Some((
                StopReason::Error,
                failure.unwrap_or_else(|| error.to_string()),
            ))
        }
    };
    Ok(turn.builder.finish(error))
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
