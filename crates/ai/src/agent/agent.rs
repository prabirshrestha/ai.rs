//! Port of `packages/agent/src/agent.ts`.
//!
//! Divergences from Pi:
//! - Pi exposes `state` and the hooks as mutable fields. Rust exposes a state
//!   snapshot ([`Agent::state`]) plus setters, and getter/setter pairs for
//!   every hook so hosts can chain the previous hook. [`Agent`] is a cheap
//!   clonable handle.
//! - Pi resolves `getDefaultStreamFn()` in the constructor and throws when
//!   none is configured. Rust resolves it in the constructor when one is
//!   configured and otherwise at run time, where a missing default fails the
//!   run like any other thrown error.
//! - `prompt(input)` overloads become [`Agent::prompt_text`] and
//!   [`Agent::prompt_messages`]; `continue()` is [`Agent::continue_run`].

use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::agent::agent_loop::{run_agent_loop, run_agent_loop_continue};
use crate::agent::error::{
    AgentError, AgentResult, CONTINUE_WHILE_PROCESSING, PROMPT_WHILE_PROCESSING,
    RESET_WHILE_PROCESSING,
};
use crate::agent::stream_fn::get_default_stream_fn;
use crate::agent::types::{
    AfterToolCallFn, AgentContext, AgentEvent, AgentEventListener, AgentEventSink, AgentLoopConfig,
    AgentLoopTurnUpdate, AgentMessage, AgentState, BeforeToolCallFn, ConvertToLlmFn, DynAgentTool,
    FinishTurnFn, GetApiKeyFn, PrepareNextTurnContext, PrepareNextTurnFn, PrepareRequestFn,
    QueueMode, StreamFn, ToolExecutionMode, TransformContextFn, default_convert_to_llm,
    user_message,
};
use crate::types::{
    AssistantContent, AssistantMessage, BoxFuture, ImageContent, Message, Model,
    ModelThinkingLevel, PayloadHook, ProviderStreamEventHook, ResponseHook, SimpleStreamOptions,
    StopReason, TextContent, ThinkingBudgets, Transport, Usage,
};
use crate::utils::time::now_millis;
use crate::utils::transcript::{
    create_initial_system_message, get_current_system_message, get_current_system_prompt,
    to_tool_declaration,
};

/// `AgentOptions.prepareNextTurn`: the legacy signal-only callback.
pub type AgentPrepareNextTurnFn =
    Arc<dyn Fn(Option<CancellationToken>) -> BoxFuture<Option<AgentLoopTurnUpdate>> + Send + Sync>;

/// `AgentOptions.prepareNextTurnWithContext`.
pub type AgentPrepareNextTurnWithContextFn = Arc<
    dyn Fn(
            PrepareNextTurnContext,
            Option<CancellationToken>,
        ) -> BoxFuture<Option<AgentLoopTurnUpdate>>
        + Send
        + Sync,
>;

/// Handle returned by [`Agent::subscribe`]. Dropping it unsubscribes.
#[must_use = "keep the subscription alive while the listener remains registered"]
pub struct AgentSubscription {
    listeners: Arc<Mutex<Vec<AgentEventListener>>>,
    listener: Option<AgentEventListener>,
}

impl AgentSubscription {
    /// Remove the listener. Returns whether it was still registered.
    pub fn unsubscribe(mut self) -> bool {
        self.remove()
    }

    fn remove(&mut self) -> bool {
        let Some(listener) = self.listener.take() else {
            return false;
        };
        let mut listeners = self.listeners.lock();
        match listeners
            .iter()
            .position(|candidate| Arc::ptr_eq(candidate, &listener))
        {
            Some(position) => {
                listeners.remove(position);
                true
            }
            None => false,
        }
    }
}

impl Drop for AgentSubscription {
    fn drop(&mut self) {
        self.remove();
    }
}

/// Pi's `DEFAULT_MODEL`.
fn default_model() -> Model {
    Model {
        id: "unknown".to_string(),
        name: "unknown".to_string(),
        api: "unknown".to_string(),
        provider: "unknown".to_string(),
        ..Default::default()
    }
}

/// Initial state for [`Agent`]. `system_prompt` and `tools` become the
/// leading system message unless `messages` already starts with one.
#[derive(Clone, Default)]
pub struct AgentInitialState {
    pub system_prompt: Option<String>,
    pub model: Option<Model>,
    pub thinking_level: Option<ModelThinkingLevel>,
    pub tools: Vec<DynAgentTool>,
    pub messages: Vec<AgentMessage>,
}

struct MutableAgentState {
    model: Model,
    thinking_level: ModelThinkingLevel,
    tools: Vec<DynAgentTool>,
    messages: Vec<AgentMessage>,
    is_streaming: bool,
    streaming_message: Option<AgentMessage>,
    pending_tool_calls: HashSet<String>,
    error_message: Option<String>,
}

fn create_mutable_agent_state(initial_state: AgentInitialState) -> MutableAgentState {
    let tools = initial_state.tools;
    let mut messages = initial_state.messages;
    let declarations: Vec<_> = tools
        .iter()
        .map(|tool| to_tool_declaration(&tool.definition()))
        .collect();
    let initial_message =
        create_initial_system_message(initial_state.system_prompt.as_deref(), Some(&declarations));
    if !matches!(messages.first(), Some(Message::System(_)))
        && let Some(initial_message) = initial_message
    {
        messages.insert(0, Message::System(initial_message));
    }

    MutableAgentState {
        model: initial_state.model.unwrap_or_else(default_model),
        thinking_level: initial_state
            .thinking_level
            .unwrap_or(ModelThinkingLevel::Off),
        tools,
        messages,
        is_streaming: false,
        streaming_message: None,
        pending_tool_calls: HashSet::new(),
        error_message: None,
    }
}

impl MutableAgentState {
    fn snapshot(&self) -> AgentState {
        AgentState {
            system_prompt: get_current_system_prompt(&self.messages),
            model: self.model.clone(),
            thinking_level: self.thinking_level,
            tools: self.tools.clone(),
            messages: self.messages.clone(),
            is_streaming: self.is_streaming,
            streaming_message: self.streaming_message.clone(),
            pending_tool_calls: self.pending_tool_calls.clone(),
            error_message: self.error_message.clone(),
        }
    }
}

/// Options for constructing an [`Agent`].
#[derive(Clone)]
pub struct AgentOptions {
    pub initial_state: AgentInitialState,
    pub convert_to_llm: Option<ConvertToLlmFn>,
    pub transform_context: Option<TransformContextFn>,
    /// Stream function; [`set_default_stream_fn`](crate::agent::set_default_stream_fn)
    /// supplies the fallback.
    pub stream_fn: Option<StreamFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub on_payload: Option<PayloadHook>,
    pub on_response: Option<ResponseHook>,
    pub on_provider_stream_event: Option<ProviderStreamEventHook>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub finish_turn: Option<FinishTurnFn>,
    pub prepare_request: Option<PrepareRequestFn>,
    pub prepare_next_turn: Option<AgentPrepareNextTurnFn>,
    pub prepare_next_turn_with_context: Option<AgentPrepareNextTurnWithContextFn>,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Transport,
    pub max_retry_delay_ms: Option<u64>,
    pub tool_execution: ToolExecutionMode,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            initial_state: AgentInitialState::default(),
            convert_to_llm: None,
            transform_context: None,
            stream_fn: None,
            get_api_key: None,
            on_payload: None,
            on_response: None,
            on_provider_stream_event: None,
            before_tool_call: None,
            after_tool_call: None,
            finish_turn: None,
            prepare_request: None,
            prepare_next_turn: None,
            prepare_next_turn_with_context: None,
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            session_id: None,
            thinking_budgets: None,
            transport: Transport::Auto,
            max_retry_delay_ms: None,
            tool_execution: ToolExecutionMode::Parallel,
        }
    }
}

impl AgentOptions {
    /// Options whose initial state uses `model`.
    pub fn new(model: Model) -> Self {
        let mut options = Self::default();
        options.initial_state.model = Some(model);
        options
    }

    pub fn builder(model: Model) -> AgentOptionsBuilder {
        AgentOptionsBuilder {
            options: Self::new(model),
        }
    }
}

/// Rust convenience builder for [`AgentOptions`].
pub struct AgentOptionsBuilder {
    options: AgentOptions,
}

macro_rules! option_setters {
    ($($name:ident: $type:ty),* $(,)?) => {
        $(
            pub fn $name(mut self, $name: $type) -> Self {
                self.options.$name = Some($name);
                self
            }
        )*
    };
}

impl AgentOptionsBuilder {
    pub fn initial_state(mut self, initial_state: AgentInitialState) -> Self {
        self.options.initial_state = initial_state;
        self
    }

    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.options.initial_state.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn thinking_level(mut self, thinking_level: ModelThinkingLevel) -> Self {
        self.options.initial_state.thinking_level = Some(thinking_level);
        self
    }

    pub fn tool(mut self, tool: DynAgentTool) -> Self {
        self.options.initial_state.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = DynAgentTool>) -> Self {
        self.options.initial_state.tools.extend(tools);
        self
    }

    pub fn message(mut self, message: impl Into<AgentMessage>) -> Self {
        self.options.initial_state.messages.push(message.into());
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = AgentMessage>) -> Self {
        self.options.initial_state.messages.extend(messages);
        self
    }

    option_setters! {
        convert_to_llm: ConvertToLlmFn,
        transform_context: TransformContextFn,
        stream_fn: StreamFn,
        get_api_key: GetApiKeyFn,
        on_payload: PayloadHook,
        on_response: ResponseHook,
        on_provider_stream_event: ProviderStreamEventHook,
        before_tool_call: BeforeToolCallFn,
        after_tool_call: AfterToolCallFn,
        finish_turn: FinishTurnFn,
        prepare_request: PrepareRequestFn,
        prepare_next_turn: AgentPrepareNextTurnFn,
        prepare_next_turn_with_context: AgentPrepareNextTurnWithContextFn,
        thinking_budgets: ThinkingBudgets,
        max_retry_delay_ms: u64,
    }

    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.options.session_id = Some(session_id.into());
        self
    }

    pub fn steering_mode(mut self, steering_mode: QueueMode) -> Self {
        self.options.steering_mode = steering_mode;
        self
    }

    pub fn follow_up_mode(mut self, follow_up_mode: QueueMode) -> Self {
        self.options.follow_up_mode = follow_up_mode;
        self
    }

    pub fn transport(mut self, transport: Transport) -> Self {
        self.options.transport = transport;
        self
    }

    pub fn tool_execution(mut self, tool_execution: ToolExecutionMode) -> Self {
        self.options.tool_execution = tool_execution;
        self
    }

    pub fn build(self) -> AgentOptions {
        self.options
    }
}

struct PendingMessageQueue {
    messages: Vec<AgentMessage>,
    mode: QueueMode,
}

impl PendingMessageQueue {
    fn new(mode: QueueMode) -> Self {
        Self {
            messages: Vec::new(),
            mode,
        }
    }

    fn enqueue(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }

    fn has_items(&self) -> bool {
        !self.messages.is_empty()
    }

    fn peek(&self) -> Vec<AgentMessage> {
        match self.mode {
            QueueMode::All => self.messages.clone(),
            QueueMode::OneAtATime => self.messages.first().cloned().into_iter().collect(),
        }
    }

    fn drain(&mut self) -> Vec<AgentMessage> {
        let drained = self.peek();
        self.messages.drain(..drained.len());
        drained
    }

    fn clear(&mut self) {
        self.messages.clear();
    }
}

/// The public mutable hook fields of Pi's `Agent`.
#[derive(Clone)]
struct AgentHooks {
    convert_to_llm: ConvertToLlmFn,
    transform_context: Option<TransformContextFn>,
    stream_function: Option<StreamFn>,
    get_api_key: Option<GetApiKeyFn>,
    on_payload: Option<PayloadHook>,
    on_response: Option<ResponseHook>,
    on_provider_stream_event: Option<ProviderStreamEventHook>,
    before_tool_call: Option<BeforeToolCallFn>,
    after_tool_call: Option<AfterToolCallFn>,
    finish_turn: Option<FinishTurnFn>,
    prepare_request: Option<PrepareRequestFn>,
    prepare_next_turn: Option<AgentPrepareNextTurnFn>,
    prepare_next_turn_with_context: Option<AgentPrepareNextTurnWithContextFn>,
    session_id: Option<String>,
    thinking_budgets: Option<ThinkingBudgets>,
    transport: Transport,
    max_retry_delay_ms: Option<u64>,
    tool_execution: ToolExecutionMode,
}

struct ActiveRun {
    abort_controller: CancellationToken,
}

struct AgentInner {
    state: Mutex<MutableAgentState>,
    listeners: Arc<Mutex<Vec<AgentEventListener>>>,
    steering_queue: Mutex<PendingMessageQueue>,
    follow_up_queue: Mutex<PendingMessageQueue>,
    hooks: Mutex<AgentHooks>,
    active_run: Mutex<Option<ActiveRun>>,
    idle: Notify,
}

/// Stateful wrapper around the low-level agent loop.
///
/// `Agent` owns the current transcript, emits lifecycle events, executes
/// tools, and exposes queueing APIs for steering and follow-up messages.
/// Clones share the same agent.
#[derive(Clone)]
pub struct Agent {
    inner: Arc<AgentInner>,
}

macro_rules! hook_accessors {
    ($($(#[$doc:meta])* $field:ident, $setter:ident: $type:ty;)*) => {
        $(
            $(#[$doc])*
            pub fn $field(&self) -> $type {
                self.inner.hooks.lock().$field.clone()
            }

            $(#[$doc])*
            pub fn $setter(&self, $field: $type) {
                self.inner.hooks.lock().$field = $field;
            }
        )*
    };
}

impl Agent {
    pub fn new(options: AgentOptions) -> Self {
        let stream_function = options.stream_fn.or_else(|| get_default_stream_fn().ok());
        Self {
            inner: Arc::new(AgentInner {
                state: Mutex::new(create_mutable_agent_state(options.initial_state)),
                listeners: Arc::new(Mutex::new(Vec::new())),
                steering_queue: Mutex::new(PendingMessageQueue::new(options.steering_mode)),
                follow_up_queue: Mutex::new(PendingMessageQueue::new(options.follow_up_mode)),
                hooks: Mutex::new(AgentHooks {
                    convert_to_llm: options
                        .convert_to_llm
                        .unwrap_or_else(default_convert_to_llm),
                    transform_context: options.transform_context,
                    stream_function,
                    get_api_key: options.get_api_key,
                    on_payload: options.on_payload,
                    on_response: options.on_response,
                    on_provider_stream_event: options.on_provider_stream_event,
                    before_tool_call: options.before_tool_call,
                    after_tool_call: options.after_tool_call,
                    finish_turn: options.finish_turn,
                    prepare_request: options.prepare_request,
                    prepare_next_turn: options.prepare_next_turn,
                    prepare_next_turn_with_context: options.prepare_next_turn_with_context,
                    session_id: options.session_id,
                    thinking_budgets: options.thinking_budgets,
                    transport: options.transport,
                    max_retry_delay_ms: options.max_retry_delay_ms,
                    tool_execution: options.tool_execution,
                }),
                active_run: Mutex::new(None),
                idle: Notify::new(),
            }),
        }
    }

    /// Subscribe to agent lifecycle events.
    ///
    /// Listener futures are awaited in subscription order and are included
    /// in the current run's settlement. Listeners also receive the active
    /// abort signal for the current run. An `Err` fails the run.
    pub fn subscribe<F, Fut>(&self, listener: F) -> AgentSubscription
    where
        F: Fn(AgentEvent, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = AgentResult<()>> + Send + 'static,
    {
        let listener: AgentEventListener =
            Arc::new(move |event, signal| Box::pin(listener(event, signal)));
        let listeners = Arc::clone(&self.inner.listeners);
        listeners.lock().push(Arc::clone(&listener));
        AgentSubscription {
            listeners,
            listener: Some(listener),
        }
    }

    /// Current agent state (a snapshot).
    pub fn state(&self) -> AgentState {
        self.inner.state.lock().snapshot()
    }

    /// Current system prompt, replayed from the transcript. To change it,
    /// append a system message with `content` or `sections`.
    pub fn system_prompt(&self) -> String {
        get_current_system_prompt(&self.inner.state.lock().messages)
    }

    /// Conversation transcript.
    pub fn messages(&self) -> Vec<AgentMessage> {
        self.inner.state.lock().messages.clone()
    }

    pub fn set_model(&self, model: Model) {
        self.inner.state.lock().model = model;
    }

    pub fn set_thinking_level(&self, thinking_level: ModelThinkingLevel) {
        self.inner.state.lock().thinking_level = thinking_level;
    }

    /// Replace the executable tools. Differences from the tools declared in
    /// the transcript are announced to the model with a system message
    /// before the next request.
    pub fn set_tools(&self, tools: Vec<DynAgentTool>) {
        self.inner.state.lock().tools = tools;
    }

    /// Replace the transcript.
    pub fn set_messages(&self, messages: Vec<AgentMessage>) {
        self.inner.state.lock().messages = messages;
    }

    /// Append to the transcript (Pi: `state.messages.push(message)`).
    pub fn push_message(&self, message: impl Into<AgentMessage>) {
        self.inner.state.lock().messages.push(message.into());
    }

    hook_accessors! {
        convert_to_llm, set_convert_to_llm: ConvertToLlmFn;
        transform_context, set_transform_context: Option<TransformContextFn>;
        /// `streamFunction`.
        stream_function, set_stream_function: Option<StreamFn>;
        get_api_key, set_get_api_key: Option<GetApiKeyFn>;
        on_payload, set_on_payload: Option<PayloadHook>;
        on_response, set_on_response: Option<ResponseHook>;
        on_provider_stream_event, set_on_provider_stream_event: Option<ProviderStreamEventHook>;
        before_tool_call, set_before_tool_call: Option<BeforeToolCallFn>;
        after_tool_call, set_after_tool_call: Option<AfterToolCallFn>;
        finish_turn, set_finish_turn: Option<FinishTurnFn>;
        prepare_request, set_prepare_request: Option<PrepareRequestFn>;
        prepare_next_turn, set_prepare_next_turn: Option<AgentPrepareNextTurnFn>;
        prepare_next_turn_with_context, set_prepare_next_turn_with_context: Option<AgentPrepareNextTurnWithContextFn>;
        /// Session identifier forwarded to providers for cache-aware backends.
        session_id, set_session_id: Option<String>;
        /// Optional per-level thinking token budgets forwarded to the stream function.
        thinking_budgets, set_thinking_budgets: Option<ThinkingBudgets>;
        /// Preferred transport forwarded to the stream function.
        transport, set_transport: Transport;
        /// Optional cap for provider-requested retry delays.
        max_retry_delay_ms, set_max_retry_delay_ms: Option<u64>;
        /// Tool execution strategy for assistant messages with multiple tool calls.
        tool_execution, set_tool_execution: ToolExecutionMode;
    }

    /// Controls how queued steering messages are drained.
    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.inner.steering_queue.lock().mode = mode;
    }

    pub fn steering_mode(&self) -> QueueMode {
        self.inner.steering_queue.lock().mode
    }

    /// Controls how queued follow-up messages are drained.
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.inner.follow_up_queue.lock().mode = mode;
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        self.inner.follow_up_queue.lock().mode
    }

    /// Queue a message to be injected after the current assistant turn finishes.
    pub fn steer(&self, message: impl Into<AgentMessage>) {
        self.inner.steering_queue.lock().enqueue(message.into());
    }

    /// Queue a message to run only after the agent would otherwise stop.
    pub fn follow_up(&self, message: impl Into<AgentMessage>) {
        self.inner.follow_up_queue.lock().enqueue(message.into());
    }

    /// Remove all queued steering messages.
    pub fn clear_steering_queue(&self) {
        self.inner.steering_queue.lock().clear();
    }

    /// Remove all queued follow-up messages.
    pub fn clear_follow_up_queue(&self) {
        self.inner.follow_up_queue.lock().clear();
    }

    /// Remove all queued steering and follow-up messages.
    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// Returns true when either queue still contains pending messages.
    pub fn has_queued_messages(&self) -> bool {
        self.inner.steering_queue.lock().has_items()
            || self.inner.follow_up_queue.lock().has_items()
    }

    /// Preview the messages selected for the next turn without consuming them.
    pub fn peek_queued_messages(&self) -> Vec<AgentMessage> {
        let steering = self.inner.steering_queue.lock().peek();
        if !steering.is_empty() {
            return steering;
        }
        self.inner.follow_up_queue.lock().peek()
    }

    /// Active abort signal for the current run, if any.
    pub fn signal(&self) -> Option<CancellationToken> {
        self.inner
            .active_run
            .lock()
            .as_ref()
            .map(|run| run.abort_controller.clone())
    }

    /// Abort the current run, if one is active.
    pub fn abort(&self) {
        if let Some(run) = self.inner.active_run.lock().as_ref() {
            run.abort_controller.cancel();
        }
    }

    /// Resolve when the current run and all awaited event listeners have
    /// finished.
    pub async fn wait_for_idle(&self) {
        loop {
            let notified = self.inner.idle.notified();
            if self.inner.active_run.lock().is_none() {
                return;
            }
            notified.await;
        }
    }

    /// Clear conversation state and queues while retaining the replayed
    /// prompt/tool baseline.
    pub fn reset(&self) -> AgentResult<()> {
        if self.inner.active_run.lock().is_some() {
            return Err(AgentError::AlreadyProcessing(RESET_WHILE_PROCESSING));
        }

        {
            let mut state = self.inner.state.lock();
            let baseline = get_current_system_message(&state.messages);
            state.messages = baseline.map(Message::System).into_iter().collect();
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls = HashSet::new();
            state.error_message = None;
        }
        self.clear_follow_up_queue();
        self.clear_steering_queue();
        Ok(())
    }

    /// Start a new prompt from text and optional images.
    pub async fn prompt_text(
        &self,
        input: impl Into<String>,
        images: Vec<ImageContent>,
    ) -> AgentResult<()> {
        self.prompt_messages(vec![user_message(input, images)])
            .await
    }

    /// Start a new prompt from a single message.
    pub async fn prompt_message(&self, message: impl Into<AgentMessage>) -> AgentResult<()> {
        self.prompt_messages(vec![message.into()]).await
    }

    /// Start a new prompt from a batch of messages.
    pub async fn prompt_messages(&self, messages: Vec<AgentMessage>) -> AgentResult<()> {
        let signal = self.start_run(PROMPT_WHILE_PROCESSING)?;
        self.run_prompt_messages(messages, false, signal).await
    }

    /// Continue from the current transcript. The last message must be a user
    /// or tool-result message.
    pub async fn continue_run(&self) -> AgentResult<()> {
        let signal = self.start_run(CONTINUE_WHILE_PROCESSING)?;

        let last_message = {
            let state = self.inner.state.lock();
            if state
                .messages
                .iter()
                .all(|message| matches!(message, Message::System(_)))
            {
                None
            } else {
                state.messages.last().cloned()
            }
        };
        let Some(last_message) = last_message else {
            self.finish_run();
            return Err(AgentError::NoMessagesToContinue);
        };

        if matches!(last_message, Message::Assistant(_)) {
            let queued_steering = self.inner.steering_queue.lock().drain();
            if !queued_steering.is_empty() {
                return self
                    .run_prompt_messages(queued_steering, true, signal)
                    .await;
            }

            let queued_follow_ups = self.inner.follow_up_queue.lock().drain();
            if !queued_follow_ups.is_empty() {
                return self
                    .run_prompt_messages(queued_follow_ups, false, signal)
                    .await;
            }

            self.finish_run();
            return Err(AgentError::CannotContinueFromAssistant);
        }

        self.run_continuation(signal).await
    }

    async fn run_prompt_messages(
        &self,
        messages: Vec<AgentMessage>,
        skip_initial_steering_poll: bool,
        signal: CancellationToken,
    ) -> AgentResult<()> {
        let context = self.create_context_snapshot();
        let config = self.create_loop_config(skip_initial_steering_poll, &signal);
        let stream_function = self.stream_function();
        let emit = self.event_sink();
        let loop_signal = signal.clone();
        self.run_with_lifecycle(signal, async move {
            run_agent_loop(
                messages,
                context,
                config,
                emit,
                Some(loop_signal),
                stream_function,
            )
            .await
            .map(|_| ())
        })
        .await
    }

    async fn run_continuation(&self, signal: CancellationToken) -> AgentResult<()> {
        let context = self.create_context_snapshot();
        let config = self.create_loop_config(false, &signal);
        let stream_function = self.stream_function();
        let emit = self.event_sink();
        let loop_signal = signal.clone();
        self.run_with_lifecycle(signal, async move {
            run_agent_loop_continue(context, config, emit, Some(loop_signal), stream_function)
                .await
                .map(|_| ())
        })
        .await
    }

    fn create_context_snapshot(&self) -> AgentContext {
        let state = self.inner.state.lock();
        AgentContext {
            messages: state.messages.clone(),
            tools: state.tools.clone(),
        }
    }

    fn create_loop_config(
        &self,
        skip_initial_steering_poll: bool,
        signal: &CancellationToken,
    ) -> AgentLoopConfig {
        let (model, thinking_level) = {
            let state = self.inner.state.lock();
            (state.model.clone(), state.thinking_level)
        };
        let hooks = self.inner.hooks.lock().clone();

        let mut options = SimpleStreamOptions {
            reasoning: thinking_level.thinking_level(),
            thinking_budgets: hooks.thinking_budgets,
            ..Default::default()
        };
        options.session_id = hooks.session_id;
        options.on_payload = hooks.on_payload;
        options.on_response = hooks.on_response;
        options.on_provider_stream_event = hooks.on_provider_stream_event;
        options.transport = Some(hooks.transport);
        options.max_retry_delay_ms = hooks.max_retry_delay_ms;

        let prepare_next_turn: Option<PrepareNextTurnFn> = if hooks
            .prepare_next_turn_with_context
            .is_some()
            || hooks.prepare_next_turn.is_some()
        {
            let with_context = hooks.prepare_next_turn_with_context;
            let legacy = hooks.prepare_next_turn;
            let signal = signal.clone();
            Some(Arc::new(move |context| {
                let with_context = with_context.clone();
                let legacy = legacy.clone();
                let signal = Some(signal.clone());
                Box::pin(async move {
                    if let Some(with_context) = with_context {
                        return with_context(context, signal).await;
                    }
                    match legacy {
                        Some(legacy) => legacy(signal).await,
                        None => None,
                    }
                })
            }))
        } else {
            None
        };

        let inner = Arc::clone(&self.inner);
        let skip_initial_steering_poll = Arc::new(Mutex::new(skip_initial_steering_poll));
        let get_steering_messages = Arc::new(move || {
            let messages = {
                let mut skip = skip_initial_steering_poll.lock();
                if *skip {
                    *skip = false;
                    Vec::new()
                } else {
                    inner.steering_queue.lock().drain()
                }
            };
            Box::pin(async move { messages }) as BoxFuture<Vec<AgentMessage>>
        });
        let inner = Arc::clone(&self.inner);
        let get_follow_up_messages = Arc::new(move || {
            let messages = inner.follow_up_queue.lock().drain();
            Box::pin(async move { messages }) as BoxFuture<Vec<AgentMessage>>
        });

        AgentLoopConfig {
            model,
            options,
            convert_to_llm: hooks.convert_to_llm,
            transform_context: hooks.transform_context,
            get_api_key: hooks.get_api_key,
            finish_turn: hooks.finish_turn,
            prepare_request: hooks.prepare_request,
            prepare_next_turn,
            get_steering_messages: Some(get_steering_messages),
            get_follow_up_messages: Some(get_follow_up_messages),
            tool_execution: hooks.tool_execution,
            before_tool_call: hooks.before_tool_call,
            after_tool_call: hooks.after_tool_call,
        }
    }

    /// Claim the active run (`runWithLifecycle`'s guard), failing with
    /// `message` when one is active.
    fn start_run(&self, message: &'static str) -> AgentResult<CancellationToken> {
        let mut active_run = self.inner.active_run.lock();
        if active_run.is_some() {
            return Err(AgentError::AlreadyProcessing(message));
        }
        let abort_controller = CancellationToken::new();
        *active_run = Some(ActiveRun {
            abort_controller: abort_controller.clone(),
        });
        Ok(abort_controller)
    }

    async fn run_with_lifecycle(
        &self,
        signal: CancellationToken,
        executor: impl Future<Output = AgentResult<()>>,
    ) -> AgentResult<()> {
        {
            let mut state = self.inner.state.lock();
            state.is_streaming = true;
            state.streaming_message = None;
            state.error_message = None;
        }

        let result = match executor.await {
            Ok(()) => Ok(()),
            Err(error) => self.handle_run_failure(error, signal.is_cancelled()).await,
        };
        self.finish_run();
        result
    }

    async fn handle_run_failure(&self, error: AgentError, aborted: bool) -> AgentResult<()> {
        let model = self.inner.state.lock().model.clone();
        let failure_message = Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::Text(TextContent::new(""))],
            usage: Usage::default(),
            stop_reason: if aborted {
                StopReason::Aborted
            } else {
                StopReason::Error
            },
            error_message: Some(error.to_string()),
            timestamp: now_millis(),
            ..AssistantMessage::empty_for(&model)
        });
        let emit = self.event_sink();
        emit(AgentEvent::MessageStart {
            message: failure_message.clone(),
        })
        .await?;
        emit(AgentEvent::MessageEnd {
            message: failure_message.clone(),
        })
        .await?;
        emit(AgentEvent::TurnEnd {
            message: failure_message.clone(),
            tool_results: Vec::new(),
        })
        .await?;
        emit(AgentEvent::AgentEnd {
            messages: vec![failure_message],
        })
        .await
    }

    fn finish_run(&self) {
        {
            let mut state = self.inner.state.lock();
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls = HashSet::new();
        }
        *self.inner.active_run.lock() = None;
        self.inner.idle.notify_waiters();
    }

    /// Reduce internal state for a loop event, then await listeners
    /// (`processEvents`).
    fn event_sink(&self) -> AgentEventSink {
        let inner = Arc::clone(&self.inner);
        Arc::new(move |event| {
            let inner = Arc::clone(&inner);
            Box::pin(async move {
                {
                    let mut state = inner.state.lock();
                    match &event {
                        AgentEvent::MessageStart { message }
                        | AgentEvent::MessageUpdate { message, .. } => {
                            state.streaming_message = Some(message.clone());
                        }
                        AgentEvent::MessageEnd { message } => {
                            state.streaming_message = None;
                            state.messages.push(message.clone());
                        }
                        AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                            state.pending_tool_calls.insert(tool_call_id.clone());
                        }
                        AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                            state.pending_tool_calls.remove(tool_call_id);
                        }
                        AgentEvent::TurnEnd { message, .. } => {
                            if let Message::Assistant(assistant) = message
                                && let Some(error_message) = &assistant.error_message
                            {
                                state.error_message = Some(error_message.clone());
                            }
                        }
                        AgentEvent::AgentEnd { .. } => {
                            state.streaming_message = None;
                        }
                        _ => {}
                    }
                }

                let signal = inner
                    .active_run
                    .lock()
                    .as_ref()
                    .map(|run| run.abort_controller.clone())
                    .ok_or(AgentError::ListenerOutsideRun)?;
                let listeners = inner.listeners.lock().clone();
                for listener in listeners {
                    listener(event.clone(), signal.clone()).await?;
                }
                Ok(())
            })
        })
    }
}

impl Default for Agent {
    fn default() -> Self {
        Self::new(AgentOptions::default())
    }
}

#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;
