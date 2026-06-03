use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;

use crate::{
    AssistantContent, AssistantMessage, ImageContent, Message, Model, SimpleStreamOptions,
    StopReason, TextContent, ThinkingBudgets, Transport, Usage,
};
use parking_lot::Mutex as SyncMutex;
use tokio::sync::{Mutex, Notify};
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{run_agent_loop, run_agent_loop_continue};
use crate::agent_types::{
    AfterToolCallFn, AgentContext, AgentEvent, AgentEventListener, AgentEventSink, AgentLoopConfig,
    AgentLoopTurnUpdate, AgentMessage, BeforeToolCallFn, ConvertToLlmFn, DynAgentTool,
    PrepareNextTurnContext, PrepareNextTurnFn, QueueMode, StreamFn, ToolExecutionMode,
    TransformContextFn, default_convert_to_llm, user_message,
};
use crate::{AgentError, AgentResult};

pub type AgentPrepareNextTurnFn = Arc<
    dyn Fn(
            Option<CancellationToken>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Option<AgentLoopTurnUpdate>> + Send>,
        > + Send
        + Sync,
>;

#[must_use = "keep the subscription alive while the listener should stay registered"]
pub struct AgentSubscription {
    listeners: Arc<SyncMutex<Vec<AgentEventListener>>>,
    listener: Option<AgentEventListener>,
}

impl AgentSubscription {
    pub fn unsubscribe(mut self) -> bool {
        self.remove()
    }

    fn remove(&mut self) -> bool {
        let Some(listener) = self.listener.take() else {
            return false;
        };
        let mut listeners = self.listeners.lock();
        if let Some(pos) = listeners
            .iter()
            .position(|candidate| Arc::ptr_eq(candidate, &listener))
        {
            listeners.remove(pos);
            true
        } else {
            false
        }
    }
}

impl Drop for AgentSubscription {
    fn drop(&mut self) {
        self.remove();
    }
}

fn default_agent_model() -> Model {
    Model {
        id: "unknown".to_string(),
        name: "unknown".to_string(),
        api: "unknown".to_string(),
        provider: "unknown".to_string(),
        ..Default::default()
    }
}

#[derive(Clone)]
pub struct AgentState {
    pub system_prompt: String,
    pub model: Model,
    pub thinking_level: crate::ModelThinkingLevel,
    pub tools: Vec<DynAgentTool>,
    pub messages: Vec<AgentMessage>,
    pub is_streaming: bool,
    pub streaming_message: Option<AgentMessage>,
    pub pending_tool_calls: HashSet<String>,
    pub error_message: Option<String>,
}

impl AgentState {
    pub fn new(model: Model) -> Self {
        Self {
            system_prompt: String::new(),
            model,
            thinking_level: crate::ModelThinkingLevel::Off,
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }

    pub fn builder(model: Model) -> AgentStateBuilder {
        AgentStateBuilder::new(model)
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new(default_agent_model())
    }
}

pub struct AgentStateBuilder {
    state: AgentState,
}

impl AgentStateBuilder {
    pub fn new(model: Model) -> Self {
        Self {
            state: AgentState::new(model),
        }
    }

    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.state.system_prompt = system_prompt.into();
        self
    }

    pub fn thinking_level(mut self, thinking_level: crate::ModelThinkingLevel) -> Self {
        self.state.thinking_level = thinking_level;
        self
    }

    pub fn tool(mut self, tool: DynAgentTool) -> Self {
        self.state.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = DynAgentTool>) -> Self {
        self.state.tools.extend(tools);
        self
    }

    pub fn message(mut self, message: AgentMessage) -> Self {
        self.state.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = AgentMessage>) -> Self {
        self.state.messages.extend(messages);
        self
    }

    pub fn build(self) -> AgentState {
        self.state
    }
}

#[derive(Clone)]
pub struct AgentOptions {
    pub initial_state: AgentState,
    pub convert_to_llm: Option<ConvertToLlmFn>,
    pub transform_context: Option<TransformContextFn>,
    pub stream_fn: Option<StreamFn>,
    pub prepare_next_turn: Option<AgentPrepareNextTurnFn>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub session_id: Option<String>,
    pub options: SimpleStreamOptions,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub tool_execution: ToolExecutionMode,
}

impl AgentOptions {
    pub fn new(model: Model) -> Self {
        Self {
            initial_state: AgentState::new(model),
            convert_to_llm: None,
            transform_context: None,
            stream_fn: None,
            prepare_next_turn: None,
            before_tool_call: None,
            after_tool_call: None,
            session_id: None,
            options: SimpleStreamOptions::default(),
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            tool_execution: ToolExecutionMode::Parallel,
        }
    }

    pub fn builder(model: Model) -> AgentOptionsBuilder {
        AgentOptionsBuilder::new(model)
    }
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self::new(default_agent_model())
    }
}

pub struct AgentOptionsBuilder {
    options: AgentOptions,
}

impl AgentOptionsBuilder {
    pub fn new(model: Model) -> Self {
        Self {
            options: AgentOptions::new(model),
        }
    }

    pub fn initial_state(mut self, initial_state: AgentState) -> Self {
        self.options.initial_state = initial_state;
        self
    }

    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.options.initial_state.system_prompt = system_prompt.into();
        self
    }

    pub fn thinking_level(mut self, thinking_level: crate::ModelThinkingLevel) -> Self {
        self.options.initial_state.thinking_level = thinking_level;
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

    pub fn message(mut self, message: AgentMessage) -> Self {
        self.options.initial_state.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = AgentMessage>) -> Self {
        self.options.initial_state.messages.extend(messages);
        self
    }

    pub fn convert_to_llm(mut self, convert_to_llm: ConvertToLlmFn) -> Self {
        self.options.convert_to_llm = Some(convert_to_llm);
        self
    }

    pub fn transform_context(mut self, transform_context: TransformContextFn) -> Self {
        self.options.transform_context = Some(transform_context);
        self
    }

    pub fn stream_fn(mut self, stream_fn: StreamFn) -> Self {
        self.options.stream_fn = Some(stream_fn);
        self
    }

    pub fn prepare_next_turn(mut self, prepare_next_turn: AgentPrepareNextTurnFn) -> Self {
        self.options.prepare_next_turn = Some(prepare_next_turn);
        self
    }

    pub fn before_tool_call(mut self, before_tool_call: BeforeToolCallFn) -> Self {
        self.options.before_tool_call = Some(before_tool_call);
        self
    }

    pub fn after_tool_call(mut self, after_tool_call: AfterToolCallFn) -> Self {
        self.options.after_tool_call = Some(after_tool_call);
        self
    }

    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.options.session_id = Some(session_id.into());
        self
    }

    pub fn options(mut self, options: SimpleStreamOptions) -> Self {
        self.options.options = options;
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

    fn drain(&mut self) -> Vec<AgentMessage> {
        match self.mode {
            QueueMode::All => std::mem::take(&mut self.messages),
            QueueMode::OneAtATime => {
                if self.messages.is_empty() {
                    Vec::new()
                } else {
                    vec![self.messages.remove(0)]
                }
            }
        }
    }

    fn clear(&mut self) {
        self.messages.clear();
    }

    fn has_items(&self) -> bool {
        !self.messages.is_empty()
    }
}

pub struct Agent {
    state: Arc<Mutex<AgentState>>,
    listeners: Arc<SyncMutex<Vec<AgentEventListener>>>,
    steering_queue: Arc<Mutex<PendingMessageQueue>>,
    follow_up_queue: Arc<Mutex<PendingMessageQueue>>,
    convert_to_llm: Option<ConvertToLlmFn>,
    transform_context: Option<TransformContextFn>,
    stream_fn: Option<StreamFn>,
    prepare_next_turn: Option<AgentPrepareNextTurnFn>,
    before_tool_call: Option<BeforeToolCallFn>,
    after_tool_call: Option<AfterToolCallFn>,
    session_id: Arc<Mutex<Option<String>>>,
    base_options: Arc<Mutex<SimpleStreamOptions>>,
    active_token: Arc<Mutex<Option<CancellationToken>>>,
    idle_notify: Arc<Notify>,
    tool_execution: Arc<Mutex<ToolExecutionMode>>,
}

impl Agent {
    pub fn new(options: AgentOptions) -> Self {
        Self {
            state: Arc::new(Mutex::new(options.initial_state)),
            listeners: Arc::new(SyncMutex::new(Vec::new())),
            steering_queue: Arc::new(Mutex::new(PendingMessageQueue::new(options.steering_mode))),
            follow_up_queue: Arc::new(Mutex::new(PendingMessageQueue::new(options.follow_up_mode))),
            convert_to_llm: options.convert_to_llm,
            transform_context: options.transform_context,
            stream_fn: options.stream_fn,
            prepare_next_turn: options.prepare_next_turn,
            before_tool_call: options.before_tool_call,
            after_tool_call: options.after_tool_call,
            session_id: Arc::new(Mutex::new(options.session_id)),
            base_options: Arc::new(Mutex::new(options.options)),
            active_token: Arc::new(Mutex::new(None)),
            idle_notify: Arc::new(Notify::new()),
            tool_execution: Arc::new(Mutex::new(options.tool_execution)),
        }
    }

    pub async fn state(&self) -> AgentState {
        self.state.lock().await.clone()
    }

    pub async fn set_system_prompt(&self, system_prompt: impl Into<String>) {
        self.state.lock().await.system_prompt = system_prompt.into();
    }

    pub async fn set_model(&self, model: Model) {
        self.state.lock().await.model = model;
    }

    pub async fn set_thinking_level(&self, thinking_level: crate::ModelThinkingLevel) {
        self.state.lock().await.thinking_level = thinking_level;
    }

    pub async fn set_tools(&self, tools: Vec<DynAgentTool>) {
        self.state.lock().await.tools = tools;
    }

    pub async fn clear_tools(&self) {
        self.state.lock().await.tools.clear();
    }

    pub async fn set_messages(&self, messages: Vec<AgentMessage>) {
        self.state.lock().await.messages = messages;
    }

    pub async fn push_message(&self, message: AgentMessage) {
        self.state.lock().await.messages.push(message);
    }

    pub async fn clear_messages(&self) {
        self.state.lock().await.messages.clear();
    }

    pub async fn set_session_id(&self, session_id: Option<String>) {
        *self.session_id.lock().await = session_id;
    }

    pub async fn session_id(&self) -> Option<String> {
        self.session_id.lock().await.clone()
    }

    pub async fn set_options(&self, options: SimpleStreamOptions) {
        *self.base_options.lock().await = options;
    }

    pub async fn options(&self) -> SimpleStreamOptions {
        self.base_options.lock().await.clone()
    }

    pub async fn set_transport(&self, transport: Option<Transport>) {
        self.base_options.lock().await.stream.transport = transport;
    }

    pub async fn set_thinking_budgets(&self, thinking_budgets: Option<ThinkingBudgets>) {
        self.base_options.lock().await.thinking_budgets = thinking_budgets;
    }

    pub async fn set_max_retry_delay_ms(&self, max_retry_delay_ms: Option<u64>) {
        self.base_options.lock().await.stream.max_retry_delay_ms = max_retry_delay_ms;
    }

    pub async fn set_tool_execution(&self, tool_execution: ToolExecutionMode) {
        *self.tool_execution.lock().await = tool_execution;
    }

    pub async fn tool_execution(&self) -> ToolExecutionMode {
        *self.tool_execution.lock().await
    }

    pub fn subscribe<F, Fut>(&self, listener: F) -> AgentSubscription
    where
        F: Fn(AgentEvent, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = AgentResult<()>> + Send + 'static,
    {
        let listener: AgentEventListener =
            Arc::new(move |event, token| Box::pin(listener(event, token)));
        self.subscribe_boxed(listener)
    }

    fn subscribe_boxed(&self, listener: AgentEventListener) -> AgentSubscription {
        let listeners = self.listeners.clone();
        listeners.lock().push(listener.clone());
        AgentSubscription {
            listeners,
            listener: Some(listener),
        }
    }

    pub async fn steer(&self, message: AgentMessage) {
        self.steering_queue.lock().await.enqueue(message);
    }

    pub async fn follow_up(&self, message: AgentMessage) {
        self.follow_up_queue.lock().await.enqueue(message);
    }

    pub async fn set_steering_mode(&self, mode: QueueMode) {
        self.steering_queue.lock().await.mode = mode;
    }

    pub async fn steering_mode(&self) -> QueueMode {
        self.steering_queue.lock().await.mode
    }

    pub async fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_queue.lock().await.mode = mode;
    }

    pub async fn follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue.lock().await.mode
    }

    pub async fn clear_steering_queue(&self) {
        self.steering_queue.lock().await.clear();
    }

    pub async fn clear_follow_up_queue(&self) {
        self.follow_up_queue.lock().await.clear();
    }

    pub async fn clear_all_queues(&self) {
        self.clear_steering_queue().await;
        self.clear_follow_up_queue().await;
    }

    pub async fn has_queued_messages(&self) -> bool {
        self.steering_queue.lock().await.has_items()
            || self.follow_up_queue.lock().await.has_items()
    }

    pub async fn abort(&self) {
        if let Some(token) = self.active_token.lock().await.as_ref() {
            token.cancel();
        }
    }

    pub async fn cancellation_token(&self) -> Option<CancellationToken> {
        self.active_token.lock().await.clone()
    }

    pub async fn wait_for_idle(&self) {
        loop {
            let notified = self.idle_notify.notified();
            if self.active_token.lock().await.is_none() {
                return;
            }
            notified.await;
        }
    }

    pub async fn reset(&self) {
        let mut state = self.state.lock().await;
        state.messages.clear();
        state.is_streaming = false;
        state.streaming_message = None;
        state.pending_tool_calls.clear();
        state.error_message = None;
        drop(state);
        self.clear_all_queues().await;
    }

    pub async fn prompt_text(
        &self,
        input: impl Into<String>,
        images: Vec<ImageContent>,
    ) -> AgentResult<()> {
        self.prompt_messages(vec![user_message(input, images)])
            .await
    }

    pub async fn prompt_messages(&self, messages: Vec<AgentMessage>) -> AgentResult<()> {
        self.run_with_lifecycle(messages, false, false).await
    }

    pub async fn continue_run(&self) -> AgentResult<()> {
        let token = self.acquire_active_token().await?;
        let last = self.state.lock().await.messages.last().cloned();
        let action = match last {
            None => Err(AgentError::NoMessagesToContinue),
            Some(Message::Assistant(_)) => {
                let queued = self.steering_queue.lock().await.drain();
                if !queued.is_empty() {
                    Ok((queued, false, true))
                } else {
                    let follow_up = self.follow_up_queue.lock().await.drain();
                    if follow_up.is_empty() {
                        Err(AgentError::CannotContinueFromAssistant)
                    } else {
                        Ok((follow_up, false, false))
                    }
                }
            }
            Some(_) => Ok((Vec::new(), true, false)),
        };

        match action {
            Ok((prompts, continue_existing, skip_initial_steering_poll)) => {
                self.run_with_lifecycle_after_acquire(
                    prompts,
                    continue_existing,
                    skip_initial_steering_poll,
                    token,
                )
                .await
            }
            Err(error) => {
                self.clear_active_token().await;
                Err(error)
            }
        }
    }

    async fn run_with_lifecycle(
        &self,
        prompts: Vec<AgentMessage>,
        continue_existing: bool,
        skip_initial_steering_poll: bool,
    ) -> AgentResult<()> {
        let token = self.acquire_active_token().await?;
        self.run_with_lifecycle_after_acquire(
            prompts,
            continue_existing,
            skip_initial_steering_poll,
            token,
        )
        .await
    }

    async fn acquire_active_token(&self) -> AgentResult<CancellationToken> {
        let mut active = self.active_token.lock().await;
        if active.is_some() {
            return Err(AgentError::AlreadyProcessing);
        }
        let token = CancellationToken::new();
        *active = Some(token.clone());
        Ok(token)
    }

    async fn clear_active_token(&self) {
        *self.active_token.lock().await = None;
        self.idle_notify.notify_waiters();
    }

    async fn run_with_lifecycle_after_acquire(
        &self,
        prompts: Vec<AgentMessage>,
        continue_existing: bool,
        skip_initial_steering_poll: bool,
        token: CancellationToken,
    ) -> AgentResult<()> {
        {
            let mut state = self.state.lock().await;
            state.is_streaming = true;
            state.streaming_message = None;
            state.error_message = None;
        }

        let result = if continue_existing {
            run_agent_loop_continue(
                self.create_context_snapshot().await,
                self.create_loop_config(skip_initial_steering_poll).await,
                self.event_sink(),
                Some(token.clone()),
                self.stream_fn.clone(),
            )
            .await
        } else {
            run_agent_loop(
                prompts,
                self.create_context_snapshot().await,
                self.create_loop_config(skip_initial_steering_poll).await,
                self.event_sink(),
                Some(token.clone()),
                self.stream_fn.clone(),
            )
            .await
        };

        let failure_result = if let Err(error) = result {
            let aborted = token.is_cancelled();
            self.emit_run_failure(error, aborted).await
        } else {
            Ok(())
        };

        let mut state = self.state.lock().await;
        state.is_streaming = false;
        state.streaming_message = None;
        state.pending_tool_calls.clear();
        drop(state);
        self.clear_active_token().await;
        failure_result
    }

    async fn emit_run_failure(&self, error: AgentError, aborted: bool) -> AgentResult<()> {
        let state = self.state.lock().await;
        let failure = Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: String::new(),
                text_signature: None,
            })],
            api: state.model.api.clone(),
            provider: state.model.provider.clone(),
            model: state.model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: if aborted {
                StopReason::Aborted
            } else {
                StopReason::Error
            },
            error_message: Some(error.to_string()),
            timestamp: crate::utils::time::now_millis(),
        });
        drop(state);

        let sink = self.event_sink();
        sink(AgentEvent::MessageStart {
            message: failure.clone(),
        })
        .await?;
        sink(AgentEvent::MessageEnd {
            message: failure.clone(),
        })
        .await?;
        sink(AgentEvent::TurnEnd {
            message: failure.clone(),
            tool_results: Vec::new(),
        })
        .await?;
        sink(AgentEvent::AgentEnd {
            messages: vec![failure],
        })
        .await
    }

    async fn create_context_snapshot(&self) -> AgentContext {
        let state = self.state.lock().await;
        AgentContext {
            system_prompt: state.system_prompt.clone(),
            messages: state.messages.clone(),
            tools: state.tools.clone(),
        }
    }

    async fn create_loop_config(&self, skip_initial_steering_poll: bool) -> AgentLoopConfig {
        let (model, thinking_level) = {
            let state = self.state.lock().await;
            (state.model.clone(), state.thinking_level)
        };
        let mut options = self.base_options.lock().await.clone();
        options.reasoning =
            (thinking_level != crate::ModelThinkingLevel::Off).then_some(thinking_level);
        options.stream.session_id = self.session_id.lock().await.clone();

        let steering_queue = self.steering_queue.clone();
        let follow_up_queue = self.follow_up_queue.clone();
        let skip_initial_steering_poll = Arc::new(Mutex::new(skip_initial_steering_poll));
        AgentLoopConfig {
            model,
            options,
            convert_to_llm: self
                .convert_to_llm
                .clone()
                .unwrap_or_else(default_convert_to_llm),
            transform_context: self.transform_context.clone(),
            should_stop_after_turn: None,
            prepare_next_turn: self.prepare_next_turn.clone().map(|prepare_next_turn| {
                Arc::new(
                    move |_context: PrepareNextTurnContext, token: Option<CancellationToken>| {
                        prepare_next_turn(token)
                    },
                ) as PrepareNextTurnFn
            }),
            get_steering_messages: Some(Arc::new(move || {
                let steering_queue = steering_queue.clone();
                let skip_initial_steering_poll = skip_initial_steering_poll.clone();
                Box::pin(async move {
                    let mut skip = skip_initial_steering_poll.lock().await;
                    if *skip {
                        *skip = false;
                        Vec::new()
                    } else {
                        drop(skip);
                        steering_queue.lock().await.drain()
                    }
                })
            })),
            get_follow_up_messages: Some(Arc::new(move || {
                let follow_up_queue = follow_up_queue.clone();
                Box::pin(async move { follow_up_queue.lock().await.drain() })
            })),
            before_tool_call: self.before_tool_call.clone(),
            after_tool_call: self.after_tool_call.clone(),
            tool_execution: *self.tool_execution.lock().await,
        }
    }

    fn event_sink(&self) -> AgentEventSink {
        let state = self.state.clone();
        let listeners = self.listeners.clone();
        let active_token = self.active_token.clone();
        Arc::new(move |event| {
            let state = state.clone();
            let listeners = listeners.clone();
            let active_token = active_token.clone();
            Box::pin(async move {
                {
                    let mut state = state.lock().await;
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
                                && let Some(error) = &assistant.error_message
                            {
                                state.error_message = Some(error.clone());
                            }
                        }
                        AgentEvent::AgentEnd { .. } => {
                            state.streaming_message = None;
                        }
                        _ => {}
                    }
                }
                let listeners = listeners.lock().clone();
                let token = active_token.lock().await.clone().ok_or_else(|| {
                    AgentError::Other("agent listener invoked outside active run".to_string())
                })?;
                for listener in listeners {
                    listener(event.clone(), token.clone()).await?;
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
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::FutureExt;
    use serde_json::{Value, json};

    use super::{AgentState, StreamFn};
    use crate::agent_types::{
        AgentTool, AgentToolResult, AgentToolUpdateCallback, ToolExecutionMode, user_message,
    };
    use crate::event_stream::create_assistant_message_event_stream;
    use crate::providers::faux::{
        FauxAssistantMessageOptions, FauxModelDefinition, FauxResponseStep, FauxTokenSize,
        RegisterFauxProviderOptions, faux_assistant_message, faux_text, faux_thinking,
        faux_tool_call, register_faux_provider,
    };
    use crate::{
        Agent, AgentError, AgentEvent, AgentOptions, AssistantContent, AssistantMessage,
        AssistantMessageEvent, ImageContent, Message, Model, ModelThinkingLevel, StopReason,
        TextContent, Tool, ToolResultContent, ToolResultMessage,
    };

    fn text_from_message(message: &Message) -> String {
        match message {
            Message::Assistant(assistant) => assistant
                .content
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            Message::ToolResult(tool_result) => tool_result
                .content
                .iter()
                .filter_map(|content| match content {
                    ToolResultContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }

    struct CalculateTool;

    #[async_trait]
    impl AgentTool for CalculateTool {
        fn definition(&self) -> Tool {
            Tool {
                name: "calculate".to_string(),
                description: "Evaluate a simple multiplication expression.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "expression": { "type": "string" }
                    },
                    "required": ["expression"]
                }),
            }
        }

        fn label(&self) -> &str {
            "Calculate"
        }

        async fn execute(
            &self,
            _tool_call_id: &str,
            args: Value,
            _cancellation_token: Option<tokio_util::sync::CancellationToken>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> crate::AgentResult<AgentToolResult> {
            let expression = args
                .get("expression")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let value = match expression {
                "123 * 456" => 56_088,
                other => {
                    return Ok(AgentToolResult::text(format!(
                        "Unsupported expression: {other}"
                    )));
                }
            };
            Ok(AgentToolResult::text(format!("{expression} = {value}")))
        }
    }

    fn assistant_text_message(model: &Model, text: impl Into<String>) -> AssistantMessage {
        let mut message = AssistantMessage::empty_for(model);
        message.content = vec![AssistantContent::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })];
        message
    }

    fn immediate_stream_fn(text: impl Into<String>) -> StreamFn {
        let text = Arc::new(text.into());
        Arc::new(move |model, _context, _options| {
            let text = Arc::clone(&text);
            async move {
                let (mut sender, stream) = create_assistant_message_event_stream();
                let message = assistant_text_message(&model, text.as_str());
                sender.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message,
                });
                Ok(stream)
            }
            .boxed()
        })
    }

    fn counted_stream_fn(counter: Arc<AtomicUsize>) -> StreamFn {
        Arc::new(move |model, _context, _options| {
            let counter = Arc::clone(&counter);
            async move {
                let count = counter.fetch_add(1, Ordering::SeqCst) + 1;
                let (mut sender, stream) = create_assistant_message_event_stream();
                let message = assistant_text_message(&model, format!("Processed {count}"));
                sender.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message,
                });
                Ok(stream)
            }
            .boxed()
        })
    }

    fn abortable_stream_fn() -> StreamFn {
        Arc::new(|model, _context, options| {
            async move {
                let (mut sender, stream) = create_assistant_message_event_stream();
                let token = options.stream.cancellation_token.clone();
                tokio::spawn(async move {
                    sender.push(AssistantMessageEvent::Start {
                        partial: assistant_text_message(&model, ""),
                    });
                    if let Some(token) = token {
                        token.cancelled().await;
                        let mut message = assistant_text_message(&model, "Aborted");
                        message.stop_reason = StopReason::Aborted;
                        message.error_message = Some("Aborted".to_string());
                        sender.push(AssistantMessageEvent::Error {
                            reason: StopReason::Aborted,
                            error: message,
                        });
                    } else {
                        std::future::pending::<()>().await;
                    }
                });
                Ok(stream)
            }
            .boxed()
        })
    }

    #[tokio::test]
    async fn should_create_an_agent_instance_with_default_state() {
        let agent = Agent::default();
        let state = agent.state().await;

        assert_eq!(state.system_prompt, "");
        assert_eq!(state.model.id, "unknown");
        assert_eq!(state.thinking_level, crate::ModelThinkingLevel::Off);
        assert!(state.tools.is_empty());
        assert!(state.messages.is_empty());
        assert!(!state.is_streaming);
        assert!(state.streaming_message.is_none());
        assert!(state.pending_tool_calls.is_empty());
        assert!(state.error_message.is_none());
    }

    #[tokio::test]
    async fn should_create_an_agent_instance_with_custom_initial_state() {
        let model = Model {
            id: "custom-model".to_string(),
            api: "openai-completions".to_string(),
            provider: "openai".to_string(),
            ..Default::default()
        };
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt: "You are a helpful assistant.".to_string(),
                model: model.clone(),
                thinking_level: ModelThinkingLevel::Low,
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });

        let state = agent.state().await;
        assert_eq!(state.system_prompt, "You are a helpful assistant.");
        assert_eq!(state.model, model);
        assert_eq!(state.thinking_level, ModelThinkingLevel::Low);
    }

    #[tokio::test]
    async fn should_build_agent_state() {
        let model = Model {
            id: "builder-model".to_string(),
            api: "openai-completions".to_string(),
            provider: "openai".to_string(),
            ..Default::default()
        };
        let state = AgentState::builder(model.clone())
            .system_prompt("You are terse.")
            .thinking_level(ModelThinkingLevel::Medium)
            .tool(Arc::new(CalculateTool))
            .message(Message::user_text("Hello"))
            .build();

        assert_eq!(state.system_prompt, "You are terse.");
        assert_eq!(state.model, model);
        assert_eq!(state.thinking_level, ModelThinkingLevel::Medium);
        assert_eq!(state.tools.len(), 1);
        assert_eq!(state.messages, vec![Message::user_text("Hello")]);
    }

    #[tokio::test]
    async fn should_build_agent_options() {
        let model = Model {
            id: "builder-model".to_string(),
            api: "openai-completions".to_string(),
            provider: "openai".to_string(),
            ..Default::default()
        };
        let message = Message::user_text("Hello");
        let options = AgentOptions::builder(model.clone())
            .system_prompt("You are precise.")
            .thinking_level(ModelThinkingLevel::High)
            .message(message.clone())
            .session_id("session-123")
            .tool_execution(ToolExecutionMode::Sequential)
            .build();

        assert_eq!(options.initial_state.system_prompt, "You are precise.");
        assert_eq!(options.initial_state.model, model);
        assert_eq!(
            options.initial_state.thinking_level,
            ModelThinkingLevel::High
        );
        assert_eq!(options.initial_state.messages, vec![message]);
        assert_eq!(options.session_id.as_deref(), Some("session-123"));
        assert_eq!(options.tool_execution, ToolExecutionMode::Sequential);
    }

    #[tokio::test]
    async fn should_update_state_with_mutators() {
        let agent = Agent::default();
        agent.set_system_prompt("Custom prompt").await;
        agent
            .set_model(Model {
                id: "new-model".to_string(),
                api: "anthropic-messages".to_string(),
                provider: "anthropic".to_string(),
                ..Default::default()
            })
            .await;
        agent.set_thinking_level(ModelThinkingLevel::High).await;
        agent.set_messages(vec![Message::user_text("Hello")]).await;
        agent.push_message(Message::user_text("Next")).await;

        let state = agent.state().await;
        assert_eq!(state.system_prompt, "Custom prompt");
        assert_eq!(state.model.id, "new-model");
        assert_eq!(state.thinking_level, ModelThinkingLevel::High);
        assert_eq!(state.messages.len(), 2);

        agent.clear_messages().await;
        assert!(agent.state().await.messages.is_empty());
    }

    #[tokio::test]
    async fn should_subscribe_to_events() {
        let agent = Agent::default();
        let event_count = Arc::new(AtomicUsize::new(0));

        let unsubscribe = agent.subscribe({
            let event_count = Arc::clone(&event_count);
            move |_event, _token| {
                let event_count = Arc::clone(&event_count);
                async move {
                    event_count.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            }
        });

        agent.set_system_prompt("Test prompt").await;
        assert_eq!(event_count.load(Ordering::SeqCst), 0);
        assert_eq!(agent.state().await.system_prompt, "Test prompt");

        assert!(unsubscribe.unsubscribe());
        agent.set_system_prompt("Another prompt").await;
        assert_eq!(event_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn should_unsubscribe_when_subscription_is_dropped() {
        let agent = Agent::new(AgentOptions {
            stream_fn: Some(immediate_stream_fn("ok")),
            ..AgentOptions::default()
        });
        let event_count = Arc::new(AtomicUsize::new(0));

        let subscription = agent.subscribe({
            let event_count = Arc::clone(&event_count);
            move |_event, _token| {
                let event_count = Arc::clone(&event_count);
                async move {
                    event_count.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            }
        });
        drop(subscription);

        agent.prompt_text("hello", Vec::new()).await.unwrap();

        assert_eq!(event_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn should_support_steering_message_queue() {
        let agent = Agent::default();
        let message = user_message("Steering message", Vec::new());

        agent.steer(message.clone()).await;

        assert!(!agent.state().await.messages.contains(&message));
        assert!(agent.has_queued_messages().await);
    }

    #[tokio::test]
    async fn should_support_follow_up_message_queue() {
        let agent = Agent::default();
        let message = user_message("Follow-up message", Vec::new());

        agent.follow_up(message.clone()).await;

        assert!(!agent.state().await.messages.contains(&message));
        assert!(agent.has_queued_messages().await);
    }

    #[tokio::test]
    async fn should_handle_abort_controller() {
        let agent = Agent::default();

        agent.abort().await;

        assert!(agent.cancellation_token().await.is_none());
    }

    #[tokio::test]
    async fn should_await_async_subscribers_before_prompt_resolves() {
        let agent = Arc::new(Agent::new(AgentOptions {
            stream_fn: Some(immediate_stream_fn("ok")),
            ..AgentOptions::default()
        }));
        let barrier = Arc::new(tokio::sync::Notify::new());
        let listener_finished = Arc::new(StdMutex::new(false));

        let _subscription = agent.subscribe({
            let barrier = Arc::clone(&barrier);
            let listener_finished = Arc::clone(&listener_finished);
            move |event, _token| {
                let barrier = Arc::clone(&barrier);
                let listener_finished = Arc::clone(&listener_finished);
                async move {
                    if matches!(event, AgentEvent::AgentEnd { .. }) {
                        barrier.notified().await;
                        *listener_finished.lock().unwrap() = true;
                    }
                    Ok(())
                }
            }
        });

        let prompt = tokio::spawn({
            let agent = Arc::clone(&agent);
            async move { agent.prompt_text("hello", Vec::new()).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!prompt.is_finished());
        assert!(!*listener_finished.lock().unwrap());
        assert!(agent.state().await.is_streaming);

        barrier.notify_waiters();
        prompt.await.unwrap().expect("prompt succeeds");

        assert!(*listener_finished.lock().unwrap());
        assert!(!agent.state().await.is_streaming);
    }

    #[tokio::test]
    async fn wait_for_idle_should_wait_for_async_subscribers() {
        let agent = Arc::new(Agent::new(AgentOptions {
            stream_fn: Some(immediate_stream_fn("ok")),
            ..AgentOptions::default()
        }));
        let barrier = Arc::new(tokio::sync::Notify::new());

        let _subscription = agent.subscribe({
            let barrier = Arc::clone(&barrier);
            move |event, _token| {
                let barrier = Arc::clone(&barrier);
                async move {
                    if matches!(
                        event,
                        AgentEvent::MessageEnd {
                            message: Message::Assistant(_)
                        }
                    ) {
                        barrier.notified().await;
                    }
                    Ok(())
                }
            }
        });

        let prompt = tokio::spawn({
            let agent = Arc::clone(&agent);
            async move { agent.prompt_text("hello", Vec::new()).await }
        });
        let idle = tokio::spawn({
            let agent = Arc::clone(&agent);
            async move { agent.wait_for_idle().await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!idle.is_finished());
        assert!(agent.state().await.is_streaming);

        barrier.notify_waiters();
        prompt.await.unwrap().expect("prompt succeeds");
        idle.await.unwrap();

        assert!(!agent.state().await.is_streaming);
    }

    #[tokio::test]
    async fn should_pass_the_active_abort_signal_to_subscribers() {
        let agent = Arc::new(Agent::new(AgentOptions {
            stream_fn: Some(abortable_stream_fn()),
            ..AgentOptions::default()
        }));
        let received_token = Arc::new(StdMutex::new(None));

        let _subscription = agent.subscribe({
            let received_token = Arc::clone(&received_token);
            move |event, token| {
                let received_token = Arc::clone(&received_token);
                async move {
                    if matches!(event, AgentEvent::AgentStart) {
                        *received_token.lock().unwrap() = Some(token);
                    }
                    Ok(())
                }
            }
        });

        let prompt = tokio::spawn({
            let agent = Arc::clone(&agent);
            async move { agent.prompt_text("hello", Vec::new()).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;

        let token = received_token
            .lock()
            .unwrap()
            .clone()
            .expect("listener saw token");
        assert!(!token.is_cancelled());

        agent.abort().await;
        prompt.await.unwrap().expect("aborted run settles");
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn emits_full_lifecycle_events_for_thrown_run_failures() {
        let agent = Agent::new(AgentOptions {
            stream_fn: Some(Arc::new(|_model, _context, _options| {
                async move { Err(crate::Error::Validation("provider exploded".to_string())) }
                    .boxed()
            })),
            ..AgentOptions::default()
        });
        let events = Arc::new(StdMutex::new(Vec::new()));
        let _subscription = agent.subscribe({
            let events = Arc::clone(&events);
            move |event, _token| {
                let events = Arc::clone(&events);
                async move {
                    events.lock().unwrap().push(match event {
                        AgentEvent::AgentStart => "agent_start",
                        AgentEvent::TurnStart => "turn_start",
                        AgentEvent::MessageStart { .. } => "message_start",
                        AgentEvent::MessageEnd { .. } => "message_end",
                        AgentEvent::TurnEnd { .. } => "turn_end",
                        AgentEvent::AgentEnd { .. } => "agent_end",
                        AgentEvent::MessageUpdate { .. } => "message_update",
                        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
                        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
                        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
                    });
                    Ok(())
                }
            }
        });

        agent
            .prompt_text("hello", Vec::new())
            .await
            .expect("failure is emitted as assistant error");

        assert_eq!(
            *events.lock().unwrap(),
            vec![
                "agent_start",
                "turn_start",
                "message_start",
                "message_end",
                "message_start",
                "message_end",
                "turn_end",
                "agent_end",
            ]
        );
        let state = agent.state().await;
        let Some(Message::Assistant(last_message)) = state.messages.last() else {
            panic!("expected assistant message");
        };
        assert_eq!(last_message.stop_reason, StopReason::Error);
        assert_eq!(
            last_message.error_message.as_deref(),
            Some("provider exploded")
        );
        assert_eq!(state.error_message.as_deref(), Some("provider exploded"));
    }

    #[tokio::test]
    async fn should_throw_when_prompt_called_while_streaming() {
        let agent = Arc::new(Agent::new(AgentOptions {
            stream_fn: Some(abortable_stream_fn()),
            ..AgentOptions::default()
        }));
        let first_prompt = tokio::spawn({
            let agent = Arc::clone(&agent);
            async move { agent.prompt_text("First message", Vec::new()).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(agent.state().await.is_streaming);

        assert!(matches!(
            agent.prompt_text("Second message", Vec::new()).await,
            Err(AgentError::AlreadyProcessing)
        ));

        agent.abort().await;
        first_prompt.await.unwrap().expect("aborted run settles");
    }

    #[tokio::test]
    async fn should_throw_when_continue_called_while_streaming() {
        let agent = Arc::new(Agent::new(AgentOptions {
            stream_fn: Some(abortable_stream_fn()),
            ..AgentOptions::default()
        }));
        let first_prompt = tokio::spawn({
            let agent = Arc::clone(&agent);
            async move { agent.prompt_text("First message", Vec::new()).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(agent.state().await.is_streaming);

        assert!(matches!(
            agent.continue_run().await,
            Err(AgentError::AlreadyProcessing)
        ));

        agent.abort().await;
        first_prompt.await.unwrap().expect("aborted run settles");
    }

    #[tokio::test]
    async fn continue_should_process_queued_follow_up_messages_after_an_assistant_turn() {
        let counter = Arc::new(AtomicUsize::new(0));
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                messages: vec![
                    Message::user_text("Initial"),
                    Message::Assistant(assistant_text_message(
                        &AgentState::default().model,
                        "Initial response",
                    )),
                ],
                ..AgentState::default()
            },
            stream_fn: Some(counted_stream_fn(Arc::clone(&counter))),
            ..AgentOptions::default()
        });

        agent
            .follow_up(user_message("Queued follow-up", Vec::new()))
            .await;
        agent.continue_run().await.expect("follow-up continues");
        let state = agent.state().await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert!(state.messages.iter().any(|message| {
            matches!(message, Message::User(user) if matches!(&user.content, crate::UserMessageContent::Parts(parts) if parts.iter().any(|part| matches!(part, crate::UserContent::Text(text) if text.text == "Queued follow-up"))))
        }));
        assert!(matches!(state.messages.last(), Some(Message::Assistant(_))));
    }

    #[tokio::test]
    async fn continue_should_keep_one_at_a_time_steering_semantics_from_assistant_tail() {
        let counter = Arc::new(AtomicUsize::new(0));
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                messages: vec![
                    Message::user_text("Initial"),
                    Message::Assistant(assistant_text_message(
                        &AgentState::default().model,
                        "Initial response",
                    )),
                ],
                ..AgentState::default()
            },
            stream_fn: Some(counted_stream_fn(Arc::clone(&counter))),
            ..AgentOptions::default()
        });
        agent.steer(user_message("Steering 1", Vec::new())).await;
        agent.steer(user_message("Steering 2", Vec::new())).await;
        agent.continue_run().await.expect("steering continues");

        let state = agent.state().await;
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        let recent_roles = state
            .messages
            .iter()
            .rev()
            .take(4)
            .map(|message| match message {
                Message::User(_) => "user",
                Message::Assistant(_) => "assistant",
                Message::ToolResult(_) => "toolResult",
                Message::Custom(_) => "custom",
            })
            .collect::<Vec<_>>();
        assert_eq!(recent_roles, vec!["assistant", "user", "assistant", "user"]);
    }

    #[tokio::test]
    async fn forwards_session_id_to_stream_fn_options() {
        let received_session_id = Arc::new(StdMutex::new(None));
        let stream_fn: StreamFn = Arc::new({
            let received_session_id = Arc::clone(&received_session_id);
            move |model, _context, options| {
                let received_session_id = Arc::clone(&received_session_id);
                async move {
                    *received_session_id.lock().unwrap() = options.stream.session_id.clone();
                    let (mut sender, stream) = create_assistant_message_event_stream();
                    let message = assistant_text_message(&model, "ok");
                    sender.push(AssistantMessageEvent::Done {
                        reason: StopReason::Stop,
                        message,
                    });
                    Ok(stream)
                }
                .boxed()
            }
        });
        let agent = Agent::new(AgentOptions {
            session_id: Some("session-abc".to_string()),
            stream_fn: Some(stream_fn),
            ..AgentOptions::default()
        });

        agent
            .prompt_text("hello", Vec::new())
            .await
            .expect("prompt succeeds");

        assert_eq!(
            *received_session_id.lock().unwrap(),
            Some("session-abc".to_string())
        );
    }

    #[tokio::test]
    async fn prepare_next_turn_receives_active_abort_signal() {
        let called = Arc::new(AtomicUsize::new(0));
        let saw_token = Arc::new(AtomicUsize::new(0));
        let agent = Agent::new(AgentOptions {
            prepare_next_turn: Some(Arc::new({
                let called = Arc::clone(&called);
                let saw_token = Arc::clone(&saw_token);
                move |token| {
                    let called = Arc::clone(&called);
                    let saw_token = Arc::clone(&saw_token);
                    async move {
                        called.fetch_add(1, Ordering::SeqCst);
                        if token.is_some() {
                            saw_token.fetch_add(1, Ordering::SeqCst);
                        }
                        None
                    }
                    .boxed()
                }
            })),
            stream_fn: Some(immediate_stream_fn("done")),
            ..AgentOptions::default()
        });

        agent
            .prompt_text("hello", Vec::new())
            .await
            .expect("prompt succeeds");

        assert_eq!(called.load(Ordering::SeqCst), 1);
        assert_eq!(saw_token.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn handles_a_basic_text_prompt() {
        let registration = register_faux_provider(None);
        registration.set_responses([faux_assistant_message("4", None)]);
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt: "You are a helpful assistant.".to_string(),
                model: registration.get_model(),
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });

        agent
            .prompt_text("What is 2+2? Answer with just the number.", Vec::new())
            .await
            .expect("prompt succeeds");

        let state = agent.state().await;
        assert!(!state.is_streaming);
        assert_eq!(state.messages.len(), 2);
        assert!(matches!(state.messages.first(), Some(Message::User(_))));
        assert!(matches!(state.messages.get(1), Some(Message::Assistant(_))));
        assert!(text_from_message(&state.messages[1]).contains('4'));
        registration.unregister();
    }

    #[tokio::test]
    async fn executes_tools_and_tracks_pending_tool_calls() {
        let registration = register_faux_provider(None);
        registration.set_responses([
            faux_assistant_message(
                vec![
                    faux_text("Let me calculate that."),
                    faux_tool_call(
                        "calculate",
                        json!({ "expression": "123 * 456" }),
                        Some("calc-1".to_string()),
                    ),
                ],
                Some(FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                }),
            ),
            faux_assistant_message("The result is 56088.", None),
        ]);
        let agent = Arc::new(Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt:
                    "You are a helpful assistant. Always use the calculator tool for math."
                        .to_string(),
                model: registration.get_model(),
                tools: vec![Arc::new(CalculateTool)],
                ..AgentState::default()
            },
            ..AgentOptions::default()
        }));

        let pending_during_events = Arc::new(StdMutex::new(Vec::new()));
        let _subscription = agent.subscribe({
            let pending_during_events = Arc::clone(&pending_during_events);
            let agent = Arc::clone(&agent);
            move |event, _token| {
                let pending_during_events = Arc::clone(&pending_during_events);
                let agent = Arc::clone(&agent);
                async move {
                    if matches!(
                        event,
                        AgentEvent::ToolExecutionStart { .. } | AgentEvent::ToolExecutionEnd { .. }
                    ) {
                        let state = agent.state().await;
                        pending_during_events.lock().unwrap().push((
                            match event {
                                AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
                                AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
                                _ => unreachable!(),
                            },
                            state.pending_tool_calls.iter().cloned().collect::<Vec<_>>(),
                        ));
                    }
                    Ok(())
                }
            }
        });

        agent
            .prompt_text("Calculate 123 * 456 using the calculator tool.", Vec::new())
            .await
            .expect("prompt succeeds");

        let state = agent.state().await;
        assert!(!state.is_streaming);
        let tool_result = state
            .messages
            .iter()
            .find(|message| matches!(message, Message::ToolResult(_)))
            .expect("tool result message");
        assert!(text_from_message(tool_result).contains("123 * 456 = 56088"));
        assert!(text_from_message(state.messages.last().unwrap()).contains("56088"));
        assert!(state.pending_tool_calls.is_empty());
        assert_eq!(
            *pending_during_events.lock().unwrap(),
            vec![
                ("tool_execution_start", vec!["calc-1".to_string()]),
                ("tool_execution_end", Vec::<String>::new()),
            ]
        );
        registration.unregister();
    }

    #[tokio::test]
    async fn handles_abort_during_streaming() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            tokens_per_second: Some(20.0),
            token_size: Some(FauxTokenSize {
                min: Some(2),
                max: Some(2),
            }),
            ..Default::default()
        }));
        registration.set_responses([faux_assistant_message(
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen",
            None,
        )]);
        let agent = Arc::new(Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt: "You are a helpful assistant.".to_string(),
                model: registration.get_model(),
                ..AgentState::default()
            },
            ..AgentOptions::default()
        }));

        let prompt = tokio::spawn({
            let agent = Arc::clone(&agent);
            async move {
                agent
                    .prompt_text("Count slowly from 1 to 20.", Vec::new())
                    .await
            }
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        agent.abort().await;
        prompt.await.unwrap().expect("aborted run settles");

        let state = agent.state().await;
        assert!(!state.is_streaming);
        let Some(Message::Assistant(last_message)) = state.messages.last() else {
            panic!("expected assistant message");
        };
        assert_eq!(last_message.stop_reason, StopReason::Aborted);
        assert!(last_message.error_message.is_some());
        assert_eq!(state.error_message, last_message.error_message);
        registration.unregister();
    }

    #[tokio::test]
    async fn emits_lifecycle_updates_while_streaming() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            ..Default::default()
        }));
        registration.set_responses([faux_assistant_message("1 2 3 4 5", None)]);
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt: "You are a helpful assistant.".to_string(),
                model: registration.get_model(),
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });
        let events = Arc::new(StdMutex::new(Vec::new()));
        let _subscription = agent.subscribe({
            let events = Arc::clone(&events);
            move |event, _token| {
                let events = Arc::clone(&events);
                async move {
                    events.lock().unwrap().push(match event {
                        AgentEvent::AgentStart => "agent_start",
                        AgentEvent::TurnStart => "turn_start",
                        AgentEvent::MessageStart { .. } => "message_start",
                        AgentEvent::MessageUpdate { .. } => "message_update",
                        AgentEvent::MessageEnd { .. } => "message_end",
                        AgentEvent::TurnEnd { .. } => "turn_end",
                        AgentEvent::AgentEnd { .. } => "agent_end",
                        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
                        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
                        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
                    });
                    Ok(())
                }
            }
        });

        agent
            .prompt_text("Count from 1 to 5.", Vec::new())
            .await
            .expect("prompt succeeds");

        {
            let events = events.lock().unwrap();
            assert!(events.contains(&"agent_start"));
            assert!(events.contains(&"turn_start"));
            assert!(events.contains(&"message_start"));
            assert!(events.contains(&"message_update"));
            assert!(events.contains(&"message_end"));
            assert!(events.contains(&"turn_end"));
            assert!(events.contains(&"agent_end"));
            assert!(
                events.iter().position(|event| *event == "agent_start")
                    < events.iter().position(|event| *event == "message_start")
            );
            assert!(
                events.iter().position(|event| *event == "message_start")
                    < events.iter().position(|event| *event == "message_end")
            );
            assert!(
                events.iter().position(|event| *event == "message_end")
                    < events.iter().rposition(|event| *event == "agent_end")
            );
        }
        let state = agent.state().await;
        assert!(!state.is_streaming);
        assert_eq!(state.messages.len(), 2);
        registration.unregister();
    }

    #[tokio::test]
    async fn maintains_context_across_multiple_turns() {
        let registration = register_faux_provider(None);
        registration.set_responses([
            FauxResponseStep::from(faux_assistant_message("Nice to meet you, Alice.", None)),
            FauxResponseStep::factory(|context, _options, _state, _model| async move {
                let has_alice = context.messages.iter().any(|message| match message {
                    Message::User(user) => match &user.content {
                        crate::UserMessageContent::Text(text) => text.contains("Alice"),
                        crate::UserMessageContent::Parts(parts) => parts.iter().any(|part| {
                            matches!(part, crate::UserContent::Text(text) if text.text.contains("Alice"))
                        }),
                    },
                    _ => false,
                });
                Ok(faux_assistant_message(
                    if has_alice {
                        "Your name is Alice."
                    } else {
                        "I do not know your name."
                    },
                    None,
                ))
            }),
        ]);
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt: "You are a helpful assistant.".to_string(),
                model: registration.get_model(),
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });

        agent
            .prompt_text("My name is Alice.", Vec::new())
            .await
            .expect("first prompt succeeds");
        assert_eq!(agent.state().await.messages.len(), 2);

        agent
            .prompt_text("What is my name?", Vec::new())
            .await
            .expect("second prompt succeeds");

        let state = agent.state().await;
        assert_eq!(state.messages.len(), 4);
        assert!(
            text_from_message(&state.messages[3])
                .to_lowercase()
                .contains("alice")
        );
        registration.unregister();
    }

    #[tokio::test]
    async fn preserves_thinking_content_blocks() {
        let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
            models: vec![FauxModelDefinition {
                id: "faux-reasoning".to_string(),
                reasoning: Some(true),
                ..Default::default()
            }],
            ..Default::default()
        }));
        registration.set_responses([faux_assistant_message(
            vec![faux_thinking("step by step"), faux_text("4")],
            None,
        )]);
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt: "You are a helpful assistant.".to_string(),
                model: registration.get_model(),
                thinking_level: ModelThinkingLevel::Low,
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });

        agent
            .prompt_text("What is 2+2?", Vec::new())
            .await
            .expect("prompt succeeds");

        let state = agent.state().await;
        let Message::Assistant(assistant) = &state.messages[1] else {
            panic!("expected assistant message");
        };
        assert_eq!(
            assistant.content,
            vec![faux_thinking("step by step"), faux_text("4")]
        );
        registration.unregister();
    }

    #[tokio::test]
    async fn continue_throws_when_no_messages_in_context() {
        let registration = register_faux_provider(None);
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt: "Test".to_string(),
                model: registration.get_model(),
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });

        assert!(matches!(
            agent.continue_run().await,
            Err(AgentError::NoMessagesToContinue)
        ));
        registration.unregister();
    }

    #[tokio::test]
    async fn continue_throws_when_last_message_is_assistant() {
        let registration = register_faux_provider(None);
        let model = registration.get_model();
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt: "Test".to_string(),
                model: model.clone(),
                messages: vec![Message::Assistant(assistant_text_message(&model, "Hello"))],
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });

        assert!(matches!(
            agent.continue_run().await,
            Err(AgentError::CannotContinueFromAssistant)
        ));
        registration.unregister();
    }

    #[tokio::test]
    async fn continues_and_gets_a_response_when_last_message_is_user() {
        let registration = register_faux_provider(None);
        registration.set_responses([faux_assistant_message("HELLO WORLD", None)]);
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt: "You are a helpful assistant. Follow instructions exactly."
                    .to_string(),
                model: registration.get_model(),
                messages: vec![user_message("Say exactly: HELLO WORLD", Vec::new())],
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });

        agent.continue_run().await.expect("continue succeeds");

        let state = agent.state().await;
        assert!(!state.is_streaming);
        assert_eq!(state.messages.len(), 2);
        assert!(matches!(state.messages[0], Message::User(_)));
        assert!(matches!(state.messages[1], Message::Assistant(_)));
        assert!(
            text_from_message(&state.messages[1])
                .to_uppercase()
                .contains("HELLO WORLD")
        );
        registration.unregister();
    }

    #[tokio::test]
    async fn continues_and_processes_tool_results() {
        let registration = register_faux_provider(None);
        let model = registration.get_model();
        registration.set_responses([faux_assistant_message("The answer is 8.", None)]);
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                system_prompt:
                    "You are a helpful assistant. After getting a calculation result, state the answer clearly."
                        .to_string(),
                model: model.clone(),
                tools: vec![Arc::new(CalculateTool)],
                messages: vec![
                    user_message("What is 5 + 3?", Vec::new()),
                    Message::Assistant({
                        let mut assistant = assistant_text_message(&model, "Let me calculate that.");
                        assistant.stop_reason = StopReason::ToolUse;
                        assistant.content.push(faux_tool_call(
                            "calculate",
                            json!({ "expression": "5 + 3" }),
                            Some("calc-1".to_string()),
                        ));
                        assistant
                    }),
                    Message::ToolResult(ToolResultMessage {
                        tool_call_id: "calc-1".to_string(),
                        tool_name: "calculate".to_string(),
                        content: vec![ToolResultContent::text("5 + 3 = 8")],
                        details: None,
                        is_error: false,
                        timestamp: crate::utils::time::now_millis(),
                    }),
                ],
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });

        agent.continue_run().await.expect("continue succeeds");

        let state = agent.state().await;
        assert!(!state.is_streaming);
        assert!(state.messages.len() >= 4);
        let Some(last_message) = state.messages.last() else {
            panic!("expected last message");
        };
        assert!(matches!(last_message, Message::Assistant(_)));
        assert!(text_from_message(last_message).contains('8'));
        registration.unregister();
    }

    #[tokio::test]
    async fn prompt_text_preserves_chat_image_input() {
        let registration = register_faux_provider(None);
        registration.set_responses([faux_assistant_message("seen", None)]);
        let agent = Agent::new(AgentOptions {
            initial_state: AgentState {
                model: registration.get_model(),
                ..AgentState::default()
            },
            ..AgentOptions::default()
        });

        agent
            .prompt_text(
                "Describe this",
                vec![ImageContent {
                    data: "abc".to_string(),
                    mime_type: "image/png".to_string(),
                }],
            )
            .await
            .expect("prompt succeeds");

        let state = agent.state().await;
        let Message::User(user) = &state.messages[0] else {
            panic!("expected user message");
        };
        let crate::UserMessageContent::Parts(parts) = &user.content else {
            panic!("expected multipart user content");
        };
        assert!(
            parts
                .iter()
                .any(|part| matches!(part, crate::UserContent::Image(_)))
        );
        registration.unregister();
    }
}
