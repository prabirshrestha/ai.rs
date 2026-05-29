use std::collections::HashSet;
use std::sync::Arc;

use ai::{
    AssistantContent, AssistantMessage, ImageContent, Message, Model, SimpleStreamOptions,
    StopReason, TextContent, Usage,
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{run_agent_loop, run_agent_loop_continue};
use crate::types::{
    AfterToolCallFn, AgentContext, AgentEvent, AgentEventSink, AgentLoopConfig, AgentMessage,
    BeforeToolCallFn, ConvertToLlmFn, DynAgentTool, GetApiKeyFn, PrepareNextTurnFn, QueueMode,
    ShouldStopAfterTurnFn, StreamFn, ToolExecutionMode, TransformContextFn, user_message,
};
use crate::{AgentError, Result};

#[derive(Clone)]
pub struct AgentState {
    pub system_prompt: Option<String>,
    pub model: Model,
    pub reasoning_level: Option<ai::ModelThinkingLevel>,
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
            system_prompt: None,
            model,
            reasoning_level: None,
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }
}

#[derive(Clone)]
pub struct AgentOptions {
    pub initial_state: AgentState,
    pub convert_to_llm: Option<ConvertToLlmFn>,
    pub transform_context: Option<TransformContextFn>,
    pub stream_fn: Option<StreamFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    pub prepare_next_turn: Option<PrepareNextTurnFn>,
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
            get_api_key: None,
            should_stop_after_turn: None,
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
    listeners: Arc<Mutex<Vec<AgentEventSink>>>,
    steering_queue: Arc<Mutex<PendingMessageQueue>>,
    follow_up_queue: Arc<Mutex<PendingMessageQueue>>,
    convert_to_llm: Option<ConvertToLlmFn>,
    transform_context: Option<TransformContextFn>,
    stream_fn: Option<StreamFn>,
    get_api_key: Option<GetApiKeyFn>,
    should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    prepare_next_turn: Option<PrepareNextTurnFn>,
    before_tool_call: Option<BeforeToolCallFn>,
    after_tool_call: Option<AfterToolCallFn>,
    session_id: Option<String>,
    base_options: SimpleStreamOptions,
    active_token: Arc<Mutex<Option<CancellationToken>>>,
    tool_execution: ToolExecutionMode,
}

impl Agent {
    pub fn new(options: AgentOptions) -> Self {
        Self {
            state: Arc::new(Mutex::new(options.initial_state)),
            listeners: Arc::new(Mutex::new(Vec::new())),
            steering_queue: Arc::new(Mutex::new(PendingMessageQueue::new(options.steering_mode))),
            follow_up_queue: Arc::new(Mutex::new(PendingMessageQueue::new(options.follow_up_mode))),
            convert_to_llm: options.convert_to_llm,
            transform_context: options.transform_context,
            stream_fn: options.stream_fn,
            get_api_key: options.get_api_key,
            should_stop_after_turn: options.should_stop_after_turn,
            prepare_next_turn: options.prepare_next_turn,
            before_tool_call: options.before_tool_call,
            after_tool_call: options.after_tool_call,
            session_id: options.session_id,
            base_options: options.options,
            active_token: Arc::new(Mutex::new(None)),
            tool_execution: options.tool_execution,
        }
    }

    pub async fn state(&self) -> AgentState {
        self.state.lock().await.clone()
    }

    pub async fn subscribe(&self, listener: AgentEventSink) {
        self.listeners.lock().await.push(listener);
    }

    pub async fn steer(&self, message: AgentMessage) {
        self.steering_queue.lock().await.enqueue(message);
    }

    pub async fn follow_up(&self, message: AgentMessage) {
        self.follow_up_queue.lock().await.enqueue(message);
    }

    pub async fn clear_all_queues(&self) {
        self.steering_queue.lock().await.clear();
        self.follow_up_queue.lock().await.clear();
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
    ) -> Result<()> {
        self.prompt_messages(vec![user_message(input, images)])
            .await
    }

    pub async fn prompt_messages(&self, messages: Vec<AgentMessage>) -> Result<()> {
        self.run_with_lifecycle(messages, false, false).await
    }

    pub async fn continue_run(&self) -> Result<()> {
        let last = self.state.lock().await.messages.last().cloned();
        match last {
            None => Err(AgentError::NoMessagesToContinue),
            Some(Message::Assistant(_)) => {
                let queued = self.steering_queue.lock().await.drain();
                if !queued.is_empty() {
                    self.run_with_lifecycle(queued, false, true).await
                } else {
                    let follow_up = self.follow_up_queue.lock().await.drain();
                    if follow_up.is_empty() {
                        Err(AgentError::CannotContinueFromAssistant)
                    } else {
                        self.run_with_lifecycle(follow_up, false, false).await
                    }
                }
            }
            Some(_) => self.run_with_lifecycle(Vec::new(), true, false).await,
        }
    }

    async fn run_with_lifecycle(
        &self,
        prompts: Vec<AgentMessage>,
        continue_existing: bool,
        skip_initial_steering_poll: bool,
    ) -> Result<()> {
        {
            let mut active = self.active_token.lock().await;
            if active.is_some() {
                return Err(AgentError::AlreadyProcessing);
            }
            *active = Some(CancellationToken::new());
        }
        {
            let mut state = self.state.lock().await;
            state.is_streaming = true;
            state.streaming_message = None;
            state.error_message = None;
        }

        let token = self.active_token.lock().await.clone();
        let result = if continue_existing {
            run_agent_loop_continue(
                self.create_context_snapshot().await,
                self.create_loop_config(skip_initial_steering_poll).await,
                self.event_sink(),
                token.clone(),
                self.stream_fn.clone(),
            )
            .await
        } else {
            run_agent_loop(
                prompts,
                self.create_context_snapshot().await,
                self.create_loop_config(skip_initial_steering_poll).await,
                self.event_sink(),
                token.clone(),
                self.stream_fn.clone(),
            )
            .await
        };

        let failure_result = if let Err(error) = result {
            let aborted = token.as_ref().is_some_and(CancellationToken::is_cancelled);
            self.emit_run_failure(error, aborted).await
        } else {
            Ok(())
        };

        let mut state = self.state.lock().await;
        state.is_streaming = false;
        state.streaming_message = None;
        state.pending_tool_calls.clear();
        drop(state);
        *self.active_token.lock().await = None;
        failure_result
    }

    async fn emit_run_failure(&self, error: AgentError, aborted: bool) -> Result<()> {
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
            timestamp: ai::utils::time::now_millis(),
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
        let state = self.state.lock().await;
        let mut options = self.base_options.clone();
        options.reasoning = state.reasoning_level;
        options.stream.session_id = self.session_id.clone();

        let steering_queue = self.steering_queue.clone();
        let follow_up_queue = self.follow_up_queue.clone();
        let skip_initial_steering_poll = Arc::new(Mutex::new(skip_initial_steering_poll));
        AgentLoopConfig {
            model: state.model.clone(),
            options,
            convert_to_llm: self.convert_to_llm.clone(),
            transform_context: self.transform_context.clone(),
            get_api_key: self.get_api_key.clone(),
            should_stop_after_turn: self.should_stop_after_turn.clone(),
            prepare_next_turn: self.prepare_next_turn.clone(),
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
            tool_execution: self.tool_execution,
        }
    }

    fn event_sink(&self) -> AgentEventSink {
        let state = self.state.clone();
        let listeners = self.listeners.clone();
        Arc::new(move |event| {
            let state = state.clone();
            let listeners = listeners.clone();
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
                            if let Message::Assistant(assistant) = message {
                                if let Some(error) = &assistant.error_message {
                                    state.error_message = Some(error.clone());
                                }
                            }
                        }
                        AgentEvent::AgentEnd { .. } => {
                            state.streaming_message = None;
                        }
                        _ => {}
                    }
                }
                for listener in listeners.lock().await.iter() {
                    listener(event.clone()).await?;
                }
                Ok(())
            })
        })
    }
}
