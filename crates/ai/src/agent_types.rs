use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, ImageContent, Message, Model, SimpleStreamOptions, TextContent, Tool,
    ToolResultContent, ToolResultMessage,
};
use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::AgentResult;

pub type AgentMessage = Message;
pub type AgentToolCall = crate::ToolCall;

pub type StreamFn = Arc<
    dyn Fn(
            Model,
            Context,
            SimpleStreamOptions,
        )
            -> Pin<Box<dyn Future<Output = crate::Result<AssistantMessageEventStream>> + Send>>
        + Send
        + Sync,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExecutionMode {
    Sequential,
    Parallel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMode {
    All,
    OneAtATime,
}

#[derive(Debug, Clone)]
pub struct AgentToolResult {
    pub content: Vec<ToolResultContent>,
    pub details: Option<Value>,
    pub terminate: bool,
}

impl AgentToolResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::text(text)],
            details: None,
            terminate: false,
        }
    }
}

pub type AgentToolUpdateCallback =
    Arc<dyn Fn(AgentToolResult) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

#[async_trait]
pub trait AgentTool: Send + Sync {
    fn definition(&self) -> Tool;
    fn label(&self) -> &str;
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }
    fn prepare_arguments(&self, args: Value) -> AgentResult<Value> {
        Ok(args)
    }
    async fn execute(
        &self,
        tool_call_id: &str,
        args: Value,
        cancellation_token: Option<CancellationToken>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> AgentResult<AgentToolResult>;
}

pub type DynAgentTool = Arc<dyn AgentTool>;

#[derive(Clone)]
pub struct AgentContext {
    pub system_prompt: Option<String>,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<DynAgentTool>,
}

impl AgentContext {
    pub fn llm_context(&self) -> Context {
        Context {
            system_prompt: self.system_prompt.clone(),
            messages: self
                .messages
                .iter()
                .filter(|message| message.is_llm_message())
                .cloned()
                .collect(),
            tools: self.tools.iter().map(|tool| tool.definition()).collect(),
        }
    }
}

pub type ConvertToLlmFn = Arc<
    dyn Fn(Vec<AgentMessage>) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send>> + Send + Sync,
>;
pub type TransformContextFn = Arc<
    dyn Fn(
            Vec<AgentMessage>,
            Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>
        + Send
        + Sync,
>;
pub type GetApiKeyFn =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> + Send + Sync>;
pub type MessageQueueFn =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>> + Send + Sync>;
pub type ShouldStopAfterTurnFn = Arc<
    dyn Fn(ShouldStopAfterTurnContext) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync,
>;
pub type PrepareNextTurnFn = Arc<
    dyn Fn(
            PrepareNextTurnContext,
            Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = Option<AgentLoopTurnUpdate>> + Send>>
        + Send
        + Sync,
>;
pub type BeforeToolCallFn = Arc<
    dyn Fn(
            BeforeToolCallContext,
            Option<CancellationToken>,
        )
            -> Pin<Box<dyn Future<Output = AgentResult<Option<BeforeToolCallResult>>> + Send>>
        + Send
        + Sync,
>;
pub type AfterToolCallFn = Arc<
    dyn Fn(
            AfterToolCallContext,
            Option<CancellationToken>,
        )
            -> Pin<Box<dyn Future<Output = AgentResult<Option<AfterToolCallResult>>> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: Model,
    pub options: SimpleStreamOptions,
    pub convert_to_llm: Option<ConvertToLlmFn>,
    pub transform_context: Option<TransformContextFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    pub prepare_next_turn: Option<PrepareNextTurnFn>,
    pub get_steering_messages: Option<MessageQueueFn>,
    pub get_follow_up_messages: Option<MessageQueueFn>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub tool_execution: ToolExecutionMode,
}

impl AgentLoopConfig {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            options: SimpleStreamOptions::default(),
            convert_to_llm: None,
            transform_context: None,
            get_api_key: None,
            should_stop_after_turn: None,
            prepare_next_turn: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
            before_tool_call: None,
            after_tool_call: None,
            tool_execution: ToolExecutionMode::Parallel,
        }
    }
}

#[derive(Clone)]
pub struct ShouldStopAfterTurnContext {
    pub message: AssistantMessage,
    pub tool_results: Vec<ToolResultMessage>,
    pub context: AgentContext,
    pub new_messages: Vec<AgentMessage>,
}

pub type PrepareNextTurnContext = ShouldStopAfterTurnContext;

#[derive(Clone)]
pub struct AgentLoopTurnUpdate {
    pub context: Option<AgentContext>,
    pub model: Option<Model>,
    pub reasoning_level: Option<crate::ModelThinkingLevel>,
}

#[derive(Clone)]
pub struct BeforeToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: crate::ToolCall,
    pub args: Arc<Mutex<Value>>,
    pub context: AgentContext,
}

#[derive(Debug, Clone, Default)]
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
}

#[derive(Clone)]
pub struct AfterToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: crate::ToolCall,
    pub args: Value,
    pub result: AgentToolResult,
    pub is_error: bool,
    pub context: AgentContext,
}

#[derive(Debug, Clone, Default)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<ToolResultContent>>,
    pub details: Option<Value>,
    pub is_error: Option<bool>,
    pub terminate: Option<bool>,
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AgentMessage,
        assistant_message_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
    },
}

pub type AgentEventSink =
    Arc<dyn Fn(AgentEvent) -> Pin<Box<dyn Future<Output = AgentResult<()>> + Send>> + Send + Sync>;

pub type AgentEventListener = Arc<
    dyn Fn(AgentEvent, CancellationToken) -> Pin<Box<dyn Future<Output = AgentResult<()>> + Send>>
        + Send
        + Sync,
>;

pub fn assistant_tool_calls(message: &AssistantMessage) -> Vec<crate::ToolCall> {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .collect()
}

pub fn text_result_message(
    tool_call_id: String,
    tool_name: String,
    text: impl Into<String>,
    is_error: bool,
) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id,
        tool_name,
        content: vec![ToolResultContent::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })],
        details: None,
        is_error,
        timestamp: crate::utils::time::now_millis(),
    }
}

pub fn user_message(text: impl Into<String>, images: Vec<ImageContent>) -> AgentMessage {
    let mut content = vec![crate::UserContent::text(text)];
    content.extend(images.into_iter().map(crate::UserContent::Image));
    Message::User(crate::UserMessage {
        content: crate::UserMessageContent::Parts(content),
        timestamp: crate::utils::time::now_millis(),
    })
}
