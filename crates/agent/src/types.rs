use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, ImageContent, Message, Model, SimpleStreamOptions, TextContent, Tool,
    ToolResultContent, ToolResultMessage,
};
use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::Result;

pub type AgentMessage = Message;

pub type StreamFn = Arc<
    dyn Fn(
            Model,
            Context,
            SimpleStreamOptions,
        ) -> Pin<Box<dyn Future<Output = ai::Result<AssistantMessageEventStream>> + Send>>
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
    fn prepare_arguments(&self, args: Value) -> Result<Value> {
        Ok(args)
    }
    async fn execute(
        &self,
        tool_call_id: &str,
        args: Value,
        cancellation_token: Option<CancellationToken>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult>;
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
            messages: self.messages.clone(),
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

#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: Model,
    pub options: SimpleStreamOptions,
    pub convert_to_llm: Option<ConvertToLlmFn>,
    pub transform_context: Option<TransformContextFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    pub get_steering_messages: Option<MessageQueueFn>,
    pub get_follow_up_messages: Option<MessageQueueFn>,
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
            get_steering_messages: None,
            get_follow_up_messages: None,
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

#[derive(Debug, Clone)]
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
    Arc<dyn Fn(AgentEvent) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync>;

pub fn assistant_tool_calls(message: &AssistantMessage) -> Vec<ai::ToolCall> {
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
        timestamp: ai::utils::time::now_millis(),
    }
}

pub fn user_message(text: impl Into<String>, images: Vec<ImageContent>) -> AgentMessage {
    if images.is_empty() {
        Message::user_text(text)
    } else {
        let mut content = vec![ai::UserContent::text(text)];
        content.extend(images.into_iter().map(ai::UserContent::Image));
        Message::User(ai::UserMessage {
            content: ai::UserMessageContent::Parts(content),
            timestamp: ai::utils::time::now_millis(),
        })
    }
}
