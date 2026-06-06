use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::{
    AssistantContent, AssistantEventStream, AssistantMessage, AssistantMessageEvent, Context,
    ImageContent, Message, Model, SimpleStreamOptions, TextContent, Tool, ToolResultContent,
    ToolResultMessage,
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
        ) -> Pin<Box<dyn Future<Output = crate::Result<AssistantEventStream>> + Send>>
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
type AgentToolExecuteFn = Arc<
    dyn Fn(
            String,
            Value,
            Option<CancellationToken>,
            Option<AgentToolUpdateCallback>,
        ) -> Pin<Box<dyn Future<Output = AgentResult<AgentToolResult>> + Send>>
        + Send
        + Sync,
>;
type AgentToolPrepareArgumentsFn = Arc<dyn Fn(Value) -> AgentResult<Value> + Send + Sync>;

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

pub struct AgentToolBuilder {
    name: String,
    description: Option<String>,
    parameters: Option<Value>,
    label: Option<String>,
    execution_mode: Option<ToolExecutionMode>,
    prepare_arguments: Option<AgentToolPrepareArgumentsFn>,
    execute: Option<AgentToolExecuteFn>,
}

impl AgentToolBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: None,
            label: None,
            execution_mode: None,
            prepare_arguments: None,
            execute: None,
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn parameters(mut self, parameters: Value) -> Self {
        self.parameters = Some(parameters);
        self
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn execution_mode(mut self, execution_mode: ToolExecutionMode) -> Self {
        self.execution_mode = Some(execution_mode);
        self
    }

    pub fn prepare_arguments<F>(mut self, prepare_arguments: F) -> Self
    where
        F: Fn(Value) -> AgentResult<Value> + Send + Sync + 'static,
    {
        self.prepare_arguments = Some(Arc::new(prepare_arguments));
        self
    }

    pub fn execute<F, Fut>(mut self, execute: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = AgentResult<AgentToolResult>> + Send + 'static,
    {
        self.execute = Some(Arc::new(
            move |_tool_call_id, args, _cancellation_token, _on_update| Box::pin(execute(args)),
        ));
        self
    }

    pub fn execute_with_context<F, Fut>(mut self, execute: F) -> Self
    where
        F: Fn(String, Value, Option<CancellationToken>, Option<AgentToolUpdateCallback>) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = AgentResult<AgentToolResult>> + Send + 'static,
    {
        self.execute = Some(Arc::new(
            move |tool_call_id, args, cancellation_token, on_update| {
                Box::pin(execute(tool_call_id, args, cancellation_token, on_update))
            },
        ));
        self
    }

    pub fn build(self) -> crate::Result<DynAgentTool> {
        let mut tool_builder = Tool::builder(self.name);
        if let Some(description) = self.description {
            tool_builder = tool_builder.description(description);
        }
        if let Some(parameters) = self.parameters {
            tool_builder = tool_builder.parameters(parameters);
        }
        let definition = tool_builder.build()?;
        let label = self.label.unwrap_or_else(|| definition.name.clone());
        let execute = self.execute.ok_or_else(|| {
            crate::Error::Validation("agent tool execute callback must be set".to_string())
        })?;

        Ok(Arc::new(ClosureAgentTool {
            definition,
            label,
            execution_mode: self.execution_mode,
            prepare_arguments: self.prepare_arguments,
            execute,
        }))
    }
}

struct ClosureAgentTool {
    definition: Tool,
    label: String,
    execution_mode: Option<ToolExecutionMode>,
    prepare_arguments: Option<AgentToolPrepareArgumentsFn>,
    execute: AgentToolExecuteFn,
}

#[async_trait]
impl AgentTool for ClosureAgentTool {
    fn definition(&self) -> Tool {
        self.definition.clone()
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.execution_mode
    }

    fn prepare_arguments(&self, args: Value) -> AgentResult<Value> {
        if let Some(prepare_arguments) = &self.prepare_arguments {
            prepare_arguments(args)
        } else {
            Ok(args)
        }
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        args: Value,
        cancellation_token: Option<CancellationToken>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> AgentResult<AgentToolResult> {
        (self.execute)(
            tool_call_id.to_string(),
            args,
            cancellation_token,
            on_update,
        )
        .await
    }
}

#[derive(Clone, Default)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<DynAgentTool>,
}

impl AgentContext {
    pub fn builder() -> AgentContextBuilder {
        AgentContextBuilder::default()
    }

    pub fn llm_context(&self) -> Context {
        Context {
            system_prompt: Some(self.system_prompt.clone()),
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

#[derive(Clone, Default)]
pub struct AgentContextBuilder {
    context: AgentContext,
}

impl AgentContextBuilder {
    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.context.system_prompt = system_prompt.into();
        self
    }

    pub fn message(mut self, message: AgentMessage) -> Self {
        self.context.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = AgentMessage>) -> Self {
        self.context.messages.extend(messages);
        self
    }

    pub fn tool(mut self, tool: DynAgentTool) -> Self {
        self.context.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = DynAgentTool>) -> Self {
        self.context.tools.extend(tools);
        self
    }

    pub fn build(self) -> AgentContext {
        self.context
    }
}

pub type ConvertToLlmFn = Arc<
    dyn Fn(Vec<AgentMessage>) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send>> + Send + Sync,
>;

pub(crate) fn default_convert_to_llm() -> ConvertToLlmFn {
    Arc::new(|messages| {
        Box::pin(async move {
            messages
                .into_iter()
                .filter(|message| message.is_llm_message())
                .collect()
        })
    })
}

pub type TransformContextFn = Arc<
    dyn Fn(
            Vec<AgentMessage>,
            Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>
        + Send
        + Sync,
>;
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
    pub convert_to_llm: ConvertToLlmFn,
    pub transform_context: Option<TransformContextFn>,
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
            convert_to_llm: default_convert_to_llm(),
            transform_context: None,
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
    pub thinking_level: Option<crate::ModelThinkingLevel>,
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    struct TestTool;

    #[async_trait]
    impl AgentTool for TestTool {
        fn definition(&self) -> Tool {
            Tool::builder("echo")
                .description("Echo a value.")
                .parameters(json!({
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    }
                }))
                .build()
                .expect("tool")
        }

        fn label(&self) -> &str {
            "Echo"
        }

        async fn execute(
            &self,
            _tool_call_id: &str,
            _args: Value,
            _cancellation_token: Option<CancellationToken>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> AgentResult<AgentToolResult> {
            Ok(AgentToolResult::text("ok"))
        }
    }

    #[test]
    fn agent_context_builder_collects_messages_and_tools() {
        let tool: DynAgentTool = Arc::new(TestTool);
        let context = AgentContext::builder()
            .system_prompt("You are concise.")
            .message(Message::user_text("Hello"))
            .tool(tool)
            .build();

        assert_eq!(context.system_prompt, "You are concise.");
        assert_eq!(context.messages.len(), 1);
        assert_eq!(context.tools.len(), 1);

        let llm_context = context.llm_context();
        assert_eq!(
            llm_context.system_prompt.as_deref(),
            Some("You are concise.")
        );
        assert_eq!(llm_context.messages.len(), 1);
        assert_eq!(llm_context.tools.len(), 1);
        assert_eq!(llm_context.tools[0].name, "echo");
    }

    #[tokio::test]
    async fn agent_tool_builder_creates_closure_tool() {
        let tool = AgentToolBuilder::new("weather")
            .description("Get weather.")
            .parameters(json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                }
            }))
            .label("Weather")
            .execute(|args| async move {
                let city = args
                    .get("city")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                Ok(AgentToolResult::text(format!("clear in {city}")))
            })
            .build()
            .expect("tool");

        let definition = tool.definition();
        assert_eq!(definition.name, "weather");
        assert_eq!(tool.label(), "Weather");

        let result = tool
            .execute("tool-1", json!({ "city": "Seattle" }), None, None)
            .await
            .expect("tool result");
        assert_eq!(
            result.content,
            vec![ToolResultContent::text("clear in Seattle")]
        );
    }

    #[tokio::test]
    async fn agent_tool_builder_supports_context_and_prepare_arguments() {
        let tool = AgentToolBuilder::new("echo")
            .description("Echo a prepared value.")
            .execution_mode(ToolExecutionMode::Sequential)
            .prepare_arguments(|mut args| {
                args["value"] = json!("prepared");
                Ok(args)
            })
            .execute_with_context(
                |tool_call_id, args, _cancellation_token, on_update| async move {
                    if let Some(on_update) = on_update {
                        on_update(AgentToolResult::text("working")).await;
                    }
                    Ok(AgentToolResult::text(format!(
                        "{tool_call_id}: {}",
                        args["value"].as_str().unwrap_or_default()
                    )))
                },
            )
            .build()
            .expect("tool");

        assert_eq!(tool.execution_mode(), Some(ToolExecutionMode::Sequential));

        let prepared = tool
            .prepare_arguments(json!({ "value": "raw" }))
            .expect("prepared args");
        assert_eq!(prepared["value"], "prepared");

        let updates = Arc::new(Mutex::new(Vec::new()));
        let on_update: AgentToolUpdateCallback = Arc::new({
            let updates = Arc::clone(&updates);
            move |result| {
                let updates = Arc::clone(&updates);
                Box::pin(async move {
                    updates.lock().expect("updates lock poisoned").push(result);
                })
            }
        });
        let result = tool
            .execute("tool-1", prepared, None, Some(on_update))
            .await
            .expect("tool result");

        assert_eq!(
            result.content,
            vec![ToolResultContent::text("tool-1: prepared")]
        );
        assert_eq!(
            updates.lock().expect("updates lock poisoned")[0].content,
            vec![ToolResultContent::text("working")]
        );
    }

    #[test]
    fn agent_tool_builder_requires_execute_callback() {
        let result = AgentToolBuilder::new("missing")
            .description("Missing execute.")
            .build();
        let error = match result {
            Ok(_) => panic!("expected missing execute callback error"),
            Err(error) => error,
        };

        assert_eq!(error.to_string(), "agent tool execute callback must be set");
    }
}
