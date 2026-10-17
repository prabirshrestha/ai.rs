//! Port of `packages/agent/src/types.ts`.
//!
//! Pi's callbacks become `Arc<dyn Fn(..) -> BoxFuture<..>>` aliases; an
//! `AbortSignal` is a [`CancellationToken`]. Pi's `AgentMessage` is
//! `Message | CustomAgentMessages[...]`; the Rust [`Message`] enum has no
//! custom roles, so [`AgentMessage`] is [`Message`].

use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::agent::error::AgentResult;
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    BoxFuture, ImageContent, Message, Model, ModelThinkingLevel, SimpleStreamOptions, Tool,
    ToolCall, ToolResultContent, ToolResultMessage, TranscriptContext, Usage, UserContent,
    UserMessage, UserMessageContent,
};

/// Stream function used by the agent loop. `Models::stream_simple` (through
/// [`stream_simple_fn`](crate::agent::stream_simple_fn)) satisfies this shape.
///
/// The loop passes a normalized transcript: the system prompt and tool
/// declarations are carried by the transcript's system messages.
///
/// Contract: it cannot fail. Failures must be encoded in the returned stream
/// via protocol events and a final [`AssistantMessage`] with stop reason
/// `Error` or `Aborted` and an error message.
pub type StreamFn = Arc<
    dyn Fn(Model, TranscriptContext, SimpleStreamOptions) -> BoxFuture<AssistantMessageEventStream>
        + Send
        + Sync,
>;

/// Configuration for how tool calls from a single assistant message are executed.
///
/// - `Sequential`: each tool call is prepared, executed, and finalized before the next one starts.
/// - `Parallel`: tool calls are prepared sequentially, then allowed tools execute concurrently.
///   `ToolExecutionEnd` is emitted in tool completion order after each tool is finalized,
///   while tool-result message artifacts are emitted later in assistant source order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExecutionMode {
    Sequential,
    Parallel,
}

/// Controls how many queued user messages are injected when the agent loop
/// reaches a queue drain point.
///
/// - `All`: drain and inject every queued message at that point.
/// - `OneAtATime`: drain and inject only the oldest queued message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMode {
    All,
    OneAtATime,
}

/// A single tool call content block emitted by an assistant message.
pub type AgentToolCall = ToolCall;

/// Result returned from `before_tool_call`.
///
/// `block: true` prevents the tool from executing. The loop emits an error
/// tool result instead. `reason` becomes the text shown in that error result.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
    /// Hint that the agent should stop after the current tool batch when this
    /// call is blocked. Early termination only happens when every finalized
    /// tool result in the batch sets this to true.
    pub terminate: bool,
}

/// Partial override returned from `after_tool_call`.
///
/// Merge semantics are field-by-field:
/// - `content`: if provided, replaces the tool result content array in full
/// - `details`: if provided, replaces the tool result details value in full
/// - `is_error`: if provided, replaces the tool result error flag
/// - `usage`: if provided, replaces the tool result usage
/// - `terminate`: if provided, replaces the early-termination hint
/// - `structured_content`: if provided, replaces the structured content. If
///   `content` is provided without it, the structured content is dropped,
///   because it may no longer match the content.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<ToolResultContent>>,
    pub details: Option<Value>,
    pub structured_content: Option<Value>,
    pub is_error: Option<bool>,
    /// Usage from the final tool execution itself, if available.
    pub usage: Option<Usage>,
    pub terminate: Option<bool>,
}

/// Context passed to `before_tool_call`.
///
/// Divergence: `context` is a clone of the loop's context; Pi passes the
/// live object, so in-place edits made here would reach the loop in Pi.
#[derive(Clone)]
pub struct BeforeToolCallContext {
    /// The assistant message that requested the tool call.
    pub assistant_message: AssistantMessage,
    /// The raw tool call block from `assistant_message.content`.
    pub tool_call: AgentToolCall,
    /// Validated tool arguments for the target tool schema. Pi hands the hook
    /// the same object the tool later receives, so mutations made here are
    /// executed without revalidation; the shared cell reproduces that.
    pub args: Arc<Mutex<Value>>,
    /// Current agent context at the time the tool call is prepared.
    pub context: AgentContext,
}

/// Context passed to `after_tool_call`.
///
/// Divergence: `context` is a clone of the loop's context; Pi passes the
/// live object, so in-place edits made here would reach the loop in Pi.
#[derive(Clone)]
pub struct AfterToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: AgentToolCall,
    /// Validated tool arguments for the target tool schema.
    pub args: Value,
    /// The executed tool result before any `after_tool_call` overrides are applied.
    pub result: AgentToolResult,
    /// Whether the executed tool result is currently treated as an error.
    pub is_error: bool,
    pub context: AgentContext,
}

/// Context passed to completed-turn callbacks.
///
/// Divergence: Pi passes the loop's live `currentContext` (and the live
/// `newMessages` array), so a hook that mutates `context.messages` in place
/// changes what the loop sends next. Rust hands the hook a clone; in-place
/// edits are lost. Return an update (`AgentRequestUpdate.context`,
/// `AgentLoopTurnUpdate.context`) to replace the loop's context instead.
#[derive(Clone)]
pub struct AgentTurnContext {
    /// The assistant message that completed the turn.
    pub message: AssistantMessage,
    /// Tool result messages emitted for the completed turn.
    pub tool_results: Vec<ToolResultMessage>,
    /// Current agent context after the turn's assistant message and tool
    /// results have been appended.
    pub context: AgentContext,
    /// Messages that this loop invocation will return if it exits at this
    /// point. Prompt runs include the initial prompt messages; continuation
    /// runs do not include pre-existing context messages.
    pub new_messages: Vec<AgentMessage>,
}

/// Decision returned by [`FinishTurnFn`]. Returning `None` preserves normal scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentTurnDecision {
    Continue,
    End,
}

/// Called after a completed assistant turn and all of its tool-result
/// messages, but before `turn_end`. On a normal turn, `Continue` ensures one
/// next provider request. Tool-result, steering, or follow-up scheduling can
/// satisfy that request and adds no extra request; otherwise the loop
/// continues once with the current context. Error and aborted responses
/// remain hard exits.
///
/// An `Err` (Pi: a thrown error) rejects the loop; `Agent` reports it with
/// its failure lifecycle events.
pub type FinishTurnFn = Arc<
    dyn Fn(
            AgentTurnContext,
            Option<CancellationToken>,
        ) -> BoxFuture<AgentResult<Option<AgentTurnDecision>>>
        + Send
        + Sync,
>;

/// Replacement runtime state used by the agent loop before starting another provider request.
#[derive(Clone, Default)]
pub struct AgentLoopTurnUpdate {
    /// Context for the next provider request.
    pub context: Option<AgentContext>,
    /// Messages to append before the next provider request, with normal lifecycle events.
    pub messages: Option<Vec<AgentMessage>>,
    /// Model for the next provider request.
    pub model: Option<Model>,
    /// Thinking level for the next provider request.
    pub thinking_level: Option<ModelThinkingLevel>,
}

/// Runtime state available immediately before a conversational provider request.
///
/// Divergence: Pi passes the loop's live `currentContext` (and the live
/// `newMessages` array), so a hook that mutates `context.messages` in place
/// changes what the loop sends next. Rust hands the hook a clone; in-place
/// edits are lost. Return an update (`AgentRequestUpdate.context`,
/// `AgentLoopTurnUpdate.context`) to replace the loop's context instead.
#[derive(Clone)]
pub struct PrepareRequestContext {
    pub context: AgentContext,
    pub model: Model,
    pub thinking_level: ModelThinkingLevel,
}

/// Replacement runtime state for the provider request being prepared.
#[derive(Clone, Default)]
pub struct AgentRequestUpdate {
    pub context: Option<AgentContext>,
    pub model: Option<Model>,
    pub thinking_level: Option<ModelThinkingLevel>,
}

/// Called immediately before every conversational provider request,
/// including the first. Pending messages have already been appended and
/// emitted when this callback runs.
///
/// An `Err` (Pi: a thrown error) rejects the loop.
pub type PrepareRequestFn = Arc<
    dyn Fn(
            PrepareRequestContext,
            Option<CancellationToken>,
        ) -> BoxFuture<AgentResult<Option<AgentRequestUpdate>>>
        + Send
        + Sync,
>;

pub type PrepareNextTurnContext = AgentTurnContext;

/// `AgentLoopConfig.prepareNextTurn`. An `Err` (Pi: a thrown error) rejects
/// the loop.
pub type PrepareNextTurnFn = Arc<
    dyn Fn(PrepareNextTurnContext) -> BoxFuture<AgentResult<Option<AgentLoopTurnUpdate>>>
        + Send
        + Sync,
>;

/// `convertToLlm`. Must not fail; return a safe fallback value instead.
pub type ConvertToLlmFn = Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<Vec<Message>> + Send + Sync>;

/// `transformContext`. Must not fail; return the original messages instead.
pub type TransformContextFn = Arc<
    dyn Fn(Vec<AgentMessage>, Option<CancellationToken>) -> BoxFuture<Vec<AgentMessage>>
        + Send
        + Sync,
>;

/// `getApiKey`: resolves an API key dynamically for each LLM call.
pub type GetApiKeyFn = Arc<dyn Fn(String) -> BoxFuture<Option<String>> + Send + Sync>;

/// `getSteeringMessages` / `getFollowUpMessages`.
pub type MessageQueueFn = Arc<dyn Fn() -> BoxFuture<Vec<AgentMessage>> + Send + Sync>;

/// `beforeToolCall`. An `Err` (Pi: a thrown error) becomes an error tool result.
pub type BeforeToolCallFn = Arc<
    dyn Fn(
            BeforeToolCallContext,
            Option<CancellationToken>,
        ) -> BoxFuture<AgentResult<Option<BeforeToolCallResult>>>
        + Send
        + Sync,
>;

/// `afterToolCall`. An `Err` (Pi: a thrown error) becomes an error tool result.
pub type AfterToolCallFn = Arc<
    dyn Fn(
            AfterToolCallContext,
            Option<CancellationToken>,
        ) -> BoxFuture<AgentResult<Option<AfterToolCallResult>>>
        + Send
        + Sync,
>;

/// Configuration of the low-level agent loop (`AgentLoopConfig extends
/// SimpleStreamOptions`). The inherited stream options live in `options`.
#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: Model,
    /// The `SimpleStreamOptions` part of Pi's config. `options.reasoning` is
    /// the requested thinking level (`None` is `off`).
    pub options: SimpleStreamOptions,
    /// Converts agent messages to LLM-compatible messages before each LLM call.
    pub convert_to_llm: ConvertToLlmFn,
    /// Optional transform applied to the context before `convert_to_llm`.
    pub transform_context: Option<TransformContextFn>,
    /// Resolves an API key dynamically for each LLM call.
    pub get_api_key: Option<GetApiKeyFn>,
    /// Called after the assistant message and all tool-result messages have
    /// been emitted, immediately before `turn_end`.
    pub finish_turn: Option<FinishTurnFn>,
    /// Called immediately before every conversational provider request,
    /// including the first.
    pub prepare_request: Option<PrepareRequestFn>,
    /// Called after `turn_end` when the loop will continue, immediately
    /// before the next turn starts.
    pub prepare_next_turn: Option<PrepareNextTurnFn>,
    /// Returns steering messages to inject into the conversation mid-run.
    pub get_steering_messages: Option<MessageQueueFn>,
    /// Returns follow-up messages to process after the agent would otherwise stop.
    pub get_follow_up_messages: Option<MessageQueueFn>,
    /// Tool execution mode. Default: `Parallel`.
    pub tool_execution: ToolExecutionMode,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
}

impl AgentLoopConfig {
    /// A config with Pi's defaults and `convert_to_llm` passing standard
    /// messages through.
    pub fn new(model: Model) -> Self {
        Self {
            model,
            options: SimpleStreamOptions::default(),
            convert_to_llm: default_convert_to_llm(),
            transform_context: None,
            get_api_key: None,
            finish_turn: None,
            prepare_request: None,
            prepare_next_turn: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
            tool_execution: ToolExecutionMode::Parallel,
            before_tool_call: None,
            after_tool_call: None,
        }
    }
}

/// Pi's `defaultConvertToLlm`: keep system, user, assistant and tool-result
/// messages. Every Rust [`Message`] is one of those.
pub fn default_convert_to_llm() -> ConvertToLlmFn {
    Arc::new(|messages| Box::pin(async move { messages }))
}

/// Agent messages: LLM messages (custom roles are not representable in Rust).
pub type AgentMessage = Message;

/// Public agent state (a snapshot returned by `Agent::state()`).
#[derive(Clone)]
pub struct AgentState {
    /// Current system prompt, replayed from the transcript's system messages.
    pub system_prompt: String,
    /// Active model used for future turns.
    pub model: Model,
    /// Requested reasoning level for future turns.
    pub thinking_level: ModelThinkingLevel,
    /// Executable tools.
    pub tools: Vec<DynAgentTool>,
    /// Conversation transcript. System messages carry the prompt and tool declarations.
    pub messages: Vec<AgentMessage>,
    /// True while the agent is processing a prompt or continuation, until
    /// awaited `agent_end` listeners settle.
    pub is_streaming: bool,
    /// Partial assistant message for the current streamed response, if any.
    pub streaming_message: Option<AgentMessage>,
    /// Tool call ids currently executing.
    pub pending_tool_calls: std::collections::HashSet<String>,
    /// Error message from the most recent failed or aborted assistant turn.
    pub error_message: Option<String>,
}

/// Final or partial result produced by a tool.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentToolResult {
    /// Text or image content returned to the model.
    pub content: Vec<ToolResultContent>,
    /// Arbitrary structured details for logs or UI rendering.
    pub details: Option<Value>,
    /// Machine-readable result matching the tool's `output_schema`, for
    /// programmatic callers. Not sent to the model.
    pub structured_content: Option<Value>,
    /// Usage from the final tool execution itself, if available.
    pub usage: Option<Usage>,
    /// Report a failure without returning `Err`. The model sees `content` as
    /// an error result, but `details` and `structured_content` are kept.
    pub is_error: bool,
    /// Hint that the agent should stop after the current tool batch. Early
    /// termination only happens when every finalized tool result in the
    /// batch sets this to true.
    pub terminate: bool,
}

impl AgentToolResult {
    /// A result with one text block (Rust convenience).
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::text(text)],
            ..Default::default()
        }
    }
}

/// Final outcome of a tool call after hooks ran.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentToolCallOutcome {
    pub tool_call: AgentToolCall,
    pub result: AgentToolResult,
    pub is_error: bool,
}

/// Callback used by tools to stream partial execution updates. Calls made
/// after the tool's `execute` settles are ignored. The returned future
/// resolves once the update has been delivered (or immediately when ignored).
pub type AgentToolUpdateCallback = Arc<dyn Fn(AgentToolResult) -> BoxFuture<()> + Send + Sync>;

/// `replay` recovery policy for an effect whose durable intent exists but
/// whose outcome is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolReplay {
    Never,
    Safe,
}

/// Tool definition used by the agent runtime (`AgentTool extends Tool`).
#[async_trait]
pub trait AgentTool: Send + Sync {
    /// The model-facing declaration (`name`, `description`, `parameters`).
    fn definition(&self) -> Tool;
    /// Human-readable label for UI display.
    fn label(&self) -> &str;
    /// Per-tool execution mode override.
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }
    /// JSON Schema of `structured_content` in successful results.
    fn output_schema(&self) -> Option<Value> {
        None
    }
    /// Recovery policy for an effect whose outcome is unknown.
    fn replay(&self) -> Option<ToolReplay> {
        None
    }
    /// Compatibility shim for raw tool-call arguments before schema validation.
    fn prepare_arguments(&self, args: Value) -> AgentResult<Value> {
        Ok(args)
    }
    /// Execute the tool call. Return `Err` on failure, or a result with
    /// `is_error: true`; do not only describe the failure in `content`.
    async fn execute(
        &self,
        tool_call_id: &str,
        args: Value,
        signal: Option<CancellationToken>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> AgentResult<AgentToolResult>;
}

pub type DynAgentTool = Arc<dyn AgentTool>;

type AgentToolExecuteFn = Arc<
    dyn Fn(
            String,
            Value,
            Option<CancellationToken>,
            Option<AgentToolUpdateCallback>,
        ) -> BoxFuture<AgentResult<AgentToolResult>>
        + Send
        + Sync,
>;
type AgentToolPrepareArgumentsFn = Arc<dyn Fn(Value) -> AgentResult<Value> + Send + Sync>;

/// Rust convenience builder for closure-backed [`AgentTool`]s.
pub struct AgentToolBuilder {
    name: String,
    description: Option<String>,
    parameters: Option<Value>,
    label: Option<String>,
    execution_mode: Option<ToolExecutionMode>,
    output_schema: Option<Value>,
    replay: Option<ToolReplay>,
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
            output_schema: None,
            replay: None,
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

    pub fn output_schema(mut self, output_schema: Value) -> Self {
        self.output_schema = Some(output_schema);
        self
    }

    pub fn replay(mut self, replay: ToolReplay) -> Self {
        self.replay = Some(replay);
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
        self.execute = Some(Arc::new(move |_tool_call_id, args, _signal, _on_update| {
            Box::pin(execute(args))
        }));
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
        self.execute = Some(Arc::new(move |tool_call_id, args, signal, on_update| {
            Box::pin(execute(tool_call_id, args, signal, on_update))
        }));
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
            output_schema: self.output_schema,
            replay: self.replay,
            prepare_arguments: self.prepare_arguments,
            execute,
        }))
    }
}

struct ClosureAgentTool {
    definition: Tool,
    label: String,
    execution_mode: Option<ToolExecutionMode>,
    output_schema: Option<Value>,
    replay: Option<ToolReplay>,
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

    fn output_schema(&self) -> Option<Value> {
        self.output_schema.clone()
    }

    fn replay(&self) -> Option<ToolReplay> {
        self.replay
    }

    fn prepare_arguments(&self, args: Value) -> AgentResult<Value> {
        match &self.prepare_arguments {
            Some(prepare_arguments) => prepare_arguments(args),
            None => Ok(args),
        }
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        args: Value,
        signal: Option<CancellationToken>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> AgentResult<AgentToolResult> {
        (self.execute)(tool_call_id.to_string(), args, signal, on_update).await
    }
}

/// Context snapshot passed into the low-level agent loop.
#[derive(Clone, Default)]
pub struct AgentContext {
    /// Transcript visible to the model.
    pub messages: Vec<AgentMessage>,
    /// Tools available for execution in this run.
    pub tools: Vec<DynAgentTool>,
}

impl AgentContext {
    pub fn builder() -> AgentContextBuilder {
        AgentContextBuilder::default()
    }
}

/// Rust convenience builder for [`AgentContext`].
#[derive(Clone, Default)]
pub struct AgentContextBuilder {
    context: AgentContext,
}

impl AgentContextBuilder {
    pub fn message(mut self, message: impl Into<AgentMessage>) -> Self {
        self.context.messages.push(message.into());
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

/// Events emitted by the agent for UI updates.
///
/// `AgentEnd` is the last event emitted for a run, but awaited
/// `Agent::subscribe()` listeners for that event are still part of run
/// settlement.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    /// A turn is one assistant response plus any tool calls/results.
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    /// Emitted for system, user, assistant, and tool-result messages.
    MessageStart {
        message: AgentMessage,
    },
    /// Only emitted for assistant messages during streaming.
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

impl AgentEvent {
    /// The Pi event `type` string.
    pub const fn event_type(&self) -> &'static str {
        match self {
            Self::AgentStart => "agent_start",
            Self::AgentEnd { .. } => "agent_end",
            Self::TurnStart => "turn_start",
            Self::TurnEnd { .. } => "turn_end",
            Self::MessageStart { .. } => "message_start",
            Self::MessageUpdate { .. } => "message_update",
            Self::MessageEnd { .. } => "message_end",
            Self::ToolExecutionStart { .. } => "tool_execution_start",
            Self::ToolExecutionUpdate { .. } => "tool_execution_update",
            Self::ToolExecutionEnd { .. } => "tool_execution_end",
        }
    }
}

/// `AgentEventSink`. An `Err` (Pi: a rejected listener) aborts the loop.
pub type AgentEventSink = Arc<dyn Fn(AgentEvent) -> BoxFuture<AgentResult<()>> + Send + Sync>;

/// Listener registered with `Agent::subscribe`.
pub type AgentEventListener =
    Arc<dyn Fn(AgentEvent, CancellationToken) -> BoxFuture<AgentResult<()>> + Send + Sync>;

/// The tool calls of an assistant message, in source order.
pub fn assistant_tool_calls(message: &AssistantMessage) -> Vec<AgentToolCall> {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .collect()
}

/// A user message with text and optional images (`Agent.prompt(text, images)`).
pub fn user_message(text: impl Into<String>, images: Vec<ImageContent>) -> AgentMessage {
    let mut content = vec![UserContent::text(text)];
    content.extend(images.into_iter().map(UserContent::Image));
    Message::User(UserMessage {
        content: UserMessageContent::Parts(content),
        timestamp: crate::utils::time::now_millis(),
    })
}
