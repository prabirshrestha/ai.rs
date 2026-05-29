use std::collections::VecDeque;
use std::sync::Arc;

use agent::{
    Agent, AgentEvent, AgentEventSink, AgentOptions, AgentTool, AgentToolResult, StreamFn,
};
use ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, Message, Model, ModelCost, SimpleStreamOptions, StopReason, TextContent, Tool,
    ToolCall, ToolResultContent, Usage,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

fn create_model() -> Model {
    Model {
        id: "mock".to_string(),
        name: "mock".to_string(),
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        base_url: "https://example.invalid".to_string(),
        reasoning: false,
        input: vec![ai::ModelInput::Text],
        cost: ModelCost::default(),
        context_window: 8192,
        max_tokens: 2048,
        ..Model::default()
    }
}

fn assistant_message(content: Vec<AssistantContent>, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content,
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        model: "mock".to_string(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::default(),
        stop_reason,
        error_message: None,
        timestamp: ai::utils::time::now_millis(),
    }
}

fn assistant_text(text: &str) -> AssistantMessage {
    assistant_message(
        vec![AssistantContent::Text(TextContent {
            text: text.to_string(),
            text_signature: None,
        })],
        StopReason::Stop,
    )
}

fn assistant_tool_call(id: &str, name: &str, args: Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments: args,
        thought_signature: None,
    })
}

fn scripted_stream(messages: Vec<AssistantMessage>) -> StreamFn {
    let messages = Arc::new(Mutex::new(VecDeque::from(messages)));
    Arc::new(
        move |_model: Model, _context: Context, _options: SimpleStreamOptions| {
            let messages = messages.clone();
            Box::pin(async move {
                let message = messages
                    .lock()
                    .await
                    .pop_front()
                    .expect("scripted stream exhausted");
                let reason = message.stop_reason;
                let (mut sender, stream) = AssistantMessageEventStream::channel();
                sender.push(AssistantMessageEvent::Done { reason, message });
                Ok(stream)
            })
        },
    )
}

fn collecting_sink(events: Arc<Mutex<Vec<AgentEvent>>>) -> AgentEventSink {
    Arc::new(move |event| {
        let events = events.clone();
        Box::pin(async move {
            events.lock().await.push(event);
            Ok(())
        })
    })
}

fn event_type(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart => "agent_start",
        AgentEvent::AgentEnd { .. } => "agent_end",
        AgentEvent::TurnStart => "turn_start",
        AgentEvent::TurnEnd { .. } => "turn_end",
        AgentEvent::MessageStart { .. } => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd { .. } => "message_end",
        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
    }
}

fn message_role(message: &Message) -> &'static str {
    match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

struct EchoTool;

#[async_trait]
impl AgentTool for EchoTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "echo".to_string(),
            description: "Echo tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            }),
        }
    }

    fn label(&self) -> &str {
        "Echo"
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: Value,
        _cancellation_token: Option<CancellationToken>,
        _on_update: Option<agent::AgentToolUpdateCallback>,
    ) -> agent::Result<AgentToolResult> {
        let value = args
            .get("value")
            .and_then(Value::as_str)
            .expect("value argument");
        Ok(AgentToolResult {
            content: vec![ToolResultContent::text(format!("echoed: {value}"))],
            details: Some(json!({ "value": value })),
            terminate: false,
        })
    }
}

#[tokio::test]
async fn stateful_agent_emits_failure_turn_for_stream_errors() {
    let mut options = AgentOptions::new(create_model());
    options.stream_fn = Some(Arc::new(
        |_model: Model, _context: Context, _options: SimpleStreamOptions| {
            Box::pin(async { Err(ai::Error::Provider("provider exploded".to_string())) })
        },
    ));
    let agent = Agent::new(options);
    let events = Arc::new(Mutex::new(Vec::new()));
    agent.subscribe(collecting_sink(events.clone())).await;

    agent.prompt_text("hello", Vec::new()).await.unwrap();

    assert_eq!(
        events
            .lock()
            .await
            .iter()
            .map(event_type)
            .collect::<Vec<_>>(),
        [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end"
        ]
    );

    let state = agent.state().await;
    assert!(!state.is_streaming);
    assert_eq!(
        state.messages.iter().map(message_role).collect::<Vec<_>>(),
        ["user", "assistant"]
    );
    let Message::Assistant(last) = state.messages.last().expect("assistant failure message") else {
        panic!("expected assistant message");
    };
    assert_eq!(last.stop_reason, StopReason::Error);
    assert_eq!(
        last.error_message.as_deref(),
        Some("provider error: provider exploded")
    );
    assert_eq!(
        state.error_message.as_deref(),
        Some("provider error: provider exploded")
    );
}

#[tokio::test]
async fn stateful_agent_records_tool_result_messages() {
    let mut options = AgentOptions::new(create_model());
    options.initial_state.tools = vec![Arc::new(EchoTool)];
    options.stream_fn = Some(scripted_stream(vec![
        assistant_message(
            vec![assistant_tool_call(
                "tool-1",
                "echo",
                json!({ "value": "hello" }),
            )],
            StopReason::ToolUse,
        ),
        assistant_text("done"),
    ]));
    let agent = Agent::new(options);

    agent
        .prompt_text("echo something", Vec::new())
        .await
        .unwrap();

    let state = agent.state().await;
    assert_eq!(
        state.messages.iter().map(message_role).collect::<Vec<_>>(),
        ["user", "assistant", "toolResult", "assistant"]
    );
    assert!(state.pending_tool_calls.is_empty());
    assert!(state.streaming_message.is_none());
}
