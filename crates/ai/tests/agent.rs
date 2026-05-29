use std::sync::Arc;

use ai::{
    Agent, AgentContext, AgentEvent, AgentEventSink, AgentLoopConfig, AgentOptions, Message,
    faux_assistant_message, faux_text, register_faux_provider, run_agent_loop,
};
use tokio::sync::Mutex;

#[tokio::test]
async fn agent_prompt_records_user_and_assistant_messages() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("hello", None)]);
    let agent = Agent::new(AgentOptions::new(registration.get_model()));

    agent.prompt_text("hi", Vec::new()).await.unwrap();

    let state = agent.state().await;
    assert_eq!(state.messages.len(), 2);
    assert!(matches!(state.messages[0], Message::User(_)));
    let Message::Assistant(assistant) = &state.messages[1] else {
        panic!("expected assistant message");
    };
    assert_eq!(assistant.content, vec![faux_text("hello")]);

    registration.unregister();
}

#[tokio::test]
async fn run_agent_loop_is_exported_from_ai_crate() {
    let registration = register_faux_provider(None);
    registration.set_responses([faux_assistant_message("done", None)]);
    let events = Arc::new(Mutex::new(Vec::new()));
    let emit: AgentEventSink = Arc::new({
        let events = Arc::clone(&events);
        move |event| {
            let events = Arc::clone(&events);
            Box::pin(async move {
                events.lock().await.push(event_name(&event));
                Ok(())
            })
        }
    });

    let messages = run_agent_loop(
        vec![Message::user_text("hi")],
        AgentContext {
            system_prompt: None,
            messages: Vec::new(),
            tools: Vec::new(),
        },
        AgentLoopConfig::new(registration.get_model()),
        emit,
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(messages.len(), 2);
    assert!(events.lock().await.contains(&"agent_end"));

    registration.unregister();
}

fn event_name(event: &AgentEvent) -> &'static str {
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
