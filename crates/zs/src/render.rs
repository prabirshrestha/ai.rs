use std::io::Write;

use ai::{
    Agent, AgentEvent, AgentSubscription, AssistantContent, AssistantMessage,
    AssistantMessageEvent, Message,
};

pub fn subscribe(agent: &Agent) -> AgentSubscription {
    agent.subscribe(|event, cancellation_token| async move {
        if cancellation_token.is_cancelled() {
            return Ok(());
        }
        render_event(&event);
        Ok(())
    })
}

pub fn help_text() -> &'static str {
    "commands:\n\
     /help                  list commands\n\
     /clear                 reset conversation context\n\
     /model [name]          show or switch the model on the active provider\n\
     /provider [name]       show or switch provider (openai, anthropic, copilot)\n\
     /login [enterprise]    log into GitHub Copilot\n\
     /exit, /quit           leave zs"
}

fn render_event(event: &AgentEvent) {
    match event {
        AgentEvent::MessageUpdate {
            assistant_message_event:
                AssistantMessageEvent::TextDelta { delta, .. }
                | AssistantMessageEvent::ThinkingDelta { delta, .. },
            ..
        } => {
            print!("{delta}");
            let _ = std::io::stdout().flush();
        }
        AgentEvent::MessageUpdate {
            assistant_message_event: AssistantMessageEvent::Error { error, .. },
            ..
        } => {
            eprintln!(
                "\nerror: {}",
                error
                    .error_message
                    .as_deref()
                    .unwrap_or("assistant stream failed")
            );
        }
        AgentEvent::MessageEnd {
            message: Message::Assistant(message),
        } => {
            if let Some(error) = &message.error_message {
                eprintln!(
                    "\nerror from {} ({}, {}): {error}",
                    message.model, message.provider, message.api
                );
            } else if assistant_visible_content(message).trim().is_empty() {
                eprintln!(
                    "\nwarning: empty response from {} ({}, {})",
                    message.model, message.provider, message.api
                );
            } else if !assistant_visible_content(message).ends_with('\n') {
                println!();
            }
        }
        AgentEvent::ToolExecutionStart {
            tool_name, args, ..
        } => {
            println!("\n{tool_name}({args})");
        }
        AgentEvent::ToolExecutionEnd {
            tool_name,
            is_error,
            ..
        } => {
            println!("{tool_name} {}", if *is_error { "error" } else { "done" });
        }
        _ => {}
    }
}

pub fn assistant_visible_content(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .map(|content| match content {
            AssistantContent::Text(text) => text.text.as_str(),
            AssistantContent::Thinking(thinking) => thinking.thinking.as_str(),
            AssistantContent::ToolCall(_) => "<tool_call>",
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ai::{AssistantContent, AssistantMessage, Model, TextContent};

    use super::{assistant_visible_content, help_text};

    #[test]
    fn visible_content_joins_text_blocks() {
        let mut message = AssistantMessage::empty_for(&Model::default());
        message.content = vec![
            AssistantContent::Text(TextContent {
                text: "hello".to_string(),
                text_signature: None,
            }),
            AssistantContent::Text(TextContent {
                text: " world".to_string(),
                text_signature: None,
            }),
        ];

        assert_eq!(assistant_visible_content(&message), "hello world");
        assert!(help_text().contains("/login"));
    }
}
