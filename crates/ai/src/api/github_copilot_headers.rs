//! Port of `api/github-copilot-headers.ts`.

use crate::types::{Message, ProviderHeaders, UserContent, UserMessageContent};

/// Copilot expects X-Initiator to indicate whether the request is user-initiated
/// or agent-initiated (e.g. follow-up after assistant/tool messages).
pub fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
    match messages.last() {
        Some(last) if !matches!(last, Message::User(_)) => "agent",
        _ => "user",
    }
}

/// Copilot requires Copilot-Vision-Request header when sending images
pub fn has_copilot_vision_input(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::User(user) => match &user.content {
            UserMessageContent::Parts(parts) => parts
                .iter()
                .any(|content| matches!(content, UserContent::Image(_))),
            UserMessageContent::Text(_) => false,
        },
        Message::ToolResult(result) => result
            .content
            .iter()
            .any(|content| matches!(content, UserContent::Image(_))),
        _ => false,
    })
}

/// `buildCopilotDynamicHeaders({ messages, hasImages })`.
pub fn build_copilot_dynamic_headers(messages: &[Message], has_images: bool) -> ProviderHeaders {
    let mut headers = ProviderHeaders::new();
    headers.insert("X-Initiator", infer_copilot_initiator(messages).to_string());
    headers.insert("Openai-Intent", "conversation-edits".to_string());

    if has_images {
        headers.insert("Copilot-Vision-Request", "true".to_string());
    }

    headers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ImageContent, SystemMessage, UserMessage};

    #[test]
    fn infers_initiator_from_the_last_message() {
        assert_eq!(infer_copilot_initiator(&[]), "user");
        assert_eq!(infer_copilot_initiator(&[Message::user_text("hi")]), "user");
        let system = Message::System(SystemMessage::default());
        assert_eq!(
            infer_copilot_initiator(&[Message::user_text("hi"), system]),
            "agent"
        );
    }

    #[test]
    fn adds_the_vision_header_only_for_images() {
        let image = Message::User(UserMessage {
            content: UserMessageContent::Parts(vec![UserContent::Image(ImageContent {
                data: "abc".to_string(),
                mime_type: "image/png".to_string(),
            })]),
            timestamp: 1,
        });
        assert!(has_copilot_vision_input(std::slice::from_ref(&image)));
        let headers = build_copilot_dynamic_headers(&[image], true);
        assert_eq!(
            headers
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            vec!["X-Initiator", "Openai-Intent", "Copilot-Vision-Request"]
        );
        assert_eq!(
            build_copilot_dynamic_headers(&[Message::user_text("hi")], false).len(),
            2
        );
    }
}
