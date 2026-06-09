use std::collections::HashMap;

use crate::types::{Message, ToolResultContent, UserContent, UserMessageContent};

pub(super) fn copilot_static_headers() -> HashMap<String, String> {
    HashMap::from([
        (
            "User-Agent".to_string(),
            "GitHubCopilotChat/0.35.0".to_string(),
        ),
        ("Editor-Version".to_string(), "vscode/1.107.0".to_string()),
        (
            "Editor-Plugin-Version".to_string(),
            "copilot-chat/0.35.0".to_string(),
        ),
        (
            "Copilot-Integration-Id".to_string(),
            "vscode-chat".to_string(),
        ),
    ])
}

pub fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
    match messages
        .iter()
        .rev()
        .find(|message| !matches!(message, Message::Custom(_)))
    {
        Some(Message::User(_)) | None => "user",
        Some(Message::Assistant(_)) | Some(Message::ToolResult(_)) => "agent",
        Some(Message::Custom(_)) => unreachable!("custom messages are filtered above"),
    }
}

pub fn has_copilot_vision_input(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::User(message) => match &message.content {
            UserMessageContent::Text(_) => false,
            UserMessageContent::Parts(parts) => parts
                .iter()
                .any(|content| matches!(content, UserContent::Image(_))),
        },
        Message::ToolResult(message) => message
            .content
            .iter()
            .any(|content| matches!(content, ToolResultContent::Image(_))),
        Message::Assistant(_) => false,
        Message::Custom(_) => false,
    })
}

pub fn build_copilot_dynamic_headers(
    messages: &[Message],
    has_images: bool,
) -> HashMap<String, String> {
    let mut headers = HashMap::from([
        (
            "X-Initiator".to_string(),
            infer_copilot_initiator(messages).to_string(),
        ),
        (
            "Openai-Intent".to_string(),
            "conversation-edits".to_string(),
        ),
    ]);

    if has_images {
        headers.insert("Copilot-Vision-Request".to_string(), "true".to_string());
    }

    headers
}

#[cfg(test)]
mod tests {
    use crate::types::{
        AssistantMessage, Context, ImageContent, Model, ModelCost, ToolResultContent,
        ToolResultMessage, UserContent, UserMessage, UserMessageContent,
    };

    use super::*;

    #[test]
    fn infers_user_initiator_for_empty_or_user_final_turns() {
        assert_eq!(infer_copilot_initiator(&[]), "user");
        assert_eq!(infer_copilot_initiator(&[Message::user_text("hi")]), "user");
    }

    #[test]
    fn infers_agent_initiator_after_assistant_or_tool_turns() {
        let model = Model {
            id: "test".to_string(),
            name: "test".to_string(),
            api: "openai-completions".to_string(),
            provider: "github-copilot".to_string(),
            base_url: "http://localhost".to_string(),
            cost: ModelCost::default(),
            ..Model::default()
        };
        let assistant = Message::Assistant(AssistantMessage::empty_for(&model));
        assert_eq!(infer_copilot_initiator(&[assistant]), "agent");

        let tool_result = Message::ToolResult(ToolResultMessage {
            tool_call_id: "tool-1".to_string(),
            tool_name: "echo".to_string(),
            content: vec![],
            details: None,
            is_error: false,
            timestamp: 1,
        });
        assert_eq!(infer_copilot_initiator(&[tool_result]), "agent");
    }

    #[test]
    fn infers_initiator_from_latest_non_custom_message() {
        let model = Model {
            id: "test".to_string(),
            name: "test".to_string(),
            api: "openai-completions".to_string(),
            provider: "github-copilot".to_string(),
            base_url: "http://localhost".to_string(),
            cost: ModelCost::default(),
            ..Model::default()
        };
        let assistant = Message::Assistant(AssistantMessage::empty_for(&model));
        let tool_result = Message::ToolResult(ToolResultMessage {
            tool_call_id: "tool-1".to_string(),
            tool_name: "echo".to_string(),
            content: vec![],
            details: None,
            is_error: false,
            timestamp: 1,
        });
        let custom = || Message::custom(serde_json::json!({ "kind": "metadata" }));

        assert_eq!(
            infer_copilot_initiator(&[Message::user_text("hi"), custom()]),
            "user"
        );
        assert_eq!(infer_copilot_initiator(&[assistant, custom()]), "agent");
        assert_eq!(infer_copilot_initiator(&[tool_result, custom()]), "agent");
        assert_eq!(infer_copilot_initiator(&[custom()]), "user");
    }

    #[test]
    fn detects_user_and_tool_result_images() {
        let user_image = Message::User(UserMessage {
            content: UserMessageContent::Parts(vec![UserContent::Image(ImageContent {
                data: "abc".to_string(),
                mime_type: "image/png".to_string(),
            })]),
            timestamp: 1,
        });
        assert!(has_copilot_vision_input(&[user_image]));

        let tool_image = Message::ToolResult(ToolResultMessage {
            tool_call_id: "tool-1".to_string(),
            tool_name: "image".to_string(),
            content: vec![ToolResultContent::Image(ImageContent {
                data: "abc".to_string(),
                mime_type: "image/png".to_string(),
            })],
            details: None,
            is_error: false,
            timestamp: 1,
        });
        assert!(has_copilot_vision_input(&[tool_image]));
    }

    #[test]
    fn builds_dynamic_headers() {
        let context = Context {
            messages: vec![Message::user_text("hi")],
            ..Context::default()
        };
        let headers = build_copilot_dynamic_headers(&context.messages, true);

        assert_eq!(headers.get("X-Initiator").map(String::as_str), Some("user"));
        assert_eq!(
            headers.get("Openai-Intent").map(String::as_str),
            Some("conversation-edits")
        );
        assert_eq!(
            headers.get("Copilot-Vision-Request").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn builds_static_headers() {
        let headers = copilot_static_headers();

        assert_eq!(
            headers.get("User-Agent").map(String::as_str),
            Some("GitHubCopilotChat/0.35.0")
        );
        assert_eq!(
            headers.get("Editor-Version").map(String::as_str),
            Some("vscode/1.107.0")
        );
        assert_eq!(
            headers.get("Editor-Plugin-Version").map(String::as_str),
            Some("copilot-chat/0.35.0")
        );
        assert_eq!(
            headers.get("Copilot-Integration-Id").map(String::as_str),
            Some("vscode-chat")
        );
    }
}
