//! Port of `utils/text.ts`.

use crate::types::{
    AssistantContent, SystemContent, SystemMessage, SystemMessageContent, UserContent,
    UserMessageContent,
};

/// Message content that [`content_text`] can read: a string or a list of
/// blocks of any kind.
pub trait ContentBlocks {
    /// The text of each `text` block, in order, or the string itself.
    fn text_parts(&self) -> TextParts<'_>;
}

pub enum TextParts<'a> {
    String(&'a str),
    Blocks(Vec<&'a str>),
}

impl ContentBlocks for str {
    fn text_parts(&self) -> TextParts<'_> {
        TextParts::String(self)
    }
}

impl ContentBlocks for String {
    fn text_parts(&self) -> TextParts<'_> {
        TextParts::String(self)
    }
}

impl ContentBlocks for [AssistantContent] {
    fn text_parts(&self) -> TextParts<'_> {
        TextParts::Blocks(
            self.iter()
                .filter_map(|block| match block {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect(),
        )
    }
}

impl ContentBlocks for Vec<AssistantContent> {
    fn text_parts(&self) -> TextParts<'_> {
        self.as_slice().text_parts()
    }
}

impl ContentBlocks for [UserContent] {
    fn text_parts(&self) -> TextParts<'_> {
        TextParts::Blocks(
            self.iter()
                .filter_map(|block| match block {
                    UserContent::Text(text) => Some(text.text.as_str()),
                    UserContent::Image(_) => None,
                })
                .collect(),
        )
    }
}

impl ContentBlocks for Vec<UserContent> {
    fn text_parts(&self) -> TextParts<'_> {
        self.as_slice().text_parts()
    }
}

impl ContentBlocks for UserMessageContent {
    fn text_parts(&self) -> TextParts<'_> {
        match self {
            Self::Text(text) => TextParts::String(text),
            Self::Parts(parts) => parts.text_parts(),
        }
    }
}

impl ContentBlocks for SystemMessageContent {
    fn text_parts(&self) -> TextParts<'_> {
        match self {
            Self::Text(text) => TextParts::String(text),
            Self::Parts(parts) => TextParts::Blocks(
                parts
                    .iter()
                    .map(|SystemContent::Text(text)| text.text.as_str())
                    .collect(),
            ),
        }
    }
}

/// Extract and join text from message content (`separator` defaults to `"\n"`
/// in Pi; see [`content_text`]).
pub fn content_text_with<C: ContentBlocks + ?Sized>(content: &C, separator: &str) -> String {
    match content.text_parts() {
        TextParts::String(text) => text.to_string(),
        TextParts::Blocks(parts) => parts.join(separator),
    }
}

/// Extract and join text from message content with `"\n"`.
pub fn content_text<C: ContentBlocks + ?Sized>(content: &C) -> String {
    content_text_with(content, "\n")
}

/// Render a system message as a complete prompt: its content followed by its sections.
pub fn get_system_message_text(message: &SystemMessage) -> String {
    let mut parts = vec![content_text(&message.content)];
    for text in message
        .sections
        .iter()
        .flat_map(|sections| sections.values())
        .flatten()
    {
        parts.push(text.clone());
    }
    parts
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Render a later system message for APIs that accept system messages
/// mid-conversation. Section changes are framed by name so the model can
/// relate them to the leading prompt.
pub fn render_system_message_update(message: &SystemMessage) -> String {
    let mut parts = Vec::new();
    let text = content_text(&message.content);
    if !text.is_empty() {
        parts.push(text);
    }
    for (name, value) in message.sections.iter().flat_map(|sections| sections.iter()) {
        parts.push(match value {
            None => format!("Removed system prompt section \"{name}\"."),
            Some(value) => format!("Updated system prompt section \"{name}\":\n\n{value}"),
        });
    }
    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use indexmap::IndexMap;

    use super::*;
    use crate::types::{ImageContent, TextContent, ThinkingContent, ToolCall};

    fn content() -> Vec<AssistantContent> {
        vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "reasoning".to_string(),
                ..Default::default()
            }),
            AssistantContent::text("first"),
            AssistantContent::ToolCall(ToolCall {
                id: "1".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({}),
                thought_signature: None,
                namespace: None,
            }),
            AssistantContent::text("second"),
        ]
    }

    #[test]
    fn extracts_assistant_text_blocks() {
        assert_eq!(content_text(&content()), "first\nsecond");
    }

    #[test]
    fn supports_custom_separators() {
        assert_eq!(content_text_with(&content(), ""), "firstsecond");
    }

    #[test]
    fn passes_string_content_through() {
        assert_eq!(content_text("hello"), "hello");
    }

    #[test]
    fn extracts_text_from_tool_result_content() {
        let tool_result_content = vec![
            UserContent::Text(TextContent::new("first")),
            UserContent::Image(ImageContent {
                data: "...".to_string(),
                mime_type: "image/png".to_string(),
            }),
            UserContent::Text(TextContent::new("second")),
        ];
        assert_eq!(content_text_with(&tool_result_content, ""), "firstsecond");
    }

    #[test]
    fn renders_system_message_text_and_updates() {
        let mut sections = IndexMap::new();
        sections.insert("a".to_string(), Some("<a/>".to_string()));
        sections.insert("b".to_string(), None);
        let message = SystemMessage {
            content: "base".into(),
            sections: Some(sections),
            ..Default::default()
        };
        assert_eq!(get_system_message_text(&message), "base\n\n<a/>");
        assert_eq!(
            render_system_message_update(&message),
            "base\n\nUpdated system prompt section \"a\":\n\n<a/>\n\nRemoved system prompt section \"b\"."
        );
    }
}
