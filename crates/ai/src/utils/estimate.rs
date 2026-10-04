//! Port of `utils/estimate.ts`. String lengths are JavaScript lengths
//! (UTF-16 code units).

use serde::Serialize;

use crate::types::{AssistantContent, Message, StopReason, Usage, UserContent, UserMessageContent};
use crate::utils::text::get_system_message_text;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    /// Estimated total context tokens.
    pub tokens: u32,
    /// Tokens reported by the most recent applicable assistant usage block.
    pub usage_tokens: u32,
    /// Estimated tokens after the most recent applicable assistant usage block.
    pub trailing_tokens: u32,
    /// Index of the applicable message that provided usage, or `None`.
    pub last_usage_index: Option<usize>,
}

const CHARS_PER_TOKEN: usize = 4;
const ESTIMATED_IMAGE_CHARS: usize = 4800;

/// JavaScript `string.length`.
pub(crate) fn js_length(value: &str) -> usize {
    value.encode_utf16().count()
}

pub fn calculate_context_tokens(usage: &Usage) -> u32 {
    if usage.total_tokens != 0 {
        usage.total_tokens
    } else {
        usage
            .input
            .saturating_add(usage.output)
            .saturating_add(usage.cache_read)
            .saturating_add(usage.cache_write)
    }
}

fn safe_json_stringify<T: Serialize + ?Sized>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_string())
}

fn estimate_text_and_image_content_chars(content: &[UserContent]) -> usize {
    content
        .iter()
        .map(|block| match block {
            UserContent::Text(text) => js_length(&text.text),
            UserContent::Image(_) => ESTIMATED_IMAGE_CHARS,
        })
        .sum()
}

pub fn estimate_text_tokens(text: &str) -> u32 {
    js_length(text).div_ceil(CHARS_PER_TOKEN) as u32
}

pub fn estimate_text_and_image_content_tokens(content: &[UserContent]) -> u32 {
    estimate_text_and_image_content_chars(content).div_ceil(CHARS_PER_TOKEN) as u32
}

fn estimate_user_content_tokens(content: &UserMessageContent) -> u32 {
    match content {
        UserMessageContent::Text(text) => estimate_text_tokens(text),
        UserMessageContent::Parts(parts) => estimate_text_and_image_content_tokens(parts),
    }
}

pub fn estimate_message_tokens(message: &Message) -> u32 {
    match message {
        Message::System(message) => {
            estimate_text_tokens(&get_system_message_text(message))
                + estimate_tools_tokens(message.tools_added.as_deref())
                + estimate_tools_tokens(message.tools_removed.as_deref())
        }
        Message::User(message) => estimate_user_content_tokens(&message.content),
        Message::ToolResult(message) => estimate_text_and_image_content_tokens(&message.content),
        Message::Assistant(message) => {
            let chars: usize = message
                .content
                .iter()
                .map(|block| match block {
                    AssistantContent::Text(text) => js_length(&text.text),
                    AssistantContent::Thinking(thinking) => js_length(&thinking.thinking),
                    AssistantContent::ToolCall(tool_call) => {
                        js_length(&tool_call.name)
                            + js_length(&safe_json_stringify(&tool_call.arguments))
                    }
                })
                .sum();
            chars.div_ceil(CHARS_PER_TOKEN) as u32
        }
    }
}

fn get_last_assistant_usage_info(messages: &[Message]) -> Option<(&Usage, usize)> {
    let mut latest_prefix_timestamp: Option<u64> = None;
    let mut usage_info = None;

    for (index, message) in messages.iter().enumerate() {
        if let Message::Assistant(assistant) = message {
            // A newer prefix message was inserted after this response (for
            // example, a compaction summary), so its usage cannot describe the
            // current prefix.
            let usage_applies_to_prefix =
                latest_prefix_timestamp.is_none_or(|latest| assistant.timestamp >= latest);
            if usage_applies_to_prefix
                && assistant.stop_reason != StopReason::Aborted
                && assistant.stop_reason != StopReason::Error
                && calculate_context_tokens(&assistant.usage) > 0
            {
                usage_info = Some((&assistant.usage, index));
            }
        }
        let timestamp = message.timestamp();
        latest_prefix_timestamp =
            Some(latest_prefix_timestamp.map_or(timestamp, |latest| latest.max(timestamp)));
    }

    usage_info
}

/// Estimate context tokens for a transcript's messages (Pi accepts a
/// `TranscriptContext` or a message list; pass `&context.messages`).
pub fn estimate_context_tokens(messages: &[Message]) -> ContextUsageEstimate {
    if let Some((usage, index)) = get_last_assistant_usage_info(messages) {
        let usage_tokens = calculate_context_tokens(usage);
        let trailing_tokens = messages[index + 1..]
            .iter()
            .map(estimate_message_tokens)
            .fold(0u32, u32::saturating_add);
        return ContextUsageEstimate {
            tokens: usage_tokens.saturating_add(trailing_tokens),
            usage_tokens,
            trailing_tokens,
            last_usage_index: Some(index),
        };
    }

    let tokens = messages
        .iter()
        .map(estimate_message_tokens)
        .fold(0u32, u32::saturating_add);
    ContextUsageEstimate {
        tokens,
        usage_tokens: 0,
        trailing_tokens: tokens,
        last_usage_index: None,
    }
}

fn estimate_tools_tokens<T: Serialize>(tools: Option<&[T]>) -> u32 {
    match tools {
        Some(tools) if !tools.is_empty() => estimate_text_tokens(&safe_json_stringify(tools)),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantMessage, Context, Model, UserMessage};
    use crate::utils::transcript::normalize_context;

    fn create_usage(total_tokens: u32) -> Usage {
        Usage {
            input: total_tokens,
            total_tokens,
            ..Default::default()
        }
    }

    fn create_assistant(timestamp: u64, total_tokens: u32) -> Message {
        let mut message = AssistantMessage::empty_for(&Model {
            id: "test-model".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            ..Default::default()
        });
        message.content = vec![AssistantContent::text("kept")];
        message.usage = create_usage(total_tokens);
        message.timestamp = timestamp;
        Message::Assistant(message)
    }

    fn user(content: impl Into<String>, timestamp: u64) -> Message {
        Message::User(UserMessage {
            content: UserMessageContent::Text(content.into()),
            timestamp,
        })
    }

    #[test]
    fn ignores_stale_assistant_usage_after_a_newer_message_is_inserted_before_it() {
        let context = normalize_context(&Context {
            system_prompt: Some("system".to_string()),
            messages: vec![
                user("summary", 200),
                create_assistant(100, 9_500),
                user("x".repeat(4_000), 300),
            ],
            ..Default::default()
        });

        assert_eq!(
            estimate_context_tokens(&context.messages),
            ContextUsageEstimate {
                tokens: 1_005,
                usage_tokens: 0,
                trailing_tokens: 1_005,
                last_usage_index: None,
            }
        );
    }

    #[test]
    fn uses_assistant_usage_again_after_a_response_to_the_inserted_context() {
        let context = normalize_context(&Context {
            messages: vec![
                user("summary", 200),
                create_assistant(100, 9_500),
                user("new prompt", 300),
                create_assistant(400, 2_000),
                user("tail", 500),
            ],
            ..Default::default()
        });

        assert_eq!(
            estimate_context_tokens(&context.messages),
            ContextUsageEstimate {
                tokens: 2_001,
                usage_tokens: 2_000,
                trailing_tokens: 1,
                last_usage_index: Some(3),
            }
        );
    }

    #[test]
    fn text_estimation_uses_javascript_utf16_string_length() {
        assert_eq!(estimate_text_tokens("😀😀"), 1);
        assert_eq!(estimate_text_tokens("😀😀a"), 2);
    }
}
