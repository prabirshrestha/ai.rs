use crate::types::{
    AssistantContent, Context, Message, StopReason, Tool, ToolResultContent, Usage, UserContent,
    UserMessageContent,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    pub tokens: u32,
    pub usage_tokens: u32,
    pub trailing_tokens: u32,
    pub last_usage_index: Option<usize>,
}

const CHARS_PER_TOKEN: usize = 4;
const ESTIMATED_IMAGE_CHARS: usize = 4_800;

fn string_length(value: &str) -> usize {
    value.encode_utf16().count()
}

pub fn calculate_context_tokens(usage: &Usage) -> u32 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage
            .input
            .saturating_add(usage.output)
            .saturating_add(usage.cache_read)
            .saturating_add(usage.cache_write)
    }
}

fn estimate_text_and_image_content_chars<'a>(
    text: Option<&str>,
    parts: impl IntoIterator<Item = TextOrImage<'a>>,
) -> usize {
    if let Some(text) = text {
        return string_length(text);
    }
    parts
        .into_iter()
        .map(|part| match part {
            TextOrImage::Text(text) => string_length(text),
            TextOrImage::Image => ESTIMATED_IMAGE_CHARS,
        })
        .sum()
}

enum TextOrImage<'a> {
    Text(&'a str),
    Image,
}

pub fn estimate_text_tokens(text: &str) -> u32 {
    string_length(text).div_ceil(CHARS_PER_TOKEN) as u32
}

pub fn estimate_message_tokens(message: &Message) -> u32 {
    let chars = match message {
        Message::User(user) => match &user.content {
            UserMessageContent::Text(text) => {
                estimate_text_and_image_content_chars(Some(text), std::iter::empty())
            }
            UserMessageContent::Parts(parts) => estimate_text_and_image_content_chars(
                None,
                parts.iter().map(|part| match part {
                    UserContent::Text(text) => TextOrImage::Text(&text.text),
                    UserContent::Image(_) => TextOrImage::Image,
                }),
            ),
        },
        Message::ToolResult(tool_result) => estimate_text_and_image_content_chars(
            None,
            tool_result.content.iter().map(|part| match part {
                ToolResultContent::Text(text) => TextOrImage::Text(&text.text),
                ToolResultContent::Image(_) => TextOrImage::Image,
            }),
        ),
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .map(|block| match block {
                AssistantContent::Text(text) => string_length(&text.text),
                AssistantContent::Thinking(thinking) => string_length(&thinking.thinking),
                AssistantContent::ToolCall(tool_call) => {
                    string_length(&tool_call.name)
                        + string_length(
                            &serde_json::to_string(&tool_call.arguments)
                                .unwrap_or_else(|_| "[unserializable]".to_string()),
                        )
                }
            })
            .sum(),
        Message::Custom(_) => 0,
    };
    chars.div_ceil(CHARS_PER_TOKEN) as u32
}

fn message_timestamp(message: &Message) -> Option<u64> {
    match message {
        Message::User(message) => Some(message.timestamp),
        Message::Assistant(message) => Some(message.timestamp),
        Message::ToolResult(message) => Some(message.timestamp),
        Message::Custom(_) => None,
    }
}

fn get_last_assistant_usage_info(messages: &[Message]) -> Option<(&Usage, usize)> {
    let mut latest_prefix_timestamp = 0;
    let mut usage_info = None;
    for (index, message) in messages.iter().enumerate() {
        if let Message::Assistant(assistant) = message {
            let usage_applies_to_prefix = assistant.timestamp >= latest_prefix_timestamp;
            if usage_applies_to_prefix
                && !matches!(
                    assistant.stop_reason,
                    StopReason::Aborted | StopReason::Error
                )
                && calculate_context_tokens(&assistant.usage) > 0
            {
                usage_info = Some((&assistant.usage, index));
            }
        }
        if let Some(timestamp) = message_timestamp(message) {
            latest_prefix_timestamp = latest_prefix_timestamp.max(timestamp);
        }
    }
    usage_info
}

fn estimate_messages(messages: &[Message]) -> ContextUsageEstimate {
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

fn estimate_tools_tokens(tools: &[Tool]) -> u32 {
    if tools.is_empty() {
        return 0;
    }
    estimate_text_tokens(
        &serde_json::to_string(tools).unwrap_or_else(|_| "[unserializable]".to_string()),
    )
}

pub fn estimate_context_tokens(context: &Context) -> ContextUsageEstimate {
    let estimate = estimate_messages(&context.messages);
    if let Some(last_usage_index) = estimate.last_usage_index {
        let added_names = context.messages[last_usage_index + 1..]
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult(result) => Some(result.added_tool_names.iter()),
                Message::User(_) | Message::Assistant(_) | Message::Custom(_) => None,
            })
            .flatten()
            .collect::<std::collections::HashSet<_>>();
        let added_tools = context
            .tools
            .iter()
            .filter(|tool| added_names.contains(&tool.name))
            .cloned()
            .collect::<Vec<_>>();
        let added_tool_tokens = estimate_tools_tokens(&added_tools);
        return ContextUsageEstimate {
            tokens: estimate.tokens.saturating_add(added_tool_tokens),
            usage_tokens: estimate.usage_tokens,
            trailing_tokens: estimate.trailing_tokens.saturating_add(added_tool_tokens),
            last_usage_index: estimate.last_usage_index,
        };
    }

    let prefix_tokens = context
        .system_prompt
        .as_deref()
        .map(estimate_text_tokens)
        .unwrap_or_default()
        .saturating_add(estimate_tools_tokens(&context.tools));
    ContextUsageEstimate {
        tokens: estimate.tokens.saturating_add(prefix_tokens),
        usage_tokens: estimate.usage_tokens,
        trailing_tokens: estimate.trailing_tokens.saturating_add(prefix_tokens),
        last_usage_index: estimate.last_usage_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AssistantMessage, Model, ModelCost, ModelInput, TextContent, UsageCost, UserMessage,
    };

    fn create_usage(total_tokens: u32) -> Usage {
        Usage {
            input: total_tokens,
            total_tokens,
            cost: UsageCost::default(),
            ..Default::default()
        }
    }

    fn create_assistant(timestamp: u64, total_tokens: u32) -> AssistantMessage {
        AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: "kept".to_string(),
                text_signature: None,
            })],
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            model: "test-model".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: create_usage(total_tokens),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp,
        }
    }

    fn model() -> Model {
        Model {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            api: "openai-responses".to_string(),
            provider: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            reasoning: false,
            input: vec![ModelInput::Text],
            cost: ModelCost::default(),
            context_window: 10_000,
            max_tokens: 8_000,
            ..Default::default()
        }
    }

    #[test]
    fn ignores_stale_assistant_usage_after_a_newer_message_is_inserted_before_it() {
        let context = Context {
            system_prompt: Some("system".to_string()),
            messages: vec![
                Message::User(UserMessage {
                    content: UserMessageContent::Text("summary".to_string()),
                    timestamp: 200,
                }),
                Message::Assistant(create_assistant(100, 9_500)),
                Message::User(UserMessage {
                    content: UserMessageContent::Text("x".repeat(4_000)),
                    timestamp: 300,
                }),
            ],
            tools: Vec::new(),
        };

        assert_eq!(
            estimate_context_tokens(&context),
            ContextUsageEstimate {
                tokens: 1_005,
                usage_tokens: 0,
                trailing_tokens: 1_005,
                last_usage_index: None,
            }
        );
        assert_eq!(
            crate::providers::simple_options::clamp_max_tokens_to_context(
                &model(),
                &context,
                8_000,
            ),
            4_899
        );
    }

    #[test]
    fn uses_assistant_usage_again_after_a_response_to_the_inserted_context() {
        let context = Context {
            messages: vec![
                Message::User(UserMessage {
                    content: UserMessageContent::Text("summary".to_string()),
                    timestamp: 200,
                }),
                Message::Assistant(create_assistant(100, 9_500)),
                Message::User(UserMessage {
                    content: UserMessageContent::Text("new prompt".to_string()),
                    timestamp: 300,
                }),
                Message::Assistant(create_assistant(400, 2_000)),
                Message::User(UserMessage {
                    content: UserMessageContent::Text("tail".to_string()),
                    timestamp: 500,
                }),
            ],
            ..Default::default()
        };

        assert_eq!(
            estimate_context_tokens(&context),
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
    }

    #[test]
    fn ignores_tool_execution_usage() {
        let tool_result = Message::ToolResult(crate::ToolResultMessage {
            tool_call_id: "call_1".to_string(),
            tool_name: "llm_tool".to_string(),
            content: vec![ToolResultContent::text("done")],
            details: None,
            usage: Some(create_usage(9_000)),
            added_tool_names: Vec::new(),
            is_error: false,
            timestamp: 1,
        });
        let mut without_usage = tool_result.clone();
        let Message::ToolResult(result) = &mut without_usage else {
            unreachable!();
        };
        result.usage = None;

        assert_eq!(
            estimate_message_tokens(&tool_result),
            estimate_message_tokens(&without_usage)
        );
    }
}
