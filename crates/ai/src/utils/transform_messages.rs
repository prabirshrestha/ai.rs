use std::collections::{HashMap, HashSet};

use crate::types::{
    AssistantContent, AssistantMessage, ImageContent, Message, Model, ModelInput, TextContent,
    ToolCall, ToolResultContent, ToolResultMessage, UserContent, UserMessage, UserMessageContent,
};

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

pub fn transform_messages<F>(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: F,
) -> Vec<Message>
where
    F: Fn(&str, &Model, &AssistantMessage) -> String,
{
    let mut tool_call_id_map: HashMap<String, String> = HashMap::new();
    let image_aware_messages = downgrade_unsupported_images(messages, model);
    let mut transformed = Vec::with_capacity(image_aware_messages.len());

    for message in image_aware_messages {
        match message {
            Message::User(_) => transformed.push(message),
            Message::ToolResult(mut tool_result) => {
                if let Some(normalized) = tool_call_id_map.get(&tool_result.tool_call_id) {
                    tool_result.tool_call_id = normalized.clone();
                }
                transformed.push(Message::ToolResult(tool_result));
            }
            Message::Assistant(assistant) => {
                let is_same_model = assistant.provider == model.provider
                    && assistant.api == model.api
                    && assistant.model == model.id;
                let mut content = Vec::new();

                for block in assistant.content.iter() {
                    match block {
                        AssistantContent::Thinking(thinking) => {
                            if thinking.redacted == Some(true) {
                                if is_same_model {
                                    content.push(block.clone());
                                }
                                continue;
                            }
                            if is_same_model && thinking.thinking_signature.is_some() {
                                content.push(block.clone());
                                continue;
                            }
                            if thinking.thinking.trim().is_empty() {
                                continue;
                            }
                            if is_same_model {
                                content.push(block.clone());
                            } else {
                                content.push(AssistantContent::Text(TextContent {
                                    text: thinking.thinking.clone(),
                                    text_signature: None,
                                }));
                            }
                        }
                        AssistantContent::Text(text) => {
                            content.push(AssistantContent::Text(TextContent {
                                text: text.text.clone(),
                                text_signature: if is_same_model {
                                    text.text_signature.clone()
                                } else {
                                    None
                                },
                            }));
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let mut normalized = tool_call.clone();
                            if !is_same_model {
                                normalized.thought_signature = None;
                                let new_id =
                                    normalize_tool_call_id(&tool_call.id, model, &assistant);
                                if new_id != tool_call.id {
                                    tool_call_id_map.insert(tool_call.id.clone(), new_id.clone());
                                    normalized.id = new_id;
                                }
                            }
                            content.push(AssistantContent::ToolCall(normalized));
                        }
                    }
                }

                transformed.push(Message::Assistant(AssistantMessage {
                    content,
                    ..assistant
                }));
            }
        }
    }

    insert_synthetic_tool_results(transformed)
}

fn downgrade_unsupported_images(messages: &[Message], model: &Model) -> Vec<Message> {
    if model.input.contains(&ModelInput::Image) {
        return messages.to_vec();
    }

    messages
        .iter()
        .cloned()
        .map(|message| match message {
            Message::User(mut user) => {
                if let UserMessageContent::Parts(parts) = user.content {
                    user.content = UserMessageContent::Parts(replace_user_images(
                        &parts,
                        NON_VISION_USER_IMAGE_PLACEHOLDER,
                    ));
                }
                Message::User(user)
            }
            Message::ToolResult(mut tool_result) => {
                tool_result.content =
                    replace_tool_images(&tool_result.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER);
                Message::ToolResult(tool_result)
            }
            other => other,
        })
        .collect()
}

fn replace_user_images(content: &[UserContent], placeholder: &str) -> Vec<UserContent> {
    let mut result = Vec::new();
    let mut previous_was_placeholder = false;
    for block in content {
        match block {
            UserContent::Image(ImageContent { .. }) => {
                if !previous_was_placeholder {
                    result.push(UserContent::Text(TextContent {
                        text: placeholder.to_string(),
                        text_signature: None,
                    }));
                }
                previous_was_placeholder = true;
            }
            UserContent::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                result.push(block.clone());
            }
        }
    }
    result
}

fn replace_tool_images(content: &[ToolResultContent], placeholder: &str) -> Vec<ToolResultContent> {
    let mut result = Vec::new();
    let mut previous_was_placeholder = false;
    for block in content {
        match block {
            ToolResultContent::Image(ImageContent { .. }) => {
                if !previous_was_placeholder {
                    result.push(ToolResultContent::Text(TextContent {
                        text: placeholder.to_string(),
                        text_signature: None,
                    }));
                }
                previous_was_placeholder = true;
            }
            ToolResultContent::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                result.push(block.clone());
            }
        }
    }
    result
}

fn insert_synthetic_tool_results(messages: Vec<Message>) -> Vec<Message> {
    let mut result = Vec::new();
    let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
    let mut existing_tool_result_ids: HashSet<String> = HashSet::new();

    fn insert(
        result: &mut Vec<Message>,
        pending_tool_calls: &mut Vec<ToolCall>,
        existing_tool_result_ids: &mut HashSet<String>,
    ) {
        for tool_call in pending_tool_calls.drain(..) {
            if existing_tool_result_ids.contains(&tool_call.id) {
                continue;
            }
            result.push(Message::ToolResult(ToolResultMessage {
                tool_call_id: tool_call.id,
                tool_name: tool_call.name,
                content: vec![ToolResultContent::text("No result provided")],
                details: None,
                is_error: true,
                timestamp: crate::utils::time::now_millis(),
            }));
        }
        existing_tool_result_ids.clear();
    }

    for message in messages {
        match &message {
            Message::Assistant(assistant) => {
                insert(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                );
                if matches!(
                    assistant.stop_reason,
                    crate::types::StopReason::Error | crate::types::StopReason::Aborted
                ) {
                    continue;
                }
                pending_tool_calls = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                        _ => None,
                    })
                    .collect();
                existing_tool_result_ids.clear();
                result.push(message);
            }
            Message::ToolResult(tool_result) => {
                existing_tool_result_ids.insert(tool_result.tool_call_id.clone());
                result.push(message);
            }
            Message::User(UserMessage { .. }) => {
                insert(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                );
                result.push(message);
            }
        }
    }
    insert(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_tool_result_ids,
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelCost, StopReason, ThinkingContent, Usage};
    use serde_json::json;

    fn copilot_claude_model() -> Model {
        Model {
            id: "claude-sonnet-4.6".to_string(),
            name: "Claude Sonnet 4.6".to_string(),
            api: "anthropic-messages".to_string(),
            provider: "github-copilot".to_string(),
            base_url: "https://api.individual.githubcopilot.com".to_string(),
            reasoning: true,
            input: vec![ModelInput::Text, ModelInput::Image],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 16_000,
            ..Default::default()
        }
    }

    fn assistant_message(content: Vec<AssistantContent>) -> AssistantMessage {
        AssistantMessage {
            content,
            api: "openai-responses".to_string(),
            provider: "github-copilot".to_string(),
            model: "gpt-5".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 0,
        }
    }

    fn normalize_for_anthropic(id: &str, _model: &Model, _source: &AssistantMessage) -> String {
        id.chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .take(64)
            .collect()
    }

    #[test]
    fn converts_cross_model_thinking_to_text_for_copilot_anthropic_handoff() {
        let model = copilot_claude_model();
        let source = AssistantMessage {
            content: vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "Let me think about this...".to_string(),
                    thinking_signature: Some("reasoning_content".to_string()),
                    redacted: None,
                }),
                AssistantContent::Text(TextContent {
                    text: "Hi there!".to_string(),
                    text_signature: None,
                }),
            ],
            api: "openai-completions".to_string(),
            provider: "github-copilot".to_string(),
            model: "gpt-4o".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        };

        let transformed = transform_messages(
            &[Message::user_text("hello"), Message::Assistant(source)],
            &model,
            normalize_for_anthropic,
        );

        let assistant = transformed
            .iter()
            .find_map(|message| match message {
                Message::Assistant(assistant) => Some(assistant),
                _ => None,
            })
            .expect("assistant message");
        assert!(
            assistant
                .content
                .iter()
                .all(|block| !matches!(block, AssistantContent::Thinking(_)))
        );
        assert_eq!(
            assistant.content,
            vec![
                AssistantContent::Text(TextContent {
                    text: "Let me think about this...".to_string(),
                    text_signature: None,
                }),
                AssistantContent::Text(TextContent {
                    text: "Hi there!".to_string(),
                    text_signature: None,
                }),
            ]
        );
    }

    #[test]
    fn removes_tool_call_thought_signatures_when_migrating_between_models() {
        let model = copilot_claude_model();
        let transformed = transform_messages(
            &[
                Message::user_text("run a command"),
                Message::Assistant(assistant_message(vec![AssistantContent::ToolCall(
                    ToolCall {
                        id: "call_123".to_string(),
                        name: "bash".to_string(),
                        arguments: json!({ "command": "ls" }),
                        thought_signature: Some(
                            json!({ "type": "reasoning.encrypted", "id": "call_123" }).to_string(),
                        ),
                    },
                )])),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_123".to_string(),
                    tool_name: "bash".to_string(),
                    content: vec![ToolResultContent::text("output")],
                    details: None,
                    is_error: false,
                    timestamp: 0,
                }),
            ],
            &model,
            normalize_for_anthropic,
        );

        let tool_call = transformed
            .iter()
            .find_map(|message| match message {
                Message::Assistant(assistant) => assistant.content.iter().find_map(|block| {
                    if let AssistantContent::ToolCall(tool_call) = block {
                        Some(tool_call)
                    } else {
                        None
                    }
                }),
                _ => None,
            })
            .expect("tool call");
        assert_eq!(tool_call.id, "call_123");
        assert_eq!(tool_call.thought_signature, None);
    }

    #[test]
    fn adds_synthetic_results_for_trailing_orphaned_tool_calls_after_normalization() {
        let model = copilot_claude_model();
        let transformed = transform_messages(
            &[
                Message::user_text("run commands"),
                Message::Assistant(assistant_message(vec![
                    AssistantContent::ToolCall(ToolCall {
                        id: "call_1|fc_1".to_string(),
                        name: "read".to_string(),
                        arguments: json!({ "path": "README.md" }),
                        thought_signature: None,
                    }),
                    AssistantContent::ToolCall(ToolCall {
                        id: "call_2|fc_2".to_string(),
                        name: "bash".to_string(),
                        arguments: json!({ "command": "pwd" }),
                        thought_signature: None,
                    }),
                ])),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_1|fc_1".to_string(),
                    tool_name: "read".to_string(),
                    content: vec![ToolResultContent::text("done")],
                    details: None,
                    is_error: false,
                    timestamp: 0,
                }),
            ],
            &model,
            normalize_for_anthropic,
        );

        let synthetic_results = transformed
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult(result) if result.is_error => Some(result),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(synthetic_results.len(), 1);
        assert_eq!(synthetic_results[0].tool_call_id, "call_2_fc_2");
        assert_eq!(synthetic_results[0].tool_name, "bash");
        assert_eq!(
            synthetic_results[0].content,
            vec![ToolResultContent::text("No result provided")]
        );
    }
}
