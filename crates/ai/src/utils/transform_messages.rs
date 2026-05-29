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
