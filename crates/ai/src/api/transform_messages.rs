//! Port of `api/transform-messages.ts`: provider-neutral replay cleanup run
//! before every provider request (image downgrades, cross-model thinking,
//! tool call id normalization, synthetic results for orphaned tool calls).
//!
//! Pi's null/missing `content` normalization happens at deserialization in
//! Rust (`deserialize_null_default` on the message types), so a [`Message`]
//! always carries content here.

use std::collections::{HashMap, HashSet};

use crate::types::{
    AssistantContent, AssistantMessage, Message, Model, ModelInput, StopReason, TextContent,
    ToolCall, ToolResultContent, ToolResultMessage, UserContent, UserMessageContent,
};
use crate::utils::time::now_millis;

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

fn replace_images_with_placeholder(content: &[UserContent], placeholder: &str) -> Vec<UserContent> {
    let mut result = Vec::new();
    let mut previous_was_placeholder = false;

    for block in content {
        match block {
            UserContent::Image(_) => {
                if !previous_was_placeholder {
                    result.push(UserContent::Text(TextContent::new(placeholder)));
                }
                previous_was_placeholder = true;
            }
            UserContent::Text(text) => {
                result.push(block.clone());
                previous_was_placeholder = text.text == placeholder;
            }
        }
    }

    result
}

fn downgrade_unsupported_images(messages: &[Message], model: &Model) -> Vec<Message> {
    if model.input.contains(&ModelInput::Image) {
        return messages.to_vec();
    }

    messages
        .iter()
        .map(|message| match message {
            Message::User(user) => match &user.content {
                UserMessageContent::Parts(parts) => {
                    let mut user = user.clone();
                    user.content = UserMessageContent::Parts(replace_images_with_placeholder(
                        parts,
                        NON_VISION_USER_IMAGE_PLACEHOLDER,
                    ));
                    Message::User(user)
                }
                UserMessageContent::Text(_) => message.clone(),
            },
            Message::ToolResult(tool_result) => {
                let mut tool_result = tool_result.clone();
                tool_result.content = replace_images_with_placeholder(
                    &tool_result.content,
                    NON_VISION_TOOL_IMAGE_PLACEHOLDER,
                );
                Message::ToolResult(tool_result)
            }
            _ => message.clone(),
        })
        .collect()
}

/// Rewrites a tool call id for the target API (`normalizeToolCallId`).
pub type NormalizeToolCallId<'a> = &'a dyn Fn(&str, &Model, &AssistantMessage) -> String;

/// Normalize a transcript for `model`: unsupported images become
/// placeholders, thinking from other models becomes text (or is dropped),
/// tool call ids from other models go through `normalize_tool_call_id`,
/// errored/aborted assistant turns are skipped, and orphaned tool calls get
/// synthetic error results.
pub fn transform_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: Option<NormalizeToolCallId>,
) -> Vec<Message> {
    // Build a map of original tool call IDs to normalized IDs
    let mut tool_call_id_map: HashMap<String, String> = HashMap::new();
    let image_aware_messages = downgrade_unsupported_images(messages, model);

    // First pass: transform messages (unsupported image downgrade, thinking blocks, tool call ID normalization)
    let transformed: Vec<Message> = image_aware_messages
        .into_iter()
        .map(|message| match message {
            // System and user messages pass through unchanged
            Message::System(_) | Message::User(_) => message,
            // Handle toolResult messages - normalize toolCallId if we have a mapping
            Message::ToolResult(mut tool_result) => {
                if let Some(normalized_id) = tool_call_id_map.get(&tool_result.tool_call_id)
                    && *normalized_id != tool_result.tool_call_id
                {
                    tool_result.tool_call_id = normalized_id.clone();
                }
                Message::ToolResult(tool_result)
            }
            // Assistant messages need transformation check
            Message::Assistant(assistant) => {
                let is_same_model = assistant.provider == model.provider
                    && assistant.api == model.api
                    && assistant.model == model.id;

                let mut transformed_content = Vec::with_capacity(assistant.content.len());
                for block in &assistant.content {
                    match block {
                        AssistantContent::Thinking(thinking) => {
                            // Redacted thinking is opaque encrypted content, only valid for the same model.
                            // Drop it for cross-model to avoid API errors.
                            if thinking.redacted == Some(true) {
                                if is_same_model {
                                    transformed_content.push(block.clone());
                                }
                                continue;
                            }
                            // For same model: keep thinking blocks with signatures (needed for replay)
                            // even if the thinking text is empty (OpenAI encrypted reasoning)
                            if is_same_model
                                && thinking
                                    .thinking_signature
                                    .as_deref()
                                    .is_some_and(|signature| !signature.is_empty())
                            {
                                transformed_content.push(block.clone());
                                continue;
                            }
                            // Skip empty thinking blocks, convert others to plain text
                            if thinking.thinking.trim().is_empty() {
                                continue;
                            }
                            if is_same_model {
                                transformed_content.push(block.clone());
                            } else {
                                transformed_content.push(AssistantContent::Text(TextContent::new(
                                    thinking.thinking.clone(),
                                )));
                            }
                        }
                        AssistantContent::Text(text) => {
                            if is_same_model {
                                transformed_content.push(block.clone());
                            } else {
                                transformed_content.push(AssistantContent::Text(TextContent::new(
                                    text.text.clone(),
                                )));
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let mut normalized_tool_call = tool_call.clone();

                            if !is_same_model && tool_call.thought_signature.is_some() {
                                normalized_tool_call.thought_signature = None;
                            }

                            if !is_same_model && let Some(normalize) = normalize_tool_call_id {
                                let normalized_id = normalize(&tool_call.id, model, &assistant);
                                if normalized_id != tool_call.id {
                                    tool_call_id_map
                                        .insert(tool_call.id.clone(), normalized_id.clone());
                                    normalized_tool_call.id = normalized_id;
                                }
                            }

                            transformed_content
                                .push(AssistantContent::ToolCall(normalized_tool_call));
                        }
                    }
                }

                Message::Assistant(AssistantMessage {
                    content: transformed_content,
                    ..assistant
                })
            }
        })
        .collect();

    // Second pass: insert synthetic empty tool results for orphaned tool calls
    // This preserves thinking signatures and satisfies API requirements
    let mut result: Vec<Message> = Vec::with_capacity(transformed.len());
    let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
    let mut existing_tool_result_ids: HashSet<String> = HashSet::new();
    // System messages are transparent to tool-call accounting: one that lands between a tool
    // call and its results is held back and emitted after the results (synthetic ones
    // included), so it never causes a duplicate result for a call that is answered later.
    let mut held_system_messages: Vec<Message> = Vec::new();
    let close_pending_tool_calls =
        |result: &mut Vec<Message>,
         pending_tool_calls: &mut Vec<ToolCall>,
         existing_tool_result_ids: &mut HashSet<String>,
         held_system_messages: &mut Vec<Message>| {
            if !pending_tool_calls.is_empty() {
                for tool_call in pending_tool_calls.drain(..) {
                    if !existing_tool_result_ids.contains(&tool_call.id) {
                        result.push(Message::ToolResult(ToolResultMessage {
                            tool_call_id: tool_call.id,
                            tool_name: tool_call.name,
                            content: vec![ToolResultContent::text("No result provided")],
                            details: None,
                            usage: None,
                            nested_calls: None,
                            is_error: true,
                            timestamp: now_millis(),
                        }));
                    }
                }
                existing_tool_result_ids.clear();
            }
            result.append(held_system_messages);
        };

    for message in transformed {
        match message {
            Message::Assistant(assistant) => {
                // If we have pending orphaned tool calls from a previous assistant, insert synthetic results now
                close_pending_tool_calls(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                    &mut held_system_messages,
                );

                // Skip errored/aborted assistant messages entirely.
                // These are incomplete turns that shouldn't be replayed:
                // - May have partial content (reasoning without message, incomplete tool calls)
                // - Replaying them can cause API errors (e.g., OpenAI "reasoning without following item")
                // - The model should retry from the last valid state
                if matches!(
                    assistant.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) {
                    continue;
                }

                // Track tool calls from this assistant message
                let tool_calls: Vec<ToolCall> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                        _ => None,
                    })
                    .collect();
                if !tool_calls.is_empty() {
                    pending_tool_calls = tool_calls;
                    existing_tool_result_ids = HashSet::new();
                }

                result.push(Message::Assistant(assistant));
            }
            Message::ToolResult(tool_result) => {
                existing_tool_result_ids.insert(tool_result.tool_call_id.clone());
                result.push(Message::ToolResult(tool_result));
            }
            Message::System(_) => {
                if !pending_tool_calls.is_empty() {
                    held_system_messages.push(message);
                } else {
                    result.push(message);
                }
            }
            Message::User(_) => {
                // A new user turn interrupts tool flow - insert synthetic results for orphaned calls
                close_pending_tool_calls(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                    &mut held_system_messages,
                );
                result.push(message);
            }
        }
    }

    // If the conversation ends with unresolved tool calls, synthesize results now.
    close_pending_tool_calls(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_tool_result_ids,
        &mut held_system_messages,
    );

    result
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::{ModelCost, SystemMessage, ThinkingContent, UserMessage};

    // Normalize function matching what anthropic-messages uses
    fn anthropic_normalize_tool_call_id(
        id: &str,
        _model: &Model,
        _source: &AssistantMessage,
    ) -> String {
        crate::api::anthropic_messages::normalize_tool_call_id(id)
    }

    fn make_copilot_claude_model() -> Model {
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

    fn assistant(
        api: &str,
        model: &str,
        content: Vec<AssistantContent>,
        stop: StopReason,
    ) -> Message {
        Message::Assistant(AssistantMessage {
            content,
            stop_reason: stop,
            ..AssistantMessage::empty_for(&Model {
                id: model.to_string(),
                api: api.to_string(),
                provider: "github-copilot".to_string(),
                ..Default::default()
            })
        })
    }

    fn tool_call(id: &str, name: &str) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: json!({}),
            thought_signature: None,
            namespace: None,
        })
    }

    fn tool_result(id: &str, name: &str) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: id.to_string(),
            tool_name: name.to_string(),
            content: vec![ToolResultContent::text("done")],
            details: None,
            usage: None,
            nested_calls: None,
            is_error: false,
            timestamp: 1,
        })
    }

    // transform-messages-copilot-openai-to-anthropic.test.ts
    #[test]
    fn converts_thinking_blocks_to_plain_text_when_source_model_differs() {
        let model = make_copilot_claude_model();
        let messages = vec![
            Message::user_text("hello"),
            assistant(
                "openai-completions",
                "gpt-4o",
                vec![
                    AssistantContent::Thinking(ThinkingContent {
                        thinking: "Let me think about this...".to_string(),
                        thinking_signature: Some("reasoning_content".to_string()),
                        redacted: None,
                    }),
                    AssistantContent::text("Hi there!"),
                ],
                StopReason::Stop,
            ),
        ];

        let result = transform_messages(&messages, &model, Some(&anthropic_normalize_tool_call_id));
        let Some(Message::Assistant(assistant)) =
            result.iter().find(|message| message.role() == "assistant")
        else {
            panic!("expected assistant");
        };
        let text_blocks = assistant
            .content
            .iter()
            .filter(|block| matches!(block, AssistantContent::Text(_)))
            .count();
        let thinking_blocks = assistant
            .content
            .iter()
            .filter(|block| matches!(block, AssistantContent::Thinking(_)))
            .count();
        assert_eq!(thinking_blocks, 0);
        assert!(text_blocks >= 2);
    }

    #[test]
    fn removes_thought_signature_from_tool_calls_when_migrating_between_models() {
        let model = make_copilot_claude_model();
        let messages = vec![
            Message::user_text("run a command"),
            assistant(
                "openai-responses",
                "gpt-5",
                vec![AssistantContent::ToolCall(ToolCall {
                    id: "call_123".to_string(),
                    name: "bash".to_string(),
                    arguments: json!({ "command": "ls" }),
                    thought_signature: Some(
                        json!({ "type": "reasoning.encrypted", "id": "call_123", "data": "encrypted" })
                            .to_string(),
                    ),
                    namespace: None,
                })],
                StopReason::ToolUse,
            ),
            tool_result("call_123", "bash"),
        ];

        let result = transform_messages(&messages, &model, Some(&anthropic_normalize_tool_call_id));
        let Some(Message::Assistant(assistant)) =
            result.iter().find(|message| message.role() == "assistant")
        else {
            panic!("expected assistant");
        };
        let Some(AssistantContent::ToolCall(tool_call)) = assistant.content.first() else {
            panic!("expected tool call");
        };
        assert_eq!(tool_call.thought_signature, None);
    }

    #[test]
    fn adds_synthetic_tool_results_for_trailing_orphaned_tool_calls() {
        let model = make_copilot_claude_model();
        let messages = vec![
            Message::user_text("read the file"),
            assistant(
                "openai-responses",
                "gpt-5",
                vec![tool_call("call_123|fc_123", "read")],
                StopReason::ToolUse,
            ),
        ];

        let result = transform_messages(&messages, &model, Some(&anthropic_normalize_tool_call_id));
        let Some(Message::ToolResult(last)) = result.last() else {
            panic!("expected a synthetic tool result");
        };
        assert_eq!(last.tool_call_id, "call_123_fc_123");
        assert_eq!(last.tool_name, "read");
        assert!(last.is_error);
        assert_eq!(
            last.content,
            vec![ToolResultContent::text("No result provided")]
        );
    }

    #[test]
    fn adds_synthetic_results_only_for_trailing_tool_calls_still_missing_results() {
        let model = make_copilot_claude_model();
        let messages = vec![
            Message::user_text("run commands"),
            assistant(
                "openai-responses",
                "gpt-5",
                vec![
                    tool_call("call_1|fc_1", "read"),
                    tool_call("call_2|fc_2", "bash"),
                ],
                StopReason::ToolUse,
            ),
            tool_result("call_1|fc_1", "read"),
        ];

        let result = transform_messages(&messages, &model, Some(&anthropic_normalize_tool_call_id));
        let synthetic: Vec<&ToolResultMessage> = result
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult(result) if result.is_error => Some(result),
                _ => None,
            })
            .collect();
        assert_eq!(synthetic.len(), 1);
        assert_eq!(synthetic[0].tool_call_id, "call_2_fc_2");
        assert_eq!(synthetic[0].tool_name, "bash");
        assert_eq!(
            synthetic[0].content,
            vec![ToolResultContent::text("No result provided")]
        );
    }

    // lax-message-content.test.ts
    #[test]
    fn normalizes_null_or_missing_content_to_an_empty_array() {
        let messages: Vec<Message> = serde_json::from_value(json!([
            { "role": "user", "content": null, "timestamp": 1 },
            {
                "role": "assistant", "content": null, "api": "openai-completions",
                "provider": "openai", "model": "test-model",
                "usage": {
                    "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                    "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 }
                },
                "stopReason": "stop", "timestamp": 2
            },
            { "role": "toolResult", "toolCallId": "call_1", "toolName": "web_search", "isError": false, "timestamp": 3 }
        ]))
        .unwrap();
        let model = Model {
            id: "test-model".to_string(),
            api: "openai-completions".to_string(),
            provider: "openai".to_string(),
            input: vec![ModelInput::Text],
            ..Default::default()
        };

        let result = transform_messages(&messages, &model, None);
        assert_eq!(result.len(), 3);
        for message in &result {
            let content = serde_json::to_value(message).unwrap()["content"].clone();
            assert_eq!(content, json!([]));
        }
    }

    #[test]
    fn downgrades_images_for_text_only_models_and_holds_system_messages_after_results() {
        let model = Model {
            id: "m".to_string(),
            api: "anthropic-messages".to_string(),
            provider: "p".to_string(),
            input: vec![ModelInput::Text],
            ..Default::default()
        };
        let image = UserContent::Image(crate::types::ImageContent {
            data: "abc".to_string(),
            mime_type: "image/png".to_string(),
        });
        let messages = vec![
            Message::User(UserMessage {
                content: UserMessageContent::Parts(vec![image.clone(), image]),
                timestamp: 1,
            }),
            assistant(
                "other",
                "x",
                vec![tool_call("call_1", "read")],
                StopReason::ToolUse,
            ),
            Message::System(SystemMessage {
                content: "update".into(),
                timestamp: 2,
                ..Default::default()
            }),
            tool_result("call_1", "read"),
        ];
        let result = transform_messages(&messages, &model, None);
        let Message::User(user) = &result[0] else {
            panic!("expected user");
        };
        assert_eq!(
            user.content,
            UserMessageContent::Parts(vec![UserContent::text(NON_VISION_USER_IMAGE_PLACEHOLDER)])
        );
        assert_eq!(
            result.iter().map(Message::role).collect::<Vec<_>>(),
            vec!["user", "assistant", "toolResult", "system"]
        );
    }
}
