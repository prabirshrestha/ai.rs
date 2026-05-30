use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::types::{
    AssistantContent, Context, ImageContent, Message, Model, ModelInput, StopReason, TextContent,
    Tool, ToolResultContent, UserContent, UserMessageContent,
};
use crate::utils::sanitize::sanitize_surrogates;
use crate::utils::transform_messages::transform_messages;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoogleThinkingLevel {
    ThinkingLevelUnspecified,
    Minimal,
    Low,
    Medium,
    High,
}

impl GoogleThinkingLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ThinkingLevelUnspecified => "THINKING_LEVEL_UNSPECIFIED",
            Self::Minimal => "MINIMAL",
            Self::Low => "LOW",
            Self::Medium => "MEDIUM",
            Self::High => "HIGH",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleContent {
    pub role: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<GooglePart>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GooglePart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<GoogleInlineData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<GoogleFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<GoogleFunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleInlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleFunctionResponse {
    pub name: String,
    pub response: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<GooglePart>,
}

pub fn is_thinking_part(part: &GooglePart) -> bool {
    part.thought == Some(true)
}

pub fn retain_thought_signature(
    existing: Option<String>,
    incoming: Option<&str>,
) -> Option<String> {
    incoming
        .filter(|signature| !signature.is_empty())
        .map(ToString::to_string)
        .or(existing)
}

pub fn requires_tool_call_id(model_id: &str) -> bool {
    model_id.starts_with("claude-") || model_id.starts_with("gpt-oss-")
}

pub fn convert_messages(model: &Model, context: &Context) -> Vec<GoogleContent> {
    let mut contents: Vec<GoogleContent> = Vec::new();
    let normalize_tool_call_id =
        |id: &str, model: &Model, _source: &crate::types::AssistantMessage| {
            if !requires_tool_call_id(&model.id) {
                return id.to_string();
            }
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
        };

    for message in transform_messages(&context.messages, model, normalize_tool_call_id) {
        match message {
            Message::User(user) => match user.content {
                UserMessageContent::Text(text) => contents.push(GoogleContent {
                    role: "user".to_string(),
                    parts: vec![GooglePart {
                        text: Some(sanitize_surrogates(&text)),
                        ..Default::default()
                    }],
                }),
                UserMessageContent::Parts(parts) => {
                    let parts = parts
                        .into_iter()
                        .map(|part| match part {
                            UserContent::Text(text) => GooglePart {
                                text: Some(sanitize_surrogates(&text.text)),
                                ..Default::default()
                            },
                            UserContent::Image(image) => GooglePart {
                                inline_data: Some(GoogleInlineData {
                                    mime_type: image.mime_type,
                                    data: image.data,
                                }),
                                ..Default::default()
                            },
                        })
                        .collect::<Vec<_>>();
                    if !parts.is_empty() {
                        contents.push(GoogleContent {
                            role: "user".to_string(),
                            parts,
                        });
                    }
                }
            },
            Message::Assistant(assistant) => {
                let is_same_provider_and_model =
                    assistant.provider == model.provider && assistant.model == model.id;
                let mut parts = Vec::new();

                for block in assistant.content {
                    match block {
                        AssistantContent::Text(text) => {
                            if text.text.trim().is_empty() {
                                continue;
                            }
                            parts.push(GooglePart {
                                text: Some(sanitize_surrogates(&text.text)),
                                thought_signature: resolve_thought_signature(
                                    is_same_provider_and_model,
                                    text.text_signature.as_deref(),
                                ),
                                ..Default::default()
                            });
                        }
                        AssistantContent::Thinking(thinking) => {
                            if thinking.thinking.trim().is_empty() {
                                continue;
                            }
                            if is_same_provider_and_model {
                                parts.push(GooglePart {
                                    thought: Some(true),
                                    text: Some(sanitize_surrogates(&thinking.thinking)),
                                    thought_signature: resolve_thought_signature(
                                        is_same_provider_and_model,
                                        thinking.thinking_signature.as_deref(),
                                    ),
                                    ..Default::default()
                                });
                            } else {
                                parts.push(GooglePart {
                                    text: Some(sanitize_surrogates(&thinking.thinking)),
                                    ..Default::default()
                                });
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            parts.push(GooglePart {
                                function_call: Some(GoogleFunctionCall {
                                    name: Some(tool_call.name),
                                    args: Some(tool_call.arguments),
                                    id: requires_tool_call_id(&model.id).then_some(tool_call.id),
                                }),
                                thought_signature: resolve_thought_signature(
                                    is_same_provider_and_model,
                                    tool_call.thought_signature.as_deref(),
                                ),
                                ..Default::default()
                            });
                        }
                    }
                }

                if !parts.is_empty() {
                    contents.push(GoogleContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
            }
            Message::ToolResult(tool_result) => {
                let text_result = tool_result
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        ToolResultContent::Text(TextContent { text, .. }) => Some(text.as_str()),
                        ToolResultContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let image_content = if model.input.contains(&ModelInput::Image) {
                    tool_result
                        .content
                        .iter()
                        .filter_map(|content| match content {
                            ToolResultContent::Image(image) => Some(image),
                            ToolResultContent::Text(_) => None,
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };

                let has_text = !text_result.is_empty();
                let has_images = !image_content.is_empty();
                let model_supports_multimodal_function_response =
                    supports_multimodal_function_response(&model.id);
                let response_value = if has_text {
                    sanitize_surrogates(&text_result)
                } else if has_images {
                    "(see attached image)".to_string()
                } else {
                    String::new()
                };
                let image_parts = image_content
                    .iter()
                    .map(|image| google_image_part(image))
                    .collect::<Vec<_>>();

                let function_response_part = GooglePart {
                    function_response: Some(GoogleFunctionResponse {
                        name: tool_result.tool_name,
                        response: if tool_result.is_error {
                            json!({ "error": response_value })
                        } else {
                            json!({ "output": response_value })
                        },
                        id: requires_tool_call_id(&model.id).then_some(tool_result.tool_call_id),
                        parts: if has_images && model_supports_multimodal_function_response {
                            image_parts.clone()
                        } else {
                            Vec::new()
                        },
                    }),
                    ..Default::default()
                };

                let mut merged = false;
                if let Some(last) = contents.last_mut() {
                    if last.role == "user"
                        && last
                            .parts
                            .iter()
                            .any(|part| part.function_response.is_some())
                    {
                        last.parts.push(function_response_part.clone());
                        merged = true;
                    }
                }
                if !merged {
                    contents.push(GoogleContent {
                        role: "user".to_string(),
                        parts: vec![function_response_part],
                    });
                }

                if has_images && !model_supports_multimodal_function_response {
                    let mut parts = vec![GooglePart {
                        text: Some("Tool result image:".to_string()),
                        ..Default::default()
                    }];
                    parts.extend(image_parts);
                    contents.push(GoogleContent {
                        role: "user".to_string(),
                        parts,
                    });
                }
            }
        }
    }

    contents
}

pub fn convert_tools(tools: &[Tool], use_parameters: bool) -> Option<Vec<Value>> {
    if tools.is_empty() {
        return None;
    }

    let function_declarations = tools
        .iter()
        .map(|tool| {
            let mut declaration = Map::new();
            declaration.insert("name".to_string(), Value::String(tool.name.clone()));
            declaration.insert(
                "description".to_string(),
                Value::String(tool.description.clone()),
            );
            if use_parameters {
                declaration.insert(
                    "parameters".to_string(),
                    sanitize_for_openapi(&tool.parameters),
                );
            } else {
                declaration.insert("parametersJsonSchema".to_string(), tool.parameters.clone());
            }
            Value::Object(declaration)
        })
        .collect::<Vec<_>>();

    Some(vec![
        json!({ "functionDeclarations": function_declarations }),
    ])
}

pub fn map_tool_choice(choice: &str) -> &'static str {
    match choice {
        "auto" => "AUTO",
        "none" => "NONE",
        "any" => "ANY",
        _ => "AUTO",
    }
}

pub fn map_stop_reason_string(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::Stop,
        "MAX_TOKENS" => StopReason::Length,
        _ => StopReason::Error,
    }
}

fn google_image_part(image: &ImageContent) -> GooglePart {
    GooglePart {
        inline_data: Some(GoogleInlineData {
            mime_type: image.mime_type.clone(),
            data: image.data.clone(),
        }),
        ..Default::default()
    }
}

fn resolve_thought_signature(
    is_same_provider_and_model: bool,
    signature: Option<&str>,
) -> Option<String> {
    if is_same_provider_and_model && is_valid_thought_signature(signature) {
        signature.map(ToString::to_string)
    } else {
        None
    }
}

fn is_valid_thought_signature(signature: Option<&str>) -> bool {
    let Some(signature) = signature else {
        return false;
    };
    if signature.is_empty() || signature.len() % 4 != 0 {
        return false;
    }
    signature
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '+' || ch == '/' || ch == '=')
}

fn get_gemini_major_version(model_id: &str) -> Option<u32> {
    let lower = model_id.to_lowercase();
    let rest = lower
        .strip_prefix("gemini-live-")
        .or_else(|| lower.strip_prefix("gemini-"))?;
    let major = rest.split('-').next()?.split('.').next()?;
    major.parse().ok()
}

fn supports_multimodal_function_response(model_id: &str) -> bool {
    get_gemini_major_version(model_id)
        .map(|version| version >= 3)
        .unwrap_or(true)
}

fn sanitize_for_openapi(schema: &Value) -> Value {
    let Value::Object(object) = schema else {
        return schema.clone();
    };

    let mut result = Map::new();
    for (key, value) in object {
        if is_json_schema_meta_declaration(key) {
            continue;
        }
        result.insert(key.clone(), sanitize_for_openapi(value));
    }
    Value::Object(result)
}

fn is_json_schema_meta_declaration(key: &str) -> bool {
    matches!(
        key,
        "$schema"
            | "$id"
            | "$anchor"
            | "$dynamicAnchor"
            | "$vocabulary"
            | "$comment"
            | "$defs"
            | "definitions"
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::{
        AssistantMessage, ModelCost, ToolCall, ToolResultMessage, Usage, UserMessage,
    };

    fn make_model(api: &str, provider: &str, id: &str) -> Model {
        Model {
            id: id.to_string(),
            name: id.to_string(),
            api: api.to_string(),
            provider: provider.to_string(),
            base_url: "https://example.invalid".to_string(),
            reasoning: true,
            input: vec![ModelInput::Text, ModelInput::Image],
            cost: ModelCost::default(),
            context_window: 1_000_000,
            max_tokens: 8192,
            ..Default::default()
        }
    }

    fn make_tool(parameters: Value) -> Tool {
        Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters,
        }
    }

    #[test]
    fn thinking_detection_uses_thought_flag_not_signature() {
        assert!(is_thinking_part(&GooglePart {
            thought: Some(true),
            thought_signature: Some("opaque-signature".to_string()),
            ..Default::default()
        }));
        assert!(!is_thinking_part(&GooglePart {
            thought_signature: Some("opaque-signature".to_string()),
            ..Default::default()
        }));
        assert!(!is_thinking_part(&GooglePart {
            thought: Some(false),
            thought_signature: Some("opaque-signature".to_string()),
            ..Default::default()
        }));
    }

    #[test]
    fn retain_thought_signature_preserves_existing_when_incoming_is_empty() {
        let first = retain_thought_signature(None, Some("sig-1"));
        assert_eq!(first.as_deref(), Some("sig-1"));
        let second = retain_thought_signature(first, None);
        assert_eq!(second.as_deref(), Some("sig-1"));
        let third = retain_thought_signature(second, Some(""));
        assert_eq!(third.as_deref(), Some("sig-1"));
        let updated = retain_thought_signature(third, Some("sig-2"));
        assert_eq!(updated.as_deref(), Some("sig-2"));
    }

    #[test]
    fn convert_tools_strips_json_schema_meta_keys_when_using_parameters() {
        let tools = [make_tool(json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "urn:bash-tool",
            "$comment": "A bash tool for demonstration",
            "$defs": { "commandDef": { "type": "string" } },
            "definitions": { "legacyDef": { "type": "number" } },
            "type": "object",
            "properties": {
                "command": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$id": "urn:nested",
                    "type": "string"
                }
            },
            "required": ["command"]
        }))];

        let converted = convert_tools(&tools, true).expect("tools");
        let declaration = &converted[0]["functionDeclarations"][0];
        assert_eq!(
            declaration["parameters"],
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            })
        );
    }

    #[test]
    fn convert_tools_preserves_schema_for_parameters_json_schema() {
        let parameters = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"]
        });
        let tools = [make_tool(parameters.clone())];

        let converted = convert_tools(&tools, false).expect("tools");
        let declaration = &converted[0]["functionDeclarations"][0];
        assert_eq!(declaration["parametersJsonSchema"], parameters);
    }

    #[test]
    fn gemini3_tool_result_images_stay_inside_function_response() {
        let model = make_model("google-generative-ai", "google", "gemini-3-pro-preview");
        let context = Context {
            messages: vec![Message::ToolResult(ToolResultMessage {
                tool_call_id: "call-1".to_string(),
                tool_name: "inspect".to_string(),
                content: vec![
                    ToolResultContent::text("done"),
                    ToolResultContent::Image(ImageContent {
                        mime_type: "image/png".to_string(),
                        data: "abc".to_string(),
                    }),
                ],
                details: None,
                is_error: false,
                timestamp: 1,
            })],
            ..Default::default()
        };

        let contents = convert_messages(&model, &context);
        assert_eq!(contents.len(), 1);
        let function_response = contents[0].parts[0]
            .function_response
            .as_ref()
            .expect("function response");
        assert_eq!(function_response.parts.len(), 1);
        assert_eq!(
            function_response.parts[0]
                .inline_data
                .as_ref()
                .unwrap()
                .mime_type,
            "image/png"
        );
    }

    #[test]
    fn gemini2_tool_result_images_are_sent_as_separate_user_turn() {
        let model = make_model("google-generative-ai", "google", "gemini-2.5-flash");
        let context = Context {
            messages: vec![Message::ToolResult(ToolResultMessage {
                tool_call_id: "call-1".to_string(),
                tool_name: "inspect".to_string(),
                content: vec![ToolResultContent::Image(ImageContent {
                    mime_type: "image/png".to_string(),
                    data: "abc".to_string(),
                })],
                details: None,
                is_error: false,
                timestamp: 1,
            })],
            ..Default::default()
        };

        let contents = convert_messages(&model, &context);
        assert_eq!(contents.len(), 2);
        assert_eq!(
            contents[0].parts[0]
                .function_response
                .as_ref()
                .unwrap()
                .parts,
            Vec::<GooglePart>::new()
        );
        assert_eq!(
            contents[1].parts[0].text.as_deref(),
            Some("Tool result image:")
        );
        assert!(contents[1].parts[1].inline_data.is_some());
    }

    #[test]
    fn same_google_model_preserves_valid_tool_call_thought_signature() {
        let model = make_model("google-generative-ai", "google", "gemini-3-pro-preview");
        let valid_signature = "YWJjZA==";
        let context = Context {
            messages: vec![Message::Assistant(AssistantMessage {
                content: vec![AssistantContent::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "lookup".to_string(),
                    arguments: json!({ "q": "rust" }),
                    thought_signature: Some(valid_signature.to_string()),
                })],
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                diagnostics: Vec::new(),
                usage: Usage::default(),
                stop_reason: StopReason::ToolUse,
                error_message: None,
                timestamp: 1,
            })],
            ..Default::default()
        };

        let contents = convert_messages(&model, &context);
        assert_eq!(
            contents[0].parts[0].thought_signature.as_deref(),
            Some(valid_signature)
        );
        assert!(
            contents[0].parts[0]
                .function_call
                .as_ref()
                .unwrap()
                .id
                .is_none()
        );
    }

    #[test]
    fn claude_models_normalize_tool_call_ids_and_include_them() {
        let model = make_model("google-generative-ai", "google", "claude-sonnet-4");
        let source = make_model("openai-responses", "openai", "gpt-5");
        let context = Context {
            messages: vec![
                Message::Assistant(AssistantMessage {
                    content: vec![AssistantContent::ToolCall(ToolCall {
                        id:
                            "call|with*bad/chars-and-a-long-long-long-long-long-long-long-long-tail"
                                .to_string(),
                        name: "lookup".to_string(),
                        arguments: json!({}),
                        thought_signature: Some("YWJjZA==".to_string()),
                    })],
                    api: source.api,
                    provider: source.provider,
                    model: source.id,
                    response_model: None,
                    response_id: None,
                    diagnostics: Vec::new(),
                    usage: Usage::default(),
                    stop_reason: StopReason::ToolUse,
                    error_message: None,
                    timestamp: 1,
                }),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id:
                        "call|with*bad/chars-and-a-long-long-long-long-long-long-long-long-tail"
                            .to_string(),
                    tool_name: "lookup".to_string(),
                    content: vec![ToolResultContent::text("ok")],
                    details: None,
                    is_error: false,
                    timestamp: 2,
                }),
            ],
            ..Default::default()
        };

        let contents = convert_messages(&model, &context);
        let call_id = contents[0].parts[0]
            .function_call
            .as_ref()
            .unwrap()
            .id
            .as_ref()
            .unwrap();
        assert_eq!(call_id.len(), 64);
        assert!(
            call_id
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
        );
        assert!(contents[0].parts[0].thought_signature.is_none());
        assert_eq!(
            contents[1].parts[0]
                .function_response
                .as_ref()
                .unwrap()
                .id
                .as_deref(),
            Some(call_id.as_str())
        );
    }

    #[test]
    fn user_messages_convert_text_and_images() {
        let model = make_model("google-generative-ai", "google", "gemini-3-flash-preview");
        let context = Context {
            messages: vec![Message::User(UserMessage {
                content: UserMessageContent::Parts(vec![
                    UserContent::text("hello"),
                    UserContent::Image(ImageContent {
                        mime_type: "image/jpeg".to_string(),
                        data: "image-data".to_string(),
                    }),
                ]),
                timestamp: 1,
            })],
            ..Default::default()
        };

        let contents = convert_messages(&model, &context);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("hello"));
        assert_eq!(
            contents[0].parts[1].inline_data.as_ref().unwrap().mime_type,
            "image/jpeg"
        );
    }
}
