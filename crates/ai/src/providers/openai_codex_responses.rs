use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use crate::models::clamp_thinking_level;
use crate::providers::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::providers::openai_responses::{
    OpenAIResponsesOptions, convert_responses_messages, convert_responses_tools,
    stream_openai_responses,
};
use crate::providers::simple_options::build_base_options;
use crate::types::{Context, Model, ModelThinkingLevel, SimpleStreamOptions, StreamOptions};

const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

#[derive(Clone, Default)]
pub struct OpenAICodexResponsesOptions {
    pub base: StreamOptions,
    pub reasoning_effort: Option<ModelThinkingLevel>,
    pub reasoning_summary: Option<Option<String>>,
    pub service_tier: Option<String>,
    pub text_verbosity: Option<String>,
}

pub fn stream_simple_openai_codex_responses(
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
) -> crate::AssistantMessageEventStream {
    let api_key = options
        .stream
        .api_key
        .clone()
        .filter(|key| !key.trim().is_empty());
    let Some(api_key) = api_key else {
        return immediate_error(model, "No API key for provider");
    };
    let base = build_base_options(&model, &options, api_key);
    let reasoning_effort = options.reasoning.and_then(|reasoning| {
        let clamped = clamp_thinking_level(&model, reasoning);
        (clamped != ModelThinkingLevel::Off).then_some(clamped)
    });

    stream_openai_codex_responses(
        model,
        context,
        OpenAICodexResponsesOptions {
            base,
            reasoning_effort,
            reasoning_summary: None,
            service_tier: None,
            text_verbosity: None,
        },
    )
}

pub fn stream_openai_codex_responses(
    model: Model,
    context: Context,
    mut options: OpenAICodexResponsesOptions,
) -> crate::AssistantMessageEventStream {
    let api_key = options
        .base
        .api_key
        .clone()
        .filter(|key| !key.trim().is_empty());
    let Some(api_key) = api_key else {
        return immediate_error(model, "No API key for provider");
    };
    let account_id = match extract_account_id(&api_key) {
        Ok(account_id) => account_id,
        Err(error) => return immediate_error(model, &error),
    };

    let request_url = resolve_codex_url(&model.base_url);
    let payload = build_request_body(&model, &context, &options);
    insert_codex_headers(
        &mut options.base.headers,
        &model.headers,
        &account_id,
        options.base.session_id.as_deref(),
    );

    stream_openai_responses(
        model,
        context,
        OpenAIResponsesOptions {
            base: options.base,
            reasoning_effort: None,
            reasoning_summary: None,
            service_tier: options.service_tier,
            request_url: Some(request_url),
            request_model: None,
            payload_override: Some(payload),
            include_store: Some(false),
            ..Default::default()
        },
    )
}

fn build_request_body(
    model: &Model,
    context: &Context,
    options: &OpenAICodexResponsesOptions,
) -> Value {
    let allowed_tool_call_providers = ["openai", "openai-codex", "opencode"]
        .into_iter()
        .collect::<HashSet<_>>();
    let messages = convert_responses_messages(model, context, &allowed_tool_call_providers, false);
    let mut body = json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": context.system_prompt.as_deref().unwrap_or("You are a helpful assistant."),
        "input": messages,
        "text": { "verbosity": options.text_verbosity.as_deref().unwrap_or("low") },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true
    });
    let object = body.as_object_mut().expect("codex request body object");

    if let Some(session_id) = &options.base.session_id {
        object.insert(
            "prompt_cache_key".to_string(),
            json!(clamp_openai_prompt_cache_key(Some(session_id))),
        );
    }
    if let Some(temperature) = options.base.temperature {
        object.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(service_tier) = &options.service_tier {
        object.insert("service_tier".to_string(), json!(service_tier));
    }
    if !context.tools.is_empty() {
        object.insert(
            "tools".to_string(),
            Value::Array(convert_responses_tools(&context.tools, None)),
        );
    }
    if let Some(reasoning_effort) = options.reasoning_effort {
        let effort = if reasoning_effort == ModelThinkingLevel::Off {
            model
                .thinking_level_map
                .get("off")
                .cloned()
                .flatten()
                .unwrap_or_else(|| "none".to_string())
        } else {
            model
                .thinking_level_map
                .get(reasoning_effort.as_str())
                .cloned()
                .flatten()
                .unwrap_or_else(|| reasoning_effort.as_str().to_string())
        };
        object.insert(
            "reasoning".to_string(),
            json!({
                "effort": effort,
                "summary": options.reasoning_summary.clone().flatten().unwrap_or_else(|| "auto".to_string())
            }),
        );
    }

    body
}

fn resolve_codex_url(base_url: &str) -> String {
    let raw = if base_url.trim().is_empty() {
        DEFAULT_CODEX_BASE_URL
    } else {
        base_url.trim()
    };
    let normalized = raw.trim_end_matches('/');
    if normalized.ends_with("/codex/responses") {
        normalized.to_string()
    } else if normalized.ends_with("/codex") {
        format!("{normalized}/responses")
    } else {
        format!("{normalized}/codex/responses")
    }
}

fn insert_codex_headers(
    headers: &mut HashMap<String, String>,
    model_headers: &HashMap<String, String>,
    account_id: &str,
    session_id: Option<&str>,
) {
    for (key, value) in model_headers {
        headers.entry(key.clone()).or_insert_with(|| value.clone());
    }
    headers.insert("chatgpt-account-id".to_string(), account_id.to_string());
    headers.insert("originator".to_string(), "pi".to_string());
    headers.insert("User-Agent".to_string(), user_agent());
    headers.insert(
        "OpenAI-Beta".to_string(),
        "responses=experimental".to_string(),
    );
    headers.insert("accept".to_string(), "text/event-stream".to_string());
    if let Some(session_id) = session_id {
        headers.insert("session-id".to_string(), session_id.to_string());
        headers.insert("x-client-request-id".to_string(), session_id.to_string());
    }
}

fn user_agent() -> String {
    format!(
        "pi ({} {}; {})",
        std::env::consts::OS,
        std::env::consts::FAMILY,
        std::env::consts::ARCH
    )
}

fn extract_account_id(token: &str) -> Result<String, String> {
    let mut parts = token.split('.');
    let _header = parts.next();
    let Some(payload) = parts.next() else {
        return Err("Failed to extract accountId from token".to_string());
    };
    if parts.next().is_none() {
        return Err("Failed to extract accountId from token".to_string());
    }
    let bytes = decode_base64_url(payload)
        .ok_or_else(|| "Failed to extract accountId from token".to_string())?;
    let parsed: Value = serde_json::from_slice(&bytes)
        .map_err(|_| "Failed to extract accountId from token".to_string())?;
    parsed
        .get(JWT_CLAIM_PATH)
        .and_then(|claim| claim.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| "Failed to extract accountId from token".to_string())
}

fn decode_base64_url(input: &str) -> Option<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0u8;
    for byte in input.bytes() {
        if byte == b'=' {
            break;
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        } as u32;
        buffer = (buffer << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push(((buffer >> bits) & 0xff) as u8);
        }
    }
    Some(output)
}

fn immediate_error(model: Model, message: &str) -> crate::AssistantMessageEventStream {
    let (mut sender, stream) = crate::AssistantMessageEventStream::channel();
    let mut output = crate::AssistantMessage::empty_for(&model);
    output.stop_reason = crate::StopReason::Error;
    output.error_message = Some(format!("{message}: {}", model.provider));
    sender.push(crate::AssistantMessageEvent::Error {
        reason: crate::StopReason::Error,
        error: output,
    });
    stream
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, ModelCost, ModelInput, Tool};

    fn model() -> Model {
        Model {
            id: "gpt-5.5".to_string(),
            name: "GPT 5.5".to_string(),
            api: "openai-codex-responses".to_string(),
            provider: "openai-codex".to_string(),
            base_url: "https://chatgpt.com/backend-api".to_string(),
            reasoning: true,
            input: vec![ModelInput::Text, ModelInput::Image],
            cost: ModelCost::default(),
            context_window: 1_000_000,
            max_tokens: 128_000,
            ..Default::default()
        }
    }

    #[test]
    fn builds_codex_request_body_with_instructions_and_cache_key() {
        let mut options = OpenAICodexResponsesOptions::default();
        options.base.session_id = Some("session-abc".to_string());
        options.reasoning_effort = Some(ModelThinkingLevel::Low);
        options.service_tier = Some("priority".to_string());
        let context = Context {
            system_prompt: Some("Be terse.".to_string()),
            messages: vec![Message::user_text("hello")],
            tools: vec![Tool {
                name: "read".to_string(),
                description: "Read".to_string(),
                parameters: json!({ "type": "object" }),
            }],
        };

        let body = build_request_body(&model(), &context, &options);

        assert_eq!(body["instructions"], json!("Be terse."));
        assert_eq!(body["input"][0]["role"], json!("user"));
        assert_eq!(body["prompt_cache_key"], json!("session-abc"));
        assert_eq!(body["tool_choice"], json!("auto"));
        assert_eq!(body["parallel_tool_calls"], json!(true));
        assert_eq!(body["reasoning"]["effort"], json!("low"));
        assert_eq!(body["service_tier"], json!("priority"));
        assert_eq!(body["tools"][0]["type"], json!("function"));
    }

    #[test]
    fn resolves_codex_url_shapes() {
        assert_eq!(
            resolve_codex_url("https://chatgpt.com/backend-api"),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url("https://chatgpt.com/backend-api/codex"),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url("https://chatgpt.com/backend-api/codex/responses"),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn extracts_account_id_from_jwt_payload() {
        let payload = "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdC0xMjMifX0";
        let token = format!("header.{payload}.sig");
        assert_eq!(extract_account_id(&token).as_deref(), Ok("acct-123"));
    }
}
