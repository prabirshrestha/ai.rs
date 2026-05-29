use crate::models::clamp_thinking_level;
use crate::providers::openai_responses::{
    OpenAIResponsesAuthHeader, OpenAIResponsesOptions, stream_openai_responses,
};
use crate::providers::simple_options::build_base_options;
use crate::types::{Context, Model, ModelThinkingLevel, SimpleStreamOptions, StreamOptions};

const DEFAULT_AZURE_API_VERSION: &str = "v1";

#[derive(Clone, Default)]
pub struct AzureOpenAIResponsesOptions {
    pub base: StreamOptions,
    pub reasoning_effort: Option<ModelThinkingLevel>,
    pub reasoning_summary: Option<Option<String>>,
    pub azure_api_version: Option<String>,
    pub azure_resource_name: Option<String>,
    pub azure_base_url: Option<String>,
    pub azure_deployment_name: Option<String>,
}

pub fn stream_simple_azure_openai_responses(
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

    stream_azure_openai_responses(
        model,
        context,
        AzureOpenAIResponsesOptions {
            base,
            reasoning_effort,
            reasoning_summary: None,
            ..Default::default()
        },
    )
}

pub fn stream_azure_openai_responses(
    model: Model,
    context: Context,
    options: AzureOpenAIResponsesOptions,
) -> crate::AssistantMessageEventStream {
    let deployment_name = resolve_deployment_name(&model, &options);
    let config = match resolve_azure_config(&model, &options) {
        Ok(config) => config,
        Err(error) => return immediate_error(model, &error),
    };

    stream_openai_responses(
        model,
        context,
        OpenAIResponsesOptions {
            base: options.base,
            reasoning_effort: options.reasoning_effort,
            reasoning_summary: options.reasoning_summary,
            service_tier: None,
            request_url: Some(format!(
                "{}/responses?api-version={}",
                trim_end_slash(&config.base_url),
                config.api_version
            )),
            request_model: Some(deployment_name),
            payload_override: None,
            include_store: Some(false),
            auth_header: OpenAIResponsesAuthHeader::ApiKey,
        },
    )
}

struct AzureConfig {
    base_url: String,
    api_version: String,
}

fn resolve_deployment_name(model: &Model, options: &AzureOpenAIResponsesOptions) -> String {
    if let Some(name) = options
        .azure_deployment_name
        .as_deref()
        .filter(|name| !name.is_empty())
    {
        return name.to_string();
    }
    if let Some(mapped) = parse_deployment_name_map(
        std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP")
            .ok()
            .as_deref(),
    )
    .get(&model.id)
    {
        return mapped.clone();
    }
    model.id.clone()
}

fn parse_deployment_name_map(value: Option<&str>) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let Some(value) = value else {
        return map;
    };
    for entry in value.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((model_id, deployment_name)) = trimmed.split_once('=') else {
            continue;
        };
        let model_id = model_id.trim();
        let deployment_name = deployment_name.trim();
        if !model_id.is_empty() && !deployment_name.is_empty() {
            map.insert(model_id.to_string(), deployment_name.to_string());
        }
    }
    map
}

fn resolve_azure_config(
    model: &Model,
    options: &AzureOpenAIResponsesOptions,
) -> Result<AzureConfig, String> {
    let api_version = options
        .azure_api_version
        .clone()
        .or_else(|| std::env::var("AZURE_OPENAI_API_VERSION").ok())
        .unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.to_string());
    let base_url = options
        .azure_base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            std::env::var("AZURE_OPENAI_BASE_URL")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .or_else(|| {
            options
                .azure_resource_name
                .clone()
                .or_else(|| std::env::var("AZURE_OPENAI_RESOURCE_NAME").ok())
                .filter(|value| !value.is_empty())
                .map(|resource| format!("https://{resource}.openai.azure.com/openai/v1"))
        })
        .or_else(|| (!model.base_url.trim().is_empty()).then(|| model.base_url.clone()))
        .ok_or_else(|| {
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azure_base_url, azure_resource_name, or model.base_url.".to_string()
        })?;

    Ok(AzureConfig {
        base_url: normalize_azure_base_url(&base_url)?,
        api_version,
    })
}

fn normalize_azure_base_url(base_url: &str) -> Result<String, String> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let mut url = reqwest::Url::parse(trimmed)
        .map_err(|_| format!("Invalid Azure OpenAI base URL: {base_url}"))?;
    let host = url.host_str().unwrap_or_default();
    let is_azure_host =
        host.ends_with(".openai.azure.com") || host.ends_with(".cognitiveservices.azure.com");
    let normalized_path = url.path().trim_end_matches('/');
    if is_azure_host
        && (normalized_path.is_empty() || normalized_path == "/" || normalized_path == "/openai")
    {
        url.set_path("/openai/v1");
        url.set_query(None);
    }
    Ok(url.to_string().trim_end_matches('/').to_string())
}

fn trim_end_slash(url: &str) -> &str {
    url.trim_end_matches('/')
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
    use serde_json::json;

    use super::*;
    use crate::providers::openai_responses::{
        OpenAIResponsesAuthHeader, build_responses_payload, get_compat,
    };
    use crate::types::{CacheRetention, Message, ModelCost, ModelInput};

    fn model() -> Model {
        Model {
            id: "gpt-4o-mini".to_string(),
            name: "GPT 4o mini".to_string(),
            api: "azure-openai-responses".to_string(),
            provider: "azure-openai-responses".to_string(),
            base_url: String::new(),
            reasoning: false,
            input: vec![ModelInput::Text, ModelInput::Image],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 16_384,
            ..Default::default()
        }
    }

    #[test]
    fn normalizes_azure_resource_base_url() {
        assert_eq!(
            normalize_azure_base_url("https://example.openai.azure.com").unwrap(),
            "https://example.openai.azure.com/openai/v1"
        );
        assert_eq!(
            normalize_azure_base_url("https://example.openai.azure.com/openai").unwrap(),
            "https://example.openai.azure.com/openai/v1"
        );
        assert_eq!(
            normalize_azure_base_url("https://example.openai.azure.com/custom").unwrap(),
            "https://example.openai.azure.com/custom"
        );
    }

    #[test]
    fn parses_deployment_name_map() {
        let map = parse_deployment_name_map(Some("gpt-4o-mini=my-mini, bad, gpt-5 = prod-gpt5"));
        assert_eq!(map.get("gpt-4o-mini").map(String::as_str), Some("my-mini"));
        assert_eq!(map.get("gpt-5").map(String::as_str), Some("prod-gpt5"));
        assert!(!map.contains_key("bad"));
    }

    #[test]
    fn builds_azure_responses_payload_with_deployment_model_and_without_store() {
        let model = model();
        let options = OpenAIResponsesOptions {
            request_model: Some("my-deployment".to_string()),
            include_store: Some(false),
            auth_header: OpenAIResponsesAuthHeader::ApiKey,
            ..Default::default()
        };
        let context = Context {
            system_prompt: Some("sys".to_string()),
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
        };
        let payload = build_responses_payload(
            &model,
            &context,
            &options,
            &get_compat(&model),
            CacheRetention::Short,
        );

        assert_eq!(payload["model"], json!("my-deployment"));
        assert!(payload.get("store").is_none());
        assert_eq!(payload["input"][0]["role"], json!("system"));
    }
}
