//! Port of `api/openrouter-images.ts` (plus `openrouter-images.lazy.ts`):
//! image generation over OpenRouter's chat completions endpoint.
//!
//! Divergences: the OpenAI SDK client is replaced by
//! `OpenAIClient` (one non-streaming
//! POST per attempt, retried by `retry_provider_request`); an HTTP error
//! message is `"<status> <body>"` rather than the SDK's text.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::openai_client::{OpenAIClient, OpenAIRequestOptions};
use crate::types::{
    AssistantImages, ImageContent, ImageModel, ImagesContext, ImagesOptions, ImagesStopReason,
    ModelOutput, ProviderHeaders, ProviderImages, ProviderResponse, TextContent, Usage, UsageCost,
    UserContent,
};
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::headers::{headers_to_record, provider_headers_to_record};
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::{Error, Result};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenRouterImageUrl {
    Url(String),
    Object { url: Option<String> },
}

#[derive(Debug, Deserialize)]
struct OpenRouterGeneratedImage {
    image_url: Option<OpenRouterImageUrl>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenRouterImageGenerationMessage {
    content: Option<Value>,
    #[serde(default)]
    images: Option<Vec<OpenRouterGeneratedImage>>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterImageGenerationChoice {
    #[serde(default)]
    message: OpenRouterImageGenerationMessage,
}

#[derive(Debug, Default, Deserialize)]
struct PromptTokensDetails {
    cached_tokens: Option<f64>,
    cache_write_tokens: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct RawUsage {
    prompt_tokens: Option<f64>,
    completion_tokens: Option<f64>,
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterImageGenerationResponse {
    id: Option<String>,
    usage: Option<RawUsage>,
    #[serde(default)]
    choices: Vec<OpenRouterImageGenerationChoice>,
}

/// Image generation over OpenRouter's chat completions endpoint
/// (`generateImages`). Never fails: errors are reported in the result.
pub async fn generate_images(
    model: ImageModel,
    context: ImagesContext,
    options: ImagesOptions,
) -> AssistantImages {
    let mut output = AssistantImages::empty_for(&model);
    if let Err(error) = run(&model, &context, &options, &mut output).await {
        output.stop_reason = if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.is_cancelled())
        {
            ImagesStopReason::Aborted
        } else {
            ImagesStopReason::Error
        };
        output.error_message = Some(format_provider_error(
            &normalize_provider_error(&error),
            None,
        ));
    }
    output
}

async fn run(
    model: &ImageModel,
    context: &ImagesContext,
    options: &ImagesOptions,
    output: &mut AssistantImages,
) -> Result<()> {
    let Some(api_key) = options.api_key.as_deref().filter(|key| !key.is_empty()) else {
        return Err(Error::message(format!(
            "No API key for provider: {}",
            model.provider
        )));
    };
    let client = create_client(
        model,
        api_key,
        options.headers.as_ref(),
        options.http_client.as_ref(),
    );
    let mut params = build_params(model, context);
    if let Some(on_payload) = &options.on_payload
        && let Some(next_params) = on_payload(params.clone(), model).await?
    {
        params = next_params;
    }
    let request_options = OpenAIRequestOptions {
        signal: options.signal.clone(),
        timeout_ms: options.timeout_ms,
    };
    let response = retry_provider_request(
        || client.post("/chat/completions", &params, &request_options),
        &ProviderRetryOptions {
            max_retries: options.max_retries,
            max_retry_delay_ms: options.max_retry_delay_ms,
            signal: options.signal.clone(),
        },
    )
    .await?;
    if let Some(on_response) = &options.on_response {
        on_response(
            ProviderResponse {
                status: response.status().as_u16(),
                headers: headers_to_record(response.headers()),
            },
            model,
        )
        .await?;
    }
    let body = response.text().await?;
    let image_response: OpenRouterImageGenerationResponse = serde_json::from_str(&body)?;
    output.response_id = image_response.id;
    if let Some(usage) = image_response.usage {
        output.usage = Some(parse_usage(&usage, model));
    }

    if let Some(choice) = image_response.choices.into_iter().next() {
        if let Some(Value::String(content)) = choice.message.content
            && !content.is_empty()
        {
            output
                .output
                .push(UserContent::Text(TextContent::new(content)));
        }
        for image in choice.message.images.unwrap_or_default() {
            let image_url = match image.image_url {
                Some(OpenRouterImageUrl::Url(url)) => Some(url),
                Some(OpenRouterImageUrl::Object { url }) => url,
                None => None,
            };
            let Some(image_url) = image_url.filter(|url| url.starts_with("data:")) else {
                continue;
            };
            let Some((mime_type, data)) = parse_data_url(&image_url) else {
                continue;
            };
            output.output.push(UserContent::Image(ImageContent {
                mime_type: mime_type.to_string(),
                data: data.to_string(),
            }));
        }
    }
    Ok(())
}

/// `/^data:([^;]+);base64,(.+)$/`.
fn parse_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (mime_type, data) = rest.split_once(";base64,")?;
    if mime_type.is_empty() || mime_type.contains(';') || data.is_empty() || data.contains('\n') {
        return None;
    }
    Some((mime_type, data))
}

fn create_client(
    model: &ImageModel,
    api_key: &str,
    options_headers: Option<&ProviderHeaders>,
    http_client: Option<&reqwest::Client>,
) -> OpenAIClient {
    let model_headers: Option<ProviderHeaders> = model.headers.clone().map(Into::into);
    let mut merged = ProviderHeaders::new();
    if let Some(model_headers) = &model_headers {
        for (name, value) in model_headers.iter() {
            merged.insert(name.clone(), value.clone());
        }
    }
    if let Some(options_headers) = options_headers {
        for (name, value) in options_headers.iter() {
            merged.insert(name.clone(), value.clone());
        }
    }
    let headers: ProviderHeaders = provider_headers_to_record(&[Some(&merged)])
        .unwrap_or_default()
        .into();
    OpenAIClient::new(api_key, &model.base_url, http_client, headers)
}

fn build_params(model: &ImageModel, context: &ImagesContext) -> Value {
    let content: Vec<Value> = context
        .input
        .iter()
        .map(|item| match item {
            UserContent::Text(text) => json!({
                "type": "text",
                "text": sanitize_surrogates(&text.text),
            }),
            UserContent::Image(image) => json!({
                "type": "image_url",
                "image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) },
            }),
        })
        .collect();
    let modalities = if model.output.contains(&ModelOutput::Text) {
        json!(["image", "text"])
    } else {
        json!(["image"])
    };
    json!({
        "model": model.id,
        "messages": [{ "role": "user", "content": content }],
        "stream": false,
        "modalities": modalities,
    })
}

fn parse_usage(raw_usage: &RawUsage, model: &ImageModel) -> Usage {
    let as_tokens = |value: Option<f64>| value.filter(|value| *value > 0.0).unwrap_or(0.0);
    let details = raw_usage.prompt_tokens_details.as_ref();
    let prompt_tokens = as_tokens(raw_usage.prompt_tokens);
    let reported_cached_tokens = as_tokens(details.and_then(|details| details.cached_tokens));
    let cache_write_tokens = as_tokens(details.and_then(|details| details.cache_write_tokens));
    let cache_read_tokens = if cache_write_tokens > 0.0 {
        (reported_cached_tokens - cache_write_tokens).max(0.0)
    } else {
        reported_cached_tokens
    };
    let input = (prompt_tokens - cache_read_tokens - cache_write_tokens).max(0.0);
    let output = as_tokens(raw_usage.completion_tokens);
    let mut cost = UsageCost {
        input: (model.cost.input / 1_000_000.0) * input,
        output: (model.cost.output / 1_000_000.0) * output,
        cache_read: (model.cost.cache_read / 1_000_000.0) * cache_read_tokens,
        cache_write: (model.cost.cache_write / 1_000_000.0) * cache_write_tokens,
        total: 0.0,
    };
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
    Usage {
        input: input as u32,
        output: output as u32,
        cache_read: cache_read_tokens as u32,
        cache_write: cache_write_tokens as u32,
        total_tokens: (input + output + cache_read_tokens + cache_write_tokens) as u32,
        cost,
        ..Default::default()
    }
}

struct OpenRouterImagesApi;

#[async_trait]
impl ProviderImages for OpenRouterImagesApi {
    async fn generate_images(
        &self,
        model: ImageModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> AssistantImages {
        generate_images(model, context, options).await
    }
}

/// `openrouterImagesApi()`.
pub fn openrouter_images_api() -> Arc<dyn ProviderImages> {
    Arc::new(OpenRouterImagesApi)
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::openai_client::test_support::{MockResponse, MockServer};

    fn image_model(value: Value) -> ImageModel {
        serde_json::from_value(value).unwrap()
    }

    fn flux(base_url: &str) -> ImageModel {
        image_model(json!({
            "type": "image",
            "id": "black-forest-labs/flux.2-pro",
            "name": "FLUX.2 Pro",
            "api": "openrouter-images",
            "provider": "openrouter",
            "baseUrl": base_url,
            "input": ["text", "image"],
            "output": ["image"],
            "cost": { "input": 0.015, "output": 0.03, "cacheRead": 0, "cacheWrite": 0 },
        }))
    }

    fn dog() -> ImagesContext {
        ImagesContext::builder().text("Generate a dog").build()
    }

    fn api_key(key: &str) -> ImagesOptions {
        ImagesOptions {
            api_key: Some(key.to_string()),
            ..Default::default()
        }
    }

    fn image_response() -> MockResponse {
        MockResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: json!({
                "id": "img-1",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 34,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                },
                "choices": [{
                    "message": {
                        "content": "Here is your image.",
                        "images": [{ "image_url": "data:image/png;base64,ZmFrZS1wbmc=" }],
                    },
                }],
            })
            .to_string(),
        }
    }

    // openrouter-images.test.ts

    #[tokio::test]
    async fn returns_text_plus_images_in_final_output() {
        let server = MockServer::start(vec![image_response()]).await;
        let model = image_model(json!({
            "type": "image",
            "id": "google/gemini-3.1-flash-image-preview",
            "name": "Gemini 3.1 Flash Image Preview",
            "api": "openrouter-images",
            "provider": "openrouter",
            "baseUrl": server.url,
            "input": ["text", "image"],
            "output": ["text", "image"],
            "cost": { "input": 0.015, "output": 0.03, "cacheRead": 0, "cacheWrite": 0 },
            "headers": { "HTTP-Referer": "https://example.com" },
        }));
        let output = generate_images(model, dog(), api_key("test")).await;
        assert_eq!(output.stop_reason, ImagesStopReason::Stop);
        assert_eq!(output.response_id.as_deref(), Some("img-1"));
        assert_eq!(
            output.output[0],
            UserContent::Text(TextContent::new("Here is your image."))
        );
        assert_eq!(
            output.output[1],
            UserContent::Image(ImageContent {
                mime_type: "image/png".to_string(),
                data: "ZmFrZS1wbmc=".to_string(),
            })
        );
        let usage = output.usage.unwrap();
        assert_eq!(
            (usage.input, usage.output, usage.total_tokens),
            (12, 34, 46)
        );

        let request = server.last();
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.header("authorization"), Some("Bearer test"));
        assert_eq!(request.header("http-referer"), Some("https://example.com"));
        assert_eq!(request.body["stream"], json!(false));
        assert_eq!(request.body["modalities"], json!(["image", "text"]));
        assert_eq!(
            request.body["messages"][0]["content"][0],
            json!({ "type": "text", "text": "Generate a dog" })
        );
    }

    #[tokio::test]
    async fn passes_through_abort_signal_and_returns_aborted_result() {
        let signal = CancellationToken::new();
        signal.cancel();
        let options = ImagesOptions {
            signal: Some(signal),
            ..api_key("test")
        };
        let output = generate_images(flux("http://127.0.0.1:9/v1"), dog(), options).await;
        assert_eq!(output.stop_reason, ImagesStopReason::Aborted);
        assert_eq!(output.error_message.as_deref(), Some("Request aborted"));
    }

    #[tokio::test]
    async fn generate_images_resolves_the_final_assistant_images_result() {
        let server = MockServer::start(vec![image_response()]).await;
        let output = generate_images(flux(&server.url), dog(), api_key("test")).await;
        assert!(
            output
                .output
                .iter()
                .any(|item| matches!(item, UserContent::Image(_)))
        );
        // Image-only models must not request text output.
        assert_eq!(server.last().body["modalities"], json!(["image"]));
    }

    // ai.rs additions

    #[tokio::test]
    async fn sends_image_inputs_as_data_urls_and_reports_cost() {
        let server = MockServer::start(vec![image_response()]).await;
        let context = ImagesContext::builder()
            .text("Make it blue")
            .image(ImageContent {
                data: "aW1n".to_string(),
                mime_type: "image/png".to_string(),
            })
            .build();
        let output = generate_images(flux(&server.url), context, api_key("test")).await;
        let usage = output.usage.unwrap();
        assert!((usage.cost.input - 0.015 * 12.0 / 1e6).abs() < 1e-12);
        assert!((usage.cost.total - (0.015 * 12.0 + 0.03 * 34.0) / 1e6).abs() < 1e-12);
        assert_eq!(
            server.last().body["messages"][0]["content"][1],
            json!({ "type": "image_url", "image_url": { "url": "data:image/png;base64,aW1n" } })
        );
    }

    #[tokio::test]
    async fn reports_missing_keys_and_http_errors_in_band() {
        let output =
            generate_images(flux("http://127.0.0.1:9/v1"), dog(), Default::default()).await;
        assert_eq!(output.stop_reason, ImagesStopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("No API key for provider: openrouter")
        );

        let server = MockServer::start(vec![MockResponse {
            status: 400,
            headers: vec![],
            body: r#"{"error":{"message":"bad prompt"}}"#.to_string(),
        }])
        .await;
        let output = generate_images(flux(&server.url), dog(), api_key("test")).await;
        assert_eq!(output.stop_reason, ImagesStopReason::Error);
        assert!(output.error_message.unwrap().contains("bad prompt"));
    }

    #[test]
    fn parses_only_base64_data_urls() {
        assert_eq!(
            parse_data_url("data:image/webp;base64,QUJD"),
            Some(("image/webp", "QUJD"))
        );
        assert_eq!(parse_data_url("https://example.com/a.png"), None);
        assert_eq!(parse_data_url("data:image/png,QUJD"), None);
        assert_eq!(parse_data_url("data:;base64,QUJD"), None);
    }
}
