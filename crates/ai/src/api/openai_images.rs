//! ai.rs extra, not part of Pi: image generation over the OpenAI-compatible
//! `/images/generations` endpoint (OpenAI, llama.cpp, MLX, Ollama, ...),
//! re-ported from the pre-1.0 crate onto the Pi 1.0 core. The request goes
//! through `OpenAIClient` and
//! `retry_provider_request`, like Pi's OpenAI API modules.
//!
//! - Input: text only; the text blocks are joined with blank lines into
//!   `prompt`. Image inputs (edits) are rejected.
//! - `response_format: "b64_json"` is sent unless the base URL is OpenAI's
//!   (where GPT image models always return base64 and reject the field).
//! - `provider_options` keys `n`, `size`, `quality`, `style`, `user`,
//!   `background`, `moderation`, `outputFormat`/`output_format`,
//!   `outputCompression`/`output_compression` and
//!   `responseFormat`/`response_format` are forwarded.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::openai_client::{OpenAIClient, OpenAIRequestOptions};
use crate::types::{
    AssistantImages, ImageContent, ImageModel, ImagesContext, ImagesOptions, ImagesStopReason,
    KnownImageApi, ProviderHeaders, ProviderImages, ProviderResponse, TextContent, Usage,
    UsageCost, UserContent,
};
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::headers::headers_to_record;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};
use crate::{Error, Result};

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug, Deserialize)]
struct OpenAIImagesResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    usage: Option<OpenAIImagesUsage>,
    #[serde(default)]
    data: Vec<OpenAIImagesData>,
}

#[derive(Debug, Deserialize)]
struct OpenAIImagesData {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIImagesUsage {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
}

/// Generate images through `/images/generations`. Never fails: errors are
/// reported in the result.
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
    if model.api != KnownImageApi::OpenaiImages.as_str() {
        return Err(Error::message(format!(
            "Mismatched api: {} expected {}",
            model.api,
            KnownImageApi::OpenaiImages.as_str()
        )));
    }
    let Some(api_key) = options.api_key.as_deref().filter(|key| !key.is_empty()) else {
        return Err(Error::message(format!(
            "No API key for provider: {}",
            model.provider
        )));
    };
    let mut payload = Value::Object(build_payload(model, context, &options.provider_options)?);
    if let Some(on_payload) = &options.on_payload
        && let Some(next_payload) = on_payload(payload.clone(), model).await?
    {
        payload = next_payload;
    }
    let mut headers: ProviderHeaders = model.headers.clone().map(Into::into).unwrap_or_default();
    for (name, value) in options.headers.iter().flatten() {
        headers.insert(name.clone(), value.clone());
    }
    let client = OpenAIClient::new(
        api_key,
        &model.base_url,
        options.http_client.as_ref(),
        headers,
    );
    let request_options = OpenAIRequestOptions {
        signal: options.signal.clone(),
        timeout_ms: options.timeout_ms,
    };
    let response = retry_provider_request(
        || client.post("/images/generations", &payload, &request_options),
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
    let response: OpenAIImagesResponse = serde_json::from_str(&body)?;
    output.response_id = response.id;
    if let Some(usage) = response.usage {
        output.usage = Some(parse_usage(usage, model));
    }
    let mime_type = output_mime_type(&payload);
    for item in response.data {
        if let Some(revised_prompt) = item.revised_prompt.filter(|value| !value.is_empty()) {
            output
                .output
                .push(UserContent::Text(TextContent::new(revised_prompt)));
        }
        if let Some(data) = item.b64_json.filter(|value| !value.is_empty()) {
            output.output.push(UserContent::Image(ImageContent {
                data,
                mime_type: mime_type.clone(),
            }));
        }
    }
    Ok(())
}

fn build_payload(
    model: &ImageModel,
    context: &ImagesContext,
    provider_options: &Map<String, Value>,
) -> Result<Map<String, Value>> {
    let mut text = Vec::new();
    for item in &context.input {
        match item {
            UserContent::Text(content) => text.push(content.text.as_str()),
            UserContent::Image(_) => {
                return Err(Error::message(
                    "openai-images generations only support text input",
                ));
            }
        }
    }
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(model.id));
    payload.insert("prompt".to_string(), json!(text.join("\n\n")));
    if model.base_url.trim_end_matches('/') != DEFAULT_OPENAI_BASE_URL {
        payload.insert("response_format".to_string(), json!("b64_json"));
    }
    for (source, target) in [
        ("n", "n"),
        ("size", "size"),
        ("quality", "quality"),
        ("style", "style"),
        ("user", "user"),
        ("background", "background"),
        ("moderation", "moderation"),
        ("outputFormat", "output_format"),
        ("output_format", "output_format"),
        ("outputCompression", "output_compression"),
        ("output_compression", "output_compression"),
        ("responseFormat", "response_format"),
        ("response_format", "response_format"),
    ] {
        if let Some(value) = provider_options.get(source) {
            payload.insert(target.to_string(), value.clone());
        }
    }
    Ok(payload)
}

fn output_mime_type(payload: &Value) -> String {
    payload
        .get("output_format")
        .and_then(Value::as_str)
        .map(|format| format.trim_start_matches("image/"))
        .filter(|format| !format.is_empty())
        .map(|format| format!("image/{format}"))
        .unwrap_or_else(|| "image/png".to_string())
}

fn parse_usage(raw_usage: OpenAIImagesUsage, model: &ImageModel) -> Usage {
    let input = raw_usage
        .input_tokens
        .or(raw_usage.prompt_tokens)
        .unwrap_or_default();
    let output = raw_usage.output_tokens.unwrap_or_default();
    let total_tokens = raw_usage
        .total_tokens
        .unwrap_or(input.saturating_add(output));
    let mut cost = UsageCost {
        input: (model.cost.input / 1_000_000.0) * f64::from(input),
        output: (model.cost.output / 1_000_000.0) * f64::from(output),
        ..Default::default()
    };
    cost.total = cost.input + cost.output;
    Usage {
        input,
        output,
        total_tokens,
        cost,
        ..Default::default()
    }
}

struct OpenAIImagesApi;

#[async_trait]
impl ProviderImages for OpenAIImagesApi {
    async fn generate_images(
        &self,
        model: ImageModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> AssistantImages {
        generate_images(model, context, options).await
    }
}

/// The `openai-images` implementation.
pub fn openai_images_api() -> Arc<dyn ProviderImages> {
    Arc::new(OpenAIImagesApi)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::api::openai_client::test_support::{MockResponse, MockServer};

    fn model(base_url: &str) -> ImageModel {
        serde_json::from_value(json!({
            "type": "image",
            "id": "gpt-image-2",
            "name": "gpt-image-2",
            "api": "openai-images",
            "provider": "openai",
            "baseUrl": base_url,
            "input": ["text"],
            "output": ["image"],
            "cost": { "input": 5, "output": 40, "cacheRead": 0, "cacheWrite": 0 },
        }))
        .unwrap()
    }

    fn json_response(body: Value) -> MockResponse {
        MockResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: body.to_string(),
        }
    }

    #[tokio::test]
    async fn generates_base64_outputs_with_revised_prompts_and_options() {
        let server = MockServer::start(vec![json_response(json!({
            "id": "img_123",
            "usage": { "input_tokens": 11, "output_tokens": 7, "total_tokens": 18 },
            "data": [{ "b64_json": "ZmFrZS1wbmc=", "revised_prompt": "A tiny robot reading a book." }],
        }))])
        .await;
        let options = ImagesOptions {
            api_key: Some("test-key".to_string()),
            provider_options:
                json!({ "size": "1024x1024", "quality": "medium", "outputFormat": "jpeg" })
                    .as_object()
                    .unwrap()
                    .clone(),
            ..Default::default()
        };
        let context = ImagesContext::builder()
            .text("A tiny robot")
            .text("reading a book")
            .build();
        let output = generate_images(model(&server.url), context, options).await;
        assert_eq!(output.stop_reason, ImagesStopReason::Stop);
        assert_eq!(output.response_id.as_deref(), Some("img_123"));
        let usage = output.usage.unwrap();
        assert_eq!((usage.input, usage.output, usage.total_tokens), (11, 7, 18));
        assert!((usage.cost.total - (5.0 * 11.0 + 40.0 * 7.0) / 1e6).abs() < 1e-12);
        assert_eq!(
            output.output,
            vec![
                UserContent::text("A tiny robot reading a book."),
                UserContent::Image(ImageContent {
                    data: "ZmFrZS1wbmc=".to_string(),
                    mime_type: "image/jpeg".to_string(),
                }),
            ]
        );
        let request = server.last();
        assert_eq!(request.path, "/v1/images/generations");
        assert_eq!(request.header("authorization"), Some("Bearer test-key"));
        assert_eq!(request.body["model"], "gpt-image-2");
        assert_eq!(request.body["prompt"], "A tiny robot\n\nreading a book");
        assert_eq!(request.body["response_format"], "b64_json");
        assert_eq!(request.body["size"], "1024x1024");
        assert_eq!(request.body["quality"], "medium");
        assert_eq!(request.body["output_format"], "jpeg");
    }

    #[tokio::test]
    async fn rejects_image_inputs_and_missing_keys() {
        let context = ImagesContext::builder()
            .text("Edit this image")
            .image(ImageContent {
                data: "abc".to_string(),
                mime_type: "image/png".to_string(),
            })
            .build();
        let options = ImagesOptions {
            api_key: Some("test-key".to_string()),
            ..Default::default()
        };
        let output = generate_images(model("http://127.0.0.1:9/v1"), context, options).await;
        assert_eq!(output.stop_reason, ImagesStopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("openai-images generations only support text input")
        );

        let output = generate_images(
            model("http://127.0.0.1:9/v1"),
            ImagesContext::builder().text("cat").build(),
            Default::default(),
        )
        .await;
        assert_eq!(
            output.error_message.as_deref(),
            Some("No API key for provider: openai")
        );
    }

    #[test]
    fn omits_response_format_for_the_official_endpoint() {
        let payload = build_payload(
            &model(DEFAULT_OPENAI_BASE_URL),
            &ImagesContext::builder().text("cat").build(),
            &Map::new(),
        )
        .unwrap();
        assert!(!payload.contains_key("response_format"));
    }
}
