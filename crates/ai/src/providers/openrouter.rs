use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::env_api_keys::{KnownProvider, get_env_api_key};
use crate::models::calculate_cost;
use crate::provider::{ImageModelApi, ModelBuilder, Provider, ProviderCapabilities};
use crate::types::{
    AssistantImages, ImageContent, ImageGenerationOptions, ImageOutput, ImagesContext,
    ImagesStopReason, KnownApi, Model, ModelInput, ModelOutput, ProviderResponse, TextContent,
    Usage, UserContent,
};
use crate::utils::headers::headers_to_record;
use crate::utils::http::{request_timeout, send_with_retries};
use crate::{Error, Result};

const DEFAULT_PROVIDER_ID: KnownProvider = KnownProvider::OpenRouter;
const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Clone)]
pub struct OpenRouter {
    provider_id: String,
    api_key: Option<String>,
    base_url: String,
    http_client: Option<reqwest::Client>,
}

impl OpenRouter {
    pub fn builder() -> OpenRouterBuilder {
        OpenRouterBuilder::default()
    }

    pub fn from_env() -> Result<Self> {
        let api_key = get_env_api_key(DEFAULT_PROVIDER_ID)
            .ok_or_else(|| Error::MissingApiKey(DEFAULT_PROVIDER_ID.into()))?;
        Self::builder().api_key(Some(api_key.as_str())).build()
    }

    pub fn model(&self, id: &str) -> ModelBuilder {
        <Self as Provider>::model(self, id)
    }
}

impl Provider for OpenRouter {
    fn id(&self) -> &str {
        &self.provider_id
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            language_models: false,
            image_models: true,
        }
    }

    fn model(&self, id: &str) -> ModelBuilder {
        let runtime = Arc::new(OpenRouterImageModelApi {
            api_key: self.api_key.clone(),
            http_client: self.http_client.clone(),
        });
        ModelBuilder::new_image(&self.provider_id, id, runtime)
            .base_url(self.base_url.clone())
            .input(vec![ModelInput::Text])
            .output(vec![ModelOutput::Image])
    }
}

#[derive(Default)]
pub struct OpenRouterBuilder {
    provider_id: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    http_client: Option<reqwest::Client>,
}

impl OpenRouterBuilder {
    pub fn provider_id(mut self, provider_id: impl Into<String>) -> Self {
        self.provider_id = Some(provider_id.into());
        self
    }

    pub fn api_key(mut self, api_key: Option<&str>) -> Self {
        self.api_key = api_key
            .map(str::trim)
            .filter(|api_key| !api_key.is_empty())
            .map(str::to_string);
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn build(self) -> Result<OpenRouter> {
        Ok(OpenRouter {
            provider_id: self
                .provider_id
                .unwrap_or_else(|| DEFAULT_PROVIDER_ID.into()),
            api_key: self.api_key,
            base_url: self
                .base_url
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            http_client: self.http_client,
        })
    }
}

#[derive(Clone)]
struct OpenRouterImageModelApi {
    api_key: Option<String>,
    http_client: Option<reqwest::Client>,
}

#[async_trait]
impl ImageModelApi for OpenRouterImageModelApi {
    fn id(&self) -> &str {
        KnownApi::OpenrouterImages.as_str()
    }

    async fn generate_images(
        &self,
        model: Model,
        context: ImagesContext,
        mut options: ImageGenerationOptions,
    ) -> Result<AssistantImages> {
        if options
            .base
            .api_key
            .as_deref()
            .is_none_or(|api_key| api_key.trim().is_empty())
        {
            options.base.api_key.clone_from(&self.api_key);
        }
        if options.base.http_client.is_none() {
            options.base.http_client.clone_from(&self.http_client);
        }
        Ok(generate_images_openrouter(model, context, options).await)
    }
}

async fn generate_images_openrouter(
    model: Model,
    context: ImagesContext,
    options: ImageGenerationOptions,
) -> AssistantImages {
    let mut output = AssistantImages::empty_for(&model);

    match run_generate_images_openrouter(&model, context, &options, &mut output).await {
        Ok(()) => output,
        Err(error) => {
            output.stop_reason = if matches!(error, Error::Cancelled) {
                ImagesStopReason::Aborted
            } else {
                ImagesStopReason::Error
            };
            output.error_message = Some(error.to_string());
            output
        }
    }
}

async fn run_generate_images_openrouter(
    model: &Model,
    context: ImagesContext,
    options: &ImageGenerationOptions,
    output: &mut AssistantImages,
) -> Result<()> {
    if model.api != KnownApi::OpenrouterImages.as_str() {
        return Err(Error::UnsupportedApi(format!(
            "Mismatched api: {} expected {}",
            model.api,
            KnownApi::OpenrouterImages.as_str()
        )));
    }

    let api_key = options
        .base
        .api_key
        .as_deref()
        .filter(|api_key| !api_key.trim().is_empty())
        .ok_or_else(|| Error::MissingApiKey(model.provider.clone()))?;

    let mut payload = build_params(model, context);
    if let Some(on_payload) = &options.base.on_payload
        && let Some(next_payload) = on_payload(payload.clone(), model).await?
    {
        payload = next_payload;
    }

    let client = options.base.http_client.clone().unwrap_or_default();
    let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));
    let headers = build_headers(api_key, &model.headers, &options.base.headers)?;
    let response = send_with_retries(&options.base, || {
        client
            .post(&url)
            .headers(headers.clone())
            .json(&payload)
            .timeout(request_timeout(options.base.timeout_ms))
    })
    .await?;
    let status = response.status();
    let response_headers = response.headers().clone();
    let body = response.text().await?;

    if !status.is_success() {
        return Err(Error::ApiStatus { status, body });
    }

    if let Some(on_response) = &options.base.on_response {
        on_response(
            ProviderResponse {
                status: status.as_u16(),
                headers: headers_to_record(&response_headers),
            },
            model,
        )
        .await?;
    }

    let response: OpenRouterImageGenerationResponse = serde_json::from_str(&body)?;
    output.response_id = Some(response.id);
    if let Some(raw_usage) = response.usage {
        output.usage = parse_usage(raw_usage, model);
    }

    if let Some(choice) = response.choices.into_iter().next() {
        if let Some(content) = choice.message.content.filter(|content| !content.is_empty()) {
            output.output.push(ImageOutput::Text(TextContent {
                text: content,
                text_signature: None,
            }));
        }

        for image in choice.message.images {
            let Some(image_url) = image.image_url.and_then(OpenRouterImageUrl::into_url) else {
                continue;
            };
            let Some(image) = parse_data_url(&image_url) else {
                continue;
            };
            output.output.push(ImageOutput::Image(image));
        }
    }

    Ok(())
}

fn build_headers(
    api_key: &str,
    model_headers: &std::collections::HashMap<String, String>,
    option_headers: &std::collections::HashMap<String, String>,
) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|error| Error::InvalidHeaderValue("authorization".to_string(), error))?,
    );
    for (name, value) in model_headers.iter().chain(option_headers.iter()) {
        let name = name
            .parse::<HeaderName>()
            .map_err(|error| Error::Provider(format!("invalid header name: {error}")))?;
        let value = HeaderValue::from_str(value)
            .map_err(|error| Error::InvalidHeaderValue(name.as_str().to_string(), error))?;
        headers.insert(name, value);
    }
    Ok(headers)
}

fn build_params(model: &Model, context: ImagesContext) -> Value {
    let content = context
        .input
        .into_iter()
        .map(|item| match item {
            UserContent::Text(text) => json!({
                "type": "text",
                "text": text.text,
            }),
            UserContent::Image(image) => json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", image.mime_type, image.data),
                },
            }),
        })
        .collect::<Vec<_>>();
    let mut modalities = vec!["image"];
    if model.output.contains(&ModelOutput::Text) {
        modalities.push("text");
    }

    json!({
        "model": model.id,
        "messages": [{
            "role": "user",
            "content": content,
        }],
        "stream": false,
        "modalities": modalities,
    })
}

fn parse_data_url(value: &str) -> Option<ImageContent> {
    let rest = value.strip_prefix("data:")?;
    let (mime_type, data) = rest.split_once(";base64,")?;
    if mime_type.is_empty() || data.is_empty() {
        return None;
    }
    Some(ImageContent {
        mime_type: mime_type.to_string(),
        data: data.to_string(),
    })
}

fn parse_usage(raw_usage: OpenRouterUsage, model: &Model) -> Usage {
    let prompt_tokens = raw_usage.prompt_tokens.unwrap_or_default();
    let reported_cached_tokens = raw_usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .unwrap_or_default();
    let cache_write_tokens = raw_usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cache_write_tokens)
        .unwrap_or_default();
    let cache_read_tokens = if cache_write_tokens > 0 {
        reported_cached_tokens.saturating_sub(cache_write_tokens)
    } else {
        reported_cached_tokens
    };
    let input = prompt_tokens
        .saturating_sub(cache_read_tokens)
        .saturating_sub(cache_write_tokens);
    let output = raw_usage.completion_tokens.unwrap_or_default();
    let mut usage = Usage {
        input,
        output,
        cache_read: cache_read_tokens,
        cache_write: cache_write_tokens,
        total_tokens: input + output + cache_read_tokens + cache_write_tokens,
        ..Usage::default()
    };
    calculate_cost(model, &mut usage);
    usage
}

#[derive(Debug, Deserialize)]
struct OpenRouterImageGenerationResponse {
    id: String,
    #[serde(default)]
    usage: Option<OpenRouterUsage>,
    choices: Vec<OpenRouterImageGenerationChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterImageGenerationChoice {
    message: OpenRouterImageGenerationMessage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterImageGenerationMessage {
    content: Option<String>,
    #[serde(default)]
    images: Vec<OpenRouterGeneratedImage>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterGeneratedImage {
    #[serde(default)]
    image_url: Option<OpenRouterImageUrl>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenRouterImageUrl {
    String(String),
    Object { url: Option<String> },
}

impl OpenRouterImageUrl {
    fn into_url(self) -> Option<String> {
        match self {
            Self::String(value) => Some(value),
            Self::Object { url } => url,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    prompt_tokens_details: Option<OpenRouterPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterPromptTokensDetails {
    cached_tokens: Option<u32>,
    cache_write_tokens: Option<u32>,
}

pub fn builder() -> OpenRouterBuilder {
    OpenRouter::builder()
}

pub fn from_env() -> Result<OpenRouter> {
    OpenRouter::from_env()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_util::sync::CancellationToken;

    use crate::types::{
        ImageGenerationOptions, ImageOutput, ImagesContext, ModelInput, StreamOptions, UserContent,
    };

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn returns_text_plus_images_and_sends_openrouter_payload() {
        let captured = Arc::new(Mutex::new(String::new()));
        let url = spawn_response_server(
            Arc::clone(&captured),
            200,
            r#"{
                "id": "img-1",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 34,
                    "prompt_tokens_details": {
                        "cached_tokens": 0
                    }
                },
                "choices": [{
                    "message": {
                        "content": "Here is your image.",
                        "images": [{
                            "image_url": "data:image/png;base64,ZmFrZS1wbmc="
                        }]
                    }
                }]
            }"#,
        )
        .await;
        let provider = builder()
            .api_key(Some("test-key"))
            .base_url(url)
            .build()
            .expect("provider");
        let model = provider
            .model("google/gemini-3.1-flash-image-preview")
            .header("HTTP-Referer", "https://example.com")
            .expect("header")
            .build_image()
            .expect("model");
        let context = ImagesContext::builder().text("Generate a dog").build();

        let output = crate::generate_images(model, context, None)
            .await
            .expect("generate images");

        assert_eq!(output.stop_reason, ImagesStopReason::Stop);
        assert_eq!(output.response_id.as_deref(), Some("img-1"));
        assert_eq!(
            output.output[0],
            ImageOutput::Text(TextContent {
                text: "Here is your image.".to_string(),
                text_signature: None,
            })
        );
        assert_eq!(
            output.output[1],
            ImageOutput::Image(ImageContent {
                mime_type: "image/png".to_string(),
                data: "ZmFrZS1wbmc=".to_string(),
            })
        );
        assert_eq!(output.usage.input, 12);
        assert_eq!(output.usage.output, 34);
        assert_eq!(output.usage.total_tokens, 46);
        assert_eq!(output.usage.cost.input, 0.0);
        assert_eq!(output.usage.cost.output, 0.0);

        let request = captured.lock().expect("request").clone();
        assert!(request.contains("authorization: Bearer test-key"));
        assert!(request.contains("http-referer: https://example.com"));
        let payload = request_body_json(&request);
        assert_eq!(payload["modalities"], serde_json::json!(["image"]));
        assert_eq!(payload["stream"], false);
        assert_eq!(payload["model"], "google/gemini-3.1-flash-image-preview");
        assert_eq!(
            payload["messages"][0]["content"][0],
            serde_json::json!({ "type": "text", "text": "Generate a dog" })
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sends_image_inputs_as_data_urls_and_image_only_modalities() {
        let captured = Arc::new(Mutex::new(String::new()));
        let url = spawn_response_server(
            Arc::clone(&captured),
            200,
            r#"{
                "id": "img-2",
                "choices": [{
                    "message": {
                        "content": "",
                        "images": [{
                            "image_url": { "url": "data:image/jpeg;base64,abc" }
                        }]
                    }
                }]
            }"#,
        )
        .await;
        let provider = builder()
            .api_key(Some("test-key"))
            .base_url(url)
            .build()
            .expect("provider");
        let model = provider
            .model("black-forest-labs/flux.2-pro")
            .input(vec![ModelInput::Text, ModelInput::Image])
            .build_image()
            .expect("model");
        let context = ImagesContext::builder()
            .input([
                UserContent::text("Edit this"),
                UserContent::Image(ImageContent {
                    mime_type: "image/png".to_string(),
                    data: "abc123".to_string(),
                }),
            ])
            .build();

        let output = crate::generate_images(model, context, None)
            .await
            .expect("generate images");

        assert_eq!(output.stop_reason, ImagesStopReason::Stop);
        assert!(matches!(output.output[0], ImageOutput::Image(_)));
        let payload = request_body_json(&captured.lock().expect("request"));
        assert_eq!(payload["modalities"], serde_json::json!(["image"]));
        assert_eq!(
            payload["messages"][0]["content"][1],
            serde_json::json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,abc123" }
            })
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancellation_returns_aborted_result() {
        let provider = builder()
            .api_key(Some("test-key"))
            .build()
            .expect("provider");
        let model = provider
            .model("black-forest-labs/flux.2-pro")
            .build_image()
            .expect("model");
        let cancellation_token = CancellationToken::new();
        cancellation_token.cancel();
        let options = ImageGenerationOptions {
            base: StreamOptions {
                cancellation_token: Some(cancellation_token),
                ..Default::default()
            },
        };

        let output = crate::generate_images(
            model,
            ImagesContext::builder().text("Generate a dog").build(),
            Some(options),
        )
        .await
        .expect("generate images");

        assert_eq!(output.stop_reason, ImagesStopReason::Aborted);
        assert_eq!(
            output.error_message.as_deref(),
            Some("request was cancelled")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missing_api_key_returns_error_result() {
        let provider = builder().build().expect("provider");
        let model = provider
            .model("black-forest-labs/flux.2-pro")
            .build_image()
            .expect("model");

        let output = crate::generate_images(
            model,
            ImagesContext::builder().text("Generate a dog").build(),
            None,
        )
        .await
        .expect("generate images");

        assert_eq!(output.stop_reason, ImagesStopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("No API key for provider: openrouter")
        );
    }

    async fn spawn_response_server(
        captured: Arc<Mutex<String>>,
        status: u16,
        body: &'static str,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let request = read_http_request(&mut socket).await;
            *captured.lock().expect("captured request") = request;
            let status_text = if status == 200 { "OK" } else { "Error" };
            let response = format!(
                "HTTP/1.1 {status} {status_text}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
        });
        format!("http://{addr}")
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
        let mut buffer = Vec::new();
        let mut temp = [0; 1024];
        loop {
            let read = socket.read(&mut temp).await.expect("read request");
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&temp[..read]);
            if let Some(header_end) = find_header_end(&buffer) {
                let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or_default();
                let expected = header_end + 4 + content_length;
                while buffer.len() < expected {
                    let read = socket.read(&mut temp).await.expect("read body");
                    if read == 0 {
                        break;
                    }
                    buffer.extend_from_slice(&temp[..read]);
                }
                break;
            }
        }
        String::from_utf8(buffer).expect("utf8 request")
    }

    fn find_header_end(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn request_body_json(request: &str) -> Value {
        let body = request
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .expect("request body");
        serde_json::from_str(body).expect("json body")
    }
}
