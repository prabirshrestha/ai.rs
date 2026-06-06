use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::models::calculate_cost;
use crate::types::{
    AssistantImages, ImageContent, ImageGenerationOptions, ImageOutput, ImagesContext,
    ImagesStopReason, KnownApi, Model, ProviderResponse, TextContent, Usage, UserContent,
};
use crate::utils::headers::headers_to_record;
use crate::utils::http::{request_timeout, send_with_retries};
use crate::{Error, Result};

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

pub async fn generate_images_openai(
    model: Model,
    context: ImagesContext,
    options: ImageGenerationOptions,
) -> AssistantImages {
    let mut output = AssistantImages::empty_for(&model);

    match run_generate_images_openai(&model, context, &options, &mut output).await {
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

async fn run_generate_images_openai(
    model: &Model,
    context: ImagesContext,
    options: &ImageGenerationOptions,
    output: &mut AssistantImages,
) -> Result<()> {
    if model.api != KnownApi::OpenaiImages.as_str() {
        return Err(Error::UnsupportedApi(format!(
            "Mismatched api: {} expected {}",
            model.api,
            KnownApi::OpenaiImages.as_str()
        )));
    }

    let api_key = options
        .base
        .api_key
        .as_deref()
        .filter(|api_key| !api_key.trim().is_empty())
        .ok_or_else(|| Error::MissingApiKey(model.provider.clone()))?;

    let mut payload = build_payload(model, context, &options.base.provider_options)?;
    if let Some(on_payload) = &options.base.on_payload
        && let Some(next_payload) = on_payload(Value::Object(payload.clone()), model).await?
    {
        payload = next_payload.as_object().cloned().ok_or_else(|| {
            Error::Provider("OpenAI Images payload hook must return a JSON object".to_string())
        })?;
    }

    let client = options.base.http_client.clone().unwrap_or_default();
    let url = format!(
        "{}/images/generations",
        model.base_url.trim_end_matches('/')
    );
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

    let response: OpenAIImagesResponse = serde_json::from_str(&body)?;
    output.response_id = response.id;
    if let Some(raw_usage) = response.usage {
        output.usage = parse_usage(raw_usage, model);
    }

    for item in response.data {
        if let Some(revised_prompt) = item.revised_prompt.filter(|value| !value.is_empty()) {
            output.output.push(ImageOutput::Text(TextContent {
                text: revised_prompt,
                text_signature: None,
            }));
        }
        if let Some(data) = item.b64_json.filter(|value| !value.is_empty()) {
            output.output.push(ImageOutput::Image(ImageContent {
                data,
                mime_type: output_mime_type(&payload),
            }));
        }
    }

    Ok(())
}

fn build_payload(
    model: &Model,
    context: ImagesContext,
    provider_options: &std::collections::HashMap<String, Value>,
) -> Result<Map<String, Value>> {
    let prompt = prompt_from_context(context)?;
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(model.id));
    payload.insert("prompt".to_string(), json!(prompt));
    if model.base_url != DEFAULT_OPENAI_BASE_URL {
        payload.insert("response_format".to_string(), json!("b64_json"));
    }
    for (key, value) in image_provider_options(provider_options) {
        payload.insert(key, value);
    }
    Ok(payload)
}

fn prompt_from_context(context: ImagesContext) -> Result<String> {
    let mut text = Vec::new();
    for item in context.input {
        match item {
            UserContent::Text(content) => text.push(content.text),
            UserContent::Image(_) => {
                return Err(Error::Provider(
                    "openai-images generations only supports text input".to_string(),
                ));
            }
        }
    }
    Ok(text.join("\n\n"))
}

fn image_provider_options(
    provider_options: &std::collections::HashMap<String, Value>,
) -> Vec<(String, Value)> {
    let mut options = Vec::new();
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
            options.push((target.to_string(), value.clone()));
        }
    }
    options
}

fn build_headers(
    api_key: &str,
    model_headers: &std::collections::HashMap<String, String>,
    option_headers: &std::collections::HashMap<String, String>,
) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if !api_key.is_empty() {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|error| Error::InvalidHeaderValue("authorization".to_string(), error))?,
        );
    }
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

fn output_mime_type(payload: &Map<String, Value>) -> String {
    payload
        .get("output_format")
        .and_then(Value::as_str)
        .map(|format| format.trim_start_matches("image/"))
        .filter(|format| !format.is_empty())
        .map(|format| format!("image/{format}"))
        .unwrap_or_else(|| "image/png".to_string())
}

fn parse_usage(raw_usage: OpenAIImagesUsage, model: &Model) -> Usage {
    let input = raw_usage
        .input_tokens
        .or(raw_usage.prompt_tokens)
        .unwrap_or_default();
    let output = raw_usage.output_tokens.unwrap_or_default();
    let total_tokens = raw_usage
        .total_tokens
        .unwrap_or(input.saturating_add(output));
    let mut usage = Usage {
        input,
        output,
        total_tokens,
        ..Usage::default()
    };
    calculate_cost(model, &mut usage);
    usage
}

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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use crate::providers::openai;
    use crate::types::{
        ImageContent, ImageGenerationOptions, ImageOutput, ImagesContext, ImagesStopReason,
        ModelInput, ModelOutput, StreamOptions, UserContent,
    };

    #[tokio::test(flavor = "current_thread")]
    async fn openai_images_generates_base64_outputs() {
        let captured = Arc::new(Mutex::new(String::new()));
        let url = spawn_response_server(
            Arc::clone(&captured),
            200,
            r#"{
                "id": "img_123",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18
                },
                "data": [{
                    "b64_json": "ZmFrZS1wbmc=",
                    "revised_prompt": "A tiny robot reading a book."
                }]
            }"#,
        )
        .await;
        let provider = openai::builder()
            .api_key(Some("test-key"))
            .base_url(url)
            .images()
            .build()
            .expect("provider");
        let model = provider.model("gpt-image-2").build_image().expect("model");
        let options = ImageGenerationOptions {
            base: StreamOptions {
                provider_options: [
                    ("size".to_string(), Value::String("1024x1024".to_string())),
                    ("quality".to_string(), Value::String("medium".to_string())),
                    (
                        "outputFormat".to_string(),
                        Value::String("jpeg".to_string()),
                    ),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            },
        };

        let output = crate::generate_images(
            model,
            ImagesContext::builder()
                .text("A tiny robot")
                .text("reading a book")
                .build(),
            Some(options),
        )
        .await
        .expect("generate images");

        assert_eq!(output.stop_reason, ImagesStopReason::Stop);
        assert_eq!(output.response_id.as_deref(), Some("img_123"));
        assert_eq!(output.usage.input, 11);
        assert_eq!(output.usage.output, 7);
        assert_eq!(output.usage.total_tokens, 18);
        assert_eq!(
            output.output[0],
            ImageOutput::Text(crate::TextContent {
                text: "A tiny robot reading a book.".to_string(),
                text_signature: None,
            })
        );
        assert_eq!(
            output.output[1],
            ImageOutput::Image(ImageContent {
                data: "ZmFrZS1wbmc=".to_string(),
                mime_type: "image/jpeg".to_string(),
            })
        );

        let request = captured.lock().expect("request").clone();
        assert!(request.contains("authorization: Bearer test-key"));
        let payload = request_body_json(&request);
        assert_eq!(payload["model"], "gpt-image-2");
        assert_eq!(payload["prompt"], "A tiny robot\n\nreading a book");
        assert_eq!(payload["response_format"], "b64_json");
        assert_eq!(payload["size"], "1024x1024");
        assert_eq!(payload["quality"], "medium");
        assert_eq!(payload["output_format"], "jpeg");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ollama_compatible_base_url_uses_openai_images_endpoint_without_env_key() {
        let captured = Arc::new(Mutex::new(String::new()));
        let url = spawn_response_server(
            Arc::clone(&captured),
            200,
            r#"{
                "created": 1710000000,
                "data": [{ "b64_json": "b2xsYW1h" }]
            }"#,
        )
        .await;
        let provider = openai::builder()
            .provider_id("ollama")
            .base_url(url)
            .images()
            .build()
            .expect("provider");
        let model = provider
            .model("x/z-image-turbo")
            .build_image()
            .expect("model");

        let output = crate::generate_images(
            model,
            ImagesContext::builder().text("Generate a cat").build(),
            None,
        )
        .await
        .expect("generate images");

        assert_eq!(output.stop_reason, ImagesStopReason::Stop);
        assert_eq!(
            output.output[0],
            ImageOutput::Image(ImageContent {
                data: "b2xsYW1h".to_string(),
                mime_type: "image/png".to_string(),
            })
        );
        let request = captured.lock().expect("request").clone();
        assert!(request.starts_with("POST /v1/images/generations HTTP/1.1"));
        assert!(request.contains("authorization: Bearer ollama"));
        let payload = request_body_json(&request);
        assert_eq!(payload["model"], "x/z-image-turbo");
        assert_eq!(payload["response_format"], "b64_json");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn image_model_convenience_works_from_non_image_openai_provider() {
        let provider = openai::builder()
            .api_key(Some("test-key"))
            .build()
            .expect("provider");
        let model = provider
            .image_model("gpt-image-2")
            .build_image()
            .expect("model");

        assert_eq!(model.api, "openai-images");
        assert_eq!(model.input, vec![ModelInput::Text]);
        assert_eq!(model.output, vec![ModelOutput::Image]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generations_endpoint_rejects_image_inputs() {
        let provider = openai::builder()
            .api_key(Some("test-key"))
            .images()
            .build()
            .expect("provider");
        let model = provider.model("gpt-image-2").build_image().expect("model");

        let output = crate::generate_images(
            model,
            ImagesContext::builder()
                .input([
                    UserContent::text("Edit this image"),
                    UserContent::Image(ImageContent {
                        data: "abc".to_string(),
                        mime_type: "image/png".to_string(),
                    }),
                ])
                .build(),
            None,
        )
        .await
        .expect("generate images");

        assert_eq!(output.stop_reason, ImagesStopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("provider error: openai-images generations only supports text input")
        );
    }

    async fn spawn_response_server(
        captured: Arc<Mutex<String>>,
        status: u16,
        body: &'static str,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
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
        format!("http://{addr}/v1")
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
