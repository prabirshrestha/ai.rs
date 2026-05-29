use std::time::Duration;

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::types::{
    AssistantImages, ImagesContent, ImagesContext, ImagesModel, ImagesOptions, ImagesStopReason,
    ModelInput, ProviderResponse, TextContent, Usage, UsageCost,
};
use crate::utils::headers::headers_to_record;
use crate::utils::sanitize::sanitize_surrogates;
use crate::{Error, Result};

pub async fn generate_images_openrouter(
    model: ImagesModel,
    context: ImagesContext,
    options: ImagesOptions,
) -> Result<AssistantImages> {
    let output = create_output(&model);
    match generate_images_openrouter_inner(model, context, options.clone(), output.clone()).await {
        Ok(output) => Ok(output),
        Err(error) => {
            let mut output = output;
            output.stop_reason = if is_cancelled(&options) {
                ImagesStopReason::Aborted
            } else {
                ImagesStopReason::Error
            };
            output.error_message = Some(error.to_string());
            Ok(output)
        }
    }
}

async fn generate_images_openrouter_inner(
    model: ImagesModel,
    context: ImagesContext,
    options: ImagesOptions,
    mut output: AssistantImages,
) -> Result<AssistantImages> {
    if is_cancelled(&options) {
        return Err(Error::Cancelled);
    }

    let api_key = options
        .api_key
        .clone()
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| Error::Validation(format!("No API key for provider: {}", model.provider)))?;
    let client = reqwest::Client::new();
    let mut payload = build_params(&model, &context);
    if let Some(on_payload) = options.on_payload.clone() {
        if let Some(next_payload) = on_payload(payload.clone(), &model).await? {
            payload = next_payload;
        }
    }

    let mut request = client
        .post(openrouter_chat_completions_url(&model))
        .headers(headers(&model, &options, &api_key)?)
        .json(&payload);
    if let Some(timeout_ms) = options.timeout_ms {
        request = request.timeout(Duration::from_millis(timeout_ms));
    }

    let response = if let Some(cancellation_token) = options.cancellation_token.as_ref() {
        tokio::select! {
            _ = cancellation_token.cancelled() => return Err(Error::Cancelled),
            response = request.send() => response?,
        }
    } else {
        request.send().await?
    };
    let status = response.status();
    let response_headers = headers_to_record(response.headers());
    if let Some(on_response) = options.on_response.clone() {
        on_response(
            ProviderResponse {
                status: status.as_u16(),
                headers: response_headers,
            },
            &model,
        )
        .await?;
    }

    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(Error::ApiStatus { status, body });
    }

    let image_response = response.json::<OpenRouterImageGenerationResponse>().await?;
    output.response_id = image_response.id;
    if let Some(usage) = image_response.usage {
        output.usage = Some(parse_usage(usage, &model));
    }

    if let Some(choice) = image_response.choices.into_iter().next() {
        if let Some(content) = choice.message.content.and_then(|content| match content {
            Value::String(content) if !content.is_empty() => Some(content),
            _ => None,
        }) {
            output.output.push(ImagesContent::Text(TextContent {
                text: content,
                text_signature: None,
            }));
        }

        for image in choice.message.images.unwrap_or_default() {
            let Some(image_url) = image.image_url.and_then(|image_url| image_url.url()) else {
                continue;
            };
            let Some((mime_type, data)) = parse_data_url(&image_url) else {
                continue;
            };
            output.output.push(ImagesContent::image(data, mime_type));
        }
    }

    Ok(output)
}

fn create_output(model: &ImagesModel) -> AssistantImages {
    AssistantImages {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        output: Vec::new(),
        response_id: None,
        usage: None,
        stop_reason: ImagesStopReason::Stop,
        error_message: None,
        timestamp: crate::utils::time::now_millis(),
    }
}

fn openrouter_chat_completions_url(model: &ImagesModel) -> String {
    format!("{}/chat/completions", model.base_url.trim_end_matches('/'))
}

fn build_params(model: &ImagesModel, context: &ImagesContext) -> Value {
    let content = context
        .input
        .iter()
        .map(|item| match item {
            ImagesContent::Text(text) => json!({
                "type": "text",
                "text": sanitize_surrogates(&text.text),
            }),
            ImagesContent::Image(image) => json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", image.mime_type, image.data),
                },
            }),
        })
        .collect::<Vec<_>>();

    let modalities = if model.output.contains(&ModelInput::Text) {
        vec!["image", "text"]
    } else {
        vec!["image"]
    };

    json!({
        "model": model.id,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "stream": false,
        "modalities": modalities,
    })
}

fn headers(model: &ImagesModel, options: &ImagesOptions, api_key: &str) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|e| Error::InvalidHeaderValue("authorization".to_string(), e))?,
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    for (name, value) in model.headers.iter().chain(options.headers.iter()) {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let value = HeaderValue::from_str(value)
            .map_err(|e| Error::InvalidHeaderValue(name.to_string(), e))?;
        headers.insert(name, value);
    }
    Ok(headers)
}

fn parse_data_url(value: &str) -> Option<(String, String)> {
    let value = value.strip_prefix("data:")?;
    let (mime_type, data) = value.split_once(";base64,")?;
    Some((mime_type.to_string(), data.to_string()))
}

fn parse_usage(raw_usage: OpenRouterRawUsage, model: &ImagesModel) -> Usage {
    let prompt_tokens = raw_usage.prompt_tokens.unwrap_or(0);
    let reported_cached_tokens = raw_usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .unwrap_or(0);
    let cache_write_tokens = raw_usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cache_write_tokens)
        .unwrap_or(0);
    let cache_read_tokens = if cache_write_tokens > 0 {
        reported_cached_tokens.saturating_sub(cache_write_tokens)
    } else {
        reported_cached_tokens
    };
    let input = prompt_tokens.saturating_sub(cache_read_tokens + cache_write_tokens);
    let output = raw_usage.completion_tokens.unwrap_or(0);
    let mut usage = Usage {
        input,
        output,
        cache_read: cache_read_tokens,
        cache_write: cache_write_tokens,
        total_tokens: input + output + cache_read_tokens + cache_write_tokens,
        cost: UsageCost {
            input: (model.cost.input / 1_000_000.0) * input as f64,
            output: (model.cost.output / 1_000_000.0) * output as f64,
            cache_read: (model.cost.cache_read / 1_000_000.0) * cache_read_tokens as f64,
            cache_write: (model.cost.cache_write / 1_000_000.0) * cache_write_tokens as f64,
            total: 0.0,
        },
    };
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage
}

fn is_cancelled(options: &ImagesOptions) -> bool {
    options
        .cancellation_token
        .as_ref()
        .is_some_and(|token| token.is_cancelled())
}

#[derive(Debug, Deserialize)]
struct OpenRouterImageGenerationResponse {
    id: Option<String>,
    #[serde(default)]
    choices: Vec<OpenRouterImageGenerationChoice>,
    usage: Option<OpenRouterRawUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterImageGenerationChoice {
    message: OpenRouterImageGenerationMessage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterImageGenerationMessage {
    content: Option<Value>,
    images: Option<Vec<OpenRouterGeneratedImage>>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterGeneratedImage {
    image_url: Option<OpenRouterImageUrl>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenRouterImageUrl {
    String(String),
    Object { url: Option<String> },
}

impl OpenRouterImageUrl {
    fn url(self) -> Option<String> {
        match self {
            Self::String(url) => Some(url),
            Self::Object { url } => url,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenRouterRawUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    prompt_tokens_details: Option<OpenRouterPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterPromptTokensDetails {
    cached_tokens: Option<u32>,
    cache_write_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use std::io;

    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    use crate::image_models::get_image_model;
    use crate::images::generate_images;
    use crate::types::{ImageContent, ImagesOutputContent};

    use super::*;

    #[derive(Debug)]
    struct CapturedRequest {
        path: String,
        authorization: Option<String>,
        body: Value,
    }

    async fn serve_once(response_body: Value) -> (String, oneshot::Receiver<CapturedRequest>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let (sender, receiver) = oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept request");
            let mut buffer = Vec::new();
            let header_end;
            loop {
                let mut chunk = [0_u8; 1024];
                let n = socket.read(&mut chunk).await.expect("read request");
                if n == 0 {
                    panic!("connection closed before headers");
                }
                buffer.extend_from_slice(&chunk[..n]);
                if let Some(index) = find_subsequence(&buffer, b"\r\n\r\n") {
                    header_end = index + 4;
                    break;
                }
            }

            let header_text = String::from_utf8_lossy(&buffer[..header_end]).to_string();
            let content_length = header_text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            while buffer.len() < header_end + content_length {
                let mut chunk = [0_u8; 1024];
                let n = socket.read(&mut chunk).await.expect("read request body");
                if n == 0 {
                    break;
                }
                buffer.extend_from_slice(&chunk[..n]);
            }

            let request_line = header_text.lines().next().expect("request line");
            let path = request_line
                .split_whitespace()
                .nth(1)
                .expect("request path")
                .to_string();
            let authorization = header_text.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("authorization")
                    .then(|| value.trim().to_string())
            });
            let body =
                serde_json::from_slice::<Value>(&buffer[header_end..header_end + content_length])
                    .expect("request json");
            sender
                .send(CapturedRequest {
                    path,
                    authorization,
                    body,
                })
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "receiver dropped"))
                .expect("send captured request");

            let response = response_body.to_string();
            let http_response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            socket
                .write_all(http_response.as_bytes())
                .await
                .expect("write response");
        });

        (format!("http://{addr}/api/v1"), receiver)
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    fn test_model(base_url: String) -> ImagesModel {
        let mut model = get_image_model("openrouter", "google/gemini-2.5-flash-image")
            .expect("openrouter image model");
        model.base_url = base_url;
        model.output = vec![ModelInput::Text, ModelInput::Image];
        model
    }

    #[tokio::test]
    async fn generate_images_returns_text_images_usage_and_sends_payload() {
        let response_body = json!({
            "id": "resp_1",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {
                    "cached_tokens": 4,
                    "cache_write_tokens": 1
                }
            },
            "choices": [
                {
                    "message": {
                        "content": "Here is your image.",
                        "images": [
                            { "image_url": "data:image/png;base64,ZmFrZS1wbmc=" }
                        ]
                    }
                }
            ]
        });
        let (base_url, request_receiver) = serve_once(response_body).await;
        let model = test_model(base_url);

        let output = generate_images(
            model,
            ImagesContext {
                input: vec![
                    ImagesContent::text("Draw a red circle."),
                    ImagesContent::Image(ImageContent {
                        data: "YWJj".to_string(),
                        mime_type: "image/png".to_string(),
                    }),
                ],
            },
            Some(ImagesOptions {
                api_key: Some("test-key".to_string()),
                ..ImagesOptions::default()
            }),
        )
        .await
        .expect("image generation");

        assert_eq!(output.response_id.as_deref(), Some("resp_1"));
        assert_eq!(
            output.output[0],
            ImagesOutputContent::text("Here is your image.")
        );
        assert_eq!(
            output.output[1],
            ImagesOutputContent::image("ZmFrZS1wbmc=", "image/png")
        );
        let usage = output.usage.expect("usage");
        assert_eq!(usage.input, 6);
        assert_eq!(usage.output, 5);
        assert_eq!(usage.cache_read, 3);
        assert_eq!(usage.cache_write, 1);
        assert_eq!(usage.total_tokens, 15);

        let request = request_receiver.await.expect("captured request");
        assert_eq!(request.path, "/api/v1/chat/completions");
        assert_eq!(request.authorization.as_deref(), Some("Bearer test-key"));
        assert_eq!(request.body["model"], "google/gemini-2.5-flash-image");
        assert_eq!(request.body["stream"], false);
        assert_eq!(request.body["modalities"], json!(["image", "text"]));
        assert_eq!(
            request.body["messages"][0]["content"][1]["image_url"]["url"],
            "data:image/png;base64,YWJj"
        );
    }

    #[tokio::test]
    async fn missing_api_key_returns_error_output() {
        let model = test_model("http://127.0.0.1:1/api/v1".to_string());
        let output = generate_images(
            model,
            ImagesContext {
                input: vec![ImagesContent::text("Draw.")],
            },
            Some(ImagesOptions::default()),
        )
        .await
        .expect("provider encodes failures as assistant images");

        assert_eq!(output.stop_reason, ImagesStopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("No API key for provider: openrouter")
        );
    }
}
