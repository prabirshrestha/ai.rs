use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::provider::EmbeddingModelApi;
use base64::Engine as _;

use crate::types::{
    EmbeddingBatch, EmbeddingInput, EmbeddingMediaFormat, EmbeddingOptions, EmbeddingUsage,
    EmbeddingVector, KnownApi, Model, ProviderResponse,
};
use crate::utils::headers::headers_to_record;
use crate::utils::http::{request_timeout, send_request_with_retries};
use crate::{Error, Result};

#[derive(Clone)]
pub(crate) struct OpenAiEmbeddingModelApi {
    api_key: Option<String>,
    allow_missing_api_key: bool,
    http_client: Option<reqwest::Client>,
}

impl OpenAiEmbeddingModelApi {
    pub(crate) fn new(
        api_key: Option<String>,
        allow_missing_api_key: bool,
        http_client: Option<reqwest::Client>,
    ) -> Self {
        Self {
            api_key,
            allow_missing_api_key,
            http_client,
        }
    }

    fn with_runtime_options(&self, mut options: EmbeddingOptions) -> EmbeddingOptions {
        if options
            .base
            .api_key
            .as_deref()
            .is_none_or(|api_key| api_key.trim().is_empty())
            && let Some(api_key) = &self.api_key
        {
            options.base.api_key = Some(api_key.clone());
        }
        if options.base.http_client.is_none() {
            options.base.http_client = self.http_client.clone();
        }
        options
    }
}

#[async_trait]
impl EmbeddingModelApi for OpenAiEmbeddingModelApi {
    fn id(&self) -> &str {
        KnownApi::OpenaiEmbeddings.as_str()
    }

    async fn embed_many(
        &self,
        model: Model,
        inputs: Vec<EmbeddingInput>,
        options: EmbeddingOptions,
    ) -> Result<EmbeddingBatch> {
        embed_many_openai(
            model,
            inputs,
            self.with_runtime_options(options),
            self.allow_missing_api_key,
        )
        .await
    }
}

pub(crate) async fn embed_many_openai(
    model: Model,
    inputs: Vec<EmbeddingInput>,
    options: EmbeddingOptions,
    allow_missing_api_key: bool,
) -> Result<EmbeddingBatch> {
    if model.api != KnownApi::OpenaiEmbeddings.as_str() {
        return Err(Error::UnsupportedApi(format!(
            "Mismatched api: {} expected {}",
            model.api,
            KnownApi::OpenaiEmbeddings.as_str()
        )));
    }
    let api_key = options
        .base
        .api_key
        .as_deref()
        .filter(|api_key| !api_key.trim().is_empty());
    if api_key.is_none() && !allow_missing_api_key {
        return Err(Error::MissingApiKey(model.provider.clone()));
    }

    // vLLM takes one media input per request, as a chat message. Text still
    // goes together in one request. Results are put back in input order.
    if options.media_format == EmbeddingMediaFormat::ChatMessages
        && inputs.iter().any(|input| !input.is_text())
    {
        let mut slots: Vec<Option<EmbeddingVector>> = vec![None; inputs.len()];
        let mut usage: Option<EmbeddingUsage> = None;
        let mut response_model = None;
        let (texts, media): (Vec<_>, Vec<_>) = inputs
            .into_iter()
            .enumerate()
            .partition(|(_, input)| input.is_text());
        if !texts.is_empty() {
            let (indices, texts): (Vec<_>, Vec<_>) = texts.into_iter().unzip();
            let payload = build_payload(&model, &texts, &options);
            let batch = send_embeddings(&model, payload, texts.len(), api_key, &options).await?;
            usage = EmbeddingUsage::sum(usage, batch.usage);
            response_model.get_or_insert(batch.model);
            for (index, embedding) in indices.into_iter().zip(batch.embeddings) {
                slots[index] = Some(embedding);
            }
        }
        for (index, input) in media {
            let payload = build_chat_media_payload(&model, &input, &options);
            let batch = send_embeddings(&model, payload, 1, api_key, &options).await?;
            usage = EmbeddingUsage::sum(usage, batch.usage);
            response_model.get_or_insert(batch.model);
            slots[index] = batch.embeddings.into_iter().next();
        }
        return Ok(EmbeddingBatch {
            embeddings: slots
                .into_iter()
                .map(|slot| slot.expect("every slot filled"))
                .collect(),
            model: response_model.unwrap_or(model.id),
            usage,
        });
    }

    let expected_count = inputs.len();
    let payload = build_payload(&model, &inputs, &options);
    send_embeddings(&model, payload, expected_count, api_key, &options).await
}

async fn send_embeddings(
    model: &Model,
    mut payload: Map<String, Value>,
    expected_count: usize,
    api_key: Option<&str>,
    options: &EmbeddingOptions,
) -> Result<EmbeddingBatch> {
    if let Some(on_payload) = &options.base.on_payload
        && let Some(next_payload) = on_payload(Value::Object(payload.clone()), model).await?
    {
        payload = next_payload.as_object().cloned().ok_or_else(|| {
            Error::Provider("OpenAI embeddings payload hook must return a JSON object".to_string())
        })?;
    }

    let client = options.base.http_client.clone().unwrap_or_default();
    let url = format!("{}/embeddings", model.base_url.trim_end_matches('/'));
    let headers = build_headers(api_key, &model.headers, &options.base.headers)?;
    let response = send_request_with_retries(&options.base, || {
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

    let response: OpenAiEmbeddingResponse = serde_json::from_str(&body).map_err(|error| {
        Error::InvalidProviderResponse(format!("could not decode embeddings response: {error}"))
    })?;
    let mut data = response.data;
    data.sort_by_key(|item| item.index);
    if data.len() != expected_count
        || data
            .iter()
            .enumerate()
            .any(|(expected_index, item)| item.index != expected_index)
    {
        return Err(Error::InvalidProviderResponse(format!(
            "expected {expected_count} indexed embeddings, received indices {:?}",
            data.iter().map(|item| item.index).collect::<Vec<_>>()
        )));
    }
    Ok(EmbeddingBatch {
        embeddings: data.into_iter().map(|item| item.embedding).collect(),
        model: response.model.unwrap_or_else(|| model.id.clone()),
        usage: response.usage.map(EmbeddingUsage::from),
    })
}

fn build_payload(
    model: &Model,
    inputs: &[EmbeddingInput],
    options: &EmbeddingOptions,
) -> Map<String, Value> {
    let input: Vec<Value> = inputs
        .iter()
        .map(|input| match input {
            EmbeddingInput::Text(text) => json!(text),
            EmbeddingInput::Image { data, mime_type } => json!({
                "content": [{"type": "image_url", "image_url": {"url": data_url(mime_type, data)}}],
            }),
            EmbeddingInput::Audio { data, mime_type } => json!({
                "content": [{
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64::engine::general_purpose::STANDARD.encode(data),
                        "format": audio_format(mime_type),
                    },
                }],
            }),
        })
        .collect();
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(model.id));
    payload.insert("input".to_string(), Value::Array(input));
    insert_common_options(&mut payload, options);
    payload
}

/// OpenAI's `input_audio.format`: the subtype of the audio MIME type.
fn audio_format(mime_type: &str) -> &str {
    match mime_type {
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/x-wav" | "audio/wave" => "wav",
        other => {
            let subtype = other.split(';').next().unwrap_or(other).trim();
            subtype.strip_prefix("audio/").unwrap_or(subtype)
        }
    }
}

fn build_chat_media_payload(
    model: &Model,
    input: &EmbeddingInput,
    options: &EmbeddingOptions,
) -> Map<String, Value> {
    let part = match input {
        EmbeddingInput::Text(text) => json!({"type": "text", "text": text}),
        EmbeddingInput::Image { data, mime_type } => json!({
            "type": "image_url",
            "image_url": {"url": data_url(mime_type, data)},
        }),
        EmbeddingInput::Audio { data, mime_type } => json!({
            "type": "audio_url",
            "audio_url": {"url": data_url(mime_type, data)},
        }),
    };
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(model.id));
    payload.insert(
        "messages".to_string(),
        json!([{"role": "user", "content": [part]}]),
    );
    insert_common_options(&mut payload, options);
    payload
}

fn data_url(mime_type: &str, data: &[u8]) -> String {
    format!(
        "data:{mime_type};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(data)
    )
}

fn insert_common_options(payload: &mut Map<String, Value>, options: &EmbeddingOptions) {
    if let Some(dimensions) = options.dimensions {
        payload.insert("dimensions".to_string(), json!(dimensions));
    }
    if let Some(encoding_format) = options.encoding_format {
        payload.insert(
            "encoding_format".to_string(),
            json!(encoding_format.as_str()),
        );
    }
    if let Some(user) = &options.user {
        payload.insert("user".to_string(), json!(user));
    }
}

fn build_headers(
    api_key: Option<&str>,
    model_headers: &std::collections::HashMap<String, String>,
    option_headers: &crate::types::ProviderHeaders,
) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Some(api_key) = api_key {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|error| Error::InvalidHeaderValue("authorization".to_string(), error))?,
        );
    }
    for (name, value) in model_headers {
        let name = name
            .parse::<HeaderName>()
            .map_err(|error| Error::Provider(format!("invalid header name: {error}")))?;
        let value = HeaderValue::from_str(value)
            .map_err(|error| Error::InvalidHeaderValue(name.as_str().to_string(), error))?;
        headers.insert(name, value);
    }
    crate::utils::headers::apply_provider_headers(&mut headers, option_headers)?;
    Ok(headers)
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<OpenAiEmbeddingUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiEmbeddingUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAiPromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

impl From<OpenAiEmbeddingUsage> for EmbeddingUsage {
    fn from(usage: OpenAiEmbeddingUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            total_tokens: usage.total_tokens,
            cached_tokens: usage
                .prompt_tokens_details
                .and_then(|details| details.cached_tokens),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingData {
    embedding: EmbeddingVector,
    index: usize,
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use crate::providers::openai;
    use crate::types::{EmbeddingEncodingFormat, EmbeddingVector};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn single_embedding_uses_one_item_upstream_array() {
        let captured = Arc::new(Mutex::new(String::new()));
        let url = spawn_response_server(
            Arc::clone(&captured),
            r#"{"data":[{"embedding":[0.25,0.5],"index":0}],"usage":{"prompt_tokens":2,"total_tokens":2}}"#,
        )
        .await;
        let provider = openai::builder()
            .api_key(Some("test-key"))
            .base_url(url)
            .build()
            .expect("provider");
        let model = provider
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .expect("model");

        let output = crate::embed(model, "hello", None).await.expect("embedding");

        assert_eq!(output.embedding, EmbeddingVector::Float(vec![0.25, 0.5]));
        assert_eq!(output.model, "text-embedding-3-small");
        let request = captured.lock().expect("request").clone();
        assert!(request.starts_with("POST /v1/embeddings HTTP/1.1"));
        assert_eq!(request_body_json(&request)["input"], json!(["hello"]));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn batch_embedding_preserves_index_order_and_options() {
        let captured = Arc::new(Mutex::new(String::new()));
        let url = spawn_response_server(
            Arc::clone(&captured),
            r#"{"data":[{"embedding":[3.0],"index":1},{"embedding":[1.0],"index":0}],"model":"upstream-model","usage":{"prompt_tokens":4,"total_tokens":4}}"#,
        )
        .await;
        let provider = openai::builder()
            .api_key(Some("test-key"))
            .base_url(url)
            .build()
            .expect("provider");
        let model = provider
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .expect("model");
        let options = EmbeddingOptions {
            dimensions: Some(256),
            encoding_format: Some(EmbeddingEncodingFormat::Float),
            user: Some("test-user".to_string()),
            ..Default::default()
        };

        let output = crate::embed_many(model, ["first", "second"], Some(options))
            .await
            .expect("embeddings");

        assert_eq!(
            output.embeddings,
            vec![
                EmbeddingVector::Float(vec![1.0]),
                EmbeddingVector::Float(vec![3.0])
            ]
        );
        assert_eq!(output.model, "upstream-model");
        let payload = request_body_json(&captured.lock().expect("request").clone());
        assert_eq!(payload["input"], json!(["first", "second"]));
        assert_eq!(payload["dimensions"], 256);
        assert_eq!(payload["encoding_format"], "float");
        assert_eq!(payload["user"], "test-user");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn custom_endpoint_without_api_key_omits_authorization() {
        let captured = Arc::new(Mutex::new(String::new()));
        let url = spawn_response_server(
            Arc::clone(&captured),
            r#"{"data":[{"embedding":[0.25],"index":0}]}"#,
        )
        .await;
        let provider = openai::builder().base_url(url).build().expect("provider");
        let model = provider
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .expect("model");

        crate::embed(model, "hello", None).await.expect("embedding");

        let request = captured.lock().expect("request").to_ascii_lowercase();
        assert!(!request.contains("\r\nauthorization:"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rejects_missing_or_duplicate_response_indices() {
        let captured = Arc::new(Mutex::new(String::new()));
        let url = spawn_response_server(
            captured,
            r#"{"data":[{"embedding":[1.0],"index":0},{"embedding":[2.0],"index":0}]}"#,
        )
        .await;
        let provider = openai::builder()
            .api_key(Some("test-key"))
            .base_url(url)
            .build()
            .expect("provider");
        let model = provider
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .expect("model");

        let error = crate::embed_many(model, ["first", "second"], None)
            .await
            .expect_err("invalid indices should fail");

        assert!(matches!(error, Error::InvalidProviderResponse(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn content_parts_media_go_in_input_next_to_text() {
        let captured = Arc::new(Mutex::new(String::new()));
        let url = spawn_response_server(
            Arc::clone(&captured),
            r#"{"data":[{"embedding":[1.0],"index":0},{"embedding":[2.0],"index":1},{"embedding":[3.0],"index":2}],"usage":{"prompt_tokens":9,"total_tokens":9,"prompt_tokens_details":{"cached_tokens":4}}}"#,
        )
        .await;
        let model = test_model(url);

        let output = crate::embed_many(
            model,
            [
                EmbeddingInput::from("text"),
                EmbeddingInput::image(b"png".to_vec(), "image/png"),
                EmbeddingInput::audio(b"wav".to_vec(), "audio/wav"),
            ],
            None,
        )
        .await
        .expect("embeddings");

        assert_eq!(output.embeddings.len(), 3);
        assert_eq!(output.usage.and_then(|u| u.cached_tokens), Some(4));
        let payload = request_body_json(&captured.lock().expect("request").clone());
        assert_eq!(
            payload["input"],
            json!([
                "text",
                {"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}]},
                {"content": [{"type": "input_audio", "input_audio": {"data": "d2F2", "format": "wav"}}]},
            ])
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn chat_messages_send_each_media_input_alone_and_keep_order() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let url = spawn_multi_response_server(
            Arc::clone(&captured),
            vec![
                r#"{"data":[{"embedding":[1.0],"index":0},{"embedding":[3.0],"index":1}],"usage":{"prompt_tokens":2,"total_tokens":2}}"#,
                r#"{"data":[{"embedding":[2.0],"index":0}],"usage":{"prompt_tokens":5,"total_tokens":5}}"#,
            ],
        )
        .await;
        let model = test_model(url);
        let options = EmbeddingOptions {
            media_format: EmbeddingMediaFormat::ChatMessages,
            ..Default::default()
        };

        let output = crate::embed_many(
            model,
            [
                EmbeddingInput::from("first"),
                EmbeddingInput::image(b"png".to_vec(), "image/png"),
                EmbeddingInput::from("third"),
            ],
            Some(options),
        )
        .await
        .expect("embeddings");

        assert_eq!(
            output.embeddings,
            vec![
                EmbeddingVector::Float(vec![1.0]),
                EmbeddingVector::Float(vec![2.0]),
                EmbeddingVector::Float(vec![3.0]),
            ]
        );
        assert_eq!(output.usage.map(|u| u.prompt_tokens), Some(7));
        let requests = captured.lock().expect("requests").clone();
        assert_eq!(
            request_body_json(&requests[0])["input"],
            json!(["first", "third"])
        );
        assert_eq!(
            request_body_json(&requests[1])["messages"],
            json!([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}
            ]}])
        );
    }

    fn test_model(url: String) -> Model {
        openai::builder()
            .api_key(Some("test-key"))
            .base_url(url)
            .build()
            .expect("provider")
            .embedding_model("embeddinggemma-2")
            .build_embedding()
            .expect("model")
    }

    async fn spawn_multi_response_server(
        captured: Arc<Mutex<Vec<String>>>,
        bodies: Vec<&'static str>,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            for body in bodies {
                let (mut socket, _) = listener.accept().await.expect("accept");
                let request = read_http_request(&mut socket).await;
                captured.lock().expect("captured requests").push(request);
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\ncontent-length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("write response");
            }
        });
        format!("http://{addr}/v1")
    }

    async fn spawn_response_server(captured: Arc<Mutex<String>>, body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let request = read_http_request(&mut socket).await;
            *captured.lock().expect("captured request") = request;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
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
            buffer.extend_from_slice(&temp[..read]);
            if let Some(header_end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&buffer[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or_default();
                while buffer.len() < header_end + 4 + content_length {
                    let read = socket.read(&mut temp).await.expect("read body");
                    buffer.extend_from_slice(&temp[..read]);
                }
                break;
            }
        }
        String::from_utf8(buffer).expect("utf8 request")
    }

    fn request_body_json(request: &str) -> Value {
        serde_json::from_str(request.split_once("\r\n\r\n").expect("body").1).expect("json")
    }
}
