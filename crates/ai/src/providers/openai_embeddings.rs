use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::provider::EmbeddingModelApi;
use crate::types::{
    EmbeddingBatch, EmbeddingOptions, EmbeddingUsage, EmbeddingVector, KnownApi, Model,
    ProviderResponse,
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
        inputs: Vec<String>,
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
    inputs: Vec<String>,
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
    let expected_count = inputs.len();
    let mut payload = build_payload(&model, inputs, &options);
    if let Some(on_payload) = &options.base.on_payload
        && let Some(next_payload) = on_payload(Value::Object(payload.clone()), &model).await?
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
            &model,
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
        model: response.model.unwrap_or(model.id),
        usage: response.usage.unwrap_or_default(),
    })
}

fn build_payload(
    model: &Model,
    inputs: Vec<String>,
    options: &EmbeddingOptions,
) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(model.id));
    payload.insert("input".to_string(), json!(inputs));
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
    payload
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
    usage: Option<EmbeddingUsage>,
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
