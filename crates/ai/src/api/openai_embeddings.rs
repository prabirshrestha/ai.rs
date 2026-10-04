//! ai.rs extra, not part of Pi: the OpenAI-compatible `/embeddings` API
//! (`openai-embeddings`) behind [`crate::embeddings`]. Requests go through
//! `OpenAIClient` and
//! `retry_provider_request`, like Pi's OpenAI API modules.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::openai_client::{OpenAIClient, OpenAIRequestOptions};
use crate::embeddings::{
    EmbeddingBatch, EmbeddingModel, EmbeddingOptions, EmbeddingUsage, EmbeddingVector,
};
use crate::types::{ProviderHeaders, ProviderResponse};
use crate::utils::headers::headers_to_record;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};
use crate::{Error, Result};

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

/// POST `{base_url}/embeddings` with every input in one array. Without an
/// API key the request fails unless `allow_missing_api_key` (keyless
/// OpenAI-compatible servers), in which case no `Authorization` header is
/// sent.
pub(crate) async fn embed_many_openai(
    model: &EmbeddingModel,
    inputs: Vec<String>,
    options: &EmbeddingOptions,
    allow_missing_api_key: bool,
) -> Result<EmbeddingBatch> {
    let api_key = options
        .api_key
        .as_deref()
        .filter(|api_key| !api_key.trim().is_empty());
    if api_key.is_none() && !allow_missing_api_key {
        return Err(Error::message(format!(
            "No API key for provider: {}",
            model.provider
        )));
    }
    let expected_count = inputs.len();
    let mut payload = Value::Object(build_payload(model, inputs, options));
    if let Some(on_payload) = &options.on_payload
        && let Some(next_payload) = on_payload(payload.clone(), model).await?
    {
        payload = next_payload;
    }

    let mut headers: ProviderHeaders = model.headers.clone().map(Into::into).unwrap_or_default();
    if api_key.is_none() {
        headers.insert("Authorization", None::<String>);
    }
    for (name, value) in options.headers.iter().flatten() {
        headers.insert(name.clone(), value.clone());
    }
    let client = OpenAIClient::new(
        api_key.unwrap_or_default(),
        &model.base_url,
        options.http_client.as_ref(),
        headers,
    );
    let request_options = OpenAIRequestOptions {
        signal: options.signal.clone(),
        timeout_ms: options.timeout_ms,
    };
    let response = retry_provider_request(
        || client.post("/embeddings", &payload, &request_options),
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
    let response: OpenAiEmbeddingResponse = serde_json::from_str(&body).map_err(|error| {
        Error::message(format!("could not decode embeddings response: {error}"))
    })?;
    let mut data = response.data;
    data.sort_by_key(|item| item.index);
    if data.len() != expected_count
        || data
            .iter()
            .enumerate()
            .any(|(expected_index, item)| item.index != expected_index)
    {
        return Err(Error::message(format!(
            "expected {expected_count} indexed embeddings, received indices {:?}",
            data.iter().map(|item| item.index).collect::<Vec<_>>()
        )));
    }
    Ok(EmbeddingBatch {
        embeddings: data.into_iter().map(|item| item.embedding).collect(),
        model: response.model.unwrap_or_else(|| model.id.clone()),
        usage: response.usage.unwrap_or_default(),
    })
}

fn build_payload(
    model: &EmbeddingModel,
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::api::openai_client::test_support::{MockResponse, MockServer};
    use crate::embeddings::{EmbeddingEncodingFormat, embed, embed_many};
    use crate::providers::{github_copilot, openai};

    fn json_response(body: Value) -> MockResponse {
        MockResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: body.to_string(),
        }
    }

    #[tokio::test]
    async fn single_embedding_uses_one_item_upstream_array() {
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": [0.25, 0.5], "index": 0 }],
            "usage": { "prompt_tokens": 2, "total_tokens": 2 },
        }))])
        .await;
        let provider = openai::builder()
            .api_key(Some("test-key"))
            .base_url(server.url.clone())
            .build()
            .unwrap();
        let model = provider
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .unwrap();

        let output = embed(model, "hello", None).await.unwrap();

        assert_eq!(output.embedding, EmbeddingVector::Float(vec![0.25, 0.5]));
        assert_eq!(output.model, "text-embedding-3-small");
        assert_eq!(output.usage.prompt_tokens, 2);
        let request = server.last();
        assert_eq!(request.path, "/v1/embeddings");
        assert_eq!(request.header("authorization"), Some("Bearer test-key"));
        assert_eq!(request.body["input"], json!(["hello"]));
    }

    #[tokio::test]
    async fn batch_embedding_preserves_index_order_and_options() {
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": [3.0], "index": 1 }, { "embedding": [1.0], "index": 0 }],
            "model": "upstream-model",
            "usage": { "prompt_tokens": 4, "total_tokens": 4 },
        }))])
        .await;
        let provider = openai::builder()
            .api_key(Some("test-key"))
            .base_url(server.url.clone())
            .build()
            .unwrap();
        let model = provider
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .unwrap();
        let options = EmbeddingOptions {
            dimensions: Some(256),
            encoding_format: Some(EmbeddingEncodingFormat::Float),
            user: Some("test-user".to_string()),
            ..Default::default()
        };

        let output = embed_many(model, ["first", "second"], Some(options))
            .await
            .unwrap();

        assert_eq!(
            output.embeddings,
            vec![
                EmbeddingVector::Float(vec![1.0]),
                EmbeddingVector::Float(vec![3.0])
            ]
        );
        assert_eq!(output.model, "upstream-model");
        let payload = server.last().body;
        assert_eq!(payload["input"], json!(["first", "second"]));
        assert_eq!(payload["dimensions"], 256);
        assert_eq!(payload["encoding_format"], "float");
        assert_eq!(payload["user"], "test-user");
    }

    #[tokio::test]
    async fn custom_endpoint_without_api_key_omits_authorization() {
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return;
        }
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": [0.25], "index": 0 }],
        }))])
        .await;
        let provider = openai::builder()
            .base_url(server.url.clone())
            .build()
            .unwrap();
        let model = provider
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .unwrap();

        embed(model, "hello", None).await.unwrap();

        assert_eq!(server.last().header("authorization"), None);
    }

    #[tokio::test]
    async fn rejects_missing_or_duplicate_response_indices() {
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": [1.0], "index": 0 }, { "embedding": [2.0], "index": 0 }],
        }))])
        .await;
        let provider = openai::builder()
            .api_key(Some("test-key"))
            .base_url(server.url.clone())
            .build()
            .unwrap();
        let model = provider
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .unwrap();

        let error = embed_many(model, ["first", "second"], None)
            .await
            .expect_err("invalid indices should fail");

        assert_eq!(
            error.to_string(),
            "expected 2 indexed embeddings, received indices [0, 0]"
        );
    }

    #[tokio::test]
    async fn copilot_embedding_models_use_the_handle_token_and_copilot_headers() {
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": [0.5], "index": 0 }],
        }))])
        .await;
        let provider = github_copilot::builder()
            .api_key("copilot-token")
            .base_url(server.url.clone())
            .build()
            .unwrap();
        let model = provider
            .embedding_model("text-embedding-3-small")
            .build_embedding()
            .unwrap();
        assert_eq!(model.api, "openai-embeddings");
        assert_eq!(model.base_url, server.url);

        embed(model, "hello", None).await.unwrap();

        let request = server.last();
        assert_eq!(request.path, "/v1/embeddings");
        assert_eq!(
            request.header("authorization"),
            Some("Bearer copilot-token")
        );
        assert_eq!(
            request.header("copilot-integration-id"),
            Some("vscode-chat")
        );
    }
}
