//! ai.rs extra, not part of Pi: embeddings over the OpenAI-compatible
//! `/embeddings` endpoint (`openai-embeddings`), shaped like the
//! `openai-images` module. The request goes through `OpenAIClient` and
//! `retry_provider_request`, like Pi's OpenAI API modules.
//!
//! - Every string of `context.input` goes upstream in one `input` array, also
//!   for a single string (some compatible servers reject scalar input).
//!   Results come back in input order; a response whose `index` values do not
//!   cover the inputs exactly once is an error.
//! - `options.dimensions` becomes `dimensions`. `provider_options` keys
//!   `encodingFormat`/`encoding_format` and `user` are forwarded.
//! - Usage reports `prompt_tokens` as input tokens, priced at the model's
//!   `cost.input` per million.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::openai_client::{OpenAIClient, OpenAIRequestOptions};
use crate::types::{
    EmbeddingModel, EmbeddingVector, EmbeddingsContext, EmbeddingsOptions, EmbeddingsResult,
    EmbeddingsStopReason, KnownEmbeddingApi, ProviderEmbeddings, ProviderHeaders, ProviderResponse,
    Usage, UsageCost,
};
use crate::utils::error_body::{format_provider_error, normalize_provider_error};
use crate::utils::headers::headers_to_record;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request};
use crate::{Error, Result};

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingsResponse {
    data: Vec<OpenAIEmbeddingsData>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<OpenAIEmbeddingsUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingsData {
    embedding: EmbeddingVector,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingsUsage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
}

/// Embed through `/embeddings`. Never fails: errors are reported in the
/// result.
pub async fn embed(
    model: EmbeddingModel,
    context: EmbeddingsContext,
    options: EmbeddingsOptions,
) -> EmbeddingsResult {
    let mut output = EmbeddingsResult::empty_for(&model);
    if let Err(error) = run(&model, context, &options, &mut output).await {
        output.embeddings.clear();
        output.stop_reason = if options
            .signal
            .as_ref()
            .is_some_and(|signal| signal.is_cancelled())
        {
            EmbeddingsStopReason::Aborted
        } else {
            EmbeddingsStopReason::Error
        };
        output.error_message = Some(format_provider_error(
            &normalize_provider_error(&error),
            None,
        ));
    }
    output
}

async fn run(
    model: &EmbeddingModel,
    context: EmbeddingsContext,
    options: &EmbeddingsOptions,
    output: &mut EmbeddingsResult,
) -> Result<()> {
    if model.api != KnownEmbeddingApi::OpenaiEmbeddings.as_str() {
        return Err(Error::message(format!(
            "Mismatched api: {} expected {}",
            model.api,
            KnownEmbeddingApi::OpenaiEmbeddings.as_str()
        )));
    }
    if context.input.is_empty() {
        return Err(Error::message(
            "embedding input must contain at least one string",
        ));
    }
    let Some(api_key) = options.api_key.as_deref().filter(|key| !key.is_empty()) else {
        return Err(Error::message(format!(
            "No API key for provider: {}",
            model.provider
        )));
    };
    let expected_count = context.input.len();
    let mut payload = Value::Object(build_payload(model, context, options));
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
    let response: OpenAIEmbeddingsResponse = serde_json::from_str(&body)?;
    if let Some(usage) = response.usage {
        output.usage = Some(parse_usage(usage, model));
    }
    output.response_model = response.model.filter(|value| *value != model.id);
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
    output.embeddings = data.into_iter().map(|item| item.embedding).collect();
    Ok(())
}

fn build_payload(
    model: &EmbeddingModel,
    context: EmbeddingsContext,
    options: &EmbeddingsOptions,
) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(model.id));
    payload.insert("input".to_string(), json!(context.input));
    if let Some(dimensions) = options.dimensions {
        payload.insert("dimensions".to_string(), json!(dimensions));
    }
    for (source, target) in [
        ("encodingFormat", "encoding_format"),
        ("encoding_format", "encoding_format"),
        ("user", "user"),
    ] {
        if let Some(value) = options.provider_options.get(source) {
            payload.insert(target.to_string(), value.clone());
        }
    }
    payload
}

fn parse_usage(raw_usage: OpenAIEmbeddingsUsage, model: &EmbeddingModel) -> Usage {
    let input = raw_usage.prompt_tokens.unwrap_or_default();
    let total_tokens = raw_usage.total_tokens.unwrap_or(input);
    let mut cost = UsageCost {
        input: (model.cost.input / 1_000_000.0) * f64::from(input),
        ..Default::default()
    };
    cost.total = cost.input;
    Usage {
        input,
        total_tokens,
        cost,
        ..Default::default()
    }
}

struct OpenAIEmbeddingsApi;

#[async_trait]
impl ProviderEmbeddings for OpenAIEmbeddingsApi {
    async fn embed(
        &self,
        model: EmbeddingModel,
        context: EmbeddingsContext,
        options: EmbeddingsOptions,
    ) -> EmbeddingsResult {
        embed(model, context, options).await
    }
}

/// The `openai-embeddings` implementation.
pub fn openai_embeddings_api() -> Arc<dyn ProviderEmbeddings> {
    Arc::new(OpenAIEmbeddingsApi)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;
    use crate::api::openai_client::test_support::{MockResponse, MockServer};
    use crate::auth::AuthContext;
    use crate::models::{CreateModelsOptions, Models};
    use crate::providers::all::builtin_models;
    use crate::types::ModelType;

    struct EnvContext(HashMap<&'static str, &'static str>);

    #[async_trait]
    impl AuthContext for EnvContext {
        async fn env(&self, name: &str) -> Option<String> {
            self.0.get(name).map(|value| value.to_string())
        }

        async fn file_exists(&self, _path: &str) -> bool {
            false
        }
    }

    /// The built-in providers with only the given environment visible to auth.
    fn models_with_env(env: &[(&'static str, &'static str)]) -> Models {
        builtin_models(CreateModelsOptions {
            auth_context: Some(Arc::new(EnvContext(env.iter().copied().collect()))),
            ..Default::default()
        })
    }

    /// A catalog embedding model pointed at the mock server.
    fn embedding_model(
        models: &Models,
        provider: &str,
        id: &str,
        base_url: &str,
    ) -> EmbeddingModel {
        let mut model = models
            .get_model_of_type(ModelType::Embedding, provider, id)
            .and_then(|model| model.as_embedding().cloned())
            .expect("catalog embedding model");
        model.base_url = base_url.to_string();
        model
    }

    fn json_response(body: Value) -> MockResponse {
        MockResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: body.to_string(),
        }
    }

    fn context(input: &[&str]) -> EmbeddingsContext {
        EmbeddingsContext {
            input: input.iter().map(|input| input.to_string()).collect(),
        }
    }

    #[tokio::test]
    async fn single_input_uses_one_item_upstream_array_and_resolves_auth() {
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": [0.25, 0.5], "index": 0 }],
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 2, "total_tokens": 2 },
        }))])
        .await;
        let models = models_with_env(&[("OPENAI_API_KEY", "test-key")]);
        let model = embedding_model(&models, "openai", "text-embedding-3-small", &server.url);

        let output = models
            .embed(&model, &context(&["hello"]), EmbeddingsOptions::default())
            .await;

        assert_eq!(output.stop_reason, EmbeddingsStopReason::Stop);
        assert_eq!(output.error_message, None);
        assert_eq!(output.api, "openai-embeddings");
        assert_eq!(output.provider, "openai");
        assert_eq!(output.model, "text-embedding-3-small");
        assert_eq!(output.response_model, None);
        assert_eq!(
            output.embeddings,
            vec![EmbeddingVector::Float(vec![0.25, 0.5])]
        );
        let usage = output.usage.unwrap();
        assert_eq!((usage.input, usage.output, usage.total_tokens), (2, 0, 2));
        assert!((usage.cost.input - 0.02 * 2.0 / 1e6).abs() < 1e-15);
        assert_eq!(usage.cost.total, usage.cost.input);
        let request = server.last();
        assert_eq!(request.path, "/v1/embeddings");
        assert_eq!(request.header("authorization"), Some("Bearer test-key"));
        assert_eq!(
            request.body,
            json!({ "model": "text-embedding-3-small", "input": ["hello"] })
        );
    }

    #[tokio::test]
    async fn batches_preserve_index_order_and_forward_options() {
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": [3.0], "index": 1 }, { "embedding": [1.0], "index": 0 }],
            "model": "upstream-model",
            "usage": { "prompt_tokens": 4, "total_tokens": 4 },
        }))])
        .await;
        let models = models_with_env(&[]);
        let model = embedding_model(&models, "openai", "text-embedding-3-large", &server.url);
        let options = EmbeddingsOptions {
            api_key: Some("explicit-key".to_string()),
            dimensions: Some(256),
            provider_options: json!({ "encodingFormat": "float", "user": "test-user" })
                .as_object()
                .unwrap()
                .clone(),
            ..Default::default()
        };

        let output = models
            .embed(&model, &context(&["first", "second"]), options)
            .await;

        assert_eq!(output.stop_reason, EmbeddingsStopReason::Stop);
        assert_eq!(
            output.embeddings,
            vec![
                EmbeddingVector::Float(vec![1.0]),
                EmbeddingVector::Float(vec![3.0])
            ]
        );
        assert_eq!(output.response_model.as_deref(), Some("upstream-model"));
        assert!((output.usage.unwrap().cost.total - 0.13 * 4.0 / 1e6).abs() < 1e-15);
        let request = server.last();
        assert_eq!(request.header("authorization"), Some("Bearer explicit-key"));
        assert_eq!(request.body["model"], "text-embedding-3-large");
        assert_eq!(request.body["input"], json!(["first", "second"]));
        assert_eq!(request.body["dimensions"], 256);
        assert_eq!(request.body["encoding_format"], "float");
        assert_eq!(request.body["user"], "test-user");
    }

    #[tokio::test]
    async fn base64_vectors_and_missing_usage() {
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": "AACAPw==", "index": 0 }],
        }))])
        .await;
        let models = models_with_env(&[("OPENAI_API_KEY", "test-key")]);
        let model = embedding_model(&models, "openai", "text-embedding-ada-002", &server.url);
        let options = EmbeddingsOptions {
            provider_options: json!({ "encoding_format": "base64" })
                .as_object()
                .unwrap()
                .clone(),
            ..Default::default()
        };

        let output = models.embed(&model, &context(&["hi"]), options).await;

        assert_eq!(
            output.embeddings,
            vec![EmbeddingVector::Base64("AACAPw==".to_string())]
        );
        assert_eq!(output.usage, None);
        assert_eq!(server.last().body["encoding_format"], "base64");
    }

    #[tokio::test]
    async fn rejects_missing_or_duplicate_response_indices_in_band() {
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": [1.0], "index": 0 }, { "embedding": [2.0], "index": 0 }],
        }))])
        .await;
        let models = models_with_env(&[("OPENAI_API_KEY", "test-key")]);
        let model = embedding_model(&models, "openai", "text-embedding-3-small", &server.url);

        let output = models
            .embed(
                &model,
                &context(&["first", "second"]),
                EmbeddingsOptions::default(),
            )
            .await;

        assert_eq!(output.stop_reason, EmbeddingsStopReason::Error);
        assert!(output.embeddings.is_empty());
        assert_eq!(
            output.error_message.as_deref(),
            Some("expected 2 indexed embeddings, received indices [0, 0]")
        );
    }

    #[tokio::test]
    async fn http_errors_are_reported_in_band() {
        let server = MockServer::start(vec![MockResponse {
            status: 400,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: json!({ "error": { "message": "bad input", "type": "invalid_request_error" } })
                .to_string(),
        }])
        .await;
        let models = models_with_env(&[("OPENAI_API_KEY", "test-key")]);
        let model = embedding_model(&models, "openai", "text-embedding-3-small", &server.url);

        let output = models
            .embed(&model, &context(&["hello"]), EmbeddingsOptions::default())
            .await;

        assert_eq!(output.stop_reason, EmbeddingsStopReason::Error);
        let message = output.error_message.unwrap();
        assert!(message.contains("400"), "{message}");
        assert!(message.contains("bad input"), "{message}");
    }

    #[tokio::test]
    async fn rejects_empty_input_unconfigured_auth_and_missing_keys() {
        let models = models_with_env(&[("OPENAI_API_KEY", "test-key")]);
        let model = embedding_model(
            &models,
            "openai",
            "text-embedding-3-small",
            "http://127.0.0.1:9/v1",
        );
        let output = models
            .embed(&model, &context(&[]), EmbeddingsOptions::default())
            .await;
        assert_eq!(output.stop_reason, EmbeddingsStopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("embedding input must contain at least one string")
        );

        let unconfigured = models_with_env(&[]);
        let output = unconfigured
            .embed(&model, &context(&["hi"]), EmbeddingsOptions::default())
            .await;
        assert_eq!(output.stop_reason, EmbeddingsStopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("Provider is not configured: openai")
        );

        // Called directly, the implementation needs a key.
        let output = embed(
            model.clone(),
            context(&["hi"]),
            EmbeddingsOptions::default(),
        )
        .await;
        assert_eq!(
            output.error_message.as_deref(),
            Some("No API key for provider: openai")
        );

        let output = embed(
            EmbeddingModel {
                api: "other".to_string(),
                ..model
            },
            context(&["hi"]),
            EmbeddingsOptions::default(),
        )
        .await;
        assert_eq!(
            output.error_message.as_deref(),
            Some("Mismatched api: other expected openai-embeddings")
        );
    }

    #[tokio::test]
    async fn copilot_embedding_models_send_copilot_headers() {
        let server = MockServer::start(vec![json_response(json!({
            "data": [{ "embedding": [0.5], "index": 0 }],
        }))])
        .await;
        let models = models_with_env(&[("COPILOT_GITHUB_TOKEN", "copilot-token")]);
        let model = embedding_model(
            &models,
            "github-copilot",
            "text-embedding-3-small",
            &server.url,
        );
        assert_eq!(model.api, "openai-embeddings");

        let output = models
            .embed(&model, &context(&["hello"]), EmbeddingsOptions::default())
            .await;

        assert_eq!(output.stop_reason, EmbeddingsStopReason::Stop);
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
