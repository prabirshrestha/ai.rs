//! Text embeddings. **ai.rs extra, not part of Pi**: Pi 1.0 has no
//! embeddings API. This module keeps the pre-1.0 ai.rs surface ([`embed`],
//! [`embed_many`], `openai.embedding_model(..).build_embedding()`) on top of
//! the Pi 1.0 core: a model built from a provider handle resolves its
//! credentials, headers and base URL through the handle's
//! `Models` collection (`Models::get_auth`), like
//! chat and image models do.
//!
//! The only embeddings API is `openai-embeddings`, the OpenAI-compatible
//! `/embeddings` endpoint (see [`crate::api::openai_embeddings`]).

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::api::openai_embeddings::embed_many_openai;
use crate::auth::{AuthResolutionOverrides, models_error};
use crate::env_api_keys::get_env_api_key;
use crate::models::Models;
use crate::types::{BoxFuture, ModelInput, ProviderHeaders, ProviderId, ProviderResponse};
use crate::utils::models_error::ModelsErrorCode;
use crate::{Error, Result};

/// The OpenAI-compatible embeddings API id.
pub const OPENAI_EMBEDDINGS_API: &str = "openai-embeddings";

/// An embedding model. Usable with [`embed`] and [`embed_many`] only.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingModel {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: ProviderId,
    pub base_url: String,
    #[serde(default)]
    pub input: Vec<ModelInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    /// The provider handle this model was built from. Not serialized and
    /// ignored by `==`.
    #[serde(skip)]
    pub(crate) binding: Option<EmbeddingBinding>,
}

/// What a provider handle contributes to its embedding models.
#[derive(Clone)]
pub(crate) struct EmbeddingBinding {
    pub models: Models,
    pub http_client: Option<reqwest::Client>,
    /// A custom base URL without a key: send no `Authorization` header.
    pub keyless: bool,
}

impl fmt::Debug for EmbeddingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddingModel")
            .field("id", &self.id)
            .field("api", &self.api)
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("bound", &self.binding.is_some())
            .finish_non_exhaustive()
    }
}

impl PartialEq for EmbeddingModel {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.name == other.name
            && self.api == other.api
            && self.provider == other.provider
            && self.base_url == other.base_url
            && self.input == other.input
            && self.headers == other.headers
    }
}

/// `onPayload` for embedding requests.
pub type EmbeddingPayloadHook = std::sync::Arc<
    dyn Fn(Value, &EmbeddingModel) -> BoxFuture<Result<Option<Value>>> + Send + Sync,
>;
/// `onResponse` for embedding requests.
pub type EmbeddingResponseHook = std::sync::Arc<
    dyn Fn(ProviderResponse, &EmbeddingModel) -> BoxFuture<Result<()>> + Send + Sync,
>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingEncodingFormat {
    #[default]
    Float,
    Base64,
}

impl EmbeddingEncodingFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Base64 => "base64",
        }
    }
}

/// Request options for [`embed`] and [`embed_many`].
#[derive(Clone, Default)]
pub struct EmbeddingOptions {
    pub signal: Option<CancellationToken>,
    pub api_key: Option<String>,
    pub http_client: Option<reqwest::Client>,
    pub on_payload: Option<EmbeddingPayloadHook>,
    pub on_response: Option<EmbeddingResponseHook>,
    pub headers: Option<ProviderHeaders>,
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
    pub dimensions: Option<u32>,
    pub encoding_format: Option<EmbeddingEncodingFormat>,
    pub user: Option<String>,
}

impl fmt::Debug for EmbeddingOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddingOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("dimensions", &self.dimensions)
            .field("encoding_format", &self.encoding_format)
            .finish_non_exhaustive()
    }
}

/// A float vector, or a base64 string with `EmbeddingEncodingFormat::Base64`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    pub embedding: EmbeddingVector,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingBatch {
    pub embeddings: Vec<EmbeddingVector>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// Embed one string. Sent upstream as a one-item array for compatibility
/// with providers that reject scalar input.
pub async fn embed(
    model: EmbeddingModel,
    input: impl Into<String>,
    options: Option<EmbeddingOptions>,
) -> Result<Embedding> {
    let batch = embed_many(model, [input.into()], options).await?;
    let embedding = batch
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| Error::message("embedding response contained no data"))?;
    Ok(Embedding {
        embedding,
        model: batch.model,
        usage: batch.usage,
    })
}

/// Embed several strings in one request. Results keep the input order.
pub async fn embed_many<I, S>(
    model: EmbeddingModel,
    inputs: I,
    options: Option<EmbeddingOptions>,
) -> Result<EmbeddingBatch>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let inputs = inputs.into_iter().map(Into::into).collect::<Vec<String>>();
    if inputs.is_empty() {
        return Err(Error::Validation(
            "embedding input must contain at least one string".to_string(),
        ));
    }
    if model.api != OPENAI_EMBEDDINGS_API {
        return Err(Error::message(format!(
            "No embeddings API registered for api: {}",
            model.api
        )));
    }
    let mut options = options.unwrap_or_default();
    let mut request_model = model.clone();
    let mut allow_missing_api_key = false;
    match &model.binding {
        Some(binding) => {
            allow_missing_api_key = binding.keyless;
            if options.http_client.is_none() {
                options.http_client.clone_from(&binding.http_client);
            }
            let resolution = binding
                .models
                .get_auth(
                    model.provider.as_str(),
                    AuthResolutionOverrides {
                        api_key: options.api_key.clone(),
                        signal: options.signal.clone(),
                        ..Default::default()
                    },
                )
                .await?;
            let Some(resolution) = resolution else {
                return Err(models_error(
                    ModelsErrorCode::Auth,
                    format!("Provider is not configured: {}", model.provider),
                ));
            };
            let auth = resolution.auth;
            options.api_key = options.api_key.or(auth.api_key);
            if let Some(auth_headers) = auth.headers {
                let mut headers = auth_headers;
                for (name, value) in options.headers.iter().flatten() {
                    headers.insert(name.clone(), value.clone());
                }
                options.headers = Some(headers);
            }
            if let Some(base_url) = auth.base_url.filter(|base_url| !base_url.is_empty()) {
                request_model.base_url = base_url;
            }
        }
        None => {
            if options
                .api_key
                .as_deref()
                .is_none_or(|key| key.trim().is_empty())
            {
                options.api_key = get_env_api_key(&model.provider, None);
            }
        }
    }
    embed_many_openai(&request_model, inputs, &options, allow_missing_api_key).await
}

/// Builder for embedding models from provider handles
/// (`provider.embedding_model("id").build_embedding()`).
#[derive(Debug, Clone)]
pub struct EmbeddingModelBuilder {
    model: EmbeddingModel,
}

impl EmbeddingModelBuilder {
    pub(crate) fn new(model: EmbeddingModel) -> Self {
        Self { model }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.model.name = name.into();
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.model.base_url = base_url.into();
        self
    }

    pub fn input(mut self, input: impl Into<Vec<ModelInput>>) -> Self {
        self.model.input = input.into();
        self
    }

    pub fn headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.model
            .headers
            .get_or_insert_with(IndexMap::new)
            .extend(headers);
        self
    }

    pub fn build_embedding(self) -> Result<EmbeddingModel> {
        Ok(self.model)
    }

    /// Alias of [`EmbeddingModelBuilder::build_embedding`].
    pub fn build(self) -> Result<EmbeddingModel> {
        self.build_embedding()
    }
}

/// An `openai-embeddings` model bound to a provider handle.
pub(crate) fn bound_embedding_model(
    id: &str,
    provider: &str,
    base_url: &str,
    headers: Option<IndexMap<String, String>>,
    binding: EmbeddingBinding,
) -> EmbeddingModelBuilder {
    EmbeddingModelBuilder::new(EmbeddingModel {
        id: id.to_string(),
        name: id.to_string(),
        api: OPENAI_EMBEDDINGS_API.to_string(),
        provider: provider.to_string(),
        base_url: base_url.to_string(),
        input: vec![ModelInput::Text],
        headers,
        binding: Some(binding),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embed_many_rejects_empty_input_before_provider_dispatch() {
        let error = embed_many(EmbeddingModel::default(), Vec::<String>::new(), None)
            .await
            .expect_err("empty input should fail");
        assert!(matches!(error, Error::Validation(_)));
    }

    #[tokio::test]
    async fn rejects_unknown_embedding_apis() {
        let model = EmbeddingModel {
            api: "other".to_string(),
            ..Default::default()
        };
        let error = embed(model, "hi", None).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "No embeddings API registered for api: other"
        );
    }
}
