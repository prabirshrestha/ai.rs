use async_trait::async_trait;
use derive_builder::Builder;
use secrecy::{ExposeSecret, SecretString};

use crate::{
    completions::{CompletionProvider, CompletionRequest, CompletionResponse},
    embeddings::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse},
    Error, Result,
};

use super::AnyClient;

pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub const OPEN_API_KEY_ENV_VAR: &str = "OPENAI_API_KEY";

#[derive(Debug, Clone, Builder)]
pub struct OpenAIClient {
    api_key: SecretString,
    base_url: String,
    http_client: reqwest::Client,
}

impl OpenAIClient {
    pub fn new(api_key: &str) -> Result<Self> {
        Self::from_url(api_key, OPENAI_BASE_URL)
    }

    pub fn from_url(api_key: &str, base_url: &str) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    reqwest::header::HeaderValue::from_str(&format!("Bearer {}", api_key))
                        .expect("Failed to create Authorization header"),
                );
                headers
            })
            .build()?;

        let api_key = SecretString::new(api_key.into());
        Ok(Self {
            api_key,
            base_url: base_url.into(),
            http_client,
        })
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|e| Error::EnvVarError(OPEN_API_KEY_ENV_VAR.into(), e))?;
        Ok(Self::new(&api_key)?)
    }
}

impl AnyClient for OpenAIClient {}

#[async_trait]
impl CompletionProvider for OpenAIClient {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        todo!()
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIClient {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        todo!()
    }
}
