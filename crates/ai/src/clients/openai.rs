use crate::chat_completions::{ChatCompletion, ChatCompletionRequest, ChatCompletionResponse};
use crate::{Error, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use secrecy::{ExposeSecret, SecretString};

pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub const OPENAI_API_KEY_ENV_VAR: &str = "OPENAI_API_KEY";

#[derive(Debug, Clone, Builder)]
pub struct Client {
    http_client: reqwest::Client,
    api_key: SecretString,
    base_url: String,
}

impl Client {
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
                        .map_err(|e| {
                            Error::InvalidHeaderValue(reqwest::header::AUTHORIZATION.to_string(), e)
                        })?,
                );
                headers
            })
            .build()?;

        Ok(Self {
            api_key: SecretString::new(api_key.into()),
            base_url: base_url.into(),
            http_client,
        })
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|e| Error::EnvVarError(OPENAI_API_KEY_ENV_VAR.into(), e))?;
        Ok(Self::new(&api_key)?)
    }
}

#[async_trait]
impl ChatCompletion for Client {
    async fn complete(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!(
                "Bearer {}",
                self.api_key.expose_secret()
            ))
            .map_err(|e| {
                Error::InvalidHeaderValue(reqwest::header::AUTHORIZATION.to_string(), e)
            })?,
        );

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        let response = self
            .http_client
            .post(format!("{}/chat/completions", self.base_url))
            .headers(headers)
            .json(request)
            .send()
            .await?;

        if !response.status().is_success() {
            todo!()
        }

        let chat_completion_response = response.json::<ChatCompletionResponse>().await?;

        Ok(chat_completion_response)
    }
}

impl super::Client for Client {}
