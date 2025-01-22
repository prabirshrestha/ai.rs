use std::str::FromStr;

use crate::chat_completions::{ChatCompletion, ChatCompletionRequest, ChatCompletionResponse};
use crate::utils::uri::ensure_no_trailing_slash;
use crate::{Error, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use reqwest::header::HeaderName;
use secrecy::{ExposeSecret, SecretString};

pub const AZURE_OPENAI_API_KEY_ENV_VAR: &str = "AZURE_OPENAI_API_KEY";

#[derive(Debug, Clone, Builder)]
#[builder(derive(Debug))]
#[builder(setter(into))]
pub struct Client {
    #[builder(default)]
    http_client: reqwest::Client,
    api_version: String,
    base_url: String,
    auth: Auth,
}

#[derive(Debug, Clone)]
pub enum Auth {
    BearerToken(SecretString),
    ApiKey(SecretString),
}

impl Client {
    pub fn new(auth: Auth, base_url: &str, api_version: &str) -> Result<Self> {
        Ok(Self {
            http_client: reqwest::Client::default(),
            api_version: api_version.to_string(),
            base_url: ensure_no_trailing_slash(base_url),
            auth,
        })
    }
}

#[async_trait]
impl ChatCompletion for Client {
    async fn chat_completions(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        if let Some(stream) = request.stream {
            if stream {
                return Err(Error::StreamingNotSupported(
                    "Streaming is not supported when using chat_completions() api".to_string(),
                ));
            }
        }

        let mut headers = reqwest::header::HeaderMap::new();
        let (auth_header_key, auth_header_value) = match &self.auth {
            Auth::BearerToken(secret_box) => (
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!(
                    "Bearer {}",
                    secret_box.expose_secret()
                ))
                .map_err(|e| {
                    Error::InvalidHeaderValue(reqwest::header::AUTHORIZATION.to_string(), e)
                })?,
            ),
            Auth::ApiKey(secret_box) => (
                HeaderName::from_str("api-key")
                    .map_err(|e| Error::InvalidHeaderName("api-key".to_owned(), e))?,
                reqwest::header::HeaderValue::from_str(&secret_box.expose_secret())
                    .map_err(|e| Error::InvalidHeaderValue("api-key".to_string(), e))?,
            ),
        };
        headers.insert(auth_header_key, auth_header_value);

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        // NOTE: use model as the deployment_id
        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.base_url, &request.model, self.api_version
        );

        let response = self
            .http_client
            .post(url)
            .headers(headers)
            .json(request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::UnknownError(response.text().await?));
        }

        let chat_completion_response = response.json::<ChatCompletionResponse>().await?;

        Ok(chat_completion_response)
    }
}

impl super::Client for Client {}
