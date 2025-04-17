use std::pin::Pin;
use std::str::FromStr;

use crate::chat_completions::{
    ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
};
use crate::embeddings::{Embeddings, EmbeddingsRequest, EmbeddingsResponse};
use crate::utils::uri::ensure_no_trailing_slash;
use crate::{Error, Result};
use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::{Stream, StreamExt};
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

impl Client {
    fn get_headers(&self) -> Result<reqwest::header::HeaderMap> {
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

        Ok(headers)
    }

    fn get_url(&self, model: &str) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.base_url, model, self.api_version
        )
    }

    fn get_embeddings_url(&self, model: &str) -> String {
        format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.base_url, model, self.api_version
        )
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

        // Check if already cancelled before making the request
        if let Some(token) = &request.cancellation_token {
            if token.is_cancelled() {
                return Err(Error::Cancelled);
            }
        }

        let headers = self.get_headers()?;
        let url = self.get_url(&request.model);

        // Create an abortable request
        let (abort_handle, abort_registration) = futures::future::AbortHandle::new_pair();

        // If we have a cancellation token, set up cancellation monitoring
        if let Some(token) = &request.cancellation_token {
            let token = token.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                abort_handle.abort();
            });
        }

        let request_future = self
            .http_client
            .post(url)
            .headers(headers)
            .json(request)
            .send();

        let response =
            match futures::future::Abortable::new(request_future, abort_registration).await {
                Ok(response) => response?,
                Err(futures::future::Aborted) => {
                    return Err(Error::Cancelled);
                }
            };

        if !response.status().is_success() {
            return Err(Error::UnknownError(response.text().await?));
        }

        let chat_completion_response = response.json::<ChatCompletionResponse>().await?;

        Ok(chat_completion_response)
    }

    async fn stream_chat_completions(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
        if let Some(stream) = request.stream {
            if !stream {
                return Err(Error::StreamingNotSupported(
                    "Streaming required when using stream_chat_completions() api".to_string(),
                ));
            }
        }

        // Check if already cancelled before making the request
        if let Some(token) = &request.cancellation_token {
            if token.is_cancelled() {
                return Ok(Box::pin(futures::stream::empty()));
            }
        }

        let mut json = serde_json::to_value(request)?;
        json["stream"] = serde_json::Value::Bool(true);

        // Create an abortable request
        let (abort_handle, abort_registration) = futures::future::AbortHandle::new_pair();

        // If we have a cancellation token, set up cancellation monitoring
        if let Some(token) = &request.cancellation_token {
            let token = token.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                abort_handle.abort();
            });
        }

        let request_future = self
            .http_client
            .post(self.get_url(&request.model))
            .headers(self.get_headers()?)
            .body(json.to_string())
            .send();

        let response =
            match futures::future::Abortable::new(request_future, abort_registration).await {
                Ok(response) => response?,
                Err(futures::future::Aborted) => {
                    return Ok(Box::pin(futures::stream::empty()));
                }
            };

        if !response.status().is_success() {
            return Err(Error::UnknownError(response.text().await?));
        }

        let byte_stream = response.bytes_stream();
        let cancellation_token = request.cancellation_token.clone();

        let result_stream = stream! {
            let mut stream = byte_stream;
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                if let Some(token) = &cancellation_token {
                    if token.is_cancelled() {
                        break;
                    }
                }
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_str = String::from_utf8_lossy(&chunk);
                        buffer.push_str(&chunk_str);

                        // Azure Open AI may send incomplete messages, so we need to buffer the response
                        // https://learn.microsoft.com/en-us/answers/questions/1693297/how-to-fix-streaming-azure-ai-responses-from-sendi
                        // Process complete messages when we have a double newline
                        while let Some(pos) = buffer.find("\n\n") {
                            let message = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();

                            if message.starts_with("data: ") {
                                let data = &message["data: ".len()..];
                                if data == "[DONE]" {
                                    break;
                                }

                                // Parse the JSON response
                                match serde_json::from_str::<ChatCompletionChunk>(data) {
                                    Ok(v) => yield Ok(v),
                                    Err(e) => yield Err(Error::SerdeJsonError(e)),
                                }
                            }
                        }
                    },
                    Err(e) => yield Err(Error::UnknownError(format!("Failed to read response: {}", e))),
                }
            }
        };

        Ok(Box::pin(result_stream))
    }
}

#[async_trait]
impl Embeddings for Client {
    async fn create_embeddings(&self, request: &EmbeddingsRequest) -> Result<EmbeddingsResponse> {
        // Check if already cancelled before making the request
        if let Some(token) = &request.cancellation_token {
            if token.is_cancelled() {
                return Err(Error::Cancelled);
            }
        }

        let headers = self.get_headers()?;
        let url = self.get_embeddings_url(&request.model);

        // Create an abortable request
        let (abort_handle, abort_registration) = futures::future::AbortHandle::new_pair();

        // If we have a cancellation token, set up cancellation monitoring
        if let Some(token) = &request.cancellation_token {
            let token = token.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                abort_handle.abort();
            });
        }

        let request_future = self
            .http_client
            .post(url)
            .headers(headers)
            .json(request)
            .send();

        let response =
            match futures::future::Abortable::new(request_future, abort_registration).await {
                Ok(response) => response?,
                Err(futures::future::Aborted) => {
                    return Err(Error::Cancelled);
                }
            };

        if !response.status().is_success() {
            return Err(Error::UnknownError(response.text().await?));
        }

        let embeddings_response = response.json::<EmbeddingsResponse>().await?;

        Ok(embeddings_response)
    }
}

impl super::Client for Client {}
