use std::pin::Pin;

use crate::chat_completions::{
    ChatCompletion, ChatCompletionChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionResponseMessage, FinishReason, Role, Usage,
};
use crate::embeddings::{Embeddings, EmbeddingsRequest, EmbeddingsResponse, EmbeddingData, EmbeddingsUsage};
use crate::utils::{
    time::deserialize_iso8601_timestamp_to_unix_timestamp, uri::ensure_no_trailing_slash,
};
use crate::{Error, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use futures::Stream;
use serde::Deserialize;
use serde_json::Value;

pub const OLLAMA_BASE_URL: &str = "http://localhost:11434";

#[derive(Debug, Clone, Builder)]
pub struct Client {
    http_client: reqwest::Client,
    base_url: String,
}

impl Client {
    pub fn new() -> Result<Self> {
        Self::from_url(OLLAMA_BASE_URL)
    }

    pub fn from_url(base_url: &str) -> Result<Self> {
        Ok(Self {
            base_url: ensure_no_trailing_slash(base_url),
            http_client: reqwest::Client::builder().build()?,
        })
    }
}

#[derive(Debug, Deserialize)]
struct OllamaChatCompletionResponse {
    #[serde(deserialize_with = "deserialize_iso8601_timestamp_to_unix_timestamp")]
    created_at: u64,
    model: String,
    message: OllamaChatCompletionResponseMessage,
    done_reason: FinishReason,
    prompt_eval_count: u32,
    eval_count: u32,
}

#[derive(Debug, Deserialize)]
struct OllamaChatCompletionResponseMessage {
    role: Role,
    content: String,
}

impl From<OllamaChatCompletionResponse> for ChatCompletionResponse {
    fn from(response: OllamaChatCompletionResponse) -> Self {
        ChatCompletionResponse {
            id: None,
            object: "".to_string(),
            created: response.created_at,
            model: response.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionResponseMessage {
                    content: Some(response.message.content),
                    rufusal: None,
                    role: response.message.role,
                    tool_calls: None,
                },
                finish_reason: Some(response.done_reason),
            }],
            usage: Usage {
                // TODO: fix usage
                prompt_tokens: response.prompt_eval_count,
                completion_tokens: (response.eval_count / 2) as u32,
                total_tokens: response.eval_count,
            },
        }
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

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        let mut request_body = serde_json::to_value(request)?;
        // Ollama defaults to streaming responses, so we need to disable it.
        request_body["stream"] = Value::from(false);

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
            .post(format!("{}/api/chat", self.base_url))
            .headers(headers)
            .json(&request_body)
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

        let chat_completion_response = response.json::<OllamaChatCompletionResponse>().await?;

        Ok(chat_completion_response.into())
    }

    async fn stream_chat_completions(
        &self,
        _request: &ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
        todo!()
    }
}

impl super::Client for Client {}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingsResponse {
    embedding: Vec<f64>,
}

#[async_trait]
impl Embeddings for Client {
    async fn create_embeddings(
        &self,
        request: &EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse> {
        // Check if already cancelled before making the request
        if let Some(token) = &request.metadata {
            if let Some(token) = token.get("cancellation_token") {
                if token.as_bool().unwrap_or(false) {
                    return Err(Error::Cancelled);
                }
            }
        }

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        let request_body = if request.input.len() == 1 {
            serde_json::json!({
                "model": request.model,
                "prompt": request.input[0],
            })
        } else {
            serde_json::json!({
                "model": request.model,
                "prompt": request.input,
            })
        };

        let response = self
            .http_client
            .post(format!("{}/api/embeddings", self.base_url))
            .headers(headers)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::UnknownError(response.text().await?));
        }

        let ollama_response = response.json::<OllamaEmbeddingsResponse>().await?;

        // Convert Ollama response to standard EmbeddingsResponse
        Ok(EmbeddingsResponse {
            object: "list".to_string(),
            data: vec![EmbeddingData {
                object: "embedding".to_string(),
                embedding: ollama_response.embedding,
                index: 0,
            }],
            model: request.model.clone(),
            usage: EmbeddingsUsage {
                prompt_tokens: 0, // Ollama doesn't provide token counts
                total_tokens: 0,
            },
        })
    }
}
