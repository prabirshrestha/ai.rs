use crate::chat_completions::{
    ChatCompletion, ChatCompletionRequest, ChatCompletionResponse, Choice, Message, Usage,
};
use crate::utils::{
    time::deserialize_iso8601_timestamp_to_unix_timestamp, uri::ensure_no_trailing_slash,
};
use crate::{Error, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use serde::Deserialize;

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
        let http_client = reqwest::Client::builder().build()?;

        Ok(Self {
            base_url: ensure_no_trailing_slash(base_url),
            http_client,
        })
    }
}

#[derive(Debug, Deserialize)]
struct OllamaChatCompletionResponse {
    #[serde(deserialize_with = "deserialize_iso8601_timestamp_to_unix_timestamp")]
    created_at: u64,
    model: String,
    message: OllamaChatCompletionResponseMessage,
    done_reason: String,
    prompt_eval_count: u32,
    eval_count: u32,
}

#[derive(Debug, Deserialize)]
struct OllamaChatCompletionResponseMessage {
    role: String,
    content: String,
}

impl From<OllamaChatCompletionResponse> for ChatCompletionResponse {
    fn from(response: OllamaChatCompletionResponse) -> Self {
        ChatCompletionResponse {
            id: None,
            object: "".to_string(),
            created: response.created_at,
            model: response.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: response.message.role,
                    content: response.message.content,
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
    async fn complete(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        let response = self
            .http_client
            .post(format!("{}/api/chat", self.base_url))
            .headers(headers)
            .json(request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::UnknownError(response.text().await?));
        }

        let chat_completion_response = response.json::<OllamaChatCompletionResponse>().await?;

        Ok(chat_completion_response.into())
    }
}

impl super::Client for Client {}
