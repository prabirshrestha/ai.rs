use crate::chat_completions::{ChatCompletion, ChatCompletionRequest, ChatCompletionResponse};
use crate::Result;
use async_trait::async_trait;
use derive_builder::Builder;

#[derive(Debug, Clone, Builder)]
pub struct Client {
    http_client: reqwest::Client,
    api_key: String,
}

impl Client {
    pub fn new(api_key: String) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            api_key,
        }
    }
}

#[async_trait]
impl ChatCompletion for Client {
    async fn complete(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        todo!()
    }
}

impl super::Client for Client {}
