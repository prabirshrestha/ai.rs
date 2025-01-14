use crate::Result;
use async_trait::async_trait;
use derive_builder::Builder;
use dyn_clone::DynClone;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl From<(&str, &str)> for Message {
    fn from((role, content): (&str, &str)) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

impl Message {
    pub fn new<R: Into<String>, C: Into<String>>(role: R, content: C) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system<S: Into<String>>(content: S) -> Self {
        Self::new("system", content.into())
    }

    pub fn user<S: Into<String>>(content: S) -> Self {
        Self::new("user", content.into())
    }

    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self::new("assistant", content.into())
    }
}

#[derive(Debug, Serialize, Deserialize, Builder)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[builder(default = "None")]
    pub temperature: Option<f64>,
    #[builder(default = "None")]
    pub n: Option<u64>,
    #[builder(default = "false")]
    pub stream: bool,
}

#[derive(Debug, Serialize, Deserialize, Builder)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn to_iso8601_created_time(&self) -> Result<OffsetDateTime> {
        Ok(OffsetDateTime::from_unix_timestamp(self.created as i64)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[async_trait]
pub trait ChatCompletion: DynClone + Send + Sync {
    async fn complete(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse>;
}

dyn_clone::clone_trait_object!(ChatCompletion);
