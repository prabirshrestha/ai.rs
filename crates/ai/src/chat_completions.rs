use crate::Result;
use async_trait::async_trait;
use derive_builder::Builder;
use dyn_clone::DynClone;
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize)]
pub struct Messages(Vec<Message>);

impl<const N: usize> From<[(&str, &str); N]> for Messages {
    fn from(arr: [(&str, &str); N]) -> Self {
        Messages(
            arr.iter()
                .map(|&(role, content)| Message::from((role, content)))
                .collect(),
        )
    }
}

impl From<Messages> for Vec<Message> {
    fn from(messages: Messages) -> Self {
        messages.0
    }
}

#[derive(Debug, Serialize, Deserialize, Builder)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub n: Option<u64>,
    pub stream: Option<bool>,
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
