use crate::Result;
use async_trait::async_trait;
use derive_builder::Builder;
use dyn_clone::DynClone;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
    Tool,
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatCompletionMessageParam {
    System(ChatCompletionSystemMessageParam),
    User(ChatCompletionUserMessageParam),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionSystemMessageParam {
    pub content: ChatCompletionSystemMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionSystemMessageContent {
    String(String),
    Array(Vec<ChatCompletionRequestSystemMessageContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ChatCompletionRequestSystemMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
pub struct ChatCompletionRequestMessageContentPartText {
    pub text: String,
}

impl<S: Into<String>> From<S> for ChatCompletionSystemMessageParam {
    fn from(content: S) -> Self {
        ChatCompletionSystemMessageParam {
            content: ChatCompletionSystemMessageContent::String(content.into()),
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionUserMessageParam {
    pub content: ChatCompletionUserMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionUserMessageContent {
    String(String),
    Array(Vec<ChatCompletionRequestUserMessageContentPart>),
}

impl<S: Into<String>> From<S> for ChatCompletionUserMessageParam {
    fn from(content: S) -> Self {
        ChatCompletionUserMessageParam {
            content: ChatCompletionUserMessageContent::String(content.into()),
            name: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ChatCompletionRequestUserMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
}

#[derive(Debug, Serialize, Deserialize, Builder)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessageParam>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Builder)]
pub struct ChatCompletionResponse {
    pub id: Option<String>,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn to_iso8601_created_time(&self) -> Result<OffsetDateTime> {
        Ok(OffsetDateTime::from_unix_timestamp(self.created as i64)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ChatCompletionResponseMessage {
    pub content: Option<String>,
    pub rufusal: Option<String>,
    pub role: Role,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionResponseMessage,
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
    async fn chat_completions(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse>;
}

dyn_clone::clone_trait_object!(ChatCompletion);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_system_message() {
        let message = ChatCompletionMessageParam::System("You are a helpful assistant.".into());
        assert_eq!(
            r#"{"role":"system","content":"You are a helpful assistant."}"#,
            serde_json::to_string(&message).unwrap()
        );

        let message = ChatCompletionMessageParam::System(ChatCompletionSystemMessageParam {
            content: ChatCompletionSystemMessageContent::Array(vec![
                ChatCompletionRequestSystemMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: "You are a helpful assistant.".into(),
                    },
                ),
            ]),
            name: None,
        });

        assert_eq!(
            r#"{"role":"system","content":[{"type":"text","text":"You are a helpful assistant."}]}"#,
            serde_json::to_string(&message).unwrap()
        );
    }

    #[test]
    fn test_user_message() {
        let message = ChatCompletionMessageParam::User("What is the capital of France?".into());
        assert_eq!(
            r#"{"role":"user","content":"What is the capital of France?"}"#,
            serde_json::to_string(&message).unwrap()
        );

        let message = ChatCompletionMessageParam::User(ChatCompletionUserMessageParam {
            content: ChatCompletionUserMessageContent::Array(vec![
                ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: "What is the capital of France?".into(),
                    },
                ),
            ]),
            name: None,
        });

        assert_eq!(
            r#"{"role":"user","content":[{"type":"text","text":"What is the capital of France?"}]}"#,
            serde_json::to_string(&message).unwrap()
        );
    }
}
