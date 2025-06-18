use std::pin::Pin;

use crate::Result;
use async_trait::async_trait;
use derive_builder::Builder;
use dyn_clone::DynClone;
use futures::Stream;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatCompletionMessage {
    Developer(ChatCompletionDeveloperMessage),
    System(ChatCompletionSystemMessage),
    User(ChatCompletionUserMessage),
    Assistant(ChatCompletionAssistantMessage),
}

impl From<(&str, &str)> for ChatCompletionMessage {
    fn from((role, content): (&str, &str)) -> Self {
        match role {
            "system" => ChatCompletionMessage::System(content.into()),
            "user" => ChatCompletionMessage::User(content.into()),
            "assistant" => ChatCompletionMessage::Assistant(content.into()),
            _ => panic!("Invalid role"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
pub struct ChatCompletionRequestMessageContentPartText {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
pub struct ChatCompletionRequestMessageContentPartRefusal {
    pub refusal: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionDeveloperMessage {
    pub content: ChatCompletionDeveloperMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionDeveloperMessageContent {
    String(String),
    Array(Vec<ChatCompletionRequestDeveloperMessageContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ChatCompletionRequestDeveloperMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
}

impl<S: Into<String>> From<S> for ChatCompletionDeveloperMessage {
    fn from(content: S) -> Self {
        ChatCompletionDeveloperMessage {
            content: ChatCompletionDeveloperMessageContent::String(content.into()),
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionSystemMessage {
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

impl<S: Into<String>> From<S> for ChatCompletionSystemMessage {
    fn from(content: S) -> Self {
        ChatCompletionSystemMessage {
            content: ChatCompletionSystemMessageContent::String(content.into()),
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionUserMessage {
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ChatCompletionRequestUserMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
}

impl<S: Into<String>> From<S> for ChatCompletionUserMessage {
    fn from(content: S) -> Self {
        ChatCompletionUserMessage {
            content: ChatCompletionUserMessageContent::String(content.into()),
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionAssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatCompletionAssistantMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionAssistantMessageContent {
    Text(String),
    Array(Vec<ChatCompletionRequestAssistantMessageContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ChatCompletionRequestAssistantMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
    Refusal(ChatCompletionRequestMessageContentPartRefusal),
}

impl<S: Into<String>> From<S> for ChatCompletionAssistantMessage {
    fn from(content: S) -> Self {
        ChatCompletionAssistantMessage {
            content: Some(ChatCompletionAssistantMessageContent::Text(content.into())),
            refusal: None,
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ChatCompletionTool {
    #[serde(rename = "function")]
    Function {
        function: ChatCompletionToolFunctionDefinition,
    },
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, Builder, PartialEq)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
pub struct ChatCompletionToolFunctionDefinition {
    pub name: String,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl From<ChatCompletionToolFunctionDefinition> for ChatCompletionTool {
    fn from(function: ChatCompletionToolFunctionDefinition) -> Self {
        ChatCompletionTool::Function { function }
    }
}

#[derive(Debug, Serialize, Deserialize, Builder)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option))]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionRequestStreamOptions>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[builder(default = "None")]
    #[serde(skip)]
    pub cancellation_token: Option<CancellationToken>,
    #[builder(default = "None")]
    #[serde(skip)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option))]
pub struct ChatCompletionRequestStreamOptions {
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
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
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rufusal: Option<String>,
    pub role: Role,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ChatCompletionMessageToolCall {
    Function { function: FunctionCall },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionResponseMessage,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder, Default)]
#[builder(setter(into, strip_option), default)]
#[builder(pattern = "mutable")]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    pub usage: Option<Usage>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
pub struct ChatCompletionChunkChoice {
    pub delta: ChatCompletionChunkChoiceDelta,
    pub index: u32,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
pub struct ChatCompletionChunkChoiceDelta {
    pub content: Option<String>,
    pub rufusal: Option<String>,
    pub role: Option<Role>,
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
}

#[async_trait]
pub trait ChatCompletion: DynClone + Send + Sync {
    async fn chat_completions(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse>;

    async fn stream_chat_completions(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>>;
}

dyn_clone::clone_trait_object!(ChatCompletion);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_developer_message() {
        let message = ChatCompletionMessage::Developer("You are a helpful assistant.".into());
        assert_eq!(
            r#"{"role":"developer","content":"You are a helpful assistant."}"#,
            serde_json::to_string(&message).unwrap()
        );

        let message = ChatCompletionMessage::Developer(ChatCompletionDeveloperMessage {
            content: ChatCompletionDeveloperMessageContent::Array(vec![
                ChatCompletionRequestDeveloperMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: "You are a helpful assistant.".into(),
                    },
                ),
            ]),
            name: None,
        });

        assert_eq!(
            r#"{"role":"developer","content":[{"type":"text","text":"You are a helpful assistant."}]}"#,
            serde_json::to_string(&message).unwrap()
        );
    }

    #[test]
    fn test_system_message() {
        let message = ChatCompletionMessage::System("You are a helpful assistant.".into());
        assert_eq!(
            r#"{"role":"system","content":"You are a helpful assistant."}"#,
            serde_json::to_string(&message).unwrap()
        );

        let message = ChatCompletionMessage::System(ChatCompletionSystemMessage {
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
        let message = ChatCompletionMessage::User("What is the capital of France?".into());
        assert_eq!(
            r#"{"role":"user","content":"What is the capital of France?"}"#,
            serde_json::to_string(&message).unwrap()
        );

        let message = ChatCompletionMessage::User(ChatCompletionUserMessage {
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

    #[test]
    fn test_max_completion_tokens() {
        let request = ChatCompletionRequestBuilder::default()
            .model("gpt-4".to_string())
            .messages(vec![ChatCompletionMessage::User("Hello".into())])
            .max_completion_tokens(100u32)
            .build()
            .unwrap();

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("max_completion_tokens"));
        assert!(json.contains("100"));
    }
}
