use crate::chat_completions::{ChatCompletion, ChatCompletionRequest, ChatCompletionResponse};
use crate::utils::uri::ensure_no_trailing_slash;
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
        Ok(Self {
            api_key: SecretString::new(api_key.into()),
            base_url: ensure_no_trailing_slash(base_url),
            http_client: reqwest::Client::default(),
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
            return Err(Error::UnknownError(response.text().await?));
        }

        let chat_completion_response = response.json::<ChatCompletionResponse>().await?;

        Ok(chat_completion_response)
    }
}

impl super::Client for Client {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        chat_completions::{
            ChatCompletionMessage, ChatCompletionRequestBuilder, FinishReason, Role,
        },
        Result,
    };
    use httpmock::prelude::*;

    const SAMPLE_RESPONSE: &str = r#"{"id":"chatcmpl-518","object":"chat.completion","created":1736868357,"model":"llama3.2","system_fingerprint":"fp_ollama","choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France is Paris."},"finish_reason":"stop"}],"usage":{"prompt_tokens":33,"completion_tokens":8,"total_tokens":41}}"#;

    #[tokio::test]
    async fn test_sending_api_key() -> Result<()> {
        let server = MockServer::start();

        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .header("Authorization", "Bearer mock_api_key");
            then.status(200)
                .header("Content-Type", "application/json")
                .body(SAMPLE_RESPONSE);
        });

        let openai = Client::from_url("mock_api_key", &server.base_url())?;

        let request = ChatCompletionRequestBuilder::default()
            .model("llama3.2".to_string())
            .messages(vec![ChatCompletionMessage::User(
                "What is the capital of France?".into(),
            )])
            .build()?;

        let response = openai.chat_completions(&request).await?;
        mock.assert();

        assert_eq!(
            response.choices[0]
                .message
                .content
                .clone()
                .unwrap_or_default(),
            "The capital of France is Paris."
        );

        assert_eq!(response.choices[0].message.role, Role::Assistant);

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_calling() -> Result<()> {
        let server = MockServer::start();

        let mock = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("Content-Type", "application/json")
                .body(
                    r#"{
  "id": "chatcmpl-357",
  "object": "chat.completion",
  "created": 1737436413,
  "model": "llama3.2",
  "system_fingerprint": "fp_ollama",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "call_qehg6bf9",
            "index": 0,
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\":\"Seattle\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 176,
    "completion_tokens": 18,
    "total_tokens": 194
  }
}"#,
                );
        });

        let openai = Client::from_url("mock_api_key", &server.base_url())?;

        let request = ChatCompletionRequestBuilder::default()
            .model("llama3.2".to_string())
            .messages(vec![ChatCompletionMessage::User(
                "What is the weather like in Paris?".into(),
            )])
            .build()?;

        let response = openai.chat_completions(&request).await?;
        mock.assert();

        assert_eq!(response.choices[0].message.role, Role::Assistant);
        assert_eq!(response.choices[0].message.content, Some("".to_string()));
        assert_eq!(
            response.choices[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );
        assert_eq!(
            response.choices[0]
                .message
                .tool_calls
                .as_ref()
                .unwrap()
                .len(),
            1
        );

        if let Some(tool_call) = response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap()
            .get(0)
        {
            match tool_call {
                crate::chat_completions::ChatCompletionMessageToolCall::Function { function } => {
                    assert_eq!(function.name, "get_current_weather");
                    assert_eq!(function.arguments, r#"{"location":"Seattle"}"#);
                }
            }
        } else {
            unreachable!("Tool call not found");
        }

        assert_eq!(response.usage.prompt_tokens, 176);
        assert_eq!(response.usage.completion_tokens, 18);
        assert_eq!(response.usage.total_tokens, 194);

        Ok(())
    }
}
