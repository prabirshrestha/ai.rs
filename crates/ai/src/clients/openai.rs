use std::pin::Pin;

use crate::chat_completions::{
    ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
};
use crate::embeddings::{
    Base64EmbeddingsResponse, Embeddings, EmbeddingsRequest, EmbeddingsResponse,
};
use crate::utils::uri::ensure_no_trailing_slash;
use crate::{Error, Result};
use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::{Stream, StreamExt};
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

impl Client {
    fn get_headers(&self) -> Result<reqwest::header::HeaderMap> {
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

        Ok(headers)
    }

    fn get_chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn get_embeddings_url(&self) -> String {
        format!("{}/embeddings", self.base_url)
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
            .post(self.get_chat_completions_url())
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
            .post(self.get_chat_completions_url())
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

            loop {
                if let Some(token) = &cancellation_token {
                    if token.is_cancelled() {
                        break;
                    }
                }

                match stream.next().await {
                    Some(chunk_result) => {
                        match chunk_result {
                            Ok(chunk) => {
                                let chunk_str = String::from_utf8_lossy(&chunk);

                                for line in chunk_str.lines() {
                                    if line.starts_with("data: ") {
                                        let data = &line[6..];

                                        // Check for stream end
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
                            Err(e) => {
                                yield Err(Error::UnknownError(format!("Failed to read response: {}", e)));
                                break;
                            }
                        }
                    }
                    None => break,
                }
            }
        };

        Ok(Box::pin(result_stream))
    }
}

impl super::Client for Client {}

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
            .post(self.get_embeddings_url())
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

    async fn create_base64_embeddings(
        &self,
        request: &EmbeddingsRequest,
    ) -> Result<Base64EmbeddingsResponse> {
        // Check if already cancelled before making the request
        if let Some(token) = &request.cancellation_token {
            if token.is_cancelled() {
                return Err(Error::Cancelled);
            }
        }

        let headers = self.get_headers()?;

        // Convert the request to a format with encoding_format=base64
        let request_body = serde_json::json!({
            "model": request.model,
            "input": request.input,
            "encoding_format": "base64",
            "user": request.user
        });

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
            .post(self.get_embeddings_url())
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

        let embeddings_response = response.json::<Base64EmbeddingsResponse>().await?;

        Ok(embeddings_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Result,
        chat_completions::{
            ChatCompletionMessage, ChatCompletionRequestBuilder, FinishReason, Role,
        },
    };
    use httpmock::prelude::*;

    const SAMPLE_RESPONSE: &str = r#"{"id":"chatcmpl-518","object":"chat.completion","created":1736868357,"model":"gemma3","system_fingerprint":"fp_ollama","choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France is Paris."},"finish_reason":"stop"}],"usage":{"prompt_tokens":33,"completion_tokens":8,"total_tokens":41}}"#;

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
            .model("gemma3".to_string())
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
  "model": "gemma3",
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
            .model("gemma3".to_string())
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
