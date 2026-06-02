use std::sync::Arc;

use crate::event_stream::AssistantMessageEventStream;
use crate::provider::{LanguageModelApi, ModelBuilder, Provider, ProviderCapabilities};
use crate::providers::{openai_completions, openai_responses, register_builtins};
use crate::types::{Context, Model, ModelInput, SimpleStreamOptions, StreamOptions};
use crate::{Error, Result};

const DEFAULT_PROVIDER_ID: &str = "openai";
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Clone)]
pub struct OpenAi {
    provider_id: String,
    api_key: Option<String>,
    base_url: String,
    api: OpenAiApi,
    http_client: Option<reqwest::Client>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpenAiApi {
    #[default]
    Responses,
    ChatCompletions,
}

impl OpenAiApi {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Responses => "openai-responses",
            Self::ChatCompletions => "openai-completions",
        }
    }
}

impl OpenAi {
    pub fn builder() -> OpenAiBuilder {
        OpenAiBuilder::default()
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|key| !key.trim().is_empty())
            .ok_or_else(|| Error::MissingApiKey(DEFAULT_PROVIDER_ID.to_string()))?;
        Self::builder().api_key(api_key).build()
    }

    pub fn model(&self, id: &str) -> ModelBuilder {
        <Self as Provider>::model(self, id)
    }
}

impl Provider for OpenAi {
    fn id(&self) -> &str {
        &self.provider_id
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            language_models: true,
            image_models: false,
        }
    }

    fn model(&self, id: &str) -> ModelBuilder {
        let runtime = Arc::new(OpenAiLanguageModelApi {
            api: self.api,
            api_key: self.api_key.clone(),
            http_client: self.http_client.clone(),
        });
        let mut builder = ModelBuilder::new(&self.provider_id, id, runtime)
            .base_url(self.base_url.clone())
            .input(vec![ModelInput::Text, ModelInput::Image]);

        if let Some(catalog_model) = crate::models::get_model(&self.provider_id, id) {
            builder = builder
                .name(catalog_model.name)
                .reasoning(catalog_model.reasoning)
                .input(catalog_model.input)
                .cost(catalog_model.cost)
                .context_window(catalog_model.context_window)
                .max_tokens(catalog_model.max_tokens)
                .compat(catalog_model.compat);
        }

        builder
    }
}

#[derive(Default)]
pub struct OpenAiBuilder {
    provider_id: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    api: OpenAiApi,
    http_client: Option<reqwest::Client>,
}

impl OpenAiBuilder {
    pub fn provider_id(mut self, provider_id: impl Into<String>) -> Self {
        self.provider_id = Some(provider_id.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn api(mut self, api: OpenAiApi) -> Self {
        self.api = api;
        self
    }

    pub fn responses(mut self) -> Self {
        self.api = OpenAiApi::Responses;
        self
    }

    pub fn chat_completions(mut self) -> Self {
        self.api = OpenAiApi::ChatCompletions;
        self
    }

    pub fn http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn build(self) -> Result<OpenAi> {
        Ok(OpenAi {
            provider_id: self
                .provider_id
                .unwrap_or_else(|| DEFAULT_PROVIDER_ID.to_string()),
            api_key: self.api_key,
            base_url: self
                .base_url
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            api: self.api,
            http_client: self.http_client,
        })
    }
}

#[derive(Clone)]
struct OpenAiLanguageModelApi {
    api: OpenAiApi,
    api_key: Option<String>,
    http_client: Option<reqwest::Client>,
}

impl OpenAiLanguageModelApi {
    fn with_api_key(&self, mut options: StreamOptions) -> StreamOptions {
        if options
            .api_key
            .as_deref()
            .is_none_or(|api_key| api_key.trim().is_empty())
        {
            options.api_key = self.api_key.clone();
        }
        if options.http_client.is_none() {
            options.http_client = self.http_client.clone();
        }
        options
    }

    fn with_api_key_simple(&self, mut options: SimpleStreamOptions) -> SimpleStreamOptions {
        options.stream = self.with_api_key(options.stream);
        options
    }
}

impl LanguageModelApi for OpenAiLanguageModelApi {
    fn id(&self) -> &str {
        self.api.id()
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: StreamOptions,
    ) -> Result<AssistantMessageEventStream> {
        let options = self.with_api_key(options);
        match self.api {
            OpenAiApi::ChatCompletions => Ok(openai_completions::stream_openai_completions(
                model,
                context,
                register_builtins::openai_completions_options_from_stream_options(options),
            )),
            OpenAiApi::Responses => Ok(openai_responses::stream_openai_responses(
                model,
                context,
                register_builtins::openai_responses_options_from_stream_options(options),
            )),
        }
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream> {
        let options = self.with_api_key_simple(options);
        match self.api {
            OpenAiApi::ChatCompletions => Ok(openai_completions::stream_simple_openai_completions(
                model, context, options,
            )),
            OpenAiApi::Responses => Ok(openai_responses::stream_simple_openai_responses(
                model, context, options,
            )),
        }
    }
}

pub fn builder() -> OpenAiBuilder {
    OpenAi::builder()
}

pub fn from_env() -> Result<OpenAi> {
    OpenAi::from_env()
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use crate::types::{AssistantMessageEvent, Context, StopReason};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn model_carries_runtime_dispatch() {
        let openai = builder()
            .provider_id("test-openai-runtime")
            .build()
            .expect("provider");
        let mut model = openai.model("gpt-test").build().expect("model");
        model.api = "not-registered".to_string();

        let mut stream =
            crate::stream_simple(model, Context::default(), None).expect("runtime stream");
        let event = stream.next().await.expect("error event");
        let AssistantMessageEvent::Error { reason, error } = event else {
            panic!("expected error event");
        };

        assert_eq!(reason, StopReason::Error);
        assert_eq!(
            error.error_message.as_deref(),
            Some("No API key for provider: test-openai-runtime")
        );
    }
}
