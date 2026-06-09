use std::sync::Arc;

use async_trait::async_trait;

use crate::env_api_keys::{KnownProvider, get_env_api_key};
use crate::event_stream::AssistantEventStream;
use crate::provider::{
    ImageModelApi, LanguageModelApi, ModelBuilder, Provider, ProviderCapabilities,
};
use crate::providers::{openai_completions, openai_images, openai_responses, simple_options};
use crate::types::{
    AssistantImages, Context, ImageGenerationOptions, ImagesContext, Model, ModelInput,
    ModelOutput, SimpleStreamOptions, StreamOptions,
};
use crate::{Error, Result};

const DEFAULT_PROVIDER_ID: KnownProvider = KnownProvider::OpenAi;
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
    Images,
}

impl OpenAiApi {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Responses => "openai-responses",
            Self::ChatCompletions => "openai-completions",
            Self::Images => "openai-images",
        }
    }
}

impl OpenAi {
    pub fn builder() -> OpenAiBuilder {
        OpenAiBuilder::default()
    }

    pub fn from_env() -> Result<Self> {
        let api_key = get_env_api_key(DEFAULT_PROVIDER_ID)
            .ok_or_else(|| Error::MissingApiKey(DEFAULT_PROVIDER_ID.into()))?;
        Self::builder().api_key(Some(api_key.as_str())).build()
    }

    pub fn model(&self, id: &str) -> ModelBuilder {
        <Self as Provider>::model(self, id)
    }

    pub fn image_model(&self, id: &str) -> ModelBuilder {
        self.image_model_builder(id)
    }

    fn image_model_builder(&self, id: &str) -> ModelBuilder {
        let runtime = Arc::new(OpenAiImageModelApi {
            api_key: self.api_key.clone(),
            allow_missing_api_key: self.api_key.is_none() && self.base_url != DEFAULT_BASE_URL,
            http_client: self.http_client.clone(),
        });
        ModelBuilder::new_image(&self.provider_id, id, runtime)
            .base_url(self.base_url.clone())
            .input(vec![ModelInput::Text])
            .output(vec![ModelOutput::Image])
    }
}

impl Provider for OpenAi {
    fn id(&self) -> &str {
        &self.provider_id
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            language_models: self.api != OpenAiApi::Images,
            image_models: true,
        }
    }

    fn model(&self, id: &str) -> ModelBuilder {
        if self.api == OpenAiApi::Images {
            return self.image_model_builder(id);
        }

        let runtime = Arc::new(OpenAiLanguageModelApi {
            api: self.api,
            api_key: self.api_key.clone(),
            allow_missing_api_key: self.api_key.is_none() && self.base_url != DEFAULT_BASE_URL,
            http_client: self.http_client.clone(),
        });
        ModelBuilder::new(&self.provider_id, id, runtime)
            .base_url(self.base_url.clone())
            .input(vec![ModelInput::Text, ModelInput::Image])
            .context_window(1_000_000)
            .max_tokens(16_384)
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

    pub fn api_key(mut self, api_key: Option<&str>) -> Self {
        self.api_key = api_key
            .map(str::trim)
            .filter(|api_key| !api_key.is_empty())
            .map(str::to_string);
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

    pub fn images(mut self) -> Self {
        self.api = OpenAiApi::Images;
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
                .unwrap_or_else(|| DEFAULT_PROVIDER_ID.into()),
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
struct OpenAiImageModelApi {
    api_key: Option<String>,
    allow_missing_api_key: bool,
    http_client: Option<reqwest::Client>,
}

impl OpenAiImageModelApi {
    fn with_api_key(&self, mut options: ImageGenerationOptions) -> ImageGenerationOptions {
        if options
            .base
            .api_key
            .as_deref()
            .is_none_or(|api_key| api_key.trim().is_empty())
        {
            if let Some(api_key) = &self.api_key {
                options.base.api_key = Some(api_key.clone());
            } else if self.allow_missing_api_key {
                options.base.api_key = Some("ollama".to_string());
            }
        }
        if options.base.http_client.is_none() {
            options.base.http_client = self.http_client.clone();
        }
        options
    }
}

#[async_trait]
impl ImageModelApi for OpenAiImageModelApi {
    fn id(&self) -> &str {
        OpenAiApi::Images.id()
    }

    async fn generate_images(
        &self,
        model: Model,
        context: ImagesContext,
        options: ImageGenerationOptions,
    ) -> Result<AssistantImages> {
        Ok(openai_images::generate_images_openai(model, context, self.with_api_key(options)).await)
    }
}

#[derive(Clone)]
struct OpenAiLanguageModelApi {
    api: OpenAiApi,
    api_key: Option<String>,
    allow_missing_api_key: bool,
    http_client: Option<reqwest::Client>,
}

impl OpenAiLanguageModelApi {
    fn with_api_key(&self, mut options: StreamOptions) -> StreamOptions {
        if options
            .api_key
            .as_deref()
            .is_none_or(|api_key| api_key.trim().is_empty())
        {
            if let Some(api_key) = &self.api_key {
                options.api_key = Some(api_key.clone());
            } else if self.allow_missing_api_key {
                options.api_key = Some(String::new());
            }
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
    ) -> Result<AssistantEventStream> {
        let options = self.with_api_key(options);
        match self.api {
            OpenAiApi::ChatCompletions => Ok(openai_completions::stream_openai_completions(
                model,
                context,
                simple_options::openai_completions_options_from_stream_options(options),
            )),
            OpenAiApi::Responses => Ok(openai_responses::stream_openai_responses(
                model,
                context,
                simple_options::openai_responses_options_from_stream_options(options),
            )),
            OpenAiApi::Images => Err(Error::unsupported_capability(
                model.provider,
                "language models",
            )),
        }
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantEventStream> {
        let options = self.with_api_key_simple(options);
        match self.api {
            OpenAiApi::ChatCompletions => {
                openai_completions::stream_simple_openai_completions(model, context, options)
            }
            OpenAiApi::Responses => {
                openai_responses::stream_simple_openai_responses(model, context, options)
            }
            OpenAiApi::Images => Err(Error::unsupported_capability(
                model.provider,
                "language models",
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
    use crate::types::Context;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn model_carries_runtime_dispatch() {
        let openai = builder()
            .provider_id("test-openai-runtime")
            .build()
            .expect("provider");
        let mut model = openai.model("gpt-test").build().expect("model");
        model.api = "not-registered".to_string();

        let error = match crate::stream_simple(model, Context::default(), None) {
            Ok(_) => panic!("missing API key should fail before stream creation"),
            Err(error) => error,
        };
        assert!(
            matches!(error, crate::Error::MissingApiKey(provider) if provider == "test-openai-runtime")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn compatible_base_url_allows_no_auth_by_default() {
        let openai = builder()
            .provider_id("ollama")
            .base_url("http://127.0.0.1:9/v1")
            .chat_completions()
            .build()
            .expect("provider");
        let model = openai.model("gemma4:12b").build().expect("model");

        let stream = crate::stream_simple(model, Context::default(), None);

        assert!(stream.is_ok());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn compatible_base_url_allows_explicit_missing_api_key() {
        let openai = builder()
            .provider_id("ollama")
            .api_key(None)
            .base_url("http://127.0.0.1:9/v1")
            .chat_completions()
            .build()
            .expect("provider");
        let model = openai.model("gemma4:12b").build().expect("model");

        let stream = crate::stream_simple(model, Context::default(), None);

        assert!(stream.is_ok());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn compatible_base_url_treats_blank_api_key_as_missing() {
        let openai = builder()
            .provider_id("ollama")
            .api_key(Some("  "))
            .base_url("http://127.0.0.1:9/v1")
            .chat_completions()
            .build()
            .expect("provider");
        let model = openai.model("gemma4:12b").build().expect("model");

        let stream = crate::stream_simple(model, Context::default(), None);

        assert!(stream.is_ok());
    }
}
