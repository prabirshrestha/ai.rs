use std::sync::Arc;

use crate::env_api_keys::get_env_api_key;
use crate::event_stream::AssistantMessageEventStream;
use crate::provider::{LanguageModelApi, ModelBuilder, Provider, ProviderCapabilities};
use crate::providers::{anthropic, openai_completions, openai_responses, register_builtins};
use crate::types::{Context, Model, ModelInput, SimpleStreamOptions, StreamOptions};
use crate::{Error, Result};

const DEFAULT_PROVIDER_ID: &str = "github-copilot";
const DEFAULT_BASE_URL: &str = "https://api.individual.githubcopilot.com";

#[derive(Clone)]
pub struct GitHubCopilot {
    provider_id: String,
    api_key: Option<String>,
    base_url: Option<String>,
    api: Option<GitHubCopilotApi>,
    http_client: Option<reqwest::Client>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GitHubCopilotApi {
    AnthropicMessages,
    OpenAiChatCompletions,
    #[default]
    OpenAiResponses,
}

impl GitHubCopilotApi {
    pub const fn id(self) -> &'static str {
        match self {
            Self::AnthropicMessages => "anthropic-messages",
            Self::OpenAiChatCompletions => "openai-completions",
            Self::OpenAiResponses => "openai-responses",
        }
    }

    fn from_id(id: &str) -> Option<Self> {
        match id {
            "anthropic-messages" => Some(Self::AnthropicMessages),
            "openai-completions" => Some(Self::OpenAiChatCompletions),
            "openai-responses" => Some(Self::OpenAiResponses),
            _ => None,
        }
    }
}

impl GitHubCopilot {
    pub fn builder() -> GitHubCopilotBuilder {
        GitHubCopilotBuilder::default()
    }

    pub fn from_env() -> Result<Self> {
        let api_key = get_env_api_key(DEFAULT_PROVIDER_ID)
            .filter(|key| !key.trim().is_empty())
            .ok_or_else(|| Error::MissingApiKey(DEFAULT_PROVIDER_ID.to_string()))?;
        Self::builder().api_key(api_key).build()
    }

    pub fn model(&self, id: &str) -> ModelBuilder {
        <Self as Provider>::model(self, id)
    }
}

impl Provider for GitHubCopilot {
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
        let catalog_model = crate::models::get_model(&self.provider_id, id);
        let api = self
            .api
            .or_else(|| {
                catalog_model
                    .as_ref()
                    .and_then(|model| GitHubCopilotApi::from_id(&model.api))
            })
            .unwrap_or_default();
        let runtime = Arc::new(GitHubCopilotLanguageModelApi {
            api,
            api_key: self.api_key.clone(),
            http_client: self.http_client.clone(),
        });
        let base_url = self
            .base_url
            .clone()
            .or_else(|| catalog_model.as_ref().map(|model| model.base_url.clone()))
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let mut builder = ModelBuilder::new(&self.provider_id, id, runtime)
            .base_url(base_url)
            .input(vec![ModelInput::Text, ModelInput::Image]);

        if let Some(catalog_model) = catalog_model {
            builder = builder
                .name(catalog_model.name)
                .reasoning(catalog_model.reasoning)
                .input(catalog_model.input)
                .cost(catalog_model.cost)
                .context_window(catalog_model.context_window)
                .max_tokens(catalog_model.max_tokens)
                .headers(catalog_model.headers)
                .compat(catalog_model.compat);
        }

        builder
    }
}

#[derive(Default)]
pub struct GitHubCopilotBuilder {
    provider_id: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    api: Option<GitHubCopilotApi>,
    http_client: Option<reqwest::Client>,
}

impl GitHubCopilotBuilder {
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

    pub fn api(mut self, api: GitHubCopilotApi) -> Self {
        self.api = Some(api);
        self
    }

    pub fn anthropic_messages(mut self) -> Self {
        self.api = Some(GitHubCopilotApi::AnthropicMessages);
        self
    }

    pub fn chat_completions(mut self) -> Self {
        self.api = Some(GitHubCopilotApi::OpenAiChatCompletions);
        self
    }

    pub fn responses(mut self) -> Self {
        self.api = Some(GitHubCopilotApi::OpenAiResponses);
        self
    }

    pub fn http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn build(self) -> Result<GitHubCopilot> {
        Ok(GitHubCopilot {
            provider_id: self
                .provider_id
                .unwrap_or_else(|| DEFAULT_PROVIDER_ID.to_string()),
            api_key: self.api_key,
            base_url: self.base_url,
            api: self.api,
            http_client: self.http_client,
        })
    }
}

#[derive(Clone)]
struct GitHubCopilotLanguageModelApi {
    api: GitHubCopilotApi,
    api_key: Option<String>,
    http_client: Option<reqwest::Client>,
}

impl GitHubCopilotLanguageModelApi {
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

impl LanguageModelApi for GitHubCopilotLanguageModelApi {
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
            GitHubCopilotApi::AnthropicMessages => Ok(anthropic::stream_anthropic(
                model,
                context,
                register_builtins::anthropic_options_from_stream_options(options),
            )),
            GitHubCopilotApi::OpenAiChatCompletions => {
                Ok(openai_completions::stream_openai_completions(
                    model,
                    context,
                    register_builtins::openai_completions_options_from_stream_options(options),
                ))
            }
            GitHubCopilotApi::OpenAiResponses => Ok(openai_responses::stream_openai_responses(
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
            GitHubCopilotApi::AnthropicMessages => {
                Ok(anthropic::stream_simple_anthropic(model, context, options))
            }
            GitHubCopilotApi::OpenAiChatCompletions => Ok(
                openai_completions::stream_simple_openai_completions(model, context, options),
            ),
            GitHubCopilotApi::OpenAiResponses => Ok(
                openai_responses::stream_simple_openai_responses(model, context, options),
            ),
        }
    }
}

pub fn builder() -> GitHubCopilotBuilder {
    GitHubCopilot::builder()
}

pub fn from_env() -> Result<GitHubCopilot> {
    GitHubCopilot::from_env()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_model_uses_catalog_api_and_metadata() {
        let provider = builder().api_key("test-token").build().expect("provider");
        let model = provider.model("claude-opus-4.5").build().expect("model");

        assert_eq!(model.provider_id(), "github-copilot");
        assert_eq!(model.api_id(), "anthropic-messages");
        assert_eq!(model.base_url, DEFAULT_BASE_URL);
        assert_eq!(
            model
                .headers
                .get("Copilot-Integration-Id")
                .map(String::as_str),
            Some("vscode-chat")
        );
    }

    #[test]
    fn explicit_api_supports_unknown_model_ids() {
        let provider = builder()
            .api_key("test-token")
            .chat_completions()
            .base_url("https://copilot.example")
            .build()
            .expect("provider");
        let model = provider.model("future-model").build().expect("model");

        assert_eq!(model.id(), "future-model");
        assert_eq!(model.api_id(), "openai-completions");
        assert_eq!(model.base_url, "https://copilot.example");
    }
}
