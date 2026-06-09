use std::sync::Arc;

use crate::env_api_keys::{KnownProvider, get_env_api_key};
use crate::event_stream::AssistantEventStream;
use crate::oauth::{GitHubCopilotOAuthProvider, OAuthApiKey, OAuthCredentials};
use crate::provider::{LanguageModelApi, ModelBuilder, Provider, ProviderCapabilities};
use crate::providers::github_copilot_headers::copilot_static_headers;
use crate::providers::{anthropic, openai_completions, openai_responses, simple_options};
use crate::types::{Context, Model, ModelInput, SimpleStreamOptions, StreamOptions};
use crate::{Error, Result};

const DEFAULT_PROVIDER_ID: KnownProvider = KnownProvider::GitHubCopilot;
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
}

impl GitHubCopilot {
    pub fn builder() -> GitHubCopilotBuilder {
        GitHubCopilotBuilder::default()
    }

    pub fn from_env() -> Result<Self> {
        let api_key = get_env_api_key(DEFAULT_PROVIDER_ID)
            .filter(|key| !key.trim().is_empty())
            .ok_or_else(|| Error::MissingApiKey(DEFAULT_PROVIDER_ID.into()))?;
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
        let api = self.api.unwrap_or_default();
        let runtime = Arc::new(GitHubCopilotLanguageModelApi {
            api,
            api_key: self.api_key.clone(),
            http_client: self.http_client.clone(),
        });
        let base_url = self
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        ModelBuilder::new(&self.provider_id, id, runtime)
            .base_url(base_url)
            .headers(copilot_static_headers())
            .input(vec![ModelInput::Text, ModelInput::Image])
            .context_window(1_000_000)
            .max_tokens(16_384)
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
                .unwrap_or_else(|| DEFAULT_PROVIDER_ID.into()),
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
    ) -> Result<AssistantEventStream> {
        let options = self.with_api_key(options);
        match self.api {
            GitHubCopilotApi::AnthropicMessages => Ok(anthropic::stream_anthropic(
                model,
                context,
                simple_options::anthropic_options_from_stream_options(options),
            )),
            GitHubCopilotApi::OpenAiChatCompletions => {
                Ok(openai_completions::stream_openai_completions(
                    model,
                    context,
                    simple_options::openai_completions_options_from_stream_options(options),
                ))
            }
            GitHubCopilotApi::OpenAiResponses => Ok(openai_responses::stream_openai_responses(
                model,
                context,
                simple_options::openai_responses_options_from_stream_options(options),
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
            GitHubCopilotApi::AnthropicMessages => {
                anthropic::stream_simple_anthropic(model, context, options)
            }
            GitHubCopilotApi::OpenAiChatCompletions => {
                openai_completions::stream_simple_openai_completions(model, context, options)
            }
            GitHubCopilotApi::OpenAiResponses => {
                openai_responses::stream_simple_openai_responses(model, context, options)
            }
        }
    }
}

pub fn builder() -> GitHubCopilotBuilder {
    GitHubCopilot::builder()
}

pub fn from_env() -> Result<GitHubCopilot> {
    GitHubCopilot::from_env()
}

pub fn oauth() -> GitHubCopilotOAuthProvider {
    crate::oauth::github_copilot_oauth_provider()
}

pub fn base_url(token: Option<&str>, enterprise_domain: Option<&str>) -> String {
    crate::oauth::get_github_copilot_base_url(token, enterprise_domain)
}

pub fn base_url_for_credentials(credentials: &OAuthCredentials) -> String {
    base_url(
        Some(&credentials.access),
        crate::oauth::github_copilot_enterprise_domain(credentials),
    )
}

pub async fn get_oauth_api_key(credentials: &OAuthCredentials) -> Result<OAuthApiKey> {
    let credentials = if crate::utils::time::now_millis() >= credentials.expires {
        oauth().refresh_token(credentials).await?
    } else {
        credentials.clone()
    };
    let api_key = oauth().get_api_key(&credentials);
    Ok(OAuthApiKey {
        new_credentials: credentials,
        api_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_uses_responses_api_without_catalog_metadata() {
        let provider = builder().api_key("test-token").build().expect("provider");
        let model = provider.model("claude-opus-4.5").build().expect("model");

        assert_eq!(model.provider_id(), "github-copilot");
        assert_eq!(model.api_id(), "openai-responses");
        assert_eq!(model.base_url, DEFAULT_BASE_URL);
        assert_eq!(
            model.headers.get("Editor-Version").map(String::as_str),
            Some("vscode/1.107.0")
        );
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
