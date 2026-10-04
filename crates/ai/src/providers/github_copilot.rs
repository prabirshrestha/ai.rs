//! Port of `providers/github-copilot.ts`, plus the pre-1.0 [`GitHubCopilot`]
//! handle.
//!
//! Auth: `COPILOT_GITHUB_TOKEN` api-key auth plus the GitHub Copilot OAuth
//! device flow (`lazyOAuth(loadGitHubCopilotOAuth)`). OAuth credentials set
//! the request base URL per credential (`toAuth`), and `filter_models` applies
//! the credential's `availableModelIds`.

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::Value;

use super::catalog::github_copilot_models;
use super::handle::{HandleAuth, HandleStreams, bind, clean_key};
use super::model_builder::ModelBuilder;
use crate::Result;
use crate::api::anthropic_messages::anthropic_messages_api;
use crate::api::openai_completions::openai_completions_api;
use crate::api::openai_responses::openai_responses_api;
use crate::auth::oauth::load_github_copilot_oauth;
pub use crate::auth::oauth::{
    get_github_copilot_base_url as base_url,
    github_copilot_base_url_for_credential as base_url_for_credentials,
    modify_github_copilot_models,
};
use crate::auth::{
    Credential, LazyOAuthInput, OAuthAuth, ProviderAuth, env_api_key_auth, lazy_oauth, models_error,
};
use crate::env_api_keys::get_env_api_key;
use crate::models::{
    CreateModelsOptions, CreateProviderOptions, FilterModels, Models, Provider, ProviderApi,
    create_models, create_provider,
};
use crate::types::{AnyModel, KnownApi, Model, ModelInput, ProviderStreams};
use crate::utils::models_error::ModelsErrorCode;

const DEFAULT_PROVIDER_ID: &str = "github-copilot";
const DEFAULT_BASE_URL: &str = "https://api.individual.githubcopilot.com";

fn filter_available_models(models: &[Model], credential: Option<&Credential>) -> Vec<Model> {
    let Some(Credential::OAuth(credential)) = credential else {
        return models.to_vec();
    };
    let Some(Value::Array(available_model_ids)) = credential.extra.get("availableModelIds") else {
        return models.to_vec();
    };
    let Some(available) = available_model_ids
        .iter()
        .map(Value::as_str)
        .collect::<Option<HashSet<&str>>>()
    else {
        return models.to_vec();
    };
    models
        .iter()
        .filter(|model| available.contains(model.id.as_str()))
        .cloned()
        .collect()
}

/// `lazyOAuth({ name: "GitHub Copilot", isSubscription: true, load: loadGitHubCopilotOAuth })`.
fn github_copilot_provider_oauth() -> Arc<dyn OAuthAuth> {
    lazy_oauth(LazyOAuthInput {
        name: "GitHub Copilot".to_string(),
        is_subscription: Some(true),
        login_label: None,
        load: Arc::new(|| Box::pin(load_github_copilot_oauth())),
    })
}

/// The GitHub Copilot OAuth implementation (pre-1.0 `github_copilot::oauth()`).
pub fn oauth() -> Arc<dyn OAuthAuth> {
    github_copilot_provider_oauth()
}

fn copilot_apis(
    wrap: impl Fn(Arc<dyn ProviderStreams>) -> Arc<dyn ProviderStreams>,
) -> ProviderApi {
    ProviderApi::ByApi(
        [
            (
                KnownApi::AnthropicMessages.as_str().to_string(),
                wrap(anthropic_messages_api()),
            ),
            (
                KnownApi::OpenaiCompletions.as_str().to_string(),
                wrap(openai_completions_api()),
            ),
            (
                KnownApi::OpenaiResponses.as_str().to_string(),
                wrap(openai_responses_api()),
            ),
        ]
        .into_iter()
        .collect::<IndexMap<_, _>>(),
    )
}

/// `githubCopilotProvider()`.
pub fn github_copilot_provider() -> Arc<dyn Provider> {
    github_copilot_provider_with_oauth(github_copilot_provider_oauth())
}

/// [`github_copilot_provider`] with a given OAuth implementation (tests
/// inject one with a stubbed `fetch`).
pub(crate) fn github_copilot_provider_with_oauth(oauth: Arc<dyn OAuthAuth>) -> Arc<dyn Provider> {
    let filter: FilterModels = Arc::new(filter_available_models);
    create_provider(CreateProviderOptions {
        id: DEFAULT_PROVIDER_ID.to_string(),
        name: Some("GitHub Copilot".to_string()),
        base_url: Some(DEFAULT_BASE_URL.to_string()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "GitHub Copilot token",
                &["COPILOT_GITHUB_TOKEN"],
            )),
            oauth: Some(oauth),
        },
        models: github_copilot_models()
            .values()
            .cloned()
            .map(AnyModel::Chat)
            .collect(),
        filter_models: Some(filter),
        api: Some(copilot_apis(|api| api)),
        ..Default::default()
    })
    .expect("the GitHub Copilot provider has API implementations")
}

/// Which API a handle's models use. Defaults to the catalog entry's API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GitHubCopilotApi {
    AnthropicMessages,
    OpenAiChatCompletions,
    OpenAiResponses,
}

impl GitHubCopilotApi {
    pub const fn id(self) -> &'static str {
        match self {
            Self::AnthropicMessages => KnownApi::AnthropicMessages.as_str(),
            Self::OpenAiChatCompletions => KnownApi::OpenaiCompletions.as_str(),
            Self::OpenAiResponses => KnownApi::OpenaiResponses.as_str(),
        }
    }
}

/// Copilot serves Claude 4.x and 5.x through the Anthropic Messages API, and
/// Grok, GPT-5, `oswe`, and MAI models only through `/responses`. Everything
/// else goes through Chat Completions. Used for ids missing from the catalog.
fn default_api_for_model(id: &str) -> GitHubCopilotApi {
    if is_copilot_claude(id) {
        GitHubCopilotApi::AnthropicMessages
    } else if id.starts_with("grok-")
        || id.starts_with("gpt-5")
        || id.starts_with("oswe")
        || id.starts_with("mai-")
    {
        GitHubCopilotApi::OpenAiResponses
    } else {
        GitHubCopilotApi::OpenAiChatCompletions
    }
}

/// Matches `claude-{haiku,sonnet,opus}-{4,5}` and its dotted or dashed revisions.
fn is_copilot_claude(id: &str) -> bool {
    let Some(rest) = id.strip_prefix("claude-") else {
        return false;
    };
    let Some(rest) = ["haiku-", "sonnet-", "opus-"]
        .iter()
        .find_map(|family| rest.strip_prefix(family))
    else {
        return false;
    };
    let mut chars = rest.chars();
    matches!(chars.next(), Some('4' | '5')) && matches!(chars.next(), None | Some('.' | '-'))
}

/// Pre-1.0 provider handle: `github_copilot::builder().api_key(..).build()?`
/// then `handle.model("gpt-5.5").build()?`. Models it builds are bound to the
/// handle's own [`Models`] collection.
#[derive(Clone, Debug)]
pub struct GitHubCopilot {
    provider_id: String,
    base_url: String,
    api: Option<GitHubCopilotApi>,
    models: Models,
}

impl GitHubCopilot {
    pub fn builder() -> GitHubCopilotBuilder {
        GitHubCopilotBuilder::default()
    }

    /// A handle that requires `COPILOT_GITHUB_TOKEN`.
    pub fn from_env() -> Result<Self> {
        if get_env_api_key(DEFAULT_PROVIDER_ID, None).is_none() {
            return Err(models_error(
                ModelsErrorCode::Auth,
                format!("Provider is not configured: {DEFAULT_PROVIDER_ID}"),
            ));
        }
        Self::builder().build()
    }

    pub fn id(&self) -> &str {
        &self.provider_id
    }

    /// The handle's collection, holding its one provider.
    pub fn models(&self) -> &Models {
        &self.models
    }

    /// Start building a chat model: the catalog entry when the id is known,
    /// otherwise a default shape with Copilot's static headers.
    pub fn model(&self, id: &str) -> ModelBuilder {
        let mut model = self
            .models
            .get_model(&self.provider_id, id)
            .unwrap_or_else(|| Model {
                id: id.to_string(),
                name: id.to_string(),
                api: default_api_for_model(id).id().to_string(),
                provider: self.provider_id.clone(),
                base_url: self.base_url.clone(),
                input: vec![ModelInput::Text, ModelInput::Image],
                context_window: 128_000,
                max_tokens: 16_384,
                headers: github_copilot_models()
                    .values()
                    .next()
                    .and_then(|model| model.headers.clone()),
                ..Default::default()
            });
        if let Some(api) = self.api {
            model.api = api.id().to_string();
        }
        ModelBuilder::new(bind(model, &self.models))
    }
}

/// `github_copilot::builder()`.
pub fn builder() -> GitHubCopilotBuilder {
    GitHubCopilot::builder()
}

/// `github_copilot::from_env()`.
pub fn from_env() -> Result<GitHubCopilot> {
    GitHubCopilot::from_env()
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

    /// A Copilot API token (what `COPILOT_GITHUB_TOKEN` holds).
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = clean_key(Some(api_key.into()));
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

    pub fn anthropic_messages(self) -> Self {
        self.api(GitHubCopilotApi::AnthropicMessages)
    }

    pub fn chat_completions(self) -> Self {
        self.api(GitHubCopilotApi::OpenAiChatCompletions)
    }

    pub fn responses(self) -> Self {
        self.api(GitHubCopilotApi::OpenAiResponses)
    }

    pub fn http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn build(self) -> Result<GitHubCopilot> {
        let provider_id = self
            .provider_id
            .unwrap_or_else(|| DEFAULT_PROVIDER_ID.to_string());
        let base_url = self
            .base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let auth = HandleAuth {
            name: "GitHub Copilot token".to_string(),
            api_key: self.api_key,
            headers: None,
            fallback: Some(env_api_key_auth(
                "GitHub Copilot token",
                &["COPILOT_GITHUB_TOKEN"],
            )),
            keyless: false,
        };
        let models = github_copilot_models()
            .values()
            .map(|model| {
                AnyModel::Chat(Model {
                    provider: provider_id.clone(),
                    base_url: base_url.clone(),
                    ..model.clone()
                })
            })
            .collect();
        let http_client = self.http_client;
        let filter: FilterModels = Arc::new(filter_available_models);
        let provider = create_provider(CreateProviderOptions {
            id: provider_id.clone(),
            name: Some("GitHub Copilot".to_string()),
            base_url: Some(base_url.clone()),
            auth: ProviderAuth {
                api_key: Some(Arc::new(auth)),
                oauth: Some(github_copilot_provider_oauth()),
            },
            models,
            filter_models: Some(filter),
            api: Some(copilot_apis(|api| HandleStreams::wrap(api, &http_client))),
            ..Default::default()
        })?;
        let collection = create_models(CreateModelsOptions::default());
        collection.set_provider(provider);
        Ok(GitHubCopilot {
            provider_id,
            base_url,
            api: self.api,
            models: collection,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::auth::OAuthCredential;

    #[test]
    fn filters_models_by_the_oauth_credential_model_ids() {
        let provider = github_copilot_provider();
        let models = provider.get_models().unwrap();
        assert_eq!(models.len(), 34);
        let credential = Credential::OAuth(OAuthCredential {
            extra: json!({ "availableModelIds": ["gpt-5-mini"] })
                .as_object()
                .unwrap()
                .clone(),
            ..Default::default()
        });
        let filtered = provider.filter_models(&models, Some(&credential)).unwrap();
        assert_eq!(
            filtered
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            vec!["gpt-5-mini"]
        );
        let api_key = Credential::ApiKey(Default::default());
        assert_eq!(
            provider
                .filter_models(&models, Some(&api_key))
                .unwrap()
                .len(),
            34
        );
        let invalid = Credential::OAuth(OAuthCredential {
            extra: json!({ "availableModelIds": [1] })
                .as_object()
                .unwrap()
                .clone(),
            ..Default::default()
        });
        assert_eq!(
            provider
                .filter_models(&models, Some(&invalid))
                .unwrap()
                .len(),
            34
        );
    }

    #[test]
    fn handle_uses_catalog_apis_and_falls_back_by_model_family() {
        let handle = builder().api_key("token").build().unwrap();
        assert_eq!(
            handle.model("gemini-3.5-flash").build().unwrap().api,
            "openai-completions"
        );
        let unknown = handle.model("claude-sonnet-4.9").build().unwrap();
        assert_eq!(unknown.api, "anthropic-messages");
        assert!(
            unknown
                .headers
                .unwrap()
                .contains_key("Copilot-Integration-Id")
        );
        assert_eq!(
            handle.model("gpt-5.9-new").build().unwrap().api,
            "openai-responses"
        );
        assert_eq!(
            handle.model("mystery").build().unwrap().api,
            "openai-completions"
        );
        let forced = builder()
            .api_key("token")
            .responses()
            .build()
            .unwrap()
            .model("gemini-3.5-flash")
            .build()
            .unwrap();
        assert_eq!(forced.api, "openai-responses");
    }
}
