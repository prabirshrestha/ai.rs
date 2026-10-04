//! Port of `providers/openai.ts`, plus the pre-1.0 [`OpenAi`] handle.
//!
//! The ChatGPT subscription OAuth (`lazyOAuth(loadOpenAIChatGPTOAuth)`) is not
//! ported; the provider offers api-key auth only.

use std::sync::Arc;

use indexmap::IndexMap;

use super::catalog::openai_models;
use super::handle::{HandleAuth, HandleStreams, bind, clean_key};
use super::model_builder::ModelBuilder;
use crate::api::openai_completions::openai_completions_api;
use crate::api::openai_responses::openai_responses_api;
use crate::auth::{ProviderAuth, env_api_key_auth};
use crate::env_api_keys::get_env_api_key;
use crate::models::{
    CreateModelsOptions, CreateProviderOptions, Models, Provider, ProviderApi, create_models,
    create_provider,
};
use crate::types::{AnyModel, KnownApi, Model, ModelInput};
use crate::utils::models_error::ModelsErrorCode;
use crate::{Result, auth::models_error};

const DEFAULT_PROVIDER_ID: &str = "openai";
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// `openaiProvider()`.
pub fn openai_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: DEFAULT_PROVIDER_ID.to_string(),
        name: Some("OpenAI".to_string()),
        base_url: Some(DEFAULT_BASE_URL.to_string()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("OpenAI API key", &["OPENAI_API_KEY"])),
            oauth: None,
        },
        models: openai_models()
            .values()
            .cloned()
            .map(AnyModel::Chat)
            .collect(),
        api: Some(ProviderApi::Single(openai_responses_api())),
        ..Default::default()
    })
    .expect("the OpenAI provider has an API implementation")
}

/// Which OpenAI API a handle's models use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpenAiApi {
    #[default]
    Responses,
    ChatCompletions,
}

impl OpenAiApi {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Responses => KnownApi::OpenaiResponses.as_str(),
            Self::ChatCompletions => KnownApi::OpenaiCompletions.as_str(),
        }
    }
}

/// Pre-1.0 provider handle: `openai::builder().api_key(..).build()?` then
/// `handle.model("gpt-5.5").build()?`. Models it builds are bound to the
/// handle's own [`Models`] collection, so the compat entry points use the
/// handle's key, base URL and HTTP client.
///
/// Unlike Pi's `openaiProvider()`, a handle also serves Chat Completions
/// (`OpenAiApi::ChatCompletions`) for OpenAI-compatible servers.
#[derive(Clone, Debug)]
pub struct OpenAi {
    provider_id: String,
    base_url: String,
    api: OpenAiApi,
    models: Models,
}

impl OpenAi {
    pub fn builder() -> OpenAiBuilder {
        OpenAiBuilder::default()
    }

    /// A handle that requires `OPENAI_API_KEY`.
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
    /// otherwise a default shape.
    pub fn model(&self, id: &str) -> ModelBuilder {
        let model = match self.models.get_model(&self.provider_id, id) {
            Some(mut model) => {
                model.api = self.api.id().to_string();
                model
            }
            None => Model {
                id: id.to_string(),
                name: id.to_string(),
                api: self.api.id().to_string(),
                provider: self.provider_id.clone(),
                base_url: self.base_url.clone(),
                input: vec![ModelInput::Text, ModelInput::Image],
                context_window: 1_000_000,
                max_tokens: 16_384,
                ..Default::default()
            },
        };
        ModelBuilder::new(bind(model, &self.models))
    }
}

/// `openai::builder()`.
pub fn builder() -> OpenAiBuilder {
    OpenAi::builder()
}

/// `openai::from_env()`.
pub fn from_env() -> Result<OpenAi> {
    OpenAi::from_env()
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
        self.api_key = clean_key(api_key.map(str::to_string));
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
        let provider_id = self
            .provider_id
            .unwrap_or_else(|| DEFAULT_PROVIDER_ID.to_string());
        let base_url = self
            .base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let keyless = self.api_key.is_none() && base_url != DEFAULT_BASE_URL;
        let auth = HandleAuth {
            name: "OpenAI API key".to_string(),
            api_key: self.api_key,
            headers: None,
            fallback: Some(env_api_key_auth("OpenAI API key", &["OPENAI_API_KEY"])),
            keyless,
        };
        let models = openai_models()
            .values()
            .map(|model| {
                AnyModel::Chat(Model {
                    provider: provider_id.clone(),
                    base_url: base_url.clone(),
                    ..model.clone()
                })
            })
            .collect();
        let streams: IndexMap<String, _> = [
            (
                OpenAiApi::Responses.id().to_string(),
                HandleStreams::wrap(openai_responses_api(), &self.http_client),
            ),
            (
                OpenAiApi::ChatCompletions.id().to_string(),
                HandleStreams::wrap(openai_completions_api(), &self.http_client),
            ),
        ]
        .into_iter()
        .collect();
        let provider = create_provider(CreateProviderOptions {
            id: provider_id.clone(),
            name: Some("OpenAI".to_string()),
            base_url: Some(base_url.clone()),
            auth: ProviderAuth {
                api_key: Some(Arc::new(auth)),
                oauth: None,
            },
            models,
            api: Some(ProviderApi::ByApi(streams)),
            ..Default::default()
        })?;
        let collection = create_models(CreateModelsOptions::default());
        collection.set_provider(provider);
        Ok(OpenAi {
            provider_id,
            base_url,
            api: self.api,
            models: collection,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_provider_lists_the_generated_catalog() {
        let provider = openai_provider();
        assert_eq!(provider.id(), "openai");
        assert_eq!(provider.name(), "OpenAI");
        assert_eq!(provider.base_url(), Some(DEFAULT_BASE_URL));
        let models = provider.get_models().unwrap();
        assert_eq!(models.len(), openai_models().len());
        assert!(models.iter().all(|model| model.api == "openai-responses"));
    }

    #[test]
    fn handle_builds_bound_catalog_and_custom_models() {
        let handle = builder()
            .api_key(Some(" test-key "))
            .chat_completions()
            .build()
            .unwrap();
        let model = handle.model("gpt-5.5").build().unwrap();
        assert_eq!(model.id, "gpt-5.5");
        assert_eq!(model.provider, "openai");
        assert_eq!(model.api, "openai-completions");
        assert_eq!(model.context_window, 272_000);
        assert!(
            model
                .bound_models
                .as_ref()
                .is_some_and(|models| models.ptr_eq(handle.models()))
        );

        let custom = builder()
            .provider_id("local")
            .base_url("http://localhost:11434/v1")
            .build()
            .unwrap()
            .model("llama")
            .build()
            .unwrap();
        assert_eq!(custom.provider, "local");
        assert_eq!(custom.api, "openai-responses");
        assert_eq!(custom.base_url, "http://localhost:11434/v1");
    }

    #[tokio::test]
    async fn handle_keys_win_and_custom_base_urls_allow_keyless_access() {
        let handle = builder().api_key(Some("test-key")).build().unwrap();
        let auth = handle
            .models()
            .get_auth("openai", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(auth.auth.api_key.as_deref(), Some("test-key"));

        let local = builder()
            .base_url("http://localhost:11434/v1")
            .build()
            .unwrap();
        let auth = local
            .models()
            .get_auth("openai", Default::default())
            .await
            .unwrap()
            .unwrap();
        if std::env::var("OPENAI_API_KEY").is_err() {
            assert_eq!(auth.auth.api_key, None);
            assert_eq!(auth.source.as_deref(), Some("keyless"));
        }
    }
}
