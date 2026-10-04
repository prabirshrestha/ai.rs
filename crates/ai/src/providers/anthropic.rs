//! Port of `providers/anthropic.ts`, plus the pre-1.0 [`Anthropic`] handle.
//!
//! The Claude Pro/Max OAuth (`lazyOAuth(loadAnthropicOAuth)`) lands with the
//! auth/OAuth commit of the rewrite; until then the provider offers api-key
//! auth only.

use std::sync::Arc;

use async_trait::async_trait;

use super::catalog::anthropic_models;
use super::handle::{HandleAuth, HandleStreams, bind, clean_key};
use super::model_builder::ModelBuilder;
use crate::Result;
use crate::api::anthropic_messages::anthropic_messages_api;
pub use crate::api::anthropic_messages::{
    AnthropicEffort, AnthropicOptions, AnthropicThinkingDisplay, AnthropicToolChoice,
    stream_anthropic, stream_simple_anthropic,
};
use crate::auth::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthPrompt, AuthResult, ModelAuth, ProviderAuth,
    ProviderAuthInteraction, models_error, throw_if_aborted,
};
use crate::env_api_keys::{
    ANTHROPIC_API_KEY_ENV, ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_FEDERATION_RULE_ID_ENV,
    ANTHROPIC_IDENTITY_TOKEN_FILE_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_ORGANIZATION_ID_ENV,
    ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, ANTHROPIC_WORKSPACE_ID_ENV, find_env_keys,
};
use crate::models::{
    CreateModelsOptions, CreateProviderOptions, Models, Provider, ProviderApi, create_models,
    create_provider,
};
use crate::types::{AnyModel, KnownApi, Model, ModelInput, ProviderEnv, ProviderHeaders};
use crate::utils::models_error::ModelsErrorCode;

const DEFAULT_PROVIDER_ID: &str = "anthropic";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

struct AnthropicApiKeyAuth;

#[async_trait]
impl ApiKeyAuth for AnthropicApiKeyAuth {
    fn name(&self) -> &str {
        "Anthropic API key"
    }

    fn supports_login(&self) -> bool {
        true
    }

    async fn login(&self, interaction: ProviderAuthInteraction) -> Result<ApiKeyCredential> {
        interaction.throw_if_aborted()?;
        let key = interaction
            .prompt(AuthPrompt::secret("Enter Anthropic API key"))
            .await?;
        interaction.throw_if_aborted()?;
        Ok(ApiKeyCredential {
            key: Some(key),
            env: None,
        })
    }

    async fn resolve(&self, input: ApiKeyAuthInput) -> Result<Option<AuthResult>> {
        let ApiKeyAuthInput {
            ctx,
            credential,
            signal,
        } = input;
        throw_if_aborted(&signal)?;
        if let Some(credential) = &credential
            && let Some(key) = credential.key.as_ref().filter(|key| !key.is_empty())
        {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key.clone()),
                    ..Default::default()
                },
                env: credential.env.clone(),
                source: Some("stored credential".to_string()),
            }));
        }

        let auth_token = ctx.env(ANTHROPIC_AUTH_TOKEN_ENV).await;
        throw_if_aborted(&signal)?;
        if let Some(auth_token) = auth_token {
            let headers: ProviderHeaders = [("Authorization", format!("Bearer {auth_token}"))]
                .into_iter()
                .collect();
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    headers: Some(headers),
                    ..Default::default()
                },
                env: None,
                source: Some(ANTHROPIC_AUTH_TOKEN_ENV.to_string()),
            }));
        }

        for env_var in [ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV] {
            let api_key = ctx.env(env_var).await;
            throw_if_aborted(&signal)?;
            if let Some(api_key) = api_key {
                return Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(api_key),
                        ..Default::default()
                    },
                    env: None,
                    source: Some(env_var.to_string()),
                }));
            }
        }

        // Workload identity federation: the Anthropic SDK exchanges the identity
        // token for a short-lived access token and refreshes it itself. Last in
        // line so keys and ANTHROPIC_AUTH_TOKEN keep winning, as in the SDK. The
        // ids are provider config rather than auth, so they travel in `env`.
        let mut federation = ProviderEnv::new();
        for env_var in [
            ANTHROPIC_FEDERATION_RULE_ID_ENV,
            ANTHROPIC_ORGANIZATION_ID_ENV,
            ANTHROPIC_IDENTITY_TOKEN_FILE_ENV,
        ] {
            let value = ctx.env(env_var).await;
            throw_if_aborted(&signal)?;
            let Some(value) = value else {
                return Ok(None);
            };
            federation.insert(env_var.to_string(), value);
        }
        for env_var in [ANTHROPIC_SERVICE_ACCOUNT_ID_ENV, ANTHROPIC_WORKSPACE_ID_ENV] {
            let value = ctx.env(env_var).await;
            throw_if_aborted(&signal)?;
            if let Some(value) = value {
                federation.insert(env_var.to_string(), value);
            }
        }
        Ok(Some(AuthResult {
            auth: ModelAuth::default(),
            env: Some(federation),
            source: Some("workload identity federation".to_string()),
        }))
    }
}

fn anthropic_api_key_auth() -> Arc<dyn ApiKeyAuth> {
    Arc::new(AnthropicApiKeyAuth)
}

/// `anthropicProvider()`.
pub fn anthropic_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: DEFAULT_PROVIDER_ID.to_string(),
        name: Some("Anthropic".to_string()),
        base_url: Some(DEFAULT_BASE_URL.to_string()),
        auth: ProviderAuth {
            api_key: Some(anthropic_api_key_auth()),
            oauth: None,
        },
        models: anthropic_models()
            .values()
            .cloned()
            .map(AnyModel::Chat)
            .collect(),
        api: Some(ProviderApi::Single(anthropic_messages_api())),
        ..Default::default()
    })
    .expect("the Anthropic provider has an API implementation")
}

/// Pre-1.0 provider handle: `anthropic::builder().api_key(..).build()?` then
/// `handle.model("claude-sonnet-5").build()?`. Models it builds are bound to
/// the handle's own [`Models`] collection.
#[derive(Clone, Debug)]
pub struct Anthropic {
    provider_id: String,
    base_url: String,
    models: Models,
}

impl Anthropic {
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::default()
    }

    /// A handle that requires Anthropic credentials in the environment
    /// (`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_OAUTH_TOKEN` or `ANTHROPIC_API_KEY`).
    pub fn from_env() -> Result<Self> {
        if find_env_keys(DEFAULT_PROVIDER_ID, None).is_none() {
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
        let model = self
            .models
            .get_model(&self.provider_id, id)
            .unwrap_or_else(|| Model {
                id: id.to_string(),
                name: id.to_string(),
                api: KnownApi::AnthropicMessages.as_str().to_string(),
                provider: self.provider_id.clone(),
                base_url: self.base_url.clone(),
                input: vec![ModelInput::Text, ModelInput::Image],
                context_window: 1_000_000,
                max_tokens: 16_384,
                ..Default::default()
            });
        ModelBuilder::new(bind(model, &self.models))
    }
}

/// `anthropic::builder()`.
pub fn builder() -> AnthropicBuilder {
    Anthropic::builder()
}

/// `anthropic::from_env()`.
pub fn from_env() -> Result<Anthropic> {
    Anthropic::from_env()
}

#[derive(Default)]
pub struct AnthropicBuilder {
    provider_id: Option<String>,
    api_key: Option<String>,
    auth_token: Option<String>,
    base_url: Option<String>,
    http_client: Option<reqwest::Client>,
}

impl AnthropicBuilder {
    pub fn provider_id(mut self, provider_id: impl Into<String>) -> Self {
        self.provider_id = Some(provider_id.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = clean_key(Some(api_key.into()));
        self
    }

    /// Use Anthropic bearer authentication (`Authorization: Bearer`), as
    /// `ANTHROPIC_AUTH_TOKEN` does.
    pub fn auth_token(mut self, auth_token: impl Into<String>) -> Self {
        self.auth_token = clean_key(Some(auth_token.into()));
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn build(self) -> Result<Anthropic> {
        let provider_id = self
            .provider_id
            .unwrap_or_else(|| DEFAULT_PROVIDER_ID.to_string());
        let base_url = self
            .base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let headers = self.auth_token.map(|auth_token| {
            [("Authorization", format!("Bearer {auth_token}"))]
                .into_iter()
                .collect::<ProviderHeaders>()
        });
        let auth = HandleAuth {
            name: "Anthropic API key".to_string(),
            api_key: self.api_key,
            headers,
            fallback: Some(anthropic_api_key_auth()),
            keyless: false,
        };
        let models = anthropic_models()
            .values()
            .map(|model| {
                AnyModel::Chat(Model {
                    provider: provider_id.clone(),
                    base_url: base_url.clone(),
                    ..model.clone()
                })
            })
            .collect();
        let provider = create_provider(CreateProviderOptions {
            id: provider_id.clone(),
            name: Some("Anthropic".to_string()),
            base_url: Some(base_url.clone()),
            auth: ProviderAuth {
                api_key: Some(Arc::new(auth)),
                oauth: None,
            },
            models,
            api: Some(ProviderApi::Single(HandleStreams::wrap(
                anthropic_messages_api(),
                &self.http_client,
            ))),
            ..Default::default()
        })?;
        let collection = create_models(CreateModelsOptions::default());
        collection.set_provider(provider);
        Ok(Anthropic {
            provider_id,
            base_url,
            models: collection,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::auth::AuthContext;

    struct EnvContext(HashMap<&'static str, &'static str>);

    #[async_trait]
    impl AuthContext for EnvContext {
        async fn env(&self, name: &str) -> Option<String> {
            self.0.get(name).map(|value| value.to_string())
        }

        async fn file_exists(&self, _path: &str) -> bool {
            false
        }
    }

    async fn resolve(env: &[(&'static str, &'static str)]) -> Option<AuthResult> {
        AnthropicApiKeyAuth
            .resolve(ApiKeyAuthInput {
                ctx: Arc::new(EnvContext(env.iter().copied().collect())),
                credential: None,
                signal: Default::default(),
            })
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn auth_token_wins_and_is_sent_as_a_bearer_header() {
        let result = resolve(&[
            (ANTHROPIC_AUTH_TOKEN_ENV, "token"),
            (ANTHROPIC_API_KEY_ENV, "key"),
        ])
        .await
        .unwrap();
        assert_eq!(result.auth.api_key, None);
        assert_eq!(
            result.auth.headers.unwrap().get("Authorization"),
            Some(&Some("Bearer token".to_string()))
        );
        assert_eq!(result.source.as_deref(), Some(ANTHROPIC_AUTH_TOKEN_ENV));
    }

    #[tokio::test]
    async fn oauth_token_then_api_key_then_federation() {
        let result = resolve(&[
            (ANTHROPIC_OAUTH_TOKEN_ENV, "oauth"),
            (ANTHROPIC_API_KEY_ENV, "key"),
        ])
        .await
        .unwrap();
        assert_eq!(result.auth.api_key.as_deref(), Some("oauth"));

        let partial = resolve(&[(ANTHROPIC_FEDERATION_RULE_ID_ENV, "rule")]).await;
        assert_eq!(partial, None);

        let federation = resolve(&[
            (ANTHROPIC_FEDERATION_RULE_ID_ENV, "rule"),
            (ANTHROPIC_ORGANIZATION_ID_ENV, "org"),
            (ANTHROPIC_IDENTITY_TOKEN_FILE_ENV, "/token"),
            (ANTHROPIC_WORKSPACE_ID_ENV, "ws"),
        ])
        .await
        .unwrap();
        assert_eq!(federation.auth, ModelAuth::default());
        assert_eq!(federation.env.unwrap().len(), 4);
        assert_eq!(
            federation.source.as_deref(),
            Some("workload identity federation")
        );
    }

    #[test]
    fn handle_models_are_bound_and_use_the_catalog() {
        let handle = builder().api_key("key").build().unwrap();
        let model = handle.model("claude-opus-5").build().unwrap();
        assert_eq!(model.api, "anthropic-messages");
        assert!(model.reasoning);
        assert!(model.bound_models.is_some());
        assert_eq!(anthropic_provider().get_models().unwrap().len(), 16);
    }
}
