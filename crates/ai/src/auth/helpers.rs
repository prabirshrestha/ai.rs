//! Port of `auth/helpers.ts`.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use super::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthPrompt, AuthResult, LoginOptions, ModelAuth,
    OAuthAuth, OAuthCredential, ProviderAuthInteraction, throw_if_aborted,
};
use crate::Result;
use crate::types::BoxFuture;

/// Standard api-key auth: a stored credential key wins, otherwise the first
/// set env var resolves. Includes a `login` that prompts for the key.
pub fn env_api_key_auth(name: impl Into<String>, env_vars: &[&str]) -> Arc<dyn ApiKeyAuth> {
    Arc::new(EnvApiKeyAuth {
        name: name.into(),
        env_vars: env_vars.iter().map(|name| name.to_string()).collect(),
    })
}

struct EnvApiKeyAuth {
    name: String,
    env_vars: Vec<String>,
}

#[async_trait]
impl ApiKeyAuth for EnvApiKeyAuth {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_login(&self) -> bool {
        true
    }

    async fn login(&self, interaction: ProviderAuthInteraction) -> Result<ApiKeyCredential> {
        interaction.throw_if_aborted()?;
        let key = interaction
            .prompt(AuthPrompt::secret(format!("Enter {}", self.name)))
            .await?;
        interaction.throw_if_aborted()?;
        Ok(ApiKeyCredential {
            key: Some(key),
            env: None,
        })
    }

    async fn resolve(&self, input: ApiKeyAuthInput) -> Result<Option<AuthResult>> {
        throw_if_aborted(&input.signal)?;
        if let Some(credential) = &input.credential
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
        for env_var in &self.env_vars {
            let value = input.ctx.env(env_var).await;
            throw_if_aborted(&input.signal)?;
            if let Some(value) = value.filter(|value| !value.is_empty()) {
                return Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(value),
                        ..Default::default()
                    },
                    env: None,
                    source: Some(env_var.clone()),
                }));
            }
        }
        Ok(None)
    }
}

/// Loads the wrapped `OAuthAuth` (`load` of [`lazy_oauth`]).
pub type OAuthLoader = Arc<dyn Fn() -> BoxFuture<Result<Arc<dyn OAuthAuth>>> + Send + Sync>;

/// Input of [`lazy_oauth`].
#[derive(Clone)]
pub struct LazyOAuthInput {
    pub name: String,
    pub is_subscription: Option<bool>,
    pub login_label: Option<String>,
    pub load: OAuthLoader,
}

/// Wraps a lazily loaded `OAuthAuth` so provider definitions can advertise
/// OAuth without building the implementation. The flow loads on the first
/// `login`/`refresh`/`to_auth` call. Unlike Pi's cached promise, a failed
/// load is retried on the next call.
pub fn lazy_oauth(input: LazyOAuthInput) -> Arc<dyn OAuthAuth> {
    Arc::new(LazyOAuth {
        input,
        loaded: OnceCell::new(),
    })
}

struct LazyOAuth {
    input: LazyOAuthInput,
    loaded: OnceCell<Arc<dyn OAuthAuth>>,
}

impl LazyOAuth {
    async fn loaded(&self) -> Result<&Arc<dyn OAuthAuth>> {
        self.loaded.get_or_try_init(|| (self.input.load)()).await
    }
}

#[async_trait]
impl OAuthAuth for LazyOAuth {
    fn name(&self) -> &str {
        &self.input.name
    }

    fn is_subscription(&self) -> Option<bool> {
        self.input.is_subscription
    }

    fn login_label(&self) -> Option<&str> {
        self.input.login_label.as_deref()
    }

    async fn login(
        &self,
        interaction: ProviderAuthInteraction,
        options: LoginOptions,
    ) -> Result<OAuthCredential> {
        self.loaded().await?.login(interaction, options).await
    }

    async fn refresh(
        &self,
        credential: OAuthCredential,
        signal: CancellationToken,
    ) -> Result<OAuthCredential> {
        self.loaded().await?.refresh(credential, signal).await
    }

    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth> {
        self.loaded().await?.to_auth(credential).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::auth::oauth::anthropic_oauth;

    #[tokio::test]
    async fn lazy_oauth_loads_once_on_first_use_and_keeps_its_metadata() {
        let loads = Arc::new(AtomicUsize::new(0));
        let counter = loads.clone();
        let oauth = lazy_oauth(LazyOAuthInput {
            name: "Lazy".to_string(),
            is_subscription: Some(true),
            login_label: Some("Sign in".to_string()),
            load: Arc::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Ok(anthropic_oauth()) })
            }),
        });
        assert_eq!(oauth.name(), "Lazy");
        assert_eq!(oauth.is_subscription(), Some(true));
        assert_eq!(oauth.login_label(), Some("Sign in"));
        assert_eq!(loads.load(Ordering::SeqCst), 0);
        let credential = OAuthCredential {
            access: "token".to_string(),
            ..Default::default()
        };
        for _ in 0..2 {
            let auth = oauth.to_auth(&credential).await.unwrap();
            assert_eq!(auth.api_key.as_deref(), Some("token"));
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }
}
