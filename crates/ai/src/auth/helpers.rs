//! Port of `auth/helpers.ts`.
//!
//! `lazyOAuth()` is not ported: Rust links OAuth implementations statically,
//! so providers hold their `OAuthAuth` directly.

use std::sync::Arc;

use async_trait::async_trait;

use super::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthPrompt, AuthResult, ModelAuth,
    ProviderAuthInteraction, throw_if_aborted,
};
use crate::Result;

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
