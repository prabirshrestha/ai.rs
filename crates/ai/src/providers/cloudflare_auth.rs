//! Port of `providers/cloudflare-auth.ts`: api-key auth for Cloudflare Workers
//! AI and the Cloudflare AI Gateway.
//!
//! Both read `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID` (the gateway
//! also `CLOUDFLARE_GATEWAY_ID`), per field from the stored credential first
//! and the ambient environment second, and return the account/gateway IDs as
//! provider env so the endpoint placeholders can be filled
//! (`providers::cloudflare_stream`). The AI Gateway provider itself is not
//! ported; its auth is kept with the shared helper it belongs to.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::Result;
use crate::auth::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthContext, AuthPrompt, AuthPromptKind,
    AuthResult, ModelAuth, ProviderAuthInteraction, throw_if_aborted,
};
use crate::types::{ProviderEnv, ProviderHeaders};

const CLOUDFLARE_API_KEY: &str = "CLOUDFLARE_API_KEY";
const CLOUDFLARE_ACCOUNT_ID: &str = "CLOUDFLARE_ACCOUNT_ID";
const CLOUDFLARE_GATEWAY_ID: &str = "CLOUDFLARE_GATEWAY_ID";

#[derive(Clone, Copy, PartialEq, Eq)]
enum CloudflareAuthKind {
    WorkersAi,
    AiGateway,
}

struct ResolvedCloudflareEnv {
    api_key: String,
    env: ProviderEnv,
    source: String,
}

async fn resolve_value(
    name: &str,
    ctx: &Arc<dyn AuthContext>,
    credential: Option<&ApiKeyCredential>,
    signal: &CancellationToken,
) -> Result<Option<String>> {
    // Per-field merge: prefer the credential value, fall back to ambient env.
    // A credential carrying only the API key must still pick up the account /
    // gateway id from the environment.
    let from_credential = credential.and_then(|credential| {
        if name == CLOUDFLARE_API_KEY {
            credential.key.clone()
        } else {
            credential
                .env
                .as_ref()
                .and_then(|env| env.get(name).cloned())
        }
    });
    if from_credential.is_some() {
        return Ok(from_credential);
    }
    throw_if_aborted(signal)?;
    let value = ctx.env(name).await;
    throw_if_aborted(signal)?;
    Ok(value)
}

async fn resolve_cloudflare_env(
    kind: CloudflareAuthKind,
    ctx: &Arc<dyn AuthContext>,
    credential: Option<&ApiKeyCredential>,
    signal: &CancellationToken,
) -> Result<Option<ResolvedCloudflareEnv>> {
    let api_key = resolve_value(CLOUDFLARE_API_KEY, ctx, credential, signal).await?;
    let account_id = resolve_value(CLOUDFLARE_ACCOUNT_ID, ctx, credential, signal).await?;
    let gateway_id = if kind == CloudflareAuthKind::AiGateway {
        resolve_value(CLOUDFLARE_GATEWAY_ID, ctx, credential, signal).await?
    } else {
        None
    };
    let present = |value: &Option<String>| value.as_ref().is_some_and(|value| !value.is_empty());
    if !present(&api_key)
        || !present(&account_id)
        || (kind == CloudflareAuthKind::AiGateway && !present(&gateway_id))
    {
        return Ok(None);
    }
    let mut env = ProviderEnv::new();
    env.insert(
        CLOUDFLARE_ACCOUNT_ID.to_string(),
        account_id.unwrap_or_default(),
    );
    if let Some(gateway_id) = gateway_id.filter(|value| !value.is_empty()) {
        env.insert(CLOUDFLARE_GATEWAY_ID.to_string(), gateway_id);
    }
    Ok(Some(ResolvedCloudflareEnv {
        api_key: api_key.unwrap_or_default(),
        env,
        source: if credential.is_some() {
            "stored credential".to_string()
        } else {
            CLOUDFLARE_API_KEY.to_string()
        },
    }))
}

fn text_prompt(message: &str) -> AuthPrompt {
    AuthPrompt {
        signal: None,
        kind: AuthPromptKind::Text {
            message: message.to_string(),
            placeholder: None,
        },
    }
}

/// `cloudflareWorkersAIAuth()`.
pub fn cloudflare_workers_ai_auth() -> Arc<dyn ApiKeyAuth> {
    Arc::new(CloudflareAuth {
        kind: CloudflareAuthKind::WorkersAi,
    })
}

/// `cloudflareAIGatewayAuth()`.
pub fn cloudflare_ai_gateway_auth() -> Arc<dyn ApiKeyAuth> {
    Arc::new(CloudflareAuth {
        kind: CloudflareAuthKind::AiGateway,
    })
}

struct CloudflareAuth {
    kind: CloudflareAuthKind,
}

#[async_trait]
impl ApiKeyAuth for CloudflareAuth {
    fn name(&self) -> &str {
        "Cloudflare API key"
    }

    fn supports_login(&self) -> bool {
        true
    }

    async fn login(&self, interaction: ProviderAuthInteraction) -> Result<ApiKeyCredential> {
        let key = interaction
            .prompt(AuthPrompt::secret("Enter Cloudflare API key"))
            .await?;
        let account_id = interaction
            .prompt(text_prompt("Enter Cloudflare account ID"))
            .await?;
        let mut env = ProviderEnv::new();
        env.insert(CLOUDFLARE_ACCOUNT_ID.to_string(), account_id);
        if self.kind == CloudflareAuthKind::AiGateway {
            let gateway_id = interaction
                .prompt(text_prompt("Enter Cloudflare AI Gateway ID"))
                .await?;
            env.insert(CLOUDFLARE_GATEWAY_ID.to_string(), gateway_id);
        }
        Ok(ApiKeyCredential {
            key: Some(key),
            env: Some(env),
        })
    }

    async fn resolve(&self, input: ApiKeyAuthInput) -> Result<Option<AuthResult>> {
        let Some(resolved) = resolve_cloudflare_env(
            self.kind,
            &input.ctx,
            input.credential.as_ref(),
            &input.signal,
        )
        .await?
        else {
            return Ok(None);
        };
        let auth = match self.kind {
            CloudflareAuthKind::WorkersAi => ModelAuth {
                api_key: Some(resolved.api_key),
                ..Default::default()
            },
            CloudflareAuthKind::AiGateway => {
                let mut headers = ProviderHeaders::new();
                headers.insert(
                    "cf-aig-authorization",
                    Some(format!("Bearer {}", resolved.api_key)),
                );
                headers.insert("Authorization", None);
                headers.insert("x-api-key", None);
                ModelAuth {
                    headers: Some(headers),
                    ..Default::default()
                }
            }
        };
        Ok(Some(AuthResult {
            auth,
            env: Some(resolved.env),
            source: Some(resolved.source),
        }))
    }
}
