//! Port of `auth/resolve.ts`.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthContext, AuthOperationOptions, AuthResult,
    Credential, CredentialStore, OAuthAuth, OAuthCredential, ProviderAuth, models_error,
    models_error_with_cause, redacted_env, throw_if_aborted,
};
use crate::types::ProviderEnv;
use crate::utils::abort::{operation_signal, race_with_abort_signal};
use crate::utils::time::now_millis;
use crate::{Error, Result};

pub use crate::utils::models_error::{ModelsError, ModelsErrorCode};

#[derive(Clone, Default)]
pub struct AuthResolutionOverrides {
    pub api_key: Option<String>,
    pub env: Option<ProviderEnv>,
    /// Require this much remaining OAuth-token validity; defaults to five minutes.
    pub min_oauth_validity_ms: Option<u64>,
    pub signal: Option<CancellationToken>,
}

impl fmt::Debug for AuthResolutionOverrides {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthResolutionOverrides")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("env", &self.env.as_ref().map(redacted_env))
            .field("min_oauth_validity_ms", &self.min_oauth_validity_ms)
            .field("signal", &self.signal)
            .finish()
    }
}

/// Auth resolution shared by all operations in a `Models` collection.
/// A stored credential owns the provider: ambient/env is consulted only when
/// nothing is stored. No silent env fallback after a failed refresh or for a
/// credential type without a matching handler.
pub async fn resolve_provider_auth(
    provider_id: &str,
    auth: &ProviderAuth,
    credentials: &Arc<dyn CredentialStore>,
    auth_context: &Arc<dyn AuthContext>,
    overrides: AuthResolutionOverrides,
) -> Result<Option<AuthResult>> {
    let signal = operation_signal(overrides.signal.as_ref());
    race_with_abort_signal(
        resolve_provider_auth_with_signal(
            provider_id,
            auth,
            credentials,
            auth_context,
            &overrides,
            &signal,
        ),
        &signal,
    )
    .await
}

async fn resolve_provider_auth_with_signal(
    provider_id: &str,
    auth: &ProviderAuth,
    credentials: &Arc<dyn CredentialStore>,
    auth_context: &Arc<dyn AuthContext>,
    overrides: &AuthResolutionOverrides,
    signal: &CancellationToken,
) -> Result<Option<AuthResult>> {
    throw_if_aborted(signal)?;
    let request_auth_context = match &overrides.env {
        Some(env) => overlay_env_auth_context(auth_context.clone(), env.clone()),
        None => auth_context.clone(),
    };

    if let (Some(api_key), Some(api_key_auth)) = (&overrides.api_key, &auth.api_key) {
        return resolve_api_key(
            request_auth_context,
            api_key_auth,
            provider_id,
            Some(ApiKeyCredential {
                key: Some(api_key.clone()),
                env: overrides.env.clone(),
            }),
            signal,
        )
        .await;
    }

    let stored = read_credential(credentials, provider_id, signal).await?;
    if let Some(stored) = stored {
        return match (stored, &auth.oauth, &auth.api_key) {
            (Credential::OAuth(stored), Some(oauth), _) => {
                resolve_stored_oauth(
                    credentials,
                    provider_id,
                    oauth,
                    stored,
                    signal,
                    overrides.min_oauth_validity_ms,
                )
                .await
            }
            (Credential::ApiKey(stored), _, Some(api_key_auth)) => {
                let credential = match &overrides.env {
                    Some(env) => {
                        let mut merged = stored.env.clone().unwrap_or_default();
                        merged.extend(env.clone());
                        ApiKeyCredential {
                            key: stored.key.clone(),
                            env: Some(merged),
                        }
                    }
                    None => stored,
                };
                resolve_api_key(
                    request_auth_context,
                    api_key_auth,
                    provider_id,
                    Some(credential),
                    signal,
                )
                .await
            }
            _ => Ok(None),
        };
    }

    // Ambient (env vars, AWS profiles, ADC files).
    match &auth.api_key {
        Some(api_key_auth) => {
            resolve_api_key(
                request_auth_context,
                api_key_auth,
                provider_id,
                None,
                signal,
            )
            .await
        }
        None => Ok(None),
    }
}

struct OverlayEnvAuthContext {
    base: Arc<dyn AuthContext>,
    env: ProviderEnv,
}

#[async_trait]
impl AuthContext for OverlayEnvAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        match self.env.get(name).filter(|value| !value.is_empty()) {
            Some(value) => Some(value.clone()),
            None => self.base.env(name).await,
        }
    }

    async fn file_exists(&self, path: &str) -> bool {
        self.base.file_exists(path).await
    }
}

fn overlay_env_auth_context(base: Arc<dyn AuthContext>, env: ProviderEnv) -> Arc<dyn AuthContext> {
    Arc::new(OverlayEnvAuthContext { base, env })
}

const DEFAULT_OAUTH_MINIMUM_VALIDITY_MS: u64 = 5 * 60 * 1000;
const DEFAULT_OAUTH_REFRESH_TIMEOUT_MS: u64 = 15_000;

/// The message of a DOM `TimeoutError` (`AbortSignal.timeout()`).
const TIMEOUT_MESSAGE: &str = "The operation was aborted due to timeout";

/// OAuth resolution with double-checked locking: tokens with less than five
/// minutes remaining lock, re-check expiry under the lock, refresh once
/// globally, and persist the rotated credential before release.
async fn resolve_stored_oauth(
    credentials: &Arc<dyn CredentialStore>,
    provider_id: &str,
    oauth: &Arc<dyn OAuthAuth>,
    stored: OAuthCredential,
    signal: &CancellationToken,
    min_oauth_validity_ms: Option<u64>,
) -> Result<Option<AuthResult>> {
    let minimum_validity_ms =
        DEFAULT_OAUTH_MINIMUM_VALIDITY_MS.max(min_oauth_validity_ms.unwrap_or(0));
    let expires_soon = move |credential: &OAuthCredential| {
        now_millis() + minimum_validity_ms >= credential.expires
    };
    let mut credential = stored;

    if expires_soon(&credential) {
        // Optimistic check said expired; the authoritative check runs under the lock.
        let oauth_for_refresh = oauth.clone();
        let refresh_parent = signal.clone();
        let provider = provider_id.to_string();
        let post = credentials
            .modify(
                provider_id,
                Box::new(move |current| {
                    Box::pin(async move {
                        let Some(Credential::OAuth(current)) = current else {
                            return Ok(None); // logged out meanwhile
                        };
                        if !expires_soon(&current) {
                            return Ok(None); // another process/request refreshed
                        }
                        // `AbortSignal.any([signal, AbortSignal.timeout(15s)])`: the
                        // timeout only aborts the refresh signal; the refresh decides
                        // when to stop, so one that ignores its signal keeps running.
                        let refresh_signal = refresh_parent.child_token();
                        let timer = tokio::spawn({
                            let refresh_signal = refresh_signal.clone();
                            async move {
                                tokio::time::sleep(Duration::from_millis(
                                    DEFAULT_OAUTH_REFRESH_TIMEOUT_MS,
                                ))
                                .await;
                                refresh_signal.cancel();
                            }
                        });
                        let refreshed = oauth_for_refresh
                            .refresh(current, refresh_signal.clone())
                            .await;
                        timer.abort();
                        let timed_out =
                            refresh_signal.is_cancelled() && !refresh_parent.is_cancelled();
                        // An abort caused by the timeout carries the timeout reason,
                        // like a fetch aborted by `AbortSignal.timeout()`.
                        let refreshed = refreshed.map_err(|error| {
                            if timed_out && error.is_abort() {
                                Error::Aborted(TIMEOUT_MESSAGE.to_string())
                            } else {
                                error
                            }
                        });
                        match refreshed {
                            Ok(refreshed) => Ok(Some(Credential::OAuth(refreshed))),
                            Err(error) => Err(models_error_with_cause(
                                ModelsErrorCode::Oauth,
                                format!("OAuth refresh failed for {provider}"),
                                &error,
                            )),
                        }
                    })
                }),
                AuthOperationOptions::with_signal(signal),
            )
            .await;
        let post = match post {
            Ok(post) => post,
            Err(error @ Error::Models(_)) => return Err(error),
            Err(error) if error.is_abort() && signal.is_cancelled() => return Err(error),
            Err(error) => {
                return Err(models_error_with_cause(
                    ModelsErrorCode::Auth,
                    format!("Credential store modify failed for {provider_id}"),
                    &error,
                ));
            }
        };
        let Some(Credential::OAuth(post)) = post else {
            return Ok(None); // logged out meanwhile
        };
        credential = post;
        // The normal five-minute window triggers a refresh but does not impose a
        // provider contract. Explicit callers (such as bearer-token export) do
        // require the requested minimum after the refresh.
        if min_oauth_validity_ms.is_some() && expires_soon(&credential) {
            return Err(models_error(
                ModelsErrorCode::Oauth,
                format!("OAuth refresh returned a token that expires too soon for {provider_id}"),
            ));
        }
    }

    match oauth.to_auth(&credential).await {
        Ok(auth) => Ok(Some(AuthResult {
            auth,
            env: None,
            source: Some("OAuth".to_string()),
        })),
        Err(error) => Err(models_error_with_cause(
            ModelsErrorCode::Oauth,
            format!("OAuth auth derivation failed for {provider_id}"),
            &error,
        )),
    }
}

async fn resolve_api_key(
    auth_context: Arc<dyn AuthContext>,
    api_key: &Arc<dyn ApiKeyAuth>,
    provider_id: &str,
    credential: Option<ApiKeyCredential>,
    signal: &CancellationToken,
) -> Result<Option<AuthResult>> {
    api_key
        .resolve(ApiKeyAuthInput {
            ctx: auth_context,
            credential,
            signal: signal.clone(),
        })
        .await
        .map_err(|error| {
            models_error_with_cause(
                ModelsErrorCode::Auth,
                format!("API key auth failed for provider {provider_id}"),
                &error,
            )
        })
}

pub(crate) async fn read_credential(
    credentials: &Arc<dyn CredentialStore>,
    provider_id: &str,
    signal: &CancellationToken,
) -> Result<Option<Credential>> {
    credentials
        .read(provider_id, AuthOperationOptions::with_signal(signal))
        .await
        .map_err(|error| {
            models_error_with_cause(
                ModelsErrorCode::Auth,
                format!("Credential store read failed for {provider_id}"),
                &error,
            )
        })
}
