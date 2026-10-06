//! Port of `auth/types.ts`.
//!
//! Pi's optional interface methods (`ApiKeyAuth.login`, `ApiKeyAuth.check`)
//! become provided trait methods plus `supports_*` probes, so `Models` can
//! still ask whether a method exists before calling it.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::types::{ProviderEnv, ProviderHeaders};
use crate::utils::models_error::{ModelsError, ModelsErrorCode};
use crate::{Error, Result};

/// Request auth for a single model request. If a value cannot be expressed as
/// `api_key`, `headers`, or `base_url`, it is provider config, not auth.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct ModelAuth {
    pub api_key: Option<String>,
    pub headers: Option<ProviderHeaders>,
    pub base_url: Option<String>,
}

impl fmt::Debug for ModelAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelAuth")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("headers", &self.headers.as_ref().map(|_| "<redacted>"))
            .field("base_url", &self.base_url)
            .finish()
    }
}

/// Stored api-key credential. `env` holds provider-scoped environment/config
/// values such as Cloudflare account/gateway ids.
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiKeyCredential {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<ProviderEnv>,
}

impl fmt::Debug for ApiKeyCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiKeyCredential")
            .field("key", &self.key.as_ref().map(|_| "<redacted>"))
            .field(
                "env",
                &self.env.as_ref().map(|env| env.keys().collect::<Vec<_>>()),
            )
            .finish()
    }
}

/// Stored canonical OAuth credential (`OAuthCredential`, which extends
/// `OAuthCredentials`). Provider-specific fields live in `extra`.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OAuthCredential {
    pub refresh: String,
    pub access: String,
    /// Expiry as Unix milliseconds.
    ///
    /// Pi's `expires` is a JS `number`. Any JSON number is accepted, so
    /// entries written by Pi (or by OAuth flows computing
    /// `Date.now() + expires_in * 1000` from fractional values, or written as
    /// `1.7e12`) load: fractions are truncated and negative values become 0,
    /// which keeps every expiry comparison unchanged. It serializes as an
    /// integer, which Pi reads as the same number.
    #[serde(deserialize_with = "deserialize_expires")]
    pub expires: u64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn deserialize_expires<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::Number(number) => Ok(number
            .as_u64()
            .unwrap_or_else(|| js_millis_to_u64(number.as_f64().unwrap_or(0.0)))),
        other => Err(serde::de::Error::invalid_type(
            serde::de::Unexpected::Other(&other.to_string()),
            &"a number",
        )),
    }
}

/// A JS millisecond `number` as `u64`: fractions are truncated, negative
/// values (and NaN) become 0, and values past `u64::MAX` saturate.
pub(crate) fn js_millis_to_u64(value: f64) -> u64 {
    // `as` saturates and maps NaN to 0.
    value as u64
}

/// OAuth token data returned by extension compatibility flows.
pub type OAuthCredentials = OAuthCredential;

impl fmt::Debug for OAuthCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OAuthCredential")
            .field("refresh", &"<redacted>")
            .field("access", &"<redacted>")
            .field("expires", &self.expires)
            .field("extra", &self.extra.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// One type-tagged credential per provider — the shape of today's auth.json.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Credential {
    #[serde(rename = "api_key")]
    ApiKey(ApiKeyCredential),
    #[serde(rename = "oauth")]
    OAuth(OAuthCredential),
}

impl Credential {
    pub fn credential_type(&self) -> AuthType {
        match self {
            Self::ApiKey(_) => AuthType::ApiKey,
            Self::OAuth(_) => AuthType::OAuth,
        }
    }

    pub fn as_api_key(&self) -> Option<&ApiKeyCredential> {
        match self {
            Self::ApiKey(credential) => Some(credential),
            Self::OAuth(_) => None,
        }
    }

    pub fn as_oauth(&self) -> Option<&OAuthCredential> {
        match self {
            Self::OAuth(credential) => Some(credential),
            Self::ApiKey(_) => None,
        }
    }
}

/// Non-secret credential metadata for account/status enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialInfo {
    pub provider_id: String,
    #[serde(rename = "type")]
    pub credential_type: AuthType,
}

/// Optional cancellation for public auth and credential operations.
#[derive(Debug, Clone, Default)]
pub struct AuthOperationOptions {
    pub signal: Option<CancellationToken>,
}

impl AuthOperationOptions {
    pub fn with_signal(signal: &CancellationToken) -> Self {
        Self {
            signal: Some(signal.clone()),
        }
    }
}

/// The write callback of [`CredentialStore::modify`]: sees the current
/// credential and returns the new one, or `None` to leave the entry unchanged.
pub type CredentialModifier = Box<
    dyn FnOnce(Option<Credential>) -> crate::types::BoxFuture<Result<Option<Credential>>> + Send,
>;

/// App-owned credential storage, keyed by `Provider.id`, one credential per
/// provider. `modify` is the only write path, so every mutation is a
/// serialized read-modify-write; `Models::get_auth()` runs OAuth refresh inside
/// `modify` so concurrent requests cannot double-refresh a rotated token. The
/// app persists a credential after login via
/// `modify(provider.id, |_| credential)`. Login/logout orchestration is
/// app-owned.
///
/// Error semantics: `read` resolves `None` for missing entries. Methods
/// reject only on storage failure; `Models` wraps such rejections in
/// `ModelsError` with code "auth".
#[async_trait]
pub trait CredentialStore: Send + Sync {
    /// Read the stored credential, possibly expired. Display/status use;
    /// resolved request auth comes from `Models::get_auth()`.
    async fn read(
        &self,
        provider_id: &str,
        options: AuthOperationOptions,
    ) -> Result<Option<Credential>>;

    /// List stored credential metadata without resolving or exposing secrets.
    /// Implementations must not execute configured API-key commands while
    /// listing.
    async fn list(&self, options: AuthOperationOptions) -> Result<Vec<CredentialInfo>>;

    /// Serialized write — the only write path. `modifier` sees the current
    /// credential; return the new credential, or `None` to leave the entry
    /// unchanged. Mutual exclusion per provider id, cross-process too where
    /// the backing store supports it (e.g. a file lock). Resolves with the
    /// post-write credential. Errors from `modifier` propagate.
    async fn modify(
        &self,
        provider_id: &str,
        modifier: CredentialModifier,
        options: AuthOperationOptions,
    ) -> Result<Option<Credential>>;

    /// Remove a credential (logout). Implementations serialize this against `modify`.
    async fn delete(&self, provider_id: &str, options: AuthOperationOptions) -> Result<()>;
}

/// Environment access for auth resolution. Injectable for tests.
#[async_trait]
pub trait AuthContext: Send + Sync {
    async fn env(&self, name: &str) -> Option<String>;
    /// Check whether a file exists. Supports a leading `~`.
    async fn file_exists(&self, path: &str) -> bool;
}

/// Result of resolving auth for a model. `Debug` shows only the names of
/// the `env` values.
#[derive(Clone, Default, PartialEq)]
pub struct AuthResult {
    pub auth: ModelAuth,
    /// Provider-scoped environment/config values resolved from credentials and ambient context.
    pub env: Option<ProviderEnv>,
    /// Human-readable label for status UI: "ANTHROPIC_API_KEY", "OAuth", "~/.aws/credentials".
    pub source: Option<String>,
}

impl fmt::Debug for AuthResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthResult")
            .field("auth", &self.auth)
            .field("env", &self.env.as_ref().map(redacted_env))
            .field("source", &self.source)
            .finish()
    }
}

/// Env values for `Debug` output: names only (sorted), values redacted.
pub(crate) fn redacted_env(env: &ProviderEnv) -> Vec<(&str, &str)> {
    let mut names: Vec<_> = env
        .keys()
        .map(|name| (name.as_str(), "<redacted>"))
        .collect();
    names.sort_unstable();
    names
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthCheck {
    pub source: Option<String>,
    pub auth_type: AuthType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuthType {
    #[serde(rename = "api_key")]
    ApiKey,
    #[serde(rename = "oauth")]
    OAuth,
}

impl AuthType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ApiKey => "api_key",
            Self::OAuth => "oauth",
        }
    }
}

impl fmt::Display for AuthType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthSelectOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthPromptKind {
    Text {
        message: String,
        placeholder: Option<String>,
    },
    Secret {
        message: String,
        placeholder: Option<String>,
    },
    Select {
        message: String,
        options: Vec<AuthSelectOption>,
    },
    ManualCode {
        message: String,
        placeholder: Option<String>,
    },
}

/// Prompt shown to the user during login. `signal` lets the flow cancel a
/// pending prompt when an out-of-band event resolves the step.
#[derive(Debug, Clone)]
pub struct AuthPrompt {
    pub signal: Option<CancellationToken>,
    pub kind: AuthPromptKind,
}

impl AuthPrompt {
    pub fn secret(message: impl Into<String>) -> Self {
        Self {
            signal: None,
            kind: AuthPromptKind::Secret {
                message: message.into(),
                placeholder: None,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthInfoLink {
    pub url: String,
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthEvent {
    Info {
        message: String,
        links: Option<Vec<AuthInfoLink>>,
    },
    AuthUrl {
        url: String,
        instructions: Option<String>,
    },
    DeviceCode {
        user_code: String,
        verification_uri: String,
        interval_seconds: Option<u64>,
        expires_in_seconds: Option<u64>,
    },
    Progress {
        message: String,
    },
}

/// Login interaction callbacks serving both api-key and OAuth flows.
///
/// `prompt()` returns the entered/selected string (`Select` returns the option
/// id) and errors on cancel/abort. `signal()` aborts the whole login flow;
/// per-prompt cancellation uses `AuthPrompt::signal`.
#[async_trait]
pub trait AuthInteraction: Send + Sync {
    fn signal(&self) -> Option<CancellationToken> {
        None
    }

    async fn prompt(&self, prompt: AuthPrompt) -> Result<String>;

    fn notify(&self, event: AuthEvent);
}

/// Normalized interaction passed to provider login implementations
/// (`AuthInteraction & { signal: AbortSignal }`).
#[derive(Clone)]
pub struct ProviderAuthInteraction {
    pub interaction: Arc<dyn AuthInteraction>,
    pub signal: CancellationToken,
}

impl ProviderAuthInteraction {
    pub async fn prompt(&self, prompt: AuthPrompt) -> Result<String> {
        self.interaction.prompt(prompt).await
    }

    pub fn notify(&self, event: AuthEvent) {
        self.interaction.notify(event);
    }

    /// `signal.throwIfAborted()`.
    pub fn throw_if_aborted(&self) -> Result<()> {
        throw_if_aborted(&self.signal)
    }
}

/// Input of [`ApiKeyAuth::check`] and [`ApiKeyAuth::resolve`].
#[derive(Clone)]
pub struct ApiKeyAuthInput {
    pub ctx: Arc<dyn AuthContext>,
    pub credential: Option<ApiKeyCredential>,
    pub signal: CancellationToken,
}

/// Api-key auth: stored key/provider env plus ambient sources (env vars, AWS
/// profiles, ADC files). Ambient-only providers do not support `login`.
#[async_trait]
pub trait ApiKeyAuth: Send + Sync {
    /// Display name, e.g. "Anthropic API key".
    fn name(&self) -> &str;

    /// Whether [`ApiKeyAuth::login`] is implemented (Pi: `login` is present).
    fn supports_login(&self) -> bool {
        false
    }

    /// Interactive setup (prompt for key/provider env).
    async fn login(&self, _interaction: ProviderAuthInteraction) -> Result<ApiKeyCredential> {
        Err(Error::message(format!(
            "{} does not support login",
            self.name()
        )))
    }

    /// Whether [`ApiKeyAuth::check`] is implemented.
    fn supports_check(&self) -> bool {
        false
    }

    /// Optional side-effect-free availability check. Use this when `resolve()`
    /// may execute commands or perform other request-time work. Missing means
    /// `Models` checks availability by resolving auth.
    async fn check(&self, _input: ApiKeyAuthInput) -> Result<Option<AuthCheck>> {
        Ok(None)
    }

    /// Resolve auth from the stored credential and/or ambient sources, merging
    /// per field. `None` = not configured.
    async fn resolve(&self, input: ApiKeyAuthInput) -> Result<Option<AuthResult>>;
}

/// App-supplied context for `Models::login`.
#[derive(Clone, Default)]
pub struct LoginOptions {
    /// Returns the stable ID of this app installation. Called only by login
    /// flows that need it.
    pub get_device_id: Option<Arc<dyn Fn() -> String + Send + Sync>>,
}

/// OAuth auth. The `refresh`/`to_auth` split lets `Models` own the locked
/// refresh pattern: `refresh` produces a credential, `to_auth` derives request
/// auth from whatever credential ends up stored.
#[async_trait]
pub trait OAuthAuth: Send + Sync {
    /// Display name, e.g. "Anthropic (Claude Pro/Max)".
    fn name(&self) -> &str;

    /// Whether access through this auth method is backed by a provider subscription.
    fn is_subscription(&self) -> Option<bool> {
        None
    }

    /// Selector label for the OAuth login option.
    fn login_label(&self) -> Option<&str> {
        None
    }

    async fn login(
        &self,
        interaction: ProviderAuthInteraction,
        options: LoginOptions,
    ) -> Result<OAuthCredential>;

    /// Exchange the refresh token. Network call; errors on failure.
    /// `Models` runs this under the store lock.
    async fn refresh(
        &self,
        credential: OAuthCredential,
        signal: CancellationToken,
    ) -> Result<OAuthCredential>;

    /// Side-effect-free derivation of request auth from a valid credential.
    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth>;
}

/// Provider auth. At least one of `api_key`/`oauth` must be present.
#[derive(Clone, Default)]
pub struct ProviderAuth {
    pub api_key: Option<Arc<dyn ApiKeyAuth>>,
    pub oauth: Option<Arc<dyn OAuthAuth>>,
}

impl fmt::Debug for ProviderAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderAuth")
            .field("api_key", &self.api_key.as_ref().map(|auth| auth.name()))
            .field("oauth", &self.oauth.as_ref().map(|auth| auth.name()))
            .finish()
    }
}

/// `signal.throwIfAborted()`.
pub fn throw_if_aborted(signal: &CancellationToken) -> Result<()> {
    if signal.is_cancelled() {
        Err(Error::aborted())
    } else {
        Ok(())
    }
}

pub(crate) fn models_error(code: ModelsErrorCode, message: impl Into<String>) -> Error {
    Error::Models(ModelsError::new(code, message))
}

pub(crate) fn models_error_with_cause(
    code: ModelsErrorCode,
    message: impl Into<String>,
    cause: &Error,
) -> Error {
    Error::Models(ModelsError::with_cause(code, message, cause))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn credentials_serialize_in_auth_json_shape() {
        let api_key = Credential::ApiKey(ApiKeyCredential {
            key: Some("secret".to_string()),
            env: None,
        });
        assert_eq!(
            serde_json::to_value(&api_key).unwrap(),
            json!({ "type": "api_key", "key": "secret" })
        );
        let oauth: Credential = serde_json::from_value(json!({
            "type": "oauth",
            "access": "a",
            "refresh": "r",
            "expires": 5,
            "availableModelIds": ["m"]
        }))
        .unwrap();
        let Credential::OAuth(credential) = &oauth else {
            panic!("expected oauth");
        };
        assert_eq!(credential.extra["availableModelIds"], json!(["m"]));
        assert_eq!(
            serde_json::to_value(&oauth).unwrap()["availableModelIds"],
            json!(["m"])
        );
        assert!(!format!("{credential:?}").contains("\"a\""));
    }

    #[test]
    fn oauth_expires_accepts_any_json_number_and_serializes_as_an_integer() {
        let expires = |value: Value| {
            let credential: Credential = serde_json::from_value(json!({
                "type": "oauth", "access": "a", "refresh": "r", "expires": value,
            }))
            .unwrap();
            let Credential::OAuth(credential) = credential else {
                panic!("expected oauth");
            };
            credential.expires
        };
        assert_eq!(expires(json!(1_730_000_000_000u64)), 1_730_000_000_000);
        assert_eq!(expires(json!(1_730_000_000_000.75)), 1_730_000_000_000);
        assert_eq!(expires(json!(1.7e12)), 1_700_000_000_000);
        assert_eq!(expires(json!(-5)), 0);
        let wrong: std::result::Result<Credential, _> = serde_json::from_value(json!({
            "type": "oauth", "access": "a", "refresh": "r", "expires": "soon",
        }));
        assert!(wrong.is_err());

        let credential = OAuthCredential {
            expires: 1_730_000_000_000,
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(&credential).unwrap()["expires"],
            json!(1_730_000_000_000u64)
        );
    }
}
