//! Port of libfx's `subscription.js`, `chatgpt.js` and `grok.js`: adds a
//! Codex (ChatGPT plan) or Grok (Grok plan) subscription's credentials to
//! the provider's own requests at the fetch edge, refreshing them near
//! expiry and once after a 401, and masks them in failed responses' bodies.
//!
//! [rust] Not ported: `refreshLock` and the credential memory engines of one
//! agent share; each authorizer keeps its own.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::fetch::{FxFetch, FxFetchError, FxHttpRequest, FxHttpResponse, header};
use crate::harness::utils::native_subscription::jwt::parse_jwt_payload;

const REFRESH_SKEW_MS: f64 = 5.0 * 60.0 * 1000.0;
const MAX_TOKEN_BYTES: usize = 64 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

pub const CHATGPT_ISSUER: &str = "https://auth.openai.com";
pub const CHATGPT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const GROK_ISSUER: &str = "https://auth.x.ai";
pub const GROK_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const CODEX_ORIGIN: &str = "https://chatgpt.com";
const GROK_PROXY_ORIGIN: &str = "https://cli-chat-proxy.grok.com";
const GROK_LANGUAGE_MODELS: &str = "https://api.x.ai/v1/language-models";
const OPENAI_AUTH_CLAIM: &str = "https://api.openai.com/auth";

/// A subscription provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FxSubscriptionProvider {
    Codex,
    Grok,
}

impl FxSubscriptionProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::Grok => "grok",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Codex => "ChatGPT",
            Self::Grok => "Grok",
        }
    }
}

/// `FxProviderCredentials`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FxProviderCredentials {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    /// Milliseconds since the epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<f64>,
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
type CredentialResult<T> = Result<T, String>;

/// The host's `credentials()` and `onCredentials()`.
pub trait FxCredentialSource: Send + Sync {
    fn read(&self) -> BoxFuture<'_, CredentialResult<FxProviderCredentials>>;
    fn write(&self, credentials: FxProviderCredentials) -> BoxFuture<'_, CredentialResult<()>>;
}

fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

fn token_string(
    value: Option<&str>,
    name: &str,
    required: bool,
    label: &str,
) -> CredentialResult<Option<String>> {
    match value {
        None if !required => Ok(None),
        Some(value) if !value.is_empty() => {
            if value.len() > MAX_TOKEN_BYTES {
                return Err(format!(
                    "{label} credentials {name} exceeds {MAX_TOKEN_BYTES} bytes"
                ));
            }
            Ok(Some(value.to_string()))
        }
        _ => Err(format!(
            "{label} credentials {name} must be a non-empty string"
        )),
    }
}

/// `normalizeCredentials`: the account and expiry default to the access
/// token's claims.
pub(crate) fn normalize_credentials(
    provider: FxSubscriptionProvider,
    value: &FxProviderCredentials,
) -> CredentialResult<FxProviderCredentials> {
    let label = provider.label();
    let access_token =
        token_string(Some(&value.access_token), "accessToken", true, label)?.unwrap();
    let refresh_token = token_string(value.refresh_token.as_deref(), "refreshToken", false, label)?;
    let claims = parse_jwt_payload(&access_token);
    let account_id = match &value.account_id {
        Some(account) => token_string(Some(account), "accountId", true, label)?,
        None => claims.as_ref().and_then(|claims| match provider {
            FxSubscriptionProvider::Codex => claims
                .get(OPENAI_AUTH_CLAIM)?
                .get("chatgpt_account_id")?
                .as_str()
                .map(str::to_string),
            FxSubscriptionProvider::Grok => claims.get("sub")?.as_str().map(str::to_string),
        }),
    };
    let Some(account_id) = account_id.filter(|account| !account.is_empty()) else {
        return Err(format!(
            "{label} credentials need an accountId; it was not in the access token"
        ));
    };
    let mut expires_at = value.expires_at;
    if expires_at.is_none() {
        expires_at = claims
            .as_ref()
            .and_then(|claims| claims.get("exp")?.as_f64())
            .filter(|exp| exp.is_finite())
            .map(|exp| exp * 1000.0);
    }
    if expires_at.is_some_and(|at| !at.is_finite()) {
        return Err(format!(
            "{label} credentials expiresAt must be milliseconds since the epoch"
        ));
    }
    Ok(FxProviderCredentials {
        access_token,
        refresh_token,
        account_id: Some(account_id),
        expires_at,
    })
}

/// `needsRefresh`.
pub(crate) fn needs_refresh(credentials: &FxProviderCredentials, now: f64) -> bool {
    credentials.refresh_token.is_some()
        && credentials
            .expires_at
            .is_some_and(|at| at - REFRESH_SKEW_MS <= now)
}

/// `origin + pathname` of a URL, or None.
pub(crate) fn endpoint_of(url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(url).ok()?;
    Some(format!(
        "{}{}",
        parsed.origin().ascii_serialization(),
        parsed.path()
    ))
}

/// `isCodexRequest` / `isGrokRequest`: the provider that authorizes
/// requests to `endpoint`.
pub(crate) fn matches(provider: FxSubscriptionProvider, endpoint: &str) -> bool {
    match provider {
        FxSubscriptionProvider::Codex => {
            endpoint == format!("{CODEX_ORIGIN}/backend-api/codex/responses")
                || endpoint == format!("{CODEX_ORIGIN}/backend-api/codex/models")
        }
        FxSubscriptionProvider::Grok => {
            endpoint == format!("{GROK_PROXY_ORIGIN}/v1/responses")
                || endpoint == format!("{GROK_PROXY_ORIGIN}/v1/models")
                || endpoint == GROK_LANGUAGE_MODELS
        }
    }
}

fn authorize(
    provider: FxSubscriptionProvider,
    request: &mut FxHttpRequest,
    endpoint: &str,
    credentials: &FxProviderCredentials,
) {
    request.set_header(
        "authorization",
        format!("Bearer {}", credentials.access_token),
    );
    let account = credentials.account_id.clone().unwrap_or_default();
    match provider {
        FxSubscriptionProvider::Codex => request.set_header("chatgpt-account-id", account),
        FxSubscriptionProvider::Grok => {
            if endpoint == format!("{GROK_PROXY_ORIGIN}/v1/responses") {
                request.set_header("x-grok-user-id", account);
            } else if endpoint == format!("{GROK_PROXY_ORIGIN}/v1/models") {
                request.set_header("x-userid", account);
            }
        }
    }
}

fn refresh_error(message: String) -> String {
    message
}

async fn json_body(response: FxHttpResponse) -> Option<Value> {
    serde_json::from_slice(&response.bytes().await.ok()?).ok()
}

fn expires_in(tokens: &Value) -> Option<f64> {
    tokens
        .get("expires_in")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .map(|seconds| now_ms() + seconds * 1000.0)
}

/// `refreshChatgptCredentials`.
async fn refresh_chatgpt(
    credentials: &FxProviderCredentials,
    fetch: &FxFetch,
    signal: CancellationToken,
) -> CredentialResult<FxProviderCredentials> {
    let Some(refresh_token) = &credentials.refresh_token else {
        return Err("ChatGPT credentials have no refreshToken".into());
    };
    let request = FxHttpRequest {
        url: format!("{CHATGPT_ISSUER}/oauth/token"),
        method: "POST".into(),
        headers: vec![("content-type".into(), "application/json".into())],
        body: Some(
            json!({
                "client_id": CHATGPT_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            })
            .to_string()
            .into(),
        ),
    };
    let response = fetch(request, signal)
        .await
        .map_err(|error| error.to_string())?;
    if !response.ok() {
        return Err(refresh_error(format!(
            "ChatGPT token refresh failed with HTTP {}",
            response.status
        )));
    }
    let tokens = json_body(response).await.unwrap_or(Value::Null);
    let next = normalize_credentials(
        FxSubscriptionProvider::Codex,
        &FxProviderCredentials {
            access_token: tokens["access_token"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            refresh_token: tokens["refresh_token"]
                .as_str()
                .map(str::to_string)
                .or_else(|| credentials.refresh_token.clone()),
            account_id: None,
            expires_at: expires_in(&tokens),
        },
    )?;
    Ok(if next.account_id == credentials.account_id {
        next
    } else {
        FxProviderCredentials {
            account_id: credentials.account_id.clone(),
            ..next
        }
    })
}

/// `grokAccountId`.
async fn grok_account_id(
    access_token: &str,
    fetch: &FxFetch,
    signal: CancellationToken,
) -> CredentialResult<String> {
    let request = FxHttpRequest {
        url: format!("{GROK_ISSUER}/oauth2/userinfo"),
        method: "GET".into(),
        headers: vec![
            ("authorization".into(), format!("Bearer {access_token}")),
            ("accept".into(), "application/json".into()),
        ],
        body: None,
    };
    let response = fetch(request, signal)
        .await
        .map_err(|error| error.to_string())?;
    if !response.ok() {
        return Err(format!(
            "xAI did not identify the account (HTTP {})",
            response.status
        ));
    }
    match json_body(response)
        .await
        .as_ref()
        .and_then(|info| info.get("sub")?.as_str())
    {
        Some(sub) if !sub.is_empty() => Ok(sub.to_string()),
        _ => Err("xAI's account response had no account id".into()),
    }
}

/// `URLSearchParams.toString()`.
fn form_encode(pairs: &[(&str, &str)]) -> String {
    let encode = |value: &str| {
        value
            .bytes()
            .map(|byte| match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'*' | b'-' | b'.' | b'_' => {
                    (byte as char).to_string()
                }
                b' ' => "+".to_string(),
                _ => format!("%{byte:02X}"),
            })
            .collect::<String>()
    };
    pairs
        .iter()
        .map(|(name, value)| format!("{}={}", encode(name), encode(value)))
        .collect::<Vec<_>>()
        .join("&")
}

/// `refreshGrokCredentials`.
async fn refresh_grok(
    credentials: &FxProviderCredentials,
    fetch: &FxFetch,
    signal: CancellationToken,
) -> CredentialResult<FxProviderCredentials> {
    let Some(refresh_token) = &credentials.refresh_token else {
        return Err("Grok credentials have no refreshToken".into());
    };
    let body = form_encode(&[
        ("grant_type", "refresh_token"),
        ("client_id", GROK_CLIENT_ID),
        ("refresh_token", refresh_token),
    ]);
    let request = FxHttpRequest {
        url: format!("{GROK_ISSUER}/oauth2/token"),
        method: "POST".into(),
        headers: vec![
            (
                "content-type".into(),
                "application/x-www-form-urlencoded".into(),
            ),
            ("accept".into(), "application/json".into()),
        ],
        body: Some(body.into()),
    };
    let response = fetch(request, signal.clone())
        .await
        .map_err(|error| error.to_string())?;
    if !response.ok() {
        return Err(format!(
            "Grok token refresh failed with HTTP {}",
            response.status
        ));
    }
    let tokens = json_body(response).await.unwrap_or(Value::Null);
    let next = normalize_credentials(
        FxSubscriptionProvider::Grok,
        &FxProviderCredentials {
            access_token: tokens["access_token"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            refresh_token: tokens["refresh_token"]
                .as_str()
                .map(str::to_string)
                .or_else(|| credentials.refresh_token.clone()),
            account_id: credentials.account_id.clone(),
            expires_at: expires_in(&tokens),
        },
    )?;
    if Some(grok_account_id(&next.access_token, fetch, signal).await?) != credentials.account_id {
        return Err("The refreshed Grok token belongs to a different account".into());
    }
    Ok(next)
}

/// `withoutCredentials`: a failed response whose body cannot repeat the
/// credentials its request carried.
async fn without_credentials(
    response: FxHttpResponse,
    credentials: &FxProviderCredentials,
) -> FxHttpResponse {
    if response.status < 400 {
        return response;
    }
    let status = response.status;
    let mut headers = response.headers.clone();
    headers.retain(|(name, _)| {
        !name.eq_ignore_ascii_case("content-length")
            && !name.eq_ignore_ascii_case("content-encoding")
    });
    let raw = bounded_text(response, MAX_ERROR_BODY_BYTES).await;
    let mut body = raw
        .clone()
        .unwrap_or_else(|| "Provider error details exceeded the local limit".into());
    if let Some(raw) = &raw {
        let trimmed = raw.trim_start();
        if trimmed.starts_with(['[', '{', '"']) {
            body = match serde_json::from_str::<Value>(raw) {
                Ok(value) => value.to_string(),
                Err(_) => "Provider error details could not be decoded".into(),
            };
        }
    }
    for secret in [
        Some(&credentials.access_token),
        credentials.refresh_token.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        let escaped = Value::String(secret.clone()).to_string();
        let escaped = escaped[1..escaped.len() - 1].to_string();
        for form in [secret.clone(), escaped] {
            if !form.is_empty() {
                body = body.replace(&form, &"*".repeat(form.chars().count()));
            }
        }
    }
    let mut masked = FxHttpResponse::new(status, body);
    masked.headers = headers;
    masked
}

async fn bounded_text(response: FxHttpResponse, limit: usize) -> Option<String> {
    use futures::StreamExt as _;
    let mut body = response.body;
    let mut bytes = Vec::new();
    while let Some(chunk) = body.next().await {
        let chunk = chunk.ok()?;
        bytes.extend_from_slice(&chunk);
        if bytes.len() > limit {
            return None;
        }
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// `createSubscriptionAuthorizer` for one provider.
pub struct SubscriptionAuthorizer {
    provider: FxSubscriptionProvider,
    source: Arc<dyn FxCredentialSource>,
    fetch: FxFetch,
    /// Supplied access token → the newest credentials refreshed from it.
    descendants: Mutex<HashMap<String, FxProviderCredentials>>,
    /// Serializes refreshes, so one refresh token is spent once.
    refreshing: Mutex<()>,
    now: fn() -> f64,
}

impl SubscriptionAuthorizer {
    pub fn new(
        provider: FxSubscriptionProvider,
        source: Arc<dyn FxCredentialSource>,
        fetch: FxFetch,
    ) -> Self {
        Self {
            provider,
            source,
            fetch,
            descendants: Mutex::default(),
            refreshing: Mutex::default(),
            now: now_ms,
        }
    }

    pub fn provider(&self) -> FxSubscriptionProvider {
        self.provider
    }

    /// `specFor(endpoint)`.
    pub fn authorizes(&self, endpoint: &str) -> bool {
        matches(self.provider, endpoint)
    }

    async fn supplied(&self) -> CredentialResult<FxProviderCredentials> {
        normalize_credentials(self.provider, &self.source.read().await?)
    }

    async fn refresh(
        &self,
        supplied: &FxProviderCredentials,
        current: &FxProviderCredentials,
        signal: CancellationToken,
    ) -> CredentialResult<FxProviderCredentials> {
        let _guard = self.refreshing.lock().await;
        // Another request may have refreshed them while this one waited.
        if let Some(latest) = self.descendants.lock().await.get(&supplied.access_token)
            && latest.access_token != current.access_token
        {
            return Ok(latest.clone());
        }
        let next = match self.provider {
            FxSubscriptionProvider::Codex => refresh_chatgpt(current, &self.fetch, signal).await?,
            FxSubscriptionProvider::Grok => refresh_grok(current, &self.fetch, signal).await?,
        };
        self.descendants
            .lock()
            .await
            .insert(supplied.access_token.clone(), next.clone());
        self.source.write(next.clone()).await?;
        Ok(next)
    }

    /// `current()`: the supplied credentials and the ones to use.
    async fn current(
        &self,
        signal: CancellationToken,
    ) -> CredentialResult<(FxProviderCredentials, FxProviderCredentials)> {
        let supplied = self.supplied().await?;
        let best = self
            .descendants
            .lock()
            .await
            .get(&supplied.access_token)
            .cloned()
            .unwrap_or_else(|| supplied.clone());
        let used = if needs_refresh(&best, (self.now)()) {
            self.refresh(&supplied, &best, signal).await?
        } else {
            best
        };
        Ok((supplied, used))
    }

    /// `fetch(spec, endpoint, hostFetch, input, init)`: sends one request with
    /// credentials, refreshing once after a 401.
    pub async fn fetch(
        &self,
        endpoint: &str,
        request: FxHttpRequest,
        signal: CancellationToken,
    ) -> Result<FxHttpResponse, FxFetchError> {
        let failed = |error: String| FxFetchError::Failed(error);
        let (_, used) = self.current(signal.clone()).await.map_err(failed)?;
        let mut authorized = request.clone();
        authorize(self.provider, &mut authorized, endpoint, &used);
        let response = (self.fetch)(authorized, signal.clone()).await?;
        if response.status != 401 || used.refresh_token.is_none() {
            return Ok(without_credentials(response, &used).await);
        }
        drop(response);
        let (supplied, latest) = self.current(signal.clone()).await.map_err(failed)?;
        let next = if latest.access_token != used.access_token {
            latest
        } else {
            self.refresh(&supplied, &used, signal.clone())
                .await
                .map_err(failed)?
        };
        let mut authorized = request;
        authorize(self.provider, &mut authorized, endpoint, &next);
        let response = (self.fetch)(authorized, signal).await?;
        Ok(without_credentials(response, &next).await)
    }
}

/// The session id a provider's request names (`spec.sessionId(headers)`).
#[allow(dead_code)]
pub(crate) fn session_id(
    provider: FxSubscriptionProvider,
    headers: &[(String, String)],
) -> Option<String> {
    let name = match provider {
        FxSubscriptionProvider::Codex => "session-id",
        FxSubscriptionProvider::Grok => "x-grok-conv-id",
    };
    header(headers, name).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use super::*;
    use crate::harness::utils::native_subscription::jwt::create_jwt;

    struct Source {
        credentials: StdMutex<FxProviderCredentials>,
        written: StdMutex<Vec<FxProviderCredentials>>,
    }

    impl FxCredentialSource for Source {
        fn read(&self) -> BoxFuture<'_, CredentialResult<FxProviderCredentials>> {
            Box::pin(async { Ok(self.credentials.lock().unwrap().clone()) })
        }
        fn write(&self, credentials: FxProviderCredentials) -> BoxFuture<'_, CredentialResult<()>> {
            Box::pin(async move {
                self.written.lock().unwrap().push(credentials);
                Ok(())
            })
        }
    }

    fn recording_fetch(
        log: Arc<StdMutex<Vec<FxHttpRequest>>>,
        respond: impl Fn(&FxHttpRequest) -> FxHttpResponse + Send + Sync + 'static,
    ) -> FxFetch {
        Arc::new(move |request, _| {
            log.lock().unwrap().push(request.clone());
            let response = respond(&request);
            Box::pin(async move { Ok(response) })
        })
    }

    #[test]
    fn normalizes_account_and_expiry_from_claims() {
        let token = create_jwt(&json!({
            "exp": 100,
            OPENAI_AUTH_CLAIM: { "chatgpt_account_id": "acct" },
        }));
        let credentials = normalize_credentials(
            FxSubscriptionProvider::Codex,
            &FxProviderCredentials {
                access_token: token,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(credentials.account_id.as_deref(), Some("acct"));
        assert_eq!(credentials.expires_at, Some(100_000.0));
        assert_eq!(
            normalize_credentials(
                FxSubscriptionProvider::Grok,
                &FxProviderCredentials {
                    access_token: "opaque".into(),
                    ..Default::default()
                },
            )
            .unwrap_err(),
            "Grok credentials need an accountId; it was not in the access token"
        );
    }

    #[test]
    fn matches_only_the_provider_endpoints() {
        assert!(matches(
            FxSubscriptionProvider::Codex,
            &endpoint_of("https://chatgpt.com/backend-api/codex/responses?x=1").unwrap()
        ));
        assert!(!matches(
            FxSubscriptionProvider::Codex,
            "https://chatgpt.com/backend-api/other"
        ));
        assert!(matches(FxSubscriptionProvider::Grok, GROK_LANGUAGE_MODELS));
        assert!(matches(
            FxSubscriptionProvider::Grok,
            "https://cli-chat-proxy.grok.com/v1/models"
        ));
    }

    #[tokio::test]
    async fn authorizes_refreshes_after_401_and_masks_tokens() {
        let log = Arc::new(StdMutex::new(Vec::new()));
        let new_token = create_jwt(&json!({ OPENAI_AUTH_CLAIM: { "chatgpt_account_id": "acct" } }));
        let token = new_token.clone();
        let fetch = recording_fetch(log.clone(), move |request| {
            if request.url.ends_with("/oauth/token") {
                return FxHttpResponse::new(
                    200,
                    json!({ "access_token": token, "expires_in": 3600 }).to_string(),
                );
            }
            if request.header("authorization") == Some("Bearer old-token") {
                return FxHttpResponse::new(401, "nope");
            }
            FxHttpResponse::new(400, json!({ "error": format!("bad {token}") }).to_string())
        });
        let source = Arc::new(Source {
            credentials: StdMutex::new(FxProviderCredentials {
                access_token: "old-token".into(),
                refresh_token: Some("refresh".into()),
                account_id: Some("acct".into()),
                expires_at: None,
            }),
            written: StdMutex::default(),
        });
        let authorizer =
            SubscriptionAuthorizer::new(FxSubscriptionProvider::Codex, source.clone(), fetch);
        let endpoint = "https://chatgpt.com/backend-api/codex/responses";
        let response = authorizer
            .fetch(
                endpoint,
                FxHttpRequest {
                    url: endpoint.into(),
                    method: "POST".into(),
                    ..Default::default()
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(response.status, 400);
        assert_eq!(
            response.text().await.unwrap(),
            json!({ "error": format!("bad {}", "*".repeat(new_token.len())) }).to_string()
        );
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 3);
        assert_eq!(log[0].header("chatgpt-account-id"), Some("acct"));
        assert_eq!(
            serde_json::from_str::<Value>(&log[1].body_text()).unwrap()["refresh_token"],
            "refresh"
        );
        assert_eq!(
            log[2].header("authorization"),
            Some(format!("Bearer {new_token}").as_str())
        );
        let written = source.written.lock().unwrap();
        assert_eq!(written.len(), 1);
        assert_eq!(written[0].access_token, new_token);
        assert_eq!(written[0].refresh_token.as_deref(), Some("refresh"));
        assert_eq!(written[0].account_id.as_deref(), Some("acct"));
    }

    #[tokio::test]
    async fn refreshes_near_expiry_and_checks_the_grok_account() {
        let log = Arc::new(StdMutex::new(Vec::new()));
        let fetch = recording_fetch(log.clone(), |request| {
            if request.url.ends_with("/oauth2/token") {
                return FxHttpResponse::new(200, json!({ "access_token": "fresh" }).to_string());
            }
            if request.url.ends_with("/oauth2/userinfo") {
                return FxHttpResponse::new(200, json!({ "sub": "other" }).to_string());
            }
            FxHttpResponse::new(200, "ok")
        });
        let source = Arc::new(Source {
            credentials: StdMutex::new(FxProviderCredentials {
                access_token: "stale".into(),
                refresh_token: Some("refresh".into()),
                account_id: Some("user".into()),
                expires_at: Some(0.0),
            }),
            written: StdMutex::default(),
        });
        let authorizer = SubscriptionAuthorizer::new(FxSubscriptionProvider::Grok, source, fetch);
        let endpoint = "https://cli-chat-proxy.grok.com/v1/responses";
        let error = authorizer
            .fetch(
                endpoint,
                FxHttpRequest {
                    url: endpoint.into(),
                    method: "POST".into(),
                    ..Default::default()
                },
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "The refreshed Grok token belongs to a different account"
        );
        let log = log.lock().unwrap();
        assert!(log[0].body_text().contains("client_id=b1a00492"));
    }
}
