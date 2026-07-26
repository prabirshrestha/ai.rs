use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::types::Model;
use crate::{Error, Result};

use super::*;

const GITHUB_COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const GITHUB_COPILOT_USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
const COPILOT_API_VERSION: &str = "2026-06-01";

const COPILOT_TOKEN_EXPIRY_SKEW_MS: u64 = 5 * 60 * 1000;

const DEFAULT_GITHUB_DOMAIN: &str = "github.com";

const DEFAULT_COPILOT_BASE_URL: &str = "https://api.individual.githubcopilot.com";

const GITHUB_COPILOT_REFRESH_TOKEN_KEY: &str = "githubRefreshToken";

const GITHUB_COPILOT_ACCESS_EXPIRES_KEY: &str = "githubAccessExpires";
const GITHUB_COPILOT_AVAILABLE_MODEL_IDS_KEY: &str = "availableModelIds";

const GITHUB_ACCESS_TOKEN_EXPIRY_SKEW_MS: u64 = 5 * 60 * 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GitHubCopilotOAuthProvider;

impl GitHubCopilotOAuthProvider {
    pub const fn id(self) -> &'static str {
        "github-copilot"
    }

    pub const fn name(self) -> &'static str {
        "GitHub Copilot"
    }

    pub async fn login(self, callbacks: OAuthLoginCallbacks) -> Result<OAuthCredentials> {
        login_github_copilot(callbacks).await
    }

    pub async fn refresh_token(self, credentials: &OAuthCredentials) -> Result<OAuthCredentials> {
        refresh_github_copilot_credentials(credentials).await
    }

    pub fn get_api_key(self, credentials: &OAuthCredentials) -> String {
        credentials.access.clone()
    }

    pub fn modify_models(
        self,
        models: impl IntoIterator<Item = Model>,
        credentials: &OAuthCredentials,
    ) -> Vec<Model> {
        modify_github_copilot_models(models, credentials)
    }
}

pub fn github_copilot_oauth_provider() -> GitHubCopilotOAuthProvider {
    GitHubCopilotOAuthProvider
}

#[async_trait]
impl OAuthProviderInterface for GitHubCopilotOAuthProvider {
    fn id(&self) -> &'static str {
        "github-copilot"
    }

    fn name(&self) -> &'static str {
        "GitHub Copilot"
    }

    async fn login(&self, callbacks: OAuthLoginCallbacks) -> Result<OAuthCredentials> {
        login_github_copilot(callbacks).await
    }

    async fn refresh_token(&self, credentials: &OAuthCredentials) -> Result<OAuthCredentials> {
        refresh_github_copilot_credentials(credentials).await
    }

    fn get_api_key(&self, credentials: &OAuthCredentials) -> String {
        credentials.access.clone()
    }

    fn modify_models(&self, models: Vec<Model>, credentials: &OAuthCredentials) -> Vec<Model> {
        modify_github_copilot_models(models, credentials)
    }
}

const GITHUB_COPILOT_ENTERPRISE_URL_KEY: &str = "enterpriseUrl";

pub(crate) fn github_copilot_enterprise_domain(credentials: &OAuthCredentials) -> Option<&str> {
    credentials
        .extra
        .get(GITHUB_COPILOT_ENTERPRISE_URL_KEY)
        .and_then(Value::as_str)
}

pub(crate) fn get_github_copilot_base_url(
    token: Option<&str>,
    enterprise_domain: Option<&str>,
) -> String {
    if let Some(token) = token
        && let Some(base_url) = get_base_url_from_token(token)
    {
        return base_url;
    }
    if let Some(domain) = enterprise_domain.and_then(normalize_domain) {
        return format!("https://copilot-api.{domain}");
    }
    DEFAULT_COPILOT_BASE_URL.to_string()
}

/// A GitHub user access token minted via the device flow (or a refresh-token
/// grant), plus the refresh token and lifetime GitHub returns when the Copilot
/// app opts into expiring user tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GithubUserToken {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

/// The GitHub refresh token (`ghr_...`) persisted alongside the credentials, if
/// the account uses expiring user tokens.
fn github_copilot_refresh_token(credentials: &OAuthCredentials) -> Option<&str> {
    credentials
        .extra
        .get(GITHUB_COPILOT_REFRESH_TOKEN_KEY)
        .and_then(Value::as_str)
}

/// The absolute expiry (Unix epoch millis, skew-adjusted) of the stored GitHub
/// user access token, if known.
fn github_copilot_access_expires(credentials: &OAuthCredentials) -> Option<u64> {
    credentials
        .extra
        .get(GITHUB_COPILOT_ACCESS_EXPIRES_KEY)
        .and_then(Value::as_u64)
}

/// Converts an `expires_in` (seconds) into a skew-adjusted absolute expiry in
/// Unix epoch millis.
fn github_access_expiry_ms(expires_in_seconds: u64) -> u64 {
    crate::utils::time::now_millis()
        .saturating_add(expires_in_seconds.saturating_mul(1000))
        .saturating_sub(GITHUB_ACCESS_TOKEN_EXPIRY_SKEW_MS)
}

/// Whether an error is GitHub rejecting our credentials (401/403), i.e. a signal
/// to try renewing the GitHub user token before giving up.
fn is_github_auth_error(error: &Error) -> bool {
    matches!(error, Error::ApiStatus { status, .. } if matches!(status.as_u16(), 401 | 403))
}

/// POSTs `fields` as `application/x-www-form-urlencoded` to a GitHub OAuth
/// endpoint with the client headers Copilot expects, returning the checked
/// response. Shared by the device-code, token-poll, and refresh-grant calls.
async fn github_oauth_form_post(
    client: &reqwest::Client,
    url: &str,
    fields: &[(&str, &str)],
) -> Result<reqwest::Response> {
    let response = client
        .post(url)
        .header("Accept", "application/json")
        .header("User-Agent", GITHUB_COPILOT_USER_AGENT)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(form_body(fields))
        .send()
        .await?;
    error_for_status(response).await
}

/// Mints a short-lived Copilot API token from a GitHub user access token.
async fn mint_copilot_token(
    github_access_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<CopilotTokenResponse> {
    let domain = enterprise_domain
        .and_then(normalize_domain)
        .unwrap_or_else(|| DEFAULT_GITHUB_DOMAIN.to_string());
    let urls = GitHubCopilotUrls::new(&domain);
    let client = reqwest::Client::new();
    let response = client
        .get(urls.copilot_token_url)
        .headers(copilot_headers(Some(github_access_token))?)
        .send()
        .await?;
    let response = error_for_status(response).await?;

    parse_copilot_token_response(response.json::<Value>().await?)
}

/// Mints Copilot credentials from a GitHub user access token without attempting
/// to renew that token. Kept for the login path and backwards compatibility;
/// callers holding full credentials should prefer
/// [`refresh_github_copilot_credentials`], which can renew an expired GitHub
/// token before minting.
pub async fn refresh_github_copilot_token(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<OAuthCredentials> {
    let token = mint_copilot_token(refresh_token, enterprise_domain).await?;
    let mut credentials = copilot_credentials_from_token(
        refresh_token,
        None,
        None,
        enterprise_domain.and_then(normalize_domain),
        token,
    );
    set_available_copilot_model_ids(&mut credentials, enterprise_domain).await?;
    Ok(credentials)
}

/// Refreshes GitHub Copilot credentials, renewing the underlying GitHub user
/// access token when it has expired.
///
/// GitHub Apps that opt into expiring user tokens issue a `ghu_` access token
/// (valid ~8h) plus a `ghr_` refresh token (valid ~6 months). Previously only the
/// access token was kept, so once it lapsed every Copilot mint failed with
/// `401 Bad credentials` until the user re-ran the device-code login. When a
/// refresh token is present we now exchange it for a fresh GitHub token —
/// proactively when the stored token is known to have expired, and reactively if
/// a mint is rejected — so a long-running proxy self-heals without re-login. With
/// no refresh token (non-expiring tokens, or credentials from before this change)
/// the behaviour is unchanged and a hard failure still surfaces for re-login.
async fn refresh_github_copilot_credentials(
    credentials: &OAuthCredentials,
) -> Result<OAuthCredentials> {
    let enterprise_domain = github_copilot_enterprise_domain(credentials).map(str::to_string);
    let domain = enterprise_domain.as_deref();
    let mut github_access = credentials.refresh.clone();
    let mut github_refresh = github_copilot_refresh_token(credentials).map(str::to_string);
    let mut github_expires = github_copilot_access_expires(credentials);

    // Proactive: the stored GitHub token has expired and we hold a refresh token,
    // so renew before spending a request on a mint that would 401 anyway.
    if let Some(refresh) = github_refresh.clone()
        && github_expires.is_some_and(|expires| crate::utils::time::now_millis() >= expires)
    {
        let renewed = refresh_github_user_token(&refresh, domain).await?;
        github_access = renewed.access_token;
        github_refresh = renewed.refresh_token.or(Some(refresh));
        github_expires = renewed.expires_in.map(github_access_expiry_ms);
    }

    // Reactive: GitHub rejected the token (expiry unknown, or it lapsed just now).
    // Renew once and retry; if the renewal itself fails, surface it for re-login.
    let token = match mint_copilot_token(&github_access, domain).await {
        Ok(token) => token,
        Err(error) => {
            let Some(refresh) = github_refresh.clone() else {
                return Err(error);
            };
            if !is_github_auth_error(&error) {
                return Err(error);
            }
            let renewed = refresh_github_user_token(&refresh, domain).await?;
            github_access = renewed.access_token;
            github_refresh = renewed.refresh_token.or(Some(refresh));
            github_expires = renewed.expires_in.map(github_access_expiry_ms);
            mint_copilot_token(&github_access, domain).await?
        }
    };

    let mut credentials = copilot_credentials_from_token(
        &github_access,
        github_refresh.as_deref(),
        github_expires,
        enterprise_domain.clone(),
        token,
    );
    set_available_copilot_model_ids(&mut credentials, domain).await?;
    Ok(credentials)
}

/// Exchanges a GitHub refresh token (`ghr_`) for a fresh user access token using
/// the device-flow client id. Public clients refresh with just `client_id` (no
/// secret), the same way the token was originally minted.
async fn refresh_github_user_token(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<GithubUserToken> {
    let domain = enterprise_domain
        .and_then(normalize_domain)
        .unwrap_or_else(|| DEFAULT_GITHUB_DOMAIN.to_string());
    let urls = GitHubCopilotUrls::new(&domain);
    let client = reqwest::Client::new();
    refresh_github_user_token_at(&client, &urls.access_token_url, refresh_token).await
}

async fn refresh_github_user_token_at(
    client: &reqwest::Client,
    access_token_url: &str,
    refresh_token: &str,
) -> Result<GithubUserToken> {
    let response = github_oauth_form_post(
        client,
        access_token_url,
        &[
            ("client_id", GITHUB_COPILOT_CLIENT_ID),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
        ],
    )
    .await?;

    match parse_device_token_response(response.json::<Value>().await?) {
        OAuthDeviceCodePollResult::Complete(token) => Ok(token),
        OAuthDeviceCodePollResult::Failed(message) => Err(Error::Provider(message)),
        OAuthDeviceCodePollResult::Pending | OAuthDeviceCodePollResult::SlowDown { .. } => {
            Err(Error::Provider(
                "Unexpected polling response while refreshing GitHub token".to_string(),
            ))
        }
    }
}

pub async fn login_github_copilot(callbacks: OAuthLoginCallbacks) -> Result<OAuthCredentials> {
    let input = (callbacks.on_prompt)(OAuthPrompt {
        message: "GitHub Enterprise URL/domain (blank for github.com)".to_string(),
        placeholder: Some("company.ghe.com".to_string()),
        allow_empty: true,
    })
    .await?;

    if callbacks
        .cancellation_token
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Err(Error::Provider(CANCEL_MESSAGE.to_string()));
    }

    let trimmed = input.trim();
    let enterprise_domain = normalize_domain(&input);
    if !trimmed.is_empty() && enterprise_domain.is_none() {
        return Err(Error::Provider(
            "Invalid GitHub Enterprise URL/domain".to_string(),
        ));
    }
    let domain = enterprise_domain
        .clone()
        .unwrap_or_else(|| DEFAULT_GITHUB_DOMAIN.to_string());

    let device = start_github_device_flow(&domain).await?;
    (callbacks.on_device_code)(OAuthDeviceCodeInfo {
        user_code: device.user_code.clone(),
        verification_uri: device.verification_uri.clone(),
        interval_seconds: device.interval,
        expires_in_seconds: Some(device.expires_in),
    });

    let github =
        poll_for_github_access_token(&domain, device, callbacks.cancellation_token.clone()).await?;
    let github_expires = github.expires_in.map(github_access_expiry_ms);
    let token = mint_copilot_token(&github.access_token, enterprise_domain.as_deref()).await?;

    let mut credentials = copilot_credentials_from_token(
        &github.access_token,
        github.refresh_token.as_deref(),
        github_expires,
        enterprise_domain,
        token,
    );
    set_available_copilot_model_ids(&mut credentials, Some(domain.as_str())).await?;
    Ok(credentials)
}

pub fn modify_github_copilot_models(
    models: impl IntoIterator<Item = Model>,
    credentials: &OAuthCredentials,
) -> Vec<Model> {
    let domain = github_copilot_enterprise_domain(credentials).and_then(normalize_domain);
    let base_url = get_github_copilot_base_url(Some(&credentials.access), domain.as_deref());
    let available_model_ids = available_copilot_model_ids(credentials);
    models
        .into_iter()
        .filter_map(|mut model| {
            if model.provider == "github-copilot" {
                if available_model_ids
                    .as_ref()
                    .is_some_and(|ids| !ids.contains(model.id.as_str()))
                {
                    return None;
                }
                model.base_url = base_url.clone();
            }
            Some(model)
        })
        .collect()
}

fn available_copilot_model_ids(credentials: &OAuthCredentials) -> Option<HashSet<&str>> {
    let ids = credentials
        .extra
        .get(GITHUB_COPILOT_AVAILABLE_MODEL_IDS_KEY)?
        .as_array()?;
    ids.iter().map(Value::as_str).collect()
}

async fn set_available_copilot_model_ids(
    credentials: &mut OAuthCredentials,
    enterprise_domain: Option<&str>,
) -> Result<()> {
    let ids = fetch_available_copilot_model_ids(&credentials.access, enterprise_domain).await?;
    credentials.extra.insert(
        GITHUB_COPILOT_AVAILABLE_MODEL_IDS_KEY.to_string(),
        Value::Array(ids.into_iter().map(Value::String).collect()),
    );
    Ok(())
}

async fn fetch_available_copilot_model_ids(
    copilot_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<Vec<String>> {
    let base_url = get_github_copilot_base_url(Some(copilot_token), enterprise_domain);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    fetch_available_copilot_model_ids_at(&client, &base_url, copilot_token).await
}

async fn fetch_available_copilot_model_ids_at(
    client: &reqwest::Client,
    base_url: &str,
    copilot_token: &str,
) -> Result<Vec<String>> {
    let response = client
        .get(format!("{base_url}/models"))
        .headers(copilot_headers(Some(copilot_token))?)
        .header("X-GitHub-Api-Version", COPILOT_API_VERSION)
        .send()
        .await?;
    let response = error_for_status(response).await?;
    parse_available_copilot_model_ids(response.json::<Value>().await?)
}

fn parse_available_copilot_model_ids(raw: Value) -> Result<Vec<String>> {
    let Some(data) = raw
        .as_object()
        .and_then(|object| object.get("data"))
        .and_then(Value::as_array)
    else {
        return Err(Error::Provider(
            "Invalid Copilot models response".to_string(),
        ));
    };

    Ok(data
        .iter()
        .filter_map(Value::as_object)
        .filter(|item| item.get("model_picker_enabled").and_then(Value::as_bool) == Some(true))
        .filter(|item| {
            item.get("policy")
                .and_then(Value::as_object)
                .and_then(|policy| policy.get("state"))
                .and_then(Value::as_str)
                != Some("disabled")
        })
        .filter(|item| {
            item.get("capabilities")
                .and_then(Value::as_object)
                .and_then(|capabilities| capabilities.get("supports"))
                .and_then(Value::as_object)
                .and_then(|supports| supports.get("tool_calls"))
                .and_then(Value::as_bool)
                != Some(false)
        })
        .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_string))
        .collect())
}

async fn start_github_device_flow(domain: &str) -> Result<DeviceCodeResponse> {
    let urls = GitHubCopilotUrls::new(domain);
    let client = reqwest::Client::new();
    let response = github_oauth_form_post(
        &client,
        &urls.device_code_url,
        &[
            ("client_id", GITHUB_COPILOT_CLIENT_ID),
            ("scope", "read:user"),
        ],
    )
    .await?;
    let device = response.json::<DeviceCodeResponse>().await?;
    if device.device_code.is_empty()
        || device.user_code.is_empty()
        || device.verification_uri.is_empty()
        || device.expires_in == 0
    {
        return Err(Error::Provider(
            "Invalid device code response fields".to_string(),
        ));
    }
    Ok(device)
}

async fn poll_for_github_access_token(
    domain: &str,
    device: DeviceCodeResponse,
    cancellation_token: Option<CancellationToken>,
) -> Result<GithubUserToken> {
    let urls = GitHubCopilotUrls::new(domain);
    let client = reqwest::Client::new();
    poll_oauth_device_code_flow_inner(
        device.interval,
        Some(device.expires_in),
        true,
        cancellation_token,
        move || {
            let client = client.clone();
            let access_token_url = urls.access_token_url.clone();
            let device_code = device.device_code.clone();
            async move {
                let response = github_oauth_form_post(
                    &client,
                    &access_token_url,
                    &[
                        ("client_id", GITHUB_COPILOT_CLIENT_ID),
                        ("device_code", device_code.as_str()),
                        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ],
                )
                .await?;
                Ok(parse_device_token_response(response.json::<Value>().await?))
            }
        },
    )
    .await
}

fn get_base_url_from_token(token: &str) -> Option<String> {
    let proxy_host = token
        .split(';')
        .find_map(|part| part.strip_prefix("proxy-ep="))?;
    let api_host = proxy_host
        .strip_prefix("proxy.")
        .map(|host| format!("api.{host}"))
        .unwrap_or_else(|| proxy_host.to_string());
    Some(format!("https://{api_host}"))
}

fn parse_device_token_response(raw: Value) -> OAuthDeviceCodePollResult<GithubUserToken> {
    let Some(object) = raw.as_object() else {
        return OAuthDeviceCodePollResult::Failed("Invalid device token response".to_string());
    };

    if let Some(access_token) = object.get("access_token").and_then(Value::as_str) {
        return OAuthDeviceCodePollResult::Complete(GithubUserToken {
            access_token: access_token.to_string(),
            refresh_token: object
                .get("refresh_token")
                .and_then(Value::as_str)
                .map(str::to_string),
            expires_in: object.get("expires_in").and_then(Value::as_u64),
        });
    }

    let Some(error) = object.get("error").and_then(Value::as_str) else {
        return OAuthDeviceCodePollResult::Failed("Invalid device token response".to_string());
    };

    match error {
        "authorization_pending" => OAuthDeviceCodePollResult::Pending,
        "slow_down" => OAuthDeviceCodePollResult::SlowDown {
            interval_seconds: object.get("interval").and_then(Value::as_u64),
        },
        error => {
            let suffix = object
                .get("error_description")
                .and_then(Value::as_str)
                .map(|description| format!(": {description}"))
                .unwrap_or_default();
            OAuthDeviceCodePollResult::Failed(format!("Device flow failed: {error}{suffix}"))
        }
    }
}

fn parse_copilot_token_response(raw: Value) -> Result<CopilotTokenResponse> {
    let Some(object) = raw.as_object() else {
        return Err(Error::Provider(
            "Invalid Copilot token response".to_string(),
        ));
    };

    let Some(token) = object.get("token").and_then(Value::as_str) else {
        return Err(Error::Provider(
            "Invalid Copilot token response fields".to_string(),
        ));
    };
    let Some(expires_at) = object.get("expires_at").and_then(Value::as_u64) else {
        return Err(Error::Provider(
            "Invalid Copilot token response fields".to_string(),
        ));
    };

    Ok(CopilotTokenResponse {
        token: token.to_string(),
        expires_at,
    })
}

fn copilot_credentials_from_token(
    github_access_token: &str,
    github_refresh_token: Option<&str>,
    github_access_expires_at_ms: Option<u64>,
    enterprise_domain: Option<String>,
    token: CopilotTokenResponse,
) -> OAuthCredentials {
    let mut extra = HashMap::new();
    if let Some(enterprise_domain) = enterprise_domain {
        extra.insert(
            GITHUB_COPILOT_ENTERPRISE_URL_KEY.to_string(),
            Value::String(enterprise_domain),
        );
    }
    if let Some(refresh_token) = github_refresh_token {
        extra.insert(
            GITHUB_COPILOT_REFRESH_TOKEN_KEY.to_string(),
            Value::String(refresh_token.to_string()),
        );
    }
    if let Some(expires_at) = github_access_expires_at_ms {
        extra.insert(
            GITHUB_COPILOT_ACCESS_EXPIRES_KEY.to_string(),
            Value::from(expires_at),
        );
    }

    OAuthCredentials {
        refresh: github_access_token.to_string(),
        access: token.token,
        expires: token
            .expires_at
            .saturating_mul(1000)
            .saturating_sub(COPILOT_TOKEN_EXPIRY_SKEW_MS),
        extra,
    }
}

fn copilot_headers(refresh_token: Option<&str>) -> Result<reqwest::header::HeaderMap> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        "Accept",
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    headers.insert(
        "User-Agent",
        reqwest::header::HeaderValue::from_static(GITHUB_COPILOT_USER_AGENT),
    );
    headers.insert(
        "Editor-Version",
        reqwest::header::HeaderValue::from_static("vscode/1.107.0"),
    );
    headers.insert(
        "Editor-Plugin-Version",
        reqwest::header::HeaderValue::from_static("copilot-chat/0.35.0"),
    );
    headers.insert(
        "Copilot-Integration-Id",
        reqwest::header::HeaderValue::from_static("vscode-chat"),
    );
    if let Some(token) = refresh_token {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|e| Error::InvalidHeaderValue("authorization".to_string(), e))?,
        );
    }
    Ok(headers)
}

#[derive(Debug)]
struct GitHubCopilotUrls {
    device_code_url: String,
    access_token_url: String,
    copilot_token_url: String,
}

impl GitHubCopilotUrls {
    fn new(domain: &str) -> Self {
        Self {
            device_code_url: format!("https://{domain}/login/device/code"),
            access_token_url: format!("https://{domain}/login/oauth/access_token"),
            copilot_token_url: format!("https://api.{domain}/copilot_internal/v2/token"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval: Option<u64>,
    expires_in: u64,
}

#[derive(Debug)]
struct CopilotTokenResponse {
    token: String,
    expires_at: u64,
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::oauth::test_support::read_http_request;
    use crate::types::ModelCost;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    #[test]
    fn resolves_copilot_base_url_from_token_or_enterprise_domain() {
        assert_eq!(
            get_github_copilot_base_url(
                Some("tid=test;proxy-ep=proxy.individual.githubcopilot.com;exp=1"),
                None,
            ),
            DEFAULT_COPILOT_BASE_URL
        );
        assert_eq!(
            get_github_copilot_base_url(None, Some("https://company.ghe.com/path")),
            "https://copilot-api.company.ghe.com"
        );
        assert_eq!(
            get_github_copilot_base_url(None, None),
            DEFAULT_COPILOT_BASE_URL
        );
    }

    #[test]
    fn copilot_credentials_apply_expiry_skew() {
        let credentials = copilot_credentials_from_token(
            "refresh-token",
            None,
            None,
            Some("company.ghe.com".to_string()),
            CopilotTokenResponse {
                token: "access-token".to_string(),
                expires_at: 1_000,
            },
        );

        assert_eq!(credentials.refresh, "refresh-token");
        assert_eq!(credentials.access, "access-token");
        assert_eq!(credentials.expires, 700_000);
        assert_eq!(
            github_copilot_enterprise_domain(&credentials),
            Some("company.ghe.com")
        );
    }

    #[test]
    fn oauth_credentials_flatten_provider_specific_extra_fields() {
        let credentials = copilot_credentials_from_token(
            "refresh-token",
            None,
            None,
            Some("company.ghe.com".to_string()),
            CopilotTokenResponse {
                token: "access-token".to_string(),
                expires_at: 1_000,
            },
        );

        let json = serde_json::to_value(&credentials).expect("credentials json");
        assert_eq!(json["enterpriseUrl"], "company.ghe.com");
        assert!(json.get("extra").is_none());

        let round_tripped: OAuthCredentials =
            serde_json::from_value(json).expect("round-tripped credentials");
        assert_eq!(
            github_copilot_enterprise_domain(&round_tripped),
            Some("company.ghe.com")
        );
    }

    #[test]
    fn provider_updates_only_github_copilot_model_base_urls() {
        let credentials = OAuthCredentials {
            refresh: "refresh".to_string(),
            access: "tid=test;proxy-ep=proxy.enterprise.example.com;exp=1".to_string(),
            expires: 1,
            extra: HashMap::new(),
        };
        let models = vec![
            Model {
                id: "gpt".to_string(),
                name: "gpt".to_string(),
                api: "openai-completions".to_string(),
                provider: "github-copilot".to_string(),
                base_url: "https://old.example.com".to_string(),
                cost: ModelCost::default(),
                ..Default::default()
            },
            Model {
                id: "claude".to_string(),
                name: "claude".to_string(),
                api: "anthropic-messages".to_string(),
                provider: "anthropic".to_string(),
                base_url: "https://api.anthropic.com".to_string(),
                cost: ModelCost::default(),
                ..Default::default()
            },
        ];

        let updated = github_copilot_oauth_provider().modify_models(models, &credentials);

        assert_eq!(updated[0].base_url, "https://api.enterprise.example.com");
        assert_eq!(updated[1].base_url, "https://api.anthropic.com");
    }

    #[test]
    fn provider_filters_copilot_models_to_account_availability() {
        let credentials = OAuthCredentials {
            refresh: "refresh".to_string(),
            access: "tid=test;proxy-ep=proxy.enterprise.example.com;exp=1".to_string(),
            expires: 1,
            extra: HashMap::from([(
                GITHUB_COPILOT_AVAILABLE_MODEL_IDS_KEY.to_string(),
                serde_json::json!(["gpt-4.1"]),
            )]),
        };
        let models = vec![
            Model {
                id: "gpt-4.1".to_string(),
                provider: "github-copilot".to_string(),
                ..Default::default()
            },
            Model {
                id: "claude-opus-4.7".to_string(),
                provider: "github-copilot".to_string(),
                ..Default::default()
            },
            Model {
                id: "claude-opus-4.7".to_string(),
                provider: "anthropic".to_string(),
                ..Default::default()
            },
        ];

        let updated = github_copilot_oauth_provider().modify_models(models, &credentials);

        assert_eq!(
            updated
                .iter()
                .map(|model| (model.provider.as_str(), model.id.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("github-copilot", "gpt-4.1"),
                ("anthropic", "claude-opus-4.7")
            ]
        );
        assert_eq!(updated[0].base_url, "https://api.enterprise.example.com");
    }

    #[test]
    fn parses_only_selectable_copilot_models() {
        let ids = parse_available_copilot_model_ids(serde_json::json!({
            "data": [
                {
                    "id": "gpt-4.1",
                    "model_picker_enabled": true,
                    "capabilities": { "supports": { "tool_calls": true } }
                },
                {
                    "id": "claude-opus-4.7",
                    "model_picker_enabled": true,
                    "policy": { "state": "disabled" },
                    "capabilities": { "supports": { "tool_calls": true } }
                },
                {
                    "id": "gpt-5.4-nano",
                    "model_picker_enabled": false,
                    "capabilities": { "supports": { "tool_calls": true } }
                },
                {
                    "id": "text-only",
                    "model_picker_enabled": true,
                    "capabilities": { "supports": { "tool_calls": false } }
                }
            ]
        }))
        .expect("models response");

        assert_eq!(ids, vec!["gpt-4.1"]);
        assert_eq!(
            parse_available_copilot_model_ids(serde_json::json!({}))
                .unwrap_err()
                .to_string(),
            "provider error: Invalid Copilot models response"
        );
    }

    #[tokio::test]
    async fn fetches_available_models_with_copilot_headers() {
        let captured_request = Arc::new(Mutex::new(None));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = {
            let captured_request = Arc::clone(&captured_request);
            tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = read_http_request(&mut socket).await;
                *captured_request.lock().expect("request lock poisoned") = Some(request);
                let body = serde_json::json!({
                    "data": [{
                        "id": "gpt-4.1",
                        "model_picker_enabled": true,
                        "capabilities": { "supports": { "tool_calls": true } }
                    }]
                })
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            })
        };
        let client = reqwest::Client::new();

        let ids = fetch_available_copilot_model_ids_at(
            &client,
            &format!("http://{addr}"),
            "copilot-token",
        )
        .await
        .expect("available models");
        server.await.unwrap();

        assert_eq!(ids, vec!["gpt-4.1"]);
        let request = captured_request
            .lock()
            .expect("request lock poisoned")
            .clone()
            .expect("captured request");
        assert!(request.starts_with("GET /models HTTP/1.1"), "{request}");
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer copilot-token"),
            "{request}"
        );
        assert!(
            request.contains("x-github-api-version: 2026-06-01"),
            "{request}"
        );
    }

    #[test]
    fn parses_device_token_response() {
        assert_eq!(
            parse_device_token_response(serde_json::json!({ "access_token": "ghu_refresh" })),
            OAuthDeviceCodePollResult::Complete(GithubUserToken {
                access_token: "ghu_refresh".to_string(),
                refresh_token: None,
                expires_in: None,
            })
        );
        assert_eq!(
            parse_device_token_response(serde_json::json!({
                "error": "authorization_pending",
                "error_description": "pending"
            })),
            OAuthDeviceCodePollResult::Pending
        );
        assert_eq!(
            parse_device_token_response(serde_json::json!({
                "error": "slow_down",
                "error_description": "slow down",
                "interval": 7
            })),
            OAuthDeviceCodePollResult::SlowDown {
                interval_seconds: Some(7),
            }
        );
        assert_eq!(
            parse_device_token_response(serde_json::json!({
                "error": "access_denied",
                "error_description": "denied"
            })),
            OAuthDeviceCodePollResult::Failed(
                "Device flow failed: access_denied: denied".to_string()
            )
        );
        assert_eq!(
            parse_device_token_response(serde_json::json!({ "access_token": 1 })),
            OAuthDeviceCodePollResult::Failed("Invalid device token response".to_string())
        );
        assert_eq!(
            parse_device_token_response(serde_json::Value::Null),
            OAuthDeviceCodePollResult::Failed("Invalid device token response".to_string())
        );
    }

    #[test]
    fn parses_copilot_token_response() {
        let token = parse_copilot_token_response(serde_json::json!({
            "token": "tid=test;exp=9999999999",
            "expires_at": 9999999999_u64
        }))
        .unwrap();

        assert_eq!(token.token, "tid=test;exp=9999999999");
        assert_eq!(token.expires_at, 9999999999);
        assert_eq!(
            parse_copilot_token_response(serde_json::Value::Null)
                .unwrap_err()
                .to_string(),
            "provider error: Invalid Copilot token response"
        );
        assert_eq!(
            parse_copilot_token_response(serde_json::json!({ "token": "tid=test" }))
                .unwrap_err()
                .to_string(),
            "provider error: Invalid Copilot token response fields"
        );
        assert_eq!(
            parse_copilot_token_response(serde_json::json!({
                "token": "tid=test",
                "expires_at": "later"
            }))
            .unwrap_err()
            .to_string(),
            "provider error: Invalid Copilot token response fields"
        );
    }

    #[test]
    fn provider_metadata_matches_expected_shape() {
        let provider = github_copilot_oauth_provider();
        assert_eq!(provider.id(), "github-copilot");
        assert_eq!(provider.name(), "GitHub Copilot");
    }

    #[test]
    fn copilot_headers_include_static_client_metadata_and_bearer() {
        let headers = copilot_headers(Some("refresh-token")).unwrap();

        assert_eq!(
            headers
                .get("user-agent")
                .and_then(|value| value.to_str().ok()),
            Some("GitHubCopilotChat/0.35.0")
        );
        assert_eq!(
            headers
                .get("editor-version")
                .and_then(|value| value.to_str().ok()),
            Some("vscode/1.107.0")
        );
        assert_eq!(
            headers
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer refresh-token")
        );
    }

    #[test]
    fn form_body_matches_url_search_params_encoding() {
        assert_eq!(
            form_body(&[
                ("client_id", GITHUB_COPILOT_CLIENT_ID),
                ("scope", "read:user"),
                ("device_code", "abc def/ghi")
            ]),
            "client_id=Iv1.b507a08c87ecfe98&scope=read%3Auser&device_code=abc+def%2Fghi"
        );
    }

    #[test]
    fn device_token_response_captures_refresh_token_and_expiry() {
        match parse_device_token_response(serde_json::json!({
            "access_token": "ghu_new",
            "refresh_token": "ghr_new",
            "expires_in": 28_800,
            "token_type": "bearer"
        })) {
            OAuthDeviceCodePollResult::Complete(token) => {
                assert_eq!(token.access_token, "ghu_new");
                assert_eq!(token.refresh_token.as_deref(), Some("ghr_new"));
                assert_eq!(token.expires_in, Some(28_800));
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn device_token_response_without_refresh_token_is_non_expiring() {
        match parse_device_token_response(
            serde_json::json!({ "access_token": "gho_classic", "token_type": "bearer" }),
        ) {
            OAuthDeviceCodePollResult::Complete(token) => {
                assert_eq!(token.access_token, "gho_classic");
                assert!(token.refresh_token.is_none());
                assert!(token.expires_in.is_none());
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn github_auth_errors_signal_token_renewal() {
        assert!(is_github_auth_error(&Error::ApiStatus {
            status: reqwest::StatusCode::UNAUTHORIZED,
            body: String::new(),
        }));
        assert!(is_github_auth_error(&Error::ApiStatus {
            status: reqwest::StatusCode::FORBIDDEN,
            body: String::new(),
        }));
        assert!(!is_github_auth_error(&Error::ApiStatus {
            status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            body: String::new(),
        }));
        assert!(!is_github_auth_error(&Error::Provider("nope".to_string())));
    }

    #[test]
    fn copilot_credentials_persist_refresh_token_and_expiry() {
        let credentials = copilot_credentials_from_token(
            "ghu_access",
            Some("ghr_refresh"),
            Some(1_700_000_000_000),
            None,
            CopilotTokenResponse {
                token: "copilot-token".to_string(),
                expires_at: 2_000,
            },
        );

        assert_eq!(credentials.refresh, "ghu_access");
        assert_eq!(
            github_copilot_refresh_token(&credentials),
            Some("ghr_refresh")
        );
        assert_eq!(
            github_copilot_access_expires(&credentials),
            Some(1_700_000_000_000)
        );

        let json = serde_json::to_value(&credentials).expect("credentials json");
        assert_eq!(json["githubRefreshToken"], "ghr_refresh");
        assert_eq!(
            json["githubAccessExpires"].as_u64(),
            Some(1_700_000_000_000)
        );
        let round_tripped: OAuthCredentials =
            serde_json::from_value(json).expect("round-tripped credentials");
        assert_eq!(
            github_copilot_refresh_token(&round_tripped),
            Some("ghr_refresh")
        );
    }

    #[test]
    fn copilot_credentials_omit_refresh_metadata_when_absent() {
        let credentials = copilot_credentials_from_token(
            "ghu_access",
            None,
            None,
            None,
            CopilotTokenResponse {
                token: "copilot-token".to_string(),
                expires_at: 2_000,
            },
        );

        assert!(github_copilot_refresh_token(&credentials).is_none());
        assert!(github_copilot_access_expires(&credentials).is_none());
        let json = serde_json::to_value(&credentials).expect("credentials json");
        assert!(json.get("githubRefreshToken").is_none());
        assert!(json.get("githubAccessExpires").is_none());
    }

    #[tokio::test]
    async fn github_user_token_refresh_uses_refresh_token_grant() {
        let captured_body = Arc::new(Mutex::new(None));
        let token_url = spawn_github_refresh_server(Arc::clone(&captured_body)).await;
        let client = reqwest::Client::new();

        let token = refresh_github_user_token_at(&client, &token_url, "ghr_old")
            .await
            .unwrap();

        assert_eq!(token.access_token, "ghu_refreshed");
        assert_eq!(token.refresh_token.as_deref(), Some("ghr_rotated"));
        assert_eq!(token.expires_in, Some(28_800));

        let body = captured_body
            .lock()
            .expect("captured body lock poisoned")
            .clone()
            .expect("captured request body");
        assert!(body.contains("grant_type=refresh_token"), "body: {body}");
        assert!(body.contains("refresh_token=ghr_old"), "body: {body}");
        assert!(
            body.contains(&format!("client_id={GITHUB_COPILOT_CLIENT_ID}")),
            "body: {body}"
        );
    }

    async fn spawn_github_refresh_server(captured_body: Arc<Mutex<Option<String>>>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut socket).await;
            let body = request
                .split_once("\r\n\r\n")
                .map(|(_, body)| body.to_string())
                .unwrap_or_default();
            *captured_body.lock().expect("captured body lock poisoned") = Some(body);
            let response_body = serde_json::json!({
                "access_token": "ghu_refreshed",
                "refresh_token": "ghr_rotated",
                "expires_in": 28_800,
                "token_type": "bearer",
                "scope": ""
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{addr}/login/oauth/access_token")
    }
}
