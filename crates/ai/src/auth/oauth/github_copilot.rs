//! Port of `auth/oauth/github-copilot.ts`: GitHub Copilot OAuth (device
//! flow), Copilot token minting, the account model catalog and model policy
//! enablement.
//!
//! Divergence from Pi (kept from the pre-1.0 ai.rs port on purpose): GitHub
//! Apps that opt into expiring user tokens return a `ghu_` access token (about
//! 8 h) plus a `ghr_` refresh token (about 6 months) from the device flow. Pi
//! keeps only the access token, so once it lapses every Copilot token mint
//! fails with `401 Bad credentials` until the user logs in again. This port
//! also stores the refresh token (`githubRefreshToken`) and the access
//! token's expiry (`githubAccessExpires`) in the credential, and `refresh`
//! renews the GitHub token with a `refresh_token` grant: proactively when the
//! stored token is known to have expired, and reactively once when a mint is
//! rejected with 401/403. Credentials without a refresh token behave exactly
//! as in Pi.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Map, Value, json};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::compat::{OAuthLoginCallbacks, callbacks_interaction};
use super::device_code::{
    OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, poll_oauth_device_code_flow,
};
use super::fetch::{
    FetchRequest, FetchResponse, OAuthFetch, default_oauth_fetch, fetch_with_timeout,
    url_search_params,
};
use crate::auth::types::{
    AuthEvent, AuthPrompt, AuthPromptKind, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential,
    ProviderAuthInteraction,
};
use crate::providers::catalog::github_copilot_models;
use crate::types::Model;
use crate::utils::sleep::sleep;
use crate::utils::time::now_millis;
use crate::{Error, Result};

fn client_id() -> String {
    String::from_utf8(
        STANDARD
            .decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=")
            .expect("static base64"),
    )
    .expect("static utf-8")
}

const USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
const COPILOT_HEADERS: [(&str, &str); 4] = [
    ("User-Agent", USER_AGENT),
    ("Editor-Version", "vscode/1.107.0"),
    ("Editor-Plugin-Version", "copilot-chat/0.35.0"),
    ("Copilot-Integration-Id", "vscode-chat"),
];
const COPILOT_API_VERSION: &str = "2026-06-01";
const DEFAULT_BASE_URL: &str = "https://api.individual.githubcopilot.com";
const EXPIRY_SKEW_MS: u64 = 5 * 60 * 1000;

const ENTERPRISE_URL_KEY: &str = "enterpriseUrl";
const AVAILABLE_MODEL_IDS_KEY: &str = "availableModelIds";
/// ai.rs-only credential fields of the `ghr_` refresh-token grant.
const GITHUB_REFRESH_TOKEN_KEY: &str = "githubRefreshToken";
const GITHUB_ACCESS_EXPIRES_KEY: &str = "githubAccessExpires";

#[derive(Debug, Clone, PartialEq)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval: Option<f64>,
    expires_in: f64,
}

/// A GitHub user access token from the device flow (or the refresh grant),
/// plus the refresh token and lifetime GitHub returns for expiring tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GitHubUserToken {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

/// Retry policy of [`fetch_with_rate_limit_retry`].
#[derive(Debug, Clone, Copy)]
struct RetryPolicy {
    max_retries: u32,
    max_elapsed_ms: u64,
}

const LOGIN_RETRY_POLICY: RetryPolicy = RetryPolicy {
    max_retries: 2,
    max_elapsed_ms: 5000,
};
const REFRESH_RETRY_POLICY: RetryPolicy = RetryPolicy {
    max_retries: 0,
    max_elapsed_ms: 0,
};

/// `normalizeDomain()`: the hostname of a URL or bare domain.
pub fn normalize_domain(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let candidate = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}")
    };
    reqwest::Url::parse(&candidate)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
}

struct Urls {
    device_code_url: String,
    access_token_url: String,
    copilot_token_url: String,
}

fn get_urls(domain: &str) -> Urls {
    Urls {
        device_code_url: format!("https://{domain}/login/device/code"),
        access_token_url: format!("https://{domain}/login/oauth/access_token"),
        copilot_token_url: format!("https://api.{domain}/copilot_internal/v2/token"),
    }
}

/// Parse the `proxy-ep` from a Copilot token and convert it to an API base
/// URL: `tid=...;proxy-ep=proxy.individual.githubcopilot.com;...` becomes
/// `https://api.individual.githubcopilot.com`.
fn get_base_url_from_token(token: &str) -> Option<String> {
    let start = token.find("proxy-ep=")? + "proxy-ep=".len();
    let proxy_host = token[start..].split(';').next()?;
    if proxy_host.is_empty() {
        return None;
    }
    let api_host = match proxy_host.strip_prefix("proxy.") {
        Some(rest) => format!("api.{rest}"),
        None => proxy_host.to_string(),
    };
    Some(format!("https://{api_host}"))
}

/// `getGitHubCopilotBaseUrl()`: the token's proxy endpoint, else the
/// enterprise endpoint, else the Individual endpoint.
pub fn get_github_copilot_base_url(token: Option<&str>, enterprise_domain: Option<&str>) -> String {
    if let Some(url) = token.and_then(get_base_url_from_token) {
        return url;
    }
    if let Some(domain) = enterprise_domain.filter(|domain| !domain.is_empty()) {
        return format!("https://copilot-api.{domain}");
    }
    DEFAULT_BASE_URL.to_string()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ModelCatalog {
    available_model_ids: Vec<String>,
    policy_model_ids: Vec<String>,
}

fn parse_github_copilot_model_catalog(
    raw: &Value,
    allow_policy_fallback: bool,
) -> Result<ModelCatalog> {
    let Some(data) = raw.get("data").and_then(Value::as_array) else {
        return Err(Error::message("Invalid Copilot models response"));
    };

    struct AccountModel<'a> {
        id: &'a str,
        picker_enabled: bool,
        policy_state: Option<&'a Value>,
    }
    let account_models: Vec<AccountModel> = data
        .iter()
        .filter_map(|item| {
            let id = item.get("id")?.as_str()?;
            if item
                .get("capabilities")
                .and_then(|capabilities| capabilities.get("supports"))
                .and_then(|supports| supports.get("tool_calls"))
                == Some(&Value::Bool(false))
            {
                return None;
            }
            Some(AccountModel {
                id,
                picker_enabled: item.get("model_picker_enabled") == Some(&Value::Bool(true)),
                policy_state: item.get("policy").and_then(|policy| policy.get("state")),
            })
        })
        .collect();
    let state_is = |model: &AccountModel, state: &str| {
        model.policy_state.and_then(Value::as_str) == Some(state)
    };
    let picker_model_ids: Vec<String> = account_models
        .iter()
        .filter(|model| model.picker_enabled && !state_is(model, "disabled"))
        .map(|model| model.id.to_string())
        .collect();
    let use_policy_fallback = allow_policy_fallback && picker_model_ids.is_empty();
    let available_model_ids = if !picker_model_ids.is_empty() || !allow_policy_fallback {
        picker_model_ids
    } else {
        account_models
            .iter()
            .filter(|model| state_is(model, "enabled"))
            .map(|model| model.id.to_string())
            .collect()
    };
    let catalog = github_copilot_models();
    let policy_model_ids = account_models
        .iter()
        .filter(|model| {
            state_is(model, "unconfigured")
                && catalog.contains_key(model.id)
                && (model.picker_enabled || use_policy_fallback)
        })
        .map(|model| model.id.to_string())
        .collect();
    Ok(ModelCatalog {
        available_model_ids,
        policy_model_ids,
    })
}

/// `Number.parseFloat()`: the longest numeric prefix.
fn parse_float_prefix(value: &str) -> Option<f64> {
    let value = value.trim_start();
    (1..=value.len())
        .rev()
        .filter(|end| value.is_char_boundary(*end))
        .find_map(|end| {
            let prefix = &value[..end];
            if prefix.ends_with(|c: char| c.is_ascii_alphabetic()) {
                return None;
            }
            prefix.parse::<f64>().ok()
        })
}

fn deadline_remaining_ms(deadline: Instant) -> u64 {
    deadline
        .saturating_duration_since(Instant::now())
        .as_millis() as u64
}

/// `fetchWithRateLimitRetry()`: retry 429 responses with exponential backoff
/// (`500 * 2^n` ms) or `Retry-After`, within a retry count and time budget.
async fn fetch_with_rate_limit_retry(
    fetch: &OAuthFetch,
    request: FetchRequest,
    signal: &CancellationToken,
    retry_policy: RetryPolicy,
) -> Result<FetchResponse> {
    let retry_deadline = (retry_policy.max_retries > 0 && retry_policy.max_elapsed_ms > 0)
        .then(|| Instant::now() + Duration::from_millis(retry_policy.max_elapsed_ms));
    let mut retry = 0u32;
    loop {
        let timeout_ms =
            retry_deadline.map_or(5000, |deadline| 5000.min(deadline_remaining_ms(deadline)));
        let response = fetch_with_timeout(
            fetch,
            request.clone(),
            signal,
            Duration::from_millis(timeout_ms),
        )
        .await?;
        if response.status != 429 || retry == retry_policy.max_retries {
            return Ok(response);
        }

        let mut delay_ms = 500.0 * 2f64.powi(retry as i32);
        if let Some(retry_after) = response.header("retry-after") {
            delay_ms = match parse_float_prefix(retry_after) {
                Some(seconds) => seconds * 1000.0,
                None => match httpdate::parse_http_date(retry_after.trim()) {
                    Ok(date) => {
                        let date_ms = date
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|duration| duration.as_millis() as f64)
                            .unwrap_or_default();
                        date_ms - now_millis() as f64
                    }
                    Err(_) => return Ok(response),
                },
            };
            if !delay_ms.is_finite() {
                return Ok(response);
            }
        }
        let delay_ms = delay_ms.max(0.0) as u64;
        if let Some(deadline) = retry_deadline
            && delay_ms >= deadline_remaining_ms(deadline)
        {
            return Ok(response);
        }
        sleep(delay_ms, signal).await?;
        retry += 1;
    }
}

/// `fetchJson()` failure: an HTTP status (kept for the `ghr_` renewal) or any
/// other error.
enum FetchJsonError {
    Status { status: u16, error: Error },
    Other(Error),
}

impl From<FetchJsonError> for Error {
    fn from(error: FetchJsonError) -> Self {
        match error {
            FetchJsonError::Status { error, .. } | FetchJsonError::Other(error) => error,
        }
    }
}

fn status_error(response: &FetchResponse) -> Error {
    Error::message(format!(
        "{} {}: {}",
        response.status, response.status_text, response.body
    ))
}

async fn fetch_json(
    fetch: &OAuthFetch,
    request: FetchRequest,
    signal: &CancellationToken,
) -> std::result::Result<Value, FetchJsonError> {
    let response = fetch(request, signal.clone())
        .await
        .map_err(FetchJsonError::Other)?;
    if !response.ok() {
        return Err(FetchJsonError::Status {
            status: response.status,
            error: status_error(&response),
        });
    }
    serde_json::from_str(&response.body).map_err(|error| FetchJsonError::Other(error.into()))
}

fn with_copilot_headers(mut request: FetchRequest) -> FetchRequest {
    for (name, value) in COPILOT_HEADERS {
        request = request.header(name, value);
    }
    request
}

async fn fetch_github_copilot_models(
    fetch: &OAuthFetch,
    copilot_token: &str,
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
    retry_policy: RetryPolicy,
) -> Result<ModelCatalog> {
    let base_url = get_github_copilot_base_url(Some(copilot_token), enterprise_domain);
    // Some Individual accounts return false for every picker flag despite explicit enabled
    // policies. Limit the fallback to that endpoint so other account types keep strict picker
    // semantics.
    let allow_policy_fallback = base_url == DEFAULT_BASE_URL;
    let request = with_copilot_headers(
        FetchRequest::get(format!("{base_url}/models"))
            .header("Accept", "application/json")
            .header("Authorization", format!("Bearer {copilot_token}")),
    )
    .header("X-GitHub-Api-Version", COPILOT_API_VERSION);
    let response = fetch_with_rate_limit_retry(fetch, request, signal, retry_policy).await?;
    if !response.ok() {
        return Err(status_error(&response));
    }
    let raw: Value = serde_json::from_str(&response.body)?;
    parse_github_copilot_model_catalog(&raw, allow_policy_fallback)
}

fn form_post(url: &str, fields: &[(&str, &str)]) -> FetchRequest {
    FetchRequest::post(url)
        .header("Accept", "application/json")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("User-Agent", USER_AGENT)
        .body(url_search_params(fields))
}

/// The verification URI is opened in the user's browser; to keep a launcher
/// from opening an executable or similar, it must be an http(s) URL.
fn normalize_verification_uri(raw: &str) -> Result<String> {
    let untrusted = || Error::message("Untrusted verification_uri in device code response");
    let url = reqwest::Url::parse(raw).map_err(|_| untrusted())?;
    if !matches!(url.scheme(), "https" | "http") {
        return Err(untrusted());
    }
    Ok(url.to_string())
}

async fn start_device_flow(
    fetch: &OAuthFetch,
    domain: &str,
    signal: &CancellationToken,
) -> Result<DeviceCodeResponse> {
    let urls = get_urls(domain);
    let client_id = client_id();
    let data = fetch_json(
        fetch,
        form_post(
            &urls.device_code_url,
            &[("client_id", client_id.as_str()), ("scope", "read:user")],
        ),
        signal,
    )
    .await?;

    if !data.is_object() {
        return Err(Error::message("Invalid device code response"));
    }
    let string = |name: &str| data.get(name).and_then(Value::as_str).map(str::to_string);
    let interval = data.get("interval");
    let (Some(device_code), Some(user_code), Some(verification_uri), Some(expires_in)) = (
        string("device_code"),
        string("user_code"),
        string("verification_uri"),
        data.get("expires_in").and_then(Value::as_f64),
    ) else {
        return Err(Error::message("Invalid device code response fields"));
    };
    if interval.is_some_and(|interval| !interval.is_number()) {
        return Err(Error::message("Invalid device code response fields"));
    }

    Ok(DeviceCodeResponse {
        device_code,
        user_code,
        verification_uri: normalize_verification_uri(&verification_uri)?,
        interval: interval.and_then(Value::as_f64),
        expires_in,
    })
}

fn parse_device_token_response(raw: &Value) -> OAuthDeviceCodePollResult<GitHubUserToken> {
    if let Some(access_token) = raw.get("access_token").and_then(Value::as_str) {
        return OAuthDeviceCodePollResult::Complete(GitHubUserToken {
            access_token: access_token.to_string(),
            refresh_token: raw
                .get("refresh_token")
                .and_then(Value::as_str)
                .map(str::to_string),
            expires_in: raw.get("expires_in").and_then(Value::as_u64),
        });
    }

    if let Some(error) = raw.get("error").and_then(Value::as_str) {
        return match error {
            "authorization_pending" => OAuthDeviceCodePollResult::Pending,
            "slow_down" => OAuthDeviceCodePollResult::SlowDown {
                interval_seconds: raw.get("interval").and_then(Value::as_f64),
            },
            _ => {
                let suffix = raw
                    .get("error_description")
                    .and_then(Value::as_str)
                    .filter(|description| !description.is_empty())
                    .map(|description| format!(": {description}"))
                    .unwrap_or_default();
                OAuthDeviceCodePollResult::Failed {
                    message: format!("Device flow failed: {error}{suffix}"),
                }
            }
        };
    }

    OAuthDeviceCodePollResult::Failed {
        message: "Invalid device token response".to_string(),
    }
}

async fn poll_for_github_access_token(
    fetch: &OAuthFetch,
    domain: &str,
    device: &DeviceCodeResponse,
    signal: &CancellationToken,
) -> Result<GitHubUserToken> {
    let urls = get_urls(domain);
    let client_id = client_id();
    poll_oauth_device_code_flow(
        OAuthDeviceCodePollOptions {
            interval_seconds: device.interval,
            expires_in_seconds: Some(device.expires_in),
            wait_before_first_poll: true,
            signal: signal.clone(),
        },
        || async {
            let raw = fetch_json(
                fetch,
                form_post(
                    &urls.access_token_url,
                    &[
                        ("client_id", client_id.as_str()),
                        ("device_code", device.device_code.as_str()),
                        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ],
                ),
                signal,
            )
            .await?;
            Ok(parse_device_token_response(&raw))
        },
    )
    .await
}

/// Mint a Copilot API token from a GitHub user access token.
async fn refresh_github_copilot_access_token(
    fetch: &OAuthFetch,
    refresh_token: &str,
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
) -> std::result::Result<OAuthCredential, FetchJsonError> {
    let domain = enterprise_domain
        .filter(|domain| !domain.is_empty())
        .unwrap_or("github.com");
    let urls = get_urls(domain);

    let raw = fetch_json(
        fetch,
        with_copilot_headers(
            FetchRequest::get(&urls.copilot_token_url)
                .header("Accept", "application/json")
                .header("Authorization", format!("Bearer {refresh_token}")),
        ),
        signal,
    )
    .await?;

    if !raw.is_object() {
        return Err(FetchJsonError::Other(Error::message(
            "Invalid Copilot token response",
        )));
    }
    let (Some(token), Some(expires_at)) = (
        raw.get("token").and_then(Value::as_str),
        raw.get("expires_at").and_then(Value::as_f64),
    ) else {
        return Err(FetchJsonError::Other(Error::message(
            "Invalid Copilot token response fields",
        )));
    };

    let mut extra = Map::new();
    if let Some(domain) = enterprise_domain {
        extra.insert(ENTERPRISE_URL_KEY.to_string(), Value::from(domain));
    }
    Ok(OAuthCredential {
        refresh: refresh_token.to_string(),
        access: token.to_string(),
        expires: ((expires_at * 1000.0) as u64).saturating_sub(EXPIRY_SKEW_MS),
        extra,
    })
}

fn set_available_model_ids(credential: &mut OAuthCredential, ids: Vec<String>) {
    credential.extra.insert(
        AVAILABLE_MODEL_IDS_KEY.to_string(),
        Value::Array(ids.into_iter().map(Value::String).collect()),
    );
}

/// ai.rs divergence: persist the GitHub refresh token and access-token expiry.
fn set_github_user_token_fields(
    credential: &mut OAuthCredential,
    refresh_token: Option<&str>,
    access_expires: Option<u64>,
) {
    if let Some(refresh_token) = refresh_token {
        credential.extra.insert(
            GITHUB_REFRESH_TOKEN_KEY.to_string(),
            Value::from(refresh_token),
        );
    }
    if let Some(expires) = access_expires {
        credential
            .extra
            .insert(GITHUB_ACCESS_EXPIRES_KEY.to_string(), Value::from(expires));
    }
}

/// Skew-adjusted absolute expiry (Unix ms) of a GitHub user token.
fn github_access_expiry_ms(expires_in_seconds: u64) -> u64 {
    (now_millis() + expires_in_seconds * 1000).saturating_sub(EXPIRY_SKEW_MS)
}

/// ai.rs divergence: exchange a GitHub `ghr_` refresh token for a new user
/// access token with the device-flow client id (public clients send no
/// secret).
async fn refresh_github_user_token(
    fetch: &OAuthFetch,
    refresh_token: &str,
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
) -> Result<GitHubUserToken> {
    let domain = enterprise_domain
        .filter(|domain| !domain.is_empty())
        .unwrap_or("github.com");
    let client_id = client_id();
    let raw = fetch_json(
        fetch,
        form_post(
            &get_urls(domain).access_token_url,
            &[
                ("client_id", client_id.as_str()),
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
            ],
        ),
        signal,
    )
    .await?;
    match parse_device_token_response(&raw) {
        OAuthDeviceCodePollResult::Complete(token) => Ok(token),
        OAuthDeviceCodePollResult::Failed { message } => Err(Error::message(message)),
        OAuthDeviceCodePollResult::Pending | OAuthDeviceCodePollResult::SlowDown { .. } => Err(
            Error::message("Unexpected polling response while refreshing GitHub token"),
        ),
    }
}

/// `refreshGitHubCopilotToken()`, plus the `ghr_` renewal (see the module docs).
async fn refresh_github_copilot_credential(
    fetch: &OAuthFetch,
    credential: &OAuthCredential,
    signal: &CancellationToken,
) -> Result<OAuthCredential> {
    let enterprise_domain = copilot_enterprise_domain(credential);
    let domain = enterprise_domain.as_deref();
    let mut github_access = credential.refresh.clone();
    let mut github_refresh = credential
        .extra
        .get(GITHUB_REFRESH_TOKEN_KEY)
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut github_expires = credential
        .extra
        .get(GITHUB_ACCESS_EXPIRES_KEY)
        .and_then(Value::as_u64);

    // Proactive: the stored GitHub token has expired and a refresh token is at hand.
    if let Some(refresh) = github_refresh.clone()
        && github_expires.is_some_and(|expires| now_millis() >= expires)
    {
        let renewed = refresh_github_user_token(fetch, &refresh, domain, signal).await?;
        github_access = renewed.access_token;
        github_refresh = renewed.refresh_token.or(Some(refresh));
        github_expires = renewed.expires_in.map(github_access_expiry_ms);
    }

    // Reactive: GitHub rejected the token. Renew once and mint again.
    let mut credentials =
        match refresh_github_copilot_access_token(fetch, &github_access, domain, signal).await {
            Ok(credentials) => credentials,
            Err(FetchJsonError::Status {
                status: 401 | 403, ..
            }) if github_refresh.is_some() => {
                let refresh = github_refresh.clone().unwrap_or_default();
                let renewed = refresh_github_user_token(fetch, &refresh, domain, signal).await?;
                github_access = renewed.access_token;
                github_refresh = renewed.refresh_token.or(Some(refresh));
                github_expires = renewed.expires_in.map(github_access_expiry_ms);
                refresh_github_copilot_access_token(fetch, &github_access, domain, signal).await?
            }
            Err(error) => return Err(error.into()),
        };

    let models = fetch_github_copilot_models(
        fetch,
        &credentials.access,
        domain,
        signal,
        REFRESH_RETRY_POLICY,
    )
    .await?;
    set_github_user_token_fields(&mut credentials, github_refresh.as_deref(), github_expires);
    set_available_model_ids(&mut credentials, models.available_model_ids);
    Ok(credentials)
}

/// Enable a model for the user's GitHub Copilot account. Some models (like
/// Claude and Grok) need this before they can be used.
async fn enable_github_copilot_model(
    fetch: &OAuthFetch,
    token: &str,
    model_id: &str,
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
) -> Result<bool> {
    let base_url = get_github_copilot_base_url(Some(token), enterprise_domain);
    let request = with_copilot_headers(
        FetchRequest::post(format!("{base_url}/models/{model_id}/policy"))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {token}")),
    )
    .header("openai-intent", "chat-policy")
    .header("x-interaction-type", "chat-policy")
    .body(json!({ "state": "enabled" }).to_string());

    let response =
        match fetch_with_rate_limit_retry(fetch, request, signal, LOGIN_RETRY_POLICY).await {
            Ok(response) => response,
            Err(error) => {
                if signal.is_cancelled() {
                    return Err(error);
                }
                return Ok(false);
            }
        };
    if response.status == 429 {
        return Err(status_error(&response));
    }
    Ok(response.ok())
}

/// Enable the requested models and return the ids that succeeded. Policy
/// updates are best effort; exhausted rate limiting stops the batch.
async fn enable_github_copilot_models(
    fetch: &OAuthFetch,
    token: &str,
    model_ids: &[String],
    enterprise_domain: Option<&str>,
    signal: &CancellationToken,
) -> Result<Vec<String>> {
    let mut enabled_model_ids = Vec::new();
    for model_id in model_ids {
        match enable_github_copilot_model(fetch, token, model_id, enterprise_domain, signal).await {
            Ok(true) => enabled_model_ids.push(model_id.clone()),
            Ok(false) => {}
            Err(error) => {
                if signal.is_cancelled() {
                    return Err(error);
                }
                break;
            }
        }
    }
    Ok(enabled_model_ids)
}

async fn login_github_copilot_with(
    fetch: &OAuthFetch,
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential> {
    let input = interaction
        .prompt(AuthPrompt {
            signal: None,
            kind: AuthPromptKind::Text {
                message: "GitHub Enterprise URL/domain (blank for github.com)".to_string(),
                placeholder: Some("company.ghe.com".to_string()),
            },
        })
        .await?;
    if interaction.signal.is_cancelled() {
        return Err(Error::message("Login cancelled"));
    }

    let trimmed = input.trim();
    let enterprise_domain = normalize_domain(&input);
    if !trimmed.is_empty() && enterprise_domain.is_none() {
        return Err(Error::message("Invalid GitHub Enterprise URL/domain"));
    }
    let domain = enterprise_domain.as_deref().unwrap_or("github.com");
    let signal = &interaction.signal;

    let device = start_device_flow(fetch, domain, signal).await?;
    interaction.notify(AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: device.verification_uri.clone(),
        interval_seconds: device.interval.map(|interval| interval as u64),
        expires_in_seconds: Some(device.expires_in as u64),
    });

    let github = poll_for_github_access_token(fetch, domain, &device, signal).await?;
    let mut credentials = refresh_github_copilot_access_token(
        fetch,
        &github.access_token,
        enterprise_domain.as_deref(),
        signal,
    )
    .await?;
    let models = fetch_github_copilot_models(
        fetch,
        &credentials.access,
        enterprise_domain.as_deref(),
        signal,
        LOGIN_RETRY_POLICY,
    )
    .await?;
    let mut enabled_model_ids = Vec::new();
    if !models.policy_model_ids.is_empty() {
        interaction.notify(AuthEvent::Progress {
            message: "Enabling models...".to_string(),
        });
        enabled_model_ids = enable_github_copilot_models(
            fetch,
            &credentials.access,
            &models.policy_model_ids,
            enterprise_domain.as_deref(),
            signal,
        )
        .await?;
    }
    let mut seen = HashSet::new();
    let available_model_ids = models
        .available_model_ids
        .into_iter()
        .chain(enabled_model_ids)
        .filter(|id| seen.insert(id.clone()))
        .collect();
    set_github_user_token_fields(
        &mut credentials,
        github.refresh_token.as_deref(),
        github.expires_in.map(github_access_expiry_ms),
    );
    set_available_model_ids(&mut credentials, available_model_ids);
    Ok(credentials)
}

/// The credential's normalized enterprise domain (`enterpriseUrl`).
fn copilot_enterprise_domain(credential: &OAuthCredential) -> Option<String> {
    credential
        .extra
        .get(ENTERPRISE_URL_KEY)
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
        .and_then(normalize_domain)
}

/// `githubCopilotOAuth`. Holds the `fetch` it sends requests through.
#[derive(Clone)]
pub struct GitHubCopilotOAuth {
    fetch: OAuthFetch,
}

impl Default for GitHubCopilotOAuth {
    fn default() -> Self {
        Self::with_fetch(default_oauth_fetch())
    }
}

impl GitHubCopilotOAuth {
    pub fn with_fetch(fetch: OAuthFetch) -> Self {
        Self { fetch }
    }
}

/// `githubCopilotOAuth`.
pub fn github_copilot_oauth() -> Arc<dyn OAuthAuth> {
    Arc::new(GitHubCopilotOAuth::default())
}

#[async_trait]
impl OAuthAuth for GitHubCopilotOAuth {
    fn name(&self) -> &str {
        "GitHub Copilot"
    }

    fn is_subscription(&self) -> Option<bool> {
        Some(true)
    }

    async fn login(
        &self,
        interaction: ProviderAuthInteraction,
        _options: LoginOptions,
    ) -> Result<OAuthCredential> {
        login_github_copilot_with(&self.fetch, &interaction).await
    }

    async fn refresh(
        &self,
        credential: OAuthCredential,
        signal: CancellationToken,
    ) -> Result<OAuthCredential> {
        refresh_github_copilot_credential(&self.fetch, &credential, &signal).await
    }

    /// Derive the credential-specific proxy endpoint for each request.
    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth> {
        Ok(ModelAuth {
            api_key: Some(credential.access.clone()),
            base_url: Some(github_copilot_base_url_for_credential(credential)),
            ..Default::default()
        })
    }
}

/// The request base URL for a stored Copilot credential.
pub fn github_copilot_base_url_for_credential(credential: &OAuthCredential) -> String {
    get_github_copilot_base_url(
        Some(&credential.access),
        copilot_enterprise_domain(credential).as_deref(),
    )
}

/// Pre-1.0 entry point: run the device-code login with legacy callbacks.
pub async fn login_github_copilot(callbacks: OAuthLoginCallbacks) -> Result<OAuthCredential> {
    let interaction = callbacks_interaction(callbacks, "");
    login_github_copilot_with(&default_oauth_fetch(), &interaction).await
}

/// Pre-1.0 entry point: mint Copilot credentials from a GitHub token
/// (`refreshGitHubCopilotToken`).
pub async fn refresh_github_copilot_token(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<OAuthCredential> {
    let credential = OAuthCredential {
        refresh: refresh_token.to_string(),
        extra: enterprise_domain
            .map(|domain| {
                [(ENTERPRISE_URL_KEY.to_string(), Value::from(domain))]
                    .into_iter()
                    .collect()
            })
            .unwrap_or_default(),
        ..Default::default()
    };
    refresh_github_copilot_credential(
        &default_oauth_fetch(),
        &credential,
        &CancellationToken::new(),
    )
    .await
}

/// Pre-1.0 `modifyModels`: keep the account's available Copilot models and
/// point them at the credential's endpoint. Pi 1.0 splits this into the
/// provider's `filterModels` and `toAuth().baseUrl`; models of other
/// providers pass through unchanged.
pub fn modify_github_copilot_models(
    models: impl IntoIterator<Item = Model>,
    credential: &OAuthCredential,
) -> Vec<Model> {
    let base_url = github_copilot_base_url_for_credential(credential);
    let available: Option<HashSet<&str>> = credential
        .extra
        .get(AVAILABLE_MODEL_IDS_KEY)
        .and_then(Value::as_array)
        .and_then(|ids| ids.iter().map(Value::as_str).collect());
    models
        .into_iter()
        .filter_map(|mut model| {
            if model.provider == "github-copilot" {
                if available
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

#[cfg(test)]
mod tests;
