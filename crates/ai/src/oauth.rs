use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::models::get_models;
use crate::types::Model;
use crate::{Error, Result};

const GITHUB_COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const COPILOT_TOKEN_EXPIRY_SKEW_MS: u64 = 5 * 60 * 1000;
const DEFAULT_GITHUB_DOMAIN: &str = "github.com";
const DEFAULT_COPILOT_BASE_URL: &str = "https://api.individual.githubcopilot.com";
const CANCEL_MESSAGE: &str = "Login cancelled";
const TIMEOUT_MESSAGE: &str = "Device flow timed out";
const SLOW_DOWN_TIMEOUT_MESSAGE: &str = "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.";
const MINIMUM_INTERVAL_MS: u64 = 1000;
const DEFAULT_POLL_INTERVAL_SECONDS: u64 = 5;
const SLOW_DOWN_INTERVAL_INCREMENT_MS: u64 = 5000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthCredentials {
    pub refresh: String,
    pub access: String,
    pub expires: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enterprise_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthPrompt {
    pub message: String,
    pub placeholder: Option<String>,
    pub allow_empty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthDeviceCodeInfo {
    pub user_code: String,
    pub verification_uri: String,
    pub interval_seconds: Option<u64>,
    pub expires_in_seconds: Option<u64>,
}

pub type OAuthPromptFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;
pub type OAuthPromptCallback = Arc<dyn Fn(OAuthPrompt) -> OAuthPromptFuture + Send + Sync>;
pub type OAuthDeviceCodeCallback = Arc<dyn Fn(OAuthDeviceCodeInfo) + Send + Sync>;
pub type OAuthProgressCallback = Arc<dyn Fn(String) + Send + Sync>;

#[derive(Clone)]
pub struct OAuthLoginCallbacks {
    pub on_device_code: OAuthDeviceCodeCallback,
    pub on_prompt: OAuthPromptCallback,
    pub on_progress: Option<OAuthProgressCallback>,
    pub cancellation_token: Option<CancellationToken>,
}

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
        refresh_github_copilot_token(&credentials.refresh, credentials.enterprise_url.as_deref())
            .await
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OAuthDeviceCodePollResult<T> {
    Pending,
    SlowDown,
    Failed(String),
    Complete(T),
}

pub async fn poll_oauth_device_code_flow<T, F, Fut>(
    interval_seconds: Option<u64>,
    expires_in_seconds: Option<u64>,
    cancellation_token: Option<CancellationToken>,
    mut poll: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<OAuthDeviceCodePollResult<T>>>,
{
    let started = crate::utils::time::now_millis();
    let deadline = expires_in_seconds.map(|seconds| started + seconds.saturating_mul(1000));
    let mut interval_ms = interval_seconds
        .unwrap_or(DEFAULT_POLL_INTERVAL_SECONDS)
        .saturating_mul(1000)
        .max(MINIMUM_INTERVAL_MS);
    let mut slow_down_responses = 0;

    loop {
        if cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(Error::Provider(CANCEL_MESSAGE.to_string()));
        }
        if deadline.is_some_and(|deadline| crate::utils::time::now_millis() >= deadline) {
            return Err(timeout_error(slow_down_responses));
        }

        match poll().await? {
            OAuthDeviceCodePollResult::Complete(value) => return Ok(value),
            OAuthDeviceCodePollResult::Failed(message) => return Err(Error::Provider(message)),
            OAuthDeviceCodePollResult::Pending => {}
            OAuthDeviceCodePollResult::SlowDown => {
                slow_down_responses += 1;
                interval_ms = interval_ms
                    .saturating_add(SLOW_DOWN_INTERVAL_INCREMENT_MS)
                    .max(MINIMUM_INTERVAL_MS);
            }
        }

        let remaining_ms = deadline
            .map(|deadline| deadline.saturating_sub(crate::utils::time::now_millis()))
            .unwrap_or(interval_ms);
        if remaining_ms == 0 {
            return Err(timeout_error(slow_down_responses));
        }
        abortable_sleep(
            Duration::from_millis(interval_ms.min(remaining_ms)),
            cancellation_token.as_ref(),
        )
        .await?;
    }
}

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
        .and_then(|url| url.host_str().map(ToString::to_string))
}

pub fn get_github_copilot_base_url(token: Option<&str>, enterprise_domain: Option<&str>) -> String {
    if let Some(token) = token {
        if let Some(base_url) = get_base_url_from_token(token) {
            return base_url;
        }
    }
    if let Some(domain) = enterprise_domain.and_then(normalize_domain) {
        return format!("https://copilot-api.{domain}");
    }
    DEFAULT_COPILOT_BASE_URL.to_string()
}

pub async fn refresh_github_copilot_token(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<OAuthCredentials> {
    let domain = enterprise_domain
        .and_then(normalize_domain)
        .unwrap_or_else(|| DEFAULT_GITHUB_DOMAIN.to_string());
    let urls = GitHubCopilotUrls::new(&domain);
    let client = reqwest::Client::new();
    let response = client
        .get(urls.copilot_token_url)
        .headers(copilot_headers(Some(refresh_token))?)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(Error::ApiStatus { status, body });
    }

    let token = response.json::<CopilotTokenResponse>().await?;
    Ok(copilot_credentials_from_token(
        refresh_token,
        enterprise_domain.and_then(normalize_domain),
        token,
    ))
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

    let refresh_token =
        poll_for_github_access_token(&domain, device, callbacks.cancellation_token.clone()).await?;
    let credentials =
        refresh_github_copilot_token(&refresh_token, enterprise_domain.as_deref()).await?;

    if let Some(on_progress) = &callbacks.on_progress {
        on_progress("Enabling models...".to_string());
    }
    enable_all_github_copilot_models(&credentials.access, enterprise_domain.as_deref()).await;
    Ok(credentials)
}

pub fn modify_github_copilot_models(
    models: impl IntoIterator<Item = Model>,
    credentials: &OAuthCredentials,
) -> Vec<Model> {
    let domain = credentials
        .enterprise_url
        .as_deref()
        .and_then(normalize_domain);
    let base_url = get_github_copilot_base_url(Some(&credentials.access), domain.as_deref());
    models
        .into_iter()
        .map(|mut model| {
            if model.provider == "github-copilot" {
                model.base_url = base_url.clone();
            }
            model
        })
        .collect()
}

async fn start_github_device_flow(domain: &str) -> Result<DeviceCodeResponse> {
    let urls = GitHubCopilotUrls::new(domain);
    let client = reqwest::Client::new();
    let response = client
        .post(urls.device_code_url)
        .header("Accept", "application/json")
        .header("User-Agent", "GitHubCopilotChat/0.35.0")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(form_body(&[
            ("client_id", GITHUB_COPILOT_CLIENT_ID),
            ("scope", "read:user"),
        ]))
        .send()
        .await?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(Error::ApiStatus { status, body });
    }
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
) -> Result<String> {
    let urls = GitHubCopilotUrls::new(domain);
    let client = reqwest::Client::new();
    poll_oauth_device_code_flow(
        device.interval,
        Some(device.expires_in),
        cancellation_token,
        move || {
            let client = client.clone();
            let access_token_url = urls.access_token_url.clone();
            let device_code = device.device_code.clone();
            async move {
                let response = client
                    .post(access_token_url)
                    .header("Accept", "application/json")
                    .header("User-Agent", "GitHubCopilotChat/0.35.0")
                    .header("Content-Type", "application/x-www-form-urlencoded")
                    .body(form_body(&[
                        ("client_id", GITHUB_COPILOT_CLIENT_ID),
                        ("device_code", device_code.as_str()),
                        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ]))
                    .send()
                    .await?;
                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    return Err(Error::ApiStatus { status, body });
                }
                let raw = response.json::<DeviceTokenResponse>().await?;
                Ok(match raw {
                    DeviceTokenResponse::Success { access_token, .. } => {
                        OAuthDeviceCodePollResult::Complete(access_token)
                    }
                    DeviceTokenResponse::Error {
                        error,
                        error_description,
                    } => match error.as_str() {
                        "authorization_pending" => OAuthDeviceCodePollResult::Pending,
                        "slow_down" => OAuthDeviceCodePollResult::SlowDown,
                        _ => {
                            let suffix = error_description
                                .map(|description| format!(": {description}"))
                                .unwrap_or_default();
                            OAuthDeviceCodePollResult::Failed(format!(
                                "Device flow failed: {error}{suffix}"
                            ))
                        }
                    },
                })
            }
        },
    )
    .await
}

async fn enable_all_github_copilot_models(token: &str, enterprise_domain: Option<&str>) {
    let models = get_models("github-copilot");
    let futures = models.into_iter().map(|model| {
        let model_id = model.id;
        async move { enable_github_copilot_model(token, &model_id, enterprise_domain).await }
    });
    futures::future::join_all(futures).await;
}

async fn enable_github_copilot_model(
    token: &str,
    model_id: &str,
    enterprise_domain: Option<&str>,
) -> bool {
    let base_url = get_github_copilot_base_url(Some(token), enterprise_domain);
    let url = format!("{base_url}/models/{model_id}/policy");
    let client = reqwest::Client::new();
    client
        .post(url)
        .headers(match copilot_headers(Some(token)) {
            Ok(headers) => headers,
            Err(_) => return false,
        })
        .header("Content-Type", "application/json")
        .header("openai-intent", "chat-policy")
        .header("x-interaction-type", "chat-policy")
        .json(&serde_json::json!({ "state": "enabled" }))
        .send()
        .await
        .is_ok_and(|response| response.status().is_success())
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

fn copilot_credentials_from_token(
    refresh_token: &str,
    enterprise_domain: Option<String>,
    token: CopilotTokenResponse,
) -> OAuthCredentials {
    OAuthCredentials {
        refresh: refresh_token.to_string(),
        access: token.token,
        expires: token
            .expires_at
            .saturating_mul(1000)
            .saturating_sub(COPILOT_TOKEN_EXPIRY_SKEW_MS),
        enterprise_url: enterprise_domain,
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
        reqwest::header::HeaderValue::from_static("GitHubCopilotChat/0.35.0"),
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

fn form_body(fields: &[(&str, &str)]) -> String {
    fields
        .iter()
        .map(|(key, value)| format!("{}={}", form_encode(key), form_encode(value)))
        .collect::<Vec<_>>()
        .join("&")
}

fn form_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'*' => {
                encoded.push(byte as char)
            }
            b' ' => encoded.push('+'),
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

async fn abortable_sleep(
    duration: Duration,
    cancellation_token: Option<&CancellationToken>,
) -> Result<()> {
    if let Some(token) = cancellation_token {
        tokio::select! {
            _ = token.cancelled() => Err(Error::Provider(CANCEL_MESSAGE.to_string())),
            _ = tokio::time::sleep(duration) => Ok(()),
        }
    } else {
        tokio::time::sleep(duration).await;
        Ok(())
    }
}

fn timeout_error(slow_down_responses: u32) -> Error {
    Error::Provider(
        if slow_down_responses > 0 {
            SLOW_DOWN_TIMEOUT_MESSAGE
        } else {
            TIMEOUT_MESSAGE
        }
        .to_string(),
    )
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DeviceTokenResponse {
    Success {
        access_token: String,
        #[serde(rename = "token_type")]
        _token_type: Option<String>,
        #[serde(rename = "scope")]
        _scope: Option<String>,
    },
    Error {
        error: String,
        error_description: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct CopilotTokenResponse {
    token: String,
    expires_at: u64,
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::types::ModelCost;

    #[test]
    fn normalizes_enterprise_domains() {
        assert_eq!(normalize_domain(""), None);
        assert_eq!(
            normalize_domain("https://company.ghe.com/path"),
            Some("company.ghe.com".to_string())
        );
        assert_eq!(
            normalize_domain("company.ghe.com"),
            Some("company.ghe.com".to_string())
        );
        assert_eq!(normalize_domain("not a host"), None);
    }

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
            credentials.enterprise_url.as_deref(),
            Some("company.ghe.com")
        );
    }

    #[test]
    fn provider_updates_only_github_copilot_model_base_urls() {
        let credentials = OAuthCredentials {
            refresh: "refresh".to_string(),
            access: "tid=test;proxy-ep=proxy.enterprise.example.com;exp=1".to_string(),
            expires: 1,
            enterprise_url: None,
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

    #[tokio::test(flavor = "current_thread")]
    async fn device_code_poll_completes_immediately() {
        let value = poll_oauth_device_code_flow(None, Some(1), None, || async {
            Ok(OAuthDeviceCodePollResult::Complete("token"))
        })
        .await
        .unwrap();

        assert_eq!(value, "token");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn device_code_poll_returns_failed_message() {
        let error = poll_oauth_device_code_flow::<(), _, _>(None, Some(1), None, || async {
            Ok(OAuthDeviceCodePollResult::Failed("nope".to_string()))
        })
        .await
        .unwrap_err();

        assert_eq!(error.to_string(), "provider error: nope");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn login_callback_prompt_can_be_constructed() {
        let seen_prompt = Arc::new(Mutex::new(None));
        let callbacks = OAuthLoginCallbacks {
            on_device_code: Arc::new(|_| {}),
            on_prompt: {
                let seen_prompt = Arc::clone(&seen_prompt);
                Arc::new(move |prompt| {
                    *seen_prompt.lock().expect("prompt lock poisoned") = Some(prompt);
                    Box::pin(async { Err(Error::Provider("stop before network".to_string())) })
                })
            },
            on_progress: None,
            cancellation_token: None,
        };

        let error = login_github_copilot(callbacks).await.unwrap_err();

        assert_eq!(error.to_string(), "provider error: stop before network");
        let prompt = seen_prompt
            .lock()
            .expect("prompt lock poisoned")
            .clone()
            .expect("prompt");
        assert_eq!(
            prompt.message,
            "GitHub Enterprise URL/domain (blank for github.com)"
        );
        assert_eq!(prompt.placeholder.as_deref(), Some("company.ghe.com"));
        assert!(prompt.allow_empty);
    }

    #[test]
    fn provider_metadata_matches_upstream() {
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
}
