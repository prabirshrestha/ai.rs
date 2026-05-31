use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::{digest, rand};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::models::get_models;
use crate::types::Model;
use crate::{Error, Result};

const GITHUB_COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const ANTHROPIC_AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const ANTHROPIC_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const ANTHROPIC_CALLBACK_PORT: u16 = 53692;
const ANTHROPIC_CALLBACK_PATH: &str = "/callback";
const ANTHROPIC_REDIRECT_URI: &str = "http://localhost:53692/callback";
const ANTHROPIC_SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const OAUTH_TOKEN_EXPIRY_SKEW_MS: u64 = 5 * 60 * 1000;
const COPILOT_TOKEN_EXPIRY_SKEW_MS: u64 = 5 * 60 * 1000;
const ANTHROPIC_OAUTH_TOKEN_TIMEOUT_MS: u64 = 30_000;
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthAuthInfo {
    pub url: String,
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthSelectOption {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthSelectPrompt {
    pub message: String,
    pub options: Vec<OAuthSelectOption>,
}

pub type OAuthPromptFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;
pub type OAuthPromptCallback = Arc<dyn Fn(OAuthPrompt) -> OAuthPromptFuture + Send + Sync>;
pub type OAuthAuthCallback = Arc<dyn Fn(OAuthAuthInfo) + Send + Sync>;
pub type OAuthDeviceCodeCallback = Arc<dyn Fn(OAuthDeviceCodeInfo) + Send + Sync>;
pub type OAuthProgressCallback = Arc<dyn Fn(String) + Send + Sync>;
pub type OAuthManualCodeInputFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;
pub type OAuthManualCodeInputCallback = Arc<dyn Fn() -> OAuthManualCodeInputFuture + Send + Sync>;
pub type OAuthSelectFuture = Pin<Box<dyn Future<Output = Result<Option<String>>> + Send>>;
pub type OAuthSelectCallback = Arc<dyn Fn(OAuthSelectPrompt) -> OAuthSelectFuture + Send + Sync>;

#[derive(Clone)]
pub struct OAuthLoginCallbacks {
    pub on_auth: Option<OAuthAuthCallback>,
    pub on_device_code: OAuthDeviceCodeCallback,
    pub on_prompt: OAuthPromptCallback,
    pub on_progress: Option<OAuthProgressCallback>,
    pub on_manual_code_input: Option<OAuthManualCodeInputCallback>,
    pub on_select: Option<OAuthSelectCallback>,
    pub cancellation_token: Option<CancellationToken>,
}

pub type OAuthProviderId = String;
pub type OAuthProvider = Arc<dyn OAuthProviderInterface>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthProviderInfo {
    pub id: OAuthProviderId,
    pub name: String,
    pub available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthApiKey {
    pub new_credentials: OAuthCredentials,
    pub api_key: String,
}

#[async_trait]
pub trait OAuthProviderInterface: Send + Sync {
    fn id(&self) -> &'static str;
    fn name(&self) -> &'static str;
    fn uses_callback_server(&self) -> bool {
        false
    }
    async fn login(&self, callbacks: OAuthLoginCallbacks) -> Result<OAuthCredentials>;
    async fn refresh_token(&self, credentials: &OAuthCredentials) -> Result<OAuthCredentials>;
    fn get_api_key(&self, credentials: &OAuthCredentials) -> String;
    fn modify_models(&self, models: Vec<Model>, _credentials: &OAuthCredentials) -> Vec<Model> {
        models
    }
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
        refresh_github_copilot_token(&credentials.refresh, credentials.enterprise_url.as_deref())
            .await
    }

    fn get_api_key(&self, credentials: &OAuthCredentials) -> String {
        credentials.access.clone()
    }

    fn modify_models(&self, models: Vec<Model>, credentials: &OAuthCredentials) -> Vec<Model> {
        modify_github_copilot_models(models, credentials)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnthropicOAuthProvider;

impl AnthropicOAuthProvider {
    pub const fn id(self) -> &'static str {
        "anthropic"
    }

    pub const fn name(self) -> &'static str {
        "Anthropic (Claude Pro/Max)"
    }

    pub async fn login(self, callbacks: OAuthLoginCallbacks) -> Result<OAuthCredentials> {
        login_anthropic(callbacks).await
    }

    pub async fn refresh_token(self, credentials: &OAuthCredentials) -> Result<OAuthCredentials> {
        refresh_anthropic_token(&credentials.refresh).await
    }

    pub fn get_api_key(self, credentials: &OAuthCredentials) -> String {
        credentials.access.clone()
    }
}

pub fn anthropic_oauth_provider() -> AnthropicOAuthProvider {
    AnthropicOAuthProvider
}

#[async_trait]
impl OAuthProviderInterface for AnthropicOAuthProvider {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    fn name(&self) -> &'static str {
        "Anthropic (Claude Pro/Max)"
    }

    fn uses_callback_server(&self) -> bool {
        true
    }

    async fn login(&self, callbacks: OAuthLoginCallbacks) -> Result<OAuthCredentials> {
        login_anthropic(callbacks).await
    }

    async fn refresh_token(&self, credentials: &OAuthCredentials) -> Result<OAuthCredentials> {
        refresh_anthropic_token(&credentials.refresh).await
    }

    fn get_api_key(&self, credentials: &OAuthCredentials) -> String {
        credentials.access.clone()
    }
}

#[derive(Clone)]
struct RegisteredOAuthProvider {
    provider: OAuthProvider,
}

fn oauth_registry() -> &'static RwLock<Vec<RegisteredOAuthProvider>> {
    static REGISTRY: OnceLock<RwLock<Vec<RegisteredOAuthProvider>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(builtin_oauth_providers()))
}

fn builtin_oauth_provider(id: &str) -> Option<OAuthProvider> {
    match id {
        "anthropic" => Some(Arc::new(AnthropicOAuthProvider)),
        "github-copilot" => Some(Arc::new(GitHubCopilotOAuthProvider)),
        _ => None,
    }
}

fn builtin_oauth_providers() -> Vec<RegisteredOAuthProvider> {
    ["anthropic", "github-copilot"]
        .into_iter()
        .filter_map(|id| {
            builtin_oauth_provider(id).map(|provider| RegisteredOAuthProvider { provider })
        })
        .collect()
}

pub fn get_oauth_provider(id: &str) -> Option<OAuthProvider> {
    oauth_registry()
        .read()
        .expect("oauth registry poisoned")
        .iter()
        .find(|entry| entry.provider.id() == id)
        .map(|entry| entry.provider.clone())
}

pub fn register_oauth_provider(provider: OAuthProvider) {
    let id = provider.id();
    let mut registry = oauth_registry().write().expect("oauth registry poisoned");
    if let Some(existing) = registry.iter_mut().find(|entry| entry.provider.id() == id) {
        *existing = RegisteredOAuthProvider { provider };
    } else {
        registry.push(RegisteredOAuthProvider { provider });
    }
}

pub fn unregister_oauth_provider(id: &str) {
    let mut registry = oauth_registry().write().expect("oauth registry poisoned");
    if let Some(provider) = builtin_oauth_provider(id) {
        if let Some(existing) = registry.iter_mut().find(|entry| entry.provider.id() == id) {
            *existing = RegisteredOAuthProvider { provider };
        } else {
            registry.push(RegisteredOAuthProvider { provider });
        }
    } else {
        registry.retain(|entry| entry.provider.id() != id);
    }
}

pub fn reset_oauth_providers() {
    *oauth_registry().write().expect("oauth registry poisoned") = builtin_oauth_providers();
}

pub fn get_oauth_providers() -> Vec<OAuthProvider> {
    oauth_registry()
        .read()
        .expect("oauth registry poisoned")
        .iter()
        .map(|entry| entry.provider.clone())
        .collect()
}

pub fn get_oauth_provider_info_list() -> Vec<OAuthProviderInfo> {
    get_oauth_providers()
        .into_iter()
        .map(|provider| OAuthProviderInfo {
            id: provider.id().to_string(),
            name: provider.name().to_string(),
            available: true,
        })
        .collect()
}

pub async fn refresh_oauth_token(
    provider_id: &str,
    credentials: &OAuthCredentials,
) -> Result<OAuthCredentials> {
    let provider = get_oauth_provider(provider_id)
        .ok_or_else(|| Error::Provider(format!("Unknown OAuth provider: {provider_id}")))?;
    provider.refresh_token(credentials).await
}

pub async fn get_oauth_api_key(
    provider_id: &str,
    credentials: &HashMap<String, OAuthCredentials>,
) -> Result<Option<OAuthApiKey>> {
    let provider = get_oauth_provider(provider_id)
        .ok_or_else(|| Error::Provider(format!("Unknown OAuth provider: {provider_id}")))?;
    let Some(mut credentials) = credentials.get(provider_id).cloned() else {
        return Ok(None);
    };

    if crate::utils::time::now_millis() >= credentials.expires {
        credentials = provider.refresh_token(&credentials).await.map_err(|_| {
            Error::Provider(format!("Failed to refresh OAuth token for {provider_id}"))
        })?;
    }

    let api_key = provider.get_api_key(&credentials);
    Ok(Some(OAuthApiKey {
        new_credentials: credentials,
        api_key,
    }))
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

    let token = parse_copilot_token_response(response.json::<Value>().await?)?;
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

pub async fn login_anthropic(callbacks: OAuthLoginCallbacks) -> Result<OAuthCredentials> {
    let Some(on_auth) = callbacks.on_auth.clone() else {
        return Err(Error::Provider(
            "Anthropic OAuth login requires an on_auth callback".to_string(),
        ));
    };
    if callbacks
        .cancellation_token
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Err(Error::Provider(CANCEL_MESSAGE.to_string()));
    }

    let pkce = generate_pkce()?;
    let callback_server = start_anthropic_callback_server(
        pkce.verifier.clone(),
        callbacks.cancellation_token.clone(),
    )
    .await?;
    let AnthropicCallbackServer {
        receiver,
        shutdown,
        task,
    } = callback_server;

    let auth_url = anthropic_authorization_url(&pkce)?;
    on_auth(OAuthAuthInfo {
        url: auth_url,
        instructions: Some(
            "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."
                .to_string(),
        ),
    });

    let result = async {
        let mut authorization =
            if let Some(on_manual_code_input) = callbacks.on_manual_code_input.clone() {
                let manual_input = on_manual_code_input();
                tokio::pin!(manual_input);
                tokio::select! {
                    result = receiver => result.ok().flatten(),
                    input = &mut manual_input => {
                        Some(parse_authorization_input(&input?))
                    }
                }
            } else {
                receiver.await.ok().flatten()
            };

        if authorization
            .as_ref()
            .and_then(|input| input.code.as_ref())
            .is_none()
        {
            if callbacks
                .cancellation_token
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return Err(Error::Provider(CANCEL_MESSAGE.to_string()));
            }
            let input = (callbacks.on_prompt)(OAuthPrompt {
                message: "Paste the authorization code or full redirect URL:".to_string(),
                placeholder: Some(ANTHROPIC_REDIRECT_URI.to_string()),
                allow_empty: false,
            })
            .await?;
            authorization = Some(parse_authorization_input(&input));
        }

        let authorization = authorization
            .ok_or_else(|| Error::Provider("Missing authorization code".to_string()))?;
        if authorization
            .state
            .as_deref()
            .is_some_and(|state| state != pkce.verifier)
        {
            return Err(Error::Provider("OAuth state mismatch".to_string()));
        }
        let code = authorization
            .code
            .ok_or_else(|| Error::Provider("Missing authorization code".to_string()))?;
        let state = authorization.state.unwrap_or_else(|| pkce.verifier.clone());

        if callbacks
            .cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(Error::Provider(CANCEL_MESSAGE.to_string()));
        }
        if let Some(on_progress) = &callbacks.on_progress {
            on_progress("Exchanging authorization code for tokens...".to_string());
        }
        exchange_anthropic_authorization_code(&code, &state, &pkce.verifier, ANTHROPIC_REDIRECT_URI)
            .await
    }
    .await;

    shutdown.cancel();
    let _ = task.await;
    result
}

pub async fn exchange_anthropic_authorization_code(
    code: &str,
    state: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<OAuthCredentials> {
    let client = reqwest::Client::new();
    exchange_anthropic_authorization_code_at(
        &client,
        ANTHROPIC_TOKEN_URL,
        code,
        state,
        verifier,
        redirect_uri,
    )
    .await
}

pub async fn refresh_anthropic_token(refresh_token: &str) -> Result<OAuthCredentials> {
    let client = reqwest::Client::new();
    refresh_anthropic_token_at(&client, ANTHROPIC_TOKEN_URL, refresh_token).await
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

#[derive(Debug, Clone)]
struct Pkce {
    verifier: String,
    challenge: String,
}

fn generate_pkce() -> Result<Pkce> {
    let rng = rand::SystemRandom::new();
    let mut verifier_bytes = [0_u8; 32];
    rand::SecureRandom::fill(&rng, &mut verifier_bytes)
        .map_err(|_| Error::Provider("Failed to generate PKCE verifier".to_string()))?;
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);
    let challenge_hash = digest::digest(&digest::SHA256, verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(challenge_hash.as_ref());
    Ok(Pkce {
        verifier,
        challenge,
    })
}

fn anthropic_authorization_url(pkce: &Pkce) -> Result<String> {
    let mut url = reqwest::Url::parse(ANTHROPIC_AUTHORIZE_URL)
        .map_err(|error| Error::Provider(format!("Invalid Anthropic authorize URL: {error}")))?;
    url.query_pairs_mut()
        .append_pair("code", "true")
        .append_pair("client_id", ANTHROPIC_CLIENT_ID)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", ANTHROPIC_REDIRECT_URI)
        .append_pair("scope", ANTHROPIC_SCOPES)
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", &pkce.verifier);
    Ok(url.to_string())
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct AuthorizationInput {
    code: Option<String>,
    state: Option<String>,
}

fn parse_authorization_input(input: &str) -> AuthorizationInput {
    let value = input.trim();
    if value.is_empty() {
        return AuthorizationInput::default();
    }

    if let Ok(url) = reqwest::Url::parse(value) {
        return AuthorizationInput {
            code: url
                .query_pairs()
                .find_map(|(key, value)| (key == "code").then(|| value.into_owned())),
            state: url
                .query_pairs()
                .find_map(|(key, value)| (key == "state").then(|| value.into_owned())),
        };
    }

    if let Some((code, state)) = value.split_once('#') {
        return AuthorizationInput {
            code: (!code.is_empty()).then(|| code.to_string()),
            state: (!state.is_empty()).then(|| state.to_string()),
        };
    }

    if value.contains("code=") {
        let value = value.strip_prefix('?').unwrap_or(value);
        return AuthorizationInput {
            code: query_param(value, "code"),
            state: query_param(value, "state"),
        };
    }

    AuthorizationInput {
        code: Some(value.to_string()),
        state: None,
    }
}

fn query_param(query: &str, name: &str) -> Option<String> {
    reqwest::Url::parse(&format!("http://localhost/?{query}"))
        .ok()
        .and_then(|url| {
            url.query_pairs()
                .find_map(|(key, value)| (key == name).then(|| value.into_owned()))
        })
        .filter(|value| !value.is_empty())
}

struct AnthropicCallbackServer {
    receiver: oneshot::Receiver<Option<AuthorizationInput>>,
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

async fn start_anthropic_callback_server(
    expected_state: String,
    external_cancellation: Option<CancellationToken>,
) -> Result<AnthropicCallbackServer> {
    let host = std::env::var("PI_OAUTH_CALLBACK_HOST")
        .ok()
        .filter(|host| !host.trim().is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let listener = TcpListener::bind((host.as_str(), ANTHROPIC_CALLBACK_PORT)).await?;
    let (sender, receiver) = oneshot::channel();
    let shutdown = CancellationToken::new();
    let task_shutdown = shutdown.clone();
    let task = tokio::spawn(async move {
        run_anthropic_callback_server(
            listener,
            expected_state,
            sender,
            task_shutdown,
            external_cancellation,
        )
        .await;
    });

    Ok(AnthropicCallbackServer {
        receiver,
        shutdown,
        task,
    })
}

async fn run_anthropic_callback_server(
    listener: TcpListener,
    expected_state: String,
    sender: oneshot::Sender<Option<AuthorizationInput>>,
    shutdown: CancellationToken,
    external_cancellation: Option<CancellationToken>,
) {
    let mut sender = Some(sender);
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = async {
                if let Some(token) = &external_cancellation {
                    token.cancelled().await;
                } else {
                    futures::future::pending::<()>().await;
                }
            } => {
                if let Some(sender) = sender.take() {
                    let _ = sender.send(None);
                }
                break;
            }
            accepted = listener.accept() => {
                let Ok((mut stream, _addr)) = accepted else {
                    continue;
                };
                let result = handle_anthropic_callback_request(&mut stream, &expected_state).await;
                if let Some(input) = result {
                    if let Some(sender) = sender.take() {
                        let _ = sender.send(Some(input));
                    }
                    break;
                }
            }
        }
    }
}

async fn handle_anthropic_callback_request(
    stream: &mut tokio::net::TcpStream,
    expected_state: &str,
) -> Option<AuthorizationInput> {
    let mut buffer = vec![0_u8; 8192];
    let Ok(bytes_read) = stream.read(&mut buffer).await else {
        return None;
    };
    let request = String::from_utf8_lossy(&buffer[..bytes_read]);
    let request_target = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");
    let url = reqwest::Url::parse(&format!("http://localhost{request_target}")).ok();
    let Some(url) = url else {
        let _ = write_oauth_response(
            stream,
            400,
            oauth_error_html("Invalid callback request.", None),
        )
        .await;
        return None;
    };

    if url.path() != ANTHROPIC_CALLBACK_PATH {
        let _ = write_oauth_response(
            stream,
            404,
            oauth_error_html("Callback route not found.", None),
        )
        .await;
        return None;
    }
    if let Some(error) = url
        .query_pairs()
        .find_map(|(key, value)| (key == "error").then(|| value.into_owned()))
    {
        let _ = write_oauth_response(
            stream,
            400,
            oauth_error_html(
                "Anthropic authentication did not complete.",
                Some(&format!("Error: {error}")),
            ),
        )
        .await;
        return None;
    }

    let code = url
        .query_pairs()
        .find_map(|(key, value)| (key == "code").then(|| value.into_owned()));
    let state = url
        .query_pairs()
        .find_map(|(key, value)| (key == "state").then(|| value.into_owned()));
    let Some(code) = code else {
        let _ = write_oauth_response(
            stream,
            400,
            oauth_error_html("Missing code or state parameter.", None),
        )
        .await;
        return None;
    };
    let Some(state) = state else {
        let _ = write_oauth_response(
            stream,
            400,
            oauth_error_html("Missing code or state parameter.", None),
        )
        .await;
        return None;
    };
    if state != expected_state {
        let _ = write_oauth_response(stream, 400, oauth_error_html("State mismatch.", None)).await;
        return None;
    }

    let _ = write_oauth_response(
        stream,
        200,
        oauth_success_html("Anthropic authentication completed. You can close this window."),
    )
    .await;
    Some(AuthorizationInput {
        code: Some(code),
        state: Some(state),
    })
}

async fn write_oauth_response(
    stream: &mut tokio::net::TcpStream,
    status: u16,
    body: String,
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await
}

fn oauth_success_html(message: &str) -> String {
    oauth_page(
        "Authentication successful",
        "Authentication successful",
        message,
        None,
    )
}

fn oauth_error_html(message: &str, details: Option<&str>) -> String {
    oauth_page(
        "Authentication failed",
        "Authentication failed",
        message,
        details,
    )
}

fn oauth_page(title: &str, heading: &str, message: &str, details: Option<&str>) -> String {
    let details_html = details
        .map(|details| format!("<div class=\"details\">{}</div>", escape_html(details)))
        .unwrap_or_default();
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{}</title></head><body><main><h1>{}</h1><p>{}</p>{}</main></body></html>",
        escape_html(title),
        escape_html(heading),
        escape_html(message),
        details_html
    )
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
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
                Ok(parse_device_token_response(response.json::<Value>().await?))
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

fn parse_device_token_response(raw: Value) -> OAuthDeviceCodePollResult<String> {
    let Some(object) = raw.as_object() else {
        return OAuthDeviceCodePollResult::Failed("Invalid device token response".to_string());
    };

    if let Some(access_token) = object.get("access_token").and_then(Value::as_str) {
        return OAuthDeviceCodePollResult::Complete(access_token.to_string());
    }

    let Some(error) = object.get("error").and_then(Value::as_str) else {
        return OAuthDeviceCodePollResult::Failed("Invalid device token response".to_string());
    };

    match error {
        "authorization_pending" => OAuthDeviceCodePollResult::Pending,
        "slow_down" => OAuthDeviceCodePollResult::SlowDown,
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

async fn exchange_anthropic_authorization_code_at(
    client: &reqwest::Client,
    token_url: &str,
    code: &str,
    state: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<OAuthCredentials> {
    let response = client
        .post(token_url)
        .header("Accept", "application/json")
        .timeout(Duration::from_millis(ANTHROPIC_OAUTH_TOKEN_TIMEOUT_MS))
        .json(&serde_json::json!({
            "grant_type": "authorization_code",
            "client_id": ANTHROPIC_CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }))
        .send()
        .await?;
    anthropic_credentials_from_response(response).await
}

async fn refresh_anthropic_token_at(
    client: &reqwest::Client,
    token_url: &str,
    refresh_token: &str,
) -> Result<OAuthCredentials> {
    let response = client
        .post(token_url)
        .header("Accept", "application/json")
        .timeout(Duration::from_millis(ANTHROPIC_OAUTH_TOKEN_TIMEOUT_MS))
        .json(&serde_json::json!({
            "grant_type": "refresh_token",
            "client_id": ANTHROPIC_CLIENT_ID,
            "refresh_token": refresh_token,
        }))
        .send()
        .await?;
    anthropic_credentials_from_response(response).await
}

async fn anthropic_credentials_from_response(
    response: reqwest::Response,
) -> Result<OAuthCredentials> {
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(Error::ApiStatus { status, body });
    }

    let token = response.json::<AnthropicTokenResponse>().await?;
    Ok(anthropic_credentials_from_token(token))
}

fn anthropic_credentials_from_token(token: AnthropicTokenResponse) -> OAuthCredentials {
    OAuthCredentials {
        refresh: token.refresh_token,
        access: token.access_token,
        expires: crate::utils::time::now_millis()
            .saturating_add(token.expires_in.saturating_mul(1000))
            .saturating_sub(OAUTH_TOKEN_EXPIRY_SKEW_MS),
        enterprise_url: None,
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

#[derive(Debug)]
struct CopilotTokenResponse {
    token: String,
    expires_at: u64,
}

#[derive(Debug, Deserialize)]
struct AnthropicTokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
    #[serde(rename = "scope")]
    _scope: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::types::ModelCost;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct TestOAuthProvider;

    #[async_trait::async_trait]
    impl OAuthProviderInterface for TestOAuthProvider {
        fn id(&self) -> &'static str {
            "test-oauth"
        }

        fn name(&self) -> &'static str {
            "Test OAuth"
        }

        async fn login(&self, _callbacks: OAuthLoginCallbacks) -> Result<OAuthCredentials> {
            Ok(OAuthCredentials {
                refresh: "login-refresh".to_string(),
                access: "login-access".to_string(),
                expires: crate::utils::time::now_millis().saturating_add(60_000),
                enterprise_url: None,
            })
        }

        async fn refresh_token(&self, credentials: &OAuthCredentials) -> Result<OAuthCredentials> {
            Ok(OAuthCredentials {
                refresh: credentials.refresh.clone(),
                access: format!("{}-refreshed", credentials.access),
                expires: crate::utils::time::now_millis().saturating_add(60_000),
                enterprise_url: credentials.enterprise_url.clone(),
            })
        }

        fn get_api_key(&self, credentials: &OAuthCredentials) -> String {
            credentials.access.clone()
        }
    }

    #[tokio::test]
    async fn oauth_registry_supports_custom_providers_and_api_keys() {
        register_oauth_provider(Arc::new(TestOAuthProvider));

        let provider = get_oauth_provider("test-oauth").expect("registered provider");
        assert_eq!(provider.id(), "test-oauth");
        assert_eq!(provider.name(), "Test OAuth");
        assert!(
            get_oauth_provider_info_list()
                .iter()
                .any(|info| info.id == "test-oauth" && info.available)
        );

        let credentials = OAuthCredentials {
            refresh: "refresh".to_string(),
            access: "access".to_string(),
            expires: 0,
            enterprise_url: None,
        };
        let refreshed = refresh_oauth_token("test-oauth", &credentials)
            .await
            .expect("refreshed credentials");
        assert_eq!(refreshed.access, "access-refreshed");

        let mut credential_map = HashMap::new();
        credential_map.insert(
            "test-oauth".to_string(),
            OAuthCredentials {
                refresh: "refresh".to_string(),
                access: "current-access".to_string(),
                expires: crate::utils::time::now_millis().saturating_add(60_000),
                enterprise_url: None,
            },
        );
        let api_key = get_oauth_api_key("test-oauth", &credential_map)
            .await
            .expect("api key result")
            .expect("api key");
        assert_eq!(api_key.api_key, "current-access");
        assert_eq!(api_key.new_credentials.access, "current-access");

        unregister_oauth_provider("test-oauth");
        assert!(get_oauth_provider("test-oauth").is_none());
    }

    #[test]
    fn oauth_registry_exposes_focused_builtins() {
        let copilot = get_oauth_provider("github-copilot").expect("copilot provider");
        assert_eq!(copilot.id(), "github-copilot");
        assert_eq!(copilot.name(), "GitHub Copilot");
        assert!(!copilot.uses_callback_server());

        let anthropic = get_oauth_provider("anthropic").expect("anthropic provider");
        assert_eq!(anthropic.id(), "anthropic");
        assert!(anthropic.uses_callback_server());
    }

    #[test]
    fn parses_anthropic_authorization_inputs() {
        assert_eq!(parse_authorization_input(""), AuthorizationInput::default());
        assert_eq!(
            parse_authorization_input("http://localhost:53692/callback?code=abc&state=verifier"),
            AuthorizationInput {
                code: Some("abc".to_string()),
                state: Some("verifier".to_string())
            }
        );
        assert_eq!(
            parse_authorization_input("abc#verifier"),
            AuthorizationInput {
                code: Some("abc".to_string()),
                state: Some("verifier".to_string())
            }
        );
        assert_eq!(
            parse_authorization_input("code=abc&state=verifier"),
            AuthorizationInput {
                code: Some("abc".to_string()),
                state: Some("verifier".to_string())
            }
        );
        assert_eq!(
            parse_authorization_input("?code=abc&state=verifier"),
            AuthorizationInput {
                code: Some("abc".to_string()),
                state: Some("verifier".to_string())
            }
        );
        assert_eq!(
            parse_authorization_input("code=abc%2Bdef&state=verifier%2Bvalue"),
            AuthorizationInput {
                code: Some("abc+def".to_string()),
                state: Some("verifier+value".to_string())
            }
        );
        assert_eq!(
            parse_authorization_input("abc"),
            AuthorizationInput {
                code: Some("abc".to_string()),
                state: None
            }
        );
    }

    #[test]
    fn generates_anthropic_pkce_authorization_url() {
        let pkce = generate_pkce().expect("pkce");
        assert!(!pkce.verifier.contains('='));
        assert!(!pkce.challenge.contains('='));

        let url = anthropic_authorization_url(&pkce).expect("auth url");
        let parsed = reqwest::Url::parse(&url).expect("parsed url");
        let query = parsed
            .query_pairs()
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect::<HashMap<_, _>>();
        assert_eq!(
            parsed.as_str().split('?').next(),
            Some(ANTHROPIC_AUTHORIZE_URL)
        );
        assert_eq!(
            query.get("client_id").map(String::as_str),
            Some(ANTHROPIC_CLIENT_ID)
        );
        assert_eq!(
            query.get("redirect_uri").map(String::as_str),
            Some(ANTHROPIC_REDIRECT_URI)
        );
        assert_eq!(
            query.get("code_challenge").map(String::as_str),
            Some(pkce.challenge.as_str())
        );
        assert_eq!(
            query.get("state").map(String::as_str),
            Some(pkce.verifier.as_str())
        );
    }

    #[tokio::test]
    async fn anthropic_provider_login_requires_auth_callback() {
        let callbacks = OAuthLoginCallbacks {
            on_auth: None,
            on_device_code: Arc::new(|_| {}),
            on_prompt: Arc::new(|_| Box::pin(async { Ok("unused".to_string()) })),
            on_progress: None,
            on_manual_code_input: None,
            on_select: None,
            cancellation_token: None,
        };

        let error = anthropic_oauth_provider()
            .login(callbacks)
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "provider error: Anthropic OAuth login requires an on_auth callback"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn anthropic_login_cancellation_does_not_fall_back_to_prompt() {
        let cancellation_token = CancellationToken::new();
        let prompt_calls = Arc::new(Mutex::new(0_usize));
        let callbacks = OAuthLoginCallbacks {
            on_auth: Some({
                let cancellation_token = cancellation_token.clone();
                Arc::new(move |_| cancellation_token.cancel())
            }),
            on_device_code: Arc::new(|_| {}),
            on_prompt: {
                let prompt_calls = Arc::clone(&prompt_calls);
                Arc::new(move |_| {
                    *prompt_calls.lock().expect("prompt lock poisoned") += 1;
                    Box::pin(async { Err(Error::Provider("prompt should not run".to_string())) })
                })
            },
            on_progress: None,
            on_manual_code_input: None,
            on_select: None,
            cancellation_token: Some(cancellation_token),
        };

        let error = tokio::time::timeout(Duration::from_secs(2), login_anthropic(callbacks))
            .await
            .expect("login should observe cancellation")
            .unwrap_err();

        assert_eq!(error.to_string(), "provider error: Login cancelled");
        assert_eq!(*prompt_calls.lock().expect("prompt lock poisoned"), 0);
    }

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

    #[test]
    fn parses_device_token_response() {
        assert_eq!(
            parse_device_token_response(serde_json::json!({ "access_token": "ghu_refresh" })),
            OAuthDeviceCodePollResult::Complete("ghu_refresh".to_string())
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
                "error_description": "slow down"
            })),
            OAuthDeviceCodePollResult::SlowDown
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
    async fn device_code_poll_reports_slow_down_timeout() {
        let attempts = Arc::new(Mutex::new(0_u32));
        let error = poll_oauth_device_code_flow::<(), _, _>(Some(1), Some(1), None, {
            let attempts = Arc::clone(&attempts);
            move || {
                let attempts = Arc::clone(&attempts);
                async move {
                    *attempts.lock().expect("attempt lock poisoned") += 1;
                    Ok(OAuthDeviceCodePollResult::SlowDown)
                }
            }
        })
        .await
        .unwrap_err();

        assert_eq!(*attempts.lock().expect("attempt lock poisoned"), 1);
        assert!(
            error
                .to_string()
                .contains("Device flow timed out after one or more slow_down responses"),
            "{error}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn device_code_poll_cancels_in_flight_wait() {
        let token = CancellationToken::new();
        let attempts = Arc::new(Mutex::new(0_u32));
        let polled = Arc::new(tokio::sync::Notify::new());
        let poll =
            poll_oauth_device_code_flow::<(), _, _>(Some(5), Some(30), Some(token.clone()), {
                let attempts = Arc::clone(&attempts);
                let polled = Arc::clone(&polled);
                move || {
                    let attempts = Arc::clone(&attempts);
                    let polled = Arc::clone(&polled);
                    async move {
                        *attempts.lock().expect("attempt lock poisoned") += 1;
                        polled.notify_one();
                        Ok(OAuthDeviceCodePollResult::Pending)
                    }
                }
            });
        tokio::pin!(poll);

        tokio::select! {
            _ = polled.notified() => {}
            result = &mut poll => panic!("poll completed before cancellation: {result:?}"),
        }

        token.cancel();
        let error = poll.await.unwrap_err();

        assert_eq!(*attempts.lock().expect("attempt lock poisoned"), 1);
        assert_eq!(error.to_string(), "provider error: Login cancelled");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn login_callback_prompt_can_be_constructed() {
        let seen_prompt = Arc::new(Mutex::new(None));
        let callbacks = OAuthLoginCallbacks {
            on_auth: None,
            on_device_code: Arc::new(|_| {}),
            on_prompt: {
                let seen_prompt = Arc::clone(&seen_prompt);
                Arc::new(move |prompt| {
                    *seen_prompt.lock().expect("prompt lock poisoned") = Some(prompt);
                    Box::pin(async { Err(Error::Provider("stop before network".to_string())) })
                })
            },
            on_progress: None,
            on_manual_code_input: None,
            on_select: None,
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
    fn provider_metadata_matches_expected_shape() {
        let provider = github_copilot_oauth_provider();
        assert_eq!(provider.id(), "github-copilot");
        assert_eq!(provider.name(), "GitHub Copilot");
    }

    #[test]
    fn anthropic_provider_metadata_matches_expected_shape() {
        let provider = anthropic_oauth_provider();
        assert_eq!(provider.id(), "anthropic");
        assert_eq!(provider.name(), "Anthropic (Claude Pro/Max)");
    }

    #[tokio::test]
    async fn refresh_anthropic_token_omits_scope() {
        let captured_body = Arc::new(Mutex::new(None));
        let token_url = spawn_anthropic_token_server(Arc::clone(&captured_body)).await;
        let client = reqwest::Client::new();

        let credentials = refresh_anthropic_token_at(&client, &token_url, "refresh-token")
            .await
            .unwrap();

        assert_eq!(credentials.access, "access-token");
        assert_eq!(credentials.refresh, "new-refresh-token");
        let body = captured_body
            .lock()
            .expect("captured body lock poisoned")
            .clone()
            .expect("captured request body");
        assert_eq!(body["grant_type"], "refresh_token");
        assert_eq!(body["client_id"], ANTHROPIC_CLIENT_ID);
        assert_eq!(body["refresh_token"], "refresh-token");
        assert!(body.get("scope").is_none());
    }

    #[tokio::test]
    async fn anthropic_authorization_exchange_preserves_redirect_uri() {
        let captured_body = Arc::new(Mutex::new(None));
        let token_url = spawn_anthropic_token_server(Arc::clone(&captured_body)).await;
        let client = reqwest::Client::new();

        let credentials = exchange_anthropic_authorization_code_at(
            &client,
            &token_url,
            "manual-code",
            "state-value",
            "verifier-value",
            "http://localhost:53692/callback",
        )
        .await
        .unwrap();

        assert_eq!(credentials.access, "access-token");
        assert_eq!(credentials.refresh, "new-refresh-token");
        let body = captured_body
            .lock()
            .expect("captured body lock poisoned")
            .clone()
            .expect("captured request body");
        assert_eq!(body["grant_type"], "authorization_code");
        assert_eq!(body["client_id"], ANTHROPIC_CLIENT_ID);
        assert_eq!(body["code"], "manual-code");
        assert_eq!(body["state"], "state-value");
        assert_eq!(body["code_verifier"], "verifier-value");
        assert_eq!(body["redirect_uri"], "http://localhost:53692/callback");
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

    async fn spawn_anthropic_token_server(
        captured_body: Arc<Mutex<Option<serde_json::Value>>>,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut socket).await;
            let body = request
                .split_once("\r\n\r\n")
                .map(|(_, body)| body)
                .unwrap_or_default();
            *captured_body.lock().expect("captured body lock poisoned") =
                Some(serde_json::from_str(body).unwrap());
            let response_body = serde_json::json!({
                "access_token": "access-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
                "scope": "ignored"
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{addr}/oauth/token")
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buffer = [0u8; 1024];
        loop {
            let read = socket.read(&mut buffer).await.unwrap();
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..read]);
            if http_request_complete(&bytes) {
                break;
            }
        }
        String::from_utf8(bytes).unwrap()
    }

    fn http_request_complete(bytes: &[u8]) -> bool {
        let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
            return false;
        };
        let headers = String::from_utf8_lossy(&bytes[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        bytes.len() >= header_end + 4 + content_length
    }
}
