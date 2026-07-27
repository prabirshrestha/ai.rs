use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::types::Model;
use crate::{Error, Result};

mod anthropic;
mod github_copilot;

pub use anthropic::*;
pub use github_copilot::*;
pub(crate) use github_copilot::{get_github_copilot_base_url, github_copilot_enterprise_domain};

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
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
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

impl OAuthLoginCallbacks {
    pub fn builder() -> OAuthLoginCallbacksBuilder {
        OAuthLoginCallbacksBuilder::default()
    }
}

#[derive(Default)]
pub struct OAuthLoginCallbacksBuilder {
    on_auth: Option<OAuthAuthCallback>,
    on_device_code: Option<OAuthDeviceCodeCallback>,
    on_prompt: Option<OAuthPromptCallback>,
    on_progress: Option<OAuthProgressCallback>,
    on_manual_code_input: Option<OAuthManualCodeInputCallback>,
    on_select: Option<OAuthSelectCallback>,
    cancellation_token: Option<CancellationToken>,
}

impl OAuthLoginCallbacksBuilder {
    pub fn on_auth<F>(mut self, callback: F) -> Self
    where
        F: Fn(OAuthAuthInfo) + Send + Sync + 'static,
    {
        self.on_auth = Some(Arc::new(callback));
        self
    }

    pub fn on_device_code<F>(mut self, callback: F) -> Self
    where
        F: Fn(OAuthDeviceCodeInfo) + Send + Sync + 'static,
    {
        self.on_device_code = Some(Arc::new(callback));
        self
    }

    pub fn on_prompt<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(OAuthPrompt) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        self.on_prompt = Some(Arc::new(move |prompt| Box::pin(callback(prompt))));
        self
    }

    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        self.on_progress = Some(Arc::new(callback));
        self
    }

    pub fn on_manual_code_input<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        self.on_manual_code_input = Some(Arc::new(move || Box::pin(callback())));
        self
    }

    pub fn on_select<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(OAuthSelectPrompt) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        self.on_select = Some(Arc::new(move |prompt| Box::pin(callback(prompt))));
        self
    }

    pub fn cancellation_token(mut self, cancellation_token: CancellationToken) -> Self {
        self.cancellation_token = Some(cancellation_token);
        self
    }

    pub fn build(self) -> OAuthLoginCallbacks {
        OAuthLoginCallbacks {
            on_auth: self.on_auth,
            on_device_code: self.on_device_code.unwrap_or_else(|| Arc::new(|_| {})),
            on_prompt: self
                .on_prompt
                .unwrap_or_else(|| Arc::new(|_| Box::pin(async { Ok(String::new()) }))),
            on_progress: self.on_progress,
            on_manual_code_input: self.on_manual_code_input,
            on_select: self.on_select,
            cancellation_token: self.cancellation_token,
        }
    }
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
    SlowDown { interval_seconds: Option<u64> },
    Failed(String),
    Complete(T),
}

pub async fn poll_oauth_device_code_flow<T, F, Fut>(
    interval_seconds: Option<u64>,
    expires_in_seconds: Option<u64>,
    cancellation_token: Option<CancellationToken>,
    poll: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<OAuthDeviceCodePollResult<T>>>,
{
    poll_oauth_device_code_flow_inner(
        interval_seconds,
        expires_in_seconds,
        false,
        cancellation_token,
        poll,
    )
    .await
}

async fn poll_oauth_device_code_flow_inner<T, F, Fut>(
    interval_seconds: Option<u64>,
    expires_in_seconds: Option<u64>,
    wait_before_first_poll: bool,
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
    if wait_before_first_poll {
        let remaining_ms = deadline
            .map(|deadline| deadline.saturating_sub(crate::utils::time::now_millis()))
            .unwrap_or(interval_ms);
        if remaining_ms > 0 {
            abortable_sleep(
                Duration::from_millis(interval_ms.min(remaining_ms)),
                cancellation_token.as_ref(),
            )
            .await?;
        }
    }

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
            OAuthDeviceCodePollResult::SlowDown { interval_seconds } => {
                slow_down_responses += 1;
                interval_ms = interval_seconds
                    .filter(|seconds| *seconds > 0)
                    .map(|seconds| seconds.saturating_mul(1000).max(MINIMUM_INTERVAL_MS))
                    .unwrap_or_else(|| {
                        interval_ms
                            .saturating_add(SLOW_DOWN_INTERVAL_INCREMENT_MS)
                            .max(MINIMUM_INTERVAL_MS)
                    });
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

/// Returns `response` unchanged when its status is a success, otherwise an
/// [`Error::ApiStatus`] capturing the status code and response body. Shared by
/// every OAuth HTTP call in this module so the error mapping stays consistent.
async fn error_for_status(response: reqwest::Response) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(Error::ApiStatus { status, body })
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

#[cfg(test)]
pub(crate) mod test_support {
    use tokio::io::AsyncReadExt;

    pub(crate) async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
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

    pub(crate) fn http_request_complete(bytes: &[u8]) -> bool {
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use super::*;

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
                extra: HashMap::new(),
            })
        }

        async fn refresh_token(&self, credentials: &OAuthCredentials) -> Result<OAuthCredentials> {
            Ok(OAuthCredentials {
                refresh: credentials.refresh.clone(),
                access: format!("{}-refreshed", credentials.access),
                expires: crate::utils::time::now_millis().saturating_add(60_000),
                extra: credentials.extra.clone(),
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
            extra: HashMap::new(),
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
                extra: HashMap::new(),
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

    #[tokio::test]
    async fn oauth_login_callbacks_builder_accepts_plain_closures() {
        let callbacks = OAuthLoginCallbacks::builder()
            .on_device_code(|_| {})
            .on_prompt(|prompt| async move { Ok(prompt.placeholder.unwrap_or_default()) })
            .on_manual_code_input(|| async { Ok("manual".to_string()) })
            .on_select(|prompt| async move {
                Ok(prompt.options.first().map(|option| option.id.clone()))
            })
            .build();

        let prompt = (callbacks.on_prompt)(OAuthPrompt {
            message: "prompt".to_string(),
            placeholder: Some("default".to_string()),
            allow_empty: true,
        })
        .await
        .expect("prompt result");
        assert_eq!(prompt, "default");

        let manual = callbacks.on_manual_code_input.expect("manual callback")()
            .await
            .expect("manual result");
        assert_eq!(manual, "manual");

        let selected = callbacks.on_select.expect("select callback")(OAuthSelectPrompt {
            message: "select".to_string(),
            options: vec![OAuthSelectOption {
                id: "first".to_string(),
                label: "First".to_string(),
            }],
        })
        .await
        .expect("select result");
        assert_eq!(selected.as_deref(), Some("first"));
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

    #[tokio::test(flavor = "current_thread")]
    async fn device_code_poll_completes_immediately() {
        let value = poll_oauth_device_code_flow(None, Some(1), None, || async {
            Ok(OAuthDeviceCodePollResult::Complete("token"))
        })
        .await
        .unwrap();

        assert_eq!(value, "token");
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn device_code_poll_polls_immediately_and_increases_interval_after_slow_down() {
        let poll_times = Arc::new(Mutex::new(Vec::new()));
        let start = tokio::time::Instant::now();
        let poll = {
            let poll_times = Arc::clone(&poll_times);
            tokio::spawn(async move {
                poll_oauth_device_code_flow(Some(1), Some(60), None, move || {
                    let poll_times = Arc::clone(&poll_times);
                    async move {
                        let mut poll_times = poll_times.lock().expect("poll times lock poisoned");
                        poll_times.push(tokio::time::Instant::now().duration_since(start));
                        Ok(match poll_times.len() {
                            1 => OAuthDeviceCodePollResult::Pending,
                            2 => OAuthDeviceCodePollResult::SlowDown {
                                interval_seconds: None,
                            },
                            3 => OAuthDeviceCodePollResult::Complete("token"),
                            _ => OAuthDeviceCodePollResult::Failed(
                                "unexpected extra access token poll".to_string(),
                            ),
                        })
                    }
                })
                .await
            })
        };

        tokio::task::yield_now().await;
        assert_eq!(
            *poll_times.lock().expect("poll times lock poisoned"),
            vec![Duration::from_millis(0)]
        );

        tokio::time::advance(Duration::from_millis(999)).await;
        tokio::task::yield_now().await;
        assert_eq!(
            poll_times.lock().expect("poll times lock poisoned").len(),
            1
        );

        tokio::time::advance(Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert_eq!(
            *poll_times.lock().expect("poll times lock poisoned"),
            vec![Duration::from_millis(0), Duration::from_millis(1000)]
        );

        tokio::time::advance(Duration::from_millis(5999)).await;
        tokio::task::yield_now().await;
        assert_eq!(
            poll_times.lock().expect("poll times lock poisoned").len(),
            2
        );

        tokio::time::advance(Duration::from_millis(1)).await;
        let value = poll.await.expect("poll task joins").expect("poll succeeds");

        assert_eq!(value, "token");
        assert_eq!(
            *poll_times.lock().expect("poll times lock poisoned"),
            vec![
                Duration::from_millis(0),
                Duration::from_millis(1000),
                Duration::from_millis(7000),
            ]
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn device_code_poll_can_wait_before_the_first_poll() {
        let poll_times = Arc::new(Mutex::new(Vec::new()));
        let start = tokio::time::Instant::now();
        let poll = {
            let poll_times = Arc::clone(&poll_times);
            tokio::spawn(async move {
                poll_oauth_device_code_flow_inner(Some(2), Some(30), true, None, move || {
                    let poll_times = Arc::clone(&poll_times);
                    async move {
                        poll_times
                            .lock()
                            .expect("poll times lock poisoned")
                            .push(tokio::time::Instant::now().duration_since(start));
                        Ok(OAuthDeviceCodePollResult::Complete("token"))
                    }
                })
                .await
            })
        };

        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_millis(1999)).await;
        tokio::task::yield_now().await;
        assert!(
            poll_times
                .lock()
                .expect("poll times lock poisoned")
                .is_empty()
        );

        tokio::time::advance(Duration::from_millis(1)).await;
        let value = poll.await.expect("poll task joins").expect("poll succeeds");

        assert_eq!(value, "token");
        assert_eq!(
            *poll_times.lock().expect("poll times lock poisoned"),
            vec![Duration::from_millis(2000)]
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn device_code_poll_honors_a_server_provided_slow_down_interval() {
        let poll_times = Arc::new(Mutex::new(Vec::new()));
        let start = tokio::time::Instant::now();
        let poll = {
            let poll_times = Arc::clone(&poll_times);
            tokio::spawn(async move {
                poll_oauth_device_code_flow(Some(2), Some(900), None, move || {
                    let poll_times = Arc::clone(&poll_times);
                    async move {
                        let mut poll_times = poll_times.lock().expect("poll times lock poisoned");
                        poll_times.push(tokio::time::Instant::now().duration_since(start));
                        Ok(match poll_times.len() {
                            1 => OAuthDeviceCodePollResult::SlowDown {
                                interval_seconds: Some(30),
                            },
                            2 => OAuthDeviceCodePollResult::Complete("token"),
                            _ => OAuthDeviceCodePollResult::Failed(
                                "unexpected extra access token poll".to_string(),
                            ),
                        })
                    }
                })
                .await
            })
        };

        tokio::task::yield_now().await;
        assert_eq!(
            *poll_times.lock().expect("poll times lock poisoned"),
            vec![Duration::from_millis(0)]
        );

        tokio::time::advance(Duration::from_millis(29_999)).await;
        tokio::task::yield_now().await;
        assert_eq!(
            poll_times.lock().expect("poll times lock poisoned").len(),
            1
        );

        tokio::time::advance(Duration::from_millis(1)).await;
        let value = poll.await.expect("poll task joins").expect("poll succeeds");

        assert_eq!(value, "token");
        assert_eq!(
            *poll_times.lock().expect("poll times lock poisoned"),
            vec![Duration::from_millis(0), Duration::from_millis(30_000)]
        );
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
                    Ok(OAuthDeviceCodePollResult::SlowDown {
                        interval_seconds: None,
                    })
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
        let callbacks = OAuthLoginCallbacks::builder()
            .on_prompt({
                let seen_prompt = Arc::clone(&seen_prompt);
                move |prompt| {
                    *seen_prompt.lock().expect("prompt lock poisoned") = Some(prompt);
                    async { Err(Error::Provider("stop before network".to_string())) }
                }
            })
            .build();

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
}
