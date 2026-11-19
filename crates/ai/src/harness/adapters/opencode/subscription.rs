//! Port of `harness-opencode/src/opencode-subscription.ts`: OpenCode's own
//! login (`auth.json`) as a credential source, and GitLab Duo direct access.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::harness::auth::HarnessAuthentication;
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::network::HarnessRequestTransformation;
use crate::harness::utils::{
    OAuthFetch, OAuthFetchRequest, RefreshOAuthAccessTokenOptions,
    create_credential_request_transformation, current_platform, is_access_token_expiring_soon,
    is_windows, parse_jwt_payload, refresh_oauth_access_token,
};

use super::OPENCODE_HARNESS_ID;
use super::auth::{
    OPENCODE_SUBSCRIPTION_ACCESS_TOKEN_ENVIRONMENT_VARIABLE, OpenCodeAuthenticationMode,
    OpenCodeResolvedAuthenticationMode, has_opencode_credential,
    resolve_opencode_authentication_mode, resolve_opencode_env,
};

const OPENAI_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const XAI_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const GITLAB_CLIENT_ID: &str = "1d89f9fdb23ee96d4e603201f6861dab6e143c5c3c00469a018a2d94bdc03d4e";
const GITLAB_AI_GATEWAY_URL: &str = "https://cloud.gitlab.com";
const GITLAB_ANTHROPIC_PROVIDER_ID: &str = "ai-sdk-gitlab-anthropic";
const GITLAB_OPENAI_CHAT_PROVIDER_ID: &str = "ai-sdk-gitlab-openai-chat";
const GITLAB_OPENAI_RESPONSES_PROVIDER_ID: &str = "ai-sdk-gitlab-openai-responses";
const WORKFLOW: &str = "workflow";

/// `OPEN_CODE_GITLAB_MODELS`: model id → (provider id, upstream model id).
const OPENCODE_GITLAB_MODELS: &[(&str, &str, &str)] = &[
    (
        "duo-chat-fable-5",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-fable-5",
    ),
    (
        "duo-chat-opus-5",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-opus-5",
    ),
    (
        "duo-chat-opus-4-8",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-opus-4-8",
    ),
    (
        "duo-chat-opus-4-7",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-opus-4-7",
    ),
    (
        "duo-chat-opus-4-6",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-opus-4-6",
    ),
    (
        "duo-chat-sonnet-5",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-sonnet-5",
    ),
    (
        "duo-chat-sonnet-4-6",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-sonnet-4-6",
    ),
    (
        "duo-chat-opus-4-5",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-opus-4-5-20251101",
    ),
    (
        "duo-chat-sonnet-4-5",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-sonnet-4-5-20250929",
    ),
    (
        "duo-chat-haiku-4-5",
        GITLAB_ANTHROPIC_PROVIDER_ID,
        "claude-haiku-4-5-20251001",
    ),
    (
        "duo-chat-gpt-5-1",
        GITLAB_OPENAI_CHAT_PROVIDER_ID,
        "gpt-5.1-2025-11-13",
    ),
    (
        "duo-chat-gpt-5-2",
        GITLAB_OPENAI_CHAT_PROVIDER_ID,
        "gpt-5.2-2025-12-11",
    ),
    (
        "duo-chat-gpt-5-4",
        GITLAB_OPENAI_CHAT_PROVIDER_ID,
        "gpt-5.4-2026-03-05",
    ),
    (
        "duo-chat-gpt-5-5",
        GITLAB_OPENAI_CHAT_PROVIDER_ID,
        "gpt-5.5-2026-04-23",
    ),
    (
        "duo-chat-gpt-5-mini",
        GITLAB_OPENAI_CHAT_PROVIDER_ID,
        "gpt-5-mini-2025-08-07",
    ),
    (
        "duo-chat-gpt-5-4-mini",
        GITLAB_OPENAI_CHAT_PROVIDER_ID,
        "gpt-5.4-mini",
    ),
    (
        "duo-chat-gpt-5-4-nano",
        GITLAB_OPENAI_CHAT_PROVIDER_ID,
        "gpt-5.4-nano",
    ),
    (
        "duo-chat-gpt-5-6-sol",
        GITLAB_OPENAI_RESPONSES_PROVIDER_ID,
        "gpt-5.6-sol",
    ),
    (
        "duo-chat-gpt-5-6-terra",
        GITLAB_OPENAI_RESPONSES_PROVIDER_ID,
        "gpt-5.6-terra",
    ),
    (
        "duo-chat-gpt-5-6-luna",
        GITLAB_OPENAI_RESPONSES_PROVIDER_ID,
        "gpt-5.6-luna",
    ),
    (
        "duo-chat-gpt-5-codex",
        GITLAB_OPENAI_RESPONSES_PROVIDER_ID,
        "gpt-5-codex",
    ),
    (
        "duo-chat-gpt-5-2-codex",
        GITLAB_OPENAI_RESPONSES_PROVIDER_ID,
        "gpt-5.2-codex",
    ),
    (
        "duo-chat-gpt-5-3-codex",
        GITLAB_OPENAI_RESPONSES_PROVIDER_ID,
        "gpt-5.3-codex",
    ),
    ("duo-workflow", WORKFLOW, "default"),
    ("duo-workflow-default", WORKFLOW, "default"),
    (
        "duo-workflow-sonnet-4-5",
        WORKFLOW,
        "anthropic/claude-sonnet-4-5-20250929",
    ),
    ("duo-workflow-sonnet-5", WORKFLOW, "claude_sonnet_5"),
    ("duo-workflow-opus-5", WORKFLOW, "claude_opus_5"),
    ("duo-workflow-sonnet-4-6", WORKFLOW, "claude_sonnet_4_6"),
    (
        "duo-workflow-opus-4-5",
        WORKFLOW,
        "anthropic/claude-opus-4-5-20251101",
    ),
    (
        "duo-workflow-haiku-4-5",
        WORKFLOW,
        "claude_haiku_4_5_20251001",
    ),
    (
        "duo-workflow-opus-4-6",
        WORKFLOW,
        "claude_opus_4_6_20260205",
    ),
];

fn gitlab_models() -> impl Iterator<Item = (&'static str, &'static str, &'static str)> {
    OPENCODE_GITLAB_MODELS.iter().copied()
}

/// `OpenCodeSubscription`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCodeSubscription {
    /// `openai`, `xai`, `github-copilot`, `poe`, `opencode-go` or `gitlab`.
    pub provider_id: OpenCodeResolvedAuthenticationMode,
    pub access_token: String,
    pub account_id: Option<String>,
    pub enterprise_url: Option<String>,
}

impl OpenCodeSubscription {
    pub fn new(
        provider_id: OpenCodeResolvedAuthenticationMode,
        access_token: impl Into<String>,
    ) -> Self {
        Self {
            provider_id,
            access_token: access_token.into(),
            account_id: None,
            enterprise_url: None,
        }
    }
}

/// `OpenCodeGitLabDirectAccess`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCodeGitLabDirectAccess {
    pub access_token: String,
    pub headers: IndexMap<String, String>,
    pub ai_gateway_url: String,
}

/// The result of [`resolve_opencode_authentication`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCodeResolvedAuthentication {
    pub environment: IndexMap<String, String>,
    pub authentication_mode: OpenCodeResolvedAuthenticationMode,
    pub subscription: Option<OpenCodeSubscription>,
}

/// `readSubscription`: called with the provider and the host environment.
pub type ReadOpenCodeSubscriptionFn = Arc<
    dyn Fn(
            OpenCodeResolvedAuthenticationMode,
            IndexMap<String, String>,
        ) -> BoxFuture<'static, HarnessResult<Option<OpenCodeSubscription>>>
        + Send
        + Sync,
>;

/// Options of [`resolve_opencode_authentication`].
#[derive(Clone, Default)]
pub struct ResolveOpenCodeAuthenticationOptions {
    pub auth: Option<OpenCodeAuthenticationMode>,
    pub model: Option<String>,
    pub provider: Option<String>,
    /// Defaults to the host's environment.
    pub process_env: Option<IndexMap<String, String>>,
    /// Defaults to [`read_opencode_subscription`].
    pub read_subscription: Option<ReadOpenCodeSubscriptionFn>,
}

/// `resolveOpenCodeAuthentication({ auth, model, provider, processEnv,
/// readSubscription })`.
pub async fn resolve_opencode_authentication(
    options: ResolveOpenCodeAuthenticationOptions,
) -> HarnessResult<OpenCodeResolvedAuthentication> {
    let process_env = options
        .process_env
        .clone()
        .unwrap_or_else(|| std::env::vars().collect());
    let auth = options.auth.as_ref();
    let model = options.model.as_deref();
    let provider = options.provider.as_deref();
    let environment = resolve_opencode_env(auth, model, provider, &process_env);
    let authentication_mode =
        resolve_opencode_authentication_mode(auth, model, provider, &process_env);
    if matches!(
        auth,
        Some(HarnessAuthentication::Environment(_) | HarnessAuthentication::AiGateway)
    ) || authentication_mode == OpenCodeResolvedAuthenticationMode::AiGateway
        || has_opencode_credential(&environment, authentication_mode)
    {
        return Ok(OpenCodeResolvedAuthentication {
            environment,
            authentication_mode,
            subscription: None,
        });
    }
    let subscription = match &options.read_subscription {
        Some(read) => read(authentication_mode, process_env).await?,
        None => {
            read_opencode_subscription(ReadOpenCodeSubscriptionOptions {
                provider_id: authentication_mode,
                env: Some(process_env),
                ..Default::default()
            })
            .await?
        }
    };
    Ok(match subscription {
        None => OpenCodeResolvedAuthentication {
            environment,
            authentication_mode,
            subscription: None,
        },
        Some(subscription) => {
            let mut environment = environment;
            environment.insert(
                OPENCODE_SUBSCRIPTION_ACCESS_TOKEN_ENVIRONMENT_VARIABLE.to_string(),
                subscription.access_token.clone(),
            );
            OpenCodeResolvedAuthentication {
                environment,
                authentication_mode,
                subscription: Some(subscription),
            }
        }
    })
}

/// Options of [`read_opencode_subscription`]. Unset fields default to the
/// host's environment, home directory and platform.
#[derive(Clone)]
pub struct ReadOpenCodeSubscriptionOptions {
    pub provider_id: OpenCodeResolvedAuthenticationMode,
    pub env: Option<IndexMap<String, String>>,
    pub home_directory: Option<String>,
    pub platform: Option<String>,
    pub fetch: Option<OAuthFetch>,
}

impl Default for ReadOpenCodeSubscriptionOptions {
    fn default() -> Self {
        Self {
            provider_id: OpenCodeResolvedAuthenticationMode::Anthropic,
            env: None,
            home_directory: None,
            platform: None,
            fetch: None,
        }
    }
}

/// `isSubscriptionProvider`.
fn is_subscription_provider(provider: OpenCodeResolvedAuthenticationMode) -> bool {
    use OpenCodeResolvedAuthenticationMode as M;
    matches!(
        provider,
        M::Openai | M::Xai | M::GithubCopilot | M::Poe | M::OpencodeGo | M::Gitlab
    )
}

/// `readOpenCodeSubscription({ providerId, env, homeDirectory, platform,
/// fetch })`.
pub async fn read_opencode_subscription(
    options: ReadOpenCodeSubscriptionOptions,
) -> HarnessResult<Option<OpenCodeSubscription>> {
    use OpenCodeResolvedAuthenticationMode as M;
    let provider = options.provider_id;
    if !is_subscription_provider(provider) {
        return Ok(None);
    }
    let provider_key = provider.as_str();
    let env = options.env.unwrap_or_else(|| std::env::vars().collect());
    let home_directory = options
        .home_directory
        .map(PathBuf::from)
        .or_else(|| std::env::home_dir().filter(|path| !path.as_os_str().is_empty()))
        .unwrap_or_default();
    let platform = options
        .platform
        .unwrap_or_else(|| current_platform().to_string());
    let Some(mut stored) = read_store(provider_key, &env, &home_directory, &platform).await else {
        return Ok(None);
    };
    let Some(Value::Object(candidate)) = stored.value.get(provider_key).cloned() else {
        return Ok(None);
    };
    let string = |key: &str| {
        candidate
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let enterprise_url = string("enterpriseUrl");

    match candidate.get("type").and_then(Value::as_str) {
        Some("api") => {
            return Ok(string("key")
                .filter(|key| !key.is_empty())
                .map(|key| OpenCodeSubscription::new(provider, key)));
        }
        Some("oauth") => {}
        _ => return Ok(None),
    }
    if provider == M::OpencodeGo {
        return Ok(None);
    }
    let (Some(access), Some(refresh), Some(expires)) = (
        string("access"),
        string("refresh"),
        candidate.get("expires").and_then(Value::as_f64),
    ) else {
        return Ok(None);
    };
    if provider == M::GithubCopilot {
        return Ok(Some(OpenCodeSubscription {
            enterprise_url,
            ..OpenCodeSubscription::new(provider, refresh)
        }));
    }
    if provider == M::Poe {
        if is_access_token_expiring_soon(expires, None, None) {
            return Err(HarnessError::new(
                "OpenCode Poe subscription API key is expiring soon. Run OpenCode login again.",
            )
            .into());
        }
        return Ok(Some(OpenCodeSubscription::new(provider, access)));
    }

    let mut access_token = access.clone();
    let mut account_id = string("accountId").or_else(|| {
        (provider == M::Openai)
            .then(|| extract_openai_account_id(&access))
            .flatten()
    });
    if is_access_token_expiring_soon(expires, None, None) {
        let (token_url, client_id) = resolve_refresh_settings(provider, &candidate, &env);
        let refreshed = refresh_oauth_access_token(RefreshOAuthAccessTokenOptions {
            token_url,
            client_id,
            refresh_token: refresh.clone(),
            fetch: options.fetch.clone(),
            ..Default::default()
        })
        .await?;
        access_token = refreshed.access_token.clone();
        if provider == M::Openai {
            account_id = extract_openai_account_id(&refreshed.access_token).or(account_id);
        }
        let mut updated = candidate.clone();
        updated.insert("access".into(), json!(refreshed.access_token));
        updated.insert(
            "refresh".into(),
            json!(refreshed.refresh_token.clone().unwrap_or(refresh)),
        );
        updated.insert("expires".into(), json!(refreshed.expires_at));
        if let Some(account_id) = &account_id {
            updated.insert("accountId".into(), json!(account_id));
        }
        stored
            .value
            .insert(provider_key.to_string(), Value::Object(updated));
        if let Some(path) = &stored.write_path {
            write_store(path, &stored.value).await?;
        }
    }
    Ok(Some(OpenCodeSubscription {
        provider_id: provider,
        access_token,
        account_id,
        enterprise_url,
    }))
}

/// Options of [`request_opencode_gitlab_direct_access`].
#[derive(Clone, Default)]
pub struct RequestOpenCodeGitLabDirectAccessOptions {
    pub access_token: String,
    /// Defaults to `https://gitlab.com`.
    pub instance_url: Option<String>,
    /// Defaults to `https://cloud.gitlab.com`.
    pub ai_gateway_url: Option<String>,
    /// Defaults to a `reqwest` client.
    pub fetch: Option<OAuthFetch>,
}

/// `requestOpenCodeGitLabDirectAccess({ accessToken, instanceUrl,
/// aiGatewayUrl, fetch })`.
pub async fn request_opencode_gitlab_direct_access(
    options: RequestOpenCodeGitLabDirectAccessOptions,
) -> HarnessResult<OpenCodeGitLabDirectAccess> {
    let instance_url = options
        .instance_url
        .unwrap_or_else(|| "https://gitlab.com".to_string());
    let request = OAuthFetchRequest {
        url: format!(
            "{}/api/v4/ai/third_party_agents/direct_access",
            instance_url.trim_end_matches('/')
        ),
        method: "POST".into(),
        headers: IndexMap::from([
            (
                "Authorization".to_string(),
                format!("Bearer {}", options.access_token),
            ),
            ("Content-Type".to_string(), "application/json".to_string()),
        ]),
        body: json!({ "feature_flags": { "DuoAgentPlatformNext": true } }).to_string(),
    };
    let response = match &options.fetch {
        Some(fetch) => fetch(request).await?,
        None => default_fetch(request).await?,
    };
    if !(200..300).contains(&response.status) {
        return Err(HarnessError::new(format!(
            "OpenCode GitLab direct access request failed with status {}.",
            response.status
        ))
        .into());
    }
    let Ok(Value::Object(parsed)) = serde_json::from_str::<Value>(&response.text) else {
        return Err(
            HarnessError::new("OpenCode GitLab direct access returned invalid JSON.").into(),
        );
    };
    let token = parsed
        .get("token")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let headers = match parsed.get("headers") {
        Some(Value::Object(headers)) if headers.values().all(Value::is_string) => Some(
            headers
                .iter()
                .map(|(key, value)| (key.clone(), value.as_str().unwrap().to_string()))
                .collect::<IndexMap<_, _>>(),
        ),
        _ => None,
    };
    let (false, Some(headers)) = (token.is_empty(), headers) else {
        return Err(HarnessError::new(
            "OpenCode GitLab direct access returned invalid credentials.",
        )
        .into());
    };
    Ok(OpenCodeGitLabDirectAccess {
        access_token: token.to_string(),
        headers,
        ai_gateway_url: options
            .ai_gateway_url
            .as_deref()
            .unwrap_or(GITLAB_AI_GATEWAY_URL)
            .trim_end_matches('/')
            .to_string(),
    })
}

async fn default_fetch(
    request: OAuthFetchRequest,
) -> HarnessResult<crate::harness::utils::OAuthFetchResponse> {
    let client = reqwest::Client::new();
    let mut builder = client.request(
        reqwest::Method::from_bytes(request.method.as_bytes())?,
        &request.url,
    );
    for (name, value) in &request.headers {
        builder = builder.header(name, value);
    }
    let response = builder.body(request.body).send().await?;
    let status = response.status().as_u16();
    let text = response.text().await?;
    Ok(crate::harness::utils::OAuthFetchResponse { status, text })
}

/// `createOpenCodeGitLabSubscriptionRequestTransformations`.
pub fn create_opencode_gitlab_subscription_request_transformations(
    direct_access: &OpenCodeGitLabDirectAccess,
    sandbox_access_token: &str,
) -> HarnessResult<Vec<HarnessRequestTransformation>> {
    let mut transform: IndexMap<String, String> = direct_access
        .headers
        .iter()
        .filter(|(name, _)| {
            let name = name.to_ascii_lowercase();
            name != "authorization" && name != "x-api-key"
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    transform.insert(
        "Authorization".into(),
        format!("Bearer {}", direct_access.access_token),
    );
    let match_headers = IndexMap::from([(
        "Authorization".to_string(),
        format!("Bearer {sandbox_access_token}"),
    )]);
    [
        format!("{}/ai/v1/proxy/anthropic", direct_access.ai_gateway_url),
        format!("{}/ai/v1/proxy/openai/v1", direct_access.ai_gateway_url),
    ]
    .iter()
    .map(|url| create_credential_request_transformation(url, &match_headers, &transform))
    .collect()
}

fn gitlab_provider_models(provider_id: &str) -> Value {
    Value::Object(
        gitlab_models()
            .filter(|(_, provider, _)| *provider == provider_id)
            .map(|(id, _, model)| (id.to_string(), json!({ "id": model, "name": id })))
            .collect(),
    )
}

/// `createOpenCodeGitLabSubscriptionConfig`.
pub fn create_opencode_gitlab_subscription_config(
    opencode_config: Option<&Map<String, Value>>,
    sandbox_access_token: &str,
    ai_gateway_url: &str,
    headers: Option<&IndexMap<String, String>>,
) -> Map<String, Value> {
    let with_headers = |mut options: Map<String, Value>| {
        if let Some(headers) = headers {
            options.insert("headers".into(), json!(headers));
        }
        Value::Object(options)
    };
    let providers = [
        (
            GITLAB_ANTHROPIC_PROVIDER_ID,
            "GitLab Anthropic",
            "@ai-sdk/anthropic",
            format!("{ai_gateway_url}/ai/v1/proxy/anthropic/v1"),
            "authToken",
        ),
        (
            GITLAB_OPENAI_CHAT_PROVIDER_ID,
            "GitLab OpenAI Chat",
            "@ai-sdk/openai-compatible",
            format!("{ai_gateway_url}/ai/v1/proxy/openai/v1"),
            "apiKey",
        ),
        (
            GITLAB_OPENAI_RESPONSES_PROVIDER_ID,
            "GitLab OpenAI Responses",
            "@ai-sdk/openai",
            format!("{ai_gateway_url}/ai/v1/proxy/openai/v1"),
            "apiKey",
        ),
    ];
    let mut config = opencode_config.cloned().unwrap_or_default();
    let mut provider = match config.get("provider") {
        Some(Value::Object(provider)) => provider.clone(),
        _ => Map::new(),
    };
    for (id, name, npm, api, credential_key) in providers {
        let mut options = Map::new();
        options.insert(credential_key.into(), json!(sandbox_access_token));
        provider.insert(
            id.to_string(),
            json!({
                "name": name,
                "npm": npm,
                "api": api,
                "options": with_headers(options),
                "models": gitlab_provider_models(id),
            }),
        );
    }
    config.insert("provider".into(), Value::Object(provider));
    config
}

/// `resolveOpenCodeGitLabSubscriptionModel({ model, provider })`.
pub fn resolve_opencode_gitlab_subscription_model(
    model: Option<&str>,
    provider: Option<&str>,
) -> HarnessResult<Option<String>> {
    let unsupported = |message: String| -> HarnessResult<Option<String>> {
        Err(HarnessError::capability_unsupported(message, Some(OPENCODE_HARNESS_ID.into())).into())
    };
    let Some(model) = model else {
        if provider == Some("gitlab") {
            return unsupported(
                "Harness 'opencode' requires an explicit GitLab model when native GitLab subscription credentials are brokered.".into(),
            );
        }
        return Ok(None);
    };
    let (model_provider, model_id) = match model.split_once('/') {
        Some((prefix, rest)) => (Some(prefix), rest),
        None => (provider, model),
    };
    if model_provider != Some("gitlab") {
        return Ok(Some(model.to_string()));
    }
    let Some((_, provider_id, _)) = gitlab_models().find(|(id, _, _)| *id == model_id) else {
        return unsupported(format!(
            "Harness 'opencode' cannot broker native GitLab subscription credentials for unknown model '{model_id}'."
        ));
    };
    if provider_id == WORKFLOW {
        return unsupported(format!(
            "Harness 'opencode' cannot broker native GitLab subscription credentials for workflow model '{model_id}' because it authenticates over WebSocket. Use a sandbox without credential brokering to use the credential-forwarding fallback."
        ));
    }
    Ok(Some(format!("{provider_id}/{model_id}")))
}

/// `createOpenCodeSubscriptionAuthContent({ authentication, accessToken })`.
pub fn create_opencode_subscription_auth_content(
    authentication: &OpenCodeSubscription,
    access_token: &str,
) -> String {
    let provider = authentication.provider_id;
    let value = if provider == OpenCodeResolvedAuthenticationMode::OpencodeGo {
        json!({ "type": "api", "key": access_token })
    } else {
        let mut value = json!({
            "type": "oauth",
            "access": access_token,
            "refresh": if provider == OpenCodeResolvedAuthenticationMode::GithubCopilot {
                access_token
            } else {
                "host-managed"
            },
            "expires": 9007199254740991u64,
        });
        if let Some(account_id) = &authentication.account_id {
            value["accountId"] = json!(account_id);
        }
        if let Some(enterprise_url) = &authentication.enterprise_url {
            value["enterpriseUrl"] = json!(enterprise_url);
        }
        value
    };
    json!({ provider.as_str(): value }).to_string()
}

/// `createOpenCodeSubscriptionRequestTransformations({ authentication,
/// sandboxAccessToken })`.
pub fn create_opencode_subscription_request_transformations(
    authentication: &OpenCodeSubscription,
    sandbox_access_token: &str,
) -> HarnessResult<Vec<HarnessRequestTransformation>> {
    use OpenCodeResolvedAuthenticationMode as M;
    let match_url = resolve_request_url(authentication);
    let mut transform = IndexMap::from([(
        "Authorization".to_string(),
        format!("Bearer {}", authentication.access_token),
    )]);
    if authentication.provider_id == M::Openai
        && let Some(account_id) = authentication
            .account_id
            .as_ref()
            .filter(|id| !id.is_empty())
    {
        transform.insert("ChatGPT-Account-Id".into(), account_id.clone());
    }
    let mut transformations = vec![create_credential_request_transformation(
        &match_url,
        &IndexMap::from([(
            "Authorization".to_string(),
            format!("Bearer {sandbox_access_token}"),
        )]),
        &transform,
    )?];
    if matches!(authentication.provider_id, M::Poe | M::OpencodeGo) {
        transformations.push(create_credential_request_transformation(
            &match_url,
            &IndexMap::from([("x-api-key".to_string(), sandbox_access_token.to_string())]),
            &IndexMap::from([("x-api-key".to_string(), authentication.access_token.clone())]),
        )?);
    }
    Ok(transformations)
}

fn resolve_refresh_settings(
    provider: OpenCodeResolvedAuthenticationMode,
    candidate: &Map<String, Value>,
    env: &IndexMap<String, String>,
) -> (String, String) {
    match provider {
        OpenCodeResolvedAuthenticationMode::Openai => (
            "https://auth.openai.com/oauth/token".into(),
            OPENAI_CLIENT_ID.into(),
        ),
        OpenCodeResolvedAuthenticationMode::Xai => (
            "https://auth.x.ai/oauth2/token".into(),
            XAI_CLIENT_ID.into(),
        ),
        _ => {
            let instance_url = candidate
                .get("enterpriseUrl")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| env.get("GITLAB_INSTANCE_URL").cloned())
                .unwrap_or_else(|| "https://gitlab.com".into());
            (
                format!("{}/oauth/token", instance_url.trim_end_matches('/')),
                env.get("GITLAB_OAUTH_CLIENT_ID")
                    .cloned()
                    .unwrap_or_else(|| GITLAB_CLIENT_ID.into()),
            )
        }
    }
}

fn resolve_request_url(authentication: &OpenCodeSubscription) -> String {
    use OpenCodeResolvedAuthenticationMode as M;
    match authentication.provider_id {
        M::Openai => "https://chatgpt.com/backend-api/codex".into(),
        M::Xai => "https://api.x.ai/v1".into(),
        M::GithubCopilot => match &authentication.enterprise_url {
            None => "https://api.githubcopilot.com".into(),
            Some(url) => format!("https://copilot-api.{}", normalize_domain(url)),
        },
        M::Poe => "https://api.poe.com".into(),
        M::OpencodeGo => "https://opencode.ai/zen/go/v1".into(),
        _ => authentication
            .enterprise_url
            .clone()
            .unwrap_or_else(|| "https://gitlab.com".into()),
    }
}

fn normalize_domain(value: &str) -> String {
    let url = if value.contains("://") {
        value.to_string()
    } else {
        format!("https://{value}")
    };
    match reqwest::Url::parse(&url) {
        Ok(url) => match url.port() {
            Some(port) => format!("{}:{port}", url.host_str().unwrap_or_default()),
            None => url.host_str().unwrap_or_default().to_string(),
        },
        Err(_) => value
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .trim_end_matches('/')
            .to_string(),
    }
}

fn extract_openai_account_id(access_token: &str) -> Option<String> {
    let payload = parse_jwt_payload(access_token)?;
    payload
        .get("https://api.openai.com/auth")?
        .get("chatgpt_account_id")?
        .as_str()
        .map(str::to_string)
}

struct Store {
    value: Map<String, Value>,
    /// Where a refreshed record is written; `None` for `OPENCODE_AUTH_CONTENT`.
    write_path: Option<PathBuf>,
}

async fn read_store(
    provider_key: &str,
    env: &IndexMap<String, String>,
    home_directory: &Path,
    platform: &str,
) -> Option<Store> {
    if let Some(content) = env.get("OPENCODE_AUTH_CONTENT") {
        return match serde_json::from_str::<Value>(content) {
            Ok(Value::Object(value)) => Some(Store {
                value,
                write_path: None,
            }),
            _ => None,
        };
    }
    // OpenCode keeps auth.json in its XDG data directory on every platform,
    // including Windows. The GitLab auth plugin (before v3) writes to
    // `~/.opencode` on Windows instead, so that store is checked as a fallback.
    let auth_paths: Vec<PathBuf> = match env.get("XDG_DATA_HOME") {
        Some(data_home) => vec![Path::new(data_home).join("opencode").join("auth.json")],
        None => {
            let mut paths = vec![home_directory.join(".local/share/opencode/auth.json")];
            if is_windows(platform) {
                paths.push(home_directory.join(".opencode").join("auth.json"));
            }
            paths
        }
    };
    for auth_path in auth_paths {
        let Ok(text) = tokio::fs::read_to_string(&auth_path).await else {
            continue;
        };
        let Ok(Value::Object(value)) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if !value.get(provider_key).is_some_and(Value::is_object) {
            continue;
        }
        return Some(Store {
            value,
            write_path: Some(auth_path),
        });
    }
    None
}

async fn write_store(auth_path: &Path, value: &Map<String, Value>) -> HarnessResult<()> {
    if let Some(parent) = auth_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut temporary_path = auth_path.as_os_str().to_owned();
    temporary_path.push(format!(".{}.tmp", std::process::id()));
    let temporary_path = PathBuf::from(temporary_path);
    tokio::fs::write(
        &temporary_path,
        format!("{}\n", serde_json::to_string_pretty(value)?),
    )
    .await?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        tokio::fs::set_permissions(&temporary_path, std::fs::Permissions::from_mode(0o600)).await?;
    }
    tokio::fs::rename(&temporary_path, auth_path).await?;
    Ok(())
}

#[cfg(test)]
#[path = "subscription_tests.rs"]
mod tests;
