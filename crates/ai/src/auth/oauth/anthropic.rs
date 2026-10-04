//! Port of `auth/oauth/anthropic.ts`: Anthropic OAuth (Claude Pro/Max).
//!
//! The browser flow serves the redirect on a loopback callback server; when
//! the port is busy it falls back to pasting the redirect URL. The copy-code
//! flow needs no local server (headless/SSH).
//!
//! [`login_anthropic`] and [`refresh_anthropic_token`] keep the pre-1.0 ai.rs
//! entry points on top of [`anthropic_oauth`].

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use super::callback_server::{
    CallbackOrManualInput, OAuthCallbackServerOptions, start_oauth_callback_server,
    wait_for_callback_or_manual_input,
};
use super::compat::{OAuthLoginCallbacks, callbacks_interaction};
use super::fetch::{FetchRequest, OAuthFetch, default_oauth_fetch, fetch_with_timeout};
use super::pkce::generate_pkce;
use crate::auth::types::{
    AuthEvent, AuthPrompt, AuthPromptKind, AuthSelectOption, LoginOptions, ModelAuth, OAuthAuth,
    OAuthCredential, ProviderAuthInteraction,
};
use crate::utils::provider_env::get_provider_env_value;
use crate::utils::time::now_millis;
use crate::{Error, Result};

fn decode(value: &str) -> String {
    String::from_utf8(STANDARD.decode(value).expect("static base64")).expect("static utf-8")
}

fn client_id() -> String {
    decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl")
}

const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const CALLBACK_PORT: u16 = 53692;
const CALLBACK_PATH: &str = "/callback";
const REDIRECT_URI: &str = "http://localhost:53692/callback";
const COPY_CODE_REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
const ANTHROPIC_BROWSER_LOGIN_METHOD: &str = "browser";
const ANTHROPIC_COPY_CODE_LOGIN_METHOD: &str = "copy_code";
const SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const TOKEN_TIMEOUT_MS: u64 = 30_000;
const EXPIRY_SKEW_MS: u64 = 5 * 60 * 1000;

/// `PI_OAUTH_CALLBACK_HOST`, read per login (Pi reads it at module load).
fn callback_host() -> String {
    get_provider_env_value("PI_OAUTH_CALLBACK_HOST", None)
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

#[derive(Debug, Default, PartialEq, Eq)]
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
        let param = |name: &str| {
            url.query_pairs()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.into_owned())
        };
        return AuthorizationInput {
            code: param("code"),
            state: param("state"),
        };
    }

    if let Some((code, state)) = value.split_once('#') {
        // `split("#", 2)` keeps only the text up to a second `#`.
        let state = state.split('#').next().unwrap_or_default();
        return AuthorizationInput {
            code: Some(code.to_string()),
            state: Some(state.to_string()),
        };
    }

    if value.contains("code=") {
        let Ok(url) = reqwest::Url::parse(&format!(
            "http://localhost/?{}",
            value.trim_start_matches('?')
        )) else {
            return AuthorizationInput::default();
        };
        let param = |name: &str| {
            url.query_pairs()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.into_owned())
        };
        return AuthorizationInput {
            code: param("code"),
            state: param("state"),
        };
    }

    AuthorizationInput {
        code: Some(value.to_string()),
        state: None,
    }
}

/// `formatErrorDetails()`. Rust errors have no `name` or stack; the message
/// and the `source()` chain are reported.
fn format_error_details(error: &(dyn std::error::Error + 'static)) -> String {
    let mut details = vec![format!("Error: {error}")];
    if let Some(cause) = error.source() {
        details.push(format!("cause={}", format_error_details(cause)));
    }
    details.join("; ")
}

async fn post_json(
    fetch: &OAuthFetch,
    url: &str,
    body: Value,
    signal: &CancellationToken,
) -> Result<String> {
    let request = FetchRequest::post(url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .body(body.to_string());
    let response = fetch_with_timeout(
        fetch,
        request,
        signal,
        Duration::from_millis(TOKEN_TIMEOUT_MS),
    )
    .await?;
    if !response.ok() {
        return Err(Error::message(format!(
            "HTTP request failed. status={}; url={url}; body={}",
            response.status, response.body
        )));
    }
    Ok(response.body)
}

fn token_credential(
    response_body: &str,
) -> std::result::Result<OAuthCredential, serde_json::Error> {
    #[derive(serde::Deserialize)]
    struct TokenData {
        access_token: String,
        refresh_token: String,
        expires_in: u64,
    }
    let data: TokenData = serde_json::from_str(response_body)?;
    Ok(OAuthCredential {
        refresh: data.refresh_token,
        access: data.access_token,
        expires: (now_millis() + data.expires_in * 1000).saturating_sub(EXPIRY_SKEW_MS),
        extra: Map::new(),
    })
}

async fn exchange_authorization_code(
    fetch: &OAuthFetch,
    code: &str,
    state: &str,
    verifier: &str,
    redirect_uri: &str,
    signal: &CancellationToken,
) -> Result<OAuthCredential> {
    let response_body = post_json(
        fetch,
        TOKEN_URL,
        json!({
            "grant_type": "authorization_code",
            "client_id": client_id(),
            "code": code,
            "state": state,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }),
        signal,
    )
    .await
    .map_err(|error| {
        Error::message(format!(
            "Token exchange request failed. url={TOKEN_URL}; redirect_uri={redirect_uri}; response_type=authorization_code; details={}",
            format_error_details(&error)
        ))
    })?;

    token_credential(&response_body).map_err(|error| {
        Error::message(format!(
            "Token exchange returned invalid JSON. url={TOKEN_URL}; body={response_body}; details={}",
            format_error_details(&error)
        ))
    })
}

fn authorize_url(challenge: &str, verifier: &str, redirect_uri: &str) -> String {
    let mut url = reqwest::Url::parse(AUTHORIZE_URL).expect("static url");
    url.query_pairs_mut().extend_pairs([
        ("code", "true"),
        ("client_id", client_id().as_str()),
        ("response_type", "code"),
        ("redirect_uri", redirect_uri),
        ("scope", SCOPES),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("state", verifier),
    ]);
    url.to_string()
}

async fn login_anthropic_browser(
    fetch: &OAuthFetch,
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential> {
    let pkce = generate_pkce()?;
    // A busy port (or any other bind failure) falls back to pasting the redirect URL.
    let callback = start_oauth_callback_server(OAuthCallbackServerOptions {
        provider_name: "Anthropic".to_string(),
        host: callback_host(),
        port: CALLBACK_PORT,
        path: CALLBACK_PATH.to_string(),
        redirect_host: None,
        state: Some(pkce.verifier.clone()),
        complete: Arc::new(|code| Box::pin(async move { Ok(code) })),
        signal: Some(interaction.signal.clone()),
        timeout_ms: None,
    })
    .await
    .ok();

    let result = async {
        interaction.notify(AuthEvent::AuthUrl {
            url: authorize_url(&pkce.challenge, &pkce.verifier, REDIRECT_URI),
            instructions: Some(
                "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here."
                    .to_string(),
            ),
        });

        let result = wait_for_callback_or_manual_input(
            interaction,
            callback.as_ref(),
            "Complete login in your browser, or paste the authorization code / redirect URL here:",
            REDIRECT_URI,
        )
        .await?;
        let (code, state) = match result {
            CallbackOrManualInput::Callback(code) => (Some(code), pkce.verifier.clone()),
            CallbackOrManualInput::Manual(input) => {
                let parsed = parse_authorization_input(&input);
                if parsed
                    .state
                    .as_ref()
                    .is_some_and(|state| !state.is_empty() && *state != pkce.verifier)
                {
                    return Err(Error::message("OAuth state mismatch"));
                }
                let state = parsed.state.unwrap_or_else(|| pkce.verifier.clone());
                (parsed.code, state)
            }
        };

        let Some(code) = code.filter(|code| !code.is_empty()) else {
            return Err(Error::message("Missing authorization code"));
        };
        interaction.notify(AuthEvent::Progress {
            message: "Exchanging authorization code for tokens...".to_string(),
        });
        exchange_authorization_code(
            fetch,
            &code,
            &state,
            &pkce.verifier,
            REDIRECT_URI,
            &interaction.signal,
        )
        .await
    }
    .await;
    if let Some(callback) = &callback {
        callback.close();
    }
    result
}

async fn login_anthropic_copy_code(
    fetch: &OAuthFetch,
    interaction: &ProviderAuthInteraction,
) -> Result<OAuthCredential> {
    let pkce = generate_pkce()?;
    interaction.notify(AuthEvent::AuthUrl {
        url: authorize_url(&pkce.challenge, &pkce.verifier, COPY_CODE_REDIRECT_URI),
        instructions: Some(
            "Complete login in your browser, then copy the code Anthropic shows and paste it here."
                .to_string(),
        ),
    });

    let input = interaction
        .prompt(AuthPrompt {
            signal: Some(interaction.signal.clone()),
            kind: AuthPromptKind::ManualCode {
                message: "Paste the code Anthropic shows after you sign in:".to_string(),
                placeholder: Some("code#state".to_string()),
            },
        })
        .await?;
    let parsed = parse_authorization_input(&input);
    if parsed
        .state
        .as_ref()
        .is_some_and(|state| !state.is_empty() && *state != pkce.verifier)
    {
        return Err(Error::message("OAuth state mismatch"));
    }
    let Some(code) = parsed.code.filter(|code| !code.is_empty()) else {
        return Err(Error::message("Missing authorization code"));
    };
    interaction.notify(AuthEvent::Progress {
        message: "Exchanging authorization code for tokens...".to_string(),
    });
    let state = parsed.state.unwrap_or_else(|| pkce.verifier.clone());
    exchange_authorization_code(
        fetch,
        &code,
        &state,
        &pkce.verifier,
        COPY_CODE_REDIRECT_URI,
        &interaction.signal,
    )
    .await
}

async fn refresh_token_with(
    fetch: &OAuthFetch,
    refresh_token: &str,
    signal: &CancellationToken,
) -> Result<OAuthCredential> {
    let response_body = post_json(
        fetch,
        TOKEN_URL,
        json!({
            "grant_type": "refresh_token",
            "client_id": client_id(),
            "refresh_token": refresh_token,
        }),
        signal,
    )
    .await
    .map_err(|error| {
        Error::message(format!(
            "Anthropic token refresh request failed. url={TOKEN_URL}; details={}",
            format_error_details(&error)
        ))
    })?;

    token_credential(&response_body).map_err(|error| {
        Error::message(format!(
            "Anthropic token refresh returned invalid JSON. url={TOKEN_URL}; body={response_body}; details={}",
            format_error_details(&error)
        ))
    })
}

/// `anthropicOAuth`. Holds the `fetch` it sends token requests through.
#[derive(Clone)]
pub struct AnthropicOAuth {
    fetch: OAuthFetch,
}

impl Default for AnthropicOAuth {
    fn default() -> Self {
        Self::with_fetch(default_oauth_fetch())
    }
}

impl AnthropicOAuth {
    pub fn with_fetch(fetch: OAuthFetch) -> Self {
        Self { fetch }
    }
}

/// `anthropicOAuth`.
pub fn anthropic_oauth() -> Arc<dyn OAuthAuth> {
    Arc::new(AnthropicOAuth::default())
}

#[async_trait]
impl OAuthAuth for AnthropicOAuth {
    fn name(&self) -> &str {
        "Anthropic (Claude Pro/Max)"
    }

    fn is_subscription(&self) -> Option<bool> {
        Some(true)
    }

    async fn login(
        &self,
        interaction: ProviderAuthInteraction,
        _options: LoginOptions,
    ) -> Result<OAuthCredential> {
        let method = interaction
            .prompt(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::Select {
                    message: "Select Anthropic login method:".to_string(),
                    options: vec![
                        AuthSelectOption {
                            id: ANTHROPIC_BROWSER_LOGIN_METHOD.to_string(),
                            label: "Browser login (default)".to_string(),
                            description: None,
                        },
                        AuthSelectOption {
                            id: ANTHROPIC_COPY_CODE_LOGIN_METHOD.to_string(),
                            label: "Copy code login (headless)".to_string(),
                            description: None,
                        },
                    ],
                },
            })
            .await?;

        if method == ANTHROPIC_COPY_CODE_LOGIN_METHOD {
            return login_anthropic_copy_code(&self.fetch, &interaction).await;
        }
        if method != ANTHROPIC_BROWSER_LOGIN_METHOD {
            return Err(Error::message(format!(
                "Unknown Anthropic login method: {method}"
            )));
        }
        login_anthropic_browser(&self.fetch, &interaction).await
    }

    async fn refresh(
        &self,
        credential: OAuthCredential,
        signal: CancellationToken,
    ) -> Result<OAuthCredential> {
        refresh_token_with(&self.fetch, &credential.refresh, &signal).await
    }

    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth> {
        Ok(ModelAuth {
            api_key: Some(credential.access.clone()),
            ..Default::default()
        })
    }
}

/// Pre-1.0 entry point: run the Anthropic login with legacy callbacks.
/// Without an `on_select` callback the browser flow is used.
pub async fn login_anthropic(callbacks: OAuthLoginCallbacks) -> Result<OAuthCredential> {
    let interaction = callbacks_interaction(callbacks, ANTHROPIC_BROWSER_LOGIN_METHOD);
    AnthropicOAuth::default()
        .login(interaction, LoginOptions::default())
        .await
}

/// Pre-1.0 entry point: exchange an Anthropic refresh token.
pub async fn refresh_anthropic_token(refresh_token: &str) -> Result<OAuthCredential> {
    refresh_token_with(
        &default_oauth_fetch(),
        refresh_token,
        &CancellationToken::new(),
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use tokio::net::TcpListener;

    use super::*;
    use crate::auth::oauth::callback_server::tests::{TestInteraction, page};
    use crate::auth::oauth::fetch::FetchResponse;
    use crate::types::BoxFuture;

    /// Serializes the tests that bind (or block) the fixed callback port.
    static CALLBACK_PORT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    type Requests = Arc<Mutex<Vec<FetchRequest>>>;

    fn token_fetch(requests: Requests, access: &'static str) -> OAuthFetch {
        Arc::new(move |request, _| {
            requests.lock().push(request);
            Box::pin(async move {
                Ok(FetchResponse::json(
                    200,
                    &json!({
                        "access_token": access,
                        "refresh_token": "refresh-token",
                        "expires_in": 3600,
                    }),
                ))
            })
        })
    }

    fn body(request: &FetchRequest) -> Value {
        serde_json::from_str(request.body.as_deref().unwrap()).unwrap()
    }

    fn url_param(url: &str, name: &str) -> Option<String> {
        reqwest::Url::parse(url)
            .unwrap()
            .query_pairs()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.into_owned())
    }

    fn interaction(
        events: Arc<Mutex<Vec<AuthEvent>>>,
        prompt: impl Fn(AuthPrompt, Option<String>) -> BoxFuture<Result<String>> + Send + Sync + 'static,
    ) -> ProviderAuthInteraction {
        let seen = events.clone();
        let auth_url = move || {
            seen.lock().iter().find_map(|event| match event {
                AuthEvent::AuthUrl { url, .. } => Some(url.clone()),
                _ => None,
            })
        };
        ProviderAuthInteraction {
            interaction: Arc::new(TestInteraction {
                prompt: Arc::new(move |p| prompt(p, auth_url())),
                notify: Arc::new(move |event| events.lock().push(event)),
            }),
            signal: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn keeps_the_localhost_redirect_uri_for_manual_callback_login() {
        let _port = CALLBACK_PORT_LOCK.lock().await;
        let requests: Requests = Arc::default();
        let oauth = AnthropicOAuth::with_fetch(token_fetch(requests.clone(), "access-token"));
        let credentials = oauth
            .login(
                interaction(Arc::default(), |prompt, auth_url| {
                    Box::pin(async move {
                        match prompt.kind {
                            AuthPromptKind::Select { .. } => Ok("browser".to_string()),
                            AuthPromptKind::ManualCode { .. } => {
                                let auth_url = auth_url.unwrap();
                                let state = url_param(&auth_url, "state").unwrap();
                                let redirect = url_param(&auth_url, "redirect_uri").unwrap();
                                Ok(format!("{redirect}?code=manual-code&state={state}"))
                            }
                            other => panic!("Unexpected prompt: {other:?}"),
                        }
                    })
                }),
                LoginOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(credentials.access, "access-token");
        assert_eq!(credentials.refresh, "refresh-token");
        let requests = requests.lock();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url, TOKEN_URL);
        assert_eq!(requests[0].method, "POST");
        let body = body(&requests[0]);
        assert_eq!(body["grant_type"], "authorization_code");
        assert_eq!(body["code"], "manual-code");
        assert_eq!(body["redirect_uri"], "http://localhost:53692/callback");
    }

    #[tokio::test]
    async fn offers_browser_login_first_and_uses_the_selected_copy_code_flow() {
        let requests: Requests = Arc::default();
        let selects = Arc::new(Mutex::new(Vec::new()));
        let events: Arc<Mutex<Vec<AuthEvent>>> = Arc::default();
        let oauth = AnthropicOAuth::with_fetch(token_fetch(requests.clone(), "access-token"));
        let seen_selects = selects.clone();
        let credentials = oauth
            .login(
                interaction(events.clone(), move |prompt, auth_url| {
                    let seen_selects = seen_selects.clone();
                    Box::pin(async move {
                        match prompt.kind {
                            kind @ AuthPromptKind::Select { .. } => {
                                seen_selects.lock().push(kind);
                                Ok("copy_code".to_string())
                            }
                            AuthPromptKind::ManualCode { placeholder, .. } => {
                                assert_eq!(placeholder.as_deref(), Some("code#state"));
                                let state = url_param(&auth_url.unwrap(), "state").unwrap();
                                Ok(format!("copied-code#{state}"))
                            }
                            other => panic!("Unexpected prompt: {other:?}"),
                        }
                    })
                }),
                LoginOptions::default(),
            )
            .await
            .unwrap();

        assert_eq!(credentials.access, "access-token");
        assert_eq!(credentials.refresh, "refresh-token");
        let auth_url = events
            .lock()
            .iter()
            .find_map(|event| match event {
                AuthEvent::AuthUrl { url, .. } => Some(url.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            url_param(&auth_url, "redirect_uri").as_deref(),
            Some(COPY_CODE_REDIRECT_URI)
        );
        let requests = requests.lock();
        assert_eq!(requests.len(), 1);
        let body = body(&requests[0]);
        assert_eq!(body["grant_type"], "authorization_code");
        assert_eq!(body["code"], "copied-code");
        assert_eq!(
            body["state"].as_str(),
            url_param(&auth_url, "state").as_deref()
        );
        assert_eq!(body["redirect_uri"], COPY_CODE_REDIRECT_URI);
        assert_eq!(
            *selects.lock(),
            vec![AuthPromptKind::Select {
                message: "Select Anthropic login method:".to_string(),
                options: vec![
                    AuthSelectOption {
                        id: "browser".to_string(),
                        label: "Browser login (default)".to_string(),
                        description: None,
                    },
                    AuthSelectOption {
                        id: "copy_code".to_string(),
                        label: "Copy code login (headless)".to_string(),
                        description: None,
                    },
                ],
            }]
        );
    }

    #[tokio::test]
    async fn cancels_when_login_method_selection_is_cancelled() {
        let oauth = AnthropicOAuth::with_fetch(token_fetch(Arc::default(), "a"));
        let error = oauth
            .login(
                interaction(Arc::default(), |_, _| {
                    Box::pin(async { Err(Error::message("Login cancelled")) })
                }),
                LoginOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "Login cancelled");
    }

    #[tokio::test]
    async fn omits_scope_from_refresh_token_requests() {
        let requests: Requests = Arc::default();
        let oauth = AnthropicOAuth::with_fetch(token_fetch(requests.clone(), "new-access-token"));
        let credentials = oauth
            .refresh(
                OAuthCredential {
                    access: "old-access-token".to_string(),
                    refresh: "refresh-token".to_string(),
                    expires: 0,
                    extra: Map::new(),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(credentials.access, "new-access-token");
        assert_eq!(credentials.refresh, "refresh-token");
        assert!(credentials.expires > now_millis());
        let requests = requests.lock();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url, TOKEN_URL);
        assert_eq!(requests[0].method, "POST");
        let body = body(&requests[0]);
        assert_eq!(body["grant_type"], "refresh_token");
        assert!(!body["client_id"].as_str().unwrap().is_empty());
        assert_eq!(body["refresh_token"], "refresh-token");
        assert!(body.get("scope").is_none());
    }

    #[tokio::test]
    async fn reports_token_endpoint_failures_with_details() {
        let fetch: OAuthFetch = Arc::new(|_, _| {
            Box::pin(async { Ok(FetchResponse::new(400, "{\"error\":\"invalid_grant\"}")) })
        });
        let error = AnthropicOAuth::with_fetch(fetch)
            .refresh(OAuthCredential::default(), CancellationToken::new())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.starts_with("Anthropic token refresh request failed. url=https://platform.claude.com/v1/oauth/token; details=Error: HTTP request failed. status=400;"),
            "{error}"
        );
        assert!(error.contains("invalid_grant"), "{error}");

        let fetch: OAuthFetch =
            Arc::new(|_, _| Box::pin(async { Ok(FetchResponse::new(200, "not json")) }));
        let error = AnthropicOAuth::with_fetch(fetch)
            .refresh(OAuthCredential::default(), CancellationToken::new())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.starts_with("Anthropic token refresh returned invalid JSON. url=https://platform.claude.com/v1/oauth/token; body=not json; details="),
            "{error}"
        );
    }

    #[tokio::test]
    async fn resolves_through_the_manual_code_prompt_and_aborts_it_after_settling() {
        let _port = CALLBACK_PORT_LOCK.lock().await;
        let events: Arc<Mutex<Vec<AuthEvent>>> = Arc::default();
        let manual_signal = Arc::new(Mutex::new(None::<CancellationToken>));
        let seen = manual_signal.clone();
        let oauth = AnthropicOAuth::with_fetch(token_fetch(Arc::default(), "access"));
        let credential = oauth
            .login(
                interaction(events.clone(), move |prompt, _| {
                    let seen = seen.clone();
                    Box::pin(async move {
                        match prompt.kind {
                            AuthPromptKind::Select { .. } => Ok("browser".to_string()),
                            AuthPromptKind::ManualCode { .. } => {
                                *seen.lock() = prompt.signal.clone();
                                Ok("the-code".to_string())
                            }
                            other => panic!("Unexpected prompt: {other:?}"),
                        }
                    })
                }),
                LoginOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(credential.access, "access");
        assert!(
            events
                .lock()
                .iter()
                .any(|event| matches!(event, AuthEvent::AuthUrl { .. }))
        );
        // The prompt's signal is aborted once login settles, so UIs can dismiss it.
        assert!(manual_signal.lock().as_ref().unwrap().is_cancelled());
    }

    #[tokio::test]
    async fn completes_login_through_the_browser_callback_and_shows_the_sign_in_page() {
        let _port = CALLBACK_PORT_LOCK.lock().await;
        let requests: Requests = Arc::default();
        let oauth = AnthropicOAuth::with_fetch(token_fetch(requests.clone(), "access"));
        let callback_page = Arc::new(Mutex::new(None));
        let pages = callback_page.clone();
        let interaction = ProviderAuthInteraction {
            interaction: Arc::new(TestInteraction {
                prompt: Arc::new(|prompt| match prompt.kind {
                    AuthPromptKind::Select { .. } => Box::pin(async { Ok("browser".to_string()) }),
                    _ => crate::auth::oauth::callback_server::tests::pending_prompt(prompt),
                }),
                notify: Arc::new(move |event| {
                    let AuthEvent::AuthUrl { url, .. } = event else {
                        return;
                    };
                    let state = url_param(&url, "state").unwrap_or_default();
                    let callback =
                        format!("http://127.0.0.1:53692/callback?code=browser-code&state={state}");
                    *pages.lock() = Some(tokio::spawn(async move { page(&callback).await }));
                }),
            }),
            signal: CancellationToken::new(),
        };
        let credential = oauth
            .login(interaction, LoginOptions::default())
            .await
            .unwrap();
        assert_eq!(credential.access, "access");
        assert_eq!(body(&requests.lock()[0])["code"], "browser-code");
        let handle = callback_page.lock().take().unwrap();
        let response = handle.await.unwrap();
        assert_eq!(response.0, 200);
        assert!(response.2.contains("Signed in to Anthropic."));
    }

    #[tokio::test]
    async fn falls_back_to_manual_input_when_the_callback_port_is_busy() {
        let _port = CALLBACK_PORT_LOCK.lock().await;
        let _blocker = TcpListener::bind(("127.0.0.1", CALLBACK_PORT)).await;
        let requests: Requests = Arc::default();
        let oauth = AnthropicOAuth::with_fetch(token_fetch(requests.clone(), "access"));
        let credential = oauth
            .login(
                interaction(Arc::default(), |prompt, _| {
                    Box::pin(async move {
                        match prompt.kind {
                            AuthPromptKind::Select { .. } => Ok("browser".to_string()),
                            _ => Ok("pasted-code".to_string()),
                        }
                    })
                }),
                LoginOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(credential.access, "access");
        assert_eq!(body(&requests.lock()[0])["code"], "pasted-code");
    }

    #[tokio::test]
    async fn fails_login_when_the_provider_redirects_with_an_error() {
        let _port = CALLBACK_PORT_LOCK.lock().await;
        let interaction = ProviderAuthInteraction {
            interaction: Arc::new(TestInteraction {
                prompt: Arc::new(|prompt| match prompt.kind {
                    AuthPromptKind::Select { .. } => Box::pin(async { Ok("browser".to_string()) }),
                    _ => crate::auth::oauth::callback_server::tests::pending_prompt(prompt),
                }),
                notify: Arc::new(|event| {
                    let AuthEvent::AuthUrl { url, .. } = event else {
                        return;
                    };
                    let state = url_param(&url, "state").unwrap_or_default();
                    let callback = format!(
                        "http://127.0.0.1:53692/callback?error=access_denied&error_description=Denied&state={state}"
                    );
                    tokio::spawn(async move { page(&callback).await });
                }),
            }),
            signal: CancellationToken::new(),
        };
        let error = AnthropicOAuth::with_fetch(token_fetch(Arc::default(), "a"))
            .login(interaction, LoginOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "Anthropic authorization failed: Denied");
    }

    #[tokio::test]
    async fn rejects_pasted_input_with_a_foreign_state() {
        let oauth = AnthropicOAuth::with_fetch(token_fetch(Arc::default(), "a"));
        let error = oauth
            .login(
                interaction(Arc::default(), |prompt, _| {
                    Box::pin(async move {
                        match prompt.kind {
                            AuthPromptKind::Select { .. } => Ok("copy_code".to_string()),
                            _ => Ok("code#other-state".to_string()),
                        }
                    })
                }),
                LoginOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "OAuth state mismatch");
    }

    #[test]
    fn parses_authorization_input_forms() {
        let parsed = |input: &str| parse_authorization_input(input);
        assert_eq!(parsed("  "), AuthorizationInput::default());
        assert_eq!(
            parsed("http://localhost:53692/callback?code=c&state=s"),
            AuthorizationInput {
                code: Some("c".to_string()),
                state: Some("s".to_string()),
            }
        );
        assert_eq!(
            parsed("c#s#extra"),
            AuthorizationInput {
                code: Some("c".to_string()),
                state: Some("s".to_string()),
            }
        );
        assert_eq!(
            parsed("code=c&state=s"),
            AuthorizationInput {
                code: Some("c".to_string()),
                state: Some("s".to_string()),
            }
        );
        assert_eq!(
            parsed("plain"),
            AuthorizationInput {
                code: Some("plain".to_string()),
                state: None,
            }
        );
    }

    #[tokio::test]
    async fn resolves_stored_oauth_credentials_through_the_providers_lazy_oauth() {
        use crate::auth::{Credential, CredentialStore, InMemoryCredentialStore};
        use crate::models::{CreateModelsOptions, create_models};
        use crate::providers::anthropic::anthropic_provider;
        use crate::providers::github_copilot::github_copilot_provider;

        let store = Arc::new(InMemoryCredentialStore::new());
        let credential = Credential::OAuth(OAuthCredential {
            access: "oauth-access-token".to_string(),
            refresh: "r".to_string(),
            // Beyond get_auth()'s refresh window.
            expires: now_millis() + 10 * 60_000,
            extra: Map::new(),
        });
        store
            .modify(
                "anthropic",
                Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
                Default::default(),
            )
            .await
            .unwrap();
        let models = create_models(CreateModelsOptions {
            credentials: Some(store),
            ..Default::default()
        });
        let provider = anthropic_provider();
        let oauth = provider.auth().oauth.clone().unwrap();
        assert_eq!(oauth.name(), "Anthropic (Claude Pro/Max)");
        assert_eq!(oauth.is_subscription(), Some(true));
        assert!(provider.auth().api_key.is_some());
        models.set_provider(provider);
        let model = models.get_models(Some("anthropic"))[0].clone();
        let result = models
            .get_auth(&model.provider, Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.auth.api_key.as_deref(), Some("oauth-access-token"));
        assert_eq!(result.source.as_deref(), Some("OAuth"));

        let copilot = github_copilot_provider();
        let oauth = copilot.auth().oauth.clone().unwrap();
        assert_eq!(oauth.name(), "GitHub Copilot");
        assert_eq!(oauth.is_subscription(), Some(true));
    }

    #[tokio::test]
    async fn to_auth_derives_the_api_key_and_flags_a_subscription() {
        let oauth = anthropic_oauth();
        assert_eq!(oauth.name(), "Anthropic (Claude Pro/Max)");
        assert_eq!(oauth.is_subscription(), Some(true));
        let auth = oauth
            .to_auth(&OAuthCredential {
                access: "token".to_string(),
                refresh: "r".to_string(),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(
            auth,
            ModelAuth {
                api_key: Some("token".to_string()),
                ..Default::default()
            }
        );
    }
}
