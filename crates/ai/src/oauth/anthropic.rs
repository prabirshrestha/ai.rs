use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::{digest, rand};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

use super::*;

const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

const ANTHROPIC_AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";

const ANTHROPIC_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";

const ANTHROPIC_CALLBACK_PORT: u16 = 53692;

const ANTHROPIC_CALLBACK_PATH: &str = "/callback";

const ANTHROPIC_REDIRECT_URI: &str = "http://localhost:53692/callback";

const ANTHROPIC_SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";

const OAUTH_TOKEN_EXPIRY_SKEW_MS: u64 = 5 * 60 * 1000;

const ANTHROPIC_OAUTH_TOKEN_TIMEOUT_MS: u64 = 30_000;

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
        extra: HashMap::new(),
    }
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
    use crate::oauth::test_support::read_http_request;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

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
        let callbacks = OAuthLoginCallbacks::builder().build();

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
        let callbacks = OAuthLoginCallbacks::builder()
            .on_auth({
                let cancellation_token = cancellation_token.clone();
                move |_| cancellation_token.cancel()
            })
            .on_prompt({
                let prompt_calls = Arc::clone(&prompt_calls);
                move |_| {
                    *prompt_calls.lock().expect("prompt lock poisoned") += 1;
                    async { Err(Error::Provider("prompt should not run".to_string())) }
                }
            })
            .cancellation_token(cancellation_token)
            .build();

        let error = tokio::time::timeout(Duration::from_secs(2), login_anthropic(callbacks))
            .await
            .expect("login should observe cancellation")
            .unwrap_err();

        assert_eq!(error.to_string(), "provider error: Login cancelled");
        assert_eq!(*prompt_calls.lock().expect("prompt lock poisoned"), 0);
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
}
