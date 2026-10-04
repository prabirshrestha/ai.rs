//! Port of `auth/oauth/callback-server.ts`: the loopback OAuth redirect
//! handler shared by the browser sign-in flows.
//!
//! Pi uses `node:http`; this port serves the few GET requests a browser
//! redirect makes with a minimal HTTP/1.1 responder on a tokio listener.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::oauth_page::{oauth_error_html, oauth_success_html};
use crate::auth::types::{AuthPrompt, AuthPromptKind, ProviderAuthInteraction};
use crate::types::BoxFuture;
use crate::{Error, Result};

/// Finishes the sign-in with the received code (`complete`).
pub type OAuthCallbackComplete<T> = Arc<dyn Fn(String) -> BoxFuture<Result<T>> + Send + Sync>;

/// Options of [`start_oauth_callback_server`].
pub struct OAuthCallbackServerOptions<T> {
    /// Provider name used on the browser page, for example `OpenAI`.
    pub provider_name: String,
    /// Address to listen on.
    pub host: String,
    /// Port to listen on; `0` picks a free port.
    pub port: u16,
    pub path: String,
    /// Host in `redirect_uri` when it differs from `host`, for example `localhost`.
    pub redirect_host: Option<String>,
    /// Expected `state` parameter. `None` when the provider does not send one.
    pub state: Option<String>,
    /// Finishes the sign-in with the received code before the browser page is
    /// sent, so the page can show exchange failures. Return the code itself to
    /// exchange it later.
    pub complete: OAuthCallbackComplete<T>,
    pub signal: Option<CancellationToken>,
    pub timeout_ms: Option<u64>,
}

enum Outcome<T> {
    Value(Option<T>),
    /// The error, handed to the first `wait()`; later waits get its message.
    Error(Option<Error>, String),
}

struct State<T> {
    claimed: bool,
    outcome: Option<Outcome<T>>,
}

struct Shared<T> {
    provider_name: String,
    path: String,
    expected_state: Option<String>,
    complete: OAuthCallbackComplete<T>,
    state: Mutex<State<T>>,
    settled: watch::Sender<bool>,
    shutdown: CancellationToken,
}

impl<T> Shared<T> {
    fn finish(&self, outcome: Outcome<T>) {
        {
            let mut state = self.state.lock();
            if state.outcome.is_some() {
                return;
            }
            state.outcome = Some(outcome);
        }
        self.settled.send_replace(true);
    }

    fn fail(&self, error: Error) {
        let message = error.to_string();
        self.finish(Outcome::Error(Some(error), message));
    }
}

/// A running loopback callback server.
pub struct OAuthCallbackServer<T> {
    pub redirect_uri: String,
    shared: Arc<Shared<T>>,
}

impl<T: Clone + Send + 'static> OAuthCallbackServer<T> {
    /// Resolves with the result of `complete`, or `None` after `cancel()`.
    /// Rejects when the provider redirects with an error, `complete` fails,
    /// the signal aborts, or the timeout elapses.
    pub async fn wait(&self) -> Result<Option<T>> {
        let mut settled = self.shared.settled.subscribe();
        let _ = settled.wait_for(|settled| *settled).await;
        let mut state = self.shared.state.lock();
        match state.outcome.as_mut() {
            Some(Outcome::Value(value)) => Ok(value.clone()),
            Some(Outcome::Error(error, message)) => Err(error
                .take()
                .unwrap_or_else(|| Error::message(message.clone()))),
            None => Err(Error::message("OAuth callback server closed")),
        }
    }

    /// Stop waiting for the browser unless a callback is already being completed.
    pub fn cancel(&self) {
        if !self.shared.state.lock().claimed {
            self.shared.finish(Outcome::Value(None));
        }
    }
}

impl<T> OAuthCallbackServer<T> {
    pub fn close(&self) {
        self.shared
            .fail(Error::message("OAuth callback server closed"));
        self.shared.shutdown.cancel();
    }
}

impl<T> Drop for OAuthCallbackServer<T> {
    fn drop(&mut self) {
        self.close();
    }
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Bad Gateway",
    }
}

async fn send_page(stream: &mut TcpStream, status: u16, html: &str) {
    let response = format!(
        "HTTP/1.1 {status} {}\r\ncontent-type: text/html; charset=utf-8\r\ncache-control: no-store\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{html}",
        status_text(status),
        html.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;
}

/// Read the request head and return `(method, target)`.
async fn read_request_line(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 2048];
    while !bytes.windows(4).any(|window| window == b"\r\n\r\n") {
        let read = stream.read(&mut buffer).await.ok()?;
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..read]);
        if bytes.len() > 64 * 1024 {
            return None;
        }
    }
    let head = String::from_utf8_lossy(&bytes);
    let mut parts = head.lines().next()?.split_whitespace();
    Some((parts.next()?.to_string(), parts.next()?.to_string()))
}

async fn handle_request<T: Send + 'static>(shared: Arc<Shared<T>>, mut stream: TcpStream) {
    let Some((method, target)) = read_request_line(&mut stream).await else {
        return;
    };
    let url = reqwest::Url::parse("http://localhost/")
        .and_then(|base| base.join(&target))
        .ok();
    let Some(url) = url.filter(|url| method == "GET" && url.path() == shared.path) else {
        send_page(
            &mut stream,
            404,
            &oauth_error_html("Callback route not found.", None),
        )
        .await;
        return;
    };
    let param = |name: &str| {
        url.query_pairs()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.into_owned())
    };
    if let Some(expected) = &shared.expected_state
        && param("state").as_deref() != Some(expected.as_str())
    {
        send_page(&mut stream, 400, &oauth_error_html("State mismatch.", None)).await;
        return;
    }
    let claimed = {
        let state = shared.state.lock();
        state.claimed || state.outcome.is_some()
    };
    if claimed {
        let html = oauth_error_html("This sign-in has already been handled.", None);
        send_page(&mut stream, 409, &html).await;
        return;
    }
    if let Some(error) = param("error").filter(|error| !error.is_empty()) {
        let description = param("error_description").unwrap_or(error);
        let provider = &shared.provider_name;
        let html = oauth_error_html(
            &format!("{provider} authorization failed."),
            Some(&description),
        );
        send_page(&mut stream, 400, &html).await;
        shared.fail(Error::message(format!(
            "{provider} authorization failed: {description}"
        )));
        return;
    }
    let Some(code) = param("code").filter(|code| !code.is_empty()) else {
        send_page(
            &mut stream,
            400,
            &oauth_error_html("Missing authorization code.", None),
        )
        .await;
        return;
    };
    let already_handled = {
        let mut state = shared.state.lock();
        let handled = state.claimed || state.outcome.is_some();
        state.claimed = true;
        handled
    };
    if already_handled {
        let html = oauth_error_html("This sign-in has already been handled.", None);
        send_page(&mut stream, 409, &html).await;
        return;
    }
    let provider = shared.provider_name.clone();
    match (shared.complete)(code).await {
        Ok(value) => {
            let html = oauth_success_html(&format!(
                "Signed in to {provider}. You may now close this page."
            ));
            send_page(&mut stream, 200, &html).await;
            shared.finish(Outcome::Value(Some(value)));
        }
        Err(error) => {
            let html = oauth_error_html(
                &format!("{provider} sign-in failed."),
                Some(&error.to_string()),
            );
            send_page(&mut stream, 502, &html).await;
            shared.fail(error);
        }
    }
}

/// `startOAuthCallbackServer()`. Fails when the port cannot be bound (it
/// never picks another port).
pub async fn start_oauth_callback_server<T: Send + 'static>(
    options: OAuthCallbackServerOptions<T>,
) -> Result<OAuthCallbackServer<T>> {
    let signal = options.signal.clone();
    if signal.as_ref().is_some_and(CancellationToken::is_cancelled) {
        return Err(Error::message("Login cancelled"));
    }

    let listener = TcpListener::bind((options.host.as_str(), options.port)).await?;
    let port = listener.local_addr()?.port();

    let (settled, _) = watch::channel(false);
    let shared = Arc::new(Shared {
        provider_name: options.provider_name,
        path: options.path.clone(),
        expected_state: options.state,
        complete: options.complete,
        state: Mutex::new(State {
            claimed: false,
            outcome: None,
        }),
        settled,
        shutdown: CancellationToken::new(),
    });

    let accept_shared = shared.clone();
    tokio::spawn(async move {
        let shutdown = accept_shared.shutdown.clone();
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                accepted = listener.accept() => match accepted {
                    Ok((stream, _)) => {
                        tokio::spawn(handle_request(accept_shared.clone(), stream));
                    }
                    Err(error) => {
                        accept_shared.fail(error.into());
                        break;
                    }
                },
            }
        }
    });

    if let Some(signal) = signal {
        let abort_shared = Arc::downgrade(&shared);
        let shutdown = shared.shutdown.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = shutdown.cancelled() => {}
                _ = signal.cancelled() => {
                    if let Some(shared) = abort_shared.upgrade() {
                        shared.fail(Error::message("Login cancelled"));
                    }
                }
            }
        });
    }
    if let Some(timeout_ms) = options.timeout_ms {
        let timer_shared = Arc::downgrade(&shared);
        let shutdown = shared.shutdown.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = shutdown.cancelled() => {}
                _ = tokio::time::sleep(Duration::from_millis(timeout_ms)) => {
                    if let Some(shared) = timer_shared.upgrade() {
                        let message = format!("{} sign-in timed out", shared.provider_name);
                        shared.fail(Error::message(message));
                    }
                }
            }
        });
    }

    let redirect_host = options.redirect_host.unwrap_or(options.host);
    let redirect_host = if redirect_host.contains(':') {
        format!("[{redirect_host}]")
    } else {
        redirect_host
    };
    Ok(OAuthCallbackServer {
        redirect_uri: format!("http://{redirect_host}:{port}{}", options.path),
        shared,
    })
}

/// Result of [`wait_for_callback_or_manual_input`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallbackOrManualInput<T> {
    Callback(T),
    Manual(String),
}

/// Wait for the browser callback, or for the user to paste the code or
/// redirect URL when the browser cannot reach the loopback server (for
/// example over SSH). Without a callback server only the manual prompt is used.
pub async fn wait_for_callback_or_manual_input<T: Clone + Send + 'static>(
    interaction: &ProviderAuthInteraction,
    callback: Option<&OAuthCallbackServer<T>>,
    message: &str,
    placeholder: &str,
) -> Result<CallbackOrManualInput<T>> {
    let manual_abort = CancellationToken::new();
    let manual = interaction.prompt(AuthPrompt {
        signal: Some(manual_abort.clone()),
        kind: AuthPromptKind::ManualCode {
            message: message.to_string(),
            placeholder: Some(placeholder.to_string()),
        },
    });
    let result = async {
        let Some(callback) = callback else {
            return manual.await.map(CallbackOrManualInput::Manual);
        };
        tokio::pin!(manual);
        let wait = callback.wait();
        tokio::pin!(wait);
        tokio::select! {
            value = &mut wait => match value? {
                Some(value) => Ok(CallbackOrManualInput::Callback(value)),
                None => manual.await.map(CallbackOrManualInput::Manual),
            },
            input = &mut manual => {
                // A claimed callback keeps completing even after the paste.
                callback.cancel();
                // As in Pi, a failed callback wins over a failed prompt.
                let value = wait.await?;
                let input = input?;
                match value {
                    Some(value) => Ok(CallbackOrManualInput::Callback(value)),
                    None => Ok(CallbackOrManualInput::Manual(input)),
                }
            }
        }
    }
    .await;
    manual_abort.cancel();
    result
}

#[cfg(test)]
pub(crate) mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::auth::types::{AuthEvent, AuthInteraction};

    pub(crate) fn callback_url(redirect_uri: &str, params: &[(&str, &str)]) -> String {
        let mut url = reqwest::Url::parse(redirect_uri).unwrap();
        url.query_pairs_mut().extend_pairs(params);
        url.to_string()
    }

    pub(crate) async fn page(url: &str) -> (u16, Option<String>, String) {
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(url)
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get("content-type")
            .map(|value| value.to_str().unwrap().to_string());
        (status, content_type, response.text().await.unwrap())
    }

    fn options(state: Option<&str>) -> OAuthCallbackServerOptions<String> {
        OAuthCallbackServerOptions {
            provider_name: "Example".to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            path: "/callback".to_string(),
            redirect_host: None,
            state: state.map(str::to_string),
            complete: Arc::new(|code| Box::pin(async move { Ok(format!("completed:{code}")) })),
            signal: None,
            timeout_ms: None,
        }
    }

    type PromptFn = Arc<dyn Fn(AuthPrompt) -> BoxFuture<Result<String>> + Send + Sync>;

    pub(crate) struct TestInteraction {
        pub prompt: PromptFn,
        pub notify: Arc<dyn Fn(AuthEvent) + Send + Sync>,
    }

    #[async_trait]
    impl AuthInteraction for TestInteraction {
        async fn prompt(&self, prompt: AuthPrompt) -> Result<String> {
            (self.prompt)(prompt).await
        }

        fn notify(&self, event: AuthEvent) {
            (self.notify)(event)
        }
    }

    pub(crate) fn interaction(
        prompt: impl Fn(AuthPrompt) -> BoxFuture<Result<String>> + Send + Sync + 'static,
    ) -> ProviderAuthInteraction {
        ProviderAuthInteraction {
            interaction: Arc::new(TestInteraction {
                prompt: Arc::new(prompt),
                notify: Arc::new(|_| {}),
            }),
            signal: CancellationToken::new(),
        }
    }

    /// A manual prompt that stays open until its signal aborts.
    pub(crate) fn pending_prompt(prompt: AuthPrompt) -> BoxFuture<Result<String>> {
        Box::pin(async move {
            match prompt.signal {
                Some(signal) => signal.cancelled().await,
                None => std::future::pending().await,
            }
            Err(Error::message("prompt aborted"))
        })
    }

    #[tokio::test]
    async fn ignores_stray_requests_and_resolves_with_the_completed_code() {
        let server = start_oauth_callback_server(options(Some("expected-state")))
            .await
            .unwrap();
        let prefix = "http://127.0.0.1:";
        assert!(server.redirect_uri.starts_with(prefix));
        assert!(server.redirect_uri.ends_with("/callback"));

        let other = server.redirect_uri.replace("/callback", "/other");
        assert_eq!(page(&other).await.0, 404);
        let wrong_state = page(&callback_url(
            &server.redirect_uri,
            &[("code", "c"), ("state", "other")],
        ))
        .await;
        assert_eq!(wrong_state.0, 400);
        assert_eq!(wrong_state.1.as_deref(), Some("text/html; charset=utf-8"));
        assert!(wrong_state.2.contains("State mismatch."));
        let post = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .post(callback_url(
                &server.redirect_uri,
                &[("code", "c"), ("state", "expected-state")],
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(post.status().as_u16(), 404);
        let missing = page(&callback_url(
            &server.redirect_uri,
            &[("state", "expected-state")],
        ))
        .await;
        assert_eq!(missing.0, 400);

        let success = page(&callback_url(
            &server.redirect_uri,
            &[("code", "the-code"), ("state", "expected-state")],
        ))
        .await;
        assert_eq!(success.0, 200);
        assert_eq!(success.1.as_deref(), Some("text/html; charset=utf-8"));
        assert!(success.2.contains("Authentication successful"));
        assert!(success.2.contains("Signed in to Example."));
        for fill in ["#F09082", "#4D9ABF", "#F1BE58"] {
            assert!(success.2.contains(&format!("fill=\"{fill}\"")));
        }
        assert_eq!(
            server.wait().await.unwrap().as_deref(),
            Some("completed:the-code")
        );
    }

    #[tokio::test]
    async fn uses_the_redirect_host_and_skips_the_state_check_when_none_is_expected() {
        let server = start_oauth_callback_server(OAuthCallbackServerOptions {
            redirect_host: Some("localhost".to_string()),
            ..options(None)
        })
        .await
        .unwrap();
        assert!(server.redirect_uri.starts_with("http://localhost:"));
        let url = callback_url(&server.redirect_uri, &[("code", "no-state")])
            .replace("localhost", "127.0.0.1");
        assert_eq!(page(&url).await.0, 200);
        assert_eq!(
            server.wait().await.unwrap().as_deref(),
            Some("completed:no-state")
        );
    }

    #[tokio::test]
    async fn shows_completion_failures_on_the_page_and_rejects_the_wait() {
        let server = start_oauth_callback_server(OAuthCallbackServerOptions {
            complete: Arc::new(|_| {
                Box::pin(async { Err(Error::message("token exchange failed")) })
            }),
            ..options(Some("expected-state"))
        })
        .await
        .unwrap();
        let failure = page(&callback_url(
            &server.redirect_uri,
            &[("code", "c"), ("state", "expected-state")],
        ))
        .await;
        assert_eq!(failure.0, 502);
        assert!(failure.2.contains("Example sign-in failed."));
        assert!(failure.2.contains("token exchange failed"));
        assert_eq!(
            server.wait().await.unwrap_err().to_string(),
            "token exchange failed"
        );
        // Later waits see the same failure.
        assert_eq!(
            server.wait().await.unwrap_err().to_string(),
            "token exchange failed"
        );
    }

    #[tokio::test]
    async fn rejects_the_wait_when_the_provider_redirects_with_an_error() {
        let server = start_oauth_callback_server(options(Some("expected-state")))
            .await
            .unwrap();
        let failure = page(&callback_url(
            &server.redirect_uri,
            &[
                ("error", "access_denied"),
                ("error_description", "User denied access"),
                ("state", "expected-state"),
            ],
        ))
        .await;
        assert_eq!(failure.0, 400);
        assert!(failure.2.contains("User denied access"));
        assert_eq!(
            server.wait().await.unwrap_err().to_string(),
            "Example authorization failed: User denied access"
        );
    }

    #[tokio::test]
    async fn completes_only_the_first_callback() {
        let (finish, finished) = tokio::sync::oneshot::channel::<String>();
        let finished = Arc::new(Mutex::new(Some(finished)));
        let (started_tx, started) = tokio::sync::oneshot::channel::<()>();
        let started_tx = Arc::new(Mutex::new(Some(started_tx)));
        let server = start_oauth_callback_server(OAuthCallbackServerOptions {
            complete: Arc::new(move |_| {
                let finished = finished.lock().take().unwrap();
                if let Some(started) = started_tx.lock().take() {
                    let _ = started.send(());
                }
                Box::pin(async move { Ok(finished.await.unwrap()) })
            }),
            ..options(Some("expected-state"))
        })
        .await
        .unwrap();
        let url = callback_url(
            &server.redirect_uri,
            &[("code", "c"), ("state", "expected-state")],
        );
        let first = tokio::spawn({
            let url = url.clone();
            async move { page(&url).await }
        });
        started.await.unwrap();
        assert_eq!(page(&url).await.0, 409);
        // A claimed callback keeps completing even when the caller switches to manual input.
        server.cancel();
        finish.send("done".to_string()).unwrap();
        assert_eq!(first.await.unwrap().0, 200);
        assert_eq!(server.wait().await.unwrap().as_deref(), Some("done"));
    }

    #[tokio::test]
    async fn resolves_with_none_after_cancel() {
        let server = start_oauth_callback_server(options(Some("expected-state")))
            .await
            .unwrap();
        server.cancel();
        assert_eq!(server.wait().await.unwrap(), None);
        let late = page(&callback_url(
            &server.redirect_uri,
            &[("code", "c"), ("state", "expected-state")],
        ))
        .await;
        assert_eq!(late.0, 409);
    }

    #[tokio::test]
    async fn rejects_the_wait_on_abort_and_on_timeout() {
        let signal = CancellationToken::new();
        let aborted = start_oauth_callback_server(OAuthCallbackServerOptions {
            signal: Some(signal.clone()),
            ..options(Some("expected-state"))
        })
        .await
        .unwrap();
        signal.cancel();
        assert_eq!(
            aborted.wait().await.unwrap_err().to_string(),
            "Login cancelled"
        );

        let timed_out = start_oauth_callback_server(OAuthCallbackServerOptions {
            timeout_ms: Some(10),
            ..options(Some("expected-state"))
        })
        .await
        .unwrap();
        assert_eq!(
            timed_out.wait().await.unwrap_err().to_string(),
            "Example sign-in timed out"
        );

        let error = start_oauth_callback_server(OAuthCallbackServerOptions {
            signal: Some(signal),
            ..options(Some("expected-state"))
        })
        .await
        .err()
        .unwrap();
        assert_eq!(error.to_string(), "Login cancelled");
    }

    #[tokio::test]
    async fn fails_instead_of_picking_another_port_when_the_requested_port_is_taken() {
        let blocker = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = blocker.local_addr().unwrap().port();
        let error = start_oauth_callback_server(OAuthCallbackServerOptions {
            port,
            ..options(Some("expected-state"))
        })
        .await
        .err()
        .unwrap();
        assert!(
            matches!(&error, Error::Io(io) if io.kind() == std::io::ErrorKind::AddrInUse),
            "{error:?}"
        );
    }

    fn no_state_server_options() -> OAuthCallbackServerOptions<String> {
        OAuthCallbackServerOptions {
            complete: Arc::new(|code| Box::pin(async move { Ok(code) })),
            ..options(None)
        }
    }

    #[tokio::test]
    async fn wait_returns_the_browser_callback_and_aborts_the_manual_prompt() {
        let server = start_oauth_callback_server(no_state_server_options())
            .await
            .unwrap();
        let manual_signal = Arc::new(Mutex::new(None::<CancellationToken>));
        let seen = manual_signal.clone();
        let interaction = interaction(move |prompt| {
            *seen.lock() = prompt.signal.clone();
            pending_prompt(prompt)
        });
        let url = callback_url(&server.redirect_uri, &[("code", "from-browser")]);
        let browser = tokio::spawn(async move { page(&url).await });
        let result = wait_for_callback_or_manual_input(
            &interaction,
            Some(&server),
            "paste",
            &server.redirect_uri,
        )
        .await
        .unwrap();
        assert_eq!(
            result,
            CallbackOrManualInput::Callback("from-browser".to_string())
        );
        browser.await.unwrap();
        assert!(manual_signal.lock().as_ref().unwrap().is_cancelled());
    }

    #[tokio::test]
    async fn wait_returns_pasted_input_and_stops_waiting_for_the_browser() {
        let server = start_oauth_callback_server(no_state_server_options())
            .await
            .unwrap();
        let interaction = interaction(|_| Box::pin(async { Ok("pasted".to_string()) }));
        let result = wait_for_callback_or_manual_input(
            &interaction,
            Some(&server),
            "paste",
            &server.redirect_uri,
        )
        .await
        .unwrap();
        assert_eq!(result, CallbackOrManualInput::Manual("pasted".to_string()));
    }

    #[tokio::test]
    async fn wait_uses_only_the_manual_prompt_without_a_callback_server() {
        let interaction = interaction(|_| Box::pin(async { Ok("pasted".to_string()) }));
        let result = wait_for_callback_or_manual_input::<String>(
            &interaction,
            None,
            "paste",
            "http://localhost/callback",
        )
        .await
        .unwrap();
        assert_eq!(result, CallbackOrManualInput::Manual("pasted".to_string()));
    }

    #[tokio::test]
    async fn wait_propagates_manual_prompt_failures() {
        let server = start_oauth_callback_server(no_state_server_options())
            .await
            .unwrap();
        let interaction =
            interaction(|_| Box::pin(async { Err(Error::message("prompt cancelled")) }));
        let error = wait_for_callback_or_manual_input(
            &interaction,
            Some(&server),
            "paste",
            &server.redirect_uri,
        )
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "prompt cancelled");
    }
}
