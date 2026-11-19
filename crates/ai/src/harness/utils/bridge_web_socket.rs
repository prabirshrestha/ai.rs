//! Port of `utils/bridge-web-socket.ts`: the `connect` function of a bridge
//! [`SandboxChannel`](crate::harness::utils::SandboxChannel).

use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::harness::errors::{AbortError, BoxError, HarnessError, HarnessResult};
use crate::harness::spec::{HarnessV1BridgeHello, HarnessV1PortEndpoint};
use crate::harness::utils::bridge_auth::{
    BridgeAuthPrivateKey, create_bridge_auth_proof, parse_bridge_auth_private_key,
};
use crate::harness::utils::plain_error;
use crate::harness::utils::sandbox_channel::{
    SandboxChannelConnect, SandboxChannelConnectOptions, SandboxChannelConnectReason,
    SharedWebSocket, WebSocketEvent, close_web_socket,
};
use crate::harness::utils::sleep::sleep;

/// `BridgeHello`.
pub type BridgeHello = HarnessV1BridgeHello;

/// `typeof WebSocket`: opens a socket to `url` with extra request headers,
/// resolving once it is open (`'open'`).
pub type WebSocketConstructor = Arc<
    dyn Fn(
            String,
            Option<IndexMap<String, String>>,
        ) -> BoxFuture<'static, HarnessResult<SharedWebSocket>>
        + Send
        + Sync,
>;

/// Called with each `bridge-hello`.
pub type OnBridgeHello = Arc<dyn Fn(&BridgeHello) + Send + Sync>;

/// `CreateBridgeWebSocketConnectOptions`.
#[derive(Clone)]
pub struct CreateBridgeWebSocketConnectOptions {
    /// Unauthenticated bridge WebSocket endpoint.
    pub endpoint: HarnessV1PortEndpoint,
    /// Host-only private key from `create_bridge_auth_key_pair()`. It answers
    /// the bridge's challenge on every connect and reconnect.
    pub private_key: String,
    /// [rust] Defaults to [`tungstenite_web_socket_constructor`](crate::harness::utils::tungstenite_web_socket_constructor).
    pub web_socket_constructor: Option<WebSocketConstructor>,
    /// Retry budget for the initial open (reason `Open`). `None` makes a
    /// single attempt. Reconnects always make one attempt.
    pub open_retry_timeout_ms: Option<u64>,
    /// Also cancels the initial open (not reconnects).
    pub start_abort_signal: Option<CancellationToken>,
    /// Called with the `bridge-hello` frame after every successful connect.
    pub on_hello: Option<OnBridgeHello>,
}

impl CreateBridgeWebSocketConnectOptions {
    pub fn new(endpoint: HarnessV1PortEndpoint, private_key: impl Into<String>) -> Self {
        Self {
            endpoint,
            private_key: private_key.into(),
            web_socket_constructor: None,
            open_retry_timeout_ms: None,
            start_abort_signal: None,
            on_hello: None,
        }
    }
}

const OPEN_TIMEOUT_MS: u64 = 10_000;
const HELLO_TIMEOUT_MS: u64 = 5_000;
const UNAUTHORIZED_CLOSE_CODE: u16 = 1008;

/// `BridgeWebSocketAuthenticationError`: never retried.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct BridgeWebSocketAuthenticationError(String);

/// `createBridgeWebSocketConnect(options)`: the `connect` function for a
/// bridge `SandboxChannel`.
///
/// A connection counts as established only once the socket is open, the
/// host answered the bridge's authentication challenge, and the bridge sent
/// `bridge-hello`. The host sends nothing until the bridge sends its
/// challenge, and the bridge sends nothing after `bridge-hello` until the
/// host sends `start` or `resume`, so no frame is lost in between.
pub fn create_bridge_web_socket_connect(
    options: CreateBridgeWebSocketConnectOptions,
) -> HarnessResult<SandboxChannelConnect> {
    let private_key = Arc::new(parse_bridge_auth_private_key(&options.private_key)?);
    let constructor = options
        .web_socket_constructor
        .clone()
        .unwrap_or_else(crate::harness::utils::tungstenite_web_socket_constructor);
    let options = Arc::new(options);
    Ok(Arc::new(move |connect: SandboxChannelConnectOptions| {
        let options = options.clone();
        let private_key = private_key.clone();
        let constructor = constructor.clone();
        async move {
            let is_initial_open = connect.reason == SandboxChannelConnectReason::Open;
            let mut signals = vec![connect.abort_signal.clone()];
            if is_initial_open && let Some(start) = &options.start_abort_signal {
                signals.push(start.clone());
            }
            let attempt = Attempt {
                endpoint: &options.endpoint,
                private_key: &private_key,
                constructor: &constructor,
                signals: &signals,
            };
            let (ws, hello) = match (is_initial_open, options.open_retry_timeout_ms) {
                (true, Some(retry_timeout_ms)) => {
                    open_bridge_web_socket_with_retry(&attempt, retry_timeout_ms).await?
                }
                _ => {
                    open_bridge_web_socket(&attempt, OPEN_TIMEOUT_MS, &|| HELLO_TIMEOUT_MS).await?
                }
            };
            if let Some(on_hello) = &options.on_hello {
                on_hello(&hello);
            }
            Ok(ws)
        }
        .boxed()
    }))
}

struct Attempt<'a> {
    endpoint: &'a HarnessV1PortEndpoint,
    private_key: &'a BridgeAuthPrivateKey,
    constructor: &'a WebSocketConstructor,
    signals: &'a [CancellationToken],
}

impl Attempt<'_> {
    fn aborted(&self) -> bool {
        self.signals.iter().any(CancellationToken::is_cancelled)
    }

    async fn abort_signal(&self) {
        futures::future::select_all(self.signals.iter().map(|s| s.cancelled().boxed())).await;
    }
}

async fn open_bridge_web_socket_with_retry(
    attempt: &Attempt<'_>,
    retry_timeout_ms: u64,
) -> HarnessResult<(SharedWebSocket, BridgeHello)> {
    let deadline = Instant::now() + Duration::from_millis(retry_timeout_ms);
    let remaining_ms = || {
        deadline
            .saturating_duration_since(Instant::now())
            .as_millis() as u64
    };
    let mut attempts: u64 = 0;
    let mut last_error: Option<BoxError> = None;

    while !attempt.aborted() && Instant::now() < deadline {
        attempts += 1;
        let open_timeout_ms = OPEN_TIMEOUT_MS.min(remaining_ms().max(1));
        let hello_timeout = || HELLO_TIMEOUT_MS.min(remaining_ms().max(1));
        match open_bridge_web_socket(attempt, open_timeout_ms, &hello_timeout).await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if attempt.aborted() {
                    return Err(AbortError.into());
                }
                if error.is::<BridgeWebSocketAuthenticationError>() {
                    return Err(error);
                }
                last_error = Some(error);
                let remaining = remaining_ms();
                if remaining == 0 {
                    break;
                }
                let first = attempt.signals.first();
                sleep((250 * attempts).min(1_000).min(remaining), first).await;
            }
        }
    }

    if attempt.aborted() {
        return Err(AbortError.into());
    }

    let message = format!(
        "The harness bridge did not complete the WebSocket handshake within {retry_timeout_ms}ms after {attempts} attempt(s)."
    );
    Err(match last_error {
        Some(cause) => HarnessError::with_cause(message, cause),
        None => HarnessError::new(message),
    }
    .into())
}

async fn open_bridge_web_socket(
    attempt: &Attempt<'_>,
    open_timeout_ms: u64,
    get_hello_timeout_ms: &(dyn Fn() -> u64 + Sync),
) -> HarnessResult<(SharedWebSocket, BridgeHello)> {
    if attempt.aborted() {
        return Err(AbortError.into());
    }

    let open = (attempt.constructor)(
        attempt.endpoint.url.clone(),
        attempt.endpoint.headers.clone(),
    );
    let ws = tokio::select! {
        result = tokio::time::timeout(Duration::from_millis(open_timeout_ms), open) => match result {
            Ok(result) => result?,
            Err(_) => {
                return Err(plain_error(format!(
                    "The harness bridge WebSocket did not open within {open_timeout_ms}ms."
                )));
            }
        },
        _ = attempt.abort_signal() => return Err(AbortError.into()),
    };

    let hello_timeout_ms = get_hello_timeout_ms();
    let hello_deadline = Instant::now() + Duration::from_millis(hello_timeout_ms);
    let (tx, mut rx) = mpsc::unbounded_channel();
    ws.set_event_handler(Some(Arc::new(move |event| {
        let _ = tx.send(event);
    })));

    let fail = |error: BoxError| {
        ws.set_event_handler(None);
        close_web_socket(ws.as_ref());
        Err(error)
    };

    let mut authenticated = false;
    loop {
        let event = tokio::select! {
            event = rx.recv() => event,
            _ = tokio::time::sleep_until(hello_deadline) => {
                return fail(plain_error(format!(
                    "The harness bridge did not send bridge-hello within {hello_timeout_ms}ms."
                )));
            }
            _ = attempt.abort_signal() => return fail(AbortError.into()),
        };
        match event {
            Some(WebSocketEvent::Message(text)) => {
                if !authenticated {
                    let Ok(proof) = create_bridge_auth_proof(attempt.private_key, &text) else {
                        return fail(Box::new(BridgeWebSocketAuthenticationError(
                            "The harness bridge did not send a valid authentication challenge."
                                .into(),
                        )));
                    };
                    authenticated = true;
                    if let Err(error) = ws.send(&proof) {
                        return fail(error);
                    }
                    continue;
                }
                if let Ok(hello) = serde_json::from_str::<BridgeHello>(&text) {
                    ws.set_event_handler(None);
                    return Ok((ws, hello));
                }
            }
            Some(WebSocketEvent::Close { code, .. }) => {
                return fail(if code == UNAUTHORIZED_CLOSE_CODE {
                    Box::new(BridgeWebSocketAuthenticationError(format!(
                        "The harness bridge rejected the WebSocket connection as unauthorized (code {code})."
                    )))
                } else {
                    plain_error(format!(
                        "The harness bridge closed the WebSocket (code {code}) before sending bridge-hello."
                    ))
                });
            }
            Some(WebSocketEvent::Error(message)) => return fail(plain_error(message)),
            Some(WebSocketEvent::Pong) => {}
            None => {
                return fail(plain_error(
                    "The harness bridge closed the WebSocket (code 1006) before sending bridge-hello.",
                ));
            }
        }
    }
}

#[cfg(test)]
#[path = "bridge_web_socket_tests.rs"]
mod tests;
