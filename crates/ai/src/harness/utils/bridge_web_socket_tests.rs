//! Port of `bridge-web-socket.test.ts`, plus a round trip over a real
//! socket.
//!
//! [rust] A socket only exists once it is open, so the TS checks that a
//! socket still connecting gets terminated have no counterpart.

use std::collections::VecDeque;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use parking_lot::Mutex;
use ring::signature;
use serde_json::{Value, json};
use tokio::sync::oneshot;

use super::*;
use crate::harness::utils::bridge_auth::{BridgeAuthKeyPair, create_bridge_auth_key_pair};
use crate::harness::utils::bridge_auth_protocol::{
    create_bridge_auth_message, parse_bridge_auth_proof_frame,
    serialize_bridge_auth_challenge_frame,
};
use crate::harness::utils::sandbox_channel::{WebSocket, WebSocketEventHandler};

static BRIDGE_AUTH: LazyLock<BridgeAuthKeyPair> =
    LazyLock::new(|| create_bridge_auth_key_pair().unwrap());

fn challenge() -> String {
    "c".repeat(64)
}

struct FakeWebSocket {
    url: String,
    headers: Option<IndexMap<String, String>>,
    handler: Mutex<Option<WebSocketEventHandler>>,
    buffered: Mutex<Vec<WebSocketEvent>>,
    sent: Mutex<Vec<String>>,
    terminated: AtomicBool,
    challenge_sent: AtomicBool,
    handler_cleared: AtomicBool,
    open: Mutex<Option<oneshot::Sender<HarnessResult<()>>>>,
}

impl FakeWebSocket {
    fn emit(&self, event: WebSocketEvent) {
        let handler = self.handler.lock().clone();
        match handler {
            Some(handler) => handler(event),
            None => self.buffered.lock().push(event),
        }
    }

    /// `emit('open')`.
    fn open(&self) {
        if let Some(open) = self.open.lock().take() {
            let _ = open.send(Ok(()));
        }
    }

    /// `emit('error', error)` before the socket opened.
    fn fail(&self, message: &str) {
        if let Some(open) = self.open.lock().take() {
            let _ = open.send(Err(plain_error(message.to_string())));
        }
    }

    fn message(&self, text: &str) {
        self.emit(WebSocketEvent::Message(text.to_string()));
    }

    fn close_with(&self, code: u16) {
        self.emit(WebSocketEvent::Close {
            code,
            reason: String::new(),
        });
    }

    fn send_challenge(&self) {
        self.challenge_sent.store(true, Ordering::SeqCst);
        self.message(&serialize_bridge_auth_challenge_frame(&challenge()));
    }

    fn send_hello(&self, capabilities: Option<Value>) {
        if !self.challenge_sent.load(Ordering::SeqCst) {
            self.send_challenge();
        }
        let mut hello = json!({ "type": "bridge-hello" });
        if let Some(capabilities) = capabilities {
            hello["capabilities"] = capabilities;
        }
        self.message(&hello.to_string());
    }
}

impl WebSocket for FakeWebSocket {
    fn set_event_handler(&self, handler: Option<WebSocketEventHandler>) {
        if handler.is_none() {
            self.handler_cleared.store(true, Ordering::SeqCst);
        }
        *self.handler.lock() = handler.clone();
        if let Some(handler) = handler {
            for event in std::mem::take(&mut *self.buffered.lock()) {
                handler(event);
            }
        }
    }

    fn is_open(&self) -> bool {
        true
    }

    fn send(&self, text: &str) -> HarnessResult<()> {
        self.sent.lock().push(text.to_string());
        Ok(())
    }

    fn close(&self, _code: Option<u16>, _reason: Option<&str>) {}

    fn terminate(&self) {
        self.terminated.store(true, Ordering::SeqCst);
    }
}

type Script = Box<dyn FnOnce(Arc<FakeWebSocket>) + Send>;

#[derive(Clone, Default)]
struct Fakes {
    sockets: Arc<Mutex<Vec<Arc<FakeWebSocket>>>>,
    scripts: Arc<Mutex<VecDeque<Script>>>,
}

impl Fakes {
    fn script(&self, script: impl FnOnce(Arc<FakeWebSocket>) + Send + 'static) {
        self.scripts.lock().push_back(Box::new(script));
    }

    fn socket(&self, index: usize) -> Arc<FakeWebSocket> {
        self.sockets.lock()[index].clone()
    }

    fn count(&self) -> usize {
        self.sockets.lock().len()
    }

    fn constructor(&self) -> WebSocketConstructor {
        let fakes = self.clone();
        Arc::new(move |url, headers| {
            let (open, opened) = oneshot::channel();
            let socket = Arc::new(FakeWebSocket {
                url,
                headers,
                handler: Mutex::default(),
                buffered: Mutex::default(),
                sent: Mutex::default(),
                terminated: AtomicBool::new(false),
                challenge_sent: AtomicBool::new(false),
                handler_cleared: AtomicBool::new(false),
                open: Mutex::new(Some(open)),
            });
            fakes.sockets.lock().push(socket.clone());
            if let Some(script) = fakes.scripts.lock().pop_front() {
                let socket = socket.clone();
                tokio::spawn(async move { script(socket) });
            }
            async move {
                match opened.await {
                    Ok(Ok(())) => Ok(socket as SharedWebSocket),
                    Ok(Err(error)) => Err(error),
                    Err(_) => futures::future::pending().await,
                }
            }
            .boxed()
        })
    }

    fn options(&self) -> CreateBridgeWebSocketConnectOptions {
        CreateBridgeWebSocketConnectOptions {
            web_socket_constructor: Some(self.constructor()),
            ..CreateBridgeWebSocketConnectOptions::new(
                HarnessV1PortEndpoint {
                    url: "wss://bridge.example/".into(),
                    headers: Some(IndexMap::from([(
                        "x-access-token".to_string(),
                        "traffic-token".to_string(),
                    )])),
                },
                BRIDGE_AUTH.private_key.clone(),
            )
        }
    }

    fn connect(&self) -> SandboxChannelConnect {
        create_bridge_web_socket_connect(self.options()).unwrap()
    }
}

fn open_with_hello(socket: Arc<FakeWebSocket>) {
    socket.open();
    socket.send_hello(Some(json!({ "experimental_userMessageResponses": true })));
}

fn connect_options(reason: SandboxChannelConnectReason) -> SandboxChannelConnectOptions {
    SandboxChannelConnectOptions {
        abort_signal: CancellationToken::new(),
        reason,
    }
}

const OPEN: SandboxChannelConnectReason = SandboxChannelConnectReason::Open;
const RECONNECT: SandboxChannelConnectReason = SandboxChannelConnectReason::Reconnect;

fn hellos() -> (Arc<Mutex<Vec<Value>>>, Option<OnBridgeHello>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    (
        seen,
        Some(Arc::new(move |hello: &BridgeHello| {
            s.lock().push(serde_json::to_value(hello).unwrap())
        })),
    )
}

#[tokio::test(start_paused = true)]
async fn resolves_once_the_socket_is_open_and_bridge_hello_arrived() {
    let fakes = Fakes::default();
    fakes.script(open_with_hello);
    let (seen, on_hello) = hellos();
    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        on_hello,
        ..fakes.options()
    })
    .unwrap();

    let ws = connect(connect_options(OPEN)).await.unwrap();
    let socket = fakes.socket(0);
    assert!(Arc::ptr_eq(&ws, &(socket.clone() as SharedWebSocket)));
    assert_eq!(socket.url, "wss://bridge.example/");
    assert_eq!(
        socket.headers,
        Some(IndexMap::from([(
            "x-access-token".to_string(),
            "traffic-token".to_string()
        )]))
    );
    assert_eq!(
        *seen.lock(),
        vec![
            json!({ "type": "bridge-hello", "capabilities": { "experimental_userMessageResponses": true } })
        ]
    );
    assert!(!socket.terminated.load(Ordering::SeqCst));
    assert!(socket.handler_cleared.load(Ordering::SeqCst));
}

#[tokio::test(start_paused = true)]
async fn ignores_messages_that_are_not_bridge_hello() {
    let fakes = Fakes::default();
    fakes.script(|socket| {
        socket.open();
        socket.send_challenge();
        socket.message("not json");
        socket.message(&json!({ "type": "text-delta" }).to_string());
    });
    let connection = tokio::spawn(fakes.connect()(connect_options(RECONNECT)));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!connection.is_finished());

    fakes.socket(0).send_hello(None);
    connection.await.unwrap().unwrap();
}

#[tokio::test(start_paused = true)]
async fn rejects_when_bridge_hello_does_not_arrive_in_time() {
    let fakes = Fakes::default();
    fakes.script(|socket| socket.open());
    let error = fakes.connect()(connect_options(RECONNECT))
        .await
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        "The harness bridge did not send bridge-hello within 5000ms."
    );
    assert!(fakes.socket(0).terminated.load(Ordering::SeqCst));
}

#[tokio::test(start_paused = true)]
async fn rejects_when_the_socket_closes_before_bridge_hello() {
    let fakes = Fakes::default();
    fakes.script(|socket| {
        socket.open();
        socket.close_with(1006);
    });
    let error = fakes.connect()(connect_options(RECONNECT))
        .await
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        "The harness bridge closed the WebSocket (code 1006) before sending bridge-hello."
    );
    assert!(fakes.socket(0).terminated.load(Ordering::SeqCst));
}

#[tokio::test(start_paused = true)]
async fn rejects_when_the_socket_does_not_open_in_time() {
    let fakes = Fakes::default();
    fakes.script(|_| {});
    let error = fakes.connect()(connect_options(RECONNECT))
        .await
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        "The harness bridge WebSocket did not open within 10000ms."
    );
}

#[tokio::test(start_paused = true)]
async fn rejects_with_an_abort_error() {
    let fakes = Fakes::default();
    let options = connect_options(RECONNECT);
    let signal = options.abort_signal.clone();
    fakes.script(move |_| signal.cancel());
    let error = fakes.connect()(options).await.err().unwrap();
    assert!(AbortError::is_instance(error.as_ref()));
}

#[tokio::test(start_paused = true)]
async fn applies_the_start_abort_signal_to_the_initial_open_only() {
    let fakes = Fakes::default();
    let start = CancellationToken::new();
    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        start_abort_signal: Some(start.clone()),
        ..fakes.options()
    })
    .unwrap();
    let s = start.clone();
    fakes.script(move |_| s.cancel());
    let error = connect(connect_options(OPEN)).await.err().unwrap();
    assert!(AbortError::is_instance(error.as_ref()));

    fakes.script(open_with_hello);
    let ws = connect(connect_options(RECONNECT)).await.unwrap();
    assert!(Arc::ptr_eq(&ws, &(fakes.socket(1) as SharedWebSocket)));
}

#[tokio::test(start_paused = true)]
async fn retries_the_initial_open_until_it_succeeds() {
    let fakes = Fakes::default();
    fakes.script(|socket| socket.fail("502"));
    fakes.script(open_with_hello);
    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        open_retry_timeout_ms: Some(20_000),
        ..fakes.options()
    })
    .unwrap();
    let ws = connect(connect_options(OPEN)).await.unwrap();
    assert!(Arc::ptr_eq(&ws, &(fakes.socket(1) as SharedWebSocket)));
}

#[tokio::test(start_paused = true)]
async fn gives_up_retrying_once_the_budget_is_used_up() {
    let fakes = Fakes::default();
    for _ in 0..10 {
        fakes.script(|socket| socket.fail("connect ECONNREFUSED 10.0.0.1:4000"));
    }
    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        open_retry_timeout_ms: Some(1_000),
        ..fakes.options()
    })
    .unwrap();
    let error = connect(connect_options(OPEN)).await.err().unwrap();
    assert!(HarnessError::is_instance(error.as_ref()));
    assert_eq!(
        error.to_string(),
        "The harness bridge did not complete the WebSocket handshake within 1000ms after 3 attempt(s)."
    );
    assert!(!error.to_string().contains("10.0.0.1:4000"));
    assert_eq!(
        error.source().unwrap().to_string(),
        "connect ECONNREFUSED 10.0.0.1:4000"
    );
}

#[tokio::test(start_paused = true)]
async fn does_not_retry_when_the_bridge_rejects_the_proof() {
    let fakes = Fakes::default();
    fakes.script(|socket| {
        socket.open();
        socket.close_with(1008);
    });
    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        open_retry_timeout_ms: Some(20_000),
        ..fakes.options()
    })
    .unwrap();
    let error = connect(connect_options(OPEN)).await.err().unwrap();
    assert_eq!(
        error.to_string(),
        "The harness bridge rejected the WebSocket connection as unauthorized (code 1008)."
    );
    assert_eq!(fakes.count(), 1);
}

#[tokio::test(start_paused = true)]
async fn caps_the_hello_timeout_by_the_remaining_retry_budget() {
    let fakes = Fakes::default();
    fakes.script(|socket| socket.open());
    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        open_retry_timeout_ms: Some(3_000),
        ..fakes.options()
    })
    .unwrap();
    let error = connect(connect_options(OPEN)).await.err().unwrap();
    assert_eq!(
        error.to_string(),
        "The harness bridge did not complete the WebSocket handshake within 3000ms after 1 attempt(s)."
    );
    assert_eq!(
        error.source().unwrap().to_string(),
        "The harness bridge did not send bridge-hello within 3000ms."
    );
    assert_eq!(fakes.count(), 1);
}

#[tokio::test(start_paused = true)]
async fn makes_a_single_attempt_when_reconnecting() {
    let fakes = Fakes::default();
    fakes.script(|socket| socket.fail("502"));
    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        open_retry_timeout_ms: Some(20_000),
        ..fakes.options()
    })
    .unwrap();
    let error = connect(connect_options(RECONNECT)).await.err().unwrap();
    assert_eq!(error.to_string(), "502");
    assert_eq!(fakes.count(), 1);
}

#[tokio::test(start_paused = true)]
async fn calls_on_hello_on_every_successful_connect() {
    let fakes = Fakes::default();
    let (seen, on_hello) = hellos();
    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        on_hello,
        ..fakes.options()
    })
    .unwrap();
    fakes.script(open_with_hello);
    fakes.script(|socket| {
        socket.open();
        socket.send_hello(None);
    });
    connect(connect_options(OPEN)).await.unwrap();
    connect(connect_options(RECONNECT)).await.unwrap();
    assert_eq!(
        *seen.lock(),
        vec![
            json!({ "type": "bridge-hello", "capabilities": { "experimental_userMessageResponses": true } }),
            json!({ "type": "bridge-hello" }),
        ]
    );
}

fn verify_proof(proof: &str) {
    let signature = parse_bridge_auth_proof_frame(proof).expect("proof frame");
    signature::UnparsedPublicKey::new(
        &signature::ED25519,
        URL_SAFE_NO_PAD.decode(&BRIDGE_AUTH.public_key).unwrap(),
    )
    .verify(
        &create_bridge_auth_message(&challenge()),
        &URL_SAFE_NO_PAD.decode(signature).unwrap(),
    )
    .unwrap();
}

#[tokio::test(start_paused = true)]
async fn answers_the_challenge_with_a_proof_before_accepting_bridge_hello() {
    let fakes = Fakes::default();
    fakes.script(open_with_hello);
    fakes.connect()(connect_options(RECONNECT)).await.unwrap();
    let sent = fakes.socket(0).sent.lock().clone();
    assert_eq!(sent.len(), 1);
    verify_proof(&sent[0]);
}

#[tokio::test(start_paused = true)]
async fn does_not_retry_when_the_bridge_sends_no_valid_challenge() {
    let fakes = Fakes::default();
    fakes.script(|socket| {
        socket.open();
        socket.message(&json!({ "type": "bridge-hello" }).to_string());
    });
    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        open_retry_timeout_ms: Some(20_000),
        ..fakes.options()
    })
    .unwrap();
    let error = connect(connect_options(OPEN)).await.err().unwrap();
    assert_eq!(
        error.to_string(),
        "The harness bridge did not send a valid authentication challenge."
    );
    assert!(!error.to_string().contains(&BRIDGE_AUTH.private_key));
    assert_eq!(fakes.count(), 1);
    assert!(fakes.socket(0).sent.lock().is_empty());
    assert!(fakes.socket(0).terminated.load(Ordering::SeqCst));
}

#[test]
fn rejects_an_invalid_private_key_before_connecting() {
    let fakes = Fakes::default();
    let error = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions {
        private_key: "invalid".into(),
        ..fakes.options()
    })
    .err()
    .unwrap();
    assert_eq!(
        error.to_string(),
        "Invalid bridge authentication private key."
    );
    assert_eq!(fakes.count(), 0);
}

/// [rust] The whole path over a real socket: handshake, then a channel.
#[tokio::test]
#[allow(clippy::result_large_err)]
async fn connects_a_channel_to_a_real_bridge_socket() {
    use crate::harness::utils::sandbox_channel::{
        SandboxChannel, SandboxChannelMessage, SandboxChannelOptions,
    };
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    #[derive(Debug, Clone, serde::Deserialize)]
    struct Frame {
        r#type: String,
        #[serde(default)]
        delta: String,
    }
    impl SandboxChannelMessage for Frame {
        fn type_name(&self) -> &str {
            &self.r#type
        }
        fn from_validation_error(_: crate::harness::provider::TypeValidationError) -> Self {
            Frame {
                r#type: "error".into(),
                delta: String::new(),
            }
        }
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (got_header, header) = oneshot::channel::<Option<String>>();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut got_header = Some(got_header);
        let callback = |request: &tokio_tungstenite::tungstenite::handshake::server::Request,
                        response| {
            let value = request
                .headers()
                .get("x-access-token")
                .map(|v| v.to_str().unwrap().to_string());
            let _ = got_header.take().unwrap().send(value);
            Ok(response)
        };
        let mut ws = tokio_tungstenite::accept_hdr_async(stream, callback)
            .await
            .unwrap();
        ws.send(Message::text(serialize_bridge_auth_challenge_frame(
            &challenge(),
        )))
        .await
        .unwrap();
        let proof = ws.next().await.unwrap().unwrap().into_text().unwrap();
        verify_proof(proof.as_str());
        ws.send(Message::text(json!({ "type": "bridge-hello" }).to_string()))
            .await
            .unwrap();
        let start = ws.next().await.unwrap().unwrap().into_text().unwrap();
        assert_eq!(start.as_str(), r#"{"type":"start"}"#);
        ws.send(Message::text(
            json!({ "type": "text-delta", "delta": "hi", "seq": 1 }).to_string(),
        ))
        .await
        .unwrap();
        // Keep the socket open until the host closes it.
        while let Some(Ok(message)) = ws.next().await {
            if message.is_close() {
                break;
            }
        }
    });

    let connect = create_bridge_web_socket_connect(CreateBridgeWebSocketConnectOptions::new(
        HarnessV1PortEndpoint {
            url: format!("ws://{address}/"),
            headers: Some(IndexMap::from([(
                "x-access-token".to_string(),
                "traffic-token".to_string(),
            )])),
        },
        BRIDGE_AUTH.private_key.clone(),
    ))
    .unwrap();
    let channel: SandboxChannel<Frame, Value> =
        SandboxChannel::new(SandboxChannelOptions::new(connect));
    channel.open(false).await.unwrap();
    assert_eq!(header.await.unwrap().as_deref(), Some("traffic-token"));

    let (tx, rx) = oneshot::channel();
    let tx = Mutex::new(Some(tx));
    let _ = channel.on("text-delta", move |frame| {
        if let Some(tx) = tx.lock().take() {
            let _ = tx.send(frame.delta.clone());
        }
    });
    channel
        .send(&json!({ "type": "start" }), Default::default())
        .unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .unwrap()
            .unwrap(),
        "hi"
    );
    assert_eq!(channel.last_seen_event_id(), 1);
    channel.close();
}
