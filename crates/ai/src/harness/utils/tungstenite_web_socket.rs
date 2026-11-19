//! [rust] The default [`WebSocket`], over `tokio-tungstenite`. TS adapters
//! pass the `ws` package's `WebSocket` class instead.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use bytes::Bytes;
use futures::{FutureExt, SinkExt, StreamExt};
use indexmap::IndexMap;
use parking_lot::Mutex;
use tokio::sync::{Notify, mpsc};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::{Message, Utf8Bytes};
use tokio_util::sync::CancellationToken;

use crate::harness::errors::HarnessResult;
use crate::harness::utils::bridge_web_socket::WebSocketConstructor;
use crate::harness::utils::plain_error;
use crate::harness::utils::sandbox_channel::{
    SharedWebSocket, WebSocket, WebSocketEvent, WebSocketEventHandler,
};

const OPEN: u8 = 1;
const CLOSING: u8 = 2;
const CLOSED: u8 = 3;

/// `ws`'s default closing handshake timeout.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Default)]
struct HandlerSlot {
    handler: Mutex<Option<WebSocketEventHandler>>,
    changed: Notify,
}

struct Shared {
    ready_state: AtomicU8,
    outgoing: mpsc::UnboundedSender<Message>,
    events: mpsc::UnboundedSender<WebSocketEvent>,
    slot: Arc<HandlerSlot>,
    stop: CancellationToken,
}

/// A client WebSocket over `tokio-tungstenite`. Events are buffered until a
/// handler is set and delivered in order from one task.
pub struct TungsteniteWebSocket {
    shared: Arc<Shared>,
}

impl TungsteniteWebSocket {
    /// Open a connection to `url` with extra request `headers`.
    pub async fn connect(
        url: &str,
        headers: Option<&IndexMap<String, String>>,
    ) -> HarnessResult<SharedWebSocket> {
        let mut request = url.into_client_request()?;
        for (name, value) in headers.into_iter().flatten() {
            request.headers_mut().insert(
                HeaderName::from_bytes(name.as_bytes())?,
                HeaderValue::from_str(value)?,
            );
        }
        let (stream, _) = tokio_tungstenite::connect_async(request).await?;
        let (mut sink, mut source) = stream.split();

        let (outgoing, mut outgoing_rx) = mpsc::unbounded_channel::<Message>();
        let (events, mut events_rx) = mpsc::unbounded_channel::<WebSocketEvent>();
        let shared = Arc::new(Shared {
            ready_state: AtomicU8::new(OPEN),
            outgoing,
            events,
            slot: Arc::default(),
            stop: CancellationToken::new(),
        });

        // Writer.
        let stop = shared.stop.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = stop.cancelled() => return,
                    message = outgoing_rx.recv() => {
                        let Some(message) = message else { return };
                        if sink.send(message).await.is_err() {
                            return;
                        }
                    }
                }
            }
        });

        // Reader.
        let reader = shared.clone();
        tokio::spawn(async move {
            let emit = |event| {
                let _ = reader.events.send(event);
            };
            loop {
                let message = tokio::select! {
                    _ = reader.stop.cancelled() => return,
                    message = source.next() => message,
                };
                match message {
                    Some(Ok(Message::Text(text))) => {
                        emit(WebSocketEvent::Message(text.to_string()))
                    }
                    Some(Ok(Message::Binary(bytes))) => emit(WebSocketEvent::Message(
                        String::from_utf8_lossy(&bytes).into_owned(),
                    )),
                    Some(Ok(Message::Pong(_))) => emit(WebSocketEvent::Pong),
                    Some(Ok(Message::Ping(_) | Message::Frame(_))) => {}
                    Some(Ok(Message::Close(frame))) => {
                        reader.ready_state.store(CLOSED, Ordering::SeqCst);
                        let (code, reason) = frame
                            .map(|frame| (u16::from(frame.code), frame.reason.to_string()))
                            .unwrap_or((1005, String::new()));
                        emit(WebSocketEvent::Close { code, reason });
                        reader.stop.cancel();
                        return;
                    }
                    Some(Err(error)) => {
                        reader.ready_state.store(CLOSED, Ordering::SeqCst);
                        emit(WebSocketEvent::Error(error.to_string()));
                        emit(WebSocketEvent::Close {
                            code: 1006,
                            reason: String::new(),
                        });
                        reader.stop.cancel();
                        return;
                    }
                    None => {
                        reader.ready_state.store(CLOSED, Ordering::SeqCst);
                        emit(WebSocketEvent::Close {
                            code: 1006,
                            reason: String::new(),
                        });
                        reader.stop.cancel();
                        return;
                    }
                }
            }
        });

        // Deliverer: one task, so the handler sees events in order.
        let slot = shared.slot.clone();
        tokio::spawn(async move {
            while let Some(event) = events_rx.recv().await {
                loop {
                    let handler = slot.handler.lock().clone();
                    if let Some(handler) = handler {
                        handler(event);
                        break;
                    }
                    slot.changed.notified().await;
                }
            }
        });

        Ok(Arc::new(Self { shared }))
    }

    fn mark_closed_and_stop(&self) {
        self.shared.ready_state.store(CLOSED, Ordering::SeqCst);
        self.shared.stop.cancel();
    }
}

impl WebSocket for TungsteniteWebSocket {
    fn set_event_handler(&self, handler: Option<WebSocketEventHandler>) {
        *self.shared.slot.handler.lock() = handler;
        self.shared.slot.changed.notify_one();
    }

    fn is_open(&self) -> bool {
        self.shared.ready_state.load(Ordering::SeqCst) == OPEN
    }

    fn send(&self, text: &str) -> HarnessResult<()> {
        self.shared
            .outgoing
            .send(Message::text(text))
            .map_err(|_| plain_error("WebSocket is not open: readyState 3 (CLOSED)"))
    }

    fn close(&self, code: Option<u16>, reason: Option<&str>) {
        if self
            .shared
            .ready_state
            .compare_exchange(OPEN, CLOSING, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let frame = code.map(|code| CloseFrame {
            code: CloseCode::from(code),
            reason: Utf8Bytes::from(reason.unwrap_or_default().to_string()),
        });
        let _ = self.shared.outgoing.send(Message::Close(frame));
        // Like `ws`, give up on the closing handshake after a while.
        let shared = Arc::downgrade(&self.shared);
        tokio::spawn(async move {
            tokio::time::sleep(CLOSE_TIMEOUT).await;
            if let Some(shared) = shared.upgrade()
                && !shared.stop.is_cancelled()
            {
                shared.ready_state.store(CLOSED, Ordering::SeqCst);
                shared.stop.cancel();
                let _ = shared.events.send(WebSocketEvent::Close {
                    code: 1006,
                    reason: String::new(),
                });
            }
        });
    }

    fn terminate(&self) {
        if self.shared.stop.is_cancelled() {
            return;
        }
        self.mark_closed_and_stop();
        let _ = self.shared.events.send(WebSocketEvent::Close {
            code: 1006,
            reason: String::new(),
        });
    }

    fn supports_ping(&self) -> bool {
        true
    }

    fn ping(&self) -> HarnessResult<()> {
        self.shared
            .outgoing
            .send(Message::Ping(Bytes::new()))
            .map_err(|_| plain_error("WebSocket is not open: readyState 3 (CLOSED)"))
    }
}

/// The default `webSocketConstructor`: [`TungsteniteWebSocket::connect`].
pub fn tungstenite_web_socket_constructor() -> WebSocketConstructor {
    Arc::new(|url, headers| {
        async move { TungsteniteWebSocket::connect(&url, headers.as_ref()).await }.boxed()
    })
}
