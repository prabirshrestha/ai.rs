//! Port of `utils/sandbox-channel.ts`: the host-side typed wrapper around a
//! bridge WebSocket connection, with buffering, reconnect and keepalive.

use std::collections::{HashMap, VecDeque};
use std::error::Error as StdError;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use parking_lot::{Mutex, ReentrantMutex};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::harness::errors::{BoxError, HarnessResult};
use crate::harness::provider::{TypeValidationError, parse_json};
use crate::harness::utils::plain_error;
use crate::harness::utils::sleep::sleep;

// ─── WebSocket ─────────────────────────────────────────────────────────

/// An event of a [`WebSocket`]: the `ws` events the channel listens to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebSocketEvent {
    Message(String),
    Pong,
    Close { code: u16, reason: String },
    Error(String),
}

/// Receives a socket's events in order (`ws.on(...)`).
pub type WebSocketEventHandler = Arc<dyn Fn(WebSocketEvent) + Send + Sync>;

/// The open `WebSocket` (from the `ws` package) that a channel carries
/// frames over.
///
/// [rust] Events go to one handler instead of `on`/`off` per event type.
/// A socket buffers events while no handler is set, so none are lost between
/// a handshake and the channel wiring the socket. Implementations must not
/// call the handler from inside `close`, `send` or `ping` while holding a
/// lock that the handler could need.
pub trait WebSocket: Send + Sync {
    /// Replace the event handler. `None` buffers events until the next one.
    fn set_event_handler(&self, handler: Option<WebSocketEventHandler>);
    /// `readyState === OPEN`.
    fn is_open(&self) -> bool;
    /// `send(text)`.
    fn send(&self, text: &str) -> HarnessResult<()>;
    /// `close(code?, reason?)`.
    fn close(&self, code: Option<u16>, reason: Option<&str>);
    /// `terminate()`: drop the connection without a closing handshake.
    /// Defaults to `close()`, as for sockets without `terminate`.
    fn terminate(&self) {
        self.close(None, None);
    }
    /// Whether `ping` is supported. Sockets without it get no keepalive.
    fn supports_ping(&self) -> bool {
        false
    }
    /// `ping()`.
    fn ping(&self) -> HarnessResult<()> {
        Err(plain_error("ping is not supported"))
    }
}

/// A shared [`WebSocket`].
pub type SharedWebSocket = Arc<dyn WebSocket>;

/// `closeWebSocket({ ws })`: best-effort terminate.
pub(crate) fn close_web_socket(ws: &dyn WebSocket) {
    ws.terminate();
}

// ─── options and events ────────────────────────────────────────────────

/// `SandboxChannelDebugEvent`: connection lifecycle diagnostics. Silent
/// unless a consumer wires `on_debug`.
#[derive(Debug, Clone)]
pub enum SandboxChannelDebugEvent {
    ReconnectAttempt {
        attempt: u32,
        last_seen_event_id: u64,
    },
    Reconnected {
        attempt: u32,
        last_seen_event_id: u64,
    },
    ReconnectFailed {
        attempts: u32,
        last_seen_event_id: u64,
        cause: Arc<dyn StdError + Send + Sync>,
    },
}

impl SandboxChannelDebugEvent {
    /// The `event` field.
    pub fn event(&self) -> &'static str {
        match self {
            Self::ReconnectAttempt { .. } => "reconnect-attempt",
            Self::Reconnected { .. } => "reconnected",
            Self::ReconnectFailed { .. } => "reconnect-failed",
        }
    }
}

/// `SandboxChannelReconnectOptions`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SandboxChannelReconnectOptions {
    /// Give up reconnecting after this many milliseconds, including
    /// connection establishment and backoff delays. Default 30_000.
    pub max_elapsed_ms: Option<u64>,
    /// First backoff delay. Default 50.
    pub initial_delay_ms: Option<u64>,
    /// Backoff ceiling. Default 2_000.
    pub max_delay_ms: Option<u64>,
}

/// Why the channel is connecting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SandboxChannelConnectReason {
    /// The initial [`SandboxChannel::open`] call.
    Open,
    /// An attempt to recover from a transient disconnect.
    Reconnect,
}

/// `SandboxChannelConnectOptions`.
#[derive(Debug, Clone)]
pub struct SandboxChannelConnectOptions {
    pub abort_signal: CancellationToken,
    pub reason: SandboxChannelConnectReason,
}

/// `connect`: open a fresh WebSocket to the bridge and resolve once it is
/// ready to carry frames (after any adapter handshake). Called once by
/// `open` and again on every transient reconnect.
pub type SandboxChannelConnect = Arc<
    dyn Fn(SandboxChannelConnectOptions) -> BoxFuture<'static, HarnessResult<SharedWebSocket>>
        + Send
        + Sync,
>;

/// `onDebug`.
pub type SandboxChannelDebugHandler = Arc<dyn Fn(&SandboxChannelDebugEvent) + Send + Sync>;

/// A listener for messages of one type.
pub type SandboxChannelListener<TOut> = Arc<dyn Fn(&TOut) + Send + Sync>;

/// Removes a listener or handler. Calling it again does nothing.
pub type Unsubscribe = Box<dyn Fn() + Send + Sync>;

/// `SandboxChannelOptions<TOut>`.
///
/// [rust] `outboundSchema` is `TOut`'s `Deserialize` implementation.
pub struct SandboxChannelOptions<TOut> {
    pub connect: SandboxChannelConnect,
    pub reconnect: Option<SandboxChannelReconnectOptions>,
    /// Interval between WebSocket pings in milliseconds. Terminates an
    /// unresponsive socket after two unanswered pings so reconnection can
    /// begin. Default 20_000; 0 disables it.
    pub keep_alive_ms: Option<u64>,
    pub on_debug: Option<SandboxChannelDebugHandler>,
    /// Sink for `sandbox-log` and `debug-event` frames. They never reach the
    /// type listeners.
    pub on_diagnostic: Option<SandboxChannelListener<TOut>>,
    /// Called for `error` frames, which are still dispatched normally.
    pub on_bridge_error: Option<SandboxChannelListener<TOut>>,
    /// Seed the cursor before the first connect (cross-process attach).
    pub initial_last_seen_event_id: Option<u64>,
}

impl<TOut> SandboxChannelOptions<TOut> {
    pub fn new(connect: SandboxChannelConnect) -> Self {
        Self {
            connect,
            reconnect: None,
            keep_alive_ms: None,
            on_debug: None,
            on_diagnostic: None,
            on_bridge_error: None,
            initial_last_seen_event_id: None,
        }
    }
}

/// Options of [`SandboxChannel::send`].
pub struct SandboxChannelSendOptions {
    /// Runs after the frame is handed to the socket (not after bridge
    /// acknowledgement), asynchronously.
    pub on_sent: Option<Box<dyn FnOnce() + Send>>,
    /// When false, an unsent frame is discarded instead of queued, and its
    /// `on_sent` never runs. Default true.
    pub queue_if_disconnected: bool,
}

impl Default for SandboxChannelSendOptions {
    fn default() -> Self {
        Self {
            on_sent: None,
            queue_if_disconnected: true,
        }
    }
}

/// A frame type a [`SandboxChannel`] receives (`TOut`).
pub trait SandboxChannelMessage: DeserializeOwned + Send + Sync + 'static {
    /// The `type` discriminator.
    fn type_name(&self) -> &str;

    /// `{ type: 'error', error }` for a frame that failed validation.
    fn from_validation_error(error: TypeValidationError) -> Self;

    /// Attach the event checkpoint, the non-enumerable property TS sets.
    /// Types that never carry one keep the default no-op.
    fn set_event_checkpoint(&mut self, _checkpoint: SandboxChannelEventCheckpoint) {}

    /// The attached event checkpoint.
    fn event_checkpoint(&self) -> Option<&SandboxChannelEventCheckpoint> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PinnedCursor {
    event_id: u64,
    token: u64,
}

#[derive(Debug, Default)]
struct PinnedSuspensionCursor {
    cursor: Mutex<Option<PinnedCursor>>,
    next_token: AtomicU64,
}

/// The suspension checkpoint attached to a dispatched event.
///
/// [rust] Compares equal to every other checkpoint, so it never affects the
/// equality of the event that carries it.
#[derive(Clone)]
pub struct SandboxChannelEventCheckpoint {
    event_id: u64,
    pinned: Arc<PinnedSuspensionCursor>,
}

impl SandboxChannelEventCheckpoint {
    /// Pin the suspension cursor at this event. Returns the release.
    pub fn pin(&self) -> Box<dyn FnOnce() + Send + Sync> {
        let token = self.pinned.next_token.fetch_add(1, Ordering::SeqCst);
        *self.pinned.cursor.lock() = Some(PinnedCursor {
            event_id: self.event_id,
            token,
        });
        let pinned = self.pinned.clone();
        Box::new(move || {
            let mut cursor = pinned.cursor.lock();
            if cursor.is_some_and(|cursor| cursor.token == token) {
                *cursor = None;
            }
        })
    }
}

impl fmt::Debug for SandboxChannelEventCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxChannelEventCheckpoint")
            .field("event_id", &self.event_id)
            .finish()
    }
}

impl PartialEq for SandboxChannelEventCheckpoint {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// `pinSandboxChannelEventCheckpoint(event)`: pins the suspension cursor at
/// a dispatched event. Returns the release, or `None` for an event without a
/// checkpoint.
pub fn pin_sandbox_channel_event_checkpoint<T: SandboxChannelMessage>(
    event: &T,
) -> Option<Box<dyn FnOnce() + Send + Sync>> {
    event
        .event_checkpoint()
        .map(SandboxChannelEventCheckpoint::pin)
}

// ─── abort controllers ─────────────────────────────────────────────────

const CONNECTION_ABORTED: &str = "SandboxChannel connection aborted";
const RECONNECT_DEADLINE_EXPIRED: &str = "Reconnect deadline expired";

/// `AbortController`: a token plus the reason it was aborted with.
#[derive(Clone)]
struct AbortController {
    token: CancellationToken,
    reason: Arc<Mutex<Option<String>>>,
    parent_reason: Option<Arc<Mutex<Option<String>>>>,
}

impl AbortController {
    fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            reason: Arc::default(),
            parent_reason: None,
        }
    }

    /// A controller that is also aborted, with the same reason, when this
    /// one is.
    fn child(&self) -> Self {
        Self {
            token: self.token.child_token(),
            reason: Arc::default(),
            parent_reason: Some(self.reason.clone()),
        }
    }

    fn abort(&self, reason: &str) {
        self.reason.lock().get_or_insert_with(|| reason.to_string());
        self.token.cancel();
    }

    fn is_aborted(&self) -> bool {
        self.token.is_cancelled()
    }

    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.reason, &other.reason)
    }

    /// `getAbortReason({ abortSignal })`.
    fn reason_error(&self) -> BoxError {
        let reason = self
            .reason
            .lock()
            .clone()
            .or_else(|| self.parent_reason.as_ref().and_then(|p| p.lock().clone()))
            .unwrap_or_else(|| CONNECTION_ABORTED.to_string());
        plain_error(reason)
    }
}

// ─── state ─────────────────────────────────────────────────────────────

struct BufferedEvent<TOut> {
    message: Arc<TOut>,
    listeners: Option<Vec<SandboxChannelListener<TOut>>>,
}

struct PendingSend {
    text: String,
    on_sent: Option<Box<dyn FnOnce() + Send>>,
}

type CloseHandler = Arc<dyn Fn(u16, &str) + Send + Sync>;
type ReconnectHandler = Arc<dyn Fn() + Send + Sync>;

struct State<TOut> {
    listeners: IndexMap<String, Vec<(u64, SandboxChannelListener<TOut>)>>,
    next_handler_id: u64,
    buffered: Vec<BufferedEvent<TOut>>,
    buffered_by_type: HashMap<String, VecDeque<Arc<TOut>>>,
    buffered_offset: usize,
    flushing_buffered: bool,
    buffered_flush_scheduled: bool,
    listener_attachment_depth: usize,
    on_close_handlers: Vec<CloseHandler>,
    on_reconnect_handlers: Vec<(u64, ReconnectHandler)>,
    stop_keepalive: Option<CancellationToken>,
    ws: Option<(u64, SharedWebSocket)>,
    next_socket_id: u64,
    connected: bool,
    /// Host has begun teardown; suppresses reconnect so a bridge-side close
    /// finalises.
    closing: bool,
    /// Host has gracefully suspended; inbound frames are ignored so the
    /// cursor stops at the last delivered event.
    suspended: bool,
    /// Fully torn down; `send` fails and `on_close` has fired.
    terminal: bool,
    last_seen_event_id: u64,
    pending_sends: VecDeque<PendingSend>,
    active_connect: Option<AbortController>,
    reconnect_abort: Option<AbortController>,
}

impl<TOut> State<TOut> {
    fn listeners_of(&self, r#type: &str) -> Option<Vec<SandboxChannelListener<TOut>>> {
        self.listeners
            .get(r#type)
            .filter(|set| !set.is_empty())
            .map(|set| set.iter().map(|(_, l)| l.clone()).collect())
    }

    fn has_listeners(&self) -> bool {
        self.listeners.values().any(|set| !set.is_empty())
    }
}

enum Work {
    Incoming(String),
    FinalizeClose(u16, String),
    Reconnect,
    Suspend,
}

/// Per-socket wiring: drop detection and keepalive.
struct Wiring {
    id: u64,
    dropped: AtomicBool,
    keepalive: CancellationToken,
    unanswered_pings: AtomicU32,
}

struct Inner<TOut: SandboxChannelMessage> {
    state: Mutex<State<TOut>>,
    /// Serialises listener calls, so events reach listeners in order.
    delivery: ReentrantMutex<()>,
    pinned: Arc<PinnedSuspensionCursor>,
    connect_thunk: SandboxChannelConnect,
    on_debug: Option<SandboxChannelDebugHandler>,
    on_diagnostic: Option<SandboxChannelListener<TOut>>,
    on_bridge_error: Option<SandboxChannelListener<TOut>>,
    max_elapsed: Duration,
    initial_delay_ms: f64,
    max_delay_ms: f64,
    keep_alive: Duration,
    work: mpsc::UnboundedSender<Work>,
    microtasks: mpsc::UnboundedSender<Box<dyn FnOnce() + Send>>,
    runtime: tokio::runtime::Handle,
    weak: Weak<Inner<TOut>>,
}

/// `SandboxChannel<TOut, TIn>`: host-side typed wrapper around the bridge
/// WebSocket connection.
///
/// Buffers inbound messages in arrival order while listeners attach, so
/// callers do not miss or reorder early frames. Registering a listener
/// drains the ordered prefix right away. Event types that stay unhandled
/// shortly after are retained per type so they cannot block subscribed
/// types. Use [`SandboxChannel::begin_listener_attachment`] to attach a
/// related set of listeners atomically. Inbound dispatch is serialised, so a
/// close that arrives right after the final `finish` fires close handlers
/// only after that message was dispatched.
///
/// Survives transient disconnects: on an unexpected drop the channel
/// re-invokes `connect`, re-wires the new socket, and asks the bridge to
/// replay everything past `last_seen_event_id`. `on_close` fires only after
/// a host-initiated close or once the reconnect budget is exhausted.
///
/// [rust] Must be created inside a Tokio runtime. Cloning shares the
/// channel.
pub struct SandboxChannel<TOut: SandboxChannelMessage, TIn = Value> {
    inner: Arc<Inner<TOut>>,
    _in: PhantomData<fn(TIn)>,
}

impl<TOut: SandboxChannelMessage, TIn> Clone for SandboxChannel<TOut, TIn> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _in: PhantomData,
        }
    }
}

impl<TOut: SandboxChannelMessage, TIn: Serialize> SandboxChannel<TOut, TIn> {
    /// `new SandboxChannel(options)`.
    pub fn new(options: SandboxChannelOptions<TOut>) -> Self {
        let runtime = tokio::runtime::Handle::current();
        let (work_tx, mut work_rx) = mpsc::unbounded_channel::<Work>();
        let (microtask_tx, mut microtask_rx) =
            mpsc::unbounded_channel::<Box<dyn FnOnce() + Send>>();
        let reconnect = options.reconnect.unwrap_or_default();
        let inner = Arc::new_cyclic(|weak| Inner {
            state: Mutex::new(State {
                listeners: IndexMap::new(),
                next_handler_id: 0,
                buffered: Vec::new(),
                buffered_by_type: HashMap::new(),
                buffered_offset: 0,
                flushing_buffered: false,
                buffered_flush_scheduled: false,
                listener_attachment_depth: 0,
                on_close_handlers: Vec::new(),
                on_reconnect_handlers: Vec::new(),
                stop_keepalive: None,
                ws: None,
                next_socket_id: 0,
                connected: false,
                closing: false,
                suspended: false,
                terminal: false,
                last_seen_event_id: options.initial_last_seen_event_id.unwrap_or(0),
                pending_sends: VecDeque::new(),
                active_connect: None,
                reconnect_abort: None,
            }),
            delivery: ReentrantMutex::new(()),
            pinned: Arc::default(),
            connect_thunk: options.connect,
            on_debug: options.on_debug,
            on_diagnostic: options.on_diagnostic,
            on_bridge_error: options.on_bridge_error,
            max_elapsed: Duration::from_millis(reconnect.max_elapsed_ms.unwrap_or(30_000)),
            initial_delay_ms: reconnect.initial_delay_ms.unwrap_or(50) as f64,
            max_delay_ms: reconnect.max_delay_ms.unwrap_or(2_000) as f64,
            keep_alive: Duration::from_millis(options.keep_alive_ms.unwrap_or(20_000)),
            work: work_tx,
            microtasks: microtask_tx,
            runtime: runtime.clone(),
            weak: weak.clone(),
        });

        let weak = Arc::downgrade(&inner);
        runtime.spawn(async move {
            while let Some(work) = work_rx.recv().await {
                let Some(inner) = weak.upgrade() else { break };
                inner.run(work).await;
            }
        });
        runtime.spawn(async move {
            while let Some(task) = microtask_rx.recv().await {
                task();
            }
        });

        Self {
            inner,
            _in: PhantomData,
        }
    }

    /// The highest bridge event `seq` observed. Persist it so a future
    /// process can seed `initial_last_seen_event_id` and attach.
    pub fn last_seen_event_id(&self) -> u64 {
        self.inner.state.lock().last_seen_event_id
    }

    /// `open({ resume })`: establish the initial connection, in a single
    /// attempt. With `resume`, send `{ type: 'resume', lastSeenEventId }`
    /// after the socket opens, so the bridge replays everything past the
    /// seeded cursor.
    pub async fn open(&self, resume: bool) -> HarnessResult<()> {
        self.inner.open(resume).await
    }

    /// `on(type, listener)`: subscribe to messages of one type.
    pub fn on(
        &self,
        r#type: &str,
        listener: impl Fn(&TOut) + Send + Sync + 'static,
    ) -> Unsubscribe {
        self.inner.on(r#type, Arc::new(listener))
    }

    /// `beginListenerAttachment()`: hold buffered delivery while a consumer
    /// attaches a related set of listeners, across asynchronous boundaries.
    /// Call the returned function once registration is complete.
    pub fn begin_listener_attachment(&self) -> Unsubscribe {
        self.inner.state.lock().listener_attachment_depth += 1;
        let finished = AtomicBool::new(false);
        let weak = Arc::downgrade(&self.inner);
        Box::new(move || {
            if finished.swap(true, Ordering::SeqCst) {
                return;
            }
            let Some(inner) = weak.upgrade() else { return };
            let types: Vec<String> = {
                let mut state = inner.state.lock();
                state.listener_attachment_depth -= 1;
                if state.listener_attachment_depth > 0 {
                    return;
                }
                state.listeners.keys().cloned().collect()
            };
            for r#type in types {
                inner.flush_buffered_type(&r#type);
            }
            inner.flush_buffered(false);
            inner.schedule_selective_buffered_flush();
        })
    }

    /// `onClose(handler)`.
    pub fn on_close(&self, handler: impl Fn(u16, &str) + Send + Sync + 'static) {
        self.inner
            .state
            .lock()
            .on_close_handlers
            .push(Arc::new(handler));
    }

    /// `onReconnect(handler)`.
    pub fn on_reconnect(&self, handler: impl Fn() + Send + Sync + 'static) -> Unsubscribe {
        let id = {
            let mut state = self.inner.state.lock();
            state.next_handler_id += 1;
            let id = state.next_handler_id;
            state.on_reconnect_handlers.push((id, Arc::new(handler)));
            id
        };
        let weak = Arc::downgrade(&self.inner);
        Box::new(move || {
            if let Some(inner) = weak.upgrade() {
                inner
                    .state
                    .lock()
                    .on_reconnect_handlers
                    .retain(|(handler_id, _)| *handler_id != id);
            }
        })
    }

    /// `send(message, { onSent, queueIfDisconnected })`: queue the frame
    /// while disconnected. Reconnect handlers run before the `on_sent`
    /// notifications of frames flushed on that connection.
    pub fn send(&self, message: &TIn, options: SandboxChannelSendOptions) -> HarnessResult<()> {
        let value = serde_json::to_value(message)?;
        if self.inner.state.lock().terminal {
            let r#type = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            return Err(plain_error(format!(
                "SandboxChannel: cannot send {type} — channel is closed."
            )));
        }
        self.inner.raw_send(
            value.to_string(),
            options.on_sent,
            options.queue_if_disconnected,
        )
    }

    /// `beginClose()`: the next socket close is terminal rather than a
    /// reconnect. Call before sending a `stop` / `destroy` frame whose
    /// completion closes the bridge socket.
    pub fn begin_close(&self) {
        self.inner.begin_close();
    }

    /// `close()`.
    pub fn close(&self) {
        if self.inner.state.lock().terminal {
            return;
        }
        self.inner.begin_close();
        if let Some(ws) = self.inner.current_ws() {
            ws.close(None, None);
        }
        self.inner
            .enqueue(Work::FinalizeClose(1000, "closed".into()));
    }

    /// `suspend()`: stop processing inbound frames, drain the frames already
    /// queued for dispatch, then close the socket with reason `suspended`.
    /// Resolves with the final cursor, or the pinned checkpoint's event when
    /// one is pinned.
    pub async fn suspend(&self) -> u64 {
        let pinned = self
            .inner
            .pinned
            .cursor
            .lock()
            .map(|cursor| cursor.event_id);
        let (tx, rx) = oneshot::channel();
        {
            let mut state = self.inner.state.lock();
            if state.terminal {
                return pinned.unwrap_or(state.last_seen_event_id);
            }
            state.suspended = true;
        }
        self.inner.begin_close();
        let tx = Mutex::new(Some(tx));
        let weak = Arc::downgrade(&self.inner);
        self.on_close(move |_, _| {
            let last_seen = weak
                .upgrade()
                .map(|inner| inner.state.lock().last_seen_event_id)
                .unwrap_or(0);
            if let Some(tx) = tx.lock().take() {
                let _ = tx.send(pinned.unwrap_or(last_seen));
            }
        });
        self.inner.enqueue(Work::Suspend);
        match rx.await {
            Ok(cursor) => cursor,
            Err(_) => pinned.unwrap_or_else(|| self.last_seen_event_id()),
        }
    }

    /// `isClosed()`.
    pub fn is_closed(&self) -> bool {
        self.inner.state.lock().terminal
    }
}

impl<TOut: SandboxChannelMessage> Inner<TOut> {
    fn enqueue(&self, work: Work) {
        let _ = self.work.send(work);
    }

    fn current_ws(&self) -> Option<SharedWebSocket> {
        self.state.lock().ws.as_ref().map(|(_, ws)| ws.clone())
    }

    async fn run(&self, work: Work) {
        match work {
            Work::Incoming(text) => self.handle_incoming(&text),
            Work::FinalizeClose(code, reason) => self.finalize_close(code, &reason),
            Work::Reconnect => self.reconnect_loop().await,
            Work::Suspend => {
                if let Some(ws) = self.current_ws() {
                    // The bridge reads this reason to keep a suspended turn
                    // running instead of treating it as a disconnected host.
                    ws.close(Some(1000), Some("suspended"));
                }
                self.finalize_close(1000, "suspended");
            }
        }
    }

    async fn open(&self, resume: bool) -> HarnessResult<()> {
        let abort = AbortController::new();
        {
            let mut state = self.state.lock();
            if state.terminal {
                return Err(plain_error("SandboxChannel: cannot open a closed channel."));
            }
            state.active_connect = Some(abort.clone());
        }
        let result = async {
            let ws = self
                .connect(&abort, SandboxChannelConnectReason::Open)
                .await?;
            self.wire(ws);
            if resume {
                let resume = self.resume_frame();
                self.raw_send(resume, None, true)?;
            }
            Ok(())
        }
        .await;
        let mut state = self.state.lock();
        if state
            .active_connect
            .as_ref()
            .is_some_and(|a| a.same(&abort))
        {
            state.active_connect = None;
        }
        result
    }

    fn resume_frame(&self) -> String {
        serde_json::json!({
            "type": "resume",
            "lastSeenEventId": self.state.lock().last_seen_event_id,
        })
        .to_string()
    }

    fn on(&self, r#type: &str, listener: SandboxChannelListener<TOut>) -> Unsubscribe {
        let id = {
            let mut state = self.state.lock();
            state.next_handler_id += 1;
            let id = state.next_handler_id;
            state
                .listeners
                .entry(r#type.to_string())
                .or_default()
                .push((id, listener));
            id
        };
        let unsubscribe: Unsubscribe = {
            let weak = self.weak.clone();
            let r#type = r#type.to_string();
            Box::new(move || {
                if let Some(inner) = weak.upgrade()
                    && let Some(set) = inner.state.lock().listeners.get_mut(&r#type)
                {
                    set.retain(|(listener_id, _)| *listener_id != id);
                }
            })
        };

        self.capture_buffered_listeners(r#type);
        if self.state.lock().listener_attachment_depth > 0 {
            return unsubscribe;
        }
        self.flush_buffered_type(r#type);
        self.flush_buffered(false);
        self.schedule_selective_buffered_flush();
        unsubscribe
    }

    fn begin_close(&self) {
        let stop = {
            let mut state = self.state.lock();
            state.closing = true;
            state.stop_keepalive.take()
        };
        if let Some(stop) = stop {
            stop.cancel();
        }
        self.abort_active_connect();
    }

    // ─── internals ─────────────────────────────────────────────────────

    /// Make `ws` the current socket and listen to it.
    fn wire(&self, ws: SharedWebSocket) {
        let wiring = {
            let mut state = self.state.lock();
            state.next_socket_id += 1;
            let wiring = Arc::new(Wiring {
                id: state.next_socket_id,
                dropped: AtomicBool::new(false),
                keepalive: CancellationToken::new(),
                unanswered_pings: AtomicU32::new(0),
            });
            state.stop_keepalive = Some(wiring.keepalive.clone());
            state.ws = Some((wiring.id, ws.clone()));
            state.connected = true;
            wiring
        };

        self.keepalive(&ws, &wiring);

        let weak = self.weak.clone();
        let handler_wiring = wiring.clone();
        ws.set_event_handler(Some(Arc::new(move |event| {
            let Some(inner) = weak.upgrade() else { return };
            match event {
                WebSocketEvent::Message(text) => {
                    // Once suspended, ignore further frames so the cursor
                    // freezes at the last delivered event.
                    if inner.state.lock().suspended {
                        return;
                    }
                    inner.enqueue(Work::Incoming(text));
                }
                WebSocketEvent::Close { code, reason } => {
                    inner.on_drop(&handler_wiring, code, reason)
                }
                WebSocketEvent::Error(_) => {
                    inner.on_drop(&handler_wiring, 1006, "socket error".into())
                }
                WebSocketEvent::Pong => handler_wiring.unanswered_pings.store(0, Ordering::SeqCst),
            }
        })));
    }

    fn on_drop(&self, wiring: &Wiring, code: u16, reason: String) {
        if wiring.dropped.swap(true, Ordering::SeqCst) {
            return;
        }
        wiring.keepalive.cancel();
        let closing = {
            let mut state = self.state.lock();
            if state.ws.as_ref().map(|(id, _)| *id) != Some(wiring.id) {
                return;
            }
            state.connected = false;
            state.closing
        };
        if closing {
            self.enqueue(Work::FinalizeClose(code, reason));
        } else {
            self.enqueue(Work::Reconnect);
        }
    }

    fn keepalive(&self, ws: &SharedWebSocket, wiring: &Arc<Wiring>) {
        if self.keep_alive.is_zero() || !ws.supports_ping() {
            return;
        }
        let period = self.keep_alive;
        let weak = self.weak.clone();
        let ws = ws.clone();
        let wiring = wiring.clone();
        self.runtime.spawn(async move {
            let mut interval = tokio::time::interval_at(Instant::now() + period, period);
            loop {
                tokio::select! {
                    _ = wiring.keepalive.cancelled() => return,
                    _ = interval.tick() => {}
                }
                if !ws.is_open() {
                    continue;
                }
                let timed_out = wiring.unanswered_pings.load(Ordering::SeqCst) >= 2 || {
                    wiring.unanswered_pings.fetch_add(1, Ordering::SeqCst);
                    ws.ping().is_err()
                };
                if timed_out {
                    close_web_socket(ws.as_ref());
                    if let Some(inner) = weak.upgrade() {
                        inner.on_drop(&wiring, 1006, "heartbeat timeout".into());
                    }
                    return;
                }
            }
        });
    }

    async fn reconnect_loop(&self) {
        let deadline = Instant::now() + self.max_elapsed;
        let mut attempt: u32 = 0;
        let mut delay = self.initial_delay_ms;
        let reconnect_abort = AbortController::new();
        {
            let mut state = self.state.lock();
            if state.terminal || state.closing {
                return;
            }
            state.reconnect_abort = Some(reconnect_abort.clone());
        }

        loop {
            {
                let state = self.state.lock();
                if state.terminal || state.closing {
                    break;
                }
            }
            let now = Instant::now();
            if now >= deadline {
                self.fail_reconnect(attempt, plain_error(RECONNECT_DEADLINE_EXPIRED));
                break;
            }
            let remaining = deadline - now;

            attempt += 1;
            self.debug(SandboxChannelDebugEvent::ReconnectAttempt {
                attempt,
                last_seen_event_id: self.state.lock().last_seen_event_id,
            });
            let abort = reconnect_abort.child();
            self.state.lock().active_connect = Some(abort.clone());
            let timeout = {
                let abort = abort.clone();
                self.runtime.spawn(async move {
                    tokio::time::sleep(remaining).await;
                    abort.abort(RECONNECT_DEADLINE_EXPIRED);
                })
            };

            let result = self
                .connect(&abort, SandboxChannelConnectReason::Reconnect)
                .await;
            let attempt_result: Result<bool, BoxError> = match result {
                Ok(ws) => {
                    let stop = {
                        let state = self.state.lock();
                        state.terminal || state.closing
                    };
                    if stop {
                        close_web_socket(ws.as_ref());
                        Ok(true)
                    } else {
                        self.wire(ws);
                        // Ask the bridge to replay everything we have not
                        // seen, then flush frames produced while disconnected.
                        let resume = self.resume_frame();
                        let (on_sents, flushed) = match self.raw_send(resume, None, true) {
                            Ok(()) => self.flush_pending(),
                            Err(error) => (Vec::new(), Err(error)),
                        };
                        if flushed.is_ok() {
                            let handlers: Vec<ReconnectHandler> = self
                                .state
                                .lock()
                                .on_reconnect_handlers
                                .iter()
                                .map(|(_, h)| h.clone())
                                .collect();
                            for handler in handlers {
                                handler();
                            }
                        }
                        for on_sent in on_sents {
                            let _ = self.microtasks.send(on_sent);
                        }
                        flushed.map(|()| {
                            self.debug(SandboxChannelDebugEvent::Reconnected {
                                attempt,
                                last_seen_event_id: self.state.lock().last_seen_event_id,
                            });
                            true
                        })
                    }
                }
                Err(error) => Err(error),
            };

            timeout.abort();
            {
                let mut state = self.state.lock();
                if state
                    .active_connect
                    .as_ref()
                    .is_some_and(|a| a.same(&abort))
                {
                    state.active_connect = None;
                }
            }

            match attempt_result {
                Ok(_) => break,
                Err(cause) => {
                    {
                        let state = self.state.lock();
                        if state.terminal || state.closing || reconnect_abort.is_aborted() {
                            break;
                        }
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        self.fail_reconnect(attempt, cause);
                        break;
                    }
                    let remaining = deadline - now;
                    let wait = Duration::from_millis(delay as u64).min(remaining);
                    sleep(wait.as_millis() as u64, Some(&reconnect_abort.token)).await;
                    delay = (delay * 1.5).min(self.max_delay_ms);
                }
            }
        }

        let mut state = self.state.lock();
        if state
            .reconnect_abort
            .as_ref()
            .is_some_and(|a| a.same(&reconnect_abort))
        {
            state.reconnect_abort = None;
        }
    }

    /// `awaitWebSocketConnection`: the connection, unless `abort` fires
    /// first, in which case a socket that still arrives is closed.
    async fn connect(
        &self,
        abort: &AbortController,
        reason: SandboxChannelConnectReason,
    ) -> HarnessResult<SharedWebSocket> {
        let connection = (self.connect_thunk)(SandboxChannelConnectOptions {
            abort_signal: abort.token.clone(),
            reason,
        });
        let mut handle = self.runtime.spawn(connection);
        let close_late = |handle: tokio::task::JoinHandle<HarnessResult<SharedWebSocket>>| {
            self.runtime.spawn(async move {
                if let Ok(Ok(ws)) = handle.await {
                    close_web_socket(ws.as_ref());
                }
            });
        };
        if abort.is_aborted() {
            close_late(handle);
            return Err(abort.reason_error());
        }
        tokio::select! {
            result = &mut handle => {
                let ws = result.map_err(|error| plain_error(error.to_string()))??;
                if abort.is_aborted() {
                    close_web_socket(ws.as_ref());
                    return Err(abort.reason_error());
                }
                Ok(ws)
            }
            _ = abort.token.cancelled() => {
                close_late(handle);
                Err(abort.reason_error())
            }
        }
    }

    fn abort_active_connect(&self) {
        let (active, reconnect) = {
            let mut state = self.state.lock();
            (state.active_connect.take(), state.reconnect_abort.clone())
        };
        if let Some(active) = active {
            active.abort(CONNECTION_ABORTED);
        }
        if let Some(reconnect) = reconnect {
            reconnect.abort(CONNECTION_ABORTED);
        }
    }

    fn fail_reconnect(&self, attempts: u32, cause: BoxError) {
        self.finalize_close(1006, "reconnect failed");
        self.debug(SandboxChannelDebugEvent::ReconnectFailed {
            attempts,
            last_seen_event_id: self.state.lock().last_seen_event_id,
            cause: Arc::from(cause),
        });
    }

    fn debug(&self, event: SandboxChannelDebugEvent) {
        if let Some(on_debug) = &self.on_debug {
            on_debug(&event);
        }
    }

    fn raw_send(
        &self,
        text: String,
        on_sent: Option<Box<dyn FnOnce() + Send>>,
        queue_if_disconnected: bool,
    ) -> HarnessResult<()> {
        let ws = {
            let state = self.state.lock();
            if state.connected {
                state.ws.as_ref().map(|(_, ws)| ws.clone())
            } else {
                None
            }
        };
        if let Some(ws) = ws.filter(|ws| ws.is_open()) {
            ws.send(&text)?;
            if let Some(on_sent) = on_sent {
                let _ = self.microtasks.send(on_sent);
            }
        } else if queue_if_disconnected {
            self.state
                .lock()
                .pending_sends
                .push_back(PendingSend { text, on_sent });
        }
        Ok(())
    }

    /// `flushPending()`. Returns the `on_sent` callbacks of the frames it
    /// sent, for the caller to schedule.
    #[allow(clippy::type_complexity)]
    fn flush_pending(&self) -> (Vec<Box<dyn FnOnce() + Send>>, HarnessResult<()>) {
        let mut on_sents = Vec::new();
        loop {
            let next = {
                let state = self.state.lock();
                match (&state.ws, state.pending_sends.front()) {
                    (Some((_, ws)), Some(pending)) if state.connected => {
                        Some((ws.clone(), pending.text.clone()))
                    }
                    _ => None,
                }
            };
            let Some((ws, text)) = next else { break };
            if !ws.is_open() {
                break;
            }
            if let Err(error) = ws.send(&text) {
                return (on_sents, Err(error));
            }
            if let Some(PendingSend {
                on_sent: Some(on_sent),
                ..
            }) = self.state.lock().pending_sends.pop_front()
            {
                on_sents.push(on_sent);
            }
        }
        (on_sents, Ok(()))
    }

    fn handle_incoming(&self, text: &str) {
        let raw = serde_json::from_str::<Value>(text).ok();
        let seq = raw
            .as_ref()
            .and_then(|value| value.get("seq"))
            .and_then(Value::as_u64);

        match parse_json::<TOut>(raw.unwrap_or(Value::Null)) {
            Ok(mut message) => {
                if let Some(seq) = seq {
                    message.set_event_checkpoint(SandboxChannelEventCheckpoint {
                        event_id: seq,
                        pinned: self.pinned.clone(),
                    });
                }
                self.dispatch(Arc::new(message));
            }
            Err(error) => self.dispatch(Arc::new(TOut::from_validation_error(error))),
        }

        if let Some(seq) = seq {
            let mut state = self.state.lock();
            if seq > state.last_seen_event_id {
                state.last_seen_event_id = seq;
            }
        }
    }

    fn dispatch(&self, message: Arc<TOut>) {
        let r#type = message.type_name().to_string();
        // Diagnostics are routed to their own sink and never enter the
        // listener/buffer path.
        if r#type == "sandbox-log" || r#type == "debug-event" {
            if let Some(on_diagnostic) = &self.on_diagnostic {
                on_diagnostic(&message);
            }
            return;
        }
        if r#type == "error"
            && let Some(on_bridge_error) = &self.on_bridge_error
        {
            on_bridge_error(&message);
        }

        let _delivery = self.delivery.lock();
        let listeners = {
            let mut state = self.state.lock();
            let set = state.listeners_of(&r#type);
            if state.listener_attachment_depth > 0
                || state.buffered_offset < state.buffered.len()
                || set.is_none()
            {
                state.buffered.push(BufferedEvent {
                    message,
                    listeners: set,
                });
                drop(state);
                self.flush_buffered(false);
                self.schedule_selective_buffered_flush();
                return;
            }
            set.unwrap_or_default()
        };
        for listener in listeners {
            listener(&message);
        }
    }

    fn flush_buffered(&self, selective: bool) {
        let _delivery = self.delivery.lock();
        {
            let mut state = self.state.lock();
            if state.flushing_buffered || state.listener_attachment_depth > 0 {
                return;
            }
            state.flushing_buffered = true;
        }

        struct Finally<'a, TOut: SandboxChannelMessage>(&'a Inner<TOut>);
        impl<TOut: SandboxChannelMessage> Drop for Finally<'_, TOut> {
            fn drop(&mut self) {
                let mut state = self.0.state.lock();
                if state.buffered_offset > 0 {
                    let offset = state.buffered_offset;
                    state.buffered.drain(..offset);
                    state.buffered_offset = 0;
                }
                state.flushing_buffered = false;
            }
        }
        let _finally = Finally(self);

        loop {
            let next = {
                let mut state = self.state.lock();
                if state.buffered_offset >= state.buffered.len() {
                    None
                } else {
                    let index = state.buffered_offset;
                    let message = state.buffered[index].message.clone();
                    let r#type = message.type_name().to_string();
                    let listeners = state.buffered[index]
                        .listeners
                        .clone()
                        .or_else(|| state.listeners_of(&r#type));
                    match listeners {
                        Some(listeners) if !listeners.is_empty() => {
                            state.buffered_offset += 1;
                            Some(Some((message, listeners)))
                        }
                        _ => {
                            if !selective {
                                None
                            } else {
                                state.buffered_offset += 1;
                                state
                                    .buffered_by_type
                                    .entry(r#type)
                                    .or_default()
                                    .push_back(message);
                                Some(None)
                            }
                        }
                    }
                }
            };
            match next {
                None => break,
                Some(None) => continue,
                Some(Some((message, listeners))) => {
                    for listener in listeners {
                        listener(&message);
                    }
                }
            }
        }
    }

    fn capture_buffered_listeners(&self, r#type: &str) {
        let mut state = self.state.lock();
        let Some(set) = state.listeners_of(r#type) else {
            return;
        };
        let offset = state.buffered_offset;
        for event in state.buffered.iter_mut().skip(offset) {
            if event.listeners.is_none() && event.message.type_name() == r#type {
                event.listeners = Some(set.clone());
            }
        }
    }

    fn flush_buffered_type(&self, r#type: &str) {
        let _delivery = self.delivery.lock();
        loop {
            let next = {
                let mut state = self.state.lock();
                let Some(set) = state.listeners_of(r#type) else {
                    return;
                };
                let Some(buffered) = state.buffered_by_type.get_mut(r#type) else {
                    return;
                };
                let message = buffered.pop_front();
                if buffered.is_empty() {
                    state.buffered_by_type.remove(r#type);
                }
                message.map(|message| (message, set))
            };
            let Some((message, listeners)) = next else {
                return;
            };
            for listener in listeners {
                listener(&message);
            }
        }
    }

    fn schedule_selective_buffered_flush(&self) {
        {
            let mut state = self.state.lock();
            if state.buffered_flush_scheduled
                || state.buffered_offset >= state.buffered.len()
                || !state.has_listeners()
                || state.listener_attachment_depth > 0
            {
                return;
            }
            state.buffered_flush_scheduled = true;
        }
        // `setTimeout(..., 0)`: listeners attached in the same task (and its
        // microtasks) still claim their buffered events first.
        let weak = self.weak.clone();
        self.runtime.spawn(async move {
            tokio::time::sleep(Duration::from_millis(1)).await;
            let Some(inner) = weak.upgrade() else { return };
            {
                let mut state = inner.state.lock();
                state.buffered_flush_scheduled = false;
                if state.listener_attachment_depth > 0 {
                    return;
                }
            }
            inner.flush_buffered(true);
        });
    }

    fn finalize_close(&self, code: u16, reason: &str) {
        let stop = {
            let mut state = self.state.lock();
            if state.terminal {
                return;
            }
            state.stop_keepalive.take()
        };
        if let Some(stop) = stop {
            stop.cancel();
        }
        self.abort_active_connect();
        let handlers = {
            let mut state = self.state.lock();
            state.terminal = true;
            state.connected = false;
            state.on_close_handlers.clone()
        };
        for handler in handlers {
            handler(code, reason);
        }
    }
}

#[cfg(test)]
#[path = "sandbox_channel_tests.rs"]
mod tests;
