//! Port of `sandbox-channel.test.ts` and `sandbox-channel-keepalive.test.ts`.

use std::sync::atomic::{AtomicU8, AtomicUsize};

use futures::FutureExt;
use serde_json::json;

use super::*;

// ─── protocol of the tests ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum Frame {
    TextDelta {
        id: String,
        delta: String,
    },
    Finish {},
    FinishStep {},
    Compaction {},
    Error {
        #[serde(default)]
        error: Value,
    },
    SandboxLog {
        source: String,
        stream: String,
        line: String,
    },
    DebugEvent {
        level: String,
        subsystem: String,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(from = "Frame")]
struct Outbound {
    frame: Frame,
    checkpoint: Option<SandboxChannelEventCheckpoint>,
}

impl From<Frame> for Outbound {
    fn from(frame: Frame) -> Self {
        Self {
            frame,
            checkpoint: None,
        }
    }
}

impl SandboxChannelMessage for Outbound {
    fn type_name(&self) -> &str {
        match &self.frame {
            Frame::TextDelta { .. } => "text-delta",
            Frame::Finish {} => "finish",
            Frame::FinishStep {} => "finish-step",
            Frame::Compaction {} => "compaction",
            Frame::Error { .. } => "error",
            Frame::SandboxLog { .. } => "sandbox-log",
            Frame::DebugEvent { .. } => "debug-event",
        }
    }

    fn from_validation_error(error: TypeValidationError) -> Self {
        Frame::Error {
            error: json!({ "message": error.message }),
        }
        .into()
    }

    fn set_event_checkpoint(&mut self, checkpoint: SandboxChannelEventCheckpoint) {
        self.checkpoint = Some(checkpoint);
    }

    fn event_checkpoint(&self) -> Option<&SandboxChannelEventCheckpoint> {
        self.checkpoint.as_ref()
    }
}

impl Outbound {
    fn delta(&self) -> String {
        match &self.frame {
            Frame::TextDelta { delta, .. } => delta.clone(),
            _ => String::new(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum Inbound {
    Start,
    Abort,
}

// ─── fake sockets ──────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Terminate {
    /// No `terminate`: falls back to `close()`.
    Close,
    /// A recorded no-op.
    Noop,
    /// The heartbeat socket's: closed with 1006.
    Heartbeat,
}

type SendHook = Arc<dyn Fn(&FakeSocket, &str) -> HarnessResult<()> + Send + Sync>;

struct FakeSocket {
    handler: Mutex<Option<WebSocketEventHandler>>,
    buffered: Mutex<Vec<WebSocketEvent>>,
    ready_state: AtomicU8,
    sent: Mutex<Vec<String>>,
    closes: Mutex<Vec<(u16, String)>>,
    send_hook: Mutex<Option<SendHook>>,
    ping_support: bool,
    pings: AtomicUsize,
    ping_fails: AtomicBool,
    terminate_mode: Terminate,
    terminates: AtomicUsize,
    close_reason: &'static str,
}

impl FakeSocket {
    fn new() -> Arc<Self> {
        Self::with(false, Terminate::Close, "")
    }

    fn heartbeat() -> Arc<Self> {
        Self::with(true, Terminate::Heartbeat, "closed")
    }

    fn with(
        ping_support: bool,
        terminate_mode: Terminate,
        close_reason: &'static str,
    ) -> Arc<Self> {
        Arc::new(Self {
            handler: Mutex::default(),
            buffered: Mutex::default(),
            ready_state: AtomicU8::new(1),
            sent: Mutex::default(),
            closes: Mutex::default(),
            send_hook: Mutex::default(),
            ping_support,
            pings: AtomicUsize::new(0),
            ping_fails: AtomicBool::new(false),
            terminate_mode,
            terminates: AtomicUsize::new(0),
            close_reason,
        })
    }

    fn emit(&self, event: WebSocketEvent) {
        let handler = self.handler.lock().clone();
        match handler {
            Some(handler) => handler(event),
            None => self.buffered.lock().push(event),
        }
    }

    fn deliver(&self, message: Value, seq: Option<u64>) {
        let mut payload = message;
        if let Some(seq) = seq {
            payload["seq"] = json!(seq);
        }
        self.emit(WebSocketEvent::Message(payload.to_string()));
    }

    fn drop_with(&self, code: u16) {
        self.ready_state.store(3, Ordering::SeqCst);
        self.emit(WebSocketEvent::Close {
            code,
            reason: String::new(),
        });
    }

    fn set_ready_state(&self, ready_state: u8) {
        self.ready_state.store(ready_state, Ordering::SeqCst);
    }

    fn sent(&self) -> Vec<String> {
        self.sent.lock().clone()
    }

    fn pings(&self) -> usize {
        self.pings.load(Ordering::SeqCst)
    }

    fn terminates(&self) -> usize {
        self.terminates.load(Ordering::SeqCst)
    }
}

impl WebSocket for FakeSocket {
    fn set_event_handler(&self, handler: Option<WebSocketEventHandler>) {
        *self.handler.lock() = handler.clone();
        if let Some(handler) = handler {
            let buffered = std::mem::take(&mut *self.buffered.lock());
            for event in buffered {
                handler(event);
            }
        }
    }

    fn is_open(&self) -> bool {
        self.ready_state.load(Ordering::SeqCst) == 1
    }

    fn send(&self, text: &str) -> HarnessResult<()> {
        let hook = self.send_hook.lock().clone();
        match hook {
            Some(hook) => hook(self, text),
            None => {
                self.sent.lock().push(text.to_string());
                Ok(())
            }
        }
    }

    fn close(&self, code: Option<u16>, reason: Option<&str>) {
        let code = code.unwrap_or(1000);
        let reason = reason.unwrap_or(self.close_reason).to_string();
        self.closes.lock().push((code, reason.clone()));
        self.ready_state.store(3, Ordering::SeqCst);
        self.emit(WebSocketEvent::Close { code, reason });
    }

    fn terminate(&self) {
        self.terminates.fetch_add(1, Ordering::SeqCst);
        match self.terminate_mode {
            Terminate::Close => self.close(None, None),
            Terminate::Noop => {}
            Terminate::Heartbeat => {
                self.ready_state.store(3, Ordering::SeqCst);
                self.emit(WebSocketEvent::Close {
                    code: 1006,
                    reason: String::new(),
                });
            }
        }
    }

    fn supports_ping(&self) -> bool {
        self.ping_support
    }

    fn ping(&self) -> HarnessResult<()> {
        self.pings.fetch_add(1, Ordering::SeqCst);
        if self.ping_fails.load(Ordering::SeqCst) {
            return Err(plain_error("socket error"));
        }
        Ok(())
    }
}

/// `makeConnector()`: a fresh fake socket per call.
#[derive(Clone)]
struct Connector {
    sockets: Arc<Mutex<Vec<Arc<FakeSocket>>>>,
    factory: Arc<dyn Fn() -> Arc<FakeSocket> + Send + Sync>,
}

impl Connector {
    fn new() -> Self {
        Self::with(FakeSocket::new)
    }

    fn with(factory: impl Fn() -> Arc<FakeSocket> + Send + Sync + 'static) -> Self {
        Self {
            sockets: Arc::default(),
            factory: Arc::new(factory),
        }
    }

    fn connect(&self) -> SandboxChannelConnect {
        let connector = self.clone();
        Arc::new(move |_| {
            let socket = (connector.factory)();
            connector.sockets.lock().push(socket.clone());
            async move { Ok(socket as SharedWebSocket) }.boxed()
        })
    }

    fn current(&self) -> Arc<FakeSocket> {
        self.sockets.lock().last().unwrap().clone()
    }

    fn count(&self) -> usize {
        self.sockets.lock().len()
    }
}

fn reconnect(initial: u64, max: u64, elapsed: u64) -> Option<SandboxChannelReconnectOptions> {
    Some(SandboxChannelReconnectOptions {
        initial_delay_ms: Some(initial),
        max_delay_ms: Some(max),
        max_elapsed_ms: Some(elapsed),
    })
}

fn options(connect: SandboxChannelConnect) -> SandboxChannelOptions<Outbound> {
    SandboxChannelOptions {
        reconnect: reconnect(1, 1, 200),
        ..SandboxChannelOptions::new(connect)
    }
}

type Channel = SandboxChannel<Outbound, Inbound>;

fn make_channel(connector: &Connector) -> Channel {
    SandboxChannel::new(options(connector.connect()))
}

/// `flush()`: let the queued work run.
async fn flush() {
    for _ in 0..6 {
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

async fn wait_for(condition: impl Fn() -> bool) {
    for _ in 0..500 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!("condition not met in time");
}

type Recorder<T> = Arc<Mutex<Vec<T>>>;

fn recorder<T: Clone + Send + 'static>() -> (Recorder<T>, Recorder<T>) {
    let list = Arc::new(Mutex::new(Vec::new()));
    (list.clone(), list)
}

fn text_delta(delta: &str) -> Value {
    json!({ "type": "text-delta", "id": "a", "delta": delta })
}

fn resume(last_seen_event_id: u64) -> String {
    json!({ "type": "resume", "lastSeenEventId": last_seen_event_id }).to_string()
}

fn abort_frame() -> String {
    json!({ "type": "abort" }).to_string()
}

fn start_frame() -> String {
    json!({ "type": "start" }).to_string()
}

fn on_sent(log: &Arc<Mutex<Vec<String>>>, name: &str) -> SandboxChannelSendOptions {
    let log = log.clone();
    let name = name.to_string();
    SandboxChannelSendOptions {
        on_sent: Some(Box::new(move || log.lock().push(name))),
        queue_if_disconnected: true,
    }
}

fn types(channel: &Channel, captured: &Arc<Mutex<Vec<String>>>, r#type: &str) -> Unsubscribe {
    let captured = captured.clone();
    channel.on(r#type, move |event| {
        captured.lock().push(event.type_name().to_string())
    })
}

// ─── SandboxChannel ────────────────────────────────────────────────────

#[tokio::test]
async fn notifies_sends_asynchronously_and_after_reconnect_handlers_for_queued_frames() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    let notifications = Arc::new(Mutex::new(Vec::<String>::new()));
    channel
        .send(&Inbound::Start, on_sent(&notifications, "start"))
        .unwrap();
    assert!(notifications.lock().is_empty());
    flush().await;
    assert_eq!(*notifications.lock(), vec!["start"]);

    connector.current().drop_with(1006);
    channel
        .send(&Inbound::Abort, on_sent(&notifications, "abort"))
        .unwrap();
    let n = notifications.clone();
    let _ = channel.on_reconnect(move || n.lock().push("reconnected".into()));
    assert_eq!(*notifications.lock(), vec!["start"]);
    flush().await;
    assert_eq!(*notifications.lock(), vec!["start", "reconnected", "abort"]);
    assert_eq!(connector.current().sent(), vec![resume(0), abort_frame()]);
    channel.close();
}

#[tokio::test]
async fn dispatches_outbound_messages_by_type() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    let (text, t) = recorder::<String>();
    let _ = channel.on("text-delta", move |event| t.lock().push(event.delta()));
    connector.current().deliver(text_delta("hello"), None);
    connector.current().deliver(text_delta(" world"), None);
    flush().await;
    assert_eq!(*text.lock(), vec!["hello", " world"]);
}

#[tokio::test]
async fn replays_messages_buffered_before_the_listener_subscribes() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    connector
        .current()
        .deliver(json!({ "type": "finish" }), None);
    flush().await;
    let captured = Arc::new(Mutex::new(Vec::new()));
    let _ = types(&channel, &captured, "finish");
    assert_eq!(captured.lock().len(), 1);
}

#[tokio::test]
async fn replays_buffered_messages_in_arrival_order_across_event_types() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    let socket = connector.current();
    socket.deliver(json!({ "type": "finish-step" }), None);
    socket.deliver(text_delta("result"), None);
    socket.deliver(json!({ "type": "finish-step" }), None);
    socket.deliver(json!({ "type": "finish" }), None);
    flush().await;

    let captured = Arc::new(Mutex::new(Vec::new()));
    let _ = types(&channel, &captured, "text-delta");
    assert!(captured.lock().is_empty());
    let _ = types(&channel, &captured, "finish-step");
    assert_eq!(
        *captured.lock(),
        vec!["finish-step", "text-delta", "finish-step"]
    );
    let _ = types(&channel, &captured, "finish");
    assert_eq!(
        *captured.lock(),
        vec!["finish-step", "text-delta", "finish-step", "finish"]
    );
}

#[tokio::test]
async fn preserves_arrival_order_when_listeners_attach_across_a_microtask() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    connector
        .current()
        .deliver(json!({ "type": "finish-step" }), None);
    connector.current().deliver(text_delta("result"), None);
    flush().await;

    let captured = Arc::new(Mutex::new(Vec::new()));
    let _ = types(&channel, &captured, "text-delta");
    tokio::task::yield_now().await;
    let _ = types(&channel, &captured, "finish-step");
    assert_eq!(*captured.lock(), vec!["finish-step", "text-delta"]);
}

#[tokio::test]
async fn preserves_arrival_order_across_an_explicit_asynchronous_listener_attachment() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    let finish_listener_attachment = channel.begin_listener_attachment();
    channel.open(false).await.unwrap();
    connector
        .current()
        .deliver(json!({ "type": "finish-step" }), None);
    connector.current().deliver(text_delta("result"), None);
    flush().await;

    let captured = Arc::new(Mutex::new(Vec::new()));
    let _ = types(&channel, &captured, "text-delta");
    tokio::time::sleep(Duration::from_millis(5)).await;
    let _ = types(&channel, &captured, "finish-step");
    assert!(captured.lock().is_empty());

    finish_listener_attachment();
    assert_eq!(*captured.lock(), vec!["finish-step", "text-delta"]);
}

#[tokio::test]
async fn holds_events_that_arrive_while_listeners_are_attaching() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    let finish_listener_attachment = channel.begin_listener_attachment();
    channel.open(false).await.unwrap();

    let (captured, c) = recorder::<String>();
    let _ = channel.on("text-delta", move |event| c.lock().push(event.delta()));
    connector.current().deliver(text_delta("result"), None);
    flush().await;
    assert!(captured.lock().is_empty());

    finish_listener_attachment();
    assert_eq!(*captured.lock(), vec!["result"]);
}

#[tokio::test]
async fn does_not_block_subscribed_events_behind_an_unhandled_buffered_type() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    connector
        .current()
        .deliver(json!({ "type": "compaction" }), None);
    connector.current().deliver(text_delta("result"), None);
    connector
        .current()
        .deliver(json!({ "type": "finish" }), None);
    flush().await;

    let captured = Arc::new(Mutex::new(Vec::new()));
    let _ = types(&channel, &captured, "text-delta");
    let _ = types(&channel, &captured, "finish");
    flush().await;
    assert_eq!(*captured.lock(), vec!["text-delta", "finish"]);

    let (compactions, c) = recorder::<Frame>();
    let _ = channel.on("compaction", move |event| {
        c.lock().push(event.frame.clone())
    });
    assert_eq!(*compactions.lock(), vec![Frame::Compaction {}]);
}

#[tokio::test]
async fn delivers_an_already_buffered_subscribed_event_after_immediate_unsubscribe() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    connector
        .current()
        .deliver(json!({ "type": "compaction" }), None);
    connector.current().deliver(text_delta("result"), None);
    flush().await;

    let (captured, c) = recorder::<String>();
    let unsubscribe = channel.on("text-delta", move |event| c.lock().push(event.delta()));
    unsubscribe();
    flush().await;
    assert_eq!(*captured.lock(), vec!["result"]);
}

#[tokio::test]
async fn suspend_freezes_the_cursor_at_the_last_delivered_event() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();

    let (text, t) = recorder::<String>();
    let _ = channel.on("text-delta", move |event| t.lock().push(event.delta()));
    let close_reason = Arc::new(Mutex::new(None::<String>));
    let r = close_reason.clone();
    channel.on_close(move |_, reason| *r.lock() = Some(reason.to_string()));

    let socket = connector.current();
    socket.deliver(text_delta("one"), Some(1));
    socket.deliver(text_delta("two"), Some(2));

    let cursor = channel.suspend().await;
    assert_eq!(cursor, 2);
    assert_eq!(close_reason.lock().as_deref(), Some("suspended"));
    assert_eq!(channel.last_seen_event_id(), 2);
    // The bridge reads the reason off the wire to tell a suspend from a drop.
    assert_eq!(*socket.closes.lock(), vec![(1000, "suspended".to_string())]);

    // Frames after suspend are ignored, so the cursor does not advance.
    socket.deliver(text_delta("three"), Some(3));
    flush().await;
    assert_eq!(*text.lock(), vec!["one", "two"]);
    assert_eq!(channel.last_seen_event_id(), 2);
    // No `resume` was sent: suspend is a one-way close.
    assert!(socket.sent().is_empty());
}

#[tokio::test]
async fn serialises_and_sends_inbound_messages() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    channel.send(&Inbound::Abort, Default::default()).unwrap();
    assert_eq!(connector.current().sent(), vec![abort_frame()]);
}

#[tokio::test]
async fn suspends_from_a_pinned_event_even_after_later_events_were_dispatched() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    let finish_step = Arc::new(Mutex::new(None::<Outbound>));
    let f = finish_step.clone();
    let _ = channel.on("finish-step", move |event| *f.lock() = Some(event.clone()));
    let _ = channel.on("text-delta", |_| {});

    let socket = connector.current();
    socket.deliver(json!({ "type": "finish-step" }), Some(1));
    socket.deliver(text_delta("next"), Some(2));
    socket.deliver(text_delta(" step"), Some(3));
    flush().await;

    assert_eq!(channel.last_seen_event_id(), 3);
    let release = pin_sandbox_channel_event_checkpoint(finish_step.lock().as_ref().unwrap());
    assert!(release.is_some());
    assert!(socket.sent().is_empty());
    assert_eq!(channel.suspend().await, 1);
}

#[tokio::test]
async fn returns_to_the_latest_cursor_after_releasing_a_checkpoint() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    let finish_step = Arc::new(Mutex::new(None::<Outbound>));
    let f = finish_step.clone();
    let _ = channel.on("finish-step", move |event| *f.lock() = Some(event.clone()));
    let _ = channel.on("text-delta", |_| {});

    connector
        .current()
        .deliver(json!({ "type": "finish-step" }), Some(1));
    connector.current().deliver(text_delta("next"), Some(2));
    flush().await;

    let release = pin_sandbox_channel_event_checkpoint(finish_step.lock().as_ref().unwrap())
        .expect("checkpoint");
    release();
    assert_eq!(channel.suspend().await, 2);
}

#[tokio::test]
async fn refuses_to_send_once_terminally_closed() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    channel.close();
    flush().await;
    let error = channel
        .send(&Inbound::Abort, Default::default())
        .unwrap_err();
    assert!(error.to_string().contains("closed"), "{error}");
}

#[tokio::test]
async fn surfaces_malformed_messages_as_error_events() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    let errors = Arc::new(Mutex::new(Vec::new()));
    let _ = types(&channel, &errors, "error");
    connector
        .current()
        .deliver(json!({ "type": "mystery" }), None);
    flush().await;
    assert_eq!(errors.lock().len(), 1);
}

#[tokio::test]
async fn reconnects_transparently_on_a_transient_drop_and_resumes_from_the_cursor() {
    let connector = Connector::new();
    let (debug, d) = recorder::<String>();
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        on_debug: Some(Arc::new(move |event| {
            d.lock().push(event.event().to_string())
        })),
        ..options(connector.connect())
    });
    channel.open(false).await.unwrap();
    let (closes, c) = recorder::<u16>();
    channel.on_close(move |code, _| c.lock().push(code));
    let (text, t) = recorder::<String>();
    let _ = channel.on("text-delta", move |event| t.lock().push(event.delta()));

    connector.current().deliver(text_delta("one"), Some(1));
    connector.current().deliver(text_delta("two"), Some(2));
    flush().await;

    connector.current().drop_with(1006);
    flush().await;

    assert_eq!(connector.count(), 2);
    assert_eq!(connector.current().sent(), vec![resume(2)]);
    assert!(closes.lock().is_empty());
    assert!(debug.lock().contains(&"reconnected".to_string()));

    connector.current().deliver(text_delta("three"), Some(3));
    flush().await;
    assert_eq!(*text.lock(), vec!["one", "two", "three"]);
}

#[tokio::test]
async fn queues_host_to_bridge_sends_while_disconnected_and_flushes_them_on_reconnect() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();

    connector.current().drop_with(1006);
    channel.send(&Inbound::Abort, Default::default()).unwrap();
    flush().await;

    assert_eq!(connector.count(), 2);
    assert_eq!(connector.current().sent(), vec![resume(0), abort_frame()]);
}

#[tokio::test]
async fn lets_the_caller_own_replay() {
    for state in [None, Some(2), Some(3)] {
        let connector = Connector::new();
        let channel = make_channel(&connector);
        channel.open(false).await.unwrap();
        let first = connector.current();
        match state {
            None => first.drop_with(1006),
            Some(state) => first.set_ready_state(state),
        }
        let called = Arc::new(AtomicBool::new(false));
        let c = called.clone();
        channel
            .send(
                &Inbound::Abort,
                SandboxChannelSendOptions {
                    on_sent: Some(Box::new(move || c.store(true, Ordering::SeqCst))),
                    queue_if_disconnected: false,
                },
            )
            .unwrap();
        if state.is_some() {
            first.drop_with(1006);
        }
        flush().await;
        assert!(first.sent().is_empty());
        assert_eq!(connector.current().sent(), vec![resume(0)]);
        assert!(!called.load(Ordering::SeqCst));
        channel.close();
    }
}

#[tokio::test]
async fn queues_sends_before_the_close_event() {
    for ready_state in [2, 3] {
        let connector = Connector::new();
        let channel = make_channel(&connector);
        channel.open(false).await.unwrap();
        let first = connector.current();
        first.set_ready_state(ready_state);

        let notifications = Arc::new(Mutex::new(Vec::<String>::new()));
        channel
            .send(&Inbound::Abort, on_sent(&notifications, "abort"))
            .unwrap();
        assert!(first.sent().is_empty());
        assert!(notifications.lock().is_empty());

        first.drop_with(1006);
        flush().await;
        assert_eq!(connector.current().sent(), vec![resume(0), abort_frame()]);
        assert_eq!(*notifications.lock(), vec!["abort"]);
    }
}

#[tokio::test]
async fn propagates_synchronous_send_errors_on_an_open_socket() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    *connector.current().send_hook.lock() = Some(Arc::new(|_, _| Err(plain_error("send failed"))));

    let notifications = Arc::new(Mutex::new(Vec::<String>::new()));
    let error = channel
        .send(&Inbound::Abort, on_sent(&notifications, "abort"))
        .unwrap_err();
    assert_eq!(error.to_string(), "send failed");
    flush().await;
    assert!(notifications.lock().is_empty());
}

#[tokio::test]
async fn retains_unsent_frames_when_a_reconnecting_socket_fails_during_the_flush() {
    for closing in [true, false] {
        let sockets = [FakeSocket::new(), FakeSocket::new(), FakeSocket::new()];
        let connections = Arc::new(AtomicUsize::new(0));
        let (list, c) = (sockets.clone(), connections.clone());
        let channel: Channel = SandboxChannel::new(options(Arc::new(move |_| {
            let socket = list[c.fetch_add(1, Ordering::SeqCst)].clone();
            async move { Ok(socket as SharedWebSocket) }.boxed()
        })));
        let [first, second, third] = sockets;
        let (start, abort) = (start_frame(), abort_frame());
        *second.send_hook.lock() = Some(Arc::new(move |socket, data| {
            if data == abort && !closing {
                return Err(plain_error("send failed"));
            }
            socket.sent.lock().push(data.to_string());
            if data == start && closing {
                socket.set_ready_state(2);
            }
            Ok(())
        }));
        channel.open(false).await.unwrap();
        first.drop_with(1006);
        let notifications = Arc::new(Mutex::new(Vec::<String>::new()));
        channel
            .send(&Inbound::Start, on_sent(&notifications, "first-start"))
            .unwrap();
        channel
            .send(&Inbound::Abort, on_sent(&notifications, "abort"))
            .unwrap();
        channel
            .send(&Inbound::Start, on_sent(&notifications, "last-start"))
            .unwrap();
        flush().await;

        assert_eq!(second.sent(), vec![resume(0), start_frame()]);
        if closing {
            second.drop_with(1006);
            flush().await;
        }
        assert_eq!(connections.load(Ordering::SeqCst), 3);
        assert_eq!(third.sent(), vec![resume(0), abort_frame(), start_frame()]);
        let mut notified = notifications.lock().clone();
        notified.sort();
        assert_eq!(notified, vec!["abort", "first-start", "last-start"]);
    }
}

#[tokio::test]
async fn fires_on_close_on_a_host_initiated_close() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    let (closes, c) = recorder::<u16>();
    channel.on_close(move |code, _| c.lock().push(code));
    channel.close();
    flush().await;
    assert_eq!(*closes.lock(), vec![1000]);
    assert!(channel.is_closed());
}

#[tokio::test]
async fn treats_a_drop_after_begin_close_as_terminal_not_a_reconnect() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    let (closes, c) = recorder::<u16>();
    channel.on_close(move |code, _| c.lock().push(code));

    channel.begin_close();
    connector.current().drop_with(1000);
    flush().await;

    assert_eq!(connector.count(), 1);
    assert_eq!(*closes.lock(), vec![1000]);
}

#[tokio::test]
async fn gives_up_and_fires_on_close_once_the_reconnect_budget_is_exhausted() {
    let first = FakeSocket::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let (f, c) = (first.clone(), calls.clone());
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        reconnect: reconnect(1, 1, 30),
        ..SandboxChannelOptions::new(Arc::new(move |_| {
            let call = c.fetch_add(1, Ordering::SeqCst);
            let socket = f.clone();
            async move {
                if call == 0 {
                    Ok(socket as SharedWebSocket)
                } else {
                    Err(plain_error("connect refused"))
                }
            }
            .boxed()
        }))
    });
    channel.open(false).await.unwrap();
    let (closes, c) = recorder::<u16>();
    channel.on_close(move |code, _| c.lock().push(code));
    first.drop_with(1006);
    wait_for(|| closes.lock().len() == 1).await;
    assert_eq!(closes.lock()[0], 1006);
}

/// A connect thunk that returns `first` once, then hands each later call's
/// signal to `on_later` and runs the future it returns.
fn first_then(
    first: Arc<FakeSocket>,
    on_later: impl Fn(CancellationToken) -> BoxFuture<'static, HarnessResult<SharedWebSocket>>
    + Send
    + Sync
    + 'static,
) -> SandboxChannelConnect {
    let calls = AtomicUsize::new(0);
    Arc::new(move |options| {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            let socket = first.clone();
            async move { Ok(socket as SharedWebSocket) }.boxed()
        } else {
            on_later(options.abort_signal)
        }
    })
}

#[tokio::test]
async fn bounds_a_hanging_reconnect_connection_by_max_elapsed_ms() {
    let first = FakeSocket::new();
    let (signals, s) = recorder::<CancellationToken>();
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        reconnect: reconnect(1, 1, 20),
        ..SandboxChannelOptions::new(first_then(first.clone(), move |signal| {
            s.lock().push(signal);
            futures::future::pending().boxed()
        }))
    });
    channel.open(false).await.unwrap();
    let (closes, c) = recorder::<u16>();
    channel.on_close(move |code, _| c.lock().push(code));

    first.drop_with(1006);
    wait_for(|| *closes.lock() == vec![1006]).await;
    assert!(signals.lock()[0].is_cancelled());
}

#[tokio::test]
async fn does_not_let_a_backoff_delay_extend_the_reconnect_deadline() {
    let first = FakeSocket::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        reconnect: reconnect(100, 100, 20),
        ..SandboxChannelOptions::new(first_then(first.clone(), move |_| {
            c.fetch_add(1, Ordering::SeqCst);
            async { Err(plain_error("connect refused")) }.boxed()
        }))
    });
    channel.open(false).await.unwrap();
    let (closes, c) = recorder::<u16>();
    channel.on_close(move |code, _| c.lock().push(code));

    let started = Instant::now();
    first.drop_with(1006);
    wait_for(|| *closes.lock() == vec![1006]).await;
    assert!(started.elapsed() < Duration::from_millis(90));
    assert!(calls.load(Ordering::SeqCst) >= 1);
}

#[tokio::test]
async fn closes_a_socket_that_resolves_after_the_reconnect_attempt_is_aborted() {
    let first = FakeSocket::new();
    let late = FakeSocket::with(false, Terminate::Noop, "");
    let resolve_late = Arc::new(Mutex::new(None::<oneshot::Sender<SharedWebSocket>>));
    let r = resolve_late.clone();
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        reconnect: reconnect(1, 1, 20),
        ..SandboxChannelOptions::new(first_then(first.clone(), move |_| {
            let (tx, rx) = oneshot::channel();
            *r.lock() = Some(tx);
            async move { rx.await.map_err(|_| plain_error("dropped")) }.boxed()
        }))
    });
    channel.open(false).await.unwrap();

    first.drop_with(1006);
    wait_for(|| resolve_late.lock().is_some()).await;
    wait_for(|| channel.is_closed()).await;
    let tx = resolve_late.lock().take().unwrap();
    let _ = tx.send(late.clone());
    flush().await;
    assert_eq!(late.terminates(), 1);
}

#[tokio::test]
async fn aborts_an_active_connection_when_the_channel_is_torn_down() {
    let first = FakeSocket::new();
    let reconnect_signal = Arc::new(Mutex::new(None::<CancellationToken>));
    let r = reconnect_signal.clone();
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        reconnect: reconnect(1, 1, 1_000),
        ..SandboxChannelOptions::new(first_then(first.clone(), move |signal| {
            *r.lock() = Some(signal);
            futures::future::pending().boxed()
        }))
    });
    channel.open(false).await.unwrap();

    first.drop_with(1006);
    wait_for(|| reconnect_signal.lock().is_some()).await;
    channel.close();
    flush().await;

    assert!(reconnect_signal.lock().as_ref().unwrap().is_cancelled());
    assert!(channel.is_closed());
}

#[tokio::test]
async fn passes_a_distinct_abort_signal_and_the_reason_to_each_connection_attempt() {
    let connector = Connector::new();
    let (seen, s) = recorder::<(CancellationToken, SandboxChannelConnectReason)>();
    let inner = connector.connect();
    let channel: Channel = SandboxChannel::new(options(Arc::new(move |options| {
        s.lock()
            .push((options.abort_signal.clone(), options.reason));
        inner(options)
    })));
    channel.open(false).await.unwrap();
    connector.current().drop_with(1006);
    wait_for(|| seen.lock().len() == 2).await;
    let seen = seen.lock();
    assert!(seen.iter().all(|(signal, _)| !signal.is_cancelled()));
    seen[0].0.cancel();
    assert!(!seen[1].0.is_cancelled());
    assert_eq!(
        seen.iter().map(|(_, reason)| *reason).collect::<Vec<_>>(),
        vec![
            SandboxChannelConnectReason::Open,
            SandboxChannelConnectReason::Reconnect
        ]
    );
}

#[tokio::test]
async fn seeds_last_seen_event_id_and_advances_it_as_events_arrive() {
    let connector = Connector::new();
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        initial_last_seen_event_id: Some(7),
        ..SandboxChannelOptions::new(connector.connect())
    });
    assert_eq!(channel.last_seen_event_id(), 7);
    channel.open(false).await.unwrap();
    let _ = channel.on("text-delta", |_| {});
    connector.current().deliver(text_delta("x"), Some(9));
    flush().await;
    assert_eq!(channel.last_seen_event_id(), 9);
}

#[tokio::test]
async fn open_with_resume_sends_a_resume_frame_with_the_seeded_cursor() {
    let connector = Connector::new();
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        initial_last_seen_event_id: Some(4),
        ..SandboxChannelOptions::new(connector.connect())
    });
    channel.open(true).await.unwrap();
    assert!(connector.current().sent().contains(&resume(4)));
}

#[tokio::test]
async fn open_without_resume_sends_no_resume_frame() {
    let connector = Connector::new();
    let channel = make_channel(&connector);
    channel.open(false).await.unwrap();
    assert!(connector.current().sent().is_empty());
}

#[tokio::test]
async fn routes_sandbox_log_and_debug_event_frames_to_on_diagnostic() {
    let connector = Connector::new();
    let (diagnostics, d) = recorder::<String>();
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        on_diagnostic: Some(Arc::new(move |event: &Outbound| {
            d.lock().push(event.type_name().to_string())
        })),
        ..SandboxChannelOptions::new(connector.connect())
    });
    channel.open(false).await.unwrap();

    let leaked = Arc::new(Mutex::new(Vec::new()));
    let _ = types(&channel, &leaked, "sandbox-log");
    let (text, t) = recorder::<String>();
    let _ = channel.on("text-delta", move |event| t.lock().push(event.delta()));

    let socket = connector.current();
    socket.deliver(
        json!({ "type": "sandbox-log", "source": "bridge", "stream": "stdout", "line": "hi" }),
        Some(1),
    );
    socket.deliver(
        json!({ "type": "debug-event", "level": "info", "subsystem": "bridge.turn", "message": "started" }),
        Some(2),
    );
    socket.deliver(text_delta("x"), Some(3));
    flush().await;

    assert_eq!(*diagnostics.lock(), vec!["sandbox-log", "debug-event"]);
    assert!(leaked.lock().is_empty());
    assert_eq!(*text.lock(), vec!["x"]);
    // Diagnostics still advance the resume cursor.
    assert_eq!(channel.last_seen_event_id(), 3);
}

#[tokio::test]
async fn reports_error_frames_to_on_bridge_error_and_still_dispatches_them() {
    let connector = Connector::new();
    let (errors, e) = recorder::<Frame>();
    let channel: Channel = SandboxChannel::new(SandboxChannelOptions {
        on_bridge_error: Some(Arc::new(move |event: &Outbound| {
            e.lock().push(event.frame.clone())
        })),
        ..SandboxChannelOptions::new(connector.connect())
    });
    channel.open(false).await.unwrap();

    let (listener_events, l) = recorder::<Frame>();
    let _ = channel.on("error", move |event| l.lock().push(event.frame.clone()));
    let error = json!({ "name": "Error", "message": "boom", "stack": "Error: boom" });
    connector
        .current()
        .deliver(json!({ "type": "error", "error": error }), Some(1));
    flush().await;

    assert_eq!(*errors.lock(), vec![Frame::Error { error }]);
    assert_eq!(*listener_events.lock(), *errors.lock());
    assert_eq!(channel.last_seen_event_id(), 1);
}

// ─── keepalive ─────────────────────────────────────────────────────────

struct Heartbeat {
    channel: SandboxChannel<Outbound, Inbound>,
    connector: Connector,
    debug: Arc<Mutex<Vec<String>>>,
}

fn heartbeat_channel(
    keep_alive_ms: Option<u64>,
    connect: Option<SandboxChannelConnect>,
) -> Heartbeat {
    let connector = Connector::with(FakeSocket::heartbeat);
    let (debug, d) = recorder::<String>();
    let channel = SandboxChannel::new(SandboxChannelOptions {
        keep_alive_ms,
        reconnect: reconnect(50, 50, 200),
        on_debug: Some(Arc::new(move |event| {
            d.lock().push(event.event().to_string())
        })),
        ..SandboxChannelOptions::new(connect.unwrap_or_else(|| connector.connect()))
    });
    Heartbeat {
        channel,
        connector,
        debug,
    }
}

/// `vi.advanceTimersByTimeAsync(ms)` on a paused clock.
async fn advance(ms: u64) {
    tokio::time::sleep(Duration::from_millis(ms)).await;
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
}

#[tokio::test(start_paused = true)]
async fn pings_every_20_seconds_by_default() {
    let h = heartbeat_channel(None, None);
    h.channel.open(false).await.unwrap();
    advance(19_999).await;
    assert_eq!(h.connector.current().pings(), 0);
    advance(1).await;
    assert_eq!(h.connector.current().pings(), 1);
    h.channel.close();
}

#[tokio::test(start_paused = true)]
async fn keeps_a_responsive_socket_connected_at_the_configured_interval() {
    let h = heartbeat_channel(Some(100), None);
    h.channel.open(false).await.unwrap();
    for _ in 0..5 {
        advance(100).await;
        h.connector.current().emit(WebSocketEvent::Pong);
    }
    let socket = h.connector.current();
    assert_eq!(socket.pings(), 5);
    assert_eq!(socket.terminates(), 0);
    assert!(h.debug.lock().is_empty());
    h.channel.close();
}

#[tokio::test(start_paused = true)]
async fn terminates_after_two_unanswered_pings_and_starts_a_fresh_heartbeat_on_reconnect() {
    let h = heartbeat_channel(Some(100), None);
    h.channel.open(false).await.unwrap();
    let first = h.connector.current();
    first.emit(WebSocketEvent::Message(
        json!({ "type": "finish", "seq": 7 }).to_string(),
    ));
    advance(200).await;
    assert_eq!(first.pings(), 2);
    assert_eq!(first.terminates(), 0);
    advance(100).await;
    assert_eq!(first.terminates(), 1);
    assert_eq!(h.connector.count(), 2);
    let second = h.connector.current();
    assert_eq!(second.sent(), vec![resume(7)]);
    assert_eq!(*h.debug.lock(), vec!["reconnect-attempt", "reconnected"]);
    first.emit(WebSocketEvent::Close {
        code: 1006,
        reason: String::new(),
    });
    advance(100).await;
    assert_eq!(first.pings(), 2);
    assert_eq!(second.pings(), 1);
    assert_eq!(second.terminates(), 0);
    h.channel.close();
}

#[tokio::test(start_paused = true)]
async fn closes_after_the_reconnect_budget_expires_for_a_dead_peer() {
    let socket = FakeSocket::heartbeat();
    let attempts = AtomicUsize::new(0);
    let s = socket.clone();
    let h = heartbeat_channel(
        Some(100),
        Some(Arc::new(move |_| {
            let first = attempts.fetch_add(1, Ordering::SeqCst) == 0;
            let socket = s.clone();
            async move {
                if first {
                    Ok(socket as SharedWebSocket)
                } else {
                    Err(plain_error("peer unavailable"))
                }
            }
            .boxed()
        })),
    );
    let (closes, c) = recorder::<(u16, String)>();
    h.channel
        .on_close(move |code, reason| c.lock().push((code, reason.to_string())));
    h.channel.open(false).await.unwrap();
    advance(500).await;
    assert_eq!(socket.terminates(), 1);
    assert!(h.channel.is_closed());
    assert_eq!(*closes.lock(), vec![(1006, "reconnect failed".to_string())]);
    assert_eq!(
        h.debug.lock().last().map(String::as_str),
        Some("reconnect-failed")
    );
}

#[tokio::test(start_paused = true)]
async fn disables_keepalive_when_keep_alive_ms_is_zero() {
    let h = heartbeat_channel(Some(0), None);
    h.channel.open(false).await.unwrap();
    advance(100_000).await;
    assert_eq!(h.connector.current().pings(), 0);
    h.channel.close();
}

#[tokio::test(start_paused = true)]
async fn leaves_sockets_without_ping_support_unchanged() {
    let socket = FakeSocket::with(false, Terminate::Heartbeat, "closed");
    let s = socket.clone();
    let h = heartbeat_channel(
        None,
        Some(Arc::new(move |_| {
            let socket = s.clone();
            async move { Ok(socket as SharedWebSocket) }.boxed()
        })),
    );
    h.channel.open(false).await.unwrap();
    advance(100_000).await;
    assert_eq!(socket.pings(), 0);
    assert_eq!(socket.terminates(), 0);
    h.channel.close();
}

#[tokio::test(start_paused = true)]
async fn cleans_up_keepalive_on_close_suspend_and_begin_close() {
    for method in ["close", "suspend", "beginClose"] {
        let h = heartbeat_channel(Some(100), None);
        h.channel.open(false).await.unwrap();
        match method {
            "close" => h.channel.close(),
            "suspend" => {
                h.channel.suspend().await;
            }
            _ => h.channel.begin_close(),
        }
        advance(1_000).await;
        let socket = h.connector.sockets.lock()[0].clone();
        assert_eq!(socket.pings(), 0, "{method}");
        assert_eq!(socket.terminates(), 0, "{method}");
        assert_eq!(h.connector.count(), 1, "{method}");
        h.channel.close();
    }
}

#[tokio::test(start_paused = true)]
async fn cleans_up_a_dropped_socket_on_close_and_error() {
    for event in ["close", "error"] {
        let h = heartbeat_channel(Some(100), None);
        h.channel.open(false).await.unwrap();
        let first = h.connector.current();
        first.emit(if event == "close" {
            WebSocketEvent::Close {
                code: 1006,
                reason: String::new(),
            }
        } else {
            WebSocketEvent::Error("boom".into())
        });
        advance(100).await;
        // The reconnect waits for its 50ms backoff only after a failure, so
        // the new socket is up and pinged once.
        assert_eq!(first.pings(), 0, "{event}");
        assert_eq!(h.connector.sockets.lock()[1].pings(), 1, "{event}");
        h.channel.close();
    }
}

#[tokio::test(start_paused = true)]
async fn does_not_ping_sockets_that_are_no_longer_open() {
    let h = heartbeat_channel(Some(100), None);
    h.channel.open(false).await.unwrap();
    let socket = h.connector.current();
    socket.set_ready_state(2);
    advance(500).await;
    assert_eq!(socket.pings(), 0);
    assert_eq!(socket.terminates(), 0);
    h.channel.close();
}

#[tokio::test(start_paused = true)]
async fn reconnects_if_sending_a_ping_throws() {
    let h = heartbeat_channel(Some(100), None);
    h.channel.open(false).await.unwrap();
    let socket = h.connector.current();
    socket.ping_fails.store(true, Ordering::SeqCst);
    advance(100).await;
    assert_eq!(socket.terminates(), 1);
    assert_eq!(h.connector.count(), 2);
    h.channel.close();
}
