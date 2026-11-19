//! Port of `claude-code-start.integration.test.ts`: the harness against a
//! real WebSocket server that plays the bridge.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt as _, StreamExt as _};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_util::sync::CancellationToken;

use super::claude_code_harness::{
    ClaudeCodeHarness, ClaudeCodeHarnessSettings, ClaudeCodeHost, create_claude_code_with_host,
};
use crate::harness::adapters::bridge_test_support::{FakeSandbox, start_options};
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::spec::diagnostic::{HarnessV1Diagnostic, HarnessV1Observability};
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1PromptControl, HarnessV1PromptTurnOptions, HarnessV1Session,
    HarnessV1StartOptions,
};
use crate::harness::spec::lifecycle_state::{HarnessV1ContinueTurnState, HarnessV1TurnSettings};
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::spec::{HarnessV1Authentication, HarnessV1ContinueTurnOptions};
use crate::harness::utils::bridge_auth::create_bridge_auth_key_pair;
use crate::harness::utils::bridge_auth_protocol::serialize_bridge_auth_challenge_frame;

// ─── The bridge side ───────────────────────────────────────────────────

enum Outgoing {
    Text(String),
    Close(u16, String),
}

/// One authenticated bridge connection, as the server sees it.
struct Connection {
    out: mpsc::UnboundedSender<Outgoing>,
    incoming: mpsc::UnboundedReceiver<Value>,
}

/// A handle that sends on one connection.
#[derive(Clone)]
struct Sender(mpsc::UnboundedSender<Outgoing>);

impl Sender {
    fn send(&self, frame: Value) {
        let _ = self.0.send(Outgoing::Text(frame.to_string()));
    }

    fn close(&self, code: u16, reason: &str) {
        let _ = self.0.send(Outgoing::Close(code, reason.to_string()));
    }
}

impl Connection {
    fn sender(&self) -> Sender {
        Sender(self.out.clone())
    }
}

type OnConnection = Arc<dyn Fn(Connection) + Send + Sync>;

/// `new WebSocketServer(...)` plus `onAuthenticatedConnection`: plays the
/// bridge side of the authentication handshake, sending a challenge and
/// handing the connection to `on_connection` once the host answered with
/// its proof.
async fn bridge_server(on_connection: OnConnection) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let on_connection = on_connection.clone();
            tokio::spawn(async move {
                let Ok(ws) = tokio_tungstenite::accept_async(stream).await else {
                    return;
                };
                let (mut sink, mut source) = ws.split();
                if sink
                    .send(Message::text(serialize_bridge_auth_challenge_frame(
                        &"a".repeat(64),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }
                // The proof.
                if !matches!(source.next().await, Some(Ok(_))) {
                    return;
                }
                let (out, mut outgoing) = mpsc::unbounded_channel::<Outgoing>();
                let (incoming_tx, incoming) = mpsc::unbounded_channel::<Value>();
                tokio::spawn(async move {
                    while let Some(message) = outgoing.recv().await {
                        match message {
                            Outgoing::Text(text) => {
                                if sink.send(Message::text(text)).await.is_err() {
                                    return;
                                }
                            }
                            Outgoing::Close(code, reason) => {
                                let _ = sink
                                    .send(Message::Close(Some(CloseFrame {
                                        code: code.into(),
                                        reason: reason.into(),
                                    })))
                                    .await;
                                let _ = sink.close().await;
                                return;
                            }
                        }
                    }
                });
                tokio::spawn(async move {
                    while let Some(Ok(message)) = source.next().await {
                        if let Message::Text(text) = message
                            && let Ok(value) = serde_json::from_str::<Value>(text.as_str())
                            && incoming_tx.send(value).is_err()
                        {
                            return;
                        }
                    }
                });
                on_connection(Connection { out, incoming });
            });
        }
    });
    address
}

// ─── The host side ─────────────────────────────────────────────────────

struct Bridge {
    harness: ClaudeCodeHarness,
    sandbox: FakeSandbox,
    _home: TempDir,
}

fn finish_frame(seq: u64) -> Value {
    json!({
        "type": "finish",
        "seq": seq,
        "finishReason": { "unified": "stop", "raw": "stop" },
        "totalUsage": { "inputTokens": {}, "outputTokens": {} },
    })
}

/// The harness and sandbox of `createBridgeSession` / `startTurn`.
fn bridge(address: SocketAddr) -> Bridge {
    let home = mkdtemp("claude-code-start-");
    let home_path = home.path_string();
    let harness = create_claude_code_with_host(
        ClaudeCodeHarnessSettings {
            auth: Some(HarnessV1Authentication::Environment(
                [("ANTHROPIC_API_KEY".to_string(), "test".to_string())]
                    .into_iter()
                    .collect(),
            )),
            port: Some(4000),
            turn_start_timeout_ms: Some(2_000),
            ..Default::default()
        },
        ClaudeCodeHost {
            web_socket_constructor: None,
            process_env: Some(
                [
                    ("HOME".to_string(), home_path.clone()),
                    ("CLAUDE_CONFIG_DIR".to_string(), home_path),
                ]
                .into_iter()
                .collect(),
            ),
        },
    )
    .unwrap();
    let sandbox = FakeSandbox {
        home_dir: "/home/user".into(),
        bridge_port_endpoint: HarnessV1PortEndpoint {
            url: format!("ws://{address}/"),
            headers: None,
        },
        spawn_stdout: Some("{\"type\":\"bridge-ready\",\"port\":4000}\n".into()),
        ports: vec![4000],
        ..Default::default()
    };
    Bridge {
        harness,
        sandbox,
        _home: home,
    }
}

impl Bridge {
    fn options(&self) -> HarnessV1StartOptions {
        let mut options = start_options("s", self.sandbox.network());
        options.session_work_dir = "/w".into();
        options
    }

    async fn start(&self, options: HarnessV1StartOptions) -> Box<dyn HarnessV1Session> {
        self.harness
            .do_start(options)
            .await
            .expect("session starts")
    }
}

fn settings() -> HarnessV1TurnSettings {
    HarnessV1TurnSettings {
        model: None,
        skills: Vec::new(),
        instructions: None,
        tools: Vec::new(),
    }
}

fn prompt(
    text: &str,
    abort_signal: Option<CancellationToken>,
    emit: impl Fn(HarnessV1StreamPart) + Send + Sync + 'static,
) -> HarnessV1PromptTurnOptions {
    HarnessV1PromptTurnOptions {
        settings: settings(),
        prompt: text.into(),
        response_format: None,
        abort_signal,
        emit: Arc::new(emit),
    }
}

fn continue_options(
    emit: impl Fn(HarnessV1StreamPart) + Send + Sync + 'static,
) -> HarnessV1ContinueTurnOptions {
    HarnessV1ContinueTurnOptions {
        settings: settings(),
        response_format: None,
        abort_signal: None,
        emit: Arc::new(emit),
    }
}

fn part_type(part: &HarnessV1StreamPart) -> String {
    serde_json::to_value(part).unwrap()["type"]
        .as_str()
        .unwrap()
        .to_string()
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("finished in time")
}

/// `report` that wakes `notify` on a `tail delivered` log line.
fn tail_reporter(notify: Arc<Notify>) -> HarnessV1Observability {
    HarnessV1Observability {
        debug: None,
        report: Some(Arc::new(move |event: HarnessV1Diagnostic| {
            if event.message == "tail delivered" {
                notify.notify_one();
            }
        })),
    }
}

fn sandbox_log(line: &str) -> Value {
    json!({ "type": "sandbox-log", "source": "test", "stream": "stdout", "line": line })
}

#[tokio::test]
async fn preserves_turn_identity_and_replay_checkpoints_across_suspension() {
    for legacy in [false, true] {
        let start_id: Arc<Mutex<Option<String>>> = Arc::default();
        let starts = Arc::new(Mutex::new(0));
        let address = bridge_server({
            let start_id = start_id.clone();
            let starts = starts.clone();
            Arc::new(move |mut connection: Connection| {
                let start_id = start_id.clone();
                let starts = starts.clone();
                tokio::spawn(async move {
                    let socket = connection.sender();
                    socket
                        .send(json!({ "type": "bridge-hello", "state": "running", "lastSeq": 6 }));
                    while let Some(message) = connection.incoming.recv().await {
                        if message["type"] == "start" {
                            *starts.lock() += 1;
                            let id = message["startId"].as_str().unwrap().to_string();
                            *start_id.lock() = Some(id.clone());
                            for (index, mut event) in [
                                json!({ "type": "stream-start" }),
                                json!({ "type": "text-start", "id": "text" }),
                                json!({ "type": "text-delta", "id": "text", "delta": "first" }),
                                json!({
                                    "type": "finish-step",
                                    "finishReason": { "unified": "stop" },
                                    "usage": { "inputTokens": {}, "outputTokens": {} },
                                }),
                                json!({ "type": "text-delta", "id": "text", "delta": "tail" }),
                                sandbox_log("tail delivered"),
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                event["seq"] = json!(index + 1);
                                event["startId"] = json!(id);
                                socket.send(event);
                            }
                        }
                        if message["type"] == "resume" {
                            assert_eq!(message["lastSeenEventId"], 4);
                            let id = start_id.lock().clone();
                            for (index, mut event) in [
                                json!({ "type": "text-delta", "id": "text", "delta": "tail" }),
                                json!({ "type": "text-end", "id": "text" }),
                                json!({
                                    "type": "finish",
                                    "finishReason": { "unified": "stop" },
                                    "totalUsage": { "inputTokens": {}, "outputTokens": {} },
                                }),
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                event["seq"] = json!(index + 5);
                                event["startId"] = json!(id);
                                socket.send(event);
                            }
                        }
                    }
                });
            })
        })
        .await;
        let bridge = bridge(address);
        let tail_received = Arc::new(Notify::new());
        let mut start_options = bridge.options();
        start_options.observability = Some(tail_reporter(tail_received.clone()));
        let session = bridge.start(start_options.clone()).await;
        let first = session
            .do_prompt_turn(prompt("Work.", None, |part| {
                if let HarnessV1StreamPart::FinishStep { checkpoint, .. } = &part {
                    let checkpoint = checkpoint.as_ref().expect("the event has a checkpoint");
                    let _release = checkpoint.pin();
                }
            }))
            .await
            .unwrap();
        within(tail_received.notified()).await;
        let mut continue_from: HarnessV1ContinueTurnState =
            within(session.do_suspend_turn()).await.unwrap();
        within(first.done()).await.unwrap();
        let start_id = start_id.lock().clone().unwrap();
        let bridge_data = continue_from.data["bridge"].as_object_mut().unwrap();
        assert_eq!(bridge_data["startId"], start_id);
        assert_eq!(bridge_data["lastSeenEventId"], 4);
        if legacy {
            bridge_data.remove("startId");
        }
        let mut options = start_options.clone();
        options.continue_from = Some(continue_from);
        let resumed = bridge.start(options).await;
        let events: Arc<Mutex<Vec<HarnessV1StreamPart>>> = Arc::default();
        let continued = resumed
            .do_continue_turn(continue_options({
                let events = events.clone();
                move |part| events.lock().push(part)
            }))
            .await
            .unwrap();
        within(continued.done()).await.unwrap();
        assert_eq!(*starts.lock(), 1);
        let parts: Vec<Value> = events
            .lock()
            .iter()
            .map(|part| serde_json::to_value(part).unwrap())
            .collect();
        assert_eq!(
            parts
                .iter()
                .map(|part| part["type"].clone())
                .collect::<Vec<_>>(),
            [json!("text-delta"), json!("text-end"), json!("finish")]
        );
        assert!(parts.iter().all(|part| part.get("startId").is_none()));
        let detached = resumed.do_detach().await.unwrap();
        assert_eq!(detached.data["bridge"]["startId"], start_id);
        let _ = session.do_destroy().await;
    }
}

#[tokio::test]
async fn keeps_the_correct_identifier_during_recovery() {
    for recovery in ["replay", "rerun"] {
        let leg = Arc::new(Mutex::new(0));
        let starts: Arc<Mutex<Vec<String>>> = Arc::default();
        let address = bridge_server({
            let leg = leg.clone();
            let starts = starts.clone();
            Arc::new(move |mut connection: Connection| {
                let socket = connection.sender();
                let current = {
                    let mut leg = leg.lock();
                    *leg += 1;
                    *leg
                };
                if current == 2 {
                    socket.close(1000, "");
                    return;
                }
                let starts = starts.clone();
                tokio::spawn(async move {
                    socket
                        .send(json!({ "type": "bridge-hello", "state": "waiting", "lastSeq": 3 }));
                    while let Some(message) = connection.incoming.recv().await {
                        if message["type"] == "start" {
                            starts
                                .lock()
                                .push(message["startId"].as_str().unwrap().to_string());
                        }
                        if message["type"] == "start" || message["type"] == "resume" {
                            let mut finish = finish_frame(3);
                            finish["finishReason"] = json!({ "unified": "stop" });
                            finish["startId"] = message
                                .get("startId")
                                .cloned()
                                .unwrap_or_else(|| json!("suspended-turn"));
                            socket.send(finish);
                        }
                    }
                });
            })
        })
        .await;
        let bridge = bridge(address);
        // `createBridgeSession` starts a session first: the bridge's first leg.
        let _session = bridge.start(bridge.options()).await;
        if recovery == "replay" {
            bridge.sandbox.record.lock().files.insert(
                "/home/user/.ai-sdk-harness/.agent-runs/s/bridge/event-log.ndjson".into(),
                "{\"type\":\"finish\"}\n".into(),
            );
        }
        let mut options = bridge.options();
        options.continue_from = Some(
            serde_json::from_value(json!({
                "type": "continue-turn",
                "harnessId": "claude-code",
                "specificationVersion": "harness-v1",
                "data": {
                    "bridge": {
                        "port": 4000,
                        "privateKey": create_bridge_auth_key_pair().unwrap().private_key,
                        "lastSeenEventId": 1,
                        "startId": "suspended-turn",
                    },
                },
            }))
            .unwrap(),
        );
        let resumed = bridge.start(options).await;
        let events: Arc<Mutex<Vec<String>>> = Arc::default();
        let control = resumed
            .do_continue_turn(continue_options({
                let events = events.clone();
                move |part| events.lock().push(part_type(&part))
            }))
            .await
            .unwrap();
        within(control.done()).await.unwrap();
        assert_eq!(*events.lock(), ["finish"], "{recovery}");
        let starts = starts.lock().clone();
        if recovery == "replay" {
            assert!(starts.is_empty());
        } else {
            assert_eq!(starts.len(), 1);
            assert_ne!(starts[0], "suspended-turn");
        }
        let _ = resumed.do_destroy().await;
    }
}

#[tokio::test]
async fn isolates_the_next_turn_from_events_of_an_aborted_turn() {
    for delivery in ["in-transit", "buffered", "replayed"] {
        isolates_the_next_turn(delivery).await;
    }
}

#[derive(Default)]
struct IsolationState {
    first_id: Option<String>,
    second_id: Option<String>,
    starts: Vec<String>,
    sequence: u64,
    leg: usize,
    answer: Option<Sender>,
}

async fn isolates_the_next_turn(delivery: &'static str) {
    let state: Arc<Mutex<IsolationState>> = Arc::default();
    let answer_ready = Arc::new(Notify::new());
    let address = bridge_server({
        let state = state.clone();
        let answer_ready = answer_ready.clone();
        Arc::new(move |mut connection: Connection| {
            let socket = connection.sender();
            let state = state.clone();
            let answer_ready = answer_ready.clone();
            let current_leg = {
                let mut state = state.lock();
                state.leg += 1;
                state.leg
            };
            tokio::spawn(async move {
                let sequence = state.lock().sequence;
                socket.send(json!({
                    "type": "bridge-hello",
                    "state": "waiting",
                    "lastSeq": sequence,
                    "capabilities": { "idempotentStart": true },
                }));
                let send = |state: &mut IsolationState, mut event: Value| {
                    state.sequence += 1;
                    event["seq"] = json!(state.sequence);
                    socket.send(event);
                };
                let send_tail = |state: &mut IsolationState| {
                    let first_id = state.first_id.clone();
                    for mut event in [
                        json!({ "type": "text-delta", "id": "old", "delta": "two" }),
                        json!({ "type": "text-end", "id": "old" }),
                        json!({ "type": "raw", "rawValue": "old" }),
                        json!({
                            "type": "finish-step",
                            "finishReason": { "unified": "stop" },
                            "usage": { "inputTokens": {}, "outputTokens": {} },
                        }),
                        json!({
                            "type": "finish",
                            "finishReason": { "unified": "stop" },
                            "totalUsage": { "inputTokens": {}, "outputTokens": {} },
                        }),
                        json!({ "type": "error", "error": "old failure" }),
                    ] {
                        event["startId"] = json!(first_id);
                        send(state, event);
                    }
                    send(state, sandbox_log("tail delivered"));
                };
                while let Some(message) = connection.incoming.recv().await {
                    let mut state = state.lock();
                    if message["type"] == "abort" && delivery == "buffered" {
                        send_tail(&mut state);
                    }
                    if message["type"] == "resume" && current_leg == 2 {
                        send_tail(&mut state);
                    }
                    if message["type"] != "start" {
                        continue;
                    }
                    let start_id = message["startId"].as_str().unwrap().to_string();
                    state.starts.push(start_id.clone());
                    if state.first_id.is_none() {
                        state.first_id = Some(start_id.clone());
                        send(&mut state, json!({ "type": "stream-start", "startId": start_id }));
                        send(
                            &mut state,
                            json!({ "type": "text-start", "id": "old", "startId": start_id }),
                        );
                        send(
                            &mut state,
                            json!({ "type": "text-delta", "id": "old", "delta": "one", "startId": start_id }),
                        );
                        continue;
                    }
                    state.second_id = Some(start_id);
                    if delivery == "replayed" && current_leg == 1 {
                        state.sequence += 10;
                        socket.close(1001, "lost new start");
                        return;
                    }
                    state.answer = Some(socket.clone());
                    answer_ready.notify_one();
                    if delivery == "in-transit" {
                        send_tail(&mut state);
                    }
                }
            });
        })
    })
    .await;
    let bridge = bridge(address);
    let tail_received = Arc::new(Notify::new());
    let mut options = bridge.options();
    options.observability = Some(tail_reporter(tail_received.clone()));
    let session = bridge.start(options).await;

    let abort = CancellationToken::new();
    let text_received = Arc::new(Notify::new());
    let first = session
        .do_prompt_turn(prompt("Count.", Some(abort.clone()), {
            let text_received = text_received.clone();
            move |part| {
                if part_type(&part) == "text-delta" {
                    text_received.notify_one();
                }
            }
        }))
        .await
        .unwrap();
    within(text_received.notified()).await;
    abort.cancel();
    assert!(within(first.done()).await.is_err());
    if delivery == "buffered" {
        within(tail_received.notified()).await;
    }
    let events: Arc<Mutex<Vec<HarnessV1StreamPart>>> = Arc::default();
    let second = session
        .do_prompt_turn(prompt("Answer pineapple.", None, {
            let events = events.clone();
            move |part| events.lock().push(part)
        }))
        .await
        .unwrap();
    if delivery != "buffered" {
        within(tail_received.notified()).await;
    }
    within(answer_ready.notified()).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(events.lock().is_empty(), "{delivery}");
    let mut done = second.done();
    assert!(futures::poll!(&mut done).is_pending());

    {
        let mut state = state.lock();
        let second_id = state.second_id.clone();
        let socket = state.answer.clone().unwrap();
        for mut event in [
            json!({ "type": "stream-start" }),
            json!({ "type": "text-start", "id": "new" }),
            json!({ "type": "text-delta", "id": "new", "delta": "pineapple" }),
            json!({ "type": "text-end", "id": "new" }),
            json!({
                "type": "finish",
                "finishReason": { "unified": "stop" },
                "totalUsage": { "inputTokens": {}, "outputTokens": {} },
            }),
        ] {
            state.sequence += 1;
            event["seq"] = json!(state.sequence);
            event["startId"] = json!(second_id);
            socket.send(event);
        }
    }
    within(done).await.unwrap();
    let parts: Vec<Value> = events
        .lock()
        .iter()
        .map(|part| serde_json::to_value(part).unwrap())
        .collect();
    assert_eq!(
        parts
            .iter()
            .map(|part| part["type"].clone())
            .collect::<Vec<_>>(),
        [
            json!("stream-start"),
            json!("text-start"),
            json!("text-delta"),
            json!("text-end"),
            json!("finish")
        ],
        "{delivery}"
    );
    assert_eq!(
        parts
            .iter()
            .filter(|part| part["type"] == "text-delta")
            .map(|part| part["delta"].clone())
            .collect::<Vec<_>>(),
        [json!("pineapple")]
    );
    assert!(parts.iter().all(|part| part.get("startId").is_none()));
    let (first_id, second_id, starts) = {
        let state = state.lock();
        (
            state.first_id.clone().unwrap(),
            state.second_id.clone().unwrap(),
            state.starts.clone(),
        )
    };
    assert_ne!(second_id, first_id);
    let expected = if delivery == "replayed" {
        vec![first_id, second_id.clone(), second_id]
    } else {
        vec![first_id, second_id]
    };
    assert_eq!(starts, expected, "{delivery}");
    let _ = session.do_destroy().await;
}

// ─── startTurn ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
struct StartTurn {
    drop: bool,
    accept_before_drop: bool,
    finish_before_reconnect: bool,
    idempotent_start: bool,
    start_during_reconnect: bool,
    submit_during_reconnect: bool,
    submit_before_first_event: bool,
}

fn start_turn(drop: bool) -> StartTurn {
    StartTurn {
        drop,
        idempotent_start: true,
        ..Default::default()
    }
}

struct StartTurnState {
    frames: Vec<(usize, Value)>,
    accepted: HashSet<String>,
    legs: usize,
    state: &'static str,
    last_seq: u64,
    first_leg: Option<Sender>,
    reconnect_hello: Option<Box<dyn FnOnce() + Send>>,
}

struct StartedTurn {
    done: crate::harness::errors::HarnessResult<()>,
    frames: Vec<(usize, Value)>,
    accepted: usize,
    events: Vec<String>,
    _session: Box<dyn HarnessV1Session>,
}

impl StartedTurn {
    fn types(&self, leg: Option<usize>) -> Vec<String> {
        self.frames
            .iter()
            .filter(|(frame_leg, _)| leg.is_none_or(|leg| *frame_leg == leg))
            .map(|(_, frame)| frame["type"].as_str().unwrap().to_string())
            .collect()
    }

    fn starts(&self) -> Vec<(usize, Value)> {
        self.frames
            .iter()
            .filter(|(_, frame)| frame["type"] == "start")
            .cloned()
            .collect()
    }
}

/// `startTurn(options)`, awaiting the turn's `done`.
async fn run_start_turn(options: StartTurn) -> StartedTurn {
    let state = Arc::new(Mutex::new(StartTurnState {
        frames: Vec::new(),
        accepted: HashSet::new(),
        legs: 0,
        state: "waiting",
        last_seq: 0,
        first_leg: None,
        reconnect_hello: None,
    }));
    let reconnecting = Arc::new(Notify::new());
    let address = bridge_server({
        let state = state.clone();
        let reconnecting = reconnecting.clone();
        Arc::new(move |mut connection: Connection| {
            let socket = connection.sender();
            let state = state.clone();
            let leg = {
                let mut state = state.lock();
                state.legs += 1;
                state.legs
            };
            let send_hello = {
                let state = state.clone();
                let socket = socket.clone();
                move || {
                    let (bridge_state, last_seq) = {
                        let state = state.lock();
                        (state.state, state.last_seq)
                    };
                    socket.send(json!({
                        "type": "bridge-hello",
                        "state": bridge_state,
                        "lastSeq": last_seq,
                        "capabilities": {
                            "idempotentStart": options.idempotent_start,
                            "experimental_userMessageResponses":
                                options.submit_during_reconnect || options.submit_before_first_event,
                        },
                    }));
                }
            };
            if leg == 1 {
                state.lock().first_leg = Some(socket.clone());
            }
            if (options.start_during_reconnect || options.submit_during_reconnect) && leg == 2 {
                state.lock().reconnect_hello = Some(Box::new(send_hello));
                reconnecting.notify_one();
            } else {
                send_hello();
            }
            tokio::spawn(async move {
                let finish = |state: &mut StartTurnState| {
                    state.state = "waiting";
                    state.last_seq = 1;
                    socket.send(finish_frame(1));
                };
                while let Some(message) = connection.incoming.recv().await {
                    let mut state = state.lock();
                    state.frames.push((leg, message.clone()));
                    if message["type"] == "resume" && !state.accepted.is_empty() {
                        finish(&mut state);
                    }
                    if message["type"] == "user-message" {
                        if options.submit_before_first_event && leg == 1 {
                            continue;
                        }
                        let accepted = !state.accepted.is_empty();
                        socket.send(json!({
                            "type": "user-message-response",
                            "messageId": message["messageId"],
                            "accepted": accepted,
                        }));
                        if accepted && state.state == "running" {
                            finish(&mut state);
                        }
                    }
                    if message["type"] != "start" {
                        continue;
                    }
                    let start_id = message["startId"].as_str().unwrap().to_string();
                    if options.drop && leg == if options.start_during_reconnect { 2 } else { 1 } {
                        if options.accept_before_drop {
                            state.accepted.insert(start_id);
                            state.state = if options.finish_before_reconnect {
                                "waiting"
                            } else {
                                "running"
                            };
                            state.last_seq = if options.finish_before_reconnect { 1 } else { 0 };
                        }
                        socket.close(1001, "leg dropped");
                        return;
                    }
                    if state.accepted.contains(&start_id) {
                        continue;
                    }
                    state.accepted.insert(start_id);
                    state.state = "running";
                    if !options.submit_during_reconnect && !options.submit_before_first_event {
                        finish(&mut state);
                    }
                }
            });
        })
    })
    .await;
    let bridge = bridge(address);
    let session = bridge.start(bridge.options()).await;
    if options.start_during_reconnect {
        let first_leg = state.lock().first_leg.clone().unwrap();
        first_leg.close(1001, "idle drop");
        within(reconnecting.notified()).await;
    }
    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let abort = CancellationToken::new();
    let mut turn = prompt("Hello", Some(abort.clone()), {
        let events = events.clone();
        move |part| events.lock().push(part_type(&part))
    });
    turn.settings.model = Some("m".into());
    let control = session.do_prompt_turn(turn).await.unwrap();
    let release_hello = || {
        let hello = state.lock().reconnect_hello.take();
        if let Some(hello) = hello {
            hello();
        }
    };
    let done = if options.submit_before_first_event {
        let (submitted, done) = within(async {
            tokio::join!(
                control.submit_user_message("Follow-up".into()),
                control.done()
            )
        })
        .await;
        submitted.unwrap();
        done
    } else if options.submit_during_reconnect {
        if !options.start_during_reconnect {
            within(reconnecting.notified()).await;
        }
        let (submitted, done, ()) = within(async {
            tokio::join!(
                control.submit_user_message("Follow-up".into()),
                control.done(),
                async {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    release_hello();
                }
            )
        })
        .await;
        submitted.unwrap();
        done
    } else {
        if options.start_during_reconnect {
            tokio::time::sleep(Duration::from_millis(20)).await;
            release_hello();
        }
        within(control.done()).await
    };
    abort.cancel();
    let state = state.lock();
    StartedTurn {
        done,
        frames: state.frames.clone(),
        accepted: state.accepted.len(),
        events: events.lock().clone(),
        _session: session,
    }
}

#[tokio::test]
async fn retries_a_lost_start_before_resending_a_pending_user_message() {
    let turn = run_start_turn(StartTurn {
        submit_before_first_event: true,
        ..start_turn(true)
    })
    .await;
    turn.done.as_ref().unwrap();
    assert_eq!(turn.types(Some(2)), ["resume", "start", "user-message"]);
    assert_eq!(turn.accepted, 1);
    assert_eq!(turn.events, ["finish"]);
}

#[tokio::test]
async fn retries_a_lost_start_before_steering_queued_during_reconnect() {
    let turn = run_start_turn(StartTurn {
        submit_during_reconnect: true,
        ..start_turn(true)
    })
    .await;
    turn.done.as_ref().unwrap();
    assert_eq!(turn.types(Some(2)), ["resume", "start", "user-message"]);
    assert_eq!(turn.accepted, 1);
    assert_eq!(turn.events, ["finish"]);
}

#[tokio::test]
async fn starts_a_turn_without_a_dropped_connection() {
    let turn = run_start_turn(start_turn(false)).await;
    turn.done.as_ref().unwrap();
    assert_eq!(turn.accepted, 1);
    assert_eq!(turn.types(None), ["start"]);
    assert_eq!(turn.events, ["finish"]);
}

#[tokio::test]
async fn resends_a_lost_start_with_the_same_identifier_after_reconnect() {
    let turn = run_start_turn(start_turn(true)).await;
    turn.done.as_ref().unwrap();
    let starts = turn.starts();
    assert_eq!(
        starts.iter().map(|(leg, _)| *leg).collect::<Vec<_>>(),
        [1, 2]
    );
    assert!(starts[0].1["startId"].is_string());
    assert_eq!(starts[1].1["startId"], starts[0].1["startId"]);
    assert_eq!(turn.accepted, 1);
    assert_eq!(turn.events, ["finish"]);
}

#[tokio::test]
async fn starts_a_turn_submitted_during_an_idle_reconnect() {
    for idempotent_start in [false, true] {
        let turn = run_start_turn(StartTurn {
            start_during_reconnect: true,
            idempotent_start,
            ..start_turn(false)
        })
        .await;
        turn.done.as_ref().unwrap();
        assert_eq!(turn.types(None), ["resume", "start"]);
        assert_eq!(turn.starts()[0].0, 2);
        assert_eq!(turn.accepted, 1);
        assert_eq!(turn.events, ["finish"]);
    }
}

#[tokio::test]
async fn preserves_queued_start_ordering_before_user_messages() {
    for idempotent_start in [false, true] {
        let turn = run_start_turn(StartTurn {
            start_during_reconnect: true,
            submit_during_reconnect: true,
            idempotent_start,
            ..start_turn(false)
        })
        .await;
        turn.done.as_ref().unwrap();
        assert_eq!(turn.types(None)[..3], ["resume", "start", "user-message"]);
        assert_eq!(turn.starts().len(), 1);
        assert!(!turn.types(None).contains(&"abort".to_string()));
        assert_eq!(turn.events, ["finish"]);
    }
}

#[tokio::test]
async fn recovers_a_start_first_sent_after_an_idle_reconnect_and_then_lost() {
    for idempotent_start in [false, true] {
        let turn = run_start_turn(StartTurn {
            start_during_reconnect: true,
            idempotent_start,
            ..start_turn(true)
        })
        .await;
        let starts = turn.starts();
        if idempotent_start {
            turn.done.as_ref().unwrap();
            assert_eq!(
                starts.iter().map(|(leg, _)| *leg).collect::<Vec<_>>(),
                [2, 3]
            );
            assert_eq!(starts[1].1["startId"], starts[0].1["startId"]);
            assert_eq!(turn.accepted, 1);
            assert_eq!(turn.events, ["finish"]);
        } else {
            let error = turn.done.as_ref().unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("does not support safe start retries"),
                "{error}"
            );
            assert_eq!(starts.iter().map(|(leg, _)| *leg).collect::<Vec<_>>(), [2]);
        }
    }
}

#[tokio::test]
async fn does_not_duplicate_an_accepted_turn() {
    for (finish_before_reconnect, idempotent_start) in
        [(false, true), (true, true), (false, false), (true, false)]
    {
        let turn = run_start_turn(StartTurn {
            accept_before_drop: true,
            finish_before_reconnect,
            idempotent_start,
            ..start_turn(true)
        })
        .await;
        turn.done.as_ref().unwrap();
        assert_eq!(turn.accepted, 1);
        let starts = turn.starts();
        assert_eq!(starts.len(), if idempotent_start { 2 } else { 1 });
        assert_eq!(
            starts
                .iter()
                .map(|(_, frame)| frame["startId"].to_string())
                .collect::<HashSet<_>>()
                .len(),
            1
        );
        assert_eq!(turn.events, ["finish"]);
    }
}

#[tokio::test]
async fn fails_rather_than_hanging_or_retrying_unsafely_with_an_older_waiting_bridge() {
    let turn = run_start_turn(StartTurn {
        idempotent_start: false,
        ..start_turn(true)
    })
    .await;
    let error = turn.done.as_ref().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not support safe start retries"),
        "{error}"
    );
    assert_eq!(turn.starts().len(), 1);
}

#[allow(dead_code)]
fn _assert_control_is_object_safe(_: &dyn HarnessV1PromptControl) {}
