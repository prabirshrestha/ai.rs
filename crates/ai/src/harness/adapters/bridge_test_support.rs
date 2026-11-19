//! [rust] Doubles shared by the adapter test ports.
//!
//! TS mocks `ws`, `SandboxChannel`, `waitForBridgeReady` and
//! `markBridgeStarting`. The ports run the real channel and readiness code
//! over a fake bridge socket and a fake sandbox whose spawned bridge prints
//! its `bridge-ready` line; `markBridgeStarting` shows up as its metadata
//! write.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::FutureExt as _;
use futures::StreamExt as _;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::*;
use crate::harness::spec::harness::{
    HarnessV1PromptControl, HarnessV1PromptTurnOptions, HarnessV1SandboxSessionRef,
    HarnessV1Session, HarnessV1StartOptions,
};
use crate::harness::spec::lifecycle_state::HarnessV1TurnSettings;
use crate::harness::spec::network_sandbox::{
    HarnessV1NetworkSandboxSession, HarnessV1PortEndpoint, HarnessV1PortProtocol,
    HarnessV1RequestTransformation,
};
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::utils::bridge_auth_protocol::serialize_bridge_auth_challenge_frame;
use crate::harness::utils::sandbox_channel::{
    SharedWebSocket, WebSocket, WebSocketEvent, WebSocketEventHandler,
};
use crate::harness::utils::{WebSocketConstructor, plain_error};

// ─── Fake bridge socket ────────────────────────────────────────────────

/// One bridge WebSocket.
pub struct FakeSocket {
    this: Weak<FakeSocket>,
    handler: Mutex<Option<WebSocketEventHandler>>,
    buffered: Mutex<Vec<WebSocketEvent>>,
    sent: Mutex<Vec<String>>,
    closed: AtomicBool,
    send_error: Arc<Mutex<Option<String>>>,
}

impl FakeSocket {
    fn emit_event(&self, event: WebSocketEvent) {
        let handler = self.handler.lock().clone();
        match handler {
            Some(handler) => handler(event),
            None => self.buffered.lock().push(event),
        }
    }

    /// `channel.emit(event)`.
    pub fn emit(&self, event: Value) {
        self.emit_event(WebSocketEvent::Message(event.to_string()));
    }

    /// `channel.emitClose(...)`.
    pub fn emit_close(&self, code: u16, reason: &str) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        self.emit_event(WebSocketEvent::Close {
            code,
            reason: reason.to_string(),
        });
    }

    /// Every frame the host sent.
    pub fn raw_sent(&self) -> Vec<Value> {
        self.sent
            .lock()
            .iter()
            .filter_map(|text| serde_json::from_str::<Value>(text).ok())
            .collect()
    }

    /// Frames the host sent, without the auth proof and resume frames.
    pub fn sent(&self) -> Vec<Value> {
        self.sent
            .lock()
            .iter()
            .filter_map(|text| serde_json::from_str::<Value>(text).ok())
            .filter(|frame| {
                let r#type = frame.get("type").and_then(Value::as_str).unwrap_or("");
                !r#type.starts_with("bridge-auth") && r#type != "resume"
            })
            .collect()
    }
}

impl WebSocket for FakeSocket {
    fn set_event_handler(&self, handler: Option<WebSocketEventHandler>) {
        *self.handler.lock() = handler.clone();
        if let Some(handler) = handler {
            for event in std::mem::take(&mut *self.buffered.lock()) {
                handler(event);
            }
        }
    }

    fn is_open(&self) -> bool {
        !self.closed.load(Ordering::SeqCst)
    }

    fn send(&self, text: &str) -> HarnessResult<()> {
        if let Some(error) = self.send_error.lock().clone() {
            return Err(plain_error(error));
        }
        self.sent.lock().push(text.to_string());
        let r#type = serde_json::from_str::<Value>(text).ok().and_then(|frame| {
            frame
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
        let socket = self.self_ref();
        match r#type.as_deref() {
            Some("destroy") => {
                tokio::spawn(async move { socket.emit_close(1000, "destroy") });
            }
            Some("stop") => {
                tokio::spawn(async move {
                    socket.emit(json!({ "type": "bridge-stop", "data": {} }));
                    socket.emit_close(1000, "stop");
                });
            }
            _ => {}
        }
        Ok(())
    }

    fn close(&self, code: Option<u16>, reason: Option<&str>) {
        let socket = self.self_ref();
        let code = code.unwrap_or(1000);
        let reason = reason.unwrap_or("").to_string();
        tokio::spawn(async move { socket.emit_close(code, &reason) });
    }

    fn terminate(&self) {
        let socket = self.self_ref();
        tokio::spawn(async move { socket.emit_close(1006, "") });
    }
}

impl FakeSocket {
    fn self_ref(&self) -> Arc<FakeSocket> {
        self.this.upgrade().expect("socket is alive")
    }
}

/// A `new WebSocket(url, { headers })` call.
pub type SocketCall = (String, Option<IndexMap<String, String>>);

/// The fake `ws` module plus `harnessUtilsMocks.channels`.
#[derive(Clone, Default)]
pub struct FakeBridge {
    sockets: Arc<Mutex<Vec<Arc<FakeSocket>>>>,
    /// `webSocketMocks.calls`.
    pub calls: Arc<Mutex<Vec<SocketCall>>>,
    /// `harnessUtilsMocks.openErrors`.
    pub open_errors: Arc<Mutex<VecDeque<String>>>,
    /// The `bridge-hello` frame each socket sends; `{ type: 'bridge-hello' }`
    /// when unset.
    pub hello: Arc<Mutex<Option<Value>>>,
    /// `channelSendError`: every socket send fails with it while set.
    pub send_error: Arc<Mutex<Option<String>>>,
}

impl FakeBridge {
    pub fn constructor(&self) -> WebSocketConstructor {
        let bridge = self.clone();
        Arc::new(move |url, headers| {
            bridge.calls.lock().push((url, headers));
            if let Some(error) = bridge.open_errors.lock().pop_front() {
                return async move { Err(plain_error(error)) }.boxed();
            }
            let socket = Arc::new_cyclic(|this| FakeSocket {
                this: this.clone(),
                handler: Mutex::default(),
                buffered: Mutex::new(vec![
                    WebSocketEvent::Message(serialize_bridge_auth_challenge_frame(&"a".repeat(64))),
                    WebSocketEvent::Message(
                        bridge
                            .hello
                            .lock()
                            .clone()
                            .unwrap_or_else(|| json!({ "type": "bridge-hello" }))
                            .to_string(),
                    ),
                ]),
                sent: Mutex::default(),
                closed: AtomicBool::new(false),
                send_error: bridge.send_error.clone(),
            });
            bridge.sockets.lock().push(socket.clone());
            async move { Ok(socket as SharedWebSocket) }.boxed()
        })
    }

    pub fn count(&self) -> usize {
        self.sockets.lock().len()
    }

    pub fn socket(&self, index: usize) -> Arc<FakeSocket> {
        self.sockets.lock()[index].clone()
    }

    /// The most recently opened socket.
    pub fn last_socket(&self) -> Arc<FakeSocket> {
        self.sockets
            .lock()
            .last()
            .expect("a socket was opened")
            .clone()
    }

    /// Frames the host sent over every socket, in socket order.
    pub fn sent(&self) -> Vec<Value> {
        self.sockets
            .lock()
            .iter()
            .flat_map(|socket| socket.sent())
            .collect()
    }
}

// ─── Fake sandbox ──────────────────────────────────────────────────────

/// A recorded `spawn`.
#[derive(Debug, Clone)]
pub struct Spawn {
    pub command: String,
    pub env: IndexMap<String, String>,
}

/// What a [`FakeSandbox`] records.
#[derive(Default)]
pub struct SandboxRecord {
    pub runs: Vec<String>,
    pub spawns: Vec<Spawn>,
    pub writes: Vec<(String, String)>,
    pub kills: Vec<String>,
    pub files: IndexMap<String, String>,
    pub transformations: Vec<Vec<HarnessV1RequestTransformation>>,
}

/// `fakeSandbox(...)`.
#[derive(Clone)]
pub struct FakeSandbox {
    pub record: Arc<Mutex<SandboxRecord>>,
    pub home_dir: String,
    pub bridge_port_endpoint: HarnessV1PortEndpoint,
    pub supports_transformations: bool,
    pub transformation_error: Option<String>,
    pub stops: Arc<AtomicUsize>,
    /// What a spawned bridge prints and how it exits; by default it prints
    /// its `bridge-ready` line on port 4319 and exits with 0.
    pub spawn_stdout: Option<String>,
    pub spawn_stderr: String,
    pub exit_code: i32,
    /// `ports`.
    pub ports: Vec<u16>,
}

impl Default for FakeSandbox {
    fn default() -> Self {
        Self {
            record: Arc::default(),
            home_dir: "/home/agent".into(),
            bridge_port_endpoint: HarnessV1PortEndpoint {
                url: "ws://127.0.0.1:4319".into(),
                headers: None,
            },
            supports_transformations: false,
            transformation_error: None,
            stops: Arc::default(),
            spawn_stdout: None,
            spawn_stderr: String::new(),
            exit_code: 0,
            ports: vec![4319],
        }
    }
}

impl FakeSandbox {
    pub fn runs(&self) -> Vec<String> {
        self.record.lock().runs.clone()
    }

    pub fn spawns(&self) -> Vec<Spawn> {
        self.record.lock().spawns.clone()
    }

    pub fn writes(&self) -> Vec<(String, String)> {
        self.record.lock().writes.clone()
    }

    /// Writes other than the `markBridgeStarting` marker, which TS mocks.
    pub fn file_writes(&self) -> Vec<(String, String)> {
        self.writes()
            .into_iter()
            .filter(|(path, _)| !path.ends_with("/bridge-meta.json"))
            .collect()
    }

    /// How often `markBridgeStarting` wrote its metadata marker.
    pub fn bridge_starting_marks(&self) -> usize {
        self.writes()
            .iter()
            .filter(|(path, content)| {
                path.ends_with("/bridge-meta.json") && content.contains("\"starting\"")
            })
            .count()
    }

    pub fn network(&self) -> HarnessV1SandboxSessionRef {
        HarnessV1SandboxSessionRef::Network(Arc::new(self.clone()))
    }

    pub fn basic(&self) -> HarnessV1SandboxSessionRef {
        HarnessV1SandboxSessionRef::Basic(Arc::new(self.clone()))
    }
}

fn text_stream(text: &str) -> ByteStream {
    let chunks: Vec<HarnessResult<Bytes>> = if text.is_empty() {
        Vec::new()
    } else {
        vec![Ok(Bytes::from(text.to_string()))]
    };
    futures::stream::iter(chunks).boxed()
}

struct FakeProcess {
    command: String,
    record: Arc<Mutex<SandboxRecord>>,
    exit_code: i32,
}

#[async_trait]
impl SandboxProcessControl for FakeProcess {
    async fn wait(&self) -> HarnessResult<SandboxExitStatus> {
        Ok(SandboxExitStatus {
            exit_code: self.exit_code,
        })
    }

    async fn kill(&self) -> HarnessResult<()> {
        self.record.lock().kills.push(self.command.clone());
        Ok(())
    }
}

#[async_trait]
impl SandboxSession for FakeSandbox {
    fn description(&self) -> String {
        "fake sandbox".into()
    }

    async fn read_file(&self, _o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
        unimplemented!("read_file")
    }

    async fn read_binary_file(
        &self,
        o: SandboxReadBinaryFileOptions,
    ) -> HarnessResult<Option<Bytes>> {
        Ok(self
            .record
            .lock()
            .files
            .get(&o.path)
            .map(|content| Bytes::from(content.clone())))
    }

    async fn read_text_file(&self, o: SandboxReadTextFileOptions) -> HarnessResult<Option<String>> {
        Ok(self.record.lock().files.get(&o.path).cloned())
    }

    async fn write_file(&self, _o: SandboxWriteFileOptions) -> HarnessResult<()> {
        unimplemented!("write_file")
    }

    async fn write_binary_file(&self, o: SandboxWriteBinaryFileOptions) -> HarnessResult<()> {
        let content = String::from_utf8_lossy(&o.content).into_owned();
        let mut record = self.record.lock();
        record.writes.push((o.path.clone(), content.clone()));
        record.files.insert(o.path, content);
        Ok(())
    }

    async fn write_text_file(&self, o: SandboxWriteTextFileOptions) -> HarnessResult<()> {
        let mut record = self.record.lock();
        record.writes.push((o.path.clone(), o.content.clone()));
        // The bridge never reaches `waiting`, so readiness comes from stdout.
        if !o.path.ends_with("/bridge-meta.json") {
            record.files.insert(o.path, o.content);
        }
        Ok(())
    }

    async fn stat(&self, _o: SandboxStatOptions) -> HarnessResult<Option<SandboxFileStat>> {
        Ok(None)
    }

    async fn list_directory(
        &self,
        _o: SandboxListDirectoryOptions,
    ) -> HarnessResult<Option<Vec<SandboxDirectoryEntry>>> {
        Ok(None)
    }

    async fn delete_file(&self, o: SandboxDeleteFileOptions) -> HarnessResult<bool> {
        Ok(self.record.lock().files.shift_remove(&o.path).is_some())
    }

    async fn spawn(&self, o: SandboxSpawnOptions) -> HarnessResult<SandboxProcess> {
        let command = o.process.command.clone();
        self.record.lock().spawns.push(Spawn {
            command: command.clone(),
            env: o.process.env.clone().unwrap_or_default(),
        });
        Ok(SandboxProcess {
            pid: None,
            stdout: text_stream(
                self.spawn_stdout
                    .as_deref()
                    .unwrap_or("{\"type\":\"bridge-ready\",\"port\":4319}\n"),
            ),
            stderr: text_stream(&self.spawn_stderr),
            stdin: None,
            control: Arc::new(FakeProcess {
                command,
                record: self.record.clone(),
                exit_code: self.exit_code,
            }),
        })
    }

    async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
        let command = o.process.command.clone();
        let mut record = self.record.lock();
        record.runs.push(command.clone());
        if command == "printf \"%s\" \"$HOME\"" {
            return Ok(SandboxRunResult {
                stdout: self.home_dir.clone(),
                ..Default::default()
            });
        }
        if command == "pwd" {
            return Ok(SandboxRunResult {
                stdout: "/workspace\n".into(),
                ..Default::default()
            });
        }
        let manifest_move = regex::Regex::new(r"^mv -f '([^']+)' '([^']+)'$").unwrap();
        if let Some(captures) = manifest_move.captures(&command)
            && let Some(content) = record.files.get(&captures[1]).cloned()
        {
            record.files.insert(captures[2].to_string(), content);
        }
        Ok(SandboxRunResult::default())
    }
}

#[async_trait]
impl HarnessV1NetworkSandboxSession for FakeSandbox {
    fn id(&self) -> String {
        "sandbox-1".into()
    }

    fn default_working_directory(&self) -> String {
        "/workspace".into()
    }

    fn ports(&self) -> Vec<u16> {
        self.ports.clone()
    }

    async fn get_port_endpoint(
        &self,
        _port: u16,
        _protocol: Option<HarnessV1PortProtocol>,
    ) -> HarnessResult<HarnessV1PortEndpoint> {
        Ok(self.bridge_port_endpoint.clone())
    }

    async fn stop(&self) -> HarnessResult<()> {
        self.stops.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn destroy(&self) -> HarnessResult<()> {
        Ok(())
    }

    fn supports_add_request_transformations(&self) -> bool {
        self.supports_transformations
    }

    async fn add_request_transformations(
        &self,
        transformations: Vec<HarnessV1RequestTransformation>,
    ) -> HarnessResult<()> {
        if let Some(error) = &self.transformation_error {
            return Err(plain_error(error.clone()));
        }
        self.record.lock().transformations.push(transformations);
        Ok(())
    }

    fn restricted(&self) -> Arc<dyn SandboxSession> {
        Arc::new(self.clone())
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────

/// Start options for `session_id` over `sandbox`.
pub fn start_options(
    session_id: &str,
    sandbox: HarnessV1SandboxSessionRef,
) -> HarnessV1StartOptions {
    HarnessV1StartOptions {
        headers: None,
        session_id: session_id.into(),
        resume_from: None,
        continue_from: None,
        permission_mode: None,
        builtin_tool_filtering: None,
        abort_signal: None,
        observability: None,
        sandbox_session: sandbox,
        session_work_dir: "/workspace/user-project".into(),
    }
}

/// Collected stream parts of a turn.
pub type Events = Arc<Mutex<Vec<HarnessV1StreamPart>>>;

/// Prompt options with an emit that records into the returned events.
pub fn prompt_options(prompt: &str) -> (HarnessV1PromptTurnOptions, Events) {
    let events: Events = Arc::default();
    let sink = events.clone();
    (
        HarnessV1PromptTurnOptions {
            settings: HarnessV1TurnSettings {
                model: None,
                skills: Vec::new(),
                instructions: None,
                tools: Vec::new(),
            },
            prompt: prompt.into(),
            response_format: None,
            abort_signal: None,
            emit: Arc::new(move |part| sink.lock().push(part)),
        },
        events,
    )
}

/// The stream parts as JSON.
pub fn events_json(events: &Events) -> Vec<Value> {
    events
        .lock()
        .iter()
        .map(|part| serde_json::to_value(part).unwrap())
        .collect()
}

/// `unknownUsage()`.
pub fn unknown_usage() -> Value {
    json!({
        "inputTokens": {},
        "outputTokens": {},
    })
}

/// A `finish` frame.
pub fn finish_frame() -> Value {
    json!({
        "type": "finish",
        "finishReason": { "unified": "stop", "raw": "end_turn" },
        "totalUsage": unknown_usage(),
    })
}

/// `vi.waitFor(...)`.
pub async fn wait_for(mut condition: impl FnMut() -> bool) {
    for _ in 0..500 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("condition was not met in time");
}

/// Let queued channel work run.
pub async fn flush() {
    tokio::time::sleep(Duration::from_millis(30)).await;
}

/// Waits until `socket` has sent `count` frames.
pub async fn wait_for_sent(socket: &FakeSocket, count: usize) -> Vec<Value> {
    wait_for(|| socket.sent().len() >= count).await;
    socket.sent()
}

/// Waits until `socket` has sent `expected`.
pub async fn wait_for_frame(socket: &FakeSocket, expected: &Value) {
    for _ in 0..500 {
        if socket.sent().contains(expected) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!(
        "frame {expected:#} was not sent; sent {:#}",
        Value::Array(socket.sent())
    );
}

/// `expect(actual).toMatchObject(expected)`.
pub fn assert_match_object(actual: &Value, expected: &Value) {
    fn matches(actual: &Value, expected: &Value) -> bool {
        match (actual, expected) {
            (Value::Object(actual), Value::Object(expected)) => expected
                .iter()
                .all(|(key, value)| actual.get(key).is_some_and(|actual| matches(actual, value))),
            (Value::Array(actual), Value::Array(expected)) => {
                actual.len() == expected.len()
                    && actual.iter().zip(expected).all(|(a, e)| matches(a, e))
            }
            _ => actual == expected,
        }
    }
    assert!(
        matches(actual, expected),
        "expected {actual:#} to match {expected:#}"
    );
}

/// Starts a session and returns it.
pub async fn start(
    harness: &dyn crate::harness::spec::harness::HarnessV1,
    options: HarnessV1StartOptions,
) -> Box<dyn HarnessV1Session> {
    harness.do_start(options).await.expect("session starts")
}

/// Awaits a turn's `done`.
pub async fn done(control: &dyn HarnessV1PromptControl) -> HarnessResult<()> {
    tokio::time::timeout(Duration::from_secs(5), control.done())
        .await
        .expect("turn settles")
}
