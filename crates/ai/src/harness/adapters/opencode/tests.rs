//! OpenCode session tests (port of `opencode-harness.test.ts` and
//! `bridge/index.test.ts`) against a fake server: an HTTP server in this
//! process that answers the OpenCode API and scripts the event stream, and a
//! fake `opencode` executable that prints its listening line.

#![allow(clippy::field_reassign_with_default, clippy::await_holding_lock)]

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use super::*;
use crate::harness::adapters::local_sandbox::{LocalSandbox, canonical};
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::session::{HarnessEvent, HarnessSandbox, HarnessTurnSettings, NoticeLevel};
use crate::harness::tools::PermissionMode;
use crate::{
    AgentToolBuilder, AgentToolResult, AssistantContent, BeforeToolCallResult, Message, StopReason,
    ToolResultContent,
};

const SESSION: &str = "ses_1";

/// One request the fake server received.
#[derive(Debug, Clone)]
struct Request {
    method: String,
    path: String,
    body: Value,
    headers: Vec<(String, String)>,
}

type Script = Arc<dyn Fn(&Request, &Fake) -> Option<Value> + Send + Sync>;

/// The fake server.
#[derive(Clone)]
struct Fake {
    requests: Arc<StdMutex<Vec<Request>>>,
    subscribers: Arc<StdMutex<Vec<mpsc::UnboundedSender<Value>>>>,
    script: Arc<StdMutex<Script>>,
    port: u16,
}

impl Fake {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let fake = Self {
            requests: Arc::default(),
            subscribers: Arc::default(),
            script: Arc::new(StdMutex::new(Arc::new(|_: &Request, _: &Fake| None))),
            port,
        };
        let server = fake.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let fake = server.clone();
                tokio::spawn(async move {
                    let _ = fake.serve(stream).await;
                });
            }
        });
        fake
    }

    fn on(&self, script: impl Fn(&Request, &Fake) -> Option<Value> + Send + Sync + 'static) {
        *self.script.lock().unwrap() = Arc::new(script);
    }

    /// Send one event to every subscriber.
    fn emit(&self, event: Value) {
        for subscriber in self.subscribers.lock().unwrap().iter() {
            let _ = subscriber.send(event.clone());
        }
    }

    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }

    fn find(&self, method: &str, path: &str) -> Vec<Request> {
        self.requests()
            .into_iter()
            .filter(|request| {
                request.method == method && request.path.split('?').next() == Some(path)
            })
            .collect()
    }

    /// The default API answers.
    fn default_response(request: &Request) -> Value {
        let path = request.path.split('?').next().unwrap();
        match (request.method.as_str(), path) {
            ("POST", "/session") => json!({ "id": SESSION }),
            ("GET", "/mcp") => json!({ "harness-tools": { "status": "connected" } }),
            ("GET", path) if path.ends_with("/message") => json!([]),
            ("GET", path) if path.starts_with("/session/") => json!({ "id": SESSION }),
            _ => json!(true),
        }
    }

    async fn serve(self, stream: tokio::net::TcpStream) -> std::io::Result<()> {
        let (read, mut write) = stream.into_split();
        let mut reader = BufReader::new(read);
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).await? == 0 {
                return Ok(());
            }
            let mut parts = line.split_whitespace();
            let method = parts.next().unwrap_or("").to_string();
            let path = parts.next().unwrap_or("").to_string();
            let mut length = 0;
            let mut headers = Vec::new();
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).await?;
                let header = header.trim_end();
                if header.is_empty() {
                    break;
                }
                if let Some((name, value)) = header.split_once(':') {
                    let name = name.trim().to_ascii_lowercase();
                    if name == "content-length" {
                        length = value.trim().parse().unwrap_or(0);
                    }
                    headers.push((name, value.trim().to_string()));
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).await?;
            let body = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let request = Request {
                method,
                path,
                body,
                headers,
            };
            self.requests.lock().unwrap().push(request.clone());
            if request.path.starts_with("/event") {
                let (sender, mut events) = mpsc::unbounded_channel();
                self.subscribers.lock().unwrap().push(sender);
                write
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                    .await?;
                write.flush().await?;
                while let Some(event) = events.recv().await {
                    write
                        .write_all(format!("data: {event}\n\n").as_bytes())
                        .await?;
                    write.flush().await?;
                }
                return Ok(());
            }
            let script = self.script.lock().unwrap().clone();
            let response =
                script(&request, &self).unwrap_or_else(|| Self::default_response(&request));
            let (status, body) = match response.get("$status").and_then(Value::as_u64) {
                Some(status) => (status, response["body"].to_string()),
                None => (200, response.to_string()),
            };
            write
                .write_all(
                    format!(
                        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await?;
            write.flush().await?;
        }
    }
}

fn is(request: &Request, method: &str, path: &str) -> bool {
    request.method == method && request.path.split('?').next() == Some(path)
}

fn ev(event_type: &str, properties: Value) -> Value {
    json!({ "type": event_type, "properties": properties })
}

fn status(kind: &str) -> Value {
    ev(
        "session.status",
        json!({ "sessionID": SESSION, "status": { "type": kind } }),
    )
}

fn assistant_updated(id: &str) -> Value {
    ev(
        "message.updated",
        json!({ "info": {
            "id": id, "sessionID": SESSION, "role": "assistant",
            "providerID": "anthropic", "modelID": "claude",
        } }),
    )
}

fn text(id: &str, text: &str) -> Vec<Value> {
    vec![
        ev(
            "session.next.text.started",
            json!({ "sessionID": SESSION, "textID": id }),
        ),
        ev(
            "session.next.text.delta",
            json!({ "sessionID": SESSION, "textID": id, "delta": text }),
        ),
        ev(
            "session.next.text.ended",
            json!({ "sessionID": SESSION, "textID": id, "text": text }),
        ),
    ]
}

fn step_ended(input: u64, output: u64) -> Value {
    ev(
        "session.next.step.ended",
        json!({
            "sessionID": SESSION,
            "finish": "stop",
            "tokens": { "input": input, "output": output, "reasoning": 0, "cache": { "read": 0, "write": 0 } },
        }),
    )
}

/// A complete text answer.
fn answer(fake: &Fake, reply: &str) {
    fake.emit(status("busy"));
    fake.emit(assistant_updated("msg_1"));
    for event in text("text-1", reply) {
        fake.emit(event);
    }
    fake.emit(step_ended(3, 2));
    fake.emit(status("idle"));
}

fn write_script(path: &str, content: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, content).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

struct Fixture {
    _temp: TempDir,
    root: String,
    fake: Fake,
}

const FAKE_OPENCODE: &str = r#"#!/bin/sh
FAKE_DIR=@DIR@
FAKE_PORT=@PORT@
echo spawn >> "$FAKE_DIR/spawns"
printf '%s\n' "$@" > "$FAKE_DIR/args"
printf '%s' "$OPENCODE_CONFIG_CONTENT" > "$FAKE_DIR/config.json"
env > "$FAKE_DIR/env"
echo "starting"
echo "opencode server listening on http://127.0.0.1:$FAKE_PORT"
exec sleep 600
"#;

async fn fixture() -> Fixture {
    let temp = mkdtemp("opencode-");
    let root = canonical(temp.path());
    for dir in ["home", "work", ".fake"] {
        std::fs::create_dir_all(format!("{root}/{dir}")).unwrap();
    }
    let fake = Fake::start().await;
    write_script(
        &format!("{root}/.fake/opencode"),
        &FAKE_OPENCODE
            .replace("@DIR@", &shell_quote(&format!("{root}/.fake")))
            .replace("@PORT@", &fake.port.to_string()),
    );
    Fixture {
        _temp: temp,
        root,
        fake,
    }
}

impl Fixture {
    fn settings(&self) -> OpenCodeHarnessSettings {
        OpenCodeHarnessSettings {
            executable_override: Some(shell_quote(&format!("{}/.fake/opencode", self.root))),
            process_env: Some(IndexMap::from([(
                "ANTHROPIC_API_KEY".to_string(),
                "sk-test".to_string(),
            )])),
            read_subscription: Some(Arc::new(|_, _| Box::pin(async { Ok(None) }))),
            startup_timeout: Some(Duration::from_secs(5)),
            ..Default::default()
        }
    }

    fn sandbox(&self) -> HarnessSandbox {
        HarnessSandbox::Basic(Arc::new(
            LocalSandbox::new(format!("{}/work", self.root))
                .with_home(format!("{}/home", self.root)),
        ))
    }

    fn start_options(&self) -> HarnessStartOptions {
        HarnessStartOptions::new("session-1", self.sandbox(), format!("{}/work", self.root))
    }

    async fn start(&self, options: HarnessStartOptions) -> Box<dyn HarnessSession> {
        self.start_with(self.settings(), options).await
    }

    async fn start_with(
        &self,
        settings: OpenCodeHarnessSettings,
        options: HarnessStartOptions,
    ) -> Box<dyn HarnessSession> {
        OpenCodeHarness::new(settings)
            .unwrap()
            .start(options)
            .await
            .unwrap()
    }

    fn fake_file(&self, name: &str) -> String {
        std::fs::read_to_string(format!("{}/.fake/{name}", self.root)).unwrap_or_default()
    }

    fn config(&self) -> Value {
        serde_json::from_str(&self.fake_file("config.json")).unwrap()
    }

    fn relay_dir(&self) -> String {
        format!(
            "{}/home/.ai-sdk-harness/.agent-runs/session-1/host-tools",
            self.root
        )
    }
}

/// Call a host tool through the relay's FIFOs, the way OpenCode's MCP
/// client does.
async fn mcp_call(dir: String, name: &str, arguments: Value) -> Value {
    let request = json!({
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": { "name": name, "arguments": arguments },
    });
    tokio::task::spawn_blocking(move || {
        use std::io::{BufRead, Write};
        let mut req = std::fs::OpenOptions::new()
            .write(true)
            .open(format!("{dir}/req"))
            .unwrap();
        writeln!(req, "{request}").unwrap();
        let res = std::fs::File::open(format!("{dir}/res")).unwrap();
        let mut line = String::new();
        std::io::BufReader::new(res).read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    })
    .await
    .unwrap()
}

fn options(settings: HarnessTurnSettings) -> HarnessTurnOptions {
    HarnessTurnOptions {
        settings,
        response_format: None,
        cancellation_token: None,
        before_tool_call: None,
        after_tool_call: None,
    }
}

async fn run(
    session: &dyn HarnessSession,
    options: HarnessTurnOptions,
) -> (Vec<HarnessEvent>, Vec<Message>) {
    let turn = session
        .prompt(Message::user_text("hi"), options)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(10), turn.collect())
        .await
        .unwrap_or_else(|_| panic!("turn timed out"))
        .unwrap()
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|message| match message {
            Message::Assistant(message) => message
                .content
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

fn tool_calls(messages: &[Message]) -> Vec<(String, String, Value)> {
    messages
        .iter()
        .flat_map(|message| match message {
            Message::Assistant(message) => message
                .content
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(call) => {
                        Some((call.id.clone(), call.name.clone(), call.arguments.clone()))
                    }
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

fn tool_results(messages: &[Message]) -> Vec<(String, String, bool, String)> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(result) => Some((
                result.tool_call_id.clone(),
                result.tool_name.clone(),
                result.is_error,
                result
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ToolResultContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<String>(),
            )),
            _ => None,
        })
        .collect()
}

fn last_assistant(messages: &[Message]) -> &crate::AssistantMessage {
    messages
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::Assistant(message) => Some(message),
            _ => None,
        })
        .unwrap()
}

#[tokio::test]
async fn streams_a_prompt_turn_and_reuses_the_server() {
    let fixture = fixture().await;
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            answer(fake, "hello world");
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    let settings = HarnessTurnSettings {
        model: Some("anthropic/claude".into()),
        instructions: Some("Be brief.".into()),
        ..Default::default()
    };
    let (_, messages) = run(session.as_ref(), options(settings.clone())).await;
    assert_eq!(texts(&messages), ["hello world"]);
    let assistant = last_assistant(&messages);
    assert_eq!(assistant.model, "anthropic/claude");
    assert_eq!(assistant.api, OPENCODE_API);
    assert_eq!(assistant.usage.input, 3);
    assert_eq!(assistant.usage.output, 2);
    assert_eq!(assistant.stop_reason, StopReason::Stop);

    let prompt = &fixture.fake.find("POST", "/session/ses_1/prompt_async")[0];
    assert_eq!(
        prompt.body,
        json!({ "parts": [{ "type": "text", "text": "hi" }], "system": "Be brief." })
    );
    assert!(
        prompt
            .headers
            .iter()
            .any(|(name, value)| name == "authorization" && value.starts_with("Basic "))
    );
    assert_eq!(
        fixture.fake.find("POST", "/api/session/ses_1/model")[0].body,
        json!({ "model": { "id": "claude", "providerID": "anthropic" } })
    );
    let event = &fixture.fake.find("GET", "/event")[0];
    assert!(event.path.contains("directory="), "{}", event.path);

    let config = fixture.config();
    assert_eq!(config["permission"]["read"], "allow");
    assert_eq!(config["share"], "disabled");
    assert_eq!(config["model"], "anthropic/claude");
    assert_eq!(
        config["provider"],
        json!({ "anthropic": { "options": { "apiKey": "sk-test" } } })
    );
    assert_eq!(
        config["instructions"],
        json!([format!("{}/work/AGENTS.md", fixture.root)])
    );
    assert!(config.get("mcp").is_none());
    let args = fixture.fake_file("args");
    assert!(
        args.contains("serve")
            && args.contains("--hostname=127.0.0.1")
            && args.contains("--port=0")
    );
    let env = fixture.fake_file("env");
    assert!(env.contains("OPENCODE_DISABLE_PROJECT_CONFIG=1"));
    assert!(env.contains("ANTHROPIC_API_KEY=sk-test"));
    assert!(env.contains(&format!(
        "AI_SDK_HARNESS_CLIENT_APP=ai-sdk-harness-opencode/{VERSION}"
    )));

    // Same configuration: the server and the session are reused.
    let (_, messages) = run(session.as_ref(), options(settings)).await;
    assert_eq!(texts(&messages), ["hello world"]);
    assert_eq!(fixture.fake_file("spawns").lines().count(), 1);
    assert_eq!(fixture.fake.find("POST", "/session").len(), 1);

    let state = session.detach().await.unwrap();
    assert_eq!(state.harness_id, OPENCODE_HARNESS_ID);
    assert_eq!(state.data["openCodeSessionId"], SESSION);
    assert!(
        state.data["server"]["url"]
            .as_str()
            .unwrap()
            .starts_with("http://127.0.0.1:")
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn resumes_the_opencode_session_on_a_new_server() {
    let fixture = fixture().await;
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            answer(fake, "again");
        }
        None
    });
    let mut options_ = fixture.start_options();
    options_.resume = Some(HarnessResumeState {
        harness_id: OPENCODE_HARNESS_ID.into(),
        data: json!({ "openCodeSessionId": SESSION }),
    });
    let session = fixture.start(options_).await;
    assert!(session.is_resume());
    let (_, messages) = run(session.as_ref(), options(HarnessTurnSettings::default())).await;
    assert_eq!(texts(&messages), ["again"]);
    assert_eq!(fixture.fake.find("GET", "/session/ses_1").len(), 1);
    assert!(fixture.fake.find("POST", "/session").is_empty());
    let state = session.stop().await.unwrap();
    assert_eq!(state.data, json!({ "openCodeSessionId": SESSION }));
}

#[tokio::test]
async fn restarts_the_server_when_the_configuration_changes() {
    let fixture = fixture().await;
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            answer(fake, "ok");
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    run(session.as_ref(), options(HarnessTurnSettings::default())).await;
    let mut settings = HarnessTurnSettings::default();
    settings.tools = vec![
        AgentToolBuilder::new("weather")
            .description("Weather")
            .parameters(json!({ "type": "object", "properties": {} }))
            .execute(|_| async { Ok(AgentToolResult::text("sunny")) })
            .build()
            .unwrap(),
    ];
    run(session.as_ref(), options(settings)).await;
    assert_eq!(fixture.fake_file("spawns").lines().count(), 2);
    let config = fixture.config();
    assert_eq!(config["mcp"]["harness-tools"]["type"], "local");
    assert_eq!(config["mcp"]["harness-tools"]["command"][0], "sh");
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn runs_host_tools_through_the_relay_after_their_tool_call_event() {
    let fixture = fixture().await;
    let relay_dir = fixture.relay_dir();
    let answers: Arc<StdMutex<Vec<Value>>> = Arc::default();
    let recorded = answers.clone();
    fixture.fake.on(move |request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            let fake = fake.clone();
            let dir = relay_dir.clone();
            let recorded = recorded.clone();
            tokio::spawn(async move {
                fake.emit(status("busy"));
                fake.emit(assistant_updated("msg_1"));
                // The MCP request arrives before OpenCode reports the call.
                let call = tokio::spawn(mcp_call(dir, "weather", json!({ "city": "Berlin" })));
                tokio::time::sleep(Duration::from_millis(100)).await;
                fake.emit(ev(
                    "message.part.updated",
                    json!({ "part": {
                        "type": "tool", "id": "prt_1", "sessionID": SESSION, "messageID": "msg_1",
                        "callID": "call_1", "tool": "harness-tools_weather",
                        "state": { "status": "running", "input": { "city": "Berlin" } },
                    } }),
                ));
                let answer = call.await.unwrap();
                recorded.lock().unwrap().push(answer);
                fake.emit(step_ended(1, 1));
                fake.emit(status("idle"));
            });
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    let mut settings = HarnessTurnSettings::default();
    settings.tools = vec![
        AgentToolBuilder::new("weather")
            .description("Weather")
            .parameters(json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
            }))
            .execute(|args| async move {
                Ok(AgentToolResult::text(format!(
                    "sunny in {}",
                    args["city"].as_str().unwrap()
                )))
            })
            .build()
            .unwrap(),
    ];
    let (_, messages) = run(session.as_ref(), options(settings)).await;
    assert_eq!(
        tool_calls(&messages),
        [(
            "call_1".to_string(),
            "weather".to_string(),
            json!({ "city": "Berlin" })
        )]
    );
    assert_eq!(
        tool_results(&messages),
        [(
            "call_1".to_string(),
            "weather".to_string(),
            false,
            "sunny in Berlin".to_string()
        )]
    );
    let answer = &answers.lock().unwrap()[0];
    assert_eq!(answer["result"]["content"][0]["text"], "sunny in Berlin");
    assert_eq!(answer["result"]["isError"], false);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn refuses_relay_calls_without_a_tool_call_event() {
    let fixture = fixture().await;
    let relay_dir = fixture.relay_dir();
    let answers: Arc<StdMutex<Vec<Value>>> = Arc::default();
    let recorded = answers.clone();
    fixture.fake.on(move |request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            let fake = fake.clone();
            let dir = relay_dir.clone();
            let recorded = recorded.clone();
            tokio::spawn(async move {
                fake.emit(status("busy"));
                // Nothing authorizes it: refused when the turn ends.
                let call = tokio::spawn(mcp_call(dir, "weather", json!({})));
                tokio::time::sleep(Duration::from_millis(100)).await;
                fake.emit(assistant_updated("msg_1"));
                for event in text("t", "done") {
                    fake.emit(event);
                }
                fake.emit(step_ended(1, 1));
                fake.emit(status("idle"));
                let answer = call.await.unwrap();
                recorded.lock().unwrap().push(answer);
            });
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    let mut settings = HarnessTurnSettings::default();
    settings.tools = vec![
        AgentToolBuilder::new("weather")
            .description("Weather")
            .parameters(json!({ "type": "object", "properties": {} }))
            .execute(|_| async { panic!("must not run") })
            .build()
            .unwrap(),
    ];
    let (_, messages) = run(session.as_ref(), options(settings)).await;
    assert!(tool_results(&messages).is_empty());
    for _ in 0..50 {
        if !answers.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let answer = answers.lock().unwrap()[0].clone();
    assert_eq!(answer["result"]["isError"], true);
    assert_eq!(
        answer["result"]["content"][0]["text"],
        "unauthorized tool relay request"
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn decides_permission_requests() {
    let fixture = fixture().await;
    std::fs::write(format!("{}/work/notes.txt", fixture.root), "notes").unwrap();
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            fake.emit(status("busy"));
            fake.emit(assistant_updated("msg_1"));
            fake.emit(ev(
                "message.part.updated",
                json!({ "part": {
                    "type": "tool", "id": "prt_1", "sessionID": SESSION, "messageID": "msg_1",
                    "callID": "call_bash", "tool": "bash",
                    "state": { "status": "running", "input": { "command": "rm -rf /" } },
                } }),
            ));
            fake.emit(ev(
                "permission.asked",
                json!({
                    "id": "per_bash", "sessionID": SESSION, "permission": "bash",
                    "patterns": ["rm -rf /"], "tool": { "callID": "call_bash", "messageID": "msg_1" },
                }),
            ));
            fake.emit(ev(
                "permission.asked",
                json!({
                    "id": "per_glob", "sessionID": SESSION, "permission": "glob",
                    "patterns": ["*.txt"], "tool": { "callID": "call_glob", "messageID": "msg_1" },
                }),
            ));
        }
        if is(request, "POST", "/permission/per_bash/reply") {
            fake.emit(ev(
                "message.part.updated",
                json!({ "part": {
                    "type": "tool", "id": "prt_1", "sessionID": SESSION, "messageID": "msg_1",
                    "callID": "call_bash", "tool": "bash",
                    "state": { "status": "error", "input": { "command": "rm -rf /" }, "error": "rejected" },
                } }),
            ));
            fake.emit(step_ended(1, 1));
            fake.emit(status("idle"));
        }
        if is(request, "GET", "/session/ses_1/message/msg_1") {
            return Some(json!({ "parts": [{
                "type": "tool", "callID": "call_glob", "tool": "glob",
                "state": { "status": "running", "input": { "pattern": "*.txt" } },
            }] }));
        }
        None
    });
    let mut start = fixture.start_options();
    start.permission_mode = PermissionMode::AllowReads;
    let session = fixture.start(start).await;
    let asked: Arc<StdMutex<Vec<String>>> = Arc::default();
    let seen = asked.clone();
    let mut turn_options = options(HarnessTurnSettings::default());
    turn_options.before_tool_call = Some(Arc::new(move |context, _| {
        seen.lock().unwrap().push(context.tool_call.name.clone());
        Box::pin(async {
            Ok(Some(BeforeToolCallResult {
                block: true,
                reason: Some("not allowed".into()),
            }))
        })
    }));
    let (_, messages) = run(session.as_ref(), turn_options).await;
    assert_eq!(*asked.lock().unwrap(), ["bash"]);
    assert_eq!(
        fixture.fake.find("POST", "/permission/per_bash/reply")[0].body,
        json!({ "reply": "reject", "message": "not allowed" })
    );
    for _ in 0..50 {
        if !fixture
            .fake
            .find("POST", "/permission/per_glob/reply")
            .is_empty()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        fixture.fake.find("POST", "/permission/per_glob/reply")[0].body,
        json!({ "reply": "once" })
    );
    assert_eq!(
        tool_calls(&messages)[0],
        (
            "call_bash".to_string(),
            "bash".to_string(),
            json!({ "command": "rm -rf /" })
        )
    );
    let results = tool_results(&messages);
    assert_eq!(results[0].0, "call_bash");
    assert!(results[0].2);

    // Restrictive modes deny OpenCode's native read and serve `read` from
    // the relay.
    let config = fixture.config();
    assert_eq!(config["permission"]["read"], "deny");
    assert_eq!(config["permission"]["glob"], "ask");
    assert!(config["mcp"]["harness-tools"].is_object());
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn serves_read_from_the_relay_in_restrictive_modes() {
    let fixture = fixture().await;
    std::fs::write(format!("{}/work/notes.txt", fixture.root), "one\ntwo\n").unwrap();
    std::fs::write(format!("{}/home/secret.txt", fixture.root), "secret").unwrap();
    let relay_dir = fixture.relay_dir();
    let answers: Arc<StdMutex<Vec<Value>>> = Arc::default();
    let recorded = answers.clone();
    let secret = format!("{}/home/secret.txt", fixture.root);
    fixture.fake.on(move |request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            let fake = fake.clone();
            let dir = relay_dir.clone();
            let recorded = recorded.clone();
            let secret = secret.clone();
            tokio::spawn(async move {
                fake.emit(status("busy"));
                fake.emit(assistant_updated("msg_1"));
                for (call_id, path) in [("call_1", "notes.txt".to_string()), ("call_2", secret)] {
                    let input = json!({ "filePath": path });
                    fake.emit(ev(
                        "message.part.updated",
                        json!({ "part": {
                            "type": "tool", "id": call_id, "sessionID": SESSION, "messageID": "msg_1",
                            "callID": call_id, "tool": "harness-tools_read",
                            "state": { "status": "running", "input": input },
                        } }),
                    ));
                    let answer = mcp_call(dir.clone(), "read", input).await;
                    recorded.lock().unwrap().push(answer);
                }
                fake.emit(step_ended(1, 1));
                fake.emit(status("idle"));
            });
        }
        None
    });
    let mut start = fixture.start_options();
    start.permission_mode = PermissionMode::AllowReads;
    let session = fixture.start(start).await;
    let (_, messages) = run(session.as_ref(), options(HarnessTurnSettings::default())).await;
    let answers = answers.lock().unwrap().clone();
    assert_eq!(
        answers[0]["result"]["content"][0]["text"],
        format!(
            "<path>{}/work/notes.txt</path>\n<type>file</type>\n<content>\n1: one\n2: two\n</content>",
            fixture.root
        )
    );
    // Outside the workspace: asked, and without a handler, denied.
    assert_eq!(answers[1]["result"]["isError"], true);
    let results = tool_results(&messages);
    assert_eq!(results[0].1, "read");
    assert!(!results[0].2);
    assert!(results[1].2);
    assert!(results[1].3.contains("requires approval"));
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn answers_questions_with_the_questions_tool() {
    let fixture = fixture().await;
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            fake.emit(status("busy"));
            fake.emit(ev(
                "question.asked",
                json!({
                    "id": "que_1", "sessionID": SESSION, "tool": { "callID": "call_q" },
                    "questions": [{
                        "question": "Pick a color", "header": "Color",
                        "options": [{ "label": "Red", "description": "Warm" }, { "label": "Blue", "description": "Cool" }],
                    }],
                }),
            ));
        }
        if is(request, "POST", "/question/que_1/reply") {
            answer(fake, "Blue it is");
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    let mut settings = HarnessTurnSettings::default();
    settings.tools = vec![
        AgentToolBuilder::new("askUserQuestions")
            .description("Ask the user")
            .parameters(crate::harness::questions_tool::questions_tool_input_schema())
            .execute(|args| async move {
                assert_eq!(args["questions"][0]["question"], "Pick a color");
                let mut result = AgentToolResult::text("answered");
                result.details = Some(json!({
                    "action": "answered",
                    "answers": { "question-1": { "optionIds": ["option-2"] } },
                }));
                Ok(result)
            })
            .build()
            .unwrap(),
    ];
    let (_, messages) = run(session.as_ref(), options(settings)).await;
    assert_eq!(
        fixture.fake.find("POST", "/question/que_1/reply")[0].body,
        json!({ "answers": [["Blue"]] })
    );
    assert_eq!(tool_calls(&messages)[0].1, "askUserQuestions");
    assert_eq!(texts(&messages), ["Blue it is"]);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn rejects_questions_without_the_questions_tool() {
    let fixture = fixture().await;
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            fake.emit(status("busy"));
            fake.emit(ev(
                "question.asked",
                json!({ "id": "que_1", "sessionID": SESSION, "questions": [{ "question": "?", "options": [] }] }),
            ));
        }
        if is(request, "POST", "/question/que_1/reject") {
            answer(fake, "fine");
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    let (events, _) = run(session.as_ref(), options(HarnessTurnSettings::default())).await;
    assert_eq!(fixture.fake.find("POST", "/question/que_1/reject").len(), 1);
    assert!(events.iter().any(|event| matches!(
        event,
        HarnessEvent::Notice { level: NoticeLevel::Warning, message } if message.contains("askUserQuestions")
    )));
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn returns_structured_output() {
    let fixture = fixture().await;
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            fake.emit(status("busy"));
            fake.emit(ev(
                "message.updated",
                json!({ "info": {
                    "id": "msg_1", "sessionID": SESSION, "role": "assistant",
                    "providerID": "anthropic", "modelID": "claude",
                    "structured": { "answer": 42 },
                } }),
            ));
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    let mut turn_options = options(HarnessTurnSettings::default());
    let schema = json!({ "type": "object", "properties": { "answer": { "type": "number" } } });
    turn_options.response_format = Some(ResponseFormat::Json {
        schema: Some(schema.clone()),
        name: None,
        description: None,
    });
    let (_, messages) = run(session.as_ref(), turn_options).await;
    assert_eq!(texts(&messages), ["{\"answer\":42}"]);
    assert_eq!(
        fixture.fake.find("POST", "/session/ses_1/prompt_async")[0].body["format"],
        json!({ "type": "json_schema", "schema": schema })
    );

    let mut missing = options(HarnessTurnSettings::default());
    missing.response_format = Some(ResponseFormat::Json {
        schema: None,
        name: None,
        description: None,
    });
    let error = session
        .prompt(Message::user_text("hi"), missing)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("requires a JSON schema"));
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn compacts_between_turns() {
    let fixture = fixture().await;
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            answer(fake, "ok");
        }
        if is(request, "POST", "/session/ses_1/summarize") {
            fake.emit(ev(
                "message.updated",
                json!({ "info": { "id": "msg_c", "sessionID": SESSION, "role": "assistant", "summary": true } }),
            ));
            fake.emit(ev(
                "message.part.updated",
                json!({ "part": { "type": "text", "id": "prt_c", "sessionID": SESSION, "messageID": "msg_c", "text": "summary text" } }),
            ));
            fake.emit(ev("session.compacted", json!({ "sessionID": SESSION })));
        }
        if is(request, "GET", "/session/ses_1/message") {
            return Some(json!([{ "info": {
                "id": "msg_1", "role": "assistant", "providerID": "anthropic", "modelID": "claude",
            }, "parts": [] }]));
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    assert!(
        session
            .compact(Some("focus".into()))
            .await
            .unwrap_err()
            .to_string()
            .contains("custom compaction instructions")
    );
    run(session.as_ref(), options(HarnessTurnSettings::default())).await;
    session.compact(None).await.unwrap();
    assert_eq!(
        fixture.fake.find("POST", "/session/ses_1/summarize")[0].body,
        json!({ "providerID": "anthropic", "modelID": "claude", "auto": false })
    );
    let (events, _) = run(session.as_ref(), options(HarnessTurnSettings::default())).await;
    let compactions: Vec<&HarnessEvent> = events
        .iter()
        .filter(|event| matches!(event, HarnessEvent::Compaction { .. }))
        .collect();
    assert_eq!(compactions.len(), 1);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn reports_session_errors_and_failed_prompts() {
    let fixture = fixture().await;
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            if request.body["parts"][0]["text"] == "fail" {
                return Some(json!({ "$status": 400, "body": { "message": "bad prompt" } }));
            }
            fake.emit(status("busy"));
            fake.emit(ev(
                "session.error",
                json!({ "sessionID": SESSION, "error": { "name": "APIError", "data": { "message": "quota" } } }),
            ));
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    let (_, messages) = run(session.as_ref(), options(HarnessTurnSettings::default())).await;
    let assistant = last_assistant(&messages);
    assert_eq!(assistant.stop_reason, StopReason::Error);
    assert!(
        assistant
            .error_message
            .as_deref()
            .unwrap()
            .contains("quota")
    );

    let turn = session
        .prompt(
            Message::user_text("fail"),
            options(HarnessTurnSettings::default()),
        )
        .await
        .unwrap();
    let (_, messages) = turn.collect().await.unwrap();
    let message = last_assistant(&messages).error_message.clone().unwrap();
    assert!(message.starts_with("OpenCode prompt failed:"), "{message}");
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn aborts_a_running_turn() {
    let fixture = fixture().await;
    let token = CancellationToken::new();
    let cancel = token.clone();
    fixture.fake.on(move |request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            fake.emit(status("busy"));
            let cancel = cancel.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                cancel.cancel();
            });
        }
        None
    });
    let session = fixture.start(fixture.start_options()).await;
    let mut turn_options = options(HarnessTurnSettings::default());
    turn_options.cancellation_token = Some(token);
    let turn = session
        .prompt(Message::user_text("hi"), turn_options)
        .await
        .unwrap();
    let (_, messages) = tokio::time::timeout(Duration::from_secs(10), turn.collect())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(last_assistant(&messages).stop_reason, StopReason::Aborted);
    assert_eq!(fixture.fake.find("POST", "/session/ses_1/abort").len(), 1);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn validates_settings_and_start_options() {
    let fixture = fixture().await;
    let mut servers = Map::new();
    servers.insert("harness-tools".into(), json!({}));
    let settings = OpenCodeHarnessSettings {
        mcp_servers: Some(servers),
        ..Default::default()
    };
    assert!(
        OpenCodeHarness::new(settings)
            .err()
            .unwrap()
            .to_string()
            .contains("reserved")
    );

    let harness = OpenCodeHarness::new(fixture.settings()).unwrap();
    assert!(harness.supports_builtin_tool_approvals());
    assert!(harness.supports_builtin_tool_filtering());
    assert_eq!(
        harness.builtin_tools().keys().cloned().collect::<Vec<_>>(),
        [
            "askUserQuestions",
            "read",
            "write",
            "edit",
            "bash",
            "glob",
            "grep",
            "ls",
            "webfetch",
            "skill",
            "todowrite",
            "agent",
        ]
    );
    let mut start = fixture.start_options();
    start.builtin_tool_filter = Some(BuiltinToolFilter::Deny {
        tool_names: vec!["write".into()],
    });
    let error = harness.start(start).await.err().unwrap();
    assert!(error.to_string().contains("'write' and 'edit'"));

    // Inactive builtins are denied in OpenCode's permissions.
    fixture.fake.on(|request, fake| {
        if is(request, "POST", "/session/ses_1/prompt_async") {
            answer(fake, "ok");
        }
        None
    });
    let mut start = fixture.start_options();
    start.builtin_tool_filter = Some(BuiltinToolFilter::Deny {
        tool_names: vec!["write".into(), "edit".into(), "bash".into()],
    });
    let session = harness.start(start).await.unwrap();
    run(session.as_ref(), options(HarnessTurnSettings::default())).await;
    let config = fixture.config();
    assert_eq!(config["permission"]["edit"], "deny");
    assert_eq!(config["permission"]["bash"], "deny");
    session.destroy().await.unwrap();

    // The working directory must not contain the home directory.
    let start = HarnessStartOptions::new(
        "session-2",
        HarnessSandbox::Basic(Arc::new(
            LocalSandbox::new(fixture.root.clone()).with_home(format!("{}/home", fixture.root)),
        )),
        fixture.root.clone(),
    );
    let session = harness.start(start).await.unwrap();
    let turn = session
        .prompt(
            Message::user_text("hi"),
            options(HarnessTurnSettings::default()),
        )
        .await
        .unwrap();
    let (_, messages) = turn.collect().await.unwrap();
    assert!(
        last_assistant(&messages)
            .error_message
            .as_deref()
            .unwrap()
            .contains("must not be or contain the sandbox home directory")
    );
}

#[tokio::test]
async fn reports_a_server_that_exits_before_listening() {
    let fixture = fixture().await;
    let mut settings = fixture.settings();
    let script = format!("{}/.fake/broken", fixture.root);
    write_script(&script, "#!/bin/sh\necho broken >&2\nexit 3\n");
    settings.executable_override = Some(shell_quote(&script));
    let session = fixture.start_with(settings, fixture.start_options()).await;
    let turn = session
        .prompt(
            Message::user_text("hi"),
            options(HarnessTurnSettings::default()),
        )
        .await
        .unwrap();
    let (_, messages) = turn.collect().await.unwrap();
    let message = last_assistant(&messages).error_message.clone().unwrap();
    assert!(message.contains("exited with code 3"), "{message}");
    assert!(message.contains("broken"), "{message}");
}

#[test]
fn builds_the_bootstrap_recipe() {
    let bootstrap = opencode_bootstrap();
    assert_eq!(bootstrap.harness_id, "opencode");
    assert_eq!(bootstrap.bootstrap_dir, ".harness-bootstrap/opencode");
    let manifest: Value = serde_json::from_str(&bootstrap.files[0].content).unwrap();
    assert_eq!(manifest["dependencies"]["opencode-ai"], OPENCODE_VERSION);
    assert_eq!(manifest["dependencies"]["@vscode/ripgrep"], "1.18.0");
    assert_eq!(
        bootstrap
            .commands
            .iter()
            .map(|command| command.command.as_str())
            .collect::<Vec<_>>(),
        [
            "npm install --no-audit --no-fund --loglevel=error",
            "./node_modules/.bin/opencode --version",
        ]
    );
}
