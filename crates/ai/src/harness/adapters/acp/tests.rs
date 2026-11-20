//! ACP tests against a fake agent: a shell script that answers scripted
//! JSON-RPC requests, prints scripted notifications and requests, and calls
//! host tools through the relay's FIFOs.

use std::sync::{Arc, Mutex as StdMutex};

use serde_json::{Value, json};

use super::*;
use crate::harness::adapters::local_sandbox::{LocalSandbox, canonical};
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::session::{
    FileChangeKind, HarnessEvent, HarnessSandbox, HarnessTurnSettings, NoticeLevel,
};
use crate::harness::tools::{BuiltinToolKind, PermissionMode};
use crate::{
    AgentToolBuilder, AgentToolResult, AssistantContent, BeforeToolCallResult, Message, StopReason,
    Tool, ToolResultContent,
};

/// Steps: `emit <json>` prints a line; `await <pattern>` reads stdin until
/// a line contains the pattern; `reply <method> <json>` answers the oldest
/// unanswered request for that method, reading stdin until one arrives;
/// `mcp <json>` sends a host tool MCP request and records the answer;
/// `exit <code>` exits.
const FAKE_AGENT: &str = r#"
echo spawn >> "$FAKE_DIR/spawns"
printf '%s\n' "$@" >> "$FAKE_DIR/args"
env > "$FAKE_DIR/env"
: > "$FAKE_DIR/seen"
: > "$FAKE_DIR/replied"
steps="$FAKE_DIR/steps"
if [ "$(wc -l < "$FAKE_DIR/spawns")" -gt 1 ] && [ -f "$FAKE_DIR/steps.2" ]; then
  steps="$FAKE_DIR/steps.2"
fi
record() {
  printf '%s\n' "$1" >> "$FAKE_DIR/stdin"
  printf '%s\n' "$1" >> "$FAKE_DIR/seen"
}
request_id() {
  printf '%s' "$1" | sed -n 's/^[^}]*"id":\([0-9]*\).*/\1/p'
}
unanswered() {
  while IFS= read -r seen; do
    case "$seen" in *"\"method\":\"$1\""*)
      grep -qx "$(request_id "$seen")" "$FAKE_DIR/replied" || { printf '%s' "$seen"; return; } ;;
    esac
  done < "$FAKE_DIR/seen"
}
exec 3< "$steps"
while IFS= read -r step <&3; do
  case "$step" in
    "emit "*) printf '%s\n' "${step#emit }" ;;
    "await "*)
      pat="${step#await }"
      while IFS= read -r line; do
        record "$line"
        case "$line" in *"$pat"*) break ;; esac
      done ;;
    "reply "*)
      rest="${step#reply }"
      method="${rest%% *}"
      result="${rest#* }"
      line=$(unanswered "$method")
      [ -n "$line" ] || while IFS= read -r line; do
        record "$line"
        case "$line" in *"\"method\":\"$method\""*) break ;; esac
      done
      id=$(request_id "$line")
      echo "$id" >> "$FAKE_DIR/replied"
      printf '{"jsonrpc":"2.0","id":%s,"result":%s}\n' "$id" "$result" ;;
    "http "*)
      url=$(sed -n 's/.*"url":"\([^"]*\)".*/\1/p' "$FAKE_DIR/seen" | head -n 1)
      auth=$(sed -n 's/.*"value":"\(Bearer [^"]*\)".*/\1/p' "$FAKE_DIR/seen" | head -n 1)
      curl -s -X POST -H "Authorization: $auth" -H 'Content-Type: application/json' \
        -d "${step#http }" "$url" >> "$FAKE_DIR/mcp"
      echo >> "$FAKE_DIR/mcp"
      curl -s -o /dev/null -w '%{http_code}\n' -X POST -H 'Authorization: Bearer wrong' \
        -d "${step#http }" "$url" >> "$FAKE_DIR/http_status" ;;
    "mcp "*)
      printf '%s\n' "${step#mcp }" > "$MCP_DIR/req"
      IFS= read -r answer < "$MCP_DIR/res"
      printf '%s\n' "$answer" >> "$FAKE_DIR/mcp" ;;
    "exit "*) echo "agent failed" >&2; exit "${step#exit }" ;;
  esac
done
while IFS= read -r line; do record "$line"; done
"#;

enum Step {
    Emit(Value),
    Await(&'static str),
    Reply(&'static str, Value),
    Mcp(Value),
    Http(Value),
    Exit(i32),
}

struct Fixture {
    _temp: TempDir,
    root: String,
    fake_dir: String,
}

fn fixture() -> Fixture {
    let temp = mkdtemp("acp-");
    let root = canonical(temp.path());
    let fake_dir = format!("{root}/.fake");
    std::fs::create_dir_all(&fake_dir).unwrap();
    std::fs::write(format!("{fake_dir}/agent.sh"), FAKE_AGENT).unwrap();
    std::fs::create_dir_all(format!("{root}/home")).unwrap();
    std::fs::create_dir_all(format!("{root}/work")).unwrap();
    Fixture {
        _temp: temp,
        root,
        fake_dir,
    }
}

fn steps_text(steps: Vec<Step>) -> String {
    steps
        .into_iter()
        .map(|step| match step {
            Step::Emit(value) => format!("emit {value}"),
            Step::Await(pattern) => format!("await {pattern}"),
            Step::Reply(method, result) => format!("reply {method} {result}"),
            Step::Mcp(value) => format!("mcp {value}"),
            Step::Http(value) => format!("http {value}"),
            Step::Exit(code) => format!("exit {code}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// The fake's tool resolver: a title `ai-sdk-harness-tools-<name>`, like
/// GitHub Copilot's.
fn resolve_by_title() -> AcpResolveHostToolCall {
    Arc::new(|options: AcpResolveHostToolCallOptions<'_>| {
        let title = options.tool_call.get("title")?.as_str()?;
        let name = options
            .tool_names
            .iter()
            .find(|name| title == format!("{}-{name}", options.server_name))?;
        Some(AcpResolvedHostToolCall {
            tool_name: name.clone(),
            input: options.tool_call.get("rawInput")?.clone(),
        })
    })
}

fn builtin(name: &str, kind: BuiltinToolKind, schema: Value) -> (String, BuiltinTool) {
    (
        name.to_string(),
        BuiltinTool {
            tool: Tool {
                name: name.to_string(),
                description: String::new(),
                parameters: schema,
                constrained_sampling: None,
            },
            native_name: Some(format!("{name}ToolCall")),
            common_name: None,
            kind: Some(kind),
        },
    )
}

impl Fixture {
    fn script(&self, steps: Vec<Step>) {
        std::fs::write(format!("{}/steps", self.fake_dir), steps_text(steps)).unwrap();
    }

    fn script_respawn(&self, steps: Vec<Step>) {
        std::fs::write(format!("{}/steps.2", self.fake_dir), steps_text(steps)).unwrap();
    }

    fn mcp_dir(&self) -> String {
        format!(
            "{}/home/.ai-sdk-harness/.agent-runs/session-1/host-tools",
            self.root
        )
    }

    fn settings(&self) -> AcpHarnessSettings {
        let mut settings = AcpHarnessSettings::new(
            "fake-agent",
            AcpClientApp {
                name: "ai-sdk-harness-fake".into(),
                version: "1.0.0".into(),
            },
            AcpSource::InstallCommand {
                command: "true".into(),
            },
            "agent",
            AcpModelMapping::SessionConfigOption {
                path: "model".into(),
            },
            resolve_by_title(),
        );
        settings.executable_override = Some(format!(
            "env FAKE_DIR={} MCP_DIR={} sh {}/agent.sh",
            shell_quote(&self.fake_dir),
            shell_quote(&self.mcp_dir()),
            shell_quote(&self.fake_dir)
        ));
        settings.args = vec!["acp".into()];
        settings.process_env = Some(IndexMap::new());
        settings.builtin_tools = IndexMap::from([
            builtin(
                "read",
                BuiltinToolKind::Readonly,
                json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }),
            ),
            builtin(
                "bash",
                BuiltinToolKind::Bash,
                json!({ "type": "object", "properties": { "command": { "type": "string" } }, "required": ["command"] }),
            ),
        ]);
        settings.builtin_tool_titles = IndexMap::from([
            ("read".to_string(), "Read".to_string()),
            ("bash".to_string(), "Terminal".to_string()),
        ]);
        settings
    }

    fn harness(&self, configure: impl FnOnce(&mut AcpHarnessSettings)) -> AcpHarness {
        let mut settings = self.settings();
        configure(&mut settings);
        AcpHarness::new(settings).unwrap()
    }

    fn start_options(&self, permission_mode: PermissionMode) -> HarnessStartOptions {
        let mut options = HarnessStartOptions::new(
            "session-1",
            HarnessSandbox::Basic(Arc::new(
                LocalSandbox::new(format!("{}/work", self.root))
                    .with_home(format!("{}/home", self.root)),
            )),
            format!("{}/work", self.root),
        );
        options.permission_mode = permission_mode;
        options
    }

    fn spawns(&self) -> usize {
        std::fs::read_to_string(format!("{}/spawns", self.fake_dir))
            .unwrap_or_default()
            .lines()
            .count()
    }

    fn stdin(&self) -> Vec<Value> {
        std::fs::read_to_string(format!("{}/stdin", self.fake_dir))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }

    fn requests(&self, method: &str) -> Vec<Value> {
        self.stdin()
            .into_iter()
            .filter(|message| message["method"] == method)
            .map(|message| message["params"].clone())
            .collect()
    }

    /// The client's answer to the agent request with `id`.
    fn answer(&self, id: u64) -> Value {
        self.stdin()
            .into_iter()
            .find(|message| message["id"] == id && message.get("method").is_none())
            .unwrap_or_else(|| panic!("no answer to request {id}"))
    }

    fn mcp_answers(&self) -> Vec<Value> {
        std::fs::read_to_string(format!("{}/mcp", self.fake_dir))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }
}

fn initialize(capabilities: Value) -> Step {
    Step::Reply(
        "initialize",
        json!({ "protocolVersion": 1, "agentCapabilities": capabilities, "authMethods": [] }),
    )
}

fn new_session(id: &str) -> Step {
    Step::Reply("session/new", json!({ "sessionId": id }))
}

fn start(id: &str) -> Vec<Step> {
    vec![initialize(json!({})), new_session(id)]
}

fn update(session: &str, update: Value) -> Step {
    Step::Emit(json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": { "sessionId": session, "update": update },
    }))
}

fn chunk(kind: &str, message_id: &str, text: &str) -> Value {
    json!({ "sessionUpdate": kind, "messageId": message_id, "content": { "type": "text", "text": text } })
}

fn end_turn() -> Step {
    Step::Reply("session/prompt", json!({ "stopReason": "end_turn" }))
}

fn permission_request(id: u64, tool_call: Value) -> Step {
    Step::Emit(json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "session/request_permission",
        "params": {
            "sessionId": "s1",
            "toolCall": tool_call,
            "options": [
                { "optionId": "allow", "name": "Allow", "kind": "allow_once" },
                { "optionId": "always", "name": "Always", "kind": "allow_always" },
                { "optionId": "reject", "name": "Reject", "kind": "reject_once" },
            ],
        },
    }))
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
                    AssistantContent::Thinking(thinking) => {
                        Some(format!("thinking: {}", thinking.thinking))
                    }
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

async fn run_with(
    session: &dyn HarnessSession,
    prompt: Message,
    options: HarnessTurnOptions,
) -> (Vec<HarnessEvent>, Vec<Message>) {
    let turn = session.prompt(prompt, options).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(8), turn.collect())
        .await
        .unwrap_or_else(|_| panic!("turn timed out"))
        .unwrap()
}

async fn run(
    session: &dyn HarnessSession,
    options: HarnessTurnOptions,
) -> (Vec<HarnessEvent>, Vec<Message>) {
    run_with(session, Message::user_text("hi"), options).await
}

#[tokio::test]
async fn streams_a_prompt_turn() {
    let fixture = fixture();
    let mut steps = start("s1");
    steps.extend([
        Step::Await("session/prompt"),
        update("s1", chunk("agent_thought_chunk", "m1", "Planning.")),
        update("s1", chunk("agent_message_chunk", "m2", "Hello")),
        update("s1", chunk("agent_message_chunk", "m2", " world")),
        // Another session's updates are ignored.
        update("other", chunk("agent_message_chunk", "m9", "stale")),
        update(
            "s1",
            json!({ "sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Read a.txt",
                "kind": "read", "status": "pending", "rawInput": { "path": "a.txt" } }),
        ),
        update(
            "s1",
            json!({ "sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed",
                "content": [{ "type": "content", "content": { "type": "text", "text": "file body" } }] }),
        ),
        update(
            "s1",
            json!({ "sessionUpdate": "tool_call", "toolCallId": "t2", "title": "Edit b.txt",
                "kind": "edit", "status": "in_progress", "name": "patchFile",
                "content": [{ "type": "diff", "path": "b.txt", "oldText": null, "newText": "x" }] }),
        ),
        update(
            "s1",
            json!({ "sessionUpdate": "tool_call_update", "toolCallId": "t2", "status": "failed",
                "rawOutput": "denied" }),
        ),
        update("s1", chunk("agent_message_chunk", "m3", "Done.")),
        Step::Reply(
            "session/prompt",
            json!({ "stopReason": "end_turn", "usage": { "inputTokens": 10, "outputTokens": 5,
                "thoughtTokens": 2, "cachedReadTokens": 3, "totalTokens": 15 } }),
        ),
    ]);
    fixture.script(steps);
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let (events, messages) = run(session.as_ref(), HarnessTurnOptions::default()).await;

    assert_eq!(
        texts(&messages),
        ["thinking: Planning.", "Hello world", "Done."]
    );
    assert_eq!(
        tool_calls(&messages),
        [
            (
                "t1".to_string(),
                "read".to_string(),
                json!({ "path": "a.txt" })
            ),
            ("t2".to_string(), "patchFile".to_string(), json!({})),
        ]
    );
    assert_eq!(
        tool_results(&messages),
        [
            (
                "t1".to_string(),
                "read".to_string(),
                false,
                "file body".to_string()
            ),
            (
                "t2".to_string(),
                "patchFile".to_string(),
                true,
                "denied".to_string()
            ),
        ]
    );
    let changes: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            HarnessEvent::FileChange { kind, path } => Some((*kind, path.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(changes, [(FileChangeKind::Create, "b.txt".to_string())]);
    let last = last_assistant(&messages);
    assert_eq!(last.stop_reason, StopReason::Stop);
    assert_eq!(last.usage.input, 10);
    assert_eq!(last.usage.output, 5);
    assert_eq!(last.usage.reasoning, Some(2));
    assert_eq!(last.usage.cache_read, 3);
    assert_eq!(last.provider, "fake-agent");
    assert_eq!(last.api, ACP_API);

    let initialize = &fixture.requests("initialize")[0];
    assert_eq!(initialize["protocolVersion"], 1);
    assert_eq!(
        initialize["clientInfo"],
        json!({ "name": "ai-sdk-harness-fake", "version": "1.0.0" })
    );
    let new = &fixture.requests("session/new")[0];
    assert_eq!(new["cwd"], json!(format!("{}/work", fixture.root)));
    let server = &new["mcpServers"][0];
    assert_eq!(server["name"], HOST_TOOL_MCP_SERVER_NAME);
    assert_eq!(server["command"], "/bin/sh");
    assert_eq!(server["args"][3], json!(fixture.mcp_dir()));
    let prompt = &fixture.requests("session/prompt")[0];
    assert_eq!(prompt["sessionId"], "s1");
    assert_eq!(prompt["prompt"], json!([{ "type": "text", "text": "hi" }]));
    let args = std::fs::read_to_string(format!("{}/args", fixture.fake_dir)).unwrap();
    assert_eq!(args.trim(), "acp");

    let state: AcpResumeState =
        serde_json::from_value(session.detach().await.unwrap().data).unwrap();
    assert_eq!(state.acp_session_id.as_deref(), Some("s1"));
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn runs_host_tools_through_the_mcp_relay() {
    let fixture = fixture();
    let mut steps = start("s1");
    steps.extend([
        Step::Await("session/prompt"),
        Step::Mcp(json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": { "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "agent", "version": "1" } } })),
        Step::Mcp(json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" })),
        update(
            "s1",
            json!({ "sessionUpdate": "tool_call", "toolCallId": "call_lookup",
                "title": "ai-sdk-harness-tools-lookup", "kind": "other", "status": "in_progress",
                "rawInput": { "query": "rust" } }),
        ),
        Step::Mcp(json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": { "name": "lookup", "arguments": { "query": "rust" } } })),
        update(
            "s1",
            json!({ "sessionUpdate": "tool_call_update", "toolCallId": "call_lookup",
                "status": "completed", "rawOutput": { "ignored": true } }),
        ),
        update("s1", chunk("agent_message_chunk", "m2", "Found it.")),
        end_turn(),
    ]);
    fixture.script(steps);
    let lookup = AgentToolBuilder::new("lookup")
        .description("Look something up")
        .parameters(json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"],
        }))
        .execute(|args| async move {
            Ok(AgentToolResult::text(format!(
                "found: {}",
                args["query"].as_str().unwrap_or_default()
            )))
        })
        .build()
        .unwrap();
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let (_, messages) = run(
        session.as_ref(),
        HarnessTurnOptions {
            settings: HarnessTurnSettings {
                tools: vec![lookup],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;

    // The ACP tool call is the host tool call, reported once.
    assert_eq!(
        tool_calls(&messages),
        [(
            "call_lookup".to_string(),
            "lookup".to_string(),
            json!({ "query": "rust" })
        )]
    );
    assert_eq!(
        tool_results(&messages),
        [(
            "call_lookup".to_string(),
            "lookup".to_string(),
            false,
            "found: rust".to_string()
        )]
    );
    assert_eq!(texts(&messages), ["Found it."]);
    let answers = fixture.mcp_answers();
    assert_eq!(
        answers[0]["result"]["serverInfo"]["name"],
        HOST_TOOL_MCP_SERVER_NAME
    );
    assert_eq!(answers[1]["result"]["tools"][0]["name"], "lookup");
    assert_eq!(
        answers[2]["result"],
        json!({ "content": [{ "type": "text", "text": "found: rust" }], "isError": false })
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn runs_host_tools_through_the_http_mcp_server() {
    let fixture = fixture();
    fixture.script(vec![
        initialize(json!({ "mcpCapabilities": { "http": true } })),
        new_session("s1"),
        Step::Await("session/prompt"),
        Step::Http(json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" })),
        update(
            "s1",
            json!({ "sessionUpdate": "tool_call", "toolCallId": "call_echo",
                "title": "ai-sdk-harness-tools-echo", "kind": "other", "status": "in_progress",
                "rawInput": { "text": "hi" } }),
        ),
        Step::Http(json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": { "name": "echo", "arguments": { "text": "hi" } } })),
        end_turn(),
    ]);
    let echo = AgentToolBuilder::new("echo")
        .description("Echo")
        .parameters(json!({ "type": "object", "properties": { "text": { "type": "string" } } }))
        .execute(|args| async move {
            Ok(AgentToolResult::text(
                args["text"].as_str().unwrap_or_default().to_string(),
            ))
        })
        .build()
        .unwrap();
    let harness = fixture.harness(|settings| {
        settings.host_tool_mcp_transport = AcpHostToolMcpTransport::Http;
    });
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let (_, messages) = run(
        session.as_ref(),
        HarnessTurnOptions {
            settings: HarnessTurnSettings {
                tools: vec![echo],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let server = &fixture.requests("session/new")[0]["mcpServers"][0];
    assert_eq!(server["type"], "http");
    assert!(
        server["url"]
            .as_str()
            .unwrap()
            .starts_with("http://127.0.0.1:")
    );
    let answers = fixture.mcp_answers();
    assert_eq!(answers[0]["result"]["tools"][0]["name"], "echo");
    assert_eq!(
        answers[1]["result"]["content"],
        json!([{ "type": "text", "text": "hi" }])
    );
    assert_eq!(
        tool_results(&messages),
        [(
            "call_echo".to_string(),
            "echo".to_string(),
            false,
            "hi".to_string()
        )]
    );
    let statuses = std::fs::read_to_string(format!("{}/http_status", fixture.fake_dir)).unwrap();
    assert_eq!(statuses.lines().collect::<Vec<_>>(), ["401", "401"]);
    session.destroy().await.unwrap();

    // Without HTTP MCP support the session cannot start.
    let fixture = self::fixture();
    fixture.script(start("s1"));
    let harness = fixture.harness(|settings| {
        settings.host_tool_mcp_transport = AcpHostToolMcpTransport::Http;
    });
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let mut turn = session
        .prompt(Message::user_text("hi"), HarnessTurnOptions::default())
        .await
        .unwrap();
    let error = turn.result().await.unwrap_err().to_string();
    assert!(
        error.contains("does not advertise support for HTTP MCP servers"),
        "{error}"
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn decides_permission_requests() {
    let fixture = fixture();
    std::fs::create_dir_all(format!("{}/work/.agents/skills", fixture.root)).unwrap();
    let work = format!("{}/work", fixture.root);
    let mut steps = start("s1");
    steps.extend([
        Step::Await("session/prompt"),
        // A read inside the work dir is approved by the mode.
        permission_request(
            100,
            json!({ "toolCallId": "t1", "title": "Read a.txt", "kind": "read",
                "locations": [{ "path": format!("{work}/a.txt") }], "rawInput": { "path": "a.txt" } }),
        ),
        Step::Await("\"id\":100"),
        // A read outside it asks.
        permission_request(
            101,
            json!({ "toolCallId": "t2", "title": "Read passwd", "kind": "read",
                "locations": [{ "path": "/etc/passwd" }], "rawInput": { "path": "/etc/passwd" } }),
        ),
        Step::Await("\"id\":101"),
        // An edit asks under allow-reads.
        permission_request(
            102,
            json!({ "toolCallId": "t3", "title": "Terminal", "kind": "execute",
                "rawInput": { "command": "rm -rf /" } }),
        ),
        Step::Await("\"id\":102"),
        // Without allow_once and reject_once the request is cancelled.
        Step::Emit(json!({ "jsonrpc": "2.0", "id": 103, "method": "session/request_permission",
            "params": { "sessionId": "s1", "toolCall": { "toolCallId": "t4" }, "options": [] } })),
        Step::Await("\"id\":103"),
        // Unknown agent requests are refused.
        Step::Emit(json!({ "jsonrpc": "2.0", "id": 104, "method": "fs/read_text_file", "params": {} })),
        Step::Await("\"id\":104"),
        end_turn(),
    ]);
    fixture.script(steps);
    let asked: Arc<StdMutex<Vec<(String, Value)>>> = Arc::default();
    let before: crate::BeforeToolCallFn = {
        let asked = asked.clone();
        Arc::new(move |context, _| {
            asked.lock().unwrap().push((
                context.tool_call.name.clone(),
                context.tool_call.arguments.clone(),
            ));
            let allow = context.tool_call.name == "read";
            Box::pin(async move {
                Ok((!allow).then(|| BeforeToolCallResult {
                    block: true,
                    reason: Some("Not allowed".into()),
                }))
            })
        })
    };
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowReads))
        .await
        .unwrap();
    let (events, messages) = run(
        session.as_ref(),
        HarnessTurnOptions {
            before_tool_call: Some(before),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        asked.lock().unwrap().as_slice(),
        [
            ("read".to_string(), json!({ "path": "/etc/passwd" })),
            ("bash".to_string(), json!({ "command": "rm -rf /" })),
        ]
    );
    let outcome = |id| fixture.answer(id)["result"]["outcome"].clone();
    assert_eq!(
        outcome(100),
        json!({ "outcome": "selected", "optionId": "allow" })
    );
    assert_eq!(
        outcome(101),
        json!({ "outcome": "selected", "optionId": "allow" })
    );
    assert_eq!(
        outcome(102),
        json!({ "outcome": "selected", "optionId": "reject" })
    );
    assert_eq!(outcome(103), json!({ "outcome": "cancelled" }));
    assert_eq!(fixture.answer(104)["error"]["code"], -32601);
    // Asked tool calls are shown before the approval.
    let names: Vec<String> = tool_calls(&messages)
        .into_iter()
        .map(|(_, name, _)| name)
        .collect();
    assert_eq!(names, ["read", "bash"]);
    assert!(events.iter().any(|event| matches!(
        event,
        HarnessEvent::Notice { level: NoticeLevel::Warning, message } if message.contains("did not advertise allow_once and reject_once")
    )));
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn edits_to_protected_paths_ask() {
    let fixture = fixture();
    let work = format!("{}/work", fixture.root);
    let mut steps = start("s1");
    steps.extend([
        Step::Await("session/prompt"),
        permission_request(
            100,
            json!({ "toolCallId": "t1", "title": "Edit", "kind": "edit",
                "locations": [{ "path": format!("{work}/src/main.rs") }] }),
        ),
        Step::Await("\"id\":100"),
        permission_request(
            101,
            json!({ "toolCallId": "t2", "title": "Edit", "kind": "edit",
                "locations": [{ "path": format!("{work}/.agents/skills/x/SKILL.md") }] }),
        ),
        Step::Await("\"id\":101"),
        permission_request(
            102,
            json!({ "toolCallId": "t3", "title": "Edit", "kind": "edit",
                "locations": [{ "path": format!("{work}/.cursor/rules") }] }),
        ),
        Step::Await("\"id\":102"),
        end_turn(),
    ]);
    fixture.script(steps);
    let harness = fixture.harness(|settings| {
        settings.write_protected_paths = vec![".cursor".into()];
    });
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowEdits))
        .await
        .unwrap();
    // No approval handler: asked calls are denied.
    run(session.as_ref(), HarnessTurnOptions::default()).await;
    let option = |id| fixture.answer(id)["result"]["outcome"]["optionId"].clone();
    assert_eq!(option(100), "allow");
    assert_eq!(option(101), "reject");
    assert_eq!(option(102), "reject");
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn cancels_a_running_prompt() {
    let fixture = fixture();
    let mut steps = start("s1");
    steps.extend([
        Step::Await("session/prompt"),
        update("s1", chunk("agent_message_chunk", "m1", "Working")),
        Step::Await("session/cancel"),
        Step::Reply("session/prompt", json!({ "stopReason": "cancelled" })),
    ]);
    fixture.script(steps);
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let cancel = CancellationToken::new();
    let turn = session
        .prompt(
            Message::user_text("hi"),
            HarnessTurnOptions {
                cancellation_token: Some(cancel.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        cancel.cancel();
    });
    let (_, messages) = tokio::time::timeout(std::time::Duration::from_secs(8), turn.collect())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(last_assistant(&messages).stop_reason, StopReason::Aborted);
    assert_eq!(
        fixture.requests("session/cancel"),
        [json!({ "sessionId": "s1" })]
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn configures_models_and_permission_modes() {
    let fixture = fixture();
    fixture.script(vec![
        initialize(json!({})),
        Step::Reply(
            "session/new",
            json!({ "sessionId": "s1",
                "modes": { "currentModeId": "default", "availableModes": [{ "id": "default", "name": "Default" }, { "id": "plan", "name": "Plan" }] },
                "configOptions": [{ "id": "model", "category": "model", "type": "select", "currentValue": "auto",
                    "options": [{ "value": "auto", "name": "Auto" }, { "value": "fast", "name": "Fast" }] }] }),
        ),
        Step::Reply("session/set_mode", json!({})),
        Step::Reply(
            "session/set_config_option",
            json!({ "configOptions": [{ "id": "model", "category": "model", "currentValue": "fast-1" }] }),
        ),
        Step::Await("session/prompt"),
        update("s1", chunk("agent_message_chunk", "m1", "Hi")),
        end_turn(),
    ]);
    let harness = fixture.harness(|settings| {
        settings.permission_mode_mapping = Some(AcpPermissionModeMapping {
            allow_reads: Some(AcpPermissionModeTarget::SessionMode {
                mode_id: "plan".into(),
            }),
            allow_edits: None,
            allow_all: Some(AcpPermissionModeTarget::SessionMode {
                mode_id: "default".into(),
            }),
        });
    });
    // An unmapped mode fails at start.
    let error = harness
        .start(fixture.start_options(PermissionMode::AllowEdits))
        .await
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("Permission mode \"allow-edits\" is not supported"),
        "{error}"
    );
    assert!(
        error.contains("Use permissionMode: 'allow-all'."),
        "{error}"
    );

    let session = harness
        .start(fixture.start_options(PermissionMode::AllowReads))
        .await
        .unwrap();
    let (_, messages) = run(
        session.as_ref(),
        HarnessTurnOptions {
            settings: HarnessTurnSettings {
                model: Some("fast".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    assert_eq!(
        fixture.requests("session/set_mode"),
        [json!({ "sessionId": "s1", "modeId": "plan" })]
    );
    assert_eq!(
        fixture.requests("session/set_config_option"),
        [json!({ "sessionId": "s1", "configId": "model", "value": "fast" })]
    );
    assert_eq!(last_assistant(&messages).model, "fast-1");
    session.destroy().await.unwrap();
}

#[test]
fn validates_permission_mode_targets() {
    let fixture = fixture();
    let settings = fixture.settings();
    let mapping = AcpPermissionModeMapping {
        allow_reads: Some(AcpPermissionModeTarget::SessionConfigOption {
            config_id: "approvals".into(),
            value: AcpConfigValue::String("ask".into()),
        }),
        allow_edits: None,
        allow_all: Some(AcpPermissionModeTarget::SessionConfigOption {
            config_id: "yolo".into(),
            value: AcpConfigValue::Boolean(true),
        }),
    };
    let session = json!({ "configOptions": [
        { "id": "approvals", "type": "select", "options": [{ "group": "g", "options": [{ "value": "ask" }] }] },
        { "id": "yolo", "type": "boolean" },
    ] });
    let (method, params) =
        turn::permission_mode_request(&settings, &mapping, PermissionMode::AllowAll, &session)
            .unwrap();
    assert_eq!(method, "session/set_config_option");
    assert_eq!(
        params,
        json!({ "configId": "yolo", "value": true, "type": "boolean" })
    );
    let missing = json!({ "configOptions": [{ "id": "yolo", "type": "boolean" }] });
    let error =
        turn::permission_mode_request(&settings, &mapping, PermissionMode::AllowAll, &missing)
            .unwrap_err()
            .to_string();
    assert!(
        error.contains("requires session configuration \"approvals\""),
        "{error}"
    );
    let wrong = json!({ "configOptions": [
        { "id": "approvals", "type": "select", "options": [{ "value": "never" }] },
        { "id": "yolo", "type": "boolean" },
    ] });
    let error =
        turn::permission_mode_request(&settings, &mapping, PermissionMode::AllowReads, &wrong)
            .unwrap_err()
            .to_string();
    assert!(
        error.contains("but the agent advertised \"never\""),
        "{error}"
    );

    // Boolean targets advertise boolean config options.
    let mut settings = fixture.settings();
    settings.permission_mode_mapping = Some(mapping);
    assert_eq!(
        turn::initialize_request(&settings)["clientCapabilities"],
        json!({ "session": { "configOptions": { "boolean": {} } } })
    );
}

#[tokio::test]
async fn maps_instructions_and_structured_output() {
    let fixture = fixture();
    let mut steps = start("s1");
    steps.extend([end_turn(), end_turn(), end_turn()]);
    fixture.script(steps);
    let harness = fixture.harness(|settings| {
        settings.instruction_mapping = Some(AcpInstructionMapping::SessionMeta {
            path: vec!["agent".into(), "systemPrompt".into()],
        });
        settings.session_meta = Some(
            json!({ "agent": { "name": "x" } })
                .as_object()
                .unwrap()
                .clone(),
        );
        settings.output_schema_mapping = Some(AcpOutputSchemaMapping::SessionPromptMeta {
            path: vec!["output".into(), "schema".into()],
        });
    });
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let with = |instructions: &str| HarnessTurnOptions {
        settings: HarnessTurnSettings {
            instructions: Some(instructions.to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    run(session.as_ref(), with("Be brief.")).await;
    run(session.as_ref(), with("Be brief.")).await;
    let mut options = with("Be thorough.");
    options.response_format = Some(ResponseFormat::Json {
        schema: Some(json!({ "type": "object" })),
        name: None,
        description: None,
    });
    run(session.as_ref(), options).await;

    let new = &fixture.requests("session/new")[0];
    assert_eq!(
        new["_meta"],
        json!({ "agent": { "name": "x", "systemPrompt": "Be brief." } })
    );
    let prompts = fixture.requests("session/prompt");
    // The first instructions went in the session; unchanged ones are not
    // repeated; changed ones lead the next prompt.
    assert_eq!(prompts[0]["prompt"].as_array().unwrap().len(), 1);
    assert_eq!(prompts[1]["prompt"].as_array().unwrap().len(), 1);
    let guidance = prompts[2]["prompt"][0]["text"].as_str().unwrap();
    assert!(guidance.starts_with("<session-guidance>"), "{guidance}");
    assert!(guidance.contains("<instructions>\nBe thorough.\n</instructions>"));
    assert_eq!(
        prompts[2]["_meta"],
        json!({ "output": { "schema": { "type": "object" } } })
    );
    assert!(prompts[0].get("_meta").is_none());

    // A schema-less JSON format and unmapped structured output are rejected.
    let error = session
        .prompt(
            Message::user_text("x"),
            HarnessTurnOptions {
                response_format: Some(ResponseFormat::Json {
                    schema: None,
                    name: None,
                    description: None,
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires a JSON schema"), "{error}");
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn writes_filesystem_instructions_and_skills() {
    let fixture = fixture();
    let mut steps = start("s1");
    steps.push(end_turn());
    fixture.script(steps);
    let harness = fixture.harness(|settings| {
        settings.instruction_mapping = Some(AcpInstructionMapping::Filesystem {
            path: ".copilot/copilot-instructions.md".into(),
        });
        settings.skills_directory = Some(".copilot/skills".into());
    });
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    run(
        session.as_ref(),
        HarnessTurnOptions {
            settings: HarnessTurnSettings {
                instructions: Some("Use tabs.".into()),
                skills: vec![crate::harness::tools::HarnessSkill {
                    name: "deploy-app".into(),
                    description: "Deploys".into(),
                    content: "Run deploy.".into(),
                    files: None,
                }],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let instructions = std::fs::read_to_string(format!(
        "{}/home/.copilot/copilot-instructions.md",
        fixture.root
    ))
    .unwrap();
    assert!(instructions.contains("Use tabs."), "{instructions}");
    let skill = std::fs::read_to_string(format!(
        "{}/home/.copilot/skills/deploy-app/SKILL.md",
        fixture.root
    ))
    .unwrap();
    assert!(skill.contains("Run deploy."), "{skill}");
    // Filesystem instructions are not repeated in the prompt.
    assert_eq!(
        fixture.requests("session/prompt")[0]["prompt"],
        json!([{ "type": "text", "text": "hi" }])
    );

    let error = session
        .prompt(
            Message::user_text("x"),
            HarnessTurnOptions {
                settings: HarnessTurnSettings {
                    skills: vec![crate::harness::tools::HarnessSkill {
                        name: "Bad Name".into(),
                        description: String::new(),
                        content: String::new(),
                        files: None,
                    }],
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Invalid ACP skill name \"Bad Name\""),
        "{error}"
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn restores_sessions() {
    let fixture = fixture();
    fixture.script(vec![
        initialize(json!({ "loadSession": true })),
        // Replayed history arrives before the load response and is dropped.
        Step::Await("session/load"),
        update("s1", chunk("agent_message_chunk", "old", "old history")),
        Step::Emit(json!({ "jsonrpc": "2.0", "id": 1, "result": {} })),
        Step::Await("session/prompt"),
        update("s1", chunk("agent_message_chunk", "m1", "Back again")),
        Step::Emit(json!({ "jsonrpc": "2.0", "id": 2, "result": { "stopReason": "end_turn" } })),
        // The agent exits after the turn; the next turn restarts it.
        Step::Exit(0),
    ]);
    fixture.script_respawn(vec![
        initialize(json!({ "sessionCapabilities": { "resume": {} } })),
        Step::Reply("session/resume", json!({})),
        end_turn(),
    ]);
    let harness = fixture.harness(|_| {});
    let mut options = fixture.start_options(PermissionMode::AllowAll);
    options.resume = Some(HarnessResumeState {
        harness_id: "fake-agent".into(),
        data: json!({ "acpSessionId": "s1", "initialGuidanceApplied": true }),
    });
    let session = harness.start(options).await.unwrap();
    assert!(session.is_resume());
    let (_, messages) = run(session.as_ref(), HarnessTurnOptions::default()).await;
    assert_eq!(texts(&messages), ["Back again"]);
    let load = &fixture.requests("session/load")[0];
    assert_eq!(load["sessionId"], "s1");
    assert!(fixture.requests("session/new").is_empty());

    // Wait for the agent to exit.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    run(session.as_ref(), HarnessTurnOptions::default()).await;
    assert_eq!(fixture.spawns(), 2);
    assert_eq!(fixture.requests("session/resume")[0]["sessionId"], "s1");
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn restoring_requires_resume_or_load() {
    let fixture = fixture();
    fixture.script(vec![initialize(json!({}))]);
    let harness = fixture.harness(|_| {});
    let mut options = fixture.start_options(PermissionMode::AllowAll);
    options.resume = Some(HarnessResumeState {
        harness_id: "fake-agent".into(),
        data: json!({ "acpSessionId": "s1" }),
    });
    let session = harness.start(options).await.unwrap();
    let mut turn = session
        .prompt(Message::user_text("hi"), HarnessTurnOptions::default())
        .await
        .unwrap();
    let error = turn.result().await.unwrap_err().to_string();
    assert!(
        error.contains("sessionCapabilities.resume or loadSession"),
        "{error}"
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn reports_agent_failures() {
    let fixture = fixture();
    let mut steps = start("s1");
    steps.extend([Step::Await("session/prompt"), Step::Exit(3)]);
    fixture.script(steps);
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let (_, messages) = run(session.as_ref(), HarnessTurnOptions::default()).await;
    let last = last_assistant(&messages);
    assert_eq!(last.stop_reason, StopReason::Error);
    let message = last.error_message.clone().unwrap();
    assert!(message.contains("agent failed"), "{message}");
    session.destroy().await.unwrap();

    // A protocol version mismatch fails the turn.
    let fixture = self::fixture();
    fixture.script(vec![Step::Reply(
        "initialize",
        json!({ "protocolVersion": 2 }),
    )]);
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let mut turn = session
        .prompt(Message::user_text("hi"), HarnessTurnOptions::default())
        .await
        .unwrap();
    let error = turn.result().await.unwrap_err().to_string();
    assert!(
        error.contains("ACP protocol negotiation failed: requested v1, agent selected v2."),
        "{error}"
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn checks_image_prompt_support_and_authenticates() {
    let fixture = fixture();
    fixture.script(vec![
        Step::Reply(
            "initialize",
            json!({ "protocolVersion": 1, "agentCapabilities": {},
                "authMethods": [{ "id": "api-key", "name": "API key" }] }),
        ),
        Step::Reply("authenticate", json!({})),
        new_session("s1"),
    ]);
    let harness = fixture.harness(|settings| {
        settings.authentication = Some(AcpAuthentication {
            method_id: "api-key".into(),
            meta: Some(json!({ "source": "env" }).as_object().unwrap().clone()),
            client_capabilities: Some(
                json!({ "auth": { "terminal": false } })
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        });
    });
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let prompt = Message::User(crate::UserMessage {
        content: crate::UserMessageContent::Parts(vec![crate::UserContent::Image(
            crate::ImageContent {
                data: "aGk=".into(),
                mime_type: "image/png".into(),
            },
        )]),
        timestamp: 0,
    });
    let mut turn = session
        .prompt(prompt, HarnessTurnOptions::default())
        .await
        .unwrap();
    let error = turn.result().await.unwrap_err().to_string();
    assert!(
        error.contains("does not advertise support for image prompts"),
        "{error}"
    );
    assert_eq!(
        fixture.requests("authenticate"),
        [json!({ "methodId": "api-key", "_meta": { "source": "env" } })]
    );
    assert_eq!(
        fixture.requests("initialize")[0]["clientCapabilities"],
        json!({ "auth": { "terminal": false } })
    );
    session.destroy().await.unwrap();
}

#[test]
fn validates_settings() {
    let fixture = fixture();
    let check = |configure: &dyn Fn(&mut AcpHarnessSettings)| {
        let mut settings = fixture.settings();
        configure(&mut settings);
        validate_acp_settings(&settings)
            .err()
            .map(|error| error.to_string())
    };
    assert_eq!(check(&|_| {}), None);
    assert!(
        check(&|s| s.harness_id = "Bad Id".into())
            .unwrap()
            .contains("kebab-case")
    );
    assert!(
        check(&|s| s.executable = "bin/agent".into())
            .unwrap()
            .contains("bare command name")
    );
    assert!(
        check(&|s| s.credential_env = Some(vec!["KEY".into()]))
            .unwrap()
            .contains("configured together")
    );
    assert!(
        check(&|s| s.mcp_servers = Some(IndexMap::from([(
            HOST_TOOL_MCP_SERVER_NAME.to_string(),
            json!({})
        )])))
        .unwrap()
        .contains("reserved")
    );
    assert!(
        check(&|s| s.write_protected_paths = vec!["../x".into()])
            .unwrap()
            .contains("writeProtectedPaths")
    );
    assert!(
        check(&|s| {
            s.forward_env = vec!["A".into()];
            s.env = IndexMap::from([("A".to_string(), "1".to_string())]);
        })
        .unwrap()
        .contains("both forwardEnv and env")
    );
    assert_eq!(
        resolve_acp_skills_directory("/home/u", Some(".copilot/skills")).unwrap(),
        "/home/u/.copilot/skills"
    );
    assert!(resolve_acp_skills_directory("/home/u", Some("../x")).is_err());
    assert!(resolve_acp_skills_directory("/home/u", Some("/abs")).is_err());
}

#[test]
fn builds_the_bootstrap_recipe() {
    let recipe = acp_bootstrap(
        "cursor",
        &AcpSource::InstallCommand {
            command: "curl https://cursor.com/install -fsS | bash".into(),
        },
    );
    assert_eq!(recipe.bootstrap_dir, ".harness-bootstrap/cursor");
    assert_eq!(recipe.files[0].path, ".harness-bootstrap/cursor/install.sh");
    assert!(
        recipe.files[0]
            .content
            .contains("export HOME=\"$ACP_BOOTSTRAP_DIR/home\"")
    );
    assert!(
        recipe.files[0]
            .content
            .ends_with("curl https://cursor.com/install -fsS | bash\n")
    );
    assert_eq!(recipe.commands[0].command, "bash install.sh");
    assert_eq!(
        resolve_acp_agent_launch(
            "/b",
            &AcpSource::InstallCommand {
                command: "x".into()
            },
            "agent"
        ),
        (
            "/b/home/.local/bin/agent".to_string(),
            Some("/b/home".to_string())
        )
    );

    let recipe = acp_bootstrap(
        "github-copilot",
        &AcpSource::Npm {
            package: "@github/copilot".into(),
            version: "1.0.106".into(),
        },
    );
    let manifest: Value = serde_json::from_str(&recipe.files[0].content).unwrap();
    assert_eq!(
        manifest["dependencies"],
        json!({ "@github/copilot": "1.0.106" })
    );
    assert_eq!(
        resolve_acp_agent_launch(
            "/b",
            &AcpSource::Npm {
                package: "p".into(),
                version: "1".into()
            },
            "copilot"
        ),
        ("/b/node_modules/.bin/copilot".to_string(), None)
    );
}

#[test]
fn maps_instruction_configuration() {
    let mut env = IndexMap::from([("CFG".to_string(), "{\"a\":1}".to_string())]);
    let meta = turn::resolve_instruction_configuration(
        Some("Be kind."),
        Some(&AcpInstructionMapping::LaunchEnvJson {
            variable: "CFG".into(),
            path: vec!["prompt".into(), "system".into()],
        }),
        None,
        &mut env,
    )
    .unwrap();
    assert_eq!(meta, None);
    assert_eq!(
        serde_json::from_str::<Value>(&env["CFG"]).unwrap(),
        json!({ "a": 1, "prompt": { "system": "Be kind." } })
    );
    let mut env = IndexMap::from([("CFG".to_string(), "[]".to_string())]);
    assert!(
        turn::resolve_instruction_configuration(
            Some("x"),
            Some(&AcpInstructionMapping::LaunchEnvJson {
                variable: "CFG".into(),
                path: vec!["p".into()],
            }),
            None,
            &mut env,
        )
        .is_err()
    );
    assert!(
        turn::resolve_instruction_configuration(
            Some("x"),
            Some(&AcpInstructionMapping::SessionMeta {
                path: vec!["__proto__".into()],
            }),
            None,
            &mut IndexMap::new(),
        )
        .is_err()
    );
}

#[test]
fn resolves_model_ids() {
    let config = AcpModelMapping::SessionConfigOption {
        path: "model".into(),
    };
    let model = AcpModelMapping::SessionModel {
        path: "modelId".into(),
    };
    let state = json!({
        "configOptions": [{ "id": "other", "category": "model", "currentValue": "by-category" }],
        "models": { "currentModelId": "by-models" },
    });
    assert_eq!(
        turn::resolve_acp_model_id(&state, Some(&config)).as_deref(),
        Some("by-category")
    );
    assert_eq!(
        turn::resolve_acp_model_id(&state, Some(&model)).as_deref(),
        Some("by-models")
    );
    assert_eq!(turn::resolve_acp_model_id(&json!({}), Some(&config)), None);
}

#[tokio::test]
async fn probes_models_and_commands_from_a_new_session() {
    let fixture = fixture();
    fixture.script(vec![
        Step::Reply("initialize", json!({ "protocolVersion": 1 })),
        Step::Reply(
            "session/new",
            json!({
                "sessionId": "probe-1",
                "models": { "currentModelId": "m/b", "availableModels": [
                    { "modelId": "m/a", "name": "A" }, { "modelId": "m/b", "name": "B" }] },
                "configOptions": [{ "id": "effort", "options": [{ "value": "low" }, { "value": "high" }] }],
            }),
        ),
        Step::Emit(json!({ "jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": "probe-1",
            "update": { "sessionUpdate": "available_commands_update", "availableCommands": [
                { "name": "plan", "description": "Make a plan", "input": { "hint": "goal" } }] } } })),
    ]);
    let session = fixture
        .harness(|_| {})
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let probe = session.probe().await.unwrap();
    assert_eq!(probe.signed_in, Some(true));
    let ids: Vec<_> = probe.models.iter().map(|model| model.id.as_str()).collect();
    assert_eq!(ids, ["m/a", "m/b"]);
    assert!(probe.models[1].is_default);
    assert_eq!(probe.models[0].thinking_levels.len(), 2);
    assert_eq!(probe.commands.len(), 1);
    assert_eq!(probe.commands[0].hint, "goal");
    let new_session = fixture
        .stdin()
        .into_iter()
        .find(|message| message["method"] == "session/new")
        .unwrap();
    assert_eq!(new_session["params"]["mcpServers"], json!([]));
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn a_probe_refused_for_sign_in_reports_signed_out() {
    let fixture = fixture();
    fixture.script(vec![Step::Await("\"method\":\"initialize\""), Step::Emit(json!({
        "jsonrpc": "2.0", "id": 0, "error": { "code": -32000, "message": "Authentication required" } }))]);
    let session = fixture
        .harness(|_| {})
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let probe = session.probe().await.unwrap();
    assert_eq!(probe.signed_in, Some(false));
    assert!(probe.error.unwrap().contains("Authentication required"));
    session.destroy().await.unwrap();
}
