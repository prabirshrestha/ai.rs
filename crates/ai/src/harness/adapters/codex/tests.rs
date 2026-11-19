//! Codex tests against a fake `codex app-server`: a shell script that
//! answers scripted JSON-RPC requests and prints scripted notifications.

use std::sync::Arc;

use serde_json::{Value, json};

use super::tools::{HOOK_COMMAND, create_trusted_apply_patch_hook};
use super::*;
use crate::harness::adapters::local_sandbox::{LocalSandbox, canonical};
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::session::{
    FileChangeKind, HarnessEvent, HarnessSandbox, HarnessTurnSettings, NoticeLevel,
};
use crate::harness::tools::PermissionMode;
use crate::{
    AgentEvent, AgentToolBuilder, AgentToolResult, AssistantContent, Message, StopReason,
    ToolResultContent,
};

/// Steps: `emit <json>` prints a line; `await <pattern>` reads stdin until a
/// line contains the pattern; `reply <method> <json>` reads stdin until that
/// method's request and answers it; `exit <code>` exits.
const FAKE_CODEX: &str = r#"
echo spawn >> "$FAKE_DIR/spawns"
printf '%s\n' "$@" >> "$FAKE_DIR/args"
steps="$FAKE_DIR/steps"
if [ "$(wc -l < "$FAKE_DIR/spawns")" -gt 1 ] && [ -f "$FAKE_DIR/steps.2" ]; then
  steps="$FAKE_DIR/steps.2"
fi
exec 3< "$steps"
while IFS= read -r step <&3; do
  case "$step" in
    "emit "*) printf '%s\n' "${step#emit }" ;;
    "await "*)
      pat="${step#await }"
      while IFS= read -r line; do
        printf '%s\n' "$line" >> "$FAKE_DIR/stdin"
        case "$line" in *"$pat"*) break ;; esac
      done ;;
    "reply "*)
      rest="${step#reply }"
      method="${rest%% *}"
      result="${rest#* }"
      while IFS= read -r line; do
        printf '%s\n' "$line" >> "$FAKE_DIR/stdin"
        case "$line" in *"\"method\":\"$method\""*) break ;; esac
      done
      id=$(printf '%s' "$line" | sed -n 's/^{"id":\([0-9]*\).*/\1/p')
      printf '{"id":%s,"result":%s}\n' "$id" "$result" ;;
    "exit "*) exit "${step#exit }" ;;
  esac
done
while IFS= read -r line; do printf '%s\n' "$line" >> "$FAKE_DIR/stdin"; done
"#;

enum Step {
    Emit(Value),
    Await(&'static str),
    Reply(&'static str, Value),
    Exit(i32),
}

struct Fixture {
    _temp: TempDir,
    root: String,
    fake_dir: String,
}

fn fixture() -> Fixture {
    let temp = mkdtemp("codex-");
    let root = canonical(temp.path());
    let fake_dir = format!("{root}/.fake");
    std::fs::create_dir_all(&fake_dir).unwrap();
    std::fs::write(format!("{fake_dir}/codex.sh"), FAKE_CODEX).unwrap();
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
            Step::Exit(code) => format!("exit {code}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

impl Fixture {
    fn script(&self, steps: Vec<Step>) {
        std::fs::write(format!("{}/steps", self.fake_dir), steps_text(steps)).unwrap();
    }

    fn script_respawn(&self, steps: Vec<Step>) {
        std::fs::write(format!("{}/steps.2", self.fake_dir), steps_text(steps)).unwrap();
    }

    fn harness(&self, configure: impl FnOnce(&mut CodexHarnessSettings)) -> CodexHarness {
        let mut settings = CodexHarnessSettings {
            executable: Some(format!(
                "FAKE_DIR={} sh {}/codex.sh",
                shell_quote(&self.fake_dir),
                shell_quote(&self.fake_dir)
            )),
            process_env: Some(IndexMap::from([(
                "CODEX_API_KEY".to_string(),
                "sk-test".to_string(),
            )])),
            ..Default::default()
        };
        configure(&mut settings);
        CodexHarness::new(settings)
    }

    fn start_options(&self) -> HarnessStartOptions {
        HarnessStartOptions::new(
            "session-1",
            HarnessSandbox::Basic(Arc::new(
                LocalSandbox::new(format!("{}/work", self.root))
                    .with_home(format!("{}/home", self.root)),
            )),
            format!("{}/work", self.root),
        )
    }

    fn args(&self) -> Vec<String> {
        std::fs::read_to_string(format!("{}/args", self.fake_dir))
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn spawns(&self) -> usize {
        std::fs::read_to_string(format!("{}/spawns", self.fake_dir))
            .unwrap_or_default()
            .lines()
            .count()
    }

    fn requests(&self, method: &str) -> Vec<Value> {
        std::fs::read_to_string(format!("{}/stdin", self.fake_dir))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|message| message["method"] == method)
            .map(|message| message["params"].clone())
            .collect()
    }

    fn stdin(&self) -> Vec<Value> {
        std::fs::read_to_string(format!("{}/stdin", self.fake_dir))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }
}

fn thread_response(id: &str) -> Value {
    json!({
        "thread": { "id": id },
        "model": "gpt-5.5-codex",
        "approvalPolicy": "never",
        "sandbox": { "type": "dangerFullAccess" },
    })
}

/// Initialize, start the thread and the turn.
fn start_steps(thread: &'static str, turn: &'static str) -> Vec<Step> {
    vec![
        Step::Reply("initialize", json!({})),
        Step::Reply("thread/start", thread_response(thread)),
        Step::Reply("turn/start", json!({ "turn": { "id": turn } })),
    ]
}

fn notify(method: &str, params: Value) -> Step {
    Step::Emit(json!({ "method": method, "params": params }))
}

fn item(method: &str, turn: &str, item: Value) -> Step {
    notify(
        method,
        json!({ "threadId": "thr_1", "turnId": turn, "item": item }),
    )
}

fn complete(turn: &str, status: &str) -> Vec<Step> {
    let mut steps = vec![notify(
        "turn/completed",
        json!({ "threadId": "thr_1", "turn": { "id": turn, "status": status, "error": null } }),
    )];
    if status == "completed" {
        steps.push(Step::Reply(
            "thread/backgroundTerminals/list",
            json!({ "data": [] }),
        ));
    }
    steps
}

fn agent_message(turn: &str, id: &str, text: &str) -> Vec<Step> {
    vec![
        item(
            "item/started",
            turn,
            json!({ "type": "agentMessage", "id": id, "text": "" }),
        ),
        notify(
            "item/agentMessage/delta",
            json!({ "threadId": "thr_1", "turnId": turn, "itemId": id, "delta": text }),
        ),
        item(
            "item/completed",
            turn,
            json!({ "type": "agentMessage", "id": id, "text": text }),
        ),
    ]
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

async fn run(
    session: &dyn HarnessSession,
    options: HarnessTurnOptions,
) -> (Vec<HarnessEvent>, Vec<Message>) {
    let turn = session
        .prompt(Message::user_text("hi"), options)
        .await
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(8), turn.collect())
        .await
        .unwrap_or_else(|_| panic!("turn timed out"))
        .unwrap()
}

#[tokio::test]
async fn streams_turns_on_one_thread() {
    let fixture = fixture();
    let mut steps = start_steps("thr_1", "turn_1");
    steps.extend([
        item("item/started", "turn_1", json!({ "type": "reasoning", "id": "rs_1", "summary": [], "content": [] })),
        notify(
            "item/reasoning/summaryTextDelta",
            json!({ "threadId": "thr_1", "turnId": "turn_1", "itemId": "rs_1", "delta": "Thinking." }),
        ),
        item("item/completed", "turn_1", json!({ "type": "reasoning", "id": "rs_1", "summary": ["Thinking."], "content": [] })),
        item("item/started", "turn_1", json!({ "type": "agentMessage", "id": "msg_1", "text": "" })),
        notify(
            "item/agentMessage/delta",
            json!({ "threadId": "thr_1", "turnId": "turn_1", "itemId": "msg_1", "delta": "Hello" }),
        ),
        notify(
            "item/agentMessage/delta",
            json!({ "threadId": "thr_1", "turnId": "turn_1", "itemId": "msg_1", "delta": " world" }),
        ),
        // Another turn's events are ignored.
        notify(
            "item/agentMessage/delta",
            json!({ "threadId": "thr_1", "turnId": "turn_0", "itemId": "msg_0", "delta": "stale" }),
        ),
        item("item/completed", "turn_1", json!({ "type": "agentMessage", "id": "msg_1", "text": "Hello world" })),
        notify(
            "thread/tokenUsage/updated",
            json!({ "threadId": "thr_1", "turnId": "turn_1", "tokenUsage": {
                "total": { "inputTokens": 100, "cachedInputTokens": 40, "outputTokens": 9 },
                "last": { "inputTokens": 100, "cachedInputTokens": 40, "outputTokens": 9 },
            } }),
        ),
        notify("warning", json!({ "message": "heads up" })),
    ]);
    steps.extend(complete("turn_1", "completed"));
    steps.push(Step::Reply(
        "turn/start",
        json!({ "turn": { "id": "turn_2" } }),
    ));
    steps.extend(agent_message("turn_2", "msg_2", "Again."));
    steps.extend(complete("turn_2", "completed"));
    fixture.script(steps);

    let harness = fixture.harness(|settings| {
        settings.codex_config = Some(
            json!({ "model_reasoning_summary": "concise", "profile": "x" })
                .as_object()
                .cloned()
                .unwrap(),
        );
        settings.mcp_servers = Some(IndexMap::from([(
            "docs".to_string(),
            json!({ "url": "https://example.com/mcp" }),
        )]));
    });
    let session = harness.start(fixture.start_options()).await.unwrap();
    let options = || HarnessTurnOptions {
        settings: HarnessTurnSettings {
            instructions: Some("Be brief.".into()),
            thinking_level: Some(ThinkingLevel::High),
            ..Default::default()
        },
        ..Default::default()
    };
    let (events, messages) = run(session.as_ref(), options()).await;

    assert_eq!(texts(&messages), ["Hello world"]);
    let Message::Assistant(reply) = &messages[1] else {
        panic!("expected an assistant message");
    };
    assert_eq!(reply.model, "gpt-5.5-codex");
    assert!(
        matches!(&reply.content[0], AssistantContent::Thinking(t) if t.thinking == "Thinking.")
    );
    assert_eq!(reply.usage.input, 60);
    assert_eq!(reply.usage.cache_read, 40);
    assert_eq!(reply.usage.output, 9);
    assert_eq!(reply.stop_reason, StopReason::Stop);
    let deltas: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            HarnessEvent::Agent(AgentEvent::MessageUpdate {
                assistant_message_event: crate::AssistantMessageEvent::TextDelta { delta, .. },
                ..
            }) => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, ["Hello", " world"]);
    assert!(events.iter().any(|event| matches!(
        event,
        HarnessEvent::Notice { level: NoticeLevel::Warning, message } if message == "heads up"
    )));

    let args = fixture.args();
    assert_eq!(
        &args[..4],
        [
            "--config",
            "sandbox_mode=\"danger-full-access\"",
            "--config",
            "approval_policy=\"never\"",
        ]
    );
    assert_eq!(&args[args.len() - 2..], ["app-server", "--stdio"]);
    assert!(args.contains(&"model_provider=\"agent_bridge_openai\"".to_string()));
    let providers = args
        .iter()
        .find(|arg| arg.starts_with("model_providers="))
        .unwrap();
    assert!(providers.contains("\"base_url\"=\"https://api.openai.com/v1\""));
    assert!(providers.contains("\"env_key\"=\"CODEX_API_KEY\""));
    assert!(providers.contains("\"supports_standalone_web_search\"=true"));

    let initialize = &fixture.requests("initialize")[0];
    assert_eq!(initialize["clientInfo"]["name"], "ai-sdk-harness-codex");
    assert_eq!(initialize["capabilities"]["experimentalApi"], true);
    assert!(
        fixture
            .stdin()
            .iter()
            .any(|line| line["method"] == "initialized")
    );
    let thread = &fixture.requests("thread/start")[0];
    assert_eq!(thread["model"], "gpt-5.5");
    assert_eq!(thread["approvalPolicy"], "never");
    assert_eq!(thread["sandbox"], "danger-full-access");
    assert_eq!(thread["experimentalRawEvents"], true);
    assert_eq!(thread["dynamicTools"], json!([]));
    let config = &thread["config"];
    assert_eq!(config["model_reasoning_summary"], "concise");
    assert_eq!(config["profile"], "x");
    assert_eq!(config["web_search"], "disabled");
    assert_eq!(
        config["mcp_servers"]["docs"]["url"],
        "https://example.com/mcp"
    );
    assert_eq!(
        config["developer_instructions"],
        format!("Be brief.\n\n{FINAL_MESSAGE_INSTRUCTION}")
    );
    assert!(config.get("model_provider").is_none());
    let turn = &fixture.requests("turn/start")[0];
    assert_eq!(turn["threadId"], "thr_1");
    assert_eq!(turn["effort"], "high");
    assert_eq!(
        turn["input"],
        json!([{ "type": "text", "text": "hi", "text_elements": [] }])
    );
    assert_eq!(
        turn["sandboxPolicy"],
        json!({ "type": "externalSandbox", "networkAccess": "enabled" })
    );
    assert!(turn.get("environments").is_none());

    // The next turn reuses the process and the loaded thread.
    let (_, messages) = run(session.as_ref(), options()).await;
    assert_eq!(texts(&messages), ["Again."]);
    assert_eq!(fixture.spawns(), 1);
    assert_eq!(fixture.requests("thread/start").len(), 1);
    assert!(fixture.requests("thread/resume").is_empty());
    assert_eq!(fixture.requests("turn/start").len(), 2);

    let state = session.stop().await.unwrap();
    assert_eq!(state.harness_id, "codex");
    assert_eq!(state.data["threadId"], "thr_1");
    assert!(
        state.data["turnConfigurationFingerprint"]
            .as_str()
            .unwrap()
            .starts_with("v2:")
    );
}

#[tokio::test]
async fn reports_native_tools() {
    let fixture = fixture();
    let mut steps = start_steps("thr_1", "turn_1");
    steps.extend([
        item("item/started", "turn_1", json!({ "type": "commandExecution", "id": "cmd_1", "command": "ls", "status": "inProgress" })),
        item("item/completed", "turn_1", json!({
            "type": "commandExecution", "id": "cmd_1", "command": "ls",
            "status": "completed", "exitCode": 0, "aggregatedOutput": "a.txt\n",
        })),
        item("item/started", "turn_1", json!({ "type": "mcpToolCall", "id": "mcp_1", "server": "docs", "tool": "search", "arguments": { "q": "x" }, "status": "inProgress" })),
        item("item/completed", "turn_1", json!({
            "type": "mcpToolCall", "id": "mcp_1", "server": "docs", "tool": "search",
            "status": "failed", "error": { "message": "boom" },
        })),
        item("item/started", "turn_1", json!({ "type": "webSearch", "id": "ws_1", "query": "rust" })),
        item("item/completed", "turn_1", json!({ "type": "webSearch", "id": "ws_1", "query": "rust", "action": { "type": "search", "query": "rust" } })),
        notify("rawResponseItem/completed", json!({ "threadId": "thr_1", "turnId": "turn_1", "item": {
            "type": "custom_tool_call", "name": "apply_patch", "call_id": "call_patch", "input": "*** Begin Patch",
        } })),
        item("item/completed", "turn_1", json!({
            "type": "fileChange", "id": "call_patch", "status": "completed",
            "changes": [
                { "path": "a.txt", "kind": { "type": "add" } },
                { "path": "b.txt", "kind": { "type": "update" } },
                { "path": "c.txt", "kind": { "type": "delete" } },
            ],
        })),
        notify("rawResponseItem/completed", json!({ "threadId": "thr_1", "turnId": "turn_1", "item": {
            "type": "function_call", "name": "view_image", "call_id": "call_image", "arguments": "{\"path\":\"a.png\"}",
        } })),
    ]);
    steps.extend(agent_message("turn_1", "msg_1", "Done."));
    steps.extend(complete("turn_1", "completed"));
    fixture.script(steps);
    let session = fixture
        .harness(|_| {})
        .start(fixture.start_options())
        .await
        .unwrap();
    let (events, messages) = run(session.as_ref(), HarnessTurnOptions::default()).await;

    assert_eq!(
        tool_calls(&messages),
        [
            (
                "cmd_1".to_string(),
                "bash".to_string(),
                json!({ "command": "ls" })
            ),
            (
                "mcp_1".to_string(),
                "mcp__docs__search".to_string(),
                json!({ "q": "x" })
            ),
            (
                "ws_1".to_string(),
                "webSearch".to_string(),
                json!({ "query": "rust" })
            ),
            (
                "call_patch".to_string(),
                "apply_patch".to_string(),
                json!("*** Begin Patch")
            ),
            (
                "call_image".to_string(),
                "view_image".to_string(),
                json!({ "path": "a.png" })
            ),
        ]
    );
    let results = tool_results(&messages);
    assert_eq!(
        results[0],
        ("cmd_1".into(), "bash".into(), false, "a.txt\n".into())
    );
    assert!(results[1].2);
    assert_eq!(results[1].3, r#"{"error":"boom"}"#);
    assert_eq!(results[2].1, "webSearch");
    // The patch result comes at turn completion, confirmed by the file change.
    assert!(results.contains(&(
        "call_patch".into(),
        "apply_patch".into(),
        false,
        "Patch applied.".into()
    )));
    assert!(results.contains(&(
        "call_image".into(),
        "view_image".into(),
        true,
        "Codex did not report the tool result.".into()
    )));
    let Some(Message::ToolResult(bash)) = messages
        .iter()
        .find(|m| matches!(m, Message::ToolResult(r) if r.tool_call_id == "cmd_1"))
    else {
        panic!("expected the bash result");
    };
    assert_eq!(
        bash.details,
        Some(json!({ "exitCode": 0, "output": "a.txt\n", "status": "completed" }))
    );
    let changes: Vec<(FileChangeKind, String)> = events
        .iter()
        .filter_map(|event| match event {
            HarnessEvent::FileChange { kind, path } => Some((*kind, path.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        changes,
        [
            (FileChangeKind::Create, "a.txt".to_string()),
            (FileChangeKind::Modify, "b.txt".to_string()),
            (FileChangeKind::Delete, "c.txt".to_string()),
        ]
    );
    assert_eq!(texts(&messages), ["Done."]);
    // Each tool result follows the assistant message that called it.
    let order: Vec<&str> = messages
        .iter()
        .map(|message| match message {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "tool",
            _ => "other",
        })
        .collect();
    assert_eq!(order[..3], ["user", "assistant", "tool"]);
    // Unconfirmed native results come when the turn completes.
    assert_eq!(order.last(), Some(&"tool"));
}

#[tokio::test]
async fn runs_dynamic_tools() {
    let fixture = fixture();
    let mut steps = start_steps("thr_1", "turn_1");
    steps.extend([
        Step::Emit(json!({
            "id": 99, "method": "item/tool/call",
            "params": { "threadId": "thr_1", "turnId": "turn_1", "callId": "call_1", "tool": "lookup", "arguments": { "query": "rust" } },
        })),
        Step::Await("\"id\":99"),
        Step::Emit(json!({
            "id": 100, "method": "item/tool/call",
            "params": { "threadId": "thr_1", "turnId": "turn_1", "callId": "call_2", "tool": "missing", "arguments": {} },
        })),
        Step::Await("\"id\":100"),
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
    let session = fixture
        .harness(|_| {})
        .start(fixture.start_options())
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

    let thread = &fixture.requests("thread/start")[0];
    assert_eq!(thread["dynamicTools"][0]["name"], "lookup");
    assert_eq!(thread["dynamicTools"][0]["type"], "function");
    let responses: Vec<Value> = fixture
        .stdin()
        .into_iter()
        .filter(|line| line["id"] == 99 || line["id"] == 100)
        .collect();
    assert_eq!(
        responses[0]["result"],
        json!({ "contentItems": [{ "type": "inputText", "text": "found: rust" }], "success": true })
    );
    assert!(
        responses[1]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unknown dynamic tool 'missing'")
    );
    assert_eq!(
        tool_results(&messages)[0],
        (
            "call_1".into(),
            "lookup".into(),
            false,
            "found: rust".into()
        )
    );
    // The unknown tool fails the turn.
    let Some(Message::Assistant(last)) = messages.last() else {
        panic!("expected a failed message");
    };
    assert_eq!(last.stop_reason, StopReason::Error);
    assert!(last.error_message.as_deref().unwrap().contains("missing"));
    // The process is closed after a protocol failure.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
}

#[tokio::test]
async fn reports_failed_turns_and_early_exits() {
    let fixture = fixture();
    let mut steps = start_steps("thr_1", "turn_1");
    steps.extend([
        notify("error", json!({ "threadId": "thr_1", "turnId": "turn_1", "willRetry": true, "error": { "message": "retrying" } })),
        notify(
            "turn/completed",
            json!({ "threadId": "thr_1", "turn": { "id": "turn_1", "status": "failed", "error": { "message": "rate limited" } } }),
        ),
        Step::Reply("turn/start", json!({ "turn": { "id": "turn_2" } })),
        Step::Exit(3),
    ]);
    fixture.script(steps);
    let session = fixture
        .harness(|_| {})
        .start(fixture.start_options())
        .await
        .unwrap();
    let (events, messages) = run(session.as_ref(), HarnessTurnOptions::default()).await;
    assert!(events.iter().any(|event| matches!(
        event,
        HarnessEvent::Notice { level: NoticeLevel::Warning, message } if message == "retrying"
    )));
    let Some(Message::Assistant(last)) = messages.last() else {
        panic!("expected a failed message");
    };
    assert_eq!(last.stop_reason, StopReason::Error);
    assert_eq!(last.error_message.as_deref(), Some("rate limited"));

    // A failed turn keeps the process; this one exits mid-turn.
    let (_, messages) = run(session.as_ref(), HarnessTurnOptions::default()).await;
    let Some(Message::Assistant(last)) = messages.last() else {
        panic!("expected a failed message");
    };
    assert_eq!(last.stop_reason, StopReason::Error);
    assert!(
        last.error_message
            .as_deref()
            .unwrap()
            .contains("exited before the turn completed")
    );
    assert_eq!(fixture.spawns(), 1);
}

#[tokio::test]
async fn interrupts_a_cancelled_turn() {
    let fixture = fixture();
    let mut steps = start_steps("thr_1", "turn_1");
    steps.extend([
        Step::Reply("turn/interrupt", json!({})),
        notify(
            "turn/completed",
            json!({ "threadId": "thr_1", "turn": { "id": "turn_1", "status": "interrupted" } }),
        ),
    ]);
    fixture.script(steps);
    let session = fixture
        .harness(|_| {})
        .start(fixture.start_options())
        .await
        .unwrap();
    let token = CancellationToken::new();
    let turn = session
        .prompt(
            Message::user_text("go"),
            HarnessTurnOptions {
                cancellation_token: Some(token.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    token.cancel();
    let (_, messages) = tokio::time::timeout(std::time::Duration::from_secs(10), turn.collect())
        .await
        .unwrap()
        .unwrap();
    let Some(Message::Assistant(last)) = messages.last() else {
        panic!("expected an aborted message");
    };
    assert_eq!(last.stop_reason, StopReason::Aborted);
    assert_eq!(
        fixture.requests("turn/interrupt")[0],
        json!({ "threadId": "thr_1", "turnId": "turn_1" })
    );
}

#[tokio::test]
async fn steers_a_running_turn() {
    let fixture = fixture();
    let mut steps = start_steps("thr_1", "turn_1");
    steps.push(Step::Reply("turn/steer", json!({ "turnId": "turn_1" })));
    steps.extend(agent_message("turn_1", "msg_1", "Steered."));
    steps.extend(complete("turn_1", "completed"));
    fixture.script(steps);
    let session = fixture
        .harness(|_| {})
        .start(fixture.start_options())
        .await
        .unwrap();
    let turn = session
        .prompt(Message::user_text("go"), HarnessTurnOptions::default())
        .await
        .unwrap();
    let mut steered = false;
    for _ in 0..50 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        if turn.steer(Message::user_text("also this")).await.is_ok() {
            steered = true;
            break;
        }
    }
    assert!(steered);
    let (_, messages) = tokio::time::timeout(std::time::Duration::from_secs(10), turn.collect())
        .await
        .unwrap()
        .unwrap();
    let steer = &fixture.requests("turn/steer")[0];
    assert_eq!(steer["threadId"], "thr_1");
    assert_eq!(steer["expectedTurnId"], "turn_1");
    assert_eq!(steer["input"][0]["text"], "also this");
    assert!(
        messages
            .iter()
            .filter(|message| matches!(message, Message::User(_)))
            .count()
            == 2
    );
    assert_eq!(texts(&messages), ["Steered."]);
}

#[tokio::test]
async fn resumes_and_restarts_threads() {
    let fixture = fixture();
    let mut steps = vec![
        Step::Reply("initialize", json!({})),
        Step::Reply("thread/resume", thread_response("thr_1")),
        Step::Reply("turn/start", json!({ "turn": { "id": "turn_1" } })),
    ];
    steps.extend(complete("turn_1", "completed"));
    steps.push(Step::Reply(
        "turn/start",
        json!({ "turn": { "id": "turn_2" } }),
    ));
    steps.extend(complete("turn_2", "completed"));
    // New skills start a new thread on the same app server.
    steps.push(Step::Reply("thread/unsubscribe", json!({})));
    steps.push(Step::Reply("thread/start", thread_response("thr_1")));
    steps.push(Step::Reply(
        "turn/start",
        json!({ "turn": { "id": "turn_3" } }),
    ));
    steps.extend(complete("turn_3", "completed"));
    fixture.script(steps);
    // New instructions change the configuration: a new app server and thread.
    let mut respawn = start_steps("thr_1", "turn_4");
    respawn.extend(complete("turn_4", "completed"));
    fixture.script_respawn(respawn);
    let mut options = fixture.start_options();
    options.resume = Some(HarnessResumeState {
        harness_id: "codex".into(),
        data: json!({ "threadId": "thr_1" }),
    });
    let session = fixture.harness(|_| {}).start(options).await.unwrap();
    assert!(session.is_resume());
    let turn_options = |instructions: &str, skill: bool| HarnessTurnOptions {
        settings: HarnessTurnSettings {
            instructions: Some(instructions.into()),
            skills: if skill {
                vec![crate::harness::tools::HarnessSkill {
                    name: "review".into(),
                    description: "Review code".into(),
                    content: "Review carefully.".into(),
                    files: None,
                }]
            } else {
                Vec::new()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    run(session.as_ref(), turn_options("A", false)).await;
    let resume = &fixture.requests("thread/resume")[0];
    assert_eq!(resume["threadId"], "thr_1");
    assert_eq!(resume["excludeTurns"], true);
    assert!(resume.get("dynamicTools").is_none());
    run(session.as_ref(), turn_options("A", false)).await;
    assert!(fixture.requests("thread/start").is_empty());
    let (_, messages) = run(session.as_ref(), turn_options("A", true)).await;
    assert!(
        !messages
            .iter()
            .any(|m| matches!(m, Message::Assistant(m) if m.stop_reason == StopReason::Error))
    );
    assert_eq!(
        fixture.requests("thread/unsubscribe")[0],
        json!({ "threadId": "thr_1" })
    );
    assert_eq!(fixture.requests("thread/start").len(), 1);
    assert_eq!(fixture.spawns(), 1);
    run(session.as_ref(), turn_options("B", true)).await;
    assert_eq!(fixture.spawns(), 2);
    assert_eq!(fixture.requests("thread/start").len(), 2);
    assert_eq!(fixture.requests("turn/start").len(), 4);
}

#[tokio::test]
async fn filters_builtin_tools() {
    let fixture = fixture();
    let hook = create_trusted_apply_patch_hook(&Map::new()).unwrap();
    let hooks = json!({ "data": [{ "hooks": [{
        "key": hook.key, "eventName": "preToolUse", "handlerType": "command",
        "command": HOOK_COMMAND, "matcher": "^apply_patch$", "currentHash": hook.hash,
        "enabled": true, "trustStatus": "trusted",
    }], "errors": [] }] });
    let mut steps = vec![
        Step::Reply("initialize", json!({})),
        Step::Reply("hooks/list", hooks),
        Step::Reply("thread/start", thread_response("thr_1")),
        Step::Reply("turn/start", json!({ "turn": { "id": "turn_1" } })),
    ];
    steps.extend(complete("turn_1", "completed"));
    fixture.script(steps);
    let mut respawn = start_steps("thr_1", "turn_2");
    respawn.extend(complete("turn_2", "completed"));
    fixture.script_respawn(respawn);

    let mut options = fixture.start_options();
    options.builtin_tool_filter = Some(BuiltinToolFilter::Deny {
        tool_names: vec!["apply_patch".into(), "bash".into()],
    });
    let session = fixture
        .harness(|settings| settings.web_search = true)
        .start(options)
        .await
        .unwrap();
    run(session.as_ref(), HarnessTurnOptions::default()).await;
    let args = fixture.args();
    assert!(args.iter().any(|arg| arg.starts_with("hooks={")));
    assert!(args.contains(&"features.hooks=true".to_string()));
    let config = &fixture.requests("thread/start")[0]["config"];
    assert_eq!(config["web_search"], "live");
    assert_eq!(config["features"], json!({ "shell_tool": false }));
    assert!(
        fixture.requests("turn/start")[0]
            .get("environments")
            .is_none()
    );
    session.destroy().await.unwrap();

    // Without any environment tool, turns run without environments.
    let mut options = fixture.start_options();
    options.builtin_tool_filter = Some(BuiltinToolFilter::Allow {
        tool_names: vec!["webSearch".into()],
    });
    let session = fixture.harness(|_| {}).start(options).await.unwrap();
    run(session.as_ref(), HarnessTurnOptions::default()).await;
    assert_eq!(fixture.spawns(), 2);
    let turn = fixture.requests("turn/start").pop().unwrap();
    assert_eq!(turn["environments"], json!([]));
    let config = &fixture.requests("thread/start").pop().unwrap()["config"];
    assert_eq!(config["web_search"], "disabled");
    assert_eq!(
        config["features"],
        json!({ "shell_tool": false, "view_image": false })
    );
}

#[tokio::test]
async fn validates_options_and_writes_skills() {
    let fixture = fixture();
    let mut steps = start_steps("thr_1", "turn_1");
    steps.extend(complete("turn_1", "completed"));
    fixture.script(steps);
    let harness = fixture.harness(|_| {});
    let mut options = fixture.start_options();
    options.permission_mode = PermissionMode::AllowReads;
    assert!(harness.start(options).await.is_err());

    let session = harness.start(fixture.start_options()).await.unwrap();
    assert!(
        session
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
            .is_err()
    );
    run(
        session.as_ref(),
        HarnessTurnOptions {
            response_format: Some(ResponseFormat::Json {
                schema: Some(json!({ "type": "object" })),
                name: None,
                description: None,
            }),
            settings: HarnessTurnSettings {
                model: Some("gpt-5.6".into()),
                skills: vec![crate::harness::tools::HarnessSkill {
                    name: "review".into(),
                    description: "Review code".into(),
                    content: "Review carefully.".into(),
                    files: None,
                }],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let skill = std::fs::read_to_string(format!(
        "{}/home/.agents/skills/review/SKILL.md",
        fixture.root
    ))
    .unwrap();
    assert!(skill.contains("Review carefully."));
    let turn = &fixture.requests("turn/start")[0];
    assert_eq!(turn["outputSchema"], json!({ "type": "object" }));
    assert_eq!(turn["model"], "gpt-5.6");
}

#[test]
fn resolves_the_runtime_configuration() {
    let gateway = IndexMap::from([
        ("AI_GATEWAY_API_KEY".to_string(), "gw".to_string()),
        (
            "AI_GATEWAY_BASE_URL".to_string(),
            "https://ai-gateway.example/v1".to_string(),
        ),
    ]);
    let headers = IndexMap::from([("x-trace".to_string(), "1".to_string())]);
    let (config, model) =
        resolve_codex_runtime(&gateway, Some("gpt-5.5"), None, None, None, Some(&headers)).unwrap();
    assert_eq!(model.as_deref(), Some("openai/gpt-5.5"));
    let provider = &config["model_providers"]["agent_bridge_openai"];
    assert_eq!(provider["base_url"], "https://ai-gateway.example/v1");
    assert!(provider.get("supports_standalone_web_search").is_none());
    assert_eq!(provider["http_headers"]["x-trace"], "1");
    assert_eq!(
        provider["http_headers"]["x-client-app"],
        format!("ai-sdk-harness-codex/{VERSION}")
    );
    assert_eq!(config["developer_instructions"], FINAL_MESSAGE_INSTRUCTION);

    let (config, model) =
        resolve_codex_runtime(&IndexMap::new(), Some("gpt-5.5"), None, None, None, None).unwrap();
    assert_eq!(model.as_deref(), Some("gpt-5.5"));
    assert!(config.get("model_provider").is_none());

    let missing = IndexMap::from([("AI_GATEWAY_API_KEY".to_string(), "gw".to_string())]);
    assert!(resolve_codex_runtime(&missing, None, None, None, None, None).is_err());
}

#[test]
fn builds_the_bootstrap_recipe() {
    let recipe = codex_bootstrap();
    assert_eq!(recipe.harness_id, "codex");
    assert!(
        recipe.files[0]
            .content
            .contains("\"@openai/codex\": \"0.167.0\"")
    );
    assert_eq!(
        recipe.commands[1].command,
        "./node_modules/.bin/codex --version"
    );
}
