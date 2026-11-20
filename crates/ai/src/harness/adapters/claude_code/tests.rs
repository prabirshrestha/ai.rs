//! Claude Code tests against a fake `claude` CLI: a shell script that
//! prints scripted stream-json lines and waits for the adapter's replies.

use std::sync::{Arc, Mutex as StdMutex};

use serde_json::{Value, json};

use super::*;
use crate::harness::adapters::local_sandbox::{LocalSandbox, canonical};
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::session::{HarnessEvent, HarnessSandbox, HarnessTurnSettings, NoticeLevel};
use crate::{
    AgentEvent, AgentToolBuilder, AgentToolResult, AssistantContent, BeforeToolCallResult, Message,
    StopReason, ToolResultContent,
};

const FAKE_CLAUDE: &str = r#"
printf '%s\n' "$@" > "$FAKE_DIR/args"
exec 3< "$FAKE_DIR/steps"
while IFS= read -r step <&3; do
  case "$step" in
    "emit "*) printf '%s\n' "${step#emit }" ;;
    "await "*)
      pat="${step#await }"
      while IFS= read -r line; do
        printf '%s\n' "$line" >> "$FAKE_DIR/stdin"
        case "$line" in *"$pat"*) break ;; esac
      done ;;
    "exit "*) exit "${step#exit }" ;;
  esac
done
while IFS= read -r line; do printf '%s\n' "$line" >> "$FAKE_DIR/stdin"; done
"#;

enum Step {
    Emit(Value),
    Await(&'static str),
}

struct Fixture {
    _temp: TempDir,
    root: String,
    fake_dir: String,
}

fn fixture() -> Fixture {
    let temp = mkdtemp("claude-code-");
    let root = canonical(temp.path());
    let fake_dir = format!("{root}/.fake");
    std::fs::create_dir_all(&fake_dir).unwrap();
    std::fs::write(format!("{fake_dir}/claude.sh"), FAKE_CLAUDE).unwrap();
    std::fs::create_dir_all(format!("{root}/home")).unwrap();
    std::fs::create_dir_all(format!("{root}/work")).unwrap();
    Fixture {
        _temp: temp,
        root,
        fake_dir,
    }
}

impl Fixture {
    fn script(&self, steps: Vec<Step>) {
        let lines: Vec<String> = steps
            .into_iter()
            .map(|step| match step {
                Step::Emit(value) => format!("emit {value}"),
                Step::Await(pattern) => format!("await {pattern}"),
            })
            .collect();
        std::fs::write(format!("{}/steps", self.fake_dir), lines.join("\n") + "\n").unwrap();
        let _ = std::fs::remove_file(format!("{}/stdin", self.fake_dir));
    }

    fn harness(&self, configure: impl FnOnce(&mut ClaudeCodeHarnessSettings)) -> ClaudeCodeHarness {
        let mut settings = ClaudeCodeHarnessSettings {
            executable: Some(format!("sh {}/claude.sh", self.fake_dir)),
            env: IndexMap::from([("FAKE_DIR".to_string(), self.fake_dir.clone())]),
            process_env: Some(IndexMap::from([(
                "ANTHROPIC_API_KEY".to_string(),
                "sk-test".to_string(),
            )])),
            ..Default::default()
        };
        configure(&mut settings);
        ClaudeCodeHarness::new(settings)
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

    fn args(&self) -> Vec<String> {
        std::fs::read_to_string(format!("{}/args", self.fake_dir))
            .unwrap()
            .lines()
            .map(str::to_string)
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

fn init() -> Step {
    Step::Emit(json!({
        "type": "system", "subtype": "init", "session_id": "claude-session-1",
        "model": "claude-opus-5-5", "permissionMode": "bypassPermissions",
    }))
}

fn result() -> Step {
    Step::Emit(json!({
        "type": "result", "subtype": "success", "is_error": false, "result": "done",
        "session_id": "claude-session-1", "total_cost_usd": 0.01,
        "usage": { "input_tokens": 10, "output_tokens": 5 },
    }))
}

fn stream(event: Value) -> Step {
    Step::Emit(
        json!({ "type": "stream_event", "event": event, "session_id": "claude-session-1", "parent_tool_use_id": null }),
    )
}

fn assistant(id: &str, content: Value) -> Step {
    Step::Emit(json!({
        "type": "assistant",
        "message": { "id": id, "role": "assistant", "content": content, "usage": { "input_tokens": 3, "output_tokens": 2 } },
        "parent_tool_use_id": null, "session_id": "claude-session-1",
    }))
}

fn tool_result(id: &str, text: &str, is_error: bool) -> Step {
    Step::Emit(json!({
        "type": "user",
        "message": { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": id, "content": text, "is_error": is_error }] },
        "parent_tool_use_id": null, "session_id": "claude-session-1",
    }))
}

fn agent_events(events: &[HarnessEvent]) -> Vec<&AgentEvent> {
    events
        .iter()
        .filter_map(|event| match event {
            HarnessEvent::Agent(event) => Some(event),
            _ => None,
        })
        .collect()
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

#[tokio::test]
async fn streams_a_text_turn_and_resumes_the_conversation() {
    let fixture = fixture();
    fixture.script(vec![
        Step::Await("\"type\":\"user\""),
        init(),
        stream(json!({ "type": "message_start", "message": { "id": "msg_1", "usage": { "input_tokens": 7 } } })),
        stream(json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "thinking", "thinking": "" } })),
        stream(json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "thinking_delta", "thinking": "Hmm." } })),
        stream(json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "signature_delta", "signature": "sig" } })),
        stream(json!({ "type": "content_block_stop", "index": 0 })),
        stream(json!({ "type": "content_block_start", "index": 1, "content_block": { "type": "text", "text": "" } })),
        stream(json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "text_delta", "text": "Hello" } })),
        stream(json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "text_delta", "text": " world" } })),
        stream(json!({ "type": "content_block_stop", "index": 1 })),
        assistant("msg_1", json!([{ "type": "text", "text": "Hello world" }])),
        stream(json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 4 } })),
        stream(json!({ "type": "message_stop" })),
        result(),
    ]);
    let harness = fixture.harness(|settings| settings.max_turns = Some(3));
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let turn = session
        .prompt(
            Message::user_text("hi"),
            HarnessTurnOptions {
                settings: HarnessTurnSettings {
                    model: Some("claude-opus-5-5".into()),
                    instructions: Some("Be brief.".into()),
                    thinking_level: Some(ThinkingLevel::High),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let (events, messages) = turn.collect().await.unwrap();

    assert_eq!(texts(&messages), ["Hello world"]);
    let Message::Assistant(reply) = &messages[1] else {
        panic!("expected an assistant message");
    };
    assert_eq!(reply.model, "claude-opus-5-5");
    assert_eq!(reply.usage.input, 7);
    assert_eq!(reply.usage.output, 4);
    assert_eq!(reply.response_id.as_deref(), Some("msg_1"));
    assert!(
        matches!(&reply.content[0], AssistantContent::Thinking(t) if t.thinking == "Hmm." && t.thinking_signature.as_deref() == Some("sig"))
    );
    let deltas: Vec<_> = agent_events(&events)
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::MessageUpdate {
                assistant_message_event: crate::AssistantMessageEvent::TextDelta { delta, .. },
                ..
            } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, ["Hello", " world"]);
    assert!(matches!(
        agent_events(&events).last(),
        Some(AgentEvent::AgentEnd { .. })
    ));

    let args = fixture.args();
    let flag = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .map(|i| args[i + 1].clone())
    };
    assert_eq!(flag("--model").as_deref(), Some("claude-opus-5-5"));
    assert_eq!(
        flag("--permission-mode").as_deref(),
        Some("bypassPermissions")
    );
    assert_eq!(flag("--effort").as_deref(), Some("high"));
    assert_eq!(flag("--max-turns").as_deref(), Some("3"));
    assert!(args.contains(&"--allow-dangerously-skip-permissions".to_string()));
    assert!(args.contains(&"--include-partial-messages".to_string()));
    assert!(!args.contains(&"--resume".to_string()));
    let stdin = fixture.stdin();
    assert_eq!(stdin[0]["request"]["subtype"], "initialize");
    assert_eq!(stdin[0]["request"]["appendSystemPrompt"], "Be brief.");
    assert_eq!(stdin[1]["message"]["content"][0]["text"], "hi");

    // The next turn resumes the conversation the CLI reported.
    fixture.script(vec![Step::Await("\"type\":\"user\""), init(), result()]);
    let turn = session
        .prompt(Message::user_text("again"), HarnessTurnOptions::default())
        .await
        .unwrap();
    turn.collect().await.unwrap();
    let args = fixture.args();
    let resume = args.iter().position(|arg| arg == "--resume").unwrap();
    assert_eq!(args[resume + 1], "claude-session-1");
    let state = session.stop().await.unwrap();
    assert_eq!(state.data["claudeSessionId"], "claude-session-1");
}

#[tokio::test]
async fn runs_host_tools_over_mcp() {
    let fixture = fixture();
    fixture.script(vec![
        Step::Await("\"type\":\"user\""),
        init(),
        assistant(
            "msg_1",
            json!([{ "type": "tool_use", "id": "toolu_1", "name": "mcp__harness-tools__lookup", "input": { "query": "rust" } }]),
        ),
        Step::Emit(json!({
            "type": "control_request", "request_id": "mcp-list",
            "request": { "subtype": "mcp_message", "server_name": "harness-tools",
                "message": { "jsonrpc": "2.0", "id": 1, "method": "tools/list" } },
        })),
        Step::Await("\"request_id\":\"mcp-list\""),
        Step::Emit(json!({
            "type": "control_request", "request_id": "mcp-call",
            "request": { "subtype": "mcp_message", "server_name": "harness-tools",
                "message": { "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": { "name": "lookup", "arguments": { "query": "rust" }, "_meta": { "claudecode/toolUseId": "toolu_1" } } } },
        })),
        Step::Await("\"request_id\":\"mcp-call\""),
        tool_result("toolu_1", "found: rust", false),
        assistant("msg_2", json!([{ "type": "text", "text": "Done." }])),
        result(),
    ]);
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
    let harness = fixture.harness(|settings| {
        settings.tool_hints = IndexMap::from([(
            "lookup".to_string(),
            ClaudeCodeToolHint {
                read_only: Some(true),
                always_load: None,
            },
        )]);
    });
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let turn = session
        .prompt(
            Message::user_text("look up rust"),
            HarnessTurnOptions {
                settings: HarnessTurnSettings {
                    tools: vec![lookup],
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let (events, messages) = turn.collect().await.unwrap();

    let roles: Vec<&str> = messages
        .iter()
        .map(|message| match message {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "toolResult",
            Message::Custom(_) => "custom",
        })
        .collect();
    assert_eq!(roles, ["user", "assistant", "toolResult", "assistant"]);
    let Message::Assistant(call) = &messages[1] else {
        unreachable!()
    };
    assert!(
        matches!(&call.content[0], AssistantContent::ToolCall(c) if c.name == "lookup" && c.id == "toolu_1")
    );
    assert_eq!(call.stop_reason, StopReason::ToolUse);
    let Message::ToolResult(result) = &messages[2] else {
        unreachable!()
    };
    assert_eq!(result.tool_name, "lookup");
    assert!(matches!(&result.content[0], ToolResultContent::Text(t) if t.text == "found: rust"));
    let tool_events = agent_events(&events)
        .into_iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::ToolExecutionStart { .. } | AgentEvent::ToolExecutionEnd { .. }
            )
        })
        .count();
    assert_eq!(tool_events, 2);

    let stdin = fixture.stdin();
    assert_eq!(
        stdin[0]["request"]["sdkMcpServers"],
        json!(["harness-tools"])
    );
    let list = stdin
        .iter()
        .find(|line| line["response"]["request_id"] == "mcp-list")
        .unwrap();
    let tool = &list["response"]["response"]["mcp_response"]["result"]["tools"][0];
    assert_eq!(tool["name"], "lookup");
    assert_eq!(tool["annotations"]["readOnlyHint"], true);
    let call = stdin
        .iter()
        .find(|line| line["response"]["request_id"] == "mcp-call")
        .unwrap();
    assert_eq!(
        call["response"]["response"]["mcp_response"]["result"]["content"][0]["text"],
        "found: rust"
    );
}

#[tokio::test]
async fn asks_before_native_tools_the_mode_does_not_allow() {
    let fixture = fixture();
    fixture.script(vec![
        Step::Await("\"type\":\"user\""),
        init(),
        assistant(
            "msg_1",
            json!([{ "type": "tool_use", "id": "toolu_bash", "name": "Bash", "input": { "command": "rm -rf /" } }]),
        ),
        Step::Emit(json!({
            "type": "control_request", "request_id": "hook-1",
            "request": { "subtype": "hook_callback", "callback_id": "harness_approval", "tool_use_id": "toolu_bash",
                "input": { "hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_use_id": "toolu_bash",
                    "tool_input": { "command": "rm -rf /" }, "cwd": format!("{}/work", fixture.root) } },
        })),
        Step::Await("\"request_id\":\"hook-1\""),
        Step::Emit(json!({
            "type": "control_request", "request_id": "perm-1",
            "request": { "subtype": "can_use_tool", "tool_name": "Read", "tool_use_id": "toolu_read",
                "input": { "file_path": format!("{}/work/a.txt", fixture.root) } },
        })),
        Step::Await("\"request_id\":\"perm-1\""),
        tool_result("toolu_bash", "Not allowed", true),
        result(),
    ]);
    let asked: Arc<StdMutex<Vec<String>>> = Arc::default();
    let before: crate::BeforeToolCallFn = {
        let asked = asked.clone();
        Arc::new(move |context, _| {
            asked.lock().unwrap().push(context.tool_call.name.clone());
            Box::pin(async {
                Ok(Some(BeforeToolCallResult {
                    block: true,
                    reason: Some("Not allowed".into()),
                }))
            })
        })
    };
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowEdits))
        .await
        .unwrap();
    let turn = session
        .prompt(
            Message::user_text("clean up"),
            HarnessTurnOptions {
                before_tool_call: Some(before),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let (_, messages) = turn.collect().await.unwrap();

    assert_eq!(asked.lock().unwrap().as_slice(), ["bash"]);
    let stdin = fixture.stdin();
    let hook = stdin
        .iter()
        .find(|line| line["response"]["request_id"] == "hook-1")
        .unwrap();
    assert_eq!(
        hook["response"]["response"]["hookSpecificOutput"],
        json!({ "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Not allowed" })
    );
    let permission = stdin
        .iter()
        .find(|line| line["response"]["request_id"] == "perm-1")
        .unwrap();
    assert_eq!(permission["response"]["response"]["behavior"], "allow");
    let Message::ToolResult(result) = &messages[2] else {
        panic!("expected the bash result");
    };
    assert_eq!(result.tool_name, "bash");
    assert!(result.is_error);

    let args = fixture.args();
    let flag = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .map(|i| args[i + 1].clone())
    };
    assert_eq!(flag("--permission-mode").as_deref(), Some("acceptEdits"));
    let settings: Value = serde_json::from_str(&flag("--settings").unwrap()).unwrap();
    assert_eq!(
        settings["permissions"]["ask"],
        json!(["Bash(*)", "Monitor(*)"])
    );
    assert_eq!(
        stdin[0]["request"]["hooks"]["PreToolUse"][0]["hookCallbackIds"],
        json!(["harness_approval"])
    );
}

#[tokio::test]
async fn answers_questions_with_the_questions_tool() {
    let fixture = fixture();
    let native = json!({ "questions": [{ "question": "Which?", "header": "Pick", "multiSelect": false,
        "options": [{ "label": "A", "description": "" }, { "label": "B", "description": "" }] }] });
    fixture.script(vec![
        Step::Await("\"type\":\"user\""),
        init(),
        assistant(
            "msg_1",
            json!([{ "type": "tool_use", "id": "toolu_q", "name": "AskUserQuestion", "input": native }]),
        ),
        Step::Emit(json!({
            "type": "control_request", "request_id": "hook-q",
            "request": { "subtype": "hook_callback", "callback_id": "harness_question", "tool_use_id": "toolu_q",
                "input": { "hook_event_name": "PreToolUse", "tool_name": "AskUserQuestion", "tool_use_id": "toolu_q", "tool_input": native } },
        })),
        Step::Await("\"request_id\":\"hook-q\""),
        tool_result("toolu_q", "User answered", false),
        result(),
    ]);
    let questions = AgentToolBuilder::new("askUserQuestions")
        .description("Ask the user")
        .parameters(crate::harness::questions_tool::questions_tool_input_schema())
        .execute(|args| async move {
            assert_eq!(args["questions"][0]["id"], "question-1");
            let mut result = AgentToolResult::text("answered");
            result.details = Some(json!({
                "action": "answered",
                "answers": { "question-1": { "optionIds": ["option-2"] } },
            }));
            Ok(result)
        })
        .build()
        .unwrap();
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let turn = session
        .prompt(
            Message::user_text("ask me"),
            HarnessTurnOptions {
                settings: HarnessTurnSettings {
                    tools: vec![questions],
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let (_, messages) = turn.collect().await.unwrap();
    let stdin = fixture.stdin();
    // Only the questions tool: no MCP server.
    assert!(stdin[0]["request"].get("sdkMcpServers").is_none());
    let hook = stdin
        .iter()
        .find(|line| line["response"]["request_id"] == "hook-q")
        .unwrap();
    let output = &hook["response"]["response"]["hookSpecificOutput"];
    assert_eq!(output["permissionDecision"], "allow");
    assert_eq!(output["updatedInput"]["answers"], json!({ "Which?": "B" }));
    let Message::Assistant(call) = &messages[1] else {
        unreachable!()
    };
    assert!(
        matches!(&call.content[0], AssistantContent::ToolCall(c) if c.name == "askUserQuestions" && c.arguments["questions"][0]["id"] == "question-1")
    );
    let Message::ToolResult(result) = &messages[2] else {
        unreachable!()
    };
    assert_eq!(result.tool_name, "askUserQuestions");
    assert!(matches!(&result.content[0], ToolResultContent::Text(t) if t.text == "answered"));
}

#[tokio::test]
async fn reports_failed_results_and_early_exits() {
    let fixture = fixture();
    fixture.script(vec![
        Step::Await("\"type\":\"user\""),
        init(),
        Step::Emit(json!({ "type": "system", "subtype": "api_retry", "attempt": 1, "max_retries": 3, "error_status": 529, "error": "overloaded" })),
        Step::Emit(json!({ "type": "result", "subtype": "error_max_turns", "is_error": true, "errors": ["Reached max turns"] })),
    ]);
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let (events, messages) = session
        .prompt(Message::user_text("go"), HarnessTurnOptions::default())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let Some(Message::Assistant(last)) = messages.last() else {
        panic!("expected an error message");
    };
    assert_eq!(last.stop_reason, StopReason::Error);
    assert_eq!(last.error_message.as_deref(), Some("Reached max turns"));
    assert!(events.iter().any(|event| matches!(event,
        HarnessEvent::Notice { level: NoticeLevel::Warning, message }
            if message == "Claude Code API retry: attempt 1/3; HTTP 529; retrying in ms; overloaded"
            || message.starts_with("Claude Code API retry: attempt 1/3; HTTP 529"))));

    std::fs::write(
        format!("{}/steps", fixture.fake_dir),
        "await \"type\":\"user\"\nemit not json: boom\nexit 1\n",
    )
    .unwrap();
    let (_, messages) = session
        .prompt(Message::user_text("again"), HarnessTurnOptions::default())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let Some(Message::Assistant(last)) = messages.last() else {
        panic!("expected an error message");
    };
    assert_eq!(last.stop_reason, StopReason::Error);
    let error = last.error_message.clone().unwrap();
    assert!(error.contains("exited before the turn finished"), "{error}");
    assert!(error.contains("boom"), "{error}");
}

#[tokio::test]
async fn interrupts_a_cancelled_turn() {
    let fixture = fixture();
    fixture.script(vec![
        Step::Await("\"type\":\"user\""),
        init(),
        Step::Await("\"subtype\":\"interrupt\""),
        Step::Emit(
            json!({ "type": "result", "subtype": "error_during_execution", "is_error": true }),
        ),
    ]);
    let harness = fixture.harness(|_| {});
    let session = harness
        .start(fixture.start_options(PermissionMode::AllowAll))
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
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    token.cancel();
    let (_, messages) = tokio::time::timeout(std::time::Duration::from_secs(10), turn.collect())
        .await
        .unwrap()
        .unwrap();
    let Some(Message::Assistant(last)) = messages.last() else {
        panic!("expected an aborted message");
    };
    assert_eq!(last.stop_reason, StopReason::Aborted);
}

#[tokio::test]
async fn writes_skills_and_validates_options() {
    let fixture = fixture();
    fixture.script(vec![Step::Await("\"type\":\"user\""), init(), result()]);
    let harness = fixture.harness(|settings| {
        settings.mcp_servers = Some(IndexMap::from([(
            "docs".to_string(),
            json!({ "type": "http", "url": "https://example.com/mcp" }),
        )]));
    });
    let mut options = fixture.start_options(PermissionMode::AllowReads);
    options.builtin_tool_filter = Some(BuiltinToolFilter::Deny {
        tool_names: vec!["webSearch".into()],
    });
    let session = harness.start(options).await.unwrap();
    assert!(
        session
            .prompt(
                Message::user_text("x"),
                HarnessTurnOptions {
                    response_format: Some(ResponseFormat::Json {
                        schema: None,
                        name: None,
                        description: None
                    }),
                    ..Default::default()
                },
            )
            .await
            .is_err()
    );
    session
        .prompt(
            Message::user_text("go"),
            HarnessTurnOptions {
                settings: HarnessTurnSettings {
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
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let skill = std::fs::read_to_string(format!(
        "{}/home/.claude/skills/review/SKILL.md",
        fixture.root
    ))
    .unwrap();
    assert!(skill.contains("Review carefully."));
    let args = fixture.args();
    let flag = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .map(|i| args[i + 1].clone())
    };
    assert_eq!(flag("--allowedTools").as_deref(), Some("Skill"));
    assert_eq!(flag("--disallowedTools").as_deref(), Some("WebSearch"));
    assert_eq!(flag("--permission-mode").as_deref(), Some("default"));
    let mcp: Value = serde_json::from_str(&flag("--mcp-config").unwrap()).unwrap();
    assert_eq!(mcp["mcpServers"]["docs"]["type"], "http");

    let reserved = fixture.harness(|settings| {
        settings.mcp_servers = Some(IndexMap::from([("harness-tools".to_string(), json!({}))]));
    });
    assert!(
        reserved
            .start(fixture.start_options(PermissionMode::AllowAll))
            .await
            .is_err()
    );
}

#[test]
fn builds_the_bootstrap_recipe() {
    let recipe = claude_code_bootstrap();
    assert_eq!(recipe.harness_id, "claude-code");
    assert!(
        recipe.files[0]
            .content
            .contains("\"@anthropic-ai/claude-code\": \"2.1.319\"")
    );
    assert_eq!(
        recipe.commands[1].command,
        "./node_modules/.bin/claude --version"
    );
}

const FAKE_CLAUDE_PROBE: &str = r#"
printf '%s\n' "$@" >> "$FAKE_DIR/probe-args"
if [ "$1" = auth ]; then
  echo '{"loggedIn":true,"authMethod":"claude.ai","email":"me@example.com"}'
  exit 0
fi
IFS= read -r line
printf '%s\n' "$line" >> "$FAKE_DIR/probe-stdin"
echo '{"type":"control_response","response":{"subtype":"success","request_id":"probe","response":{"models":[{"value":"default","displayName":"Default"},{"value":"haiku","displayName":"Haiku"}],"commands":[{"name":"review","description":"Review","argumentHint":"<pr>"},{"name":"skill","description":"A skill (user)"}]}}}'
while IFS= read -r line; do :; done
"#;

#[tokio::test]
async fn probes_sign_in_models_and_commands() {
    let fixture = fixture();
    std::fs::write(format!("{}/probe.sh", fixture.fake_dir), FAKE_CLAUDE_PROBE).unwrap();
    let session = fixture
        .harness(|settings| {
            settings.executable = Some(format!("sh {}/probe.sh", fixture.fake_dir));
            settings.extra_args = vec!["--add-dir".into(), "/extra".into()];
        })
        .start(fixture.start_options(PermissionMode::AllowAll))
        .await
        .unwrap();
    let probe = session.probe().await.unwrap();
    assert_eq!(probe.signed_in, Some(true));
    assert_eq!(probe.account.as_deref(), Some("me@example.com"));
    assert_eq!(probe.models.len(), 2);
    assert!(probe.models[0].is_default);
    assert!(probe.models[1].thinking_levels.is_empty());
    assert_eq!(probe.commands.len(), 1);
    let stdin = std::fs::read_to_string(format!("{}/probe-stdin", fixture.fake_dir)).unwrap();
    assert!(stdin.contains("\"subtype\":\"initialize\""));
    let args = std::fs::read_to_string(format!("{}/probe-args", fixture.fake_dir)).unwrap();
    assert!(args.contains("--add-dir\n/extra"));
    session.destroy().await.unwrap();
}
