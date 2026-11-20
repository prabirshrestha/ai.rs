//! Port of `harness-fx/src/fx-harness.test.ts` for what ai.rs sessions do:
//! the engine is a script that plays the core's events.

use std::sync::Mutex as StdMutex;
use std::sync::atomic::AtomicUsize;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::oneshot;

use super::super::{FxHarness, FxHarnessSettings, FxServiceTier, FxTurnResult};
use super::*;
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::sandbox::SandboxRunResult;
use crate::harness::session::{Harness, HarnessSandbox, HarnessStartOptions, HarnessTurnSettings};
use crate::harness::test_support::MockSandbox;
use crate::harness::tools::{BuiltinToolFilter, HarnessSkill};
use crate::{AgentToolBuilder, AssistantContent, BeforeToolCallResult};

type Script = Arc<
    dyn Fn(FxEngineOptions, Vec<FxPromptBlock>) -> BoxFuture<'static, HarnessResult<FxTurnResult>>
        + Send
        + Sync,
>;

#[derive(Default)]
struct Recorder {
    options: Mutex<Vec<FxEngineOptions>>,
    prompts: Mutex<Vec<Vec<FxPromptBlock>>>,
    steers: Mutex<Vec<String>>,
    closes: AtomicUsize,
}

struct MockEngine {
    options: FxEngineOptions,
    script: Script,
    recorder: Arc<Recorder>,
}

struct MockControl {
    cancelled: CancellationToken,
    recorder: Arc<Recorder>,
}

impl FxTurnControl for MockControl {
    fn steer(&self, text: &str) -> HarnessResult<()> {
        self.recorder.steers.lock().push(text.to_string());
        Ok(())
    }

    fn cancel(&self) {
        self.cancelled.cancel();
    }
}

#[async_trait]
impl FxEngine for MockEngine {
    fn prompt(&self, prompt: Vec<FxPromptBlock>) -> HarnessResult<FxTurn> {
        self.recorder.prompts.lock().push(prompt.clone());
        let cancelled = CancellationToken::new();
        let run = (self.script)(self.options.clone(), prompt);
        let token = cancelled.clone();
        Ok(FxTurn {
            result: Box::pin(async move {
                tokio::select! {
                    result = run => result,
                    _ = token.cancelled() => Err(HarnessError::new("cancelled").into()),
                }
            }),
            control: Arc::new(MockControl {
                cancelled,
                recorder: self.recorder.clone(),
            }),
        })
    }

    async fn checkpoint(&self) -> HarnessResult<Vec<u8>> {
        Ok(vec![1, 2, 3])
    }

    async fn close(&self) {
        self.recorder
            .closes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

fn factory(recorder: Arc<Recorder>, script: Script) -> FxRuntimeFactory {
    Arc::new(move |options| {
        recorder.options.lock().push(options.clone());
        let engine: Arc<dyn FxEngine> = Arc::new(MockEngine {
            options,
            script: script.clone(),
            recorder: recorder.clone(),
        });
        Box::pin(async move { Ok(engine) })
    })
}

fn ended() -> HarnessResult<FxTurnResult> {
    Ok(FxTurnResult {
        stop_reason: "end_turn".into(),
        ..Default::default()
    })
}

/// A script that does nothing.
fn idle() -> Script {
    Arc::new(|_, _| Box::pin(async { ended() }))
}

fn emit(options: &FxEngineOptions, event: FxEngineEvent) {
    options.events.send(event).unwrap();
}

fn request(options: &FxEngineOptions, index: usize) {
    emit(
        options,
        FxEngineEvent::Transport(FxTransportSignal::Request {
            index,
            model_id: Some("model-1".into()),
        }),
    );
}

async fn call_tool(options: &FxEngineOptions, name: &str, id: &str, input: Value) -> FxToolResult {
    let (reply, receiver) = oneshot::channel();
    emit(
        options,
        FxEngineEvent::ToolCall {
            name: name.into(),
            input,
            call_id: id.into(),
            signal: CancellationToken::new(),
            reply,
        },
    );
    receiver.await.unwrap()
}

struct Fixture {
    home: TempDir,
    recorder: Arc<Recorder>,
    sandbox: MockSandbox,
}

fn sandbox() -> MockSandbox {
    MockSandbox::new(|options, _| {
        let command = &options.process.command;
        Ok(if command.contains("realpath") {
            // `realpath -mz -- '<path>' | base64`
            let path = command
                .split("realpath -mz -- ")
                .nth(1)
                .and_then(|rest| rest.split('\'').find(|part| part.starts_with('/')))
                .unwrap_or("/workspace")
                .to_string();
            SandboxRunResult {
                exit_code: 0,
                stdout: base64::engine::general_purpose::STANDARD.encode(format!("{path}\0")),
                ..Default::default()
            }
        } else {
            SandboxRunResult {
                exit_code: 0,
                stdout: format!("ran: {command}"),
                ..Default::default()
            }
        })
    })
}

impl Fixture {
    fn new() -> Self {
        Self {
            home: mkdtemp("fx-session-"),
            recorder: Arc::default(),
            sandbox: sandbox(),
        }
    }

    fn harness(&self, script: Script, configure: impl FnOnce(&mut FxHarnessSettings)) -> FxHarness {
        let mut settings = FxHarnessSettings {
            process_env: Some(IndexMap::from([(
                "AI_GATEWAY_API_KEY".to_string(),
                "test-key".to_string(),
            )])),
            home_directory: Some(self.home.path().to_path_buf()),
            runtime_factory: Some(factory(self.recorder.clone(), script)),
            ..Default::default()
        };
        configure(&mut settings);
        FxHarness::new(settings)
    }

    fn start_options(&self, session_id: &str) -> HarnessStartOptions {
        HarnessStartOptions::new(
            session_id,
            HarnessSandbox::Basic(Arc::new(self.sandbox.clone())),
            "/workspace",
        )
    }

    fn put(&self, file: &str, value: Value) {
        let path = self.home.path().join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, value.to_string()).unwrap();
    }
}

fn fx_credentials(token: &str) -> Value {
    json!({
        "version": 1,
        "access_token": token,
        "refresh_token": "refresh",
        "account_id": "account",
        "expires_at_ms": 4_000_000_000_000_i64,
    })
}

async fn run(
    session: &dyn HarnessSession,
    prompt: AgentMessage,
    options: HarnessTurnOptions,
) -> Vec<AgentMessage> {
    session
        .prompt(prompt, options)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .1
}

fn assistants(messages: &[AgentMessage]) -> Vec<&AssistantMessage> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(message) => Some(message),
            _ => None,
        })
        .collect()
}

fn texts(message: &AssistantMessage) -> Vec<String> {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

fn session_state(state: &HarnessResumeState) -> FxSessionState {
    serde_json::from_value(state.data["session"].clone()).unwrap()
}

#[tokio::test]
async fn constructs_without_a_bootstrap() {
    let harness = FxHarness::default();
    assert_eq!(harness.id(), "fx");
    assert!(harness.supports_builtin_tool_approvals());
    assert!(harness.supports_builtin_tool_filtering());
    assert!(harness.bootstrap().await.unwrap().is_none());
    assert_eq!(
        harness.builtin_tools().keys().collect::<Vec<_>>(),
        vec![
            "read",
            "write",
            "edit",
            "bash",
            "grep",
            "glob",
            "ls",
            "askUserQuestions"
        ]
    );
    let bash = &harness.builtin_tools()["bash"];
    assert_eq!(bash.kind, Some(BuiltinToolKind::Bash));
    assert_eq!(bash.tool.parameters["required"], json!(["command"]));
}

#[tokio::test]
async fn translates_text_reasoning_usage_and_the_finish() {
    let fixture = Fixture::new();
    let harness = fixture.harness(
        Arc::new(|options, _| {
            Box::pin(async move {
                request(&options, 0);
                emit(
                    &options,
                    FxEngineEvent::Turn(FxRuntimeEvent::ReasoningDelta("think".into())),
                );
                emit(
                    &options,
                    FxEngineEvent::Turn(FxRuntimeEvent::TextDelta("Hel".into())),
                );
                emit(
                    &options,
                    FxEngineEvent::Turn(FxRuntimeEvent::TextDelta("lo".into())),
                );
                emit(
                    &options,
                    FxEngineEvent::ModelEnd(FxModelEnd {
                        finish_reason: Some("stop".into()),
                        usage: Some(super::super::FxModelUsage {
                            input_tokens: Some(100),
                            output_tokens: Some(20),
                            cache_read_tokens: Some(30),
                            cache_write_tokens: Some(10),
                            reasoning_tokens: Some(5),
                        }),
                    }),
                );
                ended()
            })
        }),
        |_| {},
    );
    let session = harness.start(fixture.start_options("text")).await.unwrap();
    let messages = run(
        session.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions::default(),
    )
    .await;
    let assistants = assistants(&messages);
    assert_eq!(assistants.len(), 1);
    let message = assistants[0];
    assert_eq!(texts(message), vec!["Hello"]);
    assert!(
        matches!(&message.content[0], AssistantContent::Thinking(thinking) if thinking.thinking == "think")
    );
    assert_eq!(message.api, "fx");
    assert_eq!(message.provider, "gateway");
    assert_eq!(message.model, "model-1");
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        (
            message.usage.input,
            message.usage.output,
            message.usage.cache_read,
            message.usage.cache_write
        ),
        (60, 20, 30, 10)
    );
    assert_eq!(message.usage.reasoning, Some(5));
    assert_eq!(message.usage.total_tokens, 120);
    let options = fixture.recorder.options.lock()[0].clone();
    assert!(options.hooks);
    assert!(
        matches!(&options.provider, FxEngineProvider::Gateway { api_key } if api_key == "test-key")
    );
    assert_eq!(
        options.instructions,
        "All filesystem and shell tools operate in the sandbox workspace /workspace."
    );
    assert_eq!(
        fixture
            .recorder
            .closes
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}

#[tokio::test]
async fn fails_output_that_matches_no_model_request() {
    for rejected in [false, true] {
        let fixture = Fixture::new();
        let harness = fixture.harness(
            Arc::new(move |options, _| {
                Box::pin(async move {
                    if rejected {
                        emit(
                            &options,
                            FxEngineEvent::Transport(FxTransportSignal::Rejected {
                                index: 0,
                                model_id: Some("gpt-test".into()),
                                status: 429,
                            }),
                        );
                    }
                    emit(
                        &options,
                        FxEngineEvent::Turn(FxRuntimeEvent::TextDelta("orphan".into())),
                    );
                    std::future::pending::<()>().await;
                    ended()
                })
            }),
            |_| {},
        );
        let session = harness
            .start(fixture.start_options("orphan"))
            .await
            .unwrap();
        let messages = run(
            session.as_ref(),
            Message::user_text("hi"),
            HarnessTurnOptions::default(),
        )
        .await;
        let last = *assistants(&messages).last().unwrap();
        assert_eq!(last.stop_reason, StopReason::Error);
        assert_eq!(
            last.error_message.as_deref(),
            Some(if rejected {
                "fx gateway model request gpt-test failed with HTTP 429."
            } else {
                "fx parsed output could not be matched to its model request."
            })
        );
    }
}

#[tokio::test]
async fn reapplies_per_turn_settings_and_leaves_the_model_default_to_libfx() {
    let fixture = Fixture::new();
    let harness = fixture.harness(idle(), |settings| {
        settings.effort = Some("high".into());
        settings.service_tier = Some(FxServiceTier::Fast);
    });
    let session = harness
        .start(fixture.start_options("settings"))
        .await
        .unwrap();
    run(
        session.as_ref(),
        Message::user_text("one"),
        HarnessTurnOptions::default(),
    )
    .await;
    run(
        session.as_ref(),
        Message::user_text("two"),
        HarnessTurnOptions {
            settings: HarnessTurnSettings {
                model: Some("openai/gpt-test".into()),
                instructions: Some("Be brief.".into()),
                skills: vec![HarnessSkill {
                    name: "review".into(),
                    description: "Review code".into(),
                    content: "Read the diff.".into(),
                    files: None,
                }],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let options = fixture.recorder.options.lock().clone();
    assert_eq!(options[0].model, None);
    assert_eq!(options[1].model.as_deref(), Some("openai/gpt-test"));
    assert_eq!(
        options[1].instructions,
        "Be brief.\n\n<skill name=\"review\">\n<description>Review code</description>\nRead the diff.\n</skill>\n\nAll filesystem and shell tools operate in the sandbox workspace /workspace."
    );
    for options in &options {
        assert_eq!(options.effort.as_deref(), Some("high"));
        assert_eq!((options.fast, options.ultrafast), (Some(true), Some(false)));
    }
    // The first turn saves a checkpoint, which the second one restores.
    assert_eq!(options[0].checkpoint, None);
    assert_eq!(options[1].checkpoint, Some(vec![1, 2, 3]));
}

#[tokio::test]
async fn maps_service_tiers_to_exclusive_libfx_flags() {
    for (tier, expected) in [
        (None, (None, None)),
        (Some(FxServiceTier::Default), (None, None)),
        (Some(FxServiceTier::Fast), (Some(true), Some(false))),
        (Some(FxServiceTier::Ultrafast), (Some(false), Some(true))),
    ] {
        let fixture = Fixture::new();
        let harness = fixture.harness(idle(), |settings| settings.service_tier = tier);
        let session = harness.start(fixture.start_options("tier")).await.unwrap();
        run(
            session.as_ref(),
            Message::user_text("hi"),
            HarnessTurnOptions::default(),
        )
        .await;
        let options = fixture.recorder.options.lock()[0].clone();
        assert_eq!((options.fast, options.ultrafast), expected);
    }
}

#[tokio::test]
async fn forwards_image_prompts() {
    let fixture = Fixture::new();
    let harness = fixture.harness(idle(), |_| {});
    let session = harness.start(fixture.start_options("image")).await.unwrap();
    let prompt = Message::User(UserMessage {
        content: UserMessageContent::Parts(vec![
            UserContent::Text(crate::TextContent {
                text: "What is this?".into(),
                text_signature: None,
            }),
            UserContent::Image(ImageContent {
                data: "aW1n".into(),
                mime_type: "image/png".into(),
            }),
        ]),
        timestamp: 0,
    });
    run(session.as_ref(), prompt, HarnessTurnOptions::default()).await;
    assert_eq!(
        fixture.recorder.prompts.lock()[0],
        vec![
            FxPromptBlock::Text("What is this?".into()),
            FxPromptBlock::Image(FxImage {
                data: "aW1n".into(),
                mime_type: "image/png".into(),
            }),
        ]
    );
}

fn echo_tool(name: &str) -> crate::DynAgentTool {
    AgentToolBuilder::new(name)
        .description("Echo the input")
        .parameters(json!({ "type": "object", "properties": { "text": { "type": "string" } } }))
        .execute(|args| async move {
            Ok(AgentToolResult::text(format!(
                "echo {}",
                args["text"].as_str().unwrap_or("")
            )))
        })
        .build()
        .unwrap()
}

#[tokio::test]
async fn filters_builtins_and_lets_custom_tools_override_them() {
    let fixture = Fixture::new();
    let harness = fixture.harness(idle(), |_| {});
    let mut start = fixture.start_options("filter");
    start.builtin_tool_filter = Some(BuiltinToolFilter::Allow {
        tool_names: vec!["read".into(), "bash".into()],
    });
    let session = harness.start(start).await.unwrap();
    run(
        session.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions {
            settings: HarnessTurnSettings {
                tools: vec![echo_tool("bash"), echo_tool("lookup")],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let options = fixture.recorder.options.lock()[0].clone();
    let names: Vec<&str> = options
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(names, vec!["read", "bash", "lookup"]);
    assert_eq!(options.tools[1].description, "Echo the input");
}

#[tokio::test]
async fn runs_host_tools_with_the_model_tool_call_ids() {
    let fixture = Fixture::new();
    let results = Arc::new(Mutex::new(Vec::new()));
    let seen = results.clone();
    let harness = fixture.harness(
        Arc::new(move |options, _| {
            let seen = seen.clone();
            Box::pin(async move {
                request(&options, 0);
                emit(
                    &options,
                    FxEngineEvent::Transport(FxTransportSignal::ToolCalls {
                        index: 0,
                        calls: vec![
                            FxToolCall {
                                tool_call_id: "call-1".into(),
                                tool_name: "lookup".into(),
                                input: json!("{\"text\":\"fx\"}"),
                            },
                            FxToolCall {
                                tool_call_id: "call-x".into(),
                                tool_name: "unknown".into(),
                                input: json!("{}"),
                            },
                        ],
                    }),
                );
                emit(
                    &options,
                    FxEngineEvent::ModelEnd(FxModelEnd {
                        finish_reason: Some("tool-calls".into()),
                        usage: None,
                    }),
                );
                let result = call_tool(&options, "lookup", "call-1", json!({ "text": "fx" })).await;
                seen.lock().push(result);
                request(&options, 1);
                emit(
                    &options,
                    FxEngineEvent::Turn(FxRuntimeEvent::TextDelta("done".into())),
                );
                ended()
            })
        }),
        |_| {},
    );
    let session = harness.start(fixture.start_options("host")).await.unwrap();
    let messages = run(
        session.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions {
            settings: HarnessTurnSettings {
                tools: vec![echo_tool("lookup")],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    assert_eq!(results.lock().clone(), vec![FxToolResult::text("echo fx")]);
    let assistants = assistants(&messages);
    assert_eq!(assistants.len(), 2);
    let calls: Vec<&ToolCall> = assistants[0]
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        (calls[0].id.as_str(), calls[0].name.as_str()),
        ("call-1", "lookup")
    );
    assert_eq!(calls[0].arguments, json!({ "text": "fx" }));
    assert_eq!(assistants[0].stop_reason, StopReason::ToolUse);
    let results: Vec<_> = messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tool_call_id, "call-1");
    assert!(!results[0].is_error);
    assert_eq!(texts(assistants[1]), vec!["done"]);
}

/// A script that calls `name` once and reports what came back.
fn one_call(name: &'static str, input: Value, seen: Arc<Mutex<Vec<FxToolResult>>>) -> Script {
    Arc::new(move |options, _| {
        let seen = seen.clone();
        let input = input.clone();
        Box::pin(async move {
            request(&options, 0);
            let result = call_tool(&options, name, "call-1", input).await;
            seen.lock().push(result);
            ended()
        })
    })
}

#[tokio::test]
async fn gates_shell_execution_before_touching_the_sandbox() {
    for (mode, hook, expected_run) in [
        (PermissionMode::AllowAll, None, true),
        (PermissionMode::AllowEdits, None, false),
        (PermissionMode::AllowReads, Some(true), true),
        (PermissionMode::AllowReads, Some(false), false),
    ] {
        let fixture = Fixture::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let harness = fixture.harness(
            one_call("bash", json!({ "command": "echo hi" }), seen.clone()),
            |_| {},
        );
        let mut start = fixture.start_options("bash");
        start.permission_mode = mode;
        let session = harness.start(start).await.unwrap();
        let before: Option<crate::BeforeToolCallFn> = hook.map(|allow| {
            let hook: crate::BeforeToolCallFn = Arc::new(move |context, _| {
                assert_eq!(context.tool_call.name, "bash");
                Box::pin(async move {
                    Ok((!allow).then(|| BeforeToolCallResult {
                        block: true,
                        reason: Some("No shell today.".into()),
                    }))
                })
            });
            hook
        });
        run(
            session.as_ref(),
            Message::user_text("hi"),
            HarnessTurnOptions {
                before_tool_call: before,
                ..Default::default()
            },
        )
        .await;
        let ran = fixture
            .sandbox
            .state
            .lock()
            .runs
            .iter()
            .any(|command| command == "echo hi");
        assert_eq!(ran, expected_run, "{mode:?} {hook:?}");
        let result = seen.lock()[0].clone();
        if expected_run {
            assert!(!result.is_error);
            assert_eq!(
                serde_json::from_str::<Value>(&result.text).unwrap(),
                json!({ "exitCode": 0, "stdout": "ran: echo hi", "stderr": "" })
            );
        } else {
            assert!(result.is_error);
            assert_eq!(
                result.text,
                if hook.is_some() {
                    "No shell today."
                } else {
                    "Tool 'bash' requires approval, but no approval handler was provided."
                }
            );
        }
    }
}

#[tokio::test]
async fn asks_before_edits_in_allow_reads_mode() {
    let fixture = Fixture::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let harness = fixture.harness(
        one_call(
            "write",
            json!({ "file_path": "notes.txt", "content": "hello" }),
            seen.clone(),
        ),
        |_| {},
    );
    let mut start = fixture.start_options("edit");
    start.permission_mode = PermissionMode::AllowReads;
    let session = harness.start(start).await.unwrap();
    run(
        session.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions::default(),
    )
    .await;
    assert!(seen.lock()[0].is_error);
    assert!(fixture.sandbox.state.lock().writes.is_empty());

    let seen = Arc::new(Mutex::new(Vec::new()));
    let fixture = Fixture::new();
    let harness = fixture.harness(
        one_call(
            "write",
            json!({ "file_path": "notes.txt", "content": "hello" }),
            seen.clone(),
        ),
        |_| {},
    );
    let session = harness.start(fixture.start_options("edit")).await.unwrap();
    run(
        session.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions::default(),
    )
    .await;
    assert_eq!(seen.lock()[0], FxToolResult::text("Wrote notes.txt"));
    assert_eq!(
        fixture.sandbox.state.lock().writes,
        vec![("/workspace/notes.txt".to_string(), "hello".to_string())]
    );
}

#[test]
fn requires_approval_by_permission_mode_and_kind() {
    use BuiltinToolKind::{Bash, Edit, Readonly};
    use PermissionMode::{AllowAll, AllowEdits, AllowReads};
    let cases = [
        (AllowAll, Bash, false),
        (AllowAll, Edit, false),
        (AllowEdits, Bash, true),
        (AllowEdits, Edit, false),
        (AllowEdits, Readonly, false),
        (AllowReads, Edit, true),
        (AllowReads, Bash, true),
        (AllowReads, Readonly, false),
    ];
    for (mode, kind, expected) in cases {
        assert_eq!(
            fx_tool_requires_approval(mode, kind),
            expected,
            "{mode:?} {kind:?}"
        );
    }
}

#[tokio::test]
async fn aborts_the_turn_and_accepts_a_fresh_prompt() {
    let fixture = Fixture::new();
    let harness = fixture.harness(
        Arc::new(|options, prompt| {
            Box::pin(async move {
                request(&options, 0);
                if prompt == vec![FxPromptBlock::Text("hang".into())] {
                    std::future::pending::<()>().await;
                }
                emit(
                    &options,
                    FxEngineEvent::Turn(FxRuntimeEvent::TextDelta("ok".into())),
                );
                ended()
            })
        }),
        |_| {},
    );
    let session = harness.start(fixture.start_options("abort")).await.unwrap();
    let token = CancellationToken::new();
    let turn = session
        .prompt(
            Message::user_text("hang"),
            HarnessTurnOptions {
                cancellation_token: Some(token.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    token.cancel();
    let (_, messages) = turn.collect().await.unwrap();
    assert_eq!(
        assistants(&messages).last().unwrap().stop_reason,
        StopReason::Aborted
    );
    let messages = run(
        session.as_ref(),
        Message::user_text("again"),
        HarnessTurnOptions::default(),
    )
    .await;
    assert_eq!(texts(assistants(&messages)[0]), vec!["ok"]);
}

#[tokio::test]
async fn rejects_a_second_concurrent_turn_and_steers_the_running_one() {
    let fixture = Fixture::new();
    let (release, released) = tokio::sync::watch::channel(false);
    let harness = fixture.harness(
        Arc::new(move |options, _| {
            let mut released = released.clone();
            Box::pin(async move {
                request(&options, 0);
                let _ = released.wait_for(|released| *released).await;
                emit(
                    &options,
                    FxEngineEvent::Turn(FxRuntimeEvent::UserMessage("also this".into())),
                );
                ended()
            })
        }),
        |_| {},
    );
    let session = harness.start(fixture.start_options("steer")).await.unwrap();
    let turn = session
        .prompt(Message::user_text("first"), HarnessTurnOptions::default())
        .await
        .unwrap();
    let second = session
        .prompt(Message::user_text("second"), HarnessTurnOptions::default())
        .await;
    assert!(
        second
            .err()
            .unwrap()
            .to_string()
            .contains("already running")
    );
    turn.steer(Message::user_text("also this")).await.unwrap();
    assert_eq!(
        fixture.recorder.steers.lock().clone(),
        vec!["also this".to_string()]
    );
    release.send(true).unwrap();
    let (_, messages) = turn.collect().await.unwrap();
    assert!(messages.iter().any(|message| matches!(
        message,
        Message::User(UserMessage { content: UserMessageContent::Text(text), .. }) if text == "also this"
    )));
}

#[tokio::test]
async fn surfaces_selected_subscription_failures_without_switching_providers() {
    let fixture = Fixture::new();
    fixture.put(".fx/chatgpt-auth.json", fx_credentials("secret"));
    let home = fixture.home.path().to_path_buf();
    let harness = fixture.harness(
        Arc::new(move |options, _| {
            let home = home.clone();
            Box::pin(async move {
                let FxEngineProvider::Subscription(authorizer) = &options.provider else {
                    panic!("Expected a subscription");
                };
                std::fs::remove_file(home.join(".fx/chatgpt-auth.json")).unwrap();
                let request = super::super::FxHttpRequest {
                    url: "https://chatgpt.com/backend-api/codex/responses".into(),
                    method: "POST".into(),
                    headers: Vec::new(),
                    body: None,
                };
                let _ = authorizer
                    .fetch(
                        "https://chatgpt.com/backend-api/codex/responses",
                        request,
                        CancellationToken::new(),
                    )
                    .await;
                Err(HarnessError::new("HostStreamFailed").into())
            })
        }),
        |settings| {
            settings.auth = Some(crate::harness::auth::HarnessAuthentication::Choice(
                crate::harness::auth::HarnessDirectAuthentication::Direct,
            ));
        },
    );
    let session = harness
        .start(fixture.start_options("credentials"))
        .await
        .unwrap();
    let messages = run(
        session.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions {
            settings: HarnessTurnSettings {
                model: Some("gpt-5".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await;
    let last = *assistants(&messages).last().unwrap();
    assert_eq!(last.stop_reason, StopReason::Error);
    assert_eq!(
        last.error_message.as_deref(),
        Some("The selected fx codex subscription credentials are no longer available.")
    );
    assert_eq!(fixture.recorder.options.lock().len(), 1);
}

#[tokio::test]
async fn pins_subscription_providers_across_resume() {
    let fixture = Fixture::new();
    fixture.put(".fx/grok-auth.json", fx_credentials("secret"));
    let harness = fixture.harness(idle(), |settings| {
        settings.process_env = Some(IndexMap::new())
    });
    let session = harness.start(fixture.start_options("pin")).await.unwrap();
    let state = session.stop().await.unwrap();
    assert_eq!(state.harness_id, "fx");
    assert_eq!(session_state(&state).provider, FxProvider::Grok);
    // A Codex credential appears; the resumed session keeps Grok.
    fixture.put(".fx/chatgpt-auth.json", fx_credentials("codex"));
    let mut start = fixture.start_options("pin");
    start.resume = Some(state);
    let resumed = harness.start(start).await.unwrap();
    assert!(resumed.is_resume());
    run(
        resumed.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions::default(),
    )
    .await;
    let options = fixture.recorder.options.lock()[0].clone();
    assert!(matches!(
        &options.provider,
        FxEngineProvider::Subscription(authorizer)
            if authorizer.provider() == crate::harness::adapters::fx::FxSubscriptionProvider::Grok
    ));
    // The saved model comes from fx's settings.
    assert_eq!(options.model, None);
}

#[tokio::test]
async fn never_stores_session_state_in_the_sandbox_and_returns_stable_state() {
    let fixture = Fixture::new();
    let harness = fixture.harness(idle(), |_| {});
    let session = harness.start(fixture.start_options("state")).await.unwrap();
    run(
        session.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions::default(),
    )
    .await;
    let state = session.detach().await.unwrap();
    assert_eq!(
        state.data,
        json!({ "session": { "provider": "gateway", "checkpoint": "AQID" } })
    );
    {
        let sandbox = fixture.sandbox.state.lock();
        assert!(sandbox.writes.is_empty());
        assert!(sandbox.runs.is_empty());
    }
    let error = session
        .prompt(Message::user_text("later"), HarnessTurnOptions::default())
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("stopped"));
}

#[tokio::test]
async fn rejects_state_from_another_harness() {
    let fixture = Fixture::new();
    let harness = fixture.harness(idle(), |_| {});
    let mut start = fixture.start_options("other");
    start.resume = Some(HarnessResumeState {
        harness_id: "codex".into(),
        data: json!({}),
    });
    let error = harness.start(start).await.err().unwrap();
    assert_eq!(
        error.to_string(),
        "Cannot import lifecycle state from a different harness."
    );
}

#[derive(Default)]
struct MemoryStorage {
    entries: Mutex<Vec<Value>>,
}

#[async_trait]
impl super::super::FxSessionStorage for MemoryStorage {
    async fn append(
        &self,
        _session_id: &str,
        entry: super::super::FxSessionStorageEntry,
    ) -> HarnessResult<()> {
        self.entries.lock().push(serde_json::to_value(entry)?);
        Ok(())
    }

    async fn read(
        &self,
        _session_id: &str,
        _leaf_id: &str,
        _token: Option<CancellationToken>,
    ) -> HarnessResult<Vec<Value>> {
        Ok(self.entries.lock().clone())
    }
}

#[tokio::test]
async fn resumes_from_the_storage_leaf_and_appends_only_changed_state() {
    let fixture = Fixture::new();
    let storage = Arc::new(MemoryStorage::default());
    let shared = storage.clone();
    let harness = fixture.harness(idle(), move |settings| {
        settings.session_storage = Some(shared)
    });
    let session = harness
        .start(fixture.start_options("stored"))
        .await
        .unwrap();
    run(
        session.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions::default(),
    )
    .await;
    let state = session.stop().await.unwrap();
    let leaf = state.data["leafId"].as_str().unwrap().to_string();
    assert_eq!(storage.entries.lock().len(), 1);
    assert_eq!(
        storage.entries.lock()[0]["data"],
        json!({ "provider": "gateway", "checkpoint": "AQID" })
    );
    let mut start = fixture.start_options("stored");
    start.resume = Some(state);
    let resumed = harness.start(start).await.unwrap();
    let state = resumed.stop().await.unwrap();
    // Nothing changed, so nothing new was appended.
    assert_eq!(state.data["leafId"], json!(leaf));
    assert_eq!(storage.entries.lock().len(), 1);
}

#[tokio::test]
async fn appends_nothing_when_the_session_is_destroyed() {
    let fixture = Fixture::new();
    let storage = Arc::new(MemoryStorage::default());
    let shared = storage.clone();
    let harness = fixture.harness(idle(), move |settings| {
        settings.session_storage = Some(shared)
    });
    let session = harness
        .start(fixture.start_options("destroyed"))
        .await
        .unwrap();
    run(
        session.as_ref(),
        Message::user_text("hi"),
        HarnessTurnOptions::default(),
    )
    .await;
    session.destroy().await.unwrap();
    assert!(storage.entries.lock().is_empty());
}

#[tokio::test]
async fn checks_the_resume_state_against_the_session_storage() {
    let fixture = Fixture::new();
    let harness = fixture.harness(idle(), |_| {});
    let mut start = fixture.start_options("leaf");
    start.resume = Some(HarnessResumeState {
        harness_id: "fx".into(),
        data: json!({ "leafId": "entry-1" }),
    });
    let error = harness.start(start).await.err().unwrap();
    assert!(error.to_string().contains("has no sessionStorage"));

    let storage: Arc<dyn super::super::FxSessionStorage> = Arc::new(MemoryStorage::default());
    let harness = fixture.harness(idle(), move |settings| {
        settings.session_storage = Some(storage)
    });
    let mut start = fixture.start_options("whole");
    start.resume = Some(HarnessResumeState {
        harness_id: "fx".into(),
        data: json!({ "session": { "provider": "gateway" } }),
    });
    let error = harness.start(start).await.err().unwrap();
    assert!(error.to_string().contains("must hold a leafId"));
}

#[tokio::test]
async fn requires_gateway_credentials_without_a_subscription() {
    let fixture = Fixture::new();
    let harness = fixture.harness(idle(), |settings| {
        settings.process_env = Some(IndexMap::new())
    });
    let error = harness
        .start(fixture.start_options("none"))
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("fx requires AI_GATEWAY_API_KEY"));
}

/// The whole session against fx's real core (`AI_FX_CORE_WASM`, or the
/// cached libfx download), with fixture model responses.
#[tokio::test(flavor = "multi_thread")]
async fn runs_a_turn_with_a_sandbox_tool_on_the_real_core() {
    if super::super::libfx::engine_tests::core_module()
        .await
        .is_none()
    {
        return;
    }
    let fixture = Fixture::new();
    let requests = Arc::new(StdMutex::new(Vec::new()));
    let fetch = super::super::libfx::engine_tests::fixture_fetch(
        vec![
            include_str!("fixtures/gateway-0.sse.txt"),
            include_str!("fixtures/gateway-1.sse.txt"),
        ],
        requests.clone(),
    );
    let harness = FxHarness::new(FxHarnessSettings {
        process_env: Some(IndexMap::from([(
            "AI_GATEWAY_API_KEY".to_string(),
            "test-key".to_string(),
        )])),
        home_directory: Some(fixture.home.path().to_path_buf()),
        fetch: Some(fetch),
        ..Default::default()
    });
    let session = harness
        .start(fixture.start_options("native"))
        .await
        .unwrap();
    let turn = session
        .prompt(
            Message::user_text("Run printf LIBFX_FIXTURE."),
            HarnessTurnOptions {
                settings: HarnessTurnSettings {
                    model: Some("anthropic/claude-haiku-4.5".into()),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let (_, messages) = tokio::time::timeout(std::time::Duration::from_secs(180), turn.collect())
        .await
        .expect("the turn ends")
        .unwrap();
    let assistants = assistants(&messages);
    assert_eq!(assistants.len(), 2, "{messages:#?}");
    assert!(assistants[0].content.iter().any(|content| matches!(
        content,
        AssistantContent::ToolCall(call) if call.id == "toolu_01KswTwrpijqi88ffraLQVsY" && call.name == "bash"
    )));
    assert!(assistants[0].usage.input > 0);
    assert!(texts(assistants[1]).concat().contains("READY"));
    assert_eq!(assistants[1].stop_reason, StopReason::Stop);
    let results: Vec<_> = messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1);
    assert!(!results[0].is_error);
    assert!(
        fixture
            .sandbox
            .state
            .lock()
            .runs
            .iter()
            .any(|command| command == "printf LIBFX_FIXTURE")
    );
    {
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].header("x-client-app"), Some(FX_CLIENT_APP));
    }
    let state = session.stop().await.unwrap();
    assert!(
        state.data["session"]["checkpoint"]
            .as_str()
            .is_some_and(|checkpoint| !checkpoint.is_empty())
    );
}
