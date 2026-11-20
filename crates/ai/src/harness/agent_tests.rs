//! `harness-agent.test.ts` (settings, sessions, prepareCall and tool
//! filtering), against a mock harness that records each start and turn.

use std::sync::Mutex as StdMutex;

use async_trait::async_trait;
use serde_json::json;

use super::*;
use crate::harness::session::HarnessTurnSender;
use crate::harness::test_support::MockSandbox;
use crate::harness::tools::BuiltinTool;
use crate::{AgentToolBuilder, AgentToolResult, Message, Tool};

#[derive(Default)]
struct Recorded {
    starts: Vec<HarnessStartOptions>,
    turns: Vec<(AgentMessage, HarnessTurnSettings, Option<ResponseFormat>)>,
}

struct MockHarness {
    builtins: IndexMap<String, BuiltinTool>,
    approvals: bool,
    filtering: bool,
    recorded: Arc<StdMutex<Recorded>>,
}

impl MockHarness {
    fn new(builtins: &[&str]) -> Self {
        Self {
            builtins: builtins
                .iter()
                .map(|name| {
                    let tool = Tool {
                        name: name.to_string(),
                        description: String::new(),
                        parameters: json!({ "type": "object" }),
                        constrained_sampling: None,
                    };
                    (name.to_string(), BuiltinTool::from(tool))
                })
                .collect(),
            approvals: true,
            filtering: true,
            recorded: Arc::default(),
        }
    }
}

struct MockSession {
    id: String,
    recorded: Arc<StdMutex<Recorded>>,
}

#[async_trait]
impl HarnessSession for MockSession {
    fn harness_id(&self) -> &str {
        "mock"
    }
    fn session_id(&self) -> &str {
        &self.id
    }
    fn is_resume(&self) -> bool {
        false
    }
    async fn prompt(
        &self,
        message: AgentMessage,
        options: HarnessTurnOptions,
    ) -> AgentResult<HarnessTurn> {
        self.recorded.lock().unwrap().turns.push((
            message,
            options.settings,
            options.response_format,
        ));
        Ok(HarnessTurn::spawn(
            |_sender: HarnessTurnSender| async move { Ok(vec![]) },
        ))
    }
    async fn detach(&self) -> AgentResult<HarnessResumeState> {
        unreachable!()
    }
    async fn stop(&self) -> AgentResult<HarnessResumeState> {
        unreachable!()
    }
    async fn destroy(&self) -> AgentResult<()> {
        Ok(())
    }
}

#[async_trait]
impl Harness for MockHarness {
    fn id(&self) -> &str {
        "mock"
    }
    fn builtin_tools(&self) -> &IndexMap<String, BuiltinTool> {
        &self.builtins
    }
    fn supports_builtin_tool_approvals(&self) -> bool {
        self.approvals
    }
    fn supports_builtin_tool_filtering(&self) -> bool {
        self.filtering
    }
    async fn start(&self, options: HarnessStartOptions) -> AgentResult<Box<dyn HarnessSession>> {
        let id = options.session_id.clone();
        self.recorded.lock().unwrap().starts.push(options);
        Ok(Box::new(MockSession {
            id,
            recorded: self.recorded.clone(),
        }))
    }
}

fn tool(name: &str) -> DynAgentTool {
    AgentToolBuilder::new(name)
        .description("A tool")
        .parameters(json!({ "type": "object" }))
        .execute(|_| async { Ok(AgentToolResult::text("ok")) })
        .build()
        .unwrap()
}

fn sandbox() -> HarnessSandbox {
    HarnessSandbox::Basic(Arc::new(MockSandbox::new(|_, _| Ok(Default::default()))))
}

fn agent(
    harness: MockHarness,
    settings: HarnessAgentSettings,
) -> (AgentResult<HarnessAgent>, Arc<StdMutex<Recorded>>) {
    let recorded = harness.recorded.clone();
    (HarnessAgent::new(Arc::new(harness), settings), recorded)
}

async fn run(agent: &HarnessAgent, session: &dyn HarnessSession, options: HarnessAgentCallOptions) {
    agent
        .prompt(session, Message::user_text("hi"), options)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
}

#[tokio::test]
async fn every_turn_gets_the_agents_settings() {
    let (agent, recorded) = agent(
        MockHarness::new(&["bash"]),
        HarnessAgentSettings {
            id: Some("auth-agent".into()),
            model: Some("claude-sonnet-4-5".into()),
            thinking_level: Some(ThinkingLevel::High),
            instructions: Some("Prefer minimal diffs.".into()),
            tools: vec![tool("lookup")],
            response_format: Some(ResponseFormat::Json {
                schema: Some(json!({ "type": "object" })),
                name: None,
                description: None,
            }),
            ..Default::default()
        },
    );
    let agent = agent.unwrap();
    assert_eq!(agent.id(), Some("auth-agent"));
    assert_eq!(agent.harness_id(), "mock");
    let session = agent
        .create_session(HarnessAgentSessionOptions::new(sandbox(), "/work"))
        .await
        .unwrap();
    run(&agent, session.as_ref(), Default::default()).await;
    run(&agent, session.as_ref(), Default::default()).await;

    let recorded = recorded.lock().unwrap();
    assert_eq!(recorded.starts.len(), 1);
    assert!(!recorded.starts[0].session_id.is_empty());
    assert_eq!(recorded.starts[0].permission_mode, PermissionMode::AllowAll);
    assert_eq!(recorded.turns.len(), 2);
    for (_, settings, format) in &recorded.turns {
        assert_eq!(settings.model.as_deref(), Some("claude-sonnet-4-5"));
        assert_eq!(settings.thinking_level, Some(ThinkingLevel::High));
        assert_eq!(
            settings.instructions.as_deref(),
            Some("Prefer minimal diffs.")
        );
        assert_eq!(tool_names(&settings.tools), ["lookup"]);
        assert!(matches!(format, Some(ResponseFormat::Json { .. })));
    }
}

#[tokio::test]
async fn prepare_call_changes_one_call() {
    let prepare: PrepareCallFn = Arc::new(|mut call: HarnessCall| {
        Box::pin(async move {
            if call.options.as_ref().and_then(|o| o["fast"].as_bool()) == Some(true) {
                call.model = Some("haiku".into());
                call.instructions = Some("Be quick.".into());
                call.prompt = Message::user_text("rewritten");
            }
            Ok(call)
        })
    });
    let (agent, recorded) = agent(
        MockHarness::new(&[]),
        HarnessAgentSettings {
            model: Some("sonnet".into()),
            prepare_call: Some(prepare),
            ..Default::default()
        },
    );
    let agent = agent.unwrap();
    let session = agent
        .create_session(HarnessAgentSessionOptions::new(sandbox(), "/work"))
        .await
        .unwrap();
    run(
        &agent,
        session.as_ref(),
        HarnessAgentCallOptions {
            options: Some(json!({ "fast": true })),
            ..Default::default()
        },
    )
    .await;
    run(&agent, session.as_ref(), Default::default()).await;

    let recorded = recorded.lock().unwrap();
    let (prompt, settings, _) = &recorded.turns[0];
    assert_eq!(settings.model.as_deref(), Some("haiku"));
    assert_eq!(settings.instructions.as_deref(), Some("Be quick."));
    assert!(matches!(
        prompt,
        Message::User(user) if matches!(&user.content, crate::UserMessageContent::Text(text) if text == "rewritten")
    ));
    // The next call is back on the agent's settings.
    assert_eq!(recorded.turns[1].1.model.as_deref(), Some("sonnet"));
    assert_eq!(recorded.turns[1].1.instructions, None);
}

#[tokio::test]
async fn active_tools_filter_host_tools_and_builtins() {
    let (agent, recorded) = agent(
        MockHarness::new(&["bash", "read"]),
        HarnessAgentSettings {
            tools: vec![tool("lookup"), tool("write_note")],
            active_tools: Some(vec!["read".into(), "lookup".into(), "lookup".into()]),
            ..Default::default()
        },
    );
    let agent = agent.unwrap();
    let session = agent
        .create_session(HarnessAgentSessionOptions::new(sandbox(), "/work"))
        .await
        .unwrap();
    run(&agent, session.as_ref(), Default::default()).await;
    let recorded = recorded.lock().unwrap();
    assert_eq!(
        recorded.starts[0].builtin_tool_filter,
        Some(BuiltinToolFilter::Allow {
            tool_names: vec!["read".into()]
        })
    );
    assert_eq!(tool_names(&recorded.turns[0].1.tools), ["lookup"]);
}

#[tokio::test]
async fn inactive_tools_deny_builtins() {
    let (agent, recorded) = agent(
        MockHarness::new(&["bash", "read"]),
        HarnessAgentSettings {
            tools: vec![tool("lookup")],
            inactive_tools: Some(vec!["bash".into()]),
            ..Default::default()
        },
    );
    let agent = agent.unwrap();
    agent
        .create_session(HarnessAgentSessionOptions::new(sandbox(), "/work"))
        .await
        .unwrap();
    assert_eq!(
        recorded.lock().unwrap().starts[0].builtin_tool_filter,
        Some(BuiltinToolFilter::Deny {
            tool_names: vec!["bash".into()]
        })
    );
}

#[test]
fn rejects_settings_the_harness_cannot_honor() {
    let both = HarnessAgentSettings {
        active_tools: Some(vec![]),
        inactive_tools: Some(vec![]),
        ..Default::default()
    };
    let error = agent(MockHarness::new(&[]), both).0.unwrap_err();
    assert!(error.to_string().contains("not both"), "{error}");

    let unknown = HarnessAgentSettings {
        active_tools: Some(vec!["nope".into()]),
        ..Default::default()
    };
    let error = agent(MockHarness::new(&["bash"]), unknown).0.unwrap_err();
    assert!(error.to_string().contains("'nope'"), "{error}");

    let mut no_filtering = MockHarness::new(&["bash"]);
    no_filtering.filtering = false;
    no_filtering.approvals = false;
    let filter = HarnessAgentSettings {
        inactive_tools: Some(vec!["bash".into()]),
        ..Default::default()
    };
    let error = agent(no_filtering, filter).0.unwrap_err();
    assert!(
        error.to_string().contains("tool filtering controls"),
        "{error}"
    );

    let mut no_approvals = MockHarness::new(&["bash"]);
    no_approvals.approvals = false;
    let ask = HarnessAgentSettings {
        permission_mode: Some(PermissionMode::AllowReads),
        ..Default::default()
    };
    let error = agent(no_approvals, ask).0.unwrap_err();
    assert!(error.to_string().contains("tool approval"), "{error}");

    let header = HarnessAgentSettings {
        headers: Some(IndexMap::from([("Authorization".into(), "x".into())])),
        ..Default::default()
    };
    let error = agent(MockHarness::new(&[]), header).0.unwrap_err();
    assert!(error.to_string().contains("`authorization`"), "{error}");

    let reserved = HarnessAgentSettings {
        tools: vec![tool(QUESTION_TOOL)],
        ..Default::default()
    };
    let error = agent(MockHarness::new(&[QUESTION_TOOL]), reserved)
        .0
        .unwrap_err();
    assert!(error.to_string().contains("reserved"), "{error}");
}

#[tokio::test]
async fn sessions_carry_mode_headers_and_resume() {
    let (agent, recorded) = agent(
        MockHarness::new(&["bash"]),
        HarnessAgentSettings {
            permission_mode: Some(PermissionMode::AllowEdits),
            headers: Some(IndexMap::from([("X-Trace".into(), "1".into())])),
            ..Default::default()
        },
    );
    let agent = agent.unwrap();
    let resume = HarnessResumeState {
        harness_id: "mock".into(),
        data: json!({}),
    };
    let mut options = HarnessAgentSessionOptions::new(sandbox(), "/work");
    options.resume = Some(resume.clone());
    let error = agent.create_session(options.clone()).await.err().unwrap();
    assert!(error.to_string().contains("session_id"), "{error}");
    options.session_id = Some("s-1".into());
    agent.create_session(options).await.unwrap();
    let recorded = recorded.lock().unwrap();
    let start = &recorded.starts[0];
    assert_eq!(start.session_id, "s-1");
    assert_eq!(start.permission_mode, PermissionMode::AllowEdits);
    assert_eq!(start.resume, Some(resume));
    assert_eq!(
        start.headers,
        Some(IndexMap::from([("x-trace".to_string(), "1".to_string())]))
    );
}
