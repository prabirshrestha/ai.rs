//! Port of `claude-code-harness.test.ts`.
//!
//! [rust] TS mocks `SandboxChannel` and records what the adapter hands it.
//! These tests run the real channel over [`FakeBridge`] sockets, so bridge
//! events are dispatched as frames on the open socket and must be valid
//! stream parts (`finish` carries `finishReason` and `totalUsage`). Checks
//! TS makes on the mocked channel itself (its constructor options, the
//! pending timer count, the listener map) and on `console.warn` have no
//! Rust counterpart; the tests check the behavior those options produce
//! where it is observable. `mintBridgeToken` and `getPortUrl` are
//! deprecated and not ported.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use futures::FutureExt as _;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::claude_code_bootstrap::get_claude_code_bootstrap;
use super::claude_code_bridge_protocol::{
    ClaudeCodeEffort, ClaudeCodeSkillOverride, ClaudeCodeToolHint,
};
use super::claude_code_harness::{
    ClaudeCodeHarness, ClaudeCodeHarnessSettings, ClaudeCodeHost, create_claude_code_with_host,
};
use super::claude_code_thinking::{ClaudeCodeThinkingConfig, ClaudeCodeThinkingDisplay};
use crate::harness::adapters::bridge_test_support::*;
use crate::harness::adapters::builtin_tool_catalog::validate_json_schema;
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::errors::{AbortError, HarnessError};
use crate::harness::provider::{DataContent, SharedV4FileData};
use crate::harness::sdk::messages::{
    FilePart, FilePartData, ImagePart, ImagePartImage, TextPart, UserContent, UserContentPart,
    UserModelMessage,
};
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1Prompt, HarnessV1Session, HarnessV1StartOptions,
};
use crate::harness::spec::lifecycle_state::HarnessV1TurnSettings;
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
use crate::harness::spec::tools::{
    HarnessV1BuiltinToolUseKind, HarnessV1PermissionMode, HarnessV1Skill, HarnessV1SkillFile,
};
use crate::harness::spec::{
    HarnessV1Authentication, HarnessV1ContinueTurnOptions, HarnessV1CredentialForwarding,
};
use crate::harness::utils::sandbox_channel::WebSocket as _;
use crate::harness::utils::{SandboxChannelDebugEvent, SandboxChannelReconnectOptions};

const PLACEHOLDER: &str = r"^aisdkhc_[A-Za-z0-9_-]{43}$";

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn auth(pairs: &[(&str, &str)]) -> Option<super::ClaudeCodeAuthenticationMode> {
    Some(HarnessV1Authentication::Environment(env(pairs)))
}

fn is_abort_error(error: &(dyn std::error::Error + 'static)) -> bool {
    error.downcast_ref::<AbortError>().is_some()
        || error.to_string().to_lowercase().contains("abort")
}

/// `beforeEach`: a temporary `HOME`/`CLAUDE_CONFIG_DIR` with no
/// credentials, the fake `ws` module and the fake sandbox.
struct Fixture {
    bridge: FakeBridge,
    sandbox: FakeSandbox,
    home: TempDir,
}

fn fixture() -> Fixture {
    let sandbox = FakeSandbox {
        home_dir: "/home/vercel-sandbox".into(),
        bridge_port_endpoint: HarnessV1PortEndpoint {
            url: "ws://127.0.0.1:1".into(),
            headers: None,
        },
        ..Default::default()
    };
    Fixture {
        bridge: FakeBridge::default(),
        sandbox,
        home: mkdtemp("claude-code-harness-"),
    }
}

impl Fixture {
    fn harness(&self, settings: ClaudeCodeHarnessSettings) -> ClaudeCodeHarness {
        let home = self.home.path_string();
        create_claude_code_with_host(
            settings,
            ClaudeCodeHost {
                web_socket_constructor: Some(self.bridge.constructor()),
                process_env: Some(env(&[
                    ("HOME", &home),
                    ("USERPROFILE", &home),
                    ("CLAUDE_CONFIG_DIR", &home),
                ])),
            },
        )
        .unwrap()
    }

    fn options(&self, session_id: &str) -> HarnessV1StartOptions {
        let mut options = start_options(session_id, self.sandbox.network());
        options.session_work_dir = format!("/vercel/sandbox/claude-code-{session_id}");
        options
    }

    async fn start(
        &self,
        harness: &ClaudeCodeHarness,
        options: HarnessV1StartOptions,
    ) -> Box<dyn HarnessV1Session> {
        start(harness, options).await
    }

    fn starts(&self) -> Vec<Value> {
        self.bridge
            .sent()
            .into_iter()
            .filter(|frame| frame["type"] == "start")
            .collect()
    }

    /// `lastStart()`, once the turn's `start` reached the bridge.
    async fn last_start(&self, count: usize) -> Value {
        wait_for(|| self.starts().len() >= count).await;
        self.starts().pop().expect("no start message was sent")
    }

    /// `dispatchChannelEvent(event)` on the open bridge socket.
    fn dispatch(&self, event: Value) {
        self.bridge.last_socket().emit(event);
    }
}

fn settings() -> ClaudeCodeHarnessSettings {
    ClaudeCodeHarnessSettings::default()
}

fn finish(start_id: &Value) -> Value {
    let mut frame = finish_frame();
    frame["startId"] = start_id.clone();
    frame
}

fn continue_options(events: Events) -> HarnessV1ContinueTurnOptions {
    HarnessV1ContinueTurnOptions {
        settings: HarnessV1TurnSettings {
            model: None,
            skills: Vec::new(),
            instructions: None,
            tools: Vec::new(),
        },
        response_format: None,
        abort_signal: None,
        emit: Arc::new(move |part| events.lock().push(part)),
    }
}

fn empty_continue_from() -> crate::harness::spec::lifecycle_state::HarnessV1ContinueTurnState {
    serde_json::from_value(json!({
        "type": "continue-turn",
        "harnessId": "claude-code",
        "specificationVersion": "harness-v1",
        "data": {},
    }))
    .unwrap()
}

fn resume_from(data: Value) -> crate::harness::spec::lifecycle_state::HarnessV1ResumeSessionState {
    serde_json::from_value(json!({
        "type": "resume-session",
        "harnessId": "claude-code",
        "specificationVersion": "harness-v1",
        "data": data,
    }))
    .unwrap()
}

/// Starts a session, runs one prompt turn and returns its `start` frame.
async fn start_frame(fx: &Fixture, settings: ClaudeCodeHarnessSettings) -> Value {
    let harness = fx.harness(settings);
    let session = fx.start(&harness, fx.options("s1")).await;
    let (options, _) = prompt_options("Hello");
    let _control = session.do_prompt_turn(options).await.unwrap();
    let start = fx.last_start(1).await;
    session.do_destroy().await.unwrap();
    start
}

/// Starts a continued session, reruns its turn and returns the `start`.
async fn continued_start_frame(fx: &Fixture, settings: ClaudeCodeHarnessSettings) -> Value {
    let harness = fx.harness(settings);
    let mut options = fx.options("s1");
    options.continue_from = Some(empty_continue_from());
    let session = fx.start(&harness, options).await;
    let _control = session
        .do_continue_turn(continue_options(Arc::default()))
        .await
        .unwrap();
    let start = fx.last_start(1).await;
    session.do_destroy().await.unwrap();
    start
}

/// Starts, detaches, attaches, runs one prompt turn and returns its `start`.
async fn attached_start_frame(fx: &Fixture, settings: ClaudeCodeHarnessSettings) -> Value {
    let harness = fx.harness(settings);
    let session = fx.start(&harness, fx.options("s1")).await;
    let resume_from = session.do_detach().await.unwrap();
    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from);
    let attached = fx.start(&harness, options).await;
    let (options, _) = prompt_options("hello again");
    let _control = attached.do_prompt_turn(options).await.unwrap();
    let start = fx.last_start(1).await;
    attached.do_destroy().await.unwrap();
    start
}

#[test]
fn declares_the_harness_id_and_builtin_tools() {
    let fx = fixture();
    let harness = fx.harness(settings());
    assert_eq!(harness.harness_id(), "claude-code");
    assert_eq!(harness.specification_version(), "harness-v1");
    assert!(harness.supports_builtin_tool_approvals());
    assert!(harness.supports_builtin_tool_filtering());
    let tools = harness.builtin_tools();
    assert_eq!(
        tools.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "read",
            "write",
            "edit",
            "bash",
            "glob",
            "grep",
            "webSearch",
            "WebFetch",
            "NotebookEdit",
            "TodoWrite",
            "Agent",
            "TaskCreate",
            "TaskGet",
            "TaskUpdate",
            "TaskList",
            "TaskStop",
            "TaskOutput",
            "Monitor",
            "ListAgents",
            "ListMcpResources",
            "ListMcpResourcesTool",
            "ReadMcpResource",
            "ReadMcpResourceTool",
            "ReadMcpResourceDirTool",
            "RefreshMcpTools",
            "ExitPlanMode",
            "EnterPlanMode",
            "EnterWorktree",
            "ExitWorktree",
            "askUserQuestions",
            "Skill",
            "ToolSearch",
            "Artifact",
            "CronCreate",
            "CronDelete",
            "CronList",
            "DesignSync",
            "LSP",
            "PowerShell",
            "PushNotification",
            "RemoteTrigger",
            "ReportFindings",
            "ScheduleWakeup",
            "SendMessage",
            "SendUserFile",
            "ShareOnboardingGuide",
            "WaitForMcpServers",
            "Workflow",
        ]
    );
    assert_eq!(tools["read"].native_name.as_deref(), Some("Read"));
    assert_eq!(
        serde_json::to_value(tools["read"].common_name).unwrap(),
        json!("read")
    );
    let kind = |name: &str| tools[name].tool_use_kind;
    assert_eq!(kind("read"), Some(HarnessV1BuiltinToolUseKind::Readonly));
    assert_eq!(kind("write"), Some(HarnessV1BuiltinToolUseKind::Edit));
    assert_eq!(kind("bash"), Some(HarnessV1BuiltinToolUseKind::Bash));
    assert_eq!(kind("Skill"), Some(HarnessV1BuiltinToolUseKind::Readonly));
    assert_eq!(
        kind("ListAgents"),
        Some(HarnessV1BuiltinToolUseKind::Readonly)
    );
    assert_eq!(
        kind("ListMcpResourcesTool"),
        Some(HarnessV1BuiltinToolUseKind::Readonly)
    );
    assert_eq!(kind("PowerShell"), Some(HarnessV1BuiltinToolUseKind::Bash));
    // WebFetch has no cross-harness common equivalent — its key is the
    // native name directly, so the entry intentionally omits both
    // `nativeName` and `commonName`.
    assert!(tools.contains_key("WebFetch"));

    let list_agents = &tools["ListAgents"].tool.input_schema.json_schema;
    assert!(validate_json_schema(list_agents, &json!({})).is_ok());
    assert!(
        validate_json_schema(
            list_agents,
            &json!({ "channel": "reviewers", "q": "available" })
        )
        .is_ok()
    );
    assert!(validate_json_schema(list_agents, &json!({ "channel": "a".repeat(257) })).is_err());
    assert!(validate_json_schema(list_agents, &json!({ "unsupported": true })).is_err());
}

#[test]
fn allows_callers_to_disable_native_agent_discovery() {
    // [rust] `new HarnessAgent({ inactiveTools: ['ListAgents'] })` checks
    // that the name is a built-in tool of the harness.
    let fx = fixture();
    assert!(
        fx.harness(settings())
            .builtin_tools()
            .contains_key("ListAgents")
    );
}

#[tokio::test]
async fn throws_harness_capability_unsupported_error_when_the_network_sandbox_session_exposes_no_ports()
 {
    let mut fx = fixture();
    fx.sandbox.ports = Vec::new();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        auth: auth(&[("ANTHROPIC_API_KEY", "test-api-key")]),
        ..settings()
    });
    let error = harness.do_start(fx.options("s1")).await.err().unwrap();
    assert!(HarnessError::is_capability_unsupported(&*error), "{error}");
}

#[tokio::test]
async fn quotes_dynamic_startup_paths_in_shell_commands() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let mut options = fx.options("s1");
    options.session_id = "s1; env > /tmp/leak #".into();
    options.session_work_dir = "/vercel/sandbox/claude-code-s1; env > /tmp/workdir-leak #".into();
    let session = fx.start(&harness, options).await;

    let session_state_dir =
        "/home/vercel-sandbox/.ai-sdk-harness/.agent-runs/s1%3B%20env%20%3E%20%2Ftmp%2Fleak%20%23";
    assert!(fx.sandbox.runs().contains(&format!(
        "mkdir -p '/vercel/sandbox/claude-code-s1; env > /tmp/workdir-leak #' '{session_state_dir}/bridge'"
    )));
    assert_eq!(
        fx.sandbox
            .spawns()
            .into_iter()
            .map(|spawn| spawn.command)
            .collect::<Vec<_>>(),
        [format!(
            "node '/home/vercel-sandbox/.ai-sdk-harness/.harness-bootstrap/claude-code/bridge.mjs' --workdir '/vercel/sandbox/claude-code-s1; env > /tmp/workdir-leak #' --bridge-state-dir '{session_state_dir}/bridge'"
        )]
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn sends_the_per_turn_model_to_the_cli() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (mut options, _) = prompt_options("Hello");
    options.settings.model = Some("agent-model".into());
    let _control = session.do_prompt_turn(options).await.unwrap();

    assert_match_object(&fx.last_start(1).await, &json!({ "model": "agent-model" }));
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn sends_configured_subagent_activity_options_to_the_bridge() {
    let fx = fixture();
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            agent_progress_summaries: Some(true),
            forward_subagent_text: Some(true),
            ..settings()
        },
    )
    .await;
    assert_match_object(
        &start,
        &json!({ "agentProgressSummaries": true, "forwardSubagentText": true }),
    );
}

#[tokio::test]
async fn sets_the_client_app_for_ai_gateway_auth() {
    let fx = fixture();
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            auth: auth(&[("AI_GATEWAY_API_KEY", "gateway-key")]),
            ..settings()
        },
    )
    .await;

    assert_match_object(
        &start,
        &json!({
            "type": "start",
            "env": {
                "CLAUDE_AGENT_SDK_CLIENT_APP":
                    format!("ai-sdk-harness-claude-code/{}", super::VERSION),
            },
        }),
    );
    let spawns = fx.sandbox.spawns();
    let public_key = &spawns[0].env["BRIDGE_AUTH_PUBLIC_KEY"];
    assert!(
        regex::Regex::new(r"^[A-Za-z0-9_-]{43}$")
            .unwrap()
            .is_match(public_key)
    );
    assert!(!spawns[0].env.contains_key("BRIDGE_CHANNEL_TOKEN"));
}

#[tokio::test]
async fn sets_custom_headers_for_gateway_and_direct_auth() {
    for (credential, session_id) in [
        (("AI_GATEWAY_API_KEY", "gateway-key"), "gateway"),
        (("ANTHROPIC_API_KEY", "anthropic-key"), "direct"),
    ] {
        let fx = fixture();
        let harness = fx.harness(ClaudeCodeHarnessSettings {
            auth: auth(&[credential]),
            ..settings()
        });
        let mut options = fx.options(session_id);
        options.headers = Some(env(&[("x-tenant", "acme")]));
        let session = fx.start(&harness, options).await;
        let (options, _) = prompt_options("Hello");
        let _control = session.do_prompt_turn(options).await.unwrap();
        assert_match_object(
            &fx.last_start(1).await,
            &json!({ "env": { "ANTHROPIC_CUSTOM_HEADERS": "x-tenant: acme" } }),
        );
        session.do_destroy().await.unwrap();
    }
}

fn recording_forwarding(
    forwarded: Arc<Mutex<Vec<(String, String)>>>,
    value: impl Fn(&str, &str) -> String + Send + Sync + 'static,
) -> Option<HarnessV1CredentialForwarding> {
    Some(Arc::new(move |options| {
        forwarded.lock().push((
            options.credential.clone(),
            options.environment_variable_name.clone(),
        ));
        let value = value(&options.credential, &options.environment_variable_name);
        async move { Ok(value) }.boxed()
    }))
}

#[tokio::test]
async fn brokers_credentials_when_the_sandbox_supports_additive_request_transformations() {
    let mut fx = fixture();
    fx.sandbox.supports_transformations = true;
    let forwarded: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            auth: auth(&[
                ("ANTHROPIC_API_KEY", "anthropic-secret"),
                ("ANTHROPIC_BASE_URL", "https://anthropic.example/v1"),
            ]),
            credential_forwarding: recording_forwarding(forwarded.clone(), |_, name| {
                format!("ephemeral-{name}")
            }),
            ..settings()
        },
    )
    .await;

    assert_eq!(
        serde_json::to_value(&fx.sandbox.record.lock().transformations).unwrap(),
        json!([[{
            "match": {
                "host": "anthropic.example",
                "path": { "startsWith": "/v1" },
                "headers": [{
                    "key": { "exact": "x-api-key" },
                    "value": { "exact": "ephemeral-ANTHROPIC_API_KEY" },
                }],
            },
            "transform": { "headers": { "x-api-key": "anthropic-secret" } },
        }]])
    );
    let forwarded = forwarded.lock().clone();
    assert_eq!(forwarded.len(), 1);
    assert!(
        regex::Regex::new(PLACEHOLDER)
            .unwrap()
            .is_match(&forwarded[0].0)
    );
    assert_eq!(forwarded[0].1, "ANTHROPIC_API_KEY");
    assert_match_object(
        &start,
        &json!({ "type": "start", "env": { "ANTHROPIC_API_KEY": "ephemeral-ANTHROPIC_API_KEY" } }),
    );
    assert!(!format!("{:?}", fx.sandbox.spawns()[0].env).contains("anthropic-secret"));
}

#[tokio::test]
async fn adds_a_gateway_placeholder_when_resuming_with_a_new_authentication_mode() {
    let mut fx = fixture();
    fx.sandbox.supports_transformations = true;
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        auth: auth(&[("AI_GATEWAY_API_KEY", "current-gateway-secret")]),
        ..settings()
    });
    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from(json!({
        "sandboxCredentialEnvironment": {
            "ANTHROPIC_API_KEY": "saved-anthropic-placeholder",
        },
    })));
    let session = fx.start(&harness, options).await;
    let (options, _) = prompt_options("Continue the session.");
    let _control = session.do_prompt_turn(options).await.unwrap();

    let sandbox_env = fx.last_start(1).await["env"].clone();
    assert_eq!(
        sandbox_env["ANTHROPIC_API_KEY"],
        "saved-anthropic-placeholder"
    );
    assert!(
        regex::Regex::new(PLACEHOLDER)
            .unwrap()
            .is_match(sandbox_env["AI_GATEWAY_API_KEY"].as_str().unwrap())
    );
    assert!(!sandbox_env.to_string().contains("current-gateway-secret"));
    assert!(!format!("{:?}", fx.sandbox.spawns()[0].env).contains("current-gateway-secret"));
    let transformations =
        serde_json::to_value(&fx.sandbox.record.lock().transformations[0]).unwrap();
    assert!(transformations.as_array().unwrap().contains(&json!({
        "match": {
            "host": "ai-gateway.vercel.sh",
            "headers": [{
                "key": { "exact": "x-api-key" },
                "value": { "exact": "saved-anthropic-placeholder" },
            }],
        },
        "transform": { "headers": { "x-api-key": "current-gateway-secret" } },
    })));
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn brokers_an_explicit_claude_oauth_token_at_the_host_boundary() {
    let mut fx = fixture();
    fx.sandbox.supports_transformations = true;
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            auth: auth(&[("CLAUDE_CODE_OAUTH_TOKEN", "host-oauth-token")]),
            credential_forwarding: recording_forwarding(Arc::default(), |_, name| {
                format!("ephemeral-{name}")
            }),
            ..settings()
        },
    )
    .await;

    assert_eq!(
        serde_json::to_value(&fx.sandbox.record.lock().transformations).unwrap(),
        json!([[{
            "match": {
                "host": "api.anthropic.com",
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer ephemeral-CLAUDE_CODE_OAUTH_TOKEN" },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer host-oauth-token" } },
        }]])
    );
    assert_match_object(
        &start,
        &json!({
            "type": "start",
            "env": { "CLAUDE_CODE_OAUTH_TOKEN": "ephemeral-CLAUDE_CODE_OAUTH_TOKEN" },
        }),
    );
    assert!(!format!("{:?}", fx.sandbox.spawns()[0].env).contains("host-oauth-token"));
}

#[tokio::test]
async fn customizes_real_credentials_when_request_transformations_are_unavailable() {
    let fx = fixture();
    let forwarded: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            auth: auth(&[("ANTHROPIC_API_KEY", "anthropic-secret")]),
            credential_forwarding: recording_forwarding(forwarded.clone(), |_, _| {
                "caller-managed-credential".into()
            }),
            ..settings()
        },
    )
    .await;

    assert_eq!(
        *forwarded.lock(),
        [(
            "anthropic-secret".to_string(),
            "ANTHROPIC_API_KEY".to_string()
        )]
    );
    assert_match_object(
        &start,
        &json!({ "type": "start", "env": { "ANTHROPIC_API_KEY": "caller-managed-credential" } }),
    );
    assert!(!format!("{:?}", fx.sandbox.spawns()[0].env).contains("anthropic-secret"));
    // [rust] The fallback warning for identity forwarding goes to stderr;
    // its wording is checked by the credential brokering utility tests.
}

#[tokio::test]
async fn customizes_credentials_forwarded_through_the_claude_process_environment() {
    let fx = fixture();
    let forwarded: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            auth: auth(&[("ANTHROPIC_API_KEY", "bridge-secret")]),
            env: Some(env(&[
                ("ANTHROPIC_API_KEY", "turn-api-key"),
                ("ANTHROPIC_AUTH_TOKEN", "turn-auth-token"),
                ("NON_SECRET", "preserved"),
            ])),
            credential_forwarding: recording_forwarding(forwarded.clone(), |credential, _| {
                format!("ephemeral-{credential}")
            }),
            ..settings()
        },
    )
    .await;

    assert_eq!(
        *forwarded.lock(),
        [
            ("turn-api-key".to_string(), "ANTHROPIC_API_KEY".to_string()),
            (
                "turn-auth-token".to_string(),
                "ANTHROPIC_AUTH_TOKEN".to_string()
            ),
        ]
    );
    assert_match_object(
        &start,
        &json!({
            "type": "start",
            "env": {
                "ANTHROPIC_API_KEY": "ephemeral-turn-api-key",
                "ANTHROPIC_AUTH_TOKEN": "ephemeral-turn-auth-token",
                "NON_SECRET": "preserved",
            },
        }),
    );
}

#[tokio::test]
async fn keeps_the_bridge_private_key_on_the_host_and_reuses_it_when_attaching() {
    let fx = fixture();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        reconnect: Some(SandboxChannelReconnectOptions {
            max_elapsed_ms: Some(120_000),
            initial_delay_ms: Some(100),
            max_delay_ms: Some(5_000),
        }),
        keep_alive_ms: Some(100),
        ..settings()
    });
    let session = fx.start(&harness, fx.options("s1")).await;

    let resume_from = session.do_detach().await.unwrap();
    let private_key = resume_from.data["bridge"]["privateKey"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        regex::Regex::new(r"^[A-Za-z0-9_-]+$")
            .unwrap()
            .is_match(&private_key)
    );
    assert!(resume_from.data["bridge"].get("token").is_none());
    let recorded = format!(
        "{:?} {:?} {:?}",
        fx.sandbox.spawns(),
        fx.sandbox.writes(),
        fx.sandbox.runs()
    );
    assert!(!recorded.contains(&private_key));

    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from);
    let attached = fx.start(&harness, options).await;
    // [rust] TS checks the reconnect and keep-alive options handed to the
    // mocked channel; here the attached channel answers the same key's
    // challenge.
    assert_eq!(fx.bridge.count(), 2);
    let attached_resume_from = attached.do_detach().await.unwrap();
    assert_match_object(
        &attached_resume_from.data,
        &json!({ "bridge": { "privateKey": private_key } }),
    );
}

#[tokio::test]
async fn forwards_channel_debug_events_for_fresh_and_attached_sessions() {
    let fx = fixture();
    let debug_events: Arc<Mutex<Vec<String>>> = Arc::default();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        on_debug: Some({
            let debug_events = debug_events.clone();
            Arc::new(move |event: &SandboxChannelDebugEvent| {
                debug_events.lock().push(format!("{event:?}"));
            })
        }),
        reconnect: Some(SandboxChannelReconnectOptions {
            max_elapsed_ms: Some(5_000),
            initial_delay_ms: Some(1),
            max_delay_ms: Some(10),
        }),
        ..settings()
    });
    let session = fx.start(&harness, fx.options("s1")).await;
    // [rust] TS checks the hook handed to the mocked channel; here a dropped
    // socket makes each channel report its reconnect.
    fx.bridge.socket(0).emit_close(1006, "");
    wait_for(|| fx.bridge.count() == 2).await;
    wait_for(|| !debug_events.lock().is_empty()).await;
    let fresh_events = debug_events.lock().len();

    let resume_from = session.do_detach().await.unwrap();
    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from);
    let attached = fx.start(&harness, options).await;
    fx.bridge.last_socket().emit_close(1006, "");
    wait_for(|| debug_events.lock().len() > fresh_events).await;
    attached.do_detach().await.unwrap();
}

#[tokio::test]
async fn resumes_the_exact_conversation_after_detaching_and_attaching() {
    let fx = fixture();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        auto_mode: Some(true),
        ..settings()
    });
    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from(
        json!({ "claudeSessionId": "claude-session-1" }),
    ));
    let session = fx.start(&harness, options).await;
    let resume_from = session.do_detach().await.unwrap();

    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from);
    let attached = fx.start(&harness, options).await;
    let (options, _) = prompt_options("Continue the work.");
    let _control = attached.do_prompt_turn(options).await.unwrap();

    let start = fx.last_start(1).await;
    assert_match_object(
        &start,
        &json!({ "resumeSessionId": "claude-session-1", "autoMode": true }),
    );
    assert!(start.get("continue").is_none());
    attached.do_destroy().await.unwrap();
}

#[tokio::test]
async fn persists_the_init_session_id_when_detaching_after_a_terminal_error_without_finish() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (options, _) = prompt_options("Trigger a terminal error.");
    let control = session.do_prompt_turn(options).await.unwrap();
    let start_id = fx.last_start(1).await["startId"].clone();

    fx.dispatch(json!({
        "type": "raw",
        "startId": start_id,
        "rawValue": { "type": "system", "subtype": "init", "sessionId": "claude-session-error" },
    }));
    fx.dispatch(json!({ "type": "error", "startId": start_id, "error": "boom" }));
    let error = done(&*control).await.unwrap_err();
    assert!(error.to_string().contains("boom"), "{error}");

    let resume_from = session.do_detach().await.unwrap();
    assert_match_object(
        &resume_from.data,
        &json!({ "claudeSessionId": "claude-session-error" }),
    );

    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from);
    let resumed = fx.start(&harness, options).await;
    let (options, _) = prompt_options("Resume the exact conversation.");
    let resumed_control = resumed.do_prompt_turn(options).await.unwrap();
    let start = fx.last_start(2).await;
    assert_match_object(
        &start,
        &json!({ "resumeSessionId": "claude-session-error" }),
    );

    fx.dispatch(finish(&start["startId"]));
    done(&*resumed_control).await.unwrap();
    resumed.do_destroy().await.unwrap();
}

#[tokio::test]
async fn persists_the_init_session_id_when_suspending_an_unfinished_turn() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (options, events) = prompt_options("Keep this turn running.");
    let _control = session.do_prompt_turn(options).await.unwrap();
    let start_id = fx.last_start(1).await["startId"].clone();

    fx.dispatch(json!({
        "type": "raw",
        "startId": start_id,
        "rawValue": { "type": "system", "subtype": "init", "sessionId": "claude-session-suspended" },
    }));
    wait_for(|| !events.lock().is_empty()).await;

    let continue_from = session.do_suspend_turn().await.unwrap();
    assert_match_object(
        &continue_from.data,
        &json!({ "claudeSessionId": "claude-session-suspended" }),
    );
}

#[tokio::test]
async fn passes_port_endpoint_headers_to_fresh_retried_and_attached_web_socket_connections() {
    let fx = fixture();
    fx.bridge
        .open_errors
        .lock()
        .push_back("mock connection failure".into());
    let headers = env(&[("E2B-Traffic-Access-Token", "traffic-token")]);
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        port_endpoint: Some(HarnessV1PortEndpoint {
            url: "wss://sandbox.example/bridge?existing=value".into(),
            headers: Some(headers.clone()),
        }),
        ..settings()
    });
    let session = fx.start(&harness, fx.options("s1")).await;

    let resume_from = session.do_detach().await.unwrap();
    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from);
    let attached = fx.start(&harness, options).await;

    let calls = fx.bridge.calls.lock().clone();
    assert_eq!(calls.len(), 3);
    for (url, call_headers) in &calls {
        assert_eq!(call_headers.as_ref(), Some(&headers));
        assert!(url.contains("existing=value"));
        assert!(!url.contains("agent_bridge_token"));
    }
    for index in 0..2 {
        let sent = fx.bridge.socket(index).raw_sent();
        assert_match_object(&sent[0], &json!({ "type": "bridge-auth-proof" }));
        assert_eq!(
            sent.iter()
                .filter(|frame| frame["type"] == "bridge-auth-proof")
                .count(),
            1
        );
    }
    attached.do_detach().await.unwrap();
}

#[tokio::test]
async fn makes_a_single_attempt_when_attaching_and_falls_back_to_respawning() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;

    let resume_from = session.do_detach().await.unwrap();
    fx.bridge
        .open_errors
        .lock()
        .push_back("mock attach failure".into());
    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from);
    let resumed = fx.start(&harness, options).await;

    // [rust] The failed attach opens no socket in the fake `ws` module.
    assert_eq!(fx.bridge.calls.lock().len(), 3);
    assert_eq!(fx.bridge.count(), 2);
    assert_eq!(fx.sandbox.spawns().len(), 2);
    assert!(fx.bridge.socket(1).is_open());
    resumed.do_detach().await.unwrap();
}

#[tokio::test]
async fn does_not_set_the_client_app_for_direct_anthropic_auth() {
    let fx = fixture();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        auth: auth(&[("ANTHROPIC_API_KEY", "anthropic-key")]),
        ..settings()
    });
    let session = fx.start(&harness, fx.options("s1")).await;
    assert!(
        !fx.sandbox.spawns()[0]
            .env
            .contains_key("CLAUDE_AGENT_SDK_CLIENT_APP")
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn sends_the_thinking_configuration_and_effort_to_the_bridge() {
    let fx = fixture();
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            thinking: Some(ClaudeCodeThinkingConfig::Enabled {
                display: Some(ClaudeCodeThinkingDisplay::Omitted),
            }),
            effort: Some(ClaudeCodeEffort::Max),
            ..settings()
        },
    )
    .await;
    assert_match_object(
        &start,
        &json!({ "thinking": { "type": "enabled", "display": "omitted" }, "effort": "max" }),
    );
}

#[tokio::test]
async fn sends_auto_mode_to_the_bridge() {
    for auto_mode in [Some(true), Some(false), None] {
        let fx = fixture();
        let harness = fx.harness(ClaudeCodeHarnessSettings {
            auto_mode,
            ..settings()
        });
        let mut options = fx.options("s1");
        options.permission_mode = Some(HarnessV1PermissionMode::AllowReads);
        let session = fx.start(&harness, options).await;
        let (options, _) = prompt_options("Inspect the project.");
        let _control = session.do_prompt_turn(options).await.unwrap();
        let start = fx.last_start(1).await;
        assert_match_object(&start, &json!({ "permissionMode": "allow-reads" }));
        match auto_mode {
            None => assert!(start.get("autoMode").is_none()),
            Some(auto_mode) => assert_match_object(&start, &json!({ "autoMode": auto_mode })),
        }
        session.do_destroy().await.unwrap();
    }
}

fn directory_settings() -> ClaudeCodeHarnessSettings {
    ClaudeCodeHarnessSettings {
        additional_directories: Some(vec!["/vercel/sandbox/shared".into()]),
        strict_mcp_config: Some(true),
        max_budget_usd: Some(0.5),
        ..settings()
    }
}

fn directory_fields() -> Value {
    json!({
        "additionalDirectories": ["/vercel/sandbox/shared"],
        "strictMcpConfig": true,
        "maxBudgetUsd": 0.5,
    })
}

#[tokio::test]
async fn sends_additional_directories_strict_mcp_config_and_the_budget_limit_to_the_bridge() {
    let fx = fixture();
    assert_match_object(
        &start_frame(&fx, directory_settings()).await,
        &directory_fields(),
    );
}

#[tokio::test]
async fn sends_configured_mcp_servers_to_the_bridge() {
    let fx = fixture();
    let mcp_servers: IndexMap<String, Value> = [(
        "context7".to_string(),
        json!({ "type": "http", "url": "https://mcp.context7.com/mcp" }),
    )]
    .into_iter()
    .collect();
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            mcp_servers: Some(mcp_servers),
            ..settings()
        },
    )
    .await;
    assert_match_object(
        &start,
        &json!({ "mcpServers": { "context7": { "type": "http", "url": "https://mcp.context7.com/mcp" } } }),
    );
}

#[test]
fn rejects_the_reserved_harness_tools_mcp_server_name() {
    let fx = fixture();
    let error = create_claude_code_with_host(
        ClaudeCodeHarnessSettings {
            mcp_servers: Some(
                [("harness-tools".to_string(), json!({}))]
                    .into_iter()
                    .collect(),
            ),
            ..settings()
        },
        ClaudeCodeHost {
            web_socket_constructor: Some(fx.bridge.constructor()),
            process_env: None,
        },
    )
    .err()
    .unwrap();
    assert_eq!(
        error.to_string(),
        "Claude Code MCP server name \"harness-tools\" is reserved for HarnessAgent tools."
    );
}

fn skill_overrides(
    pairs: &[(&str, ClaudeCodeSkillOverride)],
) -> IndexMap<String, ClaudeCodeSkillOverride> {
    pairs
        .iter()
        .map(|(name, value)| (name.to_string(), *value))
        .collect()
}

fn tool_hints(read_only: bool, always_load: Option<bool>) -> IndexMap<String, ClaudeCodeToolHint> {
    [(
        "search".to_string(),
        ClaudeCodeToolHint {
            read_only: Some(read_only),
            always_load,
        },
    )]
    .into_iter()
    .collect()
}

#[tokio::test]
async fn sends_skill_overrides_and_host_tool_hints_to_the_bridge() {
    let fx = fixture();
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            skill_overrides: Some(skill_overrides(&[
                ("update-config", ClaudeCodeSkillOverride::Off),
                ("fewer-permission-prompts", ClaudeCodeSkillOverride::Off),
                ("init", ClaudeCodeSkillOverride::Off),
            ])),
            tool_hints: Some(tool_hints(true, Some(true))),
            ..settings()
        },
    )
    .await;
    assert_match_object(
        &start,
        &json!({
            "skillOverrides": {
                "update-config": "off",
                "fewer-permission-prompts": "off",
                "init": "off",
            },
            "toolHints": { "search": { "readOnly": true, "alwaysLoad": true } },
        }),
    );
}

#[tokio::test]
async fn omits_skill_overrides_when_they_are_not_configured() {
    let fx = fixture();
    assert!(
        start_frame(&fx, settings())
            .await
            .get("skillOverrides")
            .is_none()
    );
}

#[tokio::test]
async fn defaults_to_summarized_adaptive_thinking() {
    let fx = fixture();
    assert_match_object(
        &start_frame(&fx, settings()).await,
        &json!({ "thinking": { "type": "adaptive", "display": "summarized" } }),
    );
}

#[tokio::test]
async fn sends_environment_configuration_to_the_bridge() {
    let fx = fixture();
    let start = start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            env: Some(env(&[("DEPLOYMENT_ENV", "staging")])),
            ..settings()
        },
    )
    .await;
    assert_match_object(&start, &json!({ "env": { "DEPLOYMENT_ENV": "staging" } }));
}

#[tokio::test]
async fn ignores_an_aborted_turn_tail_while_the_next_turn_waits_for_its_own_answer() {
    let fx = fixture();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        auth: auth(&[("ANTHROPIC_API_KEY", "test")]),
        ..settings()
    });
    let session = fx.start(&harness, fx.options("s1")).await;
    let abort = CancellationToken::new();
    let (mut options, _) = prompt_options("Count.");
    options.abort_signal = Some(abort.clone());
    let first = session.do_prompt_turn(options).await.unwrap();
    let first_id = fx.last_start(1).await["startId"].clone();
    abort.cancel();
    let error = done(&*first).await.unwrap_err();
    assert!(is_abort_error(&*error), "{error}");

    let (options, events) = prompt_options("Answer pineapple.");
    let second = session.do_prompt_turn(options).await.unwrap();
    let second_id = fx.last_start(2).await["startId"].clone();
    assert_ne!(second_id, first_id);
    let usage = unknown_usage();
    let stale = [
        json!({ "type": "stream-start" }),
        json!({ "type": "text-start", "id": "old" }),
        json!({ "type": "text-delta", "id": "old", "delta": "two" }),
        json!({ "type": "text-end", "id": "old" }),
        json!({ "type": "reasoning-start", "id": "old-reasoning" }),
        json!({ "type": "reasoning-delta", "id": "old-reasoning", "delta": "old" }),
        json!({ "type": "reasoning-end", "id": "old-reasoning" }),
        json!({ "type": "tool-input-start", "id": "old-tool", "toolName": "bash" }),
        json!({ "type": "tool-input-delta", "id": "old-tool", "delta": "{}" }),
        json!({ "type": "tool-input-end", "id": "old-tool" }),
        json!({ "type": "tool-call", "toolCallId": "old-tool", "toolName": "bash", "input": "{}" }),
        json!({ "type": "tool-result", "toolCallId": "old-tool", "toolName": "bash", "result": "old" }),
        json!({ "type": "tool-approval-request", "toolCallId": "old-tool", "approvalId": "old-approval" }),
        json!({ "type": "compaction", "trigger": "auto", "summary": "old" }),
        json!({
            "type": "raw",
            "rawValue": { "type": "system", "subtype": "init", "sessionId": "wrong-conversation" },
        }),
        json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "end_turn" }, "usage": usage }),
        {
            let mut frame = finish_frame();
            frame["harnessMetadata"] =
                json!({ "claude-code": { "sessionId": "wrong-conversation" } });
            frame
        },
        json!({ "type": "error", "error": "old failure" }),
    ];
    for mut event in stale {
        event["startId"] = first_id.clone();
        fx.dispatch(event);
    }
    flush().await;
    assert!(events.lock().is_empty());
    assert!(second.done().now_or_never().is_none());

    for event in [
        json!({ "type": "stream-start" }),
        json!({ "type": "text-start", "id": "new" }),
        json!({ "type": "text-delta", "id": "new", "delta": "pineapple" }),
        json!({ "type": "text-end", "id": "new" }),
    ] {
        let mut event = event;
        event["startId"] = second_id.clone();
        fx.dispatch(event);
    }
    fx.dispatch(finish(&second_id));
    done(&*second).await.unwrap();
    let parts = events_json(&events);
    assert_eq!(
        parts
            .iter()
            .map(|part| part["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "stream-start",
            "text-start",
            "text-delta",
            "text-end",
            "finish"
        ]
    );
    assert_eq!(parts[2]["delta"], "pineapple");

    let (options, _) = prompt_options("Again.");
    let third = session.do_prompt_turn(options).await.unwrap();
    let start = fx.last_start(3).await;
    assert!(start.get("resumeSessionId").is_none());
    fx.dispatch(finish(&start["startId"]));
    done(&*third).await.unwrap();
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn does_not_clear_the_startup_deadline_for_events_from_another_turn() {
    let fx = fixture();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        auth: auth(&[("ANTHROPIC_API_KEY", "test")]),
        turn_start_timeout_ms: Some(100),
        ..settings()
    });
    let mut options = fx.options("s1");
    options.session_work_dir = "/w".into();
    let session = fx.start(&harness, options).await;
    let (options, _) = prompt_options("Hello");
    let control = session.do_prompt_turn(options).await.unwrap();
    fx.last_start(1).await;
    fx.dispatch(json!({ "type": "stream-start", "startId": "previous-turn" }));
    fx.dispatch(finish(&json!("previous-turn")));
    let error = done(&*control).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("did not emit a turn event within 100ms"),
        "{error}"
    );
    wait_for_frame(&fx.bridge.last_socket(), &json!({ "type": "abort" })).await;
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn does_not_start_a_bridge_turn_when_the_signal_is_already_aborted() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let abort = CancellationToken::new();
    abort.cancel();
    let (mut options, _) = prompt_options("never runs");
    options.abort_signal = Some(abort);
    let control = session.do_prompt_turn(options).await.unwrap();

    // The turn settles as the caller's own abort…
    let error = done(&*control).await.unwrap_err();
    assert!(is_abort_error(&*error), "{error}");
    // …and no `start` is sent: the bridge must not run an unattended turn
    // the caller has already observed as cancelled.
    flush().await;
    assert!(fx.starts().is_empty());
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn passes_the_host_disconnect_timeout_to_the_bridge() {
    let pattern = regex::Regex::new(r"--host-disconnect-timeout-ms '?(\d+)'?").unwrap();
    for (timeout, expected) in [(Some(60_000), Some("60000")), (None, None)] {
        let fx = fixture();
        let harness = fx.harness(ClaudeCodeHarnessSettings {
            host_disconnect_timeout_ms: timeout,
            ..settings()
        });
        let session = fx.start(&harness, fx.options("s1")).await;
        let spawns = fx.sandbox.spawns();
        assert_eq!(
            pattern
                .captures(&spawns[0].command)
                .map(|captures| captures[1].to_string())
                .as_deref(),
            expected
        );
        assert!(
            !spawns[0]
                .env
                .contains_key("BRIDGE_HOST_DISCONNECT_TIMEOUT_MS")
        );
        session.do_destroy().await.unwrap();
    }
}

#[tokio::test]
async fn sends_environment_configuration_and_tool_hints_when_rerunning_a_continued_turn() {
    let fx = fixture();
    let start = continued_start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            env: Some(env(&[("DEPLOYMENT_ENV", "staging")])),
            tool_hints: Some(tool_hints(true, None)),
            ..settings()
        },
    )
    .await;
    assert_match_object(
        &start,
        &json!({
            "env": { "DEPLOYMENT_ENV": "staging" },
            "toolHints": { "search": { "readOnly": true } },
            "continue": true,
        }),
    );
}

#[tokio::test]
async fn sends_skill_overrides_and_auto_mode_when_rerunning_a_continued_turn() {
    let fx = fixture();
    let start = continued_start_frame(
        &fx,
        ClaudeCodeHarnessSettings {
            skill_overrides: Some(skill_overrides(&[("init", ClaudeCodeSkillOverride::Off)])),
            auto_mode: Some(true),
            ..settings()
        },
    )
    .await;
    assert_match_object(
        &start,
        &json!({ "skillOverrides": { "init": "off" }, "autoMode": true, "continue": true }),
    );
}

#[tokio::test]
async fn sends_additional_directories_strict_mcp_config_and_the_budget_limit_when_rerunning_a_continued_turn()
 {
    let fx = fixture();
    let mut expected = directory_fields();
    expected["continue"] = json!(true);
    assert_match_object(
        &continued_start_frame(&fx, directory_settings()).await,
        &expected,
    );
}

#[tokio::test]
async fn sends_additional_directories_strict_mcp_config_and_the_budget_limit_after_attaching() {
    let fx = fixture();
    assert_match_object(
        &attached_start_frame(&fx, directory_settings()).await,
        &directory_fields(),
    );
}

fn question_settings() -> ClaudeCodeHarnessSettings {
    ClaudeCodeHarnessSettings {
        question_timeout_ms: Some(600_000),
        exclude_dynamic_sections: Some(true),
        ..settings()
    }
}

#[tokio::test]
async fn sends_the_question_timeout_and_exclude_dynamic_sections_to_the_bridge() {
    let fx = fixture();
    assert_match_object(
        &start_frame(&fx, question_settings()).await,
        &json!({ "questionTimeoutMs": 600_000, "excludeDynamicSections": true }),
    );
}

#[tokio::test]
async fn sends_the_question_timeout_and_exclude_dynamic_sections_when_rerunning_a_continued_turn() {
    let fx = fixture();
    assert_match_object(
        &continued_start_frame(&fx, question_settings()).await,
        &json!({ "questionTimeoutMs": 600_000, "excludeDynamicSections": true, "continue": true }),
    );
}

#[tokio::test]
async fn resumes_a_deferred_approval_when_rerunning_a_continued_turn() {
    for lifecycle in ["continueFrom", "resumeFrom"] {
        let fx = fixture();
        let continue_from = json!({
            "type": "continue-turn",
            "harnessId": "claude-code",
            "specificationVersion": "harness-v1",
            "data": {},
            "pendingToolApprovals": [{
                "approvalId": "call-1",
                "toolCallId": "call-1",
                "toolName": "bash",
                "input": "{\"command\":\"pwd\"}",
                "kind": "builtin",
                "providerExecuted": true,
                "nativeName": "Bash",
            }],
        });
        let harness = fx.harness(settings());
        let mut options = fx.options("s1");
        options.permission_mode = Some(HarnessV1PermissionMode::AllowReads);
        if lifecycle == "continueFrom" {
            options.continue_from = Some(serde_json::from_value(continue_from).unwrap());
        } else {
            options.resume_from = Some(
                serde_json::from_value(json!({
                    "type": "resume-session",
                    "harnessId": "claude-code",
                    "specificationVersion": "harness-v1",
                    "data": {},
                    "continueFrom": continue_from,
                }))
                .unwrap(),
            );
        }
        let session = fx.start(&harness, options).await;
        let _control = session
            .do_continue_turn(continue_options(Arc::default()))
            .await
            .unwrap();
        assert_match_object(
            &fx.last_start(1).await,
            &json!({ "deferredToolUseId": "call-1" }),
        );
        session.do_destroy().await.unwrap();
    }
}

#[tokio::test]
async fn sends_the_question_timeout_and_exclude_dynamic_sections_after_attaching() {
    let fx = fixture();
    assert_match_object(
        &attached_start_frame(&fx, question_settings()).await,
        &json!({ "questionTimeoutMs": 600_000, "excludeDynamicSections": true }),
    );
}

fn skill(name: &str, files: Option<Vec<HarnessV1SkillFile>>) -> HarnessV1Skill {
    HarnessV1Skill {
        name: name.into(),
        description: "unsafe".into(),
        content: "unsafe".into(),
        files,
    }
}

fn skill_file(path: &str, content: &str) -> HarnessV1SkillFile {
    HarnessV1SkillFile {
        path: path.into(),
        content: content.into(),
    }
}

#[tokio::test]
async fn writes_standard_claude_skill_files_and_enables_their_names_on_start() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (mut options, _) = prompt_options("which skills do you have available?");
    options.settings.skills = vec![
        HarnessV1Skill {
            name: "weather-forecast".into(),
            description: "Use the weather forecast tool.".into(),
            content: "Call `get_weather` before answering forecasts.".into(),
            files: Some(vec![skill_file("reference.md", "# Forecast reference")]),
        },
        HarnessV1Skill {
            name: "weather-codes".into(),
            description: "Read weather code reference.".into(),
            content: "Read `weather-codes.md` for code descriptions.".into(),
            files: None,
        },
    ];
    let _control = session.do_prompt_turn(options).await.unwrap();

    assert_match_object(
        &fx.last_start(1).await,
        &json!({
            "skills": ["weather-forecast", "weather-codes"],
            "readableSkillRoots": [
                "/home/vercel-sandbox/.claude/skills/weather-forecast",
                "/home/vercel-sandbox/.claude/skills/weather-codes",
            ],
        }),
    );

    let writes = fx.sandbox.writes();
    let skill_writes: Vec<_> = writes
        .iter()
        .filter(|(path, _)| path.contains("/weather-"))
        .collect();
    assert!(
        fx.sandbox
            .runs()
            .contains(&"mkdir -p '/home/vercel-sandbox/.claude/skills'".to_string())
    );
    assert!(writes.contains(&(
        "/home/vercel-sandbox/.ai-sdk-harness/.agent-runs/s1/bridge/bridge-meta.json".to_string(),
        json!({ "type": "claude-code", "state": "starting" }).to_string(),
    )));
    let paths: Vec<&str> = skill_writes.iter().map(|(path, _)| path.as_str()).collect();
    for expected in [
        "/home/vercel-sandbox/.claude/skills/weather-forecast/SKILL.md",
        "/home/vercel-sandbox/.claude/skills/weather-forecast/reference.md",
        "/home/vercel-sandbox/.claude/skills/weather-codes/SKILL.md",
    ] {
        assert!(paths.contains(&expected), "{paths:?}");
    }
    assert_eq!(skill_writes.len(), 3);
    assert!(
        skill_writes
            .iter()
            .find(|(path, _)| path.ends_with("/SKILL.md"))
            .unwrap()
            .1
            .contains("name: weather-")
    );
    assert_eq!(
        skill_writes
            .iter()
            .find(|(path, _)| path.ends_with("/reference.md"))
            .unwrap()
            .1,
        "# Forecast reference"
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn rejects_unsafe_skill_names_before_writing_skill_files() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (mut options, _) = prompt_options("Use the skill.");
    options.settings.skills = vec![skill("../weather", None)];
    let error = session.do_prompt_turn(options).await.err().unwrap();
    assert!(
        error.to_string().contains("Invalid Claude Code skill name"),
        "{error}"
    );
    assert!(
        !fx.sandbox
            .writes()
            .iter()
            .any(|(path, _)| path.contains("../weather"))
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn rejects_unsafe_skill_file_paths_before_writing_skill_files() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (mut options, _) = prompt_options("Use the skill.");
    options.settings.skills = vec![skill(
        "weather",
        Some(vec![skill_file("../weather-codes.md", "unsafe")]),
    )];
    let error = session.do_prompt_turn(options).await.err().unwrap();
    assert!(
        error
            .to_string()
            .contains("Invalid Claude Code skill file path"),
        "{error}"
    );
    assert!(
        !fx.sandbox
            .writes()
            .iter()
            .any(|(path, _)| path.contains("weather-codes.md"))
    );
    assert!(
        !fx.sandbox
            .runs()
            .contains(&"mkdir -p '/home/vercel-sandbox/.claude/skills'".to_string())
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn includes_bridge_startup_stdout_stderr_and_exit_code_when_ready_never_arrives() {
    let mut fx = fixture();
    fx.sandbox.spawn_stdout = Some(
        json!({ "type": "bridge-fatal", "message": "Missing --workdir argument." }).to_string()
            + "\n",
    );
    fx.sandbox.spawn_stderr = "Cannot find module @anthropic-ai/claude-agent-sdk".into();
    fx.sandbox.exit_code = 1;
    let harness = fx.harness(settings());

    let error = harness.do_start(fx.options("s1")).await.err().unwrap();
    let message = error.to_string();
    for expected in [
        "claude-code bridge exited before becoming ready.",
        "Exit code: 1.",
        "bridge-fatal",
        "Missing --workdir argument.",
        "Cannot find module @anthropic-ai/claude-agent-sdk",
    ] {
        assert!(message.contains(expected), "{message}");
    }
}

// ─── user message content ──────────────────────────────────────────────

const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn message(parts: Vec<UserContentPart>) -> HarnessV1Prompt {
    HarnessV1Prompt::Message(UserModelMessage {
        content: UserContent::Parts(parts),
        provider_options: None,
    })
}

fn text(text: &str) -> UserContentPart {
    UserContentPart::Text(TextPart {
        text: text.into(),
        provider_options: None,
    })
}

fn file(media_type: &str, data: FilePartData) -> UserContentPart {
    UserContentPart::File(FilePart {
        data,
        filename: None,
        media_type: media_type.into(),
        provider_options: None,
    })
}

/// `promptTurn(prompt)`.
async fn prompt_turn(fx: &Fixture, prompt: HarnessV1Prompt) -> Result<(), String> {
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (mut options, _) = prompt_options("");
    options.prompt = prompt;
    let result = session.do_prompt_turn(options).await;
    let outcome = match result {
        Ok(_control) => {
            fx.last_start(1).await;
            Ok(())
        }
        Err(error) => {
            assert!(HarnessError::is_capability_unsupported(&*error), "{error}");
            Err(error.to_string())
        }
    };
    session.do_destroy().await.unwrap();
    outcome
}

#[tokio::test]
async fn sends_a_text_only_user_message_as_a_string_prompt() {
    let fx = fixture();
    prompt_turn(&fx, message(vec![text("First"), text("Second")]))
        .await
        .unwrap();
    assert_eq!(fx.starts().pop().unwrap()["prompt"], "First\n\nSecond");
}

#[tokio::test]
async fn sends_inline_images_as_image_blocks() {
    let fx = fixture();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(PNG)
        .unwrap();
    prompt_turn(
        &fx,
        message(vec![
            text("What is in these images?"),
            file(
                "image/png",
                FilePartData::FileData(SharedV4FileData::Url {
                    url: format!("data:image/png;base64,{PNG}"),
                    original_url: None,
                }),
            ),
            file(
                "image",
                FilePartData::FileData(SharedV4FileData::Data {
                    data: DataContent::Bytes(bytes),
                }),
            ),
            file(
                "image/png",
                FilePartData::Content(DataContent::String(PNG.into())),
            ),
            UserContentPart::Image(ImagePart {
                image: ImagePartImage::Content(DataContent::String(PNG.into())),
                media_type: None,
                provider_options: None,
            }),
        ]),
    )
    .await
    .unwrap();

    let image = json!({ "type": "image", "mediaType": "image/png", "data": PNG });
    assert_eq!(
        fx.starts().pop().unwrap()["prompt"],
        json!([
            { "type": "text", "text": "What is in these images?" },
            image,
            image,
            image,
            image,
        ])
    );
}

#[tokio::test]
async fn rejects_unsupported_user_message_content() {
    let cases = [
        // remote image URLs
        file(
            "image/png",
            FilePartData::FileData(SharedV4FileData::Url {
                url: "https://example.com/a.png".into(),
                original_url: None,
            }),
        ),
        // provider references
        file(
            "image/png",
            FilePartData::FileData(SharedV4FileData::Reference {
                reference: [("anthropic".to_string(), "file_1".to_string())]
                    .into_iter()
                    .collect(),
            }),
        ),
        // non-image files
        file(
            "application/pdf",
            FilePartData::FileData(SharedV4FileData::Data {
                data: DataContent::String(PNG.into()),
            }),
        ),
        // data URLs that are not base64 encoded
        file(
            "image/svg+xml",
            FilePartData::Content(DataContent::String(
                "data:image/svg+xml,%3Csvg%3E%3C/svg%3E".into(),
            )),
        ),
        // images whose media type cannot be detected
        file(
            "image",
            FilePartData::FileData(SharedV4FileData::Data {
                data: DataContent::String("aGVsbG8=".into()),
            }),
        ),
    ];
    for part in cases {
        let fx = fixture();
        assert!(prompt_turn(&fx, message(vec![part])).await.is_err());
        assert!(fx.starts().is_empty());
    }
}

// ─── bridge WebSocket startup ──────────────────────────────────────────

#[tokio::test]
async fn cleans_up_a_turn_whose_initial_start_cannot_be_sent() {
    let fx = fixture();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        turn_start_timeout_ms: Some(100),
        ..settings()
    });
    let session = fx.start(&harness, fx.options("s1")).await;
    *fx.bridge.send_error.lock() = Some("channel is closed".into());
    let (options, events) = prompt_options("Hello");
    let error = session.do_prompt_turn(options).await.err().unwrap();
    assert!(error.to_string().contains("channel is closed"), "{error}");
    // [rust] The turn's listeners and startup deadline are gone: a late
    // event reaches nobody and the deadline sends no `abort`.
    *fx.bridge.send_error.lock() = None;
    fx.dispatch(json!({ "type": "text-delta", "id": "m", "delta": "late" }));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(events.lock().is_empty());
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn clears_the_startup_deadline_when_the_caller_aborts() {
    let fx = fixture();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        turn_start_timeout_ms: Some(100),
        ..settings()
    });
    let session = fx.start(&harness, fx.options("s1")).await;
    let abort = CancellationToken::new();
    let (mut options, _) = prompt_options("Hello");
    options.abort_signal = Some(abort.clone());
    let control = session.do_prompt_turn(options).await.unwrap();
    fx.last_start(1).await;
    abort.cancel();
    let error = done(&*control).await.unwrap_err();
    assert!(is_abort_error(&*error), "{error}");
    let aborts = || {
        fx.bridge
            .sent()
            .iter()
            .filter(|frame| frame["type"] == "abort")
            .count()
    };
    let after_abort = aborts();
    tokio::time::sleep(Duration::from_millis(200)).await;
    // The deadline no longer fires a second `abort`.
    assert_eq!(aborts(), after_abort);
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn fails_a_silent_turn_after_the_startup_deadline() {
    let fx = fixture();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        turn_start_timeout_ms: Some(100),
        ..settings()
    });
    let session = fx.start(&harness, fx.options("s1")).await;
    let (options, _) = prompt_options("Hello");
    let control = session.do_prompt_turn(options).await.unwrap();
    let error = done(&*control).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("did not emit a turn event within 100ms"),
        "{error}"
    );
    wait_for_frame(&fx.bridge.last_socket(), &json!({ "type": "abort" })).await;
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn clears_the_startup_deadline_on_the_first_bridge_event() {
    let fx = fixture();
    let harness = fx.harness(ClaudeCodeHarnessSettings {
        turn_start_timeout_ms: Some(100),
        ..settings()
    });
    let session = fx.start(&harness, fx.options("s1")).await;
    let (options, _) = prompt_options("Hello");
    let control = session.do_prompt_turn(options).await.unwrap();
    fx.last_start(1).await;
    fx.dispatch(json!({ "type": "text-delta", "id": "m", "delta": "Hello" }));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!fx.bridge.sent().contains(&json!({ "type": "abort" })));
    fx.dispatch(finish_frame());
    done(&*control).await.unwrap();
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn exposes_steering_when_the_bridge_advertises_acknowledged_user_messages() {
    let fx = fixture();
    *fx.bridge.hello.lock() = Some(json!({
        "type": "bridge-hello",
        "capabilities": { "experimental_userMessageResponses": true },
    }));
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (options, _) = prompt_options("Weather in Paris?");
    let control = session.do_prompt_turn(options).await.unwrap();
    assert!(control.supports_submit_user_message());
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn does_not_expose_steering_without_acknowledged_user_messages() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (options, _) = prompt_options("Weather in Paris?");
    let control = session.do_prompt_turn(options).await.unwrap();
    assert!(!control.supports_submit_user_message());
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn submits_compaction_without_requiring_an_active_acknowledged_turn() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    session
        .do_compact(Some("keep the error trace".into()))
        .await
        .unwrap();
    wait_for_frame(
        &fx.bridge.last_socket(),
        &json!({ "type": "user-message", "text": "/compact keep the error trace" }),
    )
    .await;
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn forwards_compaction_events_without_blocking_turn_completion() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (options, events) = prompt_options("Continue");
    let control = session.do_prompt_turn(options).await.unwrap();
    fx.last_start(1).await;

    fx.dispatch(json!({ "type": "compaction", "trigger": "auto", "summary": "Compacted context" }));
    fx.dispatch(finish_frame());

    done(&*control).await.unwrap();
    let parts = events_json(&events);
    assert_eq!(parts.len(), 2);
    assert_eq!(
        parts[0],
        json!({ "type": "compaction", "trigger": "auto", "summary": "Compacted context" })
    );
    assert_eq!(parts[1]["type"], "finish");
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn forwards_raw_response_boundaries_without_ending_the_turn() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let session = fx.start(&harness, fx.options("s1")).await;
    let (options, events) = prompt_options("Continue");
    let control = session.do_prompt_turn(options).await.unwrap();
    fx.last_start(1).await;
    let boundary = json!({
        "type": "raw",
        "rawValue": {
            "type": "stream_event",
            "event": { "type": "message_stop" },
            "usage": { "input_tokens": 10, "output_tokens": 2 },
        },
    });

    fx.dispatch(boundary.clone());
    wait_for(|| !events.lock().is_empty()).await;
    assert_eq!(events_json(&events), [boundary]);
    assert!(control.done().now_or_never().is_none());

    fx.dispatch(finish_frame());
    done(&*control).await.unwrap();
    session.do_destroy().await.unwrap();
}

// ─── getBootstrap ──────────────────────────────────────────────────────

#[tokio::test]
async fn returns_a_recipe_with_the_expected_harness_id_and_bootstrap_dir() {
    let fx = fixture();
    let harness = fx.harness(settings());
    assert!(harness.supports_get_bootstrap());
    let recipe = harness.get_bootstrap(None).await.unwrap();
    assert_eq!(recipe.harness_id, "claude-code");
    assert_eq!(recipe.bootstrap_dir, ".harness-bootstrap/claude-code");
}

#[tokio::test]
async fn includes_bridge_and_package_manager_assets_under_the_bootstrap_dir() {
    let fx = fixture();
    let recipe = fx.harness(settings()).get_bootstrap(None).await.unwrap();
    let mut paths: Vec<&str> = recipe.files.iter().map(|file| file.path.as_str()).collect();
    paths.sort();
    assert_eq!(
        paths,
        [
            ".harness-bootstrap/claude-code/bridge.mjs",
            ".harness-bootstrap/claude-code/package.json",
            ".harness-bootstrap/claude-code/pnpm-lock.yaml",
            ".harness-bootstrap/claude-code/pnpm-workspace.yaml",
        ]
    );
    for file in &recipe.files {
        assert!(!file.content.is_empty());
    }
}

#[tokio::test]
async fn allows_the_pinned_claude_code_build_and_verifies_the_installed_cli() {
    let fx = fixture();
    let recipe = fx.harness(settings()).get_bootstrap(None).await.unwrap();
    let commands: Vec<&str> = recipe
        .commands
        .iter()
        .map(|command| command.command.as_str())
        .collect();
    assert_eq!(
        commands,
        [
            "pnpm install --frozen-lockfile --store-dir .pnpm-store",
            "./node_modules/.bin/claude --version",
        ]
    );
    let workspace = recipe
        .files
        .iter()
        .find(|file| file.path.ends_with("/pnpm-workspace.yaml"))
        .unwrap();
    assert!(
        String::from_utf8_lossy(workspace.content.as_ref())
            .contains("'@anthropic-ai/claude-code': true")
    );
}

#[test]
fn caches_the_recipe_across_calls_and_shares_it_across_configured_harness_instances() {
    assert!(Arc::ptr_eq(
        &get_claude_code_bootstrap(),
        &get_claude_code_bootstrap()
    ));
}
