//! Port of `codex-instructions.test.ts`.
//!
//! The Codex adapter sends session instructions separately from user text.
//!
//! [rust] TS stubs `SandboxChannel` so `send()` records messages and
//! `open()` records its options; these tests run the real channel over
//! [`FakeBridge`], where `open({ resume: true })` shows up as a `resume`
//! frame and the stubbed `suspend()` cursor of 7 is an event the bridge
//! sends with that sequence number.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use super::codex_harness::{CodexHarness, CodexHarnessSettings, CodexHost, create_codex_with_host};
use crate::harness::adapters::bridge_test_support::*;
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::spec::HarnessV1ContinueTurnOptions;
use crate::harness::spec::harness::{
    HarnessV1PromptTurnOptions, HarnessV1Session, HarnessV1StartOptions,
};
use crate::harness::spec::lifecycle_state::{
    HarnessV1ContinueTurnState, HarnessV1ResumeSessionState, HarnessV1TurnSettings,
};
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
use crate::harness::spec::tools::{HarnessV1Skill, HarnessV1SkillFile, HarnessV1ToolSpec};
use crate::harness::utils::bridge_auth::create_bridge_auth_key_pair;

const INSTRUCTIONS: &str = "Use turbo build --concurrency=4.";

struct Fixture {
    bridge: FakeBridge,
    sandbox: FakeSandbox,
    home: TempDir,
}

fn fixture() -> Fixture {
    Fixture {
        bridge: FakeBridge::default(),
        sandbox: FakeSandbox {
            home_dir: "/home/vercel-sandbox".into(),
            bridge_port_endpoint: HarnessV1PortEndpoint {
                url: "ws://127.0.0.1:4317".into(),
                headers: None,
            },
            spawn_stdout: Some(json!({ "type": "bridge-ready", "port": 4317 }).to_string() + "\n"),
            ports: vec![4317],
            ..Default::default()
        },
        home: mkdtemp("codex-instructions-"),
    }
}

impl Fixture {
    fn harness(&self) -> CodexHarness {
        let home = self.home.path_string();
        create_codex_with_host(
            CodexHarnessSettings::default(),
            CodexHost {
                web_socket_constructor: Some(self.bridge.constructor()),
                process_env: Some(
                    [
                        ("HOME".to_string(), home.clone()),
                        ("USERPROFILE".to_string(), home),
                    ]
                    .into_iter()
                    .collect(),
                ),
            },
        )
    }

    fn options(&self) -> HarnessV1StartOptions {
        let mut options = start_options("s1", self.sandbox.network());
        options.session_work_dir = "/wd/codex-s1".into();
        options
    }

    /// `startSession({ resumeFrom, continueFrom })`.
    async fn start_session(
        &self,
        resume_from: Option<HarnessV1ResumeSessionState>,
        continue_from: Option<HarnessV1ContinueTurnState>,
    ) -> Box<dyn HarnessV1Session> {
        let mut options = self.options();
        options.resume_from = resume_from;
        options.continue_from = continue_from;
        start(&self.harness(), options).await
    }

    fn starts(&self) -> Vec<Value> {
        self.bridge
            .sent()
            .into_iter()
            .filter(|frame| frame["type"] == "start")
            .collect()
    }

    /// `waitForStart({ count })`.
    async fn wait_for_start(&self, count: usize) -> Value {
        wait_for(|| self.starts().len() >= count).await;
        let starts = self.starts();
        assert_eq!(starts.len(), count);
        starts.last().cloned().unwrap()
    }
}

/// `waitForDeferredStarts()`.
async fn wait_for_deferred_starts() {
    tokio::time::sleep(Duration::from_millis(10)).await;
}

fn resume_from(data: Value) -> HarnessV1ResumeSessionState {
    serde_json::from_value(json!({
        "type": "resume-session",
        "harnessId": "codex",
        "specificationVersion": "harness-v1",
        "data": data,
    }))
    .unwrap()
}

fn continue_from(data: Value) -> HarnessV1ContinueTurnState {
    serde_json::from_value(json!({
        "type": "continue-turn",
        "harnessId": "codex",
        "specificationVersion": "harness-v1",
        "data": data,
    }))
    .unwrap()
}

fn prompt(text: &str, instructions: Option<&str>) -> HarnessV1PromptTurnOptions {
    let (mut options, _) = prompt_options(text);
    options.settings.instructions = instructions.map(str::to_string);
    options
}

fn prompt_with_tools(
    text: &str,
    instructions: Option<&str>,
    tools: Vec<HarnessV1ToolSpec>,
) -> HarnessV1PromptTurnOptions {
    let mut options = prompt(text, instructions);
    options.settings.tools = tools;
    options
}

fn continue_options(instructions: Option<&str>) -> HarnessV1ContinueTurnOptions {
    HarnessV1ContinueTurnOptions {
        settings: HarnessV1TurnSettings {
            model: None,
            skills: Vec::new(),
            instructions: instructions.map(str::to_string),
            tools: Vec::new(),
        },
        response_format: None,
        abort_signal: None,
        emit: Arc::new(|_| {}),
    }
}

fn weather_tool(input_schema: Value) -> HarnessV1ToolSpec {
    HarnessV1ToolSpec {
        name: "get_weather".into(),
        description: Some("Get weather".into()),
        input_schema: Some(input_schema),
    }
}

fn resume_frames(socket: &FakeSocket) -> usize {
    socket
        .raw_sent()
        .into_iter()
        .filter(|frame| frame["type"] == "resume")
        .count()
}

/// Makes the bridge's event cursor 7, the cursor TS's stubbed `suspend()`
/// returns.
async fn advance_cursor_to_7(fx: &Fixture) {
    let mut event = json!({ "type": "raw", "rawValue": {} });
    event["seq"] = json!(7);
    fx.bridge.last_socket().emit(event);
    flush().await;
}

// describe('codex adapter — instructions transport')

#[tokio::test]
async fn defers_the_start_frame_until_after_prompt_control_is_returned() {
    let fx = fixture();
    let session = fx.start_session(None, None).await;

    let _control = session
        .do_prompt_turn(prompt("first turn", None))
        .await
        .unwrap();

    assert!(fx.starts().is_empty());
    fx.wait_for_start(1).await;
}

#[tokio::test]
async fn keeps_instructions_separate_from_every_user_message() {
    let fx = fixture();
    let session = fx.start_session(None, None).await;

    let _first = session
        .do_prompt_turn(prompt("first turn", Some(INSTRUCTIONS)))
        .await
        .unwrap();
    let first_start = fx.wait_for_start(1).await;
    assert_eq!(first_start["prompt"], "first turn");
    assert_eq!(first_start["instructions"], INSTRUCTIONS);

    let _second = session
        .do_prompt_turn(prompt("second turn", Some(INSTRUCTIONS)))
        .await
        .unwrap();
    let last_start = fx.wait_for_start(2).await;
    assert_eq!(last_start["prompt"], "second turn");
    assert_eq!(last_start["instructions"], INSTRUCTIONS);
}

#[tokio::test]
async fn passes_dynamic_tools_without_changing_user_messages() {
    let fx = fixture();
    let session = fx.start_session(None, None).await;
    let tools = vec![weather_tool(json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
    }))];
    let tools_json = json!([{
        "name": "get_weather",
        "description": "Get weather",
        "inputSchema": {
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        },
    }]);

    let _first = session
        .do_prompt_turn(prompt_with_tools(
            "use the weather tool",
            None,
            tools.clone(),
        ))
        .await
        .unwrap();
    let first_start = fx.wait_for_start(1).await;
    assert_eq!(first_start["prompt"], "use the weather tool");
    assert_eq!(first_start["tools"], tools_json);

    let _second = session
        .do_prompt_turn(prompt_with_tools("use it again", None, tools))
        .await
        .unwrap();
    let second_start = fx.wait_for_start(2).await;
    assert_eq!(second_start["prompt"], "use it again");
    assert_eq!(second_start["tools"], tools_json);
}

#[tokio::test]
async fn keeps_instructions_separate_when_resuming_a_session() {
    let fx = fixture();
    let session = fx
        .start_session(Some(resume_from(json!({ "threadId": "thread-abc" }))), None)
        .await;

    let _control = session
        .do_prompt_turn(prompt("resumed turn", Some(INSTRUCTIONS)))
        .await
        .unwrap();
    let start = fx.wait_for_start(1).await;
    assert_eq!(start["prompt"], "resumed turn");
    assert_eq!(start["instructions"], INSTRUCTIONS);
    assert_eq!(start["resumeThreadId"], "thread-abc");
}

#[tokio::test]
async fn starts_a_fresh_native_thread_when_resumed_turn_configuration_changes() {
    let fx = fixture();
    let first_session = fx
        .start_session(Some(resume_from(json!({ "threadId": "thread-abc" }))), None)
        .await;

    let _first = first_session
        .do_prompt_turn(prompt(
            "first turn",
            Some("Use the previous turn instructions."),
        ))
        .await
        .unwrap();
    fx.wait_for_start(1).await;
    let persisted_state = first_session.do_detach().await.unwrap();
    let resumed_session = fx.start_session(Some(persisted_state), None).await;

    let _resumed = resumed_session
        .do_prompt_turn(prompt(
            "resumed turn",
            Some("Use the current turn instructions."),
        ))
        .await
        .unwrap();

    let start = fx.wait_for_start(2).await;
    assert_eq!(start["restartThread"], true);
    assert!(start.get("resumeThreadId").is_none());
}

#[tokio::test]
async fn preserves_the_native_thread_when_persisted_tool_keys_are_reordered() {
    let fx = fixture();
    let first_session = fx
        .start_session(Some(resume_from(json!({ "threadId": "thread-abc" }))), None)
        .await;
    let tools_before = vec![weather_tool(json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
    }))];

    let _first = first_session
        .do_prompt_turn(prompt_with_tools(
            "first turn",
            Some("Use the weather tool."),
            tools_before,
        ))
        .await
        .unwrap();
    fx.wait_for_start(1).await;
    let persisted_state = first_session.do_detach().await.unwrap();
    let resumed_session = fx.start_session(Some(persisted_state), None).await;
    let tools_after = vec![weather_tool(json!({
        "required": ["city"],
        "properties": { "city": { "type": "string" } },
        "type": "object",
    }))];

    let _resumed = resumed_session
        .do_prompt_turn(prompt_with_tools(
            "resumed turn",
            Some("Use the weather tool."),
            tools_after,
        ))
        .await
        .unwrap();

    let start = fx.wait_for_start(2).await;
    assert!(start.get("restartThread").is_none());
}

#[tokio::test]
async fn preserves_the_native_thread_while_replacing_a_legacy_fingerprint() {
    let fx = fixture();
    let session = fx
        .start_session(
            Some(resume_from(json!({
                "threadId": "thread-abc",
                "turnConfigurationFingerprint": "legacy-fingerprint",
            }))),
            None,
        )
        .await;

    let _control = session
        .do_prompt_turn(prompt(
            "resumed turn",
            Some("Use the current turn instructions."),
        ))
        .await
        .unwrap();

    let start = fx.wait_for_start(1).await;
    assert!(start.get("restartThread").is_none());
    assert_eq!(start["resumeThreadId"], "thread-abc");
    let persisted_state = session.do_detach().await.unwrap();
    assert!(
        persisted_state.data["turnConfigurationFingerprint"]
            .as_str()
            .unwrap()
            .starts_with("v2:")
    );
}

#[tokio::test]
async fn forwards_instructions_when_rerunning_a_suspended_turn() {
    let fx = fixture();
    let session = fx
        .start_session(
            None,
            Some(continue_from(json!({ "threadId": "thread-abc" }))),
        )
        .await;

    let _control = session
        .do_continue_turn(continue_options(Some(INSTRUCTIONS)))
        .await
        .unwrap();

    let start = fx.wait_for_start(1).await;
    assert_eq!(start["prompt"], "Continue.");
    assert_eq!(start["instructions"], INSTRUCTIONS);
    assert_eq!(start["resumeThreadId"], "thread-abc");
}

#[tokio::test]
async fn reruns_a_recovered_suspended_turn_only_once() {
    let fx = fixture();
    let session = fx
        .start_session(
            None,
            Some(continue_from(json!({ "threadId": "thread-abc" }))),
        )
        .await;

    let _first = session
        .do_continue_turn(continue_options(None))
        .await
        .unwrap();
    fx.wait_for_start(1).await;

    let _second = session
        .do_continue_turn(continue_options(None))
        .await
        .unwrap();
    wait_for_deferred_starts().await;

    let starts = fx.starts();
    assert_eq!(starts.len(), 1);
    assert_match_object(
        &starts[0],
        &json!({ "prompt": "Continue.", "resumeThreadId": "thread-abc" }),
    );
}

#[tokio::test]
async fn does_not_rerun_a_recovered_session_after_its_first_prompt_starts() {
    let fx = fixture();
    let session = fx
        .start_session(Some(resume_from(json!({ "threadId": "thread-abc" }))), None)
        .await;

    let _prompt = session
        .do_prompt_turn(prompt("Recovered prompt", None))
        .await
        .unwrap();
    fx.wait_for_start(1).await;

    let _continued = session
        .do_continue_turn(continue_options(None))
        .await
        .unwrap();
    wait_for_deferred_starts().await;

    let starts = fx.starts();
    assert_eq!(starts.len(), 1);
    assert_match_object(
        &starts[0],
        &json!({ "prompt": "Recovered prompt", "resumeThreadId": "thread-abc" }),
    );
}

// describe('codex adapter — attach replay mode')

fn bridge_data() -> Value {
    json!({
        "threadId": "thread-abc",
        "bridge": {
            "port": 4317,
            "privateKey": create_bridge_auth_key_pair().unwrap().private_key,
            "lastSeenEventId": 7,
        },
    })
}

#[tokio::test]
async fn attaches_a_parked_session_without_replaying_old_turn_events() {
    let fx = fixture();
    let session = fx
        .start_session(Some(resume_from(bridge_data())), None)
        .await;
    flush().await;

    assert_eq!(fx.sandbox.spawns().len(), 0);
    assert_eq!(resume_frames(&fx.bridge.socket(0)), 0);

    let _control = session
        .do_prompt_turn(prompt("next user turn", None))
        .await
        .unwrap();
    let start = fx.wait_for_start(1).await;
    assert_match_object(
        &start,
        &json!({ "type": "start", "prompt": "next user turn" }),
    );
    assert!(start.get("resumeThreadId").is_none());
}

#[tokio::test]
async fn attaches_a_suspended_turn_by_requesting_replay_from_the_cursor() {
    let fx = fixture();
    let _session = fx
        .start_session(None, Some(continue_from(bridge_data())))
        .await;

    assert_eq!(fx.sandbox.spawns().len(), 0);
    let socket = fx.bridge.socket(0);
    wait_for(|| resume_frames(&socket) > 0).await;
    let resume = socket
        .raw_sent()
        .into_iter()
        .find(|frame| frame["type"] == "resume")
        .unwrap();
    assert_eq!(resume["lastSeenEventId"], 7);
}

#[tokio::test]
async fn marks_detach_as_parked_and_suspend_as_replayable() {
    let fx = fixture();
    let session = fx.start_session(None, None).await;
    advance_cursor_to_7(&fx).await;
    let detached = session.do_detach().await.unwrap();
    assert_eq!(
        serde_json::to_value(&detached).unwrap()["type"],
        "resume-session"
    );
    assert_match_object(
        &detached.data,
        &json!({ "bridge": { "lastSeenEventId": 7 } }),
    );

    let session = fx.start_session(None, None).await;
    advance_cursor_to_7(&fx).await;
    let suspended = session.do_suspend_turn().await.unwrap();
    assert_eq!(
        serde_json::to_value(&suspended).unwrap()["type"],
        "continue-turn"
    );
    assert_match_object(
        &suspended.data,
        &json!({ "bridge": { "lastSeenEventId": 7 } }),
    );
}

// describe('codex adapter — skills')

fn demo_skill(content: &str, files: Option<Vec<HarnessV1SkillFile>>) -> HarnessV1Skill {
    HarnessV1Skill {
        name: "demo".into(),
        description: "Demo skill.".into(),
        content: content.into(),
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
async fn writes_skills_to_sandbox_home_without_sending_skill_metadata() {
    let fx = fixture();
    let session = fx.start_session(None, None).await;

    let mut options = prompt("use demo", None);
    options.settings.skills = vec![demo_skill(
        "Use reference.md.",
        Some(vec![skill_file("reference.md", "# Reference")]),
    )];
    let _control = session.do_prompt_turn(options).await.unwrap();

    let runs = fx.sandbox.runs();
    assert!(!runs.contains(&"mkdir -p '/home/vercel-sandbox/.codex'".to_string()));
    assert!(runs.contains(&"mkdir -p '/home/vercel-sandbox/.agents/skills'".to_string()));
    let writes = fx.sandbox.writes();
    let skill_writes: Vec<(String, String)> = writes
        .iter()
        .filter(|(path, _)| path.contains("/demo/"))
        .cloned()
        .collect();
    let bridge_meta_write = writes
        .iter()
        .find(|(path, _)| path.ends_with("/bridge-meta.json"))
        .cloned();
    assert_eq!(
        bridge_meta_write,
        Some((
            "/home/vercel-sandbox/.ai-sdk-harness/.agent-runs/s1/bridge/bridge-meta.json".into(),
            json!({ "type": "codex", "state": "starting" }).to_string(),
        ))
    );
    for expected in [
        (
            "/home/vercel-sandbox/.agents/skills/demo/SKILL.md".to_string(),
            "---\nname: demo\ndescription: Demo skill.\n---\n\nUse reference.md.".to_string(),
        ),
        (
            "/home/vercel-sandbox/.agents/skills/demo/reference.md".to_string(),
            "# Reference".to_string(),
        ),
    ] {
        assert!(skill_writes.contains(&expected), "{skill_writes:?}");
    }
    assert_eq!(skill_writes.len(), 2);
    let spawn_env = &fx.sandbox.spawns()[0].env;
    assert!(!spawn_env.contains_key("HOME"));
    assert!(!spawn_env.contains_key("CODEX_HOME"));
    let start = fx.wait_for_start(1).await;
    assert!(start.get("skills").is_none());
    let serialized = start.to_string();
    for secret in ["Demo skill.", "Use reference.md.", "# Reference"] {
        assert!(!serialized.contains(secret));
    }
}

#[tokio::test]
async fn starts_a_fresh_native_thread_when_skills_change_on_a_resumed_session() {
    let fx = fixture();
    let session = fx
        .start_session(Some(resume_from(json!({ "threadId": "thread-abc" }))), None)
        .await;

    let mut options = prompt("use demo", None);
    options.settings.skills = vec![demo_skill("Use the current skill.", None)];
    let _control = session.do_prompt_turn(options).await.unwrap();

    let start = fx.wait_for_start(1).await;
    assert_eq!(start["restartThread"], true);
    assert!(start.get("resumeThreadId").is_none());
}

#[tokio::test]
async fn rejects_unsafe_skill_file_paths_before_writing_files() {
    let fx = fixture();
    let session = fx.start_session(None, None).await;
    let mut options = prompt("Use demo.", None);
    options.settings.skills = vec![demo_skill(
        "Use reference.md.",
        Some(vec![skill_file("../reference.md", "# Reference")]),
    )];
    let error = session.do_prompt_turn(options).await.err().unwrap();
    assert!(
        error.to_string().contains("Invalid Codex skill file path"),
        "{error}"
    );
    assert!(
        !fx.sandbox
            .writes()
            .iter()
            .any(|(path, _)| path.contains("../reference.md"))
    );
    session.do_destroy().await.unwrap();
}
