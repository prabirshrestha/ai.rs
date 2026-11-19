//! Port of `claude-code-instructions.test.ts`.
//!
//! The claude-code adapter sends `instructions` separately from the user
//! text in each `start` message.
//!
//! [rust] TS stubs `SandboxChannel` and counts its `open` and listener
//! attachment calls; these tests run the real channel over [`FakeBridge`],
//! where `open({ resume: true })` shows up as a `resume` frame and the
//! listener attachment as events buffered for the continued turn.
//! `claude-code-test-isolation.test.ts` reruns the TS suites with an
//! expired ambient Claude login to check they stay offline; the Rust suites
//! hand the harness an explicit environment and an empty temporary home, so
//! they never read the ambient login.

use std::sync::Arc;

use serde_json::{Value, json};

use super::claude_code_harness::{
    ClaudeCodeHarness, ClaudeCodeHarnessSettings, ClaudeCodeHost, create_claude_code_with_host,
};
use crate::harness::adapters::bridge_test_support::*;
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::spec::HarnessV1ContinueTurnOptions;
use crate::harness::spec::harness::{HarnessV1Session, HarnessV1StartOptions};
use crate::harness::spec::lifecycle_state::{
    HarnessV1ContinueTurnState, HarnessV1ResumeSessionState, HarnessV1TurnSettings,
};
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
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
                url: "ws://127.0.0.1:4319".into(),
                headers: None,
            },
            ..Default::default()
        },
        home: mkdtemp("claude-code-instructions-"),
    }
}

impl Fixture {
    fn harness(&self) -> ClaudeCodeHarness {
        let home = self.home.path_string();
        create_claude_code_with_host(
            ClaudeCodeHarnessSettings::default(),
            ClaudeCodeHost {
                web_socket_constructor: Some(self.bridge.constructor()),
                process_env: Some(
                    [
                        ("HOME".to_string(), home.clone()),
                        ("USERPROFILE".to_string(), home.clone()),
                        ("CLAUDE_CONFIG_DIR".to_string(), home),
                    ]
                    .into_iter()
                    .collect(),
                ),
            },
        )
        .unwrap()
    }

    fn options(&self) -> HarnessV1StartOptions {
        let mut options = start_options("s1", self.sandbox.network());
        options.session_work_dir = "/wd/claude-code-s1".into();
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

    async fn last_start(&self, count: usize) -> Value {
        wait_for(|| self.starts().len() >= count).await;
        self.starts().pop().expect("no start message was sent")
    }
}

fn resume_from(data: Value) -> HarnessV1ResumeSessionState {
    serde_json::from_value(json!({
        "type": "resume-session",
        "harnessId": "claude-code",
        "specificationVersion": "harness-v1",
        "data": data,
    }))
    .unwrap()
}

fn continue_from(data: Value) -> HarnessV1ContinueTurnState {
    serde_json::from_value(json!({
        "type": "continue-turn",
        "harnessId": "claude-code",
        "specificationVersion": "harness-v1",
        "data": data,
    }))
    .unwrap()
}

fn prompt(
    text: &str,
    instructions: Option<&str>,
) -> crate::harness::spec::harness::HarnessV1PromptTurnOptions {
    let (mut options, _) = prompt_options(text);
    options.settings.instructions = instructions.map(str::to_string);
    options
}

fn continue_options(instructions: Option<&str>, events: Events) -> HarnessV1ContinueTurnOptions {
    HarnessV1ContinueTurnOptions {
        settings: HarnessV1TurnSettings {
            model: None,
            skills: Vec::new(),
            instructions: instructions.map(str::to_string),
            tools: Vec::new(),
        },
        response_format: None,
        abort_signal: None,
        emit: Arc::new(move |part| events.lock().push(part)),
    }
}

fn resume_frames(socket: &FakeSocket) -> Vec<Value> {
    socket
        .raw_sent()
        .into_iter()
        .filter(|frame| frame["type"] == "resume")
        .map(|frame| frame["lastSeenEventId"].clone())
        .collect()
}

#[tokio::test]
async fn keeps_instructions_separate_from_every_user_message() {
    let fx = fixture();
    let session = fx.start_session(None, None).await;

    let _first = session
        .do_prompt_turn(prompt("first turn", Some(INSTRUCTIONS)))
        .await
        .unwrap();
    let start = fx.last_start(1).await;
    assert_eq!(start["prompt"], "first turn");
    assert_eq!(start["instructions"], INSTRUCTIONS);
    let mut finish = finish_frame();
    finish["startId"] = start["startId"].clone();
    fx.bridge.last_socket().emit(finish);

    let _second = session
        .do_prompt_turn(prompt("second turn", Some(INSTRUCTIONS)))
        .await
        .unwrap();
    let start = fx.last_start(2).await;
    assert_eq!(start["prompt"], "second turn");
    assert_eq!(start["instructions"], INSTRUCTIONS);
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn keeps_instructions_separate_when_resuming_a_session() {
    let fx = fixture();
    let session = fx.start_session(Some(resume_from(json!({}))), None).await;

    let _control = session
        .do_prompt_turn(prompt("resumed turn", Some(INSTRUCTIONS)))
        .await
        .unwrap();
    let start = fx.last_start(1).await;
    assert_eq!(start["prompt"], "resumed turn");
    assert_eq!(start["instructions"], INSTRUCTIONS);
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn forwards_instructions_when_rerunning_a_suspended_turn() {
    let fx = fixture();
    let session = fx.start_session(None, Some(continue_from(json!({})))).await;

    let _control = session
        .do_continue_turn(continue_options(Some(INSTRUCTIONS), Arc::default()))
        .await
        .unwrap();

    let start = fx.last_start(1).await;
    assert_eq!(start["prompt"], "Continue.");
    assert_eq!(start["instructions"], INSTRUCTIONS);
    assert_eq!(start["continue"], true);
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn reruns_a_recovered_suspended_turn_only_once() {
    let fx = fixture();
    let session = fx.start_session(None, Some(continue_from(json!({})))).await;

    let _first = session
        .do_continue_turn(continue_options(None, Arc::default()))
        .await
        .unwrap();
    fx.last_start(1).await;
    let _second = session
        .do_continue_turn(continue_options(None, Arc::default()))
        .await
        .unwrap();
    flush().await;

    let starts = fx.starts();
    assert_eq!(starts.len(), 1);
    assert_match_object(
        &starts[0],
        &json!({ "prompt": "Continue.", "continue": true }),
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn does_not_rerun_a_recovered_session_after_its_first_prompt_starts() {
    let fx = fixture();
    let session = fx.start_session(Some(resume_from(json!({}))), None).await;

    let _prompt = session
        .do_prompt_turn(prompt("Recovered prompt", None))
        .await
        .unwrap();
    fx.last_start(1).await;
    let _continued = session
        .do_continue_turn(continue_options(None, Arc::default()))
        .await
        .unwrap();
    flush().await;

    let starts = fx.starts();
    assert_eq!(starts.len(), 1);
    assert_match_object(
        &starts[0],
        &json!({ "prompt": "Recovered prompt", "continue": true }),
    );
    session.do_destroy().await.unwrap();
}

fn bridge_data() -> Value {
    json!({
        "bridge": {
            "port": 4319,
            "privateKey": create_bridge_auth_key_pair().unwrap().private_key,
            "lastSeenEventId": 7,
        },
    })
}

#[tokio::test]
async fn distinguishes_parked_session_resume_from_suspended_turn_continuation() {
    let fx = fixture();
    let _parked = fx
        .start_session(Some(resume_from(bridge_data())), None)
        .await;
    flush().await;
    // A parked session attaches without asking for a replay.
    assert!(resume_frames(&fx.bridge.socket(0)).is_empty());

    let continued = fx
        .start_session(None, Some(continue_from(bridge_data())))
        .await;
    let socket = fx.bridge.socket(1);
    wait_for(|| !resume_frames(&socket).is_empty()).await;
    assert_eq!(resume_frames(&socket), [json!(7)]);
    // Events that arrive before the continued turn wires up are buffered
    // for it.
    let mut delta = json!({ "type": "text-delta", "id": "t", "delta": "replayed" });
    delta["seq"] = json!(8);
    socket.emit(delta);
    flush().await;

    let events: Events = Arc::default();
    let _control = continued
        .do_continue_turn(continue_options(None, events.clone()))
        .await
        .unwrap();
    wait_for(|| !events.lock().is_empty()).await;
    assert_eq!(events_json(&events)[0]["delta"], "replayed");
}
