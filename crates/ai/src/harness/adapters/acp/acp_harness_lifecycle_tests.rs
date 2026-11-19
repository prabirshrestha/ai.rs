//! Port of `acp-harness.test.ts`: detach, stop, cold restore, suspension,
//! disk replay and lossy rerun.
//!
//! [rust] The TS fake channel's `nextSuspensionCursor` becomes the `seq` of
//! the last frame before the suspension, and its `openOptions` /
//! `initialLastSeenEventId` show up as the `resume` frame the real channel
//! sends.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::FutureExt as _;
use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::harness::errors::HarnessError;
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1ContinueTurnOptions, HarnessV1PromptTurnOptions, HarnessV1Session,
};
use crate::harness::spec::lifecycle_state::{HarnessV1ResumeSessionState, HarnessV1TurnSettings};
use crate::harness::spec::tools::{HarnessV1PermissionMode, HarnessV1Skill, HarnessV1ToolSpec};

use super::acp_auth::ACPAuthenticationMode;
use super::acp_harness::create_acp;
use super::acp_harness_test_support::*;
use super::v1::bridge::bridge_environment::ACP_BRIDGE_CONFIGURATION_ENV;
use super::v1::harness::ACPV1Harness;
use super::v1::settings::{
    ACPAuthentication, ACPGatewayProviderAuthentication, ACPInstructionMapping,
    ACPProviderAuthentication, ACPSessionSettings, ACPV1Settings,
};

fn skill(value: Value) -> HarnessV1Skill {
    serde_json::from_value(value).unwrap()
}

fn tool(value: Value) -> HarnessV1ToolSpec {
    serde_json::from_value(value).unwrap()
}

fn harness_with_env(
    settings: ACPV1Settings,
    bridge: &FakeBridge,
    process_env: IndexMap<String, String>,
) -> ACPV1Harness {
    let mut settings = harness_settings(settings, bridge);
    settings.process_env = Some(process_env);
    settings.client_app = Some(super::acp_auth::ACPClientApp {
        name: "ai-sdk-harness-acp".into(),
        version: "0.0.0-test".into(),
    });
    create_acp(settings).unwrap()
}

/// The `resume` frames the channel sent, as `lastSeenEventId`s.
fn resume_frames(socket: &FakeSocket) -> Vec<Value> {
    socket
        .raw_sent()
        .into_iter()
        .filter(|frame| frame["type"] == "resume")
        .map(|frame| frame["lastSeenEventId"].clone())
        .collect()
}

fn continue_options(emit: Events) -> HarnessV1ContinueTurnOptions {
    HarnessV1ContinueTurnOptions {
        settings: HarnessV1TurnSettings {
            model: None,
            skills: Vec::new(),
            instructions: None,
            tools: Vec::new(),
        },
        response_format: None,
        abort_signal: None,
        emit: Arc::new(move |part| emit.lock().push(part)),
    }
}

fn types(events: &Events) -> Vec<String> {
    events_json(events)
        .into_iter()
        .map(|event| event["type"].as_str().unwrap().to_string())
        .collect()
}

/// Runs a turn whose bridge reports `thread_id` and then finishes.
async fn establish_turn(
    session: &dyn HarnessV1Session,
    socket: &FakeSocket,
    options: HarnessV1PromptTurnOptions,
    thread_id: &str,
) {
    let before = socket.sent().len();
    let control = session.do_prompt_turn(options).await.unwrap();
    wait_for_sent(socket, before + 1).await;
    socket.emit(json!({ "type": "bridge-thread", "threadId": thread_id }));
    socket.emit(finish_frame());
    done(&*control).await.unwrap();
}

#[tokio::test]
async fn detaches_between_turns_and_attaches_without_spawning_or_reapplying_guidance_and_skills() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let skills = vec![skill(json!({
        "name": "private-context",
        "description": "Use private context.",
        "content": "Private skill body.",
    }))];
    let mut settings = agent_settings("codex-acp");
    settings.authentication = Some(ACPAuthentication {
        method_id: "api-key".into(),
        meta: None,
        client_capabilities: None,
    });
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let first_session = start(&harness, start_options("session-1", sandbox.network())).await;
    let first_channel = bridge.socket(0);
    let (mut options, _) = prompt_options("Remember the number 42.");
    options.settings.skills = skills.clone();
    options.settings.instructions = Some("Remember user-provided facts.".into());
    let first_turn = first_session.do_prompt_turn(options).await.unwrap();
    wait_for_sent(&first_channel, 1).await;
    first_channel.emit(json!({ "type": "bridge-thread", "threadId": "acp-session-1" }));
    let mut finish = finish_frame();
    finish["seq"] = json!(17);
    first_channel.emit(finish);
    done(&*first_turn).await.unwrap();

    let resume_from = first_session.do_detach().await.unwrap();
    assert!(resume_from.data["agentIdentity"].is_string());
    assert!(
        regex::Regex::new("^[a-f0-9]{64}$").unwrap().is_match(
            resume_from.data["authenticationProfile"]["digest"]
                .as_str()
                .unwrap()
        )
    );
    assert!(
        regex::Regex::new("^[A-Za-z0-9_-]+$")
            .unwrap()
            .is_match(resume_from.data["bridge"]["privateKey"].as_str().unwrap())
    );
    assert_match_object(
        &resume_from.data,
        &json!({
            "authenticationProfile": { "acpMethodId": "api-key", "providerKind": "direct" },
            "acpSessionId": "acp-session-1",
            "bridge": { "port": 4319, "lastSeenEventId": 17, "sandboxId": "sandbox-1" },
            "initialGuidanceApplied": true,
            "skillsDirectory": "/home/agent/.agents/skills",
        }),
    );
    assert!(
        !serde_json::to_string(&resume_from)
            .unwrap()
            .contains("test-key")
    );
    let write_count = sandbox.file_writes().len();

    let mut options = start_options("session-1", sandbox.network());
    options.resume_from = Some(resume_from);
    let second_session = start(&harness, options).await;
    let second_channel = bridge.socket(1);
    assert!(second_session.is_resume());
    assert_eq!(sandbox.spawns().len(), 1);
    assert_eq!(sandbox.file_writes().len(), write_count);
    flush().await;
    assert!(resume_frames(&second_channel).is_empty());

    let (mut options, _) = prompt_options("What number did I ask you to remember?");
    options.settings.skills = skills;
    options.settings.instructions = Some("Remember user-provided facts.".into());
    let second_turn = second_session.do_prompt_turn(options).await.unwrap();
    let sent = wait_for_sent(&second_channel, 1).await;
    assert_match_object(
        &sent[0],
        &json!({
            "type": "start",
            "prompt": [{ "type": "text", "text": "What number did I ask you to remember?" }],
        }),
    );
    assert!(!sent[0].to_string().contains("<session-guidance>"));
    assert!(!sent[0].to_string().contains("Private skill body."));
    second_channel.emit(finish_frame());
    done(&*second_turn).await.unwrap();
    second_session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn persists_prompt_free_cold_state_and_restores_with_fresh_gateway_credentials_before_the_next_turn()
 {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let create_harness = |api_key: &str| {
        let mut settings = agent_settings("cold-gateway-acp");
        settings.prepare_sandbox = Some({
            let prepare_calls = prepare_calls.clone();
            Arc::new(move |_| {
                prepare_calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(()) }.boxed()
            })
        });
        settings.auth = Some(ACPAuthenticationMode::AiGateway);
        settings.forward_env = Some(Vec::new());
        settings.session = Some(ACPSessionSettings {
            meta: Some(IndexMap::from([("profile".to_string(), json!("restored"))])),
        });
        settings.permission_mode_mapping = Some(permission_mode_mapping());
        settings.provider_authentication = Some(ACPProviderAuthentication {
            gateway: ACPGatewayProviderAuthentication {
                env: IndexMap::new(),
            },
        });
        let mut process_env = test_env();
        process_env.insert("AI_GATEWAY_API_KEY".into(), api_key.into());
        process_env.insert(
            "AI_GATEWAY_BASE_URL".into(),
            "https://gateway.example/custom".into(),
        );
        harness_with_env(settings, &bridge, process_env)
    };
    let skills = vec![skill(json!({
        "name": "cold-context",
        "description": "Preserve cold context.",
        "content": "Use the restored context.",
    }))];
    let harness = create_harness("gateway-key-before-stop");
    let mut options = start_options("session-1", sandbox.network());
    options.permission_mode = Some(HarnessV1PermissionMode::AllowEdits);
    let first_session = start(&harness, options).await;
    let first_channel = bridge.socket(0);
    let (mut options, _) = prompt_options("Remember the private phrase cedar-lantern.");
    options.settings.skills = skills.clone();
    options.settings.instructions = Some("Retain facts across native session restoration.".into());
    options.settings.tools = vec![tool(json!({
        "name": "firstTool",
        "description": "The first-turn tool.",
        "inputSchema": { "type": "object", "properties": { "value": { "type": "string" } } },
    }))];
    establish_turn(&*first_session, &first_channel, options, "acp-session-1").await;

    let stopped = first_session.do_stop().await.unwrap();
    let serialized_stopped = serde_json::to_string(&stopped).unwrap();
    assert_match_object(
        &stopped.data,
        &json!({
            "acpSessionId": "acp-session-1",
            "coldSession": { "version": 1, "permissionMode": "allow-edits" },
            "initialGuidanceApplied": true,
            "skillsDirectory": "/home/agent/.agents/skills",
        }),
    );
    assert_eq!(stopped.data["coldSession"]["tools"][0]["name"], "firstTool");
    assert!(stopped.data.get("bridge").is_none());
    assert!(stopped.data.get("turnStartConfig").is_none());
    for hidden in [
        "Remember the private phrase cedar-lantern.",
        "gateway-key-before-stop",
        "clientApp",
    ] {
        assert!(!serialized_stopped.contains(hidden), "{hidden}");
    }
    assert_eq!(
        first_channel.sent().last().unwrap(),
        &json!({ "type": "stop" })
    );
    assert_eq!(sandbox.stops.load(Ordering::SeqCst), 0);
    let write_count = sandbox.file_writes().len();

    let harness = Arc::new(create_harness("gateway-key-after-stop"));
    let mut options = start_options("session-1", sandbox.network());
    options.resume_from = Some(stopped);
    options.permission_mode = Some(HarnessV1PermissionMode::AllowEdits);
    let resumed = tokio::spawn({
        let harness = harness.clone();
        async move { harness.do_start(options).await }
    });
    wait_for(|| bridge.count() == 2 && bridge.socket(1).sent().len() == 1).await;
    let resumed_channel = bridge.socket(1);
    let restore_frame = resumed_channel.sent()[0].clone();
    assert_eq!(prepare_calls.load(Ordering::SeqCst), 2);
    assert_match_object(
        &restore_frame,
        &json!({
            "type": "start",
            "prompt": [],
            "permissionMode": "allow-edits",
            "turnStartConfig": { "prompt": [] },
            "recoveryMode": { "type": "cold-restore", "acpSessionId": "acp-session-1" },
        }),
    );
    assert_eq!(restore_frame["tools"][0]["name"], "firstTool");
    for hidden in [
        "gateway-key-before-stop",
        "clientApp",
        "gateway-key-after-stop",
    ] {
        assert!(!restore_frame.to_string().contains(hidden), "{hidden}");
    }
    assert!(restore_frame.get("sessionMeta").is_none());
    let spawns = sandbox.spawns();
    let configuration: Value =
        serde_json::from_str(&spawns[1].env[ACP_BRIDGE_CONFIGURATION_ENV]).unwrap();
    assert_match_object(
        &configuration,
        &json!({
            "providerAuthentication": { "type": "ai-gateway", "env": {} },
            "sessionMeta": { "profile": "restored" },
        }),
    );
    assert_eq!(
        spawns[1].env["AI_SDK_ACP_GATEWAY_API_KEY"],
        "gateway-key-after-stop"
    );
    assert_eq!(
        spawns[1].env["AI_SDK_ACP_GATEWAY_BASE_URL"],
        "https://gateway.example/custom"
    );
    assert_eq!(
        spawns[1].env["AI_SDK_ACP_CLIENT_APP_NAME"],
        "ai-sdk-harness-acp"
    );
    assert_eq!(spawns[1].env["AI_SDK_ACP_CLIENT_APP_VERSION"], "0.0.0-test");
    assert_eq!(sandbox.file_writes().len(), write_count);

    emit_cold_restoration(&resumed_channel, "resume", "acp-session-1");
    let resumed_session = resumed.await.unwrap().unwrap();
    let (mut options, events) = prompt_options("What phrase did I ask you to remember?");
    options.settings.skills = skills;
    options.settings.instructions = Some("Retain facts across native session restoration.".into());
    options.settings.tools = vec![tool(json!({
        "name": "secondTool",
        "description": "A legitimate next-turn catalog change.",
        "inputSchema": { "type": "object", "properties": {} },
    }))];
    let resumed_turn = resumed_session.do_prompt_turn(options).await.unwrap();
    let sent = wait_for_sent(&resumed_channel, 2).await;
    assert_match_object(
        &sent[1],
        &json!({
            "type": "start",
            "prompt": [{ "type": "text", "text": "What phrase did I ask you to remember?" }],
        }),
    );
    assert_eq!(sent[1]["tools"][0]["name"], "secondTool");
    assert!(!sent[1].to_string().contains("<session-guidance>"));
    resumed_channel.emit(json!({ "type": "text-start", "id": "answer" }));
    resumed_channel.emit(json!({ "type": "text-delta", "id": "answer", "delta": "cedar-lantern" }));
    resumed_channel.emit(json!({ "type": "text-end", "id": "answer" }));
    resumed_channel.emit(finish_frame());
    done(&*resumed_turn).await.unwrap();
    assert_eq!(
        types(&events),
        vec!["text-start", "text-delta", "text-end", "finish"]
    );

    let restopped = resumed_session.do_stop().await.unwrap();
    assert_match_object(
        &restopped.data,
        &json!({ "restoration": { "method": "resume" } }),
    );
    assert_eq!(
        restopped.data["coldSession"]["tools"][0]["name"],
        "secondTool"
    );
    assert!(
        !serde_json::to_string(&restopped)
            .unwrap()
            .contains("gateway-key-after-stop")
    );
    assert_eq!(sandbox.stops.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn propagates_an_unsupported_cold_restore_error_and_terminates_the_replacement_bridge() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = Arc::new(
        create_acp(harness_settings(
            agent_settings("unsupported-cold-acp"),
            &bridge,
        ))
        .unwrap(),
    );
    let first_session = start(&*harness, start_options("session-1", sandbox.network())).await;
    let (options, _) = prompt_options("Establish a native session.");
    establish_turn(&*first_session, &bridge.socket(0), options, "acp-session-1").await;
    let resume_from = first_session.do_stop().await.unwrap();

    let mut options = start_options("session-1", sandbox.network());
    options.resume_from = Some(resume_from);
    let resumed = tokio::spawn({
        let harness = harness.clone();
        async move { harness.do_start(options).await }
    });
    wait_for(|| bridge.count() == 2 && bridge.socket(1).sent().len() == 1).await;
    let replacement_channel = bridge.socket(1);
    replacement_channel.emit(json!({
        "type": "error",
        "error": {
            "name": "AI_HarnessBridgeCapabilityUnsupportedError",
            "message": "Cold ACP session restoration requires the agent to advertise sessionCapabilities.resume or loadSession; a fresh unrelated ACP session will not be created.",
        },
    }));

    let error = resumed.await.unwrap().err().unwrap();
    assert!(HarnessError::is_capability_unsupported(&*error), "{error}");
    let sent = replacement_channel.sent();
    assert_eq!(sent.len(), 2);
    assert_match_object(
        &sent[0],
        &json!({
            "type": "start",
            "recoveryMode": { "type": "cold-restore", "acpSessionId": "acp-session-1" },
        }),
    );
    assert_eq!(sent[1], json!({ "type": "destroy" }));
    assert_eq!(sandbox.record.lock().kills.len(), 2);
    assert_eq!(sandbox.stops.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn restores_cold_lifecycle_state_independently_of_the_per_turn_model() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let first_harness = create_acp(harness_settings(
        agent_settings("model-identity-acp"),
        &bridge,
    ))
    .unwrap();
    let first_session = start(
        &first_harness,
        start_options("session-1", sandbox.network()),
    )
    .await;
    let (mut options, _) = prompt_options("Establish model-scoped state.");
    options.settings.model = Some("model-before-stop".into());
    establish_turn(&*first_session, &bridge.socket(0), options, "acp-session-1").await;
    let resume_from = first_session.do_stop().await.unwrap();

    let changed_harness = Arc::new(
        create_acp(harness_settings(
            agent_settings("model-identity-acp"),
            &bridge,
        ))
        .unwrap(),
    );
    let mut options = start_options("session-1", sandbox.network());
    options.resume_from = Some(resume_from);
    let resumed = tokio::spawn({
        let harness = changed_harness.clone();
        async move { harness.do_start(options).await }
    });
    wait_for(|| bridge.count() == 2 && bridge.socket(1).sent().len() == 1).await;
    let resumed_channel = bridge.socket(1);
    emit_cold_restoration(&resumed_channel, "resume", "acp-session-1");
    let resumed_session = resumed.await.unwrap().unwrap();
    let (mut options, _) = prompt_options("Continue model-independent state.");
    options.settings.model = Some("model-after-stop".into());
    let resumed_turn = resumed_session.do_prompt_turn(options).await.unwrap();
    let sent = wait_for_sent(&resumed_channel, 2).await;
    assert_match_object(
        &sent[1],
        &json!({
            "type": "start",
            "model": "model-after-stop",
            "modelMapping": { "type": "session-config-option", "path": "model" },
        }),
    );
    resumed_channel.emit(finish_frame());
    done(&*resumed_turn).await.unwrap();
    resumed_session.do_destroy().await.unwrap();
}

fn resume_with_continuation(
    continue_from: crate::harness::spec::lifecycle_state::HarnessV1ContinueTurnState,
) -> HarnessV1ResumeSessionState {
    let mut resume_from = HarnessV1ResumeSessionState::new(
        continue_from.harness_id.clone(),
        continue_from.data.clone(),
    );
    resume_from.continue_from = Some(continue_from);
    resume_from
}

#[tokio::test]
async fn suspends_at_an_exact_cursor_and_resumes_a_nested_continuation_by_replay_only() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let first_session = start(&harness, start_options("session-1", sandbox.network())).await;
    let first_channel = bridge.socket(0);
    let (options, delivered) = prompt_options("Work for a while.");
    let first_turn = first_session.do_prompt_turn(options).await.unwrap();
    wait_for_sent(&first_channel, 1).await;
    first_channel.emit(json!({ "type": "bridge-thread", "threadId": "acp-session-1" }));
    first_channel.emit(json!({ "type": "stream-start" }));
    first_channel.emit(json!({ "type": "text-start", "id": "text-1" }));
    first_channel
        .emit(json!({ "type": "text-delta", "id": "text-1", "delta": "before", "seq": 23 }));
    wait_for(|| delivered.lock().len() == 3).await;

    let continue_from = first_session.do_suspend_turn().await.unwrap();
    done(&*first_turn).await.unwrap();
    assert_eq!(
        types(&delivered),
        vec!["stream-start", "text-start", "text-delta"]
    );
    assert_match_object(
        &continue_from.data,
        &json!({
            "acpSessionId": "acp-session-1",
            "bridge": { "lastSeenEventId": 23 },
            "initialGuidanceApplied": true,
        }),
    );

    let mut options = start_options("session-1", sandbox.network());
    options.resume_from = Some(resume_with_continuation(continue_from));
    let resumed_session = start(&harness, options).await;
    let resumed_channel = bridge.socket(1);
    assert!(resumed_session.is_resume());
    assert_eq!(sandbox.spawns().len(), 1);
    wait_for(|| !resume_frames(&resumed_channel).is_empty()).await;
    assert_eq!(resume_frames(&resumed_channel), vec![json!(23)]);

    let replayed: Events = Arc::default();
    let continued = resumed_session
        .do_continue_turn(continue_options(replayed.clone()))
        .await
        .unwrap();
    flush().await;
    assert!(resumed_channel.sent().is_empty());
    resumed_channel.emit(json!({ "type": "text-delta", "id": "text-1", "delta": " after" }));
    resumed_channel.emit(json!({ "type": "text-end", "id": "text-1" }));
    resumed_channel.emit(finish_frame());
    done(&*continued).await.unwrap();
    assert_eq!(types(&replayed), vec!["text-delta", "text-end", "finish"]);
    resumed_session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn continues_a_terminal_event_replayed_before_the_continuation_is_wired() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let first_session = start(&harness, start_options("session-1", sandbox.network())).await;
    let first_channel = bridge.socket(0);
    let (options, _) = prompt_options("Work for a while.");
    let first_turn = first_session.do_prompt_turn(options).await.unwrap();
    wait_for_sent(&first_channel, 1).await;
    first_channel.emit(json!({ "type": "bridge-thread", "threadId": "acp-session-1" }));
    flush().await;

    let continue_from = first_session.do_suspend_turn().await.unwrap();
    done(&*first_turn).await.unwrap();

    let mut options = start_options("session-1", sandbox.network());
    options.resume_from = Some(resume_with_continuation(continue_from));
    let resumed_session = start(&harness, options).await;
    let resumed_channel = bridge.socket(1);
    resumed_channel.emit(finish_frame());
    flush().await;

    let replayed: Events = Arc::default();
    let continued = resumed_session
        .do_continue_turn(continue_options(replayed.clone()))
        .await
        .unwrap();
    done(&*continued).await.unwrap();

    assert_eq!(types(&replayed), vec!["finish"]);
    resumed_session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn respawns_a_replay_only_bridge_for_a_coherent_completed_disk_tail() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let mut settings = agent_settings("codex-acp");
    settings.prepare_sandbox = Some({
        let prepare_calls = prepare_calls.clone();
        Arc::new(move |_| {
            prepare_calls.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }.boxed()
        })
    });
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let initial_session = start(&harness, start_options("session-replay", sandbox.network())).await;
    let initial_channel = bridge.socket(0);
    let (options, _) = prompt_options("Complete this while the host is disconnected.");
    let initial_turn = initial_session.do_prompt_turn(options).await.unwrap();
    wait_for_sent(&initial_channel, 1).await;
    initial_channel.emit(json!({
        "type": "bridge-thread",
        "threadId": "acp-replay-session",
        "seq": 10,
    }));
    flush().await;
    let continue_from = initial_session.do_suspend_turn().await.unwrap();
    done(&*initial_turn).await.unwrap();
    let state_dir = continue_from.data["bridge"]["stateDir"]
        .as_str()
        .unwrap()
        .to_string();
    let mut text_delta =
        json!({ "type": "text-delta", "id": "text-1", "delta": "completed offline", "seq": 11 });
    let mut finish = finish_frame();
    finish["seq"] = json!(12);
    sandbox.record.lock().files.insert(
        format!("{state_dir}/event-log.ndjson"),
        format!("{text_delta}\n{finish}\n"),
    );

    bridge
        .open_errors
        .lock()
        .push_back("bridge process exited".into());
    let mut options = start_options("session-replay", sandbox.network());
    options.continue_from = Some(continue_from);
    let recovered_session = start(&harness, options).await;
    let replay_channel = bridge.socket(1);
    let spawns = sandbox.spawns();
    assert_eq!(spawns.len(), 2);
    assert_eq!(prepare_calls.load(Ordering::SeqCst), 2);
    assert_eq!(spawns[1].env["BRIDGE_REPLAY_FROM_DISK"], "1");
    wait_for(|| !resume_frames(&replay_channel).is_empty()).await;
    assert_eq!(resume_frames(&replay_channel), vec![json!(10)]);

    let replayed: Events = Arc::default();
    let continued = recovered_session
        .do_continue_turn(continue_options(replayed.clone()))
        .await
        .unwrap();
    flush().await;
    assert!(replay_channel.sent().is_empty());
    text_delta.as_object_mut().unwrap().remove("seq");
    replay_channel.emit(text_delta);
    replay_channel.emit(finish_frame());
    done(&*continued).await.unwrap();
    assert_eq!(types(&replayed), vec!["text-delta", "finish"]);
    let (options, _) = prompt_options("Do unrelated work.");
    let error = recovered_session
        .do_prompt_turn(options)
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("disk replay only"), "{error}");
    let stopped = recovered_session.do_stop().await.unwrap();
    assert_match_object(
        &stopped.data,
        &json!({ "recovery": { "mode": "disk-replay", "reason": "completed coherent event log" } }),
    );
}

#[tokio::test]
async fn reruns_an_incomplete_turn_only_through_session_resume_with_fresh_gateway_credentials() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let create_harness = |api_key: &str| {
        let mut settings = agent_settings("codex-acp");
        settings.auth = Some(ACPAuthenticationMode::AiGateway);
        settings.instruction_mapping = Some(ACPInstructionMapping::LaunchEnvJson {
            variable: "CODEX_CONFIG".into(),
            path: vec!["developer_instructions".into()],
        });
        settings.provider_authentication = Some(ACPProviderAuthentication {
            gateway: ACPGatewayProviderAuthentication {
                env: serde_json::from_value(json!({
                    "AI_GATEWAY_API_KEY": { "$source": "gateway-api-key" },
                    "AI_GATEWAY_BASE_URL": { "$source": "gateway-base-url" },
                    "AI_GATEWAY_CLIENT": { "$source": "client-app" },
                }))
                .unwrap(),
            },
        });
        let mut process_env = test_env();
        process_env.insert("AI_GATEWAY_API_KEY".into(), api_key.into());
        process_env.insert(
            "AI_GATEWAY_BASE_URL".into(),
            "https://gateway.example.test/v1".into(),
        );
        harness_with_env(settings, &bridge, process_env)
    };
    let harness = create_harness("gateway-secret-before");
    let initial_session = start(&harness, start_options("session-rerun", sandbox.network())).await;
    let initial_channel = bridge.socket(0);
    let (mut options, _) = prompt_options("Finish this durable operation.");
    options.settings.instructions = Some("Keep working until the operation is complete.".into());
    let initial_turn = initial_session.do_prompt_turn(options).await.unwrap();
    wait_for_sent(&initial_channel, 1).await;
    initial_channel.emit(json!({
        "type": "bridge-thread",
        "threadId": "acp-rerun-session",
        "seq": 20,
    }));
    flush().await;
    let continue_from = initial_session.do_suspend_turn().await.unwrap();
    done(&*initial_turn).await.unwrap();
    let serialized = serde_json::to_string(&continue_from).unwrap();
    assert!(!serialized.contains("gateway-secret-before"));
    assert!(!serialized.contains("clientApp"));
    assert_match_object(
        &continue_from.data["turnStartConfig"]["prompt"],
        &json!([{ "type": "text", "text": "Finish this durable operation." }]),
    );
    let state_dir = continue_from.data["bridge"]["stateDir"]
        .as_str()
        .unwrap()
        .to_string();
    sandbox.record.lock().files.insert(
        format!("{state_dir}/event-log.ndjson"),
        format!(
            "{}\n",
            json!({ "type": "text-delta", "id": "text-1", "delta": "incomplete", "seq": 21 })
        ),
    );
    let harness = create_harness("gateway-secret-after");
    bridge
        .open_errors
        .lock()
        .push_back("bridge process exited".into());
    let mut options = start_options("session-rerun", sandbox.network());
    options.continue_from = Some(continue_from);
    let recovered_session = start(&harness, options).await;
    let spawns = sandbox.spawns();
    assert_eq!(spawns.len(), 2);
    assert_eq!(
        spawns[1].env["AI_SDK_ACP_GATEWAY_API_KEY"],
        "gateway-secret-after"
    );
    assert!(!spawns[1].env.contains_key("BRIDGE_REPLAY_FROM_DISK"));

    let rerun_channel = bridge.socket(1);
    let mut continue_turn = continue_options(Arc::default());
    continue_turn.settings.instructions =
        Some("Keep working until the operation is complete.".into());
    let continued = recovered_session
        .do_continue_turn(continue_turn)
        .await
        .unwrap();
    let sent = wait_for_sent(&rerun_channel, 1).await;
    assert_match_object(
        &sent[0],
        &json!({
            "type": "start",
            "prompt": [{ "type": "text", "text": "Finish this durable operation." }],
            "recoveryMode": {
                "type": "lossy-rerun",
                "acpSessionId": "acp-rerun-session",
                "reason": "event log not replayable",
            },
            "instructions": "Keep working until the operation is complete.",
            "instructionMapping": {
                "type": "launch-env-json",
                "variable": "CODEX_CONFIG",
                "path": ["developer_instructions"],
            },
        }),
    );
    assert!(!sent[0].to_string().contains("gateway-secret-before"));
    assert!(!sent[0].to_string().contains("gateway-secret-after"));
    rerun_channel.emit(finish_frame());
    done(&*continued).await.unwrap();
    let detached = recovered_session.do_detach().await.unwrap();
    assert_match_object(
        &detached.data,
        &json!({ "recovery": { "mode": "lossy-rerun", "reason": "event log not replayable" } }),
    );
}
