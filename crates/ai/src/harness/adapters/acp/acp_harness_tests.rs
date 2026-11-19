//! Port of `acp-harness.test.ts`: construction, startup, preparation and
//! turn basics.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::FutureExt as _;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::errors::{AbortError, HarnessError};
use crate::harness::sdk::messages::{ToolResultOutput, ToolResultPart};
use crate::harness::sdk::tool::json_schema;
use crate::harness::spec::harness::{HarnessV1, HarnessV1SubmitToolResult};
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::stream_part::HarnessV1StreamPart;
use crate::harness::spec::tools::{
    CommonToolOptions, HarnessV1BuiltinToolFiltering, HarnessV1BuiltinToolName,
    HarnessV1BuiltinToolUseKind, HarnessV1PermissionMode, common_tool,
};

use super::acp_auth::ACPAuthenticationMode;
use super::acp_harness::create_acp;
use super::acp_harness_test_support::*;
use super::v1::bridge::bridge_environment::ACP_BRIDGE_CONFIGURATION_ENV;
use super::v1::harness::serialize_builtin_tools;
use super::v1::settings::{
    ACPAskUserQuestionsSettings, ACPAuthentication, ACPAuthenticationFile,
    ACPGatewayProviderAuthentication, ACPOutputSchemaMapping, ACPPermissionModeMapping,
    ACPPermissionModeTarget, ACPProviderAuthentication,
};

fn is_abort_error(error: &(dyn std::error::Error + 'static)) -> bool {
    error.downcast_ref::<AbortError>().is_some()
        || error.to_string().to_lowercase().contains("abort")
}

#[tokio::test]
async fn awaits_sandbox_preparation_after_authentication_files_and_before_bridge_startup() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let (release, preparation) = tokio::sync::oneshot::channel::<()>();
    let preparation = preparation.shared();
    let prepare_calls: Arc<Mutex<Vec<(String, bool)>>> = Arc::default();
    let mut settings = agent_settings("prepared-acp");
    settings.authentication_files = Some(Arc::new(|_| {
        vec![ACPAuthenticationFile {
            path: ".agent/auth.json".into(),
            content: "private-auth".into(),
        }]
    }));
    settings.prepare_sandbox = Some({
        let prepare_calls = prepare_calls.clone();
        Arc::new(move |options| {
            prepare_calls
                .lock()
                .push((options.work_dir.clone(), options.abort_signal.is_some()));
            let preparation = preparation.clone();
            async move {
                let _ = preparation.await;
                Ok(())
            }
            .boxed()
        })
    });
    let harness = Arc::new(create_acp(harness_settings(settings, &bridge)).unwrap());
    let controller = CancellationToken::new();
    let mut options = start_options("session-prepared", sandbox.network());
    options.abort_signal = Some(controller.clone());

    let startup = tokio::spawn({
        let harness = harness.clone();
        async move { harness.do_start(options).await }
    });
    wait_for(|| prepare_calls.lock().len() == 1).await;

    assert_eq!(
        prepare_calls.lock()[0],
        ("/workspace/user-project".to_string(), true)
    );
    assert!(sandbox.writes().contains(&(
        "/home/agent/.agent/auth.json".to_string(),
        "private-auth".to_string()
    )));
    assert!(sandbox.spawns().is_empty());
    assert_eq!(sandbox.bridge_starting_marks(), 0);

    release.send(()).unwrap();
    let session = startup.await.unwrap().unwrap();
    assert_eq!(sandbox.spawns().len(), 1);
    assert_eq!(sandbox.bridge_starting_marks(), 1);
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn supports_synchronous_preparation_and_skips_it_when_reattaching_to_a_live_bridge() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let mut settings = agent_settings("prepared-acp");
    settings.prepare_sandbox = Some({
        let prepare_calls = prepare_calls.clone();
        Arc::new(move |_| {
            prepare_calls.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }.boxed()
        })
    });
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let session = start(
        &harness,
        start_options("session-prepared", sandbox.network()),
    )
    .await;
    let resume_from = session.do_detach().await.unwrap();
    let mut options = start_options("session-prepared", sandbox.network());
    options.resume_from = Some(resume_from);
    let resumed = start(&harness, options).await;

    assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
    resumed.do_destroy().await.unwrap();
}

#[tokio::test]
async fn passes_host_disconnect_timeout_ms_to_the_bridge() {
    let pattern = regex::Regex::new(r"--host-disconnect-timeout-ms '?(\d+)'?").unwrap();
    for (host_disconnect_timeout_ms, expected) in [(Some(60_000), Some("60000")), (None, None)] {
        let bridge = FakeBridge::default();
        let sandbox = FakeSandbox::default();
        let mut settings = harness_settings(agent_settings("timeout-acp"), &bridge);
        settings.host_disconnect_timeout_ms = host_disconnect_timeout_ms;
        let harness = create_acp(settings).unwrap();
        let session = start(
            &harness,
            start_options("session-timeout", sandbox.network()),
        )
        .await;

        let spawns = sandbox.spawns();
        let actual = pattern
            .captures(&spawns[0].command)
            .map(|captures| captures[1].to_string());
        assert_eq!(actual.as_deref(), expected);
        session.do_destroy().await.unwrap();
    }
}

#[tokio::test]
async fn does_not_start_a_bridge_when_sandbox_preparation_fails() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("prepared-acp");
    settings.prepare_sandbox = Some(Arc::new(|_| {
        async { Err(crate::harness::utils::plain_error("Preparation failed.")) }.boxed()
    }));
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();

    let error = harness
        .do_start(start_options("session-prepared", sandbox.network()))
        .await
        .err()
        .unwrap();
    assert_eq!(error.to_string(), "Preparation failed.");
    assert!(sandbox.spawns().is_empty());
    assert_eq!(sandbox.bridge_starting_marks(), 0);
}

#[tokio::test]
async fn does_not_start_a_bridge_when_startup_is_cancelled_before_or_during_preparation() {
    for before in [true, false] {
        let bridge = FakeBridge::default();
        let sandbox = FakeSandbox::default();
        let controller = CancellationToken::new();
        let prepare_calls = Arc::new(AtomicUsize::new(0));
        let mut settings = agent_settings("prepared-acp");
        settings.prepare_sandbox = Some({
            let prepare_calls = prepare_calls.clone();
            let controller = controller.clone();
            Arc::new(move |_| {
                prepare_calls.fetch_add(1, Ordering::SeqCst);
                controller.cancel();
                async { Ok(()) }.boxed()
            })
        });
        if before {
            controller.cancel();
        }
        let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
        let mut options = start_options("session-prepared", sandbox.network());
        options.abort_signal = Some(controller);

        let error = harness.do_start(options).await.err().unwrap();
        assert!(is_abort_error(&*error), "{error}");
        assert_eq!(
            prepare_calls.load(Ordering::SeqCst),
            if before { 0 } else { 1 }
        );
        assert!(sandbox.spawns().is_empty());
        assert_eq!(sandbox.bridge_starting_marks(), 0);
    }
}

/// The TS `toolResult` object, with its `type` tag.
fn tagged_tool_result(tool_result: &ToolResultPart) -> Value {
    let mut value = json!({ "type": "tool-result" });
    value.as_object_mut().unwrap().extend(
        serde_json::to_value(tool_result)
            .unwrap()
            .as_object()
            .unwrap()
            .clone(),
    );
    value
}

#[tokio::test]
async fn translates_acp_question_requests_and_client_results() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let from_native_calls: Arc<Mutex<Vec<(Value, bool)>>> = Arc::default();
    let to_native_calls: Arc<Mutex<Vec<(Value, Value)>>> = Arc::default();
    let question_input = json!({
        "allowPartialAnswers": false,
        "questions": [{ "id": "q1", "question": "Framework?" }],
    })
    .to_string();
    let mut settings = agent_settings("test-acp");
    settings.ask_user_questions = Some(ACPAskUserQuestionsSettings {
        request_method: "test/ask".into(),
        is_native_tool_call: None,
        from_native_request: {
            let from_native_calls = from_native_calls.clone();
            let question_input = question_input.clone();
            Arc::new(move |options| {
                from_native_calls.lock().push((
                    options.native_request.clone(),
                    options.native_tool_call.is_some(),
                ));
                Ok(Some(
                    serde_json::from_value(json!({
                        "type": "tool-call",
                        "toolCallId": "question-1",
                        "toolName": "askUserQuestions",
                        "input": question_input,
                        "providerExecuted": false,
                        "providerMetadata": { "test": { "preserved": true } },
                    }))
                    .unwrap(),
                ))
            })
        },
        to_native_response: {
            let to_native_calls = to_native_calls.clone();
            Arc::new(move |options| {
                let tool_result = tagged_tool_result(options.tool_result);
                to_native_calls
                    .lock()
                    .push((options.native_request.clone(), tool_result.clone()));
                Ok(json!({ "native": tool_result }))
            })
        },
        matches_native_request: None,
    });
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    assert!(harness.builtin_tools().contains_key("askUserQuestions"));
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let (options, events) = prompt_options("Ask.");
    let control = session.do_prompt_turn(options).await.unwrap();
    let channel = bridge.socket(0);
    let native_request = json!({
        "sessionId": "native-session",
        "toolCallId": "native-question-1",
    });

    channel.emit(json!({
        "type": "acp-question-request",
        "requestId": "request-1",
        "nativeRequest": native_request,
    }));
    let sent = wait_for_sent(&channel, 2).await;

    assert_eq!(
        *from_native_calls.lock(),
        vec![(native_request.clone(), false)]
    );
    assert!(events_json(&events).contains(&json!({
        "type": "tool-call",
        "toolCallId": "question-1",
        "toolName": "askUserQuestions",
        "input": question_input,
        "providerExecuted": false,
        "providerMetadata": {
            "test": { "preserved": true },
            "test-acp": { "nativeRequest": native_request },
        },
    })));
    assert!(sent.contains(&json!({
        "type": "tool-result",
        "toolCallId": "request-1",
        "output": { "type": "handled", "toolCallId": "question-1" },
    })));

    let tool_result: ToolResultPart = serde_json::from_value(json!({
        "type": "tool-result",
        "toolCallId": "question-1",
        "toolName": "askUserQuestions",
        "output": {
            "type": "json",
            "value": { "action": "answered", "answers": { "q1": { "optionIds": [] } } },
        },
    }))
    .unwrap();
    let ToolResultOutput::Json { value, .. } = &tool_result.output else {
        unreachable!()
    };
    control
        .submit_tool_result(HarnessV1SubmitToolResult {
            tool_call_id: "question-1".into(),
            output: value.clone(),
            is_error: None,
            tool_result: Some(tool_result.clone()),
        })
        .await
        .unwrap();

    let tool_result_json = tagged_tool_result(&tool_result);
    assert_eq!(
        *to_native_calls.lock(),
        vec![(native_request, tool_result_json.clone())]
    );
    let sent = wait_for_sent(&channel, 3).await;
    assert!(sent.contains(&json!({
        "type": "tool-result",
        "toolCallId": "question-1",
        "output": { "native": tool_result_json },
        "toolResult": tool_result_json,
    })));
}

#[tokio::test]
async fn constructs_the_default_v1_harness_synchronously_without_reading_assets() {
    let bridge = FakeBridge::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();

    assert_eq!(harness.harness_id(), "codex-acp");
    assert!(harness.builtin_tools().is_empty());
    assert!(harness.supports_builtin_tool_approvals());
    assert!(!harness.supports_builtin_tool_filtering());
}

#[tokio::test]
async fn accepts_explicit_v1_selection() {
    let bridge = FakeBridge::default();
    let mut settings = agent_settings("codex-acp");
    settings.version = Some("v1".into());
    assert!(create_acp(harness_settings(settings, &bridge)).is_ok());
}

#[tokio::test]
async fn maps_the_harness_agent_model_into_per_turn_session_settings() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(
        agent_settings("model-mapping-acp"),
        &bridge,
    ))
    .unwrap();

    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let (mut options, _) = prompt_options("Hello");
    options.settings.model = Some("agent-model".into());
    let control = session.do_prompt_turn(options).await.unwrap();

    let sent = wait_for_sent(&bridge.socket(0), 1).await;
    assert_match_object(
        &sent[0],
        &json!({
            "type": "start",
            "model": "agent-model",
            "modelMapping": { "type": "session-config-option", "path": "model" },
        }),
    );
    bridge.socket(0).emit(finish_frame());
    done(&*control).await.unwrap();

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn sends_the_model_mapping_without_a_model_so_the_bridge_can_report_the_agent_default() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(
        agent_settings("model-mapping-acp"),
        &bridge,
    ))
    .unwrap();

    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let (options, _) = prompt_options("Hello");
    let control = session.do_prompt_turn(options).await.unwrap();

    let sent = wait_for_sent(&bridge.socket(0), 1).await;
    assert_match_object(
        &sent[0],
        &json!({
            "type": "start",
            "modelMapping": { "type": "session-config-option", "path": "model" },
        }),
    );
    assert!(sent[0].get("model").is_none());
    bridge.socket(0).emit(finish_frame());
    done(&*control).await.unwrap();

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn requires_credential_environment_and_brokering_settings_together() {
    let bridge = FakeBridge::default();
    let mut settings = agent_settings("codex-acp");
    settings.credential_env = Some(vec!["CODEX_API_KEY".into()]);
    let error = create_acp(harness_settings(settings, &bridge))
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("ACP credentialEnv and credentialBrokering must be configured together.")
    );

    let mut settings = agent_settings("codex-acp");
    settings.credential_brokering = Some(Arc::new(|_| Ok(Vec::new())));
    let error = create_acp(harness_settings(settings, &bridge))
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("ACP credentialEnv and credentialBrokering must be configured together.")
    );
}

#[tokio::test]
async fn reserves_the_host_tool_mcp_server_name() {
    let bridge = FakeBridge::default();
    let mut settings = agent_settings("codex-acp");
    settings.mcp_servers = Some(IndexMap::from([(
        "ai-sdk-harness-tools".to_string(),
        json!({}),
    )]));
    let error = create_acp(harness_settings(settings, &bridge))
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("reserved for HarnessAgent tools")
    );
}

#[tokio::test]
async fn advertises_approvals_with_and_without_a_complete_mapping() {
    let bridge = FakeBridge::default();
    let mut mapped = agent_settings("codex-acp");
    mapped.permission_mode_mapping = Some(permission_mode_mapping());
    let mapped = create_acp(harness_settings(mapped, &bridge)).unwrap();
    let mut incomplete = agent_settings("codex-acp-incomplete");
    incomplete.permission_mode_mapping = Some(ACPPermissionModeMapping {
        allow_reads: None,
        allow_edits: None,
        allow_all: permission_mode_mapping().allow_all,
    });
    let incomplete = create_acp(harness_settings(incomplete, &bridge)).unwrap();

    assert!(mapped.supports_builtin_tool_approvals());
    assert!(!mapped.supports_builtin_tool_filtering());
    assert!(incomplete.supports_builtin_tool_approvals());
    assert!(!incomplete.supports_builtin_tool_filtering());
}

#[tokio::test]
async fn supports_restrictive_permission_modes_without_a_mapping() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("grok-build-acp"), &bridge)).unwrap();
    let mut options = start_options("session-1", sandbox.network());
    options.permission_mode = Some(HarnessV1PermissionMode::AllowReads);
    let session = start(&harness, options).await;
    let (options, _) = prompt_options("Check permissions.");
    let control = session.do_prompt_turn(options).await.unwrap();
    let channel = bridge.socket(0);

    let sent = wait_for_sent(&channel, 1).await;
    assert_match_object(
        &sent[0],
        &json!({ "type": "start", "permissionMode": "allow-reads" }),
    );
    assert!(sent[0].get("permissionModeMapping").is_none());
    channel.emit(finish_frame());
    done(&*control).await.unwrap();
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn transports_explicit_unsupported_permission_mode_mappings() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let unsupported_mapping = ACPPermissionModeMapping {
        allow_reads: None,
        allow_edits: None,
        allow_all: Some(ACPPermissionModeTarget::SessionMode {
            mode_id: "agent-full-access".into(),
        }),
    };
    let mut settings = agent_settings("codex-acp");
    settings.permission_mode_mapping = Some(unsupported_mapping);
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let mut options = start_options("session-1", sandbox.network());
    options.permission_mode = Some(HarnessV1PermissionMode::AllowAll);
    let session = start(&harness, options).await;
    let (options, _) = prompt_options("Check permissions.");
    let control = session.do_prompt_turn(options).await.unwrap();
    let channel = bridge.socket(0);

    let sent = wait_for_sent(&channel, 1).await;
    assert_match_object(
        &sent[0],
        &json!({
            "type": "start",
            "permissionModeMapping": {
                "allow-reads": null,
                "allow-edits": null,
                "allow-all": { "type": "session-mode", "modeId": "agent-full-access" },
            },
        }),
    );
    channel.emit(finish_frame());
    done(&*control).await.unwrap();
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn does_not_claim_native_filtering_when_approval_mapping_is_complete() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("codex-acp");
    settings.permission_mode_mapping = Some(permission_mode_mapping());
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let mut options = start_options("session-1", sandbox.network());
    options.builtin_tool_filtering = Some(
        serde_json::from_value::<HarnessV1BuiltinToolFiltering>(
            json!({ "mode": "deny", "toolNames": ["bash"] }),
        )
        .unwrap(),
    );

    let error = harness.do_start(options).await.err().unwrap();
    assert!(
        error
            .to_string()
            .contains("built-in tool filtering is not available")
    );
    assert_eq!(bridge.count(), 0);
}

#[tokio::test]
async fn rejects_unsupported_versions_clearly() {
    let bridge = FakeBridge::default();
    let mut settings = agent_settings("codex-acp");
    settings.version = Some("v2".into());
    let error = create_acp(harness_settings(settings, &bridge))
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("Unsupported ACP protocol version \"v2\"")
    );
}

#[tokio::test]
async fn rejects_unstable_harness_ids() {
    let bridge = FakeBridge::default();
    for harness_id in ["CodexACP", "codex_acp", "codex/acp", "codex--acp", ""] {
        let error = create_acp(harness_settings(agent_settings(harness_id), &bridge))
            .err()
            .unwrap();
        assert!(
            error.to_string().contains("stable kebab-case identifier"),
            "{harness_id}: {error}"
        );
    }
}

fn bash_tool(description: Option<&str>) -> crate::harness::spec::tools::HarnessV1BuiltinTool {
    common_tool(
        HarnessV1BuiltinToolName::Bash,
        CommonToolOptions {
            native_name: "shell".into(),
            tool_use_kind: Some(HarnessV1BuiltinToolUseKind::Bash),
            description: description.map(str::to_string),
            input_schema: json_schema(json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"],
                "additionalProperties": false,
            })),
        },
    )
    .unwrap()
}

#[tokio::test]
async fn preserves_supplied_built_in_definitions() {
    let bridge = FakeBridge::default();
    let mut settings = harness_settings(agent_settings("codex-acp"), &bridge);
    settings.builtin_tools = Some(IndexMap::from([("bash".to_string(), bash_tool(None))]));
    let harness = create_acp(settings).unwrap();

    assert_eq!(
        harness.builtin_tools()["bash"].native_name.as_deref(),
        Some("shell")
    );
}

#[test]
fn serializes_built_in_matching_fields_and_input_schemas_across_the_bridge() {
    let mut bash = bash_tool(Some("Execute a command"));
    bash.tool.title = Some("Terminal".into());
    let builtin_tools = IndexMap::from([("bash".to_string(), bash)]);

    assert_eq!(
        serde_json::to_value(serialize_builtin_tools(&builtin_tools)).unwrap(),
        json!([{
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "additionalProperties": false,
                "properties": { "command": { "type": "string" } },
                "required": ["command"],
                "type": "object",
            },
            "nativeName": "shell",
            "title": "Terminal",
            "toolName": "bash",
            "toolUseKind": "bash",
        }])
    );
    assert!(
        !serde_json::to_string(&serialize_builtin_tools(&builtin_tools))
            .unwrap()
            .contains("Execute a command")
    );
}

#[tokio::test]
async fn caches_bootstrap_per_factory_without_sharing_configuration_globally() {
    let bridge = FakeBridge::default();
    let first = create_acp(harness_settings(agent_settings("first-acp"), &bridge)).unwrap();
    let second = create_acp(harness_settings(agent_settings("second-acp"), &bridge)).unwrap();
    let first_bootstrap = first.get_bootstrap(None).await.unwrap();
    let second_bootstrap = second.get_bootstrap(None).await.unwrap();

    assert_eq!(first.get_bootstrap(None).await.unwrap(), first_bootstrap);
    assert_eq!(second.get_bootstrap(None).await.unwrap(), second_bootstrap);
    assert_ne!(first_bootstrap, second_bootstrap);
    assert_eq!(
        first_bootstrap.bootstrap_dir,
        ".harness-bootstrap/first-acp"
    );
    assert_eq!(
        second_bootstrap.bootstrap_dir,
        ".harness-bootstrap/second-acp"
    );
}

#[tokio::test]
async fn uses_the_supplied_sandbox_and_workdir_while_separating_bridge_and_direct_provider_auth() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("codex-acp");
    settings.auth = Some(ACPAuthenticationMode::Choice(
        crate::harness::spec::authentication::HarnessV1DirectAuthentication::Direct,
    ));
    settings.authentication = Some(ACPAuthentication {
        method_id: "api-key".into(),
        meta: None,
        client_capabilities: None,
    });
    settings.provider_authentication = Some(ACPProviderAuthentication {
        gateway: ACPGatewayProviderAuthentication {
            env: IndexMap::new(),
        },
    });
    let mut harness_settings = harness_settings(settings, &bridge);
    harness_settings.client_app = Some(super::acp_auth::ACPClientApp {
        name: "ai-sdk-harness-acp".into(),
        version: "0.0.0-test".into(),
    });
    let harness = create_acp(harness_settings).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    let runs = sandbox.runs();
    assert_eq!(runs[0], "printf \"%s\" \"$HOME\"");
    assert_eq!(
        runs[1],
        "mkdir -p '/workspace/user-project' '/home/agent/.ai-sdk-harness/.agent-runs/session-1/bridge'"
    );
    let spawns = sandbox.spawns();
    assert!(
        spawns[0]
            .command
            .contains("--workdir '/workspace/user-project'")
    );
    assert!(
        spawns[0]
            .command
            .contains("node '/home/agent/.ai-sdk-harness/.harness-bootstrap/codex-acp/bridge.mjs'")
    );
    let configuration: Value =
        serde_json::from_str(&spawns[0].env[ACP_BRIDGE_CONFIGURATION_ENV]).unwrap();
    assert_match_object(
        &configuration["agent"],
        &json!({
            "command": "/home/agent/.ai-sdk-harness/.harness-bootstrap/codex-acp/node_modules/.bin/codex-acp",
            "args": ["--example"],
        }),
    );
    let env = &spawns[0].env;
    assert_eq!(env["CODEX_API_KEY"], "test-key");
    assert!(
        regex::Regex::new(r"^[A-Za-z0-9_-]{43}$")
            .unwrap()
            .is_match(&env["BRIDGE_AUTH_PUBLIC_KEY"])
    );
    assert!(!env.contains_key("BRIDGE_CHANNEL_TOKEN"));
    assert!(!env.contains_key("AI_SDK_ACP_GATEWAY_API_KEY"));
    assert_eq!(env["AI_SDK_ACP_CLIENT_APP_NAME"], "ai-sdk-harness-acp");
    assert_eq!(env["AI_SDK_ACP_CLIENT_APP_VERSION"], "0.0.0-test");
    assert_eq!(sandbox.stops.load(Ordering::SeqCst), 0);

    session.do_destroy().await.unwrap();
    assert_eq!(sandbox.stops.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn passes_write_protected_paths_to_the_bridge_configuration() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("codex-acp");
    settings.auth = Some(ACPAuthenticationMode::Choice(
        crate::harness::spec::authentication::HarnessV1DirectAuthentication::Direct,
    ));
    settings.write_protected_paths = Some(vec![".example".into(), ".example-config/hooks".into()]);
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    let configuration: Value =
        serde_json::from_str(&sandbox.spawns()[0].env[ACP_BRIDGE_CONFIGURATION_ENV]).unwrap();
    assert_eq!(
        configuration["writeProtectedPaths"],
        json!([".example", ".example-config/hooks"])
    );

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn requires_explicit_bridge_settings_for_a_basic_sandbox_session() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();

    let mut settings = harness_settings(agent_settings("codex-acp"), &bridge);
    settings.port_endpoint = Some(HarnessV1PortEndpoint {
        url: "ws://127.0.0.1:4319".into(),
        headers: None,
    });
    let error = create_acp(settings)
        .unwrap()
        .do_start(start_options("session-1", sandbox.basic()))
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("explicit `port`"), "{error}");

    let mut settings = harness_settings(agent_settings("codex-acp"), &bridge);
    settings.port = Some(4319);
    let error = create_acp(settings)
        .unwrap()
        .do_start(start_options("session-1", sandbox.basic()))
        .await
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("explicit `portEndpoint`"),
        "{error}"
    );
}

#[tokio::test]
async fn uses_a_basic_sandbox_session_with_explicit_bridge_settings() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = harness_settings(agent_settings("codex-acp"), &bridge);
    settings.port = Some(4319);
    settings.port_endpoint = Some(HarnessV1PortEndpoint {
        url: "ws://127.0.0.1:4319".into(),
        headers: None,
    });
    let session = start(
        &create_acp(settings).unwrap(),
        start_options("session-1", sandbox.basic()),
    )
    .await;

    assert_eq!(sandbox.runs()[0], "printf \"%s\" \"$HOME\"");
    let resume_from = session.do_detach().await.unwrap();
    assert_match_object(
        &resume_from.data,
        &json!({ "bridge": { "port": 4319, "lastSeenEventId": 0 } }),
    );
    assert!(resume_from.data["bridge"]["privateKey"].is_string());
    assert!(resume_from.data["bridge"].get("sandboxId").is_none());
}

#[tokio::test]
async fn keeps_the_bridge_private_key_on_the_host_and_passes_endpoint_headers_when_attaching() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox {
        bridge_port_endpoint: HarnessV1PortEndpoint {
            url: "ws://unused.example".into(),
            headers: None,
        },
        ..FakeSandbox::default()
    };
    let port_endpoint = HarnessV1PortEndpoint {
        url: "wss://sandbox.example/bridge?existing=value".into(),
        headers: Some(IndexMap::from([(
            "E2B-Traffic-Access-Token".to_string(),
            "traffic-token".to_string(),
        )])),
    };
    let mut settings = harness_settings(agent_settings("codex-acp"), &bridge);
    settings.port_endpoint = Some(port_endpoint.clone());
    settings.keep_alive_ms = Some(100);
    let harness = create_acp(settings).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    assert!(
        regex::Regex::new(r"^[A-Za-z0-9_-]{43}$")
            .unwrap()
            .is_match(&sandbox.spawns()[0].env["BRIDGE_AUTH_PUBLIC_KEY"])
    );

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
    assert!(!format!("{:?}", sandbox.spawns()).contains(&private_key));

    let mut options = start_options("session-1", sandbox.network());
    options.resume_from = Some(resume_from);
    let attached = start(&harness, options).await;
    let expected = (port_endpoint.url.clone(), port_endpoint.headers.clone());
    assert_eq!(*bridge.calls.lock(), vec![expected.clone(), expected]);
    attached.do_detach().await.unwrap();
}

#[tokio::test]
async fn rejects_an_already_aborted_turn_without_sending_a_start_frame() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let abort_controller = CancellationToken::new();
    abort_controller.cancel();
    let (mut options, _) = prompt_options("Do not start");
    options.abort_signal = Some(abort_controller);

    let error = session.do_prompt_turn(options).await.err().unwrap();
    assert!(is_abort_error(&*error), "{error}");
    flush().await;
    assert!(bridge.socket(0).sent().is_empty());

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn rejects_structured_output_when_the_acp_profile_has_no_schema_mapping() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let (mut options, _) = prompt_options("Answer.");
    options.response_format = Some(HarnessV1ResponseFormat::Json {
        schema: Some(json!({ "type": "object" }).as_object().unwrap().clone()),
        name: None,
        description: None,
    });

    let error = session.do_prompt_turn(options).await.err().unwrap();
    assert!(HarnessError::is_capability_unsupported(&*error), "{error}");
    flush().await;
    assert!(bridge.socket(0).sent().is_empty());

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn passes_structured_output_configuration_to_a_mapped_acp_profile() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("grok-build-acp");
    settings.output_schema_mapping = Some(ACPOutputSchemaMapping::SessionPromptMeta {
        path: vec!["outputSchema".into()],
    });
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let schema = json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
    });
    let (mut options, _) = prompt_options("Answer.");
    options.response_format = Some(HarnessV1ResponseFormat::Json {
        schema: Some(schema.as_object().unwrap().clone()),
        name: None,
        description: None,
    });

    let control = session.do_prompt_turn(options).await.unwrap();
    let channel = bridge.socket(0);
    let sent = wait_for_sent(&channel, 1).await;
    assert_match_object(
        &sent[0],
        &json!({
            "type": "start",
            "responseFormat": { "type": "json", "schema": schema },
            "outputSchemaMapping": { "type": "session-prompt-meta", "path": ["outputSchema"] },
        }),
    );
    channel.emit(finish_frame());
    done(&*control).await.unwrap();
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn rejects_standard_acp_v1_manual_compaction_without_sending_a_command() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    let error = session.do_compact(None).await.err().unwrap();
    assert!(HarnessError::is_capability_unsupported(&*error));
    assert!(
        error
            .to_string()
            .contains("ACP v1 does not define manual session compaction")
    );
    flush().await;
    assert!(bridge.socket(0).sent().is_empty());
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn preserves_capability_errors_reported_by_the_sandbox_bridge() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let channel = bridge.socket(0);
    let (options, events) = prompt_options("Use time.");
    let turn = session.do_prompt_turn(options).await.unwrap();

    channel.emit(json!({
        "type": "error",
        "error": {
            "name": "AI_HarnessBridgeCapabilityUnsupportedError",
            "message": "The catalog was not refreshed.",
        },
    }));

    let error = done(&*turn).await.err().unwrap();
    assert!(HarnessError::is_capability_unsupported(&*error), "{error}");
    assert_eq!(error.to_string(), "The catalog was not refreshed.");
    assert_eq!(
        HarnessError::downcast(&*error).and_then(HarnessError::harness_id),
        Some("codex-acp")
    );
    let events = events.lock().clone();
    assert_eq!(events.len(), 1);
    let HarnessV1StreamPart::Error { error } = &events[0] else {
        panic!("expected an error part");
    };
    assert!(HarnessError::is_capability_unsupported(&*error.to_box()));
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn keeps_stop_and_destroy_idempotent_without_stopping_the_framework_sandbox() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(
        agent_settings("idempotent-lifecycle-acp"),
        &bridge,
    ))
    .unwrap();
    let mut options = start_options("session-stop", sandbox.network());
    options.session_work_dir = "/workspace/stop".into();
    let stopped_session = start(&harness, options).await;
    let stop_channel = bridge.socket(0);
    let (options, _) = prompt_options("Establish a stoppable session.");
    let stop_turn = stopped_session.do_prompt_turn(options).await.unwrap();
    stop_channel.emit(json!({ "type": "bridge-thread", "threadId": "acp-stop" }));
    stop_channel.emit(finish_frame());
    done(&*stop_turn).await.unwrap();
    let first_stop = stopped_session.do_stop().await.unwrap();
    let second_stop = stopped_session.do_stop().await.unwrap();
    stopped_session.do_destroy().await.unwrap();
    assert_eq!(second_stop, first_stop);
    let stops: Vec<Value> = stop_channel
        .sent()
        .into_iter()
        .filter(|message| message["type"] == "stop")
        .collect();
    assert_eq!(stops, vec![json!({ "type": "stop" })]);

    let mut options = start_options("session-destroy", sandbox.network());
    options.session_work_dir = "/workspace/destroy".into();
    let destroyed_session = start(&harness, options).await;
    let destroy_channel = bridge.socket(1);
    let (options, _) = prompt_options("Establish a destroyable session.");
    let destroy_turn = destroyed_session.do_prompt_turn(options).await.unwrap();
    destroy_channel.emit(json!({ "type": "bridge-thread", "threadId": "acp-destroy" }));
    destroy_channel.emit(finish_frame());
    done(&*destroy_turn).await.unwrap();
    destroyed_session.do_destroy().await.unwrap();
    destroyed_session.do_destroy().await.unwrap();
    let destroys: Vec<Value> = destroy_channel
        .sent()
        .into_iter()
        .filter(|message| message["type"] == "destroy")
        .collect();
    assert_eq!(destroys, vec![json!({ "type": "destroy" })]);
    assert_eq!(sandbox.record.lock().kills.len(), 2);
    assert_eq!(sandbox.stops.load(Ordering::SeqCst), 0);
}
