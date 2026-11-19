//! Port of `acp-harness.test.ts`: tool flows through `HarnessAgent`,
//! approvals, catalogs, cancellation, provider authentication and the
//! lifecycle state schema.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt as _;
use indexmap::IndexMap;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::agent::harness_agent::HarnessAgentCallParameters;
use crate::harness::agent::harness_agent::{
    HarnessAgent, HarnessAgentContinueParameters, HarnessAgentCreateSessionOptions,
};
use crate::harness::agent::settings::HarnessAgentSettings;
use crate::harness::errors::AbortError;
use crate::harness::sdk::messages::{ToolApprovalResponse, ToolResultPart};
use crate::harness::sdk::tool::{Tool, json_schema};
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1ContinueTurnOptions, HarnessV1SubmitToolApproval, HarnessV1SubmitToolResult,
};
use crate::harness::spec::lifecycle_state::HarnessV1TurnSettings;
use crate::harness::spec::tools::{
    CommonToolOptions, HarnessV1BuiltinToolName, HarnessV1BuiltinToolUseKind,
    HarnessV1PermissionMode, HarnessV1ToolSpec, common_tool,
};

use super::acp_auth::{ACPAuthenticationMode, ACPClientApp};
use super::acp_harness::create_acp;
use super::acp_harness_test_support::*;
use super::v1::bridge::bridge_environment::ACP_BRIDGE_CONFIGURATION_ENV;
use super::v1::harness::ACPV1Harness;
use super::v1::settings::{
    ACPGatewayProviderAuthentication, ACPProviderAuthentication, ACPV1Settings,
};

fn tool_spec(value: Value) -> HarnessV1ToolSpec {
    serde_json::from_value(value).unwrap()
}

fn continue_options() -> HarnessV1ContinueTurnOptions {
    HarnessV1ContinueTurnOptions {
        settings: HarnessV1TurnSettings {
            model: None,
            skills: Vec::new(),
            instructions: None,
            tools: Vec::new(),
        },
        response_format: None,
        abort_signal: None,
        emit: Arc::new(|_| {}),
    }
}

fn types(events: &Events) -> Vec<String> {
    events_json(events)
        .into_iter()
        .map(|event| event["type"].as_str().unwrap().to_string())
        .collect()
}

fn harness_with_env(
    settings: ACPV1Settings,
    bridge: &FakeBridge,
    process_env: IndexMap<String, String>,
) -> ACPV1Harness {
    let mut settings = harness_settings(settings, bridge);
    settings.process_env = Some(process_env);
    create_acp(settings).unwrap()
}

fn gateway(env: Value) -> Option<ACPProviderAuthentication> {
    Some(ACPProviderAuthentication {
        gateway: ACPGatewayProviderAuthentication {
            env: serde_json::from_value(env).unwrap(),
        },
    })
}

fn object_schema(properties: Value) -> crate::harness::sdk::tool::Schema {
    json_schema(json!({
        "type": "object",
        "properties": properties,
        "required": properties.as_object().unwrap().keys().collect::<Vec<_>>(),
        "additionalProperties": false,
    }))
}

// ─── HarnessAgent flows ────────────────────────────────────────────────

fn agent(harness: ACPV1Harness, configure: impl FnOnce(&mut HarnessAgentSettings)) -> HarnessAgent {
    let mut settings = HarnessAgentSettings::new(Arc::new(harness));
    configure(&mut settings);
    HarnessAgent::new(settings).unwrap()
}

fn session_options(sandbox: &FakeSandbox) -> HarnessAgentCreateSessionOptions {
    HarnessAgentCreateSessionOptions {
        sandbox_session: Some(sandbox.network().into()),
        ..Default::default()
    }
}

fn tool_result_part(value: Value) -> ToolResultPart {
    serde_json::from_value(value).unwrap()
}

#[tokio::test]
async fn submits_converted_output_once_while_preserving_raw_results_and_complete_response_content()
{
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let executions = Arc::new(AtomicUsize::new(0));
    let conversions = Arc::new(AtomicUsize::new(0));
    let inspect = Tool::new(object_schema(json!({})))
        .with_execute({
            let executions = executions.clone();
            move |_, _| {
                executions.fetch_add(1, Ordering::SeqCst);
                async { Ok(json!({ "status": "ready" })) }
            }
        })
        .with_to_model_output({
            let conversions = conversions.clone();
            move |_| {
                conversions.fetch_add(1, Ordering::SeqCst);
                async {
                    Ok(serde_json::from_value(json!({
                        "type": "content",
                        "value": [
                            { "type": "text", "text": "marker" },
                            { "type": "file", "mediaType": "image/png", "data": { "type": "data", "data": "AQID" } },
                        ],
                    }))
                    .unwrap())
                }
            }
        });
    let harness = create_acp(harness_settings(agent_settings("test-acp"), &bridge)).unwrap();
    let agent = agent(harness, |settings| {
        settings.tools = Some(IndexMap::from([("inspect".to_string(), inspect)]));
    });
    let session = agent
        .create_session(session_options(&sandbox))
        .await
        .unwrap();
    let channel = bridge.socket(0);
    let result = agent
        .stream(HarnessAgentCallParameters::new(session.clone()).with_prompt("Inspect the image."))
        .await
        .unwrap();
    let consumed = tokio::spawn({
        let result = result.clone();
        async move { result.consume_stream().await }
    });
    wait_for_sent(&channel, 1).await;
    channel.emit(json!({ "type": "stream-start" }));
    channel.emit(json!({
        "type": "tool-call",
        "toolCallId": "call",
        "toolName": "inspect",
        "input": "{}",
        "providerExecuted": false,
    }));
    let output = json!({
        "type": "content",
        "value": [
            { "type": "text", "text": "marker" },
            { "type": "file", "mediaType": "image/png", "data": { "type": "data", "data": "AQID" } },
        ],
    });
    let expected = json!({
        "type": "tool-result",
        "toolCallId": "call",
        "output": { "status": "ready" },
        "toolResult": {
            "type": "tool-result",
            "toolCallId": "call",
            "toolName": "inspect",
            "output": output,
        },
    });
    wait_for_frame(&channel, &expected).await;
    channel.emit(json!({
        "type": "tool-result",
        "toolCallId": "call",
        "toolName": "inspect",
        "result": { "status": "ready" },
    }));
    channel.emit(json!({
        "type": "finish-step",
        "finishReason": { "unified": "tool-calls", "raw": "tool-calls" },
        "usage": unknown_usage(),
    }));
    channel.emit(finish_frame());
    consumed.await.unwrap();
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(conversions.load(Ordering::SeqCst), 1);
    let tool_results = serde_json::to_value(result.tool_results().await.unwrap()).unwrap();
    assert_match_object(&tool_results, &json!([{ "output": { "status": "ready" } }]));
    let response_messages =
        serde_json::to_value(result.response_messages().await.unwrap()).unwrap();
    assert!(
        response_messages.as_array().unwrap().contains(&json!({
            "role": "tool",
            "content": [{
                "type": "tool-result",
                "toolCallId": "call",
                "toolName": "inspect",
                "output": output,
            }],
        })),
        "{response_messages:#}"
    );
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn preserves_a_pending_client_tool_result_across_cross_process_continuation() {
    for convert_model_output in [false, true] {
        let bridge = FakeBridge::default();
        let sandbox = FakeSandbox::default();
        let model_output = json!({
            "type": "content",
            "value": [
                { "type": "text", "text": "marker" },
                { "type": "file", "mediaType": "image/png", "data": { "type": "data", "data": "AQID" } },
            ],
        });
        let conversions = Arc::new(AtomicUsize::new(0));
        let mut client_tool = Tool::new(object_schema(json!({ "key": { "type": "string" } })))
            .with_description("Get a value from the client.");
        if convert_model_output {
            let conversions = conversions.clone();
            let model_output = model_output.clone();
            client_tool = client_tool.with_to_model_output(move |_| {
                conversions.fetch_add(1, Ordering::SeqCst);
                let model_output = model_output.clone();
                async move { Ok(serde_json::from_value(model_output).unwrap()) }
            });
        }
        let tools = IndexMap::from([("clientTool".to_string(), client_tool)]);
        let harness =
            Arc::new(create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap());
        let make_agent = || {
            let mut settings = HarnessAgentSettings::new(harness.clone());
            settings.tools = Some(tools.clone());
            HarnessAgent::new(settings).unwrap()
        };
        let first_agent = make_agent();
        let first_session = first_agent
            .create_session(HarnessAgentCreateSessionOptions {
                session_id: Some("session-1".into()),
                ..session_options(&sandbox)
            })
            .await
            .unwrap();
        let first_channel = bridge.socket(0);
        let first = first_agent
            .stream(
                HarnessAgentCallParameters::new(first_session.clone())
                    .with_prompt("Use the client tool."),
            )
            .await
            .unwrap();
        let first_parts = tokio::spawn({
            let first = first.clone();
            async move { first.full_stream().collect::<Vec<_>>().await }
        });
        wait_for_sent(&first_channel, 1).await;
        first_channel.emit(json!({ "type": "stream-start" }));
        first_channel.emit(json!({
            "type": "tool-call",
            "toolCallId": "client-call",
            "toolName": "clientTool",
            "input": json!({ "key": "answer" }).to_string(),
            "providerExecuted": false,
        }));
        first_channel.emit(json!({
            "type": "finish-step",
            "finishReason": { "unified": "tool-calls", "raw": "host-input" },
            "usage": unknown_usage(),
            "harnessMetadata": { "acp": { "inferredStep": true } },
            "seq": 31,
        }));
        first_parts.await.unwrap();

        let continue_from = first_session.suspend_turn().await.unwrap();
        assert_eq!(
            serde_json::to_value(&continue_from.pending_tool_results).unwrap(),
            json!([{
                "toolCallId": "client-call",
                "toolName": "clientTool",
                "input": json!({ "key": "answer" }).to_string(),
            }])
        );

        let second_agent = make_agent();
        let second_session = second_agent
            .create_session(HarnessAgentCreateSessionOptions {
                session_id: Some("session-1".into()),
                continue_from: Some(continue_from),
                ..session_options(&sandbox)
            })
            .await
            .unwrap();
        let second_channel = bridge.socket(1);
        let continuation_output = if convert_model_output {
            model_output.clone()
        } else {
            json!({ "type": "json", "value": { "value": 42 } })
        };
        let mut params = HarnessAgentContinueParameters::new(second_session.clone());
        params.tool_result_continuations = vec![tool_result_part(json!({
            "type": "tool-result",
            "toolCallId": "client-call",
            "toolName": "clientTool",
            "output": continuation_output,
        }))];
        let continued = second_agent.continue_stream(params).await.unwrap();
        let continued_parts = tokio::spawn({
            let continued = continued.clone();
            async move { continued.full_stream().collect::<Vec<_>>().await }
        });
        let raw_output = if convert_model_output {
            model_output.clone()
        } else {
            json!({ "value": 42 })
        };
        let expected = json!({
            "type": "tool-result",
            "toolCallId": "client-call",
            "output": raw_output,
            "toolResult": {
                "type": "tool-result",
                "toolCallId": "client-call",
                "toolName": "clientTool",
                "output": continuation_output,
            },
        });
        wait_for_frame(&second_channel, &expected).await;
        second_channel.emit(json!({
            "type": "tool-result",
            "toolCallId": "client-call",
            "toolName": "clientTool",
            "result": raw_output,
            "providerExecuted": false,
        }));
        second_channel.emit(json!({
            "type": "finish-step",
            "finishReason": { "unified": "tool-calls", "raw": "tool-calls" },
            "usage": unknown_usage(),
        }));
        second_channel.emit(finish_frame());
        continued_parts.await.unwrap();
        assert_eq!(conversions.load(Ordering::SeqCst), 0);
        second_session.destroy().await.unwrap();
    }
}

#[tokio::test]
async fn preserves_a_pending_native_approval_across_cross_process_continuation() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = harness_settings(agent_settings("codex-acp"), &bridge);
    settings.settings.permission_mode_mapping = Some(permission_mode_mapping());
    settings.builtin_tools = Some(IndexMap::from([(
        "bash".to_string(),
        common_tool(
            HarnessV1BuiltinToolName::Bash,
            CommonToolOptions {
                native_name: "shell".into(),
                tool_use_kind: Some(HarnessV1BuiltinToolUseKind::Bash),
                description: None,
                input_schema: object_schema(json!({ "command": { "type": "string" } })),
            },
        )
        .unwrap(),
    )]));
    let harness = Arc::new(create_acp(settings).unwrap());
    let make_agent = || {
        let mut settings = HarnessAgentSettings::new(harness.clone());
        settings.permission_mode = Some(HarnessV1PermissionMode::AllowEdits);
        HarnessAgent::new(settings).unwrap()
    };
    let first_agent = make_agent();
    let first_session = first_agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("session-1".into()),
            ..session_options(&sandbox)
        })
        .await
        .unwrap();
    let first_channel = bridge.socket(0);
    let first = first_agent
        .stream(HarnessAgentCallParameters::new(first_session.clone()).with_prompt("Run pwd."))
        .await
        .unwrap();
    let first_parts = tokio::spawn({
        let first = first.clone();
        async move { first.full_stream().collect::<Vec<_>>().await }
    });
    wait_for_sent(&first_channel, 1).await;
    first_channel.emit(json!({ "type": "stream-start" }));
    first_channel.emit(json!({
        "type": "tool-call",
        "toolCallId": "native-call",
        "toolName": "bash",
        "input": json!({ "command": "pwd" }).to_string(),
        "providerExecuted": true,
    }));
    first_channel.emit(json!({
        "type": "tool-approval-request",
        "approvalId": "native-approval",
        "toolCallId": "native-call",
    }));
    first_channel.emit(json!({
        "type": "finish-step",
        "finishReason": { "unified": "tool-calls", "raw": "host-input" },
        "usage": unknown_usage(),
        "harnessMetadata": { "acp": { "inferredStep": true } },
        "seq": 37,
    }));
    first_parts.await.unwrap();

    let continue_from = first_session.suspend_turn().await.unwrap();
    assert_eq!(
        serde_json::to_value(&continue_from.pending_tool_approvals).unwrap(),
        json!([{
            "approvalId": "native-approval",
            "toolCallId": "native-call",
            "toolName": "bash",
            "input": json!({ "command": "pwd" }).to_string(),
            "kind": "builtin",
            "providerExecuted": true,
        }])
    );

    let second_agent = make_agent();
    let second_session = second_agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("session-1".into()),
            continue_from: Some(continue_from),
            ..session_options(&sandbox)
        })
        .await
        .unwrap();
    let second_channel = bridge.socket(1);
    let mut params = HarnessAgentContinueParameters::new(second_session.clone());
    params.tool_approval_continuations = vec![ToolApprovalResponse {
        approval_id: "native-approval".into(),
        approved: true,
        reason: None,
        provider_executed: None,
    }];
    let continued = second_agent.continue_stream(params).await.unwrap();
    let continued_parts = tokio::spawn({
        let continued = continued.clone();
        async move { continued.full_stream().collect::<Vec<_>>().await }
    });
    let expected = json!({
        "type": "tool-approval-response",
        "approvalId": "native-approval",
        "approved": true,
    });
    wait_for_frame(&second_channel, &expected).await;
    second_channel.emit(json!({
        "type": "tool-result",
        "toolCallId": "native-call",
        "toolName": "bash",
        "result": { "output": "/workspace/user-project" },
        "providerExecuted": true,
    }));
    second_channel.emit(finish_frame());
    continued_parts.await.unwrap();
    second_session.destroy().await.unwrap();
}

#[tokio::test]
async fn executes_host_tools_in_the_framework_and_returns_only_the_final_generator_value() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let weather = Tool::new(object_schema(json!({ "city": { "type": "string" } })))
        .with_streaming_execute(|_, _| {
            futures::stream::iter(vec![
                Ok(json!({ "state": "loading" })),
                Ok(json!({ "state": "ready", "celsius": 19 })),
            ])
            .boxed()
        });
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let agent = agent(harness, |settings| {
        settings.tools = Some(IndexMap::from([("weather".to_string(), weather)]));
    });
    let session = agent
        .create_session(session_options(&sandbox))
        .await
        .unwrap();
    let channel = bridge.socket(0);
    let result = agent
        .stream(
            HarnessAgentCallParameters::new(session.clone()).with_prompt("Use weather for Lima."),
        )
        .await
        .unwrap();
    let parts = tokio::spawn({
        let result = result.clone();
        async move { result.full_stream().collect::<Vec<_>>().await }
    });
    wait_for_sent(&channel, 1).await;

    channel.emit(json!({ "type": "stream-start" }));
    channel.emit(json!({
        "type": "tool-call",
        "toolCallId": "weather-call",
        "toolName": "weather",
        "input": json!({ "city": "Lima" }).to_string(),
        "providerExecuted": false,
    }));
    wait_for(|| {
        channel
            .sent()
            .iter()
            .filter(|message| message["type"] == "tool-result")
            .count()
            == 1
    })
    .await;
    assert_eq!(
        channel.sent().last().unwrap(),
        &json!({
            "type": "tool-result",
            "toolCallId": "weather-call",
            "output": { "state": "ready", "celsius": 19 },
        })
    );

    channel.emit(json!({
        "type": "tool-result",
        "toolCallId": "weather-call",
        "toolName": "weather",
        "result": { "state": "ready", "celsius": 19 },
    }));
    channel.emit(json!({
        "type": "finish-step",
        "finishReason": { "unified": "tool-calls", "raw": "tool-calls" },
        "usage": unknown_usage(),
    }));
    channel.emit(finish_frame());

    let parts: Vec<Value> = parts
        .await
        .unwrap()
        .iter()
        .map(|part| part.to_json())
        .filter(|part| part["type"] == "tool-result" && part["preliminary"] == true)
        .collect();
    assert_match_object(
        &json!(parts),
        &json!([
            { "output": { "state": "loading" }, "preliminary": true },
            { "output": { "state": "ready", "celsius": 19 }, "preliminary": true },
        ]),
    );
    assert_eq!(result.tool_calls().await.unwrap().len(), 1);
    session.destroy().await.unwrap();
}

#[tokio::test]
async fn pauses_a_non_executable_tool_and_continues_the_same_live_acp_turn() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let client_tool = Tool::new(object_schema(json!({ "question": { "type": "string" } })));
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let agent = agent(harness, |settings| {
        settings.tools = Some(IndexMap::from([("clientTool".to_string(), client_tool)]));
    });
    let session = agent
        .create_session(session_options(&sandbox))
        .await
        .unwrap();
    let channel = bridge.socket(0);
    let first = agent
        .stream(
            HarnessAgentCallParameters::new(session.clone()).with_prompt("Use the client tool."),
        )
        .await
        .unwrap();
    let first_parts = tokio::spawn({
        let first = first.clone();
        async move { first.full_stream().collect::<Vec<_>>().await }
    });
    wait_for_sent(&channel, 1).await;
    channel.emit(json!({ "type": "stream-start" }));
    channel.emit(json!({
        "type": "tool-call",
        "toolCallId": "client-call",
        "toolName": "clientTool",
        "input": json!({ "question": "name" }).to_string(),
        "providerExecuted": false,
    }));
    channel.emit(json!({
        "type": "finish-step",
        "finishReason": { "unified": "tool-calls", "raw": "host-input" },
        "usage": unknown_usage(),
        "harnessMetadata": { "acp": { "inferredStep": true } },
    }));
    first_parts.await.unwrap();
    assert!(session.has_unfinished_turn());

    let mut params = HarnessAgentContinueParameters::new(session.clone());
    params.tool_result_continuations = vec![tool_result_part(json!({
        "type": "tool-result",
        "toolCallId": "client-call",
        "toolName": "clientTool",
        "output": { "type": "json", "value": { "answer": "Ada" } },
    }))];
    let continued = agent.continue_stream(params).await.unwrap();
    let continued_parts = tokio::spawn({
        let continued = continued.clone();
        async move { continued.full_stream().collect::<Vec<_>>().await }
    });
    let expected = json!({
        "type": "tool-result",
        "toolCallId": "client-call",
        "output": { "answer": "Ada" },
        "toolResult": {
            "type": "tool-result",
            "toolCallId": "client-call",
            "toolName": "clientTool",
            "output": { "type": "json", "value": { "answer": "Ada" } },
        },
    });
    wait_for_frame(&channel, &expected).await;
    assert_eq!(channel.sent().last(), Some(&expected));
    channel.emit(json!({
        "type": "tool-result",
        "toolCallId": "client-call",
        "toolName": "clientTool",
        "result": { "answer": "Ada" },
    }));
    channel.emit(json!({
        "type": "finish-step",
        "finishReason": { "unified": "tool-calls", "raw": "tool-calls" },
        "usage": unknown_usage(),
    }));
    channel.emit(finish_frame());
    continued_parts.await.unwrap();

    assert!(!session.has_unfinished_turn());
    session.destroy().await.unwrap();
}

// ─── Direct session flows ──────────────────────────────────────────────

#[tokio::test]
async fn sends_the_permission_mode_mapping_to_the_acp_bridge() {
    for (permission_mode, name) in [
        (HarnessV1PermissionMode::AllowReads, "allow-reads"),
        (HarnessV1PermissionMode::AllowEdits, "allow-edits"),
        (HarnessV1PermissionMode::AllowAll, "allow-all"),
    ] {
        let bridge = FakeBridge::default();
        let sandbox = FakeSandbox::default();
        let mut settings = agent_settings("codex-acp");
        settings.permission_mode_mapping = Some(permission_mode_mapping());
        let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
        let mut options = start_options("session-1", sandbox.network());
        options.permission_mode = Some(permission_mode);
        let session = start(&harness, options).await;
        let channel = bridge.socket(0);
        let (options, _) = prompt_options("Check permissions.");
        let control = session.do_prompt_turn(options).await.unwrap();

        let sent = wait_for_sent(&channel, 1).await;
        assert_match_object(
            &sent[0],
            &json!({
                "type": "start",
                "permissionMode": name,
                "permissionModeMapping": serde_json::to_value(permission_mode_mapping()).unwrap(),
            }),
        );
        channel.emit(finish_frame());
        done(&*control).await.unwrap();
        session.do_destroy().await.unwrap();
    }
}

#[tokio::test]
async fn submits_a_host_approval_decision_to_the_pending_acp_turn() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("codex-acp");
    settings.permission_mode_mapping = Some(permission_mode_mapping());
    let harness = create_acp(harness_settings(settings, &bridge)).unwrap();
    let mut options = start_options("session-1", sandbox.network());
    options.permission_mode = Some(HarnessV1PermissionMode::AllowEdits);
    let session = start(&harness, options).await;
    let channel = bridge.socket(0);
    let (options, _) = prompt_options("Run a native command.");
    let initial = session.do_prompt_turn(options).await.unwrap();
    let continued = session.do_continue_turn(continue_options()).await.unwrap();

    continued
        .submit_tool_approval(HarnessV1SubmitToolApproval {
            approval_id: "approval-1".into(),
            approved: false,
            reason: Some("Not allowed".into()),
        })
        .await
        .unwrap();
    let sent = wait_for_sent(&channel, 2).await;
    assert_eq!(
        sent.last().unwrap(),
        &json!({
            "type": "tool-approval-response",
            "approvalId": "approval-1",
            "approved": false,
            "reason": "Not allowed",
        })
    );

    channel.emit(finish_frame());
    done(&*initial).await.unwrap();
    done(&*continued).await.unwrap();
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn sends_the_exact_host_tool_catalog_and_submits_a_continued_client_result() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let channel = bridge.socket(0);
    let recursive_schema = json!({
        "type": "object",
        "properties": { "node": { "$ref": "#/$defs/node" } },
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "children": { "type": "array", "items": { "$ref": "#/$defs/node" } },
                },
            },
        },
    });
    let (mut options, _) = prompt_options("Use the client tool.");
    options.settings.tools = vec![tool_spec(json!({
        "name": "client_tree",
        "description": "Inspect a recursive tree.",
        "inputSchema": recursive_schema,
    }))];
    let initial = session.do_prompt_turn(options).await.unwrap();

    let sent = wait_for_sent(&channel, 1).await;
    assert_match_object(
        &sent[0],
        &json!({
            "type": "start",
            "tools": [{
                "name": "client_tree",
                "description": "Inspect a recursive tree.",
                "inputSchema": recursive_schema,
            }],
        }),
    );

    let continued = session.do_continue_turn(continue_options()).await.unwrap();
    continued
        .submit_tool_result(HarnessV1SubmitToolResult {
            tool_call_id: "client-call-1".into(),
            output: json!({ "accepted": true }),
            is_error: None,
            tool_result: None,
        })
        .await
        .unwrap();
    let sent = wait_for_sent(&channel, 2).await;
    assert_eq!(
        sent[1],
        json!({
            "type": "tool-result",
            "toolCallId": "client-call-1",
            "output": { "accepted": true },
        })
    );

    channel.emit(finish_frame());
    done(&*initial).await.unwrap();
    done(&*continued).await.unwrap();
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn sends_changed_removed_and_unchanged_catalogs_on_consecutive_turns() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let channel = bridge.socket(0);
    let weather = json!({ "name": "weather", "description": "Read weather.", "inputSchema": { "type": "object" } });
    let time =
        json!({ "name": "time", "description": "Read time.", "inputSchema": { "type": "object" } });

    for (index, (prompt, tool)) in [
        ("Use weather.", &weather),
        ("Use time.", &time),
        ("Use time again.", &time),
    ]
    .into_iter()
    .enumerate()
    {
        let (mut options, _) = prompt_options(prompt);
        options.settings.tools = vec![tool_spec(tool.clone())];
        let control = session.do_prompt_turn(options).await.unwrap();
        wait_for_sent(&channel, index + 1).await;
        channel.emit(finish_frame());
        done(&*control).await.unwrap();
    }

    let catalogs: Vec<Value> = channel
        .sent()
        .into_iter()
        .filter(|message| message["type"] == "start")
        .map(|message| message["tools"].clone())
        .collect();
    assert_eq!(
        catalogs,
        vec![json!([weather]), json!([time]), json!([time])]
    );

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn waits_for_the_bridge_terminal_sequence_after_cancellation_before_rejecting_the_turn() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let channel = bridge.socket(0);
    let abort_controller = CancellationToken::new();
    let (mut options, events) = prompt_options("Start, then cancel");
    options.abort_signal = Some(abort_controller.clone());
    let control = session.do_prompt_turn(options).await.unwrap();

    let sent = wait_for_sent(&channel, 1).await;
    assert_match_object(&json!(sent), &json!([{ "type": "start" }]));
    abort_controller.cancel();
    let sent = wait_for_sent(&channel, 2).await;
    assert_match_object(
        &json!(sent),
        &json!([{ "type": "start" }, { "type": "abort" }]),
    );

    let done_future = control.done();
    flush().await;
    let settled = futures::FutureExt::now_or_never(control.done());
    assert!(settled.is_none());

    let finish_reason = json!({ "unified": "other", "raw": "cancelled" });
    channel.emit(json!({ "type": "text-start", "id": "partial" }));
    channel.emit(json!({ "type": "text-end", "id": "partial" }));
    channel.emit(json!({
        "type": "finish-step",
        "finishReason": finish_reason,
        "usage": unknown_usage(),
        "harnessMetadata": { "acp": { "inferredStep": true } },
    }));
    channel.emit(json!({
        "type": "finish",
        "finishReason": finish_reason,
        "totalUsage": unknown_usage(),
        "harnessMetadata": { "acp": { "stopReason": "cancelled" } },
    }));

    let error = done_future.await.err().unwrap();
    // [rust] `done` carries the abort through an `ErrorValue`, so the
    // check is by message, not by `toBe(abortError)` identity.
    assert_eq!(error.to_string(), AbortError.to_string());
    assert_eq!(
        types(&events),
        vec!["text-start", "text-end", "finish-step", "finish"]
    );

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn rejects_a_cancelled_turn_when_the_bridge_connection_fails() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = harness_settings(agent_settings("codex-acp"), &bridge);
    // [rust] The real channel reconnects after a drop; a zero reconnect
    // window makes the drop terminal, as the TS fake channel's close is.
    settings.reconnect = Some(crate::harness::utils::SandboxChannelReconnectOptions {
        max_elapsed_ms: Some(0),
        initial_delay_ms: None,
        max_delay_ms: None,
    });
    let harness = create_acp(settings).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let channel = bridge.socket(0);
    let abort_controller = CancellationToken::new();
    let (mut options, events) = prompt_options("Start, then disconnect");
    options.abort_signal = Some(abort_controller.clone());
    let control = session.do_prompt_turn(options).await.unwrap();
    wait_for_sent(&channel, 1).await;

    abort_controller.cancel();
    wait_for_sent(&channel, 2).await;
    channel.emit(json!({ "type": "reasoning-start", "id": "partial" }));
    bridge
        .open_errors
        .lock()
        .extend(["bridge gone".to_string()]);
    channel.emit_close(1006, "closed");
    let error = done(&*control).await.err().unwrap();
    // [rust] `done` carries the abort through an `ErrorValue`, so the
    // check is by message, not by `toBe(abortError)` identity.
    assert_eq!(error.to_string(), AbortError.to_string());
    assert_eq!(types(&events), vec!["reasoning-start", "reasoning-end"]);

    session.do_destroy().await.unwrap();
}

// ─── Provider authentication ───────────────────────────────────────────

#[tokio::test]
async fn resolves_downstream_gateway_credentials_separately_from_bridge_auth() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("codex-acp-gateway");
    settings.auth = Some(ACPAuthenticationMode::AiGateway);
    settings.forward_env = Some(Vec::new());
    settings.provider_authentication = gateway(json!({}));
    let mut settings = harness_settings(settings, &bridge);
    settings.client_app = Some(ACPClientApp {
        name: "custom-client".into(),
        version: "1.2.3".into(),
    });
    let mut process_env = test_env();
    process_env.insert("AI_GATEWAY_API_KEY".into(), "gateway-key".into());
    process_env.insert(
        "AI_GATEWAY_BASE_URL".into(),
        "https://gateway.example/custom".into(),
    );
    settings.process_env = Some(process_env);
    let harness = create_acp(settings).unwrap();
    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    let env = sandbox.spawns()[0].env.clone();
    assert_eq!(env["AI_SDK_ACP_GATEWAY_API_KEY"], "gateway-key");
    assert_eq!(
        env["AI_SDK_ACP_GATEWAY_BASE_URL"],
        "https://gateway.example/custom"
    );
    assert_eq!(env["AI_SDK_ACP_CLIENT_APP_NAME"], "custom-client");
    assert_eq!(env["AI_SDK_ACP_CLIENT_APP_VERSION"], "1.2.3");
    assert_ne!(env["BRIDGE_AUTH_PUBLIC_KEY"], "gateway-key");
    assert!(!env.contains_key("CODEX_API_KEY"));

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn resolves_automatic_provider_authentication_from_the_session_start_environment() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("late-auth-acp");
    settings.forward_env = Some(Vec::new());
    settings.provider_authentication = gateway(json!({
        "PROVIDER_API_KEY": { "$source": "gateway-api-key" },
    }));
    let mut process_env = test_env();
    process_env.insert("AI_GATEWAY_API_KEY".into(), "late-gateway-key".into());
    process_env.insert(
        "AI_GATEWAY_BASE_URL".into(),
        "https://gateway.example/late".into(),
    );
    let harness = harness_with_env(settings, &bridge, process_env);
    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    let spawn = sandbox.spawns()[0].clone();
    assert_eq!(spawn.env["AI_SDK_ACP_GATEWAY_API_KEY"], "late-gateway-key");
    assert_eq!(
        spawn.env["AI_SDK_ACP_GATEWAY_BASE_URL"],
        "https://gateway.example/late"
    );
    let configuration: Value =
        serde_json::from_str(&spawn.env[ACP_BRIDGE_CONFIGURATION_ENV]).unwrap();
    assert_match_object(
        &configuration,
        &json!({ "providerAuthentication": { "type": "ai-gateway" } }),
    );
    assert_match_object(
        &session.do_stop().await.unwrap().data,
        &json!({
            "authenticationProfile": {
                "providerKind": "ai-gateway",
                "providerMode": "auto",
                "gatewayCredentialSource": "AI_GATEWAY_API_KEY",
            },
        }),
    );
}

#[tokio::test]
async fn excludes_resolved_credentials_from_bootstrap_and_lifecycle_state() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("secret-safe-acp");
    settings.auth = Some(ACPAuthenticationMode::AiGateway);
    settings.forward_env = Some(vec!["PROVIDER_API_KEY".into()]);
    settings.env = Some(IndexMap::from([(
        "PROVIDER_BASE_URL".to_string(),
        "https://provider.example".to_string(),
    )]));
    settings.provider_authentication = gateway(json!({
        "PROVIDER_API_KEY": { "$source": "gateway-api-key" },
    }));
    let mut process_env = test_env();
    process_env.insert("PROVIDER_API_KEY".into(), "direct-secret".into());
    process_env.insert("AI_GATEWAY_API_KEY".into(), "gateway-secret".into());
    process_env.insert(
        "AI_GATEWAY_BASE_URL".into(),
        "https://gateway.example".into(),
    );
    let harness = harness_with_env(settings, &bridge, process_env);
    let bootstrap = serde_json::to_string(&harness.get_bootstrap(None).await.unwrap()).unwrap();

    for hidden in ["direct-secret", "gateway-secret", "AI_GATEWAY_API_KEY"] {
        assert!(!bootstrap.contains(hidden), "{hidden}");
    }

    let session = start(&harness, start_options("session-1", sandbox.network())).await;
    let env = sandbox.spawns()[0].env.clone();
    assert_eq!(env["PROVIDER_API_KEY"], "direct-secret");
    assert_eq!(env["PROVIDER_BASE_URL"], "https://provider.example");
    assert_eq!(env["AI_SDK_ACP_GATEWAY_API_KEY"], "gateway-secret");

    let lifecycle_state = session.do_stop().await.unwrap();
    let serialized = serde_json::to_string(&lifecycle_state).unwrap();
    assert!(!serialized.contains("direct-secret"));
    assert!(!serialized.contains("gateway-secret"));
    assert!(lifecycle_state.data["agentIdentity"].is_string());
    assert!(
        regex::Regex::new("^[a-f0-9]{64}$").unwrap().is_match(
            lifecycle_state.data["authenticationProfile"]["digest"]
                .as_str()
                .unwrap()
        )
    );
    assert_match_object(
        &lifecycle_state.data,
        &json!({
            "authenticationProfile": {
                "providerKind": "ai-gateway",
                "providerMode": "ai-gateway",
                "gatewayCredentialSource": "AI_GATEWAY_API_KEY",
            },
        }),
    );
}

#[tokio::test]
async fn excludes_direct_provider_credentials_from_its_authentication_profile() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let mut settings = agent_settings("direct-secret-safe-acp");
    settings.auth = Some(ACPAuthenticationMode::Choice(
        crate::harness::spec::authentication::HarnessV1DirectAuthentication::Direct,
    ));
    settings.forward_env = Some(vec!["PROVIDER_API_KEY".into()]);
    settings.provider_authentication = gateway(json!({
        "PROVIDER_API_KEY": { "$source": "gateway-api-key" },
    }));
    let mut process_env = test_env();
    process_env.insert("PROVIDER_API_KEY".into(), "direct-profile-secret".into());
    process_env.insert("AI_GATEWAY_API_KEY".into(), "unused-gateway-secret".into());
    let harness = harness_with_env(settings, &bridge, process_env);
    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    let env = sandbox.spawns()[0].env.clone();
    assert_eq!(env["PROVIDER_API_KEY"], "direct-profile-secret");
    assert!(!env.contains_key("AI_SDK_ACP_GATEWAY_API_KEY"));

    let lifecycle_state = session.do_stop().await.unwrap();
    let serialized = serde_json::to_string(&lifecycle_state).unwrap();
    assert!(!serialized.contains("direct-profile-secret"));
    assert!(!serialized.contains("unused-gateway-secret"));
    assert!(lifecycle_state.data["agentIdentity"].is_string());
    assert_match_object(
        &lifecycle_state.data,
        &json!({
            "authenticationProfile": { "providerKind": "direct", "providerMode": "direct" },
        }),
    );
}

// ─── Lifecycle state schema ────────────────────────────────────────────

#[tokio::test]
async fn validates_lifecycle_state_structurally_and_rejects_incompatible_identities_at_start() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let first = create_acp(harness_settings(agent_settings("identity-acp"), &bridge)).unwrap();
    let mut second_settings = agent_settings("identity-acp");
    second_settings.args = Some(vec!["different".into()]);
    let second = create_acp(harness_settings(second_settings, &bridge)).unwrap();
    let first_session = start(&first, start_options("session-1", sandbox.network())).await;
    let resume_from = first_session.do_stop().await.unwrap();

    assert!(
        second
            .lifecycle_state_schema()
            .unwrap()
            .validate_value(&resume_from.data)
            .is_ok()
    );
    let mut options = start_options("session-1", sandbox.network());
    options.resume_from = Some(resume_from);
    let error = second.do_start(options).await.err().unwrap();
    assert!(
        error
            .to_string()
            .contains("ACP lifecycle state is incompatible with the configured agent."),
        "{error}"
    );
}

#[tokio::test]
async fn rejects_lifecycle_state_with_a_legacy_bridge_token() {
    let bridge = FakeBridge::default();
    let harness = create_acp(harness_settings(agent_settings("codex-acp"), &bridge)).unwrap();

    let result = harness
        .lifecycle_state_schema()
        .unwrap()
        .validate_value(&json!({
            "agentIdentity": "agent",
            "bridge": { "port": 4319, "token": "legacy-token", "lastSeenEventId": 0 },
        }));

    assert!(result.is_err());
}
