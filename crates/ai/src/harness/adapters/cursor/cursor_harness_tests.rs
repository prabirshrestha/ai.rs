//! Port of `harness-cursor/src/cursor-harness.test.ts`.
//!
//! [rust] The TS tests mock `createACP` and inspect the settings it is
//! called with; these tests inspect [`cursor_acp_settings`] instead.

use std::sync::Arc;

use futures::FutureExt as _;
use indexmap::IndexMap;
use serde_json::{Value as JsonValue, json};

use super::cursor_harness::{
    CursorHarnessSettings, create_cursor, cursor_acp_settings, cursor_authentication_warning,
};
use crate::harness::adapters::acp::v1::settings::ACPCredentialBrokeringOptions;
use crate::harness::adapters::acp::{
    ACPHarnessSettings, ACPResolveHostToolCallOptions, ACPResolvedHostToolCall, ACPToolCall,
};
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
use crate::harness::spec::{
    HarnessV1Authentication, HarnessV1CredentialForwarding, HarnessV1DirectAuthentication,
};
use crate::harness::utils::SandboxChannelReconnectOptions;

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn brokering(
    settings: &ACPHarnessSettings,
    env: &IndexMap<String, String>,
    sandbox_env: Option<&IndexMap<String, String>>,
    headers: Option<&IndexMap<String, String>>,
) -> JsonValue {
    let brokering = settings.settings.credential_brokering.as_ref().unwrap();
    serde_json::to_value(
        brokering(ACPCredentialBrokeringOptions {
            env,
            sandbox_env,
            headers,
        })
        .unwrap(),
    )
    .unwrap()
}

fn tool_call(value: JsonValue) -> ACPToolCall {
    serde_json::from_value(value).unwrap()
}

#[test]
fn enforces_the_cursor_acp_agent() {
    let settings = cursor_acp_settings(CursorHarnessSettings::default());
    let acp = &settings.settings;
    assert_eq!(
        acp.non_text_content_types,
        Some(vec![
            crate::harness::adapters::acp::v1::settings::ACPNonTextContentType::Image
        ])
    );
    assert_eq!(acp.write_protected_paths, Some(vec![".cursor".to_string()]));
    let builtin_tool_names: Vec<&str> = settings
        .builtin_tools
        .as_ref()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        json!({
            "version": acp.version,
            "harnessId": acp.harness_id,
            "clientApp": settings.client_app,
            "source": acp.source,
            "executable": acp.executable,
            "args": acp.args,
            "credentialEnv": acp.credential_env,
            "authentication": acp.authentication,
            "providerAuthentication": acp.provider_authentication,
            "builtinToolNames": builtin_tool_names,
        }),
        json!({
            "args": ["--disable-auto-update", "--disable-project-configs", "acp"],
            "authentication": null,
            "builtinToolNames": [
                "bash",
                "delete",
                "glob",
                "grep",
                "read",
                "write",
                "updateTodos",
                "readTodos",
                "edit",
                "ls",
                "readLints",
                "semanticSearch",
                "createPlan",
                "webSearch",
                "task",
                "listMcpResources",
                "readMcpResource",
                "applyAgentDiff",
                "fetch",
                "switchMode",
                "generateImage",
                "recordScreen",
                "computerUse",
                "writeShellStdin",
                "reflect",
                "setupVmEnvironment",
                "replaceEnv",
                "startGrindExecution",
                "startGrindPlanning",
                "webFetch",
                "reportBugfixResults",
            ],
            "clientApp": {
                "name": "ai-sdk-harness-cursor",
                "version": super::VERSION,
            },
            "credentialEnv": ["CURSOR_API_KEY"],
            "executable": "agent",
            "harnessId": "cursor",
            "providerAuthentication": null,
            "source": {
                "command": "curl https://cursor.com/install -fsS | bash",
                "type": "install-command",
            },
            "version": "v1",
        })
    );

    let write_tool = &settings.builtin_tools.as_ref().unwrap()["write"];
    assert_eq!(write_tool.tool.title.as_deref(), Some("Write"));
    assert_eq!(write_tool.native_name.as_deref(), Some("writeToolCall"));
    assert_eq!(
        write_tool.tool_use_kind,
        Some(crate::harness::spec::tools::HarnessV1BuiltinToolUseKind::Edit)
    );
    let schema = &write_tool.tool.input_schema.json_schema;
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["path"], json!({ "type": "string" }));
    assert_eq!(schema["required"], json!(["path"]));

    assert_eq!(
        brokering(
            &settings,
            &env(&[("CURSOR_API_KEY", "cursor-secret")]),
            Some(&env(&[("CURSOR_API_KEY", "sandbox-cursor-secret")])),
            None,
        ),
        json!([
            {
                "match": {
                    "host": "api2.cursor.sh",
                    "path": { "exact": "/auth/exchange_user_api_key" },
                    "method": ["POST"],
                    "headers": [
                        {
                            "key": { "exact": "Authorization" },
                            "value": { "exact": "Bearer sandbox-cursor-secret" },
                        },
                    ],
                },
                "transform": {
                    "headers": { "Authorization": "Bearer cursor-secret" },
                },
            },
        ])
    );
    assert_eq!(
        brokering(&settings, &IndexMap::new(), None, None),
        json!([])
    );
    assert_eq!(
        serde_json::to_value(&acp.model_mapping).unwrap(),
        json!({ "type": "session-config-option", "path": "model" })
    );
    assert_eq!(
        serde_json::to_value(&acp.client_capabilities).unwrap(),
        json!({ "_meta": { "parameterizedModelPicker": true } })
    );
}

#[test]
fn forwards_user_configurable_settings() {
    let credential_forwarding: HarnessV1CredentialForwarding =
        Arc::new(|options| async move { Ok(format!("ephemeral-{}", options.credential)) }.boxed());
    let port_endpoint = HarnessV1PortEndpoint {
        url: "wss://sandbox.example/bridge".into(),
        headers: None,
    };
    let reconnect = SandboxChannelReconnectOptions {
        max_elapsed_ms: Some(120_000),
        initial_delay_ms: Some(100),
        max_delay_ms: Some(5_000),
    };
    let settings = cursor_acp_settings(CursorHarnessSettings {
        credential_forwarding: Some(credential_forwarding.clone()),
        port: Some(4319),
        port_endpoint: Some(port_endpoint.clone()),
        startup_timeout_ms: Some(45_000),
        host_disconnect_timeout_ms: Some(60_000),
        reconnect: Some(reconnect),
        keep_alive_ms: Some(100),
        mcp_servers: Some(IndexMap::from([(
            "external".to_string(),
            json!({ "command": "external-mcp" }),
        )])),
        ..Default::default()
    });

    assert!(Arc::ptr_eq(
        settings.settings.credential_forwarding.as_ref().unwrap(),
        &credential_forwarding
    ));
    assert_eq!(settings.port, Some(4319));
    assert_eq!(settings.port_endpoint, Some(port_endpoint));
    assert_eq!(settings.startup_timeout_ms, Some(45_000));
    assert_eq!(settings.host_disconnect_timeout_ms, Some(60_000));
    assert_eq!(settings.reconnect, Some(reconnect));
    assert_eq!(settings.keep_alive_ms, Some(100));
    assert_eq!(
        serde_json::to_value(&settings.settings.mcp_servers).unwrap(),
        json!({ "external": { "command": "external-mcp" } })
    );
}

#[test]
fn applies_headers_to_configured_model_request_routes() {
    let headers = env(&[("x-tenant", "acme")]);
    let gateway_settings = cursor_acp_settings(CursorHarnessSettings {
        auth: Some(HarnessV1Authentication::AiGateway),
        ..Default::default()
    });
    assert_eq!(
        brokering(&gateway_settings, &IndexMap::new(), None, Some(&headers)),
        json!([
            {
                "match": {
                    "host": "ai-gateway.vercel.sh",
                    "path": { "startsWith": "/cursor/v1" },
                },
                "transform": { "headers": { "x-tenant": "acme" } },
            },
        ])
    );

    let auto_settings = cursor_acp_settings(CursorHarnessSettings::default());
    assert_eq!(
        brokering(&auto_settings, &IndexMap::new(), None, Some(&headers)),
        json!([
            {
                "match": { "host": "api2.cursor.sh" },
                "transform": { "headers": { "x-tenant": "acme" } },
            },
        ])
    );
}

#[test]
fn accepts_direct_and_ai_gateway_auth_and_warns_that_cursor_configuration_controls_routing() {
    for (auth, name) in [
        (
            HarnessV1Authentication::Choice(HarnessV1DirectAuthentication::Direct),
            "direct",
        ),
        (HarnessV1Authentication::AiGateway, "ai-gateway"),
    ] {
        let warning = cursor_authentication_warning(Some(&auth)).unwrap();
        assert!(warning.contains(&format!("auth: \"{name}\"")));
        assert!(warning.contains("CURSOR_API_KEY"));
        assert!(
            create_cursor(CursorHarnessSettings {
                auth: Some(auth.clone()),
                ..Default::default()
            })
            .is_ok()
        );
        let settings = cursor_acp_settings(CursorHarnessSettings {
            auth: Some(auth.clone()),
            ..Default::default()
        });
        assert_eq!(settings.settings.auth, Some(auth));
        assert!(settings.settings.provider_authentication.is_none());
    }
}

#[test]
fn accepts_auto_auth_without_warning() {
    assert_eq!(
        cursor_authentication_warning(Some(&HarnessV1Authentication::Auto)),
        None
    );
    let settings = cursor_acp_settings(CursorHarnessSettings {
        auth: Some(HarnessV1Authentication::Auto),
        ..Default::default()
    });
    assert_eq!(settings.settings.auth, Some(HarnessV1Authentication::Auto));
    assert!(settings.settings.provider_authentication.is_none());
}

#[test]
fn forwards_a_supplied_authentication_environment_for_cursor_credentials() {
    let auth =
        HarnessV1Authentication::Environment(env(&[("CURSOR_API_KEY", "programmatic-cursor-key")]));
    let settings = cursor_acp_settings(CursorHarnessSettings {
        auth: Some(auth.clone()),
        ..Default::default()
    });
    assert_eq!(settings.settings.auth, Some(auth));
    assert!(settings.settings.provider_authentication.is_none());
}

#[test]
fn classifies_cursor_mcp_calls_from_their_raw_input() {
    let settings = cursor_acp_settings(CursorHarnessSettings::default());
    let is_mcp_tool_call = &settings.settings.is_mcp_tool_call;
    assert!(
        is_mcp_tool_call(&tool_call(json!({
            "toolCallId": "call-1",
            "title": "ai-sdk-harness-tools: weather",
            "rawInput": {
                "providerIdentifier": "ai-sdk-harness-tools",
                "toolName": "weather",
                "args": { "city": "Lima" },
            },
        })))
        .unwrap()
    );
    assert!(
        !is_mcp_tool_call(&tool_call(json!({
            "toolCallId": "call-2",
            "title": "Read README.md",
            "rawInput": { "path": "README.md" },
        })))
        .unwrap()
    );
}

// resolveHostToolCall
const SERVER_NAME: &str = "ai-sdk-harness-tools";

fn resolve(value: JsonValue) -> Option<ACPResolvedHostToolCall> {
    let settings = cursor_acp_settings(CursorHarnessSettings::default());
    let tool_names = vec!["get_weather".to_string()];
    (settings.settings.resolve_host_tool_call)(ACPResolveHostToolCallOptions {
        tool_call: &tool_call(value),
        server_name: SERVER_NAME,
        tool_names: &tool_names,
    })
    .unwrap()
}

#[test]
fn resolves_the_captured_host_tool_call_update() {
    let resolved = resolve(json!({
        "toolCallId": "call-1",
        "title": "ai-sdk-harness-tools: get_weather",
        "kind": "other",
        "status": "pending",
        "rawInput": {
            "providerIdentifier": "ai-sdk-harness-tools",
            "toolName": "get_weather",
            "args": { "city": "Austin" },
        },
    }))
    .unwrap();
    assert_eq!(resolved.tool_name, "get_weather");
    assert_eq!(
        serde_json::to_value(&resolved.input).unwrap(),
        json!({ "city": "Austin" })
    );
}

#[test]
fn does_not_resolve_before_the_identifying_update_arrives() {
    assert!(
        resolve(json!({ "toolCallId": "call-1", "title": "MCP: tool", "rawInput": {} })).is_none()
    );
}

#[test]
fn does_not_resolve_a_mismatched_raw_input() {
    for raw_input in [
        json!({
            "providerIdentifier": "other-server",
            "toolName": "get_weather",
            "args": { "city": "Austin" },
        }),
        json!({
            "providerIdentifier": SERVER_NAME,
            "toolName": "unknown_tool",
            "args": { "city": "Austin" },
        }),
        json!({ "providerIdentifier": SERVER_NAME, "toolName": "get_weather", "args": null }),
    ] {
        assert!(
            resolve(json!({ "toolCallId": "call-1", "title": "Tool", "rawInput": raw_input }))
                .is_none(),
            "{raw_input}"
        );
    }
}
