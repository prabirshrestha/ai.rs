use base64::Engine as _;
use serde_json::json;

use super::*;
use crate::harness::adapters::acp::{AcpBuiltinToolMapping, HOST_TOOL_MCP_SERVER_NAME};
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::auth::HarnessDirectAuthentication;
use crate::harness::tools::BuiltinToolKind;

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn brokering(
    settings: &AcpHarnessSettings,
    env: &IndexMap<String, String>,
    sandbox_env: Option<&IndexMap<String, String>>,
    headers: Option<&IndexMap<String, String>>,
) -> Value {
    let brokering = settings.credential_brokering.as_ref().unwrap();
    serde_json::to_value(
        brokering(
            crate::harness::adapters::acp::AcpCredentialBrokeringOptions {
                env,
                sandbox_env,
                headers,
            },
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn enforces_the_cursor_acp_agent() {
    let settings = cursor_acp_settings(CursorHarnessSettings::default());
    assert!(settings.image_content);
    assert_eq!(settings.write_protected_paths, [".cursor"]);
    assert_eq!(settings.harness_id, "cursor");
    assert_eq!(
        settings.client_app,
        AcpClientApp {
            name: "ai-sdk-harness-cursor".into(),
            version: VERSION.into()
        }
    );
    assert_eq!(
        settings.source,
        AcpSource::InstallCommand {
            command: "curl https://cursor.com/install -fsS | bash".into()
        }
    );
    assert_eq!(settings.executable, "agent");
    assert_eq!(
        settings.args,
        ["--disable-auto-update", "--disable-project-configs", "acp"]
    );
    assert_eq!(settings.credential_env, Some(vec!["CURSOR_API_KEY".into()]));
    assert!(settings.authentication.is_none());
    assert!(settings.provider_authentication.is_none());
    assert_eq!(
        settings.builtin_tools.keys().collect::<Vec<_>>(),
        [
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
        ]
    );
    let write = &settings.builtin_tools["write"];
    assert_eq!(settings.builtin_tool_titles["write"], "Write");
    assert_eq!(write.native_name.as_deref(), Some("writeToolCall"));
    assert_eq!(write.kind, Some(BuiltinToolKind::Edit));
    assert_eq!(write.tool.parameters["type"], "object");
    assert_eq!(
        write.tool.parameters["properties"]["path"]["type"],
        "string"
    );
    assert_eq!(write.tool.parameters["required"], json!(["path"]));
    assert_eq!(settings.builtin_tool_titles["bash"], "Terminal");

    assert_eq!(
        brokering(
            &settings,
            &env(&[("CURSOR_API_KEY", "cursor-secret")]),
            Some(&env(&[("CURSOR_API_KEY", "sandbox-cursor-secret")])),
            None,
        ),
        json!([{
            "match": {
                "host": "api2.cursor.sh",
                "path": { "exact": "/auth/exchange_user_api_key" },
                "method": ["POST"],
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer sandbox-cursor-secret" },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer cursor-secret" } },
        }])
    );
    assert_eq!(
        brokering(&settings, &IndexMap::new(), None, None),
        json!([])
    );
    assert_eq!(
        settings.model_mapping,
        AcpModelMapping::SessionConfigOption {
            path: "model".into()
        }
    );
    assert_eq!(
        Value::Object(settings.client_capabilities.clone().unwrap()),
        json!({ "_meta": { "parameterizedModelPicker": true } })
    );
    // The settings are valid ACP settings.
    create_cursor(CursorHarnessSettings::default()).unwrap();
}

#[test]
fn forwards_user_configurable_settings() {
    let settings = cursor_acp_settings(CursorHarnessSettings {
        mcp_servers: Some(IndexMap::from([(
            "external".to_string(),
            json!({ "command": "external-mcp" }),
        )])),
        ..Default::default()
    });
    assert_eq!(
        settings.mcp_servers,
        Some(IndexMap::from([(
            "external".to_string(),
            json!({ "command": "external-mcp" }),
        )]))
    );
}

#[test]
fn applies_headers_to_configured_model_request_routes() {
    let headers = env(&[("x-tenant", "acme")]);
    let gateway = cursor_acp_settings(CursorHarnessSettings {
        auth: Some(HarnessAuthentication::AiGateway),
        ..Default::default()
    });
    assert_eq!(
        brokering(&gateway, &IndexMap::new(), None, Some(&headers)),
        json!([{
            "match": { "host": "ai-gateway.vercel.sh", "path": { "startsWith": "/cursor/v1" } },
            "transform": { "headers": { "x-tenant": "acme" } },
        }])
    );
    let auto = cursor_acp_settings(CursorHarnessSettings::default());
    assert_eq!(
        brokering(&auto, &IndexMap::new(), None, Some(&headers)),
        json!([{
            "match": { "host": "api2.cursor.sh" },
            "transform": { "headers": { "x-tenant": "acme" } },
        }])
    );
}

#[test]
fn warns_that_cursor_configuration_controls_routing() {
    for (auth, mode) in [
        (
            HarnessAuthentication::Choice(HarnessDirectAuthentication::Direct),
            "direct",
        ),
        (HarnessAuthentication::AiGateway, "ai-gateway"),
    ] {
        let warning = cursor_authentication_warning(Some(&auth)).unwrap();
        assert!(warning.contains(&format!("auth: \"{mode}\"")), "{warning}");
        assert!(warning.contains("CURSOR_API_KEY"), "{warning}");
        let settings = cursor_acp_settings(CursorHarnessSettings {
            auth: Some(auth.clone()),
            ..Default::default()
        });
        assert_eq!(settings.auth, Some(auth));
        assert!(settings.provider_authentication.is_none());
    }
    assert_eq!(
        cursor_authentication_warning(Some(&HarnessAuthentication::Auto)),
        None
    );
    assert_eq!(cursor_authentication_warning(None), None);
    let supplied =
        HarnessAuthentication::Environment(env(&[("CURSOR_API_KEY", "programmatic-cursor-key")]));
    assert_eq!(cursor_authentication_warning(Some(&supplied)), None);
    let settings = cursor_acp_settings(CursorHarnessSettings {
        auth: Some(supplied.clone()),
        ..Default::default()
    });
    assert_eq!(settings.auth, Some(supplied));
}

fn resolve(tool_call: Value) -> Option<AcpResolvedHostToolCall> {
    let tool_names = vec!["get_weather".to_string()];
    resolve_cursor_host_tool_call()(AcpResolveHostToolCallOptions {
        tool_call: &tool_call,
        server_name: HOST_TOOL_MCP_SERVER_NAME,
        tool_names: &tool_names,
    })
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
    assert_eq!(resolved.input, json!({ "city": "Austin" }));
    // Not before the identifying update arrives.
    assert!(
        resolve(json!({ "toolCallId": "call-1", "title": "MCP: tool", "rawInput": {} })).is_none()
    );
    for raw_input in [
        json!({ "providerIdentifier": "other-server", "toolName": "get_weather", "args": { "city": "Austin" } }),
        json!({ "providerIdentifier": HOST_TOOL_MCP_SERVER_NAME, "toolName": "unknown_tool", "args": { "city": "Austin" } }),
        json!({ "providerIdentifier": HOST_TOOL_MCP_SERVER_NAME, "toolName": "get_weather", "args": null }),
    ] {
        assert!(
            resolve(json!({ "toolCallId": "call-1", "title": "Tool", "rawInput": raw_input }))
                .is_none(),
            "{raw_input}"
        );
    }
}

#[test]
fn maps_cursor_tool_calls_to_builtin_tools() {
    // A Cursor `Write <path>` edit with a path-only input is the write tool.
    let settings = cursor_acp_settings(CursorHarnessSettings::default());
    let mappings: Vec<AcpBuiltinToolMapping> =
        crate::harness::adapters::acp::serialize_builtin_tools(
            &settings.builtin_tools,
            &settings.builtin_tool_titles,
        );
    let write = mappings
        .iter()
        .find(|mapping| mapping.tool_name == "write")
        .unwrap();
    assert_eq!(write.title.as_deref(), Some("Write"));
    assert_eq!(write.native_name.as_deref(), Some("writeToolCall"));
}

// resolveCursorSubscriptionEnvironment
#[tokio::test]
async fn does_not_inspect_subscriptions_for_gateway_auth() {
    assert_eq!(
        resolve_cursor_subscription_environment(
            Some(&HarnessAuthentication::AiGateway),
            env(&[("AI_GATEWAY_API_KEY", "gateway")]),
        )
        .await
        .unwrap(),
        env(&[("AI_GATEWAY_API_KEY", "gateway")])
    );
}

#[tokio::test]
async fn prefers_cursor_api_key_over_native_storage() {
    assert_eq!(
        resolve_cursor_subscription_environment(
            Some(&HarnessAuthentication::Choice(
                HarnessDirectAuthentication::Direct
            )),
            env(&[("CURSOR_API_KEY", "environment-key")]),
        )
        .await
        .unwrap(),
        env(&[("CURSOR_API_KEY", "environment-key")])
    );
}

fn cursor_home(auth: &Value) -> TempDir {
    let home = mkdtemp("cursor-auth-");
    let cursor_directory = home.path().join(".cursor");
    std::fs::create_dir_all(&cursor_directory).unwrap();
    std::fs::write(cursor_directory.join("auth.json"), auth.to_string()).unwrap();
    home
}

fn darwin(home: &TempDir) -> ReadCursorSubscriptionOptions {
    ReadCursorSubscriptionOptions {
        env: Some(env(&[("AGENT_CLI_CREDENTIAL_STORE", "file")])),
        home_directory: Some(home.path().to_string_lossy().into_owned()),
        platform: Some("darwin".into()),
    }
}

// readCursorSubscription
#[tokio::test]
async fn reads_a_file_backed_browser_access_token() {
    let home = cursor_home(&json!({
        "accessToken": "opaque-access-token",
        "refreshToken": "stored-but-not-forwarded",
    }));
    assert_eq!(
        read_cursor_subscription(darwin(&home)).await.unwrap(),
        Some("opaque-access-token".into())
    );
}

#[tokio::test]
async fn rejects_an_expiring_jwt_instead_of_inventing_a_refresh_endpoint() {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(json!({ "exp": now + 60 }).to_string());
    let home = cursor_home(&json!({
        "accessToken": format!("header.{payload}.signature"),
        "refreshToken": "refresh-token",
    }));
    let error = read_cursor_subscription(darwin(&home)).await.unwrap_err();
    assert!(error.to_string().contains("Run Cursor login again."));
}

#[tokio::test]
async fn skips_in_memory_credential_stores() {
    let home = cursor_home(&json!({ "accessToken": "token" }));
    let mut options = darwin(&home);
    options.env = Some(env(&[("AGENT_CLI_CREDENTIAL_STORE", "memory")]));
    assert_eq!(read_cursor_subscription(options).await.unwrap(), None);
}

// resolveCursorAuthPath
#[test]
fn uses_each_native_platform_path() {
    assert_eq!(
        resolve_cursor_auth_path(&IndexMap::new(), "/home/me", "linux"),
        "/home/me/.config/cursor/auth.json"
    );
    assert!(
        resolve_cursor_auth_path(
            &env(&[("APPDATA", "C:\\Users\\me\\AppData\\Roaming")]),
            "C:\\Users\\me",
            "win32",
        )
        .contains("Cursor/auth.json")
    );
    assert_eq!(
        resolve_cursor_auth_path(&IndexMap::new(), "/Users/me", "darwin"),
        "/Users/me/.cursor/auth.json"
    );
}
