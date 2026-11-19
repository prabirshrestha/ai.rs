//! Port of `harness-github-copilot/src/github-copilot-harness.test.ts`.
//!
//! [rust] The TS tests mock `createACP` and inspect the settings it is
//! called with; these tests inspect [`github_copilot_acp_settings`] instead.

use std::sync::Arc;

use futures::FutureExt as _;
use indexmap::IndexMap;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{Value as JsonValue, json};

use super::github_copilot_harness::{
    GitHubCopilotHarnessSettings, GitHubCopilotReasoningEffort, create_github_copilot,
    github_copilot_acp_settings,
};
use crate::harness::adapters::acp::v1::settings::{
    ACPCredentialBrokeringOptions, ACPHostToolMCPTransport, ACPInstructionMapping,
    ACPNonTextContentType, ACPSource,
};
use crate::harness::adapters::acp::{
    ACPHarnessSettings, ACPResolveHostToolCallOptions, ACPResolvedHostToolCall, ACPToolCall,
};
use crate::harness::spec::harness::HarnessV1 as _;
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
use crate::harness::spec::{
    HarnessV1Authentication, HarnessV1CredentialForwarding, HarnessV1CredentialForwardingOptions,
    HarnessV1DirectAuthentication,
};
use crate::harness::utils::SandboxChannelReconnectOptions;

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn settings() -> ACPHarnessSettings {
    github_copilot_acp_settings(GitHubCopilotHarnessSettings::default())
}

fn brokering(
    settings: &ACPHarnessSettings,
    env: &IndexMap<String, String>,
    sandbox_env: Option<&IndexMap<String, String>>,
    headers: Option<&IndexMap<String, String>>,
) -> Vec<JsonValue> {
    let brokering = settings.settings.credential_brokering.as_ref().unwrap();
    brokering(ACPCredentialBrokeringOptions {
        env,
        sandbox_env,
        headers,
    })
    .unwrap()
    .into_iter()
    .map(|transformation| serde_json::to_value(transformation).unwrap())
    .collect()
}

async fn forward(settings: &ACPHarnessSettings, credential: &str, name: &str) -> String {
    (settings.settings.credential_forwarding.as_ref().unwrap())(
        HarnessV1CredentialForwardingOptions {
            credential: credential.into(),
            environment_variable_name: name.into(),
        },
    )
    .await
    .unwrap()
}

fn tool_call(value: JsonValue) -> ACPToolCall {
    serde_json::from_value(value).unwrap()
}

#[test]
fn configures_the_github_copilot_acp_agent() {
    let settings = settings();
    let acp = &settings.settings;
    assert_eq!(
        acp.non_text_content_types,
        Some(vec![ACPNonTextContentType::Image])
    );
    assert_eq!(
        acp.write_protected_paths,
        Some(
            [
                ".github/hooks",
                ".github/mcp.json",
                ".github/copilot",
                ".github/skills",
                ".github/agents",
                ".github/extensions",
                ".claude",
                ".mcp.json",
                ".vscode/mcp.json",
            ]
            .map(String::from)
            .to_vec()
        )
    );
    let source_type = serde_json::to_value(&acp.source).unwrap()["type"].clone();
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
            "sourceType": source_type,
            "executable": acp.executable,
            "args": acp.args,
            "forwardEnv": acp.forward_env,
            "credentialEnv": acp.credential_env,
            "hasCredentialBrokering": acp.credential_brokering.is_some(),
            "hasAuthenticationResolver": acp.resolve_authentication_environment.is_some(),
            "providerAuthentication": acp.provider_authentication,
            "modelMapping": acp.model_mapping,
            "skillsDirectory": acp.skills_directory,
            "builtinToolNames": builtin_tool_names,
        }),
        json!({
            "args": ["--acp", "--stdio", "--no-auto-update"],
            "builtinToolNames": [
                "bash",
                "read_bash",
                "stop_bash",
                "list_bash",
                "view",
                "create",
                "edit",
                "web_fetch",
                "skill",
                "sql",
                "read_agent",
                "list_agents",
                "write_agent",
                "grep",
                "glob",
                "task",
            ],
            "clientApp": {
                "name": "ai-sdk-harness-github-copilot",
                "version": super::VERSION,
            },
            "credentialEnv": ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"],
            "executable": "copilot",
            "forwardEnv": ["COPILOT_GH_HOST", "GH_HOST"],
            "harnessId": "github-copilot",
            "hasAuthenticationResolver": true,
            "hasCredentialBrokering": true,
            "modelMapping": { "path": "model", "type": "session-config-option" },
            "providerAuthentication": {
                "gateway": {
                    "env": {
                        "COPILOT_MODEL": "openai/gpt-5.5",
                        "COPILOT_PROVIDER_API_KEY": { "$source": "gateway-api-key" },
                        "COPILOT_PROVIDER_BASE_URL": {
                            "$source": "gateway-base-url",
                            "ensureSuffix": "/v1",
                        },
                        "COPILOT_PROVIDER_HEADERS": {
                            "$source": "client-app",
                            "prefix": "x-client-app: ",
                        },
                        "COPILOT_PROVIDER_TYPE": "openai",
                        "COPILOT_PROVIDER_WIRE_API": "responses",
                    },
                },
            },
            "skillsDirectory": ".copilot/skills",
            "sourceType": "npm-locked",
            "version": "v1",
        })
    );

    let ACPSource::NpmLocked(source) = &acp.source else {
        panic!("Expected a locked NPM source.");
    };
    assert!(
        source
            .pnpm_workspace_yaml
            .as_deref()
            .unwrap()
            .contains("'@github/copilot")
    );
}

#[test]
fn brokers_github_credentials_using_their_sandbox_placeholders() {
    let transformations = brokering(
        &settings(),
        &env(&[
            ("COPILOT_GITHUB_TOKEN", "github-secret"),
            ("GH_TOKEN", "second-github-secret"),
        ]),
        Some(&env(&[
            ("COPILOT_GITHUB_TOKEN", "copilot-placeholder"),
            ("GH_TOKEN", "gh-placeholder"),
        ])),
        None,
    );
    for expected in [
        json!({
            "match": {
                "host": "github.com",
                "headers": [
                    {
                        "key": { "exact": "Authorization" },
                        "value": { "exact": "Bearer copilot-placeholder" },
                    },
                ],
            },
            "transform": { "headers": { "Authorization": "Bearer github-secret" } },
        }),
        json!({
            "match": {
                "host": "*.github.com",
                "headers": [
                    {
                        "key": { "exact": "Authorization" },
                        "value": { "exact": "token gh-placeholder" },
                    },
                ],
            },
            "transform": { "headers": { "Authorization": "token second-github-secret" } },
        }),
    ] {
        assert!(transformations.contains(&expected), "{expected}");
    }
}

#[tokio::test]
async fn formats_each_token_sandbox_placeholder_as_a_github_oauth_token() {
    let pattern = Regex::new(r"^gho_aisdkhc[0-9a-f]{29}$").unwrap();
    for name in ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"] {
        let credential = forward(&settings(), &format!("aisdkhc_{}", "A".repeat(43)), name).await;
        assert!(pattern.is_match(&credential), "{name}: {credential}");
        assert_eq!(credential.len(), 40);
    }
}

#[tokio::test]
async fn does_not_format_real_credentials_or_unrelated_placeholders() {
    let settings = settings();
    assert_eq!(
        forward(&settings, "gho_real-credential", "COPILOT_GITHUB_TOKEN").await,
        "gho_real-credential"
    );
    let placeholder = format!("aisdkhc_{}", "A".repeat(43));
    assert_eq!(
        forward(&settings, &placeholder, "COPILOT_PROVIDER_API_KEY").await,
        placeholder
    );
}

#[test]
fn brokers_github_credentials_for_the_configured_enterprise_host() {
    let transformations = brokering(
        &settings(),
        &env(&[
            ("COPILOT_GH_HOST", "example.ghe.com"),
            ("GITHUB_TOKEN", "github-secret"),
        ]),
        Some(&env(&[
            ("COPILOT_GH_HOST", "example.ghe.com"),
            ("GITHUB_TOKEN", "github-placeholder"),
        ])),
        None,
    );
    let hosts: Vec<&str> = transformations
        .iter()
        .map(|transformation| transformation["match"]["host"].as_str().unwrap())
        .collect();
    assert!(hosts.contains(&"example.ghe.com"));
    assert!(hosts.contains(&"*.example.ghe.com"));
    assert!(!hosts.contains(&"*.github.com"));
}

#[test]
fn brokers_the_ai_gateway_provider_credential() {
    let transformations = brokering(
        &settings(),
        &env(&[
            ("COPILOT_PROVIDER_API_KEY", "gateway-secret"),
            ("COPILOT_PROVIDER_BASE_URL", "https://gateway.example/v1"),
        ]),
        Some(&env(&[
            ("COPILOT_PROVIDER_API_KEY", "gateway-placeholder"),
            ("COPILOT_PROVIDER_BASE_URL", "https://gateway.example/v1"),
        ])),
        Some(&env(&[("x-tenant", "acme")])),
    );
    let expected = json!({
        "match": {
            "host": "gateway.example",
            "path": { "startsWith": "/v1" },
            "headers": [
                {
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer gateway-placeholder" },
                },
            ],
        },
        "transform": {
            "headers": {
                "x-tenant": "acme",
                "Authorization": "Bearer gateway-secret",
            },
        },
    });
    assert!(transformations.contains(&expected), "{transformations:?}");
}

#[test]
fn applies_headers_to_direct_copilot_model_request_routes() {
    let settings = github_copilot_acp_settings(GitHubCopilotHarnessSettings {
        auth: Some(HarnessV1Authentication::Choice(
            HarnessV1DirectAuthentication::Direct,
        )),
        ..Default::default()
    });
    assert_eq!(
        brokering(
            &settings,
            &IndexMap::new(),
            None,
            Some(&env(&[("x-tenant", "acme")]))
        ),
        vec![
            json!({
                "match": { "host": "githubcopilot.com" },
                "transform": { "headers": { "x-tenant": "acme" } },
            }),
            json!({
                "match": { "host": "*.githubcopilot.com" },
                "transform": { "headers": { "x-tenant": "acme" } },
            }),
        ]
    );
}

#[test]
fn exposes_host_tools_through_an_http_mcp_server() {
    assert_eq!(
        settings().settings.host_tool_mcp_transport,
        Some(ACPHostToolMCPTransport::Http)
    );
}

#[test]
fn routes_instructions_to_the_home_directory_copilot_instructions_md_file() {
    assert_eq!(
        settings().settings.instruction_mapping,
        Some(ACPInstructionMapping::Filesystem {
            path: ".copilot/copilot-instructions.md".into(),
        })
    );
}

#[tokio::test]
async fn forwards_launch_and_bridge_settings() {
    let calls: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let recorded = calls.clone();
    let credential_forwarding: HarnessV1CredentialForwarding = Arc::new(move |options| {
        recorded.lock().push((
            options.credential.clone(),
            options.environment_variable_name.clone(),
        ));
        async move { Ok(options.credential) }.boxed()
    });
    let mcp_servers =
        IndexMap::from([("docs".to_string(), json!({ "url": "https://mcp.example" }))]);
    let port_endpoint = HarnessV1PortEndpoint {
        url: "wss://sandbox.example/bridge".into(),
        headers: None,
    };
    let reconnect = SandboxChannelReconnectOptions {
        max_elapsed_ms: Some(120_000),
        initial_delay_ms: Some(100),
        max_delay_ms: Some(5_000),
    };
    let direct = HarnessV1Authentication::Choice(HarnessV1DirectAuthentication::Direct);

    let settings = github_copilot_acp_settings(GitHubCopilotHarnessSettings {
        auth: Some(direct.clone()),
        credential_forwarding: Some(credential_forwarding),
        reasoning_effort: Some(GitHubCopilotReasoningEffort::High),
        mcp_servers: Some(mcp_servers.clone()),
        port: Some(4319),
        port_endpoint: Some(port_endpoint.clone()),
        startup_timeout_ms: Some(45_000),
        host_disconnect_timeout_ms: Some(60_000),
        reconnect: Some(reconnect),
        keep_alive_ms: Some(100),
    });

    assert_eq!(settings.settings.auth, Some(direct));
    assert_eq!(
        settings.settings.args,
        Some(
            [
                "--acp",
                "--stdio",
                "--no-auto-update",
                "--reasoning-effort=high"
            ]
            .map(String::from)
            .to_vec()
        )
    );
    assert_eq!(settings.settings.mcp_servers, Some(mcp_servers));
    assert_eq!(settings.port, Some(4319));
    assert_eq!(settings.port_endpoint, Some(port_endpoint));
    assert_eq!(settings.startup_timeout_ms, Some(45_000));
    assert_eq!(settings.host_disconnect_timeout_ms, Some(60_000));
    assert_eq!(settings.reconnect, Some(reconnect));
    assert_eq!(settings.keep_alive_ms, Some(100));
    forward(&settings, "real-credential", "COPILOT_GITHUB_TOKEN").await;
    assert_eq!(
        *calls.lock(),
        vec![(
            "real-credential".to_string(),
            "COPILOT_GITHUB_TOKEN".to_string()
        )]
    );
}

#[test]
fn encodes_each_reasoning_effort() {
    use GitHubCopilotReasoningEffort as E;
    for effort in [
        E::None,
        E::Minimal,
        E::Low,
        E::Medium,
        E::High,
        E::Xhigh,
        E::Max,
    ] {
        let settings = github_copilot_acp_settings(GitHubCopilotHarnessSettings {
            reasoning_effort: Some(effort),
            ..Default::default()
        });
        assert!(
            settings
                .settings
                .args
                .unwrap()
                .contains(&format!("--reasoning-effort={}", effort.as_str()))
        );
    }
}

#[test]
fn classifies_built_in_and_configured_mcp_title_prefixes() {
    let settings = github_copilot_acp_settings(GitHubCopilotHarnessSettings {
        mcp_servers: Some(IndexMap::from([(
            "docs".to_string(),
            json!({ "command": "docs-mcp" }),
        )])),
        ..Default::default()
    });
    let is_mcp_tool_call = &settings.settings.is_mcp_tool_call;
    let classify = |id: &str, title: &str| {
        is_mcp_tool_call(&tool_call(json!({ "toolCallId": id, "title": title }))).unwrap()
    };
    assert!(classify("github", "github-mcp-server-search_code"));
    assert!(classify("configured", "docs-search"));
    assert!(!classify("native", "Viewing README.md"));
}

#[test]
fn uses_loose_schemas_for_captured_copilot_tool_inputs() {
    let settings = settings();
    let tools = settings.builtin_tools.as_ref().unwrap();
    let validate = |name: &str, input: JsonValue| {
        (tools[name].tool.input_schema.validate.as_ref().unwrap())(&input)
    };
    assert!(
        validate(
            "bash",
            json!({
                "command": "pwd",
                "description": "Print the working directory",
                "mode": "async",
                "futureField": true,
            })
        )
        .is_ok()
    );
    assert!(
        validate(
            "task",
            json!({
                "name": "review",
                "prompt": "Review this repository",
                "agent_type": "code-review",
                "description": "Review code",
                "context_tier": "long_context",
                "mode": "background",
            })
        )
        .is_ok()
    );
}

#[test]
fn create_github_copilot_wraps_the_acp_harness() {
    let harness = create_github_copilot(GitHubCopilotHarnessSettings::default()).unwrap();
    assert_eq!(harness.harness_id(), "github-copilot");
    assert_eq!(harness.builtin_tools().len(), 16);
}

// resolveHostToolCall
const SERVER_NAME: &str = "ai-sdk-harness-tools";

fn resolve(value: JsonValue) -> Option<ACPResolvedHostToolCall> {
    let settings = settings();
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
        "title": "ai-sdk-harness-tools-get_weather",
        "kind": "read",
        "status": "pending",
        "rawInput": { "city": "Austin" },
    }))
    .unwrap();
    assert_eq!(resolved.tool_name, "get_weather");
    assert_eq!(
        serde_json::to_value(&resolved.input).unwrap(),
        json!({ "city": "Austin" })
    );
}

#[test]
fn does_not_resolve_the_shortened_permission_request_title() {
    assert!(
        resolve(json!({
            "toolCallId": "call-1",
            "title": "get_weather",
            "rawInput": { "city": "Austin" },
        }))
        .is_none()
    );
}

#[test]
fn does_not_resolve_a_mismatched_tool_call() {
    for tool_call in [
        json!({ "title": "github-mcp-server-get_weather", "rawInput": { "city": "Austin" } }),
        json!({
            "title": "ai-sdk-harness-tools-get_weather",
            "name": "get_weather",
            "rawInput": { "city": "Austin" },
        }),
        json!({ "title": "ai-sdk-harness-tools-get_weather", "rawInput": null }),
    ] {
        let mut value = json!({ "toolCallId": "call-1" });
        value
            .as_object_mut()
            .unwrap()
            .extend(tool_call.as_object().unwrap().clone());
        assert!(resolve(value).is_none(), "{tool_call}");
    }
}
