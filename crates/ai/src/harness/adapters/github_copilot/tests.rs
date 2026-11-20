//! Port of `harness-github-copilot/src/github-copilot-harness.test.ts`.
//!
//! [rust] The TS tests mock `createACP` and inspect the settings it is
//! called with; these tests inspect [`github_copilot_acp_settings`].

use std::sync::{Arc, Mutex};

use regex::Regex;

use super::*;
use crate::harness::adapters::acp::{AcpCredentialBrokeringOptions, HOST_TOOL_MCP_SERVER_NAME};
use crate::harness::auth::{HarnessAuthentication, HarnessDirectAuthentication};

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn settings() -> AcpHarnessSettings {
    github_copilot_acp_settings(GitHubCopilotHarnessSettings::default())
}

fn brokering(
    settings: &AcpHarnessSettings,
    env: &IndexMap<String, String>,
    sandbox_env: Option<&IndexMap<String, String>>,
    headers: Option<&IndexMap<String, String>>,
) -> Vec<Value> {
    let brokering = settings.credential_brokering.as_ref().unwrap();
    brokering(AcpCredentialBrokeringOptions {
        env,
        sandbox_env,
        headers,
    })
    .unwrap()
    .into_iter()
    .map(|transformation| serde_json::to_value(transformation).unwrap())
    .collect()
}

async fn forward(settings: &AcpHarnessSettings, credential: &str, name: &str) -> String {
    (settings.credential_forwarding.as_ref().unwrap())(HarnessCredentialForwardingOptions {
        credential: credential.into(),
        environment_variable_name: name.into(),
    })
    .await
    .unwrap()
}

#[test]
fn configures_the_github_copilot_acp_agent() {
    let acp = settings();
    assert!(acp.image_content);
    assert_eq!(
        acp.write_protected_paths,
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
    );
    let builtin_tool_names: Vec<&str> = acp.builtin_tools.keys().map(String::as_str).collect();
    assert_eq!(
        json!({
            "harnessId": acp.harness_id,
            "clientApp": acp.client_app,
            "source": serde_json::to_value(&acp.source).unwrap(),
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
                "bash", "read_bash", "stop_bash", "list_bash", "view", "create", "edit",
                "web_fetch", "skill", "sql", "read_agent", "list_agents", "write_agent",
                "grep", "glob", "task",
            ],
            "clientApp": { "name": "ai-sdk-harness-github-copilot", "version": VERSION },
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
            "source": { "type": "npm", "package": "@github/copilot", "version": "1.0.106" },
        })
    );
    assert_eq!(
        acp.fallback_model.as_deref(),
        Some(GITHUB_COPILOT_GATEWAY_MODEL)
    );
    // The settings are valid ACP settings.
    create_github_copilot(GitHubCopilotHarnessSettings::default()).unwrap();
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
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer copilot-placeholder" },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer github-secret" } },
        }),
        json!({
            "match": {
                "host": "*.github.com",
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "token gh-placeholder" },
                }],
            },
            "transform": { "headers": { "Authorization": "token second-github-secret" } },
        }),
        json!({
            "match": {
                "host": "*.githubcopilot.com",
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer copilot-placeholder" },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer github-secret" } },
        }),
    ] {
        assert!(transformations.contains(&expected), "{expected}");
    }
    // Two schemes for each of two tokens on four hosts.
    assert_eq!(transformations.len(), 16);
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
    assert!(!hosts.contains(&"githubcopilot.com"));
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
            "headers": [{
                "key": { "exact": "Authorization" },
                "value": { "exact": "Bearer gateway-placeholder" },
            }],
        },
        "transform": {
            "headers": { "x-tenant": "acme", "Authorization": "Bearer gateway-secret" },
        },
    });
    assert_eq!(transformations, [expected]);
}

#[test]
fn applies_headers_to_direct_copilot_model_request_routes() {
    let settings = github_copilot_acp_settings(GitHubCopilotHarnessSettings {
        auth: Some(HarnessAuthentication::Choice(
            HarnessDirectAuthentication::Direct,
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
        [
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
    // An enterprise host serves Copilot itself.
    let hosts: Vec<Value> = brokering(
        &settings,
        &env(&[("GH_HOST", "https://example.ghe.com")]),
        None,
        Some(&env(&[("x-tenant", "acme")])),
    )
    .into_iter()
    .map(|transformation| transformation["match"]["host"].clone())
    .collect();
    assert_eq!(
        hosts,
        [json!("example.ghe.com"), json!("*.example.ghe.com")]
    );
}

#[test]
fn exposes_host_tools_through_an_http_mcp_server() {
    assert_eq!(
        settings().host_tool_mcp_transport,
        AcpHostToolMcpTransport::Http
    );
}

#[test]
fn routes_instructions_to_the_home_directory_copilot_instructions_md_file() {
    assert_eq!(
        settings().instruction_mapping,
        Some(AcpInstructionMapping::Filesystem {
            path: ".copilot/copilot-instructions.md".into(),
        })
    );
}

#[tokio::test]
async fn forwards_launch_settings() {
    let calls: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let recorded = calls.clone();
    let credential_forwarding: HarnessCredentialForwarding = Arc::new(move |options| {
        recorded.lock().unwrap().push((
            options.credential.clone(),
            options.environment_variable_name.clone(),
        ));
        async move { Ok(options.credential) }.boxed()
    });
    let mcp_servers =
        IndexMap::from([("docs".to_string(), json!({ "url": "https://mcp.example" }))]);
    let direct = HarnessAuthentication::Choice(HarnessDirectAuthentication::Direct);
    let settings = github_copilot_acp_settings(GitHubCopilotHarnessSettings {
        auth: Some(direct.clone()),
        credential_forwarding: Some(credential_forwarding),
        reasoning_effort: Some(GitHubCopilotReasoningEffort::High),
        mcp_servers: Some(mcp_servers.clone()),
        ..Default::default()
    });
    assert_eq!(settings.auth, Some(direct));
    assert_eq!(
        settings.args,
        [
            "--acp",
            "--stdio",
            "--no-auto-update",
            "--reasoning-effort=high"
        ]
    );
    assert_eq!(settings.mcp_servers, Some(mcp_servers));
    forward(&settings, "real-credential", "COPILOT_GITHUB_TOKEN").await;
    // A GitHub placeholder reaches the caller's forwarding reshaped.
    let placeholder = format!("aisdkhc_{}", "A".repeat(43));
    let forwarded = forward(&settings, &placeholder, "GH_TOKEN").await;
    assert!(forwarded.starts_with("gho_aisdkhc"), "{forwarded}");
    assert_eq!(
        *calls.lock().unwrap(),
        [
            (
                "real-credential".to_string(),
                "COPILOT_GITHUB_TOKEN".to_string()
            ),
            (forwarded, "GH_TOKEN".to_string()),
        ]
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
                .args
                .contains(&format!("--reasoning-effort={}", effort.as_str()))
        );
    }
}

#[test]
fn uses_loose_schemas_for_captured_copilot_tool_inputs() {
    let settings = settings();
    for (name, required) in [
        ("bash", json!(["command"])),
        (
            "task",
            json!(["name", "prompt", "agent_type", "description"]),
        ),
    ] {
        let schema = &settings.builtin_tools[name].tool.parameters;
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"], required, "{name}");
        // Unknown fields are allowed.
        assert_ne!(
            schema.get("additionalProperties"),
            Some(&json!(false)),
            "{name}"
        );
    }
    assert_eq!(settings.builtin_tool_titles["view"], "Viewing ");
    assert_eq!(settings.builtin_tool_titles["grep"], "Searching for ");
}

// resolveHostToolCall
fn resolve(value: Value) -> Option<AcpResolvedHostToolCall> {
    let tool_names = vec!["get_weather".to_string()];
    resolve_github_copilot_host_tool_call()(AcpResolveHostToolCallOptions {
        tool_call: &value,
        server_name: HOST_TOOL_MCP_SERVER_NAME,
        tool_names: &tool_names,
    })
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
    assert_eq!(resolved.input, json!({ "city": "Austin" }));
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
