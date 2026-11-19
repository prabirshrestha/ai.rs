//! Port of `acp-harness.test.ts`: credential brokering and forwarding.
//!
//! [rust] The `console.warn` spies are not ported: the Rust warnings go to
//! the shared warning sink, which these tests do not capture.

use std::sync::Arc;

use futures::FutureExt as _;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::harness::spec::authentication::{
    HarnessV1CredentialForwarding, HarnessV1CredentialForwardingOptions,
};
use crate::harness::spec::harness::HarnessV1;
use crate::harness::spec::network_sandbox::HarnessV1RequestTransformation;
use crate::harness::spec::tools::HarnessV1PermissionMode;
use crate::harness::utils::plain_error;

use super::acp_auth::ACPAuthenticationMode;
use super::acp_harness::create_acp;
use super::acp_harness_test_support::*;
use super::v1::bridge::bridge_environment::ACP_BRIDGE_CONFIGURATION_ENV;
use super::v1::settings::{
    ACPAuthenticationFile, ACPCredentialBrokering, ACPGatewayProviderAuthentication,
    ACPProviderAuthentication, ACPV1Settings,
};

type Env = IndexMap<String, String>;

fn env(pairs: &[(&str, &str)]) -> Env {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn placeholder_pattern() -> regex::Regex {
    regex::Regex::new(r"^aisdkhc_[A-Za-z0-9_-]{43}$").unwrap()
}

fn transformation(
    host: &str,
    path_starts_with: Option<&str>,
    placeholder: &str,
    secret: &str,
) -> HarnessV1RequestTransformation {
    let mut matcher = json!({
        "host": host,
        "headers": [{
            "key": { "exact": "Authorization" },
            "value": { "exact": format!("Bearer {placeholder}") },
        }],
    });
    if let Some(prefix) = path_starts_with {
        matcher["path"] = json!({ "startsWith": prefix });
    }
    serde_json::from_value(json!({
        "match": matcher,
        "transform": { "headers": { "Authorization": format!("Bearer {secret}") } },
    }))
    .unwrap()
}

/// A brokering call: `env`, `sandboxEnv`, `headers`.
type BrokeringCall = (Env, Option<Env>, Option<Env>);

/// `credentialBrokering` building one transformation for `variable`.
fn brokering(
    host: &'static str,
    path_starts_with: Option<&'static str>,
    variable: &'static str,
) -> (ACPCredentialBrokering, Arc<Mutex<Vec<BrokeringCall>>>) {
    let calls: Arc<Mutex<Vec<BrokeringCall>>> = Arc::default();
    let recorded = calls.clone();
    (
        Arc::new(move |options| {
            recorded.lock().push((
                options.env.clone(),
                options.sandbox_env.cloned(),
                options.headers.cloned(),
            ));
            let Some(placeholder) = options.sandbox_env.and_then(|env| env.get(variable)) else {
                return Vec::new();
            };
            vec![transformation(
                host,
                path_starts_with,
                placeholder,
                options.env.get(variable).map(String::as_str).unwrap_or(""),
            )]
        }),
        calls,
    )
}

type ForwardingCall = (String, String);

fn forwarding(
    value: impl Fn(&HarnessV1CredentialForwardingOptions) -> Result<String, String>
    + Send
    + Sync
    + 'static,
) -> (
    HarnessV1CredentialForwarding,
    Arc<Mutex<Vec<ForwardingCall>>>,
) {
    let calls: Arc<Mutex<Vec<ForwardingCall>>> = Arc::default();
    let recorded = calls.clone();
    (
        Arc::new(move |options| {
            recorded.lock().push((
                options.credential.clone(),
                options.environment_variable_name.clone(),
            ));
            let result = value(&options).map_err(plain_error);
            async move { result }.boxed()
        }),
        calls,
    )
}

fn harness_with_env(
    settings: ACPV1Settings,
    bridge: &FakeBridge,
    process_env: Env,
) -> super::v1::harness::ACPV1Harness {
    let mut settings = harness_settings(settings, bridge);
    settings.process_env = Some(process_env);
    create_acp(settings).unwrap()
}

fn brokering_sandbox() -> FakeSandbox {
    FakeSandbox {
        supports_transformations: true,
        ..FakeSandbox::default()
    }
}

fn gateway_mode() -> Option<ACPAuthenticationMode> {
    Some(ACPAuthenticationMode::AiGateway)
}

fn gateway_env(pairs: Value) -> Option<ACPProviderAuthentication> {
    Some(ACPProviderAuthentication {
        gateway: ACPGatewayProviderAuthentication {
            env: serde_json::from_value(pairs).unwrap(),
        },
    })
}

fn bridge_configuration(spawn: &Spawn) -> Value {
    serde_json::from_str(&spawn.env[ACP_BRIDGE_CONFIGURATION_ENV]).unwrap()
}

#[tokio::test]
async fn brokers_host_resolved_direct_credentials_before_launching_the_sandbox_bridge() {
    let bridge = FakeBridge::default();
    let sandbox = brokering_sandbox();
    let (credential_brokering, brokering_calls) =
        brokering("provider.example", Some("/v1"), "PROVIDER_API_KEY");
    let (credential_forwarding, forwarding_calls) =
        forwarding(|options| Ok(format!("ephemeral-{}", options.environment_variable_name)));
    let mut settings = agent_settings("direct-brokered-acp");
    settings.forward_env = Some(vec![
        "PROVIDER_BASE_URL".into(),
        "HOST_PROVIDER_BASE_URL".into(),
    ]);
    settings.credential_env = Some(vec!["PROVIDER_API_KEY".into()]);
    settings.credential_brokering = Some(credential_brokering);
    settings.credential_forwarding = Some(credential_forwarding);
    settings.resolve_authentication_environment = Some(Arc::new(|_| {
        async {
            Ok(env(&[
                ("PROVIDER_API_KEY", "direct-secret"),
                ("PROVIDER_BASE_URL", "https://provider.example/v1"),
            ]))
        }
        .boxed()
    }));
    settings.env = Some(env(&[("STATIC_SETTING", "literal-value")]));
    let mut process_env = test_env();
    process_env.insert(
        "HOST_PROVIDER_BASE_URL".into(),
        "https://host-provider.example/v1".into(),
    );
    let harness = harness_with_env(settings, &bridge, process_env);

    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    let expected_env = env(&[
        ("PROVIDER_API_KEY", "direct-secret"),
        ("PROVIDER_BASE_URL", "https://provider.example/v1"),
        ("HOST_PROVIDER_BASE_URL", "https://host-provider.example/v1"),
        ("STATIC_SETTING", "literal-value"),
    ]);
    let expected_sandbox_env = env(&[
        ("PROVIDER_API_KEY", "ephemeral-PROVIDER_API_KEY"),
        ("PROVIDER_BASE_URL", "https://provider.example/v1"),
        ("HOST_PROVIDER_BASE_URL", "https://host-provider.example/v1"),
        ("STATIC_SETTING", "literal-value"),
    ]);
    assert!(
        brokering_calls
            .lock()
            .contains(&(expected_env, Some(expected_sandbox_env), None)),
        "{:?}",
        brokering_calls.lock()
    );
    assert_eq!(
        sandbox.record.lock().transformations,
        vec![vec![transformation(
            "provider.example",
            Some("/v1"),
            "ephemeral-PROVIDER_API_KEY",
            "direct-secret",
        )]]
    );
    let spawn_env = sandbox.spawns()[0].env.clone();
    assert_eq!(spawn_env["PROVIDER_API_KEY"], "ephemeral-PROVIDER_API_KEY");
    assert_eq!(
        spawn_env["PROVIDER_BASE_URL"],
        "https://provider.example/v1"
    );
    assert_eq!(
        spawn_env["HOST_PROVIDER_BASE_URL"],
        "https://host-provider.example/v1"
    );
    assert_eq!(spawn_env["STATIC_SETTING"], "literal-value");
    let forwarding_calls = forwarding_calls.lock().clone();
    assert_eq!(forwarding_calls.len(), 1);
    assert!(placeholder_pattern().is_match(&forwarding_calls[0].0));
    assert_eq!(forwarding_calls[0].1, "PROVIDER_API_KEY");
    assert!(!format!("{spawn_env:?}").contains("direct-secret"));

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn materializes_private_authentication_files_with_brokered_credentials() {
    let bridge = FakeBridge::default();
    let sandbox = brokering_sandbox();
    let file_calls: Arc<Mutex<Vec<(Env, Env, bool)>>> = Arc::default();
    let (credential_forwarding, _) = forwarding(|_| Ok("sandbox-secret".into()));
    let mut settings = agent_settings("authentication-file-acp");
    settings.credential_env = Some(vec!["PROVIDER_API_KEY".into()]);
    settings.credential_brokering = Some(Arc::new(|_| Vec::new()));
    settings.credential_forwarding = Some(credential_forwarding);
    settings.authentication_files = Some({
        let file_calls = file_calls.clone();
        Arc::new(move |options| {
            file_calls.lock().push((
                options.env.clone(),
                options.sandbox_env.clone(),
                options.credential_brokering_available,
            ));
            vec![ACPAuthenticationFile {
                path: ".config/provider/auth.json".into(),
                content: json!({ "token": options.sandbox_env["PROVIDER_API_KEY"] }).to_string(),
            }]
        })
    });
    let mut process_env = test_env();
    process_env.insert("PROVIDER_API_KEY".into(), "host-secret".into());
    let harness = harness_with_env(settings, &bridge, process_env);

    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    let file_calls = file_calls.lock().clone();
    assert_eq!(file_calls.len(), 1);
    assert_eq!(file_calls[0].0["PROVIDER_API_KEY"], "host-secret");
    assert_eq!(file_calls[0].1["PROVIDER_API_KEY"], "sandbox-secret");
    assert!(file_calls[0].2);
    assert!(sandbox.writes().contains(&(
        "/home/agent/.config/provider/auth.json".to_string(),
        "{\"token\":\"sandbox-secret\"}".to_string()
    )));
    assert!(
        sandbox
            .runs()
            .contains(&"chmod 600 -- '/home/agent/.config/provider/auth.json'".to_string())
    );
    assert!(!format!("{:?}", sandbox.writes()).contains("host-secret"));

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn uses_resolved_gateway_profile_values_for_brokering_without_serializing_the_credential() {
    let bridge = FakeBridge::default();
    let sandbox = brokering_sandbox();
    let (credential_brokering, brokering_calls) =
        brokering("gateway.example", Some("/v1"), "PROVIDER_API_KEY");
    let (credential_forwarding, forwarding_calls) =
        forwarding(|options| Ok(format!("ephemeral-{}", options.environment_variable_name)));
    let mut settings = agent_settings("gateway-brokered-acp");
    settings.auth = gateway_mode();
    settings.forward_env = Some(Vec::new());
    settings.credential_env = Some(vec!["PROVIDER_API_KEY".into()]);
    settings.credential_brokering = Some(credential_brokering);
    settings.credential_forwarding = Some(credential_forwarding);
    settings.provider_authentication = gateway_env(json!({
        "PROVIDER_API_KEY": { "$source": "gateway-api-key" },
        "PROVIDER_BASE_URL": { "$source": "gateway-base-url", "ensureSuffix": "/v1" },
    }));
    let mut process_env = test_env();
    process_env.insert("PROVIDER_API_KEY".into(), "direct-secret".into());
    process_env.insert("AI_GATEWAY_API_KEY".into(), "gateway-secret".into());
    process_env.insert(
        "AI_GATEWAY_BASE_URL".into(),
        "https://gateway.example".into(),
    );
    let harness = harness_with_env(settings, &bridge, process_env);
    let mut options = start_options("session-1", sandbox.network());
    options.headers = Some(env(&[("x-tenant", "acme")]));

    let session = start(&harness, options).await;

    assert!(
        brokering_calls.lock().contains(&(
            env(&[
                ("PROVIDER_API_KEY", "gateway-secret"),
                ("PROVIDER_BASE_URL", "https://gateway.example/v1"),
            ]),
            Some(env(&[
                ("PROVIDER_API_KEY", "ephemeral-PROVIDER_API_KEY"),
                ("PROVIDER_BASE_URL", "https://gateway.example/v1"),
            ])),
            Some(env(&[("x-tenant", "acme")])),
        )),
        "{:?}",
        brokering_calls.lock()
    );
    assert_eq!(
        sandbox.record.lock().transformations,
        vec![vec![transformation(
            "gateway.example",
            Some("/v1"),
            "ephemeral-PROVIDER_API_KEY",
            "gateway-secret",
        )]]
    );
    let spawn = sandbox.spawns()[0].clone();
    assert_eq!(spawn.env["PROVIDER_API_KEY"], "ephemeral-PROVIDER_API_KEY");
    assert_eq!(spawn.env["PROVIDER_BASE_URL"], "https://gateway.example/v1");
    assert!(!spawn.env.contains_key("AI_SDK_ACP_GATEWAY_API_KEY"));
    assert!(!format!("{:?}", spawn.env).contains("gateway-secret"));
    assert_match_object(
        &bridge_configuration(&spawn),
        &json!({
            "providerAuthentication": { "type": "ai-gateway" },
            "providerEnvironment": {},
        }),
    );
    let forwarding_calls = forwarding_calls.lock().clone();
    assert_eq!(forwarding_calls.len(), 1);
    assert!(placeholder_pattern().is_match(&forwarding_calls[0].0));
    assert_eq!(forwarding_calls[0].1, "PROVIDER_API_KEY");

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn uses_the_exposed_environment_variable_name_for_brokered_gateway_credentials() {
    let bridge = FakeBridge::default();
    let sandbox = brokering_sandbox();
    let (credential_brokering, brokering_calls) =
        brokering("gateway.example", None, "RUNTIME_PROVIDER_API_KEY");
    let (credential_forwarding, forwarding_calls) =
        forwarding(|options| Ok(options.credential.clone()));
    let mut settings = agent_settings("renamed-gateway-credential-acp");
    settings.auth = gateway_mode();
    settings.forward_env = Some(Vec::new());
    settings.credential_env = Some(Vec::new());
    settings.credential_brokering = Some(credential_brokering);
    settings.credential_forwarding = Some(credential_forwarding);
    settings.provider_authentication = gateway_env(json!({
        "RUNTIME_PROVIDER_API_KEY": { "$source": "gateway-api-key" },
    }));
    let mut process_env = test_env();
    process_env.insert("AI_GATEWAY_API_KEY".into(), "gateway-secret".into());
    let harness = harness_with_env(settings, &bridge, process_env);

    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    let first_call = brokering_calls.lock()[0].clone();
    let sandbox_env = first_call.1.clone().unwrap();
    let placeholder = sandbox_env["RUNTIME_PROVIDER_API_KEY"].clone();
    assert!(placeholder_pattern().is_match(&placeholder));
    assert!(brokering_calls.lock().contains(&(
        env(&[("RUNTIME_PROVIDER_API_KEY", "gateway-secret")]),
        Some(sandbox_env),
        None,
    )));
    assert_eq!(
        sandbox.record.lock().transformations,
        vec![vec![transformation(
            "gateway.example",
            None,
            &placeholder,
            "gateway-secret"
        )]]
    );
    assert_eq!(
        *forwarding_calls.lock(),
        vec![(placeholder.clone(), "RUNTIME_PROVIDER_API_KEY".to_string())]
    );
    let spawn = sandbox.spawns()[0].clone();
    assert_eq!(spawn.env["RUNTIME_PROVIDER_API_KEY"], placeholder);
    assert!(!spawn.env.contains_key("AI_SDK_ACP_GATEWAY_API_KEY"));
    assert!(!format!("{:?}", spawn.env).contains("gateway-secret"));

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn persists_sandbox_credentials_and_reinstalls_the_same_matchers_across_process_resumes() {
    let bridge = FakeBridge::default();
    let sandbox = brokering_sandbox();
    let create_harness = || {
        let (credential_brokering, _) = brokering("provider.example", None, "PROVIDER_API_KEY");
        let mut settings = agent_settings("resumable-brokered-acp");
        settings.forward_env = Some(Vec::new());
        settings.credential_env = Some(vec!["PROVIDER_API_KEY".into()]);
        settings.credential_brokering = Some(credential_brokering);
        let mut process_env = test_env();
        process_env.insert("PROVIDER_API_KEY".into(), "direct-secret".into());
        harness_with_env(settings, &bridge, process_env)
    };

    let first_session = start(
        &create_harness(),
        start_options("session-1", sandbox.network()),
    )
    .await;
    let resume_from = first_session.do_detach().await.unwrap();
    let persisted = resume_from.data["sandboxCredentialEnvironment"]["PROVIDER_API_KEY"]
        .as_str()
        .unwrap()
        .to_string();

    assert!(placeholder_pattern().is_match(&persisted));
    assert_eq!(sandbox.spawns()[0].env["PROVIDER_API_KEY"], persisted);
    assert!(
        !serde_json::to_string(&resume_from)
            .unwrap()
            .contains("direct-secret")
    );

    let mut options = start_options("session-1", sandbox.network());
    options.resume_from =
        Some(serde_json::from_value(serde_json::to_value(&resume_from).unwrap()).unwrap());
    let resumed_session = start(&create_harness(), options).await;

    assert_eq!(sandbox.spawns().len(), 1);
    let transformations = sandbox.record.lock().transformations.clone();
    assert_eq!(transformations.len(), 2);
    assert_eq!(transformations[1], transformations[0]);
    assert_eq!(
        transformations[1],
        vec![transformation(
            "provider.example",
            None,
            &persisted,
            "direct-secret"
        )]
    );

    resumed_session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn uses_new_placeholders_after_a_brokered_cold_restore_changes_credential_names() {
    let bridge = FakeBridge::default();
    let sandbox = brokering_sandbox();
    let credential_brokering: ACPCredentialBrokering = Arc::new(|options| {
        ["GITHUB_TOKEN", "GITLAB_TOKEN"]
            .into_iter()
            .filter_map(|name| {
                let secret = options.env.get(name)?;
                let placeholder = options.sandbox_env?.get(name)?;
                Some(transformation("tokens.example", None, placeholder, secret))
            })
            .collect()
    });
    let create_harness = |process_env: Env| {
        let mut settings = agent_settings("changing-credentials-acp");
        settings.forward_env = Some(Vec::new());
        settings.credential_env = Some(vec!["GITHUB_TOKEN".into(), "GITLAB_TOKEN".into()]);
        settings.permission_mode_mapping = Some(permission_mode_mapping());
        settings.credential_brokering = Some(credential_brokering.clone());
        harness_with_env(settings, &bridge, process_env)
    };
    let mut first_env = test_env();
    first_env.insert("GITHUB_TOKEN".into(), "old-github-secret".into());
    let harness = create_harness(first_env);
    let mut options = start_options("session-1", sandbox.network());
    options.permission_mode = Some(HarnessV1PermissionMode::AllowEdits);

    let first_session = start(&harness, options).await;
    let first_placeholder = sandbox.spawns()[0].env["GITHUB_TOKEN"].clone();
    assert!(placeholder_pattern().is_match(&first_placeholder));
    let (turn_options, _) = prompt_options("Remember this.");
    let first_turn = first_session.do_prompt_turn(turn_options).await.unwrap();
    let first_channel = bridge.socket(0);
    first_channel.emit(json!({ "type": "bridge-thread", "threadId": "acp-session-1" }));
    first_channel.emit(finish_frame());
    done(&*first_turn).await.unwrap();
    let resume_from = first_session.do_stop().await.unwrap();
    assert_eq!(
        resume_from.data["sandboxCredentialEnvironment"],
        json!({ "GITHUB_TOKEN": first_placeholder })
    );

    let mut second_env = test_env();
    second_env.insert("GITLAB_TOKEN".into(), "new-gitlab-secret".into());
    let harness = Arc::new(create_harness(second_env));
    let mut options = start_options("session-1", sandbox.network());
    options.resume_from = Some(resume_from);
    options.permission_mode = Some(HarnessV1PermissionMode::AllowEdits);
    let resumed = tokio::spawn({
        let harness = harness.clone();
        async move { harness.do_start(options).await }
    });
    wait_for(|| bridge.count() == 2 && bridge.socket(1).sent().len() == 1).await;
    let spawns = sandbox.spawns();
    let new_placeholder = spawns[1].env["GITLAB_TOKEN"].clone();
    assert!(placeholder_pattern().is_match(&new_placeholder));
    assert!(!spawns[1].env.contains_key("GITHUB_TOKEN"));
    assert!(!format!("{:?}", spawns[1].env).contains("new-gitlab-secret"));
    assert_eq!(
        sandbox.record.lock().transformations[1],
        vec![transformation(
            "tokens.example",
            None,
            &new_placeholder,
            "new-gitlab-secret"
        )]
    );
    emit_cold_restoration(&bridge.socket(1), "resume", "acp-session-1");
    let resumed_session = resumed.await.unwrap().unwrap();
    let next_state = resumed_session.do_stop().await.unwrap();
    assert_eq!(
        next_state.data["sandboxCredentialEnvironment"],
        json!({ "GITLAB_TOKEN": new_placeholder })
    );
}

#[tokio::test]
async fn preserves_real_credential_forwarding_when_additive_transformations_are_unavailable() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let (credential_brokering, brokering_calls) =
        brokering("provider.example", None, "PROVIDER_API_KEY");
    let mut settings = agent_settings("legacy-credentials-acp");
    settings.forward_env = Some(Vec::new());
    settings.credential_env = Some(vec!["PROVIDER_API_KEY".into()]);
    settings.credential_brokering = Some(credential_brokering);
    let mut process_env = test_env();
    process_env.insert("PROVIDER_API_KEY".into(), "legacy-secret".into());
    let harness = harness_with_env(settings, &bridge, process_env);

    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    assert!(brokering_calls.lock().is_empty());
    assert_eq!(sandbox.spawns()[0].env["PROVIDER_API_KEY"], "legacy-secret");

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn customizes_direct_and_gateway_credentials_under_their_sandbox_environment_names() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let (credential_brokering, brokering_calls) =
        brokering("provider.example", None, "PROVIDER_API_KEY");
    let (credential_forwarding, forwarding_calls) = forwarding(|options| {
        Ok(if options.credential == "gateway-secret" {
            "ephemeral-gateway-credential".into()
        } else {
            "ephemeral-direct-credential".into()
        })
    });
    let mut settings = agent_settings("custom-forwarding-acp");
    settings.auth = gateway_mode();
    settings.forward_env = Some(Vec::new());
    settings.credential_env = Some(vec!["PROVIDER_API_KEY".into()]);
    settings.credential_brokering = Some(credential_brokering);
    settings.credential_forwarding = Some(credential_forwarding);
    settings.provider_authentication = gateway_env(json!({
        "RUNTIME_PROVIDER_API_KEY": { "$source": "gateway-api-key" },
    }));
    let mut process_env = test_env();
    process_env.insert("PROVIDER_API_KEY".into(), "direct-secret".into());
    process_env.insert("AI_GATEWAY_API_KEY".into(), "gateway-secret".into());
    let harness = harness_with_env(settings, &bridge, process_env);

    let session = start(&harness, start_options("session-1", sandbox.network())).await;

    assert!(brokering_calls.lock().is_empty());
    let forwarding_calls = forwarding_calls.lock().clone();
    assert_eq!(
        forwarding_calls[0],
        ("direct-secret".to_string(), "PROVIDER_API_KEY".to_string())
    );
    assert_eq!(
        forwarding_calls[1],
        (
            "gateway-secret".to_string(),
            "RUNTIME_PROVIDER_API_KEY".to_string()
        )
    );
    let spawn = sandbox.spawns()[0].clone();
    assert_eq!(spawn.env["PROVIDER_API_KEY"], "ephemeral-direct-credential");
    assert_eq!(
        spawn.env["RUNTIME_PROVIDER_API_KEY"],
        "ephemeral-gateway-credential"
    );
    assert!(!spawn.env.contains_key("AI_SDK_ACP_GATEWAY_API_KEY"));
    assert_match_object(
        &bridge_configuration(&spawn),
        &json!({ "providerEnvironment": {} }),
    );

    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn aborts_before_spawning_when_credential_forwarding_fails() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox::default();
    let (credential_forwarding, _) = forwarding(|_| Err("credential forwarding failed".into()));
    let mut settings = agent_settings("failed-forwarding-acp");
    settings.forward_env = Some(Vec::new());
    settings.credential_env = Some(vec!["PROVIDER_API_KEY".into()]);
    settings.credential_brokering = Some(Arc::new(|_| Vec::new()));
    settings.credential_forwarding = Some(credential_forwarding);
    let mut process_env = test_env();
    process_env.insert("PROVIDER_API_KEY".into(), "direct-secret".into());
    let harness = harness_with_env(settings, &bridge, process_env);

    let error = harness
        .do_start(start_options("session-1", sandbox.network()))
        .await
        .err()
        .unwrap();
    assert_eq!(error.to_string(), "credential forwarding failed");
    assert!(sandbox.spawns().is_empty());
}

#[tokio::test]
async fn aborts_before_attach_or_spawn_when_credential_brokering_fails() {
    let bridge = FakeBridge::default();
    let sandbox = FakeSandbox {
        transformation_error: Some("request transformation installation failed".into()),
        ..brokering_sandbox()
    };
    let (credential_brokering, _) = brokering("provider.example", None, "PROVIDER_API_KEY");
    let mut settings = agent_settings("failed-brokering-acp");
    settings.forward_env = Some(Vec::new());
    settings.credential_env = Some(vec!["PROVIDER_API_KEY".into()]);
    settings.credential_brokering = Some(credential_brokering);
    let mut process_env = test_env();
    process_env.insert("PROVIDER_API_KEY".into(), "direct-secret".into());
    let harness = harness_with_env(settings, &bridge, process_env);

    let error = harness
        .do_start(start_options("session-1", sandbox.network()))
        .await
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        "request transformation installation failed"
    );
    assert!(sandbox.spawns().is_empty());
    assert_eq!(bridge.count(), 0);
}
