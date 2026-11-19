//! Port of `codex-harness.test.ts`.
//!
//! [rust] TS mocks `SandboxChannel` and records what the adapter hands it.
//! These tests run the real channel over [`FakeBridge`] sockets, so bridge
//! events are dispatched as frames on the open socket and must be valid
//! stream parts. Checks TS makes on the mocked channel itself (its
//! constructor options) and on `console.warn` have no Rust counterpart; the
//! tests check the behavior those options produce where it is observable.
//! `mintBridgeToken` and `getPortUrl` are deprecated and not ported.

use std::sync::Arc;

use futures::FutureExt as _;
use indexmap::IndexMap;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{Value, json};

use super::codex_bootstrap::get_codex_bootstrap;
use super::codex_bridge_protocol::CodexReasoningEffort;
use super::codex_harness::{CodexHarness, CodexHarnessSettings, CodexHost, create_codex_with_host};
use crate::harness::adapters::bridge_test_support::*;
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::errors::HarnessError;
use crate::harness::provider::DataContent;
use crate::harness::sdk::messages::{
    FilePart, FilePartData, TextPart, UserContent, UserContentPart, UserModelMessage,
};
use crate::harness::spec::harness::{HarnessV1, HarnessV1Prompt, HarnessV1StartOptions};
use crate::harness::spec::network_sandbox::HarnessV1PortEndpoint;
use crate::harness::spec::tools::{
    HarnessV1BuiltinToolFiltering, HarnessV1BuiltinToolUseKind, HarnessV1PermissionMode,
};
use crate::harness::spec::{HarnessV1Authentication, HarnessV1CredentialForwarding};

const PLACEHOLDER: &str = r"^aisdkhc_[A-Za-z0-9_-]{43}$";

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn auth(pairs: &[(&str, &str)]) -> Option<super::CodexAuthenticationMode> {
    Some(HarnessV1Authentication::Environment(env(pairs)))
}

fn is_match(pattern: &str, value: &str) -> bool {
    Regex::new(pattern).unwrap().is_match(value)
}

/// `beforeEach`: an empty temporary `HOME`, the fake `ws` module and the
/// fake sandbox (`fakeNetworkSandboxSessionForStartupSuccess`), which
/// supports additive request transformations.
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
                url: "ws://127.0.0.1:1".into(),
                headers: None,
            },
            supports_transformations: true,
            ..Default::default()
        },
        home: mkdtemp("codex-harness-"),
    }
}

impl Fixture {
    fn harness(&self, settings: CodexHarnessSettings) -> CodexHarness {
        let home = self.home.path_string();
        create_codex_with_host(
            settings,
            CodexHost {
                web_socket_constructor: Some(self.bridge.constructor()),
                process_env: Some(env(&[("HOME", &home), ("USERPROFILE", &home)])),
            },
        )
    }

    fn options(&self, session_id: &str) -> HarnessV1StartOptions {
        let mut options = start_options(session_id, self.sandbox.network());
        options.session_work_dir = format!("/vercel/sandbox/codex-{session_id}");
        options
    }

    fn starts(&self) -> Vec<Value> {
        self.bridge
            .sent()
            .into_iter()
            .filter(|frame| frame["type"] == "start")
            .collect()
    }

    /// `sentMessages.at(-1)`, once the turn's deferred `start` was sent.
    async fn last_start(&self, count: usize) -> Value {
        wait_for(|| self.starts().len() >= count).await;
        self.starts().pop().expect("no start message was sent")
    }

    fn spawn_env(&self) -> IndexMap<String, String> {
        self.sandbox.spawns()[0].env.clone()
    }
}

fn settings() -> CodexHarnessSettings {
    CodexHarnessSettings::default()
}

/// Starts a session, runs one prompt turn and returns its `start` frame.
async fn start_frame(
    fx: &Fixture,
    harness: &CodexHarness,
    options: HarnessV1StartOptions,
) -> Value {
    let session = start(harness, options).await;
    let (prompt, _) = prompt_options("Hello");
    let _control = session.do_prompt_turn(prompt).await.unwrap();
    let start = fx.last_start(fx.starts().len() + 1).await;
    session.do_destroy().await.unwrap();
    start
}

fn recording_forwarding(
    forwarded: Arc<Mutex<Vec<(String, String)>>>,
    value: impl Fn(&str, &str) -> String + Send + Sync + 'static,
) -> Option<HarnessV1CredentialForwarding> {
    Some(Arc::new(move |options| {
        forwarded.lock().push((
            options.credential.clone(),
            options.environment_variable_name.clone(),
        ));
        let value = value(&options.credential, &options.environment_variable_name);
        async move { Ok(value) }.boxed()
    }))
}

fn transformations(fx: &Fixture) -> Value {
    serde_json::to_value(&fx.sandbox.record.lock().transformations).unwrap()
}

#[test]
fn declares_the_harness_id_and_builtin_tools() {
    let fx = fixture();
    let harness = fx.harness(settings());
    assert_eq!(harness.harness_id(), "codex");
    assert_eq!(harness.specification_version(), "harness-v1");
    assert!(!harness.supports_builtin_tool_approvals());
    let tools = harness.builtin_tools();
    assert_eq!(
        tools.keys().map(String::as_str).collect::<Vec<_>>(),
        ["bash", "webSearch", "apply_patch", "view_image"]
    );
    assert_eq!(tools["bash"].native_name.as_deref(), Some("shell"));
    assert_eq!(
        serde_json::to_value(tools["bash"].common_name).unwrap(),
        json!("bash")
    );
    assert_eq!(
        tools["webSearch"].native_name.as_deref(),
        Some("web_search")
    );
    assert_eq!(
        serde_json::to_value(tools["webSearch"].common_name).unwrap(),
        json!("webSearch")
    );
    assert_eq!(
        tools["apply_patch"].tool_use_kind,
        Some(HarnessV1BuiltinToolUseKind::Edit)
    );
    assert_eq!(
        tools["view_image"].tool_use_kind,
        Some(HarnessV1BuiltinToolUseKind::Readonly)
    );
}

#[tokio::test]
async fn waits_for_codex_to_accept_steering_and_rejects_messages_after_the_turn_finishes() {
    let fx = fixture();
    let session = start(&fx.harness(settings()), fx.options("s1")).await;
    let (options, _) = prompt_options("Weather in Paris?");
    let control = session.do_prompt_turn(options).await.unwrap();
    let socket = fx.bridge.last_socket();
    let steering = tokio::spawn({
        let control: Arc<dyn crate::harness::spec::harness::HarnessV1PromptControl> =
            Arc::from(control);
        let submit = control.clone();
        async move {
            let result = submit
                .submit_user_message("Actually, Paris, Texas.".into())
                .await
                .map_err(|error| error.to_string());
            (control, result)
        }
    });
    wait_for(|| {
        socket
            .sent()
            .iter()
            .any(|frame| frame["type"] == "user-message")
    })
    .await;
    let request = socket
        .sent()
        .into_iter()
        .find(|frame| frame["type"] == "user-message")
        .unwrap();
    assert_eq!(request["text"], "Actually, Paris, Texas.");
    assert!(request["messageId"].is_string());
    assert_eq!(request.as_object().unwrap().len(), 3);
    socket.emit(json!({
        "type": "user-message-response",
        "messageId": request["messageId"],
        "accepted": true,
    }));
    let (control, result) = steering.await.unwrap();
    assert_eq!(result, Ok(()));

    socket.emit(finish_frame());
    done(&*control).await.unwrap();
    let error = control
        .submit_user_message("Too late.".into())
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("no longer accepting user messages"),
        "{error}"
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn rejects_built_in_permission_modes_other_than_allow_all() {
    let fx = fixture();
    let mut options = fx.options("s1");
    options.permission_mode = Some(HarnessV1PermissionMode::AllowEdits);
    let error = fx
        .harness(settings())
        .do_start(options)
        .await
        .err()
        .unwrap();
    assert!(HarnessError::is_capability_unsupported(&*error), "{error}");
}

#[tokio::test]
async fn throws_harness_capability_unsupported_error_when_the_network_sandbox_session_exposes_no_ports()
 {
    let mut fx = fixture();
    fx.sandbox.ports = Vec::new();
    let error = fx
        .harness(settings())
        .do_start(fx.options("s1"))
        .await
        .err()
        .unwrap();
    assert!(HarnessError::is_capability_unsupported(&*error), "{error}");
}

#[tokio::test]
async fn requires_explicit_bridge_settings_for_a_basic_sandbox_session() {
    let fx = fixture();
    let basic_options = || {
        let mut options = start_options("s1", fx.sandbox.basic());
        options.session_work_dir = "/vercel/sandbox/codex-s1".into();
        options
    };
    let error = fx
        .harness(CodexHarnessSettings {
            port_endpoint: Some(HarnessV1PortEndpoint {
                url: "ws://127.0.0.1:4319".into(),
                headers: None,
            }),
            ..settings()
        })
        .do_start(basic_options())
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("explicit `port`"), "{error}");

    let error = fx
        .harness(CodexHarnessSettings {
            port: Some(4319),
            ..settings()
        })
        .do_start(basic_options())
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
    let mut fx = fixture();
    fx.sandbox.bridge_port_endpoint = HarnessV1PortEndpoint {
        url: "ws://unused.example".into(),
        headers: None,
    };
    let port_endpoint = HarnessV1PortEndpoint {
        url: "wss://sandbox.example/bridge".into(),
        headers: Some(env(&[("authorization", "endpoint-token")])),
    };
    let mut options = start_options("s1", fx.sandbox.basic());
    options.session_work_dir = "/vercel/sandbox/codex-s1".into();
    let session = start(
        &fx.harness(CodexHarnessSettings {
            port: Some(4319),
            port_endpoint: Some(port_endpoint.clone()),
            ..settings()
        }),
        options,
    )
    .await;

    assert!(
        fx.sandbox
            .runs()
            .contains(&"printf \"%s\" \"$HOME\"".to_string())
    );
    let (url, headers) = fx.bridge.calls.lock().last().cloned().unwrap();
    assert!(url.contains("wss://sandbox.example/bridge"), "{url}");
    assert_eq!(headers, port_endpoint.headers);
    let resume_from = session.do_detach().await.unwrap();
    let data = resume_from.data.as_object().unwrap();
    assert_eq!(data.keys().collect::<Vec<_>>(), ["bridge"]);
    assert_eq!(
        resume_from.data["bridge"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["port", "privateKey", "lastSeenEventId"]
    );
    assert_match_object(
        &resume_from.data,
        &json!({ "bridge": { "port": 4319, "lastSeenEventId": 0 } }),
    );
    assert!(resume_from.data["bridge"]["privateKey"].is_string());
}

#[test]
fn rejects_lifecycle_state_with_a_legacy_bridge_token() {
    let fx = fixture();
    let harness = fx.harness(settings());
    let schema = harness.lifecycle_state_schema().unwrap();
    assert!(
        schema
            .validate_value(&json!({
                "bridge": { "port": 4319, "token": "legacy-token", "lastSeenEventId": 0 },
            }))
            .is_err()
    );
}

#[tokio::test]
async fn quotes_dynamic_startup_paths_in_shell_commands() {
    let fx = fixture();
    let mut options = fx.options("s1");
    options.session_id = "s1; env > /tmp/leak #".into();
    options.session_work_dir = "/vercel/sandbox/codex-s1; env > /tmp/workdir-leak #".into();
    let session = start(&fx.harness(settings()), options).await;

    let session_state_dir =
        "/home/vercel-sandbox/.ai-sdk-harness/.agent-runs/s1%3B%20env%20%3E%20%2Ftmp%2Fleak%20%23";
    assert!(fx.sandbox.runs().contains(&format!(
        "mkdir -p '/vercel/sandbox/codex-s1; env > /tmp/workdir-leak #' '{session_state_dir}/bridge'"
    )));
    assert_eq!(
        fx.sandbox
            .spawns()
            .into_iter()
            .map(|spawn| spawn.command)
            .collect::<Vec<_>>(),
        [format!(
            "node '/home/vercel-sandbox/.ai-sdk-harness/.harness-bootstrap/codex/bridge.mjs' --workdir '/vercel/sandbox/codex-s1; env > /tmp/workdir-leak #' --bridge-state-dir '{session_state_dir}/bridge'"
        )]
    );
    let spawn_env = fx.spawn_env();
    // [rust] TS stubs the package version as `0.0.0-test`.
    assert_eq!(
        spawn_env["AI_SDK_HARNESS_CLIENT_APP"],
        format!("ai-sdk-harness-codex/{}", super::VERSION)
    );
    assert!(is_match(
        r"^[A-Za-z0-9_-]{43}$",
        &spawn_env["BRIDGE_AUTH_PUBLIC_KEY"]
    ));
    assert!(!spawn_env.contains_key("BRIDGE_CHANNEL_TOKEN"));
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn passes_host_disconnect_timeout_ms_to_the_bridge() {
    let timeout_flag = Regex::new(r"--host-disconnect-timeout-ms '?(\d+)'?").unwrap();
    for (host_disconnect_timeout_ms, expected) in [(Some(60_000), Some("60000")), (None, None)] {
        let fx = fixture();
        let session = start(
            &fx.harness(CodexHarnessSettings {
                host_disconnect_timeout_ms,
                ..settings()
            }),
            fx.options("s1"),
        )
        .await;
        let command = fx.sandbox.spawns()[0].command.clone();
        let found = timeout_flag
            .captures(&command)
            .map(|captures| captures[1].to_string());
        assert_eq!(found.as_deref(), expected);
        session.do_destroy().await.unwrap();
    }
}

#[tokio::test]
async fn falls_back_to_the_default_model_overridden_by_the_per_turn_model() {
    let fx = fixture();
    let session = start(&fx.harness(settings()), fx.options("s1")).await;
    let (options, _) = prompt_options("Hello");
    let _first = session.do_prompt_turn(options).await.unwrap();
    assert_match_object(
        &fx.last_start(1).await,
        &json!({ "type": "start", "model": "gpt-5.5" }),
    );

    let (mut options, _) = prompt_options("Hello again");
    options.settings.model = Some("agent-model".into());
    let _second = session.do_prompt_turn(options).await.unwrap();
    assert_match_object(
        &fx.last_start(2).await,
        &json!({ "type": "start", "model": "agent-model" }),
    );
    session.do_destroy().await.unwrap();
}

const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

#[tokio::test]
async fn forwards_inline_image_content_to_the_bridge() {
    use base64::Engine as _;
    let fx = fixture();
    let session = start(&fx.harness(settings()), fx.options("s1")).await;
    let (mut options, _) = prompt_options("");
    options.prompt = HarnessV1Prompt::Message(UserModelMessage {
        content: UserContent::Parts(vec![
            UserContentPart::Text(TextPart {
                text: "What is in this image?".into(),
                provider_options: None,
            }),
            UserContentPart::File(FilePart {
                data: FilePartData::Content(DataContent::Bytes(
                    base64::engine::general_purpose::STANDARD
                        .decode(PNG)
                        .unwrap(),
                )),
                filename: None,
                media_type: "image/png".into(),
                provider_options: None,
            }),
        ]),
        provider_options: None,
    });
    let _control = session.do_prompt_turn(options).await.unwrap();

    assert_match_object(
        &fx.last_start(1).await,
        &json!({
            "type": "start",
            "prompt": [
                { "type": "text", "text": "What is in this image?" },
                { "type": "image", "mediaType": "image/png", "data": PNG },
            ],
        }),
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn forwards_built_in_tool_filtering_to_the_bridge() {
    let fx = fixture();
    let builtin_tool_filtering: HarnessV1BuiltinToolFiltering =
        serde_json::from_value(json!({ "mode": "deny", "toolNames": ["bash"] })).unwrap();
    let mut options = fx.options("s1");
    options.builtin_tool_filtering = Some(builtin_tool_filtering);
    let start = start_frame(&fx, &fx.harness(settings()), options).await;
    assert_eq!(
        start["builtinToolFiltering"],
        json!({ "mode": "deny", "toolNames": ["bash"] })
    );
}

#[tokio::test]
async fn passes_headers_to_the_bridge_for_gateway_and_direct_auth() {
    for (credential, session_id) in [
        (("AI_GATEWAY_API_KEY", "gateway-key"), "gateway"),
        (("OPENAI_API_KEY", "openai-key"), "direct"),
    ] {
        let fx = fixture();
        let mut options = fx.options(session_id);
        options.headers = Some(env(&[("x-tenant", "acme")]));
        let start = start_frame(
            &fx,
            &fx.harness(CodexHarnessSettings {
                auth: auth(&[credential]),
                ..settings()
            }),
            options,
        )
        .await;
        assert_match_object(
            &start,
            &json!({ "type": "start", "headers": { "x-tenant": "acme" } }),
        );
    }
}

#[tokio::test]
async fn brokers_credentials_when_the_sandbox_supports_additive_request_transformations() {
    let fx = fixture();
    let forwarded: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let session = start(
        &fx.harness(CodexHarnessSettings {
            auth: auth(&[
                ("OPENAI_API_KEY", "openai-secret"),
                ("OPENAI_BASE_URL", "https://openai.example/v1"),
            ]),
            credential_forwarding: recording_forwarding(forwarded.clone(), |_, name| {
                format!("ephemeral-{name}")
            }),
            ..settings()
        }),
        fx.options("s1"),
    )
    .await;

    assert_eq!(
        transformations(&fx),
        json!([[{
            "match": {
                "host": "openai.example",
                "path": { "startsWith": "/v1" },
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer ephemeral-CODEX_API_KEY" },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer openai-secret" } },
        }]])
    );
    let forwarded = forwarded.lock().clone();
    assert_eq!(forwarded.len(), 1);
    assert!(is_match(PLACEHOLDER, &forwarded[0].0));
    assert_eq!(forwarded[0].1, "CODEX_API_KEY");
    assert_eq!(fx.spawn_env()["CODEX_API_KEY"], "ephemeral-CODEX_API_KEY");
    assert!(!format!("{:?}", fx.spawn_env()).contains("openai-secret"));
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn customizes_real_credentials_when_request_transformations_are_unavailable() {
    let mut fx = fixture();
    fx.sandbox.supports_transformations = false;
    let forwarded: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let session = start(
        &fx.harness(CodexHarnessSettings {
            auth: auth(&[("OPENAI_API_KEY", "openai-secret")]),
            credential_forwarding: recording_forwarding(forwarded.clone(), |_, _| {
                "caller-managed-credential".into()
            }),
            ..settings()
        }),
        fx.options("s1"),
    )
    .await;

    assert_eq!(
        *forwarded.lock(),
        [("openai-secret".to_string(), "CODEX_API_KEY".to_string())]
    );
    assert_eq!(fx.spawn_env()["CODEX_API_KEY"], "caller-managed-credential");
    assert!(!format!("{:?}", fx.spawn_env()).contains("openai-secret"));
    session.do_destroy().await.unwrap();
    // [rust] The fallback warning for identity forwarding goes to stderr;
    // its wording is checked by the credential brokering utility tests.
}

#[tokio::test]
async fn adds_a_gateway_placeholder_when_resuming_direct_authentication() {
    let fx = fixture();
    let mut options = fx.options("s1");
    options.resume_from = Some(
        serde_json::from_value(json!({
            "type": "resume-session",
            "harnessId": "codex",
            "specificationVersion": "harness-v1",
            "data": {
                "sandboxCredentialEnvironment": { "CODEX_API_KEY": "saved-codex-placeholder" },
            },
        }))
        .unwrap(),
    );
    let session = start(
        &fx.harness(CodexHarnessSettings {
            auth: auth(&[("AI_GATEWAY_API_KEY", "current-gateway-secret")]),
            ..settings()
        }),
        options,
    )
    .await;

    let spawn_env = fx.spawn_env();
    assert_eq!(spawn_env["CODEX_API_KEY"], "saved-codex-placeholder");
    assert!(is_match(PLACEHOLDER, &spawn_env["AI_GATEWAY_API_KEY"]));
    assert!(!format!("{spawn_env:?}").contains("current-gateway-secret"));
    assert_eq!(
        transformations(&fx),
        json!([[{
            "match": {
                "host": "ai-gateway.vercel.sh",
                "path": { "startsWith": "/v1" },
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer saved-codex-placeholder" },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer current-gateway-secret" } },
        }]])
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn configures_the_standard_openai_url_for_brokered_direct_auth() {
    let fx = fixture();
    let session = start(
        &fx.harness(CodexHarnessSettings {
            auth: auth(&[("OPENAI_API_KEY", "openai-secret")]),
            ..settings()
        }),
        fx.options("s1"),
    )
    .await;

    let sandbox_credential = fx.spawn_env()["CODEX_API_KEY"].clone();
    assert!(is_match(PLACEHOLDER, &sandbox_credential));
    assert_eq!(
        transformations(&fx),
        json!([[{
            "match": {
                "host": "api.openai.com",
                "path": { "startsWith": "/v1" },
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": format!("Bearer {sandbox_credential}") },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer openai-secret" } },
        }]])
    );
    assert_eq!(
        fx.spawn_env()["OPENAI_BASE_URL"],
        "https://api.openai.com/v1"
    );
    session.do_destroy().await.unwrap();
}

#[tokio::test]
async fn sends_configured_mcp_servers_to_the_bridge() {
    let fx = fixture();
    let mcp_servers: IndexMap<String, Value> = IndexMap::from([(
        "context7".to_string(),
        json!({ "url": "https://mcp.context7.com/mcp" }),
    )]);
    let start = start_frame(
        &fx,
        &fx.harness(CodexHarnessSettings {
            mcp_servers: Some(mcp_servers),
            ..settings()
        }),
        fx.options("s1"),
    )
    .await;
    assert_match_object(
        &start,
        &json!({
            "type": "start",
            "mcpServers": { "context7": { "url": "https://mcp.context7.com/mcp" } },
        }),
    );
}

#[tokio::test]
async fn sends_configured_codex_config_to_the_bridge() {
    let fx = fixture();
    let codex_config: IndexMap<String, Value> = IndexMap::from([
        ("model_verbosity".to_string(), json!("low")),
        ("features".to_string(), json!({ "multi_agent": false })),
    ]);
    let start = start_frame(
        &fx,
        &fx.harness(CodexHarnessSettings {
            codex_config: Some(codex_config),
            reasoning_effort: Some(CodexReasoningEffort::High),
            ..settings()
        }),
        fx.options("s1"),
    )
    .await;
    assert_match_object(
        &start,
        &json!({
            "type": "start",
            "codexConfig": { "model_verbosity": "low", "features": { "multi_agent": false } },
        }),
    );
}

#[tokio::test]
async fn keeps_the_bridge_private_key_on_the_host_and_reuses_it_when_attaching() {
    let fx = fixture();
    let harness = fx.harness(CodexHarnessSettings {
        reconnect: Some(crate::harness::utils::SandboxChannelReconnectOptions {
            max_elapsed_ms: Some(120_000),
            initial_delay_ms: Some(100),
            max_delay_ms: Some(5_000),
        }),
        keep_alive_ms: Some(100),
        ..settings()
    });
    let session = start(&harness, fx.options("s1")).await;

    let resume_from = session.do_detach().await.unwrap();
    let private_key = resume_from.data["bridge"]["privateKey"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(is_match(r"^[A-Za-z0-9_-]+$", &private_key));
    assert!(resume_from.data["bridge"].get("token").is_none());
    let recorded = format!(
        "{:?} {:?} {:?}",
        fx.sandbox.spawns(),
        fx.sandbox.writes(),
        fx.sandbox.runs()
    );
    assert!(!recorded.contains(&private_key));

    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from);
    let attached = start(&harness, options).await;
    // [rust] TS checks the reconnect and keep-alive options handed to the
    // mocked channel; here the attached channel answers the same key's
    // challenge without a second spawn.
    assert_eq!(fx.bridge.count(), 2);
    assert_eq!(fx.sandbox.spawns().len(), 1);
    let attached_resume_from = attached.do_detach().await.unwrap();
    assert_match_object(
        &attached_resume_from.data,
        &json!({ "bridge": { "privateKey": private_key } }),
    );
}

#[tokio::test]
async fn passes_port_endpoint_headers_to_fresh_and_attached_web_socket_connections() {
    let mut fx = fixture();
    let headers = env(&[("E2B-Traffic-Access-Token", "traffic-token")]);
    fx.sandbox.bridge_port_endpoint = HarnessV1PortEndpoint {
        url: "wss://sandbox.example/bridge?existing=value".into(),
        headers: Some(headers.clone()),
    };
    let harness = fx.harness(settings());
    let session = start(&harness, fx.options("s1")).await;

    let resume_from = session.do_detach().await.unwrap();
    let mut options = fx.options("s1");
    options.resume_from = Some(resume_from);
    let attached = start(&harness, options).await;

    let calls = fx.bridge.calls.lock().clone();
    assert_eq!(calls.len(), 2);
    for (url, call_headers) in &calls {
        assert_eq!(call_headers.as_ref(), Some(&headers));
        assert!(url.contains("existing=value"));
        assert!(!url.contains("agent_bridge_token"));
    }
    attached.do_detach().await.unwrap();
}

// describe('getBootstrap')

#[tokio::test]
async fn returns_a_recipe_with_the_expected_harness_id_and_bootstrap_dir() {
    let fx = fixture();
    let harness = fx.harness(settings());
    assert!(harness.supports_get_bootstrap());
    let recipe = harness.get_bootstrap(None).await.unwrap();
    assert_eq!(recipe.harness_id, "codex");
    assert_eq!(recipe.bootstrap_dir, ".harness-bootstrap/codex");
}

#[tokio::test]
async fn includes_bridge_mjs_package_json_and_pnpm_lock_yaml_under_the_bootstrap_dir() {
    let fx = fixture();
    let recipe = fx.harness(settings()).get_bootstrap(None).await.unwrap();
    let mut paths: Vec<&str> = recipe.files.iter().map(|file| file.path.as_str()).collect();
    paths.sort();
    assert_eq!(
        paths,
        [
            ".harness-bootstrap/codex/bridge.mjs",
            ".harness-bootstrap/codex/package.json",
            ".harness-bootstrap/codex/pnpm-lock.yaml",
        ]
    );
    for file in &recipe.files {
        assert!(!file.content.is_empty());
    }
}

#[tokio::test]
async fn declares_a_pnpm_install_command_for_the_bootstrap_cwd() {
    let fx = fixture();
    let recipe = fx.harness(settings()).get_bootstrap(None).await.unwrap();
    assert_eq!(
        recipe
            .commands
            .iter()
            .map(|command| command.command.as_str())
            .collect::<Vec<_>>(),
        ["pnpm install --frozen-lockfile --store-dir .pnpm-store"]
    );
}

#[test]
fn caches_the_recipe_across_calls() {
    assert!(Arc::ptr_eq(&get_codex_bootstrap(), &get_codex_bootstrap()));
}

#[tokio::test]
async fn shares_the_getter_across_configured_harness_instances() {
    // [rust] Every harness instance reads the one cached recipe.
    let fx = fixture();
    let first = fx.harness(CodexHarnessSettings {
        reasoning_effort: Some(CodexReasoningEffort::Low),
        ..settings()
    });
    let second = fx.harness(CodexHarnessSettings {
        reasoning_effort: Some(CodexReasoningEffort::High),
        web_search: Some(true),
        ..settings()
    });
    assert_eq!(
        first.get_bootstrap(None).await.unwrap(),
        second.get_bootstrap(None).await.unwrap()
    );
}
