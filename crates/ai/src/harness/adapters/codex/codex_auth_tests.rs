//! Port of `codex-auth.test.ts`.
//!
//! [rust] TS mocks the `@ai-sdk/harness/utils` credential-store readers;
//! these tests hand `read_codex_subscription` a mocked `execFile`, which
//! the platform readers run, and check the command each one issues.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::FutureExt as _;
use indexmap::IndexMap;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{Value, json};

use super::codex_auth::*;
use super::codex_subscription::*;
use crate::harness::adapters::bridge_test_support::assert_match_object;
use crate::harness::adapters::test_temp_dir::mkdtemp;
use crate::harness::errors::HarnessResult;
use crate::harness::spec::network_sandbox::{
    HarnessV1RequestTransformation, HarnessV1RequestTransformationSources,
};
use crate::harness::spec::{HarnessV1Authentication, HarnessV1DirectAuthentication};
use crate::harness::utils::native_subscription::jwt::create_jwt;
use crate::harness::utils::native_subscription::test_exec::MockExecFile;
use crate::harness::utils::oauth_access_token::OAuthFetchResponse;

const DIRECT: CodexAuthenticationMode =
    HarnessV1Authentication::Choice(HarnessV1DirectAuthentication::Direct);
const GATEWAY: CodexAuthenticationMode = HarnessV1Authentication::AiGateway;
const ACCOUNT: &str = r"^cli\|[0-9a-f]{16}$";

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn now_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// `jwt(expiresAt)`.
fn jwt(expires_at: u64) -> String {
    create_jwt(&json!({ "exp": expires_at }))
}

fn assert_matches(pattern: &str, value: &str) {
    assert!(
        Regex::new(pattern).unwrap().is_match(value),
        "{value:?} does not match {pattern}"
    );
}

// describe('resolveCodexEnv')

#[test]
fn uses_direct_openai_auth_when_selected() {
    let env = resolve_codex_env(
        Some(&DIRECT),
        &env(&[
            ("OPENAI_API_KEY", "sk-direct"),
            ("OPENAI_ORGANIZATION", "org_1"),
        ]),
    );
    assert_eq!(env["CODEX_API_KEY"], "sk-direct");
    assert_eq!(env["OPENAI_ORGANIZATION"], "org_1");
}

fn gateway_env(key: &str, base_url: &str) -> IndexMap<String, String> {
    env(&[
        ("AI_GATEWAY_API_KEY", key),
        ("CODEX_API_KEY", key),
        ("AI_GATEWAY_BASE_URL", base_url),
        ("OPENAI_BASE_URL", base_url),
    ])
}

#[test]
fn routes_through_the_gateway_when_gateway_mode_is_selected() {
    assert_eq!(
        resolve_codex_env(Some(&GATEWAY), &env(&[("AI_GATEWAY_API_KEY", "gw-key")])),
        gateway_env("gw-key", "https://ai-gateway.vercel.sh/v1")
    );
}

#[test]
fn appends_v1_to_gateway_base_urls_for_codex() {
    assert_eq!(
        resolve_codex_env(
            Some(&GATEWAY),
            &env(&[
                ("AI_GATEWAY_BASE_URL", "https://gw.example"),
                ("VERCEL_OIDC_TOKEN", "oidc-env"),
            ])
        ),
        gateway_env("oidc-env", "https://gw.example/v1")
    );
}

#[test]
fn preserves_v1_on_gateway_base_urls() {
    assert_eq!(
        resolve_codex_env(
            Some(&GATEWAY),
            &env(&[
                ("AI_GATEWAY_BASE_URL", "https://gw.example/v1"),
                ("VERCEL_OIDC_TOKEN", "oidc-env"),
            ])
        ),
        gateway_env("oidc-env", "https://gw.example/v1")
    );
}

#[test]
fn auto_detects_gateway_when_ai_gateway_api_key_is_set() {
    assert_eq!(
        resolve_codex_env(None, &env(&[("AI_GATEWAY_API_KEY", "gw-auto")])),
        gateway_env("gw-auto", "https://ai-gateway.vercel.sh/v1")
    );
}

#[test]
fn auto_detects_gateway_when_vercel_oidc_token_is_set() {
    assert_eq!(
        resolve_codex_env(None, &env(&[("VERCEL_OIDC_TOKEN", "oidc-auto")])),
        gateway_env("oidc-auto", "https://ai-gateway.vercel.sh/v1")
    );
}

#[test]
fn auto_detects_direct_openai_when_only_openai_api_key_is_set() {
    assert_eq!(
        resolve_codex_env(None, &env(&[("OPENAI_API_KEY", "sk-auto")])),
        env(&[("CODEX_API_KEY", "sk-auto")])
    );
}

#[test]
fn uses_a_supplied_authentication_environment_instead_of_ambient_credentials() {
    let auth =
        HarnessV1Authentication::Environment(env(&[("OPENAI_API_KEY", "programmatic-openai-key")]));
    let ambient = env(&[("AI_GATEWAY_API_KEY", "ambient-gateway-key")]);
    assert_eq!(
        resolve_codex_env(Some(&auth), &ambient),
        env(&[("CODEX_API_KEY", "programmatic-openai-key")])
    );
    assert_eq!(
        resolve_codex_authentication_mode(Some(&auth), &ambient),
        CodexResolvedAuthenticationMode::Direct
    );
}

/// [rust] A typed mode rejects the nested object when it deserializes; the
/// untyped guard reports the TS message.
#[test]
fn rejects_nested_authentication_objects_before_reading_ambient_credentials() {
    let nested = json!({ "openai": { "apiKey": "legacy-key" } });
    assert!(serde_json::from_value::<CodexAuthenticationMode>(nested.clone()).is_err());
    assert_eq!(
        crate::harness::utils::is_harness_authentication_environment(Some(&nested))
            .unwrap_err()
            .to_string(),
        "Invalid auth: expected an authentication mode or a flat record with string values."
    );
}

#[test]
fn forwards_host_openai_base_url_alongside_the_api_key() {
    assert_eq!(
        resolve_codex_env(
            None,
            &env(&[
                ("OPENAI_API_KEY", "sk-auto"),
                ("OPENAI_BASE_URL", "https://ai-gateway.vercel.sh"),
            ])
        ),
        env(&[
            ("CODEX_API_KEY", "sk-auto"),
            ("OPENAI_BASE_URL", "https://ai-gateway.vercel.sh"),
        ])
    );
}

#[test]
fn returns_an_empty_env_when_nothing_is_configured() {
    assert_eq!(resolve_codex_env(None, &env(&[])), env(&[]));
}

#[test]
fn supports_string_authentication_modes() {
    assert_eq!(
        resolve_codex_env(Some(&DIRECT), &env(&[("OPENAI_API_KEY", "sk-direct")])),
        env(&[("CODEX_API_KEY", "sk-direct")])
    );
    assert_eq!(
        resolve_codex_env(Some(&GATEWAY), &env(&[("AI_GATEWAY_API_KEY", "gw-mode")])),
        gateway_env("gw-mode", "https://ai-gateway.vercel.sh/v1")
    );
}

// describe('resolveCodexAuthenticationMode')

#[test]
fn preserves_direct_auth_despite_ambient_gateway_credentials() {
    assert_eq!(
        resolve_codex_authentication_mode(
            Some(&DIRECT),
            &env(&[("AI_GATEWAY_API_KEY", "gateway-key")])
        ),
        CodexResolvedAuthenticationMode::Direct
    );
}

#[test]
fn resolves_ambient_gateway_credentials_to_gateway_auth() {
    assert_eq!(
        resolve_codex_authentication_mode(None, &env(&[("VERCEL_OIDC_TOKEN", "oidc-token")])),
        CodexResolvedAuthenticationMode::AiGateway
    );
}

// describe('resolveCodexAuthentication')

fn counting_read_subscription(calls: Arc<AtomicUsize>) -> ReadCodexSubscriptionFn {
    Arc::new(move |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        async { Ok(None) }.boxed()
    })
}

#[tokio::test]
async fn never_reads_native_authentication_for_gateway_auth() {
    let calls = Arc::new(AtomicUsize::new(0));
    let resolved = resolve_codex_authentication(ResolveCodexAuthenticationOptions {
        auth: Some(GATEWAY),
        process_env: Some(env(&[("AI_GATEWAY_API_KEY", "gateway")])),
        read_subscription: Some(counting_read_subscription(calls.clone())),
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(resolved.environment["AI_GATEWAY_API_KEY"], "gateway");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn prefers_a_direct_environment_api_key_over_native_authentication() {
    let calls = Arc::new(AtomicUsize::new(0));
    assert_eq!(
        resolve_codex_authentication(ResolveCodexAuthenticationOptions {
            auth: Some(DIRECT),
            process_env: Some(env(&[("OPENAI_API_KEY", "environment-key")])),
            read_subscription: Some(counting_read_subscription(calls.clone())),
            ..Default::default()
        })
        .await
        .unwrap(),
        CodexResolvedAuthentication {
            environment: env(&[("CODEX_API_KEY", "environment-key")]),
            request_headers: None,
        }
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn uses_a_fresh_file_backed_chatgpt_subscription() {
    let codex_home = mkdtemp("codex-auth-");
    std::fs::write(
        codex_home.path().join("auth.json"),
        json!({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": jwt(now_seconds() + 3600),
                "refresh_token": "refresh-token",
                "account_id": "account-id",
            },
        })
        .to_string(),
    )
    .unwrap();

    let resolved = read_codex_subscription(ReadCodexSubscriptionOptions {
        env: Some(env(&[("CODEX_HOME", &codex_home.path_string())])),
        ..Default::default()
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        resolved.environment.keys().collect::<Vec<_>>(),
        ["CODEX_API_KEY", "OPENAI_BASE_URL"]
    );
    assert_matches(r"^header\.", &resolved.environment["CODEX_API_KEY"]);
    assert_eq!(
        resolved.environment["OPENAI_BASE_URL"],
        "https://chatgpt.com/backend-api/codex"
    );
    assert_eq!(
        resolved.request_headers,
        Some(env(&[("ChatGPT-Account-ID", "account-id")]))
    );
}

#[tokio::test]
async fn refreshes_and_persists_an_expiring_subscription_without_losing_fields() {
    let codex_home = mkdtemp("codex-auth-");
    let auth_path = codex_home.path().join("auth.json");
    std::fs::write(
        &auth_path,
        json!({
            "auth_mode": "chatgpt",
            "preserved": true,
            "tokens": {
                "id_token": "id-token",
                "access_token": jwt(now_seconds() + 60),
                "refresh_token": "old-refresh",
                "account_id": "account-id",
            },
        })
        .to_string(),
    )
    .unwrap();

    read_codex_subscription(ReadCodexSubscriptionOptions {
        env: Some(env(&[("CODEX_HOME", &codex_home.path_string())])),
        fetch: Some(Arc::new(|_| {
            async {
                Ok(OAuthFetchResponse {
                    status: 200,
                    text: json!({
                        "access_token": jwt(now_seconds() + 3600),
                        "refresh_token": "new-refresh",
                    })
                    .to_string(),
                })
            }
            .boxed()
        })),
        ..Default::default()
    })
    .await
    .unwrap();
    let persisted: Value =
        serde_json::from_str(&std::fs::read_to_string(&auth_path).unwrap()).unwrap();
    assert_match_object(
        &persisted,
        &json!({
            "preserved": true,
            "tokens": {
                "id_token": "id-token",
                "refresh_token": "new-refresh",
                "account_id": "account-id",
            },
        }),
    );
}

/// `{ read: vi.fn(), write: vi.fn() }`.
#[derive(Default)]
struct MockKeyring {
    value: Option<String>,
    reads: Mutex<Vec<(String, String)>>,
}

#[async_trait]
impl CodexKeyring for MockKeyring {
    async fn read(&self, service: &str, account: &str) -> Option<String> {
        self.reads
            .lock()
            .push((service.to_string(), account.to_string()));
        self.value.clone()
    }

    async fn write(&self, _service: &str, _account: &str, _value: &str) -> HarnessResult<()> {
        Ok(())
    }
}

#[tokio::test]
async fn uses_the_configured_codex_keyring_before_the_auth_file_in_auto_mode() {
    let codex_home = mkdtemp("codex-auth-");
    std::fs::write(
        codex_home.path().join("auth.json"),
        json!({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": jwt(now_seconds() + 3600),
                "refresh_token": "file-refresh-token",
                "account_id": "file-account",
            },
        })
        .to_string(),
    )
    .unwrap();
    let keyring_access_token = jwt(now_seconds() + 3600);
    let keyring = Arc::new(MockKeyring {
        value: Some(
            json!({
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": keyring_access_token,
                    "refresh_token": "keyring-refresh-token",
                    "account_id": "keyring-account",
                },
            })
            .to_string(),
        ),
        ..Default::default()
    });

    assert_eq!(
        read_codex_subscription(ReadCodexSubscriptionOptions {
            env: Some(env(&[("CODEX_HOME", &codex_home.path_string())])),
            auth_credentials_store_mode: Some(json!("auto")),
            keyring: Some(keyring),
            ..Default::default()
        })
        .await
        .unwrap(),
        Some(CodexResolvedAuthentication {
            environment: env(&[
                ("CODEX_API_KEY", &keyring_access_token),
                ("OPENAI_BASE_URL", "https://chatgpt.com/backend-api/codex"),
            ]),
            request_headers: Some(env(&[("ChatGPT-Account-ID", "keyring-account")])),
        })
    );
}

#[tokio::test]
async fn reads_the_codex_credential_store_mode_from_config_toml() {
    let codex_home = mkdtemp("codex-auth-");
    std::fs::write(
        codex_home.path().join("config.toml"),
        "cli_auth_credentials_store = \"keyring\"\n",
    )
    .unwrap();
    let keyring_access_token = jwt(now_seconds() + 3600);
    let keyring = Arc::new(MockKeyring {
        value: Some(
            json!({
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": keyring_access_token,
                    "refresh_token": "refresh-token",
                },
            })
            .to_string(),
        ),
        ..Default::default()
    });

    let resolved = read_codex_subscription(ReadCodexSubscriptionOptions {
        env: Some(env(&[("CODEX_HOME", &codex_home.path_string())])),
        keyring: Some(keyring.clone()),
        ..Default::default()
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(resolved.environment["CODEX_API_KEY"], keyring_access_token);
    let reads = keyring.reads.lock().clone();
    assert_eq!(reads.len(), 1);
    assert_eq!(reads[0].0, "Codex Auth");
    assert_matches(ACCOUNT, &reads[0].1);
}

/// `createCodexKeyringCredential()`.
fn create_codex_keyring_credential() -> String {
    json!({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": jwt(now_seconds() + 3600),
            "refresh_token": "refresh-token",
        },
    })
    .to_string()
}

/// Reads a keyring credential on `platform` through a mocked `execFile`
/// and returns the one command the platform reader ran.
async fn read_platform_keyring(platform: &str) -> (String, Vec<String>) {
    let codex_home = mkdtemp("codex-auth-");
    let exec = Arc::new(MockExecFile::resolves(&create_codex_keyring_credential()));
    let resolved = read_codex_subscription(ReadCodexSubscriptionOptions {
        env: Some(env(&[("CODEX_HOME", &codex_home.path_string())])),
        auth_credentials_store_mode: Some(json!("keyring")),
        platform: Some(platform.into()),
        exec_file: Some(exec.clone()),
        ..Default::default()
    })
    .await
    .unwrap()
    .unwrap();
    assert_matches(r"^header\.", &resolved.environment["CODEX_API_KEY"]);
    let calls = exec.calls.lock().clone();
    assert_eq!(calls.len(), 1);
    let (file, args, _) = calls.into_iter().next().unwrap();
    (file, args)
}

#[tokio::test]
async fn reads_the_codex_credential_from_the_macos_keychain() {
    let (file, args) = read_platform_keyring("darwin").await;
    assert_eq!(file, "/usr/bin/security");
    assert_eq!(args[..3], ["find-generic-password", "-s", "Codex Auth"]);
    assert_eq!(args[3], "-a");
    assert_matches(ACCOUNT, &args[4]);
}

#[tokio::test]
async fn reads_the_codex_credential_from_linux_secret_service() {
    let (file, args) = read_platform_keyring("linux").await;
    assert_eq!(file, "secret-tool");
    assert_eq!(args[..4], ["lookup", "service", "Codex Auth", "username"]);
    assert_matches(ACCOUNT, &args[4]);
    assert_eq!(args[5..], ["target", "default"]);
}

#[tokio::test]
async fn reads_the_codex_credential_from_windows_credential_manager() {
    let codex_home = mkdtemp("codex-auth-");
    let exec = Arc::new(MockExecFile::resolves(&create_codex_keyring_credential()));
    let resolved = read_codex_subscription(ReadCodexSubscriptionOptions {
        env: Some(env(&[("CODEX_HOME", &codex_home.path_string())])),
        auth_credentials_store_mode: Some(json!("keyring")),
        platform: Some("win32".into()),
        exec_file: Some(exec.clone()),
        ..Default::default()
    })
    .await
    .unwrap()
    .unwrap();
    assert_matches(r"^header\.", &resolved.environment["CODEX_API_KEY"]);
    let calls = exec.calls.lock().clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "powershell.exe");
    let target = &calls[0].2.as_ref().unwrap().env["AI_SDK_WINDOWS_CREDENTIAL_MANAGER_TARGET"];
    assert_matches(r"^cli\|[0-9a-f]{16}\.Codex Auth$", target);
}

#[tokio::test]
async fn does_not_inspect_persistent_storage_in_ephemeral_mode() {
    let keyring = Arc::new(MockKeyring::default());
    assert_eq!(
        read_codex_subscription(ReadCodexSubscriptionOptions {
            auth_credentials_store_mode: Some(json!("ephemeral")),
            keyring: Some(keyring.clone()),
            ..Default::default()
        })
        .await
        .unwrap(),
        None
    );
    assert!(keyring.reads.lock().is_empty());
}

// describe('createCodexRequestTransformations')

fn sources(
    env_pairs: &[(&str, &str)],
    sandbox_pairs: &[(&str, &str)],
    auth: CodexResolvedAuthenticationMode,
) -> HarnessV1RequestTransformationSources<CodexResolvedAuthenticationMode> {
    HarnessV1RequestTransformationSources {
        env: env(env_pairs),
        sandbox_env: env(sandbox_pairs),
        auth,
    }
}

fn transformations_json(transformations: Vec<HarnessV1RequestTransformation>) -> Value {
    serde_json::to_value(transformations).unwrap()
}

#[test]
fn uses_the_configured_openai_compatible_route_for_direct_auth() {
    assert_eq!(
        transformations_json(
            create_codex_request_transformations(&sources(
                &[
                    ("CODEX_API_KEY", "openai-secret"),
                    ("OPENAI_BASE_URL", "https://openai.example/v1"),
                ],
                &[("CODEX_API_KEY", "sandbox-openai-secret")],
                CodexResolvedAuthenticationMode::Direct,
            ))
            .unwrap()
        ),
        json!([{
            "match": {
                "host": "openai.example",
                "path": { "startsWith": "/v1" },
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer sandbox-openai-secret" },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer openai-secret" } },
        }])
    );
}

#[test]
fn falls_back_to_the_ai_gateway_endpoint_for_gateway_auth() {
    assert_eq!(
        transformations_json(
            create_codex_request_transformations(&sources(
                &[("CODEX_API_KEY", "gateway-secret")],
                &[("CODEX_API_KEY", "sandbox-gateway-secret")],
                CodexResolvedAuthenticationMode::AiGateway,
            ))
            .unwrap()
        ),
        json!([{
            "match": {
                "host": "ai-gateway.vercel.sh",
                "path": { "startsWith": "/v1" },
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer sandbox-gateway-secret" },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer gateway-secret" } },
        }])
    );
}

#[test]
fn does_not_create_a_transformation_without_a_credential() {
    assert!(
        create_codex_request_transformations(&sources(
            &[],
            &[],
            CodexResolvedAuthenticationMode::Direct
        ))
        .unwrap()
        .is_empty()
    );
}

#[test]
fn adds_the_chatgpt_account_header_only_at_the_host_boundary() {
    let transformations = create_codex_subscription_request_transformations(
        &sources(
            &[
                ("CODEX_API_KEY", "host-access-token"),
                ("OPENAI_BASE_URL", "https://chatgpt.com/backend-api/codex"),
            ],
            &[("CODEX_API_KEY", "sandbox-placeholder")],
            CodexResolvedAuthenticationMode::Direct,
        ),
        Some(&env(&[("ChatGPT-Account-ID", "account-id")])),
    )
    .unwrap();
    assert_eq!(transformations.len(), 1);
    assert_eq!(
        serde_json::to_value(&transformations[0].transform).unwrap(),
        json!({
            "headers": {
                "Authorization": "Bearer host-access-token",
                "ChatGPT-Account-ID": "account-id",
            },
        })
    );
}
