//! Port of `claude-code-auth.test.ts` and `claude-code-subscription.test.ts`.
//!
//! [rust] TS mocks `node:child_process`; these tests hand
//! `read_api_key_helper_with` the shell runners and
//! `read_claude_code_subscription` the `execFile` seam instead.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::FutureExt as _;
use indexmap::IndexMap;
use serde_json::json;

use super::auth::*;
use super::subscription::*;
use crate::harness::adapters::test_temp_dir::mkdtemp;
use crate::harness::auth::{HarnessAuthentication, HarnessDirectAuthentication};
use crate::harness::network::HarnessRequestTransformationSources;
use crate::harness::test_support::assert_match_object;
use crate::harness::utils::native_subscription::test_exec::MockExecFile;
use crate::harness::utils::oauth_access_token::OAuthFetchResponse;

fn now_ms_for_tests() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

const DIRECT: ClaudeCodeAuthenticationMode =
    HarnessAuthentication::Choice(HarnessDirectAuthentication::Direct);
const GATEWAY: ClaudeCodeAuthenticationMode = HarnessAuthentication::AiGateway;

fn helper(value: Option<&'static str>) -> ResolveClaudeCodeEnvOptions {
    ResolveClaudeCodeEnvOptions {
        read_api_key_helper: Some(Arc::new(move || value.map(str::to_string))),
    }
}

fn no_helper() -> ResolveClaudeCodeEnvOptions {
    helper(None)
}

// describe('resolveClaudeCodeEnv')

#[test]
fn pins_direct_anthropic_auth_when_selected() {
    assert_eq!(
        resolve_claude_code_env(
            Some(&DIRECT),
            &env(&[
                ("ANTHROPIC_API_KEY", "sk-direct"),
                ("AI_GATEWAY_API_KEY", "gw-key")
            ]),
            &no_helper(),
        ),
        env(&[("ANTHROPIC_API_KEY", "sk-direct")])
    );
}

#[test]
fn uses_anthropic_env_in_direct_mode() {
    let process_env = env(&[
        ("ANTHROPIC_API_KEY", "sk-process"),
        ("ANTHROPIC_BASE_URL", "https://api.example.com"),
    ]);
    assert_eq!(
        resolve_claude_code_env(Some(&DIRECT), &process_env, &no_helper()),
        process_env
    );
}

#[test]
fn routes_through_the_gateway_when_gateway_mode_is_selected() {
    let env = resolve_claude_code_env(
        Some(&GATEWAY),
        &env(&[("AI_GATEWAY_API_KEY", "gw-explicit")]),
        &no_helper(),
    );
    assert_eq!(env["AI_GATEWAY_API_KEY"], "gw-explicit");
    assert_eq!(env["ANTHROPIC_API_KEY"], "gw-explicit");
    assert_eq!(env["ANTHROPIC_BASE_URL"], "https://ai-gateway.vercel.sh");
}

#[test]
fn uses_gateway_auth_from_the_environment() {
    assert_eq!(
        resolve_claude_code_env(
            Some(&GATEWAY),
            &env(&[
                ("AI_GATEWAY_BASE_URL", "https://gw.example"),
                ("VERCEL_OIDC_TOKEN", "oidc-env"),
            ]),
            &no_helper(),
        ),
        env(&[
            ("AI_GATEWAY_API_KEY", "oidc-env"),
            ("ANTHROPIC_API_KEY", "oidc-env"),
            ("AI_GATEWAY_BASE_URL", "https://gw.example"),
            ("ANTHROPIC_BASE_URL", "https://gw.example"),
        ])
    );
}

#[test]
fn auto_detects_gateway_when_ai_gateway_api_key_is_set() {
    let env = resolve_claude_code_env(
        None,
        &env(&[("AI_GATEWAY_API_KEY", "gw-auto")]),
        &no_helper(),
    );
    assert_eq!(env["AI_GATEWAY_API_KEY"], "gw-auto");
    assert_eq!(env["ANTHROPIC_API_KEY"], "gw-auto");
}

#[test]
fn auto_detects_gateway_when_vercel_oidc_token_is_set() {
    let env = resolve_claude_code_env(
        None,
        &env(&[("VERCEL_OIDC_TOKEN", "oidc-auto")]),
        &no_helper(),
    );
    assert_eq!(env["AI_GATEWAY_API_KEY"], "oidc-auto");
    assert_eq!(env["ANTHROPIC_API_KEY"], "oidc-auto");
    assert_eq!(env["AI_GATEWAY_BASE_URL"], "https://ai-gateway.vercel.sh");
    assert_eq!(env["ANTHROPIC_BASE_URL"], "https://ai-gateway.vercel.sh");
}

#[test]
fn auto_detects_direct_anthropic_when_only_anthropic_api_key_is_set() {
    assert_eq!(
        resolve_claude_code_env(
            None,
            &env(&[("ANTHROPIC_API_KEY", "sk-auto")]),
            &no_helper()
        ),
        env(&[("ANTHROPIC_API_KEY", "sk-auto")])
    );
}

#[test]
fn uses_a_supplied_authentication_environment_without_ambient_or_helper_fallback() {
    let calls = Arc::new(AtomicUsize::new(0));
    let options = ResolveClaudeCodeEnvOptions {
        read_api_key_helper: Some({
            let calls = calls.clone();
            Arc::new(move || {
                calls.fetch_add(1, Ordering::SeqCst);
                Some("helper-key".into())
            })
        }),
    };
    let auth = HarnessAuthentication::Environment(env(&[(
        "ANTHROPIC_API_KEY",
        "programmatic-anthropic-key",
    )]));
    let ambient = env(&[("AI_GATEWAY_API_KEY", "ambient-gateway-key")]);

    assert_eq!(
        resolve_claude_code_env(Some(&auth), &ambient, &options),
        env(&[("ANTHROPIC_API_KEY", "programmatic-anthropic-key")])
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        resolve_claude_code_authentication_mode(Some(&auth), &ambient),
        ClaudeCodeResolvedAuthenticationMode::Direct
    );
}

/// [rust] A typed mode rejects the nested object when it deserializes; the
/// untyped guard reports the TS message.
#[test]
fn rejects_nested_authentication_objects_before_reading_ambient_credentials() {
    let nested = json!({ "anthropic": { "apiKey": "legacy-key" } });
    assert!(serde_json::from_value::<ClaudeCodeAuthenticationMode>(nested.clone()).is_err());
    assert_eq!(
        crate::harness::utils::is_harness_authentication_environment(Some(&nested))
            .unwrap_err()
            .to_string(),
        "Invalid auth: expected an authentication mode or a flat record with string values."
    );
}

#[test]
fn forwards_host_anthropic_base_url_alongside_the_api_key() {
    let process_env = env(&[
        ("ANTHROPIC_API_KEY", "sk-auto"),
        ("ANTHROPIC_BASE_URL", "https://ai-gateway.vercel.sh"),
    ]);
    assert_eq!(
        resolve_claude_code_env(None, &process_env, &no_helper()),
        process_env
    );
}

#[test]
fn forwards_a_base_url_alongside_the_api_key_helper_supplied_credentials() {
    assert_eq!(
        resolve_claude_code_env(
            None,
            &env(&[("ANTHROPIC_BASE_URL", "https://ai-gateway.vercel.sh")]),
            &helper(Some("vck_from_helper")),
        ),
        env(&[
            ("ANTHROPIC_API_KEY", "vck_from_helper"),
            ("ANTHROPIC_AUTH_TOKEN", "vck_from_helper"),
            ("ANTHROPIC_BASE_URL", "https://ai-gateway.vercel.sh"),
        ])
    );
}

#[test]
fn populates_both_anthropic_api_key_and_anthropic_auth_token_from_the_api_key_helper() {
    assert_eq!(
        resolve_claude_code_env(None, &IndexMap::new(), &helper(Some("vck_from_helper"))),
        env(&[
            ("ANTHROPIC_API_KEY", "vck_from_helper"),
            ("ANTHROPIC_AUTH_TOKEN", "vck_from_helper"),
        ])
    );
}

#[test]
fn prefers_a_static_anthropic_api_key_over_the_api_key_helper() {
    let env = resolve_claude_code_env(
        None,
        &env(&[("ANTHROPIC_API_KEY", "sk-static")]),
        &helper(Some("sk-from-helper")),
    );
    assert_eq!(env["ANTHROPIC_API_KEY"], "sk-static");
}

#[test]
fn returns_an_empty_env_when_nothing_is_configured() {
    assert!(resolve_claude_code_env(None, &IndexMap::new(), &no_helper()).is_empty());
}

#[test]
fn supports_string_authentication_modes() {
    assert_eq!(
        resolve_claude_code_env(
            Some(&serde_json::from_value(json!("direct")).unwrap()),
            &env(&[("ANTHROPIC_API_KEY", "sk-direct")]),
            &no_helper(),
        ),
        env(&[("ANTHROPIC_API_KEY", "sk-direct")])
    );
    assert_eq!(
        resolve_claude_code_env(
            Some(&serde_json::from_value(json!("ai-gateway")).unwrap()),
            &env(&[("AI_GATEWAY_API_KEY", "gw-mode")]),
            &no_helper(),
        ),
        env(&[
            ("AI_GATEWAY_API_KEY", "gw-mode"),
            ("ANTHROPIC_API_KEY", "gw-mode"),
            ("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh"),
            ("ANTHROPIC_BASE_URL", "https://ai-gateway.vercel.sh"),
        ])
    );
}

// describe('default apiKeyHelper reader')

fn write_api_key_helper(home: &Path, command: &str) {
    std::fs::create_dir_all(home.join(".claude")).unwrap();
    std::fs::write(
        home.join(".claude").join("settings.json"),
        json!({ "apiKeyHelper": command }).to_string(),
    )
    .unwrap();
}

/// `resolveClaudeCodeEnv(undefined, {})` with the default helper reader over
/// `home` and the given shells.
fn resolve_with_shells(
    home: &Path,
    sh: &(dyn Fn(&str) -> Result<String, ShellRunError> + Sync),
    system_shell: &(dyn Fn(&str) -> Result<String, ShellRunError> + Sync),
) -> IndexMap<String, String> {
    let value = read_api_key_helper_with(home, sh, system_shell);
    resolve_claude_code_env(
        None,
        &IndexMap::new(),
        &ResolveClaudeCodeEnvOptions {
            read_api_key_helper: Some(Arc::new(move || value.clone())),
        },
    )
}

#[cfg(unix)]
#[test]
fn falls_back_to_the_system_shell_when_sh_is_not_on_path() {
    let home = mkdtemp("claude-code-auth-");
    write_api_key_helper(home.path(), "echo sk-from-helper");
    // No `sh` on PATH, as in a PowerShell or cmd session on Windows.
    assert_eq!(
        resolve_with_shells(
            home.path(),
            &|_| Err(ShellRunError::NotFound),
            &run_system_shell
        ),
        env(&[
            ("ANTHROPIC_API_KEY", "sk-from-helper"),
            ("ANTHROPIC_AUTH_TOKEN", "sk-from-helper"),
        ])
    );
}

#[cfg(unix)]
#[test]
fn ignores_an_api_key_helper_command_that_fails() {
    let home = mkdtemp("claude-code-auth-");
    write_api_key_helper(home.path(), "exit 1");
    assert!(
        resolve_with_shells(
            home.path(),
            &|_| Err(ShellRunError::NotFound),
            &run_system_shell
        )
        .is_empty()
    );
    assert!(resolve_with_shells(home.path(), &run_sh, &run_system_shell).is_empty());
}

#[test]
fn preserves_sh_based_helpers_when_sh_is_available() {
    let home = mkdtemp("claude-code-auth-");
    let command = r#"printf "%s" "${HELPER_KEY:-sk-from-sh}""#;
    write_api_key_helper(home.path(), command);
    let sh_calls = parking_lot::Mutex::new(Vec::new());
    let system_shell_calls = AtomicUsize::new(0);

    assert_eq!(
        resolve_with_shells(
            home.path(),
            &|command| {
                sh_calls.lock().push(command.to_string());
                Ok("sk-from-sh\n".into())
            },
            &|_| {
                system_shell_calls.fetch_add(1, Ordering::SeqCst);
                Ok(String::new())
            },
        ),
        env(&[
            ("ANTHROPIC_API_KEY", "sk-from-sh"),
            ("ANTHROPIC_AUTH_TOKEN", "sk-from-sh"),
        ])
    );
    assert_eq!(*sh_calls.lock(), [command]);
    assert_eq!(system_shell_calls.load(Ordering::SeqCst), 0);
}

/// `it.each([1, 127])('does not retry a helper that exits with status %s')`
/// and `does not fall back when sh cannot be launched due to permissions`:
/// both are failures other than a missing `sh`.
#[test]
fn does_not_retry_a_helper_that_fails_or_cannot_be_launched() {
    for command in ["exit 1", "exit 127", "echo sk-from-helper"] {
        let home = mkdtemp("claude-code-auth-");
        write_api_key_helper(home.path(), command);
        let sh_calls = AtomicUsize::new(0);
        let system_shell_calls = AtomicUsize::new(0);
        assert!(
            resolve_with_shells(
                home.path(),
                &|_| {
                    sh_calls.fetch_add(1, Ordering::SeqCst);
                    Err(ShellRunError::Failed)
                },
                &|_| {
                    system_shell_calls.fetch_add(1, Ordering::SeqCst);
                    Ok("sk-from-helper".into())
                },
            )
            .is_empty()
        );
        assert_eq!(sh_calls.load(Ordering::SeqCst), 1);
        assert_eq!(system_shell_calls.load(Ordering::SeqCst), 0);
    }
}

// describe('resolveClaudeCodeAuthentication')

fn counting_subscription(
    calls: Arc<AtomicUsize>,
    result: Option<IndexMap<String, String>>,
) -> Option<ReadClaudeCodeSubscriptionFn> {
    Some(Arc::new(move || {
        calls.fetch_add(1, Ordering::SeqCst);
        let result = result.clone();
        async move { Ok(result) }.boxed()
    }))
}

#[tokio::test]
async fn does_not_inspect_native_storage_in_gateway_mode() {
    let calls = Arc::new(AtomicUsize::new(0));
    resolve_claude_code_authentication(ResolveClaudeCodeAuthenticationOptions {
        auth: Some(GATEWAY),
        process_env: Some(env(&[("AI_GATEWAY_API_KEY", "gateway")])),
        read_subscription: counting_subscription(calls.clone(), None),
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn prefers_an_explicit_process_oauth_token_over_native_storage() {
    let calls = Arc::new(AtomicUsize::new(0));
    assert_eq!(
        resolve_claude_code_authentication(ResolveClaudeCodeAuthenticationOptions {
            auth: Some(DIRECT),
            process_env: Some(env(&[("CLAUDE_CODE_OAUTH_TOKEN", "process-oauth-token")])),
            read_subscription: counting_subscription(calls.clone(), None),
            ..Default::default()
        })
        .await
        .unwrap(),
        env(&[("CLAUDE_CODE_OAUTH_TOKEN", "process-oauth-token")])
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn uses_a_supplied_oauth_authentication_environment_without_ambient_or_native_discovery() {
    let calls = Arc::new(AtomicUsize::new(0));
    assert_eq!(
        resolve_claude_code_authentication(ResolveClaudeCodeAuthenticationOptions {
            auth: Some(HarnessAuthentication::Environment(env(&[(
                "CLAUDE_CODE_OAUTH_TOKEN",
                "supplied-oauth-token"
            )]))),
            process_env: Some(env(&[("AI_GATEWAY_API_KEY", "ambient-gateway-key")])),
            read_subscription: counting_subscription(calls.clone(), None),
            ..Default::default()
        })
        .await
        .unwrap(),
        env(&[("CLAUDE_CODE_OAUTH_TOKEN", "supplied-oauth-token")])
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn uses_native_storage_after_direct_environment_credentials() {
    let subscription = env(&[("CLAUDE_CODE_OAUTH_TOKEN", "subscription-access-token")]);
    assert_eq!(
        resolve_claude_code_authentication(ResolveClaudeCodeAuthenticationOptions {
            auth: Some(DIRECT),
            process_env: Some(IndexMap::new()),
            read_subscription: counting_subscription(Arc::default(), Some(subscription.clone())),
            ..Default::default()
        })
        .await
        .unwrap(),
        subscription
    );
}

#[tokio::test]
async fn refreshes_and_persists_file_backed_claude_credentials() {
    let config_directory = mkdtemp("claude-auth-");
    let credential_path = config_directory.path().join(".credentials.json");
    std::fs::write(
        &credential_path,
        json!({
            "preserved": true,
            "claudeAiOauth": {
                "accessToken": "old-access-token",
                "refreshToken": "old-refresh-token",
                "expiresAt": now_ms_for_tests() + 60_000,
            },
        })
        .to_string(),
    )
    .unwrap();

    assert_eq!(
        read_claude_code_subscription(ReadClaudeCodeSubscriptionOptions {
            env: Some(env(&[(
                "CLAUDE_CONFIG_DIR",
                &config_directory.path_string()
            )])),
            fetch: Some(Arc::new(|_| {
                async {
                    Ok(OAuthFetchResponse {
                        status: 200,
                        text: json!({
                            "access_token": "new-access-token",
                            "refresh_token": "new-refresh-token",
                            "expires_in": 3600,
                        })
                        .to_string(),
                    })
                }
                .boxed()
            })),
            ..Default::default()
        })
        .await
        .unwrap(),
        Some(env(&[
            ("CLAUDE_CODE_OAUTH_TOKEN", "new-access-token"),
            ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ]))
    );
    let persisted: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&credential_path).unwrap()).unwrap();
    assert_match_object(
        &persisted,
        &json!({
            "preserved": true,
            "claudeAiOauth": {
                "accessToken": "new-access-token",
                "refreshToken": "new-refresh-token",
            },
        }),
    );
}

// describe('resolveClaudeCodeAuthenticationMode')

#[test]
fn preserves_direct_auth_despite_ambient_gateway_credentials() {
    assert_eq!(
        resolve_claude_code_authentication_mode(
            Some(&DIRECT),
            &env(&[("AI_GATEWAY_API_KEY", "gateway-key")])
        ),
        ClaudeCodeResolvedAuthenticationMode::Direct
    );
}

#[test]
fn resolves_ambient_gateway_credentials_to_gateway_auth() {
    assert_eq!(
        resolve_claude_code_authentication_mode(None, &env(&[("VERCEL_OIDC_TOKEN", "oidc-token")])),
        ClaudeCodeResolvedAuthenticationMode::AiGateway
    );
}

// describe('createClaudeCodeRequestTransformations')

fn sources(
    env: IndexMap<String, String>,
    sandbox_env: IndexMap<String, String>,
    auth: ClaudeCodeResolvedAuthenticationMode,
) -> HarnessRequestTransformationSources<ClaudeCodeResolvedAuthenticationMode> {
    HarnessRequestTransformationSources {
        env,
        sandbox_env,
        auth,
    }
}

#[test]
fn injects_anthropic_api_key_and_auth_token_headers_at_the_configured_endpoint() {
    let transformations = create_claude_code_request_transformations(&sources(
        env(&[
            ("ANTHROPIC_API_KEY", "api-secret"),
            ("ANTHROPIC_AUTH_TOKEN", "token-secret"),
            ("ANTHROPIC_BASE_URL", "https://anthropic.example/v1"),
        ]),
        env(&[
            ("ANTHROPIC_API_KEY", "sandbox-api-secret"),
            ("ANTHROPIC_AUTH_TOKEN", "sandbox-token-secret"),
        ]),
        ClaudeCodeResolvedAuthenticationMode::Direct,
    ))
    .unwrap();
    assert_eq!(
        serde_json::to_value(transformations).unwrap(),
        json!([
            {
                "match": {
                    "host": "anthropic.example",
                    "path": { "startsWith": "/v1" },
                    "headers": [{
                        "key": { "exact": "x-api-key" },
                        "value": { "exact": "sandbox-api-secret" },
                    }],
                },
                "transform": { "headers": { "x-api-key": "api-secret" } },
            },
            {
                "match": {
                    "host": "anthropic.example",
                    "path": { "startsWith": "/v1" },
                    "headers": [{
                        "key": { "exact": "Authorization" },
                        "value": { "exact": "Bearer sandbox-token-secret" },
                    }],
                },
                "transform": { "headers": { "Authorization": "Bearer token-secret" } },
            },
        ])
    );
}

#[test]
fn uses_the_resolved_gateway_route() {
    let transformations = create_claude_code_request_transformations(&sources(
        env(&[
            ("ANTHROPIC_API_KEY", "gateway-secret"),
            ("ANTHROPIC_BASE_URL", "https://gateway.example"),
        ]),
        env(&[("ANTHROPIC_API_KEY", "sandbox-gateway-secret")]),
        ClaudeCodeResolvedAuthenticationMode::AiGateway,
    ))
    .unwrap();
    assert_eq!(
        serde_json::to_value(transformations).unwrap(),
        json!([{
            "match": {
                "host": "gateway.example",
                "headers": [{
                    "key": { "exact": "x-api-key" },
                    "value": { "exact": "sandbox-gateway-secret" },
                }],
            },
            "transform": { "headers": { "x-api-key": "gateway-secret" } },
        }])
    );
}

// claude-code-subscription.test.ts

#[tokio::test]
async fn falls_back_to_macos_keychain_when_the_credential_file_has_no_claude_oauth_credential() {
    let home = mkdtemp("claude-home-");
    let config_directory = home.path().join(".claude");
    std::fs::create_dir_all(&config_directory).unwrap();
    std::fs::write(
        config_directory.join(".credentials.json"),
        json!({ "mcpOAuth": {} }).to_string(),
    )
    .unwrap();
    let exec = Arc::new(MockExecFile::resolves(
        &json!({
            "claudeAiOauth": {
                "accessToken": "keychain-access-token",
                "refreshToken": "keychain-refresh-token",
                "expiresAt": now_ms_for_tests() + 60 * 60 * 1000,
            },
        })
        .to_string(),
    ));

    assert_eq!(
        read_claude_code_subscription(ReadClaudeCodeSubscriptionOptions {
            env: Some(IndexMap::new()),
            home_directory: Some(home.path_string()),
            platform: Some("darwin".into()),
            exec_file: Some(exec.clone()),
            ..Default::default()
        })
        .await
        .unwrap(),
        Some(env(&[
            ("CLAUDE_CODE_OAUTH_TOKEN", "keychain-access-token"),
            ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ]))
    );
    assert_eq!(exec.calls.lock().len(), 1);
}

#[test]
fn replaces_only_the_oauth_authorization_header() {
    let transformations = create_claude_code_subscription_request_transformations(&sources(
        env(&[
            ("CLAUDE_CODE_OAUTH_TOKEN", "host-access-token"),
            ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ]),
        env(&[
            ("CLAUDE_CODE_OAUTH_TOKEN", "sandbox-placeholder"),
            ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ]),
        ClaudeCodeResolvedAuthenticationMode::Direct,
    ))
    .unwrap();
    assert_eq!(
        serde_json::to_value(transformations).unwrap(),
        json!([{
            "match": {
                "host": "api.anthropic.com",
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer sandbox-placeholder" },
                }],
            },
            "transform": { "headers": { "Authorization": "Bearer host-access-token" } },
        }])
    );
}
