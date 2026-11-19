//! Port of `opencode-auth.test.ts`.

use std::sync::{Arc, Mutex};

use futures::FutureExt as _;
use indexmap::IndexMap;
use serde_json::{Value, json};

use super::*;
use crate::harness::HarnessCredentialForwarding;
use crate::harness::adapters::opencode::subscription::{
    OpenCodeResolvedAuthentication, OpenCodeSubscription, ReadOpenCodeSubscriptionFn,
    ResolveOpenCodeAuthenticationOptions, resolve_opencode_authentication,
};
use crate::harness::utils::resolve_sandbox_credential_environment;

use OpenCodeAuthenticationChoice as C;
use OpenCodeResolvedAuthenticationMode as M;

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn mode(choice: C) -> Option<OpenCodeAuthenticationMode> {
    Some(HarnessAuthentication::Choice(choice))
}

fn transformations(
    env: IndexMap<String, String>,
    sandbox_env: IndexMap<String, String>,
    auth: M,
) -> Value {
    serde_json::to_value(
        create_opencode_request_transformations(&HarnessRequestTransformationSources {
            env,
            sandbox_env,
            auth,
        })
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn resolves_provider_from_explicit_provider_or_model_prefix() {
    assert_eq!(resolve_opencode_provider(None, Some("openai")), M::Openai);
    assert_eq!(
        resolve_opencode_provider(Some("openai/gpt-5.1"), None),
        M::Openai
    );
    assert_eq!(
        resolve_opencode_provider(Some("anthropic/claude-sonnet-4-5"), None),
        M::Anthropic
    );
    assert_eq!(
        resolve_opencode_provider(Some("custom/model"), None),
        M::Anthropic
    );
    assert_eq!(
        resolve_opencode_provider(Some("google/gemini-3.8-flash"), None),
        M::Google
    );
    assert_eq!(resolve_opencode_provider(Some("xai/grok-4"), None), M::Xai);
}

#[test]
fn splits_provider_prefixed_models() {
    assert_eq!(
        split_opencode_model(Some("anthropic/claude-sonnet-4-5"), None),
        OpenCodeModelParts {
            provider_id: Some("anthropic".into()),
            model_id: Some("claude-sonnet-4-5".into()),
            model: Some("anthropic/claude-sonnet-4-5".into()),
        }
    );
    assert_eq!(
        split_opencode_model(Some("gpt-5.1"), Some("openai")),
        OpenCodeModelParts {
            provider_id: Some("openai".into()),
            model_id: Some("gpt-5.1".into()),
            model: Some("openai/gpt-5.1".into()),
        }
    );
}

#[test]
fn uses_ai_gateway_env_including_oidc_fallback() {
    assert_eq!(
        resolve_opencode_env(
            None,
            None,
            None,
            &env(&[
                ("VERCEL_OIDC_TOKEN", "oidc-token"),
                ("AI_GATEWAY_BASE_URL", "https://gateway.example"),
            ])
        ),
        env(&[
            ("AI_GATEWAY_API_KEY", "oidc-token"),
            ("AI_GATEWAY_BASE_URL", "https://gateway.example/v1"),
        ])
    );
}

#[test]
fn prefers_the_selected_direct_provider_before_ambient_gateway_fallback() {
    assert_eq!(
        resolve_opencode_env(
            mode(C::Openai).as_ref(),
            None,
            Some("openai"),
            &env(&[
                ("OPENAI_API_KEY", "openai-key"),
                ("AI_GATEWAY_API_KEY", "gateway-key"),
                ("AI_GATEWAY_BASE_URL", "https://gateway.example/v1"),
            ])
        ),
        env(&[("OPENAI_API_KEY", "openai-key")])
    );
}

#[test]
fn uses_a_supplied_authentication_environment_instead_of_ambient_credentials() {
    let auth = Some(HarnessAuthentication::Environment(env(&[(
        "OPENAI_API_KEY",
        "programmatic-openai-key",
    )])));
    let process_env = env(&[("AI_GATEWAY_API_KEY", "ambient-gateway-key")]);
    assert_eq!(
        resolve_opencode_env(auth.as_ref(), None, Some("openai"), &process_env),
        env(&[("OPENAI_API_KEY", "programmatic-openai-key")])
    );
    assert_eq!(
        resolve_opencode_authentication_mode(auth.as_ref(), None, Some("openai"), &process_env),
        M::Openai
    );
}

#[test]
fn resolves_google_credentials_from_a_supplied_authentication_environment() {
    let auth = Some(HarnessAuthentication::Environment(env(&[(
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "programmatic-google-key",
    )])));
    let process_env = IndexMap::new();
    assert_eq!(
        resolve_opencode_env(auth.as_ref(), None, None, &process_env),
        env(&[("GOOGLE_GENERATIVE_AI_API_KEY", "programmatic-google-key")])
    );
    assert_eq!(
        resolve_opencode_authentication_mode(auth.as_ref(), None, None, &process_env),
        M::Google
    );
}

#[test]
fn ignores_an_empty_ambient_google_credential_when_selecting_the_fallback_provider() {
    let auth = Some(HarnessAuthentication::Auto);
    let process_env = env(&[("GOOGLE_GENERATIVE_AI_API_KEY", "")]);
    assert_eq!(
        resolve_opencode_env(auth.as_ref(), None, None, &process_env),
        IndexMap::<String, String>::new()
    );
    assert_eq!(
        resolve_opencode_authentication_mode(auth.as_ref(), None, None, &process_env),
        M::Anthropic
    );
}

#[test]
fn selects_a_valid_google_credential_when_unrelated_provider_credentials_are_empty() {
    let auth = Some(HarnessAuthentication::Auto);
    let process_env = env(&[
        ("GOOGLE_GENERATIVE_AI_API_KEY", "google-key"),
        ("OPENAI_API_KEY", ""),
        ("ANTHROPIC_API_KEY", ""),
        ("ANTHROPIC_AUTH_TOKEN", ""),
        ("XAI_API_KEY", ""),
        ("GITHUB_TOKEN", ""),
        ("GITHUB_COPILOT_TOKEN", ""),
        ("POE_API_KEY", ""),
        ("OPENCODE_API_KEY", ""),
        ("GITLAB_TOKEN", ""),
    ]);
    assert_eq!(
        resolve_opencode_env(auth.as_ref(), None, None, &process_env),
        env(&[("GOOGLE_GENERATIVE_AI_API_KEY", "google-key")])
    );
    assert_eq!(
        resolve_opencode_authentication_mode(auth.as_ref(), None, None, &process_env),
        M::Google
    );
}

#[test]
fn keeps_explicit_authentication_authoritative_over_ambient_google_credentials() {
    let process_env = env(&[
        ("ANTHROPIC_API_KEY", "anthropic-key"),
        ("OPENAI_API_KEY", "openai-key"),
        ("GOOGLE_GENERATIVE_AI_API_KEY", "google-key"),
    ]);
    for (choice, expected_env, expected_mode) in [
        (
            C::Anthropic,
            env(&[("ANTHROPIC_API_KEY", "anthropic-key")]),
            M::Anthropic,
        ),
        (
            C::Openai,
            env(&[("OPENAI_API_KEY", "openai-key")]),
            M::Openai,
        ),
    ] {
        let auth = mode(choice);
        assert_eq!(
            resolve_opencode_env(auth.as_ref(), None, None, &process_env),
            expected_env
        );
        assert_eq!(
            resolve_opencode_authentication_mode(auth.as_ref(), None, None, &process_env),
            expected_mode
        );
    }
}

#[test]
fn uses_only_the_selected_provider_credentials_with_direct_auth() {
    let process_env = env(&[
        ("AI_GATEWAY_API_KEY", "gateway-key"),
        ("VERCEL_OIDC_TOKEN", "oidc-token"),
        ("GITHUB_TOKEN", "github-token"),
        ("OPENAI_API_KEY", "openai-key"),
    ]);
    let auth = mode(C::Direct);
    assert_eq!(
        resolve_opencode_env(auth.as_ref(), None, Some("github-copilot"), &process_env),
        env(&[("GITHUB_TOKEN", "github-token")])
    );
    assert_eq!(
        resolve_opencode_authentication_mode(
            auth.as_ref(),
            None,
            Some("github-copilot"),
            &process_env
        ),
        M::GithubCopilot
    );
}

#[test]
fn resolves_direct_auth_without_a_provider_to_the_default_provider() {
    let process_env = env(&[
        ("VERCEL_OIDC_TOKEN", "oidc-token"),
        ("ANTHROPIC_API_KEY", "anthropic-key"),
    ]);
    let auth = mode(C::Direct);
    assert_eq!(
        resolve_opencode_env(auth.as_ref(), None, None, &process_env),
        env(&[("ANTHROPIC_API_KEY", "anthropic-key")])
    );
    assert_eq!(
        resolve_opencode_authentication_mode(auth.as_ref(), None, None, &process_env),
        M::Anthropic
    );
}

#[test]
fn preserves_the_anthropic_fallback_for_mixed_direct_provider_credentials() {
    let process_env = env(&[
        ("ANTHROPIC_API_KEY", "anthropic-key"),
        ("GOOGLE_GENERATIVE_AI_API_KEY", "google-key"),
    ]);
    let auth = Some(HarnessAuthentication::Auto);
    assert_eq!(
        resolve_opencode_env(auth.as_ref(), None, None, &process_env),
        env(&[("ANTHROPIC_API_KEY", "anthropic-key")])
    );
    assert_eq!(
        resolve_opencode_authentication_mode(auth.as_ref(), None, None, &process_env),
        M::Anthropic
    );
}

/// [rust] A nested object is not an [`OpenCodeAuthenticationMode`]; the
/// untyped guard reports the TS message.
#[test]
fn rejects_nested_authentication_objects_before_reading_ambient_credentials() {
    let nested = json!({ "openai": { "apiKey": "legacy-key" } });
    assert!(serde_json::from_value::<OpenCodeAuthenticationMode>(nested.clone()).is_err());
    assert_eq!(
        crate::harness::utils::is_harness_authentication_environment(Some(&nested))
            .unwrap_err()
            .to_string(),
        "Invalid auth: expected an authentication mode or a flat record with string values."
    );
}

#[test]
fn normalizes_opencode_gateway_base_urls_to_v1() {
    assert_eq!(
        to_opencode_gateway_base_url("https://ai-gateway.vercel.sh"),
        "https://ai-gateway.vercel.sh/v1"
    );
    assert_eq!(
        to_opencode_gateway_base_url("https://ai-gateway.vercel.sh/v1"),
        "https://ai-gateway.vercel.sh/v1"
    );
}

#[test]
fn supports_string_authentication_modes() {
    let openai: OpenCodeAuthenticationMode = serde_json::from_value(json!("openai")).unwrap();
    assert_eq!(
        resolve_opencode_env(
            Some(&openai),
            None,
            Some("openai"),
            &env(&[
                ("OPENAI_API_KEY", "sk-openai"),
                ("OPENAI_BASE_URL", "https://compat.example/v1"),
            ])
        ),
        env(&[
            ("OPENAI_API_KEY", "sk-openai"),
            ("OPENAI_BASE_URL", "https://compat.example/v1"),
        ])
    );
    let gateway: OpenCodeAuthenticationMode = serde_json::from_value(json!("ai-gateway")).unwrap();
    assert_eq!(
        resolve_opencode_env(
            Some(&gateway),
            None,
            None,
            &env(&[("AI_GATEWAY_API_KEY", "gw-mode")])
        ),
        env(&[
            ("AI_GATEWAY_API_KEY", "gw-mode"),
            ("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1"),
        ])
    );
}

// --- resolveOpenCodeAuthentication -------------------------------------------

type Calls = Arc<Mutex<Vec<(M, IndexMap<String, String>)>>>;

fn reader(result: Option<OpenCodeSubscription>) -> (ReadOpenCodeSubscriptionFn, Calls) {
    let calls: Calls = Arc::default();
    let recorded = calls.clone();
    let read: ReadOpenCodeSubscriptionFn = Arc::new(move |provider, env| {
        recorded.lock().unwrap().push((provider, env));
        let result = result.clone();
        async move { Ok(result) }.boxed()
    });
    (read, calls)
}

async fn resolve(
    auth: Option<OpenCodeAuthenticationMode>,
    provider: &str,
    process_env: IndexMap<String, String>,
    read: ReadOpenCodeSubscriptionFn,
) -> OpenCodeResolvedAuthentication {
    resolve_opencode_authentication(ResolveOpenCodeAuthenticationOptions {
        auth,
        model: None,
        provider: Some(provider.into()),
        process_env: Some(process_env),
        read_subscription: Some(read),
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn never_reads_native_subscriptions_for_gateway_auth() {
    let (read, calls) = reader(None);
    let resolved = resolve(
        Some(HarnessAuthentication::AiGateway),
        "openai",
        env(&[("AI_GATEWAY_API_KEY", "gateway-key")]),
        read,
    )
    .await;
    assert_eq!(resolved.authentication_mode, M::AiGateway);
    assert!(calls.lock().unwrap().is_empty());
}

async fn assert_api_key_wins(
    auth: OpenCodeAuthenticationMode,
    process_env: IndexMap<String, String>,
) {
    let (read, calls) = reader(None);
    let resolved = resolve(Some(auth), "xai", process_env, read).await;
    assert_eq!(
        resolved,
        OpenCodeResolvedAuthentication {
            environment: env(&[("XAI_API_KEY", "environment-key")]),
            authentication_mode: M::Xai,
            subscription: None,
        }
    );
    assert!(calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn prefers_a_provider_api_key_before_native_subscriptions() {
    assert_api_key_wins(
        HarnessAuthentication::Auto,
        env(&[("XAI_API_KEY", "environment-key")]),
    )
    .await;
}

#[tokio::test]
async fn prefers_a_provider_api_key_before_native_subscriptions_with_direct_auth() {
    assert_api_key_wins(
        HarnessAuthentication::Choice(C::Direct),
        env(&[
            ("VERCEL_OIDC_TOKEN", "oidc-token"),
            ("XAI_API_KEY", "environment-key"),
        ]),
    )
    .await;
}

#[tokio::test]
async fn reads_the_native_subscription_with_direct_auth_despite_ambient_gateway_credentials() {
    for provider in [
        M::Openai,
        M::Xai,
        M::GithubCopilot,
        M::Poe,
        M::OpencodeGo,
        M::Gitlab,
    ] {
        let process_env = env(&[
            ("AI_GATEWAY_API_KEY", "gateway-key"),
            ("VERCEL_OIDC_TOKEN", "oidc-token"),
        ]);
        let subscription = OpenCodeSubscription::new(provider, "subscription-token");
        let (read, calls) = reader(Some(subscription.clone()));
        let resolved = resolve(
            mode(C::Direct),
            provider.as_str(),
            process_env.clone(),
            read,
        )
        .await;
        assert_eq!(
            resolved,
            OpenCodeResolvedAuthentication {
                environment: env(&[("AI_SDK_OPENCODE_NATIVE_ACCESS_TOKEN", "subscription-token")]),
                authentication_mode: provider,
                subscription: Some(subscription),
            },
            "{provider:?}"
        );
        assert_eq!(*calls.lock().unwrap(), vec![(provider, process_env)]);
    }
}

#[tokio::test]
async fn keeps_resolving_an_ambient_oidc_token_to_gateway_auth() {
    for auth in [None, Some(HarnessAuthentication::Auto)] {
        let (read, calls) = reader(None);
        let resolved = resolve(
            auth,
            "github-copilot",
            env(&[("VERCEL_OIDC_TOKEN", "oidc-token")]),
            read,
        )
        .await;
        assert_eq!(
            resolved,
            OpenCodeResolvedAuthentication {
                environment: env(&[
                    ("AI_GATEWAY_API_KEY", "oidc-token"),
                    ("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1"),
                ]),
                authentication_mode: M::AiGateway,
                subscription: None,
            }
        );
        assert!(calls.lock().unwrap().is_empty());
    }
}

// --- resolveOpenCodeAuthenticationMode ---------------------------------------

#[test]
fn preserves_selected_provider_auth_despite_ambient_gateway_credentials() {
    assert_eq!(
        resolve_opencode_authentication_mode(
            mode(C::Openai).as_ref(),
            None,
            Some("openai"),
            &env(&[("AI_GATEWAY_API_KEY", "gateway-key")])
        ),
        M::Openai
    );
}

#[test]
fn resolves_direct_auth_despite_ambient_gateway_credentials() {
    for provider in [M::Xai, M::GithubCopilot, M::Poe, M::OpencodeGo, M::Gitlab] {
        assert_eq!(
            resolve_opencode_authentication_mode(
                mode(C::Direct).as_ref(),
                None,
                Some(provider.as_str()),
                &env(&[
                    ("AI_GATEWAY_API_KEY", "gateway-key"),
                    ("VERCEL_OIDC_TOKEN", "oidc-token"),
                ])
            ),
            provider
        );
    }
}

#[test]
fn resolves_ambient_gateway_credentials_to_gateway_auth() {
    assert_eq!(
        resolve_opencode_authentication_mode(
            None,
            None,
            None,
            &env(&[("VERCEL_OIDC_TOKEN", "oidc-token")])
        ),
        M::AiGateway
    );
}

// --- createOpenCodeRequestTransformations ------------------------------------

#[test]
fn restricts_gitlab_credentials_to_the_post_chat_exchange() {
    for (instance_url, host, path) in [
        (
            None,
            "gitlab.com",
            "/api/v4/ai/third_party_agents/direct_access",
        ),
        (
            Some("https://gitlab.example"),
            "gitlab.example",
            "/api/v4/ai/third_party_agents/direct_access",
        ),
        (
            Some("https://gitlab.example/gitlab"),
            "gitlab.example",
            "/gitlab/api/v4/ai/third_party_agents/direct_access",
        ),
        (
            Some("https://gitlab.example/"),
            "gitlab.example",
            "//api/v4/ai/third_party_agents/direct_access",
        ),
        (
            Some("https://gitlab.example/gitlab/"),
            "gitlab.example",
            "/gitlab//api/v4/ai/third_party_agents/direct_access",
        ),
    ] {
        let mut host_env = env(&[("GITLAB_TOKEN", "gitlab-secret")]);
        if let Some(instance_url) = instance_url {
            host_env.insert("GITLAB_INSTANCE_URL".into(), instance_url.into());
        }
        assert_eq!(
            transformations(
                host_env,
                env(&[("GITLAB_TOKEN", "sandbox-gitlab-placeholder")]),
                M::Gitlab
            ),
            json!([{
                "match": {
                    "host": host,
                    "path": { "exact": path },
                    "method": ["POST"],
                    "headers": [{
                        "key": { "exact": "Authorization" },
                        "value": { "exact": "Bearer sandbox-gitlab-placeholder" },
                    }],
                },
                "transform": { "headers": { "Authorization": "Bearer gitlab-secret" } },
            }]),
            "{instance_url:?}"
        );
    }
}

#[test]
fn omits_gitlab_transformations_when_a_credential_is_missing() {
    for (host_env, sandbox_env) in [
        (
            IndexMap::new(),
            env(&[("GITLAB_TOKEN", "sandbox-gitlab-placeholder")]),
        ),
        (env(&[("GITLAB_TOKEN", "gitlab-secret")]), IndexMap::new()),
    ] {
        assert_eq!(transformations(host_env, sandbox_env, M::Gitlab), json!([]));
    }
}

#[test]
fn uses_the_resolved_openai_route() {
    assert_eq!(
        transformations(
            env(&[
                ("OPENAI_API_KEY", "openai-secret"),
                ("OPENAI_BASE_URL", "https://openai.example/v1"),
                ("AI_GATEWAY_API_KEY", "unselected-gateway-secret"),
                (
                    "AI_GATEWAY_BASE_URL",
                    "https://unselected-gateway.example/v1"
                ),
            ]),
            env(&[("OPENAI_API_KEY", "sandbox-openai-secret")]),
            M::Openai
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
fn matches_both_supported_gateway_credential_headers() {
    let rule = |key: &str, value: &str| {
        json!({
            "match": {
                "host": "gateway.example",
                "path": { "startsWith": "/v1" },
                "headers": [{ "key": { "exact": key }, "value": { "exact": value } }],
            },
            "transform": { "headers": { "Authorization": "Bearer gateway-secret" } },
        })
    };
    assert_eq!(
        transformations(
            env(&[
                ("OPENAI_API_KEY", "unselected-openai-secret"),
                ("OPENAI_BASE_URL", "https://unselected-openai.example/v1"),
                ("AI_GATEWAY_API_KEY", "gateway-secret"),
                ("AI_GATEWAY_BASE_URL", "https://gateway.example/v1"),
            ]),
            env(&[("AI_GATEWAY_API_KEY", "sandbox-gateway-secret")]),
            M::AiGateway
        ),
        json!([
            rule("x-api-key", "sandbox-gateway-secret"),
            rule("Authorization", "Bearer sandbox-gateway-secret"),
        ])
    );
}

#[test]
fn injects_both_supported_anthropic_credential_headers() {
    assert_eq!(
        transformations(
            env(&[
                ("ANTHROPIC_API_KEY", "api-secret"),
                ("ANTHROPIC_AUTH_TOKEN", "token-secret"),
            ]),
            env(&[
                ("ANTHROPIC_API_KEY", "sandbox-api-secret"),
                ("ANTHROPIC_AUTH_TOKEN", "sandbox-token-secret"),
            ]),
            M::Anthropic
        ),
        json!([
            {
                "match": {
                    "host": "api.anthropic.com",
                    "headers": [{
                        "key": { "exact": "x-api-key" },
                        "value": { "exact": "sandbox-api-secret" },
                    }],
                },
                "transform": { "headers": { "x-api-key": "api-secret" } },
            },
            {
                "match": {
                    "host": "api.anthropic.com",
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

#[tokio::test]
async fn brokers_the_google_api_key_header() {
    let environment = env(&[("GOOGLE_GENERATIVE_AI_API_KEY", "google-secret")]);
    let forwarding: HarnessCredentialForwarding =
        Arc::new(|_| async { Ok("sandbox-google-secret".to_string()) }.boxed());
    let names: Vec<String> = OPENCODE_CREDENTIAL_ENVIRONMENT_VARIABLES
        .iter()
        .map(|name| name.to_string())
        .collect();
    let sandbox_environment =
        resolve_sandbox_credential_environment(&environment, &names, Some(&forwarding), None)
            .await
            .unwrap();
    assert_eq!(
        sandbox_environment,
        env(&[("GOOGLE_GENERATIVE_AI_API_KEY", "sandbox-google-secret")])
    );
    assert_eq!(
        transformations(environment, sandbox_environment, M::Google),
        json!([{
            "match": {
                "host": "generativelanguage.googleapis.com",
                "headers": [{
                    "key": { "exact": "x-goog-api-key" },
                    "value": { "exact": "sandbox-google-secret" },
                }],
            },
            "transform": { "headers": { "x-goog-api-key": "google-secret" } },
        }])
    );
}
