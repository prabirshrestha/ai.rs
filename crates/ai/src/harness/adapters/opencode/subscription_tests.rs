//! Port of `opencode-subscription.test.ts`.

use std::sync::{Arc, Mutex};

use futures::FutureExt as _;
use indexmap::IndexMap;
use serde_json::{Value, json};

use super::*;
use crate::harness::adapters::test_temp_dir::mkdtemp;
use crate::harness::utils::{OAuthFetchRequest, OAuthFetchResponse};

use OpenCodeResolvedAuthenticationMode as M;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn auth_content(provider: M, record: Value) -> IndexMap<String, String> {
    IndexMap::from([(
        "OPENCODE_AUTH_CONTENT".to_string(),
        json!({ provider.as_str(): record }).to_string(),
    )])
}

fn fetch_returning(text: Value) -> (OAuthFetch, Arc<Mutex<Vec<OAuthFetchRequest>>>) {
    let calls: Arc<Mutex<Vec<OAuthFetchRequest>>> = Arc::default();
    let recorded = calls.clone();
    let fetch: OAuthFetch = Arc::new(move |request| {
        recorded.lock().unwrap().push(request);
        let text = text.to_string();
        async move { Ok(OAuthFetchResponse { status: 200, text }) }.boxed()
    });
    (fetch, calls)
}

#[tokio::test]
async fn reads_subscriptions_from_opencode_auth_content() {
    let hour = now_ms() + 60 * 60 * 1000;
    for (provider, record, access_token) in [
        (
            M::Xai,
            json!({ "type": "oauth", "access": "xai-access", "refresh": "xai-refresh", "expires": hour }),
            "xai-access",
        ),
        (
            M::GithubCopilot,
            json!({ "type": "oauth", "access": "github-token", "refresh": "github-token", "expires": 0 }),
            "github-token",
        ),
        (
            M::Poe,
            json!({ "type": "oauth", "access": "poe-key", "refresh": "poe-key", "expires": hour }),
            "poe-key",
        ),
        (
            M::OpencodeGo,
            json!({ "type": "api", "key": "go-key" }),
            "go-key",
        ),
        (
            M::Gitlab,
            json!({ "type": "oauth", "access": "gitlab-access", "refresh": "gitlab-refresh", "expires": hour }),
            "gitlab-access",
        ),
    ] {
        let subscription = read_opencode_subscription(ReadOpenCodeSubscriptionOptions {
            provider_id: provider,
            env: Some(auth_content(provider, record)),
            ..Default::default()
        })
        .await
        .unwrap()
        .unwrap_or_else(|| panic!("{provider:?}"));
        assert_eq!(subscription.provider_id, provider);
        assert_eq!(subscription.access_token, access_token);
    }
}

#[tokio::test]
async fn refreshes_an_expiring_openai_record_in_the_xdg_auth_store() {
    let home = mkdtemp("opencode-auth-");
    let auth_directory = home.path().join(".local/share/opencode");
    let auth_path = auth_directory.join("auth.json");
    std::fs::create_dir_all(&auth_directory).unwrap();
    std::fs::write(
        &auth_path,
        json!({
            "openai": {
                "type": "oauth",
                "access": "old-access",
                "refresh": "old-refresh",
                "expires": now_ms() + 60_000,
                "accountId": "account-1",
            },
        })
        .to_string(),
    )
    .unwrap();
    let (fetch, calls) = fetch_returning(json!({
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_in": 3600,
    }));

    let subscription = read_opencode_subscription(ReadOpenCodeSubscriptionOptions {
        provider_id: M::Openai,
        env: Some(IndexMap::new()),
        home_directory: Some(home.path_string()),
        platform: Some("linux".into()),
        fetch: Some(fetch),
    })
    .await
    .unwrap();
    assert_eq!(
        subscription,
        Some(OpenCodeSubscription {
            provider_id: M::Openai,
            access_token: "new-access".into(),
            account_id: Some("account-1".into()),
            enterprise_url: None,
        })
    );
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].url, "https://auth.openai.com/oauth/token");
    assert!(calls[0].body.contains("refresh_token=old-refresh"));
    assert!(
        std::fs::read_to_string(&auth_path)
            .unwrap()
            .contains("new-refresh")
    );
}

#[tokio::test]
async fn reads_the_gitlab_plugin_auth_store_on_windows() {
    let home = mkdtemp("opencode-auth-");
    let auth_directory = home.path().join(".opencode");
    std::fs::create_dir_all(&auth_directory).unwrap();
    std::fs::write(
        auth_directory.join("auth.json"),
        json!({
            "gitlab": {
                "type": "oauth",
                "access": "gitlab-access",
                "refresh": "gitlab-refresh",
                "expires": now_ms() + 60 * 60 * 1000,
                "enterpriseUrl": "https://gitlab.example.com",
            },
        })
        .to_string(),
    )
    .unwrap();

    assert_eq!(
        read_opencode_subscription(ReadOpenCodeSubscriptionOptions {
            provider_id: M::Gitlab,
            env: Some(IndexMap::new()),
            home_directory: Some(home.path_string()),
            platform: Some("win32".into()),
            fetch: None,
        })
        .await
        .unwrap(),
        Some(OpenCodeSubscription {
            provider_id: M::Gitlab,
            access_token: "gitlab-access".into(),
            account_id: None,
            enterprise_url: Some("https://gitlab.example.com".into()),
        })
    );
}

#[tokio::test]
async fn reads_the_opencode_data_directory_auth_store_on_windows() {
    let home = mkdtemp("opencode-auth-");
    let auth_directory = home.path().join(".local/share/opencode");
    std::fs::create_dir_all(&auth_directory).unwrap();
    std::fs::write(
        auth_directory.join("auth.json"),
        json!({ "xai": { "type": "api", "key": "xai-key" } }).to_string(),
    )
    .unwrap();

    assert_eq!(
        read_opencode_subscription(ReadOpenCodeSubscriptionOptions {
            provider_id: M::Xai,
            env: Some(IndexMap::new()),
            home_directory: Some(home.path_string()),
            platform: Some("win32".into()),
            fetch: None,
        })
        .await
        .unwrap(),
        Some(OpenCodeSubscription::new(M::Xai, "xai-key"))
    );
}

#[tokio::test]
async fn falls_back_to_the_gitlab_plugin_auth_store_on_windows_for_providers_missing_from_the_data_directory()
 {
    let home = mkdtemp("opencode-auth-");
    let data_directory = home.path().join(".local/share/opencode");
    let plugin_directory = home.path().join(".opencode");
    std::fs::create_dir_all(&data_directory).unwrap();
    std::fs::create_dir_all(&plugin_directory).unwrap();
    std::fs::write(
        data_directory.join("auth.json"),
        json!({ "xai": { "type": "api", "key": "xai-key" } }).to_string(),
    )
    .unwrap();
    std::fs::write(
        plugin_directory.join("auth.json"),
        json!({ "gitlab": { "type": "api", "key": "gitlab-key" } }).to_string(),
    )
    .unwrap();

    assert_eq!(
        read_opencode_subscription(ReadOpenCodeSubscriptionOptions {
            provider_id: M::Gitlab,
            env: Some(IndexMap::new()),
            home_directory: Some(home.path_string()),
            platform: Some("win32".into()),
            fetch: None,
        })
        .await
        .unwrap(),
        Some(OpenCodeSubscription::new(M::Gitlab, "gitlab-key"))
    );
}

// --- OpenCode GitLab direct access -------------------------------------------

#[tokio::test]
async fn requests_an_ai_access_token_from_gitlab_on_the_host() {
    let (fetch, calls) = fetch_returning(json!({
        "token": "direct-access-token",
        "headers": { "x-gitlab-token": "routing-token" },
    }));
    assert_eq!(
        request_opencode_gitlab_direct_access(RequestOpenCodeGitLabDirectAccessOptions {
            access_token: "oauth-access-token".into(),
            instance_url: Some("https://gitlab.example.com/".into()),
            ai_gateway_url: Some("https://gateway.example.com/".into()),
            fetch: Some(fetch),
        })
        .await
        .unwrap(),
        OpenCodeGitLabDirectAccess {
            access_token: "direct-access-token".into(),
            headers: IndexMap::from([("x-gitlab-token".to_string(), "routing-token".to_string())]),
            ai_gateway_url: "https://gateway.example.com".into(),
        }
    );
    let calls = calls.lock().unwrap();
    assert_eq!(
        *calls,
        vec![OAuthFetchRequest {
            url: "https://gitlab.example.com/api/v4/ai/third_party_agents/direct_access".into(),
            method: "POST".into(),
            headers: IndexMap::from([
                (
                    "Authorization".to_string(),
                    "Bearer oauth-access-token".to_string()
                ),
                ("Content-Type".to_string(), "application/json".to_string()),
            ]),
            body: json!({ "feature_flags": { "DuoAgentPlatformNext": true } }).to_string(),
        }]
    );
}

#[tokio::test]
async fn rejects_invalid_direct_access_credentials_without_exposing_them() {
    let (fetch, _) = fetch_returning(json!({ "token": "sensitive-token" }));
    let error = request_opencode_gitlab_direct_access(RequestOpenCodeGitLabDirectAccessOptions {
        access_token: "sensitive-oauth-token".into(),
        fetch: Some(fetch),
        ..Default::default()
    })
    .await
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("OpenCode GitLab direct access returned invalid credentials."),
        "{error}"
    );
    assert!(!error.contains("sensitive"), "{error}");
}

#[test]
fn brokers_the_ai_access_token_and_response_headers_for_both_proxy_apis() {
    let transformations = create_opencode_gitlab_subscription_request_transformations(
        &OpenCodeGitLabDirectAccess {
            access_token: "host-direct-access-token".into(),
            headers: IndexMap::from([
                ("x-gitlab-token".to_string(), "routing-token".to_string()),
                ("x-api-key".to_string(), "must-not-be-forwarded".to_string()),
                (
                    "Authorization".to_string(),
                    "must-not-be-forwarded".to_string(),
                ),
            ]),
            ai_gateway_url: "https://cloud.gitlab.test".into(),
        },
        "sandbox-access-token",
    )
    .unwrap();
    let rule = |path: &str| {
        json!({
            "match": {
                "host": "cloud.gitlab.test",
                "path": { "startsWith": path },
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer sandbox-access-token" },
                }],
            },
            "transform": {
                "headers": {
                    "x-gitlab-token": "routing-token",
                    "Authorization": "Bearer host-direct-access-token",
                },
            },
        })
    };
    assert_eq!(
        serde_json::to_value(transformations).unwrap(),
        json!([
            rule("/ai/v1/proxy/anthropic"),
            rule("/ai/v1/proxy/openai/v1")
        ])
    );
}

/// `toMatchObject`: every field of `expected` is in `actual`.
fn assert_matches_object(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            for (key, expected) in expected {
                let actual = actual
                    .get(key)
                    .unwrap_or_else(|| panic!("missing {key} in {actual:?}"));
                assert_matches_object(actual, expected);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
fn configures_native_providers_for_each_gitlab_chat_api() {
    let opencode_config =
        json!({ "provider": { "custom": { "options": { "apiKey": "custom" } } } });
    let config = create_opencode_gitlab_subscription_config(
        opencode_config.as_object(),
        "sandbox-access-token",
        "https://cloud.gitlab.test",
        None,
    );
    let providers = &config["provider"];
    assert_eq!(
        providers["custom"],
        json!({ "options": { "apiKey": "custom" } })
    );
    assert_matches_object(
        &providers["ai-sdk-gitlab-anthropic"],
        &json!({
            "npm": "@ai-sdk/anthropic",
            "api": "https://cloud.gitlab.test/ai/v1/proxy/anthropic/v1",
            "models": { "duo-chat-sonnet-4-5": { "id": "claude-sonnet-4-5-20250929" } },
        }),
    );
    assert_matches_object(
        &providers["ai-sdk-gitlab-openai-chat"],
        &json!({
            "npm": "@ai-sdk/openai-compatible",
            "models": { "duo-chat-gpt-5-4": { "id": "gpt-5.4-2026-03-05" } },
        }),
    );
    assert_matches_object(
        &providers["ai-sdk-gitlab-openai-responses"],
        &json!({
            "npm": "@ai-sdk/openai",
            "models": { "duo-chat-gpt-5-6-sol": { "id": "gpt-5.6-sol" } },
        }),
    );
}

#[test]
fn maps_gitlab_chat_models_and_rejects_workflow_models() {
    assert_eq!(
        resolve_opencode_gitlab_subscription_model(
            Some("gitlab/duo-chat-sonnet-4-5"),
            Some("gitlab")
        )
        .unwrap()
        .as_deref(),
        Some("ai-sdk-gitlab-anthropic/duo-chat-sonnet-4-5")
    );
    assert_eq!(
        resolve_opencode_gitlab_subscription_model(Some("duo-chat-gpt-5-4"), Some("gitlab"))
            .unwrap()
            .as_deref(),
        Some("ai-sdk-gitlab-openai-chat/duo-chat-gpt-5-4")
    );
    let error =
        resolve_opencode_gitlab_subscription_model(Some("gitlab/duo-workflow"), Some("gitlab"))
            .unwrap_err()
            .to_string();
    assert!(
        error.contains("because it authenticates over WebSocket"),
        "{error}"
    );
}

// --- OpenCode sandbox authentication -----------------------------------------

fn openai_subscription() -> OpenCodeSubscription {
    OpenCodeSubscription {
        provider_id: M::Openai,
        access_token: "host-access".into(),
        account_id: Some("account-1".into()),
        enterprise_url: None,
    }
}

#[test]
fn creates_a_sanitized_non_expiring_oauth_record() {
    let content =
        create_opencode_subscription_auth_content(&openai_subscription(), "sandbox-access");
    assert_eq!(
        serde_json::from_str::<Value>(&content).unwrap(),
        json!({
            "openai": {
                "type": "oauth",
                "access": "sandbox-access",
                "refresh": "host-managed",
                "expires": 9007199254740991u64,
                "accountId": "account-1",
            },
        })
    );
    assert!(!content.contains("host-access"));
}

#[test]
fn brokers_the_sandbox_token_at_the_subscription_request_route() {
    assert_eq!(
        serde_json::to_value(
            create_opencode_subscription_request_transformations(
                &openai_subscription(),
                "sandbox-access"
            )
            .unwrap()
        )
        .unwrap(),
        json!([{
            "match": {
                "host": "chatgpt.com",
                "path": { "startsWith": "/backend-api/codex" },
                "headers": [{
                    "key": { "exact": "Authorization" },
                    "value": { "exact": "Bearer sandbox-access" },
                }],
            },
            "transform": {
                "headers": {
                    "Authorization": "Bearer host-access",
                    "ChatGPT-Account-Id": "account-1",
                },
            },
        }])
    );
}
