//! Ports of Pi's `github-copilot-oauth.test.ts` and the Copilot cases of
//! `oauth-auth.test.ts`, with `fetch` stubbed through [`OAuthFetch`], plus
//! tests of the ai.rs `ghr_` refresh-token divergence.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::auth::oauth::callback_server::tests::TestInteraction;
use crate::auth::{AuthType, Credential, CredentialStore, InMemoryCredentialStore};
use crate::models::{CreateModelsOptions, create_models};
use crate::providers::github_copilot::github_copilot_provider_with_oauth;

const TEST_COPILOT_ACCESS_TOKEN: &str =
    "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;";
const TEST_COPILOT_MODELS_URL: &str = "https://api.individual.githubcopilot.com/models";

type Handler = Arc<dyn Fn(&FetchRequest) -> Result<FetchResponse> + Send + Sync>;
type Log = Arc<Mutex<Vec<(FetchRequest, u64)>>>;

/// `vi.stubGlobal("fetch", ...)`: records each request with its time (ms
/// since the stub was made, on the tokio clock).
fn stub_fetch(handler: Handler) -> (OAuthFetch, Log) {
    let log: Log = Arc::default();
    let start = Instant::now();
    let seen = log.clone();
    let fetch: OAuthFetch = Arc::new(move |request, _signal| {
        let elapsed = Instant::now().duration_since(start).as_millis() as u64;
        seen.lock().push((request.clone(), elapsed));
        let result = handler(&request);
        Box::pin(async move { result })
    });
    (fetch, log)
}

fn json_response(body: Value) -> Result<FetchResponse> {
    Ok(FetchResponse::json(200, &body))
}

fn urls(log: &Log, filter: impl Fn(&str) -> bool) -> Vec<(String, u64)> {
    log.lock()
        .iter()
        .filter(|(request, _)| filter(&request.url))
        .map(|(request, at)| (request.url.clone(), *at))
        .collect()
}

fn model_id(index: usize) -> String {
    github_copilot_models()
        .keys()
        .nth(index)
        .expect("a GitHub Copilot model")
        .clone()
}

fn available_ids(credential: &OAuthCredential) -> Value {
    credential.extra[AVAILABLE_MODEL_IDS_KEY].clone()
}

/// Responses for every login step; `models` and `policy` serve the catalog
/// and policy endpoints.
fn login_handler(
    models: impl Fn() -> Result<FetchResponse> + Send + Sync + 'static,
    policy: impl Fn(&str) -> Result<FetchResponse> + Send + Sync + 'static,
) -> Handler {
    Arc::new(move |request| {
        let url = request.url.as_str();
        if url.ends_with("/login/device/code") {
            return json_response(json!({
                "device_code": "device-code",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "interval": 1,
                "expires_in": 900,
            }));
        }
        if url.ends_with("/login/oauth/access_token") {
            return json_response(json!({ "access_token": "ghu_refresh_token" }));
        }
        if url.contains("/copilot_internal/v2/token") {
            return json_response(
                json!({ "token": TEST_COPILOT_ACCESS_TOKEN, "expires_at": 9999999999u64 }),
            );
        }
        if url == TEST_COPILOT_MODELS_URL {
            return models();
        }
        if let Some(rest) = url.strip_prefix(&format!("{TEST_COPILOT_MODELS_URL}/"))
            && let Some(model) = rest.strip_suffix("/policy")
        {
            return policy(model);
        }
        panic!("Unexpected fetch URL: {url}");
    })
}

fn empty_catalog_handler() -> Handler {
    login_handler(
        || json_response(json!({ "data": [] })),
        |_| Ok(FetchResponse::new(200, "")),
    )
}

#[derive(Default)]
struct Seen {
    prompts: Vec<AuthPrompt>,
    events: Vec<AuthEvent>,
}

fn login_interaction(seen: Arc<Mutex<Seen>>, signal: CancellationToken) -> ProviderAuthInteraction {
    let prompts = seen.clone();
    ProviderAuthInteraction {
        interaction: Arc::new(TestInteraction {
            prompt: Arc::new(move |prompt| {
                assert!(
                    matches!(prompt.kind, AuthPromptKind::Text { .. }),
                    "Unexpected prompt: {prompt:?}"
                );
                prompts.lock().prompts.push(prompt);
                Box::pin(async { Ok(String::new()) })
            }),
            notify: Arc::new(move |event| seen.lock().events.push(event)),
        }),
        signal,
    }
}

async fn login_with(handler: Handler) -> (Result<OAuthCredential>, Log, Arc<Mutex<Seen>>) {
    let (fetch, log) = stub_fetch(handler);
    let seen: Arc<Mutex<Seen>> = Arc::default();
    let result = GitHubCopilotOAuth::with_fetch(fetch)
        .login(
            login_interaction(seen.clone(), CancellationToken::new()),
            LoginOptions::default(),
        )
        .await;
    (result, log, seen)
}

async fn refresh_models_for_test(data: Value, proxy_host: &str) -> Result<OAuthCredential> {
    let access_token = format!("tid=test;exp=9999999999;proxy-ep={proxy_host};");
    let models_url = format!(
        "https://{}/models",
        proxy_host.replacen("proxy.", "api.", 1)
    );
    let (fetch, _) = stub_fetch(Arc::new(move |request| {
        if request.url.contains("/copilot_internal/v2/token") {
            return json_response(json!({ "token": access_token, "expires_at": 9999999999u64 }));
        }
        if request.url == models_url {
            assert_eq!(
                request.header_value("Authorization"),
                Some(format!("Bearer {access_token}").as_str())
            );
            return json_response(json!({ "data": data }));
        }
        panic!("Unexpected fetch URL: {}", request.url);
    }));
    GitHubCopilotOAuth::with_fetch(fetch)
        .refresh(
            OAuthCredential {
                access: "old-access-token".to_string(),
                refresh: "ghu_refresh_token".to_string(),
                expires: 0,
                extra: Map::new(),
            },
            CancellationToken::new(),
        )
        .await
}

async fn available_through_models(credential: OAuthCredential) -> Vec<String> {
    let store = Arc::new(InMemoryCredentialStore::new());
    store
        .modify(
            "github-copilot",
            Box::new(move |_| Box::pin(async move { Ok(Some(Credential::OAuth(credential))) })),
            Default::default(),
        )
        .await
        .unwrap();
    let models = create_models(CreateModelsOptions {
        credentials: Some(store),
        ..Default::default()
    });
    models.set_provider(github_copilot_provider_with_oauth(github_copilot_oauth()));
    models
        .get_available(Some("github-copilot"), Default::default())
        .await
        .unwrap()
        .into_iter()
        .map(|model| model.id)
        .collect()
}

#[tokio::test]
async fn filters_models_to_the_authenticated_account_picker_catalog() {
    let (picker, disabled, hidden) = (model_id(0), model_id(1), model_id(2));
    let credential = refresh_models_for_test(
        json!([
            {
                "id": picker,
                "model_picker_enabled": true,
                "capabilities": { "supports": { "tool_calls": true } },
            },
            {
                "id": disabled,
                "model_picker_enabled": true,
                "policy": { "state": "disabled" },
                "capabilities": { "supports": { "tool_calls": true } },
            },
            {
                "id": hidden,
                "model_picker_enabled": false,
                "policy": { "state": "enabled" },
                "capabilities": { "supports": { "tool_calls": true } },
            },
        ]),
        "proxy.individual.githubcopilot.com",
    )
    .await
    .unwrap();
    assert_eq!(available_ids(&credential), json!([picker]));
    assert_eq!(available_through_models(credential).await, vec![picker]);
}

#[tokio::test]
async fn falls_back_to_explicitly_enabled_policy_models_when_the_picker_catalog_is_empty() {
    let enabled = model_id(0);
    let credential = refresh_models_for_test(
        json!([
            {
                "id": enabled,
                "model_picker_enabled": false,
                "policy": { "state": "enabled" },
                "capabilities": { "supports": { "tool_calls": true } },
            },
            {
                "id": "policy-disabled-model",
                "model_picker_enabled": false,
                "policy": { "state": "disabled" },
                "capabilities": { "supports": { "tool_calls": true } },
            },
            {
                "id": "unconfigured-model",
                "model_picker_enabled": false,
                "capabilities": { "supports": { "tool_calls": true } },
            },
            {
                "id": "tool-incapable-model",
                "model_picker_enabled": false,
                "policy": { "state": "enabled" },
                "capabilities": { "supports": { "tool_calls": false } },
            },
        ]),
        "proxy.individual.githubcopilot.com",
    )
    .await
    .unwrap();
    assert_eq!(available_ids(&credential), json!([enabled]));
    assert_eq!(available_through_models(credential).await, vec![enabled]);
}

#[tokio::test]
async fn does_not_fall_back_to_policy_models_for_non_individual_accounts() {
    let credential = refresh_models_for_test(
        json!([{
            "id": "gpt-4.1",
            "model_picker_enabled": false,
            "policy": { "state": "enabled" },
            "capabilities": { "supports": { "tool_calls": true } },
        }]),
        "proxy.business.githubcopilot.com",
    )
    .await
    .unwrap();
    assert_eq!(available_ids(&credential), json!([]));
}

#[tokio::test]
async fn does_not_retry_model_catalog_throttling_during_credential_refresh() {
    let (fetch, log) = stub_fetch(Arc::new(|request| {
        if request.url.contains("/copilot_internal/v2/token") {
            return json_response(
                json!({ "token": TEST_COPILOT_ACCESS_TOKEN, "expires_at": 9999999999u64 }),
            );
        }
        if request.url == TEST_COPILOT_MODELS_URL {
            return Ok(
                FetchResponse::json(429, &json!({ "error": "too many requests" }))
                    .with_header("Retry-After", "0"),
            );
        }
        panic!("Unexpected fetch URL: {}", request.url);
    }));
    let error = GitHubCopilotOAuth::with_fetch(fetch)
        .refresh(
            OAuthCredential {
                access: "old-access-token".to_string(),
                refresh: "ghu_refresh_token".to_string(),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("429"), "{error}");
    assert_eq!(urls(&log, |url| url == TEST_COPILOT_MODELS_URL).len(), 1);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn reports_device_code_details_through_notify() {
    let (result, _, seen) = login_with(empty_catalog_handler()).await;
    let credential = result.unwrap();
    assert_eq!(credential.access, TEST_COPILOT_ACCESS_TOKEN);
    assert_eq!(credential.refresh, "ghu_refresh_token");
    let seen = seen.lock();
    assert_eq!(
        seen.events[0],
        AuthEvent::DeviceCode {
            user_code: "ABCD-EFGH".to_string(),
            verification_uri: "https://github.com/login/device".to_string(),
            interval_seconds: Some(1),
            expires_in_seconds: Some(900),
        }
    );
    assert_eq!(
        seen.prompts[0].kind,
        AuthPromptKind::Text {
            message: "GitHub Enterprise URL/domain (blank for github.com)".to_string(),
            placeholder: Some("company.ghe.com".to_string()),
        }
    );
    // No policy work, so no progress message.
    assert_eq!(seen.events.len(), 1);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn updates_only_known_tool_capable_unconfigured_account_model_policies() {
    let (configured, unconfigured, tool_incapable) = (model_id(0), model_id(1), model_id(2));
    let catalog = json!({
        "data": [
            {
                "id": configured,
                "model_picker_enabled": true,
                "policy": { "state": "enabled" },
                "capabilities": { "supports": { "tool_calls": true } },
            },
            {
                "id": unconfigured,
                "model_picker_enabled": true,
                "policy": { "state": "unconfigured" },
                "capabilities": { "supports": { "tool_calls": true } },
            },
            {
                "id": "remote-only-model",
                "model_picker_enabled": true,
                "policy": { "state": "unconfigured" },
                "capabilities": { "supports": { "tool_calls": true } },
            },
            {
                "id": tool_incapable,
                "model_picker_enabled": true,
                "policy": { "state": "unconfigured" },
                "capabilities": { "supports": { "tool_calls": false } },
            },
        ],
    });
    let (result, log, seen) = login_with(login_handler(
        move || json_response(catalog.clone()),
        |_| Ok(FetchResponse::new(200, "")),
    ))
    .await;
    let credential = result.unwrap();
    assert_eq!(urls(&log, |url| url == TEST_COPILOT_MODELS_URL).len(), 1);
    let policies = urls(&log, |url| url.ends_with("/policy"));
    assert_eq!(
        policies
            .iter()
            .map(|(url, _)| url.as_str())
            .collect::<Vec<_>>(),
        vec![format!("{TEST_COPILOT_MODELS_URL}/{unconfigured}/policy")]
    );
    let policy_request = log
        .lock()
        .iter()
        .find(|(request, _)| request.url.ends_with("/policy"))
        .unwrap()
        .0
        .clone();
    assert_eq!(policy_request.method, "POST");
    assert_eq!(
        policy_request.header_value("openai-intent"),
        Some("chat-policy")
    );
    assert_eq!(
        policy_request.body.as_deref(),
        Some(r#"{"state":"enabled"}"#)
    );
    assert!(seen.lock().events.contains(&AuthEvent::Progress {
        message: "Enabling models...".to_string(),
    }));
    assert_eq!(
        available_ids(&credential),
        json!([configured, unconfigured, "remote-only-model"])
    );
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn retries_a_throttled_policy_update_after_retry_after() {
    let model = model_id(0);
    let catalog = json!({
        "data": [{ "id": model, "model_picker_enabled": true, "policy": { "state": "unconfigured" } }],
    });
    let attempts = Arc::new(Mutex::new(0));
    let (result, log, _) = login_with(login_handler(
        move || json_response(catalog.clone()),
        move |_| {
            let mut attempts = attempts.lock();
            *attempts += 1;
            Ok(if *attempts == 1 {
                FetchResponse::json(429, &json!({ "error": "too many requests" }))
                    .with_header("Retry-After", "1")
            } else {
                FetchResponse::new(200, "")
            })
        },
    ))
    .await;
    result.unwrap();
    let times: Vec<u64> = urls(&log, |url| url.ends_with("/policy"))
        .into_iter()
        .map(|(_, at)| at)
        .collect();
    assert_eq!(times, vec![1000, 2000]);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn continues_policy_updates_after_a_transport_failure() {
    let ids = [model_id(0), model_id(1)];
    let catalog = json!({
        "data": ids
            .iter()
            .map(|id| json!({ "id": id, "model_picker_enabled": true, "policy": { "state": "unconfigured" } }))
            .collect::<Vec<_>>(),
    });
    let attempts = Arc::new(Mutex::new(0));
    let (result, log, _) = login_with(login_handler(
        move || json_response(catalog.clone()),
        move |_| {
            let mut attempts = attempts.lock();
            *attempts += 1;
            if *attempts == 1 {
                return Err(Error::message("fetch failed"));
            }
            Ok(FetchResponse::new(200, ""))
        },
    ))
    .await;
    let credential = result.unwrap();
    let policies: Vec<String> = urls(&log, |url| url.ends_with("/policy"))
        .into_iter()
        .map(|(url, _)| url)
        .collect();
    assert_eq!(
        policies,
        ids.iter()
            .map(|id| format!("{TEST_COPILOT_MODELS_URL}/{id}/policy"))
            .collect::<Vec<_>>()
    );
    assert_eq!(available_ids(&credential), json!(ids));
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn stops_policy_updates_and_persists_authentication_when_the_retry_delay_exceeds_the_budget()
{
    let (first, second) = (model_id(0), model_id(1));
    let catalog = json!({
        "data": [
            { "id": first, "model_picker_enabled": true, "policy": { "state": "unconfigured" } },
            { "id": second, "model_picker_enabled": true, "policy": { "state": "unconfigured" } },
        ],
    });
    let (fetch, log) = stub_fetch(login_handler(
        move || json_response(catalog.clone()),
        |_| {
            Ok(
                FetchResponse::json(429, &json!({ "error": "too many requests" }))
                    .with_header("Retry-After", "5"),
            )
        },
    ));
    let store = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(store.clone()),
        ..Default::default()
    });
    models.set_provider(github_copilot_provider_with_oauth(Arc::new(
        GitHubCopilotOAuth::with_fetch(fetch),
    )));
    let interaction = login_interaction(Arc::default(), CancellationToken::new());
    let credential = models
        .login(
            "github-copilot",
            AuthType::OAuth,
            interaction.interaction,
            LoginOptions::default(),
        )
        .await
        .unwrap();
    let Credential::OAuth(oauth) = &credential else {
        panic!("expected an OAuth credential");
    };
    assert_eq!(oauth.access, TEST_COPILOT_ACCESS_TOKEN);
    assert_eq!(
        urls(&log, |url| url.ends_with("/policy"))
            .into_iter()
            .map(|(url, _)| url)
            .collect::<Vec<_>>(),
        vec![format!("{TEST_COPILOT_MODELS_URL}/{first}/policy")]
    );
    assert_eq!(
        store
            .read("github-copilot", Default::default())
            .await
            .unwrap(),
        Some(credential)
    );
}

fn device_code_handler(verification_uri: &'static str) -> Handler {
    let base = empty_catalog_handler();
    Arc::new(move |request| {
        if request.url.ends_with("/login/device/code") {
            return json_response(json!({
                "device_code": "device-code",
                "user_code": "ABCD-EFGH",
                "verification_uri": verification_uri,
                "interval": 1,
                "expires_in": 900,
            }));
        }
        base(request)
    })
}

#[tokio::test]
async fn rejects_a_non_http_verification_uri_before_it_is_reported() {
    let (result, log, seen) = login_with(device_code_handler("$(id>/tmp/pwned)")).await;
    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("Untrusted verification_uri"),
        "{error}"
    );
    assert!(seen.lock().events.is_empty());
    assert_eq!(log.lock().len(), 1);
    assert!(normalize_verification_uri("javascript:alert(1)").is_err());
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn normalizes_the_verification_uri_before_it_is_reported() {
    let raw = "https://github.com/login/\u{1b}]8;;evil";
    let (result, _, seen) = login_with(device_code_handler(raw)).await;
    result.unwrap();
    let AuthEvent::DeviceCode {
        verification_uri, ..
    } = &seen.lock().events[0]
    else {
        panic!("expected a device code event");
    };
    assert_eq!(verification_uri, "https://github.com/login/%1B]8;;evil");
    assert_ne!(verification_uri, raw);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn waits_before_polling_and_increases_the_interval_after_slow_down() {
    let responses = Arc::new(Mutex::new(std::collections::VecDeque::from([
        json!({ "error": "authorization_pending", "error_description": "pending" }),
        json!({ "error": "slow_down", "error_description": "slow down", "interval": 7 }),
        json!({ "access_token": "ghu_refresh_token" }),
    ])));
    let base = empty_catalog_handler();
    let (result, log, _) = login_with(Arc::new(move |request| {
        if request.url.ends_with("/login/device/code") {
            assert_eq!(request.method, "POST");
            assert_eq!(request.header_value("Accept"), Some("application/json"));
            assert_eq!(
                request.header_value("Content-Type"),
                Some("application/x-www-form-urlencoded")
            );
            let body = request.body.as_deref().unwrap();
            assert!(body.contains("client_id="));
            assert!(body.contains("scope=read%3Auser"));
            return json_response(json!({
                "device_code": "device-code",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "interval": 5,
                "expires_in": 900,
            }));
        }
        if request.url.ends_with("/login/oauth/access_token") {
            let body = request.body.as_deref().unwrap();
            assert!(body.contains("device_code=device-code"));
            assert!(
                body.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code")
            );
            let response = responses
                .lock()
                .pop_front()
                .expect("Unexpected extra access token poll");
            return json_response(response);
        }
        base(request)
    }))
    .await;
    result.unwrap();
    let polls: Vec<u64> = urls(&log, |url| url.ends_with("/login/oauth/access_token"))
        .into_iter()
        .map(|(_, at)| at)
        .collect();
    assert_eq!(polls, vec![5000, 10_000, 17_000]);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn times_out_after_repeated_slow_down_responses() {
    let responses = Arc::new(Mutex::new(std::collections::VecDeque::from([
        json!({ "error": "slow_down", "error_description": "slow down" }),
        json!({ "error": "slow_down", "error_description": "still too fast" }),
        json!({ "error": "authorization_pending", "error_description": "pending" }),
    ])));
    let (result, log, _) = login_with(Arc::new(move |request| {
        if request.url.ends_with("/login/device/code") {
            return json_response(json!({
                "device_code": "device-code",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "interval": 5,
                "expires_in": 25,
            }));
        }
        if request.url.ends_with("/login/oauth/access_token") {
            let response = responses
                .lock()
                .pop_front()
                .expect("Unexpected extra access token poll");
            return json_response(response);
        }
        panic!("Unexpected fetch URL: {}", request.url);
    }))
    .await;
    let error = result.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Device flow timed out after one or more slow_down responses"),
        "{error}"
    );
    let polls: Vec<u64> = urls(&log, |url| url.ends_with("/login/oauth/access_token"))
        .into_iter()
        .map(|(_, at)| at)
        .collect();
    assert_eq!(polls, vec![5000, 15_000]);
}

#[tokio::test]
async fn rejects_an_invalid_enterprise_domain() {
    let (fetch, log) = stub_fetch(empty_catalog_handler());
    let interaction = ProviderAuthInteraction {
        interaction: Arc::new(TestInteraction {
            prompt: Arc::new(|_| Box::pin(async { Ok("not a host".to_string()) })),
            notify: Arc::new(|_| {}),
        }),
        signal: CancellationToken::new(),
    };
    let error = GitHubCopilotOAuth::with_fetch(fetch)
        .login(interaction, LoginOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "Invalid GitHub Enterprise URL/domain");
    assert!(log.lock().is_empty());
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn enterprise_logins_use_the_enterprise_hosts_and_store_the_domain() {
    let (fetch, log) = stub_fetch(Arc::new(|request| {
        let url = request.url.as_str();
        if url == "https://company.ghe.com/login/device/code" {
            return json_response(json!({
                "device_code": "d",
                "user_code": "u",
                "verification_uri": "https://company.ghe.com/login/device",
                "expires_in": 900,
            }));
        }
        if url == "https://company.ghe.com/login/oauth/access_token" {
            return json_response(json!({ "access_token": "ghu_token" }));
        }
        if url == "https://api.company.ghe.com/copilot_internal/v2/token" {
            return json_response(json!({ "token": "no-proxy-ep", "expires_at": 9999999999u64 }));
        }
        if url == "https://copilot-api.company.ghe.com/models" {
            return json_response(json!({ "data": [] }));
        }
        panic!("Unexpected fetch URL: {url}");
    }));
    let interaction = ProviderAuthInteraction {
        interaction: Arc::new(TestInteraction {
            prompt: Arc::new(|_| Box::pin(async { Ok(" https://company.ghe.com/x ".to_string()) })),
            notify: Arc::new(|_| {}),
        }),
        signal: CancellationToken::new(),
    };
    let credential = GitHubCopilotOAuth::with_fetch(fetch)
        .login(interaction, LoginOptions::default())
        .await
        .unwrap();
    assert_eq!(credential.extra[ENTERPRISE_URL_KEY], "company.ghe.com");
    // Without `interval` the device flow waits the RFC 8628 default of 5 s.
    assert_eq!(urls(&log, |url| url.ends_with("/access_token"))[0].1, 5000);
    let auth = github_copilot_oauth().to_auth(&credential).await.unwrap();
    assert_eq!(
        auth.base_url.as_deref(),
        Some("https://copilot-api.company.ghe.com")
    );
}

#[tokio::test]
async fn to_auth_derives_the_base_url_from_the_token_proxy_endpoint() {
    let access = "tid=abc;exp=123;proxy-ep=proxy.enterprise.example;rest";
    let auth = github_copilot_oauth()
        .to_auth(&OAuthCredential {
            access: access.to_string(),
            refresh: "r".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        auth,
        ModelAuth {
            api_key: Some(access.to_string()),
            base_url: Some("https://api.enterprise.example".to_string()),
            ..Default::default()
        }
    );
}

#[tokio::test]
async fn to_auth_falls_back_to_the_enterprise_domain_then_the_individual_endpoint() {
    let oauth = github_copilot_oauth();
    let mut credential = OAuthCredential {
        access: "no-proxy-ep".to_string(),
        refresh: "r".to_string(),
        ..Default::default()
    };
    credential.extra.insert(
        ENTERPRISE_URL_KEY.to_string(),
        json!("https://company.ghe.com"),
    );
    let enterprise = oauth.to_auth(&credential).await.unwrap();
    assert_eq!(
        enterprise.base_url.as_deref(),
        Some("https://copilot-api.company.ghe.com")
    );
    credential.extra.clear();
    let individual = oauth.to_auth(&credential).await.unwrap();
    assert_eq!(individual.base_url.as_deref(), Some(DEFAULT_BASE_URL));
    assert_eq!(oauth.is_subscription(), Some(true));
}

#[tokio::test]
async fn refresh_preserves_the_enterprise_domain() {
    let (fetch, log) = stub_fetch(Arc::new(|request| {
        if request.url.ends_with("/models") {
            return json_response(json!({ "data": [] }));
        }
        json_response(json!({ "token": "new-token", "expires_at": 9999999999u64 }))
    }));
    let mut credential = OAuthCredential {
        access: "old".to_string(),
        refresh: "gh-token".to_string(),
        ..Default::default()
    };
    credential
        .extra
        .insert(ENTERPRISE_URL_KEY.to_string(), json!("company.ghe.com"));
    let refreshed = GitHubCopilotOAuth::with_fetch(fetch)
        .refresh(credential, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(refreshed.access, "new-token");
    assert_eq!(refreshed.refresh, "gh-token");
    assert_eq!(refreshed.expires, 9999999999 * 1000 - EXPIRY_SKEW_MS);
    assert_eq!(refreshed.extra[ENTERPRISE_URL_KEY], "company.ghe.com");
    let first = log.lock()[0].0.clone();
    assert_eq!(
        first.url,
        "https://api.company.ghe.com/copilot_internal/v2/token"
    );
    assert_eq!(first.header_value("Authorization"), Some("Bearer gh-token"));
    assert_eq!(first.header_value("Editor-Version"), Some("vscode/1.107.0"));
    let models = log.lock()[1].0.clone();
    assert_eq!(
        models.header_value("X-GitHub-Api-Version"),
        Some(COPILOT_API_VERSION)
    );
}

#[tokio::test]
async fn resolves_stored_oauth_credentials_through_models_get_auth_with_the_credential_base_url() {
    let access = "tid=abc;exp=123;proxy-ep=proxy.business.githubcopilot.com;rest";
    let store = Arc::new(InMemoryCredentialStore::new());
    let credential = Credential::OAuth(OAuthCredential {
        access: access.to_string(),
        refresh: "r".to_string(),
        // Beyond get_auth()'s refresh window.
        expires: now_millis() + 10 * 60_000,
        extra: Map::new(),
    });
    store
        .modify(
            "github-copilot",
            Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
            Default::default(),
        )
        .await
        .unwrap();
    let models = create_models(CreateModelsOptions {
        credentials: Some(store),
        ..Default::default()
    });
    models.set_provider(crate::providers::github_copilot::github_copilot_provider());
    let model = models.get_models(Some("github-copilot"))[0].clone();
    let result = models
        .get_auth(&model.provider, Default::default())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.auth.api_key.as_deref(), Some(access));
    assert_eq!(
        result.auth.base_url.as_deref(),
        Some("https://api.business.githubcopilot.com")
    );
    assert_eq!(result.source.as_deref(), Some("OAuth"));
}

#[test]
fn parses_device_token_responses() {
    assert_eq!(
        parse_device_token_response(
            &json!({ "access_token": "ghu", "refresh_token": "ghr", "expires_in": 28_800 })
        ),
        OAuthDeviceCodePollResult::Complete(GitHubUserToken {
            access_token: "ghu".to_string(),
            refresh_token: Some("ghr".to_string()),
            expires_in: Some(28_800),
        })
    );
    assert_eq!(
        parse_device_token_response(&json!({ "error": "authorization_pending" })),
        OAuthDeviceCodePollResult::Pending
    );
    assert_eq!(
        parse_device_token_response(&json!({ "error": "slow_down", "interval": 7 })),
        OAuthDeviceCodePollResult::SlowDown {
            interval_seconds: Some(7.0)
        }
    );
    assert_eq!(
        parse_device_token_response(
            &json!({ "error": "access_denied", "error_description": "denied" })
        ),
        OAuthDeviceCodePollResult::Failed {
            message: "Device flow failed: access_denied: denied".to_string()
        }
    );
    assert_eq!(
        parse_device_token_response(&json!({ "access_token": 1 })),
        OAuthDeviceCodePollResult::Failed {
            message: "Invalid device token response".to_string()
        }
    );
}

#[test]
fn resolves_base_urls_and_normalizes_domains() {
    assert_eq!(
        get_github_copilot_base_url(
            Some("tid=test;proxy-ep=proxy.individual.githubcopilot.com;exp=1"),
            None
        ),
        DEFAULT_BASE_URL
    );
    assert_eq!(
        get_github_copilot_base_url(None, Some("company.ghe.com")),
        "https://copilot-api.company.ghe.com"
    );
    assert_eq!(get_github_copilot_base_url(None, None), DEFAULT_BASE_URL);
    assert_eq!(normalize_domain(""), None);
    assert_eq!(
        normalize_domain("https://company.ghe.com/path").as_deref(),
        Some("company.ghe.com")
    );
    assert_eq!(
        normalize_domain("company.ghe.com").as_deref(),
        Some("company.ghe.com")
    );
    assert_eq!(normalize_domain("not a host"), None);
    assert_eq!(parse_float_prefix("1.5abc"), Some(1.5));
    assert_eq!(parse_float_prefix("Wed, 21 Oct 2015 07:28:00 GMT"), None);
}

#[test]
fn modify_models_filters_available_models_and_sets_the_credential_base_url() {
    let mut credential = OAuthCredential {
        access: "tid=test;proxy-ep=proxy.enterprise.example.com;exp=1".to_string(),
        ..Default::default()
    };
    credential
        .extra
        .insert(AVAILABLE_MODEL_IDS_KEY.to_string(), json!(["gpt-4.1"]));
    let model = |id: &str, provider: &str| Model {
        id: id.to_string(),
        provider: provider.to_string(),
        base_url: "https://old.example.com".to_string(),
        ..Default::default()
    };
    let updated = modify_github_copilot_models(
        vec![
            model("gpt-4.1", "github-copilot"),
            model("claude-opus-4.7", "github-copilot"),
            model("claude-opus-4.7", "anthropic"),
        ],
        &credential,
    );
    assert_eq!(
        updated
            .iter()
            .map(|model| (
                model.provider.as_str(),
                model.id.as_str(),
                model.base_url.as_str()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                "github-copilot",
                "gpt-4.1",
                "https://api.enterprise.example.com"
            ),
            ("anthropic", "claude-opus-4.7", "https://old.example.com"),
        ]
    );
}

// --- ai.rs divergence: the GitHub `ghr_` refresh-token grant ---

fn renewal_handler(mint_statuses: Vec<u16>) -> Handler {
    let mint_statuses = Arc::new(Mutex::new(std::collections::VecDeque::from(mint_statuses)));
    Arc::new(move |request| {
        let url = request.url.as_str();
        if url.ends_with("/login/oauth/access_token") {
            return json_response(json!({
                "access_token": "ghu_renewed",
                "refresh_token": "ghr_rotated",
                "expires_in": 28_800,
            }));
        }
        if url.contains("/copilot_internal/v2/token") {
            let status = mint_statuses.lock().pop_front().unwrap_or(200);
            if status != 200 {
                return Ok(FetchResponse::new(status, "Bad credentials"));
            }
            return json_response(
                json!({ "token": TEST_COPILOT_ACCESS_TOKEN, "expires_at": 9999999999u64 }),
            );
        }
        if url == TEST_COPILOT_MODELS_URL {
            return json_response(json!({ "data": [] }));
        }
        panic!("Unexpected fetch URL: {url}");
    })
}

fn expiring_credential(github_access_expires: u64) -> OAuthCredential {
    let mut credential = OAuthCredential {
        access: "old".to_string(),
        refresh: "ghu_old".to_string(),
        ..Default::default()
    };
    credential
        .extra
        .insert(GITHUB_REFRESH_TOKEN_KEY.to_string(), json!("ghr_old"));
    credential.extra.insert(
        GITHUB_ACCESS_EXPIRES_KEY.to_string(),
        json!(github_access_expires),
    );
    credential
}

#[tokio::test]
async fn renews_an_expired_github_token_before_minting() {
    let (fetch, log) = stub_fetch(renewal_handler(vec![]));
    let refreshed = GitHubCopilotOAuth::with_fetch(fetch)
        .refresh(expiring_credential(1), CancellationToken::new())
        .await
        .unwrap();
    let log = log.lock();
    let grant = &log[0].0;
    assert_eq!(grant.url, "https://github.com/login/oauth/access_token");
    let body = grant.body.as_deref().unwrap();
    assert!(body.contains("grant_type=refresh_token"), "{body}");
    assert!(body.contains("refresh_token=ghr_old"), "{body}");
    assert!(body.contains("client_id=Iv1.b507a08c87ecfe98"), "{body}");
    assert_eq!(
        log[1].0.header_value("Authorization"),
        Some("Bearer ghu_renewed")
    );
    assert_eq!(refreshed.refresh, "ghu_renewed");
    assert_eq!(refreshed.extra[GITHUB_REFRESH_TOKEN_KEY], "ghr_rotated");
    assert!(refreshed.extra[GITHUB_ACCESS_EXPIRES_KEY].as_u64().unwrap() > now_millis());
}

#[tokio::test]
async fn renews_the_github_token_once_when_a_mint_is_rejected() {
    let (fetch, log) = stub_fetch(renewal_handler(vec![401]));
    let refreshed = GitHubCopilotOAuth::with_fetch(fetch)
        .refresh(
            expiring_credential(now_millis() + 3_600_000),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let urls: Vec<String> = log.lock().iter().map(|(r, _)| r.url.clone()).collect();
    assert_eq!(
        urls,
        vec![
            "https://api.github.com/copilot_internal/v2/token",
            "https://github.com/login/oauth/access_token",
            "https://api.github.com/copilot_internal/v2/token",
            TEST_COPILOT_MODELS_URL,
        ]
    );
    assert_eq!(refreshed.refresh, "ghu_renewed");
    assert_eq!(refreshed.access, TEST_COPILOT_ACCESS_TOKEN);
}

#[tokio::test]
async fn without_a_refresh_token_a_rejected_mint_fails_as_in_pi() {
    let (fetch, log) = stub_fetch(renewal_handler(vec![401]));
    let error = GitHubCopilotOAuth::with_fetch(fetch)
        .refresh(
            OAuthCredential {
                refresh: "ghu_old".to_string(),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "401 Unauthorized: Bad credentials");
    assert_eq!(log.lock().len(), 1);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn login_stores_the_github_refresh_token_and_expiry() {
    let base = empty_catalog_handler();
    let (result, _, _) = login_with(Arc::new(move |request| {
        if request.url.ends_with("/login/oauth/access_token") {
            return json_response(json!({
                "access_token": "ghu_new",
                "refresh_token": "ghr_new",
                "expires_in": 28_800,
            }));
        }
        base(request)
    }))
    .await;
    let credential = result.unwrap();
    assert_eq!(credential.refresh, "ghu_new");
    assert_eq!(credential.extra[GITHUB_REFRESH_TOKEN_KEY], "ghr_new");
    assert!(credential.extra.contains_key(GITHUB_ACCESS_EXPIRES_KEY));
    let json = serde_json::to_value(Credential::OAuth(credential)).unwrap();
    assert_eq!(json["githubRefreshToken"], "ghr_new");
    assert_eq!(json["type"], "oauth");
}
