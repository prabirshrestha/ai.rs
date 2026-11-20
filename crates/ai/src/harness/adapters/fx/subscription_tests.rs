//! Port of `harness-fx/src/fx-subscription.test.ts`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use base64::Engine as _;
use indexmap::IndexMap;
use serde_json::{Value, json};

use super::*;
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};

const SCOPE: &str = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828";
const EXPIRES_AT: f64 = 2_000_000_000_000.0;

fn fx_credentials(access_token: &str) -> Value {
    json!({
        "version": 1,
        "access_token": access_token,
        "refresh_token": "refresh",
        "account_id": "account",
        "expires_at_ms": EXPIRES_AT as i64,
    })
}

fn codex_credentials() -> Value {
    json!({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "native-codex",
            "refresh_token": "refresh",
            "account_id": "account",
            "id_token": "preserve-id",
        },
        "extra": true,
    })
}

fn grok_credentials() -> Value {
    json!({
        SCOPE: {
            "key": "native-grok",
            "refresh_token": "refresh",
            "user_id": "account",
            "expires_at": iso_timestamp(EXPIRES_AT as i64),
            "extra": true,
        },
        "other": { "key": "preserve" },
    })
}

fn put(directory: &TempDir, file: &str, value: &Value) {
    let path = directory.path().join(file);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, value.to_string()).unwrap();
}

fn read_json(directory: &TempDir, file: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(directory.path().join(file)).unwrap()).unwrap()
}

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn options(directory: &TempDir) -> ResolveFxAuthenticationOptions {
    ResolveFxAuthenticationOptions {
        env: Some(IndexMap::new()),
        home_directory: Some(directory.path().to_path_buf()),
        ..Default::default()
    }
}

async fn resolve(options: ResolveFxAuthenticationOptions) -> HarnessResult<FxResolvedAuthentication> {
    resolve_fx_authentication(options).await
}

fn subscription(auth: FxResolvedAuthentication) -> FxSubscription {
    match auth {
        FxResolvedAuthentication::Subscription(subscription) => subscription,
        FxResolvedAuthentication::Gateway { .. } => panic!("Expected subscription"),
    }
}

fn error_text<T: std::fmt::Debug>(result: HarnessResult<T>) -> String {
    result.expect_err("expected an error").to_string()
}

fn credentials(access_token: &str) -> FxProviderCredentials {
    FxProviderCredentials {
        access_token: access_token.into(),
        refresh_token: None,
        account_id: None,
        expires_at: None,
    }
}

#[tokio::test]
async fn uses_fx_codex_fx_grok_native_codex_then_native_grok() {
    let directory = mkdtemp("fx-subscription-");
    let sources = [
        (".fx/chatgpt-auth.json", fx_credentials("fx-codex"), FxProvider::Codex, "fx-codex"),
        (".fx/grok-auth.json", fx_credentials("fx-grok"), FxProvider::Grok, "fx-grok"),
        (".codex/auth.json", codex_credentials(), FxProvider::Codex, "native-codex"),
        (".grok/auth.json", grok_credentials(), FxProvider::Grok, "native-grok"),
    ];
    for (file, value, ..) in &sources {
        put(&directory, file, value);
    }
    for (file, _, provider, token) in &sources {
        let auth = resolve(options(&directory)).await.unwrap();
        assert_eq!(auth.provider(), *provider);
        assert_eq!(subscription(auth).read().await.unwrap().access_token, *token);
        std::fs::remove_file(directory.path().join(file)).unwrap();
    }
    assert!(error_text(resolve(options(&directory)).await).contains("No subscription credentials were available"));
}

#[tokio::test]
async fn prefers_gateway_without_direct_mode() {
    for auth in [None, Some(HarnessAuthentication::Auto), Some(HarnessAuthentication::AiGateway)] {
        let directory = mkdtemp("fx-subscription-");
        put(&directory, ".fx/chatgpt-auth.json", &fx_credentials("subscription"));
        let resolved = resolve(ResolveFxAuthenticationOptions {
            auth,
            env: Some(env(&[("AI_GATEWAY_API_KEY", "gateway")])),
            ..options(&directory)
        })
        .await
        .unwrap();
        let FxResolvedAuthentication::Gateway { api_key, base_url } = resolved else {
            panic!("Expected Gateway");
        };
        assert_eq!(api_key, "gateway");
        assert_eq!(base_url, "https://ai-gateway.vercel.sh");
    }
}

#[tokio::test]
async fn discovers_subscriptions_in_direct_mode_and_retains_gateway_fallback() {
    let directory = mkdtemp("fx-subscription-");
    put(&directory, ".fx/chatgpt-auth.json", &fx_credentials("subscription"));
    let direct = || Some(HarnessAuthentication::Choice(HarnessDirectAuthentication::Direct));
    let resolved = resolve(ResolveFxAuthenticationOptions {
        auth: direct(),
        env: Some(env(&[("AI_GATEWAY_API_KEY", "gateway")])),
        ..options(&directory)
    })
    .await
    .unwrap();
    assert_eq!(resolved.provider(), FxProvider::Codex);
    std::fs::remove_file(directory.path().join(".fx/chatgpt-auth.json")).unwrap();
    let resolved = resolve(ResolveFxAuthenticationOptions {
        auth: direct(),
        env: Some(env(&[("VERCEL_OIDC_TOKEN", "gateway")])),
        ..options(&directory)
    })
    .await
    .unwrap();
    assert_eq!(resolved.provider(), FxProvider::Gateway);
}

#[tokio::test]
async fn isolates_explicit_authentication_environments() {
    let directory = mkdtemp("fx-subscription-");
    put(&directory, ".fx/chatgpt-auth.json", &fx_credentials("subscription"));
    let result = resolve(ResolveFxAuthenticationOptions {
        auth: Some(HarnessAuthentication::Environment(IndexMap::new())),
        env: Some(env(&[("AI_GATEWAY_API_KEY", "host")])),
        ..options(&directory)
    })
    .await;
    assert!(error_text(result).contains("fx requires"));
    let resolved = resolve(ResolveFxAuthenticationOptions {
        auth: Some(HarnessAuthentication::Environment(env(&[("AI_GATEWAY_API_KEY", "explicit")]))),
        ..options(&directory)
    })
    .await
    .unwrap();
    assert_eq!(resolved.provider(), FxProvider::Gateway);
    let result = resolve(ResolveFxAuthenticationOptions {
        auth: Some(HarnessAuthentication::AiGateway),
        ..options(&directory)
    })
    .await;
    assert!(error_text(result).contains("fx requires"));
}

#[tokio::test]
async fn skips_malformed_stores_and_ignores_native_api_key_records() {
    let directory = mkdtemp("fx-subscription-");
    put(&directory, ".fx/chatgpt-auth.json", &json!({ "version": 2 }));
    put(&directory, ".fx/grok-auth.json", &json!({}));
    put(
        &directory,
        ".codex/auth.json",
        &json!({ "auth_mode": "apikey", "OPENAI_API_KEY": "ignore" }),
    );
    put(&directory, ".grok/auth.json", &grok_credentials());
    assert_eq!(resolve(options(&directory)).await.unwrap().provider(), FxProvider::Grok);
}

#[tokio::test]
async fn honors_native_home_overrides() {
    let directory = mkdtemp("fx-subscription-");
    put(&directory, "custom-codex/auth.json", &codex_credentials());
    put(&directory, "custom-grok/auth.json", &grok_credentials());
    let homes = env(&[
        ("CODEX_HOME", &directory.path().join("custom-codex").to_string_lossy()),
        ("GROK_HOME", &directory.path().join("custom-grok").to_string_lossy()),
    ]);
    let with_env = || ResolveFxAuthenticationOptions {
        env: Some(homes.clone()),
        ..options(&directory)
    };
    assert_eq!(resolve(with_env()).await.unwrap().provider(), FxProvider::Codex);
    std::fs::remove_file(directory.path().join("custom-codex/auth.json")).unwrap();
    assert_eq!(resolve(with_env()).await.unwrap().provider(), FxProvider::Grok);
}

#[tokio::test]
async fn pins_the_provider_on_resume_without_overriding_an_explicit_gateway_mode() {
    let directory = mkdtemp("fx-subscription-");
    put(&directory, ".fx/chatgpt-auth.json", &fx_credentials("codex"));
    put(&directory, ".fx/grok-auth.json", &fx_credentials("grok"));
    let gateway = || Some(env(&[("AI_GATEWAY_API_KEY", "gateway")]));
    let resolved = resolve(ResolveFxAuthenticationOptions {
        required_provider: Some(FxProvider::Grok),
        env: gateway(),
        ..options(&directory)
    })
    .await
    .unwrap();
    assert_eq!(resolved.provider(), FxProvider::Grok);
    let resolved = resolve(ResolveFxAuthenticationOptions {
        required_provider: Some(FxProvider::Gateway),
        auth: Some(HarnessAuthentication::Choice(HarnessDirectAuthentication::Direct)),
        env: gateway(),
        ..options(&directory)
    })
    .await
    .unwrap();
    assert_eq!(resolved.provider(), FxProvider::Gateway);
    let result = resolve(ResolveFxAuthenticationOptions {
        required_provider: Some(FxProvider::Codex),
        auth: Some(HarnessAuthentication::AiGateway),
        ..options(&directory)
    })
    .await;
    assert!(error_text(result).contains("does not allow"));
    std::fs::remove_file(directory.path().join(".fx/chatgpt-auth.json")).unwrap();
    let result = resolve(ResolveFxAuthenticationOptions {
        required_provider: Some(FxProvider::Codex),
        env: gateway(),
        ..options(&directory)
    })
    .await;
    assert!(error_text(result).contains("no codex subscription"));
}

#[tokio::test]
async fn rereads_and_updates_the_selected_fx_store() {
    for name in ["chatgpt", "grok"] {
        let directory = mkdtemp("fx-subscription-");
        let file = format!(".fx/{name}-auth.json");
        let with_extra = |token: &str| {
            let mut value = fx_credentials(token);
            value["extra"] = json!(true);
            value
        };
        put(&directory, &file, &with_extra("initial"));
        let auth = subscription(resolve(options(&directory)).await.unwrap());
        put(&directory, &file, &with_extra("external-rotation"));
        assert_eq!(auth.read().await.unwrap().access_token, "external-rotation");
        auth.write(&FxProviderCredentials {
            access_token: "rotated".into(),
            refresh_token: Some("rotated-refresh".into()),
            account_id: Some("account".into()),
            expires_at: Some(EXPIRES_AT + 1000.0),
        })
        .await
        .unwrap();
        assert_eq!(
            read_json(&directory, &file),
            json!({
                "access_token": "rotated",
                "account_id": "account",
                "expires_at_ms": 2_000_000_001_000_i64,
                "extra": true,
                "refresh_token": "rotated-refresh",
                "version": 1,
            })
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(directory.path().join(&file)).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let entries: Vec<String> = std::fs::read_dir(directory.path().join(".fx"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec![format!("{name}-auth.json")]);
        std::fs::remove_file(directory.path().join(&file)).unwrap();
        assert!(error_text(auth.read().await).contains("no longer available"));
        assert!(error_text(auth.write(&credentials("lost")).await).contains("Cannot update"));
    }
}

#[tokio::test]
async fn preserves_native_codex_metadata_and_derives_jwt_expiry() {
    let directory = mkdtemp("fx-subscription-");
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json!({ "exp": 2_000_000_000 }).to_string());
    let token = format!("header.{payload}.signature");
    let mut value = codex_credentials();
    value["tokens"]["access_token"] = json!(token);
    put(&directory, ".codex/auth.json", &value);
    let auth = subscription(resolve(options(&directory)).await.unwrap());
    assert_eq!(auth.read().await.unwrap().expires_at, Some(EXPIRES_AT));
    auth.write(&credentials("rotated")).await.unwrap();
    let saved = read_json(&directory, ".codex/auth.json");
    assert_eq!(
        saved["tokens"],
        json!({
            "access_token": "rotated",
            "account_id": "account",
            "id_token": "preserve-id",
            "refresh_token": "refresh",
        })
    );
    assert_eq!(saved["extra"], json!(true));
    assert!(parse_iso_millis(saved["last_refresh"].as_str().unwrap()).is_some());
}

#[tokio::test]
async fn reads_grok_expiry_and_writes_native_compatible_timestamps() {
    for expiry in [
        json!(iso_timestamp(EXPIRES_AT as i64)),
        json!(EXPIRES_AT as i64),
        json!((EXPIRES_AT / 1000.0) as i64),
    ] {
        let directory = mkdtemp("fx-subscription-");
        let mut value = grok_credentials();
        value[SCOPE]["expires_at"] = expiry;
        put(&directory, ".grok/auth.json", &value);
        let auth = subscription(resolve(options(&directory)).await.unwrap());
        assert_eq!(
            auth.read().await.unwrap(),
            FxProviderCredentials {
                access_token: "native-grok".into(),
                refresh_token: Some("refresh".into()),
                account_id: Some("account".into()),
                expires_at: Some(EXPIRES_AT),
            }
        );
        auth.write(&FxProviderCredentials {
            expires_at: Some(EXPIRES_AT + 1000.0),
            ..credentials("rotated")
        })
        .await
        .unwrap();
        let mut expected = grok_credentials();
        expected[SCOPE]["key"] = json!("rotated");
        expected[SCOPE]["expires_at"] = json!(iso_timestamp(EXPIRES_AT as i64 + 1000));
        assert_eq!(read_json(&directory, ".grok/auth.json"), expected);
    }
}

#[tokio::test]
async fn does_not_fall_back_after_a_selected_store_disappears() {
    let directory = mkdtemp("fx-subscription-");
    put(&directory, ".fx/chatgpt-auth.json", &fx_credentials("selected"));
    put(&directory, ".fx/grok-auth.json", &fx_credentials("fallback"));
    let auth = subscription(resolve(options(&directory)).await.unwrap());
    std::fs::remove_file(directory.path().join(".fx/chatgpt-auth.json")).unwrap();
    assert!(error_text(auth.read().await).contains("no longer available"));
}

#[derive(Default)]
struct MockKeyring {
    stored: Mutex<Option<String>>,
    reads: Mutex<Vec<(String, String)>>,
    fail_next_write: Mutex<bool>,
}

#[async_trait]
impl FxCodexKeyring for MockKeyring {
    async fn read(&self, service: &str, account: &str) -> Option<String> {
        self.reads.lock().unwrap().push((service.into(), account.into()));
        self.stored.lock().unwrap().clone()
    }

    async fn write(&self, _service: &str, _account: &str, value: &str) -> HarnessResult<()> {
        if std::mem::take(&mut *self.fail_next_write.lock().unwrap()) {
            return Err(HarnessError::new("write failed").into());
        }
        *self.stored.lock().unwrap() = Some(value.into());
        Ok(())
    }
}

#[tokio::test]
async fn honors_each_native_codex_credential_store_mode() {
    for mode in ["file", "keyring", "auto", "ephemeral"] {
        let directory = mkdtemp("fx-subscription-");
        put(&directory, ".codex/auth.json", &codex_credentials());
        std::fs::write(
            directory.path().join(".codex/config.toml"),
            format!("cli_auth_credentials_store = \"{mode}\"\n"),
        )
        .unwrap();
        let mut in_keyring = codex_credentials();
        in_keyring["tokens"]["access_token"] = json!("keyring");
        let keyring = Arc::new(MockKeyring {
            stored: Mutex::new(Some(in_keyring.to_string())),
            ..Default::default()
        });
        let with_keyring = || ResolveFxAuthenticationOptions {
            keyring: Some(keyring.clone()),
            ..options(&directory)
        };
        if mode == "ephemeral" {
            assert!(error_text(resolve(with_keyring()).await).contains("No subscription credentials"));
            assert!(keyring.reads.lock().unwrap().is_empty());
            continue;
        }
        let auth = subscription(resolve(with_keyring()).await.unwrap());
        assert_eq!(
            auth.read().await.unwrap().access_token,
            if mode == "file" { "native-codex" } else { "keyring" }
        );
        auth.write(&credentials("rotated")).await.unwrap();
        if mode == "file" {
            assert!(keyring.reads.lock().unwrap().is_empty());
            continue;
        }
        let canonical = std::fs::canonicalize(directory.path().join(".codex")).unwrap();
        let digest = ring::digest::digest(&ring::digest::SHA256, canonical.to_string_lossy().as_bytes());
        let hex: String = digest.as_ref().iter().map(|byte| format!("{byte:02x}")).collect();
        let account = format!("cli|{}", &hex[..16]);
        assert!(keyring
            .reads
            .lock()
            .unwrap()
            .contains(&("Codex Auth".to_string(), account)));
        let stored: Value = serde_json::from_str(keyring.stored.lock().unwrap().as_deref().unwrap()).unwrap();
        assert_eq!(stored["tokens"]["access_token"], json!("rotated"));
        assert_eq!(
            read_json(&directory, ".codex/auth.json")["tokens"]["access_token"],
            json!("native-codex")
        );
        *keyring.fail_next_write.lock().unwrap() = true;
        assert!(error_text(auth.write(&credentials("failure")).await).contains("write failed"));
    }
}

#[tokio::test]
async fn falls_back_to_the_file_in_auto_mode_when_the_keyring_has_no_credentials() {
    let directory = mkdtemp("fx-subscription-");
    put(&directory, ".codex/auth.json", &codex_credentials());
    std::fs::write(
        directory.path().join(".codex/config.toml"),
        "cli_auth_credentials_store = \"auto\"",
    )
    .unwrap();
    let auth = subscription(
        resolve(ResolveFxAuthenticationOptions {
            keyring: Some(Arc::new(MockKeyring::default())),
            ..options(&directory)
        })
        .await
        .unwrap(),
    );
    assert_eq!(auth.read().await.unwrap().access_token, "native-codex");
}

#[tokio::test]
async fn resolve_fx_model_prefers_the_explicit_model() {
    let directory = mkdtemp("fx-subscription-");
    put(&directory, ".fx/settings.json", &json!({ "models": { "grok": "saved-grok" } }));
    assert_eq!(
        resolve_fx_model(FxProvider::Grok, Some("explicit".into()), Some(directory.path())).await,
        Some("explicit".into())
    );
}

#[tokio::test]
async fn resolve_fx_model_leaves_the_gateway_model_to_libfx() {
    let directory = mkdtemp("fx-subscription-");
    assert_eq!(resolve_fx_model(FxProvider::Gateway, None, Some(directory.path())).await, None);
}

#[tokio::test]
async fn resolve_fx_model_uses_the_model_saved_for_the_subscription_provider() {
    let directory = mkdtemp("fx-subscription-");
    put(
        &directory,
        ".fx/settings.json",
        &json!({
            "codex_model": "legacy-codex",
            "grok_model": "legacy-grok",
            "models": { "gateway": "saved-gateway", "codex": "saved-codex" },
        }),
    );
    assert_eq!(
        resolve_fx_model(FxProvider::Codex, None, Some(directory.path())).await,
        Some("saved-codex".into())
    );
    assert_eq!(
        resolve_fx_model(FxProvider::Grok, None, Some(directory.path())).await,
        Some("legacy-grok".into())
    );
}

#[tokio::test]
async fn resolve_fx_model_returns_none_when_no_subscription_model_is_saved() {
    let directory = mkdtemp("fx-subscription-");
    put(&directory, ".fx/settings.json", &json!({ "models": { "gateway": "saved-gateway" } }));
    assert_eq!(resolve_fx_model(FxProvider::Grok, None, Some(directory.path())).await, None);
}

#[test]
fn parses_iso_timestamps_like_date_parse() {
    assert_eq!(parse_iso_millis("2033-05-18T03:33:20.000Z"), Some(EXPIRES_AT));
    assert_eq!(parse_iso_millis("2033-05-18T03:33:20Z"), Some(EXPIRES_AT));
    assert_eq!(parse_iso_millis("2033-05-18T05:33:20+02:00"), Some(EXPIRES_AT));
    assert_eq!(parse_iso_millis("1970-01-01"), Some(0.0));
    assert_eq!(parse_iso_millis("not a date"), None);
}
