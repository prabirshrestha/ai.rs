//! Port of `harness-github-copilot/src/github-copilot-subscription.test.ts`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::FutureExt as _;
use indexmap::IndexMap;
use parking_lot::Mutex;

use super::*;
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
use crate::harness::auth::{HarnessAuthentication, HarnessDirectAuthentication};

fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn direct() -> HarnessAuthentication {
    HarnessAuthentication::Choice(HarnessDirectAuthentication::Direct)
}

/// `vi.fn()` standing in for `readSubscription`.
fn read_subscription(
    result: Option<GitHubCopilotSubscription>,
) -> (ReadGitHubCopilotSubscription, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    (
        Arc::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            let result = result.clone();
            async move { Ok(result) }.boxed()
        }),
        calls,
    )
}

// resolveGitHubCopilotSubscriptionEnvironment
#[tokio::test]
async fn uses_an_explicit_authentication_environment_without_native_discovery() {
    let auth = env(&[("COPILOT_GITHUB_TOKEN", "explicit-token")]);
    let (read, calls) = read_subscription(None);
    assert_eq!(
        resolve_github_copilot_subscription_environment(
            Some(&HarnessAuthentication::Environment(auth.clone())),
            env(&[("COPILOT_GITHUB_TOKEN", "process-token")]),
            Some(read),
        )
        .await
        .unwrap(),
        auth
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn does_not_inspect_native_credentials_for_explicit_gateway_auth() {
    let environment = env(&[("AI_GATEWAY_API_KEY", "gateway-token")]);
    let (read, calls) = read_subscription(None);
    assert_eq!(
        resolve_github_copilot_subscription_environment(
            Some(&HarnessAuthentication::AiGateway),
            environment.clone(),
            Some(read),
        )
        .await
        .unwrap(),
        environment
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn does_not_inspect_native_credentials_when_auto_selects_gateway_auth() {
    let environment = env(&[("AI_GATEWAY_API_KEY", "gateway-token")]);
    let (read, calls) = read_subscription(None);
    assert_eq!(
        resolve_github_copilot_subscription_environment(
            Some(&HarnessAuthentication::Auto),
            environment.clone(),
            Some(read),
        )
        .await
        .unwrap(),
        environment
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn prefers_each_token_variable_over_native_credentials() {
    for variable in ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"] {
        let environment = env(&[(variable, "environment-token")]);
        let (read, calls) = read_subscription(None);
        assert_eq!(
            resolve_github_copilot_subscription_environment(
                Some(&direct()),
                environment.clone(),
                Some(read),
            )
            .await
            .unwrap(),
            environment
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0, "{variable}");
    }
}

#[tokio::test]
async fn discovers_native_credentials_for_direct_auth_even_with_gateway_credentials() {
    let (read, _) = read_subscription(Some(GitHubCopilotSubscription {
        token: "subscription-token".into(),
        host: "https://enterprise.example".into(),
    }));
    assert_eq!(
        resolve_github_copilot_subscription_environment(
            Some(&direct()),
            env(&[("AI_GATEWAY_API_KEY", "gateway-token")]),
            Some(read),
        )
        .await
        .unwrap(),
        env(&[
            ("AI_GATEWAY_API_KEY", "gateway-token"),
            ("COPILOT_GITHUB_TOKEN", "subscription-token"),
            ("COPILOT_GH_HOST", "enterprise.example"),
        ])
    );
}

#[tokio::test]
async fn keeps_the_original_environment_when_discovery_finds_no_token() {
    let environment = env(&[("PATH", "/usr/bin")]);
    let (read, _) = read_subscription(None);
    assert_eq!(
        resolve_github_copilot_subscription_environment(
            Some(&direct()),
            environment.clone(),
            Some(read),
        )
        .await
        .unwrap(),
        environment
    );
}

// readGitHubCopilotSubscription

/// The mocked `readLinuxSecretServicePassword`, `readMacOSKeychainPassword`
/// and `readWindowsCredentialManagerPassword`.
#[derive(Default)]
struct MockStores {
    result: Option<String>,
    macos: Mutex<Vec<(String, String)>>,
    linux: Mutex<Vec<IndexMap<String, String>>>,
    windows: Mutex<Vec<String>>,
}

#[async_trait]
impl SecureCredentialStores for MockStores {
    async fn read_macos_keychain_password(&self, service: &str, account: &str) -> Option<String> {
        self.macos.lock().push((service.into(), account.into()));
        self.result.clone()
    }

    async fn read_linux_secret_service_password(
        &self,
        attributes: &IndexMap<String, String>,
    ) -> Option<String> {
        self.linux.lock().push(attributes.clone());
        self.result.clone()
    }

    async fn read_windows_credential_manager_password(&self, target_name: &str) -> Option<String> {
        self.windows.lock().push(target_name.into());
        self.result.clone()
    }
}

fn create_copilot_home(config: &str) -> TempDir {
    let directory = mkdtemp("copilot-home-");
    std::fs::write(directory.path().join("config.json"), config).unwrap();
    directory
}

fn no_gh() -> Option<FindGitHubCliExecutable> {
    Some(Arc::new(|_, _| async { None }.boxed()))
}

fn gh_at(path: &'static str) -> Option<FindGitHubCliExecutable> {
    Some(Arc::new(move |_, _| {
        async move { Some(path.to_string()) }.boxed()
    }))
}

type SecureCall = (String, String);

/// `vi.fn(async () => result)` standing in for `readSecureCredential`.
fn secure_credential(
    result: Option<&'static str>,
) -> (Option<ReadSecureCredential>, Arc<Mutex<Vec<SecureCall>>>) {
    let calls: Arc<Mutex<Vec<SecureCall>>> = Arc::default();
    let recorded = calls.clone();
    (
        Some(Arc::new(move |service, account| {
            recorded.lock().push((service, account));
            async move { Ok(result.map(str::to_string)) }.boxed()
        })),
        calls,
    )
}

type CliCall = (String, String, IndexMap<String, String>);

fn gh_token(
    result: Option<&'static str>,
) -> (Option<ReadGitHubCliToken>, Arc<Mutex<Vec<CliCall>>>) {
    let calls: Arc<Mutex<Vec<CliCall>>> = Arc::default();
    let recorded = calls.clone();
    (
        Some(Arc::new(move |executable, hostname, env| {
            recorded.lock().push((executable, hostname, env));
            async move { Ok(result.map(str::to_string)) }.boxed()
        })),
        calls,
    )
}

fn home_options(
    home: &TempDir,
    platform: &str,
    stores: &Arc<MockStores>,
) -> ReadGitHubCopilotSubscriptionOptions {
    ReadGitHubCopilotSubscriptionOptions {
        env: Some(env(&[("COPILOT_HOME", &home.path_string())])),
        platform: Some(platform.into()),
        find_github_cli_executable: no_gh(),
        secure_credential_stores: Some(stores.clone()),
        ..Default::default()
    }
}

fn subscription(token: &str, host: &str) -> Option<GitHubCopilotSubscription> {
    Some(GitHubCopilotSubscription {
        token: token.into(),
        host: host.into(),
    })
}

#[tokio::test]
async fn reads_the_last_logged_in_account_from_the_macos_keychain() {
    let home = create_copilot_home(
        r#"
        // This file is managed by Copilot CLI.
        {
          "last_logged_in_user": {
            "host": "https://enterprise.example/",
            "login": "last-user",
          },
          "logged_in_users": [
            { "host": "https://github.com", "login": "first-user" }
          ],
          "copilotTokens": {
            "https://enterprise.example:last-user": "plaintext-token"
          },
        }
      "#,
    );
    let stores = Arc::new(MockStores {
        result: Some("secure-token".into()),
        ..Default::default()
    });
    assert_eq!(
        read_github_copilot_subscription(home_options(&home, "darwin", &stores))
            .await
            .unwrap(),
        subscription("secure-token", "https://enterprise.example")
    );
    assert_eq!(
        *stores.macos.lock(),
        vec![(
            "copilot-cli".to_string(),
            "https://enterprise.example:last-user".to_string()
        )]
    );
}

fn octocat_config() -> String {
    serde_json::json!({
        "lastLoggedInUser": { "host": "https://github.com", "login": "octocat" },
    })
    .to_string()
}

#[tokio::test]
async fn reads_the_selected_account_from_linux_secret_service() {
    let home = create_copilot_home(&octocat_config());
    let stores = Arc::new(MockStores {
        result: Some("secure-token".into()),
        ..Default::default()
    });
    assert_eq!(
        read_github_copilot_subscription(home_options(&home, "linux", &stores))
            .await
            .unwrap(),
        subscription("secure-token", "https://github.com")
    );
    assert_eq!(
        *stores.linux.lock(),
        vec![env(&[
            ("service", "copilot-cli"),
            ("username", "https://github.com:octocat"),
        ])]
    );
}

#[tokio::test]
async fn reads_the_selected_account_from_windows_credential_manager() {
    let home = create_copilot_home(&octocat_config());
    let stores = Arc::new(MockStores {
        result: Some("secure-token".into()),
        ..Default::default()
    });
    assert_eq!(
        read_github_copilot_subscription(home_options(&home, "win32", &stores))
            .await
            .unwrap(),
        subscription("secure-token", "https://github.com")
    );
    assert_eq!(
        *stores.windows.lock(),
        vec!["https://github.com:octocat.copilot-cli".to_string()]
    );
}

#[tokio::test]
async fn does_not_read_a_secure_credential_on_unsupported_platforms() {
    let home = create_copilot_home(&octocat_config());
    let stores = Arc::new(MockStores::default());
    assert_eq!(
        read_github_copilot_subscription(home_options(&home, "freebsd", &stores))
            .await
            .unwrap(),
        None
    );
    assert!(stores.linux.lock().is_empty());
    assert!(stores.macos.lock().is_empty());
    assert!(stores.windows.lock().is_empty());
}

#[tokio::test]
async fn uses_the_first_logged_in_account_when_no_last_account_is_stored() {
    let home = create_copilot_home(
        &serde_json::json!({
            "logged_in_users": [
                { "host": "invalid", "login": "" },
                { "host": "https://github.com", "login": "first-valid-user" },
                { "host": "https://enterprise.example", "login": "second-user" },
            ],
        })
        .to_string(),
    );
    let (read_secure_credential, calls) = secure_credential(Some("secure-token"));
    read_github_copilot_subscription(ReadGitHubCopilotSubscriptionOptions {
        env: Some(env(&[("COPILOT_HOME", &home.path_string())])),
        read_secure_credential,
        find_github_cli_executable: no_gh(),
        ..Default::default()
    })
    .await
    .unwrap();
    assert!(calls.lock().contains(&(
        "copilot-cli".to_string(),
        "https://github.com:first-valid-user".to_string()
    )));
}

#[tokio::test]
async fn falls_back_to_the_plaintext_token_without_writing_native_state() {
    let home = create_copilot_home(
        &serde_json::json!({
            "lastLoggedInUser": { "host": "https://github.com", "login": "octocat" },
            "copilotTokens": { "https://github.com:octocat": "plaintext-token" },
        })
        .to_string(),
    );
    let (read_secure_credential, _) = secure_credential(None);
    assert_eq!(
        read_github_copilot_subscription(ReadGitHubCopilotSubscriptionOptions {
            env: Some(env(&[("COPILOT_HOME", &home.path_string())])),
            read_secure_credential,
            find_github_cli_executable: no_gh(),
            ..Default::default()
        })
        .await
        .unwrap(),
        subscription("plaintext-token", "https://github.com")
    );
}

#[tokio::test]
async fn does_not_select_another_stored_account_when_the_selected_account_has_no_token() {
    let home = create_copilot_home(
        &serde_json::json!({
            "lastLoggedInUser": {
                "host": "https://enterprise.example",
                "login": "selected-user",
            },
            "loggedInUsers": [{ "host": "https://github.com", "login": "other-user" }],
            "copilotTokens": { "https://github.com:other-user": "other-token" },
        })
        .to_string(),
    );
    let (read_secure_credential, calls) = secure_credential(None);
    assert_eq!(
        read_github_copilot_subscription(ReadGitHubCopilotSubscriptionOptions {
            env: Some(env(&[("COPILOT_HOME", &home.path_string())])),
            read_secure_credential,
            find_github_cli_executable: no_gh(),
            ..Default::default()
        })
        .await
        .unwrap(),
        None
    );
    assert_eq!(
        *calls.lock(),
        vec![(
            "copilot-cli".to_string(),
            "https://enterprise.example:selected-user".to_string()
        )]
    );
}

#[tokio::test]
async fn uses_the_selected_account_host_for_the_gh_fallback() {
    let home = create_copilot_home(
        &serde_json::json!({
            "lastLoggedInUser": { "host": "https://enterprise.example", "login": "octocat" },
        })
        .to_string(),
    );
    let (read_secure_credential, _) = secure_credential(None);
    let (read_github_cli_token, calls) = gh_token(Some(" gh-token\n"));
    let environment = env(&[("COPILOT_HOME", &home.path_string()), ("PATH", "/usr/bin")]);
    assert_eq!(
        read_github_copilot_subscription(ReadGitHubCopilotSubscriptionOptions {
            env: Some(environment.clone()),
            read_secure_credential,
            find_github_cli_executable: gh_at("/usr/bin/gh"),
            read_github_cli_token,
            ..Default::default()
        })
        .await
        .unwrap(),
        subscription("gh-token", "https://enterprise.example")
    );
    assert_eq!(
        *calls.lock(),
        vec![(
            "/usr/bin/gh".to_string(),
            "enterprise.example".to_string(),
            environment
        )]
    );
}

#[tokio::test]
async fn uses_the_configured_host_for_gh_when_no_native_account_is_stored() {
    let (read_secure_credential, _) = secure_credential(None);
    let (read_github_cli_token, calls) = gh_token(Some("gh-token"));
    assert_eq!(
        read_github_copilot_subscription(ReadGitHubCopilotSubscriptionOptions {
            env: Some(env(&[
                ("GH_HOST", "enterprise.example"),
                ("PATH", "/usr/local/bin")
            ])),
            home_directory: Some("/missing-home".into()),
            read_secure_credential,
            find_github_cli_executable: gh_at("/usr/local/bin/gh"),
            read_github_cli_token,
            ..Default::default()
        })
        .await
        .unwrap(),
        subscription("gh-token", "https://enterprise.example")
    );
    assert_eq!(calls.lock()[0].1, "enterprise.example");
}

#[tokio::test]
async fn does_not_execute_gh_when_it_is_unavailable_on_the_host() {
    let (read_secure_credential, _) = secure_credential(None);
    let (read_github_cli_token, calls) = gh_token(None);
    assert_eq!(
        read_github_copilot_subscription(ReadGitHubCopilotSubscriptionOptions {
            env: Some(env(&[("PATH", "/missing-bin")])),
            home_directory: Some("/missing-home".into()),
            read_secure_credential,
            find_github_cli_executable: no_gh(),
            read_github_cli_token,
            ..Default::default()
        })
        .await
        .unwrap(),
        None
    );
    assert!(calls.lock().is_empty());
}

#[tokio::test]
async fn returns_no_token_when_gh_returns_nothing_or_blank_output() {
    for stdout in [None, Some(""), Some("   ")] {
        let (read_secure_credential, _) = secure_credential(None);
        let (read_github_cli_token, _) = gh_token(stdout);
        assert_eq!(
            read_github_copilot_subscription(ReadGitHubCopilotSubscriptionOptions {
                env: Some(env(&[("PATH", "/usr/bin")])),
                home_directory: Some("/missing-home".into()),
                read_secure_credential,
                find_github_cli_executable: gh_at("/usr/bin/gh"),
                read_github_cli_token,
                ..Default::default()
            })
            .await
            .unwrap(),
            None,
            "{stdout:?}"
        );
    }
}

#[tokio::test]
async fn ignores_malformed_native_configuration() {
    let home = create_copilot_home("{ invalid");
    let (read_secure_credential, calls) = secure_credential(None);
    assert_eq!(
        read_github_copilot_subscription(ReadGitHubCopilotSubscriptionOptions {
            env: Some(env(&[("COPILOT_HOME", &home.path_string())])),
            read_secure_credential,
            find_github_cli_executable: no_gh(),
            ..Default::default()
        })
        .await
        .unwrap(),
        None
    );
    assert!(calls.lock().is_empty());
}

// findHostGitHubCliExecutable
#[cfg(unix)]
#[tokio::test]
async fn finds_an_executable_in_a_unix_path() {
    use std::os::unix::fs::PermissionsExt as _;
    let directory = mkdtemp("copilot-gh-bin-");
    let executable = directory.path().join("gh");
    std::fs::write(&executable, "").unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert_eq!(
        find_host_github_cli_executable(
            &env(&[("PATH", &format!("/missing:{}", directory.path_string()))]),
            "linux",
        )
        .await,
        Some(executable.to_string_lossy().into_owned())
    );
}

#[tokio::test]
async fn honors_pathext_when_finding_a_windows_executable() {
    let directory = mkdtemp("copilot-gh-bin-");
    let executable = directory.path().join("gh.EXE");
    std::fs::write(&executable, "").unwrap();
    assert_eq!(
        find_host_github_cli_executable(
            &env(&[
                ("PATH", &format!("C:\\missing;{}", directory.path_string())),
                ("PATHEXT", ".EXE;.CMD"),
            ]),
            "win32",
        )
        .await,
        Some(executable.to_string_lossy().into_owned())
    );
}

#[tokio::test]
async fn returns_none_without_a_host_path() {
    assert_eq!(
        find_host_github_cli_executable(&IndexMap::new(), "linux").await,
        None
    );
}
