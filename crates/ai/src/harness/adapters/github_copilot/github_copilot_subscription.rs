//! Port of `harness-github-copilot/src/github-copilot-subscription.ts`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Map, Value as JsonValue};

use crate::harness::adapters::acp::ACPAuthenticationMode;
use crate::harness::adapters::jsonc::parse_jsonc;
use crate::harness::errors::HarnessResult;
use crate::harness::spec::HarnessV1Authentication;
use crate::harness::utils::{
    current_platform, is_linux, is_macos, is_windows, plain_error,
    read_linux_secret_service_password, read_macos_keychain_password,
    read_windows_credential_manager_password, should_resolve_native_subscription,
};

const COPILOT_KEYRING_SERVICE: &str = "copilot-cli";

#[derive(Debug, Clone, PartialEq, Eq)]
struct GitHubCopilotAccount {
    host: String,
    login: String,
}

/// `GitHubCopilotSubscription`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitHubCopilotSubscription {
    pub token: String,
    pub host: String,
}

/// `ReadSecureCredential`: `({ service, account }) => Promise<string | undefined>`.
pub type ReadSecureCredential =
    Arc<dyn Fn(String, String) -> BoxFuture<'static, HarnessResult<Option<String>>> + Send + Sync>;

/// `FindGitHubCliExecutable`: `({ env, platform }) => Promise<string | undefined>`.
pub type FindGitHubCliExecutable = Arc<
    dyn Fn(IndexMap<String, String>, String) -> BoxFuture<'static, Option<String>> + Send + Sync,
>;

/// `ReadGitHubCliToken`: `({ executable, hostname, env }) => Promise<string | undefined>`.
pub type ReadGitHubCliToken = Arc<
    dyn Fn(
            String,
            String,
            IndexMap<String, String>,
        ) -> BoxFuture<'static, HarnessResult<Option<String>>>
        + Send
        + Sync,
>;

/// `readSubscription` of [`resolve_github_copilot_subscription_environment`].
pub type ReadGitHubCopilotSubscription = Arc<
    dyn Fn(
            IndexMap<String, String>,
        ) -> BoxFuture<'static, HarnessResult<Option<GitHubCopilotSubscription>>>
        + Send
        + Sync,
>;

/// `resolveGitHubCopilotSubscriptionEnvironment({ auth, env, readSubscription })`.
/// `read_subscription` defaults to [`read_github_copilot_subscription`].
pub async fn resolve_github_copilot_subscription_environment(
    auth: Option<&ACPAuthenticationMode>,
    env: IndexMap<String, String>,
    read_subscription: Option<ReadGitHubCopilotSubscription>,
) -> HarnessResult<IndexMap<String, String>> {
    if let Some(HarnessV1Authentication::Environment(environment)) = auth {
        return Ok(environment.clone());
    }
    let has_direct_credential = ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]
        .iter()
        .any(|name| env.contains_key(*name));
    if !should_resolve_native_subscription(auth, &env, has_direct_credential) {
        return Ok(env);
    }

    let subscription = match read_subscription {
        Some(read) => read(env.clone()).await?,
        None => {
            read_github_copilot_subscription(ReadGitHubCopilotSubscriptionOptions {
                env: Some(env.clone()),
                ..Default::default()
            })
            .await?
        }
    };
    let Some(subscription) = subscription else {
        return Ok(env);
    };

    let mut env = env;
    env.insert("COPILOT_GITHUB_TOKEN".into(), subscription.token);
    env.insert(
        "COPILOT_GH_HOST".into(),
        normalize_github_host(&subscription.host)?,
    );
    Ok(env)
}

/// [rust] The platform credential stores `readGitHubCopilotSecureCredential`
/// dispatches to, a seam for the tests, which mock these readers.
#[async_trait]
pub(crate) trait SecureCredentialStores: Send + Sync {
    async fn read_macos_keychain_password(&self, service: &str, account: &str) -> Option<String>;
    async fn read_linux_secret_service_password(
        &self,
        attributes: &IndexMap<String, String>,
    ) -> Option<String>;
    async fn read_windows_credential_manager_password(&self, target_name: &str) -> Option<String>;
}

struct SystemSecureCredentialStores;

#[async_trait]
impl SecureCredentialStores for SystemSecureCredentialStores {
    async fn read_macos_keychain_password(&self, service: &str, account: &str) -> Option<String> {
        read_macos_keychain_password(service, account).await
    }

    async fn read_linux_secret_service_password(
        &self,
        attributes: &IndexMap<String, String>,
    ) -> Option<String> {
        read_linux_secret_service_password(attributes).await
    }

    async fn read_windows_credential_manager_password(&self, target_name: &str) -> Option<String> {
        read_windows_credential_manager_password(target_name).await
    }
}

/// Options of [`read_github_copilot_subscription`]. Unset fields default to
/// the host's `process.env`, `homedir()`, `process.platform`, the platform
/// credential store, [`find_host_github_cli_executable`] and running
/// `gh auth token`.
#[derive(Clone, Default)]
pub struct ReadGitHubCopilotSubscriptionOptions {
    pub env: Option<IndexMap<String, String>>,
    pub home_directory: Option<String>,
    pub platform: Option<String>,
    pub read_secure_credential: Option<ReadSecureCredential>,
    pub find_github_cli_executable: Option<FindGitHubCliExecutable>,
    pub read_github_cli_token: Option<ReadGitHubCliToken>,
    /// [rust] Test seam standing in for the mocked credential store readers.
    pub(crate) secure_credential_stores: Option<Arc<dyn SecureCredentialStores>>,
}

/// [rust] Node's `path.resolve`.
fn resolve_path(path: PathBuf) -> PathBuf {
    std::path::absolute(&path).unwrap_or(path)
}

/// `readGitHubCopilotSubscription(options)`.
pub async fn read_github_copilot_subscription(
    options: ReadGitHubCopilotSubscriptionOptions,
) -> HarnessResult<Option<GitHubCopilotSubscription>> {
    let env = options.env.unwrap_or_else(|| std::env::vars().collect());
    let home_directory = options.home_directory.unwrap_or_else(|| {
        std::env::home_dir()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    let platform = options
        .platform
        .unwrap_or_else(|| current_platform().to_string());

    let copilot_home = resolve_path(
        env.get("COPILOT_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(&home_directory).join(".copilot")),
    );
    let config = read_github_copilot_config(&copilot_home.join("config.json")).await;
    let account = select_github_copilot_account(config.as_ref());

    if let Some(account) = &account {
        let account_key = format!("{}:{}", account.host, account.login);
        let secure_token = match &options.read_secure_credential {
            Some(read) => read(COPILOT_KEYRING_SERVICE.into(), account_key.clone())
                .await
                .ok()
                .flatten(),
            None => {
                let stores = options
                    .secure_credential_stores
                    .clone()
                    .unwrap_or_else(|| Arc::new(SystemSecureCredentialStores));
                read_github_copilot_secure_credential(
                    &*stores,
                    COPILOT_KEYRING_SERVICE,
                    &account_key,
                    &platform,
                )
                .await
            }
        };
        if let Some(token) = secure_token.filter(|token| is_non_empty_string(token)) {
            return Ok(Some(GitHubCopilotSubscription {
                token,
                host: account.host.clone(),
            }));
        }

        if let Some(token) = get_plaintext_token(config.as_ref(), &account_key) {
            return Ok(Some(GitHubCopilotSubscription {
                token,
                host: account.host.clone(),
            }));
        }
    }

    let host = match account {
        Some(account) => Some(account.host),
        None => normalize_github_origin(
            env.get("COPILOT_GH_HOST")
                .or_else(|| env.get("GH_HOST"))
                .map(String::as_str)
                .unwrap_or("github.com"),
        ),
    };
    let Some(host) = host else {
        return Ok(None);
    };

    let executable = match &options.find_github_cli_executable {
        Some(find) => find(env.clone(), platform.clone()).await,
        None => find_host_github_cli_executable(&env, &platform).await,
    };
    let Some(executable) = executable else {
        return Ok(None);
    };

    let hostname = normalize_github_host(&host)?;
    let token = match &options.read_github_cli_token {
        Some(read) => read(executable, hostname, env.clone()).await.ok().flatten(),
        None => read_host_github_cli_token(&executable, &hostname, &env).await,
    };
    Ok(token
        .filter(|token| is_non_empty_string(token))
        .map(|token| GitHubCopilotSubscription {
            token: token.trim().to_string(),
            host,
        }))
}

/// `findHostGitHubCliExecutable({ env, platform })`.
pub async fn find_host_github_cli_executable(
    env: &IndexMap<String, String>,
    platform: &str,
) -> Option<String> {
    let path = env
        .get("PATH")
        .or_else(|| env.get("Path"))
        .or_else(|| env.get("path"))?;

    let windows = is_windows(platform);
    let separator = if windows { ';' } else { ':' };
    let names = if windows {
        get_windows_github_cli_executable_names(env.get("PATHEXT").map(String::as_str))
    } else {
        vec!["gh".to_string()]
    };

    for directory in path.split(separator) {
        let directory = directory.strip_prefix('"').unwrap_or(directory);
        let directory = directory.strip_suffix('"').unwrap_or(directory);
        if directory.is_empty() {
            continue;
        }
        for name in &names {
            let candidate = resolve_path(Path::new(directory).join(name));
            if is_accessible(&candidate, !windows).await {
                return Some(candidate.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// `access(candidate, executable ? X_OK : F_OK)`.
async fn is_accessible(path: &Path, executable: bool) -> bool {
    let Ok(metadata) = tokio::fs::metadata(path).await else {
        return false;
    };
    if !executable {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        true
    }
}

async fn read_github_copilot_config(path: &Path) -> Option<Map<String, JsonValue>> {
    let text = tokio::fs::read_to_string(path).await.ok()?;
    match parse_jsonc(&text)? {
        JsonValue::Object(config) => Some(config),
        _ => None,
    }
}

fn select_github_copilot_account(
    config: Option<&Map<String, JsonValue>>,
) -> Option<GitHubCopilotAccount> {
    let config = config?;
    for key in ["last_logged_in_user", "lastLoggedInUser"] {
        if let Some(account) = config.get(key).and_then(to_github_copilot_account) {
            return Some(account);
        }
    }
    for key in ["logged_in_users", "loggedInUsers"] {
        let Some(values) = config.get(key).and_then(JsonValue::as_array) else {
            continue;
        };
        if let Some(account) = values.iter().find_map(to_github_copilot_account) {
            return Some(account);
        }
    }
    None
}

fn to_github_copilot_account(value: &JsonValue) -> Option<GitHubCopilotAccount> {
    let value = value.as_object()?;
    let host = value
        .get("host")?
        .as_str()
        .filter(|s| is_non_empty_string(s))?;
    let login = value
        .get("login")?
        .as_str()
        .filter(|s| is_non_empty_string(s))?;
    Some(GitHubCopilotAccount {
        host: normalize_github_origin(host)?,
        login: login.to_string(),
    })
}

fn get_plaintext_token(
    config: Option<&Map<String, JsonValue>>,
    account_key: &str,
) -> Option<String> {
    config?
        .get("copilotTokens")?
        .as_object()?
        .get(account_key)?
        .as_str()
        .filter(|token| is_non_empty_string(token))
        .map(str::to_string)
}

async fn read_github_copilot_secure_credential(
    stores: &dyn SecureCredentialStores,
    service: &str,
    account: &str,
    platform: &str,
) -> Option<String> {
    if is_macos(platform) {
        return stores.read_macos_keychain_password(service, account).await;
    }
    if is_linux(platform) {
        let attributes = IndexMap::from([
            ("service".to_string(), service.to_string()),
            ("username".to_string(), account.to_string()),
        ]);
        return stores.read_linux_secret_service_password(&attributes).await;
    }
    if is_windows(platform) {
        return stores
            .read_windows_credential_manager_password(&format!("{account}.{service}"))
            .await;
    }
    None
}

/// `readHostGitHubCliToken`: `gh auth token --hostname <hostname>` with
/// `env` as the whole environment, a 10 second timeout and a hidden window.
async fn read_host_github_cli_token(
    executable: &str,
    hostname: &str,
    env: &IndexMap<String, String>,
) -> Option<String> {
    let mut command = tokio::process::Command::new(executable);
    command
        .args(["auth", "token", "--hostname", hostname])
        .env_clear()
        .envs(env)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let output = tokio::time::timeout(Duration::from_secs(10), command.output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    is_non_empty_string(&stdout).then(|| stdout.trim().to_string())
}

fn get_windows_github_cli_executable_names(path_ext: Option<&str>) -> Vec<String> {
    std::iter::once("gh".to_string())
        .chain(
            path_ext
                .unwrap_or(".COM;.EXE;.BAT;.CMD")
                .split(';')
                .filter(|extension| is_non_empty_string(extension))
                .map(|extension| format!("gh{extension}")),
        )
        .collect()
}

fn parse_url(host: &str) -> Option<reqwest::Url> {
    let url = if host.contains("://") {
        host.to_string()
    } else {
        format!("https://{host}")
    };
    reqwest::Url::parse(&url).ok()
}

fn normalize_github_origin(host: &str) -> Option<String> {
    parse_url(host).map(|url| url.origin().ascii_serialization())
}

/// `normalizeGitHubHost(host)`: the hostname of `host`, read as an
/// `https://` URL when it has no scheme. Fails on an invalid URL, as
/// `new URL` throws.
pub fn normalize_github_host(host: &str) -> HarnessResult<String> {
    parse_url(host)
        .map(|url| url.host_str().unwrap_or_default().to_string())
        .ok_or_else(|| plain_error("TypeError: Invalid URL"))
}

fn is_non_empty_string(value: &str) -> bool {
    !value.trim().is_empty()
}
