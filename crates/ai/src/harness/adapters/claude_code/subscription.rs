//! Port of `harness-claude-code/src/claude-code-subscription.ts`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Map, Value as JsonValue};

use crate::harness::auth::HarnessAuthentication;
use crate::harness::error::HarnessResult;
use crate::harness::network::{HarnessRequestTransformation, HarnessRequestTransformationSources};
use crate::harness::utils::native_subscription::macos_keychain;
use crate::harness::utils::native_subscription::{ExecFile, SystemExecFile};
use crate::harness::utils::{
    OAuthFetch, OAuthRequestFormat, RefreshOAuthAccessTokenOptions,
    create_credential_request_transformation, current_platform, is_access_token_expiring_soon,
    is_macos, plain_error, refresh_oauth_access_token, should_resolve_native_subscription,
};

use super::auth::{
    ClaudeCodeAuthenticationMode, ClaudeCodeResolvedAuthenticationMode,
    ResolveClaudeCodeEnvOptions, claude_code_match_url, host_home_directory,
    resolve_claude_code_env,
};

const CLAUDE_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const KEYCHAIN_SERVICE: &str = "Claude Code-credentials";

/// `createClaudeCodeSubscriptionRequestTransformations({ env, sandboxEnv,
/// auth })`.
pub fn create_claude_code_subscription_request_transformations(
    sources: &HarnessRequestTransformationSources<ClaudeCodeResolvedAuthenticationMode>,
) -> HarnessResult<Vec<HarnessRequestTransformation>> {
    let (Some(token), Some(sandbox_token)) = (
        sources.env.get("CLAUDE_CODE_OAUTH_TOKEN"),
        sources.sandbox_env.get("CLAUDE_CODE_OAUTH_TOKEN"),
    ) else {
        return Ok(Vec::new());
    };
    Ok(vec![create_credential_request_transformation(
        &claude_code_match_url(&sources.env, sources.auth),
        &IndexMap::from([(
            "Authorization".to_string(),
            format!("Bearer {sandbox_token}"),
        )]),
        &IndexMap::from([("Authorization".to_string(), format!("Bearer {token}"))]),
    )?])
}

/// `readSubscription` of [`ResolveClaudeCodeAuthenticationOptions`].
pub type ReadClaudeCodeSubscriptionFn = Arc<
    dyn Fn() -> BoxFuture<'static, HarnessResult<Option<IndexMap<String, String>>>> + Send + Sync,
>;

/// Options of [`resolve_claude_code_authentication`].
#[derive(Clone, Default)]
pub struct ResolveClaudeCodeAuthenticationOptions {
    pub auth: Option<ClaudeCodeAuthenticationMode>,
    /// Defaults to the host's `process.env`.
    pub process_env: Option<IndexMap<String, String>>,
    pub options: ResolveClaudeCodeEnvOptions,
    /// Defaults to [`read_claude_code_subscription`].
    pub read_subscription: Option<ReadClaudeCodeSubscriptionFn>,
}

/// `resolveClaudeCodeAuthentication({ auth, processEnv, options,
/// readSubscription })`.
pub async fn resolve_claude_code_authentication(
    options: ResolveClaudeCodeAuthenticationOptions,
) -> HarnessResult<IndexMap<String, String>> {
    let auth = options.auth.as_ref();
    let process_env = options
        .process_env
        .clone()
        .unwrap_or_else(|| std::env::vars().collect());
    let no_helper = ResolveClaudeCodeEnvOptions {
        read_api_key_helper: Some(Arc::new(|| None)),
    };
    let environment_without_helper = resolve_claude_code_env(auth, &process_env, &no_helper);
    if matches!(auth, Some(HarnessAuthentication::Environment(_))) {
        return Ok(environment_without_helper);
    }
    let has_direct_credential = [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ]
    .iter()
    .any(|name| environment_without_helper.contains_key(*name));
    if !should_resolve_native_subscription(auth, &environment_without_helper, has_direct_credential)
    {
        return Ok(environment_without_helper);
    }

    let subscription = match &options.read_subscription {
        Some(read) => read().await?,
        None => read_claude_code_subscription(ReadClaudeCodeSubscriptionOptions::default()).await?,
    };
    Ok(match subscription {
        Some(subscription) => subscription,
        None => resolve_claude_code_env(auth, &process_env, &options.options),
    })
}

/// Options of [`read_claude_code_subscription`]. Unset fields default to
/// the host's `process.env`, `homedir()` and `process.platform`.
#[derive(Clone, Default)]
pub struct ReadClaudeCodeSubscriptionOptions {
    pub env: Option<IndexMap<String, String>>,
    pub home_directory: Option<String>,
    pub platform: Option<String>,
    pub fetch: Option<OAuthFetch>,
    /// [rust] The `execFile` seam the TS tests reach by mocking
    /// `node:child_process`, for the macOS keychain.
    pub(crate) exec_file: Option<Arc<dyn ExecFile>>,
}

/// `path.resolve(path)`: absolute and normalized.
fn resolve_path(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                resolved.pop();
            }
            component => resolved.push(component),
        }
    }
    resolved
}

/// `readClaudeCodeSubscription({ env, homeDirectory, platform, fetch })`.
pub async fn read_claude_code_subscription(
    options: ReadClaudeCodeSubscriptionOptions,
) -> HarnessResult<Option<IndexMap<String, String>>> {
    let env = options.env.unwrap_or_else(|| std::env::vars().collect());
    let home_directory = options
        .home_directory
        .map(PathBuf::from)
        .or_else(host_home_directory)
        .unwrap_or_default();
    let platform = options
        .platform
        .unwrap_or_else(|| current_platform().to_string());
    let exec_file: Arc<dyn ExecFile> = options
        .exec_file
        .unwrap_or_else(|| Arc::new(SystemExecFile));

    let config_directory = resolve_path(
        &env.get("CLAUDE_CONFIG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| home_directory.join(".claude")),
    );
    let credential_path = config_directory.join(".credentials.json");
    let Some(stored) = read_claude_credential_store(
        &credential_path,
        &home_directory,
        &config_directory,
        &platform,
        exec_file.as_ref(),
    )
    .await
    else {
        return Ok(None);
    };
    let Some(oauth) = get_claude_oauth_credential(&stored.value) else {
        return Ok(None);
    };

    let mut resolved_access_token = oauth.access_token.clone();
    if is_access_token_expiring_soon(oauth.expires_at, None, None) {
        let refreshed = refresh_oauth_access_token(RefreshOAuthAccessTokenOptions {
            token_url: CLAUDE_TOKEN_URL.into(),
            client_id: CLAUDE_CLIENT_ID.into(),
            refresh_token: oauth.refresh_token.clone(),
            request_format: OAuthRequestFormat::Json,
            headers: None,
            fetch: options.fetch,
        })
        .await?;
        resolved_access_token = refreshed.access_token.clone();
        let mut updated_oauth = oauth.record.clone();
        updated_oauth.insert("accessToken".into(), refreshed.access_token.into());
        updated_oauth.insert(
            "refreshToken".into(),
            refreshed
                .refresh_token
                .unwrap_or(oauth.refresh_token)
                .into(),
        );
        updated_oauth.insert("expiresAt".into(), refreshed.expires_at.into());
        let mut updated = stored.value.clone();
        updated.insert("claudeAiOauth".into(), JsonValue::Object(updated_oauth));
        match stored.location {
            CredentialStoreLocation::File(path) => {
                write_claude_credential_file(&path, &updated).await?;
            }
            CredentialStoreLocation::Keychain => {
                let args = [
                    "add-generic-password",
                    "-U",
                    "-s",
                    KEYCHAIN_SERVICE,
                    "-a",
                    &keychain_account(),
                    "-w",
                    &serde_json::to_string(&updated)?,
                ]
                .map(String::from);
                exec_file
                    .exec_file("/usr/bin/security", &args, None)
                    .await
                    .map_err(|error| plain_error(error.to_string()))?;
            }
        }
    }

    Ok(Some(IndexMap::from([
        ("CLAUDE_CODE_OAUTH_TOKEN".to_string(), resolved_access_token),
        (
            "ANTHROPIC_BASE_URL".to_string(),
            "https://api.anthropic.com".to_string(),
        ),
    ])))
}

/// `process.env.USER ?? ''`.
fn keychain_account() -> String {
    std::env::var("USER").unwrap_or_default()
}

enum CredentialStoreLocation {
    File(PathBuf),
    Keychain,
}

struct ClaudeCredentialStore {
    value: Map<String, JsonValue>,
    location: CredentialStoreLocation,
}

struct ClaudeOAuthCredential {
    record: Map<String, JsonValue>,
    access_token: String,
    refresh_token: String,
    expires_at: f64,
}

async fn read_claude_credential_store(
    credential_path: &Path,
    home_directory: &Path,
    config_directory: &Path,
    platform: &str,
    exec_file: &dyn ExecFile,
) -> Option<ClaudeCredentialStore> {
    if let Ok(text) = tokio::fs::read_to_string(credential_path).await
        && let Some(value) = parse_claude_credential_file(&text)
        && get_claude_oauth_credential(&value).is_some()
    {
        return Some(ClaudeCredentialStore {
            value,
            location: CredentialStoreLocation::File(credential_path.to_path_buf()),
        });
    }

    if !is_macos(platform) {
        return None;
    }
    if config_directory != resolve_path(&home_directory.join(".claude")) {
        return None;
    }
    let keychain_text =
        macos_keychain::read_with(exec_file, KEYCHAIN_SERVICE, &keychain_account()).await?;
    let value = parse_claude_credential_file(&keychain_text)?;
    Some(ClaudeCredentialStore {
        value,
        location: CredentialStoreLocation::Keychain,
    })
}

fn get_claude_oauth_credential(value: &Map<String, JsonValue>) -> Option<ClaudeOAuthCredential> {
    let oauth = value.get("claudeAiOauth")?.as_object()?;
    Some(ClaudeOAuthCredential {
        access_token: oauth.get("accessToken")?.as_str()?.to_string(),
        refresh_token: oauth.get("refreshToken")?.as_str()?.to_string(),
        expires_at: oauth.get("expiresAt")?.as_f64()?,
        record: oauth.clone(),
    })
}

fn parse_claude_credential_file(text: &str) -> Option<Map<String, JsonValue>> {
    match serde_json::from_str(text).ok()? {
        JsonValue::Object(value) => Some(value),
        _ => None,
    }
}

async fn write_claude_credential_file(
    credential_path: &Path,
    value: &Map<String, JsonValue>,
) -> HarnessResult<()> {
    if let Some(parent) = credential_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut temporary_path = credential_path.as_os_str().to_owned();
    temporary_path.push(format!(".{}.tmp", std::process::id()));
    let temporary_path = PathBuf::from(temporary_path);
    let content = format!("{}\n", serde_json::to_string_pretty(value)?);
    let mut open = tokio::fs::OpenOptions::new();
    open.write(true).create(true).truncate(true);
    #[cfg(unix)]
    open.mode(0o600);
    {
        use tokio::io::AsyncWriteExt as _;
        let mut file = open.open(&temporary_path).await?;
        file.write_all(content.as_bytes()).await?;
        file.flush().await?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        tokio::fs::set_permissions(&temporary_path, std::fs::Permissions::from_mode(0o600)).await?;
    }
    tokio::fs::rename(&temporary_path, credential_path).await?;
    Ok(())
}
