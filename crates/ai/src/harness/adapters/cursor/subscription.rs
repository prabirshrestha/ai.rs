//! Port of `harness-cursor/src/cursor-subscription.ts`: the Cursor CLI
//! login stored on the host, used when no `CURSOR_API_KEY` is set.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde_json::Value;

use crate::harness::adapters::acp::AcpAuthenticationMode;
use crate::harness::auth::HarnessAuthentication;
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::utils::native_subscription::jwt::get_jwt_expires_at;
use crate::harness::utils::native_subscription::macos_keychain::read_macos_keychain_password;
use crate::harness::utils::native_subscription::should_resolve_native::should_resolve_native_subscription;
use crate::harness::utils::oauth_access_token::is_access_token_expiring_soon;
use crate::harness::utils::os::{current_platform, is_linux, is_macos, is_windows};

/// `resolveCursorSubscriptionEnvironment({ auth, env })`.
pub async fn resolve_cursor_subscription_environment(
    auth: Option<&AcpAuthenticationMode>,
    env: IndexMap<String, String>,
) -> HarnessResult<IndexMap<String, String>> {
    if let Some(HarnessAuthentication::Environment(environment)) = auth {
        return Ok(environment.clone());
    }
    if !should_resolve_native_subscription(auth, &env, env.contains_key("CURSOR_API_KEY")) {
        return Ok(env);
    }
    let subscription = read_cursor_subscription(ReadCursorSubscriptionOptions {
        env: Some(env.clone()),
        ..Default::default()
    })
    .await?;
    Ok(match subscription {
        None => env,
        Some(token) => {
            let mut env = env;
            env.insert("CURSOR_API_KEY".into(), token);
            env
        }
    })
}

/// Options of [`read_cursor_subscription`]. Unset fields default to the
/// host's environment, home directory and platform.
#[derive(Debug, Clone, Default)]
pub struct ReadCursorSubscriptionOptions {
    pub env: Option<IndexMap<String, String>>,
    pub home_directory: Option<String>,
    pub platform: Option<String>,
}

/// `readCursorSubscription({ env, homeDirectory, platform })`.
pub async fn read_cursor_subscription(
    options: ReadCursorSubscriptionOptions,
) -> HarnessResult<Option<String>> {
    let env = options.env.unwrap_or_else(|| std::env::vars().collect());
    let home_directory = options.home_directory.unwrap_or_else(|| {
        std::env::home_dir()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    let platform = options
        .platform
        .unwrap_or_else(|| current_platform().to_string());
    let store = env.get("AGENT_CLI_CREDENTIAL_STORE").map(String::as_str);
    if store == Some("memory") {
        return Ok(None);
    }
    let auth_path = resolve_cursor_auth_path(&env, &home_directory, &platform);
    let text = tokio::fs::read_to_string(&auth_path).await.ok();
    let file_credential = text.as_deref().and_then(parse_access_token);
    let access_token = match file_credential {
        Some(token) => Some(token),
        None if is_macos(&platform) && store != Some("file") => {
            read_macos_keychain_password("cursor-access-token", "cursor-user").await
        }
        None => None,
    };
    let Some(access_token) = access_token else {
        return Ok(None);
    };
    if let Some(expires_at) = get_jwt_expires_at(&access_token)
        && is_access_token_expiring_soon(expires_at, None, None)
    {
        return Err(HarnessError::new(
            "Cursor subscription access token is expiring soon. Run Cursor login again.",
        )
        .into());
    }
    Ok(Some(access_token))
}

/// `resolveCursorAuthPath({ env, homeDirectory, platform })`.
pub fn resolve_cursor_auth_path(
    env: &IndexMap<String, String>,
    home_directory: &str,
    platform: &str,
) -> String {
    let join = |base: PathBuf, parts: &[&str]| -> String {
        parts
            .iter()
            .fold(base, |path, part| path.join(part))
            .to_string_lossy()
            .into_owned()
    };
    let home = Path::new(home_directory);
    if is_windows(platform) {
        let base = env
            .get("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join("AppData").join("Roaming"));
        return join(base, &["Cursor", "auth.json"]);
    }
    if is_linux(platform) {
        let base = env
            .get("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".config"));
        return join(base, &["cursor", "auth.json"]);
    }
    join(home.to_path_buf(), &[".cursor", "auth.json"])
}

fn parse_access_token(text: &str) -> Option<String> {
    let value: Value = serde_json::from_str(text).ok()?;
    value
        .as_object()?
        .get("accessToken")
        .and_then(Value::as_str)
        .map(str::to_string)
}
