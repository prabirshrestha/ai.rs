//! Port of `harness-cursor/src/cursor-subscription.ts`.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde_json::Value as JsonValue;

use crate::harness::adapters::acp::ACPAuthenticationMode;
use crate::harness::errors::HarnessResult;
use crate::harness::provider::safe_parse_json;
use crate::harness::spec::HarnessV1Authentication;
use crate::harness::utils::{
    current_platform, get_jwt_expires_at, is_access_token_expiring_soon, is_linux, is_macos,
    is_windows, plain_error, read_macos_keychain_password, should_resolve_native_subscription,
};

/// `resolveCursorSubscriptionEnvironment({ auth, env })`.
pub async fn resolve_cursor_subscription_environment(
    auth: Option<&ACPAuthenticationMode>,
    env: IndexMap<String, String>,
) -> HarnessResult<IndexMap<String, String>> {
    if let Some(HarnessV1Authentication::Environment(environment)) = auth {
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
/// host's `process.env`, `homedir()` and `process.platform`.
#[derive(Debug, Clone, Default)]
pub struct ReadCursorSubscriptionOptions {
    pub env: Option<IndexMap<String, String>>,
    pub home_directory: Option<String>,
    pub platform: Option<String>,
}

/// [rust] `os.homedir()`.
fn host_home_directory() -> String {
    std::env::home_dir()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `readCursorSubscription({ env, homeDirectory, platform })`.
pub async fn read_cursor_subscription(
    options: ReadCursorSubscriptionOptions,
) -> HarnessResult<Option<String>> {
    let env = options.env.unwrap_or_else(|| std::env::vars().collect());
    let home_directory = options.home_directory.unwrap_or_else(host_home_directory);
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
        return Err(plain_error(
            "Cursor subscription access token is expiring soon. Run Cursor login again.",
        ));
    }
    Ok(Some(access_token))
}

/// `resolveCursorAuthPath({ env, homeDirectory, platform })`.
///
/// [rust] Like Node's `path.join`, the separator is the host's.
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
    let value = safe_parse_json(text, None).ok()?;
    value
        .as_object()?
        .get("accessToken")
        .and_then(JsonValue::as_str)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use serde_json::json;

    use super::*;
    use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};
    use crate::harness::spec::HarnessV1DirectAuthentication;

    fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    // resolveCursorSubscriptionEnvironment
    #[tokio::test]
    async fn does_not_inspect_subscriptions_for_gateway_auth() {
        assert_eq!(
            resolve_cursor_subscription_environment(
                Some(&HarnessV1Authentication::AiGateway),
                env(&[("AI_GATEWAY_API_KEY", "gateway")]),
            )
            .await
            .unwrap(),
            env(&[("AI_GATEWAY_API_KEY", "gateway")])
        );
    }

    #[tokio::test]
    async fn prefers_cursor_api_key_over_native_storage() {
        assert_eq!(
            resolve_cursor_subscription_environment(
                Some(&HarnessV1Authentication::Choice(
                    HarnessV1DirectAuthentication::Direct
                )),
                env(&[("CURSOR_API_KEY", "environment-key")]),
            )
            .await
            .unwrap(),
            env(&[("CURSOR_API_KEY", "environment-key")])
        );
    }

    fn cursor_home(auth: &JsonValue) -> TempDir {
        let home = mkdtemp("cursor-auth-");
        let cursor_directory = home.path().join(".cursor");
        std::fs::create_dir_all(&cursor_directory).unwrap();
        std::fs::write(cursor_directory.join("auth.json"), auth.to_string()).unwrap();
        home
    }

    fn darwin(home: &TempDir) -> ReadCursorSubscriptionOptions {
        ReadCursorSubscriptionOptions {
            env: Some(IndexMap::new()),
            home_directory: Some(home.path_string()),
            platform: Some("darwin".into()),
        }
    }

    // readCursorSubscription
    #[tokio::test]
    async fn reads_a_file_backed_browser_access_token() {
        let home = cursor_home(&json!({
            "accessToken": "opaque-access-token",
            "refreshToken": "stored-but-not-forwarded",
        }));
        assert_eq!(
            read_cursor_subscription(darwin(&home)).await.unwrap(),
            Some("opaque-access-token".into())
        );
    }

    #[tokio::test]
    async fn rejects_an_expiring_jwt_instead_of_inventing_a_refresh_endpoint() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(json!({ "exp": now + 60 }).to_string());
        let home = cursor_home(&json!({
            "accessToken": format!("header.{payload}.signature"),
            "refreshToken": "refresh-token",
        }));
        let error = read_cursor_subscription(darwin(&home)).await.unwrap_err();
        assert!(error.to_string().contains("Run Cursor login again."));
    }

    // resolveCursorAuthPath
    #[test]
    fn uses_each_native_platform_path() {
        assert_eq!(
            resolve_cursor_auth_path(&IndexMap::new(), "/home/me", "linux"),
            "/home/me/.config/cursor/auth.json"
        );
        assert!(
            resolve_cursor_auth_path(
                &env(&[("APPDATA", "C:\\Users\\me\\AppData\\Roaming")]),
                "C:\\Users\\me",
                "win32",
            )
            .contains("Cursor/auth.json")
        );
    }
}
