//! Port of `harness-codex/src/codex-subscription.ts`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Map, Value as JsonValue};

use crate::harness::auth::HarnessAuthentication;
use crate::harness::error::HarnessResult;
use crate::harness::network::{HarnessRequestTransformation, HarnessRequestTransformationSources};
use crate::harness::utils::native_subscription::{
    ExecFile, SystemExecFile, linux_secret_service, macos_keychain, windows_credential_manager,
};
use crate::harness::utils::{
    OAuthFetch, OAuthRequestFormat, RefreshOAuthAccessTokenOptions, current_platform,
    get_jwt_expires_at, is_access_token_expiring_soon, is_linux, is_macos, is_windows, plain_error,
    refresh_oauth_access_token, should_resolve_native_subscription,
};

use super::auth::{
    CodexAuthenticationMode, CodexResolvedAuthenticationMode, create_codex_request_transformations,
    resolve_codex_env,
};

const CHATGPT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const OPENAI_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const OPENAI_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const KEYRING_SERVICE: &str = "Codex Auth";

/// `CodexResolvedAuthentication`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodexResolvedAuthentication {
    pub environment: IndexMap<String, String>,
    pub request_headers: Option<IndexMap<String, String>>,
}

/// `createCodexSubscriptionRequestTransformations({ requestHeaders,
/// ...sources })`.
pub fn create_codex_subscription_request_transformations(
    sources: &HarnessRequestTransformationSources<CodexResolvedAuthenticationMode>,
    request_headers: Option<&IndexMap<String, String>>,
) -> HarnessResult<Vec<HarnessRequestTransformation>> {
    let mut transformations = create_codex_request_transformations(sources)?;
    if let Some(request_headers) = request_headers {
        for transformation in &mut transformations {
            for (name, value) in request_headers {
                transformation
                    .transform
                    .headers
                    .insert(name.clone(), value.clone());
            }
        }
    }
    Ok(transformations)
}

/// `CodexAuthCredentialsStoreMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodexAuthCredentialsStoreMode {
    File,
    Keyring,
    Auto,
    Ephemeral,
}

/// `CodexKeyring`: where Codex keeps its credential when the store mode is
/// `keyring` or `auto`.
#[async_trait]
pub trait CodexKeyring: Send + Sync {
    async fn read(&self, service: &str, account: &str) -> Option<String>;
    async fn write(&self, service: &str, account: &str, value: &str) -> HarnessResult<()>;
}

/// `readSubscription` of [`ResolveCodexAuthenticationOptions`]: called with
/// `authCredentialsStoreMode`.
pub type ReadCodexSubscriptionFn = Arc<
    dyn Fn(
            Option<JsonValue>,
        ) -> BoxFuture<'static, HarnessResult<Option<CodexResolvedAuthentication>>>
        + Send
        + Sync,
>;

/// Options of [`resolve_codex_authentication`].
#[derive(Clone, Default)]
pub struct ResolveCodexAuthenticationOptions {
    pub auth: Option<CodexAuthenticationMode>,
    /// Defaults to the host's `process.env`.
    pub process_env: Option<IndexMap<String, String>>,
    pub auth_credentials_store_mode: Option<JsonValue>,
    /// Defaults to [`read_codex_subscription`].
    pub read_subscription: Option<ReadCodexSubscriptionFn>,
}

/// `resolveCodexAuthentication({ auth, processEnv, authCredentialsStoreMode,
/// readSubscription })`.
pub async fn resolve_codex_authentication(
    options: ResolveCodexAuthenticationOptions,
) -> HarnessResult<CodexResolvedAuthentication> {
    let auth = options.auth.as_ref();
    let process_env = options
        .process_env
        .clone()
        .unwrap_or_else(|| std::env::vars().collect());
    let environment = resolve_codex_env(auth, &process_env);
    if matches!(auth, Some(HarnessAuthentication::Environment(_))) {
        return Ok(CodexResolvedAuthentication {
            environment,
            request_headers: None,
        });
    }
    let has_direct_credential = environment.contains_key("CODEX_API_KEY");
    if !should_resolve_native_subscription(auth, &environment, has_direct_credential) {
        return Ok(CodexResolvedAuthentication {
            environment,
            request_headers: None,
        });
    }

    let subscription = match &options.read_subscription {
        Some(read) => read(options.auth_credentials_store_mode.clone()).await?,
        None => {
            read_codex_subscription(ReadCodexSubscriptionOptions {
                auth_credentials_store_mode: options.auth_credentials_store_mode.clone(),
                ..ReadCodexSubscriptionOptions::default()
            })
            .await?
        }
    };
    Ok(subscription.unwrap_or(CodexResolvedAuthentication {
        environment,
        request_headers: None,
    }))
}

/// Options of [`read_codex_subscription`]. Unset fields default to the
/// host's `process.env`, `homedir()`, `process.platform` and the
/// platform's keyring.
#[derive(Clone, Default)]
pub struct ReadCodexSubscriptionOptions {
    pub env: Option<IndexMap<String, String>>,
    pub home_directory: Option<String>,
    pub auth_credentials_store_mode: Option<JsonValue>,
    pub platform: Option<String>,
    pub keyring: Option<Arc<dyn CodexKeyring>>,
    pub fetch: Option<OAuthFetch>,
    /// [rust] The `execFile` seam the platform keyrings read through, which
    /// the TS tests reach by mocking the credential-store readers.
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

/// [rust] `os.homedir()`.
fn host_home_directory() -> Option<PathBuf> {
    std::env::home_dir().filter(|path| !path.as_os_str().is_empty())
}

/// `readCodexSubscription({ env, homeDirectory, authCredentialsStoreMode,
/// platform, keyring, fetch })`.
pub async fn read_codex_subscription(
    options: ReadCodexSubscriptionOptions,
) -> HarnessResult<Option<CodexResolvedAuthentication>> {
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

    let codex_home = resolve_path(
        &env.get("CODEX_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home_directory.join(".codex")),
    );
    let auth_path = codex_home.join("auth.json");
    let storage_mode =
        match to_codex_auth_credentials_store_mode(options.auth_credentials_store_mode.as_ref()) {
            Some(mode) => mode,
            None => read_codex_auth_credentials_store_mode(&codex_home)
                .await
                .unwrap_or(CodexAuthCredentialsStoreMode::File),
        };
    let keyring = options
        .keyring
        .or_else(|| create_codex_keyring(&platform, exec_file));
    let Some(stored) =
        read_codex_auth_store(&codex_home, &auth_path, storage_mode, keyring.as_deref()).await
    else {
        return Ok(None);
    };

    let Some(credential) = to_codex_credential(&stored.value) else {
        return Ok(None);
    };

    let mut access_token = credential.access_token.clone();
    if is_access_token_expiring_soon(credential.expires_at, None, None) {
        let refreshed = refresh_oauth_access_token(RefreshOAuthAccessTokenOptions {
            token_url: OPENAI_TOKEN_URL.into(),
            client_id: OPENAI_CLIENT_ID.into(),
            refresh_token: credential.refresh_token.clone(),
            request_format: OAuthRequestFormat::Json,
            headers: None,
            fetch: options.fetch,
        })
        .await?;
        access_token = refreshed.access_token.clone();
        let mut updated = stored.value.clone();
        updated.insert(
            "last_refresh".into(),
            crate::harness::utils::iso_timestamp_now().into(),
        );
        let mut tokens = match stored.value.get("tokens") {
            Some(JsonValue::Object(tokens)) => tokens.clone(),
            _ => Map::new(),
        };
        tokens.insert("access_token".into(), refreshed.access_token.into());
        tokens.insert(
            "refresh_token".into(),
            refreshed
                .refresh_token
                .unwrap_or(credential.refresh_token)
                .into(),
        );
        updated.insert("tokens".into(), JsonValue::Object(tokens));
        stored.write(&updated, keyring.as_deref()).await?;
    }

    Ok(Some(CodexResolvedAuthentication {
        environment: IndexMap::from([
            ("CODEX_API_KEY".to_string(), access_token),
            (
                "OPENAI_BASE_URL".to_string(),
                CHATGPT_CODEX_BASE_URL.to_string(),
            ),
        ]),
        request_headers: credential
            .account_id
            .map(|account_id| IndexMap::from([("ChatGPT-Account-ID".to_string(), account_id)])),
    }))
}

type CodexAuthFile = Map<String, JsonValue>;

enum CodexAuthStoreLocation {
    File(PathBuf),
    Keyring { account: String },
}

struct CodexAuthStore {
    value: CodexAuthFile,
    location: CodexAuthStoreLocation,
}

impl CodexAuthStore {
    async fn write(
        &self,
        updated: &CodexAuthFile,
        keyring: Option<&dyn CodexKeyring>,
    ) -> HarnessResult<()> {
        match &self.location {
            CodexAuthStoreLocation::File(auth_path) => {
                write_codex_auth_file(auth_path, updated).await
            }
            CodexAuthStoreLocation::Keyring { account } => {
                let keyring = keyring.ok_or_else(|| plain_error("missing keyring"))?;
                keyring
                    .write(KEYRING_SERVICE, account, &serde_json::to_string(updated)?)
                    .await
            }
        }
    }
}

async fn read_codex_auth_store(
    codex_home: &Path,
    auth_path: &Path,
    storage_mode: CodexAuthCredentialsStoreMode,
    keyring: Option<&dyn CodexKeyring>,
) -> Option<CodexAuthStore> {
    let read_file_store = || async {
        let value = read_codex_auth_file(auth_path).await?;
        Some(CodexAuthStore {
            value,
            location: CodexAuthStoreLocation::File(auth_path.to_path_buf()),
        })
    };
    let read_keyring_store = || async {
        let keyring = keyring?;
        let canonical_home = tokio::fs::canonicalize(codex_home)
            .await
            .unwrap_or_else(|_| codex_home.to_path_buf());
        let digest = super::tools::sha256_hex(&canonical_home.to_string_lossy());
        let account = format!("cli|{}", &digest[..16]);
        let text = keyring.read(KEYRING_SERVICE, &account).await?;
        let value = parse_codex_auth_file(&text)?;
        Some(CodexAuthStore {
            value,
            location: CodexAuthStoreLocation::Keyring { account },
        })
    };

    match storage_mode {
        CodexAuthCredentialsStoreMode::Ephemeral => None,
        CodexAuthCredentialsStoreMode::File => read_file_store().await,
        CodexAuthCredentialsStoreMode::Keyring => read_keyring_store().await,
        CodexAuthCredentialsStoreMode::Auto => match read_keyring_store().await {
            Some(store) => Some(store),
            None => read_file_store().await,
        },
    }
}

/// `cli_auth_credentials_store` from the top level of `config.toml`.
async fn read_codex_auth_credentials_store_mode(
    codex_home: &Path,
) -> Option<CodexAuthCredentialsStoreMode> {
    static LINE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r#"^\s*cli_auth_credentials_store\s*=\s*["'](file|keyring|auto|ephemeral)["']\s*(?:#.*)?$"#,
        )
        .expect("valid regex")
    });
    let text = tokio::fs::read_to_string(codex_home.join("config.toml"))
        .await
        .ok()?;
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.trim_start().starts_with('[') {
            return None;
        }
        if let Some(captures) = LINE.captures(line) {
            return to_codex_auth_credentials_store_mode(Some(&JsonValue::String(
                captures[1].to_string(),
            )));
        }
    }
    None
}

fn to_codex_auth_credentials_store_mode(
    value: Option<&JsonValue>,
) -> Option<CodexAuthCredentialsStoreMode> {
    match value?.as_str()? {
        "file" => Some(CodexAuthCredentialsStoreMode::File),
        "keyring" => Some(CodexAuthCredentialsStoreMode::Keyring),
        "auto" => Some(CodexAuthCredentialsStoreMode::Auto),
        "ephemeral" => Some(CodexAuthCredentialsStoreMode::Ephemeral),
        _ => None,
    }
}

async fn read_codex_auth_file(auth_path: &Path) -> Option<CodexAuthFile> {
    let text = tokio::fs::read_to_string(auth_path).await.ok()?;
    parse_codex_auth_file(&text)
}

fn parse_codex_auth_file(text: &str) -> Option<CodexAuthFile> {
    match serde_json::from_str(text).ok()? {
        JsonValue::Object(value) => Some(value),
        _ => None,
    }
}

struct CodexCredential {
    access_token: String,
    refresh_token: String,
    expires_at: f64,
    account_id: Option<String>,
}

fn to_codex_credential(value: &CodexAuthFile) -> Option<CodexCredential> {
    if value.get("auth_mode").and_then(JsonValue::as_str) != Some("chatgpt") {
        return None;
    }
    let tokens = value.get("tokens")?.as_object()?;
    let access_token = tokens.get("access_token")?.as_str()?.to_string();
    let refresh_token = tokens.get("refresh_token")?.as_str()?.to_string();
    let expires_at = get_jwt_expires_at(&access_token)?;
    Some(CodexCredential {
        access_token,
        refresh_token,
        expires_at,
        account_id: tokens
            .get("account_id")
            .and_then(JsonValue::as_str)
            .map(str::to_string),
    })
}

async fn write_codex_auth_file(auth_path: &Path, value: &CodexAuthFile) -> HarnessResult<()> {
    if let Some(parent) = auth_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut temporary_path = auth_path.as_os_str().to_owned();
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
    tokio::fs::rename(&temporary_path, auth_path).await?;
    Ok(())
}

/// `createCodexKeyring({ platform })`.
fn create_codex_keyring(
    platform: &str,
    exec_file: Arc<dyn ExecFile>,
) -> Option<Arc<dyn CodexKeyring>> {
    let kind = if is_macos(platform) {
        PlatformKeyringKind::MacOS
    } else if is_linux(platform) {
        PlatformKeyringKind::Linux
    } else if is_windows(platform) {
        PlatformKeyringKind::Windows
    } else {
        return None;
    };
    Some(Arc::new(PlatformKeyring { kind, exec_file }))
}

#[derive(Clone, Copy)]
enum PlatformKeyringKind {
    MacOS,
    Linux,
    Windows,
}

struct PlatformKeyring {
    kind: PlatformKeyringKind,
    exec_file: Arc<dyn ExecFile>,
}

const WINDOWS_CREDENTIAL_SOURCE: &str = r#"
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class CodexCredentialManager {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct Credential {
    public UInt32 Flags; public UInt32 Type; public string TargetName;
    public string Comment; public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public UInt32 CredentialBlobSize; public IntPtr CredentialBlob;
    public UInt32 Persist; public UInt32 AttributeCount; public IntPtr Attributes;
    public string TargetAlias; public string UserName;
  }
  [DllImport("advapi32.dll", EntryPoint = "CredWriteW", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern bool CredWrite(ref Credential credential, UInt32 flags);
  public static void Write(string target, string userName, string value) {
    byte[] bytes = Encoding.Unicode.GetBytes(value);
    IntPtr blob = Marshal.AllocHGlobal(bytes.Length);
    try {
      Marshal.Copy(bytes, 0, blob, bytes.Length);
      Credential credential = new Credential {
        Type = 1, TargetName = target, CredentialBlobSize = (UInt32)bytes.Length,
        CredentialBlob = blob, Persist = 3, UserName = userName
      };
      if (!CredWrite(ref credential, 0)) throw new System.ComponentModel.Win32Exception();
    } finally { Marshal.FreeHGlobal(blob); }
  }
}"#;

const WINDOWS_WRITE_SCRIPT: &str = r#"Add-Type -TypeDefinition $env:AI_SDK_CODEX_CREDENTIAL_SOURCE; $value = [Console]::In.ReadToEnd(); [CodexCredentialManager]::Write($env:AI_SDK_CODEX_CREDENTIAL_TARGET, $env:AI_SDK_CODEX_CREDENTIAL_USER, $value)"#;

#[async_trait]
impl CodexKeyring for PlatformKeyring {
    async fn read(&self, service: &str, account: &str) -> Option<String> {
        let exec = self.exec_file.as_ref();
        match self.kind {
            PlatformKeyringKind::MacOS => macos_keychain::read_with(exec, service, account).await,
            PlatformKeyringKind::Linux => {
                linux_secret_service::read_with(
                    exec,
                    &IndexMap::from([
                        ("service".to_string(), service.to_string()),
                        ("username".to_string(), account.to_string()),
                        ("target".to_string(), "default".to_string()),
                    ]),
                )
                .await
            }
            PlatformKeyringKind::Windows => {
                windows_credential_manager::read_with(exec, &format!("{account}.{service}")).await
            }
        }
    }

    async fn write(&self, service: &str, account: &str, value: &str) -> HarnessResult<()> {
        match self.kind {
            PlatformKeyringKind::MacOS => {
                let hex: String = value.bytes().map(|byte| format!("{byte:02x}")).collect();
                let command = format!(
                    "add-generic-password -U -s {} -a {} -X {hex}\n",
                    shell_quote_for_security(service),
                    shell_quote_for_security(account),
                );
                run_command_with_input("/usr/bin/security", &["-i".into()], &command, None).await
            }
            PlatformKeyringKind::Linux => {
                let args = [
                    "store".to_string(),
                    "--label".into(),
                    format!("{account}@{service}:default"),
                    "service".into(),
                    service.into(),
                    "username".into(),
                    account.into(),
                    "target".into(),
                    "default".into(),
                    "application".into(),
                    "rust-keyring".into(),
                ];
                run_command_with_input("secret-tool", &args, value, None).await
            }
            PlatformKeyringKind::Windows => {
                let args = [
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    WINDOWS_WRITE_SCRIPT,
                ]
                .map(String::from);
                let env = IndexMap::from([
                    (
                        "AI_SDK_CODEX_CREDENTIAL_SOURCE".to_string(),
                        WINDOWS_CREDENTIAL_SOURCE.to_string(),
                    ),
                    (
                        "AI_SDK_CODEX_CREDENTIAL_TARGET".to_string(),
                        format!("{account}.{service}"),
                    ),
                    (
                        "AI_SDK_CODEX_CREDENTIAL_USER".to_string(),
                        account.to_string(),
                    ),
                ]);
                run_command_with_input("powershell.exe", &args, value, Some(env)).await
            }
        }
    }
}

fn shell_quote_for_security(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Runs `command` with `input` on stdin, rejecting with its stderr (or the
/// exit code) when it fails. `env` is added to the inherited environment.
async fn run_command_with_input(
    command: &str,
    args: &[String],
    input: &str,
    env: Option<IndexMap<String, String>>,
) -> HarnessResult<()> {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt as _;

    let mut child = tokio::process::Command::new(command);
    child
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(env) = env {
        child.envs(env);
    }
    let mut child = child.spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(input.as_bytes()).await?;
    }
    let output = child.wait_with_output().await?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    Err(plain_error(if stderr.is_empty() {
        format!(
            "{command} exited with code {}.",
            output
                .status
                .code()
                .map_or_else(|| "null".to_string(), |code| code.to_string())
        )
    } else {
        stderr
    }))
}
