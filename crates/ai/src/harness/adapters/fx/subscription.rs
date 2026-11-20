//! Port of `harness-fx/src/fx-auth.ts` and `fx-subscription.ts`: how an fx
//! session authenticates. An AI Gateway key, or the credentials of a Codex
//! (ChatGPT plan) or Grok subscription that fx, Codex or Grok saved on the
//! host, which the session reads again and updates when libfx refreshes them.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use super::libfx::authorizer::{FxCredentialSource, FxProviderCredentials, FxSubscriptionProvider};
use super::transport::FxProvider;
use crate::harness::auth::{HarnessAuthentication, HarnessDirectAuthentication};
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::utils::{
    current_platform, get_ai_gateway_auth_from_env, get_jwt_expires_at, is_linux, is_macos,
    is_windows, iso_timestamp, iso_timestamp_now, read_linux_secret_service_password,
    read_macos_keychain_password, read_windows_credential_manager_password,
    should_resolve_native_subscription,
};

/// `FxAuthenticationMode`.
pub type FxAuthenticationMode = HarnessAuthentication;

const GROK_SCOPE: &str = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828";
const KEYRING_SERVICE: &str = "Codex Auth";

/// `resolveFxEnv({ auth, env })`.
pub fn resolve_fx_env(auth: Option<&FxAuthenticationMode>, env: &IndexMap<String, String>) -> IndexMap<String, String> {
    let gateway = match auth {
        Some(HarnessAuthentication::Environment(environment)) => get_ai_gateway_auth_from_env(environment),
        _ => get_ai_gateway_auth_from_env(env),
    };
    let mut resolved = IndexMap::new();
    if let Some(api_key) = gateway.api_key {
        resolved.insert("AI_GATEWAY_API_KEY".to_string(), api_key);
    }
    resolved.insert("AI_GATEWAY_BASE_URL".to_string(), gateway.base_url);
    resolved
}

/// Which file format a credential store holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CredentialFormat {
    Fx,
    Codex,
    Grok,
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `CredentialStore`.
#[async_trait]
trait CredentialStore: Send + Sync {
    async fn read(&self) -> Option<Map<String, Value>>;
    async fn write(&self, value: Map<String, Value>) -> HarnessResult<()>;
}

/// `CodexKeyring`: where Codex keeps its credential in `keyring` or `auto`
/// mode.
#[async_trait]
pub trait FxCodexKeyring: Send + Sync {
    async fn read(&self, service: &str, account: &str) -> Option<String>;
    async fn write(&self, service: &str, account: &str, value: &str) -> HarnessResult<()>;
}

/// `FxSubscription`: the selected store, read again on every use.
#[derive(Clone)]
pub struct FxSubscription {
    pub provider: FxSubscriptionProvider,
    format: CredentialFormat,
    store: Arc<dyn CredentialStore>,
}

impl std::fmt::Debug for FxSubscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FxSubscription")
            .field("provider", &self.provider)
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

/// `FxResolvedAuthentication`.
#[derive(Debug, Clone)]
pub enum FxResolvedAuthentication {
    Gateway { api_key: String, base_url: String },
    Subscription(FxSubscription),
}

impl FxResolvedAuthentication {
    pub fn provider(&self) -> FxProvider {
        match self {
            Self::Gateway { .. } => FxProvider::Gateway,
            Self::Subscription(subscription) => match subscription.provider {
                FxSubscriptionProvider::Codex => FxProvider::Codex,
                FxSubscriptionProvider::Grok => FxProvider::Grok,
            },
        }
    }
}

/// Options of [`resolve_fx_authentication`].
#[derive(Clone, Default)]
pub struct ResolveFxAuthenticationOptions {
    pub auth: Option<FxAuthenticationMode>,
    /// Defaults to the process environment.
    pub env: Option<IndexMap<String, String>>,
    /// Defaults to the host's home directory.
    pub home_directory: Option<PathBuf>,
    pub platform: Option<String>,
    /// The provider a resumed session's credentials must come from.
    pub required_provider: Option<FxProvider>,
    pub keyring: Option<Arc<dyn FxCodexKeyring>>,
}

fn host_home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default()
}

fn absolute(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir().map(|dir| dir.join(&path)).unwrap_or(path)
    }
}

/// `resolveFxAuthentication(options)`.
pub async fn resolve_fx_authentication(options: ResolveFxAuthenticationOptions) -> HarnessResult<FxResolvedAuthentication> {
    let env = options
        .env
        .clone()
        .unwrap_or_else(|| std::env::vars().collect());
    let home = options.home_directory.clone().unwrap_or_else(host_home);
    let platform = options
        .platform
        .clone()
        .unwrap_or_else(|| current_platform().to_string());
    let auth = options.auth.as_ref();
    let environment = resolve_fx_env(auth, &env);
    let isolated = matches!(auth, Some(HarnessAuthentication::Environment(_)));
    let required = options.required_provider;
    let native_required = required.is_some_and(|provider| provider != FxProvider::Gateway);
    if native_required && (isolated || matches!(auth, Some(HarnessAuthentication::AiGateway))) {
        return Err(HarnessError::new(
            "Cannot resume fx: authentication does not allow the saved subscription provider.",
        )
        .into());
    }
    let direct = HarnessAuthentication::Choice(HarnessDirectAuthentication::Direct);
    if !isolated
        && required != Some(FxProvider::Gateway)
        && should_resolve_native_subscription(
            if native_required { Some(&direct) } else { auth },
            &environment,
            false,
        )
    {
        let wants = |provider: FxProvider| required.is_none() || required == Some(provider);
        for (provider, file) in [
            (FxSubscriptionProvider::Codex, "chatgpt-auth.json"),
            (FxSubscriptionProvider::Grok, "grok-auth.json"),
        ] {
            if !wants(provider_of(provider)) {
                continue;
            }
            let store = FileStore::new(home.join(".fx").join(file));
            if let Some(subscription) = bind_subscription(provider, CredentialFormat::Fx, store).await {
                return Ok(FxResolvedAuthentication::Subscription(subscription));
            }
        }
        if wants(FxProvider::Codex) {
            let codex_home = absolute(env.get("CODEX_HOME").map_or_else(|| home.join(".codex"), PathBuf::from));
            let mode = codex_store_mode(&codex_home).await;
            if matches!(mode, "keyring" | "auto") {
                let keyring = options
                    .keyring
                    .clone()
                    .or_else(|| create_codex_keyring(&platform));
                if let Some(keyring) = keyring {
                    let canonical = tokio::fs::canonicalize(&codex_home)
                        .await
                        .unwrap_or_else(|_| codex_home.clone());
                    let digest = ring::digest::digest(
                        &ring::digest::SHA256,
                        canonical.to_string_lossy().as_bytes(),
                    );
                    let hex: String = digest.as_ref().iter().map(|byte| format!("{byte:02x}")).collect();
                    let store = KeyringStore {
                        keyring,
                        account: format!("cli|{}", &hex[..16]),
                    };
                    if let Some(subscription) =
                        bind_subscription(FxSubscriptionProvider::Codex, CredentialFormat::Codex, store).await
                    {
                        return Ok(FxResolvedAuthentication::Subscription(subscription));
                    }
                }
            }
            if matches!(mode, "file" | "auto") {
                let store = FileStore::new(codex_home.join("auth.json"));
                if let Some(subscription) =
                    bind_subscription(FxSubscriptionProvider::Codex, CredentialFormat::Codex, store).await
                {
                    return Ok(FxResolvedAuthentication::Subscription(subscription));
                }
            }
        }
        if wants(FxProvider::Grok) {
            let grok_home = absolute(env.get("GROK_HOME").map_or_else(|| home.join(".grok"), PathBuf::from));
            let store = FileStore::new(grok_home.join("auth.json"));
            if let Some(subscription) = bind_subscription(FxSubscriptionProvider::Grok, CredentialFormat::Grok, store).await {
                return Ok(FxResolvedAuthentication::Subscription(subscription));
            }
        }
    }
    if let Some(provider) = required.filter(|_| native_required) {
        return Err(HarnessError::new(format!(
            "Cannot resume fx: no {} subscription credentials are available.",
            provider.as_str()
        ))
        .into());
    }
    let Some(api_key) = environment.get("AI_GATEWAY_API_KEY").cloned() else {
        return Err(HarnessError::new(
            "fx requires AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN in the host environment or an explicit authentication environment. No subscription credentials were available.",
        )
        .into());
    };
    Ok(FxResolvedAuthentication::Gateway {
        api_key,
        base_url: environment["AI_GATEWAY_BASE_URL"].clone(),
    })
}

fn provider_of(provider: FxSubscriptionProvider) -> FxProvider {
    match provider {
        FxSubscriptionProvider::Codex => FxProvider::Codex,
        FxSubscriptionProvider::Grok => FxProvider::Grok,
    }
}

/// `cli_auth_credentials_store` from the top of Codex's `config.toml`.
async fn codex_store_mode(codex_home: &Path) -> &'static str {
    static LINE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r#"^\s*cli_auth_credentials_store\s*=\s*["'](file|keyring|auto|ephemeral)["']\s*(?:#.*)?$"#,
        )
        .expect("valid regex")
    });
    let config = tokio::fs::read_to_string(codex_home.join("config.toml"))
        .await
        .unwrap_or_default();
    for line in config.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.trim_start().starts_with('[') {
            break;
        }
        if let Some(captures) = LINE.captures(line) {
            return match &captures[1] {
                "keyring" => "keyring",
                "auto" => "auto",
                "ephemeral" => "ephemeral",
                _ => "file",
            };
        }
    }
    "file"
}

async fn bind_subscription(
    provider: FxSubscriptionProvider,
    format: CredentialFormat,
    store: impl CredentialStore + 'static,
) -> Option<FxSubscription> {
    let value = store.read().await?;
    credentials_from_document(&value, format)?;
    Some(FxSubscription {
        provider,
        format,
        store: Arc::new(store),
    })
}

impl FxSubscription {
    /// `read()`.
    pub async fn read(&self) -> HarnessResult<FxProviderCredentials> {
        let credentials = self
            .store
            .read()
            .await
            .and_then(|value| credentials_from_document(&value, self.format));
        credentials.ok_or_else(|| {
            HarnessError::new(format!(
                "The selected fx {} subscription credentials are no longer available.",
                self.provider.as_str()
            ))
            .into()
        })
    }

    /// `write({ credentials })`.
    pub async fn write(&self, credentials: &FxProviderCredentials) -> HarnessResult<()> {
        let value = self.store.read().await;
        let Some((value, mut record)) = value.and_then(|value| {
            credentials_from_document(&value, self.format)?;
            let record = credential_record(&value, self.format)?.clone();
            Some((value, record))
        }) else {
            return Err(HarnessError::new("Cannot update the selected fx subscription credential store.").into());
        };
        let grok = self.format == CredentialFormat::Grok;
        record.insert(
            if grok { "key" } else { "access_token" }.into(),
            json!(credentials.access_token),
        );
        if let Some(refresh_token) = &credentials.refresh_token {
            record.insert("refresh_token".into(), json!(refresh_token));
        }
        if let Some(account_id) = &credentials.account_id {
            record.insert(if grok { "user_id" } else { "account_id" }.into(), json!(account_id));
        }
        if let Some(expires_at) = credentials.expires_at {
            match self.format {
                CredentialFormat::Fx => {
                    record.insert("expires_at_ms".into(), json_number(expires_at));
                }
                CredentialFormat::Grok => {
                    record.insert("expires_at".into(), json!(iso_timestamp(expires_at as i64)));
                }
                CredentialFormat::Codex => {}
            }
        }
        let next = match self.format {
            CredentialFormat::Fx => record,
            CredentialFormat::Codex => {
                let mut next = value;
                next.insert("tokens".into(), Value::Object(record));
                next.insert("last_refresh".into(), json!(iso_timestamp_now()));
                next
            }
            CredentialFormat::Grok => {
                let mut next = value;
                next.insert(GROK_SCOPE.into(), Value::Object(record));
                next
            }
        };
        self.store.write(next).await
    }
}

impl FxCredentialSource for FxSubscription {
    fn read(&self) -> BoxFuture<'_, Result<FxProviderCredentials, String>> {
        Box::pin(async { FxSubscription::read(self).await.map_err(|error| error.to_string()) })
    }

    fn write(&self, credentials: FxProviderCredentials) -> BoxFuture<'_, Result<(), String>> {
        Box::pin(async move {
            FxSubscription::write(self, &credentials)
                .await
                .map_err(|error| error.to_string())
        })
    }
}

/// A whole number as a JSON integer, like `JSON.stringify`.
fn json_number(value: f64) -> Value {
    if value.fract() == 0.0 && value.abs() < 9.0e15 {
        json!(value as i64)
    } else {
        json!(value)
    }
}

fn credential_record(value: &Map<String, Value>, format: CredentialFormat) -> Option<&Map<String, Value>> {
    match format {
        CredentialFormat::Fx => (value.get("version") == Some(&json!(1))).then_some(value),
        CredentialFormat::Codex => {
            if value.get("auth_mode").and_then(Value::as_str) != Some("chatgpt") {
                return None;
            }
            value.get("tokens")?.as_object()
        }
        CredentialFormat::Grok => value.get(GROK_SCOPE)?.as_object(),
    }
}

fn non_empty(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// `credentialsFromDocument({ value, format })`.
fn credentials_from_document(value: &Map<String, Value>, format: CredentialFormat) -> Option<FxProviderCredentials> {
    let record = credential_record(value, format)?;
    let grok = format == CredentialFormat::Grok;
    let access_token = non_empty(record.get(if grok { "key" } else { "access_token" }))?;
    let refresh_token = non_empty(record.get("refresh_token"));
    let account_id = non_empty(record.get(if grok { "user_id" } else { "account_id" }));
    let expires_at = match format {
        CredentialFormat::Fx => {
            refresh_token.as_ref()?;
            account_id.as_ref()?;
            Some(record.get("expires_at_ms")?.as_f64().filter(|value| value.is_finite())?)
        }
        CredentialFormat::Codex => get_jwt_expires_at(&access_token),
        CredentialFormat::Grok => match record.get("expires_at") {
            Some(Value::String(text)) => parse_iso_millis(text),
            Some(Value::Number(number)) => number.as_f64().filter(|value| value.is_finite()).map(|value| {
                if value < 10_000_000_000.0 {
                    value * 1000.0
                } else {
                    value
                }
            }),
            _ => None,
        },
    };
    Some(FxProviderCredentials {
        access_token,
        refresh_token,
        account_id,
        expires_at,
    })
}

/// `Date.parse` for ISO 8601 date-times (`2033-05-18T03:33:20.000Z`,
/// with an optional fraction and a `Z` or `±hh:mm` offset).
fn parse_iso_millis(text: &str) -> Option<f64> {
    static ISO: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r"^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?(Z|[+-]\d{2}:\d{2})?)?$",
        )
        .expect("valid regex")
    });
    let captures = ISO.captures(text.trim())?;
    let number = |index: usize| captures.get(index).map_or(Some(0), |value| value.as_str().parse::<i64>().ok());
    let (year, month, day) = (number(1)?, number(2)?, number(3)?);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let (hour, minute, second) = (number(4)?, number(5)?, number(6)?);
    let millis = captures.get(7).map_or(0, |fraction| {
        let digits = format!("{:0<3}", fraction.as_str());
        digits[..3].parse::<i64>().unwrap_or(0)
    });
    // Howard Hinnant's days_from_civil.
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let offset = match captures.get(8).map(|offset| offset.as_str()) {
        None | Some("Z") => 0,
        Some(offset) => {
            let sign = if offset.starts_with('-') { -1 } else { 1 };
            let hours: i64 = offset[1..3].parse().ok()?;
            let minutes: i64 = offset[4..6].parse().ok()?;
            sign * (hours * 60 + minutes) * 60_000
        }
    };
    Some((((days * 24 + hour) * 60 + minute) * 60 + second) as f64 * 1000.0 + millis as f64 - offset as f64)
}

fn parse_document(text: &str) -> Option<Map<String, Value>> {
    match serde_json::from_str(text).ok()? {
        Value::Object(value) => Some(value),
        _ => None,
    }
}

/// `fileStore({ filePath })`.
struct FileStore {
    path: PathBuf,
}

impl FileStore {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

#[async_trait]
impl CredentialStore for FileStore {
    async fn read(&self) -> Option<Map<String, Value>> {
        parse_document(&tokio::fs::read_to_string(&self.path).await.ok()?)
    }

    async fn write(&self, value: Map<String, Value>) -> HarnessResult<()> {
        let temporary = PathBuf::from(format!(
            "{}.{}.tmp",
            self.path.display(),
            crate::harness::utils::random_uuid()
        ));
        let text = format!("{}\n", serde_json::to_string_pretty(&Value::Object(value))?);
        let result = async {
            let mut options = tokio::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            use tokio::io::AsyncWriteExt as _;
            let mut file = options.open(&temporary).await?;
            file.write_all(text.as_bytes()).await?;
            file.flush().await?;
            drop(file);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                tokio::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o600)).await?;
            }
            tokio::fs::rename(&temporary, &self.path).await?;
            Ok::<_, std::io::Error>(())
        }
        .await;
        let _ = tokio::fs::remove_file(&temporary).await;
        result.map_err(Into::into)
    }
}

struct KeyringStore {
    keyring: Arc<dyn FxCodexKeyring>,
    account: String,
}

#[async_trait]
impl CredentialStore for KeyringStore {
    async fn read(&self) -> Option<Map<String, Value>> {
        parse_document(&self.keyring.read(KEYRING_SERVICE, &self.account).await?)
    }

    async fn write(&self, value: Map<String, Value>) -> HarnessResult<()> {
        self.keyring
            .write(KEYRING_SERVICE, &self.account, &Value::Object(value).to_string())
            .await
    }
}

/// `createCodexKeyring({ platform })`.
fn create_codex_keyring(platform: &str) -> Option<Arc<dyn FxCodexKeyring>> {
    if is_macos(platform) || is_linux(platform) || is_windows(platform) {
        Some(Arc::new(PlatformKeyring {
            platform: platform.to_string(),
        }))
    } else {
        None
    }
}

struct PlatformKeyring {
    platform: String,
}

const WINDOWS_CREDENTIAL_SOURCE: &str = r#"
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class FxCodexCredentialManager {
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

#[async_trait]
impl FxCodexKeyring for PlatformKeyring {
    async fn read(&self, service: &str, account: &str) -> Option<String> {
        if is_macos(&self.platform) {
            read_macos_keychain_password(service, account).await
        } else if is_linux(&self.platform) {
            read_linux_secret_service_password(&IndexMap::from([
                ("service".to_string(), service.to_string()),
                ("username".to_string(), account.to_string()),
                ("target".to_string(), "default".to_string()),
            ]))
            .await
        } else {
            read_windows_credential_manager_password(&format!("{account}.{service}")).await
        }
    }

    async fn write(&self, service: &str, account: &str, value: &str) -> HarnessResult<()> {
        let quote = |value: &str| format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""));
        if is_macos(&self.platform) {
            let hex: String = value.bytes().map(|byte| format!("{byte:02x}")).collect();
            let input = format!(
                "add-generic-password -U -s {} -a {} -X {hex}\n",
                quote(service),
                quote(account)
            );
            run_command_with_input("/usr/bin/security", &["-i"], &input, &[]).await
        } else if is_linux(&self.platform) {
            let label = format!("{account}@{service}:default");
            let args = [
                "store", "--label", &label, "service", service, "username", account, "target", "default",
                "application", "rust-keyring",
            ];
            run_command_with_input("secret-tool", &args, value, &[]).await
        } else {
            let target = format!("{account}.{service}");
            run_command_with_input(
                "powershell.exe",
                &[
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Add-Type -TypeDefinition $env:AI_SDK_FX_CREDENTIAL_SOURCE; $value = [Console]::In.ReadToEnd(); [FxCodexCredentialManager]::Write($env:AI_SDK_FX_CREDENTIAL_TARGET, $env:AI_SDK_FX_CREDENTIAL_USER, $value)",
                ],
                value,
                &[
                    ("AI_SDK_FX_CREDENTIAL_SOURCE", WINDOWS_CREDENTIAL_SOURCE),
                    ("AI_SDK_FX_CREDENTIAL_TARGET", &target),
                    ("AI_SDK_FX_CREDENTIAL_USER", account),
                ],
            )
            .await
        }
    }
}

/// `runCommandWithInput({ command, args, input, env })`.
async fn run_command_with_input(command: &str, args: &[&str], input: &str, env: &[(&str, &str)]) -> HarnessResult<()> {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt as _;
    let mut child = tokio::process::Command::new(command);
    child
        .args(args)
        .envs(env.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let mut child = child
        .spawn()
        .map_err(|_| HarnessError::new("Unable to access the Codex credential store."))?;
    let update_failed = || HarnessError::new("Unable to update the Codex credential store.");
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(input.as_bytes()).await.map_err(|_| update_failed())?;
    }
    let status = child.wait().await.map_err(|_| update_failed())?;
    if status.success() {
        Ok(())
    } else {
        Err(update_failed().into())
    }
}

/// `resolveFxModel({ provider, model, homeDirectory })`: an explicit model,
/// else the model the user's fx settings save for a subscription provider
/// (`models.<provider>`, or the legacy `<provider>_model`).
pub async fn resolve_fx_model(provider: FxProvider, model: Option<String>, home_directory: Option<&Path>) -> Option<String> {
    if model.as_deref().is_some_and(|model| !model.is_empty()) || provider == FxProvider::Gateway {
        return model;
    }
    let home = home_directory.map_or_else(host_home, Path::to_path_buf);
    let settings = tokio::fs::read_to_string(home.join(".fx").join("settings.json"))
        .await
        .ok()
        .and_then(|text| parse_document(&text))?;
    let saved = settings
        .get("models")
        .and_then(Value::as_object)
        .and_then(|models| models.get(provider.as_str()))
        .filter(|saved| !saved.is_null())
        .or_else(|| settings.get(&format!("{}_model", provider.as_str())))?;
    saved
        .as_str()
        .filter(|saved| !saved.trim().is_empty())
        .map(str::to_string)
}

#[cfg(test)]
#[path = "subscription_tests.rs"]
mod tests;
