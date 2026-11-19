//! Port of `harness-claude-code/src/claude-code-auth.ts`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use indexmap::IndexMap;

use crate::harness::errors::HarnessResult;
use crate::harness::spec::HarnessV1Authentication;
use crate::harness::spec::network_sandbox::{
    HarnessV1RequestTransformation, HarnessV1RequestTransformationSources,
};
use crate::harness::utils::{
    AiGatewayAuth, create_credential_request_transformation, get_ai_gateway_auth_from_env,
};

/// `CLAUDE_CODE_CREDENTIAL_ENVIRONMENT_VARIABLES`.
pub const CLAUDE_CODE_CREDENTIAL_ENVIRONMENT_VARIABLES: [&str; 4] = [
    "AI_GATEWAY_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
];

const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";

/// The URL credential transformations match: the configured base URL, which
/// for direct auth defaults to the Anthropic API.
pub(crate) fn claude_code_match_url(
    environment: &IndexMap<String, String>,
    authentication_mode: ClaudeCodeResolvedAuthenticationMode,
) -> String {
    match authentication_mode {
        // [rust] `new URL(undefined)` throws in TS; an empty URL fails the
        // same way.
        ClaudeCodeResolvedAuthenticationMode::AiGateway => environment
            .get("ANTHROPIC_BASE_URL")
            .cloned()
            .unwrap_or_default(),
        ClaudeCodeResolvedAuthenticationMode::Direct => environment
            .get("ANTHROPIC_BASE_URL")
            .cloned()
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_BASE_URL.to_string()),
    }
}

/// `createClaudeCodeRequestTransformations({ env, sandboxEnv, auth })`.
pub fn create_claude_code_request_transformations(
    sources: &HarnessV1RequestTransformationSources<ClaudeCodeResolvedAuthenticationMode>,
) -> HarnessResult<Vec<HarnessV1RequestTransformation>> {
    let environment = &sources.env;
    let sandbox_environment = &sources.sandbox_env;
    let match_url = claude_code_match_url(environment, sources.auth);
    let mut transformations = Vec::new();

    if let (Some(api_key), Some(sandbox_api_key)) = (
        environment.get("ANTHROPIC_API_KEY"),
        sandbox_environment.get("ANTHROPIC_API_KEY"),
    ) {
        transformations.push(create_credential_request_transformation(
            &match_url,
            &IndexMap::from([("x-api-key".to_string(), sandbox_api_key.clone())]),
            &IndexMap::from([("x-api-key".to_string(), api_key.clone())]),
        )?);
    }

    if let (Some(auth_token), Some(sandbox_auth_token)) = (
        environment.get("ANTHROPIC_AUTH_TOKEN"),
        sandbox_environment.get("ANTHROPIC_AUTH_TOKEN"),
    ) {
        transformations.push(create_credential_request_transformation(
            &match_url,
            &IndexMap::from([(
                "Authorization".to_string(),
                format!("Bearer {sandbox_auth_token}"),
            )]),
            &IndexMap::from([("Authorization".to_string(), format!("Bearer {auth_token}"))]),
        )?);
    }

    Ok(transformations)
}

/// `ClaudeCodeResolvedAuthenticationMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClaudeCodeResolvedAuthenticationMode {
    Direct,
    AiGateway,
}

/// `ClaudeCodeAuthenticationMode`: `HarnessV1Authentication`.
pub type ClaudeCodeAuthenticationMode = HarnessV1Authentication;

/// Returns an API key from a custom source.
pub type ReadApiKeyHelper = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// `ResolveClaudeCodeEnvOptions`.
///
/// Resolve the environment-variable blob the bridge needs to authenticate
/// with Anthropic (directly or via the Vercel AI Gateway). Precedence:
///
///   1. An explicit authentication mode pins the selected route.
///   2. Auto-detect from the host process env: gateway first
///      (`AI_GATEWAY_API_KEY` / `VERCEL_OIDC_TOKEN`), then direct
///      (`ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`).
#[derive(Clone, Default)]
pub struct ResolveClaudeCodeEnvOptions {
    /// Returns an API key from a custom source (e.g. a password manager).
    /// Used as the last-resort fallback in the auto-detect branch when no
    /// static env vars or explicit auth are configured. Defaults to running
    /// the `apiKeyHelper` command from `~/.claude/settings.json`, matching
    /// the `claude` CLI's own behaviour.
    pub read_api_key_helper: Option<ReadApiKeyHelper>,
}

/// `resolveClaudeCodeEnv(auth, processEnv, options)`.
///
/// [rust] `processEnv` has no `process.env` default; pass the host
/// environment explicitly.
pub fn resolve_claude_code_env(
    auth: Option<&ClaudeCodeAuthenticationMode>,
    process_env: &IndexMap<String, String>,
    options: &ResolveClaudeCodeEnvOptions,
) -> IndexMap<String, String> {
    let supplied_environment = match auth {
        Some(HarnessV1Authentication::Environment(environment)) => Some(environment),
        _ => None,
    };
    let authentication_environment = supplied_environment.unwrap_or(process_env);
    let is_direct = matches!(auth, Some(HarnessV1Authentication::Choice(_)));

    let read_api_key = || -> Option<String> {
        if supplied_environment.is_some() || is_direct {
            return None;
        }
        match &options.read_api_key_helper {
            Some(read) => read(),
            None => read_api_key_helper(),
        }
    };
    if is_direct {
        return pick_anthropic(authentication_environment, read_api_key);
    }

    let gateway_auth_from_env = get_ai_gateway_auth_from_env(authentication_environment);
    if matches!(auth, Some(HarnessV1Authentication::AiGateway))
        || gateway_auth_from_env.api_key.is_some()
    {
        return pick_gateway(gateway_auth_from_env);
    }

    pick_anthropic(authentication_environment, read_api_key)
}

/// `resolveClaudeCodeAuthenticationMode(auth, processEnv)`.
pub fn resolve_claude_code_authentication_mode(
    auth: Option<&ClaudeCodeAuthenticationMode>,
    process_env: &IndexMap<String, String>,
) -> ClaudeCodeResolvedAuthenticationMode {
    let from_env = |env| {
        if get_ai_gateway_auth_from_env(env).api_key.is_some() {
            ClaudeCodeResolvedAuthenticationMode::AiGateway
        } else {
            ClaudeCodeResolvedAuthenticationMode::Direct
        }
    };
    match auth {
        Some(HarnessV1Authentication::Environment(environment)) => from_env(environment),
        Some(HarnessV1Authentication::Choice(_)) => ClaudeCodeResolvedAuthenticationMode::Direct,
        Some(HarnessV1Authentication::AiGateway) => ClaudeCodeResolvedAuthenticationMode::AiGateway,
        Some(HarnessV1Authentication::Auto) | None => from_env(process_env),
    }
}

fn pick_anthropic(
    process_env: &IndexMap<String, String>,
    read_api_key: impl FnOnce() -> Option<String>,
) -> IndexMap<String, String> {
    let mut env = IndexMap::new();
    let helper_key = read_api_key();
    let api_key = process_env
        .get("ANTHROPIC_API_KEY")
        .cloned()
        .or_else(|| helper_key.clone());
    let auth_token = process_env
        .get("ANTHROPIC_AUTH_TOKEN")
        .cloned()
        .or(helper_key);
    let oauth_token = process_env.get("CLAUDE_CODE_OAUTH_TOKEN");
    if let Some(api_key) = api_key.filter(|value| !value.is_empty()) {
        env.insert("ANTHROPIC_API_KEY".into(), api_key);
    }
    if let Some(auth_token) = auth_token.filter(|value| !value.is_empty()) {
        env.insert("ANTHROPIC_AUTH_TOKEN".into(), auth_token);
    }
    if let Some(oauth_token) = oauth_token.filter(|value| !value.is_empty()) {
        env.insert("CLAUDE_CODE_OAUTH_TOKEN".into(), oauth_token.clone());
    }
    if let Some(base_url) = process_env
        .get("ANTHROPIC_BASE_URL")
        .filter(|value| !value.is_empty())
    {
        env.insert("ANTHROPIC_BASE_URL".into(), base_url.clone());
    }
    env
}

/// How an `apiKeyHelper` shell run failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShellRunError {
    /// The shell could not be found (`ENOENT`).
    NotFound,
    /// Any other failure: a non-zero exit or another launch error.
    Failed,
}

/// Runs a helper command and returns its stdout.
pub(crate) type RunShell<'a> = &'a dyn Fn(&str) -> Result<String, ShellRunError>;

fn run_to_string(mut command: std::process::Command) -> Result<String, ShellRunError> {
    use std::process::Stdio;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let output = command.output().map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ShellRunError::NotFound
        } else {
            ShellRunError::Failed
        }
    })?;
    if !output.status.success() {
        return Err(ShellRunError::Failed);
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `execFileSync('sh', ['-c', command], ...)`.
pub(crate) fn run_sh(command: &str) -> Result<String, ShellRunError> {
    let mut sh = std::process::Command::new("sh");
    sh.arg("-c").arg(command);
    run_to_string(sh)
}

/// `execSync(command, ...)`: the system shell, `/bin/sh` on macOS and
/// Linux and `cmd.exe` on Windows.
pub(crate) fn run_system_shell(command: &str) -> Result<String, ShellRunError> {
    #[cfg(windows)]
    let shell = {
        let mut shell = std::process::Command::new(
            std::env::var("ComSpec").unwrap_or_else(|_| "cmd.exe".into()),
        );
        shell.args(["/d", "/s", "/c", command]);
        shell
    };
    #[cfg(not(windows))]
    let shell = {
        let mut shell = std::process::Command::new("/bin/sh");
        shell.arg("-c").arg(command);
        shell
    };
    run_to_string(shell)
}

/// [rust] `os.homedir()`.
pub(crate) fn host_home_directory() -> Option<PathBuf> {
    std::env::home_dir().filter(|path| !path.as_os_str().is_empty())
}

/// Read the `apiKeyHelper` setting from `~/.claude/settings.json` and run
/// it. The `claude` CLI uses this hook to fetch credentials from password
/// managers and similar tools; mirroring it here lets users with that setup
/// run the harness without having to set `ANTHROPIC_API_KEY` explicitly. It
/// preserves existing helper commands by using `sh` from PATH when
/// available. If `sh` cannot be found, it uses the system shell: `/bin/sh`
/// on macOS and Linux, `cmd.exe` on Windows. Only a missing shell triggers
/// fallback, so a helper that fails is never executed twice.
fn read_api_key_helper() -> Option<String> {
    read_api_key_helper_with(&host_home_directory()?, &run_sh, &run_system_shell)
}

/// [rust] [`read_api_key_helper`] over a home directory and shell runners,
/// the seam the TS tests reach by mocking `node:child_process`.
pub(crate) fn read_api_key_helper_with(
    home: &Path,
    sh: RunShell<'_>,
    system_shell: RunShell<'_>,
) -> Option<String> {
    let raw = std::fs::read_to_string(home.join(".claude").join("settings.json")).ok()?;
    let settings: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let command = settings.get("apiKeyHelper")?.as_str()?;
    if command.is_empty() {
        return None;
    }
    let output = match sh(command) {
        Ok(output) => output,
        Err(ShellRunError::NotFound) => system_shell(command).ok()?,
        Err(ShellRunError::Failed) => return None,
    };
    let trimmed = output.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn pick_gateway(gateway_auth_from_env: AiGatewayAuth) -> IndexMap<String, String> {
    let mut env = IndexMap::new();
    if let Some(api_key) = gateway_auth_from_env.api_key {
        env.insert("AI_GATEWAY_API_KEY".into(), api_key.clone());
        env.insert("ANTHROPIC_API_KEY".into(), api_key);
    }
    env.insert(
        "AI_GATEWAY_BASE_URL".into(),
        gateway_auth_from_env.base_url.clone(),
    );
    env.insert("ANTHROPIC_BASE_URL".into(), gateway_auth_from_env.base_url);
    env
}
