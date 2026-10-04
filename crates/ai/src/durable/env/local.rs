//! Port of durable `src/env/node.ts`: [`LocalExecutionEnv`], the local
//! filesystem and shell over `tokio::fs` and `tokio::process`, behind the
//! `durable-local-env` feature.
//!
//! Divergences from Pi:
//! - `NodeExecutionEnv` is `LocalExecutionEnv`; its namespace id stays
//!   `"node:local"`, so local TS and Rust environments share one namespace.
//! - Paths follow POSIX semantics (`path.resolve`/`path.join` are lexical
//!   normalizations); Windows shells and `taskkill` are not ported, and shell
//!   lookup checks `/bin/bash`, then `bash` on `PATH`, then `sh`. The legacy WSL
//!   stdin transport is kept for a custom shell path that names one.
//! - Commands run in their own process group (Node's `detached`); abort,
//!   timeout and `cleanup` kill the group with `SIGKILL`.
//! - Output is read in one loop that writes the spill file inline, so a slow
//!   spill write backpressures the pipes directly. After the shell exits, the
//!   loop waits for both pipes to close, or for 100 ms without output (Pi's
//!   exit stdio grace).
//! - Error messages are Rust's `std::io::Error` text, not Node's
//!   `ENOENT: ..., lstat '...'` strings; codes and paths match.
//! - `readTextFile` decodes lossily, as Node's `utf8` decoding does.

use std::collections::HashSet;
use std::io::SeekFrom;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use ring::rand::{SecureRandom, SystemRandom};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::Instant;

use crate::chord::Context;

use super::{
    CreateDirOptions, ExecutionError, ExecutionErrorCode, FileError, FileErrorCode, FileInfo,
    FileKind, FileSystem, ReadTextLinesOptions, RemoveOptions, Shell, ShellExecOptions,
    ShellExecResult, TempFileOptions, TextLine, TextLineReader,
};

const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;
const MAX_TIMEOUT_SECONDS: f64 = MAX_TIMEOUT_MS / 1000.0;
const EXIT_STDIO_GRACE_MS: u64 = 100;
const READ_CHUNK_BYTES: usize = 64 * 1024;
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

fn resolve_timeout_ms(timeout: Option<f64>) -> Result<Option<f64>, ExecutionError> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    if !timeout.is_finite() || timeout <= 0.0 {
        return Err(ExecutionError::new(
            ExecutionErrorCode::Timeout,
            "Invalid timeout: must be a finite number of seconds",
        ));
    }
    let timeout_ms = timeout * 1000.0;
    if timeout_ms > MAX_TIMEOUT_MS {
        return Err(ExecutionError::new(
            ExecutionErrorCode::Timeout,
            format!("Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"),
        ));
    }
    Ok(Some(timeout_ms))
}

/// Lexical normalization shared by `resolve` and `join` (Node's `normalizeString`).
fn normalize(path: &str) -> String {
    let absolute = path.starts_with('/');
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                if segments.last().is_some_and(|last| *last != "..") {
                    segments.pop();
                } else if !absolute {
                    segments.push("..");
                }
            }
            segment => segments.push(segment),
        }
    }
    let joined = segments.join("/");
    if absolute {
        format!("/{joined}")
    } else if joined.is_empty() {
        ".".into()
    } else {
        joined
    }
}

/// Node's `path.join`.
fn join(parts: &[&str]) -> String {
    let joined = parts
        .iter()
        .filter(|part| !part.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join("/");
    if joined.is_empty() {
        return ".".into();
    }
    let trailing = joined.ends_with('/');
    let mut normalized = normalize(&joined);
    if trailing && normalized != "/" {
        normalized.push('/');
    }
    normalized
}

/// Node's `path.resolve(cwd, path)` for an already-joined pair.
fn resolve(cwd: &str, path: &str) -> String {
    if path.starts_with('/') {
        normalize(path)
    } else {
        normalize(&format!("{cwd}/{path}"))
    }
}

fn homedir() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/".into())
}

fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let hex = text.get(index + 1..index + 3)?;
            decoded.push(u8::from_str_radix(hex, 16).ok()?);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

/// Node's `fileURLToPath` for POSIX `file:` URLs; `None` for a malformed URL.
fn file_url_to_path(url: &str) -> Option<String> {
    let rest = url.strip_prefix("file://")?;
    let path = match rest.strip_prefix("localhost") {
        Some(path) if path.starts_with('/') => path,
        _ if rest.starts_with('/') => rest,
        _ => return None,
    };
    let path = path.split(['?', '#']).next().unwrap_or_default();
    // Encoded slashes are rejected, as in Node.
    if path.to_ascii_lowercase().contains("%2f") {
        return None;
    }
    percent_decode(path)
}

fn resolve_path(cwd: &str, path: &str) -> String {
    let mut normalized = path.to_owned();
    if normalized == "~" {
        normalized = homedir();
    } else if let Some(rest) = normalized.strip_prefix("~/") {
        normalized = join(&[&homedir(), rest]);
    } else if normalized.starts_with("file://") {
        // Keep malformed URLs as ordinary paths so filesystem methods preserve their non-throwing contract.
        if let Some(path) = file_url_to_path(&normalized) {
            normalized = path;
        }
    }
    resolve(cwd, &normalized)
}

fn basename(path: &str) -> String {
    path.trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .to_owned()
}

fn parent(path: &str) -> String {
    resolve(path, "..")
}

fn file_kind(metadata: &std::fs::Metadata) -> Option<FileKind> {
    let file_type = metadata.file_type();
    if file_type.is_file() {
        Some(FileKind::File)
    } else if file_type.is_dir() {
        Some(FileKind::Directory)
    } else if file_type.is_symlink() {
        Some(FileKind::Symlink)
    } else {
        None
    }
}

fn file_info_from_metadata(
    path: &str,
    metadata: &std::fs::Metadata,
) -> Result<FileInfo, FileError> {
    let kind = file_kind(metadata).ok_or_else(|| {
        FileError::new(
            FileErrorCode::Invalid,
            "Unsupported file type",
            Some(path.to_owned()),
        )
    })?;
    let mtime_ms = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0.0, |duration| duration.as_secs_f64() * 1000.0);
    Ok(FileInfo {
        name: basename(path),
        path: path.to_owned(),
        kind,
        size: metadata.len(),
        mtime_ms,
    })
}

fn to_file_error(error: std::io::Error, path: &str) -> FileError {
    use std::io::ErrorKind;
    let code = match error.kind() {
        ErrorKind::NotFound => FileErrorCode::NotFound,
        ErrorKind::PermissionDenied => FileErrorCode::PermissionDenied,
        ErrorKind::NotADirectory => FileErrorCode::NotDirectory,
        ErrorKind::IsADirectory => FileErrorCode::IsDirectory,
        ErrorKind::InvalidInput => FileErrorCode::Invalid,
        _ => match error.raw_os_error() {
            #[cfg(unix)]
            Some(libc::EPERM) => FileErrorCode::PermissionDenied,
            #[cfg(unix)]
            Some(libc::EINVAL) => FileErrorCode::Invalid,
            _ => FileErrorCode::Unknown,
        },
    };
    let message = error.to_string();
    FileError::with_cause(code, message, Some(path.to_owned()), error)
}

fn is_aborted(context: &Context) -> bool {
    context
        .abort_signal()
        .is_some_and(|signal| signal.aborted())
}

fn abort_result(context: &Context, path: Option<&str>) -> Result<(), FileError> {
    if is_aborted(context) {
        return Err(FileError::new(
            FileErrorCode::Aborted,
            "aborted",
            path.map(str::to_owned),
        ));
    }
    Ok(())
}

async fn path_exists(path: &str) -> bool {
    tokio::fs::metadata(path).await.is_ok()
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0; N];
    SystemRandom::new()
        .fill(&mut bytes)
        .expect("system randomness is available");
    bytes
}

/// `crypto.randomUUID()`.
fn random_uuid() -> String {
    let mut bytes: [u8; 16] = random_bytes();
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn kill_process_tree(pid: u32) {
    #[cfg(unix)]
    {
        let Ok(pid) = i32::try_from(pid) else {
            return;
        };
        // SAFETY: kill(2) has no memory-safety preconditions.
        unsafe {
            if libc::kill(-pid, libc::SIGKILL) != 0 {
                // Process already dead, or not a group leader.
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
    }
}

async fn find_bash_on_path() -> Option<String> {
    let output = tokio::time::timeout(
        Duration::from_millis(5000),
        Command::new("which")
            .arg("bash")
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first = stdout.trim().lines().next()?.to_owned();
    (!first.is_empty() && path_exists(&first).await).then_some(first)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShellConfig {
    shell: String,
    args: Vec<String>,
    command_from_stdin: bool,
}

fn is_legacy_wsl_bash_path(path: &str) -> bool {
    let normalized = path.replace('/', "\\").to_lowercase();
    let bytes = normalized.as_bytes();
    bytes.len() > 3
        && bytes[0].is_ascii_lowercase()
        && normalized[1..].starts_with(":\\windows\\")
        && matches!(
            &normalized[11..],
            "system32\\bash.exe" | "sysnative\\bash.exe"
        )
}

fn get_bash_shell_config(shell: &str) -> ShellConfig {
    if is_legacy_wsl_bash_path(shell) {
        ShellConfig {
            shell: shell.to_owned(),
            args: vec!["-s".into()],
            command_from_stdin: true,
        }
    } else {
        ShellConfig {
            shell: shell.to_owned(),
            args: vec!["-c".into()],
            command_from_stdin: false,
        }
    }
}

async fn get_shell_config(custom_shell_path: Option<&str>) -> Result<ShellConfig, ExecutionError> {
    if let Some(custom) = custom_shell_path.filter(|custom| !custom.is_empty()) {
        if path_exists(custom).await {
            return Ok(get_bash_shell_config(custom));
        }
        return Err(ExecutionError::new(
            ExecutionErrorCode::ShellUnavailable,
            format!("Custom shell path not found: {custom}"),
        ));
    }
    if path_exists("/bin/bash").await {
        return Ok(get_bash_shell_config("/bin/bash"));
    }
    if let Some(bash) = find_bash_on_path().await {
        return Ok(get_bash_shell_config(&bash));
    }
    Ok(ShellConfig {
        shell: "sh".into(),
        args: vec!["-c".into()],
        command_from_stdin: false,
    })
}

fn get_shell_env(
    base_env: Option<&IndexMap<String, String>>,
    extra_env: Option<&IndexMap<String, String>>,
    inherit_env: bool,
) -> IndexMap<String, String> {
    let mut env = IndexMap::new();
    if inherit_env {
        env.extend(std::env::vars());
        if let Some(base) = base_env {
            env.extend(base.iter().map(|(key, value)| (key.clone(), value.clone())));
        }
    }
    if let Some(extra) = extra_env {
        env.extend(
            extra
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    env
}

/// A streaming UTF-8 decoder with WHATWG replacement (`TextDecoder`).
#[derive(Debug, Default)]
pub(crate) struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    pub(crate) fn decode(&mut self, chunk: &[u8], stream: bool) -> String {
        self.pending.extend_from_slice(chunk);
        let mut output = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    output.push_str(text);
                    self.pending.clear();
                    return output;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    output.push_str(
                        std::str::from_utf8(&self.pending[..valid]).expect("valid prefix"),
                    );
                    match error.error_len() {
                        Some(length) => {
                            output.push('\u{fffd}');
                            self.pending.drain(..valid + length);
                        }
                        None if stream => {
                            self.pending.drain(..valid);
                            return output;
                        }
                        None => {
                            output.push('\u{fffd}');
                            self.pending.clear();
                            return output;
                        }
                    }
                }
            }
        }
    }
}

/// Strict LF reader; reports whether its final line was newline-terminated (`NodeTextLineReader`).
struct LocalTextLineReader {
    file: Option<tokio::fs::File>,
    path: String,
    decoder: Utf8Decoder,
    chunk: Vec<u8>,
    byte_offset: u64,
    buffered: String,
    ended: bool,
    closed: bool,
}

#[async_trait]
impl TextLineReader for LocalTextLineReader {
    async fn read_line(&mut self, context: &Context) -> Result<Option<TextLine>, FileError> {
        abort_result(context, Some(&self.path))?;
        if self.closed {
            return Err(FileError::new(
                FileErrorCode::Invalid,
                "Text line reader is closed",
                Some(self.path.clone()),
            ));
        }
        loop {
            if let Some(newline) = self.buffered.find('\n') {
                let text = self.buffered[..newline].to_owned();
                self.buffered.drain(..=newline);
                return Ok(Some(TextLine {
                    text,
                    terminated: true,
                }));
            }
            if self.ended {
                if self.buffered.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(TextLine {
                    text: std::mem::take(&mut self.buffered),
                    terminated: false,
                }));
            }

            // Explicit positions allow an aborted read to be retried without skipping bytes.
            let file = self.file.as_mut().expect("open while not closed");
            let read = async {
                file.seek(SeekFrom::Start(self.byte_offset)).await?;
                file.read(&mut self.chunk).await
            }
            .await
            .map_err(|error| to_file_error(error, &self.path))?;
            abort_result(context, Some(&self.path))?;
            self.byte_offset += read as u64;
            if read == 0 {
                let rest = self.decoder.decode(&[], false);
                self.buffered.push_str(&rest);
                self.ended = true;
            } else {
                let text = self.decoder.decode(&self.chunk[..read], true);
                self.buffered.push_str(&text);
            }
        }
    }

    async fn close(&mut self, _context: &Context) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.buffered.clear();
        // Closing is best-effort, including after cancellation or an earlier I/O failure.
        self.file.take();
    }
}

/// `NodeExecutionEnv` options.
#[derive(Debug, Clone, Default)]
pub struct LocalExecutionEnvOptions {
    pub cwd: String,
    pub shell_path: Option<String>,
    pub shell_env: Option<IndexMap<String, String>>,
}

/// The local filesystem and shell (`NodeExecutionEnv`).
pub struct LocalExecutionEnv {
    cwd: RwLock<String>,
    shell_path: Option<String>,
    shell_env: Option<IndexMap<String, String>>,
    active_child_pids: Mutex<HashSet<u32>>,
    #[cfg(test)]
    pub(crate) spill_path_override: Mutex<Option<String>>,
}

impl std::fmt::Debug for LocalExecutionEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalExecutionEnv")
            .field("cwd", &*self.cwd.read())
            .field("shell_path", &self.shell_path)
            .finish_non_exhaustive()
    }
}

impl LocalExecutionEnv {
    /// Every local environment sees the same files.
    pub const ID: &'static str = "node:local";

    pub fn new(options: LocalExecutionEnvOptions) -> Self {
        Self {
            cwd: RwLock::new(options.cwd),
            shell_path: options.shell_path,
            shell_env: options.shell_env,
            active_child_pids: Mutex::new(HashSet::new()),
            #[cfg(test)]
            spill_path_override: Mutex::new(None),
        }
    }

    /// An environment rooted at `cwd` with the default shell.
    pub fn at(cwd: impl Into<String>) -> Self {
        Self::new(LocalExecutionEnvOptions {
            cwd: cwd.into(),
            ..Default::default()
        })
    }

    /// `cleanup`: kill every active shell process group.
    pub async fn cleanup(&self, _context: &Context) {
        let pids: Vec<u32> = self.active_child_pids.lock().drain().collect();
        for pid in pids {
            kill_process_tree(pid);
        }
    }

    fn resolve(&self, path: &str) -> String {
        resolve_path(&self.cwd.read(), path)
    }

    async fn spill_file(&self, context: &Context) -> Result<String, FileError> {
        #[cfg(test)]
        if let Some(path) = self.spill_path_override.lock().clone() {
            return Ok(path);
        }
        self.create_temp_file(
            TempFileOptions {
                prefix: Some("pi-output-".into()),
                suffix: Some(".log".into()),
            },
            context,
        )
        .await
    }
}

/// Spill state of one command (`spillPrefix`, `spillPath`, the write stream).
struct Spill {
    prefix: Vec<Vec<u8>>,
    seen_bytes: u64,
    seen_newlines: u64,
    started: bool,
    path: Option<String>,
    file: Option<tokio::fs::File>,
    error: Option<ExecutionError>,
}

impl Spill {
    fn fail(&mut self, message: &str, cause: Arc<dyn std::error::Error + Send + Sync>) {
        if self.error.is_none() {
            self.error = Some(ExecutionError::with_cause(
                ExecutionErrorCode::Unknown,
                format!("Failed to preserve complete shell output: {message}"),
                cause,
            ));
        }
        self.file = None;
    }

    async fn write(&mut self, chunk: &[u8]) {
        let Some(file) = self.file.as_mut() else {
            return;
        };
        if chunk.is_empty() {
            return;
        }
        if let Err(error) = file.write_all(chunk).await {
            let message = error.to_string();
            self.fail(&message, Arc::new(error));
        }
    }
}

#[async_trait]
impl Shell for LocalExecutionEnv {
    async fn exec(
        &self,
        command: &str,
        options: ShellExecOptions,
        context: &Context,
    ) -> Result<ShellExecResult, ExecutionError> {
        let signal = context.abort_signal().cloned();
        if signal.as_ref().is_some_and(|signal| signal.aborted()) {
            return Err(ExecutionError::new(ExecutionErrorCode::Aborted, "aborted"));
        }
        let timeout_ms = resolve_timeout_ms(options.timeout)?;

        let base_cwd = self.cwd.read().clone();
        let cwd = match options.cwd.as_deref().filter(|cwd| !cwd.is_empty()) {
            Some(cwd) => resolve_path(&base_cwd, cwd),
            None => base_cwd,
        };
        let shell_config = get_shell_config(self.shell_path.as_deref()).await?;
        if let Err(error) = tokio::fs::metadata(&cwd).await {
            return Err(ExecutionError::with_cause(
                ExecutionErrorCode::SpawnError,
                format!("Working directory does not exist: {cwd}\nCannot execute bash commands."),
                Arc::new(error),
            ));
        }

        let mut child_command = Command::new(&shell_config.shell);
        child_command.args(&shell_config.args);
        if !shell_config.command_from_stdin {
            child_command.arg(command);
        }
        child_command
            .current_dir(&cwd)
            .env_clear()
            .envs(get_shell_env(
                self.shell_env.as_ref(),
                options.env.as_ref(),
                options.inherit_env.unwrap_or(true),
            ))
            .stdin(if shell_config.command_from_stdin {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        child_command.process_group(0);
        let mut child = match child_command.spawn() {
            Ok(child) => child,
            Err(error) => {
                return Err(ExecutionError::with_cause(
                    ExecutionErrorCode::SpawnError,
                    error.to_string(),
                    Arc::new(error),
                ));
            }
        };
        let pid = child.id();
        if let Some(pid) = pid {
            self.active_child_pids.lock().insert(pid);
        }
        if shell_config.command_from_stdin
            && let Some(mut stdin) = child.stdin.take()
        {
            let command = command.to_owned();
            tokio::spawn(async move {
                let _ = stdin.write_all(command.as_bytes()).await;
            });
        }
        let kill = || {
            if let Some(pid) = pid {
                kill_process_tree(pid);
            }
        };

        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        // One decoder per stream, so a character split across chunks of one stream survives interleaving.
        let mut decoders = [Utf8Decoder::default(), Utf8Decoder::default()];
        let mut stdout_buffer = vec![0; READ_CHUNK_BYTES];
        let mut stderr_buffer = vec![0; READ_CHUNK_BYTES];
        let mut callback_error: Option<ExecutionError> = None;
        let mut timed_out = false;
        let mut abort_seen = false;
        let mut exit_status: Option<std::process::ExitStatus> = None;
        let mut post_exit_deadline: Option<Instant> = None;
        let mut spill = Spill {
            prefix: Vec::new(),
            seen_bytes: 0,
            seen_newlines: 0,
            started: false,
            path: None,
            file: None,
            error: None,
        };
        let deadline = timeout_ms.map(|ms| Instant::now() + Duration::from_secs_f64(ms / 1000.0));
        let timeout = async {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending().await,
            }
        };
        tokio::pin!(timeout);
        let aborted = async {
            match &signal {
                Some(signal) => signal.cancelled().await,
                None => std::future::pending().await,
            }
        };
        tokio::pin!(aborted);

        // No output reaches the caller after exec() settled, for example from a descendant holding stdio open.
        let emit = |text: &str, callback_error: &mut Option<ExecutionError>| {
            if text.is_empty() || callback_error.is_some() {
                return;
            }
            let Some(on_output) = &options.on_output else {
                return;
            };
            if let Err(error) = on_output(text, context) {
                *callback_error = Some(ExecutionError::with_cause(
                    ExecutionErrorCode::CallbackError,
                    error.to_string(),
                    Arc::new(error),
                ));
                kill();
            }
        };

        let wait_error = loop {
            if exit_status.is_some() && stdout.is_none() && stderr.is_none() {
                break None;
            }
            let (chunk, stream) = tokio::select! {
                read = async { stdout.as_mut().expect("guarded").read(&mut stdout_buffer).await }, if stdout.is_some() => {
                    (read, 0)
                }
                read = async { stderr.as_mut().expect("guarded").read(&mut stderr_buffer).await }, if stderr.is_some() => {
                    (read, 1)
                }
                status = child.wait(), if exit_status.is_none() => {
                    match status {
                        Ok(status) => {
                            exit_status = Some(status);
                            post_exit_deadline = Some(Instant::now() + Duration::from_millis(EXIT_STDIO_GRACE_MS));
                            continue;
                        }
                        Err(error) => break Some(error),
                    }
                }
                () = &mut timeout, if !timed_out => {
                    timed_out = true;
                    kill();
                    continue;
                }
                () = &mut aborted, if !abort_seen => {
                    abort_seen = true;
                    kill();
                    continue;
                }
                () = async { tokio::time::sleep_until(post_exit_deadline.expect("guarded")).await }, if post_exit_deadline.is_some() => {
                    break None;
                }
            };
            let read = chunk.unwrap_or(0);
            if read == 0 {
                if stream == 0 {
                    stdout = None;
                } else {
                    stderr = None;
                }
                continue;
            }
            if exit_status.is_some() {
                post_exit_deadline =
                    Some(Instant::now() + Duration::from_millis(EXIT_STDIO_GRACE_MS));
            }
            let chunk = if stream == 0 {
                &stdout_buffer[..read]
            } else {
                &stderr_buffer[..read]
            };
            let text = decoders[stream].decode(chunk, true);
            emit(&text, &mut callback_error);
            let Some(spill_options) = options.spill else {
                continue;
            };
            if spill.error.is_some() {
                continue;
            }
            if spill.started {
                spill.write(chunk).await;
            } else {
                spill.seen_bytes += chunk.len() as u64;
                spill.seen_newlines += chunk.iter().filter(|&&byte| byte == b'\n').count() as u64;
                let lines = spill.seen_newlines + u64::from(chunk.last() != Some(&b'\n'));
                if spill.seen_bytes <= spill_options.after_bytes
                    && lines <= spill_options.after_lines
                {
                    spill.prefix.push(chunk.to_vec());
                    continue;
                }
                spill.started = true;
                match self.spill_file(context).await {
                    Ok(path) => {
                        spill.path = Some(path.clone());
                        match tokio::fs::OpenOptions::new()
                            .append(true)
                            .create(true)
                            .open(&path)
                            .await
                        {
                            Ok(file) => {
                                spill.file = Some(file);
                                for prefix in std::mem::take(&mut spill.prefix) {
                                    spill.write(&prefix).await;
                                }
                                spill.write(chunk).await;
                            }
                            Err(error) => {
                                let message = error.to_string();
                                spill.fail(&message, Arc::new(error));
                            }
                        }
                    }
                    Err(error) => {
                        let message = error.message.clone();
                        spill.fail(&message, Arc::new(error));
                    }
                }
                if spill.error.is_some() {
                    kill();
                }
            }
        };

        if let Some(pid) = pid {
            self.active_child_pids.lock().remove(&pid);
        }
        if let Some(error) = wait_error {
            return Err(ExecutionError::with_cause(
                ExecutionErrorCode::SpawnError,
                error.to_string(),
                Arc::new(error),
            ));
        }
        if let Some(mut file) = spill.file.take()
            && let Err(error) = file.flush().await
        {
            let message = error.to_string();
            spill.fail(&message, Arc::new(error));
        }
        for decoder in &mut decoders {
            let rest = decoder.decode(&[], false);
            emit(&rest, &mut callback_error);
        }
        if let Some(error) = callback_error {
            return Err(error);
        }
        let interrupted = if timed_out {
            Some(ExecutionError::new(
                ExecutionErrorCode::Timeout,
                format!("timeout:{}", options.timeout.unwrap_or_default()),
            ))
        } else if signal.as_ref().is_some_and(|signal| signal.aborted()) {
            Some(ExecutionError::new(ExecutionErrorCode::Aborted, "aborted"))
        } else {
            None
        };
        if let Some(mut interrupted) = interrupted {
            interrupted.spill_path = spill.path;
            return Err(interrupted);
        }
        if let Some(error) = spill.error {
            return Err(error);
        }
        // A process killed by a signal (e.g. OOM killer) has no exit code; map it
        // to the conventional 128 + signal number so callers do not mistake it
        // for a successful exit.
        let exit_code = match exit_status {
            Some(status) => status.code().unwrap_or_else(|| {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    status.signal().map_or(1, |signal| 128 + signal)
                }
                #[cfg(not(unix))]
                {
                    1
                }
            }),
            // The pipes closed without an observed exit: reap the child now.
            None => match child.wait().await {
                Ok(status) => status.code().unwrap_or(1),
                Err(_) => 1,
            },
        };
        Ok(ShellExecResult {
            exit_code,
            spill_path: spill.path,
        })
    }

    async fn cleanup(&self, context: &Context) {
        LocalExecutionEnv::cleanup(self, context).await;
    }
}

#[async_trait]
impl FileSystem for LocalExecutionEnv {
    fn id(&self) -> &str {
        Self::ID
    }

    fn cwd(&self) -> String {
        self.cwd.read().clone()
    }

    fn set_cwd(&self, cwd: String) {
        *self.cwd.write() = cwd;
    }

    async fn absolute_path(&self, path: &str, _context: &Context) -> Result<String, FileError> {
        Ok(self.resolve(path))
    }

    async fn join_path(&self, parts: &[&str], _context: &Context) -> Result<String, FileError> {
        Ok(join(parts))
    }

    async fn read_text_file(&self, path: &str, context: &Context) -> Result<String, FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        let bytes = tokio::fs::read(&resolved)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        abort_result(context, Some(&resolved))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn open_text_line_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> Result<Box<dyn TextLineReader>, FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        let file = tokio::fs::File::open(&resolved)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        abort_result(context, Some(&resolved))?;
        Ok(Box::new(LocalTextLineReader {
            file: Some(file),
            path: resolved,
            decoder: Utf8Decoder::default(),
            chunk: vec![0; READ_CHUNK_BYTES],
            byte_offset: 0,
            buffered: String::new(),
            ended: false,
            closed: false,
        }))
    }

    async fn read_text_lines(
        &self,
        path: &str,
        options: ReadTextLinesOptions,
        context: &Context,
    ) -> Result<Vec<String>, FileError> {
        if options.max_lines == Some(0) {
            return Ok(Vec::new());
        }
        let mut reader = self.open_text_line_reader(path, context).await?;
        let mut lines = Vec::new();
        let result = async {
            while options.max_lines.is_none_or(|max| lines.len() < max) {
                match reader.read_line(context).await? {
                    Some(line) => lines.push(line.text),
                    None => break,
                }
            }
            Ok(())
        }
        .await;
        reader.close(context).await;
        result.map(|()| lines)
    }

    async fn read_binary_file(&self, path: &str, context: &Context) -> Result<Vec<u8>, FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        let bytes = tokio::fs::read(&resolved)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        abort_result(context, Some(&resolved))?;
        Ok(bytes)
    }

    async fn write_file(
        &self,
        path: &str,
        content: &[u8],
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        tokio::fs::create_dir_all(parent(&resolved))
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        abort_result(context, Some(&resolved))?;
        tokio::fs::write(&resolved, content)
            .await
            .map_err(|error| to_file_error(error, &resolved))
    }

    async fn append_file(
        &self,
        path: &str,
        content: &[u8],
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        tokio::fs::create_dir_all(parent(&resolved))
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        abort_result(context, Some(&resolved))?;
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&resolved)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        file.write_all(content)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        file.flush()
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        abort_result(context, Some(&resolved))
    }

    async fn truncate_file(
        &self,
        path: &str,
        size: u64,
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        if size > MAX_SAFE_INTEGER {
            return Err(FileError::new(
                FileErrorCode::Invalid,
                "File size must be a non-negative safe integer",
                Some(resolved),
            ));
        }
        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&resolved)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        file.set_len(size)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        abort_result(context, Some(&resolved))
    }

    async fn flush_file(&self, path: &str, context: &Context) -> Result<(), FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&resolved)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        file.sync_all()
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        abort_result(context, Some(&resolved))
    }

    async fn rename_file(
        &self,
        source_path: &str,
        destination_path: &str,
        context: &Context,
    ) -> Result<(), FileError> {
        let source = self.resolve(source_path);
        let destination = self.resolve(destination_path);
        abort_result(context, Some(&destination))?;
        tokio::fs::rename(&source, &destination)
            .await
            .map_err(|error| to_file_error(error, &source))
    }

    async fn file_info(&self, path: &str, context: &Context) -> Result<FileInfo, FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        let metadata = tokio::fs::symlink_metadata(&resolved)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        file_info_from_metadata(&resolved, &metadata)
    }

    async fn list_dir(&self, path: &str, context: &Context) -> Result<Vec<FileInfo>, FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        let mut entries = tokio::fs::read_dir(&resolved)
            .await
            .map_err(|error| to_file_error(error, &resolved))?;
        let mut infos = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| to_file_error(error, &resolved))?
        {
            abort_result(context, Some(&resolved))?;
            let entry_path = resolve(&resolved, &entry.file_name().to_string_lossy());
            let metadata = tokio::fs::symlink_metadata(&entry_path)
                .await
                .map_err(|error| to_file_error(error, &entry_path))?;
            if let Ok(info) = file_info_from_metadata(&entry_path, &metadata) {
                infos.push(info);
            }
        }
        Ok(infos)
    }

    async fn canonical_path(&self, path: &str, context: &Context) -> Result<String, FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        tokio::fs::canonicalize(&resolved)
            .await
            .map(|path| path.to_string_lossy().into_owned())
            .map_err(|error| to_file_error(error, &resolved))
    }

    async fn exists(&self, path: &str, context: &Context) -> Result<bool, FileError> {
        match self.file_info(path, context).await {
            Ok(_) => Ok(true),
            Err(error) if error.code == FileErrorCode::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    async fn create_dir(
        &self,
        path: &str,
        options: CreateDirOptions,
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        let created = if options.recursive.unwrap_or(true) {
            tokio::fs::create_dir_all(&resolved).await
        } else {
            tokio::fs::create_dir(&resolved).await
        };
        created.map_err(|error| to_file_error(error, &resolved))
    }

    async fn remove(
        &self,
        path: &str,
        options: RemoveOptions,
        context: &Context,
    ) -> Result<(), FileError> {
        let resolved = self.resolve(path);
        abort_result(context, Some(&resolved))?;
        let force = options.force.unwrap_or(false);
        let metadata = match tokio::fs::symlink_metadata(&resolved).await {
            Ok(metadata) => metadata,
            Err(error) if force && error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(to_file_error(error, &resolved)),
        };
        let removed = if metadata.is_dir() {
            if !options.recursive.unwrap_or(false) {
                return Err(FileError::new(
                    FileErrorCode::Unknown,
                    format!("Path is a directory: rm returned EISDIR (is a directory) {resolved}"),
                    Some(resolved),
                ));
            }
            tokio::fs::remove_dir_all(&resolved).await
        } else {
            tokio::fs::remove_file(&resolved).await
        };
        match removed {
            Err(error) if force && error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            removed => removed.map_err(|error| to_file_error(error, &resolved)),
        }
    }

    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        context: &Context,
    ) -> Result<String, FileError> {
        abort_result(context, None)?;
        const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        let base = std::env::temp_dir().to_string_lossy().into_owned();
        let prefix = join(&[&base, prefix.unwrap_or("tmp-")]);
        loop {
            let suffix: String = random_bytes::<6>()
                .iter()
                .map(|byte| CHARS[usize::from(*byte) % CHARS.len()] as char)
                .collect();
            let path = format!("{prefix}{suffix}");
            match tokio::fs::create_dir(&path).await {
                Ok(()) => return Ok(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(to_file_error(error, &path)),
            }
        }
    }

    async fn create_temp_file(
        &self,
        options: TempFileOptions,
        context: &Context,
    ) -> Result<String, FileError> {
        let dir = self.create_temp_dir(Some("tmp-"), context).await?;
        let file_path = join(&[
            &dir,
            &format!(
                "{}{}{}",
                options.prefix.unwrap_or_default(),
                random_uuid(),
                options.suffix.unwrap_or_default()
            ),
        ]);
        tokio::fs::write(&file_path, b"")
            .await
            .map_err(|error| to_file_error(error, &file_path))?;
        Ok(file_path)
    }

    async fn cleanup(&self, context: &Context) {
        LocalExecutionEnv::cleanup(self, context).await;
    }
}

#[cfg(test)]
mod tests;
