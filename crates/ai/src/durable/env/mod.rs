//! Port of durable `src/env/index.ts`: the portable filesystem and shell
//! capabilities durable tools and storage run on. [`local`] (behind the
//! `durable-local-env` feature) is Pi's `env/node.ts`.
//!
//! Divergences from Pi:
//! - TS's `Result<TValue, TError>` (`{ ok, value } | { ok, error }`) is Rust's
//!   [`Result`]; `ok`/`err` are `Ok`/`Err`, [`get_or_throw`] converts the
//!   error into a durable [`Error`], and `getOrUndefined` is `Result::ok`.
//!   `toError` has no counterpart (Rust errors are already typed).
//! - `cwd` is a getter/setter pair. Optional option bags are `Default`
//!   structs passed by value (`Default::default()` is TS `undefined`).
//! - File contents are bytes (`&[u8]`); TS strings are written as UTF-8, so
//!   callers pass `text.as_bytes()`.
//! - Truncation sizes are `u64`: negative and fractional sizes are
//!   unrepresentable, sizes above `Number.MAX_SAFE_INTEGER` are still rejected.
//! - `FileSystem` and `Shell` both declare `cleanup`, as in TS; on a value of a
//!   type implementing both, name the trait (`FileSystem::cleanup(&env, ctx)`).
//!   [`local::LocalExecutionEnv`] also has an inherent `cleanup` doing both.
//! - `onOutput` callbacks return a `Result` instead of throwing.

#[cfg(feature = "durable-local-env")]
pub mod local;

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use indexmap::IndexMap;

use crate::chord::Context;
use crate::durable::errors::Error;

/// `getOrThrow`: the value, or the expected failure as a thrown durable [`Error`].
pub fn get_or_throw<T, E>(result: Result<T, E>) -> crate::durable::Result<T>
where
    E: std::error::Error + Send + Sync + 'static,
{
    result.map_err(Error::thrown)
}

/// `getOrUndefined`.
pub fn get_or_undefined<T, E>(result: Result<T, E>) -> Option<T> {
    result.ok()
}

/// `FileKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileKind {
    File,
    Directory,
    Symlink,
}

impl FileKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Directory => "directory",
            Self::Symlink => "symlink",
        }
    }
}

/// `FileErrorCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileErrorCode {
    Aborted,
    NotFound,
    PermissionDenied,
    NotDirectory,
    IsDirectory,
    Invalid,
    NotSupported,
    Unknown,
}

impl FileErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aborted => "aborted",
            Self::NotFound => "not_found",
            Self::PermissionDenied => "permission_denied",
            Self::NotDirectory => "not_directory",
            Self::IsDirectory => "is_directory",
            Self::Invalid => "invalid",
            Self::NotSupported => "not_supported",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for FileErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An expected filesystem failure (`FileError`, `name === "FileError"`).
#[derive(Debug, Clone)]
pub struct FileError {
    pub code: FileErrorCode,
    pub message: String,
    pub path: Option<String>,
    pub cause: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl FileError {
    pub fn new(code: FileErrorCode, message: impl Into<String>, path: Option<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path,
            cause: None,
        }
    }

    pub fn with_cause(
        code: FileErrorCode,
        message: impl Into<String>,
        path: Option<String>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            path,
            cause: Some(Arc::new(cause)),
        }
    }

    /// The JS `error.name`.
    pub fn name(&self) -> &'static str {
        "FileError"
    }
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_deref()
            .map(|cause| cause as &(dyn std::error::Error + 'static))
    }
}

/// `ExecutionErrorCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionErrorCode {
    Aborted,
    Timeout,
    ShellUnavailable,
    SpawnError,
    CallbackError,
    Unknown,
}

impl ExecutionErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aborted => "aborted",
            Self::Timeout => "timeout",
            Self::ShellUnavailable => "shell_unavailable",
            Self::SpawnError => "spawn_error",
            Self::CallbackError => "callback_error",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for ExecutionErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An expected shell failure (`ExecutionError`, `name === "ExecutionError"`).
#[derive(Debug, Clone)]
pub struct ExecutionError {
    pub code: ExecutionErrorCode,
    pub message: String,
    /// Spill file of a command that timed out or was aborted after its output crossed the spill thresholds.
    pub spill_path: Option<String>,
    pub cause: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl ExecutionError {
    pub fn new(code: ExecutionErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            spill_path: None,
            cause: None,
        }
    }

    pub fn with_cause(
        code: ExecutionErrorCode,
        message: impl Into<String>,
        cause: Arc<dyn std::error::Error + Send + Sync>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            spill_path: None,
            cause: Some(cause),
        }
    }

    /// The JS `error.name`.
    pub fn name(&self) -> &'static str {
        "ExecutionError"
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_deref()
            .map(|cause| cause as &(dyn std::error::Error + 'static))
    }
}

/// `FileInfo`.
#[derive(Debug, Clone, PartialEq)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub kind: FileKind,
    pub size: u64,
    pub mtime_ms: f64,
}

/// `TextLine`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextLine {
    pub text: String,
    pub terminated: bool,
}

/// `TextLineReader`.
#[async_trait]
pub trait TextLineReader: Send {
    async fn read_line(&mut self, context: &Context) -> Result<Option<TextLine>, FileError>;
    async fn close(&mut self, context: &Context);
}

/// `readTextLines` options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadTextLinesOptions {
    pub max_lines: Option<usize>,
}

/// `createDir` options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CreateDirOptions {
    pub recursive: Option<bool>,
}

/// `remove` options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemoveOptions {
    pub recursive: Option<bool>,
    pub force: Option<bool>,
}

/// `createTempFile` options.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TempFileOptions {
    pub prefix: Option<String>,
    pub suffix: Option<String>,
}

/// Portable filesystem capability (`FileSystem`). Operations return failures rather than throwing.
#[async_trait]
pub trait FileSystem: Send + Sync {
    /// The file namespace: equal ids see the same files at the same paths, whatever their `cwd`. Every local
    /// environment shares one id; each container or remote host has its own.
    fn id(&self) -> &str;
    fn cwd(&self) -> String;
    fn set_cwd(&self, cwd: String);
    async fn absolute_path(&self, path: &str, context: &Context) -> Result<String, FileError>;
    async fn join_path(&self, parts: &[&str], context: &Context) -> Result<String, FileError>;
    async fn read_text_file(&self, path: &str, context: &Context) -> Result<String, FileError>;
    async fn open_text_line_reader(
        &self,
        path: &str,
        context: &Context,
    ) -> Result<Box<dyn TextLineReader>, FileError>;
    async fn read_text_lines(
        &self,
        path: &str,
        options: ReadTextLinesOptions,
        context: &Context,
    ) -> Result<Vec<String>, FileError>;
    async fn read_binary_file(&self, path: &str, context: &Context) -> Result<Vec<u8>, FileError>;
    async fn write_file(
        &self,
        path: &str,
        content: &[u8],
        context: &Context,
    ) -> Result<(), FileError>;
    async fn append_file(
        &self,
        path: &str,
        content: &[u8],
        context: &Context,
    ) -> Result<(), FileError>;
    /// Truncate or extend a file to exactly `size` bytes.
    async fn truncate_file(
        &self,
        path: &str,
        size: u64,
        context: &Context,
    ) -> Result<(), FileError>;
    /// Flush file contents and metadata needed to retrieve them from an open file handle.
    async fn flush_file(&self, path: &str, context: &Context) -> Result<(), FileError>;
    async fn rename_file(
        &self,
        source_path: &str,
        destination_path: &str,
        context: &Context,
    ) -> Result<(), FileError>;
    async fn file_info(&self, path: &str, context: &Context) -> Result<FileInfo, FileError>;
    async fn list_dir(&self, path: &str, context: &Context) -> Result<Vec<FileInfo>, FileError>;
    async fn canonical_path(&self, path: &str, context: &Context) -> Result<String, FileError>;
    async fn exists(&self, path: &str, context: &Context) -> Result<bool, FileError>;
    async fn create_dir(
        &self,
        path: &str,
        options: CreateDirOptions,
        context: &Context,
    ) -> Result<(), FileError>;
    async fn remove(
        &self,
        path: &str,
        options: RemoveOptions,
        context: &Context,
    ) -> Result<(), FileError>;
    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        context: &Context,
    ) -> Result<String, FileError>;
    async fn create_temp_file(
        &self,
        options: TempFileOptions,
        context: &Context,
    ) -> Result<String, FileError>;
    async fn cleanup(&self, context: &Context);
}

/// Spill the complete output to a temporary file once it exceeds either threshold (`ShellSpillOptions`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShellSpillOptions {
    pub after_bytes: u64,
    /// Complete or partial lines.
    pub after_lines: u64,
}

/// `ShellExecResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellExecResult {
    pub exit_code: i32,
    /// Temporary file holding the complete raw output, when the spill thresholds were exceeded.
    pub spill_path: Option<String>,
}

/// Every decoded chunk of combined stdout and stderr as it arrives: raw, unbounded, and unthrottled.
pub type OnOutput = Arc<dyn Fn(&str, &Context) -> Result<(), Error> + Send + Sync>;

/// `ShellExecOptions`.
#[derive(Clone, Default)]
pub struct ShellExecOptions {
    pub cwd: Option<String>,
    pub env: Option<IndexMap<String, String>>,
    pub inherit_env: Option<bool>,
    /// Seconds.
    pub timeout: Option<f64>,
    pub on_output: Option<OnOutput>,
    pub spill: Option<ShellSpillOptions>,
}

impl fmt::Debug for ShellExecOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShellExecOptions")
            .field("cwd", &self.cwd)
            .field("env", &self.env)
            .field("inherit_env", &self.inherit_env)
            .field("timeout", &self.timeout)
            .field("on_output", &self.on_output.is_some())
            .field("spill", &self.spill)
            .finish()
    }
}

/// `Shell`.
#[async_trait]
pub trait Shell: Send + Sync {
    async fn exec(
        &self,
        command: &str,
        options: ShellExecOptions,
        context: &Context,
    ) -> Result<ShellExecResult, ExecutionError>;
    async fn cleanup(&self, context: &Context);
}

/// `ExecutionEnv`: a filesystem plus a shell.
pub trait ExecutionEnv: FileSystem + Shell {}

impl<T: FileSystem + Shell + ?Sized> ExecutionEnv for T {}
