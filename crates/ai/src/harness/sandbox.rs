//! Port of `@ai-sdk/provider-utils` `types/sandbox.ts`
//! (`Experimental_SandboxSession` and its option and result types).
//!
//! Relative paths in every method, including `working_directory`, resolve
//! against the directory that `run` and `spawn` start in.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::harness::error::{BoxError, HarnessResult};

/// `ReadableStream<Uint8Array>`.
pub type ByteStream = BoxStream<'static, Result<Bytes, BoxError>>;

/// `WritableStream<Uint8Array>`. Shutting it down delivers EOF.
pub type SandboxStdin = Pin<Box<dyn tokio::io::AsyncWrite + Send>>;

/// Options for executing a command via `run` or `spawn`.
#[derive(Debug, Clone, Default)]
pub struct SandboxProcessOptions {
    pub command: String,
    pub working_directory: Option<String>,
    /// Merged with the sandbox's default environment; these win.
    pub env: Option<IndexMap<String, String>>,
    /// Kills the process when cancelled.
    pub abort_signal: Option<CancellationToken>,
}

impl SandboxProcessOptions {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            ..Default::default()
        }
    }
}

/// `SandboxRunOptions`.
#[derive(Debug, Clone, Default)]
pub struct SandboxRunOptions {
    pub process: SandboxProcessOptions,
    /// Limits each returned stream to at most this many UTF-16 code units.
    pub max_output_length: Option<usize>,
}

/// `SandboxRunResult`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SandboxRunResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub stdout_length: Option<usize>,
    pub stderr_length: Option<usize>,
}

/// `SandboxSpawnOptions`.
#[derive(Debug, Clone, Default)]
pub struct SandboxSpawnOptions {
    pub process: SandboxProcessOptions,
    /// Keep stdin open and expose it on the returned process.
    pub stdin: Option<bool>,
}

/// `PathOptions` (read, stat, list, delete).
#[derive(Debug, Clone, Default)]
pub struct SandboxPathOptions {
    pub path: String,
    pub abort_signal: Option<CancellationToken>,
}

impl SandboxPathOptions {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            abort_signal: None,
        }
    }
}

/// `SandboxReadFileOptions`.
pub type SandboxReadFileOptions = SandboxPathOptions;
/// `SandboxReadBinaryFileOptions`.
pub type SandboxReadBinaryFileOptions = SandboxPathOptions;
/// `SandboxStatOptions`.
pub type SandboxStatOptions = SandboxPathOptions;
/// `SandboxListDirectoryOptions`.
pub type SandboxListDirectoryOptions = SandboxPathOptions;
/// `SandboxDeleteFileOptions`.
pub type SandboxDeleteFileOptions = SandboxPathOptions;

/// `SandboxReadTextFileOptions`.
#[derive(Debug, Clone, Default)]
pub struct SandboxReadTextFileOptions {
    pub path: String,
    pub abort_signal: Option<CancellationToken>,
    /// Defaults to `"utf-8"`.
    pub encoding: Option<String>,
    /// 1-based inclusive start line.
    pub start_line: Option<usize>,
    /// 1-based inclusive end line.
    pub end_line: Option<usize>,
}

/// `WriteFileOptions<CONTENT>`.
pub struct SandboxWriteFileOptionsOf<C> {
    pub path: String,
    pub content: C,
    pub abort_signal: Option<CancellationToken>,
}

impl<C> fmt::Debug for SandboxWriteFileOptionsOf<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxWriteFileOptions")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// `SandboxWriteFileOptions`.
pub type SandboxWriteFileOptions = SandboxWriteFileOptionsOf<ByteStream>;
/// `SandboxWriteBinaryFileOptions`.
pub type SandboxWriteBinaryFileOptions = SandboxWriteFileOptionsOf<Bytes>;

/// `SandboxWriteTextFileOptions`.
#[derive(Debug, Clone, Default)]
pub struct SandboxWriteTextFileOptions {
    pub path: String,
    pub content: String,
    pub abort_signal: Option<CancellationToken>,
    /// Defaults to `"utf-8"`.
    pub encoding: Option<String>,
}

/// `SandboxFileStat['type']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SandboxFileType {
    File,
    Directory,
    Other,
}

/// `SandboxFileStat`. Symbolic links are followed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxFileStat {
    pub r#type: SandboxFileType,
    pub size: u64,
}

/// `SandboxDirectoryEntry['type']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SandboxEntryType {
    File,
    Directory,
    Symlink,
    Other,
}

/// `SandboxDirectoryEntry`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxDirectoryEntry {
    pub name: String,
    pub r#type: SandboxEntryType,
}

/// `wait()` result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SandboxExitStatus {
    pub exit_code: i32,
}

/// The `wait` / `kill` half of a [`SandboxProcess`].
#[async_trait]
pub trait SandboxProcessControl: Send + Sync {
    /// Resolve when the process exits.
    async fn wait(&self) -> HarnessResult<SandboxExitStatus>;
    /// Terminate the process. Idempotent.
    async fn kill(&self) -> HarnessResult<()>;
}

/// `SandboxProcess` / `WritableSandboxProcess`.
///
/// [rust] TS selects one of two types through `spawn` overloads. Here
/// `stdin` is `Some` exactly when spawned with `stdin: Some(true)`.
pub struct SandboxProcess {
    pub pid: Option<u32>,
    pub stdout: ByteStream,
    pub stderr: ByteStream,
    pub stdin: Option<SandboxStdin>,
    pub control: Arc<dyn SandboxProcessControl>,
}

impl fmt::Debug for SandboxProcess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxProcess")
            .field("pid", &self.pid)
            .field("stdin", &self.stdin.is_some())
            .finish_non_exhaustive()
    }
}

impl SandboxProcess {
    /// `wait()`.
    pub async fn wait(&self) -> HarnessResult<SandboxExitStatus> {
        self.control.wait().await
    }

    /// `kill()`.
    pub async fn kill(&self) -> HarnessResult<()> {
        self.control.kill().await
    }
}

/// `Experimental_SandboxSession`.
#[async_trait]
pub trait SandboxSession: Send + Sync {
    /// Description of the environment, for the agent's instructions.
    fn description(&self) -> String;

    /// Read one file as a byte stream. `None` when it does not exist.
    async fn read_file(&self, options: SandboxReadFileOptions)
    -> HarnessResult<Option<ByteStream>>;

    /// Read one file as bytes. `None` when it does not exist.
    async fn read_binary_file(
        &self,
        options: SandboxReadBinaryFileOptions,
    ) -> HarnessResult<Option<Bytes>>;

    /// Read one text file. `None` when it does not exist.
    async fn read_text_file(
        &self,
        options: SandboxReadTextFileOptions,
    ) -> HarnessResult<Option<String>>;

    /// Write one file from a byte stream, creating parent directories.
    async fn write_file(&self, options: SandboxWriteFileOptions) -> HarnessResult<()>;

    /// Write one file from bytes, creating parent directories.
    async fn write_binary_file(&self, options: SandboxWriteBinaryFileOptions) -> HarnessResult<()>;

    /// Write one file from a string, creating parent directories.
    async fn write_text_file(&self, options: SandboxWriteTextFileOptions) -> HarnessResult<()>;

    /// Stat a path, following symlinks. `None` when it does not exist.
    async fn stat(&self, options: SandboxStatOptions) -> HarnessResult<Option<SandboxFileStat>>;

    /// List one directory. `None` when it does not exist.
    async fn list_directory(
        &self,
        options: SandboxListDirectoryOptions,
    ) -> HarnessResult<Option<Vec<SandboxDirectoryEntry>>>;

    /// Delete one file. `false` when it does not exist.
    async fn delete_file(&self, options: SandboxDeleteFileOptions) -> HarnessResult<bool>;

    /// Spawn a long-running process.
    async fn spawn(&self, options: SandboxSpawnOptions) -> HarnessResult<SandboxProcess>;

    /// Run a command to completion.
    async fn run(&self, options: SandboxRunOptions) -> HarnessResult<SandboxRunResult>;
}
