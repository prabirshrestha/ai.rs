use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use tokio::process::Command;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::harness::types::{
    ExecutionError, ExecutionErrorCode, ExecutionResult, FileError, FileErrorCode, FileKind,
    FileMetadata, FileResult,
};

pub type ExecChunkCallback = Arc<dyn Fn(&str) -> ExecutionResult<()> + Send + Sync>;

#[derive(Clone, Default)]
pub struct ExecOptions {
    pub cwd: Option<PathBuf>,
    pub env: HashMap<String, String>,
    pub timeout: Option<Duration>,
    pub cancellation_token: Option<CancellationToken>,
    pub on_stdout: Option<ExecChunkCallback>,
    pub on_stderr: Option<ExecChunkCallback>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

#[derive(Debug, Clone)]
pub struct NodeExecutionEnv {
    pub cwd: PathBuf,
    shell_path: Option<PathBuf>,
    shell_env: HashMap<String, String>,
}

impl NodeExecutionEnv {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            shell_path: None,
            shell_env: HashMap::new(),
        }
    }

    pub fn with_shell_path(mut self, shell_path: impl Into<PathBuf>) -> Self {
        self.shell_path = Some(shell_path.into());
        self
    }

    pub fn with_shell_env(mut self, shell_env: HashMap<String, String>) -> Self {
        self.shell_env = shell_env;
        self
    }

    pub fn absolute_path(&self, path: impl AsRef<Path>) -> FileResult<PathBuf> {
        Ok(resolve_path(&self.cwd, path.as_ref()))
    }

    pub fn join_path(&self, parts: &[impl AsRef<Path>]) -> FileResult<PathBuf> {
        let mut joined = PathBuf::new();
        for part in parts {
            joined.push(part.as_ref());
        }
        Ok(joined)
    }

    pub async fn exec(&self, command: &str, options: ExecOptions) -> ExecutionResult<ExecOutput> {
        if options
            .cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(ExecutionError::new(ExecutionErrorCode::Aborted, "aborted"));
        }

        let shell_config = self.get_shell_config().await?;
        let cwd = options
            .cwd
            .as_deref()
            .map(|cwd| resolve_path(&self.cwd, cwd))
            .unwrap_or_else(|| self.cwd.clone());
        let mut cmd = Command::new(shell_config.shell);
        cmd.args(shell_config.args)
            .arg(command)
            .current_dir(cwd)
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        for (key, value) in &self.shell_env {
            cmd.env(key, value);
        }
        for (key, value) in &options.env {
            cmd.env(key, value);
        }

        let run = cmd.output();
        let output = if let Some(timeout) = options.timeout {
            match tokio::time::timeout(timeout, run).await {
                Ok(output) => output,
                Err(_) => {
                    return Err(ExecutionError::new(
                        ExecutionErrorCode::Timeout,
                        format!("timeout:{}", timeout.as_secs_f64()),
                    ));
                }
            }
        } else if let Some(token) = options.cancellation_token.clone() {
            tokio::select! {
                output = run => output,
                _ = token.cancelled() => return Err(ExecutionError::new(ExecutionErrorCode::Aborted, "aborted")),
            }
        } else {
            run.await
        }
        .map_err(|err| ExecutionError::new(ExecutionErrorCode::SpawnError, err.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if let Some(callback) = options.on_stdout.as_ref()
            && !stdout.is_empty()
        {
            callback(&stdout).map_err(|err| {
                ExecutionError::new(ExecutionErrorCode::CallbackError, err.message().to_string())
            })?;
        }
        if let Some(callback) = options.on_stderr.as_ref()
            && !stderr.is_empty()
        {
            callback(&stderr).map_err(|err| {
                ExecutionError::new(ExecutionErrorCode::CallbackError, err.message().to_string())
            })?;
        }

        Ok(ExecOutput {
            stdout,
            stderr,
            exit_code: output.status.code().unwrap_or_default(),
        })
    }

    pub async fn read_text_file(&self, path: impl AsRef<Path>) -> FileResult<String> {
        let resolved = resolve_path(&self.cwd, path.as_ref());
        tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|err| to_file_error(err, Some(resolved)))
    }

    pub async fn read_text_lines(
        &self,
        path: impl AsRef<Path>,
        max_lines: Option<usize>,
    ) -> FileResult<Vec<String>> {
        if max_lines == Some(0) {
            return Ok(Vec::new());
        }
        let text = self.read_text_file(path).await?;
        Ok(text
            .lines()
            .take(max_lines.unwrap_or(usize::MAX))
            .map(ToString::to_string)
            .collect())
    }

    pub async fn read_binary_file(&self, path: impl AsRef<Path>) -> FileResult<Vec<u8>> {
        let resolved = resolve_path(&self.cwd, path.as_ref());
        tokio::fs::read(&resolved)
            .await
            .map_err(|err| to_file_error(err, Some(resolved)))
    }

    pub async fn write_file(
        &self,
        path: impl AsRef<Path>,
        content: impl AsRef<[u8]>,
    ) -> FileResult<()> {
        let resolved = resolve_path(&self.cwd, path.as_ref());
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|err| to_file_error(err, Some(parent)))?;
        }
        tokio::fs::write(&resolved, content)
            .await
            .map_err(|err| to_file_error(err, Some(resolved)))
    }

    pub async fn append_file(
        &self,
        path: impl AsRef<Path>,
        content: impl AsRef<[u8]>,
    ) -> FileResult<()> {
        use tokio::io::AsyncWriteExt;

        let resolved = resolve_path(&self.cwd, path.as_ref());
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|err| to_file_error(err, Some(parent)))?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&resolved)
            .await
            .map_err(|err| to_file_error(err, Some(resolved.clone())))?;
        file.write_all(content.as_ref())
            .await
            .map_err(|err| to_file_error(err, Some(resolved)))
    }

    pub async fn file_info(&self, path: impl AsRef<Path>) -> FileResult<FileMetadata> {
        let resolved = resolve_path(&self.cwd, path.as_ref());
        let metadata = tokio::fs::symlink_metadata(&resolved)
            .await
            .map_err(|err| to_file_error(err, Some(resolved.clone())))?;
        file_info_from_metadata(&resolved, metadata)
    }

    pub async fn list_dir(&self, path: impl AsRef<Path>) -> FileResult<Vec<FileMetadata>> {
        let resolved = resolve_path(&self.cwd, path.as_ref());
        let mut entries = tokio::fs::read_dir(&resolved)
            .await
            .map_err(|err| to_file_error(err, Some(resolved.clone())))?;
        let mut infos = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|err| to_file_error(err, Some(resolved.clone())))?
        {
            let path = entry.path();
            let metadata = tokio::fs::symlink_metadata(&path)
                .await
                .map_err(|err| to_file_error(err, Some(path.clone())))?;
            infos.push(file_info_from_metadata(&path, metadata)?);
        }
        Ok(infos)
    }

    pub async fn canonical_path(&self, path: impl AsRef<Path>) -> FileResult<PathBuf> {
        let resolved = resolve_path(&self.cwd, path.as_ref());
        tokio::fs::canonicalize(&resolved)
            .await
            .map_err(|err| to_file_error(err, Some(resolved)))
    }

    pub async fn exists(&self, path: impl AsRef<Path>) -> FileResult<bool> {
        match self.file_info(path).await {
            Ok(_) => Ok(true),
            Err(err) if err.code == FileErrorCode::NotFound => Ok(false),
            Err(err) => Err(err),
        }
    }

    pub async fn create_dir(&self, path: impl AsRef<Path>, recursive: bool) -> FileResult<()> {
        let resolved = resolve_path(&self.cwd, path.as_ref());
        if recursive {
            tokio::fs::create_dir_all(&resolved)
                .await
                .map_err(|err| to_file_error(err, Some(resolved)))
        } else {
            tokio::fs::create_dir(&resolved)
                .await
                .map_err(|err| to_file_error(err, Some(resolved)))
        }
    }

    pub async fn remove(
        &self,
        path: impl AsRef<Path>,
        recursive: bool,
        force: bool,
    ) -> FileResult<()> {
        let resolved = resolve_path(&self.cwd, path.as_ref());
        let result = if recursive {
            tokio::fs::remove_dir_all(&resolved).await
        } else {
            match tokio::fs::symlink_metadata(&resolved).await {
                Ok(metadata) if metadata.is_dir() => tokio::fs::remove_dir(&resolved).await,
                Ok(_) => tokio::fs::remove_file(&resolved).await,
                Err(err) => Err(err),
            }
        };
        match result {
            Ok(()) => Ok(()),
            Err(err) if force && err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(to_file_error(err, Some(resolved))),
        }
    }

    pub async fn create_temp_dir(&self, prefix: Option<&str>) -> FileResult<PathBuf> {
        let path = std::env::temp_dir().join(format!(
            "{}{}",
            prefix.unwrap_or("tmp-"),
            crate::harness::session::create_session_id()
        ));
        tokio::fs::create_dir_all(&path)
            .await
            .map_err(|err| to_file_error(err, Some(path.clone())))?;
        Ok(path)
    }

    pub async fn create_temp_file(
        &self,
        prefix: Option<&str>,
        suffix: Option<&str>,
    ) -> FileResult<PathBuf> {
        let dir = self.create_temp_dir(Some("tmp-")).await?;
        let path = dir.join(format!(
            "{}{}{}",
            prefix.unwrap_or_default(),
            crate::harness::session::create_session_id(),
            suffix.unwrap_or_default()
        ));
        tokio::fs::write(&path, b"")
            .await
            .map_err(|err| to_file_error(err, Some(path.clone())))?;
        Ok(path)
    }

    pub async fn cleanup(&self) {}

    async fn get_shell_config(&self) -> ExecutionResult<ShellConfig> {
        if let Some(shell_path) = self.shell_path.as_ref() {
            if tokio::fs::metadata(shell_path).await.is_ok() {
                return Ok(ShellConfig {
                    shell: shell_path.clone(),
                    args: vec!["-c".to_string()],
                });
            }
            return Err(ExecutionError::new(
                ExecutionErrorCode::ShellUnavailable,
                format!("Custom shell path not found: {}", shell_path.display()),
            ));
        }

        if tokio::fs::metadata("/bin/bash").await.is_ok() {
            return Ok(ShellConfig {
                shell: PathBuf::from("/bin/bash"),
                args: vec!["-c".to_string()],
            });
        }
        Ok(ShellConfig {
            shell: PathBuf::from("sh"),
            args: vec!["-c".to_string()],
        })
    }
}

struct ShellConfig {
    shell: PathBuf,
    args: Vec<String>,
}

fn resolve_path(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn file_info_from_metadata(path: &Path, metadata: std::fs::Metadata) -> FileResult<FileMetadata> {
    let file_type = metadata.file_type();
    let kind = if file_type.is_file() {
        FileKind::File
    } else if file_type.is_dir() {
        FileKind::Directory
    } else if file_type.is_symlink() {
        FileKind::Symlink
    } else {
        return Err(FileError::new(
            FileErrorCode::Invalid,
            "Unsupported file type",
            Some(path.to_string_lossy().into_owned()),
        ));
    };
    let mtime_ms = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .unwrap_or_default();
    Ok(FileMetadata {
        name: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string(),
        path: path.to_string_lossy().into_owned(),
        kind,
        size: metadata.len(),
        mtime_ms,
    })
}

fn to_file_error(error: std::io::Error, path: Option<impl AsRef<Path>>) -> FileError {
    let path = path.map(|path| path.as_ref().to_string_lossy().into_owned());
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => FileErrorCode::NotFound,
        std::io::ErrorKind::PermissionDenied => FileErrorCode::PermissionDenied,
        std::io::ErrorKind::NotADirectory => FileErrorCode::NotDirectory,
        std::io::ErrorKind::IsADirectory => FileErrorCode::IsDirectory,
        std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData => {
            FileErrorCode::Invalid
        }
        _ => FileErrorCode::Unknown,
    };
    FileError::new(code, error.to_string(), path)
}
