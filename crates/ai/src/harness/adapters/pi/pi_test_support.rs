//! [rust] Test sandboxes for the Pi adapter: a local `bash` sandbox standing
//! in for `just-bash` and the native-shell sandboxes of the TS tests.

#![allow(dead_code)]

use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt as _;
use parking_lot::Mutex;
use tokio::io::AsyncReadExt as _;

use crate::harness::error::HarnessResult;
use crate::harness::sandbox::*;

/// What a [`LocalSandbox`] was asked to do.
#[derive(Debug, Default, Clone)]
pub(crate) struct LocalSandboxCalls {
    pub runs: Vec<(String, Option<String>)>,
    pub binary_reads: Vec<String>,
    pub text_writes: Vec<(String, String)>,
}

/// A sandbox that runs `bash -c` and file IO on the local machine.
#[derive(Clone, Default)]
pub(crate) struct LocalSandbox {
    pub cwd: Option<String>,
    pub calls: Arc<Mutex<LocalSandboxCalls>>,
}

impl LocalSandbox {
    pub fn new(cwd: impl Into<String>) -> Self {
        Self {
            cwd: Some(cwd.into()),
            calls: Arc::default(),
        }
    }

    fn command(&self, options: &SandboxProcessOptions) -> tokio::process::Command {
        let mut command = tokio::process::Command::new("bash");
        command.arg("-c").arg(&options.command);
        if let Some(dir) = options.working_directory.as_ref().or(self.cwd.as_ref()) {
            command.current_dir(dir);
        }
        if let Some(env) = &options.env {
            command.envs(env);
        }
        command.kill_on_drop(true);
        command
    }
}

struct LocalProcess {
    child: tokio::sync::Mutex<tokio::process::Child>,
}

#[async_trait]
impl SandboxProcessControl for LocalProcess {
    async fn wait(&self) -> HarnessResult<SandboxExitStatus> {
        let status = self.child.lock().await.wait().await?;
        Ok(SandboxExitStatus {
            exit_code: status.code().unwrap_or(-1),
        })
    }

    async fn kill(&self) -> HarnessResult<()> {
        let _ = self.child.lock().await.start_kill();
        Ok(())
    }
}

fn reader_stream(reader: impl tokio::io::AsyncRead + Send + Unpin + 'static) -> ByteStream {
    futures::stream::unfold(reader, |mut reader| async move {
        let mut buffer = vec![0u8; 8192];
        match reader.read(&mut buffer).await {
            Ok(0) => None,
            Ok(n) => {
                buffer.truncate(n);
                Some((Ok(Bytes::from(buffer)), reader))
            }
            Err(error) => Some((Err(error.into()), reader)),
        }
    })
    .boxed()
}

#[async_trait]
impl SandboxSession for LocalSandbox {
    fn description(&self) -> String {
        "local bash".into()
    }

    async fn read_file(&self, o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
        Ok(self
            .read_binary_file(o)
            .await?
            .map(|bytes| futures::stream::iter([Ok(bytes)]).boxed()))
    }

    async fn read_binary_file(
        &self,
        o: SandboxReadBinaryFileOptions,
    ) -> HarnessResult<Option<Bytes>> {
        self.calls.lock().binary_reads.push(o.path.clone());
        match std::fs::read(&o.path) {
            Ok(bytes) => Ok(Some(Bytes::from(bytes))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    async fn read_text_file(&self, o: SandboxReadTextFileOptions) -> HarnessResult<Option<String>> {
        match std::fs::read(&o.path) {
            Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    async fn write_file(&self, o: SandboxWriteFileOptions) -> HarnessResult<()> {
        let mut content = Vec::new();
        let mut stream = o.content;
        while let Some(chunk) = stream.next().await {
            content.extend_from_slice(&chunk?);
        }
        self.write_binary_file(SandboxWriteBinaryFileOptions {
            path: o.path,
            content: Bytes::from(content),
            abort_signal: None,
        })
        .await
    }

    async fn write_binary_file(&self, o: SandboxWriteBinaryFileOptions) -> HarnessResult<()> {
        if let Some(parent) = std::path::Path::new(&o.path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&o.path, &o.content)?;
        Ok(())
    }

    async fn write_text_file(&self, o: SandboxWriteTextFileOptions) -> HarnessResult<()> {
        self.calls
            .lock()
            .text_writes
            .push((o.path.clone(), o.content.clone()));
        if let Some(parent) = std::path::Path::new(&o.path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&o.path, o.content)?;
        Ok(())
    }

    async fn stat(&self, o: SandboxStatOptions) -> HarnessResult<Option<SandboxFileStat>> {
        match std::fs::metadata(&o.path) {
            Ok(metadata) => Ok(Some(SandboxFileStat {
                r#type: if metadata.is_dir() {
                    SandboxFileType::Directory
                } else if metadata.is_file() {
                    SandboxFileType::File
                } else {
                    SandboxFileType::Other
                },
                size: metadata.len(),
            })),
            Err(_) => Ok(None),
        }
    }

    async fn list_directory(
        &self,
        o: SandboxListDirectoryOptions,
    ) -> HarnessResult<Option<Vec<SandboxDirectoryEntry>>> {
        let Ok(entries) = std::fs::read_dir(&o.path) else {
            return Ok(None);
        };
        Ok(Some(
            entries
                .flatten()
                .map(|entry| {
                    let file_type = entry.file_type().ok();
                    SandboxDirectoryEntry {
                        name: entry.file_name().to_string_lossy().into_owned(),
                        r#type: match file_type {
                            Some(t) if t.is_dir() => SandboxEntryType::Directory,
                            Some(t) if t.is_file() => SandboxEntryType::File,
                            Some(t) if t.is_symlink() => SandboxEntryType::Symlink,
                            _ => SandboxEntryType::Other,
                        },
                    }
                })
                .collect(),
        ))
    }

    async fn delete_file(&self, o: SandboxDeleteFileOptions) -> HarnessResult<bool> {
        Ok(std::fs::remove_file(&o.path).is_ok())
    }

    async fn spawn(&self, o: SandboxSpawnOptions) -> HarnessResult<SandboxProcess> {
        let mut command = self.command(&o.process);
        command
            .stdin(if o.stdin == Some(true) {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn()?;
        let stdout = reader_stream(child.stdout.take().expect("stdout"));
        let stderr = reader_stream(child.stderr.take().expect("stderr"));
        let stdin = child
            .stdin
            .take()
            .map(|stdin| Box::pin(stdin) as SandboxStdin);
        Ok(SandboxProcess {
            pid: child.id(),
            stdout,
            stderr,
            stdin,
            control: Arc::new(LocalProcess {
                child: tokio::sync::Mutex::new(child),
            }),
        })
    }

    async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
        self.calls.lock().runs.push((
            o.process.command.clone(),
            o.process.working_directory.clone(),
        ));
        let mut command = self.command(&o.process);
        command.stdin(Stdio::null());
        let output = command.output().await?;
        Ok(SandboxRunResult {
            exit_code: output.status.code().unwrap_or(1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            stdout_length: None,
            stderr_length: None,
        })
    }
}

/// `realpathSync.native(path)` as a string.
pub(crate) fn canonical(path: &std::path::Path) -> String {
    std::fs::canonicalize(path)
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

/// Whether the tests run as root (file modes do not deny root).
pub(crate) fn is_root() -> bool {
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim() == "0")
        .unwrap_or(false)
}
