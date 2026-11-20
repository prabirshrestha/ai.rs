//! [rust] A sandbox that is the host machine itself: commands run with
//! `bash -c` as this user, and files are read and written in place. It is
//! what a desktop or server app uses to run the harness CLIs a person has
//! installed and signed in to. It gives no isolation.

use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt as _;
use tokio::io::AsyncReadExt as _;

use crate::harness::error::HarnessResult;
use crate::harness::sandbox::*;

/// Runs commands and file operations on this machine.
#[derive(Debug, Clone, Default)]
pub struct HostSandbox {
    /// The directory commands run in when they name none.
    pub cwd: Option<String>,
    /// Variables set for every command, under each command's own.
    pub env: Vec<(String, String)>,
}

impl HostSandbox {
    pub fn new(cwd: impl Into<String>) -> Self {
        Self {
            cwd: Some(cwd.into()),
            env: Vec::new(),
        }
    }

    /// Set `name` for every command.
    pub fn with_env(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((name.into(), value.into()));
        self
    }

    fn command(&self, options: &SandboxProcessOptions) -> tokio::process::Command {
        let mut command = tokio::process::Command::new("bash");
        command.arg("-c").arg(&options.command);
        if let Some(dir) = options.working_directory.as_ref().or(self.cwd.as_ref()) {
            command.current_dir(dir);
        }
        command.envs(self.env.iter().cloned());
        if let Some(env) = &options.env {
            command.envs(env);
        }
        #[cfg(unix)]
        command.process_group(0);
        command.kill_on_drop(true);
        command
    }
}

struct LocalProcess {
    child: tokio::sync::Mutex<tokio::process::Child>,
    pid: Option<u32>,
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
        // The process leads its own group: stop the whole tree.
        #[cfg(unix)]
        if let Some(pid) = self.pid {
            let _ = std::process::Command::new("kill")
                .args(["-KILL", "--", &format!("-{pid}")])
                .stderr(Stdio::null())
                .status();
        }
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
impl SandboxSession for HostSandbox {
    fn description(&self) -> String {
        format!(
            "the host machine ({}), commands run with bash",
            std::env::consts::OS
        )
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
        let pid = child.id();
        let stdout = reader_stream(child.stdout.take().expect("stdout"));
        let stderr = reader_stream(child.stderr.take().expect("stderr"));
        let stdin = child
            .stdin
            .take()
            .map(|stdin| Box::pin(stdin) as SandboxStdin);
        Ok(SandboxProcess {
            pid,
            stdout,
            stderr,
            stdin,
            control: Arc::new(LocalProcess {
                child: tokio::sync::Mutex::new(child),
                pid,
            }),
        })
    }

    async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn runs_commands_and_files_on_the_host() {
        let dir = std::env::temp_dir().join(format!("host-sandbox-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dir = dir.to_string_lossy().into_owned();
        let sandbox = HostSandbox::new(dir.clone()).with_env("GREETING", "hi");
        let mut process = SandboxProcessOptions::new("printf '%s %s' \"$GREETING\" \"$(pwd)\"");
        process.env = None;
        let result = sandbox
            .run(SandboxRunOptions {
                process,
                max_output_length: None,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.starts_with("hi "));
        let path = format!("{dir}/a/b.txt");
        sandbox
            .write_text_file(SandboxWriteTextFileOptions {
                path: path.clone(),
                content: "text".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        let read = sandbox
            .read_text_file(SandboxReadTextFileOptions {
                path,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(read.as_deref(), Some("text"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
