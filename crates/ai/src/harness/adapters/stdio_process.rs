//! [rust] An agent process in the sandbox that speaks newline-delimited JSON
//! over stdio (Claude Code's stream-json, Codex's app server, ACP agents).

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};

use futures::StreamExt as _;
use serde_json::Value;
use tokio::io::AsyncWriteExt as _;
use tokio::sync::{Mutex, mpsc};

use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::sandbox::{
    SandboxExitStatus, SandboxProcessControl, SandboxProcessOptions, SandboxSession,
    SandboxSpawnOptions, SandboxStdin,
};

const STDERR_TAIL_LINES: usize = 50;

/// A running JSON-lines process.
pub struct JsonLineProcess {
    stdin: Mutex<Option<SandboxStdin>>,
    control: Arc<dyn SandboxProcessControl>,
    stderr_tail: Arc<StdMutex<VecDeque<String>>>,
    label: String,
}

fn split_lines(buffer: &mut Vec<u8>, chunk: &[u8]) -> Vec<String> {
    buffer.extend_from_slice(chunk);
    let mut lines = Vec::new();
    while let Some(position) = buffer.iter().position(|&b| b == b'\n') {
        let line: Vec<u8> = buffer.drain(..=position).collect();
        let line = String::from_utf8_lossy(&line[..line.len() - 1])
            .trim_end_matches('\r')
            .to_string();
        lines.push(line);
    }
    lines
}

impl JsonLineProcess {
    /// Spawn `options` with stdin open. Each JSON line on stdout arrives on
    /// the receiver; it closes when stdout ends.
    pub async fn spawn(
        sandbox: &dyn SandboxSession,
        options: SandboxProcessOptions,
        label: impl Into<String>,
    ) -> HarnessResult<(Arc<Self>, mpsc::UnboundedReceiver<Value>)> {
        let label = label.into();
        let mut process = sandbox
            .spawn(SandboxSpawnOptions {
                process: options,
                stdin: Some(true),
            })
            .await?;
        let stdin = process
            .stdin
            .take()
            .ok_or_else(|| HarnessError::new(format!("The {label} process has no stdin.")))?;
        let stderr_tail: Arc<StdMutex<VecDeque<String>>> = Arc::default();
        let (sender, receiver) = mpsc::unbounded_channel();

        let mut stdout = std::mem::replace(&mut process.stdout, futures::stream::empty().boxed());
        {
            let stderr_tail = stderr_tail.clone();
            tokio::spawn(async move {
                let mut buffer = Vec::new();
                while let Some(chunk) = stdout.next().await {
                    let Ok(chunk) = chunk else { break };
                    for line in split_lines(&mut buffer, &chunk) {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<Value>(trimmed) {
                            Ok(value) => {
                                let _ = sender.send(value);
                            }
                            Err(_) => push_tail(&stderr_tail, line),
                        }
                    }
                }
                if !buffer.is_empty()
                    && let Ok(value) = serde_json::from_slice::<Value>(&buffer)
                {
                    let _ = sender.send(value);
                }
            });
        }
        let mut stderr = std::mem::replace(&mut process.stderr, futures::stream::empty().boxed());
        {
            let stderr_tail = stderr_tail.clone();
            tokio::spawn(async move {
                let mut buffer = Vec::new();
                while let Some(chunk) = stderr.next().await {
                    let Ok(chunk) = chunk else { break };
                    for line in split_lines(&mut buffer, &chunk) {
                        push_tail(&stderr_tail, line);
                    }
                }
                if !buffer.is_empty() {
                    push_tail(&stderr_tail, String::from_utf8_lossy(&buffer).into_owned());
                }
            });
        }
        Ok((
            Arc::new(Self {
                stdin: Mutex::new(Some(stdin)),
                control: process.control.clone(),
                stderr_tail,
                label,
            }),
            receiver,
        ))
    }

    /// Write one JSON line.
    pub async fn send(&self, value: &Value) -> HarnessResult<()> {
        let mut line = serde_json::to_vec(value)?;
        line.push(b'\n');
        let mut stdin = self.stdin.lock().await;
        let Some(stdin) = stdin.as_mut() else {
            return Err(
                HarnessError::new(format!("The {} process stdin is closed.", self.label)).into(),
            );
        };
        stdin.write_all(&line).await?;
        stdin.flush().await?;
        Ok(())
    }

    /// Close stdin, delivering EOF.
    pub async fn close_stdin(&self) {
        if let Some(mut stdin) = self.stdin.lock().await.take() {
            let _ = stdin.shutdown().await;
        }
    }

    pub async fn kill(&self) {
        let _ = self.control.kill().await;
    }

    pub async fn wait(&self) -> HarnessResult<SandboxExitStatus> {
        self.control.wait().await
    }

    /// The last lines of stderr (and of non-JSON stdout).
    pub fn stderr_tail(&self) -> String {
        self.stderr_tail
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

fn push_tail(tail: &StdMutex<VecDeque<String>>, line: String) {
    let mut tail = tail.lock().unwrap();
    tail.push_back(line);
    while tail.len() > STDERR_TAIL_LINES {
        tail.pop_front();
    }
}

/// The error for a process that ended early, with its stderr.
pub fn process_exit_error(process: &JsonLineProcess, what: &str) -> HarnessError {
    let tail = process.stderr_tail();
    if tail.is_empty() {
        HarnessError::new(format!("The {} process {what}.", process.label()))
    } else {
        HarnessError::new(format!("The {} process {what}.\n{tail}", process.label()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::adapters::local_sandbox::LocalSandbox;
    use serde_json::json;

    #[test]
    fn splits_lines_across_chunks() {
        let mut buffer = Vec::new();
        assert!(split_lines(&mut buffer, b"{\"a\"").is_empty());
        assert_eq!(
            split_lines(&mut buffer, b":1}\r\n{}\n"),
            ["{\"a\":1}", "{}"]
        );
        assert!(buffer.is_empty());
    }

    #[tokio::test]
    async fn exchanges_json_lines_with_a_process() {
        let sandbox = LocalSandbox::new("/");
        let (process, mut receiver) = JsonLineProcess::spawn(
            &sandbox,
            SandboxProcessOptions::new(
                "while read -r line; do echo \"$line\"; echo not-json; echo oops >&2; done",
            ),
            "echo",
        )
        .await
        .unwrap();
        process.send(&json!({ "hello": 1 })).await.unwrap();
        assert_eq!(receiver.recv().await.unwrap(), json!({ "hello": 1 }));
        process.close_stdin().await;
        assert_eq!(receiver.recv().await, None);
        process.wait().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(process.stderr_tail().contains("not-json"));
        assert!(process.stderr_tail().contains("oops"));
    }
}
