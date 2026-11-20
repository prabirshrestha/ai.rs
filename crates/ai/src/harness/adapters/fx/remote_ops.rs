//! Port of `harness-fx/src/fx-remote-ops.ts`: fx's builtin tools, run in the
//! sandbox and confined to the session's workspace.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::glob::matches_glob;
use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::paths::resolve_posix;
use crate::harness::sandbox::{
    SandboxProcessOptions, SandboxReadTextFileOptions, SandboxRunOptions, SandboxRunResult,
    SandboxSession, SandboxWriteTextFileOptions,
};
use crate::harness::utils::shell_quote;

/// `path.posix.relative(from, to)` for absolute paths.
fn relative(from: &str, to: &str) -> String {
    let from: Vec<&str> = from.split('/').filter(|part| !part.is_empty()).collect();
    let to: Vec<&str> = to.split('/').filter(|part| !part.is_empty()).collect();
    let common = from
        .iter()
        .zip(&to)
        .take_while(|(left, right)| left == right)
        .count();
    let mut parts = vec![".."; from.len() - common];
    parts.extend(&to[common..]);
    parts.join("/")
}

/// `assertContained({ root, candidate })`.
fn assert_contained(root: &str, candidate: &str) -> HarnessResult<String> {
    let relative = relative(root, candidate);
    if relative == ".." || relative.starts_with("../") || relative.starts_with('/') {
        return Err(
            HarnessError::new(format!("fx path escapes the workspace: {candidate}")).into(),
        );
    }
    Ok(candidate.to_string())
}

#[derive(Deserialize)]
struct PathInput {
    file_path: String,
}

#[derive(Deserialize)]
struct WriteInput {
    file_path: String,
    content: String,
}

#[derive(Deserialize)]
struct EditInput {
    file_path: String,
    old_string: String,
    new_string: String,
}

#[derive(Deserialize)]
struct BashInput {
    command: String,
    timeout: Option<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GrepInput {
    pattern: String,
    path: Option<String>,
    glob: Option<String>,
    ignore_case: Option<bool>,
    literal: Option<bool>,
    context: Option<f64>,
    limit: Option<f64>,
}

#[derive(Deserialize)]
struct GlobInput {
    pattern: String,
    path: Option<String>,
    limit: Option<f64>,
}

#[derive(Deserialize)]
struct LsInput {
    path: Option<String>,
    limit: Option<f64>,
}

fn parse<T: for<'de> Deserialize<'de>>(input: Value) -> HarnessResult<T> {
    serde_json::from_value(input)
        .map_err(|error| HarnessError::new(format!("Invalid tool input: {error}")).into())
}

/// `createFxRemoteOps({ sandbox, workDir })`.
#[derive(Clone)]
pub(crate) struct FxRemoteOps {
    sandbox: Arc<dyn SandboxSession>,
    work_dir: String,
    canonical_root: Arc<OnceCell<String>>,
}

impl FxRemoteOps {
    pub fn new(sandbox: Arc<dyn SandboxSession>, work_dir: &str) -> Self {
        Self {
            sandbox,
            work_dir: resolve_posix("/", work_dir),
            canonical_root: Arc::default(),
        }
    }

    /// Runs operation `name` with the model's `input`.
    pub async fn call(
        &self,
        name: &str,
        input: Value,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<Value> {
        match name {
            "read" => Ok(Value::String(
                self.read(parse::<PathInput>(input)?.file_path, signal)
                    .await?,
            )),
            "write" => {
                let input: WriteInput = parse(input)?;
                Ok(Value::String(
                    self.write(&input.file_path, input.content, signal).await?,
                ))
            }
            "edit" => {
                let input: EditInput = parse(input)?;
                Ok(Value::String(self.edit(input, signal).await?))
            }
            "bash" => {
                let input: BashInput = parse(input)?;
                let result = self.bash(input.command, input.timeout, signal).await?;
                Ok(json!({
                    "exitCode": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }))
            }
            "grep" => Ok(Value::String(self.grep(parse(input)?, signal).await?)),
            "glob" => Ok(json!(self.glob(parse(input)?, signal).await?)),
            "ls" => Ok(json!(self.ls(parse(input)?, signal).await?)),
            _ => Err(HarnessError::new(format!("Unsupported sandbox tool {name}")).into()),
        }
    }

    async fn run(
        &self,
        command: String,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<SandboxRunResult> {
        if signal.as_ref().is_some_and(CancellationToken::is_cancelled) {
            return Err(HarnessError::new("This operation was aborted").into());
        }
        self.sandbox
            .run(SandboxRunOptions {
                process: SandboxProcessOptions {
                    command,
                    working_directory: Some(self.work_dir.clone()),
                    env: None,
                    abort_signal: signal,
                },
                max_output_length: None,
            })
            .await
    }

    async fn realpath(
        &self,
        target: &str,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<String> {
        let inner = format!("realpath -mz -- {} | base64", shell_quote(target));
        let result = self
            .run(
                format!("bash -o pipefail -c {}", shell_quote(&inner)),
                signal,
            )
            .await?;
        let encoded: String = result.stdout.split_whitespace().collect();
        if result.exit_code != 0 || encoded.is_empty() {
            return Err(
                HarnessError::new(format!("Unable to resolve sandbox path: {target}")).into(),
            );
        }
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| HarnessError::new(format!("Unable to resolve sandbox path: {target}")))?;
        let decoded = String::from_utf8_lossy(&decoded).into_owned();
        Ok(decoded
            .strip_suffix('\0')
            .map(str::to_string)
            .unwrap_or(decoded))
    }

    /// `resolvePath({ filePath })`: the canonical path, inside the workspace.
    async fn resolve_path(
        &self,
        file_path: &str,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<String> {
        let candidate =
            assert_contained(&self.work_dir, &resolve_posix(&self.work_dir, file_path))?;
        let root = self
            .canonical_root
            .get_or_try_init(|| self.realpath(&self.work_dir, None))
            .await?
            .clone();
        assert_contained(&root, &self.realpath(&candidate, signal).await?)
    }

    pub async fn read(
        &self,
        file_path: String,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<String> {
        let path = self.resolve_path(&file_path, signal.clone()).await?;
        self.sandbox
            .read_text_file(SandboxReadTextFileOptions {
                path,
                abort_signal: signal,
                ..Default::default()
            })
            .await?
            .ok_or_else(|| HarnessError::new(format!("File not found: {file_path}")).into())
    }

    pub async fn write(
        &self,
        file_path: &str,
        content: String,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<String> {
        let path = self.resolve_path(file_path, signal.clone()).await?;
        self.sandbox
            .write_text_file(SandboxWriteTextFileOptions {
                path,
                content,
                abort_signal: signal,
                encoding: None,
            })
            .await?;
        Ok(format!("Wrote {file_path}"))
    }

    async fn edit(
        &self,
        input: EditInput,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<String> {
        let content = self.read(input.file_path.clone(), signal.clone()).await?;
        let index = content.find(&input.old_string);
        let (Some(index), false) = (index, input.old_string.is_empty()) else {
            return Err(HarnessError::new("Text to replace was not found or is empty.").into());
        };
        if content[index + input.old_string.len()..].contains(&input.old_string) {
            return Err(HarnessError::new(
                "Text to replace occurs more than once. Include more context.",
            )
            .into());
        }
        let next = format!(
            "{}{}{}",
            &content[..index],
            input.new_string,
            &content[index + input.old_string.len()..]
        );
        self.write(&input.file_path, next, signal).await?;
        Ok(format!("Edited {}", input.file_path))
    }

    async fn bash(
        &self,
        command: String,
        timeout: Option<f64>,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<SandboxRunResult> {
        let Some(seconds) = timeout.filter(|seconds| *seconds > 0.0) else {
            return self.run(command, signal).await;
        };
        let token = signal.map_or_else(CancellationToken::new, |signal| signal.child_token());
        let timer = token.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
            timer.cancel();
        });
        let result = self.run(command, Some(token)).await;
        handle.abort();
        result
    }

    async fn grep(
        &self,
        input: GrepInput,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<String> {
        let target = self
            .resolve_path(input.path.as_deref().unwrap_or("."), signal.clone())
            .await?;
        let mut flags = vec![
            "-r".to_string(),
            "-n".into(),
            "--binary-files=without-match".into(),
        ];
        if input.ignore_case == Some(true) {
            flags.push("-i".into());
        }
        if input.literal == Some(true) {
            flags.push("-F".into());
        }
        if let Some(glob) = input.glob.filter(|glob| !glob.is_empty()) {
            flags.push(format!("--include={glob}"));
        }
        if let Some(context) = input.context.filter(|context| *context != 0.0) {
            flags.push("-C".into());
            flags.push(js_number(context));
        }
        let limit = input.limit.unwrap_or(100.0).floor().max(1.0);
        let command = format!(
            "grep {} -- {} {} | head -n {}",
            flags
                .iter()
                .map(|flag| shell_quote(flag))
                .collect::<Vec<_>>()
                .join(" "),
            shell_quote(&input.pattern),
            shell_quote(&target),
            js_number(limit),
        );
        let result = self.run(command, signal).await?;
        Ok(if !result.stdout.is_empty() {
            result.stdout
        } else if !result.stderr.is_empty() {
            result.stderr
        } else {
            "No matches found".into()
        })
    }

    async fn glob(
        &self,
        input: GlobInput,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<Vec<String>> {
        let target = self
            .resolve_path(input.path.as_deref().unwrap_or("."), signal.clone())
            .await?;
        let result = self
            .run(
                format!("find {} -type f -print0", shell_quote(&target)),
                signal,
            )
            .await?;
        if result.exit_code != 0 {
            return Err(HarnessError::new(result.stderr).into());
        }
        let mut files: Vec<String> = result
            .stdout
            .split('\0')
            .filter(|file| !file.is_empty())
            .map(|file| relative(&target, file))
            .filter(|file| matches_glob(file, &input.pattern))
            .collect();
        files.sort();
        files.truncate(input.limit.map_or(1000, |limit| limit.max(0.0) as usize));
        Ok(files)
    }

    async fn ls(
        &self,
        input: LsInput,
        signal: Option<CancellationToken>,
    ) -> HarnessResult<Vec<String>> {
        let target = self
            .resolve_path(input.path.as_deref().unwrap_or("."), signal.clone())
            .await?;
        let result = self
            .run(
                format!(
                    "find {} -mindepth 1 -maxdepth 1 -printf '%f\\0'",
                    shell_quote(&target)
                ),
                signal,
            )
            .await?;
        if result.exit_code != 0 {
            return Err(HarnessError::new(result.stderr).into());
        }
        let mut entries: Vec<String> = result
            .stdout
            .split('\0')
            .filter(|entry| !entry.is_empty())
            .map(str::to_string)
            .collect();
        entries.sort();
        entries.truncate(input.limit.map_or(500, |limit| limit.max(0.0) as usize));
        Ok(entries)
    }
}

/// `String(number)` for the whole numbers these commands take.
fn js_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

#[cfg(test)]
#[path = "remote_ops_tests.rs"]
mod tests;
