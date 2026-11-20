//! Port of `harness-pi/src/pi-remote-ops.ts`: Pi's native file and shell
//! tool operations, run against the sandbox.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use indexmap::{IndexMap, IndexSet};
use tokio_util::sync::CancellationToken;

use super::pi_paths::{
    CreatePiPathMapperOptions, PiPathMapper, PiReadablePathRoot, basename_posix,
    create_pi_path_mapper, relative_posix,
};
use crate::harness::error::{BoxError, HarnessResult};
use crate::harness::paths::{dirname_posix, is_absolute_posix};
use crate::harness::sandbox::{
    SandboxPathOptions, SandboxProcessOptions, SandboxRunOptions, SandboxSession,
    SandboxWriteTextFileOptions,
};
use crate::harness::utils::shell_quote::shell_quote;
use crate::harness::utils::{plain_error, random_uuid};

/// `PiRemoteFileChangeKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiRemoteFileChangeKind {
    Create,
    Modify,
}

impl PiRemoteFileChangeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Modify => "modify",
        }
    }
}

/// `onFileChange(event, relativePath, content)`.
pub type PiRemoteFileChangeHandler = Arc<dyn Fn(PiRemoteFileChangeKind, &str, &[u8]) + Send + Sync>;

/// `PiRemoteOpsOptions`.
#[derive(Clone)]
pub struct PiRemoteOpsOptions {
    pub sandbox: Arc<dyn SandboxSession>,
    pub paths: PiPathMapper,
    pub env: Option<IndexMap<String, String>>,
    pub on_file_change: Option<PiRemoteFileChangeHandler>,
}

/// The `grepFiles` input.
#[derive(Debug, Clone, Default)]
pub struct PiGrepInput {
    pub path: Option<String>,
    pub glob: Option<String>,
    pub ignore_case: bool,
    pub literal: bool,
    pub context: Option<i64>,
    pub limit: Option<i64>,
}

/// The `exec` input.
#[derive(Default)]
pub struct PiExecInput<'a> {
    pub on_data: Option<&'a mut (dyn FnMut(&[u8]) + Send)>,
    pub signal: Option<CancellationToken>,
    /// Seconds (Pi's `bash` tool contract).
    pub timeout: Option<f64>,
}

/// `PiSandboxPathResolutionError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiSandboxPathResolutionError {
    pub message: String,
}

impl fmt::Display for PiSandboxPathResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PiSandboxPathResolutionError {}

impl PiSandboxPathResolutionError {
    pub const NAME: &'static str = "PiSandboxPathResolutionError";

    fn boxed(message: String) -> BoxError {
        Box::new(Self { message })
    }
}

struct RunShellResult {
    exit_code: i32,
    output: Vec<u8>,
    stdout: String,
    stderr: String,
}

fn escape_find_path_pattern(input: &str) -> String {
    input
        .chars()
        .map(|character| match character {
            '\\' => r"[\\]".to_string(),
            '*' => "[*]".to_string(),
            '?' => "[?]".to_string(),
            '[' => "[[]".to_string(),
            ']' => "[]]".to_string(),
            other => other.to_string(),
        })
        .collect()
}

const MAX_GREP_DIAGNOSTIC_BYTES: usize = 8_192;
const REALPATH_FRAME: &str = "__PI_REALPATH_FRAME__";

fn realpath_shell_lines(target_variable: &str) -> Vec<String> {
    vec![
        "set -o pipefail".into(),
        format!("realpath_marker={}", shell_quote(REALPATH_FRAME)),
        format!(
            "realpath_framed=$(realpath \"${target_variable}\" 2>/dev/null; realpath_status=$?; printf '%s' \"$realpath_marker\"; exit \"$realpath_status\")"
        ),
        "realpath_status=$?".into(),
        "if [ \"$realpath_status\" -ne 0 ]; then echo \"__PI_REALPATH_FAILED__\"; exit 3; fi"
            .into(),
        "resolved=${realpath_framed%$realpath_marker}".into(),
        "resolved=${resolved%$'\\n'}".into(),
    ]
}

fn frame_path_shell(path_expression: &str) -> String {
    format!(
        "if ! printf '%s' {path_expression} | base64 | tr -d '\\r\\n'; then exit 4; fi; printf '%s' \"$realpath_marker\""
    )
}

fn is_base64_path(encoded: &str) -> bool {
    let bytes = encoded.as_bytes();
    if bytes.len() % 4 != 0 {
        return false;
    }
    let is_char = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/';
    let padding = bytes.iter().rev().take_while(|byte| **byte == b'=').count();
    padding <= 2
        && bytes[..bytes.len() - padding]
            .iter()
            .all(|byte| is_char(*byte))
}

fn parse_framed_path(output: &str) -> Option<String> {
    let encoded_path = output.strip_suffix(REALPATH_FRAME)?;
    if encoded_path.is_empty() || !is_base64_path(encoded_path) {
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded_path)
        .ok()?;
    String::from_utf8(bytes).ok()
}

fn denied_root_prune(search_root: &str, relative_denied_roots: &[String]) -> Vec<String> {
    if relative_denied_roots.is_empty() {
        return Vec::new();
    }
    let prefix = if search_root.ends_with('/') {
        search_root.to_string()
    } else {
        format!("{search_root}/")
    };
    let tests: Vec<String> = relative_denied_roots
        .iter()
        .map(|root| {
            format!(
                "-path {}",
                shell_quote(&escape_find_path_pattern(&format!("{prefix}{root}")))
            )
        })
        .collect();
    vec![format!("\\( {} \\) -prune -o", tests.join(" -o "))]
}

struct ResolveTarget {
    remote_path: String,
    input_path: String,
    must_exist: bool,
}

async fn resolve_sandbox_paths(
    sandbox: &dyn SandboxSession,
    targets: &[ResolveTarget],
    env: Option<&IndexMap<String, String>>,
    abort_signal: Option<&CancellationToken>,
) -> HarnessResult<Vec<String>> {
    let mut lines: Vec<String> = Vec::new();
    for target in targets {
        lines.push(format!("target={}", shell_quote(&target.remote_path)));
        if target.must_exist {
            lines.extend(realpath_shell_lines("target"));
            lines.push(
                "if [ ! -e \"$resolved\" ]; then echo \"__PI_REALPATH_NOT_FOUND__\"; exit 2; fi"
                    .into(),
            );
        } else {
            lines.push("if [ -e \"$target\" ] || [ -L \"$target\" ]; then".into());
            lines.extend(realpath_shell_lines("target"));
            lines.extend(
                [
                    "else",
                    "dir=${target%/*}",
                    "base=${target##*/}",
                    "[ -n \"$dir\" ] || dir=/",
                    "missing=\"$base\"",
                    "while [ ! -e \"$dir\" ] && [ ! -L \"$dir\" ]; do parent=${dir%/*}; [ -n \"$parent\" ] || parent=/; if [ \"$parent\" = \"$dir\" ]; then echo \"__PI_REALPATH_NOT_FOUND__\"; exit 2; fi; missing=${dir##*/}/$missing; dir=$parent; done",
                ]
                .map(String::from),
            );
            lines.extend(realpath_shell_lines("dir"));
            lines.push("resolved=\"${resolved%/}/$missing\"".into());
            lines.push("fi".into());
        }
        lines.push(frame_path_shell("\"$resolved\""));
    }
    let result = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command: lines.join("\n"),
                working_directory: None,
                env: env.cloned(),
                abort_signal: abort_signal.cloned(),
            },
            max_output_length: None,
        })
        .await?;
    let mut records: Vec<&str> = result.stdout.split(REALPATH_FRAME).collect();
    let failed_target =
        &targets[(records.len().saturating_sub(1)).min(targets.len().saturating_sub(1))];
    let output = format!("{}{}", result.stdout, result.stderr);
    if output.contains("__PI_REALPATH_NOT_FOUND__") {
        return Err(PiSandboxPathResolutionError::boxed(format!(
            "Path not found: {}",
            failed_target.input_path
        )));
    }
    let last = records.pop();
    if result.exit_code != 0 || last != Some("") || records.len() != targets.len() {
        return Err(PiSandboxPathResolutionError::boxed(format!(
            "Unable to resolve path: {}",
            failed_target.input_path
        )));
    }
    records
        .iter()
        .zip(targets)
        .map(
            |(record, target)| match parse_framed_path(&format!("{record}{REALPATH_FRAME}")) {
                Some(resolved) if is_absolute_posix(&resolved) => Ok(resolved),
                _ => Err(PiSandboxPathResolutionError::boxed(format!(
                    "Unable to resolve path: {}",
                    target.input_path
                ))),
            },
        )
        .collect()
}

/// `resolvePiSandboxPathOrParent(input)`: the canonical sandbox path of
/// `remote_path`, or of its nearest existing parent plus the missing tail.
pub async fn resolve_pi_sandbox_path_or_parent(
    sandbox: &dyn SandboxSession,
    remote_path: &str,
    input_path: &str,
    env: Option<&IndexMap<String, String>>,
    abort_signal: Option<&CancellationToken>,
) -> HarnessResult<String> {
    let mut resolved = resolve_sandbox_paths(
        sandbox,
        &[ResolveTarget {
            remote_path: remote_path.into(),
            input_path: input_path.into(),
            must_exist: false,
        }],
        env,
        abort_signal,
    )
    .await?;
    Ok(resolved.remove(0))
}

struct ResolvedSandboxPath {
    path: String,
    paths: PiPathMapper,
}

/// `localeCompare` of lowercased names.
///
/// [rust] Compares by code point instead of the ICU collation order.
fn compare_case_insensitive(left: &str, right: &str) -> Ordering {
    left.to_lowercase().cmp(&right.to_lowercase())
}

/// `PiRemoteOps`.
#[derive(Clone)]
pub struct PiRemoteOps {
    options: PiRemoteOpsOptions,
}

impl fmt::Debug for PiRemoteOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PiRemoteOps")
            .field("paths", &self.options.paths)
            .finish_non_exhaustive()
    }
}

/// `createPiRemoteOps(options)`.
pub fn create_pi_remote_ops(options: PiRemoteOpsOptions) -> PiRemoteOps {
    PiRemoteOps { options }
}

impl PiRemoteOps {
    /// `paths`.
    pub fn paths(&self) -> &PiPathMapper {
        &self.options.paths
    }

    async fn run_shell(
        &self,
        command: String,
        cwd: Option<&str>,
        signal: Option<CancellationToken>,
        on_data: Option<&mut (dyn FnMut(&[u8]) + Send)>,
    ) -> HarnessResult<RunShellResult> {
        // `sandbox.run({ command })` already wraps in `bash -c`.
        let working_directory = match cwd {
            Some(cwd) if !cwd.is_empty() => Some(self.options.paths.to_sandbox_path(cwd)?),
            _ => None,
        };
        let result = self
            .options
            .sandbox
            .run(SandboxRunOptions {
                process: SandboxProcessOptions {
                    command,
                    working_directory,
                    env: self.options.env.clone(),
                    abort_signal: signal,
                },
                max_output_length: None,
            })
            .await?;
        let output = format!("{}{}", result.stdout, result.stderr).into_bytes();
        if !output.is_empty()
            && let Some(on_data) = on_data
        {
            on_data(&output);
        }
        Ok(RunShellResult {
            exit_code: result.exit_code,
            output,
            stdout: result.stdout,
            stderr: result.stderr,
        })
    }

    async fn resolve_sandbox_path(
        &self,
        remote_path: &str,
        input_path: &str,
        must_exist: bool,
    ) -> HarnessResult<ResolvedSandboxPath> {
        let paths = &self.options.paths;
        let roots: IndexSet<String> = std::iter::once(paths.sandbox_work_dir.clone())
            .chain(paths.readable_sandbox_roots.iter().cloned())
            .chain(paths.writable_sandbox_roots.iter().cloned())
            .chain(paths.denied_sandbox_roots.iter().cloned())
            .collect();
        let mut targets = vec![ResolveTarget {
            remote_path: remote_path.into(),
            input_path: input_path.into(),
            must_exist,
        }];
        targets.extend(roots.iter().map(|root| ResolveTarget {
            remote_path: root.clone(),
            input_path: root.clone(),
            must_exist: false,
        }));
        let resolved = resolve_sandbox_paths(
            self.options.sandbox.as_ref(),
            &targets,
            self.options.env.as_ref(),
            None,
        )
        .await?;
        let (resolved_target, resolved_roots) = resolved.split_first().expect("target");
        let resolved_root =
            |root: &String| resolved_roots[roots.get_index_of(root).expect("root")].clone();
        let denied_roots: IndexSet<String> = paths
            .denied_sandbox_roots
            .iter()
            .cloned()
            .chain(paths.denied_sandbox_roots.iter().map(resolved_root))
            .collect();
        let resolved_paths = create_pi_path_mapper(CreatePiPathMapperOptions {
            sandbox_work_dir: resolved_root(&paths.sandbox_work_dir),
            readable_roots: Some(
                paths
                    .readable_sandbox_roots
                    .iter()
                    .map(|root| PiReadablePathRoot {
                        sandbox_dir: resolved_root(root),
                    })
                    .collect(),
            ),
            writable_roots: Some(
                paths
                    .writable_sandbox_roots
                    .iter()
                    .map(resolved_root)
                    .collect(),
            ),
            denied_roots: Some(denied_roots.into_iter().collect()),
            home_dir: None,
        })?;
        let path = if must_exist {
            resolved_paths.assert_readable_sandbox_path(resolved_target)?
        } else {
            resolved_paths.assert_sandbox_path(resolved_target)?
        };
        Ok(ResolvedSandboxPath {
            path,
            paths: resolved_paths,
        })
    }

    fn relative_denied_roots(
        &self,
        remote_path: &str,
        resolved: &ResolvedSandboxPath,
    ) -> Vec<String> {
        let roots: IndexSet<String> = self
            .options
            .paths
            .relative_denied_roots_under(remote_path)
            .into_iter()
            .chain(resolved.paths.relative_denied_roots_under(&resolved.path))
            .collect();
        roots.into_iter().collect()
    }

    /// `readBuffer(inputPath)`.
    pub async fn read_buffer(&self, input_path: &str) -> HarnessResult<Vec<u8>> {
        let remote_path = self.options.paths.to_readable_sandbox_path(input_path)?;
        let resolved = self
            .resolve_sandbox_path(&remote_path, input_path, true)
            .await?;
        let bytes = self
            .options
            .sandbox
            .read_binary_file(SandboxPathOptions::new(resolved.path))
            .await?;
        match bytes {
            Some(bytes) => Ok(bytes.to_vec()),
            None => Err(plain_error(format!("Path not found: {input_path}"))),
        }
    }

    /// `writeFile(inputPath, content)`.
    pub async fn write_file(&self, input_path: &str, content: &str) -> HarnessResult<()> {
        let remote_path = self.options.paths.to_sandbox_path(input_path)?;
        let resolved = self
            .resolve_sandbox_path(&remote_path, input_path, false)
            .await?;
        let resolved_path = resolved.path.clone();
        let previous = self
            .options
            .sandbox
            .read_binary_file(SandboxPathOptions::new(resolved_path.clone()))
            .await?;
        self.run_shell(
            format!("mkdir -p {}", shell_quote(&dirname_posix(&resolved_path))),
            None,
            None,
            None,
        )
        .await?;
        self.options
            .sandbox
            .write_text_file(SandboxWriteTextFileOptions {
                path: resolved_path.clone(),
                content: content.into(),
                ..Default::default()
            })
            .await?;
        if resolved.paths.is_workspace_path(&resolved_path)
            && let Some(on_file_change) = &self.options.on_file_change
        {
            // `previous ? 'modify' : 'create'`: an empty buffer is truthy.
            let kind = if previous.is_some() {
                PiRemoteFileChangeKind::Modify
            } else {
                PiRemoteFileChangeKind::Create
            };
            on_file_change(
                kind,
                &resolved.paths.to_relative_path(&resolved_path),
                content.as_bytes(),
            );
        }
        Ok(())
    }

    /// `editFile(inputPath, oldText, newText)`: replace the first occurrence.
    pub async fn edit_file(
        &self,
        input_path: &str,
        old_text: &str,
        new_text: &str,
    ) -> HarnessResult<String> {
        let current = String::from_utf8_lossy(&self.read_buffer(input_path).await?).into_owned();
        let Some(index) = current.find(old_text) else {
            return Err(plain_error(format!(
                "Text to replace was not found in {input_path}"
            )));
        };
        let updated = format!(
            "{}{}{}",
            &current[..index],
            new_text,
            &current[index + old_text.len()..]
        );
        self.write_file(input_path, &updated).await?;
        Ok(updated)
    }

    /// `listDirectory(inputPath = '.', limit = 500)`.
    pub async fn list_directory(
        &self,
        input_path: Option<&str>,
        limit: Option<usize>,
    ) -> HarnessResult<Vec<String>> {
        let input_path = input_path.unwrap_or(".");
        let limit = limit.unwrap_or(500);
        let remote_path = self.options.paths.to_readable_sandbox_path(input_path)?;
        let resolved = self
            .resolve_sandbox_path(&remote_path, input_path, true)
            .await?;
        let quoted = shell_quote(&resolved.path);
        let result = self
            .run_shell(
                [
                    format!("if [ ! -e {quoted} ]; then echo \"__PI_LS_NOT_FOUND__\"; exit 2; fi"),
                    format!("if [ ! -d {quoted} ]; then echo \"__PI_LS_NOT_DIR__\"; exit 3; fi"),
                    format!("cd {quoted}"),
                    "ls -1AF".into(),
                ]
                .join("; "),
                None,
                None,
                None,
            )
            .await?;
        let output = String::from_utf8_lossy(&result.output).trim().to_string();
        if output.contains("__PI_LS_NOT_FOUND__") {
            return Err(plain_error(format!("Path not found: {input_path}")));
        }
        if output.contains("__PI_LS_NOT_DIR__") {
            return Err(plain_error(format!("Not a directory: {input_path}")));
        }
        let denied_entries: IndexSet<String> = self
            .relative_denied_roots(&remote_path, &resolved)
            .into_iter()
            .filter(|root| !root.is_empty() && !root.contains('/'))
            .collect();
        let strip_marker = |line: &str| {
            line.strip_suffix(['*', '=', '@', '|'])
                .unwrap_or(line)
                .to_string()
        };
        let mut entries: Vec<String> = output
            .split('\n')
            .filter(|line| !line.is_empty())
            .filter(|line| {
                let name = match line.strip_suffix('/') {
                    Some(name) => name.to_string(),
                    None => strip_marker(line),
                };
                !denied_entries.contains(&name)
            })
            .map(strip_marker)
            .collect();
        entries.sort_by(|left, right| compare_case_insensitive(left, right));
        entries.truncate(limit);
        Ok(entries)
    }

    /// `findFiles(pattern, inputPath = '.', limit = 1000)`.
    pub async fn find_files(
        &self,
        pattern: &str,
        input_path: Option<&str>,
        limit: Option<usize>,
    ) -> HarnessResult<Vec<String>> {
        let input_path = input_path.unwrap_or(".");
        let limit = limit.unwrap_or(1_000);
        let remote_path = self.options.paths.to_readable_sandbox_path(input_path)?;
        let resolved = self
            .resolve_sandbox_path(&remote_path, input_path, true)
            .await?;
        let resolved_path = resolved.path.clone();
        let quoted = shell_quote(&resolved_path);
        let mut find_flags = vec![quoted.clone()];
        find_flags.extend(denied_root_prune(
            &resolved_path,
            &self.relative_denied_roots(&remote_path, &resolved),
        ));
        let find_flags = find_flags.join(" ");
        let result = self
            .run_shell(
                [
                    format!(
                        "if [ ! -e {quoted} ]; then echo \"__PI_FIND_NOT_FOUND__\"; exit 2; fi"
                    ),
                    format!(
                        "if [ -d {quoted} ]; then find {find_flags} -type f -print; else printf '%s\\n' {quoted}; fi"
                    ),
                ]
                .join("; "),
                None,
                None,
                None,
            )
            .await?;
        let output = String::from_utf8_lossy(&result.output).trim().to_string();
        if output.contains("__PI_FIND_NOT_FOUND__") {
            return Err(plain_error(format!("Path not found: {input_path}")));
        }
        let search_root = &resolved_path;
        let mut files: Vec<String> = output
            .split('\n')
            .filter(|line| !line.is_empty())
            .map(|absolute_path| {
                if absolute_path == search_root {
                    basename_posix(absolute_path)
                } else {
                    relative_posix(search_root, absolute_path)
                }
            })
            .filter(|candidate| !candidate.is_empty() && matches_glob(candidate, pattern))
            .collect();
        files.sort_by(|left, right| compare_case_insensitive(left, right));
        files.truncate(limit);
        Ok(files)
    }

    /// `grepFiles(pattern, input)`.
    pub async fn grep_files(&self, pattern: &str, input: PiGrepInput) -> HarnessResult<String> {
        let input_path = input.path.clone().unwrap_or_else(|| ".".into());
        let remote_path = self.options.paths.to_readable_sandbox_path(&input_path)?;
        let resolved = self
            .resolve_sandbox_path(&remote_path, &input_path, true)
            .await?;
        let resolved_path = resolved.path.clone();
        let relative_target = resolved.paths.to_relative_path(&resolved_path);
        let target_path =
            if relative_target.starts_with("../") || is_absolute_posix(&relative_target) {
                resolved_path.clone()
            } else if relative_target.starts_with('-') {
                format!("./{relative_target}")
            } else {
                relative_target
            };
        let relative_denied_roots = self.relative_denied_roots(&remote_path, &resolved);
        let limit = input.limit.unwrap_or(100).max(1);
        let mut common_flags: Vec<String> = vec!["-n".into()];
        if input.ignore_case {
            common_flags.push("-i".into());
        }
        if input.literal {
            common_flags.push("-F".into());
        }
        if let Some(context) = input.context.filter(|context| *context > 0) {
            common_flags.push("-C".into());
            common_flags.push(context.to_string());
        }
        let mut recursive_flags: Vec<String> = vec!["-r".into()];
        recursive_flags.extend(common_flags.iter().cloned());
        recursive_flags.push("-m".into());
        recursive_flags.push(limit.to_string());
        if let Some(glob) = &input.glob {
            recursive_flags.push(format!("--include={glob}"));
        }
        let quote_all = |flags: &[String]| {
            flags
                .iter()
                .map(|flag| shell_quote(flag))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let temporary_path_prefix = format!("/tmp/.ai-sdk-harness-pi-grep-{}", random_uuid());
        let output_path = format!("{temporary_path_prefix}.stdout");
        let stderr_path = format!("{temporary_path_prefix}.stderr");
        let status_path = format!("{temporary_path_prefix}.status");
        let file_output_path = format!("{temporary_path_prefix}.file.stdout");
        let file_stderr_path = format!("{temporary_path_prefix}.file.stderr");
        let find_stderr_path = format!("{temporary_path_prefix}.find.stderr");
        let quoted_pattern = shell_quote(pattern);
        let bounded_file_grep_script = [
            format!("grep_limit={limit}"),
            format!("grep_output={}", shell_quote(&output_path)),
            format!("grep_stderr={}", shell_quote(&stderr_path)),
            format!("grep_status={}", shell_quote(&status_path)),
            format!("grep_file_output={}", shell_quote(&file_output_path)),
            format!("grep_file_stderr={}", shell_quote(&file_stderr_path)),
            "grep_output_lines=$(wc -l < \"$grep_output\")".into(),
            "grep_overall_status=$(cat \"$grep_status\")".into(),
            "for grep_file in \"$@\"; do".into(),
            "grep_remaining=$((grep_limit - grep_output_lines))".into(),
            "if [ \"$grep_remaining\" -le 0 ]; then break; fi".into(),
            format!(
                "grep $binary_option {} -m \"$grep_remaining\" -e {quoted_pattern} \"$grep_file\" > \"$grep_file_output\" 2> \"$grep_file_stderr\"",
                quote_all(&common_flags)
            ),
            "grep_file_status=$?".into(),
            "if [ \"$grep_file_status\" -eq 0 ]; then".into(),
            "grep_display_file=${grep_file#./}".into(),
            "grep_file_lines=0".into(),
            "while IFS= read -r grep_line; do".into(),
            "if [ \"$grep_file_lines\" -ge \"$grep_remaining\" ]; then break; fi".into(),
            "case \"$grep_line\" in".into(),
            "'--') printf '%s\\n' \"$grep_line\" ;;".into(),
            "[0-9]*:*) printf '%s:%s\\n' \"$grep_display_file\" \"$grep_line\" ;;".into(),
            "[0-9]*-*) printf '%s-%s\\n' \"$grep_display_file\" \"$grep_line\" ;;".into(),
            "*) printf '%s:%s\\n' \"$grep_display_file\" \"$grep_line\" ;;".into(),
            "esac".into(),
            "grep_file_lines=$((grep_file_lines + 1))".into(),
            "done < \"$grep_file_output\" >> \"$grep_output\"".into(),
            "grep_output_lines=$((grep_output_lines + grep_file_lines))".into(),
            "grep_overall_status=0".into(),
            "elif [ \"$grep_file_status\" -gt 1 ]; then".into(),
            "grep_diagnostic_size=$(wc -c < \"$grep_stderr\")".into(),
            format!(
                "grep_diagnostic_remaining=$(({MAX_GREP_DIAGNOSTIC_BYTES} - grep_diagnostic_size))"
            ),
            "if [ \"$grep_diagnostic_remaining\" -gt 0 ]; then head -c \"$grep_diagnostic_remaining\" \"$grep_file_stderr\" >> \"$grep_stderr\"; fi".into(),
            "grep_overall_status=2".into(),
            "break".into(),
            "fi".into(),
            "done".into(),
            "printf '%s\\n' \"$grep_overall_status\" > \"$grep_status\"".into(),
        ]
        .join("\n");
        let quoted_target = shell_quote(&target_path);
        let mut find_flags = vec![quoted_target.clone()];
        find_flags.extend(denied_root_prune(&target_path, &relative_denied_roots));
        find_flags.push("-type".into());
        find_flags.push("f".into());
        if let Some(glob) = &input.glob {
            find_flags.push("-name".into());
            find_flags.push(shell_quote(glob));
        }
        let find_flags = find_flags.join(" ");
        let cleanup = [
            &output_path,
            &stderr_path,
            &status_path,
            &file_output_path,
            &file_stderr_path,
            &find_stderr_path,
        ]
        .iter()
        .map(|path| shell_quote(path))
        .collect::<Vec<_>>()
        .join(" ");
        let command = [
            format!(
                "if [ ! -e {} ]; then echo \"__PI_GREP_NOT_FOUND__\"; exit 2; fi",
                shell_quote(&resolved_path)
            ),
            format!("cd {}", shell_quote(&resolved.paths.sandbox_work_dir)),
            // Preserve binary skipping where grep supports it without passing
            // an unsupported option to just-bash.
            "binary_option_error=$(grep --binary-files=without-match -e '' /dev/null 2>&1)".into(),
            "if [ -z \"$binary_option_error\" ]; then binary_option='--binary-files=without-match'; else binary_option=''; fi".into(),
            "export binary_option".into(),
            format!("grep_stderr={}", shell_quote(&stderr_path)),
            format!("grep_output={}", shell_quote(&output_path)),
            format!("grep_status_file={}", shell_quote(&status_path)),
            format!("grep_find_stderr={}", shell_quote(&find_stderr_path)),
            ": > \"$grep_stderr\"".into(),
            // just-bash buffers every pipeline stage and applies grep -m per
            // file. Search files individually there so the producer never
            // emits more than the remaining global result limit. grep can only
            // exclude directories by base name. Use exact-path find pruning
            // when denied roots are present so unrelated directories with the
            // same base name remain searchable.
            format!(
                "if {{ [ -n \"$binary_option\" ] && [ {denied_count} -eq 0 ]; }} || [ ! -d {quoted_target} ]; then set -o pipefail; grep $binary_option {recursive} -e {quoted_pattern} {quoted_target} 2>\"$grep_stderr\" | head -n {limit}; grep_status=$?; else : > \"$grep_output\"; printf '1\\n' > \"$grep_status_file\"; find {find_flags} -exec bash -c {script} bash {{}} + 2>\"$grep_find_stderr\"; find_status=$?; grep_diagnostic_size=$(wc -c < \"$grep_stderr\"); grep_diagnostic_remaining=$(({MAX_GREP_DIAGNOSTIC_BYTES} - grep_diagnostic_size)); if [ \"$grep_diagnostic_remaining\" -gt 0 ]; then head -c \"$grep_diagnostic_remaining\" \"$grep_find_stderr\" >> \"$grep_stderr\"; fi; grep_status=$(cat \"$grep_status_file\"); if [ \"$find_status\" -ne 0 ]; then grep_status=2; fi; cat \"$grep_output\"; fi",
                denied_count = relative_denied_roots.len(),
                recursive = quote_all(&recursive_flags),
                script = shell_quote(&bounded_file_grep_script),
            ),
            format!("head -c {MAX_GREP_DIAGNOSTIC_BYTES} \"$grep_stderr\" >&2"),
            format!("rm -f {cleanup}"),
            "exit \"$grep_status\"".into(),
        ]
        .join("; ");
        let result = self.run_shell(command, None, None, None).await?;

        let output = String::from_utf8_lossy(&result.output).trim().to_string();
        if output.contains("__PI_GREP_NOT_FOUND__") {
            return Err(plain_error(format!("Path not found: {input_path}")));
        }
        let stdout = result.stdout.trim();
        let stderr = result.stderr.trim();

        // GNU grep can receive SIGPIPE (141) after head reaches the limit.
        if matches!(result.exit_code, 0 | 1 | 141) {
            if !stdout.is_empty() {
                return Ok([stdout, stderr]
                    .into_iter()
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n"));
            }
            if !stderr.is_empty() {
                return Err(plain_error(stderr));
            }
            return Ok("No matches found".into());
        }

        // GNU grep exits 2 when recursive traversal encounters unreadable
        // entries, even if it also found useful matches in readable files.
        if result.exit_code == 2 && !stdout.is_empty() && !stderr.is_empty() {
            return Ok(format!("{stdout}\n{stderr}"));
        }
        if !stderr.is_empty() {
            return Err(plain_error(stderr));
        }
        if output.is_empty() {
            Err(plain_error(format!(
                "grep failed with exit code {}",
                result.exit_code
            )))
        } else {
            Err(plain_error(output))
        }
    }

    /// `access(inputPath)`.
    pub async fn access(&self, input_path: &str) -> HarnessResult<()> {
        self.read_buffer(input_path).await.map(|_| ())
    }

    /// `exec(command, cwd, input)`: run a shell command, streaming its output
    /// to `on_data`. `timeout` is in seconds.
    pub async fn exec(
        &self,
        command: &str,
        cwd: &str,
        input: PiExecInput<'_>,
    ) -> HarnessResult<Option<i32>> {
        let controller = CancellationToken::new();
        let timer = input
            .timeout
            .filter(|timeout| *timeout > 0.0)
            .map(|timeout| {
                let controller = controller.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_secs_f64(timeout)).await;
                    controller.cancel();
                })
            });
        let forwarder = input.signal.map(|signal| {
            let controller = controller.clone();
            tokio::spawn(async move {
                signal.cancelled().await;
                controller.cancel();
            })
        });
        let result = self
            .run_shell(
                command.into(),
                Some(cwd),
                Some(controller.clone()),
                input.on_data,
            )
            .await;
        if let Some(forwarder) = forwarder {
            forwarder.abort();
        }
        if let Some(timer) = timer {
            timer.abort();
        }
        Ok(Some(result?.exit_code))
    }
}

pub(crate) use crate::harness::adapters::glob::matches_glob;

#[cfg(test)]
#[path = "pi_remote_ops_tests.rs"]
mod tests;
