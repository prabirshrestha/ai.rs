//! Port of `write-instructions.ts`: append harness instructions to a file
//! in the sandbox, with companion metadata so they can be replaced or
//! removed later without touching the user's own content.

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use super::{plain_error, shell_quote};
use crate::harness::error::HarnessResult;
use crate::harness::paths::{normalize_posix, posix_join};
use crate::harness::sandbox::{
    SandboxProcessOptions, SandboxReadTextFileOptions, SandboxRunOptions, SandboxSession,
    SandboxWriteTextFileOptions,
};

const INSTRUCTIONS_METADATA_VERSION: u32 = 1;

/// `WriteInstructionsOptions`.
pub struct WriteInstructionsOptions<'a> {
    pub sandbox: &'a dyn SandboxSession,
    pub home_path: String,
    pub instructions_file: String,
    pub instructions: Option<String>,
    pub abort_signal: Option<CancellationToken>,
}

/// `WriteInstructionsResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteInstructionsResult {
    pub changed: bool,
    pub file_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InstructionsMetadata {
    version: u32,
    original_content: Option<String>,
    instructions: String,
    applied_content: String,
}

fn trim_trailing_newlines(s: &str) -> &str {
    s.trim_end_matches('\n')
}

/// `writeInstructions`.
pub async fn write_instructions(
    options: WriteInstructionsOptions<'_>,
) -> HarnessResult<WriteInstructionsResult> {
    let WriteInstructionsOptions {
        sandbox,
        home_path,
        instructions_file,
        instructions,
        abort_signal,
    } = options;
    let (target_path, metadata_path) =
        resolve_instructions_file_path(&home_path, &instructions_file)?;
    let trimmed = instructions
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let current = sandbox
        .read_text_file(SandboxReadTextFileOptions {
            path: target_path.clone(),
            abort_signal: abort_signal.clone(),
            ..Default::default()
        })
        .await?;
    let existing =
        read_instructions_metadata(sandbox, &metadata_path, abort_signal.clone()).await?;

    let unchanged = || WriteInstructionsResult {
        changed: false,
        file_path: target_path.clone(),
    };
    let changed = || WriteInstructionsResult {
        changed: true,
        file_path: target_path.clone(),
    };

    if let Some(trimmed) = trimmed {
        let original = derive_original_content(current.as_deref(), existing.as_ref());
        let target_content = match original.as_deref() {
            Some(o) if !o.trim().is_empty() => {
                format!("{}\n\n{trimmed}\n", trim_trailing_newlines(o))
            }
            _ => format!("{trimmed}\n"),
        };
        if current.as_deref() == Some(target_content.as_str())
            && existing
                .as_ref()
                .is_some_and(|m| m.instructions == trimmed && m.original_content == original)
        {
            return Ok(unchanged());
        }
        sandbox
            .write_text_file(SandboxWriteTextFileOptions {
                path: target_path.clone(),
                content: target_content.clone(),
                abort_signal: abort_signal.clone(),
                encoding: None,
            })
            .await?;
        write_instructions_metadata(
            sandbox,
            &metadata_path,
            &InstructionsMetadata {
                version: INSTRUCTIONS_METADATA_VERSION,
                original_content: original,
                instructions: trimmed,
                applied_content: target_content,
            },
            abort_signal,
        )
        .await?;
        return Ok(changed());
    }

    let Some(existing) = existing else {
        return Ok(unchanged());
    };
    match derive_restored_content(current.as_deref(), &existing) {
        Some(restored) => {
            sandbox
                .write_text_file(SandboxWriteTextFileOptions {
                    path: target_path.clone(),
                    content: format!("{}\n", trim_trailing_newlines(&restored)),
                    abort_signal: abort_signal.clone(),
                    encoding: None,
                })
                .await?;
        }
        None => {
            run_sandbox_command(
                sandbox,
                format!("rm -f -- {}", shell_quote(&target_path)),
                abort_signal.clone(),
                format!("Failed to remove instructions file: {target_path}"),
            )
            .await?;
        }
    }
    run_sandbox_command(
        sandbox,
        format!("rm -f -- {}", shell_quote(&metadata_path)),
        abort_signal,
        format!("Failed to remove instructions metadata: {metadata_path}"),
    )
    .await?;
    Ok(changed())
}

fn user_base(disk: &str, instructions: &str) -> Option<Option<String>> {
    let trimmed = trim_trailing_newlines(disk);
    let suffix = format!("\n\n{instructions}");
    if let Some(base) = trimmed.strip_suffix(&suffix) {
        return Some(if base.is_empty() {
            None
        } else {
            Some(base.to_string())
        });
    }
    if trimmed == instructions {
        return Some(None);
    }
    None
}

fn derive_original_content(
    current: Option<&str>,
    existing: Option<&InstructionsMetadata>,
) -> Option<String> {
    let Some(existing) = existing else {
        return current.map(str::to_string);
    };
    let Some(current) = current else {
        return existing.original_content.clone();
    };
    if current == existing.applied_content {
        return existing.original_content.clone();
    }
    if let Some(base) = user_base(current, &existing.instructions) {
        return base;
    }
    existing
        .original_content
        .clone()
        .or_else(|| Some(current.to_string()))
}

fn derive_restored_content(
    current: Option<&str>,
    existing: &InstructionsMetadata,
) -> Option<String> {
    let current = current?;
    if current == existing.applied_content {
        return existing
            .original_content
            .clone()
            .filter(|o| !o.trim().is_empty());
    }
    if let Some(base) = user_base(current, &existing.instructions) {
        return base;
    }
    Some(current.to_string())
}

fn is_win32_absolute(path: &str) -> bool {
    let b = path.as_bytes();
    path.starts_with('\\')
        || path.starts_with('/')
        || (b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':')
}

fn resolve_instructions_file_path(
    home_path: &str,
    instructions_file: &str,
) -> HarnessResult<(String, String)> {
    let json = |s: &str| serde_json::to_string(s).unwrap_or_default();
    if home_path.trim().is_empty() {
        return Err(plain_error(
            "Invalid homePath: expected a non-empty string.",
        ));
    }
    if !home_path.starts_with('/') {
        return Err(plain_error(format!(
            "Invalid homePath {}: expected an absolute POSIX path.",
            json(home_path)
        )));
    }
    let invalid = || {
        plain_error(format!(
            "Invalid instructionsFile {}: expected a relative POSIX path without traversal.",
            json(instructions_file)
        ))
    };
    if instructions_file.trim().is_empty() {
        return Err(invalid());
    }
    let contains_traversal = instructions_file.split(['\\', '/']).any(|s| s == "..");
    let normalized = normalize_posix(instructions_file.trim());
    let no_trailing = normalized.trim_end_matches('/').to_string();
    if instructions_file.contains('\\')
        || instructions_file.starts_with('/')
        || is_win32_absolute(instructions_file)
        || contains_traversal
        || instructions_file.ends_with('/')
        || instructions_file.ends_with('\\')
        || instructions_file.ends_with("/.")
        || instructions_file.ends_with("/..")
        || no_trailing.is_empty()
        || no_trailing == "."
        || normalized.starts_with("../")
        || normalized.contains("/../")
        || normalized.ends_with("/..")
    {
        return Err(invalid());
    }
    let target_path = posix_join(&[home_path, &no_trailing]);
    let home = normalize_posix(home_path);
    let home = home.trim_end_matches('/');
    let inside = target_path
        .strip_prefix(home)
        .is_some_and(|rest| rest.starts_with('/') && rest.len() > 1);
    if !inside {
        return Err(plain_error(format!(
            "Invalid instructionsFile {}: must be a subpath within homePath {}.",
            json(instructions_file),
            json(home_path)
        )));
    }
    let (dir, base) = match target_path.rfind('/') {
        Some(0) => ("/", &target_path[1..]),
        Some(i) => (&target_path[..i], &target_path[i + 1..]),
        None => (".", target_path.as_str()),
    };
    let metadata_path = posix_join(&[dir, &format!(".{base}.ai-sdk-harness-instructions.json")]);
    Ok((target_path, metadata_path))
}

async fn read_instructions_metadata(
    sandbox: &dyn SandboxSession,
    metadata_path: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<Option<InstructionsMetadata>> {
    let content = sandbox
        .read_text_file(SandboxReadTextFileOptions {
            path: metadata_path.into(),
            abort_signal,
            ..Default::default()
        })
        .await?;
    let Some(content) = content else {
        return Ok(None);
    };
    let invalid = || {
        plain_error(format!(
            "Invalid AI SDK harness instructions metadata: {metadata_path}"
        ))
    };
    let value: serde_json::Value = serde_json::from_str(&content).map_err(|_| invalid())?;
    let ok = value.as_object().is_some_and(|m| {
        m.get("version") == Some(&serde_json::json!(INSTRUCTIONS_METADATA_VERSION))
            && matches!(
                m.get("originalContent"),
                Some(serde_json::Value::Null | serde_json::Value::String(_))
            )
            && m.get("instructions").is_some_and(|v| v.is_string())
            && m.get("appliedContent").is_some_and(|v| v.is_string())
    });
    if !ok {
        return Err(invalid());
    }
    serde_json::from_value(value)
        .map(Some)
        .map_err(|_| invalid())
}

async fn write_instructions_metadata(
    sandbox: &dyn SandboxSession,
    metadata_path: &str,
    metadata: &InstructionsMetadata,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    let temporary = format!("{metadata_path}.tmp");
    sandbox
        .write_text_file(SandboxWriteTextFileOptions {
            path: temporary.clone(),
            content: format!(
                "{}\n",
                serde_json::to_string_pretty(metadata).expect("serializable")
            ),
            abort_signal: abort_signal.clone(),
            encoding: None,
        })
        .await?;
    run_sandbox_command(
        sandbox,
        format!(
            "mv -f {} {}",
            shell_quote(&temporary),
            shell_quote(metadata_path)
        ),
        abort_signal,
        format!("Failed to update instructions metadata: {metadata_path}"),
    )
    .await
}

async fn run_sandbox_command(
    sandbox: &dyn SandboxSession,
    command: String,
    abort_signal: Option<CancellationToken>,
    error_message: String,
) -> HarnessResult<()> {
    let result = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command,
                abort_signal,
                ..Default::default()
            },
            max_output_length: None,
        })
        .await?;
    if result.exit_code != 0 {
        let stderr = if result.stderr.is_empty() {
            String::new()
        } else {
            format!(": {}", result.stderr)
        };
        return Err(plain_error(format!(
            "{error_message} (exit {}){stderr}",
            result.exit_code
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sandbox::SandboxRunResult;
    use crate::harness::test_support::{MockSandbox, quoted_values};
    use serde_json::json;

    fn make_sandbox() -> MockSandbox {
        MockSandbox::new(|o, s| {
            let command = &o.process.command;
            if command.starts_with("mv -f ") {
                let q = quoted_values(command);
                let content = s.files.remove(&q[0]).unwrap();
                s.files.insert(q[1].clone(), content);
            }
            if command.starts_with("rm -f -- ") {
                for target in quoted_values(command) {
                    s.files.remove(&target);
                }
            }
            Ok(SandboxRunResult::default())
        })
    }

    async fn write(
        sb: &MockSandbox,
        file: &str,
        instructions: Option<&str>,
    ) -> HarnessResult<WriteInstructionsResult> {
        write_instructions(WriteInstructionsOptions {
            sandbox: sb,
            home_path: "/home/user".into(),
            instructions_file: file.into(),
            instructions: instructions.map(str::to_string),
            abort_signal: None,
        })
        .await
    }

    fn res(changed: bool, path: &str) -> WriteInstructionsResult {
        WriteInstructionsResult {
            changed,
            file_path: path.into(),
        }
    }

    fn file(sb: &MockSandbox, path: &str) -> Option<String> {
        sb.state.lock().files.get(path).cloned()
    }

    fn meta(sb: &MockSandbox) -> serde_json::Value {
        serde_json::from_str(
            &file(sb, "/home/user/.AGENTS.md.ai-sdk-harness-instructions.json").unwrap(),
        )
        .unwrap()
    }

    fn set(sb: &MockSandbox, path: &str, content: &str) {
        sb.state.lock().files.insert(path.into(), content.into());
    }

    // write-instructions.test.ts
    #[tokio::test]
    async fn writes_instructions_to_a_new_file_and_creates_companion_metadata() {
        let sb = make_sandbox();
        let r = write(&sb, "AGENTS.md", Some("Always run tests."))
            .await
            .unwrap();
        assert_eq!(r, res(true, "/home/user/AGENTS.md"));
        assert_eq!(
            file(&sb, "/home/user/AGENTS.md").unwrap(),
            "Always run tests.\n"
        );
        assert_eq!(
            meta(&sb),
            json!({ "version": 1, "originalContent": null, "instructions": "Always run tests.", "appliedContent": "Always run tests.\n" })
        );
    }

    #[tokio::test]
    async fn appends_instructions_without_overriding_pre_existing_content() {
        let sb = make_sandbox();
        set(
            &sb,
            "/home/user/AGENTS.md",
            "# Project Guidelines\n\nBe thorough.",
        );
        let r = write(&sb, "AGENTS.md", Some("Always run tests."))
            .await
            .unwrap();
        assert_eq!(r, res(true, "/home/user/AGENTS.md"));
        assert_eq!(
            file(&sb, "/home/user/AGENTS.md").unwrap(),
            "# Project Guidelines\n\nBe thorough.\n\nAlways run tests.\n"
        );
        assert_eq!(
            meta(&sb),
            json!({
                "version": 1,
                "originalContent": "# Project Guidelines\n\nBe thorough.",
                "instructions": "Always run tests.",
                "appliedContent": "# Project Guidelines\n\nBe thorough.\n\nAlways run tests.\n"
            })
        );
    }

    #[tokio::test]
    async fn subsequent_call_with_different_instructions_replaces_previous() {
        let sb = make_sandbox();
        set(
            &sb,
            "/home/user/AGENTS.md",
            "# Project Guidelines\n\nBe thorough.",
        );
        write(&sb, "AGENTS.md", Some("First instructions."))
            .await
            .unwrap();
        let r = write(&sb, "AGENTS.md", Some("Updated instructions."))
            .await
            .unwrap();
        assert_eq!(r, res(true, "/home/user/AGENTS.md"));
        assert_eq!(
            file(&sb, "/home/user/AGENTS.md").unwrap(),
            "# Project Guidelines\n\nBe thorough.\n\nUpdated instructions.\n"
        );
        assert_eq!(
            meta(&sb),
            json!({
                "version": 1,
                "originalContent": "# Project Guidelines\n\nBe thorough.",
                "instructions": "Updated instructions.",
                "appliedContent": "# Project Guidelines\n\nBe thorough.\n\nUpdated instructions.\n"
            })
        );
    }

    #[tokio::test]
    async fn idempotent_call_returns_unchanged_with_zero_writes() {
        let sb = make_sandbox();
        write(&sb, "AGENTS.md", Some("Always run tests."))
            .await
            .unwrap();
        let (w, r) = {
            let s = sb.state.lock();
            (s.writes.len(), s.runs.len())
        };
        let result = write(&sb, "AGENTS.md", Some("Always run tests."))
            .await
            .unwrap();
        assert_eq!(result, res(false, "/home/user/AGENTS.md"));
        let s = sb.state.lock();
        assert_eq!(s.writes.len(), w);
        assert_eq!(s.runs.len(), r);
    }

    #[tokio::test]
    async fn clearing_restores_pre_existing_content_and_deletes_metadata() {
        let sb = make_sandbox();
        set(
            &sb,
            "/home/user/AGENTS.md",
            "# Project Guidelines\n\nBe thorough.\n",
        );
        write(&sb, "AGENTS.md", Some("Always run tests."))
            .await
            .unwrap();
        let r = write(&sb, "AGENTS.md", None).await.unwrap();
        assert_eq!(r, res(true, "/home/user/AGENTS.md"));
        assert_eq!(
            file(&sb, "/home/user/AGENTS.md").unwrap(),
            "# Project Guidelines\n\nBe thorough.\n"
        );
        assert!(
            file(
                &sb,
                "/home/user/.AGENTS.md.ai-sdk-harness-instructions.json"
            )
            .is_none()
        );
    }

    #[tokio::test]
    async fn clearing_on_a_harness_created_file_removes_file_and_metadata() {
        let sb = make_sandbox();
        write(&sb, "AGENTS.md", Some("Temporary instructions."))
            .await
            .unwrap();
        assert!(file(&sb, "/home/user/AGENTS.md").is_some());
        let r = write(&sb, "AGENTS.md", Some("")).await.unwrap();
        assert_eq!(r, res(true, "/home/user/AGENTS.md"));
        assert!(file(&sb, "/home/user/AGENTS.md").is_none());
        assert!(
            file(
                &sb,
                "/home/user/.AGENTS.md.ai-sdk-harness-instructions.json"
            )
            .is_none()
        );
    }

    #[tokio::test]
    async fn clearing_without_metadata_is_a_no_op() {
        let sb = make_sandbox();
        set(&sb, "/home/user/AGENTS.md", "# Unmanaged");
        let r = write(&sb, "AGENTS.md", None).await.unwrap();
        assert_eq!(r, res(false, "/home/user/AGENTS.md"));
        assert_eq!(file(&sb, "/home/user/AGENTS.md").unwrap(), "# Unmanaged");
    }

    #[tokio::test]
    async fn preserves_external_edits_made_while_instructions_were_active() {
        let sb = make_sandbox();
        set(&sb, "/home/user/AGENTS.md", "# Base\n");
        write(&sb, "AGENTS.md", Some("First instructions."))
            .await
            .unwrap();
        set(
            &sb,
            "/home/user/AGENTS.md",
            "# Base\n\n# User Section\n\nFirst instructions.\n",
        );
        write(&sb, "AGENTS.md", Some("Second instructions."))
            .await
            .unwrap();
        assert_eq!(
            file(&sb, "/home/user/AGENTS.md").unwrap(),
            "# Base\n\n# User Section\n\nSecond instructions.\n"
        );
        write(&sb, "AGENTS.md", Some("")).await.unwrap();
        assert_eq!(
            file(&sb, "/home/user/AGENTS.md").unwrap(),
            "# Base\n\n# User Section\n"
        );
    }

    #[tokio::test]
    async fn manages_multiple_target_files_in_the_same_directory_independently() {
        let sb = make_sandbox();
        write(&sb, "AGENTS.md", Some("Codex instructions."))
            .await
            .unwrap();
        write(&sb, "CLAUDE.md", Some("Claude instructions."))
            .await
            .unwrap();
        assert_eq!(
            file(&sb, "/home/user/AGENTS.md").unwrap(),
            "Codex instructions.\n"
        );
        assert_eq!(
            file(&sb, "/home/user/CLAUDE.md").unwrap(),
            "Claude instructions.\n"
        );
        assert!(
            file(
                &sb,
                "/home/user/.AGENTS.md.ai-sdk-harness-instructions.json"
            )
            .is_some()
        );
        assert!(
            file(
                &sb,
                "/home/user/.CLAUDE.md.ai-sdk-harness-instructions.json"
            )
            .is_some()
        );
        write(&sb, "CLAUDE.md", None).await.unwrap();
        assert!(file(&sb, "/home/user/CLAUDE.md").is_none());
        assert!(
            file(
                &sb,
                "/home/user/.CLAUDE.md.ai-sdk-harness-instructions.json"
            )
            .is_none()
        );
        assert_eq!(
            file(&sb, "/home/user/AGENTS.md").unwrap(),
            "Codex instructions.\n"
        );
        assert!(
            file(
                &sb,
                "/home/user/.AGENTS.md.ai-sdk-harness-instructions.json"
            )
            .is_some()
        );
    }

    #[tokio::test]
    async fn manages_nested_target_paths_cleanly() {
        let sb = make_sandbox();
        let r = write(&sb, ".claude/CLAUDE.md", Some("Nested instructions."))
            .await
            .unwrap();
        assert_eq!(r, res(true, "/home/user/.claude/CLAUDE.md"));
        assert_eq!(
            file(&sb, "/home/user/.claude/CLAUDE.md").unwrap(),
            "Nested instructions.\n"
        );
        assert!(
            file(
                &sb,
                "/home/user/.claude/.CLAUDE.md.ai-sdk-harness-instructions.json"
            )
            .is_some()
        );
    }

    #[tokio::test]
    async fn rejects_invalid_home_path() {
        let sb = make_sandbox();
        for (home, msg) in [
            (
                "relative/home",
                "Invalid homePath \"relative/home\": expected an absolute POSIX path.",
            ),
            ("   ", "Invalid homePath: expected a non-empty string."),
        ] {
            let err = write_instructions(WriteInstructionsOptions {
                sandbox: &sb,
                home_path: home.into(),
                instructions_file: "AGENTS.md".into(),
                instructions: Some("Test".into()),
                abort_signal: None,
            })
            .await
            .unwrap_err();
            assert!(err.to_string().contains(msg));
        }
    }

    #[tokio::test]
    async fn rejects_invalid_instructions_files() {
        for f in [
            "/AGENTS.md",
            "C:\\AGENTS.md",
            "sub\\AGENTS.md",
            "../AGENTS.md",
            "foo/../../bar.md",
            "",
            "   ",
            ".",
            "./",
            "sub/",
            "sub/.",
            "sub/..",
        ] {
            let sb = make_sandbox();
            let err = write(&sb, f, Some("Test")).await.unwrap_err();
            assert!(
                err.to_string()
                    .contains("expected a relative POSIX path without traversal"),
                "{f:?}: {err}"
            );
        }
    }
}
