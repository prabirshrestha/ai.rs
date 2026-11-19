//! Port of `harness-v1-bootstrap.ts`.

use serde::{Deserialize, Serialize};

/// `HarnessBootstrapFile`: one file to write into the sandbox. Relative
/// paths resolve against the harness state directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessBootstrapFile {
    pub path: String,
    pub content: String,
}

/// `HarnessBootstrapCommand`: run sequentially after all files are
/// written; a non-zero exit aborts the bootstrap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessBootstrapCommand {
    pub command: String,
}

/// `HarnessBootstrap`: adapter-owned bootstrap recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessBootstrap {
    pub harness_id: String,
    pub bootstrap_dir: String,
    pub files: Vec<HarnessBootstrapFile>,
    pub commands: Vec<HarnessBootstrapCommand>,
}

/// Version of the bootstrap recipe shape itself.
pub const BOOTSTRAP_SCHEMA_VERSION: u32 = 1;

fn hash_strings(values: &[&str], bytes: usize) -> String {
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
    for value in values {
        ctx.update(value.as_bytes());
        ctx.update(b"\0");
    }
    ctx.finish().as_ref()[..bytes]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `hashHarnessBootstrap`: deterministic 16-char hex identity of a recipe.
///
/// [rust] Files sort by code point; TS uses `localeCompare`.
pub fn hash_harness_bootstrap(recipe: &HarnessBootstrap) -> String {
    let state_directory = crate::harness::paths::harness_state_directory_path("$HOME");
    let mut files: Vec<_> = recipe.files.iter().collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let commands = serde_json::to_string(&recipe.commands).expect("commands serialize");
    let version = BOOTSTRAP_SCHEMA_VERSION.to_string();
    let mut values: Vec<&str> = vec![&state_directory, &recipe.harness_id, &recipe.bootstrap_dir];
    for file in files {
        values.push(&file.path);
        values.push(&file.content);
    }
    values.push(&commands);
    values.push(&version);
    hash_strings(&values, 8)
}

fn resolve_bootstrap_path(path: &str, state_directory: &str) -> String {
    if crate::harness::paths::is_absolute_posix(path) {
        path.to_string()
    } else {
        crate::harness::paths::resolve_posix(state_directory, path)
    }
}

fn or_stdout<'a>(stderr: &'a str, stdout: &'a str) -> &'a str {
    if stderr.is_empty() { stdout } else { stderr }
}

/// `applyBootstrapRecipe`: install the agent in the sandbox once. A marker
/// named after the recipe's hash skips later runs. Returns the bootstrap
/// directory.
pub async fn apply_harness_bootstrap(
    session: &dyn crate::harness::sandbox::SandboxSession,
    recipe: &HarnessBootstrap,
    abort_signal: Option<tokio_util::sync::CancellationToken>,
) -> crate::harness::error::HarnessResult<String> {
    use crate::harness::sandbox::{
        SandboxProcessOptions, SandboxReadTextFileOptions, SandboxRunOptions,
        SandboxWriteTextFileOptions,
    };
    use crate::harness::utils::plain_error;

    let home = crate::harness::utils::sandbox_home_dir::resolve_sandbox_home_dir(
        session,
        abort_signal.clone(),
    )
    .await?;
    let state_directory = crate::harness::paths::harness_state_directory_path(&home);
    let bootstrap_dir = resolve_bootstrap_path(&recipe.bootstrap_dir, &state_directory);
    let marker_path = crate::harness::paths::posix_join(&[
        &bootstrap_dir,
        &format!(".bootstrap-{}.ok", hash_harness_bootstrap(recipe)),
    ]);
    let existing = session
        .read_text_file(SandboxReadTextFileOptions {
            path: marker_path.clone(),
            abort_signal: abort_signal.clone(),
            ..Default::default()
        })
        .await?;
    if existing.is_some() {
        return Ok(bootstrap_dir);
    }
    let mkdir = session
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command: "mkdir -p \"$BOOTSTRAP_DIR\"".into(),
                env: Some([("BOOTSTRAP_DIR".to_string(), bootstrap_dir.clone())].into()),
                abort_signal: abort_signal.clone(),
                ..Default::default()
            },
            max_output_length: None,
        })
        .await?;
    if mkdir.exit_code != 0 {
        return Err(plain_error(format!(
            "Failed to create bootstrap directory for harness '{}' (exit {}): {bootstrap_dir}\n{}",
            recipe.harness_id,
            mkdir.exit_code,
            or_stdout(&mkdir.stderr, &mkdir.stdout)
        )));
    }
    for file in &recipe.files {
        session
            .write_text_file(SandboxWriteTextFileOptions {
                path: resolve_bootstrap_path(&file.path, &state_directory),
                content: file.content.clone(),
                abort_signal: abort_signal.clone(),
                encoding: None,
            })
            .await?;
    }
    for cmd in &recipe.commands {
        let result = session
            .run(SandboxRunOptions {
                process: SandboxProcessOptions {
                    command: cmd.command.clone(),
                    working_directory: Some(bootstrap_dir.clone()),
                    abort_signal: abort_signal.clone(),
                    ..Default::default()
                },
                max_output_length: None,
            })
            .await?;
        if result.exit_code != 0 {
            return Err(plain_error(format!(
                "Bootstrap command failed for harness '{}' (exit {}): {}\n{}",
                recipe.harness_id,
                result.exit_code,
                cmd.command,
                or_stdout(&result.stderr, &result.stdout)
            )));
        }
    }
    session
        .write_text_file(SandboxWriteTextFileOptions {
            path: marker_path,
            content: String::new(),
            abort_signal,
            encoding: None,
        })
        .await?;
    Ok(bootstrap_dir)
}
