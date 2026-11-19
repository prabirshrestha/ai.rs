//! Port of `agent/internal/bootstrap-recipe.ts`.

use tokio_util::sync::CancellationToken;

use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::{
    SandboxProcessOptions, SandboxReadTextFileOptions, SandboxRunOptions, SandboxSession,
    SandboxWriteTextFileOptions,
};
use crate::harness::spec::bootstrap::HarnessV1Bootstrap;
use crate::harness::spec::paths::{
    harness_state_directory_path, is_absolute_posix, posix_join, resolve_posix,
};
use crate::harness::utils::plain_error;

/// Version of the bootstrap recipe shape itself.
pub const BOOTSTRAP_SCHEMA_VERSION: u32 = 1;

/// `hashHarnessBootstrap`: deterministic 16-char hex identity of a recipe.
///
/// [rust] Files sort by code point; TS uses `localeCompare`.
pub fn hash_harness_bootstrap(recipe: &HarnessV1Bootstrap) -> String {
    let state_directory = harness_state_directory_path("$HOME");
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
    super::hash_strings(&values, 8)
}

/// `bootstrapMarkerPath`: the marker written after a recipe applies.
pub fn bootstrap_marker_path(
    recipe: &HarnessV1Bootstrap,
    identity: &str,
    state_directory: &str,
) -> String {
    posix_join(&[
        &resolve_bootstrap_path(&recipe.bootstrap_dir, state_directory),
        &format!(".bootstrap-{identity}.ok"),
    ])
}

/// `applyBootstrapRecipe`: apply a recipe idempotently.
pub async fn apply_bootstrap_recipe(
    session: &dyn SandboxSession,
    recipe: &HarnessV1Bootstrap,
    identity: &str,
    state_directory: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    let marker_path = bootstrap_marker_path(recipe, identity, state_directory);
    let existing = session
        .read_text_file(SandboxReadTextFileOptions {
            path: marker_path.clone(),
            abort_signal: abort_signal.clone(),
            ..Default::default()
        })
        .await?;
    if existing.is_some() {
        return Ok(());
    }

    let bootstrap_dir = resolve_bootstrap_path(&recipe.bootstrap_dir, state_directory);
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
                path: resolve_bootstrap_path(&file.path, state_directory),
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
        .await
}

/// `a || b` on strings.
pub(crate) fn or_stdout<'a>(stderr: &'a str, stdout: &'a str) -> &'a str {
    if stderr.is_empty() { stdout } else { stderr }
}

fn resolve_bootstrap_path(path: &str, state_directory: &str) -> String {
    if is_absolute_posix(path) {
        path.to_string()
    } else {
        resolve_posix(state_directory, path)
    }
}

#[cfg(test)]
#[path = "bootstrap_recipe_tests.rs"]
mod tests;
