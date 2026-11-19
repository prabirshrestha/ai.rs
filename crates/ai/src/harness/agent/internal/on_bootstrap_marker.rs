//! Port of `agent/internal/on-bootstrap-marker.ts`.

use tokio_util::sync::CancellationToken;

use super::bootstrap_recipe::or_stdout;
use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::{
    SandboxProcessOptions, SandboxReadTextFileOptions, SandboxRunOptions, SandboxSession,
    SandboxWriteTextFileOptions,
};
use crate::harness::spec::paths::{dirname_posix, harness_state_directory_path, posix_join};
use crate::harness::utils::plain_error;
use crate::harness::utils::sandbox_home_dir::resolve_sandbox_home_dir;

/// `onBootstrapMarkerPath`:
/// `<home>/.ai-sdk-harness/.on-bootstrap/<sha256(bootstrapHash)>.ok`.
pub async fn on_bootstrap_marker_path(
    session: &dyn SandboxSession,
    bootstrap_hash: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<String> {
    let filename = super::sha256_hex(bootstrap_hash);
    let home = resolve_sandbox_home_dir(session, abort_signal).await?;
    Ok(posix_join(&[
        &harness_state_directory_path(&home),
        ".on-bootstrap",
        &format!("{filename}.ok"),
    ]))
}

/// `hasOnBootstrapMarker`.
pub async fn has_on_bootstrap_marker(
    session: &dyn SandboxSession,
    bootstrap_hash: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<bool> {
    let path = on_bootstrap_marker_path(session, bootstrap_hash, abort_signal.clone()).await?;
    Ok(session
        .read_text_file(SandboxReadTextFileOptions {
            path,
            abort_signal,
            ..Default::default()
        })
        .await?
        .is_some())
}

/// `writeOnBootstrapMarker`.
pub async fn write_on_bootstrap_marker(
    session: &dyn SandboxSession,
    bootstrap_hash: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    let path = on_bootstrap_marker_path(session, bootstrap_hash, abort_signal.clone()).await?;
    let result = session
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command: "mkdir -p \"$MARKER_DIR\"".into(),
                env: Some([("MARKER_DIR".to_string(), dirname_posix(&path))].into()),
                abort_signal: abort_signal.clone(),
                ..Default::default()
            },
            max_output_length: None,
        })
        .await?;
    if result.exit_code != 0 {
        return Err(plain_error(format!(
            "Failed to create onBootstrap marker directory: {}",
            or_stdout(&result.stderr, &result.stdout)
        )));
    }
    session
        .write_text_file(SandboxWriteTextFileOptions {
            path,
            content: String::new(),
            abort_signal,
            encoding: None,
        })
        .await
}
