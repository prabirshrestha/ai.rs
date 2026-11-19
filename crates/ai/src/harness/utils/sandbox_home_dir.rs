//! Port of `sandbox-home-dir.ts`,
//! `resolve-sandbox-default-working-directory.ts` and
//! `get-restricted-sandbox-session.ts`.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use super::plain_error;
use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::{SandboxProcessOptions, SandboxRunOptions, SandboxSession};
use crate::harness::spec::harness::HarnessV1SandboxSessionRef;

fn run_options(command: &str, abort_signal: Option<CancellationToken>) -> SandboxRunOptions {
    SandboxRunOptions {
        process: SandboxProcessOptions {
            command: command.into(),
            abort_signal,
            ..Default::default()
        },
        max_output_length: None,
    }
}

/// `resolveSandboxHomeDir`: `$HOME` inside the sandbox.
pub async fn resolve_sandbox_home_dir(
    sandbox: &dyn SandboxSession,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<String> {
    let result = sandbox
        .run(run_options("printf \"%s\" \"$HOME\"", abort_signal))
        .await?;
    let home_dir = result.stdout.trim();
    if result.exit_code != 0 || home_dir.is_empty() || !home_dir.starts_with('/') {
        let detail = if result.stderr.is_empty() {
            &result.stdout
        } else {
            &result.stderr
        };
        return Err(plain_error(format!(
            "Unable to resolve sandbox HOME directory: {detail}"
        )));
    }
    Ok(home_dir.to_string())
}

/// `resolveSandboxDefaultWorkingDirectory`.
pub async fn resolve_sandbox_default_working_directory(
    sandbox_session: &HarnessV1SandboxSessionRef,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<String> {
    let session = match sandbox_session {
        HarnessV1SandboxSessionRef::Network(s) => return Ok(s.default_working_directory()),
        HarnessV1SandboxSessionRef::Basic(s) => s,
    };
    let result = session.run(run_options("pwd", abort_signal)).await?;
    if result.exit_code != 0 {
        let detail = if result.stderr.is_empty() {
            &result.stdout
        } else {
            &result.stderr
        };
        return Err(plain_error(format!(
            "Failed to resolve sandbox default working directory (exit {}): {detail}",
            result.exit_code
        )));
    }
    let cwd = result.stdout.trim();
    if !cwd.starts_with('/') {
        return Err(plain_error(format!(
            "Failed to resolve sandbox default working directory: expected an absolute path, got {}.",
            serde_json::to_string(cwd).unwrap_or_default()
        )));
    }
    Ok(if cwd == "/" {
        cwd.to_string()
    } else {
        cwd.trim_end_matches('/').to_string()
    })
}

/// `getRestrictedSandboxSession`.
pub fn get_restricted_sandbox_session(
    sandbox_session: &HarnessV1SandboxSessionRef,
) -> Arc<dyn SandboxSession> {
    sandbox_session.session()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sandbox::SandboxRunResult;
    use crate::harness::test_support::MockSandbox;

    // sandbox-home-dir.test.ts
    #[tokio::test]
    async fn returns_an_absolute_home_directory() {
        let sandbox = MockSandbox::with_result(SandboxRunResult {
            exit_code: 0,
            stdout: "/home/vercel-sandbox\n".into(),
            ..Default::default()
        });
        assert_eq!(
            resolve_sandbox_home_dir(&sandbox, None).await.unwrap(),
            "/home/vercel-sandbox"
        );
    }

    #[tokio::test]
    async fn rejects_non_absolute_output() {
        let sandbox = MockSandbox::with_result(SandboxRunResult {
            exit_code: 0,
            stdout: "home/user".into(),
            ..Default::default()
        });
        let err = resolve_sandbox_home_dir(&sandbox, None).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Unable to resolve sandbox HOME directory")
        );
    }

    #[tokio::test]
    async fn resolves_and_trims_a_basic_sandbox_working_directory() {
        let sandbox = HarnessV1SandboxSessionRef::Basic(Arc::new(MockSandbox::with_result(
            SandboxRunResult {
                exit_code: 0,
                stdout: "/work//\n".into(),
                ..Default::default()
            },
        )));
        assert_eq!(
            resolve_sandbox_default_working_directory(&sandbox, None)
                .await
                .unwrap(),
            "/work"
        );
    }
}
