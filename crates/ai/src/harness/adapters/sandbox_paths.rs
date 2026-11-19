//! [rust] Path checks run inside the sandbox, shared by the adapters that
//! auto-approve file tools (Claude Code's `permission-paths.ts` and ACP's
//! `permission-locations.ts`). The TS bridges run in the sandbox and call
//! `realpath` directly; the host resolves paths with one shell command.

#![allow(dead_code)]

use tokio_util::sync::CancellationToken;

use crate::harness::paths::{is_absolute_posix, normalize_posix};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxRunOptions, SandboxSession};
use crate::harness::utils::shell_quote;

/// Resolves paths in the sandbox the way `resolvePermissionPath` does: the
/// real path of the deepest existing ancestor plus the missing segments.
/// Prints one line per path, or `!` when it cannot be resolved.
const RESOLVE_PATHS_SCRIPT: &str = r#"
resolve() {
  p=$1; missing=""
  while [ ! -e "$p" ] && [ ! -L "$p" ]; do
    seg=${p##*/}; parent=${p%/*}
    [ -z "$parent" ] && parent=/
    if [ "$parent" = "$p" ] || [ "$seg" = "." ] || [ "$seg" = ".." ]; then echo '!'; return; fi
    missing="/$seg$missing"; p=$parent
  done
  real=$(realpath "$p" 2>/dev/null || readlink -f "$p" 2>/dev/null) || { echo '!'; return; }
  [ -z "$real" ] && { echo '!'; return; }
  [ "$real" = "/" ] && [ -n "$missing" ] && real=""
  printf '%s%s\n' "$real" "$missing"
}
for p in "$@"; do resolve "$p"; done
"#;

/// Resolve `paths` in the sandbox. `None` when the command fails.
pub(crate) async fn resolve_sandbox_paths(
    sandbox: &dyn SandboxSession,
    paths: &[String],
    abort: Option<CancellationToken>,
) -> Option<Vec<String>> {
    let args: Vec<String> = paths.iter().map(|path| shell_quote(path)).collect();
    let command = format!(
        "sh -c {} resolve {}",
        shell_quote(RESOLVE_PATHS_SCRIPT),
        args.join(" ")
    );
    let result = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command,
                abort_signal: abort,
                ..Default::default()
            },
            max_output_length: None,
        })
        .await
        .ok()?;
    if result.exit_code != 0 {
        return None;
    }
    let lines: Vec<String> = result.stdout.lines().map(str::to_string).collect();
    (lines.len() == paths.len()).then_some(lines)
}

/// Whether `candidate` is `root` or below it.
pub(crate) fn relative_inside(root: &str, candidate: &str) -> bool {
    let root = root.trim_end_matches('/');
    let root = if root.is_empty() { "/" } else { root };
    candidate == root
        || root == "/"
        || candidate
            .strip_prefix(root)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// Every path, taken relative to `cwd`, stays inside `required_root` or one
/// of `optional_roots` after symlinks, both as written and normalized.
/// `required_root` must resolve; missing optional roots are skipped. Paths
/// that are empty, padded or use `~` are never trusted.
pub(crate) async fn are_paths_inside_roots(
    sandbox: &dyn SandboxSession,
    cwd: &str,
    required_root: &str,
    optional_roots: &[String],
    paths: &[String],
    abort: Option<CancellationToken>,
) -> bool {
    if paths.is_empty() {
        return false;
    }
    let mut root_inputs = vec![required_root.to_string()];
    root_inputs.extend(optional_roots.iter().cloned());
    let mut candidates = Vec::new();
    for path in paths {
        if path.is_empty() || path != path.trim() || path == "~" || path.starts_with("~/") {
            return false;
        }
        if !is_absolute_posix(path) && !is_absolute_posix(cwd) {
            return false;
        }
        let absolute = if is_absolute_posix(path) {
            path.clone()
        } else {
            format!("{cwd}/{path}")
        };
        let normalized = normalize_posix(&absolute);
        candidates.push(absolute);
        candidates.push(normalized);
    }
    let mut all = root_inputs.clone();
    all.extend(candidates.iter().cloned());
    let Some(resolved) = resolve_sandbox_paths(sandbox, &all, abort).await else {
        return false;
    };
    let (roots, resolved_candidates) = resolved.split_at(root_inputs.len());
    if roots[0] == "!" {
        return false;
    }
    let roots: Vec<&String> = roots.iter().filter(|root| *root != "!").collect();
    resolved_candidates.iter().all(|candidate| {
        candidate != "!" && roots.iter().any(|root| relative_inside(root, candidate))
    })
}
