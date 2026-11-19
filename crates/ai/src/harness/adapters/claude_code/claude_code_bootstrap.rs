//! Port of `harness-claude-code/src/claude-code-bootstrap.ts`.
//!
//! [rust] The bridge bundle is the `@ai-sdk/harness-claude-code` dist built
//! from the ported vercel/ai commit by `scripts/refresh-harness-bridges.sh`,
//! vendored under `crates/ai/assets/harness/claude-code` and embedded at
//! compile time instead of read from disk.

use std::sync::{Arc, OnceLock};

use crate::harness::spec::bootstrap::{
    HarnessV1Bootstrap, HarnessV1BootstrapCommand, HarnessV1BootstrapFile,
};

const BRIDGE_PACKAGE_JSON: &str =
    include_str!("../../../../assets/harness/claude-code/package.json");
/// Shipped as `pnpm-lockfile.yaml` because `pnpm pack` excludes files named
/// `pnpm-lock.yaml`.
const BRIDGE_PNPM_LOCK_YAML: &str =
    include_str!("../../../../assets/harness/claude-code/pnpm-lockfile.yaml");
const BRIDGE_PNPM_WORKSPACE_YAML: &str =
    include_str!("../../../../assets/harness/claude-code/pnpm-workspace.yaml");
const BRIDGE_INDEX_MJS: &str = include_str!("../../../../assets/harness/claude-code/index.mjs");

/// `CLAUDE_CODE_BOOTSTRAP_DIR`.
///
/// Bootstrap is derived state stored under `$HOME/.ai-sdk-harness`, outside
/// the agent's working directory. Snapshot-capable providers preserve the
/// installed CLI, bridge, and recipe marker there.
///
/// The session work dir (`startOpts.sessionWorkDir`) lives under the
/// sandbox's default working directory, while the bridge-state dir lives
/// under `$HOME/.ai-sdk-harness/.agent-runs`. Claude's project history is
/// keyed by the working directory, so the same work dir is needed across
/// resumes.
pub const CLAUDE_CODE_BOOTSTRAP_DIR: &str = ".harness-bootstrap/claude-code";

/// `getClaudeCodeBootstrap()`: built once, then cached.
pub fn get_claude_code_bootstrap() -> Arc<HarnessV1Bootstrap> {
    static CACHED: OnceLock<Arc<HarnessV1Bootstrap>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let file = |name: &str, content: &str| HarnessV1BootstrapFile {
                path: format!("{CLAUDE_CODE_BOOTSTRAP_DIR}/{name}"),
                content: content.to_string(),
            };
            Arc::new(HarnessV1Bootstrap {
                harness_id: "claude-code".into(),
                bootstrap_dir: CLAUDE_CODE_BOOTSTRAP_DIR.into(),
                files: vec![
                    file("package.json", BRIDGE_PACKAGE_JSON),
                    file("pnpm-lock.yaml", BRIDGE_PNPM_LOCK_YAML),
                    file("pnpm-workspace.yaml", BRIDGE_PNPM_WORKSPACE_YAML),
                    file("bridge.mjs", BRIDGE_INDEX_MJS),
                ],
                commands: vec![
                    HarnessV1BootstrapCommand {
                        command: "pnpm install --frozen-lockfile --store-dir .pnpm-store".into(),
                    },
                    HarnessV1BootstrapCommand {
                        command: "./node_modules/.bin/claude --version".into(),
                    },
                ],
            })
        })
        .clone()
}
