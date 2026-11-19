//! Port of `harness-codex/src/codex-bootstrap.ts`.
//!
//! [rust] The bridge bundle is the `@ai-sdk/harness-codex` dist built
//! from the ported vercel/ai commit by `scripts/refresh-harness-bridges.sh`,
//! vendored under `crates/ai/assets/harness/codex` and embedded at
//! compile time instead of read from disk.

use std::sync::{Arc, OnceLock};

use crate::harness::spec::bootstrap::{
    HarnessV1Bootstrap, HarnessV1BootstrapCommand, HarnessV1BootstrapFile,
};

const BRIDGE_PACKAGE_JSON: &str = include_str!("../../../../assets/harness/codex/package.json");
/// Shipped as `pnpm-lockfile.yaml` because `pnpm pack` excludes files named
/// `pnpm-lock.yaml`.
const BRIDGE_PNPM_LOCK_YAML: &str =
    include_str!("../../../../assets/harness/codex/pnpm-lockfile.yaml");
const BRIDGE_INDEX_MJS: &str = include_str!("../../../../assets/harness/codex/index.mjs");

/// `CODEX_BOOTSTRAP_DIR`.
///
/// Bootstrap is derived state stored under `$HOME/.ai-sdk-harness`, outside
/// the agent's working directory. Snapshot-capable providers preserve the
/// installed CLI, bridge, and recipe marker there.
///
/// The session work dir (`startOpts.sessionWorkDir`) lives under the
/// sandbox's default working directory (the provider's persistent mount),
/// so any files the agent edits survive both detach -> attach and stop ->
/// snapshot -> resume cycles. Session-specific harness state lives under
/// `$HOME/.ai-sdk-harness/.agent-runs`, outside the agent workdir.
pub const CODEX_BOOTSTRAP_DIR: &str = ".harness-bootstrap/codex";

/// `getCodexBootstrap()`: built once, then cached.
pub fn get_codex_bootstrap() -> Arc<HarnessV1Bootstrap> {
    static CACHED: OnceLock<Arc<HarnessV1Bootstrap>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let file = |name: &str, content: &str| HarnessV1BootstrapFile {
                path: format!("{CODEX_BOOTSTRAP_DIR}/{name}"),
                content: content.to_string(),
            };
            Arc::new(HarnessV1Bootstrap {
                harness_id: "codex".into(),
                bootstrap_dir: CODEX_BOOTSTRAP_DIR.into(),
                files: vec![
                    file("package.json", BRIDGE_PACKAGE_JSON),
                    file("pnpm-lock.yaml", BRIDGE_PNPM_LOCK_YAML),
                    file("bridge.mjs", BRIDGE_INDEX_MJS),
                ],
                commands: vec![HarnessV1BootstrapCommand {
                    command: "pnpm install --frozen-lockfile --store-dir .pnpm-store".into(),
                }],
            })
        })
        .clone()
}
