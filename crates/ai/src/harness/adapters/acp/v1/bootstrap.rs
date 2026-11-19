//! Port of `harness-acp/src/v1/acp-v1-bootstrap.ts`.
//!
//! [rust] The bridge bundle is the `@ai-sdk/harness-acp` dist built from the
//! ported vercel/ai commit by `scripts/refresh-harness-bridges.sh`, vendored
//! under `crates/ai/assets/harness/acp` and embedded at compile time instead
//! of read from disk.

use std::sync::{Arc, OnceLock};

use crate::harness::spec::bootstrap::{
    HarnessV1Bootstrap, HarnessV1BootstrapCommand, HarnessV1BootstrapFile,
};

use super::bridge::bridge_environment::ACPAgentLaunch;
use super::settings::ACPSource;

/// The bridge bundle files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ACPBridgeAssets {
    pub package_json: &'static str,
    pub pnpm_lock_yaml: &'static str,
    pub index_mjs: &'static str,
    pub host_tool_mcp_mjs: &'static str,
}

/// The vendored bridge bundle. The lockfile is shipped as
/// `pnpm-lockfile.yaml` because `pnpm pack` excludes files named
/// `pnpm-lock.yaml`.
pub const ACP_BRIDGE_ASSETS: ACPBridgeAssets = ACPBridgeAssets {
    package_json: include_str!("../../../../../assets/harness/acp/package.json"),
    pnpm_lock_yaml: include_str!("../../../../../assets/harness/acp/pnpm-lockfile.yaml"),
    index_mjs: include_str!("../../../../../assets/harness/acp/index.mjs"),
    host_tool_mcp_mjs: include_str!("../../../../../assets/harness/acp/host-tool-mcp.mjs"),
};

const PNPM_INSTALL_COMMAND: &str = "pnpm install --frozen-lockfile --prod --store-dir .pnpm-store";

/// What [`create_acp_bootstrap`] returns: the bootstrap directory and a
/// cached `getBootstrap`.
#[derive(Debug, Clone)]
pub struct ACPBootstrap {
    pub bootstrap_dir: String,
    harness_id: String,
    source: ACPSource,
    assets: ACPBridgeAssets,
    cached: Arc<OnceLock<Arc<HarnessV1Bootstrap>>>,
}

impl ACPBootstrap {
    /// `getBootstrap()`: built once, then cached.
    pub fn get_bootstrap(&self) -> Arc<HarnessV1Bootstrap> {
        self.cached.get_or_init(|| Arc::new(self.build())).clone()
    }

    fn build(&self) -> HarnessV1Bootstrap {
        let mut files = vec![
            ("bridge.mjs", self.assets.index_mjs.to_string()),
            (
                "host-tool-mcp.mjs",
                self.assets.host_tool_mcp_mjs.to_string(),
            ),
        ];
        files.extend(read_install_files(&self.source, &self.assets));
        let mut commands = vec![HarnessV1BootstrapCommand {
            command: PNPM_INSTALL_COMMAND.to_string(),
        }];
        if matches!(self.source, ACPSource::InstallCommand(_)) {
            commands.push(HarnessV1BootstrapCommand {
                command: "bash install.sh".to_string(),
            });
        }
        HarnessV1Bootstrap {
            harness_id: self.harness_id.clone(),
            bootstrap_dir: self.bootstrap_dir.clone(),
            files: files
                .into_iter()
                .map(|(name, content)| HarnessV1BootstrapFile {
                    path: format!("{}/{name}", self.bootstrap_dir),
                    content,
                })
                .collect(),
            commands,
        }
    }
}

/// `createACPBootstrap`.
pub fn create_acp_bootstrap(harness_id: &str, source: &ACPSource) -> ACPBootstrap {
    create_acp_bootstrap_with_assets(harness_id, source, ACP_BRIDGE_ASSETS)
}

/// [`create_acp_bootstrap`] over other bridge assets.
pub fn create_acp_bootstrap_with_assets(
    harness_id: &str,
    source: &ACPSource,
    assets: ACPBridgeAssets,
) -> ACPBootstrap {
    ACPBootstrap {
        bootstrap_dir: format!(".harness-bootstrap/{harness_id}"),
        harness_id: harness_id.to_string(),
        source: source.clone(),
        assets,
        cached: Arc::default(),
    }
}

/// `resolveACPAgentLaunch`.
pub fn resolve_acp_agent_launch(
    bootstrap_directory: &str,
    source: &ACPSource,
    executable: &str,
    args: Option<&[String]>,
) -> ACPAgentLaunch {
    let args = args.map(<[String]>::to_vec).unwrap_or_default();
    match source {
        ACPSource::InstallCommand(_) => {
            let home = format!("{bootstrap_directory}/home");
            ACPAgentLaunch {
                command: format!("{home}/.local/bin/{executable}"),
                args,
                home: Some(home),
            }
        }
        ACPSource::NpmLocked(_) => ACPAgentLaunch {
            command: format!("{bootstrap_directory}/node_modules/.bin/{executable}"),
            args,
            home: None,
        },
    }
}

/// A locked source is the complete bridge installation and therefore
/// replaces the bridge manifest. An install command source installs the
/// agent outside of pnpm, so the bridge manifest only provides the bridge's
/// own runtime dependencies.
fn read_install_files(source: &ACPSource, assets: &ACPBridgeAssets) -> Vec<(&'static str, String)> {
    match source {
        ACPSource::NpmLocked(source) => {
            let mut files = vec![
                ("package.json", source.package_json.clone()),
                ("pnpm-lock.yaml", source.pnpm_lock_yaml.clone()),
            ];
            if let Some(workspace) = &source.pnpm_workspace_yaml {
                files.push(("pnpm-workspace.yaml", workspace.clone()));
            }
            files
        }
        ACPSource::InstallCommand(source) => vec![
            ("package.json", assets.package_json.to_string()),
            ("pnpm-lock.yaml", assets.pnpm_lock_yaml.to_string()),
            ("install.sh", create_install_script(&source.command)),
        ],
    }
}

fn create_install_script(command: &str) -> String {
    format!(
        r#"#!/usr/bin/env bash
set -euo pipefail

ACP_BOOTSTRAP_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export HOME="$ACP_BOOTSTRAP_DIR/home"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$HOME/.local/bin"
cd "$HOME"

{command}
"#
    )
}

#[cfg(test)]
#[path = "bootstrap_tests.rs"]
mod tests;
