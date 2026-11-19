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
