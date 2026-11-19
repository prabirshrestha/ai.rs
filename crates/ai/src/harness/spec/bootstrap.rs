//! Port of `harness-v1-bootstrap.ts`.

use serde::{Deserialize, Serialize};

/// `HarnessV1BootstrapFile`: one file to write into the sandbox. Relative
/// paths resolve against the harness state directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1BootstrapFile {
    pub path: String,
    pub content: String,
}

/// `HarnessV1BootstrapCommand`: run sequentially after all files are
/// written; a non-zero exit aborts the bootstrap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1BootstrapCommand {
    pub command: String,
}

/// `HarnessV1Bootstrap`: adapter-owned bootstrap recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1Bootstrap {
    pub harness_id: String,
    pub bootstrap_dir: String,
    pub files: Vec<HarnessV1BootstrapFile>,
    pub commands: Vec<HarnessV1BootstrapCommand>,
}
