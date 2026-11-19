//! Port of `harness-claude-code/src/claude-code-thinking.ts`.

use serde::{Deserialize, Serialize};

/// `display` of an adaptive or enabled [`ClaudeCodeThinkingConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClaudeCodeThinkingDisplay {
    Summarized,
    Omitted,
}

/// `ClaudeCodeThinkingConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ClaudeCodeThinkingConfig {
    Adaptive {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display: Option<ClaudeCodeThinkingDisplay>,
    },
    Enabled {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display: Option<ClaudeCodeThinkingDisplay>,
    },
    Disabled,
}
