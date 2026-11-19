//! Harness adapters (`@ai-sdk/harness-*`), each behind its own cargo
//! feature.

#[cfg(feature = "harness-acp")]
pub mod acp;
#[cfg(feature = "harness-cursor")]
pub mod cursor;
#[cfg(feature = "harness-github-copilot")]
pub mod github_copilot;

#[cfg(feature = "harness-claude-code")]
pub mod claude_code;
#[cfg(feature = "harness-codex")]
pub mod codex;
#[cfg(feature = "harness-pi")]
pub mod pi;

#[cfg(all(
    test,
    any(
        feature = "harness-acp",
        feature = "harness-claude-code",
        feature = "harness-codex",
        feature = "harness-pi"
    )
))]
pub(crate) mod bridge_test_support;
#[cfg(any(
    feature = "harness-acp",
    feature = "harness-claude-code",
    feature = "harness-codex"
))]
pub(crate) mod builtin_tool_catalog;
#[cfg(feature = "harness-acp")]
pub(crate) mod jsonc;
#[cfg(all(
    test,
    any(
        feature = "harness-acp",
        feature = "harness-claude-code",
        feature = "harness-codex",
        feature = "harness-pi"
    )
))]
pub(crate) mod test_temp_dir;
