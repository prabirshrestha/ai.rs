//! Harness adapters, each behind its own feature.

#[cfg(feature = "harness-acp")]
pub mod acp;
pub mod builtin_catalog;
#[cfg(feature = "harness-claude-code")]
pub mod claude_code;
#[cfg(feature = "harness-codex")]
pub mod codex;
#[cfg(feature = "harness-cursor")]
pub mod cursor;
#[cfg(feature = "harness-fx")]
pub mod fx;
#[cfg(feature = "harness-github-copilot")]
pub mod github_copilot;
#[cfg(any(feature = "harness-acp", feature = "harness-opencode"))]
pub(crate) mod host_relay;
pub(crate) mod glob;
pub mod host_tools;
#[cfg(feature = "harness-github-copilot")]
pub(crate) mod jsonc;
#[cfg(any(feature = "harness-native", feature = "harness-pi"))]
pub mod loop_session;
pub mod mcp_host;
#[cfg(feature = "harness-native")]
pub mod native;
#[cfg(feature = "harness-opencode")]
pub mod opencode;
#[cfg(any(feature = "harness-native", feature = "harness-pi"))]
pub mod pi;
pub(crate) mod sandbox_paths;
pub mod stdio_process;
pub mod turn_builder;

#[cfg(test)]
#[path = "pi/pi_test_support.rs"]
pub(crate) mod local_sandbox;
#[cfg(test)]
pub(crate) mod test_temp_dir;
