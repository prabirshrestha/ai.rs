//! Harness adapters, each behind its own feature.

#[cfg(feature = "harness-acp")]
pub mod acp;
pub mod builtin_catalog;
#[cfg(feature = "harness-claude-code")]
pub mod claude_code;
#[cfg(feature = "harness-codex")]
pub mod codex;
pub mod host_tools;
#[cfg(any(feature = "harness-native", feature = "harness-pi"))]
pub mod loop_session;
pub mod mcp_host;
#[cfg(feature = "harness-native")]
pub mod native;
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
