//! Pi, run in-process on ai.rs's agent loop, with its file and shell tools
//! working against the sandbox. Enabled by the `harness-pi` feature. The
//! same tools back the native ai adapter (`harness-native`).

pub mod pi_paths;
pub mod pi_remote_ops;
pub mod pi_tool_result;
pub mod pi_tools;
pub mod pi_utils;

#[cfg(test)]
pub(crate) mod pi_test_support;
