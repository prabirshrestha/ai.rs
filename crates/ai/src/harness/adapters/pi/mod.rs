//! Pi, run in-process on ai.rs's agent loop, with its file and shell tools
//! working against the sandbox. Enabled by the `harness-pi` feature. The
//! same tools back the native ai adapter (`harness-native`).

#[cfg(feature = "harness-pi")]
pub mod pi_harness;
pub mod pi_model_resolver;
pub mod pi_paths;
pub mod pi_remote_ops;
pub mod pi_session_log;
pub mod pi_skill_command;
pub mod pi_system_prompt;
pub mod pi_tool_result;
pub mod pi_tools;
pub mod pi_utils;

#[cfg(test)]
pub(crate) use super::local_sandbox as pi_test_support;

#[cfg(feature = "harness-pi")]
pub use pi_harness::{PI_HARNESS_ID, PiHarness, PiHarnessSettings, PiResources, PiSandboxSkill};
