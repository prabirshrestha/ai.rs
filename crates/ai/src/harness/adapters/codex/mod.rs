//! Port of `@ai-sdk/harness-codex`: the Codex app server through its own
//! bridge. Enabled by the `harness-codex` cargo feature.

pub mod codex_auth;
pub mod codex_bootstrap;
pub mod codex_bridge_protocol;
pub mod codex_harness;
pub mod codex_subscription;

#[cfg(test)]
mod codex_auth_tests;
#[cfg(test)]
mod codex_harness_tests;
#[cfg(test)]
mod codex_instructions_tests;

pub use codex_auth::CodexAuthenticationMode;
pub use codex_harness::{
    CodexHarness, CodexHarnessSettings, codex, codex_builtin_tools, create_codex,
};

/// `VERSION` of the ported `@ai-sdk/harness-codex` package.
pub const VERSION: &str = "1.0.190";
