//! Port of `@ai-sdk/harness-acp`: run any Agent Client Protocol agent as a
//! harness. Enabled by the `harness-acp` cargo feature.

pub mod acp_auth;
pub mod acp_harness;
pub mod acp_tool_call;
pub mod v1;

#[cfg(test)]
mod acp_harness_test_support;
#[cfg(test)]
mod acp_harness_tests;

pub use acp_auth::*;
pub use acp_harness::{ACPHarnessSettings, acp_client_app, create_acp};
pub use acp_tool_call::*;

/// `VERSION` of the ported `@ai-sdk/harness-acp` package.
pub const VERSION: &str = "1.0.128";
