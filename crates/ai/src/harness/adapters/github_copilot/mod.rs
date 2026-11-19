//! Port of `@ai-sdk/harness-github-copilot`: the GitHub Copilot CLI agent
//! over ACP. Enabled by the `harness-github-copilot` cargo feature.

pub mod github_copilot_gateway_model;
pub mod github_copilot_harness;
pub mod github_copilot_subscription;

#[cfg(test)]
mod github_copilot_harness_tests;
#[cfg(test)]
mod github_copilot_subscription_tests;

pub use github_copilot_harness::{
    GitHubCopilotAuthenticationMode, GitHubCopilotHarness, GitHubCopilotHarnessSettings,
    GitHubCopilotReasoningEffort, create_github_copilot, github_copilot,
    github_copilot_builtin_tools,
};

/// `VERSION` of the ported `@ai-sdk/harness-github-copilot` package.
pub const VERSION: &str = "1.0.85";
