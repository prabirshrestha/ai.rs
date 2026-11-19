//! Port of `@ai-sdk/harness-claude-code`: the Claude Code CLI through its
//! own bridge. Enabled by the `harness-claude-code` cargo feature.

pub mod claude_code_auth;
pub mod claude_code_bootstrap;
pub mod claude_code_bridge_protocol;
pub mod claude_code_harness;
pub mod claude_code_subscription;
pub mod claude_code_thinking;

#[cfg(test)]
mod claude_code_auth_tests;
#[cfg(test)]
mod claude_code_harness_tests;
#[cfg(test)]
mod claude_code_instructions_tests;
#[cfg(test)]
mod claude_code_start_integration_tests;

pub use claude_code_auth::ClaudeCodeAuthenticationMode;
pub use claude_code_harness::{
    ClaudeCodeHarness, ClaudeCodeHarnessSettings, claude_code, claude_code_builtin_tools,
    create_claude_code,
};
pub use claude_code_thinking::ClaudeCodeThinkingConfig;

/// `VERSION` of the ported `@ai-sdk/harness-claude-code` package.
pub const VERSION: &str = "1.0.192";
