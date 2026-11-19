//! Port of `@ai-sdk/harness-cursor`: the Cursor CLI agent over ACP. Enabled
//! by the `harness-cursor` cargo feature.

pub mod cursor_harness;
pub mod cursor_subscription;

#[cfg(test)]
mod cursor_harness_tests;

pub use cursor_harness::{
    CursorAuthenticationMode, CursorHarnessSettings, create_cursor, cursor, cursor_builtin_tools,
};

/// `VERSION` of the ported `@ai-sdk/harness-cursor` package.
pub const VERSION: &str = "1.0.103";
