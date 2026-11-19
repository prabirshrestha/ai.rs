//! Port of the Vercel AI SDK harness (`@ai-sdk/harness`, spec `harness-v1`).
//!
//! A harness wraps a third-party coding-agent runtime (Claude Code, Codex,
//! …) behind one contract: [`HarnessV1`] starts a [`HarnessV1Session`], each
//! prompt turn emits [`HarnessV1StreamPart`]s and returns a
//! [`HarnessV1PromptControl`].
//!
//! Module map:
//! - this module re-exports the spec (`@ai-sdk/harness`);
//! - [`sandbox`]: the sandbox contract (`Experimental_SandboxSession`);
//! - [`provider`] and [`sdk`]: the `@ai-sdk/provider` and AI SDK types the
//!   harness API is defined in.
//!
//! Enabled by the `harness` cargo feature (on by default).

// Wire enums mirror TS unions variant for variant; boxing would change the
// public shape for no benefit.
#![allow(clippy::large_enum_variant)]

pub mod adapters;
pub mod agent;
pub mod errors;
pub mod provider;
pub mod sandbox;
pub mod sdk;
pub mod spec;
pub mod utils;

#[cfg(test)]
pub(crate) mod test_support;

pub use errors::*;
pub use spec::*;
