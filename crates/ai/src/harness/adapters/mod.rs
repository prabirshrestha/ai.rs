//! Harness adapters (`@ai-sdk/harness-*`), each behind its own cargo
//! feature.

#[cfg(feature = "harness-acp")]
pub mod acp;
