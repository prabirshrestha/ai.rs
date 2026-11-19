//! Harness adapters (`@ai-sdk/harness-*`), each behind its own cargo
//! feature.

#[cfg(feature = "harness-acp")]
pub mod acp;
#[cfg(feature = "harness-cursor")]
pub mod cursor;
#[cfg(feature = "harness-github-copilot")]
pub mod github_copilot;

#[cfg(feature = "harness-acp")]
pub(crate) mod jsonc;
#[cfg(all(test, feature = "harness-acp"))]
pub(crate) mod test_temp_dir;
#[cfg(feature = "harness-acp")]
pub(crate) mod zod_schema;
