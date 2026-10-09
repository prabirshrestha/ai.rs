//! Port of `packages/ai/src/providers` for the in-scope providers, plus the
//! pre-1.0 provider handles (`openai::builder()` and friends).

pub mod all;
pub mod anthropic;
pub mod catalog;
pub mod cloudflare_auth;
pub mod cloudflare_stream;
pub mod cloudflare_workers_ai;
pub mod faux;
pub mod github_copilot;
pub(crate) mod handle;
pub mod model_builder;
pub mod openai;
pub mod openrouter;
pub mod typesafe;

pub use model_builder::{ImageModelBuilder, ModelBuilder};
