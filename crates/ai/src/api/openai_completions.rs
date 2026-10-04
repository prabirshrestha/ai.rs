//! Port of `api/openai-completions.ts`: the OpenAI Chat Completions API.
//!
//! Not ported yet: [`openai_completions_api()`] streams an error until the implementation
//! lands in a later commit of the rewrite.

use std::sync::Arc;

use super::UnportedApi;
use crate::types::ProviderStreams;

/// The `openai-completions` implementation as `ProviderStreams`.
pub fn openai_completions_api() -> Arc<dyn ProviderStreams> {
    Arc::new(UnportedApi("openai-completions"))
}
