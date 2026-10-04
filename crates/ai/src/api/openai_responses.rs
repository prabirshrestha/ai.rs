//! Port of `api/openai-responses.ts`: the OpenAI Responses API.
//!
//! Not ported yet: [`openai_responses_api()`] streams an error until the implementation
//! lands in a later commit of the rewrite.

use std::sync::Arc;

use super::UnportedApi;
use crate::types::ProviderStreams;

/// The `openai-responses` implementation as `ProviderStreams`.
pub fn openai_responses_api() -> Arc<dyn ProviderStreams> {
    Arc::new(UnportedApi("openai-responses"))
}
