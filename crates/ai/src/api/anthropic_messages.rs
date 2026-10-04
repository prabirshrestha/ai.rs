//! Port of `api/anthropic-messages.ts`: the Anthropic Messages API.
//!
//! Not ported yet: [`anthropic_messages_api()`] streams an error until the implementation
//! lands in a later commit of the rewrite.

use std::sync::Arc;

use super::UnportedApi;
use crate::types::ProviderStreams;

/// The `anthropic-messages` implementation as `ProviderStreams`.
pub fn anthropic_messages_api() -> Arc<dyn ProviderStreams> {
    Arc::new(UnportedApi("anthropic-messages"))
}
