//! Port of `packages/ai/src/api`: API implementation modules and shared
//! request helpers.

pub mod anthropic_messages;
pub mod lazy;
pub mod openai_completions;
pub mod openai_responses;
pub mod simple_options;

use crate::types::{Model, ProviderStreams, SimpleStreamOptions, StreamOptions, TranscriptContext};
use crate::utils::event_stream::AssistantMessageEventStream;

/// Stand-in for an API implementation that has not been ported yet. Every
/// request ends with an error event. Replaced module by module as the
/// rewrite lands the real implementations.
pub(crate) struct UnportedApi(pub &'static str);

impl ProviderStreams for UnportedApi {
    fn stream(
        &self,
        model: Model,
        _context: TranscriptContext,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        lazy::error_stream(&model, format!("API {} is not implemented yet", self.0))
    }

    fn stream_simple(
        &self,
        model: Model,
        _context: TranscriptContext,
        _options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        lazy::error_stream(&model, format!("API {} is not implemented yet", self.0))
    }
}
