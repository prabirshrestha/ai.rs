//! Port of `packages/agent/src/stream-fn.ts`.

use std::sync::Arc;

use parking_lot::RwLock;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::types::StreamFn;
use crate::api::lazy::error_stream;
use crate::types::{
    AssistantMessageEventStream, Context, Model, SimpleStreamOptions, TranscriptContext,
};

static DEFAULT_STREAM_FN: RwLock<Option<StreamFn>> = RwLock::new(None);

/// Configure the fallback used by `Agent` and the low-level loops when
/// callers omit a stream function.
///
/// Hosts that provide a default model runtime can install its stream
/// function here.
pub fn set_default_stream_fn(stream_fn: Option<StreamFn>) {
    *DEFAULT_STREAM_FN.write() = stream_fn;
}

/// The configured default stream function.
pub fn get_default_stream_fn() -> AgentResult<StreamFn> {
    DEFAULT_STREAM_FN
        .read()
        .clone()
        .ok_or(AgentError::NoDefaultStreamFn)
}

/// Wrap a closure as a [`StreamFn`] (Rust convenience).
pub fn stream_fn<F>(stream_fn: F) -> StreamFn
where
    F: Fn(Model, TranscriptContext, SimpleStreamOptions) -> AssistantMessageEventStream
        + Send
        + Sync
        + 'static,
{
    Arc::new(move |model, context, options| {
        let stream = stream_fn(model, context, options);
        Box::pin(async move { stream })
    })
}

/// [`crate::compat::stream_simple`] as a [`StreamFn`], the Rust form of
/// passing Pi's `streamSimple` as `streamFn`. A missing API implementation
/// becomes an error stream, so the function never fails.
pub fn stream_simple_fn() -> StreamFn {
    stream_fn(|model, context, options| {
        let context = Context {
            system_prompt: None,
            messages: context.messages,
            tools: None,
        };
        match crate::compat::stream_simple(model.clone(), context, Some(options)) {
            Ok(stream) => stream,
            Err(error) => error_stream(&model, error),
        }
    })
}

/// Serializes tests that replace the default stream function.
#[cfg(test)]
pub(crate) static DEFAULT_STREAM_FN_TEST_LOCK: tokio::sync::Mutex<()> =
    tokio::sync::Mutex::const_new(());
