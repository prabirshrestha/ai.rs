//! Port of `api/lazy.ts`.
//!
//! `lazyApi()` is not ported: Rust links API implementations statically, so
//! there is no module to load on first use. `lazy_stream()` keeps its role of
//! returning a stream synchronously while async setup (auth resolution) runs
//! behind it.
//!
//! A panic in the setup or while forwarding is caught like a TS throw (the
//! `.catch()` of `lazyStream`): the stream ends with an error event carrying
//! the panic message.

use std::any::Any;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::{FutureExt, StreamExt};

use crate::Result;
use crate::types::{AssistantMessage, AssistantMessageEvent, Model, StopReason, Usage};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::time::now_millis;

fn create_setup_error_message(model: &Model, error: impl ToString) -> AssistantMessage {
    let mut message = AssistantMessage::empty_for(model);
    message.usage = Usage::default();
    message.stop_reason = StopReason::Error;
    message.error_message = Some(error.to_string());
    message.timestamp = now_millis();
    message
}

async fn forward_stream(
    target: &AssistantMessageEventStream,
    mut source: AssistantMessageEventStream,
) {
    while let Some(event) = source.next().await {
        target.push(event);
    }
    target.end(Some(source.result().await));
}

/// The message of a caught panic: its `&str` or `String` payload, as a thrown
/// JS error's `message`.
pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "panic".to_string()
    }
}

/// Returns a stream synchronously while running async setup (auth resolution)
/// behind it on a spawned Tokio task. Setup failures, and panics in the setup
/// or while forwarding, terminate the stream with an error event.
pub fn lazy_stream<F>(model: &Model, setup: F) -> AssistantMessageEventStream
where
    F: Future<Output = Result<AssistantMessageEventStream>> + Send + 'static,
{
    let outer = AssistantMessageEventStream::new();
    let target = outer.clone();
    let model = model.clone();
    tokio::spawn(async move {
        let run = AssertUnwindSafe(async {
            match setup.await {
                Ok(inner) => {
                    forward_stream(&target, inner).await;
                    Ok(())
                }
                Err(error) => Err(error.to_string()),
            }
        });
        let error = match run.catch_unwind().await {
            Ok(Ok(())) => return,
            Ok(Err(error)) => error,
            Err(payload) => panic_message(payload.as_ref()),
        };
        let message = create_setup_error_message(&model, error);
        target.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: message.clone(),
        });
        target.end(Some(message));
    });
    outer
}

/// A stream that has already failed with `message`, the synchronous
/// equivalent of a `lazy_stream()` whose setup throws.
pub fn error_stream(model: &Model, message: impl ToString) -> AssistantMessageEventStream {
    let stream = AssistantMessageEventStream::new();
    let error = create_setup_error_message(model, message);
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: error.clone(),
    });
    stream.end(Some(error));
    stream
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[tokio::test]
    async fn setup_failures_become_error_events() {
        let model = Model {
            id: "m".to_string(),
            api: "test-api".to_string(),
            provider: "p".to_string(),
            ..Default::default()
        };
        let stream = lazy_stream(&model, async { Err(Error::message("setup failed")) });
        let events: Vec<_> = stream.clone().collect().await;
        assert_eq!(events.len(), 1);
        let result = stream.result().await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some("setup failed"));
        assert_eq!(result.provider, "p");
    }

    #[tokio::test]
    async fn setup_panics_become_error_events() {
        let model = Model {
            id: "m".to_string(),
            provider: "p".to_string(),
            ..Default::default()
        };
        let stream = lazy_stream(&model, async {
            if true {
                panic!("setup panicked");
            }
            Ok(AssistantMessageEventStream::new())
        });
        let types: Vec<_> = stream
            .clone()
            .map(|event| event.event_type())
            .collect()
            .await;
        assert_eq!(types, vec!["error"]);
        let result = stream.result().await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some("setup panicked"));
        assert_eq!(result.provider, "p");
    }

    #[tokio::test]
    async fn forwards_inner_events_and_result() {
        let model = Model::default();
        let inner_model = model.clone();
        let stream = lazy_stream(&model, async move {
            let inner = AssistantMessageEventStream::new();
            let message = AssistantMessage::empty_for(&inner_model);
            inner.push(AssistantMessageEvent::Start {
                partial: message.clone(),
            });
            inner.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message,
            });
            Ok(inner)
        });
        let types: Vec<_> = stream
            .clone()
            .map(|event| event.event_type())
            .collect()
            .await;
        assert_eq!(types, vec!["start", "done"]);
        assert_eq!(stream.result().await.stop_reason, StopReason::Stop);
    }
}
