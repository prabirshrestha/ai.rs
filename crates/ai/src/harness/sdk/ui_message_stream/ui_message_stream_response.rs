//! Mirror of `ai` `ui-message-stream/create-ui-message-stream-response.ts`,
//! `json-to-sse-transform-stream.ts`, `create-sse-stream-with-keep-alive.ts`,
//! `ui-message-stream-headers.ts`, `ui-message-stream-response-init.ts` and
//! `util/prepare-headers.ts`.
//!
//! [rust] There is no fetch `Response`: [`UIMessageStreamResponse`] carries
//! the status, the headers and the SSE body as a byte stream, ready for any
//! HTTP server.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use indexmap::IndexMap;

use crate::harness::sdk::ui_message_stream::process_ui_message_stream::UIMessageStreamError;
use crate::harness::sdk::ui_message_stream::to_ui_message_stream::UIMessageStream;

/// `UI_MESSAGE_STREAM_HEADERS`.
pub const UI_MESSAGE_STREAM_HEADERS: [(&str, &str); 5] = [
    ("content-type", "text/event-stream"),
    ("cache-control", "no-cache"),
    ("connection", "keep-alive"),
    ("x-vercel-ai-ui-message-stream", "v1"),
    ("x-accel-buffering", "no"), // disable nginx buffering
];

const STREAM_OPEN_COMMENT: &str = ": stream-open\n\n";
const KEEP_ALIVE_COMMENT: &str = ": keep-alive\n\n";

/// An SSE text stream.
pub type SseStream = BoxStream<'static, Result<String, UIMessageStreamError>>;

/// `consumeSseStream: ({ stream }) => void | PromiseLike<void>`.
pub type ConsumeSseStreamFn = Arc<dyn Fn(SseStream) -> BoxFuture<'static, ()> + Send + Sync>;

/// `UIMessageStreamResponseInit`: `ResponseInit` plus the SSE options.
#[derive(Clone, Default)]
pub struct UIMessageStreamResponseInit {
    pub status: Option<u16>,
    pub status_text: Option<String>,
    pub headers: Option<IndexMap<String, String>>,
    /// Send `: keep-alive` comments when the stream is idle this long.
    pub keep_alive_ms: Option<u64>,
    /// Receives a copy of the SSE stream, consumed independently of the
    /// response (it is not awaited).
    pub consume_sse_stream: Option<ConsumeSseStreamFn>,
}

/// The `Response` `createUIMessageStreamResponse` returns.
pub struct UIMessageStreamResponse {
    /// Defaults to 200.
    pub status: u16,
    pub status_text: Option<String>,
    pub headers: IndexMap<String, String>,
    /// The UTF-8 encoded SSE body.
    pub body: BoxStream<'static, Result<Bytes, UIMessageStreamError>>,
}

/// A `createSseStreamWithKeepAlive` failure.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("keepAliveMs must be a positive finite timer duration no greater than 2147483647")]
pub struct InvalidKeepAliveError;

/// `prepareHeaders(headers, defaultHeaders)`: `headers`, plus each default
/// whose name (case-insensitively) is not set.
pub fn prepare_headers(
    headers: Option<&IndexMap<String, String>>,
    default_headers: &[(&str, &str)],
) -> IndexMap<String, String> {
    // `new Headers(init)` lower-cases names.
    let mut result: IndexMap<String, String> = headers
        .into_iter()
        .flatten()
        .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
        .collect();
    for (key, value) in default_headers {
        result
            .entry(key.to_ascii_lowercase())
            .or_insert_with(|| value.to_string());
    }
    result
}

/// `JsonToSseTransformStream`: each chunk as a `data:` event, then
/// `data: [DONE]`.
pub fn json_to_sse(stream: UIMessageStream) -> SseStream {
    async_stream::stream! {
        let mut stream = stream;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(chunk) => {
                    yield Ok(format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk).unwrap_or_default()
                    ))
                }
                Err(error) => {
                    yield Err(error);
                    return;
                }
            }
        }
        yield Ok("data: [DONE]\n\n".to_string());
    }
    .boxed()
}

/// `createSseStreamWithKeepAlive({ stream, keepAliveMs })`.
pub fn create_sse_stream_with_keep_alive(
    stream: SseStream,
    keep_alive_ms: Option<u64>,
) -> Result<SseStream, InvalidKeepAliveError> {
    let Some(keep_alive_ms) = keep_alive_ms else {
        return Ok(stream);
    };
    if keep_alive_ms == 0 || keep_alive_ms > 2_147_483_647 {
        return Err(InvalidKeepAliveError);
    }
    let interval = Duration::from_millis(keep_alive_ms);
    Ok(async_stream::stream! {
        let mut stream = stream;
        yield Ok(STREAM_OPEN_COMMENT.to_string());
        loop {
            match tokio::time::timeout(interval, stream.next()).await {
                Ok(Some(item)) => yield item,
                Ok(None) => break,
                Err(_) => yield Ok(KEEP_ALIVE_COMMENT.to_string()),
            }
        }
    }
    .boxed())
}

/// `stream.tee()`: two copies of `stream`, read from a task so each copy
/// is consumed independently.
fn tee(stream: SseStream) -> (SseStream, SseStream) {
    let (tx1, rx1) = futures::channel::mpsc::unbounded();
    let (tx2, rx2) = futures::channel::mpsc::unbounded();
    tokio::spawn(async move {
        let mut stream = stream;
        while let Some(item) = stream.next().await {
            let a = tx1.unbounded_send(item.clone());
            let b = tx2.unbounded_send(item);
            if a.is_err() && b.is_err() {
                break;
            }
        }
    });
    (rx1.boxed(), rx2.boxed())
}

/// `createUIMessageStreamResponse({ stream, status, statusText, headers,
/// keepAliveMs, consumeSseStream })`.
pub fn create_ui_message_stream_response(
    stream: UIMessageStream,
    init: UIMessageStreamResponseInit,
) -> Result<UIMessageStreamResponse, InvalidKeepAliveError> {
    let mut sse_stream =
        create_sse_stream_with_keep_alive(json_to_sse(stream), init.keep_alive_ms)?;
    // with consumeSseStream, tee the stream and hand the second copy to it
    if let Some(consume) = &init.consume_sse_stream {
        let (first, second) = tee(sse_stream);
        sse_stream = first;
        tokio::spawn(consume(second)); // no await (do not block the response)
    }
    Ok(UIMessageStreamResponse {
        status: init.status.unwrap_or(200),
        status_text: init.status_text,
        headers: prepare_headers(init.headers.as_ref(), &UI_MESSAGE_STREAM_HEADERS),
        body: sse_stream.map(|item| item.map(Bytes::from)).boxed(),
    })
}

#[cfg(test)]
#[path = "ui_message_stream_response_tests.rs"]
mod tests;
