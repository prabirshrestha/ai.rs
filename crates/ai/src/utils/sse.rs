//! Server-sent events decoding for the API modules (Rust plumbing; Pi uses
//! the provider SDKs' SSE parsers and an inline decoder for Anthropic).
//!
//! Two line splitters are reproduced, selected by [`SseLineMode`]:
//!
//! - [`SseLineMode::Pi`]: Pi's inline Anthropic decoder (`consumeLine` in
//!   `anthropic-messages.ts`). A `\r` ends a line as soon as it is read, so a
//!   `\r\n` split across two reads gives an extra empty line (which flushes
//!   the pending event early), exactly like Pi.
//! - [`SseLineMode::OpenAISdk`]: the OpenAI SDK's `LineDecoder`. A chunk-final
//!   `\r` also ends its line at once, but a `\n` at the start of the next
//!   chunk is skipped as its continuation.
//!
//! Field decoding (`event`, `data`, comments, the blank-line flush) is the
//! same in both, and neither imposes a line or event size limit.
//!
//! On cancellation, the Pi mode fails with `"Request was aborted"` (Pi's
//! `iterateSseMessages` check before each read); the SDK mode ends the stream
//! without an error and stops yielding already decoded events, like the SDK's
//! `_iterSSEMessages`, so callers see a normal end of stream.

use async_stream::try_stream;
use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub raw: Vec<String>,
}

/// Which upstream line splitter and abort behaviour to reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseLineMode {
    /// Pi's inline Anthropic decoder.
    Pi,
    /// The OpenAI SDK's `Stream.fromSSEResponse()` decoder.
    OpenAISdk,
}

#[derive(Default)]
struct SseDecoderState {
    event: Option<String>,
    data: Vec<String>,
    raw: Vec<String>,
}

/// Pi's inline Anthropic SSE decoder (`iterateSseMessages`).
pub fn events(
    response: reqwest::Response,
    cancellation_token: Option<CancellationToken>,
) -> impl Stream<Item = Result<SseEvent>> + Send + 'static {
    events_with_mode(response, cancellation_token, SseLineMode::Pi)
}

/// The OpenAI SDK's SSE decoder (`_iterSSEMessages`).
pub fn sdk_events(
    response: reqwest::Response,
    cancellation_token: Option<CancellationToken>,
) -> impl Stream<Item = Result<SseEvent>> + Send + 'static {
    events_with_mode(response, cancellation_token, SseLineMode::OpenAISdk)
}

pub fn events_with_mode(
    response: reqwest::Response,
    cancellation_token: Option<CancellationToken>,
    mode: SseLineMode,
) -> impl Stream<Item = Result<SseEvent>> + Send + 'static {
    try_stream! {
        let mut byte_stream = response.bytes_stream();
        let mut state = SseDecoderState::default();
        let mut lines = LineSplitter::new(mode);
        let cancelled = || {
            cancellation_token
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
        };

        loop {
            let chunk = if let Some(cancellation_token) = cancellation_token.as_ref() {
                tokio::select! {
                    _ = cancellation_token.cancelled() => None,
                    chunk = byte_stream.next() => Some(chunk),
                }
            } else {
                Some(byte_stream.next().await)
            };
            let Some(chunk) = chunk else {
                // Cancelled while waiting for the next chunk.
                if mode == SseLineMode::Pi {
                    Err(Error::Aborted("Request was aborted".to_string()))?;
                }
                return;
            };
            let Some(chunk) = chunk else {
                break;
            };
            let chunk = chunk?;
            for line in lines.push(&chunk) {
                if mode == SseLineMode::OpenAISdk && cancelled() {
                    return;
                }
                if let Some(event) = decode_line(&line, &mut state) {
                    yield event;
                }
            }
        }

        if mode == SseLineMode::OpenAISdk && cancelled() {
            return;
        }
        if let Some(line) = lines.finish()
            && let Some(event) = decode_line(&line, &mut state)
        {
            yield event;
        }
        if let Some(event) = flush(&mut state) {
            yield event;
        }
    }
}

/// Incremental line splitter over raw bytes. `\r`, `\n` and `\r\n` end a
/// line; a `\r` at the end of the buffered bytes ends its line at once.
struct LineSplitter {
    mode: SseLineMode,
    buffer: Vec<u8>,
    /// Bytes of `buffer` already searched for a line break.
    searched: usize,
    /// SDK mode: the last chunk ended with `\r`, so a leading `\n` is skipped.
    skip_leading_lf: bool,
}

impl LineSplitter {
    fn new(mode: SseLineMode) -> Self {
        Self {
            mode,
            buffer: Vec::new(),
            searched: 0,
            skip_leading_lf: false,
        }
    }

    fn push(&mut self, mut chunk: &[u8]) -> Vec<String> {
        if std::mem::take(&mut self.skip_leading_lf) && chunk.first() == Some(&b'\n') {
            chunk = &chunk[1..];
        }
        self.buffer.extend_from_slice(chunk);
        let mut lines = Vec::new();
        let mut start = 0;
        let mut search = self.searched;
        while let Some(offset) = self.buffer[search..]
            .iter()
            .position(|byte| matches!(byte, b'\r' | b'\n'))
        {
            let index = search + offset;
            lines.push(String::from_utf8_lossy(&self.buffer[start..index]).into_owned());
            let mut next = index + 1;
            if self.buffer[index] == b'\r' {
                if self.buffer.get(next) == Some(&b'\n') {
                    next += 1;
                } else if next == self.buffer.len() && self.mode == SseLineMode::OpenAISdk {
                    self.skip_leading_lf = true;
                }
            }
            start = next;
            search = next;
        }
        self.buffer.drain(..start);
        self.searched = self.buffer.len();
        lines
    }

    /// The unterminated last line, if any.
    fn finish(&mut self) -> Option<String> {
        self.skip_leading_lf = false;
        if self.buffer.is_empty() {
            return None;
        }
        let line = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        self.searched = 0;
        Some(line)
    }
}

/// `flushSseEvent()`: an event with no (or an empty) name and no data is
/// skipped.
fn flush(state: &mut SseDecoderState) -> Option<SseEvent> {
    if state.event.as_deref().is_none_or(str::is_empty) && state.data.is_empty() {
        return None;
    }
    Some(SseEvent {
        event: state.event.take(),
        data: std::mem::take(&mut state.data).join("\n"),
        raw: std::mem::take(&mut state.raw),
    })
}

/// `decodeSseLine()`.
fn decode_line(line: &str, state: &mut SseDecoderState) -> Option<SseEvent> {
    if line.is_empty() {
        return flush(state);
    }

    state.raw.push(line.to_string());
    if line.starts_with(':') {
        return None;
    }

    let (field, value) = match line.split_once(':') {
        Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
        None => (line, ""),
    };

    match field {
        "event" => state.event = Some(value.to_string()),
        "data" => state.data.push(value.to_string()),
        _ => {}
    }
    None
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_util::sync::CancellationToken;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn cancellation_interrupts_stalled_body_read() {
        let url = spawn_stalled_sse_server().await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();
        let cancellation_token = CancellationToken::new();
        let cancel = cancellation_token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            cancel.cancel();
        });

        let mut events = Box::pin(events(response, Some(cancellation_token)));
        let item = tokio::time::timeout(Duration::from_millis(500), events.next())
            .await
            .expect("SSE read should be cancelled while waiting for a body chunk");

        assert!(matches!(item, Some(Err(Error::Aborted(_)))));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_utf8_split_across_body_chunks() {
        let url = spawn_chunked_sse_server(vec![
            b"data: {\"text\":\"".to_vec(),
            vec![0xF0, 0x9F],
            vec![0x98, 0x80],
            b"\"}\n\n".to_vec(),
        ])
        .await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();

        let mut events = Box::pin(events(response, None));
        let event = events
            .next()
            .await
            .expect("event")
            .expect("valid sse event");

        assert_eq!(event.data, "{\"text\":\"😀\"}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_crlf_split_across_body_chunks() {
        let url = spawn_chunked_sse_server(vec![
            b"data: {\"text\":\"hello\"}\r".to_vec(),
            b"\n\r".to_vec(),
            b"\n".to_vec(),
        ])
        .await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();

        let events = events(response, None)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .expect("valid sse events");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"text\":\"hello\"}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_escaped_json_crlf_inside_data() {
        let url = spawn_chunked_sse_server(vec![
            br#"data: {"text":"a\r\nb"}"#.to_vec(),
            b"\r\n\r\n".to_vec(),
        ])
        .await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();

        let mut events = Box::pin(events(response, None));
        let event = events
            .next()
            .await
            .expect("event")
            .expect("valid sse event");

        assert_eq!(event.data, r#"{"text":"a\r\nb"}"#);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_multiline_data_until_blank_line() {
        let url = spawn_chunked_sse_server(vec![
            b"data: first\r\n".to_vec(),
            b"data: second\r\n".to_vec(),
            b"\r\n".to_vec(),
        ])
        .await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();

        let events = events(response, None)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .expect("valid sse events");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "first\nsecond");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn raw_crlf_injection_becomes_separate_sse_fields() {
        let url = spawn_chunked_sse_server(vec![
            b"data: {\"text\":\"ok\"}\n".to_vec(),
            b"event: error\n".to_vec(),
            b"data: {\"message\":\"boom\"}\n\n".to_vec(),
        ])
        .await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();

        let mut events = Box::pin(events(response, None));
        let event = events
            .next()
            .await
            .expect("event")
            .expect("valid sse event");

        assert_eq!(event.event.as_deref(), Some("error"));
        assert_eq!(event.data, "{\"text\":\"ok\"}\n{\"message\":\"boom\"}");
        assert_eq!(
            event.raw,
            vec![
                "data: {\"text\":\"ok\"}".to_string(),
                "event: error".to_string(),
                "data: {\"message\":\"boom\"}".to_string()
            ]
        );
    }

    async fn collect_events(mode: SseLineMode, chunks: Vec<Vec<u8>>) -> Vec<SseEvent> {
        let url = spawn_chunked_sse_server(chunks).await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();
        events_with_mode(response, None, mode)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .expect("valid sse events")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pi_mode_ends_a_line_at_a_chunk_final_carriage_return() {
        // Pi's `consumeLine` takes the trailing `\r` as a line break, so the
        // `\n` of the split CRLF is an empty line that flushes the event.
        let events = collect_events(
            SseLineMode::Pi,
            vec![b"data: a\r".to_vec(), b"\ndata: b\n\n".to_vec()],
        )
        .await;
        assert_eq!(
            events
                .iter()
                .map(|event| event.data.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sdk_mode_skips_the_newline_of_a_split_crlf() {
        let events = collect_events(
            SseLineMode::OpenAISdk,
            vec![b"data: a\r".to_vec(), b"\ndata: b\n\n".to_vec()],
        )
        .await;
        assert_eq!(
            events
                .iter()
                .map(|event| event.data.as_str())
                .collect::<Vec<_>>(),
            ["a\nb"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn flushes_an_unterminated_last_event_and_skips_nameless_empty_events() {
        for mode in [SseLineMode::Pi, SseLineMode::OpenAISdk] {
            let events = collect_events(
                mode,
                vec![b"event:\n\n: comment\n\nevent: done\ndata: [DONE]".to_vec()],
            )
            .await;
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].event.as_deref(), Some("done"));
            assert_eq!(events[0].data, "[DONE]");
            assert_eq!(
                events[0].raw,
                ["event:", ": comment", "event: done", "data: [DONE]"]
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn has_no_line_or_event_size_limit() {
        let data = "x".repeat(3 * 1024 * 1024);
        let chunks = format!("data: {data}\ndata: {data}\n\n")
            .into_bytes()
            .chunks(256 * 1024)
            .map(<[u8]>::to_vec)
            .collect();
        let events = collect_events(SseLineMode::Pi, chunks).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data.len(), 2 * data.len() + 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sdk_mode_ends_the_stream_on_cancellation() {
        let url = spawn_stalled_sse_server().await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();
        let cancellation_token = CancellationToken::new();
        let cancel = cancellation_token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            cancel.cancel();
        });

        let mut events = Box::pin(sdk_events(response, Some(cancellation_token)));
        let item = tokio::time::timeout(Duration::from_millis(500), events.next())
            .await
            .expect("SSE read should be cancelled while waiting for a body chunk");

        assert!(item.is_none());
    }

    async fn spawn_stalled_sse_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 1024];
            let _ = socket.read(&mut buffer).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: keep-alive\r\n\r\n",
                )
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });
        format!("http://{addr}")
    }

    async fn spawn_chunked_sse_server(chunks: Vec<Vec<u8>>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 1024];
            let _ = socket.read(&mut buffer).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            for chunk in chunks {
                socket.write_all(&chunk).await.unwrap();
                socket.flush().await.unwrap();
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });
        format!("http://{addr}")
    }
}
