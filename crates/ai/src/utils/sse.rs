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

#[derive(Default)]
struct SseDecoderState {
    event: Option<String>,
    data: Vec<String>,
    raw: Vec<String>,
}

pub fn events(
    response: reqwest::Response,
    cancellation_token: Option<CancellationToken>,
) -> impl Stream<Item = Result<SseEvent>> + Send + 'static {
    try_stream! {
        let mut byte_stream = response.bytes_stream();
        let mut state = SseDecoderState::default();
        let mut buffer = String::new();

        while let Some(chunk) = byte_stream.next().await {
            if cancellation_token.as_ref().is_some_and(CancellationToken::is_cancelled) {
                Err(Error::Cancelled)?;
            }

            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some((line, rest)) = consume_line(&buffer) {
                buffer = rest;
                if let Some(event) = decode_line(&line, &mut state) {
                    yield event;
                }
            }
        }

        if !buffer.is_empty() {
            if let Some(event) = decode_line(&buffer, &mut state) {
                yield event;
            }
        }

        if let Some(event) = flush(&mut state) {
            yield event;
        }
    }
}

fn flush(state: &mut SseDecoderState) -> Option<SseEvent> {
    if state.event.is_none() && state.data.is_empty() {
        return None;
    }
    Some(SseEvent {
        event: state.event.take(),
        data: std::mem::take(&mut state.data).join("\n"),
        raw: std::mem::take(&mut state.raw),
    })
}

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

fn consume_line(text: &str) -> Option<(String, String)> {
    let index = text.find(['\r', '\n'])?;
    let mut next = index + 1;
    if text.as_bytes().get(index) == Some(&b'\r') && text.as_bytes().get(next) == Some(&b'\n') {
        next += 1;
    }
    Some((text[..index].to_string(), text[next..].to_string()))
}
