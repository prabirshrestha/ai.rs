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
        let mut buffer = Vec::new();

        loop {
            let chunk = if let Some(cancellation_token) = cancellation_token.as_ref() {
                tokio::select! {
                    _ = cancellation_token.cancelled() => Err(Error::Cancelled),
                    chunk = byte_stream.next() => Ok(chunk),
                }?
            } else {
                byte_stream.next().await
            };

            let Some(chunk) = chunk else {
                break;
            };
            let chunk = chunk?;
            buffer.extend_from_slice(&chunk);
            while let Some((line, rest)) = consume_line(&buffer) {
                buffer = rest;
                if let Some(event) = decode_line(&line, &mut state) {
                    yield event;
                }
            }
        }

        if !buffer.is_empty()
            && let Some(event) = decode_line(&String::from_utf8_lossy(&buffer), &mut state) {
                yield event;
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

fn consume_line(buffer: &[u8]) -> Option<(String, Vec<u8>)> {
    let index = buffer
        .iter()
        .position(|byte| matches!(byte, b'\r' | b'\n'))?;
    let mut next = index + 1;
    if buffer.get(index) == Some(&b'\r') && buffer.get(next) == Some(&b'\n') {
        next += 1;
    }
    Some((
        String::from_utf8_lossy(&buffer[..index]).into_owned(),
        buffer[next..].to_vec(),
    ))
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

        assert!(matches!(item, Some(Err(Error::Cancelled))));
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
