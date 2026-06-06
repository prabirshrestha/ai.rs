use async_stream::try_stream;
use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;
const MAX_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;

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
    events_with_limits(
        response,
        cancellation_token,
        MAX_SSE_LINE_BYTES,
        MAX_SSE_EVENT_BYTES,
    )
}

fn events_with_limits(
    response: reqwest::Response,
    cancellation_token: Option<CancellationToken>,
    max_line_bytes: usize,
    max_event_bytes: usize,
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
            if buffer.len() > max_line_bytes
                && !buffer.iter().any(|byte| matches!(byte, b'\r' | b'\n'))
            {
                Err(Error::Provider(format!(
                    "SSE line exceeded {max_line_bytes} bytes"
                )))?;
            }
            while let Some((line, rest)) = consume_line(&buffer) {
                buffer = rest;
                if line.len() > max_line_bytes {
                    Err(Error::Provider(format!(
                        "SSE line exceeded {max_line_bytes} bytes"
                    )))?;
                }
                if let Some(event) = decode_line(&line, &mut state, max_event_bytes)? {
                    yield event;
                }
            }
        }

        if buffer.len() > max_line_bytes {
            Err(Error::Provider(format!(
                "SSE line exceeded {max_line_bytes} bytes"
            )))?;
        }
        if !buffer.is_empty() {
            let line = String::from_utf8_lossy(trim_final_line_ending(&buffer));
            if let Some(event) = decode_line(&line, &mut state, max_event_bytes)? {
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

fn decode_line(
    line: &str,
    state: &mut SseDecoderState,
    max_event_bytes: usize,
) -> Result<Option<SseEvent>> {
    if line.is_empty() {
        return Ok(flush(state));
    }

    let projected_event_bytes = state
        .raw
        .iter()
        .map(String::len)
        .sum::<usize>()
        .saturating_add(line.len());
    if projected_event_bytes > max_event_bytes {
        return Err(Error::Provider(format!(
            "SSE event exceeded {max_event_bytes} bytes"
        )));
    }

    state.raw.push(line.to_string());
    if line.starts_with(':') {
        return Ok(None);
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
    Ok(None)
}

fn consume_line(buffer: &[u8]) -> Option<(String, Vec<u8>)> {
    let index = buffer
        .iter()
        .position(|byte| matches!(byte, b'\r' | b'\n'))?;
    if buffer.get(index) == Some(&b'\r') && index + 1 == buffer.len() {
        return None;
    }
    let mut next = index + 1;
    if buffer.get(index) == Some(&b'\r') && buffer.get(next) == Some(&b'\n') {
        next += 1;
    }
    Some((
        String::from_utf8_lossy(&buffer[..index]).into_owned(),
        buffer[next..].to_vec(),
    ))
}

fn trim_final_line_ending(buffer: &[u8]) -> &[u8] {
    buffer
        .strip_suffix(b"\r")
        .or_else(|| buffer.strip_suffix(b"\n"))
        .unwrap_or(buffer)
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

    #[tokio::test(flavor = "current_thread")]
    async fn rejects_overlong_unterminated_line() {
        let max_line_bytes = 1024;
        let url = spawn_chunked_sse_server(vec![vec![b'x'; max_line_bytes + 1]]).await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();

        let mut events = Box::pin(events_with_limits(
            response,
            None,
            max_line_bytes,
            MAX_SSE_EVENT_BYTES,
        ));
        let error = events
            .next()
            .await
            .expect("error")
            .expect_err("overlong line should fail");

        assert!(matches!(error, Error::Provider(message) if message.contains("SSE line exceeded")));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rejects_overlong_event() {
        let max_event_bytes = 255;
        let mut event = Vec::new();
        while event.len() <= max_event_bytes {
            event.extend_from_slice(b"data: 0123456789\n");
        }
        event.extend_from_slice(b"\n");
        let url = spawn_chunked_sse_server(vec![event]).await;
        let response = reqwest::Client::new().get(url).send().await.unwrap();

        let mut events = Box::pin(events_with_limits(
            response,
            None,
            MAX_SSE_LINE_BYTES,
            max_event_bytes,
        ));
        let error = events
            .next()
            .await
            .expect("error")
            .expect_err("overlong event should fail");

        assert!(
            matches!(error, Error::Provider(message) if message.contains("SSE event exceeded"))
        );
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
