//! Port of `utils/bridge-diagnostics.ts`: logging and stream helpers for
//! bridge processes.

use std::error::Error as StdError;
use std::io::Write as _;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::future::{BoxFuture, Shared};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::harness::errors::{AbortError, BoxError, HarnessError};
use crate::harness::sandbox::{ByteStream, SandboxProcessControl};
use crate::harness::utils::plain_error;

const DEFAULT_TAIL_LIMIT: usize = 20;

/// `BridgeProcessStreamName`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BridgeProcessStreamName {
    Stdout,
    Stderr,
}

impl BridgeProcessStreamName {
    fn as_str(self) -> &'static str {
        match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        }
    }
}

/// The `unknown` error that [`format_bridge_error`] accepts.
#[derive(Debug, Clone, Copy)]
pub enum BridgeErrorInput<'a> {
    /// An `Error` instance.
    Error(&'a (dyn StdError + 'static)),
    /// A value from a bridge frame, possibly a serialized error.
    Value(&'a Value),
}

/// `formatBridgeError(error)`.
///
/// [rust] Rust errors have no `stack`, so an error formats as
/// `name: message`.
pub fn format_bridge_error(error: BridgeErrorInput<'_>) -> String {
    match error {
        BridgeErrorInput::Error(error) => {
            let name = match error.downcast_ref::<HarnessError>() {
                Some(error) => error.name(),
                None if AbortError::is_instance(error) => AbortError.name(),
                None => "Error",
            };
            format!("{name}: {error}")
        }
        BridgeErrorInput::Value(value) => {
            if let Some(object) = value.as_object()
                && let Some(message) = object.get("message").and_then(Value::as_str)
            {
                // `error.stack ?? ...`: only `undefined` and `null` fall through.
                if let Some(stack) = object.get("stack").filter(|stack| !stack.is_null()) {
                    return stringify_unknown(stack);
                }
                let name = object
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .map(|name| format!("{name}: "))
                    .unwrap_or_default();
                return format!("{name}{message}");
            }
            stringify_unknown(value)
        }
    }
}

fn stringify_unknown(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn write_to_stderr(line: &str) {
    let _ = std::io::stderr().write_all(line.as_bytes());
}

/// Options of [`log_bridge_error`].
pub struct LogBridgeErrorOptions<'a> {
    pub harness_id: &'a str,
    pub session_id: Option<&'a str>,
    pub context: Option<&'a str>,
    pub error: BridgeErrorInput<'a>,
    /// Defaults to writing to stderr.
    pub write: Option<&'a mut dyn FnMut(&str)>,
}

/// `logBridgeError(...)`: writes an error, one prefixed line at a time.
pub fn log_bridge_error(options: LogBridgeErrorOptions<'_>) {
    let prefix = format!(
        "[harness:{}:error{}]",
        options.harness_id,
        options
            .session_id
            .filter(|id| !id.is_empty())
            .map(|id| format!(" session={id}"))
            .unwrap_or_default()
    );
    let formatted = format_bridge_error(options.error);
    let message = match options.context.filter(|context| !context.is_empty()) {
        Some(context) => format!("{context}: {formatted}"),
        None => formatted,
    };
    let mut default_write = write_to_stderr;
    let write: &mut dyn FnMut(&str) = match options.write {
        Some(write) => write,
        None => &mut default_write,
    };
    for line in message.split('\n') {
        if !line.trim().is_empty() {
            write(&format!("{prefix} {line}\n"));
        }
    }
}

/// `createBridgeErrorHandler({ harnessId, sessionId })`: logs the `error`
/// of a bridge `error` event.
pub fn create_bridge_error_handler(
    harness_id: impl Into<String>,
    session_id: Option<String>,
) -> impl Fn(&Value) + Send + Sync + 'static {
    let harness_id = harness_id.into();
    move |error: &Value| {
        log_bridge_error(LogBridgeErrorOptions {
            harness_id: &harness_id,
            session_id: session_id.as_deref(),
            context: Some("bridge emitted an error frame"),
            error: BridgeErrorInput::Value(error),
            write: None,
        });
    }
}

/// A shared "stream finished" signal, as the TS `Promise<void>`.
pub type BridgeStreamDone = Shared<BoxFuture<'static, ()>>;

/// `createBridgeStartupError(...)`: an error with the exit code and the
/// collected output tails.
pub async fn create_bridge_startup_error(
    message: &str,
    proc: &dyn SandboxProcessControl,
    stdout_tail: &[String],
    stderr_tail: &Mutex<Vec<String>>,
    stderr_done: Option<BridgeStreamDone>,
) -> BoxError {
    if let Some(stderr_done) = stderr_done {
        let _ = tokio::time::timeout(Duration::from_millis(250), stderr_done).await;
    }

    let mut exit_status = String::new();
    if let Ok(Ok(result)) = tokio::time::timeout(Duration::from_millis(250), proc.wait()).await {
        exit_status = format!(" Exit code: {}.", result.exit_code);
    }

    let mut details = Vec::new();
    if !stdout_tail.is_empty() {
        details.push(format!("stdout:\n{}", stdout_tail.join("\n")));
    }
    let stderr_tail = stderr_tail.lock().clone();
    if !stderr_tail.is_empty() {
        details.push(format!("stderr:\n{}", stderr_tail.join("\n")));
    }

    plain_error(format!(
        "{message}{exit_status}{}",
        if details.is_empty() {
            String::new()
        } else {
            format!("\n\n{}", details.join("\n\n"))
        }
    ))
}

/// Options of [`forward_bridge_process_stream`].
pub struct ForwardBridgeProcessStreamOptions {
    pub stream: ByteStream,
    pub stream_name: BridgeProcessStreamName,
    /// Defaults to `"bridge"`.
    pub source: Option<String>,
    pub collect_tail: Option<Arc<Mutex<Vec<String>>>>,
    /// Defaults to 20.
    pub tail_limit: Option<usize>,
}

/// `forwardBridgeProcessStream(...)`: copies a process stream to stderr,
/// line by line, keeping a tail. Never fails.
pub async fn forward_bridge_process_stream(options: ForwardBridgeProcessStreamOptions) {
    let source = options.source.as_deref().unwrap_or("bridge");
    let tail_limit = options.tail_limit.unwrap_or(DEFAULT_TAIL_LIMIT);
    let mut stream = options.stream;
    let mut decoder = LineDecoder::default();
    loop {
        let (lines, done) = match stream.next().await {
            None => (decoder.flush(), true),
            Some(Ok(chunk)) => (decoder.push(&chunk), false),
            Some(Err(_)) => return,
        };
        for line in lines {
            if let Some(tail) = &options.collect_tail {
                push_tail(&mut tail.lock(), line.clone(), tail_limit);
            }
            write_to_stderr(&format!(
                "[harness:{source}:{}] {line}\n",
                options.stream_name.as_str()
            ));
        }
        if done {
            return;
        }
    }
}

/// `drainBridgeProcessStream(stream)`: reads a stream to its end, discarding
/// it. Never fails.
pub async fn drain_bridge_process_stream(mut stream: ByteStream) {
    while let Some(Ok(_)) = stream.next().await {}
}

pub(crate) fn push_tail(lines: &mut Vec<String>, line: String, limit: usize) {
    lines.push(line);
    while lines.len() > limit {
        lines.remove(0);
    }
}

/// `lineDecoder()`: splits decoded text into trimmed, non-empty lines.
///
/// [rust] Buffers bytes so UTF-8 sequences split across chunks decode as
/// `TextDecoderStream` does.
#[derive(Debug, Default)]
pub(crate) struct LineDecoder {
    buffer: Vec<u8>,
}

impl LineDecoder {
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|&byte| byte == b'\n') {
            let raw: Vec<u8> = self.buffer.drain(..=newline).collect();
            if let Some(line) = Self::line(&raw[..raw.len() - 1]) {
                lines.push(line);
            }
        }
        lines
    }

    pub(crate) fn flush(&mut self) -> Vec<String> {
        let raw = std::mem::take(&mut self.buffer);
        Self::line(&raw).into_iter().collect()
    }

    fn line(raw: &[u8]) -> Option<String> {
        let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
        let line = String::from_utf8_lossy(raw).trim().to_string();
        (!line.is_empty()).then_some(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn formats_errors_and_serialized_errors() {
        let error = HarnessError::new("boom");
        assert_eq!(
            format_bridge_error(BridgeErrorInput::Error(&error)),
            "AI_HarnessError: boom"
        );
        assert_eq!(
            format_bridge_error(BridgeErrorInput::Value(
                &json!({ "name": "TypeError", "message": "bad" })
            )),
            "TypeError: bad"
        );
        assert_eq!(
            format_bridge_error(BridgeErrorInput::Value(
                &json!({ "message": "bad", "stack": "at x" })
            )),
            "at x"
        );
        assert_eq!(
            format_bridge_error(BridgeErrorInput::Value(&json!("text"))),
            "text"
        );
        assert_eq!(
            format_bridge_error(BridgeErrorInput::Value(&json!({ "code": 1 }))),
            r#"{"code":1}"#
        );
    }

    #[test]
    fn logs_each_non_empty_line_with_the_prefix() {
        let mut lines = Vec::new();
        let mut write = |line: &str| lines.push(line.to_string());
        log_bridge_error(LogBridgeErrorOptions {
            harness_id: "codex",
            session_id: Some("s1"),
            context: Some("ctx"),
            error: BridgeErrorInput::Value(&json!({ "message": "one\n\ntwo" })),
            write: Some(&mut write),
        });
        assert_eq!(
            lines,
            vec![
                "[harness:codex:error session=s1] ctx: one\n",
                "[harness:codex:error session=s1] two\n",
            ]
        );
    }

    #[test]
    fn line_decoder_splits_trims_and_flushes() {
        let mut decoder = LineDecoder::default();
        assert_eq!(decoder.push(b" a \r\n\nb"), vec!["a"]);
        assert_eq!(
            decoder.push("\u{e4}".as_bytes()[..1].as_ref()),
            Vec::<String>::new()
        );
        assert_eq!(
            decoder.push(&"\u{e4}".as_bytes()[1..]),
            Vec::<String>::new()
        );
        assert_eq!(decoder.flush(), vec!["b\u{e4}"]);
    }
}
