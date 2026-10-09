//! Port of durable `src/harness/output.ts`: bounded tool output and adaptive progress commits.
//!
//! Divergences from Pi:
//! - `OutputBuffer::push` takes text (`push`) or bytes (`push_bytes`); the streaming UTF-8 decoder replaces invalid
//!   sequences with U+FFFD like `TextDecoder`.
//! - `Progress` runs its commits as Tokio tasks and measures pauses with the Tokio clock.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use parking_lot::Mutex;
use tokio::sync::oneshot;
use tokio::time::Instant;

use crate::durable::errors::{Error, Result};

use super::types::Retain;

/// Retention limits of one tool's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputLimits {
    pub max_bytes: usize,
    pub max_lines: usize,
    pub retain: Retain,
}

/// Retained output and what the limits dropped.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoundedOutput {
    pub text: String,
    pub dropped_bytes: usize,
    pub dropped_lines: usize,
}

/// An exact slice of the input within the limits, and what it left out.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OutputSlice {
    pub text: String,
    pub bytes: usize,
    pub dropped_bytes: usize,
    pub dropped_lines: usize,
}

const NEWLINE: u8 = b'\n';

fn invalid_output(c: char) -> bool {
    matches!(c, '\u{00}'..='\u{08}' | '\u{0b}'..='\u{1f}' | '\u{fff9}'..='\u{fffb}')
}

/// Remove control characters that break display and transcripts; tabs and newlines stay.
pub fn sanitize_output(text: &str) -> String {
    if !text.chars().any(invalid_output) {
        return text.to_string();
    }
    text.chars().filter(|c| !invalid_output(*c)).collect()
}

/// Bound `text` to whole lines within the limits: the first lines for `head`, the last lines for `tail`. The result is
/// an exact slice, trailing newline included. A single line longer than `max_bytes` is cut at the byte limit on a
/// character boundary.
pub fn bound_output(text: &str, limits: OutputLimits) -> OutputSlice {
    let bytes = text.as_bytes();
    let (from, to) = match limits.retain {
        Retain::Head => head_range(bytes, limits),
        Retain::Tail => tail_range(bytes, limits),
    };
    let kept = &bytes[from..to];
    OutputSlice {
        text: text[from..to].to_string(),
        bytes: kept.len(),
        dropped_bytes: bytes.len() - kept.len(),
        dropped_lines: line_count(bytes) - line_count(kept),
    }
}

fn index_of(bytes: &[u8], from: usize) -> Option<usize> {
    bytes
        .get(from..)?
        .iter()
        .position(|b| *b == NEWLINE)
        .map(|index| index + from)
}

/// `lastIndexOf(NEWLINE, at)`: the last newline at or before `at`.
fn last_index_of(bytes: &[u8], at: usize) -> Option<usize> {
    if bytes.is_empty() {
        return None;
    }
    let end = at.min(bytes.len() - 1);
    bytes[..=end].iter().rposition(|b| *b == NEWLINE)
}

fn head_range(bytes: &[u8], limits: OutputLimits) -> (usize, usize) {
    if limits.max_lines == 0 || limits.max_bytes == 0 {
        return (0, 0);
    }
    let mut end = bytes.len();
    let mut lines = 0;
    let mut next = index_of(bytes, 0);
    while let Some(index) = next {
        lines += 1;
        if lines == limits.max_lines {
            end = index + 1;
            break;
        }
        next = index_of(bytes, index + 1);
    }
    if end > limits.max_bytes {
        end = match last_index_of(bytes, limits.max_bytes - 1) {
            None => character_end(bytes, limits.max_bytes),
            Some(newline) => newline + 1,
        };
    }
    (0, end)
}

fn tail_range(bytes: &[u8], limits: OutputLimits) -> (usize, usize) {
    if limits.max_lines == 0 || limits.max_bytes == 0 {
        return (bytes.len(), bytes.len());
    }
    // A trailing newline ends the last line rather than starting another.
    let last: isize = if bytes.last() == Some(&NEWLINE) {
        bytes.len() as isize - 2
    } else {
        bytes.len() as isize - 1
    };
    let mut start = 0;
    let mut lines = 1;
    let mut next = if last < 0 {
        None
    } else {
        last_index_of(bytes, last as usize)
    };
    while let Some(index) = next {
        if lines == limits.max_lines {
            start = index + 1;
            break;
        }
        lines += 1;
        next = if index == 0 {
            None
        } else {
            last_index_of(bytes, index - 1)
        };
    }
    if bytes.len() - start > limits.max_bytes {
        let from = bytes.len() - limits.max_bytes;
        // The first line starting inside the byte window, or a cut of the last line when it alone is too long.
        start = match index_of(bytes, from.saturating_sub(1)) {
            Some(newline) if newline + 1 < bytes.len() => newline + 1,
            _ => character_start(bytes, from),
        };
    }
    (start, bytes.len())
}

/// The last character boundary at or before `index`.
pub fn character_end(bytes: &[u8], index: usize) -> usize {
    let mut end = index;
    while end > 0 && (bytes.get(end).copied().unwrap_or(0) & 0xc0) == 0x80 {
        end -= 1;
    }
    end
}

/// The first character boundary at or after `index`.
fn character_start(bytes: &[u8], index: usize) -> usize {
    let mut start = index;
    while start < bytes.len() && (bytes[start] & 0xc0) == 0x80 {
        start += 1;
    }
    start
}

fn line_count(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }
    let newlines = bytes.iter().filter(|b| **b == NEWLINE).count();
    newlines + usize::from(bytes.last() != Some(&NEWLINE))
}

/// A streaming UTF-8 decoder (`TextDecoder` with `stream`).
#[derive(Debug, Default)]
struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    /// Decode `chunk` after the pending bytes; with `stream`, an incomplete trailing character stays pending.
    fn decode(&mut self, chunk: &[u8], stream: bool) -> String {
        let mut bytes = std::mem::take(&mut self.pending);
        bytes.extend_from_slice(chunk);
        let mut out = String::new();
        let mut rest: &[u8] = &bytes;
        loop {
            match std::str::from_utf8(rest) {
                Ok(text) => {
                    out.push_str(text);
                    break;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    out.push_str(std::str::from_utf8(&rest[..valid]).expect("valid prefix"));
                    match error.error_len() {
                        Some(len) => {
                            out.push('\u{fffd}');
                            rest = &rest[valid + len..];
                        }
                        None => {
                            if stream {
                                self.pending = rest[valid..].to_vec();
                            } else {
                                out.push('\u{fffd}');
                            }
                            break;
                        }
                    }
                }
            }
        }
        out
    }
}

#[derive(Debug)]
struct Chunk {
    text: String,
    bytes: usize,
    newlines: usize,
}

/// Bounded running output of one tool call. Accepting a chunk costs time proportional to the chunk: head retention
/// stops storing once the window is full, and tail retention drops stored text the window no longer needs when it
/// snapshots. Counts of the whole stream are kept so the dropped totals stay exact.
#[derive(Debug)]
pub struct OutputBuffer {
    limits: OutputLimits,
    decoder: Utf8Decoder,
    /// Stored chunks: for head the start of the stream, for tail a suffix that still contains the next window.
    chunks: Vec<Chunk>,
    stored_bytes: usize,
    stored_newlines: usize,
    full: bool,
    total_bytes: usize,
    total_newlines: usize,
    ends_with_newline: bool,
}

impl OutputBuffer {
    pub fn new(limits: OutputLimits) -> Self {
        Self {
            limits,
            decoder: Utf8Decoder::default(),
            chunks: Vec::new(),
            stored_bytes: 0,
            stored_newlines: 0,
            full: false,
            total_bytes: 0,
            total_newlines: 0,
            ends_with_newline: true,
        }
    }

    /// Bytes currently held; bounded by the limits plus one chunk.
    pub fn stored_bytes(&self) -> usize {
        self.stored_bytes
    }

    /// Accept a text chunk; returns whether anything was accepted.
    pub fn push(&mut self, chunk: &str) -> bool {
        // Bytes of an incomplete character from an earlier byte chunk come first.
        let mut text = self.decoder.decode(&[], false);
        text.push_str(chunk);
        self.accept(text)
    }

    /// Accept a byte chunk; returns whether anything was accepted.
    pub fn push_bytes(&mut self, chunk: &[u8]) -> bool {
        let text = self.decoder.decode(chunk, true);
        self.accept(text)
    }

    /// Flush an incomplete trailing character as a replacement character; call when the stream ends.
    pub fn end(&mut self) {
        let text = self.decoder.decode(&[], false);
        self.accept(text);
    }

    fn accept(&mut self, text: String) -> bool {
        if text.is_empty() {
            return false;
        }
        let bytes = text.len();
        let newlines = count_newlines(&text);
        self.total_bytes += bytes;
        self.total_newlines += newlines;
        self.ends_with_newline = text.ends_with('\n');
        if self.full {
            return true;
        }
        self.chunks.push(Chunk {
            text,
            bytes,
            newlines,
        });
        self.stored_bytes += bytes;
        self.stored_newlines += newlines;
        if self.limits.retain == Retain::Head {
            // Nothing past a full window is ever needed.
            self.full = self.stored_bytes > self.limits.max_bytes
                || self.stored_newlines >= self.limits.max_lines;
            return true;
        }
        // Drop leading chunks while the rest still holds more than a window: more than `max_bytes` bytes or
        // `max_lines` newlines, plus one, so the window's line start can still be found. Each chunk is dropped once.
        let mut drop = 0;
        while self.chunks.len() - drop > 1 {
            let first = &self.chunks[drop];
            let bytes_after = self.stored_bytes - first.bytes;
            let newlines_after = self.stored_newlines - first.newlines;
            if bytes_after <= self.limits.max_bytes + 1
                && newlines_after <= self.limits.max_lines + 1
            {
                break;
            }
            drop += 1;
            self.stored_bytes = bytes_after;
            self.stored_newlines = newlines_after;
        }
        self.chunks.drain(..drop);
        true
    }

    /// Retained, sanitized output and what the limits dropped from the whole stream.
    pub fn snapshot(&mut self) -> BoundedOutput {
        let stored = if self.chunks.len() == 1 {
            self.chunks[0].text.clone()
        } else {
            self.chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect()
        };
        let kept = bound_output(&stored, self.limits);
        let stored_lines = lines(
            self.stored_newlines,
            stored.is_empty() || stored.ends_with('\n'),
        );
        let kept_lines = stored_lines - kept.dropped_lines;
        // Tail windows never reach back before this one, so only the kept slice needs storing.
        if self.limits.retain == Retain::Tail || self.chunks.len() > 1 {
            let (text, bytes) = if self.limits.retain == Retain::Tail {
                (kept.text.clone(), kept.bytes)
            } else {
                (stored, self.stored_bytes)
            };
            let newlines = count_newlines(&text);
            self.chunks = if text.is_empty() {
                Vec::new()
            } else {
                vec![Chunk {
                    text,
                    bytes,
                    newlines,
                }]
            };
            self.stored_bytes = bytes;
            self.stored_newlines = self.chunks.first().map_or(0, |chunk| chunk.newlines);
        }
        BoundedOutput {
            text: sanitize_output(&kept.text),
            dropped_bytes: self.total_bytes - kept.bytes,
            dropped_lines: lines(self.total_newlines, self.ends_with_newline) - kept_lines,
        }
    }
}

/// Lines of text with `newlines` newlines; a final unterminated line counts.
fn lines(newlines: usize, terminated: bool) -> usize {
    newlines + usize::from(!terminated)
}

fn count_newlines(text: &str) -> usize {
    text.bytes().filter(|b| *b == NEWLINE).count()
}

/// Minimum pause between progress commits; each commit also buys a pause proportional to what it wrote.
const MIN_PROGRESS_INTERVAL_MS: u64 = 100;
const PROGRESS_BYTES_PER_SECOND: u64 = 100 * 1024;

/// A progress commit: resolves with the bytes it wrote.
pub type ProgressWrite = Arc<dyn Fn() -> BoxFuture<'static, Result<usize>> + Send + Sync>;
/// Receives a failed progress commit.
pub type ProgressError = Arc<dyn Fn(Error) + Send + Sync>;
/// Settles with the commit that includes a change.
pub type ProgressWaiter = oneshot::Sender<Result<()>>;

struct ProgressState {
    waiters: Vec<ProgressWaiter>,
    timer: Option<tokio::task::JoinHandle<()>>,
    in_flight: Option<Shared<BoxFuture<'static, ()>>>,
    next_at: Option<Instant>,
    dirty: bool,
    stopped: bool,
}

/// Adaptive progress commits, like the environment's shell output capture: the first change after an idle period
/// commits at once; each commit then delays the next by at least 100 ms and by its written size at 100 KiB/s. At most
/// one commit is in flight; changes made meanwhile coalesce into the next one.
#[derive(Clone)]
pub struct Progress {
    write: ProgressWrite,
    on_error: ProgressError,
    state: Arc<Mutex<ProgressState>>,
}

impl Progress {
    pub fn new(write: ProgressWrite, on_error: ProgressError) -> Self {
        Self {
            write,
            on_error,
            state: Arc::new(Mutex::new(ProgressState {
                waiters: Vec::new(),
                timer: None,
                in_flight: None,
                next_at: None,
                dirty: false,
                stopped: false,
            })),
        }
    }

    /// Schedule a commit.
    pub fn mark(&self) {
        self.state.lock().dirty = true;
        self.schedule();
    }

    /// Schedule a commit; the future settles with the commit that includes this change.
    pub fn mark_and_wait(&self) -> impl Future<Output = Result<()>> + Send + 'static {
        let (sender, receiver) = oneshot::channel();
        self.state.lock().waiters.push(sender);
        self.mark();
        async move {
            receiver
                .await
                .unwrap_or_else(|_| Err(Error::message("Progress stopped")))
        }
    }

    /// Stop committing and wait for the commit in flight; returns the waiters the final commit must settle.
    pub async fn stop(&self) -> Vec<ProgressWaiter> {
        let in_flight = {
            let mut state = self.state.lock();
            state.stopped = true;
            if let Some(timer) = state.timer.take() {
                timer.abort();
            }
            state.in_flight.clone()
        };
        if let Some(in_flight) = in_flight {
            in_flight.await;
        }
        std::mem::take(&mut self.state.lock().waiters)
    }

    fn schedule(&self) {
        let wait = {
            let state = self.state.lock();
            if state.stopped || state.timer.is_some() || state.in_flight.is_some() {
                return;
            }
            state
                .next_at
                .map(|at| at.saturating_duration_since(Instant::now()))
                .unwrap_or(Duration::ZERO)
        };
        if wait.is_zero() {
            self.flush();
            return;
        }
        let this = self.clone();
        let deadline = Instant::now() + wait;
        let timer = tokio::spawn(async move {
            tokio::time::sleep_until(deadline).await;
            this.state.lock().timer = None;
            this.flush();
        });
        let mut state = self.state.lock();
        if state.stopped {
            timer.abort();
        } else {
            state.timer = Some(timer);
        }
    }

    fn flush(&self) {
        let mut state = self.state.lock();
        if state.stopped || !state.dirty {
            return;
        }
        state.dirty = false;
        let waiters = std::mem::take(&mut state.waiters);
        let started = Instant::now();
        let write = (self.write)();
        let this = self.clone();
        let commit = async move {
            match write.await {
                Ok(bytes) => {
                    let pause = MIN_PROGRESS_INTERVAL_MS
                        .max(bytes as u64 * 1000 / PROGRESS_BYTES_PER_SECOND);
                    this.state.lock().next_at = Some(started + Duration::from_millis(pause));
                    for waiter in waiters {
                        let _ = waiter.send(Ok(()));
                    }
                }
                Err(error) => {
                    this.state.lock().next_at =
                        Some(started + Duration::from_millis(MIN_PROGRESS_INTERVAL_MS));
                    for waiter in waiters {
                        let _ = waiter.send(Err(error.clone()));
                    }
                    (this.on_error)(error);
                }
            }
            let dirty = {
                let mut state = this.state.lock();
                state.in_flight = None;
                state.dirty
            };
            if dirty {
                this.schedule();
            }
        }
        .boxed()
        .shared();
        state.in_flight = Some(commit.clone());
        drop(state);
        tokio::spawn(commit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_keeps_incomplete_characters_pending() {
        let mut decoder = Utf8Decoder::default();
        let bytes = "é".as_bytes();
        assert_eq!(decoder.decode(&bytes[..1], true), "");
        assert_eq!(decoder.decode(&bytes[1..], true), "é");
        assert_eq!(decoder.decode(&bytes[..1], true), "");
        assert_eq!(decoder.decode(&[], false), "\u{fffd}");
    }
}
