//! Port of `test/harness-output.test.ts`.
//!
//! Divergences: Rust strings cannot hold lone surrogates, so the surrogate edge cases are skipped and the fuzz
//! alphabet keeps only valid characters. Fake timers are Tokio's paused clock, whose timers fire on millisecond ticks,
//! so commit times are checked within one millisecond.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::Mutex;

use crate::durable::errors::Error;
use crate::durable::harness::output::{
    BoundedOutput, OutputBuffer, OutputLimits, Progress, bound_output, sanitize_output,
};
use crate::durable::harness::types::Retain;

fn head(max_lines: usize, max_bytes: usize) -> OutputLimits {
    OutputLimits {
        max_bytes,
        max_lines,
        retain: Retain::Head,
    }
}

fn tail(max_lines: usize, max_bytes: usize) -> OutputLimits {
    OutputLimits {
        max_bytes,
        max_lines,
        retain: Retain::Tail,
    }
}

fn buffer_tail(content: &str, max_bytes: usize) -> String {
    let bytes = content.as_bytes();
    if bytes.len() <= max_bytes {
        return content.to_string();
    }
    let mut start = bytes.len() - max_bytes;
    while start < bytes.len() && (bytes[start] & 0xc0) == 0x80 {
        start += 1;
    }
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

/// A single line longer than the byte limit is cut like Buffer's tail slice on a character boundary.
fn assert_matches_buffer_tail(input: &str, max_byte_values: Option<Vec<usize>>) {
    let values = max_byte_values.unwrap_or_else(|| (0..input.len() + 5).collect());
    for max_bytes in values {
        let kept = bound_output(input, tail(10, max_bytes)).text;
        let expected = buffer_tail(input, max_bytes);
        assert_eq!(
            kept, expected,
            "tail mismatch input={input:?} maxBytes={max_bytes}"
        );
        assert!(
            kept.len() <= max_bytes,
            "tail output exceeded {max_bytes} bytes"
        );
    }
}

fn sampled_byte_limits(input: &str) -> Vec<usize> {
    let total = input.len() as isize;
    let mut candidates: Vec<usize> = [
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        total / 2,
        total - 4,
        total - 1,
        total,
        total + 1,
    ]
    .into_iter()
    .filter(|value| *value >= 0)
    .map(|value| value as usize)
    .collect();
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn bound(text: &str, limits: OutputLimits) -> (String, usize, usize) {
    let slice = bound_output(text, limits);
    (slice.text, slice.dropped_bytes, slice.dropped_lines)
}

fn kept(text: &str, dropped_bytes: usize, dropped_lines: usize) -> (String, usize, usize) {
    (text.to_string(), dropped_bytes, dropped_lines)
}

#[test]
fn removes_control_characters_but_keeps_tabs_newlines_and_other_text() {
    assert_eq!(
        sanitize_output("a\0b\tc\nd\re\u{7}f\u{fff9}g\u{fffb}h😀"),
        "ab\tc\ndefgh😀"
    );
}

#[test]
fn keeps_output_within_the_limits_unchanged() {
    assert_eq!(bound("a\nb\n", head(2, 1000)), kept("a\nb\n", 0, 0));
    assert_eq!(bound("a\nb", tail(2, 1000)), kept("a\nb", 0, 0));
    assert_eq!(bound("", tail(2, 1000)), kept("", 0, 0));
}

#[test]
fn keeps_nothing_with_a_zero_limit() {
    assert_eq!(bound("ab\ncd\n", head(10, 0)), kept("", 6, 2));
    assert_eq!(bound("ab\ncd\n", tail(0, 1000)), kept("", 6, 2));
}

#[test]
fn keeps_exact_slices_of_whole_lines_trailing_newline_included() {
    assert_eq!(bound("a\nb\nc\n", head(2, 1000)), kept("a\nb\n", 2, 1));
    assert_eq!(bound("a\nb\nc\n", tail(2, 1000)), kept("b\nc\n", 2, 1));
    assert_eq!(bound("a\nb\nc", tail(2, 1000)), kept("b\nc", 2, 1));
    // Blank lines are lines.
    assert_eq!(bound("a\nb\nc\n\n", tail(3, 1000)), kept("b\nc\n\n", 2, 1));
}

#[test]
fn cuts_at_the_byte_limit_on_whole_lines_when_possible() {
    assert_eq!(bound("aa\nbb\ncc\n", head(10, 7)), kept("aa\nbb\n", 3, 1));
    assert_eq!(bound("aa\nbb\ncc\n", tail(10, 7)), kept("bb\ncc\n", 3, 1));
}

#[test]
fn cuts_a_single_line_longer_than_the_byte_limit_on_a_character_boundary() {
    // "é" is two bytes; five bytes hold two whole characters.
    assert_eq!(bound("ééé\n", head(10, 5)), kept("éé", 3, 0));
    assert_eq!(bound("x\néééé", tail(10, 5)), kept("éé", 6, 1));
}

#[test]
fn cuts_tails_exactly_like_buffer_across_deterministic_fuzz_cases() {
    let alphabet = [
        "a",
        "\u{7f}",
        "\u{80}",
        "é",
        "\u{7ff}",
        "\u{800}",
        "中",
        "\u{d7ff}",
        "🙂",
        "\u{e000}",
        "\u{ffff}",
        "👩‍💻",
    ];
    fn check_exhaustive(alphabet: &[&str], prefix: &str, depth: usize) {
        assert_matches_buffer_tail(prefix, Some(sampled_byte_limits(prefix)));
        if depth == 0 {
            return;
        }
        for character in alphabet {
            check_exhaustive(alphabet, &format!("{prefix}{character}"), depth - 1);
        }
    }
    check_exhaustive(&alphabet, "", 3);
    let mut seed: u32 = 0x12345678;
    let mut random = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        f64::from(seed) / 4294967296.0
    };
    for _ in 0..1000 {
        let length = (random() * 80.0) as usize;
        let mut input = String::new();
        for _ in 0..length {
            input.push_str(alphabet[(random() * alphabet.len() as f64) as usize]);
        }
        assert_matches_buffer_tail(&input, Some(sampled_byte_limits(&input)));
    }
    for input in ["a🙂", "🙂b", "👩‍💻"] {
        assert_matches_buffer_tail(input, None);
    }
}

fn output(text: &str, dropped_bytes: usize, dropped_lines: usize) -> BoundedOutput {
    BoundedOutput {
        text: text.to_string(),
        dropped_bytes,
        dropped_lines,
    }
}

#[test]
fn keeps_exact_totals_across_chunks_and_decodes_utf8_split_across_byte_chunks() {
    let mut buffer = OutputBuffer::new(tail(2, 1000));
    let bytes = "😀\n".as_bytes();
    buffer.push("a\nb\n");
    buffer.push_bytes(&bytes[..2]);
    buffer.push_bytes(&bytes[2..]);
    assert_eq!(buffer.snapshot(), output("b\n😀\n", 2, 1));
}

#[test]
fn sanitizes_the_retained_text_but_counts_the_raw_stream() {
    let mut buffer = OutputBuffer::new(tail(1, 1000));
    buffer.push("a\u{7}\n");
    assert_eq!(buffer.snapshot(), output("a\n", 0, 0));
    buffer.push("b\u{1b}\n");
    assert_eq!(buffer.snapshot(), output("b\n", 3, 1));
}

#[test]
fn flushes_an_incomplete_character_before_a_string_chunk_and_at_the_end() {
    let mut buffer = OutputBuffer::new(tail(10, 1000));
    let euro = "€".as_bytes();
    buffer.push_bytes(&euro[..1]);
    buffer.push("x");
    buffer.push_bytes(&euro[..2]);
    buffer.end();
    assert_eq!(buffer.snapshot().text, "\u{fffd}x\u{fffd}");
}

#[test]
fn matches_bounding_the_whole_stream_when_several_chunks_arrive_between_snapshots() {
    for limits in [head(3, 40), tail(3, 40), head(50, 25), tail(50, 25)] {
        let mut buffer = OutputBuffer::new(limits);
        let mut stream = String::new();
        for index in 0..300 {
            let chunk = if index % 7 == 0 {
                format!("{}\n", "é".repeat(index % 30))
            } else {
                format!("line {index}\n")
            };
            stream.push_str(&chunk);
            buffer.push(&chunk);
            if index % 5 != 4 {
                continue;
            }
            let expected = bound_output(&stream, limits);
            assert_eq!(
                buffer.snapshot(),
                output(
                    &expected.text,
                    expected.dropped_bytes,
                    expected.dropped_lines
                )
            );
        }
    }
}

#[test]
fn stops_storing_head_output_once_the_window_is_full() {
    let mut buffer = OutputBuffer::new(head(2, 1000));
    for index in 0..1000 {
        buffer.push(&format!("line {index}\n"));
    }
    assert!(buffer.stored_bytes() < 20);
    assert_eq!(buffer.snapshot(), output("line 0\nline 1\n", 8876, 998));
}

#[test]
fn stores_only_the_tail_window_after_each_snapshot() {
    let mut buffer = OutputBuffer::new(tail(3, 100));
    let mut stream = String::new();
    for index in 0..2000 {
        let chunk = format!("line {index}\n\n");
        stream.push_str(&chunk);
        buffer.push(&chunk);
        let snapshot = buffer.snapshot();
        assert!(buffer.stored_bytes() <= 100);
        assert_eq!(snapshot.text, bound_output(&stream, tail(3, 100)).text);
    }
}

async fn advance(ms: u64) {
    tokio::time::advance(Duration::from_millis(ms)).await;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
}

#[tokio::test(start_paused = true)]
async fn commits_the_first_change_at_once_then_waits_at_least_100_ms_and_the_written_size_at_100_kib_per_s()
 {
    let start = tokio::time::Instant::now();
    let commits: Arc<Mutex<Vec<u64>>> = Arc::default();
    let size = Arc::new(AtomicUsize::new(50 * 1024));
    let (log, written) = (commits.clone(), size.clone());
    let progress = Progress::new(
        Arc::new(move || {
            log.lock().push(start.elapsed().as_millis() as u64);
            let size = written.load(Ordering::SeqCst);
            Box::pin(async move { Ok(size) })
        }),
        Arc::new(|_| {}),
    );
    progress.mark();
    advance(0).await;
    assert_eq!(*commits.lock(), [0]);
    // 50 KiB buys 500 ms; changes meanwhile coalesce into one commit.
    size.store(10, Ordering::SeqCst);
    progress.mark();
    progress.mark();
    advance(499).await;
    assert_eq!(*commits.lock(), [0]);
    // Tokio rounds timer deadlines up to the next millisecond tick.
    advance(2).await;
    let second = commits.lock()[1];
    assert_eq!(commits.lock().len(), 2);
    assert!((500..=501).contains(&second));
    // A small commit still waits the minimum 100 ms.
    progress.mark();
    advance(98).await;
    assert_eq!(commits.lock().len(), 2);
    advance(2).await;
    assert_eq!(commits.lock().len(), 3);
    assert!((second + 100..=second + 101).contains(&commits.lock()[2]));
}

#[tokio::test]
async fn rejects_the_waiters_of_a_failed_commit_and_reports_its_error() {
    let errors: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = errors.clone();
    let progress = Progress::new(
        Arc::new(|| Box::pin(async { Err(Error::message("commit failed")) })),
        Arc::new(move |error| sink.lock().push(error.to_string())),
    );
    let error = progress.mark_and_wait().await.unwrap_err();
    assert_eq!(error.to_string(), "commit failed");
    assert_eq!(*errors.lock(), ["commit failed"]);
}

#[tokio::test(start_paused = true)]
async fn stops_waits_for_the_commit_in_flight_and_hands_back_waiters_no_commit_covered_yet() {
    let gate = crate::durable::session::tests::support::Deferred::default();
    let release = gate.clone();
    let progress = Progress::new(
        Arc::new(move || {
            let gate = gate.clone();
            Box::pin(async move {
                gate.wait().await;
                Ok(0)
            })
        }),
        Arc::new(|_| {}),
    );
    let first = progress.mark_and_wait();
    let second = progress.mark_and_wait();
    let stopping = {
        let progress = progress.clone();
        tokio::spawn(async move { progress.stop().await })
    };
    tokio::task::yield_now().await;
    release.resolve();
    let pending = stopping.await.unwrap();
    first.await.unwrap();
    assert_eq!(pending.len(), 1);
    for waiter in pending {
        let _ = waiter.send(Ok(()));
    }
    second.await.unwrap();
}
