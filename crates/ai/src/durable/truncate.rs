//! Port of durable `src/truncate.ts`: shared truncation utilities for tool
//! outputs.
//!
//! Truncation is based on two independent limits - whichever is hit first wins:
//! - Line limit (default: 2000 lines)
//! - Byte limit (default: 50KB)
//!
//! Never returns partial lines. Tool output streams are bounded by the harness
//! output module instead.
//!
//! Divergences from Pi:
//! - Rust strings are UTF-8, so [`utf8_byte_length`] is `str::len`; the
//!   no-`Buffer` fallback (and its lone-surrogate cases) has no counterpart.
//! - `truncatedBy: null` is `None`; sizes and counts are `usize`.

/// `DEFAULT_MAX_LINES`.
pub const DEFAULT_MAX_LINES: usize = 2000;
/// `DEFAULT_MAX_BYTES` (50KB).
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// Which limit was hit (`"lines" | "bytes"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

impl TruncatedBy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lines => "lines",
            Self::Bytes => "bytes",
        }
    }
}

/// `TruncationResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncationResult {
    /// The truncated content.
    pub content: String,
    /// Whether truncation occurred.
    pub truncated: bool,
    /// Which limit was hit, or `None` if not truncated.
    pub truncated_by: Option<TruncatedBy>,
    /// Total number of lines in the original content.
    pub total_lines: usize,
    /// Total number of bytes in the original content.
    pub total_bytes: usize,
    /// Number of complete lines in the truncated output.
    pub output_lines: usize,
    /// Number of bytes in the truncated output.
    pub output_bytes: usize,
    /// Whether the last line was partially truncated (only for tail truncation edge case).
    pub last_line_partial: bool,
    /// Whether the first line exceeded the byte limit (for head truncation).
    pub first_line_exceeds_limit: bool,
    /// The max lines limit that was applied.
    pub max_lines: usize,
    /// The max bytes limit that was applied.
    pub max_bytes: usize,
}

/// `TruncationOptions`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TruncationOptions {
    /// Maximum number of lines (default: 2000).
    pub max_lines: Option<usize>,
    /// Maximum number of bytes (default: 50KB).
    pub max_bytes: Option<usize>,
}

/// `utf8ByteLength`.
pub fn utf8_byte_length(content: &str) -> usize {
    content.len()
}

fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// `toFixed(1)` of `numerator / denominator`, rounding the exact quotient half up.
fn to_fixed_1(numerator: u64, denominator: u64) -> String {
    let tenths =
        (u128::from(numerator) * 10 + u128::from(denominator) / 2) / u128::from(denominator);
    format!("{}.{}", tenths / 10, tenths % 10)
}

/// Format bytes as human-readable size (`formatSize`).
pub fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{}KB", to_fixed_1(bytes, 1024))
    } else {
        format!("{}MB", to_fixed_1(bytes, 1024 * 1024))
    }
}

/// Truncate content from the head (keep first N lines/bytes); `truncateHead`.
/// Suitable for file reads where you want to see the beginning.
///
/// Never returns partial lines. If first line exceeds byte limit,
/// returns empty content with `first_line_exceeds_limit = true`.
pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines.unwrap_or(DEFAULT_MAX_LINES);
    let max_bytes = options.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);

    let total_bytes = utf8_byte_length(content);
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    // Check if no truncation needed
    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncationResult {
            content: content.to_owned(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            output_bytes: total_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
            max_lines,
            max_bytes,
        };
    }

    // Check if first line alone exceeds byte limit
    let first_line_bytes = utf8_byte_length(lines.first().copied().unwrap_or_default());
    if first_line_bytes > max_bytes {
        return TruncationResult {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines,
            total_bytes,
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
            max_lines,
            max_bytes,
        };
    }

    // Collect complete lines that fit
    let mut output_lines_arr: Vec<&str> = Vec::new();
    let mut output_bytes_count = 0;
    let mut truncated_by = TruncatedBy::Lines;

    for (index, line) in lines.iter().enumerate().take(max_lines) {
        let line_bytes = utf8_byte_length(line) + usize::from(index > 0); // +1 for newline

        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }

        output_lines_arr.push(line);
        output_bytes_count += line_bytes;
    }

    // Without a byte break, only omitted lines prove the line limit was reached; otherwise a trailing newline exceeded bytes.
    if truncated_by != TruncatedBy::Bytes {
        truncated_by = if output_lines_arr.len() < total_lines {
            TruncatedBy::Lines
        } else {
            TruncatedBy::Bytes
        };
    }

    let output_content = output_lines_arr.join("\n");
    let final_output_bytes = utf8_byte_length(&output_content);

    TruncationResult {
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output_lines_arr.len(),
        output_bytes: final_output_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Port of `test/env-truncate.test.ts`. The no-`Buffer` runtime case has no Rust counterpart.
#[cfg(test)]
mod tests {
    use super::*;

    fn options(max_bytes: usize, max_lines: usize) -> TruncationOptions {
        TruncationOptions {
            max_lines: Some(max_lines),
            max_bytes: Some(max_bytes),
        }
    }

    #[test]
    fn reports_utf8_byte_counts_in_truncation_results() {
        let content = "aé🙂\nb";
        let result = truncate_head(content, options(100, 10));

        assert!(!result.truncated);
        assert_eq!(result.total_bytes, content.len());
        assert_eq!(result.output_bytes, content.len());
        assert_eq!(result.total_bytes, 9);
    }

    #[test]
    fn truncates_multibyte_content_on_byte_limits() {
        // The truncation half of the no-Buffer runtime case.
        let result = truncate_head("aé🙂\nb", options(7, 10));
        assert_eq!(result.content, "aé🙂");
        assert_eq!(result.output_bytes, 7);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        for (input, length) in [("", 0), ("ascii", 5), ("é", 2), ("中", 3), ("🙂", 4)] {
            assert_eq!(utf8_byte_length(input), length);
        }
    }

    #[test]
    fn does_not_count_a_trailing_newline_as_an_extra_line() {
        let content = format!("{}\n", ["line"; 3].join("\n"));
        let head = truncate_head(&content, options(100, 3));

        assert!(!head.truncated);
        assert_eq!(head.total_lines, 3);
        assert_eq!(head.output_lines, 3);
    }

    #[test]
    fn truncates_head_by_line_limits() {
        let result = truncate_head("one\ntwo\nthree\nfour", options(100, 2));
        assert_eq!(result.content, "one\ntwo");
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(result.total_lines, 4);
        assert_eq!(result.output_lines, 2);
    }

    #[test]
    fn reports_bytes_when_only_a_trailing_newline_exceeds_limits_at_the_line_cap() {
        let result = truncate_head("hello\nworld\n", options(11, 2));
        assert_eq!(result.content, "hello\nworld");
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert_eq!(result.total_lines, 2);
        assert_eq!(result.output_lines, 2);
    }

    #[test]
    fn truncates_head_on_utf8_byte_limits_without_partial_lines() {
        let result = truncate_head("éé\nabc", options(4, 10));

        assert_eq!(result.content, "éé");
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert_eq!(result.output_bytes, 4);
        assert!(!result.first_line_exceeds_limit);
    }

    #[test]
    fn reports_head_truncation_when_the_first_line_exceeds_the_byte_limit() {
        let result = truncate_head("éé\nabc", options(3, 10));

        assert_eq!(result.content, "");
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert!(result.first_line_exceeds_limit);
    }

    #[test]
    fn formats_sizes() {
        assert_eq!(format_size(1023), "1023B");
        assert_eq!(format_size(1536), "1.5KB");
        assert_eq!(format_size(3 * 1024 * 1024), "3.0MB");
        // toFixed rounds the exact quotient half up.
        assert_eq!(format_size(1024 + 51), "1.0KB");
        assert_eq!(format_size(1024 + 52), "1.1KB");
    }
}
