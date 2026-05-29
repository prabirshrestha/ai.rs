use agent::{
    TruncatedBy, TruncationOptions, format_size, truncate_head, truncate_line, truncate_tail,
};

#[test]
fn formats_byte_sizes() {
    assert_eq!(format_size(10), "10B");
    assert_eq!(format_size(1536), "1.5KB");
    assert_eq!(format_size(2 * 1024 * 1024), "2.0MB");
}

#[test]
fn truncate_head_keeps_complete_lines_by_line_limit() {
    let result = truncate_head(
        "one\ntwo\nthree",
        TruncationOptions {
            max_lines: 2,
            max_bytes: 100,
        },
    );

    assert_eq!(result.content, "one\ntwo");
    assert!(result.truncated);
    assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
    assert_eq!(result.total_lines, 3);
    assert_eq!(result.output_lines, 2);
}

#[test]
fn truncate_head_keeps_complete_lines_by_byte_limit() {
    let result = truncate_head(
        "one\ntwo\nthree",
        TruncationOptions {
            max_lines: 10,
            max_bytes: 6,
        },
    );

    assert_eq!(result.content, "one");
    assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    assert!(!result.last_line_partial);
}

#[test]
fn truncate_head_reports_first_line_exceeding_byte_limit() {
    let result = truncate_head(
        "abcdef\nsecond",
        TruncationOptions {
            max_lines: 10,
            max_bytes: 3,
        },
    );

    assert_eq!(result.content, "");
    assert_eq!(result.output_lines, 0);
    assert!(result.first_line_exceeds_limit);
}

#[test]
fn truncate_tail_keeps_complete_tail_lines() {
    let result = truncate_tail(
        "one\ntwo\nthree\n",
        TruncationOptions {
            max_lines: 2,
            max_bytes: 100,
        },
    );

    assert_eq!(result.content, "two\nthree");
    assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
    assert_eq!(result.total_lines, 3);
    assert_eq!(result.output_lines, 2);
}

#[test]
fn truncate_tail_keeps_partial_last_line_when_needed() {
    let result = truncate_tail(
        "abcdef",
        TruncationOptions {
            max_lines: 10,
            max_bytes: 3,
        },
    );

    assert_eq!(result.content, "def");
    assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    assert!(result.last_line_partial);
}

#[test]
fn truncate_tail_respects_utf8_boundaries() {
    let result = truncate_tail(
        "aé日",
        TruncationOptions {
            max_lines: 10,
            max_bytes: 4,
        },
    );

    assert_eq!(result.content, "日");
    assert_eq!(result.output_bytes, 3);
    assert!(result.last_line_partial);
}

#[test]
fn truncate_line_adds_suffix_when_over_limit() {
    assert_eq!(truncate_line("abc", Some(3)), ("abc".to_string(), false));
    assert_eq!(
        truncate_line("abcdef", Some(3)),
        ("abc... [truncated]".to_string(), true)
    );
}
