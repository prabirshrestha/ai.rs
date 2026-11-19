//! Port of `harness-pi/src/pi-tool-result.ts`, with the truncation
//! helpers it imports from `@earendil-works/pi-coding-agent`
//! (`core/tools/truncate.js`: `DEFAULT_MAX_BYTES`, `formatSize`,
//! `truncateHead`, `truncateTail`).

use crate::harness::utils::HarnessToolModelOutputContent;

/// Pi's `DEFAULT_MAX_LINES`.
pub const DEFAULT_MAX_LINES: usize = 2000;
/// Pi's `DEFAULT_MAX_BYTES` (50KB).
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// Pi's `TruncationResult['truncatedBy']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

/// Pi's `TruncationResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncationResult {
    pub content: String,
    pub truncated: bool,
    pub truncated_by: Option<TruncatedBy>,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub output_lines: usize,
    pub output_bytes: usize,
    pub last_line_partial: bool,
    pub first_line_exceeds_limit: bool,
    pub max_lines: usize,
    pub max_bytes: usize,
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

/// Pi's `formatSize(bytes)`.
pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn untruncated(content: &str, total_lines: usize) -> TruncationResult {
    TruncationResult {
        content: content.to_string(),
        truncated: false,
        truncated_by: None,
        total_lines,
        total_bytes: content.len(),
        output_lines: total_lines,
        output_bytes: content.len(),
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines: DEFAULT_MAX_LINES,
        max_bytes: DEFAULT_MAX_BYTES,
    }
}

/// Pi's `truncateHead(content)`: keep the first lines that fit. Never
/// returns partial lines.
pub fn truncate_head(content: &str) -> TruncationResult {
    let (max_lines, max_bytes) = (DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();
    if total_lines <= max_lines && total_bytes <= max_bytes {
        return untruncated(content, total_lines);
    }
    if lines[0].len() > max_bytes {
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
    let mut output: Vec<&str> = Vec::new();
    let mut output_bytes = 0;
    let mut truncated_by = TruncatedBy::Lines;
    for (index, line) in lines.iter().take(max_lines).enumerate() {
        let line_bytes = line.len() + usize::from(index > 0);
        if output_bytes + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        output.push(line);
        output_bytes += line_bytes;
    }
    if output.len() >= max_lines && output_bytes <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }
    let content = output.join("\n");
    TruncationResult {
        output_bytes: content.len(),
        content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output.len(),
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

fn truncate_string_to_bytes_from_end(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len() - max_bytes;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

/// Pi's `truncateTail(content)`: keep the last lines that fit. May return
/// a partial first line if the last line exceeds the byte limit.
pub fn truncate_tail(content: &str) -> TruncationResult {
    let (max_lines, max_bytes) = (DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();
    if total_lines <= max_lines && total_bytes <= max_bytes {
        return untruncated(content, total_lines);
    }
    let mut output: Vec<&str> = Vec::new();
    let mut output_bytes = 0;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;
    for line in lines.iter().rev() {
        if output.len() >= max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(!output.is_empty());
        if output_bytes + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            if output.is_empty() {
                let truncated = truncate_string_to_bytes_from_end(line, max_bytes);
                output.insert(0, truncated);
                output_bytes = truncated.len();
                last_line_partial = true;
            }
            break;
        }
        output.insert(0, line);
        output_bytes += line_bytes;
    }
    if output.len() >= max_lines && output_bytes <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }
    let content = output.join("\n");
    TruncationResult {
        output_bytes: content.len(),
        content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output.len(),
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

fn join_content_and_notice(content: &str, notice: &str) -> String {
    if content.is_empty() {
        format!("[{notice}]")
    } else {
        format!("{content}\n\n[{notice}]")
    }
}

fn truncate_utf8_start(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn limit_suffix(truncation: &TruncationResult) -> String {
    if truncation.truncated_by == Some(TruncatedBy::Bytes) {
        format!(" ({} limit)", format_size(truncation.max_bytes))
    } else {
        String::new()
    }
}

fn get_pi_tool_output_head_truncation(text: &str, continuation: &str) -> (String, Option<String>) {
    let truncation = truncate_head(text);
    if !truncation.truncated {
        return (text.to_string(), None);
    }
    if truncation.first_line_exceeds_limit {
        let content = truncate_utf8_start(text, DEFAULT_MAX_BYTES);
        let notice = format!(
            "Output truncated: showing the first {} of a {} line. {continuation}",
            format_size(content.len()),
            format_size(truncation.total_bytes)
        );
        return (content.to_string(), Some(notice));
    }
    let notice = format!(
        "Output truncated: showing the first {} of {} lines{}. {continuation}",
        truncation.output_lines,
        truncation.total_lines,
        limit_suffix(&truncation)
    );
    (truncation.content, Some(notice))
}

/// `truncatePiToolOutputHead(text, continuation)`.
pub fn truncate_pi_tool_output_head(text: &str, continuation: &str) -> String {
    match get_pi_tool_output_head_truncation(text, continuation) {
        (retained, None) => retained,
        (retained, Some(notice)) => join_content_and_notice(&retained, &notice),
    }
}

/// `truncatePiToolModelOutput({ content, continuation })`.
///
/// [rust] TS returns the input array itself when nothing is truncated;
/// this returns an equal copy.
pub fn truncate_pi_tool_model_output(
    content: &[HarnessToolModelOutputContent],
    continuation: &str,
) -> Vec<HarnessToolModelOutputContent> {
    let text = content
        .iter()
        .filter_map(|part| match part {
            HarnessToolModelOutputContent::Text { text } => Some(text.as_str()),
            HarnessToolModelOutputContent::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let (retained, Some(notice)) = get_pi_tool_output_head_truncation(&text, continuation) else {
        return content.to_vec();
    };
    let mut truncated = Vec::new();
    let mut offset = 0;
    let mut last_text_index = None;
    for part in content {
        let HarnessToolModelOutputContent::Text { text } = part else {
            truncated.push(part.clone());
            continue;
        };
        let start = offset.min(retained.len());
        let end = (offset + text.len()).min(retained.len());
        let prefix = &retained[start..end];
        offset += text.len() + 1;
        if !prefix.is_empty() {
            last_text_index = Some(truncated.len());
            truncated.push(HarnessToolModelOutputContent::Text {
                text: prefix.to_string(),
            });
        }
    }
    match last_text_index {
        Some(index) => {
            if let HarnessToolModelOutputContent::Text { text } = &mut truncated[index] {
                *text = join_content_and_notice(text, &notice);
            }
        }
        None => {
            let index = content
                .iter()
                .position(|part| matches!(part, HarnessToolModelOutputContent::Text { .. }))
                .unwrap_or(0);
            truncated.insert(
                index,
                HarnessToolModelOutputContent::Text {
                    text: join_content_and_notice("", &notice),
                },
            );
        }
    }
    truncated
}

/// `truncatePiToolOutputTail(text, continuation)`.
pub fn truncate_pi_tool_output_tail(text: &str, continuation: &str) -> String {
    let truncation = truncate_tail(text);
    if !truncation.truncated {
        return text.to_string();
    }
    let notice = format!(
        "Output truncated: showing the last {} of {} lines{}. {continuation}",
        truncation.output_lines,
        truncation.total_lines,
        limit_suffix(&truncation)
    );
    join_content_and_notice(&truncation.content, &notice)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTINUATION: &str = "Call the tool again with narrower parameters.";

    fn image() -> HarnessToolModelOutputContent {
        HarnessToolModelOutputContent::Image {
            data: "iVBORw==".into(),
            media_type: "image/png".into(),
        }
    }

    fn text(text: impl Into<String>) -> HarnessToolModelOutputContent {
        HarnessToolModelOutputContent::Text { text: text.into() }
    }

    fn texts(parts: &[HarnessToolModelOutputContent]) -> String {
        parts
            .iter()
            .filter_map(|part| match part {
                HarnessToolModelOutputContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn preserves_untruncated_content() {
        for content in [vec![], vec![image()], vec![text("small"), image()]] {
            assert_eq!(truncate_pi_tool_model_output(&content, CONTINUATION), content);
        }
    }

    #[test]
    fn bounds_a_single_oversized_text_part_safely() {
        for character in ["x", "😀", "€"] {
            let value = character.repeat(100_000);
            let content = vec![text(value.clone())];
            let result = truncate_pi_tool_model_output(&content, CONTINUATION);
            assert_eq!(
                result,
                vec![text(truncate_pi_tool_output_head(&value, CONTINUATION))]
            );
            assert!(!serde_json::to_string(&result).unwrap().contains('�'));
        }
    }

    #[test]
    fn shares_the_budget_across_text_parts_and_preserves_every_image() {
        let cases = [
            ("bytes", "x".repeat(30_000), "y".repeat(30_000)),
            (
                "lines",
                vec!["first"; 1_501].join("\n"),
                vec!["second"; 1_001].join("\n"),
            ),
        ];
        for (limit, first, second) in cases {
            let content = vec![
                image(),
                text(first.clone()),
                image(),
                text(second.clone()),
                image(),
                text("omitted"),
                image(),
            ];
            let result = truncate_pi_tool_model_output(&content, CONTINUATION);
            let joined = texts(&result);
            assert_eq!(
                joined,
                truncate_pi_tool_output_head(&format!("{first}\n{second}\nomitted"), CONTINUATION)
            );
            assert_eq!(joined.matches("Output truncated").count(), 1);
            assert_eq!(
                result
                    .iter()
                    .filter(|part| matches!(part, HarnessToolModelOutputContent::Image { .. }))
                    .count(),
                4
            );
            assert_eq!(result[0], image());
            assert_eq!(result.last(), Some(&image()));
            let kinds: Vec<&str> = result
                .iter()
                .map(|part| match part {
                    HarnessToolModelOutputContent::Text { .. } => "text",
                    HarnessToolModelOutputContent::Image { .. } => "image",
                })
                .collect();
            let expected = if limit == "bytes" {
                vec!["image", "text", "image", "image", "image"]
            } else {
                vec!["image", "text", "image", "text", "image", "image"]
            };
            assert_eq!(kinds, expected, "{limit}");
        }
    }

    #[test]
    fn inserts_a_notice_at_the_first_text_position_when_no_text_is_retained() {
        let big = "x".repeat(60_000);
        let content = vec![image(), text(""), image(), text(big.clone()), image()];
        assert_eq!(
            truncate_pi_tool_model_output(&content, CONTINUATION),
            vec![
                image(),
                text(truncate_pi_tool_output_head(&format!("\n{big}"), CONTINUATION)),
                image(),
                image(),
            ]
        );
    }

    #[test]
    fn keeps_small_tool_results_unchanged() {
        assert_eq!(
            truncate_pi_tool_output_head("small output", "Continue another way."),
            "small output"
        );
        assert_eq!(
            truncate_pi_tool_output_tail("small output", "Continue another way."),
            "small output"
        );
    }

    #[test]
    fn bounds_single_line_head_output_and_adds_continuation_guidance() {
        let result = truncate_pi_tool_output_head(&"x".repeat(1_000_000), CONTINUATION);
        assert!(result.len() < 64 * 1024);
        assert!(result.contains("Output truncated"));
        assert!(result.contains(CONTINUATION));
    }

    #[test]
    fn keeps_the_tail_of_oversized_command_output() {
        let result = truncate_pi_tool_output_tail(
            &format!("{}last", "first\n".repeat(10_000)),
            "Redirect the command to a file.",
        );
        assert!(result.len() < 64 * 1024);
        assert!(!result.contains(&"first\n".repeat(2_000)));
        assert!(result.contains("last"));
        assert!(result.contains("Output truncated"));
        assert!(result.contains("Redirect the command to a file."));
    }
}
