pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
pub const GREP_MAX_LINE_LENGTH: usize = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncationOptions {
    pub max_lines: usize,
    pub max_bytes: usize,
}

impl Default for TruncationOptions {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn truncate_head(
    content: &str,
    options: impl Into<Option<TruncationOptions>>,
) -> TruncationResult {
    let options = options.into().unwrap_or_default();
    let total_bytes = utf8_byte_length(content);
    let lines = content.split('\n').collect::<Vec<_>>();
    let total_lines = lines.len();

    if total_lines <= options.max_lines && total_bytes <= options.max_bytes {
        return unchanged(content, total_lines, total_bytes, options);
    }

    let first_line_bytes = utf8_byte_length(lines.first().copied().unwrap_or_default());
    if first_line_bytes > options.max_bytes {
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
            max_lines: options.max_lines,
            max_bytes: options.max_bytes,
        };
    }

    let mut output_lines = Vec::new();
    let mut output_bytes_count = 0;
    let mut truncated_by = TruncatedBy::Lines;

    for (index, line) in lines.iter().enumerate().take(options.max_lines) {
        let line_bytes = utf8_byte_length(line) + usize::from(index > 0);
        if output_bytes_count + line_bytes > options.max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        output_lines.push(*line);
        output_bytes_count += line_bytes;
    }

    if output_lines.len() >= options.max_lines && output_bytes_count <= options.max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output_lines.join("\n");
    let final_output_bytes = utf8_byte_length(&output_content);

    TruncationResult {
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output_lines.len(),
        output_bytes: final_output_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

pub fn truncate_tail(
    content: &str,
    options: impl Into<Option<TruncationOptions>>,
) -> TruncationResult {
    let options = options.into().unwrap_or_default();
    let total_bytes = utf8_byte_length(content);
    let mut lines = content.split('\n').collect::<Vec<_>>();
    if lines.len() > 1 && lines.last() == Some(&"") {
        lines.pop();
    }
    let total_lines = lines.len();

    if total_lines <= options.max_lines && total_bytes <= options.max_bytes {
        return unchanged(content, total_lines, total_bytes, options);
    }

    let mut output_lines = Vec::new();
    let mut output_bytes_count = 0;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;

    for line in lines.iter().rev().take(options.max_lines) {
        let line_bytes = utf8_byte_length(line) + usize::from(!output_lines.is_empty());
        if output_bytes_count + line_bytes > options.max_bytes {
            truncated_by = TruncatedBy::Bytes;
            if output_lines.is_empty() {
                let truncated_line = truncate_string_to_bytes_from_end(line, options.max_bytes);
                output_bytes_count = utf8_byte_length(&truncated_line);
                output_lines.insert(0, truncated_line);
                last_line_partial = true;
            }
            break;
        }
        output_lines.insert(0, (*line).to_string());
        output_bytes_count += line_bytes;
    }

    if output_lines.len() >= options.max_lines && output_bytes_count <= options.max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output_lines.join("\n");
    let final_output_bytes = utf8_byte_length(&output_content);

    TruncationResult {
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output_lines.len(),
        output_bytes: final_output_bytes,
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

pub fn truncate_line(line: &str, max_chars: Option<usize>) -> (String, bool) {
    let max_chars = max_chars.unwrap_or(GREP_MAX_LINE_LENGTH);
    if line.chars().count() <= max_chars {
        return (line.to_string(), false);
    }
    let truncated = line.chars().take(max_chars).collect::<String>();
    (format!("{truncated}... [truncated]"), true)
}

fn unchanged(
    content: &str,
    total_lines: usize,
    total_bytes: usize,
    options: TruncationOptions,
) -> TruncationResult {
    TruncationResult {
        content: content.to_string(),
        truncated: false,
        truncated_by: None,
        total_lines,
        total_bytes,
        output_lines: total_lines,
        output_bytes: total_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

fn truncate_string_to_bytes_from_end(value: &str, max_bytes: usize) -> String {
    if max_bytes == 0 {
        return String::new();
    }

    let mut output_bytes = 0;
    let mut start = value.len();
    for (index, character) in value.char_indices().rev() {
        let character_bytes = character.len_utf8();
        if output_bytes + character_bytes > max_bytes {
            break;
        }
        output_bytes += character_bytes;
        start = index;
    }
    value[start..].to_string()
}

fn utf8_byte_length(content: &str) -> usize {
    content.len()
}
