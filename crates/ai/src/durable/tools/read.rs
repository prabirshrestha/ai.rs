//! Port of durable `src/tools/read.ts`.
//!
//! Divergences from Pi:
//! - The TypeBox schema is a JSON Schema literal; `ReadToolInput` is a serde struct and `ReadToolDetails` is built as
//!   JSON (`{ truncation?: Omit<TruncationResult, "content"> }`, camelCase keys in TS order).
//! - `new TextDecoder().decode(bytes)` is a lossy UTF-8 decode that drops a leading byte order mark, as the
//!   decoder does.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::chord::{Context, JsonValue};
use crate::durable::env::get_or_throw;
use crate::durable::errors::{Error, Result};
use crate::durable::harness::define::define_tool;
use crate::durable::harness::output::character_end;
use crate::durable::harness::tool::ToolExecutionApi;
use crate::durable::harness::types::{
    DiagnosticSeverity, ToolDiagnostic, ToolExecutionResult, ToolRegistration,
};
use crate::durable::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationOptions, TruncationResult,
    format_size, truncate_head,
};
use crate::types::UserContent;

use super::env::require_env;
use super::image::detect_supported_image_mime_type;
use super::path_utils::resolve_read_tool_path;
use super::{js_number, js_slice_index, parse_args};

fn read_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to the file to read (relative or absolute)" },
            "offset": { "type": "number", "description": "Line number to start reading from (1-indexed)" },
            "limit": { "type": "number", "description": "Maximum number of lines to read" },
        },
        "required": ["path"],
    })
}

/// `ReadToolInput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadToolInput {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
}

/// `Omit<TruncationResult, "content">` as JSON, in TS key order.
fn truncation_json(truncation: &TruncationResult) -> JsonValue {
    json!({
        "truncated": truncation.truncated,
        "truncatedBy": truncation.truncated_by.map(TruncatedBy::as_str),
        "totalLines": truncation.total_lines,
        "totalBytes": truncation.total_bytes,
        "outputLines": truncation.output_lines,
        "outputBytes": truncation.output_bytes,
        "lastLinePartial": truncation.last_line_partial,
        "firstLineExceedsLimit": truncation.first_line_exceeds_limit,
        "maxLines": truncation.max_lines,
        "maxBytes": truncation.max_bytes,
    })
}

/// `new TextDecoder().decode(bytes)`.
fn decode(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    match text.strip_prefix('\u{FEFF}') {
        Some(rest) => rest.to_string(),
        None => text.into_owned(),
    }
}

/// Reads text files. Remarks about truncation and continuation are diagnostics; the content is only file text.
pub fn create_read_tool() -> ToolRegistration {
    define_tool(
        "read",
        format!(
            "Read the contents of a text file. Output is truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
            DEFAULT_MAX_BYTES / 1024
        ),
        read_schema(),
        execute,
    )
}

async fn execute(
    args: JsonValue,
    api: ToolExecutionApi,
    context: Context,
) -> Result<ToolExecutionResult> {
    let ReadToolInput {
        path,
        offset,
        limit,
    } = parse_args(args)?;
    let env = require_env(&api)?;
    let absolute_path = resolve_read_tool_path(env.as_ref(), &path, &context).await?;
    let bytes = get_or_throw(env.read_binary_file(&absolute_path, &context).await)?;
    if let Some(mime_type) = detect_supported_image_mime_type(&bytes) {
        // Image content is not supported yet.
        return Ok(ToolExecutionResult {
            content: Some(Vec::new()),
            is_error: Some(true),
            diagnostics: Some(vec![ToolDiagnostic {
                severity: DiagnosticSeverity::Error,
                code: Some("unsupported_image".into()),
                message: format!(
                    "{path} is an image ({mime_type}); reading images is not supported"
                ),
            }]),
            ..ToolExecutionResult::default()
        });
    }

    let text_content = decode(&bytes);
    let all_lines: Vec<&str> = text_content.split('\n').collect();
    let total_file_lines = all_lines.len();
    let line_count = all_lines.len() as f64;
    // `offset ? Math.max(0, offset - 1) : 0`
    let start_line = match offset {
        Some(offset) if offset != 0.0 && !offset.is_nan() => (offset - 1.0).max(0.0),
        _ => 0.0,
    };
    let start_line_display = start_line + 1.0;
    if start_line >= line_count {
        return Err(Error::message(format!(
            "Offset {} is beyond end of file ({} lines total)",
            offset.map(js_number).unwrap_or_else(|| "undefined".into()),
            all_lines.len()
        )));
    }

    let start = js_slice_index(start_line, all_lines.len());
    let selected_content: String;
    let mut user_limited_lines: Option<f64> = None;
    if let Some(limit) = limit {
        let end_line = (start_line + limit).min(line_count);
        let end = js_slice_index(end_line, all_lines.len());
        selected_content = all_lines[start..end.max(start)].join("\n");
        user_limited_lines = Some(end_line - start_line);
    } else {
        selected_content = all_lines[start..].join("\n");
    }

    let truncation = truncate_head(&selected_content, TruncationOptions::default());
    let mut diagnostics: Vec<ToolDiagnostic> = Vec::new();
    let mut output_text = truncation.content.clone();
    let mut details: Option<JsonValue> = None;
    if truncation.first_line_exceeds_limit {
        // Show the start of the line, cut at the byte limit on a character boundary.
        let line_bytes = all_lines[start].as_bytes();
        let end = character_end(line_bytes, DEFAULT_MAX_BYTES).min(line_bytes.len());
        output_text = String::from_utf8_lossy(&line_bytes[..end]).into_owned();
        diagnostics.push(ToolDiagnostic {
            severity: DiagnosticSeverity::Warn,
            code: Some("truncated".into()),
            message: format!(
                "Line {} is {}, exceeds the {} limit; showing its first {}. Use bash: sed -n '{}p' {path} | tail -c +{}",
                js_number(start_line_display),
                format_size(line_bytes.len() as u64),
                format_size(DEFAULT_MAX_BYTES as u64),
                format_size(end as u64),
                js_number(start_line_display),
                end + 1
            ),
        });
        let mut truncation = truncation_json(&truncation);
        truncation["outputBytes"] = json!(end);
        truncation["outputLines"] = json!(1);
        details = Some(json!({ "truncation": truncation }));
    } else if truncation.truncated {
        let end_line_display = start_line_display + truncation.output_lines as f64 - 1.0;
        let next_offset = end_line_display + 1.0;
        let limit_text = if truncation.truncated_by == Some(TruncatedBy::Lines) {
            String::new()
        } else {
            format!(" ({} limit)", format_size(DEFAULT_MAX_BYTES as u64))
        };
        diagnostics.push(ToolDiagnostic {
            severity: DiagnosticSeverity::Info,
            code: Some("truncated".into()),
            message: format!(
                "Showing lines {}-{} of {total_file_lines}{limit_text}. Use offset={} to continue.",
                js_number(start_line_display),
                js_number(end_line_display),
                js_number(next_offset)
            ),
        });
        details = Some(json!({ "truncation": truncation_json(&truncation) }));
    } else if let Some(user_limited_lines) = user_limited_lines
        && start_line + user_limited_lines < line_count
    {
        let remaining = line_count - (start_line + user_limited_lines);
        let next_offset = start_line + user_limited_lines + 1.0;
        diagnostics.push(ToolDiagnostic {
            severity: DiagnosticSeverity::Info,
            code: None,
            message: format!(
                "{} more lines in file. Use offset={} to continue.",
                js_number(remaining),
                js_number(next_offset)
            ),
        });
    }

    Ok(ToolExecutionResult {
        content: Some(if output_text.is_empty() {
            Vec::new()
        } else {
            vec![UserContent::text(output_text)]
        }),
        details,
        diagnostics: Some(diagnostics),
        ..ToolExecutionResult::default()
    })
}
