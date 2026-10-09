//! Port of durable `src/tools/edit.ts`.
//!
//! Divergence from Pi: `prepareArguments` repairs a JSON value (TS works on a shallow copy of the object); the
//! caller's value is untouched either way.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::chord::{Context, JsonValue};
use crate::durable::env::{FileError, FileKind};
use crate::durable::errors::{Error, Result};
use crate::durable::harness::define::define_tool;
use crate::durable::harness::tool::ToolExecutionApi;
use crate::durable::harness::types::{ToolExecutionResult, ToolRegistration};
use crate::types::UserContent;

use super::edit_diff::{
    Edit, apply_edits_to_normalized_content, detect_line_ending, generate_diff_string,
    generate_unified_patch, normalize_to_lf, restore_line_endings, strip_bom,
};
use super::env::require_env;
use super::file_mutation_queue::with_file_mutation_queue;
use super::path_utils::resolve_tool_path;
use super::throw_if_aborted;

fn edit_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to the file to edit (relative or absolute)" },
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.",
                        },
                        "newText": { "type": "string", "description": "Replacement text for this targeted edit." },
                    },
                    "required": ["oldText", "newText"],
                },
                "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
            },
        },
        "required": ["path", "edits"],
    })
}

/// `EditToolInput`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditToolInput {
    pub path: String,
    pub edits: Vec<Edit>,
}

/// `EditToolDetails`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditToolDetails {
    pub diff: String,
    pub patch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_changed_line: Option<usize>,
}

fn is_single_edit_input(value: &JsonValue) -> bool {
    value.as_object().is_some_and(|edit| {
        edit.get("oldText").is_some_and(JsonValue::is_string)
            && edit.get("newText").is_some_and(JsonValue::is_string)
    })
}

/// Repair shapes models commonly send: `edits` as a JSON string or as a single edit object, and a top-level
/// `oldText`/`newText` pair. Works on a copy; the call's arguments stay unchanged.
pub fn prepare_edit_arguments(input: JsonValue) -> JsonValue {
    let JsonValue::Object(mut args) = input else {
        return input;
    };
    match args.get("edits") {
        Some(JsonValue::String(text)) => {
            if let Ok(parsed) = serde_json::from_str::<JsonValue>(text) {
                if parsed.is_array() {
                    args.insert("edits".into(), parsed);
                } else if is_single_edit_input(&parsed) {
                    args.insert("edits".into(), JsonValue::Array(vec![parsed]));
                }
            }
        }
        Some(edits) if is_single_edit_input(edits) => {
            let edit = edits.clone();
            args.insert("edits".into(), JsonValue::Array(vec![edit]));
        }
        _ => {}
    }

    let (Some(JsonValue::String(old_text)), Some(JsonValue::String(new_text))) =
        (args.get("oldText"), args.get("newText"))
    else {
        return JsonValue::Object(args);
    };
    let pair = json!({ "oldText": old_text, "newText": new_text });
    let mut edits = match args.get("edits") {
        Some(JsonValue::Array(edits)) => edits.clone(),
        _ => Vec::new(),
    };
    edits.push(pair);
    args.shift_remove("oldText");
    args.shift_remove("newText");
    args.insert("edits".into(), JsonValue::Array(edits));
    JsonValue::Object(args)
}

fn validate_edit_input(input: JsonValue) -> Result<EditToolInput> {
    let has_edits = input
        .get("edits")
        .and_then(JsonValue::as_array)
        .is_some_and(|edits| !edits.is_empty());
    if !has_edits {
        return Err(Error::message(
            "Edit tool input is invalid. edits must contain at least one replacement.",
        ));
    }
    super::parse_args(input)
}

fn edit_access_error(path: &str, error: FileError) -> Error {
    Error::with_cause(
        format!(
            "Could not edit file: {path}. Error code: {}.",
            error.code.as_str()
        ),
        Error::thrown(error),
    )
}

pub fn create_edit_tool() -> ToolRegistration {
    ToolRegistration {
        prepare_arguments: Some(Arc::new(|args| Ok(prepare_edit_arguments(args)))),
        ..define_tool(
            "edit",
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
            edit_schema(),
            execute,
        )
    }
}

async fn execute(
    args: JsonValue,
    api: ToolExecutionApi,
    context: Context,
) -> Result<ToolExecutionResult> {
    let EditToolInput { path, edits } = validate_edit_input(args)?;
    let env = require_env(&api)?;
    let absolute_path = resolve_tool_path(env.as_ref(), &path, &context).await?;
    with_file_mutation_queue(
        env.as_ref(),
        &absolute_path,
        || async {
            throw_if_aborted(&context)?;
            let info = env
                .file_info(&absolute_path, &context)
                .await
                .map_err(|error| edit_access_error(&path, error))?;
            if info.kind != FileKind::File && info.kind != FileKind::Symlink {
                return Err(Error::message(format!(
                    "Could not edit file: {path}. Path is not a file."
                )));
            }

            let read = env
                .read_text_file(&absolute_path, &context)
                .await
                .map_err(|error| edit_access_error(&path, error))?;
            throw_if_aborted(&context)?;

            let (bom, content) = strip_bom(&read);
            let original_ending = detect_line_ending(content);
            let normalized_content = normalize_to_lf(content);
            let applied = apply_edits_to_normalized_content(&normalized_content, &edits, &path)?;
            throw_if_aborted(&context)?;

            let final_content = format!(
                "{bom}{}",
                restore_line_endings(&applied.new_content, original_ending)
            );
            env.write_file(&absolute_path, final_content.as_bytes(), &context)
                .await
                .map_err(|error| edit_access_error(&path, error))?;
            throw_if_aborted(&context)?;

            let diff_result =
                generate_diff_string(&applied.base_content, &applied.new_content, None);
            let details = EditToolDetails {
                diff: diff_result.diff,
                patch: generate_unified_patch(
                    &path,
                    &applied.base_content,
                    &applied.new_content,
                    None,
                ),
                first_changed_line: diff_result.first_changed_line,
            };
            Ok(ToolExecutionResult {
                content: Some(vec![UserContent::text(format!(
                    "Successfully replaced {} block(s) in {path}.",
                    edits.len()
                ))]),
                details: Some(serde_json::to_value(details).map_err(Error::thrown)?),
                ..ToolExecutionResult::default()
            })
        },
        &context,
    )
    .await
}
