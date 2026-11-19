//! Pi's builtin tools (`buildBuiltinToolDefinition` in `pi-session.ts`
//! and Pi's own `read` tool), as ai.rs [`AgentTool`]s over the sandbox.
//!
//! Approvals are not handled here: the session asks `before_tool_call`
//! before any builtin runs that the permission mode does not allow.

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use base64::Engine as _;
use indexmap::IndexMap;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::pi_remote_ops::{PiExecInput, PiGrepInput, PiRemoteOps};
use super::pi_tool_result::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head, truncate_pi_tool_output_head,
    truncate_pi_tool_output_tail,
};
use crate::harness::error::into_agent_error;
use crate::harness::tools::{
    BuiltinTool, BuiltinToolKind, BuiltinToolName, CommonToolOptions, common_tool,
};
use crate::{
    AgentResult, AgentTool, AgentToolResult, AgentToolUpdateCallback, DynAgentTool, ImageContent,
    Tool, ToolResultContent,
};

/// Pi's own names for its builtin tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiNativeTool {
    Read,
    Write,
    Edit,
    Bash,
    Grep,
    Find,
    Ls,
}

/// `PI_NATIVE_BUILTIN_NAMES`, in registration order.
pub const PI_NATIVE_TOOLS: [PiNativeTool; 7] = [
    PiNativeTool::Read,
    PiNativeTool::Write,
    PiNativeTool::Edit,
    PiNativeTool::Bash,
    PiNativeTool::Grep,
    PiNativeTool::Find,
    PiNativeTool::Ls,
];

impl PiNativeTool {
    /// The name the model sees.
    pub fn native_name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Edit => "edit",
            Self::Bash => "bash",
            Self::Grep => "grep",
            Self::Find => "find",
            Self::Ls => "ls",
        }
    }

    /// The harness name (`find` is the common `glob`).
    pub fn harness_name(self) -> &'static str {
        match self {
            Self::Find => "glob",
            other => other.native_name(),
        }
    }

    pub fn kind(self) -> BuiltinToolKind {
        match self {
            Self::Write | Self::Edit => BuiltinToolKind::Edit,
            Self::Bash => BuiltinToolKind::Bash,
            _ => BuiltinToolKind::Readonly,
        }
    }

    pub fn from_native_name(name: &str) -> Option<Self> {
        PI_NATIVE_TOOLS
            .into_iter()
            .find(|tool| tool.native_name() == name)
    }
}

/// `piBuiltinToolRequiresApproval({ permissionMode, kind })`.
pub fn pi_builtin_tool_requires_approval(
    permission_mode: crate::harness::tools::PermissionMode,
    kind: BuiltinToolKind,
) -> bool {
    use crate::harness::tools::PermissionMode as M;
    match permission_mode {
        M::AllowAll => false,
        M::AllowEdits => kind == BuiltinToolKind::Bash,
        M::AllowReads => matches!(kind, BuiltinToolKind::Edit | BuiltinToolKind::Bash),
    }
}

fn common(
    name: BuiltinToolName,
    native_name: &str,
    kind: BuiltinToolKind,
    description: &str,
    parameters: Value,
) -> BuiltinTool {
    common_tool(
        name,
        CommonToolOptions {
            native_name: native_name.into(),
            kind: Some(kind),
            description: Some(description.into()),
            parameters,
        },
    )
    .expect("Pi builtin schemas extend the standard schemas")
}

/// `PI_BUILTIN_TOOLS`: the builtins as the harness reports them.
pub fn pi_builtin_tools() -> &'static IndexMap<String, BuiltinTool> {
    static TOOLS: OnceLock<IndexMap<String, BuiltinTool>> = OnceLock::new();
    TOOLS.get_or_init(|| {
        use BuiltinToolKind as K;
        use BuiltinToolName as N;
        let mut tools = IndexMap::new();
        tools.insert(
            "read".into(),
            common(
                N::Read,
                "read",
                K::Readonly,
                "Read file contents. Output is limited to 2,000 lines or 50KB. Use offset and limit to read large files in pages.",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": { "type": "string" },
                        "offset": { "type": "integer", "exclusiveMinimum": 0 },
                        "limit": { "type": "integer", "exclusiveMinimum": 0 },
                    },
                    "required": ["file_path"],
                }),
            ),
        );
        tools.insert(
            "write".into(),
            common(
                N::Write,
                "write",
                K::Edit,
                "Overwrite or create a file.",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": { "type": "string" },
                        "content": { "type": "string" },
                    },
                    "required": ["file_path", "content"],
                }),
            ),
        );
        tools.insert(
            "edit".into(),
            common(
                N::Edit,
                "edit",
                K::Edit,
                "Edit a file by exact string replacement.",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": { "type": "string" },
                        "old_string": { "type": "string" },
                        "new_string": { "type": "string" },
                    },
                    "required": ["file_path", "old_string", "new_string"],
                }),
            ),
        );
        tools.insert(
            "bash".into(),
            common(
                N::Bash,
                "bash",
                K::Bash,
                "Execute a shell command in the sandbox.",
                json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "timeout": { "type": "number" },
                    },
                    "required": ["command"],
                }),
            ),
        );
        tools.insert(
            "grep".into(),
            common(
                N::Grep,
                "grep",
                K::Readonly,
                "Search file contents with regex.",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string" },
                        "path": { "type": "string" },
                        "glob": { "type": "string" },
                        "ignoreCase": { "type": "boolean" },
                        "literal": { "type": "boolean" },
                        "context": { "type": "number" },
                        "limit": { "type": "number" },
                    },
                    "required": ["pattern"],
                }),
            ),
        );
        tools.insert(
            "glob".into(),
            common(
                N::Glob,
                "find",
                K::Readonly,
                "Find files matching a glob pattern.",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string" },
                        "path": { "type": "string" },
                        "limit": { "type": "number" },
                    },
                    "required": ["pattern"],
                }),
            ),
        );
        tools.insert(
            "ls".into(),
            BuiltinTool {
                tool: Tool {
                    name: "ls".into(),
                    description: "List directory entries.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" },
                            "limit": { "type": "number" },
                        },
                    }),
                    constrained_sampling: None,
                },
                native_name: Some("ls".into()),
                common_name: None,
                kind: Some(K::Readonly),
            },
        );
        tools
    })
}

/// The description of Pi's `read` tool (`createReadToolDefinition`).
pub fn pi_read_tool_description() -> String {
    format!(
        "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
        DEFAULT_MAX_BYTES / 1024
    )
}

fn native_definition(tool: PiNativeTool) -> Tool {
    let (description, parameters) = match tool {
        PiNativeTool::Read => (
            pi_read_tool_description(),
            json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string" },
                    "offset": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Line number to start reading from (1-indexed).",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum number of lines to read.",
                    },
                },
                "required": ["file_path"],
            }),
        ),
        PiNativeTool::Write => (
            "Write content to a file.".into(),
            json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string" },
                    "content": { "type": "string" },
                },
                "required": ["file_path", "content"],
            }),
        ),
        PiNativeTool::Edit => (
            "Edit a file by exact-string replacement.".into(),
            json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" },
                },
                "required": ["file_path", "old_string", "new_string"],
            }),
        ),
        PiNativeTool::Bash => (
            "Execute a shell command.".into(),
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "timeout": { "type": "number", "description": "Timeout in seconds." },
                },
                "required": ["command"],
            }),
        ),
        PiNativeTool::Grep => (
            "Search file contents with regex.".into(),
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" },
                    "glob": { "type": "string" },
                    "ignoreCase": { "type": "boolean" },
                    "literal": { "type": "boolean" },
                    "context": { "type": "number" },
                    "limit": { "type": "number" },
                },
                "required": ["pattern"],
            }),
        ),
        PiNativeTool::Find => (
            "Find files matching a glob pattern.".into(),
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" },
                    "limit": { "type": "number" },
                },
                "required": ["pattern"],
            }),
        ),
        PiNativeTool::Ls => (
            "List directory entries.".into(),
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "limit": { "type": "number" },
                },
            }),
        ),
    };
    Tool {
        name: tool.native_name().into(),
        description,
        parameters,
        constrained_sampling: None,
    }
}

/// One Pi builtin, run against the sandbox.
pub struct PiBuiltinAgentTool {
    tool: PiNativeTool,
    definition: Tool,
    ops: Arc<PiRemoteOps>,
}

impl PiBuiltinAgentTool {
    pub fn new(tool: PiNativeTool, ops: Arc<PiRemoteOps>) -> Self {
        Self {
            tool,
            definition: native_definition(tool),
            ops,
        }
    }

    pub fn native_tool(&self) -> PiNativeTool {
        self.tool
    }
}

/// Every Pi builtin over `ops`, in registration order.
pub fn create_pi_builtin_agent_tools(ops: Arc<PiRemoteOps>) -> Vec<Arc<PiBuiltinAgentTool>> {
    PI_NATIVE_TOOLS
        .into_iter()
        .map(|tool| Arc::new(PiBuiltinAgentTool::new(tool, ops.clone())))
        .collect()
}

/// As [`DynAgentTool`]s.
pub fn pi_builtin_dyn_tools(ops: Arc<PiRemoteOps>) -> Vec<DynAgentTool> {
    create_pi_builtin_agent_tools(ops)
        .into_iter()
        .map(|tool| tool as DynAgentTool)
        .collect()
}

fn text_result(text: impl Into<String>) -> AgentToolResult {
    AgentToolResult::text(text)
}

fn string_arg<'a>(args: &'a Value, key: &str) -> AgentResult<&'a str> {
    args.get(key).and_then(Value::as_str).ok_or_else(|| {
        crate::AgentError::Other(format!("Missing required string argument '{key}'."))
    })
}

fn optional_string(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(String::from)
}

fn optional_number(args: &Value, key: &str) -> Option<f64> {
    args.get(key).and_then(Value::as_f64)
}

/// `detectSupportedImageMimeType`: sniff the formats Pi's `read` sends as
/// images.
pub fn detect_supported_image_mime_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        Some("image/png")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") && bytes.len() >= 14 {
        Some("image/bmp")
    } else {
        None
    }
}

/// Pi's `read` over the sandbox (`executePiSandboxRead`).
///
/// [rust] Images are sent as read, without Pi's resizing.
pub async fn execute_pi_read(
    ops: &PiRemoteOps,
    path: &str,
    offset: Option<i64>,
    limit: Option<i64>,
) -> AgentResult<AgentToolResult> {
    // Pi expands `~` against the host home, so map the path first.
    let sandbox_path = ops
        .paths()
        .to_readable_sandbox_path(path)
        .map_err(crate::AgentError::from)?;
    let buffer = ops
        .read_buffer(&sandbox_path)
        .await
        .map_err(into_agent_error)?;
    if let Some(mime_type) = detect_supported_image_mime_type(&buffer) {
        return Ok(AgentToolResult {
            content: vec![
                ToolResultContent::text(format!("Read image file [{mime_type}]")),
                ToolResultContent::Image(ImageContent {
                    data: base64::engine::general_purpose::STANDARD.encode(&buffer),
                    mime_type: mime_type.into(),
                }),
            ],
            ..AgentToolResult::text("")
        });
    }
    let text = String::from_utf8_lossy(&buffer);
    let all_lines: Vec<&str> = text.split('\n').collect();
    let total_file_lines = all_lines.len();
    let start_line = offset.map_or(0, |offset| (offset - 1).max(0) as usize);
    let start_line_display = start_line + 1;
    if start_line >= all_lines.len() {
        return Err(crate::AgentError::Other(format!(
            "Offset {} is beyond end of file ({} lines total)",
            offset.unwrap_or_default(),
            all_lines.len()
        )));
    }
    let (selected, user_limited_lines) = match limit {
        Some(limit) => {
            let end_line = (start_line + limit.max(0) as usize).min(all_lines.len());
            (
                all_lines[start_line..end_line].join("\n"),
                Some(end_line - start_line),
            )
        }
        None => (all_lines[start_line..].join("\n"), None),
    };
    let truncation = truncate_head(&selected);
    let mut details = None;
    let output = if truncation.first_line_exceeds_limit {
        let first_line_size = format_size(all_lines[start_line].len());
        details = Some(json!({ "truncation": truncation_json(&truncation) }));
        format!(
            "[Line {start_line_display} is {first_line_size}, exceeds {} limit. Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]",
            format_size(DEFAULT_MAX_BYTES)
        )
    } else if truncation.truncated {
        let end_line_display = start_line_display + truncation.output_lines - 1;
        let next_offset = end_line_display + 1;
        details = Some(json!({ "truncation": truncation_json(&truncation) }));
        if truncation.truncated_by == Some(super::pi_tool_result::TruncatedBy::Lines) {
            format!(
                "{}\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. Use offset={next_offset} to continue.]",
                truncation.content
            )
        } else {
            format!(
                "{}\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} ({} limit). Use offset={next_offset} to continue.]",
                truncation.content,
                format_size(DEFAULT_MAX_BYTES)
            )
        }
    } else if let Some(user_limited_lines) = user_limited_lines
        && start_line + user_limited_lines < all_lines.len()
    {
        let remaining = all_lines.len() - (start_line + user_limited_lines);
        let next_offset = start_line + user_limited_lines + 1;
        format!(
            "{}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]",
            truncation.content
        )
    } else {
        truncation.content.clone()
    };
    Ok(AgentToolResult {
        details,
        ..text_result(output)
    })
}

fn truncation_json(truncation: &super::pi_tool_result::TruncationResult) -> Value {
    json!({
        "truncated": truncation.truncated,
        "truncatedBy": truncation.truncated_by.map(|by| match by {
            super::pi_tool_result::TruncatedBy::Lines => "lines",
            super::pi_tool_result::TruncatedBy::Bytes => "bytes",
        }),
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

#[async_trait]
impl AgentTool for PiBuiltinAgentTool {
    fn definition(&self) -> Tool {
        self.definition.clone()
    }

    fn label(&self) -> &str {
        self.tool.native_name()
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        args: Value,
        cancellation_token: Option<CancellationToken>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> AgentResult<AgentToolResult> {
        let ops = &self.ops;
        match self.tool {
            PiNativeTool::Read => {
                execute_pi_read(
                    ops,
                    string_arg(&args, "file_path")?,
                    args.get("offset").and_then(Value::as_i64),
                    args.get("limit").and_then(Value::as_i64),
                )
                .await
            }
            PiNativeTool::Write => {
                let file_path = string_arg(&args, "file_path")?;
                ops.write_file(file_path, string_arg(&args, "content")?)
                    .await
                    .map_err(into_agent_error)?;
                Ok(text_result(format!("Wrote {file_path}")))
            }
            PiNativeTool::Edit => {
                let file_path = string_arg(&args, "file_path")?;
                ops.edit_file(
                    file_path,
                    string_arg(&args, "old_string")?,
                    string_arg(&args, "new_string")?,
                )
                .await
                .map_err(into_agent_error)?;
                Ok(text_result(format!("Edited {file_path}")))
            }
            PiNativeTool::Bash => {
                let mut chunks: Vec<u8> = Vec::new();
                let mut on_data = |data: &[u8]| chunks.extend_from_slice(data);
                let exit_code = ops
                    .exec(
                        string_arg(&args, "command")?,
                        ".",
                        PiExecInput {
                            on_data: Some(&mut on_data),
                            signal: cancellation_token,
                            timeout: optional_number(&args, "timeout"),
                        },
                    )
                    .await
                    .map_err(into_agent_error)?;
                let out = String::from_utf8_lossy(&chunks);
                let text = match exit_code {
                    Some(code) => format!("{out}\n\n(exit {code})"),
                    None => out.into_owned(),
                };
                Ok(text_result(truncate_pi_tool_output_tail(
                    text.trim(),
                    "Re-run the command with narrower output, or redirect it to a file and use read with offset and limit.",
                )))
            }
            PiNativeTool::Grep => {
                let output = ops
                    .grep_files(
                        string_arg(&args, "pattern")?,
                        PiGrepInput {
                            path: optional_string(&args, "path"),
                            glob: optional_string(&args, "glob"),
                            ignore_case: args.get("ignoreCase").and_then(Value::as_bool)
                                == Some(true),
                            literal: args.get("literal").and_then(Value::as_bool) == Some(true),
                            context: optional_number(&args, "context").map(|value| value as i64),
                            limit: optional_number(&args, "limit").map(|value| value as i64),
                        },
                    )
                    .await
                    .map_err(into_agent_error)?;
                Ok(text_result(truncate_pi_tool_output_head(
                    &output,
                    "Narrow the pattern or path, or read the matching file with offset and limit.",
                )))
            }
            PiNativeTool::Find => {
                let matches = ops
                    .find_files(
                        string_arg(&args, "pattern")?,
                        Some(optional_string(&args, "path").as_deref().unwrap_or(".")),
                        Some(optional_number(&args, "limit").map_or(1_000, |limit| limit as usize)),
                    )
                    .await
                    .map_err(into_agent_error)?;
                Ok(text_result(truncate_pi_tool_output_head(
                    &matches.join("\n"),
                    "Narrow the pattern or search path to inspect the omitted matches.",
                )))
            }
            PiNativeTool::Ls => {
                let entries = ops
                    .list_directory(
                        Some(optional_string(&args, "path").as_deref().unwrap_or(".")),
                        Some(optional_number(&args, "limit").map_or(500, |limit| limit as usize)),
                    )
                    .await
                    .map_err(into_agent_error)?;
                Ok(text_result(truncate_pi_tool_output_head(
                    &entries.join("\n"),
                    "List a narrower directory path to inspect the omitted entries.",
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::adapters::pi::pi_paths::{
        CreatePiPathMapperOptions, create_pi_path_mapper,
    };
    use crate::harness::adapters::pi::pi_remote_ops::{PiRemoteOpsOptions, create_pi_remote_ops};
    use crate::harness::adapters::pi::pi_test_support::{LocalSandbox, canonical};
    use crate::harness::adapters::test_temp_dir::mkdtemp;
    use crate::harness::tools::PermissionMode;

    fn tool_text(result: &AgentToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|part| match part {
                ToolResultContent::Text(text) => Some(text.text.as_str()),
                ToolResultContent::Image(_) => None,
            })
            .collect()
    }

    fn workspace() -> (
        crate::harness::adapters::test_temp_dir::TempDir,
        String,
        Vec<Arc<PiBuiltinAgentTool>>,
    ) {
        let temp = mkdtemp("pi-tools-");
        let root = canonical(temp.path());
        let ops = create_pi_remote_ops(PiRemoteOpsOptions {
            sandbox: Arc::new(LocalSandbox::new(root.clone())),
            paths: create_pi_path_mapper(CreatePiPathMapperOptions {
                sandbox_work_dir: root.clone(),
                ..Default::default()
            })
            .unwrap(),
            env: None,
            on_file_change: None,
        });
        let tools = create_pi_builtin_agent_tools(Arc::new(ops));
        (temp, root, tools)
    }

    fn tool(tools: &[Arc<PiBuiltinAgentTool>], tool: PiNativeTool) -> Arc<PiBuiltinAgentTool> {
        tools
            .iter()
            .find(|candidate| candidate.native_tool() == tool)
            .unwrap()
            .clone()
    }

    #[test]
    fn reports_pi_builtins_with_harness_names() {
        let names: Vec<_> = pi_builtin_tools().keys().cloned().collect();
        assert_eq!(
            names,
            ["read", "write", "edit", "bash", "grep", "glob", "ls"]
        );
        assert_eq!(
            pi_builtin_tools()["glob"].native_name.as_deref(),
            Some("find")
        );
        for native in PI_NATIVE_TOOLS {
            let builtin = &pi_builtin_tools()[native.harness_name()];
            assert_eq!(builtin.kind, Some(native.kind()));
            assert_eq!(
                PiNativeTool::from_native_name(native.native_name()),
                Some(native)
            );
        }
    }

    #[test]
    fn approval_follows_the_permission_mode() {
        use BuiltinToolKind as K;
        assert!(!pi_builtin_tool_requires_approval(
            PermissionMode::AllowAll,
            K::Bash
        ));
        assert!(pi_builtin_tool_requires_approval(
            PermissionMode::AllowEdits,
            K::Bash
        ));
        assert!(!pi_builtin_tool_requires_approval(
            PermissionMode::AllowEdits,
            K::Edit
        ));
        assert!(pi_builtin_tool_requires_approval(
            PermissionMode::AllowReads,
            K::Edit
        ));
        assert!(!pi_builtin_tool_requires_approval(
            PermissionMode::AllowReads,
            K::Readonly
        ));
    }

    #[tokio::test]
    async fn writes_reads_edits_and_lists_through_the_sandbox() {
        let (_temp, root, tools) = workspace();
        let run = |native, args: Value| {
            let tool = tool(&tools, native);
            async move { tool.execute("call", args, None, None).await }
        };
        let wrote = run(
            PiNativeTool::Write,
            json!({ "file_path": "notes.txt", "content": "one\ntwo\nthree" }),
        )
        .await
        .unwrap();
        assert_eq!(tool_text(&wrote), "Wrote notes.txt");
        assert_eq!(
            std::fs::read_to_string(format!("{root}/notes.txt")).unwrap(),
            "one\ntwo\nthree"
        );

        let read = run(
            PiNativeTool::Read,
            json!({ "file_path": "notes.txt", "offset": 2, "limit": 1 }),
        )
        .await
        .unwrap();
        assert_eq!(
            tool_text(&read),
            "two\n\n[1 more lines in file. Use offset=3 to continue.]"
        );

        let error = run(
            PiNativeTool::Read,
            json!({ "file_path": "notes.txt", "offset": 9 }),
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Offset 9 is beyond end of file (3 lines total)"
        );

        run(
            PiNativeTool::Edit,
            json!({ "file_path": "notes.txt", "old_string": "two", "new_string": "TWO" }),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(format!("{root}/notes.txt")).unwrap(),
            "one\nTWO\nthree"
        );

        let listed = run(PiNativeTool::Ls, json!({})).await.unwrap();
        assert_eq!(tool_text(&listed), "notes.txt");
        let found = run(PiNativeTool::Find, json!({ "pattern": "*.txt" }))
            .await
            .unwrap();
        assert_eq!(tool_text(&found), "notes.txt");
        let grep = run(
            PiNativeTool::Grep,
            json!({ "pattern": "TWO", "literal": true }),
        )
        .await
        .unwrap();
        assert!(tool_text(&grep).contains("notes.txt:2:TWO"));
    }

    #[tokio::test]
    async fn bash_appends_the_exit_code() {
        let (_temp, _root, tools) = workspace();
        let result = tool(&tools, PiNativeTool::Bash)
            .execute("call", json!({ "command": "echo hi; exit 3" }), None, None)
            .await
            .unwrap();
        assert_eq!(tool_text(&result), "hi\n\n\n(exit 3)");
    }

    #[tokio::test]
    async fn read_sends_images_as_attachments() {
        let (_temp, root, tools) = workspace();
        std::fs::write(
            format!("{root}/pixel.png"),
            [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0],
        )
        .unwrap();
        let result = tool(&tools, PiNativeTool::Read)
            .execute("call", json!({ "file_path": "pixel.png" }), None, None)
            .await
            .unwrap();
        assert_eq!(tool_text(&result), "Read image file [image/png]");
        assert!(matches!(
            &result.content[1],
            ToolResultContent::Image(image) if image.mime_type == "image/png"
        ));
    }
}
