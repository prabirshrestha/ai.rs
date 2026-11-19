//! [rust] Port of `bridge/opencode-read-plugin.ts`: the `read` tool OpenCode
//! uses in restrictive permission modes, served by the host tool relay and
//! read through the sandbox. Like the plugin, it never discovers or opens
//! instruction files.

use base64::Engine as _;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::sandbox::{SandboxFileType, SandboxPathOptions, SandboxSession};
use crate::{AgentToolResult, Tool};

pub(super) const READ_TOOL_NAME: &str = "read";

const MAX_BYTES: usize = 50 * 1024;
const MAX_LINES: usize = 2000;
const MAX_LINE_LENGTH: usize = 2000;

/// The tool definition OpenCode's MCP client lists.
pub(super) fn read_tool_definition() -> Tool {
    Tool {
        name: READ_TOOL_NAME.to_string(),
        description: "Read a text file, image, PDF, or list a directory. filePath is absolute or relative to the working directory. offset is 1-based; limit defaults to 2000 lines. Does not load nearby instruction files.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "filePath": { "type": "string" },
                "offset": { "type": "integer", "minimum": 0 },
                "limit": { "type": "integer", "minimum": 0 },
            },
            "required": ["filePath"],
            "additionalProperties": false,
        }),
        constrained_sampling: None,
    }
}

/// The parsed arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ReadArgs {
    /// Absolute, resolved against the working directory.
    pub file_path: String,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

fn non_negative(args: &Value, name: &str) -> Result<Option<usize>, String> {
    match args.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(|value| Some(value as usize))
            .ok_or_else(|| format!("Invalid {name}: expected a non-negative integer.")),
    }
}

/// `argsSchema.parse` and `path.resolve(ctx.directory, filePath)`.
pub(super) fn parse_read_args(args: &Value, work_dir: &str) -> Result<ReadArgs, String> {
    let file_path = args
        .get("filePath")
        .and_then(Value::as_str)
        .ok_or_else(|| "Invalid filePath: expected a string.".to_string())?;
    let absolute = if file_path.starts_with('/') {
        file_path.to_string()
    } else {
        format!("{}/{file_path}", work_dir.trim_end_matches('/'))
    };
    Ok(ReadArgs {
        file_path: crate::harness::paths::normalize_posix(&absolute),
        offset: non_negative(args, "offset")?,
        limit: non_negative(args, "limit")?,
    })
}

/// `attachmentMime`.
fn attachment_mime(header: &[u8]) -> Option<&'static str> {
    if header.starts_with(b"%PDF-") {
        Some("application/pdf")
    } else if header.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]) {
        Some("image/png")
    } else if header.starts_with(&[255, 216, 255]) {
        Some("image/jpeg")
    } else if header.starts_with(b"GIF87a") || header.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if header.len() >= 12 && &header[..4] == b"RIFF" && &header[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// The result: the host-facing tool result and the MCP `CallToolResult`.
pub(super) struct ReadOutput {
    pub result: AgentToolResult,
    pub mcp: Value,
    pub is_error: bool,
}

impl ReadOutput {
    fn text(text: String, is_error: bool) -> Self {
        Self {
            mcp: json!({ "content": [{ "type": "text", "text": text }], "isError": is_error }),
            result: AgentToolResult::text(text),
            is_error,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::text(message.into(), true)
    }
}

fn truncate_line(line: &str) -> String {
    if line.chars().count() > MAX_LINE_LENGTH {
        let head: String = line.chars().take(MAX_LINE_LENGTH).collect();
        format!("{head}... (line truncated)")
    } else {
        line.to_string()
    }
}

/// Render text the way the plugin's line reader does.
pub(super) fn render_text(file_path: &str, text: &str, offset: usize, limit: usize) -> String {
    let mut lines = Vec::new();
    let mut bytes = 0usize;
    let mut truncated = false;
    let body = text.strip_suffix('\n').unwrap_or(text);
    if !text.is_empty() {
        for (index, line) in body.split('\n').enumerate() {
            let line_number = index + 1;
            if line_number < offset {
                continue;
            }
            let line = line.strip_suffix('\r').unwrap_or(line);
            let rendered = format!("{line_number}: {}", truncate_line(line));
            if lines.len() >= limit || bytes + rendered.len() + 1 > MAX_BYTES {
                truncated = true;
                break;
            }
            bytes += rendered.len() + 1;
            lines.push(rendered);
        }
    }
    format!(
        "<path>{file_path}</path>\n<type>file</type>\n<content>\n{}\n</content>{}",
        lines.join("\n"),
        if truncated {
            "\n(Output truncated; use offset to read more.)"
        } else {
            ""
        }
    )
}

/// Run the tool. The caller already approved `args`.
pub(super) async fn read(
    sandbox: &dyn SandboxSession,
    args: &ReadArgs,
    abort: Option<CancellationToken>,
) -> ReadOutput {
    let options = || SandboxPathOptions {
        path: args.file_path.clone(),
        abort_signal: abort.clone(),
    };
    let offset = args.offset.unwrap_or(1).max(1);
    let limit = args.limit.unwrap_or(MAX_LINES).min(MAX_LINES);
    let stat = match sandbox.stat(options()).await {
        Ok(Some(stat)) => stat,
        Ok(None) => {
            return ReadOutput::error(format!(
                "ENOENT: no such file or directory, open '{}'",
                args.file_path
            ));
        }
        Err(error) => return ReadOutput::error(error.to_string()),
    };
    match stat.r#type {
        SandboxFileType::Directory => {
            let entries = match sandbox.list_directory(options()).await {
                Ok(entries) => entries.unwrap_or_default(),
                Err(error) => return ReadOutput::error(error.to_string()),
            };
            let mut names: Vec<String> = entries
                .into_iter()
                .map(|entry| {
                    if entry.r#type == crate::harness::sandbox::SandboxEntryType::Directory {
                        format!("{}/", entry.name)
                    } else {
                        entry.name
                    }
                })
                .collect();
            names.sort();
            let selected: Vec<String> =
                names.iter().skip(offset - 1).take(limit).cloned().collect();
            ReadOutput::text(
                format!(
                    "<path>{}</path>\n<type>directory</type>\n<entries>\n{}\n</entries>",
                    args.file_path,
                    selected.join("\n")
                ),
                false,
            )
        }
        SandboxFileType::Other => {
            ReadOutput::error("Only regular files and directories can be read.")
        }
        SandboxFileType::File => {
            let bytes = match sandbox.read_binary_file(options()).await {
                Ok(Some(bytes)) => bytes,
                Ok(None) => {
                    return ReadOutput::error(format!(
                        "ENOENT: no such file or directory, open '{}'",
                        args.file_path
                    ));
                }
                Err(error) => return ReadOutput::error(error.to_string()),
            };
            let header = &bytes[..bytes.len().min(4096)];
            if let Some(mime) = attachment_mime(header) {
                let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
                if mime == "application/pdf" {
                    let text = "PDF read successfully".to_string();
                    return ReadOutput {
                        mcp: json!({
                            "content": [
                                { "type": "text", "text": text },
                                { "type": "resource", "resource": {
                                    "uri": format!("file://{}", args.file_path),
                                    "mimeType": mime,
                                    "blob": data,
                                } },
                            ],
                            "isError": false,
                        }),
                        result: AgentToolResult::text(text),
                        is_error: false,
                    };
                }
                let text = "Image read successfully";
                let mut result = AgentToolResult::text(text);
                result
                    .content
                    .push(crate::ToolResultContent::Image(crate::ImageContent {
                        data: data.clone(),
                        mime_type: mime.to_string(),
                    }));
                return ReadOutput {
                    mcp: json!({
                        "content": [
                            { "type": "text", "text": text },
                            { "type": "image", "data": data, "mimeType": mime },
                        ],
                        "isError": false,
                    }),
                    result,
                    is_error: false,
                };
            }
            if header.contains(&0) {
                return ReadOutput::error(
                    "Binary files are not supported by the restrictive-mode read tool.",
                );
            }
            let text = String::from_utf8_lossy(&bytes);
            ReadOutput::text(render_text(&args.file_path, &text, offset, limit), false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::adapters::local_sandbox::{LocalSandbox, canonical};
    use crate::harness::adapters::test_temp_dir::mkdtemp;

    fn text_of(output: &ReadOutput) -> String {
        output.mcp["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn parses_and_resolves_arguments() {
        assert_eq!(
            parse_read_args(&json!({ "filePath": "a/../b.txt", "offset": 2 }), "/work"),
            Ok(ReadArgs {
                file_path: "/work/b.txt".into(),
                offset: Some(2),
                limit: None,
            })
        );
        assert!(parse_read_args(&json!({}), "/work").is_err());
        assert!(parse_read_args(&json!({ "filePath": "x", "limit": -1 }), "/work").is_err());
    }

    #[test]
    fn renders_numbered_lines_with_offset_limit_and_truncation() {
        assert_eq!(
            render_text("/w/a.txt", "one\r\ntwo\nthree\n", 2, 1),
            "<path>/w/a.txt</path>\n<type>file</type>\n<content>\n2: two\n</content>\n(Output truncated; use offset to read more.)"
        );
        assert_eq!(
            render_text("/w/a.txt", "", 1, 2000),
            "<path>/w/a.txt</path>\n<type>file</type>\n<content>\n\n</content>"
        );
        let long = "x".repeat(MAX_LINE_LENGTH + 5);
        assert!(render_text("/w/a", &long, 1, 10).contains("... (line truncated)"));
        let many = "y\n".repeat(30_000);
        let rendered = render_text("/w/a", &many, 1, 2000);
        assert!(rendered.contains("\n2000: y\n"));
        assert!(rendered.ends_with("(Output truncated; use offset to read more.)"));
    }

    #[tokio::test]
    async fn reads_files_directories_and_images_through_the_sandbox() {
        let temp = mkdtemp("opencode-read-");
        let root = canonical(temp.path());
        std::fs::create_dir_all(format!("{root}/dir/sub")).unwrap();
        std::fs::write(format!("{root}/dir/b.txt"), "b").unwrap();
        std::fs::write(format!("{root}/a.txt"), "alpha\nbeta\n").unwrap();
        std::fs::write(format!("{root}/bin"), [1u8, 0, 2]).unwrap();
        std::fs::write(
            format!("{root}/i.png"),
            [137, 80, 78, 71, 13, 10, 26, 10, 0],
        )
        .unwrap();
        let sandbox = LocalSandbox::new(root.clone());
        let args = |path: &str| parse_read_args(&json!({ "filePath": path }), &root).unwrap();

        let file = read(&sandbox, &args("a.txt"), None).await;
        assert!(!file.is_error);
        assert_eq!(
            text_of(&file),
            format!(
                "<path>{root}/a.txt</path>\n<type>file</type>\n<content>\n1: alpha\n2: beta\n</content>"
            )
        );
        let dir = read(&sandbox, &args("dir"), None).await;
        assert_eq!(
            text_of(&dir),
            format!(
                "<path>{root}/dir</path>\n<type>directory</type>\n<entries>\nb.txt\nsub/\n</entries>"
            )
        );
        let binary = read(&sandbox, &args("bin"), None).await;
        assert!(binary.is_error);
        let image = read(&sandbox, &args("i.png"), None).await;
        assert_eq!(image.mcp["content"][1]["mimeType"], "image/png");
        assert_eq!(image.result.content.len(), 2);
        assert!(read(&sandbox, &args("missing"), None).await.is_error);
    }
}
