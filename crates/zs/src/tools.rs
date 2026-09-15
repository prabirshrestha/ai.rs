use std::time::Duration;

use ai::{AgentError, AgentToolBuilder, AgentToolResult, DynAgentTool, Result};
use serde_json::{Value, json};
use tokio::process::Command;

const BASH_TOOL_TIMEOUT: Duration = Duration::from_secs(60);
const BASH_TOOL_OUTPUT_LIMIT: usize = 16 * 1024;

pub fn build_bash_tool() -> Result<DynAgentTool> {
    AgentToolBuilder::new("bash")
        .description(
            "Run a bash command in the current working directory and return stdout, stderr, and exit status.",
        )
        .parameters(json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run in the agent process's current working directory."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        }))
        .execute(|args| async move {
            let command = args
                .get("command")
                .and_then(Value::as_str)
                .ok_or_else(|| AgentError::Other("missing string argument: command".to_string()))?;

            let output = tokio::time::timeout(
                BASH_TOOL_TIMEOUT,
                Command::new("bash")
                    .kill_on_drop(true)
                    .arg("-lc")
                    .arg(command)
                    .output(),
            )
            .await
            .map_err(|_| {
                AgentError::Other(format!("bash command timed out after {BASH_TOOL_TIMEOUT:?}"))
            })?
            .map_err(|error| AgentError::Other(format!("failed to run bash: {error}")))?;

            let text = format_bash_output(
                output.status.to_string(),
                &output.stdout,
                &output.stderr,
                BASH_TOOL_OUTPUT_LIMIT,
            );

            Ok(AgentToolResult::text(text))
        })
        .build()
}

pub fn format_bash_output(
    status: impl std::fmt::Display,
    stdout: &[u8],
    stderr: &[u8],
    limit: usize,
) -> String {
    let mut text = format!("exit status: {status}\n");
    text.push_str("\nstdout:\n");
    append_limited_utf8(&mut text, stdout, limit);
    text.push_str("\n\nstderr:\n");
    append_limited_utf8(&mut text, stderr, limit);
    text
}

fn append_limited_utf8(text: &mut String, bytes: &[u8], limit: usize) {
    let shown = bytes.len().min(limit);
    text.push_str(&String::from_utf8_lossy(&bytes[..shown]));
    if bytes.len() > shown {
        text.push_str(&format!("\n[truncated {} bytes]", bytes.len() - shown));
    }
}

#[cfg(test)]
mod tests {
    use super::format_bash_output;

    #[test]
    fn bash_output_is_truncated_per_stream() {
        let text = format_bash_output("exit 0", b"abcdef", b"12345", 3);

        assert!(text.contains("stdout:\nabc\n[truncated 3 bytes]"));
        assert!(text.contains("stderr:\n123\n[truncated 2 bytes]"));
        assert!(!text.contains("def"));
        assert!(!text.contains("45"));
    }

    #[test]
    fn bash_output_keeps_short_streams_intact() {
        let text = format_bash_output("exit 0", b"ok", b"", 16);

        assert!(text.contains("exit status: exit 0"));
        assert!(text.contains("stdout:\nok"));
        assert!(text.ends_with("stderr:\n"));
    }
}
