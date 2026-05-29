use std::collections::HashSet;

use ai::{AssistantContent, Message, ToolResultContent, UserContent, UserMessageContent};
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileOperations {
    pub read: HashSet<String>,
    pub written: HashSet<String>,
    pub edited: HashSet<String>,
}

pub fn create_file_ops() -> FileOperations {
    FileOperations::default()
}

pub fn extract_file_ops_from_message(message: &Message, file_ops: &mut FileOperations) {
    let Message::Assistant(message) = message else {
        return;
    };

    for block in &message.content {
        let AssistantContent::ToolCall(tool_call) = block else {
            continue;
        };
        let Some(path) = tool_call.arguments.get("path").and_then(Value::as_str) else {
            continue;
        };
        match tool_call.name.as_str() {
            "read" => {
                file_ops.read.insert(path.to_string());
            }
            "write" => {
                file_ops.written.insert(path.to_string());
            }
            "edit" => {
                file_ops.edited.insert(path.to_string());
            }
            _ => {}
        }
    }
}

pub fn compute_file_lists(file_ops: &FileOperations) -> (Vec<String>, Vec<String>) {
    let modified = file_ops
        .edited
        .union(&file_ops.written)
        .cloned()
        .collect::<HashSet<_>>();
    let mut read_files = file_ops
        .read
        .iter()
        .filter(|path| !modified.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    let mut modified_files = modified.into_iter().collect::<Vec<_>>();
    read_files.sort();
    modified_files.sort();
    (read_files, modified_files)
}

pub fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
    let mut sections = Vec::new();
    if !read_files.is_empty() {
        sections.push(format!(
            "<read-files>\n{}\n</read-files>",
            read_files.join("\n")
        ));
    }
    if !modified_files.is_empty() {
        sections.push(format!(
            "<modified-files>\n{}\n</modified-files>",
            modified_files.join("\n")
        ));
    }
    if sections.is_empty() {
        String::new()
    } else {
        format!("\n\n{}", sections.join("\n\n"))
    }
}

const TOOL_RESULT_MAX_CHARS: usize = 2000;

pub fn serialize_conversation(messages: &[Message]) -> String {
    let mut parts = Vec::new();

    for message in messages {
        match message {
            Message::User(message) => {
                let content = match &message.content {
                    UserMessageContent::Text(text) => text.clone(),
                    UserMessageContent::Parts(parts) => parts
                        .iter()
                        .filter_map(|part| match part {
                            UserContent::Text(text) => Some(text.text.as_str()),
                            UserContent::Image(_) => None,
                        })
                        .collect::<String>(),
                };
                if !content.is_empty() {
                    parts.push(format!("[User]: {content}"));
                }
            }
            Message::Assistant(message) => {
                let mut text_parts = Vec::new();
                let mut thinking_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for block in &message.content {
                    match block {
                        AssistantContent::Text(text) => text_parts.push(text.text.clone()),
                        AssistantContent::Thinking(thinking) => {
                            thinking_parts.push(thinking.thinking.clone());
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let args = match &tool_call.arguments {
                                Value::Object(object) => object
                                    .iter()
                                    .map(|(key, value)| {
                                        format!("{key}={}", safe_json_stringify(value))
                                    })
                                    .collect::<Vec<_>>()
                                    .join(", "),
                                _ => String::new(),
                            };
                            tool_calls.push(format!("{}({args})", tool_call.name));
                        }
                    }
                }

                if !thinking_parts.is_empty() {
                    parts.push(format!(
                        "[Assistant thinking]: {}",
                        thinking_parts.join("\n")
                    ));
                }
                if !text_parts.is_empty() {
                    parts.push(format!("[Assistant]: {}", text_parts.join("\n")));
                }
                if !tool_calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
                }
            }
            Message::ToolResult(message) => {
                let content = message
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        ToolResultContent::Text(text) => Some(text.text.as_str()),
                        ToolResultContent::Image(_) => None,
                    })
                    .collect::<String>();
                if !content.is_empty() {
                    parts.push(format!(
                        "[Tool result]: {}",
                        truncate_for_summary(&content, TOOL_RESULT_MAX_CHARS)
                    ));
                }
            }
        }
    }

    parts.join("\n\n")
}

fn safe_json_stringify(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_string())
}

fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    let truncated = text.chars().take(max_chars).collect::<String>();
    let truncated_chars = char_count - max_chars;
    format!("{truncated}\n\n[... {truncated_chars} more characters truncated]")
}
