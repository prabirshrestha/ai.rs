use ai::{
    AssistantContent, Message, StopReason, ToolResultContent, Usage, UserContent, UserMessage,
    UserMessageContent,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::utils::{FileOperations, create_file_ops, extract_file_ops_from_message};
use crate::harness::session::build_session_context;
use crate::harness::types::{
    CompactionError, CompactionErrorCode, CompactionResult, SessionTreeEntry,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSettings {
    pub enabled: bool,
    pub reserve_tokens: usize,
    pub keep_recent_tokens: usize,
}

pub const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = CompactionSettings {
    enabled: true,
    reserve_tokens: 16_384,
    keep_recent_tokens: 20_000,
};

pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

pub fn calculate_context_tokens(usage: &Usage) -> usize {
    let total = usage.total_tokens;
    if total > 0 {
        total as usize
    } else {
        (usage.input + usage.output + usage.cache_read + usage.cache_write) as usize
    }
}

pub fn get_last_assistant_usage(entries: &[SessionTreeEntry]) -> Option<Usage> {
    entries.iter().rev().find_map(|entry| match entry {
        SessionTreeEntry::Message(entry) => get_assistant_usage(&entry.message),
        _ => None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextUsageEstimate {
    pub tokens: usize,
    pub usage_tokens: usize,
    pub trailing_tokens: usize,
    pub last_usage_index: Option<usize>,
}

pub fn estimate_context_tokens(messages: &[Message]) -> ContextUsageEstimate {
    let usage_info = messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| get_assistant_usage(message).map(|usage| (index, usage)));

    let Some((last_usage_index, usage)) = usage_info else {
        let estimated = messages.iter().map(estimate_tokens).sum();
        return ContextUsageEstimate {
            tokens: estimated,
            usage_tokens: 0,
            trailing_tokens: estimated,
            last_usage_index: None,
        };
    };

    let usage_tokens = calculate_context_tokens(&usage);
    let trailing_tokens = messages
        .iter()
        .skip(last_usage_index + 1)
        .map(estimate_tokens)
        .sum::<usize>();
    ContextUsageEstimate {
        tokens: usage_tokens + trailing_tokens,
        usage_tokens,
        trailing_tokens,
        last_usage_index: Some(last_usage_index),
    }
}

pub fn should_compact(
    context_tokens: usize,
    context_window: usize,
    settings: &CompactionSettings,
) -> bool {
    settings.enabled && context_tokens > context_window.saturating_sub(settings.reserve_tokens)
}

const ESTIMATED_IMAGE_CHARS: usize = 4800;

pub fn estimate_tokens(message: &Message) -> usize {
    let chars = match message {
        Message::User(message) => estimate_user_content_chars(&message.content),
        Message::Assistant(message) => message
            .content
            .iter()
            .map(|block| match block {
                AssistantContent::Text(text) => text.text.chars().count(),
                AssistantContent::Thinking(thinking) => thinking.thinking.chars().count(),
                AssistantContent::ToolCall(tool_call) => {
                    tool_call.name.chars().count()
                        + safe_json_stringify(&tool_call.arguments).chars().count()
                }
            })
            .sum(),
        Message::ToolResult(message) => message
            .content
            .iter()
            .map(|content| match content {
                ToolResultContent::Text(text) => text.text.chars().count(),
                ToolResultContent::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
    };
    chars.div_ceil(4)
}

pub fn find_turn_start_index(
    entries: &[SessionTreeEntry],
    entry_index: usize,
    start_index: usize,
) -> Option<usize> {
    (start_index..=entry_index).rev().find(|index| {
        let entry = &entries[*index];
        matches!(
            entry,
            SessionTreeEntry::BranchSummary(_) | SessionTreeEntry::CustomMessage(_)
        ) || matches!(
            entry,
            SessionTreeEntry::Message(message) if matches!(message.message, Message::User(_))
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CutPointResult {
    pub first_kept_entry_index: usize,
    pub turn_start_index: Option<usize>,
    pub is_split_turn: bool,
}

pub fn find_cut_point(
    entries: &[SessionTreeEntry],
    start_index: usize,
    end_index: usize,
    keep_recent_tokens: usize,
) -> CutPointResult {
    let cut_points = find_valid_cut_points(entries, start_index, end_index);
    if cut_points.is_empty() {
        return CutPointResult {
            first_kept_entry_index: start_index,
            turn_start_index: None,
            is_split_turn: false,
        };
    }

    let mut accumulated_tokens = 0;
    let mut cut_index = cut_points[0];
    for index in (start_index..end_index).rev() {
        let SessionTreeEntry::Message(entry) = &entries[index] else {
            continue;
        };
        accumulated_tokens += estimate_tokens(&entry.message);
        if accumulated_tokens >= keep_recent_tokens {
            if let Some(candidate) = cut_points.iter().find(|candidate| **candidate >= index) {
                cut_index = *candidate;
            }
            break;
        }
    }

    while cut_index > start_index {
        match &entries[cut_index - 1] {
            SessionTreeEntry::Compaction(_) | SessionTreeEntry::Message(_) => break,
            _ => cut_index -= 1,
        }
    }

    let is_user_message = matches!(
        &entries[cut_index],
        SessionTreeEntry::Message(entry) if matches!(entry.message, Message::User(_))
    );
    let turn_start_index = if is_user_message {
        None
    } else {
        find_turn_start_index(entries, cut_index, start_index)
    };

    CutPointResult {
        first_kept_entry_index: cut_index,
        turn_start_index,
        is_split_turn: !is_user_message && turn_start_index.is_some(),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPreparation {
    pub first_kept_entry_id: String,
    pub messages_to_summarize: Vec<Message>,
    pub turn_prefix_messages: Vec<Message>,
    pub is_split_turn: bool,
    pub tokens_before: usize,
    pub previous_summary: Option<String>,
    pub file_ops: FileOperations,
    pub settings: CompactionSettings,
}

pub fn prepare_compaction(
    path_entries: &[SessionTreeEntry],
    settings: CompactionSettings,
) -> CompactionResult<Option<CompactionPreparation>> {
    if path_entries.is_empty()
        || matches!(path_entries.last(), Some(SessionTreeEntry::Compaction(_)))
    {
        return Ok(None);
    }

    let prev_compaction_index = path_entries
        .iter()
        .rposition(|entry| matches!(entry, SessionTreeEntry::Compaction(_)));

    let mut previous_summary = None;
    let mut boundary_start = 0;
    if let Some(index) = prev_compaction_index {
        let SessionTreeEntry::Compaction(entry) = &path_entries[index] else {
            unreachable!();
        };
        previous_summary = Some(entry.summary.clone());
        boundary_start = path_entries
            .iter()
            .position(|candidate| candidate.id() == entry.first_kept_entry_id)
            .unwrap_or(index + 1);
    }

    let boundary_end = path_entries.len();
    let tokens_before =
        estimate_context_tokens(&build_session_context(path_entries).messages).tokens;
    let cut_point = find_cut_point(
        path_entries,
        boundary_start,
        boundary_end,
        settings.keep_recent_tokens,
    );
    let Some(first_kept_entry) = path_entries.get(cut_point.first_kept_entry_index) else {
        return Err(CompactionError::new(
            CompactionErrorCode::InvalidSession,
            "First kept entry has no UUID - session may need migration",
        ));
    };
    let first_kept_entry_id = first_kept_entry.id().to_string();

    let history_end = if cut_point.is_split_turn {
        cut_point
            .turn_start_index
            .unwrap_or(cut_point.first_kept_entry_index)
    } else {
        cut_point.first_kept_entry_index
    };
    let messages_to_summarize = (boundary_start..history_end)
        .filter_map(|index| get_message_from_entry_for_compaction(&path_entries[index]))
        .collect::<Vec<_>>();
    let turn_prefix_messages = if let Some(turn_start_index) = cut_point.turn_start_index {
        (turn_start_index..cut_point.first_kept_entry_index)
            .filter_map(|index| get_message_from_entry_for_compaction(&path_entries[index]))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let mut file_ops =
        extract_file_operations(&messages_to_summarize, path_entries, prev_compaction_index);
    if cut_point.is_split_turn {
        for message in &turn_prefix_messages {
            extract_file_ops_from_message(message, &mut file_ops);
        }
    }

    Ok(Some(CompactionPreparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn: cut_point.is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings,
    }))
}

fn get_assistant_usage(message: &Message) -> Option<Usage> {
    let Message::Assistant(message) = message else {
        return None;
    };
    if matches!(message.stop_reason, StopReason::Aborted | StopReason::Error)
        || calculate_context_tokens(&message.usage) == 0
    {
        None
    } else {
        Some(message.usage.clone())
    }
}

fn estimate_user_content_chars(content: &UserMessageContent) -> usize {
    match content {
        UserMessageContent::Text(text) => text.chars().count(),
        UserMessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                UserContent::Text(text) => text.text.chars().count(),
                UserContent::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
    }
}

fn safe_json_stringify(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_string())
}

fn find_valid_cut_points(
    entries: &[SessionTreeEntry],
    start_index: usize,
    end_index: usize,
) -> Vec<usize> {
    let mut cut_points = Vec::new();
    for index in start_index..end_index {
        match &entries[index] {
            SessionTreeEntry::Message(entry) => {
                if !matches!(entry.message, Message::ToolResult(_)) {
                    cut_points.push(index);
                }
            }
            SessionTreeEntry::BranchSummary(_) | SessionTreeEntry::CustomMessage(_) => {
                cut_points.push(index);
            }
            _ => {}
        }
    }
    cut_points
}

fn get_message_from_entry_for_compaction(entry: &SessionTreeEntry) -> Option<Message> {
    match entry {
        SessionTreeEntry::Compaction(_) => None,
        _ => get_message_from_entry(entry),
    }
}

fn get_message_from_entry(entry: &SessionTreeEntry) -> Option<Message> {
    match entry {
        SessionTreeEntry::Message(entry) => Some(entry.message.clone()),
        SessionTreeEntry::CustomMessage(entry) => Some(Message::User(UserMessage {
            content: match &entry.content {
                crate::harness::types::CustomMessageContent::Text(text) => {
                    UserMessageContent::Text(text.clone())
                }
                crate::harness::types::CustomMessageContent::Parts(parts) => {
                    UserMessageContent::Parts(parts.clone())
                }
            },
            timestamp: timestamp_to_millis(&entry.timestamp),
        })),
        SessionTreeEntry::BranchSummary(entry) => Some(summary_message(
            format!(
                "{}{}{}",
                crate::harness::messages::BRANCH_SUMMARY_PREFIX,
                entry.summary,
                crate::harness::messages::BRANCH_SUMMARY_SUFFIX
            ),
            &entry.timestamp,
        )),
        SessionTreeEntry::Compaction(entry) => Some(summary_message(
            format!(
                "{}{}{}",
                crate::harness::messages::COMPACTION_SUMMARY_PREFIX,
                entry.summary,
                crate::harness::messages::COMPACTION_SUMMARY_SUFFIX
            ),
            &entry.timestamp,
        )),
        _ => None,
    }
}

fn summary_message(text: String, timestamp: &str) -> Message {
    Message::User(UserMessage {
        content: UserMessageContent::Parts(vec![UserContent::text(text)]),
        timestamp: timestamp_to_millis(timestamp),
    })
}

fn timestamp_to_millis(timestamp: &str) -> u64 {
    if let Ok(value) = timestamp.parse::<u64>() {
        return value;
    }
    time::OffsetDateTime::parse(timestamp, &time::format_description::well_known::Rfc3339)
        .map(|value| value.unix_timestamp_nanos().max(0) as u64 / 1_000_000)
        .unwrap_or_default()
}

fn extract_file_operations(
    messages: &[Message],
    entries: &[SessionTreeEntry],
    prev_compaction_index: Option<usize>,
) -> FileOperations {
    let mut file_ops = create_file_ops();
    if let Some(index) = prev_compaction_index
        && let Some(SessionTreeEntry::Compaction(entry)) = entries.get(index)
        && entry.from_hook != Some(true)
        && let Some(details) = entry.details.as_ref()
    {
        if let Some(read_files) = details.get("readFiles").and_then(Value::as_array) {
            for path in read_files.iter().filter_map(Value::as_str) {
                file_ops.read.insert(path.to_string());
            }
        }
        if let Some(modified_files) = details.get("modifiedFiles").and_then(Value::as_array) {
            for path in modified_files.iter().filter_map(Value::as_str) {
                file_ops.edited.insert(path.to_string());
            }
        }
    }
    for message in messages {
        extract_file_ops_from_message(message, &mut file_ops);
    }
    file_ops
}
