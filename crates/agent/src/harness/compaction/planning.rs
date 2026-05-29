use std::collections::HashMap;

use ai::{
    AssistantContent, Context, Message, Model, ModelThinkingLevel, SimpleStreamOptions, StopReason,
    StreamOptions, ToolResultContent, Usage, UserContent, UserMessage, UserMessageContent,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::utils::{
    FileOperations, compute_file_lists, create_file_ops, extract_file_ops_from_message,
    format_file_operations, serialize_conversation,
};
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

const SUMMARIZATION_PROMPT: &str = r#"The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."#;

const UPDATE_SUMMARIZATION_PROMPT: &str = r#"The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."#;

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = r#"This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."#;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionDetails {
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GenerateSummaryOptions {
    pub model: Model,
    pub reserve_tokens: usize,
    pub api_key: String,
    pub headers: HashMap<String, String>,
    pub cancellation_token: Option<CancellationToken>,
    pub custom_instructions: Option<String>,
    pub previous_summary: Option<String>,
    pub thinking_level: Option<ModelThinkingLevel>,
}

impl GenerateSummaryOptions {
    pub fn new(model: Model, reserve_tokens: usize, api_key: impl Into<String>) -> Self {
        Self {
            model,
            reserve_tokens,
            api_key: api_key.into(),
            headers: HashMap::new(),
            cancellation_token: None,
            custom_instructions: None,
            previous_summary: None,
            thinking_level: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactOptions {
    pub model: Model,
    pub api_key: String,
    pub headers: HashMap<String, String>,
    pub custom_instructions: Option<String>,
    pub cancellation_token: Option<CancellationToken>,
    pub thinking_level: Option<ModelThinkingLevel>,
}

impl CompactOptions {
    pub fn new(model: Model, api_key: impl Into<String>) -> Self {
        Self {
            model,
            api_key: api_key.into(),
            headers: HashMap::new(),
            custom_instructions: None,
            cancellation_token: None,
            thinking_level: None,
        }
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeneratedCompaction {
    pub summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: usize,
    pub details: CompactionDetails,
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

pub async fn generate_summary(
    current_messages: &[Message],
    options: GenerateSummaryOptions,
) -> CompactionResult<String> {
    let mut base_prompt = if options.previous_summary.is_some() {
        UPDATE_SUMMARIZATION_PROMPT.to_string()
    } else {
        SUMMARIZATION_PROMPT.to_string()
    };
    if let Some(custom_instructions) = options.custom_instructions.as_deref() {
        base_prompt.push_str("\n\nAdditional focus: ");
        base_prompt.push_str(custom_instructions);
    }

    let conversation_text = serialize_conversation(current_messages);
    let mut prompt_text = format!("<conversation>\n{conversation_text}\n</conversation>\n\n");
    if let Some(previous_summary) = options.previous_summary.as_deref() {
        prompt_text.push_str(&format!(
            "<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        ));
    }
    prompt_text.push_str(&base_prompt);

    let max_tokens = summary_max_tokens(options.reserve_tokens, 0.8, options.model.max_tokens);
    let response = run_summarization_request(
        options.model,
        prompt_text,
        max_tokens,
        options.api_key,
        options.headers,
        options.cancellation_token,
        options.thinking_level,
    )
    .await
    .map_err(|error| {
        CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            format!("Summarization failed: {error}"),
        )
    })?;

    if response.stop_reason == StopReason::Aborted {
        return Err(CompactionError::new(
            CompactionErrorCode::Aborted,
            response
                .error_message
                .unwrap_or_else(|| "Summarization aborted".to_string()),
        ));
    }
    if response.stop_reason == StopReason::Error {
        return Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            format!(
                "Summarization failed: {}",
                response
                    .error_message
                    .unwrap_or_else(|| "Unknown error".to_string())
            ),
        ));
    }

    Ok(extract_text_content(&response.content))
}

pub async fn compact(
    preparation: CompactionPreparation,
    options: CompactOptions,
) -> CompactionResult<GeneratedCompaction> {
    if preparation.first_kept_entry_id.is_empty() {
        return Err(CompactionError::new(
            CompactionErrorCode::InvalidSession,
            "First kept entry has no UUID - session may need migration",
        ));
    }

    let summary = if preparation.is_split_turn && !preparation.turn_prefix_messages.is_empty() {
        let history_options = GenerateSummaryOptions {
            model: options.model.clone(),
            reserve_tokens: preparation.settings.reserve_tokens,
            api_key: options.api_key.clone(),
            headers: options.headers.clone(),
            cancellation_token: options.cancellation_token.clone(),
            custom_instructions: options.custom_instructions.clone(),
            previous_summary: preparation.previous_summary.clone(),
            thinking_level: options.thinking_level,
        };
        let turn_prefix_options = GenerateSummaryOptions {
            model: options.model.clone(),
            reserve_tokens: preparation.settings.reserve_tokens,
            api_key: options.api_key.clone(),
            headers: options.headers.clone(),
            cancellation_token: options.cancellation_token.clone(),
            custom_instructions: None,
            previous_summary: None,
            thinking_level: options.thinking_level,
        };
        let (history_summary, turn_prefix_summary) = tokio::try_join!(
            async {
                if preparation.messages_to_summarize.is_empty() {
                    Ok("No prior history.".to_string())
                } else {
                    generate_summary(&preparation.messages_to_summarize, history_options).await
                }
            },
            generate_turn_prefix_summary(&preparation.turn_prefix_messages, turn_prefix_options)
        )?;
        format!(
            "{history_summary}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_summary}"
        )
    } else {
        generate_summary(
            &preparation.messages_to_summarize,
            GenerateSummaryOptions {
                model: options.model,
                reserve_tokens: preparation.settings.reserve_tokens,
                api_key: options.api_key,
                headers: options.headers,
                cancellation_token: options.cancellation_token,
                custom_instructions: options.custom_instructions,
                previous_summary: preparation.previous_summary.clone(),
                thinking_level: options.thinking_level,
            },
        )
        .await?
    };

    let (read_files, modified_files) = compute_file_lists(&preparation.file_ops);
    let mut summary = summary;
    summary.push_str(&format_file_operations(&read_files, &modified_files));

    Ok(GeneratedCompaction {
        summary,
        first_kept_entry_id: preparation.first_kept_entry_id,
        tokens_before: preparation.tokens_before,
        details: CompactionDetails {
            read_files,
            modified_files,
        },
    })
}

async fn generate_turn_prefix_summary(
    messages: &[Message],
    options: GenerateSummaryOptions,
) -> CompactionResult<String> {
    let conversation_text = serialize_conversation(messages);
    let prompt_text = format!(
        "<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    );
    let max_tokens = summary_max_tokens(options.reserve_tokens, 0.5, options.model.max_tokens);
    let response = run_summarization_request(
        options.model,
        prompt_text,
        max_tokens,
        options.api_key,
        options.headers,
        options.cancellation_token,
        options.thinking_level,
    )
    .await
    .map_err(|error| {
        CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            format!("Turn prefix summarization failed: {error}"),
        )
    })?;

    if response.stop_reason == StopReason::Aborted {
        return Err(CompactionError::new(
            CompactionErrorCode::Aborted,
            response
                .error_message
                .unwrap_or_else(|| "Turn prefix summarization aborted".to_string()),
        ));
    }
    if response.stop_reason == StopReason::Error {
        return Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            format!(
                "Turn prefix summarization failed: {}",
                response
                    .error_message
                    .unwrap_or_else(|| "Unknown error".to_string())
            ),
        ));
    }

    Ok(extract_text_content(&response.content))
}

async fn run_summarization_request(
    model: Model,
    prompt_text: String,
    max_tokens: u32,
    api_key: String,
    headers: HashMap<String, String>,
    cancellation_token: Option<CancellationToken>,
    thinking_level: Option<ModelThinkingLevel>,
) -> ai::Result<ai::AssistantMessage> {
    let reasoning = if model.reasoning {
        thinking_level.filter(|level| *level != ModelThinkingLevel::Off)
    } else {
        None
    };
    ai::complete_simple(
        model,
        Context {
            system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
            messages: vec![Message::User(UserMessage {
                content: UserMessageContent::Parts(vec![UserContent::text(prompt_text)]),
                timestamp: ai::utils::time::now_millis(),
            })],
            tools: Vec::new(),
        },
        Some(SimpleStreamOptions {
            stream: StreamOptions {
                max_tokens: Some(max_tokens),
                cancellation_token,
                api_key: Some(api_key),
                headers,
                ..Default::default()
            },
            reasoning,
            ..Default::default()
        }),
    )
    .await
}

fn summary_max_tokens(reserve_tokens: usize, ratio: f64, model_max_tokens: u32) -> u32 {
    let reserved = ((reserve_tokens as f64) * ratio).floor() as u64;
    let reserved = reserved.min(u32::MAX as u64) as u32;
    if model_max_tokens > 0 {
        reserved.min(model_max_tokens)
    } else {
        reserved
    }
}

fn extract_text_content(content: &[AssistantContent]) -> String {
    content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
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
