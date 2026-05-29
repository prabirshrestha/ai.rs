use std::collections::{HashMap, HashSet};

use ai::{
    AssistantContent, Context, Message, Model, SimpleStreamOptions, StopReason, StreamOptions,
    UserContent, UserMessage, UserMessageContent,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::planning::{SUMMARIZATION_SYSTEM_PROMPT, estimate_tokens};
use super::utils::{
    FileOperations, compute_file_lists, create_file_ops, extract_file_ops_from_message,
    format_file_operations, serialize_conversation,
};
use crate::harness::session::Session;
use crate::harness::types::{
    BranchSummaryError, BranchSummaryErrorCode, BranchSummaryGenerationResult, SessionError,
    SessionErrorCode, SessionResult, SessionTreeEntry,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryDetails {
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BranchPreparation {
    pub messages: Vec<Message>,
    pub file_ops: FileOperations,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollectEntriesResult {
    pub entries: Vec<SessionTreeEntry>,
    pub common_ancestor_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GenerateBranchSummaryOptions {
    pub model: Model,
    pub api_key: String,
    pub headers: HashMap<String, String>,
    pub cancellation_token: Option<CancellationToken>,
    pub custom_instructions: Option<String>,
    pub replace_instructions: bool,
    pub reserve_tokens: usize,
}

impl GenerateBranchSummaryOptions {
    pub fn new(model: Model, api_key: impl Into<String>) -> Self {
        Self {
            model,
            api_key: api_key.into(),
            headers: HashMap::new(),
            cancellation_token: None,
            custom_instructions: None,
            replace_instructions: false,
            reserve_tokens: DEFAULT_BRANCH_SUMMARY_RESERVE_TOKENS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryResult {
    pub summary: String,
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

pub async fn collect_entries_for_branch_summary<TMetadata>(
    session: &Session<TMetadata>,
    old_leaf_id: Option<&str>,
    target_id: &str,
) -> SessionResult<CollectEntriesResult>
where
    TMetadata: Clone + Send + Sync + 'static,
{
    let Some(old_leaf_id) = old_leaf_id else {
        return Ok(CollectEntriesResult {
            entries: Vec::new(),
            common_ancestor_id: None,
        });
    };

    let old_path = session
        .get_branch(Some(old_leaf_id))
        .await?
        .into_iter()
        .map(|entry| entry.id().to_string())
        .collect::<HashSet<_>>();
    let target_path = session.get_branch(Some(target_id)).await?;
    let common_ancestor_id = target_path
        .iter()
        .rev()
        .find(|entry| old_path.contains(entry.id()))
        .map(|entry| entry.id().to_string());

    let mut entries = Vec::new();
    let mut current = Some(old_leaf_id.to_string());
    while let Some(current_id) = current {
        if common_ancestor_id.as_deref() == Some(current_id.as_str()) {
            break;
        }
        let Some(entry) = session.get_entry(&current_id).await? else {
            return Err(SessionError::new(
                SessionErrorCode::InvalidSession,
                format!("Entry {current_id} not found"),
            ));
        };
        current = entry.parent_id().map(ToString::to_string);
        entries.push(entry);
    }
    entries.reverse();

    Ok(CollectEntriesResult {
        entries,
        common_ancestor_id,
    })
}

pub fn prepare_branch_entries(
    entries: &[SessionTreeEntry],
    token_budget: Option<usize>,
) -> BranchPreparation {
    let token_budget = token_budget.unwrap_or_default();
    let mut messages = Vec::new();
    let mut file_ops = create_file_ops();
    let mut total_tokens = 0;

    for entry in entries {
        if let SessionTreeEntry::BranchSummary(entry) = entry
            && entry.from_hook != Some(true)
            && let Some(details) = entry.details.as_ref()
        {
            apply_file_details(details, &mut file_ops);
        }
    }

    for entry in entries.iter().rev() {
        let Some(message) = get_message_from_entry(entry) else {
            continue;
        };
        extract_file_ops_from_message(&message, &mut file_ops);
        let tokens = estimate_tokens(&message);
        if token_budget > 0 && total_tokens + tokens > token_budget {
            if matches!(
                entry,
                SessionTreeEntry::Compaction(_) | SessionTreeEntry::BranchSummary(_)
            ) && total_tokens.saturating_mul(10) < token_budget.saturating_mul(9)
            {
                messages.insert(0, message);
                total_tokens += tokens;
            }
            break;
        }
        messages.insert(0, message);
        total_tokens += tokens;
    }

    BranchPreparation {
        messages,
        file_ops,
        total_tokens,
    }
}

const DEFAULT_BRANCH_SUMMARY_RESERVE_TOKENS: usize = 16_384;

const BRANCH_SUMMARY_PREAMBLE: &str = "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

const BRANCH_SUMMARY_PROMPT: &str = r#"Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."#;

pub async fn generate_branch_summary(
    entries: &[SessionTreeEntry],
    options: GenerateBranchSummaryOptions,
) -> BranchSummaryGenerationResult<BranchSummaryResult> {
    let context_window = if options.model.context_window == 0 {
        128_000
    } else {
        options.model.context_window as usize
    };
    let token_budget = context_window.saturating_sub(options.reserve_tokens);
    let BranchPreparation {
        messages, file_ops, ..
    } = prepare_branch_entries(entries, Some(token_budget));

    if messages.is_empty() {
        return Ok(BranchSummaryResult {
            summary: "No content to summarize".to_string(),
            read_files: Vec::new(),
            modified_files: Vec::new(),
        });
    }

    let conversation_text = serialize_conversation(&messages);
    let instructions = match (
        options.replace_instructions,
        options.custom_instructions.as_deref(),
    ) {
        (true, Some(custom_instructions)) => custom_instructions.to_string(),
        (false, Some(custom_instructions)) => {
            format!("{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}")
        }
        _ => BRANCH_SUMMARY_PROMPT.to_string(),
    };
    let prompt_text =
        format!("<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}");
    let summarization_messages = vec![Message::User(UserMessage {
        content: UserMessageContent::Parts(vec![UserContent::text(prompt_text)]),
        timestamp: ai::utils::time::now_millis(),
    })];

    let response = ai::complete_simple(
        options.model,
        Context {
            system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
            messages: summarization_messages,
            tools: Vec::new(),
        },
        Some(SimpleStreamOptions {
            stream: StreamOptions {
                max_tokens: Some(2048),
                cancellation_token: options.cancellation_token,
                api_key: Some(options.api_key),
                headers: options.headers,
                ..Default::default()
            },
            ..Default::default()
        }),
    )
    .await
    .map_err(|error| {
        BranchSummaryError::new(
            BranchSummaryErrorCode::SummarizationFailed,
            format!("Branch summary failed: {error}"),
        )
    })?;

    if response.stop_reason == StopReason::Aborted {
        return Err(BranchSummaryError::new(
            BranchSummaryErrorCode::Aborted,
            response
                .error_message
                .unwrap_or_else(|| "Branch summary aborted".to_string()),
        ));
    }
    if response.stop_reason == StopReason::Error {
        return Err(BranchSummaryError::new(
            BranchSummaryErrorCode::SummarizationFailed,
            format!(
                "Branch summary failed: {}",
                response
                    .error_message
                    .unwrap_or_else(|| "Unknown error".to_string())
            ),
        ));
    }

    let mut summary = response
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    summary = format!("{BRANCH_SUMMARY_PREAMBLE}{summary}");
    let (read_files, modified_files) = compute_file_lists(&file_ops);
    summary.push_str(&format_file_operations(&read_files, &modified_files));

    Ok(BranchSummaryResult {
        summary: if summary.is_empty() {
            "No summary generated".to_string()
        } else {
            summary
        },
        read_files,
        modified_files,
    })
}

fn get_message_from_entry(entry: &SessionTreeEntry) -> Option<Message> {
    match entry {
        SessionTreeEntry::Message(entry) if matches!(entry.message, Message::ToolResult(_)) => None,
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

fn apply_file_details(details: &Value, file_ops: &mut FileOperations) {
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
