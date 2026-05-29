use std::collections::HashSet;

use ai::{Message, UserContent, UserMessage, UserMessageContent};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::planning::estimate_tokens;
use super::utils::{FileOperations, create_file_ops, extract_file_ops_from_message};
use crate::harness::session::Session;
use crate::harness::types::{SessionError, SessionErrorCode, SessionResult, SessionTreeEntry};

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
