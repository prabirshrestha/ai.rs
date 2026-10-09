//! Port of durable `src/harness/context.ts`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::chord::Context;
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, EntryId};
use crate::durable::session::SessionImpl;
use crate::durable::types::{ContextEdit, ContextEditAction, EntryQuery, EntryRecord, Storage};
use crate::types::{
    AssistantContent, Message, StopReason, TextContent, ToolCall, ToolResultMessage, UserContent,
};

use super::types::ContextView;

const SCAN_PAGE_SIZE: usize = 256;
const MISSING_RESULT_TEXT: &str =
    "Tool result unavailable: history ends before this call completed.";

fn excluded(reason: StopReason) -> bool {
    matches!(
        reason,
        StopReason::Aborted | StopReason::Error | StopReason::Deferred
    )
}

/// Head marker and newest visible entry that fix one committed context range.
#[derive(Debug, Clone)]
pub struct ContextBounds {
    pub head: Option<EntryRecord>,
    pub tail: EntryId,
}

/// Capture the bounds of the current context, or of the context cut off at the visible entry `at`, with two O(1)
/// reads. Run this on the Session line; entries at or below the tail are immutable, so `derive_context()` can then scan
/// them off the line.
pub async fn capture_context_bounds(
    storage: &Arc<dyn Storage>,
    conversation_id: ConversationId,
    context: &Context,
    at: Option<EntryId>,
) -> Result<Option<ContextBounds>> {
    let tail = match at {
        None => {
            let page = storage
                .scan_entries(&EntryQuery::conversation(conversation_id), 1, None, context)
                .await?;
            match page.items.first() {
                None => return Ok(None),
                Some(entry) => entry.id,
            }
        }
        Some(at) => {
            if storage
                .entry_in(conversation_id, at, context)
                .await?
                .is_none()
            {
                return Err(Error::message(format!(
                    "Entry {at} is not visible from conversation {conversation_id}"
                )));
            }
            at
        }
    };
    let head = storage
        .find_latest_head_marker(conversation_id, Some(tail), context)
        .await?;
    Ok(Some(ContextBounds { head, tail }))
}

/// Committed context of one conversation: bounds captured on the Session line, entries derived off it.
pub async fn read_context(
    session: &SessionImpl,
    conversation_id: ConversationId,
    context: &Context,
    at: Option<EntryId>,
) -> Result<ContextView> {
    let storage = session.storage().clone();
    let bounds = {
        let storage = storage.clone();
        let context = context.clone();
        session
            .read_on_line(async move {
                capture_context_bounds(&storage, conversation_id, &context, at).await
            })
            .await?
    };
    derive_context(&storage, conversation_id, bounds.as_ref(), context).await
}

/// Derive the active transcript and model context of one conversation within captured bounds.
///
/// H = newest visible head marker; the range runs from `H.head` (or transcript start) through the tail. Per target,
/// the newest edit in the range wins. Context entries are H followed by the range's non-head entries.
pub async fn derive_context(
    storage: &Arc<dyn Storage>,
    conversation_id: ConversationId,
    bounds: Option<&ContextBounds>,
    context: &Context,
) -> Result<ContextView> {
    let Some(bounds) = bounds else {
        return Ok(ContextView::default());
    };
    let range = scan_range(storage, conversation_id, bounds, context).await?;
    let mut edits: HashMap<EntryId, &ContextEdit> = HashMap::new();
    // Edits of every entry in the range count, including older head markers that `select_active()` drops.
    for entry in &range {
        for edit in entry.edits.iter().flatten() {
            edits.insert(edit.target, edit);
        }
    }
    let entries = select_active(bounds.head.as_ref(), &range);
    let contributions: Vec<Vec<Message>> = entries
        .iter()
        .map(|entry| {
            let contributed = match edits.get(&entry.id).map(|edit| &edit.action) {
                Some(ContextEditAction::Omit) => return Vec::new(),
                Some(ContextEditAction::Replace { messages }) => messages.clone(),
                None => entry.model.clone().unwrap_or_default(),
            };
            contributed
                .into_iter()
                .filter(|message| match message {
                    Message::Assistant(assistant) => !excluded(assistant.stop_reason),
                    _ => true,
                })
                .collect()
        })
        .collect();
    let messages = order_tool_results(&contributions.concat());
    Ok(ContextView {
        head: bounds.head.clone(),
        entries,
        contributions,
        messages,
    })
}

/// The raw active entries within captured bounds, without deriving model context.
pub async fn active_entries(
    storage: &Arc<dyn Storage>,
    conversation_id: ConversationId,
    bounds: Option<&ContextBounds>,
    context: &Context,
) -> Result<Vec<EntryRecord>> {
    let Some(bounds) = bounds else {
        return Ok(Vec::new());
    };
    let range = scan_range(storage, conversation_id, bounds, context).await?;
    Ok(select_active(bounds.head.as_ref(), &range))
}

/// Visible entries from the head marker's head, or transcript start, through the tail, oldest first.
async fn scan_range(
    storage: &Arc<dyn Storage>,
    conversation_id: ConversationId,
    bounds: &ContextBounds,
    context: &Context,
) -> Result<Vec<EntryRecord>> {
    let query = EntryQuery {
        conversation_id,
        min_entry_id: bounds.head.as_ref().and_then(|head| head.head),
        max_entry_id: Some(bounds.tail),
    };
    let mut range = Vec::new();
    let mut cursor = None;
    loop {
        let page = storage
            .scan_entries(&query, SCAN_PAGE_SIZE, cursor.as_ref(), context)
            .await?;
        range.extend(page.items);
        cursor = page.next;
        if cursor.is_none() {
            break;
        }
    }
    range.reverse();
    Ok(range)
}

/// The head marker followed by the range's non-head entries, or the whole range without a marker.
fn select_active(head: Option<&EntryRecord>, range: &[EntryRecord]) -> Vec<EntryRecord> {
    match head {
        None => range.to_vec(),
        Some(head) => std::iter::once(head.clone())
            .chain(range.iter().filter(|entry| entry.head.is_none()).cloned())
            .collect(),
    }
}

/// Place each assistant's tool results directly after it in call order. Results are taken from the messages before
/// the next assistant; a missing result is synthesized and unmatched results are dropped.
pub fn order_tool_results(messages: &[Message]) -> Vec<Message> {
    let mut ordered = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if matches!(message, Message::ToolResult(_)) {
            continue;
        }
        ordered.push(message.clone());
        let Message::Assistant(assistant) = message else {
            continue;
        };
        let calls: Vec<&ToolCall> = assistant
            .content
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();
        if calls.is_empty() {
            continue;
        }
        let mut results: HashMap<&str, usize> = HashMap::new();
        for (next, candidate) in messages.iter().enumerate().skip(index + 1) {
            match candidate {
                Message::Assistant(_) => break,
                Message::ToolResult(result) => {
                    results.entry(result.tool_call_id.as_str()).or_insert(next);
                }
                _ => {}
            }
        }
        for call in calls {
            ordered.push(match results.get(call.id.as_str()) {
                None => missing_result(call, assistant.timestamp),
                Some(&index) => messages[index].clone(),
            });
        }
    }
    ordered
}

fn missing_result(call: &ToolCall, timestamp: u64) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content: vec![UserContent::Text(TextContent::new(MISSING_RESULT_TEXT))],
        details: Some(serde_json::json!({ "reason": "missing_result" })),
        usage: None,
        nested_calls: None,
        is_error: true,
        timestamp,
    })
}
