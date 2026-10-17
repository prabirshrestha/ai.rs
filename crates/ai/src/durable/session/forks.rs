//! Port of durable `src/session/forks.ts`: selecting the conversation
//! documents a fork copies.

use std::collections::HashSet;

use crate::chord::Context;

use crate::durable::documents::address_id;
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, DocumentId, EntryId};
use crate::durable::types::{
    Cursor, DocumentCopySource, DocumentCreate, DocumentPoint, DocumentQuery, DocumentScope,
    ForkPolicy, History, Storage,
};

const SCAN_PAGE_SIZE: usize = 256;

/// One definition-free document copy to create with a forked conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct ForkDocumentCopy {
    pub record: DocumentCreate,
    pub source: DocumentCopySource,
}

/// Select every persisted conversation document copied by one fork.
pub async fn prepare_fork_document_copies(
    storage: &dyn Storage,
    parent_conversation_id: ConversationId,
    at: EntryId,
    child_conversation_id: ConversationId,
    context: &Context,
) -> Result<Vec<ForkDocumentCopy>> {
    let Some(entry) = storage
        .entry_in(parent_conversation_id, at, context)
        .await?
    else {
        return Err(Error::message(format!(
            "Entry {at} is not visible from conversation {parent_conversation_id}"
        )));
    };

    let mut copies = Vec::new();
    let mut copied_addresses = HashSet::new();
    collect_copies(
        storage,
        entry.entry.conversation_id,
        DocumentPoint::Seq(entry.commit_seq),
        ForkPolicy::AsOf,
        child_conversation_id,
        &mut copies,
        &mut copied_addresses,
        context,
    )
    .await?;
    collect_copies(
        storage,
        parent_conversation_id,
        DocumentPoint::Current,
        ForkPolicy::Current,
        child_conversation_id,
        &mut copies,
        &mut copied_addresses,
        context,
    )
    .await?;
    Ok(copies)
}

#[allow(clippy::too_many_arguments)]
async fn collect_copies(
    storage: &dyn Storage,
    conversation_id: ConversationId,
    at: DocumentPoint,
    policy: ForkPolicy,
    child_conversation_id: ConversationId,
    copies: &mut Vec<ForkDocumentCopy>,
    copied_addresses: &mut HashSet<String>,
    context: &Context,
) -> Result<()> {
    let query = DocumentQuery {
        scope: DocumentScope::Conversation { conversation_id },
        at,
        kind: None,
    };
    let mut cursor: Option<Cursor> = None;
    loop {
        let page = storage
            .scan_documents(&query, SCAN_PAGE_SIZE, cursor.as_ref(), context)
            .await?;
        for source in page.items {
            if !matches!(source.scope, DocumentScope::Conversation { .. })
                || source.fork != Some(policy)
            {
                continue;
            }
            let id = DocumentId(storage.mint_id().await?);
            let record = DocumentCreate {
                id,
                kind: source.kind.clone(),
                key: source.key.clone(),
                scope: DocumentScope::Conversation {
                    conversation_id: child_conversation_id,
                },
                history: Some(if source.history == Some(History::Latest) {
                    History::Latest
                } else {
                    History::Rewindable
                }),
                fork: source.fork,
            };
            let copy_address = address_id(&record.address());
            if copied_addresses.contains(&copy_address) {
                let member = match &record.key {
                    None => record.kind.clone(),
                    Some(key) => format!("{}/{key}", record.kind),
                };
                return Err(Error::message(format!(
                    "Fork selects multiple source documents for {member}"
                )));
            }
            copied_addresses.insert(copy_address);
            copies.push(ForkDocumentCopy {
                record,
                source: DocumentCopySource { id: source.id, at },
            });
        }
        cursor = page.next;
        if cursor.is_none() {
            return Ok(());
        }
    }
}
