//! Port of `test/memory-storage.test.ts`.

use std::sync::Arc;

use serde_json::json;

use crate::chord::BACKGROUND_CONTEXT;
use crate::durable::ids::{EntryId, ROOT_CONVERSATION_ID, Seq};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::testing::StorageConformanceProvider;
use crate::durable::types::{ConversationRecord, EntryRecord, Storage, StorageWrite};

mod memory_storage {
    use super::*;

    crate::storage_conformance_tests!({
        let provider: StorageConformanceProvider =
            Arc::new(|test| Box::pin(async move { test(Arc::new(MemoryStorage::new())).await }));
        provider
    });
}

#[tokio::test]
async fn does_not_expose_retained_state_through_a_prepared_commit() {
    let storage = MemoryStorage::new();
    storage
        .commit(
            &[StorageWrite::Conversation {
                value: ConversationRecord::new(ROOT_CONVERSATION_ID),
            }],
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap();
    let entry_id = EntryId(2);
    let mut data = json!({ "nested": [1] });
    let prepared = storage
        .prepare_commit(
            &[StorageWrite::Entry {
                value: EntryRecord {
                    id: entry_id,
                    conversation_id: ROOT_CONVERSATION_ID,
                    kind: "test".into(),
                    data: Some(data.clone()),
                    ..EntryRecord::default()
                },
            }],
            None,
        )
        .unwrap();
    // Divergence: TS checks the exposed writes are frozen; Rust only lends them
    // immutably, so mutating the caller's input is the observable equivalent.
    data["nested"].as_array_mut().unwrap().push(json!(2));
    assert!(matches!(prepared.writes[0], StorageWrite::Entry { .. }));

    assert_eq!(prepared.apply(), Seq(2));
    assert_eq!(prepared.apply(), Seq(2));
    let stored = storage
        .entry(entry_id, &BACKGROUND_CONTEXT)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.entry.data, Some(json!({ "nested": [1] })));
}

#[tokio::test]
async fn runs_the_runner_independent_conformance_cases() {
    let provider: StorageConformanceProvider =
        Arc::new(|test| Box::pin(async move { test(Arc::new(MemoryStorage::new())).await }));
    let cases = crate::durable::testing::create_storage_conformance(provider);
    assert_eq!(cases.len(), 23);
    for case in cases {
        case.run().await;
    }
}
