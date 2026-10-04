//! Port of `test/session-states.test.ts`.
//!
//! Divergences: value identity is `Arc::ptr_eq`; subtree identity
//! (`retained`) and the "without freezing" case are JS-only and skipped.

use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::*;
use crate::chord::delta::Op;
use crate::chord::{JsonValue, ListenerOutcome};
use crate::durable::documents::{define_doc, define_doc_family};
use crate::durable::types::{
    ConversationOwnership, DocDefinition, DocFamilyDefinition, EntryDraft, LatestConversation,
    LatestFork, SessionScope,
};
use crate::durable::{DocFamilyToken, DocToken, SessionImpl};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Retained {
    label: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct State {
    value: i64,
    retained: Retained,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Value {
    value: i64,
}

fn state_doc() -> DocToken<State, SessionScope> {
    define_doc(DocDefinition::new("state.state", 1, SessionScope, || {
        State {
            value: 0,
            retained: Retained {
                label: "stable".into(),
            },
        }
    }))
    .unwrap()
}

fn family_doc() -> DocFamilyToken<Value, String, SessionScope> {
    define_doc_family(DocFamilyDefinition::new(
        "state.family",
        1,
        SessionScope,
        |seed: String| Value {
            value: seed.len() as i64,
        },
    ))
    .unwrap()
}

async fn create_state() -> TestSession {
    let test = open_test_session();
    commit(&test.session, |tx| async move {
        tx.doc(&state_doc(), ()).await.map(|_| ())
    })
    .await;
    flush().await;
    test
}

async fn set_value(session: &SessionImpl, value: i64) {
    commit(session, move |tx| async move {
        tx.doc(&state_doc(), ())
            .await?
            .edit(|state| state.value = value)
    })
    .await;
}

fn number(value: &Option<Arc<JsonValue>>) -> Option<i64> {
    value.as_ref().and_then(|value| value["value"].as_i64())
}

#[tokio::test]
async fn never_creates_an_absent_document() {
    let test = open_test_session();
    let commits = test.storage.commit_count();
    assert!(
        test.session
            .document_state(&state_doc(), (), &context())
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        test.session
            .document_state(&family_doc(), "missing".into(), &context())
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(test.storage.mints(), 0);
}

#[tokio::test]
async fn returns_an_immediately_hydrated_read_only_state_with_contiguous_chord_deliveries() {
    let test = create_state().await;
    let session = &test.session;
    let baseline = session
        .snapshot_json(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let state = session
        .document_state(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let deliveries: Arc<Mutex<Vec<(Option<i64>, u64)>>> = Arc::default();
    let sink = deliveries.clone();
    let _unsubscribe = state.subscribe(move |value, _, delivery| {
        sink.lock().push((number(&value), delivery.sequence));
        ListenerOutcome::ok()
    });
    assert!(Arc::ptr_eq(state.value().as_ref().unwrap(), &baseline));

    set_value(session, 1).await;
    set_value(session, 2).await;
    flush().await;

    assert_eq!(
        *deliveries.lock(),
        [(Some(0), 0), (Some(1), 1), (Some(2), 2)]
    );
    let snapshot = session
        .snapshot_json(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(state.value().as_ref().unwrap(), &snapshot));
    state.dispose().unwrap();
}

#[tokio::test]
async fn creates_independent_disposable_states_for_one_incarnation() {
    let test = create_state().await;
    let session = &test.session;
    let first = session
        .document_state(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let second = session
        .document_state(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    set_value(session, 1).await;
    flush().await;
    assert_eq!(number(&first.value()), Some(1));
    assert_eq!(number(&second.value()), Some(1));

    first.dispose().unwrap();
    set_value(session, 2).await;
    flush().await;
    assert_eq!(number(&first.value()), Some(1));
    assert_eq!(number(&second.value()), Some(2));
    second.dispose().unwrap();
}

#[tokio::test]
async fn shares_exact_committed_value_and_operation_references_with_chord() {
    let test = create_state().await;
    let session = &test.session;
    let state = session
        .document_state(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let received: Arc<Mutex<Option<Arc<[Op]>>>> = Arc::default();
    let sink = received.clone();
    let _unsubscribe = state.internals().subscribe(move |ops, _, _| {
        *sink.lock() = Some(ops.clone());
        Ok(())
    });
    set_value(session, 4).await;
    flush().await;
    let published = document_changes(&test.last_publication())[0].clone();
    assert!(Arc::ptr_eq(
        state.value().as_ref().unwrap(),
        published.value.as_ref().unwrap()
    ));
    assert!(Arc::ptr_eq(
        received.lock().as_ref().unwrap(),
        &published.ops
    ));
    state.dispose().unwrap();
}

#[tokio::test]
async fn captures_a_late_baseline_without_redelivering_an_already_covered_commit() {
    let test = create_state().await;
    let session = &test.session;
    set_value(session, 1).await;
    let state = session
        .document_state(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let deliveries: Arc<Mutex<Vec<u64>>> = Arc::default();
    let sink = deliveries.clone();
    let _unsubscribe = state.subscribe(move |_, _, delivery| {
        sink.lock().push(delivery.sequence);
        ListenerOutcome::ok()
    });
    flush().await;
    assert_eq!(number(&state.value()), Some(1));
    assert_eq!(*deliveries.lock(), [0]);
    state.dispose().unwrap();
}

#[tokio::test]
async fn publishes_null_retirement_and_never_follows_a_replacement_incarnation() {
    let test = create_state().await;
    let session = &test.session;
    let old_state = session
        .document_state(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let retirement: Arc<Mutex<Option<Arc<[Op]>>>> = Arc::default();
    let sink = retirement.clone();
    let _unsubscribe = old_state.internals().subscribe(move |ops, _, _| {
        *sink.lock() = Some(ops.clone());
        Ok(())
    });
    commit(session, |tx| async move {
        tx.retire_doc(&state_doc(), ()).await?;
        tx.doc(&state_doc(), ())
            .await?
            .edit(|state| state.value = 10)
    })
    .await;
    flush().await;
    assert!(old_state.value().is_none());
    assert_eq!(
        to_json(&retirement.lock().as_ref().unwrap().to_vec()),
        json!([["r", null]])
    );

    let replacement = session
        .document_state(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(number(&replacement.value()), Some(10));
    set_value(session, 11).await;
    flush().await;
    assert!(old_state.value().is_none());
    assert_eq!(number(&replacement.value()), Some(11));
    old_state.dispose().unwrap();
    replacement.dispose().unwrap();
}

#[tokio::test]
async fn cold_loads_a_definition_free_fork_copy() {
    let copied: DocToken<Value, LatestConversation> = define_doc(DocDefinition::new(
        "state.copied",
        1,
        LatestConversation {
            fork: LatestFork::Current,
        },
        Value::default,
    ))
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let token = copied.clone();
    let entry = commit(session, move |tx| async move {
        let created = tx.append_entry(parent_id, EntryDraft::new("point")).await?;
        tx.doc(&token, parent_id)
            .await?
            .edit(|value| value.value = 7)?;
        Ok(created)
    })
    .await;
    let child_id = commit(session, move |tx| async move {
        Ok(tx
            .fork_conversation(parent_id, entry.id, ConversationOwnership::Ownerless)
            .await?
            .id)
    })
    .await;
    let reads = test.storage.document_reads();
    let state = session
        .document_state(&copied, child_id, &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(*state.value().unwrap(), json!({ "value": 7 }));
    assert!(test.storage.document_reads() > reads);
    state.dispose().unwrap();
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Migrated {
    value: i64,
    migrated: bool,
}

#[tokio::test]
async fn hydrates_a_migrated_tracker_without_writing_and_skips_an_equal_version_base_update() {
    let old: DocToken<Value, SessionScope> = define_doc(DocDefinition::new(
        "state.migration",
        1,
        SessionScope,
        || Value { value: 3 },
    ))
    .unwrap();
    let current: DocToken<Migrated, SessionScope> = define_doc(
        DocDefinition::new("state.migration", 2, SessionScope, Migrated::default).migrate(
            |value, _| {
                Ok(Migrated {
                    value: value["value"].as_i64().unwrap(),
                    migrated: true,
                })
            },
        ),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let token = old.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, ()).await.map(|_| ())
    })
    .await;
    session.unload_documents().await;
    let commits = test.storage.commit_count();
    let state = session
        .document_state(&current, (), &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        *state.value().unwrap(),
        json!({ "value": 3, "migrated": true })
    );
    assert_eq!(test.storage.commit_count(), commits);
    let baseline = state.value().unwrap();

    let token = current.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, ()).await.map(|_| ())
    })
    .await;
    flush().await;
    assert_eq!(test.storage.commit_count(), commits + 1);
    assert!(Arc::ptr_eq(&state.value().unwrap(), &baseline));
    let token = current.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, ()).await?.edit(|value| value.value = 4)
    })
    .await;
    flush().await;
    assert_eq!(
        *state.value().unwrap(),
        json!({ "value": 4, "migrated": true })
    );
    state.dispose().unwrap();
}

#[tokio::test]
async fn continues_from_exact_committed_values_after_the_tracker_cache_unloads() {
    let test = create_state().await;
    let session = &test.session;
    let state = session
        .document_state(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let baseline = state.value().unwrap();
    let reads = test.storage.document_reads();
    session.unload_documents().await;
    let reloaded = session
        .snapshot_json(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reloaded, baseline);
    assert!(!Arc::ptr_eq(&reloaded, &baseline));
    assert!(test.storage.document_reads() > reads);
    set_value(session, 6).await;
    flush().await;
    assert_eq!(number(&state.value()), Some(6));
    state.dispose().unwrap();
}

#[tokio::test]
async fn exposes_trusted_shared_immutable_values() {
    let test = create_state().await;
    let snapshot = test
        .session
        .snapshot_json(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let state = test
        .session
        .document_state(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(state.value().as_ref().unwrap(), &snapshot));
    state.dispose().unwrap();
}
