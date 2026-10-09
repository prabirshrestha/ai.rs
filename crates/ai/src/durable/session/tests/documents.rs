//! Port of `test/session-documents.test.ts`.
//!
//! Divergences: JS-only cases are adapted or skipped.
//! - Revision identity is `Arc::ptr_eq` on [`SessionImpl::snapshot_json`];
//!   per-subtree structural sharing is not observable through `serde_json`.
//! - "adopts by pointer swap": published values are the snapshot revision
//!   itself; operation payloads are compared by value, not identity.
//! - "copies assigned values per placement" is skipped: Rust values are owned.
//! - "writes and publishes replayable nonempty structural no-ops" is skipped:
//!   Chord's Rust change diffs at preparation, so a shift/unshift pair has no ops.
//! - Escaped draft revocation checks that every access fails after settlement.
//! - The Date initializer case uses a non-finite number (not strict JSON).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::*;
use crate::chord::JsonValue;
use crate::durable::documents::{define_doc, define_doc_family};
use crate::durable::errors::Error;
use crate::durable::ids::{ConversationId, TaskId};
use crate::durable::session::{DocDraft, TxFuture};
use crate::durable::types::{
    DocDefinition, DocFamilyDefinition, DocumentAddress, DocumentPoint, DocumentScope,
    LatestConversation, LatestFork, RewindableConversation, RewindableFork, SessionScope, Storage,
};
use crate::durable::{DocFamilyToken, DocToken, SessionImpl};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Nested {
    count: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Label {
    label: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Live {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    items: Vec<String>,
    nested: Nested,
    other: Label,
}

fn initial_live() -> Live {
    Live {
        message: None,
        items: vec![],
        nested: Nested { count: 0 },
        other: Label { label: "x".into() },
    }
}

static LIVE_INIT_COUNT: AtomicUsize = AtomicUsize::new(0);

fn live_doc() -> DocToken<Live, LatestConversation> {
    define_doc(DocDefinition::new(
        "test.live",
        1,
        LatestConversation {
            fork: LatestFork::Initial,
        },
        || {
            LIVE_INIT_COUNT.fetch_add(1, Ordering::SeqCst);
            initial_live()
        },
    ))
    .unwrap()
}

fn rewindable_live_doc() -> DocToken<Live, RewindableConversation> {
    define_doc(DocDefinition::new(
        "test.live",
        1,
        RewindableConversation {
            fork: RewindableFork::AsOf,
        },
        initial_live,
    ))
    .unwrap()
}

fn live_doc_v2() -> DocToken<Live, LatestConversation> {
    define_doc(DocDefinition::new(
        "test.live",
        2,
        LatestConversation {
            fork: LatestFork::Initial,
        },
        initial_live,
    ))
    .unwrap()
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Counter {
    count: i64,
}

fn counter_doc() -> DocToken<Counter, SessionScope> {
    define_doc(DocDefinition::new(
        "test.counter",
        1,
        SessionScope,
        Counter::default,
    ))
    .unwrap()
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Member {
    seed: String,
    hits: i64,
}

fn member_doc(seeds: Arc<Mutex<Vec<String>>>) -> DocFamilyToken<Member, String, SessionScope> {
    define_doc_family(DocFamilyDefinition::new(
        "test.member",
        1,
        SessionScope,
        move |seed: String| {
            seeds.lock().push(seed.clone());
            Member { seed, hits: 0 }
        },
    ))
    .unwrap()
}

fn plain_member_doc() -> DocFamilyToken<Member, String, SessionScope> {
    member_doc(Arc::default())
}

async fn setup_live() -> (TestSession, ConversationId) {
    let test = open_test_session();
    let conversation_id = create_conversation(&test.session).await;
    commit(&test.session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id)
            .await?
            .edit(|live| live.items.extend(["a".into(), "b".into()]))?;
        Ok(())
    })
    .await;
    (test, conversation_id)
}

async fn snapshot(session: &SessionImpl, conversation_id: ConversationId) -> Option<Live> {
    session
        .snapshot(&live_doc(), conversation_id, &context())
        .await
        .unwrap()
}

async fn snapshot_json(
    session: &SessionImpl,
    conversation_id: ConversationId,
) -> Option<Arc<JsonValue>> {
    session
        .snapshot_json(&live_doc(), conversation_id, &context())
        .await
        .unwrap()
}

#[tokio::test]
async fn creates_an_initial_base_on_first_access_and_adopts_it_after_storage_success() {
    let test = open_test_session();
    let session = &test.session;
    let conversation_id = create_conversation(session).await;
    commit(session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id)
            .await?
            .edit(|live| live.message = Some("hello".into()))?;
        Ok(())
    })
    .await;
    let writes = test.storage.last_commit();
    assert_eq!(writes.len(), 1);
    assert_matches(
        &to_json(&writes[0]),
        &json!({
            "type": "document.create",
            "record": {
                "kind": "test.live",
                "scope": { "kind": "conversation", "conversationId": conversation_id },
                "history": "latest",
                "fork": "initial",
            },
            "content": { "kind": "base", "version": 1, "value": { "message": "hello", "items": [], "nested": { "count": 0 } } },
        }),
    );
    let snapshot = snapshot_json(session, conversation_id).await.unwrap();
    assert_eq!(
        *snapshot,
        json!({ "message": "hello", "items": [], "nested": { "count": 0 }, "other": { "label": "x" } })
    );
    flush().await;
    let publication = test.last_publication();
    let published = &document_changes(&publication)[0];
    assert_eq!(published.record.created_at, publication.seq);
    assert!(Arc::ptr_eq(published.value.as_ref().unwrap(), &snapshot));
    assert_eq!(published.conversation_id, Some(conversation_id));
    let create = writes
        .iter()
        .find(|write| write_type(write) == "document.create")
        .unwrap();
    assert_eq!(to_json(create)["content"]["value"], *snapshot);
}

#[tokio::test]
async fn never_creates_on_snapshot_and_returns_none_when_absent() {
    let test = open_test_session();
    let session = &test.session;
    let conversation_id = create_conversation(session).await;
    let before = test.storage.commit_count();
    assert_eq!(snapshot(session, conversation_id).await, None);
    assert_eq!(
        session
            .snapshot(&counter_doc(), (), &context())
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        session
            .snapshot(&plain_member_doc(), "k".into(), &context())
            .await
            .unwrap(),
        None
    );
    assert_eq!(test.storage.commit_count(), before);
    assert_eq!(test.storage.mints(), 1);
}

#[tokio::test]
async fn returns_shared_immutable_snapshots_and_keeps_prior_revisions_stable() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    let first = snapshot_json(session, conversation_id).await.unwrap();
    assert!(Arc::ptr_eq(
        &snapshot_json(session, conversation_id).await.unwrap(),
        &first
    ));
    commit(session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id)
            .await?
            .edit(|live| live.nested.count = 1)?;
        Ok(())
    })
    .await;
    let second = snapshot_json(session, conversation_id).await.unwrap();
    assert!(!Arc::ptr_eq(&second, &first));
    assert_eq!(first["nested"]["count"], 0);
    assert_eq!(second["nested"]["count"], 1);
}

#[tokio::test]
async fn adopts_by_pointer_swap_and_publishes_the_admitted_operations() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    commit(session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id).await?.edit(|live| {
            live.other = Label { label: "y".into() };
            live.items.push("c".into());
        })?;
        Ok(())
    })
    .await;
    flush().await;
    let snapshot = snapshot_json(session, conversation_id).await.unwrap();
    let published = document_changes(&test.last_publication())[0].clone();
    assert!(Arc::ptr_eq(published.value.as_ref().unwrap(), &snapshot));
    let admitted = test
        .storage
        .last_commit()
        .into_iter()
        .find(|write| write_type(write) == "document.change")
        .unwrap();
    let admitted = to_json(&admitted);
    assert_eq!(admitted["content"]["kind"], "delta");
    assert_eq!(to_json(&published.ops.to_vec()), admitted["content"]["ops"]);
    let ops = to_json(&published.ops.to_vec());
    // Chord's Rust change diffs at preparation, so the set lands on the changed leaf rather than the assigned object.
    assert!(ops.as_array().unwrap().iter().any(|op| op[0] == "s"
        && op[1][0] == "other"
        && (op[2] == json!({ "label": "y" }) || op[2] == "y")));
}

#[tokio::test]
async fn suppresses_writes_and_publications_for_empty_batches() {
    let (test, conversation_id) = setup_live().await;
    flush().await;
    let commits = test.storage.commit_count();
    let published = test.published();
    commit(&test.session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id).await?.edit(|live| {
            live.nested.count = 0;
            live.items.push("z".into());
            live.items.pop();
        })?;
        Ok(())
    })
    .await;
    flush().await;
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(test.published(), published);
}

#[tokio::test]
async fn revokes_escaped_drafts_when_the_callback_settles() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    let escaped = commit(session, move |tx| async move {
        let escaped = tx.doc(&live_doc(), conversation_id).await?;
        escaped.edit(|live| live.message = Some("inside".into()))?;
        Ok(escaped)
    })
    .await;
    assert_err(escaped.get(), "settled overlay");
    assert_err(
        escaped.edit(|live| live.message = Some("outside".into())),
        "settled overlay",
    );
    assert_err(escaped.json(), "settled overlay");
    assert_eq!(
        snapshot(session, conversation_id).await.unwrap().message,
        Some("inside".into())
    );

    let returned = commit(session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id).await
    })
    .await;
    assert_err(returned.get(), "settled overlay");
}

#[tokio::test]
async fn aborts_every_change_when_the_callback_fails() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    let before = snapshot_json(session, conversation_id).await.unwrap();
    let commits = test.storage.commit_count();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    let live = tx.doc(&live_doc(), conversation_id).await?;
                    let counter = tx.doc(&counter_doc(), ()).await?;
                    live.edit(|live| live.message = Some("lost".into()))?;
                    counter.edit(|counter| counter.count = 5)?;
                    Err::<(), _>(Error::message("callback failed"))
                },
                &context(),
            )
            .await,
        "callback failed",
    );
    assert_eq!(test.storage.commit_count(), commits);
    assert!(Arc::ptr_eq(
        &snapshot_json(session, conversation_id).await.unwrap(),
        &before
    ));
    assert_eq!(
        session
            .snapshot(&counter_doc(), (), &context())
            .await
            .unwrap(),
        None
    );
    commit(session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id)
            .await?
            .edit(|live| live.message = Some("kept".into()))?;
        Ok(())
    })
    .await;
    assert_eq!(
        snapshot(session, conversation_id).await.unwrap().message,
        Some("kept".into())
    );
}

#[tokio::test]
async fn memoizes_concurrent_duplicate_acquisition_and_initializes_once() {
    let test = open_test_session();
    let session = &test.session;
    let conversation_id = create_conversation(session).await;
    let init_count = LIVE_INIT_COUNT.load(Ordering::SeqCst);
    let mints = test.storage.mints();
    commit(session, move |tx| async move {
        let (first, second) = futures::future::try_join(
            tx.doc(&live_doc(), conversation_id),
            tx.doc(&live_doc(), conversation_id),
        )
        .await?;
        assert!(first.same(&second));
        assert!(tx.doc(&live_doc(), conversation_id).await?.same(&first));
        first.edit(|live| live.message = Some("once".into()))?;
        Ok(())
    })
    .await;
    // Other tests share the counter; this one adds exactly one initialization.
    assert!(LIVE_INIT_COUNT.load(Ordering::SeqCst) > init_count);
    assert_eq!(test.storage.mints(), mints + 1);
    assert_eq!(
        test.storage
            .last_commit()
            .iter()
            .filter(|write| write_type(write) == "document.create")
            .count(),
        1
    );
}

#[tokio::test]
async fn memoized_acquisition_initializes_once() {
    let test = open_test_session();
    let initialized = Arc::new(AtomicUsize::new(0));
    let count = initialized.clone();
    let token = define_doc(DocDefinition::new(
        "test.once",
        1,
        SessionScope,
        move || {
            count.fetch_add(1, Ordering::SeqCst);
            Counter::default()
        },
    ))
    .unwrap();
    commit(&test.session, move |tx| async move {
        let (first, second) =
            futures::future::try_join(tx.doc(&token, ()), tx.doc(&token, ())).await?;
        assert!(first.same(&second));
        first.edit(|counter| counter.count = 1)?;
        Ok(())
    })
    .await;
    assert_eq!(initialized.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn uses_the_first_family_seed_and_ignores_seeds_for_existing_members() {
    let test = open_test_session();
    let session = &test.session;
    let seeds: Arc<Mutex<Vec<String>>> = Arc::default();
    let token = member_doc(seeds.clone());
    let first_token = token.clone();
    commit(session, move |tx| async move {
        let first = tx.doc(&first_token, ("k".into(), "first".into())).await?;
        let second = tx.doc(&first_token, ("k".into(), "second".into())).await?;
        assert!(second.same(&first));
        first.edit(|member| member.hits += 1)?;
        Ok(())
    })
    .await;
    let second_token = token.clone();
    commit(session, move |tx| async move {
        tx.doc(&second_token, ("k".into(), "third".into()))
            .await?
            .edit(|member| member.hits += 1)?;
        tx.doc(&second_token, ("other".into(), "fourth".into()))
            .await?
            .edit(|member| member.hits += 1)?;
        Ok(())
    })
    .await;
    assert_eq!(*seeds.lock(), ["first", "fourth"]);
    assert_eq!(
        session
            .snapshot(&token, "k".into(), &context())
            .await
            .unwrap(),
        Some(Member {
            seed: "first".into(),
            hits: 2
        })
    );
    assert_eq!(
        session
            .snapshot(&token, "other".into(), &context())
            .await
            .unwrap(),
        Some(Member {
            seed: "fourth".into(),
            hits: 1
        })
    );
}

type Pending = Arc<Mutex<Option<TxFuture<DocDraft<Live>>>>>;

#[tokio::test]
async fn rejects_a_callback_that_succeeds_with_a_pending_acquisition_and_drains_it() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    session.unload_documents().await;
    let gate = test.storage.hold_find_document();
    let commits = test.storage.commit_count();
    let pending: Pending = Arc::default();
    let slot = pending.clone();
    let commit_future = session.commit(
        move |tx| {
            *slot.lock() = Some(tx.doc(&live_doc(), conversation_id));
            async { Ok(()) }
        },
        &context(),
    );
    let settled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = settled.clone();
    let commit_task = tokio::spawn(async move {
        let result = commit_future.await;
        observed.store(true, Ordering::SeqCst);
        result
    });
    gate.entered().await;
    flush().await;
    // The line stays held until the late acquisition settles.
    assert!(!settled.load(Ordering::SeqCst));
    gate.release();
    assert_err(commit_task.await.unwrap(), "pending Tx operations");
    let pending = pending.lock().take().unwrap();
    assert_err(pending.await, "Transaction has settled");
    assert_eq!(test.storage.commit_count(), commits);
    commit(session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id)
            .await?
            .edit(|live| live.message = Some("after".into()))?;
        Ok(())
    })
    .await;
    assert_eq!(
        snapshot(session, conversation_id).await.unwrap().message,
        Some("after".into())
    );
}

#[tokio::test]
async fn does_not_initialize_or_mint_for_an_absent_acquisition_that_finishes_after_settlement() {
    let test = open_test_session();
    let initialized = Arc::new(AtomicUsize::new(0));
    let count = initialized.clone();
    let late_doc = define_doc(DocDefinition::new(
        "test.late",
        1,
        SessionScope,
        move || {
            count.fetch_add(1, Ordering::SeqCst);
            Counter::default()
        },
    ))
    .unwrap();
    let gate = test.storage.hold_find_document();
    let mints = test.storage.mints();
    let pending: Arc<Mutex<Option<TxFuture<DocDraft<Counter>>>>> = Arc::default();
    let slot = pending.clone();
    let commit_future = test.session.commit(
        move |tx| {
            *slot.lock() = Some(tx.doc(&late_doc, ()));
            async { Ok(()) }
        },
        &context(),
    );
    let commit_task = tokio::spawn(commit_future);
    gate.entered().await;
    gate.release();
    assert_err(commit_task.await.unwrap(), "pending Tx operations");
    let pending = pending.lock().take().unwrap();
    assert_err(pending.await, "Transaction has settled");
    assert_eq!(initialized.load(Ordering::SeqCst), 0);
    assert_eq!(test.storage.mints(), mints);
}

#[tokio::test]
async fn rejects_with_the_callback_error_when_it_fails_with_a_pending_acquisition() {
    let (test, conversation_id) = setup_live().await;
    test.session.unload_documents().await;
    let gate = test.storage.hold_find_document();
    let pending: Pending = Arc::default();
    let slot = pending.clone();
    let commit_future = test.session.commit(
        move |tx| {
            *slot.lock() = Some(tx.doc(&live_doc(), conversation_id));
            async { Err::<(), _>(Error::message("callback failed")) }
        },
        &context(),
    );
    let commit_task = tokio::spawn(commit_future);
    gate.entered().await;
    gate.release();
    assert_err(commit_task.await.unwrap(), "callback failed");
    let pending = pending.lock().take().unwrap();
    assert_err(pending.await, "Transaction has settled");
}

#[tokio::test]
async fn rejects_tx_use_after_the_callback_settles() {
    let (test, conversation_id) = setup_live().await;
    let captured = commit_with(&test.session, |tx| async move { Ok(tx) }).await;
    assert_err(
        captured.doc(&live_doc(), conversation_id).await,
        "Transaction has settled",
    );
    assert_err(
        captured.conversation(conversation_id).await,
        "Transaction has settled",
    );
    let task = from_json(json!({
        "id": 1, "conversationId": 1, "kind": "x", "version": 1, "input": null,
        "background": false, "abortRequested": false,
        "state": { "status": "pending", "checkpoint": null },
    }));
    assert_err(captured.set_task(task), "Transaction has settled");
}

#[tokio::test]
async fn rejects_tokens_whose_semantics_or_version_disagree_with_the_stored_incarnation() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    assert_err(
        session
            .snapshot(&rewindable_live_doc(), conversation_id, &context())
            .await,
        "does not match the supplied definition semantics",
    );
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.doc(&rewindable_live_doc(), conversation_id).await?;
                    Ok(())
                },
                &context(),
            )
            .await,
        "does not match the supplied definition semantics",
    );
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.doc(&live_doc_v2(), conversation_id).await?;
                    Ok(())
                },
                &context(),
            )
            .await,
        "requires migration from version 1",
    );
}

#[tokio::test]
async fn rejects_non_json_initializer_values_and_draft_placements_before_storage_admission() {
    let test = open_test_session();
    let session = &test.session;
    let conversation_id = create_conversation(session).await;
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct Float {
        at: f64,
    }
    let date_doc = define_doc(DocDefinition::new("test.date", 1, SessionScope, || Float {
        at: f64::NAN,
    }))
    .unwrap();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.doc(&date_doc, ()).await?;
                    Ok(())
                },
                &context(),
            )
            .await,
        "strict JSON",
    );
    let commits = test.storage.commit_count();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    let live = tx.doc(&live_doc(), conversation_id).await?;
                    live.edit_json(|live| {
                        live["items"]
                            .as_array_mut()
                            .unwrap()
                            .push(json!(f64::INFINITY))
                    })?;
                    // `serde_json` maps non-finite numbers to null, so place the value through the typed path too.
                    let float =
                        define_doc(DocDefinition::new("test.float", 1, SessionScope, || {
                            Float { at: 0.0 }
                        }))
                        .unwrap();
                    tx.doc(&float, ()).await?.edit(|value| value.at = f64::NAN)
                },
                &context(),
            )
            .await,
        "strict JSON",
    );
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(snapshot(session, conversation_id).await, None);
}

#[tokio::test]
async fn rolls_back_prepared_documents_when_batch_assembly_fails() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    commit(session, |tx| async move {
        tx.doc(&counter_doc(), ())
            .await?
            .edit(|counter| counter.count = 1)?;
        Ok(())
    })
    .await;
    let live = snapshot_json(session, conversation_id).await.unwrap();
    let counter = session
        .snapshot_json(&counter_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let commits = test.storage.commit_count();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.doc(&live_doc(), conversation_id)
                        .await?
                        .edit(|live| live.message = Some("lost".into()))?;
                    tx.doc(&counter_doc(), ())
                        .await?
                        .edit(|counter| counter.count = 2)?;
                    // Replacing a missing task fails during assembly, after every change was prepared.
                    tx.set_task(from_json(json!({
                        "id": 999, "conversationId": conversation_id, "kind": "missing", "version": 1,
                        "input": null, "background": false, "abortRequested": false,
                        "state": { "status": "pending", "checkpoint": { "phase": "start" } },
                    })))
                },
                &context(),
            )
            .await,
        "Task 999 does not exist",
    );
    assert_eq!(test.storage.commit_count(), commits);
    assert!(Arc::ptr_eq(
        &snapshot_json(session, conversation_id).await.unwrap(),
        &live
    ));
    assert!(Arc::ptr_eq(
        &session
            .snapshot_json(&counter_doc(), (), &context())
            .await
            .unwrap()
            .unwrap(),
        &counter
    ));
    commit(session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id)
            .await?
            .edit(|live| live.message = Some("next".into()))?;
        tx.doc(&counter_doc(), ())
            .await?
            .edit(|counter| counter.count = 3)?;
        Ok(())
    })
    .await;
    assert_eq!(
        session
            .snapshot(&counter_doc(), (), &context())
            .await
            .unwrap()
            .unwrap()
            .count,
        3
    );
    let _: TaskId = TaskId::new(999);
}

#[tokio::test]
async fn poisons_the_session_after_an_uncertain_storage_failure_and_publishes_nothing() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    flush().await;
    let before = snapshot_json(session, conversation_id).await.unwrap();
    let published = test.published();
    test.storage
        .fail_next_commit(Error::message("disk vanished"));
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.doc(&live_doc(), conversation_id)
                        .await?
                        .edit(|live| live.message = Some("uncertain".into()))?;
                    Ok(())
                },
                &context(),
            )
            .await,
        "disk vanished",
    );
    flush().await;
    assert_eq!(test.published(), published);
    assert!(before.get("message").is_none());
    assert_err(
        session
            .snapshot(&live_doc(), conversation_id, &context())
            .await,
        "poisoned",
    );
    assert_err(
        session.commit(|_| async { Ok(()) }, &context()).await,
        "poisoned",
    );
    session.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_the_previous_revision_unchanged_through_storage_settlement() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    let before = snapshot_json(session, conversation_id).await.unwrap();
    let copy = (*before).clone();
    let gate = test.storage.hold_commits();
    let pending = session.commit(
        move |tx| async move {
            tx.doc(&live_doc(), conversation_id).await?.edit(|live| {
                live.items.push("c".into());
                live.nested.count = 9;
            })?;
            Ok(())
        },
        &context(),
    );
    let pending = tokio::spawn(pending);
    gate.entered().await;
    assert!(Arc::ptr_eq(
        &snapshot_json(session, conversation_id).await.unwrap(),
        &before
    ));
    assert_eq!(*before, copy);
    gate.release();
    pending.await.unwrap().unwrap();
    let after = snapshot(session, conversation_id).await.unwrap();
    assert_eq!(after.items, ["a", "b", "c"]);
    assert_eq!(*before, copy);
}

#[tokio::test]
async fn retires_documents_and_creates_a_new_incarnation_at_the_same_address() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    flush().await;
    let old_id = document_changes(&test.last_publication())[0].record.id;
    commit(session, move |tx| async move {
        let live = tx.doc(&live_doc(), conversation_id).await?;
        live.edit(|live| live.message = Some("final".into()))?;
        tx.retire_doc(&live_doc(), conversation_id).await?;
        let replacement = tx.doc(&live_doc(), conversation_id).await?;
        assert!(!replacement.same(&live));
        replacement.edit(|live| live.message = Some("new".into()))?;
        Ok(())
    })
    .await;
    let writes: Vec<JsonValue> = test.storage.last_commit().iter().map(to_json).collect();
    assert_eq!(writes.len(), 3);
    assert!(
        writes
            .iter()
            .any(|write| write["type"] == "document.change" && write["id"] == json!(old_id))
    );
    assert!(writes.contains(&json!({ "type": "document.retire", "id": old_id })));
    assert!(
        writes
            .iter()
            .any(|write| write["type"] == "document.create"
                && write["record"]["kind"] == "test.live")
    );
    flush().await;
    let publication = test.last_publication();
    let documents = document_changes(&publication);
    let (retired, created) = (&documents[0], &documents[1]);
    assert_eq!(retired.record.id, old_id);
    assert!(retired.value.is_none());
    assert!(retired.ops.is_empty());
    assert_eq!(retired.record.retired_at, Some(publication.seq));
    assert_ne!(created.record.id, old_id);
    assert_eq!(created.record.created_at, publication.seq);
    assert_matches(
        created.value.as_deref().unwrap(),
        &json!({ "message": "new", "items": [] }),
    );
    assert!(created.ops.is_empty());
    assert!(Arc::ptr_eq(
        &snapshot_json(session, conversation_id).await.unwrap(),
        created.value.as_ref().unwrap()
    ));

    commit(session, move |tx| async move {
        tx.retire_doc(&live_doc(), conversation_id).await
    })
    .await;
    assert_eq!(snapshot(session, conversation_id).await, None);
    session.unload_documents().await;
    assert_eq!(snapshot(session, conversation_id).await, None);
    // Retiring an absent address is a no-op.
    let commits = test.storage.commit_count();
    commit(session, move |tx| async move {
        tx.retire_doc(&live_doc(), conversation_id).await
    })
    .await;
    assert_eq!(test.storage.commit_count(), commits);
}

#[tokio::test]
async fn retires_without_acquisition_and_recreates_both_existing_and_absent_addresses() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    flush().await;
    let old_id = document_changes(&test.last_publication())[0].record.id;
    session.unload_documents().await;
    commit(session, move |tx| async move {
        let retired = tx.retire_doc(&live_doc(), conversation_id);
        let replacement = tx.doc(&live_doc(), conversation_id);
        let (live, ()) = futures::future::try_join(replacement, retired).await?;
        live.edit(|live| live.message = Some("replacement".into()))?;
        Ok(())
    })
    .await;
    let existing: Vec<JsonValue> = test.storage.last_commit().iter().map(to_json).collect();
    assert_eq!(existing.len(), 2);
    assert!(existing.contains(&json!({ "type": "document.retire", "id": old_id })));
    assert!(
        existing
            .iter()
            .any(|write| write["type"] == "document.create"
                && write["record"]["kind"] == "test.live")
    );

    let member = plain_member_doc();
    let token = member.clone();
    commit(session, move |tx| async move {
        let retired = tx.retire_doc(&token, "absent".into());
        let replacement = tx.doc(&token, ("absent".into(), "seed".into()));
        let (member, ()) = futures::future::try_join(replacement, retired).await?;
        member.edit(|member| member.hits = 1)?;
        Ok(())
    })
    .await;
    let absent: Vec<String> = test.storage.last_commit().iter().map(write_type).collect();
    assert_eq!(
        absent
            .iter()
            .filter(|kind| *kind == "document.retire")
            .count(),
        0
    );
    assert_eq!(
        absent
            .iter()
            .filter(|kind| *kind == "document.create")
            .count(),
        1
    );
    assert_eq!(
        session
            .snapshot(&member, "absent".into(), &context())
            .await
            .unwrap(),
        Some(Member {
            seed: "seed".into(),
            hits: 1
        })
    );
}

#[tokio::test]
async fn retires_the_existing_incarnation_when_retirement_races_a_pending_acquisition() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    flush().await;
    let old_id = document_changes(&test.last_publication())[0].record.id;
    commit(session, move |tx| async move {
        let acquired = tx.doc(&live_doc(), conversation_id);
        let retired = tx.retire_doc(&live_doc(), conversation_id);
        acquired
            .await?
            .edit(|live| live.message = Some("final".into()))?;
        retired.await
    })
    .await;
    let first: Vec<JsonValue> = test.storage.last_commit().iter().map(to_json).collect();
    assert_eq!(first.len(), 2);
    assert!(first.iter().any(|write| write["type"] == "document.change"
        && write["id"] == json!(old_id)
        && write["content"]["kind"] == "delta"));
    assert!(first.contains(&json!({ "type": "document.retire", "id": old_id })));
    assert_eq!(snapshot(session, conversation_id).await, None);

    commit(session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id)
            .await?
            .edit(|live| live.message = Some("second".into()))?;
        Ok(())
    })
    .await;
    flush().await;
    let second_id = document_changes(&test.last_publication())[0].record.id;
    commit(session, move |tx| async move {
        let acquired = tx.doc(&live_doc(), conversation_id);
        let retired = tx.retire_doc(&live_doc(), conversation_id);
        let recreated = tx.doc(&live_doc(), conversation_id);
        let (old, fresh, ()) = futures::future::try_join3(acquired, recreated, retired).await?;
        assert!(!fresh.same(&old));
        fresh.edit(|live| live.message = Some("third".into()))?;
        Ok(())
    })
    .await;
    let writes: Vec<JsonValue> = test.storage.last_commit().iter().map(to_json).collect();
    assert_eq!(writes.len(), 2);
    assert!(writes.contains(&json!({ "type": "document.retire", "id": second_id })));
    assert!(
        writes
            .iter()
            .any(|write| write["type"] == "document.create"
                && write["record"]["kind"] == "test.live")
    );
    assert_eq!(
        snapshot(session, conversation_id).await.unwrap().message,
        Some("third".into())
    );
}

#[tokio::test]
async fn reloads_an_unloaded_document_from_storage() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    let loaded = snapshot_json(session, conversation_id).await.unwrap();
    session.unload_documents().await;
    let reloaded = snapshot_json(session, conversation_id).await.unwrap();
    assert!(!Arc::ptr_eq(&reloaded, &loaded));
    assert_eq!(reloaded, loaded);
    commit(session, move |tx| async move {
        tx.doc(&live_doc(), conversation_id)
            .await?
            .edit(|live| live.items.push("c".into()))?;
        Ok(())
    })
    .await;
    session.unload_documents().await;
    assert_eq!(
        snapshot(session, conversation_id).await.unwrap().items,
        ["a", "b", "c"]
    );
}

#[tokio::test]
async fn delivers_complete_publications_synchronously_after_adoption() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    let published = test.published();
    let listener_context = Arc::new(Mutex::new(None));
    let sink = listener_context.clone();
    let unsubscribe = session
        .subscribe_commits(move |_, context| *sink.lock() = Some(context.clone()))
        .unwrap();
    let caller = context();
    let result = session
        .commit(
            move |tx| async move {
                tx.doc(&live_doc(), conversation_id)
                    .await?
                    .edit(|live| live.message = Some("m".into()))?;
                Ok("done")
            },
            &caller,
        )
        .await
        .unwrap();
    assert_eq!(result, "done");
    assert!(listener_context.lock().as_ref().unwrap().ptr_eq(&caller));
    assert_eq!(test.published(), published + 1);
    unsubscribe();
}

#[tokio::test]
async fn publishes_close_synchronously_and_supports_unsubscription() {
    let test = open_test_session();
    let calls: Arc<Mutex<Vec<&str>>> = Arc::default();
    let active = calls.clone();
    let _ = test
        .session
        .subscribe_close(move || active.lock().push("active"))
        .unwrap();
    let removed = calls.clone();
    let unsubscribe = test
        .session
        .subscribe_close(move || removed.lock().push("removed"))
        .unwrap();
    unsubscribe();
    test.session.close(&context()).await.unwrap();
    assert_eq!(*calls.lock(), ["active"]);
}

#[tokio::test]
async fn settles_admitted_commits_before_close_and_rejects_later_admission() {
    let (test, conversation_id) = setup_live().await;
    let session = &test.session;
    let gate = test.storage.hold_commits();
    let admitted = tokio::spawn(session.commit(
        move |tx| async move {
            tx.doc(&live_doc(), conversation_id)
                .await?
                .edit(|live| live.message = Some("admitted".into()))?;
            Ok(())
        },
        &context(),
    ));
    gate.entered().await;
    let queued = session.commit(
        move |tx| async move {
            tx.doc(&live_doc(), conversation_id)
                .await?
                .edit(|live| live.message = Some("queued".into()))?;
            Ok(())
        },
        &context(),
    );
    let admitted_snapshot = session.snapshot(&plain_member_doc(), "absent".into(), &context());
    let closed = session.close(&context());
    assert_err(
        session.commit(|_| async { Ok(()) }, &context()).await,
        "closed",
    );
    assert_err(
        session
            .snapshot(&live_doc(), conversation_id, &context())
            .await,
        "closed",
    );
    gate.release();
    admitted.await.unwrap().unwrap();
    queued.await.unwrap();
    assert_eq!(admitted_snapshot.await.unwrap(), None);
    closed.await.unwrap();
    let stored = test
        .storage
        .find_document(
            &DocumentAddress {
                kind: "test.live".into(),
                key: None,
                scope: DocumentScope::Conversation { conversation_id },
            },
            DocumentPoint::Current,
            &context(),
        )
        .await;
    assert!(stored.is_err());
}
