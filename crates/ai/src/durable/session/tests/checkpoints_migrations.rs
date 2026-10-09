//! Port of `test/session-checkpoints-migrations.test.ts`.
//!
//! Divergences:
//! - `checkpointWhen` has no `this` receiver in Rust; the receiver case checks
//!   the same base selection with a closure.
//! - Exact value/ops identity passed to the predicate is compared by value
//!   (the predicate borrows `&JsonValue`/`&[Op]`).
//! - The nonempty structural no-op half of the predicate case is skipped:
//!   Chord's Rust change diffs at preparation, so a shift/unshift pair is empty.
//! - The Date migration result is a non-finite number (not strict JSON).
//! - The root-replacement case checks that the delta replays; Chord's Rust
//!   diff decides whether a large change is one root replacement.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use futures::FutureExt;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::*;
use crate::chord::JsonValue;
use crate::durable::documents::{define_doc, define_doc_family};
use crate::durable::errors::Error;
use crate::durable::ids::EntryId;
use crate::durable::types::{
    ConversationOwnership, DocDefinition, DocFamilyDefinition, DocumentAddress, DocumentPoint,
    DocumentScope, EntryDraft, RewindableConversation, RewindableFork, SessionScope, Storage,
    StorageWrite,
};
use crate::durable::{DocAccess, DocToken, SessionImpl};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Count {
    count: i64,
}

fn count_doc(kind: &str, version: u32, initial: i64) -> DocDefinition<Count, SessionScope> {
    DocDefinition::new(kind, version, SessionScope, move || Count {
        count: initial,
    })
}

fn document_writes(writes: &[StorageWrite]) -> Vec<JsonValue> {
    writes
        .iter()
        .filter(|write| {
            matches!(
                write_type(write).as_str(),
                "document.create" | "document.change" | "document.retire"
            )
        })
        .map(to_json)
        .collect()
}

fn change_kinds(writes: &[Vec<StorageWrite>]) -> Vec<String> {
    writes
        .iter()
        .flatten()
        .map(to_json)
        .filter(|write| write["type"] == "document.change")
        .map(|write| write["content"]["kind"].as_str().unwrap().to_string())
        .collect()
}

async fn touch<A>(session: &SessionImpl, token: &A)
where
    A: DocAccess<Args = ()> + Clone + 'static,
{
    let token = token.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, ()).await.map(|_| ())
    })
    .await;
}

async fn set_count(session: &SessionImpl, token: &DocToken<Count, SessionScope>, count: i64) {
    let token = token.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, ()).await?.edit(|value| value.count = count)
    })
    .await;
}

async fn snap<A: DocAccess<Address = ()>>(session: &SessionImpl, token: &A) -> Option<A::Value> {
    session.snapshot(token, (), &context()).await.unwrap()
}

// ─── Checkpoints ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn calls_checkpoint_when_for_ordinary_changes() {
    let token = define_doc(
        count_doc("checkpoint.receiver", 1, 0)
            .checkpoint_when(|value, _, _| Ok(value["count"] == 2)),
    )
    .unwrap();
    let test = open_test_session();
    for count in 0..=2 {
        set_count(&test.session, &token, count).await;
    }
    assert_eq!(
        change_kinds(&test.storage.commits.lock()),
        ["delta", "base"]
    );
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Items {
    items: Vec<String>,
}

#[tokio::test]
async fn selects_bases_only_for_nonempty_ordinary_batches_and_passes_the_prepared_revision_and_ops()
{
    type Calls = Arc<Mutex<Vec<(JsonValue, JsonValue)>>>;
    let calls: Calls = Arc::default();
    let false_calls: Calls = Arc::default();
    let sink = calls.clone();
    let base_doc = define_doc(
        DocDefinition::new("checkpoint.base", 1, SessionScope, Items::default).checkpoint_when(
            move |value, ops, _| {
                sink.lock().push((value.clone(), to_json(&ops.to_vec())));
                Ok(true)
            },
        ),
    )
    .unwrap();
    let sink = false_calls.clone();
    let delta_doc = define_doc(count_doc("checkpoint.delta", 1, 0).checkpoint_when(
        move |value, ops, _| {
            sink.lock().push((value.clone(), to_json(&ops.to_vec())));
            Ok(false)
        },
    ))
    .unwrap();
    let default_doc = define_doc(count_doc("checkpoint.default", 1, 0)).unwrap();
    let test = open_test_session();
    let session = &test.session;

    let tokens = (base_doc.clone(), delta_doc.clone(), default_doc.clone());
    commit(session, move |tx| async move {
        let (base, delta, default) = tokens;
        tx.doc(&base, ()).await?;
        tx.doc(&delta, ()).await?;
        tx.doc(&default, ()).await?;
        Ok(())
    })
    .await;
    assert!(calls.lock().is_empty());
    assert!(false_calls.lock().is_empty());
    assert!(
        document_writes(&test.storage.last_commit())
            .iter()
            .all(|write| write["type"] == "document.create" && write["content"]["kind"] == "base")
    );

    let tokens = (base_doc.clone(), delta_doc.clone(), default_doc.clone());
    commit(session, move |tx| async move {
        let (base, delta, default) = tokens;
        tx.doc(&base, ())
            .await?
            .edit(|value| value.items.push("x".into()))?;
        tx.doc(&delta, ()).await?.edit(|value| value.count += 1)?;
        tx.doc(&default, ()).await?.edit(|value| value.count += 1)?;
        Ok(())
    })
    .await;
    flush().await;
    let writes: Vec<JsonValue> = test
        .storage
        .last_commit()
        .iter()
        .map(to_json)
        .filter(|write| write["type"] == "document.change")
        .collect();
    let kinds: Vec<_> = writes
        .iter()
        .map(|write| write["content"]["kind"].clone())
        .collect();
    assert_eq!(kinds, [json!("base"), json!("delta"), json!("delta")]);
    assert_eq!(calls.lock().len(), 1);
    assert_eq!(false_calls.lock().len(), 1);
    let snapshot = session
        .snapshot_json(&base_doc, (), &context())
        .await
        .unwrap()
        .unwrap();
    let delta_snapshot = session
        .snapshot_json(&delta_doc, (), &context())
        .await
        .unwrap()
        .unwrap();
    let documents = document_changes(&test.last_publication());
    let (call, false_call) = (calls.lock()[0].clone(), false_calls.lock()[0].clone());
    assert_eq!(call.0, *snapshot);
    assert_eq!(call.1, to_json(&documents[0].ops.to_vec()));
    assert_eq!(false_call.0, *delta_snapshot);
    assert_eq!(false_call.1, to_json(&documents[1].ops.to_vec()));
    assert_eq!(writes[1]["content"]["ops"], false_call.1);
    assert_eq!(writes[0]["content"]["value"], *snapshot);
}

#[tokio::test]
async fn passes_the_stored_delta_count_since_the_newest_base_including_after_unload_and_version_bases()
 {
    let seen: Arc<Mutex<Vec<u64>>> = Arc::default();
    let sink = seen.clone();
    let v1 = define_doc(
        count_doc("checkpoint.deltas-since-base", 1, 0).checkpoint_when(move |_, _, info| {
            sink.lock().push(info.deltas_since_base);
            Ok(info.deltas_since_base >= 2)
        }),
    )
    .unwrap();
    let sink = seen.clone();
    let v2 = define_doc(
        count_doc("checkpoint.deltas-since-base", 2, 0)
            .migrate(|value, _| {
                Ok(Count {
                    count: value["count"].as_i64().unwrap(),
                })
            })
            .checkpoint_when(move |_, _, info| {
                sink.lock().push(info.deltas_since_base);
                Ok(false)
            }),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let increment = |token: DocToken<Count, SessionScope>| async move {
        commit(session, move |tx| async move {
            tx.doc(&token, ()).await?.edit(|value| value.count += 1)
        })
        .await;
    };
    touch(session, &v1).await;
    increment(v1.clone()).await;
    increment(v1.clone()).await;
    increment(v1.clone()).await;
    session.unload_documents().await;
    increment(v1.clone()).await;
    assert_eq!(*seen.lock(), [0, 1, 2, 0]);
    let commits = test.storage.commits.lock().clone();
    let kinds: Vec<String> = commits[commits.len() - 4..]
        .iter()
        .map(|writes| {
            let write = to_json(&writes[0]);
            if write["type"] == "document.change" {
                write["content"]["kind"].as_str().unwrap().to_string()
            } else {
                write["type"].as_str().unwrap().to_string()
            }
        })
        .collect();
    assert_eq!(kinds, ["delta", "delta", "base", "delta"]);

    // A required version base resets the count without calling the predicate.
    increment(v2.clone()).await;
    increment(v2.clone()).await;
    assert_eq!(*seen.lock(), [0, 1, 2, 0, 0]);
    let commits = test.storage.commits.lock().clone();
    assert_matches(
        &to_json(&commits[commits.len() - 2][0]),
        &json!({ "content": { "kind": "base", "version": 2 } }),
    );
}

#[tokio::test]
async fn skips_the_predicate_for_empty_batches() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let token = define_doc(
        DocDefinition::new("checkpoint.no-op", 1, SessionScope, || Items {
            items: vec!["a".into(), "b".into()],
        })
        .checkpoint_when(move |_, _, _| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(false)
        }),
    )
    .unwrap();
    let test = open_test_session();
    touch(&test.session, &token).await;
    let commits = test.storage.commit_count();
    let staged = token.clone();
    commit(&test.session, move |tx| async move {
        tx.doc(&staged, ()).await?.edit(|value| {
            value.items.push("x".into());
            value.items.pop();
        })
    })
    .await;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(test.storage.commit_count(), commits);
}

#[tokio::test]
async fn rolls_back_every_prepared_document_when_a_checkpoint_predicate_throws() {
    let throw_checkpoint = Arc::new(AtomicBool::new(true));
    let first_calls = Arc::new(AtomicUsize::new(0));
    let counted = first_calls.clone();
    let first = define_doc(
        count_doc("checkpoint.rollback.first", 1, 0).checkpoint_when(move |_, _, _| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(false)
        }),
    )
    .unwrap();
    let throwing = throw_checkpoint.clone();
    let second = define_doc(
        count_doc("checkpoint.rollback.second", 1, 0).checkpoint_when(move |_, _, _| {
            if throwing.load(Ordering::SeqCst) {
                return Err(Error::message("checkpoint failed"));
            }
            Ok(false)
        }),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let tokens = (first.clone(), second.clone());
    commit(session, move |tx| async move {
        tx.doc(&tokens.0, ()).await?;
        tx.doc(&tokens.1, ()).await?;
        Ok(())
    })
    .await;
    flush().await;
    let first_value = session
        .snapshot_json(&first, (), &context())
        .await
        .unwrap()
        .unwrap();
    let second_value = session
        .snapshot_json(&second, (), &context())
        .await
        .unwrap()
        .unwrap();
    let commits = test.storage.commit_count();
    let published = test.published();

    let tokens = (first.clone(), second.clone());
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.doc(&tokens.0, ()).await?.edit(|value| value.count = 1)?;
                    tx.doc(&tokens.1, ()).await?.edit(|value| value.count = 2)
                },
                &context(),
            )
            .await,
        "checkpoint failed",
    );
    flush().await;
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(test.published(), published);
    assert!(Arc::ptr_eq(
        &session
            .snapshot_json(&first, (), &context())
            .await
            .unwrap()
            .unwrap(),
        &first_value
    ));
    assert!(Arc::ptr_eq(
        &session
            .snapshot_json(&second, (), &context())
            .await
            .unwrap()
            .unwrap(),
        &second_value
    ));

    throw_checkpoint.store(false, Ordering::SeqCst);
    let tokens = (first.clone(), second.clone());
    commit(session, move |tx| async move {
        tx.doc(&tokens.0, ()).await?.edit(|value| value.count = 3)?;
        tx.doc(&tokens.1, ()).await?.edit(|value| value.count = 4)
    })
    .await;
    assert_eq!(snap(session, &first).await, Some(Count { count: 3 }));
    assert_eq!(snap(session, &second).await, Some(Count { count: 4 }));
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Values {
    values: Vec<i64>,
}

#[tokio::test]
async fn persists_repeated_false_decisions_as_deltas_and_replays_the_complete_tail() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let token = define_doc(
        DocDefinition::new("checkpoint.tail", 1, SessionScope, Values::default).checkpoint_when(
            move |_, _, _| {
                counted.fetch_add(1, Ordering::SeqCst);
                Ok(false)
            },
        ),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    touch(session, &token).await;
    for value in 1..=8 {
        let staged = token.clone();
        commit(session, move |tx| async move {
            tx.doc(&staged, ())
                .await?
                .edit(|doc| doc.values.push(value))
        })
        .await;
    }
    let kinds = change_kinds(&test.storage.commits.lock());
    assert_eq!(kinds.len(), 8);
    assert!(kinds.iter().all(|kind| kind == "delta"));
    assert_eq!(calls.load(Ordering::SeqCst), 8);
    session.unload_documents().await;
    assert_eq!(
        snap(session, &token).await,
        Some(Values {
            values: (1..=8).collect()
        })
    );
}

#[tokio::test]
async fn keeps_a_large_change_as_a_replayable_delta_when_the_predicate_is_false() {
    let initial: BTreeMap<String, i64> = (0..4_100)
        .map(|index| (format!("field{index}"), 0))
        .collect();
    let token = define_doc(
        DocDefinition::new("checkpoint.root-replacement", 1, SessionScope, move || {
            initial.clone()
        })
        .checkpoint_when(|_, _, _| Ok(false)),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    touch(session, &token).await;
    let staged = token.clone();
    commit(session, move |tx| async move {
        tx.doc(&staged, ()).await?.edit(|value| {
            for field in value.values_mut() {
                *field = 1;
            }
        })
    })
    .await;
    let write = to_json(&test.storage.last_commit()[0]);
    assert_eq!(write["type"], "document.change");
    assert_eq!(write["content"]["kind"], "delta");
    session.unload_documents().await;
    assert_eq!(snap(session, &token).await.unwrap()["field4099"], 1);
}

#[tokio::test]
async fn uses_ordinary_checkpoint_selection_before_retirement() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let token = define_doc(
        count_doc("checkpoint.retire", 1, 0).checkpoint_when(move |_, _, _| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(true)
        }),
    )
    .unwrap();
    let test = open_test_session();
    touch(&test.session, &token).await;
    let staged = token.clone();
    commit(&test.session, move |tx| async move {
        tx.doc(&staged, ()).await?.edit(|value| value.count = 1)?;
        tx.retire_doc(&staged, ()).await
    })
    .await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let writes = document_writes(&test.storage.last_commit());
    assert_eq!(writes.len(), 2);
    assert_matches(
        &writes[0],
        &json!({ "type": "document.change", "content": { "kind": "base" } }),
    );
    assert_eq!(writes[1]["type"], "document.retire");
}

// ─── Migrations ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Labeled {
    count: i64,
    labels: Vec<String>,
}

#[tokio::test]
async fn migrates_read_only_once_per_cold_load_and_writes_nothing() {
    let old = define_doc(count_doc("migration.read-only", 1, 2)).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let current = define_doc(
        DocDefinition::new("migration.read-only", 3, SessionScope, Labeled::default).migrate(
            move |value, from_version| {
                assert_eq!(from_version, 1);
                counted.fetch_add(1, Ordering::SeqCst);
                Ok(Labeled {
                    count: value["count"].as_i64().unwrap(),
                    labels: vec!["migrated".into()],
                })
            },
        ),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    touch(session, &old).await;
    session.unload_documents().await;
    let commits = test.storage.commit_count();

    let first = session
        .snapshot_json(&current, (), &context())
        .await
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(
        &session
            .snapshot_json(&current, (), &context())
            .await
            .unwrap()
            .unwrap(),
        &first
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(*first, json!({ "count": 2, "labels": ["migrated"] }));

    session.unload_documents().await;
    let second = session
        .snapshot_json(&current, (), &context())
        .await
        .unwrap()
        .unwrap();
    assert!(!Arc::ptr_eq(&second, &first));
    assert_eq!(second, first);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(test.storage.commit_count(), commits);
}

#[tokio::test]
async fn writes_the_required_base_on_the_first_successful_transaction_then_writes_deltas() {
    let old = define_doc(count_doc("migration.transition", 1, 4)).unwrap();
    let checkpoints = Arc::new(AtomicUsize::new(0));
    let counted = checkpoints.clone();
    let current = define_doc(
        count_doc("migration.transition", 3, 0)
            .migrate(|value, from_version| {
                Ok(Count {
                    count: value["count"].as_i64().unwrap() + i64::from(from_version) - 1,
                })
            })
            .checkpoint_when(move |_, _, _| {
                counted.fetch_add(1, Ordering::SeqCst);
                Ok(false)
            }),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    touch(session, &old).await;
    session.unload_documents().await;
    // An observer of the older shape.
    let watch = session
        .watch_doc(&old, (), &context())
        .await
        .unwrap()
        .unwrap();
    let frames: Arc<Mutex<Vec<(JsonValue, JsonValue)>>> = Arc::default();
    let sink = frames.clone();
    watch
        .start(move |value, ops, _| {
            sink.lock().push((
                value.as_deref().cloned().unwrap_or(JsonValue::Null),
                to_json(&ops.to_vec()),
            ));
            async { Ok(()) }.boxed()
        })
        .unwrap();
    assert_eq!(snap(session, &current).await, Some(Count { count: 4 }));
    let snapshot = session
        .snapshot_json(&current, (), &context())
        .await
        .unwrap()
        .unwrap();
    // An observer of the new shape: the migration changes nothing it sees.
    let current_watch = session
        .watch_doc(&current, (), &context())
        .await
        .unwrap()
        .unwrap();
    let current_frames = Arc::new(AtomicUsize::new(0));
    let counted = current_frames.clone();
    current_watch
        .start(move |_, _, _| {
            counted.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }.boxed()
        })
        .unwrap();

    touch(session, &current).await;
    flush().await;
    assert_matches(
        &to_json(&test.storage.last_commit()[0]),
        &json!({ "type": "document.change", "content": { "kind": "base", "version": 3, "value": { "count": 4 } } }),
    );
    assert_eq!(checkpoints.load(Ordering::SeqCst), 0);
    // The migration-only base is published, so the older-shape watch receives the new value as a root replacement.
    let documents = document_changes(&test.last_publication());
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].version, Some(3));
    assert_eq!(
        **documents[0].value.as_ref().unwrap(),
        json!({ "count": 4 })
    );
    assert!(documents[0].ops.is_empty());
    assert_eq!(
        *frames.lock(),
        [(json!({ "count": 4 }), json!([["r", { "count": 4 }]]))]
    );
    assert_eq!(current_frames.load(Ordering::SeqCst), 0);
    watch.stop().await;
    current_watch.stop().await;
    assert!(Arc::ptr_eq(
        &session
            .snapshot_json(&current, (), &context())
            .await
            .unwrap()
            .unwrap(),
        &snapshot
    ));

    set_count(session, &current, 7).await;
    assert_matches(
        &to_json(&test.storage.last_commit()[0]),
        &json!({ "type": "document.change", "content": { "kind": "delta", "version": 3 } }),
    );
    assert_eq!(checkpoints.load(Ordering::SeqCst), 1);
    session.unload_documents().await;
    assert_eq!(snap(session, &current).await, Some(Count { count: 7 }));
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Migrated {
    count: i64,
    migrated: bool,
}

#[tokio::test]
async fn rolls_migration_and_edits_back_with_the_callback_then_coalesces_later_edits_into_one_base()
{
    let old = define_doc(count_doc("migration.rollback", 1, 1)).unwrap();
    let migrations = Arc::new(AtomicUsize::new(0));
    let counted = migrations.clone();
    let current = define_doc(
        DocDefinition::new("migration.rollback", 2, SessionScope, Migrated::default).migrate(
            move |value, _| {
                counted.fetch_add(1, Ordering::SeqCst);
                Ok(Migrated {
                    count: value["count"].as_i64().unwrap(),
                    migrated: true,
                })
            },
        ),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    touch(session, &old).await;
    session.unload_documents().await;
    let commits = test.storage.commit_count();

    let staged = current.clone();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.doc(&staged, ()).await?.edit(|value| value.count = 8)?;
                    Err::<(), _>(Error::message("rollback"))
                },
                &context(),
            )
            .await,
        "rollback",
    );
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(
        snap(session, &current).await,
        Some(Migrated {
            count: 1,
            migrated: true
        })
    );
    assert_eq!(migrations.load(Ordering::SeqCst), 1);

    let staged = current.clone();
    commit(session, move |tx| async move {
        tx.doc(&staged, ()).await?.edit(|value| {
            value.count = 9;
            value.migrated = false;
        })
    })
    .await;
    let writes = document_writes(&test.storage.last_commit());
    assert_eq!(writes.len(), 1);
    assert_matches(
        &writes[0],
        &json!({ "type": "document.change", "content": { "kind": "base", "version": 2, "value": { "count": 9, "migrated": false } } }),
    );
    flush().await;
    let published = document_changes(&test.last_publication())[0].clone();
    let snapshot = session
        .snapshot_json(&current, (), &context())
        .await
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(published.value.as_ref().unwrap(), &snapshot));
    assert_eq!(writes[0]["content"]["value"], *snapshot);
    assert!(!published.ops.is_empty());
    assert_eq!(migrations.load(Ordering::SeqCst), 1);
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Float {
    invalid: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Okay {
    ok: bool,
}

#[tokio::test]
async fn strict_checks_migration_results_before_tracker_ownership_and_remains_usable_after_rejection()
 {
    let old = define_doc(count_doc("migration.invalid", 1, 1)).unwrap();
    let invalid = define_doc(
        DocDefinition::new("migration.invalid", 2, SessionScope, Float::default)
            .migrate(|_, _| Result::Ok(Float { invalid: f64::NAN })),
    )
    .unwrap();
    let other = define_doc(DocDefinition::new(
        "migration.invalid.other",
        1,
        SessionScope,
        || Okay { ok: true },
    ))
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    touch(session, &old).await;
    session.unload_documents().await;
    let commits = test.storage.commit_count();
    assert_err(
        session.snapshot(&invalid, (), &context()).await,
        "strict JSON",
    );
    let staged = invalid.clone();
    assert_err(
        session
            .commit(
                move |tx| async move { tx.doc(&staged, ()).await.map(|_| ()) },
                &context(),
            )
            .await,
        "strict JSON",
    );
    assert_eq!(test.storage.commit_count(), commits);
    touch(session, &other).await;
    assert_eq!(snap(session, &other).await, Some(Okay { ok: true }));
}

#[tokio::test]
async fn rejects_newer_stored_versions_and_older_versions_without_migration_for_snapshots_and_transactions()
 {
    let v2 = define_doc(count_doc("migration.compatibility", 2, 2)).unwrap();
    let v1 = define_doc(count_doc("migration.compatibility", 1, 1)).unwrap();
    let v3 = define_doc(count_doc("migration.compatibility", 3, 3)).unwrap();
    let test = open_test_session();
    let session = &test.session;
    touch(session, &v2).await;
    session.unload_documents().await;
    let commits = test.storage.commit_count();

    let reject = |token: DocToken<Count, SessionScope>, needle: &'static str| async move {
        assert_err(session.snapshot(&token, (), &context()).await, needle);
        assert_err(
            session
                .commit(
                    move |tx| async move { tx.doc(&token, ()).await.map(|_| ()) },
                    &context(),
                )
                .await,
            needle,
        );
    };
    reject(v1, "newer version 2 than 1").await;
    reject(v3, "requires migration from version 2").await;
    assert_eq!(test.storage.commit_count(), commits);
}

#[tokio::test]
async fn persists_a_required_migration_base_before_retirement_without_consulting_the_checkpoint_predicate()
 {
    let old = define_doc(count_doc("migration.retire", 1, 1)).unwrap();
    let current = define_doc(
        count_doc("migration.retire", 2, 0)
            .migrate(|value, _| {
                Result::Ok(Count {
                    count: value["count"].as_i64().unwrap(),
                })
            })
            .checkpoint_when(|_, _, _| panic!("must not run")),
    )
    .unwrap();
    let test = open_test_session();
    touch(&test.session, &old).await;
    test.session.unload_documents().await;
    let staged = current.clone();
    commit(&test.session, move |tx| async move {
        tx.doc(&staged, ()).await?;
        tx.retire_doc(&staged, ()).await
    })
    .await;
    let writes = document_writes(&test.storage.last_commit());
    assert_eq!(writes.len(), 2);
    assert_matches(
        &writes[0],
        &json!({ "type": "document.change", "content": { "kind": "base", "version": 2, "value": { "count": 1 } } }),
    );
    assert_eq!(writes[1]["type"], "document.retire");
}

#[tokio::test]
async fn leaves_unaccessed_older_documents_and_unavailable_definitions_untouched() {
    let first_v1 = define_doc(count_doc("migration.lazy.first", 1, 1)).unwrap();
    let second_v1 = define_doc(count_doc("migration.lazy.second", 1, 2)).unwrap();
    let second_migrations = Arc::new(AtomicUsize::new(0));
    let first_v2 = define_doc(count_doc("migration.lazy.first", 2, 0).migrate(|value, _| {
        Result::Ok(Count {
            count: value["count"].as_i64().unwrap(),
        })
    }))
    .unwrap();
    let counted = second_migrations.clone();
    let _second_v2 = define_doc(count_doc("migration.lazy.second", 2, 0).migrate(
        move |value, _| {
            counted.fetch_add(1, Ordering::SeqCst);
            Result::Ok(Count {
                count: value["count"].as_i64().unwrap(),
            })
        },
    ))
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let tokens = (first_v1.clone(), second_v1.clone());
    commit(session, move |tx| async move {
        tx.doc(&tokens.0, ()).await?;
        tx.doc(&tokens.1, ()).await?;
        Result::Ok(())
    })
    .await;
    session.unload_documents().await;
    let commits = test.storage.commit_count();
    assert_eq!(snap(session, &first_v2).await, Some(Count { count: 1 }));
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(second_migrations.load(Ordering::SeqCst), 0);
    let second_record = test
        .storage
        .find_document(
            &DocumentAddress {
                kind: "migration.lazy.second".into(),
                key: None,
                scope: DocumentScope::Session,
            },
            DocumentPoint::Current,
            &context(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        test.storage
            .document(second_record.id, DocumentPoint::Current, &context())
            .await
            .unwrap()
            .unwrap()
            .version,
        1
    );
}

// ─── Historical snapshots ────────────────────────────────────────────────────

const AS_OF: RewindableConversation = RewindableConversation {
    fork: RewindableFork::AsOf,
};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Versioned {
    count: i64,
    version: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Seeded {
    seed: String,
    count: i64,
}

#[tokio::test]
async fn migrates_current_and_historical_rewindable_values_independently_and_follows_fork_ancestry()
{
    let v1 = define_doc(DocDefinition::new(
        "history.migration",
        1,
        AS_OF,
        Count::default,
    ))
    .unwrap();
    let family = define_doc_family(DocFamilyDefinition::new(
        "history.family",
        1,
        AS_OF,
        |seed: String| Seeded { seed, count: 0 },
    ))
    .unwrap();
    let migrations: Arc<Mutex<Vec<u32>>> = Arc::default();
    let sink = migrations.clone();
    let v3 = define_doc(
        DocDefinition::new("history.migration", 3, AS_OF, || Versioned {
            count: 0,
            version: 3,
        })
        .migrate(move |value, from_version| {
            sink.lock().push(from_version);
            Result::Ok(Versioned {
                count: value["count"].as_i64().unwrap(),
                version: 3,
            })
        }),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let conversation_id = create_conversation(session).await;
    let tokens = (v1.clone(), family.clone());
    let first_entry = commit(session, move |tx| async move {
        let entry = tx
            .append_entry(conversation_id, EntryDraft::new("first"))
            .await?
            .id;
        tx.doc(&tokens.0, conversation_id)
            .await?
            .edit(|value| value.count = 1)?;
        tx.doc(&tokens.1, (conversation_id, "member".into(), "seed".into()))
            .await?
            .edit(|value| value.count = 1)?;
        Result::Ok(entry)
    })
    .await;
    let staged = v1.clone();
    let second_entry = commit(session, move |tx| async move {
        let entry = tx
            .append_entry(conversation_id, EntryDraft::new("second"))
            .await?
            .id;
        tx.doc(&staged, conversation_id)
            .await?
            .edit(|value| value.count = 2)?;
        Result::Ok(entry)
    })
    .await;
    session.unload_documents().await;
    let commits = test.storage.commit_count();
    let versioned = |count| Some(Versioned { count, version: 3 });
    assert_eq!(
        session
            .snapshot(&v3, conversation_id, &context())
            .await
            .unwrap(),
        versioned(2)
    );
    assert_eq!(*migrations.lock(), [1]);
    assert_eq!(test.storage.commit_count(), commits);

    let staged = v3.clone();
    let third_entry = commit(session, move |tx| async move {
        let entry = tx
            .append_entry(conversation_id, EntryDraft::new("third"))
            .await?
            .id;
        tx.doc(&staged, conversation_id).await?;
        Result::Ok(entry)
    })
    .await;
    assert!(
        test.storage
            .last_commit()
            .iter()
            .map(to_json)
            .any(|write| write["type"] == "document.change"
                && write["content"]["kind"] == "base"
                && write["content"]["version"] == 3)
    );

    let as_of = |conversation, at: EntryId| {
        let v3 = v3.clone();
        async move {
            session
                .snapshot_as_of(&v3, conversation, at, &context())
                .await
        }
    };
    assert_eq!(
        as_of(conversation_id, first_entry).await.unwrap(),
        versioned(1)
    );
    assert_eq!(
        as_of(conversation_id, second_entry).await.unwrap(),
        versioned(2)
    );
    assert_eq!(
        as_of(conversation_id, third_entry).await.unwrap(),
        versioned(2)
    );
    assert_eq!(*migrations.lock(), [1, 1, 1]);
    let seeded = Some(Seeded {
        seed: "seed".into(),
        count: 1,
    });
    assert_eq!(
        session
            .snapshot_as_of(
                &family,
                (conversation_id, "member".into()),
                first_entry,
                &context()
            )
            .await
            .unwrap(),
        seeded
    );

    let child_id = commit(session, move |tx| async move {
        Result::Ok(
            tx.fork_conversation(
                conversation_id,
                second_entry,
                ConversationOwnership::Ownerless,
            )
            .await?
            .id,
        )
    })
    .await;
    assert_eq!(as_of(child_id, first_entry).await.unwrap(), versioned(1));
    assert_eq!(as_of(child_id, second_entry).await.unwrap(), versioned(2));
    assert_eq!(
        session
            .snapshot_as_of(
                &family,
                (child_id, "member".into()),
                first_entry,
                &context()
            )
            .await
            .unwrap(),
        seeded
    );
    assert_err(
        as_of(child_id, third_entry).await,
        &format!("Entry {third_entry} is not visible"),
    );
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Text {
    value: String,
}

#[tokio::test]
async fn selects_the_incarnation_alive_at_the_entry_commit_across_retirement_and_recreation() {
    let token = define_doc(DocDefinition::new("history.incarnation", 1, AS_OF, || {
        Text {
            value: "initial".into(),
        }
    }))
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let conversation_id = create_conversation(session).await;
    let append = |kind: &'static str, change: u8| {
        let token = token.clone();
        commit(session, move |tx| async move {
            let entry = tx
                .append_entry(conversation_id, EntryDraft::new(kind))
                .await?
                .id;
            match change {
                1 => tx
                    .doc(&token, conversation_id)
                    .await?
                    .edit(|value| value.value = "old".into())?,
                2 => tx.retire_doc(&token, conversation_id).await?,
                3 => tx
                    .doc(&token, conversation_id)
                    .await?
                    .edit(|value| value.value = "new".into())?,
                _ => {}
            }
            Result::Ok(entry)
        })
    };
    let before_creation = append("before", 0).await;
    let created_at = append("create", 1).await;
    let retired_at = append("retire", 2).await;
    let recreated_at = append("recreate", 3).await;

    let as_of = |at| session.snapshot_as_of(&token, conversation_id, at, &context());
    assert_eq!(as_of(before_creation).await.unwrap(), None);
    assert_eq!(
        as_of(created_at).await.unwrap(),
        Some(Text {
            value: "old".into()
        })
    );
    assert_eq!(as_of(retired_at).await.unwrap(), None);
    assert_eq!(
        as_of(recreated_at).await.unwrap(),
        Some(Text {
            value: "new".into()
        })
    );
    session.close(&context()).await.unwrap();
    assert_err(as_of(recreated_at).await, "closed");
}

// ─── Tracker cache across definition versions ───────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct V1 {
    name: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct V2 {
    names: Vec<String>,
}

fn v1_doc() -> DocToken<V1, SessionScope> {
    define_doc(DocDefinition::new(
        "cache.versioned",
        1,
        SessionScope,
        || V1 {
            name: "first".into(),
        },
    ))
    .unwrap()
}

fn v2_doc() -> DocToken<V2, SessionScope> {
    define_doc(
        DocDefinition::new("cache.versioned", 2, SessionScope, V2::default).migrate(|value, _| {
            Result::Ok(V2 {
                names: vec![value["name"].as_str().unwrap().to_string()],
            })
        }),
    )
    .unwrap()
}

fn names(names: &[&str]) -> V2 {
    V2 {
        names: names.iter().map(|name| name.to_string()).collect(),
    }
}

#[tokio::test]
async fn migrates_a_document_cached_by_an_older_token_without_unloading() {
    let test = open_test_session();
    let session = &test.session;
    touch(session, &v1_doc()).await;
    assert_eq!(
        snap(session, &v1_doc()).await,
        Some(V1 {
            name: "first".into()
        })
    );

    // Reloaded extension code accesses the still-cached document with a newer token.
    assert_eq!(snap(session, &v2_doc()).await, Some(names(&["first"])));
    commit(session, |tx| async move {
        tx.doc(&v2_doc(), ())
            .await?
            .edit(|value| value.names.push("second".into()))
    })
    .await;
    let write = document_writes(&test.storage.last_commit())[0].clone();
    assert_matches(
        &write,
        &json!({ "type": "document.change", "content": { "version": 2, "kind": "base", "value": { "names": ["first", "second"] } } }),
    );
    assert_err(
        session.snapshot(&v1_doc(), (), &context()).await,
        "newer version 2",
    );
}

#[tokio::test]
async fn sends_observers_of_an_older_shape_a_root_replacement_after_a_newer_token_writes() {
    let test = open_test_session();
    let session = &test.session;
    touch(session, &v1_doc()).await;
    let state = session
        .document_state(&v1_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let watch = session
        .watch_doc(&v1_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    let frames: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let delivered = Deferred::default();
    let (sink, signal) = (frames.clone(), delivered.clone());
    watch
        .start(move |_, ops, _| {
            sink.lock().push(to_json(&ops.to_vec()));
            signal.resolve();
            async { Result::Ok(()) }.boxed()
        })
        .unwrap();
    commit(session, |tx| async move {
        tx.doc(&v2_doc(), ())
            .await?
            .edit(|value| value.names.push("second".into()))
    })
    .await;
    delivered.wait().await;
    let expected = json!({ "names": ["first", "second"] });
    assert_eq!(*state.value().unwrap(), expected);
    assert_eq!(*watch.value().unwrap(), expected);
    assert_eq!(*frames.lock(), [json!([["r", expected]])]);
    state.dispose().unwrap();
    watch.stop().await;
}

#[tokio::test]
async fn serves_an_older_token_from_storage_after_a_newer_token_migrated_only_in_memory() {
    let test = open_test_session();
    let session = &test.session;
    touch(session, &v1_doc()).await;
    session.unload_documents().await;
    assert_eq!(snap(session, &v2_doc()).await, Some(names(&["first"])));
    assert_eq!(
        snap(session, &v1_doc()).await,
        Some(V1 {
            name: "first".into()
        })
    );
    commit(session, |tx| async move {
        tx.doc(&v1_doc(), ())
            .await?
            .edit(|value| value.name = "renamed".into())
    })
    .await;
    assert_eq!(snap(session, &v2_doc()).await, Some(names(&["renamed"])));
}
