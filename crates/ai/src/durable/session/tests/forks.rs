//! Port of `test/session-forks.test.ts`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::*;
use super::tables::{WorkInput, work_task};
use crate::chord::JsonValue;
use crate::durable::documents::{define_doc, define_doc_family};
use crate::durable::errors::{Error, StorageRejected};
use crate::durable::ids::{ConversationId, EntryId};
use crate::durable::types::{
    ConversationOwnership, DocDefinition, DocFamilyDefinition, DocScope, DocumentAddress,
    DocumentPoint, DocumentScope, EntryDraft, EntryQuery, LatestConversation, LatestFork,
    RewindableConversation, RewindableFork, SessionScope, Storage, StorageWrite, TaskOptions,
    TaskScope,
};
use crate::durable::{DocFamilyToken, DocToken, SessionImpl};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct V {
    value: String,
}

fn v(value: &str) -> V {
    V {
        value: value.into(),
    }
}

fn doc<S: DocScope>(kind: &str, scope: S, initial: &'static str) -> DocToken<V, S> {
    define_doc(DocDefinition::new(kind, 1, scope, move || v(initial))).unwrap()
}

fn family<S: DocScope>(kind: &str, scope: S) -> DocFamilyToken<V, String, S> {
    define_doc_family(DocFamilyDefinition::new(kind, 1, scope, |seed: String| V {
        value: seed,
    }))
    .unwrap()
}

const AS_OF: RewindableConversation = RewindableConversation {
    fork: RewindableFork::AsOf,
};
const CURRENT: LatestConversation = LatestConversation {
    fork: LatestFork::Current,
};
const INITIAL: LatestConversation = LatestConversation {
    fork: LatestFork::Initial,
};

fn copies_of(writes: &[StorageWrite]) -> Vec<JsonValue> {
    writes
        .iter()
        .filter(|write| write_type(write) == "document.copy")
        .map(to_json)
        .collect()
}

fn count_type(writes: &[StorageWrite], kind: &str) -> usize {
    writes
        .iter()
        .filter(|write| write_type(write) == kind)
        .count()
}

async fn fork(
    session: &SessionImpl,
    parent: ConversationId,
    at: EntryId,
) -> crate::durable::Result<crate::durable::types::ConversationRecord> {
    session
        .commit(
            move |tx| async move {
                tx.fork_conversation(parent, at, ConversationOwnership::Ownerless)
                    .await
            },
            &context(),
        )
        .await
}

async fn append(
    tx: &crate::durable::Transaction,
    conversation: ConversationId,
    kind: &str,
) -> crate::durable::Result<EntryId> {
    Ok(tx
        .append_entry(conversation, EntryDraft::new(kind))
        .await?
        .id)
}

async fn set<A>(
    tx: &crate::durable::Transaction,
    token: &A,
    args: A::Args,
    value: &str,
) -> crate::durable::Result<()>
where
    A: crate::durable::DocAccess<Value = V>,
{
    let value = value.to_string();
    tx.doc(token, args).await?.edit(|doc| doc.value = value)
}

async fn snap<A: crate::durable::DocAccess<Value = V>>(
    session: &SessionImpl,
    token: &A,
    address: A::Address,
) -> Option<V> {
    session.snapshot(token, address, &context()).await.unwrap()
}

#[tokio::test]
async fn copies_as_of_and_current_singleton_and_family_bases_while_leaving_initial_documents_absent()
 {
    let as_of = doc("fork.policies.as-of", AS_OF, "as-initial");
    let current = doc("fork.policies.current", CURRENT, "current-initial");
    let initial = doc("fork.policies.initial", INITIAL, "fresh");
    let as_of_family = family("fork.policies.as-of-family", AS_OF);
    let current_family = family("fork.policies.current-family", CURRENT);
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let tokens = (
        as_of.clone(),
        current.clone(),
        initial.clone(),
        as_of_family.clone(),
        current_family.clone(),
    );
    let fork_at = commit(session, move |tx| async move {
        let (as_of, current, initial, as_of_family, current_family) = tokens;
        let at = append(&tx, parent_id, "fork-point").await?;
        set(&tx, &as_of, parent_id, "as-at-fork").await?;
        set(&tx, &current, parent_id, "current-at-fork").await?;
        set(&tx, &initial, parent_id, "parent-only").await?;
        set(
            &tx,
            &as_of_family,
            (parent_id, "a".into(), "unused".into()),
            "family-as-a",
        )
        .await?;
        set(
            &tx,
            &as_of_family,
            (parent_id, "b".into(), "unused".into()),
            "family-as-b",
        )
        .await?;
        set(
            &tx,
            &current_family,
            (parent_id, "a".into(), "unused".into()),
            "family-current-a",
        )
        .await?;
        Ok(at)
    })
    .await;
    let tokens = (
        as_of.clone(),
        current.clone(),
        as_of_family.clone(),
        current_family.clone(),
    );
    commit(session, move |tx| async move {
        let (as_of, current, as_of_family, current_family) = tokens;
        set(&tx, &as_of, parent_id, "as-after-fork").await?;
        set(&tx, &current, parent_id, "current-when-copied").await?;
        set(
            &tx,
            &as_of_family,
            (parent_id, "a".into(), "unused".into()),
            "family-as-after",
        )
        .await?;
        set(
            &tx,
            &current_family,
            (parent_id, "a".into(), "unused".into()),
            "family-current-when-copied",
        )
        .await
    })
    .await;
    flush().await;
    let document_reads = test.storage.document_reads();

    let child = fork(session, parent_id, fork_at).await.unwrap();
    flush().await;
    assert_eq!(test.storage.document_reads(), document_reads);

    assert_eq!(snap(session, &as_of, child.id).await, Some(v("as-at-fork")));
    assert_eq!(
        snap(session, &current, child.id).await,
        Some(v("current-when-copied"))
    );
    assert_eq!(snap(session, &initial, child.id).await, None);
    assert_eq!(
        snap(session, &as_of_family, (child.id, "a".into())).await,
        Some(v("family-as-a"))
    );
    assert_eq!(
        snap(session, &as_of_family, (child.id, "b".into())).await,
        Some(v("family-as-b"))
    );
    assert_eq!(
        snap(session, &current_family, (child.id, "a".into())).await,
        Some(v("family-current-when-copied"))
    );

    let copies = copies_of(&test.storage.last_commit());
    assert_eq!(copies.len(), 5);
    assert!(copies.iter().all(|write| write["record"]["scope"]
        == json!({ "kind": "conversation", "conversationId": child.id })));
    let publication = test.last_publication();
    let copied = document_copy_changes(&publication);
    assert_eq!(copied.len(), 5);
    for change in &copied {
        assert_eq!(change.record.created_at, publication.seq);
        assert_eq!(change.conversation_id, child.id);
        let write = copies
            .iter()
            .find(|write| write["record"]["id"] == json!(change.record.id))
            .unwrap();
        assert_eq!(to_json(&change.source), write["source"]);
    }

    let address = |conversation_id| DocumentAddress {
        kind: as_of.definition().kind.clone(),
        key: None,
        scope: DocumentScope::Conversation { conversation_id },
    };
    let parent_record = test
        .storage
        .find_document(&address(parent_id), DocumentPoint::Current, &context())
        .await
        .unwrap()
        .unwrap();
    let child_record = test
        .storage
        .find_document(&address(child.id), DocumentPoint::Current, &context())
        .await
        .unwrap()
        .unwrap();
    assert_ne!(child_record.id, parent_record.id);

    let token = as_of.clone();
    let child_id = child.id;
    commit(session, move |tx| async move {
        set(&tx, &token, child_id, "child-independent").await
    })
    .await;
    assert_eq!(
        snap(session, &as_of, parent_id).await,
        Some(v("as-after-fork"))
    );
    assert_eq!(
        snap(session, &as_of, child.id).await,
        Some(v("child-independent"))
    );
    let token = initial.clone();
    commit(session, move |tx| async move {
        set(&tx, &token, child_id, "child-created").await
    })
    .await;
    assert_eq!(
        snap(session, &initial, child.id).await,
        Some(v("child-created"))
    );
}

#[tokio::test]
async fn uses_final_document_state_from_the_fork_entry_commit_while_excluding_later_same_commit_entries()
 {
    let token = doc("fork.same-commit", AS_OF, "initial");
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let staged = token.clone();
    let (fork_at, excluded) = commit(session, move |tx| async move {
        let fork_at = append(&tx, parent_id, "included").await?;
        set(&tx, &staged, parent_id, "final-state-of-commit").await?;
        let excluded = append(&tx, parent_id, "excluded").await?;
        Ok((fork_at, excluded))
    })
    .await;
    let child = fork(session, parent_id, fork_at).await.unwrap();
    assert_eq!(
        snap(session, &token, child.id).await,
        Some(v("final-state-of-commit"))
    );
    let child_id = child.id;
    let visible = commit(session, move |tx| async move {
        tx.scan_entries(EntryQuery::conversation(child_id), 10, None)
            .await
    })
    .await;
    let ids: Vec<_> = visible.items.iter().map(|entry| entry.id).collect();
    assert!(ids.contains(&fork_at));
    assert!(!ids.contains(&excluded));
}

#[tokio::test]
async fn selects_the_entry_owning_ancestor_for_as_of_copies_and_the_immediate_parent_for_current_copies()
 {
    let as_of = doc("fork.ancestry.as-of", AS_OF, "initial");
    let current = doc("fork.ancestry.current", CURRENT, "initial");
    let test = open_test_session();
    let session = &test.session;
    let root_id = create_conversation(session).await;
    let tokens = (as_of.clone(), current.clone());
    let inherited = commit(session, move |tx| async move {
        let (as_of, current) = tokens;
        let inherited = append(&tx, root_id, "root").await?;
        set(&tx, &as_of, root_id, "root-at-entry").await?;
        set(&tx, &current, root_id, "root-current").await?;
        Ok(inherited)
    })
    .await;
    let parent = fork(session, root_id, inherited).await.unwrap();
    let parent_id = parent.id;
    let tokens = (as_of.clone(), current.clone());
    let parent_entry = commit(session, move |tx| async move {
        let (as_of, current) = tokens;
        let entry = append(&tx, parent_id, "parent").await?;
        set(&tx, &as_of, parent_id, "parent-at-own-entry").await?;
        set(&tx, &current, parent_id, "parent-current").await?;
        Ok(entry)
    })
    .await;

    let inherited_fork = fork(session, parent_id, inherited).await.unwrap();
    assert_eq!(
        snap(session, &as_of, inherited_fork.id).await,
        Some(v("root-at-entry"))
    );
    assert_eq!(
        snap(session, &current, inherited_fork.id).await,
        Some(v("parent-current"))
    );

    let own_entry_fork = fork(session, parent_id, parent_entry).await.unwrap();
    assert_eq!(
        snap(session, &as_of, own_entry_fork.id).await,
        Some(v("parent-at-own-entry"))
    );
    assert_eq!(
        snap(session, &current, own_entry_fork.id).await,
        Some(v("parent-current"))
    );
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Count {
    count: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Migrated {
    count: i64,
    migrated: bool,
}

#[tokio::test]
async fn copies_the_stored_value_and_version_without_consulting_migration_definitions_or_migrated_caches()
 {
    let v1 = define_doc(DocDefinition::new("fork.stored-version", 1, AS_OF, || {
        Count { count: 1 }
    }))
    .unwrap();
    let migrations = Arc::new(AtomicUsize::new(0));
    let counted = migrations.clone();
    let v3 = define_doc(
        DocDefinition::new("fork.stored-version", 3, AS_OF, Migrated::default).migrate(
            move |value, from_version| {
                assert_eq!(from_version, 1);
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
    let parent_id = create_conversation(session).await;
    let token = v1.clone();
    let fork_at = commit(session, move |tx| async move {
        let at = append(&tx, parent_id, "point").await?;
        tx.doc(&token, parent_id).await?;
        Ok(at)
    })
    .await;
    session.unload_documents().await;
    assert_eq!(
        session.snapshot(&v3, parent_id, &context()).await.unwrap(),
        Some(Migrated {
            count: 1,
            migrated: true
        })
    );
    assert_eq!(migrations.load(Ordering::SeqCst), 1);

    let child = fork(session, parent_id, fork_at).await.unwrap();
    assert_eq!(migrations.load(Ordering::SeqCst), 1);
    let child_record = test
        .storage
        .find_document(
            &DocumentAddress {
                kind: v1.definition().kind.clone(),
                key: None,
                scope: DocumentScope::Conversation {
                    conversation_id: child.id,
                },
            },
            DocumentPoint::Current,
            &context(),
        )
        .await
        .unwrap()
        .unwrap();
    let child_stored = test
        .storage
        .document(child_record.id, DocumentPoint::Current, &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(child_stored.version, 1);
    assert_eq!(JsonValue::Object(child_stored.value), json!({ "count": 1 }));
    assert_eq!(
        session.snapshot(&v3, child.id, &context()).await.unwrap(),
        Some(Migrated {
            count: 1,
            migrated: true
        })
    );
    assert_eq!(migrations.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn coalesces_a_typed_migration_and_override_into_the_copied_creation_base() {
    let v1 = define_doc(DocDefinition::new("fork.override", 1, AS_OF, || Count {
        count: 2,
    }))
    .unwrap();
    let checkpoints = Arc::new(AtomicUsize::new(0));
    let counted = checkpoints.clone();
    let v2 = define_doc(
        DocDefinition::new("fork.override", 2, AS_OF, Migrated::default)
            .migrate(|value, _| {
                Ok(Migrated {
                    count: value["count"].as_i64().unwrap(),
                    migrated: true,
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
    let parent_id = create_conversation(session).await;
    let token = v1.clone();
    let fork_at = commit(session, move |tx| async move {
        let at = append(&tx, parent_id, "point").await?;
        tx.doc(&token, parent_id).await?;
        Ok(at)
    })
    .await;

    let document_reads = test.storage.document_reads();
    let token = v2.clone();
    let child = commit(session, move |tx| async move {
        let created = tx
            .fork_conversation(parent_id, fork_at, ConversationOwnership::Ownerless)
            .await?;
        tx.doc(&token, created.id)
            .await?
            .edit(|value| value.count = 9)?;
        Ok(created)
    })
    .await;
    flush().await;
    assert_eq!(test.storage.document_reads(), document_reads + 1);
    let creates: Vec<JsonValue> = test
        .storage
        .last_commit()
        .iter()
        .filter(|write| write_type(write) == "document.create")
        .map(to_json)
        .collect();
    assert_eq!(creates.len(), 1);
    assert_matches(
        &creates[0]["content"],
        &json!({ "kind": "base", "version": 2, "value": { "count": 9, "migrated": true } }),
    );
    assert_eq!(checkpoints.load(Ordering::SeqCst), 0);
    let documents = document_changes(&test.last_publication());
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].version, Some(2));
    assert_eq!(
        session.snapshot(&v2, child.id, &context()).await.unwrap(),
        Some(Migrated {
            count: 9,
            migrated: true
        })
    );
    let parent_record = test
        .storage
        .find_document(
            &DocumentAddress {
                kind: v1.definition().kind.clone(),
                key: None,
                scope: DocumentScope::Conversation {
                    conversation_id: parent_id,
                },
            },
            DocumentPoint::Current,
            &context(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        test.storage
            .document(parent_record.id, DocumentPoint::Current, &context())
            .await
            .unwrap()
            .unwrap()
            .version,
        1
    );
}

#[tokio::test]
async fn copies_the_incarnation_alive_at_the_fork_point_across_retirement_and_recreation() {
    let token = doc("fork.incarnations", AS_OF, "initial");
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let staged = token.clone();
    let old_at = commit(session, move |tx| async move {
        let at = append(&tx, parent_id, "old").await?;
        set(&tx, &staged, parent_id, "old").await?;
        Ok(at)
    })
    .await;
    let staged = token.clone();
    let retired_at = commit(session, move |tx| async move {
        let at = append(&tx, parent_id, "retired").await?;
        tx.retire_doc(&staged, parent_id).await?;
        Ok(at)
    })
    .await;
    let staged = token.clone();
    let new_at = commit(session, move |tx| async move {
        let at = append(&tx, parent_id, "new").await?;
        set(&tx, &staged, parent_id, "new").await?;
        Ok(at)
    })
    .await;

    let old_child = fork(session, parent_id, old_at).await.unwrap();
    let empty_child = fork(session, parent_id, retired_at).await.unwrap();
    let new_child = fork(session, parent_id, new_at).await.unwrap();
    assert_eq!(snap(session, &token, old_child.id).await, Some(v("old")));
    assert_eq!(snap(session, &token, empty_child.id).await, None);
    assert_eq!(snap(session, &token, new_child.id).await, Some(v("new")));
}

#[tokio::test]
async fn rejects_invisible_fork_points_before_admission_and_remains_usable() {
    let test = open_test_session();
    let session = &test.session;
    let root_id = create_conversation(session).await;
    let (visible, hidden) = commit(session, move |tx| async move {
        let visible = append(&tx, root_id, "visible").await?;
        let hidden = append(&tx, root_id, "hidden").await?;
        Ok((visible, hidden))
    })
    .await;
    let parent = fork(session, root_id, visible).await.unwrap();
    flush().await;
    let commits = test.storage.commit_count();
    let published = test.published();
    assert_err(
        fork(session, parent.id, hidden).await,
        &format!("Entry {hidden} is not visible"),
    );
    flush().await;
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(test.published(), published);
    let independent = create_conversation(session).await;
    assert!(
        test.storage
            .conversation(independent, &context())
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn rejects_duplicate_as_of_and_current_selections_for_one_child_address_before_admission() {
    let as_of = doc("fork.duplicate-policy", AS_OF, "as-of");
    let current = doc("fork.duplicate-policy", CURRENT, "current");
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let token = as_of.clone();
    let old_at = commit(session, move |tx| async move {
        let at = append(&tx, parent_id, "old").await?;
        tx.doc(&token, parent_id).await?;
        Ok(at)
    })
    .await;
    let token = as_of.clone();
    commit(session, move |tx| async move {
        tx.retire_doc(&token, parent_id).await
    })
    .await;
    let token = current.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, parent_id).await.map(|_| ())
    })
    .await;
    let commits = test.storage.commit_count();
    assert_err(
        fork(session, parent_id, old_at).await,
        "Fork selects multiple source documents",
    );
    assert_eq!(test.storage.commit_count(), commits);
    let independent = create_conversation(session).await;
    assert!(
        test.storage
            .conversation(independent, &context())
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn rejects_current_and_as_of_source_writes_in_the_fork_transaction() {
    let current = doc("fork.same-transaction-current", CURRENT, "committed");
    let as_of = doc("fork.same-transaction-as-of", AS_OF, "committed");
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let tokens = (current.clone(), as_of.clone());
    let fork_at = commit(session, move |tx| async move {
        let (current, as_of) = tokens;
        let at = append(&tx, parent_id, "point").await?;
        tx.doc(&current, parent_id).await?;
        tx.doc(&as_of, parent_id).await?;
        Ok(at)
    })
    .await;
    let commits = test.storage.commit_count();
    let token = current.clone();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    set(&tx, &token, parent_id, "before-fork").await?;
                    tx.fork_conversation(parent_id, fork_at, ConversationOwnership::Ownerless)
                        .await?;
                    Ok(())
                },
                &context(),
            )
            .await,
        "Cannot change fork source document",
    );
    let token = current.clone();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.fork_conversation(parent_id, fork_at, ConversationOwnership::Ownerless)
                        .await?;
                    set(&tx, &token, parent_id, "after-fork").await
                },
                &context(),
            )
            .await,
        "Cannot change fork source document",
    );
    let token = as_of.clone();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    set(&tx, &token, parent_id, "as-of-write").await?;
                    tx.fork_conversation(parent_id, fork_at, ConversationOwnership::Ownerless)
                        .await?;
                    Ok(())
                },
                &context(),
            )
            .await,
        "Cannot change fork source document",
    );
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(
        snap(session, &current, parent_id).await,
        Some(v("committed"))
    );
    let child = fork(session, parent_id, fork_at).await.unwrap();
    assert_eq!(
        snap(session, &current, child.id).await,
        Some(v("committed"))
    );
}

#[tokio::test]
async fn rolls_every_copied_base_back_when_later_pre_admission_assembly_fails() {
    let copied = doc("fork.rollback.copied", AS_OF, "copied");
    let fail_checkpoint = Arc::new(AtomicBool::new(true));
    let failing = fail_checkpoint.clone();
    let failure = define_doc(
        DocDefinition::new("fork.rollback.failure", 1, SessionScope, Count::default)
            .checkpoint_when(move |_, _, _| {
                if failing.load(Ordering::SeqCst) {
                    return Err(Error::message("checkpoint failed"));
                }
                Ok(false)
            }),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let tokens = (copied.clone(), failure.clone());
    let fork_at = commit(session, move |tx| async move {
        let (copied, failure) = tokens;
        let at = append(&tx, parent_id, "point").await?;
        tx.doc(&copied, parent_id).await?;
        tx.doc(&failure, ()).await?;
        Ok(at)
    })
    .await;
    flush().await;
    let commits = test.storage.commit_count();
    let published = test.published();
    let child_id = Arc::new(Mutex::new(None));
    let slot = child_id.clone();
    let token = failure.clone();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    *slot.lock() = Some(
                        tx.fork_conversation(parent_id, fork_at, ConversationOwnership::Ownerless)
                            .await?
                            .id,
                    );
                    tx.doc(&token, ()).await?.edit(|counter| counter.count = 1)
                },
                &context(),
            )
            .await,
        "checkpoint failed",
    );
    flush().await;
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(test.published(), published);
    let child_id = child_id.lock().unwrap();
    assert_eq!(
        test.storage
            .conversation(child_id, &context())
            .await
            .unwrap(),
        None
    );
    fail_checkpoint.store(false, Ordering::SeqCst);
    let token = failure.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, ()).await?.edit(|counter| counter.count = 2)
    })
    .await;
    assert_eq!(
        session.snapshot(&failure, (), &context()).await.unwrap(),
        Some(Count { count: 2 })
    );
    assert_eq!(snap(session, &copied, parent_id).await, Some(v("copied")));
}

#[tokio::test]
async fn rolls_back_a_guaranteed_storage_rejection_without_poisoning_the_session() {
    let token = doc("fork.storage-rejected", AS_OF, "source");
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let staged = token.clone();
    let fork_at = commit(session, move |tx| async move {
        let at = append(&tx, parent_id, "point").await?;
        tx.doc(&staged, parent_id).await?;
        Ok(at)
    })
    .await;
    let rejected = Arc::new(Mutex::new(None));
    let slot = rejected.clone();
    test.storage
        .fail_next_commit(StorageRejected::new("copy rejected").into());
    assert_err(
        session
            .commit(
                move |tx| async move {
                    *slot.lock() = Some(
                        tx.fork_conversation(parent_id, fork_at, ConversationOwnership::Ownerless)
                            .await?
                            .id,
                    );
                    Ok(())
                },
                &context(),
            )
            .await,
        "copy rejected",
    );
    let rejected = rejected.lock().unwrap();
    assert_eq!(
        test.storage
            .conversation(rejected, &context())
            .await
            .unwrap(),
        None
    );
    let next = create_conversation(session).await;
    assert!(
        test.storage
            .conversation(next, &context())
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn retires_a_copied_document_and_can_recreate_the_address_in_the_fork_transaction() {
    let token = doc("fork.retire-recreate", AS_OF, "fresh");
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let staged = token.clone();
    let fork_at = commit(session, move |tx| async move {
        let at = append(&tx, parent_id, "point").await?;
        set(&tx, &staged, parent_id, "copied").await?;
        Ok(at)
    })
    .await;
    let staged = token.clone();
    let child = commit(session, move |tx| async move {
        let created = tx
            .fork_conversation(parent_id, fork_at, ConversationOwnership::Ownerless)
            .await?;
        tx.retire_doc(&staged, created.id).await?;
        set(&tx, &staged, created.id, "replacement").await?;
        Ok(created)
    })
    .await;
    let writes = test.storage.last_commit();
    assert_eq!(count_type(&writes, "document.copy"), 1);
    assert_eq!(count_type(&writes, "document.create"), 1);
    assert_eq!(count_type(&writes, "document.retire"), 1);
    assert_eq!(
        snap(session, &token, child.id).await,
        Some(v("replacement"))
    );
}

#[tokio::test]
async fn copies_only_conversation_documents_leaving_session_and_task_documents_in_their_original_scopes()
 {
    let copied = doc("fork.scope.conversation", CURRENT, "conversation");
    let session_only = doc("fork.scope.session", SessionScope, "session");
    let task_only = doc("fork.scope.task", TaskScope, "task");
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let tokens = (copied.clone(), session_only.clone(), task_only.clone());
    let (fork_at, task_id) = commit(session, move |tx| async move {
        let (copied, session_only, task_only) = tokens;
        let at = append(&tx, parent_id, "point").await?;
        let task_id = tx
            .create_task(
                &work_task(),
                WorkInput { path: "x".into() },
                TaskOptions::conversation(Some(parent_id)),
            )
            .await?
            .erase();
        tx.doc(&copied, parent_id).await?;
        tx.doc(&session_only, ()).await?;
        tx.doc(&task_only, task_id).await?;
        Ok((at, task_id))
    })
    .await;
    let child = fork(session, parent_id, fork_at).await.unwrap();
    flush().await;
    let copies = copies_of(&test.storage.last_commit());
    let kinds: Vec<_> = copies
        .iter()
        .map(|write| write["record"]["kind"].clone())
        .collect();
    assert_eq!(kinds, [json!(copied.definition().kind)]);
    let changes: Vec<_> = document_copy_changes(&test.last_publication())
        .into_iter()
        .map(|change| change.record.kind)
        .collect();
    assert_eq!(changes, [copied.definition().kind.clone()]);
    assert_eq!(
        snap(session, &copied, child.id).await,
        Some(v("conversation"))
    );
    assert_eq!(snap(session, &session_only, ()).await, Some(v("session")));
    assert_eq!(snap(session, &task_only, task_id).await, Some(v("task")));
}

#[tokio::test]
async fn copies_every_family_member_across_storage_scan_pages() {
    let token: DocFamilyToken<Count, i64, LatestConversation> = define_doc_family(
        DocFamilyDefinition::new("fork.pagination", 1, CURRENT, |seed: i64| Count {
            count: seed,
        }),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let parent_id = create_conversation(session).await;
    let fork_at = commit(session, move |tx| async move {
        append(&tx, parent_id, "point").await
    })
    .await;
    let staged = token.clone();
    commit(session, move |tx| async move {
        futures::future::try_join_all(
            (0..260).map(|index| tx.doc(&staged, (parent_id, format!("member-{index}"), index))),
        )
        .await?;
        Ok(())
    })
    .await;
    let child = fork(session, parent_id, fork_at).await.unwrap();
    assert_eq!(copies_of(&test.storage.last_commit()).len(), 260);
    assert_eq!(
        session
            .snapshot(&token, (child.id, "member-0".into()), &context())
            .await
            .unwrap(),
        Some(Count { count: 0 })
    );
    assert_eq!(
        session
            .snapshot(&token, (child.id, "member-259".into()), &context())
            .await
            .unwrap(),
        Some(Count { count: 259 })
    );
}
