//! Port of durable `src/testing/storage-conformance.ts`.
//!
//! Each TS case is a `pub async fn` taking the storage under test. Records are
//! written from JSON literals in Pi's wire shape (so every case also checks
//! that the Rust records decode Pi's JSON) and compared through their JSON
//! encoding, which is what TS `toEqual`/`toMatchObject` compare.
//!
//! Divergences: "detaches ..." cases mutate the caller's and the returned
//! values, which Rust ownership already isolates; the prototype-key case
//! checks that `__proto__`-like keys are stored and read back as ordinary
//! keys; the lossless-identity case uses `"\u{d7ff}"`/`"\u{e000}"` because
//! Rust strings cannot hold lone surrogates.

use std::sync::Arc;

use futures::future::BoxFuture;
use serde::Serialize;
use serde_json::json;

use crate::chord::{BACKGROUND_CONTEXT, Context, JsonValue};
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{
    ConversationId, DocumentId, EntryId, MAX_SAFE_INTEGER, ROOT_CONVERSATION_ID, Seq, SubmissionId,
    TaskId,
};
use crate::durable::types::{
    ConversationQuery, Cursor, DocumentAddress, DocumentPoint, DocumentQuery, DocumentScope,
    EntryQuery, Page, Storage, StorageWrite, StoredDocument, SubmissionQuery, SubmissionStatus,
    TaskQuery, TaskStatus,
};

use super::{ConformanceTest, StorageConformanceCase, StorageConformanceProvider};

fn context() -> &'static Context {
    &BACKGROUND_CONTEXT
}

fn writes(value: JsonValue) -> Vec<StorageWrite> {
    serde_json::from_value(value).unwrap_or_else(|error| panic!("invalid storage writes: {error}"))
}

async fn commit(storage: &dyn Storage, value: JsonValue) -> Result<Seq> {
    storage.commit(&writes(value), context()).await
}

async fn create_root(storage: &dyn Storage) -> u64 {
    commit(
        storage,
        json!([{ "type": "conversation", "value": { "id": ROOT_CONVERSATION_ID } }]),
    )
    .await
    .expect("create root");
    ROOT_CONVERSATION_ID.0
}

async fn mint(storage: &dyn Storage) -> u64 {
    storage.mint_id().await.expect("mint id")
}

fn pending_task(id: u64, conversation_id: u64) -> JsonValue {
    json!({
        "id": id,
        "conversationId": conversation_id,
        "kind": "test.task",
        "version": 1,
        "input": { "value": id },
        "state": { "status": "pending", "checkpoint": { "phase": "ready" } },
        "background": false,
        "abortRequested": false,
    })
}

fn entry(id: u64, conversation_id: u64, kind: &str) -> JsonValue {
    json!({ "id": id, "conversationId": conversation_id, "kind": kind })
}

/// `{ ...base, ...extra }`.
fn with(base: JsonValue, extra: JsonValue) -> JsonValue {
    let (JsonValue::Object(mut base), JsonValue::Object(extra)) = (base, extra) else {
        panic!("with() spreads objects");
    };
    base.extend(extra);
    JsonValue::Object(base)
}

fn to_json<T: Serialize>(value: &T) -> JsonValue {
    serde_json::to_value(value).expect("serializable record")
}

fn stored_json(stored: &StoredDocument) -> JsonValue {
    json!({
        "record": to_json(&stored.record),
        "version": stored.version,
        "value": stored.value,
        "deltasSinceBase": stored.deltas_since_base,
    })
}

fn ids<T>(page: &Page<T>, id: impl Fn(&T) -> u64) -> Vec<u64> {
    page.items.iter().map(id).collect()
}

fn session_scope() -> DocumentScope {
    DocumentScope::Session
}

fn conversation_scope(conversation_id: u64) -> DocumentScope {
    DocumentScope::Conversation {
        conversation_id: ConversationId(conversation_id),
    }
}

fn address(kind: &str, scope: DocumentScope, key: Option<&str>) -> DocumentAddress {
    DocumentAddress {
        kind: kind.to_string(),
        scope,
        key: key.map(str::to_string),
    }
}

fn at(seq: Seq) -> DocumentPoint {
    DocumentPoint::Seq(seq)
}

const CURRENT: DocumentPoint = DocumentPoint::Current;

/// `toMatchObject`: every expected object property matches recursively; arrays match element-wise.
fn assert_match(actual: &JsonValue, expected: &JsonValue) {
    fn matches(actual: &JsonValue, expected: &JsonValue) -> bool {
        match (actual, expected) {
            (JsonValue::Object(actual), JsonValue::Object(expected)) => {
                expected.iter().all(|(key, expected)| {
                    actual
                        .get(key)
                        .is_some_and(|actual| matches(actual, expected))
                })
            }
            (JsonValue::Array(actual), JsonValue::Array(expected)) => {
                actual.len() == expected.len()
                    && actual
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| matches(actual, expected))
            }
            _ => actual == expected,
        }
    }
    assert!(
        matches(actual, expected),
        "expected {actual} to match {expected}"
    );
}

#[track_caller]
fn assert_rejects<T: std::fmt::Debug>(result: Result<T>, message_includes: &str) -> Error {
    match result {
        Ok(value) => panic!("expected a rejection including {message_includes:?}, got {value:?}"),
        Err(error) => {
            assert!(
                error.to_string().contains(message_includes),
                "expected {:?} to include {message_includes:?}",
                error.to_string()
            );
            error
        }
    }
}

macro_rules! cases {
    ($($name:literal => $case:ident),* $(,)?) => {
        /// Creates runner-independent cases. `with_storage` must call and await its callback exactly once per case.
        pub fn create_storage_conformance(
            with_storage: StorageConformanceProvider,
        ) -> Vec<StorageConformanceCase> {
            vec![$({
                let with_storage = with_storage.clone();
                StorageConformanceCase::new($name, move || {
                    let test: ConformanceTest = Box::new(|storage| Box::pin($case(storage)) as BoxFuture<'static, ()>);
                    with_storage(test)
                })
            }),*]
        }
    };
}

cases! {
    "reserves ID 1 for the immutable root conversation" => reserves_id_1_for_the_immutable_root_conversation,
    "commits mixed table writes atomically and rolls all of them back on failure" => commits_mixed_table_writes_atomically_and_rolls_all_of_them_back_on_failure,
    "detaches retained writes and every returned record" => detaches_retained_writes_and_every_returned_record,
    "detaches prototype-like JSON keys without changing object prototypes" => detaches_prototype_like_json_keys_without_changing_object_prototypes,
    "indexes entries committed out of ID order" => indexes_entries_committed_out_of_id_order,
    "continues an entry cursor below its last item after a newer commit" => continues_an_entry_cursor_below_its_last_item_after_a_newer_commit,
    "paginates conversations by opaque cursor in ascending ID order" => paginates_conversations_by_opaque_cursor_in_ascending_id_order,
    "filters and pages conversations by durable owner edges" => filters_and_pages_conversations_by_durable_owner_edges,
    "scans deep fork history newest-first through every ancestor cap" => scans_deep_fork_history_newest_first_through_every_ancestor_cap,
    "replaces complete task records and pages filtered task scans" => replaces_complete_task_records_and_pages_filtered_task_scans,
    "stores owners and scans waiting and completing tasks by status" => stores_owners_and_scans_waiting_and_completing_tasks_by_status,
    "indexes request IDs per conversation and replaces complete submission records" => indexes_request_ids_per_conversation_and_replaces_complete_submission_records,
    "stores passive write submissions without input-only lifecycle states" => stores_passive_write_submissions_without_input_only_lifecycle_states,
    "reconstructs rewindable documents and preserves half-open incarnations" => reconstructs_rewindable_documents_and_preserves_half_open_incarnations,
    "streams long document tails across root replacement deltas" => streams_long_document_tails_across_root_replacement_deltas,
    "copies stored document bases independently and rejects ambiguous sources" => copies_stored_document_bases_independently_and_rejects_ambiguous_sources,
    "uses bases for version transitions and rejects historical reads of current-only documents" => uses_bases_for_version_transitions_and_rejects_historical_reads_of_current_only_documents,
    "indexes logical addresses and exact-scope scans independently" => indexes_logical_addresses_and_exact_scope_scans_independently,
    "keeps document lifecycle failures atomic and gives create-plus-retire an empty lifetime" => keeps_document_lifecycle_failures_atomic_and_gives_create_plus_retire_an_empty_lifetime,
    "rolls back record tables and secondary indexes when a document command fails" => rolls_back_record_tables_and_secondary_indexes_when_a_document_command_fails,
    "keeps indexed string identities lossless" => keeps_indexed_string_identities_lossless,
    "keeps one global record ID namespace and rejects exhausted ID minting" => keeps_one_global_record_id_namespace_and_rejects_exhausted_id_minting,
    "rejects every operation after close" => rejects_every_operation_after_close,
}

pub async fn reserves_id_1_for_the_immutable_root_conversation(storage: Arc<dyn Storage>) {
    let storage = &*storage;
    assert_eq!(mint(storage).await, 2);
    assert_eq!(create_root(storage).await, ROOT_CONVERSATION_ID.0);
    let root = storage
        .conversation(ROOT_CONVERSATION_ID, context())
        .await
        .unwrap();
    assert_eq!(to_json(&root), json!({ "id": 1 }));
    assert_rejects(
        commit(
            storage,
            json!([{ "type": "conversation", "value": { "id": ROOT_CONVERSATION_ID } }]),
        )
        .await,
        &format!("ID {ROOT_CONVERSATION_ID} already belongs to conversation"),
    );
}

pub async fn commits_mixed_table_writes_atomically_and_rolls_all_of_them_back_on_failure(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let entry_id = mint(storage).await;
    let task_id = mint(storage).await;
    let submission_id = mint(storage).await;
    let task = pending_task(task_id, root_id);
    let input = json!({
        "id": submission_id,
        "conversationId": root_id,
        "requestId": "request-1",
        "type": "input",
        "status": "placed",
        "entry": entry_id,
    });
    let user_entry = with(
        entry(entry_id, root_id, "user"),
        json!({ "data": { "text": "hello" } }),
    );
    let initial_seq = commit(
        storage,
        json!([
            { "type": "entry", "value": user_entry },
            { "type": "task", "value": task },
            { "type": "submission", "value": input },
        ]),
    )
    .await
    .unwrap();

    let stored = storage
        .entry(EntryId(entry_id), context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(to_json(&stored.entry), user_entry);
    assert_eq!(stored.commit_seq, initial_seq);
    let read_task = |id| async move { storage.task(TaskId::new(id), context()).await.unwrap() };
    let read_submission = |id| async move {
        storage
            .submission(SubmissionId(id), context())
            .await
            .unwrap()
    };
    assert_eq!(to_json(&read_task(task_id).await), task);
    assert_eq!(to_json(&read_submission(submission_id).await), input);

    let transient_entry_id = mint(storage).await;
    let running_task = with(
        task.clone(),
        json!({ "state": { "status": "running", "checkpoint": { "phase": "effect" } } }),
    );
    let done_input = with(
        input.clone(),
        json!({ "status": "done", "answer": transient_entry_id }),
    );
    assert_rejects(
        commit(
            storage,
            json!([
                { "type": "task", "value": running_task },
                { "type": "submission", "value": done_input },
                { "type": "entry", "value": entry(transient_entry_id, root_id, "assistant") },
                { "type": "conversation", "value": { "id": root_id } },
            ]),
        )
        .await,
        &format!("ID {root_id} already belongs to conversation"),
    );

    assert_eq!(to_json(&read_task(task_id).await), task);
    assert_eq!(to_json(&read_submission(submission_id).await), input);
    assert!(
        storage
            .entry(EntryId(transient_entry_id), context())
            .await
            .unwrap()
            .is_none()
    );
    let after_id = mint(storage).await;
    let after_rollback_seq = commit(
        storage,
        json!([{ "type": "entry", "value": entry(after_id, root_id, "after-rollback") }]),
    )
    .await
    .unwrap();
    assert!(after_rollback_seq > initial_seq);
}

pub async fn detaches_retained_writes_and_every_returned_record(storage: Arc<dyn Storage>) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let entry_id = mint(storage).await;
    let task_id = mint(storage).await;
    let submission_id = mint(storage).await;
    let mut entry_data = json!({ "nested": [1, 2] });
    let mut checkpoint = json!({ "phase": "ready", "nested": { "count": 1 } });
    let mut detail = json!({ "codes": ["initial"] });
    let mut batch = writes(json!([
        { "type": "entry", "value": with(entry(entry_id, root_id, "note"), json!({ "data": entry_data })) },
        { "type": "task", "value": with(pending_task(task_id, root_id), json!({
            "state": { "status": "pending", "checkpoint": checkpoint },
        })) },
        { "type": "submission", "value": {
            "id": submission_id,
            "conversationId": root_id,
            "type": "input",
            "status": "unanswered",
            "reason": "failed",
            "detail": detail,
        } },
    ]));
    storage.commit(&batch, context()).await.unwrap();

    // Mutate the caller's copies, including the committed batch itself.
    entry_data["nested"].as_array_mut().unwrap().push(json!(3));
    checkpoint["nested"]["count"] = json!(2);
    detail["codes"]
        .as_array_mut()
        .unwrap()
        .push(json!("mutated"));
    if let StorageWrite::Entry { value } = &mut batch[0] {
        value.data = Some(entry_data.clone());
    }

    let expected_task_state = json!({
        "status": "pending",
        "checkpoint": { "phase": "ready", "nested": { "count": 1 } },
    });
    let check = || async {
        let stored = storage
            .entry(EntryId(entry_id), context())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.entry.data, Some(json!({ "nested": [1, 2] })));
        let task = storage
            .task(TaskId::new(task_id), context())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(to_json(&task.state), expected_task_state);
        let submission = storage
            .submission(SubmissionId(submission_id), context())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(submission.detail, Some(json!({ "codes": ["initial"] })));
    };
    check().await;

    let mut read_entry = storage
        .entry(EntryId(entry_id), context())
        .await
        .unwrap()
        .unwrap()
        .entry;
    read_entry.data.as_mut().unwrap()["nested"]
        .as_array_mut()
        .unwrap()
        .push(json!(9));
    let mut read_task = storage
        .task(TaskId::new(task_id), context())
        .await
        .unwrap()
        .unwrap();
    if let crate::durable::types::TaskState::Pending { checkpoint } = &mut read_task.state {
        checkpoint["nested"]["count"] = json!(9);
    }
    let mut read_input = storage
        .submission(SubmissionId(submission_id), context())
        .await
        .unwrap()
        .unwrap();
    read_input.detail.as_mut().unwrap()["codes"]
        .as_array_mut()
        .unwrap()
        .push(json!("read mutation"));

    check().await;
}

pub async fn detaches_prototype_like_json_keys_without_changing_object_prototypes(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let entry_id = mint(storage).await;
    let mut data: JsonValue = serde_json::from_str(
        r#"{"__proto__":{"polluted":false},"constructor":{"label":"stored"},"toString":"value"}"#,
    )
    .unwrap();
    commit(
        storage,
        json!([{ "type": "entry", "value": with(entry(entry_id, root_id, "note"), json!({ "data": data })) }]),
    )
    .await
    .unwrap();

    data["__proto__"]["polluted"] = json!(true);
    data["constructor"]["label"] = json!("mutated");
    let read = || async {
        storage
            .entry(EntryId(entry_id), context())
            .await
            .unwrap()
            .unwrap()
            .entry
            .data
            .unwrap()
    };
    let mut first_read = read().await;
    let object = first_read.as_object().unwrap();
    assert_eq!(
        object.keys().collect::<Vec<_>>(),
        ["__proto__", "constructor", "toString"]
    );
    assert_eq!(object["__proto__"], json!({ "polluted": false }));
    assert_eq!(object["constructor"], json!({ "label": "stored" }));
    assert_eq!(object["toString"], json!("value"));

    first_read["__proto__"]["polluted"] = json!(true);
    let second_read = read().await;
    assert_eq!(second_read["__proto__"], json!({ "polluted": false }));
    assert_eq!(second_read["constructor"], json!({ "label": "stored" }));
    assert_eq!(second_read["toString"], json!("value"));
}

pub async fn indexes_entries_committed_out_of_id_order(storage: Arc<dyn Storage>) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    commit(
        storage,
        json!([
            { "type": "entry", "value": entry(30, root_id, "message") },
            { "type": "entry", "value": entry(10, root_id, "message") },
            { "type": "entry", "value": with(entry(20, root_id, "marker"), json!({ "head": 10 })) },
        ]),
    )
    .await
    .unwrap();

    let page = storage
        .scan_entries(
            &EntryQuery::conversation(ConversationId(root_id)),
            10,
            None,
            context(),
        )
        .await
        .unwrap();
    assert_eq!(ids(&page, |entry| entry.id.0), [30, 20, 10]);
    let marker = storage
        .find_latest_head_marker(ConversationId(root_id), None, context())
        .await
        .unwrap();
    assert_eq!(marker.map(|marker| marker.id), Some(EntryId(20)));
}

pub async fn continues_an_entry_cursor_below_its_last_item_after_a_newer_commit(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let oldest_id = mint(storage).await;
    let middle_id = mint(storage).await;
    let newest_id = mint(storage).await;
    commit(
        storage,
        json!([
            { "type": "entry", "value": entry(oldest_id, root_id, "message") },
            { "type": "entry", "value": entry(middle_id, root_id, "message") },
            { "type": "entry", "value": entry(newest_id, root_id, "message") },
        ]),
    )
    .await
    .unwrap();

    let query = EntryQuery::conversation(ConversationId(root_id));
    let first = storage
        .scan_entries(&query, 2, None, context())
        .await
        .unwrap();
    assert_eq!(ids(&first, |entry| entry.id.0), [newest_id, middle_id]);
    let appended_id = mint(storage).await;
    commit(
        storage,
        json!([{ "type": "entry", "value": entry(appended_id, root_id, "message") }]),
    )
    .await
    .unwrap();
    let second = storage
        .scan_entries(&query, 2, first.next.as_ref(), context())
        .await
        .unwrap();
    assert_eq!(ids(&second, |entry| entry.id.0), [oldest_id]);
    assert!(second.next.is_none());
}

pub async fn paginates_conversations_by_opaque_cursor_in_ascending_id_order(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let second_id = mint(storage).await;
    let third_id = mint(storage).await;
    commit(
        storage,
        json!([
            { "type": "conversation", "value": { "id": third_id } },
            { "type": "conversation", "value": { "id": second_id } },
        ]),
    )
    .await
    .unwrap();

    let query = ConversationQuery::default();
    let first = storage
        .scan_conversations(&query, 2, None, context())
        .await
        .unwrap();
    assert_eq!(ids(&first, |record| record.id.0), [root_id, second_id]);
    let next = first.next.expect("a continuation cursor");
    let round_tripped: Cursor =
        serde_json::from_str(&serde_json::to_string(&next).unwrap()).unwrap();
    let second = storage
        .scan_conversations(&query, 2, Some(&round_tripped), context())
        .await
        .unwrap();
    assert_eq!(ids(&second, |record| record.id.0), [third_id]);
    assert!(second.next.is_none());
}

pub async fn filters_and_pages_conversations_by_durable_owner_edges(storage: Arc<dyn Storage>) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let other_owner_id = mint(storage).await;
    let first_task_id = mint(storage).await;
    let second_task_id = mint(storage).await;
    let first_id = mint(storage).await;
    let second_id = mint(storage).await;
    let third_id = mint(storage).await;
    commit(
        storage,
        json!([
            { "type": "conversation", "value": { "id": other_owner_id } },
            { "type": "conversation", "value": { "id": first_id, "owner": { "conversationId": root_id, "taskId": first_task_id } } },
            { "type": "conversation", "value": { "id": second_id, "owner": { "conversationId": root_id, "taskId": second_task_id } } },
            { "type": "conversation", "value": { "id": third_id, "owner": { "conversationId": other_owner_id, "taskId": first_task_id } } },
        ]),
    )
    .await
    .unwrap();

    let by_root = ConversationQuery {
        owner_conversation_id: Some(ConversationId(root_id)),
        owner_task_id: None,
    };
    let first = storage
        .scan_conversations(&by_root, 1, None, context())
        .await
        .unwrap();
    assert_eq!(ids(&first, |record| record.id.0), [first_id]);
    assert!(first.next.is_some());
    let second = storage
        .scan_conversations(&by_root, 1, first.next.as_ref(), context())
        .await
        .unwrap();
    assert_eq!(ids(&second, |record| record.id.0), [second_id]);
    assert!(second.next.is_none());
    let by_task = ConversationQuery {
        owner_conversation_id: None,
        owner_task_id: Some(TaskId::new(first_task_id)),
    };
    assert_eq!(
        ids(
            &storage
                .scan_conversations(&by_task, 10, None, context())
                .await
                .unwrap(),
            |record| record.id.0
        ),
        [first_id, third_id]
    );
    let both = ConversationQuery {
        owner_conversation_id: Some(ConversationId(root_id)),
        owner_task_id: Some(TaskId::new(first_task_id)),
    };
    assert_eq!(
        ids(
            &storage
                .scan_conversations(&both, 10, None, context())
                .await
                .unwrap(),
            |record| record.id.0
        ),
        [first_id]
    );
}

pub async fn scans_deep_fork_history_newest_first_through_every_ancestor_cap(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let root_first = mint(storage).await;
    let root_fork_point = mint(storage).await;
    let root_excluded_same_commit = mint(storage).await;
    let root_entries_seq = commit(
        storage,
        json!([
            { "type": "entry", "value": entry(root_first, root_id, "message") },
            { "type": "entry", "value": with(entry(root_fork_point, root_id, "marker"), json!({ "head": root_first })) },
            { "type": "entry", "value": entry(root_excluded_same_commit, root_id, "message") },
        ]),
    )
    .await
    .unwrap();
    let child_id = mint(storage).await;
    commit(
        storage,
        json!([{ "type": "conversation", "value": { "id": child_id, "parent": { "conversationId": root_id, "at": root_fork_point } } }]),
    )
    .await
    .unwrap();
    let child_fork_point = mint(storage).await;
    let child_excluded = mint(storage).await;
    commit(
        storage,
        json!([
            { "type": "entry", "value": entry(child_fork_point, child_id, "note") },
            { "type": "entry", "value": entry(child_excluded, child_id, "message") },
        ]),
    )
    .await
    .unwrap();
    let root_excluded_later = mint(storage).await;
    commit(
        storage,
        json!([{ "type": "entry", "value": entry(root_excluded_later, root_id, "message") }]),
    )
    .await
    .unwrap();
    let grandchild_id = mint(storage).await;
    commit(
        storage,
        json!([{ "type": "conversation", "value": { "id": grandchild_id, "parent": { "conversationId": child_id, "at": child_fork_point } } }]),
    )
    .await
    .unwrap();
    let grandchild_head = mint(storage).await;
    let grandchild_tail = mint(storage).await;
    let grandchild_entries_seq = commit(
        storage,
        json!([
            { "type": "entry", "value": with(entry(grandchild_head, grandchild_id, "marker"), json!({ "head": grandchild_head })) },
            { "type": "entry", "value": entry(grandchild_tail, grandchild_id, "message") },
        ]),
    )
    .await
    .unwrap();
    let child_excluded_later = mint(storage).await;
    commit(
        storage,
        json!([{ "type": "entry", "value": entry(child_excluded_later, child_id, "message") }]),
    )
    .await
    .unwrap();

    let grandchild = ConversationId(grandchild_id);
    let query = EntryQuery::conversation(grandchild);
    let first = storage
        .scan_entries(&query, 2, None, context())
        .await
        .unwrap();
    assert_eq!(
        ids(&first, |entry| entry.id.0),
        [grandchild_tail, grandchild_head]
    );
    let second = storage
        .scan_entries(&query, 2, first.next.as_ref(), context())
        .await
        .unwrap();
    assert_eq!(
        ids(&second, |entry| entry.id.0),
        [child_fork_point, root_fork_point]
    );
    let third = storage
        .scan_entries(&query, 2, second.next.as_ref(), context())
        .await
        .unwrap();
    assert_eq!(ids(&third, |entry| entry.id.0), [root_first]);
    assert!(third.next.is_none());

    let current_marker = storage
        .find_latest_head_marker(grandchild, None, context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(current_marker.id, EntryId(grandchild_head));
    assert_eq!(current_marker.head, Some(EntryId(grandchild_head)));
    let historical_marker = storage
        .find_latest_head_marker(grandchild, Some(EntryId(child_fork_point)), context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(historical_marker.id, EntryId(root_fork_point));
    assert_eq!(historical_marker.head, Some(EntryId(root_first)));
    assert!(
        storage
            .find_latest_head_marker(grandchild, Some(EntryId(root_first)), context())
            .await
            .unwrap()
            .is_none()
    );

    let active = EntryQuery {
        conversation_id: grandchild,
        min_entry_id: current_marker.head,
        max_entry_id: None,
    };
    let active_first = storage
        .scan_entries(&active, 1, None, context())
        .await
        .unwrap();
    assert_eq!(ids(&active_first, |entry| entry.id.0), [grandchild_tail]);
    assert!(active_first.next.is_some());
    let active_second = storage
        .scan_entries(&active, 1, active_first.next.as_ref(), context())
        .await
        .unwrap();
    assert_eq!(ids(&active_second, |entry| entry.id.0), [grandchild_head]);
    assert!(active_second.next.is_none());

    let historical = EntryQuery {
        conversation_id: grandchild,
        min_entry_id: historical_marker.head,
        max_entry_id: Some(EntryId(child_fork_point)),
    };
    assert_eq!(
        ids(
            &storage
                .scan_entries(&historical, 10, None, context())
                .await
                .unwrap(),
            |entry| entry.id.0
        ),
        [child_fork_point, root_fork_point, root_first]
    );

    let global = |id| async move { storage.entry(EntryId(id), context()).await.unwrap() };
    let root_first_stored = global(root_first).await.unwrap();
    assert_eq!(
        to_json(&root_first_stored.entry),
        entry(root_first, root_id, "message")
    );
    assert_eq!(root_first_stored.commit_seq, root_entries_seq);
    assert_eq!(
        global(root_fork_point).await.unwrap().commit_seq,
        root_entries_seq
    );
    assert_eq!(
        global(grandchild_head).await.unwrap().commit_seq,
        grandchild_entries_seq
    );
    assert_eq!(
        global(grandchild_tail).await.unwrap().commit_seq,
        grandchild_entries_seq
    );
    assert!(global(999_999).await.is_none());

    let visible = |conversation_id, id| async move {
        storage
            .entry_in(ConversationId(conversation_id), EntryId(id), context())
            .await
    };
    let through = visible(grandchild_id, root_first).await.unwrap().unwrap();
    assert_eq!(
        to_json(&through.entry),
        entry(root_first, root_id, "message")
    );
    assert_eq!(through.commit_seq, root_entries_seq);
    assert_eq!(
        visible(grandchild_id, child_fork_point)
            .await
            .unwrap()
            .unwrap()
            .entry
            .conversation_id,
        ConversationId(child_id)
    );
    assert_eq!(
        visible(grandchild_id, grandchild_tail)
            .await
            .unwrap()
            .unwrap()
            .commit_seq,
        grandchild_entries_seq
    );
    for (conversation_id, id) in [
        (grandchild_id, root_excluded_same_commit),
        (grandchild_id, root_excluded_later),
        (grandchild_id, child_excluded),
        (grandchild_id, child_excluded_later),
        (root_id, grandchild_head),
        (grandchild_id, 999_999),
    ] {
        assert!(
            visible(conversation_id, id).await.unwrap().is_none(),
            "entry {id} must not be visible through {conversation_id}"
        );
    }
    assert_rejects(visible(999_999, root_first).await, "Unknown conversation");
    assert_rejects(
        storage
            .scan_entries(
                &EntryQuery::conversation(ConversationId(999_999)),
                10,
                None,
                context(),
            )
            .await,
        "Unknown conversation",
    );
}

pub async fn replaces_complete_task_records_and_pages_filtered_task_scans(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let first_id = mint(storage).await;
    let second_id = mint(storage).await;
    let third_id = mint(storage).await;
    let first = with(
        pending_task(first_id, root_id),
        json!({ "memos": { "winner": "first" } }),
    );
    let second = with(
        pending_task(second_id, root_id),
        json!({ "background": true }),
    );
    let third = with(
        pending_task(third_id, root_id),
        json!({ "abortRequested": true }),
    );
    commit(
        storage,
        json!([
            { "type": "task", "value": first },
            { "type": "task", "value": second },
            { "type": "task", "value": third },
        ]),
    )
    .await
    .unwrap();

    let read =
        |id| async move { to_json(&storage.task(TaskId::new(id), context()).await.unwrap()) };
    let running = with(
        first.clone(),
        json!({
            "state": { "status": "running", "checkpoint": { "phase": "effect", "attempt": 1 } },
            "abortRequested": true,
        }),
    );
    commit(storage, json!([{ "type": "task", "value": running }]))
        .await
        .unwrap();
    assert_eq!(read(first_id).await, running);
    let terminal = json!({
        "id": first_id,
        "conversationId": root_id,
        "kind": first["kind"],
        "version": first["version"],
        "input": first["input"],
        "state": { "status": "terminal", "outcome": { "status": "completed", "result": { "entryId": 99 } } },
        "background": false,
        "abortRequested": true,
    });
    commit(storage, json!([{ "type": "task", "value": terminal }]))
        .await
        .unwrap();
    assert_eq!(read(first_id).await, terminal);

    let pending = TaskQuery {
        status: Some(TaskStatus::Pending),
        ..TaskQuery::default()
    };
    let pending_page = storage
        .scan_tasks(&pending, 1, None, context())
        .await
        .unwrap();
    assert_eq!(ids(&pending_page, |task| task.id.0), [second_id]);
    assert!(pending_page.next.is_some());
    assert_eq!(
        ids(
            &storage
                .scan_tasks(&pending, 1, pending_page.next.as_ref(), context())
                .await
                .unwrap(),
            |task| task.id.0
        ),
        [third_id]
    );
    let terminal_aborted = TaskQuery {
        status: Some(TaskStatus::Terminal),
        abort_requested: Some(true),
        ..TaskQuery::default()
    };
    assert_eq!(
        to_json(
            &storage
                .scan_tasks(&terminal_aborted, 10, None, context())
                .await
                .unwrap()
                .items
        ),
        json!([terminal])
    );
    let background = TaskQuery {
        background: Some(true),
        ..TaskQuery::default()
    };
    assert_eq!(
        ids(
            &storage
                .scan_tasks(&background, 10, None, context())
                .await
                .unwrap(),
            |task| task.id.0
        ),
        [second_id]
    );
}

pub async fn stores_owners_and_scans_waiting_and_completing_tasks_by_status(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let owner_id = mint(storage).await;
    let waiting_id = mint(storage).await;
    let completing_id = mint(storage).await;
    let owner = pending_task(owner_id, root_id);
    let waiting = with(
        pending_task(waiting_id, root_id),
        json!({
            "owner": owner_id,
            "state": { "status": "waiting", "checkpoint": { "phase": "next" }, "on": [owner_id], "policy": "allSettled" },
            "memos": { "kept": true },
        }),
    );
    let completing = with(
        pending_task(completing_id, root_id),
        json!({
            "owner": owner_id,
            "state": { "status": "completing", "outcome": { "status": "failed", "error": { "message": "held" } } },
        }),
    );
    commit(
        storage,
        json!([
            { "type": "task", "value": owner },
            { "type": "task", "value": waiting },
            { "type": "task", "value": completing },
        ]),
    )
    .await
    .unwrap();
    let read =
        |id| async move { to_json(&storage.task(TaskId::new(id), context()).await.unwrap()) };
    assert_eq!(read(waiting_id).await, waiting);
    assert_eq!(read(completing_id).await, completing);
    let scan = |status| async move {
        let query = TaskQuery {
            status: Some(status),
            ..TaskQuery::default()
        };
        storage
            .scan_tasks(&query, 10, None, context())
            .await
            .unwrap()
    };
    assert_eq!(
        to_json(&scan(TaskStatus::Waiting).await.items),
        json!([waiting])
    );
    assert_eq!(
        to_json(&scan(TaskStatus::Completing).await.items),
        json!([completing])
    );
    assert_eq!(
        ids(&scan(TaskStatus::Pending).await, |task| task.id.0),
        [owner_id]
    );
    let terminal = with(
        completing.clone(),
        json!({ "state": { "status": "terminal", "outcome": completing["state"]["outcome"] } }),
    );
    commit(storage, json!([{ "type": "task", "value": terminal }]))
        .await
        .unwrap();
    assert_eq!(
        to_json(&scan(TaskStatus::Completing).await.items),
        json!([])
    );
    assert_eq!(
        to_json(&scan(TaskStatus::Terminal).await.items),
        json!([terminal])
    );
}

pub async fn indexes_request_ids_per_conversation_and_replaces_complete_submission_records(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let second_conversation_id = mint(storage).await;
    commit(
        storage,
        json!([{ "type": "conversation", "value": { "id": second_conversation_id } }]),
    )
    .await
    .unwrap();
    let first_id = mint(storage).await;
    let second_id = mint(storage).await;
    let other_conversation_id = mint(storage).await;
    let submission = |id: u64, conversation_id: u64, request_id: &str| {
        json!({
            "id": id,
            "conversationId": conversation_id,
            "requestId": request_id,
            "type": "input",
            "status": "queued",
        })
    };
    let first = submission(first_id, root_id, "same");
    let second = submission(second_id, root_id, "other");
    let other_conversation = submission(other_conversation_id, second_conversation_id, "same");
    commit(
        storage,
        json!([
            { "type": "submission", "value": first },
            { "type": "submission", "value": second },
            { "type": "submission", "value": other_conversation },
        ]),
    )
    .await
    .unwrap();
    let by_request = |conversation_id, request_id: &'static str| async move {
        to_json(
            &storage
                .submission_by_request(ConversationId(conversation_id), request_id, context())
                .await
                .unwrap(),
        )
    };
    assert_eq!(by_request(root_id, "same").await, first);
    assert_eq!(
        by_request(second_conversation_id, "same").await,
        other_conversation
    );

    let placed_second = with(
        second.clone(),
        json!({ "status": "placed", "entry": mint(storage).await }),
    );
    commit(
        storage,
        json!([{ "type": "submission", "value": placed_second }]),
    )
    .await
    .unwrap();
    assert_eq!(
        to_json(
            &storage
                .submission(SubmissionId(second_id), context())
                .await
                .unwrap()
        ),
        placed_second
    );
    assert_eq!(by_request(root_id, "other").await, placed_second);

    let scan_ids = |query: SubmissionQuery| async move {
        let mut found = Vec::new();
        let mut cursor: Option<Cursor> = None;
        loop {
            let page = storage
                .scan_submissions(&query, 1, cursor.as_ref(), context())
                .await
                .unwrap();
            found.extend(page.items.iter().map(|submission| submission.id.0));
            cursor = page.next;
            if cursor.is_none() {
                return found;
            }
        }
    };
    assert_eq!(
        scan_ids(SubmissionQuery::default()).await,
        [first_id, second_id, other_conversation_id]
    );
    assert_eq!(
        scan_ids(SubmissionQuery {
            conversation_id: Some(ConversationId(root_id)),
            status: None,
        })
        .await,
        [first_id, second_id]
    );
    // A status change moves the record between status scans.
    assert_eq!(
        scan_ids(SubmissionQuery {
            conversation_id: None,
            status: Some(SubmissionStatus::Queued),
        })
        .await,
        [first_id, other_conversation_id]
    );
    let placed = SubmissionQuery {
        conversation_id: None,
        status: Some(SubmissionStatus::Placed),
    };
    assert_eq!(
        to_json(
            &storage
                .scan_submissions(&placed, 10, None, context())
                .await
                .unwrap()
                .items
        ),
        json!([placed_second])
    );
}

pub async fn stores_passive_write_submissions_without_input_only_lifecycle_states(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let done_id = mint(storage).await;
    let failed_id = mint(storage).await;
    let queued = |id: u64, request_id: &str| {
        json!({
            "id": id,
            "conversationId": root_id,
            "requestId": request_id,
            "type": "write",
            "status": "queued",
        })
    };
    let queued_done = queued(done_id, "passive-done");
    let queued_failed = queued(failed_id, "passive-failed");
    commit(
        storage,
        json!([
            { "type": "submission", "value": queued_done },
            { "type": "submission", "value": queued_failed },
        ]),
    )
    .await
    .unwrap();

    let done = with(
        queued_done,
        json!({ "status": "done", "entry": mint(storage).await }),
    );
    let unanswered = with(
        queued_failed,
        json!({ "status": "unanswered", "reason": "closed", "detail": { "retryable": false } }),
    );
    commit(
        storage,
        json!([
            { "type": "submission", "value": done },
            { "type": "submission", "value": unanswered },
        ]),
    )
    .await
    .unwrap();
    let by_id = |id| async move {
        to_json(
            &storage
                .submission(SubmissionId(id), context())
                .await
                .unwrap(),
        )
    };
    let by_request = |request_id: &'static str| async move {
        to_json(
            &storage
                .submission_by_request(ConversationId(root_id), request_id, context())
                .await
                .unwrap(),
        )
    };
    assert_eq!(by_id(done_id).await, done);
    assert_eq!(by_request("passive-done").await, done);
    assert_eq!(by_id(failed_id).await, unanswered);
    assert_eq!(by_request("passive-failed").await, unanswered);
}

async fn read_document(
    storage: &dyn Storage,
    id: u64,
    point: DocumentPoint,
) -> Result<Option<StoredDocument>> {
    storage.document(DocumentId(id), point, context()).await
}

async fn document_json(storage: &dyn Storage, id: u64, point: DocumentPoint) -> JsonValue {
    read_document(storage, id, point)
        .await
        .unwrap()
        .as_ref()
        .map_or(JsonValue::Null, stored_json)
}

async fn document_value(storage: &dyn Storage, id: u64, point: DocumentPoint) -> JsonValue {
    read_document(storage, id, point)
        .await
        .unwrap()
        .map_or(JsonValue::Null, |stored| JsonValue::Object(stored.value))
}

pub async fn reconstructs_rewindable_documents_and_preserves_half_open_incarnations(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let first_id = mint(storage).await;
    let first_record = json!({
        "id": first_id,
        "kind": "conversation.notes",
        "scope": { "kind": "conversation", "conversationId": root_id },
        "history": "rewindable",
        "fork": "asOf",
    });
    let mut initial = json!({ "items": ["a"], "nested": { "count": 1 } });
    let created_at = commit(
        storage,
        json!([{ "type": "document.create", "record": first_record, "content": { "kind": "base", "version": 1, "value": initial } }]),
    )
    .await
    .unwrap();
    let mut appended = json!(["b"]);
    let changed_at = commit(
        storage,
        json!([{ "type": "document.change", "id": first_id, "content": { "kind": "delta", "version": 1, "ops": [
            ["p", ["items"], 1, 0, appended],
            ["s", ["nested", "count"], 2],
        ] } }]),
    )
    .await
    .unwrap();

    initial["items"]
        .as_array_mut()
        .unwrap()
        .push(json!("caller mutation"));
    appended
        .as_array_mut()
        .unwrap()
        .push(json!("caller mutation"));
    assert_match(
        &document_json(storage, first_id, at(created_at)).await,
        &json!({ "version": 1, "value": { "items": ["a"], "nested": { "count": 1 } }, "deltasSinceBase": 0 }),
    );
    let mut changed = read_document(storage, first_id, at(changed_at))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        JsonValue::Object(changed.value.clone()),
        json!({ "items": ["a", "b"], "nested": { "count": 2 } })
    );
    assert_eq!(changed.deltas_since_base, 1);
    changed.value["items"]
        .as_array_mut()
        .unwrap()
        .push(json!("read mutation"));
    assert_eq!(
        document_value(storage, first_id, CURRENT).await,
        json!({ "items": ["a", "b"], "nested": { "count": 2 } })
    );

    let checkpoint_at = commit(
        storage,
        json!([{ "type": "document.change", "id": first_id, "content": { "kind": "base", "version": 2, "value": { "items": ["checkpoint"], "nested": { "count": 3 } } } }]),
    )
    .await
    .unwrap();
    let replaced_at = commit(
        storage,
        json!([{ "type": "document.change", "id": first_id, "content": { "kind": "delta", "version": 2, "ops": [
            ["r", { "items": ["replacement"], "nested": { "count": 4 } }],
        ] } }]),
    )
    .await
    .unwrap();
    assert_match(
        &document_json(storage, first_id, at(changed_at)).await,
        &json!({ "version": 1, "value": { "items": ["a", "b"], "nested": { "count": 2 } } }),
    );
    assert_match(
        &document_json(storage, first_id, at(checkpoint_at)).await,
        &json!({ "version": 2, "value": { "items": ["checkpoint"], "nested": { "count": 3 } }, "deltasSinceBase": 0 }),
    );
    assert_match(
        &document_json(storage, first_id, at(replaced_at)).await,
        &json!({ "value": { "items": ["replacement"], "nested": { "count": 4 } }, "deltasSinceBase": 1 }),
    );
    assert_eq!(
        read_document(storage, first_id, CURRENT)
            .await
            .unwrap()
            .unwrap()
            .deltas_since_base,
        1
    );

    let second_id = mint(storage).await;
    let retired_at = commit(
        storage,
        json!([
            { "type": "document.create", "record": with(first_record.clone(), json!({ "id": second_id })), "content": { "kind": "base", "version": 1, "value": { "items": ["new"] } } },
            { "type": "document.retire", "id": first_id },
            { "type": "document.change", "id": first_id, "content": { "kind": "delta", "version": 2, "ops": [["s", ["retiring"], true]] } },
        ]),
    )
    .await
    .unwrap();
    let notes = address("conversation.notes", conversation_scope(root_id), None);
    assert_eq!(
        storage
            .find_document(&notes, at(changed_at), context())
            .await
            .unwrap()
            .map(|record| record.id),
        Some(DocumentId(first_id))
    );
    assert_match(
        &to_json(
            &storage
                .find_document(&notes, at(retired_at), context())
                .await
                .unwrap(),
        ),
        &json!({ "id": second_id, "createdAt": retired_at }),
    );
    let scan = |point| {
        let notes = DocumentQuery {
            scope: conversation_scope(root_id),
            at: point,
            kind: None,
        };
        async move {
            ids(
                &storage
                    .scan_documents(&notes, 10, None, context())
                    .await
                    .unwrap(),
                |record| record.id.0,
            )
        }
    };
    assert_eq!(scan(at(changed_at)).await, [first_id]);
    assert_eq!(scan(at(retired_at)).await, [second_id]);
    assert!(
        read_document(storage, first_id, at(retired_at))
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        document_value(storage, second_id, CURRENT).await,
        json!({ "items": ["new"] })
    );
}

pub async fn streams_long_document_tails_across_root_replacement_deltas(storage: Arc<dyn Storage>) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let id = mint(storage).await;
    let record = json!({
        "id": id,
        "kind": "conversation.long-tail",
        "scope": { "kind": "conversation", "conversationId": root_id },
        "history": "rewindable",
        "fork": "asOf",
    });
    let initial = json!({
        "revision": 0,
        "rows": (0..512).map(|value| json!({ "value": value, "stable": format!("row-{value}") })).collect::<Vec<_>>(),
    });
    let created_at = commit(
        storage,
        json!([{ "type": "document.create", "record": record, "content": { "kind": "base", "version": 1, "value": initial } }]),
    )
    .await
    .unwrap();
    let change = |revision: i64, index: usize| {
        json!([{ "type": "document.change", "id": id, "content": { "kind": "delta", "version": 1, "ops": [
            ["s", ["rows", index, "value"], -revision],
            ["s", ["revision"], revision],
        ] } }])
    };
    let mut before_replacement = initial.clone();
    let mut before_replacement_at = created_at;
    for revision in 1..=24_i64 {
        let index = (revision as usize * 17) % 512;
        before_replacement["rows"][index]["value"] = json!(-revision);
        before_replacement["revision"] = json!(revision);
        before_replacement_at = commit(storage, change(revision, index)).await.unwrap();
    }

    let mut replacement = json!({
        "revision": 100,
        "rows": (0..512).map(|value| json!({ "value": 10_000 + value, "stable": format!("new-{value}") })).collect::<Vec<_>>(),
    });
    let replacement_snapshot = replacement.clone();
    let replacement_at = commit(
        storage,
        json!([{ "type": "document.change", "id": id, "content": { "kind": "delta", "version": 1, "ops": [["r", replacement]] } }]),
    )
    .await
    .unwrap();
    replacement["rows"][0]["value"] = json!(-999);

    let mut current = replacement_snapshot.clone();
    for revision in 101..=124_i64 {
        let index = (revision as usize * 19) % 512;
        current["rows"][index]["value"] = json!(-revision);
        current["revision"] = json!(revision);
        commit(storage, change(revision, index)).await.unwrap();
    }

    assert_eq!(document_value(storage, id, at(created_at)).await, initial);
    assert_eq!(
        document_value(storage, id, at(before_replacement_at)).await,
        before_replacement
    );
    assert_eq!(
        document_value(storage, id, at(replacement_at)).await,
        replacement_snapshot
    );
    let mut read = read_document(storage, id, CURRENT).await.unwrap().unwrap();
    assert_eq!(JsonValue::Object(read.value.clone()), current);
    read.value["rows"][0]["value"] = json!(-1_000);
    assert_eq!(document_value(storage, id, CURRENT).await, current);
}

pub async fn copies_stored_document_bases_independently_and_rejects_ambiguous_sources(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let child_id = mint(storage).await;
    let second_child_id = mint(storage).await;
    commit(
        storage,
        json!([
            { "type": "conversation", "value": { "id": child_id } },
            { "type": "conversation", "value": { "id": second_child_id } },
        ]),
    )
    .await
    .unwrap();
    let source_id = mint(storage).await;
    let source_record = json!({
        "id": source_id,
        "kind": "copy.source",
        "scope": { "kind": "conversation", "conversationId": root_id },
        "history": "rewindable",
        "fork": "asOf",
    });
    let created_at = commit(
        storage,
        json!([{ "type": "document.create", "record": source_record, "content": { "kind": "base", "version": 2, "value": { "count": 1, "rows": [{ "value": "base" }] } } }]),
    )
    .await
    .unwrap();
    commit(
        storage,
        json!([{ "type": "document.change", "id": source_id, "content": { "kind": "delta", "version": 2, "ops": [
            ["s", ["count"], 2],
            ["p", ["rows"], 1, 0, [{ "value": "current" }]],
        ] } }]),
    )
    .await
    .unwrap();
    let historical_copy_id = mint(storage).await;
    let current_copy_id = mint(storage).await;
    let retired_copy_id = mint(storage).await;
    let child_record = |id: u64, conversation_id: u64| {
        json!({
            "id": id,
            "kind": "copy.source",
            "scope": { "kind": "conversation", "conversationId": conversation_id },
            "history": "rewindable",
            "fork": "asOf",
        })
    };
    commit(
        storage,
        json!([
            { "type": "document.copy", "record": child_record(historical_copy_id, child_id), "source": { "id": source_id, "at": created_at } },
            { "type": "document.copy", "record": child_record(current_copy_id, second_child_id), "source": { "id": source_id, "at": "current" } },
            { "type": "document.copy", "record": child_record(retired_copy_id, root_id), "source": { "id": source_id, "at": "current" } },
            { "type": "document.retire", "id": retired_copy_id },
        ]),
    )
    .await
    .unwrap();
    assert_match(
        &document_json(storage, historical_copy_id, CURRENT).await,
        &json!({ "version": 2, "value": { "count": 1, "rows": [{ "value": "base" }] } }),
    );
    let current_value =
        json!({ "count": 2, "rows": [{ "value": "base" }, { "value": "current" }] });
    assert_match(
        &document_json(storage, current_copy_id, CURRENT).await,
        &json!({ "version": 2, "value": current_value }),
    );
    assert!(
        read_document(storage, retired_copy_id, CURRENT)
            .await
            .unwrap()
            .is_none()
    );

    commit(
        storage,
        json!([
            { "type": "document.change", "id": source_id, "content": { "kind": "base", "version": 2, "value": { "count": 99, "rows": [] } } },
            { "type": "document.retire", "id": source_id },
        ]),
    )
    .await
    .unwrap();
    assert_eq!(
        document_value(storage, current_copy_id, CURRENT).await,
        current_value
    );

    let latest_source_id = mint(storage).await;
    let latest_copy_id = mint(storage).await;
    let latest_source = json!({
        "id": latest_source_id,
        "kind": "copy.latest",
        "scope": { "kind": "conversation", "conversationId": root_id },
        "history": "latest",
        "fork": "current",
    });
    commit(
        storage,
        json!([{ "type": "document.create", "record": latest_source, "content": { "kind": "base", "version": 4, "value": { "retained": "copy" } } }]),
    )
    .await
    .unwrap();
    commit(
        storage,
        json!([{ "type": "document.copy", "record": with(latest_source.clone(), json!({
            "id": latest_copy_id,
            "scope": { "kind": "conversation", "conversationId": child_id },
        })), "source": { "id": latest_source_id, "at": "current" } }]),
    )
    .await
    .unwrap();
    commit(
        storage,
        json!([
            { "type": "document.change", "id": latest_source_id, "content": { "kind": "base", "version": 4, "value": { "retained": "source-only" } } },
            { "type": "document.retire", "id": latest_source_id },
        ]),
    )
    .await
    .unwrap();
    assert_match(
        &document_json(storage, latest_copy_id, CURRENT).await,
        &json!({ "version": 4, "value": { "retained": "copy" } }),
    );

    let conflict_id = mint(storage).await;
    let conflict = commit(
        storage,
        json!([
            { "type": "document.copy", "record": child_record(conflict_id, child_id), "source": { "id": current_copy_id, "at": "current" } },
            { "type": "document.retire", "id": current_copy_id },
        ]),
    )
    .await;
    assert_eq!(
        conflict.expect_err("conflicting copy").name(),
        "StorageRejected"
    );
    assert!(
        read_document(storage, conflict_id, CURRENT)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        document_value(storage, current_copy_id, CURRENT).await,
        current_value
    );

    let mismatch_id = mint(storage).await;
    let mismatch = commit(
        storage,
        json!([{ "type": "document.copy", "record": with(child_record(mismatch_id, child_id), json!({ "kind": "copy.mismatch" })), "source": { "id": current_copy_id, "at": "current" } }]),
    )
    .await;
    assert_eq!(
        mismatch.expect_err("mismatched copy").name(),
        "StorageRejected"
    );
    assert!(
        read_document(storage, mismatch_id, CURRENT)
            .await
            .unwrap()
            .is_none()
    );
}

pub async fn uses_bases_for_version_transitions_and_rejects_historical_reads_of_current_only_documents(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    create_root(storage).await;
    let id = mint(storage).await;
    let record = json!({ "id": id, "kind": "session.settings", "scope": { "kind": "session" } });
    commit(
        storage,
        json!([{ "type": "document.create", "record": record, "content": { "kind": "base", "version": 1, "value": { "count": 1 } } }]),
    )
    .await
    .unwrap();
    commit(
        storage,
        json!([{ "type": "document.change", "id": id, "content": { "kind": "delta", "version": 1, "ops": [["s", ["count"], 2]] } }]),
    )
    .await
    .unwrap();
    let migrated_at = commit(
        storage,
        json!([{ "type": "document.change", "id": id, "content": { "kind": "base", "version": 2, "value": { "count": 3 } } }]),
    )
    .await
    .unwrap();
    assert_match(
        &document_json(storage, id, CURRENT).await,
        &json!({ "version": 2, "value": { "count": 3 } }),
    );
    assert_rejects(
        read_document(storage, id, at(migrated_at)).await,
        "does not retain historical content",
    );

    assert_rejects(
        commit(
            storage,
            json!([{ "type": "document.change", "id": id, "content": { "kind": "delta", "version": 1, "ops": [["s", ["count"], 4]] } }]),
        )
        .await,
        "version transition requires a base",
    );
    assert_eq!(
        document_value(storage, id, CURRENT).await,
        json!({ "count": 3 })
    );
    commit(storage, json!([{ "type": "document.retire", "id": id }]))
        .await
        .unwrap();
    assert!(read_document(storage, id, CURRENT).await.unwrap().is_none());
}

pub async fn indexes_logical_addresses_and_exact_scope_scans_independently(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let first_id = mint(storage).await;
    let second_id = mint(storage).await;
    let conversation_id = mint(storage).await;
    let task_id = mint(storage).await;
    let task_singleton_id = mint(storage).await;
    let task_family_id = mint(storage).await;
    let task_other_kind_id = mint(storage).await;
    let base = |owner: &str| json!({ "kind": "base", "version": 1, "value": { "owner": owner } });
    let created_at = commit(
        storage,
        json!([
            { "type": "task", "value": pending_task(task_id, root_id) },
            { "type": "document.create", "record": { "id": first_id, "kind": "cache", "scope": { "kind": "session" }, "key": "__proto__" }, "content": base("first") },
            { "type": "document.create", "record": { "id": second_id, "kind": "cache", "scope": { "kind": "session" }, "key": "constructor" }, "content": base("second") },
            { "type": "document.create", "record": {
                "id": conversation_id,
                "kind": "cache",
                "scope": { "kind": "conversation", "conversationId": root_id },
                "history": "latest",
                "fork": "current",
                "key": "__proto__",
            }, "content": base("conversation") },
            { "type": "document.create", "record": { "id": task_singleton_id, "kind": "task.cache", "scope": { "kind": "task", "taskId": task_id } }, "content": base("singleton") },
            { "type": "document.create", "record": { "id": task_family_id, "kind": "task.cache", "scope": { "kind": "task", "taskId": task_id }, "key": "member" }, "content": base("family") },
            { "type": "document.create", "record": { "id": task_other_kind_id, "kind": "task.other", "scope": { "kind": "task", "taskId": task_id } }, "content": base("other") },
        ]),
    )
    .await
    .unwrap();

    let find = |address: DocumentAddress| async move {
        storage
            .find_document(&address, CURRENT, context())
            .await
            .unwrap()
            .map(|record| record.id.0)
    };
    assert_eq!(
        find(address("cache", session_scope(), Some("__proto__"))).await,
        Some(first_id)
    );
    let sessions = DocumentQuery {
        scope: session_scope(),
        at: CURRENT,
        kind: None,
    };
    assert_eq!(
        storage
            .scan_documents(&sessions, 1, None, context())
            .await
            .unwrap()
            .items
            .len(),
        1
    );
    let first = storage
        .scan_documents(&sessions, 1, None, context())
        .await
        .unwrap();
    let second = storage
        .scan_documents(&sessions, 1, first.next.as_ref(), context())
        .await
        .unwrap();
    assert_eq!(
        first
            .items
            .iter()
            .chain(&second.items)
            .map(|record| record.id.0)
            .collect::<Vec<_>>(),
        [first_id, second_id]
    );
    let conversations = DocumentQuery {
        scope: conversation_scope(root_id),
        at: CURRENT,
        kind: None,
    };
    assert_eq!(
        ids(
            &storage
                .scan_documents(&conversations, 10, None, context())
                .await
                .unwrap(),
            |record| record.id.0
        ),
        [conversation_id]
    );
    let task_scope = DocumentScope::Task {
        task_id: TaskId::new(task_id),
    };
    assert_eq!(
        find(address("task.cache", task_scope, None)).await,
        Some(task_singleton_id)
    );
    assert_eq!(
        find(address("task.cache", task_scope, Some("member"))).await,
        Some(task_family_id)
    );
    let task_cache = DocumentQuery {
        scope: task_scope,
        at: CURRENT,
        kind: Some("task.cache".into()),
    };
    assert_eq!(
        ids(
            &storage
                .scan_documents(&task_cache, 10, None, context())
                .await
                .unwrap(),
            |record| record.id.0
        ),
        [task_singleton_id, task_family_id]
    );
    assert_rejects(
        read_document(storage, task_singleton_id, at(created_at)).await,
        "does not retain historical content",
    );
}

pub async fn keeps_document_lifecycle_failures_atomic_and_gives_create_plus_retire_an_empty_lifetime(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let first_id = mint(storage).await;
    let second_id = mint(storage).await;
    let record = json!({ "id": first_id, "kind": "singleton", "scope": { "kind": "session" } });
    commit(
        storage,
        json!([{ "type": "document.create", "record": record, "content": { "kind": "base", "version": 1, "value": { "value": 1 } } }]),
    )
    .await
    .unwrap();
    assert_rejects(
        commit(
            storage,
            json!([
                { "type": "document.create", "record": with(record.clone(), json!({ "id": second_id })), "content": { "kind": "base", "version": 1, "value": { "value": 2 } } },
                { "type": "document.change", "id": first_id, "content": { "kind": "delta", "version": 1, "ops": [] } },
            ]),
        )
        .await,
        "already has a current incarnation",
    );
    assert_eq!(
        document_value(storage, first_id, CURRENT).await,
        json!({ "value": 1 })
    );
    assert!(
        read_document(storage, second_id, CURRENT)
            .await
            .unwrap()
            .is_none()
    );

    let empty_id = mint(storage).await;
    let empty_at = commit(
        storage,
        json!([
            { "type": "document.create", "record": {
                "id": empty_id,
                "kind": "singleton",
                "key": "empty",
                "scope": { "kind": "conversation", "conversationId": root_id },
                "history": "rewindable",
                "fork": "initial",
            }, "content": { "kind": "base", "version": 1, "value": {} } },
            { "type": "document.retire", "id": empty_id },
        ]),
    )
    .await
    .unwrap();
    assert!(
        read_document(storage, empty_id, CURRENT)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        read_document(storage, empty_id, at(empty_at))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        storage
            .find_document(
                &address("singleton", conversation_scope(root_id), Some("empty")),
                at(empty_at),
                context(),
            )
            .await
            .unwrap()
            .is_none()
    );
}

pub async fn rolls_back_record_tables_and_secondary_indexes_when_a_document_command_fails(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let task_id = mint(storage).await;
    let submission_id = mint(storage).await;
    let document_id = mint(storage).await;
    let task = pending_task(task_id, root_id);
    let submission = json!({
        "id": submission_id,
        "conversationId": root_id,
        "requestId": "atomic",
        "type": "input",
        "status": "queued",
    });
    let record = json!({ "id": document_id, "kind": "atomic", "scope": { "kind": "session" } });
    let baseline_seq = commit(
        storage,
        json!([
            { "type": "task", "value": task },
            { "type": "submission", "value": submission },
            { "type": "document.create", "record": record, "content": { "kind": "base", "version": 1, "value": { "count": 1 } } },
        ]),
    )
    .await
    .unwrap();

    let entry_id = mint(storage).await;
    let conflicting_document_id = mint(storage).await;
    assert_rejects(
        commit(
            storage,
            json!([
                { "type": "task", "value": with(task.clone(), json!({ "state": { "status": "running", "checkpoint": { "phase": "effect" } } })) },
                { "type": "submission", "value": with(submission.clone(), json!({ "status": "unanswered", "reason": "failed" })) },
                { "type": "entry", "value": entry(entry_id, root_id, "transient") },
                { "type": "document.create", "record": with(record.clone(), json!({ "id": conflicting_document_id })), "content": { "kind": "base", "version": 1, "value": { "count": 2 } } },
            ]),
        )
        .await,
        "already has a current incarnation",
    );

    assert_eq!(
        to_json(&storage.task(TaskId::new(task_id), context()).await.unwrap()),
        task
    );
    let pending = TaskQuery {
        status: Some(TaskStatus::Pending),
        ..TaskQuery::default()
    };
    assert_eq!(
        to_json(
            &storage
                .scan_tasks(&pending, 10, None, context())
                .await
                .unwrap()
                .items
        ),
        json!([task])
    );
    assert_eq!(
        to_json(
            &storage
                .submission_by_request(ConversationId(root_id), "atomic", context())
                .await
                .unwrap()
        ),
        submission
    );
    assert!(
        storage
            .entry(EntryId(entry_id), context())
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        read_document(storage, conflicting_document_id, CURRENT)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        storage
            .find_document(
                &address("atomic", session_scope(), None),
                CURRENT,
                context()
            )
            .await
            .unwrap()
            .map(|record| record.id.0),
        Some(document_id)
    );
    let after_rollback_seq = commit(
        storage,
        json!([{ "type": "document.change", "id": document_id, "content": { "kind": "delta", "version": 1, "ops": [["s", ["count"], 3]] } }]),
    )
    .await
    .unwrap();
    assert!(after_rollback_seq > baseline_seq);
}

pub async fn keeps_indexed_string_identities_lossless(storage: Arc<dyn Storage>) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    // Pi uses the lone surrogates "\ud800" and "\ud801"; Rust strings cannot
    // hold them, so the neighbours of the surrogate range stand in.
    let first = "\u{d7ff}";
    let second = "\u{e000}";
    let first_task_id = mint(storage).await;
    let second_task_id = mint(storage).await;
    let first_submission_id = mint(storage).await;
    let second_submission_id = mint(storage).await;
    let first_kind_document_id = mint(storage).await;
    let second_kind_document_id = mint(storage).await;
    let first_key_document_id = mint(storage).await;
    let second_key_document_id = mint(storage).await;
    let queued = |id: u64, request_id: &str| json!({ "id": id, "conversationId": root_id, "requestId": request_id, "type": "input", "status": "queued" });
    commit(
        storage,
        json!([
            { "type": "task", "value": with(pending_task(first_task_id, root_id), json!({ "kind": first })) },
            { "type": "task", "value": with(pending_task(second_task_id, root_id), json!({ "kind": second })) },
            { "type": "submission", "value": queued(first_submission_id, first) },
            { "type": "submission", "value": queued(second_submission_id, second) },
            { "type": "document.create", "record": { "id": first_kind_document_id, "kind": first, "scope": { "kind": "session" } }, "content": { "kind": "base", "version": 1, "value": { "identity": "first kind" } } },
            { "type": "document.create", "record": { "id": second_kind_document_id, "kind": second, "scope": { "kind": "session" } }, "content": { "kind": "base", "version": 1, "value": { "identity": "second kind" } } },
            { "type": "document.create", "record": { "id": first_key_document_id, "kind": "family", "key": first, "scope": { "kind": "session" } }, "content": { "kind": "base", "version": 1, "value": { "identity": "first key" } } },
            { "type": "document.create", "record": { "id": second_key_document_id, "kind": "family", "key": second, "scope": { "kind": "session" } }, "content": { "kind": "base", "version": 1, "value": { "identity": "second key" } } },
        ]),
    )
    .await
    .unwrap();

    let scan_kind = |kind: &str| {
        let query = TaskQuery {
            kind: Some(kind.to_string()),
            ..TaskQuery::default()
        };
        async move {
            ids(
                &storage
                    .scan_tasks(&query, 10, None, context())
                    .await
                    .unwrap(),
                |task| task.id.0,
            )
        }
    };
    assert_eq!(scan_kind(first).await, [first_task_id]);
    assert_eq!(scan_kind(second).await, [second_task_id]);
    let task_kind = |id| async move {
        storage
            .task(TaskId::new(id), context())
            .await
            .unwrap()
            .unwrap()
            .kind
    };
    assert_eq!(task_kind(first_task_id).await, first);
    assert_eq!(task_kind(second_task_id).await, second);
    let by_request = |request_id: &'static str| async move {
        storage
            .submission_by_request(ConversationId(root_id), request_id, context())
            .await
            .unwrap()
            .unwrap()
    };
    assert_eq!(by_request(first).await.request_id.as_deref(), Some(first));
    assert_eq!(
        by_request(first).await.id,
        SubmissionId(first_submission_id)
    );
    assert_eq!(
        by_request(second).await.id,
        SubmissionId(second_submission_id)
    );
    let find = |kind: &'static str, key: Option<&'static str>| async move {
        storage
            .find_document(&address(kind, session_scope(), key), CURRENT, context())
            .await
            .unwrap()
            .map(|record| record.id.0)
    };
    assert_eq!(find(first, None).await, Some(first_kind_document_id));
    assert_eq!(find(second, None).await, Some(second_kind_document_id));
    assert_eq!(
        find("family", Some(first)).await,
        Some(first_key_document_id)
    );
    assert_eq!(
        find("family", Some(second)).await,
        Some(second_key_document_id)
    );
    let by_kind = DocumentQuery {
        scope: session_scope(),
        at: CURRENT,
        kind: Some(first.to_string()),
    };
    assert_eq!(
        ids(
            &storage
                .scan_documents(&by_kind, 10, None, context())
                .await
                .unwrap(),
            |record| record.id.0
        ),
        [first_kind_document_id]
    );
}

pub async fn keeps_one_global_record_id_namespace_and_rejects_exhausted_id_minting(
    storage: Arc<dyn Storage>,
) {
    let storage = &*storage;
    let root_id = create_root(storage).await;
    let explicit_entry_id = 100;
    commit(
        storage,
        json!([{ "type": "entry", "value": entry(explicit_entry_id, root_id, "message") }]),
    )
    .await
    .unwrap();
    assert_eq!(mint(storage).await, 101);
    assert_rejects(
        commit(
            storage,
            json!([{ "type": "task", "value": pending_task(explicit_entry_id, root_id) }]),
        )
        .await,
        &format!("ID {explicit_entry_id} already belongs to entry"),
    );

    commit(
        storage,
        json!([{ "type": "entry", "value": entry(MAX_SAFE_INTEGER, root_id, "last-id") }]),
    )
    .await
    .unwrap();
    assert_rejects(storage.mint_id().await, "ID space is exhausted");
    assert_rejects(storage.mint_id().await, "ID space is exhausted");
}

pub async fn rejects_every_operation_after_close(storage: Arc<dyn Storage>) {
    let storage = &*storage;
    create_root(storage).await;
    storage.close(context()).await.unwrap();
    assert_rejects(
        storage.conversation(ROOT_CONVERSATION_ID, context()).await,
        "closed",
    );
    assert_rejects(storage.commit(&[], context()).await, "closed");
    assert_rejects(storage.mint_id().await, "closed");
}
