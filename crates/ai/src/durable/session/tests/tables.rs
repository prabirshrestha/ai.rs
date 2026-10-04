//! Port of `test/session-tables.test.ts`.
//!
//! Divergence: the "takes ownership of table JSON" case keeps the omitted-data
//! and strict-JSON checks; caller mutation after the call is impossible in
//! Rust, and NaN is represented by a non-finite `f64` field.

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::*;
use crate::chord::JsonValue;
use crate::durable::documents::{define_doc, define_doc_family};
use crate::durable::errors::Error;
use crate::durable::ids::{ConversationId, EntryId, TaskId};
use crate::durable::tasks::{Task, TaskDefinition, define_task};
use crate::durable::types::{
    ConversationOwnership, ConversationQuery, DocDefinition, DocFamilyDefinition, DocumentPoint,
    DocumentQuery, DocumentScope, EntryDraft, EntryHead, EntryQuery, RewindableConversation,
    RewindableFork, Storage, TaskOptions, TaskQuery, TaskRecord, TaskScope,
};
use crate::durable::{DocFamilyToken, DocToken, SessionImpl};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkInput {
    pub path: String,
}

pub type WorkTask = Task<WorkInput, JsonValue, JsonValue>;

pub fn work_task() -> WorkTask {
    define_task(TaskDefinition::new(
        "test.work",
        1,
        |_: &WorkInput| json!({ "phase": "start" }),
    ))
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Progress {
    lines: Vec<String>,
}

fn progress_doc() -> DocToken<Progress, TaskScope> {
    define_doc(DocDefinition::new(
        "test.progress",
        1,
        TaskScope,
        Progress::default,
    ))
    .unwrap()
}

fn step_doc() -> DocFamilyToken<Progress, (), TaskScope> {
    define_doc_family(DocFamilyDefinition::new("test.step", 1, TaskScope, |()| {
        Progress::default()
    }))
    .unwrap()
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Notes {
    text: String,
}

fn notes_doc() -> DocToken<Notes, RewindableConversation> {
    define_doc(DocDefinition::new(
        "test.notes",
        1,
        RewindableConversation {
            fork: RewindableFork::AsOf,
        },
        Notes::default,
    ))
    .unwrap()
}

fn terminal(task: &TaskRecord) -> TaskRecord {
    let mut value = to_json(task);
    value["state"] = json!({ "status": "terminal", "outcome": { "status": "completed", "result": { "ok": true } } });
    value.as_object_mut().unwrap().remove("memos");
    from_json(value)
}

fn task_record(value: JsonValue) -> TaskRecord {
    from_json(value)
}

fn pending(id: TaskId, conversation_id: ConversationId, path: &str, abort: bool) -> TaskRecord {
    task_record(json!({
        "id": id,
        "conversationId": conversation_id,
        "kind": "test.work",
        "version": 1,
        "input": { "path": path },
        "background": false,
        "abortRequested": abort,
        "state": { "status": "pending", "checkpoint": { "phase": "start" } },
    }))
}

async fn create_task(
    session: &SessionImpl,
    conversation_id: ConversationId,
    with_document: bool,
) -> TaskId {
    commit(session, move |tx| async move {
        let task_id = tx
            .create_task(
                &work_task(),
                WorkInput { path: "a".into() },
                TaskOptions::conversation(Some(conversation_id)),
            )
            .await?
            .erase();
        if with_document {
            tx.doc(&progress_doc(), task_id)
                .await?
                .edit(|progress| progress.lines.push("started".into()))?;
        }
        Ok(task_id)
    })
    .await
}

#[tokio::test]
async fn allows_table_reads_only_before_the_first_table_write() {
    let TestSession { session, .. } = open_test_session();
    let conversation_id = create_conversation(&session).await;
    commit(&session, move |tx| async move {
        assert_eq!(
            to_json(&tx.conversation(conversation_id).await?),
            json!({ "id": conversation_id })
        );
        let page = tx
            .scan_conversations(ConversationQuery::default(), 1, None)
            .await?;
        assert_eq!(to_json(&page.items), json!([{ "id": conversation_id }]));
        let tasks = tx
            .scan_tasks(
                TaskQuery {
                    conversation_id: Some(conversation_id),
                    ..TaskQuery::default()
                },
                10,
                None,
            )
            .await?;
        assert!(tasks.items.is_empty());
        let entries = tx
            .scan_entries(EntryQuery::conversation(conversation_id), 10, None)
            .await?;
        assert!(entries.items.is_empty());
        tx.append_entry(conversation_id, EntryDraft::new("note"))
            .await?;
        let error = tx.conversation(conversation_id).await.unwrap_err();
        assert!(matches!(error, Error::ReadAfterWrite(_)));
        assert_err(
            tx.task(TaskId::new(1)).await,
            "Tx.task() cannot read tables after the first table write",
        );
        assert!(matches!(
            tx.entry(EntryId(1)).await,
            Err(Error::ReadAfterWrite(_))
        ));
        assert!(matches!(
            tx.scan_conversations(ConversationQuery::default(), 10, None)
                .await,
            Err(Error::ReadAfterWrite(_))
        ));
        assert!(matches!(
            tx.scan_entries(EntryQuery::conversation(conversation_id), 10, None)
                .await,
            Err(Error::ReadAfterWrite(_))
        ));
        // Document access remains available after table writes.
        tx.doc(&notes_doc(), conversation_id)
            .await?
            .edit(|notes| notes.text = "after write".into())?;
        Ok(())
    })
    .await;
    assert_eq!(
        session
            .snapshot(&notes_doc(), conversation_id, &context())
            .await
            .unwrap(),
        Some(Notes {
            text: "after write".into()
        })
    );
}

#[tokio::test]
async fn passes_caller_selected_limits_and_cursors_through_table_scans() {
    let TestSession { session, .. } = open_test_session();
    let ids = [
        create_conversation(&session).await,
        create_conversation(&session).await,
        create_conversation(&session).await,
    ];
    commit(&session, move |tx| async move {
        let first = tx
            .scan_conversations(ConversationQuery::default(), 2, None)
            .await?;
        let first_ids: Vec<_> = first.items.iter().map(|record| record.id).collect();
        assert_eq!(first_ids, ids[..2]);
        assert!(first.next.is_some());
        let second = tx
            .scan_conversations(ConversationQuery::default(), 2, first.next)
            .await?;
        let second_ids: Vec<_> = second.items.iter().map(|record| record.id).collect();
        assert_eq!(second_ids, ids[2..]);
        assert!(second.next.is_none());
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn treats_synchronous_set_task_as_the_first_table_write() {
    let TestSession { session, .. } = open_test_session();
    let conversation_id = create_conversation(&session).await;
    let task_id = create_task(&session, conversation_id, false).await;
    commit_with(&session, move |tx| async move {
        let task = tx.task(task_id).await?.unwrap();
        tx.set_task(task)?;
        assert!(matches!(
            tx.task(task_id).await,
            Err(Error::ReadAfterWrite(_))
        ));
        Ok(())
    })
    .await;
}

#[tokio::test]
async fn creates_conversations_entries_and_tasks_with_minted_ids() {
    let TestSession {
        session,
        storage,
        publications,
    } = open_test_session();
    let (conversation, first, headed, task) = commit(&session, |tx| async move {
        let conversation = tx
            .create_conversation(ConversationOwnership::Ownerless)
            .await?;
        let first = tx
            .append_entry(
                conversation.id,
                EntryDraft {
                    data: Some(json!("one")),
                    ..EntryDraft::new("note")
                },
            )
            .await?;
        let headed = tx
            .append_entry(
                conversation.id,
                EntryDraft {
                    head: Some(EntryHead::SelfEntry),
                    ..EntryDraft::new("summary")
                },
            )
            .await?;
        let task = tx
            .create_task(
                &work_task(),
                WorkInput { path: "x".into() },
                TaskOptions {
                    background: Some(true),
                    ..TaskOptions::conversation(Some(conversation.id))
                },
            )
            .await?;
        Ok((conversation, first, headed, task.erase()))
    })
    .await;
    let ids: std::collections::HashSet<u64> =
        [conversation.id.0, first.id.0, headed.id.0, task.0].into();
    assert_eq!(ids.len(), 4);
    assert_eq!(headed.head, Some(headed.id));
    assert_eq!(
        to_json(&first),
        json!({ "id": first.id, "conversationId": conversation.id, "kind": "note", "data": "one" })
    );
    assert_eq!(
        storage
            .entry(headed.id, &context())
            .await
            .unwrap()
            .unwrap()
            .entry,
        headed
    );
    assert_eq!(
        to_json(&storage.task(task, &context()).await.unwrap()),
        json!({
            "id": task,
            "conversationId": conversation.id,
            "kind": "test.work",
            "version": 1,
            "input": { "path": "x" },
            "background": true,
            "abortRequested": false,
            "state": { "status": "pending", "checkpoint": { "phase": "start" } },
        })
    );
    flush().await;
    let changes = publications.lock().last().unwrap().changes.clone();
    assert_eq!(changes.len(), 4);
    let mut types: Vec<_> = changes.iter().map(|change| change.type_name()).collect();
    types.sort_unstable();
    assert_eq!(types, ["conversation", "entry", "entry", "task"]);
    for change in &changes {
        match change {
            crate::durable::types::CommitChange::Conversation(value) => {
                assert_eq!(value, &conversation)
            }
            crate::durable::types::CommitChange::Entry(value) => {
                assert!(value == &first || value == &headed)
            }
            _ => {}
        }
    }
    assert_err(
        session
            .commit(
                |tx| async move {
                    tx.create_task(
                        &work_task(),
                        WorkInput { path: "x".into() },
                        TaskOptions::conversation(None),
                    )
                    .await
                },
                &context(),
            )
            .await,
        "requires options.conversationId",
    );
    assert_err(
        session
            .commit(
                |tx| async move {
                    tx.append_entry(ConversationId(12345), EntryDraft::new("note"))
                        .await
                },
                &context(),
            )
            .await,
        "Conversation 12345 does not exist",
    );
}

#[tokio::test]
async fn creates_a_conversation_with_an_explicitly_staged_task_owner() {
    let TestSession {
        session, storage, ..
    } = open_test_session();
    let parent_id = create_conversation(&session).await;
    let (supervisor_id, child) = commit(&session, move |tx| async move {
        let supervisor_id = tx
            .create_task(
                &work_task(),
                WorkInput {
                    path: "background".into(),
                },
                TaskOptions {
                    background: Some(true),
                    ..TaskOptions::conversation(Some(parent_id))
                },
            )
            .await?
            .erase();
        let child = tx
            .create_conversation(ConversationOwnership::Task {
                task_id: supervisor_id,
            })
            .await?;
        Ok((supervisor_id, child))
    })
    .await;
    assert_eq!(
        to_json(&child.owner),
        json!({ "conversationId": parent_id, "taskId": supervisor_id })
    );
    assert_eq!(
        storage.conversation(child.id, &context()).await.unwrap(),
        Some(child.clone())
    );

    let child_id = child.id;
    assert_err(
        session
            .commit(
                move |tx| async move {
                    let mut supervisor = tx.task(supervisor_id).await?.unwrap();
                    supervisor.conversation_id = child_id;
                    tx.set_task(supervisor)
                },
                &context(),
            )
            .await,
        &format!("Task {supervisor_id} cannot change conversations"),
    );
    assert_eq!(
        storage
            .task(supervisor_id, &context())
            .await
            .unwrap()
            .unwrap()
            .conversation_id,
        parent_id
    );
}

#[tokio::test]
async fn rejects_missing_terminal_and_abort_marked_conversation_owners_atomically() {
    let TestSession {
        session, storage, ..
    } = open_test_session();
    let parent_id = create_conversation(&session).await;
    let moved = std::sync::Arc::new(parking_lot::Mutex::new(None));
    let staged = moved.clone();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    let task_id = tx
                        .create_task(
                            &work_task(),
                            WorkInput {
                                path: "move".into(),
                            },
                            TaskOptions::conversation(Some(parent_id)),
                        )
                        .await?
                        .erase();
                    *staged.lock() = Some(task_id);
                    tx.set_task(pending(task_id, ConversationId(998), "move", false))
                },
                &context(),
            )
            .await,
        "cannot change conversations",
    );
    let moved = moved.lock().expect("a staged task ID");
    assert_eq!(storage.task(moved, &context()).await.unwrap(), None);

    assert_err(
        session
            .commit(
                |tx| async move {
                    tx.create_conversation(ConversationOwnership::Task {
                        task_id: TaskId::new(999),
                    })
                    .await
                },
                &context(),
            )
            .await,
        "Conversation owner task 999 does not exist",
    );

    let rejected = std::sync::Arc::new(parking_lot::Mutex::new(None));
    let child = rejected.clone();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    let supervisor_id = tx
                        .create_task(
                            &work_task(),
                            WorkInput {
                                path: "aborting".into(),
                            },
                            TaskOptions::conversation(Some(parent_id)),
                        )
                        .await?
                        .erase();
                    *child.lock() = Some(
                        tx.create_conversation(ConversationOwnership::Task {
                            task_id: supervisor_id,
                        })
                        .await?
                        .id,
                    );
                    tx.set_task(pending(supervisor_id, parent_id, "aborting", true))
                },
                &context(),
            )
            .await,
        "is abort-marked",
    );
    let rejected = rejected.lock().expect("a rejected child ID");
    assert_eq!(
        storage.conversation(rejected, &context()).await.unwrap(),
        None
    );

    assert_err(
        session
            .commit(
                move |tx| async move {
                    let supervisor_id = tx
                        .create_task(
                            &work_task(),
                            WorkInput {
                                path: "terminal".into(),
                            },
                            TaskOptions::conversation(Some(parent_id)),
                        )
                        .await?
                        .erase();
                    tx.create_conversation(ConversationOwnership::Task {
                        task_id: supervisor_id,
                    })
                    .await?;
                    tx.set_task(terminal(&pending(
                        supervisor_id,
                        parent_id,
                        "terminal",
                        false,
                    )))
                },
                &context(),
            )
            .await,
        "is terminal",
    );

    let terminal_owner = create_task(&session, parent_id, false).await;
    commit_with(&session, move |tx| async move {
        tx.set_task(terminal(&tx.task(terminal_owner).await?.unwrap()))
    })
    .await;
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.create_conversation(ConversationOwnership::Task {
                        task_id: terminal_owner,
                    })
                    .await
                },
                &context(),
            )
            .await,
        "is terminal",
    );
}

#[tokio::test]
async fn takes_ownership_of_table_json_and_rejects_non_strict_values() {
    let TestSession {
        session, storage, ..
    } = open_test_session();
    let conversation_id = create_conversation(&session).await;
    let omitted = commit(&session, move |tx| async move {
        tx.append_entry(conversation_id, EntryDraft::new("omitted"))
            .await
    })
    .await;
    assert!(to_json(&omitted).get("data").is_none());
    let stored = storage
        .entry(omitted.id, &context())
        .await
        .unwrap()
        .unwrap();
    assert!(to_json(&stored.entry).get("data").is_none());

    #[derive(Serialize)]
    struct Invalid {
        value: f64,
    }
    let token = crate::durable::entries::define_entry::<Invalid>("invalid").unwrap();
    let commits = storage.commit_count();
    assert_err(
        session
            .commit(
                move |tx| async move {
                    tx.append_entry_of(
                        &token,
                        conversation_id,
                        crate::durable::types::TypedEntryDraft {
                            data: Some(Invalid { value: f64::NAN }),
                            model: None,
                            head: None,
                            edits: None,
                        },
                    )
                    .await
                    .map(|_| ())
                },
                &context(),
            )
            .await,
        "strict JSON",
    );
    assert_eq!(storage.commit_count(), commits);
}

#[tokio::test]
async fn replaces_task_records_completely() {
    let TestSession {
        session, storage, ..
    } = open_test_session();
    let conversation_id = create_conversation(&session).await;
    let task_id = create_task(&session, conversation_id, false).await;
    commit_with(&session, move |tx| async move {
        let task = tx.task(task_id).await?.unwrap();
        let mut value = to_json(&task);
        value["state"] =
            json!({ "status": "running", "checkpoint": { "phase": "next", "step": 2 } });
        value["memos"] = json!({ "choice": "b" });
        tx.set_task(from_json(value))
    })
    .await;
    assert_matches(
        &to_json(&storage.task(task_id, &context()).await.unwrap()),
        &json!({
            "state": { "status": "running", "checkpoint": { "phase": "next", "step": 2 } },
            "memos": { "choice": "b" },
        }),
    );
}

#[tokio::test]
async fn creates_a_task_and_then_its_document_in_one_transaction_without_read_after_write() {
    let test = open_test_session();
    let session = &test.session;
    let conversation_id = create_conversation(session).await;
    let task_id = commit(session, move |tx| async move {
        let task_id = tx
            .create_task(
                &work_task(),
                WorkInput { path: "a".into() },
                TaskOptions::conversation(Some(conversation_id)),
            )
            .await?
            .erase();
        // Validation uses the candidate task record, not a caller table read.
        tx.doc(&progress_doc(), task_id)
            .await?
            .edit(|progress| progress.lines.push("created".into()))?;
        tx.doc(&step_doc(), (task_id, "one".into(), ()))
            .await?
            .edit(|progress| progress.lines.push("step".into()))?;
        Ok(task_id)
    })
    .await;
    assert_eq!(
        session
            .snapshot(&progress_doc(), task_id, &context())
            .await
            .unwrap(),
        Some(Progress {
            lines: vec!["created".into()]
        })
    );
    flush().await;
    let publication = test.last_publication();
    assert!(publication.changes.iter().any(|change| matches!(
        change,
        crate::durable::types::CommitChange::Task(task) if task.id == task_id
    )));
    let documents = document_changes(&publication);
    assert_eq!(documents.len(), 2);
    // Task documents derive their conversation from the task record.
    for document in &documents {
        assert_eq!(document.conversation_id, Some(conversation_id));
    }

    commit(session, move |tx| async move {
        tx.create_conversation(ConversationOwnership::Ownerless)
            .await?;
        tx.doc(&progress_doc(), task_id)
            .await?
            .edit(|progress| progress.lines.push("committed task".into()))?;
        Ok(())
    })
    .await;
    flush().await;
    assert_eq!(
        document_changes(&test.last_publication())[0].conversation_id,
        Some(conversation_id)
    );
}

#[tokio::test]
async fn rejects_task_documents_after_a_terminal_candidate() {
    let TestSession { session, .. } = open_test_session();
    let conversation_id = create_conversation(&session).await;
    let task_id = create_task(&session, conversation_id, true).await;
    commit_with(&session, move |tx| async move {
        let task = tx.task(task_id).await?.unwrap();
        let progress = tx.doc(&progress_doc(), task_id).await?;
        tx.set_task(terminal(&task))?;
        assert_err(
            tx.doc(&progress_doc(), task_id).await,
            &format!("Task {task_id} is terminal"),
        );
        assert_err(
            tx.doc(&step_doc(), (task_id, "late".into(), ())).await,
            &format!("Task {task_id} is terminal"),
        );
        assert_err(tx.set_task(task), "terminal candidate");
        progress.edit(|progress| progress.lines.push("final".into()))?;
        Ok(())
    })
    .await;
    assert_err(
        session
            .commit(
                move |tx| async move { tx.doc(&progress_doc(), task_id).await.map(|_| ()) },
                &context(),
            )
            .await,
        &format!("Task {task_id} is terminal"),
    );
}

#[tokio::test]
async fn retires_task_documents_at_terminal_settlement_including_documents_created_in_the_same_transaction()
 {
    let test = open_test_session();
    let TestSession {
        session, storage, ..
    } = &test;
    let conversation_id = create_conversation(session).await;
    let task_id = create_task(session, conversation_id, true).await;
    commit(session, move |tx| async move {
        tx.doc(&step_doc(), (task_id, "committed".into(), ()))
            .await?
            .edit(|progress| progress.lines.push("x".into()))?;
        Ok(())
    })
    .await;
    flush().await;
    let published = test.published();
    commit_with(session, move |tx| async move {
        let task = tx.task(task_id).await?.unwrap();
        tx.doc(&step_doc(), (task_id, "new".into(), ()))
            .await?
            .edit(|progress| progress.lines.push("created then retired".into()))?;
        tx.set_task(terminal(&task))
    })
    .await;
    let writes = storage.last_commit();
    assert_eq!(writes.len(), 5);
    let types: Vec<_> = writes.iter().map(write_type).collect();
    assert_eq!(types.iter().filter(|kind| *kind == "task").count(), 1);
    let creations: Vec<_> = writes
        .iter()
        .filter(|write| write_type(write) == "document.create")
        .collect();
    let retirements: Vec<_> = writes
        .iter()
        .filter(|write| write_type(write) == "document.retire")
        .map(|write| to_json(write)["id"].clone())
        .collect();
    assert_eq!(creations.len(), 1);
    assert_eq!(retirements.len(), 3);
    assert!(retirements.contains(&to_json(creations[0])["record"]["id"]));
    flush().await;
    assert_eq!(test.published(), published + 1);
    let publication = test.last_publication();
    let documents = document_changes(&publication);
    assert_eq!(
        publication
            .changes
            .iter()
            .filter(|change| change.type_name() == "task")
            .count(),
        1
    );
    assert_eq!(documents.len(), 3);
    for document in &documents {
        assert!(document.value.is_none());
        assert!(document.ops.is_empty());
        assert_eq!(document.conversation_id, Some(conversation_id));
    }
    assert_eq!(
        session
            .snapshot(&progress_doc(), task_id, &context())
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        session
            .snapshot(&step_doc(), (task_id, "committed".into()), &context())
            .await
            .unwrap(),
        None
    );
    let alive = storage
        .scan_documents(
            &DocumentQuery {
                scope: DocumentScope::Task { task_id },
                at: DocumentPoint::Current,
                kind: None,
            },
            10,
            None,
            &context(),
        )
        .await
        .unwrap();
    assert!(alive.items.is_empty());
    assert_err(
        session
            .commit(
                move |tx| async move { tx.set_task(terminal(&tx.task(task_id).await?.unwrap())) },
                &context(),
            )
            .await,
        &format!("Task {task_id} is already terminal"),
    );
}

#[tokio::test]
async fn validates_document_owners() {
    let TestSession { session, .. } = open_test_session();
    assert_err(
        session
            .commit(
                |tx| async move { tx.doc(&progress_doc(), TaskId::new(4242)).await.map(|_| ()) },
                &context(),
            )
            .await,
        "Task 4242 does not exist",
    );
    assert_err(
        session
            .commit(
                |tx| async move { tx.doc(&notes_doc(), ConversationId(4242)).await.map(|_| ()) },
                &context(),
            )
            .await,
        "Conversation 4242 does not exist",
    );
}
