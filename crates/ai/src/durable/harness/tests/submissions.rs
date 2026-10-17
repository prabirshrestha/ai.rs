//! Port of `test/harness-submissions.test.ts`.
//!
//! Divergences: the reopened SQLite file is `ControlledStorage::persistent()`; the TS storage subclass that holds a
//! submission read is `ControlledStorage::hold_submission_reads()`.

use std::sync::Arc;

use serde_json::json;

use super::chat::*;
use super::support::{assert_err, context, signal_context, to_json};
use crate::chord::{AbortController, AbortReason};
use crate::durable::entries::{ASSISTANT_ENTRY, USER_ENTRY, define_entry};
use crate::durable::errors::Error;
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::submissions::AbortSubmissionResult;
use crate::durable::harness::types::{ConversationCreateOptions, SubmissionDraft, WhenBusy};
use crate::durable::ids::{EntryId, SubmissionId};
use crate::durable::session::tests::support::{ControlledStorage, Deferred};
use crate::durable::types::{
    EntryDraft, Storage, SubmissionCreate, SubmissionSettlement, SubmissionStatus, SubmissionType,
    TaskOptions, TaskQuery, TypedEntryDraft,
};
use crate::providers::faux::{FauxMessageOptions, FauxResponseStep, faux_assistant_message};

fn controlled() -> Arc<ControlledStorage> {
    Arc::new(ControlledStorage::new())
}

fn note(data: Option<serde_json::Value>) -> SubmissionDraft {
    let mut entry = EntryDraft::new("note");
    entry.data = data;
    SubmissionDraft::write(entry)
}

#[tokio::test]
async fn appends_an_idle_write_and_settles_it_done_without_a_turn() {
    let (harness, root) = open_chat(controlled(), &chat_setup()).await;
    let submission = root
        .submit(note(Some(json!({ "text": "x" }))), &context())
        .await
        .unwrap();
    let settled = submission.wait(&context()).await.unwrap();
    let entry = settled.entry.unwrap();
    assert_eq!(
        to_json(&settled),
        json!({ "id": submission.id, "conversationId": root.id, "type": "write", "status": "done", "entry": entry })
    );
    let entries = all_entries(&root).await;
    assert_eq!(
        to_json(&entries),
        json!([{ "id": entry, "conversationId": root.id, "kind": "note", "data": { "text": "x" } }])
    );
    assert_eq!(live_state(&harness, &root).await, Some(Default::default()));
    let id = root.id;
    let tasks = harness
        .commit(
            move |tx| async move {
                tx.scan_tasks(
                    TaskQuery {
                        conversation_id: Some(id),
                        ..TaskQuery::default()
                    },
                    10,
                    None,
                )
                .await
            },
            &context(),
        )
        .await
        .unwrap();
    assert!(tasks.items.is_empty());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn places_idle_input_and_rejects_busy_input_with_when_busy_reject_without_writing() {
    let storage = controlled();
    let setup = chat_setup();
    setup.set_now(|| 42);
    let (step, reached) = unanswered();
    setup.faux.set_responses([step]);
    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    let submission = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap();
    reached.wait().await;
    let record = submission.status(&context()).await.unwrap();
    assert_eq!(record.status, SubmissionStatus::Placed);
    let entry_id = record.entry.unwrap();
    let entry = root
        .commit(
            move |tx| async move { tx.entry_of(&USER_ENTRY, entry_id).await },
            &context(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        to_json(&entry.model),
        json!([{ "role": "user", "content": "hi", "timestamp": 42 }])
    );
    let live = live_state(&harness, &root).await.unwrap();
    let run = live.run.unwrap();
    assert_eq!(run.inputs, [submission.id]);
    assert_eq!(
        harness
            .get_task(run.task_id, &context())
            .await
            .unwrap()
            .unwrap()
            .kind,
        "pi.generation"
    );

    let commits = storage.commit_count();
    let rejected = root
        .submit(
            SubmissionDraft::input("again").when_busy(WhenBusy::Reject),
            &context(),
        )
        .await
        .unwrap_err();
    match rejected {
        Error::ConversationBusy(busy) => assert_eq!(busy.conversation_id, root.id),
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(storage.commit_count(), commits);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn deduplicates_request_ids_per_conversation_before_any_write() {
    let storage = controlled();
    let setup = chat_setup();
    let (step, reached) = unanswered();
    setup.faux.set_responses([step]);
    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    let first = root
        .submit(
            SubmissionDraft::input("hi").with_request_id("r1"),
            &context(),
        )
        .await
        .unwrap();
    reached.wait().await;
    let commits = storage.commit_count();
    // Deduplication runs before the busy check.
    let again = root
        .submit(
            SubmissionDraft::input("different").with_request_id("r1"),
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(again.id, first.id);
    assert_eq!(storage.commit_count(), commits);
    assert_err(
        root.submit(note(None).with_request_id("r1"), &context())
            .await,
        "Request r1 already identifies a submission of type input",
    );
    let status = first.status(&context()).await.unwrap();
    assert_eq!(
        (status.request_id.as_deref(), status.status),
        (Some("r1"), SubmissionStatus::Placed)
    );

    let other = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    let write = other
        .submit(note(None).with_request_id("r1"), &context())
        .await
        .unwrap();
    assert_ne!(write.id, first.id);
    assert_eq!(
        other
            .submit(note(None).with_request_id("r1"), &context())
            .await
            .unwrap()
            .id,
        write.id
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reports_abort_results_and_looks_submissions_up_by_conversation() {
    let setup = chat_setup();
    let release = Deferred::default();
    let gate = release.clone();
    setup
        .faux
        .set_responses([FauxResponseStep::async_factory(move |_, _, _, _| {
            let gate = gate.clone();
            async move {
                gate.wait().await;
                Ok(faux_assistant_message(
                    "answer",
                    FauxMessageOptions::default(),
                ))
            }
        })]);
    let (harness, root) = open_chat(controlled(), &setup).await;
    let submission = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap();
    assert_eq!(
        submission.abort(&context()).await.unwrap(),
        AbortSubmissionResult::AlreadyPlaced
    );
    assert_eq!(
        harness
            .abort_submission(submission.id, &context(), Some(root.id))
            .await
            .unwrap(),
        AbortSubmissionResult::AlreadyPlaced
    );
    let other = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    assert_eq!(
        harness
            .abort_submission(submission.id, &context(), Some(other.id))
            .await
            .unwrap(),
        AbortSubmissionResult::NotFound
    );
    assert_eq!(
        harness
            .abort_submission(SubmissionId(999_999), &context(), None)
            .await
            .unwrap(),
        AbortSubmissionResult::NotFound
    );
    assert!(
        harness
            .submission(SubmissionId(999_999), &context())
            .await
            .unwrap()
            .is_none()
    );

    release.resolve();
    submission.wait(&context()).await.unwrap();
    assert_eq!(
        submission.abort(&context()).await.unwrap(),
        AbortSubmissionResult::Settled
    );
    assert_eq!(
        harness
            .abort_submission(submission.id, &context(), None)
            .await
            .unwrap(),
        AbortSubmissionResult::Settled
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cancels_only_a_wait_and_rejects_pending_waits_on_close() {
    let setup = chat_setup();
    setup.faux.set_responses([unanswered().0]);
    let (harness, root) = open_chat(controlled(), &setup).await;
    let submission = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap();
    let controller = AbortController::new();
    let cancelled = {
        let submission = harness
            .submission(submission.id, &context())
            .await
            .unwrap()
            .unwrap();
        let ctx = signal_context(controller.signal());
        tokio::spawn(async move { submission.wait(&ctx).await })
    };
    let pending = {
        let submission = harness
            .submission(submission.id, &context())
            .await
            .unwrap()
            .unwrap();
        tokio::spawn(async move { submission.wait(&context()).await })
    };
    tokio::task::yield_now().await;
    controller.abort(Some(AbortReason::message("stop waiting")));
    let error = cancelled.await.unwrap().unwrap_err();
    assert!(error.to_string().contains("stop waiting"), "{error}");
    assert_eq!(
        submission.status(&context()).await.unwrap().status,
        SubmissionStatus::Placed
    );
    harness.close(&context()).await.unwrap();
    let error = pending.await.unwrap().unwrap_err();
    assert!(error.to_string().contains("Harness is closed"), "{error}");
}

#[tokio::test]
async fn rejects_a_wait_whose_submission_read_spans_the_start_of_close() {
    let storage = Arc::new(ControlledStorage::new());
    let setup = chat_setup();
    setup.faux.set_responses([unanswered().0]);
    let (harness, root) = open_chat(storage.clone(), &setup).await;
    let submission = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap();
    let held = storage.hold_submission_reads();
    let waiting = tokio::spawn(async move { submission.wait(&context()).await });
    held.entered().await;
    let closing = harness.close(&context());
    held.release();
    let error = waiting.await.unwrap().unwrap_err();
    assert!(error.to_string().contains("Harness is closed"), "{error}");
    closing.await.unwrap();
}

#[tokio::test]
async fn reacquires_a_submission_after_reopen_and_settles_it_durably() {
    let storage = Arc::new(ControlledStorage::persistent());
    let setup = chat_setup();
    // The first process never answers; the reopened one does.
    let (step, reached) = unanswered();
    setup.faux.set_responses([
        step,
        faux_assistant_message("after reopen", FauxMessageOptions::default()).into(),
    ]);
    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    let id = root
        .submit(
            SubmissionDraft::input("hi").with_request_id("print"),
            &context(),
        )
        .await
        .unwrap()
        .id;
    reached.wait().await;
    harness.close(&context()).await.unwrap();

    let (harness, _) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    let submission = harness.submission(id, &context()).await.unwrap().unwrap();
    assert_eq!(
        submission.status(&context()).await.unwrap().status,
        SubmissionStatus::Placed
    );
    harness.resume().unwrap();
    let settled = submission.wait(&context()).await.unwrap();
    assert_eq!(
        (settled.status, settled.type_),
        (SubmissionStatus::Done, SubmissionType::Input)
    );
    harness.close(&context()).await.unwrap();

    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    assert_eq!(
        harness
            .submission(id, &context())
            .await
            .unwrap()
            .unwrap()
            .wait(&context())
            .await
            .unwrap(),
        settled
    );
    let again = root
        .submit(
            SubmissionDraft::input("hi").with_request_id("print"),
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(again.id, id);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn enables_scheduling_when_a_caller_submits_or_waits() {
    let setup = chat_setup();
    setup
        .faux
        .set_responses([faux_assistant_message("answer", FauxMessageOptions::default()).into()]);
    let (harness, root) = open_chat(controlled(), &setup).await;
    // No resume(): submitting asks for progress.
    assert_eq!(
        root.submit(SubmissionDraft::input("hi"), &context())
            .await
            .unwrap()
            .wait(&context())
            .await
            .unwrap()
            .status,
        SubmissionStatus::Done
    );
    harness.close(&context()).await.unwrap();

    let (passive, passive_root) = open_chat(controlled(), &chat_setup()).await;
    let task_id = passive_root
        .commit(
            |tx| async move {
                tx.create_task(
                    &*GENERATION_TASK,
                    Default::default(),
                    TaskOptions::conversation(None),
                )
                .await
            },
            &context(),
        )
        .await
        .unwrap();
    // A committed task alone does not start scheduling; waiting for it does.
    let state = passive
        .get_task(task_id, &context())
        .await
        .unwrap()
        .unwrap()
        .state;
    assert_eq!(to_json(&state)["status"], json!("pending"));
    let settled = passive.wait_for_task(task_id, &context()).await.unwrap();
    assert_eq!(to_json(&settled.state)["status"], json!("terminal"));
    passive.close(&context()).await.unwrap();
}

#[tokio::test]
async fn settles_submissions_by_their_current_record_in_the_transaction() {
    let (harness, root) = open_chat(controlled(), &chat_setup()).await;
    let root_id = root.id;
    let entry = root
        .commit(
            move |tx| async move { Ok(tx.append_entry(root_id, EntryDraft::new("note")).await?.id) },
            &context(),
        )
        .await
        .unwrap();
    let create = |type_: SubmissionType| {
        let harness = harness.clone();
        async move {
            harness
                .commit_with(
                    move |tx| async move {
                        Ok(tx
                            .create_submission(SubmissionCreate {
                                conversation_id: root_id,
                                type_,
                                status: SubmissionStatus::Queued,
                                ..SubmissionCreate::default()
                            })
                            .await?
                            .id)
                    },
                    &context(),
                    Default::default(),
                )
                .await
                .unwrap()
        }
    };
    let queued = create(SubmissionType::Input).await;
    let write = create(SubmissionType::Write).await;
    let answer = SubmissionSettlement::Done { answer: entry };
    for id in [queued, write] {
        let answer = answer.clone();
        assert_err(
            root.commit(
                move |tx| async move { tx.settle_submission(id, answer) },
                &context(),
            )
            .await,
            "is not a placed input",
        );
    }
    assert_err(
        root.commit(
            |tx| async move {
                tx.settle_submission(
                    SubmissionId(999_999),
                    SubmissionSettlement::Unanswered {
                        reason: "x".into(),
                        detail: None,
                    },
                )
            },
            &context(),
        )
        .await,
        "does not exist",
    );

    // A submission created earlier in the same commit settles; a second settlement leaves the first.
    let settle = answer.clone();
    let placed = harness
        .commit_with(
            move |tx| async move {
                let id = tx
                    .create_submission(SubmissionCreate {
                        conversation_id: root_id,
                        type_: SubmissionType::Input,
                        status: SubmissionStatus::Placed,
                        entry: Some(entry),
                        ..SubmissionCreate::default()
                    })
                    .await?
                    .id;
                tx.settle_submission(id, settle)?;
                tx.settle_submission(
                    id,
                    SubmissionSettlement::Unanswered {
                        reason: "late".into(),
                        detail: None,
                    },
                )?;
                Ok(id)
            },
            &context(),
            Default::default(),
        )
        .await
        .unwrap();
    let record = harness
        .submission(placed, &context())
        .await
        .unwrap()
        .unwrap()
        .status(&context())
        .await
        .unwrap();
    assert_eq!(
        (record.status, record.entry, record.answer),
        (SubmissionStatus::Done, Some(entry), Some(entry))
    );
    harness.close(&context()).await.unwrap();
}

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
struct Counter {
    n: i64,
}

#[tokio::test]
async fn appends_and_reads_typed_entries_through_tokens() {
    let counter_token = define_entry::<Counter>("app.counter").unwrap();
    let marker_token = define_entry::<()>("app.marker").unwrap();
    let (harness, root) = open_chat(controlled(), &chat_setup()).await;
    let root_id = root.id;
    let token = counter_token.clone();
    let counter = root
        .commit(
            move |tx| async move {
                tx.append_entry_of(
                    &token,
                    root_id,
                    TypedEntryDraft {
                        data: Some(Counter { n: 1 }),
                        ..TypedEntryDraft::default()
                    },
                )
                .await
            },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(
        to_json(&counter),
        json!({ "id": counter.id, "conversationId": root.id, "kind": "app.counter", "data": { "n": 1 } })
    );
    let token = marker_token.clone();
    let marker = root
        .commit(
            move |tx| async move {
                tx.append_entry_of(&token, root_id, TypedEntryDraft::default())
                    .await
            },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(marker.kind, "app.marker");
    let read = |token: crate::durable::entries::Entry<Counter>, id: EntryId| {
        let root = root.clone();
        async move {
            root.commit(
                move |tx| async move { tx.entry_of(&token, id).await },
                &context(),
            )
            .await
            .unwrap()
        }
    };
    assert_eq!(
        read(counter_token.clone(), counter.id).await,
        Some(counter.clone())
    );
    let token = marker_token.clone();
    let id = counter.id;
    assert!(
        root.commit(
            move |tx| async move { tx.entry_of(&token, id).await },
            &context()
        )
        .await
        .unwrap()
        .is_none()
    );
    assert!(
        read(counter_token.clone(), EntryId(999_999))
            .await
            .is_none()
    );
    assert!(counter_token.is(Some(&counter)));
    assert!(!ASSISTANT_ENTRY.is(Some(&counter)));
    assert_eq!(
        [USER_ENTRY.kind(), ASSISTANT_ENTRY.kind()],
        ["pi.user", "pi.assistant"]
    );
    harness.close(&context()).await.unwrap();
}
