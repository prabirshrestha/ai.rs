//! Port of `test/harness-lifecycle.test.ts` (Harness open, close, and pausing).
//!
//! The SQLite reopen case runs over `ControlledStorage::persistent()`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use serde_json::json;

use super::support::*;
use crate::chord::{AbortController, AbortReason, Context};
use crate::durable::documents::define_doc;
use crate::durable::errors::Error;
use crate::durable::harness::types::{
    AgentChange, ConversationAbortOptions, ConversationCreateOptions, HarnessOptions, Scheduling,
    SubmissionDraft, TaskInspectionState,
};
use crate::durable::harness::{CreateOptions, Harness, create_registry};
use crate::durable::ids::{ConversationId, EntryId, TaskId};
use crate::durable::session::tests::support::{ControlledStorage, Deferred, flush};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{
    ConversationRecord, DocDefinition, EntryDraft, SessionScope, Storage, StorageWrite,
    SubmissionCreate, SubmissionStatus, SubmissionType, TaskRecord, TaskState, WatchEnd,
};
use crate::models::Models;
use crate::types::ModelThinkingLevel;

fn with_signal(controller: &AbortController) -> Context {
    signal_context(controller.signal())
}

/// Commit a conversation with a `running` task, as a crash leaves it, so open has a reconciliation commit to fail.
async fn seed_running_task(storage: &dyn Storage) {
    let conversation_id = ConversationId(storage.mint_id().await.unwrap());
    let id = TaskId::new(storage.mint_id().await.unwrap());
    let task = TaskRecord {
        id,
        conversation_id,
        kind: "test.seeded".into(),
        version: 1,
        input: json!(null),
        owner: None,
        background: false,
        abort_requested: false,
        state: TaskState::Running {
            checkpoint: run_phase(),
        },
        memos: None,
    };
    storage
        .commit(
            &[
                StorageWrite::Conversation {
                    value: ConversationRecord::new(conversation_id),
                },
                StorageWrite::Task { value: task },
            ],
            &context(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn closes_without_a_cancelled_caller_context_and_returns_the_original_error_when_open_fails()
{
    let storage = Arc::new(ControlledStorage::new());
    seed_running_task(&*storage).await;
    let reader = CountingReader::new(create_registry());
    let held = storage.hold_commits();
    storage.fail_next_commit(Error::message("disk full"));
    let controller = AbortController::new();
    let options = HarnessOptions::new(Models::default(), reader.clone());
    let signalled = with_signal(&controller);
    let shared: Arc<dyn Storage> = storage.clone();
    let opening = tokio::spawn(async move { Harness::open(shared, options, &signalled).await });
    held.entered().await;
    controller.abort(Some(AbortReason::message("caller gave up")));
    held.release();
    assert_err(opening.await.unwrap(), "disk full");
    assert_eq!(reader.subscriptions(), 0);
    let storage: Arc<dyn Storage> = storage;
    assert!(!storage_open(&storage).await);
}

#[tokio::test]
async fn reports_a_failing_close_and_still_returns_the_open_error() {
    let storage = Arc::new(ControlledStorage::new());
    seed_running_task(&*storage).await;
    storage.fail_next_commit(Error::message("disk full"));
    storage.fail_close(Error::message("close failed"));
    let reports = Reports::default();
    let mut options = HarnessOptions::new(Models::default(), Arc::new(create_registry()));
    options.on_report = Some(reports.callback());
    assert_err(
        Harness::open(storage, options, &context()).await,
        "disk full",
    );
    assert_eq!(reports.messages(), ["close failed"]);
}

#[tokio::test]
async fn joins_a_task_handler_that_ignores_its_signal_before_closing_storage() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let reached = Deferred::default();
    let gate = Deferred::default();
    let read_after_release = Arc::new(Mutex::new(None::<bool>));
    let stubborn = {
        let (reached, gate, storage, read) = (
            reached.clone(),
            gate.clone(),
            storage.clone(),
            read_after_release.clone(),
        );
        one_step("test.close-stubborn", move || {
            let (reached, gate, storage, read) =
                (reached.clone(), gate.clone(), storage.clone(), read.clone());
            async move {
                reached.resolve();
                gate.wait().await;
                *read.lock() = Some(storage_open(&storage).await);
            }
        })
    };
    let (harness, _, _) = open_tasks(
        storage.clone(),
        vec![stubborn.any()],
        TaskOptions::default(),
    )
    .await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    start(&root, &stubborn, (), false).await;
    harness.resume().unwrap();
    reached.wait().await;
    let closing = {
        let harness = harness.clone();
        async move { harness.close(&context()).await }
    };
    let (done, closing) = settled(closing).await;
    assert!(!done);
    assert!(storage_open(&storage).await);
    gate.resolve();
    closing.await.unwrap().unwrap();
    assert_eq!(*read_after_release.lock(), Some(true));
    assert!(!storage_open(&storage).await);
}

#[tokio::test]
async fn joins_a_tool_execute_and_a_hook_that_ignore_their_signal_before_closing_storage() {
    for in_tool in [true, false] {
        let setup = super::chat::chat_setup();
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let reached = Deferred::default();
        let gate = Deferred::default();
        let read_after_release = Arc::new(Mutex::new(None::<bool>));
        let stubborn = {
            let (reached, gate, storage, read) = (
                reached.clone(),
                gate.clone(),
                storage.clone(),
                read_after_release.clone(),
            );
            move || {
                let (reached, gate, storage, read) =
                    (reached.clone(), gate.clone(), storage.clone(), read.clone());
                async move {
                    reached.resolve();
                    gate.wait().await;
                    *read.lock() = Some(storage_open(&storage).await);
                }
            }
        };
        let in_execute = stubborn.clone();
        add_tool(
            &setup.registry,
            crate::durable::harness::define_tool(
                "wait",
                "wait",
                json!({ "type": "object", "properties": {} }),
                move |_, _, _| {
                    let stubborn = in_execute.clone();
                    async move {
                        if in_tool {
                            stubborn().await;
                        }
                        Ok(crate::durable::harness::types::ToolExecutionResult {
                            content: Some(Vec::new()),
                            ..Default::default()
                        })
                    }
                },
            ),
        );
        add_hooks(
            &setup.registry,
            crate::durable::harness::hook(
                &*crate::durable::harness::generation::GENERATION_TASK,
                crate::durable::harness::types::GenerationHooks {
                    before_request: Some(Arc::new(move |_, _, _| {
                        let stubborn = stubborn.clone();
                        Box::pin(async move {
                            if !in_tool {
                                stubborn().await;
                            }
                            Ok(None)
                        })
                    })),
                    ..Default::default()
                },
            ),
        );
        setup.faux.set_responses([
            crate::providers::faux::faux_assistant_message(
                vec![crate::providers::faux::faux_tool_call(
                    "wait",
                    json!({}),
                    Some("c1"),
                )],
                crate::providers::faux::FauxMessageOptions {
                    stop_reason: Some(crate::types::StopReason::ToolUse),
                    ..Default::default()
                },
            )
            .into(),
            crate::providers::faux::faux_assistant_message(
                "done",
                crate::providers::faux::FauxMessageOptions::default(),
            )
            .into(),
        ]);
        let (harness, root) = super::chat::open_chat(storage.clone(), &setup).await;
        root.submit(SubmissionDraft::input("go"), &context())
            .await
            .unwrap();
        reached.wait().await;
        let closing = {
            let harness = harness.clone();
            async move { harness.close(&context()).await }
        };
        let (done, closing) = settled(closing).await;
        assert!(!done);
        assert!(storage_open(&storage).await);
        gate.resolve();
        closing.await.unwrap().unwrap();
        assert_eq!(*read_after_release.lock(), Some(true));
        assert!(!storage_open(&storage).await);
    }
}

#[tokio::test]
async fn keeps_shutting_down_after_a_cancelled_close_and_a_second_close_awaits_the_same_shutdown() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let reached = Deferred::default();
    let gate = Deferred::default();
    let stubborn = {
        let (reached, gate) = (reached.clone(), gate.clone());
        one_step("test.close-cancelled", move || {
            let (reached, gate) = (reached.clone(), gate.clone());
            async move {
                reached.resolve();
                gate.wait().await;
            }
        })
    };
    let (harness, _, _) = open_tasks(
        storage.clone(),
        vec![stubborn.any()],
        TaskOptions::default(),
    )
    .await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    start(&root, &stubborn, (), false).await;
    harness.resume().unwrap();
    reached.wait().await;
    let controller = AbortController::new();
    let cancelled = {
        let harness = harness.clone();
        let signalled = with_signal(&controller);
        tokio::spawn(async move { harness.close(&signalled).await })
    };
    flush().await;
    controller.abort(Some(AbortReason::message("stop waiting")));
    assert_err(cancelled.await.unwrap(), "stop waiting");
    // Admission stays sealed and the invocation still holds Storage open.
    assert_err(
        harness.commit(|_| async { Ok(()) }, &context()).await,
        "closed",
    );
    assert!(storage_open(&storage).await);
    let second = {
        let harness = harness.clone();
        async move { harness.close(&context()).await }
    };
    let (done, second) = settled(second).await;
    assert!(!done);
    gate.resolve();
    second.await.unwrap().unwrap();
    assert!(!storage_open(&storage).await);
}

#[tokio::test]
async fn settles_durably_a_commit_whose_committer_was_cancelled_while_it_was_in_storage() {
    let storage = Arc::new(ControlledStorage::new());
    let (harness, _, _) = open_tasks(storage.clone(), vec![], TaskOptions::default()).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    let held = storage.hold_commits();
    let controller = AbortController::new();
    let committing = {
        let root = root.clone();
        let signalled = with_signal(&controller);
        tokio::spawn(async move {
            let id = root.id;
            root.commit(
                move |tx| async move { Ok(tx.append_entry(id, EntryDraft::new("note")).await?.id) },
                &signalled,
            )
            .await
        })
    };
    held.entered().await;
    controller.abort(Some(AbortReason::message("committer gave up")));
    held.release();
    let id = committing.await.unwrap().unwrap();
    let page = root
        .entries(None, None, 10, None, &context())
        .await
        .unwrap();
    assert_eq!(
        page.items
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<EntryId>>(),
        [id]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_conversation_and_harness_operations_once_close_begins_inspect_queued_before_reports_closing()
 {
    let storage = Arc::new(ControlledStorage::new());
    let (harness, _, _) = open_tasks(storage.clone(), vec![], TaskOptions::default()).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    let root_id = root.id;
    let entry = root
        .commit(
            move |tx| async move { tx.append_entry(root_id, EntryDraft::new("note")).await },
            &context(),
        )
        .await
        .unwrap();
    let submission = root
        .submit(SubmissionDraft::write(EntryDraft::new("note")), &context())
        .await
        .unwrap();

    let held = storage.hold_commits();
    let blocking = {
        let harness = harness.clone();
        tokio::spawn(async move {
            harness
                .commit(
                    move |tx| async move {
                        tx.append_entry(root_id, EntryDraft::new("blocker")).await?;
                        Ok(())
                    },
                    &context(),
                )
                .await
        })
    };
    held.entered().await;
    let queued_inspect = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.inspect(&context()).await })
    };
    flush().await;
    let closing = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.close(&context()).await })
    };
    flush().await;
    let mut outcomes = Vec::new();
    macro_rules! check {
        ($name:expr, $operation:expr) => {{
            let outcome = match $operation.await {
                Ok(_) => "resolved".to_string(),
                Err(error) if error.to_string().contains("closed") => "closed".to_string(),
                Err(error) => error.to_string(),
            };
            outcomes.push(($name, outcome));
        }};
    }
    let ctx = context();
    check!("submit", root.submit(SubmissionDraft::input("x"), &ctx));
    check!("agent", root.agent(&ctx));
    check!(
        "configure",
        root.configure(
            AgentChange::default().thinking_level(ModelThinkingLevel::Low),
            &ctx
        )
    );
    check!("commit", root.commit(|_| async { Ok(()) }, &ctx));
    check!("context", root.context(&ctx));
    check!("entries", root.entries(None, None, 10, None, &ctx));
    check!(
        "fork",
        root.fork(entry.id, ConversationCreateOptions::ownerless(), &ctx)
    );
    check!("reset", root.reset(None, &ctx));
    check!(
        "abort",
        root.abort(&ctx, ConversationAbortOptions::default())
    );
    check!("conversationIdle", root.wait_for_idle(&ctx));
    check!("status", submission.status(&ctx));
    check!("wait", submission.wait(&ctx));
    check!("abortSubmission", submission.abort(&ctx));
    check!("root", harness.root(&ctx, CreateOptions::default()));
    check!("conversation", harness.conversation(root.id, &ctx));
    check!(
        "createConversation",
        harness.create_conversation(ConversationCreateOptions::ownerless(), &ctx)
    );
    check!("getTask", harness.get_task(TaskId::<()>::new(1), &ctx));
    check!("inspect", harness.inspect(&ctx));
    check!("submission", harness.submission(submission.id, &ctx));
    check!("abortTask", harness.abort_task(TaskId::<()>::new(1), &ctx));
    check!(
        "waitForTask",
        harness.wait_for_task(TaskId::<()>::new(1), &ctx)
    );
    check!("harnessIdle", harness.wait_for_idle(&ctx));
    check!("usage", harness.usage(&ctx));
    for (name, outcome) in &outcomes {
        assert_eq!(outcome, "closed", "{name}");
    }
    assert_err(harness.resume(), "closed");
    held.release();
    blocking.await.unwrap().unwrap();
    assert_eq!(
        queued_inspect.await.unwrap().unwrap().scheduling,
        Scheduling::Closing
    );
    closing.await.unwrap().unwrap();
}

#[tokio::test]
async fn completes_reads_queued_at_the_seal_and_rejects_queued_waits_and_acquisitions_that_would_follow_commits()
 {
    #[derive(serde::Serialize, serde::Deserialize, Default)]
    struct Notes {
        text: String,
    }
    let notes = define_doc(DocDefinition::new(
        "test.queued-at-seal",
        1,
        SessionScope,
        Notes::default,
    ))
    .unwrap();
    let absent = define_doc(DocDefinition::new(
        "test.queued-absent",
        1,
        SessionScope,
        Notes::default,
    ))
    .unwrap();
    let pending = noop_step("test.queued-pending");
    let storage = Arc::new(ControlledStorage::new());
    let (harness, _, _) = open_tasks(storage.clone(), vec![], TaskOptions::default()).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    {
        let notes = notes.clone();
        harness
            .commit(
                move |tx| async move {
                    tx.doc(&notes, ()).await?;
                    Ok(())
                },
                &context(),
            )
            .await
            .unwrap();
    }
    let task_id = start(&root, &pending, (), false).await;
    let write = root
        .submit(SubmissionDraft::write(EntryDraft::new("note")), &context())
        .await
        .unwrap();
    write.wait(&context()).await.unwrap();

    let held = storage.hold_commits();
    let root_id = root.id;
    let blocking = {
        let harness = harness.clone();
        tokio::spawn(async move {
            harness
                .commit(
                    move |tx| async move {
                        tx.append_entry(root_id, EntryDraft::new("blocker")).await?;
                        Ok(())
                    },
                    &context(),
                )
                .await
        })
    };
    held.entered().await;
    fn settle<T: Send + 'static>(
        future: impl Future<Output = crate::durable::errors::Result<Option<T>>> + Send + 'static,
    ) -> tokio::task::JoinHandle<String> {
        tokio::spawn(async move {
            match future.await {
                Ok(None) => "undefined".to_string(),
                Ok(Some(_)) => "resolved".to_string(),
                Err(error) if error.to_string().contains("closed") => "closed".to_string(),
                Err(error) => error.to_string(),
            }
        })
    }
    let queued = vec![
        ("context", {
            let root = root.clone();
            settle(async move { root.context(&context()).await.map(Some) })
        }),
        ("snapshot", settle(harness.snapshot(&notes, (), &context()))),
        ("settledWait", {
            let write = write.clone();
            settle(async move { write.wait(&context()).await.map(Some) })
        }),
        (
            "absentWatch",
            settle(harness.watch_doc(&absent, (), &context())),
        ),
        (
            "absentState",
            settle(harness.document_state(&absent, (), &context())),
        ),
        ("waitForTask", {
            let harness = harness.clone();
            settle(async move { harness.wait_for_task(task_id, &context()).await.map(Some) })
        }),
        (
            "documentState",
            settle(harness.document_state(&notes, (), &context())),
        ),
        (
            "watchDoc",
            settle(harness.watch_doc(&notes, (), &context())),
        ),
    ];
    flush().await;
    let closing = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.close(&context()).await })
    };
    flush().await;
    held.release();
    blocking.await.unwrap().unwrap();
    let mut outcomes = Vec::new();
    for (name, outcome) in queued {
        outcomes.push((name, outcome.await.unwrap()));
    }
    let expected = [
        ("context", "resolved"),
        ("snapshot", "resolved"),
        ("settledWait", "resolved"),
        ("absentWatch", "undefined"),
        ("absentState", "undefined"),
        ("waitForTask", "closed"),
        ("documentState", "closed"),
        ("watchDoc", "closed"),
    ];
    assert_eq!(
        outcomes,
        expected
            .iter()
            .map(|(name, outcome)| (*name, outcome.to_string()))
            .collect::<Vec<_>>()
    );
    closing.await.unwrap().unwrap();
}

/// Open a paused Harness with one background task pending and a queued write submission.
async fn paused() -> (
    Harness,
    crate::durable::harness::Conversation,
    TaskId,
    crate::durable::ids::SubmissionId,
    Arc<AtomicBool>,
) {
    let ran = Arc::new(AtomicBool::new(false));
    let marker = {
        let ran = ran.clone();
        one_step("test.marker", move || {
            let ran = ran.clone();
            async move { ran.store(true, Ordering::SeqCst) }
        })
    };
    let (harness, _, _) = open_tasks(
        Arc::new(MemoryStorage::new()),
        vec![marker.any()],
        TaskOptions::default(),
    )
    .await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    // Background, so idle waits and conversation abort leave it alone.
    let id = start(&root, &marker, (), true).await;
    let root_id = root.id;
    let submission_id = harness
        .commit(
            move |tx| async move {
                Ok(tx
                    .create_submission(SubmissionCreate {
                        conversation_id: root_id,
                        type_: SubmissionType::Write,
                        status: SubmissionStatus::Queued,
                        ..SubmissionCreate::default()
                    })
                    .await?
                    .id)
            },
            &context(),
        )
        .await
        .unwrap();
    (harness, root, id, submission_id, ran)
}

#[tokio::test]
async fn never_schedules_from_a_read_only_viewer() {
    let (harness, root, id, submission_id, ran) = paused().await;
    #[derive(serde::Serialize, serde::Deserialize, Default)]
    struct Notes {
        text: String,
    }
    let notes = define_doc(DocDefinition::new(
        "test.viewer-notes",
        1,
        SessionScope,
        Notes::default,
    ))
    .unwrap();
    {
        let notes = notes.clone();
        harness
            .commit(
                move |tx| async move {
                    tx.doc(&notes, ()).await?;
                    Ok(())
                },
                &context(),
            )
            .await
            .unwrap();
    }
    harness.inspect(&context()).await.unwrap();
    harness.get_task(id, &context()).await.unwrap();
    harness.usage(&context()).await.unwrap();
    harness.conversation(root.id, &context()).await.unwrap();
    harness.snapshot(&notes, (), &context()).await.unwrap();
    harness
        .document_state(&notes, (), &context())
        .await
        .unwrap()
        .unwrap()
        .dispose()
        .unwrap();
    harness
        .watch_doc(&notes, (), &context())
        .await
        .unwrap()
        .unwrap()
        .stop()
        .await;
    let submission = harness
        .submission(submission_id, &context())
        .await
        .unwrap()
        .unwrap();
    submission.status(&context()).await.unwrap();
    root.agent(&context()).await.unwrap();
    root.context(&context()).await.unwrap();
    root.entries(None, None, 10, None, &context())
        .await
        .unwrap();
    flush().await;
    flush().await;
    assert!(!ran.load(Ordering::SeqCst));
    assert_eq!(
        harness.inspect(&context()).await.unwrap().scheduling,
        Scheduling::Paused
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn schedules_from_every_progress_call() {
    for name in [
        "submit",
        "abort",
        "conversationIdle",
        "submissionWait",
        "waitForTask",
        "harnessIdle",
    ] {
        let (harness, root, id, submission_id, ran) = paused().await;
        let pending = {
            let harness = harness.clone();
            tokio::spawn(async move {
                let ctx = context();
                let _ = match name {
                    "submit" => root
                        .submit(SubmissionDraft::write(EntryDraft::new("note")), &ctx)
                        .await
                        .map(|_| ()),
                    "abort" => root.abort(&ctx, ConversationAbortOptions::default()).await,
                    "conversationIdle" => root.wait_for_idle(&ctx).await,
                    "submissionWait" => harness
                        .submission(submission_id, &ctx)
                        .await
                        .unwrap()
                        .unwrap()
                        .wait(&ctx)
                        .await
                        .map(|_| ()),
                    "waitForTask" => harness.wait_for_task(id, &ctx).await.map(|_| ()),
                    _ => harness.wait_for_idle(&ctx).await,
                };
            })
        };
        let flag = ran.clone();
        eventually(|| {
            let flag = flag.clone();
            async move { flag.load(Ordering::SeqCst) }
        })
        .await;
        harness.close(&context()).await.unwrap();
        pending.await.unwrap();
    }
}

#[tokio::test]
async fn runs_what_the_registry_holds_at_resume_a_definition_installed_or_replaced_after_open() {
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let define = |label: &'static str| {
        let log = log.clone();
        one_step("test.late-definition", move || {
            let log = log.clone();
            async move { log.lock().push(label.to_string()) }
        })
    };
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let (harness, registry, _) = open_tasks(storage, vec![], TaskOptions::default()).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    let missing = start(&root, &define("unused"), (), false).await;
    let tasks = harness.inspect(&context()).await.unwrap().tasks;
    assert!(matches!(
        tasks[0].state,
        TaskInspectionState::Blocked {
            reason: crate::durable::harness::BlockedReason::MissingTask,
            ..
        }
    ));
    let installed = add_task(&registry, define("v1").any());
    let tasks = harness.inspect(&context()).await.unwrap().tasks;
    assert!(matches!(tasks[0].state, TaskInspectionState::Ready { .. }));
    // The same extension name replaces the definition in place, still before resume.
    installed.dispose();
    add_task(&registry, define("v2").any());
    harness.resume().unwrap();
    harness.wait_for_task(missing, &context()).await.unwrap();
    assert_eq!(*log.lock(), ["v2"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn lets_a_new_harness_open_the_same_storage_once_close_resolved_no_old_invocation_code_runs()
{
    let storage = Arc::new(ControlledStorage::persistent());
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let gate = Deferred::default();
    let generation = Arc::new(std::sync::atomic::AtomicUsize::new(1));
    let stubborn = {
        let (log, gate, generation) = (log.clone(), gate.clone(), generation.clone());
        one_step("test.generations", move || {
            let (log, gate) = (log.clone(), gate.clone());
            let mine = generation.load(Ordering::SeqCst);
            async move {
                log.lock().push(format!("start {mine}"));
                if mine == 1 {
                    gate.wait().await;
                }
                log.lock().push(format!("end {mine}"));
            }
        })
    };
    let (first, _, _) = open_tasks(
        storage.clone(),
        vec![stubborn.any()],
        TaskOptions::default(),
    )
    .await;
    let root = first
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    let id = start(&root, &stubborn, (), false).await;
    first.resume().unwrap();
    eventually(|| {
        let log = log.clone();
        async move { log.lock().len() == 1 }
    })
    .await;
    let closing = {
        let (closing, log) = (first.close(&context()), log.clone());
        tokio::spawn(async move {
            closing.await.unwrap();
            log.lock().push("closed".into());
        })
    };
    flush().await;
    assert!(!closing.is_finished());
    gate.resolve();
    closing.await.unwrap();
    assert_eq!(*log.lock(), vec!["start 1", "end 1", "closed"]);
    generation.store(2, Ordering::SeqCst);
    let (second, _, _) = open_tasks(
        storage.clone(),
        vec![stubborn.any()],
        TaskOptions::default(),
    )
    .await;
    second.resume().unwrap();
    second.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(
        *log.lock(),
        vec!["start 1", "end 1", "closed", "start 2", "end 2"]
    );
    second.close(&context()).await.unwrap();
}

#[tokio::test]
async fn publishes_no_frame_to_states_and_watches_from_a_commit_that_settles_during_close() {
    #[derive(serde::Serialize, serde::Deserialize, Default)]
    struct Notes {
        text: String,
    }
    let notes = define_doc(DocDefinition::new(
        "test.close-frames",
        1,
        SessionScope,
        Notes::default,
    ))
    .unwrap();
    let storage = Arc::new(ControlledStorage::new());
    let (harness, _, _) = open_tasks(storage.clone(), vec![], TaskOptions::default()).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    {
        let notes = notes.clone();
        harness
            .commit(
                move |tx| async move {
                    tx.doc(&notes, ())
                        .await?
                        .edit(|notes| notes.text = "before".into())
                },
                &context(),
            )
            .await
            .unwrap();
    }
    let doc_state = harness
        .document_state(&notes, (), &context())
        .await
        .unwrap()
        .unwrap();
    let doc_watch = harness
        .watch_doc(&notes, (), &context())
        .await
        .unwrap()
        .unwrap();
    let view_state = root.view_state(&context()).await.unwrap();
    let view_watch = root.watch(&context()).await.unwrap();
    let graph_state = harness.task_graph(&context()).await.unwrap();
    let graph_watch = harness.watch_task_graph(&context()).await.unwrap();
    let frames: Arc<Mutex<Vec<&'static str>>> = Arc::default();
    let record = |name: &'static str| {
        let frames = frames.clone();
        move || frames.lock().push(name)
    };
    {
        let push = record("graphState");
        let _ = graph_state.subscribe(move |_, _, delivery| {
            if delivery.kind == crate::chord::DeliveryKind::Update {
                push();
            }
            crate::chord::ListenerOutcome::ok()
        });
        let push = record("graphWatch");
        graph_watch
            .start(move |_, _, _| {
                push();
                Box::pin(async { Ok(()) })
            })
            .unwrap();
        let push = record("docState");
        let _ = doc_state.subscribe(move |_, _, delivery| {
            if delivery.kind == crate::chord::DeliveryKind::Update {
                push();
            }
            crate::chord::ListenerOutcome::ok()
        });
        let push = record("viewState");
        let _ = view_state.subscribe(move |_, _, delivery| {
            if delivery.kind == crate::chord::DeliveryKind::Update {
                push();
            }
            crate::chord::ListenerOutcome::ok()
        });
        let push = record("docWatch");
        doc_watch
            .start(move |_, _, _| {
                push();
                Box::pin(async { Ok(()) })
            })
            .unwrap();
        let push = record("viewWatch");
        view_watch
            .start(move |_, _, _| {
                push();
                Box::pin(async { Ok(()) })
            })
            .unwrap();
    }
    let doc_value = doc_state.value().unwrap();
    let view_value = view_state.value();

    let held = storage.hold_commits();
    let root_id = root.id;
    let committing = tokio::spawn(harness.commit(
        move |tx| async move {
            tx.doc(&notes, ())
                .await?
                .edit(|notes| notes.text = "during close".into())?;
            tx.append_entry(root_id, EntryDraft::new("note")).await?;
            tx.create_task(
                &noop_step("test.close-graph"),
                (),
                crate::durable::types::TaskOptions::conversation(Some(root_id)),
            )
            .await?;
            Ok(())
        },
        &context(),
    ));
    held.entered().await;
    let closing = tokio::spawn(harness.close(&context()));
    flush().await;
    held.release();
    committing.await.unwrap().unwrap();
    closing.await.unwrap().unwrap();
    flush().await;
    assert!(frames.lock().is_empty());
    assert!(Arc::ptr_eq(&doc_state.value().unwrap(), &doc_value));
    assert!(Arc::ptr_eq(
        &view_state.value().entries,
        &view_value.entries
    ));
    assert!(Arc::ptr_eq(&view_state.value().docs, &view_value.docs));
    assert_eq!(to_json(&graph_state.value()), json!({ "tasks": {} }));
    assert!(matches!(
        graph_watch.closed().await,
        WatchEnd::SessionClosed
    ));
    assert!(matches!(doc_watch.closed().await, WatchEnd::SessionClosed));
    assert!(matches!(view_watch.closed().await, WatchEnd::SessionClosed));
    // The commit itself settled.
    assert!(
        storage
            .last_commit()
            .iter()
            .any(|write| matches!(write, StorageWrite::Entry { .. }))
    );
}
