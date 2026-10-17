//! Port of `test/harness-inspect.test.ts`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value as JsonValue, json};

use super::chat::{chat_setup, open_chat, unanswered, wait_for};
use super::support::*;
use super::tasks::Rt;
use crate::chord::Context;
use crate::durable::errors::Error;
use crate::durable::harness::CreateOptions;
use crate::durable::harness::types::{
    BlockedReason, ConversationCreateOptions, Scheduling, SubmissionDraft, TaskInspection,
    TaskInspectionState,
};
use crate::durable::ids::TaskId;
use crate::durable::session::tests::support::Deferred;
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{NextTaskState, TaskDefinition, define_task};
use crate::durable::types::{EntryDraft, JoinPolicy, Storage, TaskOptions as CreateTaskOptions};

type Migrate = Arc<dyn Fn() -> crate::durable::errors::Result<()> + Send + Sync>;

async fn complete(runtime: &Rt, ctx: &Context) -> crate::durable::errors::Result<()> {
    runtime
        .commit(
            |_, _| async { Ok(Some(NextTaskState::completed(JsonValue::Null))) },
            ctx,
        )
        .await
}

/// A one-phase task that completes once `gate` resolves; `migrate` runs when an older stored version migrates.
fn task(name: &str, version: u32, gate: Option<Deferred>, migrate: Option<Migrate>) -> StepTask {
    let definition = TaskDefinition::new(name, version, |_: &()| run_phase())
        .phase("run", move |_, runtime: Rt, ctx| {
            let gate = gate.clone();
            async move {
                if let Some(gate) = gate {
                    gate.wait().await;
                }
                complete(&runtime, &ctx).await
            }
        })
        .abort(|_, _, _| async { Ok(()) });
    define_task(match migrate {
        Some(migrate) => definition.migrate(move |_, _, _| {
            migrate()?;
            Ok(((), run_phase()))
        }),
        None => definition,
    })
}

fn state_of(tasks: &[TaskInspection], id: TaskId) -> Option<TaskInspectionState> {
    tasks
        .iter()
        .find(|entry| entry.record.id == id)
        .map(|entry| entry.state.clone())
}

fn conversation_owned() -> CreateTaskOptions {
    CreateTaskOptions::conversation(None)
}

#[tokio::test]
async fn derives_every_live_tasks_state_without_running_task_code() {
    let gate = Deferred::default();
    let gate_task = task("test.gate", 1, Some(gate.clone()), None);
    let gate_id = Arc::new(parking_lot::Mutex::new(None::<TaskId>));
    let dependent: StepTask = {
        let gate_id = gate_id.clone();
        define_task(
            TaskDefinition::new("test.dependent", 1, |_: &()| json!({ "phase": "wait" }))
                .phase("wait", move |_, runtime: Rt, ctx| {
                    let on = vec![gate_id.lock().unwrap()];
                    async move {
                        runtime
                            .commit(
                                move |_, _| async move {
                                    Ok(Some(NextTaskState::waiting(
                                        json!({ "phase": "done" }),
                                        on,
                                        JoinPolicy::AllSettled,
                                    )))
                                },
                                &ctx,
                            )
                            .await
                    }
                })
                .phase("done", |_, runtime: Rt, ctx| async move {
                    complete(&runtime, &ctx).await
                })
                .abort(|_, _, _| async { Ok(()) }),
        )
    };
    let migrations = Arc::new(AtomicUsize::new(0));
    let counted = migrations.clone();
    let registered = vec![
        gate_task.any(),
        dependent.any(),
        task(
            "test.migrating",
            2,
            None,
            Some(Arc::new(move || {
                counted.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })),
        )
        .any(),
        task("test.no-migration", 2, None, None).any(),
        task(
            "test.failing",
            2,
            None,
            Some(Arc::new(|| Err(Error::message("cannot migrate")))),
        )
        .any(),
        task("test.too-old", 1, None, None).any(),
    ];
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let (harness, _, _) = open_tasks(storage, registered, TaskOptions::default()).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    let ids: Vec<TaskId> = {
        let gate_id = gate_id.clone();
        root.commit(
            move |tx| async move {
                let gate = tx
                    .create_task(&gate_task, (), conversation_owned())
                    .await?
                    .erase();
                *gate_id.lock() = Some(gate);
                let mut ids = vec![gate];
                ids.push(
                    tx.create_task(&dependent, (), conversation_owned())
                        .await?
                        .erase(),
                );
                // Stored by definitions other than the registered ones.
                for (name, version) in [
                    ("test.migrating", 1),
                    ("test.no-migration", 1),
                    ("test.failing", 1),
                    ("test.too-old", 2),
                    ("test.missing", 1),
                ] {
                    let stored = task(name, version, None, None);
                    ids.push(
                        tx.create_task(&stored, (), conversation_owned())
                            .await?
                            .erase(),
                    );
                }
                Ok(ids)
            },
            &context(),
        )
        .await
        .unwrap()
    };
    let [
        gate_id,
        dependent_id,
        migrating,
        no_migration,
        failing,
        too_old,
        missing,
    ] = ids[..]
    else {
        unreachable!()
    };

    let paused = harness.inspect(&context()).await.unwrap();
    assert_eq!(paused.scheduling, Scheduling::Paused);
    let listed: Vec<TaskId> = paused.tasks.iter().map(|entry| entry.record.id).collect();
    assert_eq!(listed, ids);
    let ready = |state: Option<TaskInspectionState>| match state {
        Some(TaskInspectionState::Ready { migrates }) => Some(migrates),
        _ => None,
    };
    assert_eq!(ready(state_of(&paused.tasks, gate_id)), Some(false));
    assert_eq!(ready(state_of(&paused.tasks, dependent_id)), Some(false));
    assert_eq!(ready(state_of(&paused.tasks, migrating)), Some(true));
    // A migration that was never tried is not run to find out.
    assert_eq!(ready(state_of(&paused.tasks, failing)), Some(true));
    let blocked = |state: Option<TaskInspectionState>| match state {
        Some(TaskInspectionState::Blocked { reason, error }) => {
            Some((reason, error.map(|error| error.to_string())))
        }
        _ => None,
    };
    assert_eq!(
        blocked(state_of(&paused.tasks, no_migration)),
        Some((
            BlockedReason::MigrationFailed,
            Some("Task test.no-migration version 2 has no migration from 1".into())
        ))
    );
    assert_eq!(
        blocked(state_of(&paused.tasks, too_old)),
        Some((BlockedReason::TaskTooOld, None))
    );
    assert_eq!(
        blocked(state_of(&paused.tasks, missing)),
        Some((BlockedReason::MissingTask, None))
    );
    assert_eq!(migrations.load(Ordering::SeqCst), 0);
    assert_eq!(
        harness.inspect(&context()).await.unwrap().scheduling,
        Scheduling::Paused
    );

    harness.resume().unwrap();
    eventually(|| {
        let done = migrations.load(Ordering::SeqCst) == 1;
        async move { done }
    })
    .await;
    harness.wait_for_task(migrating, &context()).await.unwrap();
    let running = Arc::new(parking_lot::Mutex::new(None));
    {
        let (harness, running) = (harness.clone(), running.clone());
        wait_for(move || {
            let (harness, running) = (harness.clone(), running.clone());
            async move {
                let inspection = harness.inspect(&context()).await.unwrap();
                let done = matches!(
                    state_of(&inspection.tasks, gate_id),
                    Some(TaskInspectionState::Running)
                ) && matches!(
                    state_of(&inspection.tasks, dependent_id),
                    Some(TaskInspectionState::Waiting { .. })
                );
                *running.lock() = Some(inspection);
                done
            }
        })
        .await;
    }
    let running = running.lock().take().unwrap();
    match state_of(&running.tasks, dependent_id) {
        Some(TaskInspectionState::Waiting { on }) => assert_eq!(on, vec![gate_id]),
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(running.scheduling, Scheduling::Running);
    assert!(state_of(&running.tasks, migrating).is_none());
    assert_eq!(
        blocked(state_of(&running.tasks, failing)),
        Some((
            BlockedReason::MigrationFailed,
            Some("cannot migrate".into())
        ))
    );

    gate.resolve();
    harness
        .wait_for_task(dependent_id, &context())
        .await
        .unwrap();
    let settled = harness.inspect(&context()).await.unwrap();
    let listed: Vec<TaskId> = settled.tasks.iter().map(|entry| entry.record.id).collect();
    assert_eq!(listed, vec![no_migration, failing, too_old, missing]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn lists_unsettled_submissions() {
    let setup = chat_setup();
    let (step, reached) = unanswered();
    setup.faux.set_responses(vec![step]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let other = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    other
        .submit(SubmissionDraft::write(EntryDraft::new("note")), &context())
        .await
        .unwrap();
    let input = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap();
    reached.wait().await;

    let inspection = harness.inspect(&context()).await.unwrap();
    assert_eq!(
        inspection.submissions,
        vec![input.status(&context()).await.unwrap()]
    );
    let tasks: Vec<(String, bool)> = inspection
        .tasks
        .iter()
        .map(|entry| {
            (
                entry.record.kind.clone(),
                matches!(entry.state, TaskInspectionState::Running),
            )
        })
        .collect();
    assert_eq!(tasks, vec![("pi.generation".to_string(), true)]);
    harness.close(&context()).await.unwrap();
}
