//! Port of `test/harness-tasks-recovery.test.ts` (recovery, blocked tasks, definition handover, and open failure).
//!
//! The SQLite close-and-reopen cases run against `ControlledStorage::persistent()`, a memory backend whose data
//! survives `close()`, standing in for a database file reopened at the same path.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use super::support::*;
use super::tasks::{Rt, abort_with, complete, outcome, shared, status};
use crate::chord::Context;
use crate::durable::documents::define_doc;
use crate::durable::errors::{Error, Result};
use crate::durable::harness::registry::Registry;
use crate::durable::harness::scheduler::{AbortTaskResult, TaskRuntime};
use crate::durable::harness::types::HarnessOptions;
use crate::durable::harness::{CreateOptions, Harness, create_registry};
use crate::durable::ids::TaskId;
use crate::durable::session::tests::support::{ControlledStorage, Deferred, flush};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{NextTaskState, Task, TaskDefinition, define_task};
use crate::durable::types::{
    DocDefinition, EntryDraft, Storage, StorageWrite, TaskOptions as CreateTaskOptions,
    TaskOutcome, TaskQuery, TaskScope,
};
use crate::models::Models;

type Opened = (Harness, Registry, Reports);

async fn open(
    storage: &Arc<ControlledStorage>,
    tasks: Vec<crate::durable::tasks::AnyTask>,
) -> Opened {
    open_tasks(storage.clone(), tasks, TaskOptions::default()).await
}

async fn create_in<I, S, R, H>(harness: &Harness, task: &Task<I, S, R, H>, input: I) -> TaskId
where
    I: Serialize + Send + 'static,
    S: Serialize + 'static,
    R: 'static,
    H: Send + Sync + 'static,
{
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    start(&root, task, input, false).await
}

fn task_state(record: Option<crate::durable::types::TaskRecord>) -> JsonValue {
    to_json(&record.expect("task"))
}

/// Fake external service whose operations are idempotent by request key.
#[derive(Default)]
struct TransferService {
    applied: Mutex<HashMap<String, u64>>,
    calls: AtomicUsize,
}

impl TransferService {
    fn apply(&self, key: &str, amount: u64) -> u64 {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self
            .applied
            .lock()
            .entry(key.to_string())
            .or_insert(amount * 10)
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[derive(Serialize, Deserialize)]
struct Amount {
    amount: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
enum TransferState {
    Prepare,
    Apply { key: String },
}

#[derive(Serialize, Deserialize)]
struct Receipt {
    receipt: u64,
}

/// Intent/effect/outcome task: `prepare` commits the intent, `apply` performs the effect and commits the outcome.
/// While `interrupt` is set, the first `apply` blocks after the effect until the invocation is signalled.
fn transfer_task(
    service: Arc<TransferService>,
    interrupt: Arc<AtomicBool>,
) -> Task<Amount, TransferState, Receipt> {
    define_task(
        TaskDefinition::new("test.transfer", 1, |_: &Amount| TransferState::Prepare)
            .phase("prepare", |task, runtime, ctx| async move {
                runtime
                    .memo_with("requested", json!(task.input.amount), &ctx)
                    .await?;
                let key = format!("transfer-{}", task.id);
                runtime
                    .commit(
                        move |_, _| async move {
                            Ok(Some(NextTaskState::running(TransferState::Apply { key })))
                        },
                        &ctx,
                    )
                    .await
            })
            .phase("apply", move |task, runtime, ctx| {
                let (service, interrupt) = (service.clone(), interrupt.clone());
                async move {
                    let TransferState::Apply { key } = &task.checkpoint else {
                        unreachable!()
                    };
                    let receipt = service.apply(key, task.input.amount);
                    if interrupt.swap(false, Ordering::SeqCst) {
                        aborted(runtime.signal()).await?;
                    }
                    runtime
                        .commit(
                            move |_, _| async move {
                                Ok(Some(NextTaskState::completed(Receipt { receipt })))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(|_, runtime, ctx| async move {
                runtime
                    .commit(
                        |_, _| async {
                            Ok(Some(NextTaskState::Terminal {
                                outcome: TaskOutcome::Aborted {
                                    reason: None,
                                    result: None,
                                },
                            }))
                        },
                        &ctx,
                    )
                    .await
            }),
    )
}

type Migrate = Arc<dyn Fn(JsonValue, JsonValue, u32) -> Result<((), JsonValue)> + Send + Sync>;

/// A versioned one-phase task that completes with `result`, optionally migrating older records.
fn versioned(version: u32, result: &str, migrate: Option<Migrate>) -> StepTask {
    let (completed, aborted) = (result.to_string(), result.to_string());
    let definition = TaskDefinition::new("test.versioned", version, |_: &()| run_phase())
        .phase("run", move |_, runtime: Rt, ctx| {
            let completed = completed.clone();
            async move { complete(&runtime, json!(completed), &ctx).await }
        })
        .abort(move |_, runtime, ctx| {
            let aborted = aborted.clone();
            async move { abort_with(&runtime, &aborted, &ctx).await }
        });
    define_task(match migrate {
        Some(migrate) => {
            definition.migrate(move |input, checkpoint, from| migrate(input, checkpoint, from))
        }
        None => definition,
    })
}

fn never<T>() -> impl std::future::Future<Output = T> {
    futures::future::pending()
}

// ─── task recovery ───────────────────────────────────────────────────────────

#[tokio::test]
async fn resumes_an_intent_effect_outcome_task_interrupted_after_its_intent_across_close_and_reopen()
 {
    let storage = Arc::new(ControlledStorage::persistent());
    let service = Arc::new(TransferService::default());
    let transfer = transfer_task(service.clone(), Arc::new(AtomicBool::new(true)));

    let (first, _, _) = open(&storage, vec![transfer.any()]).await;
    let id = create_in(&first, &transfer, Amount { amount: 7 }).await;
    first.resume().unwrap();
    eventually(|| {
        let service = service.clone();
        async move { service.calls() == 1 }
    })
    .await;
    first.close(&context()).await.unwrap();

    let (second, _, _) = open(&storage, vec![transfer.any()]).await;
    // Open reconciled `running` to `pending` and kept the checkpoint and memos; nothing ran yet.
    assert_matches(
        &task_state(second.get_task(id, &context()).await.unwrap()),
        &json!({
            "abortRequested": false,
            "memos": { "requested": 7 },
            "state": { "status": "pending", "checkpoint": { "phase": "apply", "key": format!("transfer-{id}") } },
        }),
    );
    assert_eq!(service.calls(), 1);
    second.resume().unwrap();
    let receipt = second.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(
        outcome(&receipt),
        json!({ "status": "completed", "result": { "receipt": 70 } })
    );
    assert_eq!(service.calls(), 2);
    assert_eq!(service.applied.lock().len(), 1);
    second.close(&context()).await.unwrap();

    let (third, _, _) = open(&storage, vec![transfer.any()]).await;
    assert_eq!(third.get_task(id, &context()).await.unwrap(), Some(receipt));
    third.close(&context()).await.unwrap();
}

#[tokio::test]
async fn resumes_abort_work_after_close_at_every_direct_task_abort_stage() {
    let storage = Arc::new(ControlledStorage::persistent());
    let log = shared::<Vec<String>>();
    let abort_gate = Arc::new(AtomicBool::new(true));
    let abort_reached = Deferred::default();
    let run_release = Deferred::default();
    let abortable = {
        let (log_run, log_abort) = (log.clone(), log.clone());
        let (abort_gate, abort_reached, run_release) = (
            abort_gate.clone(),
            abort_reached.clone(),
            run_release.clone(),
        );
        step_task_with(
            "test.abortable",
            move |_, _, _| {
                log_run.lock().push("run".into());
                // Ignores the abort signal until released, so the mark is durable while the run is active.
                let run_release = run_release.clone();
                async move {
                    run_release.wait().await;
                    Ok(())
                }
            },
            move |runtime, ctx| {
                log_abort.lock().push("abort".into());
                let (abort_gate, abort_reached) = (abort_gate.clone(), abort_reached.clone());
                async move {
                    if abort_gate.load(Ordering::SeqCst) {
                        abort_reached.resolve();
                        aborted(runtime.signal()).await?;
                    }
                    abort_with(&runtime, "stop", &ctx).await
                }
            },
        )
    };
    let log_is = |expected: &[&str]| assert_eq!(*log.lock(), expected);

    // Stage 1: the mark is committed while the run invocation is still active.
    let (harness, _, _) = open(&storage, vec![abortable.any()]).await;
    let id = create_in(&harness, &abortable, ()).await;
    harness.resume().unwrap();
    eventually(|| {
        let log = log.clone();
        async move { log.lock().len() == 1 }
    })
    .await;
    let aborting = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.abort_task(id, &context()).await })
    };
    while !harness
        .get_task(id, &context())
        .await
        .unwrap()
        .is_some_and(|task| task.abort_requested)
    {
        flush().await;
    }
    let closing = harness.close(&context());
    run_release.resolve();
    closing.await.unwrap();
    assert_eq!(aborting.await.unwrap().unwrap(), AbortTaskResult::Marked);
    log_is(&["run"]);

    // Stage 2: reopen dispatches the abort invocation, never the run; close while it is active.
    let (harness, _, _) = open(&storage, vec![abortable.any()]).await;
    assert_matches(
        &task_state(harness.get_task(id, &context()).await.unwrap()),
        &json!({ "abortRequested": true, "state": { "status": "pending" } }),
    );
    harness.resume().unwrap();
    abort_reached.wait().await;
    harness.close(&context()).await.unwrap();
    log_is(&["run", "abort"]);

    // Stage 3: a fresh abort invocation settles the task.
    abort_gate.store(false, Ordering::SeqCst);
    let (harness, _, _) = open(&storage, vec![abortable.any()]).await;
    harness.resume().unwrap();
    assert_eq!(
        outcome(&harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "stop" })
    );
    harness.close(&context()).await.unwrap();
    log_is(&["run", "abort", "abort"]);

    // Stage 4: the terminal receipt survives reopen and nothing runs again.
    let (harness, _, _) = open(&storage, vec![abortable.any()]).await;
    harness.resume().unwrap();
    flush().await;
    assert_eq!(
        harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Terminal
    );
    harness.close(&context()).await.unwrap();
    log_is(&["run", "abort", "abort"]);
}

/// `oneStep`-like task with an explicit abort handler.
fn step_task_with<F, Fut, A, AFut>(name: &str, run: F, abort: A) -> StepTask
where
    F: Fn(TaskId, Rt, Context) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    A: Fn(Rt, Context) -> AFut + Send + Sync + 'static,
    AFut: std::future::Future<Output = Result<()>> + Send + 'static,
{
    super::tasks::step_with(name, run, abort)
}

// ─── task crash recovery ─────────────────────────────────────────────────────
//
// Crash simulation: the crashed Harness is abandoned without close, its held storage commit never lands, and its
// blocked handlers never return. A new Harness then opens the same storage.

type Log = Arc<Mutex<Vec<String>>>;

/// A task whose run ignores its signal and whose abort handler blocks while `block_abort` is set.
fn crash_task(log: &Log, block_abort: bool) -> StepTask {
    let (log_run, log_abort) = (log.clone(), log.clone());
    step_task_with(
        "test.crash",
        move |_, _, _| {
            log_run.lock().push("run".into());
            never()
        },
        move |runtime, ctx| {
            log_abort.lock().push("abort".into());
            async move {
                if block_abort {
                    never::<()>().await;
                }
                abort_with(&runtime, "recovered", &ctx).await
            }
        },
    )
}

async fn recover(storage: &Arc<ControlledStorage>, log: &Log) -> (Harness, TaskId) {
    storage.crash();
    let (harness, _, _) = open(storage, vec![crash_task(log, false).any()]).await;
    let page = harness
        .commit(
            |tx| async move {
                tx.scan_tasks(
                    TaskQuery {
                        kind: Some("test.crash".into()),
                        ..TaskQuery::default()
                    },
                    1,
                    None,
                )
                .await
            },
            &context(),
        )
        .await
        .unwrap();
    let id = page.items[0].id;
    (harness, id)
}

async fn crashed_run(log: &Log) -> (Arc<ControlledStorage>, Harness, TaskId) {
    let storage = Arc::new(ControlledStorage::new());
    let (harness, _, _) = open(&storage, vec![crash_task(log, true).any()]).await;
    let id = create_in(&harness, &crash_task(log, true), ()).await;
    harness.resume().unwrap();
    eventually(|| {
        let log = log.clone();
        async move { log.lock().len() == 1 }
    })
    .await;
    (storage, harness, id)
}

async fn log_len(log: &Log, len: usize) {
    eventually(|| {
        let log = log.clone();
        async move { log.lock().len() == len }
    })
    .await;
}

#[tokio::test]
async fn crash_while_the_mark_commit_is_in_storage_the_run_resumes() {
    let log = shared::<Vec<String>>();
    let (storage, harness, id) = crashed_run(&log).await;
    let held = storage.hold_commits();
    tokio::spawn(async move {
        let _ = harness.abort_task(id, &context()).await;
    });
    held.entered().await;
    let (recovered, _) = recover(&storage, &log).await;
    assert_matches(
        &task_state(recovered.get_task(id, &context()).await.unwrap()),
        &json!({ "abortRequested": false, "state": { "status": "pending" } }),
    );
    recovered.resume().unwrap();
    log_len(&log, 2).await;
    assert_eq!(*log.lock(), vec!["run", "run"]);
}

#[tokio::test]
async fn crash_after_the_mark_before_the_run_joins_only_the_abort_handler_runs() {
    let log = shared::<Vec<String>>();
    let (storage, harness, id) = crashed_run(&log).await;
    // The run ignores its signal, so abort_task never finishes joining it.
    let joined = Arc::new(AtomicBool::new(false));
    {
        let (harness, joined) = (harness.clone(), joined.clone());
        tokio::spawn(async move {
            let _ = harness.abort_task(id, &context()).await;
            joined.store(true, Ordering::SeqCst);
        });
    }
    while !harness
        .get_task(id, &context())
        .await
        .unwrap()
        .is_some_and(|task| task.abort_requested)
    {
        flush().await;
    }
    flush().await;
    assert!(!joined.load(Ordering::SeqCst));
    let (recovered, _) = recover(&storage, &log).await;
    assert_matches(
        &task_state(recovered.get_task(id, &context()).await.unwrap()),
        &json!({ "abortRequested": true, "state": { "status": "pending" } }),
    );
    recovered.resume().unwrap();
    assert_eq!(
        outcome(&recovered.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "recovered" })
    );
    assert_eq!(*log.lock(), vec!["run", "abort"]);
    recovered.close(&context()).await.unwrap();
}

#[tokio::test]
async fn crash_while_the_abort_handler_runs_a_fresh_abort_invocation_settles_the_task() {
    let log = shared::<Vec<String>>();
    let storage = Arc::new(ControlledStorage::new());
    let crash = crash_task(&log, true);
    let (harness, _, _) = open(&storage, vec![crash.any()]).await;
    let id = create_in(&harness, &crash, ()).await;
    harness.abort_task(id, &context()).await.unwrap();
    harness.resume().unwrap();
    log_len(&log, 1).await;
    assert_eq!(*log.lock(), vec!["abort"]);
    assert_eq!(
        status(harness.get_task(id, &context()).await.unwrap()),
        "running"
    );
    let (recovered, _) = recover(&storage, &log).await;
    recovered.resume().unwrap();
    assert_eq!(
        outcome(&recovered.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "recovered" })
    );
    assert_eq!(*log.lock(), vec!["abort", "abort"]);
    recovered.close(&context()).await.unwrap();
}

#[tokio::test]
async fn crash_while_the_abort_outcome_is_in_storage_the_abort_handler_runs_again() {
    let log = shared::<Vec<String>>();
    let storage = Arc::new(ControlledStorage::new());
    let reached = Deferred::default();
    let proceed = Deferred::default();
    let crash = {
        let log = log.clone();
        let (reached, proceed) = (reached.clone(), proceed.clone());
        step_task_with(
            "test.crash",
            |_, _, _| async { Ok(()) },
            move |runtime, ctx| {
                log.lock().push("abort".into());
                let (reached, proceed) = (reached.clone(), proceed.clone());
                async move {
                    reached.resolve();
                    proceed.wait().await;
                    abort_with(&runtime, "lost", &ctx).await
                }
            },
        )
    };
    let (harness, _, _) = open(&storage, vec![crash.any()]).await;
    let id = create_in(&harness, &crash, ()).await;
    harness.abort_task(id, &context()).await.unwrap();
    harness.resume().unwrap();
    reached.wait().await;
    let held = storage.hold_commits();
    proceed.resolve();
    held.entered().await;
    let (recovered, _) = recover(&storage, &log).await;
    recovered.resume().unwrap();
    assert_eq!(
        outcome(&recovered.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "recovered" })
    );
    assert_eq!(*log.lock(), vec!["abort", "abort"]);
    recovered.close(&context()).await.unwrap();
}

#[tokio::test]
async fn crash_after_the_terminal_outcome_nothing_runs_again() {
    let log = shared::<Vec<String>>();
    let storage = Arc::new(ControlledStorage::new());
    let crash = crash_task(&log, false);
    let (harness, _, _) = open(&storage, vec![crash.any()]).await;
    let id = create_in(&harness, &crash, ()).await;
    harness.abort_task(id, &context()).await.unwrap();
    harness.resume().unwrap();
    harness.wait_for_task(id, &context()).await.unwrap();
    let (recovered, _) = recover(&storage, &log).await;
    recovered.resume().unwrap();
    flush().await;
    assert_eq!(*log.lock(), vec!["abort"]);
    assert_eq!(
        to_json(
            &recovered
                .get_task(id, &context())
                .await
                .unwrap()
                .unwrap()
                .state
        ),
        json!({ "status": "terminal", "outcome": { "status": "aborted", "reason": "recovered" } })
    );
    recovered.close(&context()).await.unwrap();
}

#[tokio::test]
async fn crash_while_the_reservation_commit_is_in_storage_the_task_is_still_pending() {
    let log = shared::<Vec<String>>();
    let storage = Arc::new(ControlledStorage::new());
    let crash = crash_task(&log, false);
    let (harness, _, _) = open(&storage, vec![crash.any()]).await;
    let id = create_in(&harness, &crash, ()).await;
    let held = storage.hold_commits();
    harness.resume().unwrap();
    held.entered().await;
    let (recovered, _) = recover(&storage, &log).await;
    assert_eq!(
        status(recovered.get_task(id, &context()).await.unwrap()),
        "pending"
    );
    assert!(log.lock().is_empty());
    recovered.abort_task(id, &context()).await.unwrap();
    recovered.resume().unwrap();
    recovered.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(*log.lock(), vec!["abort"]);
    recovered.close(&context()).await.unwrap();
}

// ─── blocked tasks ───────────────────────────────────────────────────────────

#[tokio::test]
async fn keeps_a_task_with_a_missing_definition_pending_and_live_for_idle_waits_until_registration()
{
    let v1 = versioned(1, "v1", None);
    let (harness, registry, _) = open_tasks(
        Arc::new(MemoryStorage::new()),
        vec![],
        TaskOptions::default(),
    )
    .await;
    let id = create_in(&harness, &v1, ()).await;
    harness.resume().unwrap();
    flush().await;
    assert_eq!(
        status(harness.get_task(id, &context()).await.unwrap()),
        "pending"
    );
    let idle = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.wait_for_idle(&context()).await })
    };
    flush().await;
    add_task(&registry, v1.any());
    idle.await.unwrap().unwrap();
    assert_eq!(
        to_json(
            &harness
                .get_task(id, &context())
                .await
                .unwrap()
                .unwrap()
                .state
        ),
        json!({ "status": "terminal", "outcome": { "status": "completed", "result": "v1" } })
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_a_task_stored_by_a_newer_version_pending_until_a_fitting_definition_is_registered() {
    let registry = create_registry();
    add_task(&registry, versioned(1, "old", None).any());
    let (harness, _, _) = open_tasks(
        Arc::new(MemoryStorage::new()),
        vec![],
        TaskOptions {
            registry: Some(registry.clone()),
            ..TaskOptions::default()
        },
    )
    .await;
    let id = create_in(&harness, &versioned(2, "new", None), ()).await;
    harness.resume().unwrap();
    flush().await;
    assert_eq!(
        status(harness.get_task(id, &context()).await.unwrap()),
        "pending"
    );
    // The same extension name replaces the old one in place.
    add_task(&registry, versioned(2, "new", None).any());
    assert_eq!(
        outcome(&harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "completed", "result": "new" })
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn migrates_at_reservation_leaves_the_record_unchanged_when_migration_fails_and_retries_only_for_a_new_definition()
 {
    let storage = Arc::new(ControlledStorage::persistent());
    let (harness, _, _) = open(&storage, vec![]).await;
    let id = create_in(&harness, &versioned(1, "v1", None), ()).await;
    harness.close(&context()).await.unwrap();

    let registry = create_registry();
    let failures = Arc::new(AtomicUsize::new(0));
    {
        let failures = failures.clone();
        add_task(
            &registry,
            versioned(
                2,
                "v2",
                Some(Arc::new(move |_, _, _| {
                    failures.fetch_add(1, Ordering::SeqCst);
                    Err(Error::message("cannot migrate"))
                })),
            )
            .any(),
        );
    }
    let (harness, _, reports) = open_tasks(
        storage.clone(),
        vec![],
        TaskOptions {
            registry: Some(registry.clone()),
            ..TaskOptions::default()
        },
    )
    .await;
    harness.resume().unwrap();
    eventually(|| {
        let reports = reports.clone();
        async move { reports.len() == 1 }
    })
    .await;
    // An unrelated registry change wakes the scheduler without retrying the same failed definition.
    add_tool(&registry, tool_described("unrelated", "unrelated"));
    flush().await;
    assert_eq!(failures.load(Ordering::SeqCst), 1);
    assert_eq!(reports.len(), 1);
    assert!(reports.messages()[0].contains("cannot migrate"));
    assert_matches(
        &task_state(harness.get_task(id, &context()).await.unwrap()),
        &json!({ "version": 1, "state": { "status": "pending", "checkpoint": { "phase": "run" } } }),
    );

    let migrations = shared::<Vec<u32>>();
    {
        let migrations = migrations.clone();
        // The same extension name replaces the old one in place.
        add_task(
            &registry,
            versioned(
                2,
                "v2",
                Some(Arc::new(move |_, checkpoint, from| {
                    migrations.lock().push(from);
                    Ok(((), checkpoint))
                })),
            )
            .any(),
        );
    }
    let receipt = harness.wait_for_task(id, &context()).await.unwrap();
    assert_matches(
        &to_json(&receipt),
        &json!({ "version": 2, "state": { "outcome": { "status": "completed", "result": "v2" } } }),
    );
    assert_eq!(*migrations.lock(), vec![1]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn blocks_an_older_record_whose_newer_definition_has_no_migration() {
    let storage = Arc::new(ControlledStorage::persistent());
    let (harness, _, _) = open(&storage, vec![]).await;
    let id = create_in(&harness, &versioned(1, "v1", None), ()).await;
    harness.close(&context()).await.unwrap();
    let (harness, _, reports) = open(&storage, vec![versioned(2, "v2", None).any()]).await;
    harness.resume().unwrap();
    eventually(|| {
        let reports = reports.clone();
        async move { reports.len() == 1 }
    })
    .await;
    assert!(
        reports.messages()[0].contains("has no migration from 1"),
        "{:?}",
        reports.messages()
    );
    assert_matches(
        &task_state(harness.get_task(id, &context()).await.unwrap()),
        &json!({ "version": 1, "state": { "status": "pending" } }),
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn settles_an_aborted_blocked_task_as_orphaned_and_retires_its_documents() {
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    struct Scratch {
        n: u32,
    }
    let scratch = define_doc(DocDefinition::new(
        "test.orphan-scratch",
        1,
        TaskScope,
        Scratch::default,
    ))
    .unwrap();
    let (harness, _, _) = open_tasks(
        Arc::new(MemoryStorage::new()),
        vec![],
        TaskOptions::default(),
    )
    .await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    let id = {
        let scratch = scratch.clone();
        let task = versioned(1, "x", None);
        root.commit(
            move |tx| async move {
                let created = tx
                    .create_task(&task, (), CreateTaskOptions::conversation(None))
                    .await?
                    .erase();
                tx.doc(&scratch, created).await?.edit(|value| value.n = 1)?;
                Ok(created)
            },
            &context(),
        )
        .await
        .unwrap()
    };
    // Before resume: the marking commit settles the blocked task directly.
    assert_eq!(
        harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    assert_eq!(
        to_json(
            &harness
                .get_task(id, &context())
                .await
                .unwrap()
                .unwrap()
                .state
        ),
        json!({ "status": "terminal", "outcome": { "status": "orphaned", "reason": "missing_task" } })
    );
    assert_eq!(
        harness.snapshot(&scratch, id, &context()).await.unwrap(),
        None
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn orphans_a_marked_task_whose_definition_disappeared_while_its_run_was_active() {
    let reached = Deferred::default();
    let running = {
        let reached = reached.clone();
        step_task_with(
            "test.vanishing",
            move |_, runtime, _| {
                let reached = reached.clone();
                async move {
                    reached.resolve();
                    aborted(runtime.signal()).await
                }
            },
            |_, _| async { Err(Error::message("must not run")) },
        )
    };
    let registry = create_registry();
    let registration = add_task(&registry, running.any());
    let (harness, _, _) = open_tasks(
        Arc::new(MemoryStorage::new()),
        vec![],
        TaskOptions {
            registry: Some(registry.clone()),
            ..TaskOptions::default()
        },
    )
    .await;
    let id = create_in(&harness, &running, ()).await;
    harness.resume().unwrap();
    reached.wait().await;
    registration.dispose();
    // An active run means the mark does not orphan directly; the scheduler orphans once the run has ended.
    assert_eq!(
        harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    assert_eq!(
        outcome(&harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "orphaned", "reason": "missing_task" })
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn orphans_a_reopened_abort_marked_task_without_a_definition_once_scheduling_resumes() {
    let storage = Arc::new(ControlledStorage::persistent());
    let (harness, _, _) = open(&storage, vec![versioned(1, "x", None).any()]).await;
    let id = create_in(&harness, &versioned(1, "x", None), ()).await;
    assert_eq!(
        harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    harness.close(&context()).await.unwrap();

    let (harness, _, _) = open(&storage, vec![]).await;
    assert_matches(
        &task_state(harness.get_task(id, &context()).await.unwrap()),
        &json!({ "abortRequested": true, "state": { "status": "pending" } }),
    );
    let idle = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.wait_for_idle(&context()).await })
    };
    flush().await;
    harness.resume().unwrap();
    idle.await.unwrap().unwrap();
    assert_eq!(
        to_json(
            &harness
                .get_task(id, &context())
                .await
                .unwrap()
                .unwrap()
                .state
        ),
        json!({ "status": "terminal", "outcome": { "status": "orphaned", "reason": "missing_task" } })
    );
    harness.close(&context()).await.unwrap();
}

// ─── definition handover ─────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct Handover {
    phase: String,
}

fn handover(phase: &str) -> Handover {
    Handover {
        phase: phase.into(),
    }
}

type HandoverRt = TaskRuntime<(), Handover, JsonValue, ()>;

#[derive(Clone, Default)]
struct Gates {
    a: Option<Deferred>,
    b: Option<Deferred>,
    on_end: Option<Arc<dyn Fn() + Send + Sync>>,
}

fn handover_task(
    label: &str,
    version: u32,
    log: &Log,
    gates: Gates,
    migrate: Option<Arc<dyn Fn() -> ((), Handover) + Send + Sync>>,
    migrate_error: Option<&str>,
) -> Task<(), Handover, JsonValue> {
    let advance = |phase: &'static str, next: &'static str| {
        let (label, log, gates) = (label.to_string(), log.clone(), gates.clone());
        move |_, runtime: HandoverRt, ctx: Context| {
            let (label, log, gates) = (label.clone(), log.clone(), gates.clone());
            async move {
                log.lock().push(format!("{label}:{phase} start"));
                let gate = if phase == "a" { &gates.a } else { &gates.b };
                if let Some(gate) = gate {
                    gate.wait().await;
                }
                runtime
                    .commit(
                        move |_, _| async move { Ok(Some(NextTaskState::running(handover(next)))) },
                        &ctx,
                    )
                    .await?;
                // Leave room for a wrongly dispatched successor before this invocation ends.
                flush().await;
                if let Some(on_end) = &gates.on_end {
                    on_end();
                }
                log.lock().push(format!("{label}:{phase} end"));
                Ok(())
            }
        }
    };
    let (label_c, log_c) = (label.to_string(), log.clone());
    let (label_abort, log_abort) = (label.to_string(), log.clone());
    let definition = TaskDefinition::new("test.handover", version, |_: &()| handover("a"))
        .phase("a", advance("a", "b"))
        .phase("b", advance("b", "c"))
        .phase("c", move |_, runtime, ctx| {
            log_c.lock().push(format!("{label_c}:c"));
            async move {
                runtime
                    .commit(
                        |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                        &ctx,
                    )
                    .await
            }
        })
        .abort(move |_, runtime, ctx| {
            log_abort.lock().push(format!("{label_abort}:abort"));
            let reason = label_abort.clone();
            async move {
                runtime
                    .commit(
                        move |_, _| async move { Ok(Some(NextTaskState::aborted(reason))) },
                        &ctx,
                    )
                    .await
            }
        });
    let definition = match (migrate, migrate_error) {
        (Some(migrate), _) => definition.migrate(move |_, _, _| Ok(migrate())),
        (None, Some(message)) => {
            let message = message.to_string();
            definition.migrate(move |_, _, _| Err(Error::message(message.clone())))
        }
        (None, None) => definition,
    };
    define_task(definition)
}

struct HandoverRun {
    harness: Harness,
    registry: Registry,
    reports: Reports,
    old: Installed,
    id: TaskId,
}

async fn start_handover(log: &Log, gates: Gates, storage: Arc<dyn Storage>) -> HandoverRun {
    let registry = create_registry();
    let old = add_task(
        &registry,
        handover_task("old", 1, log, gates, None, None).any(),
    );
    let (harness, registry, reports) = open_tasks(
        storage,
        vec![],
        TaskOptions {
            registry: Some(registry),
            ..TaskOptions::default()
        },
    )
    .await;
    let id = create_in(
        &harness,
        &handover_task("old", 1, log, Gates::default(), None, None),
        (),
    )
    .await;
    harness.resume().unwrap();
    log_len(log, 1).await;
    HandoverRun {
        harness,
        registry,
        reports,
        old,
        id,
    }
}

fn gates_a(gate: &Deferred) -> Gates {
    Gates {
        a: Some(gate.clone()),
        ..Gates::default()
    }
}

#[tokio::test]
async fn hands_over_at_the_next_phase_boundary_to_a_same_version_replacement_without_overlap() {
    let log = shared::<Vec<String>>();
    let gate = Deferred::default();
    let run = start_handover(&log, gates_a(&gate), Arc::new(MemoryStorage::new())).await;
    // The same extension name replaces the old one in place.
    add_task(
        &run.registry,
        handover_task("new", 1, &log, Gates::default(), None, None).any(),
    );
    gate.resolve();
    run.harness.wait_for_task(run.id, &context()).await.unwrap();
    assert_eq!(
        *log.lock(),
        vec![
            "old:a start",
            "old:a end",
            "new:b start",
            "new:b end",
            "new:c"
        ]
    );
    run.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_the_memos_across_a_handover() {
    #[derive(Serialize, Deserialize)]
    struct Memoed {
        phase: String,
    }
    let log = shared::<Vec<String>>();
    let gate = Deferred::default();
    let reached = Deferred::default();
    let memo_task = |label: &str| {
        let (label_a, label_b) = (label.to_string(), label.to_string());
        let (gate, reached, log) = (gate.clone(), reached.clone(), log.clone());
        define_task(
            TaskDefinition::<(), Memoed, String>::new("test.handover-memo", 1, |_| Memoed {
                phase: "a".into(),
            })
            .phase("a", move |_, runtime, ctx| {
                let (label, gate, reached) = (label_a.clone(), gate.clone(), reached.clone());
                async move {
                    runtime.memo_with("picked", json!(label), &ctx).await?;
                    reached.resolve();
                    gate.wait().await;
                    runtime
                        .commit(
                            |_, _| async {
                                Ok(Some(NextTaskState::running(Memoed { phase: "b".into() })))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .phase("b", move |_, runtime, ctx| {
                let (label, log) = (label_b.clone(), log.clone());
                async move {
                    // The candidate loses to the memo the old definition stored.
                    let picked = runtime.memo_with("picked", json!(label), &ctx).await?;
                    log.lock()
                        .push(format!("{label}:b {}", picked.as_str().unwrap()));
                    runtime
                        .commit(
                            move |_, _| async move { Ok(Some(NextTaskState::completed(picked))) },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(|_, _, _| async { Ok(()) }),
        )
    };
    let registry = create_registry();
    add_task(&registry, memo_task("old").any());
    let (harness, _, _) = open_tasks(
        Arc::new(MemoryStorage::new()),
        vec![],
        TaskOptions {
            registry: Some(registry.clone()),
            ..TaskOptions::default()
        },
    )
    .await;
    let id = create_in(&harness, &memo_task("old"), ()).await;
    harness.resume().unwrap();
    reached.wait().await;
    add_task(&registry, memo_task("new").any());
    gate.resolve();
    let receipt = harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(
        outcome(&receipt),
        json!({ "status": "completed", "result": "old" })
    );
    assert_eq!(*log.lock(), vec!["new:b old"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn hands_over_to_a_newer_version_with_a_migration() {
    let log = shared::<Vec<String>>();
    let gate = Deferred::default();
    let run = start_handover(&log, gates_a(&gate), Arc::new(MemoryStorage::new())).await;
    // The same extension name replaces the old one in place.
    add_task(
        &run.registry,
        handover_task(
            "v2",
            2,
            &log,
            Gates::default(),
            Some(Arc::new(|| ((), handover("c")))),
            None,
        )
        .any(),
    );
    gate.resolve();
    let receipt = run.harness.wait_for_task(run.id, &context()).await.unwrap();
    assert_eq!(receipt.version, 2);
    assert_eq!(*log.lock(), vec!["old:a start", "old:a end", "v2:c"]);
    run.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn hands_over_to_a_newer_version_whose_migration_fails_and_leaves_the_task_blocked() {
    let log = shared::<Vec<String>>();
    let gate = Deferred::default();
    let run = start_handover(&log, gates_a(&gate), Arc::new(MemoryStorage::new())).await;
    // The same extension name replaces the old one in place.
    add_task(
        &run.registry,
        handover_task(
            "broken",
            2,
            &log,
            Gates::default(),
            None,
            Some("broken migration"),
        )
        .any(),
    );
    gate.resolve();
    eventually(|| {
        let reports = run.reports.clone();
        async move { reports.len() == 1 }
    })
    .await;
    flush().await;
    assert_matches(
        &task_state(run.harness.get_task(run.id, &context()).await.unwrap()),
        &json!({ "version": 1, "state": { "status": "pending", "checkpoint": { "phase": "b" } } }),
    );
    assert_eq!(*log.lock(), vec!["old:a start", "old:a end"]);
    run.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_running_under_the_old_definition_when_the_replacement_is_missing_or_cannot_take_the_task()
 {
    let log = shared::<Vec<String>>();
    let gate_a = Deferred::default();
    let gate_b = Deferred::default();
    let run = start_handover(
        &log,
        Gates {
            a: Some(gate_a.clone()),
            b: Some(gate_b.clone()),
            on_end: None,
        },
        Arc::new(MemoryStorage::new()),
    )
    .await;
    run.old.dispose();
    gate_a.resolve();
    eventually(|| {
        let log = log.clone();
        async move { log.lock().iter().any(|line| line == "old:b start") }
    })
    .await;
    // A newer definition without a migration cannot take the task either.
    add_task(
        &run.registry,
        handover_task("incompatible", 2, &log, Gates::default(), None, None).any(),
    );
    gate_b.resolve();
    run.harness.wait_for_task(run.id, &context()).await.unwrap();
    assert_eq!(
        *log.lock(),
        vec![
            "old:a start",
            "old:a end",
            "old:b start",
            "old:b end",
            "old:c"
        ]
    );
    let causes: Vec<String> = run
        .reports
        .errors()
        .iter()
        .map(|error| match error {
            Error::Message {
                cause: Some(cause), ..
            } => cause.to_string(),
            other => format!("no cause: {other}"),
        })
        .collect();
    assert_eq!(causes, vec!["missing_task", "incompatible_task"]);
    run.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_a_runtime_commit_of_the_old_invocation_queued_behind_its_handover_commit() {
    let log = shared::<Vec<String>>();
    let gate = Deferred::default();
    let storage = Arc::new(ControlledStorage::new());
    let held = shared::<Option<Arc<crate::durable::session::tests::support::Gate>>>();
    let old_runtime = shared::<Option<HandoverRt>>();
    let harness_ref = shared::<Option<Harness>>();
    let registry = create_registry();
    let old = {
        let (gate, storage, held, old_runtime, harness_ref) = (
            gate.clone(),
            storage.clone(),
            held.clone(),
            old_runtime.clone(),
            harness_ref.clone(),
        );
        define_task(
            TaskDefinition::<(), Handover, JsonValue>::new("test.handover", 1, |_| handover("a"))
                .phase("a", move |task, runtime, ctx| {
                    *old_runtime.lock() = Some(runtime.clone());
                    let (gate, storage, held, harness_ref) = (
                        gate.clone(),
                        storage.clone(),
                        held.clone(),
                        harness_ref.clone(),
                    );
                    async move {
                        gate.wait().await;
                        runtime
                            .commit(
                                |_, _| async { Ok(Some(NextTaskState::running(handover("b")))) },
                                &ctx,
                            )
                            .await?;
                        // Hold the line with an unrelated commit so the handover commit queues behind it.
                        *held.lock() = Some(Arc::new(storage.hold_commits()));
                        let harness = harness_ref.lock().clone().unwrap();
                        let conversation_id = task.conversation_id;
                        tokio::spawn(async move {
                            let _ = harness
                                .commit(
                                    move |tx| async move {
                                        tx.append_entry(
                                            conversation_id,
                                            EntryDraft::new("blocker"),
                                        )
                                        .await?;
                                        Ok(())
                                    },
                                    &context(),
                                )
                                .await;
                        });
                        super::tasks::settle_spawned().await;
                        Ok(())
                    }
                })
                .phase("b", |_, _, _| async { Ok(()) })
                .phase("c", |_, _, _| async { Ok(()) })
                .abort(|_, _, _| async { Ok(()) }),
        )
    };
    add_task(&registry, old.any());
    let (harness, _, _) = open_tasks(
        storage.clone(),
        vec![],
        TaskOptions {
            registry: Some(registry.clone()),
            ..TaskOptions::default()
        },
    )
    .await;
    *harness_ref.lock() = Some(harness.clone());
    let id = create_in(&harness, &old, ()).await;
    harness.resume().unwrap();
    eventually(|| {
        let old_runtime = old_runtime.clone();
        async move { old_runtime.lock().is_some() }
    })
    .await;
    // The same extension name replaces the old one in place.
    add_task(
        &registry,
        handover_task("new", 1, &log, Gates::default(), None, None).any(),
    );
    gate.resolve();
    eventually(|| {
        let held = held.clone();
        async move { held.lock().is_some() }
    })
    .await;
    let gate = held.lock().clone().unwrap();
    gate.entered().await;
    flush().await;
    // Queued behind the handover commit while the invocation has not ended yet.
    let late = {
        let runtime = old_runtime.lock().clone().unwrap();
        tokio::spawn(async move {
            runtime
                .commit(
                    |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                    &context(),
                )
                .await
        })
    };
    flush().await;
    gate.release();
    assert_err(late.await.unwrap(), "invocation has ended");
    harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(*log.lock(), vec!["new:b start", "new:b end", "new:c"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn preserves_an_abort_mark_that_races_the_handover_commit_the_new_definition_aborts() {
    let log = shared::<Vec<String>>();
    let gate = Deferred::default();
    let storage = Arc::new(ControlledStorage::new());
    let held = shared::<Option<Arc<crate::durable::session::tests::support::Gate>>>();
    // The progress commit landed; hold the next commit, the handover, and queue the abort mark behind it.
    let gates = {
        let (storage, held) = (storage.clone(), held.clone());
        Gates {
            a: Some(gate.clone()),
            b: None,
            on_end: Some(Arc::new(move || {
                let mut held = held.lock();
                if held.is_none() {
                    *held = Some(Arc::new(storage.hold_commits()));
                }
            })),
        }
    };
    let run = start_handover(&log, gates, storage.clone()).await;
    // The same extension name replaces the old one in place.
    add_task(
        &run.registry,
        handover_task("new", 1, &log, Gates::default(), None, None).any(),
    );
    gate.resolve();
    eventually(|| {
        let held = held.clone();
        async move { held.lock().is_some() }
    })
    .await;
    let held_gate = held.lock().clone().unwrap();
    held_gate.entered().await;
    let id = run.id;
    let aborting = {
        let harness = run.harness.clone();
        tokio::spawn(async move { harness.abort_task(id, &context()).await })
    };
    flush().await;
    held_gate.release();
    assert_eq!(aborting.await.unwrap().unwrap(), AbortTaskResult::Marked);
    assert_eq!(
        outcome(&run.harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "new" })
    );
    let states: Vec<String> = storage
        .commits
        .lock()
        .iter()
        .flatten()
        .filter_map(|write| match write {
            StorageWrite::Task { value } if value.id == id => Some(format!(
                "{}{}",
                value.state.status(),
                if value.abort_requested { "+mark" } else { "" }
            )),
            _ => None,
        })
        .collect();
    // created, reserved, progress, handover, mark, abort reservation, aborted
    assert_eq!(
        states,
        vec![
            "pending",
            "running",
            "running",
            "pending",
            "pending+mark",
            "running+mark",
            "terminal+mark",
        ]
    );
    assert_eq!(*log.lock(), vec!["old:a start", "old:a end", "new:abort"]);
    run.harness.close(&context()).await.unwrap();
}

// ─── Harness open ────────────────────────────────────────────────────────────

#[tokio::test]
async fn releases_its_registry_subscription_and_closes_the_session_when_open_fails() {
    let storage = Arc::new(ControlledStorage::new());
    let log = shared::<Vec<String>>();
    let stuck = {
        let log = log.clone();
        step_task_with(
            "test.stuck",
            move |_, _, _| {
                log.lock().push("run".into());
                never()
            },
            |_, _| async { Ok(()) },
        )
    };
    // Leave a running task behind so open has a reconciliation commit to fail.
    let (first, _, _) = open(&storage, vec![stuck.any()]).await;
    create_in(&first, &stuck, ()).await;
    first.resume().unwrap();
    log_len(&log, 1).await;

    let reader = CountingReader::new(create_registry());
    storage.fail_next_commit(Error::message("disk full"));
    let opened = Harness::open(
        storage.clone(),
        HarnessOptions::new(Models::default(), reader.clone()),
        &context(),
    )
    .await;
    assert_err(opened.map(|_| ()), "disk full");
    assert_eq!(reader.subscriptions(), 0);
    assert!(!storage_open(&(storage.clone() as Arc<dyn Storage>)).await);
}
