//! Port of `test/harness-tasks.test.ts` (task phases, runtime, scheduling, abort, and close).
//!
//! Divergences: `Agent` values are cloned per `runtime.agent()` call, so the "same agent within a phase" identity check
//! compares the resolved values instead. Skipped: the throwing settings getter case, since Rust settings are a plain
//! value returned by an infallible callback.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use futures::FutureExt;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use super::support::*;
use crate::chord::{AbortController, AbortReason, Context};
use crate::durable::documents::define_doc;
use crate::durable::entries::define_entry;
use crate::durable::errors::{Error, Result, StorageRejected};
use crate::durable::harness::registry::{Registry, RegistryReader, RegistrySnapshot};
use crate::durable::harness::scheduler::{AbortTaskResult, TaskRuntime};
use crate::durable::harness::types::{AgentChange, ConversationCreateOptions, HarnessOptions};
use crate::durable::harness::{Conversation, CreateOptions, Harness, create_registry, hook};
use crate::durable::ids::TaskId;
use crate::durable::session::Unsubscribe;
use crate::durable::session::tests::support::{ControlledStorage, Deferred, flush};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{AnyTask, NextTaskState, RunningTask, TaskDefinition, define_task};
use crate::durable::types::{
    DocDefinition, EntryDraft, JoinPolicy, RewindableConversation, RewindableFork, SessionScope,
    Storage, StorageWrite, TaskRecord, TaskScope, WatchEnd,
};
use crate::models::Models;
use crate::types::ModelThinkingLevel;

pub(super) type Rt = TaskRuntime<(), JsonValue, JsonValue, ()>;

pub(super) fn owned() -> crate::durable::types::TaskOptions {
    crate::durable::types::TaskOptions::conversation(None)
}

/// A one-phase task with an abort handler.
pub(super) fn step_with<F, Fut, A, AFut>(name: &str, run: F, abort: A) -> StepTask
where
    F: Fn(TaskId, Rt, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
    A: Fn(Rt, Context) -> AFut + Send + Sync + 'static,
    AFut: Future<Output = Result<()>> + Send + 'static,
{
    define_task(
        TaskDefinition::new(name, 1, |_: &()| run_phase())
            .phase(
                "run",
                move |task: RunningTask<(), JsonValue>, runtime, ctx| run(task.id, runtime, ctx),
            )
            .abort(move |_, runtime, ctx| abort(runtime, ctx)),
    )
}

/// A one-phase task; the default abort handler settles `aborted` with reason "test".
pub(super) fn step<F, Fut>(name: &str, run: F) -> StepTask
where
    F: Fn(TaskId, Rt, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    step_with(name, run, |runtime, ctx| async move {
        abort_with(&runtime, "test", &ctx).await
    })
}

pub(super) async fn complete(runtime: &Rt, result: JsonValue, ctx: &Context) -> Result<()> {
    runtime
        .commit(
            move |_, _| async move { Ok(Some(NextTaskState::completed(result))) },
            ctx,
        )
        .await
}

pub(super) async fn abort_with(runtime: &Rt, reason: &str, ctx: &Context) -> Result<()> {
    let reason = reason.to_string();
    runtime
        .commit(
            move |_, _| async move { Ok(Some(NextTaskState::aborted(reason))) },
            ctx,
        )
        .await
}

pub(super) async fn no_state(runtime: &Rt, ctx: &Context) -> Result<()> {
    runtime.commit(|_, _| async { Ok(None) }, ctx).await
}

/// A one-phase task that waits for `gate` and completes with null.
pub(super) fn gated(name: &str, gate: Deferred) -> StepTask {
    step(name, move |_, runtime, ctx| {
        let gate = gate.clone();
        async move {
            gate.wait().await;
            complete(&runtime, json!(null), &ctx).await
        }
    })
}

pub(super) struct Opened {
    pub harness: Harness,
    pub registry: Registry,
    pub reports: Reports,
    pub root: Conversation,
}

pub(super) async fn open_root(tasks: Vec<AnyTask>, options: TaskOptions) -> Opened {
    open_root_in(Arc::new(MemoryStorage::new()), tasks, options).await
}

pub(super) async fn open_root_in(
    storage: Arc<dyn Storage>,
    tasks: Vec<AnyTask>,
    options: TaskOptions,
) -> Opened {
    let (harness, registry, reports) = open_tasks(storage, tasks, options).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    Opened {
        harness,
        registry,
        reports,
        root,
    }
}

/// Start `abort_task()` and resolve once its mark is durable, before it has joined the run.
pub(super) async fn mark_durably(
    harness: &Harness,
    id: TaskId,
) -> tokio::task::JoinHandle<Result<AbortTaskResult>> {
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
    aborting
}

pub(super) fn outcome(record: &TaskRecord) -> JsonValue {
    to_json(&record.state)["outcome"].clone()
}

pub(super) fn status(record: Option<TaskRecord>) -> String {
    record
        .map(|record| record.state.status().to_string())
        .unwrap_or_default()
}

pub(super) fn faulted(message: &str) -> JsonValue {
    json!({ "status": "faulted", "error": { "message": message } })
}

pub(super) fn shared<T: Default>() -> Arc<Mutex<T>> {
    Arc::new(Mutex::new(T::default()))
}

pub(super) fn spawn_commit_blocker(
    harness: &Harness,
    conversation_id: crate::durable::ids::ConversationId,
) {
    let harness = harness.clone();
    tokio::spawn(async move {
        let _ = harness
            .commit(
                move |tx| async move {
                    tx.append_entry(conversation_id, EntryDraft::new("blocker"))
                        .await?;
                    Ok(())
                },
                &context(),
            )
            .await;
    });
}

/// Let spawned tasks run until they block, as TS microtasks queued by a handler run before it returns.
pub(super) async fn settle_spawned() {
    for _ in 0..16 {
        tokio::task::yield_now().await;
    }
}

pub(super) fn task_statuses(storage: &ControlledStorage, id: TaskId) -> Vec<String> {
    storage
        .commits
        .lock()
        .iter()
        .flatten()
        .filter_map(|write| match write {
            StorageWrite::Task { value } if value.id == id => {
                Some(value.state.status().to_string())
            }
            _ => None,
        })
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Notes {
    text: String,
}

// ─── task phases ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn continues_one_invocation_through_checkpoint_progress_and_completes_with_a_typed_result() {
    #[derive(Serialize, Deserialize)]
    struct Input {
        to: u32,
    }
    #[derive(Serialize, Deserialize)]
    struct Count {
        phase: String,
        n: u32,
    }
    let seen = shared::<Vec<u32>>();
    let runtimes = shared::<Vec<Arc<crate::durable::harness::scheduler::RuntimeCore>>>();
    let counter = {
        let (seen, runtimes) = (seen.clone(), runtimes.clone());
        define_task(
            TaskDefinition::<Input, Count, u32>::new("test.counter", 1, |_| Count {
                phase: "count".into(),
                n: 0,
            })
            .phase("count", move |task, runtime, ctx| {
                seen.lock().push(task.checkpoint.n);
                runtimes.lock().push(runtime.core().clone());
                async move {
                    runtime
                        .commit(
                            |_, current| async move {
                                let n = current.checkpoint.n + 1;
                                Ok(Some(if n == current.input.to {
                                    NextTaskState::completed(n)
                                } else {
                                    NextTaskState::running(Count {
                                        phase: "count".into(),
                                        n,
                                    })
                                }))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(|_, _, _| async { Ok(()) }),
        )
    };
    let opened = open_root(vec![counter.any()], TaskOptions::default()).await;
    let id = {
        let counter = counter.clone();
        opened
            .root
            .commit(
                move |tx| async move { tx.create_task(&counter, Input { to: 3 }, owned()).await },
                &context(),
            )
            .await
            .unwrap()
    };
    opened.harness.resume().unwrap();
    let receipt = opened.harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(
        outcome(&receipt),
        json!({ "status": "completed", "result": 3 })
    );
    assert_eq!(*seen.lock(), vec![0, 1, 2]);
    {
        let runtimes = runtimes.lock();
        assert!(runtimes.iter().all(|core| Arc::ptr_eq(core, &runtimes[0])));
    }
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn faults_a_phase_without_durable_progress_and_a_throwing_phase() {
    let idle = step("test.idle", |_, _, _| async { Ok(()) });
    let document_only = step("test.document-only", |_, runtime, ctx| async move {
        // A commit that returns no state is not progress.
        no_state(&runtime, &ctx).await
    });
    let throws = step("test.throws", |_, _, _| async {
        Err(Error::message("boom"))
    });
    let opened = open_root(
        vec![idle.any(), document_only.any(), throws.any()],
        TaskOptions::default(),
    )
    .await;
    let ids = [
        start(&opened.root, &idle, (), false).await,
        start(&opened.root, &document_only, (), false).await,
        start(&opened.root, &throws, (), false).await,
    ];
    opened.harness.resume().unwrap();
    let mut outcomes = Vec::new();
    for id in ids {
        outcomes.push(outcome(
            &opened.harness.wait_for_task(id, &context()).await.unwrap(),
        ));
    }
    assert_eq!(
        outcomes,
        vec![
            faulted("Task test.idle phase run returned without durable progress"),
            faulted("Task test.document-only phase run returned without durable progress"),
            faulted("boom"),
        ]
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_a_committed_terminal_outcome_when_the_handler_throws_afterwards() {
    let late_commit = shared::<Option<String>>();
    let done = {
        let late_commit = late_commit.clone();
        step("test.done", move |_, runtime, ctx| {
            let late_commit = late_commit.clone();
            async move {
                complete(&runtime, json!("ok"), &ctx).await?;
                let error = no_state(&runtime, &ctx).await.unwrap_err();
                *late_commit.lock() = Some(error.to_string());
                Err(Error::message("after terminal"))
            }
        })
    };
    let opened = open_root(vec![done.any()], TaskOptions::default()).await;
    let id = start(&opened.root, &done, (), false).await;
    opened.harness.resume().unwrap();
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "completed", "result": "ok" })
    );
    eventually(|| {
        let late_commit = late_commit.clone();
        async move { late_commit.lock().is_some() }
    })
    .await;
    assert!(
        late_commit
            .lock()
            .as_ref()
            .unwrap()
            .contains(&format!("Task {id} is terminal"))
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn compares_checkpoints_by_value_including_arrays() {
    #[derive(Serialize, Deserialize)]
    struct Collect {
        phase: String,
        items: Vec<String>,
    }
    let collect = define_task(
        TaskDefinition::<(), Collect, ()>::new("test.collect", 1, |_| Collect {
            phase: "collect".into(),
            items: Vec::new(),
        })
        .phase("collect", |task, runtime, ctx| async move {
            let mut items = task.checkpoint.items;
            // Two rounds of progress, then an equal copy of the checkpoint, which is no progress.
            if items.len() < 2 {
                items.push(format!("item{}", items.len()));
            }
            runtime
                .commit(
                    move |_, _| async move {
                        Ok(Some(NextTaskState::running(Collect {
                            phase: "collect".into(),
                            items,
                        })))
                    },
                    &ctx,
                )
                .await
        })
        .abort(|_, _, _| async { Ok(()) }),
    );
    let opened = open_root(vec![collect.any()], TaskOptions::default()).await;
    let id = {
        let collect = collect.clone();
        opened
            .root
            .commit(
                move |tx| async move { tx.create_task(&collect, (), owned()).await },
                &context(),
            )
            .await
            .unwrap()
    };
    opened.harness.resume().unwrap();
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        faulted("Task test.collect phase collect returned without durable progress")
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn commits_results_with_entries_atomically_keeps_memos_until_terminal_and_retires_task_documents()
 {
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    struct Progress {
        lines: Vec<String>,
    }
    #[derive(Serialize, Deserialize)]
    struct Phase {
        phase: String,
    }
    #[derive(Serialize, Deserialize)]
    struct Answer {
        #[serde(rename = "entryId")]
        entry_id: crate::durable::ids::EntryId,
    }
    let progress = define_doc(DocDefinition::new(
        "test.task-progress",
        1,
        TaskScope,
        Progress::default,
    ))
    .unwrap();
    let answer = define_entry::<()>("answer").unwrap();
    let child = step("test.child", |_, runtime, ctx| async move {
        complete(&runtime, json!(null), &ctx).await
    });
    let child_id = shared::<Option<TaskId>>();
    let progress_seen = shared::<Option<Vec<String>>>();
    let writer = {
        let (progress_w, progress_a) = (progress.clone(), progress.clone());
        let child = child.clone();
        let child_id = child_id.clone();
        let progress_seen = progress_seen.clone();
        define_task(
            TaskDefinition::<(), Phase, Answer>::new("test.writer", 1, |_| Phase {
                phase: "write".into(),
            })
            .phase("write", move |task, runtime, ctx| {
                let progress = progress_w.clone();
                let child = child.clone();
                let child_id = child_id.clone();
                async move {
                    assert_eq!(runtime.memo::<JsonValue>("choice").await?, None);
                    let (a, b) = futures::join!(
                        runtime.memo_with("choice", json!("a"), &ctx),
                        runtime.memo_with("choice", json!("b"), &ctx)
                    );
                    assert_eq!((a?, b?), (json!("a"), json!("a")));
                    assert_eq!(runtime.memo::<JsonValue>("choice").await?, Some(json!("a")));
                    // Memo names never resolve to inherited object properties.
                    assert_eq!(runtime.memo::<JsonValue>("toString").await?, None);
                    assert_eq!(
                        runtime.memo_with("toString", json!("own"), &ctx).await?,
                        json!("own")
                    );
                    let task_id = task.id.erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                tx.doc(&progress, task_id)
                                    .await?
                                    .edit(|value| value.lines.push("wrote".into()))?;
                                // Task creation defaults to the task's own conversation.
                                let id = tx.create_task(&child, (), owned()).await?;
                                *child_id.lock() = Some(id.erase());
                                Ok(Some(NextTaskState::running(Phase {
                                    phase: "answer".into(),
                                })))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .phase("answer", move |task, runtime, ctx| {
                let progress = progress_a.clone();
                let progress_seen = progress_seen.clone();
                async move {
                    assert_eq!(
                        task.memos.as_ref().map(to_json),
                        Some(json!({ "choice": "a", "toString": "own" }))
                    );
                    let task_id = task.id.erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let lines = tx.doc(&progress, task_id).await?.get()?.lines.clone();
                                *progress_seen.lock() = Some(lines);
                                Ok(None)
                            },
                            &ctx,
                        )
                        .await?;
                    runtime
                        .commit(
                            |tx, current| async move {
                                let mut draft = EntryDraft::new("answer");
                                draft.model = Some(vec![user("done")]);
                                let entry = tx.append_entry(current.conversation_id, draft).await?;
                                Ok(Some(NextTaskState::completed(Answer {
                                    entry_id: entry.id,
                                })))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(|_, _, _| async { Ok(()) }),
        )
    };
    let opened = open_root(vec![writer.any(), child.any()], TaskOptions::default()).await;
    let conversation = opened
        .harness
        .create_conversation(ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    let id = {
        let writer = writer.clone();
        conversation
            .commit(
                move |tx| async move { tx.create_task(&writer, (), owned()).await },
                &context(),
            )
            .await
            .unwrap()
    };
    opened.harness.resume().unwrap();
    let receipt = opened.harness.wait_for_task(id, &context()).await.unwrap();
    assert!(to_json(&receipt.state).get("checkpoint").is_none());
    assert!(receipt.memos.is_none());
    let result = outcome(&receipt);
    assert_eq!(result["status"], "completed");
    let entry_id: crate::durable::ids::EntryId =
        serde_json::from_value(result["result"]["entryId"].clone()).unwrap();
    let entry = opened
        .harness
        .commit(
            move |tx| async move { tx.entry(entry_id).await },
            &context(),
        )
        .await
        .unwrap();
    assert!(answer.is(entry.as_ref()));
    assert_eq!(entry.unwrap().conversation_id, conversation.id);
    assert_eq!(*progress_seen.lock(), Some(vec!["wrote".to_string()]));
    assert_eq!(
        opened
            .harness
            .snapshot(&progress, id.erase(), &context())
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        opened.harness.get_task(id, &context()).await.unwrap(),
        Some(receipt)
    );
    let child_id = child_id.lock().unwrap();
    assert_eq!(
        opened
            .harness
            .wait_for_task(child_id, &context())
            .await
            .unwrap()
            .conversation_id,
        conversation.id
    );
    assert_ne!(opened.root.id, conversation.id);
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn resumes_a_waiting_task_once_every_task_in_on_is_terminal_whatever_the_outcome() {
    #[derive(Serialize, Deserialize)]
    struct Phase {
        phase: String,
    }
    let gate = Deferred::default();
    let order = shared::<Vec<String>>();
    let on = shared::<Vec<TaskId>>();
    let outcomes = shared::<Vec<String>>();
    let waiter = {
        let (order_w, order_r) = (order.clone(), order.clone());
        let (on_w, on_r) = (on.clone(), on.clone());
        let outcomes = outcomes.clone();
        define_task(
            TaskDefinition::<(), Phase, ()>::new("test.waiter", 1, |_| Phase {
                phase: "wait".into(),
            })
            .phase("wait", move |_, runtime, ctx| {
                order_w.lock().push("wait".into());
                let on = on_w.lock().clone();
                async move {
                    runtime
                        .commit(
                            move |_, _| async move {
                                Ok(Some(NextTaskState::waiting(
                                    Phase {
                                        phase: "resume".into(),
                                    },
                                    on,
                                    JoinPolicy::AllSettled,
                                )))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .phase("resume", move |_, runtime, ctx| {
                order_r.lock().push("resume".into());
                let on = on_r.lock().clone();
                let outcomes = outcomes.clone();
                async move {
                    *outcomes.lock() = runtime
                        .outcomes(&on, &ctx)
                        .await?
                        .iter()
                        .map(|outcome| to_json(outcome)["status"].as_str().unwrap().to_string())
                        .collect();
                    runtime
                        .commit(
                            |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(|_, runtime, ctx| async move {
                runtime
                    .commit(
                        |_, _| async { Ok(Some(NextTaskState::aborted("test"))) },
                        &ctx,
                    )
                    .await
            }),
        )
    };
    let first = {
        let (order, gate) = (order.clone(), gate.clone());
        step("test.first", move |_, runtime, ctx| {
            order.lock().push("first".into());
            let gate = gate.clone();
            async move {
                gate.wait().await;
                complete(&runtime, json!(null), &ctx).await
            }
        })
    };
    let faulting = {
        let order = order.clone();
        step("test.faulting", move |_, _, _| {
            order.lock().push("faulting".into());
            async { Err(Error::message("fails")) }
        })
    };
    let opened = open_root(
        vec![first.any(), faulting.any(), waiter.any()],
        TaskOptions::default(),
    )
    .await;
    let first_id = start(&opened.root, &first, (), false).await;
    let faulting_id = start(&opened.root, &faulting, (), false).await;
    *on.lock() = vec![first_id, faulting_id];
    let waiter_id = {
        let waiter = waiter.clone();
        opened
            .root
            .commit(
                move |tx| async move { tx.create_task(&waiter, (), owned()).await },
                &context(),
            )
            .await
            .unwrap()
    };
    opened.harness.resume().unwrap();
    eventually(|| {
        let order = order.clone();
        async move { order.lock().len() == 3 }
    })
    .await;
    flush().await;
    let mut sorted = order.lock().clone();
    sorted.sort();
    assert_eq!(sorted, vec!["faulting", "first", "wait"]);
    let state = opened
        .harness
        .get_task(waiter_id, &context())
        .await
        .unwrap()
        .unwrap()
        .state;
    assert_matches(
        &to_json(&state),
        &json!({ "status": "waiting", "on": [first_id, faulting_id] }),
    );
    gate.resolve();
    opened
        .harness
        .wait_for_task(waiter_id, &context())
        .await
        .unwrap();
    assert_eq!(order.lock().last().unwrap(), "resume");
    assert_eq!(*outcomes.lock(), vec!["completed", "faulted"]);
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_one_registry_snapshot_per_phase_and_refreshes_it_at_the_phase_boundary() {
    #[derive(Serialize, Deserialize)]
    struct Phase {
        phase: String,
    }
    let seen = shared::<Vec<(String, Vec<String>, bool)>>();
    let phase_gate = Deferred::default();
    let entered = Deferred::default();
    fn tools(snapshot: &RegistrySnapshot) -> Vec<String> {
        snapshot
            .tools()
            .into_iter()
            .map(|(_, tool)| tool.name.clone())
            .collect()
    }
    let snapshots = {
        let (seen_a, seen_b) = (seen.clone(), seen.clone());
        let (phase_gate, entered) = (phase_gate.clone(), entered.clone());
        define_task(
            TaskDefinition::<(), Phase, ()>::new("test.snapshots", 1, |_| Phase {
                phase: "a".into(),
            })
            .phase("a", move |_, runtime, ctx| {
                let seen = seen_a.clone();
                let (phase_gate, entered) = (phase_gate.clone(), entered.clone());
                async move {
                    let before = runtime.registry();
                    entered.resolve();
                    phase_gate.wait().await;
                    let now = runtime.registry();
                    seen.lock()
                        .push(("a".into(), tools(&now), Arc::ptr_eq(&now, &before)));
                    runtime
                        .commit(
                            |_, _| async {
                                Ok(Some(NextTaskState::running(Phase { phase: "b".into() })))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .phase("b", move |_, runtime, ctx| {
                seen_b
                    .lock()
                    .push(("b".into(), tools(&runtime.registry()), true));
                async move {
                    runtime
                        .commit(
                            |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(|_, _, _| async { Ok(()) }),
        )
    };
    let opened = open_root(vec![snapshots.any()], TaskOptions::default()).await;
    let id = {
        let snapshots = snapshots.clone();
        opened
            .root
            .commit(
                move |tx| async move { tx.create_task(&snapshots, (), owned()).await },
                &context(),
            )
            .await
            .unwrap()
    };
    opened.harness.resume().unwrap();
    entered.wait().await;
    add_tool(&opened.registry, tool_described("late", "late"));
    phase_gate.resolve();
    opened.harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(
        *seen.lock(),
        vec![
            ("a".to_string(), vec![], true),
            ("b".to_string(), vec!["late".to_string()], true),
        ]
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn resolves_the_agent_at_first_use_in_a_phase_from_its_snapshot_keeps_it_for_the_phase_and_anew_at_the_next()
 {
    #[derive(Serialize, Deserialize)]
    struct Phase {
        phase: String,
    }
    #[derive(Clone)]
    struct PingHooks {
        ping: Arc<dyn Fn() + Send + Sync>,
    }
    type PingRt = TaskRuntime<(), Phase, (), PingHooks>;
    let pings = shared::<Vec<String>>();
    let seen = shared::<Vec<JsonValue>>();
    let before_use = Deferred::default();
    let after_use = Deferred::default();
    let waiting_before_use = Deferred::default();
    let waiting_after_use = Deferred::default();
    async fn ping(runtime: &PingRt, pings: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        pings.lock().clear();
        runtime
            .hooks()
            .each(
                |hooks| Some(hooks.ping.clone()),
                |handler| async move {
                    handler();
                    Ok(())
                },
            )
            .await
            .unwrap();
        pings.lock().clone()
    }
    fn thinking(level: ModelThinkingLevel) -> JsonValue {
        to_json(&level)
    }
    let phases = {
        let (pings_a, pings_b) = (pings.clone(), pings.clone());
        let (seen_a, seen_b) = (seen.clone(), seen.clone());
        let (before_use, after_use) = (before_use.clone(), after_use.clone());
        let (waiting_before_use, waiting_after_use) =
            (waiting_before_use.clone(), waiting_after_use.clone());
        define_task(
            TaskDefinition::<(), Phase, (), PingHooks>::new("test.agent-phases", 1, |_| Phase {
                phase: "a".into(),
            })
            .phase("a", move |_, runtime, ctx| {
                let (pings, seen) = (pings_a.clone(), seen_a.clone());
                let (before_use, after_use) = (before_use.clone(), after_use.clone());
                let (waiting_before_use, waiting_after_use) =
                    (waiting_before_use.clone(), waiting_after_use.clone());
                async move {
                    waiting_before_use.resolve();
                    before_use.wait().await;
                    let first = runtime.agent(&ctx).await?;
                    let first_pings = ping(&runtime, &pings).await;
                    seen.lock().push(json!({
                        "phase": "a",
                        "pings": first_pings,
                        "thinking": thinking(first.thinking_level),
                    }));
                    waiting_after_use.resolve();
                    after_use.wait().await;
                    let second = runtime.agent(&ctx).await?;
                    let second_pings = ping(&runtime, &pings).await;
                    seen.lock().push(json!({
                        "phase": "a",
                        "pings": second_pings,
                        "thinking": thinking(second.thinking_level),
                        "same": second.thinking_level == first.thinking_level
                            && second.extensions.len() == first.extensions.len(),
                    }));
                    runtime
                        .commit(
                            |_, _| async {
                                Ok(Some(NextTaskState::running(Phase { phase: "b".into() })))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .phase("b", move |_, runtime, ctx| {
                let (pings, seen) = (pings_b.clone(), seen_b.clone());
                async move {
                    let agent = runtime.agent(&ctx).await?;
                    let pings = ping(&runtime, &pings).await;
                    seen.lock().push(json!({
                        "phase": "b",
                        "pings": pings,
                        "thinking": thinking(agent.thinking_level),
                    }));
                    runtime
                        .commit(
                            |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(|_, _, _| async { Ok(()) }),
        )
    };
    let opened = open_root(vec![phases.any()], TaskOptions::default()).await;
    let push = |label: &'static str| {
        let pings = pings.clone();
        PingHooks {
            ping: Arc::new(move || pings.lock().push(label.into())),
        }
    };
    add_hooks(&opened.registry, hook(&phases, push("before")));
    let id = {
        let phases = phases.clone();
        opened
            .root
            .commit(
                move |tx| async move { tx.create_task(&phases, (), owned()).await },
                &context(),
            )
            .await
            .unwrap()
    };
    opened.harness.resume().unwrap();

    // Configured during the phase but before its first use: the lazy resolution reads it.
    waiting_before_use.wait().await;
    opened
        .root
        .configure(
            AgentChange::default().thinking_level(ModelThinkingLevel::Low),
            &context(),
        )
        .await
        .unwrap();
    before_use.resolve();
    // Configured and installed after the first use: not seen until the next phase.
    waiting_after_use.wait().await;
    add_hooks(&opened.registry, hook(&phases, push("late")));
    opened
        .root
        .configure(
            AgentChange::default().thinking_level(ModelThinkingLevel::High),
            &context(),
        )
        .await
        .unwrap();
    after_use.resolve();

    opened.harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(
        *seen.lock(),
        vec![
            json!({ "phase": "a", "pings": ["before"], "thinking": "low" }),
            json!({ "phase": "a", "pings": ["before"], "thinking": "low", "same": true }),
            json!({ "phase": "b", "pings": ["before", "late"], "thinking": "high" }),
        ]
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_an_agent_wait_whose_caller_context_is_cancelled_without_affecting_the_phases_resolution()
 {
    let cancelled = shared::<Option<String>>();
    let resolved = shared::<Option<JsonValue>>();
    let agent = {
        let (cancelled, resolved) = (cancelled.clone(), resolved.clone());
        step("test.agent-cancel", move |_, runtime, ctx| {
            let (cancelled, resolved) = (cancelled.clone(), resolved.clone());
            async move {
                let caller = AbortController::new();
                caller.abort(Some(AbortReason::message("caller gone")));
                let caller_context = crate::chord::with_abort_signal(caller.signal(), &ctx);
                *cancelled.lock() = Some(
                    runtime
                        .agent(&caller_context)
                        .await
                        .unwrap_err()
                        .to_string(),
                );
                *resolved.lock() = Some(to_json(&runtime.agent(&ctx).await?.thinking_level));
                complete(&runtime, json!(null), &ctx).await
            }
        })
    };
    let opened = open_root(vec![agent.any()], TaskOptions::default()).await;
    let id = start(&opened.root, &agent, (), false).await;
    opened.harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(cancelled.lock().as_deref(), Some("caller gone"));
    assert_eq!(*resolved.lock(), Some(json!("off")));
    opened.harness.close(&context()).await.unwrap();
}

// ─── task runtime ────────────────────────────────────────────────────────────

#[tokio::test]
async fn rejects_runtime_operations_after_the_invocation_ends_and_stops_its_watches() {
    let notes = define_doc(DocDefinition::new(
        "test.task-notes",
        1,
        SessionScope,
        Notes::default,
    ))
    .unwrap();
    let absent_doc = define_doc(DocDefinition::new(
        "test.task-absent",
        1,
        SessionScope,
        Notes::default,
    ))
    .unwrap();
    let captured = shared::<Option<(Rt, Context)>>();
    let watch_closed =
        shared::<Option<futures::future::Shared<futures::future::BoxFuture<'static, WatchEnd>>>>();
    let delivered = shared::<Vec<String>>();
    let absent = shared::<Option<bool>>();
    let watcher = {
        let (notes, absent_doc) = (notes.clone(), absent_doc.clone());
        let (captured, watch_closed, delivered, absent) = (
            captured.clone(),
            watch_closed.clone(),
            delivered.clone(),
            absent.clone(),
        );
        step("test.watcher", move |_, runtime, ctx| {
            let (notes, absent_doc) = (notes.clone(), absent_doc.clone());
            let (captured, watch_closed, delivered, absent) = (
                captured.clone(),
                watch_closed.clone(),
                delivered.clone(),
                absent.clone(),
            );
            async move {
                *captured.lock() = Some((runtime.clone(), ctx.clone()));
                *absent.lock() = Some(runtime.watch_doc(&absent_doc, (), &ctx).await?.is_none());
                let watch = runtime.watch_doc(&notes, (), &ctx).await?.unwrap();
                watch.start(move |value, _, _| {
                    let delivered = delivered.clone();
                    Box::pin(async move {
                        delivered.lock().push(
                            value
                                .map(|value| value["text"].as_str().unwrap_or("").to_string())
                                .unwrap_or_else(|| "retired".into()),
                        );
                        Ok(())
                    })
                })?;
                *watch_closed.lock() = Some(watch.closed());
                complete(&runtime, json!(null), &ctx).await
            }
        })
    };
    let opened = open_root(vec![watcher.any()], TaskOptions::default()).await;
    let set_text = |text: &'static str| {
        let notes = notes.clone();
        move |tx: crate::durable::session::Transaction| async move {
            tx.doc(&notes, ())
                .await?
                .edit(|notes| notes.text = text.into())?;
            Ok(())
        }
    };
    opened
        .harness
        .commit(set_text("hello"), &context())
        .await
        .unwrap();
    let id = start(&opened.root, &watcher, (), false).await;
    opened.harness.resume().unwrap();
    opened.harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(*absent.lock(), Some(true));
    // The watch stops when the step after the phase ends the invocation; later commits deliver nothing.
    let closed = watch_closed.lock().clone().unwrap();
    assert_eq!(closed.await.reason(), "stopped");
    opened
        .harness
        .commit(set_text("after"), &context())
        .await
        .unwrap();
    flush().await;
    assert!(delivered.lock().is_empty());
    let (runtime, handler_context) = captured.lock().clone().unwrap();
    assert_err(no_state(&runtime, &context()).await, "invocation has ended");
    assert_err(runtime.memo::<JsonValue>("x").await, "invocation has ended");
    assert_err(
        runtime.memo_with("x", json!(1), &context()).await,
        "invocation has ended",
    );
    assert_err(runtime.sleep(0, &context()).await, "invocation has ended");
    assert_err(
        runtime.watch_doc(&notes, (), &context()).await,
        "invocation has ended",
    );
    // The handler's own context is cancelled by now; the ended invocation still wins.
    assert_err(
        runtime.agent(&handler_context).await,
        "invocation has ended",
    );
    assert_err(runtime.now(), "invocation has ended");
    assert_err(
        runtime.report(Error::message("late")),
        "invocation has ended",
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reads_committed_documents_and_context_through_the_runtime_and_forwards_the_clock_and_reports()
 {
    let notes = define_doc(DocDefinition::new(
        "test.runtime-notes",
        1,
        RewindableConversation {
            fork: RewindableFork::AsOf,
        },
        Notes::default,
    ))
    .unwrap();
    let captured = shared::<Option<Rt>>();
    let seen = shared::<Vec<JsonValue>>();
    let reader = {
        let (notes, captured, seen) = (notes.clone(), captured.clone(), seen.clone());
        step("test.reader", move |_, runtime, ctx| {
            let (notes, captured, seen) = (notes.clone(), captured.clone(), seen.clone());
            async move {
                *captured.lock() = Some(runtime.clone());
                let conversation_id = runtime.conversation_id();
                let entries = runtime.context(conversation_id, &ctx, None).await?.entries;
                let (first, second) = (entries[0].id, entries[1].id);
                let mut values = Vec::new();
                values.push(json!(
                    runtime
                        .snapshot(&notes, conversation_id, &ctx)
                        .await?
                        .unwrap()
                        .text
                ));
                values.push(json!(
                    runtime
                        .snapshot_as_of(&notes, conversation_id, first, &ctx)
                        .await?
                        .unwrap()
                        .text
                ));
                values.push(json!(
                    runtime
                        .context(conversation_id, &ctx, Some(first))
                        .await?
                        .entries
                        .len()
                ));
                values.push(json!(
                    runtime
                        .context(conversation_id, &ctx, Some(second))
                        .await?
                        .messages
                        .len()
                ));
                values.push(json!(runtime.now()?));
                runtime.report(Error::message("reported"))?;
                seen.lock().extend(values);
                complete(&runtime, json!(null), &ctx).await
            }
        })
    };
    let opened = open_root(
        vec![reader.any()],
        TaskOptions {
            now: Some(Arc::new(|| 1234)),
            ..TaskOptions::default()
        },
    )
    .await;
    let root_id = opened.root.id;
    for text in ["one", "two"] {
        let notes = notes.clone();
        opened
            .root
            .commit(
                move |tx| async move {
                    tx.doc(&notes, root_id)
                        .await?
                        .edit(|notes| notes.text = text.into())?;
                    let mut draft = EntryDraft::new("message");
                    draft.model = Some(vec![user(text)]);
                    tx.append_entry(root_id, draft).await?;
                    Ok(())
                },
                &context(),
            )
            .await
            .unwrap();
    }
    let id = start(&opened.root, &reader, (), false).await;
    opened.harness.resume().unwrap();
    opened.harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(
        *seen.lock(),
        vec![json!("two"), json!("one"), json!(1), json!(2), json!(1234)]
    );
    assert_eq!(opened.reports.messages(), vec!["reported"]);
    // The step after the phase ends the invocation.
    flush().await;
    let runtime = captured.lock().clone().unwrap();
    assert_err(
        runtime.snapshot(&notes, root_id, &context()).await,
        "invocation has ended",
    );
    assert_err(
        runtime.context(root_id, &context(), None).await,
        "invocation has ended",
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn orders_runtime_commits_against_the_step_one_queued_before_it_lands_one_after_it_rejects() {
    let storage = Arc::new(ControlledStorage::new());
    let before = shared::<Option<tokio::task::JoinHandle<Result<()>>>>();
    let after = shared::<Option<tokio::task::JoinHandle<Result<()>>>>();
    let held = shared::<Option<Arc<crate::durable::session::tests::support::Gate>>>();
    let harness_ref = shared::<Option<Harness>>();
    let detached = {
        let (storage, before, after, held, harness_ref) = (
            storage.clone(),
            before.clone(),
            after.clone(),
            held.clone(),
            harness_ref.clone(),
        );
        step("test.detached", move |_, runtime, _| {
            // Hold the line, queue a commit without awaiting it, and return; the step queues behind that commit.
            *held.lock() = Some(Arc::new(storage.hold_commits()));
            let harness = harness_ref.lock().clone().unwrap();
            spawn_commit_blocker(&harness, runtime.conversation_id());
            let first = runtime.clone();
            *before.lock() = Some(tokio::spawn(async move {
                complete(&first, json!("before"), &context()).await
            }));
            // Queued after the handler returned, so after the step.
            let after = after.clone();
            tokio::spawn(async move {
                flush().await;
                *after.lock() = Some(tokio::spawn(async move {
                    complete(&runtime, json!("after"), &context()).await
                }));
            });
            settle_spawned().map(Ok)
        })
    };
    let opened = open_root_in(
        storage.clone(),
        vec![detached.any()],
        TaskOptions::default(),
    )
    .await;
    *harness_ref.lock() = Some(opened.harness.clone());
    let id = start(&opened.root, &detached, (), false).await;
    opened.harness.resume().unwrap();
    eventually(|| {
        let held = held.clone();
        async move { held.lock().is_some() }
    })
    .await;
    let gate = held.lock().clone().unwrap();
    gate.entered().await;
    eventually(|| {
        let after = after.clone();
        async move { after.lock().is_some() }
    })
    .await;
    gate.release();
    let before = before.lock().take().unwrap();
    before.await.unwrap().unwrap();
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "completed", "result": "before" })
    );
    let after = after.lock().take().unwrap();
    let error = after.await.unwrap().unwrap_err().to_string();
    assert!(
        error.contains("invocation has ended") || error.contains("is terminal"),
        "{error}"
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn stops_a_watch_whose_acquisition_finishes_after_the_invocation_ended() {
    let notes = define_doc(DocDefinition::new(
        "test.late-watch",
        1,
        SessionScope,
        Notes::default,
    ))
    .unwrap();
    let storage = Arc::new(ControlledStorage::new());
    let watching = shared::<Option<tokio::task::JoinHandle<String>>>();
    let held = shared::<Option<Arc<crate::durable::session::tests::support::Gate>>>();
    let harness_ref = shared::<Option<Harness>>();
    let late = {
        let (notes, storage, watching, held, harness_ref) = (
            notes.clone(),
            storage.clone(),
            watching.clone(),
            held.clone(),
            harness_ref.clone(),
        );
        step("test.late-watch", move |_, runtime, _| {
            *held.lock() = Some(Arc::new(storage.hold_commits()));
            let harness = harness_ref.lock().clone().unwrap();
            spawn_commit_blocker(&harness, runtime.conversation_id());
            // Starts after the handler returned, so its line job queues behind the step that ends the invocation.
            let (notes, watching) = (notes.clone(), watching.clone());
            tokio::spawn(async move {
                flush().await;
                *watching.lock() = Some(tokio::spawn(async move {
                    match runtime.watch_doc(&notes, (), &context()).await {
                        Ok(_) => "watching".to_string(),
                        Err(error) => error.to_string(),
                    }
                }));
            });
            settle_spawned().map(Ok)
        })
    };
    let opened = open_root_in(storage.clone(), vec![late.any()], TaskOptions::default()).await;
    *harness_ref.lock() = Some(opened.harness.clone());
    {
        let notes = notes.clone();
        opened
            .harness
            .commit(
                move |tx| async move {
                    tx.doc(&notes, ())
                        .await?
                        .edit(|notes| notes.text = "x".into())?;
                    Ok(())
                },
                &context(),
            )
            .await
            .unwrap();
    }
    let id = start(&opened.root, &late, (), false).await;
    opened.harness.resume().unwrap();
    eventually(|| {
        let held = held.clone();
        async move { held.lock().is_some() }
    })
    .await;
    let gate = held.lock().clone().unwrap();
    gate.entered().await;
    eventually(|| {
        let watching = watching.clone();
        async move { watching.lock().is_some() }
    })
    .await;
    gate.release();
    let watching = watching.lock().take().unwrap();
    assert!(watching.await.unwrap().contains("invocation has ended"));
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap())["status"],
        "faulted"
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn sleeps_until_the_harness_clock_reaches_the_deadline_rechecking_after_each_timer() {
    let clock = Arc::new(AtomicUsize::new(1_000));
    let woke = Deferred::default();
    let sleeper = {
        let woke = woke.clone();
        step("test.sleeper", move |_, runtime, ctx| {
            let woke = woke.clone();
            async move {
                runtime.sleep(900, &ctx).await?;
                runtime.sleep(1_005, &ctx).await?;
                woke.resolve();
                complete(&runtime, json!(null), &ctx).await
            }
        })
    };
    let now = {
        let clock = clock.clone();
        Arc::new(move || clock.load(Ordering::SeqCst) as u64)
    };
    let opened = open_root(
        vec![sleeper.any()],
        TaskOptions {
            now: Some(now),
            ..TaskOptions::default()
        },
    )
    .await;
    let id = start(&opened.root, &sleeper, (), false).await;
    opened.harness.resume().unwrap();
    // The clock stands still, so real timers keep firing without waking the task.
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    let (done, _) = settled({
        let woke = woke.clone();
        async move { woke.wait().await }
    })
    .await;
    assert!(!done);
    clock.store(1_005, Ordering::SeqCst);
    opened.harness.wait_for_task(id, &context()).await.unwrap();
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_a_sleep_when_the_invocation_is_signalled_or_the_sleeps_own_context_is_cancelled() {
    let results = shared::<Vec<String>>();
    let sleeping = Deferred::default();
    fn far() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
            + 60_000
    }
    let signalled = {
        let (results, sleeping) = (results.clone(), sleeping.clone());
        step("test.sleep-signalled", move |_, runtime, ctx| {
            let (results, sleeping) = (results.clone(), sleeping.clone());
            async move {
                sleeping.resolve();
                let error = runtime.sleep(far(), &ctx).await.unwrap_err();
                results.lock().push(format!("signalled:{}", error.name()));
                Err(error)
            }
        })
    };
    let cancelled = {
        let results = results.clone();
        step("test.sleep-cancelled", move |_, runtime, ctx| {
            let results = results.clone();
            async move {
                let controller = AbortController::new();
                let sleep_context = signal_context(controller.signal());
                let sleeping = {
                    let runtime = runtime.clone();
                    tokio::spawn(async move { runtime.sleep(far(), &sleep_context).await })
                };
                controller.abort(Some(AbortReason::message("stop sleeping")));
                let error = sleeping.await.unwrap().unwrap_err();
                results.lock().push(format!("cancelled:{error}"));
                complete(&runtime, json!(null), &ctx).await
            }
        })
    };
    let opened = open_root(
        vec![signalled.any(), cancelled.any()],
        TaskOptions::default(),
    )
    .await;
    let signalled_id = start(&opened.root, &signalled, (), false).await;
    let cancelled_id = start(&opened.root, &cancelled, (), false).await;
    opened.harness.resume().unwrap();
    opened
        .harness
        .wait_for_task(cancelled_id, &context())
        .await
        .unwrap();
    sleeping.wait().await;
    opened
        .harness
        .abort_task(signalled_id, &context())
        .await
        .unwrap();
    assert_eq!(
        outcome(
            &opened
                .harness
                .wait_for_task(signalled_id, &context())
                .await
                .unwrap()
        ),
        json!({ "status": "aborted", "reason": "test" })
    );
    assert_eq!(
        *results.lock(),
        vec!["cancelled:stop sleeping", "signalled:AbortError"]
    );
    opened.harness.close(&context()).await.unwrap();
}

// ─── task scheduling ─────────────────────────────────────────────────────────

#[tokio::test]
async fn retries_reservation_on_the_next_wakeup_after_a_rejected_reservation_commit() {
    let runs = Arc::new(AtomicUsize::new(0));
    let once = {
        let runs = runs.clone();
        step("test.once", move |_, runtime, ctx| {
            runs.fetch_add(1, Ordering::SeqCst);
            async move { complete(&runtime, json!(null), &ctx).await }
        })
    };
    let storage = Arc::new(ControlledStorage::new());
    let opened = open_root_in(storage.clone(), vec![once.any()], TaskOptions::default()).await;
    let id = start(&opened.root, &once, (), false).await;
    storage.fail_next_commit(StorageRejected::new("busy").into());
    opened.harness.resume().unwrap();
    eventually(|| {
        let reports = opened.reports.clone();
        async move { reports.len() == 1 }
    })
    .await;
    assert_eq!(
        status(opened.harness.get_task(id, &context()).await.unwrap()),
        "pending"
    );
    // Any wakeup, here a registry change, reserves again.
    add_tool(&opened.registry, tool_described("wake", "wake"));
    opened.harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_a_wakeup_that_arrives_while_a_rejected_reservation_commit_is_in_storage() {
    let first = step("test.wake-first", |_, runtime, ctx| async move {
        complete(&runtime, json!(null), &ctx).await
    });
    let late = step("test.wake-late", |_, runtime, ctx| async move {
        complete(&runtime, json!(null), &ctx).await
    });
    let storage = Arc::new(ControlledStorage::new());
    let opened = open_root_in(storage.clone(), vec![first.any()], TaskOptions::default()).await;
    let first_id = start(&opened.root, &first, (), false).await;
    let late_id = start(&opened.root, &late, (), false).await;
    let held = storage.hold_commits();
    storage.fail_next_commit(StorageRejected::new("busy").into());
    opened.harness.resume().unwrap();
    held.entered().await;
    // Registering the missing definition wakes the scheduler while the doomed reservation is in storage.
    add_task(&opened.registry, late.any());
    held.release();
    opened
        .harness
        .wait_for_task(first_id, &context())
        .await
        .unwrap();
    opened
        .harness
        .wait_for_task(late_id, &context())
        .await
        .unwrap();
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reruns_a_task_whose_fault_write_was_rejected() {
    let runs = Arc::new(AtomicUsize::new(0));
    let storage = Arc::new(ControlledStorage::new());
    let throws = {
        let (runs, storage) = (runs.clone(), storage.clone());
        step("test.rejected-fault", move |_, _, _| {
            // The next commit is the step's fault write.
            if runs.fetch_add(1, Ordering::SeqCst) == 0 {
                storage.fail_next_commit(StorageRejected::new("busy").into());
            }
            async { Err(Error::message("boom")) }
        })
    };
    let opened = open_root_in(storage.clone(), vec![throws.any()], TaskOptions::default()).await;
    let id = start(&opened.root, &throws, (), false).await;
    opened.harness.resume().unwrap();
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        faulted("boom")
    );
    assert_eq!(runs.load(Ordering::SeqCst), 2);
    let reports: Vec<String> = opened
        .reports
        .errors()
        .iter()
        .map(|error| format!("{}: {error}", error.name()))
        .collect();
    assert_eq!(reports, vec!["StorageRejected: busy"]);
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn waits_for_harness_and_conversation_idleness_counting_blocked_work_and_ignoring_background_tasks()
 {
    let gates = shared::<Vec<(TaskId, Deferred)>>();
    let gated_task = {
        let gates = gates.clone();
        step("test.gated", move |id, runtime, ctx| {
            let gate = Deferred::default();
            gates.lock().push((id, gate.clone()));
            async move {
                gate.wait().await;
                complete(&runtime, json!(null), &ctx).await
            }
        })
    };
    let open = |id: TaskId| {
        let gate = gates
            .lock()
            .iter()
            .find(|(task, _)| *task == id)
            .map(|(_, gate)| gate.clone())
            .unwrap();
        gate.resolve();
    };
    let opened = open_root(vec![gated_task.any()], TaskOptions::default()).await;
    let harness = opened.harness.clone();
    harness.wait_for_idle(&context()).await.unwrap();
    let other = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    let foreground = start(&opened.root, &gated_task, (), false).await;
    let background = start(&opened.root, &gated_task, (), true).await;
    let elsewhere = start(&other, &gated_task, (), false).await;
    // Work that has not started yet is live; cancelling a wait only rejects that wait.
    let cancelled = AbortController::new();
    let cancelled_wait = {
        let harness = harness.clone();
        let wait_context = signal_context(cancelled.signal());
        tokio::spawn(async move { harness.wait_for_idle(&wait_context).await })
    };
    flush().await;
    cancelled.abort(Some(AbortReason::message("stop waiting")));
    assert_err(cancelled_wait.await.unwrap(), "stop waiting");
    let aborted = AbortController::new();
    aborted.abort(Some(AbortReason::message("already cancelled")));
    assert_err(
        opened
            .root
            .wait_for_idle(&signal_context(aborted.signal()))
            .await,
        "already cancelled",
    );
    harness.resume().unwrap();
    eventually(|| {
        let gates = gates.clone();
        async move { gates.lock().len() == 3 }
    })
    .await;
    let root_idle = {
        let root = opened.root.clone();
        tokio::spawn(async move { root.wait_for_idle(&context()).await })
    };
    let harness_idle = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.wait_for_idle(&context()).await })
    };
    flush().await;
    open(foreground);
    root_idle.await.unwrap().unwrap();
    flush().await;
    assert!(!harness_idle.is_finished());
    open(elsewhere);
    harness_idle.await.unwrap().unwrap();
    assert_eq!(
        status(harness.get_task(background, &context()).await.unwrap()),
        "running"
    );
    open(background);
    harness.wait_for_task(background, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();
    assert_err(harness.wait_for_idle(&context()).await, "closed");
    assert_err(opened.root.wait_for_idle(&context()).await, "closed");
}

#[tokio::test]
async fn rejects_unknown_tasks_and_reports_terminal_tasks() {
    let done = step("test.quick", |_, runtime, ctx| async move {
        complete(&runtime, json!(null), &ctx).await
    });
    let opened = open_root(vec![done.any()], TaskOptions::default()).await;
    let harness = &opened.harness;
    let unknown: TaskId = TaskId::new(999_999);
    assert_eq!(harness.get_task(unknown, &context()).await.unwrap(), None);
    assert_err(
        harness.wait_for_task(unknown, &context()).await,
        "does not exist",
    );
    assert_err(
        harness.abort_task(unknown, &context()).await,
        "does not exist",
    );
    let id = start(&opened.root, &done, (), false).await;
    harness.resume().unwrap();
    harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(
        status(Some(harness.wait_for_task(id, &context()).await.unwrap())),
        "terminal"
    );
    assert_eq!(
        harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Terminal
    );
    harness.close(&context()).await.unwrap();
    assert_err(harness.resume(), "closed");
}

#[tokio::test]
async fn rejects_task_waits_cancelled_or_closed_while_queued_on_the_line_and_pending_waits_on_close()
 {
    let gate = Deferred::default();
    let blocking = gated("test.wait-close", gate.clone());
    let storage = Arc::new(ControlledStorage::new());
    let opened = open_root_in(
        storage.clone(),
        vec![blocking.any()],
        TaskOptions::default(),
    )
    .await;
    let harness = opened.harness.clone();
    let root_id = opened.root.id;
    let id = start(&opened.root, &blocking, (), false).await;
    let blocker = |harness: &Harness| {
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
    let wait = |harness: &Harness, context: Context| {
        let harness = harness.clone();
        tokio::spawn(async move { harness.wait_for_task(id, &context).await })
    };

    // Hold the line with a commit, then queue waits behind it.
    let held = storage.hold_commits();
    let first_blocker = blocker(&harness);
    held.entered().await;
    let controller = AbortController::new();
    let cancelled_while_queued = wait(&harness, signal_context(controller.signal()));
    flush().await;
    controller.abort(Some(AbortReason::message("wait cancelled")));
    held.release();
    first_blocker.await.unwrap().unwrap();
    assert_err(cancelled_while_queued.await.unwrap(), "wait cancelled");

    let pending = wait(&harness, context());
    let idle = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.wait_for_idle(&context()).await })
    };
    let conversation_idle = {
        let root = opened.root.clone();
        tokio::spawn(async move { root.wait_for_idle(&context()).await })
    };
    flush().await;
    let held_again = storage.hold_commits();
    let second_blocker = blocker(&harness);
    held_again.entered().await;
    let closed_while_queued = wait(&harness, context());
    flush().await;
    let closing = tokio::spawn(harness.close(&context()));
    held_again.release();
    second_blocker.await.unwrap().unwrap();
    assert_err(closed_while_queued.await.unwrap(), "closed");
    assert_err(pending.await.unwrap(), "closed");
    assert_err(idle.await.unwrap(), "closed");
    assert_err(conversation_idle.await.unwrap(), "closed");
    gate.resolve();
    closing.await.unwrap().unwrap();
}

// ─── task abort ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn rejects_the_runs_commits_and_memo_writes_after_the_mark_and_settles_through_a_fresh_abort_invocation()
 {
    let reached = Deferred::default();
    let errors = shared::<Vec<String>>();
    let memo_read = shared::<Option<JsonValue>>();
    let abort_runtimes = shared::<Vec<Arc<crate::durable::harness::scheduler::RuntimeCore>>>();
    let run_runtime = shared::<Option<Arc<crate::durable::harness::scheduler::RuntimeCore>>>();
    let marked = {
        let (reached, errors, memo_read, run_runtime) = (
            reached.clone(),
            errors.clone(),
            memo_read.clone(),
            run_runtime.clone(),
        );
        let abort_runtimes = abort_runtimes.clone();
        step_with(
            "test.marked",
            move |_, runtime, ctx| {
                let (reached, errors, memo_read, run_runtime) = (
                    reached.clone(),
                    errors.clone(),
                    memo_read.clone(),
                    run_runtime.clone(),
                );
                async move {
                    *run_runtime.lock() = Some(runtime.core().clone());
                    runtime.memo_with("kept", json!(1), &ctx).await?;
                    reached.resolve();
                    // Keep working after the signal: every later write of this run must reject.
                    let _ = aborted(runtime.signal()).await;
                    *memo_read.lock() = runtime.memo::<JsonValue>("kept").await?;
                    if let Err(error) = runtime.memo_with("late", json!(2), &context()).await {
                        errors.lock().push(error.to_string());
                    }
                    if let Err(error) = complete(&runtime, json!(null), &context()).await {
                        errors.lock().push(error.to_string());
                    }
                    Ok(())
                }
            },
            move |runtime, ctx| {
                abort_runtimes.lock().push(runtime.core().clone());
                async move {
                    runtime
                        .commit(
                            |_, current| async move {
                                assert!(current.abort_requested);
                                assert_eq!(
                                    current.memos.as_ref().map(to_json),
                                    Some(json!({ "kept": 1 }))
                                );
                                Ok(Some(NextTaskState::aborted("mark")))
                            },
                            &ctx,
                        )
                        .await
                }
            },
        )
    };
    let opened = open_root(vec![marked.any()], TaskOptions::default()).await;
    let id = start(&opened.root, &marked, (), false).await;
    opened.harness.resume().unwrap();
    reached.wait().await;
    assert_eq!(
        opened.harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "mark" })
    );
    assert_eq!(*memo_read.lock(), Some(json!(1)));
    let mark = format!("Task {id} has a durable abort mark");
    assert_eq!(*errors.lock(), vec![mark.clone(), mark]);
    {
        let abort_runtimes = abort_runtimes.lock();
        assert_eq!(abort_runtimes.len(), 1);
        assert!(!Arc::ptr_eq(
            &abort_runtimes[0],
            run_runtime.lock().as_ref().unwrap()
        ));
    }
    opened.harness.close(&context()).await.unwrap();
}

#[derive(Serialize, Deserialize)]
struct TwoPhase {
    phase: String,
}

fn next_phase() -> NextTaskState<TwoPhase> {
    NextTaskState::running(TwoPhase {
        phase: "two".into(),
    })
}

#[tokio::test]
async fn starts_no_further_phase_after_a_mark_that_lands_during_a_phase_with_progress() {
    let reached = Deferred::default();
    let proceed = Deferred::default();
    let phases = shared::<Vec<String>>();
    let two = {
        let (reached, proceed) = (reached.clone(), proceed.clone());
        let (phases_1, phases_2, phases_a) = (phases.clone(), phases.clone(), phases.clone());
        define_task(
            TaskDefinition::<(), TwoPhase, ()>::new("test.mark-boundary", 1, |_| TwoPhase {
                phase: "one".into(),
            })
            .phase("one", move |_, runtime, ctx| {
                phases_1.lock().push("one".into());
                let (reached, proceed) = (reached.clone(), proceed.clone());
                async move {
                    runtime
                        .commit(|_, _| async { Ok(Some(next_phase())) }, &ctx)
                        .await?;
                    reached.resolve();
                    // Ignores the signal and returns normally after the mark.
                    proceed.wait().await;
                    Ok(())
                }
            })
            .phase("two", move |_, _, _| {
                phases_2.lock().push("two".into());
                async { Ok(()) }
            })
            .abort(move |_, runtime, ctx| {
                phases_a.lock().push("abort".into());
                async move {
                    runtime
                        .commit(
                            |_, _| async { Ok(Some(NextTaskState::aborted("boundary"))) },
                            &ctx,
                        )
                        .await
                }
            }),
        )
    };
    let opened = open_root(vec![two.any()], TaskOptions::default()).await;
    let id = {
        let two = two.clone();
        opened
            .root
            .commit(
                move |tx| async move { tx.create_task(&two, (), owned()).await.map(|id| id.erase()) },
                &context(),
            )
            .await
            .unwrap()
    };
    opened.harness.resume().unwrap();
    reached.wait().await;
    let aborting = mark_durably(&opened.harness, id).await;
    proceed.resolve();
    assert_eq!(aborting.await.unwrap().unwrap(), AbortTaskResult::Marked);
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "boundary" })
    );
    assert_eq!(*phases.lock(), vec!["one", "abort"]);
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn signals_and_joins_the_run_before_returning_then_runs_the_abort_handler() {
    let reached = Deferred::default();
    let run_ended = Arc::new(AtomicBool::new(false));
    // Tokio may run the whole abort invocation before the caller resumes, where TS microtask order leaves it pending;
    // the abort handler waits for this gate so the second abort deterministically sees the live mark.
    let settle = Deferred::default();
    let signalled = {
        let (reached, run_ended) = (reached.clone(), run_ended.clone());
        let settle = settle.clone();
        step_with(
            "test.signalled",
            move |_, runtime, _| {
                let (reached, run_ended) = (reached.clone(), run_ended.clone());
                async move {
                    reached.resolve();
                    let result = aborted(runtime.signal()).await;
                    run_ended.store(true, Ordering::SeqCst);
                    result
                }
            },
            move |runtime, ctx| {
                let settle = settle.clone();
                async move {
                    settle.wait().await;
                    abort_with(&runtime, "test", &ctx).await
                }
            },
        )
    };
    let opened = open_root(vec![signalled.any()], TaskOptions::default()).await;
    let id = start(&opened.root, &signalled, (), false).await;
    opened.harness.resume().unwrap();
    reached.wait().await;
    assert_eq!(
        opened.harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    assert!(run_ended.load(Ordering::SeqCst));
    assert_eq!(
        opened.harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    settle.resolve();
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "test" })
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn aborts_waiting_work_before_its_wait_ends_and_faults_abort_handlers_that_throw_or_settle_nothing()
 {
    let gate = Deferred::default();
    let first = gated("test.dependency", gate.clone());
    let first_id = shared::<Option<TaskId>>();
    let wait = {
        let first_id = first_id.clone();
        move |_: TaskId, runtime: Rt, ctx: Context| {
            let on = vec![first_id.lock().unwrap()];
            async move {
                runtime
                    .commit(
                        move |_, _| async move {
                            Ok(Some(NextTaskState::waiting(
                                run_phase(),
                                on,
                                JoinPolicy::AllSettled,
                            )))
                        },
                        &ctx,
                    )
                    .await
            }
        }
    };
    let lazy = step_with("test.lazy-abort", wait.clone(), |_, _| async { Ok(()) });
    let throwing = step_with("test.throwing-abort", wait, |_, _| async {
        Err(Error::message("abort failed"))
    });
    let opened = open_root(
        vec![first.any(), lazy.any(), throwing.any()],
        TaskOptions::default(),
    )
    .await;
    let harness = opened.harness.clone();
    let first_task = start(&opened.root, &first, (), false).await;
    *first_id.lock() = Some(first_task);
    let lazy_id = start(&opened.root, &lazy, (), false).await;
    let throwing_id = start(&opened.root, &throwing, (), false).await;
    harness.resume().unwrap();
    for id in [throwing_id, lazy_id] {
        eventually(|| {
            let harness = harness.clone();
            async move { status(harness.get_task(id, &context()).await.unwrap()) == "waiting" }
        })
        .await;
    }
    harness.abort_task(lazy_id, &context()).await.unwrap();
    harness.abort_task(throwing_id, &context()).await.unwrap();
    assert_eq!(
        outcome(&harness.wait_for_task(lazy_id, &context()).await.unwrap()),
        faulted(&format!(
            "Abort handler of task {lazy_id} returned without a terminal outcome"
        ))
    );
    assert_eq!(
        outcome(
            &harness
                .wait_for_task(throwing_id, &context())
                .await
                .unwrap()
        ),
        faulted("abort failed")
    );
    assert_eq!(
        status(harness.get_task(first_task, &context()).await.unwrap()),
        "running"
    );
    gate.resolve();
    harness.wait_for_task(first_task, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn does_not_signal_a_running_abort_handler_when_aborted_again_and_a_cancelled_caller_leaves_the_mark_durable()
 {
    let reached = Deferred::default();
    let proceed = Deferred::default();
    let run_release = Deferred::default();
    let abort_signalled = shared::<Option<bool>>();
    let aborts = Arc::new(AtomicUsize::new(0));
    let run = {
        let run_release = run_release.clone();
        let (reached, proceed, abort_signalled, aborts) = (
            reached.clone(),
            proceed.clone(),
            abort_signalled.clone(),
            aborts.clone(),
        );
        step_with(
            "test.abort-again",
            move |_, _, _| {
                // Ignores the signal, so the first caller is still joining when it gives up.
                let run_release = run_release.clone();
                async move {
                    run_release.wait().await;
                    Ok(())
                }
            },
            move |runtime, ctx| {
                aborts.fetch_add(1, Ordering::SeqCst);
                let (reached, proceed, abort_signalled) =
                    (reached.clone(), proceed.clone(), abort_signalled.clone());
                async move {
                    reached.resolve();
                    proceed.wait().await;
                    *abort_signalled.lock() = Some(runtime.signal().aborted());
                    abort_with(&runtime, "once", &ctx).await
                }
            },
        )
    };
    let opened = open_root(vec![run.any()], TaskOptions::default()).await;
    let harness = opened.harness.clone();
    let id = start(&opened.root, &run, (), false).await;
    harness.resume().unwrap();
    flush().await;
    // This caller gives up while joining; the mark and the abort invocation are unaffected.
    let controller = AbortController::new();
    let cancelled = {
        let harness = harness.clone();
        let abort_context = signal_context(controller.signal());
        tokio::spawn(async move { harness.abort_task(id, &abort_context).await })
    };
    while !harness
        .get_task(id, &context())
        .await
        .unwrap()
        .is_some_and(|task| task.abort_requested)
    {
        flush().await;
    }
    controller.abort(Some(AbortReason::message("caller gave up")));
    assert_err(cancelled.await.unwrap(), "caller gave up");
    run_release.resolve();
    reached.wait().await;
    assert_eq!(
        harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    proceed.resolve();
    assert_eq!(
        outcome(&harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "once" })
    );
    assert_eq!(*abort_signalled.lock(), Some(false));
    assert_eq!(aborts.load(Ordering::SeqCst), 1);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_a_terminal_outcome_committed_by_an_abort_handler_that_throws_afterwards() {
    let run = step_with(
        "test.abort-then-throw",
        |_, runtime, _| async move { aborted(runtime.signal()).await },
        |runtime, ctx| async move {
            abort_with(&runtime, "done", &ctx).await?;
            Err(Error::message("after terminal"))
        },
    );
    let opened = open_root(vec![run.any()], TaskOptions::default()).await;
    let id = start(&opened.root, &run, (), false).await;
    opened.harness.resume().unwrap();
    flush().await;
    opened.harness.abort_task(id, &context()).await.unwrap();
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "done" })
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn never_starts_phase_one_when_the_abort_lands_while_the_reservation_settles() {
    let ran = Arc::new(AtomicBool::new(false));
    let reserved = {
        let ran = ran.clone();
        step("test.reserved", move |_, runtime, _| {
            ran.store(true, Ordering::SeqCst);
            async move { aborted(runtime.signal()).await }
        })
    };
    let storage = Arc::new(ControlledStorage::new());
    let opened = open_root_in(
        storage.clone(),
        vec![reserved.any()],
        TaskOptions::default(),
    )
    .await;
    let id = start(&opened.root, &reserved, (), false).await;
    let held = storage.hold_commits();
    opened.harness.resume().unwrap();
    held.entered().await;
    // The reservation commit is in storage; the abort mark commit queues behind it.
    let aborting = {
        let harness = opened.harness.clone();
        tokio::spawn(async move { harness.abort_task(id, &context()).await })
    };
    flush().await;
    held.release();
    assert_eq!(aborting.await.unwrap().unwrap(), AbortTaskResult::Marked);
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "test" })
    );
    assert!(!ran.load(Ordering::SeqCst));
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn lets_an_abort_mark_win_over_a_fault_that_races_it() {
    let storage = Arc::new(ControlledStorage::new());
    let marking = shared::<Option<tokio::task::JoinHandle<Result<AbortTaskResult>>>>();
    let held = shared::<Option<Arc<crate::durable::session::tests::support::Gate>>>();
    let harness_ref = shared::<Option<Harness>>();
    let racing = {
        let (storage, marking, held, harness_ref) = (
            storage.clone(),
            marking.clone(),
            held.clone(),
            harness_ref.clone(),
        );
        step("test.racing-fault", move |id, _, _| {
            let gate = Arc::new(storage.hold_commits());
            *held.lock() = Some(gate.clone());
            let harness = harness_ref.lock().clone().unwrap();
            *marking.lock() = Some(tokio::spawn(async move {
                harness.abort_task(id, &context()).await
            }));
            async move {
                gate.entered().await;
                // The mark is in storage but not committed when the handler throws, so the fault commit queues
                // behind it.
                Err(Error::message("would fault"))
            }
        })
    };
    let opened = open_root_in(storage.clone(), vec![racing.any()], TaskOptions::default()).await;
    *harness_ref.lock() = Some(opened.harness.clone());
    let id = start(&opened.root, &racing, (), false).await;
    opened.harness.resume().unwrap();
    eventually(|| {
        let held = held.clone();
        async move { held.lock().is_some() }
    })
    .await;
    let gate = held.lock().clone().unwrap();
    gate.entered().await;
    flush().await;
    gate.release();
    assert_eq!(
        outcome(&opened.harness.wait_for_task(id, &context()).await.unwrap()),
        json!({ "status": "aborted", "reason": "test" })
    );
    let marking = marking.lock().take().unwrap();
    assert_eq!(marking.await.unwrap().unwrap(), AbortTaskResult::Marked);
    opened.harness.close(&context()).await.unwrap();
}

// ─── task close ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn stops_without_outcomes_and_starts_no_fresh_phase_or_abort_invocation_while_closing() {
    let reached = Deferred::default();
    let proceed = Deferred::default();
    let phases = shared::<Vec<String>>();
    let abort_ran = Arc::new(AtomicBool::new(false));
    let two = {
        let (reached, proceed) = (reached.clone(), proceed.clone());
        let (phases_1, phases_2) = (phases.clone(), phases.clone());
        let abort_ran = abort_ran.clone();
        define_task(
            TaskDefinition::<(), TwoPhase, ()>::new("test.two", 1, |_| TwoPhase {
                phase: "one".into(),
            })
            .phase("one", move |_, runtime, ctx| {
                phases_1.lock().push("one".into());
                let (reached, proceed) = (reached.clone(), proceed.clone());
                async move {
                    runtime
                        .commit(|_, _| async { Ok(Some(next_phase())) }, &ctx)
                        .await?;
                    reached.resolve();
                    // Ignores signals and returns normally once released; the closing rule wins over the next phase.
                    proceed.wait().await;
                    Ok(())
                }
            })
            .phase("two", move |_, _, _| {
                phases_2.lock().push("two".into());
                async { Ok(()) }
            })
            .abort(move |_, _, _| {
                abort_ran.store(true, Ordering::SeqCst);
                async { Ok(()) }
            }),
        )
    };
    let storage = Arc::new(ControlledStorage::new());
    let opened = open_root_in(storage.clone(), vec![two.any()], TaskOptions::default()).await;
    let id = {
        let two = two.clone();
        opened
            .root
            .commit(
                move |tx| async move { tx.create_task(&two, (), owned()).await.map(|id| id.erase()) },
                &context(),
            )
            .await
            .unwrap()
    };
    opened.harness.resume().unwrap();
    reached.wait().await;
    let aborting = mark_durably(&opened.harness, id).await;
    let record = opened.harness.get_task(id, &context()).await.unwrap();
    let commits = storage.commit_count();
    let closing = opened.harness.close(&context());
    proceed.resolve();
    closing.await.unwrap();
    assert_eq!(aborting.await.unwrap().unwrap(), AbortTaskResult::Marked);
    assert_eq!(storage.commit_count(), commits);
    assert_eq!(*phases.lock(), vec!["one"]);
    assert!(!abort_ran.load(Ordering::SeqCst));
    assert_matches(
        &to_json(&record),
        &json!({
            "abortRequested": true,
            "state": { "status": "running", "checkpoint": { "phase": "two" } },
        }),
    );
    assert_err(
        opened
            .harness
            .commit(|_| async { Ok(()) }, &context())
            .await,
        "closed",
    );
}

#[tokio::test]
async fn seals_admission_before_signalling_handlers_and_stops_watches_before_joining_them() {
    let notes = define_doc(DocDefinition::new(
        "test.close-notes",
        1,
        SessionScope,
        Notes::default,
    ))
    .unwrap();
    let reached = Deferred::default();
    let from_listener = shared::<Option<tokio::task::JoinHandle<Result<()>>>>();
    let harness_ref = shared::<Option<Harness>>();
    let watch_end = shared::<Option<String>>();
    let stubborn = {
        let (notes, reached, from_listener, harness_ref, watch_end) = (
            notes.clone(),
            reached.clone(),
            from_listener.clone(),
            harness_ref.clone(),
            watch_end.clone(),
        );
        step("test.stubborn", move |_, runtime, _| {
            let (notes, reached, from_listener, harness_ref, watch_end) = (
                notes.clone(),
                reached.clone(),
                from_listener.clone(),
                harness_ref.clone(),
                watch_end.clone(),
            );
            async move {
                // Uses a context close does not cancel, then waits for the watch to close.
                let watch = runtime.watch_doc(&notes, (), &context()).await?.unwrap();
                runtime.signal().add_listener(move |_| {
                    let harness = harness_ref.lock().clone().unwrap();
                    let commit = harness.commit(|_| async { Ok(()) }, &context());
                    *from_listener.lock() = Some(tokio::spawn(commit));
                });
                reached.resolve();
                *watch_end.lock() = Some(watch.closed().await.reason().to_string());
                Ok(())
            }
        })
    };
    let opened = open_root(vec![stubborn.any()], TaskOptions::default()).await;
    *harness_ref.lock() = Some(opened.harness.clone());
    {
        let notes = notes.clone();
        opened
            .harness
            .commit(
                move |tx| async move {
                    tx.doc(&notes, ())
                        .await?
                        .edit(|notes| notes.text = "x".into())?;
                    Ok(())
                },
                &context(),
            )
            .await
            .unwrap();
    }
    start(&opened.root, &stubborn, (), false).await;
    opened.harness.resume().unwrap();
    reached.wait().await;
    opened.harness.close(&context()).await.unwrap();
    let from_listener = from_listener.lock().take().unwrap();
    assert_err(from_listener.await.unwrap(), "closed");
    assert_eq!(watch_end.lock().as_deref(), Some("session_closed"));
}

#[tokio::test]
async fn writes_no_fault_when_close_seals_while_the_step_after_a_failed_phase_is_queued() {
    let storage = Arc::new(ControlledStorage::new());
    let held = shared::<Option<Arc<crate::durable::session::tests::support::Gate>>>();
    let harness_ref = shared::<Option<Harness>>();
    let throws = {
        let (storage, held, harness_ref) = (storage.clone(), held.clone(), harness_ref.clone());
        step("test.close-fault", move |_, runtime, _| {
            *held.lock() = Some(Arc::new(storage.hold_commits()));
            let harness = harness_ref.lock().clone().unwrap();
            spawn_commit_blocker(&harness, runtime.conversation_id());
            async {
                settle_spawned().await;
                Err(Error::message("would fault"))
            }
        })
    };
    let opened = open_root_in(storage.clone(), vec![throws.any()], TaskOptions::default()).await;
    *harness_ref.lock() = Some(opened.harness.clone());
    let id = start(&opened.root, &throws, (), false).await;
    opened.harness.resume().unwrap();
    eventually(|| {
        let held = held.clone();
        async move { held.lock().is_some() }
    })
    .await;
    let gate = held.lock().clone().unwrap();
    gate.entered().await;
    flush().await;
    let closing = opened.harness.close(&context());
    gate.release();
    closing.await.unwrap();
    assert_eq!(task_statuses(&storage, id), vec!["pending", "running"]);
}

#[tokio::test]
async fn starts_no_abort_handler_whose_reservation_settles_while_closing() {
    let ran = Arc::new(AtomicBool::new(false));
    let marked = {
        let ran = ran.clone();
        step_with(
            "test.close-abort-reservation",
            |_, _, _| async { Ok(()) },
            move |_, _| {
                ran.store(true, Ordering::SeqCst);
                async { Ok(()) }
            },
        )
    };
    let storage = Arc::new(ControlledStorage::new());
    let opened = open_root_in(storage.clone(), vec![marked.any()], TaskOptions::default()).await;
    let id = start(&opened.root, &marked, (), false).await;
    assert_eq!(
        opened.harness.abort_task(id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    let held = storage.hold_commits();
    opened.harness.resume().unwrap();
    held.entered().await;
    let closing = opened.harness.close(&context());
    held.release();
    closing.await.unwrap();
    assert!(!ran.load(Ordering::SeqCst));
    let writes: Vec<JsonValue> = storage
        .last_commit()
        .iter()
        .filter(|write| matches!(write, StorageWrite::Task { .. }))
        .map(to_json)
        .collect();
    assert_eq!(writes.len(), 1);
    assert_matches(
        &writes[0],
        &json!({ "value": { "id": id, "abortRequested": true, "state": { "status": "running" } } }),
    );
}

#[tokio::test]
async fn rejects_a_runtime_commit_that_was_queued_on_the_line_when_close_sealed_it() {
    let storage = Arc::new(ControlledStorage::new());
    let queued = shared::<Option<tokio::task::JoinHandle<Result<()>>>>();
    let held = shared::<Option<Arc<crate::durable::session::tests::support::Gate>>>();
    let harness_ref = shared::<Option<Harness>>();
    let proceed = Deferred::default();
    let task = {
        let (storage, queued, held, harness_ref, proceed) = (
            storage.clone(),
            queued.clone(),
            held.clone(),
            harness_ref.clone(),
            proceed.clone(),
        );
        step("test.close-queued-commit", move |_, runtime, _| {
            *held.lock() = Some(Arc::new(storage.hold_commits()));
            let harness = harness_ref.lock().clone().unwrap();
            spawn_commit_blocker(&harness, runtime.conversation_id());
            let committing = runtime.clone();
            *queued.lock() = Some(tokio::spawn(async move {
                complete(&committing, json!(null), &context()).await
            }));
            // Ignores the close signal, so the invocation is still alive when the queued commit reaches the line.
            let proceed = proceed.clone();
            async move {
                proceed.wait().await;
                Ok(())
            }
        })
    };
    let opened = open_root_in(storage.clone(), vec![task.any()], TaskOptions::default()).await;
    *harness_ref.lock() = Some(opened.harness.clone());
    let id = start(&opened.root, &task, (), false).await;
    opened.harness.resume().unwrap();
    eventually(|| {
        let held = held.clone();
        async move { held.lock().is_some() }
    })
    .await;
    let gate = held.lock().clone().unwrap();
    gate.entered().await;
    flush().await;
    let closing = opened.harness.close(&context());
    gate.release();
    let queued = queued.lock().take().unwrap();
    assert_err(queued.await.unwrap(), "Harness is closed");
    proceed.resolve();
    closing.await.unwrap();
    assert_eq!(task_statuses(&storage, id).len(), 2);
}

#[tokio::test]
async fn starts_no_next_phase_when_close_seals_during_a_step_that_decided_to_continue() {
    struct ClosingReader {
        registry: Registry,
        harness: Mutex<Option<Harness>>,
        close_on_snapshot: AtomicBool,
        closing: Mutex<Option<tokio::task::JoinHandle<Result<()>>>>,
    }
    impl RegistryReader for ClosingReader {
        fn snapshot(&self) -> RegistrySnapshot {
            if self.close_on_snapshot.swap(false, Ordering::SeqCst) {
                let harness = self.harness.lock().clone().unwrap();
                *self.closing.lock() = Some(tokio::spawn(harness.close(&context())));
            }
            self.registry.snapshot()
        }

        fn subscribe(&self, listener: Arc<dyn Fn() + Send + Sync>) -> Unsubscribe {
            self.registry.subscribe(listener)
        }
    }
    let phases = shared::<Vec<String>>();
    let registry = create_registry();
    // The step refreshes the snapshot after progress, inside its line callback and after its closing check.
    let reader = Arc::new(ClosingReader {
        registry: registry.clone(),
        harness: Mutex::new(None),
        close_on_snapshot: AtomicBool::new(false),
        closing: Mutex::new(None),
    });
    let two = {
        let (phases_1, phases_2) = (phases.clone(), phases.clone());
        let reader = reader.clone();
        define_task(
            TaskDefinition::<(), TwoPhase, ()>::new("test.close-in-step", 1, |_| TwoPhase {
                phase: "one".into(),
            })
            .phase("one", move |_, runtime, ctx| {
                phases_1.lock().push("one".into());
                let reader = reader.clone();
                async move {
                    runtime
                        .commit(|_, _| async { Ok(Some(next_phase())) }, &ctx)
                        .await?;
                    reader.close_on_snapshot.store(true, Ordering::SeqCst);
                    Ok(())
                }
            })
            .phase("two", move |_, _, _| {
                phases_2.lock().push("two".into());
                async { Ok(()) }
            })
            .abort(|_, _, _| async { Ok(()) }),
        )
    };
    add_task(&registry, two.any());
    let harness = Harness::open(
        Arc::new(MemoryStorage::new()),
        HarnessOptions::new(Models::default(), reader.clone()),
        &context(),
    )
    .await
    .unwrap();
    *reader.harness.lock() = Some(harness.clone());
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    root.commit(
        move |tx| async move { tx.create_task(&two, (), owned()).await.map(|_| ()) },
        &context(),
    )
    .await
    .unwrap();
    harness.resume().unwrap();
    eventually(|| {
        let reader = reader.clone();
        async move { reader.closing.lock().is_some() }
    })
    .await;
    let closing = reader.closing.lock().take().unwrap();
    closing.await.unwrap().unwrap();
    assert_eq!(*phases.lock(), vec!["one"]);
}

#[tokio::test]
async fn joins_a_reservation_that_settles_while_closing_without_starting_its_handler() {
    let ran = Arc::new(AtomicBool::new(false));
    let never = {
        let ran = ran.clone();
        step("test.close-reservation", move |_, _, _| {
            ran.store(true, Ordering::SeqCst);
            async { Ok(()) }
        })
    };
    let storage = Arc::new(ControlledStorage::new());
    let opened = open_root_in(storage.clone(), vec![never.any()], TaskOptions::default()).await;
    let id = start(&opened.root, &never, (), false).await;
    let held = storage.hold_commits();
    opened.harness.resume().unwrap();
    held.entered().await;
    let closing = opened.harness.close(&context());
    held.release();
    closing.await.unwrap();
    assert!(!ran.load(Ordering::SeqCst));
    let writes: Vec<JsonValue> = storage
        .last_commit()
        .iter()
        .filter(|write| matches!(write, StorageWrite::Task { .. }))
        .map(to_json)
        .collect();
    assert_eq!(writes.len(), 1);
    assert_matches(
        &writes[0],
        &json!({ "type": "task", "value": { "id": id, "state": { "status": "running" } } }),
    );
}
