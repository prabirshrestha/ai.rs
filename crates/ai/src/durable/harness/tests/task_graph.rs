//! Port of `test/harness-task-graph.test.ts`.
//!
//! Divergences: the reopened SQLite file is `ControlledStorage::persistent()`; graphs and operations are compared in
//! their TS JSON form, and TS identity checks (`toBe`) are `Arc` pointer checks on the task map.

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use super::support::*;
use crate::chord::delta::{Op, apply_immutable};
use crate::chord::{AbortController, AbortReason, DeliveryKind, ListenerOutcome};
use crate::durable::harness::scheduler::AbortTaskResult;
use crate::durable::harness::task_graph::TaskGraph;
use crate::durable::harness::types::TaskInspectionState;
use crate::durable::harness::{Conversation, CreateOptions, Harness};
use crate::durable::ids::{ConversationId, EntryId, TaskId};
use crate::durable::session::tests::support::{ControlledStorage, Deferred, flush};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{NextTaskState, Task, TaskDefinition, define_task};
use crate::durable::types::{
    ConversationOwnership, EntryDraft, JoinPolicy, Storage, TaskOptions as CreateTaskOptions,
    TaskOutcome, TaskOwnership,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Late {
    late: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
enum Work {
    Work,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
enum ParentState {
    Spawn,
    Join { child: TaskId },
    Finish,
}

fn conversation_owned() -> CreateTaskOptions {
    CreateTaskOptions::conversation(None)
}

fn task_owned(task_id: TaskId) -> CreateTaskOptions {
    CreateTaskOptions {
        ownership: TaskOwnership::Task { task_id },
        conversation_id: None,
        background: None,
    }
}

type ChildTask = Task<Late, Work, ()>;
type ParentTask = Task<(), ParentState, ()>;

/// A parent that creates a child task and a conversation it owns, waits for the child, creates a second child, and
/// completes while that child is still live, so its outcome is held as `completing`.
fn family(child_gate: Deferred, late_gate: Deferred) -> (ParentTask, ChildTask) {
    let child: ChildTask = define_task(
        TaskDefinition::new("test.graph-child", 1, |_: &Late| Work::Work)
            .phase("work", move |task, runtime, ctx| {
                let gate = if task.input.late {
                    late_gate.clone()
                } else {
                    child_gate.clone()
                };
                async move {
                    gate.wait().await;
                    runtime
                        .commit(
                            |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(|_, _, _| async { Ok(()) }),
    );
    let spawn_child = child.clone();
    let join_child = child.clone();
    let parent: ParentTask = define_task(
        TaskDefinition::new("test.graph-parent", 1, |_: &()| ParentState::Spawn)
            .phase("spawn", move |task, runtime, ctx| {
                let child = spawn_child.clone();
                async move {
                    let ownership = ConversationOwnership::Task {
                        task_id: task.id.erase(),
                    };
                    // Two conversations in a commit that leaves the parent's record unchanged.
                    runtime
                        .commit(
                            move |tx, _| async move {
                                tx.create_conversation(ownership).await?;
                                tx.create_conversation(ownership).await?;
                                Ok(None)
                            },
                            &ctx,
                        )
                        .await?;
                    let owner = task.id.erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let id = tx
                                    .create_task(&child, Late { late: false }, task_owned(owner))
                                    .await?
                                    .erase();
                                Ok(Some(NextTaskState::waiting(
                                    ParentState::Join { child: id },
                                    vec![id],
                                    JoinPolicy::AllSettled,
                                )))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .phase("join", move |task, runtime, ctx| {
                let child = join_child.clone();
                async move {
                    let owner = task.id.erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                tx.create_task(&child, Late { late: true }, task_owned(owner))
                                    .await?;
                                Ok(Some(NextTaskState::running(ParentState::Finish)))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .phase("finish", |_, runtime, ctx| async move {
                runtime
                    .commit(
                        |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                        &ctx,
                    )
                    .await
            })
            .abort(|_, _, _| async { Ok(()) }),
    );
    (parent, child)
}

/// Node states by task kind, for compact assertions.
fn statuses(graph: &TaskGraph) -> BTreeMap<String, String> {
    graph
        .tasks
        .values()
        .map(|node| {
            (
                format!("{}#{}", node.kind, node.id),
                to_json(&node.state)["status"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

async fn root_of(harness: &Harness) -> Conversation {
    harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap()
}

fn status_of(graph: &TaskGraph, id: TaskId) -> Option<String> {
    graph
        .tasks
        .get(&id)
        .map(|node| to_json(&node.state)["status"].as_str().unwrap().to_string())
}

#[tokio::test]
async fn follows_every_live_task_through_its_statuses_owner_edges_and_owned_conversations() {
    let (child_gate, late_gate) = (Deferred::default(), Deferred::default());
    let (parent_task, child_task) = family(child_gate.clone(), late_gate.clone());
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let (harness, _, _) = open_tasks(
        storage,
        vec![parent_task.any(), child_task.any()],
        TaskOptions::default(),
    )
    .await;
    let root = root_of(&harness).await;
    let seen: Arc<Mutex<Vec<TaskGraph>>> = Arc::default();
    let observe = || {
        let harness = harness.clone();
        let seen = seen.clone();
        async move {
            let opened = harness.task_graph(&context()).await.unwrap();
            let unsubscribe = opened.subscribe(move |value, _, delivery| {
                if delivery.kind == DeliveryKind::Update {
                    seen.lock().push(value);
                }
                ListenerOutcome::ok()
            });
            std::mem::forget(unsubscribe);
            opened
        }
    };
    let mut graph = observe().await;
    assert_eq!(to_json(&graph.value()), json!({ "tasks": {} }));

    let parent = {
        let parent_task = parent_task.clone();
        root.commit(
            move |tx| async move { tx.create_task(&parent_task, (), conversation_owned()).await },
            &context(),
        )
        .await
        .unwrap()
        .erase()
    };
    flush().await;
    assert_eq!(
        to_json(&graph.value())["tasks"][parent.0.to_string()],
        json!({
            "id": parent,
            "kind": "test.graph-parent",
            "conversationId": root.id,
            "background": false,
            "abortRequested": false,
            "state": { "status": "pending", "phase": "spawn" },
            "conversations": [],
        })
    );

    harness.resume().unwrap();
    let child_id = |graph: &TaskGraph| {
        graph
            .tasks
            .values()
            .find(|node| node.kind == "test.graph-child")
            .map(|node| node.id)
    };
    eventually(|| {
        let value = graph.value();
        let ready = value.tasks.len() == 2
            && child_id(&value)
                .is_some_and(|id| status_of(&value, id).as_deref() == Some("running"));
        async move { ready }
    })
    .await;
    let first = child_id(&graph.value()).unwrap();
    let parent_node = graph.value().tasks[&parent].clone();
    assert_eq!(
        to_json(&parent_node.state),
        json!({ "status": "waiting", "phase": "join", "on": [first], "policy": "allSettled" })
    );
    assert_eq!(parent_node.conversations.len(), 2);
    let mut sorted = parent_node.conversations.clone();
    sorted.sort();
    assert_eq!(sorted, parent_node.conversations);
    let owned = parent_node.conversations[0];
    let first_node = graph.value().tasks[&first].clone();
    assert_eq!(first_node.owner, Some(parent));
    assert_eq!(first_node.conversation_id, root.id);
    assert!(
        harness
            .conversation(owned, &context())
            .await
            .unwrap()
            .is_some()
    );
    // The advanced value equals a fresh build from Storage once the last observer left.
    let advanced = graph.value();
    graph.dispose().unwrap();
    graph = observe().await;
    assert!(!Arc::ptr_eq(&graph.value().tasks, &advanced.tasks));
    assert_eq!(graph.value(), advanced);

    child_gate.resolve();
    // The parent completes while the late child lives: its outcome is held.
    eventually(|| {
        let ready = status_of(&graph.value(), parent).as_deref() == Some("completing");
        async move { ready }
    })
    .await;
    assert_eq!(
        to_json(&graph.value().tasks[&parent].state),
        json!({ "status": "completing", "outcome": "completed" })
    );
    assert!(!graph.value().tasks.contains_key(&first));
    // Owned conversations stay listed while the owner lives.
    assert_eq!(
        graph.value().tasks[&parent].conversations,
        parent_node.conversations
    );
    let advanced = graph.value();
    graph.dispose().unwrap();
    graph = observe().await;
    assert!(!Arc::ptr_eq(&graph.value().tasks, &advanced.tasks));
    assert_eq!(graph.value(), advanced);

    late_gate.resolve();
    harness.wait_for_task(parent, &context()).await.unwrap();
    flush().await;
    assert_eq!(to_json(&graph.value()), json!({ "tasks": {} }));
    // Every revision is one commit that changed a node; none repeats its predecessor.
    let seen = seen.lock().clone();
    for pair in seen.windows(2) {
        assert_ne!(pair[0], pair[1]);
    }
    // The commit that created the two conversations published one revision setting them.
    assert!(seen.iter().any(|value| {
        value
            .tasks
            .get(&parent)
            .is_some_and(|node| node.conversations.len() == 2)
    }));
    graph.dispose().unwrap();
    harness.close(&context()).await.unwrap();
}

fn work_task(gate: Deferred) -> Task<(), Work, ()> {
    define_task(
        TaskDefinition::new("test.graph-work", 1, |_: &()| Work::Work)
            .phase("work", move |_, runtime, ctx| {
                let gate = gate.clone();
                async move {
                    tokio::select! {
                        _ = gate.wait() => {}
                        result = aborted(runtime.signal()) => result?,
                    }
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
                        |_, _| async {
                            Ok(Some(NextTaskState::Terminal {
                                outcome: TaskOutcome::Aborted {
                                    reason: Some("test".into()),
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

#[tokio::test]
async fn builds_from_committed_tasks_shows_surviving_tasks_as_pending_after_reopen_and_marks_aborts()
 {
    let storage = Arc::new(ControlledStorage::persistent());
    let gate = Deferred::default();
    let work = work_task(gate.clone());
    let (first, _, _) = open_tasks(storage.clone(), vec![work.any()], TaskOptions::default()).await;
    let first_root = root_of(&first).await;
    let (foreground, background, owned) = {
        let work = work.clone();
        first_root
            .commit(
                move |tx| async move {
                    let foreground = tx
                        .create_task(&work, (), conversation_owned())
                        .await?
                        .erase();
                    let background = tx
                        .create_task(
                            &work,
                            (),
                            CreateTaskOptions {
                                background: Some(true),
                                ..conversation_owned()
                            },
                        )
                        .await?
                        .erase();
                    let owned = tx
                        .create_conversation(ConversationOwnership::Task {
                            task_id: foreground,
                        })
                        .await?;
                    Ok((foreground, background, owned.id))
                },
                &context(),
            )
            .await
            .unwrap()
    };
    first.resume().unwrap();
    eventually(|| {
        let first = first.clone();
        async move {
            first
                .inspect(&context())
                .await
                .unwrap()
                .tasks
                .iter()
                .all(|task| matches!(task.state, TaskInspectionState::Running))
        }
    })
    .await;
    let running = first.task_graph(&context()).await.unwrap();
    assert_eq!(
        statuses(&running.value()),
        BTreeMap::from([
            (format!("test.graph-work#{foreground}"), "running".into()),
            (format!("test.graph-work#{background}"), "running".into()),
        ])
    );
    first.close(&context()).await.unwrap();

    // Acquired after reopen: built from the committed records and owner edges; open reconciled running to pending.
    let (harness, _, _) = open_tasks(storage, vec![work.any()], TaskOptions::default()).await;
    let watch = harness.watch_task_graph(&context()).await.unwrap();
    assert_eq!(
        statuses(&watch.value()),
        BTreeMap::from([
            (format!("test.graph-work#{foreground}"), "pending".into()),
            (format!("test.graph-work#{background}"), "pending".into()),
        ])
    );
    assert_eq!(watch.value().tasks[&foreground].conversations, vec![owned]);
    assert!(watch.value().tasks[&background].background);

    // Exact frames: replaying their operations from the acquisition revision gives each delivered value.
    let replica = Arc::new(Mutex::new(to_json(&watch.value())));
    let frames: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    {
        let (replica, frames) = (replica.clone(), frames.clone());
        watch
            .start(move |value, ops: Arc<[Op]>, _| {
                let mut replica = replica.lock();
                *replica = apply_immutable(&replica, &ops).unwrap();
                assert_eq!(*replica, to_json(&value));
                frames.lock().push(to_json(&ops.to_vec()));
                Box::pin(async { Ok(()) })
            })
            .unwrap();
    }
    harness.resume().unwrap();
    eventually(|| {
        let ready = status_of(&watch.value(), background).as_deref() == Some("running");
        async move { ready }
    })
    .await;
    assert_eq!(
        harness.abort_task(background, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    harness.wait_for_task(background, &context()).await.unwrap();
    flush().await;
    let frames = frames.lock().clone();
    let key = background.0.to_string();
    assert!(frames.iter().any(|frame| {
        frame.as_array().unwrap().len() == 1
            && frame[0][0] == "s"
            && frame[0][1] == json!(["tasks", key])
            && frame[0][2]["abortRequested"] == true
    }));
    assert_eq!(
        frames.last().unwrap(),
        &json!([["d", ["tasks", background.0.to_string()]]])
    );
    let keys: Vec<String> = replica.lock()["tasks"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(keys, vec![foreground.0.to_string()]);
    watch.stop().await;
    gate.resolve();
    harness.wait_for_task(foreground, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
enum Spawn {
    Spawn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct At {
    at: EntryId,
}

#[tokio::test]
async fn lists_owned_conversations_in_id_order_whatever_order_one_commit_creates_them_in() {
    let gate = Deferred::default();
    let created: Arc<Mutex<Vec<ConversationId>>> = Arc::default();
    let spawner: Task<At, Spawn, ()> = {
        let (gate, created) = (gate.clone(), created.clone());
        define_task(
            TaskDefinition::new("test.graph-spawner", 1, |_: &At| Spawn::Spawn)
                .phase("spawn", move |task, runtime, ctx| {
                    let (gate, created) = (gate.clone(), created.clone());
                    async move {
                        let ownership = ConversationOwnership::Task {
                            task_id: task.id.erase(),
                        };
                        let (conversation_id, at) = (task.conversation_id, task.input.at);
                        runtime
                            .commit(
                                move |tx, _| async move {
                                    let forked =
                                        tx.fork_conversation(conversation_id, at, ownership);
                                    let fresh = tx.create_conversation(ownership);
                                    let (forked, fresh) = futures::try_join!(forked, fresh)?;
                                    created.lock().extend([forked.id, fresh.id]);
                                    Ok(None)
                                },
                                &ctx,
                            )
                            .await?;
                        gate.wait().await;
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
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let (harness, _, _) = open_tasks(storage, vec![spawner.any()], TaskOptions::default()).await;
    let root = root_of(&harness).await;
    let graph = harness.task_graph(&context()).await.unwrap();
    let id = {
        let root_id = root.id;
        root.commit(
            move |tx| async move {
                let entry = tx.append_entry(root_id, EntryDraft::new("note")).await?;
                tx.create_task(&spawner, At { at: entry.id }, conversation_owned())
                    .await
            },
            &context(),
        )
        .await
        .unwrap()
        .erase()
    };
    harness.resume().unwrap();
    eventually(|| {
        let ready = graph
            .value()
            .tasks
            .get(&id)
            .is_some_and(|node| node.conversations.len() == 2);
        async move { ready }
    })
    .await;
    let advanced = graph.value();
    let mut expected = created.lock().clone();
    expected.sort();
    assert_eq!(advanced.tasks[&id].conversations, expected);
    graph.dispose().unwrap();
    let rebuilt = harness.task_graph(&context()).await.unwrap();
    assert_eq!(rebuilt.value(), advanced);
    rebuilt.dispose().unwrap();
    gate.resolve();
    harness.wait_for_task(id, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn registers_nothing_for_an_acquisition_cancelled_while_it_waits_for_the_line() {
    let storage = Arc::new(ControlledStorage::new());
    let (harness, _, _) = open_tasks(storage.clone(), vec![], TaskOptions::default()).await;
    let root = root_of(&harness).await;
    let (_, child) = family(Deferred::default(), Deferred::default());
    root.commit(
        move |tx| async move {
            tx.create_task(&child, Late { late: false }, conversation_owned())
                .await
        },
        &context(),
    )
    .await
    .unwrap();
    let held = storage.hold_commits();
    let root_id = root.id;
    let blocking = tokio::spawn(harness.commit_with(
        move |tx| async move {
            tx.append_entry(root_id, EntryDraft::new("blocker")).await?;
            Ok(())
        },
        &context(),
        Default::default(),
    ));
    held.entered().await;
    let controller = AbortController::new();
    let cancelled = {
        let harness = harness.clone();
        let signalled = signal_context(controller.signal());
        tokio::spawn(async move { harness.watch_task_graph(&signalled).await })
    };
    flush().await;
    controller.abort(Some(AbortReason::message("cancelled")));
    held.release();
    blocking.await.unwrap().unwrap();
    assert_err(cancelled.await.unwrap(), "cancelled");
    // No observer kept the mount: each new observer builds a new revision.
    let first = harness.task_graph(&context()).await.unwrap();
    let value = first.value();
    first.dispose().unwrap();
    let second = harness.task_graph(&context()).await.unwrap();
    assert!(!Arc::ptr_eq(&second.value().tasks, &value.tasks));
    assert_eq!(second.value(), value);
    second.dispose().unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn publishes_no_revision_for_a_commit_that_changes_no_node_and_shares_one_mount_between_observers()
 {
    let gate = Deferred::default();
    let reached = Deferred::default();
    let memo: Task<(), Work, ()> = {
        let (gate, reached) = (gate.clone(), reached.clone());
        define_task(
            TaskDefinition::new("test.graph-memo", 1, |_: &()| Work::Work)
                .phase("work", move |_, runtime, ctx| {
                    let (gate, reached) = (gate.clone(), reached.clone());
                    async move {
                        runtime.memo_with("seen", json!(true), &ctx).await?;
                        reached.resolve();
                        gate.wait().await;
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
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let (harness, _, _) = open_tasks(storage, vec![memo.any()], TaskOptions::default()).await;
    let root = root_of(&harness).await;
    let id = root
        .commit(
            move |tx| async move { tx.create_task(&memo, (), conversation_owned()).await },
            &context(),
        )
        .await
        .unwrap()
        .erase();
    let first = harness.task_graph(&context()).await.unwrap();
    let second = harness.task_graph(&context()).await.unwrap();
    assert!(Arc::ptr_eq(&second.value().tasks, &first.value().tasks));
    let updates: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = updates.clone();
    let unsubscribe = first.subscribe(move |value, _, delivery| {
        if delivery.kind == DeliveryKind::Update {
            sink.lock()
                .push(status_of(&value, id).unwrap_or_else(|| "gone".into()));
        }
        ListenerOutcome::ok()
    });
    harness.resume().unwrap();
    reached.wait().await;
    flush().await;
    // Reservation changed the node; the memo commit did not.
    assert_eq!(*updates.lock(), vec!["running"]);
    gate.resolve();
    harness.wait_for_task(id, &context()).await.unwrap();
    flush().await;
    assert_eq!(*updates.lock(), vec!["running", "gone"]);
    unsubscribe.unsubscribe();
    let last = first.value();
    first.dispose().unwrap();
    second.dispose().unwrap();
    // No observer is left, so the mount was dropped: a new observer builds a new revision.
    let rebuilt = harness.task_graph(&context()).await.unwrap();
    assert_eq!(rebuilt.value(), last);
    assert!(!Arc::ptr_eq(&rebuilt.value().tasks, &last.tasks));
    rebuilt.dispose().unwrap();
    harness.close(&context()).await.unwrap();
}
