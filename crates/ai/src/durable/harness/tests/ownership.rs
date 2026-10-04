//! Port of the `ownership` block of `test/harness-ownership.test.ts`. The "owned conversations from tools and
//! supervisors" block runs a faux chat and is ported with the generation milestone. SQLite reopen cases use
//! `ControlledStorage::persistent()`; the subclassed rejecting storages use `ControlledStorage::filter_commits()`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use tokio::sync::watch;

use super::support::*;
use super::tasks::{settle_spawned, shared};
use crate::chord::{AbortController, AbortReason};
use crate::durable::errors::{Error, StorageRejected};
use crate::durable::harness::live::{LIVE_DOC, RunState};
use crate::durable::harness::types::{ConversationAbortOptions, SubmissionDraft};
use crate::durable::harness::{Conversation, CreateOptions, Harness};
use crate::durable::ids::{ConversationId, TaskId};
use crate::durable::session::create_session;
use crate::durable::session::tests::support::ControlledStorage;
use crate::durable::tasks::{NextTaskState, Task, TaskDefinition, define_task};
use crate::durable::types::{
    ConversationOwnership, EntryDraft, JoinPolicy, Storage, StorageWrite,
    TaskOptions as CreateTaskOptions, TaskOutcome, TaskOutcomeError, TaskOwnership, TaskRecord,
    TaskState,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    Completed,
    Failed,
}

/// Named gates and run counts shared by the tasks of one test.
#[derive(Default)]
struct World {
    gates: Mutex<HashMap<String, Arc<watch::Sender<Option<Ending>>>>>,
    runs: Mutex<HashMap<String, usize>>,
}

impl World {
    fn gate(&self, name: &str) -> Arc<watch::Sender<Option<Ending>>> {
        self.gates
            .lock()
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(watch::channel(None).0))
            .clone()
    }

    fn open(&self, name: &str, ending: Ending) {
        self.gate(name).send_replace(Some(ending));
    }

    async fn wait(&self, name: &str) -> Ending {
        let mut receiver = self.gate(name).subscribe();
        let ending = *receiver
            .wait_for(Option::is_some)
            .await
            .expect("gate kept alive");
        ending.unwrap()
    }

    fn runs(&self, name: &str) -> usize {
        self.runs.lock().get(name).copied().unwrap_or(0)
    }
}

#[derive(Serialize, Deserialize)]
struct HoldInput {
    name: String,
    #[serde(
        rename = "slowAbort",
        default,
        skip_serializing_if = "std::ops::Not::not"
    )]
    slow_abort: bool,
}

fn hold_input(name: &str, slow_abort: bool) -> HoldInput {
    HoldInput {
        name: name.into(),
        slow_abort,
    }
}

#[derive(Serialize, Deserialize)]
struct Phase {
    phase: String,
}

fn phase(name: &str) -> Phase {
    Phase { phase: name.into() }
}

fn aborted_outcome() -> TaskOutcome {
    TaskOutcome::Aborted {
        reason: None,
        result: None,
    }
}

/// A task that holds until its named gate opens or it is aborted. With `slowAbort`, its abort handler first waits for
/// the gate `abort.<name>`.
fn hold_task(world: &Arc<World>) -> Task<HoldInput, Phase, JsonValue> {
    let (run_world, abort_world) = (world.clone(), world.clone());
    define_task(
        TaskDefinition::new("test.hold", 1, |_: &HoldInput| phase("hold"))
            .phase("hold", move |task, runtime, ctx| {
                let world = run_world.clone();
                async move {
                    *world
                        .runs
                        .lock()
                        .entry(task.input.name.clone())
                        .or_default() += 1;
                    let ending = tokio::select! {
                        ending = world.wait(&task.input.name) => ending,
                        result = aborted(runtime.signal()) => return result,
                    };
                    runtime
                        .commit(
                            move |_, _| async move {
                                Ok(Some(match ending {
                                    Ending::Completed => NextTaskState::completed(()),
                                    Ending::Failed => NextTaskState::Terminal {
                                        outcome: TaskOutcome::Failed {
                                            error: TaskOutcomeError {
                                                message: "gate failed".into(),
                                                detail: None,
                                            },
                                            result: None,
                                        },
                                    },
                                }))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(move |task, runtime, ctx| {
                let world = abort_world.clone();
                async move {
                    if task.input.slow_abort {
                        // Unlike TS, also end on the invocation signal: Tokio may start this handler before a
                        // following close, which then joins it, where TS microtask order seals first.
                        let gate = format!("abort.{}", task.input.name);
                        tokio::select! {
                            _ = world.wait(&gate) => {}
                            result = aborted(runtime.signal()) => return result,
                        }
                    }
                    runtime
                        .commit(
                            |_, _| async {
                                Ok(Some(NextTaskState::Terminal {
                                    outcome: aborted_outcome(),
                                }))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
    )
}

#[derive(Serialize, Deserialize)]
struct WaiterInput {
    on: Vec<TaskId>,
}

/// Waits on the tasks in its input, then completes; its abort handler ends it `aborted`.
fn waiter_task() -> Task<WaiterInput, Phase, JsonValue> {
    define_task(
        TaskDefinition::new("test.waiter", 1, |_: &WaiterInput| phase("wait"))
            .phase("wait", |task, runtime, ctx| async move {
                let on = task.input.on.clone();
                runtime
                    .commit(
                        move |_, _| async move {
                            Ok(Some(NextTaskState::waiting(
                                phase("done"),
                                on,
                                JoinPolicy::AllSettled,
                            )))
                        },
                        &ctx,
                    )
                    .await
            })
            .phase("done", |_, runtime, ctx| async move {
                runtime
                    .commit(
                        |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                        &ctx,
                    )
                    .await
            })
            .abort(|_, runtime, ctx| async move {
                runtime
                    .commit(
                        |_, _| async {
                            Ok(Some(NextTaskState::Terminal {
                                outcome: aborted_outcome(),
                            }))
                        },
                        &ctx,
                    )
                    .await
            }),
    )
}

/// Never registered: aborting it can only orphan it.
fn unregistered_task() -> Task<HoldInput, Phase, JsonValue> {
    define_task(
        TaskDefinition::new("test.unregistered", 1, |_: &HoldInput| phase("hold"))
            .phase("hold", |_, _, _| async { Ok(()) })
            .abort(|_, _, _| async { Ok(()) }),
    )
}

#[derive(Debug, Clone, Copy)]
struct Tree {
    owner: TaskId,
    child: ConversationId,
    inner: TaskId,
}

#[derive(Default, Clone, Copy)]
struct TreeOptions {
    background: bool,
    slow_inner: bool,
    slow_owner: bool,
}

fn in_conversation(conversation_id: Option<ConversationId>, background: bool) -> CreateTaskOptions {
    CreateTaskOptions {
        ownership: TaskOwnership::Conversation,
        conversation_id,
        background: Some(background),
    }
}

struct Setup {
    world: Arc<World>,
    hold: Task<HoldInput, Phase, JsonValue>,
    waiter: Task<WaiterInput, Phase, JsonValue>,
}

impl Setup {
    fn new() -> Self {
        let world = Arc::new(World::default());
        Self {
            hold: hold_task(&world),
            waiter: waiter_task(),
            world,
        }
    }

    fn open(&self, name: &str, ending: Ending) {
        self.world.open(name, ending);
    }

    /// In `parent`, stage task `owner` and a conversation it owns holding task `inner`, in one commit.
    async fn owned_child(&self, parent: &Conversation, name: &str, options: TreeOptions) -> Tree {
        let hold = self.hold.clone();
        let name = name.to_string();
        parent
            .commit(
                move |tx| async move {
                    let owner = tx
                        .create_task(
                            &hold,
                            hold_input(&name, options.slow_owner),
                            in_conversation(None, options.background),
                        )
                        .await?
                        .erase();
                    let child = tx
                        .create_conversation(ConversationOwnership::Task { task_id: owner })
                        .await?;
                    let inner = tx
                        .create_task(
                            &hold,
                            hold_input(&format!("{name}.inner"), options.slow_inner),
                            in_conversation(Some(child.id), false),
                        )
                        .await?
                        .erase();
                    Ok(Tree {
                        owner,
                        child: child.id,
                        inner,
                    })
                },
                &context(),
            )
            .await
            .unwrap()
    }

    async fn open_harness(&self, storage: Arc<dyn Storage>) -> (Harness, Conversation, Reports) {
        let (harness, _, reports) = open_tasks(
            storage,
            vec![self.hold.any(), self.waiter.any()],
            TaskOptions::default(),
        )
        .await;
        let root = harness
            .root(&context(), CreateOptions::default())
            .await
            .unwrap();
        harness.resume().unwrap();
        (harness, root, reports)
    }
}

async fn wait_until<F, Fut>(mut check: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    for _ in 0..500 {
        if check().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(check().await, "Condition was not reached");
}

/// Mark a conversation busy with `task` standing in for its run, so submissions queue.
async fn busy(conversation: &Conversation, task: TaskId) {
    let id = conversation.id;
    conversation
        .commit(
            move |tx| async move {
                tx.doc(&*LIVE_DOC, id).await?.edit(|live| {
                    live.run = Some(RunState {
                        task_id: task,
                        inputs: Vec::new(),
                    })
                })?;
                Ok(())
            },
            &context(),
        )
        .await
        .unwrap();
}

async fn state(harness: &Harness, id: TaskId) -> TaskState {
    harness
        .get_task(id, &context())
        .await
        .unwrap()
        .unwrap()
        .state
}

async fn status_of(harness: &Harness, id: TaskId) -> String {
    state(harness, id).await.status().to_string()
}

async fn outcome_status(harness: &Harness, id: TaskId) -> String {
    let record = harness.wait_for_task(id, &context()).await.unwrap();
    to_json(&record.state)["outcome"]["status"]
        .as_str()
        .unwrap()
        .to_string()
}

async fn conversation(harness: &Harness, id: ConversationId) -> Conversation {
    harness.conversation(id, &context()).await.unwrap().unwrap()
}

async fn submission_status(harness: &Harness, id: crate::durable::ids::SubmissionId) -> JsonValue {
    let submission = harness.submission(id, &context()).await.unwrap().unwrap();
    to_json(&submission.status(&context()).await.unwrap())
}

async fn idle_settles(conversation: &Conversation) -> bool {
    let conversation = conversation.clone();
    settled(async move { conversation.wait_for_idle(&context()).await })
        .await
        .0
}

async fn harness_idle_settles(harness: &Harness) -> bool {
    let harness = harness.clone();
    settled(async move { harness.wait_for_idle(&context()).await })
        .await
        .0
}

fn memory() -> Arc<dyn Storage> {
    Arc::new(crate::durable::storage::memory::MemoryStorage::new())
}

#[tokio::test]
async fn keeps_a_conversation_busy_while_its_owned_foreground_subtree_has_live_work_holding_the_completed_owner()
 {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let tree = setup
        .owned_child(&root, "owner", TreeOptions::default())
        .await;
    setup.open("owner", Ending::Completed);
    wait_until(|| async { status_of(&harness, tree.owner).await == "completing" }).await;
    let idle = {
        let root = root.clone();
        tokio::spawn(async move { root.wait_for_idle(&context()).await })
    };
    flush_all().await;
    assert!(!idle.is_finished());
    assert!(!harness_idle_settles(&harness).await);
    setup.open("owner.inner", Ending::Completed);
    idle.await.unwrap().unwrap();
    harness.wait_for_idle(&context()).await.unwrap();
    assert_eq!(outcome_status(&harness, tree.owner).await, "completed");
    harness.close(&context()).await.unwrap();
}

async fn flush_all() {
    crate::durable::session::tests::support::flush().await;
}

#[tokio::test]
async fn stops_idle_traversal_at_a_background_owner() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let tree = setup
        .owned_child(
            &root,
            "background",
            TreeOptions {
                background: true,
                ..TreeOptions::default()
            },
        )
        .await;
    root.wait_for_idle(&context()).await.unwrap();
    harness.wait_for_idle(&context()).await.unwrap();
    // The background child is its own scope.
    let child = conversation(&harness, tree.child).await;
    assert!(!idle_settles(&child).await);
    child
        .abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cascades_an_abort_mark_to_the_owned_foreground_subtree_and_withdraws_its_queued_inputs() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let tree = setup
        .owned_child(&root, "owner", TreeOptions::default())
        .await;
    let nested = conversation(&harness, tree.child).await;
    let deeper = setup
        .owned_child(&nested, "deeper", TreeOptions::default())
        .await;
    let shielded = setup
        .owned_child(
            &nested,
            "shielded",
            TreeOptions {
                background: true,
                ..TreeOptions::default()
            },
        )
        .await;
    // Busy conversations queue submissions; the inner task stands in for the child's run.
    busy(&nested, tree.inner).await;
    let queued = nested
        .submit(SubmissionDraft::input("later"), &context())
        .await
        .unwrap()
        .id;
    assert_eq!(
        harness.abort_task(tree.owner, &context()).await.unwrap(),
        crate::durable::harness::AbortTaskResult::Marked
    );
    for id in [tree.owner, tree.inner, deeper.owner, deeper.inner] {
        assert_eq!(outcome_status(&harness, id).await, "aborted");
    }
    // A nested background owner is a boundary.
    assert_ne!(status_of(&harness, shielded.owner).await, "terminal");
    assert_ne!(status_of(&harness, shielded.inner).await, "terminal");
    assert_matches(
        &submission_status(&harness, queued).await,
        &json!({ "status": "unanswered", "reason": "aborted" }),
    );
    harness
        .abort_task(shielded.owner, &context())
        .await
        .unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cascades_a_failed_owner_but_not_a_completed_one() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let failed = setup
        .owned_child(&root, "failed", TreeOptions::default())
        .await;
    let completed = setup
        .owned_child(&root, "done", TreeOptions::default())
        .await;
    setup.open("failed", Ending::Failed);
    setup.open("done", Ending::Completed);
    assert_eq!(outcome_status(&harness, failed.inner).await, "aborted");
    assert_eq!(outcome_status(&harness, failed.owner).await, "failed");
    wait_until(|| async { status_of(&harness, completed.owner).await == "completing" }).await;
    assert_ne!(status_of(&harness, completed.inner).await, "terminal");
    setup.open("done.inner", Ending::Completed);
    assert_eq!(outcome_status(&harness, completed.owner).await, "completed");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn aborts_a_background_task_directly_with_its_ordinary_subtree() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let tree = setup
        .owned_child(
            &root,
            "background",
            TreeOptions {
                background: true,
                ..TreeOptions::default()
            },
        )
        .await;
    harness.abort_task(tree.owner, &context()).await.unwrap();
    assert_eq!(outcome_status(&harness, tree.inner).await, "aborted");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn aborts_a_conversation_queued_inputs_withdrawn_writes_kept_foreground_work_aborted_background_kept()
 {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let foreground = setup
        .owned_child(&root, "foreground", TreeOptions::default())
        .await;
    let background = setup
        .owned_child(
            &root,
            "background",
            TreeOptions {
                background: true,
                ..TreeOptions::default()
            },
        )
        .await;
    busy(&root, foreground.owner).await;
    let input = root
        .submit(SubmissionDraft::input("later"), &context())
        .await
        .unwrap()
        .id;
    let write = root
        .submit(SubmissionDraft::write(EntryDraft::new("note")), &context())
        .await
        .unwrap()
        .id;
    root.abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    for id in [foreground.owner, foreground.inner] {
        assert_eq!(status_of(&harness, id).await, "terminal");
    }
    assert_ne!(status_of(&harness, background.owner).await, "terminal");
    assert_ne!(status_of(&harness, background.inner).await, "terminal");
    assert_matches(
        &submission_status(&harness, input).await,
        &json!({ "reason": "aborted" }),
    );
    assert_eq!(submission_status(&harness, write).await["status"], "queued");
    harness
        .abort_task(background.owner, &context())
        .await
        .unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn aborts_work_created_below_a_held_failed_owner_but_not_below_a_terminal_one() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let tree = setup
        .owned_child(
            &root,
            "owner",
            TreeOptions {
                slow_inner: true,
                ..TreeOptions::default()
            },
        )
        .await;
    setup.open("owner", Ending::Failed);
    wait_until(|| async {
        harness
            .get_task(tree.inner, &context())
            .await
            .unwrap()
            .unwrap()
            .abort_requested
    })
    .await;
    assert_eq!(status_of(&harness, tree.owner).await, "completing");
    let child = conversation(&harness, tree.child).await;
    let create = |name: &str| {
        let (child, hold, name) = (child.clone(), setup.hold.clone(), name.to_string());
        async move {
            child
                .commit(
                    move |tx| async move {
                        tx.create_task(
                            &hold,
                            hold_input(&name, false),
                            in_conversation(None, false),
                        )
                        .await
                        .map(|id| id.erase())
                    },
                    &context(),
                )
                .await
                .unwrap()
        }
    };
    let during = create("during").await;
    assert_eq!(outcome_status(&harness, during).await, "aborted");
    setup.open("abort.owner.inner", Ending::Completed);
    assert_eq!(outcome_status(&harness, tree.owner).await, "failed");
    // A terminal owner never cascades: interrogating its conversation runs normally.
    let after = create("after").await;
    setup.open("after", Ending::Completed);
    assert_eq!(outcome_status(&harness, after).await, "completed");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn withdraws_queued_inputs_below_the_aborted_task_but_keeps_its_own_conversations_queue_and_queued_writes()
 {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let tree = setup
        .owned_child(&root, "owner", TreeOptions::default())
        .await;
    let child = conversation(&harness, tree.child).await;
    busy(&root, tree.owner).await;
    busy(&child, tree.inner).await;
    let own = root
        .submit(SubmissionDraft::input("own"), &context())
        .await
        .unwrap()
        .id;
    let below = child
        .submit(SubmissionDraft::input("below"), &context())
        .await
        .unwrap()
        .id;
    let write = child
        .submit(SubmissionDraft::write(EntryDraft::new("note")), &context())
        .await
        .unwrap()
        .id;
    harness.abort_task(tree.owner, &context()).await.unwrap();
    harness.wait_for_task(tree.inner, &context()).await.unwrap();
    assert_eq!(submission_status(&harness, own).await["status"], "queued");
    assert_eq!(
        submission_status(&harness, below).await["status"],
        "unanswered"
    );
    assert_eq!(submission_status(&harness, write).await["status"], "queued");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_a_nested_background_owners_subtree_when_its_cancelled_background_owner_is_aborted() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let background = TreeOptions {
        background: true,
        ..TreeOptions::default()
    };
    let outer = setup.owned_child(&root, "outer", background).await;
    let outer_child = conversation(&harness, outer.child).await;
    let inner = setup.owned_child(&outer_child, "inner", background).await;
    harness.abort_task(outer.owner, &context()).await.unwrap();
    assert_eq!(outcome_status(&harness, outer.inner).await, "aborted");
    assert_ne!(status_of(&harness, inner.owner).await, "terminal");
    assert_ne!(status_of(&harness, inner.inner).await, "terminal");
    harness.abort_task(inner.owner, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn decides_idle_after_reopen_from_owner_edges_it_has_to_load_first() {
    let setup = Setup::new();
    let storage = Arc::new(ControlledStorage::persistent());
    let (harness, root, _) = setup.open_harness(storage.clone()).await;
    let foreground = setup.owned_child(&root, "fg", TreeOptions::default()).await;
    let fg_child = conversation(&harness, foreground.child).await;
    let deeper = setup
        .owned_child(&fg_child, "fg2", TreeOptions::default())
        .await;
    let background = setup
        .owned_child(
            &root,
            "bg",
            TreeOptions {
                background: true,
                ..TreeOptions::default()
            },
        )
        .await;
    setup.open("fg", Ending::Completed);
    setup.open("fg2", Ending::Completed);
    // Both owners hold their outcomes while the work below them runs.
    wait_until(|| async { status_of(&harness, foreground.owner).await == "completing" }).await;
    wait_until(|| async { status_of(&harness, deeper.owner).await == "completing" }).await;
    harness.close(&context()).await.unwrap();

    let (harness, root, _) = setup.open_harness(storage.clone()).await;
    // Two levels below completed foreground owners, the inner tasks keep the root busy.
    assert!(!idle_settles(&root).await);
    assert!(!harness_idle_settles(&harness).await);
    setup.open("fg.inner", Ending::Completed);
    setup.open("fg2.inner", Ending::Completed);
    // The background subtree does not count.
    root.wait_for_idle(&context()).await.unwrap();
    harness.wait_for_idle(&context()).await.unwrap();
    assert_eq!(status_of(&harness, foreground.owner).await, "terminal");
    assert_ne!(status_of(&harness, background.inner).await, "terminal");
    harness
        .abort_task(background.owner, &context())
        .await
        .unwrap();
    harness.close(&context()).await.unwrap();
}

async fn derives_marks_a_crash_left_unapplied(abort_requested: bool, owner_state: TaskState) {
    let setup = Setup::new();
    let storage = Arc::new(ControlledStorage::persistent());
    // Without a Harness, nothing derives marks: a cancelled owner with a live task below it.
    let session = create_session(storage.clone());
    let hold = setup.hold.clone();
    let (owner, inner) = session
        .commit(
            move |tx| async move {
                let root = tx
                    .create_conversation(ConversationOwnership::Ownerless)
                    .await?;
                let owner = tx
                    .create_task(
                        &hold,
                        hold_input("gone", false),
                        in_conversation(Some(root.id), false),
                    )
                    .await?
                    .erase();
                let child = tx
                    .create_conversation(ConversationOwnership::Task { task_id: owner })
                    .await?;
                let inner = tx
                    .create_task(
                        &hold,
                        hold_input("orphan", false),
                        in_conversation(Some(child.id), false),
                    )
                    .await?
                    .erase();
                Ok((owner, inner))
            },
            &context(),
        )
        .await
        .unwrap();
    session
        .commit(
            move |tx| async move {
                let record = tx.task(owner).await?.unwrap();
                tx.set_task(TaskRecord {
                    abort_requested,
                    state: owner_state,
                    ..record
                })
            },
            &context(),
        )
        .await
        .unwrap();
    session.close(&context()).await.unwrap();

    let (harness, _, _) = setup.open_harness(storage.clone()).await;
    assert_eq!(outcome_status(&harness, inner).await, "aborted");
    // The owner finishes only after the work below it.
    assert_eq!(
        outcome_status(&harness, owner).await,
        if abort_requested { "aborted" } else { "failed" }
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn derives_marks_a_crash_left_unapplied_below_a_held_failed_owner_at_open() {
    derives_marks_a_crash_left_unapplied(
        false,
        TaskState::Completing {
            outcome: TaskOutcome::Failed {
                error: TaskOutcomeError {
                    message: "crash".into(),
                    detail: None,
                },
                result: None,
            },
        },
    )
    .await;
}

#[tokio::test]
async fn derives_marks_a_crash_left_unapplied_below_an_abort_marked_owner_at_open() {
    derives_marks_a_crash_left_unapplied(
        true,
        TaskState::Pending {
            checkpoint: json!({ "phase": "hold" }),
        },
    )
    .await;
}

#[tokio::test]
async fn withdraws_an_input_queued_below_a_held_failed_owner_after_its_cascade() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let tree = setup
        .owned_child(
            &root,
            "owner",
            TreeOptions {
                slow_inner: true,
                ..TreeOptions::default()
            },
        )
        .await;
    let child = conversation(&harness, tree.child).await;
    // The inner task stands in for the child's run, which stays busy after the cascade.
    busy(&child, tree.inner).await;
    setup.open("owner", Ending::Failed);
    wait_until(|| async {
        harness
            .get_task(tree.inner, &context())
            .await
            .unwrap()
            .unwrap()
            .abort_requested
    })
    .await;
    let late = child
        .submit(SubmissionDraft::input("late"), &context())
        .await
        .unwrap();
    assert_matches(
        &to_json(&late.wait(&context()).await.unwrap()),
        &json!({ "status": "unanswered", "reason": "aborted" }),
    );
    setup.open("abort.owner.inner", Ending::Completed);
    harness.wait_for_task(tree.owner, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn marks_work_admitted_after_reopen_below_a_cancelled_owner_whose_edge_was_not_loaded() {
    let setup = Setup::new();
    let storage = Arc::new(ControlledStorage::persistent());
    let (harness, root, _) = setup.open_harness(storage.clone()).await;
    let tree = setup
        .owned_child(
            &root,
            "owner",
            TreeOptions {
                slow_owner: true,
                ..TreeOptions::default()
            },
        )
        .await;
    setup.open("owner.inner", Ending::Completed);
    harness.wait_for_task(tree.inner, &context()).await.unwrap();
    // The owner stays live and cancelled in its slow abort handler.
    harness.abort_task(tree.owner, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();

    // The child is empty at open, so nothing loads its edge until new work arrives.
    let (harness, _, _) = setup.open_harness(storage.clone()).await;
    let child = conversation(&harness, tree.child).await;
    let hold = setup.hold.clone();
    let late = child
        .commit(
            move |tx| async move {
                tx.create_task(
                    &hold,
                    hold_input("late", false),
                    in_conversation(None, false),
                )
                .await
                .map(|id| id.erase())
            },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(outcome_status(&harness, late).await, "aborted");
    setup.open("abort.owner", Ending::Completed);
    assert_eq!(outcome_status(&harness, tree.owner).await, "aborted");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cascades_from_an_owner_the_scheduler_orphans() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let hold = setup.hold.clone();
    let (owner, inner) = root
        .commit(
            move |tx| async move {
                let owner = tx
                    .create_task(
                        &unregistered_task(),
                        hold_input("unregistered", false),
                        in_conversation(None, false),
                    )
                    .await?
                    .erase();
                let child = tx
                    .create_conversation(ConversationOwnership::Task { task_id: owner })
                    .await?;
                let inner = tx
                    .create_task(
                        &hold,
                        hold_input("below", false),
                        in_conversation(Some(child.id), false),
                    )
                    .await?
                    .erase();
                Ok((owner, inner))
            },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(
        harness.abort_task(owner, &context()).await.unwrap(),
        crate::durable::harness::AbortTaskResult::Marked
    );
    assert_eq!(
        to_json(
            &harness
                .wait_for_task(owner, &context())
                .await
                .unwrap()
                .state
        )["outcome"],
        json!({ "status": "orphaned", "reason": "missing_task" })
    );
    assert_eq!(outcome_status(&harness, inner).await, "aborted");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cancels_only_the_callers_wait_never_the_shared_work() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let tree = setup
        .owned_child(&root, "owner", TreeOptions::default())
        .await;
    let waiting = AbortController::new();
    let idle = {
        let root = root.clone();
        let wait_context = signal_context(waiting.signal());
        tokio::spawn(async move { root.wait_for_idle(&wait_context).await })
    };
    flush_all().await;
    waiting.abort(Some(AbortReason::message("stop waiting")));
    assert_err(idle.await.unwrap(), "stop waiting");
    assert_ne!(status_of(&harness, tree.inner).await, "terminal");

    // Cancelling an abort after its commit leaves the marks in place.
    let slow = setup
        .owned_child(&root, "slow", TreeOptions::default())
        .await;
    root.commit(
        move |tx| async move {
            let record = tx.task(slow.owner).await?.unwrap();
            tx.set_task(TaskRecord {
                input: json!({ "name": "slow", "slowAbort": true }),
                ..record
            })
        },
        &context(),
    )
    .await
    .unwrap();
    let aborting = AbortController::new();
    let abort = {
        let root = root.clone();
        let abort_context = signal_context(aborting.signal());
        tokio::spawn(async move {
            root.abort(&abort_context, ConversationAbortOptions::default())
                .await
        })
    };
    wait_until(|| async {
        harness
            .get_task(slow.owner, &context())
            .await
            .unwrap()
            .unwrap()
            .abort_requested
    })
    .await;
    aborting.abort(Some(AbortReason::message("stop aborting")));
    assert_err(abort.await.unwrap(), "stop aborting");
    setup.open("abort.slow", Ending::Completed);
    for id in [tree.owner, tree.inner, slow.owner, slow.inner] {
        assert_eq!(outcome_status(&harness, id).await, "aborted");
    }
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn aborts_a_waiting_child_whose_awaited_task_completes_in_the_commit_that_marks_its_owner() {
    let setup = Setup::new();
    let (harness, root, _) = setup.open_harness(memory()).await;
    let (hold, waiter) = (setup.hold.clone(), setup.waiter.clone());
    let (owner, dependency, blocked) = root
        .commit(
            move |tx| async move {
                let owner = tx
                    .create_task(
                        &hold,
                        hold_input("owner", false),
                        in_conversation(None, false),
                    )
                    .await?
                    .erase();
                let child = tx
                    .create_conversation(ConversationOwnership::Task { task_id: owner })
                    .await?;
                let dependency = tx
                    .create_task(
                        &hold,
                        hold_input("dependency", false),
                        in_conversation(None, false),
                    )
                    .await?
                    .erase();
                let blocked = tx
                    .create_task(
                        &waiter,
                        WaiterInput {
                            on: vec![dependency],
                        },
                        in_conversation(Some(child.id), false),
                    )
                    .await?
                    .erase();
                Ok((owner, dependency, blocked))
            },
            &context(),
        )
        .await
        .unwrap();
    wait_until(|| async { setup.world.runs("dependency") == 1 && setup.world.runs("owner") == 1 })
        .await;
    wait_until(|| async { status_of(&harness, blocked).await == "waiting" }).await;
    // One commit completes the dependency and marks the owner.
    root.commit(
        move |tx| async move {
            let completed = tx.task(dependency).await?.unwrap();
            let marked = tx.task(owner).await?.unwrap();
            tx.set_task(TaskRecord {
                state: TaskState::Terminal {
                    outcome: TaskOutcome::Completed {
                        result: JsonValue::Null,
                    },
                },
                ..completed
            })?;
            tx.set_task(TaskRecord {
                abort_requested: true,
                ..marked
            })
        },
        &context(),
    )
    .await
    .unwrap();
    assert_eq!(outcome_status(&harness, blocked).await, "aborted");
    harness.close(&context()).await.unwrap();
}

/// Reject, once, the first batch that marks `target`.
fn reject_mark_once(storage: &ControlledStorage, target: Arc<Mutex<Option<TaskId>>>) {
    storage.filter_commits(Box::new(move |writes| {
        let mut target = target.lock();
        let marks = writes.iter().any(|write| {
            matches!(write, StorageWrite::Task { value } if Some(value.id) == *target && value.abort_requested)
        });
        if marks {
            *target = None;
            return Some(StorageRejected::new("rejected once").into());
        }
        None
    }));
}

#[tokio::test]
async fn retries_marks_found_through_an_edge_loaded_after_reopen_when_their_commit_is_rejected() {
    let setup = Setup::new();
    let storage = Arc::new(ControlledStorage::persistent());
    let (harness, root, _) = setup.open_harness(storage.clone()).await;
    let tree = setup
        .owned_child(
            &root,
            "owner",
            TreeOptions {
                slow_owner: true,
                ..TreeOptions::default()
            },
        )
        .await;
    setup.open("owner.inner", Ending::Completed);
    harness.wait_for_task(tree.inner, &context()).await.unwrap();
    harness.abort_task(tree.owner, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();

    let reject = shared::<Option<TaskId>>();
    reject_mark_once(&storage, reject.clone());
    let (harness, _, _) = setup.open_harness(storage.clone()).await;
    let child = conversation(&harness, tree.child).await;
    let (hold, target) = (setup.hold.clone(), reject.clone());
    let late = child
        .commit(
            move |tx| async move {
                let id = tx
                    .create_task(
                        &hold,
                        hold_input("late", false),
                        in_conversation(None, false),
                    )
                    .await?
                    .erase();
                *target.lock() = Some(id);
                Ok(id)
            },
            &context(),
        )
        .await
        .unwrap();
    wait_until(|| async { reject.lock().is_none() }).await;
    // Any later commit, here the reservation of the new task, retries the cascade.
    assert_eq!(outcome_status(&harness, late).await, "aborted");
    setup.open("abort.owner", Ending::Completed);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn retries_a_cascade_whose_commit_the_storage_rejected() {
    let setup = Setup::new();
    let storage = Arc::new(ControlledStorage::new());
    let reject = shared::<Option<TaskId>>();
    reject_mark_once(&storage, reject.clone());
    let (harness, root, reports) = setup.open_harness(storage.clone()).await;
    let tree = setup
        .owned_child(&root, "owner", TreeOptions::default())
        .await;
    *reject.lock() = Some(tree.inner);
    setup.open("owner", Ending::Failed);
    wait_until(|| async {
        reports
            .errors()
            .iter()
            .any(|error| matches!(error, Error::StorageRejected(_)))
    })
    .await;
    assert_eq!(status_of(&harness, tree.owner).await, "completing");
    assert_ne!(status_of(&harness, tree.inner).await, "terminal");
    // The next commit retries the cascade.
    let root_id = root.id;
    root.commit(
        move |tx| async move {
            tx.append_entry(root_id, EntryDraft::new("note")).await?;
            Ok(())
        },
        &context(),
    )
    .await
    .unwrap();
    assert_eq!(outcome_status(&harness, tree.inner).await, "aborted");
    assert_eq!(outcome_status(&harness, tree.owner).await, "failed");
    settle_spawned().await;
    harness.close(&context()).await.unwrap();
}
