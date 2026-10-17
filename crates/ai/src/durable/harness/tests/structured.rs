//! Port of `test/harness-structured.test.ts`: task ownership, waits, held outcomes, abort order, background
//! boundaries, recovery, and the built-in tool rounds as structured work.
//!
//! Divergences: the scripted `Node` task, its gates and its log live in a per-test `World` instead of module globals.
//! The SQLite files are `ControlledStorage::persistent()`, and the seeded crash records go through the internal
//! `Transaction::set_task()`. The subclassed rejecting storages use `ControlledStorage::filter_commits()`. Values that
//! are not strict JSON (a function in TS) are non-finite usage costs here. Skipped: creating a task with no ownership,
//! which `TaskOptions` cannot express.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Weak};
use std::time::Duration;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use tokio::sync::watch;

use super::chat::{
    ChatOptions, ChatSetup, ProxyOverrides, all_entries, chat_setup, entry_kinds, live_state,
    open_chat_with, proxy_models, scripted_stream, wait_for,
};
use super::support::*;
use crate::chord::Context;
use crate::durable::documents::{DocToken, define_doc};
use crate::durable::errors::{Error, Result, StorageRejected};
use crate::durable::harness::events::{AgentEventStream, EventBatch, watch_events};
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::live::SlotStatus;
use crate::durable::harness::registry::Registry;
use crate::durable::harness::scheduler::{AbortTaskResult, HookApi, TaskRuntime};
use crate::durable::harness::submissions::Submission;
use crate::durable::harness::types::{
    BlockedReason, ConversationAbortOptions, ConversationCreateOptions, GenerationHooks,
    SubmissionDraft, TaskInspectionState, ToolExecutionMode, ToolExecutionResult, ToolRegistration,
};
use crate::durable::harness::{Conversation, CreateOptions, Harness, define_tool, hook};
use crate::durable::ids::{ConversationId, TaskId};
use crate::durable::session::Transaction;
use crate::durable::session::create_session;
use crate::durable::session::tests::support::{ControlledStorage, Deferred};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{NextTaskState, Task, TaskDefinition, define_task};
use crate::durable::types::{
    ConversationOwnership, DocDefinition, EntryDraft, JoinPolicy, Storage, StorageWrite,
    TaskOptions as CreateTaskOptions, TaskOutcome, TaskOwnership, TaskQuery, TaskRecord, TaskScope,
    TaskState,
};
use crate::providers::faux::{
    FauxMessageOptions, FauxResponseStep, faux_assistant_message, faux_tool_call,
};
use crate::types::{AssistantMessageEvent, StopReason, Usage};

// ─── A scriptable task ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeInput {
    name: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
enum NodeCheckpoint {
    Run,
    Resume { round: u32 },
}

type NodeRuntime = TaskRuntime<NodeInput, NodeCheckpoint, String, ()>;
type NodeTask = Task<NodeInput, NodeCheckpoint, String>;
type Next = NextTaskState<NodeCheckpoint>;
type Handler = Arc<dyn Fn(NodeRuntime, Context) -> BoxFuture<'static, Result<()>> + Send + Sync>;
type ResumeHandler =
    Arc<dyn Fn(NodeRuntime, Context, u32) -> BoxFuture<'static, Result<()>> + Send + Sync>;

/// Per-name behavior of `Node`; unscripted parts use the defaults.
#[derive(Clone, Default)]
struct Behavior {
    run: Option<Handler>,
    resume: Option<ResumeHandler>,
    abort: Option<Handler>,
}

impl Behavior {
    fn run<F, Fut>(mut self, run: F) -> Self
    where
        F: Fn(NodeRuntime, Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        self.run = Some(Arc::new(move |runtime, ctx| Box::pin(run(runtime, ctx))));
        self
    }

    fn resume<F, Fut>(mut self, resume: F) -> Self
    where
        F: Fn(NodeRuntime, Context, u32) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        self.resume = Some(Arc::new(move |runtime, ctx, round| {
            Box::pin(resume(runtime, ctx, round))
        }));
        self
    }

    fn abort<F, Fut>(mut self, abort: F) -> Self
    where
        F: Fn(NodeRuntime, Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        self.abort = Some(Arc::new(move |runtime, ctx| Box::pin(abort(runtime, ctx))));
        self
    }
}

/// How a default run ends once its gate opens: its outcome, or a throw that faults it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    Completed,
    Failed,
    Throw,
}

/// The behaviors, gates and handler log of one test, and the `Node` task that runs them.
struct World {
    behaviors: Mutex<HashMap<String, Behavior>>,
    gates: Mutex<HashMap<String, Arc<watch::Sender<Option<Ending>>>>>,
    /// Handler starts in order, such as `run:a`, `abort:a`.
    log: Mutex<Vec<String>>,
    node: NodeTask,
}

impl World {
    fn new() -> Arc<Self> {
        Arc::new_cyclic(|world| Self {
            behaviors: Mutex::default(),
            gates: Mutex::default(),
            log: Mutex::default(),
            node: node_task(world.clone()),
        })
    }

    fn gate(&self, name: &str) -> Arc<watch::Sender<Option<Ending>>> {
        self.gates
            .lock()
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(watch::channel(None).0))
            .clone()
    }

    fn open(&self, name: &str) {
        self.end(name, Ending::Completed);
    }

    fn end(&self, name: &str, ending: Ending) {
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

    fn script(&self, name: &str, behavior: Behavior) {
        self.behaviors.lock().insert(name.to_string(), behavior);
    }

    fn behavior(&self, name: &str) -> Behavior {
        self.behaviors.lock().get(name).cloned().unwrap_or_default()
    }

    fn push(&self, line: String) {
        self.log.lock().push(line);
    }

    fn log(&self) -> Vec<String> {
        self.log.lock().clone()
    }

    fn logged(&self, line: &str) -> bool {
        self.log.lock().iter().any(|entry| entry == line)
    }

    fn aborts(&self) -> Vec<String> {
        self.log()
            .into_iter()
            .filter(|line| line.starts_with("abort:"))
            .collect()
    }

    /// A child task named `name`, owned by `owner`.
    async fn spawn(&self, tx: &Transaction, owner: TaskId, name: &str) -> Result<TaskId> {
        Ok(tx
            .create_task(&self.node, input(name), owned(owner))
            .await?
            .erase())
    }

    /// A task named `name` in `conversation`, which owns it.
    async fn create_in(
        &self,
        tx: &Transaction,
        conversation: ConversationId,
        name: &str,
    ) -> Result<TaskId> {
        Ok(tx
            .create_task(&self.node, input(name), in_conversation(conversation))
            .await?
            .erase())
    }

    /// Wait for the gate or the abort signal, then end as the gate says.
    async fn default_run(&self, runtime: NodeRuntime, ctx: Context, name: String) -> Result<()> {
        let ending = tokio::select! {
            ending = self.wait(&name) => ending,
            result = aborted(runtime.signal()) => return result,
        };
        if ending == Ending::Throw {
            return Err(Error::message(format!("{name} threw")));
        }
        commit_next(&runtime, &ctx, end(ending, &name)).await
    }

    async fn open_nodes(self: &Arc<Self>, storage: Arc<dyn Storage>) -> Opened {
        let (harness, registry, reports) =
            open_tasks(storage, vec![self.node.any()], TaskOptions::default()).await;
        let root = harness
            .root(&context(), CreateOptions::default())
            .await
            .unwrap();
        harness.resume().unwrap();
        Opened {
            harness,
            root,
            registry,
            reports,
        }
    }

    async fn start(&self, conversation: &Conversation, name: &str) -> TaskId {
        self.start_with(conversation, name, own_conversation())
            .await
    }

    async fn start_with(
        &self,
        conversation: &Conversation,
        name: &str,
        options: CreateTaskOptions,
    ) -> TaskId {
        let node = self.node.clone();
        let input = input(name);
        conversation
            .commit(
                move |tx| async move { Ok(tx.create_task(&node, input, options).await?.erase()) },
                &context(),
            )
            .await
            .unwrap()
    }
}

fn node_task(world: Weak<World>) -> NodeTask {
    let (run_world, resume_world, abort_world) = (world.clone(), world.clone(), world);
    define_task(
        TaskDefinition::new("test.node", 1, |_: &NodeInput| NodeCheckpoint::Run)
            .phase("run", move |task, runtime, ctx| {
                let world = run_world.upgrade().expect("world");
                async move {
                    let name = task.input.name;
                    world.push(format!("run:{name}"));
                    match world.behavior(&name).run {
                        Some(run) => run(runtime, ctx).await,
                        None => world.default_run(runtime, ctx, name).await,
                    }
                }
            })
            .phase("resume", move |task, runtime, ctx| {
                let world = resume_world.upgrade().expect("world");
                async move {
                    let name = task.input.name;
                    world.push(format!("resume:{name}"));
                    let round = match task.checkpoint {
                        NodeCheckpoint::Resume { round } => round,
                        NodeCheckpoint::Run => 0,
                    };
                    match world.behavior(&name).resume {
                        Some(resume) => resume(runtime, ctx, round).await,
                        None => commit_next(&runtime, &ctx, end(Ending::Completed, &name)).await,
                    }
                }
            })
            .abort(move |task, runtime, ctx| {
                let world = abort_world.upgrade().expect("world");
                async move {
                    let name = task.input.name;
                    world.push(format!("abort:{name}"));
                    match world.behavior(&name).abort {
                        Some(abort) => abort(runtime, ctx).await,
                        None => {
                            let outcome = TaskOutcome::Aborted {
                                reason: None,
                                result: Some(json!(name)),
                            };
                            commit_next(&runtime, &ctx, NextTaskState::Terminal { outcome }).await
                        }
                    }
                }
            }),
    )
}

/// Never registered: aborting it can only orphan it.
fn unregistered_task() -> NodeTask {
    define_task(
        TaskDefinition::new("test.unregistered", 1, |_: &NodeInput| NodeCheckpoint::Run)
            .phase("run", |_, _, _| async { Ok(()) })
            .abort(|_, _, _| async { Ok(()) }),
    )
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Notes {
    text: String,
}

fn task_notes() -> DocToken<Notes, TaskScope> {
    define_doc(DocDefinition::new(
        "test.task-notes",
        1,
        TaskScope,
        Notes::default,
    ))
    .unwrap()
}

fn input(name: &str) -> NodeInput {
    NodeInput { name: name.into() }
}

fn end(ending: Ending, name: &str) -> Next {
    match ending {
        Ending::Completed => NextTaskState::completed(name),
        _ => NextTaskState::failed(format!("{name} failed")),
    }
}

fn aborted_plain() -> Next {
    NextTaskState::Terminal {
        outcome: TaskOutcome::Aborted {
            reason: None,
            result: None,
        },
    }
}

/// Wait on `on`, resuming in round `round`.
fn wait_on(on: Vec<TaskId>, policy: JoinPolicy, round: u32) -> Next {
    NextTaskState::waiting(NodeCheckpoint::Resume { round }, on, policy)
}

fn resume_next(round: u32) -> Next {
    NextTaskState::running(NodeCheckpoint::Resume { round })
}

async fn commit_next(runtime: &NodeRuntime, ctx: &Context, next: Next) -> Result<()> {
    runtime
        .commit(move |_, _| async move { Ok(Some(next)) }, ctx)
        .await
}

fn own_conversation() -> CreateTaskOptions {
    CreateTaskOptions::conversation(None)
}

fn background() -> CreateTaskOptions {
    CreateTaskOptions {
        background: Some(true),
        ..own_conversation()
    }
}

fn in_conversation(conversation: ConversationId) -> CreateTaskOptions {
    CreateTaskOptions::conversation(Some(conversation))
}

fn owned(owner: TaskId) -> CreateTaskOptions {
    CreateTaskOptions {
        ownership: TaskOwnership::Task { task_id: owner },
        conversation_id: None,
        background: None,
    }
}

fn owned_by(owner: TaskId) -> ConversationOwnership {
    ConversationOwnership::Task { task_id: owner }
}

struct Opened {
    harness: Harness,
    root: Conversation,
    registry: Registry,
    reports: Reports,
}

fn memory() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new())
}

fn slot<T>() -> Arc<Mutex<Option<T>>> {
    Arc::new(Mutex::new(None))
}

/// The value a slot was filled with.
fn taken<T: Copy>(slot: &Arc<Mutex<Option<T>>>) -> T {
    slot.lock().expect("slot filled")
}

async fn record(harness: &Harness, id: TaskId) -> TaskRecord {
    harness.get_task(id, &context()).await.unwrap().unwrap()
}

async fn state(harness: &Harness, id: TaskId) -> TaskState {
    record(harness, id).await.state
}

async fn status(harness: &Harness, id: TaskId) -> String {
    state(harness, id).await.status().to_string()
}

async fn marked(harness: &Harness, id: TaskId) -> bool {
    record(harness, id).await.abort_requested
}

async fn settle(harness: &Harness, id: TaskId) -> TaskRecord {
    harness.wait_for_task(id, &context()).await.unwrap()
}

fn outcome_json(record: &TaskRecord) -> JsonValue {
    to_json(&record.state)["outcome"].clone()
}

async fn outcome_of(harness: &Harness, id: TaskId) -> String {
    outcome_json(&settle(harness, id).await)["status"]
        .as_str()
        .unwrap()
        .to_string()
}

async fn inspected(harness: &Harness, id: TaskId) -> Option<TaskInspectionState> {
    harness
        .inspect(&context())
        .await
        .unwrap()
        .tasks
        .into_iter()
        .find(|task| task.record.id == id)
        .map(|task| task.state)
}

/// Poll `check` in real time until it holds.
async fn until<F, Fut>(mut check: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    for _ in 0..1000 {
        if check().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("Condition was not reached");
}

async fn idle_settles(conversation: &Conversation) -> bool {
    let conversation = conversation.clone();
    settled(async move { conversation.wait_for_idle(&context()).await })
        .await
        .0
}

async fn task_settles(harness: &Harness, id: TaskId) -> bool {
    let harness = harness.clone();
    settled(async move { harness.wait_for_task(id, &context()).await })
        .await
        .0
}

fn statuses(outcomes: &[TaskOutcome]) -> Vec<String> {
    outcomes
        .iter()
        .map(|outcome| to_json(outcome)["status"].as_str().unwrap().to_string())
        .collect()
}

/// An abort handler that waits for the gate `abort.<name>`, then ends the task aborted. Unlike TS, it also ends on
/// the invocation signal, since Tokio may start it before a following close that then joins it.
fn slow_abort(world: &Arc<World>, name: &str) -> Behavior {
    let world = world.clone();
    let gate = format!("abort.{name}");
    Behavior::default().abort(move |runtime, ctx| {
        let (world, gate) = (world.clone(), gate.clone());
        async move {
            tokio::select! {
                _ = world.wait(&gate) => {}
                result = aborted(runtime.signal()) => return result,
            }
            commit_next(&runtime, &ctx, aborted_plain()).await
        }
    })
}

/// A run that spawns `child` and moves on to resume round 1; the default resume completes, holding while the child
/// lives. The child's ID lands in the returned slot.
fn spawn_and_finish(world: &Arc<World>, child: &str) -> (Behavior, Arc<Mutex<Option<TaskId>>>) {
    let found = slot();
    let (world, sink, child) = (world.clone(), found.clone(), child.to_string());
    let behavior = Behavior::default().run(move |runtime, ctx| {
        let (world, sink, child) = (world.clone(), sink.clone(), child.clone());
        async move {
            let owner = runtime.task_id().erase();
            runtime
                .commit(
                    move |tx, _| async move {
                        *sink.lock() = Some(world.spawn(&tx, owner, &child).await?);
                        Ok(Some(resume_next(1)))
                    },
                    &ctx,
                )
                .await
        }
    });
    (behavior, found)
}

/// A run that creates a conversation it owns and moves on to resume round 1; its ID lands in the returned slot.
fn host_conversation() -> (Behavior, Arc<Mutex<Option<ConversationId>>>) {
    let found = slot();
    let sink = found.clone();
    let behavior = Behavior::default().run(move |runtime, ctx| {
        let sink = sink.clone();
        async move {
            let owner = runtime.task_id().erase();
            runtime
                .commit(
                    move |tx, _| async move {
                        *sink.lock() = Some(tx.create_conversation(owned_by(owner)).await?.id);
                        Ok(Some(resume_next(1)))
                    },
                    &ctx,
                )
                .await
        }
    });
    (behavior, found)
}

/// A resume that blocks until the invocation is aborted.
fn resume_until_aborted(behavior: Behavior) -> Behavior {
    behavior.resume(|runtime, _, _| async move { aborted(runtime.signal()).await })
}

// ─── Ownership ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn creates_a_child_in_its_owners_conversation_with_its_owner_recorded() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let child = slot::<TaskId>();
    {
        let (world, sink) = (w.clone(), child.clone());
        w.script(
            "parent",
            Behavior::default().run(move |runtime, ctx| {
                let (world, sink) = (world.clone(), sink.clone());
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let id = world.spawn(&tx, owner, "child").await?;
                                *sink.lock() = Some(id);
                                Ok(Some(wait_on(vec![id], JoinPolicy::AllSettled, 1)))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
        );
    }
    let parent = w.start(&root, "parent").await;
    until(|| async { child.lock().is_some() }).await;
    let child = taken(&child);
    let created = record(&harness, child).await;
    assert_eq!(created.owner, Some(parent));
    assert_eq!(created.conversation_id, root.id);
    assert!(!created.background);
    assert_eq!(record(&harness, parent).await.owner, None);
    w.open("child");
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_a_missing_owner_a_child_in_another_conversation_and_a_background_child() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let parent = w.start(&root, "parent").await;
    let other = harness
        .create_conversation(
            ConversationCreateOptions::new(ConversationOwnership::Ownerless),
            &context(),
        )
        .await
        .unwrap();
    let create = |options: CreateTaskOptions| {
        let node = w.node.clone();
        let root = root.clone();
        async move {
            root.commit(
                move |tx| async move { Ok(tx.create_task(&node, input("x"), options).await?.erase()) },
                &context(),
            )
            .await
        }
    };
    let rejection = |result: Result<TaskId>| result.unwrap_err().to_string();
    assert!(rejection(create(owned(TaskId::new(999_999))).await).contains("does not exist"));
    assert!(
        rejection(
            create(CreateTaskOptions {
                conversation_id: Some(other.id),
                ..owned(parent)
            })
            .await
        )
        .contains("owner's conversation")
    );
    assert!(
        rejection(
            create(CreateTaskOptions {
                background: Some(true),
                ..owned(parent)
            })
            .await
        )
        .contains("cannot be background")
    );
    // The owner's conversation is the default, even from a commit bound to another conversation.
    let world = w.clone();
    let child = other
        .commit(
            move |tx| async move { world.spawn(&tx, parent, "child").await },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(record(&harness, child).await.conversation_id, root.id);
    w.open("parent");
    w.open("child");
    settle(&harness, parent).await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_new_owned_work_below_an_owner_that_is_completing_terminal_or_abort_marked() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let (behavior, _) = spawn_and_finish(&w, "child");
    w.script("parent", behavior);
    let parent = w.start(&root, "parent").await;
    until(|| async { status(&harness, parent).await == "completing" }).await;
    let create_child = |owner: TaskId| {
        let (world, root) = (w.clone(), root.clone());
        async move {
            root.commit(
                move |tx| async move { world.spawn(&tx, owner, "late").await },
                &context(),
            )
            .await
            .unwrap_err()
            .to_string()
        }
    };
    let create_conversation = || {
        let root = root.clone();
        async move {
            root.commit(
                move |tx| async move { Ok(tx.create_conversation(owned_by(parent)).await?.id) },
                &context(),
            )
            .await
            .unwrap_err()
            .to_string()
        }
    };
    assert!(create_child(parent).await.contains("is completing"));
    assert!(create_conversation().await.contains("is completing"));
    w.open("child");
    settle(&harness, parent).await;
    assert!(create_child(parent).await.contains("is terminal"));
    assert!(create_conversation().await.contains("is terminal"));

    w.script("slow", slow_abort(&w, "slow"));
    let slow = w.start(&root, "slow").await;
    until(|| async { w.logged("run:slow") }).await;
    harness.abort_task(slow, &context()).await.unwrap();
    assert!(create_child(slow).await.contains("is abort-marked"));
    w.open("abort.slow");
    assert_eq!(outcome_of(&harness, slow).await, "aborted");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cannot_create_a_child_in_its_finishing_commit_but_work_it_starts_in_an_owned_conversation_holds_it()
 {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    {
        let world = w.clone();
        w.script(
            "eager",
            Behavior::default().run(move |runtime, ctx| {
                let world = world.clone();
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                world.spawn(&tx, owner, "never").await?;
                                Ok(Some(end(Ending::Completed, "eager")))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
        );
    }
    let eager = w.start(&root, "eager").await;
    let outcome = outcome_json(&settle(&harness, eager).await);
    assert_eq!(outcome["status"], "faulted");
    assert!(
        outcome["error"]["message"]
            .as_str()
            .unwrap()
            .contains("is completing")
    );
    // The rejected commit wrote nothing.
    let nodes = harness
        .commit(
            |tx| async move {
                tx.scan_tasks(
                    TaskQuery {
                        kind: Some("test.node".into()),
                        ..TaskQuery::default()
                    },
                    20,
                    None,
                )
                .await
            },
            &context(),
        )
        .await
        .unwrap();
    let names: Vec<JsonValue> = nodes
        .items
        .iter()
        .map(|task| task.input["name"].clone())
        .collect();
    assert_eq!(names, vec![json!("eager")]);

    let (host, child) = host_conversation();
    {
        let (world, child) = (w.clone(), child.clone());
        w.script(
            "host",
            host.resume(move |runtime, ctx, _| {
                let (world, child) = (world.clone(), taken(&child));
                async move {
                    runtime
                        .commit(
                            move |tx, _| async move {
                                world.create_in(&tx, child, "inner").await?;
                                Ok(Some(end(Ending::Completed, "host")))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
        );
    }
    let host = w.start(&root, "host").await;
    until(|| async { status(&harness, host).await == "completing" }).await;
    assert!(!task_settles(&harness, host).await);
    w.open("inner");
    assert_eq!(outcome_of(&harness, host).await, "completed");
    harness.close(&context()).await.unwrap();
}

// ─── Waiting ────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Found {
    ids: Mutex<Vec<TaskId>>,
    outcomes: Mutex<Vec<String>>,
}

impl Found {
    fn ids(&self) -> Vec<TaskId> {
        self.ids.lock().clone()
    }

    fn outcomes(&self) -> Vec<String> {
        self.outcomes.lock().clone()
    }
}

/// A resume that records the outcomes of `found.ids` and completes.
fn record_outcomes(behavior: Behavior, name: &str, found: &Arc<Found>) -> Behavior {
    let (found, name) = (found.clone(), name.to_string());
    behavior.resume(move |runtime, ctx, _| {
        let (found, name) = (found.clone(), name.clone());
        async move {
            let outcomes = runtime.outcomes(&found.ids(), &ctx).await?;
            *found.outcomes.lock() = statuses(&outcomes);
            commit_next(&runtime, &ctx, end(Ending::Completed, &name)).await
        }
    })
}

/// Script `parent` to spawn `children` in one commit and wait on them with `policy`; it records their outcomes.
fn parent_of(
    world: &Arc<World>,
    parent: &str,
    children: &[&str],
    policy: JoinPolicy,
) -> Arc<Found> {
    let found = Arc::new(Found::default());
    let (spawner, sink) = (world.clone(), found.clone());
    let children: Vec<String> = children.iter().map(|name| name.to_string()).collect();
    let behavior = Behavior::default().run(move |runtime, ctx| {
        let (world, sink, children) = (spawner.clone(), sink.clone(), children.clone());
        async move {
            let owner = runtime.task_id().erase();
            runtime
                .commit(
                    move |tx, _| async move {
                        for name in &children {
                            let id = world.spawn(&tx, owner, name).await?;
                            sink.ids.lock().push(id);
                        }
                        Ok(Some(wait_on(sink.ids(), policy, 1)))
                    },
                    &ctx,
                )
                .await
        }
    });
    world.script(parent, record_outcomes(behavior, parent, &found));
    found
}

#[tokio::test]
async fn resumes_once_every_awaited_task_is_terminal_and_reads_their_outcomes_in_order_all_settled()
{
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let found = Arc::new(Found::default());
    {
        let (world, sink) = (w.clone(), found.clone());
        let behavior = Behavior::default().run(move |runtime, ctx| {
            let (world, sink) = (world.clone(), sink.clone());
            async move {
                let owner = runtime.task_id().erase();
                runtime
                    .commit(
                        move |tx, _| async move {
                            for name in ["ok", "fails", "throws", "aborted"] {
                                let id = world.spawn(&tx, owner, name).await?;
                                sink.ids.lock().push(id);
                            }
                            let orphan = tx
                                .create_task(&unregistered_task(), input("orphan"), owned(owner))
                                .await?
                                .erase();
                            sink.ids.lock().push(orphan);
                            Ok(Some(wait_on(sink.ids(), JoinPolicy::AllSettled, 1)))
                        },
                        &ctx,
                    )
                    .await
            }
        });
        w.script("parent", record_outcomes(behavior, "parent", &found));
    }
    let parent = w.start(&root, "parent").await;
    until(|| async { found.ids().len() == 5 }).await;
    until(|| async { status(&harness, parent).await == "waiting" }).await;
    w.open("ok");
    w.end("fails", Ending::Failed);
    w.end("throws", Ending::Throw);
    let ids = found.ids();
    harness.abort_task(ids[3], &context()).await.unwrap();
    assert_eq!(
        harness.abort_task(ids[4], &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    assert_eq!(
        found.outcomes(),
        ["completed", "failed", "faulted", "aborted", "orphaned"]
    );
    // allSettled never marks siblings.
    assert_eq!(w.aborts(), ["abort:aborted"]);
    harness.close(&context()).await.unwrap();
}

async fn fails_fast_when_a_child_ends(ending: Ending) {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let found = parent_of(
        &w,
        "checkout",
        &["p1", "p2", "p3", "p4"],
        JoinPolicy::FailFast,
    );
    let parent = w.start(&root, "checkout").await;
    until(|| async { status(&harness, parent).await == "waiting" }).await;
    w.end("p2", ending);
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    let failed = if ending == Ending::Throw {
        "faulted"
    } else {
        "failed"
    };
    assert_eq!(found.outcomes(), ["aborted", failed, "aborted", "aborted"]);
    assert!(!marked(&harness, parent).await);
    let mut aborts = w.aborts();
    aborts.sort();
    assert_eq!(aborts, ["abort:p1", "abort:p3", "abort:p4"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn fails_fast_when_a_child_ends_failed_its_live_siblings_are_aborted_the_parent_is_not() {
    fails_fast_when_a_child_ends(Ending::Failed).await;
}

#[tokio::test]
async fn fails_fast_when_a_child_ends_faulted_its_live_siblings_are_aborted_the_parent_is_not() {
    fails_fast_when_a_child_ends(Ending::Throw).await;
}

#[tokio::test]
async fn fails_fast_on_a_held_failure_before_the_failing_child_drains() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let (p1, _) = spawn_and_finish(&w, "grandchild");
    w.script(
        "p1",
        p1.resume(|runtime, ctx, _| async move {
            commit_next(&runtime, &ctx, end(Ending::Failed, "p1")).await
        }),
    );
    w.script("grandchild", slow_abort(&w, "grandchild"));
    let found = Arc::new(Found::default());
    let outside = slot::<TaskId>();
    {
        let (world, sink, outside) = (w.clone(), found.clone(), outside.clone());
        w.script(
            "parent",
            Behavior::default().run(move |runtime, ctx| {
                let (world, sink, outside) = (world.clone(), sink.clone(), outside.clone());
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                for name in ["p1", "p2"] {
                                    let id = world.spawn(&tx, owner, name).await?;
                                    sink.ids.lock().push(id);
                                }
                                *outside.lock() = Some(world.spawn(&tx, owner, "outside").await?);
                                Ok(Some(wait_on(sink.ids(), JoinPolicy::FailFast, 1)))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
        );
    }
    let parent = w.start(&root, "parent").await;
    until(|| async { found.ids().len() == 2 }).await;
    let ids = found.ids();
    let outside = taken(&outside);
    // p1 holds `failed` while its grandchild's abort handler runs; p2 is already aborted.
    assert_eq!(outcome_of(&harness, ids[1]).await, "aborted");
    let held = to_json(&state(&harness, ids[0]).await);
    assert_eq!(held["status"], "completing");
    assert_eq!(held["outcome"]["status"], "failed");
    assert_eq!(status(&harness, parent).await, "waiting");
    w.open("abort.grandchild");
    until(|| async { w.logged("resume:parent") }).await;
    // Only the other tasks in `on` are marked: not the failed one, not the parent, not a child outside `on`.
    assert_eq!(outcome_of(&harness, ids[0]).await, "failed");
    assert_eq!(
        [
            marked(&harness, ids[0]).await,
            marked(&harness, ids[1]).await,
            marked(&harness, outside).await
        ],
        [false, true, false]
    );
    assert!(!marked(&harness, parent).await);
    assert_eq!(status(&harness, outside).await, "running");
    // Finished, the parent holds for the child outside `on`.
    until(|| async { status(&harness, parent).await == "completing" }).await;
    w.open("outside");
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn waits_with_all_settled_on_tasks_it_does_not_own_including_already_terminal_ones() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let done = w.start(&root, "done").await;
    w.open("done");
    settle(&harness, done).await;
    let live = w.start(&root, "live").await;
    let found = Arc::new(Found::default());
    found.ids.lock().extend([done, live]);
    w.script(
        "parent",
        record_outcomes(
            Behavior::default().run(move |runtime, ctx| async move {
                commit_next(
                    &runtime,
                    &ctx,
                    wait_on(vec![done, live], JoinPolicy::AllSettled, 1),
                )
                .await
            }),
            "parent",
            &found,
        ),
    );
    let parent = w.start(&root, "parent").await;
    until(|| async { status(&harness, parent).await == "waiting" }).await;
    // A task it does not own is not its work: the parent may finish while it lives, but here it waits for it.
    w.end("live", Ending::Failed);
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    assert_eq!(found.outcomes(), ["completed", "failed"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_waits_on_itself_its_owner_a_missing_task_and_fail_fast_on_a_task_it_does_not_own()
{
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let other = w.start(&root, "other").await;
    let cases: [(&str, Option<TaskId>, JoinPolicy, &str); 3] = [
        (
            "self",
            None,
            JoinPolicy::AllSettled,
            "cannot wait on itself or its owner",
        ),
        (
            "missing",
            Some(TaskId::new(999_999)),
            JoinPolicy::AllSettled,
            "does not exist",
        ),
        (
            "foreign",
            Some(other),
            JoinPolicy::FailFast,
            "only on tasks it owns",
        ),
    ];
    for (name, on, policy, message) in cases {
        w.script(
            name,
            Behavior::default().run(move |runtime, ctx| async move {
                let on = on.unwrap_or_else(|| runtime.task_id().erase());
                commit_next(&runtime, &ctx, wait_on(vec![on], policy, 1)).await
            }),
        );
        let id = w.start(&root, name).await;
        let outcome = outcome_json(&settle(&harness, id).await);
        assert_eq!(outcome["status"], "faulted", "{name}");
        assert!(
            outcome["error"]["message"]
                .as_str()
                .unwrap()
                .contains(message),
            "{name}: {outcome}"
        );
    }
    // A child waiting on its owner could never resume.
    w.script(
        "child",
        Behavior::default().run(|runtime, ctx| async move {
            let task = runtime.get_task(runtime.task_id(), &ctx).await?.unwrap();
            let owner = task.owner.unwrap();
            commit_next(
                &runtime,
                &ctx,
                wait_on(vec![owner], JoinPolicy::AllSettled, 1),
            )
            .await
        }),
    );
    let child = slot::<TaskId>();
    {
        let (world, sink) = (w.clone(), child.clone());
        w.script(
            "parent",
            Behavior::default().run(move |runtime, ctx| {
                let (world, sink) = (world.clone(), sink.clone());
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let id = world.spawn(&tx, owner, "child").await?;
                                *sink.lock() = Some(id);
                                Ok(Some(wait_on(vec![id], JoinPolicy::AllSettled, 1)))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
        );
    }
    let parent = w.start(&root, "parent").await;
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    let outcome = outcome_json(&settle(&harness, taken(&child)).await);
    assert_eq!(outcome["status"], "faulted");
    assert!(
        outcome["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot wait on itself or its owner")
    );
    w.open("other");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn resumes_at_the_next_pass_when_it_waits_on_nothing() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    w.script(
        "parent",
        Behavior::default().run(|runtime, ctx| async move {
            commit_next(&runtime, &ctx, wait_on(Vec::new(), JoinPolicy::FailFast, 1)).await
        }),
    );
    let parent = w.start(&root, "parent").await;
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    assert_eq!(w.log(), ["run:parent", "resume:parent"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_running_phases_after_spawning_and_waits_on_subsets_in_sequence() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let ids: Arc<Mutex<Vec<TaskId>>> = Arc::default();
    {
        let (world, sink, rounds_world, read) = (w.clone(), ids.clone(), w.clone(), ids.clone());
        w.script(
            "parent",
            Behavior::default()
                .run(move |runtime, ctx| {
                    let (world, sink) = (world.clone(), sink.clone());
                    async move {
                        let owner = runtime.task_id().erase();
                        runtime
                            .commit(
                                move |tx, _| async move {
                                    for name in ["a", "b", "c"] {
                                        let id = world.spawn(&tx, owner, name).await?;
                                        sink.lock().push(id);
                                    }
                                    Ok(Some(resume_next(0)))
                                },
                                &ctx,
                            )
                            .await
                    }
                })
                .resume(move |runtime, ctx, round| {
                    let (world, ids) = (rounds_world.clone(), read.lock().clone());
                    async move {
                        world.push(format!("round:{round}"));
                        let next = match round {
                            0 => wait_on(vec![ids[0]], JoinPolicy::AllSettled, 1),
                            1 => wait_on(vec![ids[1], ids[2]], JoinPolicy::FailFast, 2),
                            _ => end(Ending::Completed, "parent"),
                        };
                        commit_next(&runtime, &ctx, next).await
                    }
                }),
        );
    }
    let parent = w.start(&root, "parent").await;
    until(|| async { w.logged("round:0") }).await;
    until(|| async { status(&harness, parent).await == "waiting" }).await;
    let children = ids.lock().clone();
    w.open("b");
    assert!(task_settles(&harness, children[1]).await);
    assert_eq!(status(&harness, parent).await, "waiting");
    w.open("a");
    until(|| async { w.logged("round:1") }).await;
    w.open("c");
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    let rounds: Vec<String> = w
        .log()
        .into_iter()
        .filter(|line| line.starts_with("round:"))
        .collect();
    assert_eq!(rounds, ["round:0", "round:1", "round:2"]);
    harness.close(&context()).await.unwrap();
}

// ─── Completing ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn holds_a_finished_task_until_its_owned_work_drains_waiters_and_task_documents_wait_for_the_final_commit()
 {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let notes = task_notes();
    let child = slot::<TaskId>();
    {
        let (world, sink, notes) = (w.clone(), child.clone(), notes.clone());
        w.script(
            "parent",
            Behavior::default().run(move |runtime, ctx| {
                let (world, sink, notes) = (world.clone(), sink.clone(), notes.clone());
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                tx.doc(&notes, owner)
                                    .await?
                                    .edit(|notes| notes.text = "notes".into())?;
                                *sink.lock() = Some(world.spawn(&tx, owner, "child").await?);
                                Ok(Some(resume_next(1)))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
        );
    }
    let parent = w.start(&root, "parent").await;
    until(|| async { status(&harness, parent).await == "completing" }).await;
    assert_eq!(
        to_json(&state(&harness, parent).await),
        json!({ "status": "completing", "outcome": { "status": "completed", "result": "parent" } })
    );
    let (done, waiter) = {
        let harness = harness.clone();
        settled(async move { harness.wait_for_task(parent, &context()).await }).await
    };
    assert!(!done);
    assert_eq!(
        harness.snapshot(&notes, parent, &context()).await.unwrap(),
        Some(Notes {
            text: "notes".into()
        })
    );
    assert!(matches!(
        inspected(&harness, parent).await,
        Some(TaskInspectionState::Completing)
    ));
    assert!(!idle_settles(&root).await);
    w.open("child");
    assert_eq!(
        outcome_json(&waiter.await.unwrap().unwrap()),
        json!({ "status": "completed", "result": "parent" })
    );
    assert_eq!(
        harness.snapshot(&notes, parent, &context()).await.unwrap(),
        None
    );
    root.wait_for_idle(&context()).await.unwrap();
    assert!(child.lock().is_some());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_holding_for_ordinary_work_created_during_the_hold() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let conversation = slot::<ConversationId>();
    {
        let (world, sink) = (w.clone(), conversation.clone());
        w.script(
            "parent",
            Behavior::default().run(move |runtime, ctx| {
                let (world, sink) = (world.clone(), sink.clone());
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let conversation =
                                    tx.create_conversation(owned_by(owner)).await?.id;
                                *sink.lock() = Some(conversation);
                                world.create_in(&tx, conversation, "first").await?;
                                Ok(Some(resume_next(1)))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
        );
    }
    let parent = w.start(&root, "parent").await;
    until(|| async { status(&harness, parent).await == "completing" }).await;
    let id = taken(&conversation);
    let child = harness.conversation(id, &context()).await.unwrap().unwrap();
    w.start(&child, "second").await;
    w.open("first");
    until(|| async { w.logged("run:second") }).await;
    assert_eq!(status(&harness, parent).await, "completing");
    w.open("second");
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    harness.close(&context()).await.unwrap();
}

async fn aborts_the_work_below_a_held_outcome(ending: Ending) {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let (behavior, child) = spawn_and_finish(&w, "child");
    w.script(
        "parent",
        behavior.resume(move |runtime, ctx, _| async move {
            if ending == Ending::Throw {
                return Err(Error::message("parent threw"));
            }
            commit_next(&runtime, &ctx, end(Ending::Failed, "parent")).await
        }),
    );
    let parent = w.start(&root, "parent").await;
    until(|| async { child.lock().is_some() }).await;
    let child = taken(&child);
    assert_eq!(outcome_of(&harness, child).await, "aborted");
    assert_eq!(
        outcome_of(&harness, parent).await,
        if ending == Ending::Throw {
            "faulted"
        } else {
            "failed"
        }
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn aborts_the_work_below_a_held_failure_then_finishes_with_the_held_outcome() {
    aborts_the_work_below_a_held_outcome(Ending::Failed).await;
}

#[tokio::test]
async fn aborts_the_work_below_a_held_scheduler_fault_then_finishes_with_the_held_outcome() {
    aborts_the_work_below_a_held_outcome(Ending::Throw).await;
}

#[tokio::test]
async fn only_marks_a_completing_task_when_aborted_the_work_below_is_aborted_the_held_outcome_stays()
 {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let (behavior, child) = spawn_and_finish(&w, "child");
    w.script("parent", behavior);
    let parent = w.start(&root, "parent").await;
    until(|| async { status(&harness, parent).await == "completing" }).await;
    assert_eq!(
        harness.abort_task(parent, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    let child = taken(&child);
    assert_eq!(outcome_of(&harness, child).await, "aborted");
    let settled_parent = settle(&harness, parent).await;
    assert_eq!(
        outcome_json(&settled_parent),
        json!({ "status": "completed", "result": "parent" })
    );
    assert!(settled_parent.abort_requested);
    assert!(!w.logged("abort:parent"));
    assert_eq!(
        harness.abort_task(parent, &context()).await.unwrap(),
        AbortTaskResult::Terminal
    );
    harness.close(&context()).await.unwrap();
}

// ─── Abort order ────────────────────────────────────────────────────────────

/// Script `name` to spawn `child` and wait on it; its abort handler records the child's status first.
fn chain(
    world: &Arc<World>,
    harness: &Arc<Mutex<Option<Harness>>>,
    name: &str,
    child: &str,
) -> Arc<Mutex<Option<TaskId>>> {
    let found = slot::<TaskId>();
    let (spawner, sink, child) = (world.clone(), found.clone(), child.to_string());
    let (logger, read, harness, label) = (
        world.clone(),
        found.clone(),
        harness.clone(),
        name.to_string(),
    );
    world.script(
        name,
        Behavior::default()
            .run(move |runtime, ctx| {
                let (world, sink, child) = (spawner.clone(), sink.clone(), child.clone());
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let id = world.spawn(&tx, owner, &child).await?;
                                *sink.lock() = Some(id);
                                Ok(Some(wait_on(vec![id], JoinPolicy::AllSettled, 1)))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(move |runtime, ctx| {
                let (world, id, harness, label) = (
                    logger.clone(),
                    taken(&read),
                    harness.lock().clone().unwrap(),
                    label.clone(),
                );
                async move {
                    let seen = status(&harness, id).await;
                    world.push(format!("{label} saw {seen}"));
                    commit_next(&runtime, &ctx, aborted_plain()).await
                }
            }),
    );
    found
}

#[tokio::test]
async fn runs_abort_handlers_bottom_up_across_three_levels_each_after_the_level_below_is_terminal()
{
    let w = World::new();
    let opened = slot::<Harness>();
    let b = chain(&w, &opened, "a", "b");
    let c = chain(&w, &opened, "b", "c");
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    *opened.lock() = Some(harness.clone());
    let a = w.start(&root, "a").await;
    until(|| async { w.logged("run:c") }).await;
    let c = taken(&c);
    until(|| async { status(&harness, c).await == "running" }).await;
    assert_eq!(
        harness.abort_task(a, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    assert_eq!(outcome_of(&harness, a).await, "aborted");
    let order: Vec<String> = w
        .log()
        .into_iter()
        .filter(|line| line.starts_with("abort:") || line.contains(" saw "))
        .collect();
    assert_eq!(
        order,
        [
            "abort:c",
            "abort:b",
            "b saw terminal",
            "abort:a",
            "a saw terminal"
        ]
    );
    assert!(b.lock().is_some());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn does_not_wait_for_a_task_it_waits_on_but_does_not_own() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let other = w.start(&root, "other").await;
    w.script(
        "parent",
        Behavior::default().run(move |runtime, ctx| async move {
            commit_next(
                &runtime,
                &ctx,
                wait_on(vec![other], JoinPolicy::AllSettled, 1),
            )
            .await
        }),
    );
    let parent = w.start(&root, "parent").await;
    until(|| async { status(&harness, parent).await == "waiting" }).await;
    harness.abort_task(parent, &context()).await.unwrap();
    assert_eq!(outcome_of(&harness, parent).await, "aborted");
    assert_eq!(status(&harness, other).await, "running");
    w.open("other");
    harness.close(&context()).await.unwrap();
}

/// A run that spawns `child` and waits on it; the child's ID lands in the returned slot.
fn spawn_and_wait(world: &Arc<World>, child: &str) -> (Behavior, Arc<Mutex<Option<TaskId>>>) {
    let found = slot();
    let (world, sink, child) = (world.clone(), found.clone(), child.to_string());
    let behavior = Behavior::default().run(move |runtime, ctx| {
        let (world, sink, child) = (world.clone(), sink.clone(), child.clone());
        async move {
            let owner = runtime.task_id().erase();
            runtime
                .commit(
                    move |tx, _| async move {
                        let id = world.spawn(&tx, owner, &child).await?;
                        *sink.lock() = Some(id);
                        Ok(Some(wait_on(vec![id], JoinPolicy::AllSettled, 1)))
                    },
                    &ctx,
                )
                .await
        }
    });
    (behavior, found)
}

#[tokio::test]
async fn reports_an_abort_marked_task_as_waiting_for_its_live_owned_work() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    w.script("child", slow_abort(&w, "child"));
    let (behavior, child) = spawn_and_wait(&w, "child");
    w.script("parent", behavior);
    let parent = w.start(&root, "parent").await;
    until(|| async { status(&harness, parent).await == "waiting" }).await;
    harness.abort_task(parent, &context()).await.unwrap();
    until(|| async { w.logged("abort:child") }).await;
    let child = taken(&child);
    assert!(matches!(
        inspected(&harness, parent).await,
        Some(TaskInspectionState::Waiting { on }) if on == vec![child]
    ));
    w.open("abort.child");
    assert_eq!(outcome_of(&harness, parent).await, "aborted");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn faults_an_abort_handler_that_tries_to_wait() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    w.script(
        "parent",
        Behavior::default().abort(|runtime, ctx| async move {
            commit_next(
                &runtime,
                &ctx,
                wait_on(Vec::new(), JoinPolicy::AllSettled, 1),
            )
            .await
        }),
    );
    let parent = w.start(&root, "parent").await;
    until(|| async { w.logged("run:parent") }).await;
    harness.abort_task(parent, &context()).await.unwrap();
    let outcome = outcome_json(&settle(&harness, parent).await);
    assert_eq!(outcome["status"], "faulted");
    assert!(
        outcome["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot wait")
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn orphans_a_blocked_task_only_after_its_owned_work_drained() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let world = w.clone();
    let (owner, child) = root
        .commit(
            move |tx| async move {
                let owner = tx
                    .create_task(&unregistered_task(), input("owner"), own_conversation())
                    .await?
                    .erase();
                Ok((owner, world.spawn(&tx, owner, "child").await?))
            },
            &context(),
        )
        .await
        .unwrap();
    until(|| async { w.logged("run:child") }).await;
    assert_eq!(
        harness.abort_task(owner, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    assert_eq!(outcome_of(&harness, child).await, "aborted");
    assert_eq!(
        outcome_json(&settle(&harness, owner).await),
        json!({ "status": "orphaned", "reason": "missing_task" })
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cascades_through_task_and_conversation_edges_bottom_up() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let (a, _) = spawn_and_wait(&w, "b");
    w.script("a", a);
    let x = slot::<TaskId>();
    {
        let (world, sink) = (w.clone(), x.clone());
        w.script(
            "b",
            resume_until_aborted(Behavior::default().run(move |runtime, ctx| {
                let (world, sink) = (world.clone(), sink.clone());
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let conversation = tx.create_conversation(owned_by(owner)).await?;
                                *sink.lock() =
                                    Some(world.create_in(&tx, conversation.id, "x").await?);
                                Ok(Some(resume_next(1)))
                            },
                            &ctx,
                        )
                        .await
                }
            })),
        );
    }
    let a = w.start(&root, "a").await;
    until(|| async { w.logged("run:x") }).await;
    harness.abort_task(a, &context()).await.unwrap();
    assert_eq!(outcome_of(&harness, a).await, "aborted");
    assert_eq!(outcome_of(&harness, taken(&x)).await, "aborted");
    assert_eq!(w.aborts(), ["abort:x", "abort:b", "abort:a"]);
    harness.close(&context()).await.unwrap();
}

// ─── Background and terminal owners ─────────────────────────────────────────

fn crossing() -> ConversationAbortOptions {
    ConversationAbortOptions { background: true }
}

#[tokio::test]
async fn keeps_background_work_through_conversation_abort_and_background_true_aborts_it_and_waits()
{
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let foreground = w.start(&root, "foreground").await;
    let found: Arc<Mutex<Vec<TaskId>>> = Arc::default();
    {
        let (world, sink) = (w.clone(), found.clone());
        w.script(
            "background",
            Behavior::default().run(move |runtime, ctx| {
                let (world, sink) = (world.clone(), sink.clone());
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let child = world.spawn(&tx, owner, "child").await?;
                                let conversation = tx.create_conversation(owned_by(owner)).await?;
                                let below = world.create_in(&tx, conversation.id, "below").await?;
                                sink.lock().extend([child, below]);
                                Ok(Some(wait_on(vec![child], JoinPolicy::AllSettled, 1)))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
        );
    }
    let background = w.start_with(&root, "background", background()).await;
    until(|| async { w.logged("run:below") && w.logged("run:child") }).await;
    root.abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    assert_eq!(outcome_of(&harness, foreground).await, "aborted");
    let (child, below) = {
        let found = found.lock();
        (found[0], found[1])
    };
    let ids = [background, child, below];
    for id in ids {
        assert_ne!(status(&harness, id).await, "terminal");
    }
    root.abort(&context(), crossing()).await.unwrap();
    for id in ids {
        assert_eq!(status(&harness, id).await, "terminal");
    }
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn never_cascades_from_a_terminal_owner_an_aborted_subagents_conversation_runs_new_work_normally()
 {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let (behavior, conversation) = host_conversation();
    w.script("agent", resume_until_aborted(behavior));
    let agent = w.start(&root, "agent").await;
    until(|| async { w.logged("resume:agent") }).await;
    harness.abort_task(agent, &context()).await.unwrap();
    assert_eq!(outcome_of(&harness, agent).await, "aborted");
    let id = taken(&conversation);
    let child = harness.conversation(id, &context()).await.unwrap().unwrap();
    let question = w.start(&child, "question").await;
    w.open("question");
    assert_eq!(outcome_of(&harness, question).await, "completed");
    harness.close(&context()).await.unwrap();
}

// ─── Recovery ───────────────────────────────────────────────────────────────

struct Seeded {
    parent: TaskId,
    children: Vec<TaskId>,
}

/// Write records a crash could leave, without a Harness: `parent` with children named `children`; then each
/// `(id, change)` from `edit` replaces a record through the internal task write.
async fn seed(
    world: &Arc<World>,
    storage: &Arc<ControlledStorage>,
    children: &[&str],
    edit: impl FnOnce(&Seeded) -> Vec<(TaskId, Box<dyn FnOnce(TaskRecord) -> TaskRecord + Send>)>,
) -> Seeded {
    let session = create_session(storage.clone());
    let world = world.clone();
    let names: Vec<String> = children.iter().map(|name| name.to_string()).collect();
    let (parent, children) = session
        .commit(
            move |tx| async move {
                let root = tx
                    .create_conversation(ConversationOwnership::Ownerless)
                    .await?;
                let parent = world.create_in(&tx, root.id, "parent").await?;
                let mut ids = Vec::new();
                for name in &names {
                    ids.push(world.spawn(&tx, parent, name).await?);
                }
                Ok((parent, ids))
            },
            &context(),
        )
        .await
        .unwrap();
    let seeded = Seeded { parent, children };
    for (id, change) in edit(&seeded) {
        session
            .commit(
                move |tx| async move {
                    let record = tx.task(id).await?.unwrap();
                    tx.set_task(change(record))
                },
                &context(),
            )
            .await
            .unwrap();
    }
    session.close(&context()).await.unwrap();
    seeded
}

type Edit = (TaskId, Box<dyn FnOnce(TaskRecord) -> TaskRecord + Send>);

fn set_state(id: TaskId, state: TaskState) -> Edit {
    (id, Box::new(move |record| TaskRecord { state, ..record }))
}

fn waiting_on(on: &[TaskId], policy: JoinPolicy) -> TaskState {
    TaskState::Waiting {
        checkpoint: json!({ "phase": "resume", "round": 1 }),
        on: on.to_vec(),
        policy,
    }
}

fn completed_done() -> TaskState {
    TaskState::Terminal {
        outcome: TaskOutcome::Completed {
            result: json!("done"),
        },
    }
}

fn held_parent() -> TaskState {
    TaskState::Completing {
        outcome: TaskOutcome::Completed {
            result: json!("parent"),
        },
    }
}

fn persistent() -> Arc<ControlledStorage> {
    Arc::new(ControlledStorage::persistent())
}

#[tokio::test]
async fn resumes_a_parent_whose_awaited_children_finished_before_the_crash() {
    let w = World::new();
    let storage = persistent();
    let seeded = seed(&w, &storage, &["c1"], |seeded| {
        vec![
            set_state(seeded.children[0], completed_done()),
            set_state(
                seeded.parent,
                waiting_on(&seeded.children, JoinPolicy::AllSettled),
            ),
        ]
    })
    .await;
    let Opened { harness, .. } = w.open_nodes(storage.clone()).await;
    assert_eq!(outcome_of(&harness, seeded.parent).await, "completed");
    assert_eq!(w.log(), ["resume:parent"]);
    assert_eq!(seeded.children.len(), 1);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn marks_fail_fast_siblings_a_crash_left_unmarked() {
    let w = World::new();
    let storage = persistent();
    let seeded = seed(&w, &storage, &["c1", "c2"], |seeded| {
        vec![
            set_state(
                seeded.children[0],
                TaskState::Terminal {
                    outcome: TaskOutcome::Failed {
                        error: crate::durable::types::TaskOutcomeError {
                            message: "declined".into(),
                            detail: None,
                        },
                        result: None,
                    },
                },
            ),
            set_state(
                seeded.parent,
                waiting_on(&seeded.children, JoinPolicy::FailFast),
            ),
        ]
    })
    .await;
    let Opened { harness, .. } = w.open_nodes(storage.clone()).await;
    assert_eq!(outcome_of(&harness, seeded.children[1]).await, "aborted");
    assert_eq!(outcome_of(&harness, seeded.parent).await, "completed");
    assert!(!w.logged("run:c2"));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn finalizes_a_held_outcome_whose_work_drained_before_the_crash_and_keeps_one_whose_work_lives()
 {
    let w = World::new();
    let storage = persistent();
    let seeded = seed(&w, &storage, &["c1"], |seeded| {
        vec![set_state(seeded.parent, held_parent())]
    })
    .await;
    let opened = w.open_nodes(storage.clone()).await;
    until(|| async { w.logged("run:c1") }).await;
    assert_eq!(state(&opened.harness, seeded.parent).await, held_parent());
    opened.harness.close(&context()).await.unwrap();

    let opened = w.open_nodes(storage.clone()).await;
    w.open("c1");
    assert_eq!(
        outcome_of(&opened.harness, seeded.children[0]).await,
        "completed"
    );
    assert_eq!(
        outcome_of(&opened.harness, seeded.parent).await,
        "completed"
    );
    opened.harness.close(&context()).await.unwrap();

    let drained = persistent();
    let second = seed(&w, &drained, &["c1"], |seeded| {
        vec![
            set_state(seeded.children[0], completed_done()),
            set_state(seeded.parent, held_parent()),
        ]
    })
    .await;
    let opened = w.open_nodes(drained.clone()).await;
    assert_eq!(
        outcome_of(&opened.harness, second.parent).await,
        "completed"
    );
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn resumes_a_bottom_up_abort_a_crash_interrupted_before_the_cascade() {
    let w = World::new();
    let storage = persistent();
    let seeded = seed(&w, &storage, &["c1"], |seeded| {
        let state = waiting_on(&seeded.children, JoinPolicy::AllSettled);
        vec![(
            seeded.parent,
            Box::new(move |record| TaskRecord {
                abort_requested: true,
                state,
                ..record
            }) as Box<dyn FnOnce(TaskRecord) -> TaskRecord + Send>,
        )]
    })
    .await;
    let Opened { harness, .. } = w.open_nodes(storage.clone()).await;
    assert_eq!(outcome_of(&harness, seeded.parent).await, "aborted");
    assert_eq!(outcome_of(&harness, seeded.children[0]).await, "aborted");
    assert_eq!(w.aborts(), ["abort:c1", "abort:parent"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reopens_a_checkout_waiting_on_live_payments_and_finishes_it() {
    let w = World::new();
    let storage = persistent();
    let found = parent_of(&w, "checkout", &["p1", "p2"], JoinPolicy::FailFast);
    let opened = w.open_nodes(storage.clone()).await;
    let parent = w.start(&opened.root, "checkout").await;
    until(|| async { status(&opened.harness, parent).await == "waiting" }).await;
    until(|| async { w.logged("run:p1") && w.logged("run:p2") }).await;
    opened.harness.close(&context()).await.unwrap();

    let opened = w.open_nodes(storage.clone()).await;
    let TaskState::Waiting { on, .. } = state(&opened.harness, parent).await else {
        panic!("checkout is not waiting");
    };
    *found.ids.lock() = on;
    w.open("p1");
    w.open("p2");
    assert_eq!(outcome_of(&opened.harness, parent).await, "completed");
    assert_eq!(found.outcomes(), ["completed", "completed"]);
    opened.harness.close(&context()).await.unwrap();
}

// ─── Definitions and waits ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VersionedInput {
    on: Vec<TaskId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
enum VersionedCheckpoint {
    Wait,
    Resume,
    Migrated,
}

/// A waiter of its own kind, so its definition can be removed or replaced.
fn versioned(version: u32) -> Task<VersionedInput, VersionedCheckpoint, String> {
    let mut definition = TaskDefinition::new("test.versioned", version, |_: &VersionedInput| {
        VersionedCheckpoint::Wait
    })
    .phase("wait", |task, runtime, ctx| async move {
        let on = task.input.on;
        runtime
            .commit(
                move |_, _| async move {
                    Ok(Some(NextTaskState::waiting(
                        VersionedCheckpoint::Resume,
                        on,
                        JoinPolicy::AllSettled,
                    )))
                },
                &ctx,
            )
            .await
    })
    .phase("resume", |_, runtime, ctx| async move {
        runtime
            .commit(
                |_, _| async { Ok(Some(NextTaskState::completed("v1"))) },
                &ctx,
            )
            .await
    })
    .phase("migrated", |_, runtime, ctx| async move {
        runtime
            .commit(
                |_, _| async { Ok(Some(NextTaskState::completed("v2"))) },
                &ctx,
            )
            .await
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
    });
    if version == 2 {
        definition = definition.migrate(|input, _, _| {
            Ok((
                serde_json::from_value(input)?,
                VersionedCheckpoint::Migrated,
            ))
        });
    }
    define_task(definition)
}

async fn start_versioned(root: &Conversation, on: TaskId) -> TaskId {
    root.commit(
        move |tx| async move {
            Ok(tx
                .create_task(
                    &versioned(1),
                    VersionedInput { on: vec![on] },
                    own_conversation(),
                )
                .await?
                .erase())
        },
        &context(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn keeps_a_waiting_task_blocked_without_its_definition_and_resumes_it_migrated_under_a_newer_one()
 {
    let w = World::new();
    let Opened {
        harness,
        root,
        registry,
        ..
    } = w.open_nodes(memory()).await;
    let registration = add_task(&registry, versioned(1).any());
    let other = w.start(&root, "other").await;
    let waiter = start_versioned(&root, other).await;
    until(|| async { status(&harness, waiter).await == "waiting" }).await;
    registration.dispose();
    w.open("other");
    settle(&harness, other).await;
    assert!(matches!(
        inspected(&harness, waiter).await,
        Some(TaskInspectionState::Blocked {
            reason: BlockedReason::MissingTask,
            ..
        })
    ));
    assert_eq!(status(&harness, waiter).await, "waiting");
    add_task(&registry, versioned(2).any());
    assert_eq!(
        outcome_json(&settle(&harness, waiter).await),
        json!({ "status": "completed", "result": "v2" })
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn orphans_an_aborted_waiting_task_without_its_definition_at_once_leaving_the_task_it_waits_on_running()
 {
    let w = World::new();
    let Opened {
        harness,
        root,
        registry,
        ..
    } = w.open_nodes(memory()).await;
    let registration = add_task(&registry, versioned(1).any());
    let other = w.start(&root, "other").await;
    let waiter = start_versioned(&root, other).await;
    until(|| async { status(&harness, waiter).await == "waiting" }).await;
    registration.dispose();
    assert_eq!(
        harness.abort_task(waiter, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    assert_eq!(
        to_json(&state(&harness, waiter).await)["outcome"],
        json!({ "status": "orphaned", "reason": "missing_task" })
    );
    assert_eq!(status(&harness, other).await, "running");
    w.open("other");
    harness.close(&context()).await.unwrap();
}

/// Completes with "held" after spawning a child, so it holds; `migrate` always fails.
fn holder(world: &Arc<World>, version: u32) -> Task<(), JsonValue, String> {
    let world = world.clone();
    define_task(
        TaskDefinition::new("test.holder", version, |_: &()| json!({ "phase": "run" }))
            .phase("run", move |task, runtime, ctx| {
                let world = world.clone();
                async move {
                    let owner = task.id.erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                world.spawn(&tx, owner, "child").await?;
                                Ok(None)
                            },
                            &ctx,
                        )
                        .await?;
                    runtime
                        .commit(
                            |_, _| async { Ok(Some(NextTaskState::completed("held"))) },
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
            })
            .migrate(|_, _, _| Err(Error::message("never migrates"))),
    )
}

#[tokio::test]
async fn never_migrates_a_held_outcome_a_newer_definition_leaves_it_and_its_version_alone() {
    let w = World::new();
    let Opened {
        harness,
        root,
        registry,
        reports,
    } = w.open_nodes(memory()).await;
    add_task(&registry, holder(&w, 1).any());
    let task = holder(&w, 1);
    let parent = root
        .commit(
            move |tx| async move { Ok(tx.create_task(&task, (), own_conversation()).await?.erase()) },
            &context(),
        )
        .await
        .unwrap();
    until(|| async { status(&harness, parent).await == "completing" }).await;
    // The same extension name replaces the old one in place.
    add_task(&registry, holder(&w, 2).any());
    w.open("child");
    let settled_parent = settle(&harness, parent).await;
    assert_eq!(
        outcome_json(&settled_parent),
        json!({ "status": "completed", "result": "held" })
    );
    assert_eq!(settled_parent.version, 1);
    assert!(reports.messages().is_empty(), "{:?}", reports.messages());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn treats_a_held_task_as_live_outcomes_rejects_and_a_task_waiting_on_it_resumes_at_its_final_commit()
 {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let (behavior, _) = spawn_and_finish(&w, "child");
    w.script("held", behavior);
    let held = w.start(&root, "held").await;
    until(|| async { status(&harness, held).await == "completing" }).await;
    let rejection = slot::<String>();
    {
        let sink = rejection.clone();
        w.script(
            "reader",
            Behavior::default().run(move |runtime, ctx| {
                let sink = sink.clone();
                async move {
                    let read = runtime.outcomes(&[held], &ctx).await;
                    *sink.lock() = Some(match read {
                        Ok(_) => "resolved".into(),
                        Err(error) => error.to_string(),
                    });
                    commit_next(
                        &runtime,
                        &ctx,
                        wait_on(vec![held], JoinPolicy::AllSettled, 1),
                    )
                    .await
                }
            }),
        );
    }
    let reader = w.start(&root, "reader").await;
    until(|| async { status(&harness, reader).await == "waiting" }).await;
    let message = rejection.lock().clone().unwrap();
    assert!(
        message.contains(&format!("Task {held} is not terminal")),
        "{message}"
    );
    w.open("child");
    assert_eq!(outcome_of(&harness, reader).await, "completed");
    assert!(w.logged("resume:reader"));
    harness.close(&context()).await.unwrap();
}

// ─── Conversation abort and boundaries ──────────────────────────────────────

#[tokio::test]
async fn marks_a_held_completed_task_with_conversation_abort_the_work_below_is_aborted_the_outcome_stays()
 {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    let (behavior, child) = spawn_and_finish(&w, "child");
    w.script("held", behavior);
    let held = w.start(&root, "held").await;
    until(|| async { status(&harness, held).await == "completing" }).await;
    root.abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    assert_eq!(outcome_of(&harness, taken(&child)).await, "aborted");
    let settled_held = settle(&harness, held).await;
    assert_eq!(outcome_json(&settled_held)["status"], "completed");
    assert!(settled_held.abort_requested);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn marks_and_awaits_only_the_background_work_reached_when_background_true_is_admitted() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    w.script("first", slow_abort(&w, "first"));
    let first = w.start_with(&root, "first", background()).await;
    until(|| async { w.logged("run:first") }).await;
    let aborting = {
        let root = root.clone();
        tokio::spawn(async move { root.abort(&context(), crossing()).await })
    };
    until(|| async { marked(&harness, first).await }).await;
    let later = w.start_with(&root, "later", background()).await;
    w.open("abort.first");
    aborting.await.unwrap().unwrap();
    assert_ne!(status(&harness, later).await, "terminal");
    assert!(!marked(&harness, later).await);
    harness.abort_task(later, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn stops_a_cascade_at_an_unmarked_background_task_but_not_at_a_marked_one() {
    let w = World::new();
    let Opened { harness, root, .. } = w.open_nodes(memory()).await;
    w.script("owner", slow_abort(&w, "owner"));
    let node = w.node.clone();
    let (owner, outer, background_task, inner) = root
        .commit(
            move |tx| async move {
                let owner = tx
                    .create_task(&node, input("owner"), own_conversation())
                    .await?
                    .erase();
                let outer = tx.create_conversation(owned_by(owner)).await?;
                let background_task = tx
                    .create_task(
                        &node,
                        input("background"),
                        CreateTaskOptions {
                            conversation_id: Some(outer.id),
                            ..background()
                        },
                    )
                    .await?
                    .erase();
                let inner = tx.create_conversation(owned_by(background_task)).await?;
                Ok((owner, outer.id, background_task, inner.id))
            },
            &context(),
        )
        .await
        .unwrap();
    until(|| async { w.logged("run:owner") && w.logged("run:background") }).await;
    // The owner's abort handler starts at once: background work is not its ordinary owned work.
    harness.abort_task(owner, &context()).await.unwrap();
    until(|| async { w.logged("abort:owner") }).await;
    let inner = harness
        .conversation(inner, &context())
        .await
        .unwrap()
        .unwrap();
    let outer = harness
        .conversation(outer, &context())
        .await
        .unwrap()
        .unwrap();
    let shielded = w.start(&inner, "shielded").await;
    let exposed = w.start(&outer, "exposed").await;
    assert_eq!(outcome_of(&harness, exposed).await, "aborted");
    assert!(!marked(&harness, shielded).await);
    // Marked directly, the background task cascades into its own subtree.
    harness
        .abort_task(background_task, &context())
        .await
        .unwrap();
    assert_eq!(outcome_of(&harness, shielded).await, "aborted");
    assert_eq!(outcome_of(&harness, background_task).await, "aborted");
    w.open("abort.owner");
    assert_eq!(outcome_of(&harness, owner).await, "aborted");
    harness.close(&context()).await.unwrap();
}

/// Reject, once, the first batch that finalizes the task in `target`.
fn reject_final_commit_once(storage: &ControlledStorage, target: Arc<Mutex<Option<TaskId>>>) {
    storage.filter_commits(Box::new(move |writes| {
        let mut target = target.lock();
        let finalizes = writes.iter().any(|write| {
            matches!(write, StorageWrite::Task { value }
                if Some(value.id) == *target && matches!(value.state, TaskState::Terminal { .. }))
        });
        if finalizes {
            *target = None;
            return Some(StorageRejected::new("rejected once").into());
        }
        None
    }));
}

fn rejected(reports: &Reports) -> bool {
    reports
        .errors()
        .iter()
        .any(|error| matches!(error, Error::StorageRejected(_)))
}

#[tokio::test]
async fn retries_a_finalization_the_storage_rejected_with_the_next_commit() {
    let w = World::new();
    let storage = Arc::new(ControlledStorage::new());
    let reject = slot::<TaskId>();
    reject_final_commit_once(&storage, reject.clone());
    let Opened {
        harness,
        root,
        reports,
        ..
    } = w.open_nodes(storage.clone()).await;
    let (behavior, child) = spawn_and_finish(&w, "child");
    w.script("parent", behavior);
    let parent = w.start(&root, "parent").await;
    until(|| async { status(&harness, parent).await == "completing" }).await;
    *reject.lock() = Some(parent);
    w.open("child");
    settle(&harness, taken(&child)).await;
    until(|| async { rejected(&reports) }).await;
    assert_eq!(status(&harness, parent).await, "completing");
    let id = root.id;
    root.commit(
        move |tx| async move {
            tx.append_entry(id, EntryDraft::new("note")).await?;
            Ok(())
        },
        &context(),
    )
    .await
    .unwrap();
    assert_eq!(outcome_of(&harness, parent).await, "completed");
    harness.close(&context()).await.unwrap();
}

// ─── Recovery through ownership edges ───────────────────────────────────────

#[tokio::test]
async fn shows_an_abort_marked_owner_waiting_for_work_in_its_owned_conversation_before_resume_after_reopen()
 {
    let w = World::new();
    let storage = persistent();
    let session = create_session(storage.clone());
    let world = w.clone();
    let (owner, inner) = session
        .commit(
            move |tx| async move {
                let conversation = tx
                    .create_conversation(ConversationOwnership::Ownerless)
                    .await?;
                let owner = world.create_in(&tx, conversation.id, "owner").await?;
                let child = tx.create_conversation(owned_by(owner)).await?;
                let inner = world.create_in(&tx, child.id, "inner").await?;
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
                    abort_requested: true,
                    ..record
                })
            },
            &context(),
        )
        .await
        .unwrap();
    session.close(&context()).await.unwrap();

    let (harness, _, _) =
        open_tasks(storage.clone(), vec![w.node.any()], TaskOptions::default()).await;
    assert!(matches!(
        inspected(&harness, owner).await,
        Some(TaskInspectionState::Waiting { on }) if on == vec![inner]
    ));
    harness.resume().unwrap();
    assert_eq!(outcome_of(&harness, owner).await, "aborted");
    assert_eq!(w.aborts(), ["abort:inner", "abort:owner"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_the_root_busy_after_reopen_for_work_below_a_child_tasks_owned_conversation() {
    let w = World::new();
    let storage = persistent();
    let inner = slot::<TaskId>();
    {
        let (world, sink) = (w.clone(), inner.clone());
        w.script(
            "child",
            Behavior::default().run(move |runtime, ctx| {
                let (world, sink) = (world.clone(), sink.clone());
                async move {
                    let owner = runtime.task_id().erase();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let conversation = tx.create_conversation(owned_by(owner)).await?;
                                *sink.lock() =
                                    Some(world.create_in(&tx, conversation.id, "inner").await?);
                                Ok(Some(resume_next(1)))
                            },
                            &ctx,
                        )
                        .await
                }
            }),
        );
    }
    parent_of(&w, "parent", &["child"], JoinPolicy::AllSettled);
    let opened = w.open_nodes(storage.clone()).await;
    let parent = w.start(&opened.root, "parent").await;
    until(|| async { w.logged("run:inner") }).await;
    opened.harness.close(&context()).await.unwrap();

    let opened = w.open_nodes(storage.clone()).await;
    assert!(!idle_settles(&opened.root).await);
    w.open("inner");
    opened.root.wait_for_idle(&context()).await.unwrap();
    assert_eq!(outcome_of(&opened.harness, parent).await, "completed");
    assert!(inner.lock().is_some());
    opened.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_a_finished_background_ancestor_a_boundary_after_reopen_which_background_true_crosses()
 {
    let w = World::new();
    let storage = persistent();
    let (behavior, conversation) = host_conversation();
    w.script("child", behavior);
    parent_of(&w, "background", &["child"], JoinPolicy::AllSettled);
    let opened = w.open_nodes(storage.clone()).await;
    let background = w.start_with(&opened.root, "background", background()).await;
    settle(&opened.harness, background).await;
    opened.harness.close(&context()).await.unwrap();

    let opened = w.open_nodes(storage.clone()).await;
    let id = taken(&conversation);
    let child = opened
        .harness
        .conversation(id, &context())
        .await
        .unwrap()
        .unwrap();
    let below = w.start(&child, "below").await;
    until(|| async { w.logged("run:below") }).await;
    opened.root.wait_for_idle(&context()).await.unwrap();
    opened
        .root
        .abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    assert_eq!(status(&opened.harness, below).await, "running");
    opened.root.abort(&context(), crossing()).await.unwrap();
    assert_eq!(outcome_of(&opened.harness, below).await, "aborted");
    opened.harness.close(&context()).await.unwrap();
}

// ─── Built-in tool rounds ───────────────────────────────────────────────────

fn empty_parameters() -> JsonValue {
    json!({ "type": "object", "properties": {} })
}

/// Writes output, signals `started`, then blocks until its invocation is cancelled.
struct Blocking {
    started: Deferred,
    registration: ToolRegistration,
}

fn blocking_tool(name: &str, mode: ToolExecutionMode) -> Blocking {
    let started = Deferred::default();
    let gate = started.clone();
    let mut registration = define_tool(
        name,
        format!("The {name} tool"),
        empty_parameters(),
        move |_, api, ctx| {
            let gate = gate.clone();
            async move {
                api.output("partial")?;
                gate.resolve();
                aborted(ctx.abort_signal().unwrap().clone()).await?;
                Ok(ToolExecutionResult::default())
            }
        },
    );
    registration.execution_mode = Some(mode);
    Blocking {
        started,
        registration,
    }
}

fn noop_tool() -> ToolRegistration {
    define_tool(
        "noop",
        "Does nothing",
        empty_parameters(),
        |_, _, _| async {
            Ok(ToolExecutionResult {
                content: Some(Vec::new()),
                ..ToolExecutionResult::default()
            })
        },
    )
}

fn tool_calls(calls: &[(&str, &str)]) -> FauxResponseStep {
    faux_assistant_message(
        calls
            .iter()
            .map(|(name, id)| faux_tool_call(*name, json!({}), Some(id)))
            .collect::<Vec<_>>(),
        FauxMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxMessageOptions::default()
        },
    )
    .into()
}

fn done() -> FauxResponseStep {
    faux_assistant_message("done", FauxMessageOptions::default()).into()
}

/// Usage that is not strict JSON, so the commit carrying it throws.
fn invalid_usage() -> Usage {
    let mut usage = Usage::default();
    usage.cost.total = f64::NAN;
    usage
}

async fn open_chat_in(
    storage: Arc<dyn Storage>,
    setup: &ChatSetup,
    models: Option<crate::models::Models>,
) -> (Harness, Conversation) {
    let (harness, root) = open_chat_with(
        storage,
        setup,
        ChatOptions {
            models,
            ..ChatOptions::default()
        },
    )
    .await;
    harness.resume().unwrap();
    (harness, root)
}

async fn submit(root: &Conversation) -> Submission {
    root.submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap()
}

async fn wait_settled(submission: &Submission) -> JsonValue {
    to_json(&submission.wait(&context()).await.unwrap())
}

async fn submission_settles(submission: &Submission) -> bool {
    let submission = submission.clone();
    settled(async move { submission.wait(&context()).await })
        .await
        .0
}

async fn scan(harness: &Harness, query: TaskQuery, limit: usize) -> Vec<TaskRecord> {
    harness
        .commit(
            move |tx| async move { tx.scan_tasks(query, limit, None).await },
            &context(),
        )
        .await
        .unwrap()
        .items
}

fn of_kind(kind: &str) -> TaskQuery {
    TaskQuery {
        kind: Some(kind.into()),
        ..TaskQuery::default()
    }
}

async fn run_of(harness: &Harness, root: &Conversation) -> Option<TaskId> {
    live_state(harness, root)
        .await
        .and_then(|live| live.run)
        .map(|run| run.task_id)
}

/// The tool results of the transcript, in their TS JSON form.
async fn tool_results(root: &Conversation) -> Vec<JsonValue> {
    all_entries(root)
        .await
        .into_iter()
        .filter(|entry| entry.kind == "pi.tool-result")
        .map(|entry| to_json(&entry.model.unwrap()[0]))
        .collect()
}

async fn note(root: &Conversation) {
    let id = root.id;
    root.commit(
        move |tx| async move {
            tx.append_entry(id, EntryDraft::new("note")).await?;
            Ok(())
        },
        &context(),
    )
    .await
    .unwrap();
}

/// A generation hook that starts a `Node` named `name`, owned by the generation, through the Harness in `opened`.
fn start_hooked(
    world: &Arc<World>,
    opened: &Arc<Mutex<Option<Harness>>>,
    api: HookApi,
    ctx: Context,
) -> BoxFuture<'static, Result<()>> {
    let (node, harness, owner) = (
        world.node.clone(),
        opened.lock().clone().unwrap(),
        api.task_id(),
    );
    Box::pin(async move {
        harness
            .commit(
                move |tx| async move {
                    tx.create_task(&node, input("hooked"), owned(owner)).await?;
                    Ok(())
                },
                &ctx,
            )
            .await
    })
}

fn after_tools_hook(world: &Arc<World>, opened: &Arc<Mutex<Option<Harness>>>) -> GenerationHooks {
    let (world, opened) = (world.clone(), opened.clone());
    GenerationHooks {
        after_tools: Some(Arc::new(move |_, _, api, ctx| {
            start_hooked(&world, &opened, api, ctx)
        })),
        ..GenerationHooks::default()
    }
}

#[tokio::test]
async fn owns_its_tool_tasks_waits_for_them_and_hands_the_run_to_a_conversation_owned_generation() {
    let setup = chat_setup();
    add_tool(&setup.registry, noop_tool());
    setup
        .faux
        .set_responses([tool_calls(&[("noop", "c1"), ("noop", "c2")]), done()]);
    let (harness, root) = open_chat_in(memory(), &setup, None).await;
    assert_eq!(wait_settled(&submit(&root).await).await["status"], "done");
    let tasks = scan(
        &harness,
        TaskQuery {
            conversation_id: Some(root.id),
            ..TaskQuery::default()
        },
        20,
    )
    .await;
    let generations: Vec<&TaskRecord> = tasks
        .iter()
        .filter(|task| task.kind == "pi.generation")
        .collect();
    let (first, second) = (generations[0], generations[1]);
    let owners: Vec<Option<TaskId>> = tasks
        .iter()
        .filter(|task| task.kind == "pi.tool")
        .map(|task| task.owner)
        .collect();
    assert_eq!(owners, [Some(first.id), Some(first.id)]);
    assert_eq!(first.owner, None);
    assert_eq!(second.owner, None);
    let assistant = all_entries(&root)
        .await
        .into_iter()
        .find(|entry| entry.kind == "pi.assistant")
        .unwrap();
    assert_eq!(
        to_json(&first.state),
        json!({
            "status": "terminal",
            "outcome": { "status": "completed", "result": { "entryId": to_json(&assistant.id) } },
        })
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn aborts_a_parallel_round_with_its_generation_tools_first_then_the_generation_with_every_result_written()
 {
    let setup = chat_setup();
    let one = blocking_tool("one", ToolExecutionMode::Parallel);
    let two = blocking_tool("two", ToolExecutionMode::Parallel);
    add_tool(&setup.registry, one.registration.clone());
    add_tool(&setup.registry, two.registration.clone());
    setup
        .faux
        .set_responses([tool_calls(&[("one", "c1"), ("two", "c2")])]);
    let (harness, root) = open_chat_in(memory(), &setup, None).await;
    let submission = submit(&root).await;
    one.started.wait().await;
    two.started.wait().await;
    let generation = run_of(&harness, &root).await.unwrap();
    harness.abort_task(generation, &context()).await.unwrap();
    let settled = wait_settled(&submission).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "aborted");
    let ids: Vec<JsonValue> = tool_results(&root)
        .await
        .iter()
        .map(|result| result["toolCallId"].clone())
        .collect();
    assert_eq!(ids, [json!("c1"), json!("c2")]);
    assert_eq!(to_json(&live_state(&harness, &root).await), json!({}));
    assert_eq!(outcome_of(&harness, generation).await, "aborted");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn answers_the_unstarted_calls_of_an_aborted_sequential_round_with_aborted_results_in_call_order()
 {
    let setup = chat_setup();
    let one = blocking_tool("one", ToolExecutionMode::Sequential);
    add_tool(&setup.registry, one.registration.clone());
    add_tool(
        &setup.registry,
        blocking_tool("two", ToolExecutionMode::Sequential).registration,
    );
    setup
        .faux
        .set_responses([tool_calls(&[("one", "c1"), ("two", "c2"), ("two", "c3")])]);
    let (harness, root) = open_chat_in(memory(), &setup, None).await;
    let submission = submit(&root).await;
    one.started.wait().await;
    let live = live_state(&harness, &root).await.unwrap();
    let started: Vec<bool> = live
        .tools
        .unwrap()
        .iter()
        .map(|slot| slot.task_id.is_some())
        .collect();
    assert_eq!(started, [true, false, false]);
    let generation = live.run.unwrap().task_id;
    harness.abort_task(generation, &context()).await.unwrap();
    let settled = wait_settled(&submission).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "aborted");
    let results = tool_results(&root).await;
    let ids: Vec<JsonValue> = results
        .iter()
        .map(|result| result["toolCallId"].clone())
        .collect();
    assert_eq!(ids, [json!("c1"), json!("c2"), json!("c3")]);
    let errors: Vec<JsonValue> = results
        .iter()
        .map(|result| result["isError"].clone())
        .collect();
    assert_eq!(errors, [json!(true), json!(true), json!(true)]);
    assert_eq!(
        results[1]["content"],
        json!([{ "type": "text", "text": "<harness>\n[error] Tool two was aborted\n</harness>" }])
    );
    let tools = scan(
        &harness,
        TaskQuery {
            conversation_id: Some(root.id),
            kind: Some("pi.tool".into()),
            ..TaskQuery::default()
        },
        20,
    )
    .await;
    assert_eq!(tools.len(), 1);
    harness.close(&context()).await.unwrap();
}

/// An event stream that records every event in its TS JSON form.
struct Listening {
    stream: AgentEventStream,
    events: Arc<Mutex<Vec<JsonValue>>>,
}

impl Listening {
    fn events(&self) -> Vec<JsonValue> {
        self.events.lock().clone()
    }

    fn types(&self, prefix: &str) -> Vec<String> {
        self.events()
            .iter()
            .filter_map(|event| event["type"].as_str())
            .filter(|kind| kind.starts_with(prefix))
            .map(str::to_string)
            .collect()
    }

    fn labels(&self) -> Vec<String> {
        labels(&self.events())
    }
}

async fn listen(harness: &Harness, conversation: &Conversation) -> Listening {
    let stream = watch_events(harness, conversation.id, &context())
        .await
        .unwrap();
    let events: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let sink = events.clone();
    stream
        .start(move |batch: EventBatch, _| {
            sink.lock().extend(batch.iter().map(to_json));
            Box::pin(async { Ok(()) })
        })
        .unwrap();
    Listening { stream, events }
}

/// Event types, with tool ends and message ends labelled by call ID and whether an entry came along.
fn labels(events: &[JsonValue]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event["type"].as_str()? {
            "tool_execution_end" => Some(format!(
                "end:{}:{}",
                event["toolCallId"].as_str().unwrap(),
                event.get("entry").is_some_and(|entry| !entry.is_null())
            )),
            "message_end" => {
                let message = &event["entry"]["model"][0];
                Some(if message["role"] == "toolResult" {
                    format!("result:{}", message["toolCallId"].as_str().unwrap())
                } else {
                    format!(
                        "message:{}",
                        message["role"].as_str().unwrap_or("undefined")
                    )
                })
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn ends_a_turn_at_the_generations_hold_before_its_successors_turn_starts() {
    let w = World::new();
    let setup = chat_setup();
    add_task(&setup.registry, w.node.any());
    add_tool(&setup.registry, noop_tool());
    // An extension's hook starts work owned by the generation, which holds it while the next turn runs.
    let opened = slot::<Harness>();
    add_hooks(
        &setup.registry,
        hook(&*GENERATION_TASK, after_tools_hook(&w, &opened)),
    );
    setup
        .faux
        .set_responses([tool_calls(&[("noop", "c1")]), done()]);
    let (harness, root) = open_chat_in(memory(), &setup, None).await;
    *opened.lock() = Some(harness.clone());
    let listening = listen(&harness, &root).await;
    wait_settled(&submit(&root).await).await;
    let first = scan(&harness, of_kind("pi.generation"), 1).await[0].id;
    assert_eq!(status(&harness, first).await, "completing");
    wait_for(|| async { listening.types("turn_").len() == 4 }).await;
    assert_eq!(
        listening.types("turn_"),
        ["turn_start", "turn_end", "turn_start", "turn_end"]
    );
    w.open("hooked");
    assert_eq!(outcome_of(&harness, first).await, "completed");
    wait_for(|| async { listening.types("turn_").len() == 4 }).await;
    listening.stream.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_run_control_with_a_faulted_generation_until_its_owned_work_drains_and_retries_a_rejected_final_commit()
 {
    let w = World::new();
    let setup = chat_setup();
    let models = proxy_models(
        &setup,
        ProxyOverrides {
            stream_simple: Some(Arc::new(|_, _, _| invalid_final_stream())),
            ..ProxyOverrides::default()
        },
    );
    add_task(&setup.registry, w.node.any());
    let opened = slot::<Harness>();
    {
        let (world, opened) = (w.clone(), opened.clone());
        add_hooks(
            &setup.registry,
            hook(
                &*GENERATION_TASK,
                GenerationHooks {
                    before_request: Some(Arc::new(move |_, api, ctx| {
                        let started = start_hooked(&world, &opened, api, ctx);
                        Box::pin(async move {
                            started.await?;
                            Ok(None)
                        })
                    })),
                    ..GenerationHooks::default()
                },
            ),
        );
    }
    w.script("hooked", slow_abort(&w, "hooked"));
    let storage = Arc::new(ControlledStorage::new());
    let reject = slot::<TaskId>();
    reject_final_commit_once(&storage, reject.clone());
    let (harness, root) = open_chat_in(storage.clone(), &setup, Some(models)).await;
    *opened.lock() = Some(harness.clone());
    let submission = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap();
    wait_for(|| async {
        match run_of(&harness, &root).await {
            Some(generation) => status(&harness, generation).await == "completing",
            None => false,
        }
    })
    .await;
    let generation = run_of(&harness, &root).await.unwrap();
    let held = to_json(&state(&harness, generation).await);
    assert_eq!(held["status"], "completing");
    assert_eq!(held["outcome"]["status"], "faulted");
    assert!(!submission_settles(&submission).await);
    assert_eq!(run_of(&harness, &root).await, Some(generation));
    // The faulted outcome is cancellation intent: the hooked work is aborted, then the run settles.
    wait_for(|| async { w.logged("abort:hooked") }).await;
    *reject.lock() = Some(generation);
    w.open("abort.hooked");
    // The final commit, with the run's cleanup, is rejected once: nothing of the cleanup lands.
    wait_for(|| async { rejected(&setup.reports) }).await;
    assert_eq!(run_of(&harness, &root).await, Some(generation));
    assert_eq!(entry_kinds(&root).await, ["pi.user"]);
    assert!(!submission_settles(&submission).await);
    // The next commit retries it; the partial becomes one aborted entry.
    note(&root).await;
    let settled = wait_settled(&submission).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "faulted");
    assert_eq!(to_json(&live_state(&harness, &root).await), json!({}));
    assert_eq!(
        entry_kinds(&root).await,
        ["pi.user", "note", "pi.assistant"]
    );
    harness.close(&context()).await.unwrap();
}

/// A stream whose final message is not strict JSON, so the classification commit throws and the task faults.
fn invalid_final_stream() -> crate::utils::event_stream::AssistantMessageEventStream {
    let partial = faux_assistant_message(
        "partial",
        FauxMessageOptions {
            stop_reason: Some(StopReason::Pending),
            ..FauxMessageOptions::default()
        },
    );
    let mut last = faux_assistant_message("final", FauxMessageOptions::default());
    last.usage = invalid_usage();
    scripted_stream(vec![AssistantMessageEvent::Start { partial }], 300, last)
}

// ─── Tool rounds and events ─────────────────────────────────────────────────

#[tokio::test]
async fn ends_an_aborted_sequential_rounds_unstarted_calls_with_their_result_entries_right_before_them()
 {
    let setup = chat_setup();
    let one = blocking_tool("one", ToolExecutionMode::Sequential);
    add_tool(&setup.registry, one.registration.clone());
    add_tool(
        &setup.registry,
        blocking_tool("two", ToolExecutionMode::Sequential).registration,
    );
    setup
        .faux
        .set_responses([tool_calls(&[("one", "c1"), ("two", "c2"), ("two", "c3")])]);
    let (harness, root) = open_chat_in(memory(), &setup, None).await;
    let listening = listen(&harness, &root).await;
    let submission = submit(&root).await;
    one.started.wait().await;
    let generation = run_of(&harness, &root).await.unwrap();
    harness.abort_task(generation, &context()).await.unwrap();
    wait_settled(&submission).await;
    wait_for(|| async { listening.labels().iter().any(|label| label == "result:c3") }).await;
    let labels: Vec<String> = listening
        .labels()
        .into_iter()
        .filter(|label| !label.starts_with("message:"))
        .collect();
    assert_eq!(
        labels,
        [
            "end:c1:true",
            "result:c1",
            "end:c2:true",
            "result:c2",
            "end:c3:true",
            "result:c3"
        ]
    );
    listening.stream.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn emits_one_turn_end_per_generation_also_for_a_stream_attached_while_it_holds() {
    let w = World::new();
    let setup = chat_setup();
    add_task(&setup.registry, w.node.any());
    add_tool(&setup.registry, noop_tool());
    let opened = slot::<Harness>();
    add_hooks(
        &setup.registry,
        hook(&*GENERATION_TASK, after_tools_hook(&w, &opened)),
    );
    setup
        .faux
        .set_responses([tool_calls(&[("noop", "c1")]), done()]);
    let (harness, root) = open_chat_in(memory(), &setup, None).await;
    *opened.lock() = Some(harness.clone());
    let early = listen(&harness, &root).await;
    wait_settled(&submit(&root).await).await;
    let late = listen(&harness, &root).await;
    w.open("hooked");
    let first = scan(&harness, of_kind("pi.generation"), 1).await[0].id;
    settle(&harness, first).await;
    note(&root).await;
    wait_for(|| async { !late.types("entry_appended").is_empty() }).await;
    assert_eq!(early.types("turn_end").len(), 2);
    assert_eq!(late.types("turn_end").len(), 0);
    early.stream.stop().await;
    late.stream.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn lets_a_tool_that_owns_live_work_finish_its_call_at_the_hold_while_the_generation_waits_for_its_final_commit()
 {
    let w = World::new();
    let setup = chat_setup();
    add_task(&setup.registry, w.node.any());
    let node = w.node.clone();
    add_tool(
        &setup.registry,
        define_tool(
            "delegate",
            "Starts work in a conversation it owns",
            empty_parameters(),
            move |_, api, ctx| {
                let node = node.clone();
                async move {
                    let owner = api.task_id();
                    api.commit(
                        move |tx| async move {
                            let child = tx.create_conversation(owned_by(owner)).await?;
                            tx.create_task(&node, input("sub"), in_conversation(child.id))
                                .await?;
                            Ok(())
                        },
                        &ctx,
                    )
                    .await?;
                    Ok(ToolExecutionResult::text("started"))
                }
            },
        ),
    );
    setup
        .faux
        .set_responses([tool_calls(&[("delegate", "c1")]), done()]);
    let (harness, root) = open_chat_in(memory(), &setup, None).await;
    let listening = listen(&harness, &root).await;
    let submission = submit(&root).await;
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.tools)
            .is_some_and(|tools| tools[0].status == SlotStatus::Done)
    })
    .await;
    let live = live_state(&harness, &root).await.unwrap();
    let slot = live.tools.unwrap()[0].clone();
    let tool = slot.task_id.unwrap();
    assert!(slot.entry.is_some());
    assert_eq!(status(&harness, tool).await, "completing");
    assert_eq!(status(&harness, live.run.unwrap().task_id).await, "waiting");
    wait_for(|| async {
        listening
            .labels()
            .iter()
            .any(|label| label == "end:c1:true")
    })
    .await;
    assert!(!submission_settles(&submission).await);
    w.open("sub");
    assert_eq!(wait_settled(&submission).await["status"], "done");
    assert_eq!(outcome_of(&harness, tool).await, "completed");
    listening.stream.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn holds_a_faulted_tools_slot_and_task_failed_until_the_work_it_owns_drained() {
    let w = World::new();
    let setup = chat_setup();
    add_task(&setup.registry, w.node.any());
    w.script("held", slow_abort(&w, "held"));
    let node = w.node.clone();
    add_tool(
        &setup.registry,
        define_tool(
            "broken",
            "Starts owned work, then returns a result that is not strict JSON",
            empty_parameters(),
            move |_, api, ctx| {
                let node = node.clone();
                async move {
                    let owner = api.task_id();
                    api.commit(
                        move |tx| async move {
                            tx.create_task(&node, input("held"), owned(owner)).await?;
                            Ok(())
                        },
                        &ctx,
                    )
                    .await?;
                    Ok(ToolExecutionResult {
                        content: Some(Vec::new()),
                        usage: Some(invalid_usage()),
                        ..ToolExecutionResult::default()
                    })
                }
            },
        ),
    );
    setup
        .faux
        .set_responses([tool_calls(&[("broken", "c1")]), done()]);
    let (harness, root) = open_chat_in(memory(), &setup, None).await;
    let listening = listen(&harness, &root).await;
    let submission = submit(&root).await;
    wait_for(|| async { w.logged("abort:held") }).await;
    let live = live_state(&harness, &root).await.unwrap();
    let slot = live.tools.unwrap()[0].clone();
    let tool = slot.task_id.unwrap();
    let held = to_json(&state(&harness, tool).await);
    assert_eq!(held["status"], "completing");
    assert_eq!(held["outcome"]["status"], "faulted");
    assert_ne!(slot.status, SlotStatus::Done);
    assert!(listening.types("task_failed").is_empty());
    w.open("abort.held");
    assert_eq!(wait_settled(&submission).await["status"], "done");
    assert_eq!(outcome_of(&harness, tool).await, "faulted");
    wait_for(|| async { !listening.types("task_failed").is_empty() }).await;
    assert!(
        listening
            .labels()
            .iter()
            .any(|label| label == "end:c1:false")
    );
    listening.stream.stop().await;
    harness.close(&context()).await.unwrap();
}
