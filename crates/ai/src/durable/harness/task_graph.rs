//! Port of durable `src/harness/task-graph.ts`: the live task graph mount (spec §9.5).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use futures::future::BoxFuture;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::Serialize;

use crate::chord::delta::{Op, Path, Seg};
use crate::chord::{
    AttachedReplicatedState, Context, JsonValue, replicated_state, without_abort_signal,
};
use crate::durable::errors::Result;
use crate::durable::ids::{ConversationId, TaskId};
use crate::durable::session::SessionImpl;
use crate::durable::session::observation::{CommittedStateSource, CommittedWatch, ObservedValue};
use crate::durable::types::{
    CommitChange, CommitPublication, ConversationQuery, JoinPolicy, Storage, TaskOutcome,
    TaskQuery, TaskRecord, TaskState, TaskStatus,
};

use super::util::{abort_error, closed_error, scan_all};

/// A live task's durable status without its checkpoint and outcome payloads (spec §9.5).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum TaskGraphState {
    Pending {
        phase: String,
    },
    Running {
        phase: String,
    },
    Waiting {
        phase: String,
        on: Vec<TaskId>,
        policy: JoinPolicy,
    },
    /// Outcome held until its ordinary owned work drains; `outcome` is its status.
    Completing {
        outcome: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskGraphNode {
    pub id: TaskId,
    pub kind: String,
    pub conversation_id: ConversationId,
    /// Owner task; absent for a conversation-owned task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<TaskId>,
    pub background: bool,
    pub abort_requested: bool,
    pub state: TaskGraphState,
    /// Conversations this task owns, in ID order.
    pub conversations: Vec<ConversationId>,
}

/// Every live task of the Session (spec §9.5), keyed by ID (serialized as its decimal string).
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct TaskGraph {
    #[serde(serialize_with = "serialize_tasks")]
    pub tasks: Arc<BTreeMap<TaskId, TaskGraphNode>>,
}

fn serialize_tasks<S: serde::Serializer>(
    tasks: &Arc<BTreeMap<TaskId, TaskGraphNode>>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    tasks.as_ref().serialize(serializer)
}

impl ObservedValue for TaskGraph {
    fn is_retired(&self) -> bool {
        false
    }

    fn root_value(&self) -> JsonValue {
        serde_json::to_value(self).expect("task graphs serialize")
    }
}

/// A task graph watch (`TaskGraphWatch`).
pub type TaskGraphWatch = CommittedWatch<TaskGraph>;

#[derive(Clone)]
enum Observer {
    State(CommittedStateSource<TaskGraph>),
    Watch(CommittedWatch<TaskGraph>),
}

impl Observer {
    fn advance(&self, value: TaskGraph, ops: Arc<[Op]>, context: Context) {
        match self {
            Self::State(source) => source.advance(value, ops, context),
            Self::Watch(watch) => watch.advance(value, ops, context),
        }
    }

    fn close_session(&self) {
        match self {
            Self::State(source) => source.close_session(),
            Self::Watch(watch) => watch.close_session(),
        }
    }
}

struct Mount {
    value: TaskGraph,
    observers: IndexMap<u64, Observer>,
}

type MountRef = Arc<Mutex<Mount>>;

const LIVE_STATUSES: [TaskStatus; 4] = [
    TaskStatus::Pending,
    TaskStatus::Running,
    TaskStatus::Waiting,
    TaskStatus::Completing,
];
const SCAN_PAGE_SIZE: usize = 256;

struct GraphInner {
    session: SessionImpl,
    mount: Mutex<Option<MountRef>>,
    next_observer: AtomicU64,
    closed: AtomicBool,
}

/// The Harness's task graph mount: built on the Session line by its first observer and dropped with its last. It
/// advances from the Session's commit publications, which are durable.
#[derive(Clone)]
pub struct TaskGraphView {
    inner: Arc<GraphInner>,
}

impl std::fmt::Debug for TaskGraphView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskGraphView").finish_non_exhaustive()
    }
}

impl TaskGraphView {
    pub fn new(session: SessionImpl) -> Result<Self> {
        let inner = Arc::new(GraphInner {
            session: session.clone(),
            mount: Mutex::new(None),
            next_observer: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        });
        let weak: Weak<GraphInner> = Arc::downgrade(&inner);
        let commits = weak.clone();
        let unsubscribe = session.subscribe_commits(move |publication, context| {
            let Some(inner) = commits.upgrade() else {
                return;
            };
            let mount = inner.mount.lock().clone();
            if let Some(mount) = mount {
                advance(&mount, publication, context);
            }
        })?;
        // The subscription lives as long as the Session.
        std::mem::forget(unsubscribe);
        let unsubscribe = session.subscribe_close(move || {
            let Some(inner) = weak.upgrade() else {
                return;
            };
            inner.closed.store(true, Ordering::SeqCst);
            let mount = inner.mount.lock().take();
            if let Some(mount) = mount {
                let observers: Vec<Observer> = mount.lock().observers.values().cloned().collect();
                for observer in observers {
                    observer.close_session();
                }
            }
        })?;
        std::mem::forget(unsubscribe);
        Ok(Self { inner })
    }

    /// A disposable read-only Chord state of the graph.
    pub async fn state(&self, context: &Context) -> Result<AttachedReplicatedState<TaskGraph>> {
        let (observer, detach) = self
            .attach(
                |value, release| Observer::State(CommittedStateSource::new(value, release)),
                context,
            )
            .await?;
        let Observer::State(source) = observer else {
            unreachable!("state() attaches a state source");
        };
        match replicated_state(&source, None) {
            Ok(state) => Ok(state),
            Err(error) => {
                detach();
                Err(error.into())
            }
        }
    }

    /// A serialized exact-frame watch of the graph; cancelling `context` stops it.
    pub async fn watch(&self, context: &Context) -> Result<TaskGraphWatch> {
        let (observer, _) = self
            .attach(
                |value, release| Observer::Watch(CommittedWatch::new(value, release, None)),
                context,
            )
            .await?;
        let Observer::Watch(watch) = observer else {
            unreachable!("watch() attaches a watch");
        };
        if let Some(signal) = context.abort_signal() {
            if let Err(reason) = signal.throw_if_aborted() {
                watch.cancel();
                return Err(abort_error(reason));
            }
            watch.observe_cancellation(signal)?;
        }
        Ok(watch)
    }

    /// Register an observer created from the current revision, atomically on the Session line.
    #[allow(clippy::type_complexity)]
    fn attach(
        &self,
        create: impl FnOnce(TaskGraph, Box<dyn FnOnce() + Send>) -> Observer + Send + 'static,
        context: &Context,
    ) -> BoxFuture<'static, Result<(Observer, Arc<dyn Fn() + Send + Sync>)>> {
        let inner = self.inner.clone();
        let context = context.clone();
        let job = async move {
            let existing = inner.mount.lock().clone();
            let mount = match existing {
                Some(mount) => mount,
                None => Arc::new(Mutex::new(Mount {
                    value: build(inner.session.storage(), &context).await?,
                    observers: IndexMap::new(),
                })),
            };
            let key = inner.next_observer.fetch_add(1, Ordering::SeqCst);
            let weak = Arc::downgrade(&inner);
            let detach_mount = mount.clone();
            let detach: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                let empty = {
                    let mut mount = detach_mount.lock();
                    mount.observers.shift_remove(&key);
                    mount.observers.is_empty()
                };
                if !empty {
                    return;
                }
                if let Some(inner) = weak.upgrade() {
                    let mut current = inner.mount.lock();
                    if current
                        .as_ref()
                        .is_some_and(|current| Arc::ptr_eq(current, &detach_mount))
                    {
                        *current = None;
                    }
                }
            });
            let value = mount.lock().value.clone();
            let release = detach.clone();
            let observer = create(value, Box::new(move || release()));
            // Close or cancellation may begin while the mount builds; register nothing then.
            if inner.closed.load(Ordering::SeqCst) {
                return Err(closed_error());
            }
            if let Some(signal) = context.abort_signal() {
                signal.throw_if_aborted().map_err(abort_error)?;
            }
            *inner.mount.lock() = Some(mount.clone());
            mount.lock().observers.insert(key, observer.clone());
            Ok((observer, detach))
        };
        self.inner.session.read_on_line(job)
    }
}

async fn build(storage: &Arc<dyn Storage>, context: &Context) -> Result<TaskGraph> {
    let mut records: Vec<TaskRecord> = Vec::new();
    for status in LIVE_STATUSES {
        let query = TaskQuery {
            status: Some(status),
            ..TaskQuery::default()
        };
        records.extend(
            scan_all(|cursor| {
                let storage = storage.clone();
                let query = query.clone();
                let context = context.clone();
                async move {
                    storage
                        .scan_tasks(&query, SCAN_PAGE_SIZE, cursor.as_ref(), &context)
                        .await
                }
            })
            .await?,
        );
    }
    records.sort_by_key(|record| record.id);
    let mut tasks = BTreeMap::new();
    for record in records {
        let query = ConversationQuery {
            owner_task_id: Some(record.id),
            ..ConversationQuery::default()
        };
        let owned = scan_all(|cursor| {
            let storage = storage.clone();
            let context = context.clone();
            async move {
                storage
                    .scan_conversations(&query, SCAN_PAGE_SIZE, cursor.as_ref(), &context)
                    .await
            }
        })
        .await?;
        let mut conversations: Vec<ConversationId> = owned
            .into_iter()
            .map(|conversation| conversation.id)
            .collect();
        conversations.sort();
        tasks.insert(record.id, node_of(&record, conversations));
    }
    Ok(TaskGraph {
        tasks: Arc::new(tasks),
    })
}

fn task_path(id: TaskId) -> Path {
    vec![Seg::from("tasks"), Seg::from(id.0.to_string())]
}

/// Derive the mount's operations from one publication, apply them, and hand the revision to every observer.
fn advance(mount: &MountRef, publication: &CommitPublication, context: &Context) {
    let (value, ops, observers) = {
        let mut mount = mount.lock();
        let mut ops: Vec<Op> = Vec::new();
        let mut tasks: Option<BTreeMap<TaskId, TaskGraphNode>> = None;
        for change in &publication.changes {
            let CommitChange::Task(record) = change else {
                continue;
            };
            let tasks = tasks.get_or_insert_with(|| (*mount.value.tasks).clone());
            let previous = tasks.get(&record.id);
            if matches!(record.state, TaskState::Terminal { .. }) {
                if previous.is_none() {
                    continue;
                }
                ops.push(Op::D(task_path(record.id)));
                tasks.remove(&record.id);
                continue;
            }
            let next = node_of(
                record,
                previous
                    .map(|node| node.conversations.clone())
                    .unwrap_or_default(),
            );
            if previous == Some(&next) {
                continue;
            }
            ops.push(Op::S(
                task_path(record.id),
                serde_json::to_value(&next).expect("task graph nodes serialize"),
            ));
            tasks.insert(record.id, next);
        }
        // After the tasks, so a conversation created with its owner task in one commit finds the owner's node.
        // Change order within a publication is unspecified, so each owner's list is sorted again.
        let mut created: IndexMap<TaskId, Vec<ConversationId>> = IndexMap::new();
        for change in &publication.changes {
            let CommitChange::Conversation(conversation) = change else {
                continue;
            };
            let Some(owner) = conversation.owner else {
                continue;
            };
            let known = match &tasks {
                Some(tasks) => tasks.contains_key(&owner.task_id),
                None => mount.value.tasks.contains_key(&owner.task_id),
            };
            if known {
                created
                    .entry(owner.task_id)
                    .or_default()
                    .push(conversation.id);
            }
        }
        if !created.is_empty() {
            let tasks = tasks.get_or_insert_with(|| (*mount.value.tasks).clone());
            for (owner, ids) in created {
                let node = tasks.get_mut(&owner).expect("owner node is live");
                node.conversations.extend(ids);
                node.conversations.sort();
                let mut path = task_path(owner);
                path.push(Seg::from("conversations"));
                ops.push(Op::S(
                    path,
                    serde_json::to_value(&node.conversations).expect("conversation IDs serialize"),
                ));
            }
        }
        if ops.is_empty() {
            return;
        }
        if let Some(tasks) = tasks {
            mount.value = TaskGraph {
                tasks: Arc::new(tasks),
            };
        }
        let observers: Vec<Observer> = mount.observers.values().cloned().collect();
        (mount.value.clone(), Arc::<[Op]>::from(ops), observers)
    };
    let frame_context = without_abort_signal(context);
    for observer in observers {
        observer.advance(value.clone(), ops.clone(), frame_context.clone());
    }
}

fn node_of(record: &TaskRecord, conversations: Vec<ConversationId>) -> TaskGraphNode {
    TaskGraphNode {
        id: record.id,
        kind: record.kind.clone(),
        conversation_id: record.conversation_id,
        owner: record.owner,
        background: record.background,
        abort_requested: record.abort_requested,
        state: state_of(&record.state),
        conversations,
    }
}

fn state_of(state: &TaskState) -> TaskGraphState {
    match state {
        TaskState::Pending { checkpoint } => TaskGraphState::Pending {
            phase: phase_of(checkpoint),
        },
        TaskState::Running { checkpoint } => TaskGraphState::Running {
            phase: phase_of(checkpoint),
        },
        TaskState::Waiting {
            checkpoint,
            on,
            policy,
        } => TaskGraphState::Waiting {
            phase: phase_of(checkpoint),
            on: on.clone(),
            policy: *policy,
        },
        // Terminal records never reach here: they leave the graph.
        TaskState::Completing { outcome } | TaskState::Terminal { outcome } => {
            TaskGraphState::Completing {
                outcome: outcome_status(outcome).to_string(),
            }
        }
    }
}

/// The outcome's `status` discriminator.
pub fn outcome_status(outcome: &TaskOutcome) -> &'static str {
    match outcome {
        TaskOutcome::Completed { .. } => "completed",
        TaskOutcome::Failed { .. } => "failed",
        TaskOutcome::Aborted { .. } => "aborted",
        TaskOutcome::Orphaned { .. } => "orphaned",
        TaskOutcome::Faulted { .. } => "faulted",
    }
}

fn phase_of(checkpoint: &JsonValue) -> String {
    checkpoint
        .get("phase")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_string()
}
