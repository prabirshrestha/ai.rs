//! Port of durable `src/harness/scheduler.ts`: the durable task scheduler of one Harness, and the task runtime it
//! hands to phase handlers.
//!
//! Divergences from Pi:
//! - Microtasks (`queueMicrotask`) are `tokio::spawn`ed tasks; listeners hold weak references.
//! - Scheduler state lives behind one mutex that is never held across an await or while calling user code.
//! - A phase handler that panics is treated like one that threw: the task faults.
//! - A checkpoint whose `phase` has no handler faults with a `TypeError`, where TS would call `undefined`.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use indexmap::{IndexMap, IndexSet};
use parking_lot::Mutex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::oneshot;

use crate::chord::{
    AbortController, AbortSignal, Context, JsonValue, await_with_context, copy_json,
    with_abort_signal,
};
use crate::durable::documents::{DocAccess, RewindableDocAccess};
use crate::durable::entries::Entry;
use crate::durable::env::ExecutionEnv;
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, EntryId, TaskId};
use crate::durable::session::{
    DocumentWatch, SessionImpl, Transaction, TransactionScope, Unsubscribe,
};
use crate::durable::tasks::{AnyTask, NextTaskState, RunningTask};
use crate::durable::types::{
    CommitChange, CommitPublication, EntryRecord, JoinPolicy, JsonObject, SubmissionQuery,
    SubmissionStatus, TaskOutcome, TaskOutcomeError, TaskQuery, TaskRecord, TaskState, TaskStatus,
};
use crate::models::Models;

use super::agent::{agent_hooks, hook_handlers};
use super::context::read_context;
use super::harness::ConversationHandle;
use super::registry::{RegistryReader, RegistrySnapshot};
use super::types::{
    Agent, BlockedReason, ContextView, DocumentReader, ErasedReader, Scheduling, Settings,
    TaskInspection, TaskInspectionState,
};
use super::util::{Waiters, abort_error, abort_reason, closed_error, panic_error, scan_all};

const SCAN_PAGE_SIZE: usize = 256;
/// Longest single timer delay; longer sleeps wait in several steps.
const MAX_TIMER_DELAY: u64 = 2_147_483_647;
const LIVE_STATUSES: [TaskStatus; 4] = [
    TaskStatus::Pending,
    TaskStatus::Running,
    TaskStatus::Waiting,
    TaskStatus::Completing,
];

/// Resolve a conversation's agent against a snapshot; the runtime calls it at most once per phase.
pub type AgentResolver = Arc<
    dyn Fn(ConversationId, RegistrySnapshot, Context) -> BoxFuture<'static, Result<Agent>>
        + Send
        + Sync,
>;
/// Build a conversation's environment with `HarnessOptions.env`.
pub type EnvBuilder = Arc<
    dyn Fn(ConversationId, Context) -> BoxFuture<'static, Result<Option<Arc<dyn ExecutionEnv>>>>
        + Send
        + Sync,
>;
/// Harness cleanup staged in the commit that makes an outcome the scheduler wrote itself terminal.
pub type SettleOutcome = Arc<
    dyn Fn(Transaction, TaskRecord, TaskOutcome) -> BoxFuture<'static, Result<()>> + Send + Sync,
>;
/// Withdraw a conversation's queued inputs, for conversation abort and abort cascades.
pub type WithdrawInputs =
    Arc<dyn Fn(Transaction, ConversationId) -> BoxFuture<'static, Result<()>> + Send + Sync>;
/// Invocation-bound handle of an existing conversation, for task runtimes and tools.
pub type ConversationOpener = Arc<
    dyn Fn(
            ConversationId,
            InvocationBinding,
            Context,
        ) -> BoxFuture<'static, Result<Option<ConversationHandle>>>
        + Send
        + Sync,
>;

/// An invocation a conversation handle is bound to: its signal, and a check that fails once it ended.
#[derive(Clone)]
pub struct InvocationBinding {
    pub signal: AbortSignal,
    pub check: Arc<dyn Fn() -> Result<()> + Send + Sync>,
}

pub struct TaskSchedulerOptions {
    pub session: SessionImpl,
    pub registry: Arc<dyn RegistryReader>,
    pub models: Models,
    pub agent: AgentResolver,
    /// Resolve the settings; read at each access.
    pub settings: Arc<dyn Fn() -> Settings + Send + Sync>,
    pub env: EnvBuilder,
    pub now: Arc<dyn Fn() -> u64 + Send + Sync>,
    pub report: Arc<dyn Fn(Error) + Send + Sync>,
    pub settle_outcome: SettleOutcome,
    pub withdraw_inputs: WithdrawInputs,
    pub conversation: ConversationOpener,
    /// Context for scheduler commits and invocations; carries no caller cancellation.
    pub context: Context,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Run,
    Abort,
}

/// One in-memory execution of a task in run or abort mode.
pub(crate) struct Invocation {
    task_id: TaskId,
    conversation_id: ConversationId,
    mode: Mode,
    controller: AbortController,
    /// Context passed to handlers; cancelled by `controller`.
    context: Context,
    /// Watches acquired through the runtime; stopped at invocation end.
    watches: Mutex<Vec<DocumentWatch>>,
    ended: AtomicBool,
    done: Shared<BoxFuture<'static, ()>>,
    finish: Mutex<Option<oneshot::Sender<()>>>,
}

impl Invocation {
    fn ended(&self) -> bool {
        self.ended.load(Ordering::SeqCst)
    }

    fn ended_error(&self) -> Error {
        Error::message(format!("Task {} invocation has ended", self.task_id))
    }

    fn check(&self) -> Result<()> {
        if self.ended() {
            return Err(self.ended_error());
        }
        Ok(())
    }

    fn finish(&self) {
        if let Some(finish) = self.finish.lock().take() {
            let _ = finish.send(());
        }
    }
}

/// What a runtime reads for the phase handler it serves: the phase's snapshot and task, and its lazily resolved agent.
struct Phase {
    snapshot: RegistrySnapshot,
    task: AnyTask,
    agent: Option<Shared<BoxFuture<'static, Result<Agent>>>>,
}

struct Reservation {
    invocation: Arc<Invocation>,
    task: AnyTask,
    snapshot: RegistrySnapshot,
}

/// Step decision: continue with the next phase, end the invocation, or end it by writing `faulted`.
enum Decision {
    Continue,
    End,
    Fault(Error),
}

/// The immutable ownership fields of a task.
#[derive(Debug, Clone, Copy)]
struct TaskNode {
    conversation_id: ConversationId,
    owner: Option<TaskId>,
    background: bool,
}

/// Where a walk up the ownership tree continues: an owner task, or a conversation.
#[derive(Debug, Clone, Copy)]
enum Up {
    Task(TaskId),
    Conversation(ConversationId),
}

/// One step of a walk up.
#[derive(Debug, Clone, Copy)]
enum Step {
    Task(TaskId, TaskNode),
    Conversation(ConversationId),
    Unknown,
}

/// Candidate records a commit staged; they override committed records in ownership walks.
#[derive(Default)]
struct Overlay {
    tasks: IndexMap<TaskId, TaskRecord>,
    edges: HashMap<ConversationId, Option<TaskId>>,
}

/// Where ordinary ownership traversal starts: one conversation, or every ownerless conversation.
#[derive(Debug, Clone, Copy)]
enum Scope {
    Conversation(ConversationId),
    Roots,
}

#[derive(Default)]
struct State {
    live: IndexMap<TaskId, TaskRecord>,
    invocations: HashMap<TaskId, Arc<Invocation>>,
    failed_migrations: HashMap<TaskId, (AnyTask, Error)>,
    edges: HashMap<ConversationId, Option<TaskId>>,
    conversation_owners: HashSet<TaskId>,
    settled: HashMap<TaskId, TaskNode>,
    fail_fast_checks: IndexSet<TaskId>,
    reconcile_scheduled: bool,
    cascade_pending: bool,
    enabled: bool,
    closing: bool,
    dirty: bool,
    draining: bool,
}

/// Durable task scheduler of one Harness.
///
/// `live` mirrors every committed non-terminal task record: pending, running, waiting, and completing. The synchronous
/// commit listener updates it on the Session line, so code running on the line reads exactly the committed state.
///
/// Invariant: every task transition is decided and written by one callback serialized on the Session line. Handlers
/// and joins run off the line.
pub struct TaskScheduler {
    options: TaskSchedulerOptions,
    state: Mutex<State>,
    task_waiters: Arc<Waiters<TaskId, TaskRecord>>,
    /// Idle waiters by conversation; `None` waits for the whole Harness.
    idle_waiters: Arc<Waiters<Option<ConversationId>, ()>>,
    /// `#unsubscribeRegistry`. The commit and close subscriptions are never
    /// removed (as in Pi): commits of invocations finishing during `join()`
    /// are still observed.
    unsubscribe_registry: Mutex<Option<Unsubscribe>>,
    me: Weak<TaskScheduler>,
}

fn node_of(record: &TaskRecord) -> TaskNode {
    TaskNode {
        conversation_id: record.conversation_id,
        owner: record.owner,
        background: record.background,
    }
}

fn parent_of(node: &TaskNode) -> Up {
    match node.owner {
        Some(owner) => Up::Task(owner),
        None => Up::Conversation(node.conversation_id),
    }
}

fn record_parent(record: &TaskRecord) -> Up {
    parent_of(&node_of(record))
}

/// Whether the record holds or ends with an outcome other than `completed`.
fn failed_outcome(record: &TaskRecord) -> bool {
    match &record.state {
        TaskState::Completing { outcome } | TaskState::Terminal { outcome } => {
            !matches!(outcome, TaskOutcome::Completed { .. })
        }
        _ => false,
    }
}

/// A live owner's durable cancellation intent: its abort mark, or a held outcome other than `completed`.
fn cancellation_intent(record: &TaskRecord) -> bool {
    record.state.status() != TaskStatus::Terminal
        && (record.abort_requested || failed_outcome(record))
}

/// Replace a live record's state; memos disappear once an outcome is decided.
pub(crate) fn with_state(record: &TaskRecord, state: TaskState) -> TaskRecord {
    let decided = matches!(
        state,
        TaskState::Terminal { .. } | TaskState::Completing { .. }
    );
    TaskRecord {
        state,
        memos: if decided { None } else { record.memos.clone() },
        ..record.clone()
    }
}

fn overlay_of(tx: &Transaction) -> Overlay {
    Overlay {
        tasks: tx
            .staged_tasks()
            .into_iter()
            .map(|record| (record.id, record))
            .collect(),
        edges: tx
            .staged_conversations()
            .into_iter()
            .map(|record| (record.id, record.owner.map(|owner| owner.task_id)))
            .collect(),
    }
}

fn checkpoint_of(record: &TaskRecord) -> Option<&JsonValue> {
    record.state.checkpoint()
}

fn missing_migration(record: &TaskRecord, task: &AnyTask) -> Error {
    Error::message(format!(
        "Task {} version {} has no migration from {}",
        record.kind,
        task.definition().version,
        record.version
    ))
}

/// Whether a definition can take the task at reservation: same version, or newer with a migration.
fn can_reserve(task: &AnyTask, record: &TaskRecord) -> bool {
    let definition = task.definition();
    definition.version == record.version
        || (definition.version > record.version && definition.has_migrate())
}

/// Own memo entry only.
fn memo_of(record: Option<&TaskRecord>, name: &str) -> Option<JsonValue> {
    record?.memos.as_ref()?.get(name).cloned()
}

enum Fit {
    Task {
        task: AnyTask,
        migrates: bool,
    },
    Blocked {
        reason: BlockedReason,
        error: Option<Error>,
    },
}

#[allow(clippy::large_enum_variant)]
enum Resolution {
    Ready { task: AnyTask, record: TaskRecord },
    Blocked { reason: BlockedReason },
}

impl State {
    fn set_edge(&mut self, conversation_id: ConversationId, owner: Option<TaskId>) {
        self.edges.insert(conversation_id, owner);
        if let Some(owner) = owner {
            self.conversation_owners.insert(owner);
        }
    }

    /// Owner task of a conversation, `Some(None)` when ownerless, `None` while not loaded.
    fn edge(&self, id: ConversationId, overlay: Option<&Overlay>) -> Option<Option<TaskId>> {
        if let Some(edge) = overlay.and_then(|overlay| overlay.edges.get(&id)) {
            return Some(*edge);
        }
        self.edges.get(&id).copied()
    }

    fn node(&self, id: TaskId, overlay: Option<&Overlay>) -> Option<TaskNode> {
        overlay
            .and_then(|overlay| overlay.tasks.get(&id))
            .or_else(|| self.live.get(&id))
            .map(node_of)
            .or_else(|| self.settled.get(&id).copied())
    }

    /// Walk up from `start`: owner tasks and conversations, ending at an ownerless root or an edge not loaded yet.
    fn above(&self, start: Up, overlay: Option<&Overlay>) -> Vec<Step> {
        let mut steps = Vec::new();
        let mut at = Some(start);
        while let Some(up) = at {
            match up {
                Up::Task(id) => {
                    let Some(node) = self.node(id, overlay) else {
                        steps.push(Step::Unknown);
                        return steps;
                    };
                    steps.push(Step::Task(id, node));
                    at = Some(parent_of(&node));
                }
                Up::Conversation(id) => {
                    steps.push(Step::Conversation(id));
                    match self.edge(id, overlay) {
                        None => {
                            steps.push(Step::Unknown);
                            return steps;
                        }
                        Some(edge) => at = edge.map(Up::Task),
                    }
                }
            }
        }
        steps
    }

    /// Whether every owner above `start` is loaded.
    fn chain_known(&self, start: Up, overlay: Option<&Overlay>) -> bool {
        !self
            .above(start, overlay)
            .iter()
            .any(|step| matches!(step, Step::Unknown))
    }

    /// Live records, with the overlay's candidates replacing committed ones; terminal candidates are gone.
    fn live_records(&self, overlay: Option<&Overlay>) -> Vec<TaskRecord> {
        let mut records = Vec::new();
        for record in self.live.values() {
            let candidate = overlay
                .and_then(|overlay| overlay.tasks.get(&record.id))
                .unwrap_or(record);
            if candidate.state.status() != TaskStatus::Terminal {
                records.push(candidate.clone());
            }
        }
        if let Some(overlay) = overlay {
            for record in overlay.tasks.values() {
                if !self.live.contains_key(&record.id)
                    && record.state.status() != TaskStatus::Terminal
                {
                    records.push(record.clone());
                }
            }
        }
        records
    }

    /// Every task with live ordinary owned work (spec §5.5), mapped to that work.
    fn owned_live(&self, overlay: Option<&Overlay>) -> HashMap<TaskId, Vec<TaskId>> {
        let mut owned: HashMap<TaskId, Vec<TaskId>> = HashMap::new();
        for record in self.live_records(overlay) {
            if record.background {
                continue;
            }
            for step in self.above(record_parent(&record), overlay) {
                match step {
                    Step::Unknown => break,
                    Step::Conversation(_) => continue,
                    Step::Task(id, node) => {
                        owned.entry(id).or_default().push(record.id);
                        if node.background {
                            break;
                        }
                    }
                }
            }
        }
        owned
    }

    /// Whether ordinary traversal from `scope` reaches `start`. `None` while an edge is not loaded.
    fn in_scope(&self, start: Up, scope: Scope, cross_background: bool) -> Option<bool> {
        for step in self.above(start, None) {
            match step {
                Step::Unknown => return None,
                Step::Conversation(id) => {
                    if let Scope::Conversation(scope) = scope
                        && id == scope
                    {
                        return Some(true);
                    }
                }
                Step::Task(_, node) => {
                    if node.background && !cross_background {
                        return Some(false);
                    }
                }
            }
        }
        Some(matches!(scope, Scope::Roots))
    }

    /// Whether a live owner's cancellation intent reaches `start`.
    fn below_cancelled(&self, start: Up) -> bool {
        for step in self.above(start, None) {
            match step {
                Step::Unknown => return false,
                Step::Conversation(_) => continue,
                Step::Task(id, node) => {
                    if self.live.get(&id).is_some_and(cancellation_intent) {
                        return true;
                    }
                    if node.background {
                        return false;
                    }
                }
            }
        }
        false
    }

    /// No live non-background task in the scope; a task whose owner edges are not loaded yet counts as inside.
    fn idle(&self, conversation_id: Option<ConversationId>) -> bool {
        let scope = match conversation_id {
            None => Scope::Roots,
            Some(id) => Scope::Conversation(id),
        };
        !self.live.values().any(|record| {
            !record.background && self.in_scope(record_parent(record), scope, false) != Some(false)
        })
    }

    /// Live tasks a task waits for before its next invocation.
    fn waiting_on(&self, record: &TaskRecord, owned: &HashMap<TaskId, Vec<TaskId>>) -> Vec<TaskId> {
        if record.abort_requested {
            return owned.get(&record.id).cloned().unwrap_or_default();
        }
        match &record.state {
            TaskState::Waiting { on, .. } => on
                .iter()
                .filter(|id| self.live.contains_key(*id))
                .copied()
                .collect(),
            _ => Vec::new(),
        }
    }

    fn fit(&self, record: &TaskRecord, task: Option<&AnyTask>) -> Fit {
        let Some(task) = task else {
            return Fit::Blocked {
                reason: BlockedReason::MissingTask,
                error: None,
            };
        };
        let version = task.definition().version;
        if version == record.version {
            return Fit::Task {
                task: task.clone(),
                migrates: false,
            };
        }
        if version < record.version {
            return Fit::Blocked {
                reason: BlockedReason::TaskTooOld,
                error: None,
            };
        }
        if let Some((failed, error)) = self.failed_migrations.get(&record.id)
            && failed.ptr_eq(task)
        {
            return Fit::Blocked {
                reason: BlockedReason::MigrationFailed,
                error: Some(error.clone()),
            };
        }
        Fit::Task {
            task: task.clone(),
            migrates: true,
        }
    }
}

impl TaskScheduler {
    pub fn new(options: TaskSchedulerOptions) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            options,
            state: Mutex::default(),
            task_waiters: Arc::default(),
            idle_waiters: Arc::default(),
            unsubscribe_registry: Mutex::default(),
            me: me.clone(),
        })
    }

    fn arc(&self) -> Arc<Self> {
        self.me.upgrade().expect("scheduler is alive")
    }

    fn session(&self) -> &SessionImpl {
        &self.options.session
    }

    fn report(&self, error: Error) {
        (self.options.report)(error);
    }

    fn closing(&self) -> bool {
        self.state.lock().closing
    }

    /// Load live tasks and change surviving `running` tasks back to `pending`. Dispatches nothing.
    pub async fn open(&self, context: &Context) -> Result<()> {
        let weak = self.me.clone();
        let _commits = self.session().subscribe_commits(move |publication, _| {
            if let Some(scheduler) = weak.upgrade() {
                scheduler.observe(publication);
            }
        })?;
        let weak = self.me.clone();
        let _close = self.session().subscribe_close(move || {
            if let Some(scheduler) = weak.upgrade() {
                scheduler.seal();
            }
        })?;
        let weak = self.me.clone();
        let registry = self.options.registry.subscribe(Arc::new(move || {
            if let Some(scheduler) = weak.upgrade() {
                scheduler.kick();
            }
        }));
        *self.unsubscribe_registry.lock() = Some(registry);
        let me = self.arc();
        self.session()
            .commit_with(
                move |tx| async move {
                    // Every table read before the first write.
                    let mut scans = Vec::new();
                    for status in LIVE_STATUSES {
                        let query = TaskQuery {
                            status: Some(status),
                            ..TaskQuery::default()
                        };
                        scans.push(
                            scan_all(|cursor| tx.scan_tasks(query.clone(), SCAN_PAGE_SIZE, cursor))
                                .await?,
                        );
                    }
                    for records in scans {
                        for record in records {
                            let mut state = me.state.lock();
                            state.live.insert(record.id, record.clone());
                            if let TaskState::Running { checkpoint } = &record.state {
                                tx.set_task(with_state(
                                    &record,
                                    TaskState::Pending {
                                        checkpoint: checkpoint.clone(),
                                    },
                                ))?;
                            }
                            if let TaskState::Waiting {
                                policy: JoinPolicy::FailFast,
                                ..
                            } = &record.state
                            {
                                state.fail_fast_checks.insert(record.id);
                            }
                        }
                    }
                    Ok(())
                },
                context,
                TransactionScope::default(),
            )
            .await?;
        // Derive abort marks a crash left unapplied below cancelled owners, and finalize held outcomes.
        self.state.lock().cascade_pending = true;
        self.schedule_reconcile();
        Ok(())
    }

    /// Enable scheduling. Idempotent; the kick does nothing once closing.
    pub fn resume(&self) {
        self.state.lock().enabled = true;
        self.kick();
    }

    /// Wait for every invocation signalled by `seal()`. Writes nothing.
    pub async fn join(&self) {
        let done: Vec<_> = self
            .state
            .lock()
            .invocations
            .values()
            .map(|invocation| invocation.done.clone())
            .collect();
        futures::future::join_all(done).await;
    }

    /// Commit the abort mark, or settle a task that no registered definition can take as `orphaned` when nothing it
    /// owns is live, then join the run invocation seen on the line.
    pub async fn abort(&self, id: TaskId, context: &Context) -> Result<AbortTaskResult> {
        let me = self.arc();
        let (result, run) = self
            .session()
            .commit_with(
                move |tx| async move {
                    let Some(current) = tx.task(id).await? else {
                        return Err(Error::message(format!("Task {id} does not exist")));
                    };
                    if current.state.status() == TaskStatus::Terminal {
                        return Ok((AbortTaskResult::Terminal, None));
                    }
                    let invocation = me.state.lock().invocations.get(&id).cloned();
                    if invocation.is_none() && current.state.status() != TaskStatus::Completing {
                        me.load_scopes(false).await?;
                        let owned = me.state.lock().owned_live(None).contains_key(&id);
                        if !owned {
                            let snapshot = me.options.registry.snapshot();
                            if let Resolution::Blocked { reason } = me.resolve(&current, &snapshot)
                            {
                                me.terminate(
                                    &tx,
                                    &current,
                                    TaskOutcome::Orphaned {
                                        reason: reason.as_str().to_string(),
                                    },
                                )
                                .await?;
                                return Ok((AbortTaskResult::Marked, None));
                            }
                        }
                    }
                    if !current.abort_requested {
                        tx.set_task(TaskRecord {
                            abort_requested: true,
                            ..current.clone()
                        })?;
                    }
                    let run = invocation.filter(|invocation| invocation.mode == Mode::Run);
                    Ok((AbortTaskResult::Marked, run))
                },
                context,
                TransactionScope::default(),
            )
            .await?;
        // The commit listener signalled the run; join it.
        if let Some(run) = run {
            await_with_context(run.done.clone(), context).await?;
        }
        Ok(result)
    }

    pub async fn wait_for_task(&self, id: TaskId, context: &Context) -> Result<TaskRecord> {
        let me = self.arc();
        let callback_context = context.clone();
        // Check and register on the line so no terminal publication falls between them.
        let found = self
            .session()
            .read_on_line(async move {
                if me.closing() {
                    return Err(closed_error());
                }
                if me.state.lock().live.contains_key(&id) {
                    return Ok(me.task_waiters.add(id, &callback_context));
                }
                match me.session().storage().task(id, &callback_context).await? {
                    None => Err(Error::message(format!("Task {id} does not exist"))),
                    Some(record) => {
                        Ok(Box::pin(futures::future::ready(Ok(record))) as BoxFuture<'static, _>)
                    }
                }
            })
            .await?;
        found.await
    }

    /// Resolve when ordinary traversal from the conversation, or from every ownerless conversation, reaches no live
    /// non-background task.
    pub fn wait_for_idle(
        &self,
        conversation_id: Option<ConversationId>,
        context: &Context,
    ) -> BoxFuture<'static, Result<()>> {
        {
            let state = self.state.lock();
            if state.closing {
                return Box::pin(futures::future::ready(Err(closed_error())));
            }
            if state.idle(conversation_id) {
                return Box::pin(futures::future::ready(Ok(())));
            }
        }
        self.schedule_reconcile();
        self.idle_waiters.add(conversation_id, context)
    }

    /// `Conversation.abort()`: in one commit, withdraw the queued inputs and mark every live non-background task that
    /// ordinary traversal from the conversation reaches; resolves once the scope is idle.
    pub async fn abort_conversation(
        &self,
        conversation_id: ConversationId,
        background: bool,
        context: &Context,
    ) -> Result<()> {
        let me = self.arc();
        let reached = self
            .session()
            .commit_with(
                move |tx| async move {
                    let queued = me.load_scopes(true).await?;
                    let scope = Scope::Conversation(conversation_id);
                    let mut reached = Vec::new();
                    let withdraw: Vec<ConversationId> = {
                        let state = me.state.lock();
                        for record in state.live.values() {
                            if record.background && !background {
                                continue;
                            }
                            if state.in_scope(record_parent(record), scope, background)
                                != Some(true)
                            {
                                continue;
                            }
                            reached.push(record.id);
                            if !record.abort_requested {
                                tx.set_task(TaskRecord {
                                    abort_requested: true,
                                    ..record.clone()
                                })?;
                            }
                        }
                        queued
                            .into_iter()
                            .filter(|id| {
                                state.in_scope(Up::Conversation(*id), scope, background)
                                    == Some(true)
                            })
                            .collect()
                    };
                    for id in withdraw {
                        (me.options.withdraw_inputs)(tx.clone(), id).await?;
                    }
                    Ok(reached)
                },
                context,
                TransactionScope::default(),
            )
            .await?;
        if background {
            for id in reached {
                self.wait_for_task(id, context).await?;
            }
        }
        self.wait_for_idle(Some(conversation_id), context).await
    }

    // ─── Scheduling ────────────────────────────────────────────────────────

    fn observe(&self, publication: &CommitPublication) {
        let mut signal = Vec::new();
        let mut resolved = Vec::new();
        let mut reconcile = false;
        let changed;
        {
            let mut state = self.state.lock();
            let mut updated = Vec::new();
            let mut failed = Vec::new();
            let mut any = false;
            for change in &publication.changes {
                let CommitChange::Task(record) = change else {
                    continue;
                };
                any = true;
                let previous = state.live.get(&record.id).cloned();
                if failed_outcome(record) && previous.as_ref().is_none_or(|p| !failed_outcome(p)) {
                    failed.push(record.id);
                }
                if record.state.status() == TaskStatus::Terminal {
                    state.live.shift_remove(&record.id);
                    state.failed_migrations.remove(&record.id);
                    state.fail_fast_checks.shift_remove(&record.id);
                    if state.conversation_owners.contains(&record.id) {
                        state.settled.insert(record.id, node_of(record));
                    }
                    resolved.push(record.clone());
                    // Its owner may finalize now.
                    reconcile = true;
                    continue;
                }
                if record.abort_requested && previous.as_ref().is_none_or(|p| !p.abort_requested) {
                    state.cascade_pending = true;
                    // Signal a run invocation of the newly marked task; its next step ends it.
                    if let Some(invocation) = state.invocations.get(&record.id)
                        && invocation.mode == Mode::Run
                    {
                        signal.push(invocation.clone());
                    }
                }
                let status = record.state.status();
                let previous_status = previous.as_ref().map(|p| p.state.status());
                if status == TaskStatus::Completing
                    && previous_status != Some(TaskStatus::Completing)
                {
                    if cancellation_intent(record) {
                        state.cascade_pending = true;
                    }
                    reconcile = true;
                }
                if let TaskState::Waiting {
                    policy: JoinPolicy::FailFast,
                    ..
                } = &record.state
                    && previous_status != Some(TaskStatus::Waiting)
                {
                    state.fail_fast_checks.insert(record.id);
                    reconcile = true;
                }
                state.live.insert(record.id, record.clone());
                updated.push(record.clone());
            }
            for id in failed {
                let waiters: Vec<TaskId> = state
                    .live
                    .values()
                    .filter(|record| match &record.state {
                        TaskState::Waiting {
                            policy: JoinPolicy::FailFast,
                            on,
                            ..
                        } => on.contains(&id),
                        _ => false,
                    })
                    .map(|record| record.id)
                    .collect();
                for waiter in waiters {
                    state.fail_fast_checks.insert(waiter);
                    reconcile = true;
                }
            }
            for change in &publication.changes {
                if let CommitChange::Conversation(record) = change
                    && !state.edges.contains_key(&record.id)
                {
                    state.set_edge(record.id, record.owner.map(|owner| owner.task_id));
                }
            }
            for change in &publication.changes {
                // A queued input below a cancelled owner is withdrawn, even after its cascade.
                let CommitChange::Submission(submission) = change else {
                    continue;
                };
                if submission.status != SubmissionStatus::Queued
                    || submission.type_ != crate::durable::types::SubmissionType::Input
                {
                    continue;
                }
                let up = Up::Conversation(submission.conversation_id);
                if !state.chain_known(up, None) || state.below_cancelled(up) {
                    state.cascade_pending = true;
                }
            }
            for record in &updated {
                // Work created below a cancelled owner, even after its cascade, is aborted too.
                if !state.chain_known(record_parent(record), None) {
                    reconcile = true;
                } else if !record.background
                    && !record.abort_requested
                    && state.below_cancelled(record_parent(record))
                {
                    state.cascade_pending = true;
                }
            }
            // Also retries, with the next commit of any kind, a cascade whose commit failed.
            if state.cascade_pending {
                reconcile = true;
            }
            changed = any;
        }
        for record in resolved {
            self.task_waiters.resolve(&record.id.clone(), record);
        }
        for invocation in signal {
            invocation.controller.abort(None);
        }
        if reconcile {
            self.schedule_reconcile();
        }
        if !changed {
            return;
        }
        self.resolve_idle_waiters();
        self.kick();
    }

    fn resolve_idle_waiters(&self) {
        for conversation_id in self.idle_waiters.keys() {
            let idle = self.state.lock().idle(conversation_id);
            if idle {
                self.idle_waiters.resolve(&conversation_id, ());
            }
        }
    }

    // ─── Ownership ───────────────────────────────────────────────────────────

    fn schedule_reconcile(&self) {
        {
            let mut state = self.state.lock();
            if state.reconcile_scheduled || state.closing {
                return;
            }
            state.reconcile_scheduled = true;
        }
        let me = self.arc();
        tokio::spawn(async move { me.reconcile().await });
    }

    /// One commit that applies what committed records imply: abort marks below live owners with cancellation intent,
    /// `failFast` marks, withdrawn queued inputs below cancelled owners, and the final terminal record of every
    /// `completing` task whose ordinary owned work is gone. Resolves idle waiters that the loaded edges decide.
    async fn reconcile(self: Arc<Self>) {
        let (cascade, checks) = {
            let mut state = self.state.lock();
            state.reconcile_scheduled = false;
            let cascade = std::mem::take(&mut state.cascade_pending);
            let checks: Vec<TaskId> = state.fail_fast_checks.drain(..).collect();
            (cascade, checks)
        };
        let me = self.clone();
        let retry_checks = checks.clone();
        let result = self
            .session()
            .commit_with(
                move |tx| async move {
                    if me.closing() {
                        return Ok(());
                    }
                    let queued = me.load_scopes(cascade).await?;
                    let mut marked: HashSet<TaskId> = HashSet::new();
                    let mark = |record: &TaskRecord, marked: &mut HashSet<TaskId>| -> Result<()> {
                        if record.abort_requested || !marked.insert(record.id) {
                            return Ok(());
                        }
                        tx.set_task(TaskRecord {
                            abort_requested: true,
                            ..record.clone()
                        })
                    };
                    // Loading edges can reveal a cancelled owner, so marks are derived on every pass.
                    let below: Vec<TaskRecord> = {
                        let state = me.state.lock();
                        state
                            .live
                            .values()
                            .filter(|record| {
                                !record.background && state.below_cancelled(record_parent(record))
                            })
                            .cloned()
                            .collect()
                    };
                    for record in &below {
                        mark(record, &mut marked)?;
                    }
                    for id in checks {
                        let waiter = me.state.lock().live.get(&id).cloned();
                        let Some(TaskRecord {
                            state: TaskState::Waiting { on, .. },
                            ..
                        }) = waiter
                        else {
                            continue;
                        };
                        if !me.any_failed(&on).await? {
                            continue;
                        }
                        // Every other live task: the failed one keeps its own outcome.
                        for member in on {
                            let record = me.state.lock().live.get(&member).cloned();
                            if let Some(record) = record
                                && !failed_outcome(&record)
                            {
                                mark(&record, &mut marked)?;
                            }
                        }
                    }
                    for id in queued {
                        let cancelled = me.state.lock().below_cancelled(Up::Conversation(id));
                        if cancelled {
                            (me.options.withdraw_inputs)(tx.clone(), id).await?;
                        }
                    }
                    me.finalize(&tx).await
                },
                &self.options.context,
                TransactionScope::default(),
            )
            .await;
        if let Err(error) = result {
            // Any pass may have staged marks, so a failed one is retried with the next commit.
            let closing = {
                let mut state = self.state.lock();
                state.cascade_pending = true;
                state.fail_fast_checks.extend(retry_checks);
                state.closing
            };
            if !closing {
                self.report(error);
            }
        }
        self.resolve_idle_waiters();
    }

    /// Whether any of `ids` holds or ended with an outcome other than `completed`.
    async fn any_failed(&self, ids: &[TaskId]) -> Result<bool> {
        for id in ids {
            let live = self.state.lock().live.get(id).cloned();
            let record = match live {
                Some(record) => Some(record),
                None => {
                    self.session()
                        .storage()
                        .task(*id, &self.options.context)
                        .await?
                }
            };
            if record.as_ref().is_some_and(failed_outcome) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Write the terminal record of every `completing` task without live ordinary owned work. Finalizing one can free
    /// its owner, so this repeats over the commit's candidates until nothing changes.
    async fn finalize(&self, tx: &Transaction) -> Result<()> {
        loop {
            let done: Vec<TaskRecord> = {
                let overlay = overlay_of(tx);
                let state = self.state.lock();
                let owned = state.owned_live(Some(&overlay));
                state
                    .live_records(Some(&overlay))
                    .into_iter()
                    .filter(|record| {
                        record.state.status() == TaskStatus::Completing
                            && !owned.contains_key(&record.id)
                    })
                    .collect()
            };
            if done.is_empty() {
                return Ok(());
            }
            for record in done {
                let TaskState::Completing { outcome } = &record.state else {
                    continue;
                };
                let outcome = outcome.clone();
                tx.set_task(with_state(
                    &record,
                    TaskState::Terminal {
                        outcome: outcome.clone(),
                    },
                ))?;
                // REMINDER: only the scheduler writes `faulted` and `orphaned` (spec §5.4); their cleanup waits for
                // this commit.
                if matches!(
                    outcome,
                    TaskOutcome::Faulted { .. } | TaskOutcome::Orphaned { .. }
                ) {
                    (self.options.settle_outcome)(tx.clone(), record.clone(), outcome).await?;
                }
            }
        }
    }

    /// Load the owner chains of every live task and, with `queued`, of every conversation with queued submissions, on
    /// the Session line; returns the latter.
    async fn load_scopes(&self, queued: bool) -> Result<Vec<ConversationId>> {
        let records: Vec<TaskRecord> = self.state.lock().live.values().cloned().collect();
        for record in records {
            let parent = record_parent(&record);
            let known = self.state.lock().chain_known(parent, None);
            if !known {
                self.load_chain(parent, None).await?;
            }
        }
        if !queued {
            return Ok(Vec::new());
        }
        let storage = self.session().storage().clone();
        let context = self.options.context.clone();
        let submissions = scan_all(|cursor| {
            let storage = storage.clone();
            let context = context.clone();
            async move {
                storage
                    .scan_submissions(
                        &SubmissionQuery {
                            conversation_id: None,
                            status: Some(SubmissionStatus::Queued),
                        },
                        SCAN_PAGE_SIZE,
                        cursor.as_ref(),
                        &context,
                    )
                    .await
            }
        })
        .await?;
        let conversations: IndexSet<ConversationId> = submissions
            .iter()
            .map(|submission| submission.conversation_id)
            .collect();
        for id in &conversations {
            self.load_chain(Up::Conversation(*id), None).await?;
        }
        Ok(conversations.into_iter().collect())
    }

    /// Load the owner edges and task nodes from `start` up to its ownerless root.
    async fn load_chain(&self, start: Up, overlay: Option<&Overlay>) -> Result<()> {
        let storage = self.session().storage().clone();
        let context = &self.options.context;
        let mut at = Some(start);
        while let Some(up) = at {
            match up {
                Up::Task(id) => {
                    let node = self.state.lock().node(id, overlay);
                    let node = match node {
                        Some(node) => node,
                        None => {
                            let Some(record) = storage.task(id, context).await? else {
                                return Ok(());
                            };
                            let node = node_of(&record);
                            if record.state.status() == TaskStatus::Terminal {
                                self.state.lock().settled.insert(record.id, node);
                            }
                            node
                        }
                    };
                    at = Some(parent_of(&node));
                }
                Up::Conversation(id) => {
                    let edge = self.state.lock().edge(id, overlay);
                    let edge = match edge {
                        Some(edge) => edge,
                        None => {
                            let record = storage.conversation(id, context).await?;
                            let edge = record
                                .and_then(|record| record.owner)
                                .map(|owner| owner.task_id);
                            self.state.lock().set_edge(id, edge);
                            edge
                        }
                    };
                    at = edge.map(Up::Task);
                }
            }
        }
        Ok(())
    }

    /// The live record the scheduler tracks for `id` (tests only).
    #[cfg(test)]
    pub(crate) fn live_task(&self, id: TaskId) -> Option<TaskRecord> {
        self.state.lock().live.get(&id).cloned()
    }

    /// Close listener: runs synchronously once admission is sealed, before `join()`.
    fn seal(&self) {
        let invocations: Vec<Arc<Invocation>> = {
            let mut state = self.state.lock();
            state.closing = true;
            state.invocations.values().cloned().collect()
        };
        let unsubscribe = self.unsubscribe_registry.lock().take();
        if let Some(unsubscribe) = unsubscribe {
            unsubscribe();
        }
        let error = closed_error();
        self.task_waiters.reject_all(error.clone());
        self.idle_waiters.reject_all(error);
        for invocation in invocations {
            invocation.controller.abort(None);
        }
    }

    fn kick(&self) {
        {
            let mut state = self.state.lock();
            state.dirty = true;
            if state.draining || !state.enabled || state.closing {
                return;
            }
            state.draining = true;
        }
        // Never commit synchronously from a commit or registry listener.
        let me = self.arc();
        tokio::spawn(async move { me.drain().await });
    }

    async fn drain(self: Arc<Self>) {
        let result: Result<()> = async {
            loop {
                {
                    let mut state = self.state.lock();
                    if !(state.dirty && state.enabled && !state.closing) {
                        return Ok(());
                    }
                    state.dirty = false;
                }
                for reservation in self.reserve().await? {
                    self.start(reservation);
                }
            }
        }
        .await;
        if let Err(error) = result
            && !self.closing()
        {
            self.report(error);
        }
        let dirty = {
            let mut state = self.state.lock();
            state.draining = false;
            state.dirty
        };
        // A wakeup that arrived during a failed pass still needs its pass.
        if dirty {
            self.kick();
        }
    }

    /// Reserve every eligible task in one commit; orphan abort-marked tasks no definition can take.
    async fn reserve(&self) -> Result<Vec<Reservation>> {
        let me = self.arc();
        let reservations: Arc<Mutex<Vec<Reservation>>> = Arc::default();
        let staged = reservations.clone();
        let result = self
            .session()
            .commit_with(
                move |tx| async move {
                    {
                        let state = me.state.lock();
                        if !state.enabled || state.closing {
                            return Ok(());
                        }
                    }
                    me.load_scopes(false).await?;
                    let (records, owned) = {
                        let state = me.state.lock();
                        (
                            state.live.values().cloned().collect::<Vec<_>>(),
                            state.owned_live(None),
                        )
                    };
                    // Taken once per pass, and only when some task is a candidate.
                    let mut snapshot: Option<RegistrySnapshot> = None;
                    for record in records {
                        {
                            let state = me.state.lock();
                            if state.invocations.contains_key(&record.id)
                                || !state.waiting_on(&record, &owned).is_empty()
                            {
                                continue;
                            }
                        }
                        if record.state.status() == TaskStatus::Completing {
                            continue;
                        }
                        let mode = if record.abort_requested {
                            Mode::Abort
                        } else {
                            Mode::Run
                        };
                        let snapshot = snapshot
                            .get_or_insert_with(|| me.options.registry.snapshot())
                            .clone();
                        match me.resolve(&record, &snapshot) {
                            Resolution::Blocked { reason } => {
                                if mode == Mode::Abort {
                                    me.terminate(
                                        &tx,
                                        &record,
                                        TaskOutcome::Orphaned {
                                            reason: reason.as_str().to_string(),
                                        },
                                    )
                                    .await?;
                                }
                            }
                            Resolution::Ready {
                                task,
                                record: resolved,
                            } => {
                                if resolved != record
                                    || record.state.status() != TaskStatus::Running
                                {
                                    let checkpoint =
                                        checkpoint_of(&resolved).cloned().unwrap_or_default();
                                    tx.set_task(with_state(
                                        &resolved,
                                        TaskState::Running { checkpoint },
                                    ))?;
                                }
                                // Registered on the line, so marks and later reservations see it and close joins it.
                                let invocation = me.create_invocation(&record, mode);
                                staged.lock().push(Reservation {
                                    invocation,
                                    task,
                                    snapshot,
                                });
                            }
                        }
                    }
                    Ok(())
                },
                &self.options.context,
                TransactionScope::default(),
            )
            .await;
        let reservations = std::mem::take(&mut *reservations.lock());
        if let Err(error) = result {
            for reservation in reservations {
                let invocation = &reservation.invocation;
                let mut state = self.state.lock();
                if state
                    .invocations
                    .get(&invocation.task_id)
                    .is_some_and(|other| Arc::ptr_eq(other, invocation))
                {
                    state.invocations.remove(&invocation.task_id);
                }
                drop(state);
                invocation.finish();
            }
            return Err(error);
        }
        Ok(reservations)
    }

    /// Resolve the record's definition by kind, migrating an older stored version.
    fn resolve(&self, record: &TaskRecord, snapshot: &RegistrySnapshot) -> Resolution {
        let fit = self.state.lock().fit(record, snapshot.task(&record.kind));
        let (task, migrates) = match fit {
            Fit::Blocked { reason, .. } => return Resolution::Blocked { reason },
            Fit::Task { task, migrates } => (task, migrates),
        };
        if !migrates {
            return Resolution::Ready {
                task,
                record: record.clone(),
            };
        }
        let checkpoint = checkpoint_of(record).cloned().unwrap_or_default();
        let migrated = match task.run_migrate(record.input.clone(), checkpoint, record.version) {
            None => Err(missing_migration(record, &task)),
            Some(result) => result,
        };
        match migrated {
            Ok((input, checkpoint)) => {
                let state = match &record.state {
                    TaskState::Pending { .. } => TaskState::Pending { checkpoint },
                    TaskState::Running { .. } => TaskState::Running { checkpoint },
                    TaskState::Waiting { on, policy, .. } => TaskState::Waiting {
                        checkpoint,
                        on: on.clone(),
                        policy: *policy,
                    },
                    other => other.clone(),
                };
                Resolution::Ready {
                    record: TaskRecord {
                        version: task.definition().version,
                        input,
                        state,
                        ..record.clone()
                    },
                    task,
                }
            }
            Err(error) => {
                self.state
                    .lock()
                    .failed_migrations
                    .insert(record.id, (task, error.clone()));
                self.report(error);
                Resolution::Blocked {
                    reason: BlockedReason::MigrationFailed,
                }
            }
        }
    }

    /// Scheduling state and every live task with its derived state, read on the Session line. Runs no task code.
    pub async fn inspect(
        &self,
        snapshot: &RegistrySnapshot,
    ) -> Result<(Scheduling, Vec<TaskInspection>)> {
        self.load_scopes(false).await?;
        let state = self.state.lock();
        let owned = state.owned_live(None);
        let tasks = state
            .live
            .values()
            .map(|record| TaskInspection {
                record: record.clone(),
                state: inspect_task(&state, record, snapshot, &owned),
            })
            .collect();
        let scheduling = if state.closing {
            Scheduling::Closing
        } else if state.enabled {
            Scheduling::Running
        } else {
            Scheduling::Paused
        };
        Ok((scheduling, tasks))
    }

    fn create_invocation(&self, record: &TaskRecord, mode: Mode) -> Arc<Invocation> {
        let controller = AbortController::new();
        let (finish, done) = oneshot::channel::<()>();
        let invocation = Arc::new(Invocation {
            task_id: record.id,
            conversation_id: record.conversation_id,
            mode,
            context: with_abort_signal(controller.signal(), &self.options.context),
            controller,
            watches: Mutex::default(),
            ended: AtomicBool::new(false),
            done: done.map(|_| ()).boxed().shared(),
            finish: Mutex::new(Some(finish)),
        });
        self.state
            .lock()
            .invocations
            .insert(record.id, invocation.clone());
        invocation
    }

    fn start(&self, reservation: Reservation) {
        let me = self.arc();
        tokio::spawn(async move {
            let invocation = reservation.invocation.clone();
            let result = match invocation.mode {
                Mode::Run => me.run(reservation).await,
                Mode::Abort => me.run_abort(reservation).await,
            };
            if let Err(error) = result {
                me.report(error);
            }
            me.end(&invocation);
            invocation.finish();
            me.kick();
        });
    }

    /// Run phase handlers, each preceded by a step that decides on the line whether the invocation continues.
    async fn run(&self, reservation: Reservation) -> Result<()> {
        let invocation = reservation.invocation;
        let phase = Arc::new(Mutex::new(Phase {
            snapshot: reservation.snapshot.clone(),
            task: reservation.task.clone(),
            agent: None,
        }));
        let core = Arc::new(RuntimeCore {
            scheduler: self.arc(),
            invocation: invocation.clone(),
            phase: phase.clone(),
        });
        let reported: Arc<Mutex<Option<Option<AnyTask>>>> = Arc::default();
        let mut previous: Option<(JsonValue, Option<Error>)> = None;
        loop {
            let me = self.arc();
            let phase_state = phase.clone();
            let reported = reported.clone();
            let prior = previous.take();
            let current = self
                .step(&invocation, move |tx, current| {
                    me.decide(tx, current, prior, &phase_state, &reported)
                })
                .await;
            // Close may seal between the decision and dispatch.
            let Some(current) = current else {
                return Ok(());
            };
            if self.closing() {
                return Ok(());
            }
            let checkpoint = checkpoint_of(&current).cloned().unwrap_or_default();
            // Each phase handler resolves its agent afresh, at first use.
            let task = {
                let mut phase = phase.lock();
                phase.agent = None;
                phase.task.clone()
            };
            let name = checkpoint
                .get("phase")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_string();
            let failure = match task.phase(&name) {
                None => Some(Error::type_error(format!(
                    "Task {} has no phase {name}",
                    current.kind
                ))),
                Some(handler) => {
                    let outcome = AssertUnwindSafe(handler(
                        current.clone(),
                        core.clone(),
                        invocation.context.clone(),
                    ))
                    .catch_unwind()
                    .await;
                    match outcome {
                        Ok(Ok(())) => None,
                        Ok(Err(error)) => Some(error),
                        Err(panic) => Some(panic_error(panic)),
                    }
                }
            };
            previous = Some((checkpoint, failure));
        }
    }

    /// Precedence rules for a run invocation, on the line. Rules 1 (terminal, `completing`, or `waiting`) and 2
    /// (closing) are applied by `step`.
    fn decide(
        &self,
        tx: &Transaction,
        current: &TaskRecord,
        previous: Option<(JsonValue, Option<Error>)>,
        phase: &Mutex<Phase>,
        reported: &Mutex<Option<Option<AnyTask>>>,
    ) -> Result<Decision> {
        // 3. abort mark: end; a fresh abort invocation starts once the task's ordinary owned work is gone.
        if current.abort_requested {
            return Ok(Decision::End);
        }
        let Some((checkpoint, failure)) = previous else {
            return Ok(Decision::Continue);
        };
        // 4. uncaught error.
        if let Some(error) = failure {
            return Ok(Decision::Fault(error));
        }
        // 6. no durable progress.
        if checkpoint_of(current) == Some(&checkpoint) {
            let name = checkpoint
                .get("phase")
                .and_then(JsonValue::as_str)
                .unwrap_or_default();
            return Ok(Decision::Fault(Error::message(format!(
                "Task {} phase {name} returned without durable progress",
                current.kind
            ))));
        }
        // 5. progress: refresh the snapshot; hand over to a replacement definition that can take the task.
        let snapshot = self.options.registry.snapshot();
        let next = snapshot.task(&current.kind).cloned();
        let mut phase = phase.lock();
        phase.snapshot = snapshot;
        let same = next.as_ref().is_some_and(|next| next.ptr_eq(&phase.task));
        if !same {
            if let Some(next) = &next
                && can_reserve(next, current)
            {
                let checkpoint = checkpoint_of(current).cloned().unwrap_or_default();
                tx.set_task(with_state(current, TaskState::Pending { checkpoint }))?;
                return Ok(Decision::End);
            }
            let mut reported = reported.lock();
            let already = reported.as_ref().is_some_and(|task| match (task, &next) {
                (None, None) => true,
                (Some(task), Some(next)) => task.ptr_eq(next),
                _ => false,
            });
            if !already {
                *reported = Some(next.clone());
                let cause = if next.is_none() {
                    "missing_task"
                } else {
                    "incompatible_task"
                };
                self.report(Error::with_cause(
                    format!(
                        "Task {} keeps running under its old {} definition",
                        current.id, current.kind
                    ),
                    Error::message(cause),
                ));
            }
        }
        Ok(Decision::Continue)
    }

    /// Run the abort handler once; rules 1, 2, and 4 apply, and returning without an outcome faults.
    async fn run_abort(&self, reservation: Reservation) -> Result<()> {
        let invocation = reservation.invocation;
        let current = self.state.lock().live.get(&invocation.task_id).cloned();
        let Some(current) = current else {
            return Ok(());
        };
        if self.closing() {
            return Ok(());
        }
        let core = Arc::new(RuntimeCore {
            scheduler: self.arc(),
            invocation: invocation.clone(),
            phase: Arc::new(Mutex::new(Phase {
                snapshot: reservation.snapshot.clone(),
                task: reservation.task.clone(),
                agent: None,
            })),
        });
        let handler = reservation.task.abort_handler();
        let outcome = AssertUnwindSafe(handler(current, core, invocation.context.clone()))
            .catch_unwind()
            .await;
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(panic) => Some(panic_error(panic)),
        };
        let message = format!(
            "Abort handler of task {} returned without a terminal outcome",
            invocation.task_id
        );
        let fault = failure.unwrap_or_else(|| Error::message(message));
        self.step(&invocation, move |_, _| Ok(Decision::Fault(fault)))
            .await;
        Ok(())
    }

    /// One synchronous decision on the Session line. A task that is no longer running or a closing Harness ends the
    /// invocation without a write; otherwise `decide` may stage a write and returns whether the invocation continues.
    async fn step(
        &self,
        invocation: &Arc<Invocation>,
        decide: impl FnOnce(&Transaction, &TaskRecord) -> Result<Decision> + Send + 'static,
    ) -> Option<TaskRecord> {
        let me = self.arc();
        let bound = invocation.clone();
        let result = self
            .session()
            .commit_with(
                move |tx| async move {
                    let (current, closing) = {
                        let state = me.state.lock();
                        let found = state.live.get(&bound.task_id).cloned();
                        (
                            found.filter(|found| found.state.status() == TaskStatus::Running),
                            state.closing,
                        )
                    };
                    let decision = match &current {
                        Some(current) if !closing => decide(&tx, current)?,
                        _ => Decision::End,
                    };
                    if let Decision::Continue = decision {
                        return Ok(current);
                    }
                    me.end(&bound);
                    if let Decision::Fault(error) = decision {
                        let current = current.expect("a fault decision has a running task");
                        me.terminate(
                            &tx,
                            &current,
                            TaskOutcome::Faulted {
                                error: TaskOutcomeError {
                                    message: error.to_string(),
                                    detail: None,
                                },
                            },
                        )
                        .await?;
                    }
                    Ok(None)
                },
                &self.options.context,
                TransactionScope::default(),
            )
            .await;
        match result {
            Ok(current) => current,
            Err(error) => {
                self.end(invocation);
                if !self.closing() {
                    self.report(error);
                }
                None
            }
        }
    }

    /// Write an outcome the scheduler decided. While the task's ordinary owned work is live it holds as `completing`
    /// and its Harness cleanup waits for the final commit; otherwise it is terminal with its cleanup.
    async fn terminate(
        &self,
        tx: &Transaction,
        record: &TaskRecord,
        outcome: TaskOutcome,
    ) -> Result<()> {
        self.load_scopes(false).await?;
        let owned = {
            let overlay = overlay_of(tx);
            self.state
                .lock()
                .owned_live(Some(&overlay))
                .contains_key(&record.id)
        };
        if owned {
            return tx.set_task(with_state(record, TaskState::Completing { outcome }));
        }
        tx.set_task(with_state(
            record,
            TaskState::Terminal {
                outcome: outcome.clone(),
            },
        ))?;
        (self.options.settle_outcome)(tx.clone(), record.clone(), outcome).await
    }

    /// Replace a running task's state with what it committed. A terminal state holds as `completing` while ordinary
    /// owned work is live, judged on the commit's candidates. A wait is validated first.
    async fn commit_state(
        &self,
        tx: &Transaction,
        invocation: &Invocation,
        current: &TaskRecord,
        next: TaskState,
    ) -> Result<()> {
        if let TaskState::Waiting { on, policy, .. } = &next {
            self.validate_wait(tx, invocation, current, on, *policy)
                .await?;
        }
        if let TaskState::Terminal { outcome } = &next {
            let overlay = overlay_of(tx);
            self.load_scopes(false).await?;
            for record in overlay.tasks.values() {
                self.load_chain(record_parent(record), Some(&overlay))
                    .await?;
            }
            let owned = self
                .state
                .lock()
                .owned_live(Some(&overlay))
                .contains_key(&current.id);
            if owned {
                return tx.set_task(with_state(
                    current,
                    TaskState::Completing {
                        outcome: outcome.clone(),
                    },
                ));
            }
        }
        tx.set_task(with_state(current, next))
    }

    /// A wait names existing tasks other than the waiter and its owners; `failFast` only tasks the waiter owns. An
    /// abort handler cannot wait.
    async fn validate_wait(
        &self,
        tx: &Transaction,
        invocation: &Invocation,
        current: &TaskRecord,
        on: &[TaskId],
        policy: JoinPolicy,
    ) -> Result<()> {
        if invocation.mode == Mode::Abort {
            return Err(Error::message(format!(
                "Abort handler of task {} cannot wait",
                current.id
            )));
        }
        let overlay = overlay_of(tx);
        self.load_chain(record_parent(current), None).await?;
        let owners: HashSet<TaskId> = self
            .state
            .lock()
            .above(record_parent(current), None)
            .into_iter()
            .filter_map(|step| match step {
                Step::Task(id, _) => Some(id),
                _ => None,
            })
            .collect();
        for id in on {
            if *id == current.id || owners.contains(id) {
                return Err(Error::message(format!(
                    "Task {} cannot wait on itself or its owner {id}",
                    current.id
                )));
            }
            let member = match overlay.tasks.get(id).cloned() {
                Some(member) => Some(member),
                None => {
                    let live = self.state.lock().live.get(id).cloned();
                    match live {
                        Some(live) => Some(live),
                        None => {
                            self.session()
                                .storage()
                                .task(*id, &self.options.context)
                                .await?
                        }
                    }
                }
            };
            let Some(member) = member else {
                return Err(Error::message(format!("Task {id} does not exist")));
            };
            if policy == JoinPolicy::FailFast && member.owner != Some(current.id) {
                return Err(Error::message(format!(
                    "Task {} can wait failFast only on tasks it owns; {id} is not one",
                    current.id
                )));
            }
        }
        Ok(())
    }

    /// End an invocation: its runtime operations reject from now on, its signal aborts, its watches stop, and its task
    /// is free.
    fn end(&self, invocation: &Arc<Invocation>) {
        if invocation.ended.swap(true, Ordering::SeqCst) {
            return;
        }
        {
            let mut state = self.state.lock();
            if state
                .invocations
                .get(&invocation.task_id)
                .is_some_and(|other| Arc::ptr_eq(other, invocation))
            {
                state.invocations.remove(&invocation.task_id);
            }
        }
        for watch in invocation.watches.lock().drain(..) {
            drop(watch.stop());
        }
        // Pending waits bound to the invocation, such as a tool's waitForTask(), reject with it.
        invocation
            .controller
            .abort(Some(abort_reason(invocation.ended_error())));
    }

    /// Wait until the Harness clock reaches `until`, rechecking it after every timer.
    async fn sleep(&self, invocation: &Invocation, until: u64, context: &Context) -> Result<()> {
        invocation.check()?;
        let mut signals = vec![invocation.controller.signal().clone()];
        if let Some(signal) = context.abort_signal() {
            signals.push(signal.clone());
        }
        let signal = AbortSignal::any(&signals);
        loop {
            if let Some(reason) = signal.reason() {
                return Err(abort_error(reason));
            }
            let now = (self.options.now)();
            if until <= now {
                return Ok(());
            }
            let delay = (until - now).min(MAX_TIMER_DELAY);
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(delay)) => {}
                _ = signal.cancelled() => {}
            }
        }
    }
}

fn inspect_task(
    state: &State,
    record: &TaskRecord,
    snapshot: &RegistrySnapshot,
    owned: &HashMap<TaskId, Vec<TaskId>>,
) -> TaskInspectionState {
    if state.invocations.contains_key(&record.id) {
        return TaskInspectionState::Running;
    }
    if record.state.status() == TaskStatus::Completing {
        return TaskInspectionState::Completing;
    }
    let on = state.waiting_on(record, owned);
    if !on.is_empty() {
        return TaskInspectionState::Waiting { on };
    }
    match state.fit(record, snapshot.task(&record.kind)) {
        Fit::Blocked { reason, error } => TaskInspectionState::Blocked { reason, error },
        Fit::Task { task, migrates } => {
            if migrates && !task.definition().has_migrate() {
                return TaskInspectionState::Blocked {
                    reason: BlockedReason::MigrationFailed,
                    error: Some(missing_migration(record, &task)),
                };
            }
            TaskInspectionState::Ready { migrates }
        }
    }
}

/// Result of `Harness.abortTask()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortTaskResult {
    Marked,
    Terminal,
}

// ─── Invocation runtime ──────────────────────────────────────────────────────

/// Erased operations of one task invocation; [`TaskRuntime`] types them.
pub struct RuntimeCore {
    scheduler: Arc<TaskScheduler>,
    invocation: Arc<Invocation>,
    phase: Arc<Mutex<Phase>>,
}

impl RuntimeCore {
    fn check(&self) -> Result<()> {
        self.invocation.check()
    }

    fn agent_future(&self) -> Shared<BoxFuture<'static, Result<Agent>>> {
        let mut phase = self.phase.lock();
        if let Some(agent) = &phase.agent {
            return agent.clone();
        }
        let agent = (self.scheduler.options.agent)(
            self.invocation.conversation_id,
            phase.snapshot.clone(),
            self.invocation.context.clone(),
        )
        .shared();
        phase.agent = Some(agent.clone());
        agent
    }

    /// The task's conversation's agent, resolved at most once per phase, at first use, and fixed for the phase.
    pub async fn agent(&self, context: &Context) -> Result<Agent> {
        self.check()?;
        await_with_context(self.agent_future(), context).await?
    }

    /// Commit after rereading the task on the line and gating the invocation.
    pub(crate) async fn gated<T, F, Fut>(&self, change: F, context: &Context) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(Transaction, TaskRecord) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        self.check()?;
        let scheduler = self.scheduler.clone();
        let invocation = self.invocation.clone();
        let scope = TransactionScope {
            conversation_id: Some(invocation.conversation_id),
            task_id: Some(invocation.task_id),
        };
        self.scheduler
            .session()
            .commit_with(
                move |tx| async move {
                    invocation.check()?;
                    if scheduler.closing() {
                        return Err(closed_error());
                    }
                    let found = scheduler
                        .state
                        .lock()
                        .live
                        .get(&invocation.task_id)
                        .cloned();
                    let Some(found) = found else {
                        return Err(Error::message(format!(
                            "Task {} is terminal",
                            invocation.task_id
                        )));
                    };
                    if found.state.status() != TaskStatus::Running {
                        return Err(Error::message(format!(
                            "Task {} is {}",
                            invocation.task_id,
                            found.state.status()
                        )));
                    }
                    if invocation.mode == Mode::Run && found.abort_requested {
                        return Err(Error::message(format!(
                            "Task {} has a durable abort mark",
                            invocation.task_id
                        )));
                    }
                    change(tx, found).await
                },
                context,
                scope,
            )
            .await
    }

    /// Commit, replacing the task's state with the returned one.
    pub async fn commit<F, Fut>(&self, change: F, context: &Context) -> Result<()>
    where
        F: FnOnce(Transaction, TaskRecord) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Option<TaskState>>> + Send + 'static,
    {
        let scheduler = self.scheduler.clone();
        let invocation = self.invocation.clone();
        self.gated(
            move |tx, current| async move {
                if let Some(next) = change(tx.clone(), current.clone()).await? {
                    scheduler
                        .commit_state(&tx, &invocation, &current, next)
                        .await?;
                }
                Ok(())
            },
            context,
        )
        .await
    }

    /// Read a durable memo of this task.
    pub async fn memo(&self, name: &str) -> Result<Option<JsonValue>> {
        self.check()?;
        let state = self.scheduler.state.lock();
        Ok(memo_of(state.live.get(&self.invocation.task_id), name))
    }

    /// Store `candidate` unless a memo already exists; return the durable winner.
    pub async fn memo_with(
        &self,
        name: &str,
        candidate: JsonValue,
        context: &Context,
    ) -> Result<JsonValue> {
        let name = name.to_string();
        self.gated(
            move |tx, current| async move {
                if let Some(winner) = memo_of(Some(&current), &name) {
                    return Ok(winner);
                }
                let mut memos = current.memos.clone().unwrap_or_default();
                memos.insert(name, candidate.clone());
                tx.set_task(TaskRecord {
                    memos: Some(memos),
                    ..current
                })?;
                Ok(candidate)
            },
            context,
        )
        .await
    }

    pub async fn get_task(&self, id: TaskId, context: &Context) -> Result<Option<TaskRecord>> {
        self.check()?;
        let storage = self.scheduler.session().storage().clone();
        let context = context.clone();
        self.scheduler
            .session()
            .read_on_line(async move { storage.task(id, &context).await })
            .await
    }

    pub async fn wait_for_task(&self, id: TaskId, context: &Context) -> Result<TaskRecord> {
        self.check()?;
        let context = with_abort_signal(self.invocation.controller.signal(), context);
        self.scheduler
            .wait_for_task(id, &context)
            .await
            .map_err(unwrap_aborted)
    }

    pub async fn outcomes(&self, ids: &[TaskId], context: &Context) -> Result<Vec<TaskOutcome>> {
        self.check()?;
        let storage = self.scheduler.session().storage().clone();
        let context = context.clone();
        let ids = ids.to_vec();
        self.scheduler
            .session()
            .read_on_line(async move {
                let mut outcomes = Vec::new();
                for id in ids {
                    match storage.task(id, &context).await?.map(|record| record.state) {
                        Some(TaskState::Terminal { outcome }) => outcomes.push(outcome),
                        _ => return Err(Error::message(format!("Task {id} is not terminal"))),
                    }
                }
                Ok(outcomes)
            })
            .await
    }

    pub async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<ConversationHandle>> {
        self.check()?;
        let invocation = self.invocation.clone();
        let binding = InvocationBinding {
            signal: self.invocation.controller.signal().clone(),
            check: Arc::new(move || invocation.check()),
        };
        (self.scheduler.options.conversation)(id, binding, context.clone()).await
    }

    pub async fn entry(&self, id: EntryId, context: &Context) -> Result<Option<EntryRecord>> {
        self.check()?;
        let storage = self.scheduler.session().storage().clone();
        let conversation_id = self.invocation.conversation_id;
        let context = context.clone();
        let found = self
            .scheduler
            .session()
            .read_on_line(async move { storage.entry_in(conversation_id, id, &context).await })
            .await?;
        Ok(found.map(|found| found.entry))
    }

    pub async fn context(
        &self,
        conversation_id: ConversationId,
        context: &Context,
        at: Option<EntryId>,
    ) -> Result<ContextView> {
        self.check()?;
        read_context(self.scheduler.session(), conversation_id, context, at).await
    }

    pub fn now(&self) -> Result<u64> {
        self.check()?;
        Ok((self.scheduler.options.now)())
    }

    pub fn report(&self, error: Error) -> Result<()> {
        self.check()?;
        self.scheduler.report(error);
        Ok(())
    }

    pub async fn sleep(&self, until: u64, context: &Context) -> Result<()> {
        self.scheduler.sleep(&self.invocation, until, context).await
    }

    pub async fn env(&self, context: &Context) -> Result<Option<Arc<dyn ExecutionEnv>>> {
        self.check()?;
        (self.scheduler.options.env)(self.invocation.conversation_id, context.clone()).await
    }

    pub fn reader(self: &Arc<Self>) -> DocumentReader {
        DocumentReader(self.clone())
    }

    pub fn signal(&self) -> AbortSignal {
        self.invocation.controller.signal().clone()
    }

    pub fn task_id(&self) -> TaskId {
        self.invocation.task_id
    }

    pub fn conversation_id(&self) -> ConversationId {
        self.invocation.conversation_id
    }

    pub fn registry(&self) -> RegistrySnapshot {
        self.phase.lock().snapshot.clone()
    }

    pub fn settings(&self) -> Settings {
        (self.scheduler.options.settings)()
    }

    pub fn models(&self) -> Models {
        self.scheduler.options.models.clone()
    }

    fn task_name(&self) -> String {
        self.phase.lock().task.name().to_string()
    }

    async fn watch(&self, watch: Option<DocumentWatch>) -> Result<Option<DocumentWatch>> {
        let Some(watch) = watch else {
            return Ok(None);
        };
        if self.invocation.ended() {
            drop(watch.stop());
            return Err(self.invocation.ended_error());
        }
        self.invocation.watches.lock().push(watch.clone());
        // `watch.closed.then(() => invocation.watches.delete(watch))`.
        let invocation = Arc::downgrade(&self.invocation);
        let (closed, tracked) = (watch.closed(), watch.clone());
        tokio::spawn(async move {
            closed.await;
            if let Some(invocation) = invocation.upgrade() {
                invocation
                    .watches
                    .lock()
                    .retain(|watch| !watch.ptr_eq(&tracked));
            }
        });
        Ok(Some(watch))
    }
}

/// A rejection through a signal whose reason is a durable error is that error.
fn unwrap_aborted(error: Error) -> Error {
    match error {
        Error::Aborted(reason) => abort_error(reason),
        error => error,
    }
}

impl ErasedReader for RuntimeCore {
    fn snapshot_json(
        &self,
        definition: Arc<crate::durable::documents::AnyDocDefinition>,
        args: crate::durable::documents::DocArgs,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<Arc<JsonValue>>>> {
        if let Err(error) = self.check() {
            return Box::pin(futures::future::ready(Err(error)));
        }
        self.scheduler
            .session()
            .snapshot_json(&RawToken(definition), args, context)
    }

    fn snapshot_as_of_json(
        &self,
        definition: Arc<crate::durable::documents::AnyDocDefinition>,
        args: crate::durable::documents::DocArgs,
        at: EntryId,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<JsonValue>>> {
        if let Err(error) = self.check() {
            return Box::pin(futures::future::ready(Err(error)));
        }
        self.scheduler
            .session()
            .snapshot_as_of(&RawToken(definition), args, at, context)
    }
}

/// An erased document token addressed by raw arguments, for erased committed reads.
pub(crate) struct RawToken(pub Arc<crate::durable::documents::AnyDocDefinition>);

impl DocAccess for RawToken {
    type Value = JsonValue;
    type Address = crate::durable::documents::DocArgs;
    type Args = crate::durable::documents::DocArgs;

    fn definition(&self) -> &Arc<crate::durable::documents::AnyDocDefinition> {
        &self.0
    }

    fn address_args(&self, address: Self::Address) -> crate::durable::documents::DocArgs {
        address
    }

    fn acquire_args(
        &self,
        args: Self::Args,
    ) -> Result<(crate::durable::documents::DocArgs, Option<JsonValue>)> {
        Ok((args, None))
    }
}

impl RewindableDocAccess for RawToken {}

impl ErasedReader for SessionImpl {
    fn snapshot_json(
        &self,
        definition: Arc<crate::durable::documents::AnyDocDefinition>,
        args: crate::durable::documents::DocArgs,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<Arc<JsonValue>>>> {
        SessionImpl::snapshot_json(self, &RawToken(definition), args, context)
    }

    fn snapshot_as_of_json(
        &self,
        definition: Arc<crate::durable::documents::AnyDocDefinition>,
        args: crate::durable::documents::DocArgs,
        at: EntryId,
        context: &Context,
    ) -> BoxFuture<'static, Result<Option<JsonValue>>> {
        SessionImpl::snapshot_as_of(self, &RawToken(definition), args, at, context)
    }
}

/// Calls hook handlers of one task name (`HookRunner<H>`).
pub struct HookRunner<H> {
    core: Arc<RuntimeCore>,
    _hooks: PhantomData<fn() -> H>,
}

impl<H: Send + Sync + 'static> HookRunner<H> {
    /// Call `invoke` with each matching handler `select` picks. An ordinary error from `invoke` is reported and the
    /// next handler runs; once the invocation is signalled, the error propagates.
    pub async fn each<F, G, Fut>(
        &self,
        select: impl Fn(&H) -> Option<F>,
        mut invoke: G,
    ) -> Result<()>
    where
        G: FnMut(F) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let agent = self.core.agent(&self.core.invocation.context).await?;
        for registration in agent_hooks(&agent, &self.core.task_name()) {
            let Some(handlers) = hook_handlers::<H>(&registration) else {
                continue;
            };
            let Some(handler) = select(&handlers) else {
                continue;
            };
            if let Err(error) = invoke(handler).await {
                if self.core.invocation.controller.signal().aborted() {
                    return Err(error);
                }
                self.core.scheduler.report(error);
            }
        }
        Ok(())
    }
}

/// What a hook may use: committed reads and the asking task's memos, which hooks and the task share (`HookApi`).
#[derive(Clone)]
pub struct HookApi(pub(crate) Arc<RuntimeCore>);

impl HookApi {
    pub fn task_id(&self) -> TaskId {
        self.0.task_id()
    }

    pub fn conversation_id(&self) -> ConversationId {
        self.0.conversation_id()
    }

    pub async fn memo<T: DeserializeOwned>(&self, name: &str) -> Result<Option<T>> {
        decode_memo(self.0.memo(name).await?)
    }

    pub async fn memo_with<T: Serialize + DeserializeOwned>(
        &self,
        name: &str,
        candidate: T,
        context: &Context,
    ) -> Result<T> {
        let candidate = copy_json(&candidate, None)?;
        let winner = self.0.memo_with(name, candidate, context).await?;
        Ok(serde_json::from_value(winner)?)
    }

    pub fn reader(&self) -> DocumentReader {
        self.0.reader()
    }

    pub async fn snapshot<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> Result<Option<A::Value>> {
        self.reader().snapshot(token, address, context).await
    }
}

fn decode_memo<T: DeserializeOwned>(value: Option<JsonValue>) -> Result<Option<T>> {
    value
        .map(|value| serde_json::from_value(value).map_err(Error::from))
        .transpose()
}

/// Marker of a runtime's task type parameters.
type Types<I, S, R, H> = PhantomData<fn() -> (I, S, R, H)>;

/// Operations of one task invocation (`TaskRuntime<I, S, R, H>`). Every operation rejects after the invocation ends;
/// watches acquired through it stop at invocation end.
pub struct TaskRuntime<I, S, R, H> {
    core: Arc<RuntimeCore>,
    _types: Types<I, S, R, H>,
}

impl<I, S, R, H> Clone for TaskRuntime<I, S, R, H> {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            _types: PhantomData,
        }
    }
}

impl<I, S, R, H> TaskRuntime<I, S, R, H> {
    pub(crate) fn new(core: Arc<RuntimeCore>) -> Self {
        Self {
            core,
            _types: PhantomData,
        }
    }

    pub fn core(&self) -> &Arc<RuntimeCore> {
        &self.core
    }

    pub fn task_id(&self) -> TaskId<R> {
        self.core.task_id().cast()
    }

    pub fn conversation_id(&self) -> ConversationId {
        self.core.conversation_id()
    }

    /// Aborted when the run is signalled by `abortTask()`, the Harness closes, or the invocation ends.
    pub fn signal(&self) -> AbortSignal {
        self.core.signal()
    }

    /// Registry snapshot of the current phase; refreshed at every phase boundary.
    pub fn registry(&self) -> RegistrySnapshot {
        self.core.registry()
    }

    pub async fn agent(&self, context: &Context) -> Result<Agent> {
        self.core.agent(context).await
    }

    /// `HarnessOptions.settings`, resolved at each access.
    pub fn settings(&self) -> Settings {
        self.core.settings()
    }

    pub fn models(&self) -> Models {
        self.core.models()
    }

    pub async fn env(&self, context: &Context) -> Result<Option<Arc<dyn ExecutionEnv>>> {
        self.core.env(context).await
    }

    /// Handlers of this task's name from the extensions its conversation selects, in extension order.
    pub fn hooks(&self) -> HookRunner<H> {
        HookRunner {
            core: self.core.clone(),
            _hooks: PhantomData,
        }
    }

    /// The runtime as the `api` hooks receive.
    pub fn hook_api(&self) -> HookApi {
        HookApi(self.core.clone())
    }

    /// Committed document reads through this invocation.
    pub fn reader(&self) -> DocumentReader {
        self.core.reader()
    }

    /// Read a durable memo of this task.
    pub async fn memo<T: DeserializeOwned>(&self, name: &str) -> Result<Option<T>> {
        decode_memo(self.core.memo(name).await?)
    }

    /// Store `candidate` unless a memo already exists; return the durable winner.
    pub async fn memo_with<T: Serialize + DeserializeOwned>(
        &self,
        name: &str,
        candidate: T,
        context: &Context,
    ) -> Result<T> {
        HookApi(self.core.clone())
            .memo_with(name, candidate, context)
            .await
    }

    /// Committed task record.
    pub async fn get_task<T>(
        &self,
        id: TaskId<T>,
        context: &Context,
    ) -> Result<Option<TaskRecord>> {
        self.core.get_task(id.erase(), context).await
    }

    /// Resolve with the task's terminal receipt; rejects when the invocation ends.
    pub async fn wait_for_task<T>(&self, id: TaskId<T>, context: &Context) -> Result<TaskRecord> {
        self.core.wait_for_task(id.erase(), context).await
    }

    /// Outcomes of terminal tasks, in order; rejects when one is missing or not terminal.
    pub async fn outcomes<T>(
        &self,
        ids: &[TaskId<T>],
        context: &Context,
    ) -> Result<Vec<TaskOutcome>> {
        let ids: Vec<TaskId> = ids.iter().map(|id| id.erase()).collect();
        self.core.outcomes(&ids, context).await
    }

    /// Invocation-bound handle of an existing conversation; `None` when absent.
    pub async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<ConversationHandle>> {
        self.core.conversation(id, context).await
    }

    /// Committed entry visible from the task's conversation.
    pub async fn entry(&self, id: EntryId, context: &Context) -> Result<Option<EntryRecord>> {
        self.core.entry(id, context).await
    }

    /// `None` when the entry is absent, not visible, or has another kind.
    pub async fn entry_of<D>(
        &self,
        token: &Entry<D>,
        id: EntryId,
        context: &Context,
    ) -> Result<Option<EntryRecord>> {
        Ok(self
            .core
            .entry(id, context)
            .await?
            .filter(|entry| entry.kind == token.kind()))
    }

    /// Committed raw active transcript and model context, optionally cut off at the visible entry `at`.
    pub async fn context(
        &self,
        conversation_id: ConversationId,
        context: &Context,
        at: Option<EntryId>,
    ) -> Result<ContextView> {
        self.core.context(conversation_id, context, at).await
    }

    /// The Harness clock.
    pub fn now(&self) -> Result<u64> {
        self.core.now()
    }

    /// Forward a non-fatal failure to `HarnessOptions.onReport`.
    pub fn report(&self, error: Error) -> Result<()> {
        self.core.report(error)
    }

    /// Resolve once the Harness clock reaches `until`; rejects when the invocation or `context` is cancelled.
    pub async fn sleep(&self, until: u64, context: &Context) -> Result<()> {
        self.core.sleep(until, context).await
    }

    pub async fn snapshot<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> Result<Option<A::Value>> {
        self.core.check()?;
        self.core
            .scheduler
            .session()
            .snapshot(token, address, context)
            .await
    }

    pub async fn snapshot_as_of<A: RewindableDocAccess>(
        &self,
        token: &A,
        address: A::Address,
        at: EntryId,
        context: &Context,
    ) -> Result<Option<A::Value>> {
        self.core.check()?;
        self.core
            .scheduler
            .session()
            .snapshot_as_of(token, address, at, context)
            .await
    }

    /// Watch a document through this invocation; the watch stops when the invocation ends.
    pub async fn watch_doc<A: DocAccess>(
        &self,
        token: &A,
        address: A::Address,
        context: &Context,
    ) -> Result<Option<DocumentWatch>> {
        self.core.check()?;
        let watch = self
            .core
            .scheduler
            .session()
            .watch_doc(token, address, context)
            .await?;
        self.core.watch(watch).await
    }
}

impl<I, S, R, H> TaskRuntime<I, S, R, H>
where
    I: DeserializeOwned + Send + 'static,
    S: Serialize + DeserializeOwned + Send + 'static,
    R: 'static,
{
    /// Commit on the Session line after rereading the task. Rejects when the task is terminal, the invocation ended,
    /// the Harness is closing, or, in a run invocation, the task carries an abort mark. A returned state replaces the
    /// task's state in the same commit; `None` leaves it unchanged. `tx.create_task()` defaults to the task's
    /// conversation.
    pub async fn commit<F, Fut>(&self, change: F, context: &Context) -> Result<()>
    where
        F: FnOnce(Transaction, RunningTask<I, S, R>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Option<NextTaskState<S>>>> + Send + 'static,
    {
        self.core
            .commit(
                move |tx, record| async move {
                    let current = RunningTask::<I, S, R>::from_record(&record)?;
                    match change(tx, current).await? {
                        None => Ok(None),
                        Some(next) => next.erase().map(Some),
                    }
                },
                context,
            )
            .await
    }
}

impl<I, S, R, H> std::fmt::Debug for TaskRuntime<I, S, R, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskRuntime")
            .field("task_id", &self.core.task_id())
            .finish_non_exhaustive()
    }
}

/// Erased JSON object helper for task inputs.
pub fn empty_object() -> JsonObject {
    JsonObject::new()
}
