//! Port of durable `src/tasks.ts` (`defineTask`) and the task definition types
//! of `src/types.ts` (`TaskDefinition`, `PhaseHandler`, `RunningTask`,
//! `NextTaskState`, `Task`).
//!
//! Divergences from Pi:
//! - The phase map is keyed by the checkpoint's `phase` string, read from the
//!   serialized checkpoint; `S` is any serde type whose JSON has a `phase`.
//!   A checkpoint whose phase has no handler faults the task.
//! - `RunningTask` exposes the running checkpoint as `checkpoint` (TS
//!   `task.state.checkpoint`).
//! - `NextTaskState` carries the erased [`TaskOutcome`]; [`NextTaskState::completed`]
//!   serializes a typed result.
//! - `define_task` also builds the erased registry form ([`AnyTask`]) once, so a
//!   task keeps one identity in registries (TS object identity).

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::chord::{Context, JsonValue, copy_json};

use super::errors::{Error, Result};
use super::harness::scheduler::{RuntimeCore, TaskRuntime};
use super::ids::{ConversationId, TaskId};
use super::types::{JoinPolicy, JsonObject, TaskOutcome, TaskOutcomeError, TaskRecord, TaskState};

/// `migrate(input, checkpoint, fromVersion)`: convert a record stored by an older supported version.
pub type TaskMigrate<I, S> = Arc<dyn Fn(JsonValue, JsonValue, u32) -> Result<(I, S)> + Send + Sync>;

/// Live task record reserved by one invocation (`RunningTask<I, S, R>`).
#[derive(Debug, Clone, PartialEq)]
pub struct RunningTask<I, S, R = JsonValue> {
    pub id: TaskId<R>,
    pub conversation_id: ConversationId,
    pub kind: String,
    pub version: u32,
    pub input: I,
    pub owner: Option<TaskId>,
    pub background: bool,
    pub abort_requested: bool,
    /// The running checkpoint (TS `state.checkpoint`).
    pub checkpoint: S,
    pub memos: Option<JsonObject>,
}

impl<I: DeserializeOwned, S: DeserializeOwned, R> RunningTask<I, S, R> {
    /// Decode an erased running record.
    pub fn from_record(record: &TaskRecord) -> Result<Self> {
        let TaskState::Running { checkpoint } = &record.state else {
            return Err(Error::message(format!(
                "Task {} is {}",
                record.id,
                record.state.status()
            )));
        };
        Ok(Self {
            id: record.id.cast(),
            conversation_id: record.conversation_id,
            kind: record.kind.clone(),
            version: record.version,
            input: decode(&record.kind, "input", &record.input)?,
            owner: record.owner,
            background: record.background,
            abort_requested: record.abort_requested,
            checkpoint: decode(&record.kind, "checkpoint", checkpoint)?,
            memos: record.memos.clone(),
        })
    }
}

fn decode<T: DeserializeOwned>(kind: &str, what: &str, value: &JsonValue) -> Result<T> {
    T::deserialize(value).map_err(|error| {
        Error::type_error(format!(
            "Task {kind} {what} does not match its definition: {error}"
        ))
    })
}

/// Next state a task commits for itself: a replacement checkpoint, a wait, or its outcome (`NextTaskState<S, R>`).
/// A returned terminal state is stored as `completing` while ordinary owned work below the task is live.
#[derive(Debug, Clone, PartialEq)]
pub enum NextTaskState<S> {
    Running {
        checkpoint: S,
    },
    Waiting {
        checkpoint: S,
        on: Vec<TaskId>,
        policy: JoinPolicy,
    },
    Terminal {
        outcome: TaskOutcome,
    },
}

impl<S> NextTaskState<S> {
    /// `{ status: "terminal", outcome: { status: "completed", result } }`.
    pub fn completed<R: Serialize>(result: R) -> Self {
        Self::Terminal {
            outcome: TaskOutcome::Completed {
                result: copy_json(&result, None).unwrap_or(JsonValue::Null),
            },
        }
    }

    /// `{ status: "terminal", outcome: { status: "aborted", reason } }`.
    pub fn aborted(reason: impl Into<String>) -> Self {
        Self::Terminal {
            outcome: TaskOutcome::Aborted {
                reason: Some(reason.into()),
                result: None,
            },
        }
    }

    /// `{ status: "terminal", outcome: { status: "failed", error: { message } } }`.
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Terminal {
            outcome: TaskOutcome::Failed {
                error: TaskOutcomeError {
                    message: message.into(),
                    detail: None,
                },
                result: None,
            },
        }
    }

    /// `{ status: "running", checkpoint }`.
    pub fn running(checkpoint: S) -> Self {
        Self::Running { checkpoint }
    }

    /// `{ status: "waiting", checkpoint, on, policy }`.
    pub fn waiting(checkpoint: S, on: Vec<TaskId>, policy: JoinPolicy) -> Self {
        Self::Waiting {
            checkpoint,
            on,
            policy,
        }
    }
}

impl<S: Serialize> NextTaskState<S> {
    /// The erased state written to the record.
    pub fn erase(self) -> Result<TaskState> {
        Ok(match self {
            Self::Running { checkpoint } => TaskState::Running {
                checkpoint: copy_json(&checkpoint, None)?,
            },
            Self::Waiting {
                checkpoint,
                on,
                policy,
            } => TaskState::Waiting {
                checkpoint: copy_json(&checkpoint, None)?,
                on,
                policy,
            },
            Self::Terminal { outcome } => TaskState::Terminal { outcome },
        })
    }
}

/// Runs one checkpoint phase (`PhaseHandler`). It must commit a changed checkpoint or a terminal outcome through
/// `runtime.commit()`; returning without durable progress faults the task.
pub type PhaseHandler<I, S, R, H> = Arc<
    dyn Fn(RunningTask<I, S, R>, TaskRuntime<I, S, R, H>, Context) -> BoxFuture<'static, Result<()>>
        + Send
        + Sync,
>;

/// Executable durable state machine definition, registered by `name` (`TaskDefinition<I, S, R, H>`).
pub struct TaskDefinition<I, S, R, H = ()> {
    /// Registered task kind persisted in `TaskRecord.kind`.
    pub name: String,
    /// Definition version persisted with live input and checkpoints.
    pub version: u32,
    /// First durable checkpoint for a newly created task.
    pub initial: Arc<dyn Fn(&I) -> S + Send + Sync>,
    /// Exhaustive phase map, keyed by the checkpoint's `phase`.
    pub phases: IndexMap<String, PhaseHandler<I, S, R, H>>,
    /// Runs in a fresh invocation after an abort mark and must commit a terminal outcome.
    pub abort: Option<PhaseHandler<I, S, R, H>>,
    pub migrate: Option<TaskMigrate<I, S>>,
    /// Typing only, as in Pi: hooks come from extensions (`hook()`).
    pub hooks: Option<H>,
    _result: PhantomData<fn() -> R>,
}

impl<I, S, R, H> TaskDefinition<I, S, R, H> {
    pub fn new(
        name: impl Into<String>,
        version: u32,
        initial: impl Fn(&I) -> S + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            version,
            initial: Arc::new(initial),
            phases: IndexMap::new(),
            abort: None,
            migrate: None,
            hooks: None,
            _result: PhantomData,
        }
    }

    /// Add the handler of phase `name`.
    pub fn phase<F, Fut>(mut self, name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(RunningTask<I, S, R>, TaskRuntime<I, S, R, H>, Context) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        self.phases.insert(
            name.into(),
            Arc::new(move |task, runtime, context| Box::pin(handler(task, runtime, context))),
        );
        self
    }

    /// Set the abort handler.
    pub fn abort<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(RunningTask<I, S, R>, TaskRuntime<I, S, R, H>, Context) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        self.abort = Some(Arc::new(move |task, runtime, context| {
            Box::pin(handler(task, runtime, context))
        }));
        self
    }

    pub fn migrate(
        mut self,
        migrate: impl Fn(JsonValue, JsonValue, u32) -> Result<(I, S)> + Send + Sync + 'static,
    ) -> Self {
        self.migrate = Some(Arc::new(migrate));
        self
    }

    pub fn hooks(mut self, hooks: H) -> Self {
        self.hooks = Some(hooks);
        self
    }
}

impl<I, S, R, H> fmt::Debug for TaskDefinition<I, S, R, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskDefinition")
            .field("name", &self.name)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

/// Typed executable task definition (`Task<I, S, R, H>`).
pub struct Task<I, S, R, H = ()> {
    pub definition: Arc<TaskDefinition<I, S, R, H>>,
    erased: AnyTask,
}

impl<I, S, R, H> Task<I, S, R, H> {
    /// The erased registry form; the same value every time (TS object identity).
    pub fn any(&self) -> AnyTask {
        self.erased.clone()
    }

    pub fn name(&self) -> &str {
        &self.definition.name
    }
}

impl<I, S, R, H> Clone for Task<I, S, R, H> {
    fn clone(&self) -> Self {
        Self {
            definition: self.definition.clone(),
            erased: self.erased.clone(),
        }
    }
}

impl<I, S, R, H> fmt::Debug for Task<I, S, R, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Task")
            .field("definition", &self.definition)
            .finish()
    }
}

impl<I, S, R, H> From<&Task<I, S, R, H>> for AnyTask {
    fn from(task: &Task<I, S, R, H>) -> Self {
        task.any()
    }
}

impl<I, S, R, H> From<Task<I, S, R, H>> for AnyTask {
    fn from(task: Task<I, S, R, H>) -> Self {
        task.erased
    }
}

/// Erased phase handler run by the scheduler on the stored record.
pub(crate) type ErasedPhase = Arc<
    dyn Fn(TaskRecord, Arc<RuntimeCore>, Context) -> BoxFuture<'static, Result<()>> + Send + Sync,
>;

type ErasedMigrate =
    Arc<dyn Fn(JsonValue, JsonValue, u32) -> Result<(JsonValue, JsonValue)> + Send + Sync>;

/// Erased executable task definition stored in the registry (`AnyTask["definition"]`).
pub struct ErasedTaskDefinition {
    pub name: String,
    pub version: u32,
    pub(crate) phases: IndexMap<String, ErasedPhase>,
    pub(crate) abort: ErasedPhase,
    pub(crate) migrate: Option<ErasedMigrate>,
}

impl ErasedTaskDefinition {
    pub fn has_migrate(&self) -> bool {
        self.migrate.is_some()
    }
}

/// Erased executable task definition stored in the registry (`AnyTask`). Compares by identity.
#[derive(Clone)]
pub struct AnyTask(Arc<ErasedTaskDefinition>);

impl AnyTask {
    pub fn definition(&self) -> &ErasedTaskDefinition {
        &self.0
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Migrate a stored input and checkpoint to this definition's version.
    pub(crate) fn run_migrate(
        &self,
        input: JsonValue,
        checkpoint: JsonValue,
        from_version: u32,
    ) -> Option<Result<(JsonValue, JsonValue)>> {
        self.0
            .migrate
            .as_ref()
            .map(|migrate| migrate(input, checkpoint, from_version))
    }

    pub(crate) fn phase(&self, name: &str) -> Option<ErasedPhase> {
        self.0.phases.get(name).cloned()
    }

    pub(crate) fn abort_handler(&self) -> ErasedPhase {
        self.0.abort.clone()
    }
}

impl PartialEq for AnyTask {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
    }
}

impl fmt::Debug for AnyTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyTask")
            .field("name", &self.0.name)
            .field("version", &self.0.version)
            .finish_non_exhaustive()
    }
}

fn erase_handler<I, S, R, H>(handler: PhaseHandler<I, S, R, H>) -> ErasedPhase
where
    I: DeserializeOwned + Send + Sync + 'static,
    S: DeserializeOwned + Send + Sync + 'static,
    R: 'static,
    H: Send + Sync + 'static,
{
    Arc::new(move |record, core, context| {
        let task = RunningTask::<I, S, R>::from_record(&record);
        let runtime = TaskRuntime::<I, S, R, H>::new(core);
        match task {
            Ok(task) => handler(task, runtime, context),
            Err(error) => Box::pin(futures::future::ready(Err(error))),
        }
    })
}

/// Define an executable task. Register it in the registry so a Harness can run tasks of its kind.
pub fn define_task<I, S, R, H>(definition: TaskDefinition<I, S, R, H>) -> Task<I, S, R, H>
where
    I: Serialize + DeserializeOwned + Send + Sync + 'static,
    S: Serialize + DeserializeOwned + Send + Sync + 'static,
    R: 'static,
    H: Send + Sync + 'static,
{
    let definition = Arc::new(definition);
    let phases = definition
        .phases
        .iter()
        .map(|(name, handler)| (name.clone(), erase_handler(handler.clone())))
        .collect();
    let abort = match &definition.abort {
        Some(handler) => erase_handler(handler.clone()),
        None => Arc::new(|_, _, _| {
            Box::pin(futures::future::ready(Ok(()))) as BoxFuture<'static, Result<()>>
        }) as ErasedPhase,
    };
    let migrate = definition.migrate.clone().map(|migrate| {
        Arc::new(move |input, checkpoint, from_version| {
            let (input, checkpoint) = migrate(input, checkpoint, from_version)?;
            Ok((copy_json(&input, None)?, copy_json(&checkpoint, None)?))
        }) as ErasedMigrate
    });
    let erased = AnyTask(Arc::new(ErasedTaskDefinition {
        name: definition.name.clone(),
        version: definition.version,
        phases,
        abort,
        migrate,
    }));
    Task { definition, erased }
}
