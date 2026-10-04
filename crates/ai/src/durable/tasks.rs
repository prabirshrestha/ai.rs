//! Port of durable `src/tasks.ts` (`defineTask`) and the task definition types
//! of `src/types.ts`.
//!
//! Types only for now: the Session needs `name`, `version` and `initial` to
//! create task records. The phase map, the abort handler and `TaskRuntime`
//! depend on the harness (registry, models, env) and arrive with the
//! scheduler, together with the typed `TaskRecord<I, S, R>` views.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::chord::JsonValue;

use super::errors::Result;

/// `migrate(input, checkpoint, fromVersion)`: convert a record stored by an older supported version.
pub type TaskMigrate<I, S> = Arc<dyn Fn(JsonValue, JsonValue, u32) -> Result<(I, S)> + Send + Sync>;

/// Executable durable state machine definition, registered by `name` (`TaskDefinition<I, S, R, H>`).
pub struct TaskDefinition<I, S, R, H = ()> {
    /// Registered task kind persisted in `TaskRecord.kind`.
    pub name: String,
    /// Definition version persisted with live input and checkpoints.
    pub version: u32,
    /// First durable checkpoint for a newly created task.
    pub initial: Arc<dyn Fn(&I) -> S + Send + Sync>,
    pub migrate: Option<TaskMigrate<I, S>>,
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
            migrate: None,
            hooks: None,
            _result: PhantomData,
        }
    }
}

impl<I, S, R, H: Clone> Clone for TaskDefinition<I, S, R, H> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            version: self.version,
            initial: self.initial.clone(),
            migrate: self.migrate.clone(),
            hooks: self.hooks.clone(),
            _result: PhantomData,
        }
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
    pub definition: TaskDefinition<I, S, R, H>,
}

impl<I, S, R, H: Clone> Clone for Task<I, S, R, H> {
    fn clone(&self) -> Self {
        Self {
            definition: self.definition.clone(),
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

/// Define an executable task. Register it in the registry so a Harness can run tasks of its kind.
pub fn define_task<I, S, R, H>(definition: TaskDefinition<I, S, R, H>) -> Task<I, S, R, H> {
    Task { definition }
}
