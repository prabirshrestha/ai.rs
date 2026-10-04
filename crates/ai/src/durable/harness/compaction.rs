//! Port of durable `src/harness/compaction.ts` (milestone 6 subset: the compaction task's identity).

use std::sync::LazyLock;

use crate::chord::JsonValue;
use crate::durable::tasks::{Task, TaskDefinition, define_task};

use super::types::{CompactionHooks, CompactionResult};

/// Built-in compaction task.
pub static COMPACTION_TASK: LazyLock<
    Task<JsonValue, JsonValue, CompactionResult, CompactionHooks>,
> = LazyLock::new(|| {
    define_task(TaskDefinition::new(
        "pi.compaction",
        1,
        |_: &JsonValue| serde_json::json!({ "phase": "summarize" }),
    ))
});
