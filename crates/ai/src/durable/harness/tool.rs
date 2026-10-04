//! Port of durable `src/harness/tool.ts` (milestone 6 subset: the tool task's identity).

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::chord::JsonValue;
use crate::durable::ids::EntryId;
use crate::durable::tasks::{Task, TaskDefinition, define_task};

use super::types::ToolHooks;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolTaskInput {
    pub assistant: EntryId,
    pub call_id: String,
}

/// What a tool's execute function may use (milestone 7).
#[derive(Clone)]
pub struct ToolExecutionApi(());

/// Built-in tool task.
pub static TOOL_TASK: LazyLock<Task<ToolTaskInput, JsonValue, JsonValue, ToolHooks>> =
    LazyLock::new(|| {
        define_task(TaskDefinition::new(
            "pi.tool",
            1,
            |_: &ToolTaskInput| serde_json::json!({ "phase": "intent" }),
        ))
    });
