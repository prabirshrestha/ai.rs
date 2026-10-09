//! Port of durable `src/tools/write.ts`.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::chord::{Context, JsonValue};
use crate::durable::env::get_or_throw;
use crate::durable::errors::Result;
use crate::durable::harness::define::define_tool;
use crate::durable::harness::tool::ToolExecutionApi;
use crate::durable::harness::types::{ToolExecutionResult, ToolRegistration};

use super::env::require_env;
use super::file_mutation_queue::with_file_mutation_queue;
use super::path_utils::resolve_tool_path;
use super::{parse_args, throw_if_aborted};

fn write_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to the file to write (relative or absolute)" },
            "content": { "type": "string", "description": "Content to write to the file" },
        },
        "required": ["path", "content"],
    })
}

/// `WriteToolInput`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteToolInput {
    pub path: String,
    pub content: String,
}

pub fn create_write_tool() -> ToolRegistration {
    define_tool(
        "write",
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        write_schema(),
        execute,
    )
}

async fn execute(
    args: JsonValue,
    api: ToolExecutionApi,
    context: Context,
) -> Result<ToolExecutionResult> {
    let WriteToolInput { path, content } = parse_args(args)?;
    let env = require_env(&api)?;
    let absolute_path = resolve_tool_path(env.as_ref(), &path, &context).await?;
    with_file_mutation_queue(
        env.as_ref(),
        &absolute_path,
        || async {
            throw_if_aborted(&context)?;
            get_or_throw(
                env.write_file(&absolute_path, content.as_bytes(), &context)
                    .await,
            )?;
            throw_if_aborted(&context)?;
            Ok(ToolExecutionResult::text(format!(
                "Successfully wrote to {path}"
            )))
        },
        &context,
    )
    .await
}
