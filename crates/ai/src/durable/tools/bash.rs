//! Port of durable `src/tools/bash.ts`.
//!
//! Divergence from Pi: `BashPrepare` takes the execution by value and returns it (TS mutates the object it gets).

use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::chord::{Context, JsonValue};
use crate::durable::env::{ExecutionErrorCode, OnOutput, ShellExecOptions, ShellSpillOptions};
use crate::durable::errors::{Error, Result};
use crate::durable::harness::define::define_tool;
use crate::durable::harness::tool::ToolExecutionApi;
use crate::durable::harness::types::{
    DiagnosticSeverity, Retain, ToolDiagnostic, ToolExecutionResult, ToolOutputLimits,
    ToolRegistration,
};
use crate::durable::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};

use super::env::require_env;
use super::{js_number, parse_args};

const MAX_TIMEOUT_SECONDS: f64 = 2_147_483_647.0 / 1000.0;

fn bash_schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "Bash command to execute" },
            "timeout": { "type": "number", "description": "Timeout in seconds (optional, no default timeout)" },
        },
        "required": ["command"],
    })
}

/// `BashToolInput`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BashToolInput {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
}

/// `BashExecution`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BashExecution {
    pub command: String,
    pub cwd: String,
    pub env: IndexMap<String, String>,
    pub inherit_env: bool,
}

/// `BashPrepare`: adjust the execution with the call's api and context.
pub type BashPrepare = Arc<
    dyn Fn(BashExecution, ToolExecutionApi, Context) -> BoxFuture<'static, Result<BashExecution>>
        + Send
        + Sync,
>;

/// `BashToolOptions`.
#[derive(Clone, Default)]
pub struct BashToolOptions {
    pub command_prefix: Option<String>,
    pub prepare: Option<BashPrepare>,
}

impl std::fmt::Debug for BashToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashToolOptions")
            .field("command_prefix", &self.command_prefix)
            .field("prepare", &self.prepare.is_some())
            .finish()
    }
}

fn validate_timeout(timeout: Option<f64>) -> Result<()> {
    let Some(timeout) = timeout else {
        return Ok(());
    };
    if !timeout.is_finite() || timeout <= 0.0 {
        return Err(Error::message(
            "Invalid timeout: must be a finite number of seconds",
        ));
    }
    if timeout > MAX_TIMEOUT_SECONDS {
        return Err(Error::message(format!(
            "Invalid timeout: maximum is {} seconds",
            js_number(MAX_TIMEOUT_SECONDS)
        )));
    }
    Ok(())
}

/// Runs a command through the environment's shell. Its output streams to `api.output()`, where the Harness keeps
/// the tail within the default limits; the result content is that retained output. Output beyond the limits is
/// spilled to a file whose path is reported as a diagnostic. A nonzero exit or timeout throws, which makes an error
/// result that still carries the output and diagnostics.
pub fn create_bash_tool(options: BashToolOptions) -> ToolRegistration {
    let options = Arc::new(options);
    ToolRegistration {
        output_limits: Some(ToolOutputLimits {
            retain: Some(Retain::Tail),
            ..ToolOutputLimits::default()
        }),
        ..define_tool(
            "bash",
            format!(
                "Execute a bash command in the current working directory. Returns combined stdout and stderr. Output is truncated to last {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
                DEFAULT_MAX_BYTES / 1024
            ),
            bash_schema(),
            move |args, api, context| execute(options.clone(), args, api, context),
        )
    }
}

async fn execute(
    options: Arc<BashToolOptions>,
    args: JsonValue,
    api: ToolExecutionApi,
    context: Context,
) -> Result<ToolExecutionResult> {
    let BashToolInput { command, timeout } = parse_args(args)?;
    validate_timeout(timeout)?;
    let env = require_env(&api)?;
    let mut execution = BashExecution {
        command: match options.command_prefix.as_deref() {
            Some(prefix) if !prefix.is_empty() => format!("{prefix}\n{command}"),
            _ => command,
        },
        cwd: env.cwd(),
        env: IndexMap::new(),
        inherit_env: true,
    };
    if let Some(prepare) = &options.prepare {
        execution = prepare(execution, api.clone(), context.clone()).await?;
    }
    let output_api = api.clone();
    let on_output: OnOutput = Arc::new(move |text, _| output_api.output(text));
    let result = env
        .exec(
            &execution.command,
            ShellExecOptions {
                cwd: Some(execution.cwd),
                env: Some(execution.env),
                inherit_env: Some(execution.inherit_env),
                timeout,
                on_output: Some(on_output),
                spill: Some(ShellSpillOptions {
                    after_bytes: DEFAULT_MAX_BYTES as u64,
                    after_lines: DEFAULT_MAX_LINES as u64,
                }),
            },
            &context,
        )
        .await;
    let spill_path = match &result {
        Ok(value) => value.spill_path.clone(),
        Err(error) => error.spill_path.clone(),
    };
    if let Some(spill_path) = spill_path {
        api.diagnostic(ToolDiagnostic {
            severity: DiagnosticSeverity::Info,
            code: Some("full_output".into()),
            message: format!("Full output: {spill_path}"),
        })?;
    }
    let value = match result {
        Ok(value) => value,
        Err(error) => {
            let context_aborted = context
                .abort_signal()
                .is_some_and(|signal| signal.aborted());
            if error.code == ExecutionErrorCode::Aborted && context_aborted {
                return Err(Error::thrown(error));
            }
            if error.code == ExecutionErrorCode::Timeout {
                return Err(Error::message(format!(
                    "Command timed out after {} seconds",
                    timeout.map(js_number).unwrap_or_else(|| "undefined".into())
                )));
            }
            if error.code == ExecutionErrorCode::Aborted {
                return Err(Error::message("Command aborted"));
            }
            return Err(Error::thrown(error));
        }
    };
    if value.exit_code != 0 {
        return Err(Error::message(format!(
            "Command exited with code {}",
            value.exit_code
        )));
    }
    Ok(ToolExecutionResult::default())
}
