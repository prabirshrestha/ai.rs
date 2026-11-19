//! Port of `agent/internal/run-prompt.ts`.
//!
//! [rust] The TS driver is an async IIFE whose closures share `let`
//! bindings. Here the bindings live in one [`State`] behind a mutex that is
//! never held across an `await`, and the driver runs as a spawned task so it
//! starts right away, like the IIFE. Host tool executions run as their own
//! tasks, joined at step boundaries.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::{BoxFuture, Shared};
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::json;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::harness::agent::internal::harness_stream_text_result::{
    HarnessStreamTextResult, HarnessStreamTextResultOptions,
};
use crate::harness::agent::internal::permission_mode::{
    CustomToolApprovalDecision, resolve_custom_tool_approval,
};
use crate::harness::agent::internal::strip_work_dir::{
    create_tool_input_work_dir_stripper, strip_parsed_tool_input_work_dir, strip_work_dir,
};
use crate::harness::agent::internal::to_harness_stream::to_harness_stream;
use crate::harness::agent::internal::translate_stream_part::{
    TranslateOptions, TranslatedToolCallInfo, translate_stream_part,
};
use crate::harness::agent::internal::turn_telemetry::{
    HarnessAgentLifecycleCallbacks, LanguageModelCallEndInput, TurnLifecycle, TurnLifecycleOptions,
    create_turn_lifecycle,
};
use crate::harness::agent::settings::HarnessAgentToolApprovalConfiguration;
use crate::harness::errors::{AbortError, HarnessError, HarnessResult};
use crate::harness::provider::{
    JsonObject, JsonValue, LanguageModelV4FinishReason, LanguageModelV4UnifiedFinishReason,
    LanguageModelV4Usage, SharedV4ProviderMetadata, safe_parse_json,
};
use crate::harness::sandbox::SandboxSession;
use crate::harness::sdk::content::{
    ContentPart, Context, ProviderMetadata, ReasoningOutput, StepResult, TextContent,
    ToolApprovalRequestOutput, ToolApprovalResponseOutput, ToolOutput, ToolsContext, TypedToolCall,
    TypedToolError, TypedToolResult,
};
use crate::harness::sdk::error_value::{ErrorValue, generate_id};
use crate::harness::sdk::messages::{
    ModelMessage, ToolApprovalResponse, ToolResultOutput, ToolResultPart, UserContent,
    UserContentPart, UserModelMessage,
};
use crate::harness::sdk::output::SharedOutput;
use crate::harness::sdk::parse_tool_call::{
    LanguageModelV4ToolCall, parse_tool_call, validate_tool_context,
};
use crate::harness::sdk::stop_condition::{StopCondition, is_stop_condition_met};
use crate::harness::sdk::telemetry::TelemetryOptions;
use crate::harness::sdk::text_stream_part::TextStreamPart;
use crate::harness::sdk::tool::{
    ExecuteToolPart, ToolExecutionOptions, ToolModelOutputErrorMode, ToolSet,
    create_tool_model_output, execute_tool, is_executable_tool,
};
use crate::harness::sdk::usage::{
    add_language_model_usage, as_language_model_usage, create_null_language_model_usage,
};
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1ContinueTurnOptions, HarnessV1Prompt, HarnessV1PromptControl,
    HarnessV1PromptTurnOptions, HarnessV1Session, HarnessV1SubmitToolApproval,
    HarnessV1SubmitToolResult,
};
use crate::harness::spec::lifecycle_state::{
    HarnessV1CompletedToolResult, HarnessV1PendingToolApproval, HarnessV1PendingToolApprovalKind,
    HarnessV1PendingToolResult, HarnessV1TurnSettings,
};
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::stream_part::{HarnessV1Metadata, HarnessV1StreamPart};
use crate::harness::spec::tools::{
    HarnessV1BuiltinToolFiltering, HarnessV1Skill, HarnessV1ToolSpec,
    get_harness_v1_builtin_tool_filtering_denial_reason, is_harness_v1_builtin_tool_included,
};
use crate::harness::utils::bridge_diagnostics::{
    BridgeErrorInput, LogBridgeErrorOptions, log_bridge_error,
};
use crate::harness::utils::plain_error;
use crate::harness::utils::tool_model_output::normalize_harness_tool_model_output;

const INVALID_TOOL_INPUT_MESSAGE: &str = "Tool input validation failed.";
const INVALID_TOOL_CONTEXT_MESSAGE: &str = "Tool context validation failed.";

/// `unwrapToolResultOutput(toolResult)`.
fn unwrap_tool_result_output(tool_result: &ToolResultPart) -> HarnessV1CompletedToolResult {
    let (output, is_error) = match &tool_result.output {
        ToolResultOutput::Text { value, .. } => (JsonValue::String(value.clone()), None),
        ToolResultOutput::Json { value, .. } => (value.clone(), None),
        ToolResultOutput::ErrorText { value, .. } => (JsonValue::String(value.clone()), Some(true)),
        ToolResultOutput::ErrorJson { value, .. } => (value.clone(), Some(true)),
        ToolResultOutput::ExecutionDenied { reason, .. } => {
            let mut o = JsonObject::new();
            o.insert("type".into(), "execution-denied".into());
            if let Some(reason) = reason {
                o.insert("reason".into(), reason.clone().into());
            }
            (JsonValue::Object(o), None)
        }
        content @ ToolResultOutput::Content { .. } => (
            serde_json::to_value(content).unwrap_or(JsonValue::Null),
            None,
        ),
    };
    HarnessV1CompletedToolResult {
        output,
        is_error,
        tool_result: None,
    }
}

/// `mode` of [`run_prompt`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RunPromptMode {
    /// Start a new turn from `prompt`.
    #[default]
    Prompt,
    /// Continue the in-flight turn via `doContinueTurn`; `prompt` is
    /// ignored.
    Continue,
}

/// A notification callback with one argument.
pub type Notify<T> = Arc<dyn Fn(T) + Send + Sync>;

/// `onStopConditionMet`.
pub type OnStopConditionMetFn =
    Arc<dyn Fn() -> BoxFuture<'static, HarnessResult<()>> + Send + Sync>;

/// [`run_prompt`] input.
#[derive(Clone)]
pub struct RunPromptInput {
    pub harness: Arc<dyn HarnessV1>,
    pub session: Arc<dyn HarnessV1Session>,
    pub mode: RunPromptMode,
    /// Required for [`RunPromptMode::Prompt`].
    pub prompt: Option<HarnessV1Prompt>,
    pub model: Option<String>,
    pub skills: Vec<HarnessV1Skill>,
    pub instructions: Option<String>,
    pub tools: ToolSet,
    pub tools_context: Option<ToolsContext>,
    pub active_tools: Option<ToolSet>,
    pub tool_specs: Vec<HarnessV1ToolSpec>,
    pub builtin_tool_filtering: Option<HarnessV1BuiltinToolFiltering>,
    pub sandbox_session: Arc<dyn SandboxSession>,
    pub session_work_dir: String,
    pub runtime_context: Context,
    pub abort_signal: Option<CancellationToken>,
    pub response_format: Option<HarnessV1ResponseFormat>,
    pub output: Option<SharedOutput>,
    pub telemetry: Option<TelemetryOptions>,
    pub callbacks: Option<HarnessAgentLifecycleCallbacks>,
    pub stop_conditions: Option<Vec<StopCondition>>,
    pub tool_approval: Option<HarnessAgentToolApprovalConfiguration>,
    pub pending_tool_approvals: Vec<HarnessV1PendingToolApproval>,
    pub pending_tool_results: Vec<HarnessV1PendingToolResult>,
    pub tool_approval_continuations: Vec<ToolApprovalResponse>,
    pub tool_result_continuations: Vec<ToolResultPart>,
    pub on_pending_tool_approval: Option<Notify<HarnessV1PendingToolApproval>>,
    pub on_tool_approval_settled: Option<Notify<String>>,
    pub on_pending_tool_result: Option<Notify<HarnessV1PendingToolResult>>,
    pub on_tool_result_settled: Option<Notify<String>>,
    pub on_turn_finished: Option<Arc<dyn Fn() + Send + Sync>>,
    pub on_turn_failed: Option<Arc<dyn Fn() + Send + Sync>>,
    pub on_prompt_control_available: Option<Notify<Arc<dyn HarnessV1PromptControl>>>,
    /// Reports that the adapter stream closed because the host suspended the
    /// still-running turn at a workflow slice boundary.
    pub is_turn_suspending: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    pub on_stop_condition_met: Option<OnStopConditionMetFn>,
}

impl RunPromptInput {
    /// The required inputs; everything else starts empty.
    pub fn new(
        harness: Arc<dyn HarnessV1>,
        session: Arc<dyn HarnessV1Session>,
        sandbox_session: Arc<dyn SandboxSession>,
        session_work_dir: impl Into<String>,
    ) -> Self {
        Self {
            harness,
            session,
            mode: RunPromptMode::Prompt,
            prompt: None,
            model: None,
            skills: Vec::new(),
            instructions: None,
            tools: ToolSet::new(),
            tools_context: None,
            active_tools: None,
            tool_specs: Vec::new(),
            builtin_tool_filtering: None,
            sandbox_session,
            session_work_dir: session_work_dir.into(),
            runtime_context: Context::new(),
            abort_signal: None,
            response_format: None,
            output: None,
            telemetry: None,
            callbacks: None,
            stop_conditions: None,
            tool_approval: None,
            pending_tool_approvals: Vec::new(),
            pending_tool_results: Vec::new(),
            tool_approval_continuations: Vec::new(),
            tool_result_continuations: Vec::new(),
            on_pending_tool_approval: None,
            on_tool_approval_settled: None,
            on_pending_tool_result: None,
            on_tool_result_settled: None,
            on_turn_finished: None,
            on_turn_failed: None,
            on_prompt_control_available: None,
            is_turn_suspending: None,
            on_stop_condition_met: None,
        }
    }
}

/// What [`run_prompt`] returns.
pub struct RunPromptOutput {
    pub result: HarnessStreamTextResult,
    /// Settles when the driver has finished. Never fails: failures surface
    /// on the result.
    pub done: Shared<BoxFuture<'static, ()>>,
}

/// The `tool-call` stream part fields.
#[derive(Debug, Clone, PartialEq)]
struct RawToolCall {
    tool_call_id: String,
    tool_name: String,
    input: String,
    provider_executed: Option<bool>,
    dynamic: Option<bool>,
    provider_metadata: Option<SharedV4ProviderMetadata>,
    native_name: Option<String>,
}

impl RawToolCall {
    fn from_part(part: &HarnessV1StreamPart) -> Option<Self> {
        match part {
            HarnessV1StreamPart::ToolCall {
                tool_call_id,
                tool_name,
                input,
                provider_executed,
                dynamic,
                provider_metadata,
                native_name,
            } => Some(Self {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                input: input.clone(),
                provider_executed: *provider_executed,
                dynamic: *dynamic,
                provider_metadata: provider_metadata.clone(),
                native_name: native_name.clone(),
            }),
            _ => None,
        }
    }
}

/// A `finish-step` part's fields.
#[derive(Debug, Clone)]
struct FinishStepPart {
    finish_reason: LanguageModelV4FinishReason,
    usage: LanguageModelV4Usage,
    harness_metadata: Option<HarnessV1Metadata>,
}

/// `HostToolOutcome`.
#[derive(Debug, Clone)]
enum HostToolOutcome {
    Ok(JsonValue),
    Err {
        /// Error for host-only consumers such as lifecycle callbacks.
        error: ErrorValue,
        /// Error for stream parts, when `error` has host-only details the
        /// runtime did not receive.
        stream_error: Option<ErrorValue>,
    },
}

/// `getStreamError(outcome)`.
fn get_stream_error(outcome: &HostToolOutcome) -> Option<ErrorValue> {
    match outcome {
        HostToolOutcome::Ok(_) => None,
        HostToolOutcome::Err { stream_error, .. } => stream_error.clone(),
    }
}

/// `toToolOutput({ toolCall, outcome })`.
fn to_tool_output(tool_call: &TypedToolCall, outcome: &HostToolOutcome) -> ToolOutput {
    match outcome {
        HostToolOutcome::Ok(output) => ToolOutput::ToolResult(tool_call.to_result(output.clone())),
        HostToolOutcome::Err { error, .. } => {
            ToolOutput::ToolError(tool_call.to_error(error.clone()))
        }
    }
}

/// `HostToolExecution`.
enum HostToolExecution {
    NotExecuted,
    Executed(HostToolOutcome),
}

#[derive(Default)]
struct ToolExecutionData {
    tool_output: Option<ToolOutput>,
    stream_error: Option<ErrorValue>,
    tool_execution_ms: Option<f64>,
    execution_started_at: Option<Instant>,
}

/// `ToolExecutionState`.
struct ToolExecution {
    tool_call: TypedToolCall,
    data: Mutex<ToolExecutionData>,
    start_notification: OnceLock<Shared<BoxFuture<'static, ()>>>,
    end_notification: OnceLock<Shared<BoxFuture<'static, ()>>>,
}

impl ToolExecution {
    fn new(tool_call: TypedToolCall) -> Arc<Self> {
        Arc::new(Self {
            tool_call,
            data: Mutex::new(ToolExecutionData::default()),
            start_notification: OnceLock::new(),
            end_notification: OnceLock::new(),
        })
    }
}

/// A consumer-visible tool outcome published at the step boundary.
enum BufferedToolOutcome {
    Host {
        tool_call: TypedToolCall,
        outcome: HostToolOutcome,
        is_continuation: Option<bool>,
    },
    Part(TextStreamPart),
}

#[derive(Default)]
struct State {
    tool_calls_by_tool_call_id: HashMap<String, TypedToolCall>,
    raw_tool_calls_by_tool_call_id: HashMap<String, RawToolCall>,
    pending_approvals_by_approval_id: IndexMap<String, HarnessV1PendingToolApproval>,
    pending_approvals_by_tool_call_id: HashMap<String, HarnessV1PendingToolApproval>,
    pending_results_by_tool_call_id: HashMap<String, HarnessV1PendingToolResult>,
    settled_host_tool_call_ids: HashSet<String>,
    settled_builtin_approval_tool_call_ids: HashSet<String>,
    completed_steps: Vec<StepResult>,
    outstanding_host_tool_executions: Vec<JoinHandle<HarnessResult<()>>>,
    step_text: String,
    step_reasoning: String,
    step_tool_calls: Vec<ContentPart>,
    step_provider_tool_results: Vec<ContentPart>,
    step_approval_requests: Vec<ContentPart>,
    buffered_tool_outcomes: Vec<BufferedToolOutcome>,
    tool_executions: IndexMap<String, Arc<ToolExecution>>,
    ignored_finish_step: Option<FinishStepPart>,
    host_tool_call_ids_awaiting_echo: HashSet<String>,
    submitted_host_tool_results: IndexMap<String, HarnessV1SubmitToolResult>,
    closing_resumed_step: bool,
}

struct Ctx {
    input: RunPromptInput,
    result: HarnessStreamTextResult,
    lifecycle: Arc<TurnLifecycle>,
    tools_context: ToolsContext,
    active_tools: ToolSet,
    continuations_by_approval_id: HashMap<String, ToolApprovalResponse>,
    continuations_by_tool_call_id: HashMap<String, ToolResultPart>,
    control: OnceLock<Arc<dyn HarnessV1PromptControl>>,
    state: Mutex<State>,
}

enum ApprovalContinuationOutcome {
    Continued,
    AwaitingToolResult,
}

fn tool_call_finish_reason() -> LanguageModelV4FinishReason {
    LanguageModelV4FinishReason {
        unified: LanguageModelV4UnifiedFinishReason::ToolCalls,
        raw: None,
    }
}

fn zero_usage() -> LanguageModelV4Usage {
    LanguageModelV4Usage::default()
}

/// The harness metadata of a step as AI SDK provider metadata.
fn as_provider_metadata(metadata: &Option<HarnessV1Metadata>) -> Option<ProviderMetadata> {
    metadata.clone()
}

/// `hasTool({ tools, toolName })`.
fn has_tool<V>(tools: &IndexMap<String, V>, tool_name: &str) -> bool {
    tools.contains_key(tool_name)
}

/// `displayHostToolCall({ toolCall, sessionWorkDir })`.
fn display_host_tool_call(tool_call: &TypedToolCall, session_work_dir: &str) -> TypedToolCall {
    let mut display = tool_call.clone();
    display.input = strip_parsed_tool_input_work_dir(&tool_call.input, session_work_dir);
    if tool_call.is_invalid() {
        display.error = Some(ErrorValue::message(INVALID_TOOL_INPUT_MESSAGE));
    }
    display
}

/// The stripped `result` of a `tool-result` part built from `output`.
fn strip_tool_result_output(
    tool_call_id: &str,
    tool_name: &str,
    output: &JsonValue,
    session_work_dir: &str,
) -> JsonValue {
    let stripped = strip_work_dir(
        &HarnessV1StreamPart::ToolResult {
            tool_call_id: tool_call_id.to_string(),
            tool_name: tool_name.to_string(),
            result: output.clone(),
            is_error: None,
            preliminary: None,
            dynamic: None,
            provider_metadata: None,
        },
        session_work_dir,
    );
    match stripped {
        HarnessV1StreamPart::ToolResult { result, .. } => result,
        _ => output.clone(),
    }
}

fn call_part(raw: &RawToolCall) -> HarnessV1StreamPart {
    HarnessV1StreamPart::ToolCall {
        tool_call_id: raw.tool_call_id.clone(),
        tool_name: raw.tool_name.clone(),
        input: raw.input.clone(),
        provider_executed: raw.provider_executed,
        dynamic: raw.dynamic,
        provider_metadata: raw.provider_metadata.clone(),
        native_name: raw.native_name.clone(),
    }
}

/// Run one prompt turn end-to-end:
///  - call `session.doPromptTurn` via `toHarnessStream`;
///  - translate harness events to AI SDK `TextStreamPart`s and push them
///    into the result;
///  - execute host-side user tools when their `tool-call` events arrive and
///    submit results back to the harness;
///  - close the result when the harness signals `finish` (or on error).
///
/// Returns right after the driver starts; await the result's accessors or
/// `done` to observe completion.
pub fn run_prompt(input: RunPromptInput) -> RunPromptOutput {
    let call_id = generate_id();
    let tools_context = input.tools_context.clone().unwrap_or_default();
    let result = HarnessStreamTextResult::new(HarnessStreamTextResultOptions {
        tools: input.tools.clone(),
        runtime_context: input.runtime_context.clone(),
        tools_context: tools_context.clone(),
        harness_id: input.harness.harness_id().to_string(),
        call_id: call_id.clone(),
        model_id: input.model.clone().unwrap_or_default(),
        output: input.output.clone(),
    });
    let active_tools = input
        .active_tools
        .clone()
        .unwrap_or_else(|| input.tools.clone());
    let builtin_tools = input.harness.builtin_tools();
    let active_tool_names: Vec<String> = input
        .tools
        .keys()
        .filter(|name| {
            active_tools.contains_key(*name)
                || (builtin_tools.contains_key(*name)
                    && is_harness_v1_builtin_tool_included(
                        name,
                        input.builtin_tool_filtering.as_ref(),
                    ))
        })
        .cloned()
        .collect();

    let lifecycle = Arc::new(create_turn_lifecycle(TurnLifecycleOptions {
        call_id,
        telemetry: input.telemetry.clone(),
        callbacks: input.callbacks.clone().unwrap_or_default(),
        harness_id: input.harness.harness_id().to_string(),
        model_id: input.model.clone(),
        instructions: input.instructions.clone(),
        tools: input.tools.clone(),
        tools_context: tools_context.clone(),
        active_tool_names,
        tool_specs: input.tool_specs.clone(),
        messages: match &input.prompt {
            None => Vec::new(),
            Some(prompt) => vec![ModelMessage::User(UserModelMessage {
                content: UserContent::Text(prompt_to_text(prompt)),
                provider_options: None,
            })],
        },
        runtime_context: input.runtime_context.clone(),
        output: input.output.clone(),
    }));

    let mut state = State::default();
    for approval in &input.pending_tool_approvals {
        state
            .pending_approvals_by_approval_id
            .insert(approval.approval_id.clone(), approval.clone());
        state
            .pending_approvals_by_tool_call_id
            .insert(approval.tool_call_id.clone(), approval.clone());
    }
    for pending in &input.pending_tool_results {
        state
            .pending_results_by_tool_call_id
            .insert(pending.tool_call_id.clone(), pending.clone());
    }
    let ctx = Arc::new(Ctx {
        continuations_by_approval_id: input
            .tool_approval_continuations
            .iter()
            .map(|c| (c.approval_id.clone(), c.clone()))
            .collect(),
        continuations_by_tool_call_id: input
            .tool_result_continuations
            .iter()
            .map(|c| (c.tool_call_id.clone(), c.clone()))
            .collect(),
        input,
        result: result.clone(),
        lifecycle,
        tools_context,
        active_tools,
        control: OnceLock::new(),
        state: Mutex::new(state),
    });

    let handle = tokio::spawn(drive(ctx));
    let done: BoxFuture<'static, ()> = Box::pin(async move {
        let _ = handle.await;
    });
    RunPromptOutput {
        result,
        done: done.shared(),
    }
}

fn is_aborted(ctx: &Ctx) -> bool {
    ctx.input
        .abort_signal
        .as_ref()
        .is_some_and(|s| s.is_cancelled())
}

fn is_turn_suspending(ctx: &Ctx) -> bool {
    ctx.input.is_turn_suspending.as_ref().is_some_and(|f| f())
}

/// Settle a failed turn. When the caller's own abort signal fired, the
/// failure is a user-initiated stop: surface it as an `abort` part. Both
/// outcomes notify `onTurnFailed` unless the turn is being suspended.
fn settle_failure(ctx: &Ctx, err: ErrorValue) {
    if !is_turn_suspending(ctx)
        && let Some(f) = &ctx.input.on_turn_failed
    {
        f();
    }
    if is_aborted(ctx) {
        ctx.result.abort(
            err,
            Some(crate::harness::sdk::error_value::get_error_message(
                &AbortError,
            )),
        );
        return;
    }
    ctx.result.fail(err);
}

async fn settle_lifecycle(ctx: &Ctx, err: ErrorValue) {
    if is_aborted(ctx) {
        ctx.lifecycle
            .abort(Some(ErrorValue::from_box(Box::new(AbortError))))
            .await;
    } else {
        ctx.lifecycle.error(err).await;
    }
}

fn log_error(ctx: &Ctx, context: &str, error: &ErrorValue) {
    let session_id = ctx.input.session.session_id();
    let input = match error {
        ErrorValue::Error(e) => BridgeErrorInput::Error(&**e),
        ErrorValue::Value(v) => BridgeErrorInput::Value(v),
    };
    log_bridge_error(LogBridgeErrorOptions {
        harness_id: ctx.input.harness.harness_id(),
        session_id: Some(&session_id),
        context: Some(context),
        error: input,
        write: None,
    });
}

async fn drive(ctx: Arc<Ctx>) {
    let invoke_ctx = ctx.clone();
    let bridge = to_harness_stream(move |emit| async move {
        let input = &invoke_ctx.input;
        let settings = HarnessV1TurnSettings {
            model: input.model.clone(),
            skills: input.skills.clone(),
            instructions: input.instructions.clone(),
            tools: input.tool_specs.clone(),
        };
        match input.mode {
            RunPromptMode::Continue => {
                input
                    .session
                    .do_continue_turn(HarnessV1ContinueTurnOptions {
                        settings,
                        response_format: input.response_format.clone(),
                        abort_signal: input.abort_signal.clone(),
                        emit,
                    })
                    .await
            }
            RunPromptMode::Prompt => {
                let Some(prompt) = input.prompt.clone() else {
                    return Err(plain_error(
                        "runPrompt: `prompt` is required for mode \"prompt\".",
                    ));
                };
                input
                    .session
                    .do_prompt_turn(HarnessV1PromptTurnOptions {
                        settings,
                        prompt,
                        response_format: input.response_format.clone(),
                        abort_signal: input.abort_signal.clone(),
                        emit,
                    })
                    .await
            }
        }
    })
    .await;
    let bridge = match bridge {
        Ok(bridge) => bridge,
        Err(err) => {
            let err = ErrorValue::from_box(err);
            settle_lifecycle(&ctx, err.clone()).await;
            log_error(&ctx, "failed to start harness turn", &err);
            settle_failure(&ctx, err);
            return;
        }
    };

    let control = bridge.control.clone();
    let _ = ctx.control.set(control.clone());
    if let Some(f) = &ctx.input.on_prompt_control_available {
        f(control);
    }
    let mut stream = bridge.stream;
    let mut pending_stop_boundary: Option<Option<Box<dyn FnOnce() + Send + Sync>>> = None;
    let outcome = run_loop(&ctx, &mut stream, &mut pending_stop_boundary).await;
    if let Err(err) = outcome {
        let _ = ctx.wait_for_outstanding_host_tool_executions().await;
        let err = ErrorValue::from_box(err);
        settle_lifecycle(&ctx, err.clone()).await;
        log_error(&ctx, "harness turn failed", &err);
        settle_failure(&ctx, err);
    }
    if let Some(Some(release)) = pending_stop_boundary.take() {
        release();
    }
}

impl Ctx {
    fn control(&self) -> &Arc<dyn HarnessV1PromptControl> {
        self.control.get().expect("prompt control is set")
    }

    fn harness_id(&self) -> &str {
        self.input.harness.harness_id()
    }

    fn start_host_tool_execution(&self, execution: JoinHandle<HarnessResult<()>>) {
        self.state
            .lock()
            .outstanding_host_tool_executions
            .push(execution);
    }

    async fn wait_for_outstanding_host_tool_executions(&self) -> HarnessResult<()> {
        let executions = std::mem::take(&mut self.state.lock().outstanding_host_tool_executions);
        if executions.is_empty() {
            return Ok(());
        }
        let results = futures::future::join_all(executions).await;
        for result in results {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(err),
                Err(join) => return Err(Box::new(join)),
            }
        }
        Ok(())
    }

    async fn publish_tool_execution_start(&self, execution: &Arc<ToolExecution>) {
        let notification = execution
            .start_notification
            .get_or_init(|| {
                let lifecycle = self.lifecycle.clone();
                let tool_call = execution.tool_call.clone();
                async move { lifecycle.tool_execution_start(tool_call).await }
                    .boxed()
                    .shared()
            })
            .clone();
        notification.await;
    }

    async fn publish_tool_execution_end(&self, execution: &Arc<ToolExecution>) {
        let (tool_output, ms) = {
            let data = execution.data.lock();
            (data.tool_output.clone(), data.tool_execution_ms)
        };
        let Some(tool_output) = tool_output else {
            return;
        };
        self.publish_tool_execution_start(execution).await;
        let notification = execution
            .end_notification
            .get_or_init(|| {
                let lifecycle = self.lifecycle.clone();
                let tool_call = execution.tool_call.clone();
                async move {
                    lifecycle
                        .tool_execution_end(tool_call, tool_output, ms.unwrap_or(0.0))
                        .await
                }
                .boxed()
                .shared()
            })
            .clone();
        notification.await;
    }

    async fn publish_tool_executions(&self) {
        let executions: Vec<Arc<ToolExecution>> = self
            .state
            .lock()
            .tool_executions
            .values()
            .cloned()
            .collect();
        for execution in &executions {
            self.publish_tool_execution_end(execution).await;
        }
        let buffered = {
            let mut state = self.state.lock();
            state.tool_executions.clear();
            std::mem::take(&mut state.buffered_tool_outcomes)
        };
        for publish in buffered {
            match publish {
                BufferedToolOutcome::Host {
                    tool_call,
                    outcome,
                    is_continuation,
                } => self.enqueue_host_tool_outcome(&tool_call, &outcome, is_continuation),
                BufferedToolOutcome::Part(part) => self.result.enqueue(part),
            }
        }
    }

    fn build_model_call_content(&self) -> Vec<ContentPart> {
        let state = self.state.lock();
        let mut parts = Vec::new();
        if !state.step_text.is_empty() {
            parts.push(ContentPart::Text(TextContent {
                text: state.step_text.clone(),
                ..Default::default()
            }));
        }
        if !state.step_reasoning.is_empty() {
            parts.push(ContentPart::Reasoning(ReasoningOutput {
                text: state.step_reasoning.clone(),
                provider_metadata: None,
            }));
        }
        parts.extend(state.step_tool_calls.iter().cloned());
        parts.extend(state.step_approval_requests.iter().cloned());
        parts.extend(state.step_provider_tool_results.iter().cloned());
        parts
    }

    fn reset_step_content(&self) {
        let mut state = self.state.lock();
        state.step_text.clear();
        state.step_reasoning.clear();
        state.step_tool_calls.clear();
        state.step_provider_tool_results.clear();
        state.step_approval_requests.clear();
        state.buffered_tool_outcomes.clear();
        state.tool_executions.clear();
        state.ignored_finish_step = None;
    }

    async fn complete_step(
        &self,
        finish_reason: &LanguageModelV4FinishReason,
        usage: &LanguageModelV4Usage,
        provider_metadata: Option<ProviderMetadata>,
    ) -> StepResult {
        self.lifecycle
            .language_model_call_end(LanguageModelCallEndInput {
                finish_reason: finish_reason.unified,
                usage: as_language_model_usage(usage),
                provider_metadata: provider_metadata.clone(),
                content: self.build_model_call_content(),
            })
            .await;
        self.publish_tool_executions().await;
        let step = self
            .result
            .finish_step(finish_reason, usage, provider_metadata);
        self.state.lock().completed_steps.push(step.clone());
        self.lifecycle.step_end(step.clone()).await;
        self.reset_step_content();
        step
    }

    async fn finish_for_host_input_pause(&self) -> HarnessResult<()> {
        self.wait_for_outstanding_host_tool_executions().await?;
        self.publish_tool_executions().await;
        let steps = self.state.lock().completed_steps.clone();
        self.lifecycle
            .end(&steps, as_language_model_usage(&zero_usage()))
            .await;
        self.result.finish(None).await;
        Ok(())
    }

    async fn pause_for_host_input(&self, finish_step: Option<FinishStepPart>) -> HarnessResult<()> {
        self.wait_for_outstanding_host_tool_executions().await?;
        let submissions: Vec<HarnessV1SubmitToolResult> = self
            .state
            .lock()
            .submitted_host_tool_results
            .values()
            .cloned()
            .collect();
        for submission in submissions {
            let tool_call_id = submission.tool_call_id.clone();
            let completed_result = HarnessV1CompletedToolResult {
                output: submission.output.clone(),
                is_error: submission.is_error,
                tool_result: submission.tool_result.clone(),
            };
            let (pending_result, tool_call, execution, settled) = {
                let state = self.state.lock();
                let raw = state.raw_tool_calls_by_tool_call_id.get(&tool_call_id);
                let pending = state
                    .pending_results_by_tool_call_id
                    .get(&tool_call_id)
                    .cloned()
                    .or_else(|| {
                        raw.map(|raw| HarnessV1PendingToolResult {
                            tool_call_id: tool_call_id.clone(),
                            tool_name: raw.tool_name.clone(),
                            input: raw.input.clone(),
                            provider_options: raw.provider_metadata.clone(),
                            completed_result: None,
                        })
                    });
                (
                    pending,
                    state.tool_calls_by_tool_call_id.get(&tool_call_id).cloned(),
                    state.tool_executions.get(&tool_call_id).cloned(),
                    state.settled_host_tool_call_ids.contains(&tool_call_id),
                )
            };
            let Some(mut pending_result) = pending_result else {
                continue;
            };
            pending_result.completed_result = Some(completed_result);
            self.on_pending_tool_result(pending_result);
            let (tool_output, stream_error) = match &execution {
                Some(execution) => {
                    let data = execution.data.lock();
                    (data.tool_output.clone(), data.stream_error.clone())
                }
                None => (None, None),
            };
            if let (Some(tool_call), Some(tool_output), false) = (tool_call, tool_output, settled) {
                let outcome = match tool_output {
                    ToolOutput::ToolError(e) => HostToolOutcome::Err {
                        error: e.error,
                        stream_error,
                    },
                    ToolOutput::ToolResult(r) => HostToolOutcome::Ok(r.output),
                };
                self.enqueue_host_tool_outcome(&tool_call, &outcome, Some(false));
            }
        }
        self.state.lock().closing_resumed_step = false;
        let usage = finish_step
            .as_ref()
            .map(|f| f.usage.clone())
            .unwrap_or_else(zero_usage);
        self.complete_step(
            &tool_call_finish_reason(),
            &usage,
            finish_step
                .as_ref()
                .and_then(|f| as_provider_metadata(&f.harness_metadata)),
        )
        .await;
        self.finish_for_host_input_pause().await
    }

    fn enqueue_approval_request(
        &self,
        approval_id: &str,
        tool_call: &TypedToolCall,
        is_automatic: Option<bool>,
    ) {
        let request = ToolApprovalRequestOutput {
            approval_id: approval_id.to_string(),
            tool_call: tool_call.clone(),
            reason: None,
            is_automatic,
            signature: None,
        };
        self.result
            .enqueue(TextStreamPart::ToolApprovalRequest(request.clone()));
        self.state
            .lock()
            .step_approval_requests
            .push(ContentPart::ToolApprovalRequest(request));
    }

    fn enqueue_automatic_approval_response(
        &self,
        approval_id: &str,
        tool_call: &TypedToolCall,
        approved: bool,
        reason: Option<String>,
        provider_executed: Option<bool>,
    ) {
        self.result.enqueue(TextStreamPart::ToolApprovalResponse(
            ToolApprovalResponseOutput {
                approval_id: approval_id.to_string(),
                tool_call: tool_call.clone(),
                approved,
                reason,
                provider_executed,
            },
        ));
    }

    fn enqueue_approval_response(
        &self,
        approval: &HarnessV1PendingToolApproval,
        continuation: &ToolApprovalResponse,
        tool_call: TypedToolCall,
    ) {
        self.result
            .enqueue_continuation(TextStreamPart::ToolApprovalResponse(
                ToolApprovalResponseOutput {
                    approval_id: approval.approval_id.clone(),
                    tool_call,
                    approved: continuation.approved,
                    reason: continuation.reason.clone(),
                    provider_executed: approval.provider_executed,
                },
            ));
    }

    fn on_pending_tool_result(&self, pending: HarnessV1PendingToolResult) {
        if let Some(f) = &self.input.on_pending_tool_result {
            f(pending);
        }
    }

    fn record_pending_tool_result(&self, tool_call: &RawToolCall) -> HarnessV1PendingToolResult {
        let pending = {
            let mut state = self.state.lock();
            let pending = state
                .pending_results_by_tool_call_id
                .get(&tool_call.tool_call_id)
                .cloned()
                .unwrap_or_else(|| HarnessV1PendingToolResult {
                    tool_call_id: tool_call.tool_call_id.clone(),
                    tool_name: tool_call.tool_name.clone(),
                    input: tool_call.input.clone(),
                    provider_options: tool_call.provider_metadata.clone(),
                    completed_result: None,
                });
            state
                .pending_results_by_tool_call_id
                .insert(pending.tool_call_id.clone(), pending.clone());
            pending
        };
        self.on_pending_tool_result(pending.clone());
        pending
    }

    fn enqueue_host_tool_outcome(
        &self,
        tool_call: &TypedToolCall,
        outcome: &HostToolOutcome,
        is_continuation: Option<bool>,
    ) {
        let enqueue = |part: TextStreamPart| {
            if is_continuation == Some(false) {
                self.result.enqueue(part);
            } else {
                self.result.enqueue_continuation(part);
            }
        };
        match outcome {
            HostToolOutcome::Ok(output) => {
                let stripped = strip_tool_result_output(
                    &tool_call.tool_call_id,
                    &tool_call.tool_name,
                    output,
                    &self.input.session_work_dir,
                );
                enqueue(TextStreamPart::ToolResult(TypedToolResult::new(
                    &tool_call.tool_call_id,
                    &tool_call.tool_name,
                    Some(tool_call.input.clone()),
                    stripped,
                )));
            }
            HostToolOutcome::Err {
                error,
                stream_error,
            } => {
                // Stream parts can reach the client and the model.
                enqueue(TextStreamPart::ToolError(TypedToolError::new(
                    &tool_call.tool_call_id,
                    &tool_call.tool_name,
                    Some(tool_call.input.clone()),
                    stream_error.clone().unwrap_or_else(|| error.clone()),
                )));
            }
        }
    }

    async fn submit_tool_result(
        &self,
        mut submission: HarnessV1SubmitToolResult,
    ) -> HarnessResult<()> {
        if let Some(tool_result) = &submission.tool_result {
            let output = match normalize_harness_tool_model_output(&tool_result.output) {
                Ok(output) => output,
                Err(error) => {
                    if !self
                        .continuations_by_tool_call_id
                        .contains_key(&submission.tool_call_id)
                        || !HarnessError::is_capability_unsupported(&error)
                    {
                        return Err(Box::new(error));
                    }
                    ToolResultOutput::Text {
                        value: submission.output.to_string(),
                        provider_options: None,
                    }
                }
            };
            let mut tool_result = tool_result.clone();
            tool_result.output = output.clone();
            submission.tool_result = Some(tool_result);
            self.result
                .set_tool_model_output(&submission.tool_call_id, output);
        }
        if !is_turn_suspending(self) {
            {
                let mut state = self.state.lock();
                if !state
                    .settled_host_tool_call_ids
                    .contains(&submission.tool_call_id)
                {
                    state
                        .host_tool_call_ids_awaiting_echo
                        .insert(submission.tool_call_id.clone());
                }
                state
                    .submitted_host_tool_results
                    .insert(submission.tool_call_id.clone(), submission.clone());
            }
            return self.control().submit_tool_result(submission).await;
        }
        let tool_call_id = submission.tool_call_id.clone();
        let tool_call = {
            let state = self.state.lock();
            state
                .raw_tool_calls_by_tool_call_id
                .get(&tool_call_id)
                .map(|r| (r.tool_name.clone(), r.input.clone()))
                .or_else(|| {
                    state
                        .pending_results_by_tool_call_id
                        .get(&tool_call_id)
                        .map(|r| (r.tool_name.clone(), r.input.clone()))
                })
                .or_else(|| {
                    self.input
                        .pending_tool_approvals
                        .iter()
                        .find(|a| a.tool_call_id == tool_call_id)
                        .map(|a| (a.tool_name.clone(), a.input.clone()))
                })
        };
        let Some((tool_name, tool_input)) = tool_call else {
            return Err(plain_error(format!(
                "Unknown suspended tool call '{tool_call_id}'."
            )));
        };
        self.on_pending_tool_result(HarnessV1PendingToolResult {
            tool_call_id: tool_call_id.clone(),
            tool_name,
            input: tool_input,
            provider_options: None,
            completed_result: Some(HarnessV1CompletedToolResult {
                output: submission.output.clone(),
                is_error: submission.is_error,
                tool_result: submission.tool_result.clone(),
            }),
        });
        let mut state = self.state.lock();
        if let Some(parsed) = state.tool_calls_by_tool_call_id.get(&tool_call_id).cloned()
            && !state.settled_host_tool_call_ids.contains(&tool_call_id)
        {
            let outcome = if submission.is_error == Some(true) {
                HostToolOutcome::Err {
                    error: ErrorValue::Value(submission.output.clone()),
                    stream_error: None,
                }
            } else {
                HostToolOutcome::Ok(submission.output.clone())
            };
            state
                .buffered_tool_outcomes
                .push(BufferedToolOutcome::Host {
                    tool_call: parsed,
                    outcome,
                    is_continuation: None,
                });
        }
        Ok(())
    }

    async fn process_pending_tool_result_continuation(
        &self,
        pending_result: &HarnessV1PendingToolResult,
        continuation: Option<&ToolResultPart>,
    ) -> HarnessResult<()> {
        let submission = match continuation {
            None => pending_result.completed_result.clone(),
            Some(continuation) => {
                let mut completed = unwrap_tool_result_output(continuation);
                let mut tool_result = continuation.clone();
                tool_result.tool_name = pending_result.tool_name.clone();
                if continuation.provider_options.is_none()
                    && pending_result.provider_options.is_some()
                {
                    tool_result.provider_options = pending_result.provider_options.clone();
                }
                completed.tool_result = Some(tool_result);
                Some(completed)
            }
        };
        let Some(submission) = submission else {
            return Ok(());
        };
        self.state
            .lock()
            .settled_host_tool_call_ids
            .insert(pending_result.tool_call_id.clone());
        if let Some(f) = &self.input.on_tool_result_settled {
            f(pending_result.tool_call_id.clone());
        }
        self.submit_tool_result(HarnessV1SubmitToolResult {
            tool_call_id: pending_result.tool_call_id.clone(),
            output: submission.output,
            is_error: submission.is_error,
            tool_result: submission.tool_result,
        })
        .await
    }

    async fn process_pending_approval_continuation(
        self: &Arc<Self>,
        approval: &HarnessV1PendingToolApproval,
        continuation: &ToolApprovalResponse,
    ) -> HarnessResult<ApprovalContinuationOutcome> {
        let raw_tool_call = (approval.kind == HarnessV1PendingToolApprovalKind::Builtin)
            .then(|| {
                self.state
                    .lock()
                    .raw_tool_calls_by_tool_call_id
                    .get(&approval.tool_call_id)
                    .cloned()
            })
            .flatten()
            .unwrap_or_else(|| RawToolCall {
                tool_call_id: approval.tool_call_id.clone(),
                tool_name: approval.tool_name.clone(),
                input: approval.input.clone(),
                provider_executed: approval.provider_executed,
                dynamic: None,
                provider_metadata: None,
                native_name: approval.native_name.clone(),
            });
        let mut validated_tool_call: Option<TypedToolCall> = None;
        let tool_call: TypedToolCall =
            if approval.kind == HarnessV1PendingToolApprovalKind::Custom && continuation.approved {
                let validated = validate_tool_call(
                    &call_part(&raw_tool_call),
                    Some(&self.input.tools),
                    self.harness_id(),
                );
                let display = display_host_tool_call(&validated, &self.input.session_work_dir);
                validated_tool_call = Some(validated);
                display
            } else {
                let parsed_input = safe_parse_json(&raw_tool_call.input, None)
                    .unwrap_or_else(|_| JsonValue::String(raw_tool_call.input.clone()));
                let mut call = TypedToolCall::new(
                    &raw_tool_call.tool_call_id,
                    &raw_tool_call.tool_name,
                    parsed_input,
                );
                call.provider_executed = raw_tool_call.provider_executed;
                call
            };

        self.enqueue_approval_response(approval, continuation, tool_call.clone());
        if let Some(f) = &self.input.on_tool_approval_settled {
            f(approval.approval_id.clone());
        }
        {
            let mut state = self.state.lock();
            state
                .pending_approvals_by_approval_id
                .shift_remove(&approval.approval_id);
            state
                .pending_approvals_by_tool_call_id
                .remove(&approval.tool_call_id);
        }

        if approval.kind == HarnessV1PendingToolApprovalKind::Builtin {
            self.state
                .lock()
                .settled_builtin_approval_tool_call_ids
                .insert(approval.tool_call_id.clone());
            if !self.control().supports_submit_tool_approval() {
                return Err(plain_error(format!(
                    "Harness '{}' emitted a built-in tool approval request but does not support approval responses.",
                    self.harness_id()
                )));
            }
            self.control()
                .submit_tool_approval(HarnessV1SubmitToolApproval {
                    approval_id: approval.approval_id.clone(),
                    approved: continuation.approved,
                    reason: continuation.reason.clone(),
                })
                .await?;
            return Ok(ApprovalContinuationOutcome::Continued);
        }

        {
            let mut state = self.state.lock();
            state
                .settled_host_tool_call_ids
                .insert(approval.tool_call_id.clone());
            state
                .raw_tool_calls_by_tool_call_id
                .insert(approval.tool_call_id.clone(), raw_tool_call.clone());
        }
        if !continuation.approved {
            let output = execution_denied(continuation.reason.clone());
            self.submit_tool_result(HarnessV1SubmitToolResult {
                tool_call_id: approval.tool_call_id.clone(),
                output: execution_denied_json(&output),
                is_error: None,
                tool_result: Some(ToolResultPart {
                    tool_call_id: approval.tool_call_id.clone(),
                    tool_name: approval.tool_name.clone(),
                    output,
                    provider_options: None,
                }),
            })
            .await?;
            return Ok(ApprovalContinuationOutcome::Continued);
        }

        let Some(validated_tool_call) = validated_tool_call else {
            return Err(plain_error(format!(
                "Harness '{}' could not validate approved host tool '{}'.",
                self.harness_id(),
                approval.tool_name
            )));
        };

        if validated_tool_call.is_invalid() {
            self.submit_tool_result(HarnessV1SubmitToolResult {
                tool_call_id: approval.tool_call_id.clone(),
                output: json!({ "error": INVALID_TOOL_INPUT_MESSAGE }),
                is_error: Some(true),
                tool_result: None,
            })
            .await?;
            self.state
                .lock()
                .buffered_tool_outcomes
                .push(BufferedToolOutcome::Host {
                    tool_call: tool_call.clone(),
                    outcome: HostToolOutcome::Err {
                        error: ErrorValue::message(INVALID_TOOL_INPUT_MESSAGE),
                        stream_error: None,
                    },
                    is_continuation: None,
                });
            self.publish_tool_executions().await;
            return Ok(ApprovalContinuationOutcome::Continued);
        }

        self.lifecycle.start(self.input.model.clone()).await;
        let execution = ToolExecution::new(tool_call.clone());
        self.state
            .lock()
            .tool_executions
            .insert(raw_tool_call.tool_call_id.clone(), execution.clone());
        self.publish_tool_execution_start(&execution).await;
        let execution_started_at = Instant::now();
        let outcome = self
            .maybe_execute_host_tool(&raw_tool_call, &validated_tool_call)
            .await?;
        let HostToolExecution::Executed(outcome) = outcome else {
            self.record_pending_tool_result(&raw_tool_call);
            self.finish_for_host_input_pause().await?;
            return Ok(ApprovalContinuationOutcome::AwaitingToolResult);
        };
        let execution = self
            .state
            .lock()
            .tool_executions
            .get(&raw_tool_call.tool_call_id)
            .cloned()
            .unwrap_or(execution);
        {
            let mut data = execution.data.lock();
            data.tool_output = Some(to_tool_output(&tool_call, &outcome));
            data.stream_error = get_stream_error(&outcome);
            data.tool_execution_ms = Some(execution_started_at.elapsed().as_secs_f64() * 1000.0);
        }
        self.publish_tool_execution_end(&execution).await;
        self.enqueue_host_tool_outcome(&tool_call, &outcome, None);
        Ok(ApprovalContinuationOutcome::Continued)
    }

    /// `maybeExecuteHostTool(...)`.
    async fn maybe_execute_host_tool(
        self: &Arc<Self>,
        event: &RawToolCall,
        parsed_tool_call: &TypedToolCall,
    ) -> HarnessResult<HostToolExecution> {
        let Some(tool) = self.active_tools.get(&event.tool_name).cloned() else {
            return Ok(HostToolExecution::NotExecuted);
        };
        if !is_executable_tool(Some(&tool)) {
            return Ok(HostToolExecution::NotExecuted);
        }

        if parsed_tool_call.is_invalid() {
            self.submit_tool_result(HarnessV1SubmitToolResult {
                tool_call_id: event.tool_call_id.clone(),
                output: json!({ "error": INVALID_TOOL_INPUT_MESSAGE }),
                is_error: Some(true),
                tool_result: None,
            })
            .await?;
            return Ok(HostToolExecution::Executed(HostToolOutcome::Err {
                error: ErrorValue::message(INVALID_TOOL_INPUT_MESSAGE),
                stream_error: None,
            }));
        }

        let context = match validate_tool_context(
            &event.tool_name,
            self.tools_context.get(&event.tool_name),
            tool.context_schema.as_ref(),
        ) {
            Ok(context) => context,
            Err(err) => {
                // Context is host-only and can contain credentials: keep the
                // detailed error host-side only.
                self.submit_tool_result(HarnessV1SubmitToolResult {
                    tool_call_id: event.tool_call_id.clone(),
                    output: json!({ "error": INVALID_TOOL_CONTEXT_MESSAGE }),
                    is_error: Some(true),
                    tool_result: None,
                })
                .await?;
                return Ok(HostToolExecution::Executed(HostToolOutcome::Err {
                    error: ErrorValue::from_box(Box::new(err)),
                    stream_error: Some(ErrorValue::message(INVALID_TOOL_CONTEXT_MESSAGE)),
                }));
            }
        };

        let ctx = self.clone();
        let tool_for_exec = tool.clone();
        let input = parsed_tool_call.input.clone();
        let tool_call_id = event.tool_call_id.clone();
        let tool_name = event.tool_name.clone();
        let execute: BoxFuture<'static, HarnessResult<JsonValue>> = Box::pin(async move {
            let mut output = JsonValue::Null;
            let mut stream = execute_tool(
                &tool_for_exec,
                input,
                ToolExecutionOptions {
                    tool_call_id: tool_call_id.clone(),
                    messages: Vec::new(),
                    abort_signal: ctx.input.abort_signal.clone(),
                    context: context.unwrap_or(JsonValue::Null),
                    experimental_sandbox: Some(ctx.input.sandbox_session.clone()),
                },
            );
            while let Some(part) = stream.next().await {
                match part? {
                    ExecuteToolPart::Preliminary(preliminary) => {
                        // Preliminary values never reach the runtime, so
                        // strip the working directory here.
                        let stripped = strip_tool_result_output(
                            &tool_call_id,
                            &tool_name,
                            &preliminary,
                            &ctx.input.session_work_dir,
                        );
                        let mut part =
                            TypedToolResult::new(&tool_call_id, &tool_name, None, stripped);
                        part.preliminary = Some(true);
                        ctx.result.enqueue(TextStreamPart::ToolResult(part));
                    }
                    ExecuteToolPart::Final(value) => output = value,
                }
            }
            Ok(output)
        });
        // The execution, the model-output conversion and the submission
        // share one `try`.
        let attempt: HarnessResult<JsonValue> = async {
            let output = self
                .lifecycle
                .execute_tool(&event.tool_call_id, execute)
                .await?;
            let tool_result = match &tool.to_model_output {
                None => None,
                Some(_) => Some(ToolResultPart {
                    tool_call_id: event.tool_call_id.clone(),
                    tool_name: event.tool_name.clone(),
                    output: create_tool_model_output(
                        &event.tool_call_id,
                        &parsed_tool_call.input,
                        &output,
                        Some(&tool),
                        ToolModelOutputErrorMode::None,
                    )
                    .await?,
                    provider_options: None,
                }),
            };
            self.submit_tool_result(HarnessV1SubmitToolResult {
                tool_call_id: event.tool_call_id.clone(),
                output: output.clone(),
                is_error: None,
                tool_result,
            })
            .await?;
            Ok(output)
        }
        .await;
        match attempt {
            Ok(output) => Ok(HostToolExecution::Executed(HostToolOutcome::Ok(output))),
            Err(err) => {
                let err = ErrorValue::from_box(err);
                let error_text = err.js_string();
                self.submit_tool_result(HarnessV1SubmitToolResult {
                    tool_call_id: event.tool_call_id.clone(),
                    output: json!({ "error": error_text }),
                    is_error: Some(true),
                    tool_result: None,
                })
                .await?;
                // The error text of an `Error` or a string is what the runtime
                // received; any other thrown value would serialize with all
                // its properties.
                let stream_error = match &err {
                    ErrorValue::Error(_) | ErrorValue::Value(JsonValue::String(_)) => None,
                    ErrorValue::Value(_) => Some(ErrorValue::Value(JsonValue::String(error_text))),
                };
                Ok(HostToolExecution::Executed(HostToolOutcome::Err {
                    error: err,
                    stream_error,
                }))
            }
        }
    }
}

fn execution_denied(reason: Option<String>) -> ToolResultOutput {
    ToolResultOutput::ExecutionDenied {
        reason,
        provider_options: None,
    }
}

/// `{ type: 'execution-denied', reason }` as the raw submitted output.
fn execution_denied_json(output: &ToolResultOutput) -> JsonValue {
    serde_json::to_value(output).unwrap_or(JsonValue::Null)
}

type BridgeStream = futures::stream::BoxStream<'static, HarnessV1StreamPart>;

async fn run_loop(
    ctx: &Arc<Ctx>,
    stream: &mut BridgeStream,
    pending_stop_boundary: &mut Option<Option<Box<dyn FnOnce() + Send + Sync>>>,
) -> HarnessResult<()> {
    let input = &ctx.input;
    let builtin_tools = input.harness.builtin_tools();
    let mut strip_tool_input_work_dir =
        create_tool_input_work_dir_stripper(&input.session_work_dir);
    let mut final_finish: Option<(
        LanguageModelV4FinishReason,
        LanguageModelV4Usage,
        Option<HarnessV1Metadata>,
    )> = None;
    let mut host_input_pause_requested = false;
    let mut deferred_pause_finish_step: Option<FinishStepPart> = None;
    let release_pending_stop_boundary =
        |boundary: &mut Option<Option<Box<dyn FnOnce() + Send + Sync>>>| {
            if let Some(Some(release)) = boundary.take() {
                release();
            }
        };

    let translate = |part: &HarnessV1StreamPart| -> Vec<TextStreamPart> {
        let get_tool_call = |id: &str| {
            ctx.state
                .lock()
                .tool_calls_by_tool_call_id
                .get(id)
                .map(|c| TranslatedToolCallInfo {
                    input: c.input.clone(),
                    provider_executed: c.provider_executed,
                })
        };
        let is_provider_executed = |id: &str| {
            ctx.state
                .lock()
                .raw_tool_calls_by_tool_call_id
                .get(id)
                .is_none_or(|raw| raw.provider_executed == Some(true))
        };
        translate_stream_part(
            part,
            &TranslateOptions {
                get_tool_call: Some(&get_tool_call),
                is_provider_executed: Some(&is_provider_executed),
            },
        )
    };

    for approval in &input.pending_tool_approvals {
        if let Some(continuation) = ctx.continuations_by_approval_id.get(&approval.approval_id) {
            let outcome = ctx
                .process_pending_approval_continuation(approval, continuation)
                .await?;
            if matches!(outcome, ApprovalContinuationOutcome::AwaitingToolResult) {
                return Ok(());
            }
            ctx.state.lock().closing_resumed_step = true;
        }
    }
    for pending_result in &input.pending_tool_results {
        let continuation = ctx
            .continuations_by_tool_call_id
            .get(&pending_result.tool_call_id);
        if continuation.is_some() || pending_result.completed_result.is_some() {
            ctx.process_pending_tool_result_continuation(pending_result, continuation)
                .await?;
            if pending_result.completed_result.is_none() {
                ctx.state.lock().closing_resumed_step = true;
            }
        }
    }

    loop {
        if deferred_pause_finish_step.is_some()
            && ctx.state.lock().host_tool_call_ids_awaiting_echo.is_empty()
        {
            ctx.pause_for_host_input(deferred_pause_finish_step.take())
                .await?;
            return Ok(());
        }

        let Some(value) = stream.next().await else {
            release_pending_stop_boundary(pending_stop_boundary);
            break;
        };

        // `raw` is out-of-band adapter data: pass it straight through.
        if let HarnessV1StreamPart::Raw { raw_value } = &value {
            ctx.result.enqueue_raw(raw_value.clone());
            continue;
        }
        if let HarnessV1StreamPart::ToolResult { tool_call_id, .. } = &value {
            let mut state = ctx.state.lock();
            state.host_tool_call_ids_awaiting_echo.remove(tool_call_id);
            state.submitted_host_tool_results.shift_remove(tool_call_id);
            state.pending_results_by_tool_call_id.remove(tool_call_id);
        }

        if pending_stop_boundary.is_some() {
            if matches!(value, HarnessV1StreamPart::Finish { .. }) {
                release_pending_stop_boundary(pending_stop_boundary);
            } else {
                let steps = ctx.state.lock().completed_steps.clone();
                let conditions = input.stop_conditions.clone().unwrap_or_default();
                if is_stop_condition_met(&conditions, &steps).await {
                    if let Some(f) = &input.on_stop_condition_met {
                        f().await?;
                    }
                    release_pending_stop_boundary(pending_stop_boundary);
                    let usage = steps
                        .iter()
                        .fold(create_null_language_model_usage(), |total, step| {
                            add_language_model_usage(&total, &step.usage)
                        });
                    ctx.lifecycle.end(&steps, usage).await;
                    ctx.result.finish(None).await;
                    return Ok(());
                } else {
                    release_pending_stop_boundary(pending_stop_boundary);
                }
            }
        }

        // Begin the operation span on stream-start.
        if let HarnessV1StreamPart::StreamStart { warnings, model_id } = &value {
            let model_id = model_id.clone().or_else(|| input.model.clone());
            if let Some(model_id) = &model_id {
                ctx.result.set_model_id(model_id.clone());
            }
            ctx.result
                .add_warnings(warnings.as_deref().unwrap_or_default());
            ctx.lifecycle.start(model_id).await;
        }

        // Open a step span lazily before the first content of each step.
        if !matches!(
            value,
            HarnessV1StreamPart::StreamStart { .. }
                | HarnessV1StreamPart::FinishStep { .. }
                | HarnessV1StreamPart::Finish { .. }
                | HarnessV1StreamPart::Error { .. }
        ) {
            ctx.lifecycle.ensure_step_open().await;
        }

        if let HarnessV1StreamPart::ToolInputStart { id, .. }
        | HarnessV1StreamPart::ToolInputDelta { id, .. }
        | HarnessV1StreamPart::ToolInputEnd { id, .. } = &value
        {
            let settled = {
                let state = ctx.state.lock();
                state.settled_host_tool_call_ids.contains(id)
                    || state.settled_builtin_approval_tool_call_ids.contains(id)
            };
            if settled {
                continue;
            }
            for display_value in strip_tool_input_work_dir(value.clone()) {
                for part in translate(&display_value) {
                    ctx.result.enqueue(part);
                }
            }
            continue;
        }

        // Strip the session working-directory prefix for everything the
        // consumer sees; `value` stays intact for host tool execution.
        let display_value = strip_work_dir(&value, &input.session_work_dir);
        let (settled_host_input_replay, settled_builtin_approval_replay) = {
            let state = ctx.state.lock();
            let id = match &display_value {
                HarnessV1StreamPart::ToolCall { tool_call_id, .. }
                | HarnessV1StreamPart::ToolResult { tool_call_id, .. }
                | HarnessV1StreamPart::ToolApprovalRequest { tool_call_id, .. } => {
                    Some(tool_call_id)
                }
                _ => None,
            };
            let host = id.is_some_and(|id| state.settled_host_tool_call_ids.contains(id));
            let builtin = !matches!(display_value, HarnessV1StreamPart::ToolResult { .. })
                && id.is_some_and(|id| state.settled_builtin_approval_tool_call_ids.contains(id));
            (host, builtin)
        };
        if settled_host_input_replay || settled_builtin_approval_replay {
            continue;
        }

        if let HarnessV1StreamPart::FinishStep {
            finish_reason,
            usage,
            harness_metadata,
            ..
        } = &value
        {
            let finish_step = FinishStepPart {
                finish_reason: finish_reason.clone(),
                usage: usage.clone(),
                harness_metadata: harness_metadata.clone(),
            };
            if host_input_pause_requested {
                ctx.wait_for_outstanding_host_tool_executions().await?;
                let (awaiting_empty, approvals_pending) = {
                    let state = ctx.state.lock();
                    (
                        state.host_tool_call_ids_awaiting_echo.is_empty(),
                        !state.pending_approvals_by_approval_id.is_empty(),
                    )
                };
                if awaiting_empty || approvals_pending {
                    ctx.pause_for_host_input(Some(finish_step)).await?;
                    return Ok(());
                }
                deferred_pause_finish_step = Some(finish_step);
                continue;
            }
            let mut state = ctx.state.lock();
            if !state.host_tool_call_ids_awaiting_echo.is_empty() {
                state.ignored_finish_step = Some(finish_step);
                continue;
            }
            state.ignored_finish_step = None;
        }
        if matches!(value, HarnessV1StreamPart::Finish { .. }) {
            if host_input_pause_requested {
                ctx.pause_for_host_input(deferred_pause_finish_step.take())
                    .await?;
                return Ok(());
            }
            let ignored = ctx.state.lock().ignored_finish_step.take();
            if let Some(finish_step) = ignored {
                ctx.wait_for_outstanding_host_tool_executions().await?;
                let closing = std::mem::take(&mut ctx.state.lock().closing_resumed_step);
                if closing {
                    ctx.publish_tool_executions().await;
                    ctx.reset_step_content();
                    ctx.result.discard_current_step_content();
                } else {
                    ctx.complete_step(
                        &finish_step.finish_reason,
                        &finish_step.usage,
                        as_provider_metadata(&finish_step.harness_metadata),
                    )
                    .await;
                }
            }
        }
        if matches!(display_value, HarnessV1StreamPart::FinishStep { .. })
            && ctx.state.lock().closing_resumed_step
        {
            ctx.state.lock().closing_resumed_step = false;
            ctx.publish_tool_executions().await;
            ctx.reset_step_content();
            ctx.result.discard_current_step_content();
            continue;
        }

        if let HarnessV1StreamPart::ToolApprovalRequest {
            approval_id,
            tool_call_id,
            ..
        } = &display_value
        {
            let (tool_call, raw_tool_call) = {
                let state = ctx.state.lock();
                (
                    state.tool_calls_by_tool_call_id.get(tool_call_id).cloned(),
                    state
                        .raw_tool_calls_by_tool_call_id
                        .get(tool_call_id)
                        .cloned(),
                )
            };
            let Some(tool_call) = tool_call else {
                return Err(plain_error(format!(
                    "Harness '{}' emitted approval request '{approval_id}' for unknown tool call '{tool_call_id}'.",
                    ctx.harness_id()
                )));
            };
            let tool_name = raw_tool_call
                .as_ref()
                .map(|r| r.tool_name.clone())
                .unwrap_or_else(|| tool_call.tool_name.clone());
            if !is_harness_v1_builtin_tool_included(
                &tool_name,
                input.builtin_tool_filtering.as_ref(),
            ) {
                if !ctx.control().supports_submit_tool_approval() {
                    return Err(plain_error(format!(
                        "Harness '{}' emitted a built-in tool approval request but does not support approval responses.",
                        ctx.harness_id()
                    )));
                }
                // The runtime may report a step boundary while it waits for
                // this decision; it must not end the step before the denied
                // call's result arrives.
                ctx.state
                    .lock()
                    .host_tool_call_ids_awaiting_echo
                    .insert(tool_call_id.clone());
                ctx.control()
                    .submit_tool_approval(HarnessV1SubmitToolApproval {
                        approval_id: approval_id.clone(),
                        approved: false,
                        reason: Some(get_harness_v1_builtin_tool_filtering_denial_reason(
                            &tool_name,
                        )),
                    })
                    .await?;
                continue;
            }
        }

        // Settle failures before the translate-and-forward step, so the
        // translated `error` part never reaches the consumer stream.
        if let (
            HarnessV1StreamPart::Error { error },
            HarnessV1StreamPart::Error {
                error: display_error,
            },
        ) = (&value, &display_value)
        {
            ctx.wait_for_outstanding_host_tool_executions().await?;
            settle_lifecycle(ctx, error.clone()).await;
            if !is_aborted(ctx) {
                log_error(ctx, "harness stream error", error);
            }
            settle_failure(ctx, display_error.clone());
            return Ok(());
        }

        let translated_parts = translate(&display_value);
        if !matches!(value, HarnessV1StreamPart::ToolResult { .. }) {
            for part in translated_parts.iter().cloned() {
                ctx.result.enqueue(part);
            }
        }

        // Tool-call validation.
        let mut validated_host_tool_call: Option<TypedToolCall> = None;
        if let HarnessV1StreamPart::ToolCall {
            tool_call_id,
            tool_name,
            provider_executed,
            dynamic,
            ..
        } = &display_value
        {
            let raw_value = RawToolCall::from_part(&value).expect("value is a tool-call");
            let is_host_tool = raw_value.provider_executed != Some(true)
                && has_tool(&ctx.active_tools, &raw_value.tool_name)
                && !(raw_value.tool_name == "askUserQuestions"
                    && has_tool(builtin_tools, &raw_value.tool_name));
            // An MCP tool can share a name with a disabled builtin whose
            // schema would reject valid MCP input.
            let is_filtered_builtin_dynamic_call = *provider_executed == Some(true)
                && *dynamic == Some(true)
                && has_tool(builtin_tools, tool_name)
                && !is_harness_v1_builtin_tool_included(
                    tool_name,
                    input.builtin_tool_filtering.as_ref(),
                )
                && !has_tool(&ctx.active_tools, tool_name);
            let parsed = validate_tool_call(
                if is_host_tool { &value } else { &display_value },
                if is_filtered_builtin_dynamic_call {
                    None
                } else {
                    Some(&input.tools)
                },
                ctx.harness_id(),
            );
            let display_tool_call = if is_host_tool {
                display_host_tool_call(&parsed, &input.session_work_dir)
            } else {
                parsed.clone()
            };
            validated_host_tool_call = is_host_tool.then_some(parsed);
            {
                let mut state = ctx.state.lock();
                state.raw_tool_calls_by_tool_call_id.insert(
                    tool_call_id.clone(),
                    RawToolCall::from_part(&display_value).expect("display value is a tool-call"),
                );
                state
                    .tool_calls_by_tool_call_id
                    .insert(tool_call_id.clone(), display_tool_call.clone());
            }
            ctx.result
                .enqueue(TextStreamPart::ToolCall(display_tool_call));
        }

        match &value {
            HarnessV1StreamPart::TextDelta { delta, .. } => {
                ctx.state.lock().step_text.push_str(delta);
            }
            HarnessV1StreamPart::ReasoningDelta { delta, .. } => {
                ctx.state.lock().step_reasoning.push_str(delta);
            }
            _ => {}
        }

        if let HarnessV1StreamPart::ToolCall {
            tool_call_id,
            provider_executed,
            ..
        } = &value
        {
            let execution = {
                let mut state = ctx.state.lock();
                match state.tool_calls_by_tool_call_id.get(tool_call_id).cloned() {
                    Some(tool_call) => {
                        state
                            .step_tool_calls
                            .push(ContentPart::ToolCall(tool_call.clone()));
                        let execution = ToolExecution::new(tool_call);
                        state
                            .tool_executions
                            .insert(tool_call_id.clone(), execution.clone());
                        Some(execution)
                    }
                    None => None,
                }
            };
            if let Some(execution) = execution
                && *provider_executed == Some(true)
            {
                ctx.publish_tool_execution_start(&execution).await;
            }
        }

        if let HarnessV1StreamPart::ToolResult {
            tool_call_id,
            result,
            is_error,
            ..
        } = &value
        {
            let (execution, provider_executed) = {
                let state = ctx.state.lock();
                (
                    state.tool_executions.get(tool_call_id).cloned(),
                    state
                        .raw_tool_calls_by_tool_call_id
                        .get(tool_call_id)
                        .is_some_and(|raw| raw.provider_executed == Some(true)),
                )
            };
            if let Some(execution) = &execution {
                let tool_output = {
                    let mut data = execution.data.lock();
                    if data.tool_output.is_none() {
                        data.tool_output = Some(if *is_error == Some(true) {
                            ToolOutput::ToolError(
                                execution
                                    .tool_call
                                    .to_error(ErrorValue::Value(result.clone())),
                            )
                        } else {
                            ToolOutput::ToolResult(execution.tool_call.to_result(result.clone()))
                        });
                    }
                    if data.tool_execution_ms.is_none()
                        && let Some(started) = data.execution_started_at
                    {
                        data.tool_execution_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                    }
                    data.tool_output.clone()
                };
                if provider_executed && let Some(tool_output) = tool_output {
                    ctx.state
                        .lock()
                        .step_provider_tool_results
                        .push(tool_output.into_content_part());
                }
                ctx.publish_tool_execution_end(execution).await;
            }
            for part in translated_parts.iter().cloned() {
                ctx.result.enqueue(part);
            }
        }

        if let HarnessV1StreamPart::ToolApprovalRequest {
            approval_id,
            tool_call_id,
            ..
        } = &value
        {
            let (tool_call, raw_tool_call, existing) = {
                let state = ctx.state.lock();
                (
                    state.tool_calls_by_tool_call_id.get(tool_call_id).cloned(),
                    state
                        .raw_tool_calls_by_tool_call_id
                        .get(tool_call_id)
                        .cloned(),
                    state
                        .pending_approvals_by_approval_id
                        .get(approval_id)
                        .cloned(),
                )
            };
            let Some(tool_call) = tool_call else {
                return Err(plain_error(format!(
                    "Harness '{}' emitted approval request '{approval_id}' for unknown tool call '{tool_call_id}'.",
                    ctx.harness_id()
                )));
            };
            let pending_approval = existing.unwrap_or_else(|| HarnessV1PendingToolApproval {
                approval_id: approval_id.clone(),
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_call.tool_name.clone(),
                input: raw_tool_call
                    .as_ref()
                    .map(|r| r.input.clone())
                    .unwrap_or_else(|| tool_call.input.to_string()),
                kind: HarnessV1PendingToolApprovalKind::Builtin,
                provider_executed: Some(
                    raw_tool_call
                        .as_ref()
                        .and_then(|r| r.provider_executed)
                        .unwrap_or(true),
                ),
                native_name: raw_tool_call.as_ref().and_then(|r| r.native_name.clone()),
            });
            {
                let mut state = ctx.state.lock();
                state.pending_approvals_by_approval_id.insert(
                    pending_approval.approval_id.clone(),
                    pending_approval.clone(),
                );
                state.pending_approvals_by_tool_call_id.insert(
                    pending_approval.tool_call_id.clone(),
                    pending_approval.clone(),
                );
            }
            if let Some(continuation) = ctx
                .continuations_by_approval_id
                .get(&pending_approval.approval_id)
            {
                let outcome = ctx
                    .process_pending_approval_continuation(&pending_approval, continuation)
                    .await?;
                if matches!(outcome, ApprovalContinuationOutcome::AwaitingToolResult) {
                    return Ok(());
                }
                ctx.state.lock().closing_resumed_step = true;
                continue;
            }
            if let Some(f) = &input.on_pending_tool_approval {
                f(pending_approval.clone());
            }
            ctx.enqueue_approval_request(&pending_approval.approval_id, &tool_call, None);
            host_input_pause_requested = true;
            continue;
        }

        // Drive step boundaries.
        if let HarnessV1StreamPart::FinishStep {
            finish_reason,
            usage,
            harness_metadata,
            checkpoint,
        } = &value
        {
            ctx.wait_for_outstanding_host_tool_executions().await?;
            ctx.complete_step(finish_reason, usage, as_provider_metadata(harness_metadata))
                .await;
            if input
                .stop_conditions
                .as_ref()
                .is_some_and(|c| !c.is_empty())
            {
                *pending_stop_boundary = Some(checkpoint.as_ref().map(|c| c.pin()));
            }
        }

        if let HarnessV1StreamPart::Finish {
            finish_reason,
            total_usage,
            harness_metadata,
        } = &value
        {
            ctx.wait_for_outstanding_host_tool_executions().await?;
            final_finish = Some((
                finish_reason.clone(),
                total_usage.clone(),
                harness_metadata.clone(),
            ));
            let steps = ctx.state.lock().completed_steps.clone();
            ctx.lifecycle
                .end(&steps, as_language_model_usage(total_usage))
                .await;
        }

        // Execute host-side tools when the harness asks for one.
        if let Some(tool_call) = RawToolCall::from_part(&value)
            && tool_call.provider_executed != Some(true)
        {
            let parsed_tool_call = ctx
                .state
                .lock()
                .tool_calls_by_tool_call_id
                .get(&tool_call.tool_call_id)
                .cloned();
            let Some(parsed_tool_call) = parsed_tool_call else {
                return Err(plain_error(format!(
                    "Harness '{}' could not find parsed tool call '{}' for custom tool approval.",
                    ctx.harness_id(),
                    tool_call.tool_call_id
                )));
            };
            let is_client_executed_builtin = tool_call.tool_name == "askUserQuestions"
                && builtin_tools.contains_key(&tool_call.tool_name);
            if !is_client_executed_builtin && !has_tool(&ctx.active_tools, &tool_call.tool_name) {
                let output = execution_denied(Some(
                    get_harness_v1_builtin_tool_filtering_denial_reason(&tool_call.tool_name),
                ));
                ctx.submit_tool_result(HarnessV1SubmitToolResult {
                    tool_call_id: tool_call.tool_call_id.clone(),
                    output: execution_denied_json(&output),
                    is_error: None,
                    tool_result: Some(ToolResultPart {
                        tool_call_id: tool_call.tool_call_id.clone(),
                        tool_name: tool_call.tool_name.clone(),
                        output: output.clone(),
                        provider_options: None,
                    }),
                })
                .await?;
                let execution = ctx
                    .state
                    .lock()
                    .tool_executions
                    .get(&tool_call.tool_call_id)
                    .cloned();
                if let Some(execution) = execution {
                    execution.data.lock().tool_output = Some(to_tool_output(
                        &parsed_tool_call,
                        &HostToolOutcome::Ok(execution_denied_json(&output)),
                    ));
                }
                continue;
            }
            if is_client_executed_builtin {
                ctx.record_pending_tool_result(&tool_call);
                host_input_pause_requested = true;
                continue;
            }
            let Some(validated_host_tool_call) = validated_host_tool_call else {
                return Err(plain_error(format!(
                    "Harness '{}' could not validate host tool '{}'.",
                    ctx.harness_id(),
                    tool_call.tool_name
                )));
            };
            if validated_host_tool_call.is_invalid() {
                {
                    let mut state = ctx.state.lock();
                    state
                        .settled_host_tool_call_ids
                        .insert(tool_call.tool_call_id.clone());
                    state.tool_executions.shift_remove(&tool_call.tool_call_id);
                    let mut error = TypedToolError::new(
                        &tool_call.tool_call_id,
                        &tool_call.tool_name,
                        Some(parsed_tool_call.input.clone()),
                        ErrorValue::message(INVALID_TOOL_INPUT_MESSAGE),
                    );
                    error.dynamic = Some(true);
                    state
                        .buffered_tool_outcomes
                        .push(BufferedToolOutcome::Part(TextStreamPart::ToolError(error)));
                }
                ctx.submit_tool_result(HarnessV1SubmitToolResult {
                    tool_call_id: tool_call.tool_call_id.clone(),
                    output: json!({ "error": INVALID_TOOL_INPUT_MESSAGE }),
                    is_error: Some(true),
                    tool_result: None,
                })
                .await?;
                continue;
            }
            let decision =
                resolve_custom_tool_approval(&tool_call.tool_name, input.tool_approval.as_ref());
            if let CustomToolApprovalDecision::Deny { reason } = &decision {
                let approval_id = generate_id();
                ctx.enqueue_approval_request(&approval_id, &parsed_tool_call, Some(true));
                ctx.enqueue_automatic_approval_response(
                    &approval_id,
                    &parsed_tool_call,
                    false,
                    reason.clone(),
                    Some(false),
                );
                let output = execution_denied(reason.clone());
                let execution = ctx
                    .state
                    .lock()
                    .tool_executions
                    .get(&tool_call.tool_call_id)
                    .cloned();
                if let Some(execution) = &execution {
                    ctx.publish_tool_execution_start(execution).await;
                }
                ctx.submit_tool_result(HarnessV1SubmitToolResult {
                    tool_call_id: tool_call.tool_call_id.clone(),
                    output: execution_denied_json(&output),
                    is_error: None,
                    tool_result: Some(ToolResultPart {
                        tool_call_id: tool_call.tool_call_id.clone(),
                        tool_name: tool_call.tool_name.clone(),
                        output: output.clone(),
                        provider_options: None,
                    }),
                })
                .await?;
                if let Some(execution) = &execution {
                    execution.data.lock().tool_output = Some(to_tool_output(
                        &parsed_tool_call,
                        &HostToolOutcome::Ok(execution_denied_json(&output)),
                    ));
                    ctx.publish_tool_execution_end(execution).await;
                }
                continue;
            }
            let pending_approval = ctx
                .state
                .lock()
                .pending_approvals_by_tool_call_id
                .get(&tool_call.tool_call_id)
                .cloned()
                .or_else(|| {
                    matches!(decision, CustomToolApprovalDecision::Request).then(|| {
                        HarnessV1PendingToolApproval {
                            approval_id: generate_id(),
                            tool_call_id: tool_call.tool_call_id.clone(),
                            tool_name: tool_call.tool_name.clone(),
                            input: tool_call.input.clone(),
                            kind: HarnessV1PendingToolApprovalKind::Custom,
                            provider_executed: Some(false),
                            native_name: tool_call.native_name.clone(),
                        }
                    })
                });
            if let Some(pending_approval) = pending_approval {
                {
                    let mut state = ctx.state.lock();
                    state.pending_approvals_by_approval_id.insert(
                        pending_approval.approval_id.clone(),
                        pending_approval.clone(),
                    );
                    state.pending_approvals_by_tool_call_id.insert(
                        pending_approval.tool_call_id.clone(),
                        pending_approval.clone(),
                    );
                }
                if let Some(continuation) = ctx
                    .continuations_by_approval_id
                    .get(&pending_approval.approval_id)
                {
                    let outcome = ctx
                        .process_pending_approval_continuation(&pending_approval, continuation)
                        .await?;
                    if matches!(outcome, ApprovalContinuationOutcome::AwaitingToolResult) {
                        return Ok(());
                    }
                    ctx.state.lock().closing_resumed_step = true;
                    continue;
                }
                let pending_parsed_tool_call = ctx
                    .state
                    .lock()
                    .tool_calls_by_tool_call_id
                    .get(&pending_approval.tool_call_id)
                    .cloned();
                let Some(pending_parsed_tool_call) = pending_parsed_tool_call else {
                    return Err(plain_error(format!(
                        "Harness '{}' could not find parsed tool call '{}' for approval request '{}'.",
                        ctx.harness_id(),
                        pending_approval.tool_call_id,
                        pending_approval.approval_id
                    )));
                };
                if let Some(f) = &input.on_pending_tool_approval {
                    f(pending_approval.clone());
                }
                ctx.enqueue_approval_request(
                    &pending_approval.approval_id,
                    &pending_parsed_tool_call,
                    None,
                );
                host_input_pause_requested = true;
                continue;
            }
            if !is_executable_tool(ctx.active_tools.get(&tool_call.tool_name)) {
                ctx.record_pending_tool_result(&tool_call);
                host_input_pause_requested = true;
                continue;
            }
            ctx.state
                .lock()
                .host_tool_call_ids_awaiting_echo
                .insert(tool_call.tool_call_id.clone());
            let task_ctx = ctx.clone();
            let handle = tokio::spawn(async move {
                let ctx = task_ctx;
                let tool_execution = ctx
                    .state
                    .lock()
                    .tool_executions
                    .get(&tool_call.tool_call_id)
                    .cloned();
                let Some(tool_execution) = tool_execution else {
                    return Err(plain_error(format!(
                        "Harness '{}' could not track host tool '{}'.",
                        ctx.harness_id(),
                        tool_call.tool_name
                    )));
                };
                ctx.publish_tool_execution_start(&tool_execution).await;
                let execution_started_at = Instant::now();
                tool_execution.data.lock().execution_started_at = Some(execution_started_at);
                let execution = ctx
                    .maybe_execute_host_tool(&tool_call, &validated_host_tool_call)
                    .await?;
                let HostToolExecution::Executed(outcome) = execution else {
                    return Err(plain_error(format!(
                        "Harness '{}' could not execute host tool '{}'.",
                        ctx.harness_id(),
                        tool_call.tool_name
                    )));
                };
                {
                    let mut data = tool_execution.data.lock();
                    data.tool_output = Some(to_tool_output(&parsed_tool_call, &outcome));
                    data.stream_error = get_stream_error(&outcome);
                    data.tool_execution_ms =
                        Some(execution_started_at.elapsed().as_secs_f64() * 1000.0);
                }
                ctx.publish_tool_execution_end(&tool_execution).await;
                Ok(())
            });
            ctx.start_host_tool_execution(handle);
        }
    }

    if host_input_pause_requested && !is_turn_suspending(ctx) {
        ctx.pause_for_host_input(deferred_pause_finish_step.take())
            .await?;
        return Ok(());
    }
    ctx.wait_for_outstanding_host_tool_executions().await?;
    let suspending = is_turn_suspending(ctx);
    if suspending {
        ctx.publish_tool_executions().await;
        if final_finish.is_none() {
            // A timed slice may stop mid-step; its partial content cannot
            // form a valid StepResult in this slice.
            ctx.result.discard_current_step_content();
        }
    } else if final_finish.is_some() {
        if let Some(f) = &input.on_turn_finished {
            f();
        }
    } else if let Some(f) = &input.on_turn_failed {
        f();
    }
    match &final_finish {
        Some((finish_reason, total_usage, harness_metadata)) => {
            ctx.result
                .finish(Some((
                    finish_reason,
                    total_usage,
                    as_provider_metadata(harness_metadata),
                )))
                .await;
        }
        None => ctx.result.finish(None).await,
    }
    Ok(())
}

/// Validate an inbound `tool-call` event against the tool set's schema with
/// the AI SDK's `parseToolCall`. Returns the parsed tool call, or a
/// `dynamic` + `invalid` one on failure (unknown tool, schema mismatch,
/// malformed JSON).
pub fn validate_tool_call(
    event: &HarnessV1StreamPart,
    tools: Option<&ToolSet>,
    harness_id: &str,
) -> TypedToolCall {
    let HarnessV1StreamPart::ToolCall {
        tool_call_id,
        tool_name,
        input,
        provider_executed,
        dynamic,
        provider_metadata,
        native_name,
    } = event
    else {
        panic!("validate_tool_call expects a tool-call part");
    };
    let provider_metadata = match native_name {
        Some(native) if native != tool_name => {
            let mut metadata = provider_metadata.clone().unwrap_or_default();
            let mut harness = metadata.get(harness_id).cloned().unwrap_or_default();
            harness.insert("nativeToolName".into(), native.clone().into());
            metadata.insert(harness_id.to_string(), harness);
            Some(metadata)
        }
        _ => provider_metadata.clone(),
    };
    parse_tool_call(
        &LanguageModelV4ToolCall {
            tool_call_id: tool_call_id.clone(),
            tool_name: tool_name.clone(),
            input: input.clone(),
            provider_executed: *provider_executed,
            dynamic: *dynamic,
            provider_metadata,
        },
        tools,
    )
}

/// Best-effort plain text of the turn's prompt, for telemetry input
/// messages.
fn prompt_to_text(prompt: &HarnessV1Prompt) -> String {
    match prompt {
        HarnessV1Prompt::Text(text) => text.clone(),
        HarnessV1Prompt::Message(message) => match &message.content {
            UserContent::Text(text) => text.clone(),
            UserContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    UserContentPart::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect(),
        },
    }
}

#[cfg(test)]
#[path = "run_prompt_tests.rs"]
mod tests;
