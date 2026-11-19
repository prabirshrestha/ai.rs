//! Port of `agent/harness-agent-session.ts`.

use std::sync::Arc;

use futures::future::{BoxFuture, FutureExt, Shared};
use indexmap::IndexMap;
use parking_lot::Mutex;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use super::internal::harness_stream_text_result::HarnessStreamTextResult;
use super::internal::lifecycle_state_validation::{
    validate_continue_turn_state, validate_resume_session_state,
};
use super::internal::run_prompt::{RunPromptInput, RunPromptMode, run_prompt};
use super::internal::turn_telemetry::HarnessAgentLifecycleCallbacks;
use super::settings::HarnessAgentToolApprovalConfiguration;
use crate::harness::errors::{BoxError, HarnessError, HarnessResult};
use crate::harness::sdk::content::{Context, ToolsContext};
use crate::harness::sdk::error_value::{SharedError, SharedErrorBox};
use crate::harness::sdk::messages::{ToolApprovalResponse, ToolResultPart};
use crate::harness::sdk::output::SharedOutput;
use crate::harness::sdk::stop_condition::StopCondition;
use crate::harness::sdk::telemetry::TelemetryOptions;
use crate::harness::sdk::tool::ToolSet;
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1Prompt, HarnessV1PromptControl, HarnessV1ReadHistoryResult,
    HarnessV1SandboxSessionRef, HarnessV1Session,
};
use crate::harness::spec::lifecycle_state::{
    HarnessV1ContinueTurnState, HarnessV1PendingToolApproval, HarnessV1PendingToolResult,
    HarnessV1ResumeSessionState, HarnessV1TurnSettings, ResumeSessionType,
};
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::tools::{
    HarnessV1BuiltinToolFiltering, HarnessV1Skill, HarnessV1ToolSpec,
};
use crate::harness::utils::plain_error;
use crate::harness::utils::sandbox_home_dir::get_restricted_sandbox_session;

/// What `promptTurn` / `continueTurn` return.
pub struct HarnessAgentTurnResult {
    pub result: HarnessStreamTextResult,
    /// Settles once the turn has wound down.
    pub done: Shared<BoxFuture<'static, ()>>,
    /// Settles once the turn's prompt control is available (or the turn has
    /// paused or ended without one).
    pub ready: BoxFuture<'static, ()>,
}

/// `HarnessAgentSessionState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarnessAgentSessionState {
    Active,
    Detached,
    Stopped,
    Destroyed,
}

/// `HarnessAgentTurnState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarnessAgentTurnState {
    Idle,
    Running,
    AwaitingApproval,
    AwaitingToolResult,
    Suspended,
}

type ControlPromise = Shared<BoxFuture<'static, Option<Arc<dyn HarnessV1PromptControl>>>>;

/// `ActivePromptControl`. [rust] `id` stands in for object identity.
struct ActivePromptControl {
    id: u64,
    turn_id: u64,
    promise: ControlPromise,
    /// `resolve`; `None` once `settled`.
    resolve: Option<oneshot::Sender<Option<Arc<dyn HarnessV1PromptControl>>>>,
}

/// `ActiveTurnSettings`.
#[derive(Clone)]
struct ActiveTurnSettings {
    persisted: HarnessV1TurnSettings,
    tools: ToolSet,
    tools_context: ToolsContext,
    runtime_context: Context,
    active_tools: ToolSet,
    builtin_tool_filtering: Option<HarnessV1BuiltinToolFiltering>,
}

type SuspendedTurnState =
    Shared<BoxFuture<'static, Result<HarnessV1ContinueTurnState, SharedError>>>;

struct State {
    underlying_session: Option<Arc<dyn HarnessV1Session>>,
    sandbox_session: Option<HarnessV1SandboxSessionRef>,
    pending_tool_approvals: IndexMap<String, HarnessV1PendingToolApproval>,
    pending_tool_results: IndexMap<String, HarnessV1PendingToolResult>,
    session_state: HarnessAgentSessionState,
    turn_state: HarnessAgentTurnState,
    turn_sequence: u64,
    active_turn_sequence: u64,
    active_prompt_done: Option<Shared<BoxFuture<'static, ()>>>,
    active_prompt_control: Option<ActivePromptControl>,
    prompt_control_ids: u64,
    suspended_turn_state: Option<SuspendedTurnState>,
    active_turn_settings: Option<ActiveTurnSettings>,
    persisted_turn_settings: Option<HarnessV1TurnSettings>,
}

struct Inner {
    session_id: String,
    harness: Arc<dyn HarnessV1>,
    session_work_dir: String,
    owns_sandbox_lifecycle: bool,
    tool_approval: Option<HarnessAgentToolApprovalConfiguration>,
    resumed_tools_context: Option<ToolsContext>,
    resumed_runtime_context: Option<Context>,
    is_resume: bool,
    state: Mutex<State>,
}

/// Options of [`HarnessAgentSession::new`].
pub struct HarnessAgentSessionOptions {
    pub session_id: String,
    pub harness: Arc<dyn HarnessV1>,
    pub underlying_session: Arc<dyn HarnessV1Session>,
    pub sandbox_session: HarnessV1SandboxSessionRef,
    /// Defaults to `true`.
    pub owns_sandbox_lifecycle: Option<bool>,
    pub session_work_dir: String,
    pub tool_approval: Option<HarnessAgentToolApprovalConfiguration>,
    pub pending_tool_approvals: Vec<HarnessV1PendingToolApproval>,
    pub pending_tool_results: Vec<HarnessV1PendingToolResult>,
    pub turn_settings: Option<HarnessV1TurnSettings>,
    pub resumed_tools_context: Option<ToolsContext>,
    pub resumed_runtime_context: Option<Context>,
    pub turn_state: Option<HarnessAgentTurnState>,
}

impl HarnessAgentSessionOptions {
    /// Options with the required fields set.
    pub fn new(
        session_id: impl Into<String>,
        harness: Arc<dyn HarnessV1>,
        underlying_session: Arc<dyn HarnessV1Session>,
        sandbox_session: HarnessV1SandboxSessionRef,
        session_work_dir: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            harness,
            underlying_session,
            sandbox_session,
            owns_sandbox_lifecycle: None,
            session_work_dir: session_work_dir.into(),
            tool_approval: None,
            pending_tool_approvals: Vec::new(),
            pending_tool_results: Vec::new(),
            turn_settings: None,
            resumed_tools_context: None,
            resumed_runtime_context: None,
            turn_state: None,
        }
    }
}

/// The turn-scoped settings of `promptTurn` / `continueTurn`.
#[derive(Clone)]
pub struct HarnessAgentTurnOptions {
    pub model: Option<String>,
    pub skills: Vec<HarnessV1Skill>,
    pub instructions: Option<String>,
    pub tools: ToolSet,
    pub tools_context: ToolsContext,
    pub active_tools: ToolSet,
    pub tool_specs: Vec<HarnessV1ToolSpec>,
    pub builtin_tool_filtering: Option<HarnessV1BuiltinToolFiltering>,
    pub runtime_context: Context,
    pub abort_signal: Option<CancellationToken>,
    pub response_format: Option<HarnessV1ResponseFormat>,
    pub output: Option<SharedOutput>,
    pub telemetry: Option<TelemetryOptions>,
    pub callbacks: HarnessAgentLifecycleCallbacks,
    pub stop_conditions: Vec<StopCondition>,
}

fn shared_to_box(error: SharedError) -> BoxError {
    Box::new(SharedErrorBox(error))
}

/// Live harness session held by the caller.
///
/// Created by `HarnessAgent::create_session`. Owns the underlying adapter
/// session and holds its sandbox session.
///
/// Pass the session back to `agent.generate` / `agent.stream` on every call;
/// end the local handle with `detach()`, `stop()`, or `destroy()`.
///
/// After any lifecycle method has resolved, the session is unusable: any
/// subsequent `generate` / `stream` call against it fails.
#[derive(Clone)]
pub struct HarnessAgentSession {
    inner: Arc<Inner>,
}

impl HarnessAgentSession {
    pub fn new(options: HarnessAgentSessionOptions) -> Self {
        let mut pending_tool_approvals = IndexMap::new();
        for approval in options.pending_tool_approvals {
            pending_tool_approvals.insert(approval.approval_id.clone(), approval);
        }
        let mut pending_tool_results = IndexMap::new();
        for pending_result in options.pending_tool_results {
            pending_tool_results.insert(pending_result.tool_call_id.clone(), pending_result);
        }
        let turn_state = options
            .turn_state
            .unwrap_or(if !pending_tool_approvals.is_empty() {
                HarnessAgentTurnState::AwaitingApproval
            } else if !pending_tool_results.is_empty() {
                HarnessAgentTurnState::AwaitingToolResult
            } else {
                HarnessAgentTurnState::Idle
            });
        let is_resume = options.underlying_session.is_resume();
        Self {
            inner: Arc::new(Inner {
                session_id: options.session_id,
                harness: options.harness,
                session_work_dir: options.session_work_dir,
                owns_sandbox_lifecycle: options.owns_sandbox_lifecycle.unwrap_or(true),
                tool_approval: options.tool_approval,
                resumed_tools_context: options.resumed_tools_context,
                resumed_runtime_context: options.resumed_runtime_context,
                is_resume,
                state: Mutex::new(State {
                    underlying_session: Some(options.underlying_session),
                    sandbox_session: Some(options.sandbox_session),
                    pending_tool_approvals,
                    pending_tool_results,
                    session_state: HarnessAgentSessionState::Active,
                    turn_state,
                    turn_sequence: 0,
                    active_turn_sequence: 0,
                    active_prompt_done: None,
                    active_prompt_control: None,
                    prompt_control_ids: 0,
                    suspended_turn_state: None,
                    active_turn_settings: None,
                    persisted_turn_settings: options.turn_settings,
                }),
            }),
        }
    }

    /// Stable identifier the harness adapter saw in `doStart`.
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    /// Whether this session was created from `resumeFrom` or `continueFrom`.
    pub fn is_resume(&self) -> bool {
        self.inner.is_resume
    }

    /// Active sandbox session.
    pub fn get_sandbox_session(&self) -> HarnessResult<HarnessV1SandboxSessionRef> {
        let state = self.inner.state.lock();
        match (&state.session_state, &state.sandbox_session) {
            (HarnessAgentSessionState::Active, Some(s)) => Ok(s.clone()),
            _ => Err(self.ended_error()),
        }
    }

    /// Working directory the agent runs in for this session.
    pub fn get_session_work_dir(&self) -> &str {
        &self.inner.session_work_dir
    }

    pub fn has_unfinished_turn(&self) -> bool {
        self.inner.state.lock().turn_state != HarnessAgentTurnState::Idle
    }

    fn ended_error(&self) -> BoxError {
        plain_error(format!(
            "Harness session {} has ended and cannot be reused.",
            self.inner.session_id
        ))
    }

    /// `promptTurn`.
    pub fn prompt_turn(
        &self,
        prompt: HarnessV1Prompt,
        options: HarnessAgentTurnOptions,
    ) -> HarnessResult<HarnessAgentTurnResult> {
        let session = self.require_reusable_session()?;
        self.require_promptable_turn()?;
        let persisted = HarnessV1TurnSettings {
            model: options.model.clone(),
            skills: options.skills.clone(),
            instructions: options.instructions.clone(),
            tools: options.tool_specs.clone(),
        };
        {
            let mut state = self.inner.state.lock();
            state.persisted_turn_settings = Some(persisted.clone());
            state.active_turn_settings = Some(ActiveTurnSettings {
                persisted,
                tools: options.tools.clone(),
                tools_context: options.tools_context.clone(),
                runtime_context: options.runtime_context.clone(),
                active_tools: options.active_tools.clone(),
                builtin_tool_filtering: options.builtin_tool_filtering.clone(),
            });
        }
        let sandbox_session = self.get_sandbox_session()?;
        let turn_id = self.start_tracked_turn();
        let mut input = self.run_input(session, &sandbox_session, turn_id, &options);
        input.prompt = Some(prompt);
        input.model = options.model;
        input.skills = options.skills;
        input.instructions = options.instructions;
        input.tools = options.tools;
        input.tools_context = Some(options.tools_context);
        input.active_tools = Some(options.active_tools);
        input.tool_specs = options.tool_specs;
        input.builtin_tool_filtering = options.builtin_tool_filtering;
        input.runtime_context = options.runtime_context;
        Ok(self.start_run(input, turn_id))
    }

    /// `continueTurn`.
    pub fn continue_turn(
        &self,
        options: HarnessAgentTurnOptions,
        tool_approval_continuations: Vec<ToolApprovalResponse>,
        tool_result_continuations: Vec<ToolResultPart>,
    ) -> HarnessResult<HarnessAgentTurnResult> {
        let session = self.require_reusable_session()?;
        self.require_continuable_turn()?;
        let turn_settings = self.resolve_active_turn_settings(&options)?;
        let sandbox_session = self.get_sandbox_session()?;
        let turn_id = self.start_tracked_turn();
        let mut input = self.run_input(session, &sandbox_session, turn_id, &options);
        input.mode = RunPromptMode::Continue;
        input.model = turn_settings.persisted.model.clone();
        input.skills = turn_settings.persisted.skills.clone();
        input.instructions = turn_settings.persisted.instructions.clone();
        input.tools = turn_settings.tools;
        input.tools_context = Some(turn_settings.tools_context);
        input.active_tools = Some(turn_settings.active_tools);
        input.tool_specs = turn_settings.persisted.tools;
        input.builtin_tool_filtering = turn_settings.builtin_tool_filtering;
        input.runtime_context = turn_settings.runtime_context;
        input.tool_approval_continuations = tool_approval_continuations;
        input.tool_result_continuations = tool_result_continuations;
        Ok(self.start_run(input, turn_id))
    }

    /// The `runPrompt` input both turn kinds share.
    fn run_input(
        &self,
        session: Arc<dyn HarnessV1Session>,
        sandbox_session: &HarnessV1SandboxSessionRef,
        turn_id: u64,
        options: &HarnessAgentTurnOptions,
    ) -> RunPromptInput {
        let mut input = RunPromptInput::new(
            self.inner.harness.clone(),
            session.clone(),
            get_restricted_sandbox_session(sandbox_session),
            self.inner.session_work_dir.clone(),
        );
        input.abort_signal = options.abort_signal.clone();
        input.response_format = options.response_format.clone();
        input.output = options.output.clone();
        input.telemetry = options.telemetry.clone();
        input.callbacks = Some(options.callbacks.clone());
        input.stop_conditions = Some(options.stop_conditions.clone());
        input.tool_approval = self.inner.tool_approval.clone();
        input.pending_tool_approvals = self.get_pending_tool_approvals();
        input.pending_tool_results = self.get_pending_tool_results();

        let this = self.clone();
        input.on_pending_tool_approval = Some(Arc::new(move |approval| {
            this.inner
                .state
                .lock()
                .pending_tool_approvals
                .insert(approval.approval_id.clone(), approval);
            this.mark_if_active(HarnessAgentTurnState::AwaitingApproval);
        }));
        let this = self.clone();
        input.on_tool_approval_settled = Some(Arc::new(move |approval_id| {
            this.inner
                .state
                .lock()
                .pending_tool_approvals
                .shift_remove(&approval_id);
        }));
        let this = self.clone();
        input.on_pending_tool_result = Some(Arc::new(move |pending_result| {
            this.inner
                .state
                .lock()
                .pending_tool_results
                .insert(pending_result.tool_call_id.clone(), pending_result);
            this.mark_if_active(HarnessAgentTurnState::AwaitingToolResult);
        }));
        let this = self.clone();
        input.on_tool_result_settled = Some(Arc::new(move |tool_call_id| {
            this.inner
                .state
                .lock()
                .pending_tool_results
                .shift_remove(&tool_call_id);
        }));
        let this = self.clone();
        input.on_turn_finished = Some(Arc::new(move || this.finish_tracked_turn(turn_id)));
        let this = self.clone();
        input.on_turn_failed = Some(Arc::new(move || this.finish_tracked_turn(turn_id)));
        let this = self.clone();
        input.on_prompt_control_available = Some(Arc::new(move |control| {
            this.set_prompt_control(turn_id, control)
        }));
        let this = self.clone();
        input.is_turn_suspending = Some(Arc::new(move || {
            let state = this.inner.state.lock();
            state.active_turn_sequence == turn_id && state.suspended_turn_state.is_some()
        }));
        let this = self.clone();
        input.on_stop_condition_met = Some(Arc::new(move || {
            let (this, session) = (this.clone(), session.clone());
            Box::pin(async move { this.capture_stop_condition_boundary(session, turn_id).await })
        }));
        input
    }

    fn start_run(&self, input: RunPromptInput, turn_id: u64) -> HarnessAgentTurnResult {
        let turn = run_prompt(input);
        self.inner.state.lock().active_prompt_done = Some(turn.done.clone());
        HarnessAgentTurnResult {
            result: turn.result,
            done: turn.done,
            ready: self.wait_for_prompt_control(turn_id),
        }
    }

    /// Ask the underlying runtime to compact its context.
    ///
    /// Fails with a capability-unsupported error for harnesses that cannot
    /// trigger compaction manually, and when the session has ended.
    pub async fn compact(&self, custom_instructions: Option<String>) -> HarnessResult<()> {
        self.require_reusable_session()?
            .do_compact(custom_instructions)
            .await
    }

    /// Submit another user message to the active turn.
    pub async fn experimental_steer_turn(&self, text: String) -> HarnessResult<()> {
        self.require_reusable_session()?;
        let (id, turn_id, promise) = {
            let state = self.inner.state.lock();
            match &state.active_prompt_control {
                Some(control)
                    if state.turn_state == HarnessAgentTurnState::Running
                        && state.suspended_turn_state.is_none() =>
                {
                    (control.id, control.turn_id, control.promise.clone())
                }
                _ => {
                    return Err(plain_error(format!(
                        "Harness session {} has no running turn to steer.",
                        self.inner.session_id
                    )));
                }
            }
        };

        let control = promise.await;
        let still_targeted = {
            let state = self.inner.state.lock();
            state.session_state == HarnessAgentSessionState::Active
                && state.turn_state == HarnessAgentTurnState::Running
                && state.suspended_turn_state.is_none()
                && state.active_prompt_control.as_ref().map(|c| c.id) == Some(id)
                && state.active_turn_sequence == turn_id
        };
        let Some(control) = control.filter(|_| still_targeted) else {
            return Err(plain_error(format!(
                "Harness session {} no longer has the running turn targeted for steering.",
                self.inner.session_id
            )));
        };

        if !control.supports_submit_user_message() {
            let harness_id = self.inner.harness.harness_id().to_string();
            return Err(HarnessError::capability_unsupported(
                format!("Harness '{harness_id}' does not support steering active turns."),
                Some(harness_id),
            )
            .into());
        }

        control.submit_user_message(text).await
    }

    fn require_active_session(&self, action: &str) -> HarnessResult<Arc<dyn HarnessV1Session>> {
        let state = self.inner.state.lock();
        match (&state.session_state, &state.underlying_session) {
            (HarnessAgentSessionState::Active, Some(session)) => Ok(session.clone()),
            _ => Err(plain_error(format!(
                "Harness session {} is not active and cannot {action}.",
                self.inner.session_id
            ))),
        }
    }

    /// Park the session, returning a payload the caller can persist and
    /// later pass to `create_session` with `resume_from` to reconnect. The
    /// runtime and sandbox keep running; this local handle becomes unusable.
    pub async fn detach(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        let session = self.require_active_session("be detached")?;
        let state = if self.has_unfinished_turn() {
            to_resume_state_with_continuation(self.finalize_current_turn_suspension(session).await?)
        } else {
            let raw = session.do_detach().await?;
            validate_resume_session_state(&*self.inner.harness, &raw)?
        };
        self.end_local_handle(HarnessAgentSessionState::Detached);
        Ok(state)
    }

    /// Read the conversation history the runtime itself persisted.
    ///
    /// Pass a previous result's `cursor` as `since` to read only the delta.
    pub async fn read_history(
        &self,
        since: Option<String>,
    ) -> HarnessResult<HarnessV1ReadHistoryResult> {
        let session = self.require_active_session("read history")?;
        if !session.supports_read_history() {
            let harness_id = self.inner.harness.harness_id().to_string();
            return Err(HarnessError::capability_unsupported(
                format!("Harness '{harness_id}' does not support reading the runtime's conversation history."),
                Some(harness_id),
            )
            .into());
        }
        session.do_read_history(since).await
    }

    /// Persist enough state to resume later, then stop the runtime and any
    /// harness-owned sandbox.
    pub async fn stop(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        let session = self.require_active_session("be stopped")?;
        let sandbox_session = self.get_sandbox_session()?;
        let result = async {
            if self.has_unfinished_turn() {
                return Ok(to_resume_state_with_continuation(
                    self.finalize_current_turn_suspension(session).await?,
                ));
            }
            let raw = session.do_stop().await?;
            validate_resume_session_state(&*self.inner.harness, &raw)
        }
        .await;
        self.end_local_handle(HarnessAgentSessionState::Stopped);
        if self.inner.owns_sandbox_lifecycle
            && let HarnessV1SandboxSessionRef::Network(sandbox) = &sandbox_session
        {
            let _ = sandbox.stop().await;
        }
        result
    }

    /// Stop the runtime and discard resumability. A harness-owned network
    /// sandbox is destroyed.
    pub async fn destroy(&self) -> HarnessResult<()> {
        let (session, sandbox_session) = {
            let state = self.inner.state.lock();
            if state.session_state != HarnessAgentSessionState::Active {
                return Ok(());
            }
            (
                state.underlying_session.clone(),
                state.sandbox_session.clone(),
            )
        };
        let Some(sandbox_session) = sandbox_session else {
            return Err(self.ended_error());
        };
        self.end_local_handle(HarnessAgentSessionState::Destroyed);
        if let Some(session) = session {
            let _ = session.do_destroy().await;
        }
        if !self.inner.owns_sandbox_lifecycle {
            return Ok(());
        }
        if let HarnessV1SandboxSessionRef::Network(sandbox) = &sandbox_session {
            let _ = sandbox.destroy().await;
        }
        Ok(())
    }

    /// Gracefully freeze the active turn at the slice boundary and return
    /// the continuation payload, leaving the sandbox and runtime running so
    /// the next process can continue. The session is detached afterwards.
    pub async fn suspend_turn(&self) -> HarnessResult<HarnessV1ContinueTurnState> {
        let session = self.require_active_session("be suspended")?;
        if !self.has_unfinished_turn() {
            return Err(plain_error(format!(
                "Harness session {} has no unfinished turn to suspend.",
                self.inner.session_id
            )));
        }
        let result = self.finalize_current_turn_suspension(session).await;
        self.end_local_handle(HarnessAgentSessionState::Detached);
        result
    }

    fn get_pending_tool_approvals(&self) -> Vec<HarnessV1PendingToolApproval> {
        self.inner
            .state
            .lock()
            .pending_tool_approvals
            .values()
            .cloned()
            .collect()
    }

    fn get_pending_tool_results(&self) -> Vec<HarnessV1PendingToolResult> {
        self.inner
            .state
            .lock()
            .pending_tool_results
            .values()
            .cloned()
            .collect()
    }

    fn add_pending_tool_state(
        &self,
        state: HarnessV1ContinueTurnState,
    ) -> HarnessV1ContinueTurnState {
        let (turn_settings, pending_tool_approvals, pending_tool_results) = {
            let inner = self.inner.state.lock();
            (
                inner.persisted_turn_settings.clone(),
                inner
                    .pending_tool_approvals
                    .values()
                    .cloned()
                    .collect::<Vec<_>>(),
                inner
                    .pending_tool_results
                    .values()
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        };
        if pending_tool_approvals.is_empty() && pending_tool_results.is_empty() {
            return HarnessV1ContinueTurnState {
                pending_tool_approvals: None,
                pending_tool_results: None,
                turn_settings,
                ..state
            };
        }
        HarnessV1ContinueTurnState {
            turn_settings: turn_settings.or(state.turn_settings),
            pending_tool_approvals: if pending_tool_approvals.is_empty() {
                state.pending_tool_approvals
            } else {
                Some(pending_tool_approvals)
            },
            pending_tool_results: if pending_tool_results.is_empty() {
                state.pending_tool_results
            } else {
                Some(pending_tool_results)
            },
            ..state
        }
    }

    async fn suspend_current_turn(
        &self,
        session: Arc<dyn HarnessV1Session>,
    ) -> HarnessResult<HarnessV1ContinueTurnState> {
        self.clear_active_prompt_control(None);
        let suspended = {
            let mut state = self.inner.state.lock();
            state
                .suspended_turn_state
                .get_or_insert_with(|| {
                    let this = self.clone();
                    async move {
                        let raw = session.do_suspend_turn().await.map_err(SharedError::from)?;
                        let validated = validate_continue_turn_state(&*this.inner.harness, &raw)
                            .map_err(SharedError::from)?;
                        Ok(this.add_pending_tool_state(validated))
                    }
                    .boxed()
                    .shared()
                })
                .clone()
        };
        let state = suspended.await.map_err(shared_to_box)?;
        self.inner.state.lock().turn_state = HarnessAgentTurnState::Suspended;
        Ok(state)
    }

    async fn finalize_current_turn_suspension(
        &self,
        session: Arc<dyn HarnessV1Session>,
    ) -> HarnessResult<HarnessV1ContinueTurnState> {
        let state = self.suspend_current_turn(session).await?;
        // Freeze ingress first, then include all dispatched host work in the
        // cursor. Keep this wait outside `suspend_current_turn` because
        // stop-condition handling invokes that helper from inside the active
        // prompt.
        let done = self.inner.state.lock().active_prompt_done.clone();
        if let Some(done) = done {
            done.await;
        }
        Ok(self.add_pending_tool_state(state))
    }

    async fn capture_stop_condition_boundary(
        &self,
        session: Arc<dyn HarnessV1Session>,
        turn_id: u64,
    ) -> HarnessResult<()> {
        {
            let state = self.inner.state.lock();
            if state.session_state != HarnessAgentSessionState::Active
                || state.active_turn_sequence != turn_id
            {
                return Ok(());
            }
        }
        self.suspend_current_turn(session).await.map(|_| ())
    }

    fn require_promptable_turn(&self) -> HarnessResult<()> {
        match self.inner.state.lock().turn_state {
            HarnessAgentTurnState::Idle => Ok(()),
            HarnessAgentTurnState::Running => Err(plain_error(format!(
                "Harness session {} already has a turn in progress.",
                self.inner.session_id
            ))),
            _ => Err(plain_error(format!(
                "Harness session {} has an unfinished turn and must be continued before accepting a new prompt.",
                self.inner.session_id
            ))),
        }
    }

    fn require_continuable_turn(&self) -> HarnessResult<()> {
        match self.inner.state.lock().turn_state {
            HarnessAgentTurnState::AwaitingApproval
            | HarnessAgentTurnState::AwaitingToolResult
            | HarnessAgentTurnState::Suspended => Ok(()),
            HarnessAgentTurnState::Running => Err(plain_error(format!(
                "Harness session {} already has a turn in progress.",
                self.inner.session_id
            ))),
            HarnessAgentTurnState::Idle => Err(plain_error(format!(
                "Harness session {} has no unfinished turn to continue.",
                self.inner.session_id
            ))),
        }
    }

    /// `markAwaitingApprovalIfActive` / `markAwaitingToolResultIfActive`.
    fn mark_if_active(&self, turn_state: HarnessAgentTurnState) {
        if self.inner.state.lock().session_state == HarnessAgentSessionState::Active {
            self.clear_active_prompt_control(None);
            self.inner.state.lock().turn_state = turn_state;
        }
    }

    fn start_tracked_turn(&self) -> u64 {
        self.clear_active_prompt_control(None);
        let mut state = self.inner.state.lock();
        state.turn_sequence += 1;
        let turn_id = state.turn_sequence;
        state.active_turn_sequence = turn_id;
        state.suspended_turn_state = None;
        state.turn_state = HarnessAgentTurnState::Running;
        let (resolve, receiver) = oneshot::channel();
        state.prompt_control_ids += 1;
        state.active_prompt_control = Some(ActivePromptControl {
            id: state.prompt_control_ids,
            turn_id,
            promise: receiver.map(|r| r.ok().flatten()).boxed().shared(),
            resolve: Some(resolve),
        });
        turn_id
    }

    fn set_prompt_control(&self, turn_id: u64, control: Arc<dyn HarnessV1PromptControl>) {
        {
            let state = self.inner.state.lock();
            if state.session_state != HarnessAgentSessionState::Active
                || state.turn_state != HarnessAgentTurnState::Running
                || state.active_turn_sequence != turn_id
                || state.active_prompt_control.as_ref().map(|c| c.turn_id) != Some(turn_id)
            {
                return;
            }
        }
        self.settle_active_prompt_control(Some(control));
    }

    fn wait_for_prompt_control(&self, turn_id: u64) -> BoxFuture<'static, ()> {
        let promise = {
            let state = self.inner.state.lock();
            match &state.active_prompt_control {
                Some(control) if control.turn_id == turn_id => Some(control.promise.clone()),
                _ => None,
            }
        };
        Box::pin(async move {
            if let Some(promise) = promise {
                promise.await;
            }
        })
    }

    fn settle_active_prompt_control(&self, control: Option<Arc<dyn HarnessV1PromptControl>>) {
        let resolve = {
            let mut state = self.inner.state.lock();
            state
                .active_prompt_control
                .as_mut()
                .and_then(|c| c.resolve.take())
        };
        if let Some(resolve) = resolve {
            let _ = resolve.send(control);
        }
    }

    fn clear_active_prompt_control(&self, turn_id: Option<u64>) {
        if let Some(turn_id) = turn_id {
            let state = self.inner.state.lock();
            if state
                .active_prompt_control
                .as_ref()
                .is_some_and(|c| c.turn_id != turn_id)
            {
                return;
            }
        }
        self.settle_active_prompt_control(None);
        self.inner.state.lock().active_prompt_control = None;
    }

    fn finish_tracked_turn(&self, turn_id: u64) {
        {
            let state = self.inner.state.lock();
            if state.session_state != HarnessAgentSessionState::Active
                || state.active_turn_sequence != turn_id
            {
                return;
            }
        }
        self.clear_active_prompt_control(Some(turn_id));
        let mut state = self.inner.state.lock();
        state.pending_tool_approvals.clear();
        state.pending_tool_results.clear();
        state.suspended_turn_state = None;
        state.active_turn_settings = None;
        state.persisted_turn_settings = None;
        state.turn_state = HarnessAgentTurnState::Idle;
    }

    fn resolve_active_turn_settings(
        &self,
        options: &HarnessAgentTurnOptions,
    ) -> HarnessResult<ActiveTurnSettings> {
        let mut state = self.inner.state.lock();
        if let Some(settings) = &state.active_turn_settings {
            return Ok(settings.clone());
        }
        let persisted =
            state
                .persisted_turn_settings
                .clone()
                .unwrap_or_else(|| HarnessV1TurnSettings {
                    model: options.model.clone(),
                    skills: options.skills.clone(),
                    instructions: options.instructions.clone(),
                    tools: options.tool_specs.clone(),
                });
        state.persisted_turn_settings = Some(persisted.clone());

        let mut active_tools = ToolSet::new();
        for tool_spec in &persisted.tools {
            let Some(tool) = options.active_tools.get(&tool_spec.name) else {
                return Err(plain_error(format!(
                    "HarnessAgent cannot continue turn because tool '{}' is no longer available.",
                    tool_spec.name
                )));
            };
            active_tools.insert(tool_spec.name.clone(), tool.clone());
        }

        let settings = ActiveTurnSettings {
            persisted,
            tools: options.tools.clone(),
            tools_context: self
                .inner
                .resumed_tools_context
                .clone()
                .unwrap_or_else(|| options.tools_context.clone()),
            runtime_context: self
                .inner
                .resumed_runtime_context
                .clone()
                .unwrap_or_else(|| options.runtime_context.clone()),
            active_tools,
            builtin_tool_filtering: options.builtin_tool_filtering.clone(),
        };
        state.active_turn_settings = Some(settings.clone());
        Ok(settings)
    }

    fn end_local_handle(&self, session_state: HarnessAgentSessionState) {
        self.clear_active_prompt_control(None);
        let mut state = self.inner.state.lock();
        state.session_state = session_state;
        state.underlying_session = None;
        state.sandbox_session = None;
    }

    fn require_reusable_session(&self) -> HarnessResult<Arc<dyn HarnessV1Session>> {
        let state = self.inner.state.lock();
        match (&state.session_state, &state.underlying_session) {
            (HarnessAgentSessionState::Active, Some(session)) => Ok(session.clone()),
            _ => Err(self.ended_error()),
        }
    }
}

fn to_resume_state_with_continuation(
    continue_from: HarnessV1ContinueTurnState,
) -> HarnessV1ResumeSessionState {
    HarnessV1ResumeSessionState {
        r#type: ResumeSessionType,
        harness_id: continue_from.harness_id.clone(),
        specification_version: continue_from.specification_version,
        data: continue_from.data.clone(),
        continue_from: Some(continue_from),
    }
}
