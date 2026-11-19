//! Port of `agent/harness-agent.ts`.

use std::sync::Arc;

use futures::future::{BoxFuture, join_all};
use indexmap::IndexMap;
use tokio_util::sync::CancellationToken;

use super::harness_agent_session::{
    HarnessAgentSession, HarnessAgentSessionOptions, HarnessAgentTurnOptions,
    HarnessAgentTurnResult, HarnessAgentTurnState,
};
use super::internal::bootstrap_recipe::{apply_bootstrap_recipe, hash_harness_bootstrap};
use super::internal::deferred_sandbox_setup::{defer_sandbox_session, defer_sandbox_setup};
use super::internal::harness_stream_text_result::HarnessStreamTextResult;
use super::internal::lifecycle_state_validation::{
    validate_continue_turn_state, validate_resume_session_state,
};
use super::internal::permission_mode::{
    permission_mode_needs_builtin_support, resolve_permission_mode,
};
use super::internal::resolve_observability::build_observability;
use super::internal::sandbox_bootstrap::{
    RunSandboxBootstrapOptions, ensure_sandbox_directory, resolve_session_work_dir,
    run_sandbox_bootstrap, validate_sandbox_bootstrap_settings,
};
use super::internal::tool_filtering::resolve_harness_agent_tool_filtering;
use super::internal::turn_telemetry::HarnessAgentLifecycleCallbacks;
use super::settings::{
    HarnessAgentCallPrompt, HarnessAgentInstructions, HarnessAgentPrepareCallOptions,
    HarnessAgentSandboxConfig, HarnessAgentSandboxCreator, HarnessAgentSettings, OnSessionOptions,
};
use super::tool_approval_continuation::collect_harness_agent_tool_approval_continuations;
use super::tool_result_continuation::collect_harness_agent_tool_result_continuations;
use crate::harness::errors::{BoxError, HarnessError, HarnessResult};
use crate::harness::provider::{JsonValue, TypeValidationContext, validate_types};
use crate::harness::sdk::content::{
    CallWarning, ContentPart, Context, FinishReason, LanguageModelRequestMetadata,
    LanguageModelResponseMetadata, ProviderMetadata, ReasoningOutput, StepResult, ToolsContext,
    TypedToolCall, TypedToolResult,
};
use crate::harness::sdk::error_value::{SharedError, SharedErrorBox, generate_id};
use crate::harness::sdk::messages::{ModelMessage, ToolApprovalResponse, ToolResultPart};
use crate::harness::sdk::stop_condition::StopCondition;
use crate::harness::sdk::telemetry::Callback;
use crate::harness::sdk::tool::ToolSet;
use crate::harness::sdk::usage::LanguageModelUsage;
use crate::harness::spec::HarnessV1SandboxTemplate;
use crate::harness::spec::harness::{
    HarnessV1, HarnessV1Prompt, HarnessV1SandboxSessionRef, HarnessV1StartOptions,
};
use crate::harness::spec::lifecycle_state::{
    HarnessV1ContinueTurnState, HarnessV1ResumeSessionState,
};
use crate::harness::spec::paths::harness_state_directory_path;
use crate::harness::spec::response_format::HarnessV1ResponseFormat;
use crate::harness::spec::tools::{
    HarnessV1BuiltinToolFiltering, HarnessV1PermissionMode, HarnessV1Skill, HarnessV1ToolSpec,
};
use crate::harness::utils::plain_error;
use crate::harness::utils::sandbox_home_dir::{
    get_restricted_sandbox_session, resolve_sandbox_default_working_directory,
    resolve_sandbox_home_dir,
};

use super::create_harness_sandbox_template::create_harness_sandbox_template;

const QUESTION_TOOL_NAME: &str = "askUserQuestions";

fn shared_to_box(error: SharedError) -> BoxError {
    Box::new(SharedErrorBox(error))
}

/// The sandbox `create_session` runs the harness in: an existing session,
/// or a [`HarnessAgentSandboxCreator`] that creates it when it is needed.
/// Either way, the caller retains ownership of the sandbox lifecycle.
#[derive(Clone)]
pub enum HarnessAgentSandboxSessionInput {
    Session(HarnessV1SandboxSessionRef),
    Creator(HarnessAgentSandboxCreator),
}

impl From<HarnessV1SandboxSessionRef> for HarnessAgentSandboxSessionInput {
    fn from(value: HarnessV1SandboxSessionRef) -> Self {
        Self::Session(value)
    }
}

impl From<HarnessAgentSandboxCreator> for HarnessAgentSandboxSessionInput {
    fn from(value: HarnessAgentSandboxCreator) -> Self {
        Self::Creator(value)
    }
}

/// Options of [`HarnessAgent::create_session`].
#[derive(Clone, Default)]
pub struct HarnessAgentCreateSessionOptions {
    /// Stable identifier for the underlying sandbox/session. Generated when
    /// omitted. Supply the original `session_id` with `resume_from` to
    /// reattach a previously ended session across processes.
    pub session_id: Option<String>,
    /// Resume payload returned by a prior `detach()` or `stop()`.
    pub resume_from: Option<HarnessV1ResumeSessionState>,
    /// Continuation payload returned by a prior `suspend_turn()`.
    pub continue_from: Option<HarnessV1ContinueTurnState>,
    /// Rebinds host-only tool context for an unfinished turn resumed with
    /// `continue_from` (directly or through `resume_from`).
    pub tools_context: Option<ToolsContext>,
    /// Rebinds host-only runtime context for an unfinished turn.
    pub runtime_context: Option<Context>,
    /// Existing sandbox session, or a creator for one.
    pub sandbox_session: Option<HarnessAgentSandboxSessionInput>,
    pub abort_signal: Option<CancellationToken>,
}

/// `AgentCallParameters & HarnessAgentCallExtensions` (and
/// `AgentStreamParameters`).
///
/// [rust] The harness ignores `timeout`, `experimental_sandbox`, and
/// `experimental_transform`, so they are not mirrored; the deprecated
/// callback aliases (`experimental_onStart`, `onStepFinish`, `onFinish`, …)
/// are not ported.
#[derive(Clone)]
pub struct HarnessAgentCallParameters {
    /// Active session returned by `create_session`.
    pub session: HarnessAgentSession,
    pub prompt: Option<HarnessAgentCallPrompt>,
    pub messages: Option<Vec<ModelMessage>>,
    /// Custom call options, validated against `call_options_schema`.
    pub options: Option<JsonValue>,
    pub abort_signal: Option<CancellationToken>,
    pub on_start: Option<Callback<crate::harness::sdk::telemetry::GenerateTextStartEvent>>,
    pub on_step_start: Option<Callback<crate::harness::sdk::telemetry::GenerateTextStepStartEvent>>,
    pub on_tool_execution_start:
        Option<Callback<crate::harness::sdk::telemetry::ToolExecutionStartEvent>>,
    pub on_tool_execution_end:
        Option<Callback<crate::harness::sdk::telemetry::ToolExecutionEndEvent>>,
    pub on_step_end: Option<Callback<StepResult>>,
    pub on_end: Option<Callback<crate::harness::sdk::telemetry::GenerateTextEndEvent>>,
}

impl HarnessAgentCallParameters {
    /// Call parameters with only `session` set.
    pub fn new(session: HarnessAgentSession) -> Self {
        Self {
            session,
            prompt: None,
            messages: None,
            options: None,
            abort_signal: None,
            on_start: None,
            on_step_start: None,
            on_tool_execution_start: None,
            on_tool_execution_end: None,
            on_step_end: None,
            on_end: None,
        }
    }

    /// Set `prompt`.
    pub fn with_prompt(mut self, prompt: impl Into<HarnessAgentCallPrompt>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set `messages`.
    pub fn with_messages(mut self, messages: Vec<ModelMessage>) -> Self {
        self.messages = Some(messages);
        self
    }
}

/// Options of `continue_generate` / `continue_stream`.
#[derive(Clone)]
pub struct HarnessAgentContinueParameters {
    pub session: HarnessAgentSession,
    pub tool_approval_continuations: Vec<ToolApprovalResponse>,
    pub tool_result_continuations: Vec<ToolResultPart>,
    pub abort_signal: Option<CancellationToken>,
}

impl HarnessAgentContinueParameters {
    pub fn new(session: HarnessAgentSession) -> Self {
        Self {
            session,
            tool_approval_continuations: Vec::new(),
            tool_result_continuations: Vec::new(),
            abort_signal: None,
        }
    }
}

/// `HarnessAgentContinueTurnInput`.
struct ContinueTurnInput {
    tool_approval_continuations: Vec<ToolApprovalResponse>,
    tool_result_continuations: Vec<ToolResultPart>,
}

/// `PreparedHarnessAgentTurnSettings`.
struct PreparedTurnSettings {
    model: Option<String>,
    skills: Vec<HarnessV1Skill>,
    instructions: Option<String>,
    tools: ToolSet,
    tools_context: ToolsContext,
    active_tools: ToolSet,
    tool_specs: Vec<HarnessV1ToolSpec>,
    builtin_tool_filtering: Option<HarnessV1BuiltinToolFiltering>,
}

/// AI SDK `Agent` implementation that drives a third-party agent runtime
/// through a harness adapter (Claude Code, Codex, …).
///
/// - **Stateless definition.** The agent holds the harness adapter, the
///   merged tool surface and other config, never a live session.
/// - **Explicit sessions.** Spawn sessions with `create_session`, pass the
///   returned session on every `generate` / `stream`, and end it with
///   `detach()`, `stop()`, or `destroy()`.
/// - **Cross-process resume.** `create_session` with `resume_from` resumes
///   from state returned by `detach()` / `stop()`; with `continue_from`, from
///   state returned by `suspend_turn()`.
/// - **Host tool execution.** User tools run on the host whenever the
///   runtime calls them; the result is submitted back to the harness.
/// - **Sandbox propagation.** The caller-provided sandbox session is passed
///   into `doStart`, and a tool-safe view of it to user tools. It remains
///   owned by the caller.
pub struct HarnessAgent {
    /// `version`.
    pub version: &'static str,
    /// `id`.
    pub id: Option<String>,
    /// Merged tool set: harness built-ins plus user tools, user tools
    /// overriding built-ins on key collision.
    pub tools: ToolSet,
    settings: HarnessAgentSettings,
    stop_conditions: Vec<StopCondition>,
    sandbox_config: HarnessAgentSandboxConfig,
    builtin_tool_filtering: Option<HarnessV1BuiltinToolFiltering>,
    permission_mode: HarnessV1PermissionMode,
    headers: Option<IndexMap<String, String>>,
}

/// `normalizeHeaders`: lowercase names, drop `undefined` values.
fn normalize_headers(
    headers: Option<&IndexMap<String, Option<String>>>,
) -> IndexMap<String, String> {
    headers
        .into_iter()
        .flatten()
        .filter_map(|(name, value)| value.as_ref().map(|v| (name.to_lowercase(), v.clone())))
        .collect()
}

/// The merged tool set of a harness and user tools.
fn merge_tools(harness: &dyn HarnessV1, user_tools: &ToolSet) -> ToolSet {
    let mut tools: ToolSet = harness
        .builtin_tools()
        .iter()
        .map(|(name, builtin)| (name.clone(), builtin.tool.clone()))
        .collect();
    for (name, tool) in user_tools {
        tools.insert(name.clone(), tool.clone());
    }
    tools
}

fn assert_no_reserved_question_tool(
    harness: &dyn HarnessV1,
    user_tools: &ToolSet,
) -> HarnessResult<()> {
    if harness.builtin_tools().contains_key(QUESTION_TOOL_NAME)
        && user_tools.contains_key(QUESTION_TOOL_NAME)
    {
        return Err(plain_error(format!(
            "HarnessAgent tool name '{QUESTION_TOOL_NAME}' is reserved for harness question requests."
        )));
    }
    Ok(())
}

/// `mergeCallbacks`: runs every callback, concurrently.
fn merge_callbacks<E: Clone + Send + 'static>(
    callbacks: impl IntoIterator<Item = Option<Callback<E>>>,
) -> Option<Callback<E>> {
    let callbacks: Vec<Callback<E>> = callbacks.into_iter().flatten().collect();
    if callbacks.is_empty() {
        return None;
    }
    Some(Arc::new(move |event: E| {
        let futures: Vec<BoxFuture<'static, ()>> = callbacks
            .iter()
            .map(|callback| callback(event.clone()))
            .collect();
        Box::pin(async move {
            join_all(futures).await;
        })
    }))
}

impl HarnessAgent {
    pub fn new(settings: HarnessAgentSettings) -> HarnessResult<Self> {
        let sandbox_config = settings.sandbox_config.clone().unwrap_or_default();
        validate_sandbox_bootstrap_settings(&sandbox_config)?;
        let stop_conditions = settings.stop_when.clone().unwrap_or_default();
        const FORBIDDEN_HEADERS: [&str; 4] =
            ["authorization", "x-api-key", "user-agent", "x-client-app"];
        if let Some(forbidden) = settings
            .headers
            .iter()
            .flat_map(|h| h.keys())
            .map(|name| name.to_lowercase())
            .find(|name| FORBIDDEN_HEADERS.contains(&name.as_str()))
        {
            return Err(plain_error(format!(
                "HarnessAgent: `headers` must not include the managed header `{forbidden}`."
            )));
        }
        let headers = normalize_headers(settings.headers.as_ref());
        let headers = (!headers.is_empty()).then_some(headers);
        let user_tools = settings.tools.clone().unwrap_or_default();
        assert_no_reserved_question_tool(&*settings.harness, &user_tools)?;
        let permission_mode = resolve_permission_mode(settings.permission_mode);
        let tools = merge_tools(&*settings.harness, &user_tools);
        let all_tool_names: Vec<String> = tools.keys().cloned().collect();
        let tool_filtering = resolve_harness_agent_tool_filtering(
            &*settings.harness,
            &user_tools,
            &all_tool_names,
            settings.active_tools.as_deref(),
            settings.inactive_tools.as_deref(),
        )?;
        if !settings.harness.builtin_tools().is_empty()
            && permission_mode_needs_builtin_support(permission_mode)
            && !settings.harness.supports_builtin_tool_approvals()
        {
            let harness_id = settings.harness.harness_id().to_string();
            return Err(HarnessError::capability_unsupported(
                format!(
                    "Harness '{harness_id}' does not support built-in tool approval requests; use permissionMode: 'allow-all'."
                ),
                Some(harness_id),
            )
            .into());
        }
        Ok(Self {
            version: "agent-v1",
            id: settings.id.clone(),
            tools,
            stop_conditions,
            sandbox_config,
            builtin_tool_filtering: tool_filtering.builtin_tool_filtering,
            permission_mode,
            headers,
            settings,
        })
    }

    /// Identifier of the harness backing this agent.
    pub fn harness_id(&self) -> &str {
        self.settings.harness.harness_id()
    }

    /// Whether this agent parses completed turns with its configured output.
    pub fn has_output(&self) -> bool {
        self.settings.output.is_some()
    }

    pub async fn get_sandbox_template(&self) -> HarnessResult<Option<HarnessV1SandboxTemplate>> {
        create_harness_sandbox_template(
            std::slice::from_ref(&self.settings.harness),
            Some(&self.sandbox_config),
        )
        .await
    }

    /// Start a fresh session, or resume from state previously returned by
    /// `detach()` or `stop()`. Pass the returned session to subsequent
    /// `generate` / `stream` calls.
    pub async fn create_session(
        &self,
        options: HarnessAgentCreateSessionOptions,
    ) -> HarnessResult<HarnessAgentSession> {
        let session_id = options.session_id.clone().unwrap_or_else(generate_id);
        let abort_signal = options.abort_signal.clone();
        let harness = self.settings.harness.clone();

        if options.resume_from.is_some() && options.continue_from.is_some() {
            return Err(plain_error(
                "HarnessAgent.createSession: pass either `resumeFrom` or `continueFrom`, not both.",
            ));
        }
        let validated_resume_from = options
            .resume_from
            .as_ref()
            .map(|state| validate_resume_session_state(&*harness, state))
            .transpose()?;
        let validated_continue_from = options
            .continue_from
            .as_ref()
            .map(|state| validate_continue_turn_state(&*harness, state))
            .transpose()?;
        let effective_continue_from = validated_continue_from.or_else(|| {
            validated_resume_from
                .as_ref()
                .and_then(|r| r.continue_from.clone())
        });
        if options.tools_context.is_some() && effective_continue_from.is_none() {
            return Err(plain_error(
                "HarnessAgent.createSession: `toolsContext` can only rebind an unfinished turn from `continueFrom` or `resumeFrom`.",
            ));
        }

        // [rust] The deprecated `sandbox` provider is not ported, so the
        // caller always provides the sandbox and keeps owning it.
        let Some(provided_sandbox_session) = options.sandbox_session.clone() else {
            return Err(plain_error(
                "HarnessAgent.createSession: pass `sandboxSession` to createSession().",
            ));
        };

        let default_working_directory = match &provided_sandbox_session {
            HarnessAgentSandboxSessionInput::Creator(creator) => {
                creator.default_working_directory.clone()
            }
            HarnessAgentSandboxSessionInput::Session(session) => {
                resolve_sandbox_default_working_directory(session, abort_signal.clone()).await?
            }
        };
        let session_work_dir = resolve_session_work_dir(
            &default_working_directory,
            harness.harness_id(),
            &session_id,
            self.sandbox_config.work_dir.as_deref(),
        );

        let sandbox_config = self.sandbox_config.clone();
        let setup_work_dir = session_work_dir.clone();
        let setup_sandbox_session = Arc::new(
            move |session: HarnessV1SandboxSessionRef,
                  abort_signal: Option<CancellationToken>|
                  -> BoxFuture<'static, HarnessResult<()>> {
                let (config, work_dir) = (sandbox_config.clone(), setup_work_dir.clone());
                Box::pin(async move {
                    let restricted = get_restricted_sandbox_session(&session);
                    let mut bootstrap = RunSandboxBootstrapOptions::new(restricted.clone());
                    bootstrap.work_dir = config.work_dir.clone();
                    bootstrap.on_bootstrap = config.on_bootstrap.clone();
                    bootstrap.bootstrap_hash = config.bootstrap_hash.clone();
                    bootstrap.skip_on_bootstrap_if_marked = true;
                    bootstrap.abort_signal = abort_signal.clone();
                    run_sandbox_bootstrap(bootstrap).await?;
                    ensure_sandbox_directory(&*restricted, &work_dir, abort_signal.clone()).await?;
                    if let Some(on_session) = &config.on_session {
                        on_session(OnSessionOptions {
                            session: restricted,
                            session_work_dir: work_dir,
                            abort_signal,
                        })
                        .await?;
                    }
                    Ok(())
                })
            },
        );

        let recipe = if harness.supports_get_bootstrap() {
            Some(harness.get_bootstrap(abort_signal.clone()).await?)
        } else {
            None
        };
        let mut adapter_sandbox_session: Option<HarnessV1SandboxSessionRef> = None;
        let sandbox_session = match provided_sandbox_session {
            HarnessAgentSandboxSessionInput::Session(session) => session,
            HarnessAgentSandboxSessionInput::Creator(creator) if recipe.is_none() => {
                // The lazy session is a plain sandbox session without the
                // network methods, because they cannot be answered before the
                // sandbox exists. Only bridge-backed harnesses that run inside
                // the sandbox need them, and those declare a bootstrap recipe.
                let setup = setup_sandbox_session.clone();
                let deferred = HarnessV1SandboxSessionRef::Basic(defer_sandbox_session(
                    creator.default_working_directory.clone(),
                    creator.create_sandbox_session.clone(),
                    Arc::new(move |session| setup(session, None)),
                ));
                adapter_sandbox_session = Some(deferred.clone());
                deferred
            }
            HarnessAgentSandboxSessionInput::Creator(creator) => {
                (creator.create_sandbox_session)(abort_signal.clone()).await?
            }
        };
        let tool_safe_sandbox_session = get_restricted_sandbox_session(&sandbox_session);
        if let Some(recipe) = &recipe {
            let recipe_identity = hash_harness_bootstrap(recipe);
            // Harness infrastructure always lives under the sandbox's own
            // HOME, never the session's working directory.
            let sandbox_home_dir =
                resolve_sandbox_home_dir(&*tool_safe_sandbox_session, abort_signal.clone()).await?;
            apply_bootstrap_recipe(
                &*tool_safe_sandbox_session,
                recipe,
                &recipe_identity,
                &harness_state_directory_path(&sandbox_home_dir),
                abort_signal.clone(),
            )
            .await?;
            setup_sandbox_session(sandbox_session.clone(), abort_signal.clone()).await?;
        }
        // Without a recipe, an existing sandbox still defers its setup to the
        // first sandbox operation.
        let adapter_sandbox_session = match adapter_sandbox_session {
            Some(session) => session,
            None if recipe.is_none() => {
                let (setup, session) = (setup_sandbox_session.clone(), sandbox_session.clone());
                defer_sandbox_setup(
                    sandbox_session.clone(),
                    Arc::new(move || setup(session.clone(), None)),
                )
            }
            None => sandbox_session.clone(),
        };

        let underlying_session = harness
            .do_start(HarnessV1StartOptions {
                headers: self.headers.clone(),
                session_id: session_id.clone(),
                resume_from: validated_resume_from,
                continue_from: effective_continue_from.clone(),
                permission_mode: Some(self.permission_mode),
                builtin_tool_filtering: self.builtin_tool_filtering.clone(),
                abort_signal,
                observability: build_observability(
                    self.settings.debug.as_ref(),
                    self.settings.on_log.clone(),
                    self.settings.telemetry.as_ref(),
                ),
                sandbox_session: adapter_sandbox_session.clone(),
                session_work_dir: session_work_dir.clone(),
            })
            .await?;

        let turn_state = match &effective_continue_from {
            None => HarnessAgentTurnState::Idle,
            Some(c)
                if c.pending_tool_approvals
                    .as_ref()
                    .is_some_and(|a| !a.is_empty()) =>
            {
                HarnessAgentTurnState::AwaitingApproval
            }
            Some(c)
                if c.pending_tool_results
                    .as_ref()
                    .is_some_and(|r| !r.is_empty()) =>
            {
                HarnessAgentTurnState::AwaitingToolResult
            }
            Some(_) => HarnessAgentTurnState::Suspended,
        };
        let mut session_options = HarnessAgentSessionOptions::new(
            session_id,
            harness,
            Arc::from(underlying_session),
            adapter_sandbox_session,
            session_work_dir,
        );
        session_options.owns_sandbox_lifecycle = Some(false);
        session_options.tool_approval = self.settings.tool_approval.clone();
        if let Some(continue_from) = effective_continue_from {
            session_options.pending_tool_approvals =
                continue_from.pending_tool_approvals.unwrap_or_default();
            session_options.pending_tool_results =
                continue_from.pending_tool_results.unwrap_or_default();
            session_options.turn_settings = continue_from.turn_settings;
        }
        session_options.resumed_tools_context = options.tools_context;
        session_options.resumed_runtime_context = options.runtime_context;
        session_options.turn_state = Some(turn_state);
        Ok(HarnessAgentSession::new(session_options))
    }

    /// Run a turn to completion.
    pub async fn generate(
        &self,
        options: HarnessAgentCallParameters,
    ) -> HarnessResult<HarnessGenerateTextResult> {
        let turn = match self
            .resolve_continue_turn_input(options.prompt.as_ref(), options.messages.as_ref())?
        {
            None => self.start_prompt_turn(options).await?,
            Some(turn_input) => {
                self.start_continue_turn(&options.session, turn_input, options.abort_signal.clone())
                    .await?
            }
        };
        turn.done.await;
        self.to_generate_result(&turn.result).await
    }

    /// Run a turn, returning its streaming result once it is ready.
    pub async fn stream(
        &self,
        options: HarnessAgentCallParameters,
    ) -> HarnessResult<HarnessStreamTextResult> {
        let turn = match self
            .resolve_continue_turn_input(options.prompt.as_ref(), options.messages.as_ref())?
        {
            None => self.start_prompt_turn(options).await?,
            Some(turn_input) => {
                self.start_continue_turn(&options.session, turn_input, options.abort_signal.clone())
                    .await?
            }
        };
        turn.ready.await;
        Ok(turn.result)
    }

    /// Continue the in-flight turn without a new prompt, draining it like
    /// [`generate`](Self::generate).
    pub async fn continue_generate(
        &self,
        options: HarnessAgentContinueParameters,
    ) -> HarnessResult<HarnessGenerateTextResult> {
        let turn = self
            .start_continue_turn(
                &options.session,
                ContinueTurnInput {
                    tool_approval_continuations: options.tool_approval_continuations,
                    tool_result_continuations: options.tool_result_continuations,
                },
                options.abort_signal,
            )
            .await?;
        turn.done.await;
        self.to_generate_result(&turn.result).await
    }

    /// Continue the in-flight turn without a new prompt, streaming its
    /// events like [`stream`](Self::stream).
    pub async fn continue_stream(
        &self,
        options: HarnessAgentContinueParameters,
    ) -> HarnessResult<HarnessStreamTextResult> {
        let turn = self
            .start_continue_turn(
                &options.session,
                ContinueTurnInput {
                    tool_approval_continuations: options.tool_approval_continuations,
                    tool_result_continuations: options.tool_result_continuations,
                },
                options.abort_signal,
            )
            .await?;
        turn.ready.await;
        Ok(turn.result)
    }

    /// Submit another user message to a currently running session turn.
    pub async fn experimental_steer(
        &self,
        session: &HarnessAgentSession,
        text: impl Into<String>,
    ) -> HarnessResult<()> {
        session.experimental_steer_turn(text.into()).await
    }

    // ─── Internals ──────────────────────────────────────────────────────

    async fn start_prompt_turn(
        &self,
        options: HarnessAgentCallParameters,
    ) -> HarnessResult<HarnessAgentTurnResult> {
        let callbacks = self.resolve_lifecycle_callbacks(Some(&options));
        let session = options.session.clone();
        let abort_signal = options.abort_signal.clone();
        let (prompt, runtime_context, settings) = self.prepare_prompt_turn_input(options).await?;
        let response_format = self.resolve_response_format().await?;
        session.prompt_turn(
            prompt,
            self.build_turn_options(
                settings,
                runtime_context,
                abort_signal,
                response_format,
                callbacks,
            ),
        )
    }

    async fn start_continue_turn(
        &self,
        session: &HarnessAgentSession,
        turn_input: ContinueTurnInput,
        abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<HarnessAgentTurnResult> {
        let runtime_context = self.settings.runtime_context.clone().unwrap_or_default();
        let settings = self.prepare_turn_settings(
            self.settings.model.clone(),
            self.settings.skills.clone(),
            self.settings.instructions.clone(),
            self.settings.tools.clone(),
            self.settings.tools_context.clone(),
        )?;
        let response_format = self.resolve_response_format().await?;
        session.continue_turn(
            self.build_turn_options(
                settings,
                runtime_context,
                abort_signal,
                response_format,
                self.resolve_lifecycle_callbacks(None),
            ),
            turn_input.tool_approval_continuations,
            turn_input.tool_result_continuations,
        )
    }

    fn build_turn_options(
        &self,
        settings: PreparedTurnSettings,
        runtime_context: Context,
        abort_signal: Option<CancellationToken>,
        response_format: Option<HarnessV1ResponseFormat>,
        callbacks: HarnessAgentLifecycleCallbacks,
    ) -> HarnessAgentTurnOptions {
        HarnessAgentTurnOptions {
            model: settings.model,
            skills: settings.skills,
            instructions: settings.instructions,
            tools: settings.tools,
            tools_context: settings.tools_context,
            active_tools: settings.active_tools,
            tool_specs: settings.tool_specs,
            builtin_tool_filtering: settings.builtin_tool_filtering,
            runtime_context,
            abort_signal,
            response_format,
            output: self.settings.output.clone(),
            telemetry: self.settings.telemetry.clone(),
            callbacks,
            stop_conditions: self.stop_conditions.clone(),
        }
    }

    fn resolve_lifecycle_callbacks(
        &self,
        call: Option<&HarnessAgentCallParameters>,
    ) -> HarnessAgentLifecycleCallbacks {
        let s = &self.settings;
        HarnessAgentLifecycleCallbacks {
            on_start: merge_callbacks([s.on_start.clone(), call.and_then(|c| c.on_start.clone())]),
            on_step_start: merge_callbacks([
                s.on_step_start.clone(),
                call.and_then(|c| c.on_step_start.clone()),
            ]),
            on_language_model_call_start: s.on_language_model_call_start.clone(),
            on_language_model_call_end: s.on_language_model_call_end.clone(),
            on_tool_execution_start: merge_callbacks([
                s.on_tool_execution_start.clone(),
                call.and_then(|c| c.on_tool_execution_start.clone()),
            ]),
            on_tool_execution_end: merge_callbacks([
                s.on_tool_execution_end.clone(),
                call.and_then(|c| c.on_tool_execution_end.clone()),
            ]),
            on_step_end: merge_callbacks([
                s.on_step_end.clone(),
                call.and_then(|c| c.on_step_end.clone()),
            ]),
            on_end: merge_callbacks([s.on_end.clone(), call.and_then(|c| c.on_end.clone())]),
        }
    }

    fn resolve_continue_turn_input(
        &self,
        prompt: Option<&HarnessAgentCallPrompt>,
        messages: Option<&Vec<ModelMessage>>,
    ) -> HarnessResult<Option<ContinueTurnInput>> {
        let messages = match prompt {
            Some(HarnessAgentCallPrompt::Text(_)) => return Ok(None),
            Some(HarnessAgentCallPrompt::Messages(messages)) => Some(messages),
            None => messages,
        };
        let Some(messages) = messages else {
            return Ok(None);
        };
        let tool_approval_continuations =
            collect_harness_agent_tool_approval_continuations(messages)?;
        let tool_result_continuations = collect_harness_agent_tool_result_continuations(messages);
        if tool_approval_continuations.is_empty() && tool_result_continuations.is_empty() {
            return Ok(None);
        }
        Ok(Some(ContinueTurnInput {
            tool_approval_continuations,
            tool_result_continuations,
        }))
    }

    /// Reduce the call input to the single user message the harness should
    /// run for this turn. The harness session owns prior-turn state, so it is
    /// never replayed: a string is forwarded as-is and a message array is
    /// collapsed to its last user message.
    fn resolve_prompt_turn_input(
        &self,
        prompt: Option<&HarnessAgentCallPrompt>,
        messages: Option<&Vec<ModelMessage>>,
    ) -> HarnessResult<HarnessV1Prompt> {
        let messages = match prompt {
            Some(HarnessAgentCallPrompt::Text(text)) => {
                return Ok(HarnessV1Prompt::Text(text.clone()));
            }
            Some(HarnessAgentCallPrompt::Messages(messages)) => Some(messages),
            None => messages,
        };
        let Some(messages) = messages else {
            return Err(plain_error(
                "HarnessAgent: either `prompt` or `messages` is required.",
            ));
        };
        messages
            .iter()
            .rev()
            .find_map(|message| match message {
                ModelMessage::User(user) => Some(HarnessV1Prompt::Message(user.clone())),
                _ => None,
            })
            .ok_or_else(|| {
                plain_error(
                    "HarnessAgent: messages must contain at least one `role: \"user\"` entry.",
                )
            })
    }

    async fn prepare_prompt_turn_input(
        &self,
        options: HarnessAgentCallParameters,
    ) -> HarnessResult<(HarnessV1Prompt, Context, PreparedTurnSettings)> {
        let mut call_options = options.options;
        if let (Some(schema), Some(value)) = (&self.settings.call_options_schema, &call_options) {
            call_options = Some(validate_types(
                value,
                schema,
                Some(TypeValidationContext {
                    field: Some("options".into()),
                    entity_name: None,
                    entity_id: None,
                }),
            )?);
        }

        let base_call_args = HarnessAgentPrepareCallOptions {
            options: call_options,
            prompt: options.prompt,
            messages: options.messages,
            model: self.settings.model.clone(),
            skills: self.settings.skills.clone(),
            instructions: self.settings.instructions.clone(),
            tools: self.settings.tools.clone(),
            runtime_context: self.settings.runtime_context.clone(),
            tools_context: self.settings.tools_context.clone().unwrap_or_default(),
        };
        let prepared = match &self.settings.prepare_call {
            Some(prepare_call) => prepare_call(base_call_args).await?,
            None => base_call_args,
        };
        if self
            .resolve_continue_turn_input(prepared.prompt.as_ref(), prepared.messages.as_ref())?
            .is_some()
        {
            return Err(plain_error(
                "HarnessAgent.prepareCall must return a fresh user prompt, not a tool continuation.",
            ));
        }

        let prompt =
            self.resolve_prompt_turn_input(prepared.prompt.as_ref(), prepared.messages.as_ref())?;
        let settings = self.prepare_turn_settings(
            prepared.model,
            prepared.skills,
            prepared.instructions,
            prepared.tools,
            Some(prepared.tools_context),
        )?;
        Ok((
            prompt,
            prepared.runtime_context.unwrap_or_default(),
            settings,
        ))
    }

    fn prepare_turn_settings(
        &self,
        model: Option<String>,
        skills: Option<Vec<HarnessV1Skill>>,
        instructions: Option<HarnessAgentInstructions>,
        tools: Option<ToolSet>,
        tools_context: Option<ToolsContext>,
    ) -> HarnessResult<PreparedTurnSettings> {
        let harness = &*self.settings.harness;
        let user_tools = tools.unwrap_or_default();
        assert_no_reserved_question_tool(harness, &user_tools)?;
        let tools = merge_tools(harness, &user_tools);
        let all_tool_names: Vec<String> = tools.keys().cloned().collect();
        let tool_filtering = resolve_harness_agent_tool_filtering(
            harness,
            &user_tools,
            &all_tool_names,
            self.settings.active_tools.as_deref(),
            self.settings.inactive_tools.as_deref(),
        )?;
        Ok(PreparedTurnSettings {
            model,
            skills: skills.unwrap_or_default(),
            instructions: instructions.map(|i| match i {
                HarnessAgentInstructions::Text(text) => text,
                HarnessAgentInstructions::System(message) => message.content,
            }),
            tools,
            tools_context: tools_context.unwrap_or_default(),
            tool_specs: to_tool_specs(&tool_filtering.active_user_tools),
            active_tools: tool_filtering.active_user_tools,
            builtin_tool_filtering: tool_filtering.builtin_tool_filtering,
        })
    }

    async fn to_generate_result(
        &self,
        result: &HarnessStreamTextResult,
    ) -> HarnessResult<HarnessGenerateTextResult> {
        // The stream is already drained (`done` has settled). `steps` is the
        // single source of truth the result derives everything else from.
        let (steps, usage, response_messages, output) = futures::join!(
            result.steps(),
            result.usage(),
            result.response_messages(),
            async {
                match self.settings.output {
                    None => Ok(None),
                    Some(_) => result.output().await.map(Some),
                }
            }
        );
        Ok(HarnessGenerateTextResult {
            steps: steps.map_err(shared_to_box)?,
            usage: usage.map_err(shared_to_box)?,
            response_messages: response_messages.map_err(shared_to_box)?,
            output: output.map_err(shared_to_box)?,
        })
    }

    async fn resolve_response_format(&self) -> HarnessResult<Option<HarnessV1ResponseFormat>> {
        match &self.settings.output {
            None => Ok(None),
            Some(output) => output.response_format().await.map(Some),
        }
    }
}

/// Wire-format projection of user-defined tools only. Harness built-ins are
/// executed by the runtime and are never re-declared over the wire.
fn to_tool_specs(active_user_tools: &ToolSet) -> Vec<HarnessV1ToolSpec> {
    active_user_tools
        .iter()
        .map(|(name, tool)| HarnessV1ToolSpec {
            name: name.clone(),
            description: tool.description.clone(),
            input_schema: Some(tool.input_schema.json_schema.clone()),
        })
        .collect()
}

/// `GenerateTextResult` view over a drained turn. Every member derives from
/// `steps`.
#[derive(Debug, Clone)]
pub struct HarnessGenerateTextResult {
    pub steps: Vec<StepResult>,
    pub usage: LanguageModelUsage,
    pub response_messages: Vec<ModelMessage>,
    /// The parsed output, when the agent has one.
    pub output: Option<JsonValue>,
}

impl HarnessGenerateTextResult {
    /// `finalStep`.
    ///
    /// # Panics
    /// When there are no steps (TS returns `undefined` and its accessors
    /// throw).
    pub fn final_step(&self) -> &StepResult {
        self.steps
            .last()
            .expect("a generate result has at least one step")
    }

    pub fn content(&self) -> Vec<ContentPart> {
        self.steps.iter().flat_map(|s| s.content.clone()).collect()
    }

    pub fn text(&self) -> String {
        self.final_step().text()
    }

    /// `files` ([rust] the harness never produces files: always empty).
    pub fn files(&self) -> Vec<JsonValue> {
        Vec::new()
    }

    /// `sources` ([rust] the harness never produces sources: always empty).
    pub fn sources(&self) -> Vec<JsonValue> {
        Vec::new()
    }

    pub fn tool_calls(&self) -> Vec<TypedToolCall> {
        self.steps.iter().flat_map(|s| s.tool_calls()).collect()
    }

    pub fn static_tool_calls(&self) -> Vec<TypedToolCall> {
        self.steps
            .iter()
            .flat_map(|s| s.static_tool_calls())
            .collect()
    }

    pub fn dynamic_tool_calls(&self) -> Vec<TypedToolCall> {
        self.steps
            .iter()
            .flat_map(|s| s.dynamic_tool_calls())
            .collect()
    }

    pub fn tool_results(&self) -> Vec<TypedToolResult> {
        self.steps.iter().flat_map(|s| s.tool_results()).collect()
    }

    pub fn static_tool_results(&self) -> Vec<TypedToolResult> {
        self.steps
            .iter()
            .flat_map(|s| s.static_tool_results())
            .collect()
    }

    pub fn dynamic_tool_results(&self) -> Vec<TypedToolResult> {
        self.steps
            .iter()
            .flat_map(|s| s.dynamic_tool_results())
            .collect()
    }

    pub fn finish_reason(&self) -> FinishReason {
        self.final_step().finish_reason
    }

    pub fn raw_finish_reason(&self) -> Option<String> {
        self.final_step().raw_finish_reason.clone()
    }

    pub fn warnings(&self) -> Vec<CallWarning> {
        self.steps
            .iter()
            .flat_map(|s| s.warnings.clone().unwrap_or_default())
            .collect()
    }

    /// The reasoning content of the final step.
    pub fn reasoning(&self) -> Vec<ReasoningOutput> {
        self.final_step()
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Reasoning(r) => Some(r.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn reasoning_text(&self) -> Option<String> {
        self.final_step().reasoning_text()
    }

    pub fn total_usage(&self) -> LanguageModelUsage {
        self.usage.clone()
    }

    pub fn request(&self) -> LanguageModelRequestMetadata {
        self.final_step().request.clone()
    }

    pub fn response(&self) -> LanguageModelResponseMetadata {
        self.final_step().response.clone()
    }

    pub fn provider_metadata(&self) -> Option<ProviderMetadata> {
        self.final_step().provider_metadata.clone()
    }
}

#[cfg(test)]
#[path = "harness_agent_tests.rs"]
mod tests;
