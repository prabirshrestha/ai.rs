//! [`HarnessAgent`]: a harness with settings that every turn of its sessions
//! uses, as the AI SDK's `HarnessAgent` (`harness-agent.ts`,
//! `harness-agent-settings.ts`, `internal/tool-filtering.ts`).
//!
//! The model, thinking level, instructions, skills and tools are set once on
//! the agent and sent with each turn; [`HarnessAgentSettings::prepare_call`]
//! can change them for one call. The permission mode, tool filtering and
//! headers apply to every session the agent creates. Sessions are explicit:
//! create one with [`HarnessAgent::create_session`], pass it to each
//! [`HarnessAgent::prompt`], and end it with its own `detach`, `stop` or
//! `destroy`.
//!
//! [rust] The AI SDK result types (`GenerateTextResult`, `StreamTextResult`,
//! steps, `stopWhen`, output parsing and the lifecycle callbacks) are not
//! ported: a turn is a [`HarnessTurn`] of ai.rs events and messages, and its
//! structured output is [`HarnessAgentSettings::response_format`]. The
//! per-tool `toolApproval` map is [`HarnessAgentSettings::before_tool_call`],
//! and `toolsContext` / `runtimeContext` are the closures' own captures.

use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::harness::error::HarnessError;
use crate::harness::session::{
    Harness, HarnessResumeState, HarnessSandbox, HarnessSession, HarnessStartOptions, HarnessTurn,
    HarnessTurnOptions, HarnessTurnSettings, ResponseFormat,
};
use crate::harness::tools::{BuiltinToolFilter, HarnessSkill, PermissionMode};
use crate::{
    AfterToolCallFn, AgentMessage, AgentResult, BeforeToolCallFn, DynAgentTool, ThinkingLevel,
};

/// The harness tool name reserved for question requests.
const QUESTION_TOOL: &str = "askUserQuestions";

/// Headers the harness manages itself.
const MANAGED_HEADERS: [&str; 4] = ["authorization", "x-api-key", "user-agent", "x-client-app"];

/// What one call runs: its prompt and the settings of its turn. This is what
/// [`HarnessAgentSettings::prepare_call`] receives, filled from the agent's
/// settings, and returns.
#[derive(Clone)]
pub struct HarnessCall {
    pub prompt: AgentMessage,
    /// The caller's own call options ([`HarnessAgentCallOptions::options`]).
    pub options: Option<Value>,
    pub model: Option<String>,
    pub thinking_level: Option<ThinkingLevel>,
    pub instructions: Option<String>,
    pub skills: Vec<HarnessSkill>,
    pub tools: Vec<DynAgentTool>,
}

impl fmt::Debug for HarnessCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessCall")
            .field("options", &self.options)
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("instructions", &self.instructions)
            .field("skills", &self.skills)
            .field("tools", &tool_names(&self.tools))
            .finish_non_exhaustive()
    }
}

/// `prepareCall`: changes one call's prompt and settings before its turn.
pub type PrepareCallFn = Arc<
    dyn Fn(HarnessCall) -> Pin<Box<dyn Future<Output = AgentResult<HarnessCall>> + Send>>
        + Send
        + Sync,
>;

/// `HarnessAgentSettings`.
#[derive(Clone, Default)]
pub struct HarnessAgentSettings {
    /// Exposed as [`HarnessAgent::id`].
    pub id: Option<String>,
    /// The model id every turn uses, in the harness's own ids. `None` runs
    /// the harness's default.
    pub model: Option<String>,
    /// [rust] The thinking level every turn uses.
    pub thinking_level: Option<ThinkingLevel>,
    pub instructions: Option<String>,
    pub skills: Vec<HarnessSkill>,
    /// Host tools, besides the harness's builtins.
    pub tools: Vec<DynAgentTool>,
    /// Extra HTTP headers for model requests. Names are lowercased; the
    /// headers the harness manages are refused.
    pub headers: Option<IndexMap<String, String>>,
    /// Defaults to [`PermissionMode::AllowAll`].
    pub permission_mode: Option<PermissionMode>,
    /// Only these tools, builtin or host, are active. Not with
    /// `inactive_tools`.
    pub active_tools: Option<Vec<String>>,
    /// These tools, builtin or host, are inactive. Not with `active_tools`.
    pub inactive_tools: Option<Vec<String>>,
    /// `output`: how the final answer is formatted.
    pub response_format: Option<ResponseFormat>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub prepare_call: Option<PrepareCallFn>,
}

impl fmt::Debug for HarnessAgentSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessAgentSettings")
            .field("id", &self.id)
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("tools", &tool_names(&self.tools))
            .field("permission_mode", &self.permission_mode)
            .field("active_tools", &self.active_tools)
            .field("inactive_tools", &self.inactive_tools)
            .finish_non_exhaustive()
    }
}

/// Options of [`HarnessAgent::create_session`].
#[derive(Debug, Clone)]
pub struct HarnessAgentSessionOptions {
    /// Generated when `None`. Pass the original id with `resume`.
    pub session_id: Option<String>,
    /// The sandbox to run in. The caller keeps owning it.
    pub sandbox: HarnessSandbox,
    pub work_dir: String,
    /// State from an earlier session's `detach` or `stop`.
    pub resume: Option<HarnessResumeState>,
    pub cancellation_token: Option<CancellationToken>,
}

impl HarnessAgentSessionOptions {
    pub fn new(sandbox: HarnessSandbox, work_dir: impl Into<String>) -> Self {
        Self {
            session_id: None,
            sandbox,
            work_dir: work_dir.into(),
            resume: None,
            cancellation_token: None,
        }
    }
}

/// Options of [`HarnessAgent::prompt`].
#[derive(Debug, Clone, Default)]
pub struct HarnessAgentCallOptions {
    /// Handed to [`HarnessAgentSettings::prepare_call`] as
    /// [`HarnessCall::options`].
    pub options: Option<Value>,
    pub cancellation_token: Option<CancellationToken>,
}

/// A harness with the settings every turn uses.
pub struct HarnessAgent {
    harness: Arc<dyn Harness>,
    settings: HarnessAgentSettings,
    permission_mode: PermissionMode,
    builtin_tool_filter: Option<BuiltinToolFilter>,
    headers: Option<IndexMap<String, String>>,
}

impl fmt::Debug for HarnessAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessAgent")
            .field("harness_id", &self.harness.id())
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

fn tool_names(tools: &[DynAgentTool]) -> Vec<String> {
    tools.iter().map(|tool| tool.definition().name).collect()
}

fn error(message: impl Into<String>) -> crate::AgentError {
    HarnessError::new(message).into()
}

fn unsupported(message: String, harness_id: &str) -> crate::AgentError {
    HarnessError::capability_unsupported(message, Some(harness_id.to_string())).into()
}

/// `resolveHarnessAgentToolFiltering`: the active host tools and the
/// builtin filter that `active_tools` / `inactive_tools` give.
fn resolve_tool_filtering(
    harness: &dyn Harness,
    user_tools: &[DynAgentTool],
    active_tools: Option<&[String]>,
    inactive_tools: Option<&[String]>,
) -> AgentResult<(Vec<DynAgentTool>, Option<BuiltinToolFilter>)> {
    if active_tools.is_some() && inactive_tools.is_some() {
        return Err(error(
            "HarnessAgent: pass either `active_tools` or `inactive_tools`, not both.",
        ));
    }
    let dedupe = |names: &[String]| {
        let mut seen = HashSet::new();
        names
            .iter()
            .filter(|name| seen.insert(name.as_str()))
            .cloned()
            .collect::<Vec<_>>()
    };
    let active = active_tools.map(dedupe);
    let inactive = inactive_tools.map(dedupe);

    let builtin_names: Vec<String> = harness.builtin_tools().keys().cloned().collect();
    let user_names = tool_names(user_tools);
    let mut all_names = builtin_names.clone();
    for name in &user_names {
        if !all_names.contains(name) {
            all_names.push(name.clone());
        }
    }
    if let Some(requested) = active.as_ref().or(inactive.as_ref()) {
        for name in requested {
            if !all_names.contains(name) {
                return Err(crate::AgentError::ToolNotFound(format!(
                    "Model tried to call unavailable tool '{name}'. Available tools: {}.",
                    all_names.join(", ")
                )));
            }
        }
    }

    let keep_user = |name: &String| match (&active, &inactive) {
        (Some(active), _) => active.contains(name),
        (_, Some(inactive)) => !inactive.contains(name),
        _ => true,
    };
    let active_user_tools = user_tools
        .iter()
        .filter(|tool| keep_user(&tool.definition().name))
        .cloned()
        .collect();

    let disabled_builtins: Vec<String> = match (&active, &inactive) {
        (Some(active), _) => builtin_names
            .iter()
            .filter(|name| !active.contains(name))
            .cloned()
            .collect(),
        (_, Some(inactive)) => builtin_names
            .iter()
            .filter(|name| inactive.contains(name))
            .cloned()
            .collect(),
        _ => Vec::new(),
    };
    let builtin_filter = if disabled_builtins.is_empty() {
        None
    } else if let Some(active) = &active {
        Some(BuiltinToolFilter::Allow {
            tool_names: builtin_names
                .iter()
                .filter(|name| active.contains(name))
                .cloned()
                .collect(),
        })
    } else {
        Some(BuiltinToolFilter::Deny {
            tool_names: disabled_builtins,
        })
    };
    if builtin_filter.is_some()
        && !harness.supports_builtin_tool_filtering()
        && !harness.supports_builtin_tool_approvals()
    {
        return Err(unsupported(
            format!(
                "Harness '{}' does not support built-in tool filtering controls.",
                harness.id()
            ),
            harness.id(),
        ));
    }
    Ok((active_user_tools, builtin_filter))
}

/// `assertNoReservedQuestionTool`.
fn assert_no_reserved_question_tool(
    harness: &dyn Harness,
    user_tools: &[DynAgentTool],
) -> AgentResult<()> {
    if harness.builtin_tools().contains_key(QUESTION_TOOL)
        && user_tools
            .iter()
            .any(|tool| tool.definition().name == QUESTION_TOOL)
    {
        return Err(error(format!(
            "HarnessAgent tool name '{QUESTION_TOOL}' is reserved for harness question requests."
        )));
    }
    Ok(())
}

impl HarnessAgent {
    /// Check the settings against the harness, as the TS constructor does.
    pub fn new(harness: Arc<dyn Harness>, settings: HarnessAgentSettings) -> AgentResult<Self> {
        let mut headers = IndexMap::new();
        for (name, value) in settings.headers.iter().flatten() {
            let name = name.to_lowercase();
            if MANAGED_HEADERS.contains(&name.as_str()) {
                return Err(error(format!(
                    "HarnessAgent: `headers` must not include the managed header `{name}`."
                )));
            }
            headers.insert(name, value.clone());
        }
        assert_no_reserved_question_tool(harness.as_ref(), &settings.tools)?;
        let permission_mode = settings.permission_mode.unwrap_or_default();
        let (_, builtin_tool_filter) = resolve_tool_filtering(
            harness.as_ref(),
            &settings.tools,
            settings.active_tools.as_deref(),
            settings.inactive_tools.as_deref(),
        )?;
        if !harness.builtin_tools().is_empty()
            && permission_mode != PermissionMode::AllowAll
            && !harness.supports_builtin_tool_approvals()
        {
            return Err(unsupported(
                format!(
                    "Harness '{}' does not support built-in tool approval requests; use PermissionMode::AllowAll.",
                    harness.id()
                ),
                harness.id(),
            ));
        }
        Ok(Self {
            harness,
            settings,
            permission_mode,
            builtin_tool_filter,
            headers: (!headers.is_empty()).then_some(headers),
        })
    }

    pub fn id(&self) -> Option<&str> {
        self.settings.id.as_deref()
    }

    /// The id of the harness behind the agent.
    pub fn harness_id(&self) -> &str {
        self.harness.id()
    }

    pub fn harness(&self) -> &Arc<dyn Harness> {
        &self.harness
    }

    pub fn settings(&self) -> &HarnessAgentSettings {
        &self.settings
    }

    /// Start a session, or resume one from `options.resume` under its
    /// original id.
    pub async fn create_session(
        &self,
        options: HarnessAgentSessionOptions,
    ) -> AgentResult<Box<dyn HarnessSession>> {
        if options.resume.is_some() && options.session_id.is_none() {
            return Err(error(
                "HarnessAgent.create_session: pass the original `session_id` with `resume`.",
            ));
        }
        let session_id = options
            .session_id
            .unwrap_or_else(crate::harness::utils::random_uuid);
        let mut start = HarnessStartOptions::new(session_id, options.sandbox, options.work_dir);
        start.resume = options.resume;
        start.permission_mode = self.permission_mode;
        start.builtin_tool_filter = self.builtin_tool_filter.clone();
        start.headers = self.headers.clone();
        start.cancellation_token = options.cancellation_token;
        self.harness.start(start).await
    }

    /// The call a prompt runs: the agent's settings, then `prepare_call`.
    pub async fn prepare_call(
        &self,
        prompt: AgentMessage,
        options: Option<Value>,
    ) -> AgentResult<HarnessCall> {
        let settings = &self.settings;
        let call = HarnessCall {
            prompt,
            options,
            model: settings.model.clone(),
            thinking_level: settings.thinking_level,
            instructions: settings.instructions.clone(),
            skills: settings.skills.clone(),
            tools: settings.tools.clone(),
        };
        match &settings.prepare_call {
            Some(prepare) => prepare(call).await,
            None => Ok(call),
        }
    }

    /// Run one turn of `session` with the agent's settings.
    pub async fn prompt(
        &self,
        session: &dyn HarnessSession,
        prompt: AgentMessage,
        options: HarnessAgentCallOptions,
    ) -> AgentResult<HarnessTurn> {
        let call = self.prepare_call(prompt, options.options).await?;
        assert_no_reserved_question_tool(self.harness.as_ref(), &call.tools)?;
        let (tools, _) = resolve_tool_filtering(
            self.harness.as_ref(),
            &call.tools,
            self.settings.active_tools.as_deref(),
            self.settings.inactive_tools.as_deref(),
        )?;
        let turn = HarnessTurnOptions {
            settings: HarnessTurnSettings {
                model: call.model,
                thinking_level: call.thinking_level,
                instructions: call.instructions.filter(|text| !text.is_empty()),
                skills: call.skills,
                tools,
            },
            response_format: self.settings.response_format.clone(),
            cancellation_token: options.cancellation_token,
            before_tool_call: self.settings.before_tool_call.clone(),
            after_tool_call: self.settings.after_tool_call.clone(),
        };
        session.prompt(call.prompt, turn).await
    }
}

#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;
