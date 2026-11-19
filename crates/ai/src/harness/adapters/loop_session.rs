//! [rust] A [`HarnessSession`] that runs ai.rs's own agent loop in-process.
//! The native ai and Pi adapters both run on it.
//!
//! It keeps the session's history, filters and approves builtin tools, queues
//! steering messages, reports builtin tool events under their harness names
//! and saves the history as the resume state.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::agent_types::StreamFn;
use crate::harness::adapters::pi::pi_utils::frame_instructions;
use crate::harness::error::HarnessError;
use crate::harness::session::{
    HarnessEvent, HarnessResumeState, HarnessSession, HarnessTurn, HarnessTurnOptions,
    HarnessTurnSender, ResponseFormat,
};
use crate::harness::tools::{BuiltinToolFilter, BuiltinToolKind, HarnessSkill, PermissionMode};
use crate::{
    AgentContext, AgentEvent, AgentEventSink, AgentLoopConfig, AgentMessage, AgentResult,
    BeforeToolCallFn, BeforeToolCallResult, DynAgentTool, Message, Model, UserContent,
    UserMessageContent, run_agent_loop,
};

/// A tool the harness itself provides.
#[derive(Clone)]
pub struct LoopBuiltinTool {
    pub tool: DynAgentTool,
    /// The name the harness reports (`glob` for Pi's `find`).
    pub harness_name: String,
    pub kind: BuiltinToolKind,
}

impl LoopBuiltinTool {
    fn native_name(&self) -> String {
        self.tool.definition().name
    }
}

/// Picks the model for a turn from `HarnessTurnSettings::model`.
pub type LoopModelResolver = Arc<dyn Fn(Option<&str>) -> AgentResult<Model> + Send + Sync>;

/// Whether a builtin of `kind` must be approved under `mode`.
pub type LoopApprovalPolicy = Arc<dyn Fn(PermissionMode, BuiltinToolKind) -> bool + Send + Sync>;

/// Called with the whole history after each turn, to persist it.
pub type LoopPersistFn = Arc<
    dyn Fn(Vec<AgentMessage>) -> Pin<Box<dyn Future<Output = AgentResult<()>> + Send>>
        + Send
        + Sync,
>;

/// Called once a turn with the sender, so session-level hooks (file changes)
/// can emit into the running turn.
pub type LoopTurnSenderSlot = Arc<StdMutex<Option<HarnessTurnSender>>>;

/// Everything a [`LoopSession`] needs.
#[derive(Clone)]
pub struct LoopSessionOptions {
    pub harness_id: String,
    pub session_id: String,
    pub is_resume: bool,
    /// The history to continue from.
    pub messages: Vec<AgentMessage>,
    pub system_prompt: String,
    pub builtins: Vec<LoopBuiltinTool>,
    pub permission_mode: PermissionMode,
    pub builtin_tool_filter: Option<BuiltinToolFilter>,
    pub requires_approval: LoopApprovalPolicy,
    pub resolve_model: LoopModelResolver,
    pub stream_fn: Option<StreamFn>,
    pub cancellation_token: Option<CancellationToken>,
    pub persist: Option<LoopPersistFn>,
    pub turn_sender: Option<LoopTurnSenderSlot>,
}

struct LoopState {
    messages: Vec<AgentMessage>,
    stopped: bool,
}

/// An in-process agent session.
pub struct LoopSession {
    options: LoopSessionOptions,
    state: Arc<Mutex<LoopState>>,
    /// Held by the running turn.
    turn_lock: Arc<Mutex<()>>,
    session_token: CancellationToken,
}

/// The resume data of a loop session: its serialized history.
pub fn loop_resume_state(harness_id: &str, messages: &[AgentMessage]) -> HarnessResumeState {
    HarnessResumeState {
        harness_id: harness_id.to_string(),
        data: json!({ "messages": messages }),
    }
}

/// The history saved in a resume state.
pub fn loop_resume_messages(
    harness_id: &str,
    state: &HarnessResumeState,
) -> AgentResult<Vec<AgentMessage>> {
    if state.harness_id != harness_id {
        return Err(HarnessError::new(format!(
            "Cannot resume a '{}' session with the '{harness_id}' harness.",
            state.harness_id
        ))
        .into());
    }
    let messages = state
        .data
        .get("messages")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    serde_json::from_value(messages).map_err(|error| {
        HarnessError::with_cause(
            format!("Invalid '{harness_id}' resume state: {error}"),
            error,
        )
        .into()
    })
}

/// Render the turn's skills for the system prompt.
pub fn render_skills_prompt(skills: &[HarnessSkill]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut text = String::from(
        "<skills>\nThe following skills are available. Follow a skill's instructions when the task matches its description.\n",
    );
    for skill in skills {
        text.push_str(&format!(
            "\n<skill name=\"{}\">\n<description>{}</description>\n{}\n",
            skill.name, skill.description, skill.content
        ));
        for file in skill.files.iter().flatten() {
            text.push_str(&format!(
                "<file path=\"{}\">\n{}\n</file>\n",
                file.path, file.content
            ));
        }
        text.push_str("</skill>\n");
    }
    text.push_str("</skills>");
    Some(text)
}

/// The instruction asking for a structured response.
pub fn render_response_format_prompt(format: &ResponseFormat) -> Option<String> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::Json {
            schema,
            name,
            description,
        } => {
            let mut text = String::from(
                "Respond with only a single JSON value, without Markdown fences or any other text.",
            );
            if let Some(name) = name {
                text.push_str(&format!(" The value is a {name}."));
            }
            if let Some(description) = description {
                text.push_str(&format!(" {description}"));
            }
            if let Some(schema) = schema {
                text.push_str(&format!(" It must match this JSON schema:\n{schema}"));
            }
            Some(text)
        }
    }
}

/// Frame `instructions` into the first user message of a fresh session.
fn frame_first_message(message: AgentMessage, instructions: &str) -> AgentMessage {
    match message {
        Message::User(mut user) => {
            user.content = match user.content {
                UserMessageContent::Text(text) => {
                    UserMessageContent::Text(frame_instructions(instructions, &text))
                }
                UserMessageContent::Parts(mut parts) => {
                    let index = parts
                        .iter()
                        .position(|part| matches!(part, UserContent::Text(_)));
                    match index {
                        Some(index) => {
                            if let UserContent::Text(text) = &mut parts[index] {
                                text.text = frame_instructions(instructions, &text.text);
                            }
                        }
                        None => parts.insert(
                            0,
                            UserContent::Text(crate::TextContent {
                                text: frame_instructions(instructions, ""),
                                text_signature: None,
                            }),
                        ),
                    }
                    UserMessageContent::Parts(parts)
                }
            };
            Message::User(user)
        }
        other => other,
    }
}

fn rename_tool_event(event: AgentEvent, names: &[(String, String)]) -> AgentEvent {
    let rename = |name: String| {
        names
            .iter()
            .find(|(native, _)| *native == name)
            .map_or(name, |(_, harness)| harness.clone())
    };
    match event {
        AgentEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
        } => AgentEvent::ToolExecutionStart {
            tool_call_id,
            tool_name: rename(tool_name),
            args,
        },
        AgentEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            args,
            partial_result,
        } => AgentEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name: rename(tool_name),
            args,
            partial_result,
        },
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result,
            is_error,
        } => AgentEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name: rename(tool_name),
            result,
            is_error,
        },
        other => other,
    }
}

impl LoopSession {
    pub fn new(options: LoopSessionOptions) -> Self {
        let session_token = options
            .cancellation_token
            .as_ref()
            .map_or_else(CancellationToken::new, CancellationToken::child_token);
        Self {
            state: Arc::new(Mutex::new(LoopState {
                messages: options.messages.clone(),
                stopped: false,
            })),
            options,
            turn_lock: Arc::new(Mutex::new(())),
            session_token,
        }
    }

    fn active_builtins(&self) -> Vec<LoopBuiltinTool> {
        self.options
            .builtins
            .iter()
            .filter(|builtin| {
                BuiltinToolFilter::includes(
                    self.options.builtin_tool_filter.as_ref(),
                    &builtin.harness_name,
                )
            })
            .cloned()
            .collect()
    }

    fn approval_hook(
        &self,
        builtins: Vec<LoopBuiltinTool>,
        user: Option<BeforeToolCallFn>,
    ) -> BeforeToolCallFn {
        let mode = self.options.permission_mode;
        let requires_approval = self.options.requires_approval.clone();
        Arc::new(move |mut context, cancellation_token| {
            let builtin = builtins
                .iter()
                .find(|builtin| builtin.native_name() == context.tool_call.name)
                .cloned();
            let user = user.clone();
            let requires_approval = requires_approval.clone();
            Box::pin(async move {
                match builtin {
                    Some(builtin) => {
                        if !requires_approval(mode, builtin.kind) {
                            return Ok(None);
                        }
                        // Approvers see the harness name.
                        context.tool_call.name = builtin.harness_name.clone();
                        match user {
                            Some(user) => user(context, cancellation_token).await,
                            None => Ok(Some(BeforeToolCallResult {
                                block: true,
                                reason: Some(format!(
                                    "Tool '{}' requires approval, but no approval handler was provided.",
                                    builtin.harness_name
                                )),
                            })),
                        }
                    }
                    None => match user {
                        Some(user) => user(context, cancellation_token).await,
                        None => Ok(None),
                    },
                }
            })
        })
    }
}

#[async_trait]
impl HarnessSession for LoopSession {
    fn harness_id(&self) -> &str {
        &self.options.harness_id
    }

    fn session_id(&self) -> &str {
        &self.options.session_id
    }

    fn is_resume(&self) -> bool {
        self.options.is_resume
    }

    async fn prompt(
        &self,
        message: AgentMessage,
        options: HarnessTurnOptions,
    ) -> AgentResult<HarnessTurn> {
        if self.state.lock().await.stopped {
            return Err(HarnessError::new("The session has been stopped.").into());
        }
        let turn_guard = self
            .turn_lock
            .clone()
            .try_lock_owned()
            .map_err(|_| HarnessError::new("A turn is already running in this session."))?;

        let HarnessTurnOptions {
            settings,
            response_format,
            cancellation_token,
            before_tool_call,
            after_tool_call,
        } = options;

        let builtins = self.active_builtins();
        let mut tools: Vec<DynAgentTool> = builtins.iter().map(|b| b.tool.clone()).collect();
        for tool in &settings.tools {
            let name = tool.definition().name;
            if self
                .options
                .builtins
                .iter()
                .any(|builtin| builtin.native_name() == name)
            {
                return Err(HarnessError::new(format!(
                    "Tool '{name}' conflicts with a built-in tool of the '{}' harness.",
                    self.options.harness_id
                ))
                .into());
            }
            tools.push(tool.clone());
        }

        let mut system_prompt = self.options.system_prompt.clone();
        let extra = [
            render_skills_prompt(&settings.skills),
            response_format
                .as_ref()
                .and_then(render_response_format_prompt),
        ];
        for part in extra.into_iter().flatten() {
            if !system_prompt.is_empty() {
                system_prompt.push_str("\n\n");
            }
            system_prompt.push_str(&part);
        }

        let model = (self.options.resolve_model)(settings.model.as_deref())?;
        let mut config = AgentLoopConfig::new(model);
        if let Some(level) = settings.thinking_level {
            config.options.reasoning = Some(level.into());
        }
        config.after_tool_call = after_tool_call;
        config.before_tool_call = Some(self.approval_hook(builtins.clone(), before_tool_call));
        let steering: Arc<StdMutex<Vec<AgentMessage>>> = Arc::default();
        {
            let steering = steering.clone();
            config.get_steering_messages = Some(Arc::new(move || {
                let messages = std::mem::take(&mut *steering.lock().unwrap());
                Box::pin(async move { messages })
            }));
        }

        let history = self.state.lock().await.messages.clone();
        let message = match settings.instructions.as_deref() {
            Some(instructions) if history.is_empty() && !instructions.is_empty() => {
                frame_first_message(message, instructions)
            }
            _ => message,
        };

        let turn_token = self.session_token.child_token();
        if let Some(token) = cancellation_token {
            let turn_token = turn_token.clone();
            tokio::spawn(async move {
                tokio::select! {
                    _ = token.cancelled() => turn_token.cancel(),
                    _ = turn_token.cancelled() => {}
                }
            });
        }

        let names: Vec<(String, String)> = builtins
            .iter()
            .map(|builtin| (builtin.native_name(), builtin.harness_name.clone()))
            .collect();
        let state = self.state.clone();
        let stream_fn = self.options.stream_fn.clone();
        let persist = self.options.persist.clone();
        let sender_slot = self.options.turn_sender.clone();

        let turn = HarnessTurn::spawn(move |sender| async move {
            let _turn_guard = turn_guard;
            if let Some(slot) = &sender_slot {
                *slot.lock().unwrap() = Some(sender.clone());
            }
            let emit: AgentEventSink = {
                let sender = sender.clone();
                Arc::new(move |event| {
                    sender.emit(HarnessEvent::Agent(rename_tool_event(event, &names)));
                    Box::pin(async { Ok(()) })
                })
            };
            let context = AgentContext {
                system_prompt,
                messages: history,
                tools,
            };
            let result = run_agent_loop(
                vec![message],
                context,
                config,
                emit,
                Some(turn_token.clone()),
                stream_fn,
            )
            .await;
            turn_token.cancel();
            if let Some(slot) = &sender_slot {
                *slot.lock().unwrap() = None;
            }
            let new_messages = result?;
            let all = {
                let mut state = state.lock().await;
                state.messages.extend(new_messages.iter().cloned());
                state.messages.clone()
            };
            if let Some(persist) = persist {
                persist(all).await?;
            }
            Ok(new_messages)
        });

        Ok(turn.with_steer(move |message| {
            let steering = steering.clone();
            async move {
                steering.lock().unwrap().push(message);
                Ok(())
            }
        }))
    }

    fn supports_history(&self) -> bool {
        true
    }

    async fn messages(&self) -> AgentResult<Vec<AgentMessage>> {
        Ok(self.state.lock().await.messages.clone())
    }

    async fn detach(&self) -> AgentResult<HarnessResumeState> {
        let state = self.state.lock().await;
        Ok(loop_resume_state(&self.options.harness_id, &state.messages))
    }

    async fn stop(&self) -> AgentResult<HarnessResumeState> {
        self.session_token.cancel();
        // Wait for a running turn to record its messages.
        let _turn = self.turn_lock.lock().await;
        let mut state = self.state.lock().await;
        state.stopped = true;
        Ok(loop_resume_state(&self.options.harness_id, &state.messages))
    }

    async fn destroy(&self) -> AgentResult<()> {
        self.session_token.cancel();
        let _turn = self.turn_lock.lock().await;
        let mut state = self.state.lock().await;
        state.stopped = true;
        state.messages.clear();
        Ok(())
    }
}
