//! The harness API: [`Harness`], [`HarnessSession`] and the turn stream.
//!
//! A harness wraps an agent that runs its own loop (Claude Code, Codex, Pi
//! and so on). A turn streams ai.rs [`AgentEvent`]s, the same events
//! [`Agent`](crate::Agent) emits, plus a few harness-only events.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use async_trait::async_trait;
use futures::Stream;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::harness::bootstrap::HarnessBootstrap;
use crate::harness::error::HarnessError;
use crate::harness::network::HarnessNetworkSandboxSession;
use crate::harness::sandbox::SandboxSession;
use crate::harness::tools::{BuiltinTool, BuiltinToolFilter, HarnessSkill, PermissionMode};
use crate::{
    AfterToolCallFn, AgentError, AgentEvent, AgentMessage, AgentResult, BeforeToolCallFn,
    DynAgentTool, ThinkingLevel,
};

/// The sandbox a session runs in.
#[derive(Clone)]
pub enum HarnessSandbox {
    /// A sandbox with ports and network policy (credential brokering).
    Network(Arc<dyn HarnessNetworkSandboxSession>),
    /// A plain sandbox.
    Basic(Arc<dyn SandboxSession>),
}

impl HarnessSandbox {
    /// The sandbox, restricted to what the agent may touch.
    pub fn session(&self) -> Arc<dyn SandboxSession> {
        match self {
            Self::Network(session) => session.clone().restricted(),
            Self::Basic(session) => session.clone(),
        }
    }

    pub fn network(&self) -> Option<&Arc<dyn HarnessNetworkSandboxSession>> {
        match self {
            Self::Network(session) => Some(session),
            Self::Basic(_) => None,
        }
    }
}

impl fmt::Debug for HarnessSandbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Network(session) => write!(f, "Network({})", session.id()),
            Self::Basic(_) => write!(f, "Basic"),
        }
    }
}

/// Saved state that reopens a session with [`HarnessStartOptions::resume`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessResumeState {
    pub harness_id: String,
    /// Adapter-specific data.
    pub data: Value,
}

/// How the final answer is formatted.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ResponseFormat {
    #[default]
    Text,
    Json {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        schema: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

/// Options of [`Harness::start`].
#[derive(Debug, Clone)]
pub struct HarnessStartOptions {
    pub session_id: String,
    pub sandbox: HarnessSandbox,
    /// The sandbox directory the agent works in.
    pub work_dir: String,
    pub resume: Option<HarnessResumeState>,
    /// Which builtin tools run without `before_tool_call`.
    pub permission_mode: PermissionMode,
    pub builtin_tool_filter: Option<BuiltinToolFilter>,
    /// Extra HTTP headers for model requests, where the agent supports them.
    pub headers: Option<IndexMap<String, String>>,
    pub cancellation_token: Option<CancellationToken>,
}

impl HarnessStartOptions {
    pub fn new(
        session_id: impl Into<String>,
        sandbox: HarnessSandbox,
        work_dir: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            sandbox,
            work_dir: work_dir.into(),
            resume: None,
            permission_mode: PermissionMode::default(),
            builtin_tool_filter: None,
            headers: None,
            cancellation_token: None,
        }
    }
}

/// Per-turn settings.
#[derive(Clone, Default)]
pub struct HarnessTurnSettings {
    /// The agent's own model id.
    pub model: Option<String>,
    pub thinking_level: Option<ThinkingLevel>,
    /// Instructions added to the agent's system prompt.
    pub instructions: Option<String>,
    pub skills: Vec<HarnessSkill>,
    /// Host tools the agent may call. The session runs them.
    pub tools: Vec<DynAgentTool>,
}

impl fmt::Debug for HarnessTurnSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessTurnSettings")
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("instructions", &self.instructions)
            .field("skills", &self.skills)
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.definition().name)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Options of [`HarnessSession::prompt`].
#[derive(Clone, Default)]
pub struct HarnessTurnOptions {
    pub settings: HarnessTurnSettings,
    pub response_format: Option<ResponseFormat>,
    pub cancellation_token: Option<CancellationToken>,
    /// Asked before a host tool, or a builtin that `permission_mode` does not
    /// allow, runs. Blocking it denies the call.
    pub before_tool_call: Option<BeforeToolCallFn>,
    /// Called after a host tool runs.
    pub after_tool_call: Option<AfterToolCallFn>,
}

impl fmt::Debug for HarnessTurnOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessTurnOptions")
            .field("settings", &self.settings)
            .field("response_format", &self.response_format)
            .finish_non_exhaustive()
    }
}

/// Whether a file was created or changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileChangeKind {
    Create,
    Modify,
    Delete,
}

/// How loud a [`HarnessEvent::Notice`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NoticeLevel {
    Info,
    Warning,
    Error,
}

/// One event of a turn.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum HarnessEvent {
    /// The same events [`Agent`](crate::Agent) emits. Builtin tools the agent
    /// runs itself show up as `ToolExecution*` events with their harness names.
    Agent(AgentEvent),
    /// The agent compacted its context.
    Compaction {
        tokens_before: Option<u64>,
        tokens_after: Option<u64>,
        summary: Option<String>,
    },
    /// The agent changed a workspace file.
    FileChange { kind: FileChangeKind, path: String },
    /// A warning or diagnostic from the agent.
    Notice { level: NoticeLevel, message: String },
}

impl From<AgentEvent> for HarnessEvent {
    fn from(event: AgentEvent) -> Self {
        Self::Agent(event)
    }
}

type SteerFn = Arc<
    dyn Fn(AgentMessage) -> Pin<Box<dyn Future<Output = AgentResult<()>> + Send>> + Send + Sync,
>;

/// A running turn: a stream of [`HarnessEvent`]s, then the new messages.
pub struct HarnessTurn {
    receiver: mpsc::UnboundedReceiver<HarnessEvent>,
    result_receiver: Option<oneshot::Receiver<AgentResult<Vec<AgentMessage>>>>,
    result: Option<Vec<AgentMessage>>,
    steer: Option<SteerFn>,
}

impl fmt::Debug for HarnessTurn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessTurn")
            .field("steer", &self.steer.is_some())
            .finish_non_exhaustive()
    }
}

/// The adapter side of a [`HarnessTurn`].
#[derive(Clone)]
pub struct HarnessTurnSender {
    sender: mpsc::UnboundedSender<HarnessEvent>,
}

impl HarnessTurnSender {
    /// Emit one event. Events sent after the caller drops the turn are lost.
    pub fn emit(&self, event: impl Into<HarnessEvent>) {
        let _ = self.sender.send(event.into());
    }
}

impl HarnessTurnSender {
    /// [rust] A sender and its receiver, for work outside a turn whose
    /// events an adapter collects itself (OpenCode compaction).
    #[allow(dead_code)]
    pub(crate) fn channel() -> (Self, mpsc::UnboundedReceiver<HarnessEvent>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (Self { sender }, receiver)
    }
}

/// [rust] A sender whose events go nowhere, for adapter unit tests.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn test_turn_sender() -> (HarnessTurnSender, mpsc::UnboundedReceiver<HarnessEvent>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    (HarnessTurnSender { sender }, receiver)
}

impl HarnessTurn {
    /// Run `run` as a turn. It emits through the sender and returns the
    /// turn's new messages.
    pub fn spawn<F, Fut>(run: F) -> Self
    where
        F: FnOnce(HarnessTurnSender) -> Fut + Send + 'static,
        Fut: Future<Output = AgentResult<Vec<AgentMessage>>> + Send + 'static,
    {
        let (sender, receiver) = mpsc::unbounded_channel();
        let (result_sender, result_receiver) = oneshot::channel();
        tokio::spawn(async move {
            let result = run(HarnessTurnSender { sender }).await;
            let _ = result_sender.send(result);
        });
        Self {
            receiver,
            result_receiver: Some(result_receiver),
            result: None,
            steer: None,
        }
    }

    /// Let callers add messages to the turn while it runs.
    pub fn with_steer<F, Fut>(mut self, steer: F) -> Self
    where
        F: Fn(AgentMessage) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = AgentResult<()>> + Send + 'static,
    {
        self.steer = Some(Arc::new(move |message| Box::pin(steer(message))));
        self
    }

    /// Whether [`steer`](Self::steer) is supported.
    pub fn supports_steer(&self) -> bool {
        self.steer.is_some()
    }

    /// Add a user message to the running turn.
    pub async fn steer(&self, message: AgentMessage) -> AgentResult<()> {
        match &self.steer {
            Some(steer) => steer(message).await,
            None => Err(HarnessError::capability_unsupported(
                "This harness does not support steering a running turn.",
                None,
            )
            .into()),
        }
    }

    /// Wait for the turn to end and return its new messages.
    pub async fn result(&mut self) -> AgentResult<Vec<AgentMessage>> {
        if let Some(result) = &self.result {
            return Ok(result.clone());
        }
        let receiver = self
            .result_receiver
            .take()
            .ok_or(AgentError::StreamClosed)?;
        let result = receiver.await.map_err(|_| AgentError::StreamClosed)??;
        self.result = Some(result.clone());
        Ok(result)
    }

    /// Drain the events and return the new messages.
    pub async fn collect(mut self) -> AgentResult<(Vec<HarnessEvent>, Vec<AgentMessage>)> {
        let mut events = Vec::new();
        while let Some(event) = self.receiver.recv().await {
            events.push(event);
        }
        let messages = self.result().await?;
        Ok((events, messages))
    }
}

impl Stream for HarnessTurn {
    type Item = HarnessEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

fn unsupported<T>(member: &str, harness_id: &str) -> AgentResult<T> {
    Err(HarnessError::capability_unsupported(
        format!("This harness does not support {member}."),
        Some(harness_id.to_string()),
    )
    .into())
}

/// One agent session inside a sandbox.
#[async_trait]
pub trait HarnessSession: Send + Sync {
    fn harness_id(&self) -> &str;
    fn session_id(&self) -> &str;
    /// Whether the session was reopened from a [`HarnessResumeState`].
    fn is_resume(&self) -> bool;

    /// Run one turn.
    async fn prompt(
        &self,
        message: AgentMessage,
        options: HarnessTurnOptions,
    ) -> AgentResult<HarnessTurn>;

    /// Compact the agent's context.
    async fn compact(&self, _instructions: Option<String>) -> AgentResult<()> {
        unsupported("manual compaction", self.harness_id())
    }

    fn supports_history(&self) -> bool {
        false
    }

    /// The session's messages so far.
    async fn messages(&self) -> AgentResult<Vec<AgentMessage>> {
        unsupported("reading history", self.harness_id())
    }

    /// [rust] What the agent offers here: whether it is signed in, its
    /// models and its commands, read from a short-lived process.
    async fn probe(&self) -> AgentResult<crate::harness::probe::HarnessProbe> {
        unsupported("probing", self.harness_id())
    }

    /// Leave the agent running and return the state to reattach to it.
    async fn detach(&self) -> AgentResult<HarnessResumeState>;

    /// Stop the agent and return the state to resume it later.
    async fn stop(&self) -> AgentResult<HarnessResumeState>;

    /// Stop the agent and delete its state.
    async fn destroy(&self) -> AgentResult<()>;
}

/// An agent that runs its own loop inside a sandbox.
#[async_trait]
pub trait Harness: Send + Sync {
    fn id(&self) -> &str;

    /// Tools the agent runs itself, keyed by harness name.
    fn builtin_tools(&self) -> &IndexMap<String, BuiltinTool>;

    /// Whether the agent can ask before a builtin tool runs. Without it, only
    /// [`PermissionMode::AllowAll`] works.
    fn supports_builtin_tool_approvals(&self) -> bool {
        false
    }

    fn supports_builtin_tool_filtering(&self) -> bool {
        false
    }

    /// Files and commands that install the agent in a fresh sandbox, for
    /// building sandbox templates ahead of time.
    async fn bootstrap(&self) -> AgentResult<Option<HarnessBootstrap>> {
        Ok(None)
    }

    async fn start(&self, options: HarnessStartOptions) -> AgentResult<Box<dyn HarnessSession>>;
}

/// Check the start options against what the harness supports.
pub fn validate_start_options(
    harness: &dyn Harness,
    options: &HarnessStartOptions,
) -> AgentResult<()> {
    if options.permission_mode != PermissionMode::AllowAll
        && !harness.supports_builtin_tool_approvals()
    {
        return Err(HarnessError::capability_unsupported(
            format!(
                "Harness '{}' does not support built-in tool approval requests; use PermissionMode::AllowAll.",
                harness.id()
            ),
            Some(harness.id().to_string()),
        )
        .into());
    }
    if options.builtin_tool_filter.is_some() && !harness.supports_builtin_tool_filtering() {
        return Err(HarnessError::capability_unsupported(
            format!(
                "Harness '{}' does not support built-in tool filtering.",
                harness.id()
            ),
            Some(harness.id().to_string()),
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;

    use super::*;

    #[tokio::test]
    async fn turn_streams_events_then_returns_messages() {
        let mut turn = HarnessTurn::spawn(|sender| async move {
            sender.emit(AgentEvent::AgentStart);
            sender.emit(HarnessEvent::Notice {
                level: NoticeLevel::Info,
                message: "hi".into(),
            });
            Ok(vec![crate::user_message("done", vec![])])
        });
        let first = turn.next().await.unwrap();
        assert!(matches!(first, HarnessEvent::Agent(AgentEvent::AgentStart)));
        let (rest, messages) = turn.collect().await.unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(messages.len(), 1);
    }

    #[tokio::test]
    async fn steer_is_unsupported_unless_set() {
        let turn = HarnessTurn::spawn(|_| async { Ok(vec![]) });
        assert!(!turn.supports_steer());
        let error = turn
            .steer(crate::user_message("more", vec![]))
            .await
            .unwrap_err();
        assert!(HarnessError::is_capability_unsupported(&error));
    }

    #[test]
    fn resume_state_round_trips() {
        let state = HarnessResumeState {
            harness_id: "pi".into(),
            data: serde_json::json!({ "file": "a.jsonl" }),
        };
        let value = serde_json::to_value(&state).unwrap();
        assert_eq!(value["harnessId"], "pi");
        assert_eq!(
            serde_json::from_value::<HarnessResumeState>(value).unwrap(),
            state
        );
    }
}
