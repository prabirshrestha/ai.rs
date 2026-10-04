//! Agent error type (no Pi counterpart: Pi throws plain `Error` objects).
//!
//! Variants keep Pi's messages verbatim in their `Display` output.

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Ai(#[from] crate::Error),

    /// A prompt, continuation, reset or run was started while another run is
    /// active. Carries Pi's message for the operation.
    #[error("{0}")]
    AlreadyProcessing(&'static str),

    /// `Agent::continue_run()` without a non-system message.
    #[error("No messages to continue from")]
    NoMessagesToContinue,

    /// `agent_loop_continue()` with an empty context.
    #[error("Cannot continue: no messages in context")]
    NoMessagesInContext,

    #[error("Cannot continue from message role: assistant")]
    CannotContinueFromAssistant,

    /// `getDefaultStreamFn()` without a configured default.
    #[error(
        "No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn()."
    )]
    NoDefaultStreamFn,

    #[error("Agent listener invoked outside active run")]
    ListenerOutsideRun,

    /// Rust addition: a stream function's stream ended without a `done` or
    /// `error` event and without a final result (Pi would wait forever).
    #[error("assistant stream ended before producing a final message")]
    StreamClosed,

    #[error("Operation aborted")]
    Aborted,

    #[error("{0}")]
    Other(String),
}

impl AgentError {
    /// `new Error(message)`.
    pub fn message(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }
}

pub type AgentResult<T> = std::result::Result<T, AgentError>;

pub(crate) const PROMPT_WHILE_PROCESSING: &str = "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.";
pub(crate) const CONTINUE_WHILE_PROCESSING: &str =
    "Agent is already processing. Wait for completion before continuing.";
pub(crate) const RESET_WHILE_PROCESSING: &str =
    "Agent is already processing. Wait for completion before resetting.";
