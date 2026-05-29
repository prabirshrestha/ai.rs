#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Ai(#[from] crate::Error),

    #[error("agent is already processing")]
    AlreadyProcessing,

    #[error("no messages to continue from")]
    NoMessagesToContinue,

    #[error("cannot continue from message role: assistant")]
    CannotContinueFromAssistant,

    #[error("tool {0} not found")]
    ToolNotFound(String),

    #[error("operation aborted")]
    Aborted,

    #[error("agent event stream closed before producing final messages")]
    StreamClosed,

    #[error("{0}")]
    Other(String),
}

pub type AgentResult<T> = std::result::Result<T, AgentError>;
