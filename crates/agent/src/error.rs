#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Ai(#[from] ai::Error),

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

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AgentError>;
