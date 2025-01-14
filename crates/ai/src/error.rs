/// The main error type for the AI [`crate`].
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Represents errors that occur during IO operations.
    #[error(transparent)]
    IOError(#[from] std::io::Error),

    /// Represents errors that occur during JSON serialization/deserialization.
    #[error(transparent)]
    SerdeJsonError(#[from] serde_json::Error),

    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),

    #[error("Invalid Header Value: {0} {1}")]
    InvalidHeaderValue(String, reqwest::header::InvalidHeaderValue),

    /// The error type for operations interacting with environment variables.
    /// Possibly returned from [`std::env::var()`].
    #[error("Environment variable error: {0} {1}")]
    EnvVarError(String, std::env::VarError),

    /// Represents [`crate::chat_completions::ChatCompletionRequestBuilder`] errors.
    #[error(transparent)]
    CompletionRequestBuilderError(
        #[from] crate::chat_completions::ChatCompletionRequestBuilderError,
    ),

    /// Represents [`crate::chat_completions::ChatCompletionResponseBuilder`] errors.
    #[error(transparent)]
    CompletionResponseBuilderError(
        #[from] crate::chat_completions::ChatCompletionResponseBuilderError,
    ),

    /// Represents errors that are uknown or not yet categorized.
    #[error("Unknown error: {0}")]
    UnknownError(String),

    /// Catches any other error types that don't fit into the above categories.
    /// Uses a boxed trait object to support a wide range of error types.
    #[error("OtherError: {0}")]
    OtherError(Box<dyn std::error::Error + Send + Sync + 'static>),
}

/// A specialized [`Result`] type for this ai [`crate`].
///
/// This type is broadly used across ai [`crate`] for any operation which may
/// produce an error.
///
/// This typedef is generally used to avoid writing out [`Error`] directly and
/// is otherwise a direct mapping to [`Result`].
///
/// # Examples
///
/// A convenience function that bubbles an `ai::Result` to its caller:
///
/// ```
///
/// fn generate_chat_completions() -> ai::Result<()> {
///     // run come code that may produce an error from the ai code
///     Ok(())
/// }
/// ```
pub type Result<T> = std::result::Result<T, Error>;
