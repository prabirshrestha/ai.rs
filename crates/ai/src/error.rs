/// The main error type for the AI [`crate`].
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Represents errors that occur during IO operations.
    #[error(transparent)]
    IOError(#[from] std::io::Error),

    #[error("Streaming not supported: {0}")]
    StreamingNotSupported(String),

    /// Represents errors that occur during JSON serialization/deserialization.
    #[error(transparent)]
    SerdeJsonError(#[from] serde_json::Error),

    /// An error type indicating that a component provided to a method was out of range, causing a
    /// failure.
    // i64 is the narrowest type fitting all use cases. This eliminates the need for a type parameter.
    #[error(transparent)]
    TimeComponentRangeError(#[from] time::error::ComponentRange),

    /// Represents errors that occur during HTTP operations.
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),

    /// Represent invalid header values.
    #[error("Invalid Header Value: {0} {1}")]
    InvalidHeaderValue(String, reqwest::header::InvalidHeaderValue),

    /// The error type for operations interacting with environment variables.
    /// Possibly returned from [`std::env::var()`].
    #[error("Environment variable error: {0} {1}")]
    EnvVarError(String, std::env::VarError),

    /// Represents [`crate::chat_completions::ChatCompletionRequestBuilder`] errors.
    #[error(transparent)]
    ChatCompletionRequestBuilderError(
        #[from] crate::chat_completions::ChatCompletionRequestBuilderError,
    ),

    /// Represents [`crate::chat_completions::ChatCompletionResponseBuilder`] errors.
    #[error(transparent)]
    ChatCompletionResponseBuilderError(
        #[from] crate::chat_completions::ChatCompletionResponseBuilderError,
    ),

    /// Represents [`crate::chat_completions::ChatCompletionChoiceBuilder`] errors.
    #[error(transparent)]
    ChatCompletionChoiceBuilderError(
        #[from] crate::chat_completions::ChatCompletionChoiceBuilderError,
    ),

    /// Represents [`crate::chat_completions::ChatCompletionToolFunctionDefinitionBuilder`] errors.
    #[error(transparent)]
    ChatCompletionToolFunctionDefinitionBuilderError(
        #[from] crate::chat_completions::ChatCompletionToolFunctionDefinitionBuilderError,
    ),

    /// Represents [`crate::clients::ollama::ClientBuilder`] errors.
    #[cfg(feature = "ollama_client")]
    #[error(transparent)]
    OllamaClientBuilderError(#[from] crate::clients::ollama::ClientBuilderError),

    /// Represents [`crate::clients::openai::ClientBuilder`] errors.
    #[cfg(feature = "openai_client")]
    #[error(transparent)]
    OpenAIClientBuilderError(#[from] crate::clients::openai::ClientBuilderError),

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
///     // run some code that may produce an error from the ai code
///     Ok(())
/// }
/// ```
pub type Result<T> = std::result::Result<T, Error>;
