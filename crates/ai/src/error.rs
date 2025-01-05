/// The main error type for the AI [`crate`].
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Represents errors that occur during IO operations.
    #[error(transparent)]
    IOError(#[from] std::io::Error),

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
