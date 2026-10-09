use crate::utils::models_error::ModelsError;
use crate::utils::provider_retry::ProviderHttpError;

/// Errors raised by the crate's fallible entry points.
///
/// Pi throws plain `Error` objects with a message. Rust needs a typed error,
/// so the variants below keep Pi's messages verbatim in their `Display`
/// output and add structured data where Pi probes error fields (HTTP status
/// and headers for retries, `ModelsError.code`).
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("invalid header value for {0}: {1}")]
    InvalidHeaderValue(String, reqwest::header::InvalidHeaderValue),

    /// A non-2xx provider response. Port of the SDK `APIError` shape that
    /// `retryProviderRequest()` and `normalizeProviderError()` probe.
    #[error(transparent)]
    ProviderHttp(Box<ProviderHttpError>),

    /// Port of Pi's `ModelsError`.
    #[error(transparent)]
    Models(#[from] ModelsError),

    /// A request or wait was aborted through its `signal`. Carries Pi's
    /// abort message (for example `"Request aborted"`).
    #[error("{0}")]
    Aborted(String),

    /// Tool argument validation failure (`validateToolArguments`).
    #[error("{0}")]
    Validation(String),

    /// Any other error Pi raises as `new Error(message)`.
    #[error("{0}")]
    Provider(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// `new Error(message)`.
    pub fn message(message: impl Into<String>) -> Self {
        Self::Provider(message.into())
    }

    /// Pi's default abort reason (`abortReason()` in `utils/abort.ts`).
    pub fn aborted() -> Self {
        Self::Aborted("The operation was aborted".to_string())
    }

    /// Whether this error is an abort (`error.name === "AbortError"`).
    pub fn is_abort(&self) -> bool {
        matches!(self, Self::Aborted(_))
    }
}

impl From<ProviderHttpError> for Error {
    fn from(error: ProviderHttpError) -> Self {
        Self::ProviderHttp(Box::new(error))
    }
}
