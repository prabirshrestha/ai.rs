use reqwest::StatusCode;

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

    #[error("No API key for provider: {0}")]
    MissingApiKey(String),

    #[error("unsupported api: {0}")]
    UnsupportedApi(String),

    #[error("provider {provider} does not support {capability}")]
    UnsupportedCapability {
        provider: String,
        capability: &'static str,
    },

    #[error("No API provider registered for api: {0}")]
    NoApiProvider(String),

    #[error("provider returned HTTP {status}: {body}")]
    ApiStatus { status: StatusCode, body: String },

    #[error("provider error: {0}")]
    Provider(String),

    #[error("{0}")]
    Validation(String),

    #[error("request was cancelled")]
    Cancelled,

    #[error("stream ended before producing a final assistant message")]
    StreamClosed,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn unsupported_capability(provider: impl Into<String>, capability: &'static str) -> Self {
        Self::UnsupportedCapability {
            provider: provider.into(),
            capability,
        }
    }
}
