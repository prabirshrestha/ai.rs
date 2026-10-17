//! Port of durable `src/errors.ts`, plus the crate-wide [`Error`] that stands in
//! for the values TS throws.
//!
//! Divergence from Pi: TS throws arbitrary values and callers match on
//! `instanceof`, `name` and message substrings. Rust uses one cloneable
//! [`Error`] enum whose `Display` strings are Pi's messages verbatim and whose
//! [`Error::name`] is the JS error name (`Error`, `TypeError`,
//! `ReadAfterWrite`, `StorageRejected`, `ConversationBusy`, ...). Errors are
//! cloneable because Pi shares one rejection between several awaiters
//! (memoized acquisitions, shared commits).

use std::sync::Arc;

use crate::chord::delta::DeltaError;
use crate::chord::{AbortReason, JsonError, StateError};

use super::ids::ConversationId;

/// `Result` with the durable [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A transaction read a table after its first table write. Read every required row before writing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Tx.{method}() cannot read tables after the first table write")]
pub struct ReadAfterWrite {
    pub method: String,
}

impl ReadAfterWrite {
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            method: method.into(),
        }
    }
}

/// Storage rejected a batch before any durable effect; the owning Session may continue safely.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct StorageRejected {
    pub message: String,
    pub cause: Option<Arc<Error>>,
}

impl StorageRejected {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            cause: None,
        }
    }

    pub fn with_cause(message: impl Into<String>, cause: Error) -> Self {
        Self {
            message: message.into(),
            cause: Some(Arc::new(cause)),
        }
    }
}

/// A submission reached a busy conversation and was not admitted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Conversation {conversation_id} is busy")]
pub struct ConversationBusy {
    pub conversation_id: ConversationId,
}

impl ConversationBusy {
    pub fn new(conversation_id: ConversationId) -> Self {
        Self { conversation_id }
    }
}

/// Every error durable raises or forwards.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    /// JS `new Error(message, { cause })`.
    #[error("{message}")]
    Message {
        message: String,
        cause: Option<Arc<Error>>,
    },
    /// JS `TypeError`.
    #[error("{0}")]
    Type(String),
    #[error(transparent)]
    ReadAfterWrite(#[from] ReadAfterWrite),
    #[error(transparent)]
    StorageRejected(#[from] StorageRejected),
    #[error(transparent)]
    ConversationBusy(#[from] ConversationBusy),
    /// A cancellation: the abort signal's reason, rethrown as is.
    #[error("{0}")]
    Aborted(AbortReason),
    /// An error thrown by user code (callbacks, listeners, definitions).
    #[error("{0}")]
    Thrown(Arc<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// `new Error(message)`.
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
            cause: None,
        }
    }

    /// `new Error(message, { cause })`.
    pub fn with_cause(message: impl Into<String>, cause: Error) -> Self {
        Self::Message {
            message: message.into(),
            cause: Some(Arc::new(cause)),
        }
    }

    /// `new TypeError(message)`.
    pub fn type_error(message: impl Into<String>) -> Self {
        Self::Type(message.into())
    }

    /// Wrap an error thrown by user code.
    pub fn thrown(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Thrown(Arc::new(error))
    }

    /// The JS `error.name`.
    pub fn name(&self) -> &str {
        match self {
            Self::Message { .. } => "Error",
            Self::Type(_) => "TypeError",
            Self::ReadAfterWrite(_) => "ReadAfterWrite",
            Self::StorageRejected(_) => "StorageRejected",
            Self::ConversationBusy(_) => "ConversationBusy",
            Self::Aborted(reason) if reason.is_abort_error() => "AbortError",
            Self::Aborted(_) => "Error",
            Self::Thrown(_) => "Error",
        }
    }

    /// The JS `error.cause`, when one was recorded.
    pub fn cause(&self) -> Option<&Error> {
        match self {
            Self::Message { cause, .. } => cause.as_deref(),
            Self::StorageRejected(rejected) => rejected.cause.as_deref(),
            _ => None,
        }
    }

    pub fn is_storage_rejected(&self) -> bool {
        matches!(self, Self::StorageRejected(_))
    }
}

impl From<DeltaError> for Error {
    fn from(error: DeltaError) -> Self {
        if error.is_type_error() {
            Self::Type(error.to_string())
        } else {
            Self::Thrown(Arc::new(error))
        }
    }
}

impl From<JsonError> for Error {
    fn from(error: JsonError) -> Self {
        Self::Type(error.0)
    }
}

impl From<AbortReason> for Error {
    fn from(reason: AbortReason) -> Self {
        Self::Aborted(reason)
    }
}

impl From<StateError> for Error {
    fn from(error: StateError) -> Self {
        Self::Thrown(Arc::new(error))
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Type(error.to_string())
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::message(message)
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::message(message)
    }
}
