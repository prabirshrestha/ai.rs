//! Harness errors.

use std::error::Error as StdError;

/// A boxed error, as sandbox implementations return them.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// The result type of [`SandboxSession`](crate::harness::sandbox::SandboxSession)
/// methods and of the harness helpers built on them.
pub type HarnessResult<T> = std::result::Result<T, BoxError>;

/// Errors raised by harnesses.
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    /// Any other harness failure.
    #[error("{message}")]
    Other {
        message: String,
        #[source]
        cause: Option<BoxError>,
    },
    /// The harness does not support the requested operation.
    #[error("{message}")]
    CapabilityUnsupported {
        message: String,
        harness_id: Option<String>,
        #[source]
        cause: Option<BoxError>,
    },
    /// The harness cannot read the session's history.
    #[error("{message}")]
    HistoryUnavailable {
        message: String,
        harness_id: Option<String>,
        #[source]
        cause: Option<BoxError>,
    },
    /// The sandbox provider rejected its credentials.
    #[error("{message}")]
    SandboxAuthentication {
        message: String,
        sandbox_provider_id: String,
        #[source]
        cause: Option<BoxError>,
    },
}

impl HarnessError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
            cause: None,
        }
    }

    pub fn with_cause(message: impl Into<String>, cause: impl Into<BoxError>) -> Self {
        Self::Other {
            message: message.into(),
            cause: Some(cause.into()),
        }
    }

    pub fn capability_unsupported(message: impl Into<String>, harness_id: Option<String>) -> Self {
        Self::CapabilityUnsupported {
            message: message.into(),
            harness_id,
            cause: None,
        }
    }

    pub fn history_unavailable(message: impl Into<String>, harness_id: Option<String>) -> Self {
        Self::HistoryUnavailable {
            message: message.into(),
            harness_id,
            cause: None,
        }
    }

    pub fn sandbox_authentication(
        message: impl Into<String>,
        sandbox_provider_id: impl Into<String>,
    ) -> Self {
        Self::SandboxAuthentication {
            message: message.into(),
            sandbox_provider_id: sandbox_provider_id.into(),
            cause: None,
        }
    }

    /// The harness error inside `error`, looking through `AgentError`.
    pub fn downcast<'a>(error: &'a (dyn StdError + 'static)) -> Option<&'a HarnessError> {
        if let Some(error) = error.downcast_ref::<HarnessError>() {
            return Some(error);
        }
        match error.downcast_ref::<crate::AgentError>() {
            Some(crate::AgentError::Harness(error)) => Some(error),
            _ => None,
        }
    }

    pub fn is_capability_unsupported(error: &(dyn StdError + 'static)) -> bool {
        matches!(
            Self::downcast(error),
            Some(HarnessError::CapabilityUnsupported { .. })
        )
    }

    pub fn is_history_unavailable(error: &(dyn StdError + 'static)) -> bool {
        matches!(
            Self::downcast(error),
            Some(HarnessError::HistoryUnavailable { .. })
        )
    }

    pub fn is_sandbox_authentication(error: &(dyn StdError + 'static)) -> bool {
        matches!(
            Self::downcast(error),
            Some(HarnessError::SandboxAuthentication { .. })
        )
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Other { message, .. }
            | Self::CapabilityUnsupported { message, .. }
            | Self::HistoryUnavailable { message, .. }
            | Self::SandboxAuthentication { message, .. } => message,
        }
    }

    pub fn harness_id(&self) -> Option<&str> {
        match self {
            Self::CapabilityUnsupported { harness_id, .. }
            | Self::HistoryUnavailable { harness_id, .. } => harness_id.as_deref(),
            _ => None,
        }
    }

    pub fn caused_by(mut self, error: impl Into<BoxError>) -> Self {
        let slot = match &mut self {
            Self::Other { cause, .. }
            | Self::CapabilityUnsupported { cause, .. }
            | Self::HistoryUnavailable { cause, .. }
            | Self::SandboxAuthentication { cause, .. } => cause,
        };
        *slot = Some(error.into());
        self
    }
}

/// Convert a sandbox or helper error into an [`AgentError`](crate::AgentError),
/// keeping harness errors typed.
// Unused when no adapter feature is on.
#[allow(dead_code)]
pub(crate) fn into_agent_error(error: BoxError) -> crate::AgentError {
    match error.downcast::<HarnessError>() {
        Ok(error) => crate::AgentError::Harness(*error),
        Err(error) => match error.downcast::<crate::AgentError>() {
            Ok(error) => *error,
            Err(error) => {
                crate::AgentError::Harness(HarnessError::with_cause(error.to_string(), error))
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_unsupported_preserves_message_and_harness_id() {
        let err = HarnessError::capability_unsupported(
            "Claude Code does not support tool approvals",
            Some("claude-code".into()),
        );
        assert_eq!(
            err.to_string(),
            "Claude Code does not support tool approvals"
        );
        assert_eq!(err.harness_id(), Some("claude-code"));
        assert!(HarnessError::is_capability_unsupported(&err));
    }

    #[test]
    fn history_unavailable_preserves_message_harness_id_and_cause() {
        let err = HarnessError::history_unavailable("No transcript", Some("claude-code".into()))
            .caused_by(std::io::Error::other("ENOENT"));
        assert!(HarnessError::is_history_unavailable(&err));
        assert_eq!(err.harness_id(), Some("claude-code"));
        assert_eq!(StdError::source(&err).unwrap().to_string(), "ENOENT");
    }

    #[test]
    fn sandbox_authentication_is_found_through_agent_error() {
        let err = crate::AgentError::Harness(HarnessError::sandbox_authentication(
            "Set a sandbox API key",
            "test-sandbox",
        ));
        assert!(HarnessError::is_sandbox_authentication(&err));
        assert!(!HarnessError::is_capability_unsupported(&err));
    }

    #[test]
    fn into_agent_error_keeps_harness_errors_typed() {
        let boxed: BoxError = HarnessError::capability_unsupported("nope", None).into();
        assert!(matches!(
            into_agent_error(boxed),
            crate::AgentError::Harness(HarnessError::CapabilityUnsupported { .. })
        ));
        let other: BoxError = Box::new(std::io::Error::other("io"));
        assert_eq!(into_agent_error(other).to_string(), "io");
    }
}
