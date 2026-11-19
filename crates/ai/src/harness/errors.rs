//! Port of `@ai-sdk/harness` `src/errors` and `agent/get-harness-error-message.ts`.
//!
//! TS has a `HarnessError` base class with three subclasses, distinguished by
//! `name` markers. Rust models the four classes as one enum; `name()` returns
//! the TS `name` so `get_harness_error_message` can apply the same rule.

use std::error::Error as StdError;

/// Boxed error used as `cause`.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// `AI_HarnessError` and its subclasses.
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    /// `HarnessError` (`AI_HarnessError`).
    #[error("{message}")]
    Harness {
        message: String,
        #[source]
        cause: Option<BoxError>,
    },
    /// `HarnessCapabilityUnsupportedError` (`AI_HarnessCapabilityUnsupportedError`).
    #[error("{message}")]
    CapabilityUnsupported {
        message: String,
        harness_id: Option<String>,
        #[source]
        cause: Option<BoxError>,
    },
    /// `HarnessHistoryUnavailableError` (`AI_HarnessHistoryUnavailableError`).
    #[error("{message}")]
    HistoryUnavailable {
        message: String,
        harness_id: Option<String>,
        #[source]
        cause: Option<BoxError>,
    },
    /// `HarnessSandboxAuthenticationError` (`AI_HarnessSandboxAuthenticationError`).
    #[error("{message}")]
    SandboxAuthentication {
        message: String,
        sandbox_provider_id: String,
        #[source]
        cause: Option<BoxError>,
    },
}

/// `Result` alias used by harness traits.
///
/// [rust] TS promises reject with `unknown`. Rust keeps that: the error is any
/// boxed error, and [`HarnessError`] is one of them. This keeps
/// [`get_harness_error_message`] masking non-harness errors exactly as TS does.
pub type HarnessResult<T> = std::result::Result<T, BoxError>;

impl HarnessError {
    /// `new HarnessError({ message })`.
    pub fn new(message: impl Into<String>) -> Self {
        Self::Harness {
            message: message.into(),
            cause: None,
        }
    }

    /// `new HarnessError({ message, cause })`.
    pub fn with_cause(message: impl Into<String>, cause: impl Into<BoxError>) -> Self {
        Self::Harness {
            message: message.into(),
            cause: Some(cause.into()),
        }
    }

    /// `new HarnessCapabilityUnsupportedError({ message, harnessId })`.
    pub fn capability_unsupported(message: impl Into<String>, harness_id: Option<String>) -> Self {
        Self::CapabilityUnsupported {
            message: message.into(),
            harness_id,
            cause: None,
        }
    }

    /// `new HarnessHistoryUnavailableError({ message, harnessId })`.
    pub fn history_unavailable(message: impl Into<String>, harness_id: Option<String>) -> Self {
        Self::HistoryUnavailable {
            message: message.into(),
            harness_id,
            cause: None,
        }
    }

    /// `new HarnessSandboxAuthenticationError({ message, sandboxProviderId })`.
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

    /// `HarnessError.isInstance(error)`: true for every variant.
    pub fn is_instance(error: &(dyn StdError + 'static)) -> bool {
        error.downcast_ref::<HarnessError>().is_some()
    }

    /// `HarnessCapabilityUnsupportedError.isInstance(error)`.
    pub fn is_capability_unsupported(error: &(dyn StdError + 'static)) -> bool {
        matches!(
            error.downcast_ref::<HarnessError>(),
            Some(HarnessError::CapabilityUnsupported { .. })
        )
    }

    /// `HarnessHistoryUnavailableError.isInstance(error)`.
    pub fn is_history_unavailable(error: &(dyn StdError + 'static)) -> bool {
        matches!(
            error.downcast_ref::<HarnessError>(),
            Some(HarnessError::HistoryUnavailable { .. })
        )
    }

    /// `HarnessSandboxAuthenticationError.isInstance(error)`.
    pub fn is_sandbox_authentication(error: &(dyn StdError + 'static)) -> bool {
        matches!(
            error.downcast_ref::<HarnessError>(),
            Some(HarnessError::SandboxAuthentication { .. })
        )
    }

    /// The TS error `name`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Harness { .. } => "AI_HarnessError",
            Self::CapabilityUnsupported { .. } => "AI_HarnessCapabilityUnsupportedError",
            Self::HistoryUnavailable { .. } => "AI_HarnessHistoryUnavailableError",
            Self::SandboxAuthentication { .. } => "AI_HarnessSandboxAuthenticationError",
        }
    }

    /// The error message.
    pub fn message(&self) -> &str {
        match self {
            Self::Harness { message, .. }
            | Self::CapabilityUnsupported { message, .. }
            | Self::HistoryUnavailable { message, .. }
            | Self::SandboxAuthentication { message, .. } => message,
        }
    }

    /// `harnessId` of the capability and history errors.
    pub fn harness_id(&self) -> Option<&str> {
        match self {
            Self::CapabilityUnsupported { harness_id, .. }
            | Self::HistoryUnavailable { harness_id, .. } => harness_id.as_deref(),
            _ => None,
        }
    }

    /// Attach a `cause` (TS constructor option).
    pub fn caused_by(mut self, error: impl Into<BoxError>) -> Self {
        let slot = match &mut self {
            Self::Harness { cause, .. }
            | Self::CapabilityUnsupported { cause, .. }
            | Self::HistoryUnavailable { cause, .. }
            | Self::SandboxAuthentication { cause, .. } => cause,
        };
        *slot = Some(error.into());
        self
    }
}

const DEFAULT_ERROR_MESSAGE: &str = "An error occurred.";

/// Port of `getHarnessErrorMessage`.
///
/// Returns a client-safe message for errors produced by the harness runtime.
/// Messages from explicitly reviewed harness error types are preserved. All
/// other errors are masked so provider, sandbox, and server details are not
/// exposed to clients by default.
pub fn get_harness_error_message(error: &(dyn StdError + 'static)) -> String {
    match error.downcast_ref::<HarnessError>() {
        Some(
            e @ (HarnessError::CapabilityUnsupported { .. }
            | HarnessError::SandboxAuthentication { .. }
            | HarnessError::Harness { .. }),
        ) => e.message().to_string(),
        _ => DEFAULT_ERROR_MESSAGE.to_string(),
    }
}

/// `new DOMException('Aborted', 'AbortError')`: what an aborted operation
/// rejects with.
///
/// [rust] A `CancellationToken` carries no `reason`, so every abort uses
/// this error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, thiserror::Error)]
#[error("Aborted")]
pub struct AbortError;

impl AbortError {
    /// `error.name`.
    pub fn name(&self) -> &'static str {
        "AbortError"
    }

    /// Whether `error` is an [`AbortError`].
    pub fn is_instance(error: &(dyn StdError + 'static)) -> bool {
        error.is::<AbortError>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // harness-capability-unsupported-error.test.ts
    #[test]
    fn capability_unsupported_is_a_harness_error() {
        let err = HarnessError::capability_unsupported(
            "Harness does not support manual compaction",
            None,
        );
        assert_eq!(err.name(), "AI_HarnessCapabilityUnsupportedError");
        assert!(HarnessError::is_instance(&err));
        assert!(HarnessError::is_capability_unsupported(&err));
    }

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
    }

    #[test]
    fn capability_unsupported_preserves_cause() {
        let err = HarnessError::capability_unsupported("sandbox cannot spawn", None)
            .caused_by(std::io::Error::other("inner"));
        assert_eq!(StdError::source(&err).unwrap().to_string(), "inner");
    }

    // harness-history-unavailable-error.test.ts
    #[test]
    fn history_unavailable_preserves_message_harness_id_and_cause() {
        let err = HarnessError::history_unavailable(
            "No transcript directory for this working directory",
            Some("claude-code".into()),
        )
        .caused_by(std::io::Error::other("ENOENT"));
        assert_eq!(err.name(), "AI_HarnessHistoryUnavailableError");
        assert!(HarnessError::is_instance(&err));
        assert!(HarnessError::is_history_unavailable(&err));
        assert_eq!(
            err.message(),
            "No transcript directory for this working directory"
        );
        assert_eq!(err.harness_id(), Some("claude-code"));
        assert_eq!(StdError::source(&err).unwrap().to_string(), "ENOENT");
    }

    // harness-sandbox-authentication-error.test.ts
    #[test]
    fn sandbox_authentication_preserves_message_provider_and_cause() {
        let err = HarnessError::sandbox_authentication("Set a sandbox API key", "test-sandbox")
            .caused_by(std::io::Error::other("inner"));
        assert_eq!(err.name(), "AI_HarnessSandboxAuthenticationError");
        assert!(HarnessError::is_instance(&err));
        assert!(HarnessError::is_sandbox_authentication(&err));
        assert_eq!(err.message(), "Set a sandbox API key");
        match &err {
            HarnessError::SandboxAuthentication {
                sandbox_provider_id,
                ..
            } => assert_eq!(sandbox_provider_id, "test-sandbox"),
            _ => unreachable!(),
        }
        assert_eq!(StdError::source(&err).unwrap().to_string(), "inner");
    }

    #[test]
    fn is_instance_returns_false_for_unrelated_errors() {
        let err = std::io::Error::other("nope");
        assert!(!HarnessError::is_instance(&err));
        assert!(!HarnessError::is_capability_unsupported(&err));
        assert!(!HarnessError::is_history_unavailable(&err));
        assert!(!HarnessError::is_sandbox_authentication(&err));
        let other = HarnessError::new("base");
        assert!(!HarnessError::is_capability_unsupported(&other));
    }

    // get-harness-error-message.test.ts
    #[test]
    fn preserves_reviewed_harness_error_messages() {
        for err in [
            HarnessError::new("Invalid harness state."),
            HarnessError::capability_unsupported("This capability is unavailable.", None),
            HarnessError::sandbox_authentication("Configure sandbox credentials.", "test"),
        ] {
            assert_eq!(get_harness_error_message(&err), err.message());
        }
    }

    #[test]
    fn masks_unknown_errors() {
        let err = std::io::Error::other("secret details");
        assert_eq!(get_harness_error_message(&err), "An error occurred.");
        let boxed: BoxError = Box::new(std::io::Error::other("secret details"));
        assert_eq!(get_harness_error_message(&*boxed), "An error occurred.");
        let boxed: BoxError = HarnessError::new("kept").into();
        assert_eq!(get_harness_error_message(&*boxed), "kept");
    }

    #[test]
    fn masks_unreviewed_harness_error_subclasses() {
        // TS masks HarnessError subclasses whose name is not reviewed; the
        // history-unavailable error is the only such class in the package.
        let err = HarnessError::history_unavailable("unreviewed details", None);
        assert_eq!(get_harness_error_message(&err), "An error occurred.");
    }
}
