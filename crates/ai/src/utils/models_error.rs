//! Port of `utils/models-error.ts`.

use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelsErrorCode {
    ModelSource,
    ModelValidation,
    Provider,
    Stream,
    Auth,
    Oauth,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelsError {
    pub code: ModelsErrorCode,
    pub message: String,
}

impl ModelsError {
    pub fn new(code: ModelsErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// `new ModelsError(code, message, { cause })`: callers surface
    /// `message` only, so the cause's text is appended to it.
    pub fn with_cause(
        code: ModelsErrorCode,
        message: impl Into<String>,
        cause: &dyn fmt::Display,
    ) -> Self {
        Self::new(code, with_cause_detail(message.into(), cause))
    }
}

fn with_cause_detail(message: String, cause: &dyn fmt::Display) -> String {
    let detail = cause.to_string();
    let detail = detail.trim();
    if detail.is_empty() || message.contains(detail) {
        return message;
    }
    format!("{message}: {detail}")
}

impl fmt::Display for ModelsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ModelsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_cause_detail_unless_already_present() {
        let error = ModelsError::with_cause(ModelsErrorCode::Auth, "Login failed", &"bad token");
        assert_eq!(error.to_string(), "Login failed: bad token");
        let error = ModelsError::with_cause(ModelsErrorCode::Auth, "bad token here", &"bad token");
        assert_eq!(error.to_string(), "bad token here");
    }
}
