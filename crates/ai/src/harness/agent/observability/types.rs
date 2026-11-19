//! Port of `agent/observability/types.ts`.

use serde::{Deserialize, Serialize};

use crate::harness::provider::JsonObject;

/// `HarnessDebugLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessDebugLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl HarnessDebugLevel {
    /// The level name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

/// `HarnessDebugConfig`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessDebugConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<HarnessDebugLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subsystems: Option<Vec<String>>,
}

/// `HarnessDiagnostic['kind']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessDiagnosticKind {
    Log,
    Event,
}

/// `HarnessDiagnostic['stream']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessDiagnosticStream {
    Stdout,
    Stderr,
}

/// `HarnessDiagnostic['error']`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessDiagnosticErrorInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
}

/// `HarnessDiagnostic`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessDiagnostic {
    pub level: HarnessDebugLevel,
    pub message: String,
    pub subsystem: String,
    pub kind: HarnessDiagnosticKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<HarnessDiagnosticStream>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attrs: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<HarnessDiagnosticErrorInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Milliseconds since the epoch.
    pub timestamp: i64,
}

/// `HarnessDiagnosticConsumer`.
pub trait HarnessDiagnosticConsumer: Send + Sync {
    /// `ingestDiagnostic?(diagnostic)`.
    fn ingest_diagnostic(&self, diagnostic: &HarnessDiagnostic);
}
