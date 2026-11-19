//! Port of `harness-v1-diagnostic.ts` and `harness-v1-observability.ts`.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::harness::provider::JsonObject;

/// `HarnessV1DebugLevel`, ordered most to least severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessV1DebugLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// `HarnessV1DebugConfig`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1DebugConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<HarnessV1DebugLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subsystems: Option<Vec<String>>,
}

/// `HarnessV1Diagnostic['kind']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessV1DiagnosticKind {
    Log,
    Event,
}

/// Standard stream of a captured console line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HarnessV1LogStream {
    Stdout,
    Stderr,
}

/// `HarnessV1Diagnostic['error']`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessV1DiagnosticError {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
}

/// `HarnessV1Diagnostic`: a diagnostic as emitted by a harness adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessV1Diagnostic {
    pub level: HarnessV1DebugLevel,
    pub message: String,
    pub subsystem: String,
    pub kind: HarnessV1DiagnosticKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<HarnessV1LogStream>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attrs: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<HarnessV1DiagnosticError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Emission time (epoch ms).
    pub timestamp: i64,
}

/// `HarnessV1Observability['report']`.
pub type HarnessV1DiagnosticReporter = Arc<dyn Fn(HarnessV1Diagnostic) + Send + Sync>;

/// `HarnessV1Observability`: diagnostics wiring handed to `doStart`.
#[derive(Clone, Default)]
pub struct HarnessV1Observability {
    pub debug: Option<HarnessV1DebugConfig>,
    pub report: Option<HarnessV1DiagnosticReporter>,
}

impl fmt::Debug for HarnessV1Observability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HarnessV1Observability")
            .field("debug", &self.debug)
            .field("report", &self.report.is_some())
            .finish()
    }
}
