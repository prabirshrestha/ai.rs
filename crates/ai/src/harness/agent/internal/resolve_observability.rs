//! Port of `agent/internal/resolve-observability.ts`.

use std::io::Write as _;
use std::sync::Arc;

use crate::harness::agent::observability::{
    HarnessDebugConfig, HarnessDebugLevel, HarnessDiagnostic, HarnessDiagnosticErrorInfo,
    HarnessDiagnosticKind, HarnessDiagnosticStream,
};
use crate::harness::sdk::telemetry::TelemetryOptions;
use crate::harness::spec::diagnostic::{
    HarnessV1DebugConfig, HarnessV1DebugLevel, HarnessV1Diagnostic, HarnessV1DiagnosticKind,
    HarnessV1LogStream, HarnessV1Observability,
};

/// `HarnessAgentSettings['onLog']`.
pub type OnLogFn = Arc<dyn Fn(&HarnessDiagnostic) + Send + Sync>;

const ENV_TRUTHY: [&str; 4] = ["1", "true", "yes", "on"];

fn parse_list(value: Option<String>) -> Option<Vec<String>> {
    let value = value.filter(|v| !v.is_empty())?;
    let items: Vec<String> = value
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect();
    (!items.is_empty()).then_some(items)
}

/// The effective debug config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDebugConfig {
    pub enabled: bool,
    pub level: HarnessDebugLevel,
    pub subsystems: Option<Vec<String>>,
}

/// `resolveDebugConfig(debug, env)`: explicit settings, with `HARNESS_DEBUG*`
/// filling unset fields.
///
/// [rust] `env` reads one variable. An unknown `HARNESS_DEBUG_LEVEL` falls
/// back to `debug`; TS passes the string through.
pub fn resolve_debug_config(
    debug: Option<&HarnessDebugConfig>,
    env: &dyn Fn(&str) -> Option<String>,
) -> ResolvedDebugConfig {
    let enabled = debug.and_then(|d| d.enabled).unwrap_or_else(|| {
        let value = env("HARNESS_DEBUG").unwrap_or_default().to_lowercase();
        ENV_TRUTHY.contains(&value.as_str())
    });
    let level = debug
        .and_then(|d| d.level)
        .or_else(|| {
            env("HARNESS_DEBUG_LEVEL")
                .and_then(|l| serde_json::from_value(serde_json::Value::String(l)).ok())
        })
        .unwrap_or(HarnessDebugLevel::Debug);
    let subsystems = match debug.and_then(|d| d.subsystems.clone()) {
        Some(s) => Some(s),
        None => parse_list(env("HARNESS_DEBUG_SUBSYSTEMS")),
    };
    ResolvedDebugConfig {
        enabled,
        level,
        subsystems,
    }
}

fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn to_v1_level(level: HarnessDebugLevel) -> HarnessV1DebugLevel {
    match level {
        HarnessDebugLevel::Error => HarnessV1DebugLevel::Error,
        HarnessDebugLevel::Warn => HarnessV1DebugLevel::Warn,
        HarnessDebugLevel::Info => HarnessV1DebugLevel::Info,
        HarnessDebugLevel::Debug => HarnessV1DebugLevel::Debug,
        HarnessDebugLevel::Trace => HarnessV1DebugLevel::Trace,
    }
}

fn from_v1_level(level: HarnessV1DebugLevel) -> HarnessDebugLevel {
    match level {
        HarnessV1DebugLevel::Error => HarnessDebugLevel::Error,
        HarnessV1DebugLevel::Warn => HarnessDebugLevel::Warn,
        HarnessV1DebugLevel::Info => HarnessDebugLevel::Info,
        HarnessV1DebugLevel::Debug => HarnessDebugLevel::Debug,
        HarnessV1DebugLevel::Trace => HarnessDebugLevel::Trace,
    }
}

/// `toHarnessDiagnostic`.
fn to_harness_diagnostic(d: HarnessV1Diagnostic) -> HarnessDiagnostic {
    HarnessDiagnostic {
        level: from_v1_level(d.level),
        message: d.message,
        subsystem: d.subsystem,
        kind: match d.kind {
            HarnessV1DiagnosticKind::Log => HarnessDiagnosticKind::Log,
            HarnessV1DiagnosticKind::Event => HarnessDiagnosticKind::Event,
        },
        source: d.source,
        stream: d.stream.map(|s| match s {
            HarnessV1LogStream::Stdout => HarnessDiagnosticStream::Stdout,
            HarnessV1LogStream::Stderr => HarnessDiagnosticStream::Stderr,
        }),
        attrs: d.attrs,
        error: d.error.map(|e| HarnessDiagnosticErrorInfo {
            name: e.name,
            message: e.message,
            stack: e.stack,
        }),
        session_id: d.session_id,
        timestamp: d.timestamp,
    }
}

/// `formatForStderr`.
fn format_for_stderr(d: &HarnessDiagnostic) -> String {
    let level = format!("[harness:{}]", d.level.as_str());
    let mut line = [level.as_str(), d.subsystem.as_str(), d.message.as_str()]
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if let Some(error) = &d.error {
        line.push_str(&format!(
            " ({}: {})",
            error.name.as_deref().unwrap_or("Error"),
            error.message
        ));
    }
    line.push('\n');
    line
}

/// `buildObservability({ settings: { debug, onLog, telemetry } })`: the
/// per-session observability handle for `doStart`, or `None` when
/// diagnostics are disabled.
pub fn build_observability(
    debug: Option<&HarnessDebugConfig>,
    on_log: Option<OnLogFn>,
    telemetry: Option<&TelemetryOptions>,
) -> Option<HarnessV1Observability> {
    build_observability_with_env(debug, on_log, telemetry, &process_env)
}

pub(crate) fn build_observability_with_env(
    debug: Option<&HarnessDebugConfig>,
    on_log: Option<OnLogFn>,
    telemetry: Option<&TelemetryOptions>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Option<HarnessV1Observability> {
    let resolved = resolve_debug_config(debug, env);
    if !resolved.enabled {
        return None;
    }
    let integrations = telemetry
        .and_then(|t| t.integrations.clone())
        .unwrap_or_default();
    let report = Arc::new(move |emitted: HarnessV1Diagnostic| {
        let diagnostic = to_harness_diagnostic(emitted);
        // Never let the stderr default break the turn.
        let _ = std::io::stderr().write_all(format_for_stderr(&diagnostic).as_bytes());
        if let Some(on_log) = &on_log {
            on_log(&diagnostic);
        }
        for integration in &integrations {
            if let Some(consumer) = integration.as_diagnostic_consumer() {
                consumer.ingest_diagnostic(&diagnostic);
            }
        }
    });
    Some(HarnessV1Observability {
        debug: Some(HarnessV1DebugConfig {
            enabled: Some(true),
            level: Some(to_v1_level(resolved.level)),
            subsystems: resolved.subsystems,
        }),
        report: Some(report),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    fn env(vars: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn env_vars_fill_only_unset_fields() {
        let vars = env(&[
            ("HARNESS_DEBUG", "Yes"),
            ("HARNESS_DEBUG_LEVEL", "trace"),
            ("HARNESS_DEBUG_SUBSYSTEMS", " bridge, ,sandbox "),
        ]);
        assert_eq!(
            resolve_debug_config(None, &vars),
            ResolvedDebugConfig {
                enabled: true,
                level: HarnessDebugLevel::Trace,
                subsystems: Some(vec!["bridge".into(), "sandbox".into()]),
            }
        );
        let explicit = HarnessDebugConfig {
            enabled: Some(false),
            level: Some(HarnessDebugLevel::Warn),
            subsystems: Some(vec![]),
        };
        assert_eq!(
            resolve_debug_config(Some(&explicit), &vars),
            ResolvedDebugConfig {
                enabled: false,
                level: HarnessDebugLevel::Warn,
                subsystems: Some(vec![]),
            }
        );
        assert_eq!(
            resolve_debug_config(None, &env(&[])),
            ResolvedDebugConfig {
                enabled: false,
                level: HarnessDebugLevel::Debug,
                subsystems: None,
            }
        );
    }

    #[test]
    fn disabled_diagnostics_build_no_observability() {
        assert!(build_observability_with_env(None, None, None, &env(&[])).is_none());
    }

    #[test]
    fn report_maps_and_fans_out_to_on_log() {
        let seen: Arc<Mutex<Vec<HarnessDiagnostic>>> = Arc::default();
        let sink = seen.clone();
        let observability = build_observability_with_env(
            Some(&HarnessDebugConfig {
                enabled: Some(true),
                ..Default::default()
            }),
            Some(Arc::new(move |d: &HarnessDiagnostic| {
                sink.lock().push(d.clone())
            })),
            None,
            &env(&[]),
        )
        .unwrap();
        assert_eq!(
            observability.debug,
            Some(HarnessV1DebugConfig {
                enabled: Some(true),
                level: Some(HarnessV1DebugLevel::Debug),
                subsystems: None,
            })
        );
        (observability.report.unwrap())(HarnessV1Diagnostic {
            level: HarnessV1DebugLevel::Warn,
            message: "hello".into(),
            subsystem: "bridge".into(),
            kind: HarnessV1DiagnosticKind::Log,
            source: None,
            stream: Some(HarnessV1LogStream::Stderr),
            attrs: None,
            error: None,
            session_id: Some("s1".into()),
            timestamp: 1,
        });
        let seen = seen.lock();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].level, HarnessDebugLevel::Warn);
        assert_eq!(seen[0].stream, Some(HarnessDiagnosticStream::Stderr));
        assert_eq!(seen[0].session_id.as_deref(), Some("s1"));
    }

    #[test]
    fn formats_stderr_lines_with_the_error() {
        let d = HarnessDiagnostic {
            level: HarnessDebugLevel::Error,
            message: "failed".into(),
            subsystem: String::new(),
            kind: HarnessDiagnosticKind::Event,
            source: None,
            stream: None,
            attrs: None,
            error: Some(HarnessDiagnosticErrorInfo {
                name: None,
                message: "boom".into(),
                stack: None,
            }),
            session_id: None,
            timestamp: 0,
        };
        assert_eq!(
            format_for_stderr(&d),
            "[harness:error] failed (Error: boom)\n"
        );
    }
}
