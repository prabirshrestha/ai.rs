//! Doubles for the `acp-harness.test.ts` port, over the shared
//! [`bridge_test_support`](crate::harness::adapters::bridge_test_support).

use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::json;

pub use crate::harness::adapters::bridge_test_support::*;

use super::acp_harness::ACPHarnessSettings;
use super::v1::settings::{
    ACPModelMapping, ACPNpmLockedSource, ACPPermissionModeMapping, ACPPermissionModeTarget,
    ACPSource, ACPV1Settings,
};

pub const LOCKED_PACKAGE_JSON: &str = r#"{
  "name": "locked-codex-acp",
  "private": true,
  "dependencies": {
    "@agentclientprotocol/codex-acp": "1.1.4"
  }
}
"#;

pub const LOCKED_PNPM_LOCK_YAML: &str = "lockfileVersion: '9.0'
importers:
  .:
    dependencies:
      '@agentclientprotocol/codex-acp':
        specifier: 1.1.4
        version: 1.1.4
";

/// `agentSettings`.
pub fn agent_settings(harness_id: &str) -> ACPV1Settings {
    let mut settings = ACPV1Settings::new(
        harness_id,
        ACPSource::NpmLocked(ACPNpmLockedSource {
            package_json: LOCKED_PACKAGE_JSON.into(),
            pnpm_lock_yaml: LOCKED_PNPM_LOCK_YAML.into(),
            pnpm_workspace_yaml: None,
        }),
        "codex-acp",
        ACPModelMapping::SessionConfigOption {
            path: "model".into(),
        },
        Arc::new(|_| Ok(false)),
        Arc::new(|_| Ok(None)),
    );
    settings.args = Some(vec!["--example".into()]);
    settings.forward_env = Some(vec!["CODEX_API_KEY".into()]);
    settings
}

/// `permissionModeMapping`.
pub fn permission_mode_mapping() -> ACPPermissionModeMapping {
    let mode = |mode_id: &str| {
        Some(ACPPermissionModeTarget::SessionMode {
            mode_id: mode_id.into(),
        })
    };
    ACPPermissionModeMapping {
        allow_reads: mode("read-only"),
        allow_edits: mode("agent"),
        allow_all: mode("agent-full-access"),
    }
}

/// The `beforeEach` environment: `CODEX_API_KEY=test-key`.
pub fn test_env() -> IndexMap<String, String> {
    IndexMap::from([("CODEX_API_KEY".to_string(), "test-key".to_string())])
}

/// Harness settings over `settings` wired to `bridge` and [`test_env`].
pub fn harness_settings(settings: ACPV1Settings, bridge: &FakeBridge) -> ACPHarnessSettings {
    let mut harness_settings = ACPHarnessSettings::new(settings);
    harness_settings.web_socket_constructor = Some(bridge.constructor());
    harness_settings.process_env = Some(test_env());
    harness_settings
}

/// `emitColdRestoration({ channel, method, sessionId })`.
pub fn emit_cold_restoration(socket: &FakeSocket, method: &str, session_id: &str) {
    socket.emit(json!({ "type": "bridge-thread", "threadId": session_id }));
    socket.emit(json!({
        "type": "raw",
        "rawValue": { "type": "acp-session-restored", "method": method },
    }));
    socket.emit(json!({
        "type": "finish",
        "finishReason": { "unified": "stop", "raw": "acp-session-restored" },
        "totalUsage": unknown_usage(),
    }));
}
