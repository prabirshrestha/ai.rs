//! Port of `harness-acp/src/v1/acp-v1-lifecycle.ts`.

use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::harness::adapters::acp::acp_auth::{
    ACPAuthenticationProfileIdentity, ACPClientApp, ACPProviderAuthenticationCompatibility,
};
use crate::harness::agent::internal::sha256_hex;
use crate::harness::errors::HarnessResult;
use crate::harness::utils::plain_error;

use super::bridge::canonical_json_fingerprint::canonical_fingerprint;
use super::bridge_protocol::{ACPColdSessionState, ACPTurnStartConfig};
use super::settings::ACPV1Settings;

/// `ACPPromptGuidanceLifecycleState`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPPromptGuidanceLifecycleState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_guidance_applied: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills_directory: Option<String>,
}

/// `ACPBridgeCoordinates`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPBridgeCoordinates {
    pub port: u16,
    pub private_key: String,
    pub last_seen_event_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_dir: Option<String>,
}

/// `ACPLifecycleData['recovery']['mode']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ACPRecoveryStatusMode {
    DiskReplay,
    LossyRerun,
}

/// `ACPLifecycleData['recovery']`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ACPRecoveryStatus {
    pub mode: ACPRecoveryStatusMode,
    pub reason: String,
}

/// `ACPLifecycleData['restoration']['method']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ACPRestorationMethod {
    Resume,
    Load,
}

/// `ACPLifecycleData['restoration']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ACPRestoration {
    pub method: ACPRestorationMethod,
}

/// `ACPLifecycleData`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPLifecycleData {
    pub agent_identity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authentication_profile: Option<ACPAuthenticationProfileIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_credential_environment: Option<IndexMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acp_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge: Option<ACPBridgeCoordinates>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cold_session: Option<ACPColdSessionState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_start_config: Option<ACPTurnStartConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery: Option<ACPRecoveryStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restoration: Option<ACPRestoration>,
    #[serde(flatten)]
    pub guidance: ACPPromptGuidanceLifecycleState,
}

impl ACPLifecycleData {
    /// Data with only the agent identity.
    pub fn new(agent_identity: impl Into<String>) -> Self {
        Self {
            agent_identity: agent_identity.into(),
            authentication_profile: None,
            sandbox_credential_environment: None,
            acp_session_id: None,
            bridge: None,
            cold_session: None,
            turn_start_config: None,
            recovery: None,
            restoration: None,
            guidance: ACPPromptGuidanceLifecycleState::default(),
        }
    }
}

/// `resolveACPInitialGuidanceApplied`.
pub fn resolve_acp_initial_guidance_applied(
    is_resume: bool,
    lifecycle_state: Option<&ACPPromptGuidanceLifecycleState>,
) -> bool {
    if !is_resume {
        return false;
    }
    lifecycle_state
        .and_then(|state| state.initial_guidance_applied)
        .unwrap_or(true)
}

fn sorted_unique<T: Ord + Clone>(values: Option<&[T]>) -> Vec<T> {
    let mut values: Vec<T> = values
        .into_iter()
        .flatten()
        .cloned()
        .collect::<std::collections::BTreeSet<T>>()
        .into_iter()
        .collect();
    values.sort();
    values
}

fn sorted_unique_strings(values: Option<&[String]>) -> Vec<String> {
    let mut values: Vec<String> = values
        .into_iter()
        .flatten()
        .cloned()
        .collect::<IndexSet<String>>()
        .into_iter()
        .collect();
    values.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    values
}

/// `createACPAgentIdentity`: digest of every non-secret input that
/// determines how the ACP agent is installed, launched, and configured.
/// Lifecycle state produced under one identity cannot be resumed under
/// another because the agent's persisted session may depend on any of these
/// inputs.
pub fn create_acp_agent_identity(
    settings: &ACPV1Settings,
    client_app: &ACPClientApp,
    provider_authentication: Option<&ACPProviderAuthenticationCompatibility>,
) -> String {
    let payload = json!({
        "harnessId": settings.harness_id,
        "acpVersion": "v1",
        "source": settings.source,
        "executable": settings.executable,
        "args": settings.args.clone().unwrap_or_default(),
        "clientApp": client_app,
        "clientCapabilities": settings.client_capabilities,
        "modelMapping": settings.model_mapping,
        "environment": {
            "forwarded": sorted_unique_strings(settings.forward_env.as_deref()),
            "credential": sorted_unique_strings(settings.credential_env.as_deref()),
            "literal": settings.env.clone().unwrap_or_default(),
        },
        "providerAuthentication": provider_authentication,
        "permissionModeMapping": settings.permission_mode_mapping,
        "nonTextContentTypes": sorted_unique(settings.non_text_content_types.as_deref()),
    });
    sha256_hex(&canonical_fingerprint(&payload))
}

fn json_string(value: &str) -> String {
    Value::String(value.to_string()).to_string()
}

/// `validateACPLifecycleCompatibility`.
pub fn validate_acp_lifecycle_compatibility(
    harness_id: &str,
    lifecycle_harness_id: &str,
    agent_identity: &str,
    authentication_profile: &ACPAuthenticationProfileIdentity,
    lifecycle_data: &ACPLifecycleData,
    sandbox_id: Option<&str>,
) -> HarnessResult<()> {
    if lifecycle_harness_id != harness_id {
        return Err(plain_error(format!(
            "ACP lifecycle state was produced by harness {}, but this harness is {}.",
            json_string(lifecycle_harness_id),
            json_string(harness_id)
        )));
    }
    if lifecycle_data.agent_identity != agent_identity {
        return Err(plain_error(
            "ACP lifecycle state is incompatible with the configured agent.",
        ));
    }
    if lifecycle_data
        .authentication_profile
        .as_ref()
        .map(|profile| profile.digest.as_str())
        != Some(authentication_profile.digest.as_str())
    {
        return Err(plain_error(
            "ACP lifecycle state is incompatible with the configured authentication profile.",
        ));
    }
    if let (Some(expected), Some(actual)) = (
        lifecycle_data
            .bridge
            .as_ref()
            .and_then(|bridge| bridge.sandbox_id.as_deref()),
        sandbox_id,
    ) && expected != actual
    {
        return Err(plain_error(format!(
            "ACP lifecycle state belongs to sandbox {}, not {}.",
            json_string(expected),
            json_string(actual)
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod tests;
