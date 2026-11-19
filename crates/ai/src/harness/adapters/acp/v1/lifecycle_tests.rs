//! Port of `harness-acp/src/v1/acp-v1-lifecycle.test.ts`.

use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::harness::adapters::acp::v1::settings::{
    ACPInstallCommandSource, ACPModelMapping, ACPNonTextContentType, ACPNpmLockedSource,
    ACPPermissionModeMapping, ACPPermissionModeTarget, ACPSource,
};

fn locked_source() -> ACPNpmLockedSource {
    ACPNpmLockedSource {
        package_json: "{\"dependencies\":{\"@example/acp-agent\":\"1.2.3\"}}\n".into(),
        pnpm_lock_yaml: "lockfileVersion: '9.0'\n".into(),
        pnpm_workspace_yaml: None,
    }
}

fn strings(values: &[&str]) -> Option<Vec<String>> {
    Some(values.iter().map(|value| value.to_string()).collect())
}

fn base_settings() -> ACPV1Settings {
    let mut settings = ACPV1Settings::new(
        "example-acp",
        ACPSource::NpmLocked(locked_source()),
        "acp-agent",
        ACPModelMapping::SessionConfigOption {
            path: "model".into(),
        },
        Arc::new(|_| Ok(false)),
        Arc::new(|_| Ok(None)),
    );
    settings.args = strings(&["stdio"]);
    settings.forward_env = strings(&["PROVIDER_API_KEY", "SECOND_PROVIDER_API_KEY"]);
    settings.env = Some(IndexMap::from([(
        "PROVIDER_BASE_URL".to_string(),
        "https://provider.example".to_string(),
    )]));
    settings
}

fn default_client_app() -> ACPClientApp {
    ACPClientApp {
        name: "ai-sdk-harness-acp".into(),
        version: "0.0.0-test".into(),
    }
}

fn identity_with(
    configure: impl FnOnce(&mut ACPV1Settings),
    client_app: Option<ACPClientApp>,
    provider_authentication: Option<&ACPProviderAuthenticationCompatibility>,
) -> String {
    let mut settings = base_settings();
    configure(&mut settings);
    create_acp_agent_identity(
        &settings,
        &client_app.unwrap_or_else(default_client_app),
        provider_authentication,
    )
}

fn identity(configure: impl FnOnce(&mut ACPV1Settings)) -> String {
    identity_with(configure, None, None)
}

fn base_identity() -> String {
    identity(|_| {})
}

// describe('createACPAgentIdentity')

#[test]
fn ignores_the_order_and_duplication_of_environment_key_lists() {
    assert_eq!(
        identity(|s| {
            s.forward_env = strings(&[
                "SECOND_PROVIDER_API_KEY",
                "PROVIDER_API_KEY",
                "PROVIDER_API_KEY",
            ])
        }),
        base_identity()
    );
    assert_eq!(
        identity(|s| s.non_text_content_types = Some(vec![])),
        base_identity()
    );
    assert_eq!(
        identity(|s| s.non_text_content_types = Some(vec![ACPNonTextContentType::Image])),
        identity(|s| {
            s.non_text_content_types = Some(vec![
                ACPNonTextContentType::Image,
                ACPNonTextContentType::Image,
            ])
        })
    );
}

#[test]
fn identifies_every_non_secret_acquisition_and_launch_input() {
    let base = base_identity();
    let session_mode = |mode: &str| {
        Some(ACPPermissionModeTarget::SessionMode {
            mode_id: mode.into(),
        })
    };
    let variants: Vec<String> = vec![
        identity(|s| s.harness_id = "other-acp".into()),
        identity(|s| {
            s.source = ACPSource::InstallCommand(ACPInstallCommandSource {
                command: "curl https://example.com/install -fsS | bash".into(),
            })
        }),
        identity(|s| s.executable = "other-agent".into()),
        identity(|s| s.args = strings(&["other"])),
        identity(|s| s.forward_env = strings(&["OTHER_PROVIDER_API_KEY"])),
        identity(|s| {
            s.forward_env = strings(&["SECOND_PROVIDER_API_KEY"]);
            s.credential_env = strings(&["PROVIDER_API_KEY"]);
        }),
        identity(|s| {
            s.env = Some(IndexMap::from([(
                "PROVIDER_BASE_URL".to_string(),
                "https://other.example".to_string(),
            )]))
        }),
        identity(|s| {
            s.client_capabilities = Some(IndexMap::from([(
                "_meta".to_string(),
                json!({ "parameterizedModelPicker": true }),
            )]))
        }),
        identity(|s| {
            s.model_mapping = ACPModelMapping::SessionModel {
                path: "modelId".into(),
            }
        }),
        identity(|s| {
            s.permission_mode_mapping = Some(ACPPermissionModeMapping {
                allow_reads: session_mode("read-only"),
                allow_edits: session_mode("agent"),
                allow_all: session_mode("full-access"),
            })
        }),
        identity(|s| s.non_text_content_types = Some(vec![ACPNonTextContentType::Image])),
        identity_with(
            |_| {},
            Some(ACPClientApp {
                name: "custom-client".into(),
                version: "4.5.6".into(),
            }),
            None,
        ),
    ];
    for variant in variants {
        assert_ne!(variant, base);
    }
}

#[test]
fn identifies_every_locked_acquisition_artifact_independently() {
    let base = base_identity();
    for source in [
        ACPNpmLockedSource {
            package_json: "{\"dependencies\":{}}\n".into(),
            ..locked_source()
        },
        ACPNpmLockedSource {
            pnpm_lock_yaml: "lockfileVersion: '9.0'\n# changed\n".into(),
            ..locked_source()
        },
        ACPNpmLockedSource {
            pnpm_workspace_yaml: Some("allowBuilds:\n  '@example/acp-agent': true\n".into()),
            ..locked_source()
        },
    ] {
        assert_ne!(identity(|s| s.source = ACPSource::NpmLocked(source)), base);
    }
}

#[test]
fn identifies_non_secret_provider_routing_without_credential_values() {
    let gateway = |patch: serde_json::Value| -> ACPProviderAuthenticationCompatibility {
        let mut value = json!({
            "type": "ai-gateway",
            "mode": "ai-gateway",
            "env": { "PROVIDER_API_KEY": { "$source": "gateway-api-key" } },
            "credentialSource": "AI_GATEWAY_API_KEY",
            "baseUrl": "https://gateway.example",
        });
        for (key, entry) in patch.as_object().unwrap() {
            value[key] = entry.clone();
        }
        serde_json::from_value(value).unwrap()
    };
    let gateway_identity = identity_with(|_| {}, None, Some(&gateway(json!({}))));
    assert_ne!(gateway_identity, base_identity());
    assert_ne!(
        identity_with(
            |_| {},
            None,
            Some(&gateway(json!({ "credentialSource": "VERCEL_OIDC_TOKEN" })))
        ),
        gateway_identity
    );
    assert_ne!(
        identity_with(
            |_| {},
            None,
            Some(&gateway(
                json!({ "baseUrl": "https://other-gateway.example" })
            ))
        ),
        gateway_identity
    );
}

// describe('ACP prompt guidance lifecycle')

fn guidance(applied: Option<bool>) -> ACPPromptGuidanceLifecycleState {
    ACPPromptGuidanceLifecycleState {
        initial_guidance_applied: applied,
        ..Default::default()
    }
}

#[test]
fn applies_guidance_to_a_fresh_session() {
    assert!(!resolve_acp_initial_guidance_applied(
        false,
        Some(&guidance(Some(true)))
    ));
}

#[test]
fn does_not_reannounce_guidance_when_a_legacy_resume_state_lacks_the_flag() {
    assert!(resolve_acp_initial_guidance_applied(true, None));
}

#[test]
fn preserves_an_explicit_resume_guidance_flag() {
    assert!(!resolve_acp_initial_guidance_applied(
        true,
        Some(&guidance(Some(false)))
    ));
}

// describe('validateACPLifecycleCompatibility')

fn authentication_profile(digest: &str) -> ACPAuthenticationProfileIdentity {
    serde_json::from_value(json!({
        "digest": digest,
        "providerKind": "direct",
        "providerMode": "direct",
    }))
    .unwrap()
}

fn lifecycle_data() -> ACPLifecycleData {
    ACPLifecycleData {
        authentication_profile: Some(authentication_profile("authentication-profile")),
        bridge: Some(ACPBridgeCoordinates {
            port: 4319,
            private_key: "bridge-private-key".into(),
            last_seen_event_id: 7,
            sandbox_id: Some("sandbox-1".into()),
            state_dir: None,
        }),
        ..ACPLifecycleData::new("agent")
    }
}

fn validate(
    lifecycle_harness_id: &str,
    agent_identity: &str,
    digest: &str,
    sandbox_id: &str,
) -> HarnessResult<()> {
    validate_acp_lifecycle_compatibility(
        "portable-acp",
        lifecycle_harness_id,
        agent_identity,
        &authentication_profile(digest),
        &lifecycle_data(),
        Some(sandbox_id),
    )
}

#[test]
fn accepts_matching_harness_agent_auth_and_sandbox_identities() {
    validate(
        "portable-acp",
        "agent",
        "authentication-profile",
        "sandbox-1",
    )
    .unwrap();
}

#[test]
fn rejects_mismatched_identities_before_attach() {
    let cases = [
        (
            "harness",
            validate("other-acp", "agent", "authentication-profile", "sandbox-1"),
            "produced by harness",
        ),
        (
            "agent",
            validate(
                "portable-acp",
                "other-agent",
                "authentication-profile",
                "sandbox-1",
            ),
            "configured agent",
        ),
        (
            "authentication profile",
            validate("portable-acp", "agent", "other-authentication", "sandbox-1"),
            "authentication profile",
        ),
        (
            "sandbox",
            validate(
                "portable-acp",
                "agent",
                "authentication-profile",
                "sandbox-2",
            ),
            "belongs to sandbox",
        ),
    ];
    for (name, result, message) in cases {
        let error = result.expect_err(name).to_string();
        assert!(error.contains(message), "{name}: {error}");
    }
}
