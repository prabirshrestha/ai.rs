//! Port of `harness-acp/src/acp-auth.test.ts`.

use indexmap::IndexMap;
use serde_json::{Value, json};

use super::*;
use crate::harness::adapters::acp::v1::settings::{
    ACPGatewayProviderAuthentication, ACPProviderAuthentication,
};

fn client_app() -> ACPClientApp {
    ACPClientApp {
        name: "ai-sdk-harness-acp".into(),
        version: "0.0.0-test".into(),
    }
}

fn gateway_env() -> IndexMap<String, JsonValue> {
    IndexMap::from([
        (
            "PROVIDER_API_KEY".into(),
            json!({ "$source": "gateway-api-key" }),
        ),
        (
            "PROVIDER_BASE_URL".into(),
            json!({ "$source": "gateway-base-url" }),
        ),
    ])
}

fn provider_authentication_with(env: IndexMap<String, JsonValue>) -> ACPProviderAuthentication {
    ACPProviderAuthentication {
        gateway: ACPGatewayProviderAuthentication { env },
    }
}

fn provider_authentication() -> ACPProviderAuthentication {
    provider_authentication_with(gateway_env())
}

fn env(entries: &[(&str, &str)]) -> IndexMap<String, String> {
    entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn mode(value: Value) -> ACPAuthenticationMode {
    serde_json::from_value(value).unwrap()
}

fn client_app_env() -> Vec<(&'static str, &'static str)> {
    vec![
        ("AI_SDK_ACP_CLIENT_APP_NAME", "ai-sdk-harness-acp"),
        ("AI_SDK_ACP_CLIENT_APP_VERSION", "0.0.0-test"),
    ]
}

fn with_client_app(entries: &[(&str, &str)]) -> IndexMap<String, String> {
    let mut all = entries.to_vec();
    all.extend(client_app_env());
    env(&all)
}

fn resolve_with(
    mode: Option<&ACPAuthenticationMode>,
    host_env: &IndexMap<String, String>,
    compatibility: Option<&ACPProviderAuthenticationCompatibility>,
) -> HarnessResult<ACPResolvedProviderAuthenticationResult> {
    let provider_authentication = provider_authentication();
    let client_app = client_app();
    resolve_acp_provider_authentication(
        ACPAuthResolutionOptions {
            mode,
            provider_authentication: Some(&provider_authentication),
            client_app: &client_app,
        },
        host_env,
        compatibility,
    )
}

fn resolve(
    mode: Option<&ACPAuthenticationMode>,
    host_env: &[(&str, &str)],
) -> HarnessResult<ACPResolvedProviderAuthenticationResult> {
    resolve_with(mode, &env(host_env), None)
}

fn gateway_result(entries: &[(&str, &str)]) -> ACPResolvedProviderAuthenticationResult {
    ACPResolvedProviderAuthenticationResult {
        provider_authentication: Some(ACPResolvedProviderAuthentication::AiGateway {
            env: gateway_env(),
        }),
        env: with_client_app(entries),
    }
}

fn direct_result() -> ACPResolvedProviderAuthenticationResult {
    ACPResolvedProviderAuthenticationResult {
        provider_authentication: Some(ACPResolvedProviderAuthentication::Direct),
        env: with_client_app(&[]),
    }
}

fn compatibility_for(
    auth: Option<&ACPAuthenticationMode>,
    provider_authentication: &ACPProviderAuthentication,
    host_env: &[(&str, &str)],
) -> Option<ACPProviderAuthenticationCompatibility> {
    resolve_acp_provider_authentication_compatibility(
        auth,
        Some(provider_authentication),
        &env(host_env),
    )
}

fn assert_gateway_compatibility(
    compatibility: &Option<ACPProviderAuthenticationCompatibility>,
    expected_source: ACPGatewayCredentialSource,
    expected_base_url: Option<&str>,
) {
    let Some(ACPProviderAuthenticationCompatibility::AiGateway {
        credential_source,
        base_url,
        ..
    }) = compatibility
    else {
        panic!("expected ai-gateway compatibility, got {compatibility:?}");
    };
    assert_eq!(*credential_source, Some(expected_source));
    if let Some(expected) = expected_base_url {
        assert_eq!(base_url, expected);
    }
}

// describe('resolveACPProviderAuthentication')

/// [rust] A typed mode rejects the nested object when it deserializes; the
/// untyped guard reports the TS message.
#[test]
fn rejects_nested_authentication_objects_before_reading_ambient_credentials() {
    let nested = json!({ "gateway": { "apiKey": "legacy-key" } });
    assert!(serde_json::from_value::<ACPAuthenticationMode>(nested.clone()).is_err());
    assert_eq!(
        crate::harness::utils::is_harness_authentication_environment(Some(&nested))
            .unwrap_err()
            .to_string(),
        "Invalid auth: expected an authentication mode or a flat record with string values."
    );
}

#[test]
fn transports_the_client_app_without_provider_authentication_configuration() {
    let client_app = client_app();
    assert_eq!(
        resolve_acp_provider_authentication(
            ACPAuthResolutionOptions {
                mode: None,
                provider_authentication: None,
                client_app: &client_app,
            },
            &IndexMap::new(),
            None,
        )
        .unwrap(),
        ACPResolvedProviderAuthenticationResult {
            provider_authentication: None,
            env: with_client_app(&[]),
        }
    );
}

#[test]
fn selects_ai_gateway_automatically_for_ai_gateway_api_key() {
    assert_eq!(
        resolve(
            None,
            &[
                ("AI_GATEWAY_API_KEY", "gateway-key"),
                ("AI_GATEWAY_BASE_URL", "https://gateway.example"),
            ]
        )
        .unwrap(),
        gateway_result(&[
            ("AI_SDK_ACP_GATEWAY_API_KEY", "gateway-key"),
            ("AI_SDK_ACP_GATEWAY_BASE_URL", "https://gateway.example"),
        ])
    );
}

#[test]
fn selects_ai_gateway_automatically_for_vercel_oidc_token() {
    assert_eq!(
        resolve(None, &[("VERCEL_OIDC_TOKEN", "oidc-token")]).unwrap(),
        gateway_result(&[
            ("AI_SDK_ACP_GATEWAY_API_KEY", "oidc-token"),
            (
                "AI_SDK_ACP_GATEWAY_BASE_URL",
                "https://ai-gateway.vercel.sh"
            ),
        ])
    );
}

#[test]
fn selects_direct_authentication_automatically_without_gateway_credentials() {
    assert_eq!(
        resolve(None, &[("DIRECT_PROVIDER_KEY", "direct-key")]).unwrap(),
        direct_result()
    );
}

#[test]
fn keeps_explicit_direct_authentication_when_gateway_credentials_are_ambient() {
    assert_eq!(
        resolve(
            Some(&mode(json!("direct"))),
            &[("AI_GATEWAY_API_KEY", "ambient-gateway-key")]
        )
        .unwrap(),
        direct_result()
    );
}

#[test]
fn uses_the_gateway_base_url_from_the_environment() {
    assert_eq!(
        resolve(
            Some(&mode(json!("ai-gateway"))),
            &[
                ("AI_GATEWAY_API_KEY", "ambient-gateway-key"),
                ("AI_GATEWAY_BASE_URL", "https://configured.example"),
            ]
        )
        .unwrap()
        .env,
        with_client_app(&[
            ("AI_SDK_ACP_GATEWAY_API_KEY", "ambient-gateway-key"),
            ("AI_SDK_ACP_GATEWAY_BASE_URL", "https://configured.example"),
        ])
    );
}

#[test]
fn uses_a_supplied_authentication_environment_without_reading_the_host_environment() {
    let host_env = env(&[("AI_GATEWAY_API_KEY", "ambient-gateway-key")]);
    let supplied = mode(json!({
        "AI_GATEWAY_API_KEY": "explicit-gateway-key",
        "AI_GATEWAY_BASE_URL": "https://explicit.example",
    }));
    assert_eq!(
        resolve_with(Some(&supplied), &host_env, None).unwrap().env,
        with_client_app(&[
            ("AI_SDK_ACP_GATEWAY_API_KEY", "explicit-gateway-key"),
            ("AI_SDK_ACP_GATEWAY_BASE_URL", "https://explicit.example"),
        ])
    );
    assert_eq!(
        host_env,
        env(&[("AI_GATEWAY_API_KEY", "ambient-gateway-key")])
    );
}

#[test]
fn fails_explicit_gateway_selection_without_credentials() {
    let error = resolve(Some(&mode(json!("ai-gateway"))), &[]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("neither AI_GATEWAY_API_KEY nor VERCEL_OIDC_TOKEN is set"),
        "{error}"
    );
}

// describe('resolveACPEnv')

#[test]
fn returns_the_environment_for_the_resolved_selection() {
    let provider_authentication = provider_authentication();
    let client_app = client_app();
    let gateway = mode(json!("ai-gateway"));
    assert_eq!(
        resolve_acp_env(
            ACPAuthResolutionOptions {
                mode: Some(&gateway),
                provider_authentication: Some(&provider_authentication),
                client_app: &client_app,
            },
            &env(&[("AI_GATEWAY_API_KEY", "gateway-key")]),
        )
        .unwrap(),
        with_client_app(&[
            ("AI_SDK_ACP_GATEWAY_API_KEY", "gateway-key"),
            (
                "AI_SDK_ACP_GATEWAY_BASE_URL",
                "https://ai-gateway.vercel.sh"
            ),
        ])
    );
}

// describe('resolveACPProviderAuthenticationCompatibility')

#[test]
fn records_credential_source_identity_without_credential_values() {
    let provider_authentication = provider_authentication();
    let first = compatibility_for(
        None,
        &provider_authentication,
        &[
            ("AI_GATEWAY_API_KEY", "first-secret"),
            ("AI_GATEWAY_BASE_URL", "https://gateway.example"),
        ],
    );
    let rotated = compatibility_for(
        None,
        &provider_authentication,
        &[
            ("AI_GATEWAY_API_KEY", "rotated-secret"),
            ("AI_GATEWAY_BASE_URL", "https://gateway.example"),
        ],
    );
    let oidc = compatibility_for(
        None,
        &provider_authentication,
        &[
            ("VERCEL_OIDC_TOKEN", "oidc-secret"),
            ("AI_GATEWAY_BASE_URL", "https://gateway.example"),
        ],
    );

    assert_eq!(first, rotated);
    assert_gateway_compatibility(
        &first,
        ACPGatewayCredentialSource::AiGatewayApiKey,
        Some("https://gateway.example"),
    );
    assert_gateway_compatibility(&oidc, ACPGatewayCredentialSource::VercelOidcToken, None);
    assert!(
        !serde_json::to_string(&first)
            .unwrap()
            .contains("first-secret")
    );
    assert!(
        !serde_json::to_string(&oidc)
            .unwrap()
            .contains("oidc-secret")
    );
}

#[test]
fn records_the_environment_gateway_source_and_effective_base_url() {
    let configured = compatibility_for(
        Some(&mode(json!("ai-gateway"))),
        &provider_authentication(),
        &[
            ("AI_GATEWAY_API_KEY", "ambient-secret"),
            ("AI_GATEWAY_BASE_URL", "https://ambient.example"),
        ],
    );
    assert_gateway_compatibility(
        &configured,
        ACPGatewayCredentialSource::AiGatewayApiKey,
        Some("https://ambient.example"),
    );
    assert!(
        !serde_json::to_string(&configured)
            .unwrap()
            .contains("ambient-secret")
    );
}

#[test]
fn excludes_supplied_credentials_from_compatibility_identity() {
    let provider_authentication = provider_authentication();
    let first = compatibility_for(
        Some(&mode(json!({
            "AI_GATEWAY_API_KEY": "first-secret",
            "AI_GATEWAY_BASE_URL": "https://gateway.example",
        }))),
        &provider_authentication,
        &[],
    );
    let rotated = compatibility_for(
        Some(&mode(json!({
            "AI_GATEWAY_API_KEY": "rotated-secret",
            "AI_GATEWAY_BASE_URL": "https://gateway.example",
        }))),
        &provider_authentication,
        &[],
    );
    assert_eq!(first, rotated);
    assert!(matches!(
        first,
        Some(ACPProviderAuthenticationCompatibility::AiGateway {
            mode: ACPGatewayProviderMode::Auto,
            ..
        })
    ));
    assert_gateway_compatibility(
        &first,
        ACPGatewayCredentialSource::AiGatewayApiKey,
        Some("https://gateway.example"),
    );
    assert!(
        !serde_json::to_string(&first)
            .unwrap()
            .contains("first-secret")
    );
}

#[test]
fn pins_selection_source_and_base_url_while_allowing_secret_rotation() {
    let compatibility = compatibility_for(
        None,
        &provider_authentication(),
        &[
            ("AI_GATEWAY_API_KEY", "first-secret"),
            ("AI_GATEWAY_BASE_URL", "https://first.example"),
        ],
    );
    assert!(matches!(
        compatibility,
        Some(ACPProviderAuthenticationCompatibility::AiGateway { .. })
    ));

    assert_eq!(
        resolve_with(
            None,
            &env(&[
                ("AI_GATEWAY_API_KEY", "rotated-secret"),
                ("AI_GATEWAY_BASE_URL", "https://changed.example"),
            ]),
            compatibility.as_ref(),
        )
        .unwrap(),
        gateway_result(&[
            ("AI_SDK_ACP_GATEWAY_API_KEY", "rotated-secret"),
            ("AI_SDK_ACP_GATEWAY_BASE_URL", "https://first.example"),
        ])
    );
    let error = resolve_with(
        None,
        &env(&[("VERCEL_OIDC_TOKEN", "different-source-secret")]),
        compatibility.as_ref(),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("AI Gateway authentication was selected"),
        "{error}"
    );
}

#[test]
fn does_not_change_an_auto_direct_identity_when_credentials_appear_later() {
    let compatibility = compatibility_for(None, &provider_authentication(), &[]);
    assert_eq!(
        resolve_with(
            None,
            &env(&[("AI_GATEWAY_API_KEY", "late-secret")]),
            compatibility.as_ref(),
        )
        .unwrap(),
        direct_result()
    );
}

// describe('createACPAuthenticationProfileIdentity')

#[test]
fn persists_only_named_selection_facts_and_a_digest() {
    let mut gateway = gateway_env();
    gateway.insert(
        "AUTHENTICATED_HEADER".into(),
        json!("authenticated-header-secret"),
    );
    let compatibility = compatibility_for(
        Some(&mode(json!("ai-gateway"))),
        &provider_authentication_with(gateway),
        &[("VERCEL_OIDC_TOKEN", "resolved-oidc-secret")],
    );
    let authentication: ACPAuthentication = serde_json::from_value(json!({
        "methodId": "api-key",
        "meta": { "token": "acp-authentication-secret" },
        "clientCapabilities": { "headers": { "Authorization": "client-capability-secret" } },
    }))
    .unwrap();
    let identity =
        create_acp_authentication_profile_identity(Some(&authentication), compatibility.as_ref());
    let serialized = serde_json::to_string(&identity).unwrap();

    assert_eq!(identity.digest.len(), 64);
    assert!(
        identity
            .digest
            .chars()
            .all(|c| matches!(c, 'a'..='f' | '0'..='9'))
    );
    assert_eq!(identity.acp_method_id.as_deref(), Some("api-key"));
    assert_eq!(identity.provider_kind, ACPProviderKind::AiGateway);
    assert_eq!(identity.provider_mode, Some(ACPProviderMode::AiGateway));
    assert_eq!(
        identity.gateway_credential_source,
        Some(Some(ACPGatewayCredentialSource::VercelOidcToken))
    );
    for secret in [
        "resolved-oidc-secret",
        "authenticated-header-secret",
        "acp-authentication-secret",
        "client-capability-secret",
        "Authorization",
    ] {
        assert!(
            !serialized.contains(secret),
            "{serialized} contains {secret}"
        );
    }
}

#[test]
fn keeps_its_digest_stable_across_secret_rotation_at_the_same_source() {
    let provider_authentication = provider_authentication();
    let first = compatibility_for(
        None,
        &provider_authentication,
        &[("AI_GATEWAY_API_KEY", "first-secret")],
    );
    let rotated = compatibility_for(
        None,
        &provider_authentication,
        &[("AI_GATEWAY_API_KEY", "rotated-secret")],
    );
    let authentication = ACPAuthentication {
        method_id: "api-key".into(),
        meta: None,
        client_capabilities: None,
    };
    assert_eq!(
        create_acp_authentication_profile_identity(Some(&authentication), first.as_ref()),
        create_acp_authentication_profile_identity(Some(&authentication), rotated.as_ref())
    );
}
