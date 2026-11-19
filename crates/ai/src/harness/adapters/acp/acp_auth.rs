//! Port of `harness-acp/src/acp-auth.ts`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::harness::agent::internal::sha256_hex;
use crate::harness::errors::HarnessResult;
use crate::harness::provider::JsonValue;
use crate::harness::spec::authentication::{
    HarnessV1Authentication, HarnessV1DirectAuthentication,
};
use crate::harness::utils::get_ai_gateway_auth_from_env;
use crate::harness::utils::plain_error;

use super::v1::bridge::bridge_environment::ACPResolvedProviderAuthentication;
use super::v1::settings::{ACPAuthentication, ACPProviderAuthentication};

const DEFAULT_AI_GATEWAY_BASE_URL: &str = "https://ai-gateway.vercel.sh";

/// `ACPAuthenticationMode`: `HarnessV1Authentication`.
pub type ACPAuthenticationMode = HarnessV1Authentication;

/// `ACPClientApp`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ACPClientApp {
    pub name: String,
    pub version: String,
}

/// `ACPAuthResolutionOptions`.
#[derive(Debug, Clone, Copy)]
pub struct ACPAuthResolutionOptions<'a> {
    pub mode: Option<&'a ACPAuthenticationMode>,
    pub provider_authentication: Option<&'a ACPProviderAuthentication>,
    pub client_app: &'a ACPClientApp,
}

/// `ACPGatewayCredentialSource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ACPGatewayCredentialSource {
    #[serde(rename = "AI_GATEWAY_API_KEY")]
    AiGatewayApiKey,
    #[serde(rename = "VERCEL_OIDC_TOKEN")]
    VercelOidcToken,
}

/// `mode` of a direct [`ACPProviderAuthenticationCompatibility`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ACPDirectProviderMode {
    Auto,
    Direct,
}

/// `mode` of an AI Gateway [`ACPProviderAuthenticationCompatibility`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ACPGatewayProviderMode {
    Auto,
    AiGateway,
}

/// `ACPProviderAuthenticationCompatibility`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum ACPProviderAuthenticationCompatibility {
    Direct {
        mode: ACPDirectProviderMode,
    },
    AiGateway {
        mode: ACPGatewayProviderMode,
        env: IndexMap<String, JsonValue>,
        credential_source: Option<ACPGatewayCredentialSource>,
        base_url: String,
    },
}

/// `providerKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ACPProviderKind {
    Direct,
    AiGateway,
}

/// `providerMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ACPProviderMode {
    Auto,
    Direct,
    AiGateway,
}

/// `ACPAuthenticationProfileIdentity`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPAuthenticationProfileIdentity {
    pub digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acp_method_id: Option<String>,
    pub provider_kind: ACPProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_mode: Option<ACPProviderMode>,
    /// `gatewayCredentialSource?: … | null`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub gateway_credential_source: Option<Option<ACPGatewayCredentialSource>>,
}

/// Keeps `null` as `Some(None)`.
pub(crate) fn deserialize_present<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// `createACPAuthenticationProfileIdentity`.
pub fn create_acp_authentication_profile_identity(
    authentication: Option<&ACPAuthentication>,
    provider_authentication_compatibility: Option<&ACPProviderAuthenticationCompatibility>,
) -> ACPAuthenticationProfileIdentity {
    let provider_kind = match provider_authentication_compatibility {
        Some(ACPProviderAuthenticationCompatibility::AiGateway { .. }) => {
            ACPProviderKind::AiGateway
        }
        _ => ACPProviderKind::Direct,
    };
    let digest = sha256_hex(&stable_stringify(&serde_json::json!({
        "authentication": authentication,
        "providerAuthentication": provider_authentication_compatibility,
    })));
    ACPAuthenticationProfileIdentity {
        digest,
        acp_method_id: authentication.map(|a| a.method_id.clone()),
        provider_kind,
        provider_mode: provider_authentication_compatibility.map(|c| match c {
            ACPProviderAuthenticationCompatibility::Direct { mode } => match mode {
                ACPDirectProviderMode::Auto => ACPProviderMode::Auto,
                ACPDirectProviderMode::Direct => ACPProviderMode::Direct,
            },
            ACPProviderAuthenticationCompatibility::AiGateway { mode, .. } => match mode {
                ACPGatewayProviderMode::Auto => ACPProviderMode::Auto,
                ACPGatewayProviderMode::AiGateway => ACPProviderMode::AiGateway,
            },
        }),
        gateway_credential_source: match provider_authentication_compatibility {
            Some(ACPProviderAuthenticationCompatibility::AiGateway {
                credential_source, ..
            }) => Some(*credential_source),
            _ => None,
        },
    }
}

/// Result of [`resolve_acp_provider_authentication`].
#[derive(Debug, Clone, PartialEq)]
pub struct ACPResolvedProviderAuthenticationResult {
    pub provider_authentication: Option<ACPResolvedProviderAuthentication>,
    pub env: IndexMap<String, String>,
}

/// `resolveACPProviderAuthentication`.
pub fn resolve_acp_provider_authentication(
    auth: ACPAuthResolutionOptions<'_>,
    env: &IndexMap<String, String>,
    compatibility: Option<&ACPProviderAuthenticationCompatibility>,
) -> HarnessResult<ACPResolvedProviderAuthenticationResult> {
    let client_app_env = IndexMap::from([
        (
            "AI_SDK_ACP_CLIENT_APP_NAME".to_string(),
            auth.client_app.name.clone(),
        ),
        (
            "AI_SDK_ACP_CLIENT_APP_VERSION".to_string(),
            auth.client_app.version.clone(),
        ),
    ]);
    let resolved_env = resolve_acp_authentication_environment(auth.mode, env);
    let Some(provider_authentication) = auth.provider_authentication else {
        return Ok(ACPResolvedProviderAuthenticationResult {
            provider_authentication: None,
            env: client_app_env,
        });
    };
    let direct = || ACPResolvedProviderAuthenticationResult {
        provider_authentication: Some(ACPResolvedProviderAuthentication::Direct),
        env: client_app_env.clone(),
    };

    if matches!(
        compatibility,
        Some(ACPProviderAuthenticationCompatibility::Direct { .. })
    ) {
        return Ok(direct());
    }

    let mode = resolve_acp_provider_authentication_mode(auth.mode);
    if mode == ACPProviderMode::Direct {
        return Ok(direct());
    }

    let (api_key, base_url) = match compatibility {
        Some(ACPProviderAuthenticationCompatibility::AiGateway {
            credential_source,
            base_url,
            ..
        }) => (
            resolve_gateway_credential(&resolved_env, *credential_source),
            base_url.clone(),
        ),
        _ => {
            let gateway = get_ai_gateway_auth_from_env(&resolved_env);
            (gateway.api_key, gateway.base_url)
        }
    };
    if compatibility.is_none() && mode == ACPProviderMode::Auto && api_key.is_none() {
        return Ok(direct());
    }
    let Some(api_key) = api_key else {
        return Err(plain_error(
            "AI Gateway authentication was selected, but neither AI_GATEWAY_API_KEY nor VERCEL_OIDC_TOKEN is set.",
        ));
    };

    let mut resolved = IndexMap::from([
        ("AI_SDK_ACP_GATEWAY_API_KEY".to_string(), api_key),
        ("AI_SDK_ACP_GATEWAY_BASE_URL".to_string(), base_url),
    ]);
    resolved.extend(client_app_env);
    Ok(ACPResolvedProviderAuthenticationResult {
        provider_authentication: Some(ACPResolvedProviderAuthentication::AiGateway {
            env: match compatibility {
                Some(ACPProviderAuthenticationCompatibility::AiGateway { env, .. }) => env.clone(),
                _ => provider_authentication.gateway.env.clone(),
            },
        }),
        env: resolved,
    })
}

/// `resolveACPProviderAuthenticationCompatibility`.
pub fn resolve_acp_provider_authentication_compatibility(
    auth: Option<&ACPAuthenticationMode>,
    provider_authentication: Option<&ACPProviderAuthentication>,
    env: &IndexMap<String, String>,
) -> Option<ACPProviderAuthenticationCompatibility> {
    let resolved_env = resolve_acp_authentication_environment(auth, env);
    let provider_authentication = provider_authentication?;

    let mode = resolve_acp_provider_authentication_mode(auth);
    if mode == ACPProviderMode::Direct {
        return Some(ACPProviderAuthenticationCompatibility::Direct {
            mode: ACPDirectProviderMode::Direct,
        });
    }

    let credential_source = resolve_gateway_credential_source(&resolved_env);
    if mode == ACPProviderMode::Auto && credential_source.is_none() {
        return Some(ACPProviderAuthenticationCompatibility::Direct {
            mode: ACPDirectProviderMode::Auto,
        });
    }

    Some(ACPProviderAuthenticationCompatibility::AiGateway {
        mode: if mode == ACPProviderMode::Auto {
            ACPGatewayProviderMode::Auto
        } else {
            ACPGatewayProviderMode::AiGateway
        },
        env: provider_authentication.gateway.env.clone(),
        credential_source,
        base_url: resolved_env
            .get("AI_GATEWAY_BASE_URL")
            .cloned()
            .unwrap_or_else(|| DEFAULT_AI_GATEWAY_BASE_URL.to_string()),
    })
}

/// `resolveACPProviderAuthenticationMode`: the string mode, `auto` for an
/// environment record.
fn resolve_acp_provider_authentication_mode(
    auth: Option<&ACPAuthenticationMode>,
) -> ACPProviderMode {
    match auth {
        None | Some(HarnessV1Authentication::Auto | HarnessV1Authentication::Environment(_)) => {
            ACPProviderMode::Auto
        }
        Some(HarnessV1Authentication::AiGateway) => ACPProviderMode::AiGateway,
        Some(HarnessV1Authentication::Choice(HarnessV1DirectAuthentication::Direct)) => {
            ACPProviderMode::Direct
        }
    }
}

/// `resolveACPAuthenticationEnvironment`: an environment record `auth`
/// replaces `env`.
pub fn resolve_acp_authentication_environment(
    auth: Option<&ACPAuthenticationMode>,
    env: &IndexMap<String, String>,
) -> IndexMap<String, String> {
    match auth {
        Some(HarnessV1Authentication::Environment(environment)) => environment.clone(),
        _ => env.clone(),
    }
}

/// `resolveACPEnv`.
pub fn resolve_acp_env(
    auth: ACPAuthResolutionOptions<'_>,
    env: &IndexMap<String, String>,
) -> HarnessResult<IndexMap<String, String>> {
    Ok(resolve_acp_provider_authentication(auth, env, None)?.env)
}

fn resolve_gateway_credential_source(
    env: &IndexMap<String, String>,
) -> Option<ACPGatewayCredentialSource> {
    let set = |name: &str| env.get(name).is_some_and(|value| !value.is_empty());
    if set("AI_GATEWAY_API_KEY") {
        Some(ACPGatewayCredentialSource::AiGatewayApiKey)
    } else if set("VERCEL_OIDC_TOKEN") {
        Some(ACPGatewayCredentialSource::VercelOidcToken)
    } else {
        None
    }
}

fn resolve_gateway_credential(
    env: &IndexMap<String, String>,
    credential_source: Option<ACPGatewayCredentialSource>,
) -> Option<String> {
    match credential_source {
        Some(ACPGatewayCredentialSource::AiGatewayApiKey) => {
            if env.contains_key("AI_GATEWAY_API_KEY") {
                get_ai_gateway_auth_from_env(env).api_key
            } else {
                None
            }
        }
        Some(ACPGatewayCredentialSource::VercelOidcToken) => {
            let only_oidc: IndexMap<String, String> = env
                .get("VERCEL_OIDC_TOKEN")
                .map(|token| ("VERCEL_OIDC_TOKEN".to_string(), token.clone()))
                .into_iter()
                .collect();
            get_ai_gateway_auth_from_env(&only_oidc).api_key
        }
        None => None,
    }
}

/// `stableStringify`: `JSON.stringify` with object keys sorted.
pub(crate) fn stable_stringify(value: &JsonValue) -> String {
    serde_json::to_string(&sort_identity_value(value)).unwrap_or_default()
}

fn sort_identity_value(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Array(items) => items.iter().map(sort_identity_value).collect(),
        JsonValue::Object(record) => {
            let mut entries: Vec<_> = record.iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            JsonValue::Object(
                entries
                    .into_iter()
                    .map(|(key, entry)| (key.clone(), sort_identity_value(entry)))
                    .collect(),
            )
        }
        other => other.clone(),
    }
}

#[cfg(test)]
#[path = "acp_auth_tests.rs"]
mod tests;
