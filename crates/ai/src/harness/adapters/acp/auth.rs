//! Port of `acp-auth.ts` and `bridge/profile-values.ts`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use serde_json::{Map, Value as JsonValue};

use crate::harness::auth::{HarnessAuthentication, HarnessDirectAuthentication};
use crate::harness::error::HarnessResult;
use crate::harness::utils::{get_ai_gateway_auth_from_env, plain_error};

use super::settings::{AcpAuthentication, AcpProviderAuthentication};

fn sha256_hex(text: &str) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, text.as_bytes());
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// `AcpResolvedProviderAuthentication`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AcpResolvedProviderAuthentication {
    Direct,
    AiGateway { env: IndexMap<String, JsonValue> },
}

const DEFAULT_AI_GATEWAY_BASE_URL: &str = "https://ai-gateway.vercel.sh";

/// `AcpAuthenticationMode`: `HarnessAuthentication`.
pub type AcpAuthenticationMode = HarnessAuthentication;

/// `AcpClientApp`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcpClientApp {
    pub name: String,
    pub version: String,
}

/// `AcpAuthResolutionOptions`.
#[derive(Debug, Clone, Copy)]
pub struct AcpAuthResolutionOptions<'a> {
    pub mode: Option<&'a AcpAuthenticationMode>,
    pub provider_authentication: Option<&'a AcpProviderAuthentication>,
    pub client_app: &'a AcpClientApp,
}

/// `AcpGatewayCredentialSource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcpGatewayCredentialSource {
    #[serde(rename = "AI_GATEWAY_API_KEY")]
    AiGatewayApiKey,
    #[serde(rename = "VERCEL_OIDC_TOKEN")]
    VercelOidcToken,
}

/// `mode` of a direct [`AcpProviderAuthenticationCompatibility`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcpDirectProviderMode {
    Auto,
    Direct,
}

/// `mode` of an AI Gateway [`AcpProviderAuthenticationCompatibility`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcpGatewayProviderMode {
    Auto,
    AiGateway,
}

/// `AcpProviderAuthenticationCompatibility`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub enum AcpProviderAuthenticationCompatibility {
    Direct {
        mode: AcpDirectProviderMode,
    },
    AiGateway {
        mode: AcpGatewayProviderMode,
        env: IndexMap<String, JsonValue>,
        credential_source: Option<AcpGatewayCredentialSource>,
        base_url: String,
    },
}

/// `providerKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcpProviderKind {
    Direct,
    AiGateway,
}

/// `providerMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcpProviderMode {
    Auto,
    Direct,
    AiGateway,
}

/// `AcpAuthenticationProfileIdentity`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpAuthenticationProfileIdentity {
    pub digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acp_method_id: Option<String>,
    pub provider_kind: AcpProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_mode: Option<AcpProviderMode>,
    /// `gatewayCredentialSource?: … | null`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub gateway_credential_source: Option<Option<AcpGatewayCredentialSource>>,
}

/// Keeps `null` as `Some(None)`.
pub(crate) fn deserialize_present<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// `createAcpAuthenticationProfileIdentity`.
pub fn create_acp_authentication_profile_identity(
    authentication: Option<&AcpAuthentication>,
    provider_authentication_compatibility: Option<&AcpProviderAuthenticationCompatibility>,
) -> AcpAuthenticationProfileIdentity {
    let provider_kind = match provider_authentication_compatibility {
        Some(AcpProviderAuthenticationCompatibility::AiGateway { .. }) => {
            AcpProviderKind::AiGateway
        }
        _ => AcpProviderKind::Direct,
    };
    let digest = sha256_hex(&stable_stringify(&serde_json::json!({
        "authentication": authentication,
        "providerAuthentication": provider_authentication_compatibility,
    })));
    AcpAuthenticationProfileIdentity {
        digest,
        acp_method_id: authentication.map(|a| a.method_id.clone()),
        provider_kind,
        provider_mode: provider_authentication_compatibility.map(|c| match c {
            AcpProviderAuthenticationCompatibility::Direct { mode } => match mode {
                AcpDirectProviderMode::Auto => AcpProviderMode::Auto,
                AcpDirectProviderMode::Direct => AcpProviderMode::Direct,
            },
            AcpProviderAuthenticationCompatibility::AiGateway { mode, .. } => match mode {
                AcpGatewayProviderMode::Auto => AcpProviderMode::Auto,
                AcpGatewayProviderMode::AiGateway => AcpProviderMode::AiGateway,
            },
        }),
        gateway_credential_source: match provider_authentication_compatibility {
            Some(AcpProviderAuthenticationCompatibility::AiGateway {
                credential_source, ..
            }) => Some(*credential_source),
            _ => None,
        },
    }
}

/// Result of [`resolve_acp_provider_authentication`].
#[derive(Debug, Clone, PartialEq)]
pub struct AcpResolvedProviderAuthenticationResult {
    pub provider_authentication: Option<AcpResolvedProviderAuthentication>,
    pub env: IndexMap<String, String>,
}

/// `resolveAcpProviderAuthentication`.
pub fn resolve_acp_provider_authentication(
    auth: AcpAuthResolutionOptions<'_>,
    env: &IndexMap<String, String>,
    compatibility: Option<&AcpProviderAuthenticationCompatibility>,
) -> HarnessResult<AcpResolvedProviderAuthenticationResult> {
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
        return Ok(AcpResolvedProviderAuthenticationResult {
            provider_authentication: None,
            env: client_app_env,
        });
    };
    let direct = || AcpResolvedProviderAuthenticationResult {
        provider_authentication: Some(AcpResolvedProviderAuthentication::Direct),
        env: client_app_env.clone(),
    };

    if matches!(
        compatibility,
        Some(AcpProviderAuthenticationCompatibility::Direct { .. })
    ) {
        return Ok(direct());
    }

    let mode = resolve_acp_provider_authentication_mode(auth.mode);
    if mode == AcpProviderMode::Direct {
        return Ok(direct());
    }

    let (api_key, base_url) = match compatibility {
        Some(AcpProviderAuthenticationCompatibility::AiGateway {
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
    if compatibility.is_none() && mode == AcpProviderMode::Auto && api_key.is_none() {
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
    Ok(AcpResolvedProviderAuthenticationResult {
        provider_authentication: Some(AcpResolvedProviderAuthentication::AiGateway {
            env: match compatibility {
                Some(AcpProviderAuthenticationCompatibility::AiGateway { env, .. }) => env.clone(),
                _ => provider_authentication.gateway.env.clone(),
            },
        }),
        env: resolved,
    })
}

/// `resolveAcpProviderAuthenticationCompatibility`.
pub fn resolve_acp_provider_authentication_compatibility(
    auth: Option<&AcpAuthenticationMode>,
    provider_authentication: Option<&AcpProviderAuthentication>,
    env: &IndexMap<String, String>,
) -> Option<AcpProviderAuthenticationCompatibility> {
    let resolved_env = resolve_acp_authentication_environment(auth, env);
    let provider_authentication = provider_authentication?;

    let mode = resolve_acp_provider_authentication_mode(auth);
    if mode == AcpProviderMode::Direct {
        return Some(AcpProviderAuthenticationCompatibility::Direct {
            mode: AcpDirectProviderMode::Direct,
        });
    }

    let credential_source = resolve_gateway_credential_source(&resolved_env);
    if mode == AcpProviderMode::Auto && credential_source.is_none() {
        return Some(AcpProviderAuthenticationCompatibility::Direct {
            mode: AcpDirectProviderMode::Auto,
        });
    }

    Some(AcpProviderAuthenticationCompatibility::AiGateway {
        mode: if mode == AcpProviderMode::Auto {
            AcpGatewayProviderMode::Auto
        } else {
            AcpGatewayProviderMode::AiGateway
        },
        env: provider_authentication.gateway.env.clone(),
        credential_source,
        base_url: resolved_env
            .get("AI_GATEWAY_BASE_URL")
            .cloned()
            .unwrap_or_else(|| DEFAULT_AI_GATEWAY_BASE_URL.to_string()),
    })
}

/// `resolveAcpProviderAuthenticationMode`: the string mode, `auto` for an
/// environment record.
fn resolve_acp_provider_authentication_mode(
    auth: Option<&AcpAuthenticationMode>,
) -> AcpProviderMode {
    match auth {
        None | Some(HarnessAuthentication::Auto | HarnessAuthentication::Environment(_)) => {
            AcpProviderMode::Auto
        }
        Some(HarnessAuthentication::AiGateway) => AcpProviderMode::AiGateway,
        Some(HarnessAuthentication::Choice(HarnessDirectAuthentication::Direct)) => {
            AcpProviderMode::Direct
        }
    }
}

/// `resolveAcpAuthenticationEnvironment`: an environment record `auth`
/// replaces `env`.
pub fn resolve_acp_authentication_environment(
    auth: Option<&AcpAuthenticationMode>,
    env: &IndexMap<String, String>,
) -> IndexMap<String, String> {
    match auth {
        Some(HarnessAuthentication::Environment(environment)) => environment.clone(),
        _ => env.clone(),
    }
}

/// `resolveACPEnv`.
pub fn resolve_acp_env(
    auth: AcpAuthResolutionOptions<'_>,
    env: &IndexMap<String, String>,
) -> HarnessResult<IndexMap<String, String>> {
    Ok(resolve_acp_provider_authentication(auth, env, None)?.env)
}

fn resolve_gateway_credential_source(
    env: &IndexMap<String, String>,
) -> Option<AcpGatewayCredentialSource> {
    let set = |name: &str| env.get(name).is_some_and(|value| !value.is_empty());
    if set("AI_GATEWAY_API_KEY") {
        Some(AcpGatewayCredentialSource::AiGatewayApiKey)
    } else if set("VERCEL_OIDC_TOKEN") {
        Some(AcpGatewayCredentialSource::VercelOidcToken)
    } else {
        None
    }
}

fn resolve_gateway_credential(
    env: &IndexMap<String, String>,
    credential_source: Option<AcpGatewayCredentialSource>,
) -> Option<String> {
    match credential_source {
        Some(AcpGatewayCredentialSource::AiGatewayApiKey) => {
            if env.contains_key("AI_GATEWAY_API_KEY") {
                get_ai_gateway_auth_from_env(env).api_key
            } else {
                None
            }
        }
        Some(AcpGatewayCredentialSource::VercelOidcToken) => {
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
            entries.sort_by_key(|(left, _)| *left);
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

// --- profile-values.ts

type Value = JsonValue;

/// `AcpGatewayValues`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcpGatewayValues {
    pub api_key: String,
    pub base_url: String,
    pub client_app_name: String,
    pub client_app_version: String,
}

/// `resolveACPProfileValue`: replaces every `{ "$source": … }` object with
/// the gateway value it names.
pub fn resolve_acp_profile_value(value: &Value, gateway: &AcpGatewayValues) -> Value {
    match value {
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| resolve_acp_profile_value(item, gateway))
                .collect(),
        ),
        Value::Object(record) => {
            if let Some(Value::String(source)) = record.get("$source") {
                let source_value = resolve_source(source, gateway);
                let prefix = record.get("prefix").and_then(Value::as_str).unwrap_or("");
                let resolved = format!("{prefix}{source_value}");
                return Value::String(match record.get("ensureSuffix").and_then(Value::as_str) {
                    None => format!(
                        "{resolved}{}",
                        record.get("suffix").and_then(Value::as_str).unwrap_or("")
                    ),
                    Some(suffix) => ensure_suffix(&resolved, suffix),
                });
            }
            Value::Object(
                record
                    .iter()
                    .map(|(key, item)| (key.clone(), resolve_acp_profile_value(item, gateway)))
                    .collect::<Map<_, _>>(),
            )
        }
        _ => value.clone(),
    }
}

fn ensure_suffix(value: &str, suffix: &str) -> String {
    let normalized = value.trim_end_matches('/');
    if normalized.ends_with(suffix) {
        normalized.to_string()
    } else {
        format!("{normalized}{suffix}")
    }
}

/// [rust] TS narrows `$source` to the enum; an unknown name resolves to
/// `undefined`, which interpolates as `"undefined"`.
fn resolve_source(source: &str, gateway: &AcpGatewayValues) -> String {
    match source {
        "gateway-api-key" => gateway.api_key.clone(),
        "gateway-base-url" => gateway.base_url.clone(),
        "gateway-authorization" => format!("Bearer {}", gateway.api_key),
        "client-app" => format!("{}/{}", gateway.client_app_name, gateway.client_app_version),
        "client-app-name" => gateway.client_app_name.clone(),
        "client-app-version" => gateway.client_app_version.clone(),
        _ => "undefined".to_string(),
    }
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
