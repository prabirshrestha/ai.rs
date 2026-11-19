//! Host-side part of `harness-acp/src/v1/bridge/protocol-configuration.ts`.
//!
//! [rust] Only `resolveACPLaunchEnvironment` runs on the host; the
//! initialize/negotiation helpers run in the vendored bridge.

use indexmap::IndexMap;
use serde_json::Value;

use crate::harness::errors::HarnessResult;
use crate::harness::utils::plain_error;

use super::bridge_environment::ACPResolvedProviderAuthentication;
use super::profile_values::{ACPGatewayValues, resolve_acp_profile_value};

/// `resolveACPLaunchEnvironment`.
pub fn resolve_acp_launch_environment(
    provider_authentication: Option<&ACPResolvedProviderAuthentication>,
    gateway: Option<&ACPGatewayValues>,
) -> HarnessResult<IndexMap<String, String>> {
    let Some(ACPResolvedProviderAuthentication::AiGateway { env }) = provider_authentication else {
        return Ok(IndexMap::new());
    };
    let gateway =
        gateway.ok_or_else(|| plain_error("ACP Gateway profile values are unavailable."))?;
    let value = Value::Object(env.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    let Value::Object(resolved) = resolve_acp_profile_value(&value, gateway) else {
        return Err(plain_error("ACP profile data must resolve to an object."));
    };
    Ok(resolved
        .into_iter()
        .map(|(key, value)| {
            let value = match value {
                Value::String(value) => value,
                other => serde_json::to_string(&other).unwrap_or_default(),
            };
            (key, value)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn gateway() -> ACPGatewayValues {
        ACPGatewayValues {
            api_key: "gateway-key".into(),
            base_url: "https://gateway.example".into(),
            client_app_name: "ai-sdk-harness-acp".into(),
            client_app_version: "0.0.0-test".into(),
        }
    }

    fn ai_gateway(env: Value) -> ACPResolvedProviderAuthentication {
        serde_json::from_value(json!({ "type": "ai-gateway", "env": env })).unwrap()
    }

    fn strings(entries: &[(&str, &str)]) -> IndexMap<String, String> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn resolves_the_gateway_launch_environment() {
        let provider_authentication = ai_gateway(json!({
            "CODEX_CONFIG": "{\"model\":\"openai/gpt-5.5\"}",
            "PROVIDER_BASE_URL": { "$source": "gateway-base-url", "ensureSuffix": "/v1" },
        }));
        assert_eq!(
            resolve_acp_launch_environment(Some(&provider_authentication), Some(&gateway()))
                .unwrap(),
            strings(&[
                ("CODEX_CONFIG", "{\"model\":\"openai/gpt-5.5\"}"),
                ("PROVIDER_BASE_URL", "https://gateway.example/v1"),
            ])
        );
        assert_eq!(
            resolve_acp_launch_environment(Some(&ACPResolvedProviderAuthentication::Direct), None)
                .unwrap(),
            IndexMap::<String, String>::new()
        );
    }

    #[test]
    fn serializes_structured_gateway_launch_environment_values_as_json() {
        let provider_authentication = ai_gateway(json!({
            "CODEX_CONFIG": {
                "model": "openai/gpt-5.5",
                "model_providers": {
                    "ai_gateway": {
                        "base_url": { "$source": "gateway-base-url", "ensureSuffix": "/v1" },
                    },
                },
            },
        }));
        assert_eq!(
            resolve_acp_launch_environment(Some(&provider_authentication), Some(&gateway()))
                .unwrap(),
            strings(&[(
                "CODEX_CONFIG",
                r#"{"model":"openai/gpt-5.5","model_providers":{"ai_gateway":{"base_url":"https://gateway.example/v1"}}}"#
            )])
        );
    }

    #[test]
    fn resolves_client_attribution_headers_in_structured_codex_configuration() {
        let provider_authentication = ai_gateway(json!({
            "CODEX_CONFIG": {
                "model_providers": {
                    "ai_gateway": {
                        "http_headers": {
                            "User-Agent": { "$source": "client-app" },
                            "x-client-app": { "$source": "client-app" },
                        },
                    },
                },
            },
        }));
        assert_eq!(
            resolve_acp_launch_environment(Some(&provider_authentication), Some(&gateway()))
                .unwrap(),
            strings(&[(
                "CODEX_CONFIG",
                r#"{"model_providers":{"ai_gateway":{"http_headers":{"User-Agent":"ai-sdk-harness-acp/0.0.0-test","x-client-app":"ai-sdk-harness-acp/0.0.0-test"}}}}"#
            )])
        );
    }
}
