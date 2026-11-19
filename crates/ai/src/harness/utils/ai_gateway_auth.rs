//! Port of `utils/ai-gateway-auth.ts`.

use indexmap::IndexMap;

/// Result of [`get_ai_gateway_auth_from_env`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiGatewayAuth {
    pub api_key: Option<String>,
    pub base_url: String,
}

/// `getAiGatewayAuthFromEnv({ env })`.
///
/// Empty values fall through, like the TS `||` chain.
pub fn get_ai_gateway_auth_from_env(env: &IndexMap<String, String>) -> AiGatewayAuth {
    let non_empty = |name: &str| env.get(name).filter(|value| !value.is_empty()).cloned();
    AiGatewayAuth {
        api_key: non_empty("AI_GATEWAY_API_KEY").or_else(|| non_empty("VERCEL_OIDC_TOKEN")),
        base_url: env
            .get("AI_GATEWAY_BASE_URL")
            .cloned()
            .unwrap_or_else(|| "https://ai-gateway.vercel.sh".to_string()),
    }
}

#[cfg(test)]
pub(crate) fn env(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_ai_gateway_api_key_over_vercel_oidc_token() {
        assert_eq!(
            get_ai_gateway_auth_from_env(&env(&[
                ("AI_GATEWAY_API_KEY", "gateway-key"),
                ("VERCEL_OIDC_TOKEN", "oidc-token"),
            ])),
            AiGatewayAuth {
                api_key: Some("gateway-key".into()),
                base_url: "https://ai-gateway.vercel.sh".into(),
            }
        );
    }

    #[test]
    fn reads_ai_gateway_base_url_from_env() {
        assert_eq!(
            get_ai_gateway_auth_from_env(&env(&[
                ("AI_GATEWAY_API_KEY", "gateway-key"),
                ("AI_GATEWAY_BASE_URL", "https://gateway.example.com"),
            ])),
            AiGatewayAuth {
                api_key: Some("gateway-key".into()),
                base_url: "https://gateway.example.com".into(),
            }
        );
    }

    #[test]
    fn falls_back_to_vercel_oidc_token() {
        assert_eq!(
            get_ai_gateway_auth_from_env(&env(&[("VERCEL_OIDC_TOKEN", "oidc-token")])),
            AiGatewayAuth {
                api_key: Some("oidc-token".into()),
                base_url: "https://ai-gateway.vercel.sh".into(),
            }
        );
    }

    #[test]
    fn returns_no_api_key_when_no_gateway_auth_env_is_configured() {
        assert_eq!(
            get_ai_gateway_auth_from_env(&env(&[])),
            AiGatewayAuth {
                api_key: None,
                base_url: "https://ai-gateway.vercel.sh".into(),
            }
        );
    }
}
