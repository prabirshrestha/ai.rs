//! Port of `harness-acp/src/v1/bridge/acp-v1-bridge-environment.ts`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::harness::errors::HarnessResult;
use crate::harness::utils::plain_error;

use super::super::settings::{
    ACPAuthentication, ACPHostToolMCPTransport, ACPNonTextContentType, ACPProfileValue,
    ACPSerializableValue,
};

/// `ACP_BRIDGE_CONFIGURATION_ENV`.
pub const ACP_BRIDGE_CONFIGURATION_ENV: &str = "AI_SDK_ACP_BRIDGE_CONFIGURATION";

/// `ACPResolvedProviderAuthentication`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ACPResolvedProviderAuthentication {
    Direct,
    AiGateway {
        env: IndexMap<String, ACPProfileValue>,
    },
}

/// `ACPAgentLaunch`: launch command of the ACP agent inside the sandbox.
/// `home` is set for agents acquired with an install command, which run with
/// a private `HOME` below the harness bootstrap directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ACPAgentLaunch {
    pub command: String,
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home: Option<String>,
}

/// `ACPBridgeConfiguration`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ACPBridgeConfiguration {
    pub agent: ACPAgentLaunch,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authentication: Option<ACPAuthentication>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_authentication: Option<ACPResolvedProviderAuthentication>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_environment: Option<IndexMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_meta: Option<IndexMap<String, ACPSerializableValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_capabilities: Option<IndexMap<String, ACPSerializableValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_user_questions_request_method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_tool_mcp_transport: Option<ACPHostToolMCPTransport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_text_content_types: Option<Vec<ACPNonTextContentType>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_protected_paths: Option<Vec<String>>,
}

impl ACPBridgeConfiguration {
    /// A configuration that only launches `agent`.
    pub fn new(agent: ACPAgentLaunch) -> Self {
        Self {
            agent,
            authentication: None,
            provider_authentication: None,
            provider_environment: None,
            session_meta: None,
            client_capabilities: None,
            ask_user_questions_request_method: None,
            host_tool_mcp_transport: None,
            non_text_content_types: None,
            write_protected_paths: None,
        }
    }
}

/// `createACPBridgeEnvironment`.
pub fn create_acp_bridge_environment(
    configuration: &ACPBridgeConfiguration,
) -> IndexMap<String, String> {
    IndexMap::from([(
        ACP_BRIDGE_CONFIGURATION_ENV.to_string(),
        serde_json::to_string(configuration).unwrap_or_default(),
    )])
}

/// `readACPBridgeEnvironment`. The error never includes the contents.
pub fn read_acp_bridge_environment(
    env: &IndexMap<String, String>,
) -> HarnessResult<ACPBridgeConfiguration> {
    env.get(ACP_BRIDGE_CONFIGURATION_ENV)
        .and_then(|value| serde_json::from_str(value).ok())
        .ok_or_else(|| plain_error("ACP bridge configuration environment is invalid."))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    const INVALID: &str = "ACP bridge configuration environment is invalid.";

    fn agent() -> ACPAgentLaunch {
        ACPAgentLaunch {
            command: "/home/agent/.harness-bootstrap/example-acp/home/.local/bin/agent".into(),
            args: vec!["acp".into()],
            home: Some("/home/agent/.harness-bootstrap/example-acp/home".into()),
        }
    }

    fn agent_json() -> Value {
        serde_json::to_value(agent()).unwrap()
    }

    fn raw(value: Value) -> IndexMap<String, String> {
        IndexMap::from([(ACP_BRIDGE_CONFIGURATION_ENV.to_string(), value.to_string())])
    }

    fn round_trip(configuration: ACPBridgeConfiguration) {
        assert_eq!(
            read_acp_bridge_environment(&create_acp_bridge_environment(&configuration)).unwrap(),
            configuration
        );
    }

    fn assert_invalid(env: IndexMap<String, String>) {
        assert_eq!(
            read_acp_bridge_environment(&env).unwrap_err().to_string(),
            INVALID
        );
    }

    #[test]
    fn round_trips_the_agent_launch() {
        round_trip(ACPBridgeConfiguration::new(agent()));
    }

    #[test]
    fn round_trips_non_text_content_types() {
        for types in [vec![], vec![ACPNonTextContentType::Image]] {
            round_trip(ACPBridgeConfiguration {
                non_text_content_types: Some(types),
                ..ACPBridgeConfiguration::new(agent())
            });
        }
    }

    #[test]
    fn rejects_invalid_non_text_content_types() {
        for types in [json!(["text"]), json!(["audio"]), json!(true)] {
            assert_invalid(raw(
                json!({ "agent": agent_json(), "nonTextContentTypes": types }),
            ));
        }
    }

    #[test]
    fn round_trips_write_protected_paths() {
        round_trip(ACPBridgeConfiguration {
            write_protected_paths: Some(vec![".github/hooks".into(), ".mcp.json".into()]),
            ..ACPBridgeConfiguration::new(agent())
        });
    }

    #[test]
    fn rejects_invalid_write_protected_paths() {
        for paths in [json!(".github/hooks"), json!([1]), json!([null])] {
            assert_invalid(raw(
                json!({ "agent": agent_json(), "writeProtectedPaths": paths }),
            ));
        }
    }

    #[test]
    fn round_trips_fixed_authentication_and_session_configuration() {
        let configuration: ACPBridgeConfiguration = serde_json::from_value(json!({
            "agent": agent_json(),
            "authentication": { "methodId": "api-key", "meta": { "apiKey": "direct-secret" } },
            "providerAuthentication": {
                "type": "ai-gateway",
                "env": {
                    "AI_GATEWAY_API_KEY": { "$source": "gateway-api-key" },
                    "AI_GATEWAY_BASE_URL": { "$source": "gateway-base-url" },
                    "CLIENT_APP": { "$source": "client-app" },
                    "CLIENT_APP_NAME": { "$source": "client-app-name" },
                    "CLIENT_APP_VERSION": { "$source": "client-app-version" },
                },
            },
            "providerEnvironment": {
                "AI_GATEWAY_API_KEY": "AI_GATEWAY_API_KEY",
                "AI_GATEWAY_BASE_URL": "https://gateway.example/v1",
            },
            "sessionMeta": { "opaqueCredential": "session-secret" },
            "clientCapabilities": { "_meta": { "parameterizedModelPicker": true } },
        }))
        .unwrap();
        round_trip(configuration);
    }

    #[test]
    fn rejects_a_missing_configuration_environment_variable() {
        assert_invalid(IndexMap::new());
    }

    #[test]
    fn round_trips_a_gateway_launch_environment() {
        round_trip(ACPBridgeConfiguration {
            provider_authentication: Some(ACPResolvedProviderAuthentication::AiGateway {
                env: IndexMap::from([(
                    "CODEX_CONFIG".into(),
                    json!("{\"model\":\"openai/gpt-5.5\"}"),
                )]),
            }),
            ..ACPBridgeConfiguration::new(agent())
        });
    }

    #[test]
    fn does_not_expose_invalid_environment_contents_in_its_error() {
        let secret = "invalid-secret";
        let error = read_acp_bridge_environment(&IndexMap::from([(
            ACP_BRIDGE_CONFIGURATION_ENV.to_string(),
            format!("{{{secret}"),
        )]))
        .unwrap_err()
        .to_string();
        assert_eq!(error, INVALID);
        assert!(!error.contains(secret));
    }

    #[test]
    fn rejects_configuration_that_does_not_match_the_schema() {
        assert_invalid(raw(json!({
            "agent": agent_json(),
            "providerAuthentication": { "type": "ai-gateway", "env": [] },
        })));
    }
}
