//! Port of `harness-acp/src/v1/bridge/profile-values.ts`.

use serde_json::{Map, Value};

use super::super::settings::{ACPProfileValue, ACPSerializableValue};

/// `ACPGatewayValues`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ACPGatewayValues {
    pub api_key: String,
    pub base_url: String,
    pub client_app_name: String,
    pub client_app_version: String,
}

/// `resolveACPProfileValue`: replaces every `{ "$source": … }` object with
/// the gateway value it names.
pub fn resolve_acp_profile_value(
    value: &ACPProfileValue,
    gateway: &ACPGatewayValues,
) -> ACPSerializableValue {
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
fn resolve_source(source: &str, gateway: &ACPGatewayValues) -> String {
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
mod tests {
    use serde_json::json;

    use super::*;

    fn gateway() -> ACPGatewayValues {
        ACPGatewayValues {
            api_key: "gateway-key".into(),
            base_url: "https://gateway.example".into(),
            client_app_name: "ai-sdk-harness-acp".into(),
            client_app_version: "1.2.3".into(),
        }
    }

    #[test]
    fn resolves_nested_opaque_gateway_profile_data() {
        assert_eq!(
            resolve_acp_profile_value(
                &json!({
                    "gateway": {
                        "baseUrl": { "$source": "gateway-base-url", "ensureSuffix": "/v1" },
                        "headers": {
                            "Authorization": { "$source": "gateway-authorization" },
                            "x-client-app": { "$source": "client-app" },
                            "x-client-app-name": { "$source": "client-app-name" },
                            "x-client-app-version": { "$source": "client-app-version" },
                        },
                    },
                }),
                &gateway(),
            ),
            json!({
                "gateway": {
                    "baseUrl": "https://gateway.example/v1",
                    "headers": {
                        "Authorization": "Bearer gateway-key",
                        "x-client-app": "ai-sdk-harness-acp/1.2.3",
                        "x-client-app-name": "ai-sdk-harness-acp",
                        "x-client-app-version": "1.2.3",
                    },
                },
            })
        );
    }

    #[test]
    fn does_not_duplicate_a_required_suffix_on_a_custom_gateway_base_url() {
        assert_eq!(
            resolve_acp_profile_value(
                &json!({ "$source": "gateway-base-url", "ensureSuffix": "/v1" }),
                &ACPGatewayValues {
                    base_url: "https://gateway.example/v1/".into(),
                    ..gateway()
                },
            ),
            json!("https://gateway.example/v1")
        );
    }
}
