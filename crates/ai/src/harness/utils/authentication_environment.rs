//! Port of `utils/authentication-environment.ts`.

use serde_json::Value;

/// The `TypeError` that [`is_harness_authentication_environment`] throws.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Invalid auth: expected an authentication mode or a flat record with string values.")]
pub struct InvalidAuthenticationEnvironmentError;

/// `isHarnessAuthenticationEnvironment(value)`: whether `value` is a flat
/// record of strings. `None` stands for `undefined`.
///
/// [rust] A typed `HarnessAuthentication` already rejects bad records when
/// it deserializes, so adapters match on `Environment(_)` instead; this
/// function serves untyped JSON input.
pub fn is_harness_authentication_environment(
    value: Option<&Value>,
) -> Result<bool, InvalidAuthenticationEnvironmentError> {
    match value {
        Some(Value::Object(record)) => {
            if record.values().all(Value::is_string) {
                Ok(true)
            } else {
                Err(InvalidAuthenticationEnvironmentError)
            }
        }
        Some(Value::Array(_)) => Err(InvalidAuthenticationEnvironmentError),
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recognizes_flat_string_records() {
        assert_eq!(
            is_harness_authentication_environment(Some(&json!({}))),
            Ok(true)
        );
        assert_eq!(
            is_harness_authentication_environment(Some(&json!({
                "AI_GATEWAY_API_KEY": "gateway-key",
                "AI_GATEWAY_BASE_URL": "https://gateway.example",
            }))),
            Ok(true)
        );
    }

    #[test]
    fn returns_false_for_authentication_modes_and_missing_values() {
        assert_eq!(
            is_harness_authentication_environment(Some(&json!("auto"))),
            Ok(false)
        );
        assert_eq!(is_harness_authentication_environment(None), Ok(false));
    }

    #[test]
    fn rejects_nested_authentication_objects() {
        let error = is_harness_authentication_environment(Some(&json!({
            "gateway": { "apiKey": "gateway-key" },
        })))
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Invalid auth: expected an authentication mode or a flat record with string values."
        );
    }
}
