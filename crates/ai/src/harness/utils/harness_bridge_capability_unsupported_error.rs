//! Port of `@ai-sdk/harness/bridge`'s
//! `harness-bridge-capability-unsupported-error.ts`, host side.
//!
//! [rust] The bridge raising the error runs as JavaScript in the sandbox;
//! the host only recognizes the serialized shape it receives in an `error`
//! frame and translates it to `HarnessCapabilityUnsupportedError`.

use serde_json::Value;

/// `error.name` of `HarnessBridgeCapabilityUnsupportedError`.
pub const HARNESS_BRIDGE_CAPABILITY_UNSUPPORTED_ERROR_NAME: &str =
    "AI_HarnessBridgeCapabilityUnsupportedError";

/// `HarnessBridgeCapabilityUnsupportedError.isInstance(error)` on a
/// serialized error: its message when `error` is one.
pub fn harness_bridge_capability_unsupported_error_message(error: &Value) -> Option<&str> {
    let record = error.as_object()?;
    if record.get("name").and_then(Value::as_str)
        != Some(HARNESS_BRIDGE_CAPABILITY_UNSUPPORTED_ERROR_NAME)
    {
        return None;
    }
    record.get("message").and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn recognizes_the_serialized_bridge_error_shape() {
        assert_eq!(
            harness_bridge_capability_unsupported_error_message(&json!({
                "name": "AI_HarnessBridgeCapabilityUnsupportedError",
                "message": "Unsupported.",
                "stack": "stack",
            })),
            Some("Unsupported.")
        );
    }

    #[test]
    fn rejects_unrelated_errors_and_malformed_serialized_values() {
        assert_eq!(
            harness_bridge_capability_unsupported_error_message(
                &json!({ "name": "Error", "message": "x" })
            ),
            None
        );
        assert_eq!(
            harness_bridge_capability_unsupported_error_message(
                &json!({ "name": "AI_HarnessBridgeCapabilityUnsupportedError" })
            ),
            None
        );
        assert_eq!(
            harness_bridge_capability_unsupported_error_message(&Value::Null),
            None
        );
    }
}
