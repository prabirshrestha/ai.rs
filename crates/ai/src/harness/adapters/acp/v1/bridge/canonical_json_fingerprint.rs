//! Port of `harness-acp/src/v1/bridge/canonical-json-fingerprint.ts`.

use serde_json::{Map, Value};

/// `canonicalFingerprint`: JSON with object keys sorted recursively.
///
/// [rust] `undefined` has no JSON representation, so there is nothing to
/// omit.
pub fn canonical_fingerprint(value: &Value) -> String {
    serde_json::to_string(&canonicalize_json(value)).unwrap_or_default()
}

fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(canonicalize_json).collect()),
        Value::Object(record) => {
            let mut keys: Vec<&String> = record.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            let mut sorted = Map::new();
            for key in keys {
                sorted.insert(key.clone(), canonicalize_json(&record[key]));
            }
            Value::Object(sorted)
        }
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn sorts_object_keys_recursively() {
        let first = canonical_fingerprint(&json!({
            "nested": { "z": 1, "b": 2 },
            "city": "Lima",
        }));
        let reordered =
            canonical_fingerprint(&json!({ "city": "Lima", "nested": { "b": 2, "z": 1 } }));
        assert_eq!(first, r#"{"city":"Lima","nested":{"b":2,"z":1}}"#);
        assert_eq!(reordered, first);
    }

    #[test]
    fn preserves_array_order_and_serializes_null_array_entries() {
        assert_eq!(
            canonical_fingerprint(&json!({ "items": [{ "z": 1, "a": 2 }, null] })),
            r#"{"items":[{"a":2,"z":1},null]}"#
        );
        assert_ne!(
            canonical_fingerprint(&json!([2, 1])),
            canonical_fingerprint(&json!([1, 2]))
        );
    }
}
