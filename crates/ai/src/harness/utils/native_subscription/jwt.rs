//! Port of `utils/native-subscription/jwt.ts`.

use base64::Engine as _;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use serde_json::{Map, Value};

// `Buffer.from(segment, 'base64url')` accepts padded and unpadded input.
const BASE64URL: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::URL_SAFE,
    GeneralPurposeConfig::new()
        .with_decode_padding_mode(DecodePaddingMode::Indifferent)
        .with_decode_allow_trailing_bits(true),
);

/// `parseJwtPayload({ token })`: the decoded payload when `token` has three
/// segments and its payload is a JSON object.
pub fn parse_jwt_payload(token: &str) -> Option<Map<String, Value>> {
    let segments: Vec<&str> = token.split('.').collect();
    if segments.len() != 3 {
        return None;
    }
    let bytes = BASE64URL.decode(segments[1]).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    match serde_json::from_str::<Value>(&text).ok()? {
        Value::Object(payload) => Some(payload),
        _ => None,
    }
}

/// `getJwtExpiresAt({ token })`: the `exp` claim in milliseconds.
pub fn get_jwt_expires_at(token: &str) -> Option<f64> {
    let payload = parse_jwt_payload(token)?;
    let expires_at = payload.get("exp")?.as_f64()?;
    expires_at.is_finite().then_some(expires_at * 1000.0)
}

#[cfg(test)]
pub(crate) fn create_jwt(payload: &Value) -> String {
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload.to_string());
    format!("header.{payload}.signature")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_an_object_payload() {
        let payload = parse_jwt_payload(&create_jwt(&json!({ "subject": "user" })));
        assert_eq!(
            payload.map(Value::Object),
            Some(json!({ "subject": "user" }))
        );
    }

    #[test]
    fn returns_none_for_invalid_tokens_and_non_object_payloads() {
        assert_eq!(parse_jwt_payload("not-a-jwt"), None);
        assert_eq!(parse_jwt_payload("header.payload"), None);
        assert_eq!(parse_jwt_payload("header.not-json.signature"), None);
        assert_eq!(
            parse_jwt_payload(&create_jwt(&json!("not-an-object"))),
            None
        );
    }

    #[test]
    fn converts_the_exp_claim_from_seconds_to_milliseconds() {
        assert_eq!(
            get_jwt_expires_at(&create_jwt(&json!({ "exp": 1234 }))),
            Some(1_234_000.0)
        );
    }

    #[test]
    fn returns_none_without_a_numeric_exp_claim() {
        assert_eq!(get_jwt_expires_at(&create_jwt(&json!({}))), None);
        assert_eq!(
            get_jwt_expires_at(&create_jwt(&json!({ "exp": "1234" }))),
            None
        );
        assert_eq!(
            get_jwt_expires_at(&create_jwt(&json!({ "exp": null }))),
            None
        );
    }
}
