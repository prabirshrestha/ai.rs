//! Port of `utils/bridge-auth-protocol.ts`.

use std::sync::LazyLock;

use regex::Regex;

pub const BRIDGE_AUTH_PUBLIC_KEY_ENV: &str = "BRIDGE_AUTH_PUBLIC_KEY";
pub const BRIDGE_AUTH_TIMEOUT_MS: u64 = 10_000;
pub const BRIDGE_AUTH_MAX_FRAME_BYTES: usize = 1_024;

const BRIDGE_AUTH_MESSAGE_PREFIX: &str = "ai-sdk-harness-bridge-auth-v1\0";

static CHALLENGE_FRAME_REGEXP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\{"type":"bridge-auth-challenge","challenge":"([a-f0-9]{64})"\}$"#).unwrap()
});
static PROOF_FRAME_REGEXP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^\{"type":"bridge-auth-proof","signature":"([A-Za-z0-9_-]{86})"\}$"#).unwrap()
});

/// `JSON.stringify(value)` for a string, keeping the TS key order around it.
fn json_string(value: &str) -> String {
    serde_json::Value::from(value).to_string()
}

/// `createBridgeAuthMessage({ challenge })`: the domain-separated bytes the
/// host signs.
pub fn create_bridge_auth_message(challenge: &str) -> Vec<u8> {
    format!("{BRIDGE_AUTH_MESSAGE_PREFIX}{challenge}").into_bytes()
}

/// `serializeBridgeAuthChallengeFrame({ challenge })`.
pub fn serialize_bridge_auth_challenge_frame(challenge: &str) -> String {
    format!(
        r#"{{"type":"bridge-auth-challenge","challenge":{}}}"#,
        json_string(challenge)
    )
}

/// `parseBridgeAuthChallengeFrame({ text })`.
///
/// Authentication frames are matched against their exact canonical
/// serialization instead of being parsed as JSON, which also rejects any
/// frame with unexpected fields or encodings.
pub fn parse_bridge_auth_challenge_frame(text: &str) -> Option<String> {
    CHALLENGE_FRAME_REGEXP
        .captures(text)
        .map(|captures| captures[1].to_string())
}

/// `serializeBridgeAuthProofFrame({ signature })`.
pub fn serialize_bridge_auth_proof_frame(signature: &str) -> String {
    format!(
        r#"{{"type":"bridge-auth-proof","signature":{}}}"#,
        json_string(signature)
    )
}

/// `parseBridgeAuthProofFrame({ text })`.
pub fn parse_bridge_auth_proof_frame(text: &str) -> Option<String> {
    PROOF_FRAME_REGEXP
        .captures(text)
        .map(|captures| captures[1].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn challenge() -> String {
        "a".repeat(64)
    }

    fn signature() -> String {
        "A".repeat(86)
    }

    #[test]
    fn creates_a_domain_separated_message_for_the_challenge() {
        assert_eq!(
            String::from_utf8(create_bridge_auth_message(&challenge())).unwrap(),
            format!("ai-sdk-harness-bridge-auth-v1\0{}", challenge())
        );
    }

    #[test]
    fn round_trips_canonical_challenge_and_proof_frames() {
        let challenge_frame = serialize_bridge_auth_challenge_frame(&challenge());
        let proof_frame = serialize_bridge_auth_proof_frame(&signature());
        assert_eq!(
            challenge_frame,
            format!(
                r#"{{"type":"bridge-auth-challenge","challenge":"{}"}}"#,
                challenge()
            )
        );
        assert_eq!(
            proof_frame,
            format!(
                r#"{{"type":"bridge-auth-proof","signature":"{}"}}"#,
                signature()
            )
        );
        assert_eq!(
            parse_bridge_auth_challenge_frame(&challenge_frame),
            Some(challenge())
        );
        assert_eq!(
            parse_bridge_auth_proof_frame(&proof_frame),
            Some(signature())
        );
    }

    #[test]
    fn rejects_non_canonical_challenge_frames() {
        let c = challenge();
        for text in [
            format!(
                r#"{{"type":"bridge-auth-challenge","challenge":"{}"}}"#,
                "A".repeat(64)
            ),
            format!(
                r#"{{"type":"bridge-auth-challenge","challenge":"{}"}}"#,
                "a".repeat(63)
            ),
            format!(r#"{{"challenge":"{c}","type":"bridge-auth-challenge"}}"#),
            format!(r#"{{"type":"bridge-auth-challenge","challenge":"{c}","extra":1}}"#),
            format!(r#" {{"type":"bridge-auth-challenge","challenge":"{c}"}}"#),
            r#"{"type":"bridge-hello"}"#.to_string(),
        ] {
            assert_eq!(parse_bridge_auth_challenge_frame(&text), None, "{text}");
        }
    }

    #[test]
    fn rejects_non_canonical_proof_frames() {
        let s = signature();
        for text in [
            format!(
                r#"{{"type":"bridge-auth-proof","signature":"{}"}}"#,
                "A".repeat(85)
            ),
            format!(
                r#"{{"type":"bridge-auth-proof","signature":"{}="}}"#,
                "A".repeat(85)
            ),
            format!(r#"{{"signature":"{s}","type":"bridge-auth-proof"}}"#),
            format!(r#"{{"type":"bridge-auth-proof","signature":"{s}","extra":1}}"#),
            r#"{"type":"start"}"#.to_string(),
        ] {
            assert_eq!(parse_bridge_auth_proof_frame(&text), None, "{text}");
        }
    }
}
