//! Port of `utils/bridge-auth.ts`: the host side of bridge authentication.

use std::fmt;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::rand::{SecureRandom as _, SystemRandom};
use ring::signature::{self, Ed25519KeyPair, KeyPair as _};

use crate::harness::errors::HarnessResult;
use crate::harness::utils::bridge_auth_protocol::{
    create_bridge_auth_message, parse_bridge_auth_challenge_frame,
    serialize_bridge_auth_proof_frame,
};
use crate::harness::utils::plain_error;

/// PKCS#8 v1 prefix of an Ed25519 private key, as Node exports it.
const ED25519_PKCS8_V1_PREFIX: [u8; 16] = [
    0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20,
];

/// `BridgeAuthKeyPair`.
#[derive(Clone, PartialEq, Eq)]
pub struct BridgeAuthKeyPair {
    /// Host-only PKCS#8 DER private key, base64url encoded. Never pass it
    /// into the sandbox.
    pub private_key: String,
    /// Raw Ed25519 public key, base64url encoded. Safe to pass into the
    /// sandbox.
    pub public_key: String,
}

impl fmt::Debug for BridgeAuthKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BridgeAuthKeyPair")
            .field("public_key", &self.public_key)
            .finish_non_exhaustive()
    }
}

/// `createBridgeAuthKeyPair()`: a fresh Ed25519 key pair.
pub fn create_bridge_auth_key_pair() -> HarnessResult<BridgeAuthKeyPair> {
    let mut seed = [0u8; 32];
    SystemRandom::new()
        .fill(&mut seed)
        .map_err(|_| plain_error("Failed to export the bridge authentication public key."))?;
    let key_pair = Ed25519KeyPair::from_seed_unchecked(&seed)
        .map_err(|_| plain_error("Failed to export the bridge authentication public key."))?;
    let mut pkcs8 = ED25519_PKCS8_V1_PREFIX.to_vec();
    pkcs8.extend_from_slice(&seed);
    Ok(BridgeAuthKeyPair {
        private_key: URL_SAFE_NO_PAD.encode(pkcs8),
        public_key: URL_SAFE_NO_PAD.encode(key_pair.public_key().as_ref()),
    })
}

/// The parsed private key (`KeyObject`).
pub struct BridgeAuthPrivateKey(Ed25519KeyPair);

impl fmt::Debug for BridgeAuthPrivateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BridgeAuthPrivateKey")
    }
}

/// `parseBridgeAuthPrivateKey({ privateKey })`.
pub fn parse_bridge_auth_private_key(private_key: &str) -> HarnessResult<BridgeAuthPrivateKey> {
    let der = base64::engine::GeneralPurpose::new(
        &base64::alphabet::URL_SAFE,
        base64::engine::GeneralPurposeConfig::new()
            .with_decode_padding_mode(base64::engine::DecodePaddingMode::Indifferent),
    )
    .decode(private_key)
    .map_err(|_| plain_error("Invalid bridge authentication private key."))?;
    if let Ok(key_pair) = Ed25519KeyPair::from_pkcs8_maybe_unchecked(&der) {
        return Ok(BridgeAuthPrivateKey(key_pair));
    }
    if is_other_pkcs8_private_key(&der) {
        return Err(plain_error(
            "Bridge authentication requires an Ed25519 private key.",
        ));
    }
    Err(plain_error("Invalid bridge authentication private key."))
}

/// Whether `der` is a valid PKCS#8 key of another algorithm.
fn is_other_pkcs8_private_key(der: &[u8]) -> bool {
    let rng = SystemRandom::new();
    [
        &signature::ECDSA_P256_SHA256_ASN1_SIGNING,
        &signature::ECDSA_P384_SHA384_ASN1_SIGNING,
    ]
    .into_iter()
    .any(|algorithm| signature::EcdsaKeyPair::from_pkcs8(algorithm, der, &rng).is_ok())
        || signature::RsaKeyPair::from_pkcs8(der).is_ok()
}

/// `createBridgeAuthProof({ privateKey, message })`: the proof frame that
/// answers a challenge frame.
pub fn create_bridge_auth_proof(
    private_key: &BridgeAuthPrivateKey,
    message: &str,
) -> HarnessResult<String> {
    let challenge = parse_bridge_auth_challenge_frame(message)
        .ok_or_else(|| plain_error("Invalid bridge authentication challenge."))?;
    let signature = private_key.0.sign(&create_bridge_auth_message(&challenge));
    Ok(serialize_bridge_auth_proof_frame(
        &URL_SAFE_NO_PAD.encode(signature.as_ref()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::utils::bridge_auth_protocol::{
        parse_bridge_auth_proof_frame, serialize_bridge_auth_challenge_frame,
    };
    use regex::Regex;

    fn challenge() -> String {
        "0123456789abcdef".repeat(4)
    }

    #[test]
    fn creates_distinct_single_line_base64url_key_pairs() {
        let first = create_bridge_auth_key_pair().unwrap();
        let second = create_bridge_auth_key_pair().unwrap();
        assert!(
            Regex::new(r"^[A-Za-z0-9_-]{43}$")
                .unwrap()
                .is_match(&first.public_key)
        );
        assert!(
            Regex::new(r"^[A-Za-z0-9_-]+$")
                .unwrap()
                .is_match(&first.private_key)
        );
        assert_ne!(first.private_key, second.private_key);
        assert_ne!(first.public_key, second.public_key);
    }

    #[test]
    fn rejects_malformed_keys() {
        assert_eq!(
            parse_bridge_auth_private_key("nope")
                .unwrap_err()
                .to_string(),
            "Invalid bridge authentication private key."
        );
    }

    #[test]
    fn rejects_non_ed25519_keys() {
        let rng = SystemRandom::new();
        let pkcs8 = signature::EcdsaKeyPair::generate_pkcs8(
            &signature::ECDSA_P256_SHA256_ASN1_SIGNING,
            &rng,
        )
        .unwrap();
        assert_eq!(
            parse_bridge_auth_private_key(&URL_SAFE_NO_PAD.encode(pkcs8.as_ref()))
                .unwrap_err()
                .to_string(),
            "Bridge authentication requires an Ed25519 private key."
        );
    }

    #[test]
    fn signs_the_challenge_with_the_private_key() {
        let key_pair = create_bridge_auth_key_pair().unwrap();
        let proof = create_bridge_auth_proof(
            &parse_bridge_auth_private_key(&key_pair.private_key).unwrap(),
            &serialize_bridge_auth_challenge_frame(&challenge()),
        )
        .unwrap();
        let signature = parse_bridge_auth_proof_frame(&proof).unwrap();
        let public_key = URL_SAFE_NO_PAD.decode(&key_pair.public_key).unwrap();
        signature::UnparsedPublicKey::new(&signature::ED25519, public_key)
            .verify(
                &create_bridge_auth_message(&challenge()),
                &URL_SAFE_NO_PAD.decode(signature).unwrap(),
            )
            .unwrap();
    }

    #[test]
    fn rejects_frames_that_are_not_authentication_challenges() {
        let key_pair = create_bridge_auth_key_pair().unwrap();
        let error = create_bridge_auth_proof(
            &parse_bridge_auth_private_key(&key_pair.private_key).unwrap(),
            r#"{"type":"bridge-hello"}"#,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Invalid bridge authentication challenge."
        );
    }
}
