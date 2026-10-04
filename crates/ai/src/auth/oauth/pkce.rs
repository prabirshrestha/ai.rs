//! Port of `auth/oauth/pkce.ts`.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::digest::{SHA256, digest};
use ring::rand::{SecureRandom, SystemRandom};

use crate::{Error, Result};

/// A PKCE code verifier and its S256 challenge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

/// Encode bytes as a base64url string without padding.
fn base64url_encode(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Generate a PKCE code verifier (32 random bytes) and its SHA-256 challenge.
pub fn generate_pkce() -> Result<Pkce> {
    let mut verifier_bytes = [0u8; 32];
    SystemRandom::new()
        .fill(&mut verifier_bytes)
        .map_err(|_| Error::message("Failed to generate PKCE verifier"))?;
    let verifier = base64url_encode(&verifier_bytes);
    let challenge = base64url_encode(digest(&SHA256, verifier.as_bytes()).as_ref());
    Ok(Pkce {
        verifier,
        challenge,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_a_base64url_verifier_and_its_s256_challenge() {
        let pkce = generate_pkce().unwrap();
        assert_eq!(pkce.verifier.len(), 43);
        assert!(
            pkce.verifier
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        );
        assert_eq!(
            pkce.challenge,
            base64url_encode(digest(&SHA256, pkce.verifier.as_bytes()).as_ref())
        );
        assert_ne!(generate_pkce().unwrap().verifier, pkce.verifier);
        // RFC 7636 appendix B test vector.
        assert_eq!(
            base64url_encode(
                digest(&SHA256, b"dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk").as_ref()
            ),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }
}
