//! Helpers shared by the harness adapters.

pub mod ai_gateway_auth;
pub mod authentication_environment;
pub mod classify_disk_log;
pub mod credential_forwarding;
pub mod is_sandbox_write_protected_path;
pub mod native_subscription;
pub mod oauth_access_token;
pub mod os;
pub mod sandbox_credential_brokering;
pub mod sandbox_home_dir;
pub mod shell_quote;
pub mod sleep;
pub mod write_instructions;
pub mod write_skills;

pub use crate::harness::paths::encode_harness_path_segment;
pub use ai_gateway_auth::*;
pub use authentication_environment::*;
pub use classify_disk_log::*;
pub use credential_forwarding::*;
pub use is_sandbox_write_protected_path::*;
pub use native_subscription::{
    get_jwt_expires_at, parse_jwt_payload, read_linux_secret_service_password,
    read_macos_keychain_password, read_windows_credential_manager_password,
    should_resolve_native_subscription,
};
pub use oauth_access_token::*;
pub use os::*;
pub use sandbox_credential_brokering::*;
pub use sandbox_home_dir::*;
pub use shell_quote::*;
pub use sleep::*;
pub use write_instructions::*;
pub use write_skills::*;

use crate::harness::error::BoxError;

/// A plain error from a message.
pub(crate) fn plain_error(message: impl Into<String>) -> BoxError {
    message.into().into()
}

/// `new Date().toISOString()`.
#[allow(dead_code)] // [rust] Used only by some adapter features.
pub(crate) fn iso_timestamp_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    iso_timestamp(now.as_millis() as i64)
}

/// `new Date(millis).toISOString()`.
#[allow(dead_code)] // [rust] Used only by some adapter features.
pub(crate) fn iso_timestamp(millis: i64) -> String {
    let secs = millis.div_euclid(1000);
    let ms = millis.rem_euclid(1000);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{ms:03}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// `crypto.randomUUID()`: a random version 4 UUID.
// Unused when no adapter feature is on.
#[allow(dead_code)]
pub(crate) fn random_uuid() -> String {
    use ring::rand::SecureRandom as _;
    let mut bytes = [0u8; 16];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .expect("the system random number generator failed");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

/// `String.prototype.localeCompare` for the identifiers and paths the
/// harness sorts.
///
/// [rust] Approximates the ICU root collation: case-insensitive first, with
/// punctuation before digits before letters, then lowercase before
/// uppercase. This keeps hashes and orderings equal to TS for skill names
/// and file paths.
pub(crate) fn locale_compare(a: &str, b: &str) -> std::cmp::Ordering {
    fn primary(c: char) -> (u8, char) {
        let lower = c.to_lowercase().next().unwrap_or(c);
        let class = if c.is_alphabetic() {
            2
        } else if c.is_numeric() {
            1
        } else {
            0
        };
        (class, lower)
    }
    let pa = a.chars().map(primary);
    let pb = b.chars().map(primary);
    pa.cmp(pb).then_with(|| {
        // Tertiary: lowercase sorts before uppercase.
        let ta = a.chars().map(|c| c.is_uppercase());
        let tb = b.chars().map(|c| c.is_uppercase());
        ta.cmp(tb)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn iso_timestamp_matches_to_iso_string() {
        assert_eq!(super::iso_timestamp(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            super::iso_timestamp(1_709_251_199_123),
            "2024-02-29T23:59:59.123Z"
        );
    }

    #[test]
    fn locale_compare_matches_icu_for_common_cases() {
        assert_eq!(locale_compare("reference.md", "SKILL.md"), Ordering::Less);
        assert_eq!(locale_compare("alpha", "zeta"), Ordering::Less);
        assert_eq!(locale_compare("a", "A"), Ordering::Less);
        assert_eq!(locale_compare("_x", "1"), Ordering::Less);
        assert_eq!(locale_compare("1", "a"), Ordering::Less);
        assert_eq!(locale_compare("same", "same"), Ordering::Equal);
    }
}
