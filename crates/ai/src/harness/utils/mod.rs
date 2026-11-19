//! Port of `@ai-sdk/harness/utils`: helpers harness adapters share.

pub mod ai_gateway_auth;
pub mod authentication_environment;
pub mod bridge_asset;
pub mod bridge_auth;
pub mod bridge_auth_protocol;
pub mod bridge_diagnostics;
pub mod bridge_endpoint;
pub mod bridge_ready;
pub mod bridge_user_message_submitter;
pub mod bridge_web_socket;
pub mod classify_disk_log;
pub mod credential_forwarding;
pub mod extract_harness_prompt_content;
pub mod is_sandbox_write_protected_path;
pub mod native_subscription;
pub mod oauth_access_token;
pub mod os;
pub mod sandbox_channel;
pub mod sandbox_credential_brokering;
pub mod sandbox_home_dir;
pub mod shell_quote;
pub mod sleep;
pub mod tool_model_output;
pub mod tungstenite_web_socket;
pub mod write_instructions;
pub mod write_skills;

pub use crate::harness::spec::paths::encode_harness_path_segment;
pub use ai_gateway_auth::*;
pub use authentication_environment::*;
pub use bridge_asset::*;
pub use bridge_auth::{BridgeAuthKeyPair, create_bridge_auth_key_pair};
pub use bridge_auth_protocol::BRIDGE_AUTH_PUBLIC_KEY_ENV;
pub use bridge_diagnostics::{
    BridgeErrorInput, BridgeProcessStreamName, BridgeStreamDone, ForwardBridgeProcessStreamOptions,
    LogBridgeErrorOptions, create_bridge_error_handler, create_bridge_startup_error,
    drain_bridge_process_stream, format_bridge_error, forward_bridge_process_stream,
    log_bridge_error,
};
pub use bridge_endpoint::*;
pub use bridge_ready::*;
pub use bridge_user_message_submitter::*;
pub use bridge_web_socket::*;
pub use classify_disk_log::*;
pub use credential_forwarding::*;
pub use extract_harness_prompt_content::{
    HarnessPromptContent, HarnessPromptContentPart, extract_harness_prompt_content,
};
pub use is_sandbox_write_protected_path::*;
pub use native_subscription::{
    get_jwt_expires_at, parse_jwt_payload, read_linux_secret_service_password,
    read_macos_keychain_password, read_windows_credential_manager_password,
    should_resolve_native_subscription,
};
pub use oauth_access_token::*;
pub use os::*;
pub use sandbox_channel::*;
pub use sandbox_credential_brokering::*;
pub use sandbox_home_dir::*;
pub use shell_quote::*;
pub use sleep::*;
pub use tool_model_output::*;
pub use tungstenite_web_socket::*;
pub use write_instructions::*;
pub use write_skills::*;

use crate::harness::errors::BoxError;

/// `new Error(message)`: a plain error that `get_harness_error_message`
/// masks.
pub(crate) fn plain_error(message: impl Into<String>) -> BoxError {
    message.into().into()
}

/// `crypto.randomUUID()`: a random version 4 UUID.
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
    fn locale_compare_matches_icu_for_common_cases() {
        assert_eq!(locale_compare("reference.md", "SKILL.md"), Ordering::Less);
        assert_eq!(locale_compare("alpha", "zeta"), Ordering::Less);
        assert_eq!(locale_compare("a", "A"), Ordering::Less);
        assert_eq!(locale_compare("_x", "1"), Ordering::Less);
        assert_eq!(locale_compare("1", "a"), Ordering::Less);
        assert_eq!(locale_compare("same", "same"), Ordering::Equal);
    }
}
