//! Host-side helpers of `harness-acp/src/v1/bridge`.
//!
//! [rust] The bridge itself runs inside the sandbox as the vendored
//! JavaScript bundle; only the modules the host imports are ported.

pub mod bridge_environment;
pub mod canonical_json_fingerprint;
pub mod profile_values;
pub mod protocol_configuration;
