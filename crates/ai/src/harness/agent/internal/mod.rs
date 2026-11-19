//! Port of `agent/internal`.
//!
//! [rust] `get-own.ts` is not ported: Rust maps have no prototype chain, so
//! `map.get(key)` already reads own entries only.

pub mod bootstrap_recipe;
pub mod deferred_sandbox_setup;
pub mod harness_stream_text_result;
pub mod lifecycle_state_validation;
pub mod on_bootstrap_marker;
pub mod permission_mode;
pub mod prepared_sandbox_identity;
pub mod resolve_observability;
pub mod run_prompt;
pub mod sandbox_bootstrap;
pub mod sandbox_operations;
pub mod strip_work_dir;
pub mod to_harness_stream;
pub mod tool_filtering;
pub mod translate_stream_part;
pub mod turn_telemetry;

/// SHA-256 of `\0`-terminated strings (the `pushString` pattern), as hex
/// of the first `bytes` digest bytes.
pub(crate) fn hash_strings(values: &[&str], bytes: usize) -> String {
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
    for value in values {
        ctx.update(value.as_bytes());
        ctx.update(b"\0");
    }
    hex(&ctx.finish().as_ref()[..bytes])
}

/// SHA-256 of `value`, as full lowercase hex.
pub(crate) fn sha256_hex(value: &str) -> String {
    hex(ring::digest::digest(&ring::digest::SHA256, value.as_bytes()).as_ref())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
