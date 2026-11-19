//! Port of `@ai-sdk/harness-pi`: Pi through the `pi` executable in RPC
//! mode. Enabled by the `harness-pi` cargo feature.
//!
//! [rust] `@ai-sdk/harness-pi` runs `@earendil-works/pi-coding-agent` as a
//! library inside the host Node process. A Rust host cannot, so this port
//! runs the real `pi` CLI (`pi --mode rpc`) inside the sandbox, with a
//! bundled extension (`ai-sdk-harness-pi-extension.ts`) that hands host
//! tools, built-in tools and approvals back to the host through Pi's RPC
//! extension-UI protocol. Built-in tools still execute host-side against
//! the sandbox, as in TS.

pub mod pi_events;
pub mod pi_paths;
pub mod pi_translate;
pub mod pi_tool_result;
pub mod pi_utils;

#[cfg(test)]
mod pi_translate_tests;

/// `VERSION` of the ported `@ai-sdk/harness-pi` package.
pub const VERSION: &str = "1.0.191";
