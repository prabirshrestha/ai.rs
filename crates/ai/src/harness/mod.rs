//! Agent harnesses: agents that run their own loop inside a sandbox
//! (Claude Code, Codex, Pi, OpenCode, ACP agents and more), driven from
//! Rust and speaking ai.rs's own [`Message`](crate::Message),
//! [`AgentTool`](crate::AgentTool) and [`AgentEvent`](crate::AgentEvent)
//! types.
//!
//! Start with a [`Harness`] from [`adapters`], open a [`HarnessSession`]
//! with [`Harness::start`], and run turns with [`HarnessSession::prompt`].

pub mod adapters;
pub mod agent;
pub mod auth;
pub mod bootstrap;
pub mod error;
pub mod host_sandbox;
pub mod network;
pub mod paths;
pub mod probe;
pub mod questions_tool;
pub mod sandbox;
pub mod session;
pub mod tools;
pub mod utils;

#[cfg(test)]
pub(crate) mod test_support;

pub use agent::{
    HarnessAgent, HarnessAgentCallOptions, HarnessAgentSessionOptions, HarnessAgentSettings,
    HarnessCall, PrepareCallFn,
};
pub use auth::*;
pub use bootstrap::*;
pub use error::{BoxError, HarnessError, HarnessResult};
pub use host_sandbox::HostSandbox;
pub use network::*;
pub use probe::{HarnessCommandInfo, HarnessModelInfo, HarnessProbe};
pub use session::*;
pub use tools::*;
