//! Port of Pi's `@earendil-works/pi-agent-core` 1.0 (`packages/agent`).
//!
//! A stateful [`Agent`] around a low-level loop ([`agent_loop`],
//! [`run_agent_loop`]) that streams assistant responses, executes tools,
//! and keeps the transcript, including the system messages that carry the
//! prompt and tool declarations. Divergences from Pi are documented on the
//! modules and items involved.

#[allow(clippy::module_inception)]
pub mod agent;
pub mod agent_loop;
pub mod error;
pub mod proxy;
pub mod stream_fn;
pub mod types;

pub use agent::{
    Agent, AgentInitialState, AgentOptions, AgentOptionsBuilder, AgentPrepareNextTurnFn,
    AgentPrepareNextTurnWithContextFn, AgentSubscription,
};
pub use agent_loop::{
    AgentEventStream, RunToolCallOptions, ToolCallHooks, ToolUpdateCallback, agent_loop,
    agent_loop_continue, run_agent_loop, run_agent_loop_continue, run_tool_call,
};
pub use error::{AgentError, AgentResult};
pub use proxy::{ProxyAssistantMessageEvent, ProxyStreamOptions, stream_proxy};
pub use stream_fn::{get_default_stream_fn, set_default_stream_fn, stream_fn, stream_simple_fn};
pub use types::*;

/// Pi's agent `ThinkingLevel` (`"off"` plus the pi-ai levels), exported from
/// `types.ts`. It lives here rather than in [`types`] because the crate root
/// re-exports [`types`] beside pi-ai's own `ThinkingLevel`, which has no
/// `off`.
pub type ThinkingLevel = crate::types::ModelThinkingLevel;
