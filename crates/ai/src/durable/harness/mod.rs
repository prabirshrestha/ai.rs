//! Port of durable `src/harness/`: the durable agent Harness over one Session.
//!
//! See `/mnt/project-files/pi-port/progress/durable-m6-m8.md` for the divergences recorded while porting.

pub mod agent;
pub mod compaction;
pub mod context;
pub mod define;
pub mod events;
pub mod generation;
#[allow(clippy::module_inception)]
pub mod harness;
pub mod inbox;
pub mod live;
pub mod output;
pub mod prompt;
pub mod provider;
pub mod registry;
pub mod scheduler;
pub mod submissions;
pub mod task_graph;
pub mod tool;
pub mod types;
pub mod usage;
pub mod util;
pub mod view;

#[cfg(test)]
mod tests;

pub use agent::{AGENT_DOC, INSTRUCTIONS_KEY};
pub use define::{
    define_extension, define_tool, hook, section, text_section, wrap_section, wrap_tool,
};
pub use harness::{Conversation, ConversationHandle, CreateOptions, Harness};
pub use registry::{Registry, RegistryReader, RegistrySnapshot, create_registry};
pub use scheduler::{AbortTaskResult, HookApi, HookRunner, TaskRuntime};
pub use submissions::{AbortSubmissionResult, Submission};
pub use types::*;
pub use usage::{USAGE_DOC, UsageState};
