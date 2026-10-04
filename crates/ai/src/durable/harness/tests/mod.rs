//! Ports of the Pi `harness-*.test.ts` suites and examples.

#![allow(clippy::type_complexity)]

pub(crate) mod support;

mod chat;
mod compaction;
mod compaction_auto;
mod compaction_more;
mod compaction_support;
mod context;
mod conversations;
mod events;
mod examples;
mod examples_chat;
mod examples_events;
mod generation;
mod generation_recovery;
mod inbox;
mod inspect;
mod lifecycle;
mod live_deltas;
mod output;
mod ownership;
mod prompt;
mod registry;
mod submissions;
mod task_graph;
mod tasks;
mod tasks_recovery;
mod tools;
mod tools_recovery;
mod view;
