//! Ports of the Pi `harness-*.test.ts` suites and examples.

#![allow(clippy::type_complexity)]

pub(crate) mod support;

mod chat;
mod context;
mod conversations;
mod examples;
mod examples_chat;
mod generation;
mod generation_recovery;
mod inbox;
mod lifecycle;
mod live_deltas;
mod output;
mod ownership;
mod prompt;
mod registry;
mod submissions;
mod tasks;
mod tasks_recovery;
mod tools;
mod tools_recovery;
