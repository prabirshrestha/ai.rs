//! Ports of the Pi `harness-*.test.ts` suites and examples.

#![allow(clippy::type_complexity)]

pub(crate) mod support;

mod chat;
mod context;
mod examples;
mod generation;
mod generation_recovery;
mod lifecycle;
mod output;
mod ownership;
mod prompt;
mod registry;
mod tasks;
mod tasks_recovery;
