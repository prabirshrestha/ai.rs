//! Storage backends (`src/storage/`). SQLite and JSONL follow in later milestones.

pub mod memory;

#[cfg(test)]
mod memory_tests;
