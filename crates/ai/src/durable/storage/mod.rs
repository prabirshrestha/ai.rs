//! Storage backends (`src/storage/`): the in-memory reference storage, the
//! portable SQLite core and the portable JSONL storage. The `rusqlite` adapter
//! needs `durable-sqlite`; the local JSONL adapter needs `durable-local-env`.
//!
//! Pi's `storage-runtime-boundary.test.ts` (the package root and the portable
//! SQLite, environment and JSONL subpaths import no Node modules) has no Rust
//! test: the boundary is the cargo features above, and
//! `cargo test -p ai --no-default-features` builds the portable cores without
//! `rusqlite` or the local environment.

pub mod jsonl;
pub mod memory;
pub mod sqlite;

#[cfg(test)]
mod memory_tests;
#[cfg(all(test, any(feature = "durable-sqlite", feature = "durable-local-env")))]
pub(crate) mod test_support;
