//! Storage backends (`src/storage/`): the in-memory reference storage, the
//! portable SQLite core and the portable JSONL storage. The `rusqlite` adapter
//! needs `durable-sqlite`; the local JSONL adapter needs `durable-local-env`.

pub mod jsonl;
pub mod memory;
pub mod sqlite;

#[cfg(test)]
mod memory_tests;
#[cfg(all(test, any(feature = "durable-sqlite", feature = "durable-local-env")))]
pub(crate) mod test_support;
