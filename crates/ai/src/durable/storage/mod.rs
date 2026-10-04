//! Storage backends (`src/storage/`): the in-memory reference storage and the
//! portable SQLite core. The `rusqlite` adapter needs `durable-sqlite`.

pub mod memory;
pub mod sqlite;

#[cfg(test)]
mod memory_tests;
#[cfg(test)]
pub(crate) mod test_support;
