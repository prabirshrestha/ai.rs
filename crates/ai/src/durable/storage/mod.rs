//! Storage backends (`src/storage/`): the in-memory reference storage, the
//! portable SQLite core and the portable JSONL storage. The `rusqlite` adapter
//! needs `durable-sqlite`; the local JSONL adapter needs `durable-local-env`.
//!
//! Pi's `storage-runtime-boundary.test.ts` (the package root and the portable
//! SQLite, environment and JSONL subpaths import no Node modules) has no Rust
//! test: the boundary is the cargo features above, and
//! `cargo test -p ai --no-default-features` builds the portable cores without
//! `rusqlite` or the local environment.

use crate::chord::JsonValue;
use crate::durable::errors::{Error, Result};

pub mod jsonl;
pub mod memory;
pub mod sqlite;

#[cfg(test)]
mod memory_tests;
#[cfg(all(test, any(feature = "durable-sqlite", feature = "durable-local-env")))]
pub(crate) mod test_support;

const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

/// A cursor's `after` ID. Like Pi's `Number.isSafeInteger` check, an
/// integral float (`3.0`) reads as its integer. Divergence: IDs are `u64`, so
/// a negative cursor is rejected; Pi accepts one (no storage produces it).
pub(crate) fn cursor_after(after: &JsonValue) -> Result<Option<u64>> {
    let invalid = || Error::type_error("Invalid storage cursor");
    if let Some(after) = after.as_u64() {
        return if after as f64 <= MAX_SAFE_INTEGER {
            Ok(Some(after))
        } else {
            Err(invalid())
        };
    }
    match after.as_f64() {
        Some(number) if number.fract() == 0.0 && (0.0..=MAX_SAFE_INTEGER).contains(&number) => {
            Ok(Some(number as u64))
        }
        _ => Err(invalid()),
    }
}
