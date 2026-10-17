//! Port of durable `src/storage/sqlite/` (`index.ts`): the portable
//! [`SqliteStorage`] core and its schema, plus the `rusqlite` adapter behind
//! the `durable-sqlite` feature (Pi's `node.ts`).

/// Build a binding list: `sqlite_params![id, "text", None::<i64>]`.
macro_rules! sqlite_params {
    ($($value:expr),* $(,)?) => {
        vec![$($crate::durable::storage::sqlite::SqliteValue::from($value)),*]
    };
}

pub mod database;
pub mod migrations;
#[cfg(feature = "durable-sqlite")]
pub mod rusqlite;
pub mod storage;

#[cfg(all(test, feature = "durable-sqlite"))]
mod tests;

pub use database::{
    AggregateError, SqliteDatabase, SqliteDatabaseExt, SqliteExecutor, SqliteRow,
    SqliteTransactionCallback, SqliteValue,
};
pub use migrations::{
    CURRENT_SQLITE_SCHEMA_VERSION, SqliteMigration, apply_sqlite_migrations, sqlite_migrations,
};
pub use storage::SqliteStorage;
