//! Port of durable `src/storage/sqlite/database.ts`: the minimal asynchronous
//! SQLite facade `SqliteStorage` runs on.
//!
//! Divergences from Pi:
//! - Methods are plain functions returning boxed futures instead of `async`
//!   methods, so an adapter can enqueue the operation when it is called, as a
//!   JS promise-returning method starts synchronously. `SqliteStorage` relies
//!   only on the order of operations it awaits.
//! - Variadic bindings become a `Vec<SqliteValue>`; rows are ordered
//!   column-name maps (`T extends object` in TS).
//! - The generic `transaction<T>` is not object safe, so adapters implement
//!   [`SqliteDatabase::transaction_dyn`] over type-erased results and callers use
//!   the generic [`SqliteDatabaseExt::transaction`]. The callback receives an
//!   owned [`Arc`] handle, which stays invalid after the transaction settles.
//! - The rollback-failure `AggregateError` is [`AggregateError`], wrapped in
//!   [`Error::Thrown`].

use std::any::Any;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;

use crate::durable::errors::{Error, Result};

/// Values supported by the portable SQLite storage core (`SqliteValue`).
#[derive(Debug, Clone, PartialEq)]
pub enum SqliteValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl SqliteValue {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            _ => None,
        }
    }
}

impl From<i64> for SqliteValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<i32> for SqliteValue {
    fn from(value: i32) -> Self {
        Self::Integer(value.into())
    }
}

impl From<u32> for SqliteValue {
    fn from(value: u32) -> Self {
        Self::Integer(value.into())
    }
}

/// IDs and sequences never exceed `Number.MAX_SAFE_INTEGER`, so they fit an `i64`.
impl From<u64> for SqliteValue {
    fn from(value: u64) -> Self {
        Self::Integer(value as i64)
    }
}

impl From<f64> for SqliteValue {
    fn from(value: f64) -> Self {
        Self::Real(value)
    }
}

impl From<&str> for SqliteValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for SqliteValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<u8>> for SqliteValue {
    fn from(value: Vec<u8>) -> Self {
        Self::Blob(value)
    }
}

impl<T: Into<SqliteValue>> From<Option<T>> for SqliteValue {
    fn from(value: Option<T>) -> Self {
        value.map_or(Self::Null, Into::into)
    }
}

/// One result row: column name to value, in column order.
pub type SqliteRow = IndexMap<String, SqliteValue>;

/// Asynchronous SQL operations shared by a database and its transaction handles (`SqliteExecutor`).
///
/// `exec` runs SQL text without bindings and may contain several statements. `run`, `get`, and `all`
/// execute one statement with positional bindings. Adapters may cache prepared statements by SQL text,
/// so callers pass values as bindings instead of interpolating them.
pub trait SqliteExecutor: Send + Sync {
    fn exec(&self, sql: &str) -> BoxFuture<'static, Result<()>>;
    fn run(&self, sql: &str, params: Vec<SqliteValue>) -> BoxFuture<'static, Result<()>>;
    fn get(
        &self,
        sql: &str,
        params: Vec<SqliteValue>,
    ) -> BoxFuture<'static, Result<Option<SqliteRow>>>;
    fn all(
        &self,
        sql: &str,
        params: Vec<SqliteValue>,
    ) -> BoxFuture<'static, Result<Vec<SqliteRow>>>;
}

/// Type-erased transaction callback (see [`SqliteDatabaseExt::transaction`]).
pub type SqliteTransactionCallback<'a> = Box<
    dyn FnOnce(Arc<dyn SqliteExecutor>) -> BoxFuture<'a, Result<Box<dyn Any + Send>>> + Send + 'a,
>;

/// Minimal database facade required by `SqliteStorage` (`SqliteDatabase`).
///
/// All operations are asynchronous so adapters may execute outside the harness runtime.
///
/// `transaction` passes the callback a transaction handle. All work in the transaction
/// must use that handle; the handle is invalid after the callback settles. Adapters must
/// queue unrelated operations and other transactions until the transaction finishes. The
/// returned future settles after commit or rollback. Calling the database itself (including
/// `transaction` or `close`) from inside a callback therefore waits for that transaction and
/// never settles.
///
/// When the callback rejects, the adapter must roll the transaction back before rejecting
/// with that same error. If rollback fails, it must reject with a different error (for
/// example an [`AggregateError`]) so callers cannot mistake the callback error for a
/// guaranteed rollback.
pub trait SqliteDatabase: SqliteExecutor {
    fn transaction_dyn<'a>(
        &'a self,
        callback: SqliteTransactionCallback<'a>,
    ) -> BoxFuture<'a, Result<Box<dyn Any + Send>>>;
    fn close(&self) -> BoxFuture<'static, Result<()>>;
}

/// The generic `transaction<T>` of `SqliteDatabase`.
pub trait SqliteDatabaseExt: SqliteDatabase {
    fn transaction<'a, T, F, Fut>(&'a self, callback: F) -> BoxFuture<'a, Result<T>>
    where
        T: Send + 'static,
        F: FnOnce(Arc<dyn SqliteExecutor>) -> Fut + Send + 'a,
        Fut: Future<Output = Result<T>> + Send + 'a,
    {
        let settled = self.transaction_dyn(Box::new(move |transaction| {
            Box::pin(async move {
                callback(transaction)
                    .await
                    .map(|value| Box::new(value) as Box<dyn Any + Send>)
            })
        }));
        Box::pin(async move {
            let value = settled.await?;
            Ok(*value
                .downcast::<T>()
                .expect("SQLite transaction adapters return the callback's value"))
        })
    }
}

impl<D: SqliteDatabase + ?Sized> SqliteDatabaseExt for D {}

/// JS `AggregateError`: several errors reported together.
#[derive(Debug, Clone)]
pub struct AggregateError {
    pub errors: Vec<Error>,
    pub message: String,
}

impl AggregateError {
    pub fn new(errors: Vec<Error>, message: impl Into<String>) -> Self {
        Self {
            errors,
            message: message.into(),
        }
    }
}

impl fmt::Display for AggregateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AggregateError {}
