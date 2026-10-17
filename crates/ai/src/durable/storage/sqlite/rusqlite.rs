//! Port of durable `src/storage/sqlite/node.ts` over `rusqlite` (bundled
//! SQLite), behind the `durable-sqlite` feature.
//!
//! Divergences from Pi:
//! - `node:sqlite`'s `DatabaseSync` plus `SerialOperationQueue` becomes one
//!   dedicated OS thread that owns the connection and runs jobs in the order
//!   they were called. A transaction occupies the thread until it settles:
//!   the thread runs `BEGIN IMMEDIATE`, then only that transaction's jobs,
//!   then `COMMIT` or `ROLLBACK`, so every other operation (including ones the
//!   callback starts on the database itself) waits behind it, exactly like
//!   the TS barrier. Blocking SQLite calls never run on tokio workers.
//! - Operations enqueue when called (they are not lazy `async fn`s), keeping
//!   Pi's call-order semantics.
//! - Prepared statements are cached per connection by `rusqlite`'s
//!   statement cache (capacity [`STATEMENT_CACHE_CAPACITY`]) rather than an
//!   unbounded map.
//! - A transaction future dropped before it settles rolls the transaction back.
//! - `openNodeSqliteDatabase`/`openNodeSqliteStorage` are
//!   [`open_rusqlite_database`]/[`open_rusqlite_storage`], with
//!   `open_local_sqlite_storage` as an alias of the latter.

use std::any::Any;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use ::rusqlite::types::{Value, ValueRef};
use ::rusqlite::{Connection, params_from_iter};
use futures::future::BoxFuture;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::durable::errors::{Error, Result};

use super::database::{
    AggregateError, SqliteDatabase, SqliteExecutor, SqliteRow, SqliteTransactionCallback,
    SqliteValue,
};
use super::storage::SqliteStorage;

/// Prepared statements cached per connection.
pub const STATEMENT_CACHE_CAPACITY: usize = 1024;

const DEFAULT_WAL_AUTO_CHECKPOINT_PAGES: u32 = 1_000;
const DEFAULT_BUSY_TIMEOUT_MS: u64 = 5_000;

/// SQLite connection settings for a durable storage file (`NodeSqliteStorageOptions`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RusqliteStorageOptions {
    /// SQLite WAL auto-checkpoint threshold. SQLite and this adapter default to 1,000 pages; 0 disables it.
    pub wal_auto_checkpoint_pages: Option<u32>,
    /// Time SQLite waits for a competing file lock. SQLite defaults to 0; this adapter defaults to 5,000 ms.
    pub busy_timeout_ms: Option<u64>,
}

type ConnectionJob = Box<dyn FnOnce(&mut Option<Connection>) + Send>;

enum Job {
    Operation(ConnectionJob),
    Transaction {
        jobs: mpsc::Receiver<TransactionJob>,
        started: oneshot::Sender<Result<()>>,
    },
}

enum TransactionJob {
    Operation(ConnectionJob),
    Finish {
        commit: bool,
        done: oneshot::Sender<Result<()>>,
    },
}

fn sqlite_error(error: ::rusqlite::Error) -> Error {
    Error::message(error.to_string())
}

fn not_open() -> Error {
    Error::message("database is not open")
}

fn stopped() -> Error {
    Error::message("SQLite connection thread stopped")
}

fn to_value(value: SqliteValue) -> Value {
    match value {
        SqliteValue::Null => Value::Null,
        SqliteValue::Integer(value) => Value::Integer(value),
        SqliteValue::Real(value) => Value::Real(value),
        SqliteValue::Text(value) => Value::Text(value),
        SqliteValue::Blob(value) => Value::Blob(value),
    }
}

fn from_value_ref(value: ValueRef<'_>) -> SqliteValue {
    match value {
        ValueRef::Null => SqliteValue::Null,
        ValueRef::Integer(value) => SqliteValue::Integer(value),
        ValueRef::Real(value) => SqliteValue::Real(value),
        ValueRef::Text(value) => SqliteValue::Text(String::from_utf8_lossy(value).into_owned()),
        ValueRef::Blob(value) => SqliteValue::Blob(value.to_vec()),
    }
}

fn open_connection(connection: &mut Option<Connection>) -> Result<&mut Connection> {
    connection.as_mut().ok_or_else(not_open)
}

fn exec_sql(connection: &mut Option<Connection>, sql: &str) -> Result<()> {
    open_connection(connection)?
        .execute_batch(sql)
        .map_err(sqlite_error)
}

fn run_sql(connection: &mut Option<Connection>, sql: &str, params: Vec<SqliteValue>) -> Result<()> {
    let connection = open_connection(connection)?;
    let mut statement = connection.prepare_cached(sql).map_err(sqlite_error)?;
    let mut rows = statement
        .query(params_from_iter(params.into_iter().map(to_value)))
        .map_err(sqlite_error)?;
    while rows.next().map_err(sqlite_error)?.is_some() {}
    Ok(())
}

fn all_sql(
    connection: &mut Option<Connection>,
    sql: &str,
    params: Vec<SqliteValue>,
    limit: Option<usize>,
) -> Result<Vec<SqliteRow>> {
    let connection = open_connection(connection)?;
    let mut statement = connection.prepare_cached(sql).map_err(sqlite_error)?;
    let names: Vec<String> = statement
        .column_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut rows = statement
        .query(params_from_iter(params.into_iter().map(to_value)))
        .map_err(sqlite_error)?;
    let mut values = Vec::new();
    while let Some(row) = rows.next().map_err(sqlite_error)? {
        let mut value = SqliteRow::with_capacity(names.len());
        for (index, name) in names.iter().enumerate() {
            value.insert(
                name.clone(),
                from_value_ref(row.get_ref(index).map_err(sqlite_error)?),
            );
        }
        values.push(value);
        if limit.is_some_and(|limit| values.len() >= limit) {
            break;
        }
    }
    Ok(values)
}

fn close_connection(connection: &mut Option<Connection>) -> Result<()> {
    let Some(open) = connection.take() else {
        return Ok(());
    };
    let checkpoint = open
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .map_err(sqlite_error);
    let closed = open.close().map_err(|(_, error)| sqlite_error(error));
    checkpoint.and(closed)
}

fn run_transaction(
    connection: &mut Option<Connection>,
    jobs: mpsc::Receiver<TransactionJob>,
    started: oneshot::Sender<Result<()>>,
) {
    if let Err(error) = exec_sql(connection, "BEGIN IMMEDIATE") {
        let _ = started.send(Err(error));
        return;
    }
    if started.send(Ok(())).is_err() {
        let _ = exec_sql(connection, "ROLLBACK");
        return;
    }
    loop {
        match jobs.recv() {
            Ok(TransactionJob::Operation(job)) => job(connection),
            Ok(TransactionJob::Finish { commit, done }) => {
                let result = exec_sql(connection, if commit { "COMMIT" } else { "ROLLBACK" });
                let failed_commit = commit && result.is_err();
                let _ = done.send(result);
                // A failed COMMIT leaves the transaction open for the caller's ROLLBACK.
                if !failed_commit {
                    return;
                }
            }
            Err(_) => {
                // The transaction future was dropped before it settled.
                let _ = exec_sql(connection, "ROLLBACK");
                return;
            }
        }
    }
}

fn connection_thread(mut connection: Option<Connection>, jobs: mpsc::Receiver<Job>) {
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Operation(job) => job(&mut connection),
            Job::Transaction { jobs, started } => run_transaction(&mut connection, jobs, started),
        }
    }
}

/// Send a connection job through `send` and await its result.
fn submit<T: Send + 'static>(
    send: impl FnOnce(ConnectionJob) -> bool,
    operation: impl FnOnce(&mut Option<Connection>) -> Result<T> + Send + 'static,
) -> BoxFuture<'static, Result<T>> {
    let (result, settled) = oneshot::channel();
    let sent = send(Box::new(move |connection| {
        let _ = result.send(operation(connection));
    }));
    Box::pin(async move {
        if !sent {
            return Err(stopped());
        }
        settled.await.map_err(|_| stopped())?
    })
}

/// Executes SQL on the connection thread (`NodeSqliteExecutor`).
trait ConnectionExecutor: Send + Sync {
    fn submit<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Option<Connection>) -> Result<T> + Send + 'static,
    ) -> BoxFuture<'static, Result<T>>;
}

macro_rules! impl_sqlite_executor {
    ($type:ty) => {
        impl SqliteExecutor for $type {
            fn exec(&self, sql: &str) -> BoxFuture<'static, Result<()>> {
                let sql = sql.to_owned();
                self.submit(move |connection| exec_sql(connection, &sql))
            }

            fn run(&self, sql: &str, params: Vec<SqliteValue>) -> BoxFuture<'static, Result<()>> {
                let sql = sql.to_owned();
                self.submit(move |connection| run_sql(connection, &sql, params))
            }

            fn get(
                &self,
                sql: &str,
                params: Vec<SqliteValue>,
            ) -> BoxFuture<'static, Result<Option<SqliteRow>>> {
                let sql = sql.to_owned();
                self.submit(move |connection| Ok(all_sql(connection, &sql, params, Some(1))?.pop()))
            }

            fn all(
                &self,
                sql: &str,
                params: Vec<SqliteValue>,
            ) -> BoxFuture<'static, Result<Vec<SqliteRow>>> {
                let sql = sql.to_owned();
                self.submit(move |connection| all_sql(connection, &sql, params, None))
            }
        }
    };
}

/// A transaction handle (`NodeSqliteTransaction`): valid only while its transaction is active.
struct RusqliteTransaction {
    jobs: Mutex<Option<mpsc::Sender<TransactionJob>>>,
    active: Arc<AtomicBool>,
}

impl ConnectionExecutor for RusqliteTransaction {
    fn submit<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Option<Connection>) -> Result<T> + Send + 'static,
    ) -> BoxFuture<'static, Result<T>> {
        if !self.active.load(Ordering::SeqCst) {
            return Box::pin(async {
                Err(Error::message(
                    "SQLite transaction handle is no longer active",
                ))
            });
        }
        let jobs = self.jobs.lock().clone();
        submit(
            move |job| jobs.is_some_and(|jobs| jobs.send(TransactionJob::Operation(job)).is_ok()),
            operation,
        )
    }
}

impl_sqlite_executor!(RusqliteTransaction);

/// Ends a transaction handle's validity when the transaction settles or its future is dropped.
struct TransactionScope {
    handle: Arc<RusqliteTransaction>,
}

impl TransactionScope {
    fn finish(&self) -> Option<mpsc::Sender<TransactionJob>> {
        self.handle.active.store(false, Ordering::SeqCst);
        self.handle.jobs.lock().take()
    }
}

impl Drop for TransactionScope {
    fn drop(&mut self) {
        self.finish();
    }
}

async fn finish_transaction(jobs: &mpsc::Sender<TransactionJob>, commit: bool) -> Result<()> {
    let (done, finished) = oneshot::channel();
    jobs.send(TransactionJob::Finish { commit, done })
        .map_err(|_| stopped())?;
    finished.await.map_err(|_| stopped())?
}

/// `SqliteDatabase` adapter backed by a bundled SQLite connection (`NodeSqliteDatabase`).
pub struct RusqliteDatabase {
    jobs: mpsc::Sender<Job>,
}

impl std::fmt::Debug for RusqliteDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RusqliteDatabase").finish_non_exhaustive()
    }
}

impl RusqliteDatabase {
    /// Wrap an open connection; its thread owns it from now on.
    pub fn new(connection: Connection) -> Self {
        connection.set_prepared_statement_cache_capacity(STATEMENT_CACHE_CAPACITY);
        let (jobs, receiver) = mpsc::channel();
        std::thread::Builder::new()
            .name("durable-sqlite".into())
            .spawn(move || connection_thread(Some(connection), receiver))
            .expect("spawn the SQLite connection thread");
        Self { jobs }
    }

    fn send(&self, job: Job) -> bool {
        self.jobs.send(job).is_ok()
    }
}

impl ConnectionExecutor for RusqliteDatabase {
    fn submit<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Option<Connection>) -> Result<T> + Send + 'static,
    ) -> BoxFuture<'static, Result<T>> {
        submit(|job| self.send(Job::Operation(job)), operation)
    }
}

impl_sqlite_executor!(RusqliteDatabase);

impl SqliteDatabase for RusqliteDatabase {
    fn transaction_dyn<'a>(
        &'a self,
        callback: SqliteTransactionCallback<'a>,
    ) -> BoxFuture<'a, Result<Box<dyn Any + Send>>> {
        let (jobs, receiver) = mpsc::channel();
        let (started, begun) = oneshot::channel();
        let sent = self.send(Job::Transaction {
            jobs: receiver,
            started,
        });
        Box::pin(async move {
            if !sent {
                return Err(stopped());
            }
            begun.await.map_err(|_| stopped())??;
            let scope = TransactionScope {
                handle: Arc::new(RusqliteTransaction {
                    jobs: Mutex::new(Some(jobs.clone())),
                    active: Arc::new(AtomicBool::new(true)),
                }),
            };
            let result = callback(scope.handle.clone()).await;
            scope.finish();
            match result {
                Ok(value) => match finish_transaction(&jobs, true).await {
                    Ok(()) => Ok(value),
                    Err(error) => match finish_transaction(&jobs, false).await {
                        Ok(()) => Err(error),
                        Err(rollback_error) => Err(Error::thrown(AggregateError::new(
                            vec![error, rollback_error],
                            "SQLite transaction failed and rollback failed",
                        ))),
                    },
                },
                Err(error) => match finish_transaction(&jobs, false).await {
                    Ok(()) => Err(error),
                    Err(rollback_error) => Err(Error::thrown(AggregateError::new(
                        vec![error, rollback_error],
                        "SQLite transaction failed and rollback failed",
                    ))),
                },
            }
        })
    }

    fn close(&self) -> BoxFuture<'static, Result<()>> {
        self.submit(close_connection)
    }
}

/// Open and configure a SQLite database facade (`openNodeSqliteDatabase`).
pub async fn open_rusqlite_database(
    path: impl AsRef<Path>,
    options: RusqliteStorageOptions,
) -> Result<RusqliteDatabase> {
    let path = path.as_ref().to_owned();
    let checkpoint_pages = options
        .wal_auto_checkpoint_pages
        .unwrap_or(DEFAULT_WAL_AUTO_CHECKPOINT_PAGES);
    let timeout = options.busy_timeout_ms.unwrap_or(DEFAULT_BUSY_TIMEOUT_MS);
    let connection = tokio::task::spawn_blocking(move || -> Result<Connection> {
        if path.as_os_str() != ":memory:"
            && let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|error| Error::message(error.to_string()))?;
        }
        let connection = Connection::open(&path).map_err(sqlite_error)?;
        connection
            .busy_timeout(Duration::from_millis(timeout))
            .map_err(sqlite_error)?;
        Ok(connection)
    })
    .await
    .map_err(|error| Error::message(error.to_string()))??;
    let adapter = RusqliteDatabase::new(connection);
    let configured = async {
        adapter.exec("PRAGMA journal_mode = WAL").await?;
        adapter.exec("PRAGMA synchronous = NORMAL").await?;
        adapter
            .exec(&format!("PRAGMA wal_autocheckpoint = {checkpoint_pages}"))
            .await
    }
    .await;
    match configured {
        Ok(()) => Ok(adapter),
        Err(error) => {
            // Preserve the configuration failure.
            let _ = adapter.close().await;
            Err(error)
        }
    }
}

/// Open or create file-backed durable storage using bundled SQLite (`openNodeSqliteStorage`).
pub async fn open_rusqlite_storage(
    path: impl AsRef<Path>,
    options: RusqliteStorageOptions,
) -> Result<SqliteStorage> {
    SqliteStorage::open(Arc::new(open_rusqlite_database(path, options).await?)).await
}

/// Alias of [`open_rusqlite_storage`] named after the local Node adapter.
pub async fn open_local_sqlite_storage(
    path: impl AsRef<Path>,
    options: RusqliteStorageOptions,
) -> Result<SqliteStorage> {
    open_rusqlite_storage(path, options).await
}
