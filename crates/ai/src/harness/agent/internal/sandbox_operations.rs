//! Port of `agent/internal/sandbox-operations.ts`.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::FutureExt as _;
use futures::future::{BoxFuture, Shared};
use parking_lot::Mutex;

use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::*;
use crate::harness::sdk::error_value::{SharedError, SharedErrorBox};

/// Resolves the session the operations run against.
pub type ResolveSandboxFn =
    Arc<dyn Fn() -> BoxFuture<'static, HarnessResult<Arc<dyn SandboxSession>>> + Send + Sync>;

/// Reads the session description at call time (TS: a `description` getter).
pub type SandboxDescriptionFn = Arc<dyn Fn() -> String + Send + Sync>;

/// `delegateSandboxOperations(resolve)`: every file and process operation
/// awaits `resolve()` and runs against the session it returns.
///
/// [rust] TS returns the operations without `description` and each caller
/// adds its own getter; here the caller passes it in.
pub fn delegate_sandbox_operations(
    resolve: ResolveSandboxFn,
    description: SandboxDescriptionFn,
) -> DelegatedSandboxSession {
    DelegatedSandboxSession {
        resolve,
        description,
    }
}

/// The session [`delegate_sandbox_operations`] returns.
#[derive(Clone)]
pub struct DelegatedSandboxSession {
    resolve: ResolveSandboxFn,
    description: SandboxDescriptionFn,
}

impl fmt::Debug for DelegatedSandboxSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DelegatedSandboxSession")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SandboxSession for DelegatedSandboxSession {
    fn description(&self) -> String {
        (self.description)()
    }

    async fn read_file(&self, o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
        (self.resolve)().await?.read_file(o).await
    }

    async fn read_binary_file(
        &self,
        o: SandboxReadBinaryFileOptions,
    ) -> HarnessResult<Option<Bytes>> {
        (self.resolve)().await?.read_binary_file(o).await
    }

    async fn read_text_file(&self, o: SandboxReadTextFileOptions) -> HarnessResult<Option<String>> {
        (self.resolve)().await?.read_text_file(o).await
    }

    async fn write_file(&self, o: SandboxWriteFileOptions) -> HarnessResult<()> {
        (self.resolve)().await?.write_file(o).await
    }

    async fn write_binary_file(&self, o: SandboxWriteBinaryFileOptions) -> HarnessResult<()> {
        (self.resolve)().await?.write_binary_file(o).await
    }

    async fn write_text_file(&self, o: SandboxWriteTextFileOptions) -> HarnessResult<()> {
        (self.resolve)().await?.write_text_file(o).await
    }

    async fn stat(&self, o: SandboxStatOptions) -> HarnessResult<Option<SandboxFileStat>> {
        (self.resolve)().await?.stat(o).await
    }

    async fn list_directory(
        &self,
        o: SandboxListDirectoryOptions,
    ) -> HarnessResult<Option<Vec<SandboxDirectoryEntry>>> {
        (self.resolve)().await?.list_directory(o).await
    }

    async fn delete_file(&self, o: SandboxDeleteFileOptions) -> HarnessResult<bool> {
        (self.resolve)().await?.delete_file(o).await
    }

    async fn spawn(&self, o: SandboxSpawnOptions) -> HarnessResult<SandboxProcess> {
        (self.resolve)().await?.spawn(o).await
    }

    async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
        (self.resolve)().await?.run(o).await
    }
}

type SharedAttempt<T> = Shared<BoxFuture<'static, Result<T, SharedError>>>;
type MakeAttempt<T> = Arc<dyn Fn() -> BoxFuture<'static, HarnessResult<T>> + Send + Sync>;

/// `promise ??= make().catch(error => { promise = undefined; throw error })`:
/// concurrent callers share one attempt, and a failed attempt is forgotten
/// so the next call retries.
pub(crate) struct RetryOnFailure<T> {
    make: MakeAttempt<T>,
    slot: Mutex<Option<SharedAttempt<T>>>,
}

impl<T: Clone + Send + Sync + 'static> RetryOnFailure<T> {
    pub(crate) fn new(make: MakeAttempt<T>) -> Self {
        Self {
            make,
            slot: Mutex::new(None),
        }
    }

    pub(crate) async fn get(&self) -> HarnessResult<T> {
        let attempt = {
            let mut slot = self.slot.lock();
            slot.get_or_insert_with(|| {
                let future = (self.make)();
                async move { future.await.map_err(SharedError::from) }
                    .boxed()
                    .shared()
            })
            .clone()
        };
        match attempt.clone().await {
            Ok(value) => Ok(value),
            Err(error) => {
                let mut slot = self.slot.lock();
                if slot.as_ref().is_some_and(|s| s.ptr_eq(&attempt)) {
                    *slot = None;
                }
                Err(Box::new(SharedErrorBox(error)))
            }
        }
    }
}
