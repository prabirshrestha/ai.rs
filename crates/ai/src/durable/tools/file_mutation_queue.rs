//! Port of durable `src/tools/file-mutation-queue.ts`.
//!
//! Divergence from Pi: the queue is a process-wide map of shared futures; a call's slot is released when its future
//! settles or is dropped.

use std::collections::HashMap;
use std::sync::LazyLock;

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::chord::Context;
use crate::durable::env::{ExecutionEnv, FileErrorCode, get_or_throw};
use crate::durable::errors::{Error, Result};

type Tail = Shared<BoxFuture<'static, ()>>;

/// Tail of the mutation chain of each file, keyed by file system id and canonical path.
static QUEUES: LazyLock<Mutex<HashMap<String, Tail>>> = LazyLock::new(Default::default);

async fn mutation_key(env: &dyn ExecutionEnv, path: &str, context: &Context) -> Result<String> {
    let absolute_path = get_or_throw(env.absolute_path(path, context).await)?;
    Ok(format!(
        "{}\0{}",
        env.id(),
        canonical(env, &absolute_path, context).await?
    ))
}

/// The canonical path; for a file that does not exist yet, its canonical parent joined with its name, so a `write`
/// that creates a file and a later mutation of it share one key even under a symlinked directory.
fn canonical<'a>(
    env: &'a dyn ExecutionEnv,
    absolute_path: &'a str,
    context: &'a Context,
) -> BoxFuture<'a, Result<String>> {
    Box::pin(async move {
        let error = match env.canonical_path(absolute_path, context).await {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        if error.code == FileErrorCode::NotSupported {
            return Ok(absolute_path.to_string());
        }
        if error.code != FileErrorCode::NotFound {
            return Err(Error::thrown(error));
        }
        // The file system splits the path, so a name may contain characters that are separators elsewhere.
        let parent = get_or_throw(env.join_path(&[absolute_path, ".."], context).await)?;
        if parent == absolute_path || !absolute_path.starts_with(parent.as_str()) {
            return Ok(absolute_path.to_string());
        }
        let skip = if parent.ends_with('/') || parent.ends_with('\\') {
            0
        } else {
            1
        };
        let name = absolute_path.get(parent.len() + skip..).unwrap_or("");
        let canonical_parent = canonical(env, &parent, context).await?;
        get_or_throw(
            env.join_path(&[canonical_parent.as_str(), name], context)
                .await,
        )
    })
}

/// Releases a call's slot when it settles or is dropped (TS `finally`).
struct Release {
    key: String,
    tail: Tail,
    done: Option<oneshot::Sender<()>>,
}

impl Drop for Release {
    fn drop(&mut self) {
        if let Some(done) = self.done.take() {
            let _ = done.send(());
        }
        let mut queues = QUEUES.lock();
        if queues
            .get(&self.key)
            .is_some_and(|tail| tail.ptr_eq(&self.tail))
        {
            queues.remove(&self.key);
        }
    }
}

/// Serialize `edit` and `write` mutations of one file within this process: same file system id and canonical path,
/// whichever environment object the call got. Other files, and other file systems, never wait. Concurrent calls on
/// one file run in the order their keys resolve. Not a lock against `bash` or other processes.
pub async fn with_file_mutation_queue<T, F, Fut>(
    env: &dyn ExecutionEnv,
    path: &str,
    f: F,
    context: &Context,
) -> Result<T>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let key = mutation_key(env, path, context).await?;
    // Take the slot without awaiting, so no other call can take it in between.
    let (done, done_receiver) = oneshot::channel::<()>();
    let (previous, release) = {
        let mut queues = QUEUES.lock();
        let previous = queues.get(&key).cloned();
        let chained = previous.clone();
        let tail: Tail = async move {
            if let Some(previous) = chained {
                previous.await;
            }
            let _ = done_receiver.await;
        }
        .boxed()
        .shared();
        queues.insert(key.clone(), tail.clone());
        (
            previous,
            Release {
                key,
                tail,
                done: Some(done),
            },
        )
    };
    if let Some(previous) = previous {
        previous.await;
    }
    let result = f().await;
    drop(release);
    result
}
