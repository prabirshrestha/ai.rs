//! Port of durable `src/storage/jsonl/node.ts` (`openNodeJsonlStorage`).

use std::sync::Arc;

use crate::chord::Context;
use crate::durable::env::local::LocalExecutionEnv;
use crate::durable::errors::Result;

use super::{JsonlStorage, JsonlStorageOptions};

/// Open or create a JSONL storage directory using the local filesystem
/// (`openNodeJsonlStorage`). Relative paths resolve against the process's
/// current directory.
pub async fn open_local_jsonl_storage(
    directory: &str,
    context: &Context,
    options: JsonlStorageOptions,
) -> Result<JsonlStorage> {
    let cwd = std::env::current_dir()
        .map(|cwd| cwd.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "/".into());
    JsonlStorage::open(
        directory,
        Arc::new(LocalExecutionEnv::at(cwd)),
        context,
        options,
    )
    .await
}
