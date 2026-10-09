//! Port of durable `src/storage/jsonl/`: the portable [`JsonlStorage`] over
//! any [`FileSystem`](crate::durable::env::FileSystem), and (behind the
//! `durable-local-env` feature) [`local::open_local_jsonl_storage`], Pi's
//! `openNodeJsonlStorage`.

#[cfg(feature = "durable-local-env")]
pub mod local;
mod storage;
#[cfg(all(test, feature = "durable-local-env"))]
mod tests;

#[cfg(feature = "durable-local-env")]
pub use local::open_local_jsonl_storage;
pub use storage::{
    JsonlCorruptionError, JsonlStorage, JsonlStorageOptions, JsonlStoragePoisonedError,
};
