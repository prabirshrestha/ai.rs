//! Rust port of Pi Durable 1.0.2 (`packages/durable`): durable conversations,
//! entries, tasks, submissions and documents over a pluggable [`Storage`].
//!
//! The module tree mirrors Pi's `src/` layout. The harness, environments,
//! scheduler and SQLite/JSONL backends are later milestones; see
//! `/mnt/project-files/pi-port/progress/durable-m2-m3.md` for the divergences
//! recorded while porting.

pub mod documents;
pub mod entries;
pub mod errors;
pub mod ids;
pub mod storage;
pub mod tasks;
#[cfg(any(test, feature = "durable-testing"))]
pub mod testing;
pub mod types;

pub use documents::{
    AnyDocDefinition, DocAccess, DocFamilyToken, DocToken, RewindableDocAccess, define_doc,
    define_doc_family,
};
pub use entries::{
    ASSISTANT_ENTRY, COMPACTION_ENTRY, Entry, RESET_ENTRY, SYSTEM_ENTRY, TOOL_RESULT_ENTRY,
    USER_ENTRY, define_entry,
};
pub use errors::{ConversationBusy, Error, ReadAfterWrite, Result, StorageRejected};
pub use ids::{
    ConversationId, DocumentId, EntryId, Id, ROOT_CONVERSATION_ID, Seq, SubmissionId, TaskId,
    id_from_number, seq_from_number,
};
pub use storage::memory::{MemoryStorage, PreparedMemoryCommit};
pub use tasks::{Task, TaskDefinition, define_task};
pub use types::*;
