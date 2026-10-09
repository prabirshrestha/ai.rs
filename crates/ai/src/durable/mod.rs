//! Rust port of Pi Durable 1.0.2 (`packages/durable`): durable conversations,
//! entries, tasks, submissions and documents over a pluggable [`Storage`].
//!
//! The module tree mirrors Pi's `src/` layout. Execution environments live in
//! [`env`](mod@env), the SQLite and JSONL backends in [`storage`], the Harness
//! and scheduler in [`harness`], and the `read`, `write`, `edit`, and `bash`
//! coding tools in [`tools`]. Divergences from Pi are documented on the modules
//! and items involved.

pub mod documents;
pub mod entries;
pub mod env;
pub mod errors;
pub mod harness;
pub mod ids;
pub mod session;
pub mod storage;
pub mod tasks;
#[cfg(any(test, feature = "durable-testing"))]
pub mod testing;
pub mod tools;
pub mod truncate;
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
pub use session::{
    DocDraft, DocumentState, DocumentWatch, Session, SessionHooks, SessionImpl, Transaction,
    TransactionScope, Tx, create_session,
};
pub use storage::memory::{MemoryStorage, PreparedMemoryCommit};
pub use tasks::{Task, TaskDefinition, define_task};
pub use types::*;
