//! Port of durable `src/session/`: the Session, its transactions, committed
//! observation, and fork document selection.

pub mod forks;
pub mod observation;
#[allow(clippy::module_inception)]
pub mod session;
pub mod transaction;

#[cfg(test)]
pub(crate) mod tests;

use crate::chord::AttachedReplicatedState;

use observation::{CommittedWatch, ObservedDocumentValue};

/// Replicated committed state of one document incarnation; `None` once retired.
pub type DocumentState = AttachedReplicatedState<ObservedDocumentValue>;

/// Serialized exact-frame watch of one document incarnation.
pub type DocumentWatch = CommittedWatch<ObservedDocumentValue>;

pub use observation::{CommittedStateSource, RETIREMENT_OPERATIONS, WatchListener};
pub use session::{
    CloseListener, CommitListener, LineDocument, Session, SessionHooks, SessionImpl, Unsubscribe,
    WeakSession, create_session,
};
pub use transaction::{DocDraft, Transaction, TransactionScope, Tx, TxFuture};
