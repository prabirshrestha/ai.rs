//! Port of durable `src/ids.ts` plus the branded ID types of `src/types.ts`.
//!
//! TS IDs are branded `number`s. Here every record kind is a `#[serde(transparent)]`
//! newtype over `u64`, so stored JSON is the bare number as in Pi. `TaskId<R>`
//! keeps the task result type as a phantom parameter (TS `TaskId<Result>`), and
//! [`TaskId::erase`] is the widening TS performs implicitly.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::chord::JsonValue;

/// `Number.MAX_SAFE_INTEGER`: the largest ID or sequence a JS backend can hold.
pub const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// Erased nominal number identifying one durable record kind (`Id<Kind, Type>`).
pub trait Id: Copy + fmt::Debug + Send + Sync + 'static {
    /// The record kind (`"conversation"`, `"entry"`, ...).
    const KIND: &'static str;
    /// Apply the brand at a trusted numeric boundary.
    fn from_number(value: u64) -> Self;
    /// The underlying number.
    fn number(self) -> u64;
}

/// Apply an erased ID brand at a trusted numeric allocation or decoding boundary.
pub fn id_from_number<I: Id>(value: u64) -> I {
    I::from_number(value)
}

/// Apply the erased commit-sequence brand at a trusted storage boundary.
pub fn seq_from_number(value: u64) -> Seq {
    Seq(value)
}

macro_rules! durable_id {
    ($(#[$meta:meta])* $name:ident, $kind:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub u64);

        impl Id for $name {
            const KIND: &'static str = $kind;
            fn from_number(value: u64) -> Self {
                Self(value)
            }
            fn number(self) -> u64 {
                self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<$name> for u64 {
            fn from(value: $name) -> u64 {
                value.0
            }
        }
    };
}

durable_id!(
    /// `ConversationId`.
    ConversationId,
    "conversation"
);
durable_id!(
    /// `EntryId`.
    EntryId,
    "entry"
);
durable_id!(
    /// `SubmissionId`.
    SubmissionId,
    "submission"
);
durable_id!(
    /// `DocumentId`.
    DocumentId,
    "document"
);
durable_id!(
    /// Strictly increasing sequence assigned to one atomic storage commit; gaps are permitted.
    Seq,
    "sequence"
);

/// The root conversation always uses this reserved ID.
pub const ROOT_CONVERSATION_ID: ConversationId = ConversationId(1);

/// `TaskId<Result>`: a task ID carrying its result type. The default is the
/// erased form (`TaskId<unknown>` in TS).
pub struct TaskId<R = JsonValue>(pub u64, PhantomData<fn() -> R>);

impl<R> TaskId<R> {
    pub const fn new(value: u64) -> Self {
        Self(value, PhantomData)
    }

    /// Widen to the erased task ID.
    pub const fn erase(self) -> TaskId {
        TaskId(self.0, PhantomData)
    }

    /// Narrow an erased ID to a result type (a TS cast).
    pub const fn cast<T>(self) -> TaskId<T> {
        TaskId(self.0, PhantomData)
    }
}

impl<R: 'static> Id for TaskId<R> {
    const KIND: &'static str = "task";
    fn from_number(value: u64) -> Self {
        Self::new(value)
    }
    fn number(self) -> u64 {
        self.0
    }
}

impl<R> Clone for TaskId<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> Copy for TaskId<R> {}

impl<R> Default for TaskId<R> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<R, S> PartialEq<TaskId<S>> for TaskId<R> {
    fn eq(&self, other: &TaskId<S>) -> bool {
        self.0 == other.0
    }
}

impl<R> Eq for TaskId<R> {}

impl<R> PartialOrd for TaskId<R> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<R> Ord for TaskId<R> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<R> Hash for TaskId<R> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<R> fmt::Debug for TaskId<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TaskId({})", self.0)
    }
}

impl<R> fmt::Display for TaskId<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<R> Serialize for TaskId<R> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de, R> Deserialize<'de> for TaskId<R> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u64::deserialize(deserializer).map(Self::new)
    }
}

impl<R> From<TaskId<R>> for u64 {
    fn from(value: TaskId<R>) -> u64 {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brands_numeric_ids_by_record_kind_and_serializes_bare_numbers() {
        let task: TaskId<f64> = id_from_number(4);
        assert_eq!(serde_json::to_string(&task).unwrap(), "4");
        assert_eq!(task.erase(), TaskId::<JsonValue>::new(4));
        let conversation: ConversationId = serde_json::from_str("1").unwrap();
        assert_eq!(conversation, ROOT_CONVERSATION_ID);
        assert_eq!(seq_from_number(1), Seq(1));
        assert_eq!(<EntryId as Id>::KIND, "entry");
    }
}
