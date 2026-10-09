//! Port of durable `src/entries.ts`: typed entry kinds and the built-in `pi.*` kinds.
//!
//! Divergences from Pi: the built-in tokens are `const` items in SCREAMING_CASE
//! (`USER_ENTRY` for `UserEntry`, ...). `is()` cannot narrow a Rust type, so
//! typed data is read with [`Entry::data`]. The data types of
//! `pi.tool-result` (`ToolDiagnostic[]`) and `pi.compaction`
//! (`CompactionReason`) are harness types that arrive with the harness; until
//! then their tokens carry raw JSON.

use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;

use serde::de::DeserializeOwned;

use crate::chord::JsonValue;

use super::errors::{Error, Result};
use super::types::EntryRecord;

/// Typed entry kind with a narrowing guard (`Entry<D>`). `D = ()` is TS `never`: the kind carries no data.
pub struct Entry<D = ()> {
    kind: Cow<'static, str>,
    _data: PhantomData<fn() -> D>,
}

impl<D> Clone for Entry<D> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            _data: PhantomData,
        }
    }
}

impl<D> fmt::Debug for Entry<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Entry").field(&self.kind).finish()
    }
}

impl<D> Entry<D> {
    /// A token for a static kind (no validation; used for the built-ins).
    pub const fn from_static(kind: &'static str) -> Self {
        Self {
            kind: Cow::Borrowed(kind),
            _data: PhantomData,
        }
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Whether `entry` exists and has this kind.
    pub fn is(&self, entry: Option<&EntryRecord>) -> bool {
        entry.is_some_and(|entry| entry.kind == self.kind)
    }
}

impl<D: DeserializeOwned> Entry<D> {
    /// Decode the typed `data` of an entry of this kind.
    pub fn data(&self, entry: &EntryRecord) -> Result<D> {
        if entry.kind != self.kind {
            return Err(Error::type_error(format!(
                "Entry {} has kind {}, not {}",
                entry.id, entry.kind, self.kind
            )));
        }
        D::deserialize(entry.data.as_ref().unwrap_or(&JsonValue::Null)).map_err(|error| {
            Error::type_error(format!(
                "Entry {} data does not match {}: {error}",
                entry.id, self.kind
            ))
        })
    }
}

/// Define a typed entry kind whose `is()` guard narrows by `EntryRecord.kind`.
pub fn define_entry<D>(kind: impl Into<String>) -> Result<Entry<D>> {
    let kind = kind.into();
    if kind.is_empty() {
        return Err(Error::type_error("Entry kind must be a non-empty string"));
    }
    Ok(Entry {
        kind: Cow::Owned(kind),
        _data: PhantomData,
    })
}

/// User input: `model` is `[UserMessage]`. Written by submissions.
pub const USER_ENTRY: Entry = Entry::from_static("pi.user");
/// Provider result with any stop reason: `model` is `[AssistantMessage]`. Written by generation.
pub const ASSISTANT_ENTRY: Entry = Entry::from_static("pi.assistant");
/// Positional prompt and tool change: `model` is `[SystemMessage]` with empty `content`.
pub const SYSTEM_ENTRY: Entry = Entry::from_static("pi.system");
/// Tool result: `model` is `[ToolResultMessage]`; `data` holds `{ diagnostics }`.
pub const TOOL_RESULT_ENTRY: Entry<JsonValue> = Entry::from_static("pi.tool-result");
/// Start of a new context: always `head: "self"`.
pub const RESET_ENTRY: Entry = Entry::from_static("pi.reset");
/// Compaction summary: `model` is `[UserMessage]` with the wrapped summary; `data` holds `{ reason }`.
pub const COMPACTION_ENTRY: Entry<JsonValue> = Entry::from_static("pi.compaction");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_kinds_and_narrows_by_kind() {
        assert_eq!(
            define_entry::<()>("").unwrap_err().to_string(),
            "Entry kind must be a non-empty string"
        );
        let note = define_entry::<String>("t.note").unwrap();
        let entry = EntryRecord {
            kind: "t.note".into(),
            data: Some(JsonValue::from("x")),
            ..EntryRecord::default()
        };
        assert!(note.is(Some(&entry)));
        assert!(!USER_ENTRY.is(Some(&entry)));
        assert!(!note.is(None));
        assert_eq!(note.data(&entry).unwrap(), "x");
    }
}
