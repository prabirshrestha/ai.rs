//! Port of chord `src/delta/index.ts`: immutable revision tracking and
//! operations over plain JSON.
//!
//! Ops are Rust enums that serialize to (and parse from) exactly the TS tuple
//! JSON, so stored and transmitted batches stay byte-compatible with Pi:
//! `["r",v]`, `["s",path,v]`, `["d",path]`, `["a",path,text]`, `["t",path,n]`,
//! `["p",path,index,remove,items]`, `["m",path,permutation]`.
//!
//! Divergences from Pi:
//! - String offsets (`t` counts, [`overlap`]) are UTF-16 code units, as in JS.
//!   A Rust `String` cannot hold a lone surrogate, so applying a `t` that would
//!   split a surrogate pair fails with [`PathError`] instead of producing one.
//! - `apply`/`apply_immutable` take the target by value/reference and return
//!   the new root. JS `undefined` as the target is [`JsonValue::Null`]; both
//!   fail the same way for any op other than `r`.
//! - Values are owned trees, so "copy each touched container once" becomes a
//!   copy-on-write clone of the target on its first non-`r` op, and payloads are
//!   cloned rather than adopted.
//! - `assertValidOp(unknown)` is [`assert_valid_op`] over a JSON value (it also
//!   returns the parsed [`Op`]); a typed [`Op`] is checked with [`Op::validate`].
//!   The serde `Deserialize` impls use the same checks and messages.

mod diff;
mod tracker;

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::de::{self, Deserialize, Deserializer};
use serde::ser::{Serialize, SerializeSeq, Serializer};
use serde_json::Value;

pub use super::json::JsonValue;
pub use diff::diff_revisions;
pub use tracker::{Change, Prepared, Tracker, track};

// ─── Paths and operations ────────────────────────────────────────────────────

/// One path segment: an object key or a non-negative integer array index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Seg {
    Key(String),
    Index(usize),
}

impl Seg {
    fn to_json(&self) -> Value {
        match self {
            Seg::Key(key) => Value::String(key.clone()),
            Seg::Index(index) => Value::from(*index),
        }
    }

    /// The object property name for this segment (JS coerces indices to strings).
    fn property(&self) -> Cow<'_, str> {
        match self {
            Seg::Key(key) => Cow::Borrowed(key),
            Seg::Index(index) => Cow::Owned(index.to_string()),
        }
    }
}

impl fmt::Display for Seg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Seg::Key(key) => f.write_str(key),
            Seg::Index(index) => write!(f, "{index}"),
        }
    }
}

impl From<&str> for Seg {
    fn from(value: &str) -> Self {
        Seg::Key(value.to_string())
    }
}

impl From<String> for Seg {
    fn from(value: String) -> Self {
        Seg::Key(value)
    }
}

impl From<usize> for Seg {
    fn from(value: usize) -> Self {
        Seg::Index(value)
    }
}

impl Serialize for Seg {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Seg::Key(key) => serializer.serialize_str(key),
            Seg::Index(index) => serializer.serialize_u64(*index as u64),
        }
    }
}

impl<'de> Deserialize<'de> for Seg {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        parse_seg(&value).map_err(de::Error::custom)
    }
}

/// `Path`: object keys and array indices from the root.
pub type Path = Vec<Seg>;

/// Build a [`Path`] from mixed segments: `path!["values", 1, "label"]`.
#[macro_export]
macro_rules! chord_path {
    ($($seg:expr),* $(,)?) => {
        vec![$($crate::chord::delta::Seg::from($seg)),*]
    };
}
pub use crate::chord_path as path;

/// A decoded operation. `s`/`d`/`a`/`t` paths must be non-empty (`NonEmptyPath`).
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// `["r", value]`: replace the complete value.
    R(JsonValue),
    /// `["s", path, value]`: set an object property or array element.
    S(Path, JsonValue),
    /// `["d", path]`: delete an object property or remove an array element.
    D(Path),
    /// `["a", path, text]`: append to a string.
    A(Path, String),
    /// `["t", path, count]`: remove `count` UTF-16 code units from a string's front.
    T(Path, usize),
    /// `["p", path, index, remove, items]`: splice an array.
    P(Path, usize, usize, Vec<JsonValue>),
    /// `["m", path, permutation]`: reorder an array, `new[i] = old[permutation[i]]`.
    M(Path, Vec<usize>),
}

impl Op {
    /// The op verb (`op[0]`).
    pub fn verb(&self) -> &'static str {
        match self {
            Op::R(_) => "r",
            Op::S(..) => "s",
            Op::D(_) => "d",
            Op::A(..) => "a",
            Op::T(..) => "t",
            Op::P(..) => "p",
            Op::M(..) => "m",
        }
    }

    /// The op path (`op[1]`), absent for `r`.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Op::R(_) => None,
            Op::S(path, _)
            | Op::D(path)
            | Op::A(path, _)
            | Op::T(path, _)
            | Op::P(path, ..)
            | Op::M(path, _) => Some(path),
        }
    }

    /// The typed half of `assertValidOp`: non-empty paths where required, safe
    /// segments, and bijective permutations. Arity and payload shapes are
    /// enforced by the type.
    pub fn validate(&self) -> Result<(), DeltaError> {
        match self {
            Op::R(_) => Ok(()),
            Op::S(path, _) | Op::D(path) | Op::A(path, _) | Op::T(path, _) => {
                if path.is_empty() {
                    return Err(DeltaError::type_error("path is empty"));
                }
                assert_safe_path(path)
            }
            Op::P(path, ..) => assert_safe_path(path),
            Op::M(path, permutation) => {
                assert_safe_path(path)?;
                assert_permutation(permutation)
            }
        }
    }

    /// The op as its TS tuple JSON.
    pub fn to_json(&self) -> JsonValue {
        serde_json::to_value(self).expect("ops always serialize")
    }
}

fn path_json(path: &[Seg]) -> Value {
    Value::Array(path.iter().map(Seg::to_json).collect())
}

impl Serialize for Op {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let len = match self {
            Op::R(_) | Op::D(_) => 2,
            Op::S(..) | Op::A(..) | Op::T(..) | Op::M(..) => 3,
            Op::P(..) => 5,
        };
        let mut seq = serializer.serialize_seq(Some(len))?;
        seq.serialize_element(self.verb())?;
        match self {
            Op::R(value) => seq.serialize_element(value)?,
            Op::S(path, value) => {
                seq.serialize_element(path)?;
                seq.serialize_element(value)?;
            }
            Op::D(path) => seq.serialize_element(path)?,
            Op::A(path, text) => {
                seq.serialize_element(path)?;
                seq.serialize_element(text)?;
            }
            Op::T(path, count) => {
                seq.serialize_element(path)?;
                seq.serialize_element(count)?;
            }
            Op::P(path, index, remove, items) => {
                seq.serialize_element(path)?;
                seq.serialize_element(index)?;
                seq.serialize_element(remove)?;
                seq.serialize_element(items)?;
            }
            Op::M(path, permutation) => {
                seq.serialize_element(path)?;
                seq.serialize_element(permutation)?;
            }
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Op {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        assert_valid_op(&value).map_err(de::Error::custom)
    }
}

/// A path inline, or an id assigned by the encoder on second use (`PathRef`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathRef {
    Path(Path),
    Id(usize),
}

impl Serialize for PathRef {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            PathRef::Path(path) => path.serialize(serializer),
            PathRef::Id(id) => serializer.serialize_u64(*id as u64),
        }
    }
}

/// What crosses a boundary (`WireOp`). A `None` path ref is the shortened tuple
/// that reuses the previous op's path; `Define` is `["#", id, path]`.
#[derive(Debug, Clone, PartialEq)]
pub enum WireOp {
    R(JsonValue),
    S(Option<PathRef>, JsonValue),
    D(Option<PathRef>),
    A(Option<PathRef>, String),
    T(Option<PathRef>, usize),
    P(Option<PathRef>, usize, usize, Vec<JsonValue>),
    M(Option<PathRef>, Vec<usize>),
    Define(usize, Path),
}

impl WireOp {
    pub fn verb(&self) -> &'static str {
        match self {
            WireOp::R(_) => "r",
            WireOp::S(..) => "s",
            WireOp::D(_) => "d",
            WireOp::A(..) => "a",
            WireOp::T(..) => "t",
            WireOp::P(..) => "p",
            WireOp::M(..) => "m",
            WireOp::Define(..) => "#",
        }
    }

    pub fn to_json(&self) -> JsonValue {
        serde_json::to_value(self).expect("wire ops always serialize")
    }
}

impl Serialize for WireOp {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(None)?;
        seq.serialize_element(self.verb())?;
        let reference = |seq: &mut S::SerializeSeq, reference: &Option<PathRef>| match reference {
            Some(reference) => seq.serialize_element(reference),
            None => Ok(()),
        };
        match self {
            WireOp::R(value) => seq.serialize_element(value)?,
            WireOp::S(path, value) => {
                reference(&mut seq, path)?;
                seq.serialize_element(value)?;
            }
            WireOp::D(path) => reference(&mut seq, path)?,
            WireOp::A(path, text) => {
                reference(&mut seq, path)?;
                seq.serialize_element(text)?;
            }
            WireOp::T(path, count) => {
                reference(&mut seq, path)?;
                seq.serialize_element(count)?;
            }
            WireOp::P(path, index, remove, items) => {
                reference(&mut seq, path)?;
                seq.serialize_element(index)?;
                seq.serialize_element(remove)?;
                seq.serialize_element(items)?;
            }
            WireOp::M(path, permutation) => {
                reference(&mut seq, path)?;
                seq.serialize_element(permutation)?;
            }
            WireOp::Define(id, path) => {
                seq.serialize_element(id)?;
                seq.serialize_element(path)?;
            }
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for WireOp {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        assert_valid_wire_op(&value).map_err(de::Error::custom)
    }
}

// ─── Classification ──────────────────────────────────────────────────────────

/// `isReplace(op)`.
pub fn is_replace(op: &Op) -> bool {
    matches!(op, Op::R(_))
}

/// A batch begins with a replacement (`isBase`). Flush guarantees `r` is at
/// index 0 or absent, so this is exact rather than a heuristic.
pub fn is_base(ops: &[Op]) -> bool {
    matches!(ops.first(), Some(Op::R(_)))
}

/// `isBase` for the wire vocabulary.
pub fn is_wire_base(ops: &[WireOp]) -> bool {
    matches!(ops.first(), Some(WireOp::R(_)))
}

// ─── UTF-16 helpers ──────────────────────────────────────────────────────────

/// `string.length`: the number of UTF-16 code units.
pub fn utf16_len(text: &str) -> usize {
    if text.is_ascii() {
        text.len()
    } else {
        text.chars().map(char::len_utf16).sum()
    }
}

/// The byte offset of UTF-16 offset `units`, clamped to the end like
/// `String.prototype.slice`. `None` when the offset splits a surrogate pair.
pub fn utf16_byte_offset(text: &str, units: usize) -> Option<usize> {
    if text.is_ascii() {
        return Some(units.min(text.len()));
    }
    let mut seen = 0;
    for (byte, character) in text.char_indices() {
        if seen == units {
            return Some(byte);
        }
        seen += character.len_utf16();
        if seen > units {
            return None;
        }
    }
    Some(text.len())
}

// ─── Overlap ─────────────────────────────────────────────────────────────────

/// Longest suffix of `a` that is a prefix of `b`, in UTF-16 code units, with
/// Pi's defaults (`probe = 64`, `maxCandidates = 8`).
///
/// Always correct: the returned n satisfies `a.slice(a.length - n) === b.slice(0, n)`.
pub fn overlap(a: &str, b: &str, scan: usize) -> usize {
    overlap_with(a, b, scan, 64, 8)
}

/// [`overlap`] with explicit `probe` and `maxCandidates`.
pub fn overlap_with(a: &str, b: &str, scan: usize, probe: usize, max_candidates: usize) -> usize {
    if a.is_empty() || b.is_empty() || scan == 0 {
        return 0;
    }
    if a.is_ascii() && b.is_ascii() {
        return overlap_units(a.as_bytes(), b.as_bytes(), scan, probe, max_candidates);
    }
    let a: Vec<u16> = a.encode_utf16().collect();
    let b: Vec<u16> = b.encode_utf16().collect();
    overlap_units(&a, &b, scan, probe, max_candidates)
}

fn index_of<T: PartialEq>(haystack: &[T], needle: &[T], from: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(from.min(haystack.len()));
    }
    if needle.len() > haystack.len() {
        return None;
    }
    let first = &needle[0];
    let last_start = haystack.len() - needle.len();
    let mut index = from;
    while index <= last_start {
        if haystack[index] == *first && haystack[index..index + needle.len()] == *needle {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn overlap_units<T: PartialEq>(
    a: &[T],
    b: &[T],
    scan: usize,
    probe: usize,
    max_candidates: usize,
) -> usize {
    let tail = if a.len() > scan {
        &a[a.len() - scan..]
    } else {
        a
    };

    // A probe of length h can only find overlaps of at least h — the head must
    // actually occur in `a`. So try a long head first, then fall back to one
    // character, which finds any overlap at the cost of more candidates.
    for h in [probe.min(b.len()), 1] {
        let head = &b[..h];
        let mut tried = 0;
        let mut found = index_of(tail, head, 0);
        while let Some(k) = found {
            tried += 1;
            if tried > max_candidates {
                break;
            }
            let n = tail.len() - k;
            if n <= b.len() && tail[k..] == b[..n] {
                return n;
            }
            found = index_of(tail, head, k + 1);
        }
        if h == 1 {
            break;
        }
    }
    0
}

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Segments that reach the JS prototype chain (`RESERVED_SEGMENTS`). Rust has
/// no prototype pollution, but the checks are kept so errors match Pi.
pub const RESERVED_SEGMENTS: [&str; 3] = ["__proto__", "constructor", "prototype"];

pub(crate) fn is_reserved(key: &str) -> bool {
    RESERVED_SEGMENTS.contains(&key)
}

/// `UnsafePathError`. `segment` is the offending segment as JSON (it may be a
/// negative or fractional number from an untrusted op).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("unsafe path segment: {}", display_segment(.segment))]
pub struct UnsafePathError {
    pub segment: Value,
}

fn display_segment(segment: &Value) -> String {
    match segment {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

impl UnsafePathError {
    fn new(segment: &Seg) -> Self {
        Self {
            segment: segment.to_json(),
        }
    }
}

/// What a [`PathError`] could not resolve: a path, or an unknown path id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathErrorTarget {
    Path(Path),
    Id(usize),
}

/// `PathError`: `unresolvable path: <JSON>`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unresolvable path: {}", display_target(.path))]
pub struct PathError {
    pub path: PathErrorTarget,
}

fn display_target(target: &PathErrorTarget) -> String {
    match target {
        PathErrorTarget::Path(path) => path_json(path).to_string(),
        PathErrorTarget::Id(id) => id.to_string(),
    }
}

impl PathError {
    pub fn new(path: &[Seg]) -> Self {
        Self {
            path: PathErrorTarget::Path(path.to_vec()),
        }
    }

    pub fn id(id: usize) -> Self {
        Self {
            path: PathErrorTarget::Id(id),
        }
    }
}

/// Errors raised by delta (`TypeError`, `PathError`, `UnsafePathError`, and
/// the tracker's plain `Error`s).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DeltaError {
    /// JS `TypeError` (malformed ops, settled drafts, ...).
    #[error("{0}")]
    Type(String),
    #[error(transparent)]
    Path(#[from] PathError),
    #[error(transparent)]
    UnsafePath(#[from] UnsafePathError),
    /// A plain JS `Error` (tracker lifecycle rejections).
    #[error("{0}")]
    Error(String),
}

impl DeltaError {
    pub(crate) fn type_error(message: impl Into<String>) -> Self {
        Self::Type(message.into())
    }

    pub(crate) fn error(message: impl Into<String>) -> Self {
        Self::Error(message.into())
    }

    pub fn is_type_error(&self) -> bool {
        matches!(self, Self::Type(_))
    }

    pub fn is_path_error(&self) -> bool {
        matches!(self, Self::Path(_))
    }

    pub fn is_unsafe_path_error(&self) -> bool {
        matches!(self, Self::UnsafePath(_))
    }
}

fn path_error(path: &[Seg]) -> DeltaError {
    DeltaError::Path(PathError::new(path))
}

fn unsafe_path(segment: &Seg) -> DeltaError {
    DeltaError::UnsafePath(UnsafePathError::new(segment))
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn non_negative_integer(value: &Value) -> Option<usize> {
    if let Some(number) = value.as_u64() {
        return usize::try_from(number).ok();
    }
    let number = value.as_f64()?;
    if number >= 0.0 && number.fract() == 0.0 && number <= usize::MAX as f64 {
        Some(number as usize)
    } else {
        None
    }
}

fn parse_seg(value: &Value) -> Result<Seg, DeltaError> {
    match value {
        Value::String(key) => {
            if is_reserved(key) {
                Err(DeltaError::UnsafePath(UnsafePathError {
                    segment: value.clone(),
                }))
            } else {
                Ok(Seg::Key(key.clone()))
            }
        }
        Value::Number(_) => non_negative_integer(value).map(Seg::Index).ok_or_else(|| {
            DeltaError::UnsafePath(UnsafePathError {
                segment: value.clone(),
            })
        }),
        // JS paths only hold strings and numbers; anything else fails `Number.isInteger`.
        other => Err(DeltaError::UnsafePath(UnsafePathError {
            segment: other.clone(),
        })),
    }
}

/// `assertSafePath` over a JSON array (the untrusted form).
fn parse_safe_path(value: &Value) -> Result<Path, DeltaError> {
    let Value::Array(segments) = value else {
        return Err(DeltaError::type_error("path is not an array"));
    };
    segments.iter().map(parse_seg).collect()
}

fn parse_path_arg(value: &Value, non_empty: bool) -> Result<Path, DeltaError> {
    let Value::Array(segments) = value else {
        return Err(DeltaError::type_error("path is not an array"));
    };
    if non_empty && segments.is_empty() {
        return Err(DeltaError::type_error("path is empty"));
    }
    parse_safe_path(value)
}

fn parse_permutation(value: &Value) -> Result<Vec<usize>, DeltaError> {
    let Value::Array(items) = value else {
        return Err(DeltaError::type_error("m permutation is not an array"));
    };
    let permutation = items
        .iter()
        .map(|item| non_negative_integer(item).filter(|index| *index < items.len()))
        .collect::<Option<Vec<usize>>>()
        .ok_or_else(|| DeltaError::type_error("m permutation is not a bijection"))?;
    assert_permutation(&permutation)?;
    Ok(permutation)
}

fn assert_permutation(permutation: &[usize]) -> Result<(), DeltaError> {
    let mut seen = vec![false; permutation.len()];
    for &index in permutation {
        if index >= permutation.len() || seen[index] {
            return Err(DeltaError::type_error("m permutation is not a bijection"));
        }
        seen[index] = true;
    }
    Ok(())
}

/// `assertSafePath(path)`.
pub fn assert_safe_path(path: &[Seg]) -> Result<(), DeltaError> {
    for seg in path {
        if let Seg::Key(key) = seg
            && is_reserved(key)
        {
            return Err(unsafe_path(seg));
        }
    }
    Ok(())
}

/// Verb, arity and payload shape for a **decoded** op (`assertValidOp`):
/// paths inline, no `#`, no short forms. Returns the parsed [`Op`].
pub fn assert_valid_op(op: &Value) -> Result<Op, DeltaError> {
    let items = match op {
        Value::Array(items) if !items.is_empty() => items,
        _ => return Err(DeltaError::type_error("op is not a tuple")),
    };
    let verb = items[0].as_str();
    let count = |value: &Value, message: &str| {
        non_negative_integer(value).ok_or_else(|| DeltaError::type_error(message))
    };
    match verb {
        Some("r") => {
            if items.len() != 2 {
                return Err(DeltaError::type_error("r arity"));
            }
            Ok(Op::R(items[1].clone()))
        }
        Some("s") => {
            if items.len() != 3 {
                return Err(DeltaError::type_error("s arity"));
            }
            Ok(Op::S(parse_path_arg(&items[1], true)?, items[2].clone()))
        }
        Some("d") => {
            if items.len() != 2 {
                return Err(DeltaError::type_error("d arity"));
            }
            Ok(Op::D(parse_path_arg(&items[1], true)?))
        }
        Some("a") => {
            let (3, Some(text)) = (items.len(), items.get(2).and_then(Value::as_str)) else {
                return Err(DeltaError::type_error("a shape"));
            };
            Ok(Op::A(parse_path_arg(&items[1], true)?, text.to_string()))
        }
        Some("t") => {
            if items.len() != 3 {
                return Err(DeltaError::type_error("t shape"));
            }
            let n = count(&items[2], "t shape")?;
            Ok(Op::T(parse_path_arg(&items[1], true)?, n))
        }
        Some("p") => {
            if items.len() != 5 {
                return Err(DeltaError::type_error("p arity"));
            }
            let path = parse_path_arg(&items[1], false)?;
            let index = count(&items[2], "p index")?;
            let remove = count(&items[3], "p remove")?;
            let Value::Array(values) = &items[4] else {
                return Err(DeltaError::type_error("p items"));
            };
            Ok(Op::P(path, index, remove, values.clone()))
        }
        Some("m") => {
            if items.len() != 3 {
                return Err(DeltaError::type_error("m arity"));
            }
            let path = parse_path_arg(&items[1], false)?;
            Ok(Op::M(path, parse_permutation(&items[2])?))
        }
        // Silently skipping an unknown verb is how a newer producer's op vanishes.
        _ => Err(DeltaError::type_error(format!(
            "unknown op verb: {}",
            verb_text(&items[0])
        ))),
    }
}

fn verb_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn parse_ref(value: &Value) -> Result<PathRef, DeltaError> {
    if value.is_number() {
        return non_negative_integer(value)
            .map(PathRef::Id)
            .ok_or_else(|| DeltaError::type_error("bad path id"));
    }
    // A string is not a path; unchecked it would resolve to the root.
    if !value.is_array() {
        return Err(DeltaError::type_error("path is not an array"));
    }
    Ok(PathRef::Path(parse_safe_path(value)?))
}

/// The same, for the wire grammar (`assertValidWireOp`): ids and short forms
/// are legal here. Returns the parsed [`WireOp`].
pub fn assert_valid_wire_op(op: &Value) -> Result<WireOp, DeltaError> {
    let items = match op {
        Value::Array(items) if !items.is_empty() => items,
        _ => return Err(DeltaError::type_error("op is not a tuple")),
    };
    let verb = items[0].as_str();
    let n = items.len();
    match verb {
        Some("r") => {
            if n != 2 {
                return Err(DeltaError::type_error("r arity"));
            }
            Ok(WireOp::R(items[1].clone()))
        }
        Some("s") => match n {
            3 => Ok(WireOp::S(Some(parse_ref(&items[1])?), items[2].clone())),
            2 => Ok(WireOp::S(None, items[1].clone())),
            _ => Err(DeltaError::type_error("s arity")),
        },
        Some("d") => match n {
            2 => Ok(WireOp::D(Some(parse_ref(&items[1])?))),
            1 => Ok(WireOp::D(None)),
            _ => Err(DeltaError::type_error("d arity")),
        },
        Some("a") => match n {
            3 => {
                let reference = parse_ref(&items[1])?;
                let text = items[2]
                    .as_str()
                    .ok_or_else(|| DeltaError::type_error("a value"))?;
                Ok(WireOp::A(Some(reference), text.to_string()))
            }
            2 => {
                let text = items[1]
                    .as_str()
                    .ok_or_else(|| DeltaError::type_error("a value"))?;
                Ok(WireOp::A(None, text.to_string()))
            }
            _ => Err(DeltaError::type_error("a arity")),
        },
        Some("t") => match n {
            3 => {
                let reference = parse_ref(&items[1])?;
                let count = non_negative_integer(&items[2])
                    .ok_or_else(|| DeltaError::type_error("t count"))?;
                Ok(WireOp::T(Some(reference), count))
            }
            2 => {
                let count = non_negative_integer(&items[1])
                    .ok_or_else(|| DeltaError::type_error("t count"))?;
                Ok(WireOp::T(None, count))
            }
            _ => Err(DeltaError::type_error("t arity")),
        },
        Some("p") => {
            let (reference, rest) = match n {
                5 => (Some(&items[1]), &items[2..]),
                4 => (None, &items[1..]),
                _ => return Err(DeltaError::type_error("p arity")),
            };
            let reference = reference.map(parse_ref).transpose()?;
            let index =
                non_negative_integer(&rest[0]).ok_or_else(|| DeltaError::type_error("p index"))?;
            let remove =
                non_negative_integer(&rest[1]).ok_or_else(|| DeltaError::type_error("p remove"))?;
            let Value::Array(values) = &rest[2] else {
                return Err(DeltaError::type_error("p items"));
            };
            Ok(WireOp::P(reference, index, remove, values.clone()))
        }
        Some("m") => {
            let reference = match n {
                3 => Some(parse_ref(&items[1])?),
                2 => None,
                _ => return Err(DeltaError::type_error("m arity")),
            };
            Ok(WireOp::M(reference, parse_permutation(&items[n - 1])?))
        }
        Some("#") => {
            let id = items.get(1).and_then(non_negative_integer);
            let (3, Some(id), Some(path @ Value::Array(_))) = (n, id, items.get(2)) else {
                return Err(DeltaError::type_error("# shape"));
            };
            Ok(WireOp::Define(id, parse_safe_path(path)?))
        }
        _ => Err(DeltaError::type_error(format!(
            "unknown op verb: {}",
            verb_text(&items[0])
        ))),
    }
}

fn validate_wire_op(op: &WireOp) -> Result<(), DeltaError> {
    let reference = match op {
        WireOp::R(_) => return Ok(()),
        WireOp::Define(_, path) => return assert_safe_path(path),
        WireOp::S(reference, _)
        | WireOp::D(reference)
        | WireOp::A(reference, _)
        | WireOp::T(reference, _)
        | WireOp::P(reference, ..) => reference,
        WireOp::M(reference, permutation) => {
            assert_permutation(permutation)?;
            reference
        }
    };
    match reference {
        Some(PathRef::Path(path)) => assert_safe_path(path),
        _ => Ok(()),
    }
}

// ─── Applier ─────────────────────────────────────────────────────────────────

/// Apply ops to a plain mutable value and return it, because `r` replaces it
/// outright (`apply`). Takes decoded ops; run a [`Decoder`] first if the ops
/// came from a boundary.
pub fn apply(target: JsonValue, ops: &[Op]) -> Result<JsonValue, DeltaError> {
    let mut root = target;
    for op in ops {
        apply_op(&mut root, op)?;
    }
    Ok(root)
}

fn apply_op(root: &mut Value, op: &Op) -> Result<(), DeltaError> {
    op.validate()?;
    match op {
        Op::R(value) => {
            *root = value.clone();
            Ok(())
        }
        Op::P(path, index, remove, items) => {
            let target = resolve(root, path)?;
            let Value::Array(target) = target else {
                return Err(path_error(path));
            };
            let start = (*index).min(target.len());
            let end = start + (*remove).min(target.len() - start);
            target.splice(start..end, items.iter().cloned());
            Ok(())
        }
        Op::M(path, permutation) => {
            let target = resolve(root, path)?;
            match target {
                Value::Array(target) if target.len() == permutation.len() => {
                    let previous = std::mem::take(target);
                    let mut slots: Vec<Option<Value>> = previous.into_iter().map(Some).collect();
                    *target = permutation
                        .iter()
                        .map(|&index| slots[index].take().expect("bijective permutation"))
                        .collect();
                    Ok(())
                }
                _ => Err(path_error(path)),
            }
        }
        Op::S(path, _) | Op::D(path) | Op::A(path, _) | Op::T(path, _) => {
            // s/d/a/t can never target the root — validation forbids it.
            let (key, parent_path) = path.split_last().expect("validated non-empty path");
            let parent = resolve(root, parent_path)?;
            if let Value::Array(array) = parent {
                let Seg::Index(index) = key else {
                    return Err(unsafe_path(key));
                };
                // An index may address an existing element or append exactly one past the end.
                if *index > array.len() {
                    return Err(unsafe_path(key));
                }
                let index = *index;
                match op {
                    Op::S(_, value) => {
                        if index == array.len() {
                            array.push(value.clone());
                        } else {
                            array[index] = value.clone();
                        }
                    }
                    Op::D(_) => {
                        if index >= array.len() {
                            return Err(path_error(path));
                        }
                        array.remove(index);
                    }
                    Op::A(_, text) => match array.get_mut(index) {
                        Some(Value::String(current)) => current.push_str(text),
                        _ => return Err(path_error(path)),
                    },
                    Op::T(_, count) => match array.get_mut(index) {
                        Some(Value::String(current)) => truncate_front(current, *count, path)?,
                        _ => return Err(path_error(path)),
                    },
                    _ => unreachable!(),
                }
                return Ok(());
            }
            let Value::Object(object) = parent else {
                unreachable!("resolve returns containers")
            };
            let property = key.property();
            match op {
                Op::S(_, value) => {
                    if let Some(slot) = object.get_mut(property.as_ref()) {
                        *slot = value.clone();
                    } else {
                        object.insert(property.into_owned(), value.clone());
                    }
                }
                Op::D(_) => {
                    object.shift_remove(property.as_ref());
                }
                Op::A(_, text) => match object.get_mut(property.as_ref()) {
                    Some(Value::String(current)) => current.push_str(text),
                    _ => return Err(path_error(path)),
                },
                Op::T(_, count) => match object.get_mut(property.as_ref()) {
                    Some(Value::String(current)) => truncate_front(current, *count, path)?,
                    _ => return Err(path_error(path)),
                },
                _ => unreachable!(),
            }
            Ok(())
        }
    }
}

fn truncate_front(current: &mut String, count: usize, path: &[Seg]) -> Result<(), DeltaError> {
    let Some(offset) = utf16_byte_offset(current, count) else {
        return Err(path_error(path));
    };
    current.drain(..offset);
    Ok(())
}

/// Walk own properties only, then require a container (`resolve`).
fn resolve<'a>(root: &'a mut Value, path: &[Seg]) -> Result<&'a mut Value, DeltaError> {
    let mut node = root;
    for seg in path {
        node = match node {
            Value::Array(array) => {
                let Seg::Index(index) = seg else {
                    return Err(unsafe_path(seg));
                };
                array.get_mut(*index).ok_or_else(|| path_error(path))?
            }
            Value::Object(object) => object
                .get_mut(seg.property().as_ref())
                .ok_or_else(|| path_error(path))?,
            _ => return Err(path_error(path)),
        };
    }
    if node.is_array() || node.is_object() {
        Ok(node)
    } else {
        Err(path_error(path))
    }
}

/// Apply one decoded operation batch without mutating the previous immutable value.
pub fn apply_immutable(target: &JsonValue, ops: &[Op]) -> Result<JsonValue, DeltaError> {
    apply_immutable_batches(target, [ops])
}

/// Apply decoded operation batches as one final-result-only replay.
///
/// The target is copied once, on the first op that is not a root replacement,
/// so a failed batch never exposes or mutates anything.
pub fn apply_immutable_batches<'a, I>(
    target: &JsonValue,
    batches: I,
) -> Result<JsonValue, DeltaError>
where
    I: IntoIterator<Item = &'a [Op]>,
{
    let mut root: Cow<'_, Value> = Cow::Borrowed(target);
    for ops in batches {
        for op in ops {
            op.validate()?;
            if let Op::R(value) = op {
                root = Cow::Owned(value.clone());
                continue;
            }
            apply_op(root.to_mut(), op)?;
        }
    }
    Ok(root.into_owned())
}

// ─── Codec ───────────────────────────────────────────────────────────────────
//
// Path interning and arity omission live between the tracker and a boundary;
// `Op` and `apply` know nothing about them. ONE PAIR PER INDEPENDENT STATE STREAM.

fn path_key(path: &[Seg]) -> String {
    path_json(path).to_string()
}

/// `encoder()`: interns on SECOND use.
#[derive(Debug, Default)]
pub struct Encoder {
    seen: HashSet<String>,
    ids: HashMap<String, usize>,
    next_id: usize,
}

/// `encoder()`.
pub fn encoder() -> Encoder {
    Encoder::default()
}

impl Encoder {
    pub fn encode(&mut self, ops: &[Op]) -> Vec<WireOp> {
        // Arity omission is scoped to a batch; ids are the only cross-batch state.
        let mut previous: Option<String> = None;
        let mut out = Vec::with_capacity(ops.len());
        for op in ops {
            let path = match op {
                Op::R(value) => {
                    out.push(WireOp::R(value.clone()));
                    // A base batch is a RECOVERY POINT: everything after it must be self-contained.
                    self.seen.clear();
                    self.ids.clear();
                    self.next_id = 0;
                    previous = None;
                    continue;
                }
                other => other.path().expect("non-replace ops have paths"),
            };
            let key = path_key(path);

            let reference = if previous.as_deref() == Some(key.as_str()) {
                // Same path as the previous op: drop the ref entirely.
                None
            } else {
                let reference = if let Some(existing) = self.ids.get(&key) {
                    PathRef::Id(*existing)
                } else if self.seen.contains(&key) {
                    let id = self.next_id;
                    self.next_id += 1;
                    self.ids.insert(key.clone(), id);
                    out.push(WireOp::Define(id, path.clone())); // second use: define, then reference
                    PathRef::Id(id)
                } else {
                    self.seen.insert(key.clone()); // first use: inline
                    PathRef::Path(path.clone())
                };
                Some(reference)
            };
            let had_reference = reference.is_some();
            out.push(match op {
                Op::S(_, value) => WireOp::S(reference, value.clone()),
                Op::D(_) => WireOp::D(reference),
                Op::A(_, text) => WireOp::A(reference, text.clone()),
                Op::T(_, count) => WireOp::T(reference, *count),
                Op::P(_, index, remove, items) => {
                    WireOp::P(reference, *index, *remove, items.clone())
                }
                Op::M(_, permutation) => WireOp::M(reference, permutation.clone()),
                Op::R(_) => unreachable!(),
            });
            if had_reference {
                previous = Some(key);
            }
        }
        out
    }
}

/// `decoder()`.
#[derive(Debug, Default)]
pub struct Decoder {
    paths: HashMap<usize, Path>,
}

/// `decoder()`.
pub fn decoder() -> Decoder {
    Decoder::default()
}

impl Decoder {
    pub fn decode(&mut self, wire: &[WireOp]) -> Result<Vec<Op>, DeltaError> {
        let mut previous: Option<Path> = None; // scoped to the batch, as in encode
        let mut out = Vec::with_capacity(wire.len());
        for op in wire {
            validate_wire_op(op)?;
            let reference = match op {
                WireOp::Define(id, path) => {
                    assert_safe_path(path)?;
                    self.paths.insert(*id, path.clone());
                    continue;
                }
                WireOp::R(value) => {
                    out.push(Op::R(value.clone()));
                    self.paths.clear();
                    previous = None;
                    continue;
                }
                WireOp::S(reference, _)
                | WireOp::D(reference)
                | WireOp::A(reference, _)
                | WireOp::T(reference, _)
                | WireOp::P(reference, ..)
                | WireOp::M(reference, _) => reference,
            };
            let path = match reference {
                None => previous.clone().ok_or_else(|| path_error(&[]))?,
                Some(PathRef::Id(id)) => {
                    let resolved = self
                        .paths
                        .get(id)
                        .ok_or(DeltaError::Path(PathError::id(*id)))?;
                    previous = Some(resolved.clone());
                    resolved.clone()
                }
                Some(PathRef::Path(path)) => {
                    previous = Some(path.clone());
                    path.clone()
                }
            };
            let needs_non_empty = !matches!(op, WireOp::P(..) | WireOp::M(..));
            if needs_non_empty && path.is_empty() {
                return Err(path_error(&path));
            }
            out.push(match op {
                WireOp::S(_, value) => Op::S(path, value.clone()),
                WireOp::D(_) => Op::D(path),
                WireOp::A(_, text) => Op::A(path, text.clone()),
                WireOp::T(_, count) => Op::T(path, *count),
                WireOp::P(_, index, remove, items) => Op::P(path, *index, *remove, items.clone()),
                WireOp::M(_, permutation) => Op::M(path, permutation.clone()),
                WireOp::R(_) | WireOp::Define(..) => unreachable!(),
            });
        }
        Ok(out)
    }

    /// Decode untrusted wire JSON (`decode(wire)` with `assertValidWireOp` on each tuple).
    pub fn decode_json(&mut self, wire: &[Value]) -> Result<Vec<Op>, DeltaError> {
        let ops = wire
            .iter()
            .map(assert_valid_wire_op)
            .collect::<Result<Vec<_>, _>>()?;
        self.decode(&ops)
    }
}

#[cfg(test)]
mod tests;
