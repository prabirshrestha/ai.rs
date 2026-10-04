//! Port of chord `src/delta/diff.ts` (`diffRevisions`).
//!
//! Divergence from Pi: JS compares containers by reference (`===`) when it
//! anchors array elements (identity subsequences, identity anchors,
//! permutations, and "shares a nested container" semantic alignment). A
//! `serde_json::Value` tree has no reference identity, so this port uses deep
//! equality wherever Pi uses identity. For revisions produced by copy-on-write
//! (where unchanged containers are shared and changed ones are fresh) the two
//! agree; they can differ when distinct containers happen to be deeply equal
//! (for example several empty objects). Results always converge: applying the
//! ops to `before` yields `after`.

use std::collections::HashMap;

use serde_json::Value;

use super::{Op, Path, Seg, is_reserved, overlap, utf16_byte_offset, utf16_len};

const DEFAULT_OVERLAP_SCAN: usize = 65_536;
const MAX_DELTA_OPERATIONS: usize = 4_096;
const MAX_IDENTITY_CANDIDATES: usize = 200_000;
const MAX_SEMANTIC_CELLS: usize = 65_536;

/// An operation batch plus the `overflowedBatches` flag.
#[derive(Default)]
struct Operations {
    ops: Vec<Op>,
    overflowed: bool,
}

impl Operations {
    fn emit(&mut self, operation: Op) {
        if self.overflowed {
            return;
        }
        if self.ops.len() >= MAX_DELTA_OPERATIONS {
            self.overflowed = true;
            return;
        }
        self.ops.push(operation);
    }

    fn emit_set(&mut self, path: &Path, value: &Value) {
        if path.is_empty() {
            self.emit(Op::R(value.clone()));
        } else {
            self.emit(Op::S(path.clone(), value.clone()));
        }
    }
}

fn is_container(value: &Value) -> bool {
    matches!(value, Value::Array(_) | Value::Object(_))
}

/// Deep JSON equality that ignores key order and compares numbers by value.
pub(crate) fn equal_json(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            left == right || (left.as_f64().is_some() && left.as_f64() == right.as_f64())
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| equal_json(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    right.get(key).is_some_and(|other| equal_json(value, other))
                })
        }
        _ => left == right,
    }
}

/// `sameValue`: identity or deep equality. Identity is deep equality here.
fn same_value(left: &Value, right: &Value) -> bool {
    equal_json(left, right)
}

/// A hashable stand-in for JS reference identity (see the module docs).
fn identity_key(value: &Value) -> String {
    match value {
        Value::Number(number) => match number.as_f64() {
            Some(float) if float.fract() == 0.0 && float.abs() < 9.007_199_254_740_992e15 => {
                format!("{}", float as i64)
            }
            _ => number.to_string(),
        },
        other => other.to_string(),
    }
}

fn permutation(before: &[Value], after: &[Value]) -> Option<Vec<usize>> {
    if before.len() != after.len() {
        return None;
    }
    let mut positions: HashMap<String, (Vec<usize>, usize)> = HashMap::new();
    for (index, value) in before.iter().enumerate() {
        positions
            .entry(identity_key(value))
            .or_default()
            .0
            .push(index);
    }
    let mut result = Vec::with_capacity(after.len());
    for value in after {
        let entry = positions.get_mut(&identity_key(value))?;
        if entry.1 == entry.0.len() {
            return None;
        }
        result.push(entry.0[entry.1]);
        entry.1 += 1;
    }
    Some(result)
}

fn emit_string(before: &str, after: &str, path: &Path, operations: &mut Operations) {
    if before == after {
        return;
    }
    if after.len() > before.len() && after.starts_with(before) {
        operations.emit(Op::A(path.clone(), after[before.len()..].to_string()));
        return;
    }
    let shared = overlap(before, after, DEFAULT_OVERLAP_SCAN);
    if shared == 0 {
        operations.emit(Op::S(path.clone(), Value::String(after.to_string())));
        return;
    }
    operations.emit(Op::T(path.clone(), utf16_len(before) - shared));
    if utf16_len(after) > shared {
        let offset =
            utf16_byte_offset(after, shared).expect("overlaps end on character boundaries");
        operations.emit(Op::A(path.clone(), after[offset..].to_string()));
    }
}

type ArrayMatch = (usize, usize);

struct MatchCandidate {
    before: usize,
    after: usize,
    previous: Option<usize>,
}

fn lcs_matches(
    before: &[Value],
    after: &[Value],
    equal: fn(&Value, &Value) -> bool,
    max_cells: usize,
) -> Option<Vec<ArrayMatch>> {
    if before.is_empty() || after.is_empty() {
        return Some(Vec::new());
    }
    if before.len() * after.len() > max_cells {
        return None;
    }
    let width = after.len() + 1;
    let mut lengths = vec![0u32; (before.len() + 1) * width];
    for left in (0..before.len()).rev() {
        for right in (0..after.len()).rev() {
            let at = left * width + right;
            lengths[at] = if equal(&before[left], &after[right]) {
                lengths[(left + 1) * width + right + 1] + 1
            } else {
                lengths[(left + 1) * width + right].max(lengths[left * width + right + 1])
            };
        }
    }
    let mut matches = Vec::new();
    let (mut left, mut right) = (0, 0);
    while left < before.len() && right < after.len() {
        if equal(&before[left], &after[right])
            && lengths[left * width + right] == lengths[(left + 1) * width + right + 1] + 1
        {
            matches.push((left, right));
            left += 1;
            right += 1;
        } else if lengths[(left + 1) * width + right] >= lengths[left * width + right + 1] {
            left += 1;
        } else {
            right += 1;
        }
    }
    Some(matches)
}

fn semantically_aligned(left: &Value, right: &Value) -> bool {
    if same_value(left, right) {
        return true;
    }
    match (left, right) {
        (Value::Array(left), Value::Array(right)) => {
            if left.len() != right.len() {
                return false;
            }
            left.iter()
                .zip(right)
                .any(|(value, other)| is_container(value) && same_value(value, other))
        }
        (Value::Object(left), Value::Object(right)) => left.iter().any(|(key, value)| {
            is_container(value) && right.get(key).is_some_and(|other| same_value(value, other))
        }),
        _ => false,
    }
}

fn lower_bound(values: &[usize], value: usize) -> usize {
    values.partition_point(|candidate| *candidate < value)
}

#[allow(clippy::needless_range_loop)]
fn identity_subsequence(
    before: &[Value],
    before_start: usize,
    before_end: usize,
    after: &[Value],
    after_start: usize,
    after_end: usize,
) -> Option<Vec<ArrayMatch>> {
    let before_count = before_end - before_start;
    let after_count = after_end - after_start;
    let mut matches = Vec::new();
    if after_count < before_count {
        let mut before_index = before_start;
        for after_index in after_start..after_end {
            while before_index < before_end
                && !same_value(&before[before_index], &after[after_index])
            {
                before_index += 1;
            }
            if before_index == before_end {
                return None;
            }
            matches.push((before_index, after_index));
            before_index += 1;
        }
        return Some(matches);
    }
    if before_count < after_count {
        let mut after_index = after_start;
        for before_index in before_start..before_end {
            while after_index < after_end && !same_value(&before[before_index], &after[after_index])
            {
                after_index += 1;
            }
            if after_index == after_end {
                return None;
            }
            matches.push((before_index, after_index));
            after_index += 1;
        }
        return Some(matches);
    }
    None
}

fn greedy_identity_anchors(
    positions: &HashMap<String, Vec<usize>>,
    after_keys: &[(usize, String)],
) -> Vec<ArrayMatch> {
    let mut matches = Vec::new();
    let mut previous: Option<usize> = None;
    for (after_index, key) in after_keys {
        let Some(candidates) = positions.get(key) else {
            continue;
        };
        let at = lower_bound(candidates, previous.map_or(0, |previous| previous + 1));
        let Some(&before_index) = candidates.get(at) else {
            continue;
        };
        matches.push((before_index, *after_index));
        previous = Some(before_index);
    }
    matches
}

fn identity_anchors(
    before: &[Value],
    before_start: usize,
    before_end: usize,
    after: &[Value],
    after_start: usize,
    after_end: usize,
) -> Vec<ArrayMatch> {
    let mut positions: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, value) in before
        .iter()
        .enumerate()
        .take(before_end)
        .skip(before_start)
    {
        positions
            .entry(identity_key(value))
            .or_default()
            .push(index);
    }
    let after_keys: Vec<(usize, String)> = (after_start..after_end)
        .map(|index| (index, identity_key(&after[index])))
        .collect();
    let mut candidate_count = 0;
    for (_, key) in &after_keys {
        candidate_count += positions.get(key).map_or(0, Vec::len);
        if candidate_count > MAX_IDENTITY_CANDIDATES {
            return greedy_identity_anchors(&positions, &after_keys);
        }
    }
    if candidate_count == 0 {
        return Vec::new();
    }

    let mut candidates: Vec<MatchCandidate> = Vec::new();
    let mut tails: Vec<usize> = Vec::new();
    let mut tail_values: Vec<usize> = Vec::new();
    for (after_index, key) in &after_keys {
        let Some(before_positions) = positions.get(key) else {
            continue;
        };
        for &before_index in before_positions.iter().rev() {
            let at = lower_bound(&tail_values, before_index);
            let candidate_index = candidates.len();
            candidates.push(MatchCandidate {
                before: before_index,
                after: *after_index,
                previous: if at == 0 { None } else { Some(tails[at - 1]) },
            });
            if at == tails.len() {
                tails.push(candidate_index);
                tail_values.push(before_index);
            } else {
                tails[at] = candidate_index;
                tail_values[at] = before_index;
            }
        }
    }
    let mut matches = Vec::new();
    let mut candidate_index = tails.last().copied();
    while let Some(index) = candidate_index {
        let candidate = &candidates[index];
        matches.push((candidate.before, candidate.after));
        candidate_index = candidate.previous;
    }
    matches.reverse();
    matches
}

struct ArrayDiff<'a> {
    before: &'a [Value],
    after: &'a [Value],
    path: &'a Path,
}

impl ArrayDiff<'_> {
    #[allow(clippy::too_many_arguments)]
    fn process_matches(
        &self,
        operations: &mut Operations,
        before_start: usize,
        before_end: usize,
        after_start: usize,
        after_end: usize,
        output_start: usize,
        matches: &[ArrayMatch],
    ) {
        let mut before_at = before_start;
        let mut after_at = after_start;
        let mut output_at = output_start;
        for &(before_match, after_match) in matches {
            if operations.overflowed {
                return;
            }
            self.region(
                operations,
                before_at,
                before_match,
                after_at,
                after_match,
                output_at,
            );
            output_at += after_match - after_at;
            if !same_value(&self.before[before_match], &self.after[after_match]) {
                let mut child = self.path.clone();
                child.push(Seg::Index(output_at));
                diff_value(
                    &self.before[before_match],
                    &self.after[after_match],
                    &child,
                    operations,
                );
            }
            output_at += 1;
            before_at = before_match + 1;
            after_at = after_match + 1;
        }
        if !operations.overflowed {
            self.region(
                operations, before_at, before_end, after_at, after_end, output_at,
            );
        }
    }

    fn splice(
        &self,
        operations: &mut Operations,
        output_start: usize,
        remove: usize,
        start: usize,
        end: usize,
    ) {
        operations.emit(Op::P(
            self.path.clone(),
            output_start,
            remove,
            self.after[start..end].to_vec(),
        ));
    }

    /// `diffArrayRegion`.
    fn region(
        &self,
        operations: &mut Operations,
        mut before_start: usize,
        mut before_end: usize,
        mut after_start: usize,
        mut after_end: usize,
        mut output_start: usize,
    ) {
        if operations.overflowed {
            return;
        }
        let (before, after) = (self.before, self.after);
        while before_start < before_end
            && after_start < after_end
            && same_value(&before[before_start], &after[after_start])
        {
            before_start += 1;
            after_start += 1;
            output_start += 1;
        }
        while before_start < before_end
            && after_start < after_end
            && same_value(&before[before_end - 1], &after[after_end - 1])
        {
            before_end -= 1;
            after_end -= 1;
        }
        let before_count = before_end - before_start;
        let after_count = after_end - after_start;
        if before_count == 0 && after_count == 0 {
            return;
        }
        if before_count == 0 || after_count == 0 {
            self.splice(
                operations,
                output_start,
                before_count,
                after_start,
                after_end,
            );
            return;
        }

        if before_count == after_count {
            let positional: Vec<ArrayMatch> = (0..before_count)
                .filter(|offset| {
                    same_value(&before[before_start + offset], &after[after_start + offset])
                })
                .map(|offset| (before_start + offset, after_start + offset))
                .collect();
            if !positional.is_empty() {
                self.process_matches(
                    operations,
                    before_start,
                    before_end,
                    after_start,
                    after_end,
                    output_start,
                    &positional,
                );
                return;
            }
        }

        if let Some(subsequence) = identity_subsequence(
            before,
            before_start,
            before_end,
            after,
            after_start,
            after_end,
        ) && !subsequence.is_empty()
        {
            self.process_matches(
                operations,
                before_start,
                before_end,
                after_start,
                after_end,
                output_start,
                &subsequence,
            );
            return;
        }

        let identity = identity_anchors(
            before,
            before_start,
            before_end,
            after,
            after_start,
            after_end,
        );
        if !identity.is_empty() {
            self.process_matches(
                operations,
                before_start,
                before_end,
                after_start,
                after_end,
                output_start,
                &identity,
            );
            return;
        }

        if let Some(semantic) = lcs_matches(
            &before[before_start..before_end],
            &after[after_start..after_end],
            semantically_aligned,
            MAX_SEMANTIC_CELLS,
        ) && !semantic.is_empty()
        {
            let absolute: Vec<ArrayMatch> = semantic
                .into_iter()
                .map(|(before_index, after_index)| {
                    (before_start + before_index, after_start + after_index)
                })
                .collect();
            self.process_matches(
                operations,
                before_start,
                before_end,
                after_start,
                after_end,
                output_start,
                &absolute,
            );
            return;
        }

        if before_count == 1 && after_count == 1 {
            let mut child = self.path.clone();
            child.push(Seg::Index(output_start));
            diff_value(
                &before[before_start],
                &after[after_start],
                &child,
                operations,
            );
            return;
        }
        self.splice(
            operations,
            output_start,
            before_count,
            after_start,
            after_end,
        );
    }
}

fn diff_array(before: &[Value], after: &[Value], path: &Path, operations: &mut Operations) {
    if before.len() == after.len()
        && before
            .iter()
            .zip(after)
            .all(|(left, right)| equal_json(left, right))
    {
        return;
    }
    if before.len() == after.len()
        && before.len() > 1
        && !same_value(&before[0], &after[0])
        && !same_value(&before[before.len() - 1], &after[after.len() - 1])
        && let Some(order) = permutation(before, after)
    {
        operations.emit(Op::M(path.clone(), order));
        return;
    }
    ArrayDiff {
        before,
        after,
        path,
    }
    .region(operations, 0, before.len(), 0, after.len(), 0);
}

fn diff_object(
    before: &serde_json::Map<String, Value>,
    after: &serde_json::Map<String, Value>,
    path: &Path,
    operations: &mut Operations,
) {
    if before
        .keys()
        .chain(after.keys())
        .any(|key| is_reserved(key))
    {
        let (before, after) = (Value::Object(before.clone()), Value::Object(after.clone()));
        if !equal_json(&before, &after) {
            operations.emit_set(path, &after);
        }
        return;
    }
    for (key, value) in after {
        if operations.overflowed {
            return;
        }
        let mut child = path.clone();
        child.push(Seg::Key(key.clone()));
        match before.get(key) {
            Some(previous) => diff_value(previous, value, &child, operations),
            None => operations.emit_set(&child, value),
        }
    }
    for key in before.keys() {
        if operations.overflowed {
            return;
        }
        if !after.contains_key(key) {
            let mut child = path.clone();
            child.push(Seg::Key(key.clone()));
            operations.emit(Op::D(child));
        }
    }
}

fn diff_value(before: &Value, after: &Value, path: &Path, operations: &mut Operations) {
    if operations.overflowed {
        return;
    }
    match (before, after) {
        (Value::String(before), Value::String(after)) if !path.is_empty() => {
            emit_string(before, after, path, operations)
        }
        (Value::Array(before), Value::Array(after)) => diff_array(before, after, path, operations),
        (Value::Object(before), Value::Object(after)) => {
            diff_object(before, after, path, operations)
        }
        _ if equal_json(before, after) => {}
        _ => operations.emit_set(path, after),
    }
}

fn json_cost(value: &Value) -> usize {
    match value {
        Value::Null => 4,
        Value::String(text) => utf16_len(text) + 2,
        Value::Number(number) => number.to_string().len(),
        Value::Bool(flag) => {
            if *flag {
                4
            } else {
                5
            }
        }
        Value::Array(items) => {
            let mut cost = 2;
            for (index, item) in items.iter().enumerate() {
                cost += json_cost(item) + usize::from(index != 0);
            }
            cost
        }
        Value::Object(object) => {
            let mut cost = 2;
            for (index, (key, item)) in object.iter().enumerate() {
                cost += utf16_len(key) + 3 + json_cost(item) + usize::from(index != 0);
            }
            cost
        }
    }
}

fn path_cost(path: &Path) -> usize {
    let mut cost = 2;
    for (index, segment) in path.iter().enumerate() {
        cost += match segment {
            Seg::Key(key) => utf16_len(key) + 2,
            Seg::Index(value) => value.to_string().len(),
        } + usize::from(index != 0);
    }
    cost
}

fn operation_cost(operation: &Op) -> usize {
    match operation {
        Op::R(value) => 6 + json_cost(value),
        Op::S(path, value) => 7 + path_cost(path) + json_cost(value),
        Op::D(path) => 6 + path_cost(path),
        Op::A(path, text) => 7 + path_cost(path) + utf16_len(text) + 2,
        Op::T(path, count) => 7 + path_cost(path) + count.to_string().len(),
        Op::P(path, index, remove, items) => {
            10 + path_cost(path)
                + index.to_string().len()
                + remove.to_string().len()
                + json_cost(&Value::Array(items.clone()))
        }
        Op::M(path, permutation) => {
            let mut cost = 2;
            for (index, value) in permutation.iter().enumerate() {
                cost += value.to_string().len() + usize::from(index != 0);
            }
            7 + path_cost(path) + cost
        }
    }
}

/// Compute a compact operation batch from two immutable JSON revisions (`diffRevisions`).
pub fn diff_revisions(before: &Value, after: &Value) -> Vec<Op> {
    let mut operations = Operations::default();
    diff_value(before, after, &Vec::new(), &mut operations);
    if operations.overflowed {
        return vec![Op::R(after.clone())];
    }
    let operations = operations.ops;
    if operations.is_empty() || matches!(operations[0], Op::R(_)) {
        return operations;
    }
    let mut delta_cost = 2;
    for operation in &operations {
        delta_cost += operation_cost(operation) + 1;
    }
    if delta_cost < 65_536 {
        return operations;
    }
    let snapshot_cost = json_cost(after) + 6;
    if delta_cost >= snapshot_cost {
        vec![Op::R(after.clone())]
    } else {
        operations
    }
}
