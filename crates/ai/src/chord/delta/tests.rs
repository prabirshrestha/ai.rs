//! Ports of chord `test/delta.test.ts`, `test/delta-diff.test.ts`,
//! `test/delta-apply-immutable.test.ts`, `test/delta-clone.test.ts` and the
//! lifecycle half of `test/delta-tracker/tracker.test.ts`, plus golden tests
//! against fixtures generated from the TS sources (`fixtures/generate.mjs`).
//!
//! Skipped (not representable in Rust): prototype/setter/getter traps,
//! `Object.freeze`, reference-identity (`toBe`) assertions on containers,
//! Proxy draft semantics (handles, array mutator coercions, placement
//! validation), and throwing iterators.

use super::*;
use serde_json::json;
use std::sync::Arc;

fn op(value: Value) -> Op {
    assert_valid_op(&value).unwrap()
}

fn ops(value: Value) -> Vec<Op> {
    value.as_array().unwrap().iter().cloned().map(op).collect()
}

fn wire(value: Value) -> Vec<WireOp> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| assert_valid_wire_op(item).unwrap())
        .collect()
}

fn json_of(ops: &[Op]) -> Value {
    serde_json::to_value(ops).unwrap()
}

// ─── delta.test.ts: immutable tracker lifecycle ──────────────────────────────

#[test]
fn keeps_a_draft_alive_and_adopts_only_a_prepared_change() {
    let input = json!({ "count": 1, "nested": { "text": "a" }, "values": [1] });
    let mut tracker = track(input.clone());
    let mut change = tracker.begin_change();
    change.state_mut().unwrap()["count"] = json!(2);
    let text = change.state_mut().unwrap()["nested"]["text"]
        .as_str()
        .unwrap()
        .to_string();
    change.state_mut().unwrap()["nested"]["text"] = json!(format!("{text}b"));
    change.state_mut().unwrap()["values"]
        .as_array_mut()
        .unwrap()
        .push(json!(2));
    assert_eq!(**tracker.value(), input);

    let prepared = change.prepare().unwrap();
    assert!(Arc::ptr_eq(prepared.base(), tracker.value()));
    assert_eq!(
        **prepared.value(),
        json!({ "count": 2, "nested": { "text": "ab" }, "values": [1, 2] })
    );
    assert_eq!(
        json_of(prepared.ops()),
        json!([
            ["s", ["count"], 2],
            ["a", ["nested", "text"], "b"],
            ["p", ["values"], 1, 0, [2]]
        ])
    );
    assert!(change.state().unwrap_err().is_type_error());
    assert_eq!(**tracker.value(), input);

    tracker.adopt(&prepared).unwrap();
    assert!(Arc::ptr_eq(tracker.value(), prepared.value()));
    assert_eq!(prepared.base_revision(), 0);
    assert_eq!(tracker.revision(), 1);
}

#[test]
fn aborts_and_revokes_without_changing_the_committed_value() {
    let mut tracker = track(json!({ "child": { "value": 1 } }));
    let mut change = tracker.begin_change();
    change.state_mut().unwrap()["child"]["value"] = json!(2);
    change.abort();
    assert_eq!(tracker.value()["child"]["value"], json!(1));
    assert!(change.state().unwrap_err().is_type_error());
    change.abort();
    assert!(
        change
            .prepare()
            .unwrap_err()
            .to_string()
            .contains("settled")
    );
    let mut next = tracker.begin_change();
    next.abort();
}

#[test]
fn grows_arrays_with_explicit_nulls_and_revokes_drafts_after_preparation() {
    let mut tracker = track(json!({ "values": [1, 2] }));
    let mut change = tracker.begin_change();
    change.state_mut().unwrap()["values"]
        .as_array_mut()
        .unwrap()
        .resize(4, Value::Null);
    let prepared = change.prepare().unwrap();
    assert_eq!(prepared.value()["values"], json!([1, 2, null, null]));
    assert!(change.state().is_err());
    tracker.adopt(&prepared).unwrap();
}

#[test]
fn normalizes_a_deep_no_op_to_exact_previous_identity() {
    let mut tracker = track(json!({ "value": { "nested": [1, 2] } }));
    let mut change = tracker.begin_change();
    change.state_mut().unwrap()["value"] = json!({ "nested": [1, 2] });
    let prepared = change.prepare().unwrap();
    assert!(Arc::ptr_eq(prepared.value(), prepared.base()));
    assert!(prepared.ops().is_empty());
    let base = prepared.base().clone();
    tracker.adopt(&prepared).unwrap();
    assert!(Arc::ptr_eq(tracker.value(), &base));
}

#[test]
fn takes_immutable_ownership_of_replacement_input_and_normalizes_no_ops() {
    let mut tracker = track(json!({ "nested": { "value": 1 } }));
    let replacement = Arc::new(json!({ "nested": { "value": 2 } }));
    let prepared = tracker.prepare_replace(replacement.clone());
    assert!(Arc::ptr_eq(prepared.value(), &replacement));
    assert_eq!(
        json_of(prepared.ops()),
        json!([["r", { "nested": { "value": 2 } }]])
    );
    tracker.adopt(&prepared).unwrap();

    let no_op = tracker.prepare_replace(json!({ "nested": { "value": 2 } }));
    assert!(Arc::ptr_eq(no_op.value(), tracker.value()));
    assert!(no_op.ops().is_empty());
}

#[test]
fn rejects_foreign_stale_and_repeated_preparations_while_allowing_competing_changes() {
    let mut first = track(json!({ "value": 0 }));
    let mut second = track(json!({ "value": 0 }));
    let mut change = first.begin_change();
    let mut competing = first.begin_change();
    change.state_mut().unwrap()["value"] = json!(1);
    competing.state_mut().unwrap()["value"] = json!(2);
    let prepared = change.prepare().unwrap();
    let competing_prepared = competing.prepare().unwrap();
    assert!(
        second
            .adopt(&prepared)
            .unwrap_err()
            .to_string()
            .contains("different tracker")
    );
    first.adopt(&prepared).unwrap();
    assert!(
        first
            .adopt(&prepared)
            .unwrap_err()
            .to_string()
            .contains("already been used")
    );
    assert!(
        first
            .adopt(&competing_prepared)
            .unwrap_err()
            .to_string()
            .contains("stale")
    );

    let stale = first.prepare_replace(json!({ "value": 2 }));
    let winner = first.prepare_replace(json!({ "value": 3 }));
    first.adopt(&winner).unwrap();
    assert!(
        first
            .adopt(&stale)
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
    assert!(
        first
            .adopt(&stale)
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
}

#[test]
fn invalidates_a_prepared_result_when_its_change_is_aborted() {
    let mut tracker = track(json!({ "value": 0 }));
    let mut change = tracker.begin_change();
    change.state_mut().unwrap()["value"] = json!(1);
    let prepared = change.prepare().unwrap();
    change.abort();
    assert!(
        tracker
            .adopt(&prepared)
            .unwrap_err()
            .to_string()
            .contains("aborted")
    );
    change.abort();
}

#[test]
fn makes_competing_same_base_preparations_stale_after_adopting_a_no_op() {
    let mut tracker = track(json!({ "value": { "count": 1 } }));
    let first = tracker.prepare_replace(json!({ "value": { "count": 1 } }));
    let competing = tracker.prepare_replace(json!({ "value": { "count": 1 } }));
    assert!(Arc::ptr_eq(first.value(), first.base()));
    assert!(Arc::ptr_eq(competing.value(), competing.base()));
    tracker.adopt(&first).unwrap();
    assert!(
        tracker
            .adopt(&first)
            .unwrap_err()
            .to_string()
            .contains("already been used")
    );
    assert!(
        tracker
            .adopt(&competing)
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
}

#[test]
fn emits_an_owned_root_replacement_for_large_replacement_input() {
    let rows: Vec<Value> = (0..10_000)
        .map(|value| json!({ "value": value, "stable": { "value": value } }))
        .collect();
    let mut tracker = track(json!({ "rows": rows }));
    let mut replacement = (**tracker.value()).clone();
    replacement["rows"][5_000]["value"] = json!(-1);
    let replacement = Arc::new(replacement);
    let prepared = tracker.prepare_replace(replacement.clone());
    assert!(Arc::ptr_eq(prepared.value(), &replacement));
    assert_eq!(prepared.ops().len(), 1);
    assert!(matches!(prepared.ops()[0], Op::R(_)));
    assert_eq!(
        apply_immutable(prepared.base(), prepared.ops()).unwrap(),
        **prepared.value()
    );
}

// ─── tracker.test.ts: lifecycle ──────────────────────────────────────────────

#[test]
fn allows_competing_contexts_and_invalidates_loser_views() {
    let mut tracker = track(json!({ "value": 0 }));
    let mut first = tracker.begin_change();
    let mut second = tracker.begin_change();
    first.state_mut().unwrap()["value"] = json!(1);
    second.state_mut().unwrap()["value"] = json!(2);
    let first_prepared = first.prepare().unwrap();
    let second_prepared = second.prepare().unwrap();
    let held = second_prepared.value().clone();
    tracker.adopt(&first_prepared).unwrap();
    assert_eq!(tracker.value()["value"], json!(1));
    assert_eq!(held["value"], json!(2));
    assert!(
        tracker
            .adopt(&second_prepared)
            .unwrap_err()
            .to_string()
            .contains("stale")
    );
    assert!(
        tracker
            .adopt(&first_prepared)
            .unwrap_err()
            .to_string()
            .contains("already been used")
    );
}

#[test]
fn invalidates_competing_open_overlays_without_changing_their_base_revision() {
    let initial = json!({ "rows": [{ "value": 0 }, { "value": 1 }] });
    let mut tracker = track(initial.clone());
    let mut stale_change = tracker.begin_change();
    stale_change.state_mut().unwrap()["rows"][1]["value"] = json!(2);

    let mut winner = tracker.begin_change();
    winner.state_mut().unwrap()["rows"][0]["value"] = json!(3);
    let prepared = winner.prepare().unwrap();
    tracker.adopt(&prepared).unwrap();

    assert_eq!(
        **tracker.value(),
        json!({ "rows": [{ "value": 3 }, { "value": 1 }] })
    );
    assert!(
        stale_change
            .state()
            .unwrap_err()
            .to_string()
            .contains("settled")
    );
    assert!(
        stale_change
            .prepare()
            .unwrap_err()
            .to_string()
            .contains("settled")
    );
    stale_change.abort();
}

#[test]
fn adopts_object_edits_and_keeps_retained_publications() {
    let mut tracker = track(json!({ "first": 1, "second": 2 }));
    let mut change = tracker.begin_change();
    let state = change.state_mut().unwrap().as_object_mut().unwrap();
    state["first"] = json!(3);
    state.shift_remove("second");
    state.insert("third".into(), json!(4));
    let prepared = change.prepare().unwrap();
    tracker.adopt(&prepared).unwrap();
    assert_eq!(**tracker.value(), json!({ "first": 3, "third": 4 }));

    let published = prepared.value().clone();
    let mut next = tracker.begin_change();
    next.state_mut().unwrap()["first"] = json!(5);
    let next_prepared = next.prepare().unwrap();
    tracker.adopt(&next_prepared).unwrap();
    assert_eq!(*published, json!({ "first": 3, "third": 4 }));
    assert_eq!(tracker.value()["first"], json!(5));
}

#[test]
fn keeps_aborted_and_stale_materialized_candidates_readable() {
    let mut tracker = track(json!({ "value": 0 }));
    let mut aborted_change = tracker.begin_change();
    aborted_change.state_mut().unwrap()["value"] = json!(1);
    let aborted = aborted_change.prepare().unwrap();
    aborted.abort();
    aborted.abort();
    assert_eq!(aborted.value()["value"], json!(1));
    assert!(
        tracker
            .adopt(&aborted)
            .unwrap_err()
            .to_string()
            .contains("aborted")
    );

    let mut loser_change = tracker.begin_change();
    loser_change.state_mut().unwrap()["value"] = json!(2);
    let loser = loser_change.prepare().unwrap();
    let winner = tracker.prepare_replace(json!({ "value": 3 }));
    tracker.adopt(&winner).unwrap();
    assert_eq!(loser.value()["value"], json!(2));
}

#[test]
fn edits_typed_drafts() {
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Doc {
        output: String,
        entries: Vec<u32>,
    }
    let mut tracker = track(json!({ "output": "", "entries": [] }));
    let mut change = tracker.begin_change();
    change
        .edit(|doc: &mut Doc| {
            doc.output.push_str("done\n");
            doc.entries.push(1);
        })
        .unwrap();
    let prepared = change.prepare().unwrap();
    assert_eq!(
        json_of(prepared.ops()),
        json!([["a", ["output"], "done\n"], ["p", ["entries"], 0, 0, [1]]])
    );
    let replica = apply_immutable(prepared.base(), prepared.ops()).unwrap();
    assert_eq!(replica, **prepared.value());
}

#[test]
fn replays_tracker_batches_across_revisions() {
    let initial = json!({
        "text": "start",
        "values": (0..8).map(|id| json!({ "id": id, "score": 0 })).collect::<Vec<_>>(),
        "revision": 0,
    });
    let mut tracker = track(initial.clone());
    let mut batches: Vec<Arc<[Op]>> = Vec::new();
    for revision in 1..=40i64 {
        let mut change = tracker.begin_change();
        let state = change.state_mut().unwrap();
        state["revision"] = json!(revision);
        let text = state["text"].as_str().unwrap()[1..].to_string();
        state["text"] = json!(format!("{text}{revision}"));
        let values = state["values"].as_array_mut().unwrap();
        match revision % 5 {
            0 => values.reverse(),
            1 => values.push(json!({ "id": 100 + revision, "score": revision })),
            2 => {
                values.remove(0);
            }
            3 => {
                let index = revision as usize % values.len();
                values[index]["score"] = json!(revision);
            }
            _ => {
                values.splice(1..2, [json!({ "id": 200 + revision, "score": revision })]);
            }
        }
        let prepared = change.prepare().unwrap();
        batches.push(prepared.ops().clone());
        tracker.adopt(&prepared).unwrap();
    }
    let replayed =
        apply_immutable_batches(&initial, batches.iter().map(|batch| &batch[..])).unwrap();
    assert_eq!(replayed, **tracker.value());
}

// ─── delta.test.ts: strings, application, validation ────────────────────────

#[test]
fn emits_append_and_rolling_window_operations() {
    let mut tracker = track(json!({ "text": "abcdefgh" }));
    let mut change = tracker.begin_change();
    change.state_mut().unwrap()["text"] = json!("abcdefghij");
    let prepared = change.prepare().unwrap();
    assert_eq!(json_of(prepared.ops()), json!([["a", ["text"], "ij"]]));
    tracker.adopt(&prepared).unwrap();

    let mut change = tracker.begin_change();
    let text = change.state().unwrap()["text"].as_str().unwrap()[3..].to_string();
    change.state_mut().unwrap()["text"] = json!(format!("{text}xyz"));
    let prepared = change.prepare().unwrap();
    assert_eq!(
        json_of(prepared.ops()),
        json!([["t", ["text"], 3], ["a", ["text"], "xyz"]])
    );
}

#[test]
fn finds_bounded_overlaps() {
    assert_eq!(overlap("abcdefgh", "defghxyz", 65_536), 5);
    assert_eq!(overlap("abcdef", "defghi", 0), 0);
    // UTF-16 code units, as in JS.
    assert_eq!(overlap("x😀y", "😀yz", 65_536), 3);
    assert_eq!(overlap("éab", "abc", 65_536), 2);
}

#[test]
fn applies_mutable_and_immutable_operations() {
    let operations = ops(json!([
        ["a", ["text"], "b"],
        ["p", ["values"], 1, 1, [3, 4]],
        ["s", ["nested", "value"], 2]
    ]));
    let base = json!({ "text": "a", "values": [1, 2], "nested": { "value": 1 }, "stable": { "value": 9 } });
    let immutable = apply_immutable(&base, &operations).unwrap();
    assert_eq!(
        immutable,
        json!({ "text": "ab", "values": [1, 3, 4], "nested": { "value": 2 }, "stable": { "value": 9 } })
    );
    assert_eq!(
        base,
        json!({ "text": "a", "values": [1, 2], "nested": { "value": 1 }, "stable": { "value": 9 } })
    );
    assert_eq!(apply(base.clone(), &operations).unwrap(), immutable);
}

#[test]
fn supports_root_replacement_root_splice_and_permutations() {
    let base = ops(json!([["r", [1, 2, 3]]]));
    assert!(is_base(&base));
    let mut value = apply(Value::Null, &base).unwrap();
    value = apply(value, &ops(json!([["p", [], 1, 1, [4]]]))).unwrap();
    value = apply(value, &ops(json!([["m", [], [2, 0, 1]]]))).unwrap();
    assert_eq!(value, json!([3, 1, 4]));
}

#[test]
fn rejects_unsafe_and_malformed_paths() {
    let error =
        assert_valid_op(&json!(["s", ["constructor", "prototype", "x"], true])).unwrap_err();
    assert!(error.is_unsafe_path_error());
    assert_eq!(error.to_string(), "unsafe path segment: constructor");
    let error = apply(
        json!({ "values": [1] }),
        &ops(json!([["s", ["values", 3], 2]])),
    )
    .unwrap_err();
    assert!(error.is_unsafe_path_error());
    assert!(apply(json!({ "value": 1 }), &ops(json!([["a", ["value"], "x"]]))).is_err());
    assert!(assert_valid_op(&json!(["s", "value", 1])).is_err());
    assert!(
        assert_valid_op(&json!(["m", [], [0, 0]]))
            .unwrap_err()
            .to_string()
            .contains("bijection")
    );
    let typed = Op::S(vec![Seg::Key("__proto__".into())], json!(1));
    assert!(
        apply(json!({}), &[typed])
            .unwrap_err()
            .is_unsafe_path_error()
    );
    let rooted = Op::S(Vec::new(), json!(1));
    assert_eq!(
        apply(json!({}), &[rooted]).unwrap_err().to_string(),
        "path is empty"
    );
}

#[test]
fn validates_decoded_and_wire_vocabularies_separately() {
    assert!(assert_valid_op(&json!(["s", ["value"], 1])).is_ok());
    assert!(assert_valid_op(&json!(["s", 1])).is_err());
    assert!(assert_valid_wire_op(&json!(["s", 1])).is_ok());
    assert!(assert_valid_wire_op(&json!(["#", 0, ["value"]])).is_ok());
}

// ─── delta.test.ts: codec ────────────────────────────────────────────────────

#[test]
fn interns_paths_omits_adjacent_paths_and_round_trips() {
    let mut enc = encoder();
    let mut dec = decoder();
    let first = ops(json!([
        ["t", ["nested", "text"], 1],
        ["a", ["nested", "text"], "x"]
    ]));
    assert_eq!(dec.decode(&enc.encode(&first)).unwrap(), first);
    let second = ops(json!([["a", ["nested", "text"], "y"]]));
    let encoded = enc.encode(&second);
    assert_eq!(
        serde_json::to_value(&encoded).unwrap(),
        json!([["#", 0, ["nested", "text"]], ["a", 0, "y"]])
    );
    assert_eq!(dec.decode(&encoded).unwrap(), second);
}

#[test]
fn resets_path_dictionaries_on_a_base() {
    let mut enc = encoder();
    enc.encode(&ops(json!([["s", ["value"], 1]])));
    enc.encode(&ops(json!([["s", ["value"], 2]])));
    let base = enc.encode(&ops(json!([["r", { "value": 3 }]])));
    assert_eq!(
        serde_json::to_value(&base).unwrap(),
        json!([["r", { "value": 3 }]])
    );
    let after = enc.encode(&ops(json!([["s", ["value"], 4]])));
    assert_eq!(
        serde_json::to_value(&after).unwrap(),
        json!([["s", ["value"], 4]])
    );
}

#[test]
fn rejects_unresolved_short_forms_and_unsafe_interned_paths() {
    assert!(decoder().decode(&wire(json!([["a", "x"]]))).is_err());
    let error = decoder()
        .decode_json(&[json!(["#", 0, ["__proto__"]]), json!(["s", 0, true])])
        .unwrap_err();
    assert!(error.is_unsafe_path_error());
    let typed = [
        WireOp::Define(0, vec![Seg::Key("__proto__".into())]),
        WireOp::S(Some(PathRef::Id(0)), json!(true)),
    ];
    assert!(decoder().decode(&typed).unwrap_err().is_unsafe_path_error());
}

#[test]
fn omits_an_adjacent_repeated_path() {
    let encoded = encoder().encode(&ops(json!([["s", ["value"], 1], ["s", ["value"], 2]])));
    assert_eq!(
        serde_json::to_value(&encoded).unwrap(),
        json!([["s", ["value"], 1], ["s", 2]])
    );
}

#[test]
fn interns_on_second_use_rather_than_first() {
    let mut enc = encoder();
    let first = enc.encode(&ops(json!([["a", ["a", "deep"], "1"]])));
    assert_eq!(
        serde_json::to_value(&first).unwrap(),
        json!([["a", ["a", "deep"], "1"]])
    );
    let second = enc.encode(&ops(json!([["a", ["a", "deep"], "2"]])));
    assert_eq!(
        serde_json::to_value(&second).unwrap(),
        json!([["#", 0, ["a", "deep"]], ["a", 0, "2"]])
    );
}

#[test]
fn does_not_collide_paths_containing_null_characters() {
    let operations = ops(json!([["s", ["a\u{0000}b"], 1], ["s", ["a", "b"], 2]]));
    assert_eq!(
        decoder().decode(&encoder().encode(&operations)).unwrap(),
        operations
    );
}

#[test]
fn clears_decoder_ids_on_a_base_batch() {
    let mut dec = decoder();
    dec.decode(&wire(json!([["#", 0, ["a"]], ["a", 0, "1"]])))
        .unwrap();
    dec.decode(&wire(json!([["r", { "a": "" }]]))).unwrap();
    let error = dec.decode(&wire(json!([["a", 0, "2"]]))).unwrap_err();
    assert_eq!(error.to_string(), "unresolvable path: 0");
}

#[test]
fn makes_batches_after_a_base_self_contained() {
    let mut enc = encoder();
    enc.encode(&ops(json!([["a", ["a", "deep"], "1"]])));
    enc.encode(&ops(json!([["a", ["a", "deep"], "2"]])));
    let base = enc.encode(&ops(json!([["r", { "a": { "deep": "x" } }]])));
    let after = enc.encode(&ops(json!([["a", ["a", "deep"], "3"]])));
    assert_eq!(
        serde_json::to_value(&after).unwrap(),
        json!([["a", ["a", "deep"], "3"]])
    );
    let mut dec = decoder();
    assert_eq!(
        json_of(&dec.decode(&base).unwrap()),
        json!([["r", { "a": { "deep": "x" } }]])
    );
    assert_eq!(
        json_of(&dec.decode(&after).unwrap()),
        json!([["a", ["a", "deep"], "3"]])
    );
}

#[test]
fn round_trips_deterministic_mixed_operation_streams() {
    let batches: Vec<Vec<Op>> = (0..100usize)
        .map(|index| {
            vec![
                Op::S(path!["rows", index, "value"], json!(index)),
                Op::A(path!["output"], index.to_string()),
                Op::P(path!["tail"], index, 0, vec![json!(index)]),
            ]
        })
        .collect();
    let mut enc = encoder();
    let mut dec = decoder();
    for batch in &batches {
        assert_eq!(&dec.decode(&enc.encode(batch)).unwrap(), batch);
    }
}

// ─── delta.test.ts: ownership, index and structure safety ────────────────────

#[test]
fn does_not_mutate_a_replacement_payload_targeted_by_a_later_operation() {
    let operations =
        ops(json!([["r", { "nested": { "value": 1 } }], ["s", ["nested", "value"], 2]]));
    let next = apply_immutable(&Value::Null, &operations).unwrap();
    assert_eq!(operations[0], Op::R(json!({ "nested": { "value": 1 } })));
    assert_eq!(next["nested"]["value"], json!(2));
}

#[test]
fn allows_reserved_names_inside_values() {
    let value: Value = serde_json::from_str(r#"{"__proto__":{"z":1}}"#).unwrap();
    let out = apply(json!({}), &[Op::S(path!["value"], value.clone())]).unwrap();
    assert_eq!(out["value"], value);
}

#[test]
fn writes_an_existing_index_and_appends_exactly_one_past_the_end() {
    assert_eq!(
        apply(
            json!({ "values": [1, 2, 3] }),
            &ops(json!([["s", ["values", 1], 9]]))
        )
        .unwrap(),
        json!({ "values": [1, 9, 3] })
    );
    assert_eq!(
        apply(
            json!({ "values": [1, 2, 3] }),
            &ops(json!([["s", ["values", 3], 9]]))
        )
        .unwrap(),
        json!({ "values": [1, 2, 3, 9] })
    );
}

#[test]
fn rejects_gaps_huge_indices_and_string_spelled_indices() {
    assert!(
        apply(
            json!({ "values": [1, 2, 3] }),
            &ops(json!([["s", ["values", 5], 9]]))
        )
        .is_err()
    );
    assert!(
        apply(
            json!({ "values": [] }),
            &ops(json!([["s", ["values", 4_294_967_290u64], 1]]))
        )
        .is_err()
    );
    assert!(
        apply(
            json!({ "values": [1] }),
            &ops(json!([["s", ["values", "0"], 9]]))
        )
        .is_err()
    );
    assert!(
        apply(
            json!({ "values": ["a"] }),
            &ops(json!([["a", ["values", "0"], "b"]]))
        )
        .is_err()
    );
}

#[test]
fn allows_explicit_growth_values_and_rejects_deletion_past_the_end() {
    assert_eq!(
        apply(
            json!({ "values": [1] }),
            &ops(json!([["p", ["values"], 1, 0, [null, null, 9]]]))
        )
        .unwrap(),
        json!({ "values": [1, null, null, 9] })
    );
    assert!(
        apply(
            json!({ "values": [1] }),
            &ops(json!([["d", ["values", 1]]]))
        )
        .is_err()
    );
}

#[test]
fn applies_large_splice_payloads() {
    let items = vec![Value::Null; 300_000];
    let result = apply(
        json!({ "values": [] }),
        &[Op::P(path!["values"], 0, 0, items)],
    )
    .unwrap();
    assert_eq!(result["values"].as_array().unwrap().len(), 300_000);
}

#[test]
fn rejects_unknown_verbs_malformed_tuples_paths_and_splice_payloads() {
    for invalid in [
        json!(["ZZZ", ["value"], 9]),
        json!(["p", ["values"], 0, 0, "not-an-array"]),
        json!(["s", "value", 9]),
        json!({ "op": "s" }),
        Value::Null,
    ] {
        assert!(assert_valid_op(&invalid).is_err(), "{invalid}");
        assert!(serde_json::from_value::<Op>(invalid).is_err());
    }
    assert_eq!(
        assert_valid_op(&json!(["ZZZ", ["value"], 9]))
            .unwrap_err()
            .to_string(),
        "unknown op verb: ZZZ"
    );
}

#[test]
fn rejects_invalid_append_and_truncation_operations() {
    assert!(
        apply(
            json!({ "value": 1 }),
            &ops(json!([["a", ["missing"], "x"]]))
        )
        .is_err()
    );
    assert!(apply(json!({ "value": 1 }), &ops(json!([["a", ["value"], "x"]]))).is_err());
    assert_eq!(
        assert_valid_op(&json!(["t", ["value"], -1]))
            .unwrap_err()
            .to_string(),
        "t shape"
    );
    assert!(
        decoder()
            .decode_json(&[json!(["t", ["value"], -1])])
            .is_err()
    );
}

#[test]
fn truncates_by_utf16_code_units() {
    let value = apply(json!({ "text": "😀ab" }), &ops(json!([["t", ["text"], 2]]))).unwrap();
    assert_eq!(value["text"], json!("ab"));
    let value = apply(json!({ "text": "ab" }), &ops(json!([["t", ["text"], 9]]))).unwrap();
    assert_eq!(value["text"], json!(""));
    // Splitting a surrogate pair is not representable in a Rust string.
    let error = apply(json!({ "text": "😀ab" }), &ops(json!([["t", ["text"], 1]]))).unwrap_err();
    assert!(error.is_path_error());
}

#[test]
fn clamps_splice_removal_past_the_end() {
    assert_eq!(
        apply(
            json!({ "values": [1, 2] }),
            &ops(json!([["p", ["values"], 0, 1e9, []]]))
        )
        .unwrap(),
        json!({ "values": [] })
    );
}

#[test]
fn accepts_decoded_operations_and_rejects_wire_only_forms() {
    for operation in [
        json!(["r", { "value": 1 }]),
        json!(["s", ["value"], 1]),
        json!(["d", ["value"]]),
        json!(["a", ["value"], "x"]),
        json!(["t", ["value"], 2]),
        json!(["p", ["value"], 0, 0, []]),
        json!(["m", ["value"], [0]]),
    ] {
        assert!(assert_valid_op(&operation).is_ok(), "{operation}");
    }
    for wire_only in [
        json!(["s", 1]),
        json!(["d"]),
        json!(["a", "x"]),
        json!(["t", 2]),
        json!(["p", 0, 0, []]),
        json!(["#", 0, ["value"]]),
        json!(["s", 0, 1]),
    ] {
        assert!(assert_valid_op(&wire_only).is_err(), "{wire_only}");
        assert!(assert_valid_wire_op(&wire_only).is_ok(), "{wire_only}");
    }
}

// ─── delta-diff.test.ts ──────────────────────────────────────────────────────

fn expect_diff(before: Value, after: Value, expected: Value) {
    let operations = diff_revisions(&before, &after);
    assert_eq!(json_of(&operations), expected);
    assert_eq!(apply_immutable(&before, &operations).unwrap(), after);
}

#[test]
fn diff_emits_sets_and_deletes() {
    expect_diff(
        json!({ "keep": 1, "change": 1, "remove": true }),
        json!({ "keep": 1, "change": 2, "add": 3 }),
        json!([["s", ["change"], 2], ["s", ["add"], 3], ["d", ["remove"]]]),
    );
}

#[test]
fn diff_emits_string_append_and_front_truncation() {
    expect_diff(
        json!({ "text": "hello" }),
        json!({ "text": "hello world" }),
        json!([["a", ["text"], " world"]]),
    );
    expect_diff(
        json!({ "text": "hello world" }),
        json!({ "text": "world" }),
        json!([["t", ["text"], 6]]),
    );
    expect_diff(
        json!({ "text": "abcdefgh" }),
        json!({ "text": "defghxyz" }),
        json!([["t", ["text"], 3], ["a", ["text"], "xyz"]]),
    );
}

#[test]
fn diff_represents_array_insertion_removal_and_shift_with_splices() {
    let (a, b, c) = (
        json!({ "id": "a" }),
        json!({ "id": "b" }),
        json!({ "id": "c" }),
    );
    expect_diff(
        json!({ "values": [a, b] }),
        json!({ "values": [a, c, b] }),
        json!([["p", ["values"], 1, 0, [c]]]),
    );
    expect_diff(
        json!({ "values": [a, b, c] }),
        json!({ "values": [b, c] }),
        json!([["p", ["values"], 0, 1, []]]),
    );
}

#[test]
fn diff_collapses_a_same_length_queue_update_to_two_splices() {
    let [a, b, c, d] = ["a", "b", "c", "d"].map(|id| json!({ "id": id }));
    expect_diff(
        json!({ "values": [a, b, c] }),
        json!({ "values": [b, c, d] }),
        json!([["p", ["values"], 0, 1, []], ["p", ["values"], 2, 0, [d]]]),
    );
}

#[test]
fn diff_emits_a_permutation_for_a_pure_reorder() {
    let [a, b, c] = ["a", "b", "c"].map(|id| json!({ "id": id }));
    expect_diff(
        json!({ "values": [a, b, c] }),
        json!({ "values": [c, a, b] }),
        json!([["m", ["values"], [2, 0, 1]]]),
    );
}

#[test]
fn diff_normalizes_reordered_deeply_equal_objects_to_a_no_op() {
    let first = json!({ "nested": { "value": 1 } });
    let second = json!({ "nested": { "value": 1 } });
    assert!(
        diff_revisions(
            &json!({ "values": [first, second] }),
            &json!({ "values": [second, first] })
        )
        .is_empty()
    );
}

#[test]
fn diff_validates_and_encodes_permutations() {
    let operations = ops(json!([
        ["m", ["values"], [2, 0, 1]],
        ["m", ["values"], [1, 2, 0]]
    ]));
    for operation in &operations {
        operation.validate().unwrap();
    }
    let encoded = encoder().encode(&operations);
    assert_eq!(
        serde_json::to_value(&encoded).unwrap(),
        json!([["m", ["values"], [2, 0, 1]], ["m", [1, 2, 0]]])
    );
    assert_eq!(decoder().decode(&encoded).unwrap(), operations);
    assert!(
        Op::M(path!["values"], vec![0, 0])
            .validate()
            .unwrap_err()
            .to_string()
            .contains("bijection")
    );
}

#[test]
fn diff_emits_nothing_for_deeply_equal_reconstructed_values() {
    assert!(
        diff_revisions(
            &json!({ "value": { "nested": [1, 2] } }),
            &json!({ "value": { "nested": [1, 2] } })
        )
        .is_empty()
    );
    assert!(
        diff_revisions(
            &json!({ "values": [{ "id": 1 }, { "id": 2 }] }),
            &json!({ "values": [{ "id": 1 }, { "id": 2 }] })
        )
        .is_empty()
    );
    assert!(
        diff_revisions(
            &json!({ "values": [true, true, true] }),
            &json!({ "values": [true, true, true] })
        )
        .is_empty()
    );
}

#[test]
fn diff_keeps_a_leaf_edit_inside_a_reconstructed_array_narrow() {
    expect_diff(
        json!({ "values": [{ "id": 1, "label": "one" }, { "id": 2, "label": "two" }] }),
        json!({ "values": [{ "id": 1, "label": "one" }, { "id": 2, "label": "changed" }] }),
        json!([["s", ["values", 1, "label"], "changed"]]),
    );
}

#[test]
fn diff_emits_payload_free_splices_for_scattered_removals() {
    let [a, b, c, d, e] = ["a", "b", "c", "d", "e"].map(|id| json!({ "id": id }));
    expect_diff(
        json!({ "values": [a, b, c, d, e] }),
        json!({ "values": [a, c, e] }),
        json!([["p", ["values"], 1, 1, []], ["p", ["values"], 2, 1, []]]),
    );
}

#[test]
fn diff_encodes_removals_canonically() {
    for (before, after, expected) in [
        (
            json!([1, 2, 3, 4]),
            json!([2, 3, 4]),
            json!([["p", ["values"], 0, 1, []]]),
        ),
        (
            json!([1, 2, 3, 4]),
            json!([1, 2, 3]),
            json!([["p", ["values"], 3, 1, []]]),
        ),
        (
            json!([1, 2, 3, 4]),
            json!([1, 3, 4]),
            json!([["p", ["values"], 1, 1, []]]),
        ),
        (
            json!([1, 2, 3, 4]),
            json!([]),
            json!([["p", ["values"], 0, 4, []]]),
        ),
        (json!([1, 2, 3, 4]), json!([1, 2, 3, 4]), json!([])),
    ] {
        expect_diff(
            json!({ "values": before }),
            json!({ "values": after }),
            expected,
        );
    }
}

#[test]
fn diff_does_not_field_diff_unrelated_shifted_objects_with_common_fields() {
    let rows = |values: [i32; 3]| values.map(|value| json!({ "type": "row", "value": value }));
    expect_diff(
        json!({ "values": rows([1, 2, 3]) }),
        json!({ "values": rows([2, 3, 4]) }),
        json!([["p", ["values"], 0, 1, []], ["p", ["values"], 2, 0, [{ "type": "row", "value": 4 }]]]),
    );
}

#[test]
fn diff_does_not_treat_coincidental_id_or_key_fields_as_structural_identity() {
    let (left, right) = (json!({ "value": "left" }), json!({ "value": "right" }));
    let before = json!([left, { "id": 1, "key": "a", "value": "first" }, { "id": 2, "key": "b", "value": "second" }, right]);
    let replacements = json!([{ "id": 2, "key": "b", "value": "edited-second" }, { "id": 1, "key": "a", "value": "edited-first" }]);
    let after = json!([left, replacements[0], replacements[1], right]);
    expect_diff(
        json!({ "values": before }),
        json!({ "values": after }),
        json!([["p", ["values"], 1, 2, replacements]]),
    );
}

/// Pi relies on `changedC.stable === c.stable` (reference identity) here. With
/// deep equality every `stable: {}` aligns, so the op shape differs (see the
/// divergence note in `diff.rs`); the result still converges.
#[test]
fn diff_combines_removals_append_and_a_survivor_edit_convergently() {
    let row = |id: &str, text: &str| json!({ "id": id, "stable": {}, "detail": { "text": text } });
    let (a, b, c, d) = (row("a", "a"), row("b", "b"), row("c", "c"), row("d", "d"));
    let changed_c = row("c", "changed");
    let appended = row("e", "e");
    let before = json!({ "values": [a, b, c, d] });
    let after = json!({ "values": [a, changed_c, d, appended] });
    let operations = diff_revisions(&before, &after);
    assert_eq!(apply_immutable(&before, &operations).unwrap(), after);

    // With distinct stable markers, deep equality reproduces Pi's exact shape.
    let row = |id: &str, text: &str| json!({ "id": id, "stable": { "of": id }, "detail": { "text": text } });
    let (a, b, c, d) = (row("a", "a"), row("b", "b"), row("c", "c"), row("d", "d"));
    let changed_c = row("c", "changed");
    let appended = row("e", "e");
    expect_diff(
        json!({ "values": [a, b, c, d] }),
        json!({ "values": [a, changed_c, d, appended] }),
        json!([
            ["p", ["values"], 1, 1, []],
            ["a", ["values", 1, "detail", "text"], "hanged"],
            ["p", ["values"], 3, 0, [appended]],
        ]),
    );
}

#[test]
fn diff_does_not_retain_a_removed_neighbor_payload() {
    let payload = "x".repeat(256 * 1024);
    let retained: Vec<Value> = ["a", "b", "c", "d", "e"]
        .map(|id| json!({ "id": id, "payload": payload }))
        .to_vec();
    let before = json!({ "values": retained });
    let after = json!({ "values": [retained[0], retained[2], retained[4]] });
    let operations = diff_revisions(&before, &after);
    assert_eq!(
        json_of(&operations),
        json!([["p", ["values"], 1, 1, []], ["p", ["values"], 2, 1, []]])
    );
    assert!(serde_json::to_string(&operations).unwrap().len() < 100);
    assert_eq!(apply_immutable(&before, &operations).unwrap(), after);
}

#[test]
fn diff_keeps_push_pop_and_middle_removal_narrow() {
    for size in [1_001usize, 10_000] {
        let values: Vec<Value> = (0..size).map(|value| json!({ "value": value })).collect();
        let appended = json!({ "value": size });
        let mut pushed = values.clone();
        pushed.push(appended.clone());
        expect_diff(
            json!({ "values": values }),
            json!({ "values": pushed }),
            json!([["p", ["values"], size, 0, [appended]]]),
        );
        expect_diff(
            json!({ "values": values }),
            json!({ "values": values[..size - 1] }),
            json!([["p", ["values"], size - 1, 1, []]]),
        );
        let middle = size / 2;
        let mut removed = values.clone();
        removed.remove(middle);
        expect_diff(
            json!({ "values": values }),
            json!({ "values": removed }),
            json!([["p", ["values"], middle, 1, []]]),
        );
    }
}

#[test]
fn diff_keeps_forty_thousand_row_sparse_edits_narrow() {
    let values: Vec<Value> = (0..40_000)
        .map(|value| json!({ "value": value, "stable": { "value": value } }))
        .collect();
    let mut after = values.clone();
    let mut expected = Vec::new();
    for index in (100..after.len()).step_by(400) {
        after[index] = json!({ "value": -(index as i64), "stable": values[index]["stable"] });
        expected.push(json!(["s", ["values", index, "value"], -(index as i64)]));
    }
    let before = json!({ "values": values });
    let after = json!({ "values": after });
    let operations = diff_revisions(&before, &after);
    assert_eq!(json_of(&operations), Value::Array(expected));
    assert!(serde_json::to_string(&operations).unwrap().len() < 7_500);
    assert_eq!(apply_immutable(&before, &operations).unwrap(), after);
}

#[test]
fn diff_keeps_a_reconstructed_large_array_leaf_edit_narrow() {
    let before: Vec<Value> = (0..1_000)
        .map(|value| json!({ "value": value, "label": format!("row-{value}") }))
        .collect();
    let mut after = before.clone();
    after[700]["label"] = json!("changed");
    let operations = diff_revisions(&json!({ "values": before }), &json!({ "values": after }));
    assert_eq!(
        json_of(&operations),
        json!([["s", ["values", 700, "label"], "changed"]])
    );
}

#[test]
fn diff_splices_an_ambiguous_equal_count_moved_and_edited_gap() {
    let (left, right) = (json!({ "value": "left" }), json!({ "value": "right" }));
    let replacements = json!([{ "id": 2, "value": "edited" }, { "id": 1, "value": "also-edited" }]);
    expect_diff(
        json!({ "values": [left, { "id": 1, "value": "a" }, { "id": 2, "value": "b" }, right] }),
        json!({ "values": [left, replacements[0], replacements[1], right] }),
        json!([["p", ["values"], 1, 2, replacements]]),
    );
}

#[test]
fn diff_keeps_a_large_rotation_payload_free() {
    let values: Vec<Value> = (0..10_000).map(|value| json!({ "value": value })).collect();
    let mut rotated = values[1_000..].to_vec();
    rotated.extend_from_slice(&values[..1_000]);
    let before = json!({ "values": values });
    let after = json!({ "values": rotated });
    let operations = diff_revisions(&before, &after);
    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].verb(), "m");
    assert!(serde_json::to_string(&operations).unwrap().len() < 60_000);
    assert_eq!(apply_immutable(&before, &operations).unwrap(), after);
}

#[test]
fn diff_encodes_five_hundred_unshifts_without_snapshotting_retained_rows() {
    let retained: Vec<Value> = (0..10_000)
        .map(|value| json!({ "value": value, "payload": "x".repeat(100) }))
        .collect();
    let inserted: Vec<Value> = (0..500)
        .map(|value| json!({ "value": -value - 1 }))
        .collect();
    let mut after = inserted.clone();
    after.extend(retained.iter().cloned());
    let operations = diff_revisions(&json!({ "values": retained }), &json!({ "values": after }));
    assert_eq!(
        json_of(&operations),
        json!([["p", ["values"], 0, 0, inserted]])
    );
}

#[test]
fn diff_bounds_wide_object_operation_emission_with_a_root_replacement() {
    let mut before = serde_json::Map::new();
    let mut after = serde_json::Map::new();
    for index in 0..20_000 {
        before.insert(format!("field{index}"), json!(0));
        after.insert(format!("field{index}"), json!(1));
    }
    let after = Value::Object(after);
    assert_eq!(
        diff_revisions(&Value::Object(before), &after),
        vec![Op::R(after.clone())]
    );
}

#[test]
fn diff_bounds_a_wide_normalized_array_fallback_with_a_root_replacement() {
    let before = json!({ "values": vec![0; 40_000] });
    let after = json!({ "values": vec![1; 40_000] });
    let operations = diff_revisions(&before, &after);
    assert_eq!(operations, vec![Op::R(after.clone())]);
    assert_eq!(apply_immutable(&before, &operations).unwrap(), after);
}

#[test]
fn diff_folds_reserved_keys_into_a_set_of_the_nearest_safe_ancestor() {
    let before: Value = serde_json::from_str(r#"{"safe":{"__proto__":{"a":1}}}"#).unwrap();
    let after: Value = serde_json::from_str(r#"{"safe":{"__proto__":{"a":2}}}"#).unwrap();
    let operations = diff_revisions(&before, &after);
    assert_eq!(
        operations,
        vec![Op::S(path!["safe"], after["safe"].clone())]
    );
}

// ─── delta-apply-immutable.test.ts ───────────────────────────────────────────

#[test]
fn copies_touched_containers_while_preserving_input_and_payloads() {
    let shared = json!({ "nested": { "value": 1 } });
    let untouched = json!({ "value": 9 });
    let row_payload = json!({ "id": 4, "label": "placed" });
    let base = json!({
        "text": "abcdef",
        "stable": { "value": 7 },
        "branch": { "value": 1 },
        "copy": null,
        "placed": null,
        "untouched": null,
        "left": null,
        "right": null,
        "meta": { "count": 0, "obsolete": true },
        "rows": [{ "id": 1, "label": "one" }, { "id": 2, "label": "two" }, { "id": 3, "label": "three" }],
    });
    let operations = ops(json!([
        ["t", ["text"], 2],
        ["a", ["text"], "!"],
        ["s", ["meta", "count"], 1],
        ["s", ["meta", "count"], 2],
        ["d", ["meta", "obsolete"]],
        ["s", ["copy"], base["branch"]],
        ["s", ["copy", "value"], 2],
        ["s", ["placed"], shared],
        ["s", ["placed", "nested", "value"], 2],
        ["s", ["untouched"], untouched],
        ["s", ["left"], shared],
        ["s", ["right"], shared],
        ["s", ["left", "nested", "value"], 3],
        ["p", ["rows"], 1, 1, [row_payload]],
        ["s", ["rows", 1, "label"], "edited"],
        ["m", ["rows"], [1, 0, 2]],
        ["s", ["rows", 0, "label"], "moved"],
    ]));
    let snapshot = base.clone();
    let result = apply_immutable(&base, &operations).unwrap();
    let mutable_result = apply(base.clone(), &operations).unwrap();
    assert_eq!(result, mutable_result);
    assert_eq!(result["text"], json!("cdef!"));
    assert_eq!(result["copy"], json!({ "value": 2 }));
    assert_eq!(result["placed"], json!({ "nested": { "value": 2 } }));
    assert_eq!(result["left"], json!({ "nested": { "value": 3 } }));
    assert_eq!(result["right"], shared);
    assert_eq!(result["rows"][0], json!({ "id": 4, "label": "moved" }));
    assert_eq!(result["meta"], json!({ "count": 2 }));
    assert_eq!(base, snapshot);
}

#[test]
fn protects_root_replacement_payloads_before_later_edits() {
    let replacement = json!({ "nested": { "value": 1 }, "values": [1, 2, 3] });
    let batches = [
        ops(json!([["r", replacement]])),
        ops(json!([["s", ["nested", "value"], 2]])),
        ops(json!([
            ["p", ["values"], 1, 1, [4, 5]],
            ["m", ["values"], [3, 0, 1, 2]]
        ])),
    ];
    let result = apply_immutable_batches(&Value::Null, batches.iter().map(Vec::as_slice)).unwrap();
    assert_eq!(
        result,
        json!({ "nested": { "value": 2 }, "values": [3, 1, 4, 5] })
    );

    let array = json!([1, 2, 3]);
    let array_result = apply_immutable(
        &array,
        &ops(json!([
            ["p", [], 1, 1, [4, 5]],
            ["m", [], [3, 0, 1, 2]],
            ["d", [1]]
        ])),
    )
    .unwrap();
    assert_eq!(array_result, json!([3, 4, 5]));
    assert_eq!(array, json!([1, 2, 3]));
}

#[test]
fn shares_one_copy_on_write_scope_across_batch_partitions() {
    let base = json!({
        "text": "abcdef",
        "meta": { "count": 0 },
        "values": [{ "id": 1, "value": 1 }, { "id": 2, "value": 2 }, { "id": 3, "value": 3 }],
    });
    let batches = [
        ops(
            json!([["s", ["meta", "count"], 1], ["p", ["values"], 1, 1, [{ "id": 4, "value": 4 }]]]),
        ),
        Vec::new(),
        ops(json!([
            ["m", ["values"], [2, 0, 1]],
            ["s", ["values", 2, "value"], 40]
        ])),
        ops(json!([["t", ["text"], 2], ["a", ["text"], "!"]])),
    ];
    let intermediate = apply_immutable(&base, &batches[0]).unwrap();
    let snapshot = intermediate.clone();
    let mut sequential = intermediate.clone();
    for batch in &batches[1..] {
        sequential = apply_immutable(&sequential, batch).unwrap();
    }
    let streamed = apply_immutable_batches(&base, batches.iter().map(Vec::as_slice)).unwrap();
    let flattened = apply_immutable(&base, &batches.concat()).unwrap();
    assert_eq!(streamed, sequential);
    assert_eq!(streamed, flattened);
    assert_eq!(intermediate, snapshot);
    assert_eq!(base["meta"]["count"], json!(0));
}

#[test]
fn does_not_expose_partial_application_when_validation_fails() {
    let base = json!({ "nested": { "value": 1 } });
    let invalid = [
        vec![Op::S(path!["nested", "value"], json!(2))],
        vec![Op::S(
            path!["constructor", "prototype", "polluted"],
            json!(true),
        )],
    ];
    assert!(apply_immutable_batches(&base, invalid.iter().map(Vec::as_slice)).is_err());
    assert_eq!(base["nested"]["value"], json!(1));

    let error = apply_immutable(
        &json!({ "values": [] }),
        &ops(json!([["s", ["values", "missing", "value"], 1]])),
    )
    .unwrap_err();
    assert!(error.is_unsafe_path_error() || error.is_path_error());
    let error = apply_immutable(
        &json!({ "values": {} }),
        &ops(json!([["s", ["values", "missing", "value"], 1]])),
    )
    .unwrap_err();
    assert!(error.is_path_error());
    let error = apply_immutable(
        &json!({ "values": [{}] }),
        &ops(json!([["s", ["values", "0", "value"], 1]])),
    )
    .unwrap_err();
    assert!(error.is_unsafe_path_error());
}

#[test]
fn copies_a_wide_object_and_leaves_the_base_untouched() {
    let mut base = serde_json::Map::new();
    for index in 0..20_000 {
        base.insert(format!("field{index}"), json!(index));
    }
    let base = Value::Object(base);
    let operations: Vec<Op> = (0..1_000)
        .map(|index| Op::S(vec![Seg::Key(format!("field{index}"))], json!(-index)))
        .collect();
    let result = apply_immutable(&base, &operations).unwrap();
    assert_eq!(result["field999"], json!(-999));
    assert_eq!(base["field999"], json!(999));
}

// ─── delta-clone.test.ts ─────────────────────────────────────────────────────

#[test]
fn takes_ownership_of_the_imported_revision() {
    let input = Arc::new(
        json!({ "point": { "x": 3, "y": 7, "pressure": 0.1 }, "rows": [{ "values": [0, false, null, "text", { "n": 1 }] }] }),
    );
    let tracker = track(input.clone());
    assert!(Arc::ptr_eq(tracker.value(), &input));
}

#[test]
fn copies_assigned_and_inserted_values_immediately() {
    let mut tracker = track(json!({ "rows": [] }));
    let mut assigned = json!({ "nested": { "value": 1 } });
    let mut change = tracker.begin_change();
    change.state_mut().unwrap()["rows"]
        .as_array_mut()
        .unwrap()
        .push(assigned.clone());
    assigned["nested"]["value"] = json!(9);
    assert_eq!(
        change.state().unwrap()["rows"][0]["nested"]["value"],
        json!(1)
    );
    let prepared = change.prepare().unwrap();
    assert_eq!(prepared.value()["rows"][0]["nested"]["value"], json!(1));
    let replica = apply((**prepared.base()).clone(), prepared.ops()).unwrap();
    assert_eq!(replica, **prepared.value());
}

// ─── Golden fixtures generated from the TS sources ───────────────────────────

fn fixtures() -> Value {
    serde_json::from_str(include_str!("../fixtures/delta.json")).unwrap()
}

/// Every op serializes to exactly `JSON.stringify(op)` and parses back.
#[test]
fn golden_op_json_matches_typescript() {
    let fixtures = fixtures();
    let cases = fixtures["ops"].as_array().unwrap();
    let constructed = vec![
        Op::R(json!({ "value": 1, "nested": { "list": [1, "two", null, true] } })),
        Op::R(Value::Null),
        Op::R(json!([1, 2, 3])),
        Op::S(path!["count"], json!(2)),
        Op::S(path!["values", 0usize, "label"], json!("changed")),
        Op::S(path!["a\u{0000}b", "é", "😀"], json!({ "x": [{}] })),
        Op::D(path!["remove"]),
        Op::D(path!["values", 3usize]),
        Op::A(path!["text"], " world".into()),
        Op::A(path!["nested", "text"], "émoji 😀 \"quoted\"\n".into()),
        Op::T(path!["text"], 6),
        Op::T(path!["rows", 2usize, "output"], 0),
        Op::P(path!["values"], 1, 0, vec![json!({ "id": "c" })]),
        Op::P(Vec::new(), 0, 4, Vec::new()),
        Op::P(
            path!["values"],
            2,
            1,
            vec![Value::Null, json!(-1.5), json!(0.1), json!(12345678)],
        ),
        Op::M(path!["values"], vec![2, 0, 1]),
        Op::M(Vec::new(), vec![0]),
    ];
    assert_eq!(cases.len(), constructed.len());
    for (case, op) in cases.iter().zip(&constructed) {
        let expected = case["json"].as_str().unwrap();
        assert_eq!(serde_json::to_string(op).unwrap(), expected);
        let parsed: Op = serde_json::from_str(expected).unwrap();
        assert_eq!(&parsed, op);
        assert_eq!(serde_json::to_string(&parsed).unwrap(), expected);
        assert_eq!(&op.to_json(), &case["op"]);
    }
}

/// The encoder produces exactly TS's wire tuples across batches, and they decode back.
#[test]
fn golden_wire_encoding_matches_typescript() {
    let fixtures = fixtures();
    let mut enc = encoder();
    let mut dec = decoder();
    for case in fixtures["wire"].as_array().unwrap() {
        let batch = ops(case["ops"].clone());
        let encoded = enc.encode(&batch);
        assert_eq!(serde_json::to_value(&encoded).unwrap(), case["wire"]);
        assert_eq!(
            serde_json::to_string(&encoded).unwrap(),
            serde_json::to_string(&case["wire"]).unwrap()
        );
        let parsed: Vec<WireOp> = serde_json::from_value(case["wire"].clone()).unwrap();
        assert_eq!(dec.decode(&parsed).unwrap(), batch);
    }
}

/// `diff_revisions` reproduces TS's ops for JSON-parsed revisions (no shared
/// containers), and TS ops apply to the same result in Rust.
#[test]
fn golden_diff_revisions_matches_typescript() {
    let fixtures = fixtures();
    let cases = fixtures["diffs"].as_array().unwrap();
    let mut identical = 0;
    for case in cases {
        let (before, after) = (&case["before"], &case["after"]);
        let expected = ops(case["ops"].clone());
        assert_eq!(apply_immutable(before, &expected).unwrap(), *after);
        let actual = diff_revisions(before, after);
        assert_eq!(apply_immutable(before, &actual).unwrap(), *after);
        if actual == expected {
            identical += 1;
        } else {
            eprintln!(
                "diff shape differs:\n  ts:   {}\n  rust: {}",
                case["ops"],
                json_of(&actual)
            );
        }
    }
    // Shapes may only differ where Pi anchors on reference identity between
    // deeply equal distinct containers (see `diff.rs`).
    assert!(
        identical * 100 >= cases.len() * 95,
        "{identical}/{} identical",
        cases.len()
    );
}

type Script = Vec<Value>;

fn run_script(state: &mut Value, steps: &Script) {
    for step in steps {
        let step = step.as_array().unwrap();
        let kind = step[0].as_str().unwrap();
        let path = step[1].as_array().unwrap();
        let (key, parents) = path.split_last().unwrap();
        let mut parent = &mut *state;
        for segment in parents {
            parent = match segment {
                Value::String(key) => &mut parent[key.as_str()],
                other => &mut parent[other.as_u64().unwrap() as usize],
            };
        }
        match (kind, key) {
            ("set", Value::String(key)) => {
                parent
                    .as_object_mut()
                    .unwrap()
                    .insert(key.clone(), step[2].clone());
                continue;
            }
            ("delete", Value::String(key)) => {
                parent.as_object_mut().unwrap().shift_remove(key.as_str());
                continue;
            }
            _ => {}
        }
        let slot = match key {
            Value::String(key) => &mut parent[key.as_str()],
            other => &mut parent[other.as_u64().unwrap() as usize],
        };
        match kind {
            "set" => *slot = step[2].clone(),
            "append" => {
                let text = format!("{}{}", slot.as_str().unwrap(), step[2].as_str().unwrap());
                *slot = Value::String(text);
            }
            "roll" => {
                let current = slot.as_str().unwrap();
                let offset =
                    utf16_byte_offset(current, step[2].as_u64().unwrap() as usize).unwrap();
                *slot = Value::String(format!(
                    "{}{}",
                    &current[offset..],
                    step[3].as_str().unwrap()
                ));
            }
            "push" => slot.as_array_mut().unwrap().push(step[2].clone()),
            "unshift" => slot.as_array_mut().unwrap().insert(0, step[2].clone()),
            "shift" => {
                slot.as_array_mut().unwrap().remove(0);
            }
            "pop" => {
                slot.as_array_mut().unwrap().pop();
            }
            "length" => slot
                .as_array_mut()
                .unwrap()
                .resize(step[2].as_u64().unwrap() as usize, Value::Null),
            other => panic!("unknown step {other}"),
        }
    }
}

/// For the mutations durable performs (set, delete, string append and
/// rolling windows, push/pop/shift/unshift, length growth), the diff-based
/// tracker emits exactly the ops of Pi's proxy tracker.
#[test]
fn golden_tracker_emission_matches_typescript_for_simple_mutations() {
    let fixtures = fixtures();
    for case in fixtures["tracker"].as_array().unwrap() {
        let mut tracker = track(case["initial"].clone());
        let mut change = tracker.begin_change();
        run_script(
            change.state_mut().unwrap(),
            case["steps"].as_array().unwrap(),
        );
        let prepared = change.prepare().unwrap();
        assert_eq!(
            json_of(prepared.ops()),
            case["ops"],
            "steps: {}",
            case["steps"]
        );
        assert_eq!(**prepared.value(), case["value"]);
        tracker.adopt(&prepared).unwrap();
    }
}
