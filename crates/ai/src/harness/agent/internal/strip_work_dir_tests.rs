//! Port of `strip-work-dir.test.ts`.

use super::*;
use serde_json::json;

const WORK_DIR: &str = "/vercel/sandbox/claude-code-abc123";

fn tool_call(input: &str) -> HarnessV1StreamPart {
    HarnessV1StreamPart::ToolCall {
        tool_call_id: "c1".into(),
        tool_name: "readFile".into(),
        input: input.into(),
        provider_executed: None,
        dynamic: None,
        provider_metadata: None,
        native_name: None,
    }
}

fn tool_result(result: JsonValue) -> HarnessV1StreamPart {
    HarnessV1StreamPart::ToolResult {
        tool_call_id: "c1".into(),
        tool_name: "bash".into(),
        result,
        is_error: None,
        preliminary: None,
        dynamic: None,
        provider_metadata: None,
    }
}

fn file_change(path: &str) -> HarnessV1StreamPart {
    HarnessV1StreamPart::FileChange {
        event: crate::harness::spec::stream_part::HarnessV1FileChangeEvent::Modify,
        path: path.into(),
        harness_metadata: None,
    }
}

fn input_of(part: &HarnessV1StreamPart) -> String {
    match part {
        HarnessV1StreamPart::ToolCall { input, .. } => input.clone(),
        _ => panic!("not a tool call"),
    }
}

fn result_of(part: &HarnessV1StreamPart) -> JsonValue {
    match part {
        HarnessV1StreamPart::ToolResult { result, .. } => result.clone(),
        _ => panic!("not a tool result"),
    }
}

fn path_of(part: &HarnessV1StreamPart) -> String {
    match part {
        HarnessV1StreamPart::FileChange { path, .. } => path.clone(),
        _ => panic!("not a file change"),
    }
}

fn start() -> HarnessV1StreamPart {
    HarnessV1StreamPart::ToolInputStart {
        id: "c1".into(),
        tool_name: "bash".into(),
        provider_metadata: None,
        provider_executed: None,
        dynamic: None,
        title: None,
    }
}

fn delta(d: &str) -> HarnessV1StreamPart {
    HarnessV1StreamPart::ToolInputDelta {
        id: "c1".into(),
        delta: d.into(),
        provider_metadata: None,
    }
}

fn end() -> HarnessV1StreamPart {
    HarnessV1StreamPart::ToolInputEnd {
        id: "c1".into(),
        provider_metadata: None,
    }
}

fn streamed(work_dir: &str, deltas: &[&str]) -> String {
    let mut strip = create_tool_input_work_dir_stripper(work_dir);
    let mut parts = vec![start()];
    parts.extend(deltas.iter().map(|d| delta(d)));
    parts.push(end());
    parts
        .into_iter()
        .flat_map(&mut strip)
        .filter_map(|p| match p {
            HarnessV1StreamPart::ToolInputDelta { delta, .. } => Some(delta),
            _ => None,
        })
        .collect()
}

#[test]
fn strips_the_prefix_from_a_tool_call_input_string() {
    let input = json!({ "path": format!("{WORK_DIR}/src/foo.ts") }).to_string();
    let out = strip_work_dir(&tool_call(&input), WORK_DIR);
    assert_eq!(out, tool_call(&json!({ "path": "src/foo.ts" }).to_string()));
}

#[test]
fn strips_the_prefix_from_a_tool_input_delta() {
    let d = json!({ "path": format!("{WORK_DIR}/src/foo.ts") }).to_string();
    let out = strip_work_dir(&delta(&d), WORK_DIR);
    assert_eq!(out, delta(&json!({ "path": "src/foo.ts" }).to_string()));
}

#[test]
fn strips_every_occurrence_in_free_form_tool_result_string_output() {
    let out = strip_work_dir(
        &tool_result(json!(format!("{WORK_DIR}/a.ts\n{WORK_DIR}/b.ts\n"))),
        WORK_DIR,
    );
    assert_eq!(result_of(&out), json!("a.ts\nb.ts\n"));
}

#[test]
fn strips_paths_nested_in_objects_and_arrays_of_a_tool_result() {
    let out = strip_work_dir(
        &tool_result(json!({
            "matches": [
                { "file": format!("{WORK_DIR}/src/a.ts"), "line": 1 },
                { "file": format!("{WORK_DIR}/src/b.ts"), "line": 2 }
            ],
            "cwd": WORK_DIR
        })),
        WORK_DIR,
    );
    assert_eq!(
        result_of(&out),
        json!({
            "matches": [{ "file": "src/a.ts", "line": 1 }, { "file": "src/b.ts", "line": 2 }],
            "cwd": "."
        })
    );
}

#[test]
fn maps_a_bare_reference_to_the_work_dir_to_dot() {
    assert_eq!(
        path_of(&strip_work_dir(&file_change(WORK_DIR), WORK_DIR)),
        "."
    );
}

#[test]
fn maps_a_bare_work_dir_followed_by_a_delimiter_to_dot_in_a_complete_value() {
    for (input, expected) in [
        ("cd /work && pwd", "cd . && pwd"),
        ("{\"cwd\":\"/work\"}", "{\"cwd\":\".\"}"),
        ("cd /work;pwd", "cd .;pwd"),
    ] {
        assert_eq!(
            input_of(&strip_work_dir(&tool_call(input), "/work")),
            expected
        );
    }
}

#[test]
fn strips_the_prefix_from_a_file_change_path() {
    assert_eq!(
        path_of(&strip_work_dir(
            &file_change(&format!("{WORK_DIR}/notes.md")),
            WORK_DIR
        )),
        "notes.md"
    );
}

#[test]
fn preserves_the_work_dir_text_inside_longer_path_segments() {
    let out = strip_work_dir(
        &tool_result(json!({
            "command": "ls .github/workflows && echo origin/workcell && cat /work/a.ts",
            "path": "/work/.github/workflows/ci.yml",
            "branch": "origin/workcell/topic"
        })),
        "/work",
    );
    assert_eq!(
        result_of(&out),
        json!({
            "command": "ls .github/workflows && echo origin/workcell && cat a.ts",
            "path": ".github/workflows/ci.yml",
            "branch": "origin/workcell/topic"
        })
    );
}

#[test]
fn preserves_repeated_work_dir_text_after_an_embedded_occurrence() {
    let value = "origin/work/work/foo";
    let streamed = streamed("/work", &[value]);
    assert_eq!(
        input_of(&strip_work_dir(&tool_call(value), "/work")),
        streamed
    );
    assert_eq!(
        result_of(&strip_work_dir(&tool_result(json!(value)), "/work")),
        json!(streamed)
    );
    assert_eq!(
        path_of(&strip_work_dir(&file_change(value), "/work")),
        streamed
    );
    assert_eq!(streamed, value);
}

#[test]
fn maps_a_streamed_bare_work_dir_followed_by_a_delimiter_to_dot() {
    for (deltas, expected) in [
        (vec!["cd /work", " && pwd"], "cd . && pwd"),
        (vec!["{\"cwd\":\"/work", "\"}"], "{\"cwd\":\".\"}"),
        (vec!["cd /work", ";pwd"], "cd .;pwd"),
    ] {
        assert_eq!(streamed("/work", &deltas), expected);
    }
}

#[test]
fn preserves_embedded_work_dir_text_split_across_tool_input_deltas() {
    assert_eq!(
        streamed(
            "/work",
            &[
                "{\"command\":\"ls .github/wor",
                "kflows && cat /wo",
                "rk/a.ts\"}"
            ]
        ),
        "{\"command\":\"ls .github/workflows && cat a.ts\"}"
    );
}

#[test]
fn preserves_a_work_dir_prefix_after_a_non_boundary_character_in_a_previous_delta() {
    assert_eq!(streamed("/work", &["foo", "/work/bar"]), "foo/work/bar");
}

#[test]
fn passes_through_variants_with_no_path_bearing_fields_unchanged() {
    let part = HarnessV1StreamPart::TextDelta {
        id: "t1".into(),
        delta: format!("wrote {WORK_DIR}/foo.ts"),
        harness_metadata: None,
    };
    assert_eq!(strip_work_dir(&part, WORK_DIR), part);
}

// stripParsedToolInputWorkDir
// [rust] JSON values cannot hold cycles or shared references; the test keeps
// the part that applies: the input is copied, not modified.
#[test]
fn strips_a_copy_without_modifying_the_validated_input() {
    let shared = json!({ "path": format!("{WORK_DIR}/shared.ts") });
    let source = json!({ "first": shared, "second": shared, "values": [1, [shared]] });
    let display = strip_parsed_tool_input_work_dir(&source, WORK_DIR);
    assert_eq!(display["first"], json!({ "path": "shared.ts" }));
    assert_eq!(display["values"][1][0], json!({ "path": "shared.ts" }));
    assert_eq!(
        source["first"]["path"],
        json!(format!("{WORK_DIR}/shared.ts"))
    );
}

#[test]
fn strips_deeply_nested_paths_without_overflowing_the_stack() {
    // A large stack only for building and dropping the 12k-deep value;
    // stripping itself is iterative.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            let depth = 12_000;
            let mut source = json!({ "path": format!("{WORK_DIR}/deep.ts") });
            for _ in 0..depth {
                let mut map = serde_json::Map::new();
                map.insert("child".into(), source);
                source = JsonValue::Object(map);
            }
            let display = strip_parsed_tool_input_work_dir(&source, WORK_DIR);
            let mut projected = &display;
            let mut original = &source;
            for _ in 0..depth {
                projected = &projected["child"];
                original = &original["child"];
            }
            assert_eq!(projected, &json!({ "path": "deep.ts" }));
            assert_eq!(original["path"], json!(format!("{WORK_DIR}/deep.ts")));
        })
        .unwrap()
        .join()
        .unwrap();
}
