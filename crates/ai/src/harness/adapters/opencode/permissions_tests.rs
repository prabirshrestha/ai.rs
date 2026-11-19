//! Port of `permission-paths.test.ts`, `question-tool.test.ts` and
//! `tool-relay-auth.test.ts`.

use std::os::unix::fs::symlink;

use serde_json::{Value, json};

use super::*;
use crate::harness::adapters::local_sandbox::{LocalSandbox, canonical};
use crate::harness::adapters::test_temp_dir::{TempDir, mkdtemp};

struct Fixture {
    _temp: TempDir,
    work_dir: String,
    skills: String,
    sandbox: LocalSandbox,
}

fn fixture() -> Fixture {
    let temp = mkdtemp("opencode-permission-paths-");
    let directory = canonical(temp.path());
    let work_dir = format!("{directory}/workspace");
    let skills = format!("{directory}/skill");
    for dir in [
        &work_dir,
        &format!("{work_dir}/nested"),
        &format!("{directory}/other"),
        &skills,
    ] {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(format!("{work_dir}/file.txt"), "workspace fixture").unwrap();
    std::fs::write(format!("{directory}/other/file.txt"), "other fixture").unwrap();
    symlink(format!("{directory}/other"), format!("{work_dir}/external")).unwrap();
    symlink(format!("{work_dir}/nested"), format!("{work_dir}/internal")).unwrap();
    symlink(
        format!("{directory}/missing"),
        format!("{work_dir}/dangling"),
    )
    .unwrap();
    Fixture {
        sandbox: LocalSandbox::new(work_dir.clone()),
        _temp: temp,
        work_dir,
        skills,
    }
}

fn call(native_name: &str, input: Value) -> PermissionToolCall {
    PermissionToolCall {
        native_name: native_name.to_string(),
        input: input.as_object().cloned().unwrap(),
    }
}

struct Case {
    action: &'static str,
    tool_call: Option<PermissionToolCall>,
    resources: Vec<String>,
    metadata: Option<Value>,
    expected: bool,
}

fn case(action: &'static str, expected: bool) -> Case {
    Case {
        action,
        tool_call: None,
        resources: Vec::new(),
        metadata: None,
        expected,
    }
}

impl Case {
    fn call(mut self, native_name: &str, input: Value) -> Self {
        self.tool_call = Some(call(native_name, input));
        self
    }

    fn resources(mut self, resources: &[&str]) -> Self {
        self.resources = resources.iter().map(|value| value.to_string()).collect();
        self
    }

    fn metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

fn options<'a>(
    fixture: &'a Fixture,
    case: &'a Case,
    skills: &'a [String],
) -> PermissionPathOptions<'a> {
    PermissionPathOptions {
        action: case.action,
        resources: &case.resources,
        metadata: case.metadata.as_ref().and_then(Value::as_object),
        tool_call: case.tool_call.as_ref(),
        work_dir: &fixture.work_dir,
        readable_skill_roots: skills,
    }
}

#[tokio::test]
async fn checks_file_permission_targets() {
    let fixture = fixture();
    let skills = vec![fixture.skills.clone()];
    let cases = vec![
        case("read", true).call("read", json!({ "filePath": "file.txt" })),
        case("read", false).call("read", json!({ "filePath": "../other/file.txt" })),
        case("read", false).call("read", json!({})),
        case("read", false).call("read", json!({ "filePath": "external/file.txt" })),
        case("read", true).call("read", json!({ "filePath": "internal" })),
        case("glob", true).call("glob", json!({ "pattern": "**/*.ts" })),
        case("grep", false).call("grep", json!({ "pattern": "text", "path": "../other" })),
        case("external_directory", false).call("write", json!({ "filePath": "../skill/new.txt" })),
        case("read", true).call(
            "read",
            json!({ "filePath": format!("{}/SKILL.md", fixture.skills) }),
        ),
        case("read", false).resources(&["../other/file.txt"]),
        case("glob", false).resources(&["**/*.ts"]),
        case("edit", false).metadata(json!({
            "files": [{ "filePath": "file.txt" }, { "filePath": "../other/file.txt" }],
        })),
        case("edit", false).metadata(json!({
            "files": [{ "filePath": "file.txt", "type": "move", "movePath": "../other/new.txt" }],
        })),
        case("edit", false)
            .metadata(json!({ "files": [{ "filePath": "file.txt", "type": "move" }] })),
        case("edit", true).metadata(json!({
            "files": [{ "filePath": "file.txt", "type": "move", "movePath": "new.txt" }],
        })),
        case("edit", false).metadata(json!({ "files": [] })),
    ];
    for (index, case) in cases.iter().enumerate() {
        let trusted =
            are_permission_paths_trusted(&fixture.sandbox, &options(&fixture, case, &skills), None)
                .await;
        assert_eq!(trusted, case.expected, "case {index} ({})", case.action);
    }
}

#[tokio::test]
async fn skill_roots_are_readable_but_not_writable() {
    let fixture = fixture();
    let skills = vec![fixture.skills.clone()];
    let path = format!("{}/SKILL.md", fixture.skills);
    let write = case("edit", false).call("write", json!({ "filePath": path }));
    assert!(
        !are_permission_paths_trusted(&fixture.sandbox, &options(&fixture, &write, &skills), None)
            .await
    );
}

#[tokio::test]
async fn reports_write_protected_targets() {
    let fixture = fixture();
    let skills = vec![fixture.skills.clone()];
    let mut cases =
        vec![case("edit", false).metadata(json!({ "files": [{ "filePath": "file.txt" }] }))];
    for path in [
        ".agents/skills/test/SKILL.md",
        ".opencode/plugin/index.ts",
        "opencode.json",
        "opencode.jsonc",
        ".claude/skills/test/SKILL.md",
    ] {
        cases.push(case("edit", true).metadata(json!({ "files": [{ "filePath": path }] })));
    }
    cases.extend([
        case("edit", false).metadata(json!({ "files": [{ "filePath": ".claude/settings.json" }] })),
        case("edit", true).metadata(json!({
            "files": [{ "filePath": "file.txt", "type": "move", "movePath": ".agents/skills/moved/SKILL.md" }],
        })),
        case("edit", true).call("write", json!({ "filePath": "opencode.json" })),
        case("read", false).call("read", json!({ "filePath": "opencode.json" })),
        case("edit", true).metadata(json!({ "files": [] })),
    ]);
    for (index, case) in cases.iter().enumerate() {
        let protected = are_permission_paths_write_protected(
            &fixture.sandbox,
            &options(&fixture, case, &skills),
            None,
        )
        .await;
        assert_eq!(protected, case.expected, "case {index}");
    }
}

#[tokio::test]
async fn selects_permission_replies() {
    let fixture = fixture();
    let read = call("read", json!({ "filePath": "file.txt" }));
    let outside = call("read", json!({ "filePath": "../other/file.txt" }));
    let request =
        |action: &'static str, tool_call: Option<&'static PermissionToolCall>| PermissionRequest {
            action,
            resources: &[],
            metadata: None,
            tool_call,
        };
    let read: &'static PermissionToolCall = Box::leak(Box::new(read));
    let outside: &'static PermissionToolCall = Box::leak(Box::new(outside));
    let select = |request: PermissionRequest<'static>,
                  mode: PermissionMode,
                  inactive: Vec<String>| {
        let sandbox = fixture.sandbox.clone();
        let work_dir = fixture.work_dir.clone();
        async move {
            select_permission_reply(&sandbox, &request, mode, &inactive, &work_dir, &[], None).await
        }
    };
    let once = PermissionDecision::Reply(PermissionReply::Once);
    assert_eq!(
        select(
            request("read", Some(read)),
            PermissionMode::AllowAll,
            vec![]
        )
        .await,
        PermissionDecision::Reply(PermissionReply::Always)
    );
    assert_eq!(
        select(
            request("read", Some(read)),
            PermissionMode::AllowReads,
            vec![]
        )
        .await,
        once
    );
    assert_eq!(
        select(
            request("read", Some(outside)),
            PermissionMode::AllowReads,
            vec![]
        )
        .await,
        PermissionDecision::Ask
    );
    assert_eq!(
        select(request("bash", None), PermissionMode::AllowEdits, vec![]).await,
        PermissionDecision::Ask
    );
    assert_eq!(
        select(
            request("webfetch", None),
            PermissionMode::AllowReads,
            vec![]
        )
        .await,
        once
    );
    assert_eq!(
        select(request("edit", None), PermissionMode::AllowReads, vec![]).await,
        PermissionDecision::Ask
    );
    assert_eq!(
        select(
            request("read", Some(read)),
            PermissionMode::AllowAll,
            vec!["read".into()]
        )
        .await,
        PermissionDecision::Ask
    );
}

#[test]
fn maps_permission_tool_names() {
    for (action, expected) in [
        ("bash", "bash"),
        ("edit", "edit"),
        ("external_directory", "external_directory"),
        ("task", "agent"),
        ("list", "ls"),
        ("webfetch", "webfetch"),
        ("question", "askUserQuestions"),
    ] {
        assert_eq!(to_permission_tool_name(action), expected, "{action}");
    }
}

// describe('OpenCode question translation')

fn native_questions() -> Vec<Value> {
    vec![
        json!({
            "question": "Pick a color",
            "header": "Color",
            "options": [
                { "label": "Red", "description": "Warm" },
                { "label": "Blue", "description": "Cool" },
            ],
            "multiple": true,
            "custom": true,
        }),
        json!({
            "question": "Pick a size",
            "header": "Size",
            "options": [{ "label": "Small", "description": "S" }],
        }),
    ]
}

#[test]
fn translates_questions_to_the_harness_input() {
    assert_eq!(
        to_harness_questions_input(&native_questions()),
        json!({
            "allowPartialAnswers": true,
            "questions": [
                {
                    "id": "question-1", "question": "Pick a color", "header": "Color",
                    "options": [
                        { "id": "option-1", "label": "Red", "description": "Warm" },
                        { "id": "option-2", "label": "Blue", "description": "Cool" },
                    ],
                    "allowMultiple": true, "allowFreeForm": true,
                },
                {
                    "id": "question-2", "question": "Pick a size", "header": "Size",
                    "options": [{ "id": "option-1", "label": "Small", "description": "S" }],
                },
            ],
        })
    );
}

#[test]
fn translates_answers_to_the_opencode_response() {
    let output = QuestionsToolOutput::parse(json!({
        "action": "partially-answered",
        "answers": {
            "question-1": { "optionIds": ["option-2", "option-9", "bogus"], "freeform": "Green" },
        },
    }))
    .unwrap();
    assert_eq!(
        to_opencode_question_response(&native_questions(), &output),
        Some(vec![vec!["Blue".to_string(), "Green".to_string()], vec![]])
    );
    for action in ["declined", "cancelled"] {
        let output = QuestionsToolOutput::parse(json!({ "action": action })).unwrap();
        assert_eq!(
            to_opencode_question_response(&native_questions(), &output),
            None
        );
    }
}

// describe('ToolRelayAuthorizer')

#[test]
fn authorizes_matching_relay_calls_once() {
    let now = Instant::now();
    let mut authorizer = ToolRelayAuthorizer::<u32>::new(Duration::from_secs(10));
    let key = tool_relay_call_key("weather", &json!({ "b": 1, "a": { "d": 2, "c": 3 } }));
    assert_eq!(
        key,
        tool_relay_call_key("weather", &json!({ "a": { "c": 3, "d": 2 }, "b": 1 }))
    );
    assert_eq!(
        authorizer.authorize(key.clone(), "call-1".into(), now),
        None
    );
    assert_eq!(authorizer.wait(key.clone(), 1, now), Some("call-1".into()));
    // Consumed.
    assert_eq!(authorizer.wait(key.clone(), 2, now), None);
    assert_eq!(
        authorizer.authorize(key.clone(), "call-2".into(), now),
        Some((2, "call-2".into()))
    );
    assert_ne!(key, tool_relay_call_key("other", &json!({})));
    assert_eq!(
        tool_relay_call_key("x", &Value::Null),
        tool_relay_call_key("x", &json!({}))
    );
}

#[test]
fn expires_relay_authorizations_and_waiters() {
    let now = Instant::now();
    let later = now + Duration::from_secs(11);
    let mut authorizer = ToolRelayAuthorizer::<u32>::new(Duration::from_secs(10));
    authorizer.authorize("a".into(), "call-1".into(), now);
    assert_eq!(authorizer.wait("a".into(), 1, later), None);
    assert_eq!(authorizer.expire(later), Vec::<u32>::new());
    assert_eq!(authorizer.expire(later + Duration::from_secs(11)), vec![1]);
    authorizer.wait("b".into(), 2, later);
    assert_eq!(authorizer.close(), vec![2]);
}
