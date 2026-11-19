use std::sync::Arc;

use parking_lot::Mutex;

use super::*;
use crate::harness::sandbox::SandboxRunResult;
use crate::harness::spec::bootstrap::{HarnessV1BootstrapCommand, HarnessV1BootstrapFile};
use crate::harness::test_support::MockSandbox;

fn base_recipe() -> HarnessV1Bootstrap {
    HarnessV1Bootstrap {
        harness_id: "demo".into(),
        bootstrap_dir: "/tmp/harness/demo".into(),
        files: vec![
            HarnessV1BootstrapFile {
                path: "/tmp/harness/demo/a.txt".into(),
                content: "one".into(),
            },
            HarnessV1BootstrapFile {
                path: "/tmp/harness/demo/b.txt".into(),
                content: "two".into(),
            },
        ],
        commands: vec![
            HarnessV1BootstrapCommand {
                command: "echo first".into(),
            },
            HarnessV1BootstrapCommand {
                command: "echo second".into(),
            },
        ],
    }
}

const STATE_DIRECTORY: &str = "/work";
const IDENTITY: &str = "idtest1234567890";

#[test]
fn produces_a_deterministic_16_char_hex_id_for_the_same_recipe() {
    let a = hash_harness_bootstrap(&base_recipe());
    let b = hash_harness_bootstrap(&base_recipe());
    assert_eq!(a, b);
    assert_eq!(a.len(), 16);
    assert!(
        a.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    );
}

#[test]
fn is_unaffected_by_file_ordering_in_the_recipe() {
    let mut reordered = base_recipe();
    reordered.files.reverse();
    assert_eq!(
        hash_harness_bootstrap(&base_recipe()),
        hash_harness_bootstrap(&reordered)
    );
}

#[test]
fn changes_when_a_file_content_changes() {
    let mut altered = base_recipe();
    altered.files[0].content = "one!".into();
    assert_ne!(
        hash_harness_bootstrap(&altered),
        hash_harness_bootstrap(&base_recipe())
    );
}

#[test]
fn changes_when_a_command_changes() {
    let mut altered = base_recipe();
    altered.commands = vec![HarnessV1BootstrapCommand {
        command: "echo different".into(),
    }];
    assert_ne!(
        hash_harness_bootstrap(&altered),
        hash_harness_bootstrap(&base_recipe())
    );
}

#[test]
fn changes_when_harness_id_changes() {
    let mut altered = base_recipe();
    altered.harness_id = "other".into();
    assert_ne!(
        hash_harness_bootstrap(&altered),
        hash_harness_bootstrap(&base_recipe())
    );
}

#[test]
fn changes_when_bootstrap_dir_changes() {
    let mut altered = base_recipe();
    altered.bootstrap_dir = "/tmp/other".into();
    assert_ne!(
        hash_harness_bootstrap(&altered),
        hash_harness_bootstrap(&base_recipe())
    );
}

#[test]
fn includes_the_harness_state_directory_in_the_recipe_identity() {
    assert_eq!(hash_harness_bootstrap(&base_recipe()), "2b1b95adf78b2953");
}

#[test]
fn marker_embeds_the_identity_in_the_filename_under_bootstrap_dir() {
    assert_eq!(
        bootstrap_marker_path(&base_recipe(), "abc1234567890def", STATE_DIRECTORY),
        "/tmp/harness/demo/.bootstrap-abc1234567890def.ok"
    );
}

#[test]
fn marker_resolves_a_relative_bootstrap_dir_against_the_state_directory() {
    let mut recipe = base_recipe();
    recipe.bootstrap_dir = ".harness-bootstrap/demo".into();
    assert_eq!(
        bootstrap_marker_path(&recipe, "abc1234567890def", STATE_DIRECTORY),
        "/work/.harness-bootstrap/demo/.bootstrap-abc1234567890def.ok"
    );
}

/// One recorded `run`: command, working directory, env, and the number of
/// writes that happened before it.
type RunRecord = (String, Option<String>, Option<Vec<(String, String)>>, usize);

struct Mock {
    session: MockSandbox,
    runs: Arc<Mutex<Vec<RunRecord>>>,
}

fn make_mock_session(
    marker_exists: bool,
    directory_exit_code: i32,
    command_exit_code: i32,
    run_failure_message: &str,
) -> Mock {
    let runs: Arc<Mutex<Vec<RunRecord>>> = Arc::default();
    let recorded = runs.clone();
    let stderr = run_failure_message.to_string();
    let session = MockSandbox::new(move |o, state| {
        recorded.lock().push((
            o.process.command.clone(),
            o.process.working_directory.clone(),
            o.process
                .env
                .as_ref()
                .map(|e| e.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            state.writes.len(),
        ));
        let is_directory_creation = o.process.command == "mkdir -p \"$BOOTSTRAP_DIR\"";
        Ok(SandboxRunResult {
            exit_code: if is_directory_creation {
                directory_exit_code
            } else {
                command_exit_code
            },
            stdout: "ok".into(),
            stderr: stderr.clone(),
            ..Default::default()
        })
    });
    if marker_exists {
        let marker = bootstrap_marker_path(&base_recipe(), IDENTITY, STATE_DIRECTORY);
        session.state.lock().files.insert(marker, String::new());
    }
    Mock { session, runs }
}

#[tokio::test]
async fn skips_when_the_marker_file_is_present() {
    let mock = make_mock_session(true, 0, 0, "");
    apply_bootstrap_recipe(
        &mock.session,
        &base_recipe(),
        IDENTITY,
        STATE_DIRECTORY,
        None,
    )
    .await
    .unwrap();
    assert_eq!(mock.session.text_reads.lock().len(), 1);
    assert!(mock.session.state.lock().writes.is_empty());
    assert!(mock.runs.lock().is_empty());
}

#[tokio::test]
async fn creates_the_bootstrap_directory_writes_files_runs_commands_and_writes_the_marker() {
    let mock = make_mock_session(false, 0, 0, "");
    let recipe = base_recipe();
    apply_bootstrap_recipe(&mock.session, &recipe, IDENTITY, STATE_DIRECTORY, None)
        .await
        .unwrap();
    let writes = mock.session.state.lock().writes.clone();
    assert_eq!(writes.len(), recipe.files.len() + 1);
    let runs = mock.runs.lock().clone();
    assert_eq!(runs.len(), recipe.commands.len() + 1);
    assert_eq!(runs[0].0, "mkdir -p \"$BOOTSTRAP_DIR\"");
    assert_eq!(
        runs[0].2,
        Some(vec![(
            "BOOTSTRAP_DIR".to_string(),
            "/tmp/harness/demo".to_string()
        )])
    );
    assert_eq!(
        runs.iter().map(|r| r.1.clone()).collect::<Vec<_>>(),
        vec![
            None,
            Some("/tmp/harness/demo".to_string()),
            Some("/tmp/harness/demo".to_string())
        ]
    );
    assert_eq!(runs[0].3, 0, "mkdir runs before any write");
    assert_eq!(
        writes.last().unwrap().0,
        bootstrap_marker_path(&recipe, IDENTITY, STATE_DIRECTORY)
    );
}

#[tokio::test]
async fn resolves_relative_paths_and_runs_commands_from_the_bootstrap_directory() {
    let relative = HarnessV1Bootstrap {
        harness_id: "demo".into(),
        bootstrap_dir: ".harness-bootstrap/demo".into(),
        files: vec![HarnessV1BootstrapFile {
            path: ".harness-bootstrap/demo/file.txt".into(),
            content: "content".into(),
        }],
        commands: vec![
            HarnessV1BootstrapCommand {
                command: "echo first".into(),
            },
            HarnessV1BootstrapCommand {
                command: "echo second".into(),
            },
        ],
    };
    let mock = make_mock_session(false, 0, 0, "");
    apply_bootstrap_recipe(&mock.session, &relative, IDENTITY, STATE_DIRECTORY, None)
        .await
        .unwrap();
    assert_eq!(
        *mock.session.text_reads.lock(),
        vec!["/work/.harness-bootstrap/demo/.bootstrap-idtest1234567890.ok".to_string()]
    );
    assert_eq!(
        mock.session
            .state
            .lock()
            .writes
            .iter()
            .map(|w| w.0.clone())
            .collect::<Vec<_>>(),
        vec![
            "/work/.harness-bootstrap/demo/file.txt".to_string(),
            "/work/.harness-bootstrap/demo/.bootstrap-idtest1234567890.ok".to_string(),
        ]
    );
    assert_eq!(
        mock.runs
            .lock()
            .iter()
            .map(|r| r.1.clone())
            .collect::<Vec<_>>(),
        vec![
            None,
            Some("/work/.harness-bootstrap/demo".to_string()),
            Some("/work/.harness-bootstrap/demo".to_string())
        ]
    );
}

#[tokio::test]
async fn throws_when_a_command_exits_non_zero_and_skips_the_marker_write() {
    let mock = make_mock_session(false, 0, 7, "boom");
    let recipe = base_recipe();
    let err = apply_bootstrap_recipe(&mock.session, &recipe, IDENTITY, STATE_DIRECTORY, None)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.starts_with("Bootstrap command failed"), "{err}");
    assert!(err.contains("exit 7") && err.contains("boom"), "{err}");
    assert_eq!(mock.runs.lock().len(), 2);
    let marker = bootstrap_marker_path(&recipe, IDENTITY, STATE_DIRECTORY);
    assert!(
        !mock
            .session
            .state
            .lock()
            .writes
            .iter()
            .any(|w| w.0 == marker)
    );
}

#[tokio::test]
async fn stops_before_writing_files_when_bootstrap_directory_creation_fails() {
    let mock = make_mock_session(false, 9, 0, "mkdir failed");
    let err = apply_bootstrap_recipe(
        &mock.session,
        &base_recipe(),
        IDENTITY,
        STATE_DIRECTORY,
        None,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.starts_with("Failed to create bootstrap directory"),
        "{err}"
    );
    assert!(
        err.contains("exit 9") && err.contains("mkdir failed"),
        "{err}"
    );
    assert_eq!(mock.runs.lock().len(), 1);
    assert!(mock.session.state.lock().writes.is_empty());
}

#[test]
fn bootstrap_schema_version_is_a_positive_integer() {
    const { assert!(BOOTSTRAP_SCHEMA_VERSION > 0) };
}
