//! Port of `fx-remote-ops.test.ts`.

use std::sync::Arc;

use base64::Engine as _;
use parking_lot::Mutex;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::FxRemoteOps;
use crate::harness::sandbox::SandboxRunResult;
use crate::harness::test_support::MockSandbox;

fn encoded(path: &str) -> String {
    base64::engine::general_purpose::STANDARD.encode(format!("{path}\0"))
}

/// A sandbox whose `realpath` answers come from `canonical`, in order, and
/// then `/workspace`.
/// Each run's command, working directory and whether it had a signal.
type Calls = Arc<Mutex<Vec<(String, Option<String>, bool)>>>;

fn setup(canonical: Vec<&'static str>) -> (MockSandbox, FxRemoteOps, Calls) {
    let answers = Arc::new(Mutex::new(canonical.into_iter()));
    let calls = Arc::new(Mutex::new(Vec::new()));
    let recorded = calls.clone();
    let sandbox = MockSandbox::new(move |options, _| {
        recorded.lock().push((
            options.process.command.clone(),
            options.process.working_directory.clone(),
            options.process.abort_signal.is_some(),
        ));
        let command = &options.process.command;
        Ok(SandboxRunResult {
            exit_code: 0,
            stdout: if command.contains("realpath") {
                encoded(answers.lock().next().unwrap_or("/workspace"))
            } else {
                "sandbox output".into()
            },
            ..Default::default()
        })
    })
    .with_read_text(|_| Ok(Some("sandbox file".into())));
    let remote = FxRemoteOps::new(Arc::new(sandbox.clone()), "/workspace");
    (sandbox, remote, calls)
}

#[tokio::test]
async fn rejects_paths_outside_the_workspace_before_any_sandbox_operation() {
    let (sandbox, remote, calls) = setup(vec![]);
    let error = remote
        .call("read", json!({ "file_path": "/etc/passwd" }), None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("escapes the workspace"));
    let error = remote
        .call(
            "write",
            json!({ "file_path": "../outside", "content": "no" }),
            None,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("escapes the workspace"));
    assert!(calls.lock().is_empty());
    assert!(sandbox.text_reads.lock().is_empty());
    assert!(sandbox.state.lock().writes.is_empty());
}

#[tokio::test]
async fn rejects_symlinks_escaping_the_canonical_workspace() {
    let (sandbox, remote, _) = setup(vec!["/workspace", "/outside/secret"]);
    let error = remote
        .call("read", json!({ "file_path": "link/secret" }), None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("escapes the workspace"));
    assert!(sandbox.text_reads.lock().is_empty());
}

#[tokio::test]
async fn runs_shell_commands_only_through_the_supplied_sandbox_and_forwards_cancellation() {
    let (_, remote, calls) = setup(vec![]);
    let output = remote
        .call(
            "bash",
            json!({ "command": "printf sandbox" }),
            Some(CancellationToken::new()),
        )
        .await
        .unwrap();
    assert_eq!(
        output,
        json!({ "exitCode": 0, "stdout": "sandbox output", "stderr": "" })
    );
    assert_eq!(
        calls.lock()[0],
        (
            "printf sandbox".to_string(),
            Some("/workspace".to_string()),
            true
        )
    );
}

#[tokio::test]
async fn does_not_mask_canonicalization_errors() {
    let sandbox = MockSandbox::with_result(SandboxRunResult {
        exit_code: 1,
        stderr: "failed".into(),
        ..Default::default()
    });
    let remote = FxRemoteOps::new(Arc::new(sandbox), "/workspace");
    let error = remote
        .call("read", json!({ "file_path": "file" }), None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Unable to resolve sandbox path"));
}

#[tokio::test]
async fn edits_unique_text_and_filters_globs() {
    let sandbox = MockSandbox::new(|options, _| {
        let command = &options.process.command;
        let stdout = if command.contains("realpath") {
            let target = regex::Regex::new(r"/workspace[^'\s]*")
                .unwrap()
                .find(command)
                .map_or("/workspace", |found| found.as_str())
                .to_string();
            encoded(&target)
        } else if command.starts_with("find") {
            "/workspace/a.ts\0/workspace/src/b.ts\0/workspace/c.md\0".into()
        } else {
            String::new()
        };
        Ok(SandboxRunResult {
            exit_code: 0,
            stdout,
            ..Default::default()
        })
    });
    sandbox
        .state
        .lock()
        .files
        .insert("/workspace/notes.txt".into(), "one two one".into());
    let remote = FxRemoteOps::new(Arc::new(sandbox.clone()), "/workspace");
    let error = remote
        .call(
            "edit",
            json!({ "file_path": "notes.txt", "old_string": "one", "new_string": "1" }),
            None,
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Text to replace occurs more than once. Include more context."
    );
    let output = remote
        .call(
            "edit",
            json!({ "file_path": "notes.txt", "old_string": "two", "new_string": "2" }),
            None,
        )
        .await
        .unwrap();
    assert_eq!(output, json!("Edited notes.txt"));
    assert_eq!(
        sandbox.state.lock().files["/workspace/notes.txt"],
        "one 2 one"
    );
    let files = remote
        .call("glob", json!({ "pattern": "**/*.ts" }), None)
        .await
        .unwrap();
    assert_eq!(files, json!(["a.ts", "src/b.ts"]));
}
