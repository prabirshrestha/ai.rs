use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

use super::*;
use crate::harness::sandbox::SandboxRunResult;
use crate::harness::spec::bootstrap::{HarnessV1BootstrapCommand, HarnessV1BootstrapFile};
use crate::harness::test_support::MockSandbox;

fn recipe() -> HarnessV1Bootstrap {
    HarnessV1Bootstrap {
        harness_id: "demo".into(),
        bootstrap_dir: ".harness-bootstrap/demo".into(),
        files: vec![HarnessV1BootstrapFile {
            path: ".harness-bootstrap/demo/a.txt".into(),
            content: "one".into(),
        }],
        commands: vec![HarnessV1BootstrapCommand {
            command: "echo ok".into(),
        }],
    }
}

type RunRecord = (String, Option<Vec<(String, String)>>);
type HookCalls = Arc<Mutex<Vec<(String, usize)>>>;

fn make_session() -> (MockSandbox, Arc<Mutex<Vec<RunRecord>>>) {
    let runs: Arc<Mutex<Vec<RunRecord>>> = Arc::default();
    let recorded = runs.clone();
    let session = MockSandbox::new(move |o, _| {
        recorded.lock().push((
            o.process.command.clone(),
            o.process
                .env
                .as_ref()
                .map(|e| e.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        ));
        let stdout = match o.process.command.as_str() {
            "pwd" => "/work\n",
            "printf \"%s\" \"$HOME\"" => "/home/agent",
            _ => "",
        };
        Ok(SandboxRunResult {
            exit_code: 0,
            stdout: stdout.into(),
            ..Default::default()
        })
    });
    (session, runs)
}

/// An `onBootstrap` hook recording each call's work dir and the number of
/// runs before it.
fn recording_hook(runs: Arc<Mutex<Vec<RunRecord>>>) -> (OnBootstrapFn, HookCalls) {
    let calls: HookCalls = Arc::default();
    let recorded = calls.clone();
    let hook: OnBootstrapFn = Arc::new(move |o| {
        recorded
            .lock()
            .push((o.work_dir.clone(), runs.lock().len()));
        Box::pin(async { Ok(()) })
    });
    (hook, calls)
}

fn noop_hook() -> OnBootstrapFn {
    Arc::new(|_| Box::pin(async { Ok(()) }))
}

fn settings(
    work_dir: Option<&str>,
    bootstrap_hash: Option<&str>,
    on_bootstrap: Option<OnBootstrapFn>,
) -> HarnessAgentSandboxConfig {
    HarnessAgentSandboxConfig {
        work_dir: work_dir.map(Into::into),
        bootstrap_hash: bootstrap_hash.map(Into::into),
        on_bootstrap,
        on_session: None,
    }
}

#[test]
fn requires_on_bootstrap_and_bootstrap_hash_together() {
    let err = validate_sandbox_bootstrap_settings(&settings(None, None, Some(noop_hook())))
        .unwrap_err()
        .to_string();
    assert!(err.contains("must be provided together"), "{err}");
    let err = validate_sandbox_bootstrap_settings(&settings(None, Some("hash"), None))
        .unwrap_err()
        .to_string();
    assert!(err.contains("must be provided together"), "{err}");
}

#[test]
fn rejects_invalid_work_dir_values() {
    for value in [
        "",
        "./",
        "repo/..",
        "/repo",
        "../repo",
        "repo/../../x",
        "a\\b",
    ] {
        let err = validate_sandbox_bootstrap_settings(&settings(Some(value), None, None))
            .unwrap_err()
            .to_string();
        assert!(err.contains("workDir"), "{value}: {err}");
    }
}

#[test]
fn normalizes_work_dir_values_that_stay_inside_the_default_cwd() {
    assert_eq!(
        normalize_sandbox_work_dir("repo/../ai-sdk").unwrap(),
        "ai-sdk"
    );
    assert_eq!(normalize_sandbox_work_dir("./ai-sdk").unwrap(), "ai-sdk");
    assert_eq!(normalize_sandbox_work_dir(".").unwrap(), ".");
}

#[test]
fn preserves_the_built_in_bootstrap_identity_when_no_new_settings_are_used() {
    let plan = create_sandbox_bootstrap_plan(Some(&recipe()), &settings(None, None, None)).unwrap();
    assert_eq!(plan.identity, Some(hash_harness_bootstrap(&recipe())));
}

#[test]
fn changes_identity_when_caller_bootstrap_hash_changes() {
    let a = create_sandbox_bootstrap_plan(
        Some(&recipe()),
        &settings(None, Some("repo-v1"), Some(noop_hook())),
    )
    .unwrap();
    let b = create_sandbox_bootstrap_plan(
        Some(&recipe()),
        &settings(None, Some("repo-v2"), Some(noop_hook())),
    )
    .unwrap();
    assert_ne!(a.identity, b.identity);
}

#[test]
fn changes_identity_when_work_dir_changes() {
    let a = create_sandbox_bootstrap_plan(Some(&recipe()), &settings(Some("repo-a"), None, None))
        .unwrap();
    let b = create_sandbox_bootstrap_plan(Some(&recipe()), &settings(Some("repo-b"), None, None))
        .unwrap();
    assert_ne!(a.identity, b.identity);
}

#[test]
fn uses_the_session_scoped_default_path_when_work_dir_is_absent() {
    assert_eq!(
        resolve_session_work_dir("/work", "mock", "s1", None),
        "/work/mock-s1"
    );
}

#[test]
fn keeps_caller_controlled_ids_within_the_default_working_directory() {
    assert_eq!(
        resolve_session_work_dir("/work", "mock", "../../../../project", None),
        "/work/mock-..%2F..%2F..%2F..%2Fproject"
    );
    assert_eq!(
        resolve_session_work_dir("/work", "../mock", "s1", None),
        "/work/..%2Fmock-s1"
    );
}

#[test]
fn uses_the_stable_work_dir_when_provided() {
    assert_eq!(
        resolve_session_work_dir("/work", "mock", "s1", Some("ai-sdk")),
        "/work/ai-sdk"
    );
}

#[test]
fn uses_the_sandbox_default_working_directory_for_work_dir_dot() {
    assert_eq!(
        resolve_session_work_dir("/work", "mock", "s1", Some(".")),
        "/work"
    );
}

#[tokio::test]
async fn runs_built_in_bootstrap_before_caller_bootstrap() {
    let (session, runs) = make_session();
    let (hook, calls) = recording_hook(runs.clone());
    let recipe_identity = hash_harness_bootstrap(&recipe());
    run_sandbox_bootstrap(RunSandboxBootstrapOptions {
        recipe: Some(recipe()),
        recipe_identity: Some(recipe_identity),
        work_dir: Some("ai-sdk".into()),
        on_bootstrap: Some(hook),
        ..RunSandboxBootstrapOptions::new(Arc::new(session.clone()))
    })
    .await
    .unwrap();

    assert_eq!(calls.lock()[0].0, "/work/ai-sdk");
    assert_eq!(
        runs.lock().iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
        vec![
            "printf \"%s\" \"$HOME\"",
            "mkdir -p \"$BOOTSTRAP_DIR\"",
            "echo ok",
            "pwd",
            "mkdir -p \"$WORK_DIR\"",
        ]
    );
    let reads = session.text_reads.lock().clone();
    assert_eq!(reads.len(), 1);
    let marker = &reads[0];
    let prefix = "/home/agent/.ai-sdk-harness/.harness-bootstrap/demo/.bootstrap-";
    assert!(
        marker.starts_with(prefix) && marker.ends_with(".ok"),
        "{marker}"
    );
    assert_eq!(marker.len(), prefix.len() + 16 + 3);
    assert!(session.state.lock().writes.contains(&(
        "/home/agent/.ai-sdk-harness/.harness-bootstrap/demo/a.txt".to_string(),
        "one".to_string()
    )));
    assert!(calls.lock()[0].1 > 2, "echo ok runs before the hook");
}

#[tokio::test]
async fn uses_the_default_cwd_for_caller_bootstrap_when_work_dir_is_absent() {
    let (session, runs) = make_session();
    let (hook, calls) = recording_hook(runs);
    run_sandbox_bootstrap(RunSandboxBootstrapOptions {
        on_bootstrap: Some(hook),
        ..RunSandboxBootstrapOptions::new(Arc::new(session))
    })
    .await
    .unwrap();
    assert_eq!(calls.lock()[0].0, "/work");
}

#[tokio::test]
async fn uses_the_sandbox_default_working_directory_for_caller_bootstrap_when_work_dir_is_dot() {
    let (session, runs) = make_session();
    let (hook, calls) = recording_hook(runs.clone());
    run_sandbox_bootstrap(RunSandboxBootstrapOptions {
        work_dir: Some(".".into()),
        on_bootstrap: Some(hook),
        ..RunSandboxBootstrapOptions::new(Arc::new(session))
    })
    .await
    .unwrap();
    assert!(runs.lock().contains(&(
        "mkdir -p \"$WORK_DIR\"".to_string(),
        Some(vec![("WORK_DIR".to_string(), "/work".to_string())])
    )));
    assert_eq!(calls.lock()[0].0, "/work");
}

// on-bootstrap-marker.test.ts

#[tokio::test]
async fn on_bootstrap_marker_uses_sandbox_home_and_only_the_hash_writing_after_success() {
    let (session, _) = make_session();
    let hook_calls = Arc::new(AtomicUsize::new(0));
    let counter = hook_calls.clone();
    let hook: OnBootstrapFn = Arc::new(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    });
    for work_dir in ["one", "two"] {
        run_sandbox_bootstrap(RunSandboxBootstrapOptions {
            work_dir: Some(work_dir.into()),
            bootstrap_hash: Some("v1".into()),
            on_bootstrap: Some(hook.clone()),
            skip_on_bootstrap_if_marked: true,
            ..RunSandboxBootstrapOptions::new(Arc::new(session.clone()))
        })
        .await
        .unwrap();
    }
    assert_eq!(hook_calls.load(Ordering::SeqCst), 1);
    let files: Vec<String> = session.state.lock().files.keys().cloned().collect();
    assert_eq!(files.len(), 1);
    let prefix = "/home/agent/.ai-sdk-harness/.on-bootstrap/";
    let path = &files[0];
    assert!(path.starts_with(prefix) && path.ends_with(".ok"), "{path}");
    let digest = &path[prefix.len()..path.len() - 3];
    assert_eq!(digest.len(), 64);
    assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
}

#[tokio::test]
async fn on_bootstrap_marker_does_not_mark_failed_hooks() {
    let (session, _) = make_session();
    let err = run_sandbox_bootstrap(RunSandboxBootstrapOptions {
        bootstrap_hash: Some("v1".into()),
        on_bootstrap: Some(Arc::new(|_| Box::pin(async { Err("failed".into()) }))),
        skip_on_bootstrap_if_marked: true,
        ..RunSandboxBootstrapOptions::new(Arc::new(session.clone()))
    })
    .await
    .unwrap_err();
    assert_eq!(err.to_string(), "failed");
    assert!(session.state.lock().writes.is_empty());
}
