//! `harness-agent.test.ts`: sandbox setup, bootstrap recipes, deferred
//! sandboxes, history, and resume-state validation.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde_json::json;

use super::*;
use crate::harness::spec::lifecycle_state::HarnessV1ResumeSessionState;

const MKDIR: &str = "mkdir -p \"$WORK_DIR\"";

fn bootstrap_marker(path: &str) -> bool {
    regex::Regex::new(r"^/home/agent/\.ai-sdk-harness/\.harness-bootstrap/mock/\.bootstrap-[0-9a-f]{16}\.ok$")
        .unwrap()
        .is_match(path)
}

/// `onSession` calls as `(session description, sessionWorkDir)`, logged
/// as `on-session` in `log`.
#[derive(Clone, Default)]
struct OnSessionRecorder {
    calls: Arc<Mutex<Vec<(String, String)>>>,
    log: Arc<Mutex<Vec<String>>>,
}

impl OnSessionRecorder {
    fn with_log(log: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            calls: Arc::default(),
            log,
        }
    }
    fn hook(&self) -> OnSessionFn {
        let recorder = self.clone();
        Arc::new(move |options: OnSessionOptions| {
            recorder.log.lock().push("on-session".into());
            recorder
                .calls
                .lock()
                .push((options.session.description(), options.session_work_dir));
            Box::pin(async { Ok(()) })
        })
    }
    fn calls(&self) -> Vec<(String, String)> {
        self.calls.lock().clone()
    }
}

fn sandbox_config(f: impl FnOnce(&mut HarnessAgentSandboxConfig)) -> Option<HarnessAgentSandboxConfig> {
    let mut config = HarnessAgentSandboxConfig::default();
    f(&mut config);
    Some(config)
}

fn index_of(log: &[String], entry: &str) -> usize {
    log.iter()
        .position(|e| e == entry)
        .unwrap_or_else(|| panic!("missing {entry} in {log:?}"))
}

fn with_recipe(mut options: MockHarnessOptions) -> MockHarnessOptions {
    options.bootstrap = Some(bootstrap_recipe());
    options
}

#[tokio::test]
async fn sandbox_config_on_session_runs_after_the_session_work_dir_exists_and_before_harness_start_for_a_harness_with_a_bootstrap_recipe()
 {
    let log: Arc<Mutex<Vec<String>>> = Arc::default();
    let mut options = with_recipe(MockHarnessOptions::script(Vec::new));
    let start_log = log.clone();
    options.on_do_start = Some(Arc::new(move |_| start_log.lock().push("do-start".into())));
    let mock = options.build();
    let sandbox = TestSandbox::with_inner(ScriptedSandbox::new("mock sandbox", log.clone()));
    let on_session = OnSessionRecorder::with_log(log.clone());
    let mut s = settings(&mock);
    s.sandbox_config = sandbox_config(|c| c.on_session = Some(on_session.hook()));
    let agent = agent(s);

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();

    assert!(
        sandbox
            .runs()
            .contains(&RunCall::new(MKDIR, &[("WORK_DIR", "/work/mock-s1")]))
    );
    assert_eq!(
        on_session.calls(),
        vec![("mock sandbox".to_string(), "/work/mock-s1".to_string())]
    );
    let log = log.lock().clone();
    assert!(index_of(&log, &format!("run:{MKDIR}")) < index_of(&log, "on-session"));
    assert!(index_of(&log, "on-session") < index_of(&log, "do-start"));

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn work_dir_dot_uses_the_sandbox_default_working_directory_for_the_session_and_harness() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let inner = ScriptedSandbox::new("mock sandbox", Arc::default()).with_run(empty_run());
    let sandbox = TestSandbox::with_inner(inner.clone()).with_restricted(inner.relabeled("restricted"));
    let on_session = OnSessionRecorder::default();
    let mut s = settings(&mock);
    s.sandbox_config = sandbox_config(|c| {
        c.work_dir = Some(".".into());
        c.on_session = Some(on_session.hook());
    });
    let agent = agent(s);

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    touch_started_sandbox(&mock).await.unwrap();

    assert!(sandbox.runs().contains(&RunCall::new(MKDIR, &[("WORK_DIR", "/work")])));
    assert_eq!(
        on_session.calls(),
        vec![("restricted".to_string(), "/work".to_string())]
    );
    assert_eq!(mock.start_call(0).session_work_dir, "/work");

    session.destroy().await.unwrap();
}

// [rust] 'deprecated top-level onSandboxSession warns and still runs' is not
// ported: the deprecated `onSandboxSession` setting is not ported.

#[test]
fn validates_sandbox_bootstrap_settings() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let with_config = |config: Option<HarnessAgentSandboxConfig>| {
        let mut s = settings(&mock);
        s.sandbox_config = config;
        err_message(HarnessAgent::new(s))
    };

    let message = with_config(sandbox_config(|c| {
        c.on_bootstrap = Some(Arc::new(|_| Box::pin(async { Ok(()) })));
    }));
    assert!(message.contains("must be provided together"), "{message}");

    let message = with_config(sandbox_config(|c| c.bootstrap_hash = Some("hash".into())));
    assert!(message.contains("must be provided together"), "{message}");

    let message = with_config(sandbox_config(|c| c.work_dir = Some("../repo".into())));
    assert!(message.contains("workDir"), "{message}");
}

#[tokio::test]
async fn requires_a_provided_sandbox_session() {
    // [rust] The deprecated constructor `sandbox` provider is not ported, so
    // the message names only `sandboxSession`.
    let mock = MockHarnessOptions::script(Vec::new).build();
    let agent = agent(settings(&mock));

    assert_eq!(
        err_message(agent.create_session(Default::default()).await),
        "HarnessAgent.createSession: pass `sandboxSession` to createSession()."
    );
}

// [rust] The deprecated constructor `sandbox` provider is not ported, so
// these are not either: 'deprecated constructor sandbox provider creates and
// resumes sessions and warns', '... is ignored when a sandbox session is
// provided', '... runs sandboxConfig.onBootstrap during onFirstCreate', and
// '... uses separate bootstrap and snapshot identities'.

#[tokio::test]
async fn uses_a_provided_basic_sandbox_session_resolves_its_working_directory_and_applies_the_harness_bootstrap_recipe()
 {
    let mock = with_recipe(MockHarnessOptions::script(Vec::new)).build();
    let restricted_session = ScriptedSandbox::new("restricted", Arc::default()).with_read_text(|_| Ok(None));
    let basic: Arc<dyn SandboxSession> = Arc::new(restricted_session.clone());
    let agent = agent(settings(&mock));

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            sandbox_session: Some(HarnessV1SandboxSessionRef::Basic(basic.clone()).into()),
            resume_from: Some(mock_resume_state()),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(
        restricted_session.writes.lock().iter().any(|path| bootstrap_marker(path)),
        "{:?}",
        restricted_session.writes.lock()
    );
    assert!(restricted_session.runs.lock().contains(&RunCall::new("pwd", &[])));
    match mock.start_call(0).sandbox_session {
        HarnessV1SandboxSessionRef::Basic(started) => {
            assert!(std::ptr::addr_eq(Arc::as_ptr(&started), Arc::as_ptr(&basic)));
        }
        other => panic!("expected the provided basic session, got {other:?}"),
    }

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn does_not_stop_a_provided_sandbox_session_when_harness_startup_fails() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    mock.record.start_errors.lock().push("start failed".into());
    let sandbox = make_sandbox_session();
    let agent = agent(settings(&mock));

    let message = err_message(agent.create_session(with_sandbox(&sandbox)).await);
    assert!(message.contains("start failed"), "{message}");
    assert_eq!(sandbox.stops(), 0);
    assert_eq!(sandbox.destroys(), 0);
}

#[tokio::test]
async fn sandbox_config_on_bootstrap_runs_for_a_resumed_session_missing_its_marker_while_on_session_still_runs()
 {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let bootstraps = Arc::new(AtomicUsize::new(0));
    let on_session = OnSessionRecorder::default();
    let mut s = settings(&mock);
    let count = bootstraps.clone();
    s.sandbox_config = sandbox_config(|c| {
        c.work_dir = Some("ai-sdk".into());
        c.bootstrap_hash = Some("repo-v1".into());
        c.on_bootstrap = Some(Arc::new(move |_| {
            count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(()) })
        }));
        c.on_session = Some(on_session.hook());
    });
    let agent = agent(s);

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            resume_from: Some(mock_resume_state()),
            ..with_sandbox(&make_sandbox_session())
        })
        .await
        .unwrap();
    touch_started_sandbox(&mock).await.unwrap();

    assert_eq!(bootstraps.load(Ordering::SeqCst), 1);
    assert_eq!(
        on_session.calls().iter().map(|(_, dir)| dir.clone()).collect::<Vec<_>>(),
        vec!["/work/ai-sdk"]
    );

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn read_history_reads_the_runtime_history_through_the_adapter() {
    let history: HarnessV1ReadHistoryResult = HarnessV1ReadHistoryResult {
        messages: serde_json::from_value(json!([
            { "role": "user", "content": [{ "type": "text", "text": "hello" }] },
            {
                "role": "assistant",
                "at": "2026-09-29T12:00:00.000Z",
                "harnessMetadata": { "mock": { "raw": { "messageId": "assistant-1" } } },
                "content": [
                    { "type": "reasoning", "text": "thinking it over" },
                    { "type": "tool-call", "toolCallId": "tool-1", "toolName": "bash", "nativeName": "Bash", "input": { "command": "ls" } },
                    { "type": "tool-result", "toolCallId": "tool-1", "toolName": "bash", "output": { "type": "text", "value": "README.md" } },
                    { "type": "text", "text": "done" },
                ],
            },
        ]))
        .unwrap(),
        cursor: "cursor-1".into(),
    };
    let since_calls: Arc<Mutex<Vec<Option<String>>>> = Arc::default();
    let mut options = MockHarnessOptions::script(Vec::new);
    let (calls, result) = (since_calls.clone(), history.clone());
    options.do_read_history = Some(Arc::new(move |since| {
        calls.lock().push(since);
        let result = result.clone();
        Box::pin(async move { Ok(result) })
    }));
    let mock = options.build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    assert_eq!(session.read_history(None).await.unwrap(), history);
    session.read_history(Some("cursor-1".into())).await.unwrap();
    assert_eq!(since_calls.lock().last().unwrap().as_deref(), Some("cursor-1"));

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn read_history_throws_harness_capability_unsupported_error_when_the_adapter_lacks_it() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;

    let error = session.read_history(None).await.unwrap_err();
    assert!(matches!(
        error.downcast_ref::<HarnessError>(),
        Some(HarnessError::CapabilityUnsupported { .. })
    ));

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn read_history_rejects_once_the_session_is_no_longer_active() {
    let mut options = MockHarnessOptions::script(Vec::new);
    options.do_read_history = Some(Arc::new(|_| {
        Box::pin(async {
            Ok(HarnessV1ReadHistoryResult {
                messages: Vec::new(),
                cursor: "c".into(),
            })
        })
    }));
    let mock = options.build();
    let agent = agent(settings(&mock));
    let session = new_session(&agent).await;
    session.destroy().await.unwrap();

    let message = err_message(session.read_history(None).await);
    assert!(message.contains("not active and cannot read history"), "{message}");
}

#[tokio::test]
async fn ensures_the_harness_bootstrap_recipe_on_resumed_sessions() {
    let mock = with_recipe(MockHarnessOptions::script(Vec::new)).build();
    // No marker: the sandbox was bootstrapped by an older recipe, or never.
    let inner = ScriptedSandbox::new("mock sandbox", Arc::default()).with_read_text(|_| Ok(None));
    let sandbox = TestSandbox::with_inner(inner.clone());
    let mut s = settings(&mock);
    s.sandbox_config = sandbox_config(|c| c.work_dir = Some("ai-sdk".into()));
    let agent = agent(s);

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            resume_from: Some(mock_resume_state()),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();

    assert!(bootstrap_marker(&inner.reads.lock()[0]), "{:?}", inner.reads.lock());
    assert!(
        bootstrap_marker(inner.writes.lock().last().unwrap()),
        "{:?}",
        inner.writes.lock()
    );

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn sandbox_config_on_session_runs_for_resumed_sessions() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let on_session = OnSessionRecorder::default();
    let mut s = settings(&mock);
    s.sandbox_config = sandbox_config(|c| c.on_session = Some(on_session.hook()));
    let agent = agent(s);

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            resume_from: Some(mock_resume_state()),
            ..with_sandbox(&make_sandbox_session())
        })
        .await
        .unwrap();
    touch_started_sandbox(&mock).await.unwrap();

    assert_eq!(
        on_session.calls().iter().map(|(_, dir)| dir.clone()).collect::<Vec<_>>(),
        vec!["/work/mock-s1"]
    );

    session.destroy().await.unwrap();
}

fn read_options(path: &str) -> SandboxReadTextFileOptions {
    SandboxReadTextFileOptions {
        path: path.into(),
        ..Default::default()
    }
}

fn run_options(command: &str) -> SandboxRunOptions {
    SandboxRunOptions {
        process: SandboxProcessOptions::new(command),
        ..Default::default()
    }
}

#[tokio::test]
async fn for_a_harness_without_a_bootstrap_recipe_sandbox_setup_runs_on_the_first_sandbox_operation_not_at_create_session()
 {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let log: Arc<Mutex<Vec<String>>> = Arc::default();
    let inner = ScriptedSandbox::new("mock sandbox", log.clone())
        .with_run(empty_run())
        .with_read_text(|_| Ok(Some("content".into())));
    let sandbox = TestSandbox::with_inner(inner.clone()).with_restricted(inner.relabeled("restricted"));
    let on_session = OnSessionRecorder::with_log(log.clone());
    let mut s = settings(&mock);
    s.sandbox_config = sandbox_config(|c| c.on_session = Some(on_session.hook()));
    let agent = agent(s);

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();

    assert_eq!(mock.start_count(), 1);
    assert!(sandbox.runs().is_empty());
    assert!(on_session.calls().is_empty());

    let started_sandbox = started_session(&mock);
    assert_eq!(
        started_sandbox
            .read_text_file(read_options("/work/mock-s1/a.txt"))
            .await
            .unwrap()
            .as_deref(),
        Some("content")
    );

    assert_eq!(
        sandbox.runs(),
        vec![RunCall::new(MKDIR, &[("WORK_DIR", "/work/mock-s1")])]
    );
    assert_eq!(
        on_session.calls(),
        vec![("restricted".to_string(), "/work/mock-s1".to_string())]
    );
    let log = log.lock().clone();
    assert!(index_of(&log, &format!("run:{MKDIR}")) < index_of(&log, "on-session"));
    assert!(index_of(&log, "on-session") < index_of(&log, "read:/work/mock-s1/a.txt"));

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn for_a_harness_without_a_bootstrap_recipe_concurrent_first_operations_on_the_sandbox_and_its_restricted_view_share_one_setup()
 {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let sandbox = make_sandbox_session();
    let agent = agent(settings(&mock));

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    assert!(sandbox.runs().is_empty());
    let HarnessV1SandboxSessionRef::Network(started_sandbox) = get_started_sandbox(&mock) else {
        panic!("expected a network sandbox");
    };

    let (restricted_a, restricted_b) = (started_sandbox.restricted(), started_sandbox.restricted());
    let (read, write, run) = tokio::join!(
        started_sandbox.read_text_file(read_options("/work/mock-s1/a.txt")),
        restricted_a.write_text_file(SandboxWriteTextFileOptions {
            path: "/work/mock-s1/b.txt".into(),
            content: "b".into(),
            ..Default::default()
        }),
        restricted_b.run(run_options("true")),
    );
    read.unwrap();
    write.unwrap();
    run.unwrap();

    let runs = sandbox.runs();
    assert_eq!(runs.iter().filter(|r| r.command == MKDIR).count(), 1);
    assert!(runs.contains(&RunCall::new("true", &[])));

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn for_a_harness_without_a_bootstrap_recipe_a_failed_sandbox_setup_rejects_its_operation_and_is_retried_on_the_next_one()
 {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let calls = Arc::new(AtomicUsize::new(0));
    let run: RunFn = Arc::new(move |_| {
        Ok(if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            SandboxRunResult {
                exit_code: 1,
                stderr: "denied".into(),
                ..Default::default()
            }
        } else {
            SandboxRunResult::default()
        })
    });
    let inner = ScriptedSandbox::new("mock sandbox", Arc::default())
        .with_run(run)
        .with_read_text(|_| Ok(Some("content".into())));
    let sandbox = TestSandbox::with_inner(inner.clone());
    let agent = agent(settings(&mock));

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    let started_sandbox = started_session(&mock);

    assert_eq!(
        err_message(started_sandbox.read_text_file(read_options("/work/mock-s1/a.txt")).await),
        "Failed to create sandbox work directory /work/mock-s1 (exit 1): denied"
    );
    assert!(inner.reads.lock().is_empty());

    assert_eq!(
        started_sandbox
            .read_text_file(read_options("/work/mock-s1/a.txt"))
            .await
            .unwrap()
            .as_deref(),
        Some("content")
    );
    assert_eq!(sandbox.runs().len(), 2);
    assert_eq!(inner.reads.lock().len(), 1);

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn for_a_harness_without_a_bootstrap_recipe_stopping_and_destroying_the_sandbox_does_not_run_setup() {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let sandbox = make_sandbox_session();
    let on_session = OnSessionRecorder::default();
    let mut s = settings(&mock);
    s.sandbox_config = sandbox_config(|c| c.on_session = Some(on_session.hook()));
    let agent = agent(s);

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            ..with_sandbox(&sandbox)
        })
        .await
        .unwrap();
    let HarnessV1SandboxSessionRef::Network(started_sandbox) = get_started_sandbox(&mock) else {
        panic!("expected a network sandbox");
    };

    started_sandbox.stop().await.unwrap();
    started_sandbox.destroy().await.unwrap();

    assert_eq!(sandbox.stops(), 1);
    assert_eq!(sandbox.destroys(), 1);
    assert!(sandbox.runs().is_empty());
    assert!(on_session.calls().is_empty());

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn create_session_reports_a_missing_sandbox_before_it_asks_the_harness_for_a_bootstrap_recipe() {
    let mock = with_recipe(MockHarnessOptions::script(Vec::new)).build();

    assert_eq!(
        err_message(agent(settings(&mock)).create_session(Default::default()).await),
        "HarnessAgent.createSession: pass `sandboxSession` to createSession()."
    );

    assert_eq!(Mock::count(&mock.record.get_bootstrap_calls), 0);
}

// [rust] 'for a harness without a bootstrap recipe, a sandbox provider is
// still called at createSession() while the work dir waits' is not ported:
// the deprecated constructor `sandbox` provider is not ported.

/// A [`HarnessAgentSandboxCreator`] over `sandbox`, counting its calls and
/// failing the first `failures` of them.
fn creator(
    default_working_directory: &str,
    sandbox: &TestSandbox,
    failures: usize,
    abort_signals: Arc<Mutex<Vec<Option<tokio_util::sync::CancellationToken>>>>,
) -> (HarnessAgentSandboxCreator, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let (count, sandbox) = (calls.clone(), sandbox.clone());
    let creator = HarnessAgentSandboxCreator {
        default_working_directory: default_working_directory.into(),
        create_sandbox_session: Arc::new(move |abort_signal| {
            abort_signals.lock().push(abort_signal);
            let call = count.fetch_add(1, Ordering::SeqCst);
            let sandbox = sandbox.sref();
            Box::pin(async move {
                if call < failures {
                    return Err(plain_error("sandbox unavailable"));
                }
                Ok(sandbox)
            })
        }),
    };
    (creator, calls)
}

#[tokio::test]
async fn for_a_harness_without_a_bootstrap_recipe_create_sandbox_session_runs_on_the_first_sandbox_operation_with_setup_before_it()
 {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let sandbox =
        TestSandbox::with_inner(ScriptedSandbox::new("Test sandbox.", Arc::default()));
    let (creator, create_calls) = creator("/work", &sandbox, 0, Arc::default());
    let on_session = OnSessionRecorder::default();
    let mut s = settings(&mock);
    s.sandbox_config = sandbox_config(|c| c.on_session = Some(on_session.hook()));
    let agent = agent(s);

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            sandbox_session: Some(creator.into()),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(create_calls.load(Ordering::SeqCst), 0);
    assert_eq!(mock.start_call(0).session_work_dir, "/work/mock-s1");
    let started_sandbox = started_session(&mock);
    assert_eq!(
        started_sandbox.description(),
        "Sandbox with the working directory /work. It starts on first use."
    );

    let (run, read) = tokio::join!(
        started_sandbox.run(run_options("ls")),
        started_sandbox.read_text_file(read_options("/work/mock-s1/a.txt")),
    );
    run.unwrap();
    read.unwrap();

    assert_eq!(create_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        sandbox.runs(),
        vec![
            RunCall::new(MKDIR, &[("WORK_DIR", "/work/mock-s1")]),
            RunCall::new("ls", &[]),
        ]
    );
    assert_eq!(on_session.calls().len(), 1);
    assert_eq!(started_sandbox.description(), "Test sandbox.");

    session.destroy().await.unwrap();
    assert_eq!(sandbox.destroys(), 0);
}

#[tokio::test]
async fn for_a_harness_without_a_bootstrap_recipe_a_failed_create_sandbox_session_is_retried_on_the_next_operation()
 {
    let mock = MockHarnessOptions::script(Vec::new).build();
    let sandbox = make_sandbox_session();
    let (creator, create_calls) = creator("/work", &sandbox, 1, Arc::default());
    let agent = agent(settings(&mock));

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            sandbox_session: Some(creator.into()),
            ..Default::default()
        })
        .await
        .unwrap();
    let started_sandbox = started_session(&mock);

    let message = err_message(started_sandbox.run(run_options("ls")).await);
    assert!(message.contains("sandbox unavailable"), "{message}");
    started_sandbox.run(run_options("ls")).await.unwrap();

    assert_eq!(create_calls.load(Ordering::SeqCst), 2);
    assert!(sandbox.runs().contains(&RunCall::new("ls", &[])));

    session.destroy().await.unwrap();
}

#[tokio::test]
async fn for_a_harness_with_a_bootstrap_recipe_create_sandbox_session_runs_in_create_session_and_skips_pwd() {
    let mock = with_recipe(MockHarnessOptions::script(Vec::new)).build();
    let sandbox = make_sandbox_session();
    let abort_signals: Arc<Mutex<Vec<Option<tokio_util::sync::CancellationToken>>>> = Arc::default();
    let (creator, _) = creator("/custom", &sandbox, 0, abort_signals.clone());
    let abort_controller = tokio_util::sync::CancellationToken::new();
    let agent = agent(settings(&mock));

    let session = agent
        .create_session(HarnessAgentCreateSessionOptions {
            session_id: Some("s1".into()),
            sandbox_session: Some(creator.into()),
            abort_signal: Some(abort_controller.clone()),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(abort_signals.lock().len(), 1);
    assert!(abort_signals.lock()[0].is_some());
    assert!(!sandbox.runs().iter().any(|r| r.command == "pwd"));
    let started = mock.start_call(0);
    assert!(matches!(&started.sandbox_session, HarnessV1SandboxSessionRef::Network(s) if s.id() == "sandbox"));
    assert_eq!(started.session_work_dir, "/custom/mock-s1");

    session.destroy().await.unwrap();
}

#[test]
fn create_session_rejects_resume_state_with_top_level_pending_tool_approvals() {
    // [rust] `HarnessV1ResumeSessionState` has no top-level pending fields,
    // so the state is rejected when it is deserialized.
    let error = serde_json::from_value::<HarnessV1ResumeSessionState>(json!({
        "type": "resume-session",
        "harnessId": "mock",
        "specificationVersion": "harness-v1",
        "data": {},
        "pendingToolApprovals": [],
    }))
    .unwrap_err();
    assert!(error.to_string().contains("cannot contain pending tool approvals"), "{error}");
}

#[test]
fn create_session_rejects_resume_state_with_top_level_pending_tool_results() {
    let error = serde_json::from_value::<HarnessV1ResumeSessionState>(json!({
        "type": "resume-session",
        "harnessId": "mock",
        "specificationVersion": "harness-v1",
        "data": {},
        "pendingToolResults": [],
    }))
    .unwrap_err();
    assert!(error.to_string().contains("cannot contain pending tool results"), "{error}");
}
