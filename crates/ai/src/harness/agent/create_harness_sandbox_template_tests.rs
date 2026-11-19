use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::harness::agent::settings::OnBootstrapFn;
use crate::harness::sandbox::{SandboxRunResult, SandboxSession};
use crate::harness::spec::bootstrap::HarnessV1BootstrapCommand;
use crate::harness::spec::harness::{HarnessV1Session, HarnessV1StartOptions};
use crate::harness::spec::tools::HarnessV1BuiltinTool;
use crate::harness::test_support::MockSandbox;

struct TemplateHarness {
    harness_id: String,
    recipe: Option<HarnessV1Bootstrap>,
    builtin_tools: IndexMap<String, HarnessV1BuiltinTool>,
    get_bootstrap_calls: AtomicUsize,
}

#[async_trait]
impl HarnessV1 for TemplateHarness {
    fn harness_id(&self) -> &str {
        &self.harness_id
    }

    fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool> {
        &self.builtin_tools
    }

    fn supports_get_bootstrap(&self) -> bool {
        self.recipe.is_some()
    }

    async fn get_bootstrap(
        &self,
        _abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<HarnessV1Bootstrap> {
        self.get_bootstrap_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.recipe.clone().expect("recipe"))
    }

    async fn do_start(
        &self,
        _options: HarnessV1StartOptions,
    ) -> HarnessResult<Box<dyn HarnessV1Session>> {
        Err("unused".into())
    }
}

fn harness(harness_id: &str, command: Option<&str>) -> Arc<TemplateHarness> {
    Arc::new(TemplateHarness {
        harness_id: harness_id.into(),
        recipe: Some(HarnessV1Bootstrap {
            harness_id: harness_id.into(),
            bootstrap_dir: format!(".bootstrap/{harness_id}"),
            files: vec![],
            commands: command
                .map(|c| vec![HarnessV1BootstrapCommand { command: c.into() }])
                .unwrap_or_default(),
        }),
        builtin_tools: IndexMap::new(),
        get_bootstrap_calls: AtomicUsize::new(0),
    })
}

fn sandbox() -> MockSandbox {
    MockSandbox::new(|o, _| {
        let stdout = match o.process.command.as_str() {
            "printf \"%s\" \"$HOME\"" => "/home/agent",
            "pwd" => "/work\n",
            _ => "",
        };
        Ok(SandboxRunResult {
            exit_code: 0,
            stdout: stdout.into(),
            ..Default::default()
        })
    })
}

fn counting_hook() -> (OnBootstrapFn, Arc<Mutex<Vec<String>>>) {
    let calls: Arc<Mutex<Vec<String>>> = Arc::default();
    let recorded = calls.clone();
    (
        Arc::new(move |o| {
            recorded.lock().push(o.work_dir.clone());
            Box::pin(async { Ok(()) })
        }),
        calls,
    )
}

fn config(
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

#[tokio::test]
async fn shares_aggregate_identity_sorts_harnesses_and_captures_recipes() {
    let alpha: Arc<dyn HarnessV1> = harness("alpha", Some("echo alpha"));
    let beta: Arc<dyn HarnessV1> = harness("beta", Some("echo beta"));
    let (hook, calls) = counting_hook();
    let settings = config(Some("repo/../app"), Some("tools-v1"), Some(hook));
    let template = create_harness_sandbox_template(&[beta.clone(), alpha.clone()], Some(&settings))
        .await
        .unwrap()
        .unwrap();
    let reversed = create_harness_sandbox_template(&[alpha, beta], Some(&settings))
        .await
        .unwrap()
        .unwrap();
    // [rust] TS also compares with the deprecated `prepareSandboxForHarness`,
    // which is not ported (and whose extra `onBootstrap` call made TS count 2).
    assert_eq!(reversed.identity, template.identity);

    let session = sandbox();
    let shared: Arc<dyn SandboxSession> = Arc::new(session.clone());
    (template.prepare)(shared.clone(), None).await.unwrap();
    (template.prepare)(shared, None).await.unwrap();
    let echoes: Vec<String> = session
        .state
        .lock()
        .runs
        .iter()
        .filter(|c| c.starts_with("echo"))
        .cloned()
        .collect();
    assert_eq!(echoes, vec!["echo alpha", "echo beta"]);
    assert_eq!(*calls.lock(), vec!["/work/app".to_string()]);
}

#[tokio::test]
async fn uses_the_last_adapter_for_a_repeated_id_and_resolves_recipes_only_during_creation() {
    let first = harness("alpha", Some("first"));
    let last = harness("alpha", Some("last"));
    let template =
        create_harness_sandbox_template(&[first.clone() as Arc<dyn HarnessV1>, last.clone()], None)
            .await
            .unwrap()
            .unwrap();
    let session = sandbox();
    (template.prepare)(Arc::new(session.clone()), None)
        .await
        .unwrap();
    assert_eq!(first.get_bootstrap_calls.load(Ordering::SeqCst), 0);
    assert_eq!(last.get_bootstrap_calls.load(Ordering::SeqCst), 1);
    assert!(session.state.lock().runs.contains(&"last".to_string()));
}

#[tokio::test]
async fn uses_the_sandbox_default_working_directory_for_dot_and_includes_it_in_the_identity() {
    let alpha: Arc<dyn HarnessV1> = harness("alpha", None);
    let (hook, calls) = counting_hook();
    let template = create_harness_sandbox_template(
        std::slice::from_ref(&alpha),
        Some(&config(Some("."), Some("tools-v1"), Some(hook.clone()))),
    )
    .await
    .unwrap()
    .unwrap();
    let subdirectory = create_harness_sandbox_template(
        &[alpha],
        Some(&config(Some("repo"), Some("tools-v1"), Some(hook))),
    )
    .await
    .unwrap()
    .unwrap();
    (template.prepare)(Arc::new(sandbox()), None).await.unwrap();
    assert_ne!(template.identity, subdirectory.identity);
    assert_eq!(*calls.lock(), vec!["/work".to_string()]);
}

#[tokio::test]
async fn rejects_empty_lists_and_unpaired_hashes_and_returns_none_without_work() {
    let err = create_harness_sandbox_template(&[], None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("at least one harness"));
    let (hook, _) = counting_hook();
    let err = create_harness_sandbox_template(
        &[harness("alpha", None) as Arc<dyn HarnessV1>],
        Some(&config(None, None, Some(hook))),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("must be provided together"));
    let empty = Arc::new(TemplateHarness {
        harness_id: "empty".into(),
        recipe: None,
        builtin_tools: IndexMap::new(),
        get_bootstrap_calls: AtomicUsize::new(0),
    });
    assert!(
        create_harness_sandbox_template(
            &[empty as Arc<dyn HarnessV1>],
            Some(&config(Some("repo"), None, None))
        )
        .await
        .unwrap()
        .is_none()
    );
}
