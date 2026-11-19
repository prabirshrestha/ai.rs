//! Port of `agent/internal/deferred-sandbox-setup.ts`.

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::sandbox_operations::{
    DelegatedSandboxSession, RetryOnFailure, delegate_sandbox_operations,
};
use crate::harness::agent::settings::CreateSandboxSessionFn;
use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::*;
use crate::harness::spec::harness::HarnessV1SandboxSessionRef;
use crate::harness::spec::network_sandbox::*;
use crate::harness::utils::sandbox_home_dir::get_restricted_sandbox_session;

/// The setup step `defer_sandbox_setup` gates operations on.
pub type SandboxSetupFn = Arc<dyn Fn() -> BoxFuture<'static, HarnessResult<()>> + Send + Sync>;

fn gate(
    target: Arc<dyn SandboxSession>,
    ensure: Arc<RetryOnFailure<()>>,
) -> DelegatedSandboxSession {
    let description_target = target.clone();
    delegate_sandbox_operations(
        Arc::new(move || {
            let ensure = ensure.clone();
            let target = target.clone();
            Box::pin(async move {
                ensure.get().await?;
                Ok(target)
            })
        }),
        Arc::new(move || description_target.description()),
    )
}

/// `deferSandboxSetup`: runs `setup` before the first file or process
/// operation. A failed setup runs again on the next operation.
pub fn defer_sandbox_setup(
    sandbox_session: HarnessV1SandboxSessionRef,
    setup: SandboxSetupFn,
) -> HarnessV1SandboxSessionRef {
    let ensure = Arc::new(RetryOnFailure::new(setup));
    match sandbox_session {
        HarnessV1SandboxSessionRef::Basic(session) => {
            HarnessV1SandboxSessionRef::Basic(Arc::new(gate(session, ensure)))
        }
        HarnessV1SandboxSessionRef::Network(original) => {
            let ops = gate(original.clone() as Arc<dyn SandboxSession>, ensure.clone());
            HarnessV1SandboxSessionRef::Network(Arc::new(GatedNetworkSandboxSession {
                original,
                ops,
                ensure,
                restricted: OnceLock::new(),
            }))
        }
    }
}

struct GatedNetworkSandboxSession {
    original: Arc<dyn HarnessV1NetworkSandboxSession>,
    ops: DelegatedSandboxSession,
    ensure: Arc<RetryOnFailure<()>>,
    restricted: OnceLock<Arc<dyn SandboxSession>>,
}

#[async_trait]
impl SandboxSession for GatedNetworkSandboxSession {
    fn description(&self) -> String {
        self.original.description()
    }

    async fn read_file(&self, o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
        self.ops.read_file(o).await
    }

    async fn read_binary_file(
        &self,
        o: SandboxReadBinaryFileOptions,
    ) -> HarnessResult<Option<Bytes>> {
        self.ops.read_binary_file(o).await
    }

    async fn read_text_file(&self, o: SandboxReadTextFileOptions) -> HarnessResult<Option<String>> {
        self.ops.read_text_file(o).await
    }

    async fn write_file(&self, o: SandboxWriteFileOptions) -> HarnessResult<()> {
        self.ops.write_file(o).await
    }

    async fn write_binary_file(&self, o: SandboxWriteBinaryFileOptions) -> HarnessResult<()> {
        self.ops.write_binary_file(o).await
    }

    async fn write_text_file(&self, o: SandboxWriteTextFileOptions) -> HarnessResult<()> {
        self.ops.write_text_file(o).await
    }

    async fn stat(&self, o: SandboxStatOptions) -> HarnessResult<Option<SandboxFileStat>> {
        self.ops.stat(o).await
    }

    async fn list_directory(
        &self,
        o: SandboxListDirectoryOptions,
    ) -> HarnessResult<Option<Vec<SandboxDirectoryEntry>>> {
        self.ops.list_directory(o).await
    }

    async fn delete_file(&self, o: SandboxDeleteFileOptions) -> HarnessResult<bool> {
        self.ops.delete_file(o).await
    }

    async fn spawn(&self, o: SandboxSpawnOptions) -> HarnessResult<SandboxProcess> {
        self.ops.spawn(o).await
    }

    async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
        self.ops.run(o).await
    }
}

#[async_trait]
impl HarnessV1NetworkSandboxSession for GatedNetworkSandboxSession {
    fn id(&self) -> String {
        self.original.id()
    }

    fn default_working_directory(&self) -> String {
        self.original.default_working_directory()
    }

    fn ports(&self) -> Vec<u16> {
        self.original.ports()
    }

    async fn get_port_endpoint(
        &self,
        port: u16,
        protocol: Option<HarnessV1PortProtocol>,
    ) -> HarnessResult<HarnessV1PortEndpoint> {
        self.original.get_port_endpoint(port, protocol).await
    }

    async fn stop(&self) -> HarnessResult<()> {
        self.original.stop().await
    }

    async fn destroy(&self) -> HarnessResult<()> {
        self.original.destroy().await
    }

    fn supports_set_network_policy(&self) -> bool {
        self.original.supports_set_network_policy()
    }

    async fn set_network_policy(&self, policy: HarnessV1NetworkPolicy) -> HarnessResult<()> {
        self.original.set_network_policy(policy).await
    }

    fn supports_set_request_transformations(&self) -> bool {
        self.original.supports_set_request_transformations()
    }

    async fn set_request_transformations(
        &self,
        transformations: Vec<HarnessV1RequestTransformation>,
    ) -> HarnessResult<()> {
        self.original
            .set_request_transformations(transformations)
            .await
    }

    fn supports_add_request_transformations(&self) -> bool {
        self.original.supports_add_request_transformations()
    }

    async fn add_request_transformations(
        &self,
        transformations: Vec<HarnessV1RequestTransformation>,
    ) -> HarnessResult<()> {
        self.original
            .add_request_transformations(transformations)
            .await
    }

    fn supports_set_ports(&self) -> bool {
        self.original.supports_set_ports()
    }

    async fn set_ports(
        &self,
        ports: Vec<u16>,
        abort_signal: Option<CancellationToken>,
    ) -> HarnessResult<()> {
        self.original.set_ports(ports, abort_signal).await
    }

    fn restricted(&self) -> Arc<dyn SandboxSession> {
        self.restricted
            .get_or_init(|| Arc::new(gate(self.original.restricted(), self.ensure.clone())))
            .clone()
    }
}

/// The setup step `defer_sandbox_session` runs on the created session.
pub type SetupSandboxSessionFn =
    Arc<dyn Fn(HarnessV1SandboxSessionRef) -> BoxFuture<'static, HarnessResult<()>> + Send + Sync>;

/// `deferSandboxSession`: creates and sets up the sandbox on the first file
/// or process operation. A failed creation or setup runs again on the next
/// operation; a created session is kept across setup failures.
pub fn defer_sandbox_session(
    default_working_directory: String,
    create_sandbox_session: CreateSandboxSessionFn,
    setup_sandbox_session: SetupSandboxSessionFn,
) -> Arc<dyn SandboxSession> {
    let created: Arc<Mutex<Option<HarnessV1SandboxSessionRef>>> = Arc::default();
    let ready = {
        let created = created.clone();
        Arc::new(RetryOnFailure::new(Arc::new(move || {
            let created = created.clone();
            let create = create_sandbox_session.clone();
            let setup = setup_sandbox_session.clone();
            Box::pin(async move {
                let existing = created.lock().clone();
                let session = match existing {
                    Some(s) => s,
                    None => {
                        let s = create(None).await?;
                        created.lock().get_or_insert(s).clone()
                    }
                };
                setup(session.clone()).await?;
                Ok(get_restricted_sandbox_session(&session))
            }) as BoxFuture<'static, HarnessResult<Arc<dyn SandboxSession>>>
        })))
    };
    Arc::new(delegate_sandbox_operations(
        Arc::new(move || {
            let ready = ready.clone();
            Box::pin(async move { ready.get().await })
        }),
        Arc::new(move || match &*created.lock() {
            Some(HarnessV1SandboxSessionRef::Network(s)) => s.description(),
            Some(HarnessV1SandboxSessionRef::Basic(s)) => s.description(),
            None => format!(
                "Sandbox with the working directory {default_working_directory}. It starts on first use."
            ),
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::test_support::MockSandbox;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn setup_runs_once_before_the_first_operation_and_retries_after_failure() {
        let calls = Arc::new(AtomicUsize::new(0));
        let setup_calls = calls.clone();
        let mock = MockSandbox::with_result(SandboxRunResult::default());
        let gated = defer_sandbox_setup(
            HarnessV1SandboxSessionRef::Basic(Arc::new(mock.clone())),
            Arc::new(move || {
                let n = setup_calls.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    if n == 0 {
                        Err("setup failed".into())
                    } else {
                        Ok(())
                    }
                })
            }),
        );
        let session = gated.session();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let err = session
            .run(SandboxRunOptions {
                process: SandboxProcessOptions::new("ls"),
                max_output_length: None,
            })
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "setup failed");
        assert!(mock.state.lock().runs.is_empty());
        for _ in 0..2 {
            session
                .run(SandboxRunOptions {
                    process: SandboxProcessOptions::new("ls"),
                    max_output_length: None,
                })
                .await
                .unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(mock.state.lock().runs, vec!["ls", "ls"]);
        assert_eq!(session.description(), "mock sandbox");
    }

    #[tokio::test]
    async fn deferred_session_is_created_on_first_use_and_kept_across_setup_failures() {
        let creates = Arc::new(AtomicUsize::new(0));
        let setups = Arc::new(AtomicUsize::new(0));
        let mock = MockSandbox::with_result(SandboxRunResult::default());
        let (c, s, m) = (creates.clone(), setups.clone(), mock.clone());
        let session = defer_sandbox_session(
            "/work".into(),
            Arc::new(move |_| {
                c.fetch_add(1, Ordering::SeqCst);
                let m = m.clone();
                Box::pin(async move { Ok(HarnessV1SandboxSessionRef::Basic(Arc::new(m))) })
            }),
            Arc::new(move |_| {
                let n = s.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    if n == 0 {
                        Err("setup failed".into())
                    } else {
                        Ok(())
                    }
                })
            }),
        );
        assert_eq!(
            session.description(),
            "Sandbox with the working directory /work. It starts on first use."
        );
        let run = || SandboxRunOptions {
            process: SandboxProcessOptions::new("ls"),
            max_output_length: None,
        };
        assert!(session.run(run()).await.is_err());
        session.run(run()).await.unwrap();
        session.run(run()).await.unwrap();
        assert_eq!(creates.load(Ordering::SeqCst), 1);
        assert_eq!(setups.load(Ordering::SeqCst), 2);
        assert_eq!(session.description(), "mock sandbox");
    }
}
