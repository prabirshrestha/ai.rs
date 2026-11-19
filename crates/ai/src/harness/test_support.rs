//! Test-only sandbox doubles, mirroring the ad-hoc `as unknown as
//! Experimental_SandboxSession` objects in Vercel's tests.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;

use crate::harness::error::HarnessResult;
use crate::harness::sandbox::*;

/// State a [`MockSandbox`] records.
#[derive(Debug, Default)]
pub struct MockState {
    pub files: BTreeMap<String, String>,
    pub directories: BTreeSet<String>,
    pub runs: Vec<String>,
    pub writes: Vec<(String, String)>,
}

type ReadTextHandler = Arc<dyn Fn(&str) -> HarnessResult<Option<String>> + Send + Sync>;

type RunHandler = Arc<
    dyn Fn(&SandboxRunOptions, &mut MockState) -> HarnessResult<SandboxRunResult> + Send + Sync,
>;

/// In-memory text-file sandbox with a programmable `run`.
#[derive(Clone)]
pub struct MockSandbox {
    pub state: Arc<Mutex<MockState>>,
    run: RunHandler,
    read_text: Option<ReadTextHandler>,
    /// Paths `read_text_file` was called with.
    pub text_reads: Arc<Mutex<Vec<String>>>,
}

impl MockSandbox {
    pub fn new(
        run: impl Fn(&SandboxRunOptions, &mut MockState) -> HarnessResult<SandboxRunResult>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState::default())),
            run: Arc::new(run),
            read_text: None,
            text_reads: Arc::default(),
        }
    }

    /// Replace `read_text_file` with `handler`.
    pub fn with_read_text(
        mut self,
        handler: impl Fn(&str) -> HarnessResult<Option<String>> + Send + Sync + 'static,
    ) -> Self {
        self.read_text = Some(Arc::new(handler));
        self
    }

    /// A sandbox whose `run` always returns `result`.
    pub fn with_result(result: SandboxRunResult) -> Self {
        Self::new(move |_, _| Ok(result.clone()))
    }
}

#[async_trait]
impl SandboxSession for MockSandbox {
    fn description(&self) -> String {
        "mock sandbox".into()
    }

    async fn read_file(&self, _o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
        unimplemented!("read_file")
    }

    async fn read_binary_file(
        &self,
        o: SandboxReadBinaryFileOptions,
    ) -> HarnessResult<Option<Bytes>> {
        Ok(self
            .state
            .lock()
            .files
            .get(&o.path)
            .map(|s| Bytes::from(s.clone().into_bytes())))
    }

    async fn read_text_file(&self, o: SandboxReadTextFileOptions) -> HarnessResult<Option<String>> {
        self.text_reads.lock().push(o.path.clone());
        if let Some(read_text) = &self.read_text {
            return read_text(&o.path);
        }
        Ok(self.state.lock().files.get(&o.path).cloned())
    }

    async fn write_file(&self, _o: SandboxWriteFileOptions) -> HarnessResult<()> {
        unimplemented!("write_file")
    }

    async fn write_binary_file(&self, o: SandboxWriteBinaryFileOptions) -> HarnessResult<()> {
        let content = String::from_utf8_lossy(&o.content).into_owned();
        let mut s = self.state.lock();
        s.writes.push((o.path.clone(), content.clone()));
        s.files.insert(o.path, content);
        Ok(())
    }

    async fn write_text_file(&self, o: SandboxWriteTextFileOptions) -> HarnessResult<()> {
        let mut s = self.state.lock();
        s.writes.push((o.path.clone(), o.content.clone()));
        s.files.insert(o.path, o.content);
        Ok(())
    }

    async fn stat(&self, _o: SandboxStatOptions) -> HarnessResult<Option<SandboxFileStat>> {
        unimplemented!("stat")
    }

    async fn list_directory(
        &self,
        _o: SandboxListDirectoryOptions,
    ) -> HarnessResult<Option<Vec<SandboxDirectoryEntry>>> {
        unimplemented!("list_directory")
    }

    async fn delete_file(&self, o: SandboxDeleteFileOptions) -> HarnessResult<bool> {
        Ok(self.state.lock().files.remove(&o.path).is_some())
    }

    async fn spawn(&self, _o: SandboxSpawnOptions) -> HarnessResult<SandboxProcess> {
        unimplemented!("spawn")
    }

    async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
        let mut s = self.state.lock();
        s.runs.push(o.process.command.clone());
        (self.run)(&o, &mut s)
    }
}

/// A process double: scripted stdout, recorded kills.
pub struct MockProcessControl {
    pub kills: std::sync::atomic::AtomicUsize,
    pub exit_code: Option<i32>,
}

#[async_trait]
impl SandboxProcessControl for MockProcessControl {
    async fn wait(&self) -> HarnessResult<SandboxExitStatus> {
        match self.exit_code {
            Some(exit_code) => Ok(SandboxExitStatus { exit_code }),
            None => futures::future::pending().await,
        }
    }

    async fn kill(&self) -> HarnessResult<()> {
        self.kills.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

/// A process whose stdout yields `chunks` then ends, or stays open when
/// `chunks` is `None`.
pub fn mock_process(chunks: Option<Vec<&str>>) -> (SandboxProcess, Arc<MockProcessControl>) {
    use futures::StreamExt;
    let stdout: ByteStream = match chunks {
        Some(chunks) => {
            let chunks: Vec<HarnessResult<Bytes>> = chunks
                .into_iter()
                .map(|chunk| Ok(Bytes::from(chunk.to_string())))
                .collect();
            futures::stream::iter(chunks).boxed()
        }
        None => futures::stream::pending().boxed(),
    };
    let control = Arc::new(MockProcessControl {
        kills: Default::default(),
        exit_code: Some(0),
    });
    (
        SandboxProcess {
            pid: None,
            stdout,
            stderr: futures::stream::empty().boxed(),
            stdin: None,
            control: control.clone(),
        },
        control,
    )
}

/// A network sandbox over a [`MockSandbox`], recording `getPortEndpoint`.
pub struct MockNetworkSandbox {
    pub inner: MockSandbox,
    pub ports: Vec<u16>,
    pub endpoint_calls: Mutex<Vec<(u16, Option<crate::harness::HarnessPortProtocol>)>>,
}

impl MockNetworkSandbox {
    pub fn new(ports: Vec<u16>) -> Self {
        Self {
            inner: MockSandbox::with_result(SandboxRunResult::default()),
            ports,
            endpoint_calls: Mutex::default(),
        }
    }
}

#[async_trait]
impl SandboxSession for MockNetworkSandbox {
    fn description(&self) -> String {
        self.inner.description()
    }
    async fn read_file(&self, o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
        self.inner.read_file(o).await
    }

    async fn read_binary_file(
        &self,
        o: SandboxReadBinaryFileOptions,
    ) -> HarnessResult<Option<Bytes>> {
        self.inner.read_binary_file(o).await
    }

    async fn read_text_file(&self, o: SandboxReadTextFileOptions) -> HarnessResult<Option<String>> {
        self.inner.read_text_file(o).await
    }

    async fn write_file(&self, o: SandboxWriteFileOptions) -> HarnessResult<()> {
        self.inner.write_file(o).await
    }

    async fn write_binary_file(&self, o: SandboxWriteBinaryFileOptions) -> HarnessResult<()> {
        self.inner.write_binary_file(o).await
    }

    async fn write_text_file(&self, o: SandboxWriteTextFileOptions) -> HarnessResult<()> {
        self.inner.write_text_file(o).await
    }

    async fn stat(&self, o: SandboxStatOptions) -> HarnessResult<Option<SandboxFileStat>> {
        self.inner.stat(o).await
    }

    async fn list_directory(
        &self,
        o: SandboxListDirectoryOptions,
    ) -> HarnessResult<Option<Vec<SandboxDirectoryEntry>>> {
        self.inner.list_directory(o).await
    }

    async fn delete_file(&self, o: SandboxDeleteFileOptions) -> HarnessResult<bool> {
        self.inner.delete_file(o).await
    }

    async fn spawn(&self, o: SandboxSpawnOptions) -> HarnessResult<SandboxProcess> {
        self.inner.spawn(o).await
    }

    async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
        self.inner.run(o).await
    }
}

#[async_trait]
impl crate::harness::HarnessNetworkSandboxSession for MockNetworkSandbox {
    fn id(&self) -> String {
        "sandbox-1".into()
    }

    fn default_working_directory(&self) -> String {
        "/workspace".into()
    }

    fn ports(&self) -> Vec<u16> {
        self.ports.clone()
    }

    async fn get_port_endpoint(
        &self,
        port: u16,
        protocol: Option<crate::harness::HarnessPortProtocol>,
    ) -> HarnessResult<crate::harness::HarnessPortEndpoint> {
        self.endpoint_calls.lock().push((port, protocol));
        Ok(crate::harness::HarnessPortEndpoint {
            url: format!("wss://sandbox.example/{port}"),
            headers: Some(indexmap::IndexMap::from([(
                "x-access-token".to_string(),
                "traffic-token".to_string(),
            )])),
        })
    }

    async fn stop(&self) -> HarnessResult<()> {
        Ok(())
    }

    async fn destroy(&self) -> HarnessResult<()> {
        Ok(())
    }

    fn restricted(&self) -> Arc<dyn SandboxSession> {
        Arc::new(self.inner.clone())
    }
}

/// Values between single quotes, as the TS tests' `quotedValues`.
pub fn quoted_values(command: &str) -> Vec<String> {
    let re = regex::Regex::new(r"'([^']*)'").unwrap();
    re.captures_iter(command)
        .map(|c| c[1].to_string())
        .collect()
}
