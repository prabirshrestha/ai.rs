//! Test-only sandbox doubles, mirroring the ad-hoc `as unknown as
//! Experimental_SandboxSession` objects in Vercel's tests.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;

use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::*;

/// State a [`MockSandbox`] records.
#[derive(Debug, Default)]
pub struct MockState {
    pub files: BTreeMap<String, String>,
    pub directories: BTreeSet<String>,
    pub runs: Vec<String>,
    pub writes: Vec<(String, String)>,
}

type RunHandler = Arc<
    dyn Fn(&SandboxRunOptions, &mut MockState) -> HarnessResult<SandboxRunResult> + Send + Sync,
>;

/// In-memory text-file sandbox with a programmable `run`.
#[derive(Clone)]
pub struct MockSandbox {
    pub state: Arc<Mutex<MockState>>,
    run: RunHandler,
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
        }
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

/// Values between single quotes, as the TS tests' `quotedValues`.
pub fn quoted_values(command: &str) -> Vec<String> {
    let re = regex::Regex::new(r"'([^']*)'").unwrap();
    re.captures_iter(command)
        .map(|c| c[1].to_string())
        .collect()
}
