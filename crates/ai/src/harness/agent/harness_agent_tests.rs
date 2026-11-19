//! Port of `agent/harness-agent.test.ts`: the shared doubles. The tests
//! live in the `harness_agent_tests_*.rs` files mounted below.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::json;

pub(crate) use crate::harness::agent::internal::run_prompt::tests::{
    Gate, contains_matching, matches, of_type, p, parts, to_json, tool, tools, types, wait_for,
    z_object,
};
pub(crate) use crate::harness::agent::harness_agent::*;
pub(crate) use crate::harness::agent::harness_agent_session::*;
pub(crate) use crate::harness::agent::settings::*;
pub(crate) use crate::harness::errors::{HarnessError, HarnessResult};
pub(crate) use crate::harness::provider::JsonValue;
pub(crate) use crate::harness::sandbox::*;
pub(crate) use crate::harness::sdk::tool::ToolSet;
pub(crate) use crate::harness::spec::bootstrap::HarnessV1Bootstrap;
pub(crate) use crate::harness::spec::harness::*;
pub(crate) use crate::harness::spec::lifecycle_state::{
    ContinueTurnType, HarnessV1ContinueTurnState, HarnessV1ResumeSessionState, ResumeSessionType,
};
pub(crate) use crate::harness::spec::literal::HarnessV1SpecificationVersion;
pub(crate) use crate::harness::spec::stream_part::HarnessV1StreamPart;
pub(crate) use crate::harness::spec::tools::HarnessV1BuiltinTool;
pub(crate) use crate::harness::spec::{HarnessV1NetworkSandboxSession, HarnessV1PortEndpoint, HarnessV1PortProtocol};

pub(crate) use crate::harness::utils::plain_error;

#[path = "harness_agent_tests_basics.rs"]
mod basics;
#[path = "harness_agent_tests_turns.rs"]
mod turns;
#[path = "harness_agent_tests_outputs.rs"]
mod outputs;
#[path = "harness_agent_tests_continue.rs"]
mod continuation;
#[path = "harness_agent_tests_sandbox.rs"]
mod sandbox;

// ─── mockHarness ───────────────────────────────────────────────────────

pub(crate) type ScriptFn = Arc<dyn Fn() -> Vec<HarnessV1StreamPart> + Send + Sync>;
pub(crate) type PromptDoneFn =
    Arc<dyn Fn(&HarnessV1PromptTurnOptions) -> BoxFuture<'static, HarnessResult<()>> + Send + Sync>;
pub(crate) type AsyncHookFn = Arc<dyn Fn() -> BoxFuture<'static, HarnessResult<()>> + Send + Sync>;
pub(crate) type SubmitHookFn = Arc<
    dyn Fn(&HarnessV1SubmitToolResult) -> BoxFuture<'static, HarnessResult<()>> + Send + Sync,
>;
pub(crate) type ReadHistoryFn = Arc<
    dyn Fn(Option<String>) -> BoxFuture<'static, HarnessResult<HarnessV1ReadHistoryResult>>
        + Send
        + Sync,
>;

/// Options of `mockHarness`.
#[derive(Clone, Default)]
pub(crate) struct MockHarnessOptions {
    pub script: Option<ScriptFn>,
    pub builtin_tools: ToolSet,
    pub supports_builtin_tool_approvals: Option<bool>,
    pub supports_builtin_tool_filtering: Option<bool>,
    pub on_do_start: Option<Arc<dyn Fn(&HarnessV1StartOptions) + Send + Sync>>,
    pub on_prompt_turn: Option<Arc<dyn Fn(&HarnessV1PromptTurnOptions) -> HarnessResult<()> + Send + Sync>>,
    pub prompt_done: Option<PromptDoneFn>,
    pub supports_steering: bool,
    pub on_suspend_turn: Option<AsyncHookFn>,
    pub on_submit_tool_result: Option<SubmitHookFn>,
    pub do_read_history: Option<ReadHistoryFn>,
    pub continue_script: Option<ScriptFn>,
    /// `withBootstrapRecipe(harness)`.
    pub bootstrap: Option<HarnessV1Bootstrap>,
}

impl MockHarnessOptions {
    pub fn script(script: impl Fn() -> Vec<HarnessV1StreamPart> + Send + Sync + 'static) -> Self {
        Self {
            script: Some(Arc::new(script)),
            ..Self::default()
        }
    }

    pub fn with_continue(
        mut self,
        script: impl Fn() -> Vec<HarnessV1StreamPart> + Send + Sync + 'static,
    ) -> Self {
        self.continue_script = Some(Arc::new(script));
        self
    }

    pub fn build(self) -> Mock {
        mock_harness(self)
    }
}

/// What a mock harness records (the `vi.fn` call logs).
#[derive(Default)]
pub(crate) struct MockRecord {
    pub prompts: Mutex<Vec<HarnessV1Prompt>>,
    pub prompt_turns: Mutex<Vec<HarnessV1PromptTurnOptions>>,
    pub continue_turns: Mutex<Vec<HarnessV1ContinueTurnOptions>>,
    pub tool_results: Mutex<Vec<HarnessV1SubmitToolResult>>,
    pub tool_approvals: Mutex<Vec<HarnessV1SubmitToolApproval>>,
    pub user_messages: Mutex<Vec<String>>,
    pub start_calls: Mutex<Vec<HarnessV1StartOptions>>,
    pub get_bootstrap_calls: AtomicUsize,
    pub detach_calls: AtomicUsize,
    pub stop_calls: AtomicUsize,
    pub destroy_calls: AtomicUsize,
    pub compact_calls: Mutex<Vec<Option<String>>>,
    pub suspend_calls: AtomicUsize,
    /// Errors the next `doDetach` / `doStop` / `doStart` calls reject with
    /// (`mockRejectedValueOnce`).
    pub detach_errors: Mutex<Vec<String>>,
    pub stop_errors: Mutex<Vec<String>>,
    pub start_errors: Mutex<Vec<String>>,
}

pub(crate) struct MockHarnessImpl {
    options: MockHarnessOptions,
    builtin_tools: IndexMap<String, HarnessV1BuiltinTool>,
    record: Arc<MockRecord>,
}

/// The result of `mockHarness`.
#[derive(Clone)]
pub(crate) struct Mock {
    pub harness: Arc<MockHarnessImpl>,
    pub record: Arc<MockRecord>,
}

impl Mock {
    pub fn harness(&self) -> Arc<dyn HarnessV1> {
        self.harness.clone()
    }
    pub fn start_call(&self, index: usize) -> HarnessV1StartOptions {
        self.record.start_calls.lock()[index].clone()
    }
    pub fn start_count(&self) -> usize {
        self.record.start_calls.lock().len()
    }
    pub fn prompts(&self) -> Vec<HarnessV1Prompt> {
        self.record.prompts.lock().clone()
    }
    pub fn tool_results(&self) -> Vec<HarnessV1SubmitToolResult> {
        self.record.tool_results.lock().clone()
    }
    pub fn tool_approvals(&self) -> Vec<HarnessV1SubmitToolApproval> {
        self.record.tool_approvals.lock().clone()
    }
    pub fn user_messages(&self) -> Vec<String> {
        self.record.user_messages.lock().clone()
    }
    pub fn continue_count(&self) -> usize {
        self.record.continue_turns.lock().len()
    }
    pub fn count(counter: &AtomicUsize) -> usize {
        counter.load(Ordering::SeqCst)
    }
}

pub(crate) fn mock_harness(options: MockHarnessOptions) -> Mock {
    let record = Arc::new(MockRecord::default());
    let builtin_tools = options
        .builtin_tools
        .iter()
        .map(|(n, t)| (n.clone(), HarnessV1BuiltinTool::from(t.clone())))
        .collect();
    Mock {
        harness: Arc::new(MockHarnessImpl {
            options,
            builtin_tools,
            record: record.clone(),
        }),
        record,
    }
}

/// The mock `resumeState`.
pub(crate) fn mock_resume_state() -> HarnessV1ResumeSessionState {
    HarnessV1ResumeSessionState {
        r#type: ResumeSessionType,
        harness_id: "mock".into(),
        specification_version: HarnessV1SpecificationVersion,
        data: json!({}),
        continue_from: None,
    }
}

/// The mock `continueState`.
pub(crate) fn mock_continue_state() -> HarnessV1ContinueTurnState {
    HarnessV1ContinueTurnState {
        r#type: ContinueTurnType,
        harness_id: "mock".into(),
        specification_version: HarnessV1SpecificationVersion,
        data: json!({}),
        pending_tool_approvals: None,
        pending_tool_results: None,
        turn_settings: None,
    }
}

#[async_trait]
impl HarnessV1 for MockHarnessImpl {
    fn harness_id(&self) -> &str {
        "mock"
    }
    fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool> {
        &self.builtin_tools
    }
    fn supports_builtin_tool_approvals(&self) -> bool {
        self.options.supports_builtin_tool_approvals.unwrap_or(false)
    }
    fn supports_builtin_tool_filtering(&self) -> bool {
        self.options.supports_builtin_tool_filtering.unwrap_or(false)
    }
    fn supports_get_bootstrap(&self) -> bool {
        self.options.bootstrap.is_some()
    }
    async fn get_bootstrap(
        &self,
        _abort_signal: Option<tokio_util::sync::CancellationToken>,
    ) -> HarnessResult<HarnessV1Bootstrap> {
        self.record.get_bootstrap_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.options.bootstrap.clone().expect("bootstrap recipe"))
    }
    async fn do_start(&self, options: HarnessV1StartOptions) -> HarnessResult<Box<dyn HarnessV1Session>> {
        self.record.start_calls.lock().push(options.clone());
        if let Some(f) = &self.options.on_do_start {
            f(&options);
        }
        if let Some(message) = pop_error(&self.record.start_errors) {
            return Err(plain_error(message));
        }
        Ok(Box::new(MockSession {
            options: self.options.clone(),
            record: self.record.clone(),
        }))
    }
}

fn pop_error(errors: &Mutex<Vec<String>>) -> Option<String> {
    let mut errors = errors.lock();
    (!errors.is_empty()).then(|| errors.remove(0))
}

struct MockSession {
    options: MockHarnessOptions,
    record: Arc<MockRecord>,
}

struct MockControl {
    options: MockHarnessOptions,
    record: Arc<MockRecord>,
    done: Shared<BoxFuture<'static, Result<(), String>>>,
}

#[async_trait]
impl HarnessV1PromptControl for MockControl {
    async fn submit_tool_result(&self, input: HarnessV1SubmitToolResult) -> HarnessResult<()> {
        if let Some(f) = &self.options.on_submit_tool_result {
            f(&input).await?;
        }
        self.record.tool_results.lock().push(input);
        Ok(())
    }
    fn supports_submit_tool_approval(&self) -> bool {
        true
    }
    async fn submit_tool_approval(&self, input: HarnessV1SubmitToolApproval) -> HarnessResult<()> {
        self.record.tool_approvals.lock().push(input);
        Ok(())
    }
    fn supports_submit_user_message(&self) -> bool {
        self.options.supports_steering
    }
    async fn submit_user_message(&self, text: String) -> HarnessResult<()> {
        self.record.user_messages.lock().push(text);
        Ok(())
    }
    fn done(&self) -> BoxFuture<'static, HarnessResult<()>> {
        let done = self.done.clone();
        Box::pin(async move { done.await.map_err(plain_error) })
    }
}

/// Emit `events`.
///
/// [rust] TS emits them in a microtask, before `done` settles; the turn's
/// channel buffers them, so emitting now is the same.
fn emit_later(emit: HarnessV1Emit, events: Vec<HarnessV1StreamPart>) {
    for event in events {
        emit(event);
    }
}

fn shared_done(
    done: Option<BoxFuture<'static, HarnessResult<()>>>,
) -> Shared<BoxFuture<'static, Result<(), String>>> {
    match done {
        Some(done) => done.map(|r| r.map_err(|e| e.to_string())).boxed().shared(),
        None => futures::future::ready(Ok(())).boxed().shared(),
    }
}

#[async_trait]
impl HarnessV1Session for MockSession {
    fn session_id(&self) -> String {
        "mock-session-1".into()
    }
    fn is_resume(&self) -> bool {
        false
    }
    async fn do_prompt_turn(
        &self,
        opts: HarnessV1PromptTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        self.record.prompts.lock().push(opts.prompt.clone());
        self.record.prompt_turns.lock().push(opts.clone());
        if let Some(f) = &self.options.on_prompt_turn {
            f(&opts)?;
        }
        let done = shared_done(self.options.prompt_done.as_ref().map(|f| f(&opts)));
        let events = self.options.script.as_ref().map(|s| s()).unwrap_or_default();
        emit_later(opts.emit.clone(), events);
        Ok(Box::new(MockControl {
            options: self.options.clone(),
            record: self.record.clone(),
            done,
        }))
    }
    async fn do_compact(&self, custom_instructions: Option<String>) -> HarnessResult<()> {
        self.record.compact_calls.lock().push(custom_instructions);
        Ok(())
    }
    fn supports_read_history(&self) -> bool {
        self.options.do_read_history.is_some()
    }
    async fn do_read_history(&self, since: Option<String>) -> HarnessResult<HarnessV1ReadHistoryResult> {
        (self.options.do_read_history.as_ref().expect("doReadHistory"))(since).await
    }
    async fn do_continue_turn(
        &self,
        opts: HarnessV1ContinueTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        self.record.continue_turns.lock().push(opts.clone());
        let events = self
            .options
            .continue_script
            .as_ref()
            .map(|s| s())
            .unwrap_or_default();
        emit_later(opts.emit.clone(), events);
        Ok(Box::new(MockControl {
            options: self.options.clone(),
            record: self.record.clone(),
            done: shared_done(None),
        }))
    }
    async fn do_suspend_turn(&self) -> HarnessResult<HarnessV1ContinueTurnState> {
        self.record.suspend_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(f) = &self.options.on_suspend_turn {
            f().await?;
        }
        Ok(mock_continue_state())
    }
    async fn do_detach(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        self.record.detach_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(message) = pop_error(&self.record.detach_errors) {
            return Err(plain_error(message));
        }
        Ok(mock_resume_state())
    }
    async fn do_stop(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        self.record.stop_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(message) = pop_error(&self.record.stop_errors) {
            return Err(plain_error(message));
        }
        Ok(mock_resume_state())
    }
    async fn do_destroy(&self) -> HarnessResult<()> {
        self.record.destroy_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

// ─── makeSandboxSession ────────────────────────────────────────────────

pub(crate) type RunFn = Arc<dyn Fn(&SandboxRunOptions) -> HarnessResult<SandboxRunResult> + Send + Sync>;
pub(crate) type ReadTextFn = Arc<dyn Fn(&str) -> HarnessResult<Option<String>> + Send + Sync>;

/// A recorded `run` call: `{ command, env }`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RunCall {
    pub command: String,
    pub env: Option<IndexMap<String, String>>,
}

impl RunCall {
    pub fn new(command: &str, env: &[(&str, &str)]) -> Self {
        Self {
            command: command.into(),
            env: (!env.is_empty()).then(|| {
                env.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect()
            }),
        }
    }
}

/// The file and process surface of a sandbox double: a programmable `run`
/// and `readTextFile` over an in-memory file map. Every operation is
/// appended to a shared event log (`invocationCallOrder`).
#[derive(Clone)]
pub(crate) struct ScriptedSandbox {
    pub label: String,
    pub run: RunFn,
    pub read_text: Option<ReadTextFn>,
    pub files: Arc<Mutex<IndexMap<String, String>>>,
    pub runs: Arc<Mutex<Vec<RunCall>>>,
    pub reads: Arc<Mutex<Vec<String>>>,
    pub writes: Arc<Mutex<Vec<String>>>,
    pub log: Arc<Mutex<Vec<String>>>,
}

/// `run` answering the HOME and `pwd` probes.
pub(crate) fn probe_run(o: &SandboxRunOptions) -> SandboxRunResult {
    let stdout = match o.process.command.as_str() {
        "printf \"%s\" \"$HOME\"" => "/home/agent",
        "pwd" => "/work\n",
        _ => "",
    };
    SandboxRunResult {
        exit_code: 0,
        stdout: stdout.into(),
        stderr: String::new(),
        ..Default::default()
    }
}

/// `run` returning an empty success.
pub(crate) fn empty_run() -> RunFn {
    Arc::new(|_| Ok(SandboxRunResult::default()))
}

impl ScriptedSandbox {
    pub fn new(label: &str, log: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            label: label.into(),
            run: Arc::new(|o| Ok(probe_run(o))),
            read_text: None,
            files: Arc::default(),
            runs: Arc::default(),
            reads: Arc::default(),
            writes: Arc::default(),
            log,
        }
    }

    pub fn with_run(mut self, run: RunFn) -> Self {
        self.run = run;
        self
    }

    pub fn with_read_text(mut self, read_text: impl Fn(&str) -> HarnessResult<Option<String>> + Send + Sync + 'static) -> Self {
        self.read_text = Some(Arc::new(read_text));
        self
    }

    /// The same operations under another label (TS's separate
    /// `restricted()` object sharing `run` / `readTextFile`).
    pub fn relabeled(&self, label: &str) -> Self {
        let mut view = self.clone();
        view.label = label.into();
        view
    }
}

#[async_trait]
impl SandboxSession for ScriptedSandbox {
    fn description(&self) -> String {
        self.label.clone()
    }
    async fn read_file(&self, _o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
        unimplemented!("read_file")
    }
    async fn read_binary_file(&self, o: SandboxReadBinaryFileOptions) -> HarnessResult<Option<bytes::Bytes>> {
        Ok(self.files.lock().get(&o.path).map(|s| bytes::Bytes::from(s.clone().into_bytes())))
    }
    async fn read_text_file(&self, o: SandboxReadTextFileOptions) -> HarnessResult<Option<String>> {
        self.log.lock().push(format!("read:{}", o.path));
        self.reads.lock().push(o.path.clone());
        if let Some(read_text) = &self.read_text {
            return read_text(&o.path);
        }
        Ok(self.files.lock().get(&o.path).cloned())
    }
    async fn write_file(&self, _o: SandboxWriteFileOptions) -> HarnessResult<()> {
        unimplemented!("write_file")
    }
    async fn write_binary_file(&self, o: SandboxWriteBinaryFileOptions) -> HarnessResult<()> {
        self.log.lock().push(format!("write:{}", o.path));
        self.writes.lock().push(o.path.clone());
        self.files.lock().insert(o.path, String::from_utf8_lossy(&o.content).into_owned());
        Ok(())
    }
    async fn write_text_file(&self, o: SandboxWriteTextFileOptions) -> HarnessResult<()> {
        self.log.lock().push(format!("write:{}", o.path));
        self.writes.lock().push(o.path.clone());
        self.files.lock().insert(o.path, o.content);
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
        Ok(self.files.lock().shift_remove(&o.path).is_some())
    }
    async fn spawn(&self, _o: SandboxSpawnOptions) -> HarnessResult<SandboxProcess> {
        unimplemented!("spawn")
    }
    async fn run(&self, o: SandboxRunOptions) -> HarnessResult<SandboxRunResult> {
        self.log.lock().push(format!("run:{}", o.process.command));
        self.runs.lock().push(RunCall {
            command: o.process.command.clone(),
            env: o.process.env.clone(),
        });
        (self.run)(&o)
    }
}

/// `makeSandboxSession()`: a network sandbox over a [`ScriptedSandbox`].
#[derive(Clone)]
pub(crate) struct TestSandbox {
    pub inner: ScriptedSandbox,
    /// What `restricted()` returns.
    pub restricted_view: ScriptedSandbox,
    pub id: String,
    pub default_working_directory: String,
    pub stop_calls: Arc<AtomicUsize>,
    pub destroy_calls: Arc<AtomicUsize>,
}

impl TestSandbox {
    pub fn new() -> Self {
        Self::with_inner(ScriptedSandbox::new("mock sandbox", Arc::default()))
    }

    pub fn with_inner(inner: ScriptedSandbox) -> Self {
        Self {
            restricted_view: inner.clone(),
            inner,
            id: "sandbox".into(),
            default_working_directory: "/work".into(),
            stop_calls: Arc::default(),
            destroy_calls: Arc::default(),
        }
    }

    pub fn with_restricted(mut self, view: ScriptedSandbox) -> Self {
        self.restricted_view = view;
        self
    }

    pub fn sref(&self) -> HarnessV1SandboxSessionRef {
        HarnessV1SandboxSessionRef::Network(Arc::new(self.clone()))
    }

    pub fn input(&self) -> Option<HarnessAgentSandboxSessionInput> {
        Some(self.sref().into())
    }

    pub fn runs(&self) -> Vec<RunCall> {
        self.inner.runs.lock().clone()
    }

    pub fn stops(&self) -> usize {
        self.stop_calls.load(Ordering::SeqCst)
    }

    pub fn destroys(&self) -> usize {
        self.destroy_calls.load(Ordering::SeqCst)
    }
}

/// `makeSandboxSession()` as the `sandboxSession` of `createSession`.
pub(crate) fn make_sandbox_session() -> TestSandbox {
    TestSandbox::new()
}

#[async_trait]
impl SandboxSession for TestSandbox {
    fn description(&self) -> String {
        self.inner.description()
    }
    async fn read_file(&self, o: SandboxReadFileOptions) -> HarnessResult<Option<ByteStream>> {
        self.inner.read_file(o).await
    }
    async fn read_binary_file(&self, o: SandboxReadBinaryFileOptions) -> HarnessResult<Option<bytes::Bytes>> {
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
impl HarnessV1NetworkSandboxSession for TestSandbox {
    fn id(&self) -> String {
        self.id.clone()
    }
    fn default_working_directory(&self) -> String {
        self.default_working_directory.clone()
    }
    fn ports(&self) -> Vec<u16> {
        Vec::new()
    }
    async fn get_port_endpoint(
        &self,
        _port: u16,
        _protocol: Option<HarnessV1PortProtocol>,
    ) -> HarnessResult<HarnessV1PortEndpoint> {
        Ok(HarnessV1PortEndpoint {
            url: "ws://example.test/".into(),
            headers: None,
        })
    }
    async fn stop(&self) -> HarnessResult<()> {
        self.stop_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn destroy(&self) -> HarnessResult<()> {
        self.destroy_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn restricted(&self) -> Arc<dyn SandboxSession> {
        Arc::new(self.restricted_view.clone())
    }
}

/// `withBootstrapRecipe`'s recipe.
pub(crate) fn bootstrap_recipe() -> HarnessV1Bootstrap {
    HarnessV1Bootstrap {
        harness_id: "mock".into(),
        bootstrap_dir: ".harness-bootstrap/mock".into(),
        files: Vec::new(),
        commands: Vec::new(),
    }
}

/// `getStartedSandbox(doStart)`.
pub(crate) fn get_started_sandbox(mock: &Mock) -> HarnessV1SandboxSessionRef {
    mock.start_call(0).sandbox_session
}

/// The started sandbox's own surface: the network session itself, or the
/// basic session.
pub(crate) fn started_session(mock: &Mock) -> Arc<dyn SandboxSession> {
    match get_started_sandbox(mock) {
        HarnessV1SandboxSessionRef::Network(session) => session,
        HarnessV1SandboxSessionRef::Basic(session) => session,
    }
}

/// `touchStartedSandbox(doStart)`.
pub(crate) async fn touch_started_sandbox(mock: &Mock) -> HarnessResult<()> {
    started_session(mock)
        .read_text_file(SandboxReadTextFileOptions {
            path: "/work/touch.txt".into(),
            ..Default::default()
        })
        .await
        .map(|_| ())
}

// ─── stream fixtures ───────────────────────────────────────────────────

pub(crate) fn zero_usage() -> JsonValue {
    json!({ "inputTokens": {}, "outputTokens": {} })
}

/// `finishEvents()`.
pub(crate) fn finish_events() -> Vec<HarnessV1StreamPart> {
    parts(vec![
        json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "tool_use" }, "usage": zero_usage() }),
        json!({ "type": "finish", "finishReason": { "unified": "tool-calls", "raw": "tool_use" }, "totalUsage": zero_usage() }),
    ])
}

/// A `finish-step` + `finish` pair with `stop` / `end_turn`.
pub(crate) fn stop_finish() -> Vec<JsonValue> {
    vec![
        json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "end_turn" }, "usage": zero_usage() }),
        json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "end_turn" }, "totalUsage": zero_usage() }),
    ]
}

/// A one-text-block turn: `text-start`, `text-delta`, `text-end`, then
/// [`stop_finish`].
pub(crate) fn text_turn(id: &str, text: &str) -> Vec<HarnessV1StreamPart> {
    let mut values = vec![
        json!({ "type": "text-start", "id": id }),
        json!({ "type": "text-delta", "id": id, "delta": text }),
        json!({ "type": "text-end", "id": id }),
    ];
    values.extend(stop_finish());
    parts(values)
}

/// `twoStepEvents()`.
pub(crate) fn two_step_events() -> Vec<HarnessV1StreamPart> {
    parts(vec![
        json!({ "type": "text-start", "id": "first" }),
        json!({ "type": "text-delta", "id": "first", "delta": "First" }),
        json!({ "type": "text-end", "id": "first" }),
        json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "tool_use" }, "usage": zero_usage() }),
        json!({ "type": "text-start", "id": "second" }),
        json!({ "type": "text-delta", "id": "second", "delta": "Second" }),
        json!({ "type": "text-end", "id": "second" }),
        json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "end_turn" }, "usage": zero_usage() }),
        json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "end_turn" }, "totalUsage": zero_usage() }),
    ])
}

/// The first step of a turn that keeps going: a `finish-step` then the
/// next step's first delta (`[finishEvents()[0], { text-delta next }]`).
pub(crate) fn one_step_then_more() -> Vec<HarnessV1StreamPart> {
    vec![
        finish_events().remove(0),
        p(json!({ "type": "text-delta", "id": "next-step", "delta": "next" })),
    ]
}

// ─── agent helpers ─────────────────────────────────────────────────────

pub(crate) fn agent(settings: HarnessAgentSettings) -> HarnessAgent {
    HarnessAgent::new(settings).expect("valid agent settings")
}

pub(crate) fn settings(mock: &Mock) -> HarnessAgentSettings {
    HarnessAgentSettings::new(mock.harness())
}

pub(crate) fn with_sandbox(sandbox: &TestSandbox) -> HarnessAgentCreateSessionOptions {
    HarnessAgentCreateSessionOptions {
        sandbox_session: sandbox.input(),
        ..Default::default()
    }
}

pub(crate) async fn new_session(agent: &HarnessAgent) -> HarnessAgentSession {
    agent
        .create_session(with_sandbox(&make_sandbox_session()))
        .await
        .expect("session")
}

pub(crate) fn call(session: &HarnessAgentSession, prompt: &str) -> HarnessAgentCallParameters {
    HarnessAgentCallParameters::new(session.clone()).with_prompt(prompt)
}

pub(crate) fn obj(value: JsonValue) -> serde_json::Map<String, JsonValue> {
    match value {
        JsonValue::Object(o) => o,
        other => panic!("not an object: {other}"),
    }
}

pub(crate) fn err_message<T>(result: HarnessResult<T>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.to_string(),
    }
}

// ─── makeLifecycleSession ──────────────────────────────────────────────

/// Options of `makeLifecycleSession`.
#[derive(Default)]
pub(crate) struct LifecycleOptions {
    pub owns_sandbox_lifecycle: Option<bool>,
    pub turn_state: Option<HarnessAgentTurnState>,
    pub detach_error: Option<String>,
    pub stop_error: Option<String>,
    pub read_history: Option<ReadHistoryFn>,
}

pub(crate) struct Lifecycle {
    pub session: HarnessAgentSession,
    pub resume_state: HarnessV1ResumeSessionState,
    pub continue_state: HarnessV1ContinueTurnState,
    pub mock: Mock,
    pub sandbox: TestSandbox,
}

impl Lifecycle {
    pub fn detaches(&self) -> usize {
        Mock::count(&self.mock.record.detach_calls)
    }
    pub fn stops(&self) -> usize {
        Mock::count(&self.mock.record.stop_calls)
    }
    pub fn destroys(&self) -> usize {
        Mock::count(&self.mock.record.destroy_calls)
    }
}

pub(crate) async fn make_lifecycle_session(options: LifecycleOptions) -> Lifecycle {
    let mut harness_options = MockHarnessOptions::default();
    harness_options.do_read_history = options.read_history;
    let mock = harness_options.build();
    if let Some(e) = options.detach_error {
        mock.record.detach_errors.lock().push(e);
    }
    if let Some(e) = options.stop_error {
        mock.record.stop_errors.lock().push(e);
    }
    let underlying = mock
        .harness
        .do_start(HarnessV1StartOptions {
            headers: None,
            session_id: "lifecycle-session".into(),
            resume_from: None,
            continue_from: None,
            permission_mode: None,
            builtin_tool_filtering: None,
            abort_signal: None,
            observability: None,
            sandbox_session: make_sandbox_session().sref(),
            session_work_dir: "/work/mock-lifecycle-session".into(),
        })
        .await
        .unwrap();
    mock.record.start_calls.lock().clear();
    let sandbox = make_sandbox_session();
    let mut session_options = HarnessAgentSessionOptions::new(
        "lifecycle-session",
        mock.harness(),
        Arc::from(underlying),
        sandbox.sref(),
        "/work/mock-lifecycle-session",
    );
    session_options.owns_sandbox_lifecycle = options.owns_sandbox_lifecycle;
    session_options.turn_state = options.turn_state;
    Lifecycle {
        session: HarnessAgentSession::new(session_options),
        resume_state: mock_resume_state(),
        continue_state: mock_continue_state(),
        mock,
        sandbox,
    }
}


// ─── ad-hoc harnesses ──────────────────────────────────────────────────

pub(crate) use crate::harness::agent::internal::run_prompt::tests::FakeSession;

/// An ad-hoc `harness` object whose `doStart` returns `session`.
pub(crate) struct SessionHarness {
    pub session: FakeSession,
    pub builtin_tools: IndexMap<String, HarnessV1BuiltinTool>,
}

impl SessionHarness {
    pub fn new(mut session: FakeSession) -> Arc<Self> {
        session.id = "mock-session-1".into();
        Arc::new(Self {
            session,
            builtin_tools: IndexMap::new(),
        })
    }
}

#[async_trait]
impl HarnessV1 for SessionHarness {
    fn harness_id(&self) -> &str {
        "mock"
    }
    fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool> {
        &self.builtin_tools
    }
    async fn do_start(&self, _options: HarnessV1StartOptions) -> HarnessResult<Box<dyn HarnessV1Session>> {
        Ok(Box::new(self.session.clone()))
    }
}

/// A `done` promise the test resolves later.
pub(crate) fn deferred_done() -> (
    Shared<BoxFuture<'static, Result<(), String>>>,
    Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
) {
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let done = async move {
        let _ = rx.await;
        Ok(())
    }
    .boxed()
    .shared();
    (done, Arc::new(Mutex::new(Some(tx))))
}

/// Resolve a [`deferred_done`].
pub(crate) fn resolve(sender: &Mutex<Option<tokio::sync::oneshot::Sender<()>>>) {
    if let Some(tx) = sender.lock().take() {
        let _ = tx.send(());
    }
}

/// A `promptDone` whose promises the test resolves in order
/// (`finishPrompts`).
#[derive(Clone, Default)]
pub(crate) struct PromptGates(pub Arc<Mutex<Vec<tokio::sync::oneshot::Sender<()>>>>);

impl PromptGates {
    pub fn prompt_done(&self) -> PromptDoneFn {
        let senders = self.0.clone();
        Arc::new(move |_| {
            let (tx, rx) = tokio::sync::oneshot::channel::<()>();
            senders.lock().push(tx);
            Box::pin(async move {
                let _ = rx.await;
                Ok(())
            })
        })
    }

    /// Resolve the oldest pending promise.
    pub fn finish_next(&self) {
        let mut senders = self.0.lock();
        if !senders.is_empty() {
            let _ = senders.remove(0).send(());
        }
    }

    /// Resolve every pending promise.
    pub fn finish_all(&self) {
        for tx in self.0.lock().drain(..) {
            let _ = tx.send(());
        }
    }
}
