//! Port of `agent/internal/run-prompt.test.ts`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::future::FutureExt;
use serde_json::json;
use tokio::sync::Notify;

use super::*;
use crate::harness::errors::BoxError;
use crate::harness::sdk::stop_condition::{has_tool_call, is_step_count, stop_condition};
use crate::harness::sdk::telemetry::{SharedTelemetry, Telemetry, TelemetryEvent};
use crate::harness::sdk::tool::{Schema, Tool, json_schema};
use crate::harness::spec::lifecycle_state::{
    ContinueTurnType, HarnessV1ContinueTurnState, HarnessV1ResumeSessionState, ResumeSessionType,
};
use crate::harness::spec::harness::HarnessV1Emit;
use crate::harness::spec::literal::HarnessV1SpecificationVersion;
use crate::harness::spec::tools::HarnessV1BuiltinTool;
use crate::harness::test_support::MockSandbox;

pub(super) const WORK_DIR: &str = "/vercel/sandbox/claude-code-abc123";

// ─── zod stand-ins ─────────────────────────────────────────────────────

/// A `z.object({...})` stand-in. Each field is `name: type`, where type is
/// `string`, `number`, `boolean`, `array`, `object`, `any`, or
/// `enum:a|b`; a trailing `?` makes it optional. Unknown keys are stripped,
/// as zod does.
pub(super) fn z_object(fields: &[(&str, &str)]) -> Schema {
    let fields: Vec<(String, String)> = fields
        .iter()
        .map(|(n, t)| (n.to_string(), t.to_string()))
        .collect();
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for (name, ty) in &fields {
        let optional = ty.ends_with('?');
        let ty = ty.trim_end_matches('?');
        let schema = match ty {
            t if t.starts_with("enum:") => {
                json!({ "type": "string", "enum": t[5..].split('|').collect::<Vec<_>>() })
            }
            "any" => json!({}),
            t => json!({ "type": t }),
        };
        properties.insert(name.clone(), schema);
        if !optional {
            required.push(name.clone());
        }
    }
    json_schema(json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    }))
    .with_validate(move |value| {
        let Some(object) = value.as_object() else {
            return Err("Invalid input: expected object".into());
        };
        let mut out = serde_json::Map::new();
        for (name, ty) in &fields {
            let optional = ty.ends_with('?');
            let ty = ty.trim_end_matches('?');
            match object.get(name) {
                None if optional => {}
                None => return Err(format!("Invalid input: missing {name}").into()),
                Some(v) => {
                    let ok = match ty {
                        "string" => v.is_string(),
                        "number" => v.is_number(),
                        "boolean" => v.is_boolean(),
                        "array" => v.is_array(),
                        "object" => v.is_object(),
                        "any" => true,
                        t if t.starts_with("enum:") => v
                            .as_str()
                            .is_some_and(|s| t[5..].split('|').any(|o| o == s)),
                        _ => false,
                    };
                    if !ok {
                        return Err(format!("Invalid input: {name} expected {ty}").into());
                    }
                    out.insert(name.clone(), v.clone());
                }
            }
        }
        Ok(JsonValue::Object(out))
    })
}

pub(super) fn tool(fields: &[(&str, &str)]) -> Tool {
    Tool::new(z_object(fields))
}

pub(super) fn tools(entries: Vec<(&str, Tool)>) -> ToolSet {
    entries
        .into_iter()
        .map(|(n, t)| (n.to_string(), t))
        .collect()
}

// ─── fakes ─────────────────────────────────────────────────────────────

pub(super) struct FakeHarness {
    pub id: String,
    pub builtin_tools: IndexMap<String, HarnessV1BuiltinTool>,
}

impl FakeHarness {
    pub fn new(id: &str) -> Arc<Self> {
        Arc::new(Self {
            id: id.into(),
            builtin_tools: IndexMap::new(),
        })
    }

    pub fn with_builtins(id: &str, builtins: &ToolSet) -> Arc<Self> {
        Arc::new(Self {
            id: id.into(),
            builtin_tools: builtins
                .iter()
                .map(|(n, t)| (n.clone(), HarnessV1BuiltinTool::from(t.clone())))
                .collect(),
        })
    }
}

#[async_trait]
impl HarnessV1 for FakeHarness {
    fn harness_id(&self) -> &str {
        &self.id
    }
    fn builtin_tools(&self) -> &IndexMap<String, HarnessV1BuiltinTool> {
        &self.builtin_tools
    }
    async fn do_start(
        &self,
        _options: crate::harness::spec::harness::HarnessV1StartOptions,
    ) -> HarnessResult<Box<dyn HarnessV1Session>> {
        Ok(Box::new(FakeSession::new(Vec::new())))
    }
}

pub(super) type SubmitFn = Arc<
    dyn Fn(HarnessV1SubmitToolResult, HarnessV1Emit) -> BoxFuture<'static, HarnessResult<()>>
        + Send
        + Sync,
>;
pub(super) type ApprovalFn = Arc<
    dyn Fn(HarnessV1SubmitToolApproval, HarnessV1Emit) -> BoxFuture<'static, HarnessResult<()>>
        + Send
        + Sync,
>;
pub(super) type TurnFn = Arc<dyn Fn(HarnessV1Emit) + Send + Sync>;

/// A scripted `HarnessV1PromptControl`.
pub(super) struct FakeControl {
    emit: HarnessV1Emit,
    on_submit: Option<SubmitFn>,
    on_approval: Option<ApprovalFn>,
    done: Shared<BoxFuture<'static, Result<(), String>>>,
}

#[async_trait]
impl HarnessV1PromptControl for FakeControl {
    async fn submit_tool_result(&self, input: HarnessV1SubmitToolResult) -> HarnessResult<()> {
        match &self.on_submit {
            Some(f) => f(input, self.emit.clone()).await,
            None => Ok(()),
        }
    }
    fn supports_submit_tool_approval(&self) -> bool {
        self.on_approval.is_some()
    }
    async fn submit_tool_approval(&self, input: HarnessV1SubmitToolApproval) -> HarnessResult<()> {
        match &self.on_approval {
            Some(f) => f(input, self.emit.clone()).await,
            None => Ok(()),
        }
    }
    fn done(&self) -> BoxFuture<'static, HarnessResult<()>> {
        let done = self.done.clone();
        Box::pin(async move { done.await.map_err(|e| plain_error(e)) })
    }
}

/// `fakeSession(script, onSubmitToolResult)` and the ad-hoc sessions.
#[derive(Clone)]
pub(super) struct FakeSession {
    pub id: String,
    /// Emitted right after the turn starts.
    pub script: Vec<HarnessV1StreamPart>,
    /// Runs right after the turn starts, in place of `script`.
    pub on_turn: Option<TurnFn>,
    pub on_submit: Option<SubmitFn>,
    pub on_approval: Option<ApprovalFn>,
    pub done: Option<Shared<BoxFuture<'static, Result<(), String>>>>,
    pub start_error: Option<String>,
    pub prompt_turns: Arc<Mutex<Vec<HarnessV1PromptTurnOptions>>>,
    pub continue_turns: Arc<Mutex<Vec<HarnessV1ContinueTurnOptions>>>,
}

impl FakeSession {
    pub fn new(script: Vec<HarnessV1StreamPart>) -> Self {
        Self {
            id: "fake-session".into(),
            script,
            on_turn: None,
            on_submit: None,
            on_approval: None,
            done: None,
            start_error: None,
            prompt_turns: Arc::default(),
            continue_turns: Arc::default(),
        }
    }

    /// Record submitted results (`fakeSession(script, onSubmitToolResult)`).
    pub fn recording(
        script: Vec<HarnessV1StreamPart>,
    ) -> (Self, Arc<Mutex<Vec<HarnessV1SubmitToolResult>>>) {
        let submitted: Arc<Mutex<Vec<HarnessV1SubmitToolResult>>> = Arc::default();
        let log = submitted.clone();
        let mut session = Self::new(script);
        session.on_submit = Some(Arc::new(move |input, _| {
            log.lock().push(input);
            Box::pin(async { Ok(()) })
        }));
        (session, submitted)
    }

    pub fn with_submit(
        mut self,
        f: impl Fn(HarnessV1SubmitToolResult, HarnessV1Emit) + Send + Sync + 'static,
    ) -> Self {
        self.on_submit = Some(Arc::new(move |input, emit| {
            f(input, emit);
            Box::pin(async { Ok(()) })
        }));
        self
    }

    pub fn with_approval(
        mut self,
        f: impl Fn(HarnessV1SubmitToolApproval, HarnessV1Emit) + Send + Sync + 'static,
    ) -> Self {
        self.on_approval = Some(Arc::new(move |input, emit| {
            f(input, emit);
            Box::pin(async { Ok(()) })
        }));
        self
    }

    fn start(&self, emit: HarnessV1Emit) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        if let Some(message) = &self.start_error {
            return Err(plain_error(message.clone()));
        }
        // [rust] TS emits the script in a microtask, before `done` settles;
        // the channel buffers, so emitting now is the same.
        match self.on_turn.clone() {
            Some(f) => {
                let task_emit = emit.clone();
                tokio::spawn(async move {
                    tokio::task::yield_now().await;
                    f(task_emit);
                });
            }
            None => {
                for event in self.script.clone() {
                    emit(event);
                }
            }
        }
        Ok(Box::new(FakeControl {
            emit,
            on_submit: self.on_submit.clone(),
            on_approval: self.on_approval.clone(),
            done: self
                .done
                .clone()
                .unwrap_or_else(|| futures::future::ready(Ok(())).boxed().shared()),
        }))
    }
}

pub(super) fn continue_state() -> HarnessV1ContinueTurnState {
    HarnessV1ContinueTurnState {
        r#type: ContinueTurnType,
        harness_id: "fake".into(),
        specification_version: HarnessV1SpecificationVersion,
        data: json!({}),
        pending_tool_approvals: None,
        pending_tool_results: None,
        turn_settings: None,
    }
}

pub(super) fn resume_state() -> HarnessV1ResumeSessionState {
    HarnessV1ResumeSessionState {
        r#type: ResumeSessionType,
        harness_id: "fake".into(),
        specification_version: HarnessV1SpecificationVersion,
        data: json!({}),
        continue_from: None,
    }
}

#[async_trait]
impl HarnessV1Session for FakeSession {
    fn session_id(&self) -> String {
        self.id.clone()
    }
    fn is_resume(&self) -> bool {
        false
    }
    async fn do_prompt_turn(
        &self,
        options: HarnessV1PromptTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        let emit = options.emit.clone();
        self.prompt_turns.lock().push(options);
        self.start(emit)
    }
    async fn do_compact(&self, _custom_instructions: Option<String>) -> HarnessResult<()> {
        Ok(())
    }
    async fn do_continue_turn(
        &self,
        options: HarnessV1ContinueTurnOptions,
    ) -> HarnessResult<Box<dyn HarnessV1PromptControl>> {
        let emit = options.emit.clone();
        self.continue_turns.lock().push(options);
        self.start(emit)
    }
    async fn do_suspend_turn(&self) -> HarnessResult<HarnessV1ContinueTurnState> {
        Ok(continue_state())
    }
    async fn do_detach(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        Ok(resume_state())
    }
    async fn do_stop(&self) -> HarnessResult<HarnessV1ResumeSessionState> {
        Ok(resume_state())
    }
    async fn do_destroy(&self) -> HarnessResult<()> {
        Ok(())
    }
}

// ─── helpers ───────────────────────────────────────────────────────────

pub(super) fn p(value: JsonValue) -> HarnessV1StreamPart {
    HarnessV1StreamPart::parse(value).expect("valid stream part")
}

pub(super) fn parts(values: Vec<JsonValue>) -> Vec<HarnessV1StreamPart> {
    values.into_iter().map(p).collect()
}

pub(super) fn empty_usage() -> JsonValue {
    json!({ "inputTokens": {}, "outputTokens": {} })
}

/// `finishEvents`.
pub(super) fn finish_events() -> Vec<HarnessV1StreamPart> {
    parts(vec![
        json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "end_turn" }, "usage": empty_usage() }),
        json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "end_turn" }, "totalUsage": empty_usage() }),
    ])
}

pub(super) fn finish_step() -> HarnessV1StreamPart {
    finish_events().remove(0)
}

pub(super) fn finish() -> HarnessV1StreamPart {
    finish_events().remove(1)
}

/// `resumableFinishStep`.
pub(super) fn resumable_finish_step() -> HarnessV1StreamPart {
    p(
        json!({ "type": "finish-step", "finishReason": { "unified": "tool-calls", "raw": "tool_use" }, "usage": empty_usage() }),
    )
}

pub(super) fn with_finish(mut script: Vec<HarnessV1StreamPart>) -> Vec<HarnessV1StreamPart> {
    script.extend(finish_events());
    script
}

pub(super) fn sandbox() -> Arc<dyn SandboxSession> {
    Arc::new(MockSandbox::new(|_, _| Err("sandbox not used".into())))
}

pub(super) fn base_input(
    harness: Arc<dyn HarnessV1>,
    session: FakeSession,
    tools: ToolSet,
) -> RunPromptInput {
    let mut input = RunPromptInput::new(harness, Arc::new(session), sandbox(), WORK_DIR);
    input.prompt = Some("go".into());
    input.tools = tools;
    input
}

pub(super) fn fake_harness() -> Arc<dyn HarnessV1> {
    FakeHarness::new("fake")
}

/// Drain `fullStream` as JSON, then await `done`.
pub(super) async fn collect(output: &RunPromptOutput) -> Vec<JsonValue> {
    let parts: Vec<JsonValue> = output
        .result
        .full_stream()
        .map(|p| p.to_json())
        .collect()
        .await;
    output.done.clone().await;
    parts
}

pub(super) async fn run_collect(input: RunPromptInput) -> (Vec<JsonValue>, RunPromptOutput) {
    let output = run_prompt(input);
    let parts = tokio::time::timeout(Duration::from_secs(10), collect(&output))
        .await
        .expect("turn finished");
    (parts, output)
}

/// `expect.objectContaining` / `toMatchObject`: every key of `expected`
/// matches, recursively for objects; arrays match element-wise.
pub(super) fn matches(actual: &JsonValue, expected: &JsonValue) -> bool {
    match (actual, expected) {
        (JsonValue::Object(a), JsonValue::Object(e)) => e
            .iter()
            .all(|(k, v)| a.get(k).is_some_and(|av| matches(av, v))),
        (JsonValue::Array(a), JsonValue::Array(e)) => {
            a.len() == e.len() && a.iter().zip(e).all(|(a, e)| matches(a, e))
        }
        _ => actual == expected,
    }
}

pub(super) fn contains_matching(parts: &[JsonValue], expected: JsonValue) -> bool {
    parts.iter().any(|p| matches(p, &expected))
}

pub(super) fn of_type<'a>(parts: &'a [JsonValue], ty: &str) -> Vec<&'a JsonValue> {
    parts.iter().filter(|p| p["type"] == ty).collect()
}

pub(super) fn types(parts: &[JsonValue]) -> Vec<String> {
    parts
        .iter()
        .map(|p| p["type"].as_str().unwrap_or_default().to_string())
        .collect()
}

pub(super) fn to_json<T: serde::Serialize>(value: &T) -> JsonValue {
    serde_json::to_value(value).unwrap()
}

pub(super) async fn wait_for(mut check: impl FnMut() -> bool) {
    for _ in 0..500 {
        if check() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!("wait_for timed out");
}

/// A gate a test opens later.
#[derive(Clone, Default)]
pub(super) struct Gate {
    open: Arc<std::sync::atomic::AtomicBool>,
    notify: Arc<Notify>,
}

impl Gate {
    pub fn open(&self) {
        self.open.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
    pub async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            if self.open.load(Ordering::SeqCst) {
                return;
            }
            notified.await;
        }
    }
}

pub(super) type Log = Arc<Mutex<Vec<String>>>;

pub(super) fn index_of(events: &[String], event: &str) -> usize {
    events
        .iter()
        .position(|e| e == event)
        .unwrap_or_else(|| panic!("missing event: {event}"))
}

// ─── tests ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn preserves_native_name_in_builtin_tool_approval_metadata() {
    for (tool_name, native_name, input, builtin) in [
        (
            "bash",
            "Bash",
            json!({ "command": "echo hello", "description": "Print hello" }),
            tool(&[("command", "string"), ("description", "string?")]),
        ),
        (
            "write",
            "Write",
            json!({ "file_path": "hello.txt", "content": "hello" }),
            tool(&[("file_path", "string"), ("content", "string")]),
        ),
        (
            "edit",
            "Edit",
            json!({ "file_path": "hello.txt", "old_string": "hello", "new_string": "hi" }),
            tool(&[
                ("file_path", "string"),
                ("old_string", "string"),
                ("new_string", "string"),
            ]),
        ),
    ] {
        let events = parts(vec![
            json!({ "type": "stream-start" }),
            json!({ "type": "tool-input-start", "id": "c1", "toolName": tool_name, "providerExecuted": true }),
            json!({ "type": "tool-input-delta", "id": "c1", "delta": input.to_string() }),
            json!({ "type": "tool-call", "toolCallId": "c1", "toolName": tool_name, "input": input.to_string(), "providerExecuted": true, "nativeName": native_name }),
            json!({ "type": "tool-input-end", "id": "c1" }),
            json!({ "type": "tool-approval-request", "approvalId": "a1", "toolCallId": "c1" }),
        ]);
        let mut run = base_input(
            FakeHarness::new("claude-code"),
            FakeSession::new(events),
            tools(vec![(tool_name, builtin)]),
        );
        run.prompt = Some("Run the shell command `echo hello`".into());
        let (parts, output) = run_collect(run).await;
        assert!(of_type(&parts, "error").is_empty(), "{parts:?}");
        let expected_tool_call = json!({
            "type": "tool-call",
            "toolCallId": "c1",
            "toolName": tool_name,
            "input": input,
            "providerExecuted": true,
            "providerMetadata": { "claude-code": { "nativeToolName": native_name } },
        });
        assert_eq!(*of_type(&parts, "tool-call")[0], expected_tool_call);
        let mut call_without_type = expected_tool_call.clone();
        call_without_type.as_object_mut().unwrap().remove("type");
        assert_eq!(
            of_type(&parts, "tool-approval-request"),
            vec![&json!({ "type": "tool-approval-request", "approvalId": "a1", "toolCall": call_without_type })]
        );
        let steps = output.result.steps().await.unwrap();
        assert_eq!(to_json(&steps[0].tool_calls()), json!([call_without_type]));
    }
}

fn questions_tools() -> ToolSet {
    tools(vec![(
        "askUserQuestions",
        tool(&[("questions", "array")]),
    )])
}

mod filtered_builtin_dynamic_calls {
    use super::*;

    fn read_tools() -> ToolSet {
        tools(vec![("read", tool(&[("file_path", "string")]))])
    }

    #[tokio::test]
    async fn accepts_a_provider_executed_dynamic_read_with_filtering() {
        for filtering in [
            HarnessV1BuiltinToolFiltering::Deny {
                tool_names: vec!["read".into()],
            },
            HarnessV1BuiltinToolFiltering::Allow { tool_names: vec![] },
        ] {
            let pending: Arc<Mutex<Vec<HarnessV1PendingToolResult>>> = Arc::default();
            let log = pending.clone();
            let mut run = base_input(
                FakeHarness::with_builtins("fake", &read_tools()),
                FakeSession::new(with_finish(parts(vec![
                    json!({ "type": "tool-input-start", "id": "replacement-call", "toolName": "read", "dynamic": true, "providerExecuted": true }),
                    json!({ "type": "tool-call", "toolCallId": "replacement-call", "toolName": "read", "input": "{\"query\":\"hello\"}", "dynamic": true, "providerExecuted": true }),
                    json!({ "type": "tool-result", "toolCallId": "replacement-call", "toolName": "read", "result": "MCP result", "dynamic": true }),
                ]))),
                read_tools(),
            );
            run.active_tools = Some(ToolSet::new());
            run.builtin_tool_filtering = Some(filtering);
            run.on_pending_tool_result = Some(Arc::new(move |p| log.lock().push(p)));
            let (parts, output) = run_collect(run).await;
            assert!(contains_matching(
                &parts,
                json!({ "type": "tool-input-start", "id": "replacement-call", "toolName": "read", "dynamic": true, "providerExecuted": true })
            ));
            let parsed = parts
                .iter()
                .find(|p| p["type"] == "tool-call" && p["toolCallId"] == "replacement-call")
                .unwrap();
            assert!(matches(
                parsed,
                &json!({ "toolName": "read", "input": { "query": "hello" }, "dynamic": true, "providerExecuted": true })
            ));
            assert!(parsed.get("invalid").is_none());
            assert!(contains_matching(
                &parts,
                json!({ "type": "tool-result", "toolCallId": "replacement-call", "output": "MCP result", "dynamic": true })
            ));
            assert_eq!(output.result.steps().await.unwrap()[0].tool_results().len(), 1);
            assert!(pending.lock().is_empty());
        }
    }

    #[tokio::test]
    async fn continues_validating_a_tool_with_an_active_builtin_schema() {
        let mut run = base_input(
            FakeHarness::with_builtins("fake", &read_tools()),
            FakeSession::new(with_finish(parts(vec![
                json!({ "type": "tool-call", "toolCallId": "builtin-call", "toolName": "read", "input": "{\"query\":\"hello\"}", "providerExecuted": true, "dynamic": true }),
            ]))),
            read_tools(),
        );
        run.active_tools = Some(ToolSet::new());
        let (parts, _) = run_collect(run).await;
        assert!(contains_matching(
            &parts,
            json!({ "type": "tool-call", "toolCallId": "builtin-call", "invalid": true })
        ));
    }

    #[tokio::test]
    async fn keeps_an_active_host_tool_schema_when_its_name_matches_a_filtered_builtin() {
        let host_read = tools(vec![("read", tool(&[("query", "string")]))]);
        let mut run = base_input(
            FakeHarness::with_builtins("fake", &read_tools()),
            FakeSession::new(with_finish(parts(vec![
                json!({ "type": "tool-call", "toolCallId": "host-name-call", "toolName": "read", "input": "{\"file_path\":\"example.txt\"}", "providerExecuted": true, "dynamic": true }),
            ]))),
            host_read.clone(),
        );
        run.active_tools = Some(host_read);
        run.builtin_tool_filtering = Some(HarnessV1BuiltinToolFiltering::Deny {
            tool_names: vec!["read".into()],
        });
        let (parts, _) = run_collect(run).await;
        assert!(contains_matching(
            &parts,
            json!({ "type": "tool-call", "toolCallId": "host-name-call", "invalid": true })
        ));
    }
}

#[tokio::test]
async fn pauses_for_ask_user_questions_and_persists_adapter_metadata_as_provider_options() {
    let provider_metadata =
        json!({ "fake": { "nativeRequest": { "questions": [{ "prompt": "Which framework?" }] } } });
    let input = json!({
        "allowPartialAnswers": true,
        "questions": [{ "id": "question-1", "question": "Which framework?" }],
    })
    .to_string();
    let (session, submitted) = FakeSession::recording(parts(vec![json!({
        "type": "tool-call", "toolCallId": "question-call", "toolName": "askUserQuestions",
        "input": input, "providerExecuted": false, "providerMetadata": provider_metadata,
    })]));
    let pending: Arc<Mutex<Vec<HarnessV1PendingToolResult>>> = Arc::default();
    let log = pending.clone();
    let mut run = base_input(
        FakeHarness::with_builtins("fake", &questions_tools()),
        session,
        questions_tools(),
    );
    run.active_tools = Some(ToolSet::new());
    run.on_pending_tool_result = Some(Arc::new(move |p| log.lock().push(p)));
    run_collect(run).await;
    assert!(submitted.lock().is_empty());
    assert_eq!(
        to_json(&*pending.lock()),
        json!([{
            "toolCallId": "question-call",
            "toolName": "askUserQuestions",
            "input": input,
            "providerOptions": provider_metadata,
        }])
    );
}

#[tokio::test]
async fn strips_the_work_dir_for_consumers_but_executes_host_tools_with_the_absolute_path() {
    let executed: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let log = executed.clone();
    let read_file = tool(&[("path", "string")])
        .with_description("Read a file")
        .with_execute(move |args, _| {
            log.lock().push(args);
            async { Ok(json!({ "ok": true })) }
        });
    let run = base_input(
        fake_harness(),
        FakeSession::new(with_finish(parts(vec![
            json!({ "type": "tool-input-start", "id": "c1", "toolName": "readFile", "providerExecuted": false }),
            json!({ "type": "tool-input-delta", "id": "c1", "delta": "{\"path\":\"/vercel/sandbox/claude" }),
            json!({ "type": "tool-input-delta", "id": "c1", "delta": "-code-abc123/src/foo.ts\"}" }),
            json!({ "type": "tool-input-end", "id": "c1" }),
            json!({ "type": "tool-call", "toolCallId": "c1", "toolName": "readFile", "input": json!({ "path": format!("{WORK_DIR}/src/foo.ts") }).to_string() }),
            json!({ "type": "tool-result", "toolCallId": "b1", "toolName": "bash", "result": format!("{WORK_DIR}/src/foo.ts\n{WORK_DIR}/src/bar.ts\n") }),
        ]))),
        tools(vec![("readFile", read_file)]),
    );
    let (parts, _) = run_collect(run).await;
    let tool_input: Vec<&JsonValue> = parts
        .iter()
        .filter(|p| p["type"].as_str().unwrap().starts_with("tool-input-"))
        .collect();
    assert_eq!(
        tool_input,
        vec![
            &json!({ "type": "tool-input-start", "id": "c1", "toolName": "readFile", "providerExecuted": false }),
            &json!({ "type": "tool-input-delta", "id": "c1", "delta": "{\"path\":\"" }),
            &json!({ "type": "tool-input-delta", "id": "c1", "delta": "src/foo.ts\"}" }),
            &json!({ "type": "tool-input-end", "id": "c1" }),
        ]
    );
    assert_eq!(
        *executed.lock(),
        vec![json!({ "path": format!("{WORK_DIR}/src/foo.ts") })]
    );
    assert_eq!(of_type(&parts, "tool-call")[0]["input"], json!({ "path": "src/foo.ts" }));
    assert_eq!(
        of_type(&parts, "tool-result")[0]["output"],
        json!("src/foo.ts\nsrc/bar.ts\n")
    );
}

#[tokio::test]
async fn uses_final_total_usage_when_it_differs_from_received_step_usage() {
    let run = base_input(
        fake_harness(),
        FakeSession::new(parts(vec![
            json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "stop" },
                "usage": { "inputTokens": { "total": 2, "noCache": 2, "cacheRead": 0, "cacheWrite": 0 },
                           "outputTokens": { "total": 5, "text": 5, "reasoning": 0 } } }),
            json!({ "type": "finish", "finishReason": { "unified": "stop", "raw": "stop" },
                "totalUsage": { "inputTokens": { "total": 10, "noCache": 4, "cacheRead": 6, "cacheWrite": 0 },
                                "outputTokens": { "total": 40, "text": 30, "reasoning": 10 } } }),
        ])),
        ToolSet::new(),
    );
    let (_, output) = run_collect(run).await;
    assert_eq!(
        to_json(&output.result.usage().await.unwrap()),
        json!({
            "inputTokens": 10,
            "inputTokenDetails": { "noCacheTokens": 4, "cacheReadTokens": 6, "cacheWriteTokens": 0 },
            "outputTokens": 40,
            "outputTokenDetails": { "textTokens": 30, "reasoningTokens": 10 },
            "totalTokens": 50,
        })
    );
}

/// An integration whose callbacks record events and wait on gates.
#[derive(Default)]
struct GatedIntegration {
    events: Log,
    language_model_end: Gate,
    step_end: Gate,
    end: Gate,
}

impl Telemetry for GatedIntegration {
    fn on_language_model_call_end(&self, _: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let (events, gate) = (self.events.clone(), self.language_model_end.clone());
        Box::pin(async move {
            events.lock().push("language-model-end:start".into());
            gate.wait().await;
            events.lock().push("language-model-end:done".into());
        })
    }
    fn on_step_end(&self, _: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let (events, gate) = (self.events.clone(), self.step_end.clone());
        Box::pin(async move {
            events.lock().push("step-end:start".into());
            gate.wait().await;
            events.lock().push("step-end:done".into());
        })
    }
    fn on_end(&self, _: &TelemetryEvent) -> BoxFuture<'static, ()> {
        let (events, gate) = (self.events.clone(), self.end.clone());
        Box::pin(async move {
            events.lock().push("end:start".into());
            gate.wait().await;
            events.lock().push("end:done".into());
        })
    }
}

fn telemetry_with(integration: SharedTelemetry) -> TelemetryOptions {
    TelemetryOptions {
        integrations: Some(vec![integration]),
        ..Default::default()
    }
}

mod telemetry_lifecycle {
    use super::*;

    #[tokio::test]
    async fn publishes_concurrent_host_tool_lifecycle_events_and_results_as_each_tool_runs() {
        let events: Log = Arc::default();
        let release_slow = Gate::default();
        let completed: Arc<Mutex<HashSet<String>>> = Arc::default();
        let done_gate = Gate::default();
        let mut session = FakeSession::new(parts(vec![
            json!({ "type": "stream-start", "modelId": "fake-model" }),
            json!({ "type": "tool-call", "toolCallId": "fast-call", "toolName": "work", "input": "{\"label\":\"fast\"}" }),
            json!({ "type": "tool-call", "toolCallId": "slow-call", "toolName": "work", "input": "{\"label\":\"slow\"}" }),
        ]));
        session.id = "concurrent-tools".into();
        let gate = done_gate.clone();
        session.done = Some(
            async move {
                gate.wait().await;
                Ok(())
            }
            .boxed()
            .shared(),
        );
        let (ev, comp, gate) = (events.clone(), completed.clone(), done_gate.clone());
        let session = session.with_submit(move |submission, emit| {
            let label = if submission.tool_call_id == "fast-call" { "fast" } else { "slow" };
            emit(p(json!({ "type": "tool-result", "toolCallId": submission.tool_call_id, "toolName": "work", "result": submission.output })));
            let count = {
                let mut c = comp.lock();
                c.insert(submission.tool_call_id.clone());
                c.len()
            };
            if count == 2 {
                for event in finish_events() {
                    emit(event);
                }
                gate.open();
            }
            ev.lock().push(format!("runtime-result:{label}"));
        });
        let (ev, slow) = (events.clone(), release_slow.clone());
        let work = tool(&[("label", "enum:fast|slow")]).with_execute(move |args, _| {
            let (ev, slow) = (ev.clone(), slow.clone());
            async move {
                let label = args["label"].as_str().unwrap().to_string();
                ev.lock().push(format!("execute-start:{label}"));
                if label == "slow" {
                    slow.wait().await;
                }
                ev.lock().push(format!("execute-end:{label}"));
                Ok(json!({ "label": label }))
            }
        });
        let mut run = base_input(fake_harness(), session, tools(vec![("work", work)]));
        let (ev1, ev2) = (events.clone(), events.clone());
        run.callbacks = Some(HarnessAgentLifecycleCallbacks {
            on_tool_execution_start: Some(Arc::new(move |e| {
                ev1.lock().push(format!(
                    "callback-start:{}",
                    e.tool_call.input["label"].as_str().unwrap()
                ));
                Box::pin(async {})
            })),
            on_tool_execution_end: Some(Arc::new(move |e| {
                ev2.lock().push(format!(
                    "callback-end:{}",
                    e.tool_call.input["label"].as_str().unwrap()
                ));
                Box::pin(async {})
            })),
            ..Default::default()
        });
        let output = run_prompt(run);
        let mut stream = output.result.full_stream();
        while let Some(part) = stream.next().await {
            if let TextStreamPart::ToolResult(r) = part {
                let label = r.output["label"].as_str().unwrap().to_string();
                events.lock().push(format!("stream-result:{label}"));
                if label == "fast" {
                    release_slow.open();
                }
            }
        }
        output.done.clone().await;
        let events = events.lock().clone();
        for label in ["fast", "slow"] {
            assert!(
                index_of(&events, &format!("callback-start:{label}"))
                    < index_of(&events, &format!("execute-start:{label}"))
            );
            assert!(
                index_of(&events, &format!("execute-end:{label}"))
                    < index_of(&events, &format!("callback-end:{label}"))
            );
            assert!(
                index_of(&events, &format!("callback-end:{label}"))
                    < index_of(&events, &format!("stream-result:{label}"))
            );
        }
        assert!(index_of(&events, "stream-result:fast") < index_of(&events, "execute-end:slow"));
    }

    #[tokio::test]
    async fn does_not_settle_until_async_end_callbacks_complete_in_order() {
        let integration = Arc::new(GatedIntegration::default());
        let mut run = base_input(
            fake_harness(),
            FakeSession::new(vec![
                p(json!({ "type": "stream-start" })),
                p(json!({ "type": "text-delta", "id": "text-1", "delta": "done" })),
                finish_step(),
                p(json!({ "type": "text-delta", "id": "text-2", "delta": "ignored" })),
            ]),
            ToolSet::new(),
        );
        run.stop_conditions = Some(vec![stop_condition(|steps| steps.len() == 1)]);
        run.telemetry = Some(telemetry_with(integration.clone()));
        let output = run_prompt(run);
        let settled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = settled.clone();
        let done = output.done.clone();
        tokio::spawn(async move {
            done.await;
            flag.store(true, Ordering::SeqCst);
        });
        let consume = tokio::spawn(output.result.full_stream().count());
        let events = integration.events.clone();
        let snapshot = || events.lock().clone();

        wait_for(|| snapshot() == ["language-model-end:start"]).await;
        assert!(!settled.load(Ordering::SeqCst));
        integration.language_model_end.open();
        wait_for(|| {
            snapshot() == ["language-model-end:start", "language-model-end:done", "step-end:start"]
        })
        .await;
        assert!(!settled.load(Ordering::SeqCst));
        integration.step_end.open();
        wait_for(|| {
            snapshot()
                == [
                    "language-model-end:start",
                    "language-model-end:done",
                    "step-end:start",
                    "step-end:done",
                    "end:start",
                ]
        })
        .await;
        assert!(!settled.load(Ordering::SeqCst));
        integration.end.open();
        output.done.clone().await;
        consume.await.unwrap();
        wait_for(|| settled.load(Ordering::SeqCst)).await;
        assert_eq!(
            snapshot(),
            [
                "language-model-end:start",
                "language-model-end:done",
                "step-end:start",
                "step-end:done",
                "end:start",
                "end:done",
            ]
        );
    }
}

#[derive(Default)]
struct EndRecorder {
    ends: Arc<Mutex<Vec<TelemetryEvent>>>,
}

impl Telemetry for EndRecorder {
    fn on_end(&self, event: &TelemetryEvent) -> BoxFuture<'static, ()> {
        self.ends.lock().push(event.clone());
        Box::pin(async {})
    }
}

#[tokio::test]
async fn reports_the_usage_of_every_step_as_the_turn_total_when_a_stop_condition_ends_the_turn() {
    let finish_step = |input: u64, output: u64| {
        p(json!({ "type": "finish-step", "finishReason": { "unified": "stop", "raw": "end_turn" },
            "usage": { "inputTokens": { "total": input, "noCache": input, "cacheRead": 0, "cacheWrite": 0 },
                       "outputTokens": { "total": output, "text": output, "reasoning": 0 } } }))
    };
    let recorder = Arc::new(EndRecorder::default());
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(vec![
            p(json!({ "type": "stream-start" })),
            finish_step(10, 1),
            finish_step(25, 4),
            // Evaluated when the next part arrives; this step must not count.
            p(json!({ "type": "text-delta", "id": "text-1", "delta": "ignored" })),
            finish_step(7, 2),
        ]),
        ToolSet::new(),
    );
    run.stop_conditions = Some(vec![stop_condition(|steps| steps.len() == 2)]);
    run.telemetry = Some(telemetry_with(recorder.clone()));
    run_collect(run).await;
    let ends = recorder.ends.lock();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0]["totalUsage"]["inputTokens"], json!(35));
}

#[tokio::test]
async fn passes_raw_parts_through_without_opening_a_step_or_adding_to_one() {
    let run = base_input(
        fake_harness(),
        FakeSession::new(vec![
            p(json!({ "type": "stream-start" })),
            p(json!({ "type": "text-start", "id": "t1" })),
            p(json!({ "type": "text-delta", "id": "t1", "delta": "one" })),
            p(json!({ "type": "text-end", "id": "t1" })),
            finish_step(),
            p(json!({ "type": "raw", "rawValue": { "marker": "between-steps" } })),
            p(json!({ "type": "text-start", "id": "t2" })),
            p(json!({ "type": "text-delta", "id": "t2", "delta": "two" })),
            p(json!({ "type": "text-end", "id": "t2" })),
            finish_step(),
            finish(),
        ]),
        ToolSet::new(),
    );
    let (parts, output) = run_collect(run).await;
    let types = types(&parts);
    let raw = types.iter().position(|t| t == "raw").unwrap();
    assert_eq!(types[raw - 1], "finish-step");
    assert_eq!(types[raw + 1], "start-step");
    assert_eq!(output.result.steps().await.unwrap().len(), 2);
}

#[tokio::test]
async fn forwards_a_raw_part_that_arrives_right_after_the_step_a_stop_condition_ends_on() {
    let mut run = base_input(
        fake_harness(),
        FakeSession::new(vec![
            p(json!({ "type": "stream-start" })),
            finish_step(),
            p(json!({ "type": "raw", "rawValue": { "marker": "usage-after-step" } })),
            p(json!({ "type": "text-delta", "id": "t", "delta": "ignored" })),
        ]),
        ToolSet::new(),
    );
    run.stop_conditions = Some(vec![stop_condition(|steps| steps.len() == 1)]);
    let (parts, _) = run_collect(run).await;
    let raws: Vec<&JsonValue> = of_type(&parts, "raw").iter().map(|p| &p["rawValue"]).collect();
    assert_eq!(raws, vec![&json!({ "marker": "usage-after-step" })]);
}

#[path = "run_prompt_tests_steps.rs"]
mod steps;
