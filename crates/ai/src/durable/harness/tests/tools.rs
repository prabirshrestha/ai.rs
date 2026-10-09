//! Port of `test/harness-tools.test.ts`.
//!
//! The environment and coding-tool cases use `LocalExecutionEnv` for `NodeExecutionEnv`. Promises a tool leaves pending are spawned Tokio tasks.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::chat::*;
use super::support::{add_hooks, add_task, add_tool, context, to_json};
use crate::chord::Context;
use crate::durable::env::local::LocalExecutionEnv;
use crate::durable::errors::Result;
use crate::durable::harness::agent::{AGENT_DOC, configure};
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::live::LIVE_DOC;
use crate::durable::harness::scheduler::HookApi;
use crate::durable::harness::tool::{TOOL_TASK, ToolExecutionApi};
use crate::durable::harness::types::{
    AgentChange, BeforeToolDecision, DiagnosticSeverity, EnvFactory, ExtensionDefinition,
    ExtensionsChange, GenerationHooks, HookRegistration, Retain, SubmissionDraft, ToolControl,
    ToolDiagnostic, ToolExecutionMode, ToolExecutionResult, ToolFilter, ToolHooks,
    ToolOutputLimits, ToolRegistration, ToolsChange, UserInput,
};
use crate::durable::harness::{
    Conversation, Harness, define_extension, define_tool, hook, section,
};
use crate::durable::ids::TaskId;
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{NextTaskState, TaskDefinition, define_task};
use crate::durable::types::{
    ConversationOwnership, EntryDraft, EntryRecord, SubmissionStatus, TaskOptions, TaskQuery,
    TaskRecord,
};
use crate::providers::faux::{
    FauxDeferredOptions, FauxMessageOptions, FauxResponseStep, RegisterFauxProviderOptions,
    faux_assistant_message, faux_tool_call,
};
use crate::types::{
    AssistantMessage, ImageContent, Message, StopReason, ToolCall, ToolResultMessage, UserContent,
    UserMessage,
};

fn echo_parameters() -> JsonValue {
    json!({ "type": "object", "properties": { "text": { "type": "string" } } })
}

fn tool<F, Fut>(name: &str, execute: F) -> ToolRegistration
where
    F: Fn(JsonValue, ToolExecutionApi, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolExecutionResult>> + Send + 'static,
{
    define_tool(name, format!("The {name} tool"), echo_parameters(), execute)
}

fn empty() -> Result<ToolExecutionResult> {
    Ok(ToolExecutionResult {
        content: Some(Vec::new()),
        ..ToolExecutionResult::default()
    })
}

fn text_result(text: &str) -> Result<ToolExecutionResult> {
    Ok(ToolExecutionResult::text(text))
}

fn noop(name: &str) -> ToolRegistration {
    tool(name, |_, _, _| async { empty() })
}

fn sequential(mut tool: ToolRegistration) -> ToolRegistration {
    tool.execution_mode = Some(ToolExecutionMode::Sequential);
    tool
}

/// A tool-calling answer with one call per `(name, args, id)`.
fn calls_message(list: &[(&str, JsonValue, &str)]) -> AssistantMessage {
    faux_assistant_message(
        list.iter()
            .map(|(name, args, id)| faux_tool_call(*name, args.clone(), Some(id)))
            .collect::<Vec<_>>(),
        FauxMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxMessageOptions::default()
        },
    )
}

fn calls(list: &[(&str, JsonValue, &str)]) -> FauxResponseStep {
    calls_message(list).into()
}

fn done() -> FauxResponseStep {
    answer("done")
}

fn answer(text: &str) -> FauxResponseStep {
    answer_message(text).into()
}

fn answer_message(text: &str) -> AssistantMessage {
    faux_assistant_message(text, FauxMessageOptions::default())
}

struct Run {
    harness: Harness,
    root: Conversation,
    entries: Vec<EntryRecord>,
    status: SubmissionStatus,
}

async fn run(setup: &ChatSetup, responses: Vec<FauxResponseStep>) -> Run {
    run_prepared(setup, responses, ChatOptions::default(), |_, _| async {}).await
}

async fn run_prepared<F, Fut>(
    setup: &ChatSetup,
    responses: Vec<FauxResponseStep>,
    options: ChatOptions,
    prepare: F,
) -> Run
where
    F: FnOnce(Harness, Conversation) -> Fut,
    Fut: Future<Output = ()>,
{
    setup.faux.set_responses(responses);
    let (harness, root) = open_chat_with(Arc::new(MemoryStorage::new()), setup, options).await;
    prepare(harness.clone(), root.clone()).await;
    let status = submit(&root, "go").await;
    let entries = all_entries(&root).await;
    Run {
        harness,
        root,
        entries,
        status,
    }
}

async fn submit(conversation: &Conversation, text: &str) -> SubmissionStatus {
    conversation
        .submit(SubmissionDraft::input(text), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap()
        .status
}

fn results(entries: &[EntryRecord]) -> Vec<ToolResultMessage> {
    entries
        .iter()
        .filter(|entry| entry.kind == "pi.tool-result")
        .map(|entry| match &entry.model.as_ref().expect("model")[0] {
            Message::ToolResult(result) => result.clone(),
            other => panic!("not a tool result: {other:?}"),
        })
        .collect()
}

fn result_text(message: &ToolResultMessage) -> String {
    message
        .content
        .iter()
        .map(|item| match item {
            UserContent::Text(text) => text.text.clone(),
            UserContent::Image(_) => "[image]".to_string(),
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn by_id(entries: &[EntryRecord]) -> std::collections::HashMap<String, (bool, String)> {
    results(entries)
        .iter()
        .map(|result| {
            (
                result.tool_call_id.clone(),
                (result.is_error, result_text(result)),
            )
        })
        .collect()
}

fn system_json(entries: &[EntryRecord]) -> Vec<JsonValue> {
    entries
        .iter()
        .filter(|entry| entry.kind == "pi.system")
        .map(|entry| to_json(&entry.model.as_ref().unwrap()[0]))
        .collect()
}

async fn tasks_of(harness: &Harness, conversation: &Conversation) -> Vec<TaskRecord> {
    let id = conversation.id;
    harness
        .commit(
            move |tx| async move {
                Ok(tx
                    .scan_tasks(
                        TaskQuery {
                            conversation_id: Some(id),
                            ..TaskQuery::default()
                        },
                        20,
                        None,
                    )
                    .await?
                    .items)
            },
            &context(),
        )
        .await
        .unwrap()
}

fn before_tool<F, Fut>(handler: F) -> HookRegistration
where
    F: Fn(ToolCall, HookApi, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Option<BeforeToolDecision>>> + Send + 'static,
{
    hook(
        &*TOOL_TASK,
        ToolHooks {
            before_tool: Some(Arc::new(move |call, api, ctx| {
                Box::pin(handler(call, api, ctx))
            })),
            ..ToolHooks::default()
        },
    )
}

fn after_tool<F, Fut>(handler: F) -> HookRegistration
where
    F: Fn(ToolCall, ToolExecutionResult, HookApi, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Option<ToolExecutionResult>>> + Send + 'static,
{
    hook(
        &*TOOL_TASK,
        ToolHooks {
            after_tool: Some(Arc::new(move |call, result, api, ctx| {
                Box::pin(handler(call, result, api, ctx))
            })),
            ..ToolHooks::default()
        },
    )
}

fn generation_hooks(hooks: GenerationHooks) -> HookRegistration {
    hook(&*GENERATION_TASK, hooks)
}

fn on_yield(handler: impl Fn() -> Option<UserInput> + Send + Sync + 'static) -> HookRegistration {
    let handler = Arc::new(handler);
    generation_hooks(GenerationHooks {
        on_yield: Some(Arc::new(move |_, _, _| {
            let handler = handler.clone();
            Box::pin(async move { Ok(handler()) })
        })),
        ..GenerationHooks::default()
    })
}

fn after_response(
    handler: impl Fn(AssistantMessage) -> Result<()> + Send + Sync + 'static,
) -> HookRegistration {
    let handler = Arc::new(handler);
    generation_hooks(GenerationHooks {
        after_response: Some(Arc::new(move |message, _, _| {
            let handler = handler.clone();
            Box::pin(async move { handler(message) })
        })),
        ..GenerationHooks::default()
    })
}

fn message_text(message: &AssistantMessage) -> String {
    super::support::text_of(Some(&Message::Assistant(message.clone()))).unwrap_or_default()
}

fn entry_text(entry: &EntryRecord) -> Option<String> {
    super::support::text_of(entry.model.as_ref().and_then(|model| model.first()))
}

fn sleep_ms(ms: u64) -> tokio::time::Sleep {
    tokio::time::sleep(Duration::from_millis(ms))
}

type Log = Arc<Mutex<Vec<String>>>;

fn push(log: &Log, value: impl Into<String>) {
    log.lock().push(value.into());
}

fn logged(log: &Log) -> Vec<String> {
    log.lock().clone()
}

// ---- tool round ----

#[tokio::test]
async fn runs_input_tool_call_tool_result_and_answer_and_settles_the_input() {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        tool("echo", |args, _, _| async move {
            text_result(&format!("echo {}", args["text"].as_str().unwrap_or("")))
        }),
    );
    let Run {
        harness,
        root,
        entries,
        status,
    } = run(
        &setup,
        vec![calls(&[("echo", json!({ "text": "hi" }), "c1")]), done()],
    )
    .await;
    assert_eq!(status, SubmissionStatus::Done);
    assert_eq!(
        entries.iter().map(|e| e.kind.as_str()).collect::<Vec<_>>(),
        [
            "pi.user",
            "pi.system",
            "pi.assistant",
            "pi.tool-result",
            "pi.assistant"
        ]
    );
    let system = to_json(&entries[1].model.as_ref().unwrap()[0]);
    assert_eq!(system["toolsAdded"].as_array().unwrap().len(), 1);
    assert_eq!(system["toolsAdded"][0]["name"], json!("echo"));
    assert_eq!(
        system["toolsAdded"][0]["description"],
        json!("The echo tool")
    );
    assert!(system["toolsAdded"][0]["parameters"].is_object());
    let result = &results(&entries)[0];
    assert_eq!(
        (
            result.tool_call_id.as_str(),
            result.tool_name.as_str(),
            result.is_error
        ),
        ("c1", "echo", false)
    );
    assert_eq!(result_text(result), "echo hi");
    assert_eq!(to_json(&entries[3].data), json!({ "diagnostics": [] }));
    assert!(entries[3].by_task_id.is_some());
    assert_eq!(setup.faux.state().call_count, 2);
    assert_eq!(
        harness
            .snapshot(&*LIVE_DOC, root.id, &context())
            .await
            .unwrap(),
        Some(Default::default())
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn answers_calls_to_tools_the_request_did_not_offer_without_a_task() {
    let setup = chat_setup();
    add_tool(&setup.registry, noop("echo"));
    let Run {
        harness,
        root,
        entries,
        status,
    } = run(
        &setup,
        vec![
            calls(&[("ghost", json!({}), "c1"), ("echo", json!({}), "c2")]),
            done(),
        ],
    )
    .await;
    assert_eq!(status, SubmissionStatus::Done);
    let found = results(&entries);
    assert_eq!(
        (found[0].tool_call_id.as_str(), found[0].is_error),
        ("c1", true)
    );
    assert_eq!(
        result_text(&found[0]),
        "<harness>\n[error] Tool ghost is not available\n</harness>"
    );
    assert_eq!(
        (found[1].tool_call_id.as_str(), found[1].is_error),
        ("c2", false)
    );
    let ghost = entries
        .iter()
        .find(|entry| entry.kind == "pi.tool-result")
        .unwrap();
    assert_eq!(
        to_json(&ghost.data),
        json!({ "diagnostics": [{ "severity": "error", "code": "tool_unavailable", "message": "Tool ghost is not available" }] })
    );
    let tools = tasks_of(&harness, &root).await;
    assert_eq!(
        tools.iter().filter(|task| task.kind == "pi.tool").count(),
        1
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn answers_a_call_to_a_tool_deactivated_after_preparation_with_tool_unavailable() {
    let setup = chat_setup();
    let seen: Log = Log::default();
    let ran = seen.clone();
    add_tool(
        &setup.registry,
        tool("echo", move |_, _, _| {
            push(&ran, "ran");
            async { empty() }
        }),
    );
    let root_slot: Arc<Mutex<Option<Conversation>>> = Arc::default();
    let slot = root_slot.clone();
    let deactivate = FauxResponseStep::async_factory(move |_, _, _, _| {
        let root = slot.lock().clone().unwrap();
        async move {
            root.configure(AgentChange::default().tools(Vec::new()), &context())
                .await
                .unwrap();
            Ok(calls_message(&[("echo", json!({}), "c1")]))
        }
    });
    let result = run_prepared(
        &setup,
        vec![deactivate, done()],
        ChatOptions::default(),
        |_, root| async move {
            *root_slot.lock() = Some(root);
        },
    )
    .await;
    assert!(logged(&seen).is_empty());
    assert_eq!(
        result_text(&results(&result.entries)[0]),
        "<harness>\n[error] Tool echo is not available\n</harness>"
    );
    let systems = system_json(&result.entries);
    assert_eq!(
        systems.last().unwrap()["toolsRemoved"],
        json!([{ "name": "echo" }])
    );
    result.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn removes_unregistered_active_tools_from_the_offer_and_adds_them_back_after_re_registration()
{
    let setup = chat_setup();
    let echo = noop("echo");
    let registration = add_tool(&setup.registry, echo.clone());
    let first = run(&setup, vec![done()]).await;
    registration.dispose();
    setup
        .faux
        .set_responses([calls(&[("echo", json!({}), "c1")]), done()]);
    assert_eq!(submit(&first.root, "again").await, SubmissionStatus::Done);
    let entries = all_entries(&first.root).await;
    assert_eq!(
        system_json(&entries).last().unwrap()["toolsRemoved"],
        json!([{ "name": "echo" }])
    );
    assert!(results(&entries)[0].is_error);
    // The stored agent is not rewritten; the tool is only not resolved.
    assert!(first.root.agent(&context()).await.unwrap().tools.is_empty());

    add_tool(&setup.registry, echo);
    setup.faux.set_responses([done()]);
    submit(&first.root, "back").await;
    let entries = all_entries(&first.root).await;
    assert_eq!(
        system_json(&entries).last().unwrap()["toolsAdded"][0]["name"],
        json!("echo")
    );
    assert_eq!(
        system_json(&entries).last().unwrap()["toolsAdded"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    first.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn produces_tool_unavailable_when_the_implementation_is_unregistered_before_its_task_runs() {
    let setup = chat_setup();
    let second: Arc<Mutex<Option<super::support::Installed>>> = Arc::default();
    let handle = second.clone();
    add_tool(
        &setup.registry,
        sequential(tool("first", move |_, _, _| {
            if let Some(second) = handle.lock().as_ref() {
                second.dispose();
            }
            async { empty() }
        })),
    );
    *second.lock() = Some(add_tool(
        &setup.registry,
        tool("second", |_, _, _| async { text_result("ran") }),
    ));
    let Run {
        harness,
        entries,
        status,
        ..
    } = run(
        &setup,
        vec![
            calls(&[("first", json!({}), "c1"), ("second", json!({}), "c2")]),
            done(),
        ],
    )
    .await;
    assert_eq!(status, SubmissionStatus::Done);
    assert_eq!(
        result_text(&results(&entries)[1]),
        "<harness>\n[error] Tool second is not available\n</harness>"
    );
    harness.close(&context()).await.unwrap();
}

fn slow_tool(
    name: &str,
    events: &Log,
    during: impl Fn() + Send + Sync + 'static,
) -> ToolRegistration {
    let events = events.clone();
    let during = Arc::new(during);
    let label = name.to_string();
    tool(name, move |_, _, _| {
        let events = events.clone();
        let during = during.clone();
        let label = label.clone();
        async move {
            push(&events, format!("start {label}"));
            during();
            sleep_ms(20).await;
            push(&events, format!("end {label}"));
            empty()
        }
    })
}

#[tokio::test]
async fn reads_the_execution_mode_when_a_round_starts_and_keeps_it_for_the_round() {
    let setup = chat_setup();
    let events = Log::default();
    for name in ["a", "b"] {
        let settings = setup.clone();
        // Changed after the round started: not seen by this round.
        add_tool(
            &setup.registry,
            slow_tool(name, &events, move || {
                settings.settings(|s| s.tool_execution = Some(ToolExecutionMode::Parallel))
            }),
        );
    }
    // Changed while the model request runs: the round that follows uses it.
    let settings = setup.clone();
    let request = FauxResponseStep::factory(move |_, _, _, _| {
        settings.settings(|s| s.tool_execution = Some(ToolExecutionMode::Sequential));
        Ok(calls_message(&[
            ("a", json!({}), "c1"),
            ("b", json!({}), "c2"),
        ]))
    });
    let result = run(&setup, vec![request, done()]).await;
    assert_eq!(logged(&events), ["start a", "end a", "start b", "end b"]);
    result.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn runs_a_round_in_parallel_by_default_and_sequentially_when_configured_or_required_by_a_tool()
 {
    async fn trace(setup: &ChatSetup) -> Vec<String> {
        let events = Log::default();
        add_tool(&setup.registry, slow_tool("a", &events, || {}));
        add_tool(&setup.registry, slow_tool("b", &events, || {}));
        let result = run(
            setup,
            vec![
                calls(&[("a", json!({}), "c1"), ("b", json!({}), "c2")]),
                done(),
            ],
        )
        .await;
        result.harness.close(&context()).await.unwrap();
        logged(&events)
    }
    assert_eq!(trace(&chat_setup()).await[..2], ["start a", "start b"]);
    let configured = chat_setup();
    configured.settings(|s| s.tool_execution = Some(ToolExecutionMode::Sequential));
    assert_eq!(
        trace(&configured).await,
        ["start a", "end a", "start b", "end b"]
    );

    let per_tool = chat_setup();
    let events = Log::default();
    let log = events.clone();
    add_tool(
        &per_tool.registry,
        sequential(tool("a", move |_, _, _| {
            let log = log.clone();
            async move {
                sleep_ms(20).await;
                push(&log, "a");
                empty()
            }
        })),
    );
    let log = events.clone();
    add_tool(
        &per_tool.registry,
        tool("b", move |_, _, _| {
            push(&log, "b");
            async { empty() }
        }),
    );
    let result = run(
        &per_tool,
        vec![
            calls(&[("a", json!({}), "c1"), ("b", json!({}), "c2")]),
            done(),
        ],
    )
    .await;
    let tasks = tasks_of(&result.harness, &result.root).await;
    let mut tools: Vec<_> = tasks.iter().filter(|task| task.kind == "pi.tool").collect();
    tools.sort_by_key(|task| task.id);
    // The generation owns both tools and creates the second only after the first ended.
    let generation = tasks
        .iter()
        .find(|task| task.kind == "pi.generation")
        .unwrap();
    assert_eq!(
        tools.iter().map(|task| task.owner).collect::<Vec<_>>(),
        [Some(generation.id), Some(generation.id)]
    );
    assert_eq!(logged(&events), ["a", "b"]);
    result.harness.close(&context()).await.unwrap();
}

// ---- tool results ----

#[tokio::test]
async fn uses_retained_output_and_the_last_details_when_the_result_omits_them_with_diagnostics_in_order()
 {
    let setup = chat_setup();
    let mut log = tool("log", |_, api, ctx| async move {
        api.output("line 1\n")?;
        api.output_bytes(b"line 2\nline 3\n")?;
        api.diagnostic(ToolDiagnostic {
            severity: DiagnosticSeverity::Info,
            message: "from api".into(),
            code: None,
        })?;
        api.details(json!({ "step": 1 }), &ctx).await?;
        api.details(json!({ "step": 2 }), &ctx).await?;
        Ok(ToolExecutionResult {
            diagnostics: Some(vec![ToolDiagnostic {
                severity: DiagnosticSeverity::Warn,
                message: "from result".into(),
                code: None,
            }]),
            ..ToolExecutionResult::default()
        })
    });
    log.output_limits = Some(ToolOutputLimits {
        max_lines: Some(2),
        ..ToolOutputLimits::default()
    });
    add_tool(&setup.registry, log);
    let Run {
        harness, entries, ..
    } = run(&setup, vec![calls(&[("log", json!({}), "c1")]), done()]).await;
    let result = &results(&entries)[0];
    assert_eq!(result.details, Some(json!({ "step": 2 })));
    assert_eq!(
        result_text(result),
        "line 1\nline 2\n|<harness>\n[info] from api\n[warn] from result\n[warn] Output truncated to its beginning: 1 lines, 7 bytes dropped\n</harness>"
    );
    let entry = entries
        .iter()
        .find(|entry| entry.kind == "pi.tool-result")
        .unwrap();
    let codes: Vec<JsonValue> = to_json(&entry.data)["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|diagnostic| diagnostic.get("code").cloned().unwrap_or(JsonValue::Null))
        .collect();
    assert_eq!(
        codes,
        [JsonValue::Null, JsonValue::Null, json!("truncated")]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn bounds_explicit_text_content_and_keeps_other_content() {
    let setup = chat_setup();
    let mut big = tool("big", |_, _, _| async {
        Ok(ToolExecutionResult {
            content: Some(vec![
                UserContent::text("a\nb\n"),
                UserContent::Image(ImageContent {
                    data: "AAAA".into(),
                    mime_type: "image/png".into(),
                }),
                UserContent::text("c\nd\n"),
            ]),
            ..ToolExecutionResult::default()
        })
    });
    big.output_limits = Some(ToolOutputLimits {
        max_lines: Some(2),
        retain: Some(Retain::Tail),
        ..ToolOutputLimits::default()
    });
    add_tool(&setup.registry, big);
    let Run {
        harness, entries, ..
    } = run(&setup, vec![calls(&[("big", json!({}), "c1")]), done()]).await;
    assert_eq!(
        result_text(&results(&entries)[0]),
        "[image]|c\nd\n|<harness>\n[warn] Output truncated to its end: 2 lines, 4 bytes dropped\n</harness>"
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn turns_a_throw_into_a_tool_error_result_with_the_partial_output() {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        tool("fail", |_, api, _| async move {
            api.output("partial\n")?;
            Err(crate::durable::errors::Error::message("boom"))
        }),
    );
    let Run {
        harness,
        entries,
        status,
        ..
    } = run(&setup, vec![calls(&[("fail", json!({}), "c1")]), done()]).await;
    assert_eq!(status, SubmissionStatus::Done);
    let result = &results(&entries)[0];
    assert!(result.is_error);
    assert_eq!(
        result_text(result),
        "partial\n|<harness>\n[error] boom\n</harness>"
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn validates_arguments_before_and_after_before_tool_and_applies_blocks_and_replacements() {
    let setup = chat_setup();
    let seen: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let sink = seen.clone();
    add_tool(
        &setup.registry,
        tool("echo", move |args, _, _| {
            sink.lock().push(args);
            async { empty() }
        }),
    );
    add_hooks(
        &setup.registry,
        before_tool(|call, _, _| async move {
            let arguments = |value: JsonValue| {
                Ok(Some(BeforeToolDecision {
                    arguments: Some(serde_json::from_value(value).unwrap()),
                    block: None,
                }))
            };
            match call.id.as_str() {
                "block" => Ok(Some(BeforeToolDecision {
                    arguments: None,
                    block: Some("not today".into()),
                })),
                "throw" => Err(crate::durable::errors::Error::message("hook failed")),
                "bad" => arguments(json!({ "text": { "not": "a string" } })),
                _ => {
                    let text = call.arguments["text"].as_str().unwrap_or("undefined");
                    arguments(json!({ "text": format!("{text}!") }))
                }
            }
        }),
    );
    let Run {
        harness, entries, ..
    } = run(
        &setup,
        vec![
            calls(&[
                ("echo", json!({ "text": 1 }), "coerced"),
                ("echo", json!({ "text": { "not": "a string" } }), "invalid"),
                ("echo", json!({}), "block"),
                ("echo", json!({}), "throw"),
                ("echo", json!({ "text": "x" }), "bad"),
                ("echo", json!({ "text": "x" }), "ok"),
            ]),
            done(),
        ],
    )
    .await;
    // Parallel tools append their results in completion order.
    let found = by_id(&entries);
    assert_eq!(
        found["block"],
        (
            true,
            "<harness>\n[error] Tool call blocked: not today\n</harness>".to_string()
        )
    );
    assert_eq!(
        found["throw"],
        (
            true,
            "<harness>\n[error] Tool call blocked: hook failed\n</harness>".to_string()
        )
    );
    assert!(found["bad"].0);
    assert!(found["bad"].1.contains("Validation failed"));
    assert_eq!(found["ok"], (false, String::new()));
    assert!(found["invalid"].0);
    assert!(found["invalid"].1.contains("Validation failed"));
    // A number is coerced to a string before the first validation.
    assert_eq!(found["coerced"], (false, String::new()));
    let mut seen = seen.lock().clone();
    seen.sort_by_key(ToString::to_string);
    assert_eq!(seen, [json!({ "text": "1!" }), json!({ "text": "x!" })]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn repairs_arguments_with_prepare_arguments_before_validation_and_a_throwing_repair_is_invalid()
 {
    let setup = chat_setup();
    let seen: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let sink = seen.clone();
    let mut echo = tool("echo", move |args, _, _| {
        sink.lock().push(args);
        async { empty() }
    });
    echo.prepare_arguments = Some(Arc::new(|args| {
        let text = args.get("text").cloned();
        match text {
            Some(JsonValue::String(text)) if text == "throw" => {
                Err(crate::durable::errors::Error::message("cannot repair"))
            }
            Some(JsonValue::Number(number)) => Ok(json!({ "text": format!("#{number}") })),
            _ => Ok(args),
        }
    }));
    add_tool(&setup.registry, echo);
    let Run {
        harness, entries, ..
    } = run(
        &setup,
        vec![
            calls(&[
                ("echo", json!({ "text": 7 }), "fixed"),
                ("echo", json!({ "text": "throw" }), "broken"),
            ]),
            done(),
        ],
    )
    .await;
    let found = by_id(&entries);
    assert_eq!(found["fixed"].1, "");
    assert_eq!(
        found["broken"].1,
        "<harness>\n[error] cannot repair\n</harness>"
    );
    assert_eq!(*seen.lock(), [json!({ "text": "#7" })]);
    // The stored call keeps what the model sent.
    let assistant = to_json(&entries[2].model.as_ref().unwrap()[0]);
    let call = assistant["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "toolCall")
        .unwrap()
        .clone();
    assert_eq!(call["arguments"], json!({ "text": 7 }));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn lets_the_first_before_tool_block_win_and_skips_later_handlers() {
    let setup = chat_setup();
    add_tool(&setup.registry, noop("echo"));
    let asked = Log::default();
    for name in ["first", "second"] {
        let asked = asked.clone();
        add_hooks(
            &setup.registry,
            before_tool(move |_, _, _| {
                push(&asked, name);
                async move {
                    Ok(Some(BeforeToolDecision {
                        arguments: None,
                        block: Some(format!("{name} says no")),
                    }))
                }
            }),
        );
    }
    let Run {
        harness, entries, ..
    } = run(&setup, vec![calls(&[("echo", json!({}), "c1")]), done()]).await;
    assert_eq!(logged(&asked), ["first"]);
    assert_eq!(
        result_text(&results(&entries)[0]),
        "<harness>\n[error] Tool call blocked: first says no\n</harness>"
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn chains_after_tool_replacements_and_observes_the_round_with_after_tools() {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        tool("echo", |_, _, _| async { text_result("raw") }),
    );
    add_hooks(
        &setup.registry,
        after_tool(|_, result, _, _| async move {
            Ok(Some(ToolExecutionResult {
                content: Some(vec![UserContent::text("first")]),
                ..result
            }))
        }),
    );
    add_hooks(
        &setup.registry,
        after_tool(|_, result, _, _| async move {
            let text = result
                .content
                .iter()
                .flatten()
                .map(|item| match item {
                    UserContent::Text(text) => text.text.clone(),
                    UserContent::Image(_) => "[image]".into(),
                })
                .collect::<Vec<_>>()
                .join("|");
            Ok(Some(ToolExecutionResult {
                details: Some(json!({ "replaced": text })),
                ..result
            }))
        }),
    );
    let observed: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let sink = observed.clone();
    add_hooks(
        &setup.registry,
        generation_hooks(GenerationHooks {
            after_tools: Some(Arc::new(move |assistant, entries, _, _| {
                sink.lock().push(json!([assistant, entries]));
                Box::pin(async { Ok(()) })
            })),
            ..GenerationHooks::default()
        }),
    );
    let Run {
        harness, entries, ..
    } = run(&setup, vec![calls(&[("echo", json!({}), "c1")]), done()]).await;
    let result = &results(&entries)[0];
    assert_eq!(result_text(result), "first");
    assert_eq!(result.details, Some(json!({ "replaced": "first" })));
    let result_entry = entries
        .iter()
        .find(|entry| entry.kind == "pi.tool-result")
        .unwrap();
    assert_eq!(
        *observed.lock(),
        [json!([entries[2].id, [result_entry.id]])]
    );
    harness.close(&context()).await.unwrap();
}

#[derive(serde::Serialize, serde::Deserialize)]
struct NeverPhase {
    phase: String,
}

fn never_task(name: &str) -> crate::durable::tasks::Task<JsonValue, NeverPhase, JsonValue> {
    define_task(
        TaskDefinition::new(name, 1, |_: &JsonValue| NeverPhase {
            phase: "never".into(),
        })
        .phase("never", |_, _, _| async { Ok(()) })
        .abort(|_, _, _| async { Ok(()) }),
    )
}

#[tokio::test]
async fn runs_the_hooks_of_the_selected_extensions_and_a_task_owned_child_copies_its_owners_selection()
 {
    let setup = chat_setup();
    let called_in: Arc<Mutex<Vec<crate::durable::ids::ConversationId>>> = Arc::default();
    let echo = define_extension(ExtensionDefinition {
        tools: vec![noop("echo")],
        ..ExtensionDefinition::new("echo")
    });
    let sink = called_in.clone();
    let audit = define_extension(ExtensionDefinition {
        hooks: vec![before_tool(move |_, api, _| {
            sink.lock().push(api.conversation_id());
            async { Ok(None) }
        })],
        ..ExtensionDefinition::new("audit")
    });
    setup.registry.install(echo.clone()).unwrap();
    setup.registry.install(audit.clone()).unwrap();
    // Audit is installed but not in the default selection.
    setup.settings(|s| s.extensions = Some(vec![echo]));
    let first = run(&setup, vec![done()]).await;
    first
        .root
        .configure(
            AgentChange {
                extensions: Some(Some(ExtensionsChange::Edit {
                    add: Some(vec![audit]),
                    remove: None,
                })),
                ..AgentChange::default()
            },
            &context(),
        )
        .await
        .unwrap();
    // An owner task no registered definition takes stays live and pending.
    let owner = never_task("test.owner");
    let root_id = first.root.id;
    let child_id = first
        .harness
        .commit(
            move |tx| async move {
                let task_id = tx
                    .create_task(&owner, json!({}), TaskOptions::conversation(Some(root_id)))
                    .await?
                    .erase();
                Ok(tx
                    .create_conversation(ConversationOwnership::Task { task_id })
                    .await?
                    .id)
            },
            &context(),
        )
        .await
        .unwrap();
    let child = first
        .harness
        .conversation(child_id, &context())
        .await
        .unwrap()
        .unwrap();
    let other = first
        .harness
        .create_conversation(
            crate::durable::harness::types::ConversationCreateOptions {
                agent: Some(AgentChange::default().model(faux_model())),
                ..crate::durable::harness::types::ConversationCreateOptions::ownerless()
            },
            &context(),
        )
        .await
        .unwrap();
    for conversation in [&first.root, &child, &other] {
        setup
            .faux
            .set_responses([calls(&[("echo", json!({}), "c1")]), done()]);
        submit(conversation, "go").await;
    }
    assert_eq!(*called_in.lock(), [first.root.id, child.id]);
    first.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn applies_add_tools_and_terminates_only_when_every_result_of_the_round_asks_to() {
    let setup = chat_setup();
    let stop = tool("stop", |_, _, _| async {
        Ok(ToolExecutionResult {
            content: Some(Vec::new()),
            control: Some(ToolControl {
                terminate: Some(true),
                ..ToolControl::default()
            }),
            ..ToolExecutionResult::default()
        })
    });
    let grow = tool("grow", |_, _, _| async {
        Ok(ToolExecutionResult {
            content: Some(Vec::new()),
            control: Some(ToolControl {
                add_tools: Some(vec!["extra".into(), "stop".into()]),
                ..ToolControl::default()
            }),
            ..ToolExecutionResult::default()
        })
    });
    add_tool(&setup.registry, stop.clone());
    add_tool(&setup.registry, grow.clone());
    add_tool(&setup.registry, noop("extra"));
    let first = run_prepared(
        &setup,
        vec![calls(&[("stop", json!({}), "c1")])],
        ChatOptions::default(),
        |_, root| async move {
            root.configure(AgentChange::default().tools(vec![stop, grow]), &context())
                .await
                .unwrap();
        },
    )
    .await;
    assert_eq!(first.status, SubmissionStatus::Done);
    assert_eq!(first.entries.last().unwrap().kind, "pi.tool-result");
    let settled = tasks_of(&first.harness, &first.root).await;
    assert!(
        settled
            .iter()
            .all(|task| to_json(&task.state)["status"] == "terminal")
    );

    setup.faux.set_responses([
        calls(&[("stop", json!({}), "c1"), ("grow", json!({}), "c2")]),
        done(),
    ]);
    assert_eq!(submit(&first.root, "again").await, SubmissionStatus::Done);
    assert_eq!(
        all_entries(&first.root).await.last().unwrap().kind,
        "pi.assistant"
    );
    // addTools appends to the stored tool array, skipping names it already holds.
    assert_eq!(
        first
            .harness
            .snapshot(&*AGENT_DOC, first.root.id, &context())
            .await
            .unwrap()
            .unwrap()
            .tools,
        Some(ToolFilter::List(vec![
            "stop".into(),
            "grow".into(),
            "extra".into()
        ]))
    );
    first.harness.close(&context()).await.unwrap();
}

// ---- generation hooks ----

#[tokio::test]
async fn replaces_request_messages_observes_responses_and_continues_on_yield() {
    let setup = chat_setup();
    let requests: Arc<Mutex<Vec<Vec<String>>>> = Arc::default();
    let sink = requests.clone();
    let record = FauxResponseStep::factory(move |request, _, _, _| {
        let mut requests = sink.lock();
        requests.push(
            request
                .messages
                .iter()
                .map(|message| {
                    let role = to_json(message)["role"].as_str().unwrap().to_string();
                    format!(
                        "{role}:{}",
                        super::support::text_of(Some(message)).unwrap_or_default()
                    )
                })
                .collect(),
        );
        Ok(answer_message(&format!("answer {}", requests.len())))
    });
    add_hooks(
        &setup.registry,
        generation_hooks(GenerationHooks {
            before_request: Some(Arc::new(|mut messages, _, _| {
                messages.push(Message::User(UserMessage {
                    content: "injected".into(),
                    timestamp: 0,
                }));
                Box::pin(async move { Ok(Some(messages)) })
            })),
            ..GenerationHooks::default()
        }),
    );
    let responses = Log::default();
    let sink = responses.clone();
    add_hooks(
        &setup.registry,
        after_response(move |message| {
            push(&sink, message_text(&message));
            Ok(())
        }),
    );
    let yields = Arc::new(Mutex::new(0));
    add_hooks(
        &setup.registry,
        on_yield(move || {
            let mut yields = yields.lock();
            *yields += 1;
            (*yields == 1).then(|| UserInput::Parts(vec![UserContent::text("keep going")]))
        }),
    );
    let Run {
        harness,
        entries,
        status,
        ..
    } = run(&setup, vec![record.clone(), record]).await;
    assert_eq!(status, SubmissionStatus::Done);
    let first = requests.lock()[0].clone();
    assert_eq!(first[first.len() - 2..], ["user:go", "user:injected"]);
    assert_eq!(logged(&responses), ["answer 1", "answer 2"]);
    assert_eq!(
        entries.iter().map(|e| e.kind.as_str()).collect::<Vec<_>>(),
        ["pi.user", "pi.assistant", "pi.user", "pi.assistant"]
    );
    // The injected message was used for the request only.
    assert!(
        !entries
            .iter()
            .any(|entry| entry_text(entry).as_deref() == Some("injected"))
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_the_runs_input_open_across_an_on_yield_continuation_and_answers_it_with_the_final_answer()
 {
    let setup = chat_setup();
    let yields = Arc::new(Mutex::new(0));
    add_hooks(
        &setup.registry,
        on_yield(move || {
            let mut yields = yields.lock();
            *yields += 1;
            (*yields == 1).then(|| UserInput::Text("again".into()))
        }),
    );
    let harness_slot: Arc<Mutex<Option<(Harness, Conversation)>>> = Arc::default();
    let observed: Arc<Mutex<Option<(crate::durable::ids::SubmissionId, SubmissionStatus)>>> =
        Arc::default();
    let (slot, sink) = (harness_slot.clone(), observed.clone());
    let second = FauxResponseStep::async_factory(move |_, _, _, _| {
        let (harness, root) = slot.lock().clone().unwrap();
        let sink = sink.clone();
        async move {
            let live = harness
                .snapshot(&*LIVE_DOC, root.id, &context())
                .await
                .unwrap()
                .unwrap();
            let input = live.run.unwrap().inputs[0];
            let status = harness
                .submission(input, &context())
                .await
                .unwrap()
                .unwrap()
                .status(&context())
                .await
                .unwrap()
                .status;
            *sink.lock() = Some((input, status));
            Ok(answer_message("second"))
        }
    });
    let result = run_prepared(
        &setup,
        vec![answer("first"), second],
        ChatOptions::default(),
        |harness, root| async move {
            *harness_slot.lock() = Some((harness, root));
        },
    )
    .await;
    let (input, status) = observed.lock().unwrap();
    assert_eq!(status, SubmissionStatus::Placed);
    let answers: Vec<_> = result
        .entries
        .iter()
        .filter(|entry| entry.kind == "pi.assistant")
        .collect();
    let settled = result
        .harness
        .submission(input, &context())
        .await
        .unwrap()
        .unwrap()
        .status(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    assert_eq!(settled.answer, Some(answers[1].id));
    result.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn observes_responses_that_arrive_by_polling_a_deferred_request() {
    let setup = chat_setup_with(RegisterFauxProviderOptions {
        deferred: Some(FauxDeferredOptions {
            pending_fetches: Some(1),
            poll_after_ms: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    let observed = Log::default();
    let sink = observed.clone();
    add_hooks(
        &setup.registry,
        after_response(move |message| {
            push(
                &sink,
                format!(
                    "{}:{}",
                    crate::durable::harness::compaction::stop_reason_str(message.stop_reason),
                    message_text(&message)
                ),
            );
            Ok(())
        }),
    );
    setup.settings(|s| {
        s.stream = Some(crate::durable::harness::types::ConversationStreamOptions {
            deferred: Some(crate::durable::harness::types::DeferredOption::Enabled(
                true,
            )),
            ..Default::default()
        })
    });
    let result = run(&setup, vec![answer("late")]).await;
    assert_eq!(result.status, SubmissionStatus::Done);
    // The still-deferred results are not terminal.
    assert_eq!(logged(&observed), ["stop:late"]);
    result.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn lets_the_first_on_yield_continuation_win_and_reports_throws_without_stopping_later_handlers()
 {
    let setup = chat_setup();
    let called = Log::default();
    let sink = called.clone();
    add_hooks(
        &setup.registry,
        after_response(move |_| {
            push(&sink, "throwing observer");
            Err(crate::durable::errors::Error::message("observer failed"))
        }),
    );
    let sink = called.clone();
    add_hooks(
        &setup.registry,
        after_response(move |_| {
            push(&sink, "next observer");
            Ok(())
        }),
    );
    let yields = Arc::new(Mutex::new(0));
    add_hooks(
        &setup.registry,
        on_yield(move || {
            let mut yields = yields.lock();
            *yields += 1;
            (*yields == 1).then(|| UserInput::Text("first".into()))
        }),
    );
    let sink = called.clone();
    add_hooks(
        &setup.registry,
        on_yield(move || {
            push(&sink, "second onYield");
            let count = logged(&sink)
                .iter()
                .filter(|name| *name == "second onYield")
                .count();
            (count == 1).then(|| UserInput::Text("second".into()))
        }),
    );
    let Run {
        harness, entries, ..
    } = run(&setup, vec![answer("a"), answer("b"), answer("c")]).await;
    // The first continuation skips the second handler; on the next answer the second handler's continuation wins.
    let users: Vec<_> = entries
        .iter()
        .filter(|entry| entry.kind == "pi.user")
        .map(|entry| entry_text(entry).unwrap())
        .collect();
    assert_eq!(users, ["go", "first", "second"]);
    let called = logged(&called);
    assert_eq!(called.iter().filter(|n| *n == "second onYield").count(), 2);
    assert_eq!(called.iter().filter(|n| *n == "next observer").count(), 3);
    assert!(
        setup
            .reports
            .messages()
            .iter()
            .any(|message| message.contains("observer failed"))
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_durable_hook_decisions_in_task_memos() {
    let setup = chat_setup();
    let asked = Arc::new(Mutex::new(0));
    add_tool(&setup.registry, noop("echo"));
    let counter = asked.clone();
    add_hooks(
        &setup.registry,
        before_tool(move |_, api, ctx| {
            *counter.lock() += 1;
            async move {
                let decision: String = api
                    .memo_with("approval:decision", "approved".to_string(), &ctx)
                    .await?;
                let again: String = api
                    .memo_with("approval:decision", "denied".to_string(), &ctx)
                    .await?;
                assert_eq!(again, decision);
                Ok(None)
            }
        }),
    );
    let result = run(&setup, vec![calls(&[("echo", json!({}), "c1")]), done()]).await;
    assert_eq!(*asked.lock(), 1);
    result.harness.close(&context()).await.unwrap();
}

// ---- tool execution api ----

#[derive(serde::Serialize, serde::Deserialize)]
struct ChildInput {
    n: i64,
}

#[tokio::test]
async fn builds_the_environment_per_call_from_the_conversations_cwd_and_runs_commits_memos_and_child_tasks()
 {
    let setup = chat_setup();
    let child: crate::durable::tasks::Task<ChildInput, NeverPhase, JsonValue> = define_task(
        TaskDefinition::new("test.child", 1, |_: &ChildInput| NeverPhase {
            phase: "run".into(),
        })
        .phase("run", |task, runtime, ctx| async move {
            let n = task.input.n;
            runtime
                .commit(
                    move |_, _| async move { Ok(Some(NextTaskState::completed(json!(n * 2)))) },
                    &ctx,
                )
                .await
        })
        .abort(|_, _, _| async { Ok(()) }),
    );
    add_task(&setup.registry, child.clone().into());
    let seen: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let sink = seen.clone();
    let probe = tool("probe", move |_, api, ctx| {
        let sink = sink.clone();
        let child = child.clone();
        async move {
            let record = |value: JsonValue| sink.lock().push(value);
            record(json!(api.env().map(|env| env.cwd())));
            record(json!(
                api.agent(&ctx)
                    .await?
                    .tools
                    .iter()
                    .map(|tool| tool.name.clone())
                    .collect::<Vec<_>>()
            ));
            record(json!(api.registry().extension("tool:probe").is_some()));
            let conversation_id = api.conversation_id();
            let call_id = api.call_id().to_string();
            let entry = api
                .commit(
                    move |tx| async move {
                        // The next call runs in the new directory: its environment is built when it executes.
                        configure(&tx, conversation_id, &AgentChange::default().cwd("/")).await?;
                        tx.append_entry(
                            conversation_id,
                            EntryDraft::new("test.note").data(json!(call_id)),
                        )
                        .await
                    },
                    &ctx,
                )
                .await?;
            record(json!(entry.by_task_id == Some(api.task_id())));
            record(json!(api.memo_with("m", 1, &ctx).await?));
            record(json!(api.memo_with("m", 2, &ctx).await?));
            let id = api
                .create_task(
                    &child,
                    ChildInput { n: 21 },
                    TaskOptions::conversation(None),
                    &ctx,
                )
                .await?;
            let done = api.wait_for_task(id, &ctx).await?;
            record(to_json(&done.state)["outcome"].clone());
            empty()
        }
    });
    add_tool(&setup.registry, probe);
    setup.faux.set_responses([
        calls(&[("probe", json!({}), "c1"), ("probe", json!({}), "c2")]),
        done(),
    ]);
    setup.settings(|s| s.tool_execution = Some(ToolExecutionMode::Sequential));
    let targets: Arc<Mutex<Vec<(crate::durable::ids::ConversationId, Option<String>)>>> =
        Arc::default();
    let sink = targets.clone();
    let env: EnvFactory = Arc::new(move |target, _| {
        sink.lock()
            .push((target.conversation_id, target.cwd.clone()));
        let env: Arc<dyn crate::durable::env::ExecutionEnv> = Arc::new(LocalExecutionEnv::at(
            target.cwd.unwrap_or_else(|| "/tmp".into()),
        ));
        Box::pin(async move { Ok(Some(env)) })
    });
    let (harness, root) = open_chat_with(
        Arc::new(MemoryStorage::new()),
        &setup,
        ChatOptions {
            env: Some(env),
            ..ChatOptions::default()
        },
    )
    .await;
    root.configure(AgentChange::default().cwd("/tmp"), &context())
        .await
        .unwrap();
    submit(&root, "go").await;
    let outcome = json!({ "status": "completed", "result": 42 });
    let call = |cwd: &str| {
        vec![
            json!(cwd),
            json!(["probe"]),
            json!(true),
            json!(true),
            json!(1),
            json!(1),
            outcome.clone(),
        ]
    };
    assert_eq!(*seen.lock(), [call("/tmp"), call("/")].concat());
    let targets = targets.lock().clone();
    assert!(targets.contains(&(root.id, Some("/tmp".into()))));
    assert!(targets.contains(&(root.id, Some("/".into()))));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn answers_a_throwing_environment_with_a_tool_error_result_and_reports_it_once_while_preparing()
 {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        tool("probe", |_, _, _| async { text_result("ran") }),
    );
    let env: EnvFactory = Arc::new(|_, _| {
        Box::pin(async { Err(crate::durable::errors::Error::message("no sandbox")) })
    });
    let Run {
        harness, entries, ..
    } = run_prepared(
        &setup,
        vec![calls(&[("probe", json!({}), "c1")]), done()],
        ChatOptions {
            env: Some(env),
            ..ChatOptions::default()
        },
        |_, _| async {},
    )
    .await;
    let result = &results(&entries)[0];
    assert!(result.is_error);
    assert_eq!(
        result_text(result),
        "<harness>\n[error] no sandbox\n</harness>"
    );
    // Each preparation reports the failure and renders without an environment.
    assert_eq!(
        setup
            .reports
            .messages()
            .iter()
            .filter(|message| message.contains("no sandbox"))
            .count(),
        2
    );
    harness.close(&context()).await.unwrap();
}

// ---- tool progress and lifetime ----

#[tokio::test]
async fn applies_the_default_output_limits() {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        tool("lines", |_, api, _| async move {
            for index in 1..=2500 {
                api.output(&format!("{index}\n"))?;
            }
            Ok(ToolExecutionResult::default())
        }),
    );
    let Run {
        harness, entries, ..
    } = run(&setup, vec![calls(&[("lines", json!({}), "c1")]), done()]).await;
    let text = result_text(&results(&entries)[0]);
    assert!(text.starts_with("1\n2\n"));
    assert!(text.ends_with(
        "\n2000\n|<harness>\n[warn] Output truncated to its beginning: 500 lines, 2500 bytes dropped\n</harness>"
    ));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn sanitizes_running_output_but_keeps_explicit_result_content_as_the_tool_returned_it() {
    let setup = chat_setup();
    let slot_output: Arc<Mutex<Option<String>>> = Arc::default();
    let sink = slot_output.clone();
    add_tool(
        &setup.registry,
        tool("noisy", move |_, api, ctx| {
            let sink = sink.clone();
            async move {
                api.output("a\u{7}b\r\n")?;
                api.details(json!({ "ready": true }), &ctx).await?;
                *sink.lock() = api
                    .snapshot(&*LIVE_DOC, api.conversation_id(), &ctx)
                    .await?
                    .and_then(|live| live.tools)
                    .and_then(|tools| tools[0].output.clone());
                Ok(ToolExecutionResult::default())
            }
        }),
    );
    add_tool(
        &setup.registry,
        tool("explicit", |_, _, _| async { text_result("c\u{1b}d") }),
    );
    let Run {
        harness, entries, ..
    } = run(
        &setup,
        vec![
            calls(&[("noisy", json!({}), "c1"), ("explicit", json!({}), "c2")]),
            done(),
        ],
    )
    .await;
    assert_eq!(slot_output.lock().as_deref(), Some("ab\n"));
    let found = by_id(&entries);
    assert_eq!(found["c1"].1, "ab\n");
    assert_eq!(found["c2"].1, "c\u{1b}d");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn drops_control_keys_set_to_undefined_instead_of_faulting() {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        tool("grow", |_, _, _| async {
            Ok(ToolExecutionResult {
                content: Some(Vec::new()),
                control: Some(ToolControl {
                    add_tools: Some(vec!["extra".into()]),
                    terminate: None,
                    ..ToolControl::default()
                }),
                ..ToolExecutionResult::default()
            })
        }),
    );
    let extra = noop("extra");
    add_tool(&setup.registry, extra.clone());
    let Run {
        harness,
        root,
        status,
        ..
    } = run_prepared(
        &setup,
        vec![calls(&[("grow", json!({}), "c1")]), done()],
        ChatOptions::default(),
        |_, root| async move {
            root.configure(
                AgentChange {
                    tools: Some(Some(ToolsChange::Remove(vec![extra]))),
                    ..AgentChange::default()
                },
                &context(),
            )
            .await
            .unwrap();
        },
    )
    .await;
    assert_eq!(status, SubmissionStatus::Done);
    // addTools deletes the name from a stored `{ remove }` filter.
    assert_eq!(
        harness
            .snapshot(&*AGENT_DOC, root.id, &context())
            .await
            .unwrap()
            .unwrap()
            .tools,
        Some(ToolFilter::Remove { remove: Vec::new() })
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn uses_explicit_null_details_instead_of_the_last_reported_value() {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        tool("null", |_, api, ctx| async move {
            api.details(json!({ "old": 1 }), &ctx).await?;
            Ok(ToolExecutionResult {
                content: Some(Vec::new()),
                details: Some(JsonValue::Null),
                ..ToolExecutionResult::default()
            })
        }),
    );
    let Run {
        harness, entries, ..
    } = run(&setup, vec![calls(&[("null", json!({}), "c1")]), done()]).await;
    assert_eq!(results(&entries)[0].details, Some(JsonValue::Null));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn settles_details_promises_with_coalesced_progress_commits_and_the_terminal_commit() {
    let setup = chat_setup();
    let settled = Log::default();
    let sink = settled.clone();
    add_tool(
        &setup.registry,
        tool("details", move |_, api, ctx| {
            let sink = sink.clone();
            async move {
                // Three updates in one throttle window coalesce; the last is still pending when execute() returns.
                let spawn = |n: i64, name: &'static str| {
                    let (api, ctx, sink) = (api.clone(), ctx.clone(), sink.clone());
                    tokio::spawn(async move {
                        if api.details(json!({ "n": n }), &ctx).await.is_ok() {
                            push(&sink, name);
                        }
                    })
                };
                let first = spawn(1, "first");
                let _second = spawn(2, "second");
                first.await.unwrap();
                let _third = spawn(3, "third");
                // Let the third update record its value before returning, as the synchronous TS call does.
                tokio::task::yield_now().await;
                empty()
            }
        }),
    );
    let Run {
        harness, entries, ..
    } = run(&setup, vec![calls(&[("details", json!({}), "c1")]), done()]).await;
    wait_for(|| {
        let settled = settled.clone();
        async move { settled.lock().len() == 3 }
    })
    .await;
    assert_eq!(logged(&settled), ["first", "second", "third"]);
    assert_eq!(results(&entries)[0].details, Some(json!({ "n": 3 })));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn finishes_a_call_under_the_implementation_it_resolved_when_the_tool_is_replaced_mid_call() {
    let setup = chat_setup();
    let started = Reached::default();
    let finished = Reached::default();
    let (start, finish) = (started.clone(), finished.clone());
    add_tool(
        &setup.registry,
        tool("work", move |_, _, _| {
            let (start, finish) = (start.clone(), finish.clone());
            async move {
                start.reach();
                finish.wait().await;
                text_result("v1")
            }
        }),
    );
    setup
        .faux
        .set_responses([calls(&[("work", json!({}), "c1")]), done()]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let submission = root
        .submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap();
    started.wait().await;
    // The same extension name replaces the old one in place.
    add_tool(
        &setup.registry,
        tool("work", |_, _, _| async { text_result("v2") }),
    );
    finished.reach();
    submission.wait(&context()).await.unwrap();
    assert_eq!(result_text(&results(&all_entries(&root).await)[0]), "v1");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn uses_a_section_and_hook_extension_reloaded_mid_run_from_the_runs_next_request() {
    let setup = chat_setup();
    let requests = Log::default();
    let prompt = |version: &'static str| {
        let sink = requests.clone();
        define_extension(ExtensionDefinition {
            sections: vec![section(
                "mode",
                move |_, _| async move { Ok(Some(version.to_string())) },
                None,
            )],
            hooks: vec![generation_hooks(GenerationHooks {
                before_request: Some(Arc::new(move |_, _, _| {
                    push(&sink, version);
                    Box::pin(async { Ok(None) })
                })),
                ..GenerationHooks::default()
            })],
            ..ExtensionDefinition::new("prompt")
        })
    };
    setup.registry.install(prompt("v1")).unwrap();
    let running = Reached::default();
    let reloaded = Reached::default();
    let (run_signal, reload_signal) = (running.clone(), reloaded.clone());
    add_tool(
        &setup.registry,
        tool("work", move |_, _, _| {
            let (run_signal, reload_signal) = (run_signal.clone(), reload_signal.clone());
            async move {
                run_signal.reach();
                reload_signal.wait().await;
                empty()
            }
        }),
    );
    setup
        .faux
        .set_responses([calls(&[("work", json!({}), "c1")]), done()]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let submission = root
        .submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap();
    running.wait().await;
    setup.registry.install(prompt("v2")).unwrap();
    reloaded.reach();
    submission.wait(&context()).await.unwrap();
    let sections: Vec<JsonValue> = system_json(&all_entries(&root).await)
        .into_iter()
        .filter_map(|system| system.get("sections").cloned())
        .collect();
    assert_eq!(
        sections,
        [
            json!({ "mode": "<mode>\nv1\n</mode>" }),
            json!({ "mode": "<mode>\nv2\n</mode>" })
        ]
    );
    assert_eq!(logged(&requests), ["v1", "v2"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_invocation_bound_waits_and_stops_watches_when_the_tools_invocation_ends() {
    let setup = chat_setup();
    let never = never_task("test.never");
    type Pending = tokio::task::JoinHandle<Result<TaskRecord>>;
    let wait: Arc<Mutex<Option<Pending>>> = Arc::default();
    let watch: Arc<Mutex<Option<crate::durable::session::DocumentWatch>>> = Arc::default();
    let (wait_slot, watch_slot) = (wait.clone(), watch.clone());
    add_tool(
        &setup.registry,
        tool("detach", move |_, api, ctx| {
            let (never, wait_slot, watch_slot) =
                (never.clone(), wait_slot.clone(), watch_slot.clone());
            async move {
                // The child's definition is not registered, so it stays pending.
                let child: TaskId<JsonValue> = api
                    .create_task(&never, json!({}), TaskOptions::conversation(None), &ctx)
                    .await?;
                let (waiter, wait_ctx) = (api.clone(), ctx.clone());
                *wait_slot.lock() = Some(tokio::spawn(async move {
                    waiter.wait_for_task(child, &wait_ctx).await
                }));
                tokio::task::yield_now().await;
                *watch_slot.lock() = api
                    .watch_doc(&*LIVE_DOC, api.conversation_id(), &ctx)
                    .await?;
                empty()
            }
        }),
    );
    let result = run(&setup, vec![calls(&[("detach", json!({}), "c1")]), done()]).await;
    let pending = wait.lock().take().unwrap();
    let error = pending.await.unwrap().expect_err("rejects");
    assert!(
        error.to_string().contains("invocation has ended"),
        "{error}"
    );
    let watch = watch.lock().take().unwrap();
    watch.closed().await;
    result.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_details_still_waiting_when_the_call_is_aborted_during_after_tool() {
    let setup = chat_setup();
    type Pending = tokio::task::JoinHandle<Result<()>>;
    let pending: Arc<Mutex<Option<Pending>>> = Arc::default();
    let slot = pending.clone();
    add_tool(
        &setup.registry,
        tool("slow", move |_, api, ctx| {
            let slot = slot.clone();
            async move {
                api.output("first\n")?;
                // The output commit is in flight, so these details wait for the next throttle window.
                let (api, ctx) = (api.clone(), ctx.clone());
                *slot.lock() = Some(tokio::spawn(async move {
                    api.details(json!({ "step": 1 }), &ctx).await
                }));
                tokio::task::yield_now().await;
                Ok(ToolExecutionResult::default())
            }
        }),
    );
    let in_after_tool = Reached::default();
    let reach = in_after_tool.clone();
    add_hooks(
        &setup.registry,
        after_tool(move |_, _, _, ctx| {
            reach.reach();
            async move {
                if let Some(signal) = ctx.abort_signal() {
                    signal.token().cancelled().await;
                }
                Ok(None)
            }
        }),
    );
    setup
        .faux
        .set_responses([calls(&[("slow", json!({}), "c1")]), done()]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let submission = root
        .submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap();
    in_after_tool.wait().await;
    let task_id = live_state(&harness, &root).await.unwrap().tools.unwrap()[0]
        .task_id
        .unwrap();
    harness.abort_task(task_id, &context()).await.unwrap();
    let details = pending.lock().take().unwrap();
    assert!(details.await.unwrap().is_err());
    submission.wait(&context()).await.unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn answers_an_aborted_tool_with_only_its_durable_output_discarding_buffered_output() {
    let setup = chat_setup();
    let buffered = Reached::default();
    let reach = buffered.clone();
    add_tool(
        &setup.registry,
        tool("slow", move |_, api, ctx| {
            let reach = reach.clone();
            async move {
                api.output("durable\n")?;
                // The first output commits at once; this one waits for the next throttle window.
                sleep_ms(20).await;
                api.output("buffered\n")?;
                reach.reach();
                if let Some(signal) = ctx.abort_signal() {
                    signal.token().cancelled().await;
                }
                Err(crate::durable::errors::Error::message("aborted"))
            }
        }),
    );
    setup
        .faux
        .set_responses([calls(&[("slow", json!({}), "c1")]), done()]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let submission = root
        .submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap();
    buffered.wait().await;
    let task_id = live_state(&harness, &root).await.unwrap().tools.unwrap()[0]
        .task_id
        .unwrap();
    harness.abort_task(task_id, &context()).await.unwrap();
    submission.wait(&context()).await.unwrap();
    assert_eq!(
        result_text(&results(&all_entries(&root).await)[0]),
        "durable\n|<harness>\n[error] Tool slow was aborted\n</harness>"
    );
    harness.close(&context()).await.unwrap();
}

// ─── Coding tools ───────────────────────────────────────────────────────────

/// A fresh directory and the fixed local environment rooted in it.
fn coding_env() -> (crate::durable::storage::test_support::TempDir, ChatOptions) {
    let dir = crate::durable::storage::test_support::TempDir::new("pi-durable-coding-");
    let cwd = dir
        .join("")
        .to_string_lossy()
        .trim_end_matches('/')
        .to_owned();
    let options = ChatOptions {
        env: Some(fixed_env(Arc::new(LocalExecutionEnv::at(cwd)))),
        ..ChatOptions::default()
    };
    (dir, options)
}

#[tokio::test]
async fn answers_a_failing_command_with_its_retained_tail_and_diagnostics_in_order() {
    let (_dir, options) = coding_env();
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        crate::durable::tools::create_bash_tool(Default::default()),
    );
    let command = "i=1; while [ $i -le 3000 ]; do echo line-$i; i=$((i + 1)); done; exit 7";
    let run = run_prepared(
        &setup,
        vec![
            calls(&[("bash", json!({ "command": command }), "b")]),
            done(),
        ],
        options,
        |_, _| async {},
    )
    .await;
    let entry = run
        .entries
        .iter()
        .find(|entry| entry.kind == "pi.tool-result")
        .unwrap();
    let result = &results(std::slice::from_ref(entry))[0];
    assert!(result.is_error);
    let text = result_text(result);
    assert!(text.starts_with("line-1001\n"), "{text}");
    let codes: Vec<JsonValue> = entry.data.as_ref().unwrap()["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|diagnostic| diagnostic["code"].clone())
        .collect();
    assert_eq!(
        codes,
        [
            json!("full_output"),
            json!("tool_error"),
            json!("truncated")
        ]
    );
    assert!(
        text.contains("line-3000\n|<harness>\n[info] Full output: "),
        "{text}"
    );
    assert!(
        text.contains(
            "\n[error] Command exited with code 7\n[warn] Output truncated to its end: 1000 lines, "
        ),
        "{text}"
    );
    run.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reads_edits_and_runs_a_command_in_one_run_then_answers() {
    let (dir, options) = coding_env();
    std::fs::write(dir.join("notes.txt"), "hello world\n").unwrap();
    let setup = chat_setup();
    setup
        .registry
        .install(define_extension(ExtensionDefinition {
            tools: vec![
                crate::durable::tools::create_read_tool(),
                crate::durable::tools::create_edit_tool(),
                crate::durable::tools::create_bash_tool(Default::default()),
            ],
            ..ExtensionDefinition::new("coding")
        }))
        .unwrap();
    let run = run_prepared(
        &setup,
        vec![
            calls(&[("read", json!({ "path": "notes.txt" }), "r")]),
            calls(&[(
                "edit",
                json!({ "path": "notes.txt", "edits": [{ "oldText": "world", "newText": "durable" }] }),
                "e",
            )]),
            calls(&[("bash", json!({ "command": "cat notes.txt" }), "b")]),
            done(),
        ],
        options,
        |_, _| async {},
    )
    .await;
    assert_eq!(run.status, SubmissionStatus::Done);
    let seen: Vec<(String, bool, String)> = results(&run.entries)
        .iter()
        .map(|result| {
            (
                result.tool_name.clone(),
                result.is_error,
                result_text(result),
            )
        })
        .collect();
    assert_eq!(
        seen,
        [
            ("read".into(), false, "hello world\n".into()),
            (
                "edit".into(),
                false,
                "Successfully replaced 1 block(s) in notes.txt.".into()
            ),
            ("bash".into(), false, "hello durable\n".into()),
        ]
    );
    assert_eq!(run.entries.last().unwrap().kind, "pi.assistant");
    assert_eq!(
        std::fs::read_to_string(dir.join("notes.txt")).unwrap(),
        "hello durable\n"
    );
    run.harness.close(&context()).await.unwrap();
}
