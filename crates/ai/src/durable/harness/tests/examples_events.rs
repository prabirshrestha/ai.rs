//! Ports of `test/examples/19`, `21`, `22`, and `25` as tests: each example runs against a Harness with the faux
//! model and asserts what the TS script prints. Example 19 uses memory storage and a stub `bash` tool (the read and
//! bash coding tools are M9), and covers both its `--events` and `--ops` modes. Example 21 joins at a gate after the
//! fifth output line instead of after 500 ms. In 25 the overflow turn first crosses the blocking threshold, so it
//! fails instead of showing an overflow summary. Example 23 (persistent background subagents) is not ported.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::compaction_support::is_summary_request;
use super::support::to_json;
use crate::chord::{BACKGROUND_CONTEXT, Context, ListenerOutcome};
use crate::durable::entries::ASSISTANT_ENTRY;
use crate::durable::harness::agent::configure;
use crate::durable::harness::events::{AgentEvent, AgentEventStream, EventBatch, watch_events};
use crate::durable::harness::live::LIVE_DOC;
use crate::durable::harness::types::{
    AgentChange, CompactionPolicyOverrides, ExtensionDefinition, ExtensionsChange, HarnessOptions,
    HarnessSettings, ModelRef, Replay, SubmissionDraft, ToolExecutionResult,
};
use crate::durable::harness::{
    Conversation, CreateOptions, Extension, Harness, create_registry, define_extension,
    define_tool, section,
};
use crate::durable::ids::ConversationId;
use crate::durable::session::tests::support::Deferred;
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{ConversationOwnership, ConversationQuery};
use crate::models::{Models, create_models};
use crate::providers::faux::{
    FauxMessageOptions, FauxModelDefinition, FauxResponseStep, FauxTokenSize,
    RegisterFauxProviderOptions, faux_assistant_message, faux_provider, faux_tool_call,
};
use crate::types::{AssistantContent, Message, StopReason, TextContent, UserContent};

fn faux_models(
    options: RegisterFauxProviderOptions,
    responses: Vec<FauxResponseStep>,
) -> (Models, crate::providers::faux::FauxProviderHandle) {
    let faux = faux_provider(options);
    let models = create_models(Default::default());
    models.set_provider(faux.provider.clone());
    faux.set_responses(responses);
    (models, faux)
}

fn tool_use(name: &str, args: JsonValue) -> FauxResponseStep {
    faux_assistant_message(
        vec![faux_tool_call(name, args, Some("call-1"))],
        FauxMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxMessageOptions::default()
        },
    )
    .into()
}

fn answer(text: &str) -> FauxResponseStep {
    faux_assistant_message(text, FauxMessageOptions::default()).into()
}

fn text_result(text: impl Into<String>) -> ToolExecutionResult {
    ToolExecutionResult {
        content: Some(vec![UserContent::Text(TextContent::new(text.into()))]),
        ..ToolExecutionResult::default()
    }
}

async fn open(
    models: Models,
    registry: &crate::durable::harness::Registry,
) -> (Harness, Conversation) {
    open_with(HarnessOptions::new(models, Arc::new(registry.clone()))).await
}

async fn open_with(options: HarnessOptions) -> (Harness, Conversation) {
    let ctx = BACKGROUND_CONTEXT.clone();
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &ctx)
        .await
        .unwrap();
    let root = harness
        .root(
            &ctx,
            CreateOptions {
                agent: Some(AgentChange::default().model(ModelRef::new("faux", "faux-1"))),
                init: None,
            },
        )
        .await
        .unwrap();
    (harness, root)
}

/// Text of an assistant message's text blocks.
fn assistant_text(message: &JsonValue) -> String {
    message["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|block| block["type"] == "text")
        .map(|block| block["text"].as_str().unwrap_or_default())
        .collect()
}

/// Record a stream's events as JSON lines.
fn record(stream: &AgentEventStream, lines: Arc<Mutex<Vec<JsonValue>>>) {
    stream
        .start(move |events: EventBatch, _| {
            lines.lock().extend(events.iter().map(to_json));
            Box::pin(async { Ok(()) })
        })
        .unwrap();
}

// ---- 19: JSON mode ----

async fn example_19(mode: &str) -> Vec<JsonValue> {
    let ctx = BACKGROUND_CONTEXT.clone();
    let (models, _faux) = faux_models(
        RegisterFauxProviderOptions {
            tokens_per_second: Some(200.0),
            ..RegisterFauxProviderOptions::default()
        },
        vec![
            tool_use("bash", json!({ "command": "ls" })),
            answer("This directory holds the durable package sources, tests, and docs."),
        ],
    );
    let registry = create_registry();
    registry
        .install(define_extension(ExtensionDefinition {
            tools: vec![define_tool(
                "bash",
                "Run a shell command",
                json!({ "type": "object", "properties": { "command": { "type": "string" } }, "required": ["command"] }),
                |_, _, _| async { Ok(text_result("docs\nsrc\ntest")) },
            )],
            sections: vec![section(
                "preamble",
                |_, _| async { Ok(Some("You are a concise coding assistant.".to_string())) },
                Some(false),
            )],
            ..ExtensionDefinition::new("coding")
        }))
        .unwrap();
    let (harness, root) = open(models, &registry).await;
    let lines: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    // Attach before submitting, so the stream covers the whole run.
    let stop: Box<dyn FnOnce() -> futures::future::BoxFuture<'static, ()>> = if mode == "events" {
        let stream = watch_events(&harness, root.id, &ctx).await.unwrap();
        lines
            .lock()
            .push(to_json(&AgentEvent::Snapshot(stream.snapshot.clone())));
        record(&stream, lines.clone());
        Box::new(move || {
            Box::pin(async move {
                stream.stop().await;
            })
        })
    } else {
        let watch = root.watch(&ctx).await.unwrap();
        lines
            .lock()
            .push(json!({ "view": watch.value().to_json() }));
        let sink = lines.clone();
        watch
            .start(move |_, ops, _| {
                sink.lock().push(json!({ "ops": to_json(&ops.to_vec()) }));
                Box::pin(async { Ok(()) })
            })
            .unwrap();
        Box::new(move || {
            Box::pin(async move {
                watch.stop().await;
            })
        })
    };
    root.submit(SubmissionDraft::input("What is in this directory?"), &ctx)
        .await
        .unwrap()
        .wait(&ctx)
        .await
        .unwrap();
    harness.wait_for_idle(&ctx).await.unwrap();
    // Let the last batch reach the listener before stopping.
    tokio::task::yield_now().await;
    stop().await;
    harness.close(&ctx).await.unwrap();
    lines.lock().clone()
}

#[tokio::test]
async fn example_19_json_events() {
    let lines = example_19("events").await;
    assert_eq!(lines[0]["type"], "snapshot");
    let types: Vec<&str> = lines
        .iter()
        .map(|line| line["type"].as_str().unwrap())
        .filter(|kind| *kind != "message_update")
        .collect();
    let position = |kind: &str, from: usize| {
        from + types[from..]
            .iter()
            .position(|candidate| *candidate == kind)
            .unwrap_or_else(|| panic!("no {kind} after {from} in {types:?}"))
    };
    // One run: the user message, the tool-calling answer, the tool round, and the final answer, in order.
    let start = position("run_start", 0);
    let tool = position("tool_execution_start", start);
    let tool_end = position("tool_execution_end", tool);
    let end = position("run_end", tool_end);
    assert!(position("message_end", start) < tool);
    assert!(position("message_end", tool_end) < end);
    let tool_start = lines
        .iter()
        .find(|line| line["type"] == "tool_execution_start")
        .unwrap();
    assert_eq!(tool_start["toolName"], "bash");
    assert_eq!(tool_start["args"], json!({ "command": "ls" }));
    let answers: Vec<String> = lines
        .iter()
        .filter(|line| line["type"] == "message_end" && line["entry"]["kind"] == "pi.assistant")
        .map(|line| assistant_text(&line["entry"]["model"][0]))
        .collect();
    assert_eq!(
        answers,
        [
            "",
            "This directory holds the durable package sources, tests, and docs."
        ]
    );
}

#[tokio::test]
async fn example_19_json_ops() {
    let lines = example_19("ops").await;
    let view = &lines[0]["view"];
    assert_eq!(view["entries"], json!([]));
    assert!(lines[1..].iter().all(|line| line["ops"].is_array()));
    // Applying every frame's ops to the first view gives the final transcript.
    let mut value = view.clone();
    for line in &lines[1..] {
        let ops: Vec<crate::chord::delta::Op> =
            serde_json::from_value(line["ops"].clone()).unwrap();
        value = crate::chord::delta::apply_immutable(&value, &ops).unwrap();
    }
    let kinds: Vec<&str> = value["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["kind"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "pi.user",
            "pi.system",
            "pi.assistant",
            "pi.tool-result",
            "pi.assistant"
        ]
    );
}

// ---- 21: late join ----

#[tokio::test]
async fn example_21_late_join() {
    let ctx = BACKGROUND_CONTEXT.clone();
    let (half, joined) = (Deferred::default(), Deferred::default());
    let registry = create_registry();
    {
        let (half, joined) = (half.clone(), joined.clone());
        registry
            .install(define_extension(ExtensionDefinition {
                tools: vec![define_tool(
                    "count",
                    "Counts to ten",
                    json!({ "type": "object", "properties": {} }),
                    move |_, api, _| {
                        let (half, joined) = (half.clone(), joined.clone());
                        async move {
                            for n in 1..=10 {
                                api.output(&format!("{n}\n"))?;
                                if n == 5 {
                                    half.resolve();
                                    joined.wait().await;
                                }
                                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            }
                            Ok(ToolExecutionResult::default())
                        }
                    },
                )],
                ..ExtensionDefinition::new("count")
            }))
            .unwrap();
    }
    let (models, _faux) = faux_models(
        RegisterFauxProviderOptions {
            tokens_per_second: Some(40.0),
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            ..RegisterFauxProviderOptions::default()
        },
        vec![
            tool_use("count", json!({})),
            answer("Counted to ten, and this answer streams slowly."),
        ],
    );
    let (harness, root) = open(models, &registry).await;
    let submission = root
        .submit(SubmissionDraft::input("Count to ten, then tell me."), &ctx)
        .await
        .unwrap();
    // Join while the tool is halfway through.
    half.wait().await;

    // Structural client: the view holds the committed transcript and documents, including the running tool's output.
    let view = root.view_state(&ctx).await.unwrap();
    let value = view.value();
    let kinds: Vec<&str> = value
        .entries
        .iter()
        .map(|entry| entry.kind.as_str())
        .collect();
    assert_eq!(kinds, ["pi.user", "pi.system", "pi.assistant"]);
    let slot = value.doc("pi.live").unwrap()["tools"][0].clone();
    assert_eq!(slot["status"], "running");
    // Output is published at commits; whatever is committed so far is a prefix of the five lines.
    let committed = slot["output"].as_str().unwrap_or_default().to_string();
    assert!("1\n2\n3\n4\n5\n".starts_with(&committed), "{committed:?}");
    let view_outputs: Arc<Mutex<Vec<String>>> = Arc::default();
    let unsubscribe = {
        let view_outputs = view_outputs.clone();
        view.subscribe(move |value, _, _| {
            if let Some(slot) = value.doc("pi.live").map(|live| live["tools"][0].clone())
                && slot["status"] == "running"
            {
                view_outputs
                    .lock()
                    .push(slot["output"].as_str().unwrap_or_default().to_string());
            }
            ListenerOutcome::ok()
        })
    };

    // Event client: the snapshot event carries the same state; later events apply on top of it.
    let stream = watch_events(&harness, root.id, &ctx).await.unwrap();
    let snapshot = to_json(&stream.snapshot);
    let tools: Vec<String> = snapshot["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|slot| {
            format!(
                "{} {}",
                slot["name"].as_str().unwrap(),
                slot["status"].as_str().unwrap()
            )
        })
        .collect();
    assert_eq!(tools, ["count running"]);
    let output = Arc::new(Mutex::new(
        snapshot["tools"][0]["output"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    ));
    let printed: Arc<Mutex<Vec<String>>> = Arc::default();
    {
        let (output, printed) = (output.clone(), printed.clone());
        stream
            .start(move |events: EventBatch, _| {
                for event in events.iter().map(to_json) {
                    let kind = event["type"].as_str().unwrap().to_string();
                    if kind == "tool_execution_update" && !event["output"].is_null() {
                        let update = &event["output"];
                        let mut output = output.lock();
                        *output = match update["set"].as_str() {
                            Some(set) => set.to_string(),
                            None => {
                                let trim = update["trimStart"].as_u64().unwrap_or(0) as usize;
                                format!(
                                    "{}{}",
                                    &output[trim..],
                                    update["append"].as_str().unwrap_or_default()
                                )
                            }
                        };
                        printed.lock().push(format!("event output now: {output:?}"));
                    } else if kind == "message_update" {
                        let deltas: String = event["changes"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|change| change["type"] == "text_delta")
                            .map(|change| change["delta"].as_str().unwrap())
                            .collect();
                        printed.lock().push(format!("event text delta: {deltas}"));
                    } else {
                        printed.lock().push(format!("event: {kind}"));
                    }
                }
                Box::pin(async { Ok(()) })
            })
            .unwrap();
    }
    joined.resolve();

    submission.wait(&ctx).await.unwrap();
    harness.wait_for_idle(&ctx).await.unwrap();
    tokio::task::yield_now().await;
    stream.stop().await;
    unsubscribe.unsubscribe();
    view.dispose().unwrap();
    harness.close(&ctx).await.unwrap();

    // The event client rebuilt the whole output from the snapshot and the later deltas.
    let printed = printed.lock().clone();
    assert!(
        printed.contains(&format!(
            "event output now: {:?}",
            "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n"
        )),
        "{printed:#?}"
    );
    // The answer streamed as text deltas within its text; the first partial carries its start in `text_start`.
    let streamed: String = printed
        .iter()
        .filter_map(|line| line.strip_prefix("event text delta: "))
        .collect();
    assert!(!streamed.is_empty(), "{printed:#?}");
    assert!(
        "Counted to ten, and this answer streams slowly.".contains(&streamed),
        "{streamed:?} {printed:#?}"
    );
    let events: Vec<&String> = printed
        .iter()
        .filter(|line| line.starts_with("event: "))
        .collect();
    assert!(events.iter().any(|line| line.as_str() == "event: run_end"));
    assert!(
        events
            .iter()
            .any(|line| line.as_str() == "event: tool_execution_end")
    );
    // The structural client saw the running output grow to all ten lines.
    let view_outputs = view_outputs.lock().clone();
    assert_eq!(
        view_outputs.last().map(String::as_str),
        Some("1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
    );
}

// ---- 22: foreground subagent ----

/// Text of the assistant entry `answer`.
async fn answer_text(
    api: &crate::durable::harness::tool::ToolExecutionApi,
    answer: crate::durable::ids::EntryId,
    context: &Context,
) -> crate::durable::errors::Result<String> {
    let entry = api
        .commit(
            move |tx| async move { tx.entry_of(&ASSISTANT_ENTRY, answer).await },
            context,
        )
        .await?;
    let model = entry.and_then(|entry| entry.model).unwrap_or_default();
    Ok(match model.first() {
        Some(Message::Assistant(message)) => message
            .content
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect(),
        _ => String::new(),
    })
}

fn subagent_extension() -> Extension {
    let this: Arc<std::sync::OnceLock<Extension>> = Arc::default();
    let own = this.clone();
    let mut tool = define_tool(
        "subagent",
        "Delegate a self-contained task to a subagent and get its answer back.",
        json!({
            "type": "object",
            "properties": { "task": { "type": "string", "description": "What the subagent should do" } },
            "required": ["task"],
        }),
        move |args, api, call_context| {
            let own = own.clone();
            async move {
                let task = args["task"].as_str().unwrap_or_default().to_string();
                let task_id = api.task_id();
                // The child is owned by this tool call's task, so aborting the call aborts the child.
                let child = api
                    .commit(
                        move |tx| async move {
                            // Ownership records the child: a rerun of this call finds it instead of creating another.
                            let existing = tx
                                .scan_conversations(
                                    ConversationQuery {
                                        owner_conversation_id: None,
                                        owner_task_id: Some(task_id),
                                    },
                                    1,
                                    None,
                                )
                                .await?;
                            if let Some(existing) = existing.items.first() {
                                return Ok(existing.id);
                            }
                            // Starts as a copy of this conversation's agent.
                            let created = tx
                                .create_conversation(ConversationOwnership::Task { task_id })
                                .await?;
                            // Without this extension, the child is not offered this tool.
                            let change = AgentChange {
                                extensions: Some(Some(ExtensionsChange::Edit {
                                    add: None,
                                    remove: Some(vec![own.get().unwrap().clone()]),
                                })),
                                ..AgentChange::default()
                            };
                            configure(&tx, created.id, &change).await?;
                            Ok(created.id)
                        },
                        &call_context,
                    )
                    .await?;
                // A UI watching the parent sees this and can attach to the child.
                api.details(json!({ "conversationId": child }), &call_context)
                    .await?;
                let handle = api.conversation(child, &call_context).await?.unwrap();
                // The request ID makes a rerun get back the submission it made before the crash.
                let request =
                    SubmissionDraft::input(task).with_request_id(format!("subagent:{task_id}"));
                let settled = handle
                    .submit(request, &call_context)
                    .await?
                    .wait(&call_context)
                    .await?;
                let Some(answer) = settled
                    .answer
                    .filter(|_| to_json(&settled)["status"] == "done")
                else {
                    return Err(crate::durable::errors::Error::message(format!(
                        "Subagent failed: {}",
                        to_json(&settled)["status"]
                    )));
                };
                let text = answer_text(&api, answer, &call_context).await?;
                Ok(ToolExecutionResult {
                    details: Some(json!({ "conversationId": child })),
                    ..text_result(text)
                })
            }
        },
    );
    // Safe to rerun after a crash: a rerun finds the child it already created and the submission it already made.
    tool.replay = Some(Replay::Safe);
    let extension = define_extension(ExtensionDefinition {
        tools: vec![tool],
        ..ExtensionDefinition::new("subagent")
    });
    this.set(extension.clone()).ok().unwrap();
    extension
}

type Printed = Arc<Mutex<Vec<String>>>;

struct Attacher {
    harness: Harness,
    /// Resolved once a child conversation's stream is attached.
    child_attached: Deferred,
    attached: Mutex<HashSet<ConversationId>>,
    streams: Mutex<Vec<AgentEventStream>>,
    printed: Printed,
}

impl Attacher {
    fn attach(
        self: Arc<Self>,
        id: ConversationId,
        indent: String,
    ) -> futures::future::BoxFuture<'static, ()> {
        Box::pin(async move {
            self.attached.lock().insert(id);
            let listener_indent = indent.clone();
            let stream = watch_events(&self.harness, id, &BACKGROUND_CONTEXT)
                .await
                .unwrap();
            let this = self.clone();
            stream
                .start(move |events: EventBatch, _| {
                    let this = this.clone();
                    let indent = listener_indent.clone();
                    Box::pin(async move {
                        for event in events.iter().map(to_json) {
                            this.print(&indent, &event);
                            if event["type"] != "tool_execution_update" {
                                continue;
                            }
                            let Some(child) = serde_json::from_value::<ConversationId>(
                                event["details"]["conversationId"].clone(),
                            )
                            .ok() else {
                                continue;
                            };
                            if !this.attached.lock().contains(&child) {
                                this.clone().attach(child, format!("{indent}  ")).await;
                            }
                        }
                        Ok(())
                    })
                })
                .unwrap();
            self.streams.lock().push(stream);
            if !indent.is_empty() {
                self.child_attached.resolve();
            }
        })
    }

    fn print(&self, indent: &str, event: &JsonValue) {
        if event["type"] == "message_end" && event["entry"]["kind"] == "pi.assistant" {
            let text = assistant_text(&event["entry"]["model"][0]);
            if !text.is_empty() {
                self.printed
                    .lock()
                    .push(format!("{indent}assistant: {text}"));
            }
        } else if event["type"] == "tool_execution_start" {
            self.printed.lock().push(format!(
                "{indent}tool {}({})",
                event["toolName"].as_str().unwrap(),
                event["args"]
            ));
        }
    }
}

#[tokio::test]
async fn example_22_subagent_foreground() {
    let ctx = BACKGROUND_CONTEXT.clone();
    // The parent delegates, the child answers, and the parent reports. The child answers once the UI attached to
    // it (TS: the faux stream's timers give the UI that time).
    let child_attached = Deferred::default();
    let child_answer = {
        let child_attached = child_attached.clone();
        FauxResponseStep::async_factory(move |_, _, _, _| {
            let child_attached = child_attached.clone();
            async move {
                child_attached.wait().await;
                Ok(faux_assistant_message(
                    "2, 3, and 5.",
                    FauxMessageOptions::default(),
                ))
            }
        })
    };
    let (models, _faux) = faux_models(
        RegisterFauxProviderOptions::default(),
        vec![
            tool_use("subagent", json!({ "task": "Name three prime numbers." })),
            child_answer,
            answer("The subagent says: 2, 3, and 5."),
        ],
    );
    let registry = create_registry();
    registry.install(subagent_extension()).unwrap();
    let (harness, root) = open(models, &registry).await;
    let attacher = Arc::new(Attacher {
        harness: harness.clone(),
        child_attached,
        attached: Mutex::default(),
        streams: Mutex::default(),
        printed: Printed::default(),
    });
    attacher.clone().attach(root.id, String::new()).await;
    root.submit(
        SubmissionDraft::input(
            "Use the subagent tool to find three prime numbers, then tell me what it said.",
        ),
        &ctx,
    )
    .await
    .unwrap()
    .wait(&ctx)
    .await
    .unwrap();
    harness.wait_for_idle(&ctx).await.unwrap();
    tokio::task::yield_now().await;
    let streams: Vec<AgentEventStream> = attacher.streams.lock().drain(..).collect();
    for stream in streams {
        stream.stop().await;
    }
    harness.close(&ctx).await.unwrap();
    assert_eq!(
        *attacher.printed.lock(),
        [
            r#"tool subagent({"task":"Name three prime numbers."})"#,
            "  assistant: 2, 3, and 5.",
            "assistant: The subagent says: 2, 3, and 5.",
        ]
    );
}

// ---- 25: compaction ----

struct Trip {
    overflow_once: Mutex<bool>,
    summaries: Mutex<usize>,
    hold: Mutex<Option<Deferred>>,
}

fn message_text(message: &Message) -> String {
    let value = to_json(message);
    match &value["content"] {
        JsonValue::String(text) => text.clone(),
        JsonValue::Array(blocks) => blocks
            .iter()
            .find(|block| block["type"] == "text")
            .and_then(|block| block["text"].as_str())
            .unwrap_or_default()
            .replace('\n', " "),
        _ => String::new(),
    }
}

/// The model context: a header line, the compaction reason, and one line per message.
async fn show(conversation: &Conversation, label: &str) -> Vec<String> {
    let ctx = BACKGROUND_CONTEXT.clone();
    let view = conversation.context(&ctx).await.unwrap();
    let stored = conversation
        .entries(None, None, 1000, None, &ctx)
        .await
        .unwrap()
        .items
        .len();
    let mut lines = vec![format!(
        "{label}: {} messages in context, {stored} entries stored",
        view.messages.len()
    )];
    if let Some(head) = view
        .head
        .as_ref()
        .filter(|head| head.kind == "pi.compaction")
    {
        let reason = to_json(head)["data"]["reason"]
            .as_str()
            .unwrap()
            .to_string();
        lines.push(format!("  ({reason} compaction summary first)"));
    }
    for message in &view.messages {
        let role = to_json(message)["role"].as_str().unwrap().to_string();
        let text = if role == "system" {
            "(system prompt)".to_string()
        } else {
            message_text(message).chars().take(70).collect()
        };
        lines.push(format!("  {role:<9} {text}"));
    }
    lines
}

#[tokio::test]
async fn example_25_compaction() {
    let ctx = BACKGROUND_CONTEXT.clone();
    let trip = Arc::new(Trip {
        overflow_once: Mutex::new(false),
        summaries: Mutex::new(0),
        hold: Mutex::new(None),
    });
    // A fake model with a tiny 3000-token window. It answers chat messages, writes summaries when asked to
    // summarize, and once rejects a request as too long.
    let respond = {
        let trip = trip.clone();
        move |transcript: crate::types::TranscriptContext, _, _, _| {
            let trip = trip.clone();
            async move {
                if is_summary_request(&transcript.messages) {
                    let mut summaries = trip.summaries.lock();
                    *summaries += 1;
                    return Ok(faux_assistant_message(
                        format!("## Goal\nPlan a week in Lisbon (summary #{summaries}).").as_str(),
                        FauxMessageOptions::default(),
                    ));
                }
                if std::mem::take(&mut *trip.overflow_once.lock()) {
                    return Ok(faux_assistant_message(
                        "",
                        FauxMessageOptions {
                            stop_reason: Some(StopReason::Error),
                            error_message: Some("prompt is too long".into()),
                            ..FauxMessageOptions::default()
                        },
                    ));
                }
                let held = trip.hold.lock().take();
                if let Some(held) = held {
                    held.wait().await;
                }
                let question = transcript
                    .messages
                    .iter()
                    .rev()
                    .find(|message| matches!(message, Message::User(_)))
                    .map(message_text)
                    .unwrap_or_default();
                Ok(faux_assistant_message(
                    format!(
                        "A detailed answer to \"{question}\": {}",
                        "details ".repeat(200)
                    )
                    .as_str(),
                    FauxMessageOptions::default(),
                ))
            }
        }
    };
    let mut tiny = FauxModelDefinition::new("tiny");
    tiny.context_window = Some(3000);
    tiny.max_tokens = Some(1000);
    let (models, _faux) = faux_models(
        RegisterFauxProviderOptions {
            models: vec![tiny],
            ..RegisterFauxProviderOptions::default()
        },
        (0..100)
            .map(|_| FauxResponseStep::async_factory(respond.clone()))
            .collect(),
    );
    // Generation blocks to compact above 3000 - 1000 = 2000 tokens and starts a background compaction above
    // 2000 - 800. Settings are read at every use, so `background_tokens` is live.
    let background_tokens = Arc::new(Mutex::new(800u64));
    let mut options = HarnessOptions::new(models, Arc::new(create_registry()));
    {
        let background_tokens = background_tokens.clone();
        options.settings = Some(Arc::new(move || HarnessSettings {
            compaction: Some(CompactionPolicyOverrides {
                enabled: None,
                reserve_tokens: Some(1000),
                keep_recent_tokens: Some(400),
                background_tokens: Some(*background_tokens.lock()),
            }),
            ..HarnessSettings::default()
        }));
    }
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &ctx)
        .await
        .unwrap();
    let root = harness
        .root(
            &ctx,
            CreateOptions {
                agent: Some(AgentChange::default().model(ModelRef::new("faux", "tiny"))),
                init: None,
            },
        )
        .await
        .unwrap();
    let shown: Arc<Mutex<Vec<Vec<String>>>> = Arc::default();
    let ask = |question: &'static str| {
        let (harness, root, shown, ctx) =
            (harness.clone(), root.clone(), shown.clone(), ctx.clone());
        async move {
            let record = to_json(
                &root
                    .submit(SubmissionDraft::input(question), &ctx)
                    .await
                    .unwrap()
                    .wait(&ctx)
                    .await
                    .unwrap(),
            );
            // Let a background compaction started by this turn finish, so its summary shows below.
            let running = harness
                .snapshot(&*LIVE_DOC, root.id, &ctx)
                .await
                .unwrap()
                .and_then(|live| live.compactions)
                .unwrap_or_default();
            for status in running {
                harness.wait_for_task(status.task_id, &ctx).await.unwrap();
            }
            let outcome = if record["status"] == "done" {
                "answered".to_string()
            } else {
                record["reason"].as_str().unwrap().to_string()
            };
            let lines = show(&root, &format!("after \"{question}\" ({outcome})")).await;
            shown.lock().push(lines);
        }
    };

    // 1. A long chat: once the context crosses the background threshold, a compaction runs while the chat goes on.
    for question in [
        "Where should we stay?",
        "What should we eat?",
        "Which day trips?",
        "Any museums?",
        "Nightlife?",
    ] {
        ask(question).await;
    }
    {
        let shown = shown.lock();
        assert!(shown.iter().all(|lines| lines[0].contains("(answered)")));
        // The summaries keep the context small while the stored transcript grows.
        assert!(
            shown
                .iter()
                .any(|lines| lines[1] == "  (threshold compaction summary first)"),
            "{shown:#?}"
        );
    }
    assert!(*trip.summaries.lock() >= 1);

    // 2. A manual compaction while an answer is still being written: the summary waits in the inbox and is placed
    // right after the answer.
    let finish = Deferred::default();
    *trip.hold.lock() = Some(finish.clone());
    let busy = root
        .submit(SubmissionDraft::input("How do we get around?"), &ctx)
        .await
        .unwrap();
    let manual = root
        .compact(Some("Keep the hotel shortlist".into()), &ctx)
        .await
        .unwrap();
    let record = harness.wait_for_task(manual, &ctx).await.unwrap();
    let placement = to_json(&record.state)["outcome"]["result"]["submissionId"].clone();
    let summary = harness
        .submission(serde_json::from_value(placement).unwrap(), &ctx)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        to_json(&summary.status(&ctx).await.unwrap())["status"],
        "queued"
    );
    finish.resolve();
    busy.wait(&ctx).await.unwrap();
    assert_eq!(
        to_json(&summary.wait(&ctx).await.unwrap())["status"],
        "done"
    );
    let after = show(&root, "after compact()").await;
    // A background compaction the busy turn started may land after the manual summary (TS shows the manual one).
    assert!(
        after[1].ends_with(" compaction summary first)"),
        "{after:#?}"
    );
    assert!(
        after[2]
            .starts_with("  user      The conversation history before this point was compacted")
    );
    let reasons: Vec<String> = root
        .entries(None, None, 1000, None, &ctx)
        .await
        .unwrap()
        .items
        .iter()
        .filter(|entry| entry.kind == "pi.compaction")
        .map(|entry| {
            to_json(entry)["data"]["reason"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert!(
        reasons.iter().any(|reason| reason == "manual"),
        "{reasons:?}"
    );

    // 3. The provider rejects a request as too long. Divergence: TS shows the overflow summary first; here the turn
    // already crosses the blocking threshold, compacts before its request, and an overflow after a blocking
    // compaction in the same generation fails (as in the compaction suite).
    *background_tokens.lock() = 0;
    ask("What should we pack?").await;
    *trip.overflow_once.lock() = true;
    ask("Summarize the plan for my partner").await;
    let last = shown.lock().last().unwrap().clone();
    assert_eq!(
        last[0].split(':').next().unwrap(),
        "after \"Summarize the plan for my partner\" (model_error)"
    );
    assert_eq!(last[1], "  (threshold compaction summary first)");
    // The next request fits again and is answered.
    ask("Thanks!").await;
    let last = shown.lock().last().unwrap().clone();
    assert!(
        last[0].starts_with("after \"Thanks!\" (answered)"),
        "{last:#?}"
    );
    harness.close(&ctx).await.unwrap();
}
