//! Port of `test/examples/23-subagent-background.ts` as a test: persistent background subagents behind one
//! `subagent` tool, with a restart while a subagent works. The main conversation's transcript, as the TS `follow()`
//! prints it, is rebuilt from its entries at the end and checked line by line where the order is fixed.
//!
//! Divergences from the TS script: the scripted faux model always runs (no `OPENAI_API_KEY` branch); the SQLite
//! file reopened by the restart is `ControlledStorage::persistent()`; colors are not printed.

use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::text_of;
use crate::chord::{BACKGROUND_CONTEXT, Context};
use crate::durable::documents::{DocToken, define_doc};
use crate::durable::entries::ASSISTANT_ENTRY;
use crate::durable::errors::Result;
use crate::durable::harness::agent::configure;
use crate::durable::harness::live::LIVE_DOC;
use crate::durable::harness::types::{
    AgentChange, Extension, ExtensionDefinition, ExtensionsChange, HarnessOptions, ModelRef,
    Replay, SubmissionDraft, ToolExecutionResult, WhenBusy,
};
use crate::durable::harness::{
    Conversation, CreateOptions, Harness, Registry, create_registry, define_extension, define_tool,
};
use crate::durable::ids::{ConversationId, EntryId, TaskId};
use crate::durable::session::tests::support::ControlledStorage;
use crate::durable::tasks::{NextTaskState, Task, TaskDefinition, define_task};
use crate::durable::types::{
    ConversationOwnership, LatestConversation, LatestFork, Storage, SubmissionStatus,
    SubmissionType, TaskOptions, TaskOutcome, TaskOwnership,
};
use crate::models::{Models, create_models};
use crate::providers::faux::{
    FauxMessageOptions, FauxResponseStep, RegisterFauxProviderOptions, faux_assistant_message,
    faux_provider, faux_tool_call,
};
use crate::types::{Message, StopReason, UserContent, UserMessageContent};

// ─── Product code ───

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Subagent {
    conversation_id: ConversationId,
    /// Answers already reported to the main agent: several messages can end in one answer, reported once.
    reported: Vec<EntryId>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
struct Subagents {
    agents: IndexMap<String, Subagent>,
    reporters: IndexMap<String, TaskId>,
}

/// A fork of the main conversation starts without subagents.
static SUBAGENTS_DOC: LazyLock<DocToken<Subagents, LatestConversation>> = LazyLock::new(|| {
    define_doc(crate::durable::types::DocDefinition::new(
        "app.subagents",
        1,
        LatestConversation {
            fork: LatestFork::Initial,
        },
        Subagents::default,
    ))
    .unwrap()
});

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
enum AnchorState {
    Done,
}

/// Owns a subagent's conversation so it outlives the main agent's turns: a background task that finishes at once.
static ANCHOR: LazyLock<Task<(), AnchorState, ()>> = LazyLock::new(|| {
    define_task(
        TaskDefinition::<(), AnchorState, ()>::new("app.subagent-anchor", 1, |_| AnchorState::Done)
            .phase("done", |_, runtime, ctx| async move {
                runtime
                    .commit(
                        |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                        &ctx,
                    )
                    .await
            })
            .abort(|_, runtime, ctx| async move { aborted(runtime, ctx).await }),
    )
});

async fn aborted<I, S, R, H>(
    runtime: crate::durable::harness::TaskRuntime<I, S, R, H>,
    ctx: Context,
) -> Result<()>
where
    I: serde::de::DeserializeOwned + Send + 'static,
    S: Serialize + serde::de::DeserializeOwned + Send + 'static,
    R: 'static,
{
    runtime
        .commit(
            |_, _| async {
                Ok(Some(NextTaskState::Terminal {
                    outcome: TaskOutcome::Aborted {
                        reason: None,
                        result: None,
                    },
                }))
            },
            &ctx,
        )
        .await
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReporterInput {
    name: String,
    conversation_id: ConversationId,
    message: String,
    follow_up: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
enum ReporterState {
    Deliver,
    Report {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        report: Option<String>,
    },
}

fn text_of_entry(message: Option<&Message>) -> String {
    match message {
        Some(Message::Assistant(assistant)) => assistant
            .content
            .iter()
            .filter_map(|content| match content {
                crate::types::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect(),
        _ => String::new(),
    }
}

/// Delivers one message to a subagent and reports the answer to the main agent; a background task too.
static REPORTER: LazyLock<Task<ReporterInput, ReporterState, ()>> = LazyLock::new(|| {
    define_task(
        TaskDefinition::<ReporterInput, ReporterState, ()>::new("app.subagent-reporter", 1, |_| {
            ReporterState::Deliver
        })
        // Send the message, wait for its answer, and decide what to report.
        .phase("deliver", |reporter, runtime, ctx| async move {
            let ReporterInput {
                name,
                conversation_id,
                message,
                follow_up,
            } = reporter.input.clone();
            // While the subagent is busy, a steer reaches it at its next step and a follow-up after its current answer.
            let subagent = runtime.conversation(conversation_id, &ctx).await?.unwrap();
            let request = SubmissionDraft::input(message)
                .when_busy(if follow_up {
                    WhenBusy::FollowUp
                } else {
                    WhenBusy::Steer
                })
                .with_request_id(format!("subagent:{}", reporter.id));
            let settled = subagent.submit(request, &ctx).await?.wait(&ctx).await?;
            // One commit decides the report and records the answer as delivered, so a restart does not decide again.
            let main = runtime.conversation_id();
            runtime
                .commit(
                    move |tx, _| async move {
                        let next = |report: Option<String>| {
                            Ok(Some(NextTaskState::running(ReporterState::Report {
                                report,
                            })))
                        };
                        // `aborted`: stopped, or withdrawn while queued. Nothing to report.
                        if settled.status == SubmissionStatus::Unanswered {
                            let reason = settled.reason.clone().unwrap_or_default();
                            return next(
                                (reason != "aborted")
                                    .then(|| format!("[subagent {name} failed: {reason}]")),
                            );
                        }
                        let Some(answer) = settled
                            .answer
                            .filter(|_| settled.type_ == SubmissionType::Input)
                        else {
                            return next(None);
                        };
                        let doc = tx.doc(&*SUBAGENTS_DOC, main).await?;
                        let fresh = doc.edit(|state| {
                            let agent = state.agents.get_mut(&name).unwrap();
                            if agent.reported.contains(&answer) {
                                return false;
                            }
                            agent.reported.push(answer);
                            true
                        })?;
                        if !fresh {
                            return next(None);
                        }
                        let entry = tx.entry_of(&ASSISTANT_ENTRY, answer).await?;
                        let text = text_of_entry(
                            entry
                                .as_ref()
                                .and_then(|entry| entry.model.as_ref())
                                .and_then(|model| model.first()),
                        );
                        next(Some(format!(
                            "[subagent {name} answered, no reply needed] {text}"
                        )))
                    },
                    &ctx,
                )
                .await
        })
        // Post the report as a follow-up input: it starts a turn when the main agent is idle, or waits for its
        // current answer.
        .phase("report", |reporter, runtime, ctx| async move {
            if let ReporterState::Report {
                report: Some(report),
            } = reporter.checkpoint.clone()
            {
                let main = runtime
                    .conversation(runtime.conversation_id(), &ctx)
                    .await?
                    .unwrap();
                let input = SubmissionDraft::input(report)
                    .when_busy(WhenBusy::FollowUp)
                    .with_request_id(format!("subagent-report:{}", reporter.id));
                main.submit(input, &ctx).await?;
            }
            runtime
                .commit(
                    |_, _| async { Ok(Some(NextTaskState::completed(()))) },
                    &ctx,
                )
                .await
        })
        .abort(|_, runtime, ctx| async move { aborted(runtime, ctx).await }),
    )
});

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubagentArgs {
    action: String,
    name: Option<String>,
    message: Option<String>,
    follow_up: Option<bool>,
}

fn reply(
    text: String,
    name: Option<&str>,
    conversation_id: Option<ConversationId>,
) -> ToolExecutionResult {
    ToolExecutionResult {
        // A UI can attach to the subagent's conversation through the call's details.
        details: match (name, conversation_id) {
            (Some(name), Some(id)) => Some(json!({ "name": name, "conversationId": id })),
            _ => None,
        },
        ..ToolExecutionResult::text(text)
    }
}

/// The task definitions come with the extension, so pending reporters resume after a restart once the host installs
/// it again.
fn subagent_tools() -> Extension {
    static TOOLS: OnceLock<Extension> = OnceLock::new();
    TOOLS
        .get_or_init(|| {
            let mut tool = define_tool(
                "subagent",
                "Manage persistent subagents that work in the background. Actions: spawn (name, message), send (name, message; followUp: true queues it after the current answer instead of steering), stop (name: aborts its current work), status (name, or all subagents without one). Answers are reported back to you when they arrive.",
                json!({
                    "type": "object",
                    "properties": {
                        "action": { "anyOf": [
                            { "const": "spawn", "type": "string" },
                            { "const": "send", "type": "string" },
                            { "const": "stop", "type": "string" },
                            { "const": "status", "type": "string" },
                        ] },
                        "name": { "type": "string" },
                        "message": { "type": "string" },
                        "followUp": { "type": "boolean" },
                    },
                    "required": ["action"],
                }),
                |args, api, call_context| async move {
                    let SubagentArgs {
                        action,
                        name,
                        message,
                        follow_up,
                    } = serde_json::from_value(args).unwrap();
                    let main = api.conversation_id();
                    let registry = api
                        .snapshot(&*SUBAGENTS_DOC, main, &call_context)
                        .await?
                        .unwrap_or_default();

                    if action == "status" {
                        let names: Vec<String> = match &name {
                            None => registry.agents.keys().cloned().collect(),
                            Some(name) => vec![name.clone()],
                        };
                        let mut lines = Vec::new();
                        for each in names {
                            let Some(found) = registry.agents.get(&each) else {
                                continue;
                            };
                            // A conversation is busy while it has a run: from an input until its final answer.
                            let busy = api
                                .snapshot(&*LIVE_DOC, found.conversation_id, &call_context)
                                .await?
                                .is_some_and(|live| live.run.is_some());
                            lines.push(format!("{each}: {}", if busy { "working" } else { "idle" }));
                        }
                        return Ok(reply(
                            if lines.is_empty() {
                                "No subagents.".into()
                            } else {
                                lines.join("\n")
                            },
                            None,
                            None,
                        ));
                    }
                    let Some(name) = name else {
                        return Ok(reply(format!("{action} needs a name."), None, None));
                    };
                    let agent = registry.agents.get(&name).cloned();
                    if action != "spawn" && agent.is_none() {
                        return Ok(reply(format!("No subagent named {name}."), None, None));
                    }

                    if action == "stop" {
                        // Aborts the subagent's current answer and tools and drops its queued messages.
                        let agent = agent.unwrap();
                        api.conversation(agent.conversation_id, &call_context)
                            .await?
                            .unwrap()
                            .abort(&call_context, Default::default())
                            .await?;
                        return Ok(reply(
                            format!("Stopped {name}."),
                            Some(&name),
                            Some(agent.conversation_id),
                        ));
                    }
                    let Some(message) = message else {
                        return Ok(reply(format!("{action} needs a message."), None, None));
                    };

                    // spawn and send: one commit starts a reporter for the message.
                    let task_id = api.task_id();
                    let (commit_name, commit_action) = (name.clone(), action.clone());
                    let result = api
                        .commit(
                            move |tx| async move {
                                let name = commit_name;
                                let action = commit_action;
                                // Both tasks belong to the main conversation and are background.
                                let background = TaskOptions {
                                    ownership: TaskOwnership::Conversation,
                                    conversation_id: None,
                                    background: Some(true),
                                };
                                let doc = tx.doc(&*SUBAGENTS_DOC, main).await?;
                                if action == "spawn" {
                                    if doc.get()?.agents.contains_key(&name) {
                                        return Ok(format!("{name} already exists; use send."));
                                    }
                                    let anchor = tx.create_task(&ANCHOR, (), background).await?;
                                    // Owned by a task of the main conversation: starts as a copy of the main agent.
                                    let child = tx
                                        .create_conversation(ConversationOwnership::Task {
                                            task_id: anchor.erase(),
                                        })
                                        .await?;
                                    // Subagents cannot start subagents, and know who they are.
                                    let change = AgentChange {
                                        extensions: Some(Some(ExtensionsChange::Edit {
                                            add: None,
                                            remove: Some(vec![subagent_tools()]),
                                        })),
                                        ..AgentChange::default().instructions(format!(
                                            "You are the subagent \"{name}\". Answer the main agent's requests."
                                        ))
                                    };
                                    configure(&tx, child.id, &change).await?;
                                    doc.edit(|state| {
                                        state.agents.insert(
                                            name.clone(),
                                            Subagent {
                                                conversation_id: child.id,
                                                reported: Vec::new(),
                                            },
                                        )
                                    })?;
                                }
                                let conversation_id = doc.get()?.agents[&name].conversation_id;
                                let input = ReporterInput {
                                    name: name.clone(),
                                    conversation_id,
                                    message,
                                    follow_up: action == "send" && follow_up == Some(true),
                                };
                                let reporter = tx.create_task(&REPORTER, input, background).await?;
                                doc.edit(|state| {
                                    state
                                        .reporters
                                        .insert(task_id.to_string(), reporter.erase())
                                })?;
                                Ok(if action == "send" {
                                    format!("Sent to {name}.")
                                } else {
                                    format!("Started {name}.")
                                })
                            },
                            &call_context,
                        )
                        .await?;
                    let current = api
                        .snapshot(&*SUBAGENTS_DOC, main, &call_context)
                        .await?
                        .and_then(|state| state.agents.get(&name).map(|agent| agent.conversation_id));
                    Ok(reply(result, Some(&name), current))
                },
            );
            // A call interrupted by a crash is not rerun: repeating `stop` could stop newer work.
            tool.replay = Some(Replay::Unsafe);
            define_extension(ExtensionDefinition {
                tasks: vec![ANCHOR.any(), REPORTER.any()],
                tools: vec![tool],
                ..ExtensionDefinition::new("subagent-tools")
            })
        })
        .clone()
}

// ─── Host setup ───

/// The main agent and the subagent share one scripted model, which answers each request by its last message.
fn route(messages: &[Message]) -> FauxResponseStep {
    let call = |input: serde_json::Value| -> FauxResponseStep {
        faux_assistant_message(
            vec![faux_tool_call("subagent", input, None)],
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxMessageOptions::default()
            },
        )
        .into()
    };
    let answer = |text: &str| -> FauxResponseStep {
        faux_assistant_message(text, FauxMessageOptions::default()).into()
    };
    // System messages carry prompt changes; the request is about the message before them.
    let last = messages
        .iter()
        .rev()
        .find(|message| !matches!(message, Message::System(_)))
        .unwrap();
    let text = match last {
        Message::User(user) => match &user.content {
            UserMessageContent::Text(text) => text.clone(),
            UserMessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    UserContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect(),
        },
        other => text_of(Some(other)).unwrap_or_default(),
    };
    // The main agent repeats what the tool said.
    if matches!(last, Message::ToolResult(_)) {
        return answer(&format!("OK. {text}"));
    }
    // The main agent.
    if text.contains("Start a subagent") {
        return call(
            json!({ "action": "spawn", "name": "reader", "message": "Summarize the plot of Moby Dick." }),
        );
    }
    if text.contains("whale's name") {
        return call(
            json!({ "action": "send", "name": "reader", "message": "What is the whale called?" }),
        );
    }
    if text.contains("every chapter") {
        return call(
            json!({ "action": "send", "name": "reader", "message": "Now go through all chapters in detail." }),
        );
    }
    if text.contains("Stop reader") {
        return call(json!({ "action": "stop", "name": "reader" }));
    }
    if text.contains("my subagents") {
        return call(json!({ "action": "status" }));
    }
    if text.contains("[subagent") {
        return answer("Noted.");
    }
    // The subagent: short answers, and a long chapter walk-through that is stopped halfway.
    if text.contains("Summarize the plot") {
        return answer("A whale, a captain, an obsession.");
    }
    if text.contains("whale called") {
        return answer("Moby Dick.");
    }
    let chapters: Vec<String> = (1..=135)
        .map(|index| format!("Chapter {index}: more whaling."))
        .collect();
    answer(&chapters.join("\n"))
}

fn scripted_models() -> Models {
    // It streams its answers at 50 tokens per second, like a slow real model.
    let faux = faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(50.0),
        ..RegisterFauxProviderOptions::default()
    });
    let models = create_models(Default::default());
    models.set_provider(faux.provider.clone());
    // More responses than the script needs; each request takes the next one.
    faux.set_responses(
        (0..40)
            .map(|_| {
                FauxResponseStep::async_factory(
                    |request: crate::types::TranscriptContext, _, _, _| async move {
                        match route(&request.messages) {
                            FauxResponseStep::Message(message) => Ok(*message),
                            FauxResponseStep::Factory(_) => unreachable!(),
                        }
                    },
                )
            })
            .collect::<Vec<_>>(),
    );
    models
}

async fn open(
    storage: &Arc<ControlledStorage>,
    models: &Models,
    registry: &Registry,
) -> (Harness, Conversation) {
    let ctx = BACKGROUND_CONTEXT.clone();
    let storage: Arc<dyn Storage> = storage.clone();
    let harness = Harness::open(
        storage,
        HarnessOptions::new(models.clone(), Arc::new(registry.clone())),
        &ctx,
    )
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

/// The main conversation's transcript as the TS `follow()` prints it (without colors).
async fn transcript(root: &Conversation) -> Vec<String> {
    let page = root
        .entries(None, None, 1000, None, &BACKGROUND_CONTEXT)
        .await
        .unwrap();
    let mut lines = Vec::new();
    for entry in page.items.into_iter().rev() {
        let Some(message) = entry.model.and_then(|model| model.into_iter().next()) else {
            continue;
        };
        match &message {
            Message::User(_) => {
                let text = text_of(Some(&message)).unwrap_or_default();
                // Input is either the user or a subagent's report, which arrives whenever the subagent is done.
                match text
                    .strip_prefix("[subagent ")
                    .and_then(|rest| rest.split_once(' '))
                    .and_then(|(name, rest)| {
                        rest.split_once(']')
                            .map(|(status, body)| (name, status, body))
                    }) {
                    None => lines.push(format!("> {text}")),
                    Some((name, status, body)) => {
                        let body = body.strip_prefix(' ').unwrap_or(body);
                        lines.push(format!(
                            "> {name}: {}",
                            if body.is_empty() { status } else { body }
                        ));
                    }
                }
            }
            Message::Assistant(assistant) => {
                for part in &assistant.content {
                    if let crate::types::AssistantContent::ToolCall(call) = part {
                        let args = &call.arguments;
                        let mut line =
                            format!("  {} {}", call.name, args["action"].as_str().unwrap_or(""));
                        if let Some(name) = args["name"].as_str() {
                            line.push_str(&format!(" {name}"));
                        }
                        if let Some(sent) = args["message"].as_str() {
                            line.push_str(&format!(" {}", json!(sent)));
                        }
                        lines.push(line);
                    }
                }
                let text = text_of_entry(Some(&message));
                if !text.is_empty() {
                    lines.push(text);
                }
            }
            Message::ToolResult(_) => {
                lines.push(format!(
                    "  → {}",
                    text_of(Some(&message)).unwrap_or_default()
                ));
            }
            Message::System(_) => {}
        }
    }
    lines
}

struct Host {
    harness: Harness,
    root: Conversation,
}

impl Host {
    /// Say something to the main agent and wait for its answer.
    async fn say(&self, text: &str) {
        let ctx = BACKGROUND_CONTEXT.clone();
        self.root
            .submit(SubmissionDraft::input(text), &ctx)
            .await
            .unwrap()
            .wait(&ctx)
            .await
            .unwrap();
    }

    /// Wait until every message to a subagent was answered and reported, and the main agent has reacted.
    async fn settle(&self) {
        let ctx = BACKGROUND_CONTEXT.clone();
        let reporters = self
            .harness
            .snapshot(&*SUBAGENTS_DOC, self.root.id, &ctx)
            .await
            .unwrap()
            .map(|state| state.reporters)
            .unwrap_or_default();
        for id in reporters.values() {
            self.harness.wait_for_task(*id, &ctx).await.unwrap();
        }
        self.root.wait_for_idle(&ctx).await.unwrap();
    }

    async fn reporters(&self) -> usize {
        self.harness
            .snapshot(&*SUBAGENTS_DOC, self.root.id, &BACKGROUND_CONTEXT)
            .await
            .unwrap()
            .map_or(0, |state| state.reporters.len())
    }

    async fn working(&self, name: &str) -> bool {
        let ctx = BACKGROUND_CONTEXT.clone();
        let Some(agent) = self
            .harness
            .snapshot(&*SUBAGENTS_DOC, self.root.id, &ctx)
            .await
            .unwrap()
            .and_then(|state| state.agents.get(name).cloned())
        else {
            return false;
        };
        self.harness
            .snapshot(&*LIVE_DOC, agent.conversation_id, &ctx)
            .await
            .unwrap()
            .is_some_and(|live| live.run.is_some())
    }
}

/// Poll `check` for up to 10 seconds.
async fn until<F, Fut>(mut check: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    for _ in 0..1000 {
        if check().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn example_23_subagent_background() {
    let ctx = BACKGROUND_CONTEXT.clone();
    let models = scripted_models();
    let registry = create_registry();
    registry.install(subagent_tools()).unwrap();
    let storage = Arc::new(ControlledStorage::persistent());
    let (harness, root) = open(&storage, &models, &registry).await;
    let host = Host { harness, root };

    // The main agent answers at once; the subagent's answer is reported back when it arrives.
    host.say("Start a subagent named reader that summarizes Moby Dick.")
        .await;
    host.settle().await;
    let first = transcript(&host.root).await;
    assert_eq!(
        first,
        [
            "> Start a subagent named reader that summarizes Moby Dick.",
            "  subagent spawn reader \"Summarize the plot of Moby Dick.\"",
            "  → Started reader.",
            "OK. Started reader.",
            "> reader: A whale, a captain, an obsession.",
            "Noted.",
        ]
    );

    // A long request, stopped while the subagent is still answering.
    host.say("Ask reader to summarize every chapter.").await;
    until(|| host.working("reader")).await;
    host.say("Stop reader.").await;
    host.settle().await;

    host.say("What are my subagents doing?").await;
    host.settle().await;
    let second = transcript(&host.root).await;
    assert_eq!(
        second[first.len()..],
        [
            "> Ask reader to summarize every chapter.",
            "  subagent send reader \"Now go through all chapters in detail.\"",
            "  → Sent to reader.",
            "OK. Sent to reader.",
            "> Stop reader.",
            "  subagent stop reader",
            "  → Stopped reader.",
            "OK. Stopped reader.",
            "> What are my subagents doing?",
            "  subagent status",
            "  → reader: idle",
            "OK. reader: idle",
        ]
    );

    // The process stops while the subagent works on a message; after the restart its answer still arrives.
    let before = host.reporters().await;
    host.root
        .submit(
            SubmissionDraft::input("Ask reader for the whale's name."),
            &ctx,
        )
        .await
        .unwrap();
    until(|| async { host.reporters().await > before }).await;
    host.harness.close(&ctx).await.unwrap();
    let (harness, root) = open(&storage, &models, &registry).await;
    let host = Host { harness, root };
    host.settle().await;
    let last = transcript(&host.root).await;
    let tail = &last[second.len()..];
    assert_eq!(tail[0], "> Ask reader for the whale's name.");
    assert_eq!(
        tail[1],
        "  subagent send reader \"What is the whale called?\""
    );
    // The report arrives once, after the restart, and the main agent reacts to it.
    let reports: Vec<&String> = tail
        .iter()
        .filter(|line| line.starts_with("> reader:"))
        .collect();
    assert_eq!(reports, ["> reader: Moby Dick."]);
    assert_eq!(tail.last().unwrap(), "Noted.");
    host.harness.close(&ctx).await.unwrap();
}
