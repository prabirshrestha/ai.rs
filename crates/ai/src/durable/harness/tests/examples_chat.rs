//! Ports of `test/examples/09`, `11`, `14`, `15`, and `20` as tests: each example runs against a Harness with the
//! faux model and asserts what the TS script prints. Example 18 (print mode) needs the read and bash coding tools of
//! M9; 22 (foreground subagent) watches events, so it lives in `examples_events`.

use std::sync::{Arc, LazyLock};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::{context, to_json};
use crate::chord::BACKGROUND_CONTEXT;
use crate::durable::DocToken;
use crate::durable::documents::define_doc;
use crate::durable::entries::{ASSISTANT_ENTRY, SYSTEM_ENTRY};
use crate::durable::env::ExecutionEnv;
use crate::durable::env::local::LocalExecutionEnv;
use crate::durable::harness::inbox::INBOX_DOC;
use crate::durable::harness::submissions::{AbortSubmissionResult, Submission};
use crate::durable::harness::types::{
    AgentChange, ConversationCreateOptions, EnvFactory, ExtensionDefinition, ExtensionsChange,
    HarnessOptions, HarnessSettings, ModelRef, QueueMode, SubmissionDraft, ToolExecutionResult,
    WhenBusy,
};
use crate::durable::harness::{
    Conversation, CreateOptions, Harness, create_registry, define_extension, define_tool, section,
    wrap_section,
};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{
    ContextEdit, ContextEditAction, EntryDraft, EntryHead, RewindableConversation, RewindableFork,
};
use crate::models::create_models;
use crate::providers::faux::{
    FauxMessageOptions, FauxResponseStep, faux_assistant_message, faux_provider, faux_tool_call,
};
use crate::types::{
    AssistantContent, Message, StopReason, SystemMessage, ToolCall, ToolResultMessage, Usage,
    UserContent, UserMessage,
};

fn model() -> ModelRef {
    ModelRef::new("faux", "faux-1")
}

fn faux_models(
    responses: Vec<FauxResponseStep>,
) -> (
    crate::models::Models,
    crate::providers::faux::FauxProviderHandle,
) {
    let faux = faux_provider(Default::default());
    let models = create_models(Default::default());
    models.set_provider(faux.provider.clone());
    faux.set_responses(responses);
    (models, faux)
}

fn answer(text: &str) -> FauxResponseStep {
    faux_assistant_message(text, FauxMessageOptions::default()).into()
}

async fn kinds(conversation: &Conversation, limit: usize) -> Vec<String> {
    let page = conversation
        .entries(None, None, limit, None, &BACKGROUND_CONTEXT)
        .await
        .unwrap();
    page.items
        .into_iter()
        .rev()
        .map(|entry| entry.kind)
        .collect()
}

// ---- 09: transcript history and model context ----

fn assistant_message(text: &str, calls: &[&str], stop_reason: Option<StopReason>) -> Message {
    let mut content: Vec<AssistantContent> = vec![AssistantContent::text(text)];
    content.extend(calls.iter().map(|id| {
        AssistantContent::ToolCall(ToolCall {
            id: (*id).into(),
            name: "read".into(),
            arguments: json!({}),
            thought_signature: None,
            namespace: None,
        })
    }));
    let stop_reason = stop_reason.unwrap_or(if calls.is_empty() {
        StopReason::Stop
    } else {
        StopReason::ToolUse
    });
    Message::Assistant(
        serde_json::from_value(json!({
            "role": "assistant",
            "content": content,
            "api": "example",
            "provider": "example",
            "model": "example",
            "usage": Usage::default(),
            "stopReason": stop_reason,
            "timestamp": 2,
        }))
        .unwrap(),
    )
}

fn tool_result_message(id: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.into(),
        tool_name: "read".into(),
        content: vec![UserContent::text(format!("file {id}"))],
        details: None,
        usage: None,
        nested_calls: None,
        is_error: false,
        timestamp: 3,
    })
}

fn user_message(text: &str, timestamp: u64) -> Message {
    Message::User(UserMessage {
        content: text.into(),
        timestamp,
    })
}

fn show(message: &Message) -> String {
    match message {
        Message::User(_) => format!(
            "user: {}",
            super::support::text_of(Some(message)).unwrap_or_default()
        ),
        Message::System(system) => format!(
            "system: {}",
            serde_json::to_string(&system.sections).unwrap()
        ),
        Message::Assistant(assistant) => format!(
            "assistant: {}",
            assistant
                .content
                .iter()
                .map(|part| match part {
                    AssistantContent::Text(text) => text.text.clone(),
                    AssistantContent::ToolCall(call) => format!("call({})", call.id),
                    _ => String::new(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        ),
        Message::ToolResult(result) => format!(
            "result({}){}",
            result.tool_call_id,
            if result.is_error { " error" } else { "" }
        ),
    }
}

#[tokio::test]
async fn example_09_context() {
    let ctx = BACKGROUND_CONTEXT.clone();
    let harness = Harness::open(
        Arc::new(MemoryStorage::new()),
        HarnessOptions::new(
            create_models(Default::default()),
            Arc::new(create_registry()),
        ),
        &ctx,
    )
    .await
    .unwrap();
    let transcript = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &ctx)
        .await
        .unwrap();
    let say = |kind: &'static str, model: Message| {
        let transcript = transcript.clone();
        async move {
            let id = transcript.id;
            let mut draft = EntryDraft::new(kind);
            draft.model = Some(vec![model]);
            transcript
                .commit(
                    move |tx| async move { tx.append_entry(id, draft).await },
                    &context(),
                )
                .await
                .unwrap()
        }
    };
    let append = |draft: EntryDraft| {
        let transcript = transcript.clone();
        async move {
            let id = transcript.id;
            transcript
                .commit(
                    move |tx| async move { tx.append_entry(id, draft).await },
                    &context(),
                )
                .await
                .unwrap()
        }
    };

    let question = say("message", user_message("read a and b", 1)).await;
    say(
        "message",
        assistant_message("I crashed", &[], Some(StopReason::Aborted)),
    )
    .await; // stored, never sent
    let calls = say("message", assistant_message("reading", &["a", "b"], None)).await;
    say("message", tool_result_message("b")).await; // results finish out of order
    let mut sections = IndexMap::new();
    sections.insert("cwd".to_string(), Some("<cwd>/repo</cwd>".to_string()));
    say(
        "pi.system",
        Message::System(SystemMessage {
            content: Default::default(),
            sections: Some(sections),
            tools_added: None,
            tools_removed: None,
            timestamp: 4,
        }),
    )
    .await;
    say("message", tool_result_message("a")).await;
    say("message", assistant_message("a and b look fine", &[], None)).await;
    let mut edit = EntryDraft::new("edit").data(json!("user fixed a typo"));
    edit.edits = Some(vec![ContextEdit {
        target: question.id,
        action: ContextEditAction::Replace {
            messages: vec![user_message("read files a and b", 1)],
        },
    }]);
    append(edit).await;
    append(EntryDraft::new("note").data(json!("display only"))).await;

    let view = transcript.context(&ctx).await.unwrap();
    assert_eq!(
        view.entries
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>(),
        [
            "message",
            "message",
            "message",
            "message",
            "pi.system",
            "message",
            "message",
            "edit",
            "note"
        ]
    );
    assert_eq!(
        view.messages.iter().map(show).collect::<Vec<_>>(),
        [
            "user: read files a and b",
            "assistant: reading call(a) call(b)",
            "result(a)",
            "result(b)",
            r#"system: {"cwd":"<cwd>/repo</cwd>"}"#,
            "assistant: a and b look fine",
        ]
    );

    // A fork at the tool call has no results yet; context() fills them in.
    let cut = transcript
        .fork(calls.id, ConversationCreateOptions::ownerless(), &ctx)
        .await
        .unwrap();
    assert_eq!(
        cut.context(&ctx)
            .await
            .unwrap()
            .messages
            .iter()
            .map(show)
            .collect::<Vec<_>>(),
        [
            "user: read a and b",
            "assistant: reading call(a) call(b)",
            "result(a) error",
            "result(b) error",
        ]
    );

    // A headed summary replaces everything before the entry it points at.
    let mut summary = EntryDraft::new("summary").head(EntryHead::SelfEntry);
    summary.model = Some(vec![user_message("Summary: a and b are fine.", 5)]);
    append(summary).await;
    let view = transcript.context(&ctx).await.unwrap();
    assert_eq!(view.head.map(|head| head.kind).as_deref(), Some("summary"));
    assert_eq!(
        view.messages.iter().map(show).collect::<Vec<_>>(),
        ["user: Summary: a and b are fine."]
    );

    // entries() pages the stored transcript, newest first. Nothing is ever deleted by heads or edits.
    let history = transcript.entries(None, None, 3, None, &ctx).await.unwrap();
    assert_eq!(
        history
            .items
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>(),
        ["summary", "note", "edit"]
    );
    assert!(history.next.is_some());
    harness.close(&ctx).await.unwrap();
}

// ---- 11: an extension that keeps its own per-conversation document ----

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
struct Todos {
    items: Vec<String>,
}

static TODOS_DOC: LazyLock<DocToken<Todos, RewindableConversation>> = LazyLock::new(|| {
    define_doc(crate::durable::types::DocDefinition::new(
        "example.todos",
        1,
        RewindableConversation {
            fork: RewindableFork::AsOf,
        },
        Todos::default,
    ))
    .unwrap()
});

#[tokio::test]
async fn example_11_extension_state() {
    let ctx = BACKGROUND_CONTEXT.clone();
    let todo = define_extension(ExtensionDefinition {
        tools: vec![define_tool(
            "todo",
            "Add an item to your todo list",
            json!({ "type": "object", "properties": { "item": { "type": "string" } }, "required": ["item"] }),
            |args, api, call_context| async move {
                let item = args["item"].as_str().unwrap_or_default().to_string();
                let id = api.conversation_id();
                let added = item.clone();
                api.commit(
                    move |tx| async move {
                        tx.doc(&*TODOS_DOC, id)
                            .await?
                            .edit(|todos| todos.items.push(added))
                    },
                    &call_context,
                )
                .await?;
                Ok(ToolExecutionResult::text(format!("added {item}")))
            },
        )],
        sections: vec![section(
            "todos",
            |input, render_context| async move {
                let todos = input
                    .read
                    .snapshot(&*TODOS_DOC, input.conversation_id, &render_context)
                    .await?;
                Ok(todos
                    .filter(|todos| !todos.items.is_empty())
                    .map(|todos| todos.items.join("\n")))
            },
            None,
        )],
        ..ExtensionDefinition::new("todo")
    });
    let (models, _faux) = faux_models(vec![
        faux_assistant_message(
            vec![faux_tool_call(
                "todo",
                json!({ "item": "fix the build" }),
                None,
            )],
            FauxMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxMessageOptions::default()
            },
        )
        .into(),
        answer("Noted."),
        answer("Working on it."),
    ]);
    let registry = create_registry();
    registry.install(todo).unwrap();
    let harness = Harness::open(
        Arc::new(MemoryStorage::new()),
        HarnessOptions::new(models, Arc::new(registry)),
        &ctx,
    )
    .await
    .unwrap();
    let root = harness
        .root(
            &ctx,
            CreateOptions {
                agent: Some(AgentChange::default().model(model())),
                init: None,
            },
        )
        .await
        .unwrap();
    root.submit(SubmissionDraft::input("Remember to fix the build."), &ctx)
        .await
        .unwrap()
        .wait(&ctx)
        .await
        .unwrap();
    assert_eq!(
        harness.snapshot(&*TODOS_DOC, root.id, &ctx).await.unwrap(),
        Some(Todos {
            items: vec!["fix the build".into()]
        })
    );

    // The next request's system prompt carries the list. The first system message only announced the todo tool.
    root.submit(SubmissionDraft::input("What is next?"), &ctx)
        .await
        .unwrap()
        .wait(&ctx)
        .await
        .unwrap();
    let messages = root.context(&ctx).await.unwrap().messages;
    let systems: Vec<_> = messages
        .iter()
        .filter_map(|message| match message {
            Message::System(system) => Some(json!({
                "sections": system.sections,
                "toolsAdded": system.tools_added.as_ref().map(|tools| tools.iter().map(|tool| tool.name.clone()).collect::<Vec<_>>()),
            })),
            _ => None,
        })
        .collect();
    assert_eq!(
        systems,
        [
            json!({ "sections": null, "toolsAdded": ["todo"] }),
            json!({ "sections": { "todos": "<todos>\nfix the build\n</todos>" }, "toolsAdded": null }),
        ]
    );
    harness.close(&ctx).await.unwrap();
}

// ---- 14: a chat turn ----

#[tokio::test]
async fn example_14_chat() {
    let ctx = BACKGROUND_CONTEXT.clone();
    let (models, _faux) = faux_models(vec![answer("Paris.")]);
    let registry = create_registry();
    registry
        .install(define_extension(ExtensionDefinition {
            sections: vec![section(
                "preamble",
                |_, _| async { Ok(Some("You answer in one word.".to_string())) },
                Some(false),
            )],
            ..ExtensionDefinition::new("terse")
        }))
        .unwrap();
    let harness = Harness::open(
        Arc::new(MemoryStorage::new()),
        HarnessOptions::new(models, Arc::new(registry)),
        &ctx,
    )
    .await
    .unwrap();
    let root = harness
        .root(
            &ctx,
            CreateOptions {
                agent: Some(AgentChange::default().model(model())),
                init: None,
            },
        )
        .await
        .unwrap();
    let answered = root
        .submit(SubmissionDraft::input("Capital of France?"), &ctx)
        .await
        .unwrap()
        .wait(&ctx)
        .await
        .unwrap();
    let answer_id = answered.answer.unwrap();
    let entry = root
        .commit(
            move |tx| async move { tx.entry_of(&ASSISTANT_ENTRY, answer_id).await },
            &ctx,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        to_json(&entry.model.unwrap()[0])["content"],
        json!([{ "type": "text", "text": "Paris." }])
    );
    assert_eq!(
        kinds(&root, 10).await,
        ["pi.user", "pi.system", "pi.assistant"]
    );
    harness.close(&ctx).await.unwrap();
}

// ---- 15: system prompt sections and per-conversation instructions ----

async fn system_entries(conversation: &Conversation) -> Vec<serde_json::Value> {
    let page = conversation
        .entries(None, None, 20, None, &BACKGROUND_CONTEXT)
        .await
        .unwrap();
    page.items
        .into_iter()
        .rev()
        .filter(|entry| SYSTEM_ENTRY.is(Some(entry)))
        .flat_map(|entry| entry.model.unwrap_or_default())
        .map(|message| {
            let mut value = to_json(&message);
            value.as_object_mut().unwrap().remove("timestamp");
            value
        })
        .collect()
}

#[tokio::test]
async fn example_15_system_prompt() {
    let ctx = BACKGROUND_CONTEXT.clone();
    let (models, _faux) = faux_models(vec![answer("Done."), answer("Done."), answer("Done.")]);
    let coding = define_extension(ExtensionDefinition {
        sections: vec![
            section(
                "preamble",
                |_, _| async { Ok(Some("You are a coding agent.".to_string())) },
                Some(false),
            ),
            section(
                "cwd",
                |input, _| async move { Ok(input.env.map(|env| env.cwd())) },
                None,
            ),
        ],
        ..ExtensionDefinition::new("coding")
    });
    let agents_md = define_extension(ExtensionDefinition {
        sections: vec![section(
            "agents_md",
            |_, _| async { Ok(Some("Run npm run check after changes.".to_string())) },
            None,
        )],
        ..ExtensionDefinition::new("agents-md")
    });
    // Another extension decorates a section by key without replacing it.
    let terse = define_extension(ExtensionDefinition {
        wraps: vec![wrap_section("preamble", |preamble| {
            let render = preamble.render.clone();
            Ok(crate::durable::harness::types::PromptSection {
                render: Arc::new(move |input, render_context| {
                    let render = render.clone();
                    Box::pin(async move {
                        Ok(render(input, render_context)
                            .await?
                            .map(|text| format!("{text} Be terse.")))
                    })
                }),
                ..preamble
            })
        })],
        ..ExtensionDefinition::new("terse")
    });
    let registry = create_registry();
    registry.install(coding).unwrap();
    registry.install(agents_md.clone()).unwrap();
    registry.install(terse).unwrap();
    // The environment follows each conversation's agent `cwd`.
    let env: EnvFactory = Arc::new(|target, _| {
        let env: Arc<dyn ExecutionEnv> = Arc::new(LocalExecutionEnv::at(
            target.cwd.unwrap_or_else(|| "/".into()),
        ));
        Box::pin(async move { Ok(Some(env)) })
    });
    let mut options = HarnessOptions::new(models, Arc::new(registry));
    options.env = Some(env);
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &ctx)
        .await
        .unwrap();
    let root = harness
        .root(
            &ctx,
            CreateOptions {
                agent: Some(AgentChange::default().model(model()).cwd("/repo")),
                init: None,
            },
        )
        .await
        .unwrap();
    // A subagent deselects AGENTS.md and gets its own instructions, rendered last as the `instructions` section.
    let subagent = harness
        .create_conversation(
            ConversationCreateOptions {
                agent: Some(AgentChange {
                    extensions: Some(Some(ExtensionsChange::Edit {
                        add: None,
                        remove: Some(vec![agents_md]),
                    })),
                    ..AgentChange::default()
                        .model(model())
                        .cwd("/repo")
                        .instructions("Only read; never edit files.")
                }),
                ..ConversationCreateOptions::ownerless()
            },
            &ctx,
        )
        .await
        .unwrap();
    for (conversation, text) in [(&root, "Fix the build."), (&subagent, "Read the logs.")] {
        conversation
            .submit(SubmissionDraft::input(text), &ctx)
            .await
            .unwrap()
            .wait(&ctx)
            .await
            .unwrap();
    }
    assert_eq!(
        system_entries(&root).await,
        [json!({
            "role": "system",
            "content": "",
            "sections": {
                "preamble": "You are a coding agent. Be terse.",
                "cwd": "<cwd>\n/repo\n</cwd>",
                "agents_md": "<agents_md>\nRun npm run check after changes.\n</agents_md>",
            },
        })]
    );
    assert_eq!(
        system_entries(&subagent).await,
        [json!({
            "role": "system",
            "content": "",
            "sections": {
                "preamble": "You are a coding agent. Be terse.",
                "cwd": "<cwd>\n/repo\n</cwd>",
                "instructions": "<instructions>\nOnly read; never edit files.\n</instructions>",
            },
        })]
    );

    // When a section's output changes, the next request appends only the change.
    root.configure(AgentChange::default().cwd("/repo/packages"), &ctx)
        .await
        .unwrap();
    root.submit(SubmissionDraft::input("Now the package."), &ctx)
        .await
        .unwrap()
        .wait(&ctx)
        .await
        .unwrap();
    let after = system_entries(&root).await;
    assert_eq!(after.len(), 2);
    assert_eq!(
        after[1],
        json!({ "role": "system", "content": "", "sections": { "cwd": "<cwd>\n/repo/packages\n</cwd>" } })
    );
    harness.close(&ctx).await.unwrap();
}

// ---- 20: the inbox ----

#[tokio::test]
async fn example_20_inbox() {
    let ctx = BACKGROUND_CONTEXT.clone();
    // The first answer waits until we let it go, so the conversation stays busy while we submit more.
    let held = crate::durable::session::tests::support::Deferred::default();
    let gate = held.clone();
    let slow = FauxResponseStep::async_factory(move |_, _, _, _| {
        let gate = gate.clone();
        async move {
            gate.wait().await;
            Ok(faux_assistant_message(
                "Answer to the first question.",
                FauxMessageOptions::default(),
            ))
        }
    });
    let (models, _faux) = faux_models(vec![slow, answer("Answer to the follow-up and the steer.")]);
    // Settings apply to every conversation: place every queued follow-up at once instead of one per run.
    let mut options = HarnessOptions::new(models, Arc::new(create_registry()));
    options.settings = Some(Arc::new(|| HarnessSettings {
        follow_up_mode: Some(QueueMode::All),
        ..HarnessSettings::default()
    }));
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &ctx)
        .await
        .unwrap();
    let root = harness
        .root(
            &ctx,
            CreateOptions {
                agent: Some(AgentChange::default().model(model())),
                init: None,
            },
        )
        .await
        .unwrap();
    let submit = |draft: SubmissionDraft| {
        let root = root.clone();
        async move { root.submit(draft, &BACKGROUND_CONTEXT).await }
    };

    let first = submit(SubmissionDraft::input("First question"))
        .await
        .unwrap();
    // While busy, input queues as a follow-up (the default) or a steer, and writes queue too.
    let follow_up = submit(SubmissionDraft::input("A follow-up")).await.unwrap();
    let steer = submit(SubmissionDraft::input("A steer").when_busy(WhenBusy::Steer))
        .await
        .unwrap();
    let note = submit(SubmissionDraft::write(
        EntryDraft::new("app.note").data(json!("noted while busy")),
    ))
    .await
    .unwrap();
    let withdrawn = submit(SubmissionDraft::input("Never mind")).await.unwrap();
    // whenBusy: "reject" refuses instead of queueing.
    let rejected = submit(SubmissionDraft::input("Now or never").when_busy(WhenBusy::Reject))
        .await
        .unwrap_err();
    assert_eq!(
        rejected.to_string(),
        format!("Conversation {} is busy", root.id)
    );
    // A queued submission can be withdrawn until a boundary places it.
    assert_eq!(
        withdrawn.abort(&ctx).await.unwrap(),
        AbortSubmissionResult::Aborted
    );

    let inbox = harness
        .snapshot(&*INBOX_DOC, root.id, &ctx)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        inbox
            .items
            .iter()
            .map(|item| format!("{} {}", item.id(), to_json(item)["mode"].as_str().unwrap()))
            .collect::<Vec<_>>(),
        [
            format!("{} followUp", follow_up.id),
            format!("{} steer", steer.id),
            format!("{} write", note.id),
        ]
    );

    // The first answer ends the run at a final boundary: the write is placed first, then the follow-up and the steer,
    // which start the next run together.
    held.resolve();
    let status = |submission: Submission| async move {
        let record = submission.wait(&BACKGROUND_CONTEXT).await.unwrap();
        format!(
            "{} {}",
            to_json(&record.status).as_str().unwrap(),
            record.reason.unwrap_or_default()
        )
    };
    assert_eq!(status(first).await, "done ");
    assert_eq!(status(follow_up).await, "done ");
    assert_eq!(status(steer).await, "done ");
    assert_eq!(status(note).await, "done ");
    assert_eq!(status(withdrawn).await, "unanswered aborted");
    assert_eq!(
        kinds(&root, 20).await,
        // No section or tool has anything to show, so no system entry is written.
        [
            "pi.user",
            "pi.assistant",
            "app.note",
            "pi.user",
            "pi.user",
            "pi.assistant"
        ]
    );
    harness.close(&ctx).await.unwrap();
}
