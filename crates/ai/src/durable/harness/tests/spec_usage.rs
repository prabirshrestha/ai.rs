//! Port of `test/spec-usage.test.ts`: the usage examples of Pi's `docs/spec.md`, compile-checked. As in TS, the
//! functions are never called; the names the spec leaves to the application are parameters. Sections 7.1 and 7.2
//! (extensions, host setup, tool filters, wraps, plan mode, reload) are in `examples`; subagents (7.3), the chat
//! extension (7.4), the payment task (5.1) and the remaining sequences are in `more_examples`.
//!
//! Divergences: the subagent extension removes itself from its children through a `OnceLock`, since a Rust value
//! cannot name itself in its own definition; live settings are a callback over the settings manager instead of
//! getters.

#![allow(dead_code)]

use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use crate::chord::Context;
use crate::durable::documents::{DocToken, define_doc};
use crate::durable::env::ExecutionEnv;
use crate::durable::errors::{Error, Result};
use crate::durable::harness::agent::configure;
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::live::LIVE_DOC;
use crate::durable::harness::tool::TOOL_TASK;
use crate::durable::harness::tool::ToolExecutionApi;
use crate::durable::harness::types::{
    AgentChange, BeforeToolDecision, CompactionPolicyOverrides, ConversationStreamOptions,
    Extension, ExtensionDefinition, ExtensionsChange, GenerationHooks, HarnessOptions,
    HarnessSettings, ModelRef, OnYieldHook, Replay, SubmissionDraft, ToolExecutionResult,
    ToolHooks, ToolRegistration, ToolsChange,
};
use crate::durable::harness::{
    CreateOptions, Harness, create_registry, define_extension, define_tool, hook, section,
    wrap_tool,
};
use crate::durable::ids::{ConversationId, EntryId, TaskId};
use crate::durable::session::Transaction;
use crate::durable::tasks::{NextTaskState, Task, TaskDefinition, define_task};
use crate::durable::tools::{
    BashToolOptions, CODING_TOOLS, create_bash_tool, create_edit_tool, create_read_tool,
};
use crate::durable::types::{
    ConversationOwnership, ConversationQuery, DocDefinition, EntryDraft, LatestConversation,
    LatestFork, Storage, SubmissionStatus, SubmissionType, TaskOptions,
};
use crate::models::Models;
use crate::types::ToolCall;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
struct PlanModeState {
    enabled: bool,
}

static PLAN_MODE_DOC: LazyLock<DocToken<PlanModeState, LatestConversation>> = LazyLock::new(|| {
    define_doc(crate::durable::types::DocDefinition::new(
        "app.plan-mode",
        1,
        LatestConversation {
            fork: LatestFork::Current,
        },
        PlanModeState::default,
    ))
    .unwrap()
});

/// The names the spec leaves to the application.
struct App {
    storage: Arc<dyn Storage>,
    models: Models,
    sonnet: ModelRef,
    local_env: Arc<dyn Fn(String) -> Arc<dyn ExecutionEnv> + Send + Sync>,
    render_agents_md: fn() -> Option<String>,
    render_skills: fn() -> Option<String>,
    is_dangerous: fn(&ToolCall) -> bool,
    writes: fn(&ToolCall) -> bool,
    request_second_pass: OnYieldHook,
    record_metric: fn(&str, u128),
}

/// Never called: compile-checks the spec's host setup.
async fn examples(app: App, context: &Context) -> Result<()> {
    let edit_tool = create_edit_tool();
    let bash_tool = create_bash_tool(BashToolOptions::default());

    // ─── Section 7.1: extensions and host setup ───

    let render_agents_md = app.render_agents_md;
    let context_files = define_extension(ExtensionDefinition {
        sections: vec![section(
            "agents-md",
            move |_, _| async move { Ok(render_agents_md()) },
            None,
        )],
        ..ExtensionDefinition::new("context-files")
    });
    let render_skills = app.render_skills;
    let skills_section = || {
        section(
            "skills",
            move |_, _| async move { Ok(render_skills()) },
            None,
        )
    };
    let skills = define_extension(ExtensionDefinition {
        sections: vec![skills_section()],
        ..ExtensionDefinition::new("skills")
    });
    let skills_v2 = define_extension(ExtensionDefinition {
        sections: vec![skills_section()],
        ..ExtensionDefinition::new("skills")
    });
    let coding = define_extension(ExtensionDefinition {
        sections: vec![
            section(
                "preamble",
                |_, _| async { Ok(Some("You are an expert coding assistant.".to_string())) },
                Some(false),
            ),
            // The environment the host built for this conversation, in the conversation's directory.
            section(
                "cwd",
                |input, _| async move {
                    Ok(input
                        .env
                        .map(|env| format!("Working directory: {}", env.cwd())))
                },
                None,
            ),
        ],
        ..ExtensionDefinition::new("coding")
    });
    let is_dangerous = app.is_dangerous;
    let permissions = define_extension(ExtensionDefinition {
        hooks: vec![hook(
            &TOOL_TASK,
            ToolHooks {
                before_tool: Some(Arc::new(move |call, _, _| {
                    let blocked = is_dangerous(&call);
                    Box::pin(async move {
                        Ok(blocked.then(|| BeforeToolDecision {
                            block: Some("Needs approval".into()),
                            ..BeforeToolDecision::default()
                        }))
                    })
                })),
                ..ToolHooks::default()
            },
        )],
        ..ExtensionDefinition::new("permissions")
    });
    // A role and a review loop, for conversations that select it.
    let reviewer = define_extension(ExtensionDefinition {
        sections: vec![section(
            "role",
            |_, _| async {
                Ok(Some(
                    "You review diffs. Report problems as a list. Never edit files.".to_string(),
                ))
            },
            None,
        )],
        hooks: vec![hook(
            &GENERATION_TASK,
            GenerationHooks {
                on_yield: Some(app.request_second_pass.clone()),
                ..GenerationHooks::default()
            },
        )],
        ..ExtensionDefinition::new("reviewer")
    });

    let record_metric = app.record_metric;
    let timing = define_extension(ExtensionDefinition {
        wraps: vec![wrap_tool(&bash_tool.name, move |tool| {
            let execute = tool.execute.clone();
            Ok(ToolRegistration {
                execute: Arc::new(move |args, api, context| {
                    let execute = execute.clone();
                    Box::pin(async move {
                        let start = Instant::now();
                        let result = execute(args, api, context).await;
                        record_metric("bash", start.elapsed().as_millis());
                        result
                    })
                }),
                ..tool
            })
        })],
        ..ExtensionDefinition::new("timing")
    });
    // A bash inside a Python virtualenv for one conversation: it replaces CODING_TOOLS' bash in place,
    // and Timing, if selected, wraps it.
    let venv = define_extension(ExtensionDefinition {
        tools: vec![create_bash_tool(BashToolOptions {
            command_prefix: Some("source .venv/bin/activate".into()),
            ..BashToolOptions::default()
        })],
        ..ExtensionDefinition::new("venv")
    });

    // ─── Section 7.2: hooks reading extension state ───

    let writes = app.writes;
    let plan_mode = define_extension(ExtensionDefinition {
        hooks: vec![hook(
            &TOOL_TASK,
            ToolHooks {
                // An absent document means plan mode is off.
                before_tool: Some(Arc::new(move |call, api, context| {
                    Box::pin(async move {
                        let enabled = api
                            .snapshot(&*PLAN_MODE_DOC, api.conversation_id(), &context)
                            .await?
                            .is_some_and(|state| state.enabled);
                        Ok((enabled && writes(&call)).then(|| BeforeToolDecision {
                            block: Some("Plan mode: read-only".into()),
                            ..BeforeToolDecision::default()
                        }))
                    })
                })),
                ..ToolHooks::default()
            },
        )],
        ..ExtensionDefinition::new("plan-mode")
    });

    // ─── Sections 2.2 and 7.1: host setup, tool filters, extension selection, plan mode, and reload ───

    let registry = create_registry();
    for extension in [
        CODING_TOOLS.clone(),
        coding.clone(),
        context_files.clone(),
        skills.clone(),
        permissions.clone(),
        reviewer,
    ] {
        registry.install(extension)?;
    }
    let mut options = HarnessOptions::new(app.models, Arc::new(registry.clone()));
    // Reviewer is installed but not selected by default: only conversations that select it get its role and hooks.
    let defaults = vec![
        CODING_TOOLS.clone(),
        coding,
        context_files,
        skills.clone(),
        permissions,
    ];
    options.settings = Some(Arc::new(move || HarnessSettings {
        extensions: Some(defaults.clone()),
        ..HarnessSettings::default()
    }));
    let local_env = app.local_env;
    options.env = Some(Arc::new(move |target, _| {
        // cached LocalExecutionEnv per directory
        let env = local_env(target.cwd.unwrap_or_else(|| "/".into()));
        Box::pin(async move { Ok(Some(env)) })
    }));
    let harness = Harness::open(app.storage, options, context).await?;
    // The conversation remembers its model and directory; a restart elsewhere keeps both.
    let root = harness
        .root(
            context,
            CreateOptions {
                agent: Some(AgentChange::default().model(app.sonnet).cwd("/work")),
                init: None,
            },
        )
        .await?;

    let remove = |tool: &ToolRegistration| AgentChange {
        tools: Some(Some(ToolsChange::Remove(vec![tool.clone()]))),
        ..AgentChange::default()
    };
    root.configure(remove(&edit_tool), context).await?;
    root.configure(remove(&bash_tool), context).await?; // edit is offered again
    root.configure(
        AgentChange {
            tools: Some(None),
            ..AgentChange::default()
        },
        context,
    )
    .await?; // every tool of the selected extensions again

    registry.install(timing)?;
    registry.install(venv.clone())?; // installed, but not in the default selection
    let conversation = &root;
    conversation
        .configure(
            AgentChange {
                extensions: Some(Some(ExtensionsChange::Edit {
                    add: Some(vec![venv]),
                    remove: None,
                })),
                ..AgentChange::default()
            },
            context,
        )
        .await?;

    registry.install(plan_mode)?;
    // PlanMode is installed after the default selection was set; /plan toggles this conversation's state.
    let id = root.id;
    root.commit(
        move |tx| async move {
            tx.doc(&*PLAN_MODE_DOC, id)
                .await?
                .edit(|state| state.enabled = true)
        },
        context,
    )
    .await?;

    registry.install(skills_v2)?; // same name: conversations selecting skills render v2 at their next request
    registry.uninstall(&skills)?; // selecting conversations get a system delta removing its section
    harness.close(context).await
}

#[test]
fn compiles_the_specs_usage_examples() {
    let _ = examples;
}

/// More names the spec leaves to the application.
struct MoreApp {
    session: crate::durable::session::Session,
    conversation_id: ConversationId,
    message: EntryDraft,
    haiku: ModelRef,
    worktree: String,
    name: String,
    containers: Arc<dyn Fn(String, String) -> Arc<dyn ExecutionEnv> + Send + Sync>,
    local_env: Arc<dyn Fn(String) -> Arc<dyn ExecutionEnv> + Send + Sync>,
    answer_text: fn(&ToolExecutionApi, EntryId) -> String,
    payments: Arc<dyn Payments>,
    new_key: fn() -> String,
    receipt_entry: fn(&Receipt) -> EntryDraft,
    manager: Arc<dyn SettingsManager>,
    anchor: Task<(), JsonValue, ()>,
    reporter: Task<(), JsonValue, ()>,
    subagent_tool: ToolRegistration,
}

struct Receipt {
    id: String,
}

#[async_trait::async_trait]
trait Payments: Send + Sync {
    async fn charge(&self, key: &str) -> Result<Receipt>;
    async fn cancel(&self, checkpoint: &Charge) -> Result<()>;
}

trait SettingsManager: Send + Sync {
    fn timeout_ms(&self) -> u64;
    fn auto_compact(&self) -> bool;
    fn set_auto_compact(&self, value: bool);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
enum Charge {
    Prepare,
    Charge { key: String },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Container {
    image: String,
}

/// Never called: compile-checks the spec's other examples and sequences.
async fn more_examples(
    app: MoreApp,
    models: Models,
    storage: Arc<dyn Storage>,
    context: &Context,
) -> Result<()> {
    let read_tool = create_read_tool();

    // ─── Section 7.3: subagents ───

    let own_extension: Arc<OnceLock<Extension>> = Arc::default();
    let itself = own_extension.clone();
    let answer_text = app.answer_text;
    let subagent = define_extension(ExtensionDefinition {
        tools: vec![ToolRegistration {
            replay: Some(Replay::Safe),
            ..define_tool(
                "subagent",
                "Delegate a self-contained task to a subagent and get its answer back.",
                json!({
                    "type": "object",
                    "properties": { "task": { "type": "string" } },
                    "required": ["task"],
                }),
                move |args, api, context| {
                    let itself = itself.clone();
                    async move {
                        let task = args["task"].as_str().unwrap_or_default().to_string();
                        let owner = api.task_id();
                        let child = api
                            .commit(
                                move |tx| async move {
                                    let query = ConversationQuery {
                                        owner_task_id: Some(owner),
                                        ..ConversationQuery::default()
                                    };
                                    if let Some(existing) =
                                        tx.scan_conversations(query, 1, None).await?.items.first()
                                    {
                                        return Ok(existing.id);
                                    }
                                    // Starts as a copy of this conversation's agent: model, thinking level, cwd,
                                    // extensions, tools.
                                    let created = tx
                                        .create_conversation(ConversationOwnership::Task {
                                            task_id: owner,
                                        })
                                        .await?;
                                    // Without this extension, the child is not offered this tool.
                                    let remove = itself.get().cloned().into_iter().collect();
                                    configure(
                                        &tx,
                                        created.id,
                                        &AgentChange {
                                            extensions: Some(Some(ExtensionsChange::Edit {
                                                add: None,
                                                remove: Some(remove),
                                            })),
                                            ..AgentChange::default()
                                        },
                                    )
                                    .await?;
                                    Ok(created.id)
                                },
                                &context,
                            )
                            .await?;
                        api.details(json!({ "conversationId": child }), &context)
                            .await?;
                        let handle = api.conversation(child, &context).await?.unwrap();
                        let request = SubmissionDraft::Input {
                            request_id: Some(format!("subagent:{owner}")),
                            content: task.into(),
                            when_busy: None,
                        };
                        let settled = handle
                            .submit(request, &context)
                            .await?
                            .wait(&context)
                            .await?;
                        if settled.status != SubmissionStatus::Done
                            || settled.type_ != SubmissionType::Input
                        {
                            return Err(Error::message(format!(
                                "Subagent failed: {:?}",
                                settled.status
                            )));
                        }
                        Ok(ToolExecutionResult::text(answer_text(
                            &api,
                            settled.answer.unwrap(),
                        )))
                    }
                },
            )
        }],
        ..ExtensionDefinition::new("subagent")
    });
    let _ = own_extension.set(subagent.clone());

    let subagent_tools = define_extension(ExtensionDefinition {
        tasks: vec![app.anchor.any(), app.reporter.any()],
        tools: vec![app.subagent_tool.clone()],
        ..ExtensionDefinition::new("subagent-tools")
    });

    // ─── Section 7.4 ───

    let chat = define_extension(ExtensionDefinition {
        sections: vec![section(
            "preamble",
            |_, _| async { Ok(Some("You are a helpful assistant.".to_string())) },
            Some(false),
        )],
        ..ExtensionDefinition::new("chat")
    });

    // ─── Section 5.1: a task with intent, effect, outcome ───

    let (payments, cancels, new_key, receipt_entry) = (
        app.payments.clone(),
        app.payments.clone(),
        app.new_key,
        app.receipt_entry,
    );
    let payment: Task<(), Charge, JsonValue> = define_task(
        TaskDefinition::new("app.payment", 1, |_: &()| Charge::Prepare)
            // Intent, effect, outcome.
            .phase("prepare", move |_, runtime, context| async move {
                runtime
                    .commit(
                        move |_, _| async move {
                            Ok(Some(NextTaskState::running(Charge::Charge {
                                key: new_key(),
                            })))
                        },
                        &context,
                    )
                    .await
            })
            .phase("charge", move |task, runtime, context| {
                let payments = payments.clone();
                async move {
                    let Charge::Charge { key } = &task.checkpoint else {
                        unreachable!("charge phase")
                    };
                    let receipt = payments.charge(key).await?; // idempotent by key
                    runtime
                        .commit(
                            move |tx, current| async move {
                                let entry = tx
                                    .append_entry(current.conversation_id, receipt_entry(&receipt))
                                    .await?;
                                Ok(Some(NextTaskState::completed(
                                    json!({ "entryId": entry.id }),
                                )))
                            },
                            &context,
                        )
                        .await
                }
            })
            // The abort handler decides the outcome; returning without one faults the task.
            .abort(move |task, runtime, context| {
                let payments = cancels.clone();
                async move {
                    payments.cancel(&task.checkpoint).await?;
                    runtime
                        .commit(
                            |_, _| async { Ok(Some(NextTaskState::aborted("user"))) },
                            &context,
                        )
                        .await
                }
            }),
    );

    let follow: Task<JsonValue, JsonValue, ()> = define_task(
        TaskDefinition::new(
            "app.follow",
            1,
            |_: &JsonValue| json!({ "phase": "follow" }),
        )
        .phase("follow", |_, _, _| async { Ok(()) })
        .abort(|_, _, _| async { Ok(()) }),
    );

    // ─── Sequences ───

    // Section 2.2: a tool's commit creates a configured child.
    let (haiku, worktree) = (app.haiku.clone(), app.worktree.clone());
    let child_in_tool_commit = move |tx: Transaction, task_id: TaskId| async move {
        // In a tool's commit. The child starts as a copy of this conversation's agent: model, extensions, tools, cwd.
        let child = tx
            .create_conversation(ConversationOwnership::Task { task_id })
            .await?;
        // A cheaper model, only the read tool, and its own worktree; everything else stays as copied.
        configure(
            &tx,
            child.id,
            &AgentChange::default()
                .model(haiku)
                .tools(vec![read_tool])
                .cwd(worktree),
        )
        .await
    };

    // Section 2.2: settings read live through a callback.
    let manager = app.manager.clone();
    let live_settings = move || -> crate::durable::harness::types::SettingsSource {
        manager.set_auto_compact(false); // no Session write; every conversation follows at its next threshold check
        let manager = manager.clone();
        Arc::new(move || HarnessSettings {
            stream: Some(ConversationStreamOptions {
                timeout_ms: Some(manager.timeout_ms()),
                ..ConversationStreamOptions::default()
            }),
            compaction: Some(CompactionPolicyOverrides {
                enabled: Some(manager.auto_compact()),
                ..CompactionPolicyOverrides::default()
            }),
            ..HarnessSettings::default()
        })
    };

    // Section 2.2: an environment per conversation.
    // Absent: the conversation runs locally. Only conversations with this document run in a container.
    // Subagents do not copy it: their creator writes it too when they should run in the container.
    let container_doc: DocToken<Container, LatestConversation> = define_doc(DocDefinition::new(
        "app.container",
        1,
        LatestConversation {
            fork: LatestFork::Current,
        },
        || Container {
            image: "node:22".into(),
        },
    ))?;
    let (containers, local_env) = (app.containers.clone(), app.local_env.clone());
    let mut options = HarnessOptions::new(models, Arc::new(create_registry()));
    options.env = Some(Arc::new(move |target, context| {
        let (container_doc, containers, local_env) =
            (container_doc.clone(), containers.clone(), local_env.clone());
        Box::pin(async move {
            let container = target
                .read
                .snapshot(&container_doc, target.conversation_id, &context)
                .await?;
            Ok(Some(match container {
                Some(container) => containers(
                    container.image,
                    target.cwd.unwrap_or_else(|| "/work".into()),
                ),
                // cached LocalExecutionEnv per directory
                None => local_env(target.cwd.unwrap_or_else(|| ".".into())),
            }))
        })
    }));
    let container_harness = Harness::open(storage, options, context).await?;

    // Section 7.3: a named subagent's spawn commit.
    let (anchor, name) = (app.anchor.clone(), app.name.clone());
    let spawn = move |tx: Transaction| async move {
        // In subagentTool's spawn commit, after the name checks:
        let anchor = tx
            .create_task(
                &anchor,
                (),
                TaskOptions {
                    background: Some(true),
                    ..TaskOptions::conversation(None)
                },
            )
            .await?;
        // Owned by a task of the parent: starts as a copy of the parent's agent.
        let child = tx
            .create_conversation(ConversationOwnership::Task {
                task_id: anchor.erase(),
            })
            .await?;
        configure(
            &tx,
            child.id,
            &AgentChange {
                extensions: Some(Some(ExtensionsChange::Edit {
                    add: None,
                    remove: Some(vec![subagent_tools]),
                })),
                ..AgentChange::default()
            }
            .instructions(format!(
                "You are the subagent \"{name}\". Answer the main agent's requests."
            )),
        )
        .await
    };

    // Section 4: table reads before the first table write; documents stay usable.
    let (conversation_id, message) = (app.conversation_id, app.message.clone());
    app.session
        .commit(
            move |tx| async move {
                let conversation = tx.conversation(conversation_id).await?; // table read
                let live = tx.doc(&*LIVE_DOC, conversation_id).await?;

                tx.append_entry(conversation_id, message).await?; // first table write
                live.edit(|live| live.generation = None)?; // document mutation remains valid
                tx.create_task(
                    &follow,
                    json!({}),
                    TaskOptions::conversation(Some(conversation_id)),
                )
                .await?; // further table writes are fine
                let _ = conversation;
                Ok(())
            },
            context,
        )
        .await?;

    // Section 3.4: drafts are revoked after their commit.
    let escaped = app
        .session
        .commit(
            move |tx| async move { tx.doc(&*LIVE_DOC, conversation_id).await },
            context,
        )
        .await?;
    escaped.edit(|live| live.generation = None)?; // fails: the draft was revoked

    let _ = (
        subagent,
        chat,
        payment,
        child_in_tool_commit,
        live_settings,
        spawn,
    );
    container_harness.close(context).await
}

#[test]
fn compiles_the_specs_other_usage_examples() {
    let _ = more_examples;
}
