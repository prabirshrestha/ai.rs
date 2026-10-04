//! Ports of `test/examples/17`, `18`, `26`-`31` as tests: each example runs against a Harness with the faux model,
//! the coding tools, and a [`LocalExecutionEnv`], and asserts what the TS script prints.
//!
//! Divergences from the TS scripts:
//! - 17 `cat`s a file that does not exist in its temporary directory instead of `/tmp/1gb.txt`, so the error result
//!   does not depend on the machine; the timing hook still sees that call.
//! - 18 always uses the scripted faux model (no `OPENAI_API_KEY` branch) and runs in a temporary directory.
//! - 31 needs `durable-sqlite` for its SQLite file.

use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::text_of;
use crate::chord::{BACKGROUND_CONTEXT, Context};
use crate::durable::documents::{DocToken, define_doc};
use crate::durable::entries::{ASSISTANT_ENTRY, TOOL_RESULT_ENTRY};
use crate::durable::env::ExecutionEnv;
use crate::durable::env::local::LocalExecutionEnv;
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::tool::TOOL_TASK;
use crate::durable::harness::types::{
    AgentChange, ConversationCreateOptions, EnvFactory, ExtensionDefinition, ExtensionsChange,
    GenerationHooks, HarnessOptions, HarnessSettings, ModelRef, RetryPolicyOverrides,
    SubmissionDraft, ToolControl, ToolExecutionMode, ToolExecutionResult, ToolHooks,
    ToolRegistration, ToolsChange,
};
use crate::durable::harness::{
    Conversation, CreateOptions, Harness, create_registry, define_extension, define_tool, hook,
    section, wrap_tool,
};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::storage::test_support::TempDir;
use crate::durable::tools::{BashToolOptions, CODING_TOOLS, create_bash_tool, create_read_tool};
use crate::durable::types::{
    LatestConversation, LatestFork, RewindableConversation, RewindableFork, Storage,
    SubmissionStatus,
};
use crate::models::{Models, create_models};
use crate::providers::faux::{
    FauxMessageOptions, FauxModelDefinition, FauxResponseStep, RegisterFauxProviderOptions,
    faux_assistant_message, faux_provider, faux_tool_call,
};
use crate::types::{Message, StopReason};

fn ctx() -> Context {
    BACKGROUND_CONTEXT.clone()
}

fn model() -> ModelRef {
    ModelRef::new("faux", "faux-1")
}

fn faux_models_with(
    options: RegisterFauxProviderOptions,
    responses: Vec<FauxResponseStep>,
) -> (Models, crate::providers::faux::FauxProviderHandle) {
    let faux = faux_provider(options);
    let models = create_models(Default::default());
    models.set_provider(faux.provider.clone());
    faux.set_responses(responses);
    (models, faux)
}

fn faux_models(
    responses: Vec<FauxResponseStep>,
) -> (Models, crate::providers::faux::FauxProviderHandle) {
    faux_models_with(Default::default(), responses)
}

fn answer(text: &str) -> FauxResponseStep {
    faux_assistant_message(text, FauxMessageOptions::default()).into()
}

fn tool_call(name: &str, args: serde_json::Value, id: Option<&str>) -> FauxResponseStep {
    faux_assistant_message(
        faux_tool_call(name, args, id),
        FauxMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..Default::default()
        },
    )
    .into()
}

fn temp_dir(prefix: &str) -> (TempDir, String) {
    let dir = TempDir::new(prefix);
    let path = dir
        .join("")
        .to_string_lossy()
        .trim_end_matches('/')
        .to_owned();
    (dir, path)
}

/// `env: ({ cwd = fallback }) => new NodeExecutionEnv({ cwd })`.
fn env_following_cwd(fallback: String) -> EnvFactory {
    Arc::new(move |target, _| {
        let env: Arc<dyn ExecutionEnv> = Arc::new(LocalExecutionEnv::at(
            target.cwd.unwrap_or_else(|| fallback.clone()),
        ));
        Box::pin(async move { Ok(Some(env)) })
    })
}

/// `env: () => new NodeExecutionEnv({ cwd })`.
fn env_at(cwd: String) -> EnvFactory {
    Arc::new(move |_, _| {
        let env: Arc<dyn ExecutionEnv> = Arc::new(LocalExecutionEnv::at(cwd.clone()));
        Box::pin(async move { Ok(Some(env)) })
    })
}

fn settings_with(extensions: Vec<crate::durable::harness::types::Extension>) -> HarnessSettings {
    HarnessSettings {
        extensions: Some(extensions),
        ..HarnessSettings::default()
    }
}

async fn submit_and_wait(
    conversation: &Conversation,
    text: &str,
) -> crate::durable::types::SubmissionRecord {
    conversation
        .submit(SubmissionDraft::input(text), &ctx())
        .await
        .unwrap()
        .wait(&ctx())
        .await
        .unwrap()
}

/// The text of the newest tool result in the last `limit` entries.
async fn last_tool_result(conversation: &Conversation, limit: usize) -> String {
    let page = conversation
        .entries(None, None, limit, None, &ctx())
        .await
        .unwrap();
    let entry = page
        .items
        .into_iter()
        .find(|entry| TOOL_RESULT_ENTRY.is(Some(entry)))
        .unwrap();
    text_of(entry.model.unwrap().first()).unwrap_or_default()
}

async fn answer_text(conversation: &Conversation, answer: crate::durable::EntryId) -> String {
    let entry = conversation
        .commit(
            move |tx| async move { tx.entry_of(&ASSISTANT_ENTRY, answer).await },
            &ctx(),
        )
        .await
        .unwrap()
        .unwrap();
    text_of(entry.model.unwrap().first()).unwrap_or_default()
}

async fn tool_names(conversation: &Conversation) -> Vec<String> {
    conversation
        .agent(&ctx())
        .await
        .unwrap()
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect()
}

// ---- 17: coding tools on JSONL storage with a timing hook ----

#[tokio::test]
async fn example_17_coding_tools() {
    let context = ctx();
    let (_dir, directory) = temp_dir("pi-durable-example-");
    std::fs::write(format!("{directory}/notes.txt"), "hello world\n").unwrap();
    let big = format!("{directory}/1gb.txt");

    // The faux provider plays the model: four tool-calling answers, then a final answer.
    let (models, _faux) = faux_models(vec![
        tool_call("read", json!({ "path": "notes.txt" }), Some("r")),
        tool_call(
            "edit",
            json!({ "path": "notes.txt", "edits": [{ "oldText": "world", "newText": "durable" }] }),
            Some("e"),
        ),
        tool_call("bash", json!({ "command": "cat notes.txt" }), Some("b")),
        tool_call(
            "bash",
            json!({ "command": format!("cat {big}") }),
            Some("c"),
        ),
        answer("The file now greets durable."),
    ]);

    // Hooks see every call before and after execution; these time the big `cat`.
    let is_big_cat = |call: &crate::types::ToolCall| {
        call.name == "bash"
            && call.arguments["command"]
                .as_str()
                .is_some_and(|command| command.contains("1gb.txt"))
    };
    let start: Arc<Mutex<Option<Instant>>> = Arc::default();
    let timings: Arc<Mutex<Vec<Duration>>> = Arc::default();
    let timing = define_extension(ExtensionDefinition {
        hooks: vec![hook(
            &TOOL_TASK,
            ToolHooks {
                before_tool: Some(Arc::new({
                    let start = start.clone();
                    move |call, _, _| {
                        if is_big_cat(&call) {
                            *start.lock() = Some(Instant::now());
                        }
                        Box::pin(async { Ok(None) })
                    }
                })),
                after_tool: Some(Arc::new({
                    let (start, timings) = (start.clone(), timings.clone());
                    move |call, _, _, _| {
                        if is_big_cat(&call)
                            && let Some(start) = *start.lock()
                        {
                            timings.lock().push(start.elapsed());
                        }
                        Box::pin(async { Ok(None) })
                    }
                })),
            },
        )],
        ..ExtensionDefinition::new("timing")
    });
    let registry = create_registry();
    // CODING_TOOLS brings read, write, edit, and bash. Tools reach files and processes only through the environment
    // the Harness builds for each call.
    registry.install(CODING_TOOLS.clone()).unwrap();
    registry.install(timing).unwrap();

    // The environment is built per use and follows the conversation's agent `cwd`.
    let storage = crate::durable::storage::jsonl::open_local_jsonl_storage(
        &format!("{directory}/storage"),
        &context,
        Default::default(),
    )
    .await
    .unwrap();
    let mut options = HarnessOptions::new(models, Arc::new(registry));
    options.env = Some(env_following_cwd(directory.clone()));
    let harness = Harness::open(Arc::new(storage), options, &context)
        .await
        .unwrap();
    let root = harness
        .root(
            &context,
            CreateOptions {
                agent: Some(AgentChange::default().model(model()).cwd(directory.clone())),
                init: None,
            },
        )
        .await
        .unwrap();

    // Each tool call runs as a durable pi.tool task owned by the generation.
    let settled = submit_and_wait(&root, "Greet durable instead.").await;
    assert_eq!(settled.status, SubmissionStatus::Done);
    let transcript = root.entries(None, None, 20, None, &context).await.unwrap();
    let mut printed = Vec::new();
    for entry in transcript.items.into_iter().rev() {
        if !TOOL_RESULT_ENTRY.is(Some(&entry)) {
            printed.push(entry.kind);
            continue;
        }
        let Some(Message::ToolResult(result)) = entry.model.unwrap().into_iter().next() else {
            panic!("tool result entry without a tool result");
        };
        let text = text_of(Some(&Message::ToolResult(result.clone()))).unwrap_or_default();
        printed.push(format!(
            "{} {}: {text:?} {}",
            entry.kind, result.tool_name, result.is_error
        ));
    }
    assert_eq!(
        printed,
        [
            "pi.user".to_string(),
            "pi.system".into(),
            "pi.assistant".into(),
            "pi.tool-result read: \"hello world\\n\" false".into(),
            "pi.assistant".into(),
            "pi.tool-result edit: \"Successfully replaced 1 block(s) in notes.txt.\" false".into(),
            "pi.assistant".into(),
            "pi.tool-result bash: \"hello durable\\n\" false".into(),
            "pi.assistant".into(),
            format!("pi.tool-result bash: \"cat: {big}: No such file or directory\\n\" true"),
            "pi.assistant".into(),
        ]
    );
    assert_eq!(
        answer_text(&root, settled.answer.unwrap()).await,
        "The file now greets durable."
    );
    assert_eq!(timings.lock().len(), 1);
    assert_eq!(
        std::fs::read_to_string(format!("{directory}/notes.txt")).unwrap(),
        "hello durable\n"
    );
    harness.close(&context).await.unwrap();
}

// ---- 18: print mode ----

#[tokio::test]
async fn example_18_print() {
    let context = ctx();
    let (_dir, directory) = temp_dir("pi-durable-print-");
    std::fs::write(format!("{directory}/README.md"), "# durable\n").unwrap();
    let (models, _faux) = faux_models(vec![
        tool_call("bash", json!({ "command": "ls" }), Some("call-1")),
        answer("This directory holds the durable package sources, tests, and docs."),
    ]);
    let registry = create_registry();
    registry
        .install(define_extension(ExtensionDefinition {
            tools: vec![
                create_read_tool(),
                create_bash_tool(BashToolOptions::default()),
            ],
            sections: vec![section(
                "preamble",
                |_, _| async { Ok(Some("You are a concise coding assistant.".to_string())) },
                Some(false),
            )],
            ..ExtensionDefinition::new("coding")
        }))
        .unwrap();
    let mut options = HarnessOptions::new(models, Arc::new(registry));
    options.env = Some(env_at(directory.clone()));
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context)
        .await
        .unwrap();
    let root = harness
        .root(
            &context,
            CreateOptions {
                agent: Some(AgentChange::default().model(model())),
                init: None,
            },
        )
        .await
        .unwrap();

    // The host awaits its own Submission, not global idle.
    let settled = submit_and_wait(&root, "What is in this directory?").await;
    assert_eq!(settled.status, SubmissionStatus::Done);
    assert_eq!(
        answer_text(&root, settled.answer.unwrap()).await,
        "This directory holds the durable package sources, tests, and docs."
    );
    assert_eq!(last_tool_result(&root, 10).await, "README.md\n");
    harness.close(&context).await.unwrap();
}

// ---- 26: a small coding agent ----

#[tokio::test]
async fn example_26_coding_agent() {
    let context = ctx();
    let (_dir, workspace) = temp_dir("pi-durable-agent-");
    std::fs::create_dir(format!("{workspace}/app")).unwrap();

    // The app's own prompt, next to the coding tools.
    let coding = define_extension(ExtensionDefinition {
        sections: vec![
            section(
                "preamble",
                |_, _| async {
                    Ok(Some(
                        "You are a coding agent. Use the tools to inspect the project.".to_string(),
                    ))
                },
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
    let registry = create_registry();
    registry.install(CODING_TOOLS.clone()).unwrap();
    registry.install(coding).unwrap();

    // Settings the user edits while the agent runs; every Harness read sees the current values.
    #[derive(Clone, Copy)]
    struct UserSettings {
        parallel_tools: bool,
        max_retries: u32,
    }
    let user_settings = Arc::new(Mutex::new(UserSettings {
        parallel_tools: true,
        max_retries: 3,
    }));
    let seen_modes: Arc<Mutex<Vec<ToolExecutionMode>>> = Arc::default();

    let pwd = || tool_call("bash", json!({ "command": "pwd" }), None);
    let (models, _faux) = faux_models(vec![pwd(), answer("Done."), pwd(), answer("Done.")]);
    let mut options = HarnessOptions::new(models, Arc::new(registry));
    options.settings = Some(Arc::new({
        let user_settings = user_settings.clone();
        let seen_modes = seen_modes.clone();
        move || {
            let current = *user_settings.lock();
            let mode = if current.parallel_tools {
                ToolExecutionMode::Parallel
            } else {
                ToolExecutionMode::Sequential
            };
            seen_modes.lock().push(mode);
            HarnessSettings {
                tool_execution: Some(mode),
                retry: Some(RetryPolicyOverrides {
                    max_retries: Some(current.max_retries),
                    ..RetryPolicyOverrides::default()
                }),
                ..HarnessSettings::default()
            }
        }
    }));
    // The Harness calls this for every tool call and request with the conversation's `cwd`.
    options.env = Some(env_following_cwd(workspace.clone()));
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context)
        .await
        .unwrap();
    let root = harness
        .root(
            &context,
            CreateOptions {
                agent: Some(AgentChange::default().model(model()).cwd(workspace.clone())),
                init: None,
            },
        )
        .await
        .unwrap();

    submit_and_wait(&root, "Where are we?").await;
    assert_eq!(last_tool_result(&root, 10).await.trim(), workspace);

    // The user switches the project directory and turns off parallel tools. Both apply from the next use.
    root.configure(
        AgentChange::default().cwd(format!("{workspace}/app")),
        &context,
    )
    .await
    .unwrap();
    user_settings.lock().parallel_tools = false;
    seen_modes.lock().clear();
    submit_and_wait(&root, "And now?").await;
    assert_eq!(
        last_tool_result(&root, 10).await.trim(),
        format!("{workspace}/app")
    );
    assert!(
        seen_modes
            .lock()
            .iter()
            .all(|mode| *mode == ToolExecutionMode::Sequential)
    );
    harness.close(&context).await.unwrap();
}

// ---- 27: plan mode ----

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
struct Plan {
    steps: Vec<String>,
}

static PLAN_DOC: LazyLock<DocToken<Plan, RewindableConversation>> = LazyLock::new(|| {
    define_doc(crate::durable::types::DocDefinition::new(
        "app.plan",
        1,
        RewindableConversation {
            fork: RewindableFork::AsOf,
        },
        Plan::default,
    ))
    .unwrap()
});

#[tokio::test]
async fn example_27_plan_mode() {
    let context = ctx();

    // ─── Product code: the plan extension ───
    let submit_plan = define_tool(
        "submit_plan",
        "Submit the plan as a list of steps.",
        json!({
            "type": "object",
            "properties": { "steps": { "type": "array", "items": { "type": "string" } } },
            "required": ["steps"],
        }),
        |args, api, call_context| async move {
            let steps: Vec<String> = serde_json::from_value(args["steps"].clone()).unwrap();
            let id = api.conversation_id();
            api.commit(
                move |tx| async move {
                    tx.doc(&*PLAN_DOC, id)
                        .await?
                        .edit(|plan| plan.steps = steps)
                },
                &call_context,
            )
            .await?;
            Ok(ToolExecutionResult {
                control: Some(ToolControl {
                    terminate: Some(true),
                    ..ToolControl::default()
                }),
                ..ToolExecutionResult::text("Plan submitted.")
            })
        },
    );
    let plan = define_extension(ExtensionDefinition {
        tools: vec![submit_plan.clone()],
        sections: vec![section(
            "plan_mode",
            |_, _| async {
                Ok(Some(
                    "You are in plan mode. Read the code, then call submit_plan. Change nothing."
                        .to_string(),
                ))
            },
            None,
        )],
        ..ExtensionDefinition::new("plan")
    });

    // Plan mode is a change to the conversation's agent: select the plan extension and offer only reading and
    // submitting. Clearing both returns to the host's default selection and every tool.
    let read = create_read_tool();
    let enter_plan_mode = AgentChange {
        extensions: Some(Some(ExtensionsChange::Edit {
            add: Some(vec![plan.clone()]),
            remove: None,
        })),
        tools: Some(Some(ToolsChange::List(vec![read, submit_plan]))),
        ..AgentChange::default()
    };
    let leave_plan_mode = AgentChange {
        extensions: Some(None),
        tools: Some(None),
        ..AgentChange::default()
    };

    // ─── Host setup ───
    let (_dir, directory) = temp_dir("pi-durable-plan-");
    std::fs::write(format!("{directory}/server.ts"), "app.listen(3000);\n").unwrap();
    let (models, _faux) = faux_models(vec![
        tool_call("read", json!({ "path": "server.ts" }), None),
        tool_call(
            "submit_plan",
            json!({ "steps": ["Read PORT from the environment", "Default to 3000"] }),
            None,
        ),
        answer("Implementing step 1."),
    ]);
    let registry = create_registry();
    registry.install(CODING_TOOLS.clone()).unwrap();
    registry.install(plan).unwrap();
    let mut options = HarnessOptions::new(models, Arc::new(registry));
    // Installed, but only CODING_TOOLS is selected by default: conversations opt into plan mode.
    options.settings = Some(Arc::new(|| settings_with(vec![CODING_TOOLS.clone()])));
    options.env = Some(env_at(directory.clone()));
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context)
        .await
        .unwrap();
    let root = harness
        .root(
            &context,
            CreateOptions {
                agent: Some(AgentChange::default().model(model())),
                init: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(tool_names(&root).await, ["read", "write", "edit", "bash"]);

    root.configure(enter_plan_mode, &context).await.unwrap();
    assert_eq!(tool_names(&root).await, ["read", "submit_plan"]);
    submit_and_wait(&root, "Make the port configurable.").await;
    assert_eq!(
        harness
            .snapshot(&*PLAN_DOC, root.id, &context)
            .await
            .unwrap()
            .unwrap()
            .steps,
        ["Read PORT from the environment", "Default to 3000"]
    );

    root.configure(leave_plan_mode, &context).await.unwrap();
    assert_eq!(tool_names(&root).await, ["read", "write", "edit", "bash"]);
    submit_and_wait(&root, "Go ahead.").await;

    // The model saw each switch as a system prompt change in its transcript.
    let messages = root.context(&context).await.unwrap().messages;
    let systems: Vec<serde_json::Value> = messages
        .iter()
        .filter_map(|message| match message {
            Message::System(system) => Some(json!({
                // A null section value removes it.
                "sections": system.sections,
                "added": system.tools_added.as_ref().map(|tools| tools.iter().map(|tool| tool.name.clone()).collect::<Vec<_>>()),
                "removed": system.tools_removed.as_ref().map(|tools| tools.iter().map(|tool| tool.name.clone()).collect::<Vec<_>>()),
            })),
            _ => None,
        })
        .collect();
    assert_eq!(
        systems,
        [
            json!({
                "sections": {
                    "plan_mode": "<plan_mode>\nYou are in plan mode. Read the code, then call submit_plan. Change nothing.\n</plan_mode>",
                },
                "added": ["read", "submit_plan"],
                "removed": null,
            }),
            json!({
                "sections": { "plan_mode": null },
                "added": ["write", "edit", "bash"],
                "removed": ["submit_plan"],
            }),
        ]
    );
    harness.close(&context).await.unwrap();
}

// ---- 28: a reviewer agent ----

#[tokio::test]
async fn example_28_reviewer() {
    let context = ctx();
    const DONE: &str = "No further findings.";
    // A role and a review loop: every answer that still has findings gets a second pass.
    let reviewer_extension = define_extension(ExtensionDefinition {
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
                on_yield: Some(Arc::new(|answer, _, _| {
                    let text = text_of(Some(&Message::Assistant(answer))).unwrap_or_default();
                    Box::pin(async move {
                        Ok(if text.contains(DONE) {
                            None
                        } else {
                            Some(
                                format!(
                                    "Look again for anything you missed. Say \"{DONE}\" when there is nothing left."
                                )
                                .into(),
                            )
                        })
                    })
                })),
                ..GenerationHooks::default()
            },
        )],
        ..ExtensionDefinition::new("reviewer")
    });

    // The reviewer works in its own checkout, in practice a `git worktree add`.
    let (_dir, worktree) = temp_dir("pi-durable-review-");
    std::fs::write(
        format!("{worktree}/user.ts"),
        "export const name = (user) => user.name;\n",
    )
    .unwrap();

    let (models, _faux) = faux_models_with(
        RegisterFauxProviderOptions {
            models: vec![
                FauxModelDefinition::new("big"),
                FauxModelDefinition::new("small"),
            ],
            ..Default::default()
        },
        vec![
            tool_call("read", json!({ "path": "user.ts" }), None),
            answer("1. `name` does not handle a missing user."),
            answer(&format!("2. `user` has no type. {DONE}")),
        ],
    );
    let registry = create_registry();
    registry.install(CODING_TOOLS.clone()).unwrap();
    registry.install(reviewer_extension.clone()).unwrap();
    let mut options = HarnessOptions::new(models, Arc::new(registry));
    // The main agent selects only CODING_TOOLS; the reviewer opts in.
    options.settings = Some(Arc::new(|| settings_with(vec![CODING_TOOLS.clone()])));
    options.env = Some(env_following_cwd(
        std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
    ));
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context)
        .await
        .unwrap();
    harness
        .root(
            &context,
            CreateOptions {
                agent: Some(AgentChange::default().model(ModelRef::new("faux", "big"))),
                init: None,
            },
        )
        .await
        .unwrap();

    // Everything the reviewer is, stored on its conversation: the model, exactly these extensions in this order,
    // only the read tool, and its directory. A restart keeps all of it.
    let reviewer = harness
        .create_conversation(
            ConversationCreateOptions {
                agent: Some(
                    AgentChange::default()
                        .model(ModelRef::new("faux", "small"))
                        .extensions(vec![CODING_TOOLS.clone(), reviewer_extension])
                        .tools(vec![create_read_tool()])
                        .cwd(worktree.clone()),
                ),
                ..ConversationCreateOptions::ownerless()
            },
            &context,
        )
        .await
        .unwrap();
    let agent = reviewer.agent(&context).await.unwrap();
    assert_eq!(agent.model.unwrap().model_id, "small");
    assert_eq!(
        agent
            .extensions
            .iter()
            .map(|extension| extension.name.clone())
            .collect::<Vec<_>>(),
        ["coding-tools", "reviewer"]
    );
    assert_eq!(
        agent
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>(),
        ["read"]
    );
    assert_eq!(agent.cwd.as_deref(), Some(worktree.as_str()));

    submit_and_wait(&reviewer, "Review user.ts.").await;
    let page = reviewer
        .entries(None, None, 20, None, &context)
        .await
        .unwrap();
    let mut printed = Vec::new();
    for entry in page.items.into_iter().rev() {
        let is_assistant = ASSISTANT_ENTRY.is(Some(&entry));
        let Some(message) = entry.model.and_then(|model| model.into_iter().next()) else {
            continue;
        };
        match &message {
            Message::User(_) => printed.push(format!("> {}", text_of(Some(&message)).unwrap())),
            Message::Assistant(_) if is_assistant => {
                if let Some(text) = text_of(Some(&message))
                    && !text.is_empty()
                {
                    printed.push(format!("reviewer: {text}"));
                }
            }
            _ => {}
        }
    }
    assert_eq!(
        printed,
        [
            "> Review user.ts.".to_string(),
            "reviewer: 1. `name` does not handle a missing user.".into(),
            format!(
                "> Look again for anything you missed. Say \"{DONE}\" when there is nothing left."
            ),
            format!("reviewer: 2. `user` has no type. {DONE}"),
        ]
    );
    harness.close(&context).await.unwrap();
}

// ---- 29: one sandbox per conversation ----

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
struct Sandbox {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<String>,
}

/// `fork: "initial"`: a fork gets no sandbox until the app assigns one.
static SANDBOX_DOC: LazyLock<DocToken<Sandbox, LatestConversation>> = LazyLock::new(|| {
    define_doc(crate::durable::types::DocDefinition::new(
        "app.sandbox",
        1,
        LatestConversation {
            fork: LatestFork::Initial,
        },
        Sandbox::default,
    ))
    .unwrap()
});

#[tokio::test]
async fn example_29_sandbox_per_conversation() {
    let context = ctx();
    let note = |text: &str| {
        tool_call(
            "write",
            json!({ "path": "note.txt", "content": text }),
            None,
        )
    };
    let (models, _faux) = faux_models(vec![
        note("from alice"),
        answer("Saved."),
        note("from bob"),
        answer("Saved."),
    ]);
    let registry = create_registry();
    registry.install(CODING_TOOLS.clone()).unwrap();
    let mut options = HarnessOptions::new(models, Arc::new(registry));
    // Committed reads only; a conversation without a sandbox gets no environment, so its tools fail cleanly.
    options.env = Some(Arc::new(|target, env_context| {
        Box::pin(async move {
            let sandbox = target
                .read
                .snapshot(&*SANDBOX_DOC, target.conversation_id, &env_context)
                .await?;
            Ok(sandbox.and_then(|sandbox| sandbox.path).map(|path| {
                let env: Arc<dyn ExecutionEnv> = Arc::new(LocalExecutionEnv::at(path));
                env
            }))
        })
    }));
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context)
        .await
        .unwrap();

    // Each user's conversation gets a fresh sandbox in the creating commit.
    let conversation_for = async |user: &str| {
        let (dir, path) = temp_dir(&format!("pi-durable-sandbox-{user}-"));
        let init_path = path.clone();
        let conversation = harness
            .create_conversation(
                ConversationCreateOptions {
                    agent: Some(AgentChange::default().model(model())),
                    init: Some(Arc::new(move |tx, id| {
                        let path = init_path.clone();
                        Box::pin(async move {
                            tx.doc(&*SANDBOX_DOC, id)
                                .await?
                                .edit(|sandbox| sandbox.path = Some(path))
                        })
                    })),
                    ..ConversationCreateOptions::ownerless()
                },
                &context,
            )
            .await
            .unwrap();
        (conversation, dir, path)
    };
    let (alice, _alice_dir, alice_path) = conversation_for("alice").await;
    let (bob, _bob_dir, bob_path) = conversation_for("bob").await;
    submit_and_wait(&alice, "Leave a note.").await;
    submit_and_wait(&bob, "Leave a note.").await;
    assert_eq!(
        std::fs::read_to_string(format!("{alice_path}/note.txt")).unwrap(),
        "from alice"
    );
    assert_eq!(
        std::fs::read_to_string(format!("{bob_path}/note.txt")).unwrap(),
        "from bob"
    );
    harness.close(&context).await.unwrap();
}

// ---- 30: tool override and wrapTool ----

#[tokio::test]
async fn example_30_tool_override() {
    let context = ctx();
    // A bash with the same name: where selected after CODING_TOOLS, it replaces CODING_TOOLS' bash in place.
    let venv = define_extension(ExtensionDefinition {
        tools: vec![create_bash_tool(BashToolOptions {
            command_prefix: Some("source .venv/bin/activate".into()),
            ..BashToolOptions::default()
        })],
        ..ExtensionDefinition::new("venv")
    });

    // Wraps the bash that won, whichever it is. Wrappers never capture a base tool, so reloading either bash keeps
    // it.
    let timings: Arc<Mutex<Vec<Duration>>> = Arc::default();
    let timing = define_extension(ExtensionDefinition {
        wraps: vec![wrap_tool("bash", {
            let timings = timings.clone();
            move |bash| {
                let execute = bash.execute.clone();
                let timings = timings.clone();
                Ok(ToolRegistration {
                    execute: Arc::new(move |args, api, call_context| {
                        let (execute, timings) = (execute.clone(), timings.clone());
                        Box::pin(async move {
                            let start = Instant::now();
                            let result = execute(args, api, call_context).await;
                            timings.lock().push(start.elapsed());
                            result
                        })
                    }),
                    ..bash
                })
            }
        })],
        ..ExtensionDefinition::new("timing")
    });

    let (_dir, project) = temp_dir("pi-durable-venv-");
    std::fs::create_dir_all(format!("{project}/.venv/bin")).unwrap();
    std::fs::write(
        format!("{project}/.venv/bin/activate"),
        format!("export VIRTUAL_ENV=\"{project}/.venv\"\n"),
    )
    .unwrap();

    let probe = || {
        tool_call(
            "bash",
            json!({ "command": "echo venv: $VIRTUAL_ENV" }),
            None,
        )
    };
    let (models, _faux) = faux_models(vec![probe(), answer("Done."), probe(), answer("Done.")]);
    let registry = create_registry();
    registry.install(CODING_TOOLS.clone()).unwrap();
    registry.install(timing.clone()).unwrap();
    registry.install(venv.clone()).unwrap();
    let mut options = HarnessOptions::new(models, Arc::new(registry));
    // Venv is installed but not selected by default.
    options.settings = Some(Arc::new(move || {
        settings_with(vec![CODING_TOOLS.clone(), timing.clone()])
    }));
    options.env = Some(env_at(project.clone()));
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context)
        .await
        .unwrap();
    let plain = harness
        .root(
            &context,
            CreateOptions {
                agent: Some(AgentChange::default().model(model())),
                init: None,
            },
        )
        .await
        .unwrap();
    let python = harness
        .create_conversation(
            ConversationCreateOptions {
                agent: Some(AgentChange {
                    extensions: Some(Some(ExtensionsChange::Edit {
                        add: Some(vec![venv]),
                        remove: None,
                    })),
                    ..AgentChange::default().model(model())
                }),
                ..ConversationCreateOptions::ownerless()
            },
            &context,
        )
        .await
        .unwrap();

    async fn probe_bash(conversation: &Conversation) -> String {
        submit_and_wait(conversation, "Which venv?").await;
        last_tool_result(conversation, 10).await.trim().to_string()
    }
    assert_eq!(probe_bash(&plain).await, "venv:");
    assert_eq!(
        probe_bash(&python).await.replace(&project, "<project>"),
        "venv: <project>/.venv"
    );
    assert_eq!(timings.lock().len(), 2);
    harness.close(&context).await.unwrap();
}

// ---- 31: reload and restart ----

#[cfg(feature = "durable-sqlite")]
#[tokio::test]
async fn example_31_reload_and_restart() {
    use crate::durable::harness::registry::Registry;
    use crate::durable::session::tests::support::Deferred;
    use crate::durable::storage::sqlite::rusqlite::open_local_sqlite_storage;

    let context = ctx();
    // Stand-in for code loaded from disk: each call builds the extension as the file currently reads.
    let started: Arc<Mutex<Deferred>> = Arc::new(Mutex::new(Deferred::default()));
    let gate: Arc<Mutex<Deferred>> = Arc::new(Mutex::new({
        let open = Deferred::default();
        open.resolve();
        open
    }));
    let load_versioned = |version: &'static str| {
        let (started, gate) = (started.clone(), gate.clone());
        define_extension(ExtensionDefinition {
            tools: vec![define_tool(
                "version",
                "Report the tool's code version",
                json!({ "type": "object", "properties": {} }),
                move |_, _, _| {
                    let started = started.lock().clone();
                    let gate = gate.lock().clone();
                    async move {
                        started.resolve();
                        gate.wait().await;
                        Ok(ToolExecutionResult::text(version))
                    }
                },
            )],
            ..ExtensionDefinition::new("versioned")
        })
    };

    let call_version = || tool_call("version", json!({}), None);
    let (models, _faux) = faux_models(vec![
        call_version(),
        answer("Done."),
        call_version(),
        answer("Done."),
    ]);

    let (_dir, directory) = temp_dir("pi-durable-reload-");
    let path = format!("{directory}/session.sqlite");
    let open = async |registry: &Registry| {
        let storage: Arc<dyn Storage> = Arc::new(
            open_local_sqlite_storage(&path, Default::default())
                .await
                .unwrap(),
        );
        Harness::open(
            storage,
            HarnessOptions::new(models.clone(), Arc::new(registry.clone())),
            &context,
        )
        .await
        .unwrap()
    };

    // First process.
    let registry = create_registry();
    registry.install(load_versioned("v1")).unwrap();
    let harness = open(&registry).await;
    let root = harness
        .root(
            &context,
            CreateOptions {
                // Selected by name. The name is what is stored, never the code.
                agent: Some(
                    AgentChange::default()
                        .model(model())
                        .extensions(vec![load_versioned("v1")]),
                ),
                init: None,
            },
        )
        .await
        .unwrap();

    // The file changes while a call runs: the running call finishes on v1, the next call uses v2.
    let release = Deferred::default();
    *gate.lock() = release.clone();
    let running = Deferred::default();
    *started.lock() = running.clone();
    let submission = root
        .submit(SubmissionDraft::input("Which version?"), &context)
        .await
        .unwrap();
    running.wait().await;
    registry.install(load_versioned("v2")).unwrap();
    release.resolve();
    submission.wait(&context).await.unwrap();
    assert_eq!(last_tool_result(&root, 10).await, "v1");
    submit_and_wait(&root, "And now?").await;
    assert_eq!(last_tool_result(&root, 10).await, "v2");
    harness.close(&context).await.unwrap();

    // Second process: the conversation still selects "versioned", but this process has not installed it yet.
    let registry = create_registry();
    let harness = open(&registry).await;
    let root = harness
        .root(&context, CreateOptions::default())
        .await
        .unwrap();
    assert!(tool_names(&root).await.is_empty());
    registry.install(load_versioned("v3")).unwrap();
    assert_eq!(tool_names(&root).await, ["version"]);
    harness.close(&context).await.unwrap();
}
