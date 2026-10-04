//! Port of the tool-related parts of `test/spec-usage.test.ts`: the usage examples of Pi's `docs/spec.md`
//! sections 7.1 and 7.2 (extensions, host setup, tool filters, wraps, plan mode, reload), compile-checked. As in TS,
//! the function is never called; the names the spec leaves to the application are parameters.
//!
//! Not ported: the subagent (7.3), payment task (5.1), table-rule and revoked-draft sequences, which do not use the
//! coding tools and whose patterns the harness suites and examples already exercise.

#![allow(dead_code)]

use std::sync::{Arc, LazyLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::chord::Context;
use crate::durable::documents::{DocToken, define_doc};
use crate::durable::env::ExecutionEnv;
use crate::durable::errors::Result;
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::tool::TOOL_TASK;
use crate::durable::harness::types::{
    AgentChange, BeforeToolDecision, ExtensionDefinition, ExtensionsChange, GenerationHooks,
    HarnessOptions, HarnessSettings, ModelRef, OnYieldHook, ToolHooks, ToolRegistration,
    ToolsChange,
};
use crate::durable::harness::{
    CreateOptions, Harness, create_registry, define_extension, hook, section, wrap_tool,
};
use crate::durable::tools::{BashToolOptions, CODING_TOOLS, create_bash_tool, create_edit_tool};
use crate::durable::types::{LatestConversation, LatestFork, Storage};
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
