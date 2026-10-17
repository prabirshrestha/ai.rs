//! Port of `test/harness-registry.test.ts`.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;

use super::support::*;
use crate::durable::errors::Error;
use crate::durable::harness::agent::{agent_hooks, resolve_agent, resolve_settings};
use crate::durable::harness::compaction::COMPACTION_TASK;
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::registry::{RegistryReader, RegistrySnapshot};
use crate::durable::harness::tool::TOOL_TASK;
use crate::durable::harness::types::{
    Agent, AgentState, Extension, ExtensionDefinition, ExtensionSelection, GenerationHooks,
    HarnessSettings, ModelRef, PromptInput, ToolFilter, ToolHooks, ToolRegistration,
};
use crate::durable::harness::{
    DocumentReader, create_registry, define_extension, hook, section, text_section, wrap_section,
    wrap_tool,
};
use crate::durable::ids::ConversationId;
use crate::durable::tasks::{AnyTask, TaskDefinition, define_task};
use crate::types::ModelThinkingLevel;

fn task(name: &str, version: u32) -> AnyTask {
    define_task(
        TaskDefinition::<(), serde_json::Value, (), ()>::new(
            name,
            version,
            |_| json!({ "phase": "run" }),
        )
        .phase("run", |_, _, _| async { Ok(()) }),
    )
    .any()
}

fn names(items: &[Extension]) -> Vec<String> {
    items.iter().map(|item| item.name.clone()).collect()
}

fn tool_names(items: &[ToolRegistration]) -> Vec<String> {
    items.iter().map(|item| item.name.clone()).collect()
}

fn ext(definition: ExtensionDefinition) -> Extension {
    define_extension(definition)
}

fn resolve(
    state: Option<AgentState>,
    snapshot: &RegistrySnapshot,
    settings: HarnessSettings,
    reports: &Mutex<Vec<Error>>,
) -> Agent {
    resolve_agent(
        state.as_ref(),
        snapshot,
        &resolve_settings(Some(&settings)),
        &|error| reports.lock().push(error),
    )
}

fn resolve_quiet(state: Option<AgentState>, snapshot: &RegistrySnapshot) -> Agent {
    resolve(
        state,
        snapshot,
        HarnessSettings::default(),
        &Mutex::default(),
    )
}

async fn rendered(agent: &Agent) -> Vec<(String, Option<String>)> {
    let reader = DocumentReader(Arc::new(crate::durable::session::SessionImpl::new(
        Arc::new(crate::durable::storage::memory::MemoryStorage::new()),
    )));
    let input = PromptInput {
        conversation_id: ConversationId(1),
        agent: agent.clone(),
        env: None,
        shown: Default::default(),
        read: reader,
    };
    let mut out = Vec::new();
    for item in &agent.sections {
        let text = (item.render)(input.clone(), context())
            .await
            .expect("render");
        out.push((item.key.clone(), text));
    }
    out
}

fn tool_with_description(name: &str, description: &str) -> ToolRegistration {
    tool_described(name, description)
}

#[test]
fn installs_replaces_in_place_and_uninstalls_extensions_by_name() {
    let registry = create_registry();
    let listener = Arc::new(Mutex::new(Vec::<String>::new()));
    {
        let listener = listener.clone();
        let reader = registry.clone();
        let _ = registry.subscribe(Arc::new(move || {
            listener
                .lock()
                .push(names(reader.snapshot().installed()).join(","))
        }));
    }
    let a = ext(ExtensionDefinition {
        tools: vec![tool_with_description("read", "Read files")],
        ..ExtensionDefinition::new("a")
    });
    let b = ext(ExtensionDefinition {
        tools: vec![tool("read"), tool("bash")],
        ..ExtensionDefinition::new("b")
    });
    registry.install(a.clone()).unwrap();
    registry.install(b).unwrap();
    let before = registry.snapshot();
    // A new object with an installed name replaces it at its position.
    let a2 = ext(ExtensionDefinition {
        tools: vec![tool("grep")],
        ..ExtensionDefinition::new("a")
    });
    registry.install(a2.clone()).unwrap();
    assert_eq!(names(registry.snapshot().installed()), ["a", "b"]);
    assert!(registry.snapshot().extension("a").unwrap().ptr_eq(&a2));
    assert_eq!(
        registry
            .snapshot()
            .tools()
            .iter()
            .map(|(extension, tool)| format!("{}@{}", tool.name, extension.name))
            .collect::<Vec<_>>(),
        ["grep@a", "read@b", "bash@b"]
    );
    // Old snapshots stay as they were.
    assert!(before.extension("a").unwrap().ptr_eq(&a));
    assert_eq!(before.tools()[0].1.description, "Read files");
    // Uninstall matches the name, whichever object; a later install appends.
    registry.uninstall(&a).unwrap();
    registry.uninstall(&a).unwrap();
    assert_eq!(names(registry.snapshot().installed()), ["b"]);
    registry.install(a).unwrap();
    assert_eq!(names(registry.snapshot().installed()), ["b", "a"]);
    assert_eq!(*listener.lock(), ["a", "a,b", "a,b", "b", "b,a"]);
}

#[test]
fn validates_the_registry_as_it_would_be_after_an_install_and_publishes_nothing_when_invalid() {
    let registry = create_registry();
    registry
        .install(ext(ExtensionDefinition {
            tasks: vec![task("app.index", 1)],
            ..ExtensionDefinition::new("tasks")
        }))
        .unwrap();
    let published = Arc::new(Mutex::new(0));
    {
        let published = published.clone();
        let _ = registry.subscribe(Arc::new(move || *published.lock() += 1));
    }
    let before = registry.snapshot();
    let invalid = [
        (
            ExtensionDefinition {
                tools: vec![tool("read"), tool("read")],
                ..ExtensionDefinition::new("x")
            },
            "two tools named read",
        ),
        (
            ExtensionDefinition {
                sections: vec![text_section("a", "1"), text_section("a", "2")],
                ..ExtensionDefinition::new("x")
            },
            "two sections",
        ),
        (
            ExtensionDefinition {
                sections: vec![text_section("Bad Key", "")],
                ..ExtensionDefinition::new("x")
            },
            "must match",
        ),
        (
            ExtensionDefinition {
                sections: vec![text_section("instructions", "")],
                ..ExtensionDefinition::new("x")
            },
            "reserved",
        ),
        (
            ExtensionDefinition {
                tasks: vec![task("pi.generation", 1)],
                ..ExtensionDefinition::new("x")
            },
            "already installed",
        ),
        (
            ExtensionDefinition {
                tasks: vec![task("app.index", 1)],
                ..ExtensionDefinition::new("x")
            },
            "already installed",
        ),
    ];
    for (extension, message) in invalid {
        assert_err(registry.install(ext(extension)), message);
    }
    assert!(Arc::ptr_eq(&registry.snapshot(), &before));
    assert_eq!(*published.lock(), 0);
    // Replacing the extension that holds a task name is valid: the check runs on the state after replacement.
    registry
        .install(ext(ExtensionDefinition {
            tasks: vec![task("app.index", 2)],
            ..ExtensionDefinition::new("tasks")
        }))
        .unwrap();
    assert_eq!(
        registry
            .snapshot()
            .task("app.index")
            .unwrap()
            .definition()
            .version,
        2
    );
}

#[test]
fn always_holds_the_built_in_tasks_which_are_not_an_extension() {
    let registry = create_registry();
    assert!(registry.snapshot().installed().is_empty());
    let builtins = [
        GENERATION_TASK.any(),
        TOOL_TASK.any(),
        COMPACTION_TASK.any(),
    ];
    assert_eq!(registry.snapshot().tasks(), builtins);
    let custom = task("app.custom", 1);
    registry
        .install(ext(ExtensionDefinition {
            tasks: vec![custom.clone()],
            ..ExtensionDefinition::new("custom")
        }))
        .unwrap();
    let mut expected = builtins.to_vec();
    expected.push(custom.clone());
    assert_eq!(registry.snapshot().tasks(), expected);
    assert!(
        registry
            .snapshot()
            .task("app.custom")
            .unwrap()
            .ptr_eq(&custom)
    );
    registry
        .uninstall(&ext(ExtensionDefinition::new("custom")))
        .unwrap();
    assert!(registry.snapshot().task("app.custom").is_none());
}

struct Fixture {
    coding: Extension,
    skills: Extension,
    snapshot: RegistrySnapshot,
}

fn fixture() -> Fixture {
    let coding = ext(ExtensionDefinition {
        tools: vec![tool("read"), tool("bash"), tool("edit")],
        sections: vec![
            section(
                "preamble",
                |_, _| async { Ok(Some("You code.".to_string())) },
                Some(false),
            ),
            text_section("cwd", "/repo"),
        ],
        ..ExtensionDefinition::new("coding")
    });
    let skills = ext(ExtensionDefinition {
        sections: vec![text_section("skills", "S")],
        ..ExtensionDefinition::new("skills")
    });
    let reviewer = ext(ExtensionDefinition {
        sections: vec![text_section("role", "Review.")],
        ..ExtensionDefinition::new("reviewer")
    });
    let registry = create_registry();
    for extension in [&coding, &skills, &reviewer] {
        registry.install(extension.clone()).unwrap();
    }
    Fixture {
        coding,
        skills,
        snapshot: registry.snapshot(),
    }
}

fn state_extensions(selection: ExtensionSelection) -> Option<AgentState> {
    Some(AgentState {
        extensions: Some(selection),
        ..AgentState::default()
    })
}

#[test]
fn selects_the_default_a_list_or_the_default_edited_by_add_and_remove() {
    let Fixture {
        coding,
        skills,
        snapshot,
    } = fixture();
    let quiet = Mutex::default();
    assert_eq!(
        names(&resolve_quiet(None, &snapshot).extensions),
        ["coding", "skills", "reviewer"]
    );
    let settings = HarnessSettings {
        extensions: Some(vec![coding.clone(), skills.clone()]),
        ..HarnessSettings::default()
    };
    assert_eq!(
        names(
            &resolve(
                Some(AgentState::default()),
                &snapshot,
                settings.clone(),
                &quiet
            )
            .extensions
        ),
        ["coding", "skills"]
    );
    assert_eq!(
        names(
            &resolve(
                state_extensions(ExtensionSelection::List(vec![
                    "reviewer".into(),
                    "coding".into()
                ])),
                &snapshot,
                settings.clone(),
                &quiet
            )
            .extensions
        ),
        ["reviewer", "coding"]
    );
    // Add appends, remove drops, duplicates keep their first position, uninstalled names are skipped.
    let edited = state_extensions(ExtensionSelection::Edit {
        add: Some(vec!["reviewer".into(), "coding".into(), "gone".into()]),
        remove: Some(vec!["skills".into()]),
    });
    assert_eq!(
        names(&resolve(edited, &snapshot, settings, &quiet).extensions),
        ["coding", "reviewer"]
    );
    // An old object stands for its name: the installed extension is selected.
    let stale = HarnessSettings {
        extensions: Some(vec![
            ext(ExtensionDefinition::new("skills")),
            skills.clone(),
        ]),
        ..HarnessSettings::default()
    };
    let resolved = resolve(Some(AgentState::default()), &snapshot, stale, &quiet).extensions;
    assert_eq!(resolved.len(), 1);
    assert!(resolved[0].ptr_eq(&skills));
}

#[test]
fn skips_uninstalled_names_and_resolves_them_again_once_they_are_installed() {
    let Fixture { coding, skills, .. } = fixture();
    let local = create_registry();
    local.install(coding).unwrap();
    let state = state_extensions(ExtensionSelection::List(vec![
        "coding".into(),
        "skills".into(),
    ]));
    assert_eq!(
        names(&resolve_quiet(state.clone(), &local.snapshot()).extensions),
        ["coding"]
    );
    local.install(skills).unwrap();
    assert_eq!(
        names(&resolve_quiet(state, &local.snapshot()).extensions),
        ["coding", "skills"]
    );
}

#[test]
fn replaces_same_name_tools_in_place_wraps_the_winner_then_applies_the_filter() {
    let Fixture { coding, .. } = fixture();
    let local = create_registry();
    let calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let venv = ext(ExtensionDefinition {
        tools: vec![tool_with_description("bash", "venv bash")],
        ..ExtensionDefinition::new("venv")
    });
    let grep_calls = calls.clone();
    let timing = ext(ExtensionDefinition {
        wraps: vec![
            wrap_tool("bash", |mut inner| {
                inner.description = format!("{} (timed)", inner.description);
                Ok(inner)
            }),
            wrap_tool("bash", |mut inner| {
                inner.description = format!("{} [2]", inner.description);
                Ok(inner)
            }),
            // No `grep` is selected: the wrapper does nothing and reports nothing.
            wrap_tool("grep", move |_| {
                grep_calls.lock().push("grep".into());
                Ok(tool("grep"))
            }),
        ],
        ..ExtensionDefinition::new("timing")
    });
    for extension in [coding, venv, timing] {
        local.install(extension).unwrap();
    }
    let reports = Mutex::default();
    let agent = resolve(
        None,
        &local.snapshot(),
        HarnessSettings::default(),
        &reports,
    );
    assert_eq!(
        agent
            .tools
            .iter()
            .map(|each| (each.name.clone(), each.description.clone()))
            .collect::<Vec<_>>(),
        [
            ("read".to_string(), "read tool".to_string()),
            ("bash".to_string(), "venv bash (timed) [2]".to_string()),
            ("edit".to_string(), "edit tool".to_string()),
        ]
    );
    assert!(reports.lock().is_empty());
    assert!(calls.lock().is_empty());

    // A list keeps exactly these names in its order, a repeated name at its first position.
    let filtered = resolve_quiet(
        Some(AgentState {
            tools: Some(ToolFilter::List(vec![
                "edit".into(),
                "missing".into(),
                "read".into(),
                "edit".into(),
            ])),
            ..AgentState::default()
        }),
        &local.snapshot(),
    );
    assert_eq!(tool_names(&filtered.tools), ["edit", "read"]);
    let removed = resolve_quiet(
        Some(AgentState {
            tools: Some(ToolFilter::Remove {
                remove: vec!["bash".into()],
            }),
            ..AgentState::default()
        }),
        &local.snapshot(),
    );
    assert_eq!(tool_names(&removed.tools), ["read", "edit"]);
}

#[test]
fn drops_a_tool_or_section_whose_wrapper_fails_or_renames_it_and_reports_the_failure() {
    let Fixture { coding, .. } = fixture();
    let local = create_registry();
    let broken = ext(ExtensionDefinition {
        wraps: vec![
            wrap_tool("read", |_| Err(Error::message("wrapper failed"))),
            wrap_tool("edit", |mut inner| {
                inner.name = "renamed".into();
                Ok(inner)
            }),
            wrap_section("cwd", |_| Err(Error::message("section wrapper failed"))),
        ],
        ..ExtensionDefinition::new("broken")
    });
    local.install(coding).unwrap();
    local.install(broken).unwrap();
    let reports = Mutex::default();
    let agent = resolve(
        None,
        &local.snapshot(),
        HarnessSettings::default(),
        &reports,
    );
    assert_eq!(tool_names(&agent.tools), ["bash"]);
    assert_eq!(
        agent
            .sections
            .iter()
            .map(|each| each.key.clone())
            .collect::<Vec<_>>(),
        ["preamble"]
    );
    assert_eq!(
        reports
            .lock()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        [
            "wrapper failed",
            "Wrapper renamed edit to renamed",
            "section wrapper failed"
        ]
    );
}

#[tokio::test]
async fn orders_sections_by_extension_replaces_same_keys_in_place_and_renders_instructions_last_and_unwrapped()
 {
    let Fixture { coding, skills, .. } = fixture();
    let local = create_registry();
    let over = ext(ExtensionDefinition {
        sections: vec![section(
            "preamble",
            |_, _| async { Ok(Some("You review.".to_string())) },
            Some(false),
        )],
        wraps: vec![
            wrap_section("cwd", |mut inner| {
                let render = inner.render.clone();
                inner.render = Arc::new(move |input, context| {
                    let render = render.clone();
                    Box::pin(async move {
                        Ok(render(input, context).await?.map(|text| format!("{text}!")))
                    })
                });
                Ok(inner)
            }),
            // Instructions are not wrapped.
            wrap_section("instructions", |_| Err(Error::message("never"))),
        ],
        ..ExtensionDefinition::new("override")
    });
    for extension in [coding, skills, over] {
        local.install(extension).unwrap();
    }
    let reports = Mutex::default();
    let agent = resolve(
        Some(AgentState {
            instructions: Some("Be terse.".into()),
            ..AgentState::default()
        }),
        &local.snapshot(),
        HarnessSettings::default(),
        &reports,
    );
    let pairs = |items: &[(&str, &str)]| -> Vec<(String, Option<String>)> {
        items
            .iter()
            .map(|(key, value)| (key.to_string(), Some(value.to_string())))
            .collect()
    };
    assert_eq!(
        rendered(&agent).await,
        pairs(&[
            ("preamble", "You review."),
            ("cwd", "/repo!"),
            ("skills", "S"),
            ("instructions", "Be terse.")
        ])
    );
    assert_eq!(agent.sections.last().unwrap().tag, None);
    assert!(reports.lock().is_empty());
}

#[test]
fn collects_hooks_of_the_selected_extensions_in_extension_order_and_applies_field_defaults() {
    let local = create_registry();
    let first = hook(&TOOL_TASK, ToolHooks::default());
    let second = hook(&TOOL_TASK, ToolHooks::default());
    let on_yield = hook(&GENERATION_TASK, GenerationHooks::default());
    local
        .install(ext(ExtensionDefinition {
            hooks: vec![first.clone(), on_yield],
            ..ExtensionDefinition::new("a")
        }))
        .unwrap();
    local
        .install(ext(ExtensionDefinition {
            hooks: vec![second.clone()],
            ..ExtensionDefinition::new("b")
        }))
        .unwrap();
    let same = |left: &[crate::durable::harness::HookRegistration],
                right: &[&crate::durable::harness::HookRegistration]| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| Arc::ptr_eq(&left.handlers, &right.handlers))
    };
    assert!(same(
        &agent_hooks(&resolve_quiet(None, &local.snapshot()), "pi.tool"),
        &[&first, &second]
    ));
    let reversed = state_extensions(ExtensionSelection::List(vec!["b".into(), "a".into()]));
    assert!(same(
        &agent_hooks(&resolve_quiet(reversed, &local.snapshot()), "pi.tool"),
        &[&second, &first]
    ));
    let only_b = state_extensions(ExtensionSelection::List(vec!["b".into()]));
    assert!(agent_hooks(&resolve_quiet(only_b, &local.snapshot()), "pi.generation").is_empty());

    let defaults = resolve_quiet(None, &local.snapshot());
    assert_eq!(defaults.model, None);
    assert_eq!(defaults.thinking_level, ModelThinkingLevel::Off);
    assert_eq!(defaults.cwd, None);
    let configured = resolve_quiet(
        Some(AgentState {
            model: Some(ModelRef::new("p", "m")),
            thinking_level: Some(ModelThinkingLevel::High),
            cwd: Some("/w".into()),
            ..AgentState::default()
        }),
        &local.snapshot(),
    );
    assert_eq!(configured.model, Some(ModelRef::new("p", "m")));
    assert_eq!(configured.thinking_level, ModelThinkingLevel::High);
    assert_eq!(configured.cwd.as_deref(), Some("/w"));
}
