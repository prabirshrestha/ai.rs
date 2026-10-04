//! Port of durable `src/harness/agent.ts`.

use std::sync::{Arc, LazyLock};

use indexmap::{IndexMap, IndexSet};

use crate::durable::documents::{DocToken, define_doc};
use crate::durable::errors::{Error, Result};
use crate::durable::ids::ConversationId;
use crate::durable::session::Transaction;
use crate::durable::types::{
    ConversationRecord, DocDefinition, RewindableConversation, RewindableFork,
};
use crate::types::ModelThinkingLevel;

use super::define::text_section;
use super::registry::RegistrySnapshot;
use super::types::{
    Agent, AgentChange, AgentState, CompactionPolicy, ConversationRetryPolicy, Extension,
    ExtensionSelection, ExtensionsChange, HarnessSettings, HookRegistration, PromptSection,
    QueueMode, Settings, ToolExecutionMode, ToolFilter, ToolRegistration, ToolsChange, Wrap,
};

pub const DEFAULT_RETRY_POLICY: ConversationRetryPolicy = ConversationRetryPolicy {
    enabled: true,
    max_retries: 3,
    base_delay_ms: 2000,
    max_agent_delay_ms: Some(60000),
};

pub const DEFAULT_COMPACTION_POLICY: CompactionPolicy = CompactionPolicy {
    enabled: true,
    reserve_tokens: 16384,
    keep_recent_tokens: 20000,
    background_tokens: 32768,
};

/// The reserved section key of the agent's `instructions`.
pub const INSTRUCTIONS_KEY: &str = "instructions";

/// Built-in agent document; rewindable so forks start from the agent at their fork entry.
pub static AGENT_DOC: LazyLock<DocToken<AgentState, RewindableConversation>> =
    LazyLock::new(|| {
        define_doc(
            DocDefinition::new(
                "pi.agent",
                1,
                RewindableConversation {
                    fork: RewindableFork::AsOf,
                },
                AgentState::default,
            )
            .checkpoint_when(|_, _, _| Ok(true)),
        )
        .expect("valid pi.agent definition")
    });

/// Resolve the host settings: every field over its built-in default, object fields merged.
pub fn resolve_settings(settings: Option<&HarnessSettings>) -> Settings {
    let retry = settings
        .and_then(|settings| settings.retry)
        .unwrap_or_default();
    let compaction = settings
        .and_then(|settings| settings.compaction)
        .unwrap_or_default();
    Settings {
        extensions: settings.and_then(|settings| settings.extensions.clone()),
        stream: settings
            .and_then(|settings| settings.stream.clone())
            .unwrap_or_default(),
        retry: ConversationRetryPolicy {
            enabled: retry.enabled.unwrap_or(DEFAULT_RETRY_POLICY.enabled),
            max_retries: retry
                .max_retries
                .unwrap_or(DEFAULT_RETRY_POLICY.max_retries),
            base_delay_ms: retry
                .base_delay_ms
                .unwrap_or(DEFAULT_RETRY_POLICY.base_delay_ms),
            max_agent_delay_ms: retry
                .max_agent_delay_ms
                .or(DEFAULT_RETRY_POLICY.max_agent_delay_ms),
        },
        compaction: CompactionPolicy {
            enabled: compaction
                .enabled
                .unwrap_or(DEFAULT_COMPACTION_POLICY.enabled),
            reserve_tokens: compaction
                .reserve_tokens
                .unwrap_or(DEFAULT_COMPACTION_POLICY.reserve_tokens),
            keep_recent_tokens: compaction
                .keep_recent_tokens
                .unwrap_or(DEFAULT_COMPACTION_POLICY.keep_recent_tokens),
            background_tokens: compaction
                .background_tokens
                .unwrap_or(DEFAULT_COMPACTION_POLICY.background_tokens),
        },
        tool_execution: settings
            .and_then(|settings| settings.tool_execution)
            .unwrap_or(ToolExecutionMode::Parallel),
        steering_mode: settings
            .and_then(|settings| settings.steering_mode)
            .unwrap_or(QueueMode::OneAtATime),
        follow_up_mode: settings
            .and_then(|settings| settings.follow_up_mode)
            .unwrap_or(QueueMode::OneAtATime),
    }
}

/// Apply one change to `pi.agent`: a given field replaces the stored one, `Some(None)` clears it, `None` changes
/// nothing.
pub async fn configure(
    tx: &Transaction,
    conversation_id: ConversationId,
    change: &AgentChange,
) -> Result<()> {
    let state = tx.doc(&*AGENT_DOC, conversation_id).await?;
    state.edit(|state| apply_change(state, change))
}

/// `addTools` of a tool round: a list gets each name it lacks appended, `{ remove }` loses the names, and unset tools
/// already offer every tool, so nothing is written.
pub async fn add_tools(
    tx: &Transaction,
    conversation_id: ConversationId,
    added: &[String],
) -> Result<()> {
    let state = tx.doc(&*AGENT_DOC, conversation_id).await?;
    state.edit(|state| match &mut state.tools {
        None => {}
        Some(ToolFilter::List(tools)) => {
            for name in added {
                if !tools.contains(name) {
                    tools.push(name.clone());
                }
            }
        }
        Some(ToolFilter::Remove { remove }) => {
            remove.retain(|name| !added.contains(name));
        }
    })
}

fn names<T>(items: &[T], name: impl Fn(&T) -> &str) -> Vec<String> {
    items.iter().map(|item| name(item).to_string()).collect()
}

fn apply_change(state: &mut AgentState, change: &AgentChange) {
    fn set<T: Clone>(slot: &mut Option<T>, value: &Option<Option<T>>) {
        if let Some(value) = value {
            *slot = value.clone();
        }
    }
    set(&mut state.model, &change.model);
    set(&mut state.thinking_level, &change.thinking_level);
    if let Some(extensions) = &change.extensions {
        state.extensions = extensions.as_ref().map(|extensions| match extensions {
            ExtensionsChange::List(list) => {
                ExtensionSelection::List(names(list, |extension| &extension.name))
            }
            ExtensionsChange::Edit { add, remove } => ExtensionSelection::Edit {
                add: add
                    .as_ref()
                    .map(|add| names(add, |extension| &extension.name)),
                remove: remove
                    .as_ref()
                    .map(|remove| names(remove, |extension| &extension.name)),
            },
        });
    }
    if let Some(tools) = &change.tools {
        state.tools = tools.as_ref().map(|tools| match tools {
            ToolsChange::List(list) => ToolFilter::List(names(list, |tool| &tool.name)),
            ToolsChange::Remove(remove) => ToolFilter::Remove {
                remove: names(remove, |tool| &tool.name),
            },
        });
    }
    set(&mut state.instructions, &change.instructions);
    set(&mut state.cwd, &change.cwd);
}

/// Built-in part of every Harness commit that creates or forks a conversation, for `pi.agent`: a fork keeps its `asOf`
/// copy; a new task-owned conversation copies the stored agent of its owner task's conversation; a new ownerless one
/// starts empty.
pub async fn create_agent(tx: &Transaction, conversation: &ConversationRecord) -> Result<()> {
    if conversation.parent.is_some() {
        return Ok(());
    }
    let agent = tx.doc(&*AGENT_DOC, conversation.id).await?;
    let Some(owner) = conversation.owner else {
        return Ok(());
    };
    let owner = tx.doc(&*AGENT_DOC, owner.conversation_id).await?;
    agent.set(owner.get()?)
}

/// Handlers of the selected extensions' hooks for a task name, in extension order.
pub fn agent_hooks(agent: &Agent, task_name: &str) -> Vec<HookRegistration> {
    agent
        .extensions
        .iter()
        .flat_map(|extension| extension.hooks.iter())
        .filter(|hook| hook.task == task_name)
        .cloned()
        .collect()
}

/// Resolve an agent from its stored state (absent: every field unset), a registry snapshot, and resolved settings. A
/// wrapper that fails or renames drops its target and is reported; a wrapper without a target does nothing.
pub fn resolve_agent(
    state: Option<&AgentState>,
    snapshot: &RegistrySnapshot,
    settings: &Settings,
    report: &dyn Fn(Error),
) -> Agent {
    let extensions = select_extensions(
        state.and_then(|state| state.extensions.as_ref()),
        snapshot,
        settings,
    );

    let mut composed: IndexMap<String, ToolRegistration> = IndexMap::new();
    for extension in &extensions {
        for tool in &extension.tools {
            composed.insert(tool.name.clone(), tool.clone());
        }
    }
    let mut sections: IndexMap<String, PromptSection> = IndexMap::new();
    for extension in &extensions {
        for section in &extension.sections {
            sections.insert(section.key.clone(), section.clone());
        }
    }
    for extension in &extensions {
        for wrap in &extension.wraps {
            match wrap {
                Wrap::Tool { tool, wrap } => apply_wrap(
                    &mut composed,
                    tool,
                    |item| wrap(item),
                    |item| &item.name,
                    report,
                ),
                Wrap::Section { section, wrap } => apply_wrap(
                    &mut sections,
                    section,
                    |item| wrap(item),
                    |item| &item.key,
                    report,
                ),
            }
        }
    }

    let tools: Vec<ToolRegistration> = match state.and_then(|state| state.tools.as_ref()) {
        None => composed.values().cloned().collect(),
        Some(ToolFilter::List(filter)) => filter
            .iter()
            .collect::<IndexSet<_>>()
            .into_iter()
            .filter_map(|name| composed.get(name).cloned())
            .collect(),
        Some(ToolFilter::Remove { remove }) => composed
            .values()
            .filter(|tool| !remove.contains(&tool.name))
            .cloned()
            .collect(),
    };

    let instructions = state.and_then(|state| state.instructions.clone());
    let mut agent_sections: Vec<PromptSection> = sections.into_values().collect();
    if let Some(instructions) = &instructions {
        agent_sections.push(text_section(INSTRUCTIONS_KEY, instructions.clone()));
    }

    Agent {
        model: state.and_then(|state| state.model.clone()),
        thinking_level: state
            .and_then(|state| state.thinking_level)
            .unwrap_or(ModelThinkingLevel::Off),
        extensions,
        tools,
        sections: agent_sections,
        instructions,
        cwd: state.and_then(|state| state.cwd.clone()),
    }
}

/// Selected installed extensions: the stored list, or the default selection edited by `{ add, remove }`.
fn select_extensions(
    stored: Option<&ExtensionSelection>,
    snapshot: &RegistrySnapshot,
    settings: &Settings,
) -> Vec<Extension> {
    let selected: Vec<String> = match stored {
        Some(ExtensionSelection::List(list)) => list.clone(),
        edit => {
            let base: Vec<String> = match &settings.extensions {
                Some(extensions) => names(extensions, |extension| &extension.name),
                None => names(snapshot.installed(), |extension| &extension.name),
            };
            let (add, remove) = match edit {
                Some(ExtensionSelection::Edit { add, remove }) => (add.clone(), remove.clone()),
                _ => (None, None),
            };
            let removed = remove.unwrap_or_default();
            base.into_iter()
                .chain(add.unwrap_or_default())
                .filter(|name| !removed.contains(name))
                .collect()
        }
    };
    selected
        .iter()
        .collect::<IndexSet<_>>()
        .into_iter()
        .filter_map(|name| snapshot.extension(name).cloned())
        .collect()
}

fn apply_wrap<T: Clone>(
    items: &mut IndexMap<String, T>,
    target: &str,
    wrap: impl Fn(T) -> Result<T>,
    name_of: impl Fn(&T) -> &String,
    report: &dyn Fn(Error),
) {
    let Some(item) = items.get(target).cloned() else {
        return;
    };
    let wrapped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| wrap(item)))
        .unwrap_or_else(|panic| Err(super::util::panic_error(panic)))
        .and_then(|wrapped| {
            if name_of(&wrapped) != target {
                return Err(Error::message(format!(
                    "Wrapper renamed {target} to {}",
                    name_of(&wrapped)
                )));
            }
            Ok(wrapped)
        });
    match wrapped {
        Ok(wrapped) => {
            items.insert(target.to_string(), wrapped);
        }
        Err(error) => {
            items.shift_remove(target);
            report(error);
        }
    }
}

/// A hook handlers struct of type `H`, when `registration` holds one.
pub fn hook_handlers<H: Send + Sync + 'static>(registration: &HookRegistration) -> Option<Arc<H>> {
    registration.handlers.clone().downcast::<H>().ok()
}
