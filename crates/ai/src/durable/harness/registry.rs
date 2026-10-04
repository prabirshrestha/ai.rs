//! Port of durable `src/harness/registry.ts`.

use std::fmt;
use std::sync::{Arc, LazyLock};

use indexmap::{IndexMap, IndexSet};
use parking_lot::Mutex;
use regex::Regex;

use crate::durable::errors::{Error, Result};
use crate::durable::session::Unsubscribe;
use crate::durable::tasks::AnyTask;

use super::agent::INSTRUCTIONS_KEY;
use super::compaction::COMPACTION_TASK;
use super::generation::GENERATION_TASK;
use super::tool::TOOL_TASK;
use super::types::{Extension, PromptSection, ToolRegistration};

static SECTION_KEY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("^[a-z][a-z0-9_-]*$").expect("valid regex"));

/// Built-in task definitions every registry holds; they are not an extension and cannot be removed or replaced.
pub fn builtin_tasks() -> Vec<AnyTask> {
    vec![
        GENERATION_TASK.any(),
        TOOL_TASK.any(),
        COMPACTION_TASK.any(),
    ]
}

/// Immutable published registry state (`RegistrySnapshot`).
pub struct RegistryState {
    extensions: Vec<Extension>,
    by_name: IndexMap<String, Extension>,
    tasks: IndexMap<String, AnyTask>,
}

/// A registry snapshot; compares by identity.
pub type RegistrySnapshot = Arc<RegistryState>;

impl RegistryState {
    fn new(extensions: Vec<Extension>) -> Result<Self> {
        let by_name = extensions
            .iter()
            .map(|extension| (extension.name.clone(), extension.clone()))
            .collect();
        let mut tasks = IndexMap::new();
        for task in builtin_tasks() {
            tasks.insert(task.name().to_string(), task);
        }
        for extension in &extensions {
            for task in &extension.tasks {
                let name = task.name();
                if tasks.contains_key(name) {
                    return Err(Error::message(format!(
                        "Task {name} of extension {} is already installed",
                        extension.name
                    )));
                }
                tasks.insert(name.to_string(), task.clone());
            }
        }
        Ok(Self {
            extensions,
            by_name,
            tasks,
        })
    }

    /// Installed extensions, in install order.
    pub fn installed(&self) -> &[Extension] {
        &self.extensions
    }

    pub fn extension(&self, name: &str) -> Option<&Extension> {
        self.by_name.get(name)
    }

    /// Every installed tool with its extension, in install order.
    pub fn tools(&self) -> Vec<(Extension, ToolRegistration)> {
        self.extensions
            .iter()
            .flat_map(|extension| {
                extension
                    .tools
                    .iter()
                    .map(move |tool| (extension.clone(), tool.clone()))
            })
            .collect()
    }

    /// Every installed section with its extension, in install order.
    pub fn sections(&self) -> Vec<(Extension, PromptSection)> {
        self.extensions
            .iter()
            .flat_map(|extension| {
                extension
                    .sections
                    .iter()
                    .map(move |section| (extension.clone(), section.clone()))
            })
            .collect()
    }

    /// The built-in tasks, then every extension task.
    pub fn tasks(&self) -> Vec<AnyTask> {
        self.tasks.values().cloned().collect()
    }

    pub fn task(&self, name: &str) -> Option<&AnyTask> {
        self.tasks.get(name)
    }
}

impl fmt::Debug for RegistryState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegistrySnapshot")
            .field("extensions", &self.extensions)
            .finish_non_exhaustive()
    }
}

/// Read side of a registry the Harness subscribes to.
pub trait RegistryReader: Send + Sync {
    /// The current immutable state.
    fn snapshot(&self) -> RegistrySnapshot;
    /// Called synchronously after each published change. Must not panic.
    fn subscribe(&self, listener: Arc<dyn Fn() + Send + Sync>) -> Unsubscribe;
}

/// Application-owned, mutable set of installed extensions.
#[derive(Clone)]
pub struct Registry(Arc<RegistryInner>);

struct RegistryInner {
    current: Mutex<RegistrySnapshot>,
    listeners: Mutex<IndexMap<u64, Arc<dyn Fn() + Send + Sync>>>,
    next: Mutex<u64>,
}

impl Registry {
    /// Install or replace the extension with the same name. Validates before publishing.
    pub fn install(&self, extension: Extension) -> Result<()> {
        validate_extension(&extension)?;
        let current = self.snapshot();
        let installed = current.installed();
        let next = match installed
            .iter()
            .position(|other| other.name == extension.name)
        {
            None => {
                let mut next = installed.to_vec();
                next.push(extension);
                next
            }
            Some(index) => {
                let mut next = installed.to_vec();
                next[index] = extension;
                next
            }
        };
        self.publish(next)
    }

    /// Remove the installed extension with the same name, if any.
    pub fn uninstall(&self, extension: &Extension) -> Result<()> {
        let current = self.snapshot();
        let installed = current.installed();
        if !installed.iter().any(|other| other.name == extension.name) {
            return Ok(());
        }
        self.publish(
            installed
                .iter()
                .filter(|other| other.name != extension.name)
                .cloned()
                .collect(),
        )
    }

    /// Build and validate the next state, which fails on a task name collision, then publish it synchronously.
    fn publish(&self, extensions: Vec<Extension>) -> Result<()> {
        let state = Arc::new(RegistryState::new(extensions)?);
        *self.0.current.lock() = state;
        let listeners: Vec<_> = self.0.listeners.lock().values().cloned().collect();
        for listener in listeners {
            listener();
        }
        Ok(())
    }
}

impl RegistryReader for Registry {
    fn snapshot(&self) -> RegistrySnapshot {
        self.0.current.lock().clone()
    }

    fn subscribe(&self, listener: Arc<dyn Fn() + Send + Sync>) -> Unsubscribe {
        let id = {
            let mut next = self.0.next.lock();
            *next += 1;
            *next
        };
        self.0.listeners.lock().insert(id, listener);
        let inner = Arc::downgrade(&self.0);
        Box::new(move || {
            if let Some(inner) = inner.upgrade() {
                inner.listeners.lock().shift_remove(&id);
            }
        })
    }
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Registry").field(&self.snapshot()).finish()
    }
}

/// Unique tool names and section keys within one extension; valid, unreserved section keys.
fn validate_extension(extension: &Extension) -> Result<()> {
    let mut tools = IndexSet::new();
    for tool in &extension.tools {
        if !tools.insert(tool.name.as_str()) {
            return Err(Error::message(format!(
                "Extension {} has two tools named {}",
                extension.name, tool.name
            )));
        }
    }
    let mut sections = IndexSet::new();
    for section in &extension.sections {
        let key = section.key.as_str();
        if !SECTION_KEY.is_match(key) {
            return Err(Error::type_error(format!(
                "Section key {} must match /^[a-z][a-z0-9_-]*$/",
                serde_json::to_string(key).unwrap_or_default()
            )));
        }
        if key == INSTRUCTIONS_KEY {
            return Err(Error::message(format!(
                "Section key {key} is reserved for the agent's instructions"
            )));
        }
        if !sections.insert(key) {
            return Err(Error::message(format!(
                "Extension {} has two sections with key {key}",
                extension.name
            )));
        }
    }
    Ok(())
}

/// Create an application-owned registry holding only the built-in tasks.
pub fn create_registry() -> Registry {
    Registry(Arc::new(RegistryInner {
        current: Mutex::new(Arc::new(
            RegistryState::new(Vec::new()).expect("built-in tasks have unique names"),
        )),
        listeners: Mutex::default(),
        next: Mutex::new(0),
    }))
}
