//! Ports of `test/harness-support.ts` and `test/task-support.ts`.
#![allow(dead_code, unused_imports)]

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde_json::json;

use crate::chord::{AbortSignal, Context};
use crate::durable::errors::{Error, Result};
use crate::durable::harness::registry::{Registry, RegistryReader, RegistrySnapshot};
use crate::durable::harness::types::{
    ExtensionDefinition, HarnessOptions, HarnessSettings, HookRegistration, ToolExecutionResult,
    ToolRegistration,
};
use crate::durable::harness::{Harness, create_registry, define_extension, define_tool};
use crate::durable::session::Unsubscribe;
use crate::durable::session::tests::support::flush;
use crate::durable::tasks::AnyTask;
use crate::durable::types::Storage;
use crate::models::Models;
use crate::types::{
    AssistantContent, AssistantMessage, Message, StopReason, SystemMessage, TextContent, ToolCall,
    ToolResultMessage, Usage, UserContent, UserMessage,
};

pub use crate::durable::session::tests::support::{assert_err, assert_matches, context, to_json};

/// A tool with an empty object schema whose execution returns no content.
pub fn tool(name: &str) -> ToolRegistration {
    tool_described(name, &format!("{name} tool"))
}

pub fn tool_described(name: &str, description: &str) -> ToolRegistration {
    define_tool(
        name,
        description,
        json!({ "type": "object", "properties": {} }),
        |_, _, _| async {
            Ok(ToolExecutionResult {
                content: Some(Vec::new()),
                ..ToolExecutionResult::default()
            })
        },
    )
}

/// Collected reports.
#[derive(Clone, Default)]
pub struct Reports(pub Arc<Mutex<Vec<Error>>>);

impl Reports {
    pub fn messages(&self) -> Vec<String> {
        self.0.lock().iter().map(ToString::to_string).collect()
    }

    pub fn len(&self) -> usize {
        self.0.lock().len()
    }

    pub fn errors(&self) -> Vec<Error> {
        self.0.lock().clone()
    }

    pub fn callback(&self) -> Arc<dyn Fn(Error) + Send + Sync> {
        let reports = self.0.clone();
        Arc::new(move |error| reports.lock().push(error))
    }
}

/// Open a Harness with a fresh registry holding the named tools.
pub async fn open_harness(
    storage: Arc<dyn Storage>,
    tool_names: &[&str],
    registry: Option<Registry>,
    reports: Option<&Reports>,
) -> (Harness, Registry) {
    let registry = registry.unwrap_or_else(create_registry);
    if !tool_names.is_empty() {
        registry
            .install(define_extension(ExtensionDefinition {
                tools: tool_names.iter().map(|name| tool(name)).collect(),
                ..ExtensionDefinition::new("tools")
            }))
            .expect("install tools");
    }
    let mut options = HarnessOptions::new(Models::default(), Arc::new(registry.clone()));
    options.on_report = reports.map(Reports::callback);
    let harness = Harness::open(storage, options, &context())
        .await
        .expect("open harness");
    (harness, registry)
}

/// Options of [`open_tasks`].
#[derive(Default)]
pub struct TaskOptions {
    pub registry: Option<Registry>,
    pub now: Option<Arc<dyn Fn() -> u64 + Send + Sync>>,
    pub settings: Option<HarnessSettings>,
}

/// Open a Harness whose registry holds `tasks`; failures passed to `onReport` are collected.
pub async fn open_tasks(
    storage: Arc<dyn Storage>,
    tasks: Vec<AnyTask>,
    options: TaskOptions,
) -> (Harness, Registry, Reports) {
    let registry = options.registry.unwrap_or_else(create_registry);
    if !tasks.is_empty() {
        registry
            .install(define_extension(ExtensionDefinition {
                tasks,
                ..ExtensionDefinition::new("tasks")
            }))
            .expect("install tasks");
    }
    let reports = Reports::default();
    let mut harness_options = HarnessOptions::new(Models::default(), Arc::new(registry.clone()));
    harness_options.on_report = Some(reports.callback());
    harness_options.now = options.now;
    if let Some(settings) = options.settings {
        harness_options.settings = Some(Arc::new(move || settings.clone()));
    }
    let harness = Harness::open(storage, harness_options, &context())
        .await
        .expect("open harness");
    (harness, registry, reports)
}

/// Flush scheduler turns until `check` holds.
pub async fn eventually<F, Fut>(mut check: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    for _ in 0..200 {
        if check().await {
            return;
        }
        flush().await;
    }
    panic!("Condition was not reached");
}

/// Whether `future` settles after pending work flushes; it keeps running in the background.
pub async fn settled<T: Send + 'static>(
    future: impl Future<Output = T> + Send + 'static,
) -> (bool, tokio::task::JoinHandle<T>) {
    let handle = tokio::spawn(future);
    flush().await;
    (handle.is_finished(), handle)
}

/// Fail with the signal's reason once it aborts; for handlers that block until cancelled.
pub async fn aborted(signal: AbortSignal) -> Result<()> {
    signal.cancelled().await;
    Err(Error::Aborted(
        signal
            .reason()
            .unwrap_or_else(crate::chord::AbortReason::abort_error),
    ))
}

/// Registry reader that counts live subscriptions.
pub struct CountingReader {
    registry: Registry,
    count: Arc<AtomicUsize>,
}

impl CountingReader {
    pub fn new(registry: Registry) -> Arc<Self> {
        Arc::new(Self {
            registry,
            count: Arc::default(),
        })
    }

    pub fn subscriptions(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

impl RegistryReader for CountingReader {
    fn snapshot(&self) -> RegistrySnapshot {
        self.registry.snapshot()
    }

    fn subscribe(&self, listener: Arc<dyn Fn() + Send + Sync>) -> Unsubscribe {
        self.count.fetch_add(1, Ordering::SeqCst);
        let unsubscribe = self.registry.subscribe(listener);
        let active = Arc::new(Mutex::new(true));
        let count = self.count.clone();
        Box::new(move || {
            let mut active = active.lock();
            if !*active {
                return;
            }
            *active = false;
            count.fetch_sub(1, Ordering::SeqCst);
            unsubscribe();
        })
    }
}

/// Uninstalls what one of the helpers below installed.
pub struct Installed {
    registry: Registry,
    extension: crate::durable::harness::Extension,
}

impl Installed {
    pub fn dispose(&self) {
        self.registry.uninstall(&self.extension).expect("uninstall");
    }
}

fn install_one(registry: &Registry, extension: ExtensionDefinition) -> Installed {
    let extension = define_extension(extension);
    registry.install(extension.clone()).expect("install");
    Installed {
        registry: registry.clone(),
        extension,
    }
}

/// Install a one-tool extension named after the tool.
pub fn add_tool(registry: &Registry, tool: ToolRegistration) -> Installed {
    let name = format!("tool:{}", tool.name);
    install_one(
        registry,
        ExtensionDefinition {
            tools: vec![tool],
            ..ExtensionDefinition::new(name)
        },
    )
}

/// Install a one-task extension named after the task.
pub fn add_task(registry: &Registry, task: AnyTask) -> Installed {
    let name = format!("task:{}", task.name());
    install_one(
        registry,
        ExtensionDefinition {
            tasks: vec![task],
            ..ExtensionDefinition::new(name)
        },
    )
}

static HOOK_EXTENSIONS: AtomicUsize = AtomicUsize::new(0);

/// Install an extension with one hook registration.
pub fn add_hooks(registry: &Registry, hook: HookRegistration) -> Installed {
    let name = format!(
        "hooks:{}",
        HOOK_EXTENSIONS.fetch_add(1, Ordering::SeqCst) + 1
    );
    install_one(
        registry,
        ExtensionDefinition {
            hooks: vec![hook],
            ..ExtensionDefinition::new(name)
        },
    )
}

/// Install a one-section extension named after the section.
pub fn add_section(
    registry: &Registry,
    section: crate::durable::harness::PromptSection,
) -> Installed {
    let name = format!("section:{}", section.key);
    install_one(
        registry,
        ExtensionDefinition {
            sections: vec![section],
            ..ExtensionDefinition::new(name)
        },
    )
}

pub fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: text.into(),
        timestamp: 1,
    })
}

pub fn zero_usage() -> Usage {
    Usage::default()
}

pub fn assistant_message(
    text: &str,
    calls: &[&str],
    stop_reason: Option<StopReason>,
) -> AssistantMessage {
    let mut content = vec![AssistantContent::Text(TextContent::new(text))];
    for id in calls {
        content.push(AssistantContent::ToolCall(ToolCall {
            id: id.to_string(),
            name: format!("tool-{id}"),
            arguments: json!({}),
            thought_signature: None,
            namespace: None,
        }));
    }
    let stop_reason = stop_reason.unwrap_or(if calls.is_empty() {
        StopReason::Stop
    } else {
        StopReason::ToolUse
    });
    serde_json::from_value(json!({
        "role": "assistant",
        "content": content,
        "api": "faux",
        "provider": "faux",
        "model": "faux",
        "usage": zero_usage(),
        "stopReason": stop_reason,
        "timestamp": 2,
    }))
    .expect("assistant message")
}

pub fn assistant(text: &str) -> Message {
    Message::Assistant(assistant_message(text, &[], None))
}

pub fn assistant_calls(text: &str, calls: &[&str]) -> Message {
    Message::Assistant(assistant_message(text, calls, None))
}

pub fn assistant_stopped(text: &str, stop_reason: StopReason) -> Message {
    Message::Assistant(assistant_message(text, &[], Some(stop_reason)))
}

pub fn tool_result(id: &str) -> Message {
    tool_result_text(id, &format!("result {id}"))
}

pub fn tool_result_text(id: &str, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: id.to_string(),
        tool_name: format!("tool-{id}"),
        content: vec![UserContent::Text(TextContent::new(text))],
        details: None,
        usage: None,
        nested_calls: None,
        is_error: false,
        timestamp: 3,
    })
}

pub fn system(sections: &[(&str, Option<&str>)]) -> Message {
    Message::System(SystemMessage {
        content: Default::default(),
        sections: Some(
            sections
                .iter()
                .map(|(key, value)| (key.to_string(), value.map(str::to_string)))
                .collect(),
        ),
        tools_added: None,
        tools_removed: None,
        timestamp: 4,
    })
}

/// Text of the first text content of a message.
pub fn text_of(message: Option<&Message>) -> Option<String> {
    match message? {
        Message::System(_) => None,
        Message::User(user) => match &user.content {
            crate::types::UserMessageContent::Text(text) => Some(text.clone()),
            crate::types::UserMessageContent::Parts(parts) => {
                parts.iter().find_map(|part| match part {
                    UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
            }
        },
        Message::Assistant(assistant) => {
            assistant.content.iter().find_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
        }
        Message::ToolResult(result) => result.content.iter().find_map(|content| match content {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        }),
    }
}

/// Compact message rendering for assertions.
pub fn describe_message(message: &Message) -> String {
    match message {
        Message::User(_) => format!("user:{}", text_of(Some(message)).unwrap_or_default()),
        Message::Assistant(_) => {
            format!("assistant:{}", text_of(Some(message)).unwrap_or_default())
        }
        Message::ToolResult(result) => format!(
            "result:{}:{}",
            result.tool_call_id,
            if result.is_error {
                "error".to_string()
            } else {
                text_of(Some(message)).unwrap_or_default()
            }
        ),
        Message::System(system) => format!(
            "system:{}",
            system
                .sections
                .as_ref()
                .map(|sections| sections.keys().cloned().collect::<Vec<_>>().join(","))
                .unwrap_or_default()
        ),
    }
}

/// A context cancelled by `signal`.
pub fn signal_context(signal: &AbortSignal) -> Context {
    crate::chord::with_abort_signal(signal, &context())
}

/// The checkpoint of a one-phase task.
pub fn run_phase() -> serde_json::Value {
    json!({ "phase": "run" })
}

/// A one-phase task type with unit input and a JSON checkpoint.
pub type StepTask = crate::durable::tasks::Task<(), serde_json::Value, serde_json::Value, ()>;

/// A one-phase task running `run` and completing with null.
pub fn one_step<F, Fut>(name: &str, run: F) -> StepTask
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let run = Arc::new(run);
    crate::durable::tasks::define_task(
        crate::durable::tasks::TaskDefinition::new(name, 1, |_: &()| run_phase()).phase(
            "run",
            move |_, runtime, context| {
                let run = run.clone();
                async move {
                    run().await;
                    runtime
                        .commit(
                            |_, _| async {
                                Ok(Some(crate::durable::tasks::NextTaskState::completed(())))
                            },
                            &context,
                        )
                        .await
                }
            },
        ),
    )
}

/// A one-phase task that completes at once.
pub fn noop_step(name: &str) -> StepTask {
    one_step(name, || async {})
}

/// Create `task` owned by `conversation`.
pub async fn start<I, S, R, H>(
    conversation: &crate::durable::harness::Conversation,
    task: &crate::durable::tasks::Task<I, S, R, H>,
    input: I,
    background: bool,
) -> crate::durable::ids::TaskId
where
    I: serde::Serialize + Send + 'static,
    S: serde::Serialize + 'static,
    R: 'static,
    H: Send + Sync + 'static,
{
    let task = task.clone();
    conversation
        .commit(
            move |tx| async move {
                tx.create_task(
                    &task,
                    input,
                    crate::durable::types::TaskOptions {
                        ownership: crate::durable::types::TaskOwnership::Conversation,
                        conversation_id: None,
                        background: Some(background),
                    },
                )
                .await
                .map(|id| id.erase())
            },
            &context(),
        )
        .await
        .expect("create task")
}

/// Whether a raw Storage read still succeeds, as code of a joined invocation relies on.
pub async fn storage_open(storage: &Arc<dyn Storage>) -> bool {
    storage
        .task(crate::durable::ids::TaskId::new(1), &context())
        .await
        .is_ok()
}
