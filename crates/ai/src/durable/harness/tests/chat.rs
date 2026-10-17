//! Port of `test/chat-support.ts`: a faux model behind a `Models` registry, and Harness helpers for chat tests.
#![allow(dead_code)]

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::sync::Notify;

use super::support::{Reports, context};
use crate::chord::Context;
use crate::durable::env::ExecutionEnv;
use crate::durable::harness::RegistryReader;
use crate::durable::harness::types::{
    AgentChange, EnvFactory, HarnessOptions, HarnessSettings, ModelRef, ToolRegistration,
};
use crate::durable::harness::{Conversation, CreateOptions, Harness, Registry, create_registry};
use crate::durable::types::{EntryRecord, Storage};
use crate::error::Error as AiError;
use crate::models::{Models, create_models};
use crate::providers::faux::{
    FauxProviderHandle, FauxResponseStep, RegisterFauxProviderOptions, faux_provider,
};

pub use super::support::text_of;

/// Models and registry that survive a close/reopen, like a host process's own objects.
#[derive(Clone)]
pub struct ChatSetup {
    pub faux: Arc<FauxProviderHandle>,
    pub models: Models,
    pub registry: Registry,
    pub reports: Reports,
    /// Live Harness settings; tests change fields between decisions.
    pub settings: Arc<Mutex<HarnessSettings>>,
    pub now: Arc<Mutex<Arc<dyn Fn() -> u64 + Send + Sync>>>,
}

impl ChatSetup {
    pub fn set_now(&self, now: impl Fn() -> u64 + Send + Sync + 'static) {
        *self.now.lock() = Arc::new(now);
    }

    pub fn settings(&self, change: impl FnOnce(&mut HarnessSettings)) {
        change(&mut self.settings.lock());
    }
}

pub fn wall_clock() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

pub fn chat_setup() -> ChatSetup {
    chat_setup_with(RegisterFauxProviderOptions::default())
}

pub fn chat_setup_with(options: RegisterFauxProviderOptions) -> ChatSetup {
    let faux = faux_provider(options);
    let models = create_models(Default::default());
    models.set_provider(faux.provider.clone());
    ChatSetup {
        faux: Arc::new(faux),
        models,
        registry: create_registry(),
        reports: Reports::default(),
        settings: Arc::default(),
        now: Arc::new(Mutex::new(Arc::new(wall_clock))),
    }
}

/// The faux model reference.
pub fn faux_model() -> ModelRef {
    ModelRef::new("faux", "faux-1")
}

/// Options of [`open_chat_with`].
#[derive(Default, Clone)]
pub struct ChatOptions {
    pub env: Option<EnvFactory>,
    pub models: Option<Models>,
}

/// One environment for every conversation.
pub fn fixed_env(env: Arc<dyn ExecutionEnv>) -> EnvFactory {
    Arc::new(move |_, _| {
        let env = env.clone();
        Box::pin(async move { Ok(Some(env)) })
    })
}

/// Open a Harness over `storage` and return its root, configured with the faux model on first creation.
pub async fn open_chat(storage: Arc<dyn Storage>, setup: &ChatSetup) -> (Harness, Conversation) {
    open_chat_with(storage, setup, ChatOptions::default()).await
}

pub fn chat_options(setup: &ChatSetup, options: &ChatOptions) -> HarnessOptions {
    let mut harness_options = HarnessOptions::new(
        options
            .models
            .clone()
            .unwrap_or_else(|| setup.models.clone()),
        Arc::new(setup.registry.clone()),
    );
    let settings = setup.settings.clone();
    harness_options.settings = Some(Arc::new(move || settings.lock().clone()));
    harness_options.env = options.env.clone();
    let now = setup.now.clone();
    harness_options.now = Some(Arc::new(move || {
        let now = now.lock().clone();
        now()
    }));
    harness_options.on_report = Some(setup.reports.callback());
    harness_options
}

pub async fn open_chat_with(
    storage: Arc<dyn Storage>,
    setup: &ChatSetup,
    options: ChatOptions,
) -> (Harness, Conversation) {
    let harness = Harness::open(storage, chat_options(setup, &options), &context())
        .await
        .expect("open harness");
    let root = harness
        .root(
            &context(),
            CreateOptions {
                agent: Some(AgentChange::default().model(faux_model())),
                init: None,
            },
        )
        .await
        .expect("root");
    (harness, root)
}

/// Raw entries of a conversation, oldest first.
pub async fn all_entries(conversation: &Conversation) -> Vec<EntryRecord> {
    all_entries_in(conversation, &context()).await
}

pub async fn all_entries_in(conversation: &Conversation, context: &Context) -> Vec<EntryRecord> {
    let page = conversation
        .entries(None, None, 1000, None, context)
        .await
        .expect("entries");
    page.items.into_iter().rev().collect()
}

/// Kinds of the entries of a conversation, oldest first.
pub async fn entry_kinds(conversation: &Conversation) -> Vec<String> {
    all_entries(conversation)
        .await
        .into_iter()
        .map(|entry| entry.kind)
        .collect()
}

/// Poll `check` in real time until it holds; for waits that span throttle windows and timers.
pub async fn wait_for<F, Fut>(check: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    wait_for_ms(check, 5000).await;
}

pub async fn wait_for_ms<F, Fut>(mut check: F, timeout_ms: u64)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    while !check().await {
        if Instant::now() > deadline {
            panic!("Condition was not reached");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Resolves once signalled; cloneable.
#[derive(Clone, Default)]
pub struct Reached {
    state: Arc<(Mutex<bool>, Notify)>,
}

impl Reached {
    pub fn reach(&self) {
        *self.state.0.lock() = true;
        self.state.1.notify_waiters();
    }

    pub fn is_reached(&self) -> bool {
        *self.state.0.lock()
    }

    pub async fn wait(&self) {
        loop {
            let notified = self.state.1.notified();
            if self.is_reached() {
                return;
            }
            notified.await;
        }
    }
}

/// Faux response that never answers; the run stays busy until its generation is cancelled. `reached` resolves once
/// the request was sent, after the generation's preparation and request commits.
pub fn unanswered() -> (FauxResponseStep, Reached) {
    let reached = Reached::default();
    let reach = reached.clone();
    let step = FauxResponseStep::async_factory(move |_, options, _, _| {
        let reach = reach.clone();
        async move {
            reach.reach();
            match options.stream.signal.clone() {
                Some(signal) => signal.cancelled().await,
                None => std::future::pending::<()>().await,
            }
            Err(AiError::Aborted("Request aborted".into()))
        }
    });
    (step, reached)
}

/// The installed tools with these names, as `configure()` takes them.
pub fn tools_named(setup: &ChatSetup, names: &[&str]) -> Vec<ToolRegistration> {
    let installed = setup.registry.snapshot().tools();
    names
        .iter()
        .map(|name| {
            installed
                .iter()
                .find(|(_, tool)| tool.name == *name)
                .unwrap_or_else(|| panic!("Tool {name} is not installed"))
                .1
                .clone()
        })
        .collect()
}

/// `stream_simple(model, context, options)` of a proxied provider.
pub type StreamOverride = Arc<
    dyn Fn(
            crate::types::Model,
            crate::types::TranscriptContext,
            crate::types::SimpleStreamOptions,
        ) -> crate::utils::event_stream::AssistantMessageEventStream
        + Send
        + Sync,
>;

/// Overrides of [`proxy_models`] (the TS tests' `Proxy` over `Models`).
#[derive(Clone, Default)]
pub struct ProxyOverrides {
    pub stream_simple: Option<StreamOverride>,
    /// `cancelDeferred` rejects with this message.
    pub cancel_error: Option<String>,
}

struct ProxyStreams {
    inner: crate::providers::faux::FauxCore,
    overrides: ProxyOverrides,
}

struct ProxyAuth;

#[async_trait::async_trait]
impl crate::auth::types::ApiKeyAuth for ProxyAuth {
    fn name(&self) -> &str {
        "Faux"
    }

    async fn resolve(
        &self,
        _input: crate::auth::types::ApiKeyAuthInput,
    ) -> crate::error::Result<Option<crate::auth::types::AuthResult>> {
        Ok(Some(crate::auth::types::AuthResult::default()))
    }
}

#[async_trait::async_trait]
impl crate::types::ProviderStreams for ProxyStreams {
    fn stream(
        &self,
        model: crate::types::Model,
        context: crate::types::TranscriptContext,
        options: crate::types::StreamOptions,
    ) -> crate::utils::event_stream::AssistantMessageEventStream {
        self.stream_simple(
            model,
            context,
            crate::types::SimpleStreamOptions::from(options),
        )
    }

    fn stream_simple(
        &self,
        model: crate::types::Model,
        context: crate::types::TranscriptContext,
        options: crate::types::SimpleStreamOptions,
    ) -> crate::utils::event_stream::AssistantMessageEventStream {
        match &self.overrides.stream_simple {
            Some(stream) => stream(model, context, options),
            None => self.inner.stream_simple(model, context, options),
        }
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn fetch_deferred(
        &self,
        model: crate::types::Model,
        handle: crate::types::DeferredHandle,
        options: crate::types::DeferredFetchOptions,
    ) -> crate::utils::event_stream::AssistantMessageEventStream {
        self.inner.fetch_deferred(model, handle, options)
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }

    async fn cancel_deferred(
        &self,
        model: crate::types::Model,
        handle: crate::types::DeferredHandle,
        options: crate::types::DeferredCancelOptions,
    ) -> crate::error::Result<()> {
        if let Some(message) = &self.overrides.cancel_error {
            return Err(AiError::message(message.clone()));
        }
        self.inner.cancel_deferred(model, handle, options).await
    }
}

/// Models whose faux provider has the given operations replaced.
pub fn proxy_models(setup: &ChatSetup, overrides: ProxyOverrides) -> Models {
    let core: crate::providers::faux::FauxCore = (**setup.faux).clone();
    let provider = crate::models::create_provider(crate::models::CreateProviderOptions {
        id: core.provider().to_string(),
        auth: crate::auth::types::ProviderAuth {
            api_key: Some(Arc::new(ProxyAuth)),
            oauth: None,
        },
        models: core
            .models()
            .iter()
            .cloned()
            .map(crate::types::AnyModel::Chat)
            .collect(),
        api: Some(crate::models::ProviderApi::Single(Arc::new(ProxyStreams {
            inner: core,
            overrides,
        }))),
        ..Default::default()
    })
    .expect("proxy provider");
    let models = create_models(Default::default());
    models.set_provider(provider);
    models
}

/// A stream of `events` (sent with `delay_ms` before ending) that ends with `last`.
pub fn scripted_stream(
    events: Vec<crate::types::AssistantMessageEvent>,
    delay_ms: u64,
    last: crate::types::AssistantMessage,
) -> crate::utils::event_stream::AssistantMessageEventStream {
    let stream = crate::utils::event_stream::AssistantMessageEventStream::new();
    let producer = stream.clone();
    tokio::spawn(async move {
        for event in events {
            producer.push(event);
        }
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        producer.push(crate::types::AssistantMessageEvent::Done {
            reason: last.stop_reason,
            message: last,
        });
    });
    stream
}

/// `addSection(registry, key, () => text, { tag })`.
pub fn add_text_section(
    registry: &Registry,
    key: &str,
    text: &str,
    tag: Option<bool>,
) -> super::support::Installed {
    let text = text.to_string();
    super::support::add_section(
        registry,
        crate::durable::harness::section(
            key,
            move |_, _| {
                let text = text.clone();
                async move { Ok(Some(text)) }
            },
            tag,
        ),
    )
}

/// The committed `pi.live` value.
pub async fn live_state(
    harness: &Harness,
    conversation: &Conversation,
) -> Option<crate::durable::harness::live::LiveState> {
    harness
        .snapshot(
            &*crate::durable::harness::live::LIVE_DOC,
            conversation.id,
            &context(),
        )
        .await
        .expect("live")
}

/// Wait until a run task holds `pi.live.run` and return it.
pub async fn run_task(
    harness: &Harness,
    conversation: &Conversation,
) -> crate::durable::ids::TaskId {
    let found: Arc<Mutex<Option<crate::durable::ids::TaskId>>> = Arc::default();
    wait_for(|| {
        let found = found.clone();
        async move {
            let id = live_state(harness, conversation)
                .await
                .and_then(|live| live.run)
                .map(|run| run.task_id);
            *found.lock() = id;
            id.is_some()
        }
    })
    .await;
    found.lock().expect("run task")
}

/// Every committed `pi.live` value, in commit order.
pub fn live_publications(
    harness: &Harness,
) -> Arc<Mutex<Vec<crate::durable::harness::live::LiveState>>> {
    let values: Arc<Mutex<Vec<crate::durable::harness::live::LiveState>>> = Arc::default();
    let sink = values.clone();
    let unsubscribe = harness
        .subscribe_commits(move |publication, _| {
            for change in &publication.changes {
                if let crate::durable::types::CommitChange::Document(change) = change
                    && change.record.kind == "pi.live"
                    && let Some(value) = &change.value
                {
                    sink.lock()
                        .push(serde_json::from_value((**value).clone()).expect("live state"));
                }
            }
        })
        .expect("subscribe");
    std::mem::forget(unsubscribe);
    values
}
