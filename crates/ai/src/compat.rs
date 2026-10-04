//! Port of `compat.ts`: the global API registry and the api-dispatch
//! `stream()`/`complete()`/`stream_simple()`/`complete_simple()` entry points
//! with env API key injection, which also keep ai.rs's pre-1.0 call shape.
//!
//! Divergences:
//! - Rust addition: a model built from a provider handle carries the handle's
//!   `Models` collection (`Model::bound_models`); the entry points dispatch
//!   through it first.
//! - The builtin APIs are the in-scope ones (anthropic-messages,
//!   openai-completions, openai-responses), and the builtin providers are
//!   anthropic, github-copilot, openai and openrouter (image models only).
//!   Image generation (`generateImages` and its registry) lives in
//!   [`crate::images`] and [`crate::images_api_registry`]. The Cloudflare auth branch has no
//!   in-scope provider and is not ported.
//! - The legacy `getModel`/`getModels`/`getProviders` aliases are the
//!   `providers::all` getters.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use indexmap::IndexMap;
use parking_lot::RwLock;

use crate::api::anthropic_messages::anthropic_messages_api;
use crate::api::openai_completions::openai_completions_api;
use crate::api::openai_responses::openai_responses_api;
use crate::env_api_keys::get_env_api_key;
use crate::models::{Models, Provider};
use crate::providers::all::builtin_models;
use crate::providers::faux::{
    FauxProviderRegistration, RegisterFauxProviderOptions, create_faux_core, random_suffix,
};
use crate::types::{
    Api, AssistantMessage, Context, KnownApi, Model, ProviderStreams, SimpleStreamOptions,
    StreamOptions, TranscriptContext,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::transcript::normalize_context;
use crate::{Error, Result};

pub use crate::providers::all::{
    get_builtin_model as get_model, get_builtin_models as get_models,
    get_builtin_providers as get_providers,
};

pub type ApiStreamFunction = Arc<
    dyn Fn(Model, TranscriptContext, StreamOptions) -> AssistantMessageEventStream + Send + Sync,
>;

pub type ApiStreamSimpleFunction = Arc<
    dyn Fn(Model, TranscriptContext, SimpleStreamOptions) -> AssistantMessageEventStream
        + Send
        + Sync,
>;

/// An API implementation registered under its api id.
#[derive(Clone)]
pub struct ApiProvider {
    pub api: Api,
    pub stream: ApiStreamFunction,
    pub stream_simple: ApiStreamSimpleFunction,
}

impl ApiProvider {
    /// Wrap a `ProviderStreams` implementation.
    pub fn from_streams(api: impl Into<Api>, streams: Arc<dyn ProviderStreams>) -> Self {
        let simple = streams.clone();
        Self {
            api: api.into(),
            stream: Arc::new(move |model, context, options| {
                streams.stream(model, context, options)
            }),
            stream_simple: Arc::new(move |model, context, options| {
                simple.stream_simple(model, context, options)
            }),
        }
    }
}

/// A registered API implementation (`ApiProviderInternal`). The stream
/// functions check that the model's api matches.
#[derive(Clone)]
pub struct RegisteredApiProvider {
    pub api: Api,
    provider: ApiProvider,
    instance: u64,
}

impl RegisteredApiProvider {
    pub fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> Result<AssistantMessageEventStream> {
        if model.api != self.api {
            return Err(Error::message(format!(
                "Mismatched api: {} expected {}",
                model.api, self.api
            )));
        }
        Ok((self.provider.stream)(model, context, options))
    }

    pub fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream> {
        if model.api != self.api {
            return Err(Error::message(format!(
                "Mismatched api: {} expected {}",
                model.api, self.api
            )));
        }
        Ok((self.provider.stream_simple)(model, context, options))
    }
}

struct RegistryEntry {
    provider: RegisteredApiProvider,
    source_id: Option<String>,
}

#[derive(Default)]
struct Registry {
    providers: IndexMap<String, RegistryEntry>,
    builtin_instances: IndexMap<String, u64>,
}

static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

static REGISTRY: LazyLock<RwLock<Registry>> = LazyLock::new(|| {
    let registry = RwLock::new(Registry::default());
    register_builtin_api_providers_into(&mut registry.write());
    registry
});

fn builtin_apis() -> Vec<(Api, Arc<dyn ProviderStreams>)> {
    vec![
        (
            KnownApi::AnthropicMessages.as_str().to_string(),
            anthropic_messages_api(),
        ),
        (
            KnownApi::OpenaiCompletions.as_str().to_string(),
            openai_completions_api(),
        ),
        (
            KnownApi::OpenaiResponses.as_str().to_string(),
            openai_responses_api(),
        ),
    ]
}

fn insert(registry: &mut Registry, provider: ApiProvider, source_id: Option<String>) {
    let api = provider.api.clone();
    registry.providers.insert(
        api.clone(),
        RegistryEntry {
            provider: RegisteredApiProvider {
                api,
                provider,
                instance: NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed),
            },
            source_id,
        },
    );
}

pub fn register_api_provider(provider: ApiProvider, source_id: Option<&str>) {
    insert(
        &mut REGISTRY.write(),
        provider,
        source_id.map(str::to_string),
    );
}

pub fn get_api_provider(api: &str) -> Option<RegisteredApiProvider> {
    REGISTRY
        .read()
        .providers
        .get(api)
        .map(|entry| entry.provider.clone())
}

pub fn get_api_providers() -> Vec<RegisteredApiProvider> {
    REGISTRY
        .read()
        .providers
        .values()
        .map(|entry| entry.provider.clone())
        .collect()
}

pub fn unregister_api_providers(source_id: &str) {
    REGISTRY
        .write()
        .providers
        .retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

/// `registerFauxProvider(options)`: register a faux API implementation under
/// its own source id. `unregister()` removes it again.
pub fn register_faux_provider(options: RegisterFauxProviderOptions) -> FauxProviderRegistration {
    let core = create_faux_core(options);
    let source_id = format!("faux-provider-{}", random_suffix());
    register_api_provider(
        ApiProvider::from_streams(core.api(), Arc::new(core.clone())),
        Some(&source_id),
    );
    FauxProviderRegistration::new(core, source_id)
}

fn register_builtin_api_providers_into(registry: &mut Registry) {
    for (api, streams) in builtin_apis() {
        if !registry.providers.contains_key(&api) {
            insert(
                registry,
                ApiProvider::from_streams(api.clone(), streams),
                None,
            );
        }
        if let Some(entry) = registry.providers.get(&api) {
            let instance = entry.provider.instance;
            registry.builtin_instances.insert(api, instance);
        }
    }
}

/// Registers the builtin API implementations into the api-registry without
/// clobbering existing entries.
pub fn register_builtin_api_providers() {
    register_builtin_api_providers_into(&mut REGISTRY.write());
}

pub fn reset_api_providers() {
    let mut registry = REGISTRY.write();
    registry.providers.clear();
    registry.builtin_instances.clear();
    register_builtin_api_providers_into(&mut registry);
}

static COMPAT_MODELS: LazyLock<Models> = LazyLock::new(|| builtin_models(Default::default()));

const AMBIENT_AUTH_MARKER: &str = "<authenticated>";

fn has_explicit_api_key(api_key: Option<&String>) -> bool {
    api_key.is_some_and(|api_key| !api_key.trim().is_empty())
}

fn with_env_api_key(model: &Model, mut options: StreamOptions) -> StreamOptions {
    if has_explicit_api_key(options.api_key.as_ref()) {
        return options;
    }
    match get_env_api_key(&model.provider, options.env.as_ref()) {
        Some(api_key) if api_key != AMBIENT_AUTH_MARKER => {
            options.api_key = Some(api_key);
            options
        }
        _ => options,
    }
}

fn with_env_api_key_simple(model: &Model, mut options: SimpleStreamOptions) -> SimpleStreamOptions {
    options.stream = with_env_api_key(model, std::mem::take(&mut options.stream));
    options
}

fn get_builtin_provider_for_model(model: &Model) -> Option<Arc<dyn Provider>> {
    {
        let registry = REGISTRY.read();
        let current = registry
            .providers
            .get(&model.api)
            .map(|entry| entry.provider.instance);
        if current != registry.builtin_instances.get(&model.api).copied() {
            return None;
        }
    }
    let provider = COMPAT_MODELS.get_provider(&model.provider)?;
    provider
        .get_models()
        .ok()?
        .iter()
        .any(|candidate| candidate.api == model.api)
        .then_some(provider)
}

fn resolve_api_provider(api: &str) -> Result<RegisteredApiProvider> {
    get_api_provider(api)
        .ok_or_else(|| Error::message(format!("No API provider registered for api: {api}")))
}

/// Stream through the model's API. Fails when no implementation is registered
/// for `model.api`; request failures arrive as error events.
pub fn stream(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessageEventStream> {
    let options = options.unwrap_or_default();
    if let Some(models) = model.bound_models.clone() {
        return Ok(models.stream(&model, &context, options));
    }
    let transcript = normalize_context(&context);
    if let Some(builtin_provider) = get_builtin_provider_for_model(&model) {
        let options = with_env_api_key(&model, options);
        return Ok(builtin_provider.stream(model, transcript, options));
    }
    let provider = resolve_api_provider(&model.api)?;
    let options = with_env_api_key(&model, options);
    provider.stream(model, transcript, options)
}

pub async fn complete(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessage> {
    Ok(stream(model, context, options)?.result().await)
}

/// Stream with unified reasoning options. Fails when no implementation is
/// registered for `model.api`; request failures arrive as error events.
pub fn stream_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessageEventStream> {
    let options = options.unwrap_or_default();
    if let Some(models) = model.bound_models.clone() {
        return Ok(models.stream_simple(&model, &context, options));
    }
    let transcript = normalize_context(&context);
    if let Some(builtin_provider) = get_builtin_provider_for_model(&model) {
        let options = with_env_api_key_simple(&model, options);
        return Ok(builtin_provider.stream_simple(model, transcript, options));
    }
    let provider = resolve_api_provider(&model.api)?;
    let options = with_env_api_key_simple(&model, options);
    provider.stream_simple(model, transcript, options)
}

pub async fn complete_simple(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessage> {
    Ok(stream_simple(model, context, options)?.result().await)
}

/// Serializes tests that replace or reset registry entries.
#[cfg(test)]
pub(crate) static REGISTRY_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use super::*;
    use crate::types::{AssistantContent, AssistantMessageEvent, Message, ModelInput, StopReason};

    fn context() -> Context {
        Context::builder().message(Message::user_text("hi")).build()
    }

    fn model() -> Model {
        Model {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            api: "openai-responses".to_string(),
            provider: "custom-openai".to_string(),
            base_url: "https://example.test/v1".to_string(),
            input: vec![ModelInput::Text],
            context_window: 128_000,
            max_tokens: 4096,
            ..Default::default()
        }
    }

    fn respond(model: &Model) -> AssistantMessageEventStream {
        let stream = AssistantMessageEventStream::new();
        let mut output = AssistantMessage::empty_for(model);
        output.content = vec![AssistantContent::text("ok")];
        stream.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });
        stream.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: output.clone(),
        });
        stream.end(Some(output));
        stream
    }

    #[tokio::test]
    async fn dispatches_unknown_providers_through_the_legacy_api_registry() {
        let _lock = REGISTRY_TEST_LOCK.lock().await;
        let captured: Arc<Mutex<Option<String>>> = Arc::default();
        let stream_captured = captured.clone();
        let simple_captured = captured.clone();
        register_api_provider(
            ApiProvider {
                api: "openai-responses".to_string(),
                stream: Arc::new(move |model, _context, options| {
                    *stream_captured.lock() = options.api_key.clone();
                    respond(&model)
                }),
                stream_simple: Arc::new(move |model, _context, options| {
                    *simple_captured.lock() = options.api_key.clone();
                    respond(&model)
                }),
            },
            None,
        );

        let result = complete(
            model(),
            context(),
            Some(StreamOptions {
                api_key: Some("request-key".to_string()),
                ..Default::default()
            }),
        )
        .await;
        reset_api_providers();

        assert_eq!(result.unwrap().stop_reason, StopReason::Stop);
        assert_eq!(captured.lock().as_deref(), Some("request-key"));
    }

    #[tokio::test]
    async fn unknown_apis_fail_before_streaming() {
        let mut model = model();
        model.api = "no-such-api".to_string();
        let error = stream_simple(model, context(), None).err().unwrap();
        assert_eq!(
            error.to_string(),
            "No API provider registered for api: no-such-api"
        );
    }

    #[tokio::test]
    async fn builtin_models_dispatch_through_their_provider() {
        let _lock = REGISTRY_TEST_LOCK.lock().await;
        let model = get_model("openai", "gpt-5.5").unwrap();
        assert!(get_builtin_provider_for_model(&model).is_some());
        let (options, payloads) = capture_payload_options();
        let result = complete_simple(model, context(), Some(options))
            .await
            .unwrap();
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some("payload captured"));
        assert_eq!(payloads.lock()[0]["model"], "gpt-5.5");
    }

    #[tokio::test]
    async fn handle_models_dispatch_through_their_bound_collection() {
        let handle = crate::providers::openai::builder()
            .provider_id("handle-only")
            .api_key(Some("handle-key"))
            .build()
            .unwrap();
        let model = handle.model("gpt-5.5").build().unwrap();
        let (options, payloads) = capture_payload_options();
        let result = complete_simple(model, context(), Some(options))
            .await
            .unwrap();
        assert_eq!(result.provider, "handle-only");
        assert_eq!(result.error_message.as_deref(), Some("payload captured"));
        assert_eq!(payloads.lock()[0]["model"], "gpt-5.5");
    }

    /// Options whose `onPayload` hook records the request body and fails
    /// the request, proving dispatch reached the API implementation without
    /// a network call.
    fn capture_payload_options() -> (
        SimpleStreamOptions,
        std::sync::Arc<parking_lot::Mutex<Vec<serde_json::Value>>>,
    ) {
        let (hook, payloads) = crate::api::openai_client::test_support::payload_hook(true);
        let mut options = SimpleStreamOptions::default();
        options.stream.api_key = Some("test-key".to_string());
        options.stream.on_payload = Some(hook);
        (options, payloads)
    }
}
