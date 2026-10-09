//! Port of `models.ts`: the `Provider` contract, the `Models` runtime
//! collection, `create_provider()` and the model helpers.
//!
//! Divergences:
//! - `Models` is one cloneable handle (Pi's `Models` + `MutableModels`).
//! - Pi's optional provider methods become provided trait methods: methods
//!   that return a value use `Option` for "absent" (`filter_models`), others
//!   are paired with a `supports_*` probe (`refresh_models`, deferred calls).
//! - Requests run on spawned Tokio tasks; abandoned operations are dropped
//!   (cancelled) instead of continuing in the background.
//! - Unknown model types cannot be represented by [`AnyModel`], so Pi's
//!   `hasKnownModelType()` filtering happens when raw models are
//!   deserialized: [`ModelsStoreEntry`] drops them, and `fetch_models`
//!   implementations that read JSON use
//!   [`known_models_from_values`](crate::types::known_models_from_values).
//! - ai.rs extra, not in Pi: embedding models (`ModelType::Embedding`),
//!   [`Provider::embed`], [`Models::embed`] and
//!   `CreateProviderOptions::embeddings`, designed like the image path
//!   (`generate_images`). `create_provider()` accepts `embeddings` as the
//!   only implementation map, and its error message names it.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all};
use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use crate::api::lazy::{error_stream, lazy_stream};
use crate::auth::resolve::read_credential;
use crate::auth::{
    ApiKeyAuthInput, AuthCheck, AuthContext, AuthInteraction, AuthOperationOptions,
    AuthResolutionOverrides, AuthResult, AuthType, Credential, CredentialStore,
    InMemoryCredentialStore, LoginOptions, ProviderAuth, ProviderAuthInteraction,
    default_provider_auth_context, models_error, models_error_with_cause, resolve_provider_auth,
    throw_if_aborted,
};
use crate::models_store::{
    InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions,
};
use crate::types::{
    AnyModel, AssistantImages, AssistantMessage, BoxFuture, ClassifierContext, ClassifierModel,
    ClassifierOptions, ClassifierResult, Context, DeferredCancelOptions, DeferredFetchOptions,
    DeferredHandle, EmbeddingModel, EmbeddingsContext, EmbeddingsOptions, EmbeddingsResult,
    ImageModel, ImagesContext, ImagesOptions, Model, ModelCost, ModelCostRates, ModelThinkingLevel,
    ModelType, ProviderClassifier, ProviderEmbeddings, ProviderEnv, ProviderHeaders,
    ProviderImages, ProviderRequestOptions, ProviderStreams, SimpleStreamOptions, StreamOptions,
    TranscriptContext, Usage, UsageCost,
};
use crate::utils::abort::{operation_signal, race_with_abort_signal};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::model_operations::{
    assert_chat_model, classifier_error_result, embeddings_error_result, image_error_result,
};

use crate::utils::time::now_millis;
use crate::utils::transcript::normalize_context;
use crate::{Error, Result};

pub use crate::utils::model_operations::{get_model_type, is_model_type};
pub use crate::utils::models_error::{ModelsError, ModelsErrorCode};

/// Synchronous update of provider-private in-memory catalog state.
pub type ModelsUpdate = Box<dyn FnOnce() + Send>;

#[derive(Default)]
pub struct ModelsPublication {
    /// Provider-selected persisted catalog. `None` leaves storage unchanged;
    /// `Some(None)` deletes it (Pi: omitted vs `null`).
    pub persist: Option<Option<ModelsStoreEntry>>,
    /// Optional synchronous update of provider-private in-memory catalog state.
    pub update: Option<ModelsUpdate>,
}

type Publisher = Arc<dyn Fn(ModelsPublication) -> BoxFuture<Result<bool>> + Send + Sync>;

#[derive(Clone)]
pub struct RefreshModelsContext {
    /// Effective configured credential. OAuth credentials are refreshed before network access.
    pub credential: Option<Credential>,
    /// Provider-scoped catalog snapshot captured before this refresh phase.
    pub stored: Option<ModelsStoreEntry>,
    /// False during offline/cache-only initialization.
    pub allow_network: bool,
    /// Bypass provider freshness checks and fetch immediately when network access is allowed.
    pub force: Option<bool>,
    /// Always present, including when the public refresh caller omits its optional signal.
    pub signal: CancellationToken,
    publisher: Publisher,
}

impl RefreshModelsContext {
    /// Generation-checked publication. Persistence policy remains
    /// provider-owned; the update runs synchronously only after the selected
    /// persistence mutation.
    pub async fn publish(&self, publication: ModelsPublication) -> Result<bool> {
        (self.publisher)(publication).await
    }
}

impl fmt::Debug for RefreshModelsContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefreshModelsContext")
            .field("credential", &self.credential)
            .field("stored", &self.stored)
            .field("allow_network", &self.allow_network)
            .field("force", &self.force)
            .field("signal", &self.signal)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelsRefreshOptions {
    /// Default `true`.
    pub allow_network: Option<bool>,
    /// Restrict refresh to these provider IDs. Unknown and static providers are ignored.
    pub providers: Option<Vec<String>>,
    /// Bypass provider freshness checks and fetch immediately when network access is allowed.
    pub force: Option<bool>,
    pub signal: Option<CancellationToken>,
}

#[derive(Debug, Default)]
pub struct ModelsRefreshResult {
    pub aborted: bool,
    pub errors: IndexMap<String, Error>,
}

/// `transformHeaders`: transform fully assembled model/auth/request headers
/// before provider dispatch.
pub type HeadersTransform =
    Arc<dyn Fn(ProviderHeaders) -> BoxFuture<Result<ProviderHeaders>> + Send + Sync>;

/// Request options plus the `Models`-only request transforms
/// (`ModelsRequestTransforms`).
#[derive(Clone, Default)]
pub struct ModelsOptions<T> {
    pub options: T,
    pub transform_headers: Option<HeadersTransform>,
}

impl<T> From<T> for ModelsOptions<T> {
    fn from(options: T) -> Self {
        Self {
            options,
            transform_headers: None,
        }
    }
}

pub type ModelsApiStreamOptions = ModelsOptions<StreamOptions>;
pub type ModelsSimpleStreamOptions = ModelsOptions<SimpleStreamOptions>;
pub type ModelsDeferredFetchOptions = ModelsOptions<DeferredFetchOptions>;
pub type ModelsDeferredCancelOptions = ModelsOptions<DeferredCancelOptions>;
pub type ModelsImagesOptions = ModelsOptions<ImagesOptions>;
pub type ModelsClassifierOptions = ModelsOptions<ClassifierOptions>;
/// ai.rs extra: [`EmbeddingsOptions`] plus the `Models` request transforms.
pub type ModelsEmbeddingsOptions = ModelsOptions<EmbeddingsOptions>;

/// A provider is the concrete runtime unit. It owns id/name/base metadata,
/// auth methods, model listing, and the operations its models support.
#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;

    fn name(&self) -> &str;

    fn base_url(&self) -> Option<&str> {
        None
    }

    fn headers(&self) -> Option<&ProviderHeaders> {
        None
    }

    /// Required: at least one of `api_key`/`oauth`. `Models::get_auth()`
    /// returns `None` when the provider is unconfigured.
    fn auth(&self) -> &ProviderAuth;

    /// Current known chat models. Static providers return their catalog;
    /// dynamic providers return the list as of the last `refresh_models()`.
    /// `Models` treats an error as having no models.
    fn get_models(&self) -> Result<Vec<Model>>;

    /// Current known models of every type, with the same contract as
    /// `get_models()`. Defaults to the chat models.
    fn get_all_models(&self) -> Result<Vec<AnyModel>> {
        Ok(self.get_models()?.into_iter().map(AnyModel::Chat).collect())
    }

    /// Whether [`Provider::refresh_models`] is implemented (dynamic providers).
    fn supports_refresh_models(&self) -> bool {
        false
    }

    /// Dynamic providers only: restore `context.stored` and optionally fetch
    /// a newer list using the effective credential.
    async fn refresh_models(&self, _context: RefreshModelsContext) -> Result<()> {
        Ok(())
    }

    /// Optional credential-specific chat model availability. `None` means the
    /// provider has no policy (all models are available).
    fn filter_models(
        &self,
        _models: &[Model],
        _credential: Option<&Credential>,
    ) -> Option<Vec<Model>> {
        None
    }

    /// Optional credential-specific availability across every model type.
    /// `None` means `Models` applies `filter_models` to chat models.
    fn filter_all_models(
        &self,
        _models: &[AnyModel],
        _credential: Option<&Credential>,
    ) -> Option<Vec<AnyModel>> {
        None
    }

    /// Stream a normalized transcript. `Models` normalizes the caller's
    /// `Context` before dispatching here.
    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream;

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream;

    fn supports_fetch_deferred(&self) -> bool {
        false
    }

    fn fetch_deferred(
        &self,
        model: Model,
        _handle: DeferredHandle,
        _options: DeferredFetchOptions,
    ) -> AssistantMessageEventStream {
        error_stream(
            &model,
            format!("Provider {} does not support deferred responses", self.id()),
        )
    }

    fn supports_cancel_deferred(&self) -> bool {
        false
    }

    async fn cancel_deferred(
        &self,
        _model: Model,
        _handle: DeferredHandle,
        _options: DeferredCancelOptions,
    ) -> Result<()> {
        Err(models_error(
            ModelsErrorCode::Provider,
            format!("Provider {} does not support deferred responses", self.id()),
        ))
    }

    /// Whether [`Provider::generate_images`] is implemented (providers with
    /// dedicated image models).
    fn supports_generate_images(&self) -> bool {
        false
    }

    /// Present when the provider supports dedicated image models. Never
    /// fails: errors are reported in the result.
    async fn generate_images(
        &self,
        model: ImageModel,
        _context: ImagesContext,
        _options: ImagesOptions,
    ) -> AssistantImages {
        image_error_result(
            &model,
            ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Provider {} does not support image generation", self.id()),
            ),
            false,
        )
    }

    /// Whether [`Provider::classify`] is implemented (providers with
    /// structured classifier models).
    fn supports_classify(&self) -> bool {
        false
    }

    /// Present when the provider supports structured classifier models.
    /// Never fails: errors are reported in the result.
    async fn classify(
        &self,
        model: ClassifierModel,
        _context: ClassifierContext,
        _options: ClassifierOptions,
    ) -> ClassifierResult {
        classifier_error_result(
            &model,
            ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Provider {} does not support classification", self.id()),
            ),
            false,
        )
    }

    /// Whether [`Provider::embed`] is implemented (providers with embedding
    /// models). ai.rs extra.
    fn supports_embed(&self) -> bool {
        false
    }

    /// Present when the provider supports embedding models. Never fails:
    /// errors are reported in the result. ai.rs extra.
    async fn embed(
        &self,
        model: EmbeddingModel,
        _context: EmbeddingsContext,
        _options: EmbeddingsOptions,
    ) -> EmbeddingsResult {
        embeddings_error_result(
            &model,
            ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Provider {} does not support embeddings", self.id()),
            ),
            false,
        )
    }
}

#[derive(Clone, Default)]
pub struct CreateModelsOptions {
    pub credentials: Option<Arc<dyn CredentialStore>>,
    pub models_store: Option<Arc<dyn ModelsStore>>,
    pub auth_context: Option<Arc<dyn AuthContext>>,
}

fn merge_headers(
    base: Option<&ProviderHeaders>,
    override_headers: Option<&ProviderHeaders>,
) -> Option<ProviderHeaders> {
    if base.is_none() && override_headers.is_none() {
        return None;
    }
    let mut merged = base.cloned().unwrap_or_default();
    for (name, value) in override_headers.into_iter().flatten() {
        let lower_name = name.to_lowercase();
        let existing: Vec<String> = merged
            .iter()
            .filter(|(existing, _)| existing.to_lowercase() == lower_name)
            .map(|(existing, _)| existing.clone())
            .collect();
        for existing in existing {
            merged.shift_remove(&existing);
        }
        merged.insert(name.clone(), value.clone());
    }
    Some(merged)
}

fn model_headers(model: &Model) -> Option<ProviderHeaders> {
    model.headers.as_ref().map(|headers| headers.clone().into())
}

/// The auth-relevant fields shared by every request option type.
trait AuthRequestOptions: Send + 'static {
    fn api_key(&self) -> Option<&String>;
    fn env(&self) -> Option<&ProviderEnv>;
    fn signal(&self) -> Option<&CancellationToken>;
    fn headers(&self) -> Option<&ProviderHeaders>;
    fn set_auth(
        &mut self,
        api_key: Option<String>,
        headers: Option<ProviderHeaders>,
        env: Option<ProviderEnv>,
    );
}

macro_rules! impl_auth_request_options {
    ($type:ty $(, $field:ident)?) => {
        impl AuthRequestOptions for $type {
            fn api_key(&self) -> Option<&String> {
                self$(.$field)?.api_key.as_ref()
            }
            fn env(&self) -> Option<&ProviderEnv> {
                self$(.$field)?.env.as_ref()
            }
            fn signal(&self) -> Option<&CancellationToken> {
                self$(.$field)?.signal.as_ref()
            }
            fn headers(&self) -> Option<&ProviderHeaders> {
                self$(.$field)?.headers.as_ref()
            }
            fn set_auth(
                &mut self,
                api_key: Option<String>,
                headers: Option<ProviderHeaders>,
                env: Option<ProviderEnv>,
            ) {
                self$(.$field)?.api_key = api_key;
                self$(.$field)?.headers = headers;
                self$(.$field)?.env = env;
            }
        }
    };
}

impl_auth_request_options!(StreamOptions);
impl_auth_request_options!(ProviderRequestOptions);
impl_auth_request_options!(SimpleStreamOptions, stream);
impl_auth_request_options!(DeferredFetchOptions, request);
impl_auth_request_options!(ImagesOptions);
impl_auth_request_options!(ClassifierOptions);
impl_auth_request_options!(EmbeddingsOptions);

/// What [`Models::get_auth`] resolves auth for: a provider id, or a model
/// (provider auth plus the model's static headers).
#[derive(Debug, Clone)]
pub struct AuthTarget {
    pub provider: String,
    pub model_headers: Option<ProviderHeaders>,
    is_model: bool,
}

impl From<&str> for AuthTarget {
    fn from(provider: &str) -> Self {
        Self {
            provider: provider.to_string(),
            model_headers: None,
            is_model: false,
        }
    }
}

impl From<&String> for AuthTarget {
    fn from(provider: &String) -> Self {
        provider.as_str().into()
    }
}

impl From<&Model> for AuthTarget {
    fn from(model: &Model) -> Self {
        Self {
            provider: model.provider.clone(),
            model_headers: model_headers(model),
            is_model: true,
        }
    }
}

impl From<&ImageModel> for AuthTarget {
    fn from(model: &ImageModel) -> Self {
        Self {
            provider: model.provider.clone(),
            model_headers: model.headers.clone().map(Into::into),
            is_model: true,
        }
    }
}

impl From<&ClassifierModel> for AuthTarget {
    fn from(model: &ClassifierModel) -> Self {
        Self {
            provider: model.provider.clone(),
            model_headers: model.headers.clone().map(Into::into),
            is_model: true,
        }
    }
}

impl From<&EmbeddingModel> for AuthTarget {
    fn from(model: &EmbeddingModel) -> Self {
        Self {
            provider: model.provider.clone(),
            model_headers: model.headers.clone().map(Into::into),
            is_model: true,
        }
    }
}

impl From<&AnyModel> for AuthTarget {
    fn from(model: &AnyModel) -> Self {
        match model {
            AnyModel::Chat(model) => model.into(),
            AnyModel::Image(model) => model.into(),
            AnyModel::Classifier(model) => model.into(),
            AnyModel::Embedding(model) => model.into(),
        }
    }
}

#[derive(Default)]
struct RefreshState {
    generations: HashMap<String, u64>,
    controllers: HashMap<String, (u64, CancellationToken)>,
}

struct ModelsInner {
    providers: RwLock<IndexMap<String, Arc<dyn Provider>>>,
    credentials: Arc<dyn CredentialStore>,
    models_store: Arc<dyn ModelsStore>,
    auth_context: Arc<dyn AuthContext>,
    refresh: Mutex<RefreshState>,
    publication_chains: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

/// Runtime collection of providers plus auth application and request
/// convenience. Providers own request behavior; `Models` resolves auth and
/// delegates each request to the provider that owns the model.
///
/// Read accessors come in three flavors: the unqualified ones (`get_models`,
/// `get_model`, `get_available`) return chat models, the `*_of_type`
/// accessors return one model type, and `get_all_models`/`get_all_available`
/// return every type.
#[derive(Clone)]
pub struct Models {
    inner: Arc<ModelsInner>,
}

impl fmt::Debug for Models {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Models")
            .field(
                "providers",
                &self.inner.providers.read().keys().collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

/// `createModels()`.
pub fn create_models(options: CreateModelsOptions) -> Models {
    Models::new(options)
}

impl Default for Models {
    fn default() -> Self {
        Self::new(CreateModelsOptions::default())
    }
}

impl Models {
    pub fn new(options: CreateModelsOptions) -> Self {
        Self {
            inner: Arc::new(ModelsInner {
                providers: RwLock::new(IndexMap::new()),
                credentials: options
                    .credentials
                    .unwrap_or_else(|| Arc::new(InMemoryCredentialStore::new())),
                models_store: options
                    .models_store
                    .unwrap_or_else(|| Arc::new(InMemoryModelsStore::new())),
                auth_context: options
                    .auth_context
                    .unwrap_or_else(default_provider_auth_context),
                refresh: Mutex::new(RefreshState::default()),
                publication_chains: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Whether two handles share one collection.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Upsert/replace by provider id. Provider ids are unique.
    pub fn set_provider(&self, provider: Arc<dyn Provider>) {
        let id = provider.id().to_string();
        self.supersede_provider_refresh(&id);
        self.inner.providers.write().insert(id, provider);
    }

    pub fn delete_provider(&self, id: &str) {
        self.supersede_provider_refresh(id);
        self.inner.providers.write().shift_remove(id);
    }

    pub fn clear_providers(&self) {
        let mut ids: Vec<String> = self.inner.providers.read().keys().cloned().collect();
        ids.extend(self.inner.refresh.lock().controllers.keys().cloned());
        let mut seen = HashSet::new();
        for id in ids {
            if seen.insert(id.clone()) {
                self.supersede_provider_refresh(&id);
            }
        }
        self.inner.providers.write().clear();
    }

    pub fn get_providers(&self) -> Vec<Arc<dyn Provider>> {
        self.inner.providers.read().values().cloned().collect()
    }

    pub fn get_provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.inner.providers.read().get(id).cloned()
    }

    /// Sync read of last-known chat models from one provider or all providers.
    /// Best-effort: a provider whose `get_models()` fails yields no models.
    pub fn get_models(&self, provider: Option<&str>) -> Vec<Model> {
        match provider {
            Some(provider) => self
                .get_provider(provider)
                .and_then(|entry| entry.get_models().ok())
                .unwrap_or_default(),
            None => self
                .get_providers()
                .iter()
                .flat_map(|entry| entry.get_models().unwrap_or_default())
                .collect(),
        }
    }

    /// Sync read of last-known models of every type from one provider or all providers.
    pub fn get_all_models(&self, provider: Option<&str>) -> Vec<AnyModel> {
        match provider {
            Some(provider) => self
                .get_provider(provider)
                .and_then(|entry| entry.get_all_models().ok())
                .unwrap_or_default(),
            None => self
                .get_providers()
                .iter()
                .flat_map(|entry| entry.get_all_models().unwrap_or_default())
                .collect(),
        }
    }

    /// Sync read of last-known models of one type from one provider or all providers.
    pub fn get_models_of_type(
        &self,
        model_type: ModelType,
        provider: Option<&str>,
    ) -> Vec<AnyModel> {
        self.get_all_models(provider)
            .into_iter()
            .filter(|model| is_model_type(model, model_type))
            .collect()
    }

    /// Sync runtime chat model lookup against last-known lists.
    pub fn get_model(&self, provider: &str, id: &str) -> Option<Model> {
        self.get_models(Some(provider))
            .into_iter()
            .find(|model| model.id == id)
    }

    /// Sync runtime lookup of a model of one type against last-known lists.
    pub fn get_model_of_type(
        &self,
        model_type: ModelType,
        provider: &str,
        id: &str,
    ) -> Option<AnyModel> {
        self.get_models_of_type(model_type, Some(provider))
            .into_iter()
            .find(|model| model.id() == id)
    }

    fn supersede_provider_refresh(&self, provider_id: &str) -> u64 {
        let mut state = self.inner.refresh.lock();
        let generation = state.generations.get(provider_id).copied().unwrap_or(0) + 1;
        state
            .generations
            .insert(provider_id.to_string(), generation);
        if let Some((_, previous)) = state.controllers.remove(provider_id) {
            previous.cancel();
        }
        generation
    }

    fn begin_provider_refresh(
        &self,
        provider_id: &str,
        caller_signal: &CancellationToken,
    ) -> (u64, CancellationToken) {
        let generation = self.supersede_provider_refresh(provider_id);
        // `AbortSignal.any([callerSignal, controller.signal])`: a child token
        // aborts with the caller and on its own when superseded.
        let controller = caller_signal.child_token();
        self.inner
            .refresh
            .lock()
            .controllers
            .insert(provider_id.to_string(), (generation, controller.clone()));
        (generation, controller)
    }

    fn current_generation(&self, provider_id: &str) -> Option<u64> {
        self.inner
            .refresh
            .lock()
            .generations
            .get(provider_id)
            .copied()
    }

    fn publication_chain(&self, provider_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        self.inner
            .publication_chains
            .lock()
            .entry(provider_id.to_string())
            .or_default()
            .clone()
    }

    async fn publish_provider_models(
        &self,
        provider_id: String,
        generation: u64,
        signal: CancellationToken,
        publication: ModelsPublication,
    ) -> Result<bool> {
        let chain = self.publication_chain(&provider_id);
        let queued = async {
            let _guard = chain.lock().await;
            if signal.is_cancelled() || self.current_generation(&provider_id) != Some(generation) {
                return Ok(false);
            }
            let store_options = ModelsStoreOperationOptions {
                signal: Some(signal.clone()),
            };
            match publication.persist {
                Some(None) => {
                    self.inner
                        .models_store
                        .delete(&provider_id, store_options)
                        .await?
                }
                Some(Some(entry)) => {
                    self.inner
                        .models_store
                        .write(&provider_id, entry, store_options)
                        .await?
                }
                None => {}
            }
            if signal.is_cancelled() || self.current_generation(&provider_id) != Some(generation) {
                return Ok(false);
            }
            if let Some(update) = publication.update {
                update();
            }
            Ok(true)
        };
        race_with_abort_signal(queued, &signal).await
    }

    async fn run_provider_refresh_phase(
        &self,
        provider: &Arc<dyn Provider>,
        credential: Option<Credential>,
        allow_network: bool,
        force: Option<bool>,
        generation: u64,
        signal: &CancellationToken,
    ) -> Result<()> {
        let stored = self
            .inner
            .models_store
            .read(
                provider.id(),
                ModelsStoreOperationOptions {
                    signal: Some(signal.clone()),
                },
            )
            .await?;
        let models = self.clone();
        let provider_id = provider.id().to_string();
        let publish_signal = signal.clone();
        let publisher: Publisher = Arc::new(move |publication| {
            let models = models.clone();
            let provider_id = provider_id.clone();
            let signal = publish_signal.clone();
            Box::pin(async move {
                models
                    .publish_provider_models(provider_id, generation, signal, publication)
                    .await
            })
        });
        provider
            .refresh_models(RefreshModelsContext {
                credential,
                stored,
                publisher,
                allow_network,
                force: if allow_network { force } else { None },
                signal: signal.clone(),
            })
            .await
    }

    /// Refresh selected configured dynamic providers concurrently (all when
    /// `providers` is omitted). Provider errors and cancellation are returned
    /// without failing; static, unknown, and unconfigured providers are
    /// skipped.
    pub async fn refresh(&self, options: ModelsRefreshOptions) -> ModelsRefreshResult {
        let allow_network = options.allow_network.unwrap_or(true);
        let caller_signal = operation_signal(options.signal.as_ref());
        let errors: Mutex<IndexMap<String, Error>> = Mutex::new(IndexMap::new());
        if caller_signal.is_cancelled() {
            return ModelsRefreshResult {
                aborted: true,
                errors: IndexMap::new(),
            };
        }
        let selected: Option<HashSet<&String>> = options
            .providers
            .as_ref()
            .map(|providers| providers.iter().collect());
        let refreshable: Vec<Arc<dyn Provider>> = self
            .get_providers()
            .into_iter()
            .filter(|provider| {
                provider.supports_refresh_models()
                    && selected
                        .as_ref()
                        .is_none_or(|selected| selected.contains(&provider.id().to_string()))
            })
            .collect();

        let refresh = join_all(refreshable.iter().map(|provider| {
            let errors = &errors;
            let caller_signal = &caller_signal;
            let force = options.force;
            async move {
                let provider_id = provider.id().to_string();
                let (generation, signal) = self.begin_provider_refresh(&provider_id, caller_signal);
                let operation = async {
                    let (stored_credential, credential_error) =
                        match read_credential(&self.inner.credentials, &provider_id, &signal).await
                        {
                            Ok(credential) => (credential, None),
                            Err(error) => (None, Some(error)),
                        };

                    // Restore cached provider state before auth resolution or network access.
                    self.run_provider_refresh_phase(
                        provider,
                        stored_credential.clone(),
                        false,
                        None,
                        generation,
                        &signal,
                    )
                    .await?;
                    if let Some(error) = credential_error {
                        return Err(error);
                    }
                    if !allow_network || signal.is_cancelled() {
                        return Ok(());
                    }

                    let Some(credential) = self
                        .resolve_refresh_credential(provider, stored_credential, &signal)
                        .await?
                    else {
                        return Ok(());
                    };
                    self.run_provider_refresh_phase(
                        provider,
                        Some(credential),
                        true,
                        force,
                        generation,
                        &signal,
                    )
                    .await
                };

                if let Err(error) = race_with_abort_signal(operation, &signal).await
                    && !signal.is_cancelled()
                {
                    errors.lock().insert(provider_id.clone(), error);
                }
                let mut state = self.inner.refresh.lock();
                if state
                    .controllers
                    .get(&provider_id)
                    .is_some_and(|(current, _)| *current == generation)
                {
                    state.controllers.remove(&provider_id);
                }
            }
        }));

        let _ = race_with_abort_signal(
            async {
                refresh.await;
                Ok(())
            },
            &caller_signal,
        )
        .await;

        ModelsRefreshResult {
            aborted: caller_signal.is_cancelled(),
            errors: errors.into_inner(),
        }
    }

    async fn resolve_refresh_credential(
        &self,
        provider: &Arc<dyn Provider>,
        stored: Option<Credential>,
        signal: &CancellationToken,
    ) -> Result<Option<Credential>> {
        if let Some(Credential::OAuth(stored)) = stored {
            let Some(oauth) = provider.auth().oauth.clone() else {
                return Ok(None);
            };
            if now_millis() < stored.expires {
                return Ok(Some(Credential::OAuth(stored)));
            }
            if signal.is_cancelled() {
                return Ok(None);
            }
            let refresh_signal = signal.clone();
            let post = self
                .inner
                .credentials
                .modify(
                    provider.id(),
                    Box::new(move |current| {
                        Box::pin(async move {
                            match current {
                                Some(Credential::OAuth(current))
                                    if now_millis() >= current.expires =>
                                {
                                    Ok(Some(Credential::OAuth(
                                        oauth.refresh(current, refresh_signal).await?,
                                    )))
                                }
                                _ => Ok(None),
                            }
                        })
                    }),
                    AuthOperationOptions::with_signal(signal),
                )
                .await?;
            return Ok(post.filter(|post| matches!(post, Credential::OAuth(_))));
        }

        let Some(api_key) = provider.auth().api_key.clone() else {
            return Ok(None);
        };
        let credential = match stored {
            Some(Credential::ApiKey(stored)) => Some(stored),
            _ => None,
        };
        let result = api_key
            .resolve(ApiKeyAuthInput {
                ctx: self.inner.auth_context.clone(),
                credential,
                signal: signal.clone(),
            })
            .await?;
        Ok(result.map(|result| {
            Credential::ApiKey(crate::auth::ApiKeyCredential {
                key: result.auth.api_key,
                env: result.env,
            })
        }))
    }

    async fn check_provider_auth(
        &self,
        provider: &Arc<dyn Provider>,
        credential: Option<&Credential>,
        signal: &CancellationToken,
    ) -> Result<Option<AuthCheck>> {
        if let Some(Credential::OAuth(_)) = credential {
            return Ok(provider.auth().oauth.is_some().then(|| AuthCheck {
                source: Some("OAuth".to_string()),
                auth_type: AuthType::OAuth,
            }));
        }
        let Some(api_key) = provider.auth().api_key.clone() else {
            return Ok(None);
        };
        if api_key.supports_check() {
            return api_key
                .check(ApiKeyAuthInput {
                    ctx: self.inner.auth_context.clone(),
                    credential: credential.and_then(Credential::as_api_key).cloned(),
                    signal: signal.clone(),
                })
                .await
                .map_err(|error| {
                    models_error_with_cause(
                        ModelsErrorCode::Auth,
                        format!("API key auth check failed for provider {}", provider.id()),
                        &error,
                    )
                });
        }

        let resolution = resolve_provider_auth(
            provider.id(),
            provider.auth(),
            &self.inner.credentials,
            &self.inner.auth_context,
            AuthResolutionOverrides {
                signal: Some(signal.clone()),
                ..Default::default()
            },
        )
        .await?;
        Ok(resolution.map(|resolution| AuthCheck {
            source: resolution.source,
            auth_type: AuthType::ApiKey,
        }))
    }

    /// Check whether a provider has complete auth configuration without refreshing OAuth.
    pub async fn check_auth(
        &self,
        provider_id: &str,
        options: AuthOperationOptions,
    ) -> Result<Option<AuthCheck>> {
        let signal = operation_signal(options.signal.as_ref());
        let check = async {
            throw_if_aborted(&signal)?;
            let Some(provider) = self.get_provider(provider_id) else {
                return Ok(None);
            };
            let credential = read_credential(&self.inner.credentials, provider_id, &signal).await?;
            self.check_provider_auth(&provider, credential.as_ref(), &signal)
                .await
        };
        race_with_abort_signal(check, &signal).await
    }

    async fn get_authenticated_providers(
        &self,
        provider_id: Option<&str>,
        signal: &CancellationToken,
    ) -> Result<Vec<(Arc<dyn Provider>, Option<Credential>)>> {
        throw_if_aborted(signal)?;
        let providers = match provider_id {
            Some(provider_id) => self.get_provider(provider_id).into_iter().collect(),
            None => self.get_providers(),
        };
        let checks = try_join_all(providers.into_iter().map(|provider| async move {
            let credential =
                read_credential(&self.inner.credentials, provider.id(), signal).await?;
            let auth = self
                .check_provider_auth(&provider, credential.as_ref(), signal)
                .await?;
            Ok::<_, Error>((provider, credential, auth))
        }))
        .await?;
        Ok(checks
            .into_iter()
            .filter(|(_, _, auth)| auth.is_some())
            .map(|(provider, credential, _)| (provider, credential))
            .collect())
    }

    /// Return chat models whose providers have complete auth configuration.
    pub async fn get_available(
        &self,
        provider_id: Option<&str>,
        options: AuthOperationOptions,
    ) -> Result<Vec<Model>> {
        let signal = operation_signal(options.signal.as_ref());
        let available = async {
            let providers = self
                .get_authenticated_providers(provider_id, &signal)
                .await?;
            let mut available = Vec::new();
            for (provider, credential) in providers {
                let models = provider.get_models()?;
                available.extend(
                    provider
                        .filter_models(&models, credential.as_ref())
                        .unwrap_or(models),
                );
            }
            Ok(available)
        };
        race_with_abort_signal(available, &signal).await
    }

    /// Return models of one type whose providers have complete auth configuration.
    pub async fn get_available_of_type(
        &self,
        model_type: ModelType,
        provider_id: Option<&str>,
        options: AuthOperationOptions,
    ) -> Result<Vec<AnyModel>> {
        Ok(self
            .get_all_available(provider_id, options)
            .await?
            .into_iter()
            .filter(|model| is_model_type(model, model_type))
            .collect())
    }

    /// Return models of every type whose providers have complete auth configuration.
    pub async fn get_all_available(
        &self,
        provider_id: Option<&str>,
        options: AuthOperationOptions,
    ) -> Result<Vec<AnyModel>> {
        let signal = operation_signal(options.signal.as_ref());
        let available = async {
            let providers = self
                .get_authenticated_providers(provider_id, &signal)
                .await?;
            let mut available = Vec::new();
            for (provider, credential) in providers {
                let models = provider.get_all_models()?;
                if let Some(filtered) = provider.filter_all_models(&models, credential.as_ref()) {
                    available.extend(filtered);
                    continue;
                }
                let chat_models = provider.get_models()?;
                let Some(filtered) = provider.filter_models(&chat_models, credential.as_ref())
                else {
                    available.extend(models);
                    continue;
                };
                let available_chat_ids: HashSet<String> =
                    filtered.into_iter().map(|model| model.id).collect();
                available.extend(models.into_iter().filter(|model| {
                    !is_model_type(model, ModelType::Chat)
                        || available_chat_ids.contains(model.id())
                }));
            }
            Ok(available)
        };
        race_with_abort_signal(available, &signal).await
    }

    /// Resolve provider-scoped auth by provider id, or provider auth plus
    /// static model headers when passed a model. Includes a source label for
    /// status UI. Resolves `None` when the provider is unknown or
    /// unconfigured. Fails with `ModelsError`: code "oauth" when a token
    /// refresh fails (the stored credential is preserved for retry), code
    /// "auth" when api-key resolution or the credential store fails.
    pub async fn get_auth(
        &self,
        target: impl Into<AuthTarget>,
        overrides: AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>> {
        let target = target.into();
        let signal = operation_signal(overrides.signal.as_ref());
        let Some(provider) = self.get_provider(&target.provider) else {
            return Ok(None);
        };
        let result = resolve_provider_auth(
            provider.id(),
            provider.auth(),
            &self.inner.credentials,
            &self.inner.auth_context,
            AuthResolutionOverrides {
                signal: Some(signal),
                ..overrides
            },
        )
        .await?;
        let Some(mut result) = result else {
            return Ok(None);
        };
        if !target.is_model {
            return Ok(Some(result));
        }
        let Some(headers) = target.model_headers else {
            return Ok(Some(result));
        };
        result.auth.headers = merge_headers(result.auth.headers.as_ref(), Some(&headers));
        Ok(Some(result))
    }

    /// Run a provider-owned login flow and persist its returned credential.
    pub async fn login(
        &self,
        provider_id: &str,
        auth_type: AuthType,
        interaction: Arc<dyn AuthInteraction>,
        options: LoginOptions,
    ) -> Result<Credential> {
        let signal = operation_signal(interaction.signal().as_ref());
        throw_if_aborted(&signal)?;
        let Some(provider) = self.get_provider(provider_id) else {
            return Err(models_error(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {provider_id}"),
            ));
        };
        let provider_interaction = ProviderAuthInteraction {
            interaction,
            signal: signal.clone(),
        };
        let unsupported = || {
            models_error(
                ModelsErrorCode::Auth,
                format!("{} does not support {auth_type} login", provider.name()),
            )
        };
        let credential = match auth_type {
            AuthType::OAuth => {
                let Some(oauth) = provider.auth().oauth.clone() else {
                    return Err(unsupported());
                };
                Credential::OAuth(
                    race_with_abort_signal(oauth.login(provider_interaction, options), &signal)
                        .await?,
                )
            }
            AuthType::ApiKey => {
                let Some(api_key) = provider
                    .auth()
                    .api_key
                    .clone()
                    .filter(|api_key| api_key.supports_login())
                else {
                    return Err(unsupported());
                };
                Credential::ApiKey(
                    race_with_abort_signal(api_key.login(provider_interaction), &signal).await?,
                )
            }
        };
        let stored = credential.clone();
        // Pi races the store mutation against the signal only until the
        // modifier starts: a store that ignores the signal while the mutation
        // is queued (e.g. waiting on a file lock) cannot hold up an aborted
        // login, but a started write is awaited.
        let mutation_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let started = mutation_started.clone();
        let mutation = self.inner.credentials.modify(
            provider_id,
            Box::new(move |_| {
                Box::pin(async move {
                    started.store(true, std::sync::atomic::Ordering::SeqCst);
                    Ok(Some(stored))
                })
            }),
            AuthOperationOptions::with_signal(&signal),
        );
        let mut mutation = std::pin::pin!(mutation);
        let mutation = tokio::select! {
            result = &mut mutation => result,
            _ = signal.cancelled() => {
                if mutation_started.load(std::sync::atomic::Ordering::SeqCst) {
                    mutation.await
                } else {
                    // The queued mutation is dropped (abandoned operations are
                    // dropped, not left running).
                    throw_if_aborted(&signal).map(|()| None)
                }
            }
        };
        if let Err(error) = mutation {
            throw_if_aborted(&signal)?;
            return Err(models_error_with_cause(
                ModelsErrorCode::Auth,
                format!("Credential store modify failed for {provider_id}"),
                &error,
            ));
        }
        Ok(credential)
    }

    /// Remove the stored credential for a provider.
    pub async fn logout(&self, provider_id: &str, options: AuthOperationOptions) -> Result<()> {
        let signal = operation_signal(options.signal.as_ref());
        throw_if_aborted(&signal)?;
        if let Err(error) = self
            .inner
            .credentials
            .delete(provider_id, AuthOperationOptions::with_signal(&signal))
            .await
        {
            throw_if_aborted(&signal)?;
            return Err(models_error_with_cause(
                ModelsErrorCode::Auth,
                format!("Credential store delete failed for {provider_id}"),
                &error,
            ));
        }
        Ok(())
    }

    fn require_provider(&self, provider_id: &str) -> Result<Arc<dyn Provider>> {
        self.get_provider(provider_id).ok_or_else(|| {
            models_error(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {provider_id}"),
            )
        })
    }

    fn require_chat_provider(&self, model: &Model) -> Result<Arc<dyn Provider>> {
        assert_chat_model(&AnyModel::Chat(model.clone()))?;
        self.require_provider(&model.provider)
    }

    async fn apply_auth<T: AuthRequestOptions>(
        &self,
        model: &Model,
        options: ModelsOptions<T>,
    ) -> Result<(Model, T)> {
        let (base_url, request_options) = self.apply_auth_to(model.into(), options).await?;
        let mut request_model = model.clone();
        if let Some(base_url) = base_url {
            request_model.base_url = base_url;
        }
        Ok((request_model, request_options))
    }

    /// `applyAuth()` for any model type: the resolved base URL override and
    /// the request options with auth merged in.
    async fn apply_auth_to<T: AuthRequestOptions>(
        &self,
        target: AuthTarget,
        options: ModelsOptions<T>,
    ) -> Result<(Option<String>, T)> {
        let provider_id = target.provider.clone();
        self.require_provider(&provider_id)?;
        let ModelsOptions {
            options: mut request_options,
            transform_headers,
        } = options;
        let resolution = self
            .get_auth(
                target,
                AuthResolutionOverrides {
                    api_key: request_options.api_key().cloned(),
                    env: request_options.env().cloned(),
                    signal: request_options.signal().cloned(),
                    ..Default::default()
                },
            )
            .await?;
        let Some(resolution) = resolution else {
            return Err(models_error(
                ModelsErrorCode::Auth,
                format!("Provider is not configured: {provider_id}"),
            ));
        };
        let auth = resolution.auth;

        // Explicit request options win per-field; the Models-only transform runs last.
        let api_key = request_options.api_key().cloned().or(auth.api_key);
        let mut headers = merge_headers(auth.headers.as_ref(), request_options.headers());
        if let Some(transform_headers) = transform_headers {
            headers = Some(transform_headers(headers.unwrap_or_default()).await?);
        }
        let env = if resolution.env.is_some() || request_options.env().is_some() {
            let mut env = resolution.env.unwrap_or_default();
            env.extend(request_options.env().cloned().unwrap_or_default());
            Some(env)
        } else {
            None
        };
        let base_url = auth.base_url.filter(|base_url| !base_url.is_empty());
        request_options.set_auth(api_key, headers, env);
        Ok((base_url, request_options))
    }

    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: impl Into<ModelsApiStreamOptions>,
    ) -> AssistantMessageEventStream {
        let transcript = normalize_context(context);
        let models = self.clone();
        let request = model.clone();
        let options = options.into();
        lazy_stream(model, async move {
            let provider = models.require_chat_provider(&request)?;
            let (request_model, request_options) = models.apply_auth(&request, options).await?;
            Ok(provider.stream(request_model, transcript, request_options))
        })
    }

    pub async fn complete(
        &self,
        model: &Model,
        context: &Context,
        options: impl Into<ModelsApiStreamOptions>,
    ) -> AssistantMessage {
        self.stream(model, context, options).result().await
    }

    pub fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: impl Into<ModelsSimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let transcript = normalize_context(context);
        let models = self.clone();
        let request = model.clone();
        let options = options.into();
        lazy_stream(model, async move {
            let provider = models.require_chat_provider(&request)?;
            let (request_model, request_options) = models.apply_auth(&request, options).await?;
            Ok(provider.stream_simple(request_model, transcript, request_options))
        })
    }

    pub async fn complete_simple(
        &self,
        model: &Model,
        context: &Context,
        options: impl Into<ModelsSimpleStreamOptions>,
    ) -> AssistantMessage {
        self.stream_simple(model, context, options).result().await
    }

    pub fn stream_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: impl Into<ModelsDeferredFetchOptions>,
    ) -> AssistantMessageEventStream {
        let models = self.clone();
        let request = model.clone();
        let handle = handle.clone();
        let options = options.into();
        lazy_stream(model, async move {
            let provider = models.require_chat_provider(&request)?;
            if !provider.supports_fetch_deferred() {
                return Err(models_error(
                    ModelsErrorCode::Provider,
                    format!(
                        "Provider {} does not support deferred responses",
                        request.provider
                    ),
                ));
            }
            let (request_model, request_options) = models.apply_auth(&request, options).await?;
            Ok(provider.fetch_deferred(request_model, handle, request_options))
        })
    }

    pub async fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: impl Into<ModelsDeferredFetchOptions>,
    ) -> AssistantMessage {
        self.stream_deferred(model, handle, options).result().await
    }

    pub async fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: impl Into<ModelsDeferredCancelOptions>,
    ) -> Result<()> {
        let provider = self.require_chat_provider(model)?;
        if !provider.supports_cancel_deferred() {
            return Err(models_error(
                ModelsErrorCode::Provider,
                format!(
                    "Provider {} does not support deferred responses",
                    model.provider
                ),
            ));
        }
        let (request_model, request_options) = self.apply_auth(model, options.into()).await?;
        provider
            .cancel_deferred(request_model, handle.clone(), request_options)
            .await
    }

    /// Generate images through the owning provider with auth resolved like
    /// `stream()`. Never fails: unknown providers, unconfigured auth, and
    /// providers without `generate_images` return an error `AssistantImages`.
    pub async fn generate_images(
        &self,
        model: &ImageModel,
        context: &ImagesContext,
        options: impl Into<ModelsImagesOptions>,
    ) -> AssistantImages {
        let options = options.into();
        let aborted = options
            .options
            .signal
            .clone()
            .map(|signal| move || signal.is_cancelled());
        let result = async {
            let provider = self.require_provider(&model.provider)?;
            if !provider.supports_generate_images() {
                return Err(models_error(
                    ModelsErrorCode::Provider,
                    format!(
                        "Provider {} does not support image generation",
                        model.provider
                    ),
                ));
            }
            let (base_url, request_options) = self.apply_auth_to(model.into(), options).await?;
            let mut request_model = model.clone();
            if let Some(base_url) = base_url {
                request_model.base_url = base_url;
            }
            Ok(provider
                .generate_images(request_model, context.clone(), request_options)
                .await)
        }
        .await;
        result.unwrap_or_else(|error| {
            image_error_result(model, error, aborted.is_some_and(|aborted| aborted()))
        })
    }

    /// Classify structured state through the owning provider. Never fails:
    /// errors are reported in the result.
    pub async fn classify(
        &self,
        model: &ClassifierModel,
        context: &ClassifierContext,
        options: impl Into<ModelsClassifierOptions>,
    ) -> ClassifierResult {
        let options = options.into();
        let aborted = options
            .options
            .signal
            .clone()
            .map(|signal| move || signal.is_cancelled());
        let result = async {
            let provider = self.require_provider(&model.provider)?;
            if !provider.supports_classify() {
                return Err(models_error(
                    ModelsErrorCode::Provider,
                    format!(
                        "Provider {} does not support classification",
                        model.provider
                    ),
                ));
            }
            let (base_url, request_options) = self.apply_auth_to(model.into(), options).await?;
            let mut request_model = model.clone();
            if let Some(base_url) = base_url {
                request_model.base_url = base_url;
            }
            Ok(provider
                .classify(request_model, context.clone(), request_options)
                .await)
        }
        .await;
        result.unwrap_or_else(|error| {
            classifier_error_result(model, error, aborted.is_some_and(|aborted| aborted()))
        })
    }

    /// Embed `context.input` through the owning provider with auth resolved
    /// like `stream()`. Never fails: unknown providers, unconfigured auth,
    /// and providers without `embed` return an error [`EmbeddingsResult`].
    /// ai.rs extra, the embedding counterpart of [`Models::generate_images`].
    pub async fn embed(
        &self,
        model: &EmbeddingModel,
        context: &EmbeddingsContext,
        options: impl Into<ModelsEmbeddingsOptions>,
    ) -> EmbeddingsResult {
        let options = options.into();
        let aborted = options
            .options
            .signal
            .clone()
            .map(|signal| move || signal.is_cancelled());
        let result = async {
            let provider = self.require_provider(&model.provider)?;
            if !provider.supports_embed() {
                return Err(models_error(
                    ModelsErrorCode::Provider,
                    format!("Provider {} does not support embeddings", model.provider),
                ));
            }
            let (base_url, request_options) = self.apply_auth_to(model.into(), options).await?;
            let mut request_model = model.clone();
            if let Some(base_url) = base_url {
                request_model.base_url = base_url;
            }
            Ok(provider
                .embed(request_model, context.clone(), request_options)
                .await)
        }
        .await;
        result.unwrap_or_else(|error| {
            embeddings_error_result(model, error, aborted.is_some_and(|aborted| aborted()))
        })
    }
}

/// `fetchModels`: fetch a dynamic model overlay of every type.
pub type FetchModels =
    Arc<dyn Fn(RefreshModelsContext) -> BoxFuture<Result<Vec<AnyModel>>> + Send + Sync>;
/// Credential-specific chat model availability. See [`Provider::filter_models`].
pub type FilterModels = Arc<dyn Fn(&[Model], Option<&Credential>) -> Vec<Model> + Send + Sync>;
/// Credential-specific availability across every model type.
pub type FilterAllModels =
    Arc<dyn Fn(&[AnyModel], Option<&Credential>) -> Vec<AnyModel> + Send + Sync>;

/// Chat implementation of a created provider: a single one for all chat
/// models, or a map keyed by `model.api` for mixed-API providers.
#[derive(Clone)]
pub enum ProviderApi {
    Single(Arc<dyn ProviderStreams>),
    ByApi(IndexMap<String, Arc<dyn ProviderStreams>>),
}

#[derive(Clone, Default)]
pub struct CreateProviderOptions {
    pub id: String,
    /// Display name. Default: `id`.
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub headers: Option<ProviderHeaders>,
    /// Required — every provider has auth semantics, even ambient/keyless ones.
    pub auth: ProviderAuth,
    /// Static baseline models of every type (empty for purely dynamic providers).
    pub models: Vec<AnyModel>,
    /// Fetch a dynamic model overlay of every type. The provider restores and
    /// publishes it transactionally.
    pub fetch_models: Option<FetchModels>,
    pub filter_models: Option<FilterModels>,
    pub filter_all_models: Option<FilterAllModels>,
    pub api: Option<ProviderApi>,
    /// Image-generation implementations keyed by `model.api`.
    pub images: Option<IndexMap<String, Arc<dyn ProviderImages>>>,
    /// Classifier implementations keyed by `model.api`.
    pub classifiers: Option<IndexMap<String, Arc<dyn ProviderClassifier>>>,
    /// Embeddings implementations keyed by `model.api`. ai.rs extra.
    pub embeddings: Option<IndexMap<String, Arc<dyn ProviderEmbeddings>>>,
}

struct CreatedProvider {
    id: String,
    name: String,
    base_url: Option<String>,
    headers: Option<ProviderHeaders>,
    auth: ProviderAuth,
    baseline_models: Vec<AnyModel>,
    dynamic_models: Arc<RwLock<Vec<AnyModel>>>,
    fetch_models: Option<FetchModels>,
    filter_models: Option<FilterModels>,
    filter_all_models: Option<FilterAllModels>,
    single: Option<Arc<dyn ProviderStreams>>,
    by_api: IndexMap<String, Arc<dyn ProviderStreams>>,
    images: Option<IndexMap<String, Arc<dyn ProviderImages>>>,
    classifiers: Option<IndexMap<String, Arc<dyn ProviderClassifier>>>,
    embeddings: Option<IndexMap<String, Arc<dyn ProviderEmbeddings>>>,
    fetch_deferred: bool,
    cancel_deferred: bool,
}

impl CreatedProvider {
    fn current_models(&self) -> Vec<AnyModel> {
        let mut merged = self.baseline_models.clone();
        for model in self.dynamic_models.read().iter() {
            let index = merged.iter().position(|entry| {
                get_model_type(entry) == get_model_type(model) && entry.id() == model.id()
            });
            match index {
                Some(index) => merged[index] = model.clone(),
                None => merged.push(model.clone()),
            }
        }
        merged
    }

    fn api_for(&self, model: &Model) -> Option<Arc<dyn ProviderStreams>> {
        self.single
            .clone()
            .or_else(|| self.by_api.get(&model.api).cloned())
    }

    fn missing_api(&self, model: &Model) -> AssistantMessageEventStream {
        error_stream(
            model,
            ModelsError::new(
                ModelsErrorCode::Stream,
                format!(
                    "Provider {} has no API implementation for \"{}\"",
                    self.id, model.api
                ),
            ),
        )
    }
}

#[async_trait]
impl Provider for CreatedProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    fn headers(&self) -> Option<&ProviderHeaders> {
        self.headers.as_ref()
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<Model>> {
        Ok(self
            .current_models()
            .into_iter()
            .filter_map(|model| match model {
                AnyModel::Chat(model) if model.model_type.is_none_or(|t| t == ModelType::Chat) => {
                    Some(model)
                }
                _ => None,
            })
            .collect())
    }

    fn get_all_models(&self) -> Result<Vec<AnyModel>> {
        Ok(self.current_models())
    }

    fn supports_refresh_models(&self) -> bool {
        self.fetch_models.is_some()
    }

    async fn refresh_models(&self, context: RefreshModelsContext) -> Result<()> {
        let Some(fetch_models) = self.fetch_models.clone() else {
            return Ok(());
        };
        if let Some(stored) = &context.stored {
            let restored: Vec<AnyModel> = stored
                .models
                .iter()
                .filter(|model| model.provider() == self.id)
                .cloned()
                .collect();
            let dynamic = self.dynamic_models.clone();
            if !context
                .publish(ModelsPublication {
                    persist: None,
                    update: Some(Box::new(move || *dynamic.write() = restored)),
                })
                .await?
            {
                return Ok(());
            }
        }
        if !context.allow_network || context.signal.is_cancelled() {
            return Ok(());
        }
        let refreshed = fetch_models(context.clone()).await?;
        if context.signal.is_cancelled() {
            return Ok(());
        }
        let dynamic = self.dynamic_models.clone();
        let published = refreshed.clone();
        context
            .publish(ModelsPublication {
                persist: Some(Some(ModelsStoreEntry {
                    models: refreshed,
                    checked_at: Some(now_millis()),
                    ..Default::default()
                })),
                update: Some(Box::new(move || *dynamic.write() = published)),
            })
            .await?;
        Ok(())
    }

    fn filter_models(
        &self,
        models: &[Model],
        credential: Option<&Credential>,
    ) -> Option<Vec<Model>> {
        self.filter_models
            .as_ref()
            .map(|filter| filter(models, credential))
    }

    fn filter_all_models(
        &self,
        models: &[AnyModel],
        credential: Option<&Credential>,
    ) -> Option<Vec<AnyModel>> {
        self.filter_all_models
            .as_ref()
            .map(|filter| filter(models, credential))
    }

    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        match self.api_for(&model) {
            Some(streams) => streams.stream(model, context, options),
            None => self.missing_api(&model),
        }
    }

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        match self.api_for(&model) {
            Some(streams) => streams.stream_simple(model, context, options),
            None => self.missing_api(&model),
        }
    }

    fn supports_fetch_deferred(&self) -> bool {
        self.fetch_deferred
    }

    fn fetch_deferred(
        &self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions,
    ) -> AssistantMessageEventStream {
        match self.api_for(&model) {
            Some(implementation) if implementation.supports_fetch_deferred() => {
                implementation.fetch_deferred(model, handle, options)
            }
            _ => {
                let message = format!(
                    "Provider {} does not support deferred responses for \"{}\"",
                    self.id, model.api
                );
                error_stream(&model, ModelsError::new(ModelsErrorCode::Provider, message))
            }
        }
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.cancel_deferred
    }

    async fn cancel_deferred(
        &self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions,
    ) -> Result<()> {
        match self.api_for(&model) {
            Some(implementation) if implementation.supports_cancel_deferred() => {
                implementation.cancel_deferred(model, handle, options).await
            }
            _ => Err(models_error(
                ModelsErrorCode::Provider,
                format!(
                    "Provider {} cannot cancel deferred responses for \"{}\"",
                    self.id, model.api
                ),
            )),
        }
    }

    fn supports_generate_images(&self) -> bool {
        self.images.is_some()
    }

    async fn generate_images(
        &self,
        model: ImageModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> AssistantImages {
        let implementation = self
            .images
            .as_ref()
            .and_then(|images| images.get(&model.api).cloned());
        match implementation {
            Some(implementation) => {
                implementation
                    .generate_images(model, context, options)
                    .await
            }
            None => {
                let message = format!(
                    "Provider {} has no image generation implementation for \"{}\"",
                    self.id, model.api
                );
                image_error_result(
                    &model,
                    ModelsError::new(ModelsErrorCode::Provider, message),
                    false,
                )
            }
        }
    }

    fn supports_classify(&self) -> bool {
        self.classifiers.is_some()
    }

    async fn classify(
        &self,
        model: ClassifierModel,
        context: ClassifierContext,
        options: ClassifierOptions,
    ) -> ClassifierResult {
        let implementation = self
            .classifiers
            .as_ref()
            .and_then(|classifiers| classifiers.get(&model.api).cloned());
        match implementation {
            Some(implementation) => implementation.classify(model, context, options).await,
            None => {
                let message = format!(
                    "Provider {} has no classifier implementation for \"{}\"",
                    self.id, model.api
                );
                classifier_error_result(
                    &model,
                    ModelsError::new(ModelsErrorCode::Provider, message),
                    false,
                )
            }
        }
    }

    fn supports_embed(&self) -> bool {
        self.embeddings.is_some()
    }

    async fn embed(
        &self,
        model: EmbeddingModel,
        context: EmbeddingsContext,
        options: EmbeddingsOptions,
    ) -> EmbeddingsResult {
        let implementation = self
            .embeddings
            .as_ref()
            .and_then(|embeddings| embeddings.get(&model.api).cloned());
        match implementation {
            Some(implementation) => implementation.embed(model, context, options).await,
            None => {
                let message = format!(
                    "Provider {} has no embeddings implementation for \"{}\"",
                    self.id, model.api
                );
                embeddings_error_result(
                    &model,
                    ModelsError::new(ModelsErrorCode::Provider, message),
                    false,
                )
            }
        }
    }
}

/// Builds a provider from parts. Built-in provider factories go through this.
/// A single `api` streams all chat models; an `api` map dispatches on
/// `model.api`, and a model whose api has no entry produces a stream error.
/// At least one concrete implementation is required; empty maps are rejected.
pub fn create_provider(input: CreateProviderOptions) -> Result<Arc<dyn Provider>> {
    let (single, by_api) = match input.api {
        Some(ProviderApi::Single(single)) => (Some(single), IndexMap::new()),
        Some(ProviderApi::ByApi(by_api)) => (None, by_api),
        None => (None, IndexMap::new()),
    };
    let streams: Vec<&Arc<dyn ProviderStreams>> = single.iter().chain(by_api.values()).collect();
    let images = input.images.filter(|images| !images.is_empty());
    let classifiers = input
        .classifiers
        .filter(|classifiers| !classifiers.is_empty());
    let embeddings = input.embeddings.filter(|embeddings| !embeddings.is_empty());
    if streams.is_empty() && images.is_none() && classifiers.is_none() && embeddings.is_none() {
        return Err(Error::message(format!(
            "Provider {}: at least one of \"api\", \"images\", \"classifiers\", or \"embeddings\" is required.",
            input.id
        )));
    }
    let fetch_deferred = streams.iter().any(|entry| entry.supports_fetch_deferred());
    let cancel_deferred = streams.iter().any(|entry| entry.supports_cancel_deferred());

    Ok(Arc::new(CreatedProvider {
        name: input.name.unwrap_or_else(|| input.id.clone()),
        id: input.id,
        base_url: input.base_url,
        headers: input.headers,
        auth: input.auth,
        baseline_models: input.models,
        dynamic_models: Arc::new(RwLock::new(Vec::new())),
        fetch_models: input.fetch_models,
        filter_models: input.filter_models,
        filter_all_models: input.filter_all_models,
        single,
        by_api,
        images,
        classifiers,
        embeddings,
        fetch_deferred,
        cancel_deferred,
    }))
}

/// Runtime-checked narrowing for dynamically looked-up models. Non-chat
/// models never match, even when their api id equals `api`.
pub fn has_api(model: &AnyModel, api: &str) -> bool {
    is_model_type(model, ModelType::Chat) && model.api() == api
}

/// `calculateCost()`: fill `usage.cost` from the model's rates and return it.
pub fn calculate_cost(model: &Model, usage: &mut Usage) -> UsageCost {
    calculate_cost_for(&model.cost, usage)
}

/// `calculateCost()` for any model type: Pi's `calculateCost(model: AnyModel,
/// usage)` reads only `model.cost`.
pub fn calculate_cost_for(cost: &ModelCost, usage: &mut Usage) -> UsageCost {
    let input_tokens =
        u64::from(usage.input) + u64::from(usage.cache_read) + u64::from(usage.cache_write);
    let mut rates = ModelCostRates {
        input: cost.input,
        output: cost.output,
        cache_read: cost.cache_read,
        cache_write: cost.cache_write,
    };
    let mut matched_threshold: i64 = -1;
    for tier in cost.tiers.iter().flatten() {
        let threshold = i64::from(tier.input_tokens_above);
        if input_tokens > u64::from(tier.input_tokens_above) && threshold > matched_threshold {
            rates = ModelCostRates {
                input: tier.input,
                output: tier.output,
                cache_read: tier.cache_read,
                cache_write: tier.cache_write,
            };
            matched_threshold = threshold;
        }
    }

    // Anthropic charges 2x base input for 1h cache writes.
    let long_write = f64::from(usage.cache_write_1h.unwrap_or(0));
    let short_write = f64::from(usage.cache_write) - long_write;
    usage.cost.input = (rates.input / 1_000_000.0) * f64::from(usage.input);
    usage.cost.output = (rates.output / 1_000_000.0) * f64::from(usage.output);
    usage.cost.cache_read = (rates.cache_read / 1_000_000.0) * f64::from(usage.cache_read);
    usage.cost.cache_write =
        (rates.cache_write * short_write + rates.input * 2.0 * long_write) / 1_000_000.0;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage.cost.clone()
}

const EXTENDED_THINKING_LEVELS: [ModelThinkingLevel; 7] = [
    ModelThinkingLevel::Off,
    ModelThinkingLevel::Minimal,
    ModelThinkingLevel::Low,
    ModelThinkingLevel::Medium,
    ModelThinkingLevel::High,
    ModelThinkingLevel::Xhigh,
    ModelThinkingLevel::Max,
];

pub fn get_supported_thinking_levels(model: &Model) -> Vec<ModelThinkingLevel> {
    if !model.reasoning {
        return vec![ModelThinkingLevel::Off];
    }

    EXTENDED_THINKING_LEVELS
        .into_iter()
        .filter(|level| {
            let mapped = model
                .thinking_level_map
                .as_ref()
                .and_then(|map| map.get(level));
            if matches!(mapped, Some(None)) {
                return false;
            }
            if matches!(level, ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max) {
                return mapped.is_some();
            }
            true
        })
        .collect()
}

pub fn clamp_thinking_level(model: &Model, level: ModelThinkingLevel) -> ModelThinkingLevel {
    let available_levels = get_supported_thinking_levels(model);
    if available_levels.contains(&level) {
        return level;
    }
    let fallback = available_levels
        .first()
        .copied()
        .unwrap_or(ModelThinkingLevel::Off);

    let Some(requested_index) = EXTENDED_THINKING_LEVELS
        .iter()
        .position(|candidate| *candidate == level)
    else {
        return fallback;
    };

    for candidate in &EXTENDED_THINKING_LEVELS[requested_index..] {
        if available_levels.contains(candidate) {
            return *candidate;
        }
    }
    for candidate in EXTENDED_THINKING_LEVELS[..requested_index].iter().rev() {
        if available_levels.contains(candidate) {
            return *candidate;
        }
    }
    fallback
}

/// Check if two models are equal by comparing their type, id, and provider.
/// Returns false if either model is missing.
pub fn models_are_equal(a: Option<&AnyModel>, b: Option<&AnyModel>) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return false;
    };
    get_model_type(a) == get_model_type(b) && a.id() == b.id() && a.provider() == b.provider()
}

#[cfg(test)]
mod tests {
    //! Port of `test/models-runtime.test.ts`, `test/model-types.test.ts` and
    //! the catalog parts of `test/max-thinking.test.ts` and
    //! `test/supports-xhigh.test.ts`.

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use serde_json::{Value, json};
    use tokio::sync::Notify;

    use super::*;
    use crate::auth::{
        ApiKeyCredential, AuthEvent, AuthPrompt, CredentialInfo, ModelAuth, OAuthAuth,
        OAuthCredential,
    };
    use crate::types::{
        AssistantContent, AssistantMessageEvent, EmbeddingsStopReason, Message, ModelCost,
        ModelCostTier, ModelInput, StopReason, UserMessage, has_known_model_type,
        known_models_from_values,
    };
    use crate::utils::event_stream::AssistantMessageEventStream;
    use futures::StreamExt;

    type ResolveFn =
        Arc<dyn Fn(ApiKeyAuthInput) -> BoxFuture<Result<Option<AuthResult>>> + Send + Sync>;
    type CheckFn =
        Arc<dyn Fn(ApiKeyAuthInput) -> BoxFuture<Result<Option<AuthCheck>>> + Send + Sync>;
    type LoginFn =
        Arc<dyn Fn(ProviderAuthInteraction) -> BoxFuture<Result<ApiKeyCredential>> + Send + Sync>;
    type RefreshFn = Arc<dyn Fn(RefreshModelsContext) -> BoxFuture<Result<()>> + Send + Sync>;
    type OAuthRefreshFn = Arc<
        dyn Fn(OAuthCredential, CancellationToken) -> BoxFuture<Result<OAuthCredential>>
            + Send
            + Sync,
    >;

    fn test_model(provider: &str, id: &str) -> Model {
        Model {
            id: id.to_string(),
            name: id.to_string(),
            api: "test-api".to_string(),
            provider: provider.to_string(),
            base_url: "https://example.test/v1".to_string(),
            input: vec![ModelInput::Text],
            context_window: 10_000,
            max_tokens: 1_000,
            ..Default::default()
        }
    }

    fn done_message(model: &Model, text: &str) -> AssistantMessage {
        let mut message = AssistantMessage::empty_for(model);
        message.content = vec![AssistantContent::text(text)];
        message
    }

    #[derive(Debug, Clone)]
    struct ProviderCall {
        model: Model,
        api_key: Option<String>,
        env: Option<ProviderEnv>,
        headers: Option<ProviderHeaders>,
    }

    struct FnApiKeyAuth {
        name: String,
        resolve: ResolveFn,
        check: Option<CheckFn>,
        login: Option<LoginFn>,
    }

    #[async_trait]
    impl crate::auth::ApiKeyAuth for FnApiKeyAuth {
        fn name(&self) -> &str {
            &self.name
        }

        fn supports_login(&self) -> bool {
            self.login.is_some()
        }

        async fn login(&self, interaction: ProviderAuthInteraction) -> Result<ApiKeyCredential> {
            (self.login.as_ref().unwrap())(interaction).await
        }

        fn supports_check(&self) -> bool {
            self.check.is_some()
        }

        async fn check(&self, input: ApiKeyAuthInput) -> Result<Option<AuthCheck>> {
            (self.check.as_ref().unwrap())(input).await
        }

        async fn resolve(&self, input: ApiKeyAuthInput) -> Result<Option<AuthResult>> {
            (self.resolve)(input).await
        }
    }

    fn api_key_auth(resolve: ResolveFn) -> FnApiKeyAuth {
        FnApiKeyAuth {
            name: "Test".to_string(),
            resolve,
            check: None,
            login: None,
        }
    }

    /// Ambient auth for keyless test providers; reports "configured" with no auth values.
    fn ambient_auth() -> ProviderAuth {
        ProviderAuth {
            api_key: Some(Arc::new(api_key_auth(Arc::new(|_| {
                Box::pin(async { Ok(Some(AuthResult::default())) })
            })))),
            oauth: None,
        }
    }

    fn env_key_auth(key: Option<&str>) -> FnApiKeyAuth {
        let key = key.map(str::to_string);
        api_key_auth(Arc::new(move |input| {
            let key = key.clone();
            Box::pin(async move {
                let stored = input
                    .credential
                    .as_ref()
                    .and_then(|credential| credential.key.clone());
                let has_credential = input.credential.is_some();
                let Some(resolved) = stored.or(key) else {
                    return Ok(None);
                };
                Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(resolved),
                        ..Default::default()
                    },
                    env: None,
                    source: Some(if has_credential { "stored" } else { "env" }.to_string()),
                }))
            })
        }))
    }

    fn api_key_provider_auth(auth: FnApiKeyAuth) -> ProviderAuth {
        ProviderAuth {
            api_key: Some(Arc::new(auth)),
            oauth: None,
        }
    }

    struct TestOAuth {
        refresh: OAuthRefreshFn,
    }

    #[async_trait]
    impl OAuthAuth for TestOAuth {
        fn name(&self) -> &str {
            "Test OAuth"
        }

        async fn login(
            &self,
            _interaction: ProviderAuthInteraction,
            _options: LoginOptions,
        ) -> Result<OAuthCredential> {
            Err(Error::message("not used"))
        }

        async fn refresh(
            &self,
            credential: OAuthCredential,
            signal: CancellationToken,
        ) -> Result<OAuthCredential> {
            (self.refresh)(credential, signal).await
        }

        async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth> {
            Ok(ModelAuth {
                api_key: Some(credential.access.clone()),
                ..Default::default()
            })
        }
    }

    fn test_oauth(refresh: Option<OAuthRefreshFn>) -> Arc<dyn OAuthAuth> {
        Arc::new(TestOAuth {
            refresh: refresh.unwrap_or_else(|| {
                Arc::new(|credential, _| Box::pin(async move { Ok(credential) }))
            }),
        })
    }

    fn oauth_credential(access: &str, refresh: &str, expires: u64) -> Credential {
        Credential::OAuth(OAuthCredential {
            access: access.to_string(),
            refresh: refresh.to_string(),
            expires,
            extra: Default::default(),
        })
    }

    fn api_key_credential(key: &str) -> Credential {
        Credential::ApiKey(ApiKeyCredential {
            key: Some(key.to_string()),
            env: None,
        })
    }

    type GetModelsFn = Arc<dyn Fn() -> Result<Vec<Model>> + Send + Sync>;
    type SeenRefresh = (Option<Credential>, Option<bool>);

    struct TestProvider {
        id: String,
        auth: ProviderAuth,
        get_models: GetModelsFn,
        get_all_models: Option<Arc<dyn Fn() -> Result<Vec<AnyModel>> + Send + Sync>>,
        refresh_models: Option<RefreshFn>,
        calls: Arc<Mutex<Vec<ProviderCall>>>,
    }

    impl TestProvider {
        fn new(id: &str) -> Self {
            let models = vec![test_model(id, "model-a")];
            Self {
                id: id.to_string(),
                auth: ambient_auth(),
                get_models: Arc::new(move || Ok(models.clone())),
                get_all_models: None,
                refresh_models: None,
                calls: Arc::default(),
            }
        }

        fn models(mut self, models: Vec<Model>) -> Self {
            self.get_models = Arc::new(move || Ok(models.clone()));
            self
        }

        fn auth(mut self, auth: ProviderAuth) -> Self {
            self.auth = auth;
            self
        }

        fn refresh(mut self, refresh: RefreshFn) -> Self {
            self.refresh_models = Some(refresh);
            self
        }

        fn calls(mut self, calls: &Arc<Mutex<Vec<ProviderCall>>>) -> Self {
            self.calls = calls.clone();
            self
        }

        fn arc(self) -> Arc<dyn Provider> {
            Arc::new(self)
        }

        fn respond(&self, call: ProviderCall) -> AssistantMessageEventStream {
            let message = done_message(&call.model, "ok");
            self.calls.lock().push(call);
            let stream = AssistantMessageEventStream::new();
            stream.push(AssistantMessageEvent::Start {
                partial: message.clone(),
            });
            stream.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: message.clone(),
            });
            stream.end(Some(message));
            stream
        }
    }

    #[async_trait]
    impl Provider for TestProvider {
        fn id(&self) -> &str {
            &self.id
        }

        fn name(&self) -> &str {
            &self.id
        }

        fn auth(&self) -> &ProviderAuth {
            &self.auth
        }

        fn get_models(&self) -> Result<Vec<Model>> {
            (self.get_models)()
        }

        fn get_all_models(&self) -> Result<Vec<AnyModel>> {
            match &self.get_all_models {
                Some(get_all_models) => get_all_models(),
                None => Ok(self.get_models()?.into_iter().map(AnyModel::Chat).collect()),
            }
        }

        fn supports_refresh_models(&self) -> bool {
            self.refresh_models.is_some()
        }

        async fn refresh_models(&self, context: RefreshModelsContext) -> Result<()> {
            (self.refresh_models.as_ref().unwrap())(context).await
        }

        fn stream(
            &self,
            model: Model,
            _context: TranscriptContext,
            options: StreamOptions,
        ) -> AssistantMessageEventStream {
            self.respond(ProviderCall {
                model,
                api_key: options.api_key,
                env: options.env,
                headers: options.headers,
            })
        }

        fn stream_simple(
            &self,
            model: Model,
            _context: TranscriptContext,
            options: SimpleStreamOptions,
        ) -> AssistantMessageEventStream {
            self.respond(ProviderCall {
                model,
                api_key: options.stream.api_key,
                env: options.stream.env,
                headers: options.stream.headers,
            })
        }
    }

    fn context() -> Context {
        Context::builder().message(Message::user_text("hi")).build()
    }

    fn models_with_credentials(credentials: &Arc<InMemoryCredentialStore>) -> Models {
        create_models(CreateModelsOptions {
            credentials: Some(credentials.clone()),
            ..Default::default()
        })
    }

    async fn store_credential(
        store: &dyn CredentialStore,
        provider_id: &str,
        credential: Credential,
    ) {
        store
            .modify(
                provider_id,
                Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
                Default::default(),
            )
            .await
            .unwrap();
    }

    async fn read_access(store: &dyn CredentialStore, provider_id: &str) -> Option<String> {
        store
            .read(provider_id, Default::default())
            .await
            .unwrap()
            .and_then(|credential| {
                credential
                    .as_oauth()
                    .map(|credential| credential.access.clone())
            })
    }

    fn chat_streams() -> ProviderApi {
        struct Empty;
        impl ProviderStreams for Empty {
            fn stream(
                &self,
                _: Model,
                _: TranscriptContext,
                _: StreamOptions,
            ) -> AssistantMessageEventStream {
                AssistantMessageEventStream::new()
            }
            fn stream_simple(
                &self,
                _: Model,
                _: TranscriptContext,
                _: SimpleStreamOptions,
            ) -> AssistantMessageEventStream {
                AssistantMessageEventStream::new()
            }
        }
        ProviderApi::Single(Arc::new(Empty))
    }

    fn provider_ids(models: &Models) -> Vec<String> {
        models
            .get_providers()
            .iter()
            .map(|provider| provider.id().to_string())
            .collect()
    }

    fn ids(models: &[Model]) -> Vec<&str> {
        models.iter().map(|model| model.id.as_str()).collect()
    }

    #[tokio::test]
    async fn enumerates_credential_metadata_without_exposing_secrets() {
        let credentials = InMemoryCredentialStore::new();
        store_credential(&credentials, "api-provider", api_key_credential("secret")).await;
        store_credential(
            &credentials,
            "oauth-provider",
            oauth_credential("access", "refresh", now_millis() + 60_000),
        )
        .await;

        assert_eq!(
            credentials.list(Default::default()).await.unwrap(),
            vec![
                CredentialInfo {
                    provider_id: "api-provider".to_string(),
                    credential_type: AuthType::ApiKey
                },
                CredentialInfo {
                    provider_id: "oauth-provider".to_string(),
                    credential_type: AuthType::OAuth
                },
            ]
        );
    }

    #[test]
    fn applies_request_wide_pricing_tiers_above_the_configured_input_threshold() {
        let mut model = test_model("openai", "gpt-5.6-sol");
        model.cost = ModelCost {
            input: 5.0,
            output: 30.0,
            cache_read: 0.5,
            cache_write: 6.25,
            tiers: Some(vec![ModelCostTier {
                input_tokens_above: 272_000,
                input: 10.0,
                output: 45.0,
                cache_read: 1.0,
                cache_write: 12.5,
            }]),
        };
        let create_usage = |cache_write: u32| Usage {
            input: 200_000,
            output: 100_000,
            cache_read: 72_000,
            cache_write,
            total_tokens: 372_000 + cache_write,
            ..Default::default()
        };

        let short = calculate_cost(&model, &mut create_usage(0));
        assert_eq!(short.input, 1.0);
        assert_eq!(short.output, 3.0);
        assert_eq!(short.cache_read, 0.036);
        assert_eq!(short.cache_write, 0.0);

        let long = calculate_cost(&model, &mut create_usage(1));
        assert_eq!(long.input, 2.0);
        assert_eq!(long.output, 4.5);
        assert_eq!(long.cache_read, 0.072);
        assert_eq!(long.cache_write, 0.000_012_5);
    }

    #[test]
    fn charges_one_hour_cache_writes_at_twice_the_base_input_rate() {
        let mut model = test_model("anthropic", "m");
        model.cost = ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
            tiers: None,
        };
        let mut usage = Usage {
            cache_write: 1_000_000,
            cache_write_1h: Some(400_000),
            ..Default::default()
        };
        let cost = calculate_cost(&model, &mut usage);
        assert!((cost.cache_write - (3.75 * 0.6 + 6.0 * 0.4)).abs() < 1e-9);
        assert_eq!(usage.cost, cost);
    }

    #[test]
    fn registers_replaces_and_deletes_providers() {
        let models = create_models(Default::default());
        models.set_provider(TestProvider::new("p1").arc());
        models.set_provider(TestProvider::new("p2").arc());
        assert_eq!(provider_ids(&models), vec!["p1", "p2"]);

        let replacement = TestProvider::new("p1").arc();
        models.set_provider(replacement.clone());
        assert!(Arc::ptr_eq(
            &models.get_provider("p1").unwrap(),
            &replacement
        ));
        assert_eq!(models.get_providers().len(), 2);

        models.delete_provider("p1");
        assert!(models.get_provider("p1").is_none());

        models.clear_providers();
        assert!(models.get_providers().is_empty());
    }

    #[test]
    fn lists_and_finds_models_per_provider() {
        let models = create_models(Default::default());
        models.set_provider(
            TestProvider::new("p1")
                .models(vec![test_model("p1", "m1"), test_model("p1", "m2")])
                .arc(),
        );
        models.set_provider(
            TestProvider::new("p2")
                .models(vec![test_model("p2", "m3")])
                .arc(),
        );

        assert_eq!(ids(&models.get_models(None)), vec!["m1", "m2", "m3"]);
        assert_eq!(ids(&models.get_models(Some("p1"))), vec!["m1", "m2"]);
        assert!(models.get_models(Some("nope")).is_empty());
        assert_eq!(models.get_model("p2", "m3").unwrap().id, "m3");
        assert!(models.get_model("p2", "missing").is_none());

        // has_api() checks dynamically looked-up models at runtime
        let found = AnyModel::Chat(models.get_model("p2", "m3").unwrap());
        assert!(!has_api(&found, "openai-completions"));
        assert!(has_api(&found, "test-api"));
    }

    #[tokio::test]
    async fn keeps_chat_reads_independent_from_the_all_model_catalog() {
        let mut provider = TestProvider::new("chat-only");
        provider.get_all_models = Some(Arc::new(|| Err(Error::message("all models unavailable"))));
        let models = create_models(Default::default());
        models.set_provider(provider.arc());

        assert_eq!(ids(&models.get_models(Some("chat-only"))), vec!["model-a"]);
        assert_eq!(
            models.get_model("chat-only", "model-a").unwrap().id,
            "model-a"
        );
        assert_eq!(
            ids(&models
                .get_available(Some("chat-only"), Default::default())
                .await
                .unwrap()),
            vec!["model-a"]
        );
        assert!(models.get_all_models(Some("chat-only")).is_empty());
    }

    #[test]
    fn swallows_provider_source_failures_for_both_all_provider_and_single_provider_listing() {
        let models = create_models(Default::default());
        let mut broken = TestProvider::new("broken");
        broken.get_models = Arc::new(|| Err(Error::message("boom")));
        models.set_provider(broken.arc());
        models.set_provider(
            TestProvider::new("ok")
                .models(vec![test_model("ok", "m1")])
                .arc(),
        );

        assert_eq!(ids(&models.get_models(None)), vec!["m1"]);
        assert!(models.get_models(Some("broken")).is_empty());
        // precise failures come from the provider directly
        assert_eq!(
            models
                .get_provider("broken")
                .unwrap()
                .get_models()
                .unwrap_err()
                .to_string(),
            "boom"
        );
    }

    #[tokio::test]
    async fn refresh_updates_every_configured_dynamic_provider_and_reports_failures() {
        let list = Arc::new(Mutex::new(vec![test_model("dyn", "before")]));
        let refreshes = Arc::new(AtomicUsize::new(0));
        let models = create_models(Default::default());
        let mut dynamic = TestProvider::new("dyn");
        let listed = list.clone();
        dynamic.get_models = Arc::new(move || Ok(listed.lock().clone()));
        let (updated, counter) = (list.clone(), refreshes.clone());
        models.set_provider(
            dynamic
                .refresh(Arc::new(move |refresh| {
                    let (updated, counter) = (updated.clone(), counter.clone());
                    Box::pin(async move {
                        if !refresh.allow_network {
                            return Ok(());
                        }
                        counter.fetch_add(1, Ordering::SeqCst);
                        refresh
                            .publish(ModelsPublication {
                                persist: None,
                                update: Some(Box::new(move || {
                                    *updated.lock() = vec![test_model("dyn", "after")];
                                })),
                            })
                            .await?;
                        Ok(())
                    })
                }))
                .arc(),
        );
        models.set_provider(
            TestProvider::new("static")
                .models(vec![test_model("static", "s1")])
                .arc(),
        );

        assert!(models.get_model("dyn", "before").is_some());
        let first = models.refresh(Default::default()).await;
        assert!(first.errors.is_empty());
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        assert!(models.get_model("dyn", "after").is_some());
        assert!(models.get_model("dyn", "before").is_none());

        models.set_provider(
            TestProvider::new("flaky")
                .refresh(Arc::new(|context| {
                    Box::pin(async move {
                        if context.allow_network {
                            return Err(Error::message("fetch failed"));
                        }
                        Ok(())
                    })
                }))
                .arc(),
        );
        let second = models.refresh(Default::default()).await;
        assert_eq!(refreshes.load(Ordering::SeqCst), 2);
        assert_eq!(second.errors["flaky"].to_string(), "fetch failed");
    }

    #[tokio::test]
    async fn restricts_refresh_work_to_selected_providers() {
        let calls: Arc<Mutex<Vec<String>>> = Arc::default();
        let models = create_models(Default::default());
        for id in ["one", "two"] {
            let calls = calls.clone();
            models.set_provider(
                TestProvider::new(id)
                    .refresh(Arc::new(move |context| {
                        let calls = calls.clone();
                        Box::pin(async move {
                            let mode = if context.allow_network {
                                "network"
                            } else {
                                "cache"
                            };
                            calls.lock().push(format!("{id}:{mode}"));
                            Ok(())
                        })
                    }))
                    .arc(),
            );
        }

        let result = models
            .refresh(ModelsRefreshOptions {
                providers: Some(vec!["two".to_string(), "unknown".to_string()]),
                ..Default::default()
            })
            .await;

        assert!(result.errors.is_empty());
        assert_eq!(*calls.lock(), vec!["two:cache", "two:network"]);
    }

    #[tokio::test]
    async fn restores_cached_models_before_waiting_for_network_auth() {
        let store = Arc::new(InMemoryModelsStore::new());
        store
            .write(
                "dynamic",
                ModelsStoreEntry {
                    models: vec![AnyModel::Chat(test_model("dynamic", "cached"))],
                    ..Default::default()
                },
                Default::default(),
            )
            .await
            .unwrap();
        let auth_started = Arc::new(Notify::new());
        let started = auth_started.clone();
        let blocked = api_key_auth(Arc::new(move |_| {
            let started = started.clone();
            Box::pin(async move {
                started.notify_one();
                std::future::pending::<()>().await;
                Ok(None)
            })
        }));
        let fetch: FetchModels =
            Arc::new(|_| Box::pin(async { Err(Error::message("must not fetch")) }));
        let provider = create_provider(CreateProviderOptions {
            id: "dynamic".to_string(),
            auth: api_key_provider_auth(blocked),
            fetch_models: Some(fetch),
            api: Some(chat_streams()),
            ..Default::default()
        })
        .unwrap();
        let models = create_models(CreateModelsOptions {
            models_store: Some(store),
            ..Default::default()
        });
        models.set_provider(provider);
        let controller = CancellationToken::new();
        let pending = tokio::spawn({
            let models = models.clone();
            let signal = controller.clone();
            async move {
                models
                    .refresh(ModelsRefreshOptions {
                        providers: Some(vec!["dynamic".to_string()]),
                        signal: Some(signal),
                        ..Default::default()
                    })
                    .await
            }
        });
        auth_started.notified().await;

        assert!(models.get_model("dynamic", "cached").is_some());
        controller.cancel();
        assert!(pending.await.unwrap().aborted);
    }

    struct SharedStore {
        entry: Arc<Mutex<Option<ModelsStoreEntry>>>,
        signals: Option<Arc<Mutex<Vec<Option<CancellationToken>>>>>,
    }

    #[async_trait]
    impl ModelsStore for SharedStore {
        async fn read(
            &self,
            _provider_id: &str,
            options: ModelsStoreOperationOptions,
        ) -> Result<Option<ModelsStoreEntry>> {
            if let Some(signals) = &self.signals {
                signals.lock().push(options.signal);
                return Ok(None);
            }
            Ok(self.entry.lock().clone())
        }

        async fn write(
            &self,
            _provider_id: &str,
            entry: ModelsStoreEntry,
            options: ModelsStoreOperationOptions,
        ) -> Result<()> {
            if let Some(signals) = &self.signals {
                signals.lock().push(options.signal);
            }
            *self.entry.lock() = Some(entry);
            Ok(())
        }

        async fn delete(
            &self,
            _provider_id: &str,
            options: ModelsStoreOperationOptions,
        ) -> Result<()> {
            if let Some(signals) = &self.signals {
                signals.lock().push(options.signal);
            }
            *self.entry.lock() = None;
            Ok(())
        }
    }

    #[tokio::test]
    async fn lets_providers_choose_persistent_deletion_and_ephemeral_publication_atomically() {
        let entry = Arc::new(Mutex::new(Some(ModelsStoreEntry {
            models: vec![AnyModel::Chat(test_model("dynamic", "stored"))],
            ..Default::default()
        })));
        let state = Arc::new(Mutex::new("initial"));
        let models = create_models(CreateModelsOptions {
            models_store: Some(Arc::new(SharedStore {
                entry: entry.clone(),
                signals: None,
            })),
            ..Default::default()
        });
        let (observed, published) = (entry.clone(), state.clone());
        models.set_provider(
            TestProvider::new("dynamic")
                .refresh(Arc::new(move |context| {
                    let (observed, published) = (observed.clone(), published.clone());
                    Box::pin(async move {
                        assert_eq!(context.stored.as_ref().unwrap().models[0].id(), "stored");
                        let deleted = published.clone();
                        context
                            .publish(ModelsPublication {
                                persist: Some(None),
                                update: Some(Box::new(move || {
                                    assert!(observed.lock().is_none());
                                    *deleted.lock() = "deleted";
                                })),
                            })
                            .await?;
                        context
                            .publish(ModelsPublication {
                                persist: None,
                                update: Some(Box::new(move || *published.lock() = "ephemeral")),
                            })
                            .await?;
                        Ok(())
                    })
                }))
                .arc(),
        );

        let result = models
            .refresh(ModelsRefreshOptions {
                allow_network: Some(false),
                ..Default::default()
            })
            .await;

        assert!(result.errors.is_empty());
        assert!(entry.lock().is_none());
        assert_eq!(*state.lock(), "ephemeral");
    }

    #[tokio::test]
    async fn persists_dynamic_catalogs_and_restores_them_without_network_access() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let models_store = Arc::new(InMemoryModelsStore::new());
        store_credential(credentials.as_ref(), "dynamic", api_key_credential("key")).await;
        let create_dynamic_provider = |fetch: FetchModels| {
            create_provider(CreateProviderOptions {
                id: "dynamic".to_string(),
                auth: api_key_provider_auth(env_key_auth(None)),
                fetch_models: Some(fetch),
                api: Some(chat_streams()),
                ..Default::default()
            })
            .unwrap()
        };
        let options = CreateModelsOptions {
            credentials: Some(credentials.clone()),
            models_store: Some(models_store.clone()),
            ..Default::default()
        };

        let online = create_models(options.clone());
        online.set_provider(create_dynamic_provider(Arc::new(|_| {
            Box::pin(async { Ok(vec![AnyModel::Chat(test_model("dynamic", "fetched"))]) })
        })));
        assert!(online.refresh(Default::default()).await.errors.is_empty());
        assert!(online.get_model("dynamic", "fetched").is_some());

        let offline = create_models(options);
        offline.set_provider(create_dynamic_provider(Arc::new(|_| {
            Box::pin(async { Err(Error::message("must not fetch")) })
        })));
        let result = offline
            .refresh(ModelsRefreshOptions {
                allow_network: Some(false),
                ..Default::default()
            })
            .await;
        assert!(result.errors.is_empty());
        assert!(offline.get_model("dynamic", "fetched").is_some());
    }

    #[tokio::test]
    async fn passes_effective_api_key_credentials_and_refresh_options_while_skipping_unconfigured_providers()
     {
        let effective: Arc<Mutex<Option<SeenRefresh>>> = Arc::default();
        let unconfigured_refreshes = Arc::new(AtomicUsize::new(0));
        let models = create_models(Default::default());
        let seen = effective.clone();
        models.set_provider(
            TestProvider::new("configured")
                .auth(api_key_provider_auth(env_key_auth(Some("ambient-key"))))
                .refresh(Arc::new(move |context| {
                    let seen = seen.clone();
                    Box::pin(async move {
                        if context.allow_network {
                            *seen.lock() = Some((context.credential.clone(), context.force));
                        }
                        Ok(())
                    })
                }))
                .arc(),
        );
        let counter = unconfigured_refreshes.clone();
        models.set_provider(
            TestProvider::new("unconfigured")
                .auth(api_key_provider_auth(env_key_auth(None)))
                .refresh(Arc::new(move |context| {
                    let counter = counter.clone();
                    Box::pin(async move {
                        if context.allow_network {
                            counter.fetch_add(1, Ordering::SeqCst);
                        }
                        Ok(())
                    })
                }))
                .arc(),
        );

        models
            .refresh(ModelsRefreshOptions {
                force: Some(true),
                ..Default::default()
            })
            .await;
        assert_eq!(
            effective.lock().clone(),
            Some((Some(api_key_credential("ambient-key")), Some(true)))
        );
        assert_eq!(unconfigured_refreshes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn refreshes_expired_oauth_before_refreshing_models() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        store_credential(
            credentials.as_ref(),
            "oauth-dynamic",
            oauth_credential("expired", "refresh", 0),
        )
        .await;
        let seen: Arc<Mutex<Option<Credential>>> = Arc::default();
        let models = models_with_credentials(&credentials);
        let observed = seen.clone();
        models.set_provider(
            TestProvider::new("oauth-dynamic")
                .auth(ProviderAuth {
                    api_key: None,
                    oauth: Some(test_oauth(Some(Arc::new(|_, _| {
                        Box::pin(async {
                            Ok(OAuthCredential {
                                access: "fresh".to_string(),
                                refresh: "rotated".to_string(),
                                expires: now_millis() + 60_000,
                                extra: Default::default(),
                            })
                        })
                    })))),
                })
                .refresh(Arc::new(move |context| {
                    let observed = observed.clone();
                    Box::pin(async move {
                        if context.allow_network {
                            *observed.lock() = context.credential.clone();
                        }
                        Ok(())
                    })
                }))
                .arc(),
        );

        assert!(models.refresh(Default::default()).await.errors.is_empty());
        let credential = seen.lock().clone().unwrap();
        let oauth = credential.as_oauth().unwrap();
        assert_eq!(
            (oauth.access.as_str(), oauth.refresh.as_str()),
            ("fresh", "rotated")
        );
        assert_eq!(
            read_access(credentials.as_ref(), "oauth-dynamic")
                .await
                .as_deref(),
            Some("fresh")
        );
    }

    #[tokio::test]
    async fn always_gives_providers_a_concrete_signal() {
        let received: Arc<Mutex<Option<CancellationToken>>> = Arc::default();
        let models = create_models(Default::default());
        let observed = received.clone();
        models.set_provider(
            TestProvider::new("dynamic")
                .refresh(Arc::new(move |context| {
                    *observed.lock() = Some(context.signal.clone());
                    Box::pin(async { Ok(()) })
                }))
                .arc(),
        );

        let result = models.refresh(Default::default()).await;
        assert!(!result.aborted);
        assert!(!received.lock().as_ref().unwrap().is_cancelled());
    }

    #[tokio::test]
    async fn binds_model_store_waits_to_the_provider_refresh_signal() {
        let storage_signals: Arc<Mutex<Vec<Option<CancellationToken>>>> = Arc::default();
        let provider_signal: Arc<Mutex<Option<CancellationToken>>> = Arc::default();
        let models = create_models(CreateModelsOptions {
            models_store: Some(Arc::new(SharedStore {
                entry: Arc::default(),
                signals: Some(storage_signals.clone()),
            })),
            ..Default::default()
        });
        let observed = provider_signal.clone();
        models.set_provider(
            TestProvider::new("dynamic")
                .auth(api_key_provider_auth(env_key_auth(Some("key"))))
                .refresh(Arc::new(move |context| {
                    let observed = observed.clone();
                    Box::pin(async move {
                        *observed.lock() = Some(context.signal.clone());
                        if !context.allow_network {
                            return Ok(());
                        }
                        context
                            .publish(ModelsPublication {
                                persist: Some(Some(ModelsStoreEntry {
                                    models: vec![AnyModel::Chat(test_model("dynamic", "fresh"))],
                                    ..Default::default()
                                })),
                                update: None,
                            })
                            .await?;
                        Ok(())
                    })
                }))
                .arc(),
        );

        let caller = CancellationToken::new();
        let result = models
            .refresh(ModelsRefreshOptions {
                providers: Some(vec!["dynamic".to_string()]),
                signal: Some(caller.clone()),
                ..Default::default()
            })
            .await;

        assert!(result.errors.is_empty());
        let signals = storage_signals.lock().clone();
        assert_eq!(signals.len(), 3);
        // Cancelling the provider signal (a child of the caller's) reaches every store wait.
        provider_signal.lock().as_ref().unwrap().cancel();
        assert!(
            signals
                .iter()
                .all(|signal| signal.as_ref().unwrap().is_cancelled())
        );
        assert!(!caller.is_cancelled());
    }

    #[tokio::test]
    async fn returns_aborted_state_without_reporting_cancellation_as_a_provider_error() {
        let controller = CancellationToken::new();
        let models = create_models(Default::default());
        let abort = controller.clone();
        models.set_provider(
            TestProvider::new("dynamic")
                .refresh(Arc::new(move |_| {
                    abort.cancel();
                    Box::pin(async { Ok(()) })
                }))
                .arc(),
        );

        let result = models
            .refresh(ModelsRefreshOptions {
                signal: Some(controller),
                ..Default::default()
            })
            .await;
        assert!(result.aborted);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn stops_waiting_on_abort_when_a_provider_ignores_its_signal() {
        let controller = CancellationToken::new();
        let started = Arc::new(Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let models = create_models(Default::default());
        let (marker, counter) = (started.clone(), calls.clone());
        models.set_provider(
            TestProvider::new("dynamic")
                .refresh(Arc::new(move |_| {
                    let (marker, counter) = (marker.clone(), counter.clone());
                    Box::pin(async move {
                        if counter.fetch_add(1, Ordering::SeqCst) != 0 {
                            return Ok(());
                        }
                        marker.notify_one();
                        std::future::pending::<()>().await;
                        Err(Error::message("late provider failure"))
                    })
                }))
                .arc(),
        );

        let pending = tokio::spawn({
            let models = models.clone();
            let signal = controller.clone();
            async move {
                models
                    .refresh(ModelsRefreshOptions {
                        signal: Some(signal),
                        ..Default::default()
                    })
                    .await
            }
        });
        started.notified().await;
        controller.cancel();

        let result = pending.await.unwrap();
        assert!(result.aborted);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn rejects_late_publication_from_a_superseded_non_cooperative_provider() {
        let store = Arc::new(InMemoryModelsStore::new());
        let state = Arc::new(Mutex::new("initial".to_string()));
        let calls = Arc::new(AtomicUsize::new(0));
        let first_started = Arc::new(Notify::new());
        let first_blocked = Arc::new(Notify::new());
        let models = create_models(CreateModelsOptions {
            models_store: Some(store.clone()),
            ..Default::default()
        });
        let (published, counter, started, blocked) = (
            state.clone(),
            calls.clone(),
            first_started.clone(),
            first_blocked.clone(),
        );
        models.set_provider(
            TestProvider::new("dynamic")
                .refresh(Arc::new(move |context| {
                    let (published, counter, started, blocked) = (
                        published.clone(),
                        counter.clone(),
                        started.clone(),
                        blocked.clone(),
                    );
                    Box::pin(async move {
                        if !context.allow_network {
                            return Ok(());
                        }
                        let current = counter.fetch_add(1, Ordering::SeqCst) + 1;
                        if current == 1 {
                            started.notify_one();
                            blocked.notified().await;
                        }
                        let value = format!("generation-{current}");
                        let update = value.clone();
                        context
                            .publish(ModelsPublication {
                                persist: Some(Some(ModelsStoreEntry {
                                    models: vec![AnyModel::Chat(test_model("dynamic", &value))],
                                    ..Default::default()
                                })),
                                update: Some(Box::new(move || *published.lock() = update)),
                            })
                            .await?;
                        Ok(())
                    })
                }))
                .arc(),
        );

        let refresh = |models: Models| async move {
            models
                .refresh(ModelsRefreshOptions {
                    providers: Some(vec!["dynamic".to_string()]),
                    ..Default::default()
                })
                .await
        };
        let first = tokio::spawn(refresh(models.clone()));
        first_started.notified().await;
        refresh(models.clone()).await;
        first.await.unwrap();
        first_blocked.notify_one();
        tokio::task::yield_now().await;

        assert_eq!(*state.lock(), "generation-2");
        let stored = store
            .read("dynamic", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.models[0].id(), "generation-2");
    }

    struct TestInteraction {
        signal: Option<CancellationToken>,
    }

    #[async_trait]
    impl AuthInteraction for TestInteraction {
        fn signal(&self) -> Option<CancellationToken> {
            self.signal.clone()
        }

        async fn prompt(&self, _prompt: AuthPrompt) -> Result<String> {
            Ok("unused".to_string())
        }

        fn notify(&self, _event: AuthEvent) {}
    }

    #[tokio::test]
    async fn passes_caller_signals_to_provider_auth_callbacks() {
        let controller = CancellationToken::new();
        let received: Arc<Mutex<Vec<CancellationToken>>> = Arc::default();
        let (on_login, on_check, on_resolve) =
            (received.clone(), received.clone(), received.clone());
        let auth = FnApiKeyAuth {
            name: "Signal auth".to_string(),
            login: Some(Arc::new(move |interaction| {
                on_login.lock().push(interaction.signal.clone());
                Box::pin(async {
                    Ok(ApiKeyCredential {
                        key: Some("saved".to_string()),
                        env: None,
                    })
                })
            })),
            check: Some(Arc::new(move |input| {
                on_check.lock().push(input.signal.clone());
                Box::pin(async {
                    Ok(Some(AuthCheck {
                        source: None,
                        auth_type: AuthType::ApiKey,
                    }))
                })
            })),
            resolve: Arc::new(move |input| {
                on_resolve.lock().push(input.signal.clone());
                Box::pin(async {
                    Ok(Some(AuthResult {
                        auth: ModelAuth {
                            api_key: Some("resolved".to_string()),
                            ..Default::default()
                        },
                        ..Default::default()
                    }))
                })
            }),
        };
        let models = create_models(Default::default());
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(auth))
                .arc(),
        );

        models
            .check_auth("p1", AuthOperationOptions::with_signal(&controller))
            .await
            .unwrap();
        models
            .get_auth(
                "p1",
                AuthResolutionOverrides {
                    signal: Some(controller.clone()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        models
            .login(
                "p1",
                AuthType::ApiKey,
                Arc::new(TestInteraction {
                    signal: Some(controller.clone()),
                }),
                LoginOptions::default(),
            )
            .await
            .unwrap();

        let received = received.lock().clone();
        assert_eq!(received.len(), 3);
        assert!(received.iter().all(|signal| !signal.is_cancelled()));
        controller.cancel();
        assert!(received.iter().all(CancellationToken::is_cancelled));
    }

    #[tokio::test]
    async fn stops_waiting_for_non_cooperative_auth_callbacks() {
        let check_started = Arc::new(Notify::new());
        let resolve_started = Arc::new(Notify::new());
        let (check_marker, resolve_marker) = (check_started.clone(), resolve_started.clone());
        let auth = FnApiKeyAuth {
            name: "Blocked auth".to_string(),
            login: None,
            check: Some(Arc::new(move |_| {
                let marker = check_marker.clone();
                Box::pin(async move {
                    marker.notify_one();
                    std::future::pending::<()>().await;
                    Ok(None)
                })
            })),
            resolve: Arc::new(move |_| {
                let marker = resolve_marker.clone();
                Box::pin(async move {
                    marker.notify_one();
                    std::future::pending::<()>().await;
                    Ok(None)
                })
            }),
        };
        let models = create_models(Default::default());
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(auth))
                .arc(),
        );

        let available_controller = CancellationToken::new();
        let available = tokio::spawn({
            let models = models.clone();
            let signal = available_controller.clone();
            async move {
                models
                    .get_available(None, AuthOperationOptions::with_signal(&signal))
                    .await
            }
        });
        check_started.notified().await;
        available_controller.cancel();
        assert!(available.await.unwrap().unwrap_err().is_abort());

        let auth_controller = CancellationToken::new();
        let auth = tokio::spawn({
            let models = models.clone();
            let signal = auth_controller.clone();
            async move {
                models
                    .get_auth(
                        "p1",
                        AuthResolutionOverrides {
                            signal: Some(signal),
                            ..Default::default()
                        },
                    )
                    .await
            }
        });
        resolve_started.notified().await;
        auth_controller.cancel();
        assert!(auth.await.unwrap().unwrap_err().is_abort());
    }

    #[tokio::test]
    async fn cancels_queued_credential_mutations_without_running_them_later() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let first_blocked = Arc::new(Notify::new());
        let first_running = Arc::new(Notify::new());
        let second_ran = Arc::new(AtomicBool::new(false));
        let first = tokio::spawn({
            let credentials = credentials.clone();
            let (blocked, running) = (first_blocked.clone(), first_running.clone());
            async move {
                credentials
                    .modify(
                        "p1",
                        Box::new(move |_| {
                            Box::pin(async move {
                                running.notify_one();
                                blocked.notified().await;
                                Ok(Some(api_key_credential("first")))
                            })
                        }),
                        Default::default(),
                    )
                    .await
            }
        });
        first_running.notified().await;
        let controller = CancellationToken::new();
        let second = tokio::spawn({
            let credentials = credentials.clone();
            let (ran, signal) = (second_ran.clone(), controller.clone());
            async move {
                credentials
                    .modify(
                        "p1",
                        Box::new(move |_| {
                            Box::pin(async move {
                                ran.store(true, Ordering::SeqCst);
                                Ok(Some(api_key_credential("second")))
                            })
                        }),
                        AuthOperationOptions::with_signal(&signal),
                    )
                    .await
            }
        });

        tokio::task::yield_now().await;
        controller.cancel();
        assert!(second.await.unwrap().unwrap_err().is_abort());
        first_blocked.notify_one();
        first.await.unwrap().unwrap();
        tokio::task::yield_now().await;

        assert!(!second_ran.load(Ordering::SeqCst));
        assert_eq!(
            credentials.read("p1", Default::default()).await.unwrap(),
            Some(api_key_credential("first"))
        );
    }

    #[tokio::test]
    async fn passes_cancellation_to_oauth_refresh_and_preserves_the_previous_credential() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let previous = oauth_credential("old", "old-refresh", 0);
        store_credential(credentials.as_ref(), "p1", previous.clone()).await;
        let refresh_started = Arc::new(Notify::new());
        let received: Arc<Mutex<Option<CancellationToken>>> = Arc::default();
        let models = models_with_credentials(&credentials);
        let (marker, observed) = (refresh_started.clone(), received.clone());
        models.set_provider(
            TestProvider::new("p1")
                .auth(ProviderAuth {
                    api_key: None,
                    oauth: Some(test_oauth(Some(Arc::new(move |credential, signal| {
                        *observed.lock() = Some(signal);
                        let marker = marker.clone();
                        Box::pin(async move {
                            marker.notify_one();
                            std::future::pending::<()>().await;
                            Ok(credential)
                        })
                    })))),
                })
                .arc(),
        );
        let controller = CancellationToken::new();
        let auth = tokio::spawn({
            let models = models.clone();
            let signal = controller.clone();
            async move {
                models
                    .get_auth(
                        "p1",
                        AuthResolutionOverrides {
                            signal: Some(signal),
                            ..Default::default()
                        },
                    )
                    .await
            }
        });
        refresh_started.notified().await;
        controller.cancel();

        assert!(auth.await.unwrap().unwrap_err().is_abort());
        assert!(received.lock().as_ref().unwrap().is_cancelled());
        tokio::task::yield_now().await;
        assert_eq!(
            credentials.read("p1", Default::default()).await.unwrap(),
            Some(previous)
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn oauth_refresh_timeout_only_aborts_the_refresh_signal() {
        // `AbortSignal.timeout(15s)` aborts the refresh signal; a refresh that
        // ignores it still completes, one that honors it fails with the
        // timeout reason.
        for honors_signal in [false, true] {
            let credentials = Arc::new(InMemoryCredentialStore::new());
            store_credential(
                credentials.as_ref(),
                "p1",
                oauth_credential("old", "old-refresh", 0),
            )
            .await;
            let models = models_with_credentials(&credentials);
            models.set_provider(
                TestProvider::new("p1")
                    .auth(ProviderAuth {
                        api_key: None,
                        oauth: Some(test_oauth(Some(Arc::new(move |_, signal| {
                            Box::pin(async move {
                                if honors_signal {
                                    signal.cancelled().await;
                                    return Err(Error::aborted());
                                }
                                tokio::time::sleep(Duration::from_secs(20)).await;
                                assert!(signal.is_cancelled());
                                Ok(OAuthCredential {
                                    access: "new".to_string(),
                                    refresh: "new-refresh".to_string(),
                                    expires: now_millis() + 60 * 60_000,
                                    extra: Default::default(),
                                })
                            })
                        })))),
                    })
                    .arc(),
            );

            let result = models.get_auth("p1", Default::default()).await;
            if honors_signal {
                let error = result.unwrap_err().to_string();
                assert_eq!(
                    error,
                    "OAuth refresh failed for p1: The operation was aborted due to timeout"
                );
            } else {
                assert_eq!(
                    result.unwrap().unwrap().auth.api_key.as_deref(),
                    Some("new")
                );
            }
        }
    }

    #[tokio::test]
    async fn resolves_auth_stored_credential_owns_the_provider_ambient_only_when_nothing_stored() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let models = models_with_credentials(&credentials);
        models.set_provider(
            TestProvider::new("p1")
                .auth(ProviderAuth {
                    api_key: Some(Arc::new(env_key_auth(Some("env-key")))),
                    oauth: Some(test_oauth(None)),
                })
                .arc(),
        );
        let model = test_model("p1", "model-a");
        let api_key = |result: Option<AuthResult>| result.unwrap().auth.api_key.unwrap();

        // model and provider-id targets resolve the same provider-scoped auth
        assert_eq!(
            api_key(models.get_auth(&model, Default::default()).await.unwrap()),
            "env-key"
        );
        assert_eq!(
            api_key(models.get_auth("p1", Default::default()).await.unwrap()),
            "env-key"
        );
        let explicit = AuthResolutionOverrides {
            api_key: Some("explicit-key".to_string()),
            ..Default::default()
        };
        assert_eq!(
            api_key(models.get_auth(&model, explicit).await.unwrap()),
            "explicit-key"
        );

        // stored oauth credential (persisted via the single write path): beats ambient env
        store_credential(
            credentials.as_ref(),
            "p1",
            oauth_credential("oauth-token", "r", now_millis() + 10 * 60_000),
        )
        .await;
        let resolution = models
            .get_auth("p1", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolution.auth.api_key.as_deref(), Some("oauth-token"));
        assert_eq!(resolution.source.as_deref(), Some("OAuth"));

        // stored api-key credential resolves through api-key auth, beats env
        store_credential(credentials.as_ref(), "p1", api_key_credential("stored-key")).await;
        let resolution = models
            .get_auth("p1", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolution.auth.api_key.as_deref(), Some("stored-key"));
        assert_eq!(resolution.source.as_deref(), Some("stored"));
    }

    #[tokio::test]
    async fn checks_provider_auth_without_refreshing_oauth_and_filters_available_models() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let refreshes = Arc::new(AtomicUsize::new(0));
        let models = models_with_credentials(&credentials);
        models.set_provider(
            TestProvider::new("ambient")
                .auth(api_key_provider_auth(env_key_auth(Some("env-key"))))
                .arc(),
        );
        models.set_provider(
            TestProvider::new("missing")
                .auth(api_key_provider_auth(env_key_auth(None)))
                .arc(),
        );
        let counter = refreshes.clone();
        models.set_provider(
            TestProvider::new("oauth")
                .auth(ProviderAuth {
                    api_key: None,
                    oauth: Some(test_oauth(Some(Arc::new(move |credential, _| {
                        counter.fetch_add(1, Ordering::SeqCst);
                        Box::pin(async move { Ok(credential) })
                    })))),
                })
                .arc(),
        );
        store_credential(
            credentials.as_ref(),
            "oauth",
            oauth_credential("expired", "refresh", 0),
        )
        .await;

        assert_eq!(
            models
                .check_auth("ambient", Default::default())
                .await
                .unwrap(),
            Some(AuthCheck {
                source: Some("env".to_string()),
                auth_type: AuthType::ApiKey
            })
        );
        assert_eq!(
            models
                .check_auth("missing", Default::default())
                .await
                .unwrap(),
            None
        );
        assert_eq!(
            models
                .check_auth("oauth", Default::default())
                .await
                .unwrap(),
            Some(AuthCheck {
                source: Some("OAuth".to_string()),
                auth_type: AuthType::OAuth
            })
        );
        assert_eq!(refreshes.load(Ordering::SeqCst), 0);
        let providers = |models: Vec<Model>| {
            models
                .into_iter()
                .map(|model| model.provider)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            providers(
                models
                    .get_available(None, Default::default())
                    .await
                    .unwrap()
            ),
            vec!["ambient", "oauth"]
        );
        assert_eq!(
            providers(
                models
                    .get_available(Some("ambient"), Default::default())
                    .await
                    .unwrap()
            ),
            vec!["ambient"]
        );
    }

    #[tokio::test]
    async fn runs_provider_login_and_logout_through_the_credential_store() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let mut auth = env_key_auth(None);
        auth.login = Some(Arc::new(|_| {
            Box::pin(async {
                Ok(ApiKeyCredential {
                    key: Some("logged-in".to_string()),
                    env: None,
                })
            })
        }));
        let models = models_with_credentials(&credentials);
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(auth))
                .arc(),
        );

        let credential = models
            .login(
                "p1",
                AuthType::ApiKey,
                Arc::new(TestInteraction { signal: None }),
                LoginOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(credential, api_key_credential("logged-in"));
        assert_eq!(
            credentials.read("p1", Default::default()).await.unwrap(),
            Some(credential)
        );

        models.logout("p1", Default::default()).await.unwrap();
        assert_eq!(
            credentials.read("p1", Default::default()).await.unwrap(),
            None
        );

        let error = models
            .login(
                "p1",
                AuthType::OAuth,
                Arc::new(TestInteraction { signal: None }),
                LoginOptions::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "p1 does not support oauth login");
    }

    /// A store whose `modify` ignores its signal: it waits for `release`,
    /// optionally after running the modifier.
    struct StalledStore {
        base: InMemoryCredentialStore,
        run_modifier_first: bool,
        modifier_ran: Arc<Notify>,
        release: Arc<Notify>,
    }

    #[async_trait]
    impl CredentialStore for StalledStore {
        async fn read(
            &self,
            provider_id: &str,
            options: AuthOperationOptions,
        ) -> Result<Option<Credential>> {
            self.base.read(provider_id, options).await
        }

        async fn list(&self, options: AuthOperationOptions) -> Result<Vec<CredentialInfo>> {
            self.base.list(options).await
        }

        async fn modify(
            &self,
            _provider_id: &str,
            modifier: crate::auth::CredentialModifier,
            _options: AuthOperationOptions,
        ) -> Result<Option<Credential>> {
            let next = if self.run_modifier_first {
                let next = modifier(None).await?;
                self.modifier_ran.notify_one();
                next
            } else {
                None
            };
            self.release.notified().await;
            Ok(next)
        }

        async fn delete(&self, provider_id: &str, options: AuthOperationOptions) -> Result<()> {
            self.base.delete(provider_id, options).await
        }
    }

    #[tokio::test]
    async fn login_abort_does_not_wait_for_a_queued_non_cooperative_store_mutation() {
        for run_modifier_first in [false, true] {
            let store = Arc::new(StalledStore {
                base: InMemoryCredentialStore::new(),
                run_modifier_first,
                modifier_ran: Arc::new(Notify::new()),
                release: Arc::new(Notify::new()),
            });
            let logged_in = Arc::new(Notify::new());
            let marker = logged_in.clone();
            let mut auth = env_key_auth(None);
            auth.login = Some(Arc::new(move |_| {
                let marker = marker.clone();
                Box::pin(async move {
                    marker.notify_one();
                    Ok(ApiKeyCredential {
                        key: Some("logged-in".to_string()),
                        env: None,
                    })
                })
            }));
            let models = create_models(CreateModelsOptions {
                credentials: Some(store.clone()),
                ..Default::default()
            });
            models.set_provider(
                TestProvider::new("p1")
                    .auth(api_key_provider_auth(auth))
                    .arc(),
            );
            let controller = CancellationToken::new();
            let login = tokio::spawn({
                let models = models.clone();
                let signal = controller.clone();
                async move {
                    models
                        .login(
                            "p1",
                            AuthType::ApiKey,
                            Arc::new(TestInteraction {
                                signal: Some(signal),
                            }),
                            LoginOptions::default(),
                        )
                        .await
                }
            });
            logged_in.notified().await;
            if run_modifier_first {
                store.modifier_ran.notified().await;
            } else {
                tokio::task::yield_now().await;
            }
            controller.cancel();

            if run_modifier_first {
                // The write started, so login waits for it and succeeds.
                tokio::time::sleep(Duration::from_millis(20)).await;
                assert!(!login.is_finished());
                store.release.notify_one();
                assert_eq!(
                    login.await.unwrap().unwrap(),
                    api_key_credential("logged-in")
                );
            } else {
                let result = tokio::time::timeout(Duration::from_secs(5), login)
                    .await
                    .expect("an aborted login does not wait for the store");
                assert!(result.unwrap().unwrap_err().is_abort());
            }
        }
    }

    #[tokio::test]
    async fn a_stored_credential_without_a_matching_handler_blocks_ambient_fallback() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let models = models_with_credentials(&credentials);
        // provider has only api-key auth, but an oauth credential is stored (stale config)
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(env_key_auth(Some("env-key"))))
                .arc(),
        );
        store_credential(credentials.as_ref(), "p1", oauth_credential("a", "r", 0)).await;

        assert_eq!(
            models.get_auth("p1", Default::default()).await.unwrap(),
            None
        );
    }

    fn rotating_oauth(counter: Option<Arc<AtomicUsize>>) -> Arc<dyn OAuthAuth> {
        test_oauth(Some(Arc::new(move |credential, _| {
            if let Some(counter) = &counter {
                counter.fetch_add(1, Ordering::SeqCst);
            }
            Box::pin(async move {
                Ok(OAuthCredential {
                    access: "new-token".to_string(),
                    expires: now_millis() + 60 * 60_000,
                    ..credential
                })
            })
        })))
    }

    fn oauth_provider(id: &str, oauth: Arc<dyn OAuthAuth>) -> Arc<dyn Provider> {
        TestProvider::new(id)
            .auth(ProviderAuth {
                api_key: None,
                oauth: Some(oauth),
            })
            .arc()
    }

    #[tokio::test]
    async fn refreshes_expired_oauth_credentials_and_persists_the_rotated_credential() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let models = models_with_credentials(&credentials);
        models.set_provider(oauth_provider("p1", rotating_oauth(None)));
        store_credential(
            credentials.as_ref(),
            "p1",
            oauth_credential("old-token", "r", 0),
        )
        .await;

        let resolution = models
            .get_auth("p1", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolution.auth.api_key.as_deref(), Some("new-token"));
        assert_eq!(
            read_access(credentials.as_ref(), "p1").await.as_deref(),
            Some("new-token")
        );
    }

    #[tokio::test]
    async fn refreshes_oauth_credentials_with_less_than_five_minutes_remaining() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let refreshes = Arc::new(AtomicUsize::new(0));
        let models = models_with_credentials(&credentials);
        models.set_provider(oauth_provider(
            "p1",
            rotating_oauth(Some(refreshes.clone())),
        ));
        store_credential(
            credentials.as_ref(),
            "p1",
            oauth_credential("old-token", "r", now_millis() + 60_000),
        )
        .await;

        let resolution = models
            .get_auth("p1", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolution.auth.api_key.as_deref(), Some("new-token"));
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn honors_a_callers_longer_oauth_minimum_validity() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let refreshes = Arc::new(AtomicUsize::new(0));
        let models = models_with_credentials(&credentials);
        models.set_provider(oauth_provider(
            "p1",
            rotating_oauth(Some(refreshes.clone())),
        ));
        store_credential(
            credentials.as_ref(),
            "p1",
            oauth_credential("old-token", "r", now_millis() + 10 * 60_000),
        )
        .await;

        let resolution = models
            .get_auth(
                "p1",
                AuthResolutionOverrides {
                    min_oauth_validity_ms: Some(30 * 60_000),
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolution.auth.api_key.as_deref(), Some("new-token"));
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
    }

    fn failing_oauth(message: &'static str) -> Arc<dyn OAuthAuth> {
        test_oauth(Some(Arc::new(move |_, _| {
            Box::pin(async move { Err(Error::message(message)) })
        })))
    }

    fn models_error_code(error: &Error) -> Option<ModelsErrorCode> {
        match error {
            Error::Models(error) => Some(error.code),
            _ => None,
        }
    }

    #[tokio::test]
    async fn rejects_with_code_oauth_when_refresh_fails_preserving_the_stored_credential() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        let models = models_with_credentials(&credentials);
        models.set_provider(oauth_provider("p1", failing_oauth("invalid_grant")));
        store_credential(credentials.as_ref(), "p1", oauth_credential("old", "r", 0)).await;

        let error = models.get_auth("p1", Default::default()).await.unwrap_err();
        assert_eq!(models_error_code(&error), Some(ModelsErrorCode::Oauth));
        // credential preserved for retry / re-login
        assert_eq!(
            read_access(credentials.as_ref(), "p1").await.as_deref(),
            Some("old")
        );
    }

    #[tokio::test]
    async fn serializes_concurrent_oauth_refreshes_through_store_modify() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        store_credential(credentials.as_ref(), "p1", oauth_credential("old", "r1", 0)).await;
        let refreshes = Arc::new(AtomicUsize::new(0));
        let counter = refreshes.clone();
        let oauth = test_oauth(Some(Arc::new(move |_, _| {
            let count = counter.fetch_add(1, Ordering::SeqCst) + 1;
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(OAuthCredential {
                    access: format!("new-{count}"),
                    refresh: "r2".to_string(),
                    expires: now_millis() + 60 * 60_000,
                    extra: Default::default(),
                })
            })
        })));
        let models = models_with_credentials(&credentials);
        models.set_provider(oauth_provider("p1", oauth));

        let (a, b) = tokio::join!(
            models.get_auth("p1", Default::default()),
            models.get_auth("p1", Default::default())
        );
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        assert_eq!(a.unwrap().unwrap().auth.api_key.as_deref(), Some("new-1"));
        assert_eq!(b.unwrap().unwrap().auth.api_key.as_deref(), Some("new-1"));
    }

    struct CountingStore {
        base: InMemoryCredentialStore,
        modifies: AtomicUsize,
        fail_read: bool,
        fail_modify: Option<Credential>,
    }

    #[async_trait]
    impl CredentialStore for CountingStore {
        async fn read(
            &self,
            provider_id: &str,
            options: AuthOperationOptions,
        ) -> Result<Option<Credential>> {
            if self.fail_read {
                return Err(Error::message("disk on fire"));
            }
            if let Some(credential) = &self.fail_modify {
                return Ok(Some(credential.clone()));
            }
            self.base.read(provider_id, options).await
        }

        async fn list(&self, options: AuthOperationOptions) -> Result<Vec<CredentialInfo>> {
            self.base.list(options).await
        }

        async fn modify(
            &self,
            provider_id: &str,
            modifier: crate::auth::CredentialModifier,
            options: AuthOperationOptions,
        ) -> Result<Option<Credential>> {
            self.modifies.fetch_add(1, Ordering::SeqCst);
            if self.fail_modify.is_some() {
                return Err(Error::message("disk on fire"));
            }
            self.base.modify(provider_id, modifier, options).await
        }

        async fn delete(&self, provider_id: &str, options: AuthOperationOptions) -> Result<()> {
            self.base.delete(provider_id, options).await
        }
    }

    fn counting_store() -> CountingStore {
        CountingStore {
            base: InMemoryCredentialStore::new(),
            modifies: AtomicUsize::new(0),
            fail_read: false,
            fail_modify: None,
        }
    }

    #[tokio::test]
    async fn valid_oauth_tokens_resolve_without_touching_modify() {
        let store = Arc::new(counting_store());
        store_credential(
            &store.base,
            "p1",
            oauth_credential("valid", "r", now_millis() + 10 * 60_000),
        )
        .await;
        let models = create_models(CreateModelsOptions {
            credentials: Some(store.clone()),
            ..Default::default()
        });
        models.set_provider(oauth_provider("p1", test_oauth(None)));

        let resolution = models
            .get_auth("p1", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolution.auth.api_key.as_deref(), Some("valid"));
        assert_eq!(store.modifies.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn wraps_credential_store_failures_in_models_error() {
        // read failure
        let models = create_models(CreateModelsOptions {
            credentials: Some(Arc::new(CountingStore {
                fail_read: true,
                ..counting_store()
            })),
            ..Default::default()
        });
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(env_key_auth(Some("env-key"))))
                .arc(),
        );
        let error = models.get_auth("p1", Default::default()).await.unwrap_err();
        assert_eq!(models_error_code(&error), Some(ModelsErrorCode::Auth));

        // modify failure during refresh
        let oauth_models = create_models(CreateModelsOptions {
            credentials: Some(Arc::new(CountingStore {
                fail_modify: Some(oauth_credential("old", "r", 0)),
                ..counting_store()
            })),
            ..Default::default()
        });
        oauth_models.set_provider(oauth_provider("p1", test_oauth(None)));
        let error = oauth_models
            .get_auth("p1", Default::default())
            .await
            .unwrap_err();
        assert_eq!(models_error_code(&error), Some(ModelsErrorCode::Auth));
    }

    #[tokio::test]
    async fn keeps_the_underlying_reason_in_wrapped_oauth_refresh_errors() {
        let credentials = Arc::new(InMemoryCredentialStore::new());
        store_credential(credentials.as_ref(), "p1", oauth_credential("old", "r", 0)).await;
        let models = models_with_credentials(&credentials);
        models.set_provider(oauth_provider(
            "p1",
            failing_oauth("token refresh failed (400): invalid_grant"),
        ));

        let error = models.get_auth("p1", Default::default()).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "OAuth refresh failed for p1: token refresh failed (400): invalid_grant"
        );
    }

    #[tokio::test]
    async fn wraps_api_key_auth_failures_in_models_error() {
        let failing = api_key_auth(Arc::new(|_| {
            Box::pin(async { Err(Error::message("nope")) })
        }));
        let models = create_models(Default::default());
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(failing))
                .arc(),
        );
        let error = models.get_auth("p1", Default::default()).await.unwrap_err();
        assert_eq!(models_error_code(&error), Some(ModelsErrorCode::Auth));
    }

    #[tokio::test]
    async fn uses_explicit_request_api_key_and_env_during_provider_auth_resolution() {
        let calls: Arc<Mutex<Vec<ProviderCall>>> = Arc::default();
        let scoped = api_key_auth(Arc::new(|input| {
            Box::pin(async move {
                let account = match input
                    .credential
                    .as_ref()
                    .and_then(|credential| credential.env.as_ref())
                    .and_then(|env| env.get("ACCOUNT_ID").cloned())
                {
                    Some(account) => Some(account),
                    None => input.ctx.env("ACCOUNT_ID").await,
                };
                let (Some(key), Some(account)) = (
                    input.credential.and_then(|credential| credential.key),
                    account,
                ) else {
                    return Ok(None);
                };
                Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(key),
                        base_url: Some(format!("https://example.test/{account}")),
                        ..Default::default()
                    },
                    env: Some([("ACCOUNT_ID".to_string(), account)].into_iter().collect()),
                    source: None,
                }))
            })
        }));
        let models = create_models(Default::default());
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(scoped))
                .calls(&calls)
                .arc(),
        );
        let model = test_model("p1", "model-a");

        let env: ProviderEnv = [("ACCOUNT_ID".to_string(), "acct".to_string())]
            .into_iter()
            .collect();
        models
            .complete_simple(
                &model,
                &context(),
                SimpleStreamOptions::from(StreamOptions {
                    api_key: Some("explicit-key".to_string()),
                    env: Some(env.clone()),
                    ..Default::default()
                }),
            )
            .await;

        let call = calls.lock()[0].clone();
        assert_eq!(call.model.base_url, "https://example.test/acct");
        assert_eq!(call.api_key.as_deref(), Some("explicit-key"));
        assert_eq!(call.env, Some(env));
    }

    #[tokio::test]
    async fn merges_resolved_auth_into_stream_options_explicit_options_win_per_field() {
        let calls: Arc<Mutex<Vec<ProviderCall>>> = Arc::default();
        let resolved = api_key_auth(Arc::new(|_| {
            Box::pin(async {
                Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some("resolved-key".to_string()),
                        headers: Some(
                            [
                                ("Authorization", "Bearer resolved-key"),
                                ("x-a", "auth"),
                                ("x-b", "auth"),
                            ]
                            .into_iter()
                            .map(|(name, value)| (name, value.to_string()))
                            .collect(),
                        ),
                        base_url: Some("https://auth.test/v1".to_string()),
                    },
                    ..Default::default()
                }))
            })
        }));
        let models = create_models(Default::default());
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(resolved))
                .calls(&calls)
                .arc(),
        );
        let model = test_model("p1", "model-a");

        let result = models
            .complete_simple(
                &model,
                &context(),
                SimpleStreamOptions::from(StreamOptions {
                    api_key: Some("explicit-key".to_string()),
                    headers: Some(
                        [("authorization", "Explicit token"), ("x-b", "explicit")]
                            .into_iter()
                            .map(|(name, value)| (name, value.to_string()))
                            .collect(),
                    ),
                    ..Default::default()
                }),
            )
            .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        let call = calls.lock()[0].clone();
        assert_eq!(call.api_key.as_deref(), Some("explicit-key"));
        assert_eq!(
            call.headers.unwrap(),
            [
                ("authorization", "Explicit token"),
                ("x-a", "auth"),
                ("x-b", "explicit")
            ]
            .into_iter()
            .map(|(name, value)| (name, value.to_string()))
            .collect()
        );
        assert_eq!(call.model.base_url, "https://auth.test/v1");

        // without explicit options, resolved auth applies
        let result = models
            .complete_simple(&model, &context(), SimpleStreamOptions::default())
            .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(calls.lock()[1].api_key.as_deref(), Some("resolved-key"));
    }

    #[tokio::test]
    async fn adds_model_headers_only_for_model_auth_and_transforms_assembled_headers_once() {
        let calls: Arc<Mutex<Vec<ProviderCall>>> = Arc::default();
        let models = create_models(Default::default());
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(env_key_auth(Some("key"))))
                .calls(&calls)
                .arc(),
        );
        let mut model = test_model("p1", "model-a");
        model.headers = Some(
            [("x-model", "model"), ("x-shared", "model")]
                .into_iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        );
        let headers = |entries: &[(&str, &str)]| -> ProviderHeaders {
            entries
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect()
        };

        assert_eq!(
            models
                .get_auth("p1", Default::default())
                .await
                .unwrap()
                .unwrap()
                .auth
                .headers,
            None
        );
        assert_eq!(
            models
                .get_auth(&model, Default::default())
                .await
                .unwrap()
                .unwrap()
                .auth
                .headers,
            Some(headers(&[("x-model", "model"), ("x-shared", "model")]))
        );

        let transforms = Arc::new(AtomicUsize::new(0));
        let counter = transforms.clone();
        let expected = headers(&[
            ("x-model", "model"),
            ("x-explicit", "explicit"),
            ("X-Shared", "explicit"),
        ]);
        let transform: HeadersTransform = Arc::new(move |assembled| {
            counter.fetch_add(1, Ordering::SeqCst);
            let expected = expected.clone();
            Box::pin(async move {
                assert_eq!(assembled, expected);
                assert_eq!(
                    assembled
                        .iter()
                        .map(|(name, _)| name.as_str())
                        .collect::<Vec<_>>(),
                    vec!["x-model", "x-explicit", "X-Shared"]
                );
                let mut transformed = assembled;
                transformed.insert("x-transformed", "yes".to_string());
                Ok(transformed)
            })
        });
        models
            .complete_simple(
                &model,
                &context(),
                ModelsOptions {
                    options: SimpleStreamOptions::from(StreamOptions {
                        headers: Some(headers(&[
                            ("x-explicit", "explicit"),
                            ("X-Shared", "explicit"),
                        ])),
                        ..Default::default()
                    }),
                    transform_headers: Some(transform),
                },
            )
            .await;

        assert_eq!(transforms.load(Ordering::SeqCst), 1);
        assert_eq!(
            calls.lock()[0].headers,
            Some(headers(&[
                ("x-model", "model"),
                ("x-explicit", "explicit"),
                ("X-Shared", "explicit"),
                ("x-transformed", "yes"),
            ]))
        );
    }

    #[tokio::test]
    async fn produces_an_error_stream_for_unknown_providers_instead_of_throwing() {
        let models = create_models(Default::default());
        let result = models
            .complete_simple(
                &test_model("ghost", "model-a"),
                &context(),
                SimpleStreamOptions::default(),
            )
            .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(
            result
                .error_message
                .unwrap()
                .contains("Unknown provider: ghost")
        );
    }

    #[tokio::test]
    async fn reports_unconfigured_providers_as_stream_errors() {
        let models = create_models(Default::default());
        models.set_provider(
            TestProvider::new("p1")
                .auth(api_key_provider_auth(env_key_auth(None)))
                .arc(),
        );
        let result = models
            .complete(
                &test_model("p1", "model-a"),
                &context(),
                StreamOptions::default(),
            )
            .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Provider is not configured: p1")
        );
    }

    #[tokio::test]
    async fn streams_through_the_provider() {
        let models = create_models(Default::default());
        models.set_provider(TestProvider::new("p1").arc());
        let model = test_model("p1", "model-a");

        let stream = models.stream_simple(&model, &context(), SimpleStreamOptions::default());
        let events: Vec<_> = stream
            .clone()
            .map(|event| event.event_type())
            .collect()
            .await;
        assert_eq!(events, vec!["start", "done"]);
        assert_eq!(stream.result().await.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn deferred_calls_fail_for_providers_without_deferred_support() {
        let models = create_models(Default::default());
        models.set_provider(TestProvider::new("p1").arc());
        let model = test_model("p1", "model-a");
        let handle = DeferredHandle {
            provider: "p1".to_string(),
            model_id: "model-a".to_string(),
            api: "test-api".to_string(),
            id: "h".to_string(),
            expires_at: None,
            poll_after_ms: None,
            data: None,
        };
        let fetched = models
            .fetch_deferred(&model, &handle, DeferredFetchOptions::default())
            .await;
        assert_eq!(
            fetched.error_message.as_deref(),
            Some("Provider p1 does not support deferred responses")
        );
        let cancelled = models
            .cancel_deferred(&model, &handle, DeferredCancelOptions::default())
            .await
            .unwrap_err();
        assert_eq!(
            cancelled.to_string(),
            "Provider p1 does not support deferred responses"
        );
    }

    #[test]
    fn created_providers_need_an_api_and_report_missing_implementations() {
        let error = create_provider(CreateProviderOptions {
            id: "empty".to_string(),
            auth: ambient_auth(),
            ..Default::default()
        })
        .err()
        .unwrap();
        assert_eq!(
            error.to_string(),
            "Provider empty: at least one of \"api\", \"images\", \"classifiers\", or \"embeddings\" is required."
        );
    }

    #[tokio::test]
    async fn created_providers_dispatch_by_api_and_error_for_unknown_apis() {
        let provider = create_provider(CreateProviderOptions {
            id: "mixed".to_string(),
            auth: ambient_auth(),
            api: Some(ProviderApi::ByApi(
                [(
                    "known".to_string(),
                    match chat_streams() {
                        ProviderApi::Single(streams) => streams,
                        ProviderApi::ByApi(_) => unreachable!(),
                    },
                )]
                .into_iter()
                .collect(),
            )),
            ..Default::default()
        })
        .unwrap();
        let result = provider
            .stream(
                test_model("mixed", "m"),
                normalize_context(&context()),
                StreamOptions::default(),
            )
            .result()
            .await;
        assert_eq!(
            result.error_message.as_deref(),
            Some("Provider mixed has no API implementation for \"test-api\"")
        );
    }

    // model-types.test.ts

    fn image_model(provider: &str, id: &str) -> ImageModel {
        ImageModel {
            id: id.to_string(),
            name: id.to_string(),
            api: "test-images".to_string(),
            provider: provider.to_string(),
            base_url: "https://example.test/v1".to_string(),
            input: vec![ModelInput::Text],
            output: vec![ModelInput::Image],
            ..Default::default()
        }
    }

    fn embedding_model(provider: &str, id: &str) -> EmbeddingModel {
        EmbeddingModel {
            id: id.to_string(),
            name: id.to_string(),
            api: "test-embeddings".to_string(),
            provider: provider.to_string(),
            base_url: "https://example.test/v1".to_string(),
            input: vec![ModelInput::Text],
            context_window: 8192,
            dimensions: 3,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn chat_models_without_a_type_work_through_a_handwritten_provider() {
        let models = create_models(Default::default());
        models.set_provider(TestProvider::new("handwritten").arc());

        let model = models.get_model("handwritten", "model-a").unwrap();
        assert_eq!(model.model_type, None);
        let any = AnyModel::Chat(model.clone());
        assert_eq!(get_model_type(&any), ModelType::Chat);
        assert!(has_api(&any, &model.api));
        let typed = AnyModel::Chat(Model {
            model_type: Some(ModelType::Chat),
            ..model.clone()
        });
        assert!(models_are_equal(Some(&any), Some(&typed)));
        let image = AnyModel::Image(image_model(&model.provider, &model.id));
        assert!(!models_are_equal(Some(&any), Some(&image)));
        assert!(!models_are_equal(Some(&any), None));
        assert_eq!(
            models.get_models_of_type(ModelType::Chat, Some("handwritten")),
            vec![any.clone()]
        );
        assert_eq!(models.get_all_models(Some("handwritten")), vec![any]);
        assert!(
            models
                .get_models_of_type(ModelType::Image, Some("handwritten"))
                .is_empty()
        );

        let result = models
            .complete(&model, &context(), StreamOptions::default())
            .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[test]
    fn narrow_mixed_lists_with_is_model_type() {
        let mixed = [
            AnyModel::Chat(test_model("p", "c")),
            AnyModel::Chat(Model {
                model_type: Some(ModelType::Chat),
                ..test_model("p", "typed")
            }),
            AnyModel::Image(image_model("p", "i")),
        ];
        let of_type = |model_type| {
            mixed
                .iter()
                .filter(|model| is_model_type(model, model_type))
                .map(|model| model.id())
                .collect::<Vec<_>>()
        };
        assert_eq!(of_type(ModelType::Chat), vec!["c", "typed"]);
        assert_eq!(of_type(ModelType::Image), vec!["i"]);
        assert!(of_type(ModelType::Classifier).is_empty());
    }

    #[tokio::test]
    async fn get_all_available_applies_chat_filters_to_mixed_catalogs() {
        let provider = create_provider(CreateProviderOptions {
            id: "mixed".to_string(),
            auth: ambient_auth(),
            models: vec![
                AnyModel::Chat(test_model("mixed", "keep")),
                AnyModel::Chat(test_model("mixed", "drop")),
                AnyModel::Image(image_model("mixed", "image")),
            ],
            filter_models: Some(Arc::new(|models, _| {
                models
                    .iter()
                    .filter(|model| model.id == "keep")
                    .cloned()
                    .collect()
            })),
            api: Some(chat_streams()),
            ..Default::default()
        })
        .unwrap();
        let models = create_models(Default::default());
        models.set_provider(provider);
        let available = models
            .get_all_available(None, Default::default())
            .await
            .unwrap();
        assert_eq!(
            available.iter().map(AnyModel::id).collect::<Vec<_>>(),
            vec!["keep", "image"]
        );
        let images = models
            .get_available_of_type(ModelType::Image, None, Default::default())
            .await
            .unwrap();
        assert_eq!(images.len(), 1);
    }

    // images-models.test.ts

    struct EnvAuthContext(HashMap<String, String>);

    #[async_trait]
    impl AuthContext for EnvAuthContext {
        async fn env(&self, name: &str) -> Option<String> {
            self.0.get(name).cloned()
        }
        async fn file_exists(&self, _path: &str) -> bool {
            false
        }
    }

    fn fake_auth_context(env: &[(&str, &str)]) -> Arc<dyn AuthContext> {
        Arc::new(EnvAuthContext(
            env.iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        ))
    }

    fn ok_images(model: &ImageModel) -> AssistantImages {
        AssistantImages {
            output: vec![crate::types::UserContent::Image(
                crate::types::ImageContent {
                    data: "aGk=".to_string(),
                    mime_type: "image/png".to_string(),
                },
            )],
            ..AssistantImages::empty_for(model)
        }
    }

    type ImageCalls = Arc<Mutex<Vec<(ImageModel, ImagesOptions)>>>;

    fn recording_images(calls: &ImageCalls) -> Arc<dyn ProviderImages> {
        let calls = calls.clone();
        struct Recording(ImageCalls);

        #[async_trait]
        impl ProviderImages for Recording {
            async fn generate_images(
                &self,
                model: ImageModel,
                _context: ImagesContext,
                options: ImagesOptions,
            ) -> AssistantImages {
                let result = ok_images(&model);
                self.0.lock().push((model, options));
                result
            }
        }

        Arc::new(Recording(calls))
    }

    fn env_var_auth(env_var: Option<&'static str>) -> ProviderAuth {
        ProviderAuth {
            api_key: Some(Arc::new(api_key_auth(Arc::new(move |input| {
                Box::pin(async move {
                    let Some(env_var) = env_var else {
                        return Ok(Some(AuthResult::default()));
                    };
                    let stored = input
                        .credential
                        .as_ref()
                        .and_then(|credential| credential.key.clone());
                    let has_credential = stored.is_some();
                    let key = match stored {
                        Some(key) => Some(key),
                        None => input.ctx.env(env_var).await,
                    };
                    Ok(key.map(|key| AuthResult {
                        auth: ModelAuth {
                            api_key: Some(key),
                            ..Default::default()
                        },
                        env: None,
                        source: Some(if has_credential { "stored" } else { env_var }.to_string()),
                    }))
                })
            })))),
            oauth: None,
        }
    }

    fn image_test_provider(
        id: &str,
        models: Option<Vec<AnyModel>>,
        env_var: Option<&'static str>,
        calls: &ImageCalls,
        image_apis: &[&str],
    ) -> Arc<dyn Provider> {
        create_provider(CreateProviderOptions {
            id: id.to_string(),
            auth: env_var_auth(env_var),
            models: models.unwrap_or_else(|| vec![AnyModel::Image(image_model(id, "model-a"))]),
            api: Some(ProviderApi::ByApi(
                [(
                    "test-chat".to_string(),
                    match chat_streams() {
                        ProviderApi::Single(streams) => streams,
                        ProviderApi::ByApi(_) => unreachable!(),
                    },
                )]
                .into_iter()
                .collect(),
            )),
            images: Some(
                image_apis
                    .iter()
                    .map(|api| (api.to_string(), recording_images(calls)))
                    .collect(),
            ),
            ..Default::default()
        })
        .unwrap()
    }

    fn images_context() -> ImagesContext {
        ImagesContext::builder().text("a red circle").build()
    }

    fn image_of(models: &Models, provider: &str, id: &str) -> ImageModel {
        models
            .get_model_of_type(ModelType::Image, provider, id)
            .and_then(|model| model.as_image().cloned())
            .unwrap()
    }

    #[test]
    fn lists_chat_image_and_all_models_through_typed_accessors() {
        let calls = ImageCalls::default();
        let models = create_models(Default::default());
        models.set_provider(image_test_provider(
            "p1",
            Some(vec![
                AnyModel::Chat(test_model("p1", "c1")),
                AnyModel::Image(image_model("p1", "i1")),
                AnyModel::Image(image_model("p1", "i2")),
            ]),
            None,
            &calls,
            &["test-images"],
        ));
        models.set_provider(image_test_provider(
            "p2",
            Some(vec![AnyModel::Image(image_model("p2", "i3"))]),
            None,
            &calls,
            &["test-images"],
        ));
        let ids = |models: Vec<AnyModel>| {
            models
                .iter()
                .map(|model| model.id().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ids(models.get_models_of_type(ModelType::Chat, None)),
            ["c1"]
        );
        assert_eq!(
            ids(models.get_models_of_type(ModelType::Image, None)),
            ["i1", "i2", "i3"]
        );
        assert_eq!(
            ids(models.get_models_of_type(ModelType::Image, Some("p1"))),
            ["i1", "i2"]
        );
        assert_eq!(ids(models.get_all_models(None)), ["c1", "i1", "i2", "i3"]);
        assert_eq!(models.get_model("p1", "c1").unwrap().id, "c1");
        assert!(models.get_model("p1", "i1").is_none());
        assert!(
            models
                .get_model_of_type(ModelType::Image, "p1", "i1")
                .is_some()
        );
        assert!(
            models
                .get_model_of_type(ModelType::Image, "p1", "c1")
                .is_none()
        );
    }

    #[tokio::test]
    async fn splits_available_models_by_type() {
        let calls = ImageCalls::default();
        let models = create_models(CreateModelsOptions {
            auth_context: Some(fake_auth_context(&[("KEY", "k")])),
            ..Default::default()
        });
        models.set_provider(image_test_provider(
            "p1",
            Some(vec![
                AnyModel::Chat(test_model("p1", "c1")),
                AnyModel::Image(image_model("p1", "i1")),
            ]),
            Some("KEY"),
            &calls,
            &["test-images"],
        ));
        models.set_provider(image_test_provider(
            "p2",
            Some(vec![AnyModel::Image(image_model("p2", "i2"))]),
            Some("MISSING"),
            &calls,
            &["test-images"],
        ));
        let chat = models
            .get_available(None, Default::default())
            .await
            .unwrap();
        assert_eq!(ids(&chat), ["c1"]);
        let images = models
            .get_available_of_type(ModelType::Image, None, Default::default())
            .await
            .unwrap();
        assert_eq!(images.iter().map(AnyModel::id).collect::<Vec<_>>(), ["i1"]);
        let all = models
            .get_all_available(None, Default::default())
            .await
            .unwrap();
        assert_eq!(
            all.iter().map(AnyModel::id).collect::<Vec<_>>(),
            ["c1", "i1"]
        );
    }

    #[tokio::test]
    async fn resolves_auth_and_merges_it_into_image_requests_explicit_options_win() {
        let calls = ImageCalls::default();
        let models = create_models(CreateModelsOptions {
            auth_context: Some(fake_auth_context(&[("TEST_KEY", "env-key")])),
            ..Default::default()
        });
        models.set_provider(image_test_provider(
            "p1",
            None,
            Some("TEST_KEY"),
            &calls,
            &["test-images"],
        ));
        let model = image_of(&models, "p1", "model-a");
        let api_key = |result: Option<AuthResult>| result.and_then(|result| result.auth.api_key);
        assert_eq!(
            api_key(models.get_auth(&model, Default::default()).await.unwrap()).as_deref(),
            Some("env-key")
        );
        assert_eq!(
            api_key(models.get_auth("p1", Default::default()).await.unwrap()).as_deref(),
            Some("env-key")
        );
        let explicit = AuthResolutionOverrides {
            api_key: Some("explicit-key".to_string()),
            ..Default::default()
        };
        assert_eq!(
            api_key(models.get_auth(&model, explicit).await.unwrap()).as_deref(),
            Some("explicit-key")
        );

        let result = models
            .generate_images(&model, &images_context(), ImagesOptions::default())
            .await;
        assert_eq!(result.stop_reason, crate::types::ImagesStopReason::Stop);
        assert_eq!(calls.lock()[0].1.api_key.as_deref(), Some("env-key"));

        models
            .generate_images(
                &model,
                &images_context(),
                ImagesOptions {
                    api_key: Some("explicit".to_string()),
                    ..Default::default()
                },
            )
            .await;
        assert_eq!(calls.lock()[1].1.api_key.as_deref(), Some("explicit"));
    }

    #[tokio::test]
    async fn image_requests_merge_provider_env_and_apply_header_transforms() {
        let calls = ImageCalls::default();
        let models = create_models(Default::default());
        let resolve: ResolveFn = Arc::new(|_| {
            Box::pin(async {
                Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some("provider-key".to_string()),
                        headers: Some([("x-base", Some("1".to_string()))].into_iter().collect()),
                        base_url: None,
                    },
                    env: Some(
                        [("PROVIDER_ONLY", "provider"), ("SHARED", "provider")]
                            .into_iter()
                            .map(|(name, value)| (name.to_string(), value.to_string()))
                            .collect(),
                    ),
                    source: None,
                }))
            })
        });
        models.set_provider(
            create_provider(CreateProviderOptions {
                id: "p1".to_string(),
                auth: ProviderAuth {
                    api_key: Some(Arc::new(api_key_auth(resolve))),
                    oauth: None,
                },
                models: vec![AnyModel::Image(image_model("p1", "model-a"))],
                images: Some(
                    [("test-images".to_string(), recording_images(&calls))]
                        .into_iter()
                        .collect(),
                ),
                ..Default::default()
            })
            .unwrap(),
        );
        let model = image_of(&models, "p1", "model-a");
        let transform: HeadersTransform = Arc::new(|mut headers: ProviderHeaders| {
            headers.insert("x-extra", Some("2".to_string()));
            Box::pin(async move { Ok(headers) })
        });
        models
            .generate_images(
                &model,
                &images_context(),
                ModelsOptions {
                    options: ImagesOptions {
                        api_key: Some("request-key".to_string()),
                        env: Some(
                            [("REQUEST_ONLY", "request"), ("SHARED", "request")]
                                .into_iter()
                                .map(|(name, value)| (name.to_string(), value.to_string()))
                                .collect(),
                        ),
                        ..Default::default()
                    },
                    transform_headers: Some(transform),
                },
            )
            .await;
        let calls = calls.lock();
        let options = &calls[0].1;
        assert_eq!(options.api_key.as_deref(), Some("request-key"));
        let env = options.env.clone().unwrap();
        assert_eq!(env.len(), 3);
        assert_eq!(env["PROVIDER_ONLY"], "provider");
        assert_eq!(env["REQUEST_ONLY"], "request");
        assert_eq!(env["SHARED"], "request");
        assert_eq!(
            options.headers,
            Some(
                [
                    ("x-base", Some("1".to_string())),
                    ("x-extra", Some("2".to_string()))
                ]
                .into_iter()
                .collect()
            )
        );
    }

    #[tokio::test]
    async fn image_generation_returns_error_results_instead_of_rejecting() {
        use crate::types::ImagesStopReason;

        let models = create_models(CreateModelsOptions {
            auth_context: Some(fake_auth_context(&[])),
            ..Default::default()
        });
        let ghost = models
            .generate_images(
                &image_model("ghost", "m"),
                &images_context(),
                ImagesOptions::default(),
            )
            .await;
        assert_eq!(ghost.stop_reason, ImagesStopReason::Error);
        assert!(
            ghost
                .error_message
                .unwrap()
                .contains("Unknown provider: ghost")
        );

        // Unconfigured auth is an error, matching stream().
        let calls = ImageCalls::default();
        models.set_provider(image_test_provider(
            "p1",
            None,
            Some("MISSING"),
            &calls,
            &["test-images"],
        ));
        let model = image_of(&models, "p1", "model-a");
        assert!(
            models
                .get_auth(&model, Default::default())
                .await
                .unwrap()
                .is_none()
        );
        let unconfigured = models
            .generate_images(&model, &images_context(), ImagesOptions::default())
            .await;
        assert_eq!(unconfigured.stop_reason, ImagesStopReason::Error);
        assert!(
            unconfigured
                .error_message
                .unwrap()
                .contains("not configured")
        );
        assert!(calls.lock().is_empty());

        let signal = CancellationToken::new();
        signal.cancel();
        let cancelled = models
            .generate_images(
                &model,
                &images_context(),
                ImagesOptions {
                    signal: Some(signal),
                    ..Default::default()
                },
            )
            .await;
        assert_eq!(cancelled.stop_reason, ImagesStopReason::Aborted);
        assert!(calls.lock().is_empty());

        // A provider without any images implementation rejects image models it lists.
        models.set_provider(
            create_provider(CreateProviderOptions {
                id: "chat-only".to_string(),
                auth: ambient_auth(),
                models: vec![AnyModel::Image(image_model("chat-only", "i"))],
                api: Some(chat_streams()),
                ..Default::default()
            })
            .unwrap(),
        );
        let unsupported = models
            .generate_images(
                &image_of(&models, "chat-only", "i"),
                &images_context(),
                ImagesOptions::default(),
            )
            .await;
        assert_eq!(unsupported.stop_reason, ImagesStopReason::Error);
        assert!(
            unsupported
                .error_message
                .unwrap()
                .contains("does not support image generation")
        );

        // An images map without the model's api yields a provider error result.
        models.set_provider(image_test_provider(
            "wrong-api",
            Some(vec![AnyModel::Image(image_model("wrong-api", "i"))]),
            None,
            &calls,
            &["other-images"],
        ));
        let missing_api = models
            .generate_images(
                &image_of(&models, "wrong-api", "i"),
                &images_context(),
                ImagesOptions::default(),
            )
            .await;
        assert_eq!(missing_api.stop_reason, ImagesStopReason::Error);
        assert!(
            missing_api
                .error_message
                .unwrap()
                .contains("no image generation implementation for \"test-images\"")
        );
    }

    #[test]
    fn requires_at_least_one_concrete_operation_implementation() {
        let create =
            |api: Option<ProviderApi>,
             images: Option<IndexMap<String, Arc<dyn ProviderImages>>>| {
                create_provider(CreateProviderOptions {
                    id: "empty".to_string(),
                    auth: ambient_auth(),
                    api,
                    images,
                    ..Default::default()
                })
            };
        let message = "at least one of \"api\", \"images\", \"classifiers\", or \"embeddings\"";
        for result in [
            create(None, None),
            create(Some(ProviderApi::ByApi(IndexMap::new())), None),
            create(None, Some(IndexMap::new())),
        ] {
            assert!(result.err().unwrap().to_string().contains(message));
        }
    }

    #[tokio::test]
    async fn stored_and_fetched_models_of_unknown_types_are_dropped_instead_of_failing_the_refresh()
    {
        let chat = |id: &str| serde_json::to_value(test_model("dyn", id)).unwrap();
        let image = |id: &str| serde_json::to_value(image_model("dyn", id)).unwrap();
        let embedding = |id: &str| serde_json::to_value(embedding_model("dyn", id)).unwrap();
        let with_type = |mut model: Value, model_type: &str| {
            model["type"] = json!(model_type);
            model
        };
        // Pi writes the raw entry into the store; a Rust store deserializes it.
        let stored: ModelsStoreEntry = serde_json::from_value(json!({
            "models": [
                chat("stored-chat"),
                image("stored-image"),
                embedding("stored-embedding"),
                with_type(chat("future-audio"), "audio"),
                with_type(image("future-video"), "video"),
            ],
        }))
        .unwrap();
        let models_store = Arc::new(InMemoryModelsStore::new());
        models_store
            .write("dyn", stored, Default::default())
            .await
            .unwrap();

        let fetched: Arc<Mutex<Vec<Value>>> = Arc::default();
        let source = fetched.clone();
        let models = create_models(CreateModelsOptions {
            models_store: Some(models_store.clone()),
            ..Default::default()
        });
        models.set_provider(
            create_provider(CreateProviderOptions {
                id: "dyn".to_string(),
                auth: ambient_auth(),
                fetch_models: Some(Arc::new(move |_| {
                    let fetched = source.lock().clone();
                    Box::pin(async move { Ok(known_models_from_values(fetched)?) })
                })),
                api: Some(chat_streams()),
                ..Default::default()
            })
            .unwrap(),
        );
        let ids = |models: &Models| {
            models
                .get_all_models(Some("dyn"))
                .iter()
                .map(|model| model.id().to_string())
                .collect::<Vec<_>>()
        };

        let restored = models
            .refresh(ModelsRefreshOptions {
                providers: Some(vec!["dyn".to_string()]),
                allow_network: Some(false),
                ..Default::default()
            })
            .await;
        assert!(restored.errors.is_empty());
        assert_eq!(
            ids(&models),
            ["stored-chat", "stored-image", "stored-embedding"]
        );
        assert_eq!(
            models
                .get_model_of_type(ModelType::Embedding, "dyn", "stored-embedding")
                .and_then(|model| model.as_embedding().map(|model| model.dimensions)),
            Some(3)
        );

        *fetched.lock() = vec![
            chat("fetched-chat"),
            embedding("fetched-embedding"),
            with_type(image("fetched-video"), "video"),
        ];
        let refreshed = models
            .refresh(ModelsRefreshOptions {
                providers: Some(vec!["dyn".to_string()]),
                ..Default::default()
            })
            .await;
        assert!(refreshed.errors.is_empty());
        assert_eq!(ids(&models), ["fetched-chat", "fetched-embedding"]);
        let entry = models_store
            .read("dyn", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            entry.models.iter().map(AnyModel::id).collect::<Vec<_>>(),
            ["fetched-chat", "fetched-embedding"]
        );
        assert!(matches!(entry.models[1], AnyModel::Embedding(_)));
        assert!(has_known_model_type(&with_type(chat("c"), "classifier")));
        assert!(has_known_model_type(&json!({ "type": null })));
    }

    // Port of `test/classifier-models.test.ts`.

    use crate::types::{ClassifierAnswer, ClassifierStopReason};

    fn classifier_model(provider: &str, id: &str) -> ClassifierModel {
        ClassifierModel {
            id: id.to_string(),
            name: id.to_string(),
            api: "test-classifier".to_string(),
            provider: provider.to_string(),
            base_url: "https://example.test/v1".to_string(),
            model_type: Default::default(),
            input: vec![ModelInput::Text],
            input_limits: None,
            cost: ModelCost::default(),
            context_window: 1000,
            headers: None,
        }
    }

    fn approval_context() -> ClassifierContext {
        serde_json::from_value(json!({
            "state": { "text": "yes" },
            "questions": {
                "approved": {
                    "type": "bool",
                    "instructions": "Does this express approval?",
                    "criteria": { "true": "Approval", "false": "No approval" },
                },
            },
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn keeps_chat_and_classifier_entries_with_the_same_provider_and_id_separate() {
        struct Approves;

        #[async_trait]
        impl ProviderClassifier for Approves {
            async fn classify(
                &self,
                model: ClassifierModel,
                _: ClassifierContext,
                _: ClassifierOptions,
            ) -> ClassifierResult {
                ClassifierResult {
                    answers: [(
                        "approved".to_string(),
                        ClassifierAnswer::Bool { probability: 0.9 },
                    )]
                    .into_iter()
                    .collect(),
                    ..ClassifierResult::empty_for(&model)
                }
            }
        }

        let chat = test_model("test", "shared");
        let classifier = classifier_model("test", "shared");
        let provider = create_provider(CreateProviderOptions {
            id: "test".to_string(),
            auth: ambient_auth(),
            models: vec![
                AnyModel::Chat(chat),
                AnyModel::Classifier(classifier.clone()),
            ],
            api: Some(chat_streams()),
            classifiers: Some(
                [(
                    "test-classifier".to_string(),
                    Arc::new(Approves) as Arc<dyn ProviderClassifier>,
                )]
                .into_iter()
                .collect(),
            ),
            ..Default::default()
        })
        .unwrap();
        let models = create_models(Default::default());
        models.set_provider(provider);

        let listed_chat = models.get_model("test", "shared").unwrap();
        assert_eq!(
            get_model_type(&AnyModel::Chat(listed_chat)),
            ModelType::Chat
        );
        assert_eq!(
            models
                .get_model_of_type(ModelType::Classifier, "test", "shared")
                .and_then(|model| model.as_classifier().map(|model| model.model_type)),
            Some(Default::default())
        );
        assert_eq!(
            models.get_models_of_type(ModelType::Classifier, None),
            vec![AnyModel::Classifier(classifier.clone())]
        );
        assert_eq!(models.get_all_models(None).len(), 2);
        assert_eq!(
            models
                .get_available_of_type(ModelType::Classifier, None, Default::default())
                .await
                .unwrap(),
            vec![AnyModel::Classifier(classifier.clone())]
        );
        assert_eq!(
            models
                .classify(
                    &classifier,
                    &approval_context(),
                    ClassifierOptions::default()
                )
                .await
                .answers["approved"],
            ClassifierAnswer::Bool { probability: 0.9 }
        );
    }

    #[tokio::test]
    async fn rejects_chat_models_at_the_classifier_entry_point_at_runtime() {
        let chat = test_model("test", "chat");
        let models = create_models(Default::default());
        models.set_provider(
            create_provider(CreateProviderOptions {
                id: "test".to_string(),
                auth: ambient_auth(),
                models: vec![AnyModel::Chat(chat.clone())],
                api: Some(chat_streams()),
                ..Default::default()
            })
            .unwrap(),
        );

        // Pi casts a chat model to `ClassifierModel`; the typed signature
        // rules that out, so the runtime check is exercised on `AnyModel`
        // and a relabeled model is refused by the chat-only provider.
        assert_eq!(
            crate::utils::model_operations::assert_classifier_model(&AnyModel::Chat(chat.clone()))
                .unwrap_err()
                .message,
            "Model test/chat is not a classifier model"
        );
        let disguised = ClassifierModel {
            id: chat.id,
            provider: chat.provider,
            api: chat.api,
            ..classifier_model("test", "chat")
        };
        let result = models
            .classify(
                &disguised,
                &approval_context(),
                ClassifierOptions::default(),
            )
            .await;
        assert_eq!(result.stop_reason, ClassifierStopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Provider test does not support classification")
        );
    }

    #[test]
    fn exposes_jev_only_through_classifier_catalog_accessors() {
        use crate::providers::all::{
            builtin_models, get_all_builtin_models, get_builtin_classifier_model,
            get_builtin_classifier_models,
        };
        let jev = get_builtin_classifier_model("typesafe", "jev-latest").unwrap();
        assert_eq!(jev.api, "typesafe-system-one");
        assert_eq!(jev.provider, "typesafe");
        assert_eq!(jev.context_window, 64000);
        assert_eq!(get_builtin_classifier_models("typesafe"), vec![jev.clone()]);
        assert_eq!(
            get_all_builtin_models("typesafe"),
            vec![AnyModel::Classifier(jev.clone())]
        );

        let models = builtin_models(Default::default());
        assert!(models.get_model("typesafe", "jev-latest").is_none());
        assert_eq!(
            models.get_model_of_type(ModelType::Classifier, "typesafe", "jev-latest"),
            Some(AnyModel::Classifier(jev))
        );
    }

    #[test]
    fn routes_openrouter_classifier_models_through_the_system_one_api() {
        use crate::providers::all::{builtin_models, get_builtin_classifier_models};
        let models = builtin_models(Default::default());
        let classifiers = get_builtin_classifier_models("openrouter");
        assert!(!classifiers.is_empty());
        for model in classifiers {
            assert_eq!(model.api, "typesafe-system-one");
            assert_eq!(model.base_url, "https://openrouter.ai/api/v1");
            assert!(models.get_model("openrouter", &model.id).is_none());
            assert_eq!(
                models.get_model_of_type(ModelType::Classifier, "openrouter", &model.id),
                Some(AnyModel::Classifier(model))
            );
        }
    }

    // Embeddings (ai.rs extra), mirroring the image tests above.

    type EmbeddingCalls = Arc<Mutex<Vec<(EmbeddingModel, EmbeddingsOptions)>>>;

    fn recording_embeddings(calls: &EmbeddingCalls) -> Arc<dyn ProviderEmbeddings> {
        struct Recording(EmbeddingCalls);

        #[async_trait]
        impl ProviderEmbeddings for Recording {
            async fn embed(
                &self,
                model: EmbeddingModel,
                context: EmbeddingsContext,
                options: EmbeddingsOptions,
            ) -> EmbeddingsResult {
                let result = EmbeddingsResult {
                    embeddings: context
                        .input
                        .iter()
                        .map(|input| crate::types::EmbeddingVector::Float(vec![input.len() as f32]))
                        .collect(),
                    ..EmbeddingsResult::empty_for(&model)
                };
                self.0.lock().push((model, options));
                result
            }
        }

        Arc::new(Recording(calls.clone()))
    }

    fn embedding_test_provider(
        id: &str,
        env_var: Option<&'static str>,
        calls: &EmbeddingCalls,
        embedding_apis: &[&str],
    ) -> Arc<dyn Provider> {
        create_provider(CreateProviderOptions {
            id: id.to_string(),
            auth: env_var_auth(env_var),
            models: vec![
                AnyModel::Chat(test_model(id, "chat")),
                AnyModel::Image(image_model(id, "image")),
                AnyModel::Embedding(embedding_model(id, "model-a")),
            ],
            embeddings: Some(
                embedding_apis
                    .iter()
                    .map(|api| (api.to_string(), recording_embeddings(calls)))
                    .collect(),
            ),
            ..Default::default()
        })
        .unwrap()
    }

    fn embedding_of(models: &Models, provider: &str, id: &str) -> EmbeddingModel {
        models
            .get_model_of_type(ModelType::Embedding, provider, id)
            .and_then(|model| model.as_embedding().cloned())
            .unwrap()
    }

    fn embeddings_context() -> EmbeddingsContext {
        EmbeddingsContext {
            input: vec!["a".to_string(), "bcd".to_string()],
        }
    }

    #[tokio::test]
    async fn lists_embedding_models_through_typed_accessors_and_resolves_auth() {
        let calls = EmbeddingCalls::default();
        let models = create_models(CreateModelsOptions {
            auth_context: Some(fake_auth_context(&[("TEST_KEY", "env-key")])),
            ..Default::default()
        });
        models.set_provider(embedding_test_provider(
            "p1",
            Some("TEST_KEY"),
            &calls,
            &["test-embeddings"],
        ));
        let ids = |list: Vec<AnyModel>| {
            list.iter()
                .map(|model| model.id().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ids(models.get_models_of_type(ModelType::Embedding, None)),
            ["model-a"]
        );
        assert_eq!(
            ids(models.get_all_models(Some("p1"))),
            ["chat", "image", "model-a"]
        );
        assert_eq!(
            models
                .get_models(Some("p1"))
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["chat"]
        );
        assert!(
            models
                .get_model_of_type(ModelType::Embedding, "p1", "chat")
                .is_none()
        );
        assert_eq!(
            ids(models
                .get_available_of_type(ModelType::Embedding, None, Default::default())
                .await
                .unwrap()),
            ["model-a"]
        );

        let model = embedding_of(&models, "p1", "model-a");
        let result = models
            .embed(&model, &embeddings_context(), EmbeddingsOptions::default())
            .await;
        assert_eq!(result.stop_reason, EmbeddingsStopReason::Stop);
        assert_eq!(result.embeddings.len(), 2);
        assert_eq!(calls.lock()[0].1.api_key.as_deref(), Some("env-key"));

        let transform: HeadersTransform = Arc::new(|mut headers: ProviderHeaders| {
            headers.insert("x-extra", Some("2".to_string()));
            Box::pin(async move { Ok(headers) })
        });
        models
            .embed(
                &model,
                &embeddings_context(),
                ModelsOptions {
                    options: EmbeddingsOptions {
                        api_key: Some("explicit".to_string()),
                        ..Default::default()
                    },
                    transform_headers: Some(transform),
                },
            )
            .await;
        let calls = calls.lock();
        assert_eq!(calls[1].1.api_key.as_deref(), Some("explicit"));
        assert_eq!(
            calls[1].1.headers,
            Some([("x-extra", Some("2".to_string()))].into_iter().collect())
        );
    }

    #[tokio::test]
    async fn embeddings_return_error_results_instead_of_rejecting() {
        let models = create_models(CreateModelsOptions {
            auth_context: Some(fake_auth_context(&[])),
            ..Default::default()
        });
        let ghost = models
            .embed(
                &embedding_model("ghost", "m"),
                &embeddings_context(),
                EmbeddingsOptions::default(),
            )
            .await;
        assert_eq!(ghost.stop_reason, EmbeddingsStopReason::Error);
        assert_eq!(ghost.provider, "ghost");
        assert_eq!(ghost.api, "test-embeddings");
        assert!(ghost.embeddings.is_empty());
        assert_eq!(
            ghost.error_message.as_deref(),
            Some("Unknown provider: ghost")
        );

        let calls = EmbeddingCalls::default();
        models.set_provider(embedding_test_provider(
            "p1",
            Some("MISSING"),
            &calls,
            &["test-embeddings"],
        ));
        let model = embedding_of(&models, "p1", "model-a");
        let unconfigured = models
            .embed(&model, &embeddings_context(), EmbeddingsOptions::default())
            .await;
        assert_eq!(unconfigured.stop_reason, EmbeddingsStopReason::Error);
        assert_eq!(
            unconfigured.error_message.as_deref(),
            Some("Provider is not configured: p1")
        );

        let signal = CancellationToken::new();
        signal.cancel();
        let cancelled = models
            .embed(
                &model,
                &embeddings_context(),
                EmbeddingsOptions {
                    signal: Some(signal),
                    ..Default::default()
                },
            )
            .await;
        assert_eq!(cancelled.stop_reason, EmbeddingsStopReason::Aborted);
        assert!(calls.lock().is_empty());

        models.set_provider(
            create_provider(CreateProviderOptions {
                id: "chat-only".to_string(),
                auth: ambient_auth(),
                models: vec![AnyModel::Embedding(embedding_model("chat-only", "e"))],
                api: Some(chat_streams()),
                ..Default::default()
            })
            .unwrap(),
        );
        let unsupported = models
            .embed(
                &embedding_of(&models, "chat-only", "e"),
                &embeddings_context(),
                EmbeddingsOptions::default(),
            )
            .await;
        assert_eq!(
            unsupported.error_message.as_deref(),
            Some("Provider chat-only does not support embeddings")
        );

        models.set_provider(embedding_test_provider(
            "wrong-api",
            None,
            &calls,
            &["other-embeddings"],
        ));
        let missing_api = models
            .embed(
                &embedding_of(&models, "wrong-api", "model-a"),
                &embeddings_context(),
                EmbeddingsOptions::default(),
            )
            .await;
        assert_eq!(missing_api.stop_reason, EmbeddingsStopReason::Error);
        assert_eq!(
            missing_api.error_message.as_deref(),
            Some("Provider wrong-api has no embeddings implementation for \"test-embeddings\"")
        );
        assert!(calls.lock().is_empty());

        // `embeddings` alone is enough for create_provider.
        assert!(
            create_provider(CreateProviderOptions {
                id: "embeddings-only".to_string(),
                auth: ambient_auth(),
                embeddings: Some(
                    [("test-embeddings".to_string(), recording_embeddings(&calls))]
                        .into_iter()
                        .collect()
                ),
                ..Default::default()
            })
            .is_ok()
        );
    }

    #[tokio::test]
    async fn supports_dynamic_providers_listing_image_models_via_refresh() {
        let fetches = Arc::new(AtomicUsize::new(0));
        let models_store = Arc::new(InMemoryModelsStore::default());
        let models = create_models(CreateModelsOptions {
            models_store: Some(models_store.clone()),
            ..Default::default()
        });
        let counter = fetches.clone();
        let calls = ImageCalls::default();
        models.set_provider(
            create_provider(CreateProviderOptions {
                id: "dyn".to_string(),
                auth: ambient_auth(),
                fetch_models: Some(Arc::new(move |_| {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async {
                        Ok(vec![
                            AnyModel::Image(image_model("dyn", "listed")),
                            AnyModel::Chat(test_model("dyn", "chat")),
                        ])
                    })
                })),
                images: Some(
                    [("test-images".to_string(), recording_images(&calls))]
                        .into_iter()
                        .collect(),
                ),
                ..Default::default()
            })
            .unwrap(),
        );
        assert!(models.get_all_models(Some("dyn")).is_empty());
        let result = models
            .refresh(ModelsRefreshOptions {
                providers: Some(vec!["dyn".to_string()]),
                ..Default::default()
            })
            .await;
        assert!(result.errors.is_empty());
        assert_eq!(fetches.load(Ordering::SeqCst), 1);
        assert!(
            models
                .get_model_of_type(ModelType::Image, "dyn", "listed")
                .is_some()
        );
        assert!(models.get_model("dyn", "chat").is_some());
        let stored = models_store
            .read("dyn", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            stored.models.iter().map(AnyModel::id).collect::<Vec<_>>(),
            ["listed", "chat"]
        );
    }

    #[test]
    fn keeps_existing_built_in_and_compat_model_reads_chat_only() {
        use crate::providers::all::{
            get_all_builtin_models, get_builtin_classifier_models, get_builtin_image_model,
            get_builtin_image_models, get_builtin_models,
        };
        let chat = get_builtin_models("openrouter");
        let images = get_builtin_image_models("openrouter");
        let classifiers = get_builtin_classifier_models("openrouter");
        let all = get_all_builtin_models("openrouter");
        assert!(
            all.iter()
                .any(|model| is_model_type(model, ModelType::Image))
        );
        assert_eq!(chat.len() + images.len() + classifiers.len(), all.len());
        assert_eq!(
            get_builtin_image_model("openrouter", "black-forest-labs/flux.2-pro")
                .unwrap()
                .model_type,
            crate::types::ImageModelType::Image
        );
    }

    #[tokio::test]
    async fn builtin_models_exposes_openrouter_image_models_under_the_openrouter_provider() {
        let models = crate::providers::all::builtin_models(CreateModelsOptions {
            auth_context: Some(fake_auth_context(&[("OPENROUTER_API_KEY", "or-key")])),
            ..Default::default()
        });
        let provider = models.get_provider("openrouter").unwrap();
        let images = models.get_models_of_type(ModelType::Image, Some("openrouter"));
        assert!(!images.is_empty());
        assert!(provider.get_models().unwrap().is_empty());
        assert!(
            provider
                .get_all_models()
                .unwrap()
                .iter()
                .any(|model| is_model_type(model, ModelType::Image))
        );
        assert!(
            images
                .iter()
                .all(|model| model.api() == "openrouter-images")
        );
        assert!(
            models
                .get_models_of_type(ModelType::Image, None)
                .iter()
                .all(|model| model.provider() == "openrouter")
        );
        let image = images[0].as_image().unwrap();
        assert_eq!(
            models
                .get_auth(image, Default::default())
                .await
                .unwrap()
                .unwrap()
                .auth
                .api_key
                .as_deref(),
            Some("or-key")
        );
        assert!(provider.supports_generate_images());
    }

    // max-thinking.test.ts and supports-xhigh.test.ts (catalog parts)

    fn reasoning_model(thinking_level_map: Option<serde_json::Value>) -> Model {
        Model {
            reasoning: true,
            thinking_level_map: thinking_level_map.map(|map| serde_json::from_value(map).unwrap()),
            ..test_model("test", "m")
        }
    }

    fn levels(levels: &[&str]) -> Vec<ModelThinkingLevel> {
        levels
            .iter()
            .map(|level| ModelThinkingLevel::parse(level).unwrap())
            .collect()
    }

    #[test]
    fn max_thinking_is_opt_in_for_ordinary_reasoning_models() {
        let model = reasoning_model(None);
        assert_eq!(
            get_supported_thinking_levels(&model),
            levels(&["off", "minimal", "low", "medium", "high"])
        );
        assert_eq!(
            clamp_thinking_level(&model, ModelThinkingLevel::Max),
            ModelThinkingLevel::High
        );
        assert_eq!(
            get_supported_thinking_levels(&test_model("p", "m")),
            levels(&["off"])
        );
    }

    #[test]
    fn supports_a_hole_between_high_and_max() {
        let model = reasoning_model(Some(json!({ "xhigh": null, "max": "max" })));
        assert_eq!(
            get_supported_thinking_levels(&model),
            levels(&["off", "minimal", "low", "medium", "high", "max"])
        );
        assert_eq!(
            clamp_thinking_level(&model, ModelThinkingLevel::Xhigh),
            ModelThinkingLevel::Max
        );
    }

    #[test]
    fn clamps_down_when_no_higher_level_exists() {
        let model = reasoning_model(Some(json!({ "high": null })));
        assert_eq!(
            clamp_thinking_level(&model, ModelThinkingLevel::Max),
            ModelThinkingLevel::Medium
        );
        let off_only = reasoning_model(Some(json!({ "off": null, "minimal": null, "low": null })));
        assert_eq!(
            clamp_thinking_level(&off_only, ModelThinkingLevel::Off),
            ModelThinkingLevel::Medium
        );
    }

    fn builtin(provider: &str, id: &str) -> Model {
        crate::providers::all::get_builtin_model(provider, id)
            .unwrap_or_else(|| panic!("missing builtin model {provider}/{id}"))
    }

    #[test]
    fn supported_thinking_levels_of_anthropic_catalog_models() {
        let supports = |id: &str, level: &str| {
            get_supported_thinking_levels(&builtin("anthropic", id))
                .contains(&ModelThinkingLevel::parse(level).unwrap())
        };
        assert!(supports("claude-opus-4-6", "max") && !supports("claude-opus-4-6", "xhigh"));
        assert!(supports("claude-opus-4-8", "xhigh") && supports("claude-opus-4-8", "max"));
        assert!(supports("claude-opus-5", "xhigh") && supports("claude-opus-5", "max"));
        assert!(supports("claude-sonnet-4-6", "max") && !supports("claude-sonnet-4-6", "xhigh"));
        assert!(supports("claude-sonnet-5", "xhigh") && supports("claude-sonnet-5", "max"));
        assert!(
            supports("claude-fable-5", "xhigh")
                && supports("claude-fable-5", "max")
                && !supports("claude-fable-5", "off")
        );
        assert!(!supports("claude-sonnet-4-5", "xhigh") && !supports("claude-sonnet-4-5", "max"));
    }

    #[test]
    fn claude_5_5_models_carry_their_effort_levels_pricing_and_compat() {
        for (id, input, output, cache_read, cache_write) in [
            ("claude-opus-5-5", 4.0, 20.0, 0.2, 5.0),
            ("claude-sonnet-5-5", 2.0, 10.0, 0.2, 2.5),
        ] {
            let model = builtin("anthropic", id);
            assert_eq!(
                (
                    model.cost.input,
                    model.cost.output,
                    model.cost.cache_read,
                    model.cost.cache_write
                ),
                (input, output, cache_read, cache_write)
            );
            assert_eq!(
                (model.context_window, model.max_tokens),
                (1_000_000, 128_000)
            );
            let compat = model.compat();
            assert_eq!(compat.force_adaptive_thinking, Some(true));
            assert_eq!(compat.supports_mid_convo_effort, Some(true));
            assert_eq!(compat.supports_mid_convo_system_messages, Some(true));
            assert_eq!(compat.supports_mid_convo_tool_changes, Some(true));
            assert_eq!(
                get_supported_thinking_levels(&model),
                levels(&["low", "medium", "high", "xhigh", "max"])
            );
        }
        assert_eq!(
            builtin("anthropic", "claude-sonnet-5-5")
                .compat()
                .supports_temperature,
            Some(false)
        );
    }

    #[test]
    fn supported_thinking_levels_of_openai_catalog_models() {
        for id in [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-6-sol",
            "gpt-6-luna",
        ] {
            assert_eq!(
                get_supported_thinking_levels(&builtin("openai", id)),
                levels(&["off", "low", "medium", "high", "xhigh", "max"]),
                "{id}"
            );
        }
        // OpenAI rejects reasoning.effort "none" for GPT-6.1 Sol.
        let sol = builtin("openai", "gpt-6.1-sol");
        assert_eq!(
            get_supported_thinking_levels(&sol),
            levels(&["low", "medium", "high", "xhigh", "max"])
        );
        assert_eq!(
            sol.thinking_level_map
                .as_ref()
                .unwrap()
                .get(&ModelThinkingLevel::Off),
            Some(&None)
        );
        assert_eq!(
            get_supported_thinking_levels(&builtin("openai", "gpt-5.5-pro")),
            levels(&["medium", "high", "xhigh"])
        );
    }

    #[test]
    fn openai_gpt_6_models_carry_official_metadata() {
        for (id, input, output, cache_read, cache_write) in [
            ("gpt-6-sol", 2.0, 10.0, 0.2, 2.5),
            ("gpt-6-luna", 0.1, 0.5, 0.01, 0.125),
            ("gpt-6.1-sol", 2.0, 10.0, 0.1, 2.5),
        ] {
            let model = builtin("openai", id);
            assert_eq!(model.input, vec![ModelInput::Text, ModelInput::Image]);
            assert_eq!(
                model.cost,
                ModelCost {
                    input,
                    output,
                    cache_read,
                    cache_write,
                    tiers: Some(vec![ModelCostTier {
                        input_tokens_above: 272_000,
                        input: input * 2.0,
                        output: output * 1.5,
                        cache_read: cache_read * 2.0,
                        cache_write: cache_write * 2.0,
                    }]),
                }
            );
            assert_eq!((model.context_window, model.max_tokens), (272_000, 128_000));
            let compat = model.compat();
            assert_eq!(compat.supports_additional_tools, Some(true));
            assert_eq!(compat.supports_mid_convo_system_messages, Some(true));
            assert_eq!(compat.supports_openai_grammar_tools, Some(true));
            assert_eq!(compat.supports_tool_search, Some(true));
        }
    }

    #[test]
    fn user_message_helper_is_used_by_context() {
        assert!(matches!(
            context().messages[0],
            Message::User(UserMessage { .. })
        ));
    }
}
