//! Port of `providers/openai.ts`, plus the pre-1.0 [`OpenAi`] handle.
//!
//! The ChatGPT subscription OAuth (`lazyOAuth(loadOpenAIChatGPTOAuth)`) is not
//! ported; the provider offers api-key auth only.
//!
//! ai.rs extras, not in Pi: the embedding models (`text-embedding-3-small`,
//! `text-embedding-3-large`, `text-embedding-ada-002`, api
//! `openai-embeddings`) served through [`openai_embeddings_api`], and on the
//! handle [`OpenAi::image_model`] (OpenAI-compatible
//! `/images/generations`, api `openai-images`). Handles also serve the
//! embedding models, keyless ones included.

use std::sync::Arc;

use indexmap::IndexMap;

use super::catalog::{openai_embedding_models, openai_models};
use super::handle::{HandleAuth, HandleEmbeddings, HandleImages, HandleStreams, clean_key};
use super::model_builder::{ImageModelBuilder, ModelBuilder};
use crate::api::openai_completions::openai_completions_api;
use crate::api::openai_embeddings::openai_embeddings_api;
use crate::api::openai_images::openai_images_api;
use crate::api::openai_responses::openai_responses_api;
use crate::auth::{ProviderAuth, env_api_key_auth};
use crate::env_api_keys::get_env_api_key;
use crate::models::{
    CreateModelsOptions, CreateProviderOptions, Models, Provider, ProviderApi, create_models,
    create_provider,
};
use crate::types::{
    AnyModel, AssistantImages, EmbeddingModel, EmbeddingsContext, EmbeddingsOptions,
    EmbeddingsResult, ImageModel, ImageModelType, ImagesContext, ImagesOptions, KnownApi,
    KnownEmbeddingApi, KnownImageApi, Model, ModelInput, ModelOutput, ProviderEmbeddings,
    ProviderHeaders, ProviderImages, ProviderStreams, SimpleStreamOptions, StreamOptions,
    TranscriptContext,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::headers::has_non_empty_header;
use crate::utils::models_error::ModelsErrorCode;
use crate::{Result, auth::models_error};

const DEFAULT_PROVIDER_ID: &str = "openai";
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// `openaiProvider()`.
pub fn openai_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: DEFAULT_PROVIDER_ID.to_string(),
        name: Some("OpenAI".to_string()),
        base_url: Some(DEFAULT_BASE_URL.to_string()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("OpenAI API key", &["OPENAI_API_KEY"])),
            oauth: None,
        },
        models: openai_models()
            .values()
            .cloned()
            .map(AnyModel::Chat)
            .chain(
                openai_embedding_models()
                    .values()
                    .cloned()
                    .map(AnyModel::Embedding),
            )
            .collect(),
        api: Some(ProviderApi::Single(openai_responses_api())),
        embeddings: Some(
            [(
                KnownEmbeddingApi::OpenaiEmbeddings.as_str().to_string(),
                openai_embeddings_api(),
            )]
            .into_iter()
            .collect(),
        ),
        ..Default::default()
    })
    .expect("the OpenAI provider has an API implementation")
}

/// Which OpenAI API a handle's models use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpenAiApi {
    #[default]
    Responses,
    ChatCompletions,
}

impl OpenAiApi {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Responses => KnownApi::OpenaiResponses.as_str(),
            Self::ChatCompletions => KnownApi::OpenaiCompletions.as_str(),
        }
    }
}

/// Pre-1.0 provider handle: `openai::builder().api_key(..).build()?` then
/// `handle.model("gpt-5.5").build()?`. Requests go through the handle's own
/// [`Models`] collection (`handle.models().complete_simple(..)`), which uses
/// the handle's key, base URL and HTTP client.
///
/// Unlike Pi's `openaiProvider()`, a handle also serves Chat Completions
/// (`OpenAiApi::ChatCompletions`) for OpenAI-compatible servers.
#[derive(Clone, Debug)]
pub struct OpenAi {
    provider_id: String,
    base_url: String,
    api: OpenAiApi,
    models: Models,
}

impl OpenAi {
    pub fn builder() -> OpenAiBuilder {
        OpenAiBuilder::default()
    }

    /// A handle that requires `OPENAI_API_KEY`.
    pub fn from_env() -> Result<Self> {
        if get_env_api_key(DEFAULT_PROVIDER_ID, None).is_none() {
            return Err(models_error(
                ModelsErrorCode::Auth,
                format!("Provider is not configured: {DEFAULT_PROVIDER_ID}"),
            ));
        }
        Self::builder().build()
    }

    pub fn id(&self) -> &str {
        &self.provider_id
    }

    /// The handle's collection, holding its one provider.
    pub fn models(&self) -> &Models {
        &self.models
    }

    /// Start building a chat model: the catalog entry when the id is known,
    /// otherwise a default shape.
    pub fn model(&self, id: &str) -> ModelBuilder {
        let model = match self.models.get_model(&self.provider_id, id) {
            Some(mut model) => {
                model.api = self.api.id().to_string();
                model
            }
            None => Model {
                id: id.to_string(),
                name: id.to_string(),
                api: self.api.id().to_string(),
                provider: self.provider_id.clone(),
                base_url: self.base_url.clone(),
                input: vec![ModelInput::Text, ModelInput::Image],
                context_window: 1_000_000,
                max_tokens: 16_384,
                ..Default::default()
            },
        };
        ModelBuilder::new(model)
    }

    /// ai.rs extra: an OpenAI-compatible image model (`openai-images`,
    /// `/images/generations`), text input and image output by default.
    pub fn image_model(&self, id: &str) -> ImageModelBuilder {
        ImageModelBuilder::new(ImageModel {
            id: id.to_string(),
            name: id.to_string(),
            api: KnownImageApi::OpenaiImages.as_str().to_string(),
            provider: self.provider_id.clone(),
            base_url: self.base_url.clone(),
            model_type: ImageModelType::Image,
            input: vec![ModelInput::Text],
            output: vec![ModelOutput::Image],
            ..Default::default()
        })
    }
}

/// `openai::builder()`.
pub fn builder() -> OpenAiBuilder {
    OpenAi::builder()
}

/// `openai::from_env()`.
pub fn from_env() -> Result<OpenAi> {
    OpenAi::from_env()
}

/// Rust addition for keyless handles (a custom base URL and no key): Pi's
/// OpenAI API modules refuse requests without an API key or an
/// `Authorization` header. When a request has neither, this adapter passes a
/// placeholder key and suppresses the `Authorization` header, so the server
/// receives no credentials.
struct KeylessStreams {
    inner: Arc<dyn ProviderStreams>,
    images: Option<Arc<dyn ProviderImages>>,
    embeddings: Option<Arc<dyn ProviderEmbeddings>>,
}

impl KeylessStreams {
    fn wrap(inner: Arc<dyn ProviderStreams>) -> Arc<dyn ProviderStreams> {
        Arc::new(Self {
            inner,
            images: None,
            embeddings: None,
        })
    }

    fn wrap_images(images: Arc<dyn ProviderImages>) -> Arc<dyn ProviderImages> {
        Arc::new(Self {
            inner: openai_responses_api(),
            images: Some(images),
            embeddings: None,
        })
    }

    fn wrap_embeddings(embeddings: Arc<dyn ProviderEmbeddings>) -> Arc<dyn ProviderEmbeddings> {
        Arc::new(Self {
            inner: openai_responses_api(),
            images: None,
            embeddings: Some(embeddings),
        })
    }

    /// Without a key or an `Authorization` header: a placeholder key and a
    /// suppressed `Authorization` header.
    fn apply_keyless(api_key: &mut Option<String>, headers: &mut Option<ProviderHeaders>) {
        let has_key = api_key.as_deref().is_some_and(|key| !key.is_empty());
        let has_authorization = headers
            .as_ref()
            .is_some_and(|headers| has_non_empty_header(headers, "authorization"));
        if !has_key && !has_authorization {
            *api_key = Some(KEYLESS_API_KEY.to_string());
            headers
                .get_or_insert_with(ProviderHeaders::new)
                .insert("Authorization", None::<String>);
        }
    }

    fn apply(options: &mut StreamOptions) {
        let has_key = options
            .api_key
            .as_deref()
            .is_some_and(|key| !key.is_empty());
        let has_authorization = options.headers.as_ref().is_some_and(|headers| {
            has_non_empty_header(headers, "authorization")
                || has_non_empty_header(headers, "cf-aig-authorization")
        });
        if has_key || has_authorization {
            return;
        }
        options.api_key = Some(KEYLESS_API_KEY.to_string());
        options
            .headers
            .get_or_insert_with(ProviderHeaders::new)
            .insert("Authorization", None::<String>);
    }
}

const KEYLESS_API_KEY: &str = "keyless";

#[async_trait::async_trait]
impl ProviderImages for KeylessStreams {
    async fn generate_images(
        &self,
        model: ImageModel,
        context: ImagesContext,
        mut options: ImagesOptions,
    ) -> AssistantImages {
        Self::apply_keyless(&mut options.api_key, &mut options.headers);
        self.images
            .as_ref()
            .expect("keyless images adapter wraps an images implementation")
            .generate_images(model, context, options)
            .await
    }
}

#[async_trait::async_trait]
impl ProviderEmbeddings for KeylessStreams {
    async fn embed(
        &self,
        model: EmbeddingModel,
        context: EmbeddingsContext,
        mut options: EmbeddingsOptions,
    ) -> EmbeddingsResult {
        Self::apply_keyless(&mut options.api_key, &mut options.headers);
        self.embeddings
            .as_ref()
            .expect("keyless embeddings adapter wraps an embeddings implementation")
            .embed(model, context, options)
            .await
    }
}

impl ProviderStreams for KeylessStreams {
    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        mut options: StreamOptions,
    ) -> AssistantMessageEventStream {
        Self::apply(&mut options);
        self.inner.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        mut options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        Self::apply(&mut options);
        self.inner.stream_simple(model, context, options)
    }
}

#[derive(Default)]
pub struct OpenAiBuilder {
    provider_id: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    api: OpenAiApi,
    http_client: Option<reqwest::Client>,
}

impl OpenAiBuilder {
    pub fn provider_id(mut self, provider_id: impl Into<String>) -> Self {
        self.provider_id = Some(provider_id.into());
        self
    }

    pub fn api_key(mut self, api_key: Option<&str>) -> Self {
        self.api_key = clean_key(api_key.map(str::to_string));
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn api(mut self, api: OpenAiApi) -> Self {
        self.api = api;
        self
    }

    pub fn responses(mut self) -> Self {
        self.api = OpenAiApi::Responses;
        self
    }

    pub fn chat_completions(mut self) -> Self {
        self.api = OpenAiApi::ChatCompletions;
        self
    }

    pub fn http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn build(self) -> Result<OpenAi> {
        let provider_id = self
            .provider_id
            .unwrap_or_else(|| DEFAULT_PROVIDER_ID.to_string());
        let base_url = self
            .base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let keyless = self.api_key.is_none() && base_url != DEFAULT_BASE_URL;
        let auth = HandleAuth {
            name: "OpenAI API key".to_string(),
            api_key: self.api_key,
            headers: None,
            fallback: Some(env_api_key_auth("OpenAI API key", &["OPENAI_API_KEY"])),
            keyless,
        };
        let models = openai_models()
            .values()
            .map(|model| {
                AnyModel::Chat(Model {
                    provider: provider_id.clone(),
                    base_url: base_url.clone(),
                    ..model.clone()
                })
            })
            .chain(openai_embedding_models().values().map(|model| {
                AnyModel::Embedding(EmbeddingModel {
                    provider: provider_id.clone(),
                    base_url: base_url.clone(),
                    ..model.clone()
                })
            }))
            .collect();
        let wrap = |api: Arc<dyn ProviderStreams>| {
            let api = if keyless {
                KeylessStreams::wrap(api)
            } else {
                api
            };
            HandleStreams::wrap(api, &self.http_client)
        };
        let streams: IndexMap<String, _> = [
            (
                OpenAiApi::Responses.id().to_string(),
                wrap(openai_responses_api()),
            ),
            (
                OpenAiApi::ChatCompletions.id().to_string(),
                wrap(openai_completions_api()),
            ),
        ]
        .into_iter()
        .collect();
        let images_api = if keyless {
            KeylessStreams::wrap_images(openai_images_api())
        } else {
            openai_images_api()
        };
        let images = [(
            KnownImageApi::OpenaiImages.as_str().to_string(),
            HandleImages::wrap(images_api, &self.http_client),
        )]
        .into_iter()
        .collect();
        let embeddings_api = if keyless {
            KeylessStreams::wrap_embeddings(openai_embeddings_api())
        } else {
            openai_embeddings_api()
        };
        let embeddings = [(
            KnownEmbeddingApi::OpenaiEmbeddings.as_str().to_string(),
            HandleEmbeddings::wrap(embeddings_api, &self.http_client),
        )]
        .into_iter()
        .collect();
        let provider = create_provider(CreateProviderOptions {
            id: provider_id.clone(),
            name: Some("OpenAI".to_string()),
            base_url: Some(base_url.clone()),
            auth: ProviderAuth {
                api_key: Some(Arc::new(auth)),
                oauth: None,
            },
            models,
            api: Some(ProviderApi::ByApi(streams)),
            images: Some(images),
            embeddings: Some(embeddings),
            ..Default::default()
        })?;
        let collection = create_models(CreateModelsOptions::default());
        collection.set_provider(provider);
        Ok(OpenAi {
            provider_id,
            base_url,
            api: self.api,
            models: collection,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::openai_client::test_support::{CapturedRequest, MockResponse, MockServer};

    #[test]
    fn openai_provider_lists_the_generated_catalog() {
        let provider = openai_provider();
        assert_eq!(provider.id(), "openai");
        assert_eq!(provider.name(), "OpenAI");
        assert_eq!(provider.base_url(), Some(DEFAULT_BASE_URL));
        let models = provider.get_models().unwrap();
        assert_eq!(models.len(), openai_models().len());
        assert!(models.iter().all(|model| model.api == "openai-responses"));
    }

    #[test]
    fn handle_builds_catalog_and_custom_models() {
        let handle = builder()
            .api_key(Some(" test-key "))
            .chat_completions()
            .build()
            .unwrap();
        let model = handle.model("gpt-5.5").build().unwrap();
        assert_eq!(model.id, "gpt-5.5");
        assert_eq!(model.provider, "openai");
        assert_eq!(model.api, "openai-completions");
        assert_eq!(model.context_window, 272_000);

        let custom = builder()
            .provider_id("local")
            .base_url("http://localhost:11434/v1")
            .build()
            .unwrap()
            .model("llama")
            .build()
            .unwrap();
        assert_eq!(custom.provider, "local");
        assert_eq!(custom.api, "openai-responses");
        assert_eq!(custom.base_url, "http://localhost:11434/v1");
    }

    #[tokio::test]
    async fn handle_keys_win_and_custom_base_urls_allow_keyless_access() {
        let handle = builder().api_key(Some("test-key")).build().unwrap();
        let auth = handle
            .models()
            .get_auth("openai", Default::default())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(auth.auth.api_key.as_deref(), Some("test-key"));

        let local = builder()
            .base_url("http://localhost:11434/v1")
            .build()
            .unwrap();
        let auth = local
            .models()
            .get_auth("openai", Default::default())
            .await
            .unwrap()
            .unwrap();
        if std::env::var("OPENAI_API_KEY").is_err() {
            assert_eq!(auth.auth.api_key, None);
            assert_eq!(auth.source.as_deref(), Some("keyless"));
        }
    }

    async fn request_through_handle(
        configure: impl FnOnce(OpenAiBuilder) -> OpenAiBuilder,
        response: MockResponse,
    ) -> (CapturedRequest, crate::types::AssistantMessage) {
        let server = MockServer::start(vec![response]).await;
        let handle = configure(builder().base_url(server.url.clone()))
            .build()
            .unwrap();
        let model = handle.model("local-model").build().unwrap();
        let context: crate::types::Context = serde_json::from_value(serde_json::json!({
            "messages": [{ "role": "user", "content": "hi", "timestamp": 1 }],
        }))
        .unwrap();
        let result = handle
            .models()
            .complete(&model, &context, StreamOptions::default())
            .await;
        (server.last(), result)
    }

    fn chat_completion_stop() -> MockResponse {
        MockResponse::sse(&[serde_json::json!({
            "id": "chatcmpl-1",
            "choices": [{ "index": 0, "delta": { "content": "ok" }, "finish_reason": "stop" }],
        })])
    }

    #[tokio::test]
    async fn chat_completions_handle_posts_to_chat_completions() {
        let (request, result) = request_through_handle(
            |builder| builder.api_key(Some("sk-handle")).chat_completions(),
            chat_completion_stop(),
        )
        .await;
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.header("authorization"), Some("Bearer sk-handle"));
        assert_eq!(request.body["model"], "local-model");
        assert_eq!(result.error_message, None);
        assert_eq!(result.stop_reason, crate::types::StopReason::Stop);
    }

    #[tokio::test]
    async fn keyless_handles_send_no_authorization_header() {
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return;
        }
        let (request, result) =
            request_through_handle(|builder| builder.chat_completions(), chat_completion_stop())
                .await;
        assert_eq!(request.header("authorization"), None);
        assert_eq!(result.error_message, None);

        let (request, _) = request_through_handle(
            |builder| builder,
            MockResponse::sse(&[serde_json::json!({
                "type": "response.completed",
                "response": { "id": "resp_1", "status": "completed", "output": [] },
            })]),
        )
        .await;
        assert_eq!(request.path, "/v1/responses");
        assert_eq!(request.header("authorization"), None);
    }

    fn images_response(body: serde_json::Value) -> MockResponse {
        MockResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: body.to_string(),
        }
    }

    #[tokio::test]
    async fn image_models_generate_through_the_handle() {
        let server = MockServer::start(vec![images_response(serde_json::json!({
            "data": [{ "b64_json": "ZmFrZS1wbmc=" }],
        }))])
        .await;
        let handle = builder()
            .api_key(Some("test-key"))
            .base_url(server.url.clone())
            .build()
            .unwrap();
        let model = handle.image_model("gpt-image-2").build_image().unwrap();
        assert_eq!(model.api, "openai-images");
        assert_eq!(model.input, vec![ModelInput::Text]);
        assert_eq!(model.output, vec![ModelOutput::Image]);
        let output = handle
            .models()
            .generate_images(
                &model,
                &ImagesContext::builder().text("A tiny robot").build(),
                ImagesOptions::default(),
            )
            .await;
        assert_eq!(output.stop_reason, crate::types::ImagesStopReason::Stop);
        let request = server.last();
        assert_eq!(request.path, "/v1/images/generations");
        assert_eq!(request.header("authorization"), Some("Bearer test-key"));
        assert_eq!(request.body["prompt"], "A tiny robot");
    }

    #[test]
    fn openai_provider_lists_embedding_models() {
        let provider = openai_provider();
        let embeddings: Vec<_> = provider
            .get_all_models()
            .unwrap()
            .into_iter()
            .filter_map(|model| model.as_embedding().cloned())
            .collect();
        assert_eq!(
            embeddings
                .iter()
                .map(|model| (model.id.as_str(), model.dimensions))
                .collect::<Vec<_>>(),
            [
                ("text-embedding-3-small", 1536),
                ("text-embedding-3-large", 3072),
                ("text-embedding-ada-002", 1536),
            ]
        );
        assert!(provider.supports_embed());
        assert_eq!(provider.get_models().unwrap().len(), openai_models().len());
    }

    #[tokio::test]
    async fn keyless_handles_embed_without_authorization() {
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return;
        }
        let server = MockServer::start(vec![images_response(serde_json::json!({
            "data": [{ "embedding": [0.25], "index": 0 }],
        }))])
        .await;
        let ollama = builder()
            .provider_id("ollama")
            .base_url(server.url.clone())
            .build()
            .unwrap();
        // Custom ids work like catalog ones: the provider dispatches on `api`.
        let model = EmbeddingModel {
            id: "nomic-embed-text".to_string(),
            name: "nomic-embed-text".to_string(),
            api: KnownEmbeddingApi::OpenaiEmbeddings.as_str().to_string(),
            provider: "ollama".to_string(),
            base_url: server.url.clone(),
            input: vec![ModelInput::Text],
            ..Default::default()
        };
        let output = ollama
            .models()
            .embed(
                &model,
                &EmbeddingsContext {
                    input: vec!["hello".to_string()],
                },
                EmbeddingsOptions::default(),
            )
            .await;
        assert_eq!(output.error_message, None);
        assert_eq!(
            output.embeddings,
            vec![crate::types::EmbeddingVector::Float(vec![0.25])]
        );
        let request = server.last();
        assert_eq!(request.path, "/v1/embeddings");
        assert_eq!(request.header("authorization"), None);
        assert_eq!(request.body["model"], "nomic-embed-text");
    }

    #[tokio::test]
    async fn handles_list_and_serve_catalog_embedding_models() {
        let server = MockServer::start(vec![images_response(serde_json::json!({
            "data": [{ "embedding": [0.5], "index": 0 }],
        }))])
        .await;
        let handle = builder()
            .api_key(Some("test-key"))
            .base_url(server.url.clone())
            .build()
            .unwrap();
        let model = handle
            .models()
            .get_model_of_type(
                crate::types::ModelType::Embedding,
                "openai",
                "text-embedding-3-small",
            )
            .and_then(|model| model.as_embedding().cloned())
            .unwrap();
        assert_eq!(model.base_url, server.url);
        let output = handle
            .models()
            .embed(
                &model,
                &EmbeddingsContext {
                    input: vec!["hello".to_string()],
                },
                EmbeddingsOptions::default(),
            )
            .await;
        assert_eq!(output.error_message, None);
        assert_eq!(
            server.last().header("authorization"),
            Some("Bearer test-key")
        );
    }

    #[tokio::test]
    async fn ollama_compatible_image_endpoints_work_without_a_key() {
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return;
        }
        let server = MockServer::start(vec![images_response(serde_json::json!({
            "created": 1710000000,
            "data": [{ "b64_json": "b2xsYW1h" }],
        }))])
        .await;
        let ollama = builder()
            .provider_id("ollama")
            .base_url(server.url.clone())
            .build()
            .unwrap();
        let model = ollama.image_model("x/z-image-turbo").build_image().unwrap();
        let output = ollama
            .models()
            .generate_images(
                &model,
                &ImagesContext::builder().text("Generate a cat").build(),
                ImagesOptions::default(),
            )
            .await;
        assert_eq!(output.stop_reason, crate::types::ImagesStopReason::Stop);
        assert_eq!(
            output.output,
            vec![crate::types::UserContent::Image(
                crate::types::ImageContent {
                    data: "b2xsYW1h".to_string(),
                    mime_type: "image/png".to_string(),
                }
            )]
        );
        let request = server.last();
        assert_eq!(request.path, "/v1/images/generations");
        assert_eq!(request.header("authorization"), None);
        assert_eq!(request.body["model"], "x/z-image-turbo");
        assert_eq!(request.body["response_format"], "b64_json");
    }
}
