//! Shared plumbing for the provider handles kept from the pre-1.0 API
//! (`providers::openai::builder()...build()`). A handle owns a private
//! `Models` collection holding one provider built with `create_provider()`
//! from the Pi provider's catalog and APIs, with the handle's explicit
//! credentials layered over the Pi provider's ambient auth. Requests go
//! through that collection (`handle.models().complete_simple(..)`).

use std::sync::Arc;

use async_trait::async_trait;

use crate::Result;
use crate::auth::{ApiKeyAuth, ApiKeyAuthInput, AuthResult, ModelAuth};
use crate::types::{
    AssistantImages, DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, EmbeddingModel,
    EmbeddingsContext, EmbeddingsOptions, EmbeddingsResult, ImageModel, ImagesContext,
    ImagesOptions, Model, ProviderEmbeddings, ProviderHeaders, ProviderImages, ProviderStreams,
    SimpleStreamOptions, StreamOptions, TranscriptContext,
};
use crate::utils::event_stream::AssistantMessageEventStream;

/// Handle credentials: an explicit request key wins, then the handle's own
/// key or headers, then the Pi provider's ambient auth (env vars), then
/// keyless access when the handle points at a custom base URL.
pub(crate) struct HandleAuth {
    pub name: String,
    pub api_key: Option<String>,
    pub headers: Option<ProviderHeaders>,
    pub fallback: Option<Arc<dyn ApiKeyAuth>>,
    pub keyless: bool,
}

#[async_trait]
impl ApiKeyAuth for HandleAuth {
    fn name(&self) -> &str {
        &self.name
    }

    async fn resolve(&self, input: ApiKeyAuthInput) -> Result<Option<AuthResult>> {
        if let Some(credential) = &input.credential
            && let Some(key) = credential.key.as_ref().filter(|key| !key.is_empty())
        {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key.clone()),
                    ..Default::default()
                },
                env: credential.env.clone(),
                source: Some("request".to_string()),
            }));
        }
        if self.api_key.is_some() || self.headers.is_some() {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: self.api_key.clone(),
                    headers: self.headers.clone(),
                    base_url: None,
                },
                env: None,
                source: Some("provider handle".to_string()),
            }));
        }
        if let Some(fallback) = &self.fallback
            && let Some(result) = fallback.resolve(input).await?
        {
            return Ok(Some(result));
        }
        Ok(self.keyless.then(|| AuthResult {
            source: Some("keyless".to_string()),
            ..Default::default()
        }))
    }
}

/// Adds the handle's HTTP client to requests that do not bring their own.
pub(crate) struct HandleStreams {
    pub inner: Arc<dyn ProviderStreams>,
    pub http_client: Option<reqwest::Client>,
}

impl HandleStreams {
    pub fn wrap(
        inner: Arc<dyn ProviderStreams>,
        http_client: &Option<reqwest::Client>,
    ) -> Arc<dyn ProviderStreams> {
        Arc::new(Self {
            inner,
            http_client: http_client.clone(),
        })
    }

    fn client(&self, client: &mut Option<reqwest::Client>) {
        if client.is_none() {
            client.clone_from(&self.http_client);
        }
    }
}

#[async_trait]
impl ProviderStreams for HandleStreams {
    fn stream(
        &self,
        model: Model,
        context: TranscriptContext,
        mut options: StreamOptions,
    ) -> AssistantMessageEventStream {
        self.client(&mut options.http_client);
        self.inner.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: Model,
        context: TranscriptContext,
        mut options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.client(&mut options.http_client);
        self.inner.stream_simple(model, context, options)
    }

    fn supports_fetch_deferred(&self) -> bool {
        self.inner.supports_fetch_deferred()
    }

    fn fetch_deferred(
        &self,
        model: Model,
        handle: DeferredHandle,
        mut options: DeferredFetchOptions,
    ) -> AssistantMessageEventStream {
        self.client(&mut options.request.http_client);
        self.inner.fetch_deferred(model, handle, options)
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.inner.supports_cancel_deferred()
    }

    async fn cancel_deferred(
        &self,
        model: Model,
        handle: DeferredHandle,
        mut options: DeferredCancelOptions,
    ) -> Result<()> {
        self.client(&mut options.http_client);
        self.inner.cancel_deferred(model, handle, options).await
    }
}

/// Adds the handle's HTTP client to image requests that do not bring their
/// own.
pub(crate) struct HandleImages {
    pub inner: Arc<dyn ProviderImages>,
    pub http_client: Option<reqwest::Client>,
}

impl HandleImages {
    pub fn wrap(
        inner: Arc<dyn ProviderImages>,
        http_client: &Option<reqwest::Client>,
    ) -> Arc<dyn ProviderImages> {
        Arc::new(Self {
            inner,
            http_client: http_client.clone(),
        })
    }
}

#[async_trait]
impl ProviderImages for HandleImages {
    async fn generate_images(
        &self,
        model: ImageModel,
        context: ImagesContext,
        mut options: ImagesOptions,
    ) -> AssistantImages {
        if options.http_client.is_none() {
            options.http_client.clone_from(&self.http_client);
        }
        self.inner.generate_images(model, context, options).await
    }
}

/// Adds the handle's HTTP client to embeddings requests that do not bring
/// their own.
pub(crate) struct HandleEmbeddings {
    pub inner: Arc<dyn ProviderEmbeddings>,
    pub http_client: Option<reqwest::Client>,
}

impl HandleEmbeddings {
    pub fn wrap(
        inner: Arc<dyn ProviderEmbeddings>,
        http_client: &Option<reqwest::Client>,
    ) -> Arc<dyn ProviderEmbeddings> {
        Arc::new(Self {
            inner,
            http_client: http_client.clone(),
        })
    }
}

#[async_trait]
impl ProviderEmbeddings for HandleEmbeddings {
    async fn embed(
        &self,
        model: EmbeddingModel,
        context: EmbeddingsContext,
        mut options: EmbeddingsOptions,
    ) -> EmbeddingsResult {
        if options.http_client.is_none() {
            options.http_client.clone_from(&self.http_client);
        }
        self.inner.embed(model, context, options).await
    }
}

/// Trim and drop empty keys, like the pre-1.0 builders.
pub(crate) fn clean_key(key: Option<String>) -> Option<String> {
    key.map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
}
