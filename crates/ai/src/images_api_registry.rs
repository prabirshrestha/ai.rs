//! Port of `images-api-registry.ts` and `providers/images/register-builtins.ts`:
//! the global registry of image-generation API implementations used by
//! [`generate_images`](crate::images::generate_images).
//!
//! Divergences:
//! - The builtin implementations register on first registry access (Pi
//!   registers them as an import side effect). There is no lazy module load,
//!   so `createLazyLoadErrorImages` has no counterpart.
//! - ai.rs extra: `openai-images` (OpenAI-compatible `/images/generations`)
//!   is registered next to Pi's `openrouter-images`.

use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::RwLock;

use crate::api::openai_images::openai_images_api;
use crate::api::openrouter_images::openrouter_images_api;
use crate::types::{
    AssistantImages, ImageApi, ImageModel, ImagesContext, ImagesOptions, KnownImageApi,
    ProviderImages,
};
use crate::{Error, Result};

/// `ImagesApiProvider`: an implementation registered under its image api id.
#[derive(Clone)]
pub struct ImagesApiProvider {
    pub api: ImageApi,
    pub generate_images: Arc<dyn ProviderImages>,
}

/// A registered implementation (`ImagesApiProviderInternal`). Its
/// `generate_images` checks that the model's api matches.
#[derive(Clone)]
pub struct RegisteredImagesApiProvider {
    pub api: ImageApi,
    implementation: Arc<dyn ProviderImages>,
}

impl RegisteredImagesApiProvider {
    pub async fn generate_images(
        &self,
        model: ImageModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> Result<AssistantImages> {
        if model.api != self.api {
            return Err(Error::message(format!(
                "Mismatched api: {} expected {}",
                model.api, self.api
            )));
        }
        Ok(self
            .implementation
            .generate_images(model, context, options)
            .await)
    }
}

struct RegistryEntry {
    provider: RegisteredImagesApiProvider,
    #[allow(dead_code)] // kept like Pi's `sourceId`; nothing unregisters by it yet
    source_id: Option<String>,
}

static REGISTRY: LazyLock<RwLock<IndexMap<String, RegistryEntry>>> = LazyLock::new(|| {
    let registry = RwLock::new(IndexMap::new());
    register_builtins_into(&mut registry.write());
    registry
});

fn insert(
    registry: &mut IndexMap<String, RegistryEntry>,
    provider: ImagesApiProvider,
    source_id: Option<String>,
) {
    registry.insert(
        provider.api.clone(),
        RegistryEntry {
            provider: RegisteredImagesApiProvider {
                api: provider.api,
                implementation: provider.generate_images,
            },
            source_id,
        },
    );
}

fn register_builtins_into(registry: &mut IndexMap<String, RegistryEntry>) {
    insert(
        registry,
        ImagesApiProvider {
            api: KnownImageApi::OpenrouterImages.as_str().to_string(),
            generate_images: openrouter_images_api(),
        },
        None,
    );
    insert(
        registry,
        ImagesApiProvider {
            api: KnownImageApi::OpenaiImages.as_str().to_string(),
            generate_images: openai_images_api(),
        },
        None,
    );
}

/// `registerImagesApiProvider(provider, sourceId?)`.
pub fn register_images_api_provider(provider: ImagesApiProvider, source_id: Option<&str>) {
    insert(
        &mut REGISTRY.write(),
        provider,
        source_id.map(str::to_string),
    );
}

/// `getImagesApiProvider(api)`.
pub fn get_images_api_provider(api: &str) -> Option<RegisteredImagesApiProvider> {
    REGISTRY.read().get(api).map(|entry| entry.provider.clone())
}

/// `registerBuiltInImagesApiProviders()`: (re-)register the builtin
/// implementations, replacing any registered under the same api.
pub fn register_builtin_images_api_providers() {
    register_builtins_into(&mut REGISTRY.write());
}

/// A [`ProviderImages`] from an async closure, for registering ad-hoc
/// implementations.
pub fn images_fn<F, Fut>(generate: F) -> Arc<dyn ProviderImages>
where
    F: Fn(ImageModel, ImagesContext, ImagesOptions) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = AssistantImages> + Send + 'static,
{
    struct FnImages<F>(F);

    #[async_trait]
    impl<F, Fut> ProviderImages for FnImages<F>
    where
        F: Fn(ImageModel, ImagesContext, ImagesOptions) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = AssistantImages> + Send + 'static,
    {
        async fn generate_images(
            &self,
            model: ImageModel,
            context: ImagesContext,
            options: ImagesOptions,
        ) -> AssistantImages {
            (self.0)(model, context, options).await
        }
    }

    Arc::new(FnImages(generate))
}
