//! Port of `providers/openrouter.ts` restricted to image generation, plus the
//! pre-1.0 [`OpenRouter`] image handle.
//!
//! Divergences from Pi's `openrouterProvider()`: only the image models
//! (`OPENROUTER_IMAGE_MODELS`) and the `openrouter-images` implementation are
//! included. OpenRouter chat models (Anthropic Messages / Chat Completions),
//! the TypeSafe System One classifiers, and the OpenRouter OAuth login are out
//! of scope; auth is `OPENROUTER_API_KEY`.

use std::sync::Arc;

use indexmap::IndexMap;

use super::catalog::openrouter_image_models;
use super::handle::{HandleAuth, HandleImages, clean_key};
use super::model_builder::ImageModelBuilder;
use crate::Result;
use crate::api::openrouter_images::openrouter_images_api;
use crate::auth::{ProviderAuth, env_api_key_auth, models_error};
use crate::env_api_keys::get_env_api_key;
use crate::models::{
    CreateModelsOptions, CreateProviderOptions, Models, Provider, create_models, create_provider,
};
use crate::types::{
    AnyModel, ImageModel, ImageModelType, KnownImageApi, ModelInput, ModelOutput, ProviderImages,
};
use crate::utils::models_error::ModelsErrorCode;

const DEFAULT_PROVIDER_ID: &str = "openrouter";
const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

fn images(
    wrap: impl Fn(Arc<dyn ProviderImages>) -> Arc<dyn ProviderImages>,
) -> IndexMap<String, Arc<dyn ProviderImages>> {
    [(
        KnownImageApi::OpenrouterImages.as_str().to_string(),
        wrap(openrouter_images_api()),
    )]
    .into_iter()
    .collect()
}

/// `openrouterProvider()` (image models only).
pub fn openrouter_provider() -> Arc<dyn Provider> {
    create_provider(CreateProviderOptions {
        id: DEFAULT_PROVIDER_ID.to_string(),
        name: Some("OpenRouter".to_string()),
        base_url: Some(DEFAULT_BASE_URL.to_string()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "OpenRouter API key",
                &["OPENROUTER_API_KEY"],
            )),
            oauth: None,
        },
        models: openrouter_image_models()
            .values()
            .cloned()
            .map(AnyModel::Image)
            .collect(),
        images: Some(images(|api| api)),
        ..Default::default()
    })
    .expect("the OpenRouter provider has an images implementation")
}

/// Pre-1.0 image handle: `openrouter::builder().api_key(Some(..)).build()?`
/// then `handle.model("google/gemini-3.1-flash-image-preview").build_image()?`.
/// Requests go through the handle's own [`Models`] collection
/// (`handle.models().generate_images(..)`), which uses the handle's key, base
/// URL and HTTP client.
#[derive(Clone, Debug)]
pub struct OpenRouter {
    provider_id: String,
    base_url: String,
    models: Models,
}

impl OpenRouter {
    pub fn builder() -> OpenRouterBuilder {
        OpenRouterBuilder::default()
    }

    /// A handle that requires `OPENROUTER_API_KEY`.
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

    /// Start building an image model: the catalog entry when the id is known,
    /// otherwise a conservative default (text input, image output).
    pub fn model(&self, id: &str) -> ImageModelBuilder {
        let model = self
            .models
            .get_all_models(Some(&self.provider_id))
            .into_iter()
            .find_map(|model| match model {
                AnyModel::Image(model) if model.id == id => Some(model),
                _ => None,
            })
            .unwrap_or_else(|| ImageModel {
                id: id.to_string(),
                name: id.to_string(),
                api: KnownImageApi::OpenrouterImages.as_str().to_string(),
                provider: self.provider_id.clone(),
                base_url: self.base_url.clone(),
                model_type: ImageModelType::Image,
                input: vec![ModelInput::Text],
                output: vec![ModelOutput::Image],
                ..Default::default()
            });
        ImageModelBuilder::new(model)
    }

    /// Alias of [`OpenRouter::model`].
    pub fn image_model(&self, id: &str) -> ImageModelBuilder {
        self.model(id)
    }
}

/// `openrouter::builder()`.
pub fn builder() -> OpenRouterBuilder {
    OpenRouter::builder()
}

/// `openrouter::from_env()`.
pub fn from_env() -> Result<OpenRouter> {
    OpenRouter::from_env()
}

#[derive(Default)]
pub struct OpenRouterBuilder {
    provider_id: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    http_client: Option<reqwest::Client>,
}

impl OpenRouterBuilder {
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

    pub fn http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn build(self) -> Result<OpenRouter> {
        let provider_id = self
            .provider_id
            .unwrap_or_else(|| DEFAULT_PROVIDER_ID.to_string());
        let base_url = self
            .base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let auth = HandleAuth {
            name: "OpenRouter API key".to_string(),
            api_key: self.api_key,
            headers: None,
            fallback: Some(env_api_key_auth(
                "OpenRouter API key",
                &["OPENROUTER_API_KEY"],
            )),
            keyless: false,
        };
        let models = openrouter_image_models()
            .values()
            .map(|model| {
                AnyModel::Image(ImageModel {
                    provider: provider_id.clone(),
                    base_url: base_url.clone(),
                    ..model.clone()
                })
            })
            .collect();
        let http_client = self.http_client;
        let provider = create_provider(CreateProviderOptions {
            id: provider_id.clone(),
            name: Some("OpenRouter".to_string()),
            base_url: Some(base_url.clone()),
            auth: ProviderAuth {
                api_key: Some(Arc::new(auth)),
                oauth: None,
            },
            models,
            images: Some(images(|api| HandleImages::wrap(api, &http_client))),
            ..Default::default()
        })?;
        let collection = create_models(CreateModelsOptions::default());
        collection.set_provider(provider);
        Ok(OpenRouter {
            provider_id,
            base_url,
            models: collection,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::api::openai_client::test_support::{MockResponse, MockServer};
    use crate::types::{ImagesContext, ImagesOptions, ImagesStopReason, UserContent};

    fn image_response() -> MockResponse {
        MockResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: json!({
                "id": "img-1",
                "choices": [{ "message": {
                    "content": "",
                    "images": [{ "image_url": { "url": "data:image/png;base64,aGk=" } }],
                } }],
            })
            .to_string(),
        }
    }

    #[test]
    fn provider_lists_only_image_models() {
        let provider = openrouter_provider();
        assert!(provider.get_models().unwrap().is_empty());
        let all = provider.get_all_models().unwrap();
        assert_eq!(all.len(), openrouter_image_models().len());
        assert!(all.iter().all(|model| model.api() == "openrouter-images"));
        assert!(provider.supports_generate_images());
    }

    #[test]
    fn handle_builds_catalog_and_default_image_models() {
        let handle = builder().api_key(Some("key")).build().unwrap();
        let catalog = handle
            .model("google/gemini-3.1-flash-image-preview")
            .build_image()
            .unwrap();
        assert_eq!(catalog.output, vec![ModelOutput::Image, ModelOutput::Text]);
        let custom = handle
            .model("vendor/new-image")
            .input(vec![ModelInput::Text, ModelInput::Image])
            .output(vec![ModelOutput::Image, ModelOutput::Text])
            .build_image()
            .unwrap();
        assert_eq!(custom.api, "openrouter-images");
        assert_eq!(custom.base_url, DEFAULT_BASE_URL);
        assert_eq!(custom.input, vec![ModelInput::Text, ModelInput::Image]);
        assert!(
            handle
                .model("vendor/x")
                .output(vec![ModelOutput::Text])
                .build_image()
                .is_err()
        );
    }

    #[tokio::test]
    async fn generate_images_uses_the_handle_key_base_url_and_client() {
        let server = MockServer::start(vec![image_response()]).await;
        let handle = builder()
            .api_key(Some(" handle-key "))
            .base_url(server.url.clone())
            .http_client(reqwest::Client::new())
            .build()
            .unwrap();
        let model = handle
            .model("black-forest-labs/flux.2-pro")
            .build_image()
            .unwrap();
        assert_eq!(model.base_url, server.url);
        let output = handle
            .models()
            .generate_images(
                &model,
                &ImagesContext::builder().text("Generate a logo.").build(),
                ImagesOptions::default(),
            )
            .await;
        assert_eq!(output.stop_reason, ImagesStopReason::Stop);
        assert!(matches!(output.output[..], [UserContent::Image(_)]));
        let request = server.last();
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.header("authorization"), Some("Bearer handle-key"));
        assert_eq!(request.body["model"], "black-forest-labs/flux.2-pro");
    }

    #[tokio::test]
    async fn missing_api_key_returns_error_result() {
        if std::env::var("OPENROUTER_API_KEY").is_ok() {
            return;
        }
        let handle = builder().build().unwrap();
        let model = handle
            .model("black-forest-labs/flux.2-pro")
            .build_image()
            .unwrap();
        let output = handle
            .models()
            .generate_images(
                &model,
                &ImagesContext::builder().text("Generate a logo.").build(),
                ImagesOptions::default(),
            )
            .await;
        assert_eq!(output.stop_reason, ImagesStopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("Provider is not configured: openrouter")
        );
    }
}
