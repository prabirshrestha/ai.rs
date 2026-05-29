use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock, RwLock};

use crate::types::{AssistantImages, ImagesContext, ImagesModel, ImagesOptions};
use crate::{Error, Result};

pub type ImagesApiFunction = Arc<
    dyn Fn(
            ImagesModel,
            ImagesContext,
            ImagesOptions,
        ) -> Pin<Box<dyn Future<Output = Result<AssistantImages>> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct ImagesApiProvider {
    pub api: String,
    pub generate_images: ImagesApiFunction,
}

#[derive(Clone)]
struct RegisteredImagesApiProvider {
    provider: ImagesApiProvider,
    source_id: Option<String>,
}

fn registry() -> &'static RwLock<HashMap<String, RegisteredImagesApiProvider>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, RegisteredImagesApiProvider>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn register_images_api_provider(provider: ImagesApiProvider, source_id: Option<String>) {
    registry()
        .write()
        .expect("images registry poisoned")
        .insert(
            provider.api.clone(),
            RegisteredImagesApiProvider {
                provider,
                source_id,
            },
        );
}

pub fn get_images_api_provider(api: &str) -> Option<ImagesApiProvider> {
    registry()
        .read()
        .expect("images registry poisoned")
        .get(api)
        .map(|entry| entry.provider.clone())
}

pub fn unregister_images_api_providers(source_id: &str) {
    registry()
        .write()
        .expect("images registry poisoned")
        .retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

pub fn clear_images_api_providers() {
    registry()
        .write()
        .expect("images registry poisoned")
        .clear();
}

pub fn wrap_generate_images<F, Fut>(api: &'static str, generate_images: F) -> ImagesApiFunction
where
    F: Fn(ImagesModel, ImagesContext, ImagesOptions) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<AssistantImages>> + Send + 'static,
{
    Arc::new(move |model, context, options| {
        if model.api != api {
            return Box::pin(async move {
                Err(Error::UnsupportedApi(format!(
                    "Mismatched api: {} expected {}",
                    model.api, api
                )))
            });
        }
        Box::pin(generate_images(model, context, options))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ImagesContent, ImagesStopReason, ModelCost, ModelInput};

    fn test_model(api: &str) -> ImagesModel {
        ImagesModel {
            id: "image-model".to_string(),
            name: "image model".to_string(),
            api: api.to_string(),
            provider: "test-provider".to_string(),
            base_url: "http://localhost".to_string(),
            input: vec![ModelInput::Text],
            output: vec![ModelInput::Image],
            cost: ModelCost::default(),
            headers: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn registered_provider_dispatches_generate_images() {
        let source_id = "images-api-registry-test";
        register_images_api_provider(
            ImagesApiProvider {
                api: "test-images".to_string(),
                generate_images: wrap_generate_images(
                    "test-images",
                    |model, _context, _options| async move {
                        Ok(AssistantImages {
                            api: model.api,
                            provider: model.provider,
                            model: model.id,
                            output: vec![ImagesContent::text("ok")],
                            response_id: None,
                            usage: None,
                            stop_reason: ImagesStopReason::Stop,
                            error_message: None,
                            timestamp: crate::utils::time::now_millis(),
                        })
                    },
                ),
            },
            Some(source_id.to_string()),
        );

        let provider = get_images_api_provider("test-images").expect("provider");
        let output = (provider.generate_images)(
            test_model("test-images"),
            ImagesContext::default(),
            ImagesOptions::default(),
        )
        .await
        .expect("images");
        assert_eq!(output.output, vec![ImagesContent::text("ok")]);

        unregister_images_api_providers(source_id);
        assert!(get_images_api_provider("test-images").is_none());
    }

    #[tokio::test]
    async fn wrapped_provider_rejects_mismatched_api() {
        let generate_images =
            wrap_generate_images("expected-images", |_model, _context, _options| async {
                unreachable!("mismatched api should fail before provider call")
            });
        let error = generate_images(
            test_model("actual-images"),
            ImagesContext::default(),
            ImagesOptions::default(),
        )
        .await
        .expect_err("mismatched api error");

        assert!(
            matches!(error, Error::UnsupportedApi(message) if message == "Mismatched api: actual-images expected expected-images")
        );
    }
}
