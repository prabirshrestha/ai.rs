//! Port of `images.ts`: global image generation dispatched on `model.api`
//! through the images api registry.
//!
//! Auth must be passed explicitly via `options.api_key`; prefer
//! [`Models::generate_images`](crate::models::Models::generate_images),
//! which resolves provider auth.
//!
//! Rust addition: an [`ImageModel`] built from a provider handle (such as
//! `providers::openrouter::builder()`) carries the handle's `Models`
//! collection (`ImageModel::bound_models`), and `generate_images()`
//! dispatches through it, resolving the handle's credentials.

use crate::images_api_registry::{RegisteredImagesApiProvider, get_images_api_provider};
use crate::types::{AssistantImages, ImageApi, ImageModel, ImagesContext, ImagesOptions};
use crate::{Error, Result};

fn resolve_images_api_provider(api: &ImageApi) -> Result<RegisteredImagesApiProvider> {
    get_images_api_provider(api)
        .ok_or_else(|| Error::message(format!("No API provider registered for api: {api}")))
}

/// `generateImages(model, context, options?)`. Fails when no implementation
/// is registered for `model.api`; request failures are reported in the
/// result (`stop_reason` error or aborted).
pub async fn generate_images(
    model: ImageModel,
    context: ImagesContext,
    options: Option<ImagesOptions>,
) -> Result<AssistantImages> {
    let options = options.unwrap_or_default();
    if let Some(models) = model.bound_models.clone() {
        return Ok(models.generate_images(&model, &context, options).await);
    }
    let provider = resolve_images_api_provider(&model.api)?;
    provider.generate_images(model, context, options).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images_api_registry::{ImagesApiProvider, images_fn, register_images_api_provider};
    use crate::types::{ImagesStopReason, UserContent};

    fn model(api: &str) -> ImageModel {
        ImageModel {
            id: "m".to_string(),
            name: "m".to_string(),
            api: api.to_string(),
            provider: "p".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn dispatches_unbound_models_through_the_images_api_registry() {
        register_images_api_provider(
            ImagesApiProvider {
                api: "test-images-registry".to_string(),
                generate_images: images_fn(|model, context, options| async move {
                    let mut output = AssistantImages::empty_for(&model);
                    output.output = context.input;
                    output.error_message = options.api_key;
                    output
                }),
            },
            Some("test"),
        );
        let output = generate_images(
            model("test-images-registry"),
            ImagesContext::builder().text("hi").build(),
            Some(ImagesOptions {
                api_key: Some("explicit".to_string()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(output.stop_reason, ImagesStopReason::Stop);
        assert_eq!(output.output, vec![UserContent::text("hi")]);
        assert_eq!(output.error_message.as_deref(), Some("explicit"));

        let registered = get_images_api_provider("test-images-registry").unwrap();
        let error = registered
            .generate_images(model("other"), ImagesContext::default(), Default::default())
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Mismatched api: other expected test-images-registry"
        );
    }

    #[tokio::test]
    async fn fails_when_no_api_is_registered_and_registers_builtins() {
        let error = generate_images(model("missing-images"), ImagesContext::default(), None)
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "No API provider registered for api: missing-images"
        );
        assert!(get_images_api_provider("openrouter-images").is_some());
        assert!(get_images_api_provider("openai-images").is_some());
        let output = generate_images(model("openrouter-images"), ImagesContext::default(), None)
            .await
            .unwrap();
        assert_eq!(output.stop_reason, ImagesStopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("No API key for provider: p")
        );
    }
}
