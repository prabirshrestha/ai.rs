use crate::images_api_registry::get_images_api_provider;
use crate::providers::images::register_builtins::ensure_builtins_registered;
use crate::types::{AssistantImages, ImagesContext, ImagesModel, ImagesOptions};
use crate::{Error, Result};

pub async fn generate_images(
    model: ImagesModel,
    context: ImagesContext,
    options: Option<ImagesOptions>,
) -> Result<AssistantImages> {
    ensure_builtins_registered();
    let provider = get_images_api_provider(&model.api)
        .ok_or_else(|| Error::UnsupportedApi(model.api.clone()))?;
    (provider.generate_images)(model, context, options.unwrap_or_default()).await
}
