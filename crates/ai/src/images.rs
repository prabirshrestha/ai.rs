use crate::types::{AssistantImages, ImageGenerationOptions, ImagesContext, Model};
use crate::{Error, Result};

pub async fn generate_images(
    model: Model,
    context: ImagesContext,
    options: Option<ImageGenerationOptions>,
) -> Result<AssistantImages> {
    let api = model
        .image_api()
        .ok_or_else(|| Error::unsupported_capability(model.provider.clone(), "image models"))?;
    api.generate_images(model, context, options.unwrap_or_default())
        .await
}
