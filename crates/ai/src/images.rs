use crate::env_api_keys::get_env_api_key;
use crate::images_api_registry::get_images_api_provider;
use crate::providers::images::register_builtins::ensure_builtins_registered;
use crate::types::{AssistantImages, ImagesContext, ImagesModel, ImagesOptions};
use crate::{Error, Result};

fn has_explicit_api_key(api_key: &Option<String>) -> bool {
    api_key
        .as_deref()
        .is_some_and(|api_key| !api_key.trim().is_empty())
}

fn with_env_api_key(model: &ImagesModel, mut options: ImagesOptions) -> ImagesOptions {
    if !has_explicit_api_key(&options.api_key) {
        if let Some(api_key) = get_env_api_key(&model.provider) {
            options.api_key = Some(api_key);
        }
    }
    options
}

pub async fn generate_images(
    model: ImagesModel,
    context: ImagesContext,
    options: Option<ImagesOptions>,
) -> Result<AssistantImages> {
    ensure_builtins_registered();
    let provider = get_images_api_provider(&model.api)
        .ok_or_else(|| Error::UnsupportedApi(model.api.clone()))?;
    let options = with_env_api_key(&model, options.unwrap_or_default());
    (provider.generate_images)(model, context, options).await
}
