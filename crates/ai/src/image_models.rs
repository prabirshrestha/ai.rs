//! Port of `image-models.ts`: compat reads of the generated catalog
//! restricted to image models. New code uses
//! `Models::get_model_of_type(ModelType::Image, ..)` or
//! `providers::all::get_builtin_image_model()`.
//!
//! The in-scope image catalog is OpenRouter's.

use crate::providers::all::{BUILTIN_PROVIDERS, get_builtin_image_model, get_builtin_image_models};
use crate::types::ImageModel;

/// `getImageModel(provider, modelId)`. Deprecated static catalog read.
pub fn get_image_model(provider: &str, model_id: &str) -> Option<ImageModel> {
    get_builtin_image_model(provider, model_id)
}

/// `getImageProviders()`: built-in providers with at least one image model.
pub fn get_image_providers() -> Vec<&'static str> {
    BUILTIN_PROVIDERS
        .iter()
        .copied()
        .filter(|provider| !get_builtin_image_models(provider).is_empty())
        .collect()
}

/// `getImageModels(provider)`.
pub fn get_image_models(provider: &str) -> Vec<ImageModel> {
    get_builtin_image_models(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_openrouter_image_catalog() {
        assert_eq!(get_image_providers(), vec!["openrouter"]);
        let model = get_image_model("openrouter", "google/gemini-2.5-flash-image").unwrap();
        assert_eq!(model.api, "openrouter-images");
        assert_eq!(model.base_url, "https://openrouter.ai/api/v1");
        assert!(get_image_model("openai", "gpt-image-1").is_none());
        assert_eq!(get_image_models("openrouter").len(), 59);
        assert!(get_image_models("anthropic").is_empty());
    }
}
