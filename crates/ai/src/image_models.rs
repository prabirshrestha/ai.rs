use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::types::ImagesModel;

type ImageModelRegistry = HashMap<String, HashMap<String, ImagesModel>>;

fn registry() -> &'static RwLock<ImageModelRegistry> {
    static REGISTRY: OnceLock<RwLock<ImageModelRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(builtin_image_models()))
}

pub fn get_image_model(provider: &str, model_id: &str) -> Option<ImagesModel> {
    registry()
        .read()
        .expect("image model registry poisoned")
        .get(provider)
        .and_then(|models| models.get(model_id))
        .cloned()
}

pub fn get_image_providers() -> Vec<String> {
    let mut providers = registry()
        .read()
        .expect("image model registry poisoned")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    providers.sort();
    providers
}

pub fn get_image_models(provider: &str) -> Vec<ImagesModel> {
    let mut models = registry()
        .read()
        .expect("image model registry poisoned")
        .get(provider)
        .map(|models| models.values().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

pub fn register_image_model(provider: impl Into<String>, model: ImagesModel) {
    registry()
        .write()
        .expect("image model registry poisoned")
        .entry(provider.into())
        .or_default()
        .insert(model.id.clone(), model);
}

fn builtin_image_models() -> ImageModelRegistry {
    serde_json::from_str(include_str!("image_models.generated.json"))
        .expect("generated image model registry should deserialize")
}

#[cfg(test)]
mod tests {
    use crate::types::ModelInput;

    use super::*;

    #[test]
    fn generated_registry_matches_upstream_image_catalog_size() {
        let registry = builtin_image_models();
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry.values().map(|models| models.len()).sum::<usize>(),
            29
        );
    }

    #[test]
    fn builtins_include_openrouter_image_models() {
        let model =
            get_image_model("openrouter", "google/gemini-2.5-flash-image").expect("image model");
        assert_eq!(model.api, "openrouter-images");
        assert_eq!(model.provider, "openrouter");
        assert!(model.input.contains(&ModelInput::Text));
        assert!(model.output.contains(&ModelInput::Image));
        assert!(get_image_providers().contains(&"openrouter".to_string()));
    }

    #[test]
    fn register_image_model_overrides_by_provider_and_id() {
        let mut model =
            get_image_model("openrouter", "google/gemini-2.5-flash-image").expect("image model");
        model.id = "custom-image-model".to_string();
        model.provider = "test-image-provider".to_string();
        register_image_model("test-image-provider", model.clone());

        assert_eq!(
            get_image_model("test-image-provider", "custom-image-model"),
            Some(model)
        );
    }
}
