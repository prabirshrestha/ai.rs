//! Port of `providers/all.ts` for the in-scope providers (anthropic,
//! github-copilot, openai, and openrouter for image models only). Classifier
//! getters are not ported.

use std::sync::Arc;

use indexmap::IndexMap;

use super::anthropic::anthropic_provider;
use super::catalog::{
    anthropic_models, github_copilot_models, openai_models, openrouter_image_models,
};
use super::github_copilot::github_copilot_provider;
use super::openai::openai_provider;
use super::openrouter::openrouter_provider;
use crate::models::{CreateModelsOptions, Models, Provider, create_models};
use crate::types::{AnyModel, ImageModel, Model};

/// Providers present in the generated catalog (`BuiltinProvider`).
pub const BUILTIN_PROVIDERS: [&str; 4] = ["anthropic", "github-copilot", "openai", "openrouter"];

fn catalog(provider: &str) -> Option<&'static IndexMap<String, Model>> {
    match provider {
        "anthropic" => Some(anthropic_models()),
        "github-copilot" => Some(github_copilot_models()),
        "openai" => Some(openai_models()),
        _ => None,
    }
}

fn image_catalog(provider: &str) -> Option<&'static IndexMap<String, ImageModel>> {
    match provider {
        "openrouter" => Some(openrouter_image_models()),
        _ => None,
    }
}

/// Read of one generated built-in image model.
pub fn get_builtin_image_model(provider: &str, model_id: &str) -> Option<ImageModel> {
    image_catalog(provider)?.get(model_id).cloned()
}

pub fn get_builtin_image_models(provider: &str) -> Vec<ImageModel> {
    image_catalog(provider)
        .map(|models| models.values().cloned().collect())
        .unwrap_or_default()
}

/// Read of one generated built-in chat model.
pub fn get_builtin_model(provider: &str, model_id: &str) -> Option<Model> {
    catalog(provider)?.get(model_id).cloned()
}

pub fn get_builtin_providers() -> Vec<&'static str> {
    BUILTIN_PROVIDERS.to_vec()
}

pub fn get_builtin_models(provider: &str) -> Vec<Model> {
    catalog(provider)
        .map(|models| models.values().cloned().collect())
        .unwrap_or_default()
}

pub fn get_all_builtin_models(provider: &str) -> Vec<AnyModel> {
    get_builtin_models(provider)
        .into_iter()
        .map(AnyModel::Chat)
        .chain(
            get_builtin_image_models(provider)
                .into_iter()
                .map(AnyModel::Image),
        )
        .collect()
}

/// All built-in providers, freshly constructed.
pub fn builtin_providers() -> Vec<Arc<dyn Provider>> {
    vec![
        anthropic_provider(),
        github_copilot_provider(),
        openai_provider(),
        openrouter_provider(),
    ]
}

/// A `Models` collection with every built-in provider registered.
pub fn builtin_models(options: CreateModelsOptions) -> Models {
    let models = create_models(options);
    for provider in builtin_providers() {
        models.set_provider(provider);
    }
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_catalog_getters_return_reassignable_model_shapes() {
        let mut model = get_builtin_model("openai", "gpt-4o-mini").unwrap();
        assert_eq!(model.id, "gpt-4o-mini");
        model = get_builtin_model("openai", "gpt-4o").unwrap();
        assert_eq!(model.id, "gpt-4o");
        assert_eq!(get_builtin_model("openai", "missing"), None);
        assert_eq!(get_builtin_model("missing", "gpt-4o"), None);
        assert_eq!(get_builtin_providers(), BUILTIN_PROVIDERS.to_vec());
        assert_eq!(get_all_builtin_models("anthropic").len(), 16);
    }

    #[test]
    fn builtin_models_registers_every_provider() {
        let models = builtin_models(Default::default());
        assert_eq!(
            models
                .get_providers()
                .iter()
                .map(|provider| provider.id().to_string())
                .collect::<Vec<_>>(),
            BUILTIN_PROVIDERS.to_vec()
        );
        assert_eq!(models.get_models(None).len(), 16 + 34 + 44);
    }
}
