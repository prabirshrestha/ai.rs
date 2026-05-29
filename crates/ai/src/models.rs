use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::types::{Model, ModelThinkingLevel, Usage, UsageCost};

const EXTENDED_THINKING_LEVELS: [ModelThinkingLevel; 6] = [
    ModelThinkingLevel::Off,
    ModelThinkingLevel::Minimal,
    ModelThinkingLevel::Low,
    ModelThinkingLevel::Medium,
    ModelThinkingLevel::High,
    ModelThinkingLevel::Xhigh,
];

pub fn calculate_cost(model: &Model, usage: &mut Usage) -> UsageCost {
    usage.cost.input = (model.cost.input / 1_000_000.0) * usage.input as f64;
    usage.cost.output = (model.cost.output / 1_000_000.0) * usage.output as f64;
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000.0) * usage.cache_read as f64;
    usage.cost.cache_write = (model.cost.cache_write / 1_000_000.0) * usage.cache_write as f64;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage.cost.clone()
}

pub fn get_supported_thinking_levels(model: &Model) -> Vec<ModelThinkingLevel> {
    if !model.reasoning {
        return vec![ModelThinkingLevel::Off];
    }

    EXTENDED_THINKING_LEVELS
        .into_iter()
        .filter(|level| {
            let mapped = model.thinking_level_map.get(level.as_str());
            if matches!(mapped, Some(None)) {
                return false;
            }
            if *level == ModelThinkingLevel::Xhigh {
                return mapped.is_some();
            }
            true
        })
        .collect()
}

pub fn clamp_thinking_level(model: &Model, level: ModelThinkingLevel) -> ModelThinkingLevel {
    let available = get_supported_thinking_levels(model);
    if available.contains(&level) {
        return level;
    }

    let requested_index = EXTENDED_THINKING_LEVELS
        .iter()
        .position(|candidate| *candidate == level);
    let Some(requested_index) = requested_index else {
        return available
            .first()
            .copied()
            .unwrap_or(ModelThinkingLevel::Off);
    };

    for candidate in EXTENDED_THINKING_LEVELS.iter().skip(requested_index) {
        if available.contains(candidate) {
            return *candidate;
        }
    }
    for candidate in EXTENDED_THINKING_LEVELS.iter().take(requested_index).rev() {
        if available.contains(candidate) {
            return *candidate;
        }
    }
    available
        .first()
        .copied()
        .unwrap_or(ModelThinkingLevel::Off)
}

pub fn models_are_equal(a: Option<&Model>, b: Option<&Model>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.id == b.id && a.provider == b.provider,
        _ => false,
    }
}

type ModelRegistry = HashMap<String, HashMap<String, Model>>;

fn registry() -> &'static RwLock<ModelRegistry> {
    static REGISTRY: OnceLock<RwLock<ModelRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(builtin_models()))
}

pub fn register_model(provider: impl Into<String>, model: Model) {
    registry()
        .write()
        .expect("model registry poisoned")
        .entry(provider.into())
        .or_default()
        .insert(model.id.clone(), model);
}

pub fn get_model(provider: &str, model_id: &str) -> Option<Model> {
    registry()
        .read()
        .expect("model registry poisoned")
        .get(provider)
        .and_then(|models| models.get(model_id))
        .cloned()
}

pub fn get_providers() -> Vec<String> {
    let mut providers = registry()
        .read()
        .expect("model registry poisoned")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    providers.sort();
    providers
}

pub fn get_models(provider: &str) -> Vec<Model> {
    let mut models = registry()
        .read()
        .expect("model registry poisoned")
        .get(provider)
        .map(|models| models.values().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

fn builtin_models() -> ModelRegistry {
    serde_json::from_str(include_str!("models.generated.json"))
        .expect("generated model registry should deserialize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_registry_matches_upstream_catalog_size() {
        let registry = builtin_models();
        assert_eq!(registry.len(), 32);
        assert_eq!(
            registry.values().map(|models| models.len()).sum::<usize>(),
            931
        );
    }

    #[test]
    fn builtins_include_priority_models() {
        let gpt = get_model("openai", "gpt-5.5").expect("gpt-5.5");
        assert_eq!(gpt.api, "openai-responses");
        assert_eq!(
            gpt.thinking_level_map.get("off"),
            Some(&Some("none".to_string()))
        );

        let opus = get_model("anthropic", "claude-opus-4-8").expect("claude-opus-4-8");
        assert_eq!(
            opus.compat.anthropic_messages.force_adaptive_thinking,
            Some(true)
        );
        assert!(get_providers().contains(&"anthropic".to_string()));
    }

    #[test]
    fn register_model_overrides_by_provider_and_id() {
        let mut model = get_model("openai", "gpt-4o-mini").expect("gpt-4o-mini");
        model.id = "custom-test-model".to_string();
        model.provider = "test-provider".to_string();
        register_model("test-provider", model.clone());

        assert_eq!(get_model("test-provider", "custom-test-model"), Some(model));
        assert_eq!(get_models("test-provider").len(), 1);
    }
}
