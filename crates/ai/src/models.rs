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
        let step =
            get_model("openrouter", "stepfun/step-3.7-flash").expect("stepfun/step-3.7-flash");
        assert_eq!(step.max_tokens, 256_000);
        assert!(get_providers().contains(&"anthropic".to_string()));
    }

    #[test]
    fn supported_thinking_levels_match_xhigh_metadata() {
        let opus46 = get_model("anthropic", "claude-opus-4-6").expect("claude-opus-4-6");
        assert!(get_supported_thinking_levels(&opus46).contains(&ModelThinkingLevel::Xhigh));

        let opus48 = get_model("anthropic", "claude-opus-4-8").expect("claude-opus-4-8");
        assert!(get_supported_thinking_levels(&opus48).contains(&ModelThinkingLevel::Xhigh));

        let sonnet45 = get_model("anthropic", "claude-sonnet-4-5").expect("claude-sonnet-4-5");
        assert!(!get_supported_thinking_levels(&sonnet45).contains(&ModelThinkingLevel::Xhigh));

        for model_id in ["gpt-5.4", "gpt-5.5"] {
            let model = get_model("openai-codex", model_id).expect(model_id);
            assert!(get_supported_thinking_levels(&model).contains(&ModelThinkingLevel::Xhigh));
        }

        let gpt55_pro = get_model("openai", "gpt-5.5-pro").expect("gpt-5.5-pro");
        assert_eq!(
            get_supported_thinking_levels(&gpt55_pro),
            vec![
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh,
            ]
        );

        let openrouter_gpt55_pro =
            get_model("openrouter", "openai/gpt-5.5-pro").expect("openai/gpt-5.5-pro");
        assert_eq!(
            get_supported_thinking_levels(&openrouter_gpt55_pro),
            vec![
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh,
            ]
        );

        for (provider, model_id, expected) in [
            (
                "deepseek",
                "deepseek-v4-flash",
                vec![
                    ModelThinkingLevel::Off,
                    ModelThinkingLevel::High,
                    ModelThinkingLevel::Xhigh,
                ],
            ),
            (
                "opencode-go",
                "deepseek-v4-flash",
                vec![
                    ModelThinkingLevel::Off,
                    ModelThinkingLevel::High,
                    ModelThinkingLevel::Xhigh,
                ],
            ),
            (
                "opencode-go",
                "kimi-k2.6",
                vec![ModelThinkingLevel::Off, ModelThinkingLevel::High],
            ),
            ("opencode", "grok-build-0.1", vec![ModelThinkingLevel::High]),
            (
                "openrouter",
                "deepseek/deepseek-v4-flash",
                vec![
                    ModelThinkingLevel::Off,
                    ModelThinkingLevel::High,
                    ModelThinkingLevel::Xhigh,
                ],
            ),
        ] {
            let model = get_model(provider, model_id).expect(model_id);
            assert_eq!(get_supported_thinking_levels(&model), expected);
        }

        let openrouter_opus46 = get_model("openrouter", "anthropic/claude-opus-4.6")
            .expect("anthropic/claude-opus-4.6");
        assert!(
            get_supported_thinking_levels(&openrouter_opus46).contains(&ModelThinkingLevel::Xhigh)
        );
    }

    #[test]
    fn anthropic_messages_adaptive_thinking_metadata_is_limited_to_supported_models() {
        let mut flagged = get_providers()
            .into_iter()
            .flat_map(|provider| get_models(&provider))
            .filter(|model| model.api == "anthropic-messages")
            .filter(|model| model.compat.anthropic_messages.force_adaptive_thinking == Some(true))
            .map(|model| format!("{}/{}", model.provider, model.id))
            .collect::<Vec<_>>();
        flagged.sort();

        for expected in [
            "anthropic/claude-opus-4-8",
            "opencode/claude-opus-4-8",
            "vercel-ai-gateway/anthropic/claude-opus-4.8",
        ] {
            assert!(flagged.contains(&expected.to_string()), "{expected}");
        }

        assert!(
            flagged.iter().all(|model_id| model_id.contains("opus-4-6")
                || model_id.contains("opus-4.6")
                || model_id.contains("opus-4-7")
                || model_id.contains("opus-4.7")
                || model_id.contains("opus-4-8")
                || model_id.contains("opus-4.8")
                || model_id.contains("sonnet-4-6")
                || model_id.contains("sonnet-4.6")),
            "{flagged:?}"
        );
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
