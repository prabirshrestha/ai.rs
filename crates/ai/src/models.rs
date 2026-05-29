use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::types::{
    AnthropicMessagesCompat, Model, ModelCompat, ModelCost, ModelInput, ModelThinkingLevel, Usage,
    UsageCost,
};

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
    let mut registry = HashMap::new();
    for model in [
        openai_gpt_55(),
        openai_gpt_4o_mini(),
        anthropic_claude_haiku_45(),
        anthropic_claude_sonnet_45(),
        anthropic_claude_opus_48(),
    ] {
        registry
            .entry(model.provider.clone())
            .or_insert_with(HashMap::new)
            .insert(model.id.clone(), model);
    }
    registry
}

fn text_image() -> Vec<ModelInput> {
    vec![ModelInput::Text, ModelInput::Image]
}

fn thinking_map(entries: &[(&str, Option<&str>)]) -> HashMap<String, Option<String>> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.map(ToString::to_string)))
        .collect()
}

fn model_cost(input: f64, output: f64, cache_read: f64, cache_write: f64) -> ModelCost {
    ModelCost {
        input,
        output,
        cache_read,
        cache_write,
    }
}

fn openai_gpt_55() -> Model {
    Model {
        id: "gpt-5.5".to_string(),
        name: "GPT-5.5".to_string(),
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        reasoning: true,
        thinking_level_map: thinking_map(&[("off", Some("none")), ("xhigh", Some("xhigh"))]),
        input: text_image(),
        cost: model_cost(5.0, 30.0, 0.5, 0.0),
        context_window: 272_000,
        max_tokens: 128_000,
        ..Default::default()
    }
}

fn openai_gpt_4o_mini() -> Model {
    Model {
        id: "gpt-4o-mini".to_string(),
        name: "GPT-4o mini".to_string(),
        api: "openai-responses".to_string(),
        provider: "openai".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        reasoning: false,
        input: text_image(),
        cost: model_cost(0.15, 0.6, 0.075, 0.0),
        context_window: 128_000,
        max_tokens: 16_384,
        ..Default::default()
    }
}

fn anthropic_claude_haiku_45() -> Model {
    Model {
        id: "claude-haiku-4-5".to_string(),
        name: "Claude Haiku 4.5 (latest)".to_string(),
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        base_url: "https://api.anthropic.com".to_string(),
        reasoning: true,
        input: text_image(),
        cost: model_cost(1.0, 5.0, 0.1, 1.25),
        context_window: 200_000,
        max_tokens: 64_000,
        ..Default::default()
    }
}

fn anthropic_claude_sonnet_45() -> Model {
    Model {
        id: "claude-sonnet-4-5".to_string(),
        name: "Claude Sonnet 4.5 (latest)".to_string(),
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        base_url: "https://api.anthropic.com".to_string(),
        reasoning: true,
        input: text_image(),
        cost: model_cost(3.0, 15.0, 0.3, 3.75),
        context_window: 200_000,
        max_tokens: 64_000,
        ..Default::default()
    }
}

fn anthropic_claude_opus_48() -> Model {
    Model {
        id: "claude-opus-4-8".to_string(),
        name: "Claude Opus 4.8".to_string(),
        api: "anthropic-messages".to_string(),
        provider: "anthropic".to_string(),
        base_url: "https://api.anthropic.com".to_string(),
        reasoning: true,
        thinking_level_map: thinking_map(&[("xhigh", Some("xhigh"))]),
        input: text_image(),
        cost: model_cost(5.0, 25.0, 0.5, 6.25),
        context_window: 1_000_000,
        max_tokens: 128_000,
        compat: ModelCompat {
            anthropic_messages: AnthropicMessagesCompat {
                force_adaptive_thinking: Some(true),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut model = openai_gpt_4o_mini();
        model.id = "custom-test-model".to_string();
        model.provider = "test-provider".to_string();
        register_model("test-provider", model.clone());

        assert_eq!(get_model("test-provider", "custom-test-model"), Some(model));
        assert_eq!(get_models("test-provider").len(), 1);
    }
}
