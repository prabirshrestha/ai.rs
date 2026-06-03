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
        (Some(a), Some(b)) => a.id == b.id && a.provider == b.provider && a.api == b.api,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_equality_distinguishes_api_mode() {
        let responses = Model {
            id: "gpt-5.5".to_string(),
            provider: "openai".to_string(),
            api: "openai-responses".to_string(),
            ..Model::default()
        };
        let chat = Model {
            api: "openai-completions".to_string(),
            ..responses.clone()
        };

        assert!(!models_are_equal(Some(&responses), Some(&chat)));
    }

    #[test]
    fn supported_thinking_levels_match_xhigh_metadata() {
        let mut gpt55_pro = Model {
            reasoning: true,
            ..Model::default()
        };
        gpt55_pro.thinking_level_map.insert("off".to_string(), None);
        gpt55_pro
            .thinking_level_map
            .insert("minimal".to_string(), None);
        gpt55_pro.thinking_level_map.insert("low".to_string(), None);
        gpt55_pro
            .thinking_level_map
            .insert("xhigh".to_string(), Some("high".to_string()));

        assert_eq!(
            get_supported_thinking_levels(&gpt55_pro),
            vec![
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh,
            ]
        );
    }

    #[test]
    fn non_reasoning_models_only_support_off() {
        let model = Model::default();

        assert_eq!(
            get_supported_thinking_levels(&model),
            vec![ModelThinkingLevel::Off]
        );
    }
}
