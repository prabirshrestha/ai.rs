use crate::types::{Model, ModelThinkingLevel, Usage, UsageCost};

const EXTENDED_THINKING_LEVELS: [ModelThinkingLevel; 7] = [
    ModelThinkingLevel::Off,
    ModelThinkingLevel::Minimal,
    ModelThinkingLevel::Low,
    ModelThinkingLevel::Medium,
    ModelThinkingLevel::High,
    ModelThinkingLevel::Xhigh,
    ModelThinkingLevel::Max,
];
pub fn calculate_cost(model: &Model, usage: &mut Usage) -> UsageCost {
    let input_tokens =
        u64::from(usage.input) + u64::from(usage.cache_read) + u64::from(usage.cache_write);
    let mut input_rate = model.cost.input;
    let mut output_rate = model.cost.output;
    let mut cache_read_rate = model.cost.cache_read;
    let mut cache_write_rate = model.cost.cache_write;
    let mut matched_threshold = None;
    for tier in &model.cost.tiers {
        if input_tokens > u64::from(tier.input_tokens_above)
            && matched_threshold.is_none_or(|threshold| tier.input_tokens_above > threshold)
        {
            input_rate = tier.input;
            output_rate = tier.output;
            cache_read_rate = tier.cache_read;
            cache_write_rate = tier.cache_write;
            matched_threshold = Some(tier.input_tokens_above);
        }
    }

    // Anthropic charges 2x the base input rate for one-hour cache writes.
    let long_write = usage.cache_write_1h.unwrap_or(0);
    let short_write = usage.cache_write.saturating_sub(long_write);
    usage.cost.input = (input_rate / 1_000_000.0) * usage.input as f64;
    usage.cost.output = (output_rate / 1_000_000.0) * usage.output as f64;
    usage.cost.cache_read = (cache_read_rate / 1_000_000.0) * usage.cache_read as f64;
    usage.cost.cache_write = (cache_write_rate * short_write as f64
        + input_rate * 2.0 * long_write as f64)
        / 1_000_000.0;
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
            if matches!(level, ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max) {
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
    use crate::types::{ModelCost, ModelCostTier, Usage};

    fn tiered_model() -> Model {
        Model {
            cost: ModelCost {
                input: 5.0,
                output: 30.0,
                cache_read: 0.5,
                cache_write: 6.25,
                tiers: vec![ModelCostTier {
                    input_tokens_above: 272_000,
                    input: 10.0,
                    output: 45.0,
                    cache_read: 1.0,
                    cache_write: 12.5,
                }],
            },
            ..Model::default()
        }
    }

    #[test]
    fn applies_request_wide_pricing_tiers_above_the_input_threshold() {
        let model = tiered_model();
        let mut short = Usage {
            input: 200_000,
            output: 100_000,
            cache_read: 72_000,
            ..Usage::default()
        };
        let mut long = Usage {
            cache_write: 1,
            ..short.clone()
        };

        let short_cost = calculate_cost(&model, &mut short);
        assert_eq!(short_cost.input, 1.0);
        assert_eq!(short_cost.output, 3.0);
        assert_eq!(short_cost.cache_read, 0.036);
        assert_eq!(short_cost.cache_write, 0.0);

        let long_cost = calculate_cost(&model, &mut long);
        assert_eq!(long_cost.input, 2.0);
        assert_eq!(long_cost.output, 4.5);
        assert_eq!(long_cost.cache_read, 0.072);
        assert_eq!(long_cost.cache_write, 0.0000125);
    }

    #[test]
    fn highest_matching_tier_applies_and_threshold_is_strict() {
        let mut model = tiered_model();
        model.cost.tiers.push(ModelCostTier {
            input_tokens_above: 400_000,
            input: 20.0,
            output: 60.0,
            cache_read: 2.0,
            cache_write: 25.0,
        });
        let mut at_threshold = Usage {
            input: 272_000,
            ..Usage::default()
        };
        let mut above_both = Usage {
            input: 400_001,
            output: 1_000_000,
            ..Usage::default()
        };

        assert_eq!(calculate_cost(&model, &mut at_threshold).input, 1.36);
        assert_eq!(calculate_cost(&model, &mut above_both).output, 60.0);
    }

    #[test]
    fn prices_anthropic_one_hour_cache_writes_at_twice_input() {
        let mut model = tiered_model();
        model.cost.tiers.clear();
        let mut usage = Usage {
            cache_write: 1_000_000,
            cache_write_1h: Some(400_000),
            ..Usage::default()
        };

        let cost = calculate_cost(&model, &mut usage);

        assert_eq!(cost.cache_write, 7.75);
    }

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
    fn supported_thinking_levels_match_extended_metadata() {
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
        gpt55_pro
            .thinking_level_map
            .insert("max".to_string(), Some("max".to_string()));

        assert_eq!(
            get_supported_thinking_levels(&gpt55_pro),
            vec![
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Xhigh,
                ModelThinkingLevel::Max,
            ]
        );
    }

    #[test]
    fn max_is_opt_in_for_ordinary_reasoning_models() {
        let model = Model {
            reasoning: true,
            ..Model::default()
        };

        assert_eq!(
            get_supported_thinking_levels(&model),
            vec![
                ModelThinkingLevel::Off,
                ModelThinkingLevel::Minimal,
                ModelThinkingLevel::Low,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
            ]
        );
        assert_eq!(
            clamp_thinking_level(&model, ModelThinkingLevel::Max),
            ModelThinkingLevel::High
        );
    }

    #[test]
    fn unsupported_xhigh_clamps_up_to_supported_max() {
        let mut model = Model {
            reasoning: true,
            ..Model::default()
        };
        model.thinking_level_map.insert("xhigh".to_string(), None);
        model
            .thinking_level_map
            .insert("max".to_string(), Some("max".to_string()));

        assert_eq!(
            get_supported_thinking_levels(&model),
            vec![
                ModelThinkingLevel::Off,
                ModelThinkingLevel::Minimal,
                ModelThinkingLevel::Low,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
                ModelThinkingLevel::Max,
            ]
        );
        assert_eq!(
            clamp_thinking_level(&model, ModelThinkingLevel::Xhigh),
            ModelThinkingLevel::Max
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
