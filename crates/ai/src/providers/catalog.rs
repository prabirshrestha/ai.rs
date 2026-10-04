//! Port of the generated `providers/<provider>.models.ts` catalogs for the
//! in-scope providers. The data files are Pi 1.0.2's generated JSON
//! (`providers/data/*.json`), embedded at compile time and flattened with
//! `flatten_chat_model_catalog()` on first use.

use std::sync::LazyLock;

use indexmap::IndexMap;

use crate::model_catalog::{ModelGroups, flatten_chat_model_catalog};
use crate::types::Model;

fn load(provider: &str, json: &str) -> IndexMap<String, Model> {
    let groups: ModelGroups =
        serde_json::from_str(json).expect("generated model catalog is valid JSON");
    flatten_chat_model_catalog(provider, &groups).expect("generated model catalog matches Model")
}

static ANTHROPIC_MODELS: LazyLock<IndexMap<String, Model>> =
    LazyLock::new(|| load("anthropic", include_str!("data/anthropic.json")));
static OPENAI_MODELS: LazyLock<IndexMap<String, Model>> =
    LazyLock::new(|| load("openai", include_str!("data/openai.json")));
static GITHUB_COPILOT_MODELS: LazyLock<IndexMap<String, Model>> =
    LazyLock::new(|| load("github-copilot", include_str!("data/github-copilot.json")));

/// `ANTHROPIC_MODELS`.
pub fn anthropic_models() -> &'static IndexMap<String, Model> {
    &ANTHROPIC_MODELS
}

/// `OPENAI_MODELS`.
pub fn openai_models() -> &'static IndexMap<String, Model> {
    &OPENAI_MODELS
}

/// `GITHUB_COPILOT_MODELS`.
pub fn github_copilot_models() -> &'static IndexMap<String, Model> {
    &GITHUB_COPILOT_MODELS
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    /// JSON has one number type; compare `10` and `10.0` as equal.
    fn normalize_numbers(value: Value) -> Value {
        match value {
            Value::Number(number) => Value::from(number.as_f64().unwrap()),
            Value::Array(items) => Value::Array(items.into_iter().map(normalize_numbers).collect()),
            Value::Object(entries) => Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, normalize_numbers(value)))
                    .collect(),
            ),
            value => value,
        }
    }

    /// Every catalog entry must survive a round trip through `Model`, so no
    /// generated field is silently dropped.
    #[test]
    fn catalog_entries_round_trip_without_losing_fields() {
        for (provider, json, models) in [
            (
                "anthropic",
                include_str!("data/anthropic.json"),
                anthropic_models(),
            ),
            ("openai", include_str!("data/openai.json"), openai_models()),
            (
                "github-copilot",
                include_str!("data/github-copilot.json"),
                github_copilot_models(),
            ),
        ] {
            let groups: ModelGroups = serde_json::from_str(json).unwrap();
            let raw: Vec<&Value> = groups.values().flat_map(|models| models.values()).collect();
            assert_eq!(raw.len(), models.len(), "{provider}");
            for value in raw {
                let id = value["id"].as_str().unwrap();
                let model = &models[id];
                assert_eq!(model.provider, provider);
                assert_eq!(
                    normalize_numbers(serde_json::to_value(model).unwrap()),
                    normalize_numbers(value.clone()),
                    "{provider}/{id}"
                );
            }
        }
        assert_eq!(anthropic_models().len(), 16);
        assert_eq!(openai_models().len(), 44);
        assert_eq!(github_copilot_models().len(), 34);
    }
}
