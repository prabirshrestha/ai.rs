//! Port of the generated `providers/<provider>.models.ts` catalogs for the
//! in-scope providers. The data files are Pi 1.0.2's generated JSON
//! (`providers/data/*.json`), embedded at compile time and flattened with
//! `flatten_chat_model_catalog()` on first use.
//!
//! ai.rs extra, not in Pi: the embedding catalogs
//! (`data/openai-embeddings.json`, `data/github-copilot-embeddings.json`),
//! hand-written in the same grouped format. The OpenAI prices (input, per
//! million tokens) are from OpenAI's model pages, the default dimensions
//! and the 8192-token input limit from OpenAI's embeddings guide and the
//! `text-embedding-ada-002` announcement. The Copilot entry repeats the
//! OpenAI numbers, like Pi's Copilot catalog lists API-equivalent prices.
//!
//! OpenRouter ships its image and classifier catalogs
//! (`OPENROUTER_IMAGE_MODELS`, `OPENROUTER_CLASSIFIER_MODELS`: the
//! `openrouter-images` and `typesafe-system-one` groups of Pi's
//! `openrouter.json`); OpenRouter chat models are out of scope. TypeSafe
//! (`typesafe.json`) and Cloudflare Workers AI (`cloudflare-workers-ai.json`,
//! chat and classifier models) ship whole.

use std::sync::LazyLock;

use indexmap::IndexMap;

use crate::model_catalog::{
    ModelGroups, flatten_chat_model_catalog, flatten_classifier_model_catalog,
    flatten_embedding_model_catalog, flatten_image_model_catalog,
};
use crate::types::{ClassifierModel, EmbeddingModel, ImageModel, Model};

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

static OPENROUTER_IMAGE_MODELS: LazyLock<IndexMap<String, ImageModel>> = LazyLock::new(|| {
    let groups: ModelGroups = serde_json::from_str(include_str!("data/openrouter-images.json"))
        .expect("generated model catalog is valid JSON");
    flatten_image_model_catalog("openrouter", &groups)
        .expect("generated model catalog matches ImageModel")
});

fn load_classifiers(provider: &str, json: &str) -> IndexMap<String, ClassifierModel> {
    let groups: ModelGroups =
        serde_json::from_str(json).expect("generated model catalog is valid JSON");
    flatten_classifier_model_catalog(provider, &groups)
        .expect("generated model catalog matches ClassifierModel")
}

static CLOUDFLARE_WORKERS_AI_MODELS: LazyLock<IndexMap<String, Model>> = LazyLock::new(|| {
    load(
        "cloudflare-workers-ai",
        include_str!("data/cloudflare-workers-ai.json"),
    )
});
static CLOUDFLARE_WORKERS_AI_CLASSIFIER_MODELS: LazyLock<IndexMap<String, ClassifierModel>> =
    LazyLock::new(|| {
        load_classifiers(
            "cloudflare-workers-ai",
            include_str!("data/cloudflare-workers-ai.json"),
        )
    });
static TYPESAFE_CLASSIFIER_MODELS: LazyLock<IndexMap<String, ClassifierModel>> =
    LazyLock::new(|| load_classifiers("typesafe", include_str!("data/typesafe.json")));
static OPENROUTER_CLASSIFIER_MODELS: LazyLock<IndexMap<String, ClassifierModel>> =
    LazyLock::new(|| {
        load_classifiers(
            "openrouter",
            include_str!("data/openrouter-classifiers.json"),
        )
    });

fn load_embeddings(provider: &str, json: &str) -> IndexMap<String, EmbeddingModel> {
    let groups: ModelGroups = serde_json::from_str(json).expect("model catalog is valid JSON");
    flatten_embedding_model_catalog(provider, &groups)
        .expect("model catalog matches EmbeddingModel")
}

static OPENAI_EMBEDDING_MODELS: LazyLock<IndexMap<String, EmbeddingModel>> =
    LazyLock::new(|| load_embeddings("openai", include_str!("data/openai-embeddings.json")));
static GITHUB_COPILOT_EMBEDDING_MODELS: LazyLock<IndexMap<String, EmbeddingModel>> =
    LazyLock::new(|| {
        load_embeddings(
            "github-copilot",
            include_str!("data/github-copilot-embeddings.json"),
        )
    });

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

/// `CLOUDFLARE_WORKERS_AI_MODELS`.
pub fn cloudflare_workers_ai_models() -> &'static IndexMap<String, Model> {
    &CLOUDFLARE_WORKERS_AI_MODELS
}

/// `CLOUDFLARE_WORKERS_AI_CLASSIFIER_MODELS`.
pub fn cloudflare_workers_ai_classifier_models() -> &'static IndexMap<String, ClassifierModel> {
    &CLOUDFLARE_WORKERS_AI_CLASSIFIER_MODELS
}

/// `TYPESAFE_CLASSIFIER_MODELS`.
pub fn typesafe_classifier_models() -> &'static IndexMap<String, ClassifierModel> {
    &TYPESAFE_CLASSIFIER_MODELS
}

/// `OPENROUTER_CLASSIFIER_MODELS`.
pub fn openrouter_classifier_models() -> &'static IndexMap<String, ClassifierModel> {
    &OPENROUTER_CLASSIFIER_MODELS
}

/// OpenAI embedding models. ai.rs extra.
pub fn openai_embedding_models() -> &'static IndexMap<String, EmbeddingModel> {
    &OPENAI_EMBEDDING_MODELS
}

/// GitHub Copilot embedding models. ai.rs extra.
pub fn github_copilot_embedding_models() -> &'static IndexMap<String, EmbeddingModel> {
    &GITHUB_COPILOT_EMBEDDING_MODELS
}

/// `OPENROUTER_IMAGE_MODELS`.
pub fn openrouter_image_models() -> &'static IndexMap<String, ImageModel> {
    &OPENROUTER_IMAGE_MODELS
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
            (
                "cloudflare-workers-ai",
                include_str!("data/cloudflare-workers-ai.json"),
                cloudflare_workers_ai_models(),
            ),
        ] {
            let groups: ModelGroups = serde_json::from_str(json).unwrap();
            let raw: Vec<&Value> = groups
                .values()
                .flat_map(|models| models.values())
                .filter(|model| model["type"] == "chat")
                .collect();
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
        let groups: ModelGroups =
            serde_json::from_str(include_str!("data/openrouter-images.json")).unwrap();
        let raw: Vec<&Value> = groups.values().flat_map(|models| models.values()).collect();
        assert_eq!(raw.len(), openrouter_image_models().len());
        for value in raw {
            let model = &openrouter_image_models()[value["id"].as_str().unwrap()];
            assert_eq!(
                normalize_numbers(serde_json::to_value(model).unwrap()),
                normalize_numbers(value.clone())
            );
        }
        for (json, models) in [
            (
                include_str!("data/openai-embeddings.json"),
                openai_embedding_models(),
            ),
            (
                include_str!("data/github-copilot-embeddings.json"),
                github_copilot_embedding_models(),
            ),
        ] {
            let groups: ModelGroups = serde_json::from_str(json).unwrap();
            let raw: Vec<&Value> = groups.values().flat_map(|models| models.values()).collect();
            assert_eq!(raw.len(), models.len());
            for value in raw {
                let model = &models[value["id"].as_str().unwrap()];
                assert_eq!(
                    normalize_numbers(serde_json::to_value(model).unwrap()),
                    normalize_numbers(value.clone())
                );
            }
        }
        assert_eq!(openrouter_image_models().len(), 59);
        assert_eq!(openai_embedding_models().len(), 3);
        for (provider, json, models) in [
            (
                "typesafe",
                include_str!("data/typesafe.json"),
                typesafe_classifier_models(),
            ),
            (
                "cloudflare-workers-ai",
                include_str!("data/cloudflare-workers-ai.json"),
                cloudflare_workers_ai_classifier_models(),
            ),
            (
                "openrouter",
                include_str!("data/openrouter-classifiers.json"),
                openrouter_classifier_models(),
            ),
        ] {
            let groups: ModelGroups = serde_json::from_str(json).unwrap();
            let raw: Vec<&Value> = groups
                .values()
                .flat_map(|models| models.values())
                .filter(|model| model["type"] == "classifier")
                .collect();
            assert_eq!(raw.len(), models.len(), "{provider}");
            for value in raw {
                let model = &models[value["id"].as_str().unwrap()];
                assert_eq!(model.provider, provider);
                assert_eq!(
                    normalize_numbers(serde_json::to_value(model).unwrap()),
                    normalize_numbers(value.clone())
                );
            }
        }
        assert_eq!(typesafe_classifier_models().len(), 1);
        assert_eq!(cloudflare_workers_ai_classifier_models().len(), 3);
        assert_eq!(openrouter_classifier_models().len(), 13);
        assert_eq!(cloudflare_workers_ai_models().len(), 18);
        assert_eq!(github_copilot_embedding_models().len(), 1);
        assert_eq!(anthropic_models().len(), 16);
        assert_eq!(openai_models().len(), 44);
        assert_eq!(github_copilot_models().len(), 34);
    }
}
