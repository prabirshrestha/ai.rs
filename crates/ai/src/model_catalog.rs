//! Port of `model-catalog.ts`: flatten generated catalog groups
//! (`{ api: { "<type>:<id>": model } }`) into id-keyed catalogs.
//!
//! Pi's mapped catalog types have no Rust equivalent; catalogs are ordered
//! maps from model id to model. Classifier catalogs are not ported.

use indexmap::IndexMap;
use serde_json::Value;

use crate::types::{ImageModel, Model, ModelType};

/// Generated catalog groups, keyed by api and then by `"<type>:<id>"`.
pub type ModelGroups = IndexMap<String, IndexMap<String, Value>>;

fn flatten_model_catalog(groups: &ModelGroups, model_type: ModelType) -> Vec<Value> {
    let type_name = match model_type {
        ModelType::Chat => "chat",
        ModelType::Image => "image",
    };
    groups
        .values()
        .flat_map(|models| models.values())
        .filter(|model| model.get("type").and_then(Value::as_str) == Some(type_name))
        .cloned()
        .collect()
}

/// `flattenChatModelCatalog()`.
pub fn flatten_chat_model_catalog(
    _provider: &str,
    groups: &ModelGroups,
) -> serde_json::Result<IndexMap<String, Model>> {
    flatten_model_catalog(groups, ModelType::Chat)
        .into_iter()
        .map(|model| serde_json::from_value::<Model>(model).map(|model| (model.id.clone(), model)))
        .collect()
}

/// `flattenImageModelCatalog()`.
pub fn flatten_image_model_catalog(
    _provider: &str,
    groups: &ModelGroups,
) -> serde_json::Result<IndexMap<String, ImageModel>> {
    flatten_model_catalog(groups, ModelType::Image)
        .into_iter()
        .map(|model| {
            serde_json::from_value::<ImageModel>(model).map(|model| (model.id.clone(), model))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn flattens_groups_by_model_type() {
        let groups: ModelGroups = serde_json::from_value(json!({
            "test-api": {
                "chat:a": { "type": "chat", "id": "a", "name": "A", "api": "test-api", "provider": "p", "baseUrl": "", "reasoning": false, "input": ["text"], "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }, "contextWindow": 1, "maxTokens": 1 },
                "image:b": { "type": "image", "id": "b", "name": "B", "api": "test-images", "provider": "p", "baseUrl": "", "input": ["text"], "output": ["image"], "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 } }
            }
        }))
        .unwrap();
        let chat = flatten_chat_model_catalog("p", &groups).unwrap();
        assert_eq!(chat.keys().collect::<Vec<_>>(), vec!["a"]);
        let images = flatten_image_model_catalog("p", &groups).unwrap();
        assert_eq!(images.keys().collect::<Vec<_>>(), vec!["b"]);
    }
}
