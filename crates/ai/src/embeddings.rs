use crate::Result;
use async_trait::async_trait;
use base64;
use base64::engine::Engine;
use derive_builder::Builder;
use dyn_clone::DynClone;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Builder)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option))]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Vec<String>,
    #[builder(default = "None")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[builder(default = "None")]
    #[serde(skip)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Builder)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

#[derive(Debug, Serialize, Deserialize, Builder)]
pub struct Base64EmbeddingsResponse {
    pub object: String,
    pub data: Vec<Base64EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

impl EmbeddingsResponse {
    pub fn to_base64(self) -> Base64EmbeddingsResponse {
        let data: Vec<Base64EmbeddingData> = self
            .data
            .into_iter()
            .map(|item| {
                // Convert the vector of f64 to bytes
                let bytes = item
                    .embedding
                    .iter()
                    .flat_map(|&f| f.to_le_bytes())
                    .collect::<Vec<u8>>();

                // Apply base64 encoding
                let base64_str = base64::engine::general_purpose::STANDARD.encode(&bytes);

                Base64EmbeddingData {
                    object: item.object,
                    embedding: base64_str,
                    index: item.index,
                }
            })
            .collect();

        Base64EmbeddingsResponse {
            object: self.object,
            data,
            model: self.model,
            usage: self.usage,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct Base64EmbeddingData {
    pub object: String,
    pub embedding: String,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder, Default)]
#[builder(setter(into, strip_option), default)]
#[builder(pattern = "mutable")]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[async_trait]
pub trait Embeddings: DynClone + Send + Sync {
    async fn create_embeddings(&self, request: &EmbeddingsRequest) -> Result<EmbeddingsResponse>;

    async fn create_base64_embeddings(
        &self,
        request: &EmbeddingsRequest,
    ) -> Result<Base64EmbeddingsResponse> {
        // Default implementation that converts float embeddings to base64
        let response = self.create_embeddings(request).await?;
        Ok(response.to_base64())
    }
}

dyn_clone::clone_trait_object!(Embeddings);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_embeddings_request_serialization() {
        let request = EmbeddingsRequestBuilder::default()
            .model("text-embedding-3-small")
            .input(vec!["Hello, world!".to_string()])
            .build()
            .unwrap();

        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"text-embedding-3-small","input":["Hello, world!"]}"#
        );
    }
}
