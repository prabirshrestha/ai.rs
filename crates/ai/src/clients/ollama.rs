use async_trait::async_trait;
use derive_builder::Builder;

use crate::completions::{CompletionProvider, CompletionRequest, CompletionResponse};
use crate::embeddings::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use crate::Result;

#[derive(Debug, Clone, Builder)]
pub struct OllamaClient {
    base_url: String,
}

#[async_trait]
impl CompletionProvider for OllamaClient {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        todo!()
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaClient {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        todo!()
    }
}
