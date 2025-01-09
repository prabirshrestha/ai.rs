use derive_builder::Builder;

use crate::completions::{CompletionProvider, CompletionRequest, CompletionResponse};
use crate::embeddings::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use crate::Result;

#[derive(Debug, Clone, Builder)]
pub struct OllamaClient {
    base_url: String,
}

impl CompletionProvider for OllamaClient {
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        todo!()
    }
}

impl EmbeddingProvider for OllamaClient {
    fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        todo!()
    }
}
