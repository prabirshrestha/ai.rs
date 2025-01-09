use derive_builder::Builder;

use crate::Result;

#[derive(Debug, Builder)]
pub struct EmbeddingRequest {}

#[derive(Debug, Builder)]
pub struct EmbeddingResponse {}

pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse>;
}
