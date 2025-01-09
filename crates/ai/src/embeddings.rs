use async_trait::async_trait;
use derive_builder::Builder;

use crate::Result;

#[derive(Debug, Builder)]
pub struct EmbeddingRequest {}

#[derive(Debug, Builder)]
pub struct EmbeddingResponse {}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse>;
}
