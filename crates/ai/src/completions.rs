use async_trait::async_trait;
use derive_builder::Builder;

use crate::Result;

#[derive(Debug, Builder)]
pub struct CompletionRequest {
    pub model: String,
}

#[derive(Debug, Builder)]
pub struct CompletionResponse {}

#[async_trait]
pub trait CompletionProvider: Send + Sync {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse>;
}
