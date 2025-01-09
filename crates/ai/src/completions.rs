use derive_builder::Builder;

use crate::Result;

#[derive(Debug, Builder)]
pub struct CompletionRequest {
    pub model: String,
}

#[derive(Debug, Builder)]
pub struct CompletionResponse {}

pub trait CompletionProvider: Send + Sync {
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse>;
}
