pub mod ollama;
pub mod openai;

use async_trait::async_trait;
use dyn_clone::DynClone;

use crate::{
    completions::{CompletionProvider, CompletionRequest, CompletionResponse},
    embeddings::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse},
};

pub trait AnyClient: DynClone + CompletionProvider + EmbeddingProvider + Send + Sync {}

dyn_clone::clone_trait_object!(AnyClient);

#[derive(Clone)]
pub enum Client {
    OpenAI(openai::OpenAIClient),
    Any(Box<dyn AnyClient>),
}

#[async_trait]
impl CompletionProvider for Client {
    async fn complete(&self, request: &CompletionRequest) -> crate::Result<CompletionResponse> {
        match self {
            Self::OpenAI(client) => client.complete(request).await,
            Self::Any(client) => client.complete(request).await,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for Client {
    async fn embed(&self, request: &EmbeddingRequest) -> crate::Result<EmbeddingResponse> {
        match self {
            Self::OpenAI(client) => client.embed(request).await,
            Self::Any(client) => client.embed(request).await,
        }
    }
}

impl AnyClient for Client {}
