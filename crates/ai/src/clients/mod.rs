pub mod ollama;
pub mod openai;

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

impl CompletionProvider for Client {
    fn complete(&self, request: &CompletionRequest) -> crate::Result<CompletionResponse> {
        match self {
            Self::OpenAI(client) => client.complete(request),
            Self::Any(client) => client.complete(request),
        }
    }
}

impl EmbeddingProvider for Client {
    fn embed(&self, request: &EmbeddingRequest) -> crate::Result<EmbeddingResponse> {
        match self {
            Self::OpenAI(client) => client.embed(request),
            Self::Any(client) => client.embed(request),
        }
    }
}

impl AnyClient for Client {}
