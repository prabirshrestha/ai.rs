use crate::Result;
use async_trait::async_trait;
use derive_builder::Builder;
use dyn_clone::DynClone;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Builder)]
pub struct ChatCompletionRequest {}

#[derive(Debug, Serialize, Deserialize, Builder)]
pub struct ChatCompletionResponse {}

#[async_trait]
pub trait ChatCompletion: DynClone + Send + Sync {
    async fn complete(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse>;
}

dyn_clone::clone_trait_object!(ChatCompletion);
