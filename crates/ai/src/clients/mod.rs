use crate::chat_completions::ChatCompletion;
use dyn_clone::DynClone;

#[cfg(feature = "ollama_client")]
pub mod ollama;

#[cfg(feature = "openai_client")]
pub mod openai;

#[cfg(feature = "azure_openai_client")]
pub mod azure_openai;

pub trait Client: DynClone + ChatCompletion + Send + Sync {}

dyn_clone::clone_trait_object!(Client);
