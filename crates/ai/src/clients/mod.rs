use crate::chat_completions::ChatCompletion;
use dyn_clone::DynClone;

pub mod ollama;
pub mod openai;

pub trait Client: DynClone + ChatCompletion + Send + Sync {}

dyn_clone::clone_trait_object!(Client);
