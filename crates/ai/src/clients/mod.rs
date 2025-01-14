pub mod openai;

use crate::chat_completions::ChatCompletion;
use dyn_clone::DynClone;

pub trait Client: DynClone + ChatCompletion + Send + Sync {}

dyn_clone::clone_trait_object!(Client);
