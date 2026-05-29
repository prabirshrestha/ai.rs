pub mod api_registry;
pub mod env_api_keys;
pub mod error;
pub mod event_stream;
pub mod models;
pub mod providers;
pub mod stream;
pub mod types;
pub mod utils;

pub use env_api_keys::{api_key_env_vars, find_env_keys, get_env_api_key};
pub use error::{Error, Result};
pub use event_stream::{AssistantMessageEventStream, AssistantMessageEventStreamSender};
pub use models::{
    calculate_cost, clamp_thinking_level, get_model, get_models, get_providers,
    get_supported_thinking_levels, models_are_equal, register_model,
};
pub use stream::{complete, complete_simple, stream, stream_simple};
pub use types::*;
