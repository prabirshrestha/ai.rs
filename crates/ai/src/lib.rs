pub mod api_registry;
pub mod env_api_keys;
pub mod error;
pub mod event_stream;
pub mod image_models;
pub mod images;
pub mod images_api_registry;
pub mod models;
pub mod providers;
pub mod stream;
pub mod types;
pub mod utils;

pub use env_api_keys::{api_key_env_vars, find_env_keys, get_env_api_key};
pub use error::{Error, Result};
pub use event_stream::{AssistantMessageEventStream, AssistantMessageEventStreamSender};
pub use image_models::{
    get_image_model, get_image_models, get_image_providers, register_image_model,
};
pub use images::generate_images;
pub use models::{
    calculate_cost, clamp_thinking_level, get_model, get_models, get_providers,
    get_supported_thinking_levels, models_are_equal, register_model,
};
pub use providers::faux::{
    FauxAssistantContent, FauxAssistantMessageOptions, FauxModelDefinition,
    FauxProviderRegistration, FauxProviderState, FauxResponseStep, FauxTokenSize,
    RegisterFauxProviderOptions, faux_assistant_message, faux_text, faux_thinking, faux_tool_call,
    register_faux_provider,
};
pub use stream::{complete, complete_simple, stream, stream_simple};
pub use types::*;
pub use utils::diagnostics::{
    AssistantMessageDiagnostic, DiagnosticErrorInfo, append_assistant_message_diagnostic,
    create_assistant_message_diagnostic, diagnostic_error_from_message, extract_diagnostic_error,
    format_thrown_value,
};
pub use utils::overflow::{get_overflow_patterns, is_context_overflow};
