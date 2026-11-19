//! [rust] Mirrors of the AI SDK (`ai`, `@ai-sdk/provider-utils`) types the
//! harness API is defined in. `ai.rs` has no `generateText` layer, so the
//! subset the harness touches lives here, named as in the AI SDK.

pub mod content;
pub mod delayed_promise;
pub mod error_value;
pub mod messages;
pub mod output;
pub mod parse_tool_call;
pub mod response_messages;
pub mod stop_condition;
pub mod telemetry;
pub mod text_stream_part;
pub mod tool;
pub mod tool_approval;
pub mod usage;

pub use content::*;
pub use delayed_promise::*;
pub use error_value::{ErrorValue, SharedError, ThrownValue, generate_id};
pub use messages::*;
pub use parse_tool_call::*;
pub use response_messages::*;
pub use stop_condition::*;
pub use text_stream_part::*;
pub use tool::*;
pub use tool_approval::*;
pub use usage::*;

/// `error.name` of the AI SDK error classes defined in this module.
pub(crate) fn named_error_name(err: &(dyn std::error::Error + 'static)) -> Option<&'static str> {
    if err.is::<NoSuchToolError>() {
        return Some("AI_NoSuchToolError");
    }
    if err.is::<InvalidToolInputError>() {
        return Some("AI_InvalidToolInputError");
    }
    if err.is::<crate::harness::provider::JsonParseError>() {
        return Some("AI_JSONParseError");
    }
    if err.is::<output::NoObjectGeneratedError>() {
        return Some("AI_NoObjectGeneratedError");
    }
    if err.is::<output::InvalidArgumentError>() {
        return Some("AI_InvalidArgumentError");
    }
    None
}
