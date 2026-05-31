pub mod agent;
pub mod agent_error;
pub mod agent_loop;
pub mod agent_types;
pub mod api_registry;
pub mod env_api_keys;
pub mod error;
pub mod event_stream;
pub mod models;
pub mod oauth;
pub mod providers;
pub mod session_resources;
pub mod stream;
pub mod types;
pub mod utils;

#[cfg(test)]
pub(crate) mod test_env;

pub use agent::{Agent, AgentListenerId, AgentOptions, AgentPrepareNextTurnFn, AgentState};
pub use agent_error::{AgentError, AgentResult};
pub use agent_loop::{
    AgentEventStream, agent_loop, agent_loop_continue, run_agent_loop, run_agent_loop_continue,
};
pub use agent_types::*;
pub use api_registry::{
    ApiProvider, ApiStreamFunction, ApiStreamSimpleFunction, clear_api_providers, get_api_provider,
    get_api_providers, register_api_provider, unregister_api_providers,
};
pub use env_api_keys::{api_key_env_vars, find_env_keys, get_env_api_key};
pub use error::{Error, Result};
pub use event_stream::{
    AssistantMessageEventStream, AssistantMessageEventStreamSender,
    create_assistant_message_event_stream,
};
pub use models::{
    calculate_cost, clamp_thinking_level, get_model, get_models, get_providers,
    get_supported_thinking_levels, models_are_equal,
};
pub use oauth::{
    AnthropicOAuthProvider, GitHubCopilotOAuthProvider, OAuthApiKey, OAuthAuthCallback,
    OAuthAuthInfo, OAuthCredentials, OAuthDeviceCodeInfo, OAuthDeviceCodePollResult,
    OAuthLoginCallbacks, OAuthManualCodeInputCallback, OAuthPrompt, OAuthProvider, OAuthProviderId,
    OAuthProviderInfo, OAuthProviderInterface, OAuthSelectCallback, OAuthSelectOption,
    OAuthSelectPrompt, anthropic_oauth_provider, exchange_anthropic_authorization_code,
    get_github_copilot_base_url, get_oauth_api_key, get_oauth_provider,
    get_oauth_provider_info_list, get_oauth_providers, github_copilot_oauth_provider,
    login_anthropic, login_github_copilot, modify_github_copilot_models, normalize_domain,
    poll_oauth_device_code_flow, refresh_anthropic_token, refresh_github_copilot_token,
    refresh_oauth_token, register_oauth_provider, reset_oauth_providers, unregister_oauth_provider,
};
pub use providers::anthropic::{
    AnthropicClient, AnthropicClientRequest, AnthropicEffort, AnthropicOptions,
    AnthropicThinkingDisplay, ResolvedAnthropicCompat, build_anthropic_payload,
    convert_messages as convert_anthropic_messages, stream_anthropic, stream_simple_anthropic,
};
pub use providers::faux::{
    FauxAssistantContent, FauxAssistantMessageOptions, FauxModelDefinition,
    FauxProviderRegistration, FauxProviderState, FauxResponseStep, FauxTokenSize,
    RegisterFauxProviderOptions, faux_assistant_message, faux_text, faux_thinking, faux_tool_call,
    register_faux_provider,
};
pub use providers::github_copilot_headers::{
    build_copilot_dynamic_headers, has_copilot_vision_input, infer_copilot_initiator,
};
pub use providers::openai_completions::{
    OpenAICompletionsOptions, ResolvedOpenAICompletionsCompat, build_chat_completions_payload,
    convert_messages as convert_openai_completions_messages,
    detect_compat as detect_openai_completions_compat, get_compat as get_openai_completions_compat,
    stream_openai_completions, stream_simple_openai_completions,
};
pub use providers::openai_responses::{
    OpenAIResponsesAuthHeader, OpenAIResponsesOptions, ResolvedOpenAIResponsesCompat,
    build_responses_payload, convert_responses_messages, convert_responses_tools,
    get_compat as get_openai_responses_compat, stream_openai_responses,
    stream_simple_openai_responses,
};
pub use providers::register_builtins::{register_builtin_api_providers, reset_api_providers};
pub use session_resources::{
    SessionResourceCleanup, SessionResourceCleanupRegistration, cleanup_session_resources,
    register_session_resource_cleanup,
};
pub use stream::{complete, complete_simple, stream, stream_simple};
pub use types::*;
pub use utils::diagnostics::{
    AssistantMessageDiagnostic, DiagnosticErrorInfo, append_assistant_message_diagnostic,
    create_assistant_message_diagnostic, diagnostic_error_from_message, extract_diagnostic_error,
    format_thrown_value,
};
pub use utils::json::{parse_json_with_repair, parse_streaming_json, repair_json};
pub use utils::overflow::{get_overflow_patterns, is_context_overflow};
pub use utils::validation::{validate_tool_arguments, validate_tool_call};
