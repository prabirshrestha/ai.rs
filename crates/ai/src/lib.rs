#![doc = include_str!("../README.md")]

// Compile-checks the snippets of the workspace README as doctests.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
struct WorkspaceReadmeDoctests;

pub mod agent;
pub mod api;
pub mod auth;
#[cfg(feature = "durable")]
pub mod chord;
#[cfg(feature = "durable")]
pub mod durable;
pub mod embeddings;
pub mod env_api_keys;
pub mod error;
pub mod model_catalog;
pub mod models;
pub mod models_store;
pub mod providers;
pub mod types;
pub mod utils;

pub use agent::types::*;
pub use agent::{
    Agent, AgentError, AgentEventStream, AgentInitialState, AgentOptions, AgentOptionsBuilder,
    AgentPrepareNextTurnFn, AgentPrepareNextTurnWithContextFn, AgentResult, AgentSubscription,
    ProxyAssistantMessageEvent, ProxyStreamOptions, RunToolCallOptions, ToolCallHooks,
    ToolUpdateCallback, agent_loop, agent_loop_continue, get_default_stream_fn, run_agent_loop,
    run_agent_loop_continue, run_tool_call, set_default_stream_fn, stream_proxy, stream_simple_fn,
};
pub use api::lazy::lazy_stream;
pub use api::openai_completions::{
    OpenAICompletionsOptions, stream_openai_completions, stream_simple_openai_completions,
};
pub use api::openai_responses::{
    OpenAIResponsesOptions, stream_openai_responses, stream_simple_openai_responses,
};
pub use auth::oauth::{
    OAuthAuthInfo, OAuthDeviceCodeInfo, OAuthLoginCallbacks, OAuthLoginCallbacksBuilder,
    OAuthPrompt, OAuthSelectOption, OAuthSelectPrompt, anthropic_oauth, get_oauth_provider,
    github_copilot_oauth, login_anthropic, login_github_copilot, modify_github_copilot_models,
    refresh_anthropic_token, refresh_github_copilot_token, register_oauth_provider,
    unregister_oauth_provider,
};
pub use auth::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthCheck, AuthContext, AuthEvent, AuthInfoLink,
    AuthInteraction, AuthOperationOptions, AuthPrompt, AuthPromptKind, AuthResolutionOverrides,
    AuthResult, AuthSelectOption, AuthType, Credential, CredentialInfo, CredentialModifier,
    CredentialStore, InMemoryCredentialStore, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential,
    OAuthCredentials, ProviderAuth, ProviderAuthInteraction, default_provider_auth_context,
    env_api_key_auth,
};
pub use embeddings::{
    Embedding, EmbeddingBatch, EmbeddingEncodingFormat, EmbeddingModel, EmbeddingModelBuilder,
    EmbeddingOptions, EmbeddingUsage, EmbeddingVector, embed, embed_many,
};
pub use env_api_keys::{find_env_keys, get_env_api_key};
pub use error::{Error, Result};
pub use models::*;
pub use models_store::*;
pub use providers::anthropic::Anthropic;
pub use providers::faux::{
    FauxContent, FauxContentBlock, FauxCore, FauxDeferredOptions, FauxMessageOptions,
    FauxModelDefinition, FauxProviderHandle, FauxProviderState, FauxResponseFactory,
    FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions, create_faux_core,
    faux_assistant_message, faux_provider, faux_text, faux_thinking, faux_tool_call,
};
pub use providers::github_copilot::{GitHubCopilot, GitHubCopilotApi};
pub use providers::openai::{OpenAi, OpenAiApi};
pub use providers::openrouter::OpenRouter;
pub use providers::{ImageModelBuilder, ModelBuilder};
pub use types::*;
pub use utils::assistant_message_frame::*;
pub use utils::diagnostics::*;
pub use utils::event_stream::*;
pub use utils::json_parse::*;
pub use utils::overflow::*;
pub use utils::retry::*;
pub use utils::text::{content_text, get_system_message_text, render_system_message_update};
pub use utils::transcript::*;
pub use utils::uuid::uuidv7;
pub use utils::validation::*;
