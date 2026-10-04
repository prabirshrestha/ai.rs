//! Rust port of Pi's `@earendil-works/pi-ai` 1.0 (`packages/ai`).
//!
//! The crate root mirrors Pi's `index.ts` (core types, the `Models` runtime,
//! auth, and the side-effect free utilities) plus the `compat` entry points
//! (`stream()`, `stream_simple()`, ...) that keep ai.rs's pre-1.0 call shape.
//! Divergences from Pi are documented on the items involved.

pub mod api;
pub mod auth;
pub mod compat;
pub mod env_api_keys;
pub mod error;
pub mod model_catalog;
pub mod models;
pub mod models_store;
pub mod providers;
pub mod types;
pub mod utils;

pub use api::lazy::lazy_stream;
pub use auth::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, AuthCheck, AuthContext, AuthEvent, AuthInfoLink,
    AuthInteraction, AuthOperationOptions, AuthPrompt, AuthPromptKind, AuthResolutionOverrides,
    AuthResult, AuthSelectOption, AuthType, Credential, CredentialInfo, CredentialModifier,
    CredentialStore, InMemoryCredentialStore, LoginOptions, ModelAuth, OAuthAuth, OAuthCredential,
    OAuthCredentials, ProviderAuth, ProviderAuthInteraction, default_provider_auth_context,
    env_api_key_auth,
};
pub use compat::{complete, complete_simple, register_faux_provider, stream, stream_simple};
pub use env_api_keys::{find_env_keys, get_env_api_key};
pub use error::{Error, Result};
pub use models::*;
pub use models_store::*;
pub use providers::ModelBuilder;
pub use providers::anthropic::Anthropic;
pub use providers::faux::{
    FauxContent, FauxContentBlock, FauxCore, FauxDeferredOptions, FauxMessageOptions,
    FauxModelDefinition, FauxProviderHandle, FauxProviderRegistration, FauxProviderState,
    FauxResponseFactory, FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions,
    create_faux_core, faux_assistant_message, faux_provider, faux_text, faux_thinking,
    faux_tool_call,
};
pub use providers::github_copilot::{GitHubCopilot, GitHubCopilotApi};
pub use providers::openai::{OpenAi, OpenAiApi};
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
