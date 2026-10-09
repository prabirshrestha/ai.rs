//! Port of `packages/ai/src/auth`: provider auth types, credential storage,
//! auth resolution and the OAuth flows (`auth/oauth`).

pub mod context;
pub mod credential_store;
pub mod helpers;
pub mod oauth;
pub mod resolve;
pub mod types;

pub use context::{DefaultProviderAuthContext, default_provider_auth_context};
pub use credential_store::InMemoryCredentialStore;
pub use helpers::{LazyOAuthInput, OAuthLoader, env_api_key_auth, lazy_oauth};
pub use resolve::{AuthResolutionOverrides, resolve_provider_auth};
pub use types::*;
