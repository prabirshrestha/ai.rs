//! Port of `packages/ai/src/auth`: provider auth types, credential storage
//! and auth resolution. OAuth flows (`auth/oauth`) land in a later commit.

pub mod context;
pub mod credential_store;
pub mod helpers;
pub mod resolve;
pub mod types;

pub use context::{DefaultProviderAuthContext, default_provider_auth_context};
pub use credential_store::InMemoryCredentialStore;
pub use helpers::env_api_key_auth;
pub use resolve::{AuthResolutionOverrides, resolve_provider_auth};
pub use types::*;
