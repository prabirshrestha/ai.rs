//! Port of `auth/oauth/load.ts`.
//!
//! Pi loads each flow through a bundler-opaque dynamic import so browser
//! bundles never see Node-only flow code; Rust links the flows statically, so
//! the loaders just build them. `registerBundledOAuthFlowLoaders()` (for
//! standalone Bun binaries) has no Rust counterpart. Only the in-scope flows
//! (Anthropic, GitHub Copilot) are ported.

use std::sync::Arc;

use crate::Result;
use crate::auth::types::OAuthAuth;

/// `loadAnthropicOAuth()`.
pub async fn load_anthropic_oauth() -> Result<Arc<dyn OAuthAuth>> {
    Ok(super::anthropic::anthropic_oauth())
}

/// `loadGitHubCopilotOAuth()`.
pub async fn load_github_copilot_oauth() -> Result<Arc<dyn OAuthAuth>> {
    Ok(super::github_copilot::github_copilot_oauth())
}
