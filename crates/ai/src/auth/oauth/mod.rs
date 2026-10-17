//! Port of `packages/ai/src/auth/oauth` for the in-scope providers
//! (Anthropic, GitHub Copilot), plus `oauth.ts` (legacy callback types) in
//! [`compat`]. The other Pi flows (ChatGPT/Codex, Kimi, Meta, OpenRouter,
//! Radius, xAI) are not ported.

pub mod anthropic;
pub mod callback_server;
pub mod compat;
pub mod device_code;
pub mod fetch;
pub mod github_copilot;
pub mod load;
pub mod oauth_page;
pub mod pkce;

pub use anthropic::{AnthropicOAuth, anthropic_oauth, login_anthropic, refresh_anthropic_token};
pub use callback_server::{
    CallbackOrManualInput, OAuthCallbackComplete, OAuthCallbackServer, OAuthCallbackServerOptions,
    start_oauth_callback_server, wait_for_callback_or_manual_input,
};
pub use compat::{
    OAuthAuthInfo, OAuthDeviceCodeInfo, OAuthLoginCallbacks, OAuthLoginCallbacksBuilder,
    OAuthPrompt, OAuthSelectOption, OAuthSelectPrompt, get_oauth_provider, register_oauth_provider,
    unregister_oauth_provider,
};
pub use device_code::{
    OAuthDeviceCodePollOptions, OAuthDeviceCodePollResult, abortable_sleep,
    poll_oauth_device_code_flow,
};
pub use fetch::{
    FetchRequest, FetchResponse, OAuthFetch, default_oauth_fetch, reqwest_oauth_fetch,
};
pub use github_copilot::{
    GitHubCopilotOAuth, get_github_copilot_base_url, github_copilot_base_url_for_credential,
    github_copilot_oauth, login_github_copilot, modify_github_copilot_models, normalize_domain,
    refresh_github_copilot_token,
};
pub use load::{load_anthropic_oauth, load_github_copilot_oauth};
pub use pkce::{Pkce, generate_pkce};
