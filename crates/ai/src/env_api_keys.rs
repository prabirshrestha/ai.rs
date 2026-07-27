pub const GITHUB_COPILOT_TOKEN_ENV_VAR: &str = "COPILOT_GITHUB_TOKEN";
pub const ANTHROPIC_AUTH_TOKEN_ENV_VAR: &str = "ANTHROPIC_AUTH_TOKEN";
pub const ANTHROPIC_OAUTH_TOKEN_ENV_VAR: &str = "ANTHROPIC_OAUTH_TOKEN";
pub const ANTHROPIC_API_KEY_ENV_VAR: &str = "ANTHROPIC_API_KEY";
pub const OPENAI_API_KEY_ENV_VAR: &str = "OPENAI_API_KEY";
pub const OPENROUTER_API_KEY_ENV_VAR: &str = "OPENROUTER_API_KEY";

#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnownProvider {
    GitHubCopilot,
    Anthropic,
    OpenAi,
    OpenRouter,
}

impl KnownProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GitHubCopilot => "github-copilot",
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::OpenRouter => "openrouter",
        }
    }
}

impl AsRef<str> for KnownProvider {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<KnownProvider> for String {
    fn from(value: KnownProvider) -> Self {
        value.as_str().to_string()
    }
}

pub(crate) fn get_anthropic_auth_token() -> Option<String> {
    get_anthropic_auth_token_with_env(&Default::default())
}

pub(crate) fn get_anthropic_auth_token_with_env(env: &crate::types::ProviderEnv) -> Option<String> {
    crate::utils::provider_env::get_provider_env_value(ANTHROPIC_AUTH_TOKEN_ENV_VAR, env)
}

pub fn get_env_api_key(provider: impl AsRef<str>) -> Option<String> {
    get_env_api_key_with_env(provider, &Default::default())
}

pub(crate) fn get_env_api_key_with_env(
    provider: impl AsRef<str>,
    env: &crate::types::ProviderEnv,
) -> Option<String> {
    match provider.as_ref() {
        provider if provider == KnownProvider::GitHubCopilot.as_str() => {
            crate::utils::provider_env::get_provider_env_value(GITHUB_COPILOT_TOKEN_ENV_VAR, env)
        }
        provider if provider == KnownProvider::Anthropic.as_str() => {
            crate::utils::provider_env::get_provider_env_value(ANTHROPIC_OAUTH_TOKEN_ENV_VAR, env)
                .or_else(|| {
                    crate::utils::provider_env::get_provider_env_value(
                        ANTHROPIC_API_KEY_ENV_VAR,
                        env,
                    )
                })
        }
        provider if provider == KnownProvider::OpenAi.as_str() => {
            crate::utils::provider_env::get_provider_env_value(OPENAI_API_KEY_ENV_VAR, env)
        }
        provider if provider == KnownProvider::OpenRouter.as_str() => {
            crate::utils::provider_env::get_provider_env_value(OPENROUTER_API_KEY_ENV_VAR, env)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SavedEnv {
        key: &'static str,
        value: Option<String>,
    }

    impl SavedEnv {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                value: std::env::var(key).ok(),
            }
        }

        fn restore(self) {
            unsafe {
                if let Some(value) = self.value {
                    std::env::set_var(self.key, value);
                } else {
                    std::env::remove_var(self.key);
                }
            }
        }
    }

    #[test]
    fn anthropic_oauth_token_precedes_api_key() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let oauth = SavedEnv::capture(ANTHROPIC_OAUTH_TOKEN_ENV_VAR);
        let api_key = SavedEnv::capture(ANTHROPIC_API_KEY_ENV_VAR);

        unsafe {
            std::env::set_var(ANTHROPIC_OAUTH_TOKEN_ENV_VAR, "oauth-token");
            std::env::set_var(ANTHROPIC_API_KEY_ENV_VAR, "api-key");
        }

        assert_eq!(
            get_env_api_key(KnownProvider::Anthropic).as_deref(),
            Some("oauth-token")
        );

        oauth.restore();
        api_key.restore();
    }

    #[test]
    fn anthropic_auth_token_is_distinct_from_api_key_lookup() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let auth = SavedEnv::capture(ANTHROPIC_AUTH_TOKEN_ENV_VAR);
        let oauth = SavedEnv::capture(ANTHROPIC_OAUTH_TOKEN_ENV_VAR);
        let api_key = SavedEnv::capture(ANTHROPIC_API_KEY_ENV_VAR);

        unsafe {
            std::env::set_var(ANTHROPIC_AUTH_TOKEN_ENV_VAR, "auth-token");
            std::env::set_var(ANTHROPIC_OAUTH_TOKEN_ENV_VAR, "oauth-token");
            std::env::set_var(ANTHROPIC_API_KEY_ENV_VAR, "api-key");
        }

        assert_eq!(get_anthropic_auth_token().as_deref(), Some("auth-token"));
        assert_eq!(
            get_env_api_key(KnownProvider::Anthropic).as_deref(),
            Some("oauth-token")
        );

        auth.restore();
        oauth.restore();
        api_key.restore();
    }

    #[test]
    fn scoped_anthropic_auth_token_takes_precedence_over_process_env() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let auth = SavedEnv::capture(ANTHROPIC_AUTH_TOKEN_ENV_VAR);
        unsafe {
            std::env::set_var(ANTHROPIC_AUTH_TOKEN_ENV_VAR, "process-auth-token");
        }
        let env = [(
            ANTHROPIC_AUTH_TOKEN_ENV_VAR.to_string(),
            "scoped-auth-token".to_string(),
        )]
        .into_iter()
        .collect();

        assert_eq!(
            get_anthropic_auth_token_with_env(&env).as_deref(),
            Some("scoped-auth-token")
        );

        auth.restore();
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let openai = SavedEnv::capture(OPENAI_API_KEY_ENV_VAR);

        unsafe {
            std::env::set_var(OPENAI_API_KEY_ENV_VAR, "");
        }

        assert_eq!(get_env_api_key(KnownProvider::OpenAi), None);

        openai.restore();
    }

    fn with_saved_github_env(test: impl FnOnce()) {
        let copilot = SavedEnv::capture(GITHUB_COPILOT_TOKEN_ENV_VAR);
        let gh = SavedEnv::capture("GH_TOKEN");
        let github = SavedEnv::capture("GITHUB_TOKEN");

        test();

        copilot.restore();
        gh.restore();
        github.restore();
    }

    #[test]
    fn does_not_treat_generic_github_tokens_as_github_copilot_credentials() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        with_saved_github_env(|| {
            unsafe {
                std::env::remove_var(GITHUB_COPILOT_TOKEN_ENV_VAR);
                std::env::set_var("GH_TOKEN", "gh-token");
                std::env::set_var("GITHUB_TOKEN", "github-token");
            }

            assert_eq!(get_env_api_key(KnownProvider::GitHubCopilot), None);
        });
    }

    #[test]
    fn resolves_github_copilot_credentials_from_copilot_github_token() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        with_saved_github_env(|| {
            unsafe {
                std::env::set_var(GITHUB_COPILOT_TOKEN_ENV_VAR, "copilot-token");
                std::env::set_var("GH_TOKEN", "gh-token");
                std::env::set_var("GITHUB_TOKEN", "github-token");
            }

            assert_eq!(
                get_env_api_key(KnownProvider::GitHubCopilot).as_deref(),
                Some("copilot-token")
            );
        });
    }

    #[test]
    fn accepts_custom_provider_strings() {
        assert_eq!(get_env_api_key("custom-provider"), None);
    }

    #[test]
    fn scoped_env_takes_precedence_over_process_env() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let openai = SavedEnv::capture(OPENAI_API_KEY_ENV_VAR);
        unsafe {
            std::env::set_var(OPENAI_API_KEY_ENV_VAR, "process-key");
        }
        let env = [(OPENAI_API_KEY_ENV_VAR.to_string(), "scoped-key".to_string())]
            .into_iter()
            .collect();

        assert_eq!(
            get_env_api_key_with_env(KnownProvider::OpenAi, &env).as_deref(),
            Some("scoped-key")
        );

        openai.restore();
    }
}
