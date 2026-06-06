pub const GITHUB_COPILOT_TOKEN_ENV_VAR: &str = "COPILOT_GITHUB_TOKEN";
pub const ANTHROPIC_OAUTH_TOKEN_ENV_VAR: &str = "ANTHROPIC_OAUTH_TOKEN";
pub const ANTHROPIC_API_KEY_ENV_VAR: &str = "ANTHROPIC_API_KEY";
pub const OPENAI_API_KEY_ENV_VAR: &str = "OPENAI_API_KEY";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnownProvider {
    GitHubCopilot,
    Anthropic,
    OpenAi,
}

impl KnownProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GitHubCopilot => "github-copilot",
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
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

fn env_value(env_var: &str) -> Option<String> {
    std::env::var(env_var)
        .ok()
        .filter(|value| !value.is_empty())
}

pub fn get_env_api_key(provider: impl AsRef<str>) -> Option<String> {
    match provider.as_ref() {
        provider if provider == KnownProvider::GitHubCopilot.as_str() => {
            env_value(GITHUB_COPILOT_TOKEN_ENV_VAR)
        }
        provider if provider == KnownProvider::Anthropic.as_str() => {
            env_value(ANTHROPIC_OAUTH_TOKEN_ENV_VAR)
                .or_else(|| env_value(ANTHROPIC_API_KEY_ENV_VAR))
        }
        provider if provider == KnownProvider::OpenAi.as_str() => env_value(OPENAI_API_KEY_ENV_VAR),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
}
