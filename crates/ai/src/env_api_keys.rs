fn api_key_env_vars(provider: &str) -> Option<&'static [&'static str]> {
    match provider {
        "github-copilot" => Some(&["COPILOT_GITHUB_TOKEN"]),
        "anthropic" => Some(&["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]),
        "openai" => Some(&["OPENAI_API_KEY"]),
        _ => None,
    }
}

pub fn find_env_keys(provider: &str) -> Option<Vec<String>> {
    let found = api_key_env_vars(provider)?
        .iter()
        .filter(|env_var| {
            std::env::var(env_var)
                .ok()
                .is_some_and(|value| !value.is_empty())
        })
        .map(|env_var| (*env_var).to_string())
        .collect::<Vec<_>>();
    (!found.is_empty()).then_some(found)
}

pub fn get_env_api_key(provider: &str) -> Option<String> {
    if let Some(env_key) = find_env_keys(provider).and_then(|keys| keys.into_iter().next()) {
        return std::env::var(env_key).ok();
    }

    None
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
        let oauth = SavedEnv::capture("ANTHROPIC_OAUTH_TOKEN");
        let api_key = SavedEnv::capture("ANTHROPIC_API_KEY");

        unsafe {
            std::env::set_var("ANTHROPIC_OAUTH_TOKEN", "oauth-token");
            std::env::set_var("ANTHROPIC_API_KEY", "api-key");
        }

        assert_eq!(
            find_env_keys("anthropic"),
            Some(vec![
                "ANTHROPIC_OAUTH_TOKEN".to_string(),
                "ANTHROPIC_API_KEY".to_string()
            ])
        );
        assert_eq!(get_env_api_key("anthropic").as_deref(), Some("oauth-token"));

        oauth.restore();
        api_key.restore();
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let openai = SavedEnv::capture("OPENAI_API_KEY");

        unsafe {
            std::env::set_var("OPENAI_API_KEY", "");
        }

        assert_eq!(find_env_keys("openai"), None);
        assert_eq!(get_env_api_key("openai"), None);

        openai.restore();
    }

    #[test]
    fn github_copilot_only_uses_copilot_token_env() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let copilot = SavedEnv::capture("COPILOT_GITHUB_TOKEN");
        let gh = SavedEnv::capture("GH_TOKEN");
        let github = SavedEnv::capture("GITHUB_TOKEN");

        unsafe {
            std::env::remove_var("COPILOT_GITHUB_TOKEN");
            std::env::set_var("GH_TOKEN", "gh-token");
            std::env::set_var("GITHUB_TOKEN", "github-token");
        }
        assert_eq!(find_env_keys("github-copilot"), None);
        assert_eq!(get_env_api_key("github-copilot"), None);

        unsafe {
            std::env::set_var("COPILOT_GITHUB_TOKEN", "copilot-token");
        }
        assert_eq!(
            find_env_keys("github-copilot"),
            Some(vec!["COPILOT_GITHUB_TOKEN".to_string()])
        );
        assert_eq!(
            get_env_api_key("github-copilot").as_deref(),
            Some("copilot-token")
        );

        copilot.restore();
        gh.restore();
        github.restore();
    }
}
