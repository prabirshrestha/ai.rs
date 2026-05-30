pub fn api_key_env_vars(provider: &str) -> Option<&'static [&'static str]> {
    match provider {
        "github-copilot" => Some(&["COPILOT_GITHUB_TOKEN"]),
        "anthropic" => Some(&["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]),
        "openai" => Some(&["OPENAI_API_KEY"]),
        "azure-openai-responses" => Some(&["AZURE_OPENAI_API_KEY"]),
        "deepseek" => Some(&["DEEPSEEK_API_KEY"]),
        "google" => Some(&["GEMINI_API_KEY"]),
        "google-vertex" => Some(&["GOOGLE_CLOUD_API_KEY"]),
        "groq" => Some(&["GROQ_API_KEY"]),
        "cerebras" => Some(&["CEREBRAS_API_KEY"]),
        "xai" => Some(&["XAI_API_KEY"]),
        "openrouter" => Some(&["OPENROUTER_API_KEY"]),
        "vercel-ai-gateway" => Some(&["AI_GATEWAY_API_KEY"]),
        "zai" => Some(&["ZAI_API_KEY"]),
        "mistral" => Some(&["MISTRAL_API_KEY"]),
        "minimax" => Some(&["MINIMAX_API_KEY"]),
        "minimax-cn" => Some(&["MINIMAX_CN_API_KEY"]),
        "moonshotai" | "moonshotai-cn" => Some(&["MOONSHOT_API_KEY"]),
        "huggingface" => Some(&["HF_TOKEN"]),
        "fireworks" => Some(&["FIREWORKS_API_KEY"]),
        "together" => Some(&["TOGETHER_API_KEY"]),
        "opencode" | "opencode-go" => Some(&["OPENCODE_API_KEY"]),
        "kimi-coding" => Some(&["KIMI_API_KEY"]),
        "cloudflare-workers-ai" | "cloudflare-ai-gateway" => Some(&["CLOUDFLARE_API_KEY"]),
        "xiaomi" => Some(&["XIAOMI_API_KEY"]),
        "xiaomi-token-plan-cn" => Some(&["XIAOMI_TOKEN_PLAN_CN_API_KEY"]),
        "xiaomi-token-plan-ams" => Some(&["XIAOMI_TOKEN_PLAN_AMS_API_KEY"]),
        "xiaomi-token-plan-sgp" => Some(&["XIAOMI_TOKEN_PLAN_SGP_API_KEY"]),
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

    if provider == "google-vertex" {
        let has_project = env_value("GOOGLE_CLOUD_PROJECT")
            .or_else(|| env_value("GCLOUD_PROJECT"))
            .is_some();
        let has_location = env_value("GOOGLE_CLOUD_LOCATION").is_some();
        if has_project && has_location && has_vertex_adc_credentials() {
            return Some("<authenticated>".to_string());
        }
    }

    if provider == "amazon-bedrock" {
        let has_iam_pair = env_value("AWS_ACCESS_KEY_ID").is_some()
            && env_value("AWS_SECRET_ACCESS_KEY").is_some();
        let has_bedrock_credentials = [
            "AWS_PROFILE",
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        ]
        .iter()
        .any(|key| env_value(key).is_some());
        if has_iam_pair || has_bedrock_credentials {
            return Some("<authenticated>".to_string());
        }
    }

    None
}

fn env_value(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

fn has_vertex_adc_credentials() -> bool {
    if let Some(path) = env_value("GOOGLE_APPLICATION_CREDENTIALS") {
        return std::path::Path::new(&path).exists();
    }

    let Some(home) = env_value("HOME") else {
        return false;
    };
    std::path::Path::new(&home)
        .join(".config")
        .join("gcloud")
        .join("application_default_credentials.json")
        .exists()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());
    static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

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

    #[test]
    fn google_vertex_uses_default_adc_file_when_project_and_location_exist() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let home = SavedEnv::capture("HOME");
        let gac = SavedEnv::capture("GOOGLE_APPLICATION_CREDENTIALS");
        let project = SavedEnv::capture("GOOGLE_CLOUD_PROJECT");
        let gcloud_project = SavedEnv::capture("GCLOUD_PROJECT");
        let location = SavedEnv::capture("GOOGLE_CLOUD_LOCATION");

        let root = std::env::temp_dir().join(format!(
            "ai-rs-vertex-adc-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let adc_dir = root.join(".config").join("gcloud");
        std::fs::create_dir_all(&adc_dir).unwrap();
        std::fs::write(adc_dir.join("application_default_credentials.json"), "{}").unwrap();

        unsafe {
            std::env::set_var("HOME", &root);
            std::env::remove_var("GOOGLE_APPLICATION_CREDENTIALS");
            std::env::set_var("GOOGLE_CLOUD_PROJECT", "project-1");
            std::env::remove_var("GCLOUD_PROJECT");
            std::env::set_var("GOOGLE_CLOUD_LOCATION", "us-central1");
        }

        assert_eq!(
            get_env_api_key("google-vertex").as_deref(),
            Some("<authenticated>")
        );

        let _ = std::fs::remove_dir_all(root);
        home.restore();
        gac.restore();
        project.restore();
        gcloud_project.restore();
        location.restore();
    }

    #[test]
    fn bedrock_iam_pair_requires_non_empty_values() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let access_key = SavedEnv::capture("AWS_ACCESS_KEY_ID");
        let secret_key = SavedEnv::capture("AWS_SECRET_ACCESS_KEY");
        let profile = SavedEnv::capture("AWS_PROFILE");
        let bearer = SavedEnv::capture("AWS_BEARER_TOKEN_BEDROCK");
        let relative_uri = SavedEnv::capture("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
        let full_uri = SavedEnv::capture("AWS_CONTAINER_CREDENTIALS_FULL_URI");
        let web_identity = SavedEnv::capture("AWS_WEB_IDENTITY_TOKEN_FILE");

        unsafe {
            std::env::set_var("AWS_ACCESS_KEY_ID", "");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "");
            std::env::remove_var("AWS_PROFILE");
            std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK");
            std::env::remove_var("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
            std::env::remove_var("AWS_CONTAINER_CREDENTIALS_FULL_URI");
            std::env::remove_var("AWS_WEB_IDENTITY_TOKEN_FILE");
        }

        assert_eq!(get_env_api_key("amazon-bedrock"), None);

        access_key.restore();
        secret_key.restore();
        profile.restore();
        bearer.restore();
        relative_uri.restore();
        full_uri.restore();
        web_identity.restore();
    }
}
