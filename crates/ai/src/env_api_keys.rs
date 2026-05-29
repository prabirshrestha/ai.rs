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
        let has_project = std::env::var("GOOGLE_CLOUD_PROJECT")
            .or_else(|_| std::env::var("GCLOUD_PROJECT"))
            .ok()
            .is_some_and(|value| !value.is_empty());
        let has_location = std::env::var("GOOGLE_CLOUD_LOCATION")
            .ok()
            .is_some_and(|value| !value.is_empty());
        let has_adc_path = std::env::var("GOOGLE_APPLICATION_CREDENTIALS")
            .ok()
            .is_some_and(|path| std::path::Path::new(&path).exists());
        if has_project && has_location && has_adc_path {
            return Some("<authenticated>".to_string());
        }
    }

    if provider == "amazon-bedrock" {
        let has_iam_pair = std::env::var("AWS_ACCESS_KEY_ID").ok().is_some()
            && std::env::var("AWS_SECRET_ACCESS_KEY").ok().is_some();
        let has_bedrock_credentials = [
            "AWS_PROFILE",
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        ]
        .iter()
        .any(|key| {
            std::env::var(key)
                .ok()
                .is_some_and(|value| !value.is_empty())
        });
        if has_iam_pair || has_bedrock_credentials {
            return Some("<authenticated>".to_string());
        }
    }

    None
}
