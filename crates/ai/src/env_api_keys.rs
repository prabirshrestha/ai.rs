//! Port of `env-api-keys.ts`.

use std::path::Path;

use parking_lot::Mutex;

use crate::auth::context::home_dir;
use crate::types::ProviderEnv;
use crate::utils::provider_env::get_provider_env_value;

pub const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
pub const ANTHROPIC_OAUTH_TOKEN_ENV: &str = "ANTHROPIC_OAUTH_TOKEN";
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_FEDERATION_RULE_ID_ENV: &str = "ANTHROPIC_FEDERATION_RULE_ID";
pub const ANTHROPIC_ORGANIZATION_ID_ENV: &str = "ANTHROPIC_ORGANIZATION_ID";
pub const ANTHROPIC_SERVICE_ACCOUNT_ID_ENV: &str = "ANTHROPIC_SERVICE_ACCOUNT_ID";
pub const ANTHROPIC_IDENTITY_TOKEN_FILE_ENV: &str = "ANTHROPIC_IDENTITY_TOKEN_FILE";
pub const ANTHROPIC_WORKSPACE_ID_ENV: &str = "ANTHROPIC_WORKSPACE_ID";

static CACHED_VERTEX_ADC_CREDENTIALS_EXISTS: Mutex<Option<bool>> = Mutex::new(None);

fn has_vertex_adc_credentials(env: Option<&ProviderEnv>) -> bool {
    if let Some(explicit_credentials_path) = env
        .and_then(|env| env.get("GOOGLE_APPLICATION_CREDENTIALS"))
        .filter(|path| !path.is_empty())
    {
        return Path::new(explicit_credentials_path).exists();
    }

    let mut cached = CACHED_VERTEX_ADC_CREDENTIALS_EXISTS.lock();
    if let Some(exists) = *cached {
        return exists;
    }
    // Check GOOGLE_APPLICATION_CREDENTIALS env var first (standard way)
    let exists = match get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env) {
        Some(gac_path) => Path::new(&gac_path).exists(),
        // Fall back to default ADC path (lazy evaluation)
        None => home_dir().is_some_and(|home| {
            Path::new(&home)
                .join(".config")
                .join("gcloud")
                .join("application_default_credentials.json")
                .exists()
        }),
    };
    *cached = Some(exists);
    exists
}

fn get_api_key_env_vars(provider: &str) -> Option<&'static [&'static str]> {
    if provider == "github-copilot" {
        return Some(&["COPILOT_GITHUB_TOKEN"]);
    }

    // ANTHROPIC_AUTH_TOKEN participates in env discovery/status, but
    // get_env_api_key() skips it because requests must pass it as Authorization: Bearer.
    if provider == "anthropic" {
        return Some(&[
            ANTHROPIC_AUTH_TOKEN_ENV,
            ANTHROPIC_OAUTH_TOKEN_ENV,
            ANTHROPIC_API_KEY_ENV,
        ]);
    }

    let env_var: &'static [&'static str] = match provider {
        "ant-ling" => &["ANT_LING_API_KEY"],
        "qwen-token-plan" => &["QWEN_TOKEN_PLAN_API_KEY"],
        "qwen-token-plan-cn" => &["QWEN_TOKEN_PLAN_CN_API_KEY"],
        "qwen-token-plan-individual" => &["QWEN_TOKEN_PLAN_API_KEY"],
        "openai" => &["OPENAI_API_KEY"],
        "azure-openai-responses" => &["AZURE_OPENAI_API_KEY"],
        "nvidia" => &["NVIDIA_API_KEY"],
        "deepseek" => &["DEEPSEEK_API_KEY"],
        "google" => &["GEMINI_API_KEY"],
        "google-vertex" => &["GOOGLE_CLOUD_API_KEY"],
        "groq" => &["GROQ_API_KEY"],
        "cerebras" => &["CEREBRAS_API_KEY"],
        "xai" => &["XAI_API_KEY"],
        "typesafe" => &["TYPESAFE_API_KEY"],
        "radius" => &["RADIUS_API_KEY"],
        "openrouter" => &["OPENROUTER_API_KEY"],
        "vercel-ai-gateway" => &["AI_GATEWAY_API_KEY"],
        "zai" => &["ZAI_API_KEY"],
        "zai-coding-cn" => &["ZAI_CODING_CN_API_KEY"],
        "mistral" => &["MISTRAL_API_KEY"],
        "minimax" => &["MINIMAX_API_KEY"],
        "minimax-cn" => &["MINIMAX_CN_API_KEY"],
        "moonshotai" => &["MOONSHOT_API_KEY"],
        "moonshotai-cn" => &["MOONSHOT_API_KEY"],
        "huggingface" => &["HF_TOKEN"],
        "fireworks" => &["FIREWORKS_API_KEY"],
        "together" => &["TOGETHER_API_KEY"],
        "baseten" => &["BASETEN_API_KEY"],
        "opencode" => &["OPENCODE_API_KEY"],
        "opencode-go" => &["OPENCODE_API_KEY"],
        "kimi-coding" => &["KIMI_API_KEY"],
        "meta" => &["META_API_KEY"],
        "cloudflare-workers-ai" => &["CLOUDFLARE_API_KEY"],
        "cloudflare-ai-gateway" => &["CLOUDFLARE_API_KEY"],
        "xiaomi" => &["XIAOMI_API_KEY"],
        "xiaomi-token-plan-cn" => &["XIAOMI_TOKEN_PLAN_CN_API_KEY"],
        "xiaomi-token-plan-ams" => &["XIAOMI_TOKEN_PLAN_AMS_API_KEY"],
        "xiaomi-token-plan-sgp" => &["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
        _ => return None,
    };
    Some(env_var)
}

/// Find configured environment variables that can provide an API key for a provider.
///
/// This only reports actual API key variables. It intentionally excludes ambient
/// credential sources such as AWS profiles, AWS IAM credentials, and Google
/// Application Default Credentials.
pub fn find_env_keys(provider: &str, env: Option<&ProviderEnv>) -> Option<Vec<String>> {
    let env_vars = get_api_key_env_vars(provider)?;
    let found: Vec<String> = env_vars
        .iter()
        .filter(|env_var| get_provider_env_value(env_var, env).is_some())
        .map(|env_var| env_var.to_string())
        .collect();
    (!found.is_empty()).then_some(found)
}

/// Get API key for provider from known environment variables, e.g. OPENAI_API_KEY.
///
/// Will not return API keys for providers that require OAuth tokens.
pub fn get_env_api_key(provider: &str, env: Option<&ProviderEnv>) -> Option<String> {
    if let Some(env_keys) = find_env_keys(provider, env) {
        let api_key_env = if provider == "anthropic" {
            env_keys
                .iter()
                .find(|key| key.as_str() != ANTHROPIC_AUTH_TOKEN_ENV)
        } else {
            env_keys.first()
        };
        if let Some(api_key_env) = api_key_env {
            return get_provider_env_value(api_key_env, env);
        }
    }

    // Vertex AI supports either an explicit API key or Application Default Credentials.
    // Auth is configured via `gcloud auth application-default login`.
    if provider == "google-vertex" {
        let has_credentials = has_vertex_adc_credentials(env);
        let has_project = get_provider_env_value("GOOGLE_CLOUD_PROJECT", env).is_some()
            || get_provider_env_value("GCLOUD_PROJECT", env).is_some();
        let has_location = get_provider_env_value("GOOGLE_CLOUD_LOCATION", env).is_some();

        if has_credentials && has_project && has_location {
            return Some("<authenticated>".to_string());
        }
    }

    if provider == "amazon-bedrock" {
        // Amazon Bedrock supports multiple credential sources:
        // 1. AWS_PROFILE - named profile from ~/.aws/credentials
        // 2. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY - standard IAM keys
        // 3. AWS_BEARER_TOKEN_BEDROCK - Bedrock bearer token
        // 4. AWS_CONTAINER_CREDENTIALS_RELATIVE_URI - ECS task roles
        // 5. AWS_CONTAINER_CREDENTIALS_FULL_URI - ECS task roles (full URI)
        // 6. AWS_WEB_IDENTITY_TOKEN_FILE - IRSA (IAM Roles for Service Accounts)
        let has = |name: &str| get_provider_env_value(name, env).is_some();
        if has("AWS_PROFILE")
            || (has("AWS_ACCESS_KEY_ID") && has("AWS_SECRET_ACCESS_KEY"))
            || has("AWS_BEARER_TOKEN_BEDROCK")
            || has("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            || has("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            || has("AWS_WEB_IDENTITY_TOKEN_FILE")
        {
            return Some("<authenticated>".to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    //! Pi mutates `process.env`; these tests pass the same values through the
    //! provider-scoped `env` overlay so they do not race other tests.

    use super::*;

    fn env(entries: &[(&str, &str)]) -> ProviderEnv {
        entries
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn unset_in_process(names: &[&str]) -> bool {
        names.iter().all(|name| std::env::var(name).is_err())
    }

    #[test]
    fn does_not_treat_generic_github_tokens_as_github_copilot_credentials() {
        if !unset_in_process(&["COPILOT_GITHUB_TOKEN"]) {
            return;
        }
        let env = env(&[("GH_TOKEN", "gh-token"), ("GITHUB_TOKEN", "github-token")]);
        assert_eq!(find_env_keys("github-copilot", Some(&env)), None);
        assert_eq!(get_env_api_key("github-copilot", Some(&env)), None);
    }

    #[test]
    fn resolves_github_copilot_credentials_from_copilot_github_token() {
        let env = env(&[
            ("COPILOT_GITHUB_TOKEN", "copilot-token"),
            ("GH_TOKEN", "gh-token"),
            ("GITHUB_TOKEN", "github-token"),
        ]);
        assert_eq!(
            find_env_keys("github-copilot", Some(&env)),
            Some(vec!["COPILOT_GITHUB_TOKEN".to_string()])
        );
        assert_eq!(
            get_env_api_key("github-copilot", Some(&env)).as_deref(),
            Some("copilot-token")
        );
    }

    #[test]
    fn resolves_zai_china_coding_plan_credentials() {
        let env = env(&[("ZAI_CODING_CN_API_KEY", "zai-coding-cn-token")]);
        assert_eq!(
            find_env_keys("zai-coding-cn", Some(&env)),
            Some(vec!["ZAI_CODING_CN_API_KEY".to_string()])
        );
        assert_eq!(
            get_env_api_key("zai-coding-cn", Some(&env)).as_deref(),
            Some("zai-coding-cn-token")
        );
    }

    #[test]
    fn reports_anthropic_auth_token_but_preserves_oauth_token_api_key_lookup() {
        let env = env(&[
            ("ANTHROPIC_AUTH_TOKEN", "auth-token"),
            ("ANTHROPIC_OAUTH_TOKEN", "oauth-token"),
            ("ANTHROPIC_API_KEY", "api-key"),
        ]);
        assert_eq!(
            find_env_keys("anthropic", Some(&env)),
            Some(vec![
                "ANTHROPIC_AUTH_TOKEN".to_string(),
                "ANTHROPIC_OAUTH_TOKEN".to_string(),
                "ANTHROPIC_API_KEY".to_string()
            ])
        );
        assert_eq!(
            get_env_api_key("anthropic", Some(&env)).as_deref(),
            Some("oauth-token")
        );
    }

    #[test]
    fn does_not_return_anthropic_auth_token_as_an_api_key() {
        if !unset_in_process(&["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]) {
            return;
        }
        let env = env(&[("ANTHROPIC_AUTH_TOKEN", "auth-token")]);
        assert_eq!(
            find_env_keys("anthropic", Some(&env)),
            Some(vec!["ANTHROPIC_AUTH_TOKEN".to_string()])
        );
        assert_eq!(get_env_api_key("anthropic", Some(&env)), None);
    }

    #[test]
    fn preserves_anthropic_oauth_token_as_an_api_key() {
        if !unset_in_process(&["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"]) {
            return;
        }
        let env = env(&[("ANTHROPIC_OAUTH_TOKEN", "oauth-token")]);
        assert_eq!(
            find_env_keys("anthropic", Some(&env)),
            Some(vec!["ANTHROPIC_OAUTH_TOKEN".to_string()])
        );
        assert_eq!(
            get_env_api_key("anthropic", Some(&env)).as_deref(),
            Some("oauth-token")
        );
    }

    #[test]
    fn falls_back_to_anthropic_api_key_for_api_key_lookup() {
        if !unset_in_process(&["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN"]) {
            return;
        }
        let env = env(&[("ANTHROPIC_API_KEY", "api-key")]);
        assert_eq!(
            get_env_api_key("anthropic", Some(&env)).as_deref(),
            Some("api-key")
        );
    }

    #[test]
    fn detects_ambient_bedrock_credentials() {
        let env = env(&[
            ("AWS_ACCESS_KEY_ID", "id"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
        ]);
        assert_eq!(
            get_env_api_key("amazon-bedrock", Some(&env)).as_deref(),
            Some("<authenticated>")
        );
        assert_eq!(find_env_keys("unknown-provider", Some(&env)), None);
    }
}
