use regex::Regex;

use crate::types::Model;
use crate::{Error, Result};

pub const CLOUDFLARE_WORKERS_AI_BASE_URL: &str =
    "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1";
pub const CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat";
pub const CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai";
pub const CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL: &str = "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic";

pub fn is_cloudflare_provider(provider: &str) -> bool {
    provider == "cloudflare-workers-ai" || provider == "cloudflare-ai-gateway"
}

pub fn resolve_cloudflare_base_url(model: &Model) -> Result<String> {
    let url = &model.base_url;
    if !url.contains('{') {
        return Ok(url.clone());
    }

    let pattern =
        Regex::new(r"\{([A-Z_][A-Z0-9_]*)\}").expect("cloudflare placeholder regex should compile");
    let mut resolved = String::new();
    let mut last_end = 0;
    for capture in pattern.captures_iter(url) {
        let whole_match = capture.get(0).expect("whole placeholder match");
        let name = capture.get(1).expect("placeholder name").as_str();
        resolved.push_str(&url[last_end..whole_match.start()]);
        let value = std::env::var(name)
            .ok()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                Error::Validation(format!(
                    "{} is required for provider {} but is not set.",
                    name, model.provider
                ))
            })?;
        resolved.push_str(&value);
        last_end = whole_match.end();
    }
    resolved.push_str(&url[last_end..]);
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use crate::types::ModelCost;

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

    fn cloudflare_model(base_url: &str) -> Model {
        Model {
            id: "test".to_string(),
            name: "test".to_string(),
            api: "openai-completions".to_string(),
            provider: "cloudflare-ai-gateway".to_string(),
            base_url: base_url.to_string(),
            cost: ModelCost::default(),
            ..Model::default()
        }
    }

    #[test]
    fn identifies_cloudflare_providers() {
        assert!(is_cloudflare_provider("cloudflare-workers-ai"));
        assert!(is_cloudflare_provider("cloudflare-ai-gateway"));
        assert!(!is_cloudflare_provider("openrouter"));
    }

    #[test]
    fn resolves_cloudflare_placeholders_from_env() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let account = SavedEnv::capture("CLOUDFLARE_ACCOUNT_ID");
        let gateway = SavedEnv::capture("CLOUDFLARE_GATEWAY_ID");

        unsafe {
            std::env::set_var("CLOUDFLARE_ACCOUNT_ID", "acct");
            std::env::set_var("CLOUDFLARE_GATEWAY_ID", "gateway");
        }

        let resolved =
            resolve_cloudflare_base_url(&cloudflare_model(CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL))
                .expect("resolved url");
        assert_eq!(
            resolved,
            "https://gateway.ai.cloudflare.com/v1/acct/gateway/compat"
        );

        account.restore();
        gateway.restore();
    }

    #[test]
    fn errors_when_placeholder_env_is_missing() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let account = SavedEnv::capture("CLOUDFLARE_ACCOUNT_ID");
        unsafe {
            std::env::remove_var("CLOUDFLARE_ACCOUNT_ID");
        }

        let error = resolve_cloudflare_base_url(&cloudflare_model(CLOUDFLARE_WORKERS_AI_BASE_URL))
            .expect_err("missing env should fail");
        assert!(
            matches!(error, Error::Validation(message) if message == "CLOUDFLARE_ACCOUNT_ID is required for provider cloudflare-ai-gateway but is not set.")
        );

        account.restore();
    }
}
