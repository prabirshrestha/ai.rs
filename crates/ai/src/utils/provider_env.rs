use crate::types::ProviderEnv;

pub fn get_provider_env_value(name: &str, env: &ProviderEnv) -> Option<String> {
    env.get(name)
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| std::env::var(name).ok().filter(|value| !value.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_values_override_process_environment() {
        let _env = crate::test_env::EnvVarGuard::set("AI_RS_PROVIDER_ENV_TEST", "process");
        let overrides = [("AI_RS_PROVIDER_ENV_TEST".to_string(), "scoped".to_string())]
            .into_iter()
            .collect();

        assert_eq!(
            get_provider_env_value("AI_RS_PROVIDER_ENV_TEST", &overrides).as_deref(),
            Some("scoped")
        );
    }
}
