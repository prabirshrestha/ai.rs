use crate::types::ProviderEnv;

pub(crate) fn get_provider_env_value(name: &str, env: &ProviderEnv) -> Option<String> {
    env.get(name)
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| std::env::var(name).ok().filter(|value| !value.is_empty()))
}
