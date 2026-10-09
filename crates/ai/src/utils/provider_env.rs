//! Port of `utils/provider-env.ts`.

use crate::types::ProviderEnv;

/// Resolve a provider env value from scoped overrides, then the process
/// environment. Empty values count as unset (JavaScript `||`). Pi's Bun
/// sandbox `/proc/self/environ` fallback does not apply to Rust.
pub fn get_provider_env_value(name: &str, env: Option<&ProviderEnv>) -> Option<String> {
    env.and_then(|env| env.get(name))
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| std::env::var(name).ok().filter(|value| !value.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_values_take_precedence_and_empty_values_fall_through() {
        let mut env = ProviderEnv::new();
        env.insert("AI_RS_TEST_PROVIDER_ENV".to_string(), "scoped".to_string());
        assert_eq!(
            get_provider_env_value("AI_RS_TEST_PROVIDER_ENV", Some(&env)).as_deref(),
            Some("scoped")
        );
        env.insert("AI_RS_TEST_PROVIDER_ENV".to_string(), String::new());
        assert_eq!(
            get_provider_env_value("AI_RS_TEST_PROVIDER_ENV", Some(&env)),
            None
        );
    }
}
