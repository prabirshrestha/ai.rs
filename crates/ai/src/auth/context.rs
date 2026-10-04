//! Port of `auth/context.ts`.

use std::sync::Arc;

use async_trait::async_trait;

use super::types::AuthContext;

/// Default auth context: env vars from the process environment, file
/// existence via the filesystem.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultProviderAuthContext;

/// `defaultProviderAuthContext()`.
pub fn default_provider_auth_context() -> Arc<dyn AuthContext> {
    Arc::new(DefaultProviderAuthContext)
}

#[async_trait]
impl AuthContext for DefaultProviderAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    }

    async fn file_exists(&self, path: &str) -> bool {
        let resolved = match path.strip_prefix('~') {
            Some(rest) => match home_dir() {
                Some(home) => format!("{home}{rest}"),
                None => return false,
            },
            None => path.to_string(),
        };
        tokio::fs::metadata(resolved).await.is_ok()
    }
}

/// `os.homedir()`.
pub(crate) fn home_dir() -> Option<String> {
    std::env::var("HOME")
        .ok()
        .or_else(|| std::env::var("USERPROFILE").ok())
        .filter(|home| !home.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reads_non_blank_env_values_and_checks_files() {
        let ctx = DefaultProviderAuthContext;
        assert_eq!(ctx.env("PI_AI_RS_SURELY_UNSET_VARIABLE").await, None);
        assert!(ctx.file_exists(env!("CARGO_MANIFEST_DIR")).await);
        assert!(!ctx.file_exists("/definitely/not/here").await);
    }
}
