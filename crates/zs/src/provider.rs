use std::env;

use ai::{
    Model, Result,
    providers::{anthropic, github_copilot, openai},
};

use crate::args::ProviderKind;

const DEFAULT_OPENAI_MODEL: &str = "gpt-5.5";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-5";
const DEFAULT_COPILOT_MODEL: &str = "gpt-5.5";

#[derive(Clone)]
pub enum ActiveProvider {
    OpenAi(openai::OpenAi),
    Anthropic(anthropic::Anthropic),
    GitHubCopilot(github_copilot::GitHubCopilot),
}

impl ActiveProvider {
    pub fn kind(&self) -> ProviderKind {
        match self {
            Self::OpenAi(_) => ProviderKind::OpenAi,
            Self::Anthropic(_) => ProviderKind::Anthropic,
            Self::GitHubCopilot(_) => ProviderKind::Copilot,
        }
    }

    pub fn model(&self, model_id: &str) -> Result<Model> {
        match self {
            Self::OpenAi(provider) => provider.model(model_id).build(),
            Self::Anthropic(provider) => provider.model(model_id).build(),
            Self::GitHubCopilot(provider) => provider.model(model_id).build(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderSelection {
    pub requested: Option<ProviderKind>,
    pub openai_base_url: Option<String>,
    pub openai_api_key: Option<String>,
    pub anthropic_configured: bool,
    pub copilot_configured: bool,
}

pub fn detect_provider(selection: &ProviderSelection) -> ProviderKind {
    if let Some(kind) = selection.requested {
        return kind;
    }
    if selection.openai_base_url.is_some() {
        return ProviderKind::OpenAi;
    }
    if let Some(api_key) = selection.openai_api_key.as_deref()
        && !looks_like_github_token(api_key)
    {
        return ProviderKind::OpenAi;
    }
    if selection.anthropic_configured {
        return ProviderKind::Anthropic;
    }
    if selection.copilot_configured {
        return ProviderKind::Copilot;
    }
    ProviderKind::OpenAi
}

pub fn selection_from_env(
    requested: Option<ProviderKind>,
    base_url: Option<String>,
) -> ProviderSelection {
    let requested = requested
        .or_else(|| env_nonempty("ZS_PROVIDER").and_then(|name| ProviderKind::parse_name(&name)));
    ProviderSelection {
        requested,
        openai_base_url: base_url.or_else(|| env_nonempty("OPENAI_BASE_URL")),
        openai_api_key: openai_api_key(),
        anthropic_configured: anthropic_configured(),
        copilot_configured: env_nonempty("COPILOT_GITHUB_TOKEN").is_some(),
    }
}

pub fn resolve_model_id(
    cli_model: Option<&str>,
    kind: ProviderKind,
    preferred_model: Option<&str>,
    provider_model: Option<&str>,
) -> String {
    cli_model
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .or_else(|| {
            preferred_model
                .map(str::trim)
                .filter(|model| !model.is_empty())
        })
        .or_else(|| {
            provider_model
                .map(str::trim)
                .filter(|model| !model.is_empty())
        })
        .unwrap_or_else(|| default_model_id(kind))
        .to_string()
}

pub fn model_id_from_env(cli_model: Option<&str>, kind: ProviderKind) -> String {
    let provider_model = match kind {
        ProviderKind::OpenAi => env_nonempty("OPENAI_MODEL"),
        ProviderKind::Anthropic => env_nonempty("ANTHROPIC_MODEL"),
        ProviderKind::Copilot => env_nonempty("COPILOT_MODEL"),
    };
    resolve_model_id(
        cli_model,
        kind,
        env_nonempty("ZS_MODEL").as_deref(),
        provider_model.as_deref(),
    )
}

pub fn default_model_id(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::OpenAi => DEFAULT_OPENAI_MODEL,
        ProviderKind::Anthropic => DEFAULT_ANTHROPIC_MODEL,
        ProviderKind::Copilot => DEFAULT_COPILOT_MODEL,
    }
}

pub fn build_provider(kind: ProviderKind, base_url: Option<&str>) -> Result<ActiveProvider> {
    match kind {
        ProviderKind::OpenAi => Ok(ActiveProvider::OpenAi(build_openai_provider(
            base_url,
            openai_api_key().as_deref(),
        )?)),
        ProviderKind::Anthropic => Ok(ActiveProvider::Anthropic(build_anthropic_provider()?)),
        ProviderKind::Copilot => Ok(ActiveProvider::GitHubCopilot(build_copilot_provider()?)),
    }
}

pub fn setup_error(kind: ProviderKind, selection: &ProviderSelection) -> Option<String> {
    match kind {
        ProviderKind::OpenAi => openai_setup_error(
            selection.openai_base_url.as_deref(),
            selection.openai_api_key.as_deref(),
        ),
        ProviderKind::Anthropic if !selection.anthropic_configured => Some(
            "no Anthropic credentials found; set ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_OAUTH_TOKEN before prompting"
                .to_string(),
        ),
        ProviderKind::Copilot if !selection.copilot_configured => Some(
            "no Copilot token found; run /login or set COPILOT_GITHUB_TOKEN before prompting"
                .to_string(),
        ),
        _ => None,
    }
}

pub fn openai_api_key() -> Option<String> {
    env_nonempty("OPENAI_API_KEY")
}

pub fn looks_like_github_token(api_key: &str) -> bool {
    ["ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"]
        .iter()
        .any(|prefix| api_key.starts_with(prefix))
}

fn openai_setup_error(base_url: Option<&str>, api_key: Option<&str>) -> Option<String> {
    if base_url.is_some() {
        return None;
    }

    match api_key {
        Some(api_key) if looks_like_github_token(api_key) => Some(
            "OPENAI_API_KEY looks like a GitHub token. Set OPENAI_API_KEY to an OpenAI key, unset it and run /login for Copilot, or set OPENAI_BASE_URL for a local OpenAI-compatible server."
                .to_string(),
        ),
        Some(_) => None,
        None => Some(
            "no OPENAI_API_KEY found; set it before prompting, run /login for Copilot, or set OPENAI_BASE_URL for a local OpenAI-compatible server"
                .to_string(),
        ),
    }
}

fn build_openai_provider(base_url: Option<&str>, api_key: Option<&str>) -> Result<openai::OpenAi> {
    match base_url {
        Some(base_url) => openai::builder()
            .api_key(api_key)
            .base_url(base_url)
            .chat_completions()
            .build(),
        None => openai::builder().api_key(api_key).build(),
    }
}

fn build_anthropic_provider() -> Result<anthropic::Anthropic> {
    if let Ok(auth_token) = env::var("ANTHROPIC_AUTH_TOKEN") {
        let auth_token = auth_token.trim();
        if !auth_token.is_empty() {
            return anthropic::builder().auth_token(auth_token).build();
        }
    }
    if let Some(api_key) =
        env_nonempty("ANTHROPIC_OAUTH_TOKEN").or_else(|| env_nonempty("ANTHROPIC_API_KEY"))
    {
        return anthropic::builder().api_key(api_key).build();
    }
    anthropic::builder().build()
}

fn build_copilot_provider() -> Result<github_copilot::GitHubCopilot> {
    if let Some(api_key) = env_nonempty("COPILOT_GITHUB_TOKEN") {
        return github_copilot::builder().api_key(api_key).build();
    }
    github_copilot::builder().build()
}

fn anthropic_configured() -> bool {
    env_nonempty("ANTHROPIC_AUTH_TOKEN").is_some()
        || env_nonempty("ANTHROPIC_OAUTH_TOKEN").is_some()
        || env_nonempty("ANTHROPIC_API_KEY").is_some()
}

fn env_nonempty(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{
        ProviderSelection, detect_provider, looks_like_github_token, resolve_model_id, setup_error,
    };
    use crate::args::ProviderKind;

    fn openai_missing() -> ProviderSelection {
        ProviderSelection::default()
    }

    #[test]
    fn detect_prefers_explicit_provider() {
        let selection = ProviderSelection {
            requested: Some(ProviderKind::Anthropic),
            openai_api_key: Some("sk-test".to_string()),
            ..ProviderSelection::default()
        };
        assert_eq!(detect_provider(&selection), ProviderKind::Anthropic);
    }

    #[test]
    fn detect_uses_local_base_url_before_other_creds() {
        let selection = ProviderSelection {
            openai_base_url: Some("http://localhost:11434/v1".to_string()),
            anthropic_configured: true,
            copilot_configured: true,
            ..ProviderSelection::default()
        };
        assert_eq!(detect_provider(&selection), ProviderKind::OpenAi);
    }

    #[test]
    fn detect_ignores_github_shaped_openai_key() {
        let selection = ProviderSelection {
            openai_api_key: Some("ghu_abc".to_string()),
            anthropic_configured: true,
            ..ProviderSelection::default()
        };
        assert_eq!(detect_provider(&selection), ProviderKind::Anthropic);
        assert!(looks_like_github_token("ghu_abc"));
    }

    #[test]
    fn setup_error_covers_missing_credentials() {
        assert!(
            setup_error(ProviderKind::OpenAi, &openai_missing())
                .expect("missing OpenAI key")
                .contains("no OPENAI_API_KEY found")
        );
        assert!(
            setup_error(ProviderKind::Anthropic, &openai_missing())
                .expect("missing Anthropic key")
                .contains("no Anthropic credentials")
        );
        assert!(
            setup_error(ProviderKind::Copilot, &openai_missing())
                .expect("missing Copilot token")
                .contains("no Copilot token found")
        );
        assert_eq!(
            setup_error(
                ProviderKind::OpenAi,
                &ProviderSelection {
                    openai_base_url: Some("http://localhost:11434/v1".to_string()),
                    ..ProviderSelection::default()
                }
            ),
            None
        );
    }

    #[test]
    fn model_id_prefers_cli_then_override_then_provider_default() {
        assert_eq!(
            resolve_model_id(
                Some("cli-model"),
                ProviderKind::OpenAi,
                Some("zs-model"),
                Some("openai-model")
            ),
            "cli-model"
        );
        assert_eq!(
            resolve_model_id(
                None,
                ProviderKind::Anthropic,
                Some("zs-model"),
                Some("anthropic-model")
            ),
            "zs-model"
        );
        assert_eq!(
            resolve_model_id(None, ProviderKind::Copilot, None, Some("copilot-model")),
            "copilot-model"
        );
        assert_eq!(
            resolve_model_id(None, ProviderKind::OpenAi, None, None),
            "gpt-5.5"
        );
    }
}
