//! Port of `harness-opencode/src/opencode-auth.ts`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::harness::auth::HarnessAuthentication;
use crate::harness::error::HarnessResult;
use crate::harness::network::{
    HarnessRequestTransformation, HarnessRequestTransformationKeyValueMatcher,
    HarnessRequestTransformationMatch, HarnessRequestTransformationMatcher,
    HarnessRequestTransformationSources, HarnessRequestTransformationTransform,
};
use crate::harness::utils::{
    create_credential_request_transformation, get_ai_gateway_auth_from_env,
};

/// `OPENCODE_SUBSCRIPTION_ACCESS_TOKEN_ENVIRONMENT_VARIABLE`.
pub const OPENCODE_SUBSCRIPTION_ACCESS_TOKEN_ENVIRONMENT_VARIABLE: &str =
    "AI_SDK_OPENCODE_NATIVE_ACCESS_TOKEN";

/// `OPENCODE_CREDENTIAL_ENVIRONMENT_VARIABLES`.
pub const OPENCODE_CREDENTIAL_ENVIRONMENT_VARIABLES: [&str; 12] = [
    "AI_GATEWAY_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "XAI_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_COPILOT_TOKEN",
    "POE_API_KEY",
    "OPENCODE_API_KEY",
    "GITLAB_TOKEN",
    OPENCODE_SUBSCRIPTION_ACCESS_TOKEN_ENVIRONMENT_VARIABLE,
];

const NON_GOOGLE_DIRECT_CREDENTIAL_ENVIRONMENT_VARIABLES: [&str; 9] = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "XAI_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_COPILOT_TOKEN",
    "POE_API_KEY",
    "OPENCODE_API_KEY",
    "GITLAB_TOKEN",
];

/// The adapter choices of [`OpenCodeAuthenticationMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OpenCodeAuthenticationChoice {
    /// The selected provider's own credentials (environment variables, then a
    /// native OpenCode subscription); AI Gateway credentials are ignored.
    Direct,
    Anthropic,
    Openai,
}

/// `OpenCodeAuthenticationMode`.
pub type OpenCodeAuthenticationMode = HarnessAuthentication<OpenCodeAuthenticationChoice>;

/// `OpenCodeResolvedAuthenticationMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OpenCodeResolvedAuthenticationMode {
    Anthropic,
    Openai,
    Google,
    Xai,
    GithubCopilot,
    Poe,
    OpencodeGo,
    Gitlab,
    AiGateway,
}

impl OpenCodeResolvedAuthenticationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
            Self::Google => "google",
            Self::Xai => "xai",
            Self::GithubCopilot => "github-copilot",
            Self::Poe => "poe",
            Self::OpencodeGo => "opencode-go",
            Self::Gitlab => "gitlab",
            Self::AiGateway => "ai-gateway",
        }
    }

    /// `isDirectProvider`.
    pub fn direct_provider(value: &str) -> Option<Self> {
        Some(match value {
            "anthropic" => Self::Anthropic,
            "openai" => Self::Openai,
            "google" => Self::Google,
            "xai" => Self::Xai,
            "github-copilot" => Self::GithubCopilot,
            "poe" => Self::Poe,
            "opencode-go" => Self::OpencodeGo,
            "gitlab" => Self::Gitlab,
            _ => return None,
        })
    }
}

fn headers(pairs: &[(&str, String)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.clone()))
        .collect()
}

fn present<'a>(env: &'a IndexMap<String, String>, name: &str) -> Option<&'a String> {
    env.get(name).filter(|value| !value.is_empty())
}

/// `createOpenCodeRequestTransformations({ env, sandboxEnv, auth })`.
pub fn create_opencode_request_transformations(
    sources: &HarnessRequestTransformationSources<OpenCodeResolvedAuthenticationMode>,
) -> HarnessResult<Vec<HarnessRequestTransformation>> {
    use OpenCodeResolvedAuthenticationMode as M;
    let env = &sources.env;
    let sandbox = &sources.sandbox_env;
    Ok(match sources.auth {
        M::AiGateway => {
            let (Some(key), Some(sandbox_key)) = (
                present(env, "AI_GATEWAY_API_KEY"),
                present(sandbox, "AI_GATEWAY_API_KEY"),
            ) else {
                return Ok(Vec::new());
            };
            let url = env
                .get("AI_GATEWAY_BASE_URL")
                .map(String::as_str)
                .unwrap_or_default();
            let transform = headers(&[("Authorization", format!("Bearer {key}"))]);
            vec![
                create_credential_request_transformation(
                    url,
                    &headers(&[("x-api-key", sandbox_key.clone())]),
                    &transform,
                )?,
                create_credential_request_transformation(
                    url,
                    &headers(&[("Authorization", format!("Bearer {sandbox_key}"))]),
                    &transform,
                )?,
            ]
        }
        M::Openai => {
            let (Some(key), Some(sandbox_key)) = (
                present(env, "OPENAI_API_KEY"),
                present(sandbox, "OPENAI_API_KEY"),
            ) else {
                return Ok(Vec::new());
            };
            vec![create_credential_request_transformation(
                env.get("OPENAI_BASE_URL")
                    .map(String::as_str)
                    .unwrap_or("https://api.openai.com/v1"),
                &headers(&[("Authorization", format!("Bearer {sandbox_key}"))]),
                &headers(&[("Authorization", format!("Bearer {key}"))]),
            )?]
        }
        M::Anthropic => {
            let url = env
                .get("ANTHROPIC_BASE_URL")
                .map(String::as_str)
                .unwrap_or("https://api.anthropic.com");
            let mut transformations = Vec::new();
            if let (Some(key), Some(sandbox_key)) = (
                env.get("ANTHROPIC_API_KEY"),
                sandbox.get("ANTHROPIC_API_KEY"),
            ) {
                transformations.push(create_credential_request_transformation(
                    url,
                    &headers(&[("x-api-key", sandbox_key.clone())]),
                    &headers(&[("x-api-key", key.clone())]),
                )?);
            }
            if let (Some(token), Some(sandbox_token)) = (
                env.get("ANTHROPIC_AUTH_TOKEN"),
                sandbox.get("ANTHROPIC_AUTH_TOKEN"),
            ) {
                transformations.push(create_credential_request_transformation(
                    url,
                    &headers(&[("Authorization", format!("Bearer {sandbox_token}"))]),
                    &headers(&[("Authorization", format!("Bearer {token}"))]),
                )?);
            }
            transformations
        }
        M::Google => {
            let (Some(key), Some(sandbox_key)) = (
                present(env, "GOOGLE_GENERATIVE_AI_API_KEY"),
                present(sandbox, "GOOGLE_GENERATIVE_AI_API_KEY"),
            ) else {
                return Ok(Vec::new());
            };
            vec![create_credential_request_transformation(
                "https://generativelanguage.googleapis.com",
                &headers(&[("x-goog-api-key", sandbox_key.clone())]),
                &headers(&[("x-goog-api-key", key.clone())]),
            )?]
        }
        M::Xai => bearer(
            sources,
            "XAI_API_KEY",
            env.get("XAI_BASE_URL")
                .map(String::as_str)
                .unwrap_or("https://api.x.ai/v1"),
        )?,
        M::GithubCopilot => bearer(
            sources,
            if env.contains_key("GITHUB_COPILOT_TOKEN") {
                "GITHUB_COPILOT_TOKEN"
            } else {
                "GITHUB_TOKEN"
            },
            "https://api.githubcopilot.com",
        )?,
        M::Poe => bearer(sources, "POE_API_KEY", "https://api.poe.com")?,
        M::OpencodeGo => bearer(sources, "OPENCODE_API_KEY", "https://opencode.ai/zen/go/v1")?,
        M::Gitlab => {
            let (Some(credential), Some(sandbox_credential)) =
                (env.get("GITLAB_TOKEN"), sandbox.get("GITLAB_TOKEN"))
            else {
                return Ok(Vec::new());
            };
            let url = reqwest::Url::parse(&format!(
                "{}/api/v4/ai/third_party_agents/direct_access",
                env.get("GITLAB_INSTANCE_URL")
                    .map(String::as_str)
                    .unwrap_or("https://gitlab.com")
            ))?;
            vec![HarnessRequestTransformation {
                r#match: HarnessRequestTransformationMatch {
                    host: url.host_str().unwrap_or_default().to_string(),
                    path: Some(HarnessRequestTransformationMatcher::Exact(
                        url.path().to_string(),
                    )),
                    method: Some(vec!["POST".to_string()]),
                    query_string: None,
                    headers: Some(vec![HarnessRequestTransformationKeyValueMatcher {
                        key: Some(HarnessRequestTransformationMatcher::Exact(
                            "Authorization".into(),
                        )),
                        value: Some(HarnessRequestTransformationMatcher::Exact(format!(
                            "Bearer {sandbox_credential}"
                        ))),
                    }]),
                },
                transform: HarnessRequestTransformationTransform {
                    headers: headers(&[("Authorization", format!("Bearer {credential}"))]),
                },
            }]
        }
    })
}

/// `createBearerTransformation`.
fn bearer(
    sources: &HarnessRequestTransformationSources<OpenCodeResolvedAuthenticationMode>,
    name: &str,
    match_url: &str,
) -> HarnessResult<Vec<HarnessRequestTransformation>> {
    let (Some(credential), Some(sandbox_credential)) =
        (sources.env.get(name), sources.sandbox_env.get(name))
    else {
        return Ok(Vec::new());
    };
    Ok(vec![create_credential_request_transformation(
        match_url,
        &headers(&[("Authorization", format!("Bearer {sandbox_credential}"))]),
        &headers(&[("Authorization", format!("Bearer {credential}"))]),
    )?])
}

/// `resolveOpenCodeProvider({ model, provider })`.
pub fn resolve_opencode_provider(
    model: Option<&str>,
    provider: Option<&str>,
) -> OpenCodeResolvedAuthenticationMode {
    if let Some(provider) = provider.and_then(OpenCodeResolvedAuthenticationMode::direct_provider) {
        return provider;
    }
    if let Some(model) = model
        && let Some((prefix, _)) = model.split_once('/')
        && let Some(provider) = OpenCodeResolvedAuthenticationMode::direct_provider(prefix)
    {
        return provider;
    }
    OpenCodeResolvedAuthenticationMode::Anthropic
}

/// The result of [`split_opencode_model`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenCodeModelParts {
    pub provider_id: Option<String>,
    pub model_id: Option<String>,
    pub model: Option<String>,
}

/// `splitOpenCodeModel(model, provider)`.
pub fn split_opencode_model(model: Option<&str>, provider: Option<&str>) -> OpenCodeModelParts {
    let Some(model) = model.filter(|model| !model.is_empty()) else {
        return OpenCodeModelParts::default();
    };
    if let Some((provider_id, model_id)) = model.split_once('/') {
        return OpenCodeModelParts {
            provider_id: Some(provider_id.to_string()),
            model_id: Some(model_id.to_string()),
            model: Some(model.to_string()),
        };
    }
    OpenCodeModelParts {
        provider_id: provider.map(str::to_string),
        model_id: Some(model.to_string()),
        model: Some(match provider {
            Some(provider) => format!("{provider}/{model}"),
            None => model.to_string(),
        }),
    }
}

fn authentication_environment<'a>(
    auth: Option<&'a OpenCodeAuthenticationMode>,
    process_env: &'a IndexMap<String, String>,
) -> &'a IndexMap<String, String> {
    match auth {
        Some(HarnessAuthentication::Environment(env)) => env,
        _ => process_env,
    }
}

fn choice(auth: Option<&OpenCodeAuthenticationMode>) -> Option<OpenCodeAuthenticationChoice> {
    match auth {
        Some(HarnessAuthentication::Choice(choice)) => Some(*choice),
        _ => None,
    }
}

/// `resolveOpenCodeEnv({ auth, model, provider, processEnv })`.
pub fn resolve_opencode_env(
    auth: Option<&OpenCodeAuthenticationMode>,
    model: Option<&str>,
    provider: Option<&str>,
    process_env: &IndexMap<String, String>,
) -> IndexMap<String, String> {
    use OpenCodeAuthenticationChoice as C;
    use OpenCodeResolvedAuthenticationMode as M;
    let environment = authentication_environment(auth, process_env);
    let selected = resolve_authentication_provider(auth, model, provider, environment);
    let choice = choice(auth);
    if choice == Some(C::Direct)
        || (selected == M::Openai && choice == Some(C::Openai))
        || (selected == M::Anthropic && choice == Some(C::Anthropic))
    {
        return pick_direct_provider(selected, environment);
    }
    let gateway = get_ai_gateway_auth_from_env(environment);
    if matches!(auth, Some(HarnessAuthentication::AiGateway)) || gateway.api_key.is_some() {
        let mut env = IndexMap::new();
        if let Some(api_key) = gateway.api_key {
            env.insert("AI_GATEWAY_API_KEY".to_string(), api_key);
        }
        env.insert(
            "AI_GATEWAY_BASE_URL".to_string(),
            to_opencode_gateway_base_url(&gateway.base_url),
        );
        return env;
    }
    pick_direct_provider(selected, environment)
}

/// `resolveOpenCodeAuthenticationMode({ auth, model, provider, processEnv })`.
pub fn resolve_opencode_authentication_mode(
    auth: Option<&OpenCodeAuthenticationMode>,
    model: Option<&str>,
    provider: Option<&str>,
    process_env: &IndexMap<String, String>,
) -> OpenCodeResolvedAuthenticationMode {
    use OpenCodeAuthenticationChoice as C;
    use OpenCodeResolvedAuthenticationMode as M;
    if let Some(HarnessAuthentication::Environment(env)) = auth {
        return if get_ai_gateway_auth_from_env(env).api_key.is_some() {
            M::AiGateway
        } else {
            resolve_authentication_provider(auth, model, provider, env)
        };
    }
    let selected = resolve_authentication_provider(auth, model, provider, process_env);
    match (auth, choice(auth)) {
        (_, Some(C::Direct)) => selected,
        (_, Some(C::Openai)) if selected == M::Openai => M::Openai,
        (_, Some(C::Anthropic)) if selected == M::Anthropic => M::Anthropic,
        (Some(HarnessAuthentication::AiGateway), _) => M::AiGateway,
        _ if get_ai_gateway_auth_from_env(process_env).api_key.is_some() => M::AiGateway,
        _ => selected,
    }
}

/// `resolveOpenCodeAuthenticationProvider`.
fn resolve_authentication_provider(
    auth: Option<&OpenCodeAuthenticationMode>,
    model: Option<&str>,
    provider: Option<&str>,
    environment: &IndexMap<String, String>,
) -> OpenCodeResolvedAuthenticationMode {
    let selected = resolve_opencode_provider(model, provider);
    if model.is_none() && provider.is_none() {
        match choice(auth) {
            Some(OpenCodeAuthenticationChoice::Anthropic) => {
                return OpenCodeResolvedAuthenticationMode::Anthropic;
            }
            Some(OpenCodeAuthenticationChoice::Openai) => {
                return OpenCodeResolvedAuthenticationMode::Openai;
            }
            _ => {}
        }
        if present(environment, "GOOGLE_GENERATIVE_AI_API_KEY").is_some()
            && !NON_GOOGLE_DIRECT_CREDENTIAL_ENVIRONMENT_VARIABLES
                .iter()
                .any(|name| present(environment, name).is_some())
        {
            return OpenCodeResolvedAuthenticationMode::Google;
        }
    }
    selected
}

/// `pickDirectProvider`: empty values are dropped for OpenAI and Anthropic,
/// kept for the other providers, like the TS.
fn pick_direct_provider(
    provider: OpenCodeResolvedAuthenticationMode,
    process_env: &IndexMap<String, String>,
) -> IndexMap<String, String> {
    use OpenCodeResolvedAuthenticationMode as M;
    let pick_non_empty = |names: &[&str]| -> IndexMap<String, String> {
        names
            .iter()
            .filter_map(|name| {
                present(process_env, name).map(|value| (name.to_string(), value.clone()))
            })
            .collect()
    };
    let names: &[&str] = match provider {
        M::Openai => {
            return pick_non_empty(&[
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "OPENAI_ORGANIZATION",
                "OPENAI_PROJECT",
            ]);
        }
        M::Anthropic => {
            return pick_non_empty(&[
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_BASE_URL",
            ]);
        }
        M::Google => &["GOOGLE_GENERATIVE_AI_API_KEY"],
        M::Xai => &["XAI_API_KEY", "XAI_BASE_URL"],
        M::GithubCopilot => &["GITHUB_COPILOT_TOKEN", "GITHUB_TOKEN"],
        M::Poe => &["POE_API_KEY"],
        M::OpencodeGo => &["OPENCODE_API_KEY"],
        M::Gitlab | M::AiGateway => &["GITLAB_TOKEN", "GITLAB_INSTANCE_URL"],
    };
    names
        .iter()
        .filter_map(|name| {
            process_env
                .get(*name)
                .map(|value| (name.to_string(), value.clone()))
        })
        .collect()
}

/// `hasOpenCodeCredential({ environment, authenticationMode })`.
pub fn has_opencode_credential(
    environment: &IndexMap<String, String>,
    authentication_mode: OpenCodeResolvedAuthenticationMode,
) -> bool {
    if authentication_mode == OpenCodeResolvedAuthenticationMode::AiGateway {
        return environment.contains_key("AI_GATEWAY_API_KEY");
    }
    environment
        .keys()
        .any(|name| name.ends_with("_API_KEY") || name.ends_with("_TOKEN"))
}

/// `toOpenCodeGatewayBaseUrl(baseUrl)`.
pub fn to_opencode_gateway_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
