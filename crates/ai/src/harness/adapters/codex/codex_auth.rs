//! Port of `harness-codex/src/codex-auth.ts`.

use indexmap::IndexMap;

use crate::harness::errors::HarnessResult;
use crate::harness::spec::HarnessV1Authentication;
use crate::harness::spec::network_sandbox::{
    HarnessV1RequestTransformation, HarnessV1RequestTransformationSources,
};
use crate::harness::utils::{
    AiGatewayAuth, create_credential_request_transformation, get_ai_gateway_auth_from_env,
};

const DEFAULT_AI_GATEWAY_BASE_URL: &str = "https://ai-gateway.vercel.sh/v1";
/// `DEFAULT_OPENAI_BASE_URL`.
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// `CODEX_CREDENTIAL_ENVIRONMENT_VARIABLES`.
pub const CODEX_CREDENTIAL_ENVIRONMENT_VARIABLES: [&str; 2] =
    ["AI_GATEWAY_API_KEY", "CODEX_API_KEY"];

/// `createCodexRequestTransformations({ env, sandboxEnv, auth })`.
pub fn create_codex_request_transformations(
    sources: &HarnessV1RequestTransformationSources<CodexResolvedAuthenticationMode>,
) -> HarnessResult<Vec<HarnessV1RequestTransformation>> {
    let (Some(api_key), Some(sandbox_api_key)) = (
        sources
            .env
            .get("CODEX_API_KEY")
            .filter(|value| !value.is_empty()),
        sources
            .sandbox_env
            .get("CODEX_API_KEY")
            .filter(|value| !value.is_empty()),
    ) else {
        return Ok(Vec::new());
    };
    let match_url = sources
        .env
        .get("OPENAI_BASE_URL")
        .map(String::as_str)
        .unwrap_or(match sources.auth {
            CodexResolvedAuthenticationMode::AiGateway => DEFAULT_AI_GATEWAY_BASE_URL,
            CodexResolvedAuthenticationMode::Direct => DEFAULT_OPENAI_BASE_URL,
        });
    Ok(vec![create_credential_request_transformation(
        match_url,
        &IndexMap::from([(
            "Authorization".to_string(),
            format!("Bearer {sandbox_api_key}"),
        )]),
        &IndexMap::from([("Authorization".to_string(), format!("Bearer {api_key}"))]),
    )?])
}

/// `CodexResolvedAuthenticationMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CodexResolvedAuthenticationMode {
    Direct,
    AiGateway,
}

/// `CodexAuthenticationMode`: `HarnessV1Authentication`.
pub type CodexAuthenticationMode = HarnessV1Authentication;

/// `resolveCodexEnv(auth, processEnv)`.
///
/// Resolve the environment-variable blob the codex bridge needs.
/// Precedence:
///
///   1. An explicit authentication mode pins the selected route.
///   2. Auto-detect from the host process env: gateway first
///      (`AI_GATEWAY_API_KEY` / `VERCEL_OIDC_TOKEN`), then `CODEX_API_KEY` /
///      `OPENAI_API_KEY`.
///
/// [rust] `processEnv` has no `process.env` default; pass the host
/// environment explicitly.
pub fn resolve_codex_env(
    auth: Option<&CodexAuthenticationMode>,
    process_env: &IndexMap<String, String>,
) -> IndexMap<String, String> {
    let authentication_environment = match auth {
        Some(HarnessV1Authentication::Environment(environment)) => environment,
        _ => process_env,
    };
    if matches!(auth, Some(HarnessV1Authentication::Choice(_))) {
        return pick_openai(authentication_environment);
    }

    let gateway_auth_from_env = get_ai_gateway_auth_from_env(authentication_environment);
    if matches!(auth, Some(HarnessV1Authentication::AiGateway))
        || gateway_auth_from_env.api_key.is_some()
    {
        return pick_gateway(gateway_auth_from_env);
    }
    pick_openai(authentication_environment)
}

/// `resolveCodexAuthenticationMode(auth, processEnv)`.
pub fn resolve_codex_authentication_mode(
    auth: Option<&CodexAuthenticationMode>,
    process_env: &IndexMap<String, String>,
) -> CodexResolvedAuthenticationMode {
    let from_env = |env| {
        if get_ai_gateway_auth_from_env(env).api_key.is_some() {
            CodexResolvedAuthenticationMode::AiGateway
        } else {
            CodexResolvedAuthenticationMode::Direct
        }
    };
    match auth {
        Some(HarnessV1Authentication::Environment(environment)) => from_env(environment),
        Some(HarnessV1Authentication::Choice(_)) => CodexResolvedAuthenticationMode::Direct,
        Some(HarnessV1Authentication::AiGateway) => CodexResolvedAuthenticationMode::AiGateway,
        Some(HarnessV1Authentication::Auto) | None => from_env(process_env),
    }
}

fn pick_openai(process_env: &IndexMap<String, String>) -> IndexMap<String, String> {
    let mut env = IndexMap::new();
    let api_key = process_env
        .get("OPENAI_API_KEY")
        .or_else(|| process_env.get("CODEX_API_KEY"));
    for (name, value) in [
        ("CODEX_API_KEY", api_key),
        ("OPENAI_BASE_URL", process_env.get("OPENAI_BASE_URL")),
        (
            "OPENAI_ORGANIZATION",
            process_env.get("OPENAI_ORGANIZATION"),
        ),
        ("OPENAI_PROJECT", process_env.get("OPENAI_PROJECT")),
    ] {
        if let Some(value) = value.filter(|value| !value.is_empty()) {
            env.insert(name.to_string(), value.clone());
        }
    }
    env
}

fn pick_gateway(gateway_auth_from_env: AiGatewayAuth) -> IndexMap<String, String> {
    let base_url = to_codex_gateway_base_url(&gateway_auth_from_env.base_url);
    let mut env = IndexMap::new();
    if let Some(api_key) = gateway_auth_from_env.api_key {
        env.insert("AI_GATEWAY_API_KEY".into(), api_key.clone());
        env.insert("CODEX_API_KEY".into(), api_key);
    }
    env.insert("AI_GATEWAY_BASE_URL".into(), base_url.clone());
    env.insert("OPENAI_BASE_URL".into(), base_url);
    env
}

fn to_codex_gateway_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}
