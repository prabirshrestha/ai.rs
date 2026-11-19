//! Port of `harness-pi/src/pi-model-resolver.ts`.
//!
//! [rust] Pi's model registry is the list of [`Model`]s the caller gives
//! [`PiHarness`](super::PiHarness), and a model has configured auth when
//! the caller's check says so (by default, when its provider's API key is in
//! the environment).

use std::sync::Arc;

use indexmap::IndexMap;

use crate::Model;
use crate::harness::utils::ai_gateway_auth::get_ai_gateway_auth_from_env;

/// Default model id used when no model is configured and gateway credentials
/// are in the environment. It must be in the model list under the
/// `vercel-ai-gateway` provider.
pub const DEFAULT_PI_GATEWAY_MODEL_ID: &str = "anthropic/claude-sonnet-4.6";

/// `modelRegistry.hasConfiguredAuth(model)`.
pub type PiHasConfiguredAuth = Arc<dyn Fn(&Model) -> bool + Send + Sync>;

/// The default auth check: the provider's API key is in the environment.
pub fn pi_env_has_configured_auth() -> PiHasConfiguredAuth {
    Arc::new(|model: &Model| crate::get_env_api_key(&model.provider).is_some())
}

fn find_scoped_match<'a>(effective_id: &str, models: &'a [Model]) -> Option<&'a Model> {
    let (prefix, bare_id) = effective_id.split_once('/')?;
    models
        .iter()
        .find(|m| m.provider == prefix && (m.id == bare_id || m.name == bare_id))
}

/// `createPiModelResolver({ modelRegistry, env })(modelId)`.
pub fn resolve_pi_model(
    models: &[Model],
    has_configured_auth: &(dyn Fn(&Model) -> bool + Send + Sync),
    env: &IndexMap<String, String>,
    model_id: Option<&str>,
) -> Option<Model> {
    let use_gateway = get_ai_gateway_auth_from_env(env).api_key.is_some();
    let effective_id = match model_id {
        Some(id) => id,
        None if use_gateway => DEFAULT_PI_GATEWAY_MODEL_ID,
        None => return None,
    };
    let matches = |m: &Model| m.id == effective_id || m.name == effective_id;

    // Pi's catalog lists the same id under several providers; with gateway
    // credentials, route through the gateway.
    if use_gateway
        && let Some(model) = models
            .iter()
            .find(|m| m.provider == "vercel-ai-gateway" && matches(m))
    {
        return Some(model.clone());
    }

    let scoped_match = find_scoped_match(effective_id, models);
    if let Some(model) = scoped_match
        && has_configured_auth(model)
    {
        return Some(model.clone());
    }

    let authenticated: Vec<&Model> = models
        .iter()
        .filter(|m| matches(m) && has_configured_auth(m))
        .collect();
    if authenticated.len() == 1 {
        return Some(authenticated[0].clone());
    }

    scoped_match
        .or_else(|| models.iter().find(|m| matches(m)))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::utils::ai_gateway_auth::env;

    fn model(id: &str, name: &str, provider: &str) -> Model {
        let mut model = crate::providers::faux::register_faux_provider(None).get_model();
        model.id = id.into();
        model.name = name.into();
        model.provider = provider.into();
        model
    }

    fn sample() -> Model {
        model("my/model", "My Model", "example")
    }
    fn default_gateway() -> Model {
        model(
            DEFAULT_PI_GATEWAY_MODEL_ID,
            "Claude Sonnet 4.6",
            "vercel-ai-gateway",
        )
    }
    fn xai() -> Model {
        model("grok-4.3", "Grok 4.3", "xai")
    }
    fn xai_gateway() -> Model {
        model("xai/grok-4.3", "Grok 4.3 (gateway)", "vercel-ai-gateway")
    }
    fn openai() -> Model {
        model("gpt-4o", "GPT-4o", "openai")
    }
    fn openai_openrouter() -> Model {
        model("openai/gpt-4o", "OpenAI: GPT-4o", "openrouter")
    }

    fn no_auth(_: &Model) -> bool {
        false
    }

    fn resolve(models: &[Model], pairs: &[(&str, &str)], id: Option<&str>) -> Option<String> {
        resolve_pi_model(models, &no_auth, &env(pairs), id)
            .map(|m| format!("{}:{}", m.provider, m.id))
    }

    #[test]
    fn matches_by_id_and_name() {
        assert_eq!(
            resolve(&[sample()], &[], Some("my/model")).unwrap(),
            "example:my/model"
        );
        assert_eq!(
            resolve(&[sample()], &[], Some("My Model")).unwrap(),
            "example:my/model"
        );
    }

    #[test]
    fn prefers_the_only_authenticated_flat_match() {
        let mut azure = openai();
        azure.provider = "azure-openai-responses".into();
        let models = [azure, openai()];
        let resolved = resolve_pi_model(
            &models,
            &|m: &Model| m.provider == "openai",
            &env(&[]),
            Some("gpt-4o"),
        )
        .unwrap();
        assert_eq!(resolved.provider, "openai");
    }

    #[test]
    fn uses_the_gateway_default_with_gateway_credentials() {
        assert_eq!(
            resolve(&[default_gateway()], &[("AI_GATEWAY_API_KEY", "sk")], None).unwrap(),
            "vercel-ai-gateway:anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            resolve(&[default_gateway()], &[("VERCEL_OIDC_TOKEN", "t")], None).unwrap(),
            "vercel-ai-gateway:anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            resolve(&[sample()], &[("AI_GATEWAY_API_KEY", "sk")], None),
            None
        );
        assert_eq!(resolve(&[sample()], &[], None), None);
        assert_eq!(
            resolve(
                &[sample()],
                &[("AI_GATEWAY_API_KEY", "sk")],
                Some("unknown")
            ),
            None
        );
    }

    #[test]
    fn resolves_scoped_references_under_their_provider() {
        assert_eq!(
            resolve(&[xai_gateway(), xai()], &[], Some("xai/grok-4.3")).unwrap(),
            "xai:grok-4.3"
        );
        assert_eq!(
            resolve(&[xai(), xai_gateway()], &[], Some("xai/Grok 4.3")).unwrap(),
            "xai:grok-4.3"
        );
        assert_eq!(
            resolve(&[openai_openrouter(), openai()], &[], Some("openai/gpt-4o")).unwrap(),
            "openai:gpt-4o"
        );
    }

    #[test]
    fn prefers_an_authenticated_proxy_over_an_unauthenticated_scoped_match() {
        let models = [openai(), openai_openrouter()];
        let resolved = resolve_pi_model(
            &models,
            &|m: &Model| m.provider == "openrouter",
            &env(&[]),
            Some("openai/gpt-4o"),
        )
        .unwrap();
        assert_eq!(resolved.provider, "openrouter");
    }

    #[test]
    fn prefers_the_gateway_with_gateway_credentials() {
        let key = [("AI_GATEWAY_API_KEY", "sk")];
        assert_eq!(
            resolve(&[xai(), xai_gateway()], &key, Some("xai/grok-4.3")).unwrap(),
            "vercel-ai-gateway:xai/grok-4.3"
        );
        assert_eq!(
            resolve(&[xai_gateway(), xai()], &key, Some("xai/grok-4.3")).unwrap(),
            "vercel-ai-gateway:xai/grok-4.3"
        );
        assert_eq!(
            resolve(&[openai()], &key, Some("openai/gpt-4o")).unwrap(),
            "openai:gpt-4o"
        );
    }

    #[test]
    fn falls_through_to_flat_matching_for_unknown_prefixes() {
        assert_eq!(
            resolve(&[sample()], &[], Some("my/model")).unwrap(),
            "example:my/model"
        );
    }
}
