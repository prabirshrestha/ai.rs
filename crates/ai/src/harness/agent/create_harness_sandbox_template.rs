//! Port of `agent/create-harness-sandbox-template.ts`.

use std::sync::Arc;

use indexmap::IndexMap;

use super::internal::bootstrap_recipe::hash_harness_bootstrap;
use super::internal::prepared_sandbox_identity::resolve_prepared_sandbox_identity;
use super::internal::sandbox_bootstrap::{
    RunSandboxBootstrapOptions, normalize_sandbox_work_dir, run_sandbox_bootstrap,
    validate_sandbox_bootstrap_settings,
};
use super::settings::HarnessAgentSandboxConfig;
use crate::harness::errors::HarnessResult;
use crate::harness::spec::bootstrap::HarnessV1Bootstrap;
use crate::harness::spec::harness::HarnessV1;
use crate::harness::spec::network_sandbox::HarnessV1SandboxTemplate;
use crate::harness::utils::plain_error;

/// `createHarnessSandboxTemplate({ harnesses, sandboxConfig })`: one
/// reusable sandbox template for several harnesses. `None` when there is
/// nothing to prepare.
///
/// `sandbox_config.on_session` is ignored.
pub async fn create_harness_sandbox_template(
    harnesses: &[Arc<dyn HarnessV1>],
    sandbox_config: Option<&HarnessAgentSandboxConfig>,
) -> HarnessResult<Option<HarnessV1SandboxTemplate>> {
    if harnesses.is_empty() {
        return Err(plain_error(
            "createHarnessSandboxTemplate: at least one harness must be provided.",
        ));
    }
    let default_settings = HarnessAgentSandboxConfig::default();
    let settings = sandbox_config.unwrap_or(&default_settings);
    validate_sandbox_bootstrap_settings(settings)?;
    let work_dir = settings
        .work_dir
        .as_deref()
        .map(normalize_sandbox_work_dir)
        .transpose()?;

    // The last adapter wins for a repeated id.
    let mut by_id: IndexMap<String, Arc<dyn HarnessV1>> = IndexMap::new();
    for harness in harnesses {
        by_id.insert(harness.harness_id().to_string(), harness.clone());
    }
    let mut unique: Vec<Arc<dyn HarnessV1>> = by_id.into_values().collect();
    unique.sort_by(|a, b| a.harness_id().cmp(b.harness_id()));

    let recipes: Vec<Option<(HarnessV1Bootstrap, String)>> =
        futures::future::try_join_all(unique.iter().map(|harness| async move {
            if !harness.supports_get_bootstrap() {
                return Ok::<_, crate::harness::errors::BoxError>(None);
            }
            let recipe = harness.get_bootstrap(None).await?;
            let identity = hash_harness_bootstrap(&recipe);
            Ok(Some((recipe, identity)))
        }))
        .await?;

    let mut recipe_identities = IndexMap::new();
    for (harness, entry) in unique.iter().zip(&recipes) {
        if let Some((_, identity)) = entry {
            recipe_identities.insert(harness.harness_id().to_string(), identity.clone());
        }
    }
    let Some(identity) = resolve_prepared_sandbox_identity(
        &recipe_identities,
        settings.bootstrap_hash.as_deref(),
        work_dir.as_deref(),
    ) else {
        return Ok(None);
    };

    let recipes: Vec<(HarnessV1Bootstrap, String)> = recipes.into_iter().flatten().collect();
    let on_bootstrap = settings.on_bootstrap.clone();
    let bootstrap_hash = settings.bootstrap_hash.clone();
    Ok(Some(HarnessV1SandboxTemplate {
        identity,
        prepare: Arc::new(move |session, abort_signal| {
            let recipes = recipes.clone();
            let work_dir = work_dir.clone();
            let on_bootstrap = on_bootstrap.clone();
            let bootstrap_hash = bootstrap_hash.clone();
            Box::pin(async move {
                for (recipe, identity) in recipes {
                    run_sandbox_bootstrap(RunSandboxBootstrapOptions {
                        recipe: Some(recipe),
                        recipe_identity: Some(identity),
                        abort_signal: abort_signal.clone(),
                        ..RunSandboxBootstrapOptions::new(session.clone())
                    })
                    .await?;
                }
                run_sandbox_bootstrap(RunSandboxBootstrapOptions {
                    work_dir,
                    on_bootstrap,
                    bootstrap_hash,
                    skip_on_bootstrap_if_marked: true,
                    abort_signal,
                    ..RunSandboxBootstrapOptions::new(session)
                })
                .await
            })
        }),
    }))
}

#[cfg(test)]
#[path = "create_harness_sandbox_template_tests.rs"]
mod tests;
