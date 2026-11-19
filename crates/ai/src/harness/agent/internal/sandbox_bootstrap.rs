//! Port of `agent/internal/sandbox-bootstrap.ts`.

use std::fmt;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use super::bootstrap_recipe::{apply_bootstrap_recipe, hash_harness_bootstrap, or_stdout};
use super::on_bootstrap_marker::{has_on_bootstrap_marker, write_on_bootstrap_marker};
use crate::harness::agent::settings::{
    HarnessAgentSandboxConfig, OnBootstrapFn, OnBootstrapOptions,
};
use crate::harness::errors::HarnessResult;
use crate::harness::sandbox::{SandboxProcessOptions, SandboxRunOptions, SandboxSession};
use crate::harness::spec::bootstrap::HarnessV1Bootstrap;
use crate::harness::spec::harness::HarnessV1SandboxSessionRef;
use crate::harness::spec::network_sandbox::HarnessV1SandboxTemplatePrepare;
use crate::harness::spec::paths::{
    encode_harness_path_segment, harness_state_directory_path, is_absolute_posix, normalize_posix,
    posix_join,
};
use crate::harness::utils::plain_error;
use crate::harness::utils::sandbox_home_dir::{
    resolve_sandbox_default_working_directory, resolve_sandbox_home_dir,
};

const SANDBOX_BOOTSTRAP_IDENTITY_VERSION: u32 = 1;

/// `SandboxBootstrapPlan`.
#[derive(Clone, Default)]
pub struct SandboxBootstrapPlan {
    pub recipe: Option<HarnessV1Bootstrap>,
    pub recipe_identity: Option<String>,
    pub identity: Option<String>,
    pub work_dir: Option<String>,
    pub on_first_create: Option<HarnessV1SandboxTemplatePrepare>,
}

impl fmt::Debug for SandboxBootstrapPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SandboxBootstrapPlan")
            .field("recipe", &self.recipe)
            .field("recipe_identity", &self.recipe_identity)
            .field("identity", &self.identity)
            .field("work_dir", &self.work_dir)
            .field("on_first_create", &self.on_first_create.is_some())
            .finish()
    }
}

/// `validateSandboxBootstrapSettings`.
pub fn validate_sandbox_bootstrap_settings(
    settings: &HarnessAgentSandboxConfig,
) -> HarnessResult<()> {
    if settings.on_bootstrap.is_none() != settings.bootstrap_hash.is_none() {
        return Err(plain_error(
            "HarnessAgent: `sandboxConfig.onBootstrap` and `sandboxConfig.bootstrapHash` must be provided together.",
        ));
    }
    if let Some(work_dir) = &settings.work_dir {
        normalize_sandbox_work_dir(work_dir)?;
    }
    Ok(())
}

/// `normalizeSandboxWorkDir`.
pub fn normalize_sandbox_work_dir(work_dir: &str) -> HarnessResult<String> {
    if work_dir.is_empty() {
        return Err(plain_error(
            "HarnessAgent: `sandboxConfig.workDir` must not be empty.",
        ));
    }
    if work_dir.contains('\0') {
        return Err(plain_error(
            "HarnessAgent: `sandboxConfig.workDir` must not contain NUL.",
        ));
    }
    if work_dir.contains('\\') {
        return Err(plain_error(
            "HarnessAgent: `sandboxConfig.workDir` must use POSIX path separators.",
        ));
    }
    if is_absolute_posix(work_dir) {
        return Err(plain_error(
            "HarnessAgent: `sandboxConfig.workDir` must be relative.",
        ));
    }
    let normalized = normalize_posix(work_dir);
    if ((normalized == "." || normalized == "./") && work_dir != ".")
        || normalized == ".."
        || normalized.starts_with("../")
    {
        return Err(plain_error(
            "HarnessAgent: `sandboxConfig.workDir` must stay inside the sandbox default working directory.",
        ));
    }
    Ok(normalized)
}

/// `resolveSessionWorkDir`.
pub fn resolve_session_work_dir(
    default_working_directory: &str,
    harness_id: &str,
    session_id: &str,
    work_dir: Option<&str>,
) -> String {
    let path = match work_dir {
        Some(w) => w.to_string(),
        None => format!(
            "{}-{}",
            encode_harness_path_segment(harness_id),
            encode_harness_path_segment(session_id)
        ),
    };
    posix_join(&[default_working_directory, &path])
}

/// `createSandboxBootstrapPlan`.
pub fn create_sandbox_bootstrap_plan(
    recipe: Option<&HarnessV1Bootstrap>,
    settings: &HarnessAgentSandboxConfig,
) -> HarnessResult<SandboxBootstrapPlan> {
    let work_dir = settings
        .work_dir
        .as_deref()
        .map(normalize_sandbox_work_dir)
        .transpose()?;
    let recipe_identity = recipe.map(hash_harness_bootstrap);
    let has_caller_bootstrap = settings.on_bootstrap.is_some();
    let needs_combined_identity =
        has_caller_bootstrap || (recipe_identity.is_some() && work_dir.is_some());
    let identity = if needs_combined_identity && (recipe_identity.is_some() || has_caller_bootstrap)
    {
        Some(hash_sandbox_bootstrap_identity(
            recipe_identity.as_deref(),
            settings.bootstrap_hash.as_deref(),
            work_dir.as_deref(),
        ))
    } else {
        recipe_identity.clone()
    };

    let on_first_create: Option<HarnessV1SandboxTemplatePrepare> =
        if recipe.is_some() || settings.on_bootstrap.is_some() {
            let recipe = recipe.cloned();
            let recipe_identity = recipe_identity.clone();
            let work_dir = work_dir.clone();
            let on_bootstrap = settings.on_bootstrap.clone();
            let bootstrap_hash = settings.bootstrap_hash.clone();
            Some(Arc::new(move |session, abort_signal| {
                let options = RunSandboxBootstrapOptions {
                    session,
                    recipe: recipe.clone(),
                    recipe_identity: recipe_identity.clone(),
                    work_dir: work_dir.clone(),
                    on_bootstrap: on_bootstrap.clone(),
                    bootstrap_hash: bootstrap_hash.clone(),
                    skip_on_bootstrap_if_marked: false,
                    default_working_directory: None,
                    abort_signal,
                };
                Box::pin(run_sandbox_bootstrap(options))
            }))
        } else {
            None
        };

    Ok(SandboxBootstrapPlan {
        recipe: recipe.cloned(),
        recipe_identity,
        identity,
        work_dir,
        on_first_create,
    })
}

/// `runSandboxBootstrap` options.
#[derive(Clone)]
pub struct RunSandboxBootstrapOptions {
    pub session: Arc<dyn SandboxSession>,
    pub recipe: Option<HarnessV1Bootstrap>,
    pub recipe_identity: Option<String>,
    pub work_dir: Option<String>,
    pub on_bootstrap: Option<OnBootstrapFn>,
    pub bootstrap_hash: Option<String>,
    pub skip_on_bootstrap_if_marked: bool,
    pub default_working_directory: Option<String>,
    pub abort_signal: Option<CancellationToken>,
}

impl RunSandboxBootstrapOptions {
    /// Options with only `session` set.
    pub fn new(session: Arc<dyn SandboxSession>) -> Self {
        Self {
            session,
            recipe: None,
            recipe_identity: None,
            work_dir: None,
            on_bootstrap: None,
            bootstrap_hash: None,
            skip_on_bootstrap_if_marked: false,
            default_working_directory: None,
            abort_signal: None,
        }
    }
}

/// `runSandboxBootstrap`.
pub async fn run_sandbox_bootstrap(options: RunSandboxBootstrapOptions) -> HarnessResult<()> {
    let RunSandboxBootstrapOptions {
        session,
        recipe,
        recipe_identity,
        work_dir,
        on_bootstrap,
        bootstrap_hash,
        skip_on_bootstrap_if_marked,
        default_working_directory,
        abort_signal,
    } = options;
    if recipe.is_none() && on_bootstrap.is_none() {
        return Ok(());
    }

    if let (Some(recipe), Some(recipe_identity)) = (&recipe, &recipe_identity) {
        // Harness infrastructure always lives under the sandbox's own HOME.
        let home = resolve_sandbox_home_dir(&*session, abort_signal.clone()).await?;
        apply_bootstrap_recipe(
            &*session,
            recipe,
            recipe_identity,
            &harness_state_directory_path(&home),
            abort_signal.clone(),
        )
        .await?;
    }

    let Some(on_bootstrap) = on_bootstrap else {
        return Ok(());
    };

    if skip_on_bootstrap_if_marked
        && let Some(hash) = &bootstrap_hash
        && has_on_bootstrap_marker(&*session, hash, abort_signal.clone()).await?
    {
        return Ok(());
    }

    let resolved_default_working_directory = match default_working_directory {
        Some(d) => d,
        None => {
            resolve_sandbox_default_working_directory(
                &HarnessV1SandboxSessionRef::Basic(session.clone()),
                abort_signal.clone(),
            )
            .await?
        }
    };
    let bootstrap_work_dir = match &work_dir {
        None => resolved_default_working_directory,
        Some(w) => posix_join(&[&resolved_default_working_directory, w]),
    };

    ensure_sandbox_directory(&*session, &bootstrap_work_dir, abort_signal.clone()).await?;
    on_bootstrap(OnBootstrapOptions {
        session: session.clone(),
        work_dir: bootstrap_work_dir,
        abort_signal: abort_signal.clone(),
    })
    .await?;
    if let Some(hash) = &bootstrap_hash {
        write_on_bootstrap_marker(&*session, hash, abort_signal).await?;
    }
    Ok(())
}

/// `ensureSandboxDirectory`.
pub async fn ensure_sandbox_directory(
    session: &dyn SandboxSession,
    work_dir: &str,
    abort_signal: Option<CancellationToken>,
) -> HarnessResult<()> {
    let result = session
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command: "mkdir -p \"$WORK_DIR\"".into(),
                env: Some([("WORK_DIR".to_string(), work_dir.to_string())].into()),
                abort_signal,
                ..Default::default()
            },
            max_output_length: None,
        })
        .await?;
    if result.exit_code != 0 {
        return Err(plain_error(format!(
            "Failed to create sandbox work directory {work_dir} (exit {}): {}",
            result.exit_code,
            or_stdout(&result.stderr, &result.stdout)
        )));
    }
    Ok(())
}

fn hash_sandbox_bootstrap_identity(
    recipe_identity: Option<&str>,
    bootstrap_hash: Option<&str>,
    work_dir: Option<&str>,
) -> String {
    let version = SANDBOX_BOOTSTRAP_IDENTITY_VERSION.to_string();
    super::hash_strings(
        &[
            &version,
            recipe_identity.unwrap_or(""),
            bootstrap_hash.unwrap_or(""),
            work_dir.unwrap_or(""),
        ],
        8,
    )
}

#[cfg(test)]
#[path = "sandbox_bootstrap_tests.rs"]
mod tests;
