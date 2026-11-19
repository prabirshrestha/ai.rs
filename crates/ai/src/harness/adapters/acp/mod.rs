//! ACP: any agent that speaks the Agent Client Protocol over stdio, run in
//! the sandbox and driven from Rust. Port of `@ai-sdk/harness-acp`. Enabled
//! by the `harness-acp` feature; Cursor and GitHub Copilot build on it.
//!
//! The agent runs for the whole session with one ACP session. The caller's
//! tools reach it as the `ai-sdk-harness-tools` MCP server, which this
//! process answers. Permission requests are approved by the permission mode
//! or asked through `before_tool_call`.
//!
//! [rust] Not ported: the TS bridge process with its reconnect, replay and
//! process-loss recovery (the agent runs under this process's control; a
//! restarted agent restores its session with `session/resume` or
//! `session/load`), `askUserQuestions`, raw stream parts and bridge
//! diagnostics.

pub mod auth;
mod client;
pub mod permissions;
pub mod settings;
pub mod translator;
mod turn;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::harness::adapters::host_tools::HostTools;
use crate::harness::bootstrap::{
    HarnessBootstrap, HarnessBootstrapCommand, HarnessBootstrapFile, apply_harness_bootstrap,
};
use crate::harness::error::{BoxError, HarnessError, into_agent_error};
use crate::harness::paths::{harness_session_data_directory_path, harness_state_directory_path};
use crate::harness::sandbox::{SandboxProcessOptions, SandboxRunOptions, SandboxSession};
use crate::harness::session::{
    Harness, HarnessResumeState, HarnessSession, HarnessStartOptions, HarnessTurn,
    HarnessTurnOptions, ResponseFormat, validate_start_options,
};
use crate::harness::tools::BuiltinTool;
use crate::harness::utils::write_skills::{WriteSkillsOptions, write_skills};
use crate::harness::utils::{
    WriteInstructionsOptions, apply_credential_forwarding, get_restricted_sandbox_session,
    resolve_sandbox_credential_environment, resolve_sandbox_home_dir, shell_quote,
    warn_credential_brokering_unavailable, write_instructions,
};
use crate::{AgentMessage, AgentResult};

pub use auth::{AcpAuthenticationMode, AcpClientApp};
pub use settings::*;
pub use translator::{AcpBuiltinToolMapping, serialize_builtin_tools};

/// `VERSION` of the ported `@ai-sdk/harness-acp` package.
pub const VERSION: &str = "1.0.128";

/// The `api` of messages from ACP agents.
pub const ACP_API: &str = "acp";

/// The adapter's resume state (`ACPLifecycleData`, without the bridge).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpResumeState {
    /// The ACP session to restore.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acp_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions_fingerprint: Option<String>,
    #[serde(default)]
    pub initial_guidance_applied: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_credential_environment: Option<IndexMap<String, String>>,
}

/// An ACP harness.
#[derive(Clone)]
pub struct AcpHarness {
    settings: Arc<AcpHarnessSettings>,
}

impl AcpHarness {
    /// `createACP`: validates the settings.
    pub fn new(settings: AcpHarnessSettings) -> Result<Self, BoxError> {
        validate_acp_settings(&settings)?;
        Ok(Self {
            settings: Arc::new(settings),
        })
    }

    pub fn settings(&self) -> &AcpHarnessSettings {
        &self.settings
    }
}

fn bootstrap_dir(harness_id: &str) -> String {
    format!(".harness-bootstrap/{harness_id}")
}

/// `createInstallScript`.
fn install_script(command: &str) -> String {
    format!(
        r#"#!/usr/bin/env bash
set -euo pipefail

ACP_BOOTSTRAP_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export HOME="$ACP_BOOTSTRAP_DIR/home"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$HOME/.local/bin"
cd "$HOME"

{command}
"#
    )
}

/// `createACPBootstrap`: installs the agent.
pub fn acp_bootstrap(harness_id: &str, source: &AcpSource) -> HarnessBootstrap {
    let dir = bootstrap_dir(harness_id);
    let (files, commands) = match source {
        AcpSource::Npm { package, version } => {
            let manifest = json!({
                "name": format!("harness-{harness_id}"),
                "version": "0.0.0",
                "private": true,
                "dependencies": { package: version },
            });
            (
                vec![HarnessBootstrapFile {
                    path: format!("{dir}/package.json"),
                    content: serde_json::to_string_pretty(&manifest).unwrap() + "\n",
                }],
                vec![HarnessBootstrapCommand {
                    command: "npm install --no-audit --no-fund --loglevel=error".into(),
                }],
            )
        }
        AcpSource::InstallCommand { command } => (
            vec![HarnessBootstrapFile {
                path: format!("{dir}/install.sh"),
                content: install_script(command),
            }],
            vec![HarnessBootstrapCommand {
                command: "bash install.sh".into(),
            }],
        ),
    };
    HarnessBootstrap {
        harness_id: harness_id.to_string(),
        bootstrap_dir: dir,
        files,
        commands,
    }
}

/// `resolveACPAgentLaunch`: the agent command and, for an install command,
/// its private `HOME`.
pub fn resolve_acp_agent_launch(
    bootstrap_directory: &str,
    source: &AcpSource,
    executable: &str,
) -> (String, Option<String>) {
    match source {
        AcpSource::InstallCommand { .. } => {
            let home = format!("{bootstrap_directory}/home");
            (format!("{home}/.local/bin/{executable}"), Some(home))
        }
        AcpSource::Npm { .. } => (
            format!("{bootstrap_directory}/node_modules/.bin/{executable}"),
            None,
        ),
    }
}

fn harness_error(message: impl Into<String>) -> crate::AgentError {
    HarnessError::new(message).into()
}

/// `fingerprintValue`.
fn fingerprint_value(value: &Value) -> String {
    let digest = ring::digest::digest(
        &ring::digest::SHA256,
        auth::stable_stringify(value).as_bytes(),
    );
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// `writeACPAuthenticationFiles`.
async fn write_authentication_files(
    sandbox: &dyn SandboxSession,
    home: &str,
    files: &[AcpAuthenticationFile],
    abort: Option<CancellationToken>,
) -> Result<(), BoxError> {
    let mut targets = Vec::new();
    for file in files {
        if file.path.is_empty()
            || file.path.starts_with('/')
            || file.path.split('/').any(|segment| segment == "..")
        {
            return Err(HarnessError::new(format!(
                "ACP authentication file path must be relative without traversal: {}.",
                serde_json::to_string(&file.path).unwrap()
            ))
            .into());
        }
        targets.push(format!("{}/{}", home.trim_end_matches('/'), file.path));
    }
    for (file, target) in files.iter().zip(&targets) {
        sandbox
            .write_text_file(crate::harness::sandbox::SandboxWriteTextFileOptions {
                path: target.clone(),
                content: file.content.clone(),
                encoding: None,
                abort_signal: abort.clone(),
            })
            .await?;
    }
    if targets.is_empty() {
        return Ok(());
    }
    let quoted: Vec<String> = targets.iter().map(|path| shell_quote(path)).collect();
    let result = sandbox
        .run(SandboxRunOptions {
            process: SandboxProcessOptions {
                command: format!("chmod 600 -- {}", quoted.join(" ")),
                abort_signal: abort,
                ..Default::default()
            },
            max_output_length: None,
        })
        .await?;
    if result.exit_code != 0 {
        return Err(HarnessError::new(format!(
            "Failed to secure ACP authentication files (exit {}){}",
            result.exit_code,
            if result.stderr.is_empty() {
                String::new()
            } else {
                format!(": {}", result.stderr)
            }
        ))
        .into());
    }
    Ok(())
}

/// `containsACPProviderCredential`.
fn contains_provider_credential(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().any(contains_provider_credential),
        Value::Object(map) => match map.get("$source") {
            Some(source) => matches!(
                source.as_str(),
                Some("gateway-api-key" | "gateway-authorization")
            ),
            None => map.values().any(contains_provider_credential),
        },
        _ => false,
    }
}

/// `resolveProviderEnvironment`: the AI Gateway profile values.
fn resolve_provider_environment(
    resolved: &auth::AcpResolvedProviderAuthenticationResult,
    client_app: &AcpClientApp,
) -> Result<IndexMap<String, String>, BoxError> {
    let Some(auth::AcpResolvedProviderAuthentication::AiGateway { env }) =
        &resolved.provider_authentication
    else {
        return Ok(IndexMap::new());
    };
    let require = |name: &str| {
        resolved.env.get(name).cloned().ok_or_else(|| {
            BoxError::from(format!(
                "ACP resolved environment value {name} is unavailable."
            ))
        })
    };
    let gateway = auth::AcpGatewayValues {
        api_key: require("AI_SDK_ACP_GATEWAY_API_KEY")?,
        base_url: require("AI_SDK_ACP_GATEWAY_BASE_URL")?,
        client_app_name: client_app.name.clone(),
        client_app_version: client_app.version.clone(),
    };
    Ok(env
        .iter()
        .map(|(key, value)| {
            let resolved = auth::resolve_acp_profile_value(value, &gateway);
            let text = match resolved {
                Value::String(text) => text,
                other => other.to_string(),
            };
            (key.clone(), text)
        })
        .collect())
}

#[async_trait]
impl Harness for AcpHarness {
    fn id(&self) -> &str {
        &self.settings.harness_id
    }

    fn builtin_tools(&self) -> &IndexMap<String, BuiltinTool> {
        &self.settings.builtin_tools
    }

    fn supports_builtin_tool_approvals(&self) -> bool {
        true
    }

    fn supports_builtin_tool_filtering(&self) -> bool {
        false
    }

    async fn bootstrap(&self) -> AgentResult<Option<HarnessBootstrap>> {
        Ok(Some(acp_bootstrap(
            &self.settings.harness_id,
            &self.settings.source,
        )))
    }

    async fn start(&self, options: HarnessStartOptions) -> AgentResult<Box<dyn HarnessSession>> {
        validate_start_options(self, &options)?;
        let settings = self.settings.clone();
        let harness_id = settings.harness_id.clone();
        let resume: AcpResumeState = match &options.resume {
            Some(state) => serde_json::from_value(state.data.clone())
                .map_err(|error| harness_error(format!("Invalid ACP resume state: {error}")))?,
            None => AcpResumeState::default(),
        };
        if let Some(mapping) = &settings.permission_mode_mapping
            && mapping.get(options.permission_mode).is_none()
        {
            // Checked again against the agent's session; fail early here.
            let mut message = format!(
                "Permission mode \"{}\" is not supported by this ACP harness.",
                serde_json::to_value(options.permission_mode)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default()
            );
            if mapping.allow_all.is_some() {
                message.push_str(" Use permissionMode: 'allow-all'.");
            }
            return Err(HarnessError::capability_unsupported(message, Some(harness_id)).into());
        }
        let sandbox = get_restricted_sandbox_session(&options.sandbox);
        let abort = options.cancellation_token.clone();

        let process_env = settings
            .process_env
            .clone()
            .unwrap_or_else(|| std::env::vars().collect());
        let authentication_env = match &settings.resolve_authentication_environment {
            Some(resolve) => resolve(settings.auth.clone(), process_env.clone())
                .await
                .map_err(into_agent_error)?,
            None => {
                auth::resolve_acp_authentication_environment(settings.auth.as_ref(), &process_env)
            }
        };
        let compatibility = auth::resolve_acp_provider_authentication_compatibility(
            settings.auth.as_ref(),
            settings.provider_authentication.as_ref(),
            &authentication_env,
        );
        let resolved_provider = auth::resolve_acp_provider_authentication(
            auth::AcpAuthResolutionOptions {
                mode: settings.auth.as_ref(),
                provider_authentication: settings.provider_authentication.as_ref(),
                client_app: &settings.client_app,
            },
            &authentication_env,
            compatibility.as_ref(),
        )
        .map_err(into_agent_error)?;
        let mut merged_env = process_env.clone();
        merged_env.extend(authentication_env.clone());
        let agent_env = resolve_acp_agent_environment(&settings, &merged_env, &authentication_env);
        let provider_env = resolve_provider_environment(&resolved_provider, &settings.client_app)
            .map_err(into_agent_error)?;
        let mut credential_names: Vec<String> = Vec::new();
        for name in settings.credential_env.iter().flatten().cloned().chain([
            "AI_GATEWAY_API_KEY".to_string(),
            "VERCEL_OIDC_TOKEN".to_string(),
        ]) {
            if !credential_names.contains(&name) {
                credential_names.push(name);
            }
        }
        if let Some(auth::AcpResolvedProviderAuthentication::AiGateway { env }) =
            &resolved_provider.provider_authentication
        {
            for (name, value) in env {
                if contains_provider_credential(value) && !credential_names.contains(name) {
                    credential_names.push(name.clone());
                }
            }
        }
        let mut brokering_env = agent_env.clone();
        brokering_env.extend(provider_env);
        let network = options
            .sandbox
            .network()
            .filter(|network| network.supports_add_request_transformations());
        let mut sandbox_credential_environment = None;
        let sandbox_env = match (&settings.credential_brokering, network) {
            (Some(brokering), Some(network)) => {
                let credentials = resolve_sandbox_credential_environment(
                    &brokering_env,
                    &credential_names,
                    settings.credential_forwarding.as_ref(),
                    resume.sandbox_credential_environment.as_ref(),
                )
                .await
                .map_err(into_agent_error)?;
                let mut sandbox_env = brokering_env.clone();
                sandbox_env.extend(credentials.clone());
                let transformations = brokering(AcpCredentialBrokeringOptions {
                    env: &brokering_env,
                    sandbox_env: Some(&sandbox_env),
                    headers: options.headers.as_ref(),
                })
                .map_err(into_agent_error)?;
                if !transformations.is_empty() {
                    network
                        .add_request_transformations(transformations)
                        .await
                        .map_err(into_agent_error)?;
                }
                sandbox_credential_environment = Some(credentials);
                sandbox_env
            }
            _ => {
                let forwarded = apply_credential_forwarding(
                    &brokering_env,
                    &credential_names,
                    settings.credential_forwarding.as_ref(),
                )
                .await
                .map_err(into_agent_error)?;
                if settings.credential_brokering.is_some() {
                    warn_credential_brokering_unavailable(
                        &brokering_env,
                        &forwarded,
                        &credential_names,
                    );
                }
                forwarded
            }
        };

        let home_dir = resolve_sandbox_home_dir(sandbox.as_ref(), abort.clone())
            .await
            .map_err(into_agent_error)?;
        let state_dir = harness_state_directory_path(&home_dir);
        let private_dir = harness_session_data_directory_path(&state_dir, &options.session_id);
        let (command, agent_home) = match &settings.executable_override {
            Some(command) => (command.clone(), None),
            None => {
                let bootstrap = acp_bootstrap(&harness_id, &settings.source);
                let directory =
                    apply_harness_bootstrap(sandbox.as_ref(), &bootstrap, abort.clone())
                        .await
                        .map_err(into_agent_error)?;
                let (executable, home) =
                    resolve_acp_agent_launch(&directory, &settings.source, &settings.executable);
                (shell_quote(&executable), home)
            }
        };
        let mut agent_command = command;
        for arg in &settings.args {
            agent_command.push(' ');
            agent_command.push_str(&shell_quote(arg));
        }
        let agent_home_dir = agent_home.clone().unwrap_or_else(|| home_dir.clone());
        let skills_directory =
            resolve_acp_skills_directory(&agent_home_dir, settings.skills_directory.as_deref())
                .map_err(into_agent_error)?;

        if let Some(files) = &settings.authentication_files {
            let files = files(AcpAuthenticationFilesOptions {
                env: &agent_env,
                sandbox_env: &sandbox_env,
                credential_brokering_available: sandbox_credential_environment.is_some(),
            });
            write_authentication_files(sandbox.as_ref(), &agent_home_dir, &files, abort.clone())
                .await
                .map_err(into_agent_error)?;
        }
        if let Some(prepare) = &settings.prepare_sandbox {
            prepare(AcpPrepareSandboxOptions {
                sandbox: sandbox.clone(),
                work_dir: options.work_dir.clone(),
                abort_signal: abort.clone(),
            })
            .await
            .map_err(into_agent_error)?;
        }
        let result = sandbox
            .run(SandboxRunOptions {
                process: SandboxProcessOptions {
                    command: format!(
                        "mkdir -p {} {}",
                        shell_quote(&options.work_dir),
                        shell_quote(&private_dir)
                    ),
                    abort_signal: abort.clone(),
                    ..Default::default()
                },
                max_output_length: None,
            })
            .await
            .map_err(into_agent_error)?;
        if result.exit_code != 0 {
            return Err(harness_error(format!(
                "Failed to create the ACP session directories: {}",
                result.stderr
            )));
        }

        let builtins =
            serialize_builtin_tools(&settings.builtin_tools, &settings.builtin_tool_titles);
        Ok(Box::new(AcpSession {
            inner: Arc::new(SessionInner {
                session_id: options.session_id,
                is_resume: options.resume.is_some(),
                config: Arc::new(turn::SessionConfig {
                    settings: settings.clone(),
                    sandbox,
                    work_dir: options.work_dir,
                    private_dir,
                    command: agent_command,
                    env: sandbox_env,
                    agent_home,
                    permission_mode: options.permission_mode,
                    builtins,
                }),
                agent_home_dir,
                skills_directory,
                sandbox_credential_environment,
                acp_session_id: Arc::new(StdMutex::new(resume.acp_session_id.clone())),
                restore: StdMutex::new(resume.acp_session_id.clone()),
                guidance: StdMutex::new((
                    resume.instructions_fingerprint.clone(),
                    resume.initial_guidance_applied || options.resume.is_some(),
                )),
                cancellation_token: options
                    .cancellation_token
                    .map(|token| token.child_token())
                    .unwrap_or_default(),
                runtime: Arc::new(tokio::sync::Mutex::new(turn::Runtime::default())),
            }),
        }))
    }
}

struct SessionInner {
    session_id: String,
    is_resume: bool,
    config: Arc<turn::SessionConfig>,
    agent_home_dir: String,
    skills_directory: String,
    sandbox_credential_environment: Option<IndexMap<String, String>>,
    acp_session_id: Arc<StdMutex<Option<String>>>,
    /// The ACP session the first turn restores.
    restore: StdMutex<Option<String>>,
    /// The instructions fingerprint and whether the first prompt went out.
    guidance: StdMutex<(Option<String>, bool)>,
    cancellation_token: CancellationToken,
    runtime: Arc<tokio::sync::Mutex<turn::Runtime>>,
}

impl SessionInner {
    fn resume_state(&self) -> HarnessResumeState {
        let (fingerprint, applied) = self.guidance.lock().unwrap().clone();
        let state = AcpResumeState {
            acp_session_id: self.acp_session_id.lock().unwrap().clone(),
            instructions_fingerprint: fingerprint,
            initial_guidance_applied: applied,
            sandbox_credential_environment: self.sandbox_credential_environment.clone(),
        };
        HarnessResumeState {
            harness_id: self.config.settings.harness_id.clone(),
            data: serde_json::to_value(state).unwrap(),
        }
    }

    async fn shut_down(&self) {
        self.cancellation_token.cancel();
        self.runtime.lock().await.close().await;
    }
}

/// An ACP session.
pub struct AcpSession {
    inner: Arc<SessionInner>,
}

#[async_trait]
impl HarnessSession for AcpSession {
    fn harness_id(&self) -> &str {
        &self.inner.config.settings.harness_id
    }

    fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    fn is_resume(&self) -> bool {
        self.inner.is_resume
    }

    async fn prompt(
        &self,
        message: AgentMessage,
        options: HarnessTurnOptions,
    ) -> AgentResult<HarnessTurn> {
        let inner = self.inner.clone();
        let mut runtime = inner
            .runtime
            .clone()
            .try_lock_owned()
            .map_err(|_| harness_error("A turn is already running in this session."))?;
        let settings = inner.config.settings.clone();
        let harness_id = settings.harness_id.clone();
        let schema = match &options.response_format {
            Some(ResponseFormat::Json { schema: None, .. }) => {
                return Err(HarnessError::capability_unsupported(
                    format!("{harness_id} requires a JSON schema for structured output."),
                    Some(harness_id),
                )
                .into());
            }
            Some(ResponseFormat::Json {
                schema: Some(schema),
                ..
            }) => {
                if settings.output_schema_mapping.is_none() {
                    return Err(HarnessError::capability_unsupported(
                        format!("{harness_id} does not support structured output through ACP."),
                        Some(harness_id),
                    )
                    .into());
                }
                Some(schema.clone())
            }
            _ => None,
        };
        let turn_settings = options.settings;
        let sandbox = inner.config.sandbox.clone();

        // `synchronizeSkills` and `synchronizeInstructions`.
        for skill in &turn_settings.skills {
            if !ACP_SKILL_NAME_PATTERN.is_match(&skill.name) {
                return Err(harness_error(format!(
                    "Invalid ACP skill name {}: expected a kebab-case slug.",
                    serde_json::to_string(&skill.name).unwrap()
                )));
            }
        }
        let skills_dir = inner
            .skills_directory
            .strip_prefix(&format!("{}/", inner.agent_home_dir.trim_end_matches('/')))
            .unwrap_or(&inner.skills_directory)
            .to_string();
        write_skills({
            let mut write = WriteSkillsOptions::new(
                sandbox.as_ref(),
                inner.agent_home_dir.clone(),
                skills_dir,
                turn_settings.skills.clone(),
            );
            write.abort_signal = options.cancellation_token.clone();
            write.skill_name_pattern = Some(ACP_SKILL_NAME_PATTERN.clone());
            write.invalid_skill_name_message = Some(Arc::new(|name| {
                format!(
                    "Invalid ACP skill name {}: expected a kebab-case slug.",
                    serde_json::to_string(name).unwrap()
                )
            }));
            write.invalid_skill_file_path_message = Some(Arc::new(|skill_name, file_path| {
                format!(
                    "Invalid ACP skill file path {} for skill {}: expected a relative POSIX path without traversal.",
                    serde_json::to_string(file_path).unwrap(),
                    serde_json::to_string(skill_name).unwrap()
                )
            }));
            write
        })
        .await
        .map_err(into_agent_error)?;
        if let Some(AcpInstructionMapping::Filesystem { path }) = &settings.instruction_mapping {
            write_instructions(WriteInstructionsOptions {
                sandbox: sandbox.as_ref(),
                home_path: inner.agent_home_dir.clone(),
                instructions_file: path.clone(),
                instructions: turn_settings.instructions.clone(),
                abort_signal: options.cancellation_token.clone(),
            })
            .await
            .map_err(into_agent_error)?;
        }

        let (blocks, has_images) = turn::prompt_content_blocks(&message);
        let next_fingerprint = fingerprint_value(
            &turn_settings
                .instructions
                .as_ref()
                .map_or(Value::Null, |instructions| json!(instructions)),
        );
        let prompt = {
            let mut guidance = inner.guidance.lock().unwrap();
            let (fingerprint, applied) = &*guidance;
            let mapping = settings.instruction_mapping.as_ref();
            let prepend = !matches!(mapping, Some(AcpInstructionMapping::Filesystem { .. }))
                && fingerprint.as_deref() != Some(next_fingerprint.as_str())
                && (mapping.is_none() || *applied);
            let prompt = if prepend {
                turn::prepend_instruction_guidance(blocks, turn_settings.instructions.as_deref())
            } else {
                blocks
            };
            *guidance = (Some(next_fingerprint), true);
            prompt
        };

        let token = inner.cancellation_token.child_token();
        if let Some(caller) = options.cancellation_token.clone() {
            let token = token.clone();
            tokio::spawn(async move {
                tokio::select! {
                    _ = caller.cancelled() => token.cancel(),
                    _ = token.cancelled() => {}
                }
            });
        }
        let config = turn::TurnConfig {
            prompt,
            has_images,
            instructions: turn_settings.instructions.clone(),
            model: turn_settings.model.clone(),
            prompt_meta: turn::output_schema_prompt_meta(
                settings.output_schema_mapping.as_ref(),
                schema.as_ref(),
            ),
            host: HostTools {
                tools: turn_settings.tools.clone(),
                before_tool_call: options.before_tool_call.clone(),
                after_tool_call: options.after_tool_call.clone(),
                cancellation_token: Some(token.clone()),
            },
            restore_session_id: inner.restore.lock().unwrap().take(),
            session_id_out: inner.acp_session_id.clone(),
        };
        let session_config = inner.config.clone();
        Ok(HarnessTurn::spawn(move |sender| async move {
            let result = turn::run_turn(
                &mut runtime,
                &session_config,
                config,
                message,
                sender,
                token.clone(),
            )
            .await;
            token.cancel();
            result
        }))
    }

    async fn detach(&self) -> AgentResult<HarnessResumeState> {
        Ok(self.inner.resume_state())
    }

    async fn stop(&self) -> AgentResult<HarnessResumeState> {
        self.inner.shut_down().await;
        Ok(self.inner.resume_state())
    }

    async fn destroy(&self) -> AgentResult<()> {
        self.inner.shut_down().await;
        Ok(())
    }
}
