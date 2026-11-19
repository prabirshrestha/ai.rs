//! Port of `harness-pi/src/pi-harness.ts` and the parts of `pi-session.ts`
//! that apply when Pi runs on ai.rs's agent loop: Pi's tools over the
//! sandbox, Pi's system prompt, skills written to `~/.agents/skills` in the
//! sandbox, `/skill:` commands, the session log and its storage, and model
//! resolution against the caller's model list.
//!
//! [rust] Not ported: Pi's auth files and credential stores (ai.rs reads
//! provider keys itself), MCP servers, extensions, cache retention and the
//! host agent directory.

use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use indexmap::IndexMap;
use tokio::sync::Mutex;

use super::pi_model_resolver::{PiHasConfiguredAuth, pi_env_has_configured_auth, resolve_pi_model};
use super::pi_paths::{
    CreatePiPathMapperOptions, PiFileToolPathPolicy, PiReadablePathRoot, basename_posix,
    create_pi_path_mapper, join_posix,
};
use super::pi_remote_ops::{
    PiRemoteFileChangeKind, PiRemoteOpsOptions, create_pi_remote_ops,
    resolve_pi_sandbox_path_or_parent,
};
use super::pi_session_log::{
    PiResumeStateData, PiSessionLog, PiSessionStorage, is_pi_session_id, read_pi_session_entries,
    with_session_id,
};
use super::pi_skill_command::{PiCommandSkill, expand_pi_skill_command};
use super::pi_system_prompt::{PiPromptSkill, PiSystemPromptOptions, build_pi_system_prompt};
use super::pi_tools::{
    create_pi_builtin_agent_tools, pi_builtin_tool_requires_approval, pi_builtin_tools,
};
use super::pi_utils::safe_pi_metadata_segment;
use crate::agent_types::StreamFn;
use crate::harness::adapters::loop_session::{
    LoopBuiltinTool, LoopSession, LoopSessionOptions, LoopTurnPrep, LoopTurnSenderSlot,
};
use crate::harness::error::{HarnessError, into_agent_error};
use crate::harness::session::{
    FileChangeKind, Harness, HarnessEvent, HarnessResumeState, HarnessSession, HarnessStartOptions,
    validate_start_options,
};
use crate::harness::tools::BuiltinTool;
use crate::harness::utils::sandbox_home_dir::{
    get_restricted_sandbox_session, resolve_sandbox_home_dir,
};
use crate::harness::utils::write_skills::{WriteSkillsOptions, write_skills};
use crate::{
    AgentResult, DynAgentTool, Message, Model, ThinkingLevel, UserContent, UserMessageContent,
};

pub const PI_HARNESS_ID: &str = "pi";

/// A skill listed to the model (`PiSandboxSkill`). Its file must already
/// exist in the sandbox at `file_path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiSandboxSkill {
    pub name: String,
    pub description: String,
    /// Absolute sandbox path of the skill's `SKILL.md`.
    pub file_path: String,
}

/// `PiResources`: project resources for the session. `AGENTS.md` and
/// `CLAUDE.md` are not read.
#[derive(Debug, Clone, Default)]
pub struct PiResources {
    pub skills: Vec<PiSandboxSkill>,
}

/// `PiHarnessSettings`.
#[derive(Clone, Default)]
pub struct PiHarnessSettings {
    /// [rust] Pi's model catalog: the models turns can name, by `id`,
    /// `name`, or `provider/id`.
    pub models: Vec<Model>,
    /// [rust] The model used until a turn names one. Without it, Pi uses the
    /// AI Gateway default when gateway credentials are in the environment.
    pub model: Option<String>,
    /// Whether a model's provider has credentials. Defaults to its API key
    /// being in the environment.
    pub has_configured_auth: Option<PiHasConfiguredAuth>,
    /// The environment the model resolver reads. Defaults to the process
    /// environment.
    pub env: Option<IndexMap<String, String>>,
    pub thinking_level: Option<ThinkingLevel>,
    pub file_tool_path_policy: Option<PiFileToolPathPolicy>,
    pub resources: PiResources,
    /// An append-only log for the session. Without it, the resume state
    /// holds the whole session.
    pub session_storage: Option<Arc<dyn PiSessionStorage>>,
    /// Stream model responses with this instead of the provider registry.
    pub stream_fn: Option<StreamFn>,
}

/// `createPi(settings)`.
#[derive(Clone, Default)]
pub struct PiHarness {
    settings: PiHarnessSettings,
}

impl PiHarness {
    pub fn new(settings: PiHarnessSettings) -> Self {
        Self { settings }
    }
}

fn configured_prompt_skills(skills: &[PiSandboxSkill]) -> Vec<PiPromptSkill> {
    skills
        .iter()
        .map(|skill| PiPromptSkill {
            name: skill.name.clone(),
            description: skill.description.clone(),
            file_path: skill.file_path.clone(),
        })
        .collect()
}

fn dirname_posix(path: &str) -> String {
    let base = basename_posix(path);
    let dir = path[..path.len() - base.len()].trim_end_matches('/');
    if dir.is_empty() {
        "/".into()
    } else {
        dir.into()
    }
}

async fn expand_message(
    message: Message,
    skills: &[PiCommandSkill],
    sandbox: &dyn crate::harness::sandbox::SandboxSession,
    abort_signal: tokio_util::sync::CancellationToken,
) -> AgentResult<Message> {
    let Message::User(mut user) = message else {
        return Ok(message);
    };
    match &mut user.content {
        UserMessageContent::Text(text) => {
            *text = expand_pi_skill_command(text, skills, sandbox, Some(abort_signal))
                .await
                .map_err(into_agent_error)?;
        }
        UserMessageContent::Parts(parts) => {
            if let Some(UserContent::Text(text)) = parts.first_mut() {
                text.text =
                    expand_pi_skill_command(&text.text, skills, sandbox, Some(abort_signal))
                        .await
                        .map_err(into_agent_error)?;
            }
        }
    }
    Ok(Message::User(user))
}

/// Read the session log a resume state points at.
async fn load_session_log(
    options: &HarnessStartOptions,
    storage: Option<&Arc<dyn PiSessionStorage>>,
) -> AgentResult<PiSessionLog> {
    let pi_session_id =
        is_pi_session_id(&options.session_id).then_some(options.session_id.as_str());
    let data = match &options.resume {
        None => None,
        Some(state) => {
            if state.harness_id != PI_HARNESS_ID {
                return Err(HarnessError::new(format!(
                    "Cannot resume a '{}' session with the 'pi' harness.",
                    state.harness_id
                ))
                .into());
            }
            Some(
                serde_json::from_value::<PiResumeStateData>(state.data.clone()).map_err(
                    |error| {
                        HarnessError::with_cause(format!("Invalid Pi resume state: {error}"), error)
                    },
                )?,
            )
        }
    };
    let data = data.unwrap_or_default();
    if let Some(leaf_id) = &data.leaf_id {
        let Some(storage) = storage else {
            return Err(HarnessError::new(
                "The Pi lifecycle state was saved through a session storage, but createPi has no sessionStorage to load it from.",
            )
            .into());
        };
        let stored = read_pi_session_entries(
            storage.as_ref(),
            &options.session_id,
            leaf_id,
            options.cancellation_token.clone(),
        )
        .await
        .map_err(into_agent_error)?;
        let saved_count = stored.len();
        let mut entries = PiSessionLog::new(pi_session_id, &options.work_dir).entries();
        entries.extend(stored);
        return PiSessionLog::from_entries(entries, Some(leaf_id), saved_count).map_err(|error| {
            HarnessError::with_cause(
                format!(
                    "The session storage has no entry \"{leaf_id}\" for the Pi session \"{}\".",
                    options.session_id
                ),
                error,
            )
            .into()
        });
    }
    if let Some(entries) = data.entries {
        let entries = match pi_session_id {
            Some(id) => with_session_id(entries, id),
            None => entries,
        };
        return Ok(PiSessionLog::from_entries(entries, None, 0)?);
    }
    Ok(PiSessionLog::new(pi_session_id, &options.work_dir))
}

#[async_trait]
impl Harness for PiHarness {
    fn id(&self) -> &str {
        PI_HARNESS_ID
    }

    fn builtin_tools(&self) -> &IndexMap<String, BuiltinTool> {
        pi_builtin_tools()
    }

    fn supports_builtin_tool_approvals(&self) -> bool {
        true
    }

    fn supports_builtin_tool_filtering(&self) -> bool {
        true
    }

    async fn start(&self, options: HarnessStartOptions) -> AgentResult<Box<dyn HarnessSession>> {
        validate_start_options(self, &options)?;
        let settings = self.settings.clone();
        let sandbox = get_restricted_sandbox_session(&options.sandbox);
        let policy = settings.file_tool_path_policy.clone();

        let home_dir =
            resolve_sandbox_home_dir(sandbox.as_ref(), options.cancellation_token.clone())
                .await
                .map_err(into_agent_error)?;
        let mut denied_roots = policy
            .as_ref()
            .and_then(|policy| policy.denied_roots.clone());
        if let Some(roots) = &mut denied_roots {
            let mut canonical = Vec::new();
            for root in roots.iter() {
                canonical.push(
                    resolve_pi_sandbox_path_or_parent(
                        sandbox.as_ref(),
                        root,
                        root,
                        None,
                        options.cancellation_token.as_ref(),
                    )
                    .await
                    .map_err(into_agent_error)?,
                );
            }
            for root in canonical {
                if !roots.contains(&root) {
                    roots.push(root);
                }
            }
        }
        let skill_root_dir = join_posix(&[&home_dir, ".agents/skills"]);
        let configured_skills = settings.resources.skills.clone();
        let mut readable_roots = vec![PiReadablePathRoot {
            sandbox_dir: skill_root_dir.clone(),
        }];
        readable_roots.extend(configured_skills.iter().map(|skill| PiReadablePathRoot {
            sandbox_dir: dirname_posix(&skill.file_path),
        }));
        readable_roots.extend(
            policy
                .iter()
                .flat_map(|policy| policy.readable_roots.iter().flatten())
                .map(|root| PiReadablePathRoot {
                    sandbox_dir: root.clone(),
                }),
        );
        let paths = create_pi_path_mapper(CreatePiPathMapperOptions {
            sandbox_work_dir: options.work_dir.clone(),
            readable_roots: Some(readable_roots),
            writable_roots: policy
                .as_ref()
                .and_then(|policy| policy.writable_roots.clone()),
            denied_roots,
            home_dir: policy.as_ref().map(|_| home_dir.clone()),
        })?;

        let turn_sender: LoopTurnSenderSlot = Arc::new(StdMutex::new(None));
        let on_file_change = {
            let turn_sender = turn_sender.clone();
            Arc::new(
                move |kind: PiRemoteFileChangeKind, path: &str, _content: &[u8]| {
                    if let Some(sender) = turn_sender.lock().unwrap().as_ref() {
                        sender.emit(HarnessEvent::FileChange {
                            kind: match kind {
                                PiRemoteFileChangeKind::Create => FileChangeKind::Create,
                                PiRemoteFileChangeKind::Modify => FileChangeKind::Modify,
                            },
                            path: path.to_string(),
                        });
                    }
                },
            )
        };
        let ops = Arc::new(create_pi_remote_ops(PiRemoteOpsOptions {
            sandbox: sandbox.clone(),
            paths,
            env: None,
            on_file_change: Some(on_file_change),
        }));
        let builtins = create_pi_builtin_agent_tools(ops)
            .into_iter()
            .map(|tool| LoopBuiltinTool {
                harness_name: tool.native_tool().harness_name().into(),
                kind: tool.native_tool().kind(),
                tool: tool as DynAgentTool,
            })
            .collect();

        let storage = settings.session_storage.clone();
        let log = load_session_log(&options, storage.as_ref()).await?;
        let is_resume = options.resume.is_some();
        let messages = log.messages()?;
        let log = Arc::new(Mutex::new(log));

        let persist = {
            let log = log.clone();
            Arc::new(move |new_messages: Vec<Message>| {
                let log = log.clone();
                Box::pin(async move {
                    let mut log = log.lock().await;
                    for message in &new_messages {
                        log.append_message(message);
                    }
                    Ok(())
                }) as _
            })
        };
        let resume_state = {
            let log = log.clone();
            let storage = storage.clone();
            let session_id = options.session_id.clone();
            Arc::new(move |_messages: Vec<Message>| {
                let log = log.clone();
                let storage = storage.clone();
                let session_id = session_id.clone();
                Box::pin(async move {
                    let data = log
                        .lock()
                        .await
                        .resume_data(storage.as_deref(), &session_id)
                        .await
                        .map_err(into_agent_error)?;
                    Ok(HarnessResumeState {
                        harness_id: PI_HARNESS_ID.into(),
                        data: serde_json::to_value(data).expect("resume data serializes"),
                    })
                }) as _
            })
        };

        let prepare_turn = {
            let sandbox = sandbox.clone();
            let home_dir = home_dir.clone();
            let work_dir = options.work_dir.clone();
            let configured = configured_prompt_skills(&configured_skills);
            Arc::new(
                move |input: crate::harness::adapters::loop_session::LoopTurnInput| {
                    let sandbox = sandbox.clone();
                    let home_dir = home_dir.clone();
                    let skill_root_dir = skill_root_dir.clone();
                    let work_dir = work_dir.clone();
                    let configured = configured.clone();
                    Box::pin(async move {
                        let mut write = WriteSkillsOptions::new(
                            sandbox.as_ref(),
                            home_dir,
                            ".agents/skills",
                            input.settings.skills.clone(),
                        );
                        write.abort_signal = Some(input.cancellation_token.clone());
                        write.invalid_skill_name_message = Some(Arc::new(|name: &str| {
                            format!("Invalid Pi skill name: {name}")
                        }));
                        write.invalid_skill_file_path_message =
                            Some(Arc::new(|skill_name: &str, file_path: &str| {
                                format!("Invalid Pi skill file path for {skill_name}: {file_path}")
                            }));
                        write_skills(write).await.map_err(into_agent_error)?;

                        let mut skills = configured;
                        for skill in &input.settings.skills {
                            let name = safe_pi_metadata_segment(&skill.name, "skill")?;
                            skills.push(PiPromptSkill {
                                name: skill.name.clone(),
                                description: skill.description.clone(),
                                file_path: join_posix(&[&skill_root_dir, name, "SKILL.md"]),
                            });
                        }
                        let system_prompt = build_pi_system_prompt(&PiSystemPromptOptions {
                            selected_tools: input.tool_names.clone(),
                            append_system_prompt: input.settings.instructions.clone(),
                            skills: skills.clone(),
                            cwd: work_dir,
                        });
                        let command_skills: Vec<PiCommandSkill> = skills
                            .iter()
                            .map(|skill| PiCommandSkill {
                                name: skill.name.clone(),
                                file_path: skill.file_path.clone(),
                                base_dir: dirname_posix(&skill.file_path),
                            })
                            .collect();
                        let message = expand_message(
                            input.message,
                            &command_skills,
                            sandbox.as_ref(),
                            input.cancellation_token,
                        )
                        .await?;
                        Ok(LoopTurnPrep {
                            system_prompt,
                            message,
                        })
                    }) as _
                },
            )
        };

        let resolve_model = {
            let models = settings.models.clone();
            let has_auth = settings
                .has_configured_auth
                .clone()
                .unwrap_or_else(pi_env_has_configured_auth);
            let env = settings
                .env
                .clone()
                .unwrap_or_else(|| std::env::vars().collect());
            let active = StdMutex::new(resolve_pi_model(
                &models,
                has_auth.as_ref(),
                &env,
                settings.model.as_deref(),
            ));
            Arc::new(move |model_id: Option<&str>| {
                let mut active = active.lock().unwrap();
                if let Some(model_id) = model_id {
                    let model = resolve_pi_model(&models, has_auth.as_ref(), &env, Some(model_id))
                        .ok_or_else(|| {
                            HarnessError::capability_unsupported(
                                format!("Harness 'pi' has no model '{model_id}' in its catalog."),
                                Some(PI_HARNESS_ID.into()),
                            )
                        })?;
                    *active = Some(model);
                }
                active.clone().ok_or_else(|| {
                    HarnessError::new(
                        "Harness 'pi' has no model to run. Set PiHarnessSettings::model, name a model in the turn settings, or set AI_GATEWAY_API_KEY.",
                    )
                    .into()
                })
            })
        };

        Ok(Box::new(LoopSession::new(LoopSessionOptions {
            harness_id: PI_HARNESS_ID.into(),
            session_id: options.session_id,
            is_resume,
            messages,
            system_prompt: String::new(),
            builtins,
            permission_mode: options.permission_mode,
            builtin_tool_filter: options.builtin_tool_filter,
            requires_approval: Arc::new(pi_builtin_tool_requires_approval),
            resolve_model,
            stream_fn: settings.stream_fn.clone(),
            cancellation_token: options.cancellation_token,
            persist: Some(persist),
            turn_sender: Some(turn_sender),
            prepare_turn: Some(prepare_turn),
            thinking_level: settings.thinking_level,
            resume_state: Some(resume_state),
        })))
    }
}

#[cfg(test)]
#[path = "pi_harness_tests.rs"]
mod tests;
