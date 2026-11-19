//! [rust] The native ai harness: ai.rs's own agent loop, run in-process with
//! Pi's file and shell tools working against the sandbox. Enabled by the
//! `harness-native` feature.
//!
//! There is no agent to install. The caller supplies the [`Model`]s, and
//! `HarnessTurnSettings::model` picks one by id.

use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use indexmap::IndexMap;

use crate::agent_types::StreamFn;
use crate::harness::adapters::loop_session::{
    LoopBuiltinTool, LoopSession, LoopSessionOptions, LoopTurnSenderSlot, loop_resume_messages,
};
use crate::harness::adapters::pi::pi_paths::{CreatePiPathMapperOptions, create_pi_path_mapper};
use crate::harness::adapters::pi::pi_remote_ops::{
    PiRemoteFileChangeKind, PiRemoteOpsOptions, create_pi_remote_ops,
};
use crate::harness::adapters::pi::pi_tools::{
    create_pi_builtin_agent_tools, pi_builtin_tool_requires_approval, pi_builtin_tools,
};
use crate::harness::error::HarnessError;
use crate::harness::session::{
    FileChangeKind, Harness, HarnessEvent, HarnessSession, HarnessStartOptions,
    validate_start_options,
};
use crate::harness::tools::BuiltinTool;
use crate::{AgentResult, DynAgentTool, Model};

pub const NATIVE_HARNESS_ID: &str = "ai";

/// The default system prompt of the native harness.
pub const NATIVE_SYSTEM_PROMPT: &str = "You are a coding agent working in a sandboxed workspace. Use the tools to inspect and change files and to run commands. Keep answers short and report what you changed.";

/// The native ai harness.
#[derive(Clone)]
pub struct NativeHarness {
    default_model: Model,
    models: IndexMap<String, Model>,
    system_prompt: String,
    stream_fn: Option<StreamFn>,
}

impl NativeHarness {
    /// A harness that runs `model` unless a turn names another one.
    pub fn new(model: Model) -> Self {
        let mut models = IndexMap::new();
        models.insert(model.id.clone(), model.clone());
        Self {
            default_model: model,
            models,
            system_prompt: NATIVE_SYSTEM_PROMPT.into(),
            stream_fn: None,
        }
    }

    /// Another model turns can pick by id.
    pub fn with_model(mut self, model: Model) -> Self {
        self.models.insert(model.id.clone(), model);
        self
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    /// Stream model responses with `stream_fn` instead of the provider registry.
    pub fn with_stream_fn(mut self, stream_fn: StreamFn) -> Self {
        self.stream_fn = Some(stream_fn);
        self
    }
}

#[async_trait]
impl Harness for NativeHarness {
    fn id(&self) -> &str {
        NATIVE_HARNESS_ID
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
        let sandbox = options.sandbox.session();
        let paths = create_pi_path_mapper(CreatePiPathMapperOptions {
            sandbox_work_dir: options.work_dir.clone(),
            ..Default::default()
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
            sandbox,
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

        let (is_resume, messages) = match &options.resume {
            Some(state) => (true, loop_resume_messages(NATIVE_HARNESS_ID, state)?),
            None => (false, Vec::new()),
        };
        let default_model = self.default_model.clone();
        let models = self.models.clone();
        Ok(Box::new(LoopSession::new(LoopSessionOptions {
            harness_id: NATIVE_HARNESS_ID.into(),
            session_id: options.session_id,
            is_resume,
            messages,
            system_prompt: self.system_prompt.clone(),
            builtins,
            permission_mode: options.permission_mode,
            builtin_tool_filter: options.builtin_tool_filter,
            requires_approval: Arc::new(pi_builtin_tool_requires_approval),
            resolve_model: Arc::new(move |id| match id {
                None => Ok(default_model.clone()),
                Some(id) => models.get(id).cloned().ok_or_else(|| {
                    HarnessError::new(format!(
                        "Unknown model '{id}'. Add it with NativeHarness::with_model."
                    ))
                    .into()
                }),
            }),
            stream_fn: self.stream_fn.clone(),
            cancellation_token: options.cancellation_token,
            persist: None,
            turn_sender: Some(turn_sender),
        })))
    }
}

#[cfg(test)]
mod tests;
