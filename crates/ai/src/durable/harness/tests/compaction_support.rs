//! Shared setup of the `test/harness-compaction.test.ts` ports: a scripted faux model that answers agent and
//! summarization requests from separate queues, and helpers over a chat Harness.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::chat::*;
use super::support::{add_hooks, context, text_of};
use crate::durable::harness::compaction::COMPACTION_TASK;
use crate::durable::harness::live::{LIVE_DOC, LiveState};
use crate::durable::harness::types::{
    CompactionDecision, CompactionHooks, CompactionPolicy, CompactionPolicyOverrides,
    CompactionRequest, CompactionResult, ConversationStreamOptions, RetryPolicyOverrides,
    SubmissionDraft,
};
use crate::durable::harness::usage::USAGE_DOC;
use crate::durable::harness::{Conversation, Harness, hook};
use crate::durable::ids::{SubmissionId, TaskId};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{Storage, TaskOutcome, TaskRecord};
use crate::error::Error as AiError;
use crate::providers::faux::{
    FauxMessageOptions, FauxModelDefinition, FauxResponseStep, RegisterFauxProviderOptions,
    faux_assistant_message,
};
use crate::types::{
    AssistantContent, AssistantMessage, CacheRetention, Message, SimpleStreamOptions, StopReason,
};

pub use crate::durable::session::tests::support::Deferred;

/// Text of about `tokens` estimated tokens, starting with `label`.
pub fn text(label: &str, tokens: usize) -> String {
    format!(
        "{label} {}",
        "x".repeat((tokens * 4).saturating_sub(label.len() + 1))
    )
}

#[derive(Clone)]
pub struct Request {
    pub messages: Vec<Message>,
    pub options: SimpleStreamOptions,
    pub model: String,
}

pub type StepFn =
    Box<dyn FnOnce(Request) -> BoxFuture<'static, Result<AssistantMessage, AiError>> + Send>;

#[allow(clippy::large_enum_variant)]
pub enum Step {
    Message(AssistantMessage),
    Run(StepFn),
}

impl From<AssistantMessage> for Step {
    fn from(message: AssistantMessage) -> Self {
        Self::Message(message)
    }
}

/// A step computed when the request arrives.
pub fn step<F, Fut>(run: F) -> Step
where
    F: FnOnce(Request) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = AssistantMessage> + Send + 'static,
{
    Step::Run(Box::new(move |request| {
        Box::pin(async move { Ok(run(request).await) })
    }))
}

/// Scripted faux model that answers agent requests and summarization requests from separate queues.
#[derive(Clone, Default)]
pub struct Script {
    agent: Arc<Mutex<VecDeque<Step>>>,
    summaries: Arc<Mutex<VecDeque<Step>>>,
    agent_requests: Arc<Mutex<Vec<Request>>>,
    summary_requests: Arc<Mutex<Vec<Request>>>,
}

impl Script {
    pub fn agent(&self, step: impl Into<Step>) {
        self.agent.lock().push_back(step.into());
    }

    pub fn summary(&self, step: impl Into<Step>) {
        self.summaries.lock().push_back(step.into());
    }

    pub fn agent_requests(&self) -> Vec<Request> {
        self.agent_requests.lock().clone()
    }

    pub fn summary_requests(&self) -> Vec<Request> {
        self.summary_requests.lock().clone()
    }

    /// Messages of the newest agent request.
    pub fn last_agent_messages(&self) -> Vec<Message> {
        self.agent_requests.lock().last().unwrap().messages.clone()
    }
}

pub fn is_summary_request(messages: &[Message]) -> bool {
    match messages.first() {
        Some(Message::System(system)) => match &system.content {
            crate::types::SystemMessageContent::Text(text) => {
                text.starts_with("You are a context summarization assistant")
            }
            _ => false,
        },
        _ => false,
    }
}

pub fn script(setup: &ChatSetup) -> Script {
    let result = Script::default();
    let steps = (0..500).map(|_| {
        let result = result.clone();
        FauxResponseStep::async_factory(move |transcript, options, _, model| {
            let request = Request {
                messages: transcript.messages.clone(),
                options,
                model: model.id.clone(),
            };
            let summary = is_summary_request(&request.messages);
            let next = if summary {
                result.summary_requests.lock().push(request.clone());
                result.summaries.lock().pop_front()
            } else {
                result.agent_requests.lock().push(request.clone());
                result.agent.lock().pop_front()
            };
            async move {
                match next {
                    None => Err(AiError::Provider(format!(
                        "No scripted {} response",
                        if summary { "summary" } else { "agent" }
                    ))),
                    Some(Step::Message(message)) => Ok(message),
                    Some(Step::Run(run)) => run(request).await,
                }
            }
        })
    });
    setup.faux.set_responses(steps.collect::<Vec<_>>());
    result
}

/// A step that waits for `gate`, or rejects when the request is cancelled.
pub fn gated(gate: Deferred, message: AssistantMessage, reached: Option<Deferred>) -> Step {
    Step::Run(Box::new(move |request| {
        Box::pin(async move {
            if let Some(reached) = reached {
                reached.resolve();
            }
            let signal = request.options.stream.signal.clone().unwrap();
            tokio::select! {
                _ = gate.wait() => Ok(message),
                _ = signal.cancelled() => Err(AiError::Aborted("Request aborted".into())),
            }
        })
    }))
}

pub fn answer(content: &str) -> AssistantMessage {
    faux_assistant_message(content, FauxMessageOptions::default())
}

pub fn failure(error_message: &str) -> AssistantMessage {
    faux_assistant_message(
        "",
        FauxMessageOptions {
            stop_reason: Some(StopReason::Error),
            error_message: Some(error_message.into()),
            ..FauxMessageOptions::default()
        },
    )
}

pub fn summary(content: &str) -> AssistantMessage {
    answer(content)
}

pub fn tool_use(content: Vec<AssistantContent>) -> AssistantMessage {
    faux_assistant_message(
        content,
        FauxMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxMessageOptions::default()
        },
    )
}

pub const OVERFLOW: &str = "prompt is too long: 250000 tokens > 200000 maximum";

/// Small thresholds: no automatic compaction unless a test enables it.
pub const MANUAL: CompactionPolicy = CompactionPolicy {
    enabled: false,
    reserve_tokens: 1000,
    keep_recent_tokens: 150,
    background_tokens: 0,
};

/// Background threshold at 500 and blocking threshold at 1500 tokens of a 2000-token window.
pub const BACKGROUND: CompactionPolicy = CompactionPolicy {
    enabled: true,
    reserve_tokens: 500,
    keep_recent_tokens: 150,
    background_tokens: 1000,
};

/// Blocking threshold at 700 tokens of a 1000-token window, no background compaction.
pub const BLOCKING: CompactionPolicy = CompactionPolicy {
    enabled: true,
    reserve_tokens: 300,
    keep_recent_tokens: 150,
    background_tokens: 0,
};

pub fn overrides(policy: CompactionPolicy) -> CompactionPolicyOverrides {
    CompactionPolicyOverrides {
        enabled: Some(policy.enabled),
        reserve_tokens: Some(policy.reserve_tokens),
        keep_recent_tokens: Some(policy.keep_recent_tokens),
        background_tokens: Some(policy.background_tokens),
    }
}

#[derive(Clone)]
pub struct Chat {
    pub harness: Harness,
    pub root: Conversation,
    pub setup: ChatSetup,
    pub faux: Script,
}

impl Chat {
    pub fn policy(&self, policy: CompactionPolicy) {
        self.setup
            .settings(|settings| settings.compaction = Some(overrides(policy)));
    }

    pub fn stream(&self, stream: ConversationStreamOptions) {
        self.setup
            .settings(|settings| settings.stream = Some(stream));
    }

    pub async fn close(&self) {
        self.harness.close(&context()).await.unwrap();
    }
}

pub fn compaction_setup(context_window: u32) -> ChatSetup {
    let mut model = FauxModelDefinition::new("faux-1");
    model.context_window = Some(context_window);
    model.max_tokens = Some(900);
    chat_setup_with(RegisterFauxProviderOptions {
        models: vec![model],
        ..RegisterFauxProviderOptions::default()
    })
}

#[derive(Default)]
pub struct OpenOptions {
    pub policy: Option<CompactionPolicy>,
    pub context_window: Option<u32>,
    pub storage: Option<Arc<dyn Storage>>,
    pub models: Option<crate::models::Models>,
}

pub async fn open(options: OpenOptions) -> Chat {
    let setup = compaction_setup(options.context_window.unwrap_or(100_000));
    let faux = script(&setup);
    open_with(options, setup, faux).await
}

pub async fn open_with(options: OpenOptions, setup: ChatSetup, faux: Script) -> Chat {
    add_text_section(&setup.registry, "preamble", "You are helpful.", Some(false));
    let storage = options
        .storage
        .unwrap_or_else(|| Arc::new(MemoryStorage::new()));
    let (harness, root) = open_chat_with(
        storage,
        &setup,
        ChatOptions {
            models: options.models,
            ..ChatOptions::default()
        },
    )
    .await;
    // These tests exercise compaction accounting and thresholds, not the faux provider's cache simulation.
    setup.settings(|settings| {
        settings.stream = Some(ConversationStreamOptions {
            cache_retention: Some(CacheRetention::None),
            ..ConversationStreamOptions::default()
        });
        settings.compaction = Some(overrides(options.policy.unwrap_or(MANUAL)));
        settings.retry = Some(RetryPolicyOverrides {
            enabled: Some(true),
            max_retries: Some(2),
            base_delay_ms: Some(1),
            max_agent_delay_ms: None,
        });
    });
    harness.resume().unwrap();
    Chat {
        harness,
        root,
        setup,
        faux,
    }
}

/// Run one turn: `user` answered by `reply`.
pub async fn turn(chat: &Chat, user: &str, reply: &str) {
    chat.faux.agent(answer(reply));
    let submission = chat
        .root
        .submit(SubmissionDraft::input(user), &context())
        .await
        .unwrap();
    assert_eq!(status(&to_record(&submission).await), "done");
}

async fn to_record(submission: &crate::durable::harness::submissions::Submission) -> JsonValue {
    serde_json::to_value(submission.wait(&context()).await.unwrap()).unwrap()
}

/// Three turns of about 100-token messages.
pub async fn history(chat: &Chat) {
    turn(chat, &text("u1", 100), &text("a1", 100)).await;
    turn(chat, &text("u2", 100), &text("a2", 100)).await;
    turn(chat, &text("u3", 100), &text("a3", 100)).await;
}

/// The settled outcome of a compaction task.
pub async fn result(chat: &Chat, id: TaskId<CompactionResult>) -> TaskOutcome {
    outcome_of(&chat.harness.wait_for_task(id, &context()).await.unwrap())
}

pub fn outcome_of(record: &TaskRecord) -> TaskOutcome {
    match &record.state {
        crate::durable::types::TaskState::Terminal { outcome } => outcome.clone(),
        state => panic!("not terminal: {state:?}"),
    }
}

/// The summary write's submission ID of a completed compaction.
pub fn submission_id(outcome: &TaskOutcome) -> Option<SubmissionId> {
    match outcome {
        TaskOutcome::Completed { result } => {
            serde_json::from_value::<CompactionResult>(result.clone())
                .unwrap()
                .submission_id
        }
        _ => None,
    }
}

pub fn outcome_status(outcome: &TaskOutcome) -> String {
    serde_json::to_value(outcome).unwrap()["status"]
        .as_str()
        .unwrap()
        .to_string()
}

pub fn status(record: &JsonValue) -> String {
    record["status"].as_str().unwrap().to_string()
}

/// The status record of a submission, as JSON.
pub async fn submission_status(chat: &Chat, id: SubmissionId) -> JsonValue {
    let submission = chat
        .harness
        .submission(id, &context())
        .await
        .unwrap()
        .unwrap();
    serde_json::to_value(submission.status(&context()).await.unwrap()).unwrap()
}

/// The settled record of a submission, as JSON.
pub async fn submission_settled(chat: &Chat, id: SubmissionId) -> JsonValue {
    let submission = chat
        .harness
        .submission(id, &context())
        .await
        .unwrap()
        .unwrap();
    serde_json::to_value(submission.wait(&context()).await.unwrap()).unwrap()
}

pub async fn kinds(conversation: &Conversation) -> Vec<String> {
    entry_kinds(conversation).await
}

pub async fn live(chat: &Chat) -> LiveState {
    chat.harness
        .snapshot(&*LIVE_DOC, chat.root.id, &context())
        .await
        .unwrap()
        .unwrap_or_default()
}

pub fn user_text(message: Option<&Message>) -> String {
    text_of(message).unwrap_or_default()
}

/// Model context of a conversation, as texts.
pub async fn context_texts(conversation: &Conversation) -> Vec<String> {
    conversation
        .context(&context())
        .await
        .unwrap()
        .messages
        .iter()
        .map(|message| user_text(Some(message)))
        .collect()
}

/// Input tokens of the faux model in the usage ledger.
pub async fn usage_input(chat: &Chat) -> u64 {
    let usage = chat
        .harness
        .snapshot(&*USAGE_DOC, chat.root.id, &context())
        .await
        .unwrap()
        .unwrap();
    serde_json::to_value(usage).unwrap()["models"]["faux/faux-1"]["input"]
        .as_u64()
        .unwrap_or(0)
}

/// Live compaction task records, from `inspect()`.
pub async fn compaction_tasks(chat: &Chat) -> Vec<TaskRecord> {
    chat.harness
        .inspect(&context())
        .await
        .unwrap()
        .tasks
        .into_iter()
        .map(|task| task.record)
        .filter(|record| record.kind == "pi.compaction")
        .collect()
}

pub fn compactions_json(live: &LiveState) -> JsonValue {
    serde_json::to_value(&live.compactions).unwrap()
}

/// Install a `beforeCompact` hook.
pub fn add_before_compact<F>(chat_setup: &ChatSetup, decide: F) -> super::support::Installed
where
    F: Fn(CompactionRequest) -> crate::durable::errors::Result<Option<CompactionDecision>>
        + Send
        + Sync
        + 'static,
{
    let decide = Arc::new(decide);
    add_hooks(
        &chat_setup.registry,
        hook(
            &*COMPACTION_TASK,
            CompactionHooks {
                before_compact: Some(Arc::new(move |request, _, _| {
                    let decision = decide(request);
                    Box::pin(async move { decision })
                })),
            },
        ),
    )
}

pub fn decline(chat_setup: &ChatSetup) -> super::support::Installed {
    add_before_compact(chat_setup, |_| Ok(Some(CompactionDecision::Decline)))
}

pub fn json_of<T: serde::Serialize>(value: &T) -> JsonValue {
    json!(value)
}
