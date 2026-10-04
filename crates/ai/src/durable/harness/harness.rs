//! Port of durable `src/harness/harness.ts`: the Harness, its conversation handles, and `Harness.open()`.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use futures::future::BoxFuture;

use crate::chord::{Context, with_abort_signal, without_abort_signal};
use crate::durable::entries::RESET_ENTRY;
use crate::durable::errors::{Error, Result};
use crate::durable::ids::{ConversationId, EntryId, ROOT_CONVERSATION_ID, SubmissionId, TaskId};
use crate::durable::session::{SessionHooks, SessionImpl, Transaction, TransactionScope};
use crate::durable::types::{
    ConversationOwnership, ConversationQuery, ConversationRecord, Cursor, EntryDraft, EntryHead,
    EntryQuery, EntryRecord, Page, Storage, SubmissionQuery, SubmissionStatus, TaskRecord,
};
use crate::types::{Message, UserMessage};

use super::agent::{AGENT_DOC, configure, create_agent, resolve_agent, resolve_settings};
use super::context::read_context;
use super::inbox::{INBOX_DOC, QueueModes, withdraw_queued_inputs};
use super::live::{LIVE_DOC, settle_scheduler_outcome};
use super::provider::PROVIDER_DOC;
use super::registry::{RegistrySnapshot, builtin_tasks};
use super::scheduler::{AbortTaskResult, InvocationBinding, TaskScheduler, TaskSchedulerOptions};
use super::submissions::{AbortSubmissionResult, Submission, Submissions};
use super::types::{
    Agent, AgentChange, ContextView, ConversationAbortOptions, ConversationCreateOptions,
    ConversationInit, DocumentReader, EnvTarget, HarnessInspection, HarnessOptions, Settings,
    SettledTask, SubmissionDraft,
};
use super::usage::{USAGE_DOC, UsageState, add_usage_state};
use super::util::scan_all;

const SCAN_PAGE_SIZE: usize = 256;

enum CreateTarget {
    Root,
    Independent(ConversationOwnership),
    Fork {
        parent_id: ConversationId,
        at: EntryId,
        ownership: ConversationOwnership,
    },
}

/// What the conveniences apply in the creating commit, after the creation hook.
#[derive(Clone, Default)]
pub struct CreateOptions {
    pub agent: Option<AgentChange>,
    pub init: Option<ConversationInit>,
}

impl From<&ConversationCreateOptions> for CreateOptions {
    fn from(options: &ConversationCreateOptions) -> Self {
        Self {
            agent: options.agent.clone(),
            init: options.init.clone(),
        }
    }
}

struct HarnessInner {
    session: SessionImpl,
    options: HarnessOptions,
    report: Arc<dyn Fn(Error) + Send + Sync>,
    now: Arc<dyn Fn() -> u64 + Send + Sync>,
    settings: Arc<dyn Fn() -> Settings + Send + Sync>,
    scheduler: Arc<TaskScheduler>,
    submissions: Arc<Submissions>,
    closed: AtomicBool,
}

/// Durable agent harness over one Session: the Session kernel extended with conversation handles and a registry.
#[derive(Clone)]
pub struct Harness {
    session: SessionImpl,
    inner: Arc<HarnessInner>,
}

impl std::fmt::Debug for Harness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Harness").finish_non_exhaustive()
    }
}

impl Deref for Harness {
    type Target = SessionImpl;

    fn deref(&self) -> &SessionImpl {
        &self.session
    }
}

/// The protected Session hooks of a Harness: the built-in creation hook and the close join.
struct HarnessHooks {
    inner: OnceLock<Weak<HarnessInner>>,
}

impl HarnessHooks {
    fn inner(&self) -> Option<Arc<HarnessInner>> {
        self.inner.get().and_then(Weak::upgrade)
    }
}

impl SessionHooks for HarnessHooks {
    /// The built-in creation hook, in every commit that creates or forks a conversation: empty `pi.live`, `pi.inbox`,
    /// and `pi.usage`, a fresh `pi.provider`, the conversation's `pi.agent`, then `HarnessOptions.conversationCreated`.
    fn conversation_created(
        &self,
        tx: &Transaction,
        record: &ConversationRecord,
    ) -> BoxFuture<'static, Result<()>> {
        let tx = tx.clone();
        let record = record.clone();
        let created = self
            .inner()
            .and_then(|inner| inner.options.conversation_created.clone());
        Box::pin(async move {
            tx.doc(&*LIVE_DOC, record.id).await?;
            tx.doc(&*INBOX_DOC, record.id).await?;
            tx.doc(&*USAGE_DOC, record.id).await?;
            tx.doc(&*PROVIDER_DOC, record.id).await?;
            create_agent(&tx, &record).await?;
            if let Some(created) = created {
                created(tx, record).await?;
            }
            Ok(())
        })
    }

    /// Join task invocations after admission is sealed and before Storage closes; writes no task outcome.
    fn before_close(&self) -> BoxFuture<'static, ()> {
        let scheduler = self.inner().map(|inner| inner.scheduler.clone());
        Box::pin(async move {
            if let Some(scheduler) = scheduler {
                scheduler.join().await;
            }
        })
    }
}

impl Harness {
    /// Open a Harness over storage. The registry may keep changing while the Harness runs.
    pub async fn open(
        storage: Arc<dyn Storage>,
        options: HarnessOptions,
        context: &Context,
    ) -> Result<Harness> {
        if let Some(signal) = context.abort_signal() {
            signal.throw_if_aborted()?;
        }
        let snapshot = options.registry.snapshot();
        let missing: Vec<String> = builtin_tasks()
            .iter()
            .filter(|task| snapshot.task(task.name()).is_none())
            .map(|task| task.name().to_string())
            .collect();
        if !missing.is_empty() {
            return Err(Error::message(format!(
                "Registry lacks built-in tasks {}; create it with create_registry()",
                missing.join(", ")
            )));
        }
        let harness = Self::new(storage, options, context)?;
        if let Err(error) = harness.inner.scheduler.open(context).await {
            // The caller's context may be what failed open: close without it, and return the open error.
            if let Err(close_error) = harness.close(&without_abort_signal(context)).await
                && let Some(report) = &harness.inner.options.on_report
            {
                report(close_error);
            }
            return Err(error);
        }
        Ok(harness)
    }

    fn new(storage: Arc<dyn Storage>, options: HarnessOptions, context: &Context) -> Result<Self> {
        let hooks = Arc::new(HarnessHooks {
            inner: OnceLock::new(),
        });
        let session = SessionImpl::with_hooks(storage, hooks.clone());
        let report: Arc<dyn Fn(Error) + Send + Sync> = match &options.on_report {
            Some(report) => report.clone(),
            None => Arc::new(|_| {}),
        };
        let now: Arc<dyn Fn() -> u64 + Send + Sync> = match &options.now {
            Some(now) => now.clone(),
            None => Arc::new(crate::utils::time::now_millis),
        };
        let source = options.settings.clone();
        let settings: Arc<dyn Fn() -> Settings + Send + Sync> =
            Arc::new(move || resolve_settings(source.as_ref().map(|source| source()).as_ref()));
        let inner = Arc::new_cyclic(|weak: &Weak<HarnessInner>| {
            let agent_session = session.clone();
            let agent_settings = settings.clone();
            let agent_report = report.clone();
            let env_session = session.clone();
            let env_build = options.env.clone();
            let opener = weak.clone();
            let resume = weak.clone();
            let scheduler = TaskScheduler::new(TaskSchedulerOptions {
                session: session.clone(),
                registry: options.registry.clone(),
                models: options.models.clone(),
                agent: Arc::new(move |id, snapshot, context| {
                    let session = agent_session.clone();
                    let settings = agent_settings.clone();
                    let report = agent_report.clone();
                    Box::pin(async move {
                        resolve_conversation_agent(
                            &session,
                            id,
                            &snapshot,
                            &settings(),
                            &report,
                            &context,
                        )
                        .await
                    })
                }),
                settings: settings.clone(),
                env: Arc::new(move |id, context| {
                    let session = env_session.clone();
                    let build = env_build.clone();
                    Box::pin(async move { build_env(&session, build, id, &context).await })
                }),
                now: now.clone(),
                report: report.clone(),
                settle_outcome: Arc::new(|tx, record, outcome| {
                    Box::pin(async move { settle_scheduler_outcome(&tx, &record, &outcome).await })
                }),
                withdraw_inputs: Arc::new(|tx, id| {
                    Box::pin(async move { withdraw_queued_inputs(&tx, id).await })
                }),
                conversation: Arc::new(move |id, binding, context| {
                    let inner = opener.clone();
                    Box::pin(async move {
                        let Some(inner) = inner.upgrade() else {
                            return Err(Error::message("Harness is closed"));
                        };
                        let session = inner.session.clone();
                        let storage = session.storage().clone();
                        let read_context = context.clone();
                        let record = session
                            .read_on_line(
                                async move { storage.conversation(id, &read_context).await },
                            )
                            .await?;
                        Ok(record.map(|_| ConversationHandle {
                            id,
                            binding,
                            submissions: inner.submissions.clone(),
                            scheduler: inner.scheduler.clone(),
                        }))
                    })
                }),
                context: without_abort_signal(context),
            });
            let queue_settings = settings.clone();
            let submissions = Submissions::new(
                session.clone(),
                now.clone(),
                Arc::new(move || QueueModes::from(&queue_settings())),
                Arc::new(move || {
                    if let Some(inner) = resume.upgrade() {
                        inner.scheduler.resume();
                    }
                }),
            )
            .expect("a new Session accepts listeners");
            HarnessInner {
                session: session.clone(),
                options,
                report,
                now,
                settings,
                scheduler,
                submissions,
                closed: AtomicBool::new(false),
            }
        });
        let _ = hooks.inner.set(Arc::downgrade(&inner));
        Ok(Self { session, inner })
    }

    /// The Session under this Harness.
    pub fn session(&self) -> &SessionImpl {
        &self.session
    }

    fn assert_open(&self) -> Result<()> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(Error::message("Harness is closed"));
        }
        Ok(())
    }

    /// The resolved settings, read now.
    pub fn settings(&self) -> Settings {
        (self.inner.settings)()
    }

    pub(crate) fn now(&self) -> u64 {
        (self.inner.now)()
    }

    #[allow(dead_code)]
    pub(crate) fn report(&self, error: Error) {
        (self.inner.report)(error)
    }

    #[allow(dead_code)]
    pub(crate) fn scheduler(&self) -> &Arc<TaskScheduler> {
        &self.inner.scheduler
    }

    #[allow(dead_code)]
    pub(crate) fn submissions(&self) -> &Arc<Submissions> {
        &self.inner.submissions
    }

    /// Resolve a conversation's committed `pi.agent` against `snapshot`, or the current one, and the current settings.
    pub async fn resolve_agent(
        &self,
        id: ConversationId,
        snapshot: Option<RegistrySnapshot>,
        context: &Context,
    ) -> Result<Agent> {
        let registry = snapshot.unwrap_or_else(|| self.inner.options.registry.snapshot());
        resolve_conversation_agent(
            &self.session,
            id,
            &registry,
            &self.settings(),
            &self.inner.report,
            context,
        )
        .await
    }

    /// Build a conversation's environment from its current `cwd`; `None` without an `env` option.
    pub async fn build_env(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<Arc<dyn crate::durable::env::ExecutionEnv>>> {
        build_env(&self.session, self.inner.options.env.clone(), id, context).await
    }

    /// Enable scheduling.
    pub fn resume(&self) -> Result<()> {
        self.assert_open()?;
        self.inner.scheduler.resume();
        Ok(())
    }

    /// The root conversation, created on first use.
    pub async fn root(&self, context: &Context, options: CreateOptions) -> Result<Conversation> {
        self.create(CreateTarget::Root, options, context).await
    }

    pub async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Result<Option<Conversation>> {
        self.assert_open()?;
        let storage = self.session.storage().clone();
        let read_context = context.clone();
        let record = self
            .session
            .read_on_line(async move { storage.conversation(id, &read_context).await })
            .await?;
        Ok(record.map(|record| Conversation {
            id: record.id,
            harness: self.clone(),
        }))
    }

    pub async fn create_conversation(
        &self,
        options: ConversationCreateOptions,
        context: &Context,
    ) -> Result<Conversation> {
        self.create(
            CreateTarget::Independent(options.ownership),
            CreateOptions::from(&options),
            context,
        )
        .await
    }

    pub async fn get_task<R>(
        &self,
        id: TaskId<R>,
        context: &Context,
    ) -> Result<Option<TaskRecord>> {
        let storage = self.session.storage().clone();
        let read_context = context.clone();
        let id = id.erase();
        self.session
            .read_on_line(async move { storage.task(id, &read_context).await })
            .await
    }

    /// Point-in-time view of live work.
    pub async fn inspect(&self, context: &Context) -> Result<HarnessInspection> {
        let scheduler = self.inner.scheduler.clone();
        let registry = self.inner.options.registry.clone();
        let storage = self.session.storage().clone();
        let read_context = context.clone();
        self.session
            .read_on_line(async move {
                let (scheduling, tasks) = scheduler.inspect(&registry.snapshot()).await?;
                let mut submissions = Vec::new();
                for status in [SubmissionStatus::Queued, SubmissionStatus::Placed] {
                    let query = SubmissionQuery {
                        conversation_id: None,
                        status: Some(status),
                    };
                    submissions.extend(
                        scan_all(|cursor| {
                            let storage = storage.clone();
                            let context = read_context.clone();
                            async move {
                                storage
                                    .scan_submissions(
                                        &query,
                                        SCAN_PAGE_SIZE,
                                        cursor.as_ref(),
                                        &context,
                                    )
                                    .await
                            }
                        })
                        .await?,
                    );
                }
                submissions.sort_by_key(|submission| submission.id);
                Ok(HarnessInspection {
                    scheduling,
                    tasks,
                    submissions,
                })
            })
            .await
    }

    pub async fn submission(
        &self,
        id: SubmissionId,
        context: &Context,
    ) -> Result<Option<Submission>> {
        self.inner.submissions.get(id, context).await
    }

    pub async fn abort_submission(
        &self,
        id: SubmissionId,
        context: &Context,
        conversation_id: Option<ConversationId>,
    ) -> Result<AbortSubmissionResult> {
        self.inner
            .submissions
            .abort(id, context, conversation_id)
            .await
    }

    pub async fn abort_task<R>(&self, id: TaskId<R>, context: &Context) -> Result<AbortTaskResult> {
        self.inner.scheduler.abort(id.erase(), context).await
    }

    pub async fn wait_for_task<R>(&self, id: TaskId<R>, context: &Context) -> Result<SettledTask> {
        self.inner.scheduler.resume();
        self.inner
            .scheduler
            .wait_for_task(id.erase(), context)
            .await
    }

    pub async fn wait_for_idle(&self, context: &Context) -> Result<()> {
        self.inner.scheduler.resume();
        self.inner.scheduler.wait_for_idle(None, context).await
    }

    /// Sum every conversation's committed `pi.usage`. Each document is read at its own point; totals only grow.
    pub async fn usage(&self, context: &Context) -> Result<UsageState> {
        let storage = self.session.storage().clone();
        let read_context = context.clone();
        let conversations = self
            .session
            .read_on_line(async move {
                scan_all(|cursor| {
                    let storage = storage.clone();
                    let context = read_context.clone();
                    async move {
                        storage
                            .scan_conversations(
                                &ConversationQuery::default(),
                                SCAN_PAGE_SIZE,
                                cursor.as_ref(),
                                &context,
                            )
                            .await
                    }
                })
                .await
            })
            .await?;
        let mut total = UsageState::default();
        for conversation in conversations {
            if let Some(state) = self
                .session
                .snapshot(&*USAGE_DOC, conversation.id, context)
                .await?
            {
                add_usage_state(&mut total, &state);
            }
        }
        Ok(total)
    }

    /// Seal admission, join task invocations, and close Storage. Idempotent. Admission is sealed when this is called,
    /// before the returned future is first polled.
    pub fn close(&self, context: &Context) -> BoxFuture<'static, Result<()>> {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.session.close(context)
    }

    async fn create(
        &self,
        target: CreateTarget,
        options: CreateOptions,
        context: &Context,
    ) -> Result<Conversation> {
        self.assert_open()?;
        let id = self
            .session
            .commit_with(
                move |tx| async move {
                    if let CreateTarget::Root = target
                        && tx.conversation(ROOT_CONVERSATION_ID).await?.is_some()
                    {
                        return Ok(ROOT_CONVERSATION_ID);
                    }
                    let record = match target {
                        CreateTarget::Root => tx.create_root_conversation().await?,
                        CreateTarget::Fork {
                            parent_id,
                            at,
                            ownership,
                        } => tx.fork_conversation(parent_id, at, ownership).await?,
                        CreateTarget::Independent(ownership) => {
                            tx.create_conversation(ownership).await?
                        }
                    };
                    if let Some(agent) = &options.agent {
                        configure(&tx, record.id, agent).await?;
                    }
                    if let Some(init) = &options.init {
                        init(tx.clone(), record.id).await?;
                    }
                    Ok(record.id)
                },
                context,
                TransactionScope::default(),
            )
            .await?;
        Ok(Conversation {
            id,
            harness: self.clone(),
        })
    }
}

async fn resolve_conversation_agent(
    session: &SessionImpl,
    id: ConversationId,
    snapshot: &RegistrySnapshot,
    settings: &Settings,
    report: &Arc<dyn Fn(Error) + Send + Sync>,
    context: &Context,
) -> Result<Agent> {
    let state = session.snapshot(&*AGENT_DOC, id, context).await?;
    Ok(resolve_agent(
        state.as_ref(),
        snapshot,
        settings,
        &|error| report(error),
    ))
}

async fn build_env(
    session: &SessionImpl,
    build: Option<super::types::EnvFactory>,
    id: ConversationId,
    context: &Context,
) -> Result<Option<Arc<dyn crate::durable::env::ExecutionEnv>>> {
    let Some(build) = build else {
        return Ok(None);
    };
    let cwd = session
        .snapshot(&*AGENT_DOC, id, context)
        .await?
        .and_then(|state| state.cwd);
    let target = EnvTarget {
        conversation_id: id,
        cwd,
        read: DocumentReader(Arc::new(session.clone())),
    };
    build(target, context.clone()).await
}

/// A conversation of a Harness (`Conversation`).
#[derive(Clone)]
pub struct Conversation {
    pub id: ConversationId,
    harness: Harness,
}

impl std::fmt::Debug for Conversation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conversation")
            .field("id", &self.id)
            .finish()
    }
}

impl Conversation {
    pub fn harness(&self) -> &Harness {
        &self.harness
    }

    /// The conversation's agent, resolved now against the current registry and settings.
    pub async fn agent(&self, context: &Context) -> Result<Agent> {
        self.harness.resolve_agent(self.id, None, context).await
    }

    /// Apply one change to the stored agent.
    pub async fn configure(&self, change: AgentChange, context: &Context) -> Result<()> {
        let id = self.id;
        self.harness
            .session
            .commit_with(
                move |tx| async move { configure(&tx, id, &change).await },
                context,
                TransactionScope::default(),
            )
            .await
    }

    /// Admit user input or a passive entry write.
    pub async fn submit(
        &self,
        submission: SubmissionDraft,
        context: &Context,
    ) -> Result<Submission> {
        self.harness
            .inner
            .submissions
            .submit(self.id, submission, context)
            .await
    }

    /// Start a fresh context with a reset entry, optionally carrying `handoff` as its user message.
    pub async fn reset(&self, handoff: Option<String>, context: &Context) -> Result<()> {
        let mut entry = EntryDraft::new(RESET_ENTRY.kind()).head(EntryHead::SelfEntry);
        if let Some(handoff) = handoff {
            entry.model = Some(vec![Message::User(UserMessage {
                content: handoff.into(),
                timestamp: self.harness.now(),
            })]);
        }
        self.harness
            .inner
            .submissions
            .submit(self.id, SubmissionDraft::write(entry), context)
            .await?;
        Ok(())
    }

    /// Commit with this conversation as the default `tx.create_task()` conversation.
    pub async fn commit<T, F, Fut>(&self, change: F, context: &Context) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(Transaction) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        self.harness
            .session
            .commit_with(
                change,
                context,
                TransactionScope {
                    conversation_id: Some(self.id),
                    task_id: None,
                },
            )
            .await
    }

    /// Committed raw active transcript and model context.
    pub async fn context(&self, context: &Context) -> Result<ContextView> {
        read_context(&self.harness.session, self.id, context, None).await
    }

    /// One page of this conversation's visible entries, newest first.
    pub async fn entries(
        &self,
        min_entry_id: Option<EntryId>,
        max_entry_id: Option<EntryId>,
        limit: usize,
        cursor: Option<Cursor>,
        context: &Context,
    ) -> Result<Page<EntryRecord>> {
        let query = EntryQuery {
            conversation_id: self.id,
            min_entry_id,
            max_entry_id,
        };
        let storage = self.harness.session.storage().clone();
        let read_context = context.clone();
        self.harness
            .session
            .read_on_line(async move {
                storage
                    .scan_entries(&query, limit, cursor.as_ref(), &read_context)
                    .await
            })
            .await
    }

    /// Fork this conversation at the visible entry `at`.
    pub async fn fork(
        &self,
        at: EntryId,
        options: ConversationCreateOptions,
        context: &Context,
    ) -> Result<Conversation> {
        self.harness
            .create(
                CreateTarget::Fork {
                    parent_id: self.id,
                    at,
                    ownership: options.ownership,
                },
                CreateOptions::from(&options),
                context,
            )
            .await
    }

    /// Withdraw queued inputs and abort the live work ordinary traversal reaches; resolves once the scope is idle.
    pub async fn abort(&self, context: &Context, options: ConversationAbortOptions) -> Result<()> {
        self.harness.inner.scheduler.resume();
        self.harness
            .inner
            .scheduler
            .abort_conversation(self.id, options.background, context)
            .await
    }

    pub async fn wait_for_idle(&self, context: &Context) -> Result<()> {
        self.harness.inner.scheduler.resume();
        self.harness
            .inner
            .scheduler
            .wait_for_idle(Some(self.id), context)
            .await
    }
}

/// Invocation-bound conversation handle for tasks and tools (`ConversationHandle`). Every operation, and every
/// operation of a submission it returns, first checks the invocation and runs under its signal, so it rejects once the
/// invocation ends; admitted work stays durable.
#[derive(Clone)]
pub struct ConversationHandle {
    pub id: ConversationId,
    binding: InvocationBinding,
    submissions: Arc<Submissions>,
    scheduler: Arc<TaskScheduler>,
}

impl std::fmt::Debug for ConversationHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationHandle")
            .field("id", &self.id)
            .finish()
    }
}

impl ConversationHandle {
    fn bind(&self, context: &Context) -> Result<Context> {
        (self.binding.check)()?;
        Ok(with_abort_signal(&self.binding.signal, context))
    }

    /// Admit user input; the handle accepts input submissions only.
    pub async fn submit(&self, draft: SubmissionDraft, context: &Context) -> Result<Submission> {
        if let SubmissionDraft::Write { .. } = draft {
            return Err(Error::type_error(
                "ConversationHandle.submit() accepts input submissions only",
            ));
        }
        let context = self.bind(context)?;
        let submission = self.submissions.submit(self.id, draft, &context).await?;
        Ok(submission.bound(self.binding.clone()))
    }

    pub async fn abort(&self, context: &Context, options: ConversationAbortOptions) -> Result<()> {
        let context = self.bind(context)?;
        self.scheduler
            .abort_conversation(self.id, options.background, &context)
            .await
    }

    pub async fn wait_for_idle(&self, context: &Context) -> Result<()> {
        let context = self.bind(context)?;
        self.scheduler.wait_for_idle(Some(self.id), &context).await
    }
}
