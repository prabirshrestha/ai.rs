//! Port of durable `src/harness/submissions.ts`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use futures::future::BoxFuture;

use crate::chord::{Context, copy_json, with_abort_signal};
use crate::durable::errors::{ConversationBusy, Error, Result};
use crate::durable::ids::{ConversationId, SubmissionId};
use crate::durable::session::{SessionImpl, Transaction, TransactionScope};
use crate::durable::types::{
    CommitChange, SubmissionCreate, SubmissionRecord, SubmissionSettlement, SubmissionStatus,
    SubmissionType,
};

use super::generation::start_run;
use super::inbox::{
    BoundaryKind, INBOX_DOC, InboxItem, QueueModes, append_user, apply_boundary, entry_draft_json,
    is_stale, prepare_boundary, remove_inbox_item,
};
use super::live::LIVE_DOC;
use super::scheduler::InvocationBinding;
use super::types::{SettledSubmissionRecord, SubmissionDraft, WhenBusy};
use super::util::{Waiters, closed_error};

/// Result of withdrawing a submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortSubmissionResult {
    Aborted,
    AlreadyPlaced,
    Settled,
    NotFound,
}

impl AbortSubmissionResult {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aborted => "aborted",
            Self::AlreadyPlaced => "already_placed",
            Self::Settled => "settled",
            Self::NotFound => "not_found",
        }
    }
}

/// Admission, waits, and withdrawal of the durable submissions of one Harness.
pub struct Submissions {
    session: SessionImpl,
    now: Arc<dyn Fn() -> u64 + Send + Sync>,
    /// Read at each admission, on the Session line.
    queue_modes: Arc<dyn Fn() -> QueueModes + Send + Sync>,
    /// Enable task scheduling; submitting or waiting asks for progress.
    resume: Arc<dyn Fn() + Send + Sync>,
    waiters: Arc<Waiters<SubmissionId, SettledSubmissionRecord>>,
    closed: AtomicBool,
    me: Weak<Submissions>,
}

fn is_settled(record: &SubmissionRecord) -> bool {
    matches!(
        record.status,
        SubmissionStatus::Done | SubmissionStatus::Unanswered
    )
}

impl Submissions {
    pub fn new(
        session: SessionImpl,
        now: Arc<dyn Fn() -> u64 + Send + Sync>,
        queue_modes: Arc<dyn Fn() -> QueueModes + Send + Sync>,
        resume: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<Arc<Self>> {
        let submissions = Arc::new_cyclic(|me| Self {
            session: session.clone(),
            now,
            queue_modes,
            resume,
            waiters: Arc::default(),
            closed: AtomicBool::new(false),
            me: me.clone(),
        });
        let weak = Arc::downgrade(&submissions);
        let _ = session.subscribe_commits(move |publication, _| {
            let Some(submissions) = weak.upgrade() else {
                return;
            };
            for change in &publication.changes {
                if let CommitChange::Submission(record) = change
                    && is_settled(record)
                {
                    submissions.waiters.resolve(&record.id, record.clone());
                }
            }
        });
        let weak = Arc::downgrade(&submissions);
        let _ = session.subscribe_close(move || {
            if let Some(submissions) = weak.upgrade() {
                submissions.closed.store(true, Ordering::SeqCst);
                submissions.waiters.reject_all(closed_error());
            }
        });
        Ok(submissions)
    }

    fn arc(&self) -> Arc<Self> {
        self.me.upgrade().expect("submissions are alive")
    }

    /// Admit a submission in one commit; see `admit_submission()`.
    pub async fn submit(
        &self,
        conversation_id: ConversationId,
        draft: SubmissionDraft,
        context: &Context,
    ) -> Result<Submission> {
        (self.resume)();
        let now = (self.now)();
        let modes = (self.queue_modes)();
        let id =
            self.session
                .commit_with(
                    move |tx| async move {
                        admit_submission(&tx, conversation_id, draft, now, modes).await
                    },
                    context,
                    TransactionScope::default(),
                )
                .await?;
        Ok(Submission {
            id,
            submissions: self.arc(),
            binding: None,
        })
    }

    /// Handle for an existing submission, or `None`.
    pub async fn get(&self, id: SubmissionId, context: &Context) -> Result<Option<Submission>> {
        let storage = self.session.storage().clone();
        let callback_context = context.clone();
        let record = self
            .session
            .read_on_line(async move { storage.submission(id, &callback_context).await })
            .await?;
        Ok(record.map(|record| Submission {
            id: record.id,
            submissions: self.arc(),
            binding: None,
        }))
    }

    pub async fn status(&self, id: SubmissionId, context: &Context) -> Result<SubmissionRecord> {
        let storage = self.session.storage().clone();
        let callback_context = context.clone();
        let record = self
            .session
            .read_on_line(async move { storage.submission(id, &callback_context).await })
            .await?;
        record.ok_or_else(|| Error::message(format!("Submission {id} does not exist")))
    }

    pub async fn wait(
        &self,
        id: SubmissionId,
        context: &Context,
    ) -> Result<SettledSubmissionRecord> {
        (self.resume)();
        let me = self.arc();
        let callback_context = context.clone();
        // Check and register on the line so no settling publication falls between them.
        let found: BoxFuture<'static, Result<SettledSubmissionRecord>> = self
            .session
            .read_on_line(async move {
                let Some(record) = me
                    .session
                    .storage()
                    .submission(id, &callback_context)
                    .await?
                else {
                    return Err(Error::message(format!("Submission {id} does not exist")));
                };
                if is_settled(&record) {
                    return Ok(
                        Box::pin(futures::future::ready(Ok(record))) as BoxFuture<'static, _>
                    );
                }
                // Close rejects registered waiters synchronously and may begin during the read.
                if me.closed.load(Ordering::SeqCst) {
                    return Err(closed_error());
                }
                Ok(me.waiters.add(id, &callback_context))
            })
            .await?;
        found.await
    }

    /// Withdraw a queued submission and remove its inbox item; placed inputs and settled submissions are reported.
    pub async fn abort(
        &self,
        id: SubmissionId,
        context: &Context,
        conversation_id: Option<ConversationId>,
    ) -> Result<AbortSubmissionResult> {
        self.session
            .commit_with(
                move |tx| async move {
                    let Some(record) = tx.submission(id).await? else {
                        return Ok(AbortSubmissionResult::NotFound);
                    };
                    if conversation_id
                        .is_some_and(|conversation_id| record.conversation_id != conversation_id)
                    {
                        return Ok(AbortSubmissionResult::NotFound);
                    }
                    match record.status {
                        SubmissionStatus::Queued => {
                            tx.settle_submission(
                                id,
                                SubmissionSettlement::Unanswered {
                                    reason: "aborted".into(),
                                    detail: None,
                                },
                            )?;
                            remove_inbox_item(&tx, record.conversation_id, id).await?;
                            Ok(AbortSubmissionResult::Aborted)
                        }
                        SubmissionStatus::Placed => Ok(AbortSubmissionResult::AlreadyPlaced),
                        _ => Ok(AbortSubmissionResult::Settled),
                    }
                },
                context,
                TransactionScope::default(),
            )
            .await
    }
}

/// Handle of one admitted submission (`Submission`). A handle a task's conversation handle returned is bound to its
/// invocation: every operation first checks it and runs under its signal.
#[derive(Clone)]
pub struct Submission {
    pub id: SubmissionId,
    submissions: Arc<Submissions>,
    binding: Option<InvocationBinding>,
}

impl std::fmt::Debug for Submission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submission").field("id", &self.id).finish()
    }
}

impl Submission {
    pub(crate) fn bound(mut self, binding: InvocationBinding) -> Self {
        self.binding = Some(binding);
        self
    }

    fn bind(&self, context: &Context) -> Result<Context> {
        match &self.binding {
            None => Ok(context.clone()),
            Some(binding) => {
                (binding.check)()?;
                Ok(with_abort_signal(&binding.signal, context))
            }
        }
    }

    pub async fn status(&self, context: &Context) -> Result<SubmissionRecord> {
        let context = self.bind(context)?;
        self.submissions.status(self.id, &context).await
    }

    pub async fn wait(&self, context: &Context) -> Result<SettledSubmissionRecord> {
        let context = self.bind(context)?;
        self.submissions.wait(self.id, &context).await
    }

    pub async fn abort(&self, context: &Context) -> Result<AbortSubmissionResult> {
        let context = self.bind(context)?;
        let result = self.submissions.abort(self.id, &context, None).await?;
        if result == AbortSubmissionResult::NotFound {
            return Err(Error::message(format!(
                "Submission {} does not exist",
                self.id
            )));
        }
        Ok(result)
    }
}

/// Admit a submission inside a commit (spec §6); `Conversation.submit()` and conversation-owned compactions share it. A
/// known request ID returns its existing submission without writing. A busy conversation queues it in `pi.inbox`, or
/// rejects `whenBusy: "reject"` input with `ConversationBusy`. An idle conversation with queued items queues it behind
/// them and runs a final boundary. Otherwise idle input places a user entry and starts a run, and an idle write appends
/// its entry and settles `done`, or `stale` when its head reaches before the active range.
pub async fn admit_submission(
    tx: &Transaction,
    conversation_id: ConversationId,
    draft: SubmissionDraft,
    now: u64,
    queue_modes: QueueModes,
) -> Result<SubmissionId> {
    let type_ = match &draft {
        SubmissionDraft::Input { .. } => SubmissionType::Input,
        SubmissionDraft::Write { .. } => SubmissionType::Write,
    };
    let request_id = draft.request_id().map(str::to_string);
    if let Some(request_id) = &request_id
        && let Some(existing) = tx
            .submission_by_request(conversation_id, request_id.clone())
            .await?
    {
        if existing.type_ != type_ {
            return Err(Error::message(format!(
                "Request {request_id} already identifies a submission of type {}",
                match existing.type_ {
                    SubmissionType::Input => "input",
                    SubmissionType::Write => "write",
                }
            )));
        }
        return Ok(existing.id);
    }
    let live = tx.doc(&*LIVE_DOC, conversation_id).await?;
    let busy = live.get()?.run.is_some();
    if busy
        && let SubmissionDraft::Input {
            when_busy: Some(WhenBusy::Reject),
            ..
        } = &draft
    {
        return Err(ConversationBusy::new(conversation_id).into());
    }
    // A boundary reads the table, so it is prepared before the first table write; a busy one needs none.
    let mut boundary = if busy {
        None
    } else {
        Some(prepare_boundary(tx, conversation_id, queue_modes).await?)
    };
    let queued_items = match &boundary {
        None => true,
        Some(boundary) => !boundary.inbox.get()?.items.is_empty(),
    };
    let create = |status: SubmissionStatus| SubmissionCreate {
        conversation_id,
        request_id: request_id.clone(),
        type_,
        status,
        ..SubmissionCreate::default()
    };
    if queued_items {
        let record = tx
            .create_submission(create(SubmissionStatus::Queued))
            .await?;
        let id = record.id;
        let item = match &draft {
            SubmissionDraft::Write { entry, .. } => InboxItem::Write {
                id,
                entry: entry_draft_json(entry)?,
            },
            SubmissionDraft::Input {
                content, when_busy, ..
            } => {
                let content = copy_json(content, None)?;
                let content = serde_json::from_value(content)?;
                if *when_busy == Some(WhenBusy::Steer) {
                    InboxItem::Steer { id, content }
                } else {
                    InboxItem::FollowUp { id, content }
                }
            }
        };
        let inbox = match &boundary {
            Some(boundary) => boundary.inbox.clone(),
            None => tx.doc(&*INBOX_DOC, conversation_id).await?,
        };
        inbox.edit(|inbox| inbox.items.push(item))?;
        let Some(boundary) = &mut boundary else {
            return Ok(id);
        };
        let users = apply_boundary(tx, boundary, BoundaryKind::Final, now)
            .await?
            .users;
        if !users.is_empty() {
            start_run(tx, conversation_id, &live, users).await?;
        }
        return Ok(id);
    }
    let boundary = boundary.expect("an idle conversation has a boundary");
    match draft {
        SubmissionDraft::Write { entry, .. } => {
            if is_stale(&boundary, &entry) {
                let mut stale = create(SubmissionStatus::Unanswered);
                stale.reason = Some("stale".into());
                return Ok(tx.create_submission(stale).await?.id);
            }
            let appended = tx.append_entry(conversation_id, entry).await?;
            let mut write = create(SubmissionStatus::Done);
            write.entry = Some(appended.id);
            Ok(tx.create_submission(write).await?.id)
        }
        SubmissionDraft::Input { content, .. } => {
            let entry = append_user(tx, conversation_id, content, now).await?;
            let mut input = create(SubmissionStatus::Placed);
            input.entry = Some(entry.id);
            let id = tx.create_submission(input).await?.id;
            start_run(tx, conversation_id, &live, vec![id]).await?;
            Ok(id)
        }
    }
}
