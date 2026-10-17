//! Port of `test/harness-inbox.test.ts`.
//!
//! Divergences: the reopened SQLite file is `ControlledStorage::persistent()`. The base/delta recording storage is
//! `ControlledStorage`'s commit log, which also holds the first push the TS suite cannot attribute yet. Operations are
//! compared in their TS JSON form; positional removals are diffed at prepare time, so several removals in one commit
//! coalesce into fewer splices than the TS draft records.

use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::chat::*;
use super::support::{add_hooks, add_tool, context, to_json};
use crate::durable::DocToken;
use crate::durable::documents::define_doc;
use crate::durable::entries::RESET_ENTRY;
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::inbox::{INBOX_DOC, InboxItem};
use crate::durable::harness::submissions::{AbortSubmissionResult, Submission};
use crate::durable::harness::tool::TOOL_TASK;
use crate::durable::harness::types::{
    ConversationCreateOptions, GenerationHooks, QueueMode, RetryPolicyOverrides, SubmissionDraft,
    ToolControl, ToolExecutionResult, ToolHooks, UserInput, WhenBusy,
};
use crate::durable::harness::usage::{USAGE_DOC, UsageBucket, record_usage};
use crate::durable::harness::{Conversation, Harness, define_tool, hook};
use crate::durable::ids::SubmissionId;
use crate::durable::session::tests::support::{ControlledStorage, Deferred, flush};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{
    CheckpointInfo, CommitChange, DocDefinition, DocumentContent, EntryDraft, EntryHead,
    EntryRecord, SessionScope, Storage, StorageWrite, SubmissionRecord, SubmissionStatus,
};
use crate::providers::faux::{
    FauxMessageOptions, FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions,
    faux_assistant_message, faux_tool_call,
};
use crate::types::{AssistantMessage, Message, StopReason, Usage, UsageCost};

fn answer(text: &str) -> AssistantMessage {
    faux_assistant_message(text, FauxMessageOptions::default())
}

fn step(text: &str) -> FauxResponseStep {
    answer(text).into()
}

fn failure(message: &str) -> AssistantMessage {
    faux_assistant_message(
        Vec::new(),
        FauxMessageOptions {
            stop_reason: Some(StopReason::Error),
            error_message: Some(message.into()),
            ..FauxMessageOptions::default()
        },
    )
}

/// A faux response held until `release` or cancellation; `reached` resolves when the request is sent.
struct Gated {
    step: FauxResponseStep,
    reached: Deferred,
    gate: Deferred,
}

impl Gated {
    fn new(message: AssistantMessage) -> Self {
        let (reached, gate) = (Deferred::default(), Deferred::default());
        let (reach, wait) = (reached.clone(), gate.clone());
        let step = FauxResponseStep::async_factory(move |_, options, _, _| {
            let (reach, wait, message) = (reach.clone(), wait.clone(), message.clone());
            async move {
                reach.resolve();
                match options.stream.signal.clone() {
                    Some(signal) => tokio::select! {
                        _ = wait.wait() => Ok(message),
                        _ = signal.cancelled() => Err(crate::error::Error::Aborted("Request aborted".into())),
                    },
                    None => {
                        wait.wait().await;
                        Ok(message)
                    }
                }
            }
        });
        Self {
            step,
            reached,
            gate,
        }
    }

    fn step(&self) -> FauxResponseStep {
        self.step.clone()
    }

    fn release(&self) {
        self.gate.resolve();
    }
}

/// Register a `hold` tool whose calls wait for `gate` and then return `result`.
fn hold_tool(setup: &ChatSetup, gate: &Deferred, result: ToolExecutionResult) {
    let gate = gate.clone();
    add_tool(
        &setup.registry,
        define_tool(
            "hold",
            "Waits for the test",
            json!({ "type": "object", "properties": {} }),
            move |_, _, _| {
                let (gate, result) = (gate.clone(), result.clone());
                async move {
                    gate.wait().await;
                    Ok(result)
                }
            },
        ),
    );
}

fn empty_result() -> ToolExecutionResult {
    ToolExecutionResult {
        content: Some(Vec::new()),
        ..ToolExecutionResult::default()
    }
}

fn control_result(control: ToolControl) -> ToolExecutionResult {
    ToolExecutionResult {
        control: Some(control),
        ..empty_result()
    }
}

fn hold_call() -> FauxResponseStep {
    calls(&[("hold", "c1")])
}

fn calls(list: &[(&str, &str)]) -> FauxResponseStep {
    faux_assistant_message(
        list.iter()
            .map(|(name, id)| faux_tool_call(*name, json!({}), Some(id)))
            .collect::<Vec<_>>(),
        FauxMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxMessageOptions::default()
        },
    )
    .into()
}

async fn status(submission: &Submission) -> SubmissionRecord {
    submission.status(&context()).await.unwrap()
}

async fn submit(root: &Conversation, draft: SubmissionDraft) -> Submission {
    root.submit(draft, &context()).await.unwrap()
}

fn input(text: &str) -> SubmissionDraft {
    SubmissionDraft::input(text)
}

fn write(kind: &str) -> SubmissionDraft {
    SubmissionDraft::write(EntryDraft::new(kind))
}

fn head_write(kind: &str, head: crate::durable::ids::EntryId) -> SubmissionDraft {
    SubmissionDraft::write(EntryDraft::new(kind).head(EntryHead::Id(head)))
}

/// Kind and text of each entry, skipping system entries.
fn transcript(entries: &[EntryRecord]) -> Vec<String> {
    entries
        .iter()
        .filter(|entry| entry.kind != "pi.system")
        .map(|entry| {
            let message = entry.model.as_ref().and_then(|model| model.first());
            let text = match message {
                Some(Message::ToolResult(_)) => None,
                other => text_of(other),
            };
            match text {
                Some(text) => format!("{}:{text}", entry.kind),
                None => entry.kind.clone(),
            }
        })
        .collect()
}

async fn transcript_of(root: &Conversation) -> Vec<String> {
    transcript(&all_entries(root).await)
}

async fn inbox(harness: &Harness, root: &Conversation) -> Vec<(SubmissionId, &'static str)> {
    harness
        .snapshot(&*INBOX_DOC, root.id, &context())
        .await
        .unwrap()
        .unwrap()
        .items
        .iter()
        .map(|item| {
            (
                item.id(),
                match item {
                    InboxItem::Steer { .. } => "steer",
                    InboxItem::FollowUp { .. } => "followUp",
                    InboxItem::Write { .. } => "write",
                },
            )
        })
        .collect()
}

async fn tool_running(harness: &Harness, root: &Conversation) {
    wait_for(|| async {
        live_state(harness, root)
            .await
            .and_then(|live| live.tools)
            .is_some_and(|tools| to_json(&tools[0].status) == json!("running"))
    })
    .await;
}

async fn assistants(root: &Conversation) -> Vec<EntryRecord> {
    all_entries(root)
        .await
        .into_iter()
        .filter(|entry| entry.kind == "pi.assistant")
        .collect()
}

fn assert_done(record: &SubmissionRecord, answer: Option<crate::durable::ids::EntryId>) {
    assert_eq!(record.status, SubmissionStatus::Done, "{record:?}");
    if answer.is_some() {
        assert_eq!(record.answer, answer, "{record:?}");
    }
}

fn assert_unanswered(record: &SubmissionRecord, reason: &str) {
    assert_eq!(record.status, SubmissionStatus::Unanswered, "{record:?}");
    assert_eq!(record.reason.as_deref(), Some(reason), "{record:?}");
}

fn on_yield(
    handler: impl Fn() -> Option<UserInput> + Send + Sync + 'static,
) -> crate::durable::harness::types::HookRegistration {
    let handler = Arc::new(handler);
    hook(
        &*GENERATION_TASK,
        GenerationHooks {
            on_yield: Some(Arc::new(move |_, _, _| {
                let handler = handler.clone();
                Box::pin(async move { Ok(handler()) })
            })),
            ..GenerationHooks::default()
        },
    )
}

fn first_yield_continues(text: &'static str) -> crate::durable::harness::types::HookRegistration {
    let yields = Arc::new(Mutex::new(0));
    on_yield(move || {
        let mut yields = yields.lock();
        *yields += 1;
        (*yields == 1).then(|| UserInput::Text(text.into()))
    })
}

fn memory() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new())
}

// ---- inbox ----

#[tokio::test]
async fn queues_busy_submissions_and_places_writes_before_user_items_at_the_final_boundary_one_follow_up_per_run()
 {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup
        .faux
        .set_responses([first.step(), step("second"), step("third")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    let f1 = submit(&root, input("f1")).await;
    let mut note = EntryDraft::new("note");
    note.data = Some(json!("w"));
    let write_submission = submit(&root, SubmissionDraft::write(note)).await;
    let f2 = submit(&root, input("f2").when_busy(WhenBusy::FollowUp)).await;
    for submission in [&f1, &write_submission, &f2] {
        assert_eq!(status(submission).await.status, SubmissionStatus::Queued);
    }
    assert_eq!(
        inbox(&harness, &root).await,
        [
            (f1.id, "followUp"),
            (write_submission.id, "write"),
            (f2.id, "followUp")
        ]
    );
    assert_eq!(transcript_of(&root).await, ["pi.user:a"]);

    first.release();
    f2.wait(&context()).await.unwrap();
    assert_eq!(
        transcript_of(&root).await,
        [
            "pi.user:a",
            "pi.assistant:first",
            "note",
            "pi.user:f1",
            "pi.assistant:second",
            "pi.user:f2",
            "pi.assistant:third",
        ]
    );
    let answers = assistants(&root).await;
    assert_done(&status(&input_submission).await, Some(answers[0].id));
    assert_done(&status(&write_submission).await, None);
    assert_done(&status(&f1).await, Some(answers[1].id));
    assert_done(&status(&f2).await, Some(answers[2].id));
    assert!(inbox(&harness, &root).await.is_empty());
    assert_eq!(live_state(&harness, &root).await, Some(Default::default()));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn places_every_follow_up_in_one_successor_run_with_follow_up_mode_all() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step(), step("both")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    setup.settings(|s| s.follow_up_mode = Some(QueueMode::All));
    submit(&root, input("a")).await;
    first.reached.wait().await;
    let f1 = submit(&root, input("f1")).await;
    let f2 = submit(&root, input("f2")).await;
    first.release();
    let settled = f2.wait(&context()).await.unwrap();
    let record = status(&f1).await;
    assert_eq!(
        (record.status, record.answer),
        (settled.status, settled.answer)
    );
    assert!(record.entry.is_some());
    assert_eq!(
        transcript_of(&root).await,
        [
            "pi.user:a",
            "pi.assistant:first",
            "pi.user:f1",
            "pi.user:f2",
            "pi.assistant:both",
        ]
    );
    assert_eq!(setup.faux.state().call_count, 2);
    harness.close(&context()).await.unwrap();
}

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
struct Marker {
    n: i64,
}

static MARKER_DOC: LazyLock<DocToken<Marker, SessionScope>> = LazyLock::new(|| {
    define_doc(DocDefinition::new(
        "test.marker",
        1,
        SessionScope,
        Marker::default,
    ))
    .unwrap()
});

#[tokio::test]
async fn reads_queue_modes_when_the_final_boundarys_commit_runs_on_the_session_line() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step(), step("both")]);
    let yielded = Deferred::default();
    let signal = yielded.clone();
    add_hooks(
        &setup.registry,
        on_yield(move || {
            signal.resolve();
            None
        }),
    );
    let storage = Arc::new(ControlledStorage::new());
    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    submit(&root, input("a")).await;
    first.reached.wait().await;
    let f1 = submit(&root, input("f1")).await;
    let f2 = submit(&root, input("f2")).await;
    // Occupy the line, let the answer queue its boundary commit behind it, then change the mode.
    let held = storage.hold_commits();
    let occupying = {
        let root = root.clone();
        tokio::spawn(async move {
            root.commit(
                |tx| async move { tx.doc(&*MARKER_DOC, ()).await?.edit(|marker| marker.n += 1) },
                &context(),
            )
            .await
        })
    };
    held.entered().await;
    first.release();
    yielded.wait().await;
    flush().await;
    setup.settings(|s| s.follow_up_mode = Some(QueueMode::All));
    held.release();
    occupying.await.unwrap().unwrap();
    let settled = f2.wait(&context()).await.unwrap();
    let record = status(&f1).await;
    assert_eq!(
        (record.status, record.answer),
        (settled.status, settled.answer)
    );
    assert_eq!(setup.faux.state().call_count, 2);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn adds_steers_to_the_run_at_the_post_tools_boundary_and_holds_follow_ups_for_the_final_boundary()
 {
    let setup = chat_setup();
    let gate = Deferred::default();
    hold_tool(&setup, &gate, empty_result());
    setup
        .faux
        .set_responses([hold_call(), step("after tools"), step("follow-up")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    tool_running(&harness, &root).await;
    let steer = submit(&root, input("s").when_busy(WhenBusy::Steer)).await;
    let follow_up = submit(&root, input("f")).await;
    gate.resolve();
    follow_up.wait(&context()).await.unwrap();
    assert_eq!(
        transcript_of(&root).await,
        [
            "pi.user:a",
            "pi.assistant",
            "pi.tool-result",
            "pi.user:s",
            "pi.assistant:after tools",
            "pi.user:f",
            "pi.assistant:follow-up",
        ]
    );
    let answers = assistants(&root).await;
    assert_done(&status(&input_submission).await, Some(answers[1].id));
    assert_done(&status(&steer).await, Some(answers[1].id));
    assert_done(&status(&follow_up).await, Some(answers[2].id));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn ends_the_run_at_a_queued_reset_after_tools_and_runs_earlier_follow_ups_in_the_new_context()
{
    let setup = chat_setup();
    let gate = Deferred::default();
    hold_tool(&setup, &gate, empty_result());
    let requests: Arc<Mutex<Vec<Vec<String>>>> = Arc::default();
    let sink = requests.clone();
    let record = FauxResponseStep::factory(move |request, _, _, _| {
        sink.lock().push(
            request
                .messages
                .iter()
                .map(|message| {
                    format!(
                        "{}:{}",
                        to_json(message)["role"].as_str().unwrap(),
                        text_of(Some(message)).unwrap_or_default()
                    )
                })
                .collect(),
        );
        Ok(answer("fresh"))
    });
    setup.faux.set_responses([hold_call(), record]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    tool_running(&harness, &root).await;
    let follow_up = submit(&root, input("f")).await;
    root.reset(None, &context()).await.unwrap();
    gate.resolve();
    follow_up.wait(&context()).await.unwrap();
    assert_unanswered(&status(&input_submission).await, "reset");
    assert_eq!(
        transcript_of(&root).await,
        [
            "pi.user:a",
            "pi.assistant",
            "pi.tool-result",
            "pi.reset",
            "pi.user:f",
            "pi.assistant:fresh",
        ]
    );
    let entries = all_entries(&root).await;
    let reset = entries
        .iter()
        .find(|entry| RESET_ENTRY.is(Some(entry)))
        .unwrap();
    assert_eq!(reset.head, Some(reset.id));
    // The follow-up's request starts at the reset: the follow-up, then the complete system baseline after the cut.
    assert_eq!(*requests.lock(), [vec!["user:f", "system:"]]);
    assert_eq!(setup.faux.state().call_count, 2);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn places_a_queued_reset_after_the_answer_at_the_final_boundary() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step()]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    root.reset(Some("handoff".into()), &context())
        .await
        .unwrap();
    first.release();
    input_submission.wait(&context()).await.unwrap();
    harness.wait_for_idle(&context()).await.unwrap();
    assert_done(&status(&input_submission).await, None);
    assert_eq!(
        transcript_of(&root).await,
        ["pi.user:a", "pi.assistant:first", "pi.reset:handoff"]
    );
    let messages = root.context(&context()).await.unwrap().messages;
    assert_eq!(messages.len(), 1);
    let message = to_json(&messages[0]);
    assert_eq!(
        (&message["role"], &message["content"]),
        (&json!("user"), &json!("handoff"))
    );
    assert!(message["timestamp"].is_number());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn resets_an_idle_conversation_at_once_with_or_without_handoff_text() {
    let setup = chat_setup();
    setup.set_now(|| 7);
    let (harness, root) = open_chat(memory(), &setup).await;
    let id = root.id;
    root.commit(
        move |tx| async move { tx.append_entry(id, EntryDraft::new("note")).await },
        &context(),
    )
    .await
    .unwrap();
    root.reset(None, &context()).await.unwrap();
    let view = root.context(&context()).await.unwrap();
    let head = view.head.unwrap();
    assert_eq!(head.kind, "pi.reset");
    assert!(head.model.is_none());
    assert!(view.messages.is_empty());
    root.reset(Some("carry on".into()), &context())
        .await
        .unwrap();
    let view = root.context(&context()).await.unwrap();
    let head = view.head.unwrap();
    assert_eq!(head.head, Some(head.id));
    assert_eq!(
        to_json(&view.messages),
        json!([{ "role": "user", "content": "carry on", "timestamp": 7 }])
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn makes_a_queued_head_write_stale_when_it_targets_an_entry_before_the_active_range() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step(), step("second")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let id = root.id;
    let old = root
        .commit(
            move |tx| async move { tx.append_entry(id, EntryDraft::new("note")).await },
            &context(),
        )
        .await
        .unwrap();
    root.reset(None, &context()).await.unwrap();
    let reset = root.context(&context()).await.unwrap().head.unwrap();
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    let stale = submit(&root, head_write("summary", old.id)).await;
    let fresh = submit(&root, head_write("summary", reset.id)).await;
    first.release();
    input_submission.wait(&context()).await.unwrap();
    assert_unanswered(&status(&stale).await, "stale");
    assert_done(&status(&fresh).await, None);
    assert!(inbox(&harness, &root).await.is_empty());

    // The fresh summary's marker starts the range at the reset: a target inside the range is not stale, even when it
    // is older than the marker itself.
    let inside = all_entries(&root)
        .await
        .into_iter()
        .find(|entry| entry.kind == "pi.user")
        .unwrap();
    let second = submit(&root, input("b")).await;
    let kept = submit(&root, head_write("summary", inside.id)).await;
    second.wait(&context()).await.unwrap();
    assert_done(&status(&kept).await, None);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn makes_a_head_write_stale_behind_a_reset_placed_earlier_in_the_same_boundary() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step()]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    let target = all_entries(&root).await[0].clone();
    root.reset(None, &context()).await.unwrap();
    let summary = submit(&root, head_write("summary", target.id)).await;
    first.release();
    input_submission.wait(&context()).await.unwrap();
    assert_unanswered(&status(&summary).await, "stale");
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn ends_the_run_with_a_pi_reset_entry_when_a_tool_requests_a_handoff() {
    let setup = chat_setup();
    let gate = Deferred::default();
    gate.resolve();
    hold_tool(
        &setup,
        &gate,
        control_result(ToolControl {
            handoff: Some("continue here".into()),
            ..ToolControl::default()
        }),
    );
    setup.faux.set_responses([hold_call()]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let settled = submit(&root, input("a"))
        .await
        .wait(&context())
        .await
        .unwrap();
    let entries = all_entries(&root).await;
    let calling = entries
        .iter()
        .find(|entry| entry.kind == "pi.assistant")
        .unwrap();
    assert_done(&settled, Some(calling.id));
    assert_eq!(
        transcript(&entries),
        [
            "pi.user:a",
            "pi.assistant",
            "pi.tool-result",
            "pi.reset:continue here"
        ]
    );
    let last = entries.last().unwrap();
    assert_eq!(last.head, Some(last.id));
    assert_eq!(setup.faux.state().call_count, 1);
    assert_eq!(live_state(&harness, &root).await, Some(Default::default()));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn drops_an_on_yield_continuation_when_the_final_boundary_selects_a_follow_up() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step(), step("second")]);
    add_hooks(&setup.registry, first_yield_continues("more"));
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    let follow_up = submit(&root, input("f")).await;
    first.release();
    follow_up.wait(&context()).await.unwrap();
    assert_done(&status(&input_submission).await, None);
    assert_eq!(
        transcript_of(&root).await,
        [
            "pi.user:a",
            "pi.assistant:first",
            "pi.user:f",
            "pi.assistant:second"
        ]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn withdraws_a_queued_submission_and_removes_its_item() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step()]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    let kept = submit(&root, write("note")).await;
    let withdrawn = submit(&root, input("f")).await;
    assert_eq!(
        withdrawn.abort(&context()).await.unwrap(),
        AbortSubmissionResult::Aborted
    );
    assert_unanswered(&withdrawn.wait(&context()).await.unwrap(), "aborted");
    assert_eq!(inbox(&harness, &root).await, [(kept.id, "write")]);
    first.release();
    input_submission.wait(&context()).await.unwrap();
    assert_done(&status(&kept).await, None);
    assert_eq!(setup.faux.state().call_count, 1);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn leaves_queued_items_after_a_failed_run_until_the_next_submission_places_them_in_order() {
    let setup = chat_setup();
    let failing = Gated::new(failure("invalid request"));
    setup
        .faux
        .set_responses([failing.step(), step("for f"), step("for g")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    failing.reached.wait().await;
    let f = submit(&root, input("f")).await;
    failing.release();
    assert_unanswered(
        &input_submission.wait(&context()).await.unwrap(),
        "model_error",
    );
    harness.wait_for_idle(&context()).await.unwrap();
    assert_eq!(status(&f).await.status, SubmissionStatus::Queued);
    assert_eq!(inbox(&harness, &root).await, [(f.id, "followUp")]);

    // Idle with a queued item: the new input queues behind it, and a final boundary places the older one first.
    let g = submit(&root, input("g").when_busy(WhenBusy::Reject)).await;
    g.wait(&context()).await.unwrap();
    assert_done(&status(&f).await, None);
    let transcript = transcript_of(&root).await;
    assert_eq!(
        transcript[transcript.len() - 4..],
        [
            "pi.user:f",
            "pi.assistant:for f",
            "pi.user:g",
            "pi.assistant:for g"
        ]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_queued_submissions_across_reopen_and_settles_them_afterwards() {
    let storage = Arc::new(ControlledStorage::persistent());
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup
        .faux
        .set_responses([first.step(), step("first again"), step("f")]);
    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    submit(&root, input("a")).await;
    first.reached.wait().await;
    let f = submit(&root, input("f")).await.id;
    harness.close(&context()).await.unwrap();

    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    let settled = harness
        .submission(f, &context())
        .await
        .unwrap()
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_done(&settled, None);
    let transcript = transcript_of(&root).await;
    assert_eq!(
        transcript[transcript.len() - 2..],
        ["pi.user:f", "pi.assistant:f"]
    );
    harness.close(&context()).await.unwrap();
}

/// Operations of every `pi.inbox` commit with operations, from now on.
fn inbox_ops(harness: &Harness) -> Arc<Mutex<Vec<Vec<JsonValue>>>> {
    let ops: Arc<Mutex<Vec<Vec<JsonValue>>>> = Arc::default();
    let sink = ops.clone();
    let unsubscribe = harness
        .subscribe_commits(move |publication, _| {
            for change in &publication.changes {
                if let CommitChange::Document(change) = change
                    && change.record.kind == "pi.inbox"
                    && !change.ops.is_empty()
                {
                    sink.lock().push(change.ops.iter().map(to_json).collect());
                }
            }
        })
        .unwrap();
    std::mem::forget(unsubscribe);
    ops
}

#[tokio::test]
async fn commits_inbox_changes_as_positional_chord_operations_and_a_base_when_empty() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step(), step("second")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let ops = inbox_ops(&harness);
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    let w1 = submit(&root, write("note")).await;
    let f1 = submit(&root, input("f1")).await;
    let w2 = submit(&root, write("note")).await;
    let f2 = submit(&root, input("f2")).await;
    let w3 = submit(&root, write("note")).await;
    let note = |id: SubmissionId| json!({ "id": id, "mode": "write", "entry": { "kind": "note" } });
    let follow_up =
        |id: SubmissionId, text: &str| json!({ "id": id, "mode": "followUp", "content": text });
    assert_eq!(
        *ops.lock(),
        [
            vec![json!(["p", ["items"], 0, 0, [note(w1.id)]])],
            vec![json!(["p", ["items"], 1, 0, [follow_up(f1.id, "f1")]])],
            vec![json!(["p", ["items"], 2, 0, [note(w2.id)]])],
            vec![json!(["p", ["items"], 3, 0, [follow_up(f2.id, "f2")]])],
            vec![json!(["p", ["items"], 4, 0, [note(w3.id)]])],
        ]
    );
    first.release();
    input_submission.wait(&context()).await.unwrap();
    // Every write and the first follow-up leave; only f2 at index 3 remains. No retained value is carried.
    let removal = ops.lock()[5].clone();
    assert!(removal.iter().all(|op| op[0] == "p" && op[4] == json!([])));
    // Rust diffs the draft at prepare time, so the removals coalesce instead of following the TS splice order.
    let removed: u64 = removal.iter().map(|op| op[3].as_u64().unwrap()).sum();
    assert_eq!(removed, 4, "{removal:?}");
    assert!(!serde_json::to_string(&removal).unwrap().contains("f2"));
    f2.wait(&context()).await.unwrap();
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn settles_a_stale_write_at_once_while_idle_and_a_queued_one_behind_waiting_items() {
    let setup = chat_setup();
    let failing = Gated::new(failure("invalid request"));
    setup.faux.set_responses([failing.step(), step("for f")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let id = root.id;
    let old = root
        .commit(
            move |tx| async move { tx.append_entry(id, EntryDraft::new("note")).await },
            &context(),
        )
        .await
        .unwrap();
    root.reset(None, &context()).await.unwrap();
    let before = all_entries(&root).await.len();
    let idle = submit(&root, head_write("summary", old.id)).await;
    assert_unanswered(&idle.wait(&context()).await.unwrap(), "stale");
    assert_eq!(all_entries(&root).await.len(), before);

    // After a failed run, a follow-up waits; a stale write queues behind it and the boundary rejects it.
    let input_submission = submit(&root, input("a")).await;
    failing.reached.wait().await;
    let f = submit(&root, input("f")).await;
    failing.release();
    input_submission.wait(&context()).await.unwrap();
    harness.wait_for_idle(&context()).await.unwrap();
    let queued = submit(&root, head_write("summary", old.id)).await;
    assert_unanswered(&queued.wait(&context()).await.unwrap(), "stale");
    assert_done(&f.wait(&context()).await.unwrap(), None);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_an_on_yield_continuation_across_a_queued_plain_write_with_the_runs_original_input() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step(), step("second")]);
    add_hooks(&setup.registry, first_yield_continues("more"));
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    let write_submission = submit(&root, write("note")).await;
    first.release();
    input_submission.wait(&context()).await.unwrap();
    let entries = all_entries(&root).await;
    assert_eq!(
        transcript(&entries),
        [
            "pi.user:a",
            "pi.assistant:first",
            "note",
            "pi.user:more",
            "pi.assistant:second",
        ]
    );
    assert_done(
        &status(&input_submission).await,
        Some(entries.last().unwrap().id),
    );
    assert_done(&status(&write_submission).await, None);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn drops_an_on_yield_continuation_for_a_queued_reset() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step()]);
    add_hooks(
        &setup.registry,
        on_yield(|| Some(UserInput::Text("more".into()))),
    );
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    root.reset(None, &context()).await.unwrap();
    first.release();
    input_submission.wait(&context()).await.unwrap();
    harness.wait_for_idle(&context()).await.unwrap();
    assert_done(&status(&input_submission).await, None);
    assert_eq!(
        transcript_of(&root).await,
        ["pi.user:a", "pi.assistant:first", "pi.reset"]
    );
    assert_eq!(setup.faux.state().call_count, 1);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn adds_every_steer_to_the_run_at_the_post_tools_boundary_with_steering_mode_all() {
    let setup = chat_setup();
    let gate = Deferred::default();
    hold_tool(&setup, &gate, empty_result());
    setup
        .faux
        .set_responses([hold_call(), step("after tools"), step("follow-up")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    setup.settings(|s| s.steering_mode = Some(QueueMode::All));
    let input_submission = submit(&root, input("a")).await;
    tool_running(&harness, &root).await;
    let s1 = submit(&root, input("s1").when_busy(WhenBusy::Steer)).await;
    let f = submit(&root, input("f")).await;
    let s2 = submit(&root, input("s2").when_busy(WhenBusy::Steer)).await;
    gate.resolve();
    f.wait(&context()).await.unwrap();
    assert_eq!(
        transcript_of(&root).await,
        [
            "pi.user:a",
            "pi.assistant",
            "pi.tool-result",
            "pi.user:s1",
            "pi.user:s2",
            "pi.assistant:after tools",
            "pi.user:f",
            "pi.assistant:follow-up",
        ]
    );
    let answers = assistants(&root).await;
    for submission in [&input_submission, &s1, &s2] {
        assert_done(&status(submission).await, Some(answers[1].id));
    }
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn queues_an_idle_steer_behind_waiting_items_and_places_it_with_the_first_follow_up_in_id_order()
 {
    let setup = chat_setup();
    let failing = Gated::new(failure("invalid request"));
    setup.faux.set_responses([failing.step(), step("both")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    failing.reached.wait().await;
    let f = submit(&root, input("f")).await;
    failing.release();
    input_submission.wait(&context()).await.unwrap();
    harness.wait_for_idle(&context()).await.unwrap();
    let steer = submit(&root, input("s").when_busy(WhenBusy::Steer)).await;
    let settled = steer.wait(&context()).await.unwrap();
    assert_done(&status(&f).await, settled.answer);
    let transcript = transcript_of(&root).await;
    assert_eq!(
        transcript[transcript.len() - 3..],
        ["pi.user:f", "pi.user:s", "pi.assistant:both"]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn starts_a_queued_follow_up_after_a_terminating_round() {
    let setup = chat_setup();
    let gate = Deferred::default();
    hold_tool(
        &setup,
        &gate,
        control_result(ToolControl {
            terminate: Some(true),
            ..ToolControl::default()
        }),
    );
    setup.faux.set_responses([hold_call(), step("follow-up")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    tool_running(&harness, &root).await;
    let f = submit(&root, input("f")).await;
    gate.resolve();
    f.wait(&context()).await.unwrap();
    let calling = assistants(&root).await[0].id;
    assert_done(&status(&input_submission).await, Some(calling));
    assert_eq!(
        transcript_of(&root).await,
        [
            "pi.user:a",
            "pi.assistant",
            "pi.tool-result",
            "pi.user:f",
            "pi.assistant:follow-up",
        ]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn writes_the_last_handoff_in_call_order_and_then_runs_queued_follow_ups_in_the_new_context()
{
    let setup = chat_setup();
    let (first_gate, second_gate) = (Deferred::default(), Deferred::default());
    hold_tool(
        &setup,
        &first_gate,
        control_result(ToolControl {
            handoff: Some("one".into()),
            ..ToolControl::default()
        }),
    );
    let gate = second_gate.clone();
    add_tool(
        &setup.registry,
        define_tool(
            "later",
            "Finishes first",
            json!({ "type": "object", "properties": {} }),
            move |_, _, _| {
                let gate = gate.clone();
                async move {
                    gate.wait().await;
                    Ok(control_result(ToolControl {
                        handoff: Some("two".into()),
                        ..ToolControl::default()
                    }))
                }
            },
        ),
    );
    setup
        .faux
        .set_responses([calls(&[("hold", "c1"), ("later", "c2")]), step("follow-up")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    tool_running(&harness, &root).await;
    let f = submit(&root, input("f")).await;
    second_gate.resolve();
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.tools)
            .is_some_and(|tools| tools.len() > 1 && to_json(&tools[1].status) == json!("done"))
    })
    .await;
    first_gate.resolve();
    f.wait(&context()).await.unwrap();
    assert_done(&status(&input_submission).await, None);
    let transcript = transcript_of(&root).await;
    assert_eq!(
        transcript[transcript.len() - 4..],
        [
            "pi.tool-result",
            "pi.reset:two",
            "pi.user:f",
            "pi.assistant:follow-up"
        ]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn leaves_the_inbox_alone_when_the_runs_task_is_aborted() {
    let setup = chat_setup();
    let first = Gated::new(answer("never"));
    setup.faux.set_responses([first.step()]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    let f = submit(&root, input("f")).await;
    let task_id = live_state(&harness, &root)
        .await
        .unwrap()
        .run
        .unwrap()
        .task_id;
    harness.abort_task(task_id, &context()).await.unwrap();
    assert_unanswered(&input_submission.wait(&context()).await.unwrap(), "aborted");
    harness.wait_for_idle(&context()).await.unwrap();
    assert_eq!(status(&f).await.status, SubmissionStatus::Queued);
    assert_eq!(inbox(&harness, &root).await, [(f.id, "followUp")]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn returns_a_queued_submission_for_its_repeated_request_id_without_a_second_item() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step()]);
    let (harness, root) = open_chat(memory(), &setup).await;
    submit(&root, input("a")).await;
    first.reached.wait().await;
    let queued = submit(&root, input("f").with_request_id("r")).await;
    let again = submit(&root, input("f").with_request_id("r")).await;
    assert_eq!(again.id, queued.id);
    assert_eq!(inbox(&harness, &root).await, [(queued.id, "followUp")]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn withdraws_a_middle_item_with_one_positional_removal() {
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step()]);
    let (harness, root) = open_chat(memory(), &setup).await;
    submit(&root, input("a")).await;
    first.reached.wait().await;
    let mut items = Vec::new();
    for text in ["x", "y", "z"] {
        items.push(submit(&root, input(text)).await);
    }
    let ops = inbox_ops(&harness);
    items[1].abort(&context()).await.unwrap();
    assert_eq!(*ops.lock(), [vec![json!(["p", ["items"], 1, 1, []])]]);
    assert_eq!(
        inbox(&harness, &root).await,
        [(items[0].id, "followUp"), (items[2].id, "followUp")]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn stores_the_inbox_as_a_base_exactly_when_it_becomes_empty() {
    let storage = Arc::new(ControlledStorage::new());
    let setup = chat_setup();
    let first = Gated::new(answer("first"));
    setup.faux.set_responses([first.step(), step("second")]);
    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    let input_submission = submit(&root, input("a")).await;
    first.reached.wait().await;
    submit(&root, input("f1")).await;
    let f2 = submit(&root, input("f2")).await;
    first.release();
    input_submission.wait(&context()).await.unwrap();
    f2.wait(&context()).await.unwrap();
    // The inbox's document ID, from its creation.
    let commits = storage.commits.lock().clone();
    let inbox_doc = commits
        .iter()
        .flatten()
        .find_map(|write| match write {
            StorageWrite::DocumentCreate { record, .. } if record.kind == "pi.inbox" => {
                Some(record.id)
            }
            _ => None,
        })
        .unwrap();
    let written: Vec<(&str, bool)> = commits
        .iter()
        .flatten()
        .filter_map(|write| match write {
            StorageWrite::DocumentChange { id, content } if *id == inbox_doc => {
                Some(match content {
                    DocumentContent::Base { value, .. } => {
                        ("base", value["items"].as_array().is_some_and(Vec::is_empty))
                    }
                    DocumentContent::Delta { .. } => ("delta", false),
                })
            }
            _ => None,
        })
        .collect();
    // The f1 push, the f2 push, removing f1, and emptying the inbox.
    assert_eq!(
        written,
        [
            ("delta", false),
            ("delta", false),
            ("delta", false),
            ("base", true)
        ]
    );
    harness.close(&context()).await.unwrap();
}

#[test]
fn keeps_a_complete_inbox_base_exactly_while_it_is_empty_and_a_usage_base_on_every_change() {
    let info = || CheckpointInfo {
        deltas_since_base: 1000,
    };
    let inbox = INBOX_DOC.definition().clone();
    let usage = USAGE_DOC.definition().clone();
    assert!(
        inbox
            .checkpoint_when(&json!({ "items": [] }), &[], info())
            .unwrap()
    );
    assert!(
        !inbox
            .checkpoint_when(
                &json!({ "items": [{ "id": 1, "mode": "followUp", "content": "x" }] }),
                &[],
                info()
            )
            .unwrap()
    );
    assert!(
        usage
            .checkpoint_when(&json!({ "models": {}, "tools": {} }), &[], info())
            .unwrap()
    );
}

// ---- usage ----

fn usage_of(input: u32, output: u32, cache_read: u32, cache_write: u32, cost: UsageCost) -> Usage {
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: input + output + cache_read + cache_write,
        cost,
        ..Usage::default()
    }
}

fn assistant_messages(entries: &[EntryRecord]) -> Vec<AssistantMessage> {
    entries
        .iter()
        .filter(|entry| entry.kind == "pi.assistant")
        .filter_map(|entry| match &entry.model.as_ref()?[0] {
            Message::Assistant(message) => Some(message.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn totals_assistant_usage_per_model_and_tool_usage_per_tool_and_sums_the_session() {
    let setup = chat_setup();
    let spent = usage_of(
        1,
        2,
        3,
        4,
        UsageCost {
            input: 0.1,
            output: 0.2,
            cache_read: 0.3,
            cache_write: 0.4,
            total: 1.0,
        },
    );
    let gate = Deferred::default();
    gate.resolve();
    hold_tool(
        &setup,
        &gate,
        ToolExecutionResult {
            usage: Some(spent.clone()),
            ..empty_result()
        },
    );
    setup
        .faux
        .set_responses([hold_call(), step("done"), step("other")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    submit(&root, input("a"))
        .await
        .wait(&context())
        .await
        .unwrap();
    let entries = all_entries(&root).await;
    let messages = assistant_messages(&entries);
    let sum = |field: fn(&Usage) -> u32| messages.iter().map(|m| field(&m.usage)).sum::<u32>();
    let usage = harness
        .snapshot(&*USAGE_DOC, root.id, &context())
        .await
        .unwrap()
        .unwrap();
    let model = &usage.models["faux/faux-1"];
    assert_eq!(
        (model.input, model.output, model.total_tokens),
        (sum(|u| u.input), sum(|u| u.output), sum(|u| u.total_tokens))
    );
    assert_eq!(usage.tools.len(), 1);
    assert_eq!(usage.tools["hold"], spent);
    let result = entries
        .iter()
        .find(|entry| entry.kind == "pi.tool-result")
        .unwrap();
    match &result.model.as_ref().unwrap()[0] {
        Message::ToolResult(result) => assert_eq!(result.usage.as_ref(), Some(&spent)),
        other => panic!("{other:?}"),
    }

    // A fork starts at zero; the Session total adds every conversation once.
    let fork = root
        .fork(
            entries.last().unwrap().id,
            ConversationCreateOptions::ownerless(),
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(
        to_json(
            &harness
                .snapshot(&*USAGE_DOC, fork.id, &context())
                .await
                .unwrap()
        ),
        json!({ "models": {}, "tools": {} })
    );
    submit(&fork, input("b"))
        .await
        .wait(&context())
        .await
        .unwrap();
    let fork_usage = harness
        .snapshot(&*USAGE_DOC, fork.id, &context())
        .await
        .unwrap()
        .unwrap()
        .models["faux/faux-1"]
        .clone();
    let total = harness.usage(&context()).await.unwrap();
    assert_eq!(
        total.models["faux/faux-1"].output,
        sum(|u| u.output) + fork_usage.output
    );
    assert_eq!(total.tools.len(), 1);
    assert_eq!(total.tools["hold"], spent);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn counts_failed_attempts_converted_partials_and_tool_usage_replaced_by_after_tool() {
    let setup = chat_setup_with(RegisterFauxProviderOptions {
        tokens_per_second: Some(200.0),
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    let spent = usage_of(5, 0, 0, 0, UsageCost::default());
    let gate = Deferred::default();
    gate.resolve();
    hold_tool(&setup, &gate, empty_result());
    let replaced = spent.clone();
    add_hooks(
        &setup.registry,
        hook(
            &*TOOL_TASK,
            ToolHooks {
                after_tool: Some(Arc::new(move |_, result, _, _| {
                    let usage = replaced.clone();
                    Box::pin(async move {
                        Ok(Some(ToolExecutionResult {
                            usage: Some(usage),
                            ..result
                        }))
                    })
                })),
                ..ToolHooks::default()
            },
        ),
    );
    setup.faux.set_responses([
        failure("503 Service Unavailable").into(),
        hold_call(),
        step("done"),
        step(&"x".repeat(400)),
    ]);
    let (harness, root) = open_chat(memory(), &setup).await;
    setup.settings(|s| {
        s.retry = Some(RetryPolicyOverrides {
            enabled: Some(true),
            max_retries: Some(1),
            base_delay_ms: Some(1),
            ..RetryPolicyOverrides::default()
        })
    });
    submit(&root, input("a"))
        .await
        .wait(&context())
        .await
        .unwrap();
    let usage = harness
        .snapshot(&*USAGE_DOC, root.id, &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(usage.tools.len(), 1);
    assert_eq!(usage.tools["hold"], spent);

    // A partial committed while streaming becomes an aborted entry on abort; its usage counts too.
    let pending = submit(&root, input("b")).await;
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.generation)
            .is_some_and(|generation| generation.message.is_some())
    })
    .await;
    let task_id = live_state(&harness, &root)
        .await
        .unwrap()
        .run
        .unwrap()
        .task_id;
    harness.abort_task(task_id, &context()).await.unwrap();
    pending.wait(&context()).await.unwrap();
    let messages = assistant_messages(&all_entries(&root).await);
    assert_eq!(
        messages
            .iter()
            .map(|message| message.stop_reason)
            .collect::<Vec<_>>(),
        [
            StopReason::Error,
            StopReason::ToolUse,
            StopReason::Stop,
            StopReason::Aborted
        ]
    );
    let output: u32 = messages.iter().map(|message| message.usage.output).sum();
    assert_eq!(
        harness
            .snapshot(&*USAGE_DOC, root.id, &context())
            .await
            .unwrap()
            .unwrap()
            .models["faux/faux-1"]
            .output,
        output
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_tools_named_like_object_prototype_keys_in_the_ledger_and_the_session_total() {
    let (harness, root) = open_chat(memory(), &chat_setup()).await;
    let usage = usage_of(1, 1, 0, 0, UsageCost::default());
    let names = ["constructor", "__proto__", "toString"];
    for name in names {
        for _ in 0..2 {
            let (id, usage) = (root.id, usage.clone());
            root.commit(
                move |tx| async move {
                    record_usage(&tx, id, UsageBucket::Tools, name, &usage).await
                },
                &context(),
            )
            .await
            .unwrap();
        }
    }
    let total = harness.usage(&context()).await.unwrap();
    assert_eq!(
        total.tools.keys().map(String::as_str).collect::<Vec<_>>(),
        names
    );
    for name in names {
        assert_eq!(total.tools[name].output, 2);
    }
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_usage_totals_exact_across_reopen_counting_a_partial_converted_after_reopen_once() {
    let storage = Arc::new(ControlledStorage::persistent());
    let setup = chat_setup_with(RegisterFauxProviderOptions {
        tokens_per_second: Some(200.0),
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    setup
        .faux
        .set_responses([step("first"), step(&"x".repeat(400)), step("again")]);
    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    submit(&root, input("a"))
        .await
        .wait(&context())
        .await
        .unwrap();
    submit(&root, input("b")).await;
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.generation)
            .is_some_and(|generation| generation.message.is_some())
    })
    .await;
    // Closing mid-stream keeps the committed partial; the reopened request converts it into an aborted entry.
    harness.close(&context()).await.unwrap();
    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    harness.resume().unwrap();
    harness.wait_for_idle(&context()).await.unwrap();
    let messages = assistant_messages(&all_entries(&root).await);
    assert_eq!(
        messages
            .iter()
            .map(|message| message.stop_reason)
            .collect::<Vec<_>>(),
        [StopReason::Stop, StopReason::Aborted, StopReason::Stop]
    );
    let total = harness.usage(&context()).await.unwrap().models["faux/faux-1"].clone();
    let sum = |field: fn(&Usage) -> u32| messages.iter().map(|m| field(&m.usage)).sum::<u32>();
    assert_eq!(
        (total.input, total.output, total.total_tokens),
        (sum(|u| u.input), sum(|u| u.output), sum(|u| u.total_tokens))
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn records_usage_as_numeric_sets_on_the_ledger_in_the_entrys_commit() {
    let setup = chat_setup();
    setup.faux.set_responses([step("one"), step("two")]);
    let (harness, root) = open_chat(memory(), &setup).await;
    let commits: Arc<Mutex<Vec<(usize, Vec<JsonValue>)>>> = Arc::default();
    let sink = commits.clone();
    let unsubscribe = harness
        .subscribe_commits(move |publication, _| {
            for change in &publication.changes {
                if let CommitChange::Document(change) = change
                    && change.record.kind == "pi.usage"
                {
                    let entries = publication
                        .changes
                        .iter()
                        .filter(|other| matches!(other, CommitChange::Entry(_)))
                        .count();
                    sink.lock()
                        .push((entries, change.ops.iter().map(to_json).collect()));
                }
            }
        })
        .unwrap();
    submit(&root, input("a"))
        .await
        .wait(&context())
        .await
        .unwrap();
    submit(&root, input("b"))
        .await
        .wait(&context())
        .await
        .unwrap();
    drop(unsubscribe);
    let commits = commits.lock().clone();
    // The root's creation also publishes its empty ledger; only updates carry operations.
    let updates: Vec<_> = commits.iter().filter(|(_, ops)| !ops.is_empty()).collect();
    assert_eq!(updates.len(), 2, "{commits:?}");
    assert_eq!(updates[0].1.len(), 1);
    assert_eq!(updates[0].1[0][0], json!("s"));
    assert_eq!(updates[0].1[0][1], json!(["models", "faux/faux-1"]));
    assert!(updates[0].1[0][2].is_object());
    assert!(
        updates[1]
            .1
            .iter()
            .all(|op| op[0] == "s" && op[1][1] == "faux/faux-1")
    );
    assert!(updates.iter().all(|(entries, _)| *entries >= 1));
    harness.close(&context()).await.unwrap();
}
