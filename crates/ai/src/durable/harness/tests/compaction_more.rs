//! Port of `test/harness-compaction.test.ts`, part 3: recovery, the inbox, edge cases, blocking and manual
//! compaction together, pinning, blocked compaction, and context contributions.
//!
//! Divergences: the reopened SQLite file is `ControlledStorage::persistent()`; the faux model window is 100k (TS: 128k)
//! and thresholds are scaled to it; the blocking compaction that faults is a panicking `beforeCompact` hook (TS: a
//! throwing `completeSimple`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use serde_json::{Value as JsonValue, json};

use super::chat::{
    ChatOptions, ChatSetup, all_entries, chat_setup, chat_setup_with, open_chat_with, wait_for,
};
use super::compaction_auto::{event_types, flat, record_events, settled_json, submit, task_state};
use super::compaction_support::*;
use super::support::{add_section, context, user};
use crate::durable::errors::StorageRejected;
use crate::durable::harness::compaction::{COMPACTION_TASK, CompactionInput};
use crate::durable::harness::events::watch_events;
use crate::durable::harness::live::{CompactionStatus, LIVE_DOC};
use crate::durable::harness::section;
use crate::durable::harness::submissions::Submission;
use crate::durable::harness::types::{
    AgentChange, BlockedReason, CompactionDecision, CompactionPolicy, CompactionReason,
    CompactionResult, ConversationAbortOptions, ModelRef, RetryPolicyOverrides, SubmissionDraft,
    TaskInspectionState,
};
use crate::durable::harness::usage::USAGE_DOC;
use crate::durable::ids::TaskId;
use crate::durable::session::tests::support::ControlledStorage;
use crate::durable::tasks::{TaskDefinition, define_task};
use crate::durable::types::{
    ContextEdit, ContextEditAction, EntryDraft, EntryHead, Storage,
    TaskOptions as CreateTaskOptions,
};
use crate::providers::faux::{
    FauxMessageOptions, FauxModelDefinition, RegisterFauxProviderOptions, faux_assistant_message,
};
use crate::types::StopReason;

async fn submission(chat: &Chat, id: crate::durable::ids::SubmissionId) -> Submission {
    chat.harness
        .submission(id, &context())
        .await
        .unwrap()
        .unwrap()
}

async fn status_of(submission: &Submission) -> String {
    status(&json_of(&submission.status(&context()).await.unwrap()))
}

async fn usage_json(chat: &Chat) -> JsonValue {
    json_of(
        &chat
            .harness
            .snapshot(&*USAGE_DOC, chat.root.id, &context())
            .await
            .unwrap(),
    )
}

fn has_compaction(kinds: &[String]) -> bool {
    kinds.iter().any(|kind| kind == "pi.compaction")
}

// ─── Compaction recovery ─────────────────────────────────────────────────

async fn reopen(storage: &Arc<ControlledStorage>, setup: ChatSetup, faux: Script) -> Chat {
    let storage: Arc<dyn Storage> = storage.clone();
    let (harness, root) = open_chat_with(storage, &setup, ChatOptions::default()).await;
    harness.resume().unwrap();
    Chat {
        harness,
        root,
        setup,
        faux,
    }
}

async fn first(storage: &Arc<ControlledStorage>) -> Chat {
    let chat = open(OpenOptions {
        storage: Some(storage.clone()),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    chat
}

async fn reopened(chat: Chat, storage: &Arc<ControlledStorage>) -> Chat {
    chat.close().await;
    reopen(storage, chat.setup, chat.faux).await
}

#[tokio::test]
async fn repeats_nothing_after_reopen_once_the_summary_is_placed() {
    let storage = Arc::new(ControlledStorage::persistent());
    let chat = first(&storage).await;
    chat.faux.summary(summary("SUMMARY"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    let entries = all_entries(&chat.root).await;
    let usage = usage_json(&chat).await;
    let chat = reopened(chat, &storage).await;
    chat.harness.wait_for_idle(&context()).await.unwrap();
    assert_eq!(chat.faux.summary_requests().len(), 1);
    assert_eq!(all_entries(&chat.root).await, entries);
    assert_eq!(usage_json(&chat).await, usage);
    let inspection = chat.harness.inspect(&context()).await.unwrap();
    assert!(inspection.tasks.is_empty());
    assert!(inspection.submissions.is_empty());
    chat.close().await;
}

#[tokio::test]
async fn fails_an_overflow_run_with_its_text_when_its_compaction_fails_after_reopen() {
    let storage = Arc::new(ControlledStorage::persistent());
    let chat = first(&storage).await;
    chat.policy(CompactionPolicy {
        enabled: true,
        ..MANUAL
    });
    let reached = Deferred::default();
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    chat.faux.agent(failure(OVERFLOW));
    let input = submit(&chat, &text("u4", 100)).await;
    reached.wait().await;
    let generation = live(&chat).await.run.unwrap().task_id;
    let state = task_state(&chat, generation).await;
    assert_eq!(state["status"], "waiting");
    assert_eq!(state["checkpoint"]["phase"], "prepare");
    assert_eq!(state["checkpoint"]["overflow"], OVERFLOW);
    let chat = reopened(chat, &storage).await;
    chat.faux.summary(failure("bad request"));
    let settled = settled_json(&submission(&chat, input.id).await).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "model_error");
    assert_eq!(settled["detail"], OVERFLOW);
    chat.close().await;
}

#[tokio::test]
async fn reruns_select_and_its_hook_after_a_crash_in_select() {
    let storage = Arc::new(ControlledStorage::persistent());
    let chat = first(&storage).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let reached = Deferred::default();
    {
        let (calls, reached) = (calls.clone(), reached.clone());
        super::support::add_hooks(
            &chat.setup.registry,
            crate::durable::harness::hook(
                &*COMPACTION_TASK,
                crate::durable::harness::types::CompactionHooks {
                    before_compact: Some(Arc::new(move |_, _, hook_context| {
                        let (calls, reached) = (calls.clone(), reached.clone());
                        Box::pin(async move {
                            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                                reached.resolve();
                                super::support::aborted(
                                    hook_context.abort_signal().unwrap().clone(),
                                )
                                .await?;
                            }
                            Ok(None)
                        })
                    })),
                },
            ),
        );
    }
    let id = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    let chat = reopened(chat, &storage).await;
    chat.faux.summary(summary("SUMMARY"));
    assert_eq!(outcome_status(&result(&chat, id).await), "completed");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    chat.close().await;
}

#[tokio::test]
async fn resends_an_interrupted_summarization_once_and_counts_only_the_answered_attempt() {
    let storage = Arc::new(ControlledStorage::persistent());
    let chat = first(&storage).await;
    let reached = Deferred::default();
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    let id = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    let before = usage_json(&chat).await["models"]["faux/faux-1"].clone();
    assert_eq!(
        task_state(&chat, id.erase()).await["checkpoint"]["phase"],
        "summarize"
    );
    let chat = reopened(chat, &storage).await;
    chat.faux.summary(summary("SUMMARY"));
    assert_eq!(outcome_status(&result(&chat, id).await), "completed");
    let requests = chat.faux.summary_requests();
    assert_eq!(requests.len(), 2);
    let without_timestamps = |messages: &[crate::types::Message]| {
        messages
            .iter()
            .map(|message| {
                let mut value = json_of(message);
                value["timestamp"] = JsonValue::Null;
                value
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        without_timestamps(&requests[1].messages),
        without_timestamps(&requests[0].messages)
    );
    let usage = usage_json(&chat).await["models"]["faux/faux-1"].clone();
    assert!(usage["input"].as_u64().unwrap() > before["input"].as_u64().unwrap());
    assert_eq!(
        usage["output"].as_u64().unwrap() - before["output"].as_u64().unwrap(),
        2
    );
    chat.close().await;
}

#[tokio::test]
async fn resumes_a_retry_backoff_after_reopen() {
    let storage = Arc::new(ControlledStorage::persistent());
    let chat = first(&storage).await;
    let now = Arc::new(AtomicU64::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    ));
    {
        let now = now.clone();
        chat.setup.set_now(move || now.load(Ordering::SeqCst));
    }
    chat.setup.settings(|settings| {
        settings.retry = Some(RetryPolicyOverrides {
            enabled: Some(true),
            max_retries: Some(2),
            base_delay_ms: Some(60_000),
            max_agent_delay_ms: None,
        })
    });
    chat.faux.summary(failure("overloaded"));
    let id = chat.root.compact(None, &context()).await.unwrap();
    {
        let chat = chat.clone();
        wait_for(move || {
            let chat = chat.clone();
            async move {
                live(&chat)
                    .await
                    .compactions
                    .and_then(|compactions| compactions.first().cloned())
                    .is_some_and(|status| status.retry.is_some())
            }
        })
        .await;
    }
    let checkpoint = task_state(&chat, id.erase()).await["checkpoint"].clone();
    assert_eq!(checkpoint["phase"], "retry");
    assert_eq!(checkpoint["attempt"], 1);
    chat.close().await;
    now.fetch_add(120_000, Ordering::SeqCst);
    let chat = reopen(&storage, chat.setup, chat.faux).await;
    chat.faux.summary(summary("SUMMARY"));
    assert_eq!(outcome_status(&result(&chat, id).await), "completed");
    assert_eq!(chat.faux.summary_requests().len(), 2);
    chat.close().await;
}

#[tokio::test]
async fn keeps_a_generation_waiting_on_its_blocking_compaction_across_reopen() {
    let storage = Arc::new(ControlledStorage::persistent());
    let chat = first(&storage).await;
    let reached = Deferred::default();
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    // The faux model window is 100k; this blocking threshold needs a small context window.
    chat.policy(CompactionPolicy {
        reserve_tokens: 100_000 - 700,
        ..BLOCKING
    });
    let input = submit(&chat, &text("u4", 200)).await;
    reached.wait().await;
    let chat = reopened(chat, &storage).await;
    chat.faux.summary(summary("SUMMARY"));
    chat.faux.agent(answer("a4"));
    assert_eq!(
        settled_json(&submission(&chat, input.id).await).await["status"],
        "done"
    );
    let kinds = kinds(&chat.root).await;
    assert_eq!(
        kinds[kinds.len() - 3..],
        ["pi.compaction", "pi.system", "pi.assistant"]
    );
    chat.close().await;
}

#[tokio::test]
async fn keeps_a_queued_summary_across_reopen_and_places_it_at_the_next_boundary() {
    let storage = Arc::new(ControlledStorage::persistent());
    let chat = first(&storage).await;
    let reached = Deferred::default();
    chat.faux.agent(gated(
        Deferred::default(),
        answer("never"),
        Some(reached.clone()),
    ));
    let input = submit(&chat, "busy").await;
    reached.wait().await;
    chat.faux.summary(summary("SUMMARY"));
    let outcome = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    let chat = reopened(chat, &storage).await;
    chat.faux.agent(answer("answered"));
    assert_eq!(
        settled_json(&submission(&chat, input.id).await).await["status"],
        "done"
    );
    assert_eq!(
        status(&submission_settled(&chat, submission_id(&outcome).unwrap()).await),
        "done"
    );
    chat.close().await;
}

// ─── Compaction and the inbox ────────────────────────────────────────────

async fn queued_summary(chat: &Chat, content: &str) -> Submission {
    chat.faux.summary(summary(content));
    let outcome = result(chat, chat.root.compact(None, &context()).await.unwrap()).await;
    submission(chat, submission_id(&outcome).unwrap()).await
}

struct Busy {
    input: Submission,
    gate: Deferred,
}

async fn busy(chat: &Chat, reply: crate::types::AssistantMessage) -> Busy {
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux
        .agent(gated(gate.clone(), reply, Some(reached.clone())));
    let input = submit(chat, "busy").await;
    reached.wait().await;
    Busy { input, gate }
}

#[tokio::test]
async fn places_a_reset_queued_after_the_summary_last_and_makes_a_summary_queued_after_a_reset_stale()
 {
    for summary_first in [true, false] {
        let chat = open(OpenOptions::default()).await;
        history(&chat).await;
        let run = busy(&chat, answer("done")).await;
        let queued = if summary_first {
            let queued = queued_summary(&chat, "SUMMARY").await;
            chat.root.reset(None, &context()).await.unwrap();
            queued
        } else {
            chat.root.reset(None, &context()).await.unwrap();
            queued_summary(&chat, "SUMMARY").await
        };
        run.gate.resolve();
        run.input.wait(&context()).await.unwrap();
        let settled = settled_json(&queued).await;
        assert_eq!(
            settled["status"],
            if summary_first { "done" } else { "unanswered" }
        );
        assert_eq!(kinds(&chat.root).await.last().unwrap(), "pi.reset");
        assert_eq!(
            chat.root
                .context(&context())
                .await
                .unwrap()
                .head
                .unwrap()
                .kind,
            "pi.reset"
        );
        chat.close().await;
    }
}

#[tokio::test]
async fn places_a_summary_left_queued_by_a_failed_run_at_the_next_submission_before_its_input() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let run = busy(&chat, failure("bad request")).await;
    let queued = queued_summary(&chat, "SUMMARY").await;
    run.gate.resolve();
    assert_eq!(settled_json(&run.input).await["status"], "unanswered");
    assert_eq!(status_of(&queued).await, "queued");
    turn(&chat, "again", "ok").await;
    assert_eq!(status_of(&queued).await, "done");
    let request = chat.faux.last_agent_messages();
    assert!(user_text(request.first()).contains("SUMMARY"));
    assert!(
        request
            .iter()
            .any(|message| user_text(Some(message)) == "again")
    );
    chat.close().await;
}

#[tokio::test]
async fn keeps_the_full_retry_budget_after_an_overflow_compaction() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    chat.policy(CompactionPolicy {
        enabled: true,
        ..MANUAL
    });
    chat.faux.summary(summary("SUMMARY"));
    chat.faux.agent(failure("prompt is too long"));
    chat.faux.agent(failure("overloaded"));
    chat.faux.agent(failure("overloaded"));
    chat.faux.agent(answer("finally"));
    let input = submit(&chat, &text("u4", 100)).await;
    assert_eq!(settled_json(&input).await["status"], "done");
    chat.close().await;
}

#[tokio::test]
async fn loses_an_application_edit_placed_while_a_compaction_summarizes() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let u1 = all_entries(&chat.root).await[0].clone();
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        gate.clone(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    let id = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    let mut draft = EntryDraft::new("app.redact");
    draft.edits = Some(vec![ContextEdit {
        target: u1.id,
        action: ContextEditAction::Replace {
            messages: vec![user("REDACTED")],
        },
    }]);
    chat.root
        .submit(SubmissionDraft::write(draft), &context())
        .await
        .unwrap();
    gate.resolve();
    result(&chat, id).await;
    // The summary was made from the unredacted entry, and the edit's target left the range.
    assert!(user_text(chat.faux.summary_requests()[0].messages.get(1)).contains("[User]: u1 "));
    assert!(
        !context_texts(&chat.root)
            .await
            .iter()
            .any(|text| text == "REDACTED")
    );
    chat.close().await;
}

#[tokio::test]
async fn keeps_the_kept_entries_mounted_in_the_conversation_view() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let state = chat.root.view_state(&context()).await.unwrap();
    chat.faux.summary(summary("SUMMARY"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    {
        let state = state.clone();
        wait_for(move || {
            let done = state
                .value()
                .entries
                .first()
                .is_some_and(|record| record.kind == "pi.compaction");
            async move { done }
        })
        .await;
    }
    assert_eq!(
        state.value().entries.to_vec(),
        chat.root.context(&context()).await.unwrap().entries
    );
    state.dispose().unwrap();
    chat.close().await;
}

// ─── Compaction edge cases ───────────────────────────────────────────────

#[tokio::test]
async fn keeps_the_one_compaction_limit_through_a_retry_backoff() {
    let chat = open(OpenOptions {
        context_window: Some(1000),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    chat.policy(BLOCKING);
    let asked = Arc::new(AtomicUsize::new(0));
    {
        let asked = asked.clone();
        add_before_compact(&chat.setup, move |_| {
            asked.fetch_add(1, Ordering::SeqCst);
            Ok(Some(CompactionDecision::Decline))
        });
    }
    chat.faux.agent(failure("overloaded"));
    turn(&chat, &text("u4", 200), "a4").await;
    // One blocking compaction, declined; the retried preparation, still above the threshold, started no other.
    assert_eq!(asked.load(Ordering::SeqCst), 1);
    assert_eq!(chat.faux.agent_requests().len(), 5);
    chat.close().await;
}

#[tokio::test]
async fn does_not_start_a_background_compaction_when_a_manual_one_was_admitted_during_preparation()
{
    let chat = open(OpenOptions {
        context_window: Some(2000),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    chat.policy(BACKGROUND);
    let (rendering, release) = (Deferred::default(), Deferred::default());
    let hold = Arc::new(AtomicBool::new(true));
    {
        let (rendering, release) = (rendering.clone(), release.clone());
        add_section(
            &chat.setup.registry,
            section(
                "slow",
                move |_, _| {
                    let (rendering, release, hold) =
                        (rendering.clone(), release.clone(), hold.clone());
                    async move {
                        if hold.swap(false, Ordering::SeqCst) {
                            rendering.resolve();
                            release.wait().await;
                        }
                        Ok(Some("slow".to_string()))
                    }
                },
                None,
            ),
        );
    }
    let reached = Deferred::default();
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    chat.faux.agent(answer("a4"));
    let input = submit(&chat, &text("u4", 100)).await;
    rendering.wait().await;
    let manual = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    release.resolve();
    assert_eq!(settled_json(&input).await["status"], "done");
    let ids: Vec<_> = compaction_tasks(&chat)
        .await
        .iter()
        .map(|task| task.id)
        .collect();
    assert_eq!(ids, vec![manual.erase()]);
    // A background compaction would have sent its own summarization request.
    assert_eq!(chat.faux.summary_requests().len(), 1);
    chat.harness.abort_task(manual, &context()).await.unwrap();
    chat.close().await;
}

#[tokio::test]
async fn starts_no_background_compaction_after_a_blocking_one_in_the_same_generation() {
    let chat = open(OpenOptions {
        context_window: Some(2000),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    // Blocking at 1500, background at 200: the kept part stays above the background threshold.
    chat.policy(CompactionPolicy {
        background_tokens: 1300,
        keep_recent_tokens: 400,
        ..BACKGROUND
    });
    chat.faux.summary(summary("SUMMARY"));
    turn(&chat, &text("u4", 1000), "a4").await;
    assert_eq!(chat.faux.summary_requests().len(), 1);
    assert!(compaction_tasks(&chat).await.is_empty());
    chat.close().await;
}

#[tokio::test]
async fn sends_the_request_anyway_when_its_blocking_compaction_faults() {
    let chat = open(OpenOptions {
        context_window: Some(1000),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    chat.policy(BLOCKING);
    let broken = add_before_compact(&chat.setup, |_| panic!("no credentials"));
    turn(&chat, &text("u4", 200), "a4").await;
    broken.dispose();
    assert!(!has_compaction(&kinds(&chat.root).await));
    assert!(live(&chat).await.compactions.is_none());
    chat.close().await;
}

#[tokio::test]
async fn fails_an_overflow_run_with_its_text_when_its_compaction_is_aborted_directly_or_itself_overflows()
 {
    for abort in [true, false] {
        let chat = open(OpenOptions::default()).await;
        history(&chat).await;
        chat.policy(CompactionPolicy {
            enabled: true,
            ..MANUAL
        });
        let reached = Deferred::default();
        if abort {
            chat.faux.summary(gated(
                Deferred::default(),
                summary("SUMMARY"),
                Some(reached.clone()),
            ));
        } else {
            chat.faux
                .summary(failure("prompt is too long for the summary"));
        }
        chat.faux.agent(failure(OVERFLOW));
        let input = submit(&chat, &text("u4", 100)).await;
        if abort {
            reached.wait().await;
            let child = compaction_tasks(&chat).await[0].clone();
            chat.harness.abort_task(child.id, &context()).await.unwrap();
        }
        let settled = settled_json(&input).await;
        assert_eq!(settled["status"], "unanswered");
        assert_eq!(settled["reason"], "model_error");
        assert_eq!(settled["detail"], OVERFLOW);
        assert!(live(&chat).await.compactions.is_none());
        chat.close().await;
    }
}

#[tokio::test]
async fn treats_a_length_stop_as_an_ordinary_answer_not_an_overflow() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    chat.policy(CompactionPolicy {
        enabled: true,
        ..MANUAL
    });
    chat.faux.agent(faux_assistant_message(
        "cut short",
        FauxMessageOptions {
            stop_reason: Some(StopReason::Length),
            error_message: Some(OVERFLOW.into()),
            ..FauxMessageOptions::default()
        },
    ));
    let input = submit(&chat, "go").await;
    assert_eq!(settled_json(&input).await["status"], "done");
    assert!(chat.faux.summary_requests().is_empty());
    chat.close().await;
}

#[tokio::test]
async fn orders_the_events_of_a_summary_placed_at_once() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (stream, batches) = record_events(&chat).await;
    chat.faux.summary(summary("SUMMARY"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    {
        let batches = batches.clone();
        wait_for(move || {
            let done = flat(&batches)
                .iter()
                .any(|event| event["type"] == "compaction_end");
            async move { done }
        })
        .await;
    }
    let types = event_types(&flat(&batches));
    let end = types
        .iter()
        .position(|kind| kind == "compaction_end")
        .unwrap();
    assert_eq!(
        types[end - 2..end + 3],
        [
            "message_start",
            "message_end",
            "compaction_end",
            "submission",
            "usage_changed"
        ]
    );
    stream.stop().await;
    chat.close().await;
}

#[tokio::test]
async fn puts_compaction_start_last_in_the_batch_of_the_preparation_commit() {
    let chat = open(OpenOptions {
        context_window: Some(2000),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    chat.policy(BACKGROUND);
    let (stream, batches) = record_events(&chat).await;
    // A new section makes the preparation commit append a system entry next to the compaction's status.
    super::chat::add_text_section(&chat.setup.registry, "extra", "extra", None);
    chat.faux
        .summary(gated(Deferred::default(), summary("SUMMARY"), None));
    turn(&chat, &text("u4", 100), "a4").await;
    let has_start = |batch: &Vec<JsonValue>| {
        batch
            .iter()
            .any(|event| event["type"] == "compaction_start")
    };
    {
        let batches = batches.clone();
        wait_for(move || {
            let done = batches.lock().iter().any(has_start);
            async move { done }
        })
        .await;
    }
    let batch = batches
        .lock()
        .iter()
        .find(|batch| has_start(batch))
        .cloned();
    assert_eq!(
        event_types(&batch.unwrap()),
        ["message_start", "message_end", "compaction_start"]
    );
    stream.stop().await;
    let task = compaction_tasks(&chat).await[0].id;
    chat.harness.abort_task(task, &context()).await.unwrap();
    chat.close().await;
}

#[tokio::test]
async fn keeps_a_background_summary_queued_through_a_retry_backoff_while_a_blocking_compaction_wins()
 {
    let chat = open(OpenOptions {
        context_window: Some(2000),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    chat.policy(BACKGROUND);
    chat.setup.settings(|settings| {
        settings.retry = Some(RetryPolicyOverrides {
            enabled: Some(true),
            max_retries: Some(2),
            base_delay_ms: Some(300),
            max_agent_delay_ms: None,
        })
    });
    let (summary_gate, summary_reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        summary_gate.clone(),
        summary("BACKGROUND"),
        Some(summary_reached.clone()),
    ));
    chat.faux.summary(summary("BLOCKING"));
    let failed = Deferred::default();
    {
        let failed = failed.clone();
        chat.faux.agent(step(move |_| async move {
            failed.resolve();
            failure("overloaded")
        }));
    }
    chat.faux.agent(answer("a4"));
    let input = submit(&chat, &text("u4", 100)).await;
    failed.wait().await;
    summary_reached.wait().await;
    // During the backoff: the background summary queues, and the thresholds drop so the retry blocks.
    {
        let chat = chat.clone();
        wait_for(move || {
            let chat = chat.clone();
            async move {
                live(&chat)
                    .await
                    .generation
                    .is_some_and(|generation| generation.retry.is_some())
            }
        })
        .await;
    }
    let background = compaction_tasks(&chat).await[0].clone();
    summary_gate.resolve();
    let queued = result(&chat, background.id.cast()).await;
    let queued = submission(&chat, submission_id(&queued).unwrap()).await;
    assert_eq!(status_of(&queued).await, "queued");
    chat.policy(CompactionPolicy {
        reserve_tokens: 1500,
        keep_recent_tokens: 50,
        ..BACKGROUND
    });
    assert_eq!(settled_json(&input).await["status"], "done");
    assert!(user_text(chat.faux.last_agent_messages().first()).contains("BLOCKING"));
    let settled = settled_json(&queued).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "stale");
    chat.close().await;
}

#[tokio::test]
async fn summarizes_the_previous_summary_in_a_second_compaction() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    chat.faux.summary(summary("FIRST"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    turn(&chat, &text("u4", 100), &text("a4", 100)).await;
    turn(&chat, &text("u5", 100), &text("a5", 100)).await;
    chat.faux.summary(summary("SECOND"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    let prompt = user_text(chat.faux.summary_requests()[1].messages.get(1));
    assert!(
        prompt.starts_with(
            "<conversation>\n[User]: The conversation history before this point was compacted"
        ),
        "{prompt}"
    );
    assert!(prompt.contains("FIRST"));
    assert!(context_texts(&chat.root).await[0].contains("SECOND"));
    chat.close().await;
}

#[tokio::test]
async fn leaves_no_usage_submission_summary_or_outcome_when_the_classifying_commit_is_rejected() {
    let storage = Arc::new(ControlledStorage::new());
    let chat = open(OpenOptions {
        storage: Some(storage.clone()),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    let before = usage_json(&chat).await["models"]["faux/faux-1"].clone();
    {
        let storage = storage.clone();
        chat.faux.summary(step(move |_| async move {
            storage.fail_next_commit(StorageRejected::new("rejected").into());
            summary("SUMMARY")
        }));
    }
    let id = chat.root.compact(None, &context()).await.unwrap();
    let outcome = result(&chat, id).await;
    assert_eq!(outcome_status(&outcome), "faulted");
    assert_eq!(usage_json(&chat).await["models"]["faux/faux-1"], before);
    assert!(!has_compaction(&kinds(&chat.root).await));
    assert!(
        chat.harness
            .inspect(&context())
            .await
            .unwrap()
            .submissions
            .is_empty()
    );
    assert!(
        storage
            .submission_by_request(chat.root.id, &format!("compaction:{id}"), &context())
            .await
            .unwrap()
            .is_none()
    );
    assert!(live(&chat).await.compactions.is_none());
    chat.close().await;
}

// ─── Blocking and manual compaction together ─────────────────────────────

struct BlockingRun {
    input: Submission,
    gate: Deferred,
    blocking: TaskId<CompactionResult>,
}

/// A run whose generation waits on a blocking compaction whose summary is held.
async fn blocking_run(chat: &Chat) -> BlockingRun {
    history(chat).await;
    chat.policy(BLOCKING);
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        gate.clone(),
        summary("BLOCKING"),
        Some(reached.clone()),
    ));
    let input = submit(chat, &text("u4", 200)).await;
    reached.wait().await;
    let blocking = compaction_tasks(chat).await[0].id.cast();
    BlockingRun {
        input,
        gate,
        blocking,
    }
}

fn small() -> OpenOptions {
    OpenOptions {
        context_window: Some(1000),
        ..OpenOptions::default()
    }
}

#[tokio::test]
async fn places_a_manual_summary_selected_before_the_blocking_one_landed_whose_equal_cut_replaces_it()
 {
    let chat = open(small()).await;
    let run = blocking_run(&chat).await;
    // Selected from the same context as the blocking compaction, so it cuts at the same entry.
    chat.faux.summary(summary("MANUAL"));
    let manual = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    let queued = submission(&chat, submission_id(&manual).unwrap()).await;
    assert_eq!(status_of(&queued).await, "queued");
    chat.faux.agent(answer("a4"));
    run.gate.resolve();
    assert_eq!(settled_json(&run.input).await["status"], "done");
    // The request after the blocking compaction used its summary; the manual one landed at the final boundary.
    assert!(user_text(chat.faux.last_agent_messages().first()).contains("BLOCKING"));
    assert_eq!(settled_json(&queued).await["status"], "done");
    let markers: Vec<_> = all_entries(&chat.root)
        .await
        .into_iter()
        .filter(|record| record.kind == "pi.compaction")
        .collect();
    assert_eq!(markers.len(), 2);
    assert_eq!(markers[1].head, markers[0].head);
    let texts = context_texts(&chat.root).await;
    assert!(texts[0].contains("MANUAL"));
    assert!(!texts.iter().any(|text| text.contains("BLOCKING")));
    chat.close().await;
}

#[tokio::test]
async fn finds_nothing_to_compact_for_a_manual_compaction_selected_after_the_blocking_summary_landed()
 {
    let chat = open(small()).await;
    let run = blocking_run(&chat).await;
    let (answer_gate, answer_reached) = (Deferred::default(), Deferred::default());
    chat.faux.agent(gated(
        answer_gate.clone(),
        answer("a4"),
        Some(answer_reached.clone()),
    ));
    run.gate.resolve();
    answer_reached.wait().await;
    let outcome = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    assert_eq!(
        json_of(&outcome),
        json!({ "status": "completed", "result": {} })
    );
    assert_eq!(chat.faux.summary_requests().len(), 1);
    answer_gate.resolve();
    assert_eq!(settled_json(&run.input).await["status"], "done");
    chat.close().await;
}

#[tokio::test]
async fn aborts_the_run_and_both_compactions_on_esc_and_appends_nothing() {
    let chat = open(small()).await;
    let run = blocking_run(&chat).await;
    let manual_reached = Deferred::default();
    chat.faux.summary(gated(
        Deferred::default(),
        summary("MANUAL"),
        Some(manual_reached.clone()),
    ));
    let manual = chat.root.compact(None, &context()).await.unwrap();
    manual_reached.wait().await;
    let blocking: Vec<bool> = live(&chat)
        .await
        .compactions
        .unwrap()
        .iter()
        .map(|status| status.blocking)
        .collect();
    assert_eq!(blocking, vec![true, false]);
    chat.root
        .abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    let settled = settled_json(&run.input).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "aborted");
    assert_eq!(
        outcome_status(&result(&chat, run.blocking).await),
        "aborted"
    );
    assert_eq!(outcome_status(&result(&chat, manual).await), "aborted");
    let state = live(&chat).await;
    assert!(state.compactions.is_none());
    assert!(state.run.is_none());
    assert!(!has_compaction(&kinds(&chat.root).await));
    assert!(
        chat.harness
            .inspect(&context())
            .await
            .unwrap()
            .submissions
            .is_empty()
    );
    chat.close().await;
}

#[tokio::test]
async fn places_a_summary_that_survived_esc_at_the_next_submission_before_its_input() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let reached = Deferred::default();
    chat.faux.agent(gated(
        Deferred::default(),
        answer("never"),
        Some(reached.clone()),
    ));
    let busy = submit(&chat, "busy").await;
    reached.wait().await;
    let queued = queued_summary(&chat, "SUMMARY").await;
    chat.root
        .abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    let settled = settled_json(&busy).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "aborted");
    assert_eq!(status_of(&queued).await, "queued");
    turn(&chat, "u5", "a5").await;
    assert_eq!(status_of(&queued).await, "done");
    let request = chat.faux.last_agent_messages();
    assert!(user_text(request.first()).contains("SUMMARY"));
    assert!(
        request
            .iter()
            .any(|message| user_text(Some(message)) == "u5")
    );
    let kinds = kinds(&chat.root).await;
    assert_eq!(
        kinds[kinds.len() - 4..],
        ["pi.compaction", "pi.user", "pi.system", "pi.assistant"]
    );
    chat.close().await;
}

// ─── Compaction pinning and late policy changes ──────────────────────────

#[tokio::test]
async fn keeps_the_pinned_model_through_a_retry_and_advances_the_live_attempt() {
    let models = ["faux-1", "faux-2"]
        .into_iter()
        .map(|id| {
            let mut model = FauxModelDefinition::new(id);
            model.context_window = Some(100_000);
            model.max_tokens = Some(900);
            model
        })
        .collect();
    let setup = chat_setup_with(RegisterFauxProviderOptions {
        models,
        ..RegisterFauxProviderOptions::default()
    });
    let faux = script(&setup);
    let chat = open_with(OpenOptions::default(), setup, faux).await;
    history(&chat).await;
    let attempt = Arc::new(parking_lot::Mutex::new(None));
    {
        let chat_ref = chat.clone();
        chat.faux.summary(step(move |_| async move {
            // Switched during the attempt: the retry still uses the pinned model.
            chat_ref
                .root
                .configure(
                    AgentChange::default().model(ModelRef::new("faux", "faux-2")),
                    &context(),
                )
                .await
                .unwrap();
            failure("overloaded")
        }));
    }
    {
        let (chat_ref, attempt) = (chat.clone(), attempt.clone());
        chat.faux.summary(step(move |_| async move {
            *attempt.lock() = live(&chat_ref)
                .await
                .compactions
                .and_then(|compactions| compactions.first().map(|status| status.attempt));
            summary("SUMMARY")
        }));
    }
    let outcome = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    assert_eq!(outcome_status(&outcome), "completed");
    let models: Vec<String> = chat
        .faux
        .summary_requests()
        .into_iter()
        .map(|request| request.model)
        .collect();
    assert_eq!(models, vec!["faux-1", "faux-1"]);
    assert_eq!(*attempt.lock(), Some(2));
    let usage = usage_json(&chat).await;
    let keys: Vec<&String> = usage["models"].as_object().unwrap().keys().collect();
    assert_eq!(keys, vec!["faux/faux-1"]);
    chat.close().await;
}

#[tokio::test]
async fn shows_a_late_joiner_a_compaction_that_is_summarizing() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let reached = Deferred::default();
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    let id = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    let stream = watch_events(&chat.harness, chat.root.id, &context())
        .await
        .unwrap();
    assert_eq!(
        json_of(&stream.snapshot)["compactions"],
        json!([{ "taskId": id, "reason": "manual", "blocking": false, "attempt": 1 }])
    );
    stream.stop().await;
    chat.harness.abort_task(id, &context()).await.unwrap();
    chat.close().await;
}

#[tokio::test]
async fn treats_silent_overflow_as_an_ordinary_answer() {
    let cases = [
        ("a stop whose input exceeds the window", answer("fine")),
        (
            "a length stop that fills the window without output",
            faux_assistant_message(
                "",
                FauxMessageOptions {
                    stop_reason: Some(StopReason::Length),
                    ..FauxMessageOptions::default()
                },
            ),
        ),
    ];
    for (name, response) in cases {
        let chat = open(OpenOptions {
            context_window: Some(300),
            ..OpenOptions::default()
        })
        .await;
        history(&chat).await;
        // Thresholds out of reach, so only overflow classification could compact.
        chat.policy(CompactionPolicy {
            enabled: true,
            reserve_tokens: -100_000,
            keep_recent_tokens: 150,
            background_tokens: 0,
        });
        chat.faux.agent(response);
        let input = submit(&chat, "go").await;
        assert_eq!(settled_json(&input).await["status"], "done", "{name}");
        let entries = all_entries(&chat.root).await;
        let last = entries.last().unwrap();
        let Some(crate::types::Message::Assistant(message)) =
            last.model.as_ref().and_then(|model| model.first())
        else {
            panic!("{name}: the last entry is not an answer");
        };
        assert!(message.usage.input >= 300, "{name}");
        assert!(chat.faux.summary_requests().is_empty(), "{name}");
        chat.close().await;
    }
}

#[tokio::test]
async fn sends_the_request_when_a_blocking_compaction_finds_nothing_under_a_policy_changed_after_preparation()
 {
    let chat = open(small()).await;
    history(&chat).await;
    chat.policy(BLOCKING);
    let changed = Arc::new(AtomicBool::new(false));
    {
        let setup = chat.setup.clone();
        add_section(
            &chat.setup.registry,
            section(
                "policy",
                move |_, _| {
                    // Rendering runs after preparation read the policy; the compaction reads this one.
                    if !changed.swap(true, Ordering::SeqCst) {
                        setup.settings(|settings| {
                            settings.compaction = Some(overrides(CompactionPolicy {
                                keep_recent_tokens: 100_000,
                                ..BLOCKING
                            }))
                        });
                    }
                    async { Ok(Some("p".to_string())) }
                },
                None,
            ),
        );
    }
    turn(&chat, &text("u4", 200), "a4").await;
    assert!(compaction_tasks(&chat).await.is_empty());
    assert!(chat.faux.summary_requests().is_empty());
    assert!(!has_compaction(&kinds(&chat.root).await));
    chat.close().await;
}

// ─── Blocked compaction ──────────────────────────────────────────────────

#[tokio::test]
async fn survives_reopen_blocked_and_is_orphaned_on_abort_with_its_status_removed() {
    let storage = Arc::new(ControlledStorage::persistent());
    let setup = chat_setup();
    let (harness, root) = open_chat_with(storage.clone(), &setup, ChatOptions::default()).await;
    // A compaction stored by a newer version than this process registers, for example after a downgrade.
    let newer = define_task(
        TaskDefinition::<CompactionInput, JsonValue, CompactionResult>::new(
            "pi.compaction",
            2,
            |_: &CompactionInput| json!({ "phase": "select" }),
        )
        .phase("select", |_, _, _| async { Ok(()) })
        .abort(|_, _, _| async { Ok(()) }),
    );
    let root_id = root.id;
    let id = root
        .commit(
            move |tx| async move {
                let task_id = tx
                    .create_task(
                        &newer,
                        CompactionInput {
                            reason: CompactionReason::Manual,
                            instructions: None,
                        },
                        CreateTaskOptions::conversation(None),
                    )
                    .await?;
                let live = tx.doc(&*LIVE_DOC, root_id).await?;
                live.edit(|live| {
                    live.compactions = Some(vec![CompactionStatus {
                        task_id: task_id.cast(),
                        reason: CompactionReason::Manual,
                        blocking: false,
                        attempt: 1,
                        retry: None,
                    }])
                })?;
                Ok(task_id.erase())
            },
            &context(),
        )
        .await
        .unwrap();
    harness.close(&context()).await.unwrap();
    let faux = script(&setup);
    let chat = reopen(&storage, setup, faux).await;
    let (stream, batches) = record_events(&chat).await;
    let inspection = chat.harness.inspect(&context()).await.unwrap();
    let state = inspection
        .tasks
        .iter()
        .find(|task| task.record.id == id)
        .map(|task| task.state.clone());
    assert!(matches!(
        state,
        Some(TaskInspectionState::Blocked {
            reason: BlockedReason::TaskTooOld,
            error: None
        })
    ));
    chat.harness.abort_task(id, &context()).await.unwrap();
    let record = chat.harness.wait_for_task(id, &context()).await.unwrap();
    assert_eq!(
        json_of(&record.state)["outcome"],
        json!({ "status": "orphaned", "reason": "task_too_old" })
    );
    assert!(live(&chat).await.compactions.is_none());
    {
        let batches = batches.clone();
        wait_for(move || {
            let done = flat(&batches)
                .iter()
                .any(|event| event["type"] == "compaction_end");
            async move { done }
        })
        .await;
    }
    stream.stop().await;
    chat.close().await;
}

// ─── Context contributions ───────────────────────────────────────────────

#[tokio::test]
async fn apply_edits_carried_by_an_older_head_marker_in_the_range() {
    let chat = open(OpenOptions::default()).await;
    let root_id = chat.root.id;
    let (a, b, c) = chat
        .root
        .commit(
            move |tx| async move {
                let note = |text: &str| {
                    let mut draft = EntryDraft::new("app.note");
                    draft.model = Some(vec![user(text)]);
                    draft
                };
                let a = tx.append_entry(root_id, note("a")).await?.id;
                let b = tx.append_entry(root_id, note("b")).await?.id;
                // An older marker that omits b, then a newer one whose range still contains the older marker.
                let mut older = EntryDraft::new("app.head");
                older.head = Some(EntryHead::Id(a));
                older.edits = Some(vec![ContextEdit {
                    target: b,
                    action: ContextEditAction::Omit,
                }]);
                tx.append_entry(root_id, older).await?;
                let c = tx.append_entry(root_id, note("c")).await?.id;
                let mut newer = EntryDraft::new("app.head");
                newer.head = Some(EntryHead::Id(a));
                newer.model = Some(vec![user("H")]);
                tx.append_entry(root_id, newer).await?;
                Ok((a, b, c))
            },
            &context(),
        )
        .await
        .unwrap();
    let view = chat.root.context(&context()).await.unwrap();
    let ids: Vec<_> = view.entries[1..].iter().map(|record| record.id).collect();
    assert_eq!(ids, vec![a, b, c]);
    let contributions: Vec<Vec<String>> = view
        .contributions
        .iter()
        .map(|messages| {
            messages
                .iter()
                .map(|message| user_text(Some(message)))
                .collect()
        })
        .collect();
    assert_eq!(contributions, vec![vec!["H"], vec!["a"], vec![], vec!["c"]]);
    assert_eq!(context_texts(&chat.root).await, vec!["H", "a", "c"]);
    // The summarizer sees the same contributions: the omitted entry stays out.
    chat.policy(CompactionPolicy {
        keep_recent_tokens: 1,
        ..MANUAL
    });
    chat.faux.summary(summary("SUMMARY"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    let prompt = user_text(chat.faux.summary_requests()[0].messages.get(1));
    assert!(
        prompt.contains("<conversation>\n[User]: H\n\n[User]: a\n</conversation>"),
        "{prompt}"
    );
    chat.close().await;
}
