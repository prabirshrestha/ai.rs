//! Port of `test/harness-compaction.test.ts`, part 2: background, blocking, and overflow compaction, estimates and
//! interactions, and compaction events.
//!
//! Divergences: a faulting compaction is a panicking `beforeCompact` hook (TS: a throwing `getModel`); Rust model
//! resolution and `complete_simple` cannot fail (a panicking provider stream ends with an error event, an ordinary
//! model error), and a hook error is reported and skipped.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::chat::{all_entries, tools_named, wait_for};
use super::compaction_support::*;
use super::support::{add_section, add_task, add_tool, context};
use crate::durable::harness::events::{EventBatch, watch_events};
use crate::durable::harness::types::{
    AgentChange, CompactionDecision, CompactionPolicy, ConversationAbortOptions,
    ConversationCreateOptions, RetryPolicyOverrides, SubmissionDraft, ToolExecutionResult,
};
use crate::durable::harness::{define_tool, section};
use crate::durable::tasks::{NextTaskState, Task, TaskDefinition, define_task};
use crate::durable::types::{
    ContextEdit, ContextEditAction, EntryDraft, TaskOptions as CreateTaskOptions, TaskOutcome,
    TaskOwnership,
};
use crate::providers::faux::faux_tool_call;
use crate::types::{Message, TextContent, UserContent, UserMessage};

fn json_of<T: serde::Serialize>(value: &T) -> JsonValue {
    serde_json::to_value(value).unwrap()
}

pub(super) async fn settled_json(
    submission: &crate::durable::harness::submissions::Submission,
) -> JsonValue {
    json_of(&submission.wait(&context()).await.unwrap())
}

pub(super) async fn submit(
    chat: &Chat,
    text: &str,
) -> crate::durable::harness::submissions::Submission {
    chat.root
        .submit(SubmissionDraft::input(text), &context())
        .await
        .unwrap()
}

pub(super) async fn task_state(chat: &Chat, id: crate::durable::ids::TaskId) -> JsonValue {
    json_of(
        &chat
            .harness
            .get_task(id, &context())
            .await
            .unwrap()
            .unwrap()
            .state,
    )
}

pub(super) fn contains(kinds: &[String], kind: &str) -> bool {
    kinds.iter().any(|candidate| candidate == kind)
}

pub(super) fn tail(kinds: &[String], count: usize) -> Vec<&str> {
    kinds[kinds.len() - count..]
        .iter()
        .map(String::as_str)
        .collect()
}

/// Record every event of the root conversation.
pub(super) async fn record_events(
    chat: &Chat,
) -> (
    crate::durable::harness::events::AgentEventStream,
    Arc<Mutex<Vec<Vec<JsonValue>>>>,
) {
    let stream = watch_events(&chat.harness, chat.root.id, &context())
        .await
        .unwrap();
    let batches: Arc<Mutex<Vec<Vec<JsonValue>>>> = Arc::default();
    let sink = batches.clone();
    stream
        .start(move |events: EventBatch, _| {
            sink.lock().push(events.iter().map(json_of).collect());
            Box::pin(async { Ok(()) })
        })
        .unwrap();
    (stream, batches)
}

pub(super) fn flat(batches: &Arc<Mutex<Vec<Vec<JsonValue>>>>) -> Vec<JsonValue> {
    batches.lock().iter().flatten().cloned().collect()
}

pub(super) fn event_types(events: &[JsonValue]) -> Vec<String> {
    events
        .iter()
        .map(|event| event["type"].as_str().unwrap().to_string())
        .collect()
}

// ─── Background threshold compaction ──────────────────────────────────────

#[tokio::test]
async fn starts_above_the_background_threshold_without_blocking_the_run_idle_waits_and_esc_ignore_it()
 {
    let chat = open(OpenOptions {
        context_window: Some(2000),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    chat.policy(BACKGROUND);
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        gate.clone(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    turn(&chat, &text("u4", 100), &text("a4", 100)).await;
    reached.wait().await;
    let tasks = compaction_tasks(&chat).await;
    let task = &tasks[0];
    assert!(task.background);
    assert_eq!(task.input["reason"], "threshold");
    assert!(task.owner.is_none());
    assert_eq!(
        compactions_json(&live(&chat).await),
        json!([{ "taskId": task.id, "reason": "threshold", "blocking": false, "attempt": 1 }])
    );
    // Background work: neither conversation idle nor Esc waits for or stops it.
    chat.root.wait_for_idle(&context()).await.unwrap();
    chat.root
        .abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    assert_eq!(task_state(&chat, task.id).await["status"], "running");
    gate.resolve();
    let outcome = result(&chat, task.id.cast()).await;
    assert_eq!(outcome_status(&outcome), "completed");
    assert_eq!(kinds(&chat.root).await.last().unwrap(), "pi.compaction");
    assert!(context_texts(&chat.root).await[0].contains("SUMMARY"));
    chat.close().await;
}

async fn does_not_start(policy: CompactionPolicy) {
    let chat = open(OpenOptions {
        context_window: Some(2000),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    chat.policy(policy);
    turn(&chat, &text("u4", 100), &text("a4", 100)).await;
    assert!(compaction_tasks(&chat).await.is_empty());
    assert!(chat.faux.summary_requests().is_empty());
    chat.close().await;
}

#[tokio::test]
async fn does_not_start_when_disabled() {
    does_not_start(CompactionPolicy {
        enabled: false,
        ..BACKGROUND
    })
    .await;
}

#[tokio::test]
async fn does_not_start_when_background_tokens_is_0() {
    does_not_start(CompactionPolicy {
        background_tokens: 0,
        ..BACKGROUND
    })
    .await;
}

#[tokio::test]
async fn does_not_start_when_there_is_no_cut() {
    does_not_start(CompactionPolicy {
        keep_recent_tokens: 100_000,
        ..BACKGROUND
    })
    .await;
}

#[tokio::test]
async fn does_not_start_while_another_compaction_is_listed() {
    let chat = open(OpenOptions {
        context_window: Some(2000),
        ..OpenOptions::default()
    })
    .await;
    history(&chat).await;
    let reached = Deferred::default();
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    let manual = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    chat.policy(BACKGROUND);
    turn(&chat, &text("u4", 100), &text("a4", 100)).await;
    let ids: Vec<_> = compaction_tasks(&chat)
        .await
        .iter()
        .map(|task| task.id)
        .collect();
    assert_eq!(ids, vec![manual.erase()]);
    chat.harness.abort_task(manual, &context()).await.unwrap();
    chat.close().await;
}

#[tokio::test]
async fn stops_through_abort_task_and_conversation_abort_with_background() {
    for stop in ["task", "conversation"] {
        let chat = open(OpenOptions {
            context_window: Some(2000),
            ..OpenOptions::default()
        })
        .await;
        history(&chat).await;
        chat.policy(BACKGROUND);
        let reached = Deferred::default();
        chat.faux.summary(gated(
            Deferred::default(),
            summary("SUMMARY"),
            Some(reached.clone()),
        ));
        turn(&chat, &text("u4", 100), &text("a4", 100)).await;
        reached.wait().await;
        let task = compaction_tasks(&chat).await[0].clone();
        if stop == "task" {
            chat.harness.abort_task(task.id, &context()).await.unwrap();
        } else {
            chat.root
                .abort(&context(), ConversationAbortOptions { background: true })
                .await
                .unwrap();
        }
        assert_eq!(
            outcome_status(&result(&chat, task.id.cast()).await),
            "aborted"
        );
        assert!(live(&chat).await.compactions.is_none());
        chat.close().await;
    }
}

// ─── Blocking threshold compaction ────────────────────────────────────────

fn small(context_window: u32) -> OpenOptions {
    OpenOptions {
        context_window: Some(context_window),
        ..OpenOptions::default()
    }
}

#[tokio::test]
async fn waits_for_its_compaction_which_appends_the_summary_before_the_request() {
    let chat = open(small(1000)).await;
    history(&chat).await;
    chat.policy(BLOCKING);
    // A new section: preparation has a system entry to append, but must not append it before the wait.
    super::chat::add_text_section(&chat.setup.registry, "extra", "EXTRA", None);
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        gate.clone(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    chat.faux.agent(answer("a4"));
    let input = submit(&chat, &text("u4", 200)).await;
    reached.wait().await;
    let child = compaction_tasks(&chat).await[0].clone();
    let generation = live(&chat).await.run.unwrap().task_id;
    assert_eq!(child.owner, Some(generation));
    assert!(!child.background);
    assert_eq!(child.input["reason"], "threshold");
    let state = task_state(&chat, generation).await;
    assert_eq!(state["status"], "waiting");
    assert_eq!(state["on"], json!([child.id]));
    assert_eq!(
        state["checkpoint"]["phase"], "prepare",
        "checkpoint: {}",
        state["checkpoint"]
    );
    assert_eq!(state["checkpoint"]["attempt"], 1);
    assert_eq!(state["checkpoint"]["compacted"], json!(child.id));
    assert_eq!(
        compactions_json(&live(&chat).await),
        json!([{ "taskId": child.id, "reason": "threshold", "blocking": true, "attempt": 1 }])
    );
    // Nothing was appended before the wait.
    assert_eq!(kinds(&chat.root).await.last().unwrap(), "pi.user");
    gate.resolve();
    assert_eq!(settled_json(&input).await["status"], "done");
    assert_eq!(
        outcome_status(&result(&chat, child.id.cast()).await),
        "completed"
    );
    assert_eq!(
        tail(&kinds(&chat.root).await, 3),
        ["pi.compaction", "pi.system", "pi.assistant"]
    );
    let request = chat.faux.last_agent_messages();
    assert!(user_text(request.first()).contains("SUMMARY"));
    let systems: Vec<&Message> = request
        .iter()
        .filter(|message| matches!(message, Message::System(_)))
        .collect();
    assert_eq!(systems.len(), 1);
    assert_eq!(
        json_of(systems[0])["sections"],
        json!({ "preamble": "You are helpful.", "extra": "<extra>\nEXTRA\n</extra>" })
    );
    chat.close().await;
}

#[tokio::test]
async fn sends_the_request_once_without_a_second_compaction_when_the_kept_part_is_still_above_the_threshold()
 {
    let chat = open(small(1000)).await;
    history(&chat).await;
    chat.policy(CompactionPolicy {
        keep_recent_tokens: 700,
        ..BLOCKING
    });
    chat.faux.summary(summary("SUMMARY"));
    turn(&chat, &text("u4", 400), "a4").await;
    assert_eq!(chat.faux.summary_requests().len(), 1);
    assert_eq!(
        kinds(&chat.root)
            .await
            .iter()
            .filter(|kind| *kind == "pi.compaction")
            .count(),
        1
    );
    chat.close().await;
}

async fn sends_the_request_anyway(prepare: impl FnOnce(&Chat)) {
    let chat = open(small(1000)).await;
    history(&chat).await;
    chat.policy(BLOCKING);
    prepare(&chat);
    turn(&chat, &text("u4", 200), "a4").await;
    assert!(!contains(&kinds(&chat.root).await, "pi.compaction"));
    assert_eq!(
        user_text(chat.faux.last_agent_messages().first()),
        text("u1", 100)
    );
    chat.close().await;
}

#[tokio::test]
async fn sends_the_request_anyway_when_its_compaction_declines() {
    sends_the_request_anyway(|chat| {
        decline(&chat.setup);
    })
    .await;
}

#[tokio::test]
async fn sends_the_request_anyway_when_its_compaction_fails() {
    sends_the_request_anyway(|chat| chat.faux.summary(failure("bad request"))).await;
}

#[tokio::test]
async fn sends_the_request_anyway_when_its_compaction_is_aborted_directly() {
    let chat = open(small(1000)).await;
    history(&chat).await;
    chat.policy(BLOCKING);
    let reached = Deferred::default();
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    chat.faux.agent(answer("a4"));
    let input = submit(&chat, &text("u4", 200)).await;
    reached.wait().await;
    let child = compaction_tasks(&chat).await[0].clone();
    chat.harness.abort_task(child.id, &context()).await.unwrap();
    assert_eq!(settled_json(&input).await["status"], "done");
    assert!(!contains(&kinds(&chat.root).await, "pi.compaction"));
    chat.close().await;
}

#[tokio::test]
async fn is_aborted_with_its_generation_by_esc_before_the_generations_abort_handler() {
    let chat = open(small(1000)).await;
    history(&chat).await;
    chat.policy(BLOCKING);
    let reached = Deferred::default();
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    let (stream, batches) = record_events(&chat).await;
    let input = submit(&chat, &text("u4", 200)).await;
    reached.wait().await;
    let child = compaction_tasks(&chat).await[0].clone();
    chat.root
        .abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    let settled = settled_json(&input).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "aborted");
    assert_eq!(
        task_state(&chat, child.id).await["outcome"]["status"],
        "aborted"
    );
    {
        let batches = batches.clone();
        wait_for(move || {
            let done = flat(&batches)
                .iter()
                .any(|event| event["type"] == "run_end");
            async move { done }
        })
        .await;
    }
    let types = event_types(&flat(&batches));
    let position = |kind: &str| {
        types
            .iter()
            .position(|candidate| candidate == kind)
            .unwrap()
    };
    assert!(position("compaction_end") < position("run_end"));
    stream.stop().await;
    chat.close().await;
}

#[tokio::test]
async fn wins_over_a_background_compaction_still_in_flight_which_then_settles_stale() {
    let chat = open(small(2000)).await;
    history(&chat).await;
    chat.policy(BACKGROUND);
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        gate.clone(),
        summary("BACKGROUND"),
        Some(reached.clone()),
    ));
    turn(&chat, &text("u4", 100), &text("a4", 100)).await;
    reached.wait().await;
    let background = compaction_tasks(&chat).await[0].clone();
    chat.faux.summary(summary("BLOCKING"));
    turn(&chat, &text("u5", 1000), "a5").await;
    assert!(context_texts(&chat.root).await[0].contains("BLOCKING"));
    gate.resolve();
    let outcome = result(&chat, background.id.cast()).await;
    let record = submission_status(&chat, submission_id(&outcome).unwrap()).await;
    assert_eq!(record["status"], "unanswered");
    assert_eq!(record["reason"], "stale");
    chat.close().await;
}

// ─── Overflow compaction ──────────────────────────────────────────────────

const ENABLED: CompactionPolicy = CompactionPolicy {
    enabled: true,
    ..MANUAL
};

#[tokio::test]
async fn compacts_and_retries_with_the_same_attempt_leaving_the_error_out_of_the_retry() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    chat.policy(ENABLED);
    chat.faux.summary(summary("SUMMARY"));
    let attempt = Arc::new(Mutex::new(None));
    chat.faux.agent(failure(OVERFLOW));
    {
        let (chat_ref, attempt) = (chat.clone(), attempt.clone());
        chat.faux.agent(step(move |_| async move {
            *attempt.lock() = live(&chat_ref)
                .await
                .generation
                .map(|generation| generation.attempt);
            answer("fits")
        }));
    }
    let input = submit(&chat, &text("u4", 100)).await;
    assert_eq!(settled_json(&input).await["status"], "done");
    assert_eq!(*attempt.lock(), Some(1));
    assert_eq!(
        tail(&kinds(&chat.root).await, 5),
        [
            "pi.user",
            "pi.assistant",
            "pi.compaction",
            "pi.system",
            "pi.assistant"
        ]
    );
    let compaction = all_entries(&chat.root)
        .await
        .into_iter()
        .find(|record| record.kind == "pi.compaction")
        .unwrap();
    assert_eq!(json_of(&compaction)["data"]["reason"], "overflow");
    let retry = chat.faux.last_agent_messages();
    assert!(user_text(retry.first()).contains("SUMMARY"));
    assert!(!retry.iter().any(|message| matches!(
        message,
        Message::Assistant(assistant) if assistant.stop_reason == crate::types::StopReason::Error
    )));
    chat.close().await;
}

#[tokio::test]
async fn fails_a_second_overflow_with_its_error_entry() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    chat.policy(ENABLED);
    chat.faux.summary(summary("SUMMARY"));
    chat.faux.agent(failure(OVERFLOW));
    chat.faux.agent(failure(OVERFLOW));
    let input = submit(&chat, &text("u4", 100)).await;
    let settled = settled_json(&input).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "model_error");
    assert_eq!(settled["detail"], OVERFLOW);
    let kinds = kinds(&chat.root).await;
    assert_eq!(
        kinds.iter().filter(|kind| *kind == "pi.compaction").count(),
        1
    );
    assert_eq!(kinds.last().unwrap(), "pi.assistant");
    chat.close().await;
}

#[tokio::test]
async fn fails_without_compacting_when_compaction_is_disabled_even_for_a_retryable_looking_overflow()
 {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    chat.faux.agent(failure(&format!("overloaded: {OVERFLOW}")));
    let input = submit(&chat, &text("u4", 100)).await;
    let settled = settled_json(&input).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "model_error");
    assert_eq!(chat.faux.agent_requests().len(), 4);
    assert!(chat.faux.summary_requests().is_empty());
    chat.close().await;
}

#[tokio::test]
async fn fails_after_a_blocking_threshold_compaction_in_the_same_generation() {
    let chat = open(small(1000)).await;
    history(&chat).await;
    chat.policy(BLOCKING);
    chat.faux.summary(summary("SUMMARY"));
    chat.faux.agent(failure(OVERFLOW));
    let input = submit(&chat, &text("u4", 200)).await;
    let settled = settled_json(&input).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "model_error");
    assert_eq!(chat.faux.summary_requests().len(), 1);
    chat.close().await;
}

async fn fails_with_the_overflow_text(prepare: impl FnOnce(&Chat), requests: usize) {
    let chat = open(OpenOptions::default()).await;
    turn(&chat, &text("u1", 100), &text("a1", 100)).await;
    chat.policy(ENABLED);
    prepare(&chat);
    chat.faux.agent(failure(OVERFLOW));
    let input = submit(&chat, &text("u4", 100)).await;
    let settled = settled_json(&input).await;
    assert_eq!(settled["status"], "unanswered");
    assert_eq!(settled["reason"], "model_error");
    assert_eq!(settled["detail"], OVERFLOW);
    assert_eq!(chat.faux.summary_requests().len(), requests);
    chat.close().await;
}

#[tokio::test]
async fn fails_with_the_overflow_text_when_compaction_declines() {
    fails_with_the_overflow_text(
        |chat| {
            decline(&chat.setup);
        },
        0,
    )
    .await;
}

#[tokio::test]
async fn fails_with_the_overflow_text_when_compaction_fails() {
    fails_with_the_overflow_text(|chat| chat.faux.summary(failure("bad request")), 1).await;
}

// Classification finds no cut, so no compaction starts and the ordinary failure carries the text.
#[tokio::test]
async fn fails_with_the_overflow_text_when_compaction_cannot_cut() {
    fails_with_the_overflow_text(
        |chat| {
            chat.policy(CompactionPolicy {
                keep_recent_tokens: 100_000,
                ..ENABLED
            })
        },
        0,
    )
    .await;
}

// ─── Compaction estimates and interactions ───────────────────────────────

async fn ignores_usage_measured_before_a_summary_placed_mid_run(fixed_clock: bool) {
    let setup = compaction_setup(2000);
    if fixed_clock {
        setup.set_now(|| 1_000);
    }
    let faux = script(&setup);
    let chat = open_with(small(2000), setup, faux).await;
    let (tool_gate, tool_reached) = (Deferred::default(), Deferred::default());
    {
        let (tool_gate, tool_reached) = (tool_gate.clone(), tool_reached.clone());
        add_tool(
            &chat.setup.registry,
            define_tool(
                "slow",
                "slow",
                json!({ "type": "object", "properties": {} }),
                move |_, _, _| {
                    let (tool_gate, tool_reached) = (tool_gate.clone(), tool_reached.clone());
                    async move {
                        tool_reached.resolve();
                        tool_gate.wait().await;
                        Ok(ToolExecutionResult {
                            content: Some(vec![UserContent::Text(TextContent::new(text(
                                "result", 200,
                            )))]),
                            ..ToolExecutionResult::default()
                        })
                    }
                },
            ),
        );
    }
    chat.root
        .configure(
            AgentChange::default().tools(tools_named(&chat.setup, &["slow"])),
            &context(),
        )
        .await
        .unwrap();
    turn(&chat, &text("u1", 220), &text("a1", 220)).await;
    turn(&chat, &text("u2", 220), &text("a2", 220)).await;
    turn(&chat, &text("u3", 220), &text("a3", 220)).await;
    // Background at 900, blocking at 1500: the tool call's usage plus its result would cross 1500.
    chat.policy(CompactionPolicy {
        background_tokens: 600,
        ..BACKGROUND
    });
    // The request starts a background compaction; its tool call's usage measures the whole context.
    chat.faux
        .agent(tool_use(vec![faux_tool_call("slow", json!({}), None)]));
    chat.faux.agent(answer("done"));
    let (summary_gate, summary_reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        summary_gate.clone(),
        summary("SUMMARY"),
        Some(summary_reached.clone()),
    ));
    let input = submit(&chat, &text("u4", 50)).await;
    tool_reached.wait().await;
    summary_reached.wait().await;
    let background = compaction_tasks(&chat).await[0].clone();
    summary_gate.resolve();
    // Queued: the run is busy in its tool round.
    let queued = result(&chat, background.id.cast()).await;
    assert_eq!(
        status(&submission_status(&chat, submission_id(&queued).unwrap()).await),
        "queued"
    );
    tool_gate.resolve();
    assert_eq!(settled_json(&input).await["status"], "done");
    // The summary landed at postTools; the successor saw a small context and did not compact again.
    assert_eq!(chat.faux.summary_requests().len(), 1);
    assert!(compaction_tasks(&chat).await.is_empty());
    assert!(user_text(chat.faux.last_agent_messages().first()).contains("SUMMARY"));
    chat.close().await;
}

#[tokio::test]
async fn ignores_usage_measured_before_a_summary_placed_mid_run_real_clock() {
    ignores_usage_measured_before_a_summary_placed_mid_run(false).await;
}

#[tokio::test]
async fn ignores_usage_measured_before_a_summary_placed_mid_run_fixed_clock() {
    ignores_usage_measured_before_a_summary_placed_mid_run(true).await;
}

#[tokio::test]
async fn rebaselines_the_system_prompt_over_kept_system_deltas() {
    let chat = open(OpenOptions::default()).await;
    let mood = Arc::new(Mutex::new("cheerful".to_string()));
    {
        let mood = mood.clone();
        add_section(
            &chat.setup.registry,
            section(
                "mood",
                move |_, _| {
                    let mood = mood.lock().clone();
                    async move { Ok(Some(mood)) }
                },
                Some(false),
            ),
        );
    }
    turn(&chat, &text("u1", 100), &text("a1", 100)).await;
    turn(&chat, &text("u2", 100), &text("a2", 100)).await;
    *mood.lock() = "terse".into();
    turn(&chat, &text("u3", 100), &text("a3", 100)).await;
    chat.policy(CompactionPolicy {
        keep_recent_tokens: 250,
        ..MANUAL
    });
    chat.faux.summary(summary("SUMMARY"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    // The kept range holds the delta for the terse mood; the next request has one complete baseline.
    assert!(
        chat.root
            .context(&context())
            .await
            .unwrap()
            .entries
            .iter()
            .any(|record| record.kind == "pi.system")
    );
    turn(&chat, "next", "ok").await;
    let systems: Vec<JsonValue> = chat
        .faux
        .last_agent_messages()
        .iter()
        .filter(|message| matches!(message, Message::System(_)))
        .map(json_of)
        .collect();
    assert_eq!(systems.len(), 1);
    assert_eq!(
        systems[0]["sections"],
        json!({ "preamble": "You are helpful.", "mood": "terse" })
    );
    chat.close().await;
}

#[tokio::test]
async fn summarizes_a_replaced_entrys_replacement_and_shows_it_to_the_hook() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let u1 = all_entries(&chat.root).await[0].clone();
    let mut draft = EntryDraft::new("app.redact");
    draft.edits = Some(vec![ContextEdit {
        target: u1.id,
        action: ContextEditAction::Replace {
            messages: vec![Message::User(UserMessage {
                content: "REDACTED".into(),
                timestamp: 0,
            })],
        },
    }]);
    chat.root
        .submit(SubmissionDraft::write(draft), &context())
        .await
        .unwrap();
    let messages: Arc<Mutex<Vec<Message>>> = Arc::default();
    {
        let messages = messages.clone();
        add_before_compact(&chat.setup, move |request| {
            *messages.lock() = request.messages;
            Ok(None)
        });
    }
    chat.faux.summary(summary("SUMMARY"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    assert_eq!(user_text(messages.lock().first()), "REDACTED");
    assert!(
        user_text(chat.faux.summary_requests()[0].messages.get(1)).contains("[User]: REDACTED")
    );
    chat.close().await;
}

#[tokio::test]
async fn compacts_a_fork_whose_cut_falls_on_a_parent_entry() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let entries = all_entries(&chat.root).await;
    let fork = chat
        .root
        .fork(
            entries.last().unwrap().id,
            ConversationCreateOptions::ownerless(),
            &context(),
        )
        .await
        .unwrap();
    chat.faux.summary(summary("SUMMARY"));
    let outcome = result(&chat, fork.compact(None, &context()).await.unwrap()).await;
    assert_eq!(
        status(&submission_settled(&chat, submission_id(&outcome).unwrap()).await),
        "done"
    );
    let u3 = entries
        .iter()
        .find(|record| user_text(record.model.as_ref().and_then(|m| m.first())).starts_with("u3"))
        .unwrap();
    let view = fork.context(&context()).await.unwrap();
    let head = view.head.clone().unwrap();
    assert_eq!(head.kind, "pi.compaction");
    assert_eq!(head.head, Some(u3.id));
    assert_eq!(head.conversation_id, fork.id);
    let texts: Vec<String> = view.messages[1..]
        .iter()
        .map(|message| user_text(Some(message)))
        .collect();
    assert_eq!(texts, vec![text("u3", 100), text("a3", 100)]);
    // The parent is untouched.
    assert!(!contains(&kinds(&chat.root).await, "pi.compaction"));
    // The fork's view keeps the parent entries the summary kept.
    let state = fork.view_state(&context()).await.unwrap();
    assert_eq!(state.value().entries.to_vec(), view.entries);
    state.dispose().unwrap();
    chat.close().await;
}

#[tokio::test]
async fn settles_stale_in_a_fork_reset_while_it_summarizes() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let entries = all_entries(&chat.root).await;
    let fork = chat
        .root
        .fork(
            entries.last().unwrap().id,
            ConversationCreateOptions::ownerless(),
            &context(),
        )
        .await
        .unwrap();
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        gate.clone(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    let id = fork.compact(None, &context()).await.unwrap();
    reached.wait().await;
    fork.reset(None, &context()).await.unwrap();
    gate.resolve();
    let outcome = result(&chat, id).await;
    let record = submission_status(&chat, submission_id(&outcome).unwrap()).await;
    assert_eq!(record["status"], "unanswered");
    assert_eq!(record["reason"], "stale");
    assert_eq!(
        fork.context(&context()).await.unwrap().head.unwrap().kind,
        "pi.reset"
    );
    chat.close().await;
}

#[tokio::test]
async fn places_an_older_queued_summary_and_the_current_one_together_when_idle_admission_drains_the_inbox()
 {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.agent(gated(
        gate.clone(),
        failure("bad request"),
        Some(reached.clone()),
    ));
    let failed = submit(&chat, "fails").await;
    reached.wait().await;
    chat.faux.summary(summary("OLDER"));
    let older = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    gate.resolve();
    failed.wait(&context()).await.unwrap();
    // Idle now, with the older summary still queued; the current one queues behind it and a final boundary runs.
    chat.faux.summary(summary("CURRENT"));
    let current = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    let mut statuses = Vec::new();
    for outcome in [older, current] {
        statuses.push(status(
            &submission_status(&chat, submission_id(&outcome).unwrap()).await,
        ));
    }
    assert_eq!(statuses, vec!["done", "done"]);
    assert!(context_texts(&chat.root).await[0].contains("CURRENT"));
    chat.close().await;
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Empty {}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
enum Run {
    Run,
}

fn child_task(gate: Deferred) -> Task<Empty, Run, ()> {
    define_task(
        TaskDefinition::new("test.child", 1, |_: &Empty| Run::Run)
            .phase("run", move |_, runtime, ctx| {
                let gate = gate.clone();
                async move {
                    gate.wait().await;
                    runtime
                        .commit(
                            |_, _| async {
                                Ok(Some(NextTaskState::Terminal {
                                    outcome: TaskOutcome::Completed {
                                        result: JsonValue::Null,
                                    },
                                }))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .abort(|_, runtime, ctx| async move {
                runtime
                    .commit(
                        |_, _| async {
                            Ok(Some(NextTaskState::Terminal {
                                outcome: TaskOutcome::Aborted {
                                    reason: None,
                                    result: None,
                                },
                            }))
                        },
                        &ctx,
                    )
                    .await
            }),
    )
}

#[tokio::test]
async fn places_a_hooks_summary_and_holds_while_work_the_hook_created_runs() {
    let chat = open(OpenOptions::default()).await;
    let child_gate = Deferred::default();
    let child = child_task(child_gate.clone());
    add_task(&chat.setup.registry, child.any());
    {
        let (harness, root_id) = (chat.harness.clone(), chat.root.id);
        let hooks = crate::durable::harness::types::CompactionHooks {
            before_compact: Some(Arc::new(move |_, api, hook_context| {
                let (harness, child) = (harness.clone(), child.clone());
                Box::pin(async move {
                    let owner = api.task_id();
                    harness
                        .commit(
                            move |tx| async move {
                                tx.create_task(
                                    &child,
                                    Empty {},
                                    CreateTaskOptions {
                                        ownership: TaskOwnership::Task { task_id: owner },
                                        conversation_id: Some(root_id),
                                        background: None,
                                    },
                                )
                                .await?;
                                Ok(())
                            },
                            &hook_context,
                        )
                        .await?;
                    Ok(Some(CompactionDecision::Summary("HOOK".into())))
                })
            })),
        };
        super::support::add_hooks(
            &chat.setup.registry,
            crate::durable::harness::hook(
                &*crate::durable::harness::compaction::COMPACTION_TASK,
                hooks,
            ),
        );
    }
    history(&chat).await;
    let id = chat.root.compact(None, &context()).await.unwrap();
    {
        let harness = chat.harness.clone();
        wait_for(move || {
            let harness = harness.clone();
            async move {
                json_of(
                    &harness
                        .get_task(id, &context())
                        .await
                        .unwrap()
                        .unwrap()
                        .state,
                )["status"]
                    == "completing"
            }
        })
        .await;
    }
    // The summary and the status removal landed at the hold.
    assert_eq!(kinds(&chat.root).await.last().unwrap(), "pi.compaction");
    assert!(live(&chat).await.compactions.is_none());
    child_gate.resolve();
    assert_eq!(outcome_status(&result(&chat, id).await), "completed");
    chat.close().await;
}

#[tokio::test]
async fn removes_the_status_of_a_faulted_compaction() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let broken = add_before_compact(&chat.setup, |_| panic!("models broke"));
    let outcome = json_of(&result(&chat, chat.root.compact(None, &context()).await.unwrap()).await);
    broken.dispose();
    assert_eq!(outcome["status"], "faulted");
    assert_eq!(outcome["error"]["message"], "models broke");
    assert!(live(&chat).await.compactions.is_none());
    chat.close().await;
}

// ─── Compaction events and live status ───────────────────────────────────

#[tokio::test]
async fn reports_start_and_end_and_the_retry_backoff_in_a_late_joiners_snapshot() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    chat.setup.settings(|settings| {
        settings.retry = Some(RetryPolicyOverrides {
            enabled: Some(true),
            max_retries: Some(2),
            base_delay_ms: Some(60_000),
            max_agent_delay_ms: None,
        })
    });
    let (stream, batches) = record_events(&chat).await;
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
    let late = watch_events(&chat.harness, chat.root.id, &context())
        .await
        .unwrap();
    let compactions = json_of(&late.snapshot)["compactions"].clone();
    assert!(compactions[0]["retry"]["at"].is_number());
    assert_eq!(
        compactions,
        json!([{
            "taskId": id,
            "reason": "manual",
            "blocking": false,
            "attempt": 1,
            "retry": { "at": compactions[0]["retry"]["at"], "error": "overloaded" },
        }])
    );
    late.stop().await;
    chat.harness.abort_task(id, &context()).await.unwrap();
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
    let compaction_events: Vec<JsonValue> = flat(&batches)
        .into_iter()
        .filter(|event| event["type"].as_str().unwrap().starts_with("compaction_"))
        .collect();
    assert_eq!(
        compaction_events,
        vec![
            json!({ "type": "compaction_start", "taskId": id, "reason": "manual", "blocking": false }),
            json!({ "type": "compaction_end", "taskId": id, "reason": "manual" }),
        ]
    );
    stream.stop().await;
    chat.close().await;
}

#[tokio::test]
async fn lists_concurrent_compactions_in_task_id_order() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (first, second) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(first.clone()),
    ));
    chat.faux.summary(gated(
        Deferred::default(),
        summary("SUMMARY"),
        Some(second.clone()),
    ));
    let a = chat.root.compact(None, &context()).await.unwrap();
    let b = chat.root.compact(None, &context()).await.unwrap();
    first.wait().await;
    second.wait().await;
    let ids: Vec<_> = live(&chat)
        .await
        .compactions
        .unwrap()
        .iter()
        .map(|status| status.task_id)
        .collect();
    assert_eq!(ids, vec![a, b]);
    chat.root
        .abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    assert!(live(&chat).await.compactions.is_none());
    chat.close().await;
}
