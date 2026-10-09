//! Port of `test/harness-compaction.test.ts`, part 1: range selection, serialization, manual compaction, and
//! compaction outcomes. The automatic, recovery, and interaction cases are in `compaction_auto.rs` and
//! `compaction_more.rs`.
//!
//! Divergences: records and outcomes are compared in their TS JSON form; the scripted faux model is a queue of
//! `async_factory` steps.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::chat::{all_entries, open_chat, tools_named};
use super::compaction_support::*;
use super::support::{add_tool, context, system};
use crate::durable::harness::compaction::{select_cut, serialize_conversation};
use crate::durable::harness::context::order_tool_results;
use crate::durable::harness::define_tool;
use crate::durable::harness::provider::PROVIDER_DOC;
use crate::durable::harness::types::{
    AgentChange, CompactionDecision, ContextView, ConversationAbortOptions,
    ConversationStreamOptions, DeferredOption, SubmissionDraft, ToolExecutionResult,
};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::EntryRecord;
use crate::providers::faux::{FauxMessageOptions, faux_assistant_message, faux_tool_call};
use crate::types::{
    AssistantContent, Message, ModelThinkingLevel, StopReason, TextContent, ThinkingContent,
    ToolResultMessage, UserContent, UserMessage,
};

// ─── Range selection ──────────────────────────────────────────────────────

struct Ids(std::sync::atomic::AtomicU64);

static NEXT_ID: Ids = Ids(std::sync::atomic::AtomicU64::new(1));

fn entry(kind: &str, model: Vec<Message>, head: Option<u64>) -> EntryRecord {
    let id = NEXT_ID.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut value = json!({ "id": id, "conversationId": 1, "kind": kind, "model": model });
    if let Some(head) = head {
        value["head"] = json!(head);
    }
    serde_json::from_value(value).unwrap()
}

fn user(content: &str) -> Message {
    Message::User(UserMessage {
        content: content.into(),
        timestamp: 0,
    })
}

fn assistant(content: &str, calls: &[&str], stop_reason: StopReason) -> Message {
    let mut blocks = vec![AssistantContent::Text(TextContent::new(content))];
    blocks.extend(
        calls
            .iter()
            .map(|id| faux_tool_call("read", json!({}), Some(id))),
    );
    Message::Assistant(faux_assistant_message(
        blocks,
        FauxMessageOptions {
            stop_reason: Some(stop_reason),
            ..FauxMessageOptions::default()
        },
    ))
}

fn stop(content: &str) -> Message {
    assistant(content, &[], StopReason::Stop)
}

fn calling(content: &str, calls: &[&str]) -> Message {
    assistant(content, calls, StopReason::Stop)
}

fn tool_result(call_id: &str, content: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: call_id.into(),
        tool_name: "read".into(),
        content: vec![UserContent::Text(TextContent::new(content))],
        details: None,
        usage: None,
        nested_calls: None,
        is_error: false,
        timestamp: 0,
    })
}

fn excluded(message: &Message) -> bool {
    matches!(
        message,
        Message::Assistant(assistant)
            if matches!(assistant.stop_reason, StopReason::Error | StopReason::Aborted | StopReason::Deferred)
    )
}

/// A view over `entries` whose contributions are their models, with excluded assistants removed.
fn view(entries: &[EntryRecord], head: Option<&EntryRecord>) -> ContextView {
    let all: Vec<EntryRecord> = head
        .into_iter()
        .cloned()
        .chain(entries.iter().cloned())
        .collect();
    let contributions: Vec<Vec<Message>> = all
        .iter()
        .map(|record| {
            record
                .model
                .clone()
                .unwrap_or_default()
                .into_iter()
                .filter(|message| !excluded(message))
                .collect()
        })
        .collect();
    let messages = order_tool_results(&contributions.concat());
    ContextView {
        head: head.cloned(),
        entries: all,
        contributions,
        messages,
    }
}

#[test]
fn keeps_about_keep_recent_tokens_and_cuts_at_the_first_candidate_at_or_after_the_budget() {
    let entries = [
        entry("pi.user", vec![user(&text("1", 10))], None),
        entry("pi.assistant", vec![calling("2", &["c1"])], None),
        entry(
            "pi.tool-result",
            vec![tool_result("c1", &text("3", 3000))],
            None,
        ),
        entry("pi.assistant", vec![stop(&text("4", 10))], None),
        entry("pi.user", vec![user(&text("5", 10))], None),
        entry("pi.assistant", vec![stop(&text("6", 10))], None),
    ];
    assert_eq!(select_cut(&view(&entries, None), 2000), Some(3));
}

#[test]
fn cuts_at_a_user_entry() {
    let entries = [
        entry("pi.user", vec![user(&text("u1", 100))], None),
        entry("pi.assistant", vec![stop(&text("a1", 100))], None),
        entry("pi.user", vec![user(&text("u2", 100))], None),
        entry("pi.assistant", vec![stop(&text("a2", 100))], None),
    ];
    assert_eq!(select_cut(&view(&entries, None), 150), Some(2));
}

#[test]
fn cuts_at_an_assistant_in_the_middle_of_one_long_run_and_never_at_a_tool_result() {
    let mut entries = vec![entry("pi.user", vec![user("do it")], None)];
    for index in 0..5 {
        let call = format!("c{index}");
        entries.push(entry(
            "pi.assistant",
            vec![calling(&format!("step {index}"), &[&call])],
            None,
        ));
        entries.push(entry(
            "pi.tool-result",
            vec![tool_result(&call, &text(&format!("r{index}"), 100))],
            None,
        ));
    }
    let cut = select_cut(&view(&entries, None), 150).unwrap();
    // The budget is reached at the fourth result; the cut is the last call, whose result it keeps.
    assert_eq!(entries[cut].kind, "pi.assistant");
    assert_eq!(cut, 9);
}

#[test]
fn keeps_a_huge_last_tool_result_together_with_its_assistant() {
    let entries = [
        entry("pi.user", vec![user("u")], None),
        entry("pi.assistant", vec![calling("a", &["c"])], None),
        entry(
            "pi.tool-result",
            vec![tool_result("c", &text("big", 5000))],
            None,
        ),
    ];
    assert_eq!(select_cut(&view(&entries, None), 100), Some(1));
}

#[test]
fn never_cuts_at_a_system_entry_or_an_excluded_error_or_aborted_answer() {
    let section = text("s", 100);
    let entries = [
        entry("pi.user", vec![user(&text("u1", 100))], None),
        entry("pi.assistant", vec![stop(&text("a1", 100))], None),
        entry("pi.system", vec![system(&[("s", Some(&section))])], None),
        entry(
            "pi.assistant",
            vec![assistant(&text("err", 100), &[], StopReason::Error)],
            None,
        ),
        entry(
            "pi.assistant",
            vec![assistant(&text("stopped", 100), &[], StopReason::Aborted)],
            None,
        ),
        entry("pi.assistant", vec![stop(&text("a2", 100))], None),
    ];
    // The walk reaches 150 at the system entry; the excluded answers after it contribute nothing.
    assert_eq!(select_cut(&view(&entries, None), 150), Some(5));
}

#[test]
fn follows_edited_contributions_an_omitted_entry_adds_nothing_and_is_no_candidate() {
    let entries = [
        entry("pi.user", vec![user(&text("u1", 100))], None),
        entry("pi.assistant", vec![stop(&text("a1", 100))], None),
        entry("pi.user", vec![user(&text("u2", 100))], None),
        entry("pi.assistant", vec![stop(&text("a2", 100))], None),
    ];
    let plain = view(&entries, None);
    assert_eq!(select_cut(&plain, 150), Some(2));
    let contributions: Vec<Vec<Message>> = plain
        .contributions
        .iter()
        .enumerate()
        .map(|(index, messages)| {
            if index == 2 {
                Vec::new()
            } else {
                messages.clone()
            }
        })
        .collect();
    let omitted = ContextView {
        messages: order_tool_results(&contributions.concat()),
        contributions,
        ..plain
    };
    assert_eq!(select_cut(&omitted, 150), Some(1));
}

#[test]
fn does_not_cut_at_a_user_entry_that_a_result_of_the_preceding_call_still_follows() {
    let entries = [
        entry("pi.user", vec![user(&text("u1", 100))], None),
        entry("pi.assistant", vec![calling("a", &["c"])], None),
        entry("pi.user", vec![user(&text("steer", 100))], None),
        entry(
            "pi.tool-result",
            vec![tool_result("c", &text("r", 100))],
            None,
        ),
        entry("pi.assistant", vec![stop(&text("a2", 100))], None),
    ];
    // The budget is reached at the steer; its result follows it, so the cut moves to the next assistant.
    assert_eq!(select_cut(&view(&entries, None), 250), Some(4));
}

#[test]
fn finds_nothing_when_the_budget_is_never_reached_or_only_the_marker_precedes_the_cut() {
    let small = [
        entry("pi.user", vec![user("hi")], None),
        entry("pi.assistant", vec![stop("hello")], None),
    ];
    assert_eq!(select_cut(&view(&small, None), 150), None);
    let marker = entry("pi.compaction", vec![user("summary")], Some(0));
    // The budget is reached at the only entry after the marker, so the marker alone would be summarized.
    let only = [entry("pi.user", vec![user(&text("u", 200))], None)];
    assert_eq!(select_cut(&view(&only, Some(&marker)), 150), None);
}

#[test]
fn summarizes_an_earlier_summary_marker_first() {
    let marker = entry("pi.compaction", vec![user("EARLIER")], Some(0));
    let kept = [
        entry("pi.user", vec![user(&text("u1", 100))], None),
        entry("pi.assistant", vec![stop(&text("a1", 100))], None),
        entry("pi.user", vec![user(&text("u2", 100))], None),
        entry("pi.assistant", vec![stop(&text("a2", 100))], None),
    ];
    let selected = view(&kept, Some(&marker));
    assert_eq!(select_cut(&selected, 150), Some(3));
    assert!(
        serialize_conversation(&selected.contributions[..3].concat())
            .starts_with("[User]: EARLIER")
    );
}

// ─── Serialization ───────────────────────────────────────────────────────

#[test]
fn writes_a_transcript_truncates_tool_results_and_omits_system_messages() {
    let call = faux_tool_call("read", json!({ "path": "a.ts" }), Some("c"));
    let messages = vec![
        system(&[("s", Some("hidden"))]),
        user("hello"),
        Message::Assistant(faux_assistant_message(
            vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "hmm".into(),
                    thinking_signature: None,
                    redacted: None,
                }),
                AssistantContent::Text(TextContent::new("sure")),
                call,
            ],
            FauxMessageOptions::default(),
        )),
        tool_result("c", &"y".repeat(2500)),
    ];
    let serialized = serialize_conversation(&messages);
    assert!(!serialized.contains("hidden"));
    assert!(serialized.contains("[User]: hello"));
    assert!(serialized.contains("[Assistant thinking]: hmm"));
    assert!(serialized.contains("[Assistant]: sure"));
    assert!(serialized.contains("[Assistant tool calls]: read(path=\"a.ts\")"));
    assert!(serialized.contains(&format!(
        "[Tool result]: {}\n\n[... 500 more characters truncated]",
        "y".repeat(2000)
    )));
}

// ─── Manual compaction ────────────────────────────────────────────────────

fn record_json(record: &EntryRecord) -> JsonValue {
    serde_json::to_value(record).unwrap()
}

async fn provider_session(chat: &Chat, id: crate::durable::ids::ConversationId) -> Option<String> {
    chat.harness
        .snapshot(&*PROVIDER_DOC, id, &context())
        .await
        .unwrap()
        .map(|state| state.session_id)
}

fn roles(messages: &[Message]) -> Vec<&'static str> {
    messages.iter().map(Message::role).collect()
}

#[tokio::test]
async fn places_the_summary_at_once_when_idle_and_keeps_raw_history() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let before = all_entries(&chat.root).await;
    let usage_before = usage_input(&chat).await;
    chat.faux.summary(summary("SUMMARY"));

    let id = chat
        .root
        .compact(Some("focus on files".into()), &context())
        .await
        .unwrap();
    let outcome = result(&chat, id).await;
    assert_eq!(outcome_status(&outcome), "completed");
    let placed = submission_settled(&chat, submission_id(&outcome).unwrap()).await;
    assert_eq!(status(&placed), "done");

    // Raw history is unchanged; one summary entry heads the first kept entry, u3.
    let after = all_entries(&chat.root).await;
    assert_eq!(after[..before.len()], before[..]);
    let marker = after.last().unwrap();
    let u3 = before
        .iter()
        .find(|record| user_text(record.model.as_ref().and_then(|m| m.first())).starts_with("u3"))
        .unwrap();
    assert_eq!(marker.kind, "pi.compaction");
    assert_eq!(marker.head, Some(u3.id));
    assert_eq!(record_json(marker)["data"]["reason"], "manual");
    assert_eq!(placed["entry"], json!(marker.id));
    let marker_text = user_text(marker.model.as_ref().and_then(|m| m.first()));
    assert_eq!(
        marker_text,
        "The conversation history before this point was compacted into the following summary:\n\n<summary>\nSUMMARY\n</summary>"
    );

    // The model context is the summary followed by the kept entries.
    assert_eq!(
        context_texts(&chat.root).await,
        vec![marker_text.clone(), text("u3", 100), text("a3", 100)]
    );

    // The summarizer saw the serialized prefix, the prompt, and the instructions, without tools or caching.
    let request = chat.faux.summary_requests()[0].clone();
    assert_eq!(request.messages.len(), 2);
    let prompt = user_text(request.messages.get(1));
    assert!(prompt.starts_with("<conversation>\n[User]: u1 "));
    assert!(prompt.contains("[Assistant]: a2 "));
    assert!(!prompt.contains("u3 "));
    assert!(prompt.contains("## Goal"));
    assert!(prompt.ends_with("\n\nAdditional focus: focus on files"));
    assert_eq!(
        request.options.stream.cache_retention,
        Some(crate::types::CacheRetention::None)
    );
    assert_eq!(request.options.stream.max_tokens, Some(800));
    assert_eq!(
        request.options.stream.session_id,
        provider_session(&chat, chat.root.id).await
    );
    assert!(request.options.deferred.is_none());

    // The summarizer's spend is in the ledger, and nothing counts it again later.
    assert!(usage_input(&chat).await > usage_before);
    turn(&chat, "next", "done").await;
    let agent = chat.faux.last_agent_messages();
    // The next request: the summary, the kept turn, the new input, then one complete system baseline.
    assert_eq!(
        roles(&agent),
        vec!["user", "user", "assistant", "user", "system"]
    );
    assert_eq!(
        serde_json::to_value(&agent[4]).unwrap()["sections"],
        json!({ "preamble": "You are helpful." })
    );
    chat.close().await;
}

// Regression coverage for #10424.
#[tokio::test]
async fn creates_provider_state_before_a_legacy_conversations_summarization_request() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let id = chat.root.id;
    chat.root
        .commit(
            move |tx| async move { tx.retire_doc(&*PROVIDER_DOC, id).await },
            &context(),
        )
        .await
        .unwrap();
    assert!(provider_session(&chat, id).await.is_none());
    chat.faux.summary(summary("SUMMARY"));
    let compaction = chat.root.compact(None, &context()).await.unwrap();
    assert_eq!(
        outcome_status(&result(&chat, compaction).await),
        "completed"
    );
    let stored = provider_session(&chat, id).await.unwrap();
    assert_eq!(stored.len(), 36);
    assert_eq!(&stored[14..15], "7");
    assert_eq!(
        chat.faux.summary_requests()[0].options.stream.session_id,
        Some(stored)
    );
    chat.close().await;
}

#[tokio::test]
async fn keeps_working_while_busy_and_places_the_summary_at_the_next_final_boundary() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.agent(gated(
        gate.clone(),
        answer("late answer"),
        Some(reached.clone()),
    ));
    let input = chat
        .root
        .submit(SubmissionDraft::input("busy"), &context())
        .await
        .unwrap();
    reached.wait().await;
    chat.faux.summary(summary("SUMMARY"));
    let outcome = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    let submission = submission_id(&outcome).unwrap();
    assert_eq!(
        status(&submission_status(&chat, submission).await),
        "queued"
    );
    gate.resolve();
    assert_eq!(
        serde_json::to_value(input.wait(&context()).await.unwrap()).unwrap()["status"],
        "done"
    );
    assert_eq!(status(&submission_settled(&chat, submission).await), "done");
    // The summary follows the answer; the kept range still includes the busy turn.
    let kinds = kinds(&chat.root).await;
    assert_eq!(
        kinds[kinds.len() - 3..],
        ["pi.user", "pi.assistant", "pi.compaction"]
    );
    chat.close().await;
}

fn wait_tool() -> crate::durable::harness::types::ToolRegistration {
    define_tool(
        "wait",
        "wait",
        json!({ "type": "object", "properties": {} }),
        |_, _, _| async {
            Ok(ToolExecutionResult {
                content: Some(vec![UserContent::Text(TextContent::new("waited"))]),
                ..ToolExecutionResult::default()
            })
        },
    )
}

#[tokio::test]
async fn places_a_queued_summary_at_post_tools_and_the_run_continues_in_the_compacted_context() {
    let chat = open(OpenOptions::default()).await;
    add_tool(&chat.setup.registry, wait_tool());
    chat.root
        .configure(
            AgentChange::default().tools(tools_named(&chat.setup, &["wait"])),
            &context(),
        )
        .await
        .unwrap();
    history(&chat).await;
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.agent(gated(
        gate.clone(),
        tool_use(vec![faux_tool_call("wait", json!({}), None)]),
        Some(reached.clone()),
    ));
    chat.faux.agent(answer("after tools"));
    let input = chat
        .root
        .submit(SubmissionDraft::input("use a tool"), &context())
        .await
        .unwrap();
    reached.wait().await;
    chat.faux.summary(summary("SUMMARY"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    gate.resolve();
    input.wait(&context()).await.unwrap();
    // The continuation request starts with the summary.
    let continuation = chat.faux.last_agent_messages();
    assert!(user_text(continuation.first()).contains("<summary>\nSUMMARY\n</summary>"));
    let kinds = kinds(&chat.root).await;
    assert_eq!(
        kinds[kinds.len() - 4..],
        [
            "pi.tool-result",
            "pi.compaction",
            "pi.system",
            "pi.assistant"
        ]
    );
    chat.close().await;
}

#[tokio::test]
async fn runs_follow_ups_left_by_a_failed_run_after_placing_the_summary() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.agent(gated(
        gate.clone(),
        failure("bad request"),
        Some(reached.clone()),
    ));
    let failed = chat
        .root
        .submit(SubmissionDraft::input("fails"), &context())
        .await
        .unwrap();
    reached.wait().await;
    let follow_up = chat
        .root
        .submit(SubmissionDraft::input("follow-up"), &context())
        .await
        .unwrap();
    gate.resolve();
    assert_eq!(
        serde_json::to_value(failed.wait(&context()).await.unwrap()).unwrap()["status"],
        "unanswered"
    );
    assert_eq!(
        serde_json::to_value(follow_up.status(&context()).await.unwrap()).unwrap()["status"],
        "queued"
    );

    chat.faux.summary(summary("SUMMARY"));
    chat.faux.agent(answer("followed"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    assert_eq!(
        serde_json::to_value(follow_up.wait(&context()).await.unwrap()).unwrap()["status"],
        "done"
    );
    let request = chat.faux.last_agent_messages();
    assert!(user_text(request.first()).contains("SUMMARY"));
    assert!(
        request
            .iter()
            .any(|message| user_text(Some(message)) == "follow-up")
    );
    chat.close().await;
}

#[tokio::test]
async fn settles_stale_when_a_reset_lands_while_it_summarizes() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        gate.clone(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    let id = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    chat.root.reset(None, &context()).await.unwrap();
    gate.resolve();
    let outcome = result(&chat, id).await;
    let record = submission_status(&chat, submission_id(&outcome).unwrap()).await;
    assert_eq!(record["status"], "unanswered");
    assert_eq!(record["reason"], "stale");
    assert_eq!(kinds(&chat.root).await.last().unwrap(), "pi.reset");
    chat.close().await;
}

#[tokio::test]
async fn does_not_make_the_conversation_busy_a_submission_during_summarization_starts_its_run_at_once()
 {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (summary_gate, summary_reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        summary_gate.clone(),
        summary("SUMMARY"),
        Some(summary_reached.clone()),
    ));
    let id = chat.root.compact(None, &context()).await.unwrap();
    summary_reached.wait().await;
    let (answer_gate, answer_reached) = (Deferred::default(), Deferred::default());
    chat.faux.agent(gated(
        answer_gate.clone(),
        answer("a4"),
        Some(answer_reached.clone()),
    ));
    let input = chat
        .root
        .submit(SubmissionDraft::input("u4"), &context())
        .await
        .unwrap();
    // Placed and answered immediately with the uncompacted context, not queued behind the compaction.
    assert_eq!(
        serde_json::to_value(input.status(&context()).await.unwrap()).unwrap()["status"],
        "placed"
    );
    answer_reached.wait().await;
    assert_eq!(
        user_text(chat.faux.last_agent_messages().first()),
        text("u1", 100)
    );
    // The summary is ready while the run is busy, so it queues and lands after the answer.
    summary_gate.resolve();
    let outcome = result(&chat, id).await;
    let submission = submission_id(&outcome).unwrap();
    assert_eq!(
        status(&submission_status(&chat, submission).await),
        "queued"
    );
    answer_gate.resolve();
    input.wait(&context()).await.unwrap();
    assert_eq!(status(&submission_settled(&chat, submission).await), "done");
    let kinds = kinds(&chat.root).await;
    assert_eq!(
        kinds[kinds.len() - 3..],
        ["pi.user", "pi.assistant", "pi.compaction"]
    );
    chat.close().await;
}

#[tokio::test]
async fn counts_the_spend_of_a_summary_that_ends_stale_and_writes_no_entry_for_it() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let before = usage_input(&chat).await;
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        gate.clone(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    let id = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    chat.root.reset(None, &context()).await.unwrap();
    gate.resolve();
    let outcome = result(&chat, id).await;
    let record = submission_status(&chat, submission_id(&outcome).unwrap()).await;
    assert_eq!(record["status"], "unanswered");
    assert_eq!(record["reason"], "stale");
    assert!(usage_input(&chat).await > before);
    assert!(
        !kinds(&chat.root)
            .await
            .contains(&"pi.compaction".to_string())
    );
    chat.close().await;
}

#[tokio::test]
async fn lets_the_compaction_that_cuts_furthest_win_whatever_finishes_first() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (first, first_reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        first.clone(),
        summary("FIRST"),
        Some(first_reached.clone()),
    ));
    let early = chat.root.compact(None, &context()).await.unwrap();
    first_reached.wait().await;
    turn(&chat, &text("u4", 100), &text("a4", 100)).await;
    chat.faux.summary(summary("SECOND"));
    let later = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    assert_eq!(outcome_status(&later), "completed");
    first.resolve();
    let outcome = result(&chat, early).await;
    // The early compaction cut at u3, before the later cut at u4.
    let record = submission_status(&chat, submission_id(&outcome).unwrap()).await;
    assert_eq!(record["status"], "unanswered");
    assert_eq!(record["reason"], "stale");
    assert!(context_texts(&chat.root).await[0].contains("SECOND"));
    chat.close().await;
}

#[tokio::test]
async fn places_an_older_selected_summary_that_cuts_later_than_the_newer_one() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    // A: small budget, late cut; selected first, finishes last.
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux
        .summary(gated(gate.clone(), summary("A"), Some(reached.clone())));
    let a = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    // B: larger budget, earlier cut; placed first.
    chat.policy(crate::durable::harness::types::CompactionPolicy {
        keep_recent_tokens: 350,
        ..MANUAL
    });
    chat.faux.summary(summary("B"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    assert!(context_texts(&chat.root).await[0].contains("B"));
    gate.resolve();
    let outcome = result(&chat, a).await;
    assert_eq!(
        status(&submission_status(&chat, submission_id(&outcome).unwrap()).await),
        "done"
    );
    let texts = context_texts(&chat.root).await;
    assert!(texts[0].contains("<summary>\nA\n</summary>"));
    assert_eq!(texts[1..], [text("u3", 100), text("a3", 100)]);
    chat.close().await;
}

async fn places_two_queued_summaries(keeps: [u64; 2], stale: bool) {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux
        .agent(gated(gate.clone(), answer("done"), Some(reached.clone())));
    let input = chat
        .root
        .submit(SubmissionDraft::input("busy"), &context())
        .await
        .unwrap();
    reached.wait().await;
    let mut submissions = Vec::new();
    for (index, keep) in keeps.into_iter().enumerate() {
        chat.policy(crate::durable::harness::types::CompactionPolicy {
            keep_recent_tokens: keep,
            ..MANUAL
        });
        chat.faux.summary(summary(&format!("S{index}")));
        let outcome = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
        submissions.push(submission_id(&outcome).unwrap());
    }
    gate.resolve();
    input.wait(&context()).await.unwrap();
    let mut statuses = Vec::new();
    for id in submissions {
        statuses.push(status(&submission_settled(&chat, id).await));
    }
    assert_eq!(
        statuses,
        vec!["done", if stale { "unanswered" } else { "done" }]
    );
    chat.close().await;
}

#[tokio::test]
async fn places_two_queued_summaries_in_one_boundary_when_the_second_cuts_before_the_first() {
    places_two_queued_summaries([150, 350], true).await;
}

#[tokio::test]
async fn places_two_queued_summaries_in_one_boundary_when_the_second_cuts_at_the_first() {
    places_two_queued_summaries([150, 150], false).await;
}

#[tokio::test]
async fn places_two_queued_summaries_in_one_boundary_when_the_second_cuts_after_the_first() {
    places_two_queued_summaries([350, 150], false).await;
}

#[tokio::test]
async fn is_aborted_by_conversation_abort_an_already_queued_summary_survives_it() {
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
    assert_eq!(live(&chat).await.compactions.unwrap().len(), 1);
    chat.root
        .abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    assert_eq!(outcome_status(&result(&chat, id).await), "aborted");
    assert!(live(&chat).await.compactions.is_none());
    assert!(
        !kinds(&chat.root)
            .await
            .contains(&"pi.compaction".to_string())
    );

    // Queued while busy, then Esc: the queued write stays and lands with the next run's boundary.
    let busy = Deferred::default();
    chat.faux.agent(gated(
        Deferred::default(),
        answer("never"),
        Some(busy.clone()),
    ));
    chat.root
        .submit(SubmissionDraft::input("busy"), &context())
        .await
        .unwrap();
    busy.wait().await;
    chat.faux.summary(summary("SUMMARY"));
    let queued = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    chat.root
        .abort(&context(), ConversationAbortOptions::default())
        .await
        .unwrap();
    let submission = chat
        .harness
        .submission(submission_id(&queued).unwrap(), &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::to_value(submission.status(&context()).await.unwrap()).unwrap()["status"],
        "queued"
    );
    assert_eq!(
        format!("{:?}", submission.abort(&context()).await.unwrap()),
        "Aborted"
    );
    turn(&chat, "next", "ok").await;
    let record = serde_json::to_value(submission.status(&context()).await.unwrap()).unwrap();
    assert_eq!(record["status"], "unanswered");
    assert_eq!(record["reason"], "aborted");
    assert!(
        !kinds(&chat.root)
            .await
            .contains(&"pi.compaction".to_string())
    );
    chat.close().await;
}

#[tokio::test]
async fn is_ordinary_work_idle_waits_include_it() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    let (gate, reached) = (Deferred::default(), Deferred::default());
    chat.faux.summary(gated(
        gate.clone(),
        summary("SUMMARY"),
        Some(reached.clone()),
    ));
    let id = chat.root.compact(None, &context()).await.unwrap();
    reached.wait().await;
    let idle = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let wait = {
        let (root, idle) = (chat.root.clone(), idle.clone());
        tokio::spawn(async move {
            root.wait_for_idle(&context()).await.unwrap();
            idle.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert!(!idle.load(std::sync::atomic::Ordering::SeqCst));
    gate.resolve();
    wait.await.unwrap();
    let record = chat
        .harness
        .get_task(id, &context())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        record.state,
        crate::durable::types::TaskState::Terminal { .. }
    ));
    chat.close().await;
}

#[tokio::test]
async fn enables_scheduling_right_after_open() {
    let setup = super::chat::chat_setup();
    let faux = script(&setup);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    setup.settings(|settings| {
        settings.compaction = Some(overrides(
            crate::durable::harness::types::CompactionPolicy {
                keep_recent_tokens: 10,
                ..MANUAL
            },
        ))
    });
    let id = root.compact(None, &context()).await.unwrap();
    // get_task() only reads, so progress here comes from compact() itself.
    {
        let harness = harness.clone();
        super::chat::wait_for(move || {
            let harness = harness.clone();
            async move {
                matches!(
                    harness
                        .get_task(id, &context())
                        .await
                        .unwrap()
                        .map(|record| record.state),
                    Some(crate::durable::types::TaskState::Terminal { .. })
                )
            }
        })
        .await;
    }
    let record = harness.get_task(id, &context()).await.unwrap().unwrap();
    assert_eq!(
        serde_json::to_value(outcome_of(&record)).unwrap(),
        json!({ "status": "completed", "result": {} })
    );
    assert!(faux.summary_requests().is_empty());
    harness.close(&context()).await.unwrap();
}

// ─── Compaction outcomes ──────────────────────────────────────────────────

#[tokio::test]
async fn completes_without_a_summary_when_there_is_nothing_to_compact() {
    let chat = open(OpenOptions::default()).await;
    turn(&chat, "hi", "hello").await;
    let outcome = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    assert_eq!(
        serde_json::to_value(&outcome).unwrap(),
        json!({ "status": "completed", "result": {} })
    );
    assert!(chat.faux.summary_requests().is_empty());
    assert!(live(&chat).await.compactions.is_none());
    chat.close().await;
}

#[tokio::test]
async fn asks_before_compact_the_first_decision_wins_a_throw_is_reported_and_skipped() {
    let chat = open(OpenOptions::default()).await;
    let seen: Arc<Mutex<Vec<crate::durable::harness::types::CompactionRequest>>> = Arc::default();
    add_before_compact(&chat.setup, |_| {
        Err(crate::durable::errors::Error::message("hook broke"))
    });
    {
        let seen = seen.clone();
        add_before_compact(&chat.setup, move |request| {
            seen.lock().push(request);
            Ok(Some(CompactionDecision::Summary("FROM HOOK".into())))
        });
    }
    decline(&chat.setup);
    history(&chat).await;
    let outcome = result(
        &chat,
        chat.root
            .compact(Some("why".into()), &context())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(outcome_status(&outcome), "completed");
    assert!(chat.faux.summary_requests().is_empty());
    assert!(context_texts(&chat.root).await[0].contains("<summary>\nFROM HOOK\n</summary>"));
    assert_eq!(chat.setup.reports.messages(), vec!["hook broke"]);
    let compaction = seen.lock()[0].clone();
    assert_eq!(
        compaction.reason,
        crate::durable::harness::types::CompactionReason::Manual
    );
    assert_eq!(compaction.instructions.as_deref(), Some("why"));
    let kinds: Vec<&str> = compaction
        .entries
        .iter()
        .map(|record| record.kind.as_str())
        .collect();
    assert_eq!(
        kinds,
        vec![
            "pi.user",
            "pi.system",
            "pi.assistant",
            "pi.user",
            "pi.assistant"
        ]
    );
    let texts: Vec<String> = compaction
        .messages
        .iter()
        .filter(|message| !matches!(message, Message::System(_)))
        .map(|message| user_text(Some(message)))
        .collect();
    assert_eq!(
        texts,
        vec![
            text("u1", 100),
            text("a1", 100),
            text("u2", 100),
            text("a2", 100)
        ]
    );
    chat.close().await;
}

#[tokio::test]
async fn completes_without_a_summary_when_a_hook_declines() {
    let chat = open(OpenOptions::default()).await;
    decline(&chat.setup);
    history(&chat).await;
    let outcome = result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    assert_eq!(
        serde_json::to_value(&outcome).unwrap(),
        json!({ "status": "completed", "result": {} })
    );
    assert!(chat.faux.summary_requests().is_empty());
    chat.close().await;
}

#[tokio::test]
async fn fails_with_no_model_without_a_configured_model() {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    chat.root
        .configure(
            AgentChange {
                model: Some(None),
                ..AgentChange::default()
            },
            &context(),
        )
        .await
        .unwrap();
    let outcome = serde_json::to_value(
        result(&chat, chat.root.compact(None, &context()).await.unwrap()).await,
    )
    .unwrap();
    assert_eq!(outcome["status"], "failed");
    assert_eq!(outcome["error"]["detail"]["reason"], "no_model");
    assert!(live(&chat).await.compactions.is_none());
    chat.close().await;
}

/// Input tokens a compaction added to the ledger.
async fn compaction_input(chat: &Chat) -> u64 {
    let before = usage_input(chat).await;
    let outcome = result(chat, chat.root.compact(None, &context()).await.unwrap()).await;
    assert_eq!(outcome_status(&outcome), "completed");
    usage_input(chat).await - before
}

#[tokio::test]
async fn retries_a_retryable_error_with_the_pinned_request_and_counts_every_attempt_once() {
    let single = open(OpenOptions::default()).await;
    history(&single).await;
    single
        .root
        .configure(
            AgentChange::default().thinking_level(ModelThinkingLevel::High),
            &context(),
        )
        .await
        .unwrap();
    single.faux.summary(summary("SUMMARY"));
    let once = compaction_input(&single).await;
    assert!(once > 0);
    single.close().await;

    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    chat.root
        .configure(
            AgentChange::default().thinking_level(ModelThinkingLevel::High),
            &context(),
        )
        .await
        .unwrap();
    chat.stream(ConversationStreamOptions {
        timeout_ms: Some(1234),
        deferred: Some(DeferredOption::Enabled(true)),
        ..ConversationStreamOptions::default()
    });
    {
        let chat = chat.clone();
        chat.faux.clone().summary(step(move |_| async move {
            // Changed during the attempt: the retry still uses the pinned request.
            chat.root
                .configure(
                    AgentChange::default().thinking_level(ModelThinkingLevel::Low),
                    &context(),
                )
                .await
                .unwrap();
            chat.stream(ConversationStreamOptions {
                timeout_ms: Some(1),
                ..ConversationStreamOptions::default()
            });
            failure("overloaded")
        }));
    }
    chat.faux.summary(summary("SUMMARY"));
    let twice = compaction_input(&chat).await;
    assert_eq!(chat.faux.summary_requests().len(), 2);
    assert_eq!(twice, 2 * once);
    let session_id = provider_session(&chat, chat.root.id).await;
    for request in chat.faux.summary_requests() {
        assert_eq!(
            request.options.reasoning,
            Some(crate::types::ThinkingLevel::High)
        );
        assert_eq!(request.options.stream.timeout_ms, Some(1234));
        assert_eq!(
            request.options.stream.cache_retention,
            Some(crate::types::CacheRetention::None)
        );
        assert_eq!(request.options.stream.session_id, session_id);
        assert!(request.options.deferred.is_none());
    }
    chat.close().await;
}

#[tokio::test]
async fn adds_no_usage_when_a_hook_declines_or_supplies_the_summary() {
    for decision in [
        CompactionDecision::Decline,
        CompactionDecision::Summary("HOOK".into()),
    ] {
        let chat = open(OpenOptions::default()).await;
        add_before_compact(&chat.setup, move |_| Ok(Some(decision.clone())));
        history(&chat).await;
        assert_eq!(compaction_input(&chat).await, 0);
        chat.close().await;
    }
}

#[tokio::test]
async fn caps_max_tokens_at_the_models_output_limit_and_sends_no_tools() {
    let chat = open(OpenOptions::default()).await;
    add_tool(
        &chat.setup.registry,
        define_tool(
            "read",
            "read",
            json!({ "type": "object", "properties": {} }),
            |_, _, _| async {
                Ok(ToolExecutionResult {
                    content: Some(Vec::new()),
                    ..ToolExecutionResult::default()
                })
            },
        ),
    );
    chat.root
        .configure(
            AgentChange::default().tools(tools_named(&chat.setup, &["read"])),
            &context(),
        )
        .await
        .unwrap();
    history(&chat).await;
    chat.policy(crate::durable::harness::types::CompactionPolicy {
        reserve_tokens: 2000,
        ..MANUAL
    });
    chat.faux.summary(summary("SUMMARY"));
    result(&chat, chat.root.compact(None, &context()).await.unwrap()).await;
    let request = chat.faux.summary_requests()[0].clone();
    // 0.8 * 2000 = 1600, above the model's 900.
    assert_eq!(request.options.stream.max_tokens, Some(900));
    assert_eq!(request.messages.len(), 2);
    assert!(!request.messages.iter().any(|message| matches!(
        message,
        Message::System(system) if system.tools_added.is_some()
    )));
    chat.close().await;
}

async fn fails_with_model_error(
    response: crate::types::AssistantMessage,
    message: &str,
    attempts: usize,
) {
    let chat = open(OpenOptions::default()).await;
    history(&chat).await;
    for _ in 0..3 {
        chat.faux.summary(response.clone());
    }
    let outcome = serde_json::to_value(
        result(&chat, chat.root.compact(None, &context()).await.unwrap()).await,
    )
    .unwrap();
    assert_eq!(outcome["status"], "failed");
    assert_eq!(outcome["error"]["message"], message);
    assert_eq!(outcome["error"]["detail"]["reason"], "model_error");
    // The retry policy allows two retries after the first attempt.
    assert_eq!(chat.faux.summary_requests().len(), attempts);
    assert!(live(&chat).await.compactions.is_none());
    assert!(
        !kinds(&chat.root)
            .await
            .contains(&"pi.compaction".to_string())
    );
    chat.close().await;
}

#[tokio::test]
async fn fails_with_model_error_on_retries_run_out() {
    fails_with_model_error(failure("overloaded"), "Summarization failed: overloaded", 3).await;
}

#[tokio::test]
async fn fails_with_model_error_on_a_non_retryable_error() {
    fails_with_model_error(
        failure("bad request"),
        "Summarization failed: bad request",
        1,
    )
    .await;
}

#[tokio::test]
async fn fails_with_model_error_on_a_length_stop() {
    fails_with_model_error(
        faux_assistant_message(
            "partial",
            FauxMessageOptions {
                stop_reason: Some(StopReason::Length),
                ..FauxMessageOptions::default()
            },
        ),
        "Summarization hit the token limit; the summary is incomplete",
        1,
    )
    .await;
}

#[tokio::test]
async fn fails_with_model_error_on_a_tool_call() {
    fails_with_model_error(
        faux_assistant_message(
            vec![faux_tool_call("read", json!({}), None)],
            FauxMessageOptions {
                stop_reason: Some(StopReason::Stop),
                ..FauxMessageOptions::default()
            },
        ),
        "Summarization attempted to call a tool",
        1,
    )
    .await;
}

#[tokio::test]
async fn fails_with_model_error_on_empty_text() {
    fails_with_model_error(answer("  "), "Summarization produced no text", 1).await;
}
