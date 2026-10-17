//! Port of `test/harness-generation.test.ts`.
//!
//! Divergences: the TS `Proxy` over `Models` is a proxied faux provider ([`proxy_models`]). The faulting case makes
//! the classification commit fail with a non-finite usage cost, which strict JSON rejects, instead of a function value.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use serde_json::json;

use super::chat::*;
use super::support::{add_section, context, to_json};
use crate::durable::documents::define_doc;
use crate::durable::entries::{ASSISTANT_ENTRY, USER_ENTRY};
use crate::durable::errors::Error;
use crate::durable::harness::RegistryReader;
use crate::durable::harness::agent::resolve_settings;
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::live::{LIVE_DOC, LiveState, RunState};
use crate::durable::harness::provider::PROVIDER_DOC;
use crate::durable::harness::scheduler::AbortTaskResult;
use crate::durable::harness::types::{
    AgentChange, CompactionPolicyOverrides, ConversationCreateOptions, ConversationRetryPolicy,
    ConversationStreamOptions, DeferredOption, ExtensionDefinition, HarnessOptions,
    HarnessSettings, RetryPolicyOverrides, SubmissionDraft, WhenBusy,
};
use crate::durable::harness::{
    Conversation, CreateOptions, Harness, create_registry, define_extension, section, wrap_section,
};
use crate::durable::session::tests::support::ControlledStorage;
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{Task, TaskDefinition, define_task};
use crate::durable::types::{
    DocDefinition, DocumentAddress, DocumentContent, DocumentPoint, DocumentScope, JsonObject,
    LatestConversation, LatestFork, Storage, StorageWrite, SubmissionCreate, SubmissionSettlement,
    SubmissionStatus, SubmissionType, TaskOptions, TaskOwnership, TaskQuery, TypedEntryDraft,
};
use crate::providers::faux::{
    FauxDeferredOptions, FauxMessageOptions, FauxResponseStep, FauxTokenSize,
    RegisterFauxProviderOptions, faux_assistant_message,
};
use crate::types::{
    AssistantMessageEvent, DeferredHandle, Message, ModelThinkingLevel, StopReason, ThinkingLevel,
    UserMessage,
};

fn error_503() -> FauxResponseStep {
    faux_assistant_message(
        Vec::new(),
        FauxMessageOptions {
            stop_reason: Some(StopReason::Error),
            error_message: Some("503 Service Unavailable".into()),
            ..FauxMessageOptions::default()
        },
    )
    .into()
}

fn answer(text: &str) -> FauxResponseStep {
    faux_assistant_message(text, FauxMessageOptions::default()).into()
}

fn input(text: &str) -> SubmissionDraft {
    SubmissionDraft::input(text)
}

async fn kinds(conversation: &Conversation) -> Vec<String> {
    entry_kinds(conversation).await
}

async fn tasks_of(
    harness: &Harness,
    conversation: &Conversation,
) -> Vec<crate::durable::types::TaskRecord> {
    let id = conversation.id;
    harness
        .commit(
            move |tx| async move {
                Ok(tx
                    .scan_tasks(
                        TaskQuery {
                            conversation_id: Some(id),
                            ..TaskQuery::default()
                        },
                        10,
                        None,
                    )
                    .await?
                    .items)
            },
            &context(),
        )
        .await
        .unwrap()
}

fn empty_live() -> Option<LiveState> {
    Some(LiveState::default())
}

#[tokio::test]
async fn answers_an_input_and_settles_its_submission() {
    let setup = chat_setup();
    add_text_section(&setup.registry, "preamble", "You are helpful.", Some(false));
    setup.faux.set_responses([answer("Hello there")]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    harness.resume().unwrap();
    let submission = root.submit(input("hi"), &context()).await.unwrap();
    let settled = submission.wait(&context()).await.unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    let answer_id = settled.answer.unwrap();
    let entry = root
        .commit(
            move |tx| async move { tx.entry(answer_id).await },
            &context(),
        )
        .await
        .unwrap()
        .unwrap();
    assert!(ASSISTANT_ENTRY.is(Some(&entry)));
    assert_eq!(
        text_of(entry.model.as_ref().unwrap().first()).as_deref(),
        Some("Hello there")
    );
    let entries = all_entries(&root).await;
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>(),
        ["pi.user", "pi.system", "pi.assistant"]
    );
    assert_eq!(Some(entries[0].id), settled.entry);
    let system = to_json(&entries[1].model);
    assert_eq!(
        system[0]["sections"],
        json!({ "preamble": "You are helpful." })
    );
    assert_eq!(system[0]["content"], json!(""));
    assert_eq!(live_state(&harness, &root).await, empty_live());
    let task = tasks_of(&harness, &root).await.remove(0);
    assert_eq!(task.kind, "pi.generation");
    // Entries written by the generation are attributed to it; the admitted user entry is not task work.
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.by_task_id)
            .collect::<Vec<_>>(),
        [None, Some(task.id), Some(task.id)]
    );
    assert_eq!(
        to_json(&task.state),
        json!({
            "status": "terminal",
            "outcome": { "status": "completed", "result": { "entryId": answer_id } },
        })
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn stores_partials_as_deltas_and_a_complete_base_once_nothing_is_in_flight() {
    let setup = chat_setup_with(RegisterFauxProviderOptions {
        tokens_per_second: Some(200.0),
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    setup.faux.set_responses([answer(&"w".repeat(200))]);
    let storage = Arc::new(ControlledStorage::new());
    let (harness, root) = open_chat(storage.clone(), &setup).await;
    let record = storage
        .find_document(
            &DocumentAddress {
                kind: "pi.live".into(),
                scope: DocumentScope::Conversation {
                    conversation_id: root.id,
                },
                key: None,
            },
            DocumentPoint::Current,
            &context(),
        )
        .await
        .unwrap()
        .unwrap();
    harness.resume().unwrap();
    root.submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    let contents: Vec<bool> = storage
        .commits
        .lock()
        .iter()
        .flatten()
        .filter_map(|write| match write {
            StorageWrite::DocumentChange { id, content } if *id == record.id => {
                Some(matches!(content, DocumentContent::Base { .. }))
            }
            _ => None,
        })
        .collect();
    // Streaming writes deltas; the commit that settles the answer clears generation and writes a base.
    assert!(contents.contains(&false));
    assert_eq!(contents.last(), Some(&true));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn still_ends_a_run_whose_input_something_else_already_settled() {
    let setup = chat_setup();
    let (step, reached) = unanswered();
    setup.faux.set_responses([step]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let submission = root.submit(input("hi"), &context()).await.unwrap();
    reached.wait().await;
    let id = submission.id;
    root.commit(
        move |tx| async move {
            tx.settle_submission(
                id,
                SubmissionSettlement::Unanswered {
                    reason: "withdrawn".into(),
                    detail: None,
                },
            )
        },
        &context(),
    )
    .await
    .unwrap();
    let task_id = run_task(&harness, &root).await;
    harness.abort_task(task_id, &context()).await.unwrap();
    let record = harness.wait_for_task(task_id, &context()).await.unwrap();
    assert_eq!(
        to_json(&record.state)["outcome"],
        json!({ "status": "aborted" })
    );
    // The earlier settlement stays; the run's own settlement leaves it unchanged.
    let status = submission.status(&context()).await.unwrap();
    assert_eq!(status.status, SubmissionStatus::Unanswered);
    assert_eq!(status.reason.as_deref(), Some("withdrawn"));
    assert_eq!(live_state(&harness, &root).await, empty_live());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn fails_with_no_model_when_no_model_is_configured_or_the_model_is_unknown() {
    let setup = chat_setup();
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let plain = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    harness.resume().unwrap();
    let unset = plain
        .submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(unset.status, SubmissionStatus::Unanswered);
    assert_eq!(unset.reason.as_deref(), Some("no_model"));

    root.configure(
        AgentChange::default().model(crate::durable::harness::types::ModelRef::new(
            "faux", "missing",
        )),
        &context(),
    )
    .await
    .unwrap();
    let unknown = root
        .submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(unknown.reason.as_deref(), Some("no_model"));
    assert!(unknown.entry.is_some());
    assert_eq!(kinds(&root).await, ["pi.user"]);
    let tasks = tasks_of(&harness, &root).await;
    assert_eq!(
        to_json(&tasks[0].state),
        json!({
            "status": "terminal",
            "outcome": {
                "status": "failed",
                "error": { "message": "Model faux/missing is not available", "detail": { "reason": "no_model" } },
            },
        })
    );
    assert_eq!(live_state(&harness, &root).await, empty_live());
    assert_eq!(live_state(&harness, &plain).await, empty_live());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn retries_a_retryable_error_after_a_durable_backoff_and_then_answers() {
    let setup = chat_setup();
    add_text_section(&setup.registry, "preamble", "p", Some(false));
    setup.faux.set_responses([error_503(), answer("recovered")]);
    setup.settings(|settings| {
        settings.retry = Some(RetryPolicyOverrides {
            enabled: Some(true),
            max_retries: Some(3),
            base_delay_ms: Some(1),
            max_agent_delay_ms: None,
        })
    });
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let values = live_publications(&harness);
    harness.resume().unwrap();
    let settled = root
        .submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    let entries = all_entries(&root).await;
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>(),
        ["pi.user", "pi.system", "pi.assistant", "pi.assistant"]
    );
    assert_eq!(to_json(&entries[2].model)[0]["stopReason"], json!("error"));
    let values = std::mem::take(&mut *values.lock());
    assert!(values.iter().any(|value| {
        value
            .generation
            .as_ref()
            .and_then(|generation| generation.retry.as_ref())
            .is_some_and(|retry| retry.error == "503 Service Unavailable")
    }));
    assert!(values.iter().any(|value| {
        value
            .generation
            .as_ref()
            .is_some_and(|generation| generation.attempt == 2)
    }));
    drop(values);
    assert_eq!(live_state(&harness, &root).await, empty_live());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn fails_with_model_error_once_retries_are_exhausted() {
    let setup = chat_setup();
    setup
        .faux
        .set_responses([error_503(), error_503(), answer("never")]);
    setup.settings(|settings| {
        settings.retry = Some(RetryPolicyOverrides {
            enabled: Some(true),
            max_retries: Some(1),
            base_delay_ms: Some(1),
            max_agent_delay_ms: None,
        })
    });
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    harness.resume().unwrap();
    let settled = root
        .submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Unanswered);
    assert_eq!(settled.reason.as_deref(), Some("model_error"));
    assert_eq!(settled.detail, Some(json!("503 Service Unavailable")));
    assert_eq!(
        kinds(&root).await,
        ["pi.user", "pi.assistant", "pi.assistant"]
    );
    assert_eq!(setup.faux.get_pending_response_count(), 1);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn fails_a_retryable_error_without_retrying_when_the_retry_policy_is_disabled() {
    let setup = chat_setup();
    setup.faux.set_responses([error_503(), answer("never")]);
    setup.settings(|settings| {
        settings.retry = Some(RetryPolicyOverrides {
            enabled: Some(false),
            ..RetryPolicyOverrides::default()
        })
    });
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    harness.resume().unwrap();
    let settled = root
        .submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.reason.as_deref(), Some("model_error"));
    assert_eq!(setup.faux.state().call_count, 1);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reports_section_wrapper_failures_while_preparing() {
    let setup = chat_setup();
    add_text_section(&setup.registry, "cwd", "/repo", None);
    setup
        .registry
        .install(define_extension(ExtensionDefinition {
            wraps: vec![wrap_section("cwd", |_| {
                Err(Error::message("wrapper failed"))
            })],
            ..ExtensionDefinition::new("broken")
        }))
        .unwrap();
    setup.faux.set_responses([answer("ok")]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    harness.resume().unwrap();
    let settled = root
        .submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    assert!(
        setup
            .reports
            .messages()
            .contains(&"wrapper failed".to_string())
    );
    // The failed section is absent, so nothing was rendered.
    assert_eq!(kinds(&root).await, ["pi.user", "pi.assistant"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn fails_a_non_retryable_error_without_retrying() {
    let setup = chat_setup();
    setup.faux.set_responses([
        faux_assistant_message(
            Vec::new(),
            FauxMessageOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("Invalid request".into()),
                ..FauxMessageOptions::default()
            },
        )
        .into(),
        answer("never"),
    ]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    harness.resume().unwrap();
    let settled = root
        .submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.reason.as_deref(), Some("model_error"));
    assert_eq!(settled.detail, Some(json!("Invalid request")));
    let tasks = tasks_of(&harness, &root).await;
    assert_eq!(
        to_json(&tasks[0].state)["outcome"],
        json!({ "status": "failed", "error": { "message": "Invalid request", "detail": { "reason": "model_error" } } })
    );
    harness.close(&context()).await.unwrap();
}

fn deferred_setup(pending_fetches: u32, poll_after_ms: u64) -> ChatSetup {
    let setup = chat_setup_with(RegisterFauxProviderOptions {
        deferred: Some(FauxDeferredOptions {
            pending_fetches: Some(pending_fetches),
            poll_after_ms: Some(poll_after_ms),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    setup.settings(|settings| {
        settings.stream = Some(ConversationStreamOptions {
            deferred: Some(DeferredOption::Enabled(true)),
            ..ConversationStreamOptions::default()
        })
    });
    setup
}

#[tokio::test]
async fn polls_a_deferred_response_until_it_is_ready() {
    let setup = deferred_setup(1, 1);
    setup.faux.set_responses([answer("deferred answer")]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let values = live_publications(&harness);
    harness.resume().unwrap();
    let settled = root
        .submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    assert_eq!(setup.faux.state().deferred_fetch_count, 2);
    let poll_times: Vec<u64> = values
        .lock()
        .iter()
        .filter_map(|value| {
            value
                .generation
                .as_ref()
                .and_then(|generation| generation.deferred)
                .map(|deferred| deferred.poll_at)
        })
        .collect();
    assert_eq!(poll_times.len(), 2);
    assert!(poll_times[1] > poll_times[0]);
    let answer_id = settled.answer.unwrap();
    let entry = root
        .commit(
            move |tx| async move { tx.entry(answer_id).await },
            &context(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        text_of(entry.model.as_ref().unwrap().first()).as_deref(),
        Some("deferred answer")
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn converts_the_committed_partial_when_aborted_during_streaming() {
    let setup = chat_setup_with(RegisterFauxProviderOptions {
        tokens_per_second: Some(20.0),
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    setup.faux.set_responses([answer(&"x".repeat(400))]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    harness.resume().unwrap();
    let submission = root.submit(input("hi"), &context()).await.unwrap();
    let task_id = run_task(&harness, &root).await;
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.generation)
            .and_then(|generation| generation.message)
            .is_some_and(|message| text_of(Some(&Message::Assistant(message))).is_some())
    })
    .await;
    let partial = live_state(&harness, &root)
        .await
        .unwrap()
        .generation
        .unwrap()
        .message
        .unwrap();
    assert_eq!(
        harness.abort_task(task_id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    let settled = submission.wait(&context()).await.unwrap();
    assert_eq!(settled.reason.as_deref(), Some("aborted"));
    let entries = all_entries(&root).await;
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>(),
        ["pi.user", "pi.assistant"]
    );
    let converted = entries[1].model.as_ref().unwrap()[0].clone();
    let Message::Assistant(message) = &converted else {
        panic!("assistant");
    };
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert!(
        text_of(Some(&converted))
            .unwrap()
            .starts_with(&text_of(Some(&Message::Assistant(partial))).unwrap())
    );
    assert_eq!(live_state(&harness, &root).await, empty_live());
    let record = harness.wait_for_task(task_id, &context()).await.unwrap();
    assert_eq!(
        to_json(&record.state)["outcome"],
        json!({ "status": "aborted" })
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cancels_a_deferred_response_when_aborted_during_polling() {
    let setup = deferred_setup(100, 60_000);
    setup.faux.set_responses([answer("never")]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    harness.resume().unwrap();
    let submission = root.submit(input("hi"), &context()).await.unwrap();
    let task_id = run_task(&harness, &root).await;
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.generation)
            .is_some_and(|generation| generation.deferred.is_some())
    })
    .await;
    harness.abort_task(task_id, &context()).await.unwrap();
    let settled = submission.wait(&context()).await.unwrap();
    assert_eq!(settled.reason.as_deref(), Some("aborted"));
    assert_eq!(setup.faux.state().cancelled_deferred.len(), 1);
    assert_eq!(live_state(&harness, &root).await, empty_live());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reports_a_failed_deferred_cancellation_and_still_ends_the_run_aborted() {
    let setup = deferred_setup(100, 60_000);
    setup.faux.set_responses([answer("never")]);
    let models = proxy_models(
        &setup,
        ProxyOverrides {
            cancel_error: Some("cancel failed".into()),
            ..ProxyOverrides::default()
        },
    );
    let (harness, root) = open_chat_with(
        Arc::new(MemoryStorage::new()),
        &setup,
        ChatOptions {
            models: Some(models),
            ..ChatOptions::default()
        },
    )
    .await;
    harness.resume().unwrap();
    let submission = root.submit(input("hi"), &context()).await.unwrap();
    let task_id = run_task(&harness, &root).await;
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.generation)
            .is_some_and(|generation| generation.deferred.is_some())
    })
    .await;
    harness.abort_task(task_id, &context()).await.unwrap();
    let settled = submission.wait(&context()).await.unwrap();
    assert_eq!(settled.reason.as_deref(), Some("aborted"));
    assert!(
        setup
            .reports
            .messages()
            .iter()
            .any(|message| message.contains("cancel failed"))
    );
    assert_eq!(live_state(&harness, &root).await, empty_live());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn forwards_stream_options_and_the_thinking_level() {
    let setup = chat_setup();
    let seen: Arc<Mutex<Vec<crate::types::SimpleStreamOptions>>> = Arc::default();
    let capture = |text: &'static str, seen: Arc<Mutex<Vec<crate::types::SimpleStreamOptions>>>| {
        FauxResponseStep::factory(move |_, options, _, _| {
            seen.lock().push(options);
            Ok(faux_assistant_message(text, FauxMessageOptions::default()))
        })
    };
    setup
        .faux
        .set_responses([capture("a", seen.clone()), capture("b", seen.clone())]);
    setup.settings(|settings| {
        settings.stream = Some(ConversationStreamOptions {
            timeout_ms: Some(1234),
            headers: Some(
                [("x-test".to_string(), "1".to_string())]
                    .into_iter()
                    .collect(),
            ),
            ..ConversationStreamOptions::default()
        })
    });
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    root.configure(
        AgentChange::default().thinking_level(ModelThinkingLevel::High),
        &context(),
    )
    .await
    .unwrap();
    harness.resume().unwrap();
    root.submit(input("one"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    // Both are read at the next preparation: the thinking level from pi.agent, the stream options live from settings.
    root.configure(
        AgentChange {
            thinking_level: Some(None),
            ..AgentChange::default()
        },
        &context(),
    )
    .await
    .unwrap();
    setup.settings(|settings| {
        settings.stream = Some(ConversationStreamOptions {
            timeout_ms: Some(99),
            ..ConversationStreamOptions::default()
        })
    });
    root.submit(input("two"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    let session_id = harness
        .snapshot(&*PROVIDER_DOC, root.id, &context())
        .await
        .unwrap()
        .unwrap()
        .session_id;
    let seen = std::mem::take(&mut *seen.lock());
    assert_eq!(seen[0].stream.timeout_ms, Some(1234));
    assert_eq!(
        seen[0]
            .stream
            .headers
            .as_ref()
            .and_then(|headers| headers.get("x-test").cloned()),
        Some(Some("1".to_string()))
    );
    assert_eq!(seen[0].reasoning, Some(ThinkingLevel::High));
    assert_eq!(
        seen[0].stream.session_id.as_deref(),
        Some(session_id.as_str())
    );
    assert!(seen[0].stream.signal.is_some());
    assert_eq!(seen[1].reasoning, None);
    assert_eq!(seen[1].stream.timeout_ms, Some(99));
    assert_eq!(
        seen[1].stream.session_id.as_deref(),
        Some(session_id.as_str())
    );
    assert!(seen[1].stream.headers.is_none());
    drop(seen);
    harness.close(&context()).await.unwrap();
}

// Regression coverage for #10424.
#[tokio::test]
async fn keeps_provider_session_ids_request_local_across_concurrent_conversations() {
    let setup = chat_setup();
    let seen: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let capture = || {
        let seen = seen.clone();
        FauxResponseStep::factory(move |request, options, _, _| {
            let text = request
                .messages
                .iter()
                .rev()
                .find_map(|message| match message {
                    Message::User(UserMessage {
                        content: crate::types::UserMessageContent::Text(text),
                        ..
                    }) => Some(text.clone()),
                    Message::User(_) => Some(String::new()),
                    _ => None,
                })
                .unwrap_or_default();
            seen.lock().push((
                text.clone(),
                options.stream.session_id.clone().unwrap_or_default(),
            ));
            Ok(faux_assistant_message(
                format!("answer:{text}").as_str(),
                FauxMessageOptions::default(),
            ))
        })
    };
    setup
        .faux
        .set_responses([capture(), capture(), capture(), capture()]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let mut options = ConversationCreateOptions::ownerless();
    options.agent = Some(AgentChange::default().model(faux_model()));
    let child = harness
        .create_conversation(options, &context())
        .await
        .unwrap();
    harness.resume().unwrap();
    for round in ["first", "second"] {
        let waits = [&root, &child]
            .into_iter()
            .enumerate()
            .map(|(index, conversation)| {
                let text = format!("{round}-{index}");
                async move {
                    conversation
                        .submit(input(&text), &context())
                        .await
                        .unwrap()
                        .wait(&context())
                        .await
                        .unwrap();
                }
            });
        futures::future::join_all(waits).await;
    }
    let root_id = harness
        .snapshot(&*PROVIDER_DOC, root.id, &context())
        .await
        .unwrap()
        .unwrap()
        .session_id;
    let child_id = harness
        .snapshot(&*PROVIDER_DOC, child.id, &context())
        .await
        .unwrap()
        .unwrap()
        .session_id;
    assert_ne!(root_id, child_id);
    let seen = std::mem::take(&mut *seen.lock());
    let of = |text: &str| {
        seen.iter()
            .filter(|(sent, _)| sent == text)
            .map(|(_, id)| id.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(of("first-0"), std::slice::from_ref(&root_id));
    assert_eq!(of("second-0"), std::slice::from_ref(&root_id));
    assert_eq!(of("first-1"), std::slice::from_ref(&child_id));
    assert_eq!(of("second-1"), std::slice::from_ref(&child_id));
    drop(seen);
    harness.close(&context()).await.unwrap();
}

// Regression coverage for #10424.
#[tokio::test]
async fn creates_and_persists_provider_state_before_a_legacy_conversations_request() {
    let setup = chat_setup();
    let sent: Arc<Mutex<Option<String>>> = Arc::default();
    let capture = sent.clone();
    setup
        .faux
        .set_responses([FauxResponseStep::factory(move |_, options, _, _| {
            *capture.lock() = options.stream.session_id.clone();
            Ok(faux_assistant_message("ok", FauxMessageOptions::default()))
        })]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let id = root.id;
    root.commit(
        move |tx| async move { tx.retire_doc(&*PROVIDER_DOC, id).await },
        &context(),
    )
    .await
    .unwrap();
    assert!(
        harness
            .snapshot(&*PROVIDER_DOC, root.id, &context())
            .await
            .unwrap()
            .is_none()
    );
    harness.resume().unwrap();
    root.submit(input("legacy"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    let stored = harness
        .snapshot(&*PROVIDER_DOC, root.id, &context())
        .await
        .unwrap()
        .unwrap()
        .session_id;
    let pattern =
        regex::Regex::new("^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
            .unwrap();
    assert!(pattern.is_match(&stored));
    assert_eq!(sent.lock().as_deref(), Some(stored.as_str()));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reads_settings_through_getters_at_every_decision() {
    let setup = chat_setup();
    let timeout_ms = Arc::new(AtomicU64::new(111));
    let seen: Arc<Mutex<Vec<Option<u64>>>> = Arc::default();
    let (first_seen, first_timeout) = (seen.clone(), timeout_ms.clone());
    let second_seen = seen.clone();
    setup.faux.set_responses([
        FauxResponseStep::factory(move |_, options, _, _| {
            first_seen.lock().push(options.stream.timeout_ms);
            // The user changes the setting while the first attempt runs.
            first_timeout.store(222, Ordering::SeqCst);
            Ok(faux_assistant_message(
                Vec::new(),
                FauxMessageOptions {
                    stop_reason: Some(StopReason::Error),
                    error_message: Some("503 Service Unavailable".into()),
                    ..FauxMessageOptions::default()
                },
            ))
        }),
        FauxResponseStep::factory(move |_, options, _, _| {
            second_seen.lock().push(options.stream.timeout_ms);
            Ok(faux_assistant_message("ok", FauxMessageOptions::default()))
        }),
    ]);
    let mut options = HarnessOptions::new(setup.models.clone(), Arc::new(setup.registry.clone()));
    let timeout = timeout_ms.clone();
    options.settings = Some(Arc::new(move || HarnessSettings {
        stream: Some(ConversationStreamOptions {
            timeout_ms: Some(timeout.load(Ordering::SeqCst)),
            ..ConversationStreamOptions::default()
        }),
        retry: Some(RetryPolicyOverrides {
            base_delay_ms: Some(1),
            ..RetryPolicyOverrides::default()
        }),
        ..HarnessSettings::default()
    }));
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context())
        .await
        .unwrap();
    let root = harness
        .root(
            &context(),
            CreateOptions {
                agent: Some(AgentChange::default().model(faux_model())),
                init: None,
            },
        )
        .await
        .unwrap();
    harness.resume().unwrap();
    let settled = root
        .submit(input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    // The retry prepares again, so it resolves the settings again and sends the new timeout.
    assert_eq!(*seen.lock(), [Some(111), Some(222)]);
    harness.close(&context()).await.unwrap();
}

#[test]
fn resolves_settings_over_the_built_in_defaults() {
    let settings = resolve_settings(None);
    assert!(settings.extensions.is_none());
    assert_eq!(settings.stream, ConversationStreamOptions::default());
    assert_eq!(
        settings.retry,
        ConversationRetryPolicy {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 2000,
            max_agent_delay_ms: Some(60000),
        }
    );
    assert_eq!(
        to_json(&settings.compaction),
        json!({ "enabled": true, "reserveTokens": 16384, "keepRecentTokens": 20000, "backgroundTokens": 32768 })
    );
    assert_eq!(to_json(&settings.tool_execution), json!("parallel"));
    assert_eq!(to_json(&settings.steering_mode), json!("one-at-a-time"));
    assert_eq!(to_json(&settings.follow_up_mode), json!("one-at-a-time"));
    let changed = resolve_settings(Some(&HarnessSettings {
        retry: Some(RetryPolicyOverrides {
            enabled: Some(false),
            ..RetryPolicyOverrides::default()
        }),
        compaction: Some(CompactionPolicyOverrides {
            background_tokens: Some(0),
            ..CompactionPolicyOverrides::default()
        }),
        ..HarnessSettings::default()
    }));
    assert!(!changed.retry.enabled);
    assert_eq!(changed.retry.max_retries, 3);
    assert_eq!(changed.retry.base_delay_ms, 2000);
    assert!(changed.compaction.enabled);
    assert_eq!(changed.compaction.background_tokens, 0);
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct TestAgent {
    cwd: String,
    kind: String,
}

#[tokio::test]
async fn renders_sections_that_read_conversation_documents_through_input_read() {
    let agent_doc = Arc::new(
        define_doc(DocDefinition::new(
            "test.agent",
            1,
            LatestConversation {
                fork: LatestFork::Current,
            },
            || TestAgent {
                cwd: "/".into(),
                kind: "main".into(),
            },
        ))
        .unwrap(),
    );
    let setup = chat_setup();
    let doc = agent_doc.clone();
    add_section(
        &setup.registry,
        section(
            "cwd",
            move |input, ctx| {
                let doc = doc.clone();
                async move {
                    Ok(input
                        .read
                        .snapshot(&*doc, input.conversation_id, &ctx)
                        .await?
                        .map(|agent| agent.cwd))
                }
            },
            None,
        ),
    );
    let doc = agent_doc.clone();
    add_section(
        &setup.registry,
        section(
            "agents",
            move |input, ctx| {
                let doc = doc.clone();
                async move {
                    let agent = input
                        .read
                        .snapshot(&*doc, input.conversation_id, &ctx)
                        .await?;
                    Ok(if agent.is_some_and(|agent| agent.kind == "sub") {
                        None
                    } else {
                        Some("Read AGENTS.md".to_string())
                    })
                }
            },
            None,
        ),
    );
    setup.faux.set_responses([answer("a"), answer("b")]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let (doc, id) = (agent_doc.clone(), root.id);
    root.commit(
        move |tx| async move {
            tx.doc(&*doc, id)
                .await?
                .edit(|agent| agent.cwd = "/repo".into())
        },
        &context(),
    )
    .await
    .unwrap();
    let mut options = ConversationCreateOptions::ownerless();
    options.agent = Some(AgentChange::default().model(faux_model()));
    let doc = agent_doc.clone();
    options.init = Some(Arc::new(move |tx, id| {
        let doc = doc.clone();
        Box::pin(async move {
            tx.doc(&*doc, id).await?.edit(|agent| {
                agent.kind = "sub".into();
                agent.cwd = "/sub".into();
            })
        })
    }));
    let sub = harness
        .create_conversation(options, &context())
        .await
        .unwrap();
    harness.resume().unwrap();
    for (conversation, text) in [(&root, "one"), (&sub, "two")] {
        conversation
            .submit(input(text), &context())
            .await
            .unwrap()
            .wait(&context())
            .await
            .unwrap();
    }
    async fn sections(conversation: &Conversation) -> serde_json::Value {
        let entry = all_entries(conversation)
            .await
            .into_iter()
            .find(|entry| entry.kind == "pi.system")
            .unwrap();
        to_json(&entry.model.unwrap()[0])
    }
    assert_eq!(
        sections(&root).await["sections"],
        json!({ "cwd": "<cwd>\n/repo\n</cwd>", "agents": "<agents>\nRead AGENTS.md\n</agents>" })
    );
    let sub_system = sections(&sub).await;
    assert_eq!(
        sub_system["sections"],
        json!({ "cwd": "<cwd>\n/sub\n</cwd>" })
    );
    assert_eq!(sub_system["content"], json!(""));
    assert!(sub_system.get("toolsAdded").is_none());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn commits_no_partial_for_a_response_that_turns_deferred_after_an_empty_start_event() {
    let setup = chat_setup();
    let handle = DeferredHandle {
        provider: "faux".into(),
        model_id: "faux-1".into(),
        api: "faux".into(),
        id: "handle-1".into(),
        expires_at: None,
        poll_after_ms: Some(60_000),
        data: None,
    };
    let models = proxy_models(
        &setup,
        ProxyOverrides {
            stream_simple: Some(Arc::new(move |_, _, _| {
                let start = faux_assistant_message(
                    Vec::new(),
                    FauxMessageOptions {
                        stop_reason: Some(StopReason::Pending),
                        ..FauxMessageOptions::default()
                    },
                );
                let last = faux_assistant_message(
                    Vec::new(),
                    FauxMessageOptions {
                        stop_reason: Some(StopReason::Deferred),
                        deferred: Some(handle.clone()),
                        ..FauxMessageOptions::default()
                    },
                );
                // Longer than the partial throttle: an empty partial would be committed here.
                scripted_stream(
                    vec![AssistantMessageEvent::Start { partial: start }],
                    300,
                    last,
                )
            })),
            ..ProxyOverrides::default()
        },
    );
    let (harness, root) = open_chat_with(
        Arc::new(MemoryStorage::new()),
        &setup,
        ChatOptions {
            models: Some(models),
            ..ChatOptions::default()
        },
    )
    .await;
    let values = live_publications(&harness);
    let submission = root.submit(input("hi"), &context()).await.unwrap();
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.generation)
            .is_some_and(|generation| generation.deferred.is_some())
    })
    .await;
    assert!(!values.lock().iter().any(|value| {
        value
            .generation
            .as_ref()
            .is_some_and(|generation| generation.message.is_some())
    }));
    let task_id = live_state(&harness, &root)
        .await
        .unwrap()
        .run
        .unwrap()
        .task_id;
    harness.abort_task(task_id, &context()).await.unwrap();
    let settled = submission.wait(&context()).await.unwrap();
    assert_eq!(settled.reason.as_deref(), Some("aborted"));
    assert_eq!(kinds(&root).await, ["pi.user"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn faults_a_run_task_settling_its_inputs_and_converting_the_committed_partial() {
    let setup = chat_setup();
    let models = proxy_models(
        &setup,
        ProxyOverrides {
            stream_simple: Some(Arc::new(|_, _, _| {
                let partial = faux_assistant_message(
                    "partial",
                    FauxMessageOptions {
                        stop_reason: Some(StopReason::Pending),
                        ..FauxMessageOptions::default()
                    },
                );
                let mut last = faux_assistant_message("final", FauxMessageOptions::default());
                // Not strict JSON, so the classification commit fails and the scheduler faults the task.
                last.usage.cost.total = f64::NAN;
                scripted_stream(vec![AssistantMessageEvent::Start { partial }], 300, last)
            })),
            ..ProxyOverrides::default()
        },
    );
    let (harness, root) = open_chat_with(
        Arc::new(MemoryStorage::new()),
        &setup,
        ChatOptions {
            models: Some(models),
            ..ChatOptions::default()
        },
    )
    .await;
    let values = live_publications(&harness);
    harness.resume().unwrap();
    let submission = root.submit(input("hi"), &context()).await.unwrap();
    let settled = submission.wait(&context()).await.unwrap();
    assert_eq!(settled.reason.as_deref(), Some("faulted"));
    assert!(
        settled
            .detail
            .as_ref()
            .is_some_and(|detail| detail.is_string())
    );
    assert!(values.lock().iter().any(|value| {
        value
            .generation
            .as_ref()
            .and_then(|generation| generation.message.clone())
            .is_some_and(|message| {
                text_of(Some(&Message::Assistant(message))).as_deref() == Some("partial")
            })
    }));
    let entries = all_entries(&root).await;
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>(),
        ["pi.user", "pi.assistant"]
    );
    let converted = entries[1].model.as_ref().unwrap()[0].clone();
    assert_eq!(to_json(&converted)["stopReason"], json!("aborted"));
    assert_eq!(text_of(Some(&converted)).as_deref(), Some("partial"));
    assert_eq!(live_state(&harness, &root).await, empty_live());
    let tasks = tasks_of(&harness, &root).await;
    assert_eq!(
        to_json(&tasks[0].state)["outcome"]["status"],
        json!("faulted")
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn orphans_a_blocked_run_task_with_full_run_cleanup() {
    let setup = chat_setup();
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    // A run whose task was stored by a newer generation definition this process cannot run.
    let newer: Task<JsonObject, serde_json::Value, serde_json::Value, ()> =
        define_task(TaskDefinition::new(
            GENERATION_TASK.name(),
            2,
            |_: &JsonObject| json!({ "phase": "prepare", "attempt": 1 }),
        ));
    let id = root.id;
    let (task_id, submission_id) = harness
        .session()
        .commit_with(
            move |tx| async move {
                let entry = tx
                    .append_entry_of(
                        &USER_ENTRY,
                        id,
                        TypedEntryDraft {
                            model: Some(vec![Message::User(UserMessage {
                                content: "hi".into(),
                                timestamp: 1,
                            })]),
                            ..TypedEntryDraft::default()
                        },
                    )
                    .await?;
                let submission = tx
                    .create_submission(SubmissionCreate {
                        conversation_id: id,
                        type_: SubmissionType::Input,
                        status: SubmissionStatus::Placed,
                        entry: Some(entry.id),
                        ..SubmissionCreate::default()
                    })
                    .await?;
                let task_id = tx
                    .create_task(
                        &newer,
                        JsonObject::new(),
                        TaskOptions {
                            ownership: TaskOwnership::Conversation,
                            conversation_id: Some(id),
                            background: None,
                        },
                    )
                    .await?
                    .erase();
                tx.doc(&*LIVE_DOC, id).await?.edit(|live| {
                    live.run = Some(RunState {
                        task_id,
                        inputs: vec![submission.id],
                    })
                })?;
                Ok((task_id, submission.id))
            },
            &context(),
            Default::default(),
        )
        .await
        .unwrap();
    harness.resume().unwrap();
    let busy = root
        .submit(input("busy").when_busy(WhenBusy::Reject), &context())
        .await
        .unwrap_err();
    assert!(busy.to_string().contains("is busy"));
    assert_eq!(
        harness.abort_task(task_id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    let record = harness.wait_for_task(task_id, &context()).await.unwrap();
    assert_eq!(
        to_json(&record.state)["outcome"],
        json!({ "status": "orphaned", "reason": "task_too_old" })
    );
    let status = harness
        .submission(submission_id, &context())
        .await
        .unwrap()
        .unwrap()
        .status(&context())
        .await
        .unwrap();
    assert_eq!(status.status, SubmissionStatus::Unanswered);
    assert_eq!(status.reason.as_deref(), Some("task_too_old"));
    assert_eq!(live_state(&harness, &root).await, empty_live());
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_a_registry_without_the_built_in_tasks() {
    struct Lacking(crate::durable::harness::RegistrySnapshot);
    impl crate::durable::harness::RegistryReader for Lacking {
        fn snapshot(&self) -> crate::durable::harness::RegistrySnapshot {
            self.0.clone()
        }
        fn subscribe(
            &self,
            _listener: Arc<dyn Fn() + Send + Sync>,
        ) -> crate::durable::session::Unsubscribe {
            Box::new(|| {})
        }
    }
    let registry = create_registry();
    let state = registry.snapshot().without_task("pi.generation");
    let options = HarnessOptions::new(chat_setup().models, Arc::new(Lacking(Arc::new(state))));
    let error = Harness::open(Arc::new(MemoryStorage::new()), options, &context())
        .await
        .expect_err("rejects");
    assert!(
        error
            .to_string()
            .contains("Registry lacks built-in tasks pi.generation")
    );
}
