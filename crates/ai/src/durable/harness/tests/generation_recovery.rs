//! Port of `test/harness-generation-recovery.test.ts`.
//!
//! Divergence: the SQLite file reopened by the TS suite is `ControlledStorage::persistent()`, reopened in place.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::chat::*;
use super::support::{aborted, add_section, context, to_json};
use crate::durable::harness::live::LIVE_DOC;
use crate::durable::harness::types::{
    ConversationStreamOptions, DeferredOption, RetryPolicyOverrides, SubmissionDraft,
};
use crate::durable::harness::{Harness, section};
use crate::durable::ids::TaskId;
use crate::durable::session::tests::support::{ControlledStorage, Deferred};
use crate::durable::types::{Storage, SubmissionStatus, TaskState};
use crate::error::Error as AiError;
use crate::models::create_models;
use crate::providers::faux::{
    FauxDeferredOptions, FauxMessageOptions, FauxResponseStep, FauxTokenSize,
    RegisterFauxProviderOptions, faux_assistant_message,
};
use crate::types::{Message, StopReason};

fn answer(text: &str) -> FauxResponseStep {
    faux_assistant_message(text, FauxMessageOptions::default()).into()
}

async fn run_task_id(harness: &Harness, root: &crate::durable::harness::Conversation) -> TaskId {
    live_state(harness, root)
        .await
        .unwrap()
        .run
        .unwrap()
        .task_id
}

async fn checkpoint(harness: &Harness, id: TaskId) -> Option<JsonValue> {
    let record = harness.get_task(id, &context()).await.unwrap()?;
    match record.state {
        TaskState::Terminal { .. } => None,
        state => Some(to_json(&state)["checkpoint"].clone()),
    }
}

fn storage() -> Arc<ControlledStorage> {
    Arc::new(ControlledStorage::persistent())
}

async fn open(
    storage: &Arc<ControlledStorage>,
    setup: &ChatSetup,
) -> (Harness, crate::durable::harness::Conversation) {
    open_chat(storage.clone() as Arc<dyn Storage>, setup).await
}

fn roles(messages: &[Message]) -> Vec<&'static str> {
    messages
        .iter()
        .map(|message| match message {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "toolResult",
            Message::System(_) => "system",
        })
        .collect()
}

#[tokio::test]
async fn reruns_preparation_interrupted_before_its_commit() {
    let storage = storage();
    let setup = chat_setup();
    let reached = Deferred::default();
    let block = Arc::new(AtomicBool::new(true));
    let (gate, blocking) = (reached.clone(), block.clone());
    add_section(
        &setup.registry,
        section(
            "preamble",
            move |_, ctx| {
                let (gate, blocking) = (gate.clone(), blocking.clone());
                async move {
                    if blocking.swap(false, Ordering::SeqCst) {
                        gate.resolve();
                        aborted(ctx.abort_signal().unwrap().clone()).await?;
                    }
                    Ok(Some("p".to_string()))
                }
            },
            None,
        ),
    );
    setup.faux.set_responses([answer("answer")]);
    let (harness, root) = open(&storage, &setup).await;
    harness.resume().unwrap();
    let id = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap()
        .id;
    reached.wait().await;
    let task_id = run_task_id(&harness, &root).await;
    harness.close(&context()).await.unwrap();

    let (harness, root) = open(&storage, &setup).await;
    assert_eq!(
        checkpoint(&harness, task_id).await,
        Some(json!({ "phase": "prepare", "attempt": 1 }))
    );
    harness.resume().unwrap();
    let settled = harness
        .submission(id, &context())
        .await
        .unwrap()
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    assert_eq!(
        entry_kinds(&root).await,
        ["pi.user", "pi.system", "pi.assistant"]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn resends_a_request_interrupted_before_any_partial_without_repeating_preparation() {
    let storage = storage();
    let setup = chat_setup();
    add_text_section(&setup.registry, "preamble", "p", Some(false));
    let reached = Reached::default();
    let sent: Arc<Mutex<Vec<Vec<&'static str>>>> = Arc::default();
    let timeouts: Arc<Mutex<Vec<Option<u64>>>> = Arc::default();
    let reach = reached.clone();
    let (sent_by, timeouts_by) = (sent.clone(), timeouts.clone());
    setup.faux.set_responses([
        FauxResponseStep::async_factory(move |_, options, _, _| {
            let reach = reach.clone();
            async move {
                reach.reach();
                options.stream.signal.clone().unwrap().cancelled().await;
                Err(AiError::Aborted("Request aborted".into()))
            }
        }),
        FauxResponseStep::factory(move |request, options, _, _| {
            sent_by.lock().push(roles(&request.messages));
            timeouts_by.lock().push(options.stream.timeout_ms);
            Ok(faux_assistant_message(
                "answer",
                FauxMessageOptions::default(),
            ))
        }),
    ]);
    let (harness, root) = open(&storage, &setup).await;
    setup.settings(|settings| {
        settings.stream = Some(ConversationStreamOptions {
            timeout_ms: Some(1234),
            ..ConversationStreamOptions::default()
        })
    });
    harness.resume().unwrap();
    let id = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap()
        .id;
    reached.wait().await;
    let task_id = run_task_id(&harness, &root).await;
    harness.close(&context()).await.unwrap();

    let (harness, root) = open(&storage, &setup).await;
    let stored = checkpoint(&harness, task_id).await.unwrap();
    assert_eq!(stored["phase"], json!("request"));
    assert_eq!(stored["attempt"], json!(1));
    assert_eq!(stored["thinkingLevel"], json!("off"));
    assert_eq!(stored["streamOptions"], json!({ "timeoutMs": 1234 }));
    // The resend uses the pinned request, not options changed after preparation.
    setup.settings(|settings| {
        settings.stream = Some(ConversationStreamOptions {
            timeout_ms: Some(999),
            ..ConversationStreamOptions::default()
        })
    });
    assert_eq!(
        live_state(&harness, &root)
            .await
            .unwrap()
            .generation
            .unwrap()
            .attempt,
        1
    );
    harness.resume().unwrap();
    let settled = harness
        .submission(id, &context())
        .await
        .unwrap()
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    assert_eq!(*sent.lock(), [vec!["user", "system"]]);
    assert_eq!(*timeouts.lock(), [Some(1234)]);
    assert_eq!(
        entry_kinds(&root).await,
        ["pi.user", "pi.system", "pi.assistant"]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn converts_a_committed_partial_into_an_aborted_entry_and_resends_the_same_messages() {
    let storage = storage();
    let slow = chat_setup_with(RegisterFauxProviderOptions {
        tokens_per_second: Some(20.0),
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    slow.faux.set_responses([answer(&"z".repeat(400))]);
    let (harness, root) = open(&storage, &slow).await;
    harness.resume().unwrap();
    let watch = harness
        .watch_doc(&*LIVE_DOC, root.id, &context())
        .await
        .unwrap()
        .unwrap();
    let watched: Arc<Mutex<Option<String>>> = Arc::default();
    let sink = watched.clone();
    watch
        .start(move |value, _, _| {
            let sink = sink.clone();
            Box::pin(async move {
                if let Some(text) = value.and_then(|value| {
                    value["generation"]["message"]["content"][0]["text"]
                        .as_str()
                        .map(str::to_string)
                }) {
                    *sink.lock() = Some(text);
                }
                Ok(())
            })
        })
        .unwrap();
    let id = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap()
        .id;
    wait_for(|| {
        let watched = watched.clone();
        async move { watched.lock().is_some() }
    })
    .await;
    harness.close(&context()).await.unwrap();

    let setup = chat_setup();
    let sent: Arc<Mutex<Vec<Vec<&'static str>>>> = Arc::default();
    let sent_by = sent.clone();
    setup
        .faux
        .set_responses([FauxResponseStep::factory(move |request, _, _, _| {
            sent_by.lock().push(roles(&request.messages));
            Ok(faux_assistant_message(
                "answer",
                FauxMessageOptions::default(),
            ))
        })]);
    let (harness, root) = open(&storage, &setup).await;
    let stored = live_state(&harness, &root).await.unwrap();
    let partial = text_of(Some(&Message::Assistant(
        stored.generation.unwrap().message.unwrap(),
    )))
    .unwrap();
    // Everything observers saw before the crash is durable.
    assert!(partial.starts_with(watched.lock().as_deref().unwrap()));
    harness.resume().unwrap();
    let settled = harness
        .submission(id, &context())
        .await
        .unwrap()
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    assert_eq!(*sent.lock(), [vec!["user"]]);
    let entries = all_entries(&root).await;
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>(),
        ["pi.user", "pi.assistant", "pi.assistant"]
    );
    let converted = entries[1].model.as_ref().unwrap()[0].clone();
    assert_eq!(to_json(&converted)["stopReason"], json!("aborted"));
    assert_eq!(text_of(Some(&converted)), Some(partial));
    assert_eq!(live_state(&harness, &root).await, Some(Default::default()));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn resumes_a_retry_backoff_after_reopen() {
    let storage = storage();
    let setup = chat_setup();
    let now = Arc::new(AtomicU64::new(1_000));
    let clock = now.clone();
    setup.set_now(move || clock.load(Ordering::SeqCst));
    setup.faux.set_responses([
        faux_assistant_message(
            Vec::new(),
            FauxMessageOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("503 Service Unavailable".into()),
                ..FauxMessageOptions::default()
            },
        )
        .into(),
        answer("recovered"),
    ]);
    let (harness, root) = open(&storage, &setup).await;
    harness.resume().unwrap();
    setup.settings(|settings| {
        settings.retry = Some(RetryPolicyOverrides {
            enabled: Some(true),
            max_retries: Some(2),
            base_delay_ms: Some(60_000),
            max_agent_delay_ms: None,
        })
    });
    let id = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap()
        .id;
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.generation)
            .is_some_and(|generation| generation.retry.is_some())
    })
    .await;
    let task_id = run_task_id(&harness, &root).await;
    harness.close(&context()).await.unwrap();

    let (harness, root) = open(&storage, &setup).await;
    assert_eq!(
        checkpoint(&harness, task_id).await,
        Some(json!({ "phase": "retry", "attempt": 1, "until": 61_000 }))
    );
    assert_eq!(
        to_json(&live_state(&harness, &root).await),
        json!({
            "run": { "taskId": task_id, "inputs": [id] },
            "generation": { "attempt": 1, "retry": { "at": 61_000, "error": "503 Service Unavailable" } },
        })
    );
    now.store(61_000, Ordering::SeqCst);
    harness.resume().unwrap();
    let settled = harness
        .submission(id, &context())
        .await
        .unwrap()
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    assert_eq!(
        entry_kinds(&root).await,
        ["pi.user", "pi.assistant", "pi.assistant"]
    );
    harness.close(&context()).await.unwrap();
}

fn deferred_setup(poll_after_ms: u64) -> ChatSetup {
    chat_setup_with(RegisterFauxProviderOptions {
        deferred: Some(FauxDeferredOptions {
            pending_fetches: None,
            poll_after_ms: Some(poll_after_ms),
        }),
        ..RegisterFauxProviderOptions::default()
    })
}

fn defer_requests(setup: &ChatSetup) {
    setup.settings(|settings| {
        settings.stream = Some(ConversationStreamOptions {
            deferred: Some(DeferredOption::Enabled(true)),
            ..ConversationStreamOptions::default()
        })
    });
}

#[tokio::test]
async fn resumes_polling_a_deferred_response_after_reopen() {
    let storage = storage();
    let setup = deferred_setup(60_000);
    let now = Arc::new(AtomicU64::new(1_000));
    let clock = now.clone();
    setup.set_now(move || clock.load(Ordering::SeqCst));
    setup.faux.set_responses([answer("deferred answer")]);
    let (harness, root) = open(&storage, &setup).await;
    harness.resume().unwrap();
    defer_requests(&setup);
    let id = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap()
        .id;
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.generation)
            .is_some_and(|generation| generation.deferred.is_some())
    })
    .await;
    let task_id = run_task_id(&harness, &root).await;
    harness.close(&context()).await.unwrap();

    let (harness, root) = open(&storage, &setup).await;
    let stored = checkpoint(&harness, task_id).await.unwrap();
    assert_eq!(stored["phase"], json!("poll"));
    assert_eq!(stored["attempt"], json!(1));
    assert_eq!(stored["pollAt"], json!(61_000));
    now.store(61_000, Ordering::SeqCst);
    harness.resume().unwrap();
    let settled = harness
        .submission(id, &context())
        .await
        .unwrap()
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
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
    assert_eq!(
        text_of(entry.model.as_ref().unwrap().first()).as_deref(),
        Some("deferred answer")
    );
    assert_eq!(setup.faux.state().deferred_fetch_count, 1);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn fails_no_model_when_the_pinned_model_is_gone_after_reopen_in_request_and_in_poll() {
    let storage = storage();
    let setup = deferred_setup(60_000);
    let (step, reached) = unanswered();
    setup.faux.set_responses([step, answer("deferred")]);
    let (harness, root) = open(&storage, &setup).await;
    harness.resume().unwrap();
    let requesting = root
        .submit(SubmissionDraft::input("one"), &context())
        .await
        .unwrap()
        .id;
    reached.wait().await;
    harness.close(&context()).await.unwrap();

    // Reopened without the faux provider: the request's pinned model is unknown.
    let empty = ChatOptions {
        models: Some(create_models(Default::default())),
        ..ChatOptions::default()
    };
    let (harness, _) = open_chat_with(storage.clone(), &setup, empty.clone()).await;
    harness.resume().unwrap();
    let settled = harness
        .submission(requesting, &context())
        .await
        .unwrap()
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Unanswered);
    assert_eq!(settled.reason.as_deref(), Some("no_model"));
    harness.close(&context()).await.unwrap();

    let (harness, root) = open(&storage, &setup).await;
    defer_requests(&setup);
    harness.resume().unwrap();
    let polling = root
        .submit(SubmissionDraft::input("two"), &context())
        .await
        .unwrap()
        .id;
    wait_for(|| async {
        live_state(&harness, &root)
            .await
            .and_then(|live| live.generation)
            .is_some_and(|generation| generation.deferred.is_some())
    })
    .await;
    harness.close(&context()).await.unwrap();

    let (harness, _) = open_chat_with(storage.clone(), &setup, empty).await;
    harness.resume().unwrap();
    let settled = harness
        .submission(polling, &context())
        .await
        .unwrap()
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    assert_eq!(settled.status, SubmissionStatus::Unanswered);
    assert_eq!(settled.reason.as_deref(), Some("no_model"));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn runs_a_print_style_turn_and_reads_the_durable_answer_after_reopen() {
    let storage = storage();
    let setup = chat_setup();
    add_text_section(&setup.registry, "preamble", "You are terse.", Some(false));
    setup.faux.set_responses([answer("42")]);
    let (harness, root) = open(&storage, &setup).await;
    harness.resume().unwrap();
    let submission = root
        .submit(SubmissionDraft::input("answer?"), &context())
        .await
        .unwrap();
    let settled = submission.wait(&context()).await.unwrap();
    assert_eq!(settled.status, SubmissionStatus::Done);
    let answer_id = settled.answer.unwrap();
    let answer = root
        .commit(
            move |tx| async move { tx.entry(answer_id).await },
            &context(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        text_of(answer.model.as_ref().unwrap().first()).as_deref(),
        Some("42")
    );
    harness.close(&context()).await.unwrap();

    let (harness, root) = open(&storage, &setup).await;
    let status = harness
        .submission(submission.id, &context())
        .await
        .unwrap()
        .unwrap()
        .status(&context())
        .await
        .unwrap();
    assert_eq!(status, settled);
    let again = root
        .commit(
            move |tx| async move { tx.entry(answer_id).await },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(again, Some(answer));
    harness.close(&context()).await.unwrap();
}
