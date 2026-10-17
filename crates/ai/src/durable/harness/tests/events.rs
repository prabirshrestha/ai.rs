//! Port of `test/harness-events.test.ts`.
//!
//! Divergences: events are compared in their TS JSON form. TS identity comparisons of tool-slot progress and of the
//! in-flight partial are value comparisons; a document's change is still the identity of its committed revision.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::chat::*;
use super::support::{aborted, add_tool, assert_err, context, signal_context};
use crate::chord::{AbortController, AbortReason};
use crate::durable::harness::agent::AGENT_DOC;
use crate::durable::harness::events::{AgentEventStream, EventBatch, watch_events};
use crate::durable::harness::live::{
    DeferredStatus, GenerationStatus, LIVE_DOC, LiveState, SlotStatus, ToolSlot,
};
use crate::durable::harness::types::{
    Retain, RetryPolicyOverrides, SubmissionDraft, ToolExecutionResult, ToolOutputLimits, WhenBusy,
};
use crate::durable::harness::usage::USAGE_DOC;
use crate::durable::harness::{Conversation, Harness, define_tool};
use crate::durable::session::tests::support::{Deferred, document_changes};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{EntryDraft, Storage, WatchEnd};
use crate::error::Error as AiError;
use crate::providers::faux::{
    FauxMessageOptions, FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions,
    faux_assistant_message, faux_text, faux_thinking, faux_tool_call,
};
use crate::types::{AssistantContent, AssistantMessage, StopReason};

struct Listening {
    stream: AgentEventStream,
    batches: Arc<Mutex<Vec<Vec<JsonValue>>>>,
}

impl Listening {
    fn events(&self) -> Vec<JsonValue> {
        self.batches.lock().iter().flatten().cloned().collect()
    }

    fn batches(&self) -> Vec<Vec<JsonValue>> {
        self.batches.lock().clone()
    }

    async fn stop(&self) {
        self.stream.stop().await;
    }
}

fn record_batches(stream: &AgentEventStream) -> Arc<Mutex<Vec<Vec<JsonValue>>>> {
    let batches: Arc<Mutex<Vec<Vec<JsonValue>>>> = Arc::default();
    let sink = batches.clone();
    stream
        .start(move |events: EventBatch, _| {
            sink.lock().push(
                events
                    .iter()
                    .map(|event| serde_json::to_value(event).unwrap())
                    .collect(),
            );
            Box::pin(async { Ok(()) })
        })
        .unwrap();
    batches
}

/// Attach and start an event stream that records every delivered batch.
async fn listen(harness: &Harness, conversation: &Conversation) -> Listening {
    let stream = watch_events(harness, conversation.id, &context())
        .await
        .unwrap();
    let batches = record_batches(&stream);
    Listening { stream, batches }
}

async fn drained() {
    for _ in 0..5 {
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(Duration::from_millis(1)).await;
    for _ in 0..5 {
        tokio::task::yield_now().await;
    }
}

fn paced(tokens_per_second: f64) -> ChatSetup {
    chat_setup_with(RegisterFauxProviderOptions {
        tokens_per_second: Some(tokens_per_second),
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    })
}

fn slow() -> ChatSetup {
    paced(400.0)
}

fn storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new())
}

fn answer(text: &str) -> FauxResponseStep {
    faux_assistant_message(text, FauxMessageOptions::default()).into()
}

fn tool_use(content: Vec<AssistantContent>) -> FauxResponseStep {
    faux_assistant_message(
        content,
        FauxMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxMessageOptions::default()
        },
    )
    .into()
}

/// A faux response held until `release`, or until the request is aborted.
fn held(release: Deferred, text: &'static str) -> FauxResponseStep {
    FauxResponseStep::async_factory(move |_, options, _, _| {
        let release = release.clone();
        async move {
            let signal = options.stream.signal.clone().unwrap();
            tokio::select! {
                _ = release.wait() => Ok(faux_assistant_message(text, FauxMessageOptions::default())),
                _ = signal.cancelled() => Err(AiError::Aborted("Request aborted".into())),
            }
        }
    })
}

fn types(events: &[JsonValue]) -> Vec<String> {
    events
        .iter()
        .map(|event| event["type"].as_str().unwrap().to_string())
        .collect()
}

fn of_type<'a>(events: &'a [JsonValue], kind: &str) -> Vec<&'a JsonValue> {
    events
        .iter()
        .filter(|event| event["type"] == kind)
        .collect()
}

/// Concatenated text blocks of a message (JSON form).
fn text_of_json(message: &JsonValue) -> String {
    message["content"]
        .as_array()
        .map(|content| {
            content
                .iter()
                .filter(|block| block["type"] == "text")
                .filter_map(|block| block["text"].as_str())
                .collect()
        })
        .unwrap_or_default()
}

/// Rebuild the streamed text of block 0 from `message_start` and the text deltas that follow it.
fn streamed_text(events: &[JsonValue]) -> String {
    let mut text = String::new();
    for event in events {
        if event["type"] == "message_start" && event["message"]["role"] == "assistant" {
            let block = &event["message"]["content"][0];
            text = if block["type"] == "text" {
                block["text"].as_str().unwrap().to_string()
            } else {
                String::new()
            };
        }
        if event["type"] != "message_update" {
            continue;
        }
        for change in event["changes"].as_array().unwrap() {
            if change["type"] == "text_start"
                && change["contentIndex"] == 0
                && change["block"]["type"] == "text"
            {
                text = change["block"]["text"].as_str().unwrap().to_string();
            }
            if change["type"] == "text_delta" && change["contentIndex"] == 0 {
                text.push_str(change["delta"].as_str().unwrap());
            }
        }
    }
    text
}

/// Apply message changes to a copy of `message`, as an events-only consumer would.
fn apply_changes(message: &JsonValue, changes: &JsonValue) -> JsonValue {
    let mut next = message.clone();
    for change in changes.as_array().unwrap() {
        let kind = change["type"].as_str().unwrap();
        let index = change["contentIndex"].as_u64().unwrap_or(0) as usize;
        match kind {
            "message" => next = change["message"].clone(),
            "block" => next["content"][index] = change["block"].clone(),
            "text_start" | "thinking_start" | "toolcall_start" => next["content"]
                .as_array_mut()
                .unwrap()
                .insert(index, change["block"].clone()),
            "text_delta" | "thinking_delta" => {
                let field = if kind == "text_delta" {
                    "text"
                } else {
                    "thinking"
                };
                let block = &mut next["content"][index][field];
                *block = json!(format!(
                    "{}{}",
                    block.as_str().unwrap(),
                    change["delta"].as_str().unwrap()
                ));
            }
            "toolcall_delta" => {
                let mut target = &mut next["content"][index]["arguments"];
                for segment in change["path"].as_array().unwrap() {
                    target = match segment {
                        JsonValue::String(key) => &mut target[key.as_str()],
                        JsonValue::Number(index) => &mut target[index.as_u64().unwrap() as usize],
                        _ => unreachable!(),
                    };
                }
                *target = json!(format!(
                    "{}{}",
                    target.as_str().unwrap(),
                    change["delta"].as_str().unwrap()
                ));
            }
            _ => unreachable!("{kind}"),
        }
    }
    next
}

/// Committed partials of the generation, one per commit that has one.
fn partials_of(harness: &Harness) -> Arc<Mutex<Vec<JsonValue>>> {
    let partials: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let sink = partials.clone();
    let unsubscribe = harness
        .subscribe_commits(move |publication, _| {
            for change in document_changes(publication) {
                if change.record.kind != "pi.live" {
                    continue;
                }
                if let Some(message) = change
                    .value
                    .as_deref()
                    .map(|value| &value["generation"]["message"])
                    .filter(|message| !message.is_null())
                {
                    sink.lock().push(message.clone());
                }
            }
        })
        .unwrap();
    std::mem::forget(unsubscribe);
    partials
}

/// The output events' rebuilt slot window after each output event.
fn output_windows(events: &[JsonValue]) -> Vec<String> {
    let mut windows = Vec::new();
    let mut text = String::new();
    for event in events {
        if event["type"] != "tool_execution_update" || event.get("output").is_none() {
            continue;
        }
        let output = &event["output"];
        if let Some(set) = output.get("set") {
            text = set.as_str().unwrap().to_string();
        } else {
            let trim = output["trimStart"].as_u64().unwrap_or(0) as usize;
            text = format!(
                "{}{}",
                &text[trim..],
                output["append"].as_str().unwrap_or("")
            );
        }
        windows.push(text.clone());
    }
    windows
}

fn empty_result() -> crate::durable::errors::Result<ToolExecutionResult> {
    Ok(ToolExecutionResult::default())
}

#[tokio::test]
async fn streams_a_run_as_lifecycle_events_and_text_deltas_that_rebuild_the_answer() {
    let setup = slow();
    let text = "streamed answer text ".repeat(20);
    setup.faux.set_responses(vec![answer(&text)]);
    let (harness, root) = open_chat(storage(), &setup).await;
    let listening = listen(&harness, &root).await;
    let snapshot = serde_json::to_value(&listening.stream.snapshot).unwrap();
    assert_eq!(snapshot["entries"], json!([]));
    assert_eq!(snapshot["tools"], json!([]));
    assert_eq!(snapshot["inbox"], json!([]));
    let partials: Arc<Mutex<Vec<String>>> = Arc::default();
    {
        let sink = partials.clone();
        let unsubscribe = harness
            .subscribe_commits(move |publication, _| {
                for change in document_changes(publication) {
                    if change.record.kind != "pi.live" {
                        continue;
                    }
                    if let Some(message) = change
                        .value
                        .as_deref()
                        .map(|value| &value["generation"]["message"])
                        .filter(|message| !message.is_null())
                    {
                        sink.lock().push(text_of_json(message));
                    }
                }
            })
            .unwrap();
        std::mem::forget(unsubscribe);
    }
    let submission = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap();
    submission.wait(&context()).await.unwrap();
    drained().await;
    let all = listening.events();
    let mut deduped = types(&all);
    deduped.dedup();
    assert_eq!(
        deduped,
        vec![
            "message_start",
            "message_end",
            "submission",
            "run_start",
            "turn_start",
            "message_start",
            "message_update",
            "message_end",
            "turn_end",
            "run_end",
            "submission",
            "usage_changed",
        ]
    );
    assert_eq!(
        *of_type(&all, "run_start")[0],
        json!({ "type": "run_start", "inputs": [submission.id] })
    );
    // After each event, the rebuilt text equals the committed partial of that commit.
    let rebuilt: Vec<String> = all
        .iter()
        .enumerate()
        .filter(|(_, event)| {
            (event["type"] == "message_start" && event["message"]["role"] == "assistant")
                || event["type"] == "message_update"
        })
        .map(|(index, _)| streamed_text(&all[..=index]))
        .collect();
    let partials = partials.lock().clone();
    assert_eq!(rebuilt, partials);
    assert!(partials.len() > 1);
    let end = of_type(&all, "message_end").pop().unwrap();
    assert_eq!(text_of_json(&end["entry"]["model"][0]), text);
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reports_tool_start_output_appends_and_the_result_entry() {
    let setup = chat_setup();
    let gate = Deferred::default();
    {
        let gate = gate.clone();
        add_tool(
            &setup.registry,
            define_tool(
                "print",
                "Prints",
                json!({ "type": "object", "properties": { "n": { "type": "number" } } }),
                move |_, api, _| {
                    let gate = gate.clone();
                    async move {
                        api.output("one\n")?;
                        tokio::time::sleep(Duration::from_millis(150)).await;
                        api.output("two\n")?;
                        gate.wait().await;
                        empty_result()
                    }
                },
            ),
        );
    }
    setup.faux.set_responses(vec![
        tool_use(vec![faux_tool_call("print", json!({ "n": 1 }), Some("c1"))]),
        answer("done"),
    ]);
    let (harness, root) = open_chat(storage(), &setup).await;
    let listening = listen(&harness, &root).await;
    let submission = root
        .submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap();
    // The output events rebuild the slot's retained window.
    let batches = listening.batches.clone();
    wait_for(|| {
        let events: Vec<JsonValue> = batches.lock().iter().flatten().cloned().collect();
        let done = output_windows(&events).last().map(String::as_str) == Some("one\ntwo\n");
        async move { done }
    })
    .await;
    gate.resolve();
    submission.wait(&context()).await.unwrap();
    drained().await;
    let all = listening.events();
    let tool: Vec<&JsonValue> = all
        .iter()
        .filter(|event| {
            event["type"]
                .as_str()
                .unwrap()
                .starts_with("tool_execution")
        })
        .collect();
    assert_eq!(
        *tool[0],
        json!({ "type": "tool_execution_start", "toolCallId": "c1", "toolName": "print", "args": { "n": 1 } })
    );
    let end = *tool.last().unwrap();
    assert_eq!(end["type"], "tool_execution_end");
    assert_eq!(end["toolCallId"], "c1");
    assert_eq!(end["entry"]["kind"], "pi.tool-result");
    // As in the coding agent, the tool ends directly before its result message.
    let end_index = all.iter().position(|event| event == end).unwrap();
    assert_eq!(
        types(&all[end_index..end_index + 3]),
        vec!["tool_execution_end", "message_start", "message_end"]
    );
    assert_eq!(all[end_index + 1]["message"]["role"], "toolResult");
    assert_eq!(all[end_index + 1]["message"]["toolCallId"], "c1");
    // Two turns: the tool round, and the answer.
    assert_eq!(of_type(&all, "turn_start").len(), 2);
    assert_eq!(of_type(&all, "turn_end").len(), 2);
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reports_queued_submissions_inbox_changes_and_retries() {
    let setup = chat_setup();
    let release = Deferred::default();
    let error = faux_assistant_message(
        Vec::<AssistantContent>::new(),
        FauxMessageOptions {
            stop_reason: Some(StopReason::Error),
            error_message: Some("503 Service Unavailable".into()),
            ..FauxMessageOptions::default()
        },
    );
    setup.faux.set_responses(vec![
        held(release.clone(), "first"),
        error.into(),
        answer("second"),
    ]);
    let (harness, root) = open_chat(storage(), &setup).await;
    setup.settings(|settings| {
        settings.retry = Some(RetryPolicyOverrides {
            enabled: Some(true),
            max_retries: Some(1),
            base_delay_ms: Some(1),
            max_agent_delay_ms: None,
        })
    });
    let listening = listen(&harness, &root).await;
    root.submit(SubmissionDraft::input("a"), &context())
        .await
        .unwrap();
    let follow_up = root
        .submit(SubmissionDraft::input("f"), &context())
        .await
        .unwrap();
    drained().await;
    let all = listening.events();
    assert_eq!(
        of_type(&all, "inbox_update"),
        vec![
            &json!({ "type": "inbox_update", "items": [{ "id": follow_up.id, "mode": "followUp" }] })
        ]
    );
    release.resolve();
    follow_up.wait(&context()).await.unwrap();
    drained().await;
    let all = listening.events();
    let kinds = types(&all);
    assert!(kinds.contains(&"auto_retry_start".to_string()));
    assert!(kinds.contains(&"auto_retry_end".to_string()));
    assert_eq!(of_type(&all, "run_start").len(), 2);
    let statuses: Vec<&JsonValue> = of_type(&all, "submission")
        .into_iter()
        .map(|event| &event["record"]["status"])
        .collect();
    assert_eq!(statuses, vec!["placed", "queued", "done", "placed", "done"]);
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}

async fn note(root: &Conversation) {
    let id = root.id;
    root.commit(
        move |tx| async move { tx.append_entry(id, EntryDraft::new("note")).await },
        &context(),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn replaces_undelivered_batches_with_one_snapshot_after_100_pending_batches() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let stream = watch_events(&harness, root.id, &context()).await.unwrap();
    for _ in 0..101 {
        note(&root).await;
    }
    let batches = record_batches(&stream);
    drained().await;
    let batches = batches.lock().clone();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].len(), 1);
    assert_eq!(batches[0][0]["type"], "snapshot");
    assert_eq!(batches[0][0]["entries"].as_array().unwrap().len(), 101);
    stream.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rebuilds_every_committed_partial_of_thinking_text_and_tool_call_arguments_from_message_changes()
 {
    let setup = paced(150.0);
    setup.faux.set_responses(vec![
        tool_use(vec![
            faux_thinking("thinking about it ".repeat(10)),
            faux_text("some text ".repeat(10)),
            faux_tool_call(
                "missing",
                json!({ "path": "a/long/path/".repeat(10), "note": "x".repeat(60) }),
                Some("c1"),
            ),
        ]),
        answer("done"),
    ]);
    let (harness, root) = open_chat(storage(), &setup).await;
    let partials = partials_of(&harness);
    let listening = listen(&harness, &root).await;
    root.submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    drained().await;
    let mut rebuilt: Vec<JsonValue> = Vec::new();
    let mut current: Option<JsonValue> = None;
    // The streamed tool-calling message, up to its end; the short final answer commits no partial.
    let all = listening.events();
    for event in &all {
        if event["type"] == "message_end" && event["entry"]["kind"] == "pi.assistant" {
            break;
        }
        if event["type"] == "message_start" && event["message"]["role"] == "assistant" {
            current = Some(event["message"].clone());
        } else if event["type"] == "message_update" {
            current = Some(apply_changes(current.as_ref().unwrap(), &event["changes"]));
        } else {
            continue;
        }
        rebuilt.push(current.clone().unwrap());
    }
    let partials = partials.lock().clone();
    assert!(partials.len() > 2);
    let contents = |messages: &[JsonValue]| -> Vec<JsonValue> {
        messages
            .iter()
            .map(|message| message["content"].clone())
            .collect()
    };
    assert_eq!(contents(&rebuilt), contents(&partials));
    let change_types: Vec<JsonValue> = of_type(&all, "message_update")
        .into_iter()
        .flat_map(|event| event["changes"].as_array().unwrap().clone())
        .map(|change| change["type"].clone())
        .collect();
    assert!(
        change_types.contains(&json!("thinking_delta"))
            || change_types.contains(&json!("text_delta"))
    );
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rebuilds_a_sliding_tail_window_from_output_trims_and_appends() {
    let setup = chat_setup();
    let gate = Deferred::default();
    {
        let gate = gate.clone();
        let mut tail = define_tool(
            "tail",
            "Prints lines",
            json!({ "type": "object", "properties": {} }),
            move |_, api, _| {
                let gate = gate.clone();
                async move {
                    for line in 0..6 {
                        api.output(&format!("line {line}\n"))?;
                        tokio::time::sleep(Duration::from_millis(120)).await;
                    }
                    gate.wait().await;
                    empty_result()
                }
            },
        );
        tail.output_limits = Some(ToolOutputLimits {
            max_bytes: None,
            max_lines: Some(3),
            retain: Some(Retain::Tail),
        });
        add_tool(&setup.registry, tail);
    }
    setup.faux.set_responses(vec![
        tool_use(vec![faux_tool_call("tail", json!({}), Some("c1"))]),
        answer("done"),
    ]);
    let (harness, root) = open_chat(storage(), &setup).await;
    let outputs: Arc<Mutex<Vec<String>>> = Arc::default();
    {
        let sink = outputs.clone();
        let unsubscribe = harness
            .subscribe_commits(move |publication, _| {
                for change in document_changes(publication) {
                    if change.record.kind != "pi.live" {
                        continue;
                    }
                    let Some(output) = change
                        .value
                        .as_deref()
                        .and_then(|value| value["tools"][0]["output"].as_str())
                    else {
                        continue;
                    };
                    let mut outputs = sink.lock();
                    if outputs.last().map(String::as_str) != Some(output) {
                        outputs.push(output.to_string());
                    }
                }
            })
            .unwrap();
        std::mem::forget(unsubscribe);
    }
    let listening = listen(&harness, &root).await;
    let submission = root
        .submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap();
    {
        let outputs = outputs.clone();
        wait_for(move || {
            let done = outputs
                .lock()
                .last()
                .is_some_and(|output| output.ends_with("line 5\n"));
            async move { done }
        })
        .await;
    }
    gate.resolve();
    submission.wait(&context()).await.unwrap();
    drained().await;
    let all = listening.events();
    assert_eq!(output_windows(&all), *outputs.lock());
    assert!(all.iter().any(|event| {
        event["type"] == "tool_execution_update" && event["output"].get("trimStart").is_some()
    }));
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn emits_one_exact_batch_when_a_run_ends_and_a_queued_follow_up_starts_the_next() {
    let setup = chat_setup();
    let release = Deferred::default();
    setup
        .faux
        .set_responses(vec![held(release.clone(), "first"), answer("second")]);
    let (harness, root) = open_chat(storage(), &setup).await;
    let input = root
        .submit(SubmissionDraft::input("a"), &context())
        .await
        .unwrap();
    let follow_up = root
        .submit(SubmissionDraft::input("f"), &context())
        .await
        .unwrap();
    let listening = listen(&harness, &root).await;
    release.resolve();
    follow_up.wait(&context()).await.unwrap();
    drained().await;
    let batches = listening.batches();
    let boundary = batches
        .iter()
        .find(|batch| batch.iter().any(|event| event["type"] == "run_end"))
        .unwrap();
    assert_eq!(
        types(boundary),
        vec![
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "run_end",
            "submission",
            "submission",
            "inbox_update",
            "usage_changed",
            "run_start",
            "turn_start",
        ]
    );
    let ids: Vec<&JsonValue> = of_type(boundary, "submission")
        .into_iter()
        .map(|event| &event["record"]["id"])
        .collect();
    assert_eq!(ids, vec![&json!(input.id), &json!(follow_up.id)]);
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}

async fn live(harness: &Harness, root: &Conversation) -> LiveState {
    harness
        .snapshot(&*LIVE_DOC, root.id, &context())
        .await
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn ends_a_call_that_never_runs_and_a_tool_aborted_with_its_generation() {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        define_tool(
            "wait",
            "Waits until aborted",
            json!({ "type": "object", "properties": {} }),
            |_, _, ctx| async move {
                aborted(ctx.abort_signal().unwrap().clone()).await?;
                empty_result()
            },
        ),
    );
    setup.faux.set_responses(vec![tool_use(vec![
        faux_tool_call("ghost", json!({}), Some("c1")),
        faux_tool_call("wait", json!({}), Some("c2")),
    ])]);
    let (harness, root) = open_chat(storage(), &setup).await;
    let listening = listen(&harness, &root).await;
    let submission = root
        .submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap();
    {
        let batches = listening.batches.clone();
        wait_for(move || {
            let started = batches
                .lock()
                .iter()
                .flatten()
                .any(|event| event["type"] == "tool_execution_start");
            async move { started }
        })
        .await;
    }
    // Aborting the generation aborts its round: the tool ends with its aborted result first (spec §8.5).
    let task_id = live(&harness, &root).await.run.unwrap().task_id;
    harness.abort_task(task_id, &context()).await.unwrap();
    submission.wait(&context()).await.unwrap();
    drained().await;
    // The call not offered ends after its calling message and directly before its result message.
    let all = listening.events();
    let round: Vec<String> = all
        .iter()
        .map(|event| match event["type"].as_str().unwrap() {
            "message_end" => format!("end:{}", event["entry"]["kind"].as_str().unwrap()),
            "message_start" => format!("start:{}", event["message"]["role"].as_str().unwrap()),
            other => other.to_string(),
        })
        .collect();
    let ghost_end = round
        .iter()
        .position(|kind| kind == "tool_execution_end")
        .unwrap();
    assert_eq!(
        round[ghost_end - 1..ghost_end + 3],
        [
            "end:pi.assistant",
            "tool_execution_end",
            "start:toolResult",
            "end:pi.tool-result"
        ]
    );
    let tool: Vec<(String, JsonValue, bool)> = all
        .iter()
        .filter(|event| {
            event["type"]
                .as_str()
                .unwrap()
                .starts_with("tool_execution")
        })
        .map(|event| {
            (
                event["type"].as_str().unwrap().to_string(),
                event["toolCallId"].clone(),
                event.get("entry").is_some(),
            )
        })
        .collect();
    assert_eq!(
        tool,
        vec![
            ("tool_execution_end".into(), json!("c1"), true),
            ("tool_execution_start".into(), json!("c2"), false),
            ("tool_execution_end".into(), json!("c2"), true),
        ]
    );
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reports_a_steer_without_run_events_a_reset_as_an_appended_entry_and_nothing_for_other_conversations()
 {
    let setup = chat_setup();
    let gate = Deferred::default();
    {
        let gate = gate.clone();
        add_tool(
            &setup.registry,
            define_tool(
                "hold",
                "Waits",
                json!({ "type": "object", "properties": {} }),
                move |_, _, _| {
                    let gate = gate.clone();
                    async move {
                        gate.wait().await;
                        empty_result()
                    }
                },
            ),
        );
    }
    setup.faux.set_responses(vec![
        tool_use(vec![faux_tool_call("hold", json!({}), Some("c1"))]),
        answer("done"),
    ]);
    let (harness, root) = open_chat(storage(), &setup).await;
    let other = harness
        .create_conversation(
            crate::durable::harness::types::ConversationCreateOptions::ownerless(),
            &context(),
        )
        .await
        .unwrap();
    let listening = listen(&harness, &root).await;
    let input = root
        .submit(SubmissionDraft::input("a"), &context())
        .await
        .unwrap();
    {
        let batches = listening.batches.clone();
        wait_for(move || {
            let started = batches
                .lock()
                .iter()
                .flatten()
                .any(|event| event["type"] == "tool_execution_start");
            async move { started }
        })
        .await;
    }
    root.submit(
        SubmissionDraft::input("s").when_busy(WhenBusy::Steer),
        &context(),
    )
    .await
    .unwrap();
    drained().await;
    let before = listening.batches().len();
    note(&other).await;
    drained().await;
    assert_eq!(listening.batches().len(), before);
    gate.resolve();
    input.wait(&context()).await.unwrap();
    root.reset(None, &context()).await.unwrap();
    drained().await;
    let all = listening.events();
    assert_eq!(of_type(&all, "run_start").len(), 1);
    assert_eq!(of_type(&all, "run_end").len(), 1);
    assert_eq!(
        types(listening.batches().last().unwrap()),
        vec!["entry_appended", "submission"]
    );
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn rejects_an_attachment_cancelled_while_it_waits_for_the_session_line() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let release = Deferred::default();
    let blocking = {
        let (root, release) = (root.clone(), release.clone());
        tokio::spawn(async move {
            root.commit(
                move |_| async move {
                    release.wait().await;
                    Ok(())
                },
                &context(),
            )
            .await
        })
    };
    tokio::task::yield_now().await;
    let controller = AbortController::new();
    let attaching = {
        let (harness, id) = (harness.clone(), root.id);
        let signalled = signal_context(controller.signal());
        tokio::spawn(async move { watch_events(&harness, id, &signalled).await })
    };
    tokio::task::yield_now().await;
    controller.abort(Some(AbortReason::message("cancelled")));
    release.resolve();
    blocking.await.unwrap().unwrap();
    assert_err(attaching.await.unwrap(), "cancelled");
    harness.close(&context()).await.unwrap();
}

fn partial_message(text: &str) -> AssistantMessage {
    faux_assistant_message(text, FauxMessageOptions::default())
}

#[tokio::test]
async fn applies_deltas_after_an_overflow_snapshot_that_holds_an_in_flight_partial() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let partial = partial_message("hel");
    let stream = watch_events(&harness, root.id, &context()).await.unwrap();
    let id = root.id;
    root.commit(
        move |tx| async move {
            tx.doc(&*LIVE_DOC, id).await?.edit(|live| {
                live.generation = Some(GenerationStatus {
                    attempt: 1,
                    message: Some(partial),
                    ..GenerationStatus::default()
                })
            })
        },
        &context(),
    )
    .await
    .unwrap();
    for _ in 0..101 {
        note(&root).await;
    }
    let batches = record_batches(&stream);
    drained().await;
    let snapshot = batches.lock()[0][0].clone();
    assert_eq!(snapshot["type"], "snapshot");
    root.commit(
        move |tx| async move {
            tx.doc(&*LIVE_DOC, id).await?.edit(|live| {
                let message = live.generation.as_mut().unwrap().message.as_mut().unwrap();
                if let AssistantContent::Text(block) = &mut message.content[0] {
                    block.text.push_str("lo");
                }
            })
        },
        &context(),
    )
    .await
    .unwrap();
    drained().await;
    let update = batches.lock().last().unwrap()[0].clone();
    assert_eq!(update["type"], "message_update");
    assert_eq!(
        update["changes"],
        json!([{ "type": "text_delta", "contentIndex": 0, "delta": "lo" }])
    );
    let rebuilt = apply_changes(&snapshot["generation"]["message"], &update["changes"]);
    assert_eq!(
        rebuilt["content"],
        json!([{ "type": "text", "text": "hello" }])
    );
    stream.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reports_usage_only_updates_cleared_tool_progress_tools_ending_without_entries_and_deferred_polls()
 {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let listening = listen(&harness, &root).await;
    let id = root.id;
    let change = |edit: Box<dyn FnOnce(&mut LiveState) + Send>| {
        let root = root.clone();
        async move {
            root.commit(
                move |tx| async move { tx.doc(&*LIVE_DOC, id).await?.edit(edit) },
                &context(),
            )
            .await
            .unwrap()
        }
    };
    let partial = partial_message("partial");
    let partial_json = serde_json::to_value(&partial).unwrap();
    {
        let partial = partial.clone();
        change(Box::new(move |live| {
            live.generation = Some(GenerationStatus {
                attempt: 1,
                message: Some(partial),
                ..GenerationStatus::default()
            })
        }))
        .await;
    }
    // A usage-only change sends the usage and no changes.
    change(Box::new(|live| {
        live.generation
            .as_mut()
            .unwrap()
            .message
            .as_mut()
            .unwrap()
            .usage
            .input = 42
    }))
    .await;
    change(Box::new(|live| {
        live.generation = Some(GenerationStatus {
            attempt: 1,
            deferred: Some(DeferredStatus { poll_at: 1 }),
            ..GenerationStatus::default()
        })
    }))
    .await;
    change(Box::new(|live| {
        live.generation
            .as_mut()
            .unwrap()
            .deferred
            .as_mut()
            .unwrap()
            .poll_at = 2
    }))
    .await;
    change(Box::new(|live| {
        live.tools = Some(vec![
            serde_json::from_value::<ToolSlot>(json!({
                "callId": "c1", "name": "t", "status": "running", "details": { "n": 1 }, "diagnostics": []
            }))
            .unwrap(),
        ])
    }))
    .await;
    // A safe replay clears the running slot's progress.
    change(Box::new(|live| {
        let slot = &mut live.tools.as_mut().unwrap()[0];
        slot.details = None;
        slot.diagnostics = None;
    }))
    .await;
    // A fault marks the slot done without an entry.
    change(Box::new(|live| {
        live.tools.as_mut().unwrap()[0].status = SlotStatus::Done
    }))
    .await;
    root.commit(
        move |tx| async move { tx.retire_doc(&*USAGE_DOC, id).await },
        &context(),
    )
    .await
    .unwrap();
    root.commit(
        move |tx| async move { tx.retire_doc(&*AGENT_DOC, id).await },
        &context(),
    )
    .await
    .unwrap();
    drained().await;
    let mut usage = partial_json["usage"].clone();
    usage["input"] = json!(42);
    let batches: Vec<Vec<JsonValue>> = listening
        .batches()
        .into_iter()
        .map(|batch| {
            batch
                .into_iter()
                .filter(|event| event["type"] != "task_failed")
                .collect()
        })
        .collect();
    assert_eq!(
        batches,
        vec![
            vec![json!({ "type": "message_start", "message": partial_json })],
            vec![json!({ "type": "message_update", "usage": usage, "changes": [] })],
            vec![json!({ "type": "deferred_poll", "pollAt": 1 })],
            vec![json!({ "type": "deferred_poll", "pollAt": 2 })],
            vec![
                json!({ "type": "tool_execution_start", "toolCallId": "c1", "toolName": "t", "args": {} })
            ],
            vec![
                json!({ "type": "tool_execution_update", "toolCallId": "c1", "toolName": "t", "details": null, "diagnostics": [] })
            ],
            vec![json!({ "type": "tool_execution_end", "toolCallId": "c1", "toolName": "t" })],
            // Retired documents read as their initial values.
            vec![json!({ "type": "usage_changed", "usage": { "models": {}, "tools": {} } })],
            vec![json!({ "type": "agent_changed", "agent": {} })],
        ]
    );
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn ends_the_stream_with_the_harness() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let stream = watch_events(&harness, root.id, &context()).await.unwrap();
    harness.close(&context()).await.unwrap();
    assert!(matches!(stream.closed().await, WatchEnd::SessionClosed));
}

#[tokio::test]
async fn starts_a_message_at_the_first_committed_partial_and_ends_it_with_the_converted_entry_on_abort()
 {
    let setup = paced(100.0);
    setup.faux.set_responses(vec![answer(&"x".repeat(400))]);
    let (harness, root) = open_chat(storage(), &setup).await;
    let listening = listen(&harness, &root).await;
    let submission = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap();
    {
        let batches = listening.batches.clone();
        wait_for(move || {
            let started = batches.lock().iter().flatten().any(|event| {
                event["type"] == "message_start" && event["message"]["role"] == "assistant"
            });
            async move { started }
        })
        .await;
    }
    let task_id = live(&harness, &root).await.run.unwrap().task_id;
    harness.abort_task(task_id, &context()).await.unwrap();
    submission.wait(&context()).await.unwrap();
    drained().await;
    let all = listening.events();
    let assistant_ends = all
        .iter()
        .filter(|event| event["type"] == "message_end" && event["entry"]["kind"] == "pi.assistant")
        .count();
    assert_eq!(assistant_ends, 1);
    let assistant_starts = all
        .iter()
        .filter(|event| event["type"] == "message_start" && event["message"]["role"] == "assistant")
        .count();
    assert_eq!(assistant_starts, 1);
    listening.stop().await;
    harness.close(&context()).await.unwrap();
}
