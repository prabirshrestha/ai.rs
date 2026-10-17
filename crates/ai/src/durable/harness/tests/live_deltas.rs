//! Port of `test/harness-live-deltas.test.ts`.
//!
//! Divergences: operations are compared in their TS JSON form. The base/delta recording storage is
//! `ControlledStorage`'s commit log paired in order with the published `pi.live` updates. The faulting tool result
//! carries a non-finite usage cost instead of a function value.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::chat::*;
use super::support::{add_tool, context, to_json};
use crate::durable::harness::live::{LIVE_DOC, LiveState};
use crate::durable::harness::tool::ToolExecutionApi;
use crate::durable::harness::types::{
    DiagnosticSeverity, Retain, SubmissionDraft, ToolDiagnostic, ToolExecutionMode,
    ToolExecutionResult, ToolOutputLimits, ToolRegistration,
};
use crate::durable::harness::{Conversation, Harness, define_tool};
use crate::durable::session::tests::support::ControlledStorage;
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{
    CheckpointInfo, CommitChange, CommitPublication, DocumentContent, Storage, StorageWrite,
    TaskQuery,
};
use crate::providers::faux::{
    FauxMessageOptions, FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions,
    faux_assistant_message, faux_tool_call,
};
use crate::types::StopReason;

type Ops = Vec<JsonValue>;

fn tool_calls(calls: &[(&str, &str)]) -> FauxResponseStep {
    faux_assistant_message(
        calls
            .iter()
            .map(|(name, id)| faux_tool_call(*name, json!({}), Some(id)))
            .collect::<Vec<_>>(),
        FauxMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            ..FauxMessageOptions::default()
        },
    )
    .into()
}

fn done() -> FauxResponseStep {
    faux_assistant_message("done", FauxMessageOptions::default()).into()
}

fn simple_tool<F, Fut>(name: &str, execute: F) -> ToolRegistration
where
    F: Fn(JsonValue, ToolExecutionApi, crate::chord::Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = crate::durable::errors::Result<ToolExecutionResult>> + Send + 'static,
{
    define_tool(
        name,
        name,
        json!({ "type": "object", "properties": {} }),
        execute,
    )
}

fn empty() -> crate::durable::errors::Result<ToolExecutionResult> {
    Ok(ToolExecutionResult {
        content: Some(Vec::new()),
        ..ToolExecutionResult::default()
    })
}

/// The operations of every `pi.live` commit with operations, in order.
fn record_live_ops(harness: &Harness) -> Arc<Mutex<Vec<Ops>>> {
    let commits: Arc<Mutex<Vec<Ops>>> = Arc::default();
    let sink = commits.clone();
    let unsubscribe = harness
        .subscribe_commits(move |publication, _| {
            for change in &publication.changes {
                if let CommitChange::Document(change) = change
                    && change.record.kind == "pi.live"
                    && !change.ops.is_empty()
                {
                    sink.lock().push(change.ops.iter().map(to_json).collect());
                }
            }
        })
        .unwrap();
    std::mem::forget(unsubscribe);
    commits
}

async fn submit_and_wait(root: &Conversation) {
    root.submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
}

/// `expect.arrayContaining(expected)`.
fn contains_all(ops: &[JsonValue], expected: &[JsonValue]) -> bool {
    expected.iter().all(|op| ops.contains(op))
}

async fn live(harness: &Harness, root: &Conversation) -> Option<LiveState> {
    harness
        .snapshot(&*LIVE_DOC, root.id, &context())
        .await
        .unwrap()
}

fn output() -> JsonValue {
    json!(["tools", 0, "output"])
}

fn is_output(op: &JsonValue) -> bool {
    op[1] == output()
}

type Action = Box<dyn FnOnce(ToolExecutionApi) -> BoxFuture<'static, bool> + Send>;

/// One tool call driven step by step, with the exact operations of every `pi.live` commit.
struct Drive {
    harness: Harness,
    root: Conversation,
    commits: Arc<Mutex<Vec<Ops>>>,
    actions: tokio::sync::mpsc::UnboundedSender<Action>,
    submission: crate::durable::harness::submissions::Submission,
}

impl Drive {
    async fn open(limits: ToolOutputLimits) -> Self {
        let setup = chat_setup();
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel::<Action>();
        let receiver = Arc::new(tokio::sync::Mutex::new(receiver));
        let mut drive = simple_tool("drive", move |_, api, ctx| {
            let receiver = receiver.clone();
            async move {
                let mut receiver = receiver.lock().await;
                let signal = ctx.abort_signal().cloned();
                loop {
                    let next = match &signal {
                        Some(signal) => tokio::select! {
                            next = receiver.recv() => next,
                            _ = signal.cancelled() => {
                                return Err(crate::durable::errors::Error::message("aborted"));
                            }
                        },
                        None => receiver.recv().await,
                    };
                    let Some(action) = next else {
                        return Ok(ToolExecutionResult::default());
                    };
                    if !action(api.clone()).await {
                        return Ok(ToolExecutionResult::default());
                    }
                }
            }
        });
        drive.output_limits = Some(limits);
        add_tool(&setup.registry, drive);
        setup
            .faux
            .set_responses([tool_calls(&[("drive", "c1")]), done()]);
        let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
        let commits = record_live_ops(&harness);
        let submission = root
            .submit(SubmissionDraft::input("go"), &context())
            .await
            .unwrap();
        wait_for(|| {
            let (harness, root) = (harness.clone(), root.clone());
            async move {
                live(&harness, &root)
                    .await
                    .and_then(|live| live.tools)
                    .is_some_and(|tools| to_json(&tools[0].status) == json!("running"))
            }
        })
        .await;
        Self {
            harness,
            root,
            commits,
            actions: sender,
            submission,
        }
    }

    /// Run one action inside the tool and return the operations of the commit it caused.
    async fn step<F, Fut>(&self, action: F) -> Ops
    where
        F: FnOnce(ToolExecutionApi) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let before = self.commits.lock().len();
        self.actions
            .send(Box::new(move |api| {
                Box::pin(async move {
                    action(api).await;
                    true
                })
            }))
            .ok()
            .unwrap();
        let commits = self.commits.clone();
        wait_for(move || {
            let commits = commits.clone();
            async move { commits.lock().len() > before }
        })
        .await;
        self.commits.lock().last().unwrap().clone()
    }

    /// Let the tool return and the run finish; returns the commits made meanwhile.
    async fn finish(&self) -> Vec<Ops> {
        let before = self.commits.lock().len();
        self.actions
            .send(Box::new(|_| Box::pin(async { false })))
            .ok()
            .unwrap();
        self.submission.wait(&context()).await.unwrap();
        self.commits.lock()[before..].to_vec()
    }
}

fn limits(
    max_bytes: Option<usize>,
    max_lines: Option<usize>,
    retain: Option<Retain>,
) -> ToolOutputLimits {
    ToolOutputLimits {
        max_bytes,
        max_lines,
        retain,
    }
}

#[tokio::test]
async fn hands_a_generation_over_to_its_tool_round_and_starts_a_tool_with_one_field_write_each() {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        simple_tool("noop", |_, _, _| async { empty() }),
    );
    setup
        .faux
        .set_responses([tool_calls(&[("noop", "c1")]), done()]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let commits = record_live_ops(&harness);
    submit_and_wait(&root).await;
    let id = root.id;
    let tasks = harness
        .commit(
            move |tx| async move {
                tx.scan_tasks(
                    TaskQuery {
                        conversation_id: Some(id),
                        ..TaskQuery::default()
                    },
                    20,
                    None,
                )
                .await
            },
            &context(),
        )
        .await
        .unwrap()
        .items;
    let ids = |kind: &str| {
        let mut ids: Vec<_> = tasks
            .iter()
            .filter(|task| task.kind == kind)
            .map(|task| task.id)
            .collect();
        ids.sort();
        ids
    };
    let (generations, tools) = (ids("pi.generation"), ids("pi.tool"));
    let entries = all_entries(&root).await;
    let result = entries
        .iter()
        .find(|entry| entry.kind == "pi.tool-result")
        .unwrap()
        .id;
    let commits = commits.lock().clone();
    assert_eq!(commits.len(), 8, "{commits:#?}");
    // submission
    assert_eq!(commits[0].len(), 1);
    assert_eq!(commits[0][0][0], json!("s"));
    assert_eq!(commits[0][0][1], json!(["run"]));
    assert_eq!(commits[0][0][2]["taskId"], json!(generations[0]));
    assert!(commits[0][0][2]["inputs"][0].is_number());
    // request
    assert_eq!(commits[1], [json!(["s", ["generation"], { "attempt": 1 }])]);
    // the generation starts its tool round and keeps the run
    assert_eq!(commits[2].len(), 2);
    assert!(contains_all(
        &commits[2],
        &[
            json!(["d", ["generation"]]),
            json!(["s", ["tools"], [{ "callId": "c1", "name": "noop", "taskId": tools[0], "status": "pending" }]]),
        ]
    ));
    // intent
    assert_eq!(
        commits[3],
        [json!(["s", ["tools", 0, "status"], "running"])]
    );
    // result
    assert_eq!(commits[4].len(), 2);
    assert!(contains_all(
        &commits[4],
        &[
            json!(["s", ["tools", 0, "status"], "done"]),
            json!(["s", ["tools", 0, "entry"], result]),
        ]
    ));
    // the generation's tools phase hands the run to the next generation
    assert_eq!(commits[5].len(), 2);
    assert!(contains_all(
        &commits[5],
        &[
            json!(["d", ["tools"]]),
            json!(["s", ["run", "taskId"], generations[1]]),
        ]
    ));
    assert_eq!(commits[6], [json!(["s", ["generation"], { "attempt": 1 }])]);
    // the answer ends the run
    assert!(contains_all(
        &commits[7],
        &[json!(["d", ["run"]]), json!(["d", ["generation"]])]
    ));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn appends_head_output_and_then_only_updates_the_dropped_counts_once_the_window_is_full() {
    let run = Drive::open(limits(None, Some(2), None)).await;
    assert_eq!(
        run.step(|api| async move { api.output("one\n").unwrap() })
            .await,
        [json!(["s", output(), "one\n"])]
    );
    assert_eq!(
        run.step(|api| async move { api.output("two\n").unwrap() })
            .await,
        [json!(["a", output(), "two\n"])]
    );
    // The window is full: the retained text stays; only the counts change.
    assert_eq!(
        run.step(|api| async move { api.output("three\n").unwrap() })
            .await,
        [
            json!(["s", ["tools", 0, "droppedBytes"], 6]),
            json!(["s", ["tools", 0, "droppedLines"], 1]),
        ]
    );
    assert_eq!(
        run.step(|api| async move { api.output("four\n").unwrap() })
            .await,
        [
            json!(["s", ["tools", 0, "droppedBytes"], 11]),
            json!(["s", ["tools", 0, "droppedLines"], 2]),
        ]
    );
    run.finish().await;
    run.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn slides_a_tail_window_as_a_front_trim_plus_an_append() {
    let run = Drive::open(limits(None, Some(3), Some(Retain::Tail))).await;
    assert_eq!(
        run.step(|api| async move { api.output("line 1\nline 2\nline 3\n").unwrap() })
            .await,
        [json!(["s", output(), "line 1\nline 2\nline 3\n"])]
    );
    assert_eq!(
        run.step(|api| async move { api.output("line 4\n").unwrap() })
            .await,
        [
            json!(["t", output(), 7]),
            json!(["a", output(), "line 4\n"]),
            json!(["s", ["tools", 0, "droppedBytes"], 7]),
            json!(["s", ["tools", 0, "droppedLines"], 1]),
        ]
    );
    // The buffer keeps only the window, and later slides stay minimal and exact.
    assert_eq!(
        run.step(|api| async move { api.output("line 5\nline 6\n").unwrap() })
            .await,
        [
            json!(["t", output(), 14]),
            json!(["a", output(), "line 5\nline 6\n"]),
            json!(["s", ["tools", 0, "droppedBytes"], 21]),
            json!(["s", ["tools", 0, "droppedLines"], 3]),
        ]
    );
    assert_eq!(
        live(&run.harness, &run.root).await.unwrap().tools.unwrap()[0]
            .output
            .as_deref(),
        Some("line 4\nline 5\nline 6\n")
    );
    run.finish().await;
    run.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn writes_the_whole_window_when_chords_overlap_search_cannot_find_the_shared_part() {
    // A retained window beyond the 64 KiB overlap scan.
    let wide = Drive::open(limits(
        Some(100 * 1024),
        Some(1_000_000),
        Some(Retain::Tail),
    ))
    .await;
    let line = |index: usize| format!("{index:010} {}\n", "x".repeat(989));
    let text: String = (0..100).map(line).collect();
    wide.step(move |api| async move { api.output(&text).unwrap() })
        .await;
    let more: String = (100..104).map(line).collect();
    let slid = wide
        .step(move |api| async move { api.output(&more).unwrap() })
        .await;
    assert_eq!(
        slid.iter()
            .filter(|op| is_output(op))
            .map(|op| op[0].clone())
            .collect::<Vec<_>>(),
        [json!("s")]
    );
    wide.finish().await;
    wide.harness.close(&context()).await.unwrap();

    // Repetitive output still finds an overlap here; Chord's bounded candidate search can give up on other inputs
    // and then writes one window.
    let repetitive = Drive::open(limits(None, Some(50), Some(Retain::Tail))).await;
    repetitive
        .step(|api| async move { api.output(&"y\n".repeat(50)).unwrap() })
        .await;
    let repeated = repetitive
        .step(|api| async move { api.output("z\n").unwrap() })
        .await;
    assert_eq!(
        repeated.into_iter().filter(is_output).collect::<Vec<_>>(),
        [json!(["t", output(), 2]), json!(["a", output(), "z\n"])]
    );
    repetitive.finish().await;
    repetitive.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn diffs_details_leaf_by_leaf_and_appends_diagnostics() {
    let run = Drive::open(ToolOutputLimits::default()).await;
    let details = json!(["tools", 0, "details"]);
    assert_eq!(
        run.step(|api| async move {
            api.details(json!({ "step": 1, "log": "a" }), &context())
                .await
                .unwrap()
        })
        .await,
        [json!(["s", details, { "step": 1, "log": "a" }])]
    );
    let changed = run
        .step(|api| async move {
            api.details(json!({ "step": 2, "log": "ab" }), &context())
                .await
                .unwrap()
        })
        .await;
    assert!(contains_all(
        &changed,
        &[
            json!(["s", ["tools", 0, "details", "step"], 2]),
            json!(["a", ["tools", 0, "details", "log"], "b"]),
        ]
    ));
    assert_eq!(
        run.step(|api| async move { api.details(json!({ "step": 2 }), &context()).await.unwrap() })
            .await,
        [json!(["d", ["tools", 0, "details", "log"]])]
    );
    let diagnostic = |severity: DiagnosticSeverity, message: &str| ToolDiagnostic {
        severity,
        message: message.into(),
        code: None,
    };
    let first = diagnostic(DiagnosticSeverity::Info, "first");
    let second = diagnostic(DiagnosticSeverity::Warn, "second");
    let diagnostics = json!(["tools", 0, "diagnostics"]);
    let sent = first.clone();
    assert_eq!(
        run.step(move |api| async move { api.diagnostic(sent).unwrap() })
            .await,
        [json!(["s", diagnostics, [to_json(&first)]])]
    );
    let sent = second.clone();
    assert_eq!(
        run.step(move |api| async move { api.diagnostic(sent).unwrap() })
            .await,
        [json!(["p", diagnostics, 1, 0, [to_json(&second)]])]
    );
    // Settlement moves everything into the result entry and keeps the slot small.
    let settled = run.finish().await;
    assert!(contains_all(
        &settled[0],
        &[
            json!(["s", ["tools", 0, "status"], "done"]),
            json!(["d", details]),
            json!(["d", diagnostics]),
        ]
    ));
    run.harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn streams_partial_text_as_appends() {
    let setup = chat_setup_with(RegisterFauxProviderOptions {
        tokens_per_second: Some(400.0),
        token_size: Some(FauxTokenSize {
            min: Some(4),
            max: Some(4),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    setup.faux.set_responses([faux_assistant_message(
        "word ".repeat(250).as_str(),
        FauxMessageOptions::default(),
    )
    .into()]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let commits = record_live_ops(&harness);
    submit_and_wait(&root).await;
    let commits = commits.lock().clone();
    let partials = &commits[2..commits.len() - 1];
    assert!(partials.len() > 2, "{}", partials.len());
    assert_eq!(partials[0].len(), 1);
    assert_eq!(partials[0][0][0], json!("s"));
    assert_eq!(partials[0][0][1], json!(["generation", "message"]));
    assert!(partials[0][0][2].is_object());
    for ops in &partials[1..] {
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0][0], json!("a"));
        assert_eq!(
            ops[0][1],
            json!(["generation", "message", "content", 0, "text"])
        );
        assert!(ops[0][2].is_string());
    }
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn stores_a_complete_base_exactly_in_the_commits_where_nothing_runs() {
    let storage = Arc::new(ControlledStorage::new());
    let setup = chat_setup();
    for name in ["first", "second"] {
        add_tool(
            &setup.registry,
            simple_tool(name, move |_, api, _| async move {
                api.output(&format!("{name} output\n"))?;
                tokio::time::sleep(Duration::from_millis(150)).await;
                api.output(&format!("{name} more\n"))?;
                Ok(ToolExecutionResult::default())
            }),
        );
    }
    setup
        .faux
        .set_responses([tool_calls(&[("first", "a"), ("second", "b")]), done()]);
    setup.settings(|s| s.tool_execution = Some(ToolExecutionMode::Sequential));
    let (harness, root) = open_chat(storage.clone() as Arc<dyn Storage>, &setup).await;
    // The published value of every ordinary `pi.live` update, in commit order.
    let values: Arc<Mutex<Vec<(crate::durable::ids::DocumentId, JsonValue)>>> = Arc::default();
    let sink = values.clone();
    let unsubscribe = harness
        .subscribe_commits(move |publication: &CommitPublication, _| {
            for change in &publication.changes {
                if let CommitChange::Document(change) = change
                    && change.record.kind == "pi.live"
                    && !change.ops.is_empty()
                    && let Some(value) = &change.value
                {
                    sink.lock().push((change.record.id, (**value).clone()));
                }
            }
        })
        .unwrap();
    submit_and_wait(&root).await;
    drop(unsubscribe);
    let values = values.lock().clone();
    let live_id = values[0].0;
    let written: Vec<&'static str> = storage
        .commits
        .lock()
        .iter()
        .flatten()
        .filter_map(|write| match write {
            StorageWrite::DocumentChange { id, content } if *id == live_id => Some(match content {
                DocumentContent::Base { .. } => "base",
                DocumentContent::Delta { .. } => "delta",
            }),
            _ => None,
        })
        .collect();
    assert_eq!(written.len(), values.len());
    let nothing_runs = |value: &JsonValue| {
        value.get("generation").is_none()
            && !value["tools"]
                .as_array()
                .is_some_and(|slots| slots.iter().any(|slot| slot["status"] == "running"))
    };
    for (kind, (_, value)) in written.iter().zip(&values) {
        assert_eq!(*kind == "base", nothing_runs(value), "{kind} {value}");
    }
    // Bases at the handover, after each sequential tool, after the round, and when the run ends.
    assert!(written.iter().filter(|kind| **kind == "base").count() >= 5);
    assert!(written.contains(&"delta"));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn starts_calls_the_request_did_not_offer_as_done_and_marks_a_faulted_tools_slot_done_without_an_entry()
 {
    let setup = chat_setup();
    // Not strict JSON: the result commit throws and the scheduler faults the task.
    add_tool(
        &setup.registry,
        simple_tool("bad", |_, _, _| async {
            let mut usage = crate::types::Usage::default();
            usage.cost.total = f64::NAN;
            Ok(ToolExecutionResult {
                content: Some(Vec::new()),
                usage: Some(usage),
                ..ToolExecutionResult::default()
            })
        }),
    );
    setup
        .faux
        .set_responses([tool_calls(&[("ghost", "g"), ("bad", "b")]), done()]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let commits = record_live_ops(&harness);
    submit_and_wait(&root).await;
    let ghost_result = all_entries(&root)
        .await
        .iter()
        .find(|entry| entry.kind == "pi.tool-result")
        .unwrap()
        .id;
    let commits = commits.lock().clone();
    let handover = commits
        .iter()
        .find(|ops| ops.iter().any(|op| op[0] == "s" && op[1][0] == "tools"))
        .unwrap();
    let tools = handover
        .iter()
        .find(|op| op[0] == "s" && op[1] == json!(["tools"]))
        .unwrap();
    assert_eq!(
        tools[2][0],
        json!({ "callId": "g", "name": "ghost", "status": "done", "entry": ghost_result })
    );
    assert_eq!(
        (
            &tools[2][1]["callId"],
            &tools[2][1]["name"],
            &tools[2][1]["status"]
        ),
        (&json!("b"), &json!("bad"), &json!("pending"))
    );
    assert!(tools[2][1]["taskId"].is_number());
    // The fault cleanup writes only the status.
    assert!(commits.contains(&vec![json!(["s", ["tools", 1, "status"], "done"])]));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn commits_the_tool_calling_answer_its_tool_tasks_the_generations_wait_and_the_tool_round_in_one_commit()
 {
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        simple_tool("noop", |_, _, _| async { empty() }),
    );
    setup
        .faux
        .set_responses([tool_calls(&[("noop", "a"), ("noop", "b")]), done()]);
    let (harness, root) = open_chat(Arc::new(MemoryStorage::new()), &setup).await;
    let handover: Arc<Mutex<Option<CommitPublication>>> = Arc::default();
    let sink = handover.clone();
    let unsubscribe = harness
        .subscribe_commits(move |publication, _| {
            for change in &publication.changes {
                if let CommitChange::Document(change) = change
                    && change.record.kind == "pi.live"
                    && change
                        .value
                        .as_ref()
                        .and_then(|value| value["tools"].as_array().map(Vec::len))
                        == Some(2)
                {
                    sink.lock().get_or_insert_with(|| publication.clone());
                }
            }
        })
        .unwrap();
    submit_and_wait(&root).await;
    drop(unsubscribe);
    let publication = handover.lock().clone().unwrap();
    let mut kinds: Vec<String> = publication
        .changes
        .iter()
        .filter_map(|change| match change {
            CommitChange::Entry(entry) => Some(entry.kind.clone()),
            CommitChange::Task(task) => Some(task.kind.clone()),
            _ => None,
        })
        .collect();
    kinds.sort();
    assert_eq!(
        kinds,
        ["pi.assistant", "pi.generation", "pi.tool", "pi.tool"]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn writes_an_aborted_tools_slot_with_field_level_ops() {
    let run = Drive::open(ToolOutputLimits::default()).await;
    run.step(|api| async move { api.output("partial\n").unwrap() })
        .await;
    run.step(|api| async move { api.details(json!({ "n": 1 }), &context()).await.unwrap() })
        .await;
    let task_id = live(&run.harness, &run.root).await.unwrap().tools.unwrap()[0]
        .task_id
        .unwrap();
    let before = run.commits.lock().len();
    run.harness.abort_task(task_id, &context()).await.unwrap();
    let is_abort_commit = |ops: &Ops| {
        ops.iter()
            .any(|op| op[0] == "s" && op[1] == json!(["tools", 0, "status"]))
    };
    let commits = run.commits.clone();
    wait_for(move || {
        let commits = commits.clone();
        async move { commits.lock()[before..].iter().any(is_abort_commit) }
    })
    .await;
    let abort_commit = run.commits.lock()[before..]
        .iter()
        .find(|ops| is_abort_commit(ops))
        .unwrap()
        .clone();
    assert_eq!(abort_commit.len(), 4, "{abort_commit:?}");
    assert!(contains_all(
        &abort_commit,
        &[
            json!(["s", ["tools", 0, "status"], "done"]),
            json!(["d", output()]),
            json!(["d", ["tools", 0, "details"]]),
        ]
    ));
    assert!(
        abort_commit
            .iter()
            .any(|op| op[0] == "s" && op[1] == json!(["tools", 0, "entry"]) && op[2].is_number())
    );
    run.submission.wait(&context()).await.unwrap();
    run.harness.close(&context()).await.unwrap();
}

#[test]
fn keeps_a_complete_base_exactly_while_nothing_runs() {
    let definition = LIVE_DOC.definition().clone();
    let base = |value: JsonValue| {
        definition
            .checkpoint_when(
                &value,
                &[],
                CheckpointInfo {
                    deltas_since_base: 1000,
                },
            )
            .unwrap()
    };
    let slot = |status: &str| json!({ "callId": "c", "name": "n", "status": status });
    let run = json!({ "taskId": 1, "inputs": [] });
    assert!(base(json!({})));
    assert!(!base(json!({ "run": run, "generation": { "attempt": 1 } })));
    assert!(base(
        json!({ "run": run, "tools": [slot("pending"), slot("done")] })
    ));
    assert!(!base(
        json!({ "run": run, "tools": [slot("done"), slot("running")] })
    ));
    assert!(base(json!({ "run": run })));
}
