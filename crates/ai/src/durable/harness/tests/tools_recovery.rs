//! Port of `test/harness-tools-recovery.test.ts`.
//!
//! Divergences: the reopened SQLite file is `ControlledStorage::persistent()`. The faulting result carries a non-finite
//! usage cost, which strict JSON rejects, instead of a function value. Skipped: the real `bash` command case (M9).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde_json::{Value as JsonValue, json};

use super::chat::*;
use super::support::{aborted, add_hooks, add_tool, context, to_json};
use crate::chord::Context;
use crate::durable::env::ExecutionEnv;
use crate::durable::env::local::LocalExecutionEnv;
use crate::durable::errors::Result;
use crate::durable::harness::generation::GENERATION_TASK;
use crate::durable::harness::live::LIVE_DOC;
use crate::durable::harness::scheduler::AbortTaskResult;
use crate::durable::harness::tool::{TOOL_TASK, ToolExecutionApi};
use crate::durable::harness::types::{
    AgentChange, DiagnosticSeverity, EnvFactory, GenerationHooks, Replay, SubmissionDraft,
    ToolDiagnostic, ToolExecutionResult, ToolHooks, ToolRegistration,
};
use crate::durable::harness::{Conversation, Harness, define_tool, hook};
use crate::durable::ids::{SubmissionId, TaskId};
use crate::durable::session::tests::support::{ControlledStorage, Deferred};
use crate::durable::types::{EntryRecord, Storage, SubmissionStatus, TaskQuery, TaskState};
use crate::providers::faux::{
    FauxMessageOptions, FauxResponseStep, faux_assistant_message, faux_tool_call,
};
use crate::types::{Message, StopReason, ToolResultMessage, UserContent};

fn storage() -> Arc<ControlledStorage> {
    Arc::new(ControlledStorage::persistent())
}

async fn open(
    storage: &Arc<ControlledStorage>,
    setup: &ChatSetup,
    env: Option<EnvFactory>,
) -> (Harness, Conversation) {
    let opened = open_chat_with(
        storage.clone() as Arc<dyn Storage>,
        setup,
        ChatOptions {
            env,
            ..ChatOptions::default()
        },
    )
    .await;
    opened.0.resume().unwrap();
    opened
}

fn tool<F, Fut>(name: &str, execute: F) -> ToolRegistration
where
    F: Fn(JsonValue, ToolExecutionApi, Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolExecutionResult>> + Send + 'static,
{
    define_tool(
        name,
        name,
        json!({ "type": "object", "properties": {} }),
        execute,
    )
}

fn call(name: &str) -> FauxResponseStep {
    faux_assistant_message(
        vec![faux_tool_call(name, json!({}), Some("c1"))],
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

fn results(entries: &[EntryRecord]) -> Vec<ToolResultMessage> {
    entries
        .iter()
        .filter(|entry| entry.kind == "pi.tool-result")
        .filter_map(|entry| match &entry.model.as_ref()?[0] {
            Message::ToolResult(result) => Some(result.clone()),
            _ => None,
        })
        .collect()
}

fn text(message: Option<&ToolResultMessage>) -> String {
    message
        .map(|message| {
            message
                .content
                .iter()
                .map(|item| match item {
                    UserContent::Text(text) => text.text.clone(),
                    UserContent::Image(_) => String::new(),
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .unwrap_or_default()
}

async fn tool_task_id(harness: &Harness, root: &Conversation) -> TaskId {
    let found: Arc<Mutex<Option<TaskId>>> = Arc::default();
    wait_for(|| {
        let found = found.clone();
        async move {
            let id = live_state(harness, root)
                .await
                .and_then(|live| live.tools)
                .and_then(|tools| tools.first().and_then(|slot| slot.task_id));
            *found.lock() = id;
            id.is_some()
        }
    })
    .await;
    found.lock().unwrap()
}

async fn submit(root: &Conversation) -> SubmissionId {
    root.submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap()
        .id
}

async fn settle(harness: &Harness, id: SubmissionId) -> SubmissionStatus {
    harness
        .submission(id, &context())
        .await
        .unwrap()
        .unwrap()
        .wait(&context())
        .await
        .unwrap()
        .status
}

/// Writes output, then blocks until its invocation is cancelled the first time it runs. `started` resolves once the
/// output is durable.
struct Blocking {
    registration: ToolRegistration,
    started: Deferred,
    runs: Arc<AtomicUsize>,
}

fn blocking_tool(name: &str, replay: Option<Replay>) -> Blocking {
    let started = Deferred::default();
    let runs = Arc::new(AtomicUsize::new(0));
    let (gate, counter) = (started.clone(), runs.clone());
    let mut registration = tool(name, move |_, api, ctx| {
        let (gate, counter) = (gate.clone(), counter.clone());
        async move {
            let run = counter.fetch_add(1, Ordering::SeqCst) + 1;
            api.output(&format!("run {run}\n"))?;
            api.details(json!({ "run": run }), &ctx).await?;
            if run <= 1 {
                gate.resolve();
                aborted(ctx.abort_signal().unwrap().clone()).await?;
            }
            Ok(ToolExecutionResult::default())
        }
    });
    registration.replay = replay;
    Blocking {
        registration,
        started,
        runs,
    }
}

#[tokio::test]
async fn answers_an_unsafe_tool_interrupted_after_intent_with_its_durable_partial_output() {
    let storage = storage();
    let setup = chat_setup();
    let blocking = blocking_tool("work", None);
    add_tool(&setup.registry, blocking.registration.clone());
    setup.faux.set_responses([call("work"), done()]);
    let (harness, root) = open(&storage, &setup, None).await;
    let id = submit(&root).await;
    blocking.started.wait().await;
    let task_id = tool_task_id(&harness, &root).await;
    harness.close(&context()).await.unwrap();

    let (harness, root) = open(&storage, &setup, None).await;
    let state = harness
        .get_task(task_id, &context())
        .await
        .unwrap()
        .unwrap()
        .state;
    assert_eq!(
        to_json(&state)["checkpoint"],
        json!({ "phase": "execute", "arguments": {}, "replay": "unsafe" })
    );
    assert_eq!(settle(&harness, id).await, SubmissionStatus::Done);
    assert_eq!(blocking.runs.load(Ordering::SeqCst), 1);
    let found = results(&all_entries(&root).await);
    assert!(found[0].is_error);
    assert_eq!(found[0].details, Some(json!({ "run": 1 })));
    assert_eq!(
        text(found.first()),
        "run 1\n|<harness>\n[error] Tool work was interrupted and may have partially run\n</harness>"
    );
    assert_eq!(live_state(&harness, &root).await, Some(Default::default()));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reruns_a_tool_only_when_both_the_stored_and_the_current_replay_policy_are_safe() {
    let cases = [
        (Replay::Safe, Replay::Safe, true),
        (Replay::Safe, Replay::Unsafe, false),
        (Replay::Unsafe, Replay::Safe, false),
    ];
    for (stored, current, reruns) in cases {
        let storage = storage();
        let setup = chat_setup();
        let blocking = blocking_tool("work", Some(stored));
        let registration = add_tool(&setup.registry, blocking.registration.clone());
        setup.faux.set_responses([call("work"), done()]);
        let (harness, root) = open(&storage, &setup, None).await;
        let id = submit(&root).await;
        blocking.started.wait().await;
        harness.close(&context()).await.unwrap();

        registration.dispose();
        let mut replaced = blocking.registration.clone();
        replaced.replay = Some(current);
        add_tool(&setup.registry, replaced);
        let (harness, root) = open(&storage, &setup, None).await;
        assert_eq!(settle(&harness, id).await, SubmissionStatus::Done);
        let found = results(&all_entries(&root).await);
        assert_eq!(
            blocking.runs.load(Ordering::SeqCst),
            if reruns { 2 } else { 1 },
            "{stored:?} {current:?}"
        );
        assert_eq!(found[0].is_error, !reruns);
        if reruns {
            assert_eq!(text(found.first()), "run 2\n");
        }
        harness.close(&context()).await.unwrap();
    }
}

#[tokio::test]
async fn reruns_a_safe_tool_with_the_environment_of_the_conversations_cwd_at_rerun_and_not_once_it_is_deselected()
 {
    for change in ["cwd", "deselect"] {
        let storage = storage();
        let setup = chat_setup();
        let cwds: Arc<Mutex<Vec<Option<String>>>> = Arc::default();
        let started = Deferred::default();
        let (sink, gate) = (cwds.clone(), started.clone());
        let mut work = tool("work", move |_, api, ctx| {
            let (sink, gate) = (sink.clone(), gate.clone());
            async move {
                let count = {
                    let mut cwds = sink.lock();
                    cwds.push(api.env().map(|env| env.cwd()));
                    cwds.len()
                };
                if count == 1 {
                    gate.resolve();
                    aborted(ctx.abort_signal().unwrap().clone()).await?;
                }
                Ok(ToolExecutionResult::default())
            }
        });
        work.replay = Some(Replay::Safe);
        add_tool(&setup.registry, work);
        setup.faux.set_responses([call("work"), done()]);
        let env: EnvFactory = Arc::new(|target, _| {
            let env: Arc<dyn ExecutionEnv> = Arc::new(LocalExecutionEnv::at(
                target.cwd.unwrap_or_else(|| "/".into()),
            ));
            Box::pin(async move { Ok(Some(env)) })
        });
        let (harness, root) = open(&storage, &setup, Some(env.clone())).await;
        root.configure(AgentChange::default().cwd("/one"), &context())
            .await
            .unwrap();
        let id = submit(&root).await;
        started.wait().await;
        let next = if change == "cwd" {
            AgentChange::default().cwd("/two")
        } else {
            AgentChange::default().extensions(Vec::new())
        };
        root.configure(next, &context()).await.unwrap();
        harness.close(&context()).await.unwrap();

        let (harness, root) = open(&storage, &setup, Some(env)).await;
        assert_eq!(settle(&harness, id).await, SubmissionStatus::Done);
        let found = results(&all_entries(&root).await);
        let cwds = cwds.lock().clone();
        if change == "cwd" {
            assert_eq!(cwds, [Some("/one".to_string()), Some("/two".to_string())]);
            assert!(!found[0].is_error);
        } else {
            // A tool that no longer resolves is treated as unsafe: interrupted, not rerun.
            assert_eq!(cwds, [Some("/one".to_string())]);
            assert!(text(found.first()).contains("was interrupted"));
        }
        harness.close(&context()).await.unwrap();
    }
}

#[tokio::test]
async fn reruns_before_tool_when_interrupted_before_intent_and_executes_once() {
    let storage = storage();
    let setup = chat_setup();
    let runs = Arc::new(AtomicUsize::new(0));
    let counter = runs.clone();
    add_tool(
        &setup.registry,
        tool("work", move |_, _, _| {
            counter.fetch_add(1, Ordering::SeqCst);
            async {
                Ok(ToolExecutionResult {
                    content: Some(Vec::new()),
                    ..ToolExecutionResult::default()
                })
            }
        }),
    );
    let reached = Deferred::default();
    let asked = Arc::new(AtomicUsize::new(0));
    let decisions: Arc<Mutex<Vec<String>>> = Arc::default();
    let (gate, count, sink) = (reached.clone(), asked.clone(), decisions.clone());
    add_hooks(
        &setup.registry,
        hook(
            &*TOOL_TASK,
            ToolHooks {
                before_tool: Some(Arc::new(move |_, api, ctx| {
                    let (gate, sink) = (gate.clone(), sink.clone());
                    let attempt = count.fetch_add(1, Ordering::SeqCst) + 1;
                    Box::pin(async move {
                        // A durable first-writer-wins decision survives the rerun.
                        let decision: String = api
                            .memo_with("test:decision", format!("attempt {attempt}"), &ctx)
                            .await?;
                        sink.lock().push(decision);
                        if attempt == 1 {
                            gate.resolve();
                            aborted(ctx.abort_signal().unwrap().clone()).await?;
                        }
                        Ok(None)
                    })
                })),
                ..ToolHooks::default()
            },
        ),
    );
    setup.faux.set_responses([call("work"), done()]);
    let (harness, root) = open(&storage, &setup, None).await;
    let id = submit(&root).await;
    reached.wait().await;
    harness.close(&context()).await.unwrap();

    let (harness, _) = open(&storage, &setup, None).await;
    assert_eq!(settle(&harness, id).await, SubmissionStatus::Done);
    assert_eq!(
        (asked.load(Ordering::SeqCst), runs.load(Ordering::SeqCst)),
        (2, 1)
    );
    assert_eq!(*decisions.lock(), ["attempt 1", "attempt 1"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reruns_the_generation_tools_phase_interrupted_before_its_commit() {
    let storage = storage();
    let setup = chat_setup();
    add_tool(
        &setup.registry,
        tool("work", |_, _, _| async {
            Ok(ToolExecutionResult {
                content: Some(Vec::new()),
                ..ToolExecutionResult::default()
            })
        }),
    );
    let reached = Deferred::default();
    let observed = Arc::new(AtomicUsize::new(0));
    let (gate, count) = (reached.clone(), observed.clone());
    add_hooks(
        &setup.registry,
        hook(
            &*GENERATION_TASK,
            GenerationHooks {
                after_tools: Some(Arc::new(move |_, _, _, ctx| {
                    let gate = gate.clone();
                    let seen = count.fetch_add(1, Ordering::SeqCst) + 1;
                    Box::pin(async move {
                        if seen == 1 {
                            gate.resolve();
                            aborted(ctx.abort_signal().unwrap().clone()).await?;
                        }
                        Ok(())
                    })
                })),
                ..GenerationHooks::default()
            },
        ),
    );
    setup.faux.set_responses([call("work"), done()]);
    let (harness, root) = open(&storage, &setup, None).await;
    let id = submit(&root).await;
    reached.wait().await;
    harness.close(&context()).await.unwrap();

    let (harness, root) = open(&storage, &setup, None).await;
    assert_eq!(settle(&harness, id).await, SubmissionStatus::Done);
    assert_eq!(observed.load(Ordering::SeqCst), 2);
    assert_eq!(
        entry_kinds(&root).await,
        [
            "pi.user",
            "pi.system",
            "pi.assistant",
            "pi.tool-result",
            "pi.assistant"
        ]
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn answers_an_aborted_tool_with_its_partial_output_and_continues_the_run() {
    let storage = storage();
    let setup = chat_setup();
    let blocking = blocking_tool("work", None);
    add_tool(&setup.registry, blocking.registration.clone());
    setup.faux.set_responses([call("work"), done()]);
    let (harness, root) = open(&storage, &setup, None).await;
    let submission = root
        .submit(SubmissionDraft::input("go"), &context())
        .await
        .unwrap();
    blocking.started.wait().await;
    let task_id = tool_task_id(&harness, &root).await;
    assert_eq!(
        harness.abort_task(task_id, &context()).await.unwrap(),
        AbortTaskResult::Marked
    );
    let settled = harness.wait_for_task(task_id, &context()).await.unwrap();
    let outcome = to_json(&settled.state)["outcome"].clone();
    assert_eq!(outcome["status"], json!("aborted"));
    assert!(outcome["result"]["entryId"].is_number());
    assert_eq!(
        submission.wait(&context()).await.unwrap().status,
        SubmissionStatus::Done
    );
    let found = results(&all_entries(&root).await);
    assert_eq!(
        text(found.first()),
        "run 1\n|<harness>\n[error] Tool work was aborted\n</harness>"
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn lets_context_derivation_answer_a_faulted_tool_and_continues_the_run() {
    let storage = storage();
    let setup = chat_setup();
    // A result that is not strict JSON makes the result commit throw, so the scheduler faults the task.
    add_tool(
        &setup.registry,
        tool("bad", |_, _, _| async {
            let mut usage = crate::types::Usage::default();
            usage.cost.total = f64::NAN;
            Ok(ToolExecutionResult {
                content: Some(Vec::new()),
                usage: Some(usage),
                ..ToolExecutionResult::default()
            })
        }),
    );
    let requests: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = requests.clone();
    setup.faux.set_responses([
        call("bad"),
        FauxResponseStep::factory(move |request, _, _, _| {
            let result = request.messages.iter().find_map(|message| match message {
                Message::ToolResult(result) => Some(result.clone()),
                _ => None,
            });
            sink.lock().push(match &result {
                Some(result) => text(Some(result)),
                None => "none".into(),
            });
            Ok(faux_assistant_message(
                "done",
                FauxMessageOptions::default(),
            ))
        }),
    ]);
    let (harness, root) = open(&storage, &setup, None).await;
    let id = submit(&root).await;
    assert_eq!(settle(&harness, id).await, SubmissionStatus::Done);
    assert!(results(&all_entries(&root).await).is_empty());
    assert_eq!(
        *requests.lock(),
        ["Tool result unavailable: history ends before this call completed."]
    );
    let root_id = root.id;
    let tasks = harness
        .commit(
            move |tx| async move {
                Ok(tx
                    .scan_tasks(
                        TaskQuery {
                            conversation_id: Some(root_id),
                            ..TaskQuery::default()
                        },
                        20,
                        None,
                    )
                    .await?
                    .items)
            },
            &context(),
        )
        .await
        .unwrap();
    let faulted = tasks.iter().find(|task| task.kind == "pi.tool").unwrap();
    assert!(matches!(faulted.state, TaskState::Terminal { .. }));
    assert_eq!(
        to_json(&faulted.state)["outcome"]["status"],
        json!("faulted")
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn clears_the_interrupted_attempts_progress_before_a_safe_rerun() {
    let storage = storage();
    let setup = chat_setup();
    let started = [Deferred::default(), Deferred::default()];
    let runs = Arc::new(AtomicUsize::new(0));
    let (gates, counter) = (started.clone(), runs.clone());
    let mut work = tool("work", move |_, api, ctx| {
        let gates = gates.clone();
        let run = counter.fetch_add(1, Ordering::SeqCst);
        async move {
            let info = |message: &str| ToolDiagnostic {
                severity: DiagnosticSeverity::Info,
                message: message.into(),
                code: None,
            };
            if run == 0 {
                api.diagnostic(info("first a"))?;
                api.diagnostic(info("first b"))?;
                api.details(json!({ "run": 1, "extra": true }), &ctx)
                    .await?;
            } else {
                api.diagnostic(info("second"))?;
                api.details(json!({ "run": 2 }), &ctx).await?;
            }
            gates[run].resolve();
            aborted(ctx.abort_signal().unwrap().clone()).await?;
            Ok(ToolExecutionResult::default())
        }
    });
    work.replay = Some(Replay::Safe);
    add_tool(&setup.registry, work);
    setup.faux.set_responses([call("work"), done()]);
    let (harness, root) = open(&storage, &setup, None).await;
    submit(&root).await;
    started[0].wait().await;
    harness.close(&context()).await.unwrap();

    let (harness, root) = open(&storage, &setup, None).await;
    started[1].wait().await;
    let slot = harness
        .snapshot(&*LIVE_DOC, root.id, &context())
        .await
        .unwrap()
        .and_then(|live| live.tools)
        .unwrap()[0]
        .clone();
    assert_eq!(
        to_json(&slot.diagnostics),
        json!([{ "severity": "info", "message": "second" }])
    );
    assert_eq!(slot.details, Some(json!({ "run": 2 })));
    harness.close(&context()).await.unwrap();
}
