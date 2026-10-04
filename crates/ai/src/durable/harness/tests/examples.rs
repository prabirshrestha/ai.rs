//! Ports of `test/examples/06`–`08`, `10`, `12`, `13`, and `24` as tests: each example runs against a Harness and
//! asserts what the TS script prints. The SQLite file of 13 and 24 is `ControlledStorage::persistent()`. Example 07
//! omits the app-tool snippet section (Rust tools have no app metadata), and 24 prints its task graph with the views
//! milestone.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use super::support::*;
use crate::chord::BACKGROUND_CONTEXT;
use crate::durable::documents::define_doc;
use crate::durable::entries::define_entry;
use crate::durable::harness::types::{
    AgentChange, ConversationCreateOptions, ExtensionDefinition, ExtensionsChange, HarnessOptions,
    HarnessSettings, ModelRef, RetryPolicyOverrides, ToolExecutionMode, ToolExecutionResult,
    ToolRegistration,
};
use crate::durable::harness::{
    AGENT_DOC, Conversation, CreateOptions, Harness, create_registry, define_extension,
    define_tool, wrap_tool,
};
use crate::durable::ids::TaskId;
use crate::durable::session::tests::support::ControlledStorage;
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{NextTaskState, TaskDefinition, define_task};
use crate::durable::types::{
    DocDefinition, JoinPolicy, RewindableConversation, RewindableFork, Storage,
    TaskOptions as CreateTaskOptions, TaskOutcome, TaskOutcomeError, TaskOwnership,
    TypedEntryDraft,
};
use crate::models::Models;
use crate::types::{ModelThinkingLevel, TextContent, UserContent};

fn example_tool(name: &str, description: &str) -> ToolRegistration {
    let name_owned = name.to_string();
    define_tool(
        name,
        description,
        json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }),
        move |args, _, _| {
            let text = format!("{name_owned} {}", args["path"].as_str().unwrap_or(""));
            async move {
                Ok(ToolExecutionResult {
                    content: Some(vec![UserContent::Text(TextContent::new(text))]),
                    ..ToolExecutionResult::default()
                })
            }
        },
    )
}

async fn open(storage: Arc<dyn Storage>, registry: &crate::durable::harness::Registry) -> Harness {
    Harness::open(
        storage,
        HarnessOptions::new(Models::default(), Arc::new(registry.clone())),
        &BACKGROUND_CONTEXT,
    )
    .await
    .unwrap()
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Notes {
    text: String,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[tokio::test]
async fn example_06_harness() {
    let context = BACKGROUND_CONTEXT.clone();
    let files = define_extension(ExtensionDefinition {
        tools: vec![example_tool("read", "Read a file")],
        ..ExtensionDefinition::new("files")
    });
    let registry = create_registry();
    registry.install(files).unwrap();
    let harness = open(Arc::new(MemoryStorage::new()), &registry).await;
    let notes = define_doc(DocDefinition::new(
        "example.notes",
        1,
        RewindableConversation {
            fork: RewindableFork::AsOf,
        },
        Notes::default,
    ))
    .unwrap();
    let init_notes = notes.clone();
    let root = harness
        .root(
            &context,
            CreateOptions {
                agent: Some(AgentChange::default().thinking_level(ModelThinkingLevel::Low)),
                init: Some(Arc::new(move |tx, root_id| {
                    let notes = init_notes.clone();
                    Box::pin(async move {
                        tx.doc(&notes, root_id)
                            .await?
                            .edit(|notes| notes.text = "root notes".into())
                    })
                })),
            },
        )
        .await
        .unwrap();
    assert_eq!(root.id.0, 1);
    assert_eq!(
        harness.snapshot(&notes, root.id, &context).await.unwrap(),
        Some(Notes {
            text: "root notes".into()
        })
    );
    assert_eq!(
        to_json(
            &harness
                .snapshot(&*AGENT_DOC, root.id, &context)
                .await
                .unwrap()
        ),
        json!({ "thinkingLevel": "low" })
    );
    let agent = root.agent(&context).await.unwrap();
    assert_eq!(to_json(&agent.thinking_level), json!("low"));
    let extensions: Vec<&str> = agent.extensions.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(extensions, vec!["files"]);
    let tools: Vec<&str> = agent.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tools, vec!["read"]);
    harness.close(&context).await.unwrap();
}

#[tokio::test]
async fn example_07_configuration() {
    let context = BACKGROUND_CONTEXT.clone();
    let read = example_tool("read", "Read a file");
    let write = example_tool("write", "Write a file");
    let grep = example_tool("grep", "Search files");
    let files = define_extension(ExtensionDefinition {
        tools: vec![read.clone(), write.clone()],
        ..ExtensionDefinition::new("files")
    });
    let search = define_extension(ExtensionDefinition {
        tools: vec![grep],
        ..ExtensionDefinition::new("search")
    });
    let registry = create_registry();
    registry.install(files.clone()).unwrap();
    registry.install(search.clone()).unwrap();

    // Settings are Harness-wide run policy, read at every use and never stored.
    let timeout_ms = Arc::new(AtomicU64::new(60_000));
    let mut options = HarnessOptions::new(Models::default(), Arc::new(registry.clone()));
    {
        let timeout_ms = timeout_ms.clone();
        options.settings = Some(Arc::new(move || {
            let stream = crate::durable::harness::types::ConversationStreamOptions {
                timeout_ms: Some(timeout_ms.load(Ordering::SeqCst)),
                ..Default::default()
            };
            HarnessSettings {
                stream: Some(stream),
                retry: Some(RetryPolicyOverrides {
                    max_retries: Some(5),
                    ..RetryPolicyOverrides::default()
                }),
                tool_execution: Some(ToolExecutionMode::Sequential),
                ..HarnessSettings::default()
            }
        }));
    }
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context)
        .await
        .unwrap();
    let root = harness
        .root(&context, CreateOptions::default())
        .await
        .unwrap();
    let tools = || async {
        root.agent(&context)
            .await
            .unwrap()
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(tools().await, vec!["read", "write", "grep"]);

    root.configure(
        AgentChange::default()
            .model(ModelRef {
                provider: "anthropic".into(),
                model_id: "claude-sonnet-4-5".into(),
            })
            .thinking_level(ModelThinkingLevel::High)
            .tools(vec![write.clone(), read.clone()]),
        &context,
    )
    .await
    .unwrap();
    assert_eq!(
        to_json(
            &harness
                .snapshot(&*AGENT_DOC, root.id, &context)
                .await
                .unwrap()
        ),
        json!({
            "model": { "provider": "anthropic", "modelId": "claude-sonnet-4-5" },
            "thinkingLevel": "high",
            "tools": ["write", "read"],
        })
    );
    let agent = root.agent(&context).await.unwrap();
    assert_eq!(
        to_json(&agent.model),
        json!({ "provider": "anthropic", "modelId": "claude-sonnet-4-5" })
    );
    assert_eq!(to_json(&agent.thinking_level), json!("high"));
    assert_eq!(tools().await, vec!["write", "read"]);

    // Deselect an extension; `None` clears a stored field back to the host default.
    root.configure(
        AgentChange {
            extensions: Some(Some(ExtensionsChange::Edit {
                add: None,
                remove: Some(vec![search.clone()]),
            })),
            tools: Some(None),
            ..AgentChange::default()
        },
        &context,
    )
    .await
    .unwrap();
    assert_eq!(tools().await, vec!["read", "write"]);

    // Stored names outlive the code.
    root.configure(
        AgentChange::default().extensions(vec![files.clone(), search.clone()]),
        &context,
    )
    .await
    .unwrap();
    registry.uninstall(&files).unwrap();
    assert_eq!(tools().await, vec!["grep"]);
    registry.install(files.clone()).unwrap();
    assert_eq!(tools().await, vec!["read", "write", "grep"]);

    // Settings changes need no commit; the next use reads the new timeout.
    timeout_ms.store(120_000, Ordering::SeqCst);
    assert_eq!(harness.settings().stream.timeout_ms, Some(120_000));
    harness.close(&context).await.unwrap();
}

#[tokio::test]
async fn example_08_harness_conversations() {
    #[derive(Default, Serialize, Deserialize)]
    struct From {
        from: String,
    }
    let context = BACKGROUND_CONTEXT.clone();
    let harness = open(Arc::new(MemoryStorage::new()), &create_registry()).await;
    let root = harness
        .root(&context, CreateOptions::default())
        .await
        .unwrap();
    root.configure(
        AgentChange::default().thinking_level(ModelThinkingLevel::High),
        &context,
    )
    .await
    .unwrap();

    let message = define_entry::<From>("message").unwrap();
    let root_id = root.id;
    let hello = {
        let message = message.clone();
        root.commit(
            move |tx| async move {
                tx.append_entry_of(
                    &message,
                    root_id,
                    TypedEntryDraft {
                        data: Some(From {
                            from: "example".into(),
                        }),
                        model: Some(vec![user("hello")]),
                        ..TypedEntryDraft::default()
                    },
                )
                .await
            },
            &context,
        )
        .await
        .unwrap()
    };
    assert!(message.is(Some(&hello)));
    assert_eq!(hello.data.as_ref().unwrap()["from"], "example");

    let helper = harness
        .create_conversation(
            ConversationCreateOptions {
                agent: Some(AgentChange::default().thinking_level(ModelThinkingLevel::Minimal)),
                ..ConversationCreateOptions::ownerless()
            },
            &context,
        )
        .await
        .unwrap();
    let retry = root
        .fork(hello.id, ConversationCreateOptions::ownerless(), &context)
        .await
        .unwrap();
    assert_eq!(
        to_json(&helper.agent(&context).await.unwrap().thinking_level),
        json!("minimal")
    );
    assert_eq!(
        to_json(&retry.agent(&context).await.unwrap().thinking_level),
        json!("high")
    );
    assert_eq!(
        harness
            .conversation(retry.id, &context)
            .await
            .unwrap()
            .map(|found| found.id),
        Some(retry.id)
    );
    harness.close(&context).await.unwrap();
}

#[tokio::test]
async fn example_10_registry_reload() {
    let context = BACKGROUND_CONTEXT.clone();
    let read = example_tool("read", "Read a file");
    let registry = create_registry();
    registry
        .install(define_extension(ExtensionDefinition {
            tools: vec![read.clone(), example_tool("grep", "Search files")],
            ..ExtensionDefinition::new("files")
        }))
        .unwrap();
    let harness = open(Arc::new(MemoryStorage::new()), &registry).await;
    let root = harness
        .root(&context, CreateOptions::default())
        .await
        .unwrap();
    let tools = || async {
        root.agent(&context)
            .await
            .unwrap()
            .tools
            .iter()
            .map(|tool| format!("{}: {}", tool.name, tool.description))
            .collect::<Vec<_>>()
    };

    registry
        .install(define_extension(ExtensionDefinition {
            tools: vec![read.clone(), example_tool("grep", "Search files, faster")],
            ..ExtensionDefinition::new("files")
        }))
        .unwrap();
    assert_eq!(
        tools().await,
        vec!["read: Read a file", "grep: Search files, faster"]
    );

    let audit = define_extension(ExtensionDefinition {
        wraps: vec![wrap_tool("read", |tool| {
            Ok(ToolRegistration {
                description: format!("{} (audited)", tool.description),
                ..tool
            })
        })],
        ..ExtensionDefinition::new("audit")
    });
    registry.install(audit.clone()).unwrap();
    assert_eq!(
        tools().await,
        vec!["read: Read a file (audited)", "grep: Search files, faster"]
    );

    registry.uninstall(&audit).unwrap();
    assert_eq!(
        tools().await,
        vec!["read: Read a file", "grep: Search files, faster"]
    );
    harness.close(&context).await.unwrap();
}

#[derive(Serialize, Deserialize)]
struct Amount {
    amount: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
enum PaymentState {
    Prepare,
    Charge { key: String },
}

#[tokio::test]
async fn example_12_tasks() {
    let context = BACKGROUND_CONTEXT.clone();
    let payments = Arc::new(Mutex::new(std::collections::HashMap::<String, u64>::new()));
    let payment = {
        let payments = payments.clone();
        define_task(
            TaskDefinition::<Amount, PaymentState, JsonValue>::new("example.payment", 1, |_| {
                PaymentState::Prepare
            })
            .phase("prepare", |task, runtime, ctx| async move {
                let key = format!("payment-{}", task.id);
                runtime
                    .commit(
                        move |_, _| async move {
                            Ok(Some(NextTaskState::running(PaymentState::Charge { key })))
                        },
                        &ctx,
                    )
                    .await
            })
            .phase("charge", move |task, runtime, ctx| {
                let PaymentState::Charge { key } = &task.checkpoint else {
                    unreachable!()
                };
                let receipt = *payments
                    .lock()
                    .entry(key.clone())
                    .or_insert(task.input.amount * 100);
                async move {
                    runtime
                        .commit(
                            move |_, _| async move {
                                Ok(Some(NextTaskState::completed(
                                    json!({ "receipt": receipt }),
                                )))
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
    };
    let registry = create_registry();
    registry
        .install(define_extension(ExtensionDefinition {
            tasks: vec![payment.any()],
            ..ExtensionDefinition::new("payments")
        }))
        .unwrap();
    let harness = open(Arc::new(MemoryStorage::new()), &registry).await;
    let root = harness
        .root(&context, CreateOptions::default())
        .await
        .unwrap();
    let payment_id = root
        .commit(
            move |tx| async move {
                tx.create_task(
                    &payment,
                    Amount { amount: 5 },
                    CreateTaskOptions::conversation(None),
                )
                .await
            },
            &context,
        )
        .await
        .unwrap();
    let paid = harness.wait_for_task(payment_id, &context).await.unwrap();
    assert_eq!(
        to_json(&paid.state)["outcome"],
        json!({ "status": "completed", "result": { "receipt": 500 } })
    );
    harness.close(&context).await.unwrap();
}

#[tokio::test]
async fn example_13_recovery() {
    #[derive(Serialize, Deserialize)]
    struct To {
        to: u32,
    }
    #[derive(Serialize, Deserialize)]
    struct Tick {
        phase: String,
        n: u32,
    }
    let context = BACKGROUND_CONTEXT.clone();
    let storage = Arc::new(ControlledStorage::persistent());
    let printed = Arc::new(Mutex::new(Vec::<String>::new()));
    let reached_two = crate::durable::session::tests::support::Deferred::default();
    let ticker = {
        let (printed, reached_two) = (printed.clone(), reached_two.clone());
        define_task(
            TaskDefinition::<To, Tick, String>::new("example.ticker", 1, |_| Tick {
                phase: "tick".into(),
                n: 1,
            })
            .phase("tick", move |task, runtime, ctx| {
                let (printed, reached_two) = (printed.clone(), reached_two.clone());
                async move {
                    let n = task.checkpoint.n;
                    // Save the intent before the effect, so a rerun does not print the same tick twice.
                    let memo = format!("printed-{n}");
                    if runtime.memo::<JsonValue>(&memo).await?.is_none() {
                        runtime.memo_with(&memo, json!(true), &ctx).await?;
                        printed.lock().push(format!("tick {n}"));
                    }
                    if n == 2 {
                        reached_two.resolve();
                    }
                    let to = task.input.to;
                    runtime
                        .commit(
                            move |_, _| async move {
                                Ok(Some(if n == to {
                                    NextTaskState::completed(format!("counted to {n}"))
                                } else {
                                    NextTaskState::running(Tick {
                                        phase: "tick".into(),
                                        n: n + 1,
                                    })
                                }))
                            },
                            &ctx,
                        )
                        .await?;
                    // Closing the Harness cancels this wait; the saved checkpoint is where the next Harness continues.
                    runtime.sleep(now_ms() + 50, &ctx).await
                }
            })
            .abort(|_, runtime, ctx| async move {
                runtime
                    .commit(|_, _| async { Ok(Some(aborted_tick())) }, &ctx)
                    .await
            }),
        )
    };
    fn aborted_tick() -> NextTaskState<Tick> {
        NextTaskState::Terminal {
            outcome: TaskOutcome::Aborted {
                reason: None,
                result: None,
            },
        }
    }
    let registry = create_registry();
    registry
        .install(define_extension(ExtensionDefinition {
            tasks: vec![ticker.any()],
            ..ExtensionDefinition::new("ticker")
        }))
        .unwrap();

    // First run: close right after tick 2 is printed, before its next checkpoint is saved.
    let first_run = open(storage.clone(), &registry).await;
    let ticker_id = first_run
        .root(&context, CreateOptions::default())
        .await
        .unwrap()
        .commit(
            move |tx| async move {
                tx.create_task(&ticker, To { to: 5 }, CreateTaskOptions::conversation(None))
                    .await
            },
            &context,
        )
        .await
        .unwrap();
    first_run.resume().unwrap();
    reached_two.wait().await;
    first_run.close(&context).await.unwrap();

    // Read the saved record through a Harness that never resumes, so nothing runs.
    let reader = open(storage.clone(), &registry).await;
    let saved = reader.get_task(ticker_id, &context).await.unwrap().unwrap();
    reader.close(&context).await.unwrap();
    let saved_n = to_json(&saved.state)["checkpoint"]["n"].as_u64().unwrap();
    assert_eq!(to_json(&saved.state)["status"], "pending");
    assert!(saved_n == 2 || saved_n == 3, "{saved_n}");
    assert_eq!(
        to_json(&saved.memos)["printed-2"],
        json!(true),
        "{:?}",
        saved.memos
    );

    // Second run: waiting for the unfinished task enables scheduling and continues it.
    let second_run = open(storage.clone(), &registry).await;
    let counted = second_run.wait_for_task(ticker_id, &context).await.unwrap();
    assert_eq!(
        to_json(&counted.state)["outcome"],
        json!({ "status": "completed", "result": "counted to 5" })
    );
    second_run.close(&context).await.unwrap();
    assert_eq!(
        *printed.lock(),
        vec!["tick 1", "tick 2", "tick 3", "tick 4", "tick 5"]
    );
}

#[derive(Serialize, Deserialize)]
struct Card {
    card: String,
}

#[derive(Serialize, Deserialize)]
struct ChargeState {
    phase: String,
    at: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
enum CheckoutState {
    Pay,
    Decide { payments: Vec<TaskId> },
}

#[derive(Serialize, Deserialize)]
struct Cards {
    cards: Vec<String>,
}

#[tokio::test]
async fn example_24_child_tasks() {
    let context = BACKGROUND_CONTEXT.clone();
    let charged = Arc::new(Mutex::new(HashSet::<String>::new()));
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let payment = {
        let (charged_run, charged_abort, log) = (charged.clone(), charged.clone(), log.clone());
        define_task(
            TaskDefinition::<Card, ChargeState, JsonValue>::new("example.payment", 1, |_| {
                ChargeState {
                    phase: "charge".into(),
                    at: now_ms() + 100,
                }
            })
            .phase("charge", move |task, runtime, ctx| {
                let charged = charged_run.clone();
                async move {
                    let card = task.input.card.clone();
                    if card.starts_with("expired") {
                        let message = format!("{card} declined");
                        return runtime
                            .commit(
                                move |_, _| async move {
                                    Ok(Some(NextTaskState::Terminal {
                                        outcome: TaskOutcome::Failed {
                                            error: TaskOutcomeError {
                                                message,
                                                detail: None,
                                            },
                                            result: None,
                                        },
                                    }))
                                },
                                &ctx,
                            )
                            .await;
                    }
                    charged.lock().insert(card.clone());
                    runtime.sleep(task.checkpoint.at, &ctx).await?;
                    runtime
                        .commit(
                            move |_, _| async move {
                                Ok(Some(NextTaskState::completed(json!({ "card": card }))))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            // Each payment undoes its own effect when aborted.
            .abort(move |task, runtime, ctx| {
                let refunded = charged_abort.lock().remove(&task.input.card);
                log.lock().push(format!(
                    "payment {} aborted{}",
                    task.input.card,
                    if refunded { ", refunded" } else { "" }
                ));
                async move {
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
                }
            }),
        )
    };
    let checkout = {
        let payment = payment.clone();
        let (log_decide, log_abort) = (log.clone(), log.clone());
        define_task(
            TaskDefinition::<Cards, CheckoutState, String>::new("example.checkout", 1, |_| {
                CheckoutState::Pay
            })
            .phase("pay", move |task, runtime, ctx| {
                let payment = payment.clone();
                async move {
                    let owner = task.id.erase();
                    let cards = task.input.cards.clone();
                    runtime
                        .commit(
                            move |tx, _| async move {
                                let mut payments = Vec::new();
                                for card in cards {
                                    payments.push(
                                        tx.create_task(
                                            &payment,
                                            Card { card },
                                            CreateTaskOptions {
                                                ownership: TaskOwnership::Task { task_id: owner },
                                                conversation_id: None,
                                                background: None,
                                            },
                                        )
                                        .await?
                                        .erase(),
                                    );
                                }
                                Ok(Some(NextTaskState::waiting(
                                    CheckoutState::Decide {
                                        payments: payments.clone(),
                                    },
                                    payments,
                                    JoinPolicy::FailFast,
                                )))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            .phase("decide", move |task, runtime, ctx| {
                let log = log_decide.clone();
                async move {
                    let CheckoutState::Decide { payments } = &task.checkpoint else {
                        unreachable!()
                    };
                    let outcomes = runtime.outcomes(payments, &ctx).await?;
                    let statuses: Vec<String> = outcomes
                        .iter()
                        .map(|outcome| to_json(outcome)["status"].as_str().unwrap().to_string())
                        .collect();
                    log.lock()
                        .push(format!("payments: {}", statuses.join(", ")));
                    let paid = statuses.iter().all(|status| status == "completed");
                    runtime
                        .commit(
                            move |_, _| async move {
                                Ok(Some(if paid {
                                    NextTaskState::completed("order placed")
                                } else {
                                    NextTaskState::failed("payment failed")
                                }))
                            },
                            &ctx,
                        )
                        .await
                }
            })
            // Runs only after every payment is done, so the refunds have already happened.
            .abort(move |_, runtime, ctx| {
                log_abort.lock().push("checkout aborted".into());
                async move {
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
                }
            }),
        )
    };
    let registry = create_registry();
    registry
        .install(define_extension(ExtensionDefinition {
            tasks: vec![payment.any(), checkout.any()],
            ..ExtensionDefinition::new("checkout")
        }))
        .unwrap();
    let storage = Arc::new(ControlledStorage::persistent());
    let start_checkout = |root: Conversation, cards: &[&str]| {
        let checkout = checkout.clone();
        let cards = Cards {
            cards: cards.iter().map(|card| card.to_string()).collect(),
        };
        let context = context.clone();
        async move {
            root.commit(
                move |tx| async move {
                    tx.create_task(&checkout, cards, CreateTaskOptions::conversation(None))
                        .await
                        .map(|id| id.erase())
                },
                &context,
            )
            .await
            .unwrap()
        }
    };
    let outcome_status = |record: crate::durable::types::TaskRecord| {
        to_json(&record.state)["outcome"]["status"]
            .as_str()
            .unwrap()
            .to_string()
    };

    let harness = open(storage.clone(), &registry).await;
    let root = harness
        .root(&context, CreateOptions::default())
        .await
        .unwrap();

    // One card is declined.
    let id = start_checkout(root.clone(), &["visa-1", "expired-2", "visa-3", "visa-4"]).await;
    assert_eq!(
        outcome_status(harness.wait_for_task(id, &context).await.unwrap()),
        "failed"
    );
    {
        let mut log = log.lock();
        let decided = log.pop().unwrap();
        log.sort();
        assert_eq!(
            *log,
            vec![
                "payment visa-1 aborted, refunded",
                "payment visa-3 aborted, refunded",
                "payment visa-4 aborted, refunded",
            ]
        );
        assert_eq!(decided, "payments: aborted, failed, aborted, aborted");
        log.clear();
    }

    // The customer cancels.
    let id = start_checkout(root.clone(), &["visa-5", "visa-6", "visa-7", "visa-8"]).await;
    harness.resume().unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    harness.abort_task(id, &context).await.unwrap();
    assert_eq!(
        outcome_status(harness.wait_for_task(id, &context).await.unwrap()),
        "aborted"
    );
    {
        let mut log = log.lock();
        let last = log.pop().unwrap();
        log.sort();
        assert_eq!(
            *log,
            vec![
                "payment visa-5 aborted, refunded",
                "payment visa-6 aborted, refunded",
                "payment visa-7 aborted, refunded",
                "payment visa-8 aborted, refunded",
            ]
        );
        assert_eq!(last, "checkout aborted");
        log.clear();
    }

    // The process stops while the payments run, and a new one continues.
    let id = start_checkout(root.clone(), &["visa-9", "visa-10", "visa-11", "visa-12"]).await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    harness.close(&context).await.unwrap();
    let harness = open(storage.clone(), &registry).await;
    harness
        .root(&context, CreateOptions::default())
        .await
        .unwrap();
    assert_eq!(
        outcome_status(harness.wait_for_task(id, &context).await.unwrap()),
        "completed"
    );
    assert_eq!(
        *log.lock(),
        vec!["payments: completed, completed, completed, completed"]
    );
    harness.close(&context).await.unwrap();
}
