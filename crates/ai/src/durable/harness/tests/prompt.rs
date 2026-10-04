//! Port of `test/harness-prompt.test.ts` (system prompt sections and tool loadout preparation).

use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::json;

use super::support::*;
use crate::chord::{AbortController, AbortReason, Context, JsonValue};
use crate::durable::documents::{AnyDocDefinition, DocArgs};
use crate::durable::entries::SYSTEM_ENTRY;
use crate::durable::errors::{Error, Result};
use crate::durable::harness::agent::{resolve_agent, resolve_settings};
use crate::durable::harness::prompt::{
    SystemDraft, plan_system_entries, render_sections, replay_sections,
};
use crate::durable::harness::types::{
    Agent, DocumentReader, ErasedReader, ExtensionDefinition, PromptInput, PromptSection,
};
use crate::durable::harness::{
    Conversation, CreateOptions, RegistryReader, create_registry, define_extension, section,
    wrap_section,
};
use crate::durable::ids::{ConversationId, EntryId};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{ContextEditAction, EntryDraft, EntryHead};
use crate::types::{Message, ModelThinkingLevel, SystemMessage, Tool};
use crate::utils::transcript::{get_current_tools, to_tool_declaration};

struct NullReader;

impl ErasedReader for NullReader {
    fn snapshot_json(
        &self,
        _definition: Arc<AnyDocDefinition>,
        _args: DocArgs,
        _context: &Context,
    ) -> BoxFuture<'static, Result<Option<Arc<JsonValue>>>> {
        Box::pin(async { Ok(None) })
    }

    fn snapshot_as_of_json(
        &self,
        _definition: Arc<AnyDocDefinition>,
        _args: DocArgs,
        _at: EntryId,
        _context: &Context,
    ) -> BoxFuture<'static, Result<Option<JsonValue>>> {
        Box::pin(async { Ok(None) })
    }
}

fn input() -> PromptInput {
    PromptInput {
        conversation_id: ConversationId(1),
        agent: Agent {
            model: None,
            thinking_level: ModelThinkingLevel::Off,
            extensions: Vec::new(),
            tools: Vec::new(),
            sections: Vec::new(),
            instructions: None,
            cwd: None,
        },
        env: None,
        shown: IndexMap::new(),
        read: DocumentReader(Arc::new(NullReader)),
    }
}

fn desired(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

#[derive(Debug, PartialEq)]
struct Planned {
    sections: JsonValue,
    omit: Option<Vec<EntryId>>,
}

fn planned(sections: JsonValue) -> Planned {
    Planned {
        sections,
        omit: None,
    }
}

fn system_of(draft: &SystemDraft) -> SystemMessage {
    match &draft.model.as_ref().unwrap()[0] {
        Message::System(message) => message.clone(),
        _ => panic!("system message"),
    }
}

async fn append_drafts(conversation: &Conversation, drafts: Vec<SystemDraft>) {
    let id = conversation.id;
    conversation
        .commit(
            move |tx| async move {
                for draft in drafts {
                    tx.append_entry_of(&SYSTEM_ENTRY, id, draft).await?;
                }
                Ok(())
            },
            &context(),
        )
        .await
        .unwrap();
}

/// Plan against the current context, append the plan, and check that replay then yields `desired` in order.
async fn apply(conversation: &Conversation, wanted: &[(&str, &str)]) -> Vec<Planned> {
    let view = conversation.context(&context()).await.unwrap();
    let drafts = plan_system_entries(&view, &desired(wanted), &[], 7);
    append_drafts(conversation, drafts.clone()).await;
    let replayed = replay_sections(&conversation.context(&context()).await.unwrap().messages);
    assert_eq!(replayed, desired(wanted));
    drafts
        .iter()
        .map(|draft| {
            let message = system_of(draft);
            assert_eq!(to_json(&message.content), json!(""));
            assert_eq!(message.timestamp, 7);
            let omit = draft.edits.as_ref().map(|edits| {
                edits
                    .iter()
                    .map(|edit| {
                        assert_eq!(edit.action, ContextEditAction::Omit);
                        edit.target
                    })
                    .collect()
            });
            Planned {
                sections: to_json(&message.sections.unwrap()),
                omit,
            }
        })
        .collect()
}

async fn root() -> Conversation {
    let (harness, _) = open_harness(Arc::new(MemoryStorage::new()), &[], None, None).await;
    harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap()
}

async fn last_system_id(conversation: &Conversation) -> EntryId {
    let page = conversation
        .entries(None, None, 100, None, &context())
        .await
        .unwrap();
    page.items
        .iter()
        .find(|entry| entry.kind == "pi.system")
        .unwrap()
        .id
}

async fn marker(conversation: &Conversation, head: EntryHead) -> EntryId {
    let id = conversation.id;
    conversation
        .commit(
            move |tx| async move {
                let mut draft = EntryDraft::new("summary").head(head);
                draft.model = Some(vec![user("summary")]);
                Ok(tx.append_entry(id, draft).await?.id)
            },
            &context(),
        )
        .await
        .unwrap()
}

fn failing(key: &str, message: &'static str) -> PromptSection {
    section(
        key,
        move |_, _| async move { Err::<Option<String>, _>(Error::message(message)) },
        None,
    )
}

#[tokio::test]
async fn renders_sections_in_order_with_tags_omissions_wrappers_and_failures() {
    let registry = create_registry();
    add_section(
        &registry,
        section(
            "preamble",
            |_, _| async { Ok(Some("You are helpful.".to_string())) },
            Some(false),
        ),
    );
    add_section(
        &registry,
        section("cwd", |_, _| async { Ok(Some("/repo".to_string())) }, None),
    );
    add_section(
        &registry,
        section("skipped", |_, _| async { Ok(None) }, None),
    );
    add_section(&registry, failing("failing", "render failed"));
    add_section(&registry, failing("new-failing", "also failed"));
    registry
        .install(define_extension(ExtensionDefinition {
            wraps: vec![wrap_section("cwd", |inner| {
                let render = inner.render.clone();
                Ok(PromptSection {
                    render: Arc::new(move |value, ctx| {
                        let render = render.clone();
                        Box::pin(async move {
                            let text = render(value, ctx).await?;
                            Ok(Some(format!("{} (git)", text.unwrap_or_default())))
                        })
                    }),
                    ..inner
                })
            })],
            ..ExtensionDefinition::new("git")
        }))
        .unwrap();
    let reports = Reports::default();
    let shown = desired(&[("failing", "<failing>\nold\n</failing>"), ("cwd", "stale")]);
    let agent = resolve_agent(
        None,
        &registry.snapshot(),
        &resolve_settings(None),
        &|error| panic!("{error}"),
    );
    let callback = reports.callback();
    let rendered = render_sections(
        &agent.sections,
        &input(),
        &shown,
        |error| callback(error),
        &context(),
    )
    .await
    .unwrap();
    assert_eq!(
        rendered,
        desired(&[
            ("preamble", "You are helpful."),
            ("cwd", "<cwd>\n/repo (git)\n</cwd>"),
            ("failing", "<failing>\nold\n</failing>"),
        ])
    );
    assert_eq!(reports.messages(), ["render failed", "also failed"]);
}

#[tokio::test]
async fn propagates_section_errors_after_cancellation() {
    let controller = AbortController::new();
    controller.abort(Some(AbortReason::message("cancelled")));
    let cancelled = signal_context(controller.signal());
    let error = render_sections(
        &[failing("a", "cancelled")],
        &input(),
        &IndexMap::new(),
        |_| {},
        &cancelled,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("cancelled"));
}

#[test]
fn replays_sections_in_place_deletes_on_none_and_appends_re_additions() {
    let shown = replay_sections(&[
        system(&[("a", Some("1")), ("b", Some("2")), ("c", Some("3"))]),
        user("x"),
        system(&[("b", Some("20")), ("a", None)]),
        system(&[("a", Some("10"))]),
    ]);
    assert_eq!(shown, desired(&[("b", "20"), ("c", "3"), ("a", "10")]));
}

#[tokio::test]
async fn emits_minimal_value_patches_removals_and_additions() {
    let conversation = root().await;
    assert_eq!(
        apply(&conversation, &[("a", "1"), ("b", "2"), ("c", "3")]).await,
        [planned(json!({ "a": "1", "b": "2", "c": "3" }))]
    );
    assert_eq!(
        apply(
            &conversation,
            &[("a", "1"), ("b", "20"), ("c", "3"), ("d", "4")]
        )
        .await,
        [planned(json!({ "b": "20", "d": "4" }))]
    );
    assert_eq!(
        apply(&conversation, &[("a", "1"), ("c", "3"), ("d", "4")]).await,
        [planned(json!({ "b": null }))]
    );
    assert_eq!(
        apply(&conversation, &[("a", "1"), ("c", "3"), ("d", "4")]).await,
        []
    );
    assert_eq!(
        apply(&conversation, &[]).await,
        [planned(json!({ "a": null, "c": null, "d": null }))]
    );
}

#[tokio::test]
async fn rewrites_order_only_changes_and_re_additions_as_two_entries() {
    let conversation = root().await;
    apply(&conversation, &[("a", "1"), ("b", "2")]).await;
    let planned_order = apply(&conversation, &[("b", "2"), ("a", "1")]).await;
    assert_eq!(
        planned_order,
        [
            planned(json!({ "a": null, "b": null })),
            planned(json!({ "b": "2", "a": "1" })),
        ]
    );
    // JSON objects compare unordered; the re-added order is checked on the section keys.
    let readded = root().await;
    apply(&readded, &[("a", "1"), ("b", "2"), ("c", "3")]).await;
    assert_eq!(
        apply(&readded, &[("a", "1"), ("c", "3")]).await,
        [planned(json!({ "b": null }))]
    );
    // Patching would append `b` after `c`.
    assert_eq!(
        apply(&readded, &[("a", "1"), ("b", "2"), ("c", "3")]).await,
        [
            planned(json!({ "a": null, "c": null })),
            planned(json!({ "a": "1", "b": "2", "c": "3" })),
        ]
    );
}

#[tokio::test]
async fn rebaselines_after_a_head_marker_omitting_retained_deltas_on_both_sides_of_it() {
    let conversation = root().await;
    apply(&conversation, &[("a", "1"), ("b", "2")]).await;
    let id = conversation.id;
    conversation
        .commit(
            move |tx| async move {
                let mut draft = EntryDraft::new("pi.user");
                draft.model = Some(vec![user("hi")]);
                tx.append_entry(id, draft).await
            },
            &context(),
        )
        .await
        .unwrap();
    apply(&conversation, &[("a", "1"), ("b", "20")]).await;
    let delta = last_system_id(&conversation).await;
    // The head keeps the delta but cuts its baseline: replay alone would show only `b`.
    marker(&conversation, EntryHead::Id(delta)).await;
    assert_eq!(
        apply(&conversation, &[("a", "1"), ("b", "20")]).await,
        [Planned {
            sections: json!({ "a": "1", "b": "20" }),
            omit: Some(vec![delta]),
        }]
    );
    let baseline = last_system_id(&conversation).await;
    // A system entry follows the marker now, so later changes are ordinary patches.
    assert_eq!(
        apply(&conversation, &[("a", "1"), ("b", "21")]).await,
        [planned(json!({ "b": "21" }))]
    );
    let after = last_system_id(&conversation).await;
    // A second marker keeps deltas from both sides of the first one.
    marker(&conversation, EntryHead::Id(delta)).await;
    assert_eq!(
        apply(&conversation, &[("a", "1"), ("b", "21")]).await,
        [Planned {
            sections: json!({ "a": "1", "b": "21" }),
            omit: Some(vec![delta, baseline, after]),
        }]
    );
}

#[tokio::test]
async fn writes_a_complete_post_head_baseline_even_when_replay_already_matches() {
    let conversation = root().await;
    apply(&conversation, &[("a", "1")]).await;
    let baseline = last_system_id(&conversation).await;
    marker(&conversation, EntryHead::Id(baseline)).await;
    assert_eq!(
        apply(&conversation, &[("a", "1")]).await,
        [Planned {
            sections: json!({ "a": "1" }),
            omit: Some(vec![baseline]),
        }]
    );
    marker(&conversation, EntryHead::SelfEntry).await;
    assert_eq!(apply(&conversation, &[]).await, [planned(json!({}))]);
    assert_eq!(apply(&conversation, &[]).await, []);
}

fn declaration(name: &str, description: &str) -> Tool {
    Tool {
        name: name.into(),
        description: description.into(),
        parameters: json!({ "type": "object", "properties": {} }),
        constrained_sampling: None,
    }
}

#[derive(Debug, PartialEq, Default)]
struct ToolPlan {
    removed: Option<Vec<String>>,
    added: Option<Vec<String>>,
    sections: Option<JsonValue>,
}

fn added(names: &[&str]) -> ToolPlan {
    ToolPlan {
        added: Some(names.iter().map(|name| name.to_string()).collect()),
        ..ToolPlan::default()
    }
}

fn names(names: &[&str]) -> Option<Vec<String>> {
    Some(names.iter().map(|name| name.to_string()).collect())
}

/// Plan tools only, append the plan, check that replay offers `tools` in order, and return each message's changes.
async fn apply_tools(
    conversation: &Conversation,
    tools: &[Tool],
    sections: &[(&str, &str)],
) -> Vec<ToolPlan> {
    let view = conversation.context(&context()).await.unwrap();
    let drafts = plan_system_entries(&view, &desired(sections), tools, 7);
    append_drafts(conversation, drafts.clone()).await;
    let offered = get_current_tools(&conversation.context(&context()).await.unwrap().messages);
    assert_eq!(
        offered,
        tools.iter().map(to_tool_declaration).collect::<Vec<_>>()
    );
    drafts
        .iter()
        .map(|draft| {
            let message = system_of(draft);
            ToolPlan {
                removed: message
                    .tools_removed
                    .map(|tools| tools.into_iter().map(|tool| tool.name).collect()),
                added: message
                    .tools_added
                    .map(|tools| tools.into_iter().map(|tool| tool.name).collect()),
                sections: message.sections.map(|sections| to_json(&sections)),
            }
        })
        .collect()
}

#[tokio::test]
async fn adds_removes_replaces_changed_declarations_and_rewrites_the_order_when_needed() {
    let conversation = root().await;
    let (a, b, c) = (
        declaration("a", "a"),
        declaration("b", "b"),
        declaration("c", "c"),
    );
    assert_eq!(
        apply_tools(&conversation, &[a.clone(), b.clone()], &[]).await,
        [added(&["a", "b"])]
    );
    assert_eq!(
        apply_tools(&conversation, &[a.clone(), b.clone()], &[]).await,
        []
    );
    assert_eq!(
        apply_tools(&conversation, &[a.clone(), b.clone(), c.clone()], &[]).await,
        [added(&["c"])]
    );
    assert_eq!(
        apply_tools(&conversation, &[a.clone(), c.clone()], &[]).await,
        [ToolPlan {
            removed: names(&["b"]),
            ..ToolPlan::default()
        }]
    );
    // A changed declaration at the end is removed and re-added in place.
    let c2 = declaration("c", "changed");
    assert_eq!(
        apply_tools(&conversation, &[a.clone(), c2.clone()], &[]).await,
        [ToolPlan {
            removed: names(&["c"]),
            added: names(&["c"]),
            sections: None,
        }]
    );
    // A changed declaration in the middle would move to the end, so the whole order is rewritten.
    let a2 = declaration("a", "changed");
    assert_eq!(
        apply_tools(&conversation, &[a2.clone(), c2.clone()], &[]).await,
        [ToolPlan {
            removed: names(&["a", "c"]),
            added: names(&["a", "c"]),
            sections: None,
        }]
    );
    // Order-only change.
    assert_eq!(
        apply_tools(&conversation, &[c2.clone(), a2.clone()], &[]).await,
        [ToolPlan {
            removed: names(&["a", "c"]),
            added: names(&["c", "a"]),
            sections: None,
        }]
    );
    assert_eq!(
        apply_tools(&conversation, &[], &[]).await,
        [ToolPlan {
            removed: names(&["c", "a"]),
            ..ToolPlan::default()
        }]
    );
}

#[tokio::test]
async fn puts_tool_changes_on_the_last_section_entry_and_re_declares_every_tool_after_a_head_cut() {
    let conversation = root().await;
    let (a, b) = (declaration("a", "a"), declaration("b", "b"));
    assert_eq!(
        apply_tools(
            &conversation,
            std::slice::from_ref(&a),
            &[("x", "1"), ("y", "2")]
        )
        .await,
        [ToolPlan {
            added: names(&["a"]),
            sections: Some(json!({ "x": "1", "y": "2" })),
            removed: None,
        }]
    );
    // Section order changes need two entries; the tool change rides on the second.
    assert_eq!(
        apply_tools(
            &conversation,
            &[a.clone(), b.clone()],
            &[("y", "2"), ("x", "1")]
        )
        .await,
        [
            ToolPlan {
                sections: Some(json!({ "x": null, "y": null })),
                ..ToolPlan::default()
            },
            ToolPlan {
                added: names(&["b"]),
                sections: Some(json!({ "y": "2", "x": "1" })),
                removed: None,
            },
        ]
    );
    marker(&conversation, EntryHead::SelfEntry).await;
    assert_eq!(
        apply_tools(&conversation, &[a, b], &[("y", "2"), ("x", "1")]).await,
        [ToolPlan {
            added: names(&["a", "b"]),
            sections: Some(json!({ "y": "2", "x": "1" })),
            removed: None,
        }]
    );
}

#[test]
fn keeps_the_planned_section_order() {
    // JSON comparisons above are unordered; check the order of an order-changing plan directly.
    let view = crate::durable::harness::types::ContextView {
        messages: vec![system(&[("a", Some("1")), ("b", Some("2"))])],
        ..Default::default()
    };
    let drafts = plan_system_entries(&view, &desired(&[("b", "2"), ("a", "1")]), &[], 1);
    let keys: Vec<Vec<String>> = drafts
        .iter()
        .map(|draft| system_of(draft).sections.unwrap().keys().cloned().collect())
        .collect();
    assert_eq!(keys, [vec!["a", "b"], vec!["b", "a"]]);
}
