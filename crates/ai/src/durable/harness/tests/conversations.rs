//! Port of `test/harness-conversations.test.ts`.
//!
//! Divergence: the reopened SQLite file is `ControlledStorage::persistent()`.

use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::{add_tool, assert_err, context, open_harness, to_json, tool, user};
use crate::durable::DocToken;
use crate::durable::documents::define_doc;
use crate::durable::entries::define_entry;
use crate::durable::errors::Error;
use crate::durable::harness::agent::{AGENT_DOC, configure};
use crate::durable::harness::live::LIVE_DOC;
use crate::durable::harness::provider::PROVIDER_DOC;
use crate::durable::harness::types::{
    AgentChange, ConversationCreateOptions, ConversationInit, ExtensionDefinition, HarnessOptions,
    ModelRef, ToolsChange,
};
use crate::durable::harness::{
    Conversation, CreateOptions, Harness, create_registry, define_extension,
};
use crate::durable::ids::{ConversationId, ROOT_CONVERSATION_ID};
use crate::durable::session::create_session;
use crate::durable::session::tests::support::{ControlledStorage, write_type};
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{TaskDefinition, define_task};
use crate::durable::types::{
    ConversationOwnership, DocDefinition, EntryDraft, EntryRecord, RewindableConversation,
    RewindableFork, Storage, TaskOptions,
};
use crate::models::create_models;
use crate::types::ModelThinkingLevel;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
struct Note {
    text: String,
}

static NOTE_DOC: LazyLock<DocToken<Note, RewindableConversation>> = LazyLock::new(|| {
    define_doc(DocDefinition::new(
        "test.note",
        1,
        RewindableConversation {
            fork: RewindableFork::AsOf,
        },
        Note::default,
    ))
    .unwrap()
});

fn is_uuid_v7(value: &str) -> bool {
    let bytes = value.as_bytes();
    value.len() == 36
        && [8, 13, 18, 23].iter().all(|&index| bytes[index] == b'-')
        && bytes[14] == b'7'
        && matches!(bytes[19], b'8' | b'9' | b'a' | b'b')
        && value.chars().enumerate().all(|(index, c)| {
            [8, 13, 18, 23].contains(&index) || c.is_ascii_hexdigit() && !c.is_ascii_uppercase()
        })
}

async fn append(conversation: &Conversation, text: &str) -> EntryRecord {
    let id = conversation.id;
    let mut draft = EntryDraft::new("message");
    draft.model = Some(vec![user(text)]);
    conversation
        .commit(
            move |tx| async move { tx.append_entry(id, draft).await },
            &context(),
        )
        .await
        .unwrap()
}

/// Texts of every entry, newest first, paging two at a time.
async fn all_texts(conversation: &Conversation) -> Vec<String> {
    let mut texts = Vec::new();
    let mut cursor = None;
    loop {
        let page = conversation
            .entries(None, None, 2, cursor, &context())
            .await
            .unwrap();
        for entry in &page.items {
            texts.push(
                super::support::text_of(entry.model.as_ref().and_then(|m| m.first())).unwrap(),
            );
        }
        cursor = page.next;
        if cursor.is_none() {
            return texts;
        }
    }
}

fn init<F, Fut>(f: F) -> Option<ConversationInit>
where
    F: Fn(crate::durable::session::Transaction, ConversationId) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = crate::durable::errors::Result<()>> + Send + 'static,
{
    Some(Arc::new(move |tx, id| Box::pin(f(tx, id))))
}

fn ownerless() -> ConversationCreateOptions {
    ConversationCreateOptions::ownerless()
}

async fn provider_session(harness: &Harness, id: ConversationId) -> String {
    harness
        .snapshot(&*PROVIDER_DOC, id, &context())
        .await
        .unwrap()
        .unwrap()
        .session_id
}

async fn agent_json(harness: &Harness, id: ConversationId) -> serde_json::Value {
    to_json(&harness.snapshot(&*AGENT_DOC, id, &context()).await.unwrap())
}

#[derive(Serialize, Deserialize)]
struct Phase {
    phase: String,
}

fn never_task(
    name: &str,
) -> crate::durable::tasks::Task<serde_json::Value, Phase, serde_json::Value> {
    define_task(
        TaskDefinition::new(name, 1, |_: &serde_json::Value| Phase {
            phase: "never".into(),
        })
        .phase("never", |_, _, _| async { Ok(()) })
        .abort(|_, _, _| async { Ok(()) }),
    )
}

// ---- Harness root and conversations ----

#[tokio::test]
async fn creates_the_root_lazily_with_its_agent_change_and_init_in_one_commit() {
    let storage = Arc::new(ControlledStorage::new());
    let (harness, _) = open_harness(storage.clone(), &["read", "bash"], None, None).await;
    assert_eq!(storage.commit_count(), 0);
    let root = harness
        .root(
            &context(),
            CreateOptions {
                agent: Some(AgentChange::default().thinking_level(ModelThinkingLevel::High)),
                init: init(|tx, id| async move {
                    // The agent change applied before init.
                    assert_eq!(
                        tx.doc(&*AGENT_DOC, id).await?.get()?.thinking_level,
                        Some(ModelThinkingLevel::High)
                    );
                    tx.doc(&*NOTE_DOC, id).await?.set(Note {
                        text: "root note".into(),
                    })
                }),
            },
        )
        .await
        .unwrap();
    assert_eq!(root.id, ROOT_CONVERSATION_ID);
    assert_eq!(storage.commit_count(), 1);
    // Conversation, five built-in documents, and the init note.
    let types: Vec<String> = storage.commits.lock()[0].iter().map(write_type).collect();
    assert_eq!(
        types,
        [
            "conversation",
            "document.create",
            "document.create",
            "document.create",
            "document.create",
            "document.create",
            "document.create"
        ]
    );
    assert_eq!(
        harness
            .snapshot(&*LIVE_DOC, root.id, &context())
            .await
            .unwrap(),
        Some(Default::default())
    );
    assert!(is_uuid_v7(&provider_session(&harness, root.id).await));
    assert_eq!(
        agent_json(&harness, root.id).await,
        json!({ "thinkingLevel": "high" })
    );
    let agent = root.agent(&context()).await.unwrap();
    assert_eq!(agent.thinking_level, ModelThinkingLevel::High);
    assert_eq!(
        agent
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["read", "bash"]
    );
    assert_eq!(
        harness
            .snapshot(&*NOTE_DOC, root.id, &context())
            .await
            .unwrap(),
        Some(Note {
            text: "root note".into()
        })
    );

    let again = harness
        .root(
            &context(),
            CreateOptions {
                agent: Some(AgentChange::default().thinking_level(ModelThinkingLevel::Low)),
                init: init(|_, _| async { unreachable!("init runs once") }),
            },
        )
        .await
        .unwrap();
    assert_eq!(again.id, root.id);
    assert_eq!(storage.commit_count(), 1);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_root_and_conversation_identity_and_state_across_reopen() {
    let storage = Arc::new(ControlledStorage::persistent());
    let (harness, _) = open_harness(storage.clone(), &["read"], None, None).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    root.configure(
        AgentChange::default()
            .model(ModelRef::new("anthropic", "claude"))
            .cwd("/repo"),
        &context(),
    )
    .await
    .unwrap();
    let entry = append(&root, "hello").await;
    let child = harness
        .create_conversation(ownerless(), &context())
        .await
        .unwrap();
    let fork = root.fork(entry.id, ownerless(), &context()).await.unwrap();
    let mut session_ids = Vec::new();
    for id in [root.id, child.id, fork.id] {
        session_ids.push(provider_session(&harness, id).await);
    }
    // Regression coverage for #10424: a fork must not inherit its parent's provider identity.
    assert!(session_ids.iter().all(|id| is_uuid_v7(id)));
    assert_eq!(
        session_ids
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    harness.close(&context()).await.unwrap();
    assert!(root.agent(&context()).await.is_err());

    // The new process installs nothing: the stored choices survive, the tools do not resolve.
    let (harness, _) = open_harness(storage.clone(), &[], None, None).await;
    let reopened = harness
        .root(
            &context(),
            CreateOptions {
                agent: None,
                init: init(|_, _| async { unreachable!("init runs once") }),
            },
        )
        .await
        .unwrap();
    assert_eq!(reopened.id, ROOT_CONVERSATION_ID);
    let agent = reopened.agent(&context()).await.unwrap();
    assert_eq!(agent.model, Some(ModelRef::new("anthropic", "claude")));
    assert_eq!(agent.cwd.as_deref(), Some("/repo"));
    assert!(agent.tools.is_empty());
    assert_eq!(all_texts(&reopened).await, ["hello"]);
    assert_eq!(
        harness
            .conversation(child.id, &context())
            .await
            .unwrap()
            .map(|c| c.id),
        Some(child.id)
    );
    let mut reopened_ids = Vec::new();
    for id in [root.id, child.id, fork.id] {
        reopened_ids.push(provider_session(&harness, id).await);
    }
    assert_eq!(reopened_ids, session_ids);
    let reopened_fork = harness
        .conversation(fork.id, &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(all_texts(&reopened_fork).await, ["hello"]);
    assert_eq!(
        reopened_fork.agent(&context()).await.unwrap().model,
        Some(ModelRef::new("anthropic", "claude"))
    );
    assert!(
        harness
            .conversation(ConversationId(999), &context())
            .await
            .unwrap()
            .is_none()
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn creates_independent_conversations_atomically_with_init_and_rolls_back_failures() {
    let storage = Arc::new(ControlledStorage::new());
    let (harness, registry) = open_harness(storage.clone(), &["read"], None, None).await;
    let created = harness
        .create_conversation(
            ConversationCreateOptions {
                init: init(|tx, id| async move {
                    let mut draft = EntryDraft::new("message");
                    draft.model = Some(vec![user("seed")]);
                    tx.append_entry(id, draft).await?;
                    Ok(())
                }),
                ..ownerless()
            },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(storage.commit_count(), 1);
    assert_eq!(all_texts(&created).await, ["seed"]);

    let before = storage.commit_count();
    assert_err(
        harness
            .create_conversation(
                ConversationCreateOptions {
                    agent: Some(AgentChange::default().thinking_level(ModelThinkingLevel::High)),
                    init: init(|_, _| async { Err(Error::message("init failed")) }),
                    ..ownerless()
                },
                &context(),
            )
            .await,
        "init failed",
    );
    assert_eq!(storage.commit_count(), before);

    // A conversation on the default selection follows installs live.
    add_tool(&registry, tool("bash"));
    let names: Vec<String> = created
        .agent(&context())
        .await
        .unwrap()
        .tools
        .iter()
        .map(|t| t.name.clone())
        .collect();
    assert_eq!(names, ["read", "bash"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn forks_at_a_concrete_entry_with_the_as_of_agent_and_applies_agent_and_init_overrides() {
    let (harness, registry) =
        open_harness(Arc::new(MemoryStorage::new()), &["read"], None, None).await;
    let legacy = tool("legacy");
    add_tool(&registry, legacy.clone());
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    root.configure(
        AgentChange::default().thinking_level(ModelThinkingLevel::Low),
        &context(),
    )
    .await
    .unwrap();
    let at = append(&root, "one").await;
    root.configure(
        AgentChange::default()
            .thinking_level(ModelThinkingLevel::High)
            .tools(vec![tool("read")]),
        &context(),
    )
    .await
    .unwrap();
    append(&root, "two").await;

    let child = root.fork(at.id, ownerless(), &context()).await.unwrap();
    assert_eq!(
        agent_json(&harness, child.id).await,
        json!({ "thinkingLevel": "low" })
    );
    assert_eq!(all_texts(&child).await, ["one"]);

    let overridden = root
        .fork(
            at.id,
            ConversationCreateOptions {
                agent: Some(
                    AgentChange::default()
                        .thinking_level(ModelThinkingLevel::Minimal)
                        .tools(vec![legacy]),
                ),
                init: init(|tx, id| async move {
                    assert_eq!(
                        tx.doc(&*AGENT_DOC, id).await?.get()?.thinking_level,
                        Some(ModelThinkingLevel::Minimal)
                    );
                    Ok(())
                }),
                ..ownerless()
            },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(
        agent_json(&harness, overridden.id).await,
        json!({ "thinkingLevel": "minimal", "tools": ["legacy"] })
    );
    assert_eq!(
        agent_json(&harness, root.id).await,
        json!({ "thinkingLevel": "high", "tools": ["read"] })
    );

    let unrelated = harness
        .create_conversation(ownerless(), &context())
        .await
        .unwrap();
    assert!(root.fork(at.id, ownerless(), &context()).await.is_ok());
    assert_err(
        unrelated.fork(at.id, ownerless(), &context()).await,
        "is not visible",
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn paginates_fork_aware_history_through_deep_ancestor_caps_and_same_commit_prefixes() {
    let (harness, _) = open_harness(Arc::new(MemoryStorage::new()), &[], None, None).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    append(&root, "r1").await;
    let root_id = root.id;
    let r2 = root
        .commit(
            move |tx| async move {
                let mut first = EntryDraft::new("message");
                first.model = Some(vec![user("r2")]);
                let mut second = EntryDraft::new("message");
                second.model = Some(vec![user("r3")]);
                let r2 = tx.append_entry(root_id, first).await?;
                tx.append_entry(root_id, second).await?;
                Ok(r2)
            },
            &context(),
        )
        .await
        .unwrap();
    let child = root.fork(r2.id, ownerless(), &context()).await.unwrap();
    let c1 = append(&child, "c1").await;
    append(&child, "c2").await;
    let grandchild = child.fork(c1.id, ownerless(), &context()).await.unwrap();
    append(&grandchild, "g1").await;

    assert_eq!(all_texts(&root).await, ["r3", "r2", "r1"]);
    assert_eq!(all_texts(&child).await, ["c2", "c1", "r2", "r1"]);
    assert_eq!(all_texts(&grandchild).await, ["g1", "c1", "r2", "r1"]);
    let bounded = grandchild
        .entries(Some(r2.id), Some(c1.id), 10, None, &context())
        .await
        .unwrap();
    assert_eq!(
        bounded.items.iter().map(|e| e.id).collect::<Vec<_>>(),
        [c1.id, r2.id]
    );
    let message = define_entry::<()>("message").unwrap();
    assert!(message.is(bounded.items.first()));
    assert!(!message.is(None));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn binds_commits_and_task_creation_to_the_conversation() {
    let (harness, _) = open_harness(Arc::new(MemoryStorage::new()), &[], None, None).await;
    let conversation = harness
        .create_conversation(ownerless(), &context())
        .await
        .unwrap();
    let task = never_task("test.work");
    let bound = task.clone();
    let task_id = conversation
        .commit(
            move |tx| async move {
                tx.create_task(&bound, json!({ "n": 1 }), TaskOptions::conversation(None))
                    .await
            },
            &context(),
        )
        .await
        .unwrap();
    let record = harness
        .commit(
            move |tx| async move { tx.task(task_id.erase()).await },
            &context(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.conversation_id, conversation.id);
    assert_err(
        harness
            .commit(
                move |tx| async move {
                    tx.create_task(&task, json!({ "n": 2 }), TaskOptions::conversation(None))
                        .await
                },
                &context(),
            )
            .await,
        "requires options.conversationId",
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn runs_conversation_created_in_every_creating_commit_after_the_built_ins_and_before_agent_and_init()
 {
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = seen.clone();
    let mut options = HarnessOptions::new(
        create_models(Default::default()),
        Arc::new(create_registry()),
    );
    options.conversation_created = Some(Arc::new(move |tx, conversation| {
        let sink = sink.clone();
        Box::pin(async move {
            let agent = tx.doc(&*AGENT_DOC, conversation.id).await?.get()?;
            let cwd = agent.cwd.clone().unwrap_or_else(|| "undefined".into());
            sink.lock().push(format!(
                "{}:{}:{cwd}",
                conversation.id,
                if conversation.parent.is_none() {
                    "new"
                } else {
                    "fork"
                }
            ));
            tx.doc(&*NOTE_DOC, conversation.id).await?.edit(|note| {
                if note.text.is_empty() {
                    note.text = "created".into();
                }
            })?;
            if agent.cwd.as_deref() == Some("/fail") {
                return Err(Error::message("no"));
            }
            Ok(())
        })
    }));
    let harness = Harness::open(Arc::new(MemoryStorage::new()), options, &context())
        .await
        .unwrap();
    let sink = seen.clone();
    let root = harness
        .root(
            &context(),
            CreateOptions {
                agent: Some(AgentChange::default().cwd("/root")),
                init: init(move |tx, id| {
                    let sink = sink.clone();
                    async move {
                        let note = tx.doc(&*NOTE_DOC, id).await?.get()?;
                        sink.lock().push(format!("init:{}", note.text));
                        Ok(())
                    }
                }),
            },
        )
        .await
        .unwrap();
    // A raw creation in a commit, as in a tool, and a fork, which already has the asOf copies.
    let raw = harness
        .commit(
            |tx| async move {
                Ok(tx
                    .create_conversation(ConversationOwnership::Ownerless)
                    .await?
                    .id)
            },
            &context(),
        )
        .await
        .unwrap();
    let entry = append(&root, "hello").await;
    let fork = root.fork(entry.id, ownerless(), &context()).await.unwrap();
    assert_eq!(
        *seen.lock(),
        [
            format!("{}:new:undefined", root.id),
            "init:created".to_string(),
            format!("{raw}:new:undefined"),
            format!("{}:fork:/root", fork.id),
        ]
    );
    assert_eq!(
        harness.snapshot(&*NOTE_DOC, raw, &context()).await.unwrap(),
        Some(Note {
            text: "created".into()
        })
    );
    // A throw fails the creating commit.
    root.configure(AgentChange::default().cwd("/fail"), &context())
        .await
        .unwrap();
    assert!(root.fork(entry.id, ownerless(), &context()).await.is_ok());
    let failing = append(&root, "after").await;
    assert_err(root.fork(failing.id, ownerless(), &context()).await, "no");
    harness.close(&context()).await.unwrap();
}

// ---- Harness agent ----

#[tokio::test]
async fn replaces_whole_fields_clears_them_with_null_and_leaves_undefined_fields_alone() {
    let (harness, registry) = open_harness(Arc::new(MemoryStorage::new()), &[], None, None).await;
    let (read, bash, edit) = (tool("read"), tool("bash"), tool("edit"));
    registry
        .install(define_extension(ExtensionDefinition {
            tools: vec![read.clone(), bash.clone(), edit.clone()],
            ..ExtensionDefinition::new("coding")
        }))
        .unwrap();
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    let offered = || {
        let root = root.clone();
        async move {
            root.agent(&context())
                .await
                .unwrap()
                .tools
                .iter()
                .map(|t| t.name.clone())
                .collect::<Vec<_>>()
        }
    };
    let agent = root.agent(&context()).await.unwrap();
    assert_eq!(agent.thinking_level, ModelThinkingLevel::Off);
    assert_eq!(offered().await, ["read", "bash", "edit"]);
    assert!(agent.model.is_none());

    root.configure(
        AgentChange::default()
            .model(ModelRef::new("openai", "gpt"))
            .thinking_level(ModelThinkingLevel::Medium),
        &context(),
    )
    .await
    .unwrap();
    root.configure(AgentChange::default().instructions("Be terse."), &context())
        .await
        .unwrap();
    assert_eq!(
        agent_json(&harness, root.id).await,
        json!({ "model": { "provider": "openai", "modelId": "gpt" }, "thinkingLevel": "medium", "instructions": "Be terse." })
    );
    root.configure(
        AgentChange {
            model: Some(None),
            instructions: Some(None),
            ..AgentChange::default()
        },
        &context(),
    )
    .await
    .unwrap();
    assert_eq!(
        agent_json(&harness, root.id).await,
        json!({ "thinkingLevel": "medium" })
    );

    let remove = |tools: Vec<crate::durable::harness::types::ToolRegistration>| AgentChange {
        tools: Some(Some(ToolsChange::Remove(tools))),
        ..AgentChange::default()
    };
    root.configure(remove(vec![edit.clone()]), &context())
        .await
        .unwrap();
    assert_eq!(offered().await, ["read", "bash"]);
    // A new filter replaces the old one: edit is offered again.
    root.configure(remove(vec![bash.clone()]), &context())
        .await
        .unwrap();
    assert_eq!(offered().await, ["read", "edit"]);
    root.configure(
        AgentChange::default().tools(vec![edit.clone(), read.clone()]),
        &context(),
    )
    .await
    .unwrap();
    assert_eq!(offered().await, ["edit", "read"]);
    root.configure(
        AgentChange {
            tools: Some(None),
            ..AgentChange::default()
        },
        &context(),
    )
    .await
    .unwrap();
    assert_eq!(offered().await, ["read", "bash", "edit"]);
    // Names are stored without checking the registry.
    root.configure(
        AgentChange::default().tools(vec![tool("missing"), read]),
        &context(),
    )
    .await
    .unwrap();
    assert_eq!(
        agent_json(&harness, root.id).await["tools"],
        json!(["missing", "read"])
    );
    assert_eq!(offered().await, ["read"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn gives_conversations_created_through_tx_their_documents_empty_an_owner_copy_or_the_forks_as_of_copy()
 {
    let (harness, _) = open_harness(Arc::new(MemoryStorage::new()), &["read"], None, None).await;
    let root = harness
        .root(
            &context(),
            CreateOptions {
                agent: Some(
                    AgentChange::default()
                        .model(ModelRef::new("faux", "m"))
                        .instructions("Main role.")
                        .cwd("/repo"),
                ),
                init: None,
            },
        )
        .await
        .unwrap();
    let owner = never_task("test.owner");
    let (task_id, plain, owned) = root
        .commit(
            move |tx| async move {
                let task_id = tx
                    .create_task(&owner, json!({}), TaskOptions::conversation(None))
                    .await?
                    .erase();
                let plain = tx
                    .create_conversation(ConversationOwnership::Ownerless)
                    .await?;
                let owned = tx
                    .create_conversation(ConversationOwnership::Task { task_id })
                    .await?;
                // The copy exists when create_conversation() returns, so a configure() in the same callback overrides it.
                assert_eq!(
                    tx.doc(&*AGENT_DOC, owned.id).await?.get()?.cwd.as_deref(),
                    Some("/repo")
                );
                configure(&tx, owned.id, &AgentChange::default().cwd("/worktree")).await?;
                Ok((task_id, plain.id, owned.id))
            },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(agent_json(&harness, plain).await, json!({}));
    assert_eq!(
        harness
            .snapshot(&*LIVE_DOC, plain, &context())
            .await
            .unwrap(),
        Some(Default::default())
    );
    assert!(is_uuid_v7(&provider_session(&harness, plain).await));
    assert_ne!(
        provider_session(&harness, owned).await,
        provider_session(&harness, root.id).await
    );
    assert_eq!(
        agent_json(&harness, owned).await,
        json!({ "model": { "provider": "faux", "modelId": "m" }, "instructions": "Main role.", "cwd": "/worktree" })
    );

    // A later owner change does not reach the child.
    root.configure(
        AgentChange::default().thinking_level(ModelThinkingLevel::High),
        &context(),
    )
    .await
    .unwrap();
    assert!(
        agent_json(&harness, owned)
            .await
            .get("thinkingLevel")
            .is_none()
    );

    // A task-owned fork keeps its as-of copy of its fork parent, not its owner's agent.
    let at = root
        .commit(
            move |tx| async move { Ok(tx.append_entry(plain, EntryDraft::new("note")).await?.id) },
            &context(),
        )
        .await
        .unwrap();
    let fork = root
        .commit(
            move |tx| async move {
                Ok(tx
                    .fork_conversation(plain, at, ConversationOwnership::Task { task_id })
                    .await?
                    .id)
            },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(agent_json(&harness, fork).await, json!({}));
    assert_eq!(
        harness
            .snapshot(&*LIVE_DOC, fork, &context())
            .await
            .unwrap(),
        Some(Default::default())
    );
    assert_ne!(
        provider_session(&harness, fork).await,
        provider_session(&harness, plain).await
    );
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn reads_an_absent_agent_for_conversations_a_plain_session_created_without_writing() {
    let storage = Arc::new(ControlledStorage::new());
    let session = create_session(storage.clone() as Arc<dyn Storage>);
    let id = session
        .commit(
            |tx| async move {
                Ok(tx
                    .create_conversation(ConversationOwnership::Ownerless)
                    .await?
                    .id)
            },
            &context(),
        )
        .await
        .unwrap();
    let (harness, _) = open_harness(storage.clone(), &["read"], None, None).await;
    let raw = harness.conversation(id, &context()).await.unwrap().unwrap();
    assert!(
        harness
            .snapshot(&*LIVE_DOC, id, &context())
            .await
            .unwrap()
            .is_none()
    );
    let commits = storage.commit_count();
    let agent = raw.agent(&context()).await.unwrap();
    assert_eq!(agent.thinking_level, ModelThinkingLevel::Off);
    assert_eq!(
        agent
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["read"]
    );
    assert_eq!(storage.commit_count(), commits);
    raw.configure(
        AgentChange::default().thinking_level(ModelThinkingLevel::Low),
        &context(),
    )
    .await
    .unwrap();
    assert_eq!(
        agent_json(&harness, id).await,
        json!({ "thinkingLevel": "low" })
    );
    harness.close(&context()).await.unwrap();
}

// ---- Harness lifecycle ----

#[tokio::test]
async fn returns_stateless_handles_and_rejects_operations_after_close() {
    let (harness, _) = open_harness(Arc::new(MemoryStorage::new()), &[], None, None).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    let again = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    assert_eq!(again.id, root.id);
    harness.close(&context()).await.unwrap();
    assert_err(
        harness.root(&context(), CreateOptions::default()).await,
        "Harness is closed",
    );
    assert_err(
        harness.create_conversation(ownerless(), &context()).await,
        "Harness is closed",
    );
    assert_err(
        harness.conversation(root.id, &context()).await,
        "Harness is closed",
    );
}

#[tokio::test]
async fn forwards_generic_session_document_apis() {
    let (harness, _) = open_harness(Arc::new(MemoryStorage::new()), &[], None, None).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    let id = root.id;
    let entry = root
        .commit(
            move |tx| async move {
                tx.doc(&*NOTE_DOC, id).await?.set(Note {
                    text: "first".into(),
                })?;
                let mut draft = EntryDraft::new("message");
                draft.model = Some(vec![user("m")]);
                tx.append_entry(id, draft).await
            },
            &context(),
        )
        .await
        .unwrap();
    harness
        .commit(
            move |tx| async move {
                tx.doc(&*NOTE_DOC, id).await?.set(Note {
                    text: "second".into(),
                })
            },
            &context(),
        )
        .await
        .unwrap();
    assert_eq!(
        harness.snapshot(&*NOTE_DOC, id, &context()).await.unwrap(),
        Some(Note {
            text: "second".into()
        })
    );
    assert_eq!(
        harness
            .snapshot_as_of(&*NOTE_DOC, id, entry.id, &context())
            .await
            .unwrap(),
        Some(Note {
            text: "first".into()
        })
    );
    let state = harness
        .document_state(&*NOTE_DOC, id, &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(state.value().as_deref(), Some(&json!({ "text": "second" })));
    state.dispose().unwrap();
    harness.close(&context()).await.unwrap();
}
