//! Ports of `test/examples/00`–`05` as tests: each example runs against a
//! plain Session over `MemoryStorage` and asserts what the TS script prints.
//! Example 03 only creates the task record; running tasks needs the Harness
//! (examples 06+).

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::FutureExt;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::{Deferred, to_json};
use crate::chord::{BACKGROUND_CONTEXT, JsonValue, ListenerOutcome};
use crate::durable::documents::define_doc;
use crate::durable::ids::ConversationId;
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::tasks::{TaskDefinition, define_task};
use crate::durable::types::{
    ConversationOwnership, DocDefinition, EntryDraft, EntryQuery, LatestConversation, LatestFork,
    RewindableConversation, RewindableFork, TaskOptions,
};
use crate::durable::{DocToken, Session, create_session};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Notes {
    text: String,
}

fn notes(text: &str) -> Notes {
    Notes { text: text.into() }
}

fn notes_doc() -> DocToken<Notes, RewindableConversation> {
    define_doc(DocDefinition::new(
        "example.notes",
        1,
        RewindableConversation {
            // A fork starts with the value these notes had at the fork entry.
            fork: RewindableFork::AsOf,
        },
        Notes::default,
    ))
    .unwrap()
}

fn new_session() -> Session {
    create_session(Arc::new(MemoryStorage::new()))
}

async fn conversation(session: &Session) -> ConversationId {
    session
        .commit(
            |tx| async move {
                Ok(tx
                    .create_conversation(ConversationOwnership::Ownerless)
                    .await?
                    .id)
            },
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap()
}

async fn note(
    session: &Session,
    chat: ConversationId,
    data: &str,
    text: &str,
) -> crate::durable::ids::EntryId {
    let (data, text) = (data.to_string(), text.to_string());
    session
        .commit(
            move |tx| async move {
                let entry = tx
                    .append_entry(
                        chat,
                        EntryDraft {
                            data: Some(json!(data)),
                            ..EntryDraft::new("note")
                        },
                    )
                    .await?;
                tx.doc(&notes_doc(), chat)
                    .await?
                    .edit(|notes| notes.text = text)?;
                Ok(entry.id)
            },
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap()
}

async fn snapshot(session: &Session, chat: ConversationId) -> Option<Notes> {
    session
        .snapshot(&notes_doc(), chat, &BACKGROUND_CONTEXT)
        .await
        .unwrap()
}

#[tokio::test]
async fn example_00_conversation() {
    let session = new_session();
    let standalone = session
        .commit(
            |tx| async move {
                tx.create_conversation(ConversationOwnership::Ownerless)
                    .await
            },
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap();
    assert!(standalone.parent.is_none());
    assert!(standalone.owner.is_none());
    session.close(&BACKGROUND_CONTEXT).await.unwrap();
}

#[tokio::test]
async fn example_01_documents() {
    let session = new_session();
    let chat = conversation(&session).await;
    let first_entry = note(&session, chat, "hello", "after hello").await;
    let second_entry = note(&session, chat, "goodbye", "after goodbye").await;
    assert_eq!(snapshot(&session, chat).await, Some(notes("after goodbye")));
    let as_of = |at| session.snapshot_as_of(&notes_doc(), chat, at, &BACKGROUND_CONTEXT);
    assert_eq!(
        as_of(first_entry).await.unwrap(),
        Some(notes("after hello"))
    );
    assert_eq!(
        as_of(second_entry).await.unwrap(),
        Some(notes("after goodbye"))
    );
    session.close(&BACKGROUND_CONTEXT).await.unwrap();
}

#[tokio::test]
async fn example_02_forks() {
    let session = new_session();
    let chat = conversation(&session).await;
    let first_entry = note(&session, chat, "hello", "after hello").await;
    note(&session, chat, "goodbye", "after goodbye").await;

    let branch = session
        .commit(
            move |tx| async move {
                tx.fork_conversation(chat, first_entry, ConversationOwnership::Ownerless)
                    .await
            },
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap();
    let branch_id = branch.id;
    let entries = session
        .commit(
            move |tx| async move {
                tx.scan_entries(EntryQuery::conversation(branch_id), 10, None)
                    .await
            },
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap();
    let transcript: Vec<_> = entries
        .items
        .iter()
        .map(|entry| entry.data.clone())
        .collect();
    assert_eq!(transcript, [Some(json!("hello"))]);
    assert_eq!(
        snapshot(&session, branch_id).await,
        Some(notes("after hello"))
    );

    session
        .commit(
            move |tx| async move {
                tx.doc(&notes_doc(), branch_id)
                    .await?
                    .edit(|notes| notes.text = "changed only in the fork".into())
            },
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap();
    assert_eq!(
        snapshot(&session, branch_id).await,
        Some(notes("changed only in the fork"))
    );
    assert_eq!(snapshot(&session, chat).await, Some(notes("after goodbye")));
    session.close(&BACKGROUND_CONTEXT).await.unwrap();
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Agent {
    conversation_id: ConversationId,
    request_id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct AgentRegistry {
    agents: BTreeMap<String, Agent>,
}

#[tokio::test]
async fn example_03_owned_conversations() {
    let session = new_session();
    // This example only creates the task record; a plain Session never runs it.
    let supervisor = define_task::<(), JsonValue, (), ()>(TaskDefinition::new(
        "example.supervisor",
        1,
        |_: &()| json!({ "phase": "ready" }),
    ));
    // "initial": forks of this conversation start without a registry.
    let registry: DocToken<AgentRegistry, LatestConversation> = define_doc(DocDefinition::new(
        "example.agent-registry",
        1,
        LatestConversation {
            fork: LatestFork::Initial,
        },
        AgentRegistry::default,
    ))
    .unwrap();
    let main = conversation(&session).await;

    let token = registry.clone();
    let (supervisor_id, child) = session
        .commit(
            move |tx| async move {
                let supervisor_id = tx
                    .create_task(
                        &supervisor,
                        (),
                        TaskOptions {
                            background: Some(true),
                            ..TaskOptions::conversation(Some(main))
                        },
                    )
                    .await?
                    .erase();
                let child = tx
                    .create_conversation(ConversationOwnership::Task {
                        task_id: supervisor_id,
                    })
                    .await?;
                let agent = Agent {
                    conversation_id: child.id,
                    request_id: format!("researcher:first-message:{supervisor_id}"),
                };
                tx.doc(&token, main).await?.edit(|registry| {
                    registry.agents.insert("researcher".into(), agent);
                })?;
                Ok((supervisor_id, child))
            },
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap();
    assert_eq!(
        to_json(&child.owner),
        json!({ "conversationId": main, "taskId": supervisor_id })
    );
    let registry = session
        .snapshot(&registry, main, &BACKGROUND_CONTEXT)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        registry.agents["researcher"],
        Agent {
            conversation_id: child.id,
            request_id: format!("researcher:first-message:{supervisor_id}"),
        }
    );
    session.close(&BACKGROUND_CONTEXT).await.unwrap();
}

async fn chat_with_first_note(session: &Session) -> ConversationId {
    session
        .commit(
            |tx| async move {
                let conversation = tx
                    .create_conversation(ConversationOwnership::Ownerless)
                    .await?;
                tx.doc(&notes_doc(), conversation.id)
                    .await?
                    .edit(|notes| notes.text = "first".into())?;
                Ok(conversation.id)
            },
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap()
}

async fn set_notes(session: &Session, chat: ConversationId, text: &'static str) {
    session
        .commit(
            move |tx| async move {
                tx.doc(&notes_doc(), chat)
                    .await?
                    .edit(|notes| notes.text = text.into())
            },
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn example_04_chord_state() {
    let session = new_session();
    let chat = chat_with_first_note(&session).await;
    // documentState() never creates a document: it returns a hydrated read-only state.
    let state = session
        .document_state(&notes_doc(), chat, &BACKGROUND_CONTEXT)
        .await
        .unwrap()
        .expect("notes are present");
    let deliveries: Arc<Mutex<Vec<(String, u64, JsonValue)>>> = Arc::default();
    let sink = deliveries.clone();
    let stop = state.subscribe(move |value, _, delivery| {
        sink.lock().push((
            format!("{:?}", delivery.kind),
            delivery.sequence,
            value.as_deref().cloned().unwrap_or(JsonValue::Null),
        ));
        ListenerOutcome::ok()
    });
    set_notes(&session, chat, "published through Chord").await;
    super::support::flush().await;
    stop.unsubscribe();
    state.dispose().unwrap();
    let deliveries = deliveries.lock().clone();
    assert_eq!(deliveries.len(), 2);
    assert_eq!(
        (deliveries[0].1, &deliveries[0].2),
        (0, &json!({ "text": "first" }))
    );
    assert_eq!(
        (deliveries[1].1, &deliveries[1].2),
        (1, &json!({ "text": "published through Chord" }))
    );
    session.close(&BACKGROUND_CONTEXT).await.unwrap();
}

#[tokio::test]
async fn example_05_watches() {
    let session = new_session();
    let chat = chat_with_first_note(&session).await;
    let watch = session
        .watch_doc(&notes_doc(), chat, &BACKGROUND_CONTEXT)
        .await
        .unwrap()
        .expect("notes are present");
    assert_eq!(*watch.value().unwrap(), json!({ "text": "first" }));
    let delivered = Deferred::default();
    let updates: Arc<Mutex<Vec<JsonValue>>> = Arc::default();
    let (signal, sink) = (delivered.clone(), updates.clone());
    watch
        .start(move |value, _, _| {
            sink.lock()
                .push(value.as_deref().cloned().unwrap_or(JsonValue::Null));
            signal.resolve();
            async { Ok(()) }.boxed()
        })
        .unwrap();
    set_notes(&session, chat, "observed asynchronously").await;
    delivered.wait().await;
    watch.stop().await;
    assert_eq!(
        *updates.lock(),
        [json!({ "text": "observed asynchronously" })]
    );
    session.close(&BACKGROUND_CONTEXT).await.unwrap();
}
