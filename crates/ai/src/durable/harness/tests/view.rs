//! Port of `test/harness-view.test.ts`.
//!
//! Divergences: views and operations are compared in their TS JSON form; TS identity checks (`toBe`) are `Arc`
//! pointer checks on the shared parts of a revision.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use super::chat::*;
use super::support::{assert_err, context, signal_context};
use crate::chord::delta::{Op, apply_immutable};
use crate::chord::{AbortController, AbortReason};
use crate::durable::documents::define_doc;
use crate::durable::harness::agent::AGENT_DOC;
use crate::durable::harness::inbox::INBOX_DOC;
use crate::durable::harness::live::LIVE_DOC;
use crate::durable::harness::provider::PROVIDER_DOC;
use crate::durable::harness::types::{AgentChange, ConversationCreateOptions, SubmissionDraft};
use crate::durable::harness::usage::USAGE_DOC;
use crate::durable::harness::view::{ConversationView, ConversationWatch};
use crate::durable::harness::{Conversation, Harness};
use crate::durable::session::tests::support::Deferred;
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{
    CommitChange, DocDefinition, EntryDraft, EntryHead, EntryRecord, LatestConversation,
    LatestFork, Storage, WatchEnd,
};
use crate::error::Error as AiError;
use crate::providers::faux::{FauxMessageOptions, FauxResponseStep, faux_assistant_message};
use crate::types::ModelThinkingLevel;

const MOUNTED: [&str; 5] = ["pi.agent", "pi.inbox", "pi.live", "pi.provider", "pi.usage"];

#[derive(Clone)]
struct Frame {
    value: ConversationView,
    ops: Vec<JsonValue>,
}

struct Recording {
    initial: ConversationView,
    frames: Arc<Mutex<Vec<Frame>>>,
    watch: ConversationWatch,
}

impl Recording {
    fn frames(&self) -> Vec<Frame> {
        self.frames.lock().clone()
    }

    async fn stop(&self) {
        self.watch.stop().await;
    }
}

fn ops_json(ops: &[Op]) -> Vec<JsonValue> {
    ops.iter()
        .map(|op| serde_json::to_value(op).unwrap())
        .collect()
}

fn start_recording(watch: &ConversationWatch) -> Arc<Mutex<Vec<Frame>>> {
    let frames: Arc<Mutex<Vec<Frame>>> = Arc::default();
    let sink = frames.clone();
    watch
        .start(move |value, ops, _| {
            sink.lock().push(Frame {
                value,
                ops: ops_json(&ops),
            });
            Box::pin(async { Ok(()) })
        })
        .unwrap();
    frames
}

/// Start a watch of `conversation` that records its acquisition revision and every delivered frame.
async fn record(conversation: &Conversation) -> Recording {
    let watch = conversation.watch(&context()).await.unwrap();
    let initial = watch.value();
    let frames = start_recording(&watch);
    Recording {
        initial,
        frames,
        watch,
    }
}

/// A freshly built view of `conversation`, for comparing with an advanced one.
async fn fresh(conversation: &Conversation) -> ConversationView {
    let state = conversation.view_state(&context()).await.unwrap();
    let value = state.value();
    state.dispose().unwrap();
    value
}

/// The view as committed state defines it, read without any mount: the active entries and the built-in documents.
async fn committed(
    harness: &Harness,
    conversation: &Conversation,
    view: &ConversationView,
) -> JsonValue {
    let id = conversation.id;
    let ctx = context();
    let mut docs = serde_json::Map::new();
    let values = [
        (
            "pi.agent",
            harness.snapshot_json(&*AGENT_DOC, id, &ctx).await,
        ),
        ("pi.live", harness.snapshot_json(&*LIVE_DOC, id, &ctx).await),
        (
            "pi.inbox",
            harness.snapshot_json(&*INBOX_DOC, id, &ctx).await,
        ),
        (
            "pi.provider",
            harness.snapshot_json(&*PROVIDER_DOC, id, &ctx).await,
        ),
        (
            "pi.usage",
            harness.snapshot_json(&*USAGE_DOC, id, &ctx).await,
        ),
    ];
    for (kind, value) in values {
        if let Some(value) = value.unwrap() {
            docs.insert(kind.to_string(), (*value).clone());
        }
    }
    let entries = conversation.context(&ctx).await.unwrap().entries;
    json!({ "conversation": view.conversation, "entries": entries, "docs": docs })
}

/// Replay every frame's operations from `initial`, checking each delivered revision on the way.
fn replay(initial: &ConversationView, frames: &[Frame]) -> JsonValue {
    let mut value = initial.to_json();
    for frame in frames {
        let ops: Vec<Op> = frame
            .ops
            .iter()
            .map(|op| serde_json::from_value(op.clone()).unwrap())
            .collect();
        value = apply_immutable(&value, &ops).unwrap();
        assert_eq!(value, frame.value.to_json());
    }
    value
}

fn sorted_keys(value: &JsonValue) -> Vec<String> {
    let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    keys
}

fn answer(text: &str) -> FauxResponseStep {
    faux_assistant_message(text, FauxMessageOptions::default()).into()
}

/// Let watch callbacks, which run after the commit, catch up.
async fn drained() {
    for _ in 0..5 {
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
}

fn touches(harness: &Harness, conversation: &Conversation) -> Arc<AtomicUsize> {
    let counter = Arc::new(AtomicUsize::new(0));
    let count = counter.clone();
    let id = conversation.id;
    let unsubscribe = harness
        .subscribe_commits(move |publication, _| {
            let touched = publication.changes.iter().any(|change| match change {
                CommitChange::Entry(entry) => entry.conversation_id == id,
                CommitChange::Document(change) => {
                    change.conversation_id == Some(id)
                        && MOUNTED.contains(&change.record.kind.as_str())
                        && !change.ops.is_empty()
                }
                _ => false,
            });
            if touched {
                count.fetch_add(1, Ordering::SeqCst);
            }
        })
        .unwrap();
    std::mem::forget(unsubscribe);
    counter
}

async fn note(conversation: &Conversation, kind: &str) -> EntryRecord {
    note_draft(conversation, EntryDraft::new(kind)).await
}

async fn note_draft(conversation: &Conversation, draft: EntryDraft) -> EntryRecord {
    let id = conversation.id;
    conversation
        .commit(
            move |tx| async move { tx.append_entry(id, draft).await },
            &context(),
        )
        .await
        .unwrap()
}

fn kinds(view: &ConversationView) -> Vec<String> {
    view.entries
        .iter()
        .map(|entry| entry.kind.clone())
        .collect()
}

fn storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new())
}

#[tokio::test]
async fn hydrates_the_active_entries_and_the_built_in_documents() {
    let setup = chat_setup();
    setup.faux.set_responses(vec![answer("hello")]);
    let (harness, root) = open_chat(storage(), &setup).await;
    root.submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap()
        .wait(&context())
        .await
        .unwrap();
    let view = fresh(&root).await;
    let json = view.to_json();
    assert_eq!(json["conversation"], json!({ "id": root.id }));
    assert_eq!(view.entries.to_vec(), all_entries(&root).await);
    assert_eq!(sorted_keys(&json["docs"]), MOUNTED);
    assert_eq!(json["docs"]["pi.live"], json!({}));
    assert_eq!(json["docs"]["pi.inbox"], json!({ "items": [] }));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn publishes_one_frame_per_touching_commit_whose_operations_rebuild_every_revision() {
    let setup = chat_setup();
    let release = Deferred::default();
    let wait = release.clone();
    let step = FauxResponseStep::async_factory(move |_, options, _, _| {
        let wait = wait.clone();
        async move {
            let signal = options.stream.signal.clone().unwrap();
            tokio::select! {
                _ = wait.wait() => Ok(faux_assistant_message("a longer answer", FauxMessageOptions::default())),
                _ = signal.cancelled() => Err(AiError::Aborted("Request aborted".into())),
            }
        }
    });
    setup.faux.set_responses(vec![step]);
    let (harness, root) = open_chat(storage(), &setup).await;
    let recording = record(&root).await;
    let touching = touches(&harness, &root);
    let submission = root
        .submit(SubmissionDraft::input("hi"), &context())
        .await
        .unwrap();
    let frames = recording.frames.clone();
    wait_for(|| {
        let found = frames.lock().iter().any(|frame| {
            frame
                .value
                .doc("pi.live")
                .is_some_and(|live| live.get("generation").is_some())
        });
        async move { found }
    })
    .await;
    release.resolve();
    submission.wait(&context()).await.unwrap();
    harness.wait_for_idle(&context()).await.unwrap();
    drained().await;
    let frames = recording.frames();
    assert_eq!(frames.len(), touching.load(Ordering::SeqCst));
    assert_eq!(
        replay(&recording.initial, &frames),
        committed(&harness, &root, &recording.initial).await
    );
    let first = &frames[0].ops;
    assert!(first.iter().any(|op| {
        op[0] == "p"
            && op[1] == json!(["entries"])
            && op[2] == 0
            && op[3] == 0
            && op[4][0]["kind"] == "pi.user"
    }));
    // Document operations keep their exact shape under the mount path.
    assert!(
        frames
            .iter()
            .flat_map(|frame| frame.ops.iter())
            .any(|op| *op == json!(["s", ["docs", "pi.live", "generation"], { "attempt": 1 }]))
    );
    recording.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn shares_unchanged_parts_between_revisions_and_skips_commits_that_touch_nothing_mounted() {
    let setup = chat_setup();
    let (harness, root) = open_chat(storage(), &setup).await;
    let other = harness
        .create_conversation(ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    let recording = record(&root).await;
    note(&other, "note").await;
    root.configure(
        AgentChange::default().thinking_level(ModelThinkingLevel::High),
        &context(),
    )
    .await
    .unwrap();
    note(&root, "note").await;
    drained().await;
    let frames = recording.frames();
    assert_eq!(frames.len(), 2);
    assert_eq!(
        frames[0].ops,
        vec![json!(["s", ["docs", "pi.agent", "thinkingLevel"], "high"])]
    );
    assert!(Arc::ptr_eq(
        &frames[0].value.entries,
        &recording.initial.entries
    ));
    assert!(Arc::ptr_eq(
        frames[0].value.doc("pi.live").unwrap(),
        recording.initial.doc("pi.live").unwrap()
    ));
    assert!(Arc::ptr_eq(&frames[1].value.docs, &frames[0].value.docs));
    recording.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cuts_the_entries_at_a_head_marker_keeping_the_entries_from_its_head() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    note(&root, "a").await;
    let b = note(&root, "b").await;
    note(&root, "c").await;
    let recording = record(&root).await;
    let summary = note_draft(&root, EntryDraft::new("summary").head(EntryHead::Id(b.id))).await;
    note(&root, "d").await;
    root.reset(None, &context()).await.unwrap();
    drained().await;
    let frames = recording.frames();
    let all: Vec<Vec<String>> = frames.iter().map(|frame| kinds(&frame.value)).collect();
    assert_eq!(
        all,
        vec![
            vec!["summary", "b", "c"],
            vec!["summary", "b", "c", "d"],
            vec!["pi.reset"]
        ]
    );
    assert_eq!(
        frames[0].ops,
        vec![json!(["p", ["entries"], 0, 1, [summary]])]
    );
    assert_eq!(
        replay(&recording.initial, &frames),
        committed(&harness, &root, &recording.initial).await
    );
    recording.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_only_mounted_entries_for_a_raw_head_write_that_targets_before_the_active_range() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let old = note(&root, "old").await;
    root.reset(None, &context()).await.unwrap();
    let recording = record(&root).await;
    note_draft(
        &root,
        EntryDraft::new("summary").head(EntryHead::Id(old.id)),
    )
    .await;
    drained().await;
    // Model context now starts at `old` again, but the mount never held it (spec §12); a rebuilt mount shows it.
    assert_eq!(kinds(&recording.frames()[0].value), vec!["summary"]);
    recording.stop().await;
    assert_eq!(kinds(&fresh(&root).await), vec!["summary", "old"]);
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cuts_a_forks_view_into_its_inherited_entries() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let a = note(&root, "a").await;
    let b = note(&root, "b").await;
    let fork = root
        .fork(b.id, ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    let recording = record(&fork).await;
    note_draft(&fork, EntryDraft::new("summary").head(EntryHead::Id(b.id))).await;
    drained().await;
    let ids: Vec<_> = recording
        .initial
        .entries
        .iter()
        .map(|entry| entry.id)
        .collect();
    assert_eq!(ids, vec![a.id, b.id]);
    let frames = recording.frames();
    assert_eq!(kinds(&frames[0].value), vec!["summary", "b"]);
    assert_eq!(
        frames[0].value.to_json(),
        committed(&harness, &fork, &recording.initial).await
    );
    recording.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn shows_a_forks_inherited_entries_and_follows_only_the_forks_own_commits() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let first = note(&root, "first").await;
    let fork = root
        .fork(first.id, ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    let recording = record(&fork).await;
    assert_eq!(kinds(&recording.initial), vec!["first"]);
    assert_eq!(
        serde_json::to_value(recording.initial.conversation.parent).unwrap(),
        json!({ "conversationId": root.id, "at": first.id })
    );
    note(&root, "parent").await;
    note(&fork, "child").await;
    drained().await;
    let all: Vec<Vec<String>> = recording
        .frames()
        .iter()
        .map(|frame| kinds(&frame.value))
        .collect();
    assert_eq!(all, vec![vec!["first", "child"]]);
    recording.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn unmounts_a_retired_document_and_mounts_its_recreation_whole() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let recording = record(&root).await;
    let id = root.id;
    root.commit(
        move |tx| async move { tx.retire_doc(&*LIVE_DOC, id).await },
        &context(),
    )
    .await
    .unwrap();
    root.commit(
        move |tx| async move {
            tx.doc(&*LIVE_DOC, id)
                .await?
                .edit(|live| live.tools = Some(Vec::new()))
        },
        &context(),
    )
    .await
    .unwrap();
    drained().await;
    let frames = recording.frames();
    let ops: Vec<Vec<JsonValue>> = frames.iter().map(|frame| frame.ops.clone()).collect();
    assert_eq!(
        ops,
        vec![
            vec![json!(["d", ["docs", "pi.live"]])],
            vec![json!(["s", ["docs", "pi.live"], { "tools": [] }])],
        ]
    );
    assert!(frames[0].value.doc("pi.live").is_none());
    recording.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn replaces_undelivered_frames_with_the_newest_view_after_100_pending_frames() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let watch = root.watch(&context()).await.unwrap();
    for _ in 0..101 {
        note(&root, "note").await;
    }
    let frames = start_recording(&watch);
    drained().await;
    let frames = frames.lock().clone();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].ops, vec![json!(["r", frames[0].value.to_json()])]);
    assert_eq!(frames[0].value.entries.len(), 101);
    watch.stop().await;
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn keeps_states_and_watches_of_one_conversation_independent_and_remounts_after_the_last_one_detaches()
 {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let state = root.view_state(&context()).await.unwrap();
    let recording = record(&root).await;
    note(&root, "one").await;
    drained().await;
    assert_eq!(kinds(&state.value()), vec!["one"]);
    recording.stop().await;
    assert_eq!(recording.frames().len(), 1);
    note(&root, "two").await;
    drained().await;
    assert_eq!(kinds(&state.value()), vec!["one", "two"]);
    let last = state.value();
    state.dispose().unwrap();
    // No observer is left, so the mount was dropped: a new observer builds a new revision from committed state.
    let rebuilt = fresh(&root).await;
    assert_eq!(rebuilt, last);
    assert!(!Arc::ptr_eq(&rebuilt.entries, &last.entries));
    harness.close(&context()).await.unwrap();
}

#[tokio::test]
async fn ends_states_and_watches_at_close_and_rejects_later_acquisition() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let watch = root.watch(&context()).await.unwrap();
    let state = root.view_state(&context()).await.unwrap();
    harness.close(&context()).await.unwrap();
    assert!(matches!(watch.closed().await, WatchEnd::SessionClosed));
    assert!(state.value().entries.is_empty());
    assert!(root.watch(&context()).await.is_err());
    assert!(root.view_state(&context()).await.is_err());
}

#[tokio::test]
async fn rejects_an_acquisition_cancelled_or_closed_while_it_waits_for_the_session_line() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let hold = |root: &Conversation| {
        let release = Deferred::default();
        let wait = release.clone();
        let root = root.clone();
        let blocking = tokio::spawn(async move {
            root.commit(
                move |_| async move {
                    wait.wait().await;
                    Ok(())
                },
                &context(),
            )
            .await
        });
        (release, blocking)
    };
    let (release, blocking) = hold(&root);
    tokio::task::yield_now().await;
    let controller = AbortController::new();
    let cancelled_root = root.clone();
    let signalled = signal_context(controller.signal());
    let cancelled = tokio::spawn(async move { cancelled_root.watch(&signalled).await });
    tokio::task::yield_now().await;
    controller.abort(Some(AbortReason::message("cancelled")));
    release.resolve();
    blocking.await.unwrap().unwrap();
    assert_err(cancelled.await.unwrap(), "cancelled");

    let (release, blocking) = hold(&root);
    tokio::task::yield_now().await;
    let queued_root = root.clone();
    let closed_while_queued = tokio::spawn(async move { queued_root.watch(&context()).await });
    tokio::task::yield_now().await;
    let closing = tokio::spawn(harness.close(&context()));
    tokio::task::yield_now().await;
    release.resolve();
    blocking.await.unwrap().unwrap();
    assert_err(closed_while_queued.await.unwrap(), "Harness is closed");
    closing.await.unwrap().unwrap();
}

#[tokio::test]
async fn shares_one_mount_between_concurrent_observers_and_isolates_a_failing_listener() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let failing = root.watch(&context()).await.unwrap();
    let recording = record(&root).await;
    let shared = root.view_state(&context()).await.unwrap();
    assert!(Arc::ptr_eq(
        &failing.value().entries,
        &shared.value().entries
    ));
    assert!(Arc::ptr_eq(&failing.value().docs, &shared.value().docs));
    shared.dispose().unwrap();
    failing
        .start(|_, _, _| {
            Box::pin(async { Err(crate::durable::errors::Error::message("listener failed")) })
        })
        .unwrap();
    note(&root, "one").await;
    note(&root, "two").await;
    drained().await;
    assert!(matches!(failing.closed().await, WatchEnd::ListenerError(_)));
    assert_eq!(recording.frames().len(), 2);
    recording.stop().await;
    harness.close(&context()).await.unwrap();
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
struct Other {
    n: u32,
}

#[tokio::test]
async fn publishes_one_frame_for_a_commit_that_appends_several_entries_and_edits_a_document() {
    let (harness, root) = open_chat(storage(), &chat_setup()).await;
    let other = define_doc(DocDefinition::new(
        "app.other",
        1,
        LatestConversation {
            fork: LatestFork::Initial,
        },
        Other::default,
    ))
    .unwrap();
    let recording = record(&root).await;
    let id = root.id;
    root.commit(
        move |tx| async move {
            tx.append_entry(id, EntryDraft::new("a")).await?;
            tx.doc(&*LIVE_DOC, id)
                .await?
                .edit(|live| live.tools = Some(Vec::new()))?;
            tx.append_entry(id, EntryDraft::new("b")).await?;
            Ok(())
        },
        &context(),
    )
    .await
    .unwrap();
    // A document that is not mounted publishes nothing.
    root.commit(
        move |tx| async move { tx.doc(&other, id).await?.edit(|value| value.n = 1) },
        &context(),
    )
    .await
    .unwrap();
    drained().await;
    let frames = recording.frames();
    assert_eq!(frames.len(), 1);
    assert_eq!(kinds(&frames[0].value), vec!["a", "b"]);
    assert_eq!(
        replay(&recording.initial, &frames),
        committed(&harness, &root, &recording.initial).await
    );
    recording.stop().await;
    harness.close(&context()).await.unwrap();
}
