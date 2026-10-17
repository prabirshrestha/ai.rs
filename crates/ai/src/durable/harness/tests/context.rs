//! Port of `test/harness-context.test.ts`.

use std::sync::Arc;

use serde_json::json;

use super::support::*;
use crate::durable::harness::types::ConversationCreateOptions;
use crate::durable::harness::{Conversation, CreateOptions, Harness};
use crate::durable::ids::EntryId;
use crate::durable::storage::memory::MemoryStorage;
use crate::durable::types::{ContextEdit, ContextEditAction, EntryDraft, EntryHead, EntryRecord};
use crate::types::{Message, StopReason};

struct Setup {
    _harness: Harness,
    root: Conversation,
}

impl Setup {
    async fn append(&self, draft: EntryDraft) -> EntryRecord {
        let id = self.root.id;
        self.root
            .commit(
                move |tx| async move { tx.append_entry(id, draft).await },
                &context(),
            )
            .await
            .unwrap()
    }

    async fn message(&self, model: Message) -> EntryRecord {
        self.message_of(model, "message").await
    }

    async fn message_of(&self, model: Message, kind: &str) -> EntryRecord {
        let mut draft = EntryDraft::new(kind);
        draft.model = Some(vec![model]);
        self.append(draft).await
    }
}

async fn setup() -> Setup {
    let (harness, _) = open_harness(Arc::new(MemoryStorage::new()), &[], None, None).await;
    let root = harness
        .root(&context(), CreateOptions::default())
        .await
        .unwrap();
    Setup {
        _harness: harness,
        root,
    }
}

fn ids(entries: &[EntryRecord]) -> Vec<EntryId> {
    entries.iter().map(|entry| entry.id).collect()
}

fn described(messages: &[Message]) -> Vec<String> {
    messages.iter().map(describe_message).collect()
}

fn edit(target: EntryId, messages: Option<Vec<Message>>) -> EntryDraft {
    let mut draft = EntryDraft::new("edit");
    draft.edits = Some(vec![ContextEdit {
        target,
        action: match messages {
            Some(messages) => ContextEditAction::Replace { messages },
            None => ContextEditAction::Omit,
        },
    }]);
    draft
}

#[tokio::test]
async fn returns_the_whole_transcript_without_a_head_and_excludes_model_less_entries_from_messages()
{
    let s = setup().await;
    let first = s.message(user("hi")).await;
    let note = s
        .append(EntryDraft::new("note").data(json!({ "text": "display only" })))
        .await;
    let answer = s.message(assistant("hello")).await;
    let view = s.root.context(&context()).await.unwrap();
    assert!(view.head.is_none());
    assert_eq!(ids(&view.entries), [first.id, note.id, answer.id]);
    assert_eq!(described(&view.messages), ["user:hi", "assistant:hello"]);
}

#[tokio::test]
async fn excludes_aborted_error_and_deferred_assistant_messages_but_keeps_their_raw_entries() {
    let s = setup().await;
    s.message(user("q")).await;
    let aborted = s
        .message(assistant_stopped("partial", StopReason::Aborted))
        .await;
    s.message(assistant_stopped("failed", StopReason::Error))
        .await;
    s.message(assistant_stopped("later", StopReason::Deferred))
        .await;
    s.message(assistant_stopped("done", StopReason::Length))
        .await;
    let view = s.root.context(&context()).await.unwrap();
    assert_eq!(view.entries.len(), 5);
    assert_eq!(view.entries[1].id, aborted.id);
    assert_eq!(described(&view.messages), ["user:q", "assistant:done"]);
}

#[tokio::test]
async fn resolves_self_heads_and_uses_the_newest_head_marker() {
    let s = setup().await;
    s.message(user("old")).await;
    let mut reset = EntryDraft::new("reset").head(EntryHead::SelfEntry);
    reset.model = Some(vec![user("fresh start")]);
    let reset = s.append(reset).await;
    assert_eq!(reset.head, Some(reset.id));
    let after = s.message(assistant("after reset")).await;
    let view = s.root.context(&context()).await.unwrap();
    assert_eq!(view.head.as_ref().map(|head| head.id), Some(reset.id));
    assert_eq!(ids(&view.entries), [reset.id, after.id]);
    assert_eq!(
        described(&view.messages),
        ["user:fresh start", "assistant:after reset"]
    );

    // A compaction summary heads an earlier kept entry; older head markers in range drop out.
    let mut summary = EntryDraft::new("summary").head(EntryHead::Id(after.id));
    summary.model = Some(vec![user("summary")]);
    let summary = s.append(summary).await;
    let tail = s.message(user("next")).await;
    let view = s.root.context(&context()).await.unwrap();
    assert_eq!(view.head.as_ref().map(|head| head.id), Some(summary.id));
    assert_eq!(ids(&view.entries), [summary.id, after.id, tail.id]);
    assert_eq!(
        described(&view.messages),
        ["user:summary", "assistant:after reset", "user:next"]
    );
}

#[tokio::test]
async fn applies_the_newest_edit_per_target_within_the_active_range() {
    let s = setup().await;
    let first = s.message(user("first")).await;
    let second = s.message(user("second")).await;
    s.append(edit(first.id, Some(vec![user("first v2")]))).await;
    s.append(edit(first.id, Some(vec![user("first v3")]))).await;
    s.append(edit(second.id, None)).await;
    let view = s.root.context(&context()).await.unwrap();
    assert_eq!(view.entries.len(), 5);
    assert_eq!(described(&view.messages), ["user:first v3"]);

    // Edits before the active range no longer apply.
    let reset = s
        .append(EntryDraft::new("reset").head(EntryHead::Id(second.id)))
        .await;
    let view = s.root.context(&context()).await.unwrap();
    assert_eq!(view.head.as_ref().map(|head| head.id), Some(reset.id));
    assert!(view.messages.is_empty());
    s.append(edit(second.id, Some(vec![user("second v2")])))
        .await;
    let view = s.root.context(&context()).await.unwrap();
    assert_eq!(described(&view.messages), ["user:second v2"]);
}

#[tokio::test]
async fn keeps_positional_system_messages_and_orders_tool_results_by_call_order() {
    let s = setup().await;
    s.message_of(system(&[("preamble", Some("You help."))]), "pi.system")
        .await;
    s.message(user("run tools")).await;
    s.message(assistant_calls("calling", &["b", "a"])).await;
    s.message(tool_result("a")).await;
    s.message_of(system(&[("cwd", Some("/repo"))]), "pi.system")
        .await;
    s.message(tool_result("b")).await;
    s.message(tool_result("zz")).await;
    s.message(assistant("done")).await;
    let view = s.root.context(&context()).await.unwrap();
    assert_eq!(
        described(&view.messages),
        [
            "system:preamble",
            "user:run tools",
            "assistant:calling",
            "result:b:result b",
            "result:a:result a",
            "system:cwd",
            "assistant:done",
        ]
    );
}

#[tokio::test]
async fn synthesizes_missing_tool_results_after_a_fork_and_drops_results_cut_from_their_call() {
    let s = setup().await;
    s.message(user("go")).await;
    let call = s.message(assistant_calls("calling", &["x", "y"])).await;
    s.message(tool_result("x")).await;
    let second = s.message(tool_result("y")).await;
    let child = s
        .root
        .fork(call.id, ConversationCreateOptions::ownerless(), &context())
        .await
        .unwrap();
    let child_view = child.context(&context()).await.unwrap();
    assert_eq!(
        described(&child_view.messages),
        [
            "user:go",
            "assistant:calling",
            "result:x:error",
            "result:y:error"
        ]
    );
    let missing = to_json(&child_view.messages[2]);
    assert_eq!(missing["role"], json!("toolResult"));
    assert_eq!(missing["toolName"], json!("tool-x"));
    assert_eq!(missing["details"], json!({ "reason": "missing_result" }));

    // A head between a call and its results leaves stray results that are not sent.
    s.append(EntryDraft::new("reset").head(EntryHead::Id(second.id)))
        .await;
    let parent_view = s.root.context(&context()).await.unwrap();
    assert!(parent_view.messages.is_empty());
    assert_eq!(
        parent_view
            .entries
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>(),
        ["reset", "message"]
    );
}
