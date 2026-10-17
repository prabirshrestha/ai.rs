//! Port of `test/session-watches.test.ts`.
//!
//! Divergences: "delivers replayable structural no-op commits" is skipped
//! (Chord's Rust change diffs at preparation, so a shift/unshift pair has no
//! ops and writes nothing); subtree identity (`retained`) is JS-only.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use futures::FutureExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::*;
use crate::chord::delta::{Op, apply_immutable};
use crate::chord::{
    AbortController, AbortReason, BACKGROUND_CONTEXT, Context, JsonValue, create_context_key,
    with_abort_signal, with_cancel, with_context_value,
};
use crate::durable::documents::define_doc;
use crate::durable::errors::{Error, Result};
use crate::durable::session::DocumentWatch;
use crate::durable::types::{DocDefinition, SessionScope, WatchEnd};
use crate::durable::{DocToken, SessionImpl};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Retained {
    label: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct State {
    value: i64,
    items: Vec<String>,
    retained: Retained,
}

fn state_doc() -> DocToken<State, SessionScope> {
    define_doc(DocDefinition::new("watch.state", 1, SessionScope, || {
        State {
            value: 0,
            items: vec!["a".into(), "b".into()],
            retained: Retained {
                label: "stable".into(),
            },
        }
    }))
    .unwrap()
}

type Value = Option<Arc<JsonValue>>;

fn number(value: &Value) -> i64 {
    value
        .as_ref()
        .and_then(|value| value["value"].as_i64())
        .unwrap_or(-1)
}

async fn create_state() -> TestSession {
    let test = open_test_session();
    commit(&test.session, |tx| async move {
        tx.doc(&state_doc(), ()).await.map(|_| ())
    })
    .await;
    test
}

async fn watch(session: &SessionImpl, context: &Context) -> DocumentWatch {
    session
        .watch_doc(&state_doc(), (), context)
        .await
        .unwrap()
        .expect("a watch")
}

async fn set_value(session: &SessionImpl, value: i64) {
    commit(session, move |tx| async move {
        tx.doc(&state_doc(), ())
            .await?
            .edit(|state| state.value = value)
    })
    .await;
}

async fn retire(session: &SessionImpl) {
    commit(session, |tx| async move {
        tx.retire_doc(&state_doc(), ()).await
    })
    .await;
}

fn done() -> BoxFuture<'static, Result<()>> {
    async { Ok(()) }.boxed()
}

#[tokio::test]
async fn never_creates_an_absent_document() {
    let test = open_test_session();
    let commits = test.storage.commit_count();
    assert!(
        test.session
            .watch_doc(&state_doc(), (), &context())
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(test.storage.commit_count(), commits);
    assert_eq!(test.storage.mints(), 0);
}

#[tokio::test]
async fn keeps_the_acquisition_revision_until_start_and_delivers_exact_committed_frames() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let initial = watch.value();
    set_value(session, 1).await;
    flush().await;
    let first = document_changes(&test.last_publication())[0].clone();
    set_value(session, 2).await;
    flush().await;
    let second = document_changes(&test.last_publication())[0].clone();
    assert!(Arc::ptr_eq(
        watch.value().as_ref().unwrap(),
        initial.as_ref().unwrap()
    ));

    let inline = Arc::new(AtomicBool::new(true));
    let deliveries: Arc<Mutex<Vec<(Value, Arc<[Op]>)>>> = Arc::default();
    let (seen_inline, sink, observed) = (inline.clone(), deliveries.clone(), watch.clone());
    watch
        .start(move |value, ops, _| {
            assert!(!seen_inline.load(Ordering::SeqCst));
            assert!(Arc::ptr_eq(
                observed.value().as_ref().unwrap(),
                value.as_ref().unwrap()
            ));
            sink.lock().push((value, ops));
            done()
        })
        .unwrap();
    inline.store(false, Ordering::SeqCst);
    assert!(deliveries.lock().is_empty());
    flush().await;
    let deliveries = deliveries.lock().clone();
    assert_eq!(
        deliveries
            .iter()
            .map(|(value, _)| number(value))
            .collect::<Vec<_>>(),
        [1, 2]
    );
    assert!(Arc::ptr_eq(
        deliveries[0].0.as_ref().unwrap(),
        first.value.as_ref().unwrap()
    ));
    assert!(Arc::ptr_eq(&deliveries[0].1, &first.ops));
    assert!(Arc::ptr_eq(
        deliveries[1].0.as_ref().unwrap(),
        second.value.as_ref().unwrap()
    ));
    assert!(Arc::ptr_eq(&deliveries[1].1, &second.ops));
    assert_eq!(number(&initial), 0);
    watch.stop().await;
}

#[tokio::test]
async fn serializes_callbacks_and_buffers_exact_frames_committed_while_one_is_in_flight() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let entered = Deferred::default();
    let release = Deferred::default();
    let values: Arc<Mutex<Vec<i64>>> = Arc::default();
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let (entered_signal, release_signal, sink, running, peak) = (
        entered.clone(),
        release.clone(),
        values.clone(),
        active.clone(),
        max_active.clone(),
    );
    watch
        .start(move |value, _, _| {
            let (entered, release, sink, running, peak) = (
                entered_signal.clone(),
                release_signal.clone(),
                sink.clone(),
                running.clone(),
                peak.clone(),
            );
            async move {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                let count = {
                    let mut values = sink.lock();
                    values.push(number(&value));
                    values.len()
                };
                if count == 1 {
                    entered.resolve();
                    release.wait().await;
                }
                running.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }
            .boxed()
        })
        .unwrap();
    set_value(session, 1).await;
    entered.wait().await;
    for value in 2..=20 {
        set_value(session, value).await;
    }
    assert_eq!(*values.lock(), [1]);
    release.resolve();
    flush().await;
    assert_eq!(max_active.load(Ordering::SeqCst), 1);
    assert_eq!(*values.lock(), (1..=20).collect::<Vec<_>>());
    watch.stop().await;
}

#[tokio::test]
async fn allows_a_listener_to_initiate_a_later_session_commit() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let completed = Deferred::default();
    let values: Arc<Mutex<Vec<i64>>> = Arc::default();
    let (session_handle, sink, finished) = (session.clone(), values.clone(), completed.clone());
    watch
        .start(move |value, _, _| {
            let (session, sink, finished) =
                (session_handle.clone(), sink.clone(), finished.clone());
            async move {
                let current = number(&value);
                sink.lock().push(current);
                if current == 1 {
                    set_value(&session, 2).await;
                } else {
                    finished.resolve();
                }
                Ok(())
            }
            .boxed()
        })
        .unwrap();
    set_value(session, 1).await;
    completed.wait().await;
    assert_eq!(*values.lock(), [1, 2]);
    watch.stop().await;
}

#[tokio::test]
async fn collapses_101_pending_commits_to_one_root_replacement() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    for value in 1..=101 {
        set_value(session, value).await;
    }
    let deliveries: Arc<Mutex<Vec<(i64, JsonValue)>>> = Arc::default();
    let sink = deliveries.clone();
    watch
        .start(move |value, ops, _| {
            sink.lock().push((number(&value), to_json(&ops.to_vec())));
            done()
        })
        .unwrap();
    flush().await;
    let current = watch.value().unwrap();
    assert_eq!(*deliveries.lock(), [(101, json!([["r", *current]]))]);
    watch.stop().await;
}

#[tokio::test]
async fn never_folds_the_in_flight_frame_into_an_overflow_reset() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let entered = Deferred::default();
    let release = Deferred::default();
    let deliveries: Arc<Mutex<Vec<(i64, JsonValue)>>> = Arc::default();
    let (entered_signal, release_signal, sink) =
        (entered.clone(), release.clone(), deliveries.clone());
    watch
        .start(move |value, ops, _| {
            let (entered, release, sink) =
                (entered_signal.clone(), release_signal.clone(), sink.clone());
            async move {
                let count = {
                    let mut deliveries = sink.lock();
                    deliveries.push((number(&value), to_json(&ops.to_vec())));
                    deliveries.len()
                };
                if count == 1 {
                    entered.resolve();
                    release.wait().await;
                }
                Ok(())
            }
            .boxed()
        })
        .unwrap();
    set_value(session, 1).await;
    entered.wait().await;
    for value in 2..=102 {
        set_value(session, value).await;
    }
    release.resolve();
    flush().await;
    let deliveries = deliveries.lock().clone();
    assert_eq!(deliveries[0].0, 1);
    let current = watch.value().unwrap();
    assert_eq!(deliveries[1], (102, json!([["r", *current]])));
    assert_eq!(deliveries.len(), 2);
    watch.stop().await;
}

#[tokio::test]
async fn folds_retirement_into_an_overflow_reset_and_then_closes() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    for value in 1..=100 {
        set_value(session, value).await;
    }
    retire(session).await;
    let deliveries: Arc<Mutex<Vec<(Value, JsonValue)>>> = Arc::default();
    let sink = deliveries.clone();
    watch
        .start(move |value, ops, _| {
            sink.lock().push((value, to_json(&ops.to_vec())));
            done()
        })
        .unwrap();
    assert_eq!(watch.closed().await.reason(), "retired");
    assert_eq!(*deliveries.lock(), [(None, json!([["r", null]]))]);
}

#[tokio::test]
async fn preserves_commit_context_values_without_inheriting_producer_cancellation() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let key = create_context_key::<String>("watch-test");
    let parent = AbortController::new();
    let commit_context = with_context_value(
        &key,
        "newest-commit".to_string(),
        &with_abort_signal(parent.signal(), &BACKGROUND_CONTEXT),
    );
    let entered = Deferred::default();
    let release = Deferred::default();
    let delivered: Arc<Mutex<Option<Context>>> = Arc::default();
    let (entered_signal, release_signal, sink) =
        (entered.clone(), release.clone(), delivered.clone());
    watch
        .start(move |_, _, context| {
            *sink.lock() = Some(context);
            let (entered, release) = (entered_signal.clone(), release_signal.clone());
            async move {
                entered.resolve();
                release.wait().await;
                Ok(())
            }
            .boxed()
        })
        .unwrap();
    session
        .commit(
            |tx| async move {
                tx.doc(&state_doc(), ())
                    .await?
                    .edit(|state| state.value = 1)
            },
            &commit_context,
        )
        .await
        .unwrap();
    entered.wait().await;
    let delivered = delivered.lock().clone().unwrap();
    assert_eq!(
        delivered.value(&key).map(String::as_str),
        Some("newest-commit")
    );
    assert!(delivered.abort_signal().is_none());
    parent.abort(Some(AbortReason::message("caller finished")));
    assert_eq!(watch.stop().await.reason(), "stopped");
    assert!(delivered.abort_signal().is_none());
    release.resolve();
}

#[tokio::test]
async fn keeps_earlier_immutable_revisions_stable() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let initial = watch.value();
    let delivered: Arc<Mutex<Option<Value>>> = Arc::default();
    let sink = delivered.clone();
    watch
        .start(move |value, _, _| {
            *sink.lock() = Some(value);
            done()
        })
        .unwrap();
    set_value(session, 1).await;
    flush().await;
    let delivered = delivered.lock().clone().unwrap();
    assert!(Arc::ptr_eq(
        delivered.as_ref().unwrap(),
        watch.value().as_ref().unwrap()
    ));
    assert!(!Arc::ptr_eq(
        delivered.as_ref().unwrap(),
        initial.as_ref().unwrap()
    ));
    assert_eq!(number(&initial), 0);
    watch.stop().await;
}

#[tokio::test]
async fn delivers_retirement_and_does_not_follow_recreation() {
    let test = create_state().await;
    let session = &test.session;
    let old_watch = watch(session, &context()).await;
    let values: Arc<Mutex<Vec<Value>>> = Arc::default();
    let sink = values.clone();
    old_watch
        .start(move |value, _, _| {
            sink.lock().push(value);
            done()
        })
        .unwrap();
    commit(session, |tx| async move {
        tx.retire_doc(&state_doc(), ()).await?;
        tx.doc(&state_doc(), ())
            .await?
            .edit(|state| state.value = 10)
    })
    .await;
    assert_eq!(old_watch.closed().await.reason(), "retired");
    assert_eq!(*values.lock(), [None]);
    assert!(old_watch.value().is_none());

    let replacement = watch(session, &context()).await;
    assert_eq!(number(&replacement.value()), 10);
    set_value(session, 11).await;
    flush().await;
    assert!(old_watch.value().is_none());
    replacement.stop().await;
}

#[tokio::test]
async fn session_close_discards_retirement_buffered_before_start() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let baseline = watch.value();
    retire(session).await;
    session.close(&context()).await.unwrap();
    assert_eq!(watch.closed().await.reason(), "session_closed");
    assert!(Arc::ptr_eq(
        watch.value().as_ref().unwrap(),
        baseline.as_ref().unwrap()
    ));
    assert_err(watch.start(|_, _, _| done()), "stopped");
}

#[tokio::test]
async fn session_close_discards_retirement_behind_an_in_flight_callback() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let entered = Deferred::default();
    let release = Deferred::default();
    let values: Arc<Mutex<Vec<Option<i64>>>> = Arc::default();
    let (entered_signal, release_signal, sink) = (entered.clone(), release.clone(), values.clone());
    watch
        .start(move |value, _, _| {
            sink.lock()
                .push(value.as_ref().and_then(|value| value["value"].as_i64()));
            let (entered, release) = (entered_signal.clone(), release_signal.clone());
            async move {
                entered.resolve();
                release.wait().await;
                Ok(())
            }
            .boxed()
        })
        .unwrap();
    set_value(session, 1).await;
    entered.wait().await;
    retire(session).await;
    session.close(&context()).await.unwrap();
    assert_eq!(watch.closed().await.reason(), "session_closed");
    release.resolve();
    flush().await;
    assert_eq!(*values.lock(), [Some(1)]);
}

#[tokio::test]
async fn supports_idempotent_stop_and_rejects_repeated_or_late_start() {
    let test = create_state().await;
    let session = &test.session;
    let started = watch(session, &context()).await;
    started.start(|_, _, _| done()).unwrap();
    assert_err(started.start(|_, _, _| done()), "already started");
    let first = started.stop();
    let second = started.stop();
    assert!(first.ptr_eq(&second));
    assert_eq!(first.await.reason(), "stopped");

    let stopped = watch(session, &context()).await;
    stopped.stop().await;
    assert_err(stopped.start(|_, _, _| done()), "stopped");
}

#[tokio::test]
async fn cancels_acquisition_without_leaking_a_registered_watch() {
    let test = create_state().await;
    let session = &test.session;
    session.unload_documents().await;
    let gate = test.storage.hold_find_document();
    let child = with_cancel(&context());
    let acquisition = tokio::spawn(session.watch_doc(&state_doc(), (), &child.context));
    gate.entered().await;
    child.cancel(Some(AbortReason::message("cancel acquisition")));
    gate.release();
    assert_err(acquisition.await.unwrap(), "cancel acquisition");
    session.close(&context()).await.unwrap();
}

#[tokio::test]
async fn cancels_future_delivery_without_aborting_an_in_flight_callback() {
    let test = create_state().await;
    let session = &test.session;
    let child = with_cancel(&context());
    let watch = watch(session, &child.context).await;
    let entered = Deferred::default();
    let release = Deferred::default();
    let delivered: Arc<Mutex<Option<Context>>> = Arc::default();
    let (entered_signal, release_signal, sink) =
        (entered.clone(), release.clone(), delivered.clone());
    watch
        .start(move |_, _, context| {
            *sink.lock() = Some(context);
            let (entered, release) = (entered_signal.clone(), release_signal.clone());
            async move {
                entered.resolve();
                release.wait().await;
                Ok(())
            }
            .boxed()
        })
        .unwrap();
    set_value(session, 1).await;
    entered.wait().await;
    child.cancel(None);
    assert_eq!(watch.closed().await.reason(), "cancelled");
    assert!(delivered.lock().as_ref().unwrap().abort_signal().is_none());
    release.resolve();
}

#[tokio::test]
async fn session_close_stops_future_delivery_without_joining_an_in_flight_callback() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let entered = Deferred::default();
    let release = Deferred::default();
    let delivered: Arc<Mutex<Option<Context>>> = Arc::default();
    let (entered_signal, release_signal, sink) =
        (entered.clone(), release.clone(), delivered.clone());
    watch
        .start(move |_, _, context| {
            *sink.lock() = Some(context);
            let (entered, release) = (entered_signal.clone(), release_signal.clone());
            async move {
                entered.resolve();
                release.wait().await;
                Ok(())
            }
            .boxed()
        })
        .unwrap();
    set_value(session, 1).await;
    entered.wait().await;
    session.close(&context()).await.unwrap();
    assert!(delivered.lock().as_ref().unwrap().abort_signal().is_none());
    assert_eq!(watch.closed().await.reason(), "session_closed");
    release.resolve();
}

#[tokio::test]
async fn settles_listener_failure_on_only_the_affected_watch() {
    let test = create_state().await;
    let session = &test.session;
    let failed = watch(session, &context()).await;
    let healthy = watch(session, &context()).await;
    failed
        .start(|_, _, _| async { Err(Error::message("listener failed")) }.boxed())
        .unwrap();
    let healthy_calls = Arc::new(AtomicUsize::new(0));
    let calls = healthy_calls.clone();
    healthy
        .start(move |_, _, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            done()
        })
        .unwrap();
    set_value(session, 1).await;
    let end = failed.closed().await;
    assert_eq!(end.reason(), "listener_error");
    let WatchEnd::ListenerError(error) = end else {
        unreachable!()
    };
    assert_eq!(error.to_string(), "listener failed");
    flush().await;
    assert_eq!(healthy_calls.load(Ordering::SeqCst), 1);
    healthy.stop().await;
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Plain {
    value: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Migrated {
    value: i64,
    migrated: bool,
}

#[tokio::test]
async fn hydrates_migration_without_writing_and_observes_the_later_exact_edit() {
    let old: DocToken<Plain, SessionScope> = define_doc(DocDefinition::new(
        "watch.migration",
        1,
        SessionScope,
        || Plain { value: 3 },
    ))
    .unwrap();
    let current: DocToken<Migrated, SessionScope> = define_doc(
        DocDefinition::new("watch.migration", 2, SessionScope, Migrated::default).migrate(
            |value, _| {
                Ok(Migrated {
                    value: value["value"].as_i64().unwrap(),
                    migrated: true,
                })
            },
        ),
    )
    .unwrap();
    let test = open_test_session();
    let session = &test.session;
    let token = old.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, ()).await.map(|_| ())
    })
    .await;
    session.unload_documents().await;
    let commits = test.storage.commit_count();
    let watch = session
        .watch_doc(&current, (), &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        *watch.value().unwrap(),
        json!({ "value": 3, "migrated": true })
    );
    assert_eq!(test.storage.commit_count(), commits);
    let values: Arc<Mutex<Vec<i64>>> = Arc::default();
    let sink = values.clone();
    watch
        .start(move |value, _, _| {
            sink.lock().push(number(&value));
            done()
        })
        .unwrap();
    let token = current.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, ()).await.map(|_| ())
    })
    .await;
    flush().await;
    assert!(values.lock().is_empty());
    let token = current.clone();
    commit(session, move |tx| async move {
        tx.doc(&token, ()).await?.edit(|value| value.value = 4)
    })
    .await;
    flush().await;
    assert_eq!(*values.lock(), [4]);
    watch.stop().await;
}

#[tokio::test]
async fn can_replay_every_delivered_exact_operation_batch_from_the_acquisition_revision() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let replica = Arc::new(Mutex::new((*watch.value().unwrap()).clone()));
    let state = replica.clone();
    watch
        .start(move |value, ops, _| {
            let mut replica = state.lock();
            *replica = apply_immutable(&replica, &ops).unwrap();
            assert_eq!(&*replica, value.as_deref().unwrap());
            done()
        })
        .unwrap();
    commit(session, |tx| async move {
        tx.doc(&state_doc(), ()).await?.edit(|state| {
            state.value = 3;
            state.items.remove(0);
        })
    })
    .await;
    flush().await;
    assert_eq!(*replica.lock(), *watch.value().unwrap());
    watch.stop().await;
}

#[tokio::test]
async fn continues_after_the_tracker_cache_unloads() {
    let test = create_state().await;
    let session = &test.session;
    let watch = watch(session, &context()).await;
    let baseline = watch.value().unwrap();
    let reads = test.storage.document_reads();
    session.unload_documents().await;
    let reloaded = session
        .snapshot_json(&state_doc(), (), &context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reloaded, baseline);
    assert!(!Arc::ptr_eq(&reloaded, &baseline));
    assert!(test.storage.document_reads() > reads);
    watch.start(|_, _, _| done()).unwrap();
    set_value(session, 7).await;
    flush().await;
    assert_eq!(number(&watch.value()), 7);
    watch.stop().await;
}
