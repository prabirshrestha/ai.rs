//! Ports of chord `test/state.test.ts` and `test/state-delivery.test.ts`
//! (mutable, attached and replica kinds). Skipped: Proxy draft escapes,
//! freezing/identity of nested containers, and `queueMicrotask` spying.

use super::*;
use crate::chord::context::with_cancel;
use crate::chord::delta::path;
use serde_json::json;
use std::time::Duration;

type Shared<T> = Arc<Mutex<T>>;

fn shared<T>(value: T) -> Shared<T> {
    Arc::new(Mutex::new(value))
}

fn bg() -> Context {
    BACKGROUND_CONTEXT.clone()
}

fn set_value(value: i64) -> Arc<[Op]> {
    Arc::from(vec![Op::S(path!["value"], json!(value))])
}

#[derive(Debug)]
struct Failure(String);

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Failure {}

fn failure(message: &str) -> SharedError {
    Arc::new(Failure(message.to_string()))
}

async fn wait_for(mut condition: impl FnMut() -> bool) {
    for _ in 0..500 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(condition(), "condition not reached");
}

// ─── state.test.ts: transactional replicated state ───────────────────────────

#[test]
fn publishes_one_immutable_revision() {
    let state =
        mutable_replicated_state(json!({ "changed": { "value": 1 }, "retained": { "value": 2 } }));
    let previous = state.value();
    let deliveries = shared(Vec::new());
    let sink = deliveries.clone();
    state.subscribe(move |_, _, delivery| {
        sink.lock().push(delivery.sequence);
        ListenerOutcome::ok()
    });
    state
        .change(&bg(), |draft| {
            draft["changed"]["value"] = json!(3);
            draft["changed"]["value"] = json!(4);
        })
        .unwrap();
    assert_eq!(
        *state.value(),
        json!({ "changed": { "value": 4 }, "retained": { "value": 2 } })
    );
    assert!(!Arc::ptr_eq(&state.value(), &previous));
    assert_eq!(*deliveries.lock(), vec![0, 1]);
}

#[test]
fn rolls_back_callback_failures() {
    let state = mutable_replicated_state(json!({ "nested": { "value": 1 } }));
    let previous = state.value();
    let error = state
        .try_change(&bg(), |draft| {
            draft["nested"]["value"] = json!(2);
            Err(failure("stop"))
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "stop");
    assert!(Arc::ptr_eq(&state.value(), &previous));
}

#[test]
fn rejects_nested_changes_and_replacements_without_losing_the_outer_rollback() {
    let state = mutable_replicated_state(json!({ "left": 0, "right": 0 }));
    let error = state
        .try_change(&bg(), |draft| {
            draft["left"] = json!(1);
            state
                .change(&bg(), |nested| nested["right"] = json!(2))
                .map_err(|error| Arc::new(error) as SharedError)
        })
        .unwrap_err();
    assert!(error.to_string().contains("reentrantly"));
    assert_eq!(*state.value(), json!({ "left": 0, "right": 0 }));
    let error = state
        .try_change(&bg(), |_| {
            state
                .replace(&bg(), json!({ "left": 1, "right": 2 }))
                .map_err(|error| Arc::new(error) as SharedError)
        })
        .unwrap_err();
    assert!(error.to_string().contains("change callback"));
    assert_eq!(*state.value(), json!({ "left": 0, "right": 0 }));
}

#[test]
fn queues_listener_triggered_changes_in_sequence_order() {
    let state = Arc::new(mutable_replicated_state(json!({ "value": 0 })));
    let source_sequences = shared(Vec::new());
    let deliveries = shared(Vec::new());
    let late = shared(Vec::new());
    let nested = Arc::new(AtomicBool::new(false));
    {
        let state_ref = Arc::downgrade(&state);
        let late = late.clone();
        state.internals().subscribe(move |_, _, _| {
            if !nested.swap(true, Ordering::SeqCst) {
                let state = state_ref.upgrade().unwrap();
                state
                    .change(&bg(), |draft| draft["value"] = json!(2))
                    .unwrap();
                let late = late.clone();
                state.subscribe(move |_, _, delivery| {
                    late.lock().push(delivery);
                    ListenerOutcome::ok()
                });
            }
            Ok(())
        });
    }
    let sink = source_sequences.clone();
    state.internals().subscribe(move |_, sequence, _| {
        sink.lock().push(sequence);
        Ok(())
    });
    let sink = deliveries.clone();
    state.subscribe(move |value, _, delivery| {
        if delivery.kind == DeliveryKind::Update {
            sink.lock()
                .push((value["value"].as_i64().unwrap(), delivery.sequence));
        }
        ListenerOutcome::ok()
    });
    state
        .change(&bg(), |draft| draft["value"] = json!(1))
        .unwrap();
    assert_eq!(*source_sequences.lock(), vec![1, 2]);
    assert_eq!(*deliveries.lock(), vec![(1, 1), (2, 2)]);
    assert_eq!(*late.lock(), vec![ReplicatedStateDelivery::hydrate(2)]);
}

#[test]
fn isolates_listener_failures_after_committing_the_revision() {
    let state = mutable_replicated_state(json!({ "value": 0 }));
    let received = shared(Vec::new());
    state
        .internals()
        .subscribe(|_, _, _| Err(failure("listener failed")));
    let sink = received.clone();
    state.internals().subscribe(move |_, sequence, _| {
        sink.lock().push(sequence);
        Ok(())
    });
    let error = state
        .change(&bg(), |draft| draft["value"] = json!(1))
        .unwrap_err();
    assert_eq!(error.to_string(), "listener failed");
    assert_eq!(*state.value(), json!({ "value": 1 }));
    assert_eq!(*received.lock(), vec![1]);
}

#[test]
fn copies_assigned_values_and_takes_ownership_of_replacements() {
    let state = mutable_replicated_state(json!({ "left": null, "right": null }));
    let external = json!({ "value": 1 });
    state
        .change(&bg(), |draft| {
            draft["left"] = external.clone();
            draft["right"] = external.clone();
            draft["left"]["value"] = json!(2);
        })
        .unwrap();
    assert_eq!(external, json!({ "value": 1 }));
    assert_eq!(
        *state.value(),
        json!({ "left": { "value": 2 }, "right": { "value": 1 } })
    );

    let replacement = Arc::new(json!({ "left": { "value": 2 }, "right": { "value": 2 } }));
    state.replace(&bg(), replacement.clone()).unwrap();
    assert!(Arc::ptr_eq(&state.value(), &replacement));
    state
        .change(&bg(), |draft| draft["left"]["value"] = json!(9))
        .unwrap();
    assert_eq!(
        *state.value(),
        json!({ "left": { "value": 9 }, "right": { "value": 2 } })
    );
}

#[test]
fn preserves_compact_string_splice_and_permutation_operations() {
    let state = mutable_replicated_state(
        json!({ "text": "abcdefgh", "values": [{ "id": "a" }, { "id": "b" }, { "id": "c" }] }),
    );
    let batches = shared(Vec::new());
    let sink = batches.clone();
    state.internals().subscribe(move |ops, _, _| {
        sink.lock().push(serde_json::to_value(&ops[..]).unwrap());
        Ok(())
    });
    state
        .change(&bg(), |draft| {
            draft["text"] = json!("defghxyz");
            draft["values"].as_array_mut().unwrap().remove(0);
        })
        .unwrap();
    state
        .change(&bg(), |draft| {
            draft["values"].as_array_mut().unwrap().reverse()
        })
        .unwrap();
    assert_eq!(
        *batches.lock(),
        vec![
            json!([
                ["t", ["text"], 3],
                ["a", ["text"], "xyz"],
                ["p", ["values"], 0, 1, []]
            ]),
            json!([["m", ["values"], [1, 0]]]),
        ]
    );
}

#[test]
fn replica_hydrates_and_applies_updates() {
    let replica = ReplicatedStateReplica::new(|_| {});
    replica
        .hydrate(0, &[Op::R(json!({ "rows": [{ "value": 1 }] }))], &bg())
        .unwrap();
    replica
        .update(
            1,
            &[Op::P(path!["rows"], 1, 0, vec![json!({ "value": 2 })])],
            &bg(),
        )
        .unwrap();
    assert_eq!(
        *replica.value().unwrap(),
        json!({ "rows": [{ "value": 1 }, { "value": 2 }] })
    );
}

#[test]
fn clears_a_replica_when_an_update_is_invalid() {
    let errors = shared(Vec::<String>::new());
    let sink = errors.clone();
    let replica = ReplicatedStateReplica::new(move |error| sink.lock().push(error.to_string()));
    replica
        .hydrate(0, &[Op::R(json!({ "values": [1, 2] }))], &bg())
        .unwrap();
    assert!(
        replica
            .update(1, &[Op::M(path!["values"], vec![0])], &bg())
            .is_err()
    );
    assert!(replica.value().is_none());
    let error = replica
        .update(2, &[Op::S(path!["values"], json!(2))], &bg())
        .unwrap_err();
    assert!(error.to_string().contains("before hydration"));
    assert!(
        replica
            .hydrate(3, &[Op::S(path!["values"], json!(2))], &bg())
            .is_err()
    );
    replica
        .hydrate(5, &[Op::R(json!({ "value": 0 }))], &bg())
        .unwrap();
    let error = replica.update(7, &set_value(7), &bg()).unwrap_err();
    assert!(error.to_string().contains("sequence has a gap"));
    assert!(errors.lock().is_empty());
}

#[test]
fn replaces_atomically_and_ignores_deeply_equal_replacements() {
    let state =
        mutable_replicated_state(json!({ "value": { "nested": 1 }, "retained": { "nested": 2 } }));
    let previous = state.value();
    let published = shared(0);
    let sink = published.clone();
    state.internals().subscribe(move |_, _, _| {
        *sink.lock() += 1;
        Ok(())
    });
    state
        .replace(
            &bg(),
            json!({ "value": { "nested": 1 }, "retained": { "nested": 2 } }),
        )
        .unwrap();
    assert!(Arc::ptr_eq(&state.value(), &previous));
    let mut next = (*state.value()).clone();
    next["value"] = json!({ "nested": 2 });
    state.replace(&bg(), next).unwrap();
    assert_eq!(
        *state.value(),
        json!({ "value": { "nested": 2 }, "retained": { "nested": 2 } })
    );
    assert_eq!(*published.lock(), 1);
}

// ─── state.test.ts: authoritative sources ────────────────────────────────────

struct SourceState {
    value: Arc<Value>,
    cursor: i64,
    attachments: Vec<(u64, Arc<TestSourceAttachment>)>,
    next_id: u64,
}

#[derive(Clone)]
struct TestSource {
    state: Shared<SourceState>,
    on_attach: Shared<Option<Box<dyn Fn() + Send>>>,
}

struct AttachmentState {
    buffer: Vec<ReplicatedStateSourceFrame<Arc<Value>>>,
    listener: Option<Arc<FrameListener<Arc<Value>>>>,
    activated: bool,
    disposed: bool,
}

struct TestSourceAttachment {
    snapshot: ReplicatedStateSnapshot<Arc<Value>>,
    state: Mutex<AttachmentState>,
    on_dispose: Box<dyn Fn() + Send + Sync>,
}

impl TestSource {
    fn new(value: Value, cursor: i64) -> Self {
        Self {
            state: shared(SourceState {
                value: Arc::new(value),
                cursor,
                attachments: Vec::new(),
                next_id: 0,
            }),
            on_attach: shared(None),
        }
    }

    fn attachments(&self) -> usize {
        self.state.lock().attachments.len()
    }

    fn commit_at(&self, value: Value, ops: Arc<[Op]>, cursor: i64) {
        let (frame, attachments) = {
            let mut state = self.state.lock();
            state.value = Arc::new(value);
            state.cursor = cursor;
            let frame = ReplicatedStateSourceFrame {
                cursor,
                value: state.value.clone(),
                ops,
                context: bg(),
            };
            let attachments: Vec<_> = state
                .attachments
                .iter()
                .map(|(_, attachment)| attachment.clone())
                .collect();
            (frame, attachments)
        };
        for attachment in attachments {
            attachment.publish(frame.clone());
        }
    }

    fn commit(&self, value: Value, ops: Arc<[Op]>) {
        let cursor = self.state.lock().cursor + 1;
        self.commit_at(value, ops, cursor);
    }
}

impl ReplicatedStateSource<Arc<Value>> for TestSource {
    fn attach(&self) -> Result<Box<dyn ReplicatedStateSourceAttachment<Arc<Value>>>, StateError> {
        let attachment = {
            let mut state = self.state.lock();
            let id = state.next_id;
            state.next_id += 1;
            let source = Arc::downgrade(&self.state);
            let attachment = Arc::new(TestSourceAttachment {
                snapshot: ReplicatedStateSnapshot {
                    value: state.value.clone(),
                    cursor: state.cursor,
                },
                state: Mutex::new(AttachmentState {
                    buffer: Vec::new(),
                    listener: None,
                    activated: false,
                    disposed: false,
                }),
                on_dispose: Box::new(move || {
                    if let Some(source) = source.upgrade() {
                        source.lock().attachments.retain(|(own, _)| *own != id);
                    }
                }),
            });
            state.attachments.push((id, attachment.clone()));
            attachment
        };
        if let Some(on_attach) = self.on_attach.lock().take() {
            on_attach();
        }
        Ok(Box::new(SharedAttachment(attachment)))
    }
}

impl TestSourceAttachment {
    fn publish(&self, frame: ReplicatedStateSourceFrame<Arc<Value>>) {
        let listener = {
            let mut state = self.state.lock();
            if state.disposed {
                return;
            }
            match &state.listener {
                None => {
                    state.buffer.push(frame);
                    return;
                }
                Some(listener) => listener.clone(),
            }
        };
        listener(frame);
    }
}

struct SharedAttachment(Arc<TestSourceAttachment>);

impl ReplicatedStateSourceAttachment<Arc<Value>> for SharedAttachment {
    fn snapshot(&self) -> ReplicatedStateSnapshot<Arc<Value>> {
        self.0.snapshot.clone()
    }

    fn activate(&self, listener: FrameListener<Arc<Value>>) -> Result<(), StateError> {
        let listener = Arc::new(listener);
        let buffered = {
            let mut state = self.0.state.lock();
            if state.activated {
                return Err(StateError::Contract("attachment is already active".into()));
            }
            state.activated = true;
            state.listener = Some(listener.clone());
            std::mem::take(&mut state.buffer)
        };
        for frame in buffered {
            listener(frame);
        }
        Ok(())
    }

    fn dispose(&self) -> Result<(), StateError> {
        {
            let mut state = self.0.state.lock();
            if state.disposed {
                return Ok(());
            }
            state.disposed = true;
            state.buffer.clear();
        }
        (self.0.on_dispose)();
        Ok(())
    }
}

#[test]
fn captures_before_activation_and_drains_queued_commits_in_order() {
    let source = TestSource::new(json!({ "value": 0 }), 10);
    let committer = source.clone();
    *source.on_attach.lock() = Some(Box::new(move || {
        committer.commit(json!({ "value": 1 }), set_value(1));
        committer.commit(json!({ "value": 2 }), set_value(2));
    }));
    let state = replicated_state(&source, None).unwrap();
    let deliveries = shared(Vec::new());
    let sink = deliveries.clone();
    state.subscribe(move |value, _, delivery| {
        sink.lock()
            .push((value["value"].as_i64().unwrap(), delivery.sequence));
        ListenerOutcome::ok()
    });
    assert_eq!(*state.value(), json!({ "value": 2 }));
    assert_eq!(*deliveries.lock(), vec![(2, 2)]);
    let (value, sequence) = state.internals().snapshot();
    assert_eq!((value["value"].as_i64(), sequence), (Some(2), 2));
}

#[test]
fn hydrates_at_sequence_zero_when_attaching_after_existing_commits() {
    let source = TestSource::new(json!({ "value": 0 }), 40);
    source.commit(json!({ "value": 1 }), set_value(1));
    let state = replicated_state(&source, None).unwrap();
    let deliveries = shared(Vec::new());
    let sink = deliveries.clone();
    state.subscribe(move |_, _, delivery| {
        sink.lock().push(delivery.sequence);
        ListenerOutcome::ok()
    });
    assert_eq!(*state.value(), json!({ "value": 1 }));
    assert_eq!(*deliveries.lock(), vec![0]);
}

#[test]
fn publishes_exact_source_value_and_operation_references() {
    let source = TestSource::new(json!({ "value": 0 }), 0);
    let state = replicated_state(&source, None).unwrap();
    let published_ops = shared(None::<Arc<[Op]>>);
    let sink = published_ops.clone();
    state.internals().subscribe(move |ops, _, _| {
        *sink.lock() = Some(ops.clone());
        Ok(())
    });
    let published_value = shared(None::<Arc<Value>>);
    let sink = published_value.clone();
    state.subscribe(move |value, _, delivery| {
        if delivery.kind == DeliveryKind::Update {
            *sink.lock() = Some(value);
        }
        ListenerOutcome::ok()
    });
    let ops = set_value(1);
    source.commit(json!({ "value": 1 }), ops.clone());
    let committed = source.state.lock().value.clone();
    assert!(Arc::ptr_eq(&state.value(), &committed));
    assert!(Arc::ptr_eq(
        published_value.lock().as_ref().unwrap(),
        &committed
    ));
    assert!(Arc::ptr_eq(published_ops.lock().as_ref().unwrap(), &ops));
}

#[test]
fn buffers_reentrant_frames_and_skips_updates_covered_by_a_late_hydration() {
    let source = TestSource::new(json!({ "value": 0 }), 0);
    let state = Arc::new(replicated_state(&source, None).unwrap());
    let received = shared(Vec::new());
    let late = shared(Vec::new());
    let nested = Arc::new(AtomicBool::new(false));
    {
        let source = source.clone();
        let state_ref = Arc::downgrade(&state);
        let late = late.clone();
        state.internals().subscribe(move |_, sequence, _| {
            if sequence != 1 || nested.swap(true, Ordering::SeqCst) {
                return Ok(());
            }
            source.commit(json!({ "value": 2 }), set_value(2));
            let late = late.clone();
            state_ref
                .upgrade()
                .unwrap()
                .subscribe(move |value, _, delivery| {
                    late.lock()
                        .push((delivery, value["value"].as_i64().unwrap()));
                    ListenerOutcome::ok()
                });
            Ok(())
        });
    }
    let sink = received.clone();
    state.subscribe(move |value, _, delivery| {
        if delivery.kind == DeliveryKind::Update {
            sink.lock().push(value["value"].as_i64().unwrap());
        }
        ListenerOutcome::ok()
    });
    source.commit(json!({ "value": 1 }), set_value(1));
    assert_eq!(*received.lock(), vec![1, 2]);
    assert_eq!(*late.lock(), vec![(ReplicatedStateDelivery::hydrate(2), 2)]);
}

fn collect_errors() -> (Shared<Vec<String>>, ReplicatedStateSourceOptions) {
    let errors = shared(Vec::new());
    let sink = errors.clone();
    let options = ReplicatedStateSourceOptions {
        on_error: Some(Arc::new(move |error: StateError| {
            sink.lock().push(error.to_string())
        })),
    };
    (errors, options)
}

#[test]
fn reports_listener_failures_without_throwing_them_into_the_source() {
    let source = TestSource::new(json!({ "value": 0 }), 0);
    let (errors, options) = collect_errors();
    let state = replicated_state(&source, Some(options)).unwrap();
    let received = shared(Vec::new());
    state.subscribe(|_, _, delivery| {
        if delivery.kind == DeliveryKind::Update {
            return ListenerOutcome::Done(Err(failure("listener failed")));
        }
        ListenerOutcome::ok()
    });
    let sink = received.clone();
    state.subscribe(move |value, _, delivery| {
        if delivery.kind == DeliveryKind::Update {
            sink.lock().push(value["value"].as_i64().unwrap());
        }
        ListenerOutcome::ok()
    });
    source.commit(json!({ "value": 1 }), set_value(1));
    source.commit(json!({ "value": 2 }), set_value(2));
    assert_eq!(*received.lock(), vec![1, 2]);
    assert_eq!(*errors.lock(), vec!["listener failed", "listener failed"]);
}

#[test]
fn reports_cursor_gaps_disposes_the_attachment_and_ignores_later_frames() {
    let source = TestSource::new(json!({ "value": 0 }), 5);
    let (errors, options) = collect_errors();
    let state = replicated_state(&source, Some(options)).unwrap();
    source.commit_at(json!({ "value": 2 }), set_value(2), 7);
    assert!(errors.lock()[0].contains("expected 6, received 7"));
    assert_eq!(source.attachments(), 0);
    assert_eq!(*state.value(), json!({ "value": 0 }));
    source.commit_at(json!({ "value": 3 }), set_value(3), 8);
    assert_eq!(*state.value(), json!({ "value": 0 }));
}

#[test]
fn keeps_attachments_independent_and_disposes_each_idempotently() {
    let source = TestSource::new(json!({ "value": 0 }), 0);
    let first = replicated_state(&source, None).unwrap();
    let second = replicated_state(&source, None).unwrap();
    assert_eq!(source.attachments(), 2);
    source.commit(json!({ "value": 1 }), set_value(1));
    assert_eq!(first.value()["value"], json!(1));
    assert_eq!(second.value()["value"], json!(1));
    first.dispose().unwrap();
    first.dispose().unwrap();
    assert_eq!(source.attachments(), 1);
    source.commit(json!({ "value": 2 }), set_value(2));
    assert_eq!(first.value()["value"], json!(1));
    assert_eq!(second.value()["value"], json!(2));
    second.dispose().unwrap();
    assert_eq!(source.attachments(), 0);
}

// ─── state-delivery.test.ts ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Kind {
    Mutable,
    Attached,
    Replica,
}

const KINDS: [Kind; 3] = [Kind::Mutable, Kind::Attached, Kind::Replica];

type Publish = Arc<dyn Fn(i64, Context) + Send + Sync>;

struct ListenerSlot(Mutex<Option<Arc<FrameListener<Arc<Value>>>>>);

struct SlotSource(Arc<ListenerSlot>);

struct SlotAttachment(Arc<ListenerSlot>);

impl ReplicatedStateSourceAttachment<Arc<Value>> for SlotAttachment {
    fn snapshot(&self) -> ReplicatedStateSnapshot<Arc<Value>> {
        ReplicatedStateSnapshot {
            value: Arc::new(json!({ "value": 0 })),
            cursor: 0,
        }
    }

    fn activate(&self, listener: FrameListener<Arc<Value>>) -> Result<(), StateError> {
        *self.0.0.lock() = Some(Arc::new(listener));
        Ok(())
    }

    fn dispose(&self) -> Result<(), StateError> {
        Ok(())
    }
}

impl ReplicatedStateSource<Arc<Value>> for SlotSource {
    fn attach(&self) -> Result<Box<dyn ReplicatedStateSourceAttachment<Arc<Value>>>, StateError> {
        Ok(Box::new(SlotAttachment(self.0.clone())))
    }
}

type Fixture = (
    Arc<dyn ReplicatedState<Arc<Value>>>,
    Publish,
    Option<ReplicatedStateInternals<Arc<Value>>>,
);

fn fixture(kind: Kind, on_error: impl Fn(StateError) + Send + Sync + 'static) -> Fixture {
    match kind {
        Kind::Mutable => {
            let state = Arc::new(mutable_replicated_state(json!({ "value": 0 })));
            let internals = state.internals();
            let target = state.clone();
            let publish: Publish = Arc::new(move |value, context| {
                let _ = target.replace(&context, json!({ "value": value }));
            });
            (state, publish, Some(internals))
        }
        Kind::Attached => {
            let slot = Arc::new(ListenerSlot(Mutex::new(None)));
            let options = ReplicatedStateSourceOptions {
                on_error: Some(Arc::new(on_error)),
            };
            let state =
                Arc::new(replicated_state(&SlotSource(slot.clone()), Some(options)).unwrap());
            let internals = state.internals();
            let publish: Publish = Arc::new(move |value, context| {
                let receive = slot.0.lock().clone().unwrap();
                receive(ReplicatedStateSourceFrame {
                    cursor: value,
                    value: Arc::new(json!({ "value": value })),
                    ops: set_value(value),
                    context,
                });
            });
            (state, publish, Some(internals))
        }
        Kind::Replica => {
            let replica = Arc::new(ReplicatedStateReplica::new(on_error));
            replica
                .hydrate(0, &[Op::R(json!({ "value": 0 }))], &bg())
                .unwrap();
            let target = replica.clone();
            let publish: Publish = Arc::new(move |value, context| {
                target
                    .update(value as u64, &set_value(value), &context)
                    .unwrap();
            });
            (replica, publish, None)
        }
    }
}

fn number(value: &Arc<Value>) -> i64 {
    value["value"].as_i64().unwrap()
}

type Gate = Shared<Option<tokio::sync::oneshot::Receiver<Result<(), String>>>>;

fn gate() -> (tokio::sync::oneshot::Sender<Result<(), String>>, Gate) {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    (sender, shared(Some(receiver)))
}

fn wait_gate(gate: &Gate) -> ListenerOutcome {
    let receiver = gate.lock().take().expect("gate used once");
    ListenerOutcome::pending(async move {
        match receiver.await {
            Ok(Ok(())) | Err(_) => Ok(()),
            Ok(Err(message)) => Err(failure(&message)),
        }
    })
}

#[tokio::test]
async fn awaits_hydration_and_each_update_independently() {
    for kind in KINDS {
        let (state, publish, internals) = fixture(kind, |_| {});
        let (hydration, hydration_gate) = gate();
        let (update, update_gate) = gate();
        let events = shared(Vec::<String>::new());
        let exact = shared(Vec::new());
        let fast = shared(Vec::new());
        if let Some(internals) = &internals {
            let sink = exact.clone();
            internals.subscribe(move |_, sequence, _| {
                sink.lock().push(sequence);
                Ok(())
            });
        }
        let sink = events.clone();
        state.subscribe(Arc::new(move |value, _, _| {
            let value = number(&value);
            sink.lock().push(format!("start:{value}"));
            let sink = sink.clone();
            let wait = match value {
                0 => Some(wait_gate(&hydration_gate)),
                1 => Some(wait_gate(&update_gate)),
                _ => None,
            };
            ListenerOutcome::pending(async move {
                if let Some(ListenerOutcome::Pending(wait)) = wait {
                    wait.await?;
                }
                sink.lock().push(format!("end:{value}"));
                Ok(())
            })
        }));
        let sink = fast.clone();
        state.subscribe(Arc::new(move |value, _, _| {
            sink.lock().push(number(&value));
            ListenerOutcome::ok()
        }));
        publish(1, bg());
        publish(2, bg());
        assert_eq!(*events.lock(), vec!["start:0"], "{kind:?}");
        assert_eq!(*fast.lock(), vec![0, 1, 2]);
        assert_eq!(*state.current().unwrap(), json!({ "value": 2 }));
        if internals.is_some() {
            assert_eq!(*exact.lock(), vec![1, 2]);
        }
        hydration.send(Ok(())).unwrap();
        wait_for(|| *events.lock() == ["start:0", "end:0", "start:1"]).await;
        update.send(Ok(())).unwrap();
        wait_for(|| *events.lock() == ["start:0", "end:0", "start:1", "end:1", "start:2", "end:2"])
            .await;
    }
}

#[tokio::test]
async fn bounds_pending_deliveries_without_changing_exact_publication() {
    for kind in KINDS {
        for count in [100i64, 101, 102, 201, 202] {
            let (state, publish, internals) = fixture(kind, |_| {});
            let (hydration, hydration_gate) = gate();
            let received = shared(Vec::new());
            let exact = shared(Vec::new());
            if let Some(internals) = &internals {
                let sink = exact.clone();
                internals.subscribe(move |_, sequence, _| {
                    sink.lock().push(sequence as i64);
                    Ok(())
                });
            }
            let sink = received.clone();
            state.subscribe(Arc::new(move |value, _, _| {
                let value = number(&value);
                sink.lock().push(value);
                if value == 0 {
                    wait_gate(&hydration_gate)
                } else {
                    ListenerOutcome::ok()
                }
            }));
            for value in 1..=count {
                publish(value, bg());
            }
            assert_eq!(*received.lock(), vec![0]);
            if internals.is_some() {
                assert_eq!(*exact.lock(), (1..=count).collect::<Vec<_>>());
            }
            hydration.send(Ok(())).unwrap();
            let first = ((count - 1) / 100) * 100 + 1;
            let mut expected = vec![0];
            expected.extend(first..=count);
            wait_for(|| *received.lock() == expected).await;
        }
    }
}

#[tokio::test]
async fn excludes_a_running_update_from_overflow_and_retains_exact_frames() {
    for kind in KINDS {
        let (state, publish, _) = fixture(kind, |_| {});
        let (release, running_gate) = gate();
        let context = with_cancel(&bg()).context;
        let received = shared(Vec::<(Arc<Value>, Context, ReplicatedStateDelivery)>::new());
        let sink = received.clone();
        state.subscribe(Arc::new(move |value, context, delivery| {
            let first_update = number(&value) == 1;
            sink.lock().push((value, context, delivery));
            if first_update {
                wait_gate(&running_gate)
            } else {
                ListenerOutcome::ok()
            }
        }));
        for value in 1..102 {
            publish(value, bg());
        }
        publish(102, context.clone());
        let adopted = state.current().unwrap();
        publish(103, bg());
        let values = || {
            received
                .lock()
                .iter()
                .map(|(value, ..)| number(value))
                .collect::<Vec<_>>()
        };
        assert_eq!(values(), vec![0, 1]);
        release.send(Ok(())).unwrap();
        wait_for(|| values() == [0, 1, 102, 103]).await;
        let received = received.lock();
        assert!(Arc::ptr_eq(&received[2].0, &adopted));
        assert!(received[2].1.ptr_eq(&context));
        assert_eq!(received[2].2, ReplicatedStateDelivery::update(102));
    }
}

#[test]
fn serializes_reentrant_hydration_and_update_callbacks() {
    for kind in KINDS {
        let (state, publish, _) = fixture(kind, |_| {});
        let events = shared(Vec::<String>::new());
        let sink = events.clone();
        let republish = publish.clone();
        state.subscribe(Arc::new(move |value, _, _| {
            let value = number(&value);
            sink.lock().push(format!("start:{value}"));
            if value < 2 {
                republish(value + 1, bg());
            }
            sink.lock().push(format!("end:{value}"));
            ListenerOutcome::ok()
        }));
        assert_eq!(
            *events.lock(),
            vec!["start:0", "end:0", "start:1", "end:1", "start:2", "end:2"],
            "{kind:?}"
        );
    }
}

#[tokio::test]
async fn treats_two_subscriptions_of_the_same_callback_independently() {
    for kind in KINDS {
        let (state, publish, _) = fixture(kind, |_| {});
        let (release, hydration_gate) = gate();
        let gates = shared(vec![hydration_gate]);
        let received = shared(Vec::new());
        let sink = received.clone();
        let listener: StateListener<Arc<Value>> = Arc::new(move |value, _, _| {
            let value = number(&value);
            sink.lock().push(value);
            if value == 0
                && let Some(gate) = gates.lock().pop()
            {
                return wait_gate(&gate);
            }
            ListenerOutcome::ok()
        });
        let stop_first = state.subscribe(listener.clone());
        let stop_second = state.subscribe(listener);
        publish(1, bg());
        stop_first.unsubscribe();
        stop_first.unsubscribe();
        release.send(Ok(())).unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(*received.lock(), vec![0, 0, 1], "{kind:?}");
        stop_second.unsubscribe();
    }
}

#[tokio::test]
async fn unsubscribe_drops_queued_callbacks_without_joining_the_running_callback() {
    for kind in KINDS {
        let (state, publish, _) = fixture(kind, |_| {});
        let (release, running_gate) = gate();
        let cancel = with_cancel(&bg());
        let received = shared(Vec::new());
        let completed = Arc::new(AtomicBool::new(false));
        let sink = received.clone();
        let done = completed.clone();
        let stop = state.subscribe(Arc::new(move |value, _, _| {
            let value = number(&value);
            sink.lock().push(value);
            if value == 1 {
                let wait = wait_gate(&running_gate);
                let done = done.clone();
                return ListenerOutcome::pending(async move {
                    if let ListenerOutcome::Pending(wait) = wait {
                        wait.await?;
                    }
                    done.store(true, Ordering::SeqCst);
                    Ok(())
                });
            }
            ListenerOutcome::ok()
        }));
        publish(1, cancel.context.clone());
        publish(2, cancel.context.clone());
        stop.unsubscribe();
        publish(3, cancel.context.clone());
        assert!(!cancel.context.abort_signal().unwrap().aborted());
        assert!(!completed.load(Ordering::SeqCst));
        release.send(Ok(())).unwrap();
        wait_for(|| completed.load(Ordering::SeqCst)).await;
        assert_eq!(*received.lock(), vec![0, 1], "{kind:?}");
    }
}

#[test]
fn isolates_a_synchronous_hydration_failure_without_removing_the_subscription() {
    for kind in [Kind::Attached, Kind::Replica] {
        let errors = shared(Vec::<String>::new());
        let sink = errors.clone();
        let (state, publish, _) = fixture(kind, move |error| sink.lock().push(error.to_string()));
        let received = shared(Vec::new());
        let sink = received.clone();
        state.subscribe(Arc::new(move |value, _, _| {
            let value = number(&value);
            sink.lock().push(value);
            if value == 0 {
                return ListenerOutcome::Done(Err(failure("sync hydration")));
            }
            ListenerOutcome::ok()
        }));
        publish(1, bg());
        assert_eq!(*errors.lock(), vec!["sync hydration"]);
        assert_eq!(*received.lock(), vec![0, 1]);
    }
}

#[tokio::test]
async fn observes_hydration_rejection_sync_throw_and_update_rejection_while_continuing() {
    for kind in [Kind::Attached, Kind::Replica] {
        let errors = shared(Vec::<String>::new());
        let sink = errors.clone();
        let (state, publish, _) = fixture(kind, move |error| sink.lock().push(error.to_string()));
        let (reject, hydration_gate) = gate();
        let received = shared(Vec::new());
        let fast = shared(Vec::new());
        let sink = received.clone();
        state.subscribe(Arc::new(move |value, _, _| {
            let value = number(&value);
            sink.lock().push(value);
            match value {
                0 => wait_gate(&hydration_gate),
                1 => ListenerOutcome::Done(Err(failure("sync update"))),
                2 => ListenerOutcome::pending(async { Err(failure("async update")) }),
                _ => ListenerOutcome::ok(),
            }
        }));
        let sink = fast.clone();
        state.subscribe(Arc::new(move |value, _, _| {
            sink.lock().push(number(&value));
            ListenerOutcome::ok()
        }));
        for value in 1..=3 {
            publish(value, bg());
        }
        reject.send(Err("async hydration".into())).unwrap();
        wait_for(|| *received.lock() == [0, 1, 2, 3]).await;
        wait_for(|| errors.lock().len() == 3).await;
        assert_eq!(
            *errors.lock(),
            vec!["async hydration", "sync update", "async update"]
        );
        assert_eq!(*fast.lock(), vec![0, 1, 2, 3]);
    }
}

#[tokio::test]
async fn replica_clear_drops_pending_work_but_waits_for_the_running_callback() {
    let replica = ReplicatedStateReplica::new(|_| {});
    let (release, hydration_gate) = gate();
    let gates = shared(vec![hydration_gate]);
    let received = shared(Vec::new());
    let sink = received.clone();
    replica.subscribe(move |_, _, delivery| {
        sink.lock().push(delivery);
        if delivery.sequence == 0
            && let Some(gate) = gates.lock().pop()
        {
            return wait_gate(&gate);
        }
        ListenerOutcome::ok()
    });
    replica
        .hydrate(0, &[Op::R(json!({ "value": 0 }))], &bg())
        .unwrap();
    replica.update(1, &set_value(1), &bg()).unwrap();
    replica.clear();
    replica
        .hydrate(50, &[Op::R(json!({ "value": 50 }))], &bg())
        .unwrap();
    replica.update(51, &set_value(51), &bg()).unwrap();
    assert_eq!(*received.lock(), vec![ReplicatedStateDelivery::hydrate(0)]);
    release.send(Ok(())).unwrap();
    let expected = vec![
        ReplicatedStateDelivery::hydrate(0),
        ReplicatedStateDelivery::hydrate(50),
        ReplicatedStateDelivery::update(51),
    ];
    wait_for(|| *received.lock() == expected).await;
}

#[test]
fn cold_replicas_hydrate_all_listeners_before_their_reentrant_updates() {
    let replica = Arc::new(ReplicatedStateReplica::new(|_| {}));
    let second = shared(Vec::new());
    let target = Arc::downgrade(&replica);
    replica.subscribe(move |_, _, delivery| {
        if delivery.kind == DeliveryKind::Hydrate {
            target
                .upgrade()
                .unwrap()
                .update(1, &set_value(1), &bg())
                .unwrap();
        }
        ListenerOutcome::ok()
    });
    let sink = second.clone();
    replica.subscribe(move |_, _, delivery| {
        sink.lock().push(delivery);
        ListenerOutcome::ok()
    });
    replica
        .hydrate(0, &[Op::R(json!({ "value": 0 }))], &bg())
        .unwrap();
    assert_eq!(
        *second.lock(),
        vec![
            ReplicatedStateDelivery::hydrate(0),
            ReplicatedStateDelivery::update(1)
        ]
    );
}

#[test]
fn rejects_unsafe_snapshot_cursors_and_disposes_the_attachment() {
    struct BadSource(Arc<AtomicBool>);
    struct BadAttachment(Arc<AtomicBool>);
    impl ReplicatedStateSourceAttachment<Arc<Value>> for BadAttachment {
        fn snapshot(&self) -> ReplicatedStateSnapshot<Arc<Value>> {
            ReplicatedStateSnapshot {
                value: Arc::new(Value::Null),
                cursor: i64::MAX,
            }
        }
        fn activate(&self, _listener: FrameListener<Arc<Value>>) -> Result<(), StateError> {
            Ok(())
        }
        fn dispose(&self) -> Result<(), StateError> {
            self.0.store(true, Ordering::SeqCst);
            Ok(())
        }
    }
    impl ReplicatedStateSource<Arc<Value>> for BadSource {
        fn attach(
            &self,
        ) -> Result<Box<dyn ReplicatedStateSourceAttachment<Arc<Value>>>, StateError> {
            Ok(Box::new(BadAttachment(self.0.clone())))
        }
    }
    let disposed = Arc::new(AtomicBool::new(false));
    let error = match replicated_state(&BadSource(disposed.clone()), None) {
        Err(error) => error,
        Ok(_) => panic!("unsafe cursor accepted"),
    };
    assert_eq!(
        error.to_string(),
        "Replicated state source snapshot cursor must be a safe integer"
    );
    assert!(disposed.load(Ordering::SeqCst));
}
