//! Port of durable `src/session/observation.ts`: the committed state source
//! bridging Session commits to Chord, and serialized exact-frame watches.
//!
//! Divergences from Pi:
//! - Observed values are generic over [`ObservedValue`]; a document observes
//!   [`ObservedDocumentValue`] (`Option<Arc<JsonValue>>`, `None` for TS `null`).
//! - `queueMicrotask` scheduling is a spawned tokio task, so both observers
//!   need a tokio runtime.
//! - A watch listener returns a boxed future of `Result`; an `Err` or a panic
//!   ends the watch with `listener_error`, as a rejection does in TS.

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, LazyLock};

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::chord::context::AbortListenerId;
use crate::chord::delta::Op;
use crate::chord::state::{FrameListener, ReplicatedStateSnapshot};
use crate::chord::{
    AbortSignal, Context, JsonValue, ReplicatedStateSource, ReplicatedStateSourceAttachment,
    ReplicatedStateSourceFrame, StateError, without_abort_signal,
};

use crate::durable::errors::{Error, Result};
use crate::durable::types::WatchEnd;

/// `Readonly<JsonObject> | null`: the shared immutable committed revision, `None` once retired.
pub type ObservedDocumentValue = Option<Arc<JsonValue>>;

/// A value observed through a committed source or watch. A retired value ends observation.
pub trait ObservedValue: Clone + Send + Sync + 'static {
    /// TS `value === null`.
    fn is_retired(&self) -> bool;
    /// The JSON root used by a root replacement (`["r", value]`).
    fn root_value(&self) -> JsonValue;
}

impl ObservedValue for ObservedDocumentValue {
    fn is_retired(&self) -> bool {
        self.is_none()
    }

    fn root_value(&self) -> JsonValue {
        self.as_deref().cloned().unwrap_or(JsonValue::Null)
    }
}

/// Maximum exact committed frames retained behind one unavailable watch listener.
const MAX_PENDING_WATCH_FRAMES: usize = 100;

/// Canonical terminal update for a retired document incarnation (`[["r", null]]`).
pub static RETIREMENT_OPERATIONS: LazyLock<Arc<[Op]>> =
    LazyLock::new(|| Arc::from(vec![Op::R(JsonValue::Null)]));

type Release = Box<dyn FnOnce() + Send>;

// ─── Committed state source ──────────────────────────────────────────────────

struct SourceState<T> {
    attachments: Vec<(u64, Arc<SessionSourceAttachment<T>>)>,
    next_attachment: u64,
    release: Option<Release>,
    /// Released on disposal.
    value: Option<T>,
    cursor: i64,
    retired: bool,
    closed: bool,
}

/// Session-to-Chord bridge owned one-to-one by one attached state: a document,
/// or a conversation view. A retired value retires it.
pub struct CommittedStateSource<T> {
    state: Arc<Mutex<SourceState<T>>>,
}

impl<T> Clone for CommittedStateSource<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<T: ObservedValue> CommittedStateSource<T> {
    pub fn new(value: T, release: impl FnOnce() + Send + 'static) -> Self {
        Self {
            state: Arc::new(Mutex::new(SourceState {
                attachments: Vec::new(),
                next_attachment: 0,
                release: Some(Box::new(release)),
                value: Some(value),
                cursor: 0,
                retired: false,
                closed: false,
            })),
        }
    }

    pub fn advance(&self, value: T, ops: Arc<[Op]>, context: Context) {
        let (attachments, frame) = {
            let mut state = self.state.lock();
            if state.closed || state.retired {
                return;
            }
            state.value = Some(value.clone());
            state.cursor += 1;
            if value.is_retired() {
                state.retired = true;
            }
            let frame = ReplicatedStateSourceFrame {
                cursor: state.cursor,
                value,
                ops,
                context,
            };
            let attachments: Vec<_> = state
                .attachments
                .iter()
                .map(|(_, attachment)| attachment.clone())
                .collect();
            (attachments, frame)
        };
        for attachment in attachments {
            attachment.publish(frame.clone());
        }
    }

    pub fn close_session(&self) {
        let attachments: Vec<_> = {
            let state = self.state.lock();
            if state.closed {
                return;
            }
            state
                .attachments
                .iter()
                .map(|(_, attachment)| attachment.clone())
                .collect()
        };
        for attachment in attachments {
            attachment.dispose();
        }
        finish_source_disposal(&self.state);
    }
}

fn finish_source_disposal<T>(state: &Mutex<SourceState<T>>) {
    let release = {
        let mut state = state.lock();
        if state.closed {
            return;
        }
        state.closed = true;
        state.value = None;
        state.release.take()
    };
    if let Some(release) = release {
        release();
    }
}

impl<T: ObservedValue> ReplicatedStateSource<T> for CommittedStateSource<T> {
    fn attach(
        &self,
    ) -> std::result::Result<Box<dyn ReplicatedStateSourceAttachment<T>>, StateError> {
        let mut state = self.state.lock();
        if state.closed {
            return Err(StateError::Contract("State source is closed".into()));
        }
        let id = state.next_attachment;
        state.next_attachment += 1;
        let source = Arc::downgrade(&self.state);
        let snapshot = ReplicatedStateSnapshot {
            value: state.value.clone().expect("an open source holds its value"),
            cursor: state.cursor,
        };
        let attachment = Arc::new(SessionSourceAttachment::new(
            snapshot,
            Box::new(move || {
                let Some(source) = source.upgrade() else {
                    return;
                };
                let empty = {
                    let mut state = source.lock();
                    state.attachments.retain(|(own, _)| *own != id);
                    state.attachments.is_empty()
                };
                if empty {
                    finish_source_disposal(&source);
                }
            }),
        ));
        state.attachments.push((id, attachment.clone()));
        Ok(Box::new(AttachmentHandle(attachment)))
    }
}

struct AttachmentState<T> {
    release: Option<Release>,
    frames: VecDeque<ReplicatedStateSourceFrame<T>>,
    listener: Option<Arc<FrameListener<T>>>,
    activated: bool,
    disposed: bool,
    scheduled: bool,
    delivering: bool,
}

struct SessionSourceAttachment<T> {
    snapshot: ReplicatedStateSnapshot<T>,
    state: Mutex<AttachmentState<T>>,
}

impl<T: ObservedValue> SessionSourceAttachment<T> {
    fn new(snapshot: ReplicatedStateSnapshot<T>, release: Release) -> Self {
        Self {
            snapshot,
            state: Mutex::new(AttachmentState {
                release: Some(release),
                frames: VecDeque::new(),
                listener: None,
                activated: false,
                disposed: false,
                scheduled: false,
                delivering: false,
            }),
        }
    }

    fn activate(&self, listener: FrameListener<T>) -> std::result::Result<(), StateError> {
        {
            let mut state = self.state.lock();
            if state.activated {
                return Err(StateError::Contract(
                    "State attachment is already active".into(),
                ));
            }
            if state.disposed {
                return Err(StateError::Contract("State attachment is disposed".into()));
            }
            state.activated = true;
            state.listener = Some(Arc::new(listener));
        }
        self.drain();
        Ok(())
    }

    fn publish(self: &Arc<Self>, frame: ReplicatedStateSourceFrame<T>) {
        {
            let mut state = self.state.lock();
            if state.disposed {
                return;
            }
            state.frames.push_back(frame);
            if !state.activated || state.delivering || state.scheduled {
                return;
            }
            state.scheduled = true;
        }
        let attachment = self.clone();
        tokio::spawn(async move {
            {
                let mut state = attachment.state.lock();
                state.scheduled = false;
                if state.disposed {
                    return;
                }
            }
            // Chord's frame listener contains source-contract failures. Isolate an
            // unexpected direct listener failure to this attachment as well.
            if std::panic::catch_unwind(AssertUnwindSafe(|| attachment.drain())).is_err() {
                attachment.dispose();
            }
        });
    }

    fn dispose(&self) {
        let release = {
            let mut state = self.state.lock();
            if state.disposed {
                return;
            }
            state.disposed = true;
            state.frames.clear();
            state.listener = None;
            state.release.take()
        };
        if let Some(release) = release {
            release();
        }
    }

    fn drain(&self) {
        let listener = {
            let mut state = self.state.lock();
            let Some(listener) = state.listener.clone() else {
                return;
            };
            if state.delivering || state.disposed {
                return;
            }
            state.delivering = true;
            listener
        };
        struct Delivering<'a, T>(&'a Mutex<AttachmentState<T>>);
        impl<T> Drop for Delivering<'_, T> {
            fn drop(&mut self) {
                self.0.lock().delivering = false;
            }
        }
        let _delivering = Delivering(&self.state);
        loop {
            let frame = {
                let mut state = self.state.lock();
                if state.disposed {
                    break;
                }
                match state.frames.pop_front() {
                    Some(frame) => frame,
                    None => break,
                }
            };
            listener(frame);
        }
    }
}

struct AttachmentHandle<T>(Arc<SessionSourceAttachment<T>>);

impl<T: ObservedValue> ReplicatedStateSourceAttachment<T> for AttachmentHandle<T> {
    fn snapshot(&self) -> ReplicatedStateSnapshot<T> {
        self.0.snapshot.clone()
    }

    fn activate(&self, listener: FrameListener<T>) -> std::result::Result<(), StateError> {
        self.0.activate(listener)
    }

    fn dispose(&self) -> std::result::Result<(), StateError> {
        self.0.dispose();
        Ok(())
    }
}

// ─── Committed watch ─────────────────────────────────────────────────────────

/// The sole asynchronous listener of a watch: `(value, ops, context) => Promise<void>`.
pub type WatchListener<T> =
    Arc<dyn Fn(T, Arc<[Op]>, Context) -> BoxFuture<'static, Result<()>> + Send + Sync>;

struct WatchFrame<T> {
    value: T,
    ops: Arc<[Op]>,
    context: Context,
}

struct WatchState<T> {
    pending: VecDeque<WatchFrame<T>>,
    value: T,
    listener: Option<WatchListener<T>>,
    started: bool,
    scheduled: bool,
    running: bool,
    retired: bool,
    end: Option<WatchEnd>,
    resolved: bool,
    cancellation: Option<(AbortSignal, AbortListenerId)>,
    cancellation_installed: bool,
}

struct WatchInner<T> {
    detach: Mutex<Option<Release>>,
    replace: Option<Arc<dyn Fn() -> T + Send + Sync>>,
    closed: Shared<BoxFuture<'static, WatchEnd>>,
    resolve_closed: Mutex<Option<oneshot::Sender<WatchEnd>>>,
    state: Mutex<WatchState<T>>,
}

/// Serialized exact-frame watch bound to one document incarnation or
/// conversation view (`WatchHandle<T>`). A retired value ends it.
pub struct CommittedWatch<T> {
    inner: Arc<WatchInner<T>>,
}

impl<T> std::fmt::Debug for CommittedWatch<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommittedWatch").finish_non_exhaustive()
    }
}

impl<T> Clone for CommittedWatch<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: ObservedValue> CommittedWatch<T> {
    /// `replace` gives the value an overflow delivers; by default the newest value.
    pub fn new(
        value: T,
        detach: impl FnOnce() + Send + 'static,
        replace: Option<Arc<dyn Fn() -> T + Send + Sync>>,
    ) -> Self {
        let (resolve_closed, closed) = oneshot::channel();
        let closed = closed
            .map(|end| end.unwrap_or(WatchEnd::SessionClosed))
            .boxed()
            .shared();
        Self {
            inner: Arc::new(WatchInner {
                detach: Mutex::new(Some(Box::new(detach))),
                replace,
                closed,
                resolve_closed: Mutex::new(Some(resolve_closed)),
                state: Mutex::new(WatchState {
                    pending: VecDeque::new(),
                    value,
                    listener: None,
                    started: false,
                    scheduled: false,
                    running: false,
                    retired: false,
                    end: None,
                    resolved: false,
                    cancellation: None,
                    cancellation_installed: false,
                }),
            }),
        }
    }

    /// Acquisition revision before start; latest delivered immutable revision afterward.
    pub fn value(&self) -> T {
        self.inner.state.lock().value.clone()
    }

    /// Settles when the watch terminates; an already-running callback remains caller-owned.
    pub fn closed(&self) -> Shared<BoxFuture<'static, WatchEnd>> {
        self.inner.closed.clone()
    }

    /// Install the sole asynchronous listener. Never invokes it inline.
    pub fn start(
        &self,
        listener: impl Fn(T, Arc<[Op]>, Context) -> BoxFuture<'static, Result<()>>
        + Send
        + Sync
        + 'static,
    ) -> Result<()> {
        let schedule = {
            let mut state = self.inner.state.lock();
            if state.started {
                return Err(Error::message("Watch is already started"));
            }
            if state.end.is_some() {
                return Err(Error::message("Watch is stopped"));
            }
            state.started = true;
            state.listener = Some(Arc::new(listener));
            !state.pending.is_empty()
        };
        if schedule {
            schedule_watch(&self.inner);
        }
        Ok(())
    }

    /// Idempotently stop future callbacks and return this watch's terminal result.
    pub fn stop(&self) -> Shared<BoxFuture<'static, WatchEnd>> {
        terminate(&self.inner, WatchEnd::Stopped);
        self.inner.closed.clone()
    }

    pub fn observe_cancellation(&self, signal: &AbortSignal) -> Result<()> {
        {
            let mut state = self.inner.state.lock();
            if state.cancellation_installed {
                return Err(Error::message("Watch cancellation is already installed"));
            }
            if state.end.is_some() {
                return Ok(());
            }
            state.cancellation_installed = true;
        }
        let weak = Arc::downgrade(&self.inner);
        let id = signal.add_listener(move |_| {
            if let Some(inner) = weak.upgrade() {
                terminate(&inner, WatchEnd::Cancelled);
            }
        });
        self.inner.state.lock().cancellation = Some((signal.clone(), id));
        if signal.aborted() {
            self.cancel();
        }
        Ok(())
    }

    pub fn cancel(&self) {
        terminate(&self.inner, WatchEnd::Cancelled);
    }

    pub fn close_session(&self) {
        terminate(&self.inner, WatchEnd::SessionClosed);
    }

    pub fn advance(&self, value: T, ops: Arc<[Op]>, context: Context) {
        let schedule = {
            let mut state = self.inner.state.lock();
            if state.end.is_some() || state.retired {
                return;
            }
            if value.is_retired() {
                state.retired = true;
            }
            if state.pending.len() >= MAX_PENDING_WATCH_FRAMES {
                state.pending.clear();
                let replacement = self
                    .inner
                    .replace
                    .as_ref()
                    .map_or_else(|| value.clone(), |replace| replace());
                let ops: Arc<[Op]> = Arc::from(vec![Op::R(replacement.root_value())]);
                state.pending.push_back(WatchFrame {
                    value: replacement,
                    ops,
                    context,
                });
            } else {
                state.pending.push_back(WatchFrame {
                    value,
                    ops,
                    context,
                });
            }
            state.started
        };
        if schedule {
            schedule_watch(&self.inner);
        }
    }
}

fn schedule_watch<T: ObservedValue>(inner: &Arc<WatchInner<T>>) {
    {
        let mut state = inner.state.lock();
        if state.scheduled || state.running || state.end.is_some() {
            return;
        }
        state.scheduled = true;
    }
    let inner = inner.clone();
    tokio::spawn(async move {
        inner.state.lock().scheduled = false;
        drain_watch(inner).await;
    });
}

async fn drain_watch<T: ObservedValue>(inner: Arc<WatchInner<T>>) {
    {
        let mut state = inner.state.lock();
        if state.running || state.end.is_some() || !state.started {
            drop(state);
            finish_if_ready(&inner);
            return;
        }
        state.running = true;
    }
    loop {
        let (frame, listener) = {
            let mut state = inner.state.lock();
            if state.end.is_some() {
                break;
            }
            let Some(frame) = state.pending.pop_front() else {
                break;
            };
            state.value = frame.value.clone();
            let listener = state
                .listener
                .clone()
                .expect("a started watch has a listener");
            (frame, listener)
        };
        let delivery_context = without_abort_signal(&frame.context);
        let delivered =
            AssertUnwindSafe(listener(frame.value.clone(), frame.ops, delivery_context))
                .catch_unwind()
                .await;
        let failure = match delivered {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(panic) => Some(Error::message(panic_message(&panic))),
        };
        if let Some(error) = failure {
            if inner.state.lock().end.is_none() {
                terminate(&inner, WatchEnd::ListenerError(error));
            }
            break;
        }
        if frame.value.is_retired() {
            terminate(&inner, WatchEnd::Retired);
            break;
        }
    }
    let reschedule = {
        let mut state = inner.state.lock();
        state.running = false;
        state.end.is_none() && !state.pending.is_empty()
    };
    if reschedule {
        schedule_watch(&inner);
    }
    finish_if_ready(&inner);
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "Watch listener panicked".to_string()
    }
}

fn terminate<T>(inner: &Arc<WatchInner<T>>, end: WatchEnd) {
    {
        let mut state = inner.state.lock();
        if state.end.is_some() {
            return;
        }
        state.end = Some(end);
        state.pending.clear();
    }
    let detach = inner.detach.lock().take();
    if let Some(detach) = detach {
        detach();
    }
    finish_if_ready(inner);
}

fn finish_if_ready<T>(inner: &Arc<WatchInner<T>>) {
    let (end, cancellation) = {
        let mut state = inner.state.lock();
        if state.resolved {
            return;
        }
        let Some(end) = state.end.clone() else {
            return;
        };
        state.resolved = true;
        (end, state.cancellation.take())
    };
    if let Some((signal, id)) = cancellation {
        signal.remove_listener(id);
    }
    if let Some(resolve) = inner.resolve_closed.lock().take() {
        let _ = resolve.send(end);
    }
}
