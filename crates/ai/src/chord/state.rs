//! Port of the replicated-state parts of chord `src/services/state.ts`
//! (`StateSubscriber`, `ReplicatedStatePublisher`, `MutableReplicatedStateImpl`,
//! `attachReplicatedStateSource`, `ReplicatedStateReplica`), the matching
//! types from `src/types.ts`, the `replicatedState()` entry point from
//! `src/api.ts`, and `services/state-internals.ts`.
//!
//! Divergences from Pi:
//! - Listeners return a [`ListenerOutcome`]: `Done` for a synchronous callback
//!   (delivery continues inline, as in JS), or `Pending(future)` for an async
//!   one. A pending future is driven on the current Tokio runtime (or a helper
//!   thread outside one), the way JS settles a promise on the microtask queue,
//!   and the subscriber's next delivery waits for it.
//! - JS errors are values; listener and source failures here are
//!   [`StateError`]s. The default reporter (JS: rethrow on a microtask, which
//!   surfaces as an uncaught error) writes the error to stderr.
//! - State runs on threads: delivery bookkeeping is mutex-protected and no
//!   lock is held while user callbacks run, so reentrant publication from a
//!   listener behaves as in JS.
//! - `MutableReplicatedState::change` takes a synchronous `FnOnce(&mut Value)`
//!   over a draft copy (see `delta::tracker`), so the "PromiseLike callback"
//!   rejection is enforced by the type.
//! - `ReplicatedStateReplica` does not run `JsonRevisionValidator`: every
//!   `serde_json::Value` is already strict JSON.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use futures::future::BoxFuture;
use parking_lot::{Mutex, ReentrantMutex};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::context::{BACKGROUND_CONTEXT, Context};
use super::delta::{DeltaError, Op, Tracker, apply_immutable, is_base, track};

/// A listener or source failure as JS would throw it.
pub type SharedError = Arc<dyn std::error::Error + Send + Sync>;

/// Errors raised or reported by replicated state.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StateError {
    /// An error thrown by user code (a listener or change callback), reported or rethrown as is.
    #[error("{0}")]
    Thrown(SharedError),
    /// JS `AggregateError(errors, message)`.
    #[error("{message}")]
    Aggregate {
        message: String,
        errors: Vec<StateError>,
    },
    /// A contract violation Pi raises as `Error`/`TypeError` with this message.
    #[error("{0}")]
    Contract(String),
    #[error(transparent)]
    Delta(#[from] DeltaError),
}

impl StateError {
    fn contract(message: impl Into<String>) -> Self {
        Self::Contract(message.into())
    }

    fn aggregate(errors: Vec<StateError>, message: &str) -> Self {
        Self::Aggregate {
            message: message.to_string(),
            errors,
        }
    }
}

/// `ReplicatedStateDelivery.kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeliveryKind {
    Hydrate,
    Update,
}

/// `ReplicatedStateDelivery`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicatedStateDelivery {
    pub kind: DeliveryKind,
    pub sequence: u64,
}

impl ReplicatedStateDelivery {
    pub fn hydrate(sequence: u64) -> Self {
        Self {
            kind: DeliveryKind::Hydrate,
            sequence,
        }
    }

    pub fn update(sequence: u64) -> Self {
        Self {
            kind: DeliveryKind::Update,
            sequence,
        }
    }
}

/// What a state listener returned: a finished synchronous call or a pending async one.
pub enum ListenerOutcome {
    Done(Result<(), SharedError>),
    Pending(BoxFuture<'static, Result<(), SharedError>>),
}

impl ListenerOutcome {
    pub fn ok() -> Self {
        Self::Done(Ok(()))
    }

    pub fn err(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Done(Err(Arc::new(error)))
    }

    pub fn pending(future: impl Future<Output = Result<(), SharedError>> + Send + 'static) -> Self {
        Self::Pending(Box::pin(future))
    }
}

impl From<()> for ListenerOutcome {
    fn from((): ()) -> Self {
        Self::ok()
    }
}

impl fmt::Debug for ListenerOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Done(result) => f.debug_tuple("Done").field(result).finish(),
            Self::Pending(_) => f.write_str("Pending(..)"),
        }
    }
}

type StateListener<T> =
    Arc<dyn Fn(T, Context, ReplicatedStateDelivery) -> ListenerOutcome + Send + Sync>;
type ErrorReporter = Arc<dyn Fn(StateError) + Send + Sync>;
/// An exact-publication listener (`ReplicatedStateInternals.subscribe`).
pub type SourceListener =
    Arc<dyn Fn(&Arc<[Op]>, u64, &Context) -> Result<(), SharedError> + Send + Sync>;

/// The function returned by `subscribe`: idempotently stop the subscription.
/// Dropping it does not unsubscribe.
#[derive(Clone)]
pub struct Unsubscribe(Arc<dyn Fn() + Send + Sync>);

impl Unsubscribe {
    fn new(stop: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(stop))
    }

    pub fn unsubscribe(&self) {
        (self.0)();
    }
}

impl fmt::Debug for Unsubscribe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Unsubscribe")
    }
}

/// The public read side of every replicated state (`ReplicatedState<T>`).
///
/// Each subscription serializes callbacks, awaiting hydration before updates.
/// At most 100 deliveries wait behind the running callback; overflow keeps only
/// the newest pending value/context/delivery. Failures are reported in
/// isolation and delivery continues. Unsubscribe discards pending work without
/// aborting or joining a running callback.
pub trait ReplicatedState<T>: Send + Sync {
    /// The immutable value, or `None` until hydration.
    fn current(&self) -> Option<T>;

    fn subscribe(
        &self,
        listener: Arc<dyn Fn(T, Context, ReplicatedStateDelivery) -> ListenerOutcome + Send + Sync>,
    ) -> Unsubscribe;
}

fn report_error_async(error: StateError) {
    eprintln!("chord: unhandled replicated state error: {error}");
}

fn run_detached(future: BoxFuture<'static, ()>) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn(future);
        }
        Err(_) => {
            std::thread::spawn(move || futures::executor::block_on(future));
        }
    }
}

// ─── Subscribers ─────────────────────────────────────────────────────────────

struct StateDelivery<T> {
    value: T,
    context: Context,
    delivery: ReplicatedStateDelivery,
}

impl<T: Clone> Clone for StateDelivery<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            context: self.context.clone(),
            delivery: self.delivery,
        }
    }
}

struct SubscriberState<T> {
    pending: VecDeque<StateDelivery<T>>,
    running: bool,
    started: bool,
    closed: bool,
}

/// One public subscription, independent of producer and other subscriber progress.
struct StateSubscriber<T> {
    listener: StateListener<T>,
    report_error: ErrorReporter,
    state: Mutex<SubscriberState<T>>,
}

impl<T: Clone + Send + 'static> StateSubscriber<T> {
    fn new(listener: StateListener<T>, report_error: ErrorReporter) -> Arc<Self> {
        Arc::new(Self {
            listener,
            report_error,
            state: Mutex::new(SubscriberState {
                pending: VecDeque::new(),
                running: false,
                started: false,
                closed: false,
            }),
        })
    }

    fn push(&self, frame: StateDelivery<T>) {
        let mut state = self.state.lock();
        if state.closed {
            return;
        }
        if state.pending.len() == 100 {
            // A cold replica can queue updates reentrantly before this subscriber's first hydration starts.
            let hydration = if state.started {
                None
            } else {
                state.pending.pop_front()
            };
            state.pending.clear();
            if let Some(hydration) = hydration {
                state.pending.push_back(hydration);
            }
        }
        state.pending.push_back(frame);
    }

    fn drain(self: &Arc<Self>) {
        {
            let mut state = self.state.lock();
            if state.running || state.closed {
                return;
            }
            state.running = true;
        }
        loop {
            let frame = {
                let mut state = self.state.lock();
                match state.pending.pop_front() {
                    Some(frame) => {
                        state.started = true;
                        frame
                    }
                    None => {
                        state.running = false;
                        return;
                    }
                }
            };
            match (self.listener)(frame.value, frame.context, frame.delivery) {
                ListenerOutcome::Done(Ok(())) => {}
                ListenerOutcome::Done(Err(error)) => self.report(StateError::Thrown(error)),
                ListenerOutcome::Pending(future) => {
                    let subscriber = self.clone();
                    run_detached(Box::pin(async move {
                        if let Err(error) = future.await {
                            subscriber.report(StateError::Thrown(error));
                        }
                        subscriber.resume();
                    }));
                    return;
                }
            }
        }
    }

    fn clear(&self) {
        self.state.lock().pending.clear();
    }

    fn close(&self) {
        let mut state = self.state.lock();
        state.closed = true;
        state.pending.clear();
    }

    fn resume(self: &Arc<Self>) {
        self.state.lock().running = false;
        self.drain();
    }

    fn report(&self, error: StateError) {
        (self.report_error)(error);
    }
}

// ─── Publisher ───────────────────────────────────────────────────────────────

struct Publication<T> {
    value: T,
    ops: Arc<[Op]>,
    sequence: u64,
    context: Context,
}

struct PublisherState<T> {
    listeners: Vec<(u64, Arc<StateSubscriber<T>>, u64)>,
    source_listeners: Vec<(u64, SourceListener)>,
    publications: VecDeque<Publication<T>>,
    value: T,
    sequence: u64,
    delivering: bool,
    next_id: u64,
}

/// Maintains local publication order independently of how revisions are produced.
struct ReplicatedStatePublisher<T> {
    report_error: ErrorReporter,
    state: Mutex<PublisherState<T>>,
}

impl<T: Clone + Send + 'static> ReplicatedStatePublisher<T> {
    fn new(initial: T, report_error: ErrorReporter) -> Arc<Self> {
        Arc::new(Self {
            report_error,
            state: Mutex::new(PublisherState {
                listeners: Vec::new(),
                source_listeners: Vec::new(),
                publications: VecDeque::new(),
                value: initial,
                sequence: 0,
                delivering: false,
                next_id: 0,
            }),
        })
    }

    fn value(&self) -> T {
        self.state.lock().value.clone()
    }

    fn snapshot(&self) -> (T, u64) {
        let state = self.state.lock();
        (state.value.clone(), state.sequence)
    }

    fn subscribe(self: &Arc<Self>, listener: StateListener<T>) -> Unsubscribe {
        let subscriber = StateSubscriber::new(listener, self.report_error.clone());
        let (id, value, sequence) = {
            let mut state = self.state.lock();
            let id = state.next_id;
            state.next_id += 1;
            let sequence = state.sequence;
            state.listeners.push((id, subscriber.clone(), sequence));
            (id, state.value.clone(), sequence)
        };
        subscriber.push(StateDelivery {
            value,
            context: service_delivery_context(),
            delivery: ReplicatedStateDelivery::hydrate(sequence),
        });
        subscriber.drain();
        let publisher = Arc::downgrade(self);
        Unsubscribe::new(move || {
            subscriber.close();
            if let Some(publisher) = publisher.upgrade() {
                publisher
                    .state
                    .lock()
                    .listeners
                    .retain(|(own, ..)| *own != id);
            }
        })
    }

    fn subscribe_source(self: &Arc<Self>, listener: SourceListener) -> Unsubscribe {
        let id = {
            let mut state = self.state.lock();
            let id = state.next_id;
            state.next_id += 1;
            state.source_listeners.push((id, listener));
            id
        };
        let publisher: Weak<Self> = Arc::downgrade(self);
        Unsubscribe::new(move || {
            if let Some(publisher) = publisher.upgrade() {
                publisher
                    .state
                    .lock()
                    .source_listeners
                    .retain(|(own, _)| *own != id);
            }
        })
    }

    /// Publish an already-prepared immutable revision and return isolated listener failures.
    fn publish(&self, value: T, ops: Arc<[Op]>, context: Context) -> Vec<SharedError> {
        {
            let mut state = self.state.lock();
            state.value = value.clone();
            state.sequence += 1;
            let sequence = state.sequence;
            state.publications.push_back(Publication {
                value,
                ops,
                sequence,
                context,
            });
            if state.delivering {
                return Vec::new();
            }
            state.delivering = true;
        }
        let mut errors = Vec::new();
        loop {
            let (publication, source_listeners, listeners) = {
                let mut state = self.state.lock();
                let Some(publication) = state.publications.pop_front() else {
                    state.delivering = false;
                    break;
                };
                let source_listeners: Vec<SourceListener> = state
                    .source_listeners
                    .iter()
                    .map(|(_, listener)| listener.clone())
                    .collect();
                let listeners: Vec<(Arc<StateSubscriber<T>>, u64)> = state
                    .listeners
                    .iter()
                    .map(|(_, subscriber, hydrated)| (subscriber.clone(), *hydrated))
                    .collect();
                (publication, source_listeners, listeners)
            };
            for listener in source_listeners {
                if let Err(error) =
                    listener(&publication.ops, publication.sequence, &publication.context)
                {
                    errors.push(error);
                }
            }
            let delivery = ReplicatedStateDelivery::update(publication.sequence);
            for (subscriber, hydrated_sequence) in listeners {
                if publication.sequence <= hydrated_sequence {
                    continue;
                }
                subscriber.push(StateDelivery {
                    value: publication.value.clone(),
                    context: publication.context.clone(),
                    delivery,
                });
                subscriber.drain();
            }
        }
        errors
    }
}

/// `ReplicatedStateInternals`: the exact publication stream behind a state.
#[derive(Clone)]
pub struct ReplicatedStateInternals<T> {
    publisher: Arc<ReplicatedStatePublisher<T>>,
}

impl<T: Clone + Send + 'static> ReplicatedStateInternals<T> {
    /// Atomically capture the immutable value and its matching publication sequence.
    pub fn snapshot(&self) -> (T, u64) {
        self.publisher.snapshot()
    }

    pub fn subscribe(
        &self,
        listener: impl Fn(&Arc<[Op]>, u64, &Context) -> Result<(), SharedError> + Send + Sync + 'static,
    ) -> Unsubscribe {
        self.publisher.subscribe_source(Arc::new(listener))
    }
}

fn throw_collected_errors(errors: Vec<SharedError>, message: &str) -> Result<(), StateError> {
    let mut errors: Vec<StateError> = errors.into_iter().map(StateError::Thrown).collect();
    match errors.len() {
        0 => Ok(()),
        1 => Err(errors.remove(0)),
        _ => Err(StateError::aggregate(errors, message)),
    }
}

// ─── Mutable state ───────────────────────────────────────────────────────────

struct MutableInner {
    tracker: std::cell::RefCell<Tracker>,
    changing: std::cell::Cell<bool>,
}

/// Authoritative state over one tracker (`MutableReplicatedState<T>`).
pub struct MutableReplicatedState {
    inner: ReentrantMutex<MutableInner>,
    publisher: Arc<ReplicatedStatePublisher<Arc<Value>>>,
}

/// Create authoritative state by taking immutable ownership of a strict-JSON root
/// (`replicatedState(initial)`).
pub fn mutable_replicated_state(initial: impl Into<Arc<Value>>) -> MutableReplicatedState {
    let tracker = track(initial);
    let publisher =
        ReplicatedStatePublisher::new(tracker.value().clone(), Arc::new(report_error_async));
    MutableReplicatedState {
        inner: ReentrantMutex::new(MutableInner {
            tracker: std::cell::RefCell::new(tracker),
            changing: std::cell::Cell::new(false),
        }),
        publisher,
    }
}

impl MutableReplicatedState {
    pub fn value(&self) -> Arc<Value> {
        let inner = self.inner.lock();
        inner.tracker.borrow().value().clone()
    }

    /// Atomically publish one synchronous draft mutation (`change(context, mutate)`).
    pub fn change(
        &self,
        context: &Context,
        mutate: impl FnOnce(&mut Value),
    ) -> Result<(), StateError> {
        self.try_change(context, |draft| {
            mutate(draft);
            Ok(())
        })
    }

    /// [`MutableReplicatedState::change`] with a callback that can throw: an
    /// error discards the draft and is returned as [`StateError::Thrown`].
    pub fn try_change(
        &self,
        context: &Context,
        mutate: impl FnOnce(&mut Value) -> Result<(), SharedError>,
    ) -> Result<(), StateError> {
        let inner = self.inner.lock();
        if inner.changing.get() {
            return Err(StateError::contract(
                "Replicated state cannot be changed reentrantly from a change callback",
            ));
        }
        inner.changing.set(true);
        let mut change = inner.tracker.borrow_mut().begin_change();
        let prepared = (|| {
            let draft = change.state_mut()?;
            mutate(draft).map_err(StateError::Thrown)?;
            Ok::<_, StateError>(change.prepare()?)
        })();
        inner.changing.set(false);
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                change.abort();
                return Err(error);
            }
        };
        inner.tracker.borrow_mut().adopt(&prepared)?;
        if prepared.ops().is_empty() {
            return Ok(());
        }
        throw_collected_errors(
            self.publisher.publish(
                prepared.value().clone(),
                prepared.ops().clone(),
                context.clone(),
            ),
            "Replicated state listeners failed",
        )
    }

    /// Atomically take immutable ownership of a replacement.
    pub fn replace(
        &self,
        context: &Context,
        value: impl Into<Arc<Value>>,
    ) -> Result<(), StateError> {
        let inner = self.inner.lock();
        if inner.changing.get() {
            return Err(StateError::contract(
                "Replicated state cannot be replaced from a change callback",
            ));
        }
        let prepared = inner.tracker.borrow_mut().prepare_replace(value);
        inner.tracker.borrow_mut().adopt(&prepared)?;
        if prepared.ops().is_empty() {
            return Ok(());
        }
        throw_collected_errors(
            self.publisher.publish(
                prepared.value().clone(),
                prepared.ops().clone(),
                context.clone(),
            ),
            "Replicated state listeners failed",
        )
    }

    pub fn subscribe(
        &self,
        listener: impl Fn(Arc<Value>, Context, ReplicatedStateDelivery) -> ListenerOutcome
        + Send
        + Sync
        + 'static,
    ) -> Unsubscribe {
        self.publisher.subscribe(Arc::new(listener))
    }

    pub fn internals(&self) -> ReplicatedStateInternals<Arc<Value>> {
        ReplicatedStateInternals {
            publisher: self.publisher.clone(),
        }
    }
}

impl ReplicatedState<Arc<Value>> for MutableReplicatedState {
    fn current(&self) -> Option<Arc<Value>> {
        Some(self.value())
    }

    fn subscribe(&self, listener: StateListener<Arc<Value>>) -> Unsubscribe {
        self.publisher.subscribe(listener)
    }
}

// ─── Source attachment ───────────────────────────────────────────────────────

/// The fixed snapshot of a source attachment.
#[derive(Debug, Clone)]
pub struct ReplicatedStateSnapshot<T> {
    pub value: T,
    pub cursor: i64,
}

/// One immutable authoritative revision committed after an attachment snapshot.
#[derive(Debug, Clone)]
pub struct ReplicatedStateSourceFrame<T> {
    /// Monotonic source cursor. The first frame after a snapshot must be `snapshot.cursor + 1`.
    pub cursor: i64,
    /// The exact immutable value produced by this commit.
    pub value: T,
    /// The exact immutable operation batch that produced `value`.
    pub ops: Arc<[Op]>,
    pub context: Context,
}

/// The listener a source attachment delivers frames to.
pub type FrameListener<T> = Box<dyn Fn(ReplicatedStateSourceFrame<T>) + Send + Sync>;

/// `ReplicatedStateSourceAttachment<T>`.
pub trait ReplicatedStateSourceAttachment<T>: Send + Sync {
    /// Fixed immutable snapshot captured at the atomic attachment boundary.
    fn snapshot(&self) -> ReplicatedStateSnapshot<T>;
    /// Install the sole listener and synchronously drain every buffered frame
    /// in source commit order. Single-use.
    fn activate(&self, listener: FrameListener<T>) -> Result<(), StateError>;
    /// Stop delivery and release source resources. Must be idempotent.
    fn dispose(&self) -> Result<(), StateError>;
}

/// An authoritative immutable revision source (`ReplicatedStateSource<T>`).
/// `attach()` must atomically capture one snapshot and buffer every later frame.
pub trait ReplicatedStateSource<T> {
    fn attach(&self) -> Result<Box<dyn ReplicatedStateSourceAttachment<T>>, StateError>;
}

/// `ReplicatedStateSourceOptions`.
#[derive(Clone, Default)]
pub struct ReplicatedStateSourceOptions {
    /// Receives source-contract and publication-listener failures without throwing them into the source.
    pub on_error: Option<Arc<dyn Fn(StateError) + Send + Sync>>,
}

struct AttachedInner<T> {
    publisher: Arc<ReplicatedStatePublisher<T>>,
    attachment: Box<dyn ReplicatedStateSourceAttachment<T>>,
    report_error: ErrorReporter,
    cursor: Mutex<i64>,
    disposed: AtomicBool,
}

/// A synchronously hydrated publication-only state backed by one source
/// attachment (`AttachedReplicatedState<T>`).
pub struct AttachedReplicatedState<T> {
    inner: Arc<AttachedInner<T>>,
}

impl<T> Clone for AttachedReplicatedState<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

fn assert_cursor(cursor: i64, kind: &str) -> Result<(), StateError> {
    if (-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&cursor) {
        Ok(())
    } else {
        Err(StateError::contract(format!(
            "Replicated state source {kind} cursor must be a safe integer"
        )))
    }
}

impl<T: Clone + Send + Sync + 'static> AttachedInner<T> {
    fn receive(&self, frame: ReplicatedStateSourceFrame<T>) {
        if self.disposed.load(Ordering::SeqCst) {
            return;
        }
        let result = (|| {
            assert_cursor(frame.cursor, "frame")?;
            {
                let mut cursor = self.cursor.lock();
                let expected = *cursor + 1;
                if frame.cursor != expected {
                    return Err(StateError::contract(format!(
                        "Replicated state source cursor has a gap: expected {expected}, received {}",
                        frame.cursor
                    )));
                }
                *cursor = frame.cursor;
            }
            let mut errors = self
                .publisher
                .publish(frame.value, frame.ops, frame.context);
            if errors.len() == 1 {
                self.report(StateError::Thrown(errors.remove(0)));
            } else if errors.len() > 1 {
                self.report(StateError::aggregate(
                    errors.into_iter().map(StateError::Thrown).collect(),
                    "Replicated state listeners failed",
                ));
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.fail(error);
        }
    }

    fn fail(&self, error: StateError) {
        if self.disposed.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Err(dispose_error) = self.attachment.dispose() {
            self.report(StateError::aggregate(
                vec![error, dispose_error],
                "Replicated state source contract failed",
            ));
            return;
        }
        self.report(error);
    }

    fn report(&self, error: StateError) {
        (self.report_error)(error);
    }
}

impl<T: Clone + Send + Sync + 'static> AttachedReplicatedState<T> {
    pub fn value(&self) -> T {
        self.inner.publisher.value()
    }

    pub fn subscribe(
        &self,
        listener: impl Fn(T, Context, ReplicatedStateDelivery) -> ListenerOutcome
        + Send
        + Sync
        + 'static,
    ) -> Unsubscribe {
        self.inner.publisher.subscribe(Arc::new(listener))
    }

    /// Idempotently release the source attachment. The last published value remains readable.
    pub fn dispose(&self) -> Result<(), StateError> {
        if self.inner.disposed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        self.inner.attachment.dispose()
    }

    pub fn internals(&self) -> ReplicatedStateInternals<T> {
        ReplicatedStateInternals {
            publisher: self.inner.publisher.clone(),
        }
    }

    fn activate(&self) -> Result<(), StateError> {
        let inner = Arc::downgrade(&self.inner);
        self.inner.attachment.activate(Box::new(move |frame| {
            if let Some(inner) = inner.upgrade() {
                inner.receive(frame);
            }
        }))
    }
}

impl<T: Clone + Send + Sync + 'static> ReplicatedState<T> for AttachedReplicatedState<T> {
    fn current(&self) -> Option<T> {
        Some(self.value())
    }

    fn subscribe(&self, listener: StateListener<T>) -> Unsubscribe {
        self.inner.publisher.subscribe(listener)
    }
}

/// Attach a publication-only replicated state to one authoritative immutable
/// source stream (`attachReplicatedStateSource`).
pub fn attach_replicated_state_source<T: Clone + Send + Sync + 'static>(
    source: &(impl ReplicatedStateSource<T> + ?Sized),
    options: ReplicatedStateSourceOptions,
) -> Result<AttachedReplicatedState<T>, StateError> {
    let attachment = source.attach()?;
    let snapshot = attachment.snapshot();
    let construct = assert_cursor(snapshot.cursor, "snapshot");
    let state = match construct {
        Ok(()) => {
            let report_error: ErrorReporter = options
                .on_error
                .unwrap_or_else(|| Arc::new(report_error_async));
            Ok(AttachedReplicatedState {
                inner: Arc::new(AttachedInner {
                    publisher: ReplicatedStatePublisher::new(snapshot.value, report_error.clone()),
                    attachment,
                    report_error,
                    cursor: Mutex::new(snapshot.cursor),
                    disposed: AtomicBool::new(false),
                }),
            })
        }
        Err(error) => Err((error, attachment)),
    };
    let (error, dispose): (StateError, Box<dyn FnOnce() -> Result<(), StateError>>) = match state {
        Ok(state) => match state.activate() {
            Ok(()) => return Ok(state),
            Err(error) => {
                let inner = state.inner.clone();
                (error, Box::new(move || inner.attachment.dispose()))
            }
        },
        Err((error, attachment)) => (error, Box::new(move || attachment.dispose())),
    };
    if let Err(dispose_error) = dispose() {
        return Err(StateError::aggregate(
            vec![error, dispose_error],
            "Failed to attach replicated state source",
        ));
    }
    Err(error)
}

/// `replicatedState(source, options)`.
pub fn replicated_state<T: Clone + Send + Sync + 'static>(
    source: &(impl ReplicatedStateSource<T> + ?Sized),
    options: Option<ReplicatedStateSourceOptions>,
) -> Result<AttachedReplicatedState<T>, StateError> {
    attach_replicated_state_source(source, options.unwrap_or_default())
}

// ─── Replica ─────────────────────────────────────────────────────────────────

struct ReplicaState {
    listeners: Vec<(u64, Arc<StateSubscriber<Arc<Value>>>)>,
    value: Option<Arc<Value>>,
    sequence: Option<u64>,
    next_id: u64,
}

/// A cold read-only state used by service consumers until a complete snapshot arrives.
pub struct ReplicatedStateReplica {
    report_error: ErrorReporter,
    state: Arc<Mutex<ReplicaState>>,
}

impl ReplicatedStateReplica {
    pub fn new(report_error: impl Fn(StateError) + Send + Sync + 'static) -> Self {
        Self {
            report_error: Arc::new(report_error),
            state: Arc::new(Mutex::new(ReplicaState {
                listeners: Vec::new(),
                value: None,
                sequence: None,
                next_id: 0,
            })),
        }
    }

    pub fn value(&self) -> Option<Arc<Value>> {
        self.state.lock().value.clone()
    }

    pub fn subscribe(
        &self,
        listener: impl Fn(Arc<Value>, Context, ReplicatedStateDelivery) -> ListenerOutcome
        + Send
        + Sync
        + 'static,
    ) -> Unsubscribe {
        self.subscribe_arc(Arc::new(listener))
    }

    fn subscribe_arc(&self, listener: StateListener<Arc<Value>>) -> Unsubscribe {
        let subscriber = StateSubscriber::new(listener, self.report_error.clone());
        let (id, hydration) = {
            let mut state = self.state.lock();
            let id = state.next_id;
            state.next_id += 1;
            state.listeners.push((id, subscriber.clone()));
            let hydration = state.value.clone().map(|value| StateDelivery {
                value,
                context: service_delivery_context(),
                delivery: ReplicatedStateDelivery::hydrate(
                    state.sequence.expect("hydrated sequence"),
                ),
            });
            (id, hydration)
        };
        if let Some(hydration) = hydration {
            subscriber.push(hydration);
            subscriber.drain();
        }
        let state = Arc::downgrade(&self.state);
        Unsubscribe::new(move || {
            subscriber.close();
            if let Some(state) = state.upgrade() {
                state.lock().listeners.retain(|(own, _)| *own != id);
            }
        })
    }

    pub fn hydrate(&self, sequence: u64, ops: &[Op], context: &Context) -> Result<(), StateError> {
        let next = (|| {
            if !is_base(ops) {
                return Err(StateError::contract(
                    "Replicated state snapshot is not a base operation batch",
                ));
            }
            Ok(apply_immutable(&Value::Null, ops)?)
        })();
        let next = match next {
            Ok(next) => next,
            Err(error) => {
                self.clear();
                return Err(error);
            }
        };
        {
            let mut state = self.state.lock();
            state.sequence = Some(sequence);
            state.value = Some(Arc::new(next));
        }
        self.deliver_all(context, ReplicatedStateDelivery::hydrate(sequence));
        Ok(())
    }

    pub fn update(&self, sequence: u64, ops: &[Op], context: &Context) -> Result<(), StateError> {
        let current = {
            let state = self.state.lock();
            match (state.sequence, state.value.clone()) {
                (Some(current), Some(value)) => (current, value),
                _ => {
                    return Err(StateError::contract(
                        "Replicated state received an update before hydration",
                    ));
                }
            }
        };
        if sequence != current.0 + 1 {
            self.clear();
            return Err(StateError::contract(
                "Replicated state update sequence has a gap",
            ));
        }
        let next = match apply_immutable(&current.1, ops) {
            Ok(next) => next,
            Err(error) => {
                self.clear();
                return Err(error.into());
            }
        };
        {
            let mut state = self.state.lock();
            state.sequence = Some(sequence);
            state.value = Some(Arc::new(next));
        }
        self.deliver_all(context, ReplicatedStateDelivery::update(sequence));
        Ok(())
    }

    pub fn clear(&self) {
        let subscribers: Vec<_> = {
            let mut state = self.state.lock();
            state.value = None;
            state.sequence = None;
            state
                .listeners
                .iter()
                .map(|(_, subscriber)| subscriber.clone())
                .collect()
        };
        for subscriber in subscribers {
            subscriber.clear();
        }
    }

    fn deliver_all(&self, context: &Context, delivery: ReplicatedStateDelivery) {
        let (value, subscribers) = {
            let state = self.state.lock();
            let Some(value) = state.value.clone() else {
                return;
            };
            let subscribers: Vec<_> = state
                .listeners
                .iter()
                .map(|(_, subscriber)| subscriber.clone())
                .collect();
            (value, subscribers)
        };
        let frame = StateDelivery {
            value,
            context: context.clone(),
            delivery,
        };
        // Enqueue for everyone before user code can publish another revision reentrantly.
        for subscriber in &subscribers {
            subscriber.push(frame.clone());
        }
        for subscriber in &subscribers {
            subscriber.drain();
        }
    }
}

impl ReplicatedState<Arc<Value>> for ReplicatedStateReplica {
    fn current(&self) -> Option<Arc<Value>> {
        self.value()
    }

    fn subscribe(&self, listener: StateListener<Arc<Value>>) -> Unsubscribe {
        self.subscribe_arc(listener)
    }
}

/// Context for synthetic service deliveries without a caller (`serviceDeliveryContext`).
pub fn service_delivery_context() -> Context {
    BACKGROUND_CONTEXT.clone()
}

#[cfg(test)]
mod tests;
